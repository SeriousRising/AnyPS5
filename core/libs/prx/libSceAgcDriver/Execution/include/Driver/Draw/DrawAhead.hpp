#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWAHEAD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWAHEAD_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/Submission.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>

namespace AgcDriver::DriverDetail {

enum DrawAheadCount : std::size_t { AheadScans, AheadPrepared, AheadHits, AheadWarm, AheadPassed, AheadPending, AheadUnsupported, AheadErrors, AheadStops, DrawAheadCountKinds };

struct DrawAhead {
    static constexpr std::size_t Running = std::numeric_limits<std::size_t>::max();
    static constexpr std::uint64_t Window = 16;

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::thread thread;
    const Submission* submission = nullptr;
    std::size_t start = 0;
    std::uint64_t firstDraw = 0;
    QueueState state;
    bool busy = false;
    bool stop = false;
    std::atomic<bool> cancel{false};
    std::atomic<std::size_t> stopped{Running};
    std::atomic<std::uint64_t> draws{0};
    std::array<std::atomic<std::uint64_t>, DrawAheadCountKinds> counts{};
    std::array<std::uint64_t, DrawAheadCountKinds> reported{};
    void Count(DrawAheadCount kind) { counts[kind].fetch_add(1, std::memory_order_relaxed); }
};

}

#endif
