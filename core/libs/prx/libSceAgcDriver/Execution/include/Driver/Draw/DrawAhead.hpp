#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWAHEAD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWAHEAD_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/Submission.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>

namespace AgcDriver::DriverDetail {

enum DrawAheadCount : std::size_t { AheadScans, AheadPrepared, AheadHits, AheadWarm, AheadPassed, AheadPending, AheadUnsupported, AheadErrors, AheadStops, AheadOverKnown, AheadOverRaw, AheadOverSync, AheadOverImage, AheadOverVertex, AheadIndirect, AheadEmpty, AheadMetadataPass, AheadNothing, AheadRejected, AheadMesh, AheadNoDevice, AheadStageRejected, AheadAwaited, DrawAheadCountKinds };

enum DrawLockRunCount : std::size_t { RunsStarted, RunDrawsHeld, RunEndedByMiss, RunEndedBySync, RunEndedByPacket, RunEndedByBound, DrawLockRunCountKinds };

struct DrawLockRun {
    static constexpr std::uint32_t MaxDraws = 8;
    static constexpr std::chrono::microseconds MaxHeld{200};
    std::unique_lock<GuestMemory::GpuMutexType> lock{GuestMemory::GpuMutex(), std::defer_lock};
    std::uint32_t draws = 0;
    std::chrono::steady_clock::time_point since{};
    bool enabled = false;
};

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
    // The ordinal (in `draws`) of the draw prepareDrawAhead works on now, 0 between draws.
    std::atomic<std::uint64_t> preparing{0};
    std::array<std::atomic<std::uint64_t>, DrawAheadCountKinds> counts{};
    std::array<std::uint64_t, DrawAheadCountKinds> reported{};
    std::array<std::atomic<std::uint64_t>, 9> refused{};
    std::array<std::uint64_t, 9> refusedReported{};
    void Count(DrawAheadCount kind) { counts[kind].fetch_add(1, std::memory_order_relaxed); }
};

}

#endif
