#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_PACKETTIMING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_PACKETTIMING_HPP

#include <chrono>
#include <string>
#include <vector>
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchTiming.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <utility>

namespace AgcDriver::DriverDetail {

struct WorkerProfile {
    double dispatchMs = 0;
    double drawMs = 0;
    double waitMs = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point reported = start;
};

struct TimestampPacketCounts {
    std::uint64_t releaseMem = 0;
    std::uint64_t releaseClock = 0;
    std::uint64_t releaseClockWrites = 0;
    std::uint64_t copyData = 0;
    std::uint64_t copyClock = 0;
    std::uint64_t releaseEop = 0;
    std::uint64_t releaseCs = 0;
    std::uint64_t releasePs = 0;
    std::uint64_t releaseOther = 0;
    std::uint32_t firstOtherEvent = 0;

    void Note(std::uint32_t opcode, std::span<const std::uint32_t> packet) {
        if (opcode == 0x49 && packet.size() >= 8) {
            ++releaseMem;
            if ((packet[2] >> 29u) == 3) {
                ++releaseClock;
                if (packet[3] != 0 || packet[4] != 0) {
                    ++releaseClockWrites;
                    const auto type = packet[1] & 0x3fu;
                    const auto index = (packet[1] >> 8u) & 0xfu;
                    if (index == 5) ++releaseEop;
                    else if (index == 6 && type == 0x2f) ++releaseCs;
                    else if (index == 6 && type == 0x30) ++releasePs;
                    else {
                        if (releaseOther == 0) firstOtherEvent = packet[1] & 0xf3fu;
                        ++releaseOther;
                    }
                }
            }
        } else if (opcode == 0x40 && packet.size() >= 6) {
            ++copyData;
            if ((((packet[1] & 0xfu) << 1u) | ((packet[1] >> 30u) & 1u)) == 18) ++copyClock;
        }
    }
};

struct PacketProfile {
    std::map<std::uint32_t, std::pair<std::uint64_t, double>> byOpcode;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    std::uint64_t submissions = 0;

    double flushMs = 0;

    std::array<std::pair<std::uint64_t, double>, static_cast<std::size_t>(DispatchOutcome::Count)> dispatchOutcomes{};
    TimestampPacketCounts timestamps;
};

std::vector<std::string> HitchLines(std::chrono::steady_clock::time_point now, double inFlightWaitMs);

struct PacketTimer {
    bool enabled; std::uint32_t key; std::uint32_t queue; PacketProfile& profile; std::chrono::steady_clock::time_point start;
    ~PacketTimer();
};

}

#endif
