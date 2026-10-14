#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <utility>

class PresentProfile {
public:
    using Clock = std::chrono::steady_clock;

    struct Summary {
        std::uint64_t intervals = 0;
        Clock::duration elapsed{};
        Clock::duration maximum{};
        std::uint64_t over60Hz = 0;
        std::uint64_t over30Hz = 0;
        std::uint64_t over50Ms = 0;
        std::uint64_t over100Ms = 0;
    };

    std::optional<Summary> Record(Clock::time_point now, bool completed, bool overlay) {
        if (!completed || overlay) return std::nullopt;
        if (!previous) {
            previous = now;
            return std::nullopt;
        }
        const auto interval = now - *previous;
        previous = now;
        ++summary.intervals;
        summary.elapsed += interval;
        summary.maximum = std::max(summary.maximum, interval);
        summary.over60Hz += interval > std::chrono::duration_cast<Clock::duration>(std::chrono::duration<int, std::ratio<1, 60>>(1));
        summary.over30Hz += interval > std::chrono::duration_cast<Clock::duration>(std::chrono::duration<int, std::ratio<1, 30>>(1));
        summary.over50Ms += interval > std::chrono::milliseconds(50);
        summary.over100Ms += interval > std::chrono::milliseconds(100);
        if (summary.elapsed >= std::chrono::seconds(10)) return std::exchange(summary, {});
        return std::nullopt;
    }

    std::optional<Summary> Finish() {
        previous.reset();
        const auto result = std::exchange(summary, {});
        if (result.intervals == 0) return std::nullopt;
        return result;
    }

private:
    std::optional<Clock::time_point> previous;
    Summary summary;
};
