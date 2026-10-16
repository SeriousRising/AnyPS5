#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

struct GpuQueueInterval {
    std::uint64_t begin, end;
    std::uint32_t queue;
};

struct GpuQueueCoverage {
    std::map<std::uint32_t, std::uint64_t> ticks;
    std::uint64_t mixedTicks = 0;
};

inline GpuQueueCoverage PartitionGpuQueueTimes(std::span<const GpuQueueInterval> intervals) {
    struct Event {
        std::uint64_t time;
        std::uint32_t queue;
        bool begins;
    };
    std::vector<Event> events;
    events.reserve(intervals.size() * 2);
    for (const auto& interval : intervals) {
        if (interval.end <= interval.begin) continue;
        events.push_back({interval.begin, interval.queue, true});
        events.push_back({interval.end, interval.queue, false});
    }
    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) { return a.time < b.time; });
    GpuQueueCoverage coverage;
    std::map<std::uint32_t, std::size_t> active;
    std::uint64_t previous = 0;
    for (const auto& event : events) {
        const auto elapsed = event.time - previous;
        if (elapsed != 0) {
            if (active.size() == 1) coverage.ticks[active.begin()->first] += elapsed;
            else if (active.size() > 1) coverage.mixedTicks += elapsed;
        }
        if (event.begins) ++active[event.queue];
        else if (--active[event.queue] == 0) active.erase(event.queue);
        previous = event.time;
    }
    return coverage;
}

}
