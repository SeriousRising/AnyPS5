#include "prx/libSceAgcDriver/Graphics/include/BdaDirectTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <string>

namespace AgcDriver::Graphics {

namespace {

// A run of guest bytes readable at one distance from their device addresses.
struct Mapping {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint64_t delta;
    bool writable;
    // DirectFollowed, and the entry that holds the seam with the next mapping (see BdaAbi::DirectSeamShift).
    std::uint32_t seam = 0;
};

// The unit sizes of the direct table's levels, as shifts. The root is the first whose units cover
// the mappings with at most DirectRootCapacity entries, or the coarsest: 16 KiB units (a guest page)
// for up to 1 GiB, 4 MiB units for up to 256 GiB, and 1 GiB units beyond.
constexpr std::array<std::uint32_t, 5> DirectShifts{30, 22, 14, 7, 0};
constexpr std::uint64_t DirectRootCapacity = 65536;
constexpr std::uint64_t DirectRootLimit = std::uint64_t{1} << 20u;
static_assert(DirectShifts.front() == ShaderRecompiler::BdaAbi::DirectMaxRootShift);

// Fills entries[first, first + count): the units of 1 << DirectShifts[level] bytes from `start`.
// `mapping` is the first one that may reach the first unit.
void fillDirectEntries(std::vector<ShaderRecompiler::BdaAbi::DirectEntry>& entries, std::span<const Mapping> mappings, std::size_t first, std::uint64_t start, std::size_t count, std::size_t level, std::size_t mapping) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    const auto shift = DirectShifts[level];
    const auto size = std::uint64_t{1} << shift;
    for (std::size_t unit = 0; unit < count;) {
        const auto unitStart = start + (static_cast<std::uint64_t>(unit) << shift);
        while (mapping < mappings.size() && mappings[mapping].end <= unitStart) ++mapping;
        if (mapping == mappings.size()) return;
        const auto& found = mappings[mapping];
        if (found.begin > unitStart + (size - 1u)) {
            // The units up to the mapping's first are left without bytes.
            unit += static_cast<std::size_t>(std::min<std::uint64_t>((found.begin - unitStart) >> shift, count - unit));
            continue;
        }
        // The last byte a unit may hold and be a leaf of the mapping: the one before the next begins.
        const auto alone = mapping + 1 == mappings.size() ? std::numeric_limits<std::uint64_t>::max() : mappings[mapping + 1].begin - 1u;
        if (found.begin <= unitStart && unitStart + (size - 1u) <= alone) {
            // The run of such units: they differ in the bytes left of the mapping only.
            const auto flags = found.writable ? Abi::DirectWritable : 0u;
            auto left = found.end - unitStart;
            auto last = unitStart + (size - 1u);
            for (; unit < count && left != 0 && last <= alone; ++unit) {
                const bool exact = left < 0xffffffffu;
                entries[first + unit] = {found.delta, exact ? static_cast<std::uint32_t>(left) : 0xffffffffu, flags | (exact ? found.seam : 0u)};
                left = left > size ? left - size : 0;
                last += size;
            }
            continue;
        }
        const auto childShift = DirectShifts[level + 1];
        const auto child = entries.size();
        const auto children = std::size_t{1} << (shift - childShift);
        Require(child + children <= 0x03ffffffu, "BDA direct table entry count overflow");
        entries.resize(child + children);
        entries[first + unit] = {child | static_cast<std::uint64_t>(childShift) << 32u, (1u << childShift) - 1u, Abi::DirectDescend};
        fillDirectEntries(entries, mappings, child, unitStart, children, level + 1, mapping);
        ++unit;
    }
}

}

std::vector<std::uint32_t> BdaDirectTable(std::span<const ShaderRecompiler::BdaAbi::Range> ranges) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    std::vector<Mapping> mappings;
    mappings.reserve(ranges.size());
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        const auto& range = ranges[index];
        Require(range.reserved == 0 && range.begin < range.end && range.deviceAddress != 0 && range.end - range.begin <= std::numeric_limits<std::uint64_t>::max() - range.deviceAddress && (range.permissions & Abi::Read) != 0, "invalid range for the BDA direct table");
        Require(index == 0 || ranges[index - 1].begin <= range.begin, "the ranges of the BDA direct table are not sorted");
        // As the search selects: a range answers from its begin to the next one's.
        const auto end = index + 1 < ranges.size() ? std::min(range.end, ranges[index + 1].begin) : range.end;
        if (range.begin >= end) continue;
        const auto delta = range.deviceAddress - range.begin;
        const bool writable = (range.permissions & Abi::Write) != 0;
        if (!mappings.empty() && mappings.back().end == range.begin && mappings.back().delta == delta && mappings.back().writable == writable) mappings.back().end = end;
        else mappings.push_back({range.begin, end, delta, writable});
    }
    Abi::DirectHeader header{Abi::DirectVersion, 0, 0, 0, 0, static_cast<std::uint32_t>(ranges.size()), 0};
    std::vector<Abi::DirectEntry> entries;
    if (!mappings.empty()) {
        Require(mappings.front().begin >= Abi::DirectMargin && mappings.back().end <= std::uint64_t{0} - Abi::DirectMargin, "guest ranges in the first or last 128 KiB of the address space are not implemented in the BDA direct table");
        std::uint64_t rootCount = 0;
        std::size_t level = 0;
        for (const std::size_t candidate : {2u, 1u, 0u}) {
            const auto shift = DirectShifts[candidate];
            level = candidate;
            header.base = mappings.front().begin >> shift << shift;
            rootCount = ((mappings.back().end - 1u - header.base) >> shift) + 1u;
            if (rootCount <= DirectRootCapacity) break;
        }
        Require(rootCount <= DirectRootLimit, "guest ranges more than " + std::to_string(DirectRootLimit) + " GiB apart are not implemented in the BDA direct table");
        header.rootShift = DirectShifts[level];
        header.rootCount = static_cast<std::uint32_t>(rootCount);
        // Consecutive tables are about as large, and growing by doubling was most of a build's time.
        static std::atomic<std::size_t> reserved{0};
        entries.reserve(reserved.load(std::memory_order_relaxed));
        entries.resize(rootCount);
        // The seams follow the root, so that no leaf's seam is entry 0.
        for (std::size_t index = 0; index + 1 < mappings.size(); ++index) {
            auto& before = mappings[index];
            const auto& after = mappings[index + 1];
            if (before.end != after.begin) continue;
            before.seam = Abi::DirectFollowed;
            const auto last = before.end - 4u + before.delta;
            const auto next = after.begin + after.delta;
            if (before.end - before.begin < 4u || after.end - after.begin < 4u || (last & 3u) != 0 || (next & 3u) != 0) continue;
            before.seam |= static_cast<std::uint32_t>(entries.size()) << Abi::DirectSeamShift;
            entries.push_back({last, static_cast<std::uint32_t>(next), static_cast<std::uint32_t>(next >> 32u)});
        }
        fillDirectEntries(entries, mappings, 0, header.base, rootCount, level, 0);
        reserved.store(entries.size(), std::memory_order_relaxed);
    }
    header.entries = static_cast<std::uint32_t>(entries.size());
    static_assert(sizeof(Abi::DirectHeader) % 4u == 0 && sizeof(Abi::DirectEntry) % 4u == 0 && sizeof(Abi::Range) % 4u == 0);
    std::vector<std::uint32_t> words((sizeof(header) + entries.size() * sizeof(Abi::DirectEntry) + ranges.size() * sizeof(Abi::Range)) / 4u);
    auto* bytes = reinterpret_cast<std::byte*>(words.data());
    std::memcpy(bytes, &header, sizeof(header));
    if (!entries.empty()) std::memcpy(bytes + sizeof(header), entries.data(), entries.size() * sizeof(Abi::DirectEntry));
    if (!ranges.empty()) std::memcpy(bytes + sizeof(header) + entries.size() * sizeof(Abi::DirectEntry), ranges.data(), ranges.size() * sizeof(Abi::Range));
    return words;
}

}
