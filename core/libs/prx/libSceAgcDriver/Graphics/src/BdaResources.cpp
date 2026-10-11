#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

namespace {

// The page tables built per device, reused by a build whose ranges (guest ranges, device
// addresses, permissions) equal one of them: consecutive address-based builds map the same registered
// set through the same imports and mirrors, so the ~1200-entry table repeats. The shader only reads
// the table (its lookup keeps its state in function variables and, for the ranges it found last,
// in per-invocation Private ones), and a batch in flight keeps its BdaResources, so sharing the
// buffer is safe. A table is written once, below, before it is bound or shared: the lookup's range
// cache (DefineBdaLookupCore) relies on it never changing afterwards. The cache holds the tables weakly: they never outlive
// the builds that own them (a static owner would destroy the buffer after its device, at exit or on
// device replacement, and would fail the tests' leak check), and with deferred lease release the
// previous builds are still in flight when the next one asks, which is what makes the hit rate.
// One table was enough while every build repeated the last one exactly; now the misaligned
// descriptor sub-ranges the GPU copies out of imports sit in pool staging buffers whose device
// addresses cycle through the few buffers the batches in flight leave free, and the three
// address-based programs of one queue interleave with the lease draws, so a table differs from the
// previous one and comes back a few builds later (69% misses in the wave-8 run, 4% before the GPU
// copies). Hence several recent tables, most recently used first, matched by a word hash before
// the byte compare. APS5_BDA_TABLE_CACHE_ENTRIES sets how many (default 8; 1 is the previous
// behaviour); APS5_NO_BDA_TABLE_CACHE=1 builds every table.
// A table built for (or found equal to) the cached address space's ranges carries the space's
// serial: a build served by that space alone binds it by the serial, with no hash and no compare.
struct TableEntry {
    std::uint64_t hash;
    std::vector<ShaderRecompiler::BdaAbi::Range> ranges;
    std::weak_ptr<Buffer> buffer;
    std::uint64_t spaceSerial = 0;
};

struct TableCache {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    std::list<TableEntry> entries;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    BdaResources::TableCacheStats classes;
};

bool sameRanges(const std::vector<ShaderRecompiler::BdaAbi::Range>& left, const std::vector<ShaderRecompiler::BdaAbi::Range>& right);

// APS5_PROFILE_DRAW: why the most recently used entry did not serve this build (see
// TableCacheStats). Under the cache mutex.
void classifyFirstEntry(TableCache& cache, std::uint64_t hash, const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& classes = cache.classes;
    if (cache.entries.empty()) {
        ++classes.firstEmpty;
        return;
    }
    const auto& first = cache.entries.front();
    if (first.hash == hash) {
        if (first.buffer.expired()) ++classes.firstExpired;
        else if (!sameRanges(first.ranges, ranges)) ++classes.firstSameHash;
        return;
    }
    if (first.buffer.expired()) {
        ++classes.firstExpired;
        return;
    }
    const auto common = std::min(first.ranges.size(), ranges.size());
    std::size_t at = 0;
    while (at < common && std::memcmp(&first.ranges[at], &ranges[at], sizeof(ranges[at])) == 0) ++at;
    const auto begin = at < ranges.size() ? ranges[at].begin : at < first.ranges.size() ? first.ranges[at].begin : 0;
    ++(begin < 0x10000000ull ? classes.firstDiffersLow : classes.firstDiffersHeap);
}

TableCache& Tables() {
    static TableCache cache;
    return cache;
}

bool tableCacheEnabled() {
    static const bool disabled = std::getenv("APS5_NO_BDA_TABLE_CACHE") != nullptr;
    return !disabled;
}

std::size_t tableCacheEntries() {
    static const std::size_t entries = [] {
        const char* value = std::getenv("APS5_BDA_TABLE_CACHE_ENTRIES");
        const auto parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 8ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(parsed, 1, 64));
    }();
    return entries;
}

bool sameRanges(const std::vector<ShaderRecompiler::BdaAbi::Range>& left, const std::vector<ShaderRecompiler::BdaAbi::Range>& right) {
    return left.size() == right.size() && (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(left.front())) == 0);
}

// A 64-bit mix of the table's words: a table is ~1200 ranges of four words, and comparing the bytes
// against every held table would cost as much as building it.
std::uint64_t hashRanges(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges) {
    static_assert(sizeof(ShaderRecompiler::BdaAbi::Range) % sizeof(std::uint64_t) == 0);
    std::uint64_t hash = 0x9e3779b97f4a7c15ull ^ ranges.size();
    for (const auto& range : ranges) {
        std::uint64_t words[sizeof(range) / sizeof(std::uint64_t)];
        std::memcpy(words, &range, sizeof(range));
        for (const auto word : words) {
            hash ^= word;
            hash *= 0xff51afd7ed558ccdull;
            hash ^= hash >> 33u;
        }
    }
    return hash;
}

}

BdaResources::BdaResources(const Context& context) {
    static_assert(std::endian::native == std::endian::little);
    Require(ShaderRecompiler::BdaAbi::FaultBufferBytes <= context.limits.maxStorageBufferRange, "BDA fault buffer exceeds storage buffer range limit");
    fault = std::make_unique<Buffer>(context, ShaderRecompiler::BdaAbi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // The slots of translated bases are reached only through the directory, which is cleared here and
    // set by writeBases after the slots it names.
    std::memset(fault->Bytes().data(), 0, ShaderRecompiler::BdaAbi::BaseSlotsWord * sizeof(std::uint32_t));
}

namespace {

struct BaseSlotCounters {
    std::atomic<std::uint64_t> translated{0}, unusable{0}, programs{0}, refused{0};
    std::atomic<std::int64_t> lastReport{0};
    std::atomic<std::uint64_t> reportedTranslated{0}, reportedUnusable{0}, reportedPrograms{0}, reportedRefused{0};
};

BaseSlotCounters& BaseSlots() {
    static BaseSlotCounters counters;
    return counters;
}

// APS5_PROFILE_GPU: every 10 s, the slots filled since the last line. A program that then reads with
// its own lookups all the same (its base address is not the one the CPU computed) is not counted:
// only the GPU sees that.
void reportBaseSlots() {
    static const bool profile = std::getenv("APS5_PROFILE_GPU") != nullptr;
    if (!profile) return;
    auto& counters = BaseSlots();
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000ll || !counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const auto since = [](const std::atomic<std::uint64_t>& total, std::atomic<std::uint64_t>& reported) {
        const auto value = total.load(std::memory_order_relaxed);
        return static_cast<unsigned long long>(value - reported.exchange(value, std::memory_order_relaxed));
    };
    const auto translated = since(counters.translated, counters.reportedTranslated);
    const auto unusable = since(counters.unusable, counters.reportedUnusable);
    const auto programs = since(counters.programs, counters.reportedPrograms);
    const auto refused = since(counters.refused, counters.reportedRefused);
    std::fprintf(stderr, "[gpu] base addresses translated by the CPU (10 s): %llu slots with a device address and %llu unusable for %llu programs; %llu programs found no room for their slots\n", translated, unusable, programs, refused);
}

}

std::uint64_t BdaResources::TranslateHull(std::span<const ShaderRecompiler::BdaAbi::Range> ranges, std::uint64_t address, std::uint32_t bytes) {
    const auto end = address + bytes;
    if (bytes == 0 || end <= address) return 0;
    // The search of lookup_bda, step for step: a table that is not sorted selects the same range.
    std::size_t low = 0;
    std::size_t high = ranges.size();
    auto ceiling = ~std::uint64_t{0};
    while (low < high) {
        const auto midpoint = low + (high - low) / 2;
        const auto visited = ranges[midpoint].begin;
        if (address < visited) {
            high = midpoint;
            ceiling = std::min(ceiling, visited);
        } else {
            low = midpoint + 1;
        }
    }
    if (low == 0) return 0;
    const auto& range = ranges[low - 1];
    if (range.reserved != 0 || range.begin >= range.end || end > range.end || (range.permissions & ShaderRecompiler::BdaAbi::Read) == 0 || range.deviceAddress == 0) return 0;
    const auto device = address + (range.deviceAddress - range.begin);
    if (device < range.deviceAddress || device + bytes <= device || end > ceiling || (device & 3u) != 0) return 0;
    return device;
}

void BdaResources::writeBases(std::span<const ShaderRecompiler::BdaAbi::Range> ranges, std::span<const StageBases> stages) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    auto* words = reinterpret_cast<std::uint32_t*>(fault->Bytes().data());
    auto& counters = BaseSlots();
    std::uint32_t next = 0;
    for (const auto& stage : stages) {
        if (stage.bases.empty()) continue;
        Require(stage.stage < Abi::BaseStages, "invalid stage for translated base addresses");
        const bool taken = std::any_of(stages.data(), &stage, [&](const StageBases& other) { return other.stage == stage.stage && !other.bases.empty(); });
        if (taken || stage.bases.size() > Abi::BaseSlotCapacity - next) {
            // The programs read with their own lookups, as without the CPU's translation: two
            // programs of one stage would read each other's slots, so neither gets a directory word.
            static std::atomic<bool> noted{false};
            if (!noted.exchange(true)) std::fprintf(stderr, "[gpu] a draw has two programs of one stage with base addresses for the CPU to translate, or more than %u such addresses; the programs without slots look their reads up themselves (noted once)\n", Abi::BaseSlotCapacity);
            if (taken) words[Abi::BaseDirectoryWord + stage.stage] = 0;
            counters.refused.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        std::uint64_t translated = 0;
        for (std::size_t index = 0; index < stage.bases.size(); ++index) {
            const auto& request = stage.bases[index];
            const std::array<std::uint64_t, 2> slot{request.base, request.bytes != 0 ? TranslateHull(ranges, request.hull, request.bytes) : 0};
            translated += slot[1] != 0;
            std::memcpy(words + Abi::BaseSlotsWord + (next + index) * Abi::BaseSlotWords, slot.data(), sizeof(slot));
        }
        words[Abi::BaseDirectoryWord + stage.stage] = next + 1;
        next += static_cast<std::uint32_t>(stage.bases.size());
        counters.translated.fetch_add(translated, std::memory_order_relaxed);
        counters.unusable.fetch_add(stage.bases.size() - translated, std::memory_order_relaxed);
        counters.programs.fetch_add(1, std::memory_order_relaxed);
    }
    if (next != 0) reportBaseSlots();
}

BdaResources::BaseSlotStats BdaResources::BaseSlotCounters() {
    const auto& counters = BaseSlots();
    return {counters.translated.load(std::memory_order_relaxed), counters.unusable.load(std::memory_order_relaxed), counters.programs.load(std::memory_order_relaxed), counters.refused.load(std::memory_order_relaxed)};
}

BdaResources::BdaResources(const Context& context, const GuestBufferMemory& memory, std::span<const StageBases> stages) : BdaResources(context) {
    const auto cached = memory.CachedAddressTable();
    std::vector<ShaderRecompiler::BdaAbi::Range> built;
    if (!cached.has_value()) built = memory.AddressRanges();
    const auto& ranges = cached.has_value() ? *cached->ranges : built;
    const auto serial = cached.has_value() ? cached->serial : 0;
    Require(ranges.size() <= std::numeric_limits<std::uint32_t>::max(), "BDA table range count overflow");
    Require(ranges.size() <= (std::numeric_limits<std::size_t>::max() - sizeof(ShaderRecompiler::BdaAbi::Header)) / sizeof(ShaderRecompiler::BdaAbi::Range), "BDA table size overflow");
    tableBytes = sizeof(ShaderRecompiler::BdaAbi::Header) + ranges.size() * sizeof(ShaderRecompiler::BdaAbi::Range);
    Require(tableBytes <= context.limits.maxStorageBufferRange && ShaderRecompiler::BdaAbi::FaultBufferBytes <= context.limits.maxStorageBufferRange, "BDA descriptors exceed storage buffer range limit");
    // Before the table is looked up or built: every path below binds a table that holds `ranges`.
    writeBases(ranges, stages);
    auto& cache = Tables();
    std::uint64_t hash = 0;
    if (tableCacheEnabled()) {
        std::lock_guard lock(cache.mutex);
        // Another device's tables are not this one's (their buffers are dead or foreign).
        if (cache.device != context.device) {
            cache.entries.clear();
            cache.device = context.device;
        }
        if (serial != 0) {
            for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
                if (it->spaceSerial != serial) continue;
                if (auto shared = it->buffer.lock(); shared != nullptr) {
                    ++cache.hits;
                    ++cache.classes.spaceTables;
                    table = std::move(shared);
                    cache.entries.splice(cache.entries.begin(), cache.entries, it);
                    return;
                }
                cache.entries.erase(it);
                break;
            }
        }
        hash = hashRanges(ranges);
        classifyFirstEntry(cache, hash, ranges);
        for (auto it = cache.entries.begin(); it != cache.entries.end();) {
            // Compare the hash before taking a strong reference: locking a non-matching entry's
            // buffer could make this thread its last owner (a concurrent reap releasing it) and
            // run the pool release under the cache mutex; expired() takes no reference.
            if (it->hash != hash) {
                if (it->buffer.expired()) {
                    it = cache.entries.erase(it);
                } else {
                    ++it;
                }
                continue;
            }
            auto shared = it->buffer.lock();
            if (shared == nullptr) {
                // Every build that owned the table is gone, and its buffer with it.
                it = cache.entries.erase(it);
                continue;
            }
            if (sameRanges(it->ranges, ranges)) {
                ++cache.hits;
                // An equal table of an earlier space (a rebuild that mapped the same ranges the
                // same way): the next build of this space binds it by the serial.
                if (serial != 0) it->spaceSerial = serial;
                table = std::move(shared);
                cache.entries.splice(cache.entries.begin(), cache.entries, it);
                return;
            }
            ++it;
        }
    }
    table = std::make_shared<Buffer>(context, tableBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const ShaderRecompiler::BdaAbi::Header header{ShaderRecompiler::BdaAbi::Version, static_cast<std::uint32_t>(ranges.size()), sizeof(ShaderRecompiler::BdaAbi::Range), 0};
    std::memcpy(table->Bytes().data(), &header, sizeof(header));
    if (!ranges.empty()) std::memcpy(table->Bytes().data() + sizeof(header), ranges.data(), ranges.size() * sizeof(ranges.front()));
    if (tableCacheEnabled()) {
        std::lock_guard lock(cache.mutex);
        ++cache.misses;
        if (cache.device != context.device) {
            cache.entries.clear();
            cache.device = context.device;
        }
        cache.entries.push_front({hash, cached.has_value() ? ranges : std::move(built), table, serial});
        while (cache.entries.size() > tableCacheEntries()) cache.entries.pop_back();
    }
}

BdaResources::TableCacheStats BdaResources::TableCacheCounters() {
    auto& cache = Tables();
    std::lock_guard lock(cache.mutex);
    auto stats = cache.classes;
    stats.hits = cache.hits;
    stats.misses = cache.misses;
    stats.held = cache.entries.size();
    return stats;
}

VkDescriptorBufferInfo BdaResources::Table() const {
    Require(table != nullptr, "BDA page table was not requested");
    return {table->Handle(), 0, tableBytes};
}

VkDescriptorBufferInfo BdaResources::Fault() const {
    return {fault->Handle(), 0, ShaderRecompiler::BdaAbi::FaultBufferBytes};
}

namespace {
std::atomic<bool> loopGuardTripped{false};
std::mutex loopGuardMutex;
LoopGuardReport lastLoopGuard;
std::vector<std::pair<std::uint64_t, std::uint32_t>> reportedLoopGuards;
}

LoopGuardReport LastLoopGuardReport() {
    std::lock_guard lock(loopGuardMutex);
    return lastLoopGuard;
}

bool LoopGuardTripped() {
    return loopGuardTripped.load(std::memory_order_relaxed);
}

void BdaResources::CheckFault() const {
    markWrittenPages();
    ShaderRecompiler::BdaAbi::Fault report{};
    std::memcpy(&report, fault->Bytes().data(), sizeof(report));
    if (report.state == ShaderRecompiler::BdaAbi::FaultState::Empty) {
        Require(static_cast<std::uint32_t>(report.reason) == 0 && report.address == 0 && report.bytes == 0 && report.stage == 0 && report.instruction == 0 && report.reserved == 0, "BDA fault record has data without publication");
        return;
    }
    const bool mismatch = report.reason == ShaderRecompiler::BdaAbi::FaultReason::RangeCacheMismatch;
    Require(report.state == ShaderRecompiler::BdaAbi::FaultState::Ready && (report.reserved == 0 || mismatch), "incomplete or invalid BDA fault record");
    if (mismatch) {
        // APS5_BDA_RANGE_CACHE_CHECK: the access is refused, so the work's results are not valid.
        std::ostringstream text;
        text << "address range cache check failed: the range kept for address 0x" << std::hex << report.address << std::dec << " (" << report.bytes << " bytes) is not the one the page table gives, at instruction 0x" << std::hex << report.instruction << " of the stage " << std::dec << report.stage << " program whose code address ends in 0x" << std::hex << report.reserved << "; the page table changed while the work ran, or the cache is wrong";
        std::fprintf(stderr, "[gpu] %s\n", text.str().c_str());
        throw std::runtime_error(text.str());
    }
    Require(report.reason != ShaderRecompiler::BdaAbi::FaultReason::InvalidRectangle, "rect-list requires finite nondegenerate axis-aligned positions with equal positive W");
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::LoopLimit) {
        // APS5_LOOP_GUARD: the shader left a loop that ran past the guard; the dispatch result is kept.
        loopGuardTripped.store(true, std::memory_order_relaxed);
        {
            std::lock_guard lock(loopGuardMutex);
            lastLoopGuard = {report.address, report.stage, report.instruction, report.bytes};
            const std::pair key{report.address, report.instruction};
            if (reportedLoopGuards.size() < 64 && std::find(reportedLoopGuards.begin(), reportedLoopGuards.end(), key) == reportedLoopGuards.end()) {
                reportedLoopGuards.push_back(key);
                constexpr std::array<const char*, 9> stages{"unknown", "vertex", "pixel", "fetch", "compute", "mesh", "local", "hull", "domain"};
                std::fprintf(stderr, "[gpu] loop guard: %s shader 0x%llx loop exit at instruction 0x%x passed %u evaluations (APS5_LOOP_GUARD; the shader left its loops unfinished)\n", report.stage < stages.size() ? stages[report.stage] : "unknown", static_cast<unsigned long long>(report.address), report.instruction, report.bytes);
            }
        }
        std::memset(fault->Bytes().data(), 0, fault->Bytes().size());
        return;
    }
    std::ostringstream message;
    message << "BDA access failed: address=0x" << std::hex << report.address << " instruction=0x" << report.instruction << std::dec << " bytes=" << report.bytes << " stage=" << report.stage << " reason=" << static_cast<std::uint32_t>(report.reason);
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::Permission && table != nullptr) {
        const auto bytes = table->Bytes();
        ShaderRecompiler::BdaAbi::Header header{};
        std::memcpy(&header, bytes.data(), sizeof(header));
        for (std::uint32_t index = 0; index < header.count; ++index) {
            ShaderRecompiler::BdaAbi::Range range{};
            std::memcpy(&range, bytes.data() + sizeof(header) + index * sizeof(range), sizeof(range));
            if (report.address < range.begin || report.address >= range.end) continue;
            message << std::hex << "; the store hit 0x" << range.begin << "+0x" << range.end - range.begin << ", read-only in the BDA table: stores through GPU-selected descriptors and FLAT/GLOBAL addresses reach only writable ranges imported in place, not ones served by a mirror or a copy (past APS5_HOST_IMPORT_MIB, or refused by the driver)";
            break;
        }
    }
    throw std::runtime_error(message.str());
}

}

namespace AgcDriver::Graphics {

void BdaResources::markWrittenPages() const {
    namespace Abi = ShaderRecompiler::BdaAbi;
    auto* words = reinterpret_cast<std::uint32_t*>(fault->Bytes().data());
    Require(words[Abi::WrittenOverflowWord] == 0, "more than " + std::to_string(Abi::WrittenPageSlots) + " pages stored to through the BDA table in one use are not implemented");
    bool any = false;
    for (std::uint32_t slot = 0; slot < Abi::WrittenPageSlots; ++slot) {
        const auto page = words[Abi::WrittenSlotsWord + slot];
        if (page == 0) continue;
        GuestMemory::MarkWritten(static_cast<std::uint64_t>(page - 1u) << Abi::WrittenPageShift, std::size_t{1} << Abi::WrittenPageShift);
        any = true;
    }
    if (any) std::memset(words + Abi::WrittenSlotsWord, 0, Abi::WrittenPageSlots * sizeof(std::uint32_t));
}

}
