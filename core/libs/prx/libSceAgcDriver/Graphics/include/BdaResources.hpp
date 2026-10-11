#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDARESOURCES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDARESOURCES_HPP

#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"

namespace AgcDriver::Graphics {

bool LoopGuardTripped();

class BdaResources {
public:
    // The bases of one stage's program that the CPU translates for this draw or dispatch
    // (RecompileResult::bdaBases and bdaBaseStage).
    struct StageBases {
        std::uint32_t stage = 0;
        std::vector<ShaderRecompiler::BdaAbi::BaseRequest> bases;
    };
    explicit BdaResources(const Context& context);
    // Fills a slot of the fault buffer for every base of `stages` from the ranges the page table
    // holds, so that a slot's device address is one the table gives for this build.
    BdaResources(const Context& context, const GuestBufferMemory& memory, std::span<const StageBases> stages = {});
    // What the shader's lookup gives a read of the `bytes` bytes at `address` asked for as one span
    // (lookup_bda_span with BdaLookupSpan): the device address of the first byte when the search of
    // `address` selects a range that holds all of them, is readable and has a device address that
    // is neither zero nor wraps, when the bytes end at or below every begin the search stopped
    // short of, and when the device address is 4-aligned; 0 otherwise.
    static std::uint64_t TranslateHull(std::span<const ShaderRecompiler::BdaAbi::Range> ranges, std::uint64_t address, std::uint32_t bytes);
    // Cumulative: slots filled with a device address, slots left unusable (a base the CPU could not
    // evaluate or whose hull the lookup refuses), and programs whose bases found no room.
    struct BaseSlotStats {
        std::uint64_t translated = 0;
        std::uint64_t unusable = 0;
        std::uint64_t programs = 0;
        std::uint64_t refused = 0;
    };
    static BaseSlotStats BaseSlotCounters();
    VkDescriptorBufferInfo Table() const;
    VkDescriptorBufferInfo Fault() const;
    void CheckFault() const;
    // APS5_PROFILE_DRAW: page tables served from the per-device cache and built anew (cumulative),
    // and how many recent tables the cache holds right now.
    struct TableCacheStats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::size_t held = 0;
        // Hits served by the cached address space's serial alone (no hash, no compare; see
        // GuestBufferMemory::CachedAddressTable), counted in `hits` too.
        std::uint64_t spaceTables = 0;
        // Why the most recently used table did not serve a build ([bda-table] first-differing entry
        // class): its owners were gone (expired weak reference), or its word hash differed, split by
        // whether the first differing range begins below 0x10000000 (a low mapping: the exe image,
        // the system heaps) or in the guest heap above; `sameHash` is a hash match whose ranges
        // still differed, `empty` a lookup with no entry held.
        std::uint64_t firstExpired = 0;
        std::uint64_t firstDiffersLow = 0;
        std::uint64_t firstDiffersHeap = 0;
        std::uint64_t firstSameHash = 0;
        std::uint64_t firstEmpty = 0;
    };
    static TableCacheStats TableCacheCounters();

private:
    // The page table is read-only to the shader, so consecutive builds mapping the same ranges to the
    // same device addresses share one buffer while one of them is alive (see the table cache in
    // BdaResources.cpp, which refers to it weakly).
    void markWrittenPages() const;
    void writeBases(std::span<const ShaderRecompiler::BdaAbi::Range> ranges, std::span<const StageBases> stages);
    std::shared_ptr<Buffer> table;
    std::unique_ptr<Buffer> fault;
    std::size_t tableBytes = 0;
};

}

#endif
