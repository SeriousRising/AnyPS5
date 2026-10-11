#ifndef CORE_SHADER_RECOMPILER_BDAABI_HPP
#define CORE_SHADER_RECOMPILER_BDAABI_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ShaderRecompiler::BdaAbi {

inline constexpr std::uint32_t Version = 1;
inline constexpr std::uint32_t Read = 1;
inline constexpr std::uint32_t Write = 2;

struct Header {
    std::uint32_t version;
    std::uint32_t count;
    std::uint32_t entryBytes;
    std::uint32_t reserved;
};

struct Range {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint64_t deviceAddress;
    std::uint32_t permissions;
    std::uint32_t reserved;
};

enum class FaultState : std::uint32_t { Empty, Writing, Ready };
enum class FaultReason : std::uint32_t { Unmapped = 1, Permission, Overflow, InvalidTable, InvalidRectangle, LoopLimit, Unaligned, InvalidDescriptor };

struct Fault {
    FaultState state;
    FaultReason reason;
    std::uint64_t address;
    std::uint32_t bytes;
    std::uint32_t stage;
    std::uint32_t instruction;
    std::uint32_t reserved;
};

static_assert(std::is_standard_layout_v<Header> && std::is_trivially_copyable_v<Header> && sizeof(Header) == 16);
static_assert(std::is_standard_layout_v<Range> && std::is_trivially_copyable_v<Range> && sizeof(Range) == 32);
static_assert(offsetof(Range, deviceAddress) == 16 && offsetof(Range, permissions) == 24);
static_assert(std::is_standard_layout_v<Fault> && std::is_trivially_copyable_v<Fault> && sizeof(Fault) == 32);
static_assert(offsetof(Fault, address) == 8 && offsetof(Fault, instruction) == 24);

// The direct table (DirectVersion). The ranges are also given as a tree the shader walks with one
// load a level and no search: `rootCount` root entries, each for the 1 << rootShift guest bytes from
// `base` on, then the entries they lead to, `entries` in all, then the `ranges` ranges the tree was
// built from (for diagnostics; the shader does not read them). An entry is four dwords. A leaf
// (flags without DirectDescend): the guest bytes of the entry's unit from its first byte up to
// `bytes` (0: none; 0xffffffff: that many or more) are readable at their guest address plus the 64-bit
// `delta` (low dword first), and writable with DirectWritable. An entry with DirectDescend has its
// unit split into units of 1 << `shift` bytes, whose entries start at entry `child`; `mask` is
// (1 << shift) - 1. A unit in which a mapping starts is always split, down to single bytes, so a
// leaf never needs the offset of a first mapped byte. Touching ranges with the same delta and
// permissions are one mapping.
inline constexpr std::uint32_t DirectVersion = 2;
inline constexpr std::uint32_t DirectDescend = 1;
inline constexpr std::uint32_t DirectWritable = 2;
inline constexpr std::uint32_t DirectMaxRootShift = 30;

// Whether `version` is a table form programs are made for.
constexpr bool TableVersion(std::uint32_t version) {
    return version == Version || version == DirectVersion;
}

struct DirectHeader {
    std::uint32_t version;
    std::uint32_t rootShift;
    std::uint32_t rootCount;
    std::uint32_t entries;
    std::uint64_t base;
    std::uint32_t ranges;
    std::uint32_t reserved;
};

struct DirectEntry {
    // A leaf's delta, or child (low dword) and shift (high dword).
    std::uint64_t delta;
    // A leaf's bytes, or mask.
    std::uint32_t bytes;
    std::uint32_t flags;
};

static_assert(std::is_standard_layout_v<DirectHeader> && std::is_trivially_copyable_v<DirectHeader> && sizeof(DirectHeader) == 32 && offsetof(DirectHeader, base) == 16);
static_assert(std::is_standard_layout_v<DirectEntry> && std::is_trivially_copyable_v<DirectEntry> && sizeof(DirectEntry) == 16 && offsetof(DirectEntry, bytes) == 8);

inline constexpr std::uint32_t WrittenPageShift = 12;
inline constexpr std::uint32_t WrittenChunkShift = 17;
inline constexpr std::uint32_t WrittenChunkPages = 1u << (WrittenChunkShift - WrittenPageShift);
inline constexpr std::uint32_t WrittenPageSlots = 2048;
inline constexpr std::uint32_t WrittenPageProbes = 8;
inline constexpr std::uint64_t WrittenAddressLimit = std::uint64_t{0xffffffffu} << WrittenChunkShift;
inline constexpr std::uint32_t WrittenOverflowWord = sizeof(Fault) / sizeof(std::uint32_t);
inline constexpr std::uint32_t WrittenCountWord = WrittenOverflowWord + 1;
inline constexpr std::uint32_t WrittenSlotsWord = WrittenCountWord + 1;
inline constexpr std::uint32_t WrittenListWord = WrittenSlotsWord + 2 * WrittenPageSlots;
inline constexpr std::size_t FaultBufferBytes = (WrittenListWord + WrittenPageSlots) * sizeof(std::uint32_t);
static_assert((WrittenPageSlots & (WrittenPageSlots - 1)) == 0);
static_assert(WrittenChunkPages == 32);

inline constexpr std::uint32_t WrittenChunkKey(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> WrittenChunkShift) + 1u;
}

template<typename TMark>
bool ForEachWrittenRange(const std::uint32_t* words, TMark&& mark) {
    const auto count = words[WrittenCountWord];
    if (count > WrittenPageSlots) return false;
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[WrittenListWord + claim];
        if (slot >= WrittenPageSlots || words[WrittenSlotsWord + 2 * slot] == 0) return false;
    }
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[WrittenListWord + claim];
        const auto key = words[WrittenSlotsWord + 2 * slot];
        auto mask = words[WrittenSlotsWord + 2 * slot + 1];
        const auto base = static_cast<std::uint64_t>(key - 1u) << WrittenChunkShift;
        while (mask != 0) {
            const auto first = static_cast<std::uint32_t>(std::countr_zero(mask));
            const auto pages = static_cast<std::uint32_t>(std::countr_one(mask >> first));
            mark(base + (static_cast<std::uint64_t>(first) << WrittenPageShift), static_cast<std::size_t>(pages) << WrittenPageShift);
            mask = pages + first >= 32u ? 0u : mask & ~((1u << (first + pages)) - 1u);
        }
    }
    return true;
}

}

#endif
