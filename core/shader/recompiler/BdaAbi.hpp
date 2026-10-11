#ifndef CORE_SHADER_RECOMPILER_BDAABI_HPP
#define CORE_SHADER_RECOMPILER_BDAABI_HPP

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
// RangeCacheMismatch: the address lookup's range cache check (APS5_BDA_RANGE_CACHE_CHECK) found a kept
// range that the page table no longer gives; `reserved` then holds the low dword of the program's
// code address.
// Split: a program for the direct table read a dword that starts in one mapping and ends in the
// next where it cannot (see DirectSeamShift); the address is the dword's.
enum class FaultReason : std::uint32_t { Unmapped = 1, Permission, Overflow, InvalidTable, InvalidRectangle, LoopLimit, Unaligned, InvalidDescriptor, RangeCacheMismatch, Split };

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
// Where a mapping is followed at once by another, the leaves that hold its end have DirectFollowed.
// A dword that starts in the first and ends in the second is read without a second walk when both
// have four bytes or more and the last four of the first and the first four of the second lie at
// 4-aligned device addresses: those leaves then also carry (flags >> DirectSeamShift) the index of
// an entry that is no entry of the tree but those two device addresses (`delta`, then `bytes` and
// `flags` as the low and high dwords of the second).
inline constexpr std::uint32_t DirectVersion = 2;
inline constexpr std::uint32_t DirectDescend = 1;
inline constexpr std::uint32_t DirectWritable = 2;
inline constexpr std::uint32_t DirectFollowed = 4;
inline constexpr std::uint32_t DirectSeamShift = 3;
inline constexpr std::uint32_t DirectMaxRootShift = 30;
// No mapping of a direct table reaches into the first or the last DirectMargin bytes of the address
// space, so that an address a program made by adding up to half that much to an address, without
// checking for a wrap, is not mapped when it did wrap, and neither is one within as much of it.
inline constexpr std::uint32_t DirectMargin = 131072;

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
inline constexpr std::uint32_t WrittenPageSlots = 4096;
inline constexpr std::uint32_t WrittenPageProbes = 8;
inline constexpr std::uint32_t WrittenOverflowWord = sizeof(Fault) / sizeof(std::uint32_t);
inline constexpr std::uint32_t WrittenSlotsWord = WrittenOverflowWord + 1;
static_assert((WrittenPageSlots & (WrittenPageSlots - 1)) == 0);

// The base addresses the CPU translated for one draw or dispatch follow the written pages. A program
// whose reads share base addresses that are the same for the whole draw numbers them from 0; the
// directory word of its stage holds 1 + the slot of its base 0, or 0 when the draw has none for it,
// and a slot is the guest base address the CPU computed and the device address of the base's hull
// (0: the hull is not in one readable range at a 4-aligned device address), each low dword first.
// A program uses a slot only where its own base address equals the slot's.
inline constexpr std::uint32_t BaseStages = 16;
inline constexpr std::uint32_t BaseSlotCapacity = 1024;
inline constexpr std::uint32_t BaseSlotWords = 4;
inline constexpr std::uint32_t BaseDirectoryWord = WrittenSlotsWord + WrittenPageSlots;
inline constexpr std::uint32_t BaseSlotsWord = BaseDirectoryWord + BaseStages;
inline constexpr std::size_t FaultBufferBytes = (BaseSlotsWord + BaseSlotCapacity * BaseSlotWords) * sizeof(std::uint32_t);

// A base as the CPU evaluated it for a draw: the base address and its hull, the `bytes` bytes from
// `hull`. No bytes: the base could not be evaluated, or its address or hull wraps.
struct BaseRequest {
    std::uint64_t base = 0;
    std::uint64_t hull = 0;
    std::uint32_t bytes = 0;

    bool operator==(const BaseRequest& other) const = default;
};

}

#endif
