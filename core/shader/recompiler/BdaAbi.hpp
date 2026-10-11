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
enum class FaultReason : std::uint32_t { Unmapped = 1, Permission, Overflow, InvalidTable, InvalidRectangle, LoopLimit, Unaligned, InvalidDescriptor, RangeCacheMismatch };

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
