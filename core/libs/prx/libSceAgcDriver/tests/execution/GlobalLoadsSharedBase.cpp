#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t ResultOffset = 0x40;
constexpr std::uint32_t Sentinel = 0x05e471e1;

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "global loads sharing a base: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() { return block; }

private:
    std::uint8_t* block = nullptr;
};

// v1:v2 hold the block's address (s8:s9) and v3:v4 the base address under test (s10:s11); v9 is the
// sum of what was read, stored at the block's ResultOffset by the last instruction before s_endpgm.
// Reads that differ only by their immediates share one translation of their base address (see
// EmitBdaSharedDwordReads); APS5_NO_BDA_SHARED_BASE=1 has every read look its own bytes up, and both
// must leave the block the same.
constexpr std::uint32_t Waitcnt = 0xbf8c3f70u;
constexpr std::uint32_t Endpgm = 0xbf810000u;

std::vector<std::uint32_t> Begin() {
    return {0x7e120280u, 0x7e020208u, 0x7e040209u, 0x7e06020au, 0x7e08020bu};
}

void End(std::vector<std::uint32_t>& code) {
    code.insert(code.end(), {0xdc708000u | ResultOffset, 0x007d0901u, Endpgm});
}

std::uint32_t Branch(std::uint32_t opcode, std::size_t from, std::size_t to) {
    return opcode | (static_cast<std::uint32_t>(to - (from + 1u)) & 0xffffu);
}

// v9 += the `dwords` dwords at [v<base>:v<base + 1> + offset], loaded into v10 and up.
void Load(std::vector<std::uint32_t>& code, std::uint32_t base, std::uint32_t offset, std::uint32_t dwords) {
    constexpr std::array<std::uint32_t, 4> opcodes{0xdc308000u, 0xdc348000u, 0xdc3c8000u, 0xdc388000u};
    code.insert(code.end(), {opcodes[dwords - 1u] | offset, 0x0a7d0000u | base, Waitcnt});
    for (std::uint32_t dword = 0; dword < dwords; ++dword) code.push_back(0x4a120109u | ((10u + dword) << 9u));
}

struct Read {
    std::uint32_t base, offset, dwords;
};

// Two bases in turn, with reads of one to four dwords on each.
constexpr std::array<Read, 8> MixedReads{{{1, 0x100, 4}, {3, 0, 1}, {1, 0x120, 2}, {3, 12, 1}, {1, 0x110, 1}, {3, 4, 2}, {1, 0x114, 3}, {3, 16, 4}}};

std::vector<std::uint32_t> MixedCode() {
    auto code = Begin();
    for (const auto& read : MixedReads) Load(code, read.base, read.offset, read.dwords);
    End(code);
    return code;
}

// What one pass of the loop reads from its base address.
constexpr std::array<Read, 3> LoopReads{{{3, 0, 1}, {3, 8, 2}, {3, 4, 1}}};
constexpr std::uint32_t LoopStep = 16;

// for (v14 = 0; v14 < passes; ++v14) { the loop's reads; v3 += LoopStep; }, then the store: the base
// address is another value in every pass. The reads are followed by a branch, so that their block is
// not the one the loop continues from (see the stopping reads' technical debt entry).
std::vector<std::uint32_t> LoopCode(std::uint32_t passes) {
    auto code = Begin();
    code.insert(code.end(), {0x7e1c0280u, 0x7e1e0280u | passes});
    const auto loop = code.size();
    code.insert(code.end(), {0x7d821f0eu, 0u});
    for (const auto& read : LoopReads) Load(code, read.base, read.offset, read.dwords);
    const auto skip = code.size();
    code.insert(code.end(), {0x7d821f0eu, 0u, 0x4a121280u});
    code[skip + 1u] = Branch(0xbf860000u, skip + 1u, code.size());
    code.insert(code.end(), {0x4a060690u, 0x4a1c1c81u, 0u});
    code[code.size() - 1u] = Branch(0xbf820000u, code.size() - 1u, loop);
    code[loop + 1u] = Branch(0xbf860000u, loop + 1u, code.size());
    End(code);
    return code;
}

// Runs `code` with `tested` in s10:s11. `expected` is the sum of the dwords read, or nothing when a
// read faults: the invocation must then end, so the block keeps every byte, the result's included.
// The recorder prints the fault ("deferred write-back failed") and does not hand it to the caller, so
// the record itself is not compared here.
void Run(AgcDriver::VulkanDevice& device, GuestBlock& guest, const std::string& name, const std::vector<std::uint32_t>& code, std::uint64_t tested, const std::vector<std::uint64_t>& read) {
    std::vector<std::uint32_t> memory(BlockBytes / 4u);
    for (std::size_t word = 0; word < memory.size(); ++word) memory[word] = static_cast<std::uint32_t>((word + 3u) * 0x9e3779b1u);
    memory[ResultOffset / 4u] = Sentinel;
    std::memcpy(guest.Data(), memory.data(), BlockBytes);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(guest.Data()));
    bool faults = false;
    std::uint32_t expected = 0;
    for (const auto dword : read) {
        if (dword < address || dword + 4u > address + BlockBytes) faults = true;
        else expected += memory[(dword - address) / 4u];
    }
    std::vector<std::uint32_t> userData(12, 0u);
    userData[8] = static_cast<std::uint32_t>(address);
    userData[9] = static_cast<std::uint32_t>(address >> 32u);
    userData[10] = static_cast<std::uint32_t>(tested);
    userData[11] = static_cast<std::uint32_t>(tested >> 32u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> regions{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(std::span(code))}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{1, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, regions},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    if (!faults) memory[ResultOffset / 4u] = expected;
    for (std::size_t word = 0; word < memory.size(); ++word) {
        std::uint32_t value = 0;
        std::memcpy(&value, guest.Data() + word * 4u, 4u);
        Require(value == memory[word], "global loads sharing a base: " + name + " left " + std::to_string(value) + " at byte " + std::to_string(word * 4u) + ", expected " + std::to_string(memory[word]));
    }
}

template<std::size_t TCount>
void Append(std::vector<std::uint64_t>& read, const std::array<Read, TCount>& reads, std::uint64_t block, std::uint64_t tested) {
    for (const auto& one : reads) {
        for (std::uint32_t dword = 0; dword < one.dwords; ++dword) read.push_back((one.base == 1u ? block : tested) + one.offset + dword * 4u);
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock guest;
        const auto block = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(guest.Data()));
        const auto mixed = [&](const std::string& name, std::uint64_t tested) {
            std::vector<std::uint64_t> read;
            Append(read, MixedReads, block, tested);
            Run(*device, guest, name, MixedCode(), tested, read);
        };
        mixed("two bases in turn", block + 0x400u);
        // The second base's hull lies across the block's end: its reads before the end succeed and the last one faults.
        mixed("a base whose last read lies past the block's end", block + BlockBytes - 24u);
        // Nothing of the second base's hull is mapped: its first read faults.
        mixed("a base past the block's end", block + BlockBytes);
        const auto loop = [&](const std::string& name, std::uint64_t tested, std::uint32_t passes) {
            std::vector<std::uint64_t> read;
            for (std::uint32_t pass = 0; pass < passes; ++pass) Append(read, LoopReads, block, tested + pass * LoopStep);
            Run(*device, guest, name, LoopCode(passes), tested, read);
        };
        loop("a base that moves in a loop", block + 0x200u, 5u);
        // The last pass's hull lies across the block's end.
        loop("a base that a loop moves past the block's end", block + BlockBytes - 40u, 3u);
        std::puts("global loads sharing a base tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
