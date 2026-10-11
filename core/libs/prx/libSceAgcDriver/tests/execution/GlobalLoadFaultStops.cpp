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
// No guest allocation holds this address.
constexpr std::uint64_t Unmapped = 0x10000;

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "global load fault stops: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true);
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

// v1:v2 hold the block's address (s8:s9) and v3:v4 the address under test (s10:s11); v9 is the
// result, stored at the block's ResultOffset by the last instruction before s_endpgm.
constexpr std::uint32_t LoadTested = 0xdc308000u;
constexpr std::uint32_t LoadTestedOperands = 0x0a7d0003u;
constexpr std::uint32_t Waitcnt = 0xbf8c3f70u;
constexpr std::uint32_t AddLoaded = 0x4a121509u;
constexpr std::uint32_t AddOne = 0x4a121281u;
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

// v9 = [block] + [tested], then the store.
std::vector<std::uint32_t> StraightCode() {
    auto code = Begin();
    code.insert(code.end(), {0xdc308000u, 0x0a7d0001u, Waitcnt, AddLoaded, LoadTested, LoadTestedOperands, Waitcnt, AddLoaded});
    End(code);
    return code;
}

// for (v11 = 0; v11 < [tested]; ++v11) ++v9, then the store.
std::vector<std::uint32_t> BoundCode() {
    auto code = Begin();
    code.insert(code.end(), {LoadTested, LoadTestedOperands, Waitcnt, 0x7e160280u});
    const auto loop = code.size();
    code.insert(code.end(), {0x7d82150bu, 0u, AddOne, 0x4a161681u, 0u});
    code[loop + 1u] = Branch(0xbf860000u, loop + 1u, loop + 5u);
    code[loop + 4u] = Branch(0xbf820000u, loop + 4u, loop);
    End(code);
    return code;
}

// for (v12 = 0; v12 < 2; ++v12) for (v11 = 0; v11 < 2; ++v11) v9 += [tested], then the store. The
// read is followed by a branch, so that its block is not the one the inner loop continues from: a
// read there returns zero and its invocation runs on (see the stopping reads' technical debt entry).
std::vector<std::uint32_t> NestedCode() {
    auto code = Begin();
    code.insert(code.end(), {0x7e1a0282u, 0x7e180280u});
    const auto outer = code.size();
    code.insert(code.end(), {0x7d821b0cu, 0u, 0x7e160280u});
    const auto inner = code.size();
    code.insert(code.end(), {0x7d821b0bu, 0u, LoadTested, LoadTestedOperands, Waitcnt, AddLoaded, 0x7d821b0bu, 0u, 0x4a121280u});
    const auto next = code.size();
    code[inner + 7u] = Branch(0xbf860000u, inner + 7u, next);
    code.insert(code.end(), {0x4a161681u, 0u});
    code[next + 1u] = Branch(0xbf820000u, next + 1u, inner);
    const auto innerEnd = code.size();
    code[inner + 1u] = Branch(0xbf860000u, inner + 1u, innerEnd);
    code.insert(code.end(), {0x4a181881u, 0u});
    code[innerEnd + 1u] = Branch(0xbf820000u, innerEnd + 1u, outer);
    code[outer + 1u] = Branch(0xbf860000u, outer + 1u, code.size());
    End(code);
    return code;
}

// Runs `code` with `tested` in s10:s11. A faulting read must end the invocation, so the block keeps
// every byte, the result's included. The recorder prints the fault ("deferred write-back failed")
// and does not hand it to the caller, so the record itself is not compared here.
void Run(AgcDriver::VulkanDevice& device, GuestBlock& guest, const std::string& name, const std::vector<std::uint32_t>& code, std::uint64_t tested, bool faults, std::uint32_t expected) {
    std::vector<std::uint32_t> memory(BlockBytes / 4u);
    for (std::size_t word = 0; word < memory.size(); ++word) memory[word] = static_cast<std::uint32_t>(word + 3u);
    memory[ResultOffset / 4u] = Sentinel;
    std::memcpy(guest.Data(), memory.data(), BlockBytes);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(guest.Data()));
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
        Require(value == memory[word], "global load fault stops: " + name + " left " + std::to_string(value) + " at byte " + std::to_string(word * 4u) + ", expected " + std::to_string(memory[word]));
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock guest;
        const auto block = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(guest.Data()));
        // The words hold their index plus three: [block] is 3 and [block + 8] is 5.
        Run(*device, guest, "reads before a store", StraightCode(), block + 8u, false, 8u);
        Run(*device, guest, "a faulting read before a store", StraightCode(), Unmapped, true, 0u);
        Run(*device, guest, "a loop bound read by address", BoundCode(), block, false, 3u);
        Run(*device, guest, "a loop bound whose read faults", BoundCode(), Unmapped, true, 0u);
        Run(*device, guest, "reads inside nested loops", NestedCode(), block + 8u, false, 20u);
        Run(*device, guest, "a faulting read inside nested loops before a store", NestedCode(), Unmapped, true, 0u);
        std::puts("global load fault stops tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
