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
#include <algorithm>
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

constexpr std::uint32_t Reads = 24;
constexpr std::uint32_t Bases = 4;
constexpr std::uint32_t BaseBytes = 4096;
constexpr std::uint32_t Sentinel = 0x05e471e1;
constexpr std::size_t BlockBytes = 65536;
alignas(256) std::array<std::uint32_t, 4> Output{};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "global loads at many sites: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, false, true);
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

std::uint32_t BaseOf(std::uint32_t read) {
    return read % 7u == 0u ? 0u : read % Bases;
}

std::uint32_t OffsetOf(std::uint32_t read) {
    return read % 5u == 0u ? 48u : 4u * (read % 61u);
}

std::vector<std::uint32_t> Code() {
    std::vector<std::uint32_t> code{0x7e120280u};
    for (std::uint32_t base = 0; base < Bases; ++base) {
        code.push_back(0x7e000200u | ((1u + 2u * base) << 17u) | (8u + 2u * base));
        code.push_back(0x7e000200u | ((2u + 2u * base) << 17u) | (9u + 2u * base));
    }
    for (std::uint32_t read = 0; read < Reads; ++read) {
        code.insert(code.end(), {0xdc308000u | OffsetOf(read), 0x0a7d0000u | (1u + 2u * BaseOf(read)), 0xbf8c3f70u, 0x4a121509u});
    }
    code.insert(code.end(), {0xe0700000u, 0x80010900u, 0xbf810000u});
    return code;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock guest;
        std::uint32_t expected = 0;
        for (std::uint32_t word = 0; word < Bases * BaseBytes / 4u; ++word) {
            const auto value = (word + 1u) * 0x9e3779b1u;
            std::memcpy(guest.Data() + word * 4u, &value, 4u);
        }
        for (std::uint32_t read = 0; read < Reads; ++read) {
            std::uint32_t value = 0;
            std::memcpy(&value, guest.Data() + BaseOf(read) * BaseBytes + OffsetOf(read), 4u);
            expected += value;
        }
        Output.fill(Sentinel);
        const auto code = Code();
        std::vector<std::uint32_t> userData(16, 0u);
        const auto output = reinterpret_cast<std::uintptr_t>(Output.data());
        userData[4] = static_cast<std::uint32_t>(output);
        userData[5] = static_cast<std::uint32_t>((output >> 32u) & 0xffffu);
        userData[6] = static_cast<std::uint32_t>(Output.size() * 4u);
        userData[7] = 0x31016facu;
        for (std::uint32_t base = 0; base < Bases; ++base) {
            const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(guest.Data() + base * BaseBytes));
            userData[8u + 2u * base] = static_cast<std::uint32_t>(address);
            userData[9u + 2u * base] = static_cast<std::uint32_t>(address >> 32u);
        }
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(std::span(code))}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{1, 1, 1}, 0, {false, false, false}, false, 1};
        ShaderRecompiler::RecompileRequest request{
            {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
            {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
            device->Target(),
            {0, 0, 0, 128}
        };
        request.useCache = false;
        const auto result = ShaderRecompiler::Recompile(request);
        device->Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
        device->WaitIdle();
        Require(Output[0] == expected, "global loads at " + std::to_string(Reads) + " sites: the sum is " + std::to_string(Output[0]) + ", expected " + std::to_string(expected));
        std::puts("global loads at many sites tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
