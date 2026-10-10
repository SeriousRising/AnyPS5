#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
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

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Limit = 100;
constexpr std::uint32_t Marker = 0x600d10adu;
constexpr std::uint32_t Sentinel = 0x05e471e1u;
constexpr std::size_t BlockBytes = 65536;
alignas(256) std::array<std::uint32_t, 4> Output{};

alignas(256) constexpr std::array<std::uint32_t, 13> ScalarLoopCode{
    0xbe890380, 0x80098109, 0xbf078009, 0xbf85fffd, 0x7e02020a, 0x7e04020b, 0xdc308000, 0x0a7d0001,
    0xbf8c3f70, 0xe0700000, 0x80010a00, 0xbf810000, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 16> WaveLoopCode{
    0x7e0a02c1, 0xbe90047e, 0x7d8a0a80, 0x87fe6a7e, 0xbf880002, 0x4a0a0ac1, 0xbf82fffb, 0xbefe0410,
    0x7e02020a, 0x7e04020b, 0xdc308000, 0x0a7d0001, 0xbf8c3f70, 0xe0700000, 0x80010a00, 0xbf810000,
};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "loop guard: cannot allocate the guest block");
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

void SetGuard(const char* value) {
#ifdef _WIN32
    _putenv_s("APS5_LOOP_GUARD", value == nullptr ? "" : value);
#else
    if (value == nullptr) unsetenv("APS5_LOOP_GUARD");
    else setenv("APS5_LOOP_GUARD", value, 1);
#endif
}

ShaderRecompiler::RecompileResult Compile(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::uint8_t* guest) {
    std::vector<std::uint32_t> userData(12, 0u);
    const auto output = reinterpret_cast<std::uintptr_t>(Output.data());
    userData[4] = static_cast<std::uint32_t>(output);
    userData[5] = static_cast<std::uint32_t>((output >> 32u) & 0xffffu);
    userData[6] = static_cast<std::uint32_t>(Output.size() * 4u);
    userData[7] = 0x31016facu;
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(guest));
    userData[10] = static_cast<std::uint32_t>(address);
    userData[11] = static_cast<std::uint32_t>(address >> 32u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

void Check(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::uint32_t exitOffset, const std::uint8_t* guest, const std::string& what) {
    SetGuard(nullptr);
    const auto plain = Compile(device, code, guest);
    SetGuard("100");
    const auto guarded = Compile(device, code, guest);
    SetGuard(nullptr);
    const auto again = Compile(device, code, guest);
    Require(plain.spirv == again.spirv, what + ": the shader without APS5_LOOP_GUARD changed after a guarded compile");
    Require(plain.spirv != guarded.spirv, what + ": APS5_LOOP_GUARD did not change the shader");
    Output.fill(Sentinel);
    device.Dispatch(guarded, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    Require(Output[0] == Marker, what + ": the guarded shader did not run to its end, its output is " + std::to_string(Output[0]));
    const auto report = AgcDriver::Graphics::LastLoopGuardReport();
    Require(AgcDriver::Graphics::LoopGuardTripped() && report.shader == reinterpret_cast<std::uintptr_t>(code.data()) && report.limit == Limit, what + ": the cut loop was not reported for this shader and limit");
    Require(report.instruction == exitOffset, what + ": the cut loop was reported at instruction " + std::to_string(report.instruction) + ", expected " + std::to_string(exitOffset));
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock guest;
        std::memcpy(guest.Data(), &Marker, sizeof(Marker));
        Check(*device, ScalarLoopCode, 4u, guest.Data(), "scalar loop");
        Check(*device, WaveLoopCode, 8u, guest.Data(), "wave mask loop");
        std::puts("loop guard tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
