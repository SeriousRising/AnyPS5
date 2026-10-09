#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Stride = 4;
constexpr std::uint32_t SelectorRegister = 8;
alignas(256) std::array<std::uint32_t, Threads * Stride> Input{};
alignas(256) std::array<std::uint32_t, Threads * Stride> Output{};

alignas(256) constexpr std::array<std::uint32_t, 21> Code{
    0x34020082, 0xe0302000, 0x80000401, 0xbf8c3f70, 0xbe890380, 0xbf068008, 0xbf85000a, 0x4a080881, 0x80098109, 0xbf060809, 0xbf850006,
    0xbf0a8409, 0xbf85fffa, 0x4a0808c0, 0xe0702000, 0x80010401, 0xbf810000, 0x4a0808a0, 0xe0702000, 0x80010401, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 18> NestedCode{
    0x34020082, 0xe0302000, 0x80000401, 0xbf8c3f70, 0xbe890380, 0xbf820007, 0x4a080881, 0xbf090809, 0xbf850006,
    0x80098109, 0x4a080882, 0xbf098209, 0xbf85fff9, 0x4a080888, 0xbf82fffa, 0xe0702000, 0x80010401, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 22> SelectionCode{
    0x34020082, 0xe0302000, 0x80000401, 0xbf8c3f70, 0xbf0a0a09, 0xbf84000d, 0xbf0a0809, 0xbf840008, 0x4a080883, 0xbf820000,
    0xbf068109, 0xbf840004, 0x4a080887, 0x80098109, 0xbf0a0a09, 0xbf85fff4, 0x4a08088b, 0x80098109, 0xbf82fff1, 0xe0702000,
    0x80010401, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 27> NestedSelectionCode{
    0x34020082, 0xe0302000, 0x80000401, 0xbf8c3f70, 0xbe890380, 0xbf0a0a09, 0xbf840011, 0xbe8b0380, 0xbf0a0809,
    0xbf84000b, 0x4a080882, 0x800b810b, 0xbf0a0c0b, 0xbf85fffc, 0xbf068109, 0xbf840003, 0x4a080883, 0xbf0a0d09,
    0xbf840002, 0x4a080887, 0xbf820000, 0x4a08088b, 0x80098109, 0xbf82ffed, 0xe0702000, 0x80010401, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 26> CyclicSelectionCode{
    0x34020082, 0xe0302000, 0x80000401, 0xbf8c3f70, 0xbe890380, 0xbf0a0a09, 0xbf840010, 0xbe8b0380, 0xbf0a0809,
    0xbf840006, 0xbf068109, 0xbf840006, 0x4a080882, 0x800b810b, 0xbf0a0c0b, 0xbf85fffc, 0x4a080883, 0xbf820002,
    0x4a080887, 0xbf820000, 0x4a08088b, 0x80098109, 0xbf82ffee, 0xe0702000, 0x80010401, 0xbf810000,
};

struct Case {
    const char* name;
    std::uint32_t selector;
    std::uint32_t added;
    std::uint32_t limit = 4;
    std::uint32_t innerLimit = 3;
    std::uint32_t tailLimit = 2;
};

const std::array<Case, 5> Cases{{
    {"early exit before the loop", 0u, 32u},
    {"early exit from the first iteration", 1u, 33u},
    {"early exit from the third iteration", 3u, 35u},
    {"early exit from the last iteration", 4u, 36u},
    {"loop exit to the end of the program", 5u, 68u},
}};

const std::array<Case, 3> NestedCases{{
    {"nested loop exit sharing the outer merge after two iterations", 2u, 21u},
    {"nested loop exit sharing the outer merge after three iterations", 3u, 24u},
    {"nested loop exit sharing the outer merge after four iterations", 4u, 27u},
}};

const std::array<Case, 8> SelectionCases{{
    {"unconditional arm bypassed", 0u, 44u},
    {"unconditional arm on the first iteration", 1u, 47u},
    {"local join and conditional repeat", 2u, 46u},
    {"unconditional arm on three iterations", 3u, 49u},
    {"unconditional arm on every iteration", 4u, 52u},
    {"short loop bypasses the arm", 0u, 22u, 2u},
    {"short loop enters the arm once", 1u, 25u, 2u},
    {"conditional repeat falls through to the local join", 2u, 35u, 2u},
}};

const std::array<Case, 9> NestedSelectionCases{{
    {"nested loop bypassed on every iteration", 0u, 44u},
    {"nested loop before the local join", 1u, 57u},
    {"nested loop and both local joins", 2u, 73u},
    {"nested loop on three outer iterations", 3u, 86u},
    {"nested loop on every outer iteration", 4u, 99u},
    {"single nested iteration", 2u, 65u, 4u, 1u},
    {"conditional arm skips the inner local join", 2u, 66u, 4u, 3u, 0u},
    {"short outer loop with both local joins", 2u, 51u, 2u},
    {"outer loop bypassed", 4u, 0u, 0u},
}};

const std::array<Case, 7> CyclicSelectionCases{{
    {"cyclic tail bypasses the nested loop", 0u, 56u},
    {"cyclic tail enters its alternate arm", 1u, 60u},
    {"cyclic tail enters the nested loop", 2u, 66u},
    {"cyclic tail takes both sides of the join", 3u, 70u},
    {"cyclic tail before every outer latch", 4u, 74u},
    {"cyclic tail with a single nested iteration", 2u, 62u, 4u, 1u},
    {"cyclic tail with a short outer loop", 2u, 38u, 2u},
}};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), count, 0x01016facu};
}

void Run(AgcDriver::VulkanDevice& device, const Case& test, std::uint32_t waveSize, std::span<const std::uint32_t> code) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) Input[tid * Stride] = tid * 0x01010101u + 7u;
    Output.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(SelectorRegister + 6u, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size()));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size()));
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    userData[SelectorRegister] = test.selector;
    userData[SelectorRegister + 2u] = test.limit;
    userData[SelectorRegister + 4u] = test.innerLimit;
    userData[SelectorRegister + 5u] = test.tailLimit;
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

void Check(const Case& test, std::uint32_t waveSize) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto expected = Input[tid * Stride] + test.added;
        const auto actual = Output[tid * Stride];
        Require(actual == expected, "loop ending exits wave" + std::to_string(waveSize) + ": " + test.name + ": thread " + std::to_string(tid) + " is " + std::to_string(actual) + ", expected " + std::to_string(expected));
        for (std::uint32_t padding = 1u; padding < Stride; ++padding) {
            Require(Output[tid * Stride + padding] == 0xdeadbeefu, "loop ending exits wave" + std::to_string(waveSize) + ": " + test.name + ": thread " + std::to_string(tid) + " overwrote padding " + std::to_string(padding));
        }
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        for (const auto waveSize : {32u, 64u}) {
            for (const auto& test : Cases) {
                Run(*device, test, waveSize, Code);
                Check(test, waveSize);
            }
            for (const auto& test : NestedCases) {
                Run(*device, test, waveSize, NestedCode);
                Check(test, waveSize);
            }
            for (const auto& test : SelectionCases) {
                Run(*device, test, waveSize, SelectionCode);
                Check(test, waveSize);
            }
            for (const auto& test : NestedSelectionCases) {
                Run(*device, test, waveSize, NestedSelectionCode);
                Check(test, waveSize);
            }
            for (const auto& test : CyclicSelectionCases) {
                Run(*device, test, waveSize, CyclicSelectionCode);
                Check(test, waveSize);
            }
        }
        std::puts("loop ending exits tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
