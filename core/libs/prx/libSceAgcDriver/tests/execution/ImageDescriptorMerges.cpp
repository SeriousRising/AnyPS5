#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Size = 8;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t SampleX = 3;
constexpr std::uint32_t ImageWord = 0;
constexpr std::uint32_t SamplerWord = 32;
constexpr std::uint32_t OutputWord = 40;
constexpr float Tolerance = 1.0f / 64.0f;
constexpr std::array<std::uint32_t, 4> PointSampler{0u, 0x00fff000u, 0u, 0u};
constexpr std::array<std::uint32_t, 4> LinearSampler{0x00000092u, 0x00fff000u, 0x00500000u, 0u};
constexpr std::array<std::uint8_t, 4> Reds{255u, 192u, 128u, 64u};
constexpr std::array<std::uint8_t, 4> Greens{40u, 100u, 160u, 220u};

struct alignas(4096) GuestData {
    std::array<std::uint32_t, 48> srt{};
    std::array<std::uint32_t, 16> output{};
};

GuestData Guest;
alignas(256) std::array<std::array<std::uint8_t, 4096>, 4> Textures{};

alignas(256) constexpr std::array<std::uint32_t, 27> SeparateCode{
    0xbf068002, 0xbf850003, 0xf40c0200, 0xfa000000, 0xbf820002, 0xf40c0200, 0xfa000020, 0xbf068003,
    0xbf850003, 0xf4080500, 0xfa000080, 0xbf820002, 0xf4080500, 0xfa000090, 0xf4080600, 0xfa0000a0,
    0xbf8cc07f, 0x7e0002ff, 0x3ef00000, 0x7e0202ff, 0x3f300000, 0xf09c0f08, 0x00a20000, 0xbf8c3f70,
    0xe0780000, 0x80060000, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 38> NestedCode{
    0xbf068002, 0xbf850008, 0xbf068003, 0xbf850003, 0xf40c0200, 0xfa000000, 0xbf820002, 0xf40c0200,
    0xfa000020, 0xbf820008, 0xbf068003, 0xbf850003, 0xf40c0200, 0xfa000040, 0xbf820002, 0xf40c0200,
    0xfa000060, 0xbf800000, 0xbf068004, 0xbf850003, 0xf4080500, 0xfa000080, 0xbf820002, 0xf4080500,
    0xfa000090, 0xf4080600, 0xfa0000a0, 0xbf8cc07f, 0x7e0002ff, 0x3ef00000, 0x7e0202ff, 0x3f300000,
    0xf09c0f08, 0x00a20000, 0xbf8c3f70, 0xe0780000, 0x80060000, 0xbf810000,
};

std::array<std::uint32_t, 8> TextureDescriptor(const void* texels) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texels));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format8888UNorm << 20u) | (((Size - 1u) & 3u) << 30u),
        ((Size - 1u) >> 2u) | ((Size - 1u) << 14u),
        0xfacu | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

void Fill() {
    const auto mips = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format8888UNorm, Size, Size, 1u);
    Require(AgcDriver::Graphics::ComputeSurfaceSize(mips, 1) <= Textures[0].size(), "image descriptor merges: a texture does not fit its storage");
    for (std::uint32_t index = 0; index < Textures.size(); ++index) {
        Textures[index].fill(0x5au);
        for (std::uint32_t y = 0; y < Size; ++y) {
            for (std::uint32_t x = 0; x < Size; ++x) {
                auto* texel = &Textures[index][static_cast<std::size_t>(y) * mips[0].pitchBytes + x * 4u];
                texel[0] = x > SampleX ? Reds[index] : std::uint8_t{0u};
                texel[1] = Greens[index];
                texel[2] = 0u;
                texel[3] = 255u;
            }
        }
        const auto descriptor = TextureDescriptor(Textures[index].data());
        std::copy(descriptor.begin(), descriptor.end(), Guest.srt.begin() + ImageWord + index * 8u);
    }
    std::copy(PointSampler.begin(), PointSampler.end(), Guest.srt.begin() + SamplerWord);
    std::copy(LinearSampler.begin(), LinearSampler.end(), Guest.srt.begin() + SamplerWord + 4u);
    const auto output = reinterpret_cast<std::uintptr_t>(Guest.output.data());
    const std::array<std::uint32_t, 4> buffer{static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), static_cast<std::uint32_t>(Guest.output.size() * 4u), 0x31016facu};
    std::copy(buffer.begin(), buffer.end(), Guest.srt.begin() + OutputWord);
}

void Check(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::array<std::uint32_t, 3>& selectors, std::uint32_t texture, bool linear, const std::string& what) {
    Guest.output.fill(0xdeadbeefu);
    const auto srt = reinterpret_cast<std::uintptr_t>(Guest.srt.data());
    const std::array<std::uint32_t, 5> userData{static_cast<std::uint32_t>(srt), static_cast<std::uint32_t>(srt >> 32u), selectors[0], selectors[1], selectors[2]};
    const std::array regions{ShaderRecompiler::MemoryRegion{reinterpret_cast<std::uintptr_t>(&Guest), std::as_bytes(std::span(&Guest, 1u))}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{1u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0u, {}},
        {32u, 0u, userData, compute, std::nullopt, std::nullopt, regions},
        device.Target(),
        {0u, 0u, 0u, 128u}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1u, 1u, 1u);
    device.WaitIdle();
    const std::array<float, 4> expected{linear ? static_cast<float>(Reds[texture]) / 1020.0f : 0.0f, static_cast<float>(Greens[texture]) / 255.0f, 0.0f, 1.0f};
    for (std::uint32_t component = 0; component < 4u; ++component) {
        const auto actual = std::bit_cast<float>(Guest.output[component]);
        Require(std::fabs(actual - expected[component]) <= Tolerance, what + ": selectors " + std::to_string(selectors[0]) + std::to_string(selectors[1]) + std::to_string(selectors[2]) + " component " + std::to_string(component) + " is " + std::to_string(actual) + ", expected " + std::to_string(expected[component]));
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Fill();
        for (std::uint32_t image = 0; image < 2u; ++image) {
            for (std::uint32_t sampler = 0; sampler < 2u; ++sampler) {
                Check(*device, SeparateCode, {image, sampler, 0u}, image != 0u ? 0u : 1u, sampler == 0u, "T# and S# selected at different merges");
            }
        }
        for (std::uint32_t outer = 0; outer < 2u; ++outer) {
            for (std::uint32_t inner = 0; inner < 2u; ++inner) {
                for (std::uint32_t sampler = 0; sampler < 2u; ++sampler) {
                    Check(*device, NestedCode, {outer, inner, sampler}, (outer != 0u ? 0u : 2u) + (inner != 0u ? 0u : 1u), sampler == 0u, "T# selected through nested merges");
                }
            }
        }
        std::puts("image descriptor merges tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
