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
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Words = 64;
constexpr std::uint32_t Size = 16;
constexpr std::uint32_t Mips = 5;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Type1D = 8;
constexpr std::uint32_t ResultWord = 16;
constexpr std::uint32_t SampleL = 0x24;
constexpr std::uint32_t SampleCl = 0x21;
constexpr std::uint32_t SampleC = 0x28;
alignas(256) std::array<std::uint32_t, Threads * Words> Buffer{};
alignas(256) std::array<std::uint8_t, 16384> Line{};

alignas(256) constexpr std::array<std::uint32_t, 93> FormsCode{
    0x34020088, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c,
    0x80000501, 0xe0301010, 0x80000601, 0xe0301014, 0x80000701, 0xe0301018, 0x80000801, 0xe030101c,
    0x80000901, 0xe0301020, 0x80000a01, 0xe0301024, 0x80000b01, 0xe0301028, 0x80000c01, 0xe030102c,
    0x80000d01, 0xe0301030, 0x80000e01, 0xe0301034, 0x80000f01, 0xe0301038, 0x80001001, 0xbf8c3f70,
    0xf0900f08, 0x00611402, 0xe0701040, 0x80001401, 0xe0701044, 0x80001501, 0xe0701048, 0x80001601,
    0xe070104c, 0x80001701, 0xf09c0f08, 0x00611402, 0xe0701050, 0x80001401, 0xe0701054, 0x80001501,
    0xe0701058, 0x80001601, 0xe070105c, 0x80001701, 0xf0800f08, 0x00611402, 0xe0701060, 0x80001401,
    0xe0701064, 0x80001501, 0xe0701068, 0x80001601, 0xe070106c, 0x80001701, 0xf0dc0f08, 0x00611405,
    0xe0701070, 0x80001401, 0xe0701074, 0x80001501, 0xe0701078, 0x80001601, 0xe070107c, 0x80001701,
    0xf0880f08, 0x00611408, 0xe0701080, 0x80001401, 0xe0701084, 0x80001501, 0xe0701088, 0x80001601,
    0xe070108c, 0x80001701, 0xf0940f08, 0x0061140e, 0xe0701090, 0x80001401, 0xe0701094, 0x80001501,
    0xe0701098, 0x80001601, 0xe070109c, 0x80001701, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 16> OneSampleCode{
    0x34020088, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c,
    0x80000501, 0xbf8c3f70, 0xf0900f08, 0x00611402, 0xbf8c3f70, 0xe0701040, 0x80001401, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 11> GatherCode{
    0x34020088, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xbf8c3f70, 0xf11c0108, 0x00611402,
    0xe0701040, 0x80001401, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 7> SizeCode{
    0x34020088, 0x7e040280, 0xf0380f08, 0x00011402, 0xe0701040, 0x80001401, 0xbf810000,
};

alignas(256) std::array<std::uint32_t, 16> PatchedCode{};

enum Form : std::uint32_t { ExplicitLod, LevelZero, Implicit, LevelZeroOffset, Derivative, Bias, FormCount };
constexpr std::array<const char*, FormCount> Names{"image_sample_l", "image_sample_lz", "image_sample", "image_sample_lz_o", "image_sample_d", "image_sample_b"};
constexpr std::array<float, 4> SecondCoordinates{0.0f, 0.8125f, 7.25f, -3.5f};

std::uint32_t Bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

std::uint32_t TexelOf(std::uint32_t tid) {
    return 1u + (tid * 5u) % 13u;
}

std::int32_t OffsetOf(std::uint32_t tid) {
    return (tid & 1u) != 0u ? 2 : -1;
}

std::uint32_t LodOf(std::uint32_t tid) {
    return tid % Mips;
}

std::uint32_t GradientLevelOf(std::uint32_t tid) {
    return (tid / 2u) % Mips;
}

std::uint8_t LevelMark(std::uint32_t level) {
    return static_cast<std::uint8_t>(10u + 40u * level);
}

std::uint8_t TexelMark(std::uint32_t x) {
    return static_cast<std::uint8_t>(x * 8u + 3u);
}

void FillInput() {
    Buffer.fill(0xdeadbeefu);
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto u = Bits((static_cast<float>(TexelOf(tid)) + 0.5f) / static_cast<float>(Size));
        const auto v = Bits(SecondCoordinates[tid % SecondCoordinates.size()]);
        const auto secondOffset = (tid & 2u) != 0u ? 31u : 32u;
        const auto offset = (static_cast<std::uint32_t>(OffsetOf(tid)) & 0x3fu) | (secondOffset << 8u);
        const auto step = Bits(static_cast<float>(1u << GradientLevelOf(tid)) / static_cast<float>(Size));
        const bool alongX = (tid & 1u) == 0u;
        const std::array<std::uint32_t, 15> input{
            u, v, Bits(static_cast<float>(LodOf(tid))),
            offset, u, v,
            alongX ? step : Bits(0.0f), Bits(64.0f), alongX ? Bits(0.0f) : step, Bits(-48.0f), u, v,
            Bits(0.0f), u, v,
        };
        std::copy(input.begin(), input.end(), &Buffer[tid * Words]);
    }
}

std::array<std::uint32_t, 4> BufferDescriptor() {
    const auto address = reinterpret_cast<std::uintptr_t>(Buffer.data());
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(Buffer.size() * 4u), 0x31016facu};
}

std::array<std::uint32_t, 8> LineDescriptor() {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Line.data()));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format8888UNorm << 20u) | (((Size - 1u) & 3u) << 30u),
        (Size - 1u) >> 2u,
        0xfacu | ((Mips - 1u) << 16u) | (Type1D << 28u),
        0u,
        (Mips - 1u) << 4u,
        0u,
        0u,
    };
}

std::array<std::uint32_t, 4> SamplerDescriptor() {
    return {0u, 0xfffu << 12u, 1u << 26u, 0u};
}

void FillTexture() {
    const auto geometry = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(LineDescriptor()));
    Require(geometry.guestBytes <= Line.size(), "sample forms through a 1D texture: the texture does not fit the texel storage");
    Require(geometry.mips.size() == Mips, "sample forms through a 1D texture: the texture has " + std::to_string(geometry.mips.size()) + " mips");
    std::fill(Line.begin(), Line.end(), 0xeeu);
    for (std::uint32_t level = 0; level < Mips; ++level) {
        const auto& mip = geometry.mips[level];
        for (std::uint32_t x = 0; x < (Size >> level); ++x) {
            auto* texel = &Line[geometry.GuestLayerOffset(0) + mip.tiledOffset + x * 4u];
            texel[0] = LevelMark(level);
            texel[1] = TexelMark(x);
            texel[2] = 99u;
            texel[3] = 255u;
        }
    }
}

void Run(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code) {
    std::vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor();
    const auto texture = LineDescriptor();
    const auto sampler = SamplerDescriptor();
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    std::copy(sampler.begin(), sampler.end(), userData.begin() + 12);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

void Check() {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t form = 0; form < FormCount; ++form) {
            const auto level = form == ExplicitLod ? LodOf(tid) : form == Derivative ? GradientLevelOf(tid) : 0u;
            const auto x = form == LevelZeroOffset ? static_cast<std::uint32_t>(static_cast<std::int32_t>(TexelOf(tid)) + OffsetOf(tid)) : TexelOf(tid) >> level;
            const std::array<std::uint32_t, 4> expected{LevelMark(level), TexelMark(x), 99u, 255u};
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const float value = std::bit_cast<float>(Buffer[tid * Words + ResultWord + form * 4u + component]) * 255.0f;
                Require(std::lround(value) == static_cast<long>(expected[component]), std::string(Names[form]) + " 2d on a 1D texture: thread " + std::to_string(tid) + " component " + std::to_string(component) + " is " + std::to_string(value) + ", expected " + std::to_string(expected[component]));
            }
        }
    }
}

std::span<const std::uint32_t> OneSample(std::uint32_t opcode) {
    PatchedCode = OneSampleCode;
    PatchedCode[10] = (PatchedCode[10] & ~(0x3fu << 18u)) | (opcode << 18u);
    return PatchedCode;
}

void RequireRefused(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::string& what) {
    std::string refusal;
    try {
        Run(device, code);
    } catch (const std::exception& error) {
        refusal = error.what();
    }
    Require(refusal.find("image descriptor is incompatible with the static runtime image interface") != std::string::npos, what + " 2d on a 1D texture was not refused: " + refusal);
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Require(OneSampleCode[10] >> 18u == (0xf0000000u >> 18u | SampleL), "sample forms through a 1D texture: the one-sample program does not hold image_sample_l");
        FillTexture();
        FillInput();
        Run(*device, FormsCode);
        Check();
        RequireRefused(*device, OneSample(SampleCl), "image_sample_cl");
        RequireRefused(*device, OneSample(SampleC), "image_sample_c");
        RequireRefused(*device, GatherCode, "image_gather4_lz");
        RequireRefused(*device, SizeCode, "image_get_resinfo");
        std::puts("image plane sample forms on line tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
