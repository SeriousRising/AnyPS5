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
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Words = 64;
constexpr std::uint32_t Width = 16;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Format16161616Float = 71;
constexpr std::uint32_t Type1D = 8;
constexpr std::uint32_t ResultWord = 16;
constexpr std::uint32_t OffsetOperation = 3;
alignas(256) std::array<std::uint32_t, Threads * Words> Buffer{};
alignas(4096) std::array<std::uint8_t, 8192> Texels{};
alignas(4096) std::array<std::uint8_t, 8192> SingleTexel{};

alignas(256) constexpr std::array<std::uint32_t, 93> SampleCode{
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

alignas(256) constexpr std::array<std::uint32_t, 11> GatherCode{
    0x34020088, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xbf8c3f70, 0xf11c0108, 0x00611402,
    0xe0701040, 0x80001401, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 7> SizeCode{
    0x34020088, 0x7e040280, 0xf0380f08, 0x00011402, 0xe0701040, 0x80001401, 0xbf810000,
};

constexpr std::array<const char*, 6> Names{"image_sample_l", "image_sample_lz", "image_sample", "image_sample_lz_o", "image_sample_d", "image_sample_b"};

std::uint32_t Bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

std::uint32_t ColumnOf(std::uint32_t tid) {
    return (tid * 7u + 3u) % (Width - 1u);
}

std::uint8_t Red(std::uint32_t x) {
    return static_cast<std::uint8_t>(x * 13u + 5u);
}

void FillInput(std::uint32_t width) {
    Buffer.fill(0xdeadbeefu);
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto u = Bits((static_cast<float>(ColumnOf(tid) % width) + 0.5f) / static_cast<float>(width));
        const auto v = Bits(0.8125f);
        const auto slope = Bits(1.0f / 1024.0f);
        const std::array<std::uint32_t, 15> input{u, v, Bits(0.0f), 1u | (3u << 8u), u, v, slope, slope, slope, slope, u, v, Bits(0.0f), u, v};
        std::copy(input.begin(), input.end(), &Buffer[tid * Words]);
    }
}

std::array<std::uint32_t, 4> BufferDescriptor() {
    const auto address = reinterpret_cast<std::uintptr_t>(Buffer.data());
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(Buffer.size() * 4u), 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(const void* data, std::uint32_t format, std::uint32_t width) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((width - 1u) & 3u) << 30u),
        (width - 1u) >> 2u,
        0xfacu | (Type1D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

std::array<std::uint32_t, 4> SamplerDescriptor() {
    return {0u, 0xfffu << 12u, 1u << 26u, 0u};
}

std::uint64_t RowOffset(const std::array<std::uint32_t, 8>& descriptor, std::size_t capacity) {
    const auto surface = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(descriptor));
    Require(surface.guestBytes <= capacity, "one-dimensional image wide sample: the row does not fit the texel storage");
    return surface.mips.at(0).tiledOffset;
}

std::array<std::uint32_t, 8> FillRow() {
    const auto descriptor = TextureDescriptor(Texels.data(), Format8888UNorm, Width);
    Texels.fill(0xeeu);
    auto* row = Texels.data() + RowOffset(descriptor, Texels.size());
    for (std::uint32_t x = 0; x < Width; ++x) {
        const std::array<std::uint8_t, 4> texel{Red(x), 0x40u, 0x80u, 0xffu};
        std::copy(texel.begin(), texel.end(), row + x * 4u);
    }
    return descriptor;
}

std::array<std::uint32_t, 8> FillSingleTexel() {
    const auto descriptor = TextureDescriptor(SingleTexel.data(), Format16161616Float, 1u);
    SingleTexel.fill(0u);
    constexpr std::array<std::uint16_t, 4> texel{0x3c00u, 0x4000u, 0x3800u, 0xc000u};
    std::memcpy(SingleTexel.data() + RowOffset(descriptor, SingleTexel.size()), texel.data(), sizeof(texel));
    return descriptor;
}

void Run(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::array<std::uint32_t, 8>& texture) {
    std::vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor();
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

float Result(std::uint32_t tid, std::uint32_t operation, std::uint32_t component) {
    return std::bit_cast<float>(Buffer[tid * Words + ResultWord + operation * 4u + component]);
}

void CheckRow() {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t operation = 0; operation < Names.size(); ++operation) {
            const auto x = ColumnOf(tid) + (operation == OffsetOperation ? 1u : 0u);
            const std::array<std::uint32_t, 4> expected{Red(x), 0x40u, 0x80u, 0xffu};
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const float value = Result(tid, operation, component) * 255.0f;
                Require(std::lround(value) == static_cast<long>(expected[component]), std::string(Names[operation]) + " 2d on a 1D texture: thread " + std::to_string(tid) + " component " + std::to_string(component) + " is " + std::to_string(value) + ", expected " + std::to_string(expected[component]));
            }
        }
    }
}

void CheckSingleTexel() {
    constexpr std::array<float, 4> expected{1.0f, 2.0f, 0.5f, -2.0f};
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t operation = 0; operation < Names.size(); ++operation) {
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const float value = Result(tid, operation, component);
                Require(value == expected[component], std::string(Names[operation]) + " 2d on a 1x1 1D texture: thread " + std::to_string(tid) + " component " + std::to_string(component) + " is " + std::to_string(value) + ", expected " + std::to_string(expected[component]));
            }
        }
    }
}

void RequireRefused(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::array<std::uint32_t, 8>& texture, const std::string& what) {
    std::string refusal;
    try {
        Run(device, code, texture);
    } catch (const std::exception& error) {
        refusal = error.what();
    }
    Require(refusal.find("image descriptor is incompatible with the static runtime image interface") != std::string::npos, what + " was not refused: " + refusal);
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto row = FillRow();
        FillInput(Width);
        Run(*device, SampleCode, row);
        CheckRow();
        const auto single = FillSingleTexel();
        FillInput(1u);
        Run(*device, SampleCode, single);
        CheckSingleTexel();
        RequireRefused(*device, GatherCode, row, "image_gather4_lz 2d on a 1D texture");
        RequireRefused(*device, SizeCode, row, "image_get_resinfo 2d on a 1D texture");
        std::puts("one-dimensional image wide sample tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
