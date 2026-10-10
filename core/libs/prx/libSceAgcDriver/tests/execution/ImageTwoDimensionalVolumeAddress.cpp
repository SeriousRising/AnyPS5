#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
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
#include <bit>
#include <cmath>
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
constexpr std::uint32_t Words = 64;
constexpr std::uint32_t Size = 16;
constexpr std::uint32_t StoreWidth = 32;
constexpr std::uint32_t StoreHeight = 4;
constexpr std::uint32_t StoreRow = 1;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Format32323232Float = 77;
constexpr std::uint32_t Type1D = 8;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t Type2DArray = 13;
constexpr std::uint32_t ResultWord = 24;
constexpr std::uint32_t OffsetOperation = 3;
constexpr std::size_t TexelBytes = 8192;
constexpr std::size_t BlockBytes = 65536;
constexpr std::array<std::uint8_t, 4> SingleTexel{0x33u, 0x66u, 0x99u, 0xffu};
alignas(256) std::array<std::uint32_t, Threads * Words> Buffer{};
alignas(256) std::array<std::uint32_t, Threads * 4> StoreInput{};
alignas(4096) std::array<std::uint8_t, TexelBytes> Texels{};
alignas(4096) std::array<std::uint8_t, TexelBytes> Single{};

alignas(256) constexpr std::array<std::uint32_t, 121> ReadCode{
    0x34020088, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c,
    0x80000501, 0xe0301010, 0x80000601, 0xe0301014, 0x80000701, 0xe0301018, 0x80000801, 0xe030101c,
    0x80000901, 0xe0301020, 0x80000a01, 0xe0301024, 0x80000b01, 0xe0301028, 0x80000c01, 0xe030102c,
    0x80000d01, 0xe0301030, 0x80000e01, 0xe0301034, 0x80000f01, 0xe0301038, 0x80001001, 0xe030103c,
    0x80001101, 0xe0301040, 0x80001201, 0xe0301044, 0x80001301, 0xe0301048, 0x80001401, 0xe030104c,
    0x80001501, 0xe0301050, 0x80001601, 0xe0301054, 0x80001701, 0xe0301058, 0x80001801, 0xe030105c,
    0x80001901, 0xbf8c3f70, 0xf0900f10, 0x00611c02, 0xe0701060, 0x80001c01, 0xe0701064, 0x80001d01,
    0xe0701068, 0x80001e01, 0xe070106c, 0x80001f01, 0xf09c0f10, 0x00611c02, 0xe0701070, 0x80001c01,
    0xe0701074, 0x80001d01, 0xe0701078, 0x80001e01, 0xe070107c, 0x80001f01, 0xf0800f10, 0x00611c02,
    0xe0701080, 0x80001c01, 0xe0701084, 0x80001d01, 0xe0701088, 0x80001e01, 0xe070108c, 0x80001f01,
    0xf0dc0f10, 0x00611c06, 0xe0701090, 0x80001c01, 0xe0701094, 0x80001d01, 0xe0701098, 0x80001e01,
    0xe070109c, 0x80001f01, 0xf0880f10, 0x00611c0a, 0xe07010a0, 0x80001c01, 0xe07010a4, 0x80001d01,
    0xe07010a8, 0x80001e01, 0xe07010ac, 0x80001f01, 0xf0940f10, 0x00611c13, 0xe07010b0, 0x80001c01,
    0xe07010b4, 0x80001d01, 0xe07010b8, 0x80001e01, 0xe07010bc, 0x80001f01, 0xf0001f10, 0x00011c17,
    0xe07010c0, 0x80001c01, 0xe07010c4, 0x80001d01, 0xe07010c8, 0x80001e01, 0xe07010cc, 0x80001f01,
    0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 10> StoreCode{
    0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0281, 0x7e400285, 0xf0201f10,
    0x0001141e, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 11> GatherCode{
    0x34020088, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xbf8c3f70, 0xf11c0110, 0x00611c02,
    0xe0701060, 0x80001c01, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 7> SizeCode{
    0x34020088, 0x7e040280, 0xf0380f10, 0x00011c02, 0xe0701060, 0x80001c01, 0xbf810000,
};

constexpr std::array<const char*, 7> Names{"image_sample_l", "image_sample_lz", "image_sample", "image_sample_lz_o", "image_sample_d", "image_sample_b", "image_load"};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "two-dimensional image volume address: cannot allocate the guest block");
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

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

std::uint32_t Bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

std::uint32_t ColumnOf(std::uint32_t tid) {
    return (tid * 7u + 3u) % (Size - 1u);
}

std::uint32_t RowOf(std::uint32_t tid) {
    return (tid * 5u + 1u) % (Size - 1u);
}

std::array<std::uint8_t, 4> Texel(std::uint32_t x, std::uint32_t y) {
    return {static_cast<std::uint8_t>(x * 13u + 5u), static_cast<std::uint8_t>(y * 11u + 3u), 0x80u, 0xffu};
}

std::uint32_t Initial(std::uint32_t x, std::uint32_t y, std::uint32_t component) {
    return Bits(-(static_cast<float>((y * StoreWidth + x) * 4u + component) + 0.5f));
}

std::uint32_t Stored(std::uint32_t x, std::uint32_t component) {
    return Bits(static_cast<float>(x * 4u + component) + 0.25f);
}

void FillInput(std::uint32_t size) {
    Buffer.fill(0xdeadbeefu);
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto x = ColumnOf(tid) % size;
        const auto y = RowOf(tid) % size;
        const auto u = Bits((static_cast<float>(x) + 0.5f) / static_cast<float>(size));
        const auto v = Bits((static_cast<float>(y) + 0.5f) / static_cast<float>(size));
        const auto w = Bits(0.8125f);
        const auto slope = Bits(1.0f / 1024.0f);
        const std::array<std::uint32_t, 24> input{u, v, w, Bits(0.0f), 1u | (1u << 8u) | (3u << 16u), u, v, w, slope, slope, slope, slope, slope, slope, u, v, w, Bits(0.0f), u, v, w, x, y, 5u};
        std::copy(input.begin(), input.end(), &Buffer[tid * Words]);
    }
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(bytes), 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(const void* data, std::uint32_t type, std::uint32_t format, std::uint32_t width, std::uint32_t height) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((width - 1u) & 3u) << 30u),
        ((width - 1u) >> 2u) | ((height - 1u) << 14u),
        0xfacu | (type << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

std::array<std::uint32_t, 4> SamplerDescriptor() {
    return {0u, 0xfffu << 12u, 1u << 26u, 0u};
}

AgcDriver::Graphics::TileMipLayout Mip(std::uint32_t format, std::uint32_t width, std::uint32_t height) {
    const auto mips = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, format, width, height, 1u);
    Require(AgcDriver::Graphics::ComputeSurfaceSize(mips, 1) <= TexelBytes, "two-dimensional image volume address: a texture does not fit its storage");
    return mips.at(0);
}

void FillTexels() {
    const auto mip = Mip(Format8888UNorm, Size, Size);
    Texels.fill(0xeeu);
    for (std::uint32_t y = 0; y < Size; ++y) {
        for (std::uint32_t x = 0; x < Size; ++x) {
            const auto texel = Texel(x, y);
            std::copy(texel.begin(), texel.end(), &Texels[mip.tiledOffset + static_cast<std::uint64_t>(y) * mip.pitchBytes + x * 4u]);
        }
    }
    Single.fill(0u);
    std::copy(SingleTexel.begin(), SingleTexel.end(), &Single[Mip(Format8888UNorm, 1u, 1u).tiledOffset]);
}

void Run(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::array<std::uint32_t, 4>& buffer, const std::array<std::uint32_t, 8>& texture) {
    std::vector<std::uint32_t> userData(16, 0u);
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

void Read(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::array<std::uint32_t, 8>& texture) {
    Run(device, code, BufferDescriptor(Buffer.data(), Buffer.size() * 4u), texture);
}

void CheckRead(bool single) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t operation = 0; operation < Names.size(); ++operation) {
            const auto shift = operation == OffsetOperation ? 1u : 0u;
            const auto expected = single ? SingleTexel : Texel(ColumnOf(tid) + shift, RowOf(tid) + shift);
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const float value = std::bit_cast<float>(Buffer[tid * Words + ResultWord + operation * 4u + component]) * 255.0f;
                Require(std::lround(value) == static_cast<long>(expected[component]), std::string(Names[operation]) + " 3d on a " + (single ? "1x1 " : "") + "2D texture: thread " + std::to_string(tid) + " component " + std::to_string(component) + " is " + std::to_string(value) + ", expected " + std::to_string(expected[component]));
            }
        }
    }
}

void CheckStore(AgcDriver::VulkanDevice& device) {
    GuestBlock block;
    auto* texels = block.Data();
    const auto mip = Mip(Format32323232Float, StoreWidth, StoreHeight);
    const auto texel = [&](std::uint32_t x, std::uint32_t y) { return texels + mip.tiledOffset + static_cast<std::uint64_t>(y) * mip.pitchBytes + x * 16u; };
    for (std::uint32_t y = 0; y < StoreHeight; ++y) {
        for (std::uint32_t x = 0; x < StoreWidth; ++x) {
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const auto value = Initial(x, y, component);
                std::memcpy(texel(x, y) + component * 4u, &value, 4u);
            }
        }
    }
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t component = 0; component < 4u; ++component) StoreInput[tid * 4u + component] = Stored(tid, component);
    }
    Run(device, StoreCode, BufferDescriptor(StoreInput.data(), StoreInput.size() * 4u), TextureDescriptor(texels, Type2D, Format32323232Float, StoreWidth, StoreHeight));
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(texels), TexelBytes, nullptr, "test");
    device.WaitIdle();
    for (std::uint32_t y = 0; y < StoreHeight; ++y) {
        for (std::uint32_t x = 0; x < StoreWidth; ++x) {
            for (std::uint32_t component = 0; component < 4u; ++component) {
                std::uint32_t actual = 0;
                std::memcpy(&actual, texel(x, y) + component * 4u, 4u);
                const auto wanted = y == StoreRow ? Stored(x, component) : Initial(x, y, component);
                Require(actual == wanted, "image_store 3d on a 2D texture: texel (" + std::to_string(x) + ", " + std::to_string(y) + ") component " + std::to_string(component) + " is " + Hex(actual) + ", expected " + Hex(wanted));
            }
        }
    }
}

void RequireRefused(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::array<std::uint32_t, 8>& texture, const std::string& reason, const std::string& what) {
    std::string refusal;
    try {
        Read(device, code, texture);
    } catch (const std::exception& error) {
        refusal = error.what();
    }
    Require(refusal.find(reason) != std::string::npos, what + " was not refused: " + refusal);
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const std::string incompatible = "image descriptor is incompatible with the static runtime image interface";
        FillTexels();
        const auto texture = TextureDescriptor(Texels.data(), Type2D, Format8888UNorm, Size, Size);
        FillInput(Size);
        Read(*device, ReadCode, texture);
        CheckRead(false);
        FillInput(1u);
        Read(*device, ReadCode, TextureDescriptor(Single.data(), Type2D, Format8888UNorm, 1u, 1u));
        CheckRead(true);
        CheckStore(*device);
        FillInput(Size);
        RequireRefused(*device, GatherCode, texture, incompatible, "image_gather4_lz 3d on a 2D texture");
        RequireRefused(*device, SizeCode, texture, incompatible, "image_get_resinfo 3d on a 2D texture");
        RequireRefused(*device, ReadCode, TextureDescriptor(Texels.data(), Type2DArray, Format8888UNorm, Size, Size), "a 3D image address on a 2D array texture is not implemented", "image_sample 3d on a 2D array texture");
        RequireRefused(*device, ReadCode, TextureDescriptor(Texels.data(), Type1D, Format8888UNorm, Size, 1u), incompatible, "image_sample 3d on a 1D texture");
        std::puts("two-dimensional image volume address tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
