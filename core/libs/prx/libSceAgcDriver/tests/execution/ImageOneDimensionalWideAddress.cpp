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
constexpr std::uint32_t Width = 32;
constexpr std::uint32_t Format32323232Float = 77;
constexpr std::uint32_t Type1D = 8;
constexpr std::uint32_t IdentitySwizzle = 0xfacu;
constexpr std::size_t TexelBytes = 8192;
constexpr std::size_t BlockBytes = 65536;

alignas(256) std::array<std::uint32_t, Threads * 4> Buffer{};

alignas(256) constexpr std::array<std::uint32_t, 9> LoadCode{
    0x34020084, 0x7e3c0300, 0x7e3e0280, 0xf0001f08, 0x0001141e, 0xbf8c3f70, 0xe0781000, 0x80001401,
    0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 9> StoreCode{
    0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0280, 0xf0201f08, 0x0001141e,
    0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 13> ExchangeCode{
    0x34020084, 0xe0381000, 0x80001401, 0x7e3c0300, 0x7e3e0285, 0xf0001f08, 0x0001181e, 0xbf8c3f70,
    0xf0201f08, 0x0001141e, 0xe0781000, 0x80001801, 0xbf810000,
};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "one-dimensional image wide address: cannot allocate the guest block");
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

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

std::uint32_t Initial(std::uint32_t x, std::uint32_t component) {
    return std::bit_cast<std::uint32_t>(-(static_cast<float>(x * 4u + component) + 0.5f));
}

std::uint32_t Stored(std::uint32_t x, std::uint32_t component) {
    return std::bit_cast<std::uint32_t>(static_cast<float>(x * 4u + component) + 0.25f);
}

std::array<std::uint32_t, 4> BufferDescriptor() {
    const auto address = reinterpret_cast<std::uintptr_t>(Buffer.data());
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(Buffer.size() * 4u), 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(const void* data) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32323232Float << 20u) | (((Width - 1u) & 3u) << 30u),
        (Width - 1u) >> 2u,
        IdentitySwizzle | (Type1D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

std::uint8_t* Texel(std::uint8_t* texels, std::uint32_t x) {
    const auto surface = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(TextureDescriptor(texels)));
    Require(surface.guestBytes <= TexelBytes, "one-dimensional image wide address: the row does not fit the texel storage");
    return texels + surface.mips.at(0).tiledOffset + static_cast<std::uint64_t>(x) * 16u;
}

void FillTexels(std::uint8_t* texels) {
    std::fill(texels, texels + TexelBytes, std::uint8_t{0xeeu});
    for (std::uint32_t x = 0; x < Width; ++x) {
        for (std::uint32_t component = 0; component < 4u; ++component) {
            const auto value = Initial(x, component);
            std::memcpy(Texel(texels, x) + component * 4u, &value, 4u);
        }
    }
}

void FillBuffer() {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t component = 0; component < 4u; ++component) Buffer[tid * 4u + component] = Stored(tid, component);
    }
}

void Dispatch(AgcDriver::VulkanDevice& device, std::uint8_t* texels, std::span<const std::uint32_t> code) {
    std::vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor();
    const auto texture = TextureDescriptor(texels);
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
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
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(texels), TexelBytes, nullptr, "test");
    device.WaitIdle();
}

void CheckBuffer(std::uint32_t (*expected)(std::uint32_t, std::uint32_t), const std::string& what) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t component = 0; component < 4u; ++component) {
            const auto actual = Buffer[tid * 4u + component];
            const auto wanted = expected(tid, component);
            Require(actual == wanted, what + ": thread " + std::to_string(tid) + " component " + std::to_string(component) + " is " + Hex(actual) + ", expected " + Hex(wanted));
        }
    }
}

void CheckTexels(std::uint8_t* texels, const std::string& what) {
    for (std::uint32_t x = 0; x < Width; ++x) {
        for (std::uint32_t component = 0; component < 4u; ++component) {
            std::uint32_t actual = 0;
            std::memcpy(&actual, Texel(texels, x) + component * 4u, 4u);
            const auto wanted = Stored(x, component);
            Require(actual == wanted, what + ": texel " + std::to_string(x) + " component " + std::to_string(component) + " is " + Hex(actual) + ", expected " + Hex(wanted));
        }
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock block;
        auto* texels = block.Data();
        FillTexels(texels);
        Buffer.fill(0xdeadbeefu);
        Dispatch(*device, texels, LoadCode);
        CheckBuffer(Initial, "image_load 2d on a 1D texture");
        FillBuffer();
        Dispatch(*device, texels, StoreCode);
        CheckTexels(texels, "image_store 2d on a 1D texture");
        Buffer.fill(0xdeadbeefu);
        Dispatch(*device, texels, LoadCode);
        CheckBuffer(Stored, "image_load 2d after image_store 2d on a 1D texture");
        FillTexels(texels);
        FillBuffer();
        Dispatch(*device, texels, ExchangeCode);
        CheckBuffer(Initial, "image_load 2d at row 5 of a 1D texture the shader also stores to");
        CheckTexels(texels, "image_store 2d at row 5 of a 1D texture the shader also loads");
        std::puts("one-dimensional image wide address tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
