#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
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
using AgcDriver::Graphics::StorageTexture;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Side = 128;
constexpr std::uint32_t Format32UInt = 20;
// Linear: tracked per layer, so the block-unit borrow does not apply and APS5_LAYER_ALIAS_COPY does.
constexpr std::uint32_t TileLinear = 0;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t Type2DArray = 13;
constexpr std::size_t SurfaceBytes = 65536;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Initial = 0x5a5a5a5au;

alignas(256) constexpr std::array<std::uint32_t, 5> StoreCode{
    0x7e020280, 0x7e04020c, 0xf0201108, 0x00010200, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 9> LoadArrayCode{
    0x7e020280, 0x7e040280, 0xf0001128, 0x00010300, 0x34080082, 0xbf8c3f70, 0xe0701000, 0x80000304, 0xbf810000,
};

alignas(256) std::array<std::uint32_t, Threads> Output{};

std::uint64_t AddressOf(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

class GuestBlock {
public:
    explicit GuestBlock(bool watched) {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
        static_cast<void>(watched);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, BlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + BlockBytes != begin + BlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            if (watched) GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "layer alias copy: cannot allocate the guest block");
        Require(!watched || AgcDriver::GuestMemory::Watched(AddressOf(block), BlockBytes), "layer alias copy: the guest block is not write-watched");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(block, BlockBytes);
        GuestArena::GuestArenaRelease_nid_postfix(block, BlockBytes);
#else
        munmap(block, BlockBytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, BlockBytes);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() { return block; }

private:
    std::uint8_t* block = nullptr;
};

std::array<std::uint32_t, 8> TextureDescriptor(const std::uint8_t* texels, std::uint32_t type) {
    const auto address = AddressOf(texels);
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32UInt << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        0xfacu | (TileLinear << 20u) | (type << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = AddressOf(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

void Dispatch(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::vector<std::uint32_t>& userData) {
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

void Store(AgcDriver::VulkanDevice& device, const std::uint8_t* texels, std::uint32_t value) {
    std::vector<std::uint32_t> userData(16, 0u);
    const auto texture = TextureDescriptor(texels, Type2D);
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    userData[12] = value;
    Dispatch(device, StoreCode, userData);
}

void LoadArray(AgcDriver::VulkanDevice& device, const std::uint8_t* texels) {
    Output.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    const auto texture = TextureDescriptor(texels, Type2DArray);
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    Dispatch(device, LoadArrayCode, userData);
}

void RequireOutput(std::uint32_t expected, const std::string& what) {
    for (std::uint32_t x = 0; x < Threads; ++x) {
        Require(Output[x] == expected, what + ": texel (" + std::to_string(x) + ", 0) reads " + Hex(Output[x]) + ", expected " + Hex(expected));
    }
}

// Guest memory still holds `before`: nothing was stored over it.
void RequireGuestUnchanged(const std::uint8_t* texels, const std::vector<std::uint32_t>& before, const std::string& what) {
    const auto* words = reinterpret_cast<const std::uint32_t*>(texels);
    Require(std::equal(before.begin(), before.end(), words), what + ": guest memory was written (the results were stored instead of handed over on the device)");
}

void Run(AgcDriver::VulkanDevice& device, std::uint8_t* texels) {
    const auto surface = AddressOf(texels);
    std::fill_n(reinterpret_cast<std::uint32_t*>(texels), SurfaceBytes / 4u, Initial);
    AgcDriver::GuestMemory::CollectWrites(surface, SurfaceBytes);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    // First round: creating the array image stores the 2D image's results (creation, not refresh).
    Store(device, texels, 0x11111111u);
    Require(StorageTexture::FindPending(surface, SurfaceBytes) != nullptr, "the 2D store left no pending image");
    LoadArray(device, texels);
    RequireOutput(0x11111111u, "the first load through the one-layer array");
    const std::vector<std::uint32_t> before(reinterpret_cast<const std::uint32_t*>(texels), reinterpret_cast<const std::uint32_t*>(texels) + SurfaceBytes / 4u);
    // Both images exist now: each refresh takes the other's pending layer on the device.
    Store(device, texels, 0x22222222u);
    RequireGuestUnchanged(texels, before, "after the second store");
    LoadArray(device, texels);
    RequireOutput(0x22222222u, "the second load through the one-layer array");
    RequireGuestUnchanged(texels, before, "after the second load");
    const auto taken = StorageTexture::FindPending(surface, SurfaceBytes);
    Require(taken != nullptr && taken->Descriptor().dimension == AgcDriver::Graphics::TextureDimension::k2DArray, "the array image did not take the pending layer");
    Store(device, texels, 0x33333333u);
    RequireGuestUnchanged(texels, before, "after the third store");
    LoadArray(device, texels);
    RequireOutput(0x33333333u, "the third load through the one-layer array");
    StorageTexture::FlushPending(surface, SurfaceBytes, nullptr, "test");
    device.WaitIdle();
    Require(std::count(reinterpret_cast<const std::uint32_t*>(texels), reinterpret_cast<const std::uint32_t*>(texels) + SurfaceBytes / 4u, 0x33333333u) == static_cast<std::ptrdiff_t>(Threads), "guest memory does not hold the 32 texels of the third store after the flush");
    AgcDriver::Graphics::ClearCachedTextures(device.Device());
}

}

int main() {
    try {
        const bool watched = AgcDriver::GuestMemory::WriteWatched();
        GuestBlock block(watched);
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device, block.Data());
        std::puts("storage layer alias copy tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
