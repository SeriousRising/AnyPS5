#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

extern "C" int APS5_VABI sceKernelMunmap(void*, std::size_t);
extern "C" int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
extern "C" int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
extern "C" int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::StorageTexture;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Side = 256;
constexpr std::uint32_t Format8888Unorm = 56;
constexpr std::uint32_t TileRenderTarget64KB = 0x1b;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t Unit = 65536;
constexpr std::size_t SurfaceBytes = 4 * Unit;
constexpr std::size_t KeyBytes = SurfaceBytes / 256;
constexpr std::size_t KeysOffset = SurfaceBytes;
constexpr std::size_t BlockBytes = SurfaceBytes + Unit;

alignas(256) constexpr std::array<std::uint32_t, 8> WriteCode{
    0x7e020280, 0x7e0402f2, 0x7e0602f2, 0x7e0802f2, 0x7e0a02f2, 0xf0201f08, 0x00010200, 0xbf810000,
};

std::uint64_t AddressOf(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

class GuestBlock {
public:
    GuestBlock() {
        Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, BlockBytes, Unit, 0, &physical) == 0, "clear capture publish: cannot allocate direct memory");
        void* mapped = nullptr;
        Require(sceKernelMapDirectMemory(&mapped, BlockBytes, 0x33, 0, physical, Unit) == 0 && mapped != nullptr, "clear capture publish: cannot map direct memory");
        block = static_cast<std::uint8_t*>(mapped);
    }

    ~GuestBlock() {
        sceKernelMunmap(block, BlockBytes);
        sceKernelReleaseDirectMemory(physical, BlockBytes);
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() { return block; }

private:
    std::int64_t physical = 0;
    std::uint8_t* block = nullptr;
};

std::array<std::uint32_t, 8> TextureDescriptor(const std::uint8_t* texels, const std::uint8_t* keys) {
    const auto address = AddressOf(texels);
    const auto meta = AddressOf(keys) >> 8u;
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format8888Unorm << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        0xfacu | (TileRenderTarget64KB << 20u) | (Type2D << 28u),
        0u,
        0u,
        (1u << 20u) | (1u << 21u) | (static_cast<std::uint32_t>(meta & 0xffu) << 24u),
        static_cast<std::uint32_t>(meta >> 8u),
    };
}

void Dispatch(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::vector<std::uint32_t>& userData, bool wait = true) {
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
    if (wait) device.WaitIdle();
}

std::size_t Count(const std::uint8_t* bytes, std::size_t size, std::uint8_t value) {
    return static_cast<std::size_t>(std::count(bytes, bytes + size, value));
}

int Run(AgcDriver::VulkanDevice& device, std::uint8_t* block) {
    using AgcDriver::Graphics::AnyShadowedOverlaps;
    auto* texels = block;
    auto* keys = block + KeysOffset;
    const auto surface = AddressOf(texels);
    std::memset(texels, 0x55, SurfaceBytes);
    std::memset(keys, 0x00, KeyBytes);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    auto plain = TextureDescriptor(texels, nullptr);
    plain[6] = 0;
    plain[7] = 0;
    std::vector<std::uint32_t> plainData(16, 0u);
    std::copy(plain.begin(), plain.end(), plainData.begin() + 4);
    const auto keyed = TextureDescriptor(texels, keys);
    std::vector<std::uint32_t> keyedData(16, 0u);
    std::copy(keyed.begin(), keyed.end(), keyedData.begin() + 4);
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    Dispatch(device, WriteCode, plainData);
    Require(StorageTexture::FindPending(surface, SurfaceBytes) != nullptr, "clear capture publish: the written surface has no pending image");
    StorageTexture::FlushPending(surface, SurfaceBytes, nullptr, "storage refresh", AgcDriver::Graphics::PublishScope::None);
    std::size_t shadowed = 0;
    for (std::size_t unit = 0; unit < SurfaceBytes / Unit; ++unit) shadowed += AnyShadowedOverlaps(surface + unit * Unit, Unit) ? 1u : 0u;
    if (shadowed < 2) {
        std::printf("skipped, the image's results were stored into %zu unit shadows, not several\n", shadowed);
        return VulkanTestSkipped;
    }
    Require(Count(texels, SurfaceBytes, 0x55) == SurfaceBytes, "clear capture publish: results stored into the unit shadow reached guest memory");
    auto* recorder = AgcDriver::Graphics::Recorder::Active();
    Require(recorder != nullptr, "clear capture publish: no active recorder");
    const auto before = recorder->Submissions();
    const auto readBefore = StorageTexture::SnapshotBytesRead();
    Dispatch(device, WriteCode, keyedData, false);
    char text[256];
    std::snprintf(text, sizeof(text), "clear capture publish: the upload under clear keys submitted %llu batches and read %llu bytes of the surface, expected neither", static_cast<unsigned long long>(recorder->Submissions() - before), static_cast<unsigned long long>(StorageTexture::SnapshotBytesRead() - readBefore));
    Require(recorder->Submissions() == before && StorageTexture::SnapshotBytesRead() == readBefore, text);
    std::size_t kept = 0;
    for (std::size_t unit = 0; unit < SurfaceBytes / Unit; ++unit) kept += AnyShadowedOverlaps(surface + unit * Unit, Unit) ? 1u : 0u;
    Require(kept == shadowed && Count(texels, SurfaceBytes, 0x55) == SurfaceBytes, "clear capture publish: the upload under clear keys published units of the surface");
    Require(StorageTexture::PublishShadowsOnly(surface, SurfaceBytes, AgcDriver::Graphics::PublishScope::Whole), "clear capture publish: the units kept in the shadow could not be published");
    device.WaitIdle();
    const auto published = Count(texels, SurfaceBytes, 0xff);
    std::snprintf(text, sizeof(text), "clear capture publish: the published results are %zu written bytes and %zu untouched of %zu, expected %u written", published, Count(texels, SurfaceBytes, 0x55), SurfaceBytes, Threads * 4u);
    Require(published == Threads * 4u && Count(texels, SurfaceBytes, 0x55) == SurfaceBytes - published, text);
    StorageTexture::FlushPending(surface, SurfaceBytes, nullptr, "test");
    device.WaitIdle();
    Require(Count(keys, KeyBytes, 0xff) == KeyBytes, "clear capture publish: the write under clear keys did not store the keys uncompressed");
    Require(Count(texels, SurfaceBytes, 0xff) == Threads * 4u && Count(texels, SurfaceBytes, 0x00) == SurfaceBytes - Threads * 4u, "clear capture publish: the write under clear keys stored other texels than its results over the clear");
    return 0;
}

}

int main() {
    try {
        std::optional<GuestBlock> watched;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (!AgcDriver::GuestMemory::WriteWatched()) {
            std::puts("skipped, guest memory has no write watch");
            return VulkanTestSkipped;
        }
        watched.emplace();
        const auto result = Run(*device, watched->Data());
        if (result == 0) std::puts("clear capture publish tests passed");
        return result;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
