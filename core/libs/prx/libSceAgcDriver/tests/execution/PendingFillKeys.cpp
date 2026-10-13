#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "VulkanTestDevice.hpp"
#include <mutex>
#include <string_view>
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
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 512;
constexpr std::uint32_t Height = 256;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Rgba8Info = 0x8028u;
constexpr std::uint32_t TiledAttrib3 = 0x4dc6c000u;

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

alignas(256) std::array<std::array<float, 4>, 3> Triangle{{
    {-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}
}};

alignas(256) std::array<std::array<std::uint32_t, 64>, 8> Programs{};
std::size_t programCount = 0;

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, 0x01016facu};
}

std::span<const std::uint32_t> Program(std::array<float, 4> srgb8, bool rgba8) {
    Require(programCount < Programs.size(), "too many test programs");
    auto& code = Programs[programCount++];
    std::size_t at = 0;
    const auto move = [&](std::uint32_t vgpr, float value) {
        code[at++] = 0x7e0002ffu | (vgpr << 17u);
        code[at++] = std::bit_cast<std::uint32_t>(value);
    };
    for (std::uint32_t channel = 0; channel < 4; ++channel) move(4 + channel, srgb8[channel]);
    if (rgba8) {
        for (std::uint32_t channel = 0; channel < 4; ++channel) move(8 + channel, channel % 2 == 0 ? 1.0f : 0.0f);
        code[at++] = 0xf800000fu;
        code[at++] = 0x0b0a0908u;
        code[at++] = 0xf800181fu;
    } else {
        code[at++] = 0xf800180fu;
    }
    code[at++] = 0x07060504u;
    code[at++] = 0xbf810000u;
    return std::span<const std::uint32_t>(code.data(), at);
}

struct Block {
    explicit Block(std::size_t bytes) : bytes(bytes), watched(AgcDriver::GuestMemory::WriteWatched()) {
#ifdef _WIN32
        if (watched) {
            data = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, BlockBytes));
            GuestArena::GuestArenaCommit_nid_postfix(data, bytes, PAGE_READWRITE, bytes);
        } else {
            data = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        }
#else
        if (watched) {
            void* raw = mmap(nullptr, bytes + BlockBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            Require(raw != MAP_FAILED, "cannot map the color target");
            const auto begin = reinterpret_cast<std::uintptr_t>(raw);
            const auto aligned = (begin + BlockBytes - 1) & ~(static_cast<std::uintptr_t>(BlockBytes) - 1);
            if (aligned != begin) munmap(raw, aligned - begin);
            if (aligned + bytes != begin + bytes + BlockBytes) munmap(reinterpret_cast<void*>(aligned + bytes), begin + BlockBytes - aligned);
            data = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(data, bytes);
        } else {
            data = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, bytes));
        }
#endif
        Require(data != nullptr, "cannot allocate the color target");
        Require(!watched || AgcDriver::GuestMemory::Watched(Address(), bytes), "the color target is not write-watched");
        GuestAllocations::Mutation mutation;
        mutation.Add(data, bytes, true, true, true);
    }
    ~Block() {
        AgcDriver::Graphics::StorageTexture::FlushPending(Address(), bytes, nullptr, "test release");
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(data);
        }
#ifdef _WIN32
        if (watched) {
            GuestArena::GuestArenaReset_nid_postfix(data, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(data, bytes);
        } else {
            VirtualFree(data, 0, MEM_RELEASE);
        }
#else
        if (watched) {
            munmap(data, bytes);
            GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(data, bytes);
        } else {
            std::free(data);
        }
#endif
    }
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(data); }
    std::size_t bytes;
    bool watched;
    std::uint8_t* data = nullptr;
};


constexpr std::uint32_t FastClear = 0x2000u;
constexpr std::uint32_t ClearWord = 0x11223344u;
constexpr std::size_t CmaskBytes = 4096;

AgcDriver::Graphics::ColorTarget DecodeTarget(const Block& block, const Block& cmask) {
    AgcDriver::Registers cx;
    cx[0x318] = static_cast<std::uint32_t>(block.Address() >> 8u);
    cx[0x31b] = 0;
    cx[0x31c] = Rgba8Info | FastClear;
    cx[0x31d] = 0;
    cx[0x31f] = static_cast<std::uint32_t>(cmask.Address() >> 8u);
    cx[0x323] = ClearWord;
    cx[0x324] = 0;
    cx[0x3b0] = ((Width - 1u) << 14u) | (Height - 1u);
    cx[0x3b8] = TiledAttrib3;
    cx[0x390] = static_cast<std::uint32_t>(block.Address() >> 40u);
    cx[0x398] = static_cast<std::uint32_t>(cmask.Address() >> 40u);
    auto color = AgcDriver::Graphics::DecodeColorBuffer(cx, 0);
    Require(color.bytes == block.bytes && color.cmaskAddress == cmask.Address() && color.cmaskBytes == CmaskBytes, "the fast-clear target did not decode over its blocks");
    color.exportIndex = 0;
    return color;
}

struct Prepared {
    ShaderRecompiler::RecompileResult vertex, pixel;
    std::uint32_t vertexPush = 0;
};

Prepared Prepare(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> pixelCode) {
    const auto target = device.Target();
    constexpr std::uint32_t waveSize = 64;
    static std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(Triangle.data(), 16u, static_cast<std::uint32_t>(Triangle.size()));
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {waveSize, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    Prepared prepared;
    prepared.vertex = ShaderRecompiler::Recompile(vertex);
    prepared.vertexPush = AgcDriver::Graphics::StagePushOffset(static_cast<std::uint32_t>(prepared.vertex.pushConstants.size()), ShaderStage::Fragment, device.GraphicsPipelineLibraries());
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode.at(0) = 9;
    pixel.targetExportMapping.fill(0xe4u);
    static const std::vector<std::uint32_t> pixelUserData(8, 0u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(pixelCode.data()), std::as_bytes(pixelCode)}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(pixelCode.data()), pixelCode, 0, {}},
        {waveSize, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        {0, 0, prepared.vertexPush, 128 - prepared.vertexPush}
    };
    fragment.useCache = false;
    prepared.pixel = ShaderRecompiler::Recompile(fragment);
    return prepared;
}

struct Scene {
    explicit Scene(AgcDriver::VulkanDevice& device) : device(device), prepared(Prepare(device, Program({0.5f, 0.25f, 0.5f, 0.75f}, false))) {
        std::memset(surface.data, 0x5a, surface.bytes);
        std::memset(cmask.data, 0xff, cmask.bytes);
        state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
        state.colors = {DecodeTarget(surface, cmask)};
        state.color = state.colors.front();
        state.hasColorTarget = true;
        state.renderExtent = {Width, Height};
        state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
        state.negativeOneToOne = false;
        state.scissor = {{0, 0}, {Width / 2, Height}};
        state.cullMode = VK_CULL_MODE_NONE;
        state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        state.blends = {VkPipelineColorBlendAttachmentState{}};
        state.blends[0].colorWriteMask = 0xf;
        state.blend = state.blends.front();
        Draw();
        device.WaitIdle();
    }

    void Draw() {
        const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
            {ShaderStage::Vertex, &prepared.vertex, 0},
            {ShaderStage::Fragment, &prepared.pixel, prepared.vertexPush}
        }};
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        device.Draw(state, AgcDriver::Pm4::DrawParameters{0, static_cast<std::uint32_t>(Triangle.size()), 0, 1, 0, false}, shaders);
    }

    void Fill(std::size_t offset, std::size_t bytes, std::array<std::uint32_t, 4> pattern) {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Require(device.FillBuffer(cmask.Address() + offset, bytes, pattern), "a fill of the CMASK was not recorded on the device");
    }

    std::uint64_t Submissions() const {
        return AgcDriver::Graphics::Recorder::Active()->Submissions();
    }

    std::vector<std::uint8_t> Stored() {
        AgcDriver::Graphics::StorageTexture::FlushPending(surface.Address(), surface.bytes, nullptr, "test read-back");
        device.WaitIdle();
        std::vector<std::uint8_t> stored(surface.bytes + cmask.bytes);
        AgcDriver::GuestMemory::Read(surface.Address(), std::as_writable_bytes(std::span(stored.data(), surface.bytes)), 1);
        AgcDriver::GuestMemory::Read(cmask.Address(), std::as_writable_bytes(std::span(stored.data() + surface.bytes, cmask.bytes)), 1);
        return stored;
    }

    AgcDriver::VulkanDevice& device;
    Prepared prepared;
    Block surface{8 * BlockBytes};
    Block cmask{BlockBytes};
    AgcDriver::Graphics::State state{};
};

constexpr std::array<std::uint32_t, 4> Cleared{0, 0, 0, 0};
constexpr std::array<std::uint32_t, 4> Unknown{0x00000000u, 0x00000000u, 0x00000000u, 0x01000000u};

enum class Fills { Whole, OverUnknown, HalfOverZeros };

std::vector<std::uint8_t> ClearedThenDrawn(AgcDriver::VulkanDevice& device, bool waited, Fills fills) {
    Scene scene(device);
    if (fills == Fills::OverUnknown) scene.Fill(0, CmaskBytes, Unknown);
    if (fills == Fills::HalfOverZeros) std::memset(scene.cmask.data + CmaskBytes / 2, 0, CmaskBytes / 2);
    scene.Fill(0, fills == Fills::HalfOverZeros ? CmaskBytes / 2 : CmaskBytes, Cleared);
    if (waited) device.WaitIdle();
    const auto before = scene.Submissions();
    scene.Draw();
    Require(waited || scene.Submissions() == before, "a draw over a CMASK that a recorded fill clears submitted and waited for the fill");
    const auto stored = scene.Stored();
    std::uint32_t drawn = ClearWord;
    std::size_t clearTexels = 0;
    for (std::size_t offset = 0; offset < scene.surface.bytes; offset += 4) {
        std::uint32_t texel;
        std::memcpy(&texel, stored.data() + offset, 4);
        if (texel == ClearWord) {
            ++clearTexels;
            continue;
        }
        if (drawn == ClearWord) drawn = texel;
        Require(texel == drawn, "a texel of the fast-cleared target is neither the clear word nor the drawn color");
    }
    Require(drawn != ClearWord && clearTexels == std::size_t{Width} * Height / 2, "the fast clear was not eliminated into the texels the draw left");
    Require(std::all_of(stored.begin() + scene.surface.bytes, stored.begin() + scene.surface.bytes + CmaskBytes, [](std::uint8_t key) { return key == 0xff; }), "the drawn target's CMASK was not stored as expanded");
    return stored;
}

void ClearedAgain(AgcDriver::VulkanDevice& device, bool waited) {
    Scene scene(device);
    scene.Fill(0, CmaskBytes, Cleared);
    scene.Draw();
    if (waited) device.WaitIdle();
    scene.Fill(0, CmaskBytes, Cleared);
    if (waited) device.WaitIdle();
    scene.state.scissor = {{Width / 2, 0}, {Width / 2, Height}};
    scene.Draw();
    const auto stored = scene.Stored();
    std::size_t clearTexels = 0;
    for (std::size_t offset = 0; offset < scene.surface.bytes; offset += 4) {
        std::uint32_t texel;
        std::memcpy(&texel, stored.data() + offset, 4);
        if (texel == ClearWord) ++clearTexels;
    }
    Require(clearTexels == std::size_t{Width} * Height / 2, std::string(waited ? "waited" : "pending") + ": a target fast-cleared, drawn over its left half, fast-cleared again by the same fill and drawn over its right half holds " + std::to_string(clearTexels) + " clear texels, expected " + std::to_string(std::size_t{Width} * Height / 2) + " (the second clear must take the first frame's left half)");
}

void PendingFillTests(AgcDriver::VulkanDevice& device) {
    const auto waited = ClearedThenDrawn(device, true, Fills::Whole);
    Require(ClearedThenDrawn(device, false, Fills::Whole) == waited, "a draw over a CMASK cleared by a pending fill stored other bytes than one that waited for the fill");
    Require(ClearedThenDrawn(device, false, Fills::OverUnknown) == waited, "a pending fill over an earlier recorded write of the CMASK stored other bytes than one that was waited for");
    Require(ClearedThenDrawn(device, false, Fills::HalfOverZeros) == waited, "a pending fill beside cleared bytes of the CMASK stored other bytes than one that was waited for");
    ClearedAgain(device, true);
    ClearedAgain(device, false);
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        PendingFillTests(*device);
        std::puts("pending fill key tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
