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
    {-1.0f, -1.0f, 0.5f, 1.0f}, {-0.98f, -1.0f, 0.5f, 1.0f}, {-1.0f, -0.98f, 0.5f, 1.0f}
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


AgcDriver::Graphics::ColorTarget DecodeTarget(std::uint32_t slot, const Block& block, std::uint32_t info, std::uint32_t attrib3) {
    const auto stride = slot * 0xfu;
    AgcDriver::Registers cx;
    cx[0x318 + stride] = static_cast<std::uint32_t>(block.Address() >> 8u);
    cx[0x31b + stride] = 0;
    cx[0x31c + stride] = info;
    cx[0x31d + stride] = 0;
    cx[0x323 + stride] = 0;
    cx[0x324 + stride] = 0;
    cx[0x3b0 + slot] = ((Width - 1u) << 14u) | (Height - 1u);
    cx[0x3b8 + slot] = attrib3;
    cx[0x390 + slot] = static_cast<std::uint32_t>(block.Address() >> 40u);
    auto color = AgcDriver::Graphics::DecodeColorBuffer(cx, slot);
    Require(color.bytes <= block.bytes, "the color target is larger than its block");
    color.exportIndex = slot;
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

using Detail = AgcDriver::Graphics::Recorder::DrawTimingDetail;
using Row = AgcDriver::Graphics::Recorder::DrawTimingRow;

AgcDriver::Graphics::State StateFor(const AgcDriver::Graphics::ColorTarget& color, std::optional<AgcDriver::Graphics::DepthTarget> depth) {
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    state.colors = {color};
    state.color = color;
    state.hasColorTarget = true;
    state.depth = depth;
    state.depthTest = depth.has_value();
    state.depthWrite = depth.has_value();
    state.depthCompare = VK_COMPARE_OP_ALWAYS;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blends = {VkPipelineColorBlendAttachmentState{}};
    state.blends[0].colorWriteMask = 0xf;
    state.blend = state.blends.front();
    return state;
}

Row Find(VkFormat depth) {
    for (const auto& row : AgcDriver::Graphics::Recorder::DrawGpuTimes()) {
        if (row.detail.depth == depth) return row;
    }
    return {};
}

void TimedDrawTests(AgcDriver::VulkanDevice& device, const Block& first, const Block& second, const Block& depthBlock) {
    const auto prepared = Prepare(device, Program({0.5f, 0.25f, 0.5f, 0.75f}, false));
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &prepared.vertex, 0},
        {ShaderStage::Fragment, &prepared.pixel, prepared.vertexPush}
    }};
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Triangle.size()), 0, 1, 0, false};
    const auto plain = StateFor(DecodeTarget(0, first, Rgba8Info, TiledAttrib3), std::nullopt);
    const auto deep = StateFor(DecodeTarget(0, second, Rgba8Info, TiledAttrib3), AgcDriver::Graphics::DepthTarget{depthBlock.Address(), 0, {Width, Height}, VK_FORMAT_D32_SFLOAT, 1.0f, 0});
    device.Draw(plain, draw, shaders);
    device.Draw(deep, draw, shaders);
    device.WaitIdle();
    const auto plainBefore = Find(VK_FORMAT_UNDEFINED);
    const auto deepBefore = Find(VK_FORMAT_D32_SFLOAT);
    Require(plainBefore.passes != 0 && deepBefore.passes != 0, "the draws of a profiled run left no draw timing rows");
    device.Draw(plain, draw, shaders);
    device.Draw(plain, draw, shaders);
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::Graphics::Recorder::Active()->Commands();
    }
    device.Draw(plain, draw, shaders);
    device.Draw(deep, draw, shaders);
    device.WaitIdle();
    const auto rows = AgcDriver::Graphics::Recorder::DrawGpuTimes();
    Require(rows.size() == 2, "two targets drawn with one pixel program gave " + std::to_string(rows.size()) + " draw timing rows");
    const auto plainAfter = Find(VK_FORMAT_UNDEFINED);
    const auto deepAfter = Find(VK_FORMAT_D32_SFLOAT);
    Require(plainAfter.passes - plainBefore.passes == 2 && plainAfter.draws - plainBefore.draws == 3, "two draws in one pass and a third in its own were timed as " + std::to_string(plainAfter.passes - plainBefore.passes) + " passes of " + std::to_string(plainAfter.draws - plainBefore.draws) + " draws");
    Require(deepAfter.passes - deepBefore.passes == 1 && deepAfter.draws - deepBefore.draws == 1, "one draw with a depth target was not timed as one pass of one draw");
    Require(plainAfter.ms > plainBefore.ms && deepAfter.ms > deepBefore.ms, "a timed pass added no GPU time");
    const auto& words = prepared.pixel.spirv.Words();
    for (const auto& row : {plainAfter, deepAfter}) {
        const auto& detail = row.detail;
        const std::uint8_t depthUse = detail.depth != VK_FORMAT_UNDEFINED ? Detail::TestsDepth | Detail::WritesDepth : 0;
        Require(detail.width == Width && detail.height == Height && detail.color == VK_FORMAT_R8G8B8A8_UNORM && detail.colors == 1, "a draw timing row does not name its target");
        Require(detail.pixelProgram == prepared.pixel.PipelineVariantId() && detail.pixelWords == words.size() && detail.pixelTraits == (AgcDriver::Graphics::PixelProgramTraits(words) | depthUse), "a draw timing row does not name its pixel program and depth use");
    }
    for (const auto* block : {&first, &second}) {
        AgcDriver::Graphics::StorageTexture::FlushPending(block->Address(), block->bytes, nullptr, "test read-back");
        device.WaitIdle();
        std::vector<std::byte> stored(block->bytes);
        AgcDriver::GuestMemory::Read(block->Address(), stored, 1);
    }
}

std::uint8_t Traits(std::initializer_list<std::uint32_t> body, std::initializer_list<std::uint32_t> helper = {}, bool called = false) {
    constexpr std::uint32_t entryPoint = (3u << 16u) | 15u, function = (5u << 16u) | 54u, functionEnd = (1u << 16u) | 56u, call = (4u << 16u) | 57u;
    std::vector<std::uint32_t> words{0x07230203u, 0x00010500u, 0u, 64u, 0u, entryPoint, 4, 40};
    words.insert(words.end(), {function, 30, 41, 0, 31});
    words.insert(words.end(), helper);
    words.push_back(functionEnd);
    words.insert(words.end(), {function, 30, 40, 0, 31});
    words.insert(words.end(), body);
    if (called) words.insert(words.end(), {call, 30, 42, 41});
    words.push_back(functionEnd);
    return AgcDriver::Graphics::PixelProgramTraits(words);
}

void TraitTests() {
    constexpr std::uint32_t pointer = (4u << 16u) | 32u, variable = (4u << 16u) | 59u, chain = (4u << 16u) | 65u, store = (3u << 16u) | 62u, mode = (3u << 16u) | 16u, kill = (1u << 16u) | 252u;
    Require(Traits({mode, 40, 9}) == Detail::EarlyTests, "an EarlyFragmentTests mode was not read");
    Require(Traits({mode, 40, 12}) == Detail::ExportsDepth, "a DepthReplacing mode was not read");
    Require(Traits({kill}) == Detail::Kills, "an OpKill in the entry point was not read");
    Require(Traits({}, {kill}, true) == Detail::Kills, "an OpKill in a called function was not read");
    Require(Traits({}, {kill}, false) == 0, "an OpKill in a function nothing calls was read as a kill");
    Require(Traits({pointer, 2, 12, 3, variable, 2, 4, 12, chain, 2, 5, 4, store, 5, 6}) == Detail::WritesMemory, "a store through a storage buffer access chain was not read");
    Require(Traits({}, {pointer, 2, 12, 3, variable, 2, 4, 12, store, 4, 6}, false) == 0, "a store in a function nothing calls was read as a store to memory");
    Require(Traits({pointer, 2, 5349, 3, (4u << 16u) | 120u, 2, 5, 7, (6u << 16u) | 232u, 8, 9, 5, 10, 11}) == Detail::WritesMemory, "an atomic update through a physical storage buffer pointer was not read");
    Require(Traits({pointer, 2, 7, 3, variable, 2, 4, 7, store, 4, 6}) == 0, "a store to a function variable was read as a store to memory");
    Require(Traits({pointer, 2, 12, 3, variable, 2, 4, 12, chain, 2, 5, 4, (4u << 16u) | 61u, 3, 6, 5}) == 0, "a load from a storage buffer was read as a store to memory");
    Require(Traits({(4u << 16u) | 99u, 4, 5, 6}) == Detail::WritesMemory, "an image write was not read");
}

}

int main() {
    try {
#ifdef _WIN32
        _putenv_s("APS5_PROFILE_GPU", "1");
#else
        setenv("APS5_PROFILE_GPU", "1", 1);
#endif
        TraitTests();
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Block first(8 * BlockBytes);
        Block second(8 * BlockBytes);
        Block depth(BlockBytes);
        TimedDrawTests(*device, first, second, depth);
        std::puts("draw GPU timing tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
