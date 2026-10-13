#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
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
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 32;
constexpr std::uint32_t Format32Float = 22;
constexpr std::size_t BlockBytes = 1048576;
constexpr std::size_t SliceBytes = 65536;
constexpr std::uint32_t Slices = 4;
constexpr std::uint32_t Type2DArray = 13;
constexpr std::size_t HtileOffset = 0x80000;
constexpr std::size_t HtileBytes = 0x8000;
constexpr std::uint32_t ClearedToOne = 0xfffffff0u;

alignas(256) std::array<std::uint32_t, Threads * 4> Buffer{};
alignas(256) std::array<std::byte, Width * Height * 4> Pixels{};

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 4> PixelCode{
    0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000,
};

std::array<std::uint32_t, 10> LoadCode(std::uint32_t slice) {
    return {0x34020084, 0x7e3c0300, 0x7e3e0280, 0x7e400280 | slice, 0xf0001f28, 0x0001141e, 0xbf8c3f70, 0xe0781000, 0x80001401, 0xbf810000};
}

constexpr std::byte Kept{0x40};

using Vertices = std::array<std::array<float, 4>, 3>;

constexpr Vertices At(float z) {
    return {{{-1.0f, -1.0f, z, 1.0f}, {3.0f, -1.0f, z, 1.0f}, {-1.0f, 3.0f, z, 1.0f}}};
}

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "depth array slice clear: cannot allocate the guest block");
        std::memset(block, 0, BlockBytes);
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

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, stride == 0u ? 0x31016facu : 0x01016facu};
}

std::array<std::uint32_t, 8> ArrayDescriptor(const void* texels) {
    const std::uint32_t format = Format32Float;
    const std::uint32_t width = Width;
    const std::uint32_t height = Height;
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texels));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((width - 1u) & 3u) << 30u),
        ((width - 1u) >> 2u) | ((height - 1u) << 14u),
        0xfacu | (Type2DArray << 28u),
        Slices - 1u,
        0u,
        0u,
        0u,
    };
}

void DrawDepth(AgcDriver::VulkanDevice& device, const std::uint8_t* base, std::uint32_t slice, const Vertices& triangle) {
    const auto* depth = base + slice * SliceBytes;
    Pixels.fill(Kept);
    const auto target = device.Target();
    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(triangle.data(), 16u, static_cast<std::uint32_t>(triangle.size()));
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {64u, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = AgcDriver::Graphics::StagePushOffset(static_cast<std::uint32_t>(vertexResult.pushConstants.size()), ShaderStage::Fragment, device.GraphicsPipelineLibraries());

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.wave32 = false;
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    const std::vector<std::uint32_t> pixelUserData(8, 0u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}},
        {64u, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        {0, 0, vertexPush, 128 - vertexPush}
    };
    fragment.useCache = false;
    const auto pixelResult = ShaderRecompiler::Recompile(fragment);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertexResult, 0},
        {ShaderStage::Fragment, &pixelResult, vertexPush}
    }};

    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.depth = AgcDriver::Graphics::DepthTarget{reinterpret_cast<std::uintptr_t>(depth), 0u, {Width, Height}, VK_FORMAT_D32_SFLOAT, 1.0f, 0u, reinterpret_cast<std::uintptr_t>(base + HtileOffset), false};
    state.depth->htileSlice = slice;
    state.depthTest = true;
    state.depthWrite = true;
    state.depthCompare = VK_COMPARE_OP_LESS;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(triangle.size()), 0, 1, 0, false};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
}

void Dispatch(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const std::array<std::uint32_t, 8>& texture) {
    std::vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor(Buffer.data(), 0u, static_cast<std::uint32_t>(Buffer.size() * 4u));
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
}

void CheckSlice(AgcDriver::VulkanDevice& device, const std::uint8_t* base, std::uint32_t slice, float expected, const std::string& what) {
    Buffer.fill(0xdeadbeefu);
    const auto code = LoadCode(slice);
    Dispatch(device, code, ArrayDescriptor(base));
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto actual = std::bit_cast<float>(Buffer[tid * 4u]);
        Require(actual == expected, what + ": thread " + std::to_string(tid) + " loaded depth " + std::to_string(actual) + " from slice " + std::to_string(slice) + ", expected " + std::to_string(expected));
    }
}

void Fill(const std::uint8_t* base, std::uint32_t firstSlice, std::uint32_t slices) {
    AgcDriver::Graphics::NoteDepthMetadataFill(reinterpret_cast<std::uintptr_t>(base + HtileOffset + firstSlice * HtileBytes), slices * HtileBytes, ClearedToOne);
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock block;
        const auto* base = block.Data();
        static constexpr Vertices first = At(0.25f);
        static constexpr Vertices second = At(0.5f);
        static constexpr Vertices third = At(0.75f);
        for (std::uint32_t slice = 0; slice < 3u; ++slice) {
            Fill(base, slice, 1u);
            DrawDepth(*device, base, slice, first);
        }
        for (std::uint32_t slice = 0; slice < 3u; ++slice) CheckSlice(*device, base, slice, 0.25f, "first frame");
        CheckSlice(*device, base, 3u, 1.0f, "slice no draw used as a depth target");
        for (std::uint32_t slice = 3u; slice-- != 0u;) {
            Fill(base, slice, 1u);
            DrawDepth(*device, base, slice, second);
        }
        for (std::uint32_t slice = 0; slice < 3u; ++slice) CheckSlice(*device, base, slice, 0.5f, "second frame, a farther draw after a fill of only this slice's HTILE tiles (0.25 is the first frame's depth, never cleared)");
        Fill(base, 0u, 1u);
        CheckSlice(*device, base, 0u, 1.0f, "the first slice after a fill of its HTILE tiles");
        CheckSlice(*device, base, 1u, 0.5f, "the second slice after a fill of only the first slice's HTILE tiles");
        CheckSlice(*device, base, 2u, 0.5f, "the third slice after a fill of only the first slice's HTILE tiles");
        Fill(base, 1u, 1u);
        DrawDepth(*device, base, 1u, third);
        CheckSlice(*device, base, 1u, 0.75f, "a slice sampled before, then cleared and drawn farther");
        Fill(base, 2u, 1u);
        CheckSlice(*device, base, 2u, 1.0f, "a slice sampled before, then cleared and not drawn");
        CheckSlice(*device, base, 1u, 0.75f, "the slice beside it");
        DrawDepth(*device, base, 0u, first);
        Fill(base, 0u, 4u);
        for (std::uint32_t slice = 0; slice < 3u; ++slice) CheckSlice(*device, base, slice, 1.0f, "a fill of the whole HTILE");
        std::puts("depth array slice clear tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
