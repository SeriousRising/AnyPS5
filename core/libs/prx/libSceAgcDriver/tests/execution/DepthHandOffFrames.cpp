#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
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
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t SurfaceBytes = 65536;
constexpr std::size_t BlockBytes = 2 * SurfaceBytes;

struct Words {
    std::uint32_t* words = nullptr;
    std::uint32_t* data() const { return words; }
    std::size_t size() const { return Threads * 4u; }
    void fill(std::uint32_t value) const { std::fill_n(words, size(), value); }
    std::uint32_t& operator[](std::size_t index) const { return words[index]; }
};

Words Buffer;
alignas(256) std::array<std::byte, Width * Height * 4> Pixels{};

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 4> PixelCode{
    0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 9> StoreCode{
    0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0280, 0xf0201f08, 0x0001141e,
    0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 9> LoadCode{
    0x34020084, 0x7e3c0300, 0x7e3e0280, 0xf0001f08, 0x0001141e, 0xbf8c3f70, 0xe0781000, 0x80001401,
    0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 9> NextRowLoadCode{
    0x34020084, 0x7e3c0300, 0x7e3e0281, 0xf0001f08, 0x0001141e, 0xbf8c3f70, 0xe0781000, 0x80001401,
    0xbf810000,
};

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
        Require(block != nullptr, "depth hand-off frames: cannot allocate the guest block");
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

std::array<std::uint32_t, 8> TextureDescriptor(const void* texels, std::uint32_t format, std::uint32_t width, std::uint32_t height) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texels));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((width - 1u) & 3u) << 30u),
        ((width - 1u) >> 2u) | ((height - 1u) << 14u),
        0xfacu | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

void DrawDepth(AgcDriver::VulkanDevice& device, std::uint8_t* depth, const Vertices& triangle, bool write = true, VkCompareOp compare = VK_COMPARE_OP_ALWAYS) {
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
    state.depth = AgcDriver::Graphics::DepthTarget{reinterpret_cast<std::uintptr_t>(depth), 0u, {Width, Height}, VK_FORMAT_D32_SFLOAT, 1.0f, 0u};
    state.depthTest = true;
    state.depthWrite = write;
    state.depthCompare = compare;
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

void Dispatch(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::uint64_t variant, const std::array<std::uint32_t, 8>& texture) {
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
    auto result = ShaderRecompiler::Recompile(request);
    result.specializationId = variant;
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

std::uint32_t Stored(std::uint32_t frame, std::uint32_t tid, std::uint32_t component) {
    return std::bit_cast<std::uint32_t>(component == 0u ? static_cast<float>(1u + tid + 64u * frame) / 256.0f : 0.0f);
}

void CheckRow(AgcDriver::VulkanDevice& device, std::uint8_t* depth, bool next, std::uint32_t frame, float drawn, const std::string& what) {
    Buffer.fill(0xdeadbeefu);
    if (next) Dispatch(device, NextRowLoadCode, 0x5103u, TextureDescriptor(depth, Format32Float, Width, Height));
    else Dispatch(device, LoadCode, 0x5102u, TextureDescriptor(depth, Format32Float, Width, Height));
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto expected = next ? drawn : std::bit_cast<float>(Stored(frame, tid, 0u));
        const auto actual = std::bit_cast<float>(Buffer[tid * 4u]);
        Require(actual == expected, "frame " + std::to_string(frame) + ", " + what + ": thread " + std::to_string(tid) + " loaded depth " + std::to_string(actual) + ", expected " + std::to_string(expected));
    }
}

void Frame(AgcDriver::VulkanDevice& device, std::uint8_t* depth, std::uint32_t frame, float drawn) {
    DrawDepth(device, depth, At(drawn));
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t component = 0; component < 4u; ++component) Buffer[tid * 4u + component] = Stored(frame, tid, component);
    }
    Dispatch(device, StoreCode, 0x5101u, TextureDescriptor(depth, Format32Float, Width, Height));
    CheckRow(device, depth, false, frame, drawn, "the row a repeated dispatch stored through a storage image of the depth plane, sampled by a repeated dispatch");
    CheckRow(device, depth, true, frame, drawn, "the row under it, which only the draw wrote");
    DrawDepth(device, depth, At(0.875f), false, VK_COMPARE_OP_LESS_OR_EQUAL);
    for (std::uint32_t pixel = 0; pixel < Width * Height; ++pixel) {
        Require(Pixels[pixel * 4u] == Kept, "frame " + std::to_string(frame) + ", a draw at depth 0.875 tested against the plane after the store: pixel " + std::to_string(pixel) + " is " + std::to_string(static_cast<unsigned>(Pixels[pixel * 4u])) + ", expected " + std::to_string(static_cast<unsigned>(Kept)) + " (every depth in the plane is nearer)");
    }
    CheckRow(device, depth, false, frame, drawn, "the stored row after the plane was bound as a depth target again");
    CheckRow(device, depth, true, frame, drawn, "the drawn row after the plane was bound as a depth target again");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock block;
        Buffer.words = reinterpret_cast<std::uint32_t*>(block.Data() + SurfaceBytes);
        Frame(*device, block.Data(), 0u, 0.75f);
        Frame(*device, block.Data(), 1u, 0.5f);
        Frame(*device, block.Data(), 2u, 0.625f);
        std::puts("depth hand-off frames tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
