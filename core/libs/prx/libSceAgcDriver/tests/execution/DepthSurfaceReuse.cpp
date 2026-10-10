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
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 32;
constexpr std::uint32_t ReusedWidth = 32;
constexpr std::uint32_t ReusedHeight = 8;
constexpr std::uint32_t Format32Float = 22;
constexpr std::uint32_t Format32323232Float = 77;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t BlockBytes = 65536;

alignas(256) std::array<std::uint32_t, Threads * 4> Buffer{};
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

constexpr std::array<std::array<float, 4>, 3> Triangle{{
    {-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}
}};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "depth surface reuse: cannot allocate the guest block");
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

std::uint32_t Stored(std::uint32_t tid, std::uint32_t component) {
    return std::bit_cast<std::uint32_t>(static_cast<float>(tid * 4u + component) + 0.25f);
}

void DrawDepth(AgcDriver::VulkanDevice& device, std::uint8_t* depth, bool write = true, VkCompareOp compare = VK_COMPARE_OP_ALWAYS) {
    Pixels.fill(Kept);
    const auto target = device.Target();
    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(Triangle.data(), 16u, static_cast<std::uint32_t>(Triangle.size()));
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
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Triangle.size()), 0, 1, 0, false};
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

void CheckDepthLoad(AgcDriver::VulkanDevice& device, std::uint8_t* depth, const std::string& what) {
    Buffer.fill(0xdeadbeefu);
    Dispatch(device, LoadCode, TextureDescriptor(depth, Format32Float, Width, Height));
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto actual = std::bit_cast<float>(Buffer[tid * 4u]);
        Require(actual == 0.5f, what + ": thread " + std::to_string(tid) + " loaded depth " + std::to_string(actual) + ", expected 0.5");
    }
}

std::uint32_t DepthStored(std::uint32_t tid, std::uint32_t component) {
    return std::bit_cast<std::uint32_t>(component == 0u ? 0.25f + static_cast<float>(tid) / 128.0f : 0.0f);
}

void CheckLiveStore(AgcDriver::VulkanDevice& device, std::uint8_t* depth) {
    const auto plane = TextureDescriptor(depth, Format32Float, Width, Height);
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t component = 0; component < 4u; ++component) Buffer[tid * 4u + component] = DepthStored(tid, component);
    }
    Dispatch(device, StoreCode, plane);
    Buffer.fill(0xdeadbeefu);
    Dispatch(device, LoadCode, plane);
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        Require(Buffer[tid * 4u] == DepthStored(tid, 0u), "depth loaded after an image_store to the live depth surface: thread " + std::to_string(tid) + " is " + std::to_string(std::bit_cast<float>(Buffer[tid * 4u])) + ", expected " + std::to_string(std::bit_cast<float>(DepthStored(tid, 0u))));
    }
    Buffer.fill(0xdeadbeefu);
    Dispatch(device, NextRowLoadCode, plane);
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        Require(std::bit_cast<float>(Buffer[tid * 4u]) == 0.5f, "depth the image_store did not write: thread " + std::to_string(tid) + " is " + std::to_string(std::bit_cast<float>(Buffer[tid * 4u])) + ", expected 0.5");
    }
    DrawDepth(device, depth, false, VK_COMPARE_OP_LESS_OR_EQUAL);
    for (std::uint32_t pixel = 0; pixel < Width * Height; ++pixel) {
        const bool stored = pixel < Threads;
        const auto expected = stored ? Kept : std::byte{255};
        Require(Pixels[pixel * 4u] == expected, std::string("a draw at depth 0.5 tested against the stored depth: pixel ") + std::to_string(pixel) + " is " + std::to_string(static_cast<unsigned>(Pixels[pixel * 4u])) + ", expected " + std::to_string(static_cast<unsigned>(expected)) + (stored ? " (the stored depth is nearer)" : ""));
    }
}

void CheckReuse(AgcDriver::VulkanDevice& device, std::uint8_t* memory) {
    const auto texture = TextureDescriptor(memory, Format32323232Float, ReusedWidth, ReusedHeight);
    const auto mip = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format32323232Float, ReusedWidth, ReusedHeight, 1u).at(0);
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t component = 0; component < 4u; ++component) Buffer[tid * 4u + component] = Stored(tid, component);
    }
    Dispatch(device, StoreCode, texture);
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(memory), BlockBytes, nullptr, "test");
    device.WaitIdle();
    for (std::uint32_t x = 0; x < Threads; ++x) {
        for (std::uint32_t component = 0; component < 4u; ++component) {
            std::uint32_t actual = 0;
            std::memcpy(&actual, memory + mip.tiledOffset + x * 16u + component * 4u, 4u);
            Require(actual == Stored(x, component), "image_store to memory that was a depth surface: texel " + std::to_string(x) + " component " + std::to_string(component) + " is " + std::to_string(std::bit_cast<float>(actual)) + ", expected " + std::to_string(std::bit_cast<float>(Stored(x, component))));
        }
    }
    Buffer.fill(0xdeadbeefu);
    Dispatch(device, LoadCode, texture);
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t component = 0; component < 4u; ++component) {
            Require(Buffer[tid * 4u + component] == Stored(tid, component), "image_load from memory that was a depth surface: thread " + std::to_string(tid) + " component " + std::to_string(component) + " is " + std::to_string(std::bit_cast<float>(Buffer[tid * 4u + component])) + ", expected " + std::to_string(std::bit_cast<float>(Stored(tid, component))));
        }
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock block;
        DrawDepth(*device, block.Data());
        CheckDepthLoad(*device, block.Data(), "live depth surface");
        CheckLiveStore(*device, block.Data());
        CheckReuse(*device, block.Data());
        DrawDepth(*device, block.Data());
        CheckDepthLoad(*device, block.Data(), "depth surface bound again after its memory was reused");
        std::puts("depth surface reuse tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
