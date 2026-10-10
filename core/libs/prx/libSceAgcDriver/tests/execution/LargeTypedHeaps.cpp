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

constexpr std::uint32_t Images = 20;
constexpr std::uint32_t Side = 4;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Format32323232Float = 77;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t SamplerByte = 0x300;
constexpr std::uint32_t OutputByte = 0x310;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 32;
constexpr std::size_t StorageBytes = 2048;
constexpr std::size_t BlockBytes = 65536;
constexpr std::array<std::uint32_t, 4> PointSampler{0u, 0x00fff000u, 0u, 0u};

struct alignas(4096) GuestData {
    std::array<std::uint32_t, 256> srt{};
    std::array<std::uint32_t, Images * 4> output{};
};

GuestData Guest;
alignas(256) std::array<std::array<std::uint8_t, 4096>, Images> Textures{};
alignas(256) std::array<std::byte, Width * Height * 4> Pixels{};

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

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
        Require(block != nullptr, "large typed heaps: cannot allocate the guest block");
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

std::array<std::uint8_t, 4> Color(std::uint32_t image) {
    std::array<std::uint8_t, 4> color{};
    color[image / 7u] = static_cast<std::uint8_t>(1u << (image % 7u));
    return color;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, stride == 0u ? 0x31016facu : 0x01016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(const void* texels, std::uint32_t format) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texels));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        0xfacu | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

void FillSampled() {
    const auto mips = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format8888UNorm, Side, Side, 1u);
    Require(AgcDriver::Graphics::ComputeSurfaceSize(mips, 1) <= Textures[0].size(), "large typed heaps: a sampled texture does not fit its storage");
    Guest.srt.fill(0u);
    for (std::uint32_t image = 0; image < Images; ++image) {
        const auto color = Color(image);
        for (std::size_t byte = 0; byte < Textures[image].size(); ++byte) Textures[image][byte] = color[byte % 4u];
        const auto descriptor = TextureDescriptor(Textures[image].data(), Format8888UNorm);
        std::copy(descriptor.begin(), descriptor.end(), Guest.srt.begin() + image * 8u);
    }
    std::copy(PointSampler.begin(), PointSampler.end(), Guest.srt.begin() + SamplerByte / 4u);
    const auto output = BufferDescriptor(Guest.output.data(), 0u, static_cast<std::uint32_t>(Guest.output.size() * 4u));
    std::copy(output.begin(), output.end(), Guest.srt.begin() + OutputByte / 4u);
}

std::vector<std::uint32_t> SampleCode(bool pixel) {
    std::vector<std::uint32_t> code{0x7e0002f0, 0x7e0202f0, 0xf4080500, 0xfa000000 | SamplerByte};
    if (pixel) {
        code.insert(code.end(), {0x7e100280, 0x7e120280, 0x7e140280, 0x7e1602f2});
    } else {
        code.insert(code.end(), {0xf4080600, 0xfa000000 | OutputByte});
    }
    for (std::uint32_t image = 0; image < Images; ++image) {
        code.insert(code.end(), {0xf40c0200, 0xfa000000 | (image * 32u), 0xbf8cc07f, 0xf09c0f08, 0x00a20400, 0xbf8c3f70});
        if (pixel) {
            code.insert(code.end(), {0x06101104, 0x06121305, 0x06141506});
        } else {
            code.insert(code.end(), {0xe0780000 | (image * 16u), 0x80060400});
        }
    }
    if (pixel) {
        code.insert(code.end(), {0xf800180f, 0x0b0a0908});
    }
    code.push_back(0xbf810000);
    return code;
}

std::vector<std::uint32_t> StoreCode() {
    std::vector<std::uint32_t> code{0x7e0a02f0, 0x7e0c02f0, 0x7e0e02f0, 0x7e3c0280, 0x7e3e0280};
    for (std::uint32_t image = 0; image < Images; ++image) {
        code.insert(code.end(), {0xf40c0200, 0xfa000000 | (image * 32u), 0xbf8cc07f, 0x7e0802ff, std::bit_cast<std::uint32_t>(static_cast<float>(image + 1u)), 0xf0201f08, 0x0002041e});
    }
    code.push_back(0xbf810000);
    return code;
}

std::array<std::uint32_t, 2> SrtUserData() {
    const auto srt = reinterpret_cast<std::uintptr_t>(Guest.srt.data());
    return {static_cast<std::uint32_t>(srt), static_cast<std::uint32_t>(srt >> 32u)};
}

void Dispatch(AgcDriver::VulkanDevice& device, const std::vector<std::uint32_t>& code) {
    const auto userData = SrtUserData();
    const std::array regions{ShaderRecompiler::MemoryRegion{reinterpret_cast<std::uintptr_t>(&Guest), std::as_bytes(std::span(&Guest, 1u))}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{1u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0u, {}},
        {32u, 0u, userData, compute, std::nullopt, std::nullopt, regions},
        device.Target(),
        {0u, 0u, 0u, 128u}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1u, 1u, 1u);
    device.WaitIdle();
}

void CheckSampledDispatch(AgcDriver::VulkanDevice& device) {
    FillSampled();
    Guest.output.fill(0xdeadbeefu);
    Dispatch(device, SampleCode(false));
    for (std::uint32_t image = 0; image < Images; ++image) {
        const auto color = Color(image);
        for (std::uint32_t component = 0; component < 4u; ++component) {
            const auto actual = std::bit_cast<float>(Guest.output[image * 4u + component]);
            const auto expected = static_cast<float>(color[component]) / 255.0f;
            Require(actual == expected, "compute shader sampling " + std::to_string(Images) + " textures: texture " + std::to_string(image) + " component " + std::to_string(component) + " is " + std::to_string(actual) + ", expected " + std::to_string(expected));
        }
    }
}

void CheckSampledDraw(AgcDriver::VulkanDevice& device) {
    FillSampled();
    Pixels.fill(std::byte{0x40});
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
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());

    const auto code = SampleCode(true);
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.wave32 = false;
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    const auto srt = SrtUserData();
    std::vector<std::uint32_t> pixelUserData(8, 0u);
    std::copy(srt.begin(), srt.end(), pixelUserData.begin());
    const std::array regions{ShaderRecompiler::MemoryRegion{reinterpret_cast<std::uintptr_t>(&Guest), std::as_bytes(std::span(&Guest, 1u))}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64u, 0, pixelUserData, std::nullopt, pixel, std::nullopt, regions},
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

    std::array<std::uint32_t, 4> expected{0u, 0u, 0u, 255u};
    for (std::uint32_t image = 0; image < Images; ++image) {
        const auto color = Color(image);
        for (std::uint32_t component = 0; component < 3u; ++component) expected[component] += color[component];
    }
    for (std::uint32_t pixelIndex = 0; pixelIndex < Width * Height; ++pixelIndex) {
        for (std::uint32_t channel = 0; channel < 4u; ++channel) {
            const auto actual = static_cast<std::uint32_t>(Pixels[pixelIndex * 4u + channel]);
            Require(actual == expected[channel], "pixel shader summing " + std::to_string(Images) + " textures: pixel " + std::to_string(pixelIndex) + " channel " + std::to_string(channel) + " is " + std::to_string(actual) + ", expected " + std::to_string(expected[channel]));
        }
    }
}

void CheckStorageDispatch(AgcDriver::VulkanDevice& device) {
    GuestBlock block;
    const auto mips = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format32323232Float, Side, Side, 1u);
    Require(AgcDriver::Graphics::ComputeSurfaceSize(mips, 1) <= StorageBytes, "large typed heaps: a storage texture does not fit its storage");
    Guest.srt.fill(0u);
    for (std::uint32_t image = 0; image < Images; ++image) {
        const auto descriptor = TextureDescriptor(block.Data() + image * StorageBytes, Format32323232Float);
        std::copy(descriptor.begin(), descriptor.end(), Guest.srt.begin() + image * 8u);
    }
    Dispatch(device, StoreCode());
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(block.Data()), BlockBytes, nullptr, "test");
    device.WaitIdle();
    for (std::uint32_t image = 0; image < Images; ++image) {
        const std::array<float, 4> expected{static_cast<float>(image + 1u), 0.5f, 0.5f, 0.5f};
        std::array<float, 4> actual{};
        std::memcpy(actual.data(), block.Data() + image * StorageBytes + mips[0].tiledOffset, sizeof(actual));
        for (std::uint32_t component = 0; component < 4u; ++component) {
            Require(actual[component] == expected[component], "compute shader storing to " + std::to_string(Images) + " textures: texture " + std::to_string(image) + " component " + std::to_string(component) + " is " + std::to_string(actual[component]) + ", expected " + std::to_string(expected[component]));
        }
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        CheckSampledDispatch(*device);
        CheckSampledDraw(*device);
        CheckStorageDispatch(*device);
        std::puts("large typed heaps tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
