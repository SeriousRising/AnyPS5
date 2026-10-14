#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Cover_vert_spv.h"
#include "Cover_frag_spv.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <span>
#include <vector>

namespace {
using namespace AgcDriver::Graphics;
constexpr std::uint32_t Width = 128;
constexpr std::uint32_t Height = 128;
constexpr std::size_t BlockBytes = 65536;
alignas(4096) std::array<std::byte, 2 * BlockBytes> SurfaceStorage{};
const std::span<std::byte, BlockBytes> Surface{reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(SurfaceStorage.data()) + BlockBytes - 1u) & ~std::uintptr_t{BlockBytes - 1u}), BlockBytes};
alignas(4096) std::array<std::byte, 4096> Keys{};

void DrawBlended(AgcDriver::VulkanDevice& device, const ColorTarget& color, VkRect2D scissor) {
    ShaderRecompiler::RecompileResult vertex;
    vertex.spirv = std::vector<std::uint32_t>(std::begin(COVER_VERT_SPV), std::end(COVER_VERT_SPV));
    ShaderRecompiler::RecompileResult fragment;
    fragment.spirv = std::vector<std::uint32_t>(std::begin(COVER_FRAG_SPV), std::end(COVER_FRAG_SPV));
    const std::array<CompiledShader, 2> shaders{{
        {ShaderRecompiler::ShaderStage::Vertex, &vertex, 0},
        {ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}
    }};
    State state{};
    state.stages = {ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    state.color = color;
    state.colors = {color};
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, 0, static_cast<float>(Width), static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = scissor;
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blend.blendEnable = VK_TRUE;
    state.blend.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
    state.blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    state.blend.colorBlendOp = VK_BLEND_OP_ADD;
    state.blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
    state.blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    state.blend.alphaBlendOp = VK_BLEND_OP_ADD;
    state.blends = {state.blend};
    state.blendConstants = {0, 0, 0, 0.5f};
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    Require(Recorder::Active() != nullptr, "recorded-clear regression requires an active recorder");
    device.Draw(state, {0, 3, 0, 1, 0, false}, shaders);
}


void Run(AgcDriver::VulkanDevice& device) {
    const ColorTargetLayout layout(Width, Height, ColorTileMode::RenderTarget, 4);
    Require(layout.Bytes() == Surface.size(), "clear attachment fixture layout changed");
    ColorTarget color{};
    color.address = reinterpret_cast<std::uintptr_t>(Surface.data());
    color.extent = {Width, Height};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.bytes = layout.Bytes();
    color.componentMapping = 0xe4;
    color.tileMode = ColorTileMode::RenderTarget;
    color.elementBytes = 4;
    color.surfaceAddress = color.address;
    color.surfaceExtent = color.extent;
    color.dccAddress = reinterpret_cast<std::uintptr_t>(Keys.data());
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    for (unsigned frame = 0; frame < 12; ++frame) {
        std::fill(Surface.begin(), Surface.end(), std::byte{0x55});
        std::fill(Keys.begin(), Keys.end(), std::byte{0});
        AgcDriver::GuestMemory::BumpCollectEpoch();
        DrawBlended(device, color, {{0, 0}, {Width / 2, Height}});
        StorageTexture::FlushPending(color.address, color.bytes, nullptr, "recorded clear attachment regression");
        device.WaitIdle();
        for (std::uint32_t y = 0; y < Height; ++y) {
            for (std::uint32_t x = 0; x < Width; ++x) {
                const auto offset = layout.Offset(x, y);
                for (unsigned channel = 0; channel < 4; ++channel) {
                    const auto actual = std::to_integer<unsigned>(Surface[offset + channel]);
                    const unsigned expected = x < Width / 2 ? 128 : 0;
                    Require(actual == expected || (expected == 128 && actual == 127),
                            "recorded DCC clear was not visible to attachment LOAD/blend");
                }
            }
        }
    }
}
}

int main() {
    try {
        for (const auto* variable : {"APS5_NO_RECORDED_CLEAR", "APS5_SYNC_DRAWS",
                "APS5_NO_RESIDENT_TARGETS", "APS5_DUMP_TARGETS", "APS5_SYNC_COMPLETION_DRAWS",
                "APS5_NO_RECORDER_SYNC_DRAWS", "APS5_NO_RECORD_COPIED_DRAWS",
                "APS5_DRAW_TRANSITIONS", "APS5_EAGER_WRITEBACK", "APS5_NO_TEXTURE_CACHE"}) {
            Require(std::getenv(variable) == nullptr,
                    std::string("recorded-clear regression rejects route-changing environment: ") + variable);
        }
        std::unique_ptr<AgcDriver::VulkanDevice> device;
        try { device = std::make_unique<AgcDriver::VulkanDevice>(); }
        catch (const std::exception& error) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::cout << "skipped, no Vulkan device: " << error.what() << '\n';
            return 77;
        }
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(SurfaceStorage.data(), SurfaceStorage.size(), true, true, true);
            mutation.Add(Keys.data(), Keys.size(), true, true, true);
        }
        Run(*device);
        device->WaitIdle();
        device.reset();
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(SurfaceStorage.data());
            mutation.Remove(Keys.data());
        }
        std::cout << "recorded clear attachment LOAD/blend passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
