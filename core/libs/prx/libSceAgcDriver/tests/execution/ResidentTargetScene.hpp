#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_RESIDENTTARGETSCENE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_RESIDENTTARGETSCENE_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
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
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

namespace ResidentTargetScene {

using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::StorageTexture;
using ShaderRecompiler::ShaderStage;

inline constexpr std::size_t BlockBytes = 65536;
inline constexpr std::size_t PageBytes = 4096;

// Write-watched guest memory at a 64 KiB boundary. On Linux it is a shared mapping of a sealed
// memory file, as a title's direct memory is: the device imports it as a dma-buf where it can,
// and such an import stays write-watched.
class WatchedBlock {
public:
    explicit WatchedBlock(std::size_t bytes) : bytes(bytes) {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, BlockBytes));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
        file = memfd_create("resident target lookup", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        Require(file >= 0 && ftruncate(file, static_cast<off_t>(bytes)) == 0 && fcntl(file, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) == 0, "resident target lookup: cannot create a memory file");
        void* reserved = mmap(nullptr, bytes + BlockBytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        Require(reserved != MAP_FAILED, "resident target lookup: cannot reserve a guest block");
        const auto begin = reinterpret_cast<std::uintptr_t>(reserved);
        const auto aligned = (begin + BlockBytes - 1) & ~(BlockBytes - 1);
        if (aligned != begin) munmap(reserved, aligned - begin);
        if (aligned + bytes != begin + bytes + BlockBytes) munmap(reinterpret_cast<void*>(aligned + bytes), begin + BlockBytes - aligned);
        Require(mmap(reinterpret_cast<void*>(aligned), bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, file, 0) != MAP_FAILED, "resident target lookup: cannot map a guest block");
        block = reinterpret_cast<std::uint8_t*>(aligned);
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
        Blocks().push_back(this);
        GuestArena::GuestArenaSetSharedBacking_nid_postfix(&Backing);
#endif
        Require(block != nullptr, "resident target lookup: cannot allocate a guest block");
        Require(AgcDriver::GuestMemory::Watched(Address(), bytes), "resident target lookup: the guest block is not write-watched");
        std::memset(block, 0, bytes);
        GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    }

    ~WatchedBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(block, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(block, bytes);
#else
        std::erase(Blocks(), this);
        munmap(block, bytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
        close(file);
#endif
    }

    WatchedBlock(const WatchedBlock&) = delete;
    WatchedBlock& operator=(const WatchedBlock&) = delete;

    std::uint8_t* Data() { return block; }
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(block); }

private:
#ifndef _WIN32
    static std::vector<WatchedBlock*>& Blocks() {
        static std::vector<WatchedBlock*> blocks;
        return blocks;
    }

    static bool Backing(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset) {
        for (const auto* block : Blocks()) {
            if (address < block->Address() || address - block->Address() > block->bytes || bytes > block->bytes - (address - block->Address())) continue;
            *file = dup(block->file);
            *offset = address - block->Address();
            return *file >= 0;
        }
        return false;
    }

    int file = -1;
#endif
    std::size_t bytes;
    std::uint8_t* block = nullptr;
};

alignas(256) inline constexpr std::array<std::uint32_t, 6> VertexCode{0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000};
alignas(256) inline std::array<std::array<float, 4>, 3> Triangle{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};
// v10 = the dword at s8:s9, exported to every channel.
alignas(256) inline constexpr std::array<std::uint32_t, 7> ReadCode{0x7e1e0280, 0xdc308000, 0x0a08000f, 0xbf8c3f70, 0xf800180f, 0x0a0a0a0a, 0xbf810000};

struct Lookup {
    bool current;
    bool walked;
};

struct Scene {
    AgcDriver::VulkanDevice& device;
    // Where the draws read 1.0, with 0.0 in the dword behind it.
    std::uint64_t values;
    AgcDriver::Graphics::ColorTarget color;
    ShaderRecompiler::RecompileResult vertex;
    // The programs live until the device is idle.
    std::list<ShaderRecompiler::RecompileResult> programs;
    std::array<const ShaderRecompiler::RecompileResult*, 2> drawn{};

    Scene(AgcDriver::VulkanDevice& device, std::uint64_t target, std::uint64_t values, VkExtent2D extent) : device(device), values(values) {
        const AgcDriver::Graphics::ColorTargetLayout layout(extent.width, extent.height, AgcDriver::Graphics::ColorTileMode::RenderTarget);
        color = {target, extent, VK_FORMAT_R8G8B8A8_UNORM, layout.Bytes(), 0xe4u};
        color.tileMode = AgcDriver::Graphics::ColorTileMode::RenderTarget;
        const auto triangle = reinterpret_cast<std::uintptr_t>(Triangle.data());
        const std::vector<std::uint32_t> userData{static_cast<std::uint32_t>(triangle), static_cast<std::uint32_t>((triangle >> 32u) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(Triangle.size()), 0x01016facu};
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
        ShaderRecompiler::RecompileRequest request{{ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}}, {64, 0, userData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, memory}, device.Target(), {0, 0, 0, 64}};
        request.useCache = false;
        vertex = ShaderRecompiler::Recompile(request);
    }

    // White (or zero) into the columns of `target` from `left` on; returns whether it was recorded.
    bool Draw(const AgcDriver::Graphics::ColorTarget& target, bool white, std::uint32_t left) {
        const auto address = values + (white ? 0u : 4u);
        std::vector<std::uint32_t> userData(12, 0u);
        userData[8] = static_cast<std::uint32_t>(address);
        userData[9] = static_cast<std::uint32_t>(address >> 32u);
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(ReadCode.data()), std::as_bytes(std::span(ReadCode))}}};
        const auto vertexPush = AgcDriver::Graphics::StagePushOffset(static_cast<std::uint32_t>(vertex.pushConstants.size()), ShaderStage::Fragment, device.GraphicsPipelineLibraries());
        ShaderRecompiler::ShaderPixelStageInfo pixel{};
        pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX);
        pixel.posX = true;
        pixel.targetOutputMode[0] = 9;
        pixel.targetExportMapping.fill(0xe4u);
        ShaderRecompiler::RecompileRequest request{{ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(ReadCode.data()), ReadCode, 0, {}}, {64, 0, userData, std::nullopt, pixel, std::nullopt, memory}, device.Target(), {0, 0, vertexPush, 128 - vertexPush}};
        request.useCache = false;
        auto& kept = drawn[white ? 1 : 0];
        if (kept == nullptr) kept = &programs.emplace_back(ShaderRecompiler::Recompile(request));
        const auto& program = *kept;
        const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{{ShaderStage::Vertex, &vertex, 0}, {ShaderStage::Fragment, &program, vertexPush}}};
        AgcDriver::Graphics::State state{};
        state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
        state.color = target;
        state.colors = {state.color};
        state.hasColorTarget = true;
        state.renderExtent = target.extent;
        state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        state.viewport = {0, static_cast<float>(target.extent.height), static_cast<float>(target.extent.width), -static_cast<float>(target.extent.height), 0, 1};
        state.negativeOneToOne = false;
        state.scissor = {{static_cast<std::int32_t>(left), 0}, {target.extent.width - left, target.extent.height}};
        state.cullMode = VK_CULL_MODE_NONE;
        state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        state.blend.colorWriteMask = 15;
        state.blends = {state.blend};
        state.blendConstants = {};
        const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Triangle.size()), 0, 1, 0, false};
        const auto before = AgcDriver::Graphics::DrawPassCounters();
        device.Draw(state, draw, shaders);
        const auto after = AgcDriver::Graphics::DrawPassCounters();
        return after.begun != before.begun || after.continued != before.continued;
    }

    // The columns of `target` up to `split` all hold `left` and the others `right`, in guest memory.
    bool Holds(const AgcDriver::Graphics::ColorTarget& target, std::uint32_t split, std::uint32_t left, std::uint32_t right) {
        StorageTexture::FlushPending(target.address, target.bytes, nullptr, "test");
        device.WaitIdle();
        drawn = {};
        programs.clear();
        std::vector<std::uint32_t> pixels(target.extent.width * target.extent.height);
        AgcDriver::Graphics::ReadColorTarget(target, std::as_writable_bytes(std::span(pixels)));
        for (std::uint32_t y = 0; y < target.extent.height; ++y) {
            for (std::uint32_t x = 0; x < target.extent.width; ++x) {
                if (pixels[y * target.extent.width + x] != (x < split ? left : right)) return false;
            }
        }
        return true;
    }

    // What every draw into the target starts with. A walk of the target's pages moves the tracker's
    // generation, and nothing else of a refresh that finds the image current does.
    Lookup Refresh(StorageTexture& image) {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const auto before = AgcDriver::GuestMemory::TrackerGeneration();
        const bool current = image.Refresh();
        return {current, AgcDriver::GuestMemory::TrackerGeneration() != before};
    }

    // The image current with what the checks before left in memory, and its walk remembered.
    void Settle(StorageTexture& image) {
        Refresh(image);
        Require(Refresh(image).current, "resident target lookup: the target was uploaded twice for one change");
    }
};

}

#endif
