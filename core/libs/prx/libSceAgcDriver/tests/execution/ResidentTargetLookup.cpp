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

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::StorageTexture;
using ShaderRecompiler::ShaderStage;

constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t PageBytes = 4096;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 64;
constexpr std::uint32_t White = 0xffffffffu;

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

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000};
alignas(256) std::array<std::array<float, 4>, 3> Triangle{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};
// v10 = the dword at s8:s9, exported to every channel.
alignas(256) constexpr std::array<std::uint32_t, 7> ReadCode{0x7e1e0280, 0xdc308000, 0x0a08000f, 0xbf8c3f70, 0xf800180f, 0x0a0a0a0a, 0xbf810000};

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

    Scene(AgcDriver::VulkanDevice& device, std::uint64_t target, std::uint64_t values) : device(device), values(values) {
        const AgcDriver::Graphics::ColorTargetLayout layout(Width, Height, AgcDriver::Graphics::ColorTileMode::RenderTarget);
        color = {target, {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, layout.Bytes(), 0xe4u};
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
        const auto& program = programs.emplace_back(ShaderRecompiler::Recompile(request));
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

int main() {
    try {
        using namespace AgcDriver::GuestMemory;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        std::lock_guard gpu(GpuMutex());
        if (!WriteWatched()) {
            std::puts("no write watching: skipped");
            return VulkanTestSkipped;
        }
        WatchedBlock values(BlockBytes);
        const float one = 1.0f;
        std::memcpy(values.Data(), &one, sizeof(one));
        WatchedBlock target(BlockBytes);
        // 64 blocks of other ranges: a collect of a page is widened to its block, so that many
        // overrun the per-thread memo.
        WatchedBlock other(64 * BlockBytes);
        // The blocks are released with nothing recorded over them, also when a check fails.
        struct Idle {
            AgcDriver::VulkanDevice& device;
            ~Idle() { device.WaitIdle(); }
        } idle{*device};
        Scene scene(*device, target.Address(), values.Address());
        auto& color = scene.color;
        Require(color.bytes == BlockBytes, "resident target lookup: the target is not one block");
        const auto walkOthers = [&] {
            for (std::size_t block = 0; block < 64; ++block) CollectWrites(other.Address() + block * BlockBytes, PageBytes);
        };

        BumpCollectEpoch();
        if (!scene.Draw(color, true, 0)) {
            std::puts("draws are not recorded on this device: lookups of a resident target not tested");
            return 0;
        }
        // An import that does not keep write protection is compared instead: nothing is walked.
        const bool watched = Watched(color.address, color.bytes);
        if (!watched) std::puts("the imported target is not write-watched on this device: its page walks not tested");
        const auto image = StorageTexture::FindPending(color.address, color.bytes);
        Require(image != nullptr, "resident target lookup: the drawn target has no resident image");
        scene.Settle(*image);
        auto lookup = scene.Refresh(*image);
        Require(lookup.current && !lookup.walked, "resident target lookup: a lookup repeated in one submission walked the target's pages");
        // A frame's draws collect far more ranges between two lookups of a target than the
        // per-thread memo holds.
        walkOthers();
        lookup = scene.Refresh(*image);
        Require(lookup.current && !lookup.walked, "resident target lookup: walks of 64 other ranges made the next lookup walk the target's pages again");
        walkOthers();
        const auto beforeDraw = TrackerGeneration();
        Require(scene.Draw(color, false, Width / 2) && TrackerGeneration() == beforeDraw, "resident target lookup: a draw into a target looked up before in its submission walked pages");
        Require(scene.Holds(color, Width / 2, White, 0u), "resident target lookup: draws after remembered walks did not keep what the draws before them drew");

        // A CPU store is collected at the next ordering point, and wins over nothing pending.
        scene.Settle(*image);
        std::memset(target.Data(), 0x5a, BlockBytes);
        BumpCollectEpoch();
        lookup = scene.Refresh(*image);
        Require(!lookup.current && lookup.walked == watched, "resident target lookup: a CPU store into the target was not collected at the next ordering point");
        Require(scene.Draw(color, true, Width / 2) && scene.Holds(color, Width / 2, 0x5a5a5a5au, White), "resident target lookup: a draw after a CPU store did not keep the stored texels");

        // The driver's own store stamps its blocks: it is seen inside the epoch, without a walk.
        scene.Settle(*image);
        walkOthers();
        StoreOwnBytes(color.address, BlockBytes, [&] { std::memset(target.Data(), 0x22, BlockBytes); });
        Require(!scene.Refresh(*image).current, "resident target lookup: a driver store into the target was missed inside its epoch");
        Require(scene.Draw(color, false, Width / 2) && scene.Holds(color, Width / 2, 0x22222222u, 0u), "resident target lookup: a draw after a driver store did not keep the stored texels");

        // A store the GPU makes into the target's memory, where the device records one.
        const std::array<std::uint32_t, 4> pattern{0x44444444u, 0x44444444u, 0x44444444u, 0x44444444u};
        scene.Settle(*image);
        walkOthers();
        if (device->FillBuffer(color.address, BlockBytes, pattern)) Require(scene.Draw(color, true, Width / 2) && scene.Holds(color, Width / 2, 0x44444444u, White), "resident target lookup: a draw after a GPU store did not keep the stored texels");
        else std::puts("fills are not recorded on this device: a GPU store into a resident target not tested");

        // The same memory as a target of another size is another image, which reads what the
        // first one drew.
        Require(scene.Draw(color, true, 0), "resident target lookup: the draw before the resize was not recorded");
        auto resized = color;
        resized.extent = {Width, Height / 2};
        resized.bytes = AgcDriver::Graphics::ColorTargetLayout(Width, Height / 2, AgcDriver::Graphics::ColorTileMode::RenderTarget).Bytes();
        Require(scene.Draw(resized, false, Width / 2), "resident target lookup: the draw into the resized target was not recorded");
        const auto second = StorageTexture::FindPending(resized.address, resized.bytes);
        Require(second != nullptr && second != image && second->Descriptor().height == Height / 2, "resident target lookup: the resized target has no image of its own");
        Require(scene.Holds(resized, Width / 2, White, 0u), "resident target lookup: the resized target did not keep what the first one drew");
        std::puts("resident target lookup tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
