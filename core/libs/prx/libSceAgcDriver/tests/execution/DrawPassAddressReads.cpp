#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 64;
constexpr std::uint32_t One = 0x3f800000u;
// Where the block holds 1.0 and 0.0 for the reads, and the words the storing draws write 1.0 to.
constexpr std::uint32_t OneOffset = 0x100;
constexpr std::uint32_t ZeroOffset = 0x104;
constexpr std::uint32_t AddressStoreOffset = 0x200;
constexpr std::uint32_t BufferStoreOffset = 0x8000;

// Zeros at a 64 KiB boundary, of which the first `registered` bytes are guest memory.
class GuestBlock {
public:
    GuestBlock(std::size_t bytes, std::size_t registered) {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(65536, bytes));
#endif
        Require(block != nullptr, "draw pass: cannot allocate a guest block");
        std::memset(block, 0, bytes);
        GuestAllocations::Mutation().Add(block, registered, true, true);
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
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(block); }
    std::uint32_t Word(std::uint32_t offset) const {
        std::uint32_t value = 0;
        std::memcpy(&value, block + offset, 4u);
        return value;
    }
    void SetWord(std::uint32_t offset, std::uint32_t value) { std::memcpy(block + offset, &value, 4u); }

private:
    std::uint8_t* block = nullptr;
};

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000};
alignas(256) std::array<std::array<float, 4>, 3> Triangle{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};
// v10 = the dword at s8:s9, exported to every channel.
alignas(256) constexpr std::array<std::uint32_t, 7> ReadCode{0x7e1e0280, 0xdc308000, 0x0a08000f, 0xbf8c3f70, 0xf800180f, 0x0a0a0a0a, 0xbf810000};
// 1.0 stored to the dword at s8:s9 (through v1:v2) and exported.
alignas(256) constexpr std::array<std::uint32_t, 8> AddressStoreCode{0x7e020208, 0x7e040209, 0x7e1202f2, 0xdc708000, 0x007d0901, 0xf800180f, 0x09090909, 0xbf810000};
// 1.0 stored to the first dword of the buffer whose descriptor is in s4..s7, and exported.
alignas(256) constexpr std::array<std::uint32_t, 7> BufferStoreCode{0x7e040280, 0x7e0202f2, 0xe0701000, 0x80010102, 0xf800180f, 0x01010101, 0xbf810000};

// What `run` wrote to the standard error stream, which gets it afterwards as well.
std::string Captured(const std::function<void()>& run) {
#ifdef _WIN32
    const auto duplicate = _dup;
    const auto redirect = _dup2;
    const auto number = _fileno;
    const auto release = _close;
#else
    const auto duplicate = dup;
    const auto redirect = dup2;
    const auto number = fileno;
    const auto release = close;
#endif
    std::FILE* file = std::tmpfile();
    Require(file != nullptr, "draw pass: cannot create a file for the error stream");
    std::fflush(stderr);
    const int saved = duplicate(number(stderr));
    Require(saved >= 0 && redirect(number(file), number(stderr)) >= 0, "draw pass: cannot redirect the error stream");
    const auto restored = [&] {
        std::fflush(stderr);
        redirect(saved, number(stderr));
        release(saved);
        std::string text;
        std::rewind(file);
        std::array<char, 4096> part{};
        for (std::size_t count = 0; (count = std::fread(part.data(), 1, part.size(), file)) != 0;) text.append(part.data(), count);
        std::fclose(file);
        std::fputs(text.c_str(), stderr);
        return text;
    };
    try {
        run();
    } catch (...) {
        restored();
        throw;
    }
    return restored();
}

std::vector<std::uint64_t> FaultAddresses(const std::string& text) {
    constexpr std::string_view marker = "BDA access failed: address=0x";
    std::vector<std::uint64_t> addresses;
    for (auto at = text.find(marker); at != std::string::npos; at = text.find(marker, at + 1)) addresses.push_back(std::strtoull(text.c_str() + at + marker.size(), nullptr, 16));
    return addresses;
}

struct Step {
    std::span<const std::uint32_t> code;
    // What s8:s9 holds, or for BufferStoreCode the buffer's address.
    std::uint64_t address;
    // The draw covers the columns from `left` on.
    std::uint32_t left;
};

enum class Pass { Begun, Continued, Synchronous };

struct Scene {
    AgcDriver::VulkanDevice& device;
    GuestBlock& target;
    AgcDriver::Graphics::ColorTarget color;
    ShaderRecompiler::RecompileResult vertex;
    // The programs live until the device is idle.
    std::list<ShaderRecompiler::RecompileResult> programs;

    Scene(AgcDriver::VulkanDevice& device, GuestBlock& target, std::size_t targetBytes) : device(device), target(target) {
        color = {target.Address(), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, targetBytes, 0xe4u};
        color.tileMode = AgcDriver::Graphics::ColorTileMode::RenderTarget;
        const auto triangle = reinterpret_cast<std::uintptr_t>(Triangle.data());
        const std::vector<std::uint32_t> userData{static_cast<std::uint32_t>(triangle), static_cast<std::uint32_t>((triangle >> 32u) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(Triangle.size()), 0x01016facu};
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
        ShaderRecompiler::RecompileRequest request{{ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}}, {64, 0, userData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, memory}, device.Target(), {0, 0, 0, 64}};
        request.useCache = false;
        vertex = ShaderRecompiler::Recompile(request);
    }

    Pass Draw(const Step& step) {
        std::vector<std::uint32_t> userData(12, 0u);
        if (step.code.data() == BufferStoreCode.data()) {
            userData[4] = static_cast<std::uint32_t>(step.address);
            userData[5] = static_cast<std::uint32_t>((step.address >> 32u) & 0xffffu);
            userData[6] = 64u;
            userData[7] = 0x31016facu;
        } else {
            userData[8] = static_cast<std::uint32_t>(step.address);
            userData[9] = static_cast<std::uint32_t>(step.address >> 32u);
        }
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(step.code.data()), std::as_bytes(step.code)}}};
        const auto vertexPush = static_cast<std::uint32_t>(vertex.pushConstants.size());
        ShaderRecompiler::ShaderPixelStageInfo pixel{};
        pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX);
        pixel.posX = true;
        pixel.targetOutputMode[0] = 9;
        pixel.targetExportMapping.fill(0xe4u);
        ShaderRecompiler::RecompileRequest request{{ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(step.code.data()), step.code, 0, {}}, {64, 0, userData, std::nullopt, pixel, std::nullopt, memory}, device.Target(), {0, 0, vertexPush, 128 - vertexPush}};
        request.useCache = false;
        const auto& program = programs.emplace_back(ShaderRecompiler::Recompile(request));
        Require(program.bdaWrites == (step.code.data() == AddressStoreCode.data()), "draw pass: the compiled program misstates whether it stores through an address");
        const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{{ShaderStage::Vertex, &vertex, 0}, {ShaderStage::Fragment, &program, vertexPush}}};
        AgcDriver::Graphics::State state{};
        state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
        state.color = color;
        state.colors = {state.color};
        state.hasColorTarget = true;
        state.renderExtent = {Width, Height};
        state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
        state.negativeOneToOne = false;
        state.scissor = {{static_cast<std::int32_t>(step.left), 0}, {Width - step.left, Height}};
        state.cullMode = VK_CULL_MODE_NONE;
        state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        state.blend.colorWriteMask = 15;
        state.blends = {state.blend};
        state.blendConstants = {};
        const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Triangle.size()), 0, 1, 0, false};
        const auto before = AgcDriver::Graphics::DrawPassCounters();
        device.Draw(state, draw, shaders);
        const auto after = AgcDriver::Graphics::DrawPassCounters();
        return after.begun != before.begun ? Pass::Begun : after.continued != before.continued ? Pass::Continued : Pass::Synchronous;
    }

    // The draws in turn, then the device idle: how each draw took its pass, and the faults reported.
    std::vector<Pass> Run(std::span<const Step> steps, std::vector<std::uint64_t>& faults) {
        std::vector<Pass> passes;
        faults = FaultAddresses(Captured([&] {
            for (const auto& step : steps) passes.push_back(Draw(step));
            device.WaitIdle();
        }));
        programs.clear();
        return passes;
    }

    // The target's columns from `left` up to `right` all hold `texel`.
    bool Holds(std::uint32_t left, std::uint32_t right, std::uint32_t texel) {
        AgcDriver::Graphics::StorageTexture::FlushPending(color.address, color.bytes, nullptr, "test");
        device.WaitIdle();
        std::vector<std::uint32_t> pixels(Width * Height);
        AgcDriver::Graphics::ReadColorTarget(color, std::as_writable_bytes(std::span(pixels)));
        for (std::uint32_t y = 0; y < Height; ++y) {
            for (std::uint32_t x = left; x < right; ++x) {
                if (pixels[y * Width + x] != texel) return false;
            }
        }
        return true;
    }
};

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock guest(BlockBytes * 2u, BlockBytes);
        const AgcDriver::Graphics::ColorTargetLayout layout(Width, Height, AgcDriver::Graphics::ColorTileMode::RenderTarget);
        const auto targetBytes = (layout.Bytes() + 65535u) & ~std::size_t{65535u};
        GuestBlock target(targetBytes, targetBytes);
        Scene scene(*device, target, layout.Bytes());
        const auto block = guest.Address();
        guest.SetWord(OneOffset, One);
        const std::span<const std::uint32_t> read(ReadCode);
        const auto one = block + OneOffset;
        const auto zero = block + ZeroOffset;
        // Behind the block's registered half.
        const auto unmapped = block + BlockBytes + 0x40u;
        std::vector<std::uint64_t> faults;
        using Passes = std::vector<Pass>;

        auto passes = scene.Run(std::array<Step, 3>{{{read, one, 0}, {read, zero, Width / 2}, {read, one, Width * 3 / 4}}}, faults);
        if (passes[0] == Pass::Synchronous) {
            std::puts("draws are not recorded on this device: render passes of draws that read by address not tested");
            return 0;
        }
        Require(passes == Passes{Pass::Begun, Pass::Continued, Pass::Continued}, "draw pass: consecutive draws that read by address did not share one render pass");
        Require(faults.empty(), "draw pass: draws that read mapped memory reported a fault");
        Require(scene.Holds(0, Width / 2, 0xffffffffu) && scene.Holds(Width / 2, Width * 3 / 4, 0u) && scene.Holds(Width * 3 / 4, Width, 0xffffffffu), "draw pass: the draws of one render pass did not each draw what they read");

        // The third draw reads what the second stored: it needs the barrier of a pass of its own.
        passes = scene.Run(std::array<Step, 4>{{{read, zero, 0}, {AddressStoreCode, block + AddressStoreOffset, Width / 2}, {read, block + AddressStoreOffset, Width * 3 / 4}, {read, zero, Width * 7 / 8}}}, faults);
        Require(passes[0] == Pass::Begun && passes[2] == Pass::Begun && passes[3] == Pass::Continued, "draw pass: a draw whose program stores through an address did not end its render pass");
        Require(faults.empty() && guest.Word(AddressStoreOffset) == One, "draw pass: the store through an address did not land");
        Require(scene.Holds(0, Width / 2, 0u) && scene.Holds(Width / 2, Width * 7 / 8, 0xffffffffu) && scene.Holds(Width * 7 / 8, Width, 0u), "draw pass: the draw after a store through an address did not read it");

        passes = scene.Run(std::array<Step, 4>{{{read, zero, 0}, {BufferStoreCode, block + BufferStoreOffset, Width / 2}, {read, block + BufferStoreOffset, Width * 3 / 4}, {read, zero, Width * 7 / 8}}}, faults);
        Require(passes[0] == Pass::Begun && passes[2] == Pass::Begun && passes[3] == Pass::Continued, "draw pass: a draw with a written storage buffer did not end its render pass");
        Require(faults.empty() && guest.Word(BufferStoreOffset) == One, "draw pass: the store to a storage buffer did not land");
        Require(scene.Holds(0, Width / 2, 0u) && scene.Holds(Width / 2, Width * 7 / 8, 0xffffffffu) && scene.Holds(Width * 7 / 8, Width, 0u), "draw pass: the draw after a store to a storage buffer did not read it");

        // Each draw has its own fault record: a fault is reported for the draw that made it, and
        // the other draw of the pass draws what it read.
        passes = scene.Run(std::array<Step, 2>{{{read, unmapped, 0}, {read, one, Width / 2}}}, faults);
        Require(passes == Passes{Pass::Begun, Pass::Continued}, "draw pass: a faulting draw and the one after it did not share one render pass");
        Require(faults == std::vector<std::uint64_t>{unmapped}, "draw pass: the fault of the first draw of a pass was not reported once, for its address");
        Require(scene.Holds(Width / 2, Width, 0xffffffffu), "draw pass: the draw after a faulting one did not draw what it read");
        passes = scene.Run(std::array<Step, 2>{{{read, unmapped + 16u, 0}, {read, unmapped + 24u, Width / 2}}}, faults);
        Require(passes == Passes{Pass::Begun, Pass::Continued}, "draw pass: two faulting draws did not share one render pass");
        Require(faults == std::vector<std::uint64_t>{unmapped + 16u, unmapped + 24u}, "draw pass: the faults of both draws of a pass were not reported, each for its address");
        passes = scene.Run(std::array<Step, 2>{{{read, zero, 0}, {read, unmapped + 8u, Width / 2}}}, faults);
        Require(passes == Passes{Pass::Begun, Pass::Continued}, "draw pass: a draw and the faulting one after it did not share one render pass");
        Require(faults == std::vector<std::uint64_t>{unmapped + 8u}, "draw pass: the fault of the second draw of a pass was not reported once, for its address");
        Require(scene.Holds(0, Width / 2, 0u), "draw pass: the draw before a faulting one did not draw what it read");
        std::puts("draw pass address reads tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
