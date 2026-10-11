#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
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
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t ResultOffset = 0x40;
constexpr std::uint32_t Sentinel = 0x05e471e1;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 64;

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "translated bases: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true);
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

    // The guest frees the block and maps it again: the driver must give it a GPU owner anew.
    void Replace() {
        GuestAllocations::Mutation().Remove(block);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true);
    }

    std::uint8_t* Data() { return block; }
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(block); }

private:
    std::uint8_t* block = nullptr;
};

alignas(256) std::array<std::byte, Width * Height * 4> Pixels{};
alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000};
alignas(256) std::array<std::array<float, 4>, 3> Triangle{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};

struct Read {
    std::uint32_t base, offset, dwords;
};

// Two bases in turn, with reads of one to four dwords on each: v1:v2 holds the block's address
// (s8:s9) and v3:v4 the base address under test (s10:s11), so both bases come from user data and the
// CPU translates them for each draw or dispatch (APS5_NO_BDA_CPU_BASE=1: the program does, lazily).
constexpr std::array<Read, 8> Reads{{{1, 0x100, 4}, {3, 0, 1}, {1, 0x120, 2}, {3, 12, 1}, {1, 0x110, 1}, {3, 4, 2}, {1, 0x114, 3}, {3, 16, 4}}};
constexpr std::int64_t TestedFirst = 0;

// v9 is the sum of what was read, stored at the block's ResultOffset. `pcs` takes each read's
// instruction offset. A pixel program then exports a color; its moves from scalar registers depend
// on the lane, so it reads at the scalar pair itself plus v15, which is 0, where the compute
// program reads at the pair of vector registers: both forms of base address. With `srt` the base
// under test is not in user data: s10:s11 points at a pair of dwords that holds it, which the
// program loads into s20:s21 and the resource walk reads for each draw.
std::vector<std::uint32_t> Code(bool pixel, bool srt, std::vector<std::uint32_t>& pcs) {
    std::vector<std::uint32_t> code{0x7e120280u, 0x7e020208u, 0x7e040209u, 0x7e06020au, 0x7e08020bu, 0x7e1e0280u};
    if (srt) code.insert(code.end(), {0xf4040000u | (20u << 6u) | 5u, 0xfa000000u, 0xbf8cc07fu});
    constexpr std::array<std::uint32_t, 4> opcodes{0xdc308000u, 0xdc348000u, 0xdc3c8000u, 0xdc388000u};
    const auto address = [&](std::uint32_t base) {
        if (base == 3u && srt) return 0x0a00000fu | (20u << 16u);
        return pixel ? 0x0a00000fu | ((base == 1u ? 8u : 10u) << 16u) : 0x0a7d0000u | base;
    };
    for (const auto& read : Reads) {
        pcs.push_back(static_cast<std::uint32_t>(code.size() * 4u));
        code.insert(code.end(), {opcodes[read.dwords - 1u] | read.offset, address(read.base), 0xbf8c3f70u});
        for (std::uint32_t dword = 0; dword < read.dwords; ++dword) code.push_back(0x4a120109u | ((10u + dword) << 9u));
    }
    code.insert(code.end(), {0xdc708000u | ResultOffset, 0x007d0901u});
    if (pixel) code.insert(code.end(), {0x7e0e02f2u, 0xf800180fu, 0x07070707u});
    code.push_back(0xbf810000u);
    return code;
}

std::uint32_t Pattern(std::size_t word) {
    return static_cast<std::uint32_t>((word + 3u) * 0x9e3779b1u);
}

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
    Require(file != nullptr, "translated bases: cannot create a file for the error stream");
    std::fflush(stderr);
    const int saved = duplicate(number(stderr));
    Require(saved >= 0 && redirect(number(file), number(stderr)) >= 0, "translated bases: cannot redirect the error stream");
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
    const auto text = restored();
    return text;
}

struct Fault {
    unsigned long long address;
    unsigned instruction, bytes, stage, reason;
};

std::optional<Fault> RecordedFault(const std::string& text) {
    const auto at = text.find("BDA access failed: ");
    if (at == std::string::npos) return std::nullopt;
    Fault fault{};
    Require(std::sscanf(text.c_str() + at, "BDA access failed: address=0x%llx instruction=0x%x bytes=%u stage=%u reason=%u", &fault.address, &fault.instruction, &fault.bytes, &fault.stage, &fault.reason) == 5, "translated bases: the fault report has another form");
    return fault;
}

struct Case {
    std::string name;
    bool pixel = false;
    std::uint64_t tested = 0;
    // The request goes through the recompiler's cache, so that a second use takes the same module.
    bool cached = false;
    // Changes the compiled result before it is used; `reads` is then where the program must read from.
    std::function<void(ShaderRecompiler::RecompileResult&)> change;
    std::optional<std::uint64_t> reads;
    // The base under test comes from a pair of dwords in guest memory instead of user data.
    bool srt = false;
};

alignas(16) std::array<std::uint32_t, 2> Table{};

bool CpuBases() {
    return std::getenv("APS5_NO_BDA_CPU_BASE") == nullptr;
}

std::uint64_t Run(AgcDriver::VulkanDevice& device, GuestBlock& guest, const Case& test, std::uint32_t seed = 0) {
    static std::vector<std::vector<std::uint32_t>> codes;
    static std::vector<std::vector<std::uint32_t>> offsets;
    if (codes.empty()) {
        for (const bool srt : {false, true}) {
            for (const bool pixel : {false, true}) {
                offsets.emplace_back();
                codes.push_back(Code(pixel, srt, offsets.back()));
            }
        }
    }
    const auto which = (test.srt ? 2u : 0u) + (test.pixel ? 1u : 0u);
    const auto& code = codes[which];
    const auto& pcs = offsets[which];
    std::vector<std::uint32_t> memory(BlockBytes / 4u);
    for (std::size_t word = 0; word < memory.size(); ++word) memory[word] = Pattern(word) + seed;
    memory[ResultOffset / 4u] = Sentinel;
    std::memcpy(guest.Data(), memory.data(), BlockBytes);
    const auto block = guest.Address();
    const auto from = test.reads.value_or(test.tested);
    std::optional<Fault> expected;
    std::uint32_t sum = 0;
    for (std::size_t read = 0; read < Reads.size() && !expected.has_value(); ++read) {
        for (std::uint32_t dword = 0; dword < Reads[read].dwords && !expected.has_value(); ++dword) {
            const auto address = (Reads[read].base == 1u ? block : from) + Reads[read].offset + dword * 4u;
            if (address < block || address + 4u > block + BlockBytes) expected = Fault{address, pcs[read], 1u, test.pixel ? 2u : 4u, 1u};
            else sum += memory[(address - block) / 4u];
        }
    }
    std::vector<std::uint32_t> userData(12, 0u);
    userData[8] = static_cast<std::uint32_t>(block);
    userData[9] = static_cast<std::uint32_t>(block >> 32u);
    Table = {static_cast<std::uint32_t>(test.tested), static_cast<std::uint32_t>(test.tested >> 32u)};
    const auto pointed = test.srt ? static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Table.data())) : test.tested;
    userData[10] = static_cast<std::uint32_t>(pointed);
    userData[11] = static_cast<std::uint32_t>(pointed >> 32u);
    std::vector<ShaderRecompiler::MemoryRegion> regions{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(std::span(code))}};
    if (test.srt) regions.push_back({reinterpret_cast<std::uintptr_t>(Table.data()), std::as_bytes(std::span(Table))});
    std::sort(regions.begin(), regions.end(), [](const auto& left, const auto& right) { return left.guestAddress < right.guestAddress; });
    std::uint64_t module = 0;
    const auto before = AgcDriver::Graphics::BdaResources::BaseSlotCounters();
    const auto prepare = [&](ShaderRecompiler::RecompileRequest& request) {
        request.useCache = test.cached;
        auto result = ShaderRecompiler::Recompile(request);
        Require(result.bdaBases.size() == (CpuBases() ? 2u : 0u), "translated bases: " + test.name + " has " + std::to_string(result.bdaBases.size()) + " bases for the CPU");
        if (CpuBases()) Require(result.bdaBases[1].base == test.tested && result.bdaBases[1].hull == test.tested + TestedFirst && result.bdaBases[1].bytes == 32u && result.bdaBases[0].base == block && result.bdaBases[0].hull == block + 0x100u && result.bdaBases[0].bytes == 0x28u, "translated bases: " + test.name + " evaluated another base or hull");
        if (test.change) test.change(result);
        module = result.PipelineVariantId();
        return result;
    };
    const auto text = Captured([&] {
        if (!test.pixel) {
            const ShaderRecompiler::ShaderComputeStageInfo compute{{1, 1, 1}, 0, {false, false, false}, false, 1};
            ShaderRecompiler::RecompileRequest request{{ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}}, {32, 0, userData, compute, std::nullopt, std::nullopt, regions}, device.Target(), {0, 0, 0, 128}};
            const auto result = prepare(request);
            device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
            device.WaitIdle();
            return;
        }
        std::vector<std::uint32_t> vertexUserData(4, 0u);
        const auto triangle = reinterpret_cast<std::uintptr_t>(Triangle.data());
        vertexUserData = {static_cast<std::uint32_t>(triangle), static_cast<std::uint32_t>((triangle >> 32u) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(Triangle.size()), 0x01016facu};
        const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
        ShaderRecompiler::RecompileRequest vertex{{ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}}, {64, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory}, device.Target(), {0, 0, 0, 64}};
        vertex.useCache = false;
        const auto vertexResult = ShaderRecompiler::Recompile(vertex);
        const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());
        ShaderRecompiler::ShaderPixelStageInfo pixel{};
        pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX);
        pixel.posX = true;
        pixel.targetOutputMode[0] = 9;
        pixel.targetExportMapping.fill(0xe4u);
        ShaderRecompiler::RecompileRequest fragment{{ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}}, {64, 0, userData, std::nullopt, pixel, std::nullopt, regions}, device.Target(), {0, 0, vertexPush, 128 - vertexPush}};
        const auto pixelResult = prepare(fragment);
        const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{{ShaderStage::Vertex, &vertexResult, 0}, {ShaderStage::Fragment, &pixelResult, vertexPush}}};
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
        // A draw's fault is thrown where a dispatch's is printed by the recorder.
        try {
            device.Draw(state, draw, shaders);
            device.WaitIdle();
        } catch (const std::runtime_error& error) {
            std::fprintf(stderr, "%s\n", error.what());
        }
    });
    const auto recorded = RecordedFault(text);
    const auto name = "translated bases: " + test.name + (test.pixel ? " (draw" : " (dispatch") + (test.srt ? ", base read from guest memory)" : ")");
    if (expected.has_value()) {
        Require(recorded.has_value(), name + " recorded no fault");
        Require(recorded->address == expected->address && recorded->instruction == expected->instruction && recorded->bytes == expected->bytes && recorded->stage == expected->stage && recorded->reason == expected->reason, name + " recorded the fault of address " + std::to_string(recorded->address) + " at instruction " + std::to_string(recorded->instruction) + ", expected address " + std::to_string(expected->address) + " at instruction " + std::to_string(expected->instruction));
    } else {
        Require(!recorded.has_value(), name + " recorded a fault");
        memory[ResultOffset / 4u] = sum;
    }
    for (std::size_t word = 0; word < memory.size(); ++word) {
        std::uint32_t value = 0;
        std::memcpy(&value, guest.Data() + word * 4u, 4u);
        Require(value == memory[word], name + " left " + std::to_string(value) + " at byte " + std::to_string(word * 4u) + ", expected " + std::to_string(memory[word]));
    }
    // The driver filled a slot for each base of the program, and gave no device address to a hull
    // that is not in one range.
    const auto after = AgcDriver::Graphics::BdaResources::BaseSlotCounters();
    if (CpuBases() && !test.change) {
        const bool whole = test.tested >= block && test.tested + 32u <= block + BlockBytes;
        Require(after.translated - before.translated == (whole ? 2u : 1u) && after.unusable - before.unusable == (whole ? 0u : 1u), name + " translated " + std::to_string(after.translated - before.translated) + " bases and left " + std::to_string(after.unusable - before.unusable));
    } else if (!CpuBases()) {
        Require(after.translated == before.translated && after.unusable == before.unusable, name + " filled slots without bases to translate");
    }
    return module;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock guest;
        const auto block = guest.Address();
        for (const bool pixel : {false, true}) {
            Run(*device, guest, {"both hulls in one range", pixel, block + 0x400u});
            // The reads of the second base before the block's end succeed; the one past it faults with its own record.
            Run(*device, guest, {"a hull across the range's end", pixel, block + BlockBytes - 24u});
            Run(*device, guest, {"a hull in the gap behind the range", pixel, block + BlockBytes + 0x40u});
            // The same module, and so the same pipeline, with another base address in each use: the
            // second must not read where the first one's slot pointed.
            const auto first = Run(*device, guest, {"a first base address", pixel, block + 0x800u, true});
            const auto second = Run(*device, guest, {"another base address in the same pipeline", pixel, block + 0xc00u, true});
            Require(first == second, "translated bases: the two uses did not share one pipeline variant");
            Run(*device, guest, {"the first base address again", pixel, block + 0x800u, true});
            // The same guest address with a new GPU owner and other bytes.
            Run(*device, guest, {"a block before it is replaced", pixel, block + 0x400u, true});
            guest.Replace();
            Run(*device, guest, {"a block that was replaced", pixel, block + 0x400u, true}, 0x01010101u);
            // The base address is read from guest memory by the walk: in one range, across the
            // range's end and in the gap, and with other dwords there for a second use of the
            // same pipeline variant, whose user data does not change.
            const auto read = [&](const std::string& name, std::uint64_t tested, bool cached) {
                Case test{name, pixel, tested, cached};
                test.srt = true;
                return Run(*device, guest, test);
            };
            read("both hulls in one range", block + 0x400u, false);
            read("a hull across the range's end", block + BlockBytes - 24u, false);
            read("a hull in the gap behind the range", block + BlockBytes + 0x40u, false);
            const auto before = read("a first base address", block + 0x800u, true);
            Require(before == read("another base address in the same pipeline", block + 0xc00u, true), "translated bases: the two uses with a base from guest memory did not share one pipeline variant");
            if (!CpuBases()) continue;
            const bool used = std::getenv("APS5_NO_BDA_SHARED_BASE") == nullptr && std::getenv("APS5_BDA_BYTE_READS") == nullptr;
            // The CPU evaluated another base address than the program computes: the program must not
            // take the slot, whose hull is mapped and holds other bytes.
            Run(*device, guest, {"a slot of another base address", pixel, block + 0x400u, false, [&](ShaderRecompiler::RecompileResult& result) {
                result.bdaBases[1].base = block + 0x2000u;
                result.bdaBases[1].hull = block + 0x2000u;
            }});
            // And that the slot is what the program reads through when the base address is its own:
            // a slot whose hull is elsewhere (which the driver never makes) moves the reads there.
            if (used) Run(*device, guest, {"a slot whose hull is elsewhere", pixel, block + 0x400u, false, [&](ShaderRecompiler::RecompileResult& result) { result.bdaBases[1].hull = block + 0x3000u; }, block + 0x3000u});
        }
        std::puts("global loads through translated bases tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
