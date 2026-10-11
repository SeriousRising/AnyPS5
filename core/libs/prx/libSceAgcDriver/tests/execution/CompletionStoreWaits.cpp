#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;
using CopyOutcome = AgcDriver::VulkanDevice::CopyOutcome;

constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Results = 4;
constexpr std::uint32_t Counters = 4;
constexpr std::uint32_t CounterOffset = 0x10;
constexpr std::uint32_t LabelOffset = 0x100;
constexpr std::uint32_t ReturnOffset = 0x124;
constexpr std::array<std::uint32_t, Counters> Starts{1000, 2000, 3000, 4000};
constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::uint32_t GdsLabel = 0x11223344u;
constexpr std::uint32_t DestinationLabel = 0x55667788u;
constexpr std::uint32_t OutputLabel = 0x99aabbccu;

alignas(256) constexpr auto Code = std::to_array<std::uint32_t>({
    0x34060082,
    0x7e0c02ff, 0x5eed0001u, 0x7e0e02ff, 0x5eed0002u,
    0xbefc03ff, 0x0000ffff,
    0xd8fa0010, 0x04000000,
    0xd8f60014, 0x05000000,
    0x7da800ff, Threads,
    0xd8fa0018, 0x06000000,
    0xd8f6001c, 0x07000000,
    0xbefe04c1,
    0xbf8cc07f,
    0xe0702000, 0x80000403, 0xe0702004, 0x80000503, 0xe0702008, 0x80000603, 0xe070200c, 0x80000703,
    0xbf810000,
});

class GuestBlock {
public:
    explicit GuestBlock(bool registered) {
#ifdef _WIN32
        block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "completion store waits: cannot allocate a guest block");
        std::memset(block, 0, BlockBytes);
        if (registered) Register();
    }

    ~GuestBlock() {
        if (listed) GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    void Register() {
        if (listed) return;
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
        listed = true;
    }

    std::uint32_t* Data() { return block; }

    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)); }

private:
    std::uint32_t* block = nullptr;
    bool listed = false;
};

struct Bytes {
    std::array<std::uint32_t, Threads * Results> output{};
    std::array<std::uint32_t, Counters> counters{};
    std::uint32_t gdsLabel = 0;
    std::uint32_t gdsReturn = 0;
    std::array<std::uint32_t, 8> destination{};

    bool operator==(const Bytes&) const = default;
};

struct Waits {
    bool completionPending = false;
    int gdsLabel = -1;
    int outputLabel = -1;
    CopyOutcome fromGds{};
    CopyOutcome toGds{};
    CopyOutcome overLabel{};
    CopyOutcome fromOutput{};
};

int StoreLabel(AgcDriver::VulkanDevice& device, std::uint64_t address, std::uint32_t value) {
    static std::uint64_t stamp = 0;
    const auto bytes = std::as_bytes(std::span(&value, 1));
    const auto reason = device.WriteLabelOnGpu(address, bytes, ++stamp, 0, false);
    if (reason == 1) AgcDriver::GuestMemory::Write(address, bytes, 4);
    else Require(reason == 0 || reason == 5, "completion store waits: a label store was refused with reason " + std::to_string(reason));
    return reason;
}

CopyOutcome Copy(AgcDriver::VulkanDevice& device, std::uint64_t destination, std::uint64_t source, std::size_t bytes) {
    const auto outcome = device.CopyBuffer(destination, source, bytes, bytes, std::numeric_limits<std::size_t>::max(), 0, 0, 0, [](std::span<const std::byte>, std::uint64_t) {});
    Require(outcome.path == 0 || outcome.path == 1, "completion store waits: a copy was refused with path " + std::to_string(outcome.path));
    return outcome;
}

Bytes Run(AgcDriver::VulkanDevice& device, GuestBlock& output, GuestBlock& destination, bool pipelined, Waits& waits) {
    const auto gds = AgcDriver::Pm4::GdsAddress();
    std::fill_n(output.Data(), BlockBytes / 4, Untouched);
    std::fill_n(destination.Data(), BlockBytes / 4, Untouched);
    AgcDriver::GuestMemory::Write(gds + CounterOffset, std::as_bytes(std::span(Starts)), 4);
    const std::uint32_t cleared = 0;
    AgcDriver::GuestMemory::Write(gds + LabelOffset, std::as_bytes(std::span(&cleared, 1)), 4);
    AgcDriver::GuestMemory::Write(gds + ReturnOffset, std::as_bytes(std::span(&cleared, 1)), 4);
    const auto outputAddress = output.Address();
    const std::vector<std::uint32_t> userData{static_cast<std::uint32_t>(outputAddress), static_cast<std::uint32_t>((outputAddress >> 32u) & 0xffffu) | (4u << 16u), Threads * Results, 0x11016facu};
    const std::span<const std::uint32_t> code(Code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto shader = ShaderRecompiler::Recompile(request);
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    const auto settle = [&] {
        if (!pipelined) device.WaitIdle();
    };
    device.Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    settle();
    if (pipelined) {
        Require(device.AfterRecordedWork([] {}, false), "completion store waits: the dispatch left no recorded work to follow");
        output.Register();
    }
    waits.completionPending = AgcDriver::Graphics::Recorder::PendingCompletionLabels() != 0;
    waits.gdsLabel = StoreLabel(device, gds + LabelOffset, GdsLabel);
    settle();
    waits.fromGds = Copy(device, destination.Address(), gds + LabelOffset, 4);
    settle();
    waits.toGds = Copy(device, gds + ReturnOffset, destination.Address(), 4);
    settle();
    StoreLabel(device, destination.Address() + 8, DestinationLabel);
    settle();
    waits.overLabel = Copy(device, destination.Address() + 8, gds + CounterOffset, 4);
    settle();
    waits.outputLabel = StoreLabel(device, outputAddress, OutputLabel);
    settle();
    waits.fromOutput = Copy(device, destination.Address() + 16, outputAddress, 16);
    device.WaitIdle();
    Bytes bytes;
    std::memcpy(bytes.output.data(), output.Data(), sizeof(bytes.output));
    std::memcpy(bytes.counters.data(), reinterpret_cast<const void*>(gds + CounterOffset), sizeof(bytes.counters));
    std::memcpy(&bytes.gdsLabel, reinterpret_cast<const void*>(gds + LabelOffset), sizeof(bytes.gdsLabel));
    std::memcpy(&bytes.gdsReturn, reinterpret_cast<const void*>(gds + ReturnOffset), sizeof(bytes.gdsReturn));
    std::memcpy(bytes.destination.data(), destination.Data(), sizeof(bytes.destination));
    return bytes;
}

}

int main() {
    try {
        GuestBlock output(false);
        GuestBlock destination(true);
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (device->Target().subgroupSize < 32) {
            std::printf("skipped, the device's subgroups are narrower than a wave32 (%u lanes)\n", device->Target().subgroupSize);
            return VulkanTestSkipped;
        }
        auto* gds = reinterpret_cast<void*>(static_cast<std::uintptr_t>(AgcDriver::Pm4::GdsAddress()));
        if (!GuestAllocations::Mutation().Covers(gds, AgcDriver::Pm4::GdsBytes)) GuestAllocations::Mutation().Add(gds, AgcDriver::Pm4::GdsBytes, true, true, true);
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            const std::array<std::uint32_t, 4> pattern{1, 2, 3, 4};
            const bool imported = device->FillBuffer(destination.Address(), 4096, pattern);
            device->WaitIdle();
            if (!imported) {
                std::printf("skipped, the device does not import guest memory\n");
                return VulkanTestSkipped;
            }
        }
        Waits serial, waits;
        const auto expected = Run(*device, output, destination, false, serial);
        Require(expected.output[1] != Untouched && expected.output[0] == OutputLabel && expected.gdsLabel == GdsLabel && expected.gdsReturn == GdsLabel && expected.counters != Starts, "completion store waits: the dispatch and the stores waited for one by one left other bytes than they store");
        Require(expected.destination[0] == GdsLabel && expected.destination[2] == expected.counters[0] && expected.destination[4] == OutputLabel && expected.destination[5] == expected.output[1] && expected.destination[7] == expected.output[3], "completion store waits: the copies waited for one by one left other bytes than their sources");
        const auto actual = Run(*device, output, destination, true, waits);
        Require(actual == expected, "completion store waits: stores and copies recorded behind a dispatch left other bytes than the same ones waited for one by one");
        Require(waits.completionPending, "completion store waits: no completion was pending behind the dispatch");
        Require(waits.gdsLabel == 0, "completion store waits: a store to memory the dispatch writes on the GPU went behind its completion (reason " + std::to_string(waits.gdsLabel) + ")");
        Require(waits.fromGds.path == 1 && !waits.fromGds.synced, "completion store waits: a copy out of memory the dispatch writes on the GPU waited (cause " + std::to_string(waits.fromGds.syncCause) + ")");
        Require(waits.toGds.path == 1 && !waits.toGds.synced, "completion store waits: a copy into the GDS out of memory a recorded copy writes waited (cause " + std::to_string(waits.toGds.syncCause) + ")");
        Require(waits.overLabel.path == 1 && !waits.overLabel.synced, "completion store waits: a copy over a label stored on the GPU waited (cause " + std::to_string(waits.overLabel.syncCause) + ")");
        Require(waits.outputLabel == 5, "completion store waits: a store over a copied buffer's pending write-back did not go behind it (reason " + std::to_string(waits.outputLabel) + ")");
        Require(waits.fromOutput.path == 1 && waits.fromOutput.synced && (waits.fromOutput.syncCause & 1) != 0, "completion store waits: a copy out of a copied buffer did not wait for its write-back (cause " + std::to_string(waits.fromOutput.syncCause) + ")");
        std::puts("completion store waits tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
