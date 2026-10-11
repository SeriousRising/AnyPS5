#include "BdaShader.hpp"
#include "BdaAbi.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaDirectTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

class Pipeline {
public:
    Pipeline(const Context& context, std::span<const std::uint32_t> code, const std::array<Buffer*, 3>& buffers) : context(context) {
        try {
            std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
            for (std::uint32_t i = 0; i < bindings.size(); ++i) bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            setInfo.bindingCount = bindings.size();
            setInfo.pBindings = bindings.data();
            Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &setInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
            const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &size;
            Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
            VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1, &setLayout};
            Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &set), "vkAllocateDescriptorSets");
            for (std::uint32_t i = 0; i < buffers.size(); ++i) {
                const VkDescriptorBufferInfo buffer{buffers[i]->Handle(), 0, buffers[i]->Bytes().size()};
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = set;
                write.dstBinding = i;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &buffer;
                context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
            }
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.setLayoutCount = 1;
            layoutInfo.pSetLayouts = &setLayout;
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout");
            VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            moduleInfo.codeSize = code.size_bytes();
            moduleInfo.pCode = code.data();
            Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            pipelineInfo.layout = layout;
            Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");
        } catch (...) { release(); throw; }
    }

    ~Pipeline() { release(); }

    void Run(std::uint32_t groups) {
        CommandBatch batch(context);
        const auto commands = batch.Handle();
        VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &upload, 0, nullptr, 0, nullptr);
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groups, 1, 1);
        VkMemoryBarrier download{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &download, 0, nullptr, 0, nullptr);
        batch.SubmitAndWait();
    }

private:
    void release() noexcept {
        if (pipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        if (layout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
        if (pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        if (setLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
    }

    const Context& context;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

}

// Each sequence runs with the range cache keeping one, two and four ranges, each also with the cache
// check, and without the cache (APS5_BDA_RANGE_CACHE_ENTRIES, APS5_BDA_RANGE_CACHE_CHECK=1 and
// APS5_NO_BDA_RANGE_CACHE=1 in the recompiler), with and without stopping invocations: the output words, the whole fault buffer
// (fault record and noted pages) and the data must be equal, and the cached runs must give the
// stated values and fault.
// Each sequence also runs with its reads sharing base addresses (EmitBdaSharedDwordReads; off with
// APS5_NO_BDA_SHARED_BASE=1 in the recompiler) by blocks of 16 bytes (a range of `sorted`), of 64 (two
// ranges and the gap behind them) and of 4096 (every read of the sequence): searched, with two kept
// ranges and the check, with one and two kept ranges, where a kept range answers for a hull without
// a search, and with the invocation ended at the faulting read. All must equal the run
// in which every read looks its own bytes up.
static void RunBdaRangeCacheTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    constexpr std::uint64_t guest = 0x7fff12340000ULL;
    constexpr std::uint32_t Sentinel = 0xdeadbeef;
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    Buffer writable(context, 32, usage);
    Buffer readOnly(context, 16, usage);
    Require(writable.DeviceAddress() % 4u == 0u && readOnly.DeviceAddress() % 4u == 0u, "BDA range cache: the test buffers are not dword-aligned");
    // [guest, +16) writable, [+16, +32) read-only and adjacent, a gap, [+64, +80) the writable buffer's second half.
    const std::vector<Abi::Range> sorted{
        {guest, guest + 16, writable.DeviceAddress(), Abi::Read | Abi::Write, 0},
        {guest + 16, guest + 32, readOnly.DeviceAddress(), Abi::Read, 0},
        {guest + 64, guest + 80, writable.DeviceAddress() + 16, Abi::Read | Abi::Write, 0}};
    const auto fill = [&] {
        for (std::size_t byte = 0; byte < writable.Bytes().size(); ++byte) writable.Bytes()[byte] = static_cast<std::byte>(byte + 1u);
        for (std::size_t byte = 0; byte < readOnly.Bytes().size(); ++byte) readOnly.Bytes()[byte] = static_cast<std::byte>(byte + 0x81u);
    };
    const auto low = [](std::uint32_t byte) { return (byte + 1u) | (byte + 2u) << 8u | (byte + 3u) << 16u | (byte + 4u) << 24u; };
    struct Expected {
        std::vector<std::uint32_t> values;
        Abi::FaultReason reason = static_cast<Abi::FaultReason>(0);
        std::uint64_t address = 0;
        std::uint32_t bytes = 0;
        std::uint32_t step = 0;
        // Whether the fault ends a stopping invocation: a faulting read does, a store or atomic is skipped.
        bool ends = true;
    };
    struct Outcome {
        std::vector<std::uint32_t> values;
        std::vector<std::byte> fault;
        std::vector<std::byte> data;
        bool operator==(const Outcome&) const = default;
    };
    // `direct`: the sequence also runs on the direct table of the ranges (sorted, valid ranges only),
    // with the invocation flagged and ended at the faulting read and with leaves walked ahead by
    // blocks of 16, 64 and 4096 bytes. All must equal the searched run, except that a dword of which
    // only the first bytes are mapped reads 0 where the search read those bytes: `directValues`,
    // when not empty, are what a run that continues past its fault then produces.
    const auto check = [&](const std::string& name, const std::vector<Abi::Range>& ranges, std::uint32_t count, const std::vector<BdaTestStep>& steps, const Expected* expected, bool direct = true, const std::vector<std::uint32_t>& directValues = {}) {
        Buffer table(context, sizeof(Abi::Header) + ranges.size() * sizeof(Abi::Range), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const Abi::Header header{Abi::Version, count, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), ranges.size() * sizeof(Abi::Range));
        Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer output(context, steps.size() * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        for (const bool stops : {true, false}) {
            const auto run = [&](std::uint32_t rangeCacheEntries, bool rangeCacheCheck = false, bool deferred = true, std::uint64_t sharedBytes = 0) {
                fill();
                std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
                for (std::size_t word = 0; word < steps.size(); ++word) std::memcpy(output.Bytes().data() + word * 4u, &Sentinel, 4u);
                Pipeline pipeline(context, MakeBdaSequenceTestShader(steps, rangeCacheEntries, stops, rangeCacheCheck, deferred, sharedBytes), {&table, &fault, &output});
                pipeline.Run(1);
                Outcome outcome{std::vector<std::uint32_t>(steps.size()), {fault.Bytes().begin(), fault.Bytes().end()}, {writable.Bytes().begin(), writable.Bytes().end()}};
                std::memcpy(outcome.values.data(), output.Bytes().data(), steps.size() * 4u);
                outcome.data.insert(outcome.data.end(), readOnly.Bytes().begin(), readOnly.Bytes().end());
                return outcome;
            };
            const auto cached = run(1);
            const auto searched = run(0);
            for (const auto entries : {1u, 2u, 4u}) {
                for (const bool checked : {false, true}) Require((entries == 1u && !checked ? cached : run(entries, checked)) == searched, "BDA range cache: " + name + (stops ? " (stopping)" : " (continuing)") + " with " + std::to_string(entries) + " kept ranges" + (checked ? " and the check" : "") + " differs from the searched lookup");
            }
            if (stops) {
                for (const auto entries : {0u, 1u, 2u, 4u}) Require(run(entries, false, false) == searched, "BDA range cache: " + name + " with " + std::to_string(entries) + " kept ranges and the invocation ended at the faulting read differs from the deferred stop");
            }
            for (const std::uint64_t sharedBytes : {16u, 64u, 4096u}) {
                const auto differs = [&](const char* how) { return "BDA shared base: " + name + (stops ? " (stopping)" : " (continuing)") + " with reads sharing bases by " + std::to_string(sharedBytes) + " bytes" + how + " differs from every read looking its own bytes up"; };
                Require(run(0, false, true, sharedBytes) == searched, differs(""));
                Require(run(2, true, true, sharedBytes) == searched, differs(", two kept ranges and the check,"));
                for (const auto entries : {1u, 2u}) Require(run(entries, false, true, sharedBytes) == searched, differs(entries == 1u ? " and one kept range" : " and two kept ranges"));
                if (stops) Require(run(0, false, false, sharedBytes) == searched, differs(" and the invocation ended at the faulting read"));
            }
            if (direct) {
                const auto words = BdaDirectTable(ranges);
                Buffer directTable(context, words.size() * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                std::memcpy(directTable.Bytes().data(), words.data(), words.size() * 4u);
                const auto runDirect = [&](bool deferred, std::uint64_t hintBytes) {
                    fill();
                    std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
                    for (std::size_t word = 0; word < steps.size(); ++word) std::memcpy(output.Bytes().data() + word * 4u, &Sentinel, 4u);
                    Pipeline pipeline(context, MakeBdaSequenceTestShader(steps, 0u, stops, false, deferred, hintBytes, true), {&directTable, &fault, &output});
                    pipeline.Run(1);
                    Outcome outcome{std::vector<std::uint32_t>(steps.size()), {fault.Bytes().begin(), fault.Bytes().end()}, {writable.Bytes().begin(), writable.Bytes().end()}};
                    std::memcpy(outcome.values.data(), output.Bytes().data(), steps.size() * 4u);
                    outcome.data.insert(outcome.data.end(), readOnly.Bytes().begin(), readOnly.Bytes().end());
                    return outcome;
                };
                auto wanted = searched;
                if (!stops && !directValues.empty()) wanted.values = directValues;
                const auto differs = [&](const std::string& how) { return "BDA direct table: " + name + (stops ? " (stopping)" : " (continuing)") + how + " differs from the searched lookup"; };
                Require(runDirect(true, 0u) == wanted, differs(""));
                if (stops) Require(runDirect(false, 0u) == wanted, differs(" with the invocation ended at the faulting read"));
                for (const std::uint64_t hintBytes : {16u, 64u, 4096u}) {
                    Require(runDirect(true, hintBytes) == wanted, differs(" with leaves walked ahead by " + std::to_string(hintBytes) + " bytes"));
                    if (stops) Require(runDirect(false, hintBytes) == wanted, differs(" with leaves walked ahead by " + std::to_string(hintBytes) + " bytes and the invocation ended at the faulting read"));
                }
            }
            if (expected == nullptr) continue;
            Abi::Fault report{};
            std::memcpy(&report, cached.fault.data(), sizeof(report));
            const bool faults = static_cast<std::uint32_t>(expected->reason) != 0u;
            if (faults) {
                Require(report.state == Abi::FaultState::Ready && report.reason == expected->reason && report.address == expected->address && report.bytes == expected->bytes && report.instruction == BdaTestStepPc + expected->step, "BDA range cache: " + name + " did not publish its fault");
            } else {
                Require(report.state == Abi::FaultState::Empty, "BDA range cache: " + name + " published a fault");
            }
            Require(cached.values == (faults && stops && expected->ends ? std::vector<std::uint32_t>(steps.size(), Sentinel) : expected->values), "BDA range cache: " + name + (stops ? " (stopping)" : " (continuing)") + " produced incorrect values");
        }
    };
    using Step = BdaTestStep;
    const auto high = [](std::uint32_t byte) { return (byte + 0x81u) | (byte + 0x82u) << 8u | (byte + 0x83u) << 16u | (byte + 0x84u) << 24u; };
    Expected expected{{low(0), low(16), low(4), low(20), high(0), low(8)}};
    check("reads alternating between ranges", sorted, 3, {{Step::Read, guest}, {Step::Read, guest + 64}, {Step::Read, guest + 4}, {Step::Read, guest + 68}, {Step::Read, guest + 16}, {Step::Read, guest + 8}}, &expected);
    expected = {{high(4), 0}, Abi::FaultReason::Unmapped, guest + 32, 1, 1};
    check("a read just past the cached range's end", sorted, 3, {{Step::Read, guest + 20}, {Step::Read, guest + 32}}, &expected);
    expected = {{high(8), high(14) & 0xffffu, high(12)}, Abi::FaultReason::Unmapped, guest + 32, 1, 1};
    check("a read across the cached range's end into the gap", sorted, 3, {{Step::Read, guest + 24}, {Step::Read, guest + 30}, {Step::Read, guest + 28}}, &expected, true, {high(8), 0u, high(12)});
    expected = {{high(0), 0x55u, high(0)}, Abi::FaultReason::Permission, guest + 20, 4, 1, false};
    check("a store into a read-only range after a cached read", sorted, 3, {{Step::Read, guest + 16}, {Step::Store, guest + 20, 0x55u}, {Step::Read, guest + 16}}, &expected);
    expected = {{high(4), 0, high(4)}, Abi::FaultReason::Permission, guest + 24, 4, 1, false};
    check("an atomic in a read-only range after a cached read", sorted, 3, {{Step::Read, guest + 20}, {Step::AtomicAdd, guest + 24, 1u}, {Step::Read, guest + 20}}, &expected);
    expected = {{low(0), (low(14) & 0xffffu) | (high(0) & 0xffffu) << 16u, high(4), low(12)}};
    check("a dword straddling two ranges", sorted, 3, {{Step::Read, guest}, {Step::Read, guest + 14}, {Step::Read, guest + 20}, {Step::Read, guest + 12}}, &expected);
    expected = {{low(0), low(1), low(7), high(2), low(4)}};
    check("unaligned dwords", sorted, 3, {{Step::Read, guest}, {Step::Read, guest + 1}, {Step::Read, guest + 7}, {Step::Read, guest + 18}, {Step::Read, guest + 4}}, &expected);
    expected = {{low(4), 0xa5a5a5a5u, low(8), 0xa5a5a5a5u, low(8) + 3u, low(20), low(8) + 3u}};
    check("a store and an atomic after a cached read", sorted, 3, {{Step::Read, guest + 4}, {Step::Store, guest + 4, 0xa5a5a5a5u}, {Step::AtomicAdd, guest + 8, 3u}, {Step::Read, guest + 4}, {Step::Read, guest + 8}, {Step::Read, guest + 68}, {Step::AtomicAdd, guest + 8, 1u}}, &expected);
    expected = {{0, low(0)}, Abi::FaultReason::Unmapped, guest - 4, 1, 0};
    check("reads below the first range", sorted, 3, {{Step::Read, guest - 4}, {Step::Read, guest}}, &expected);
    expected = {{0, 0}, Abi::FaultReason::InvalidTable, guest, 1, 0};
    check("an invalid table", sorted, 2, {{Step::Read, guest}, {Step::Read, guest + 4}}, &expected, false);
    // Reads sharing a base (see the blocks above): the base's hull inside one range, across a range's
    // end into the gap (the reads before the end succeed and the one past it faults with its own
    // address), across two adjacent ranges, in the gap, two bases in turn, and reads of other sizes
    // and alignments on one base (a wide read inside the hull, one across the range's end, and
    // unaligned dwords, which share a base of their own whose device address is not 4-aligned).
    expected = {{low(0), low(4), low(8), low(12), low(4), low(0)}};
    check("a base's reads inside one range", sorted, 3, {{Step::Read, guest}, {Step::Read, guest + 4}, {Step::Read, guest + 8}, {Step::Read, guest + 12}, {Step::Read, guest + 4}, {Step::Read, guest}}, &expected);
    expected = {{high(0), high(8), high(12), 0, high(4)}, Abi::FaultReason::Unmapped, guest + 32, 1, 3};
    check("a base's reads across a range's end into the gap", sorted, 3, {{Step::Read, guest + 16}, {Step::Read, guest + 24}, {Step::Read, guest + 28}, {Step::Read, guest + 32}, {Step::Read, guest + 20}}, &expected);
    expected = {{low(8), high(0), low(12), high(12), low(0)}};
    check("a base's reads across two adjacent ranges", sorted, 3, {{Step::Read, guest + 8}, {Step::Read, guest + 16}, {Step::Read, guest + 12}, {Step::Read, guest + 28}, {Step::Read, guest}}, &expected);
    expected = {{0, 0, low(0), 0}, Abi::FaultReason::Unmapped, guest + 40, 1, 0};
    check("a base's reads in the gap", sorted, 3, {{Step::Read, guest + 40}, {Step::Read, guest + 44}, {Step::Read, guest}, {Step::Read, guest + 48}}, &expected);
    expected = {{low(0), low(16), high(0), low(4), low(20), high(4), low(8), low(24), high(8), low(12), low(28), high(12)}};
    {
        std::vector<Step> steps;
        for (const auto offset : {0u, 4u, 8u, 12u}) steps.insert(steps.end(), {{Step::Read, guest + offset}, {Step::Read, guest + 64 + offset}, {Step::Read, guest + 16 + offset}});
        check("three bases' reads in turn", sorted, 3, steps, &expected);
    }
    expected = {{low(0), low(4) + 3u * low(8), low(12), low(1), low(5), low(8) + 3u * low(12) + 5u * high(0), low(0) + 3u * low(4) + 5u * low(8) + 7u * low(12), low(9)}};
    check("a base's reads of several sizes and alignments", sorted, 3, {{Step::Read, guest}, {Step::Read, guest + 4, 0, 2}, {Step::Read, guest + 12}, {Step::Read, guest + 1}, {Step::Read, guest + 5}, {Step::Read, guest + 8, 0, 3}, {Step::Read, guest, 0, 4}, {Step::Read, guest + 9}}, &expected);
    expected = {{high(4), high(8) + 3u * high(12), high(12), high(0)}, Abi::FaultReason::Unmapped, guest + 32, 1, 2};
    check("a base's wide read across a range's end into the gap", sorted, 3, {{Step::Read, guest + 20}, {Step::Read, guest + 24, 0, 2}, {Step::Read, guest + 28, 0, 2}, {Step::Read, guest + 16}}, &expected);
    // A store keeps its range before a base's first read asks for the hull: a hull inside the kept
    // range, and one that begins in it and ends in the next range (by blocks of 64 and 4096 bytes).
    for (const auto& order : {std::array{0u, 1u, 2u}, std::array{2u, 0u, 1u}, std::array{1u, 2u, 0u}, std::array{2u, 1u, 0u}, std::array{0u, 2u, 1u}}) {
        const std::vector<Abi::Range> table{sorted[order[0]], sorted[order[1]], sorted[order[2]]};
        const bool inOrder = order[0] == 0u && order[1] == 1u;
        expected = {{0xa5a5a5a5u, low(0), 0xa5a5a5a5u, low(12), low(8)}};
        // The direct table is built from sorted ranges only.
        const bool sortedTable = inOrder && order[2] == 2u;
        check("a base's reads inside the range a store kept", table, 3, {{Step::Store, guest + 4, 0xa5a5a5a5u}, {Step::Read, guest}, {Step::Read, guest + 4}, {Step::Read, guest + 12}, {Step::Read, guest + 8}}, inOrder ? &expected : nullptr, sortedTable);
        expected = {{0x5a5a5a5au, low(8), high(0), 0x5a5a5a5au, high(4)}};
        check("a base's reads from the range a store kept into the next", table, 3, {{Step::Store, guest + 12, 0x5a5a5a5au}, {Step::Read, guest + 8}, {Step::Read, guest + 16}, {Step::Read, guest + 12}, {Step::Read, guest + 20}}, inOrder ? &expected : nullptr, sortedTable);
    }
    // More ranges than any cache keeps, each read twice in turn, then a store and reads behind it.
    {
        std::vector<Abi::Range> many;
        for (std::uint32_t range = 0; range < 6u; ++range) many.push_back({guest + range * 8u, guest + range * 8u + 4u, writable.DeviceAddress() + range * 4u, Abi::Read | Abi::Write, 0});
        std::vector<Step> steps;
        expected = {};
        for (const auto range : {0u, 1u, 2u, 3u, 4u, 5u, 0u, 1u, 0u, 5u, 3u, 2u, 2u, 4u, 1u}) {
            steps.push_back({Step::Read, guest + range * 8u});
            expected.values.push_back(low(range * 4u));
        }
        steps.push_back({Step::Store, guest + 24, 0x11223344u});
        expected.values.push_back(0x11223344u);
        for (const auto range : {0u, 3u, 5u, 3u}) {
            steps.push_back({Step::Read, guest + range * 8u});
            expected.values.push_back(range == 3u ? 0x11223344u : low(range * 4u));
        }
        check("reads over six ranges", many, 6, steps, &expected);
        expected = {{low(0), low(4), low(8), 0, low(0), low(4)}, Abi::FaultReason::Unmapped, guest + 4, 1, 3};
        check("a read in the gap behind a kept range", many, 6, {{Step::Read, guest}, {Step::Read, guest + 8}, {Step::Read, guest + 16}, {Step::Read, guest + 4}, {Step::Read, guest}, {Step::Read, guest + 8}}, &expected);
    }
    // Ranges the cache does not keep, because the checks a kept range skips could fail in them: a
    // device address of zero, and device addresses that wrap.
    {
        auto unkept = sorted;
        unkept[1].deviceAddress = 0;
        // No byte read below has a device address short of the wrap: it would be loaded.
        unkept[2].deviceAddress = ~std::uint64_t{0} - 3u;
        expected = {{low(0), 0, 0, low(4)}, Abi::FaultReason::InvalidTable, guest + 16, 1, 1};
        // The direct table refuses these ranges where it is built.
        check("a range without a device address", unkept, 3, {{Step::Read, guest}, {Step::Read, guest + 16}, {Step::Read, guest + 20}, {Step::Read, guest + 4}}, &expected, false);
        expected = {{low(0), 0, 0, low(4)}, Abi::FaultReason::Overflow, guest + 76, 1, 1};
        check("a range whose device addresses wrap", unkept, 3, {{Step::Read, guest}, {Step::Read, guest + 76}, {Step::Read, guest + 72}, {Step::Read, guest + 4}}, &expected, false);
    }
    // The search does not require sorted begins; the cache keeps the bounds of the steps it took, so
    // it must agree on any table.
    for (const auto& order : {std::array{2u, 0u, 1u}, std::array{1u, 2u, 0u}, std::array{2u, 1u, 0u}, std::array{0u, 2u, 1u}}) {
        const std::vector<Abi::Range> unsorted{sorted[order[0]], sorted[order[1]], sorted[order[2]]};
        std::vector<Step> steps;
        for (const auto offset : {0u, 64u, 16u, 4u, 68u, 20u, 14u, 30u, 76u, 12u, 40u, 8u}) steps.push_back({Step::Read, guest + offset});
        check("an unsorted table", unsorted, 3, steps, nullptr, false);
    }
}

// What the direct table gives the bytes at `address`, walked as the shader walks it: the device
// address of the first, or 0, and whether they may be written.
static std::pair<std::uint64_t, bool> WalkDirectTable(std::span<const std::uint32_t> words, std::uint64_t address, std::uint32_t bytes) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Abi::DirectHeader header{};
    std::memcpy(&header, words.data(), sizeof(header));
    Require(header.version == Abi::DirectVersion && header.reserved == 0 && header.rootShift <= Abi::DirectMaxRootShift && header.rootCount <= header.entries && words.size() == sizeof(header) / 4u + header.entries * 4u + header.ranges * 8u, "BDA direct table: invalid header");
    const auto entry = [&](std::uint64_t index) {
        Require(index < header.entries, "BDA direct table: an entry lies outside the table");
        Abi::DirectEntry value{};
        std::memcpy(&value, words.data() + sizeof(header) / 4u + index * 4u, sizeof(value));
        return value;
    };
    const auto relative = address - header.base;
    if ((relative >> header.rootShift) >= header.rootCount) return {0, false};
    auto found = entry(relative >> header.rootShift);
    auto offset = static_cast<std::uint32_t>(relative & ((std::uint64_t{1} << header.rootShift) - 1u));
    for (int level = 0; (found.flags & Abi::DirectDescend) != 0; ++level) {
        Require(level < 8, "BDA direct table: the walk does not end");
        const auto shift = static_cast<std::uint32_t>(found.delta >> 32u);
        Require(found.bytes == (1u << shift) - 1u, "BDA direct table: an entry's mask is not its shift's");
        const auto child = static_cast<std::uint32_t>(found.delta) + (offset >> shift);
        offset &= found.bytes;
        found = entry(child);
    }
    if (static_cast<std::uint64_t>(offset) + bytes > found.bytes) return {0, false};
    return {address + found.delta, (found.flags & Abi::DirectWritable) != 0};
}

// The direct table (BdaAbi::DirectVersion) beyond what the sequences above compare with the search:
// its tree on the CPU against the ranges it was built from (every root size, units split down to
// single bytes, mappings longer than a leaf can count), reads at the starts and ends of mappings
// far apart and through every level, dwords across two mappings where the table has a seam and where
// it has none, the immediate a read adds itself, an invalid header, and one shader on tables that
// change between its runs.
static void RunBdaDirectTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    using Step = BdaTestStep;
    constexpr std::uint32_t Sentinel = 0xdeadbeef;
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    constexpr std::uint64_t guest = 0x7fff12340000ULL;
    Buffer first(context, 64, usage);
    Buffer second(context, 32, usage);
    Require(first.DeviceAddress() % 4u == 0u && second.DeviceAddress() % 4u == 0u, "BDA direct table: the test buffers are not dword-aligned");
    const auto fill = [&] {
        for (std::size_t byte = 0; byte < first.Bytes().size(); ++byte) first.Bytes()[byte] = static_cast<std::byte>(byte + 1u);
        for (std::size_t byte = 0; byte < second.Bytes().size(); ++byte) second.Bytes()[byte] = static_cast<std::byte>(byte + 0x81u);
    };
    const auto low = [](std::uint32_t byte) { return (byte + 1u) | (byte + 2u) << 8u | (byte + 3u) << 16u | (byte + 4u) << 24u; };
    const auto high = [](std::uint32_t byte) { return (byte + 0x81u) | (byte + 0x82u) << 8u | (byte + 0x83u) << 16u | (byte + 0x84u) << 24u; };

    // The tree against its ranges, on the CPU: every byte near a range's begin and end, and far from both.
    const auto verify = [&](const std::string& name, const std::vector<Abi::Range>& ranges, std::uint32_t rootShift) {
        const auto words = BdaDirectTable(ranges);
        Abi::DirectHeader header{};
        std::memcpy(&header, words.data(), sizeof(header));
        Require(header.rootShift == rootShift && header.ranges == ranges.size(), "BDA direct table: " + name + " has root units of 1 << " + std::to_string(header.rootShift) + " bytes");
        std::vector<std::uint64_t> addresses;
        for (const auto& range : ranges) {
            for (std::uint64_t near = 0; near < 9u; ++near) addresses.insert(addresses.end(), {range.begin - 4u + near, range.end - 5u + near, range.begin + (range.end - range.begin) / 2u + near, range.begin - 0x12345u + near, range.end + 0x54321u + near});
        }
        for (const auto address : addresses) {
            for (const std::uint32_t bytes : {1u, 4u}) {
                std::uint64_t device = 0;
                bool writable = false;
                // As the search selects, with touching ranges of one distance and permission as one.
                for (std::size_t index = 0; index < ranges.size(); ++index) {
                    if (address < ranges[index].begin || address >= ranges[index].end) continue;
                    auto end = ranges[index].end;
                    for (auto next = index + 1; next < ranges.size() && ranges[next].begin == end && ranges[next].deviceAddress - ranges[next].begin == ranges[index].deviceAddress - ranges[index].begin && ranges[next].permissions == ranges[index].permissions; ++next) end = ranges[next].end;
                    if (address + bytes > end) break;
                    device = address + (ranges[index].deviceAddress - ranges[index].begin);
                    writable = (ranges[index].permissions & Abi::Write) != 0;
                }
                const auto walked = WalkDirectTable(words, address, bytes);
                Require(walked.first == device && (device == 0 || walked.second == writable), "BDA direct table: " + name + " gives another device address for " + std::to_string(bytes) + " bytes at guest offset " + std::to_string(static_cast<std::int64_t>(address - ranges.front().begin)));
            }
        }
        return words.size();
    };
    const std::uint64_t device = 0x1000000000ULL;
    verify("one small range", {{guest, guest + 80, device, Abi::Read, 0}}, 14);
    verify("ranges that split their units down to bytes", {{guest + 1, guest + 4, device + 1, Abi::Read, 0}, {guest + 4, guest + 9, device + 0x100, Abi::Read | Abi::Write, 0}, {guest + 9, guest + 0x4003, device + 0x209, Abi::Read, 0}, {guest + 0x4003, guest + 0x4004, device + 0x8003, Abi::Read, 0}, {guest + 0x4080, guest + 0x9000, device + 0x9080, Abi::Read, 0}}, 14);
    verify("touching ranges of one distance", {{guest, guest + 0x4000, device, Abi::Read, 0}, {guest + 0x4000, guest + 0x8001, device + 0x4000, Abi::Read, 0}, {guest + 0x8001, guest + 0x8010, device + 0x8001, Abi::Read | Abi::Write, 0}}, 14);
    verify("ranges more than 1 GiB apart", {{guest + 0x123, guest + 0x40000, device, Abi::Read, 0}, {guest + 0x80000000ULL, guest + 0x80800000ULL, device + 0x100000, Abi::Read | Abi::Write, 0}, {guest + 0x80800000ULL, guest + 0x80800010ULL, device + 0x4000000, Abi::Read, 0}}, 22);
    verify("ranges more than 256 GiB apart", {{0x200000000ULL, 0x200004000ULL, device, Abi::Read, 0}, {0x4000000000ULL + 7, 0x4000000000ULL + 0x12345, device + 0x8000, Abi::Read, 0}, {guest, guest + 0x40000000ULL, device + 0x100000, Abi::Read | Abi::Write, 0}}, 30);
    verify("a range longer than a leaf counts", {{0x200000000ULL, 0x380000000ULL, device, Abi::Read, 0}, {0x380000000ULL, 0x380000100ULL, device * 2, Abi::Read, 0}}, 22);
    Require(BdaDirectTable({}).size() == sizeof(Abi::DirectHeader) / 4u, "BDA direct table: the table of no range has entries");
    // A table that would be refused where it is built.
    for (const auto& refused : {std::vector<Abi::Range>{{guest + 16, guest + 32, device, Abi::Read, 0}, {guest, guest + 16, device, Abi::Read, 0}}, std::vector<Abi::Range>{{guest, guest + 16, 0, Abi::Read, 0}}, std::vector<Abi::Range>{{guest, guest + 16, device, Abi::Write, 0}}, std::vector<Abi::Range>{{0x1000, 0x2000, device, Abi::Read, 0}}, std::vector<Abi::Range>{{guest, guest + 16, ~std::uint64_t{0} - 8u, Abi::Read, 0}}}) {
        bool thrown = false;
        try {
            BdaDirectTable(refused);
        } catch (const std::runtime_error&) {
            thrown = true;
        }
        Require(thrown, "BDA direct table: an unsorted or invalid range was accepted");
    }

    struct Outcome {
        std::vector<std::uint32_t> values;
        Abi::Fault report{};
    };
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto runWords = [&](std::span<const std::uint32_t> code, std::span<const std::uint32_t> table, std::size_t count) {
        Buffer buffer(context, table.size() * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memcpy(buffer.Bytes().data(), table.data(), table.size() * 4u);
        Buffer output(context, count * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        for (std::size_t word = 0; word < count; ++word) std::memcpy(output.Bytes().data() + word * 4u, &Sentinel, 4u);
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        fill();
        Pipeline pipeline(context, code, {&buffer, &fault, &output});
        pipeline.Run(1);
        Outcome outcome{std::vector<std::uint32_t>(count)};
        std::memcpy(outcome.values.data(), output.Bytes().data(), count * 4u);
        std::memcpy(&outcome.report, fault.Bytes().data(), sizeof(outcome.report));
        return outcome;
    };
    const auto searchedTable = [](const std::vector<Abi::Range>& ranges) {
        std::vector<std::uint32_t> words((sizeof(Abi::Header) + ranges.size() * sizeof(Abi::Range)) / 4u);
        const Abi::Header header{Abi::Version, static_cast<std::uint32_t>(ranges.size()), sizeof(Abi::Range), 0};
        std::memcpy(words.data(), &header, sizeof(header));
        std::memcpy(words.data() + sizeof(header) / 4u, ranges.data(), ranges.size() * sizeof(Abi::Range));
        return words;
    };
    const auto same = [](const Outcome& left, const Outcome& right) { return left.values == right.values && std::memcmp(&left.report, &right.report, sizeof(left.report)) == 0; };
    // The steps on the direct table, in every form, against the search of the same ranges.
    const auto compare = [&](const std::string& name, const std::vector<Abi::Range>& ranges, const std::vector<Step>& steps) {
        const auto direct = BdaDirectTable(ranges);
        const auto searched = searchedTable(ranges);
        for (const bool stops : {true, false}) {
            const auto wanted = runWords(MakeBdaSequenceTestShader(steps, 0u, stops), searched, steps.size());
            for (const std::uint64_t hintBytes : {0u, 16u, 4096u}) {
                for (const bool deferred : {true, false}) {
                    if (!deferred && !stops) continue;
                    Require(same(runWords(MakeBdaSequenceTestShader(steps, 0u, stops, false, deferred, hintBytes, true), direct, steps.size()), wanted), "BDA direct table: " + name + (stops ? " (stopping" : " (continuing") + (deferred ? "" : ", ended at the read") + ", leaves walked ahead by " + std::to_string(hintBytes) + " bytes) differs from the searched lookup");
                }
            }
        }
        return runWords(MakeBdaSequenceTestShader(steps, 0u, false), searched, steps.size());
    };
    // The steps on the direct table where it differs from the search: the values of a run that
    // continues, and the fault.
    const auto expect = [&](const std::string& name, std::span<const std::uint32_t> direct, const std::vector<Step>& steps, const std::vector<std::uint32_t>& values, Abi::FaultReason reason, std::uint64_t address, std::uint32_t bytes, std::uint32_t step) {
        for (const bool stops : {true, false}) {
            for (const std::uint64_t hintBytes : {0u, 16u, 4096u}) {
                for (const bool deferred : {true, false}) {
                    if (!deferred && !stops) continue;
                    const auto outcome = runWords(MakeBdaSequenceTestShader(steps, 0u, stops, false, deferred, hintBytes, true), direct, steps.size());
                    const auto form = name + (stops ? " (stopping" : " (continuing") + (deferred ? "" : ", ended at the read") + ", leaves walked ahead by " + std::to_string(hintBytes) + " bytes)";
                    Require(outcome.report.state == Abi::FaultState::Ready && outcome.report.reason == reason && outcome.report.address == address && outcome.report.bytes == bytes && outcome.report.instruction == BdaTestStepPc + step && outcome.report.reserved == 0, "BDA direct table: " + form + " did not publish its fault");
                    Require(outcome.values == (stops ? std::vector<std::uint32_t>(steps.size(), Sentinel) : values), "BDA direct table: " + form + " produced incorrect values");
                }
            }
        }
    };

    // Three mappings far apart, the middle one from an odd address to an odd address: its reads go
    // through every level of the tree. Its device addresses are odd where the guest's are, so a
    // dword at a multiple of 4 is loaded as one and any other byte by byte.
    const std::uint64_t far = guest + 0x4000000000ULL;
    const std::uint64_t odd = guest + 0x40000003ULL;
    const std::vector<Abi::Range> apart{{guest, guest + 16, first.DeviceAddress(), Abi::Read | Abi::Write, 0}, {odd, odd + 13, second.DeviceAddress() + 3, Abi::Read, 0}, {far, far + 16, first.DeviceAddress() + 16, Abi::Read | Abi::Write, 0}};
    {
        Abi::DirectHeader header{};
        const auto words = BdaDirectTable(apart);
        std::memcpy(&header, words.data(), sizeof(header));
        Require(header.rootShift == 30, "BDA direct table: the far mappings have another root");
    }
    const auto inside = compare("reads at the starts and ends of mappings", apart, {{Step::Read, guest}, {Step::Read, guest + 12}, {Step::Read, odd}, {Step::Read, odd + 1}, {Step::Read, odd + 9}, {Step::Read, far}, {Step::Read, far + 12}, {Step::Read, odd + 5, 0, 2}, {Step::Store, far + 4, 0x600df00du}, {Step::Read, far + 4}, {Step::AtomicAdd, guest + 8, 5u}, {Step::Read, guest + 8}});
    Require(inside.report.state == Abi::FaultState::Empty && inside.values == std::vector<std::uint32_t>({low(0), low(12), high(3), high(4), high(12), low(16), low(28), high(8) + 3u * high(12), 0x600df00du, 0x600df00du, low(8), low(8) + 5u}), "BDA direct table: reads at the starts and ends of mappings produced incorrect values");
    // A read outside every mapping: reported, and a stopping invocation leaves no value.
    std::uint32_t outside = 0;
    for (const std::uint64_t address : std::initializer_list<std::uint64_t>{guest + 16, guest - 4, odd - 4, odd + 13, far + 16, far - 0x100000u, guest + 0x100000000ULL, far + 0x40000000u, 0x1000u}) {
        const auto unmapped = compare("a read outside every mapping", apart, {{Step::Read, guest + 4}, {Step::Read, address}, {Step::Read, far + 8}});
        Require(unmapped.report.state == Abi::FaultState::Ready && unmapped.report.reason == Abi::FaultReason::Unmapped && unmapped.report.address == address && unmapped.report.bytes == 1 && unmapped.report.instruction == BdaTestStepPc + 1, "BDA direct table: a read outside every mapping was not reported");
        ++outside;
    }
    Require(outside == 9, "BDA direct table: not every read outside the mappings ran");
    // A dword whose first bytes only are mapped: the fault is the search's, the first unmapped byte.
    expect("a dword across a mapping's end", BdaDirectTable(apart), {{Step::Read, odd + 4}, {Step::Read, odd + 11}, {Step::Read, guest}}, {high(7), 0u, low(0)}, Abi::FaultReason::Unmapped, odd + 13, 1, 1);
    expect("a dword across a mapping's begin", BdaDirectTable(apart), {{Step::Read, odd - 2}, {Step::Read, guest}}, {0u, low(0)}, Abi::FaultReason::Unmapped, odd - 2, 1, 0);
    // A store and an atomic on a mapping that is not writable are refused as the search refuses them.
    compare("a store to a read-only mapping", apart, {{Step::Read, odd + 1}, {Step::Store, odd + 5, 0x55u}, {Step::Read, odd + 5}});
    compare("an atomic on a read-only mapping", apart, {{Step::AtomicAdd, odd + 5, 1u}, {Step::Read, odd + 5}});

    // Two mappings that touch at a multiple of 4, both at 4-aligned device addresses: the dwords
    // across them are read as the search's byte path reads them, and so are those on either side.
    const std::vector<Abi::Range> seamed{{guest, guest + 16, first.DeviceAddress(), Abi::Read, 0}, {guest + 16, guest + 32, second.DeviceAddress(), Abi::Read, 0}, {guest + 32, guest + 48, first.DeviceAddress() + 32, Abi::Read, 0}};
    const auto across = compare("dwords across neighbouring mappings", seamed, {{Step::Read, guest + 12}, {Step::Read, guest + 13}, {Step::Read, guest + 14}, {Step::Read, guest + 15}, {Step::Read, guest + 16}, {Step::Read, guest + 29}, {Step::Read, guest + 30}, {Step::Read, guest + 31}, {Step::Read, guest + 32}, {Step::Read, guest + 12, 0, 4}, {Step::Read, guest + 26, 0, 3}});
    Require(across.report.state == Abi::FaultState::Empty && across.values[1] == ((low(13) & 0xffffffu) | (high(0) & 0xffu) << 24u) && across.values[3] == ((low(15) & 0xffu) | (high(0) & 0xffffffu) << 8u) && across.values[6] == ((high(14) & 0xffffu) | (low(32) & 0xffffu) << 16u), "BDA direct table: dwords across neighbouring mappings produced incorrect values");
    // Touching mappings where the table has no seam: the first one's last bytes at an odd device
    // address, and mappings shorter than four bytes. The dwords inside each are read; one across
    // them is refused as Split, which the search reads byte by byte.
    const std::vector<Abi::Range> crooked{{guest, guest + 16, first.DeviceAddress() + 1, Abi::Read, 0}, {guest + 16, guest + 32, second.DeviceAddress(), Abi::Read, 0}};
    compare("dwords on either side of mappings without a seam", crooked, {{Step::Read, guest}, {Step::Read, guest + 12}, {Step::Read, guest + 16}, {Step::Read, guest + 5}, {Step::Read, guest + 28}});
    expect("a dword across mappings without a seam", BdaDirectTable(crooked), {{Step::Read, guest + 8}, {Step::Read, guest + 14}, {Step::Read, guest + 16}}, {low(9), 0u, high(0)}, Abi::FaultReason::Split, guest + 14, 4, 1);
    const std::vector<Abi::Range> tiny{{guest + 1, guest + 4, first.DeviceAddress(), Abi::Read, 0}, {guest + 4, guest + 6, second.DeviceAddress(), Abi::Read, 0}};
    expect("a dword across mappings shorter than four bytes", BdaDirectTable(tiny), {{Step::Read, guest + 2}}, {0u}, Abi::FaultReason::Split, guest + 2, 4, 0);

    // One shader on tables that change between its runs: the same guest bytes at other device
    // addresses, and then not mapped at all.
    {
        const std::vector<Step> steps{{Step::Read, guest}, {Step::Read, guest + 12}, {Step::Read, guest + 20}};
        const auto code = MakeBdaSequenceTestShader(steps, 0u, true, false, true, 16u, true);
        const auto before = runWords(code, BdaDirectTable(std::vector<Abi::Range>{{guest, guest + 32, first.DeviceAddress(), Abi::Read, 0}}), steps.size());
        const auto after = runWords(code, BdaDirectTable(std::vector<Abi::Range>{{guest, guest + 16, second.DeviceAddress() + 16, Abi::Read, 0}, {guest + 16, guest + 32, second.DeviceAddress(), Abi::Read, 0}}), steps.size());
        const auto gone = runWords(code, BdaDirectTable(std::vector<Abi::Range>{{guest, guest + 16, first.DeviceAddress(), Abi::Read, 0}}), steps.size());
        Require(before.report.state == Abi::FaultState::Empty && before.values == std::vector<std::uint32_t>({low(0), low(12), low(20)}), "BDA direct table: the first table's bytes were not read");
        Require(after.report.state == Abi::FaultState::Empty && after.values == std::vector<std::uint32_t>({high(16), high(28), high(4)}), "BDA direct table: a changed table's bytes were not read");
        Require(gone.report.state == Abi::FaultState::Ready && gone.report.reason == Abi::FaultReason::Unmapped && gone.report.address == guest + 20 && gone.report.bytes == 1 && gone.report.instruction == BdaTestStepPc + 2 && gone.values == std::vector<std::uint32_t>(steps.size(), Sentinel), "BDA direct table: a read of a range the table no longer maps was not reported");
    }

    // A header that is not a direct table's: every access is refused, as with the ranges' header.
    {
        auto words = BdaDirectTable(apart);
        words[0] = Abi::Version;
        expect("an invalid header", words, {{Step::Read, guest}, {Step::Read, guest + 4}}, {0u, 0u}, Abi::FaultReason::InvalidTable, guest, 1, 0);
        for (const bool stops : {true, false}) {
            const std::vector<Step> store{{Step::Store, guest, 1u}};
            const auto outcome = runWords(MakeBdaSequenceTestShader(store, 0u, stops, false, true, 0u, true), words, 1);
            Require(outcome.report.state == Abi::FaultState::Ready && outcome.report.reason == Abi::FaultReason::InvalidTable && outcome.report.address == guest && outcome.report.bytes == 4, "BDA direct table: a store under an invalid header was not refused");
        }
        Require(first.Bytes()[0] == std::byte{1}, "BDA direct table: a store under an invalid header reached its range");
        fill();
    }

    // The immediate a dword read adds to its base itself: one that wraps is the fault the checked
    // sum of the search records, at the base and of no bytes, and a dword that wraps is its own.
    {
        Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const auto tables = std::array{searchedTable(apart), BdaDirectTable(apart)};
        const auto read = [&](bool direct, bool deferred, std::uint64_t address, std::int64_t offset) {
            const auto& words = tables[direct ? 1u : 0u];
            Buffer buffer(context, words.size() * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            std::memcpy(buffer.Bytes().data(), words.data(), words.size() * 4u);
            std::memcpy(output.Bytes().data(), &Sentinel, 4u);
            std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
            Pipeline pipeline(context, MakeBdaTestShader(address, 32u, offset, deferred, direct), {&buffer, &fault, &output});
            pipeline.Run(1);
            Outcome outcome{std::vector<std::uint32_t>(1)};
            std::memcpy(outcome.values.data(), output.Bytes().data(), 4u);
            std::memcpy(&outcome.report, fault.Bytes().data(), sizeof(outcome.report));
            return outcome;
        };
        struct Case {
            std::uint64_t address;
            std::int64_t offset;
            Abi::FaultReason reason;
            std::uint64_t reported;
            std::uint32_t bytes;
            std::uint32_t value;
        };
        const auto top = std::numeric_limits<std::uint64_t>::max();
        for (const auto& test : {Case{guest - 8, 8, static_cast<Abi::FaultReason>(0), 0, 0, low(0)}, Case{guest + 0x1000c, -0x10000, static_cast<Abi::FaultReason>(0), 0, 0, low(12)}, Case{far + 20, -8, static_cast<Abi::FaultReason>(0), 0, 0, low(28)}, Case{top - 2, 4, Abi::FaultReason::Overflow, top - 2, 0, 0}, Case{top, 0x10000, Abi::FaultReason::Overflow, top, 0, 0}, Case{1, -4, Abi::FaultReason::Overflow, 1, 0, 0}, Case{0xffff, -0x10000, Abi::FaultReason::Overflow, 0xffff, 0, 0}, Case{top - 1, 0, Abi::FaultReason::Overflow, top - 1, 4, 0}, Case{top - 9, 6, Abi::FaultReason::Overflow, top - 3, 4, 0}, Case{guest + 8, 8, Abi::FaultReason::Unmapped, guest + 16, 1, 0}, Case{4, -4, Abi::FaultReason::Unmapped, 0, 1, 0}}) {
            const auto wanted = read(false, true, test.address, test.offset);
            const bool faults = static_cast<std::uint32_t>(test.reason) != 0u;
            Require(faults ? wanted.report.reason == test.reason && wanted.report.address == test.reported && wanted.report.bytes == test.bytes && wanted.values[0] == Sentinel : wanted.report.state == Abi::FaultState::Empty && wanted.values[0] == test.value, "BDA direct table: the searched read with an immediate is not the expected one");
            for (const bool deferred : {true, false}) Require(same(read(true, deferred, test.address, test.offset), wanted), "BDA direct table: a read at base " + std::to_string(test.address) + " with immediate " + std::to_string(test.offset) + " differs from the searched lookup");
        }
    }
}

void RunBdaExecutionTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Buffer first(context, 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer second(context, 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    first.Bytes()[0] = std::byte{0x11};
    first.Bytes()[1] = std::byte{0x22};
    first.Bytes()[2] = std::byte{0x33};
    second.Bytes()[0] = std::byte{0x44};
    second.Bytes()[1] = std::byte{0x55};
    constexpr std::uint64_t guest = 0x7fff12340001ULL;
    std::array<Abi::Range, 2> ranges{{{guest, guest + 3, first.DeviceAddress(), Abi::Read, 0}, {guest + 3, guest + 5, second.DeviceAddress(), Abi::Read, 0}}};
    Buffer table(context, sizeof(Abi::Header) + sizeof(ranges), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, sizeof(Abi::Fault), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto runWith = [&](bool deferred, std::uint64_t address, std::uint32_t bits, std::uint32_t expected, Abi::FaultReason reason, std::uint32_t count, std::uint32_t groups, std::int64_t offset) {
        const Abi::Header header{Abi::Version, count, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::uint32_t sentinel = 0xdeadbeef;
        std::memcpy(output.Bytes().data(), &sentinel, sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaTestShader(address, bits, offset, deferred), {&table, &fault, &output});
        pipeline.Run(groups);
        Abi::Fault report{};
        std::uint32_t result = 0;
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(&result, output.Bytes().data(), sizeof(result));
        if (static_cast<std::uint32_t>(reason) == 0) {
            Require(report.state == Abi::FaultState::Empty && result == expected, "BDA GPU read produced incorrect data or a fault");
        } else {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.instruction == 0x1234, "BDA GPU fault was not published correctly");
            Require(result == sentinel, "faulting BDA shader continued to output a substitute value");
        }
    };
    const auto run = [&](std::uint64_t address, std::uint32_t bits, std::uint32_t expected, Abi::FaultReason reason, std::uint32_t count = 2, std::uint32_t groups = 1, std::int64_t offset = 0) {
        for (const bool deferred : {true, false}) runWith(deferred, address, bits, expected, reason, count, groups, offset);
    };
    run(guest, 8, 0x11, static_cast<Abi::FaultReason>(0));
    run(guest + 1, 16, 0x3322, static_cast<Abi::FaultReason>(0));
    run(guest + 1, 32, 0x55443322, static_cast<Abi::FaultReason>(0));
    run(guest + 5, 8, 0, Abi::FaultReason::Unmapped, 2, 64);
    run(guest - 1, 8, 0, Abi::FaultReason::Unmapped);
    run(std::numeric_limits<std::uint64_t>::max() - 1, 32, 0, Abi::FaultReason::Overflow);
    run(guest, 8, 0, Abi::FaultReason::InvalidTable, 3);
    run(std::numeric_limits<std::uint64_t>::max() - 2, 8, 0, Abi::FaultReason::Overflow, 2, 1, 4);
    run(1, 8, 0, Abi::FaultReason::Overflow, 2, 1, -4);
    Buffer words(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto read = [&](std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops, std::uint32_t value, bool faults, bool deferred) {
        const Abi::Header header{Abi::Version, 2, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::array<std::uint32_t, 4> sentinel{0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
        std::memcpy(words.Bytes().data(), sentinel.data(), sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaDwordReadTestShader(address, dwords, coherent, stops, deferred), {&table, &fault, &words});
        pipeline.Run(1);
        Abi::Fault report{};
        std::array<std::uint32_t, 4> result{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(result.data(), words.Bytes().data(), sizeof(result));
        auto expected = sentinel;
        if (!faults || !stops) {
            expected[0] = value;
            for (std::uint32_t dword = 1; dword < dwords; ++dword) expected[dword] = 0;
        }
        if (faults) {
            Require(report.state == Abi::FaultState::Ready && report.reason == Abi::FaultReason::Unmapped && report.address == guest + 5 && report.bytes == 1 && report.instruction == 0x1234, "BDA dword read did not publish its first fault");
        } else {
            Require(report.state == Abi::FaultState::Empty, "mapped BDA dword read published a fault");
        }
        Require(result == expected, faults && stops ? "faulting BDA dword read continued to output a substitute value" : "BDA dword read produced incorrect data");
    };
    for (const bool stops : {true, false}) {
        for (const bool coherent : {false, true}) {
            for (const bool deferred : {true, false}) {
                if (deferred && !stops) continue;
                read(guest + 1, 1, coherent, stops, 0x55443322, false, deferred);
                read(guest + 3, 1, coherent, stops, 0x5544, true, deferred);
                read(guest + 1, 4, coherent, stops, 0x55443322, true, deferred);
            }
        }
    }
    ranges[0].permissions = 0;
    run(guest, 8, 0, Abi::FaultReason::Permission);
    RunBdaRangeCacheTests(context);
    RunBdaDirectTests(context);
}
