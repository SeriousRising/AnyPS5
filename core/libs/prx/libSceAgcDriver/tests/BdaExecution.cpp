#include "BdaShader.hpp"
#include "BdaAbi.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <array>
#include <cstring>
#include <limits>
#include <string>
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

// Each sequence runs with the range cache and without it (APS5_NO_BDA_RANGE_CACHE=1 in the
// recompiler), with and without stopping invocations: the output words, the whole fault buffer
// (fault record and noted pages) and the data must be equal, and the cached run must give the
// stated values and fault.
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
    const auto check = [&](const std::string& name, const std::vector<Abi::Range>& ranges, std::uint32_t count, const std::vector<BdaTestStep>& steps, const Expected* expected) {
        Buffer table(context, sizeof(Abi::Header) + ranges.size() * sizeof(Abi::Range), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const Abi::Header header{Abi::Version, count, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), ranges.size() * sizeof(Abi::Range));
        Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer output(context, steps.size() * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        for (const bool stops : {true, false}) {
            const auto run = [&](bool rangeCache) {
                fill();
                std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
                for (std::size_t word = 0; word < steps.size(); ++word) std::memcpy(output.Bytes().data() + word * 4u, &Sentinel, 4u);
                Pipeline pipeline(context, MakeBdaSequenceTestShader(steps, rangeCache, stops), {&table, &fault, &output});
                pipeline.Run(1);
                Outcome outcome{std::vector<std::uint32_t>(steps.size()), {fault.Bytes().begin(), fault.Bytes().end()}, {writable.Bytes().begin(), writable.Bytes().end()}};
                std::memcpy(outcome.values.data(), output.Bytes().data(), steps.size() * 4u);
                outcome.data.insert(outcome.data.end(), readOnly.Bytes().begin(), readOnly.Bytes().end());
                return outcome;
            };
            const auto cached = run(true);
            Require(cached == run(false), "BDA range cache: " + name + (stops ? " (stopping)" : " (continuing)") + " differs from the searched lookup");
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
    check("a read across the cached range's end into the gap", sorted, 3, {{Step::Read, guest + 24}, {Step::Read, guest + 30}, {Step::Read, guest + 28}}, &expected);
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
    check("an invalid table", sorted, 2, {{Step::Read, guest}, {Step::Read, guest + 4}}, &expected);
    // The search does not require sorted begins; the cache keeps the bounds of the steps it took, so
    // it must agree on any table.
    for (const auto& order : {std::array{2u, 0u, 1u}, std::array{1u, 2u, 0u}, std::array{2u, 1u, 0u}, std::array{0u, 2u, 1u}}) {
        const std::vector<Abi::Range> unsorted{sorted[order[0]], sorted[order[1]], sorted[order[2]]};
        std::vector<Step> steps;
        for (const auto offset : {0u, 64u, 16u, 4u, 68u, 20u, 14u, 30u, 76u, 12u, 40u, 8u}) steps.push_back({Step::Read, guest + offset});
        check("an unsorted table", unsorted, 3, steps, nullptr);
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
    const auto run = [&](std::uint64_t address, std::uint32_t bits, std::uint32_t expected, Abi::FaultReason reason, std::uint32_t count = 2, std::uint32_t groups = 1, std::int64_t offset = 0) {
        const Abi::Header header{Abi::Version, count, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::uint32_t sentinel = 0xdeadbeef;
        std::memcpy(output.Bytes().data(), &sentinel, sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaTestShader(address, bits, offset), {&table, &fault, &output});
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
    const auto read = [&](std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops, std::uint32_t value, bool faults) {
        const Abi::Header header{Abi::Version, 2, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::array<std::uint32_t, 4> sentinel{0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
        std::memcpy(words.Bytes().data(), sentinel.data(), sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaDwordReadTestShader(address, dwords, coherent, stops), {&table, &fault, &words});
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
            read(guest + 1, 1, coherent, stops, 0x55443322, false);
            read(guest + 3, 1, coherent, stops, 0x5544, true);
            read(guest + 1, 4, coherent, stops, 0x55443322, true);
        }
    }
    ranges[0].permissions = 0;
    run(guest, 8, 0, Abi::FaultReason::Permission);
    RunBdaRangeCacheTests(context);
}
