#include "BdaShader.hpp"
#include "BdaAbi.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaDirectTable.hpp"
#include "IntermediateRepresentation/IrMetadata/ShaderStage.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <set>
#include <span>
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

static void RunWrittenRangeDecodeTests() {
    namespace Abi = ShaderRecompiler::BdaAbi;
    std::vector<std::uint32_t> words(Abi::FaultBufferBytes / sizeof(std::uint32_t));
    std::vector<std::pair<std::uint64_t, std::size_t>> marked;
    const auto decode = [&] {
        marked.clear();
        return Abi::ForEachWrittenRange(words.data(), [&](std::uint64_t address, std::size_t bytes) { marked.emplace_back(address, bytes); });
    };
    const auto claim = [&](std::uint32_t index, std::uint32_t slot, std::uint64_t address, std::uint32_t mask) {
        words[Abi::WrittenListWord + index] = slot;
        words[Abi::WrittenSlotsWord + 2 * slot] = Abi::WrittenChunkKey(address);
        words[Abi::WrittenSlotsWord + 2 * slot + 1] = mask;
        words[Abi::WrittenCountWord] = std::max(words[Abi::WrittenCountWord], index + 1);
    };
    constexpr std::uint64_t page = std::uint64_t{1} << Abi::WrittenPageShift;
    Require(decode() && marked.empty(), "an empty written-page record marked pages");
    constexpr std::uint64_t top = 0x7ffffffe0000ULL;
    claim(0, 5, top, 0b1011u);
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{top, 2 * page}, {top + 3 * page, page}}, "written pages near the top of the address space were not decoded as written");
    std::fill(words.begin(), words.end(), 0u);
    constexpr std::uint64_t high = 0x200000001000ULL;
    claim(0, Abi::WrittenPageSlots - 1, high, 1u << ((high >> Abi::WrittenPageShift) & (Abi::WrittenChunkPages - 1u)));
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{high, page}}, "a written page above 16 TiB was decoded at another address");
    std::fill(words.begin(), words.end(), 0u);
    constexpr std::uint64_t chunk = 0x3e0000000ULL;
    claim(0, 7, chunk, 0xffffffffu);
    claim(1, 8, chunk + (std::uint64_t{1} << Abi::WrittenChunkShift), 1u);
    claim(2, 9, chunk - (std::uint64_t{1} << Abi::WrittenChunkShift), 0x80000000u);
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{chunk, 32 * page}, {chunk + 32 * page, page}, {chunk - page, page}}, "whole, first-page and last-page chunks were not decoded exactly");
    words[Abi::WrittenListWord + 1] = Abi::WrittenPageSlots;
    Require(!decode(), "a claim outside the slots was accepted");
    std::fill(words.begin(), words.end(), 0u);
    words[Abi::WrittenCountWord] = 1;
    Require(!decode(), "a claim of an empty slot was accepted");
    words[Abi::WrittenCountWord] = Abi::WrittenPageSlots + 1;
    Require(!decode(), "more claims than slots were accepted");
}

static void RunBdaDwordWriteTests(const Context& context) {
    RunWrittenRangeDecodeTests();
    namespace Abi = ShaderRecompiler::BdaAbi;
    constexpr std::uint64_t guest = 0x7fff12340f80ULL;
    constexpr std::size_t firstBytes = 160;
    Buffer first(context, firstBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer second(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer readOnly(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    const std::array<Abi::Range, 3> ranges{{
        {guest, guest + firstBytes, first.DeviceAddress(), Abi::Read | Abi::Write, 0},
        {guest + firstBytes, guest + firstBytes + 16, second.DeviceAddress(), Abi::Read | Abi::Write, 0},
        {guest + firstBytes + 16, guest + firstBytes + 32, readOnly.DeviceAddress(), Abi::Read, 0},
    }};
    Buffer table(context, sizeof(Abi::Header) + sizeof(ranges), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const std::array<std::uint32_t, 4> values{0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
    const auto bytesAt = [&](std::uint64_t address) -> std::byte* {
        for (std::size_t index = 0; index < ranges.size(); ++index) {
            if (address < ranges[index].begin || address >= ranges[index].end) continue;
            auto& buffer = index == 0 ? first : index == 1 ? second : readOnly;
            return buffer.Bytes().data() + (address - ranges[index].begin);
        }
        return nullptr;
    };
    const auto notedBytes = [&](std::uint64_t first, std::uint64_t last) {
        std::uint64_t covered = 0, total = 0;
        const auto* faultWords = reinterpret_cast<const std::uint32_t*>(fault.Bytes().data());
        Require(Abi::ForEachWrittenRange(faultWords, [&](std::uint64_t begin, std::size_t bytes) {
            total += bytes;
            for (const auto address : {first, last}) covered += address >= begin && address < begin + bytes ? 1u : 0u;
        }), "a BDA dword write left an invalid written-page record");
        return std::pair{covered, total};
    };
    const auto write = [&](std::uint64_t address, std::uint32_t dwords, std::uint32_t storedBytes, Abi::FaultReason reason, std::uint32_t groups = 1) {
        const Abi::Header header{Abi::Version, static_cast<std::uint32_t>(ranges.size()), sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        for (auto* buffer : {&first, &second, &readOnly}) std::memset(buffer->Bytes().data(), 0xa5, buffer->Bytes().size());
        Pipeline pipeline(context, MakeBdaDwordWriteTestShader(address, dwords, values.data()), {&table, &fault, &output});
        pipeline.Run(groups);
        Abi::Fault report{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        for (std::uint32_t byte = 0; byte < dwords * 4u; ++byte) {
            const auto* stored = bytesAt(address + byte);
            const auto expected = byte < storedBytes ? static_cast<std::byte>(values[byte / 4u] >> ((byte % 4u) * 8u)) : std::byte{0xa5};
            Require(stored != nullptr && *stored == expected, "BDA dword write stored incorrect bytes at offset " + std::to_string(byte) + " of guest + " + std::to_string(address - guest));
        }
        Require(*bytesAt(address - 1) == std::byte{0xa5}, "BDA dword write touched the byte before it");
        if (address + dwords * 4u < guest + firstBytes + 32) Require(*bytesAt(address + dwords * 4u) == std::byte{0xa5}, "BDA dword write touched the byte after it");
        if (static_cast<std::uint32_t>(reason) == 0) {
            Require(report.state == Abi::FaultState::Empty, "a mapped BDA dword write published a fault");
        } else {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.instruction == 0x1234, "a BDA dword write did not publish its fault");
        }
        const auto [covered, total] = notedBytes(address, address + (storedBytes != 0 ? storedBytes - 1u : 0u));
        if (storedBytes == 0) {
            Require(total == 0, "a BDA dword write that stored nothing noted pages");
        } else {
            const auto pages = ((address + storedBytes - 1u) >> Abi::WrittenPageShift) - (address >> Abi::WrittenPageShift) + 1u;
            Require(covered == 2 && total == pages << Abi::WrittenPageShift, "a BDA dword write did not note exactly its written pages");
            const auto chunks = ((address + storedBytes - 1u) >> Abi::WrittenChunkShift) - (address >> Abi::WrittenChunkShift) + 1u;
            std::uint32_t claims = 0;
            std::memcpy(&claims, fault.Bytes().data() + Abi::WrittenCountWord * sizeof(std::uint32_t), sizeof(claims));
            Require(claims == chunks, "a BDA dword write claimed a written-page slot more than once per chunk");
        }
    };
    write(guest + 16, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 20, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 8, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 12, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 32, 3, 12, static_cast<Abi::FaultReason>(0));
    write(guest + 0x78, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 0x7c, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 41, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + firstBytes - 8, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + firstBytes + 12, 4, 4, Abi::FaultReason::Permission);
    write(guest + 0x7c, 2, 8, static_cast<Abi::FaultReason>(0), 64);
}

// What the direct table gives the bytes at `address`, walked as the shader walks it: the device
// address of the first, or 0, and whether they may be written.
static std::pair<std::uint64_t, bool> WalkDirectTable(std::span<const std::uint32_t> words, std::uint64_t address, std::uint32_t bytes) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Abi::DirectHeader header{};
    std::memcpy(&header, words.data(), sizeof(header));
    Require(header.version == Abi::DirectVersion && header.reserved == 0 && header.rootShift <= Abi::DirectMaxRootShift && header.rootCount <= header.entries && words.size() == sizeof(header) / 4u + header.entries * 4u + header.ranges * 8u, "BDA direct table: the header does not describe the table");
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

// The direct table (BdaAbi::DirectVersion): its tree on the CPU against the ranges it was built from
// (every root size, units split down to single bytes, mappings longer than a leaf can count, the
// ranges it refuses), then reads, stores and dword writes through the emitter's lookups on the
// direct table against the same shader made for the sorted ranges: outputs, the whole fault buffer
// and the data must be equal.
static void RunBdaDirectTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    constexpr std::uint64_t guest = 0x7fff12340000ULL;
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
    };
    const std::uint64_t device = 0x1000000000ULL;
    verify("one small range", {{guest, guest + 80, device, Abi::Read, 0}}, 14);
    verify("ranges that split their units down to bytes", {{guest + 1, guest + 4, device + 1, Abi::Read, 0}, {guest + 4, guest + 9, device + 0x100, Abi::Read | Abi::Write, 0}, {guest + 9, guest + 0x4003, device + 0x209, Abi::Read, 0}, {guest + 0x4003, guest + 0x4004, device + 0x8003, Abi::Read, 0}, {guest + 0x4080, guest + 0x9000, device + 0x9080, Abi::Read, 0}}, 14);
    verify("touching ranges of one distance", {{guest, guest + 0x4000, device, Abi::Read, 0}, {guest + 0x4000, guest + 0x8001, device + 0x4000, Abi::Read, 0}, {guest + 0x8001, guest + 0x8010, device + 0x8001, Abi::Read | Abi::Write, 0}}, 14);
    verify("ranges more than 1 GiB apart", {{guest + 0x123, guest + 0x40000, device, Abi::Read, 0}, {guest + 0x80000000ULL, guest + 0x80800000ULL, device + 0x100000, Abi::Read | Abi::Write, 0}, {guest + 0x80800000ULL, guest + 0x80800010ULL, device + 0x4000000, Abi::Read, 0}}, 22);
    verify("ranges more than 256 GiB apart", {{0x200000000ULL, 0x200004000ULL, device, Abi::Read, 0}, {0x4000000000ULL + 7, 0x4000000000ULL + 0x12345, device + 0x8000, Abi::Read, 0}, {guest, guest + 0x40000000ULL, device + 0x100000, Abi::Read | Abi::Write, 0}}, 30);
    verify("a range longer than a leaf counts", {{0x200000000ULL, 0x380000000ULL, device, Abi::Read, 0}, {0x380000000ULL, 0x380000100ULL, device * 2, Abi::Read, 0}}, 22);
    verify("a range in the first pages of the address space", {{0x1000, 0x2000, device, Abi::Read, 0}}, 14);
    Require(BdaDirectTable({}).size() == sizeof(Abi::DirectHeader) / 4u, "BDA direct table: the table of no range has entries");
    for (const auto& refused : {std::vector<Abi::Range>{{guest + 16, guest + 32, device, Abi::Read, 0}, {guest, guest + 16, device, Abi::Read, 0}}, std::vector<Abi::Range>{{guest, guest + 16, 0, Abi::Read, 0}}, std::vector<Abi::Range>{{guest, guest + 16, device, Abi::Write, 0}}, std::vector<Abi::Range>{{guest, guest + 16, ~std::uint64_t{0} - 8u, Abi::Read, 0}}}) {
        bool thrown = false;
        try {
            BdaDirectTable(refused);
        } catch (const std::runtime_error&) {
            thrown = true;
        }
        Require(thrown, "BDA direct table: an unsorted or invalid range was accepted");
    }

    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    Buffer writable(context, 32, usage);
    Buffer readOnly(context, 16, usage);
    Require(writable.DeviceAddress() % 4u == 0u && readOnly.DeviceAddress() % 4u == 0u, "BDA direct table: the test buffers are not dword-aligned");
    // [guest, +16) writable, [+16, +32) read-only and adjacent, a gap, [+67, +80) the writable
    // buffer's second half from an odd guest address at an odd distance.
    const std::vector<Abi::Range> ranges{
        {guest, guest + 16, writable.DeviceAddress(), Abi::Read | Abi::Write, 0},
        {guest + 16, guest + 32, readOnly.DeviceAddress(), Abi::Read, 0},
        {guest + 67, guest + 80, writable.DeviceAddress() + 16, Abi::Read | Abi::Write, 0}};
    const auto direct = BdaDirectTable(ranges);
    std::vector<std::uint32_t> searched((sizeof(Abi::Header) + ranges.size() * sizeof(Abi::Range)) / 4u);
    const Abi::Header header{Abi::Version, static_cast<std::uint32_t>(ranges.size()), sizeof(Abi::Range), 0};
    std::memcpy(searched.data(), &header, sizeof(header));
    std::memcpy(searched.data() + sizeof(header) / 4u, ranges.data(), ranges.size() * sizeof(Abi::Range));
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    struct Outcome {
        std::vector<std::byte> output;
        std::vector<std::byte> fault;
        std::vector<std::byte> data;
        bool operator==(const Outcome&) const = default;
    };
    const auto run = [&](bool onDirect, std::span<const std::uint32_t> words, const auto& make) {
        Buffer table(context, words.size() * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memcpy(table.Bytes().data(), words.data(), words.size() * 4u);
        for (std::size_t byte = 0; byte < writable.Bytes().size(); ++byte) writable.Bytes()[byte] = static_cast<std::byte>(byte + 1u);
        for (std::size_t byte = 0; byte < readOnly.Bytes().size(); ++byte) readOnly.Bytes()[byte] = static_cast<std::byte>(byte + 0x81u);
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        std::memset(output.Bytes().data(), 0xee, output.Bytes().size());
        SetBdaTestTable(onDirect);
        const auto code = make();
        SetBdaTestTable(false);
        Pipeline pipeline(context, code, {&table, &fault, &output});
        pipeline.Run(1);
        Outcome outcome{{output.Bytes().begin(), output.Bytes().end()}, {fault.Bytes().begin(), fault.Bytes().end()}, {writable.Bytes().begin(), writable.Bytes().end()}};
        outcome.data.insert(outcome.data.end(), readOnly.Bytes().begin(), readOnly.Bytes().end());
        return outcome;
    };
    std::size_t faults = 0;
    std::size_t stored = 0;
    const auto compare = [&](const std::string& name, std::uint64_t address, const auto& make) {
        const auto wanted = run(false, searched, make);
        const auto found = run(true, direct, make);
        Require(found == wanted, "BDA direct table: " + name + " at guest + " + std::to_string(static_cast<std::int64_t>(address - guest)) + " differs from the search of the sorted ranges");
        Abi::Fault report{};
        std::memcpy(&report, found.fault.data(), sizeof(report));
        faults += report.state == Abi::FaultState::Ready;
        stored += std::find(found.data.begin(), found.data.end(), std::byte{0xd4}) != found.data.end();
    };
    const std::array<std::uint32_t, 4> values{0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
    for (const std::int64_t offset : {-1, 0, 1, 13, 15, 16, 29, 31, 32, 64, 66, 67, 68, 77, 79, 80, 0x40000000}) {
        const auto address = guest + static_cast<std::uint64_t>(offset);
        for (const std::uint32_t bits : {8u, 16u, 32u}) {
            compare(std::to_string(bits) + "-bit read", address, [&] { return MakeBdaTestShader(address, bits); });
            for (const bool runtime : {false, true}) compare(std::to_string(bits) + "-bit store", address, [&] { return runtime ? MakeBdaWriteTestShader(address, 0xa1b2c3d4u, bits, false) : MakeBdaStoreTestShader(address, 0xa1b2c3d4u, bits, false); });
        }
        for (const bool stops : {true, false}) {
            for (const bool coherent : {false, true}) {
                for (const std::uint32_t dwords : {1u, 2u, 4u}) compare(std::to_string(dwords) + " dwords read", address, [&] { return MakeBdaDwordReadTestShader(address, dwords, coherent, stops); });
                compare("a span read", address, [&] { return MakeBdaSpanReadTestShader(address - 8u, 8u, 0b1010u, coherent, stops); });
            }
        }
        for (const std::uint32_t dwords : {2u, 3u, 4u}) compare(std::to_string(dwords) + " dwords written", address, [&] { return MakeBdaDwordWriteTestShader(address, dwords, values.data()); });
    }
    compare("a negative dword offset", guest + 4u, [&] { return MakeBdaDwordReadTestShader(guest + 4u, 1u, false, true, static_cast<std::uint32_t>(-4)); });
    compare("a positive dword offset", guest, [&] { return MakeBdaDwordReadTestShader(guest, 1u, true, false, 4u); });
    compare("an offset across a range seam", guest, [&] { return MakeBdaDwordReadTestShader(guest, 1u, true, false, 13u); });
    compare("a dword offset underflow", 2u, [&] { return MakeBdaDwordReadTestShader(2u, 1u, true, false, static_cast<std::uint32_t>(-4)); });
    compare("a dword offset overflow", UINT64_MAX - 1u, [&] { return MakeBdaDwordReadTestShader(UINT64_MAX - 1u, 1u, false, true, 4u); });
    compare("a dword extent overflow", UINT64_MAX - 2u, [&] { return MakeBdaDwordReadTestShader(UINT64_MAX - 2u, 1u, false, false); });
    Require(faults != 0 && stored != 0, "BDA direct table: the comparison with the search met no fault or no store");
    std::vector<std::uint32_t> invalid = direct;
    invalid[0] = Abi::Version;
    for (const bool store : {false, true}) {
        const auto outcome = run(true, invalid, [&] { return store ? MakeBdaStoreTestShader(guest, 0xa1b2c3d4u, 32, false) : MakeBdaTestShader(guest, 32); });
        Abi::Fault report{};
        std::memcpy(&report, outcome.fault.data(), sizeof(report));
        Require(report.state == Abi::FaultState::Ready && report.reason == Abi::FaultReason::InvalidTable && outcome.data[0] == std::byte{1}, "BDA direct table: a header that is not a direct table's was not refused");
    }
}

void RunBdaExecutionTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    RunBdaDirectTests(context);
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
    Buffer wide(context, 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    for (std::uint32_t dword = 0; dword < 8u; ++dword) {
        const std::uint32_t value = 0xa0b0c000u + dword;
        std::memcpy(wide.Bytes().data() + dword * 4u, &value, sizeof(value));
    }
    constexpr std::uint64_t spanGuest = 0x7fff56780000ULL;
    const std::array<Abi::Range, 3> spanRanges{{ranges[0], ranges[1], {spanGuest, spanGuest + 32, wide.DeviceAddress(), Abi::Read, 0}}};
    Buffer spanTable(context, sizeof(Abi::Header) + sizeof(spanRanges), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto span = [&](std::uint64_t address, std::int32_t offset, std::uint32_t extracted, bool coherent, bool stops, std::array<std::uint32_t, 4> values, Abi::FaultReason reason = {}, std::uint64_t faultAddress = 0, std::uint32_t faultBytes = 0) {
        const Abi::Header header{Abi::Version, 3, sizeof(Abi::Range), 0};
        std::memcpy(spanTable.Bytes().data(), &header, sizeof(header));
        std::memcpy(spanTable.Bytes().data() + sizeof(header), spanRanges.data(), sizeof(spanRanges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::array<std::uint32_t, 4> sentinel{0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
        std::memcpy(words.Bytes().data(), sentinel.data(), sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaSpanReadTestShader(address, static_cast<std::uint32_t>(offset), extracted, coherent, stops), {&spanTable, &fault, &words});
        pipeline.Run(1);
        Abi::Fault report{};
        std::array<std::uint32_t, 4> result{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(result.data(), words.Bytes().data(), sizeof(result));
        const bool faults = static_cast<std::uint32_t>(reason) != 0u;
        if (faults) {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.address == faultAddress && report.bytes == faultBytes && report.instruction == 0x1234, "BDA span read did not publish its first fault");
        } else {
            Require(report.state == Abi::FaultState::Empty, "mapped BDA span read published a fault");
        }
        Require(result == (faults && stops ? sentinel : values), faults && stops ? "faulting BDA span read continued to output a substitute value" : "BDA span read produced incorrect data");
    };
    const auto at = [&](std::uint32_t dword) { return 0xa0b0c000u + dword; };
    for (const bool stops : {true, false}) {
        for (const bool coherent : {false, true}) {
            span(spanGuest + 16, -12, 0b1010u, coherent, stops, {0, at(2), 0, at(4)});
            span(spanGuest, 4, 0b0101u, coherent, stops, {at(1), 0, at(3), 0});
            span(spanGuest + 24, 0, 0b0011u, coherent, stops, {at(6), at(7), 0, 0});
            span(spanGuest + 20, 8, 0b0001u, coherent, stops, {at(7), 0, 0, 0});
            span(spanGuest + 28, 0, 0b0101u, coherent, stops, {at(7), 0, 0, 0}, Abi::FaultReason::Unmapped, spanGuest + 36, 1);
            span(guest, 1, 0b1001u, coherent, stops, {0x55443322, 0, 0, 0}, Abi::FaultReason::Unmapped, guest + 13, 1);
            span(4, -8, 0b0001u, coherent, stops, {0, 0, 0, 0}, Abi::FaultReason::Overflow, 4, 0);
        }
    }
    ranges[0].permissions = 0;
    run(guest, 8, 0, Abi::FaultReason::Permission);
    RunBdaDwordWriteTests(context);
}

void RunBdaStoreExecutionTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Buffer first(context, 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer second(context, 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer table(context, sizeof(Abi::Header) + 2 * sizeof(Abi::Range), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    constexpr std::uint64_t guest = 0x7fff1235fffdULL;
    constexpr std::uint32_t writable = Abi::Read | Abi::Write;
    const Abi::Header header{Abi::Version, 2, sizeof(Abi::Range), 0};
    std::array<Abi::Range, 2> ranges{};
    bool runtime = false;
    const auto run = [&](const char* name, std::uint64_t address, std::uint32_t bits, bool coherent, const std::array<std::uint8_t, 5>& expected, std::initializer_list<std::uint64_t> writtenAddresses, Abi::FaultReason reason = {}, std::uint64_t faultAddress = 0, std::uint32_t faultBytes = 0) {
        const std::string label = std::string(runtime ? "BDA runtime write " : "BDA store ") + name + (coherent ? " coherent: " : ": ");
        std::memset(first.Bytes().data(), 0xcd, first.Bytes().size());
        std::memset(second.Bytes().data(), 0xcd, second.Bytes().size());
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        std::memset(output.Bytes().data(), 0, output.Bytes().size());
        Pipeline pipeline(context, runtime ? MakeBdaWriteTestShader(address, 0xa1b2c3d4u, bits, coherent) : MakeBdaStoreTestShader(address, 0xa1b2c3d4u, bits, coherent), {&table, &fault, &output});
        pipeline.Run(1);
        std::array<std::uint8_t, 5> actual{};
        std::memcpy(actual.data(), first.Bytes().data(), 2);
        std::memcpy(actual.data() + 2, second.Bytes().data(), 3);
        Require(actual == expected, label + "incorrect stored bytes or writes outside the mapped span");
        Abi::Fault report{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        if (reason == Abi::FaultReason{}) {
            Require(report.state == Abi::FaultState::Empty && report.reason == Abi::FaultReason{} && report.address == 0 && report.bytes == 0 && report.stage == 0 && report.instruction == 0 && report.reserved == 0, label + "mapped store published a fault");
        } else {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.address == faultAddress && report.bytes == faultBytes && report.stage == static_cast<std::uint32_t>(ShaderRecompiler::IrShaderStage::Compute) && report.instruction == 0x1234 && report.reserved == 0, label + "incorrect first fault metadata");
        }
        std::array<std::uint32_t, Abi::FaultBufferBytes / sizeof(std::uint32_t)> words{};
        std::memcpy(words.data(), fault.Bytes().data(), sizeof(words));
        Require(words[Abi::WrittenOverflowWord] == 0, label + "written-page table overflowed");
        std::set<std::uint64_t> recordedPages;
        Require(Abi::ForEachWrittenRange(words.data(), [&](std::uint64_t address, std::size_t bytes) {
            for (std::size_t offset = 0; offset < bytes; offset += std::size_t{1} << Abi::WrittenPageShift) {
                Require(recordedPages.insert((address + offset) >> Abi::WrittenPageShift).second, label + "written-page table contains duplicate pages");
            }
        }), label + "invalid written-page record");
        std::set<std::uint64_t> expectedPages;
        std::set<std::uint32_t> expectedChunks;
        for (const auto writtenAddress : writtenAddresses) {
            expectedPages.insert(writtenAddress >> Abi::WrittenPageShift);
            expectedChunks.insert(Abi::WrittenChunkKey(writtenAddress));
        }
        Require(recordedPages == expectedPages, label + "incorrect written-page set");
        Require(words[Abi::WrittenCountWord] == expectedChunks.size(), label + "incorrect written-chunk claim count");
        std::uint32_t completed = 0;
        std::memcpy(&completed, output.Bytes().data(), sizeof(completed));
        Require(completed == 1, label + "store did not finish its per-byte accesses");
    };
    for (const auto& [coherent, writes] : std::initializer_list<std::pair<bool, bool>>{{false, false}, {false, true}, {true, false}, {true, true}}) {
        runtime = writes;
        ranges = {{{guest, guest + 2, first.DeviceAddress(), writable, 0}, {guest + 2, guest + 5, second.DeviceAddress(), writable, 0}}};
        run("byte", guest + 1, 8, coherent, {0xcd, 0xd4, 0xcd, 0xcd, 0xcd}, {guest + 1});
        run("split short", guest + 1, 16, coherent, {0xcd, 0xd4, 0xc3, 0xcd, 0xcd}, {guest + 1, guest + 2});
        run("split unaligned dword", guest, 32, coherent, {0xd4, 0xc3, 0xb2, 0xa1, 0xcd}, {guest, guest + 3});
        ranges[1].permissions = Abi::Read;
        run("read-only byte", guest + 2, 8, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {}, Abi::FaultReason::Permission, guest + 2, 1);
        run("read-only short suffix", guest + 1, 16, coherent, {0xcd, 0xd4, 0xcd, 0xcd, 0xcd}, {guest + 1}, Abi::FaultReason::Permission, guest + 2, 1);
        run("read-only suffix", guest, 32, coherent, {0xd4, 0xc3, 0xcd, 0xcd, 0xcd}, {guest}, Abi::FaultReason::Permission, guest + 2, 1);
        ranges[0].end = guest + 1;
        ranges[1].permissions = writable;
        run("unmapped middle byte", guest, 32, coherent, {0xd4, 0xcd, 0xb2, 0xa1, 0xcd}, {guest, guest + 3}, Abi::FaultReason::Unmapped, guest + 1, 1);
        ranges = {{{guest + 3, guest + 5, first.DeviceAddress(), writable, 0}, {guest + 5, guest + 8, second.DeviceAddress(), writable, 0}}};
        run("split aligned dword", guest + 3, 32, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {}, Abi::FaultReason::Unmapped, guest + 3, 4);
    }
}
