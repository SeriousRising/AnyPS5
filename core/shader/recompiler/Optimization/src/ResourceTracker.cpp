#include "Optimization/ResourceTracker.hpp"
#include "Optimization/SrtWalker.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "IntermediateRepresentation/IrBuilder.hpp"
#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ShaderRecompiler {
namespace {

constexpr std::uint32_t samplerBorderClampMask = (1u << 2u) | (1u << 5u) | (1u << 8u);
constexpr std::uint32_t samplerDword3ReservedMask = 0x3ffff000u;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

// Debug aid: APS5_TRACE_BDA=1 names the accesses that make a program address-based (see Collect).
bool bdaTraceEnabled() {
    static const bool enabled = std::getenv("APS5_TRACE_BDA") != nullptr;
    return enabled;
}
constexpr unsigned bdaTraceLimit = 8;

std::string formatHex32(std::uint32_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string hex(8u, '0');
    for (std::uint32_t index = 0; index < 8u; index++) {
        hex[7u - index] = digits[(value >> (index * 4u)) & 0xfu];
    }
    return hex;
}

std::string describeValueChain(const IrValue* value, std::uint32_t depth) {
    value = value->Resolve();
    if (value->HasImmediate()) {
        return "Immediate";
    }
    std::string text = std::string(IrOpcodeName(value->Opcode()));
    if (value->Opcode() == IrOpcode::LoadAddressU32 || value->Opcode() == IrOpcode::ReadConstBuffer) {
        text += "@pc=0x" + formatHex32(value->Flags<MemoryFlags>().pc);
    }
    if (depth == 0u || value->ArgumentCount() == 0u) {
        return text;
    }
    text += "(";
    for (std::uint32_t index = 0; index < value->ArgumentCount(); index++) {
        if (index != 0u) {
            text += ", ";
        }
        text += describeValueChain(value->Argument(index), depth - 1u);
    }
    text += ")";
    return text;
}

std::uint32_t possibleU32Bits(const IrValue* value) {
    value = value->Resolve();
    if (value->HasImmediate()) {
        return value->Type() == IrType::U32 ? value->ImmediateU32() : std::numeric_limits<std::uint32_t>::max();
    }
    switch (value->Opcode()) {
        case IrOpcode::BitwiseAnd32:
            return possibleU32Bits(value->Argument(0)) & possibleU32Bits(value->Argument(1));
        case IrOpcode::BitwiseOr32:
            return possibleU32Bits(value->Argument(0)) | possibleU32Bits(value->Argument(1));
        case IrOpcode::ShiftLeftLogical32: {
            const IrValue* shift = value->Argument(1)->Resolve();
            return shift->HasImmediate() && shift->Type() == IrType::U32 ? possibleU32Bits(value->Argument(0)) << (shift->ImmediateU32() & 31u) : std::numeric_limits<std::uint32_t>::max();
        }
        default:
            return std::numeric_limits<std::uint32_t>::max();
    }
}

std::uint32_t byteExtent(const MemoryInfo& memory) {
    const auto bytes = std::max((memory.dataBits + 7u) / 8u, 1u);
    const auto count = std::max(memory.dataDwords, 1u);
    const auto end = static_cast<std::uint64_t>(memory.offset) + static_cast<std::uint64_t>(bytes) * count;
    return end > std::numeric_limits<std::uint32_t>::max() ? std::numeric_limits<std::uint32_t>::max() : static_cast<std::uint32_t>(end);
}

class Tracker {
public:
    explicit Tracker(IrProgram& program) : m_program(program), m_info(program.Resources().info), m_builder(program) {
        m_info.buffers.clear();
        m_info.images.clear();
        m_info.samplers.clear();
        m_info.sampledPairs.clear();
        m_info.usesDma = false;
        m_info.bdaWrites = false;
    }

    void Run() {
        if (m_program.Resources().resourceTrackingComplete) {
            fail("resources already tracked");
        }
        if (!m_program.Resources().srtPlanComplete) {
            fail("SRT plan is not ready");
        }
        SplitDescriptorPhis();
        PlanIndirectImages();
        for (auto& block : m_program.Blocks()) {
            for (IrValue* inst : block->Instructions()) {
                Collect(*inst);
            }
        }
        if (m_bdaTraces > bdaTraceLimit) std::fprintf(stderr, "[bda] %u more address accesses in this program not shown\n", m_bdaTraces - bdaTraceLimit);
        LinkImageAliases();
        for (const auto& patch : m_handlePatches) {
            patch.handle->SetFlags<std::uint32_t>(patch.resource);
        }
        for (const auto& patch : m_memoryPatches) {
            auto& memory = m_program.Resources().memoryInfo[patch.index];
            memory.resource = patch.resource;
            if (patch.hasSampler) {
                memory.sampler = patch.sampler;
            }
        }
        for (const auto& plan : m_indirectImages) {
            plan.handle->ReplaceArgument(0, plan.key);
            for (std::uint32_t dword = 0; dword < 4u; dword++) {
                plan.handle->ReplaceArgument(dword + 1u, plan.roots[dword]);
            }
            for (std::uint32_t dword = 5u; dword < plan.roots.size(); dword++) {
                plan.handle->ReplaceArgument(dword, plan.key);
            }
            for (const auto index : plan.memory) {
                m_program.Resources().memoryInfo[index].planningOnly = true;
            }
        }
        std::erase_if(m_program.Metadata().dynamicReads, [&](IrValue* value) {
            const IrValue* inst = value->Resolve();
            return std::ranges::any_of(m_indirectImages, [&](const IndirectImagePlan& plan) {
                return std::ranges::find(plan.reads, inst) != plan.reads.end();
            });
        });
        m_program.Resources().descriptorSources = std::move(m_sources);
        m_program.Resources().info = std::move(m_info);
        m_program.Resources().resourceTrackingComplete = true;
    }

private:
    struct HandlePatch {
        IrValue* handle = nullptr;
        std::uint32_t resource = 0;
    };

    struct MemoryPatch {
        std::uint32_t index = 0;
        std::uint32_t resource = 0;
        std::uint32_t sampler = 0;
        bool hasSampler = false;
    };

    struct IndirectImagePlan {
        IrValue* handle = nullptr;
        std::uint32_t source = 0;
        IrValue* key = nullptr;
        std::array<IrValue*, 8> roots {};
        std::array<std::uint32_t, 8> memory {};
        std::array<const IrValue*, 8> reads {};
    };

    void MakeSource(const IrValue& handle, std::uint32_t width, bool sampler, bool sampleAdjust, DescriptorSource& descriptor) {
        if (handle.ArgumentCount() != width) {
            fail(std::string(IrOpcodeName(handle.Opcode())) + " has " + std::to_string(handle.ArgumentCount()) + " descriptor dwords, expected " + std::to_string(width));
        }
        descriptor.dwordCount = width;
        for (std::uint32_t i = 0; i < width; i++) {
            descriptor.dwords[i] = handle.Argument(i)->Resolve();
        }
        if (sampleAdjust) {
            descriptor.dwords[3] = canonicalizeSampleAdjustDword3(descriptor.dwords[3]);
        }
        const IrValue* dword0 = descriptor.dwords[0]->Resolve();
        if (sampler && dword0->HasImmediate() && dword0->Type() == IrType::U32 && (dword0->ImmediateU32() & samplerBorderClampMask) == 0u) {
            descriptor.dwords[3] = &m_builder.Constant(0u);
        }
    }

    IrValue* canonicalizeSampleAdjustDword3(IrValue* value) {
        for (;;) {
            value = value->Resolve();
            if (value->Opcode() != IrOpcode::BitwiseOr32) {
                return value;
            }
            IrValue* left = value->Argument(0)->Resolve();
            IrValue* right = value->Argument(1)->Resolve();
            const bool leftReserved = (possibleU32Bits(left) & ~samplerDword3ReservedMask) == 0u;
            const bool rightReserved = (possibleU32Bits(right) & ~samplerDword3ReservedMask) == 0u;
            if (leftReserved && rightReserved) {
                return &m_builder.Constant(0u);
            }
            if (leftReserved) {
                value = right;
            } else if (rightReserved) {
                value = left;
            } else {
                return value;
            }
        }
    }

    bool ValidateSource(const DescriptorSource& descriptor, std::uint32_t& badDword) const {
        constexpr SrtWalker walker;
        for (std::uint32_t i = 0; i < descriptor.dwordCount; i++) {
            badDword = i;
            if (descriptor.dwords[i]->Resolve()->Type() != IrType::U32) {
                return false;
            }
            if (!walker.ValidateRuntimeValue(m_program.Resources(), descriptor.dwords[i])) {
                return false;
            }
        }
        return true;
    }

    std::uint32_t InternSource(const DescriptorSource& descriptor) {
        for (std::uint32_t candidate = 0; candidate < m_sources.size(); candidate++) {
            const auto& current = m_sources[candidate];
            if (current.dwordCount != descriptor.dwordCount || current.indirectImage != descriptor.indirectImage) {
                continue;
            }
            bool same = true;
            for (std::uint32_t i = 0; i < descriptor.dwordCount; i++) {
                same = same && EquivalentValue(m_program.Resources(), current.dwords[i], descriptor.dwords[i]);
            }
            if (same) {
                return candidate;
            }
        }
        m_sources.push_back(descriptor);
        return static_cast<std::uint32_t>(m_sources.size() - 1);
    }

    static bool immediateU32(IrValue* value, std::uint32_t& result) {
        value = value->Resolve();
        if (!value->HasImmediate() || value->Type() != IrType::U32) {
            return false;
        }
        result = value->ImmediateU32();
        return true;
    }

    static bool usesOnly(const IrValue& value, std::span<const IrValue* const> users) {
        return !value.Uses().empty() && std::ranges::all_of(value.Uses(), [&](const IrValue* user) {
            return std::ranges::find(users, user) != users.end();
        });
    }

    const MemoryInfo* ScalarReadMemory(const IrValue& read, std::uint32_t& index) const {
        if (read.Opcode() != IrOpcode::ReadConstBuffer || read.ArgumentCount() != 2u) {
            return nullptr;
        }
        index = read.Flags<MemoryFlags>().index;
        if (index >= m_program.Resources().memoryInfo.size()) {
            return nullptr;
        }
        const auto& memory = m_program.Resources().memoryInfo[index];
        return memory.kind == ResourceKind::ScalarBuffer && memory.dataBits == 32u && memory.dataDwords == 1u ? &memory : nullptr;
    }

    bool MemoryIndexBelongsTo(std::uint32_t index, const IrValue& owner) const {
        for (const auto& block : m_program.Blocks()) {
            for (const IrValue* inst : block->Instructions()) {
                const auto op = inst->Opcode();
                if ((BufferAccessOf(op) == BufferAccess::None && AddressOpcodeInfoOf(op).access == AddressAccess::None && ImageOpcodeInfoOf(op).access == ImageAccess::None) || inst == &owner) {
                    continue;
                }
                if (inst->Flags<MemoryFlags>().index == index) {
                    return false;
                }
            }
        }
        return true;
    }

    bool MakeRuntimeBufferSource(const IrValue& handle, std::uint32_t& source, DescriptorSource& descriptor) {
        if (handle.Opcode() != IrOpcode::GetBufferResource) {
            return false;
        }
        MakeSource(handle, 4u, false, false, descriptor);
        std::uint32_t badDword = 0;
        if (!ValidateSource(descriptor, badDword)) {
            return false;
        }
        source = InternSource(descriptor);
        return true;
    }

    bool MatchMaterialOffset(IrValue* value, IrValue*& selector, std::uint32_t& stride, std::uint32_t& offset) const {
        value = value->Resolve();
        offset = 0;
        IrValue* candidate = value;
        if (candidate->Opcode() == IrOpcode::IAdd32 && candidate->ArgumentCount() == 2u) {
            std::uint32_t immediate = 0;
            if (immediateU32(candidate->Argument(0), immediate)) {
                value = candidate->Argument(1)->Resolve();
            } else if (immediateU32(candidate->Argument(1), immediate)) {
                value = candidate->Argument(0)->Resolve();
            } else {
                return false;
            }
            offset = immediate;
        }
        IrValue* multiply = value;
        if (multiply->Opcode() != IrOpcode::IMul32 || multiply->ArgumentCount() != 2u) {
            return false;
        }
        if (immediateU32(multiply->Argument(0), stride)) {
            selector = multiply->Argument(1)->Resolve();
        } else if (immediateU32(multiply->Argument(1), stride)) {
            selector = multiply->Argument(0)->Resolve();
        } else {
            return false;
        }
        return stride != 0u && selector->Opcode() == IrOpcode::ReadFirstLane;
    }

    // The table entry offset `key * 32` (a shift or a multiply), optionally plus an immediate.
    static bool MatchTableOffset(IrValue* value, IrValue*& key, std::uint32_t& entryOffset) {
        value = value->Resolve();
        entryOffset = 0;
        if (value->Opcode() == IrOpcode::IAdd32 && value->ArgumentCount() == 2u) {
            std::uint32_t immediate = 0;
            if (immediateU32(value->Argument(0), immediate)) {
                value = value->Argument(1)->Resolve();
            } else if (immediateU32(value->Argument(1), immediate)) {
                value = value->Argument(0)->Resolve();
            } else {
                return false;
            }
            entryOffset = immediate;
        }
        if (value->ArgumentCount() != 2u) {
            return false;
        }
        std::uint32_t scale = 0;
        if (value->Opcode() == IrOpcode::ShiftLeftLogical32) {
            if (!immediateU32(value->Argument(1), scale) || scale != 5u) {
                return false;
            }
            key = value->Argument(0)->Resolve();
            return true;
        }
        if (value->Opcode() != IrOpcode::IMul32) {
            return false;
        }
        if (immediateU32(value->Argument(1), scale)) {
            key = value->Argument(0)->Resolve();
        } else if (immediateU32(value->Argument(0), scale)) {
            key = value->Argument(1)->Resolve();
        } else {
            return false;
        }
        return scale == 32u;
    }

    // A T# whose eight dwords are scalar reads of one V# (the table) at `entryOffset + key * 32`
    // with a wave-uniform runtime key: a bindless image table. The key stays an ordinary value
    // (the SPIR-V selects the bound slot from it); the eight reads become planning-only.
    bool TryMakeTableImage(IrValue& handle, IndirectImagePlan& plan) {
        if (handle.Opcode() != IrOpcode::GetImageResource || handle.ArgumentCount() != 8u) {
            return false;
        }

        std::array<IrValue*, 8> heapReads {};
        IrValue* heapHandle = nullptr;
        IrValue* heapOffset = nullptr;
        std::uint32_t immediateOffset = 0;
        for (std::uint32_t dword = 0; dword < heapReads.size(); dword++) {
            heapReads[dword] = handle.Argument(dword)->Resolve();
            std::uint32_t memoryIndex = 0;
            const MemoryInfo* memory = ScalarReadMemory(*heapReads[dword], memoryIndex);
            if (memory == nullptr || !MemoryIndexBelongsTo(memoryIndex, *heapReads[dword])) {
                return false;
            }
            if (dword == 0u) {
                immediateOffset = memory->offset;
            } else if (memory->offset != immediateOffset + dword * sizeof(std::uint32_t)) {
                return false;
            }
            IrValue* currentHandle = heapReads[dword]->Argument(0)->Resolve();
            if (heapHandle != nullptr && currentHandle != heapHandle) {
                return false;
            }
            heapHandle = currentHandle;
            if (dword == 0u) {
                heapOffset = heapReads[dword]->Argument(1)->Resolve();
            } else if (!EquivalentValue(m_program.Resources(), heapOffset, heapReads[dword]->Argument(1))) {
                return false;
            }
            plan.memory[dword] = memoryIndex;
            plan.reads[dword] = heapReads[dword];
        }

        IrValue* key = nullptr;
        std::uint32_t entryOffset = 0;
        if (!MatchTableOffset(heapOffset, key, entryOffset) || key->Type() != IrType::U32) {
            return false;
        }
        entryOffset += immediateOffset;

        const std::array<const IrValue*, 1> imageUsers {&handle};
        for (const auto* read : heapReads) {
            if (!usesOnly(*read, imageUsers)) {
                return false;
            }
        }

        DescriptorSource heapSource;
        std::uint32_t heapSourceIndex = 0;
        if (!MakeRuntimeBufferSource(*heapHandle, heapSourceIndex, heapSource)) {
            return false;
        }

        DescriptorSource::IndirectImage table;
        table.heapSource = heapSourceIndex;
        table.materialSource = heapSourceIndex;
        table.entryOffset = entryOffset;
        DescriptorSource materialSource = heapSource;
        std::uint32_t materialMemoryIndex = 0;
        const MemoryInfo* materialMemory = ScalarReadMemory(*key, materialMemoryIndex);
        if (materialMemory != nullptr && MemoryIndexBelongsTo(materialMemoryIndex, *key)) {
            IrValue* selector = nullptr;
            std::uint32_t selectorStride = 0;
            std::uint32_t selectorOffset = 0;
            DescriptorSource candidateSource;
            std::uint32_t candidateIndex = 0;
            if (MatchMaterialOffset(key->Argument(1), selector, selectorStride, selectorOffset) && MakeRuntimeBufferSource(*key->Argument(0)->Resolve(), candidateIndex, candidateSource)) {
                table.hasMaterial = true;
                table.materialSource = candidateIndex;
                table.selectorStride = selectorStride;
                table.selectorOffset = selectorOffset + materialMemory->offset;
                materialSource = candidateSource;
            }
        }

        DescriptorSource imageSource;
        imageSource.dwordCount = 8u;
        std::copy(heapSource.dwords.begin(), heapSource.dwords.begin() + 4u, imageSource.dwords.begin());
        std::copy(materialSource.dwords.begin(), materialSource.dwords.begin() + 4u, imageSource.dwords.begin() + 4u);
        imageSource.indirectImage = table;

        plan.handle = &handle;
        plan.source = InternSource(imageSource);
        plan.key = key;
        plan.roots = imageSource.dwords;
        return true;
    }

    const IndirectImagePlan* FindIndirectImage(const IrValue& handle) const {
        const auto found = std::ranges::find_if(m_indirectImages, [&](const IndirectImagePlan& plan) {
            return plan.handle == &handle;
        });
        return found == m_indirectImages.end() ? nullptr : &*found;
    }

    bool IsIndirectPlanningMemory(std::uint32_t index) const {
        return std::ranges::any_of(m_indirectImages, [&](const IndirectImagePlan& plan) {
            return std::ranges::find(plan.memory, index) != plan.memory.end();
        });
    }

    static constexpr std::uint32_t phiSearchDepth = 8u;
    static constexpr std::size_t maxDescriptorPaths = 64u;

    static bool SplittableImageRead(const IrValue& inst) {
        switch (inst.Opcode()) {
            case IrOpcode::ImageSampleRaw:
            case IrOpcode::ImageGatherRaw:
            case IrOpcode::ImageQueryLod:
            case IrOpcode::ImageQueryDimensions:
            case IrOpcode::ImageRead: return inst.Type() == IrType::U32x4;
            default: return false;
        }
    }

    void CollectDescriptorPhis(IrValue* value, std::vector<IrValue*>& phis, std::uint32_t depth) const {
        value = value->Resolve();
        if (value->HasImmediate()) {
            return;
        }
        if (value->IsPhi()) {
            if (ResolveInvariantPhi(m_program.Resources(), value) == nullptr && std::ranges::find(phis, value) == phis.end()) {
                phis.push_back(value);
            }
            return;
        }
        if (depth == 0u || !Detail::IsRuntimeUniformOp(value->Opcode())) {
            return;
        }
        for (std::size_t index = 0; index < value->ArgumentCount(); index++) {
            CollectDescriptorPhis(value->Argument(index), phis, depth - 1u);
        }
    }

    static IrValue* PhiIncoming(const IrValue& phi, const IrBlock* predecessor) {
        for (std::size_t index = 0; index < phi.ArgumentCount(); index++) {
            if (phi.PhiBlock(index) == predecessor) {
                return phi.Argument(index);
            }
        }
        return nullptr;
    }

    IrValue* EmitBefore(IrValue& position, IrOpcode op, IrType type, std::span<IrValue* const> arguments, std::uint64_t flags = 0) {
        IrValue& value = m_program.CreateValue(op, type, flags);
        for (IrValue* argument : arguments) {
            value.AddArgument(argument);
        }
        position.Parent()->InsertInstructionBefore(&position, &value);
        return &value;
    }

    IrValue* EmitCopy(const IrValue& original, std::span<IrValue* const> arguments, IrValue& position) {
        return EmitBefore(position, original.Opcode(), original.Type(), arguments, original.Flags<std::uint64_t>());
    }

    struct PhiChoice {
        IrBlock* block = nullptr;
        std::size_t edge = 0;
        std::int32_t parent = -1;
    };
    using PhiPath = std::vector<PhiChoice>;

    static std::int32_t ChoiceOf(const PhiPath& path, const IrBlock* block) {
        for (std::size_t index = 0; index < path.size(); index++) {
            if (path[index].block == block) {
                return static_cast<std::int32_t>(index);
            }
        }
        return -1;
    }

    static IrValue* ChosenIncoming(const IrValue& phi, const PhiChoice& choice) {
        return PhiIncoming(phi, choice.block->Predecessors()[choice.edge]);
    }

    const char* FindPendingPhi(IrValue* value, const PhiPath& path, std::int32_t parent, std::vector<const IrBlock*>& crossed, std::uint32_t depth, IrValue*& pending, std::int32_t& pendingParent) const {
        value = value->Resolve();
        if (pending != nullptr || value->HasImmediate()) {
            return nullptr;
        }
        if (value->IsPhi()) {
            if (ResolveInvariantPhi(m_program.Resources(), value) != nullptr) {
                return nullptr;
            }
            const IrBlock* block = value->Parent();
            if (block == nullptr || block->Predecessors().size() < 2u || value->PhiBlockCount() != block->Predecessors().size()) {
                return "a descriptor phi does not have one operand for each predecessor of its block";
            }
            const auto choice = ChoiceOf(path, block);
            if (choice < 0) {
                pending = value;
                pendingParent = parent;
                return nullptr;
            }
            if (std::ranges::find(crossed, block) != crossed.end()) {
                return "a descriptor phi is carried around a loop";
            }
            if (path[choice].parent != parent) {
                return "the phis of one block are reached both directly and through another phi, or through different phis";
            }
            IrValue* incoming = ChosenIncoming(*value, path[choice]);
            if (incoming == nullptr) {
                return "a descriptor phi does not have one operand for each predecessor of its block";
            }
            crossed.push_back(block);
            const char* refusal = FindPendingPhi(incoming, path, choice, crossed, phiSearchDepth, pending, pendingParent);
            crossed.pop_back();
            return refusal;
        }
        if (depth == 0u || !Detail::IsRuntimeUniformOp(value->Opcode())) {
            return nullptr;
        }
        for (std::size_t index = 0; index < value->ArgumentCount(); index++) {
            if (const char* refusal = FindPendingPhi(value->Argument(index), path, parent, crossed, depth - 1u, pending, pendingParent)) {
                return refusal;
            }
        }
        return nullptr;
    }

    const char* EnumeratePhiPaths(std::span<IrValue* const> handles, PhiPath& path, std::vector<PhiPath>& paths) const {
        IrValue* pending = nullptr;
        std::int32_t parent = -1;
        std::vector<const IrBlock*> crossed;
        for (IrValue* handle : handles) {
            for (std::size_t dword = 0; dword < handle->ArgumentCount(); dword++) {
                if (const char* refusal = FindPendingPhi(handle->Argument(dword), path, -1, crossed, phiSearchDepth, pending, parent)) {
                    return refusal;
                }
            }
        }
        if (pending == nullptr) {
            if (paths.size() == maxDescriptorPaths) {
                return "the descriptor is selected along more than 64 control-flow paths";
            }
            paths.push_back(path);
            return nullptr;
        }
        IrBlock* block = pending->Parent();
        for (std::size_t edge = 0; edge < block->Predecessors().size(); edge++) {
            path.push_back({block, edge, parent});
            const char* refusal = EnumeratePhiPaths(handles, path, paths);
            path.pop_back();
            if (refusal != nullptr) {
                return refusal;
            }
        }
        return nullptr;
    }

    IrValue* SubstitutePath(IrValue* value, const PhiPath& path, IrValue& position, bool emit, bool crossed, std::uint32_t depth) {
        value = value->Resolve();
        if (value->HasImmediate()) {
            return value;
        }
        if (value->IsPhi()) {
            if (IrValue* invariant = ResolveInvariantPhi(m_program.Resources(), value)) {
                if (!crossed) {
                    return value;
                }
                return invariant->IsPhi() ? nullptr : SubstitutePath(invariant, path, position, emit, true, depth);
            }
            const auto choice = ChoiceOf(path, value->Parent());
            IrValue* incoming = choice < 0 ? nullptr : ChosenIncoming(*value, path[choice]);
            return incoming == nullptr ? nullptr : SubstitutePath(incoming, path, position, emit, true, phiSearchDepth);
        }
        const auto op = value->Opcode();
        const bool register_ = op == IrOpcode::GetUserData;
        if (crossed) {
            const bool argumentless = (op == IrOpcode::GetShaderBase || op == IrOpcode::GetSrtResource) && value->ArgumentCount() == 0u;
            const bool srtRead = op == IrOpcode::ReadConst && value->ArgumentCount() == 2u && value->Argument(0)->Resolve()->Opcode() == IrOpcode::GetSrtResource && value->Argument(1)->Resolve()->HasImmediate();
            if (depth == 0u || (!register_ && !argumentless && !srtRead && !Detail::IsRuntimeUniformOp(op))) {
                return nullptr;
            }
        } else {
            std::vector<IrValue*> phis;
            CollectDescriptorPhis(value, phis, depth);
            if (phis.empty()) {
                return value;
            }
        }
        std::vector<IrValue*> arguments;
        for (std::size_t index = 0; index < value->ArgumentCount(); index++) {
            IrValue* argument = crossed && register_ ? value->Argument(index) : SubstitutePath(value->Argument(index), path, position, emit, crossed, depth - 1u);
            if (argument == nullptr) {
                return nullptr;
            }
            arguments.push_back(argument);
        }
        return emit ? EmitCopy(*value, arguments, position) : value;
    }

    IrValue* PathHead(IrValue* value, const PhiPath& path) const {
        value = value->Resolve();
        while (value->IsPhi()) {
            if (IrValue* invariant = ResolveInvariantPhi(m_program.Resources(), value)) {
                return invariant;
            }
            const auto choice = ChoiceOf(path, value->Parent());
            IrValue* incoming = choice < 0 ? nullptr : ChosenIncoming(*value, path[choice]);
            if (incoming == nullptr) {
                return value;
            }
            value = incoming->Resolve();
        }
        return value;
    }

    bool SameOnPaths(IrValue* left, const PhiPath& leftPath, IrValue* right, const PhiPath& rightPath, std::uint32_t depth) const {
        left = PathHead(left, leftPath);
        right = PathHead(right, rightPath);
        std::vector<IrValue*> phis;
        CollectDescriptorPhis(left, phis, phiSearchDepth);
        CollectDescriptorPhis(right, phis, phiSearchDepth);
        if (phis.empty()) {
            return EquivalentValue(m_program.Resources(), left, right);
        }
        if (depth == 0u || left->IsPhi() || right->IsPhi() || left->Opcode() != right->Opcode() || left->Type() != right->Type() || left->ArgumentCount() != right->ArgumentCount() || left->Flags<std::uint64_t>() != right->Flags<std::uint64_t>()) {
            return false;
        }
        for (std::size_t index = 0; index < left->ArgumentCount(); index++) {
            if (!SameOnPaths(left->Argument(index), leftPath, right->Argument(index), rightPath, depth - 1u)) {
                return false;
            }
        }
        return true;
    }

    IrValue* EdgeSelector(IrBlock& block, std::size_t edge, IrValue* taken) {
        for (const auto& selector : m_edgeSelectors) {
            if (selector.block == &block && selector.edge == edge && selector.taken == taken) {
                return selector.value;
            }
        }
        IrValue& phi = m_program.CreateValue(IrOpcode::Phi, IrType::Bool);
        const auto& predecessors = block.Predecessors();
        for (std::size_t index = 0; index < predecessors.size(); index++) {
            phi.AddPhiOperand(predecessors[index], index != edge ? &m_builder.ConstantBool(false) : taken != nullptr ? taken : &m_builder.ConstantBool(true));
        }
        block.InsertInstructionBefore(nullptr, &phi);
        m_edgeSelectors.push_back({&block, edge, taken, &phi});
        return &phi;
    }

    IrValue* PathSelector(const PhiPath& path, IrValue& position) {
        IrValue* selector = nullptr;
        for (const auto& choice : path) {
            IrValue* taken = EdgeSelector(*choice.block, choice.edge, nullptr);
            for (auto parent = choice.parent; parent >= 0; parent = path[parent].parent) {
                taken = EdgeSelector(*path[parent].block, path[parent].edge, taken);
            }
            const std::array<IrValue*, 2> arguments {selector, taken};
            selector = selector == nullptr ? taken : EmitBefore(position, IrOpcode::LogicalAnd, IrType::Bool, arguments);
        }
        return selector;
    }

    void SplitDescriptorPhi(IrValue& inst) {
        const auto info = ImageOpcodeInfoOf(inst.Opcode());
        const std::uint32_t handleCount = info.needsSampler ? 2u : 1u;
        if (inst.ArgumentCount() < handleCount || inst.Parent() == nullptr) {
            return;
        }
        std::array<IrValue*, 2> handles {};
        std::array<bool, 2> split {};
        bool any = false;
        for (std::uint32_t slot = 0; slot < handleCount; slot++) {
            IrValue* handle = inst.Argument(slot)->Resolve();
            const auto expected = slot == 0u ? IrOpcode::GetImageResource : IrOpcode::GetSamplerResource;
            if (handle->Opcode() != expected || handle->ArgumentCount() != (slot == 0u ? 8u : 4u)) {
                return;
            }
            handles[slot] = handle;
            std::vector<IrValue*> handlePhis;
            for (std::size_t dword = 0; dword < handle->ArgumentCount(); dword++) {
                CollectDescriptorPhis(handle->Argument(dword), handlePhis, phiSearchDepth);
            }
            split[slot] = !handlePhis.empty();
            any = any || split[slot];
        }
        if (!any) {
            return;
        }
        const std::span<IrValue* const> used(handles.data(), handleCount);
        const auto refuse = [&](const char* reason) {
            for (IrValue* handle : used) {
                m_splitRefusals.push_back({handle, reason});
            }
        };
        std::vector<PhiPath> paths;
        PhiPath prefix;
        if (const char* refusal = EnumeratePhiPaths(used, prefix, paths)) {
            refuse(refusal);
            return;
        }
        for (const auto& path : paths) {
            for (std::uint32_t slot = 0; slot < handleCount; slot++) {
                for (std::size_t dword = 0; split[slot] && dword < handles[slot]->ArgumentCount(); dword++) {
                    if (SubstitutePath(handles[slot]->Argument(dword), path, inst, false, false, phiSearchDepth) == nullptr) {
                        refuse("a selected descriptor is not read from the SRT or the user data");
                        return;
                    }
                }
            }
        }

        std::vector<std::uint32_t> pathArm(paths.size());
        std::vector<std::size_t> arms;
        for (std::size_t path = 0; path < paths.size(); path++) {
            std::uint32_t arm = 0;
            for (; arm < arms.size(); arm++) {
                bool same = true;
                for (std::uint32_t slot = 0; same && slot < handleCount; slot++) {
                    for (std::size_t dword = 0; same && split[slot] && dword < handles[slot]->ArgumentCount(); dword++) {
                        same = SameOnPaths(handles[slot]->Argument(dword), paths[path], handles[slot]->Argument(dword), paths[arms[arm]], phiSearchDepth);
                    }
                }
                if (same) {
                    break;
                }
            }
            if (arm == arms.size()) {
                arms.push_back(path);
            }
            pathArm[path] = arm;
        }

        std::vector<IrValue*> copies;
        const auto flags = inst.Flags<MemoryFlags>();
        const auto memory = m_program.Resources().memoryInfo.at(flags.index);
        for (std::uint32_t arm = 0; arm < arms.size(); arm++) {
            std::vector<IrValue*> arguments(inst.Arguments().begin(), inst.Arguments().end());
            for (std::uint32_t slot = 0; slot < handleCount; slot++) {
                if (!split[slot]) {
                    continue;
                }
                std::vector<IrValue*> dwords;
                for (std::size_t dword = 0; dword < handles[slot]->ArgumentCount(); dword++) {
                    dwords.push_back(SubstitutePath(handles[slot]->Argument(dword), paths[arms[arm]], inst, true, false, phiSearchDepth));
                }
                arguments[slot] = EmitCopy(*handles[slot], dwords, inst);
            }
            auto copyFlags = flags;
            if (arm != 0u) {
                copyFlags.index = static_cast<std::uint32_t>(m_program.Resources().memoryInfo.size());
                m_program.Resources().memoryInfo.push_back(memory);
            }
            std::uint64_t rawFlags = 0;
            std::memcpy(&rawFlags, &copyFlags, sizeof(copyFlags));
            copies.push_back(EmitBefore(inst, inst.Opcode(), inst.Type(), arguments, rawFlags));
        }

        IrValue* result = copies.back();
        if (copies.size() > 1u) {
            std::vector<IrValue*> selectors(copies.size() - 1u);
            for (std::size_t path = 0; path < paths.size(); path++) {
                if (pathArm[path] == selectors.size()) {
                    continue;
                }
                IrValue*& selector = selectors[pathArm[path]];
                const std::array<IrValue*, 2> arguments {selector, PathSelector(paths[path], inst)};
                selector = selector == nullptr ? arguments[1] : EmitBefore(inst, IrOpcode::LogicalOr, IrType::Bool, arguments);
            }
            std::array<IrValue*, 4> components {};
            for (std::uint32_t component = 0; component < components.size(); component++) {
                const auto extract = [&](IrValue* vector) {
                    const std::array<IrValue*, 2> arguments {vector, &m_builder.Constant(component)};
                    return EmitBefore(inst, IrOpcode::CompositeExtractU32x4, IrType::U32, arguments);
                };
                IrValue* selected = extract(copies.back());
                for (std::uint32_t arm = static_cast<std::uint32_t>(copies.size()) - 1u; arm-- > 0u;) {
                    const std::array<IrValue*, 3> arguments {selectors[arm], extract(copies[arm]), selected};
                    selected = EmitBefore(inst, IrOpcode::SelectU32, IrType::U32, arguments);
                }
                components[component] = selected;
            }
            result = EmitBefore(inst, IrOpcode::CompositeConstructU32x4, IrType::U32x4, components);
        }
        inst.ReplaceAllUsesWith(result);
        inst.Invalidate();
        inst.Parent()->RemoveInstruction(&inst);
    }

    void SplitDescriptorPhis() {
        std::vector<IrValue*> candidates;
        for (auto& block : m_program.Blocks()) {
            for (IrValue* inst : block->Instructions()) {
                if (SplittableImageRead(*inst)) {
                    candidates.push_back(inst);
                }
            }
        }
        for (IrValue* inst : candidates) {
            SplitDescriptorPhi(*inst);
        }
    }

    void PlanIndirectImages() {
        // Kill switch: APS5_NO_BINDLESS_IMAGES=1 leaves table-loaded T#s unplanned (GetHandle then
        // rejects them as before).
        static const bool enabled = std::getenv("APS5_NO_BINDLESS_IMAGES") == nullptr;
        if (!enabled) {
            return;
        }
        for (auto& block : m_program.Blocks()) {
            for (IrValue* inst : block->Instructions()) {
                const auto imageInfo = ImageOpcodeInfoOf(inst->Opcode());
                if (imageInfo.access == ImageAccess::None || inst->ArgumentCount() == 0u) {
                    continue;
                }
                IrValue* handle = inst->Argument(0)->Resolve();
                const IndirectImagePlan* planned = FindIndirectImage(*handle);
                IndirectImagePlan plan;
                if (planned == nullptr && !TryMakeTableImage(*handle, plan)) {
                    continue;
                }
                // A store through a table would mark every bound slot pending write-back.
                if (imageInfo.resourceClass == ImageResourceClass::Storage) {
                    ResourceMaterializer::CountBindlessRejection(BindlessRejection::Storage);
                    fail("bindless storage image tables are unsupported");
                }
                if (planned == nullptr) {
                    m_indirectImages.push_back(std::move(plan));
                }
            }
        }
    }

    void GetHandle(IrValue* value, IrOpcode expected, std::uint32_t width, IrValue*& handle, std::uint32_t& source, bool sampler = false, bool sampleAdjust = false) {
        handle = value->Resolve();
        if (handle->Opcode() != expected) {
            fail("memory operation requires " + std::string(IrOpcodeName(expected)));
        }
        DescriptorSource descriptor;
        MakeSource(*handle, width, sampler, sampleAdjust, descriptor);
        std::uint32_t badDword = 0;
        if (expected == IrOpcode::GetImageResource) {
            for (; badDword < descriptor.dwordCount; badDword++) {
                const IrValue* value2 = descriptor.dwords[badDword]->Resolve();
                if (value2->Opcode() == IrOpcode::ReadConstBuffer) {
                    fail(std::string(IrOpcodeName(expected)) + " dword " + std::to_string(badDword) + " is not a valid runtime value; chain: " + describeValueChain(descriptor.dwords[badDword], 8u));
                }
            }
            badDword = 0;
        }
        if (!ValidateSource(descriptor, badDword)) {
            const auto refusal = std::ranges::find(m_splitRefusals, handle, [](const auto& entry) { return entry.first; });
            fail(std::string(IrOpcodeName(expected)) + " dword " + std::to_string(badDword) + " is not a valid runtime value; chain: " + describeValueChain(descriptor.dwords[badDword], 8u) + (refusal == m_splitRefusals.end() ? std::string() : std::string("; its descriptor phis were not split: ") + refusal->second));
        }
        source = InternSource(descriptor);
    }

    bool TakeGpuDescriptor(IrValue& inst, std::uint32_t memoryIndex) {
        const IrValue* handle = inst.Argument(0)->Resolve();
        if (handle->Opcode() != IrOpcode::GetBufferResource || handle->ArgumentCount() != 4u) {
            return false;
        }
        DescriptorSource descriptor;
        MakeSource(*handle, 4u, false, false, descriptor);
        std::uint32_t badDword = 0;
        auto& memory = m_program.Resources().memoryInfo[memoryIndex];
        const auto access = BufferAccessOf(inst.Opcode());
        if (ValidateSource(descriptor, badDword)) {
            const auto source = InternSource(descriptor);
            const auto resource = AddBuffer(source, memory, inst.Opcode(), inst.Flags<MemoryFlags>().pc);
            if (resource == std::numeric_limits<std::uint32_t>::max()) fail("buffer resource limit exceeded");
            AddMemoryPatch(memoryIndex, resource, 0u, false);
            memory.gpuDescriptor = false;
            m_info.usesDma = m_info.usesDma || access == BufferAccess::Atomic;
            return true;
        }
        memory.gpuDescriptor = true;
        m_info.usesDma = true;
        m_info.bdaWrites = m_info.bdaWrites || access == BufferAccess::Write || access == BufferAccess::Atomic;
        return true;
    }

    void ValidateAddressHandle(IrValue* value) const {
        const IrValue* handle = value->Resolve();
        if (handle->Opcode() != IrOpcode::GetAddressResource) {
            fail("address operation requires GetAddressResource");
        }
        if (handle->ArgumentCount() != 2) {
            fail("GetAddressResource must have two address dwords");
        }
    }

    std::uint32_t AddBuffer(std::uint32_t source, const MemoryInfo& memory, IrOpcode op, std::uint32_t pc) {
        for (std::uint32_t i = 0; i < m_info.buffers.size(); i++) {
            if (m_info.buffers[i].source == source) {
                Merge(m_info.buffers[i], memory, op, pc);
                return i;
            }
        }
        BufferResource resource;
        resource.source = source;
        resource.firstUsePc = pc;
        Merge(resource, memory, op, pc);
        m_info.buffers.push_back(resource);
        return static_cast<std::uint32_t>(m_info.buffers.size() - 1);
    }

    static void Merge(BufferResource& resource, const MemoryInfo& memory, IrOpcode op, std::uint32_t pc) {
        const auto access = BufferAccessOf(op);
        const bool atomic = access == BufferAccess::Atomic;
        const bool write = access == BufferAccess::Write || atomic;
        resource.firstUsePc = std::min(resource.firstUsePc, pc);
        resource.maxByteExtent = std::max(resource.maxByteExtent, byteExtent(memory));
        resource.read = resource.read || !write || atomic;
        resource.written = resource.written || write;
        resource.atomic = resource.atomic || atomic;
        resource.formatted = resource.formatted || memory.formatted;
        resource.descriptorFormatted = resource.descriptorFormatted || (memory.formatted && !memory.typed);
        if (memory.formatted && memory.typed) {
            const auto format = GetFormatInfo(DecodeTBufferFormat(memory.dataFormat, memory.numberFormat));
            if (format.byteSize == 0u) fail("typed buffer instruction has an invalid format");
            resource.typedAlignment = std::max(resource.typedAlignment, static_cast<std::uint8_t>(std::min(format.byteSize, 4u)));
        }
        if (memory.formatted && !memory.typed && !write) resource.formattedReadMask |= (1u << std::min(memory.dataDwords, 4u)) - 1u;
        resource.scalar = resource.scalar || op == IrOpcode::ReadConstBuffer || memory.kind == ResourceKind::ScalarBuffer;
    }

    std::uint32_t AddImage(std::uint32_t source, const MemoryInfo& memory, IrOpcode op, std::uint32_t pc) {
        const auto resourceClass = ImageOpcodeInfoOf(op).resourceClass;
        const auto mip = resourceClass == ImageResourceClass::Storage && memory.imageHasMip ? ImageMipMode::DynamicStorage : ImageMipMode::None;
        const bool depth = (memory.imageSampleFlags & RdnaImageSampleFlagCompare) != 0;
        for (std::uint32_t i = 0; i < m_info.images.size(); i++) {
            auto& image = m_info.images[i];
            if (image.source == source && image.resourceClass == resourceClass && image.dimension == memory.imageDimension && image.mipMode == mip && image.depthCompare == depth && image.r128 == memory.imageR128 && image.packed == memory.imagePacked && image.byElements == memory.imageByElements && image.byComponents == (memory.imageByElements != 0u ? memory.dataDwords / memory.imageByElements : 0u)) {
                Merge(image, memory, op, pc);
                return i;
            }
        }
        if (m_info.images.size() >= ShaderInfo::MaxImages) {
            return std::numeric_limits<std::uint32_t>::max();
        }
        ImageResource image;
        image.source = source;
        image.firstUsePc = pc;
        image.resourceClass = resourceClass;
        image.dimension = memory.imageDimension;
        image.mipMode = mip;
        image.depthCompare = depth;
        image.r128 = memory.imageR128;
        image.packed = memory.imagePacked;
        image.byElements = memory.imageByElements;
        image.byComponents = memory.imageByElements != 0u ? memory.dataDwords / memory.imageByElements : 0u;
        Merge(image, memory, op, pc);
        m_info.images.push_back(image);
        return static_cast<std::uint32_t>(m_info.images.size() - 1);
    }

    static void Merge(ImageResource& image, const MemoryInfo& memory, IrOpcode op, std::uint32_t pc) {
        const auto access = ImageOpcodeInfoOf(op).access;
        const bool atomic = access == ImageAccess::Atomic;
        const bool write = access == ImageAccess::Write || atomic;
        const bool atomic64 = IsImageAtomic64Opcode(op);
        if ((image.read || image.written) && image.atomic64 != atomic64) {
            throw std::runtime_error("an image accessed by 64-bit atomics is also accessed in another way");
        }
        image.atomic64 = atomic64;
        image.firstUsePc = std::min(image.firstUsePc, pc);
        image.read = image.read || !write || atomic;
        image.written = image.written || write;
        image.atomic = image.atomic || atomic;
        image.srgbDecodeCompatible = image.srgbDecodeCompatible && !ImageOpcodeInfoOf(op).needsSampler;
        image.fmaskCompatible = image.fmaskCompatible && op == IrOpcode::ImageRead && memory.dataBits == 32u;
        image.depthBitsCompatible = image.depthBitsCompatible && memory.dataBits == 32u;
        if ((memory.imageSampleFlags & RdnaImageSampleFlagCompare) != 0u) {
            constexpr auto unsupported = RdnaImageSampleFlagLod | RdnaImageSampleFlagDerivative;
            if (op == IrOpcode::ImageGatherRaw || (memory.imageSampleFlags & unsupported) != 0u) image.emulatedCompare |= EmulatedCompare::Unsupported;
            if ((memory.imageSampleFlags & RdnaImageSampleFlagLevelZero) == 0u) image.emulatedCompare |= EmulatedCompare::RequiresSingleLevel;
        }
    }

    std::uint32_t AddSampler(std::uint32_t source, std::uint32_t pc) {
        for (std::uint32_t i = 0; i < m_info.samplers.size(); i++) {
            if (m_info.samplers[i].source == source) {
                m_info.samplers[i].firstUsePc = std::min(m_info.samplers[i].firstUsePc, pc);
                return i;
            }
        }
        if (m_info.samplers.size() >= ShaderInfo::MaxSamplers) {
            return std::numeric_limits<std::uint32_t>::max();
        }
        m_info.samplers.push_back({source, pc});
        return static_cast<std::uint32_t>(m_info.samplers.size() - 1);
    }

    void AddSampledPair(std::uint32_t image, std::uint32_t sampler, std::uint32_t pc) {
        for (auto& pair : m_info.sampledPairs) {
            if (pair.image == image && pair.sampler == sampler) {
                pair.firstUsePc = std::min(pair.firstUsePc, pc);
                return;
            }
        }
        if (m_info.sampledPairs.size() >= ShaderInfo::MaxSampledPairs) {
            fail("sampled image/sampler pair limit exceeded");
        }
        m_info.sampledPairs.push_back({image, sampler, pc});
    }

    void AddHandlePatch(IrValue* handle, std::uint32_t resource) {
        for (const auto& patch : m_handlePatches) {
            if (patch.handle == handle) {
                if (patch.resource != resource) {
                    fail(std::string(IrOpcodeName(handle->Opcode())) + " is reused with incompatible resource classes");
                }
                return;
            }
        }
        m_handlePatches.push_back({handle, resource});
    }

    void AddMemoryPatch(std::uint32_t index, std::uint32_t resource, std::uint32_t sampler, bool hasSampler) {
        for (auto& patch : m_memoryPatches) {
            if (patch.index != index) {
                continue;
            }
            if (patch.resource != resource || (hasSampler && patch.hasSampler && patch.sampler != sampler)) {
                fail("memory metadata is reused with incompatible resources");
            }
            if (hasSampler) {
                patch.sampler = sampler;
                patch.hasSampler = true;
            }
            return;
        }
        m_memoryPatches.push_back({index, resource, sampler, hasSampler});
    }

    void Collect(IrValue& inst) {
        const auto op = inst.Opcode();
        if (op == IrOpcode::ImageBvhIntersectRay) {
            m_info.usesDma = true;
            return;
        }
        const auto buffer = BufferAccessOf(op);
        const auto addressInfo = AddressOpcodeInfoOf(op);
        const auto imageInfo = ImageOpcodeInfoOf(op);
        if (buffer == BufferAccess::None && addressInfo.access == AddressAccess::None && imageInfo.access == ImageAccess::None) {
            return;
        }
        const auto flags = inst.Flags<MemoryFlags>();
        if (flags.index >= m_program.Resources().memoryInfo.size()) {
            fail("memory metadata index " + std::to_string(flags.index) + " is out of range");
        }
        if (inst.ArgumentCount() == 0) {
            fail("memory operation has no resource handle");
        }
        const auto& memory = m_program.Resources().memoryInfo[flags.index];
        if (memory.planningOnly || IsIndirectPlanningMemory(flags.index)) {
            return;
        }
        IrValue* handle = nullptr;
        std::uint32_t source = 0;
        std::uint32_t resource = 0;

        if (buffer != BufferAccess::None) {
            if (!TakeGpuDescriptor(inst, flags.index)) fail("buffer operation requires a four-dword runtime V#");
            return;
        }
        if (addressInfo.access != AddressAccess::None) {
            if (!IsAddressResourceKind(memory.kind)) {
                fail("address operation has invalid resource kind");
            }
            if (memory.kind == ResourceKind::Scratch) {
                handle = inst.Argument(0)->Resolve();
                if (handle->Opcode() != IrOpcode::GetScratchResource || handle->ArgumentCount() != 0) {
                    fail("scratch operation requires GetScratchResource");
                }
                if (m_info.scratchDwords == 0) {
                    fail("scratch operation requires a nonzero AGC per-thread size");
                }
                return;
            }
            ValidateAddressHandle(inst.Argument(0));
            // Debug aid: APS5_TRACE_BDA=1 names every access that makes the program address-based
            // (a raw scalar load the SRT walker left in place, or a flat/global access), so the
            // reason a stage takes the BDA path can be read off without a shader dump.
            // The first few accesses of a program are printed (a Bink kernel has hundreds); Run
            // reports how many more there were.
            if (bdaTraceEnabled() && ++m_bdaTraces <= bdaTraceLimit) {
                const auto* kind = memory.kind == ResourceKind::ScalarAddress ? "scalar address" : memory.kind == ResourceKind::Global ? "global" : "flat";
                const IrValue* offset = inst.ArgumentCount() > 1 ? inst.Argument(1)->Resolve() : nullptr;
                const bool immediateOffset = offset != nullptr && offset->HasImmediate();
                std::fprintf(stderr, "[bda] %s at pc 0x%08x: %s access, offset %s%s\n", std::string(IrOpcodeName(op)).c_str(), flags.pc, kind, immediateOffset ? "immediate" : "dynamic", memory.kind == ResourceKind::ScalarAddress && !immediateOffset ? " (a register offset is not planned by the SRT walker)" : "");
            }
            m_info.usesDma = true;
            m_info.bdaWrites = m_info.bdaWrites || addressInfo.access == AddressAccess::Write || addressInfo.access == AddressAccess::Atomic;
            return;
        }

        if (memory.kind != ResourceKind::Image || imageInfo.resourceClass == ImageResourceClass::None) {
            fail("image operation has invalid resource kind");
        }
        handle = inst.Argument(0)->Resolve();
        const IndirectImagePlan* indirect = FindIndirectImage(*handle);
        if (indirect != nullptr) {
            source = indirect->source;
        } else {
            GetHandle(inst.Argument(0), IrOpcode::GetImageResource, 8, handle, source);
        }
        resource = AddImage(source, memory, op, flags.pc);
        if (resource == std::numeric_limits<std::uint32_t>::max()) {
            fail("image resource limit exceeded");
        }
        AddHandlePatch(handle, resource);
        std::uint32_t sampler = 0;
        if (imageInfo.needsSampler) {
            if (inst.ArgumentCount() < 2) {
                fail("sampled image operation has no sampler handle");
            }
            IrValue* samplerHandle = nullptr;
            std::uint32_t samplerSource = 0;
            const bool sampleAdjust = (memory.imageSampleFlags & RdnaImageSampleFlagAdjust) != 0;
            GetHandle(inst.Argument(1), IrOpcode::GetSamplerResource, 4, samplerHandle, samplerSource, true, sampleAdjust);
            sampler = AddSampler(samplerSource, flags.pc);
            if (sampler == std::numeric_limits<std::uint32_t>::max()) {
                fail("sampler resource limit exceeded");
            }
            AddHandlePatch(samplerHandle, sampler);
            AddSampledPair(resource, sampler, flags.pc);
        }
        AddMemoryPatch(flags.index, resource, sampler, imageInfo.needsSampler);
    }

    const DescriptorSource* Source(std::uint32_t source) const {
        return source < m_sources.size() ? &m_sources[source] : nullptr;
    }

    void LinkImageAliases() {
        for (auto& buffer : m_info.buffers) {
            const DescriptorSource* bufferSource = Source(buffer.source);
            if (bufferSource == nullptr || bufferSource->dwordCount != 4) {
                continue;
            }
            for (std::uint32_t image = 0; image < m_info.images.size(); image++) {
                const DescriptorSource* imageSource = Source(m_info.images[image].source);
                if (imageSource == nullptr || imageSource->dwordCount != 8 || imageSource->indirectImage.has_value()) {
                    continue;
                }
                bool alias = true;
                for (std::uint32_t dword = 0; dword < 4; dword++) {
                    alias = alias && EquivalentValue(m_program.Resources(), bufferSource->dwords[dword], imageSource->dwords[dword]);
                }
                if (alias) {
                    buffer.imageAlias = image;
                    break;
                }
            }
        }
    }

    IrProgram& m_program;
    ShaderInfo m_info;
    IrBuilder m_builder;
    std::vector<DescriptorSource> m_sources;
    std::vector<HandlePatch> m_handlePatches;
    std::vector<MemoryPatch> m_memoryPatches;
    // APS5_TRACE_BDA: address accesses seen by Collect (the first bdaTraceLimit are printed).
    unsigned m_bdaTraces = 0;
    std::vector<IndirectImagePlan> m_indirectImages;
    struct EdgeSelectorEntry {
        IrBlock* block = nullptr;
        std::size_t edge = 0;
        IrValue* taken = nullptr;
        IrValue* value = nullptr;
    };
    std::vector<EdgeSelectorEntry> m_edgeSelectors;
    std::vector<std::pair<const IrValue*, const char*>> m_splitRefusals;
};

}

void ResourceTracker::Track(IrProgram& program) const {
    Tracker(program).Run();
}

}
