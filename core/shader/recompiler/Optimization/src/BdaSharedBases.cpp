#include "Optimization/BdaSharedBases.hpp"
#include "Optimization/SrtWalker.hpp"
#include "BdaAbi.hpp"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <unordered_set>

namespace ShaderRecompiler {

std::uint32_t UsedBdaDwords(const IrValue& inst, std::uint32_t dwords) {
    const auto all = (1u << dwords) - 1u;
    if (dwords == 1u) return all;
    std::uint32_t used = 0;
    for (const IrValue* user : inst.Uses()) {
        const auto opcode = user->Opcode();
        if (opcode != IrOpcode::CompositeExtractU32x2 && opcode != IrOpcode::CompositeExtractU32x3 && opcode != IrOpcode::CompositeExtractU32x4) return all;
        const IrValue* index = user->ArgumentCount() > 1u ? user->Argument(1) : nullptr;
        if (index == nullptr || !index->HasImmediate() || index->ImmediateU32() >= dwords) return all;
        used |= 1u << index->ImmediateU32();
    }
    return used;
}

std::pair<std::uint32_t, std::uint32_t> ExtractedBdaDwords(const IrValue& inst, std::uint32_t dwords, bool everyDword) {
    const auto used = everyDword ? (1u << dwords) - 1u : UsedBdaDwords(inst, dwords);
    std::uint32_t first = dwords;
    std::uint32_t last = 0;
    for (std::uint32_t dword = 0; dword < dwords; ++dword) {
        if ((used & (1u << dword)) == 0u) continue;
        first = std::min(first, dword);
        last = dword;
    }
    return {first, last};
}

std::vector<BdaBaseGroup> CollectBdaBaseGroups(const IrProgram& program) {
    std::vector<BdaBaseGroup> groups;
    std::map<BdaBaseKey, std::size_t> found;
    // The blocks and activity conditions under which a read of each base already tries the hull.
    std::vector<std::set<std::pair<const IrBlock*, const IrValue*>>> tried;
    const auto& memory = program.Resources().memoryInfo;
    for (const auto* block : program.BlockOrder()) {
        for (const auto* inst : block->Instructions()) {
            std::uint32_t dwords = 0;
            switch (inst->Opcode()) {
                case IrOpcode::LoadAddressU32: dwords = 1u; break;
                case IrOpcode::LoadAddressU32x2: dwords = 2u; break;
                case IrOpcode::LoadAddressU32x3: dwords = 3u; break;
                case IrOpcode::LoadAddressU32x4: dwords = 4u; break;
                default: continue;
            }
            const auto index = inst->Flags<MemoryFlags>().index;
            if (index >= memory.size() || inst->ArgumentCount() < 4u) continue;
            const auto& mem = memory[index];
            const bool scalar = mem.kind == ResourceKind::ScalarAddress;
            if (mem.planningOnly || (!scalar && mem.kind != ResourceKind::Flat && mem.kind != ResourceKind::Global) || (scalar && dwords != 1u)) continue;
            const auto [first, last] = ExtractedBdaDwords(*inst, dwords, false);
            if (first > last) continue;
            // As the emitter's GuestAddress and EmitBdaDwordReads add them.
            const auto at = [&](std::uint32_t dword) { return static_cast<std::int64_t>(static_cast<std::int32_t>(dwords == 1u ? (scalar ? mem.offset & ~3u : mem.offset) : mem.offset + dword * 4u)); };
            if (at(last) - at(first) != static_cast<std::int64_t>(last - first) * 4) continue;
            const IrValue* low = inst->Argument(1) != nullptr ? inst->Argument(1)->Resolve() : nullptr;
            const IrValue* pairLow = nullptr;
            const IrValue* pairHigh = nullptr;
            if (mem.addressIsFull) {
                pairHigh = inst->Argument(2) != nullptr ? inst->Argument(2)->Resolve() : nullptr;
            } else {
                const IrValue* handle = inst->Argument(0) != nullptr ? inst->Argument(0)->Resolve() : nullptr;
                if (handle == nullptr || handle->Opcode() != IrOpcode::GetAddressResource || handle->ArgumentCount() != 2u || handle->Argument(0) == nullptr || handle->Argument(1) == nullptr) continue;
                pairLow = handle->Argument(0)->Resolve();
                pairHigh = handle->Argument(1)->Resolve();
            }
            if (low == nullptr || pairHigh == nullptr) continue;
            const BdaBaseKey key{pairLow, pairHigh, low, scalar, static_cast<std::uint32_t>(at(first)) & 3u};
            const auto [entry, added] = found.try_emplace(key, groups.size());
            if (added) {
                groups.push_back({key, {}, {}, at(first), at(last) + 4});
                tried.emplace_back();
            }
            auto& group = groups[entry->second];
            const IrValue* active = inst->Argument(inst->ArgumentCount() - 1u);
            group.tries.push_back(tried[entry->second].emplace(block, active != nullptr ? active->Resolve() : nullptr).second);
            group.reads.emplace_back(inst, at(first));
            group.first = std::min(group.first, at(first));
            group.end = std::max(group.end, at(last) + 4);
        }
    }
    return groups;
}

bool BdaCpuBases() {
    static const bool refused = std::getenv("APS5_NO_BDA_CPU_BASE") != nullptr;
    return !refused;
}

namespace {

// Whether evaluating `root` reads guest memory only through the SRT reads of the plan, which the
// resource walk evaluates for every draw: a base may then be evaluated without a read the draw
// would not have made (and waited for, where recorded GPU work writes the page).
bool ReadsOnlyThroughSrt(const IrValue* root) {
    std::unordered_set<const IrValue*> visited;
    std::vector<const IrValue*> pending{root};
    while (!pending.empty()) {
        const IrValue* value = pending.back()->Resolve();
        pending.pop_back();
        if (!visited.insert(value).second || value->Opcode() == IrOpcode::ReadConst) continue;
        switch (value->Opcode()) {
            case IrOpcode::LoadAddressU32:
            case IrOpcode::LoadAddressU32x2:
            case IrOpcode::LoadAddressU32x3:
            case IrOpcode::LoadAddressU32x4:
            case IrOpcode::ReadConstBuffer: return false;
            default: break;
        }
        for (std::size_t index = 0; index < value->ArgumentCount(); ++index) {
            if (value->Argument(index) != nullptr) pending.push_back(value->Argument(index));
        }
    }
    return true;
}

}

void PlanBdaBases(IrProgram& program) {
    auto& plan = program.Resources();
    plan.bdaBases.clear();
    if (!BdaCpuBases()) return;
    const SrtWalker walker;
    for (const auto& group : CollectBdaBaseGroups(program)) {
        if (group.reads.size() < 2u || plan.bdaBases.size() == BdaAbi::BaseSlotCapacity) continue;
        bool uniform = true;
        for (const IrValue* value : {group.key.pairLow, group.key.pairHigh, group.key.low}) uniform = uniform && (value == nullptr || (walker.ValidateRuntimeValue(plan, value, RuntimeValueType::Integer) && ReadsOnlyThroughSrt(value)));
        if (!uniform) continue;
        plan.bdaBases.push_back({const_cast<IrValue*>(group.key.pairLow), const_cast<IrValue*>(group.key.pairHigh), const_cast<IrValue*>(group.key.low), group.key.scalar, group.key.residue, group.first, static_cast<std::uint32_t>(group.end - group.first)});
    }
}

}
