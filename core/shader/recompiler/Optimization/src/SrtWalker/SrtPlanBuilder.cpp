#include "Optimization/SrtWalker/SrtPlanBuilder.hpp"
#include "Optimization/SrtWalker/SrtDiagnostics.hpp"
#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"
#include "Optimization/SrtWalker/SrtRuntimeValidator.hpp"

#include "IntermediateRepresentation/IrBuilder.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace ShaderRecompiler::Detail {

void PlanBuilder::Run() {
    _program.Resources().srtReads.clear();
    _program.Metadata().dynamicReads.clear();
    classifyControlFlow();
    for (auto& block : _program.Blocks()) {
        for (IrValue* inst : block->Instructions()) {
            const auto op = inst->Opcode();
            if (op == IrOpcode::LoadAddressU32 || op == IrOpcode::ReadConstBuffer) {
                const auto flags = inst->Flags<MemoryFlags>();
                if (flags.index < _program.Resources().memoryInfo.size()) {
                    const auto kind = _program.Resources().memoryInfo[flags.index].kind;
                    const bool crosswired = (op == IrOpcode::LoadAddressU32 && kind == ResourceKind::ScalarBuffer) || (op == IrOpcode::ReadConstBuffer && kind == ResourceKind::ScalarAddress);
                    if (crosswired) {
                        Fail(_program.Resources(), flags.pc, std::string(IrOpcodeName(op)) + " has incompatible scalar memory metadata");
                    }
                }
            }
            if (op == IrOpcode::GetImageResource || op == IrOpcode::GetSamplerResource || (op == IrOpcode::GetBufferResource && canCapture(inst, true))) {
                for (std::size_t index = 0; index < inst->ArgumentCount(); index++) {
                    Collect(inst->Argument(index), 0);
                }
            }
        }
    }
    for (auto& block : _program.Blocks()) {
        for (IrValue* inst : block->Instructions()) {
            if (inst->Opcode() == IrOpcode::LoadAddressU32 && IsRawRead(_program.Resources(), *inst) && inst->Argument(1)->Resolve()->HasImmediate() && canCapture(inst, false)) {
                Collect(inst, inst->Flags<MemoryFlags>().pc);
            }
        }
    }
    PatchReads();
    buildControlFlow();
}

bool PlanBuilder::canCapture(IrValue* raw, bool scalarBuffer) const {
    if (raw == nullptr) return false;
    RuntimeValidator validator(_program.Resources(), RuntimeValueType::Any);
    std::unordered_set<IrValue*> validated;
    std::unordered_set<IrValue*> visiting;
    const std::function<bool(IrValue*)> visit = [&](IrValue* value) {
        value = value->Resolve();
        if (validated.contains(value)) return true;
        if (!validator.Run(value)) return false;
        if (!visiting.insert(value).second) return value->Opcode() == IrOpcode::Phi;
        const auto finish = [&](bool valid) {
            visiting.erase(value);
            if (valid) validated.insert(value);
            return valid;
        };
        const auto op = value->Opcode();
        if (op == IrOpcode::Void || op == IrOpcode::GetUserData || op == IrOpcode::GetShaderBase || op == IrOpcode::GetSrtResource) return finish(true);
        if (op == IrOpcode::Phi && ResolveInvariantPhi(_program.Resources(), value) == nullptr) return finish(false);
        if (op == IrOpcode::ReadFirstLane) {
            auto* mask = value->Argument(1)->Resolve();
            if (!mask->HasImmediate() || mask->Type() != IrType::Bool || !mask->ImmediateBool()) return finish(false);
        } else if (op == IrOpcode::ReadConst) {
            return finish(false);
        } else if (op == IrOpcode::LoadAddressU32 || op == IrOpcode::ReadConstBuffer) {
            if (!IsRawRead(_program.Resources(), *value) || (!scalarBuffer && op == IrOpcode::ReadConstBuffer) || value->Parent() == nullptr || _gpuControlled.contains(value->Parent())) return finish(false);
            auto* offset = value->Argument(1)->Resolve();
            if (!offset->HasImmediate() || offset->Type() != IrType::U32) return finish(false);
        } else if (op != IrOpcode::Phi && !IsDescriptorHandle(op) && !IsRuntimeUniformOp(op)) {
            return finish(false);
        }
        for (std::size_t index = 0; index < value->ArgumentCount(); ++index) {
            if (!visit(value->Argument(index))) return finish(false);
        }
        return finish(true);
    };
    return visit(raw);
}

void PlanBuilder::classifyControlFlow() {
    const auto& blocks = _program.BlockOrder();
    const auto& metadata = _program.Metadata().blockInfo;
    if (metadata.size() != blocks.size() || blocks.empty()) {
        for (const auto& block : _program.Blocks()) _gpuControlled.insert(block.get());
        return;
    }
    bool changed;
    do {
        changed = false;
        for (std::size_t index = 0; index < blocks.size(); ++index) {
            const auto& info = metadata[index];
            const bool unknown = _gpuControlled.contains(blocks[index]) ||
                (info.terminator.kind == TerminatorKind::ConditionalBranch && !canCapture(info.condition, false)) ||
                info.terminator.kind == TerminatorKind::IndirectBranch || info.terminator.kind == TerminatorKind::Unsupported;
            if (unknown) {
                for (const auto* successor : blocks[index]->Successors()) changed = _gpuControlled.insert(successor).second || changed;
            }
        }
    } while (changed);
}

void PlanBuilder::buildControlFlow() {
    auto& flow = _program.Resources().controlFlow;
    flow.clear();
    const auto& blocks = _program.BlockOrder();
    const auto& metadata = _program.Metadata().blockInfo;
    if (_patches.empty() || metadata.size() != blocks.size()) return;
    std::unordered_map<const IrBlock*, std::uint32_t> indices;
    std::unordered_map<std::uint32_t, std::uint32_t> targets;
    for (std::uint32_t index = 0; index < blocks.size(); ++index) {
        indices.emplace(blocks[index], index);
        targets.emplace(metadata[index].id, index);
    }
    flow.resize(blocks.size());
    for (std::uint32_t index = 0; index < blocks.size(); ++index) {
        auto& block = flow[index];
        const auto& info = metadata[index];
        if (info.terminator.kind == TerminatorKind::ConditionalBranch) {
            block.condition = info.condition;
            block.successors = {targets.at(info.terminator.trueBlock), targets.at(info.terminator.falseBlock)};
        } else {
            for (const auto* successor : blocks[index]->Successors()) block.successors.push_back(indices.at(successor));
        }
    }
    for (const auto& patch : _patches) {
        auto& reads = flow.at(indices.at(patch.inst->Parent())).reads;
        if (std::ranges::find(reads, patch.slot) == reads.end()) reads.push_back(patch.slot);
    }
}

void PlanBuilder::Collect(IrValue* raw, std::uint32_t usePc) {
    IrValue* value = raw->Resolve();
    if (value->Opcode() == IrOpcode::Void) {
        return;
    }
    IrValue* inst = value;
    const auto cycle = std::find(_visiting.begin(), _visiting.end(), inst);
    if (cycle != _visiting.end()) {
        const auto containsPhi = std::any_of(cycle, _visiting.end(), [](IrValue* candidate) { return candidate->Opcode() == IrOpcode::Phi; });
        if (containsPhi) {
            return;
        }
        Fail(_program.Resources(), usePc, "cyclic typed planning value " + std::string(IrOpcodeName(inst->Opcode())) + " without a phi");
    }
    if (std::find(_visited.begin(), _visited.end(), inst) != _visited.end()) {
        return;
    }
    _visiting.push_back(inst);
    for (std::size_t index = 0; index < inst->ArgumentCount(); index++) {
        Collect(inst->Argument(index), usePc);
    }
    _visiting.pop_back();
    _visited.push_back(inst);
    if (!IsRawRead(_program.Resources(), *inst)) {
        return;
    }
    IrValue* offset = inst->Argument(1)->Resolve();
    const auto foldable = offset->HasImmediate() && offset->Type() == IrType::U32 && RuntimeValidator(_program.Resources(), RuntimeValueType::Any).Run(inst);
    if (!foldable) {
        if (std::find(_program.Metadata().dynamicReads.begin(), _program.Metadata().dynamicReads.end(), value) == _program.Metadata().dynamicReads.end()) {
            _program.Metadata().dynamicReads.push_back(value);
        }
        return;
    }
    for (std::uint32_t slot = 0; slot < _program.Resources().srtReads.size(); slot++) {
        if (EquivalentValue(_program.Resources(), value, _program.Resources().srtReads[slot].value)) {
            _patches.push_back({inst, slot, false});
            return;
        }
    }
    const auto slot = static_cast<std::uint32_t>(_program.Resources().srtReads.size());
    _program.Resources().srtReads.push_back({value, slot});
    _patches.push_back({inst, slot, true});
}

void PlanBuilder::PatchReads() {
    IrBuilder builder(_program);
    for (const auto& patch : _patches) {
        IrBlock* block = patch.inst->Parent();
        if (block == nullptr) {
            throw std::runtime_error("SrtWalker::BuildPlan patched value has no parent block");
        }
        builder.SetInsertionPoint(*block);
        IrValue& resource = builder.Emit(IrOpcode::GetSrtResource, IrType::SrtResource, {});
        block->RemoveInstruction(&resource);
        block->InsertInstructionBefore(patch.inst, &resource);
        IrValue& slotConstant = builder.Constant(patch.slot);
        IrValue& flat = builder.Emit(IrOpcode::ReadConst, IrType::U32, {&resource, &slotConstant});
        block->RemoveInstruction(&flat);
        block->InsertInstructionBefore(patch.inst, &flat);
        for (const auto& use : std::vector<IrUse>(patch.inst->OperandUses())) {
            use.user->ReplaceArgument(use.operand, &flat);
        }
        for (auto& info : _program.Metadata().blockInfo) {
            if (info.condition != nullptr && info.condition->Resolve() == patch.inst) {
                info.condition = &flat;
            }
            if (info.indirectTarget != nullptr && info.indirectTarget->Resolve() == patch.inst) {
                info.indirectTarget = &flat;
            }
        }
        if (patch.keep) {
            const auto memory = patch.inst->Flags<MemoryFlags>().index;
            if (memory < _program.Resources().memoryInfo.size()) {
                _program.Resources().memoryInfo[memory].planningOnly = true;
            }
            const auto& irVal = builder.Emit(IrOpcode::ReferenceU32, IrType::Void, {patch.inst});
        }
    }
}

}
