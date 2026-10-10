#include "Optimization/SrtWalker/SrtRuntimeValidator.hpp"
#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"

#include "IntermediateRepresentation/IrBuilder.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace ShaderRecompiler::Detail {

void RuntimeValidator::analyzeCycles(IrValue* root) {
    std::unordered_map<IrValue*, std::pair<std::size_t, std::size_t>> indices;
    std::unordered_set<IrValue*> onStack;
    std::vector<IrValue*> stack;
    const auto visit = [&](auto&& self, IrValue* value) -> void {
        if (_analyzedValues.contains(value)) return;
        const auto index = indices.size();
        indices.emplace(value, std::pair{index, index});
        stack.push_back(value);
        onStack.insert(value);
        const auto follow = [&](IrValue* child) {
            if (child == value) _cyclicValues.insert(value);
            if (_analyzedValues.contains(child)) return;
            if (!indices.contains(child)) {
                self(self, child);
                indices.at(value).second = std::min(indices.at(value).second, indices.at(child).second);
            } else if (onStack.contains(child)) {
                indices.at(value).second = std::min(indices.at(value).second, indices.at(child).first);
            }
        };
        for (auto* argument : value->Arguments()) if (argument != nullptr) follow(argument);
        if (_type == RuntimeValueType::Integer && value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() == 2u) {
            auto* slot = value->Argument(1);
            std::unordered_set<IrValue*> identities;
            while (slot != nullptr && slot->IsIdentity()) {
                if (!identities.insert(slot).second || slot->ArgumentCount() == 0u) {
                    slot = nullptr;
                    break;
                }
                slot = slot->Argument(0);
            }
            if (slot != nullptr && slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < _program.srtReads.size()) {
                auto* read = _program.srtReads[slot->ImmediateU32()].value;
                if (read != nullptr) follow(read);
            }
        }
        if (indices.at(value).second != index) return;
        const auto component = std::find(stack.rbegin(), stack.rend(), value).base() - 1;
        if (stack.end() - component > 1) _cyclicValues.insert(component, stack.end());
        for (auto member = component; member != stack.end(); ++member) {
            onStack.erase(*member);
            _analyzedValues.insert(*member);
        }
        stack.erase(component, stack.end());
    };
    visit(visit, root);
}

bool RuntimeValidator::ValidateArguments(IrValue& inst, bool requireUniform) {
    for (std::size_t index = 0; index < inst.ArgumentCount(); index++) {
        if (!Validate(inst.Argument(index), requireUniform)) return false;
    }
    return true;
}

bool RuntimeValidator::Validate(IrValue* raw, bool requireUniform) {
    IrValue* value = raw->Resolve();
    if (_type == RuntimeValueType::Integer && TypesOverlap(value->Type(), IrType::F16 | IrType::F32 | IrType::Vec2F32)) {
        return false;
    }
    if (value->Opcode() == IrOpcode::Void) {
        if (!requireUniform) return true;
        switch (value->Type()) {
            case IrType::Bool:
            case IrType::U8:
            case IrType::U16:
            case IrType::U32:
            case IrType::U64:
            case IrType::F32: return true;
            default: return false;
        }
    }
    IrValue& inst = *value;
    if (!requireUniform && _validatedDependencies.contains(&inst)) return true;
    const auto activeMask = _activeMask;
    if (requireUniform && !_cyclicValues.contains(&inst)) {
        const auto context = _validatedUniform.find(activeMask);
        if (context != _validatedUniform.end() && context->second.contains(&inst)) return true;
    }
    if (!_visiting.insert(&inst).second) {
        return !requireUniform;
    }
    const auto finish = [&](bool valid) {
        _visiting.erase(&inst);
        if (valid) {
            if (requireUniform) {
                if (!_cyclicValues.contains(&inst)) _validatedUniform[activeMask].insert(&inst);
            } else _validatedDependencies.insert(&inst);
        }
        return valid;
    };
    const auto op = inst.Opcode();
    if (op == IrOpcode::ReadConst) {
        IrValue* slot = inst.ArgumentCount() == 2 ? inst.Argument(1)->Resolve() : nullptr;
        if (inst.ArgumentCount() != 2 || inst.Argument(0)->Resolve()->Opcode() == IrOpcode::Void || inst.Argument(0)->Resolve()->Opcode() != IrOpcode::GetSrtResource || slot == nullptr || !slot->HasImmediate() || slot->Type() != IrType::U32 || slot->ImmediateU32() >= _program.srtReads.size()) {
            return finish(false);
        }
        if (_type == RuntimeValueType::Integer) {
            const auto activeMask = _activeMask;
            _activeMask = nullptr;
            const bool valid = Validate(_program.srtReads[slot->ImmediateU32()].value);
            _activeMask = activeMask;
            if (!valid) return finish(false);
        }
    }
    if (!requireUniform) return finish(ValidateArguments(inst, false));
    if (_activeMask != nullptr && IsRuntimeSelect(op) && inst.ArgumentCount() == 3 && inst.Argument(0)->Resolve() == _activeMask) {
        if (_type == RuntimeValueType::Integer && !Validate(inst.Argument(2), false)) {
            return finish(false);
        }
        return finish(Validate(inst.Argument(1)));
    }
    if (op == IrOpcode::UndefU1 || op == IrOpcode::UndefU8 || op == IrOpcode::UndefU16 || op == IrOpcode::UndefU32 || op == IrOpcode::UndefU64 || op == IrOpcode::Void) {
        return finish(false);
    }
    if (op == IrOpcode::GetUserData) {
        if (inst.ArgumentCount() != 1 || inst.Argument(0)->Type() != IrType::ScalarReg) {
            return finish(false);
        }
        const auto reg = RegIndex(static_cast<ScalarReg>(inst.Argument(0)->Register().index));
        if (reg < _program.userDataBase || reg - _program.userDataBase >= _program.userDataCount) {
            return finish(false);
        }
        return finish(true);
    }
    if (op == IrOpcode::GetShaderBase) {
        if (inst.ArgumentCount() != 0) {
            return finish(false);
        }
        return finish(true);
    }
    if (op == IrOpcode::Phi) {
        if (_type == RuntimeValueType::Integer && !ValidateArguments(inst, false)) {
            return finish(false);
        }
        IrValue* invariant = ResolveInvariantPhi(_program, &inst);
        if (invariant == nullptr) {
            return finish(false);
        }
        return finish(Validate(invariant));
    }
    if (op == IrOpcode::ReadFirstLane) {
        if (inst.ArgumentCount() != 2 || inst.Argument(0)->Type() != IrType::U32 || inst.Argument(1)->Type() != IrType::Bool) {
            return finish(false);
        }
        if (_type == RuntimeValueType::Integer && !Validate(inst.Argument(1), false)) {
            return finish(false);
        }
        const auto activeMask = _activeMask;
        _activeMask = inst.Argument(1)->Resolve();
        const bool valid = Validate(inst.Argument(0));
        _activeMask = activeMask;
        return finish(valid);
    }
    if (op == IrOpcode::GetSrtResource) {
        if (inst.ArgumentCount() != 0) {
            return finish(false);
        }
        return finish(true);
    }
    if (op == IrOpcode::LoadAddressU32 || op == IrOpcode::ReadConstBuffer) {
        const auto expected = op == IrOpcode::LoadAddressU32 ? IrOpcode::GetAddressResource : IrOpcode::GetBufferResource;
        IrValue* handle = inst.ArgumentCount() != 0 ? inst.Argument(0)->Resolve() : nullptr;
        if (!IsRawRead(_program, inst) || handle == nullptr || handle->Opcode() != expected) {
            return finish(false);
        }
    } else if (op == IrOpcode::CompositeExtractU64) {
        IrValue* index = inst.ArgumentCount() == 2 ? inst.Argument(1)->Resolve() : nullptr;
        if (index == nullptr || !index->HasImmediate() || index->Type() != IrType::U32 || index->ImmediateU32() >= 2u) {
            return finish(false);
        }
    } else if (op == IrOpcode::CompositeExtractU32x2) {
        IrValue* source = inst.ArgumentCount() == 2 ? inst.Argument(0)->Resolve() : nullptr;
        IrValue* index = inst.ArgumentCount() == 2 ? inst.Argument(1)->Resolve() : nullptr;
        if (source == nullptr || index == nullptr || !index->HasImmediate() || index->Type() != IrType::U32 || index->ImmediateU32() >= 2u || (source->Opcode() != IrOpcode::CompositeConstructU32x2 && source->Opcode() != IrOpcode::IAddCarry32)) {
            return finish(false);
        }
    }
    if (IsDescriptorHandle(op)) {
        std::size_t expected = 4u;
        if (op == IrOpcode::GetImageResource) {
            expected = 8u;
        } else if (op == IrOpcode::GetAddressResource) {
            expected = 2u;
        }
        if (inst.ArgumentCount() != expected) {
            return finish(false);
        }
    } else if (op != IrOpcode::ReadConst && op != IrOpcode::ReadConstBuffer && op != IrOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
        return finish(false);
    }
    return finish(ValidateArguments(inst, true));
}

}
