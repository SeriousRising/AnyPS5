#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ShaderRecompiler {

std::uint32_t BdaConstant(SpirvEmitterState& state, std::uint64_t value) {
    return state.module.Constant(spv::OpConstant, TypeScalarU64(state), static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32u));
}

std::uint32_t BdaWord(SpirvEmitterState& state, std::uint32_t variable, std::uint32_t index) {
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer, variable, ConstantU32(state, 0u), index);
    return pointer;
}

std::uint32_t BdaLoadWord(SpirvEmitterState& state, std::uint32_t index) {
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), value, BdaWord(state, state.bdaPagetableVariable, index));
    return value;
}

std::uint32_t BdaAddressFromWords(SpirvEmitterState& state, std::uint32_t low, std::uint32_t high) {
    const auto words = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 2), words, low, high);
    return Unary(state, spv::OpBitcast, TypeScalarU64(state), words);
}

std::uint32_t BdaLoadAddress(SpirvEmitterState& state, std::uint32_t index) {
    const auto low = Unary(state, spv::OpUConvert, TypeScalarU64(state), BdaLoadWord(state, index));
    const auto high = Unary(state, spv::OpUConvert, TypeScalarU64(state), BdaLoadWord(state, Binary(state, spv::OpIAdd, TypeU32(state), index, ConstantU32(state, 1u))));
    return Binary(state, spv::OpBitwiseOr, TypeScalarU64(state), low, Binary(state, spv::OpShiftLeftLogical, TypeScalarU64(state), high, BdaConstant(state, 32u)));
}

namespace {

// Publishes a fault in the fault buffer, unless another invocation's is there already.
void EmitBdaFaultPublication(SpirvEmitterState& state, std::uint32_t reason, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction) {
    const auto pointer = BdaWord(state, state.faultBufferVariable, ConstantU32(state, 0u));
    const auto previous = state.module.AllocateId();
    const auto scope = ConstantU32(state, spv::ScopeDevice);
    const auto relaxed = ConstantU32(state, spv::MemorySemanticsMaskNone);
    state.module.AddFunction(spv::OpAtomicCompareExchange, TypeU32(state), previous, pointer, scope, relaxed, relaxed, ConstantU32(state, static_cast<std::uint32_t>(BdaAbi::FaultState::Writing)), ConstantU32(state, 0u));
    const auto won = Binary(state, spv::OpIEqual, TypeBool(state), previous, ConstantU32(state, 0u));
    EmitIfCondition(state, won, [&] {
        const auto store = [&](std::uint32_t index, std::uint32_t value) {
            state.module.AddFunction(spv::OpStore, BdaWord(state, state.faultBufferVariable, ConstantU32(state, index)), value);
        };
        store(1, reason);
        store(2, Unary(state, spv::OpUConvert, TypeU32(state), address));
        store(3, Unary(state, spv::OpUConvert, TypeU32(state), Binary(state, spv::OpShiftRightLogical, TypeScalarU64(state), address, BdaConstant(state, 32u))));
        store(4, bytes);
        store(5, ConstantU32(state, static_cast<std::uint32_t>(state.program.Resources().stage)));
        store(6, instruction);
        const auto mismatch = static_cast<std::uint32_t>(BdaAbi::FaultReason::RangeCacheMismatch);
        store(7, state.bdaRangeCacheCheck ? Select(state, TypeU32(state), Binary(state, spv::OpIEqual, TypeBool(state), reason, ConstantU32(state, mismatch)), ConstantU32(state, static_cast<std::uint32_t>(state.program.Resources().shaderHash)), ConstantU32(state, 0u)) : ConstantU32(state, 0u));
        const auto release = ConstantU32(state, spv::MemorySemanticsReleaseMask | spv::MemorySemanticsUniformMemoryMask);
        state.module.AddFunction(spv::OpAtomicStore, pointer, scope, release, ConstantU32(state, static_cast<std::uint32_t>(BdaAbi::FaultState::Ready)));
    });
}

}

// A program for the direct table keeps its first fault in Private variables and publishes it where
// the invocation ends (PublishBdaFault): every helper call is inlined, and the publication at each
// read site was most of what a site cost. The fault published is the one that was recorded first,
// as before; another invocation's may now reach the buffer earlier.
void DefineBdaFaultFunction(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    state.bdaFaultFunction = state.module.AllocateId();
    state.module.AddName(state.bdaFaultFunction, "record_bda_fault");
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), state.bdaFaultFunction, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, TypeVoid(state), u32, u64, u32, u32));
    const std::array<std::uint32_t, 4> types{u32, u64, u32, u32};
    std::array<std::uint32_t, 4> parameters{};
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        parameters[index] = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionParameter, types[index], parameters[index]);
    }
    EmitLabel(state, state.module.AllocateId());
    if (!state.bdaDirect) {
        EmitBdaFaultPublication(state, parameters[0], parameters[1], parameters[2], parameters[3]);
        state.module.AddFunction(spv::OpReturn);
        state.module.AddFunction(spv::OpFunctionEnd);
        return;
    }
    for (std::size_t index = 0; index < parameters.size(); ++index) state.bdaPendingFault[index] = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, types[index]), spv::StorageClassPrivate);
    const auto load = [&](std::size_t index) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, types[index], value, state.bdaPendingFault[index]);
        return value;
    };
    EmitIfCondition(state, Binary(state, spv::OpIEqual, TypeBool(state), load(0), ConstantU32(state, 0u)), [&] {
        for (std::size_t index = 0; index < parameters.size(); ++index) state.module.AddFunction(spv::OpStore, state.bdaPendingFault[index], parameters[index]);
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.bdaPublishFunction = state.module.AllocateId();
    state.module.AddName(state.bdaPublishFunction, "publish_bda_fault");
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), state.bdaPublishFunction, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, TypeVoid(state)));
    EmitLabel(state, state.module.AllocateId());
    const auto recorded = load(0);
    EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), recorded, ConstantU32(state, 0u)), [&] {
        // A direct dword read left its base address, immediate and instruction (BdaDirectReadFault):
        // what failed is worked out here, once, with a walk of its own.
        const auto boolean = TypeBool(state);
        const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
        const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
        const auto both = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalAnd, boolean, left, right); };
        const auto either = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalOr, boolean, left, right); };
        const auto reasonOf = [&](BdaAbi::FaultReason reason) { return constant(static_cast<std::uint32_t>(reason)); };
        const auto base = load(1);
        const auto immediate = load(2);
        const auto offsetBy = Unary(state, spv::OpSConvert, u64, immediate);
        const auto address = binary(spv::OpIAdd, u64, base, offsetBy);
        const auto leaf = EmitBdaDirectWalk(state, address);
        const auto room = Select(state, u32, binary(spv::OpULessThan, boolean, leaf.offset, leaf.bytes), binary(spv::OpISub, u32, leaf.bytes, leaf.offset), constant(0u));
        // A wrapped address is below a positive immediate, and at or above 2^64 minus a negative one's size.
        const auto wrapped = both(binary(spv::OpINotEqual, boolean, immediate, constant(0u)), binary(spv::OpLogicalNotEqual, boolean, binary(spv::OpSLessThan, boolean, immediate, constant(0u)), binary(spv::OpULessThan, boolean, address, offsetBy)));
        const auto overflow = binary(spv::OpUGreaterThan, boolean, address, BdaConstant(state, std::numeric_limits<std::uint64_t>::max() - 4u));
        const auto invalid = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, invalid, state.bdaTableInvalid);
        const auto broken = either(overflow, binary(spv::OpINotEqual, boolean, invalid, constant(0u)));
        const auto split = both(binary(spv::OpINotEqual, boolean, room, constant(0u)), binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, leaf.flags, constant(BdaAbi::DirectFollowed)), constant(0u)));
        // In the order the byte path met them: the immediate, the dword's wrap, the table, the bytes.
        auto reason = Select(state, u32, split, reasonOf(BdaAbi::FaultReason::Split), reasonOf(BdaAbi::FaultReason::Unmapped));
        reason = Select(state, u32, broken, reasonOf(BdaAbi::FaultReason::InvalidTable), reason);
        reason = Select(state, u32, either(wrapped, overflow), reasonOf(BdaAbi::FaultReason::Overflow), reason);
        const auto whole = either(broken, split);
        const auto failed = Select(state, u64, wrapped, base, binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, Select(state, u32, whole, constant(0u), room))));
        const auto bytes = Select(state, u32, wrapped, constant(0u), Select(state, u32, either(overflow, both(split, Unary(state, spv::OpLogicalNot, boolean, broken))), constant(4u), constant(1u)));
        const auto direct = binary(spv::OpIEqual, boolean, recorded, constant(BdaDirectReadFault));
        EmitBdaFaultPublication(state, Select(state, u32, direct, reason, recorded), Select(state, u64, direct, failed, base), Select(state, u32, direct, bytes, immediate), load(3));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
}

void PublishBdaFault(SpirvEmitterState& state) {
    if (state.bdaPublishFunction != 0u) state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaPublishFunction);
}

void RecordBdaFault(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason) {
    if (state.bdaFaultFunction == 0) throw std::runtime_error("BDA fault function is missing");
    state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaFaultFunction, ConstantU32(state, static_cast<std::uint32_t>(reason)), address, bytes, instruction);
}

void ReturnBdaFailureIf(SpirvEmitterState& state, std::uint32_t condition, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason) {
    const auto failed = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
    EmitLabel(state, failed);
    RecordBdaFault(state, address, bytes, instruction, reason);
    state.module.AddFunction(spv::OpReturnValue, BdaConstant(state, 0u));
    EmitLabel(state, next);
}

namespace {

std::uint32_t LoadBdaStopped(SpirvEmitterState& state) {
    const auto stopped = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeBool(state), stopped, state.bdaStopped);
    return stopped;
}

void EndBdaInvocationIf(SpirvEmitterState& state, std::uint32_t condition) {
    const auto failed = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
    EmitLabel(state, failed);
    PublishBdaFault(state);
    if (state.bdaStopValue != 0) {
        state.module.AddFunction(spv::OpReturnValue, state.bdaStopValue);
    } else {
        if (OrderedPixelShader(state)) {
            state.module.AddFunction(spv::OpEndInvocationInterlockEXT);
        }
        state.module.AddFunction(state.program.Resources().stage == IrShaderStage::Pixel ? spv::OpKill : spv::OpReturn);
    }
    EmitLabel(state, next);
}

}

void StopBdaInvocationIf(SpirvEmitterState& state, std::uint32_t condition) {
    if (!state.bdaStopsInvocations) return;
    if (!state.bdaStopDeferred) {
        EndBdaInvocationIf(state, condition);
        return;
    }
    state.module.AddFunction(spv::OpStore, state.bdaStopped, Binary(state, spv::OpLogicalOr, TypeBool(state), LoadBdaStopped(state), condition));
}

void EndStoppedBdaInvocation(SpirvEmitterState& state) {
    if (state.bdaStopsInvocations && state.bdaStopDeferred) EndBdaInvocationIf(state, LoadBdaStopped(state));
}

std::uint32_t UnlessBdaStopped(SpirvEmitterState& state, std::uint32_t condition) {
    if (!state.bdaStopDeferred) return condition;
    return Binary(state, spv::OpLogicalAnd, TypeBool(state), condition, Unary(state, spv::OpLogicalNot, TypeBool(state), LoadBdaStopped(state)));
}

void EmitUnlessBdaStopped(SpirvEmitterState& state, const std::function<void()>& emit) {
    if (!state.bdaStopDeferred) {
        emit();
        return;
    }
    EmitIfCondition(state, UnlessBdaStopped(state, ConstantBool(state, true)), emit);
}

bool BdaStoppedInvocationSkips(const IrValue& instruction) {
    const auto opcode = instruction.Opcode();
    return instruction.Type() == IrType::Void && (BufferAccessOf(opcode) == BufferAccess::Write || ImageOpcodeInfoOf(opcode).access == ImageAccess::Write || SharedAccessOf(opcode) == SharedAccess::Write);
}

bool BdaStopCanBeDeferred(const SpirvEmitterState& state) {
    static const bool perSite = std::getenv("APS5_BDA_STOP_PER_SITE") != nullptr;
    const auto& program = state.program;
    const auto stage = program.Resources().stage;
    if (perSite || !program.Info().usesDma || !state.bdaStopsInvocations || (stage != IrShaderStage::Pixel && stage != IrShaderStage::Compute)) return false;
    // APS5_TRACE_BDA=1 names what keeps a program on one stop per read site.
    static const bool traced = std::getenv("APS5_TRACE_BDA") != nullptr;
    // Untraced, the first reason ends the search; traced, the line names every one, so that a program
    // is not found refused for the next reason once the first is lifted.
    std::set<std::string> reasons;
    const auto refuse = [&](std::string_view reason) { reasons.emplace(reason); };
    const auto refused = [&] {
        if (reasons.empty()) return false;
        std::string text;
        for (const auto& reason : reasons) text += (text.empty() ? "" : ", ") + reason;
        if (traced) std::fprintf(stderr, "[gpu] shader 0x%llx stage %u ends a faulting invocation at each address read: %s\n", static_cast<unsigned long long>(program.Resources().shaderHash), static_cast<unsigned>(stage), text.c_str());
        return true;
    };
    if (state.laneCount != 1u) refuse("two lanes per invocation");
    for (const auto* block : program.BlockOrder()) {
        for (const auto* instruction : block->Instructions()) {
            if (!traced && !reasons.empty()) return false;
            const auto opcode = instruction->Opcode();
            switch (opcode) {
                case IrOpcode::ReadLane:
                case IrOpcode::WriteLane:
                case IrOpcode::Permlane16U32:
                case IrOpcode::SwizzleU32:
                case IrOpcode::PermuteU32:
                case IrOpcode::BpermuteU32:
                    refuse(IrOpcodeName(opcode));
                    continue;
                default:
                    break;
            }
            if (!IrOpcodeHasSideEffects(opcode) || AddressOpcodeInfoOf(opcode).access != AddressAccess::None) continue;
            if (BdaStoppedInvocationSkips(*instruction)) continue;
            switch (opcode) {
                case IrOpcode::SetRegister:
                case IrOpcode::Branch:
                case IrOpcode::BranchConditional:
                case IrOpcode::Loop:
                case IrOpcode::LoopMerge:
                case IrOpcode::Return:
                case IrOpcode::Unreachable:
                case IrOpcode::Reference:
                case IrOpcode::ReferenceU32:
                case IrOpcode::SetThreadBitScalarRegister:
                case IrOpcode::SetScalarMaskTag:
                case IrOpcode::SetScalarRegister:
                case IrOpcode::SetVectorRegister:
                case IrOpcode::SetGotoVariable:
                case IrOpcode::SetScc:
                case IrOpcode::SetExec:
                case IrOpcode::SetExecLo:
                case IrOpcode::SetExecHi:
                case IrOpcode::SetVcc:
                case IrOpcode::SetVccLo:
                case IrOpcode::SetVccHi:
                case IrOpcode::SetM0:
                case IrOpcode::Waitcnt:
                case IrOpcode::Sendmsg:
                case IrOpcode::TtraceData:
                case IrOpcode::InstPrefetch:
                    continue;
                case IrOpcode::Discard:
                case IrOpcode::SetAttribute:
                    if (stage != IrShaderStage::Pixel) refuse(IrOpcodeName(opcode));
                    continue;
                default:
                    refuse(IrOpcodeName(opcode));
            }
        }
    }
    return !refused();
}

}
