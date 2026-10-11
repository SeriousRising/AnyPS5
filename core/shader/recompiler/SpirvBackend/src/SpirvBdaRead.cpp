#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "Optimization/SrtWalker.hpp"
#include <algorithm>
#include <compare>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

namespace {

void EmitBdaOverflowCheck(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction) {
    const auto overflow = Binary(state, spv::OpUGreaterThan, TypeBool(state), address, BdaConstant(state, std::numeric_limits<std::uint64_t>::max() - bytes));
    EmitIfCondition(state, overflow, [&] { RecordBdaFault(state, address, ConstantU32(state, bytes), instruction, BdaAbi::FaultReason::Overflow); });
    StopBdaInvocationIf(state, overflow);
}

bool BdaCoherent(const SpirvEmitterState& state, const IrValue& inst) {
    const auto index = inst.Flags<MemoryFlags>().index;
    const auto& memory = state.program.Resources().memoryInfo;
    return index < memory.size() && memory[index].coherent;
}

std::uint32_t BdaAccessMask(bool coherent) {
    return spv::MemoryAccessAlignedMask | (coherent ? spv::MemoryAccessVolatileMask : 0u);
}

std::uint32_t BdaAccessMask(const SpirvEmitterState& state, const IrValue& inst) {
    return BdaAccessMask(BdaCoherent(state, inst));
}

std::uint32_t EmitBdaByte(SpirvEmitterState& state, std::uint32_t guest, std::uint32_t instruction, std::uint32_t accessMask) {
    const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
    const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
    const auto physical = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaPointerFunction, guest, ConstantU32(state, 1u), instruction);
    // Unmapped bytes read as zero; the fault is recorded and every invocation reaches the
    // program's barriers, or runs on with its stop pending. The load is guarded also where the
    // invocation ends at the read: a pixel's OpKill does not end it on every driver (RADV compiles
    // it to a demotion and runs the rest of the program for the helper).
    const auto mapped = Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u));
    const auto before = state.currentLabel;
    const auto loadLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, mapped, loadLabel, merge);
    EmitLabel(state, loadLabel);
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, physical);
    const auto loaded = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, accessMask, 1u);
    const auto widened = Unary(state, spv::OpUConvert, TypeU32(state), loaded);
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, TypeU32(state), value, widened, loadLabel, ConstantU32(state, 0u), before);
    StopBdaInvocationIf(state, Unary(state, spv::OpLogicalNot, TypeBool(state), mapped));
    return value;
}

std::uint32_t EmitBdaBytes(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t accessMask) {
    auto result = ConstantU32(state, 0u);
    for (std::uint32_t byte = 0; byte < bytes; ++byte) {
        const auto guest = Binary(state, spv::OpIAdd, TypeScalarU64(state), address, BdaConstant(state, byte));
        const auto value = EmitBdaByte(state, guest, instruction, accessMask);
        result = Binary(state, spv::OpBitwiseOr, TypeU32(state), result, Binary(state, spv::OpShiftLeftLogical, TypeU32(state), value, ConstantU32(state, byte * 8u)));
    }
    return result;
}

// The dwords of a wide load that the program extracts (all of them when the composite is used
// whole). Dead-code elimination has dropped the unused extracts, as it dropped unused dword loads
// when every dword was its own instruction.
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

void EmitBdaStoreAt(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t pointerType, std::uint32_t value, std::uint32_t instruction, std::uint32_t accessMask) {
    const auto physical = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaWritePointerFunction, address, ConstantU32(state, bytes), instruction);
    EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), [&] {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, pointerType, pointer, physical);
        state.module.AddFunction(spv::OpStore, pointer, value, accessMask, bytes);
        state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, address);
    });
}

std::uint32_t BdaAddressUnaligned(SpirvEmitterState& state, std::uint32_t address) {
    return Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), address, BdaConstant(state, 3u)), BdaConstant(state, 0u));
}

}

// (u64 address, u32 instruction) -> (u32 dword, u32 stopped). One loop holds the only lookup and the
// only fault site: step 0 probes the whole dword, as EmitBdaDwordReads does, and loads it when it
// lies in one range at a 4-aligned device address; step 1 is the byte path's overflow check and
// steps 2 to 5 its bytes, each fault recorded in that order. Without the probe the loop starts at
// step 1.
// `shared` adds two parameters and a step before the others: (u64 address, u32 instruction, u64 hull
// address, u32 hull bytes) -> (u64 dword | stopped << 32, u64 the hull's device address or 1). The
// hull is tried when its bytes are not 0, with the same lookup, asked for the whole span; found at
// a 4-aligned device address, the dword is loaded at its place in the hull (the address lies in the
// hull at a multiple of 4 from its start) and the loop ends as after a successful probe. Nothing is
// recorded for a hull that fails; the remaining steps then run as without it.
static std::uint32_t DefineBdaDwordRead(SpirvEmitterState& state, const std::string& name, bool probe, bool stop, bool coherent, bool shared = false) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto pair = shared ? state.module.Type(spv::OpTypeVector, u64, 2u) : TypeU32Vector(state, 2);
    const std::uint32_t skew = shared ? 1u : 0u;
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalAnd, boolean, left, right); };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    state.module.AddFunction(spv::OpFunction, pair, function, spv::FunctionControlDontInlineMask, shared ? state.module.Type(spv::OpTypeFunction, pair, u64, u32, u64, u32) : state.module.Type(spv::OpTypeFunction, pair, u64, u32));
    const auto address = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    const auto hullAddress = shared ? state.module.AllocateId() : 0u;
    const auto hullBytes = shared ? state.module.AllocateId() : 0u;
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    if (shared) {
        state.module.AddFunction(spv::OpFunctionParameter, u64, hullAddress);
        state.module.AddFunction(spv::OpFunctionParameter, u32, hullBytes);
    }
    EmitLabel(state, state.module.AllocateId());
    const auto variable = [&](std::uint32_t initial) {
        const auto id = state.module.AllocateId();
        state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, u32), id, spv::StorageClassFunction);
        return std::pair{id, initial};
    };
    const auto step = variable(probe ? 0u : 1u);
    const auto value = variable(0u);
    const auto stopped = variable(0u);
    const auto done = variable(0u);
    const auto hull = shared ? state.module.AllocateId() : 0u;
    if (shared) state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, u64), hull, spv::StorageClassFunction);
    for (const auto& [id, initial] : {step, value, stopped, done}) state.module.AddFunction(spv::OpStore, id, constant(initial));
    if (shared) {
        state.module.AddFunction(spv::OpStore, step.first, Select(state, u32, binary(spv::OpINotEqual, boolean, hullBytes, constant(0u)), constant(0u), constant(1u)));
        state.module.AddFunction(spv::OpStore, hull, BdaConstant(state, 0u));
    }
    const auto load = [&](std::uint32_t pointer) {
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, loaded, pointer);
        return loaded;
    };
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto current = load(step.first);
    const auto running = both(binary(spv::OpULessThan, boolean, current, constant(6u + skew)), binary(spv::OpIEqual, boolean, load(done.first), constant(0u)));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, running, body, merge);
    EmitLabel(state, body);
    const auto hulling = shared ? binary(spv::OpIEqual, boolean, current, constant(0u)) : 0u;
    const auto probing = binary(spv::OpIEqual, boolean, current, constant(skew));
    const auto checking = binary(spv::OpIEqual, boolean, current, constant(1u + skew));
    const auto reading = binary(spv::OpUGreaterThan, boolean, current, constant(1u + skew));
    const auto byte = Select(state, u32, reading, binary(spv::OpISub, u32, current, constant(2u + skew)), constant(0u));
    auto guest = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, byte));
    auto bytes = Select(state, u32, reading, constant(1u), constant(4u));
    if (shared) {
        guest = Select(state, u64, hulling, hullAddress, guest);
        bytes = Select(state, u32, hulling, hullBytes, bytes);
    }
    const auto looked = shared ? BdaSpanLookup(state, guest, bytes, Select(state, u32, hulling, constant(BdaAbi::Read | BdaLookupSpan), constant(BdaAbi::Read))) : BdaLookup(state, guest, bytes, BdaAbi::Read);
    const auto overflow = binary(spv::OpUGreaterThan, boolean, address, BdaConstant(state, std::numeric_limits<std::uint64_t>::max() - 4u));
    // A cache check failure is recorded at whichever step meets it, the probe included, the hull excepted.
    auto mismatched = state.bdaRangeCacheCheck ? binary(spv::OpIEqual, boolean, looked.reason, constant(static_cast<std::uint32_t>(BdaAbi::FaultReason::RangeCacheMismatch))) : 0u;
    if (mismatched != 0u && shared) mismatched = both(mismatched, Unary(state, spv::OpLogicalNot, boolean, hulling));
    const auto reported = mismatched != 0u ? binary(spv::OpLogicalOr, boolean, reading, mismatched) : reading;
    const auto reason = Select(state, u32, reported, looked.reason, Select(state, u32, both(checking, overflow), constant(static_cast<std::uint32_t>(BdaAbi::FaultReason::Overflow)), constant(0u)));
    const auto failed = binary(spv::OpINotEqual, boolean, reason, constant(0u));
    EmitIfCondition(state, failed, [&] {
        state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaFaultFunction, reason, guest, bytes, instruction);
        if (!stop) return;
        state.module.AddFunction(spv::OpStore, stopped.first, constant(1u));
        state.module.AddFunction(spv::OpStore, done.first, constant(1u));
    });
    const auto found = binary(spv::OpIEqual, boolean, looked.reason, constant(0u));
    if (probe) {
        const auto aligned = binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u64, looked.address, BdaConstant(state, 3u)), BdaConstant(state, 0u));
        EmitIfCondition(state, both(shared ? binary(spv::OpLogicalOr, boolean, hulling, probing) : probing, both(found, aligned)), [&] {
            auto device = looked.address;
            if (shared) {
                device = binary(spv::OpIAdd, u64, device, Select(state, u64, hulling, binary(spv::OpISub, u64, address, hullAddress), BdaConstant(state, 0u)));
                state.module.AddFunction(spv::OpStore, hull, Select(state, u64, hulling, looked.address, BdaConstant(state, 0u)));
            }
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, device);
            const auto loaded = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, u32, loaded, pointer, BdaAccessMask(coherent), 4u);
            state.module.AddFunction(spv::OpStore, value.first, loaded);
            state.module.AddFunction(spv::OpStore, done.first, constant(1u));
        });
    }
    EmitIfCondition(state, both(reading, found), [&] {
        const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType), pointer, looked.address);
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, BdaAccessMask(coherent), 1u);
        const auto shifted = binary(spv::OpShiftLeftLogical, u32, Unary(state, spv::OpUConvert, u32, loaded), binary(spv::OpIMul, u32, byte, constant(8u)));
        state.module.AddFunction(spv::OpStore, value.first, binary(spv::OpBitwiseOr, u32, load(value.first), shifted));
    });
    state.module.AddFunction(spv::OpStore, step.first, binary(spv::OpIAdd, u32, current, constant(1u)));
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto result = state.module.AllocateId();
    if (shared) {
        const auto wide = [&](std::uint32_t word) { return Unary(state, spv::OpUConvert, u64, word); };
        const auto held = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u64, held, hull);
        const auto packed = binary(spv::OpBitwiseOr, u64, wide(load(value.first)), binary(spv::OpShiftLeftLogical, u64, wide(load(stopped.first)), BdaConstant(state, 32u)));
        state.module.AddFunction(spv::OpCompositeConstruct, pair, result, packed, Select(state, u64, binary(spv::OpINotEqual, boolean, held, BdaConstant(state, 0u)), held, BdaConstant(state, 1u)));
    } else {
        state.module.AddFunction(spv::OpCompositeConstruct, pair, result, load(value.first), load(stopped.first));
    }
    state.module.AddFunction(spv::OpReturnValue, result);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

// (u64 address, u32 bytes, u64 hull address, u32 hull bytes) -> (u64 device address of the bytes, or
// 0; u64 the hull's device address or 1). probe_bda_pointer for a wide read of a shared base: one
// lookup tries the hull when its bytes are not 0 and, when that fails, the read's own span.
static std::uint32_t DefineBdaSharedProbe(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto pair = state.module.Type(spv::OpTypeVector, u64, 2u);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalAnd, boolean, left, right); };
    const auto zero = BdaConstant(state, 0u);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "probe_bda_shared");
    state.module.AddFunction(spv::OpFunction, pair, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, pair, u64, u32, u64, u32));
    std::array<std::uint32_t, 4> parameters{};
    for (std::uint32_t index = 0; index < parameters.size(); ++index) {
        parameters[index] = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionParameter, index % 2u == 0u ? u64 : u32, parameters[index]);
    }
    const auto [address, bytes, hullAddress, hullBytes] = parameters;
    EmitLabel(state, state.module.AllocateId());
    const auto variable = [&](std::uint32_t type) {
        const auto id = state.module.AllocateId();
        state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, type), id, spv::StorageClassFunction);
        return id;
    };
    const auto step = variable(u32);
    const auto device = variable(u64);
    const auto hull = variable(u64);
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, loaded, pointer);
        return loaded;
    };
    state.module.AddFunction(spv::OpStore, step, Select(state, u32, binary(spv::OpINotEqual, boolean, hullBytes, constant(0u)), constant(0u), constant(1u)));
    state.module.AddFunction(spv::OpStore, device, zero);
    state.module.AddFunction(spv::OpStore, hull, zero);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto current = load(u32, step);
    const auto running = both(binary(spv::OpULessThan, boolean, current, constant(2u)), binary(spv::OpIEqual, boolean, load(u64, device), zero));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, running, body, merge);
    EmitLabel(state, body);
    const auto hulling = binary(spv::OpIEqual, boolean, current, constant(0u));
    const auto looked = BdaSpanLookup(state, Select(state, u64, hulling, hullAddress, address), Select(state, u32, hulling, hullBytes, bytes), Select(state, u32, hulling, constant(BdaAbi::Read | BdaLookupSpan), constant(BdaAbi::Read)));
    // The lookup's address is 0 when it fails.
    const auto aligned = binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u64, looked.address, BdaConstant(state, 3u)), zero);
    const auto held = Select(state, u64, both(hulling, aligned), looked.address, zero);
    state.module.AddFunction(spv::OpStore, hull, held);
    const auto within = Select(state, u64, binary(spv::OpINotEqual, boolean, held, zero), binary(spv::OpIAdd, u64, held, binary(spv::OpISub, u64, address, hullAddress)), zero);
    state.module.AddFunction(spv::OpStore, device, Select(state, u64, hulling, within, looked.address));
    state.module.AddFunction(spv::OpStore, step, binary(spv::OpIAdd, u32, current, constant(1u)));
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto found = load(u64, hull);
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, pair, result, load(u64, device), Select(state, u64, binary(spv::OpINotEqual, boolean, found, zero), found, BdaConstant(state, 1u)));
    state.module.AddFunction(spv::OpReturnValue, result);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void DefineBdaDwordReadFunctions(SpirvEmitterState& state) {
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    for (const bool stop : {false, true}) {
        for (const bool coherent : {false, true}) {
            if ((stop && !state.bdaStopsInvocations) || (coherent && !coherentAccesses)) continue;
            const auto suffix = std::string(stop ? "_stop" : "") + (coherent ? "_coherent" : "");
            state.bdaDwordReadFunctions[stop][coherent] = DefineBdaDwordRead(state, "read_bda_dword_bytes" + suffix, false, stop, coherent);
            if (!BdaByteReadsForced()) state.bdaProbedDwordReadFunctions[stop][coherent] = DefineBdaDwordRead(state, "read_bda_dword" + suffix, true, stop, coherent);
            if (state.bdaSharedBase) state.bdaSharedDwordReadFunctions[stop][coherent] = DefineBdaDwordRead(state, "read_bda_dword_shared" + suffix, true, stop, coherent, true);
        }
    }
    if (state.bdaSharedBase) state.bdaSharedProbeFunction = DefineBdaSharedProbe(state);
}

// APS5_BDA_BYTE_READS=1 restores one lookup and one byte load per byte for every read.
bool BdaByteReadsForced() {
    static const bool forced = std::getenv("APS5_BDA_BYTE_READS") != nullptr;
    return forced;
}

std::uint32_t AddBdaAddress(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, bool subtract) {
    auto& state = ctx.state;
    const auto result = Binary(state, subtract ? spv::OpISub : spv::OpIAdd, TypeScalarU64(state), address, offset);
    const auto overflow = subtract ? Binary(state, spv::OpUGreaterThan, TypeBool(state), offset, address) : Binary(state, spv::OpULessThan, TypeBool(state), result, address);
    EmitIfCondition(state, overflow, [&] { RecordBdaFault(state, address, ConstantU32(state, 0u), ConstantU32(state, inst.Flags<MemoryFlags>().pc), BdaAbi::FaultReason::Overflow); });
    StopBdaInvocationIf(state, overflow);
    return result;
}

std::uint32_t AddBdaImmediate(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::int32_t immediate) {
    if (immediate == 0) {
        return address;
    }
    const auto magnitude = immediate < 0 ? -static_cast<std::int64_t>(immediate) : static_cast<std::int64_t>(immediate);
    return AddBdaAddress(ctx, inst, address, BdaConstant(ctx.state, static_cast<std::uint64_t>(magnitude)), immediate < 0);
}

void ValidateBdaTarget(const IrProgram& program, const SpirvTargetOptions& target) {
    if (!program.Info().usesDma) return;
    if (target.bdaAbiVersion != BdaAbi::Version) throw std::runtime_error("unsupported BDA ABI version");
    for (const auto capability : {spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess}) {
        if (std::find(target.supportedCapabilities.begin(), target.supportedCapabilities.end(), static_cast<std::uint32_t>(capability)) == target.supportedCapabilities.end()) throw std::runtime_error("BDA requires unsupported SPIR-V capability " + std::to_string(capability));
    }
    for (const auto extension : {"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"}) {
        if (std::find(target.supportedExtensions.begin(), target.supportedExtensions.end(), extension) == target.supportedExtensions.end()) throw std::runtime_error(std::string("BDA requires unsupported extension ") + extension);
    }
    if (program.Resources().stage == IrShaderStage::TessellationControl) throw std::runtime_error("BDA fault termination requires a barrier-safe tessellation-control execution protocol");
}

bool BdaInvocationsMayStop(const IrProgram& program) {
    for (const auto* block : program.BlockOrder()) {
        for (const auto* instruction : block->Instructions()) {
            if (instruction->Opcode() == IrOpcode::Barrier) return false;
        }
    }
    return true;
}

std::uint32_t EmitBdaRead(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bits) {
    auto& state = ctx.state;
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA read width");
    if (state.bdaPointerFunction == 0) ctx.Fail(inst, "BDA lookup function is missing");
    if (bits == 32u) return EmitBdaDwordReads(ctx, inst, address, 0u, 1u)[0];
    const auto instruction = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    EmitBdaOverflowCheck(state, address, bits / 8u, instruction);
    return EmitBdaBytes(state, address, bits / 8u, instruction, BdaAccessMask(state, inst));
}

void EmitBdaWrite(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits) {
    auto& state = ctx.state;
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto instruction = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA write width");
    if (bits != 32u) {
        EmitBdaOverflowCheck(state, address, bits / 8u, instruction);
        for (std::uint32_t byte = 0u; byte < bits / 8u; ++byte) {
            const auto guest = Binary(state, spv::OpIAdd, TypeScalarU64(state), address, BdaConstant(state, byte));
            const auto physical = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaWritePointerFunction, guest, ConstantU32(state, 1u), instruction);
            EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), [&] {
                const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
                const auto pointer = state.module.AllocateId();
                state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType), pointer, physical);
                const auto shifted = Binary(state, spv::OpShiftRightLogical, TypeU32(state), value, ConstantU32(state, byte * 8u));
                state.module.AddFunction(spv::OpStore, pointer, Unary(state, spv::OpUConvert, byteType, shifted), BdaAccessMask(state, inst), 1u);
                state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, guest);
            });
        }
        return;
    }
    const auto unaligned = BdaAddressUnaligned(state, address);
    EmitIfCondition(state, unaligned, [&] { RecordBdaFault(state, address, ConstantU32(state, 4u), instruction, BdaAbi::FaultReason::Unaligned); });
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), [&] {
        EmitBdaStoreAt(state, address, 4u, TypePhysicalU32Pointer(state), value, instruction, BdaAccessMask(state, inst));
    });
}

void EmitBdaStore(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits) {
    auto& state = ctx.state;
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA store width");
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto instruction = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    const auto accessMask = BdaAccessMask(state, inst);
    const auto storeBytes = [&] {
        const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
        const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
        for (std::uint32_t byte = 0; byte < bits / 8u; ++byte) {
            const auto guest = byte == 0u ? address : Binary(state, spv::OpIAdd, TypeScalarU64(state), address, BdaConstant(state, byte));
            const auto shifted = byte == 0u ? value : Binary(state, spv::OpShiftRightLogical, TypeU32(state), value, ConstantU32(state, byte * 8u));
            EmitBdaStoreAt(state, guest, 1u, bytePointer, Unary(state, spv::OpUConvert, byteType, shifted), instruction, accessMask);
        }
    };
    if (bits != 32u) {
        storeBytes();
        return;
    }
    const auto unaligned = BdaAddressUnaligned(state, address);
    EmitIfCondition(state, unaligned, storeBytes);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), [&] {
        EmitBdaStoreAt(state, address, 4u, TypePhysicalU32Pointer(state), value, instruction, accessMask);
    });
}

std::uint32_t EmitBdaAtomic(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bytes, const std::function<std::uint32_t(std::uint32_t)>& operation) {
    auto& state = ctx.state;
    if (bytes != 4u && bytes != 8u) ctx.Fail(inst, "unsupported BDA atomic width");
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto instruction = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    const auto type = bytes == 8u ? TypeScalarU64(state) : TypeU32(state);
    const auto zero = bytes == 8u ? BdaConstant(state, 0u) : ConstantU32(state, 0u);
    const auto unaligned = Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), address, BdaConstant(state, bytes - 1u)), BdaConstant(state, 0u));
    EmitIfCondition(state, unaligned, [&] { RecordBdaFault(state, address, ConstantU32(state, bytes), instruction, BdaAbi::FaultReason::Unaligned); });
    return EmitValueOrDefaultIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), type, zero, [&] {
        const auto physical = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaWritePointerFunction, address, ConstantU32(state, bytes), instruction);
        return EmitValueOrDefaultIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), type, zero, [&] {
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, type), pointer, physical);
            const auto old = operation(pointer);
            state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, address);
            return old;
        });
    });
}

// One probe of the span from the first to the last extracted dword replaces 4 lookups per dword (a
// Bink DC pass spent ~450 dependent table levels per iteration on them). The span is loaded
// dword-wise from the returned device address when it lies in one range and that address is
// 4-aligned; a span crossing ranges (the table is byte-granular, see tests/BdaExecution.cpp), an
// unaligned mirror or a wrapping span falls back to the byte path. That path repeats, per extracted
// dword in order, the address, overflow check and byte lookups of a separate dword load, so every
// fault is recorded exactly as one; the probe records none, and a span it accepts cannot fault.
std::array<std::uint32_t, 4> EmitBdaDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword, const BdaSharedTry* shared) {
    auto& state = ctx.state;
    if (dwords == 0u || dwords > 4u) ctx.Fail(inst, "unsupported BDA read width");
    if (state.bdaPointerFunction == 0) ctx.Fail(inst, "BDA lookup function is missing");
    const auto u64 = TypeScalarU64(state);
    const auto instruction = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    const auto used = everyDword ? (1u << dwords) - 1u : UsedBdaDwords(inst, dwords);
    const auto isUsed = [&](std::uint32_t dword) { return (used & (1u << dword)) != 0u; };
    std::array<std::uint32_t, 4> values{};
    std::uint32_t first = dwords;
    std::uint32_t last = 0;
    for (std::uint32_t dword = 0; dword < dwords; ++dword) {
        if (!isUsed(dword)) {
            values[dword] = ConstantU32(state, 0u);
            continue;
        }
        first = std::min(first, dword);
        last = dword;
    }
    if (used == 0u) return values;
    std::array<std::uint32_t, 4> addresses{};
    const auto dwordAddress = [&](std::uint32_t dword) { return AddBdaImmediate(ctx, inst, address, static_cast<std::int32_t>(offset + dword * 4u)); };
    const auto readBytes = [&](std::array<std::uint32_t, 4>& into) {
        const auto function = state.bdaDwordReadFunctions[state.bdaStopsInvocations][BdaCoherent(state, inst)];
        if (function == 0) ctx.Fail(inst, "BDA dword read function is missing");
        for (std::uint32_t dword = first; dword <= last; ++dword) {
            if (!isUsed(dword)) continue;
            if (dword != first) addresses[dword] = dwordAddress(dword);
            const auto read = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 2), read, function, addresses[dword], instruction);
            into[dword] = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), into[dword], read, 0u);
            if (!state.bdaStopsInvocations) continue;
            const auto stopped = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), stopped, read, 1u);
            StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, TypeBool(state), stopped, ConstantU32(state, 0u)));
        }
    };
    addresses[first] = dwordAddress(first);
    if (state.bdaProbeFunction == 0) {
        readBytes(values);
        return values;
    }
    // The hull is tried only while the base's variable is 0; the callee's answer replaces it.
    const auto sharedCall = [&](std::uint32_t function, std::uint32_t second) {
        if (function == 0) ctx.Fail(inst, "BDA shared read function is missing");
        const auto pair = state.module.Type(spv::OpTypeVector, u64, 2u);
        const auto hull = shared->back != 0u ? Binary(state, spv::OpISub, u64, addresses[first], BdaConstant(state, shared->back)) : addresses[first];
        const auto bytes = Select(state, TypeU32(state), Binary(state, spv::OpIEqual, TypeBool(state), shared->held, BdaConstant(state, 0u)), ConstantU32(state, shared->bytes), ConstantU32(state, 0u));
        const auto called = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, pair, called, function, addresses[first], second, hull, bytes);
        const auto held = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u64, held, called, 1u);
        state.module.AddFunction(spv::OpStore, shared->variable, held);
        const auto answer = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u64, answer, called, 0u);
        return answer;
    };
    if (first == last && shared != nullptr) {
        const auto read = sharedCall(state.bdaSharedDwordReadFunctions[state.bdaStopsInvocations][BdaCoherent(state, inst)], instruction);
        values[first] = Unary(state, spv::OpUConvert, TypeU32(state), read);
        if (state.bdaStopsInvocations) StopBdaInvocationIf(state, Binary(state, spv::OpUGreaterThan, TypeBool(state), read, BdaConstant(state, 0xffffffffu)));
        return values;
    }
    if (first == last) {
        const auto function = state.bdaProbedDwordReadFunctions[state.bdaStopsInvocations][BdaCoherent(state, inst)];
        if (function == 0) ctx.Fail(inst, "BDA dword read function is missing");
        const auto read = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 2), read, function, addresses[first], instruction);
        values[first] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), values[first], read, 0u);
        if (state.bdaStopsInvocations) {
            const auto stopped = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), stopped, read, 1u);
            StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, TypeBool(state), stopped, ConstantU32(state, 0u)));
        }
        return values;
    }
    auto physical = state.module.AllocateId();
    if (shared != nullptr) physical = sharedCall(state.bdaSharedProbeFunction, ConstantU32(state, (last - first + 1u) * 4u));
    else state.module.AddFunction(spv::OpFunctionCall, u64, physical, state.bdaProbeFunction, addresses[first], ConstantU32(state, (last - first + 1u) * 4u), instruction);
    const auto mapped = Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u));
    const auto aligned = Binary(state, spv::OpIEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, u64, physical, BdaConstant(state, 3u)), BdaConstant(state, 0u));
    const auto wide = Binary(state, spv::OpLogicalAnd, TypeBool(state), mapped, aligned);
    const auto wideLabel = state.module.AllocateId();
    const auto byteLabel = state.module.AllocateId();
    const auto byteExit = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, wide, wideLabel, byteLabel);
    EmitLabel(state, wideLabel);
    std::array<std::uint32_t, 4> wideValues{};
    for (std::uint32_t dword = first; dword <= last; ++dword) {
        if (!isUsed(dword)) continue;
        const auto element = dword == first ? physical : Binary(state, spv::OpIAdd, u64, physical, BdaConstant(state, (dword - first) * 4u));
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, element);
        wideValues[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), wideValues[dword], pointer, BdaAccessMask(state, inst), 4u);
    }
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, byteLabel);
    std::array<std::uint32_t, 4> byteValues{};
    readBytes(byteValues);
    state.module.AddFunction(spv::OpBranch, byteExit);
    EmitLabel(state, byteExit);
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    for (std::uint32_t dword = first; dword <= last; ++dword) {
        if (!isUsed(dword)) continue;
        values[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpPhi, TypeU32(state), values[dword], wideValues[dword], wideLabel, byteValues[dword], byteExit);
    }
    return values;
}

namespace {

// The dwords the read extracts, as EmitBdaDwordReads finds them: {first, last}; first > last when none.
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

}

// Why a member that skips its own path changes nothing. It runs only after a member of the same base,
// with the same base address (ResetBdaSharedBases), took its own path and found the hull in one
// range: every byte of the hull takes the search the hull's first byte took (BdaLookupSpan), so the
// member's own probe would select that range, find its span inside, at the hull's device address
// plus the same distance, 4-aligned as the hull is, and load the same dwords. The caller has checked
// `address` for overflow as for any read; the check a wide member skips is that of its first dword's
// immediate: when the hull's start wrapped, the member that translated wrapped too, has recorded its
// fault (the record keeps the first) and has stopped the invocation or flagged it, and when it did
// not, no address inside the hull wraps.
std::array<std::uint32_t, 4> EmitBdaSharedDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, const BdaSharedBase& base, std::int64_t begin, bool tries, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword) {
    auto& state = ctx.state;
    const auto u64 = TypeScalarU64(state);
    const auto [first, last] = ExtractedBdaDwords(inst, dwords, everyDword);
    if (first > last || base.variable == 0u || begin < base.first || static_cast<std::uint64_t>(begin - base.first) + (last - first + 1u) * 4u > base.bytes) ctx.Fail(inst, "BDA shared read lies outside its base's hull");
    const auto back = static_cast<std::uint32_t>(begin - base.first);
    const auto held = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u64, held, base.variable);
    const auto translated = Binary(state, spv::OpUGreaterThan, TypeBool(state), held, BdaConstant(state, 1u));
    const auto directLabel = state.module.AllocateId();
    const auto ownLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, translated, directLabel, ownLabel);
    EmitLabel(state, directLabel);
    std::array<std::uint32_t, 4> direct{};
    for (std::uint32_t dword = first; dword <= last; ++dword) {
        if (!everyDword && (UsedBdaDwords(inst, dwords) & (1u << dword)) == 0u) continue;
        const auto distance = back + (dword - first) * 4u;
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, distance != 0u ? Binary(state, spv::OpIAdd, u64, held, BdaConstant(state, distance)) : held);
        direct[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), direct[dword], pointer, BdaAccessMask(state, inst), 4u);
    }
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, ownLabel);
    const BdaSharedTry shared{base.variable, held, back, base.bytes};
    auto values = EmitBdaDwordReads(ctx, inst, address, offset, dwords, everyDword, tries ? &shared : nullptr);
    const auto ownExit = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    for (std::uint32_t dword = first; dword <= last; ++dword) {
        if (direct[dword] == 0u) continue;
        const auto own = values[dword];
        values[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpPhi, TypeU32(state), values[dword], direct[dword], directLabel, own, ownExit);
    }
    return values;
}

std::uint32_t AddBdaSharedBase(SpirvEmitterState& state, std::int64_t first, std::uint32_t bytes) {
    const auto variable = state.bdaLookupFunction != 0u ? state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeScalarU64(state)), spv::StorageClassPrivate) : 0u;
    state.bdaSharedBases.push_back({variable, first, bytes});
    return static_cast<std::uint32_t>(state.bdaSharedBases.size() - 1u);
}

const BdaSharedRead* SharedBdaRead(const SpirvEmitterState& state, const IrValue& inst) {
    const auto found = state.bdaSharedReads.find(&inst);
    return found != state.bdaSharedReads.end() ? &found->second : nullptr;
}

void ResetBdaSharedBases(SpirvEmitterState& state, const IrBlock* block) {
    const auto found = state.bdaSharedBaseResets.find(block);
    if (found == state.bdaSharedBaseResets.end()) return;
    for (const auto base : found->second) state.module.AddFunction(spv::OpStore, state.bdaSharedBases[base].variable, BdaConstant(state, 0u));
}

void FindBdaSharedBases(SpirvEmitterState& state) {
    static const bool refused = std::getenv("APS5_NO_BDA_SHARED_BASE") != nullptr;
    const auto& program = state.program;
    // Two lanes per invocation would need a translation for each.
    if (refused || !program.Info().usesDma || state.laneCount != 1u || BdaByteReadsForced()) return;
    // What the base address is computed from: the address pair's dwords (or the high dword of a full
    // address) and the low dword, and whether the low dword and the immediate lose their low bits.
    struct Key {
        const IrValue* pairLow;
        const IrValue* pairHigh;
        const IrValue* low;
        bool scalar;
        std::uint32_t residue;
        auto operator<=>(const Key&) const = default;
    };
    struct Found {
        std::vector<std::pair<const IrValue*, std::int64_t>> reads;
        std::int64_t first = 0;
        std::int64_t end = 0;
        // The blocks and activity conditions under which a read of the base already tries the hull.
        std::set<std::pair<const IrBlock*, const IrValue*>> tried;
        std::vector<bool> tries;
    };
    std::map<Key, Found> found;
    std::vector<Key> order;
    std::unordered_map<const IrValue*, const IrBlock*> definedIn;
    const auto& memory = program.Resources().memoryInfo;
    for (const auto* block : program.BlockOrder()) {
        for (const auto* inst : block->Instructions()) {
            definedIn.emplace(inst, block);
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
            // As GuestAddress and EmitBdaDwordReads add them.
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
            const Key key{pairLow, pairHigh, low, scalar, static_cast<std::uint32_t>(at(first)) & 3u};
            auto [entry, added] = found.try_emplace(key);
            if (added) {
                order.push_back(key);
                entry->second.first = at(first);
                entry->second.end = at(last) + 4;
            }
            const IrValue* active = inst->Argument(inst->ArgumentCount() - 1u);
            entry->second.tries.push_back(entry->second.tried.emplace(block, active != nullptr ? active->Resolve() : nullptr).second);
            entry->second.reads.emplace_back(inst, at(first));
            entry->second.first = std::min(entry->second.first, at(first));
            entry->second.end = std::max(entry->second.end, at(last) + 4);
        }
    }
    std::size_t reads = 0;
    std::size_t sharing = 0;
    for (const auto& key : order) {
        const auto& group = found.at(key);
        reads += group.reads.size();
        if (group.reads.size() < 2u) continue;
        sharing += group.reads.size();
        const auto base = AddBdaSharedBase(state, group.first, static_cast<std::uint32_t>(group.end - group.first));
        for (std::size_t read = 0; read < group.reads.size(); ++read) state.bdaSharedReads.emplace(group.reads[read].first, BdaSharedRead{base, group.reads[read].second, group.tries[read]});
        for (const IrValue* value : {key.pairLow, key.pairHigh, key.low}) {
            const auto block = value != nullptr ? definedIn.find(value) : definedIn.end();
            if (block == definedIn.end()) continue;
            auto& resets = state.bdaSharedBaseResets[block->second];
            if (std::find(resets.begin(), resets.end(), base) == resets.end()) resets.push_back(base);
        }
    }
    state.bdaSharedBase = !state.bdaSharedBases.empty();
    // APS5_TRACE_BDA=1 prints how a program's address-based dword reads share base addresses, and
    // how many shared bases are computed only from values that are the same for the whole draw or
    // dispatch (user data, constants and what the resource walk reads): the bases the CPU could
    // translate once per draw. Nothing uses that count yet.
    static const bool traced = std::getenv("APS5_TRACE_BDA") != nullptr;
    if (traced && reads != 0u) {
        std::map<std::size_t, std::size_t> sizes;
        for (const auto& key : order) ++sizes[found.at(key).reads.size()];
        std::string text;
        for (const auto& [size, count] : sizes) text += (text.empty() ? "" : ", ") + std::to_string(count) + " of " + std::to_string(size);
        std::size_t uniformBases = 0;
        std::size_t uniformReads = 0;
        const SrtWalker walker;
        for (const auto& key : order) {
            const auto& group = found.at(key);
            if (group.reads.size() < 2u) continue;
            bool uniform = true;
            for (const IrValue* value : {key.pairLow, key.pairHigh, key.low}) uniform = uniform && (value == nullptr || walker.ValidateRuntimeValue(program.Resources(), value, RuntimeValueType::Integer));
            if (!uniform) continue;
            ++uniformBases;
            uniformReads += group.reads.size();
        }
        std::fprintf(stderr, "[gpu] shader 0x%llx stage %u: %zu of %zu address-based dword reads share %zu base addresses (bases by reads: %s); %zu bases with %zu reads are the same for the whole draw or dispatch\n", static_cast<unsigned long long>(program.Resources().shaderHash), static_cast<unsigned>(program.Resources().stage), sharing, reads, state.bdaSharedBases.size(), text.c_str(), uniformBases, uniformReads);
    }
}

}
