#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <array>
#include <bit>
#include <limits>

namespace ShaderRecompiler {

namespace {

void ReturnBdaZeroIf(SpirvEmitterState& state, std::uint32_t condition) {
    const auto failed = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
    EmitLabel(state, failed);
    state.module.AddFunction(spv::OpReturnValue, BdaConstant(state, 0u));
    EmitLabel(state, next);
}

// What the direct table holds for an address (BdaAbi::DirectEntry): the leaf of its unit, and the
// address's offset in that unit.
struct BdaDirectLeaf {
    std::uint32_t delta;
    std::uint32_t bytes;
    std::uint32_t flags;
    std::uint32_t offset;
};

// The direct table's leaf for `address`, walked where the function stands: one entry is loaded a
// level, and a program has no more levels than the table's builder makes. An address outside the
// root's units, or any address under an invalid header (the root count is then 0), gets a leaf
// without bytes at offset 0 and loads nothing.
BdaDirectLeaf EmitBdaDirectWalk(SpirvEmitterState& state, std::uint32_t address) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer);
        return value;
    };
    const auto phi = [&](std::uint32_t type, std::uint32_t value, std::uint32_t from, std::uint32_t otherwise, std::uint32_t before) {
        const auto merged = state.module.AllocateId();
        state.module.AddFunction(spv::OpPhi, type, merged, value, from, otherwise, before);
        return merged;
    };
    const auto relative = binary(spv::OpISub, u64, address, load(u64, state.bdaDirectBase));
    const auto unit = binary(spv::OpShiftRightLogical, u64, relative, Unary(state, spv::OpUConvert, u64, load(u32, state.bdaDirectShift)));
    const auto inRoot = binary(spv::OpULessThan, boolean, unit, Unary(state, spv::OpUConvert, u64, load(u32, state.bdaDirectCount)));
    const auto firstEntry = Unary(state, spv::OpUConvert, u32, unit);
    const auto firstOffset = binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, relative), load(u32, state.bdaDirectMask));
    const auto before = state.currentLabel;
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    const auto walked = state.module.AllocateId();
    const auto nextEntry = state.module.AllocateId();
    const auto nextOffset = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, walked, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, inRoot, header, walked);
    EmitLabel(state, header);
    const auto entry = phi(u32, firstEntry, before, nextEntry, continuation);
    const auto offset = phi(u32, firstOffset, before, nextOffset, continuation);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranch, body);
    EmitLabel(state, body);
    const auto word = binary(spv::OpIAdd, u32, constant(sizeof(BdaAbi::DirectHeader) / 4u), binary(spv::OpShiftLeftLogical, u32, entry, constant(2u)));
    std::array<std::uint32_t, 4> words{};
    for (std::uint32_t index = 0; index < words.size(); ++index) words[index] = BdaLoadWord(state, index != 0u ? binary(spv::OpIAdd, u32, word, constant(index)) : word);
    const auto descends = binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, words[3], constant(BdaAbi::DirectDescend)), constant(0u));
    // An entry that descends holds its first child, the children's shift and their mask.
    state.module.AddFunction(spv::OpIAdd, u32, nextEntry, words[0], binary(spv::OpShiftRightLogical, u32, offset, words[1]));
    state.module.AddFunction(spv::OpBitwiseAnd, u32, nextOffset, offset, words[2]);
    state.module.AddFunction(spv::OpBranchConditional, descends, continuation, merge);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto delta = binary(spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, words[0]), binary(spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, words[1]), BdaConstant(state, 32u)));
    state.module.AddFunction(spv::OpBranch, walked);
    EmitLabel(state, walked);
    // The delta and flags of a leaf without bytes are not used: the lookup fails on its bytes first.
    return {phi(u64, delta, merge, BdaConstant(state, 0u), before), phi(u32, words[2], merge, constant(0u), before), phi(u32, words[3], merge, constant(0u), before), phi(u32, offset, merge, constant(0u), before)};
}

// (u64 address, u32 bytes, u32 instruction) -> u64 device address of the range holding all the bytes,
std::uint32_t DefineBdaLookup(SpirvEmitterState& state, const char* name, bool recordFaults, std::uint32_t permission = BdaAbi::Read) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto load = [&](std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, value, pointer);
        return value;
    };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    state.module.AddFunction(spv::OpFunction, u64, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, u64, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto fail = [&](std::uint32_t condition, BdaAbi::FaultReason reason) {
        if (recordFaults) {
            ReturnBdaFailureIf(state, condition, address, bytes, instruction, reason);
        } else {
            ReturnBdaZeroIf(state, condition);
        }
    };
    const auto returns = [&](std::uint32_t result) {
        if (state.bdaStopped != 0u && (permission & BdaAbi::Write) != 0u) {
            const auto stopped = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, boolean, stopped, state.bdaStopped);
            result = Select(state, u64, stopped, BdaConstant(state, 0u), result);
        }
        state.module.AddFunction(spv::OpReturnValue, result);
        state.module.AddFunction(spv::OpFunctionEnd);
        return function;
    };
    if (state.bdaDirect) {
        // The same result and the same reasons in the same order as the search below (the header,
        // overflow of the access, a first byte that is not mapped or a last byte past the mapping,
        // the permission), from the leaf of the address. A mapping's device addresses are valid
        // by the table's construction (BdaDirectTable refuses the ranges for which the search
        // reports InvalidTable or a device overflow).
        const auto invalid = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, invalid, state.bdaTableInvalid);
        fail(binary(spv::OpINotEqual, boolean, invalid, constant(0u)), BdaAbi::FaultReason::InvalidTable);
        fail(binary(spv::OpIEqual, boolean, bytes, constant(0u)), BdaAbi::FaultReason::Overflow);
        fail(binary(spv::OpULessThanEqual, boolean, binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes)), address), BdaAbi::FaultReason::Overflow);
        const auto leaf = EmitBdaDirectWalk(state, address);
        // The offset is below 1 << 30; an access as long as the rest of a dword's range is not mapped.
        fail(binary(spv::OpLogicalOr, boolean, binary(spv::OpUGreaterThan, boolean, bytes, constant(0x7fffffffu)), binary(spv::OpUGreaterThan, boolean, binary(spv::OpIAdd, u32, leaf.offset, bytes), leaf.bytes)), BdaAbi::FaultReason::Unmapped);
        static_assert(BdaAbi::DirectWritable == BdaAbi::Write);
        const auto held = binary(spv::OpBitwiseOr, u32, binary(spv::OpBitwiseAnd, u32, leaf.flags, constant(BdaAbi::DirectWritable)), constant(BdaAbi::Read));
        fail(binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, held, constant(permission)), constant(permission)), BdaAbi::FaultReason::Permission);
        return returns(binary(spv::OpIAdd, u64, address, leaf.delta));
    }
    const auto low = state.module.AllocateId();
    const auto high = state.module.AllocateId();
    const auto pointer = TypePointer(state, spv::StorageClassFunction, u32);
    state.module.AddFunction(spv::OpVariable, pointer, low, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpVariable, pointer, high, spv::StorageClassFunction);
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    fail(binary(spv::OpULessThan, boolean, length, constant(4)), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(0)), constant(BdaAbi::Version)), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(2)), constant(sizeof(BdaAbi::Range))), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(3)), constant(0)), BdaAbi::FaultReason::InvalidTable);
    const auto count = BdaLoadWord(state, constant(1));
    const auto available = binary(spv::OpISub, u32, length, constant(4));
    fail(binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, available, constant(7)), constant(0)), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpINotEqual, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3))), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpIEqual, boolean, bytes, constant(0)), BdaAbi::FaultReason::Overflow);
    const auto end = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
    fail(binary(spv::OpULessThanEqual, boolean, end, address), BdaAbi::FaultReason::Overflow);
    state.module.AddFunction(spv::OpStore, low, constant(0));
    state.module.AddFunction(spv::OpStore, high, count);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto lower = load(low);
    const auto upper = load(high);
    const auto search = binary(spv::OpULessThan, boolean, lower, upper);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, search, body, merge);
    EmitLabel(state, body);
    const auto midpoint = binary(spv::OpIAdd, u32, lower, binary(spv::OpShiftRightLogical, u32, binary(spv::OpISub, u32, upper, lower), constant(1)));
    const auto entry = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, midpoint, constant(8)));
    const auto before = binary(spv::OpULessThan, boolean, address, BdaLoadAddress(state, entry));
    const auto newLow = state.module.AllocateId();
    const auto newHigh = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelect, u32, newLow, before, lower, binary(spv::OpIAdd, u32, midpoint, constant(1)));
    state.module.AddFunction(spv::OpSelect, u32, newHigh, before, midpoint, upper);
    state.module.AddFunction(spv::OpStore, low, newLow);
    state.module.AddFunction(spv::OpStore, high, newHigh);
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto index = load(low);
    fail(binary(spv::OpIEqual, boolean, index, constant(0)), BdaAbi::FaultReason::Unmapped);
    const auto selected = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(8)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    const auto begin = BdaLoadAddress(state, selected);
    const auto finish = BdaLoadAddress(state, at(2));
    const auto base = BdaLoadAddress(state, at(4));
    const auto permissions = BdaLoadWord(state, at(6));
    fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, at(7)), constant(0)), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpUGreaterThanEqual, boolean, begin, finish), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpUGreaterThan, boolean, end, finish), BdaAbi::FaultReason::Unmapped);
    fail(binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, constant(permission)), constant(permission)), BdaAbi::FaultReason::Permission);
    fail(binary(spv::OpIEqual, boolean, base, BdaConstant(state, 0)), BdaAbi::FaultReason::InvalidTable);
    const auto offset = binary(spv::OpISub, u64, address, begin);
    const auto result = binary(spv::OpIAdd, u64, base, offset);
    fail(binary(spv::OpULessThan, boolean, result, base), BdaAbi::FaultReason::Overflow);
    const auto deviceEnd = binary(spv::OpIAdd, u64, result, Unary(state, spv::OpUConvert, u64, bytes));
    fail(binary(spv::OpULessThanEqual, boolean, deviceEnd, result), BdaAbi::FaultReason::Overflow);
    return returns(result);
}

}

std::uint32_t EmitBdaDirectDwordPointer(SpirvEmitterState& state, std::uint32_t address) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto leaf = EmitBdaDirectWalk(state, address);
    const auto physical = Binary(state, spv::OpIAdd, u64, address, leaf.delta);
    const auto fits = Binary(state, spv::OpULessThanEqual, boolean, Binary(state, spv::OpIAdd, u32, leaf.offset, ConstantU32(state, 4u)), leaf.bytes);
    const auto extent = Binary(state, spv::OpULessThanEqual, boolean, address, BdaConstant(state, std::numeric_limits<std::uint64_t>::max() - 4u));
    const auto mapped = Binary(state, spv::OpINotEqual, boolean, physical, BdaConstant(state, 0u));
    const auto aligned = Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u64, physical, BdaConstant(state, 3u)), BdaConstant(state, 0u));
    const auto usable = Binary(state, spv::OpLogicalAnd, boolean, Binary(state, spv::OpLogicalAnd, boolean, fits, extent), Binary(state, spv::OpLogicalAnd, boolean, mapped, aligned));
    return Select(state, u64, usable, physical, BdaConstant(state, 0u));
}

static std::uint32_t DefineBdaNoteWrite(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_bda_write");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlMaskNone, state.module.Type(spv::OpTypeFunction, voidType, TypeScalarU64(state)));
    const auto address = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, TypeScalarU64(state), address);
    EmitLabel(state, state.module.AllocateId());
    const auto done = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, boolean), done, spv::StorageClassFunction);
    const auto counter = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, u32), counter, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpStore, done, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, counter, constant(0u));
    const auto u64 = TypeScalarU64(state);
    const auto key = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenChunkShift))), constant(1));
    const auto page = binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenPageShift))), constant(BdaAbi::WrittenChunkPages - 1u));
    const auto bit = binary(spv::OpShiftLeftLogical, u32, constant(1u), page);
    const auto hash = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, key, constant(0x9e3779b1u)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::WrittenPageSlots))));
    const auto scope = constant(spv::ScopeDevice);
    const auto relaxed = constant(spv::MemorySemanticsMaskNone);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto probe = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, probe, counter);
    const auto found = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, found, done);
    const auto searching = binary(spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, found), binary(spv::OpULessThan, boolean, probe, constant(BdaAbi::WrittenPageProbes)));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, searching, body, merge);
    EmitLabel(state, body);
    const auto slot = binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, hash, probe), constant(BdaAbi::WrittenPageSlots - 1u));
    const auto keyWord = binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenSlotsWord), binary(spv::OpShiftLeftLogical, u32, slot, constant(1u)));
    const auto current = state.module.AllocateId();
    state.module.AddFunction(spv::OpAtomicLoad, u32, current, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed);
    const auto same = binary(spv::OpIEqual, boolean, current, key);
    state.module.AddFunction(spv::OpStore, done, same);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, same), [&] {
        const auto previous = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed, relaxed, key, constant(0u));
        const auto claimed = binary(spv::OpIEqual, boolean, previous, constant(0u));
        state.module.AddFunction(spv::OpStore, done, binary(spv::OpLogicalOr, boolean, claimed, binary(spv::OpIEqual, boolean, previous, key)));
        EmitIfCondition(state, claimed, [&] {
            const auto index = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicIAdd, u32, index, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenCountWord)), scope, relaxed, constant(1u));
            state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenListWord), index)), scope, relaxed, slot);
        });
    });
    const auto holds = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, holds, done);
    EmitIfCondition(state, holds, [&] {
        const auto maskWord = binary(spv::OpIAdd, u32, keyWord, constant(1u));
        const auto mask = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicLoad, u32, mask, BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed);
        EmitIfCondition(state, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, mask, bit), constant(0u)), [&] {
            state.module.AddFunction(spv::OpAtomicOr, u32, state.module.AllocateId(), BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed, bit);
        });
    });
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpStore, counter, binary(spv::OpIAdd, u32, probe, constant(1u)));
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, noted, done);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, noted), [&] {
        state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenOverflowWord)), scope, relaxed, constant(1u));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void PrepareBdaDirectLookups(SpirvEmitterState& state) {
    if (state.bdaTableInvalid == 0u) return;
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto differs = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpINotEqual, boolean, left, right); };
    const auto either = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalOr, boolean, left, right); };
    constexpr std::uint32_t headerWords = sizeof(BdaAbi::DirectHeader) / 4u;
    state.module.AddFunction(spv::OpStore, state.bdaTableInvalid, constant(1u));
    state.module.AddFunction(spv::OpStore, state.bdaDirectBase, BdaConstant(state, 0u));
    for (const auto variable : {state.bdaDirectShift, state.bdaDirectMask, state.bdaDirectCount}) state.module.AddFunction(spv::OpStore, variable, constant(0u));
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    EmitIfCondition(state, binary(spv::OpUGreaterThanEqual, boolean, length, constant(headerWords)), [&] {
        const auto shift = BdaLoadWord(state, constant(1u));
        const auto count = BdaLoadWord(state, constant(2u));
        const auto entries = BdaLoadWord(state, constant(3u));
        const auto ranges = BdaLoadWord(state, constant(6u));
        // Counts that would wrap the length below are not a table's.
        auto invalid = differs(BdaLoadWord(state, constant(0u)), constant(BdaAbi::DirectVersion));
        invalid = either(invalid, differs(BdaLoadWord(state, constant(7u)), constant(0u)));
        invalid = either(invalid, binary(spv::OpUGreaterThan, boolean, shift, constant(BdaAbi::DirectMaxRootShift)));
        invalid = either(invalid, binary(spv::OpUGreaterThan, boolean, count, entries));
        invalid = either(invalid, binary(spv::OpUGreaterThan, boolean, binary(spv::OpBitwiseOr, u32, entries, ranges), constant(0x03ffffffu)));
        invalid = either(invalid, differs(length, binary(spv::OpIAdd, u32, constant(headerWords), binary(spv::OpIAdd, u32, binary(spv::OpShiftLeftLogical, u32, entries, constant(2u)), binary(spv::OpShiftLeftLogical, u32, ranges, constant(3u))))));
        state.module.AddFunction(spv::OpStore, state.bdaTableInvalid, Select(state, u32, invalid, constant(1u), constant(0u)));
        state.module.AddFunction(spv::OpStore, state.bdaDirectBase, BdaLoadAddress(state, constant(4u)));
        state.module.AddFunction(spv::OpStore, state.bdaDirectShift, shift);
        state.module.AddFunction(spv::OpStore, state.bdaDirectMask, binary(spv::OpISub, u32, binary(spv::OpShiftLeftLogical, u32, constant(1u), shift), constant(1u)));
        state.module.AddFunction(spv::OpStore, state.bdaDirectCount, Select(state, u32, invalid, constant(0u), count));
    });
}

void DefineGetBdaPointer(SpirvEmitterState& state) {
    if (state.faultBufferVariable != 0) DefineBdaFaultFunction(state);
    if (!state.program.Info().usesDma) return;
    if (state.bdaDirect) {
        const auto word = [&](std::uint32_t type) { return state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, type), spv::StorageClassPrivate); };
        state.bdaTableInvalid = word(TypeU32(state));
        state.bdaDirectBase = word(TypeScalarU64(state));
        state.bdaDirectShift = word(TypeU32(state));
        state.bdaDirectMask = word(TypeU32(state));
        state.bdaDirectCount = word(TypeU32(state));
    }
    if (state.bdaStopDeferred) state.bdaStopped = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeBool(state)), spv::StorageClassPrivate);
    state.bdaPointerFunction = DefineBdaLookup(state, "get_bda_pointer", true);
    if (!BdaByteReadsForced()) state.bdaProbeFunction = DefineBdaLookup(state, "probe_bda_pointer", false);
    DefineBdaDwordReadFunctions(state);
    DefineBdaSpanReadFunctions(state);
    if (state.program.Info().bdaWrites) {
        state.bdaWritePointerFunction = DefineBdaLookup(state, "get_bda_write_pointer", true, BdaAbi::Write);
        if (!BdaByteReadsForced()) state.bdaWriteProbeFunction = DefineBdaLookup(state, "probe_bda_write_pointer", false, BdaAbi::Write);
        state.bdaAtomicPointerFunction = DefineBdaLookup(state, "get_bda_atomic_pointer", true, BdaAbi::Read | BdaAbi::Write);
        state.bdaNoteWriteFunction = DefineBdaNoteWrite(state);
        DefineBdaByteWriteFunctions(state);
    }
}

}
