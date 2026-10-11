#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <bit>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

namespace {

// (u64 address, u32 bytes, u32 permission) -> (u64 device address of the range holding all the bytes, u64 fault reason);
// the address is 0 when the reason is not. It records nothing, so that each caller has one fault site.
std::uint32_t DefineBdaLookupCore(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto pair = state.module.Type(spv::OpTypeVector, u64, 2u);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto either = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalOr, boolean, left, right); };
    const auto load = [&](std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, value, pointer);
        return value;
    };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "lookup_bda");
    state.module.AddFunction(spv::OpFunction, pair, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, pair, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto permission = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, permission);
    EmitLabel(state, state.module.AllocateId());
    const auto low = state.module.AllocateId();
    const auto high = state.module.AllocateId();
    const auto pointer = TypePointer(state, spv::StorageClassFunction, u32);
    state.module.AddFunction(spv::OpVariable, pointer, low, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpVariable, pointer, high, spv::StorageClassFunction);
    // With the cache check: whether a kept range holds the address, while the search runs. A search
    // that fails then contradicts the kept range.
    std::uint32_t contradicts = 0;
    const auto fail = [&](std::uint32_t condition, BdaAbi::FaultReason reason) {
        const auto failed = state.module.AllocateId();
        const auto next = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
        EmitLabel(state, failed);
        if (contradicts != 0) {
            const auto returned = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, pair, returned, BdaConstant(state, 0u), Select(state, u64, contradicts, BdaConstant(state, static_cast<std::uint32_t>(BdaAbi::FaultReason::RangeCacheMismatch)), BdaConstant(state, static_cast<std::uint32_t>(reason))));
            state.module.AddFunction(spv::OpReturnValue, returned);
        } else {
            state.module.AddFunction(spv::OpReturnValue, state.module.Constant(spv::OpConstantComposite, pair, BdaConstant(state, 0u), BdaConstant(state, static_cast<std::uint32_t>(reason))));
        }
        EmitLabel(state, next);
    };
    // The range cache. The page table does not change while a draw or dispatch runs (BdaResources
    // fills it once, before it is bound), so the outcome of a search depends only on how the address
    // compares with the begins the search visits: an address at or above every begin it stepped past
    // and below every begin it stopped short of takes the same steps and selects the same range,
    // sorted table or not. Those two bounds and the selected range are kept in Private variables,
    // per invocation, and an address between the bounds skips the header checks and the search.
    // The selected range's begin is one of the begins stepped past, so such an address is not below
    // it. A range is kept only when its device address is not zero and the device addresses of its
    // bytes do not wrap: the checks of those two that follow the permission check then pass for any
    // access the earlier checks let through, so a kept range skips them too and keeps, instead of
    // its begin and device address, their difference. The checks that depend on the address, bytes
    // and permission (overflow, the range's end, the permission) run on the kept words, in the same
    // order, so every result and fault reason is the one the search gives. The entry point empties
    // the cache (ResetBdaRangeCache) by storing to every one of these variables: a variable only
    // this function used would be made a local of it by the optimizer's private-to-local pass and
    // lose its value between calls.
    // The check (bdaRangeCacheCheck) searches at every lookup, uses what the search selected, and
    // returns RangeCacheMismatch when a kept range holds the address and the search fails or selects
    // a range with another begin, end, device address or permissions.
    // Several ranges are kept (bdaRangeCacheEntries), tried together and replaced oldest first.
    const bool cache = state.bdaRangeCacheEntries != 0;
    const bool check = cache && state.bdaRangeCacheCheck;
    const auto loadWide = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u64, value, variable);
        return value;
    };
    // `begin` only with the check.
    struct Kept {
        std::uint32_t floor, ceiling, finish, delta, permissions, begin;
    };
    std::vector<Kept> kept;
    std::uint32_t validated = 0;
    // The kept range as the lookup found it on entry, and the block that goes to `found` with it.
    struct {
        std::uint32_t finish, delta, permissions, begin, block;
    } hit{};
    std::uint32_t within = 0;
    std::uint32_t floor = 0;
    std::uint32_t ceiling = 0;
    std::uint32_t found = 0;
    if (cache) {
        const auto wide = [&] { return state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, u64), spv::StorageClassPrivate); };
        const auto word = [&] { return state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, u32), spv::StorageClassPrivate); };
        for (std::uint32_t entry = 0; entry < state.bdaRangeCacheEntries; ++entry) {
            kept.push_back({wide(), wide(), wide(), wide(), word(), check ? wide() : 0u});
            state.bdaRangeCache64.insert(state.bdaRangeCache64.end(), {kept.back().floor, kept.back().ceiling, kept.back().finish, kept.back().delta});
            if (check) state.bdaRangeCache64.push_back(kept.back().begin);
            state.bdaRangeCache32.push_back(kept.back().permissions);
        }
        if (!check) {
            validated = word();
            state.bdaRangeCache32.push_back(validated);
        }
        found = state.module.AllocateId();
        const auto local = [&](std::uint32_t type) {
            const auto variable = state.module.AllocateId();
            state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, type), variable, spv::StorageClassFunction);
            return variable;
        };
        floor = local(u64);
        ceiling = local(u64);
    }
    // Returns the header's range count.
    const auto validate = [&] {
        const auto length = state.module.AllocateId();
        state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
        fail(binary(spv::OpULessThan, boolean, length, constant(4)), BdaAbi::FaultReason::InvalidTable);
        const auto count = BdaLoadWord(state, constant(1));
        const auto available = binary(spv::OpISub, u32, length, constant(4));
        auto invalid = binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(0)), constant(BdaAbi::Version));
        invalid = either(invalid, binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(2)), constant(sizeof(BdaAbi::Range))));
        invalid = either(invalid, binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(3)), constant(0)));
        invalid = either(invalid, binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, available, constant(7)), constant(0)));
        invalid = either(invalid, binary(spv::OpINotEqual, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3))));
        fail(invalid, BdaAbi::FaultReason::InvalidTable);
        return count;
    };
    if (cache) {
        for (const auto& entry : kept) {
            const auto holds = binary(spv::OpLogicalAnd, boolean, binary(spv::OpULessThanEqual, boolean, loadWide(entry.floor), address), binary(spv::OpULessThan, boolean, address, loadWide(entry.ceiling)));
            const Kept values{0, 0, loadWide(entry.finish), loadWide(entry.delta), load(entry.permissions), check ? loadWide(entry.begin) : 0u};
            if (within == 0) {
                within = holds;
                hit = {values.finish, values.delta, values.permissions, values.begin, 0};
            } else {
                hit = {Select(state, u64, holds, values.finish, hit.finish), Select(state, u64, holds, values.delta, hit.delta), Select(state, u32, holds, values.permissions, hit.permissions), check ? Select(state, u64, holds, values.begin, hit.begin) : 0u, 0};
                within = either(within, holds);
            }
        }
        hit.block = state.currentLabel;
    }
    if (check) contradicts = within;
    auto count = cache && !check ? 0u : validate();
    contradicts = 0;
    const auto end = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
    const auto overflowed = either(binary(spv::OpIEqual, boolean, bytes, constant(0)), binary(spv::OpULessThanEqual, boolean, end, address));
    if (cache && !check) {
        const auto missed = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, found, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, within, found, missed);
        EmitLabel(state, missed);
        EmitIfCondition(state, binary(spv::OpIEqual, boolean, load(validated), constant(0)), [&] {
            validate();
            state.module.AddFunction(spv::OpStore, validated, constant(1));
        });
        count = BdaLoadWord(state, constant(1));
    }
    fail(overflowed, BdaAbi::FaultReason::Overflow);
    if (check) contradicts = within;
    state.module.AddFunction(spv::OpStore, low, constant(0));
    state.module.AddFunction(spv::OpStore, high, count);
    if (cache) {
        state.module.AddFunction(spv::OpStore, floor, BdaConstant(state, 0u));
        state.module.AddFunction(spv::OpStore, ceiling, BdaConstant(state, ~std::uint64_t{0}));
    }
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
    const auto visited = BdaLoadAddress(state, entry);
    const auto before = binary(spv::OpULessThan, boolean, address, visited);
    const auto newLow = state.module.AllocateId();
    const auto newHigh = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelect, u32, newLow, before, lower, binary(spv::OpIAdd, u32, midpoint, constant(1)));
    state.module.AddFunction(spv::OpSelect, u32, newHigh, before, midpoint, upper);
    state.module.AddFunction(spv::OpStore, low, newLow);
    state.module.AddFunction(spv::OpStore, high, newHigh);
    if (cache) {
        const auto oldFloor = loadWide(floor);
        const auto oldCeiling = loadWide(ceiling);
        const auto raises = binary(spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, before), binary(spv::OpUGreaterThan, boolean, visited, oldFloor));
        const auto lowers = binary(spv::OpLogicalAnd, boolean, before, binary(spv::OpULessThan, boolean, visited, oldCeiling));
        state.module.AddFunction(spv::OpStore, floor, Select(state, u64, raises, visited, oldFloor));
        state.module.AddFunction(spv::OpStore, ceiling, Select(state, u64, lowers, visited, oldCeiling));
    }
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto index = load(low);
    fail(binary(spv::OpIEqual, boolean, index, constant(0)), BdaAbi::FaultReason::Unmapped);
    const auto selected = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(8)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    auto begin = BdaLoadAddress(state, selected);
    auto finish = BdaLoadAddress(state, at(2));
    auto base = BdaLoadAddress(state, at(4));
    auto permissions = BdaLoadWord(state, at(6));
    fail(either(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, at(7)), constant(0)), binary(spv::OpUGreaterThanEqual, boolean, begin, finish)), BdaAbi::FaultReason::InvalidTable);
    contradicts = 0;
    std::uint32_t delta = 0;
    // Whether the device checks are still due: not for a kept range.
    std::uint32_t unchecked = 0;
    if (cache) {
        delta = binary(spv::OpISub, u64, base, begin);
        const auto last = binary(spv::OpIAdd, u64, base, binary(spv::OpISub, u64, finish, begin));
        const auto keeps = binary(spv::OpLogicalAnd, boolean, binary(spv::OpINotEqual, boolean, base, BdaConstant(state, 0u)), binary(spv::OpUGreaterThan, boolean, last, base));
        if (check) {
            const auto differs = [&](std::uint32_t looked, std::uint32_t held) { return binary(spv::OpINotEqual, boolean, looked, held); };
            fail(binary(spv::OpLogicalAnd, boolean, within, either(either(differs(begin, hit.begin), differs(finish, hit.finish)), either(differs(delta, hit.delta), differs(permissions, hit.permissions)))), BdaAbi::FaultReason::RangeCacheMismatch);
        }
        EmitIfCondition(state, check ? binary(spv::OpLogicalAnd, boolean, keeps, Unary(state, spv::OpLogicalNot, boolean, within)) : keeps, [&] {
            for (std::size_t entry = kept.size(); --entry != 0;) {
                const auto& from = kept[entry - 1];
                const auto& to = kept[entry];
                for (const auto& [target, source] : std::initializer_list<std::pair<std::uint32_t, std::uint32_t>>{{to.floor, from.floor}, {to.ceiling, from.ceiling}, {to.finish, from.finish}, {to.delta, from.delta}}) state.module.AddFunction(spv::OpStore, target, loadWide(source));
                state.module.AddFunction(spv::OpStore, to.permissions, load(from.permissions));
                if (check) state.module.AddFunction(spv::OpStore, to.begin, loadWide(from.begin));
            }
            if (check) state.module.AddFunction(spv::OpStore, kept[0].begin, begin);
            state.module.AddFunction(spv::OpStore, kept[0].floor, loadWide(floor));
            state.module.AddFunction(spv::OpStore, kept[0].ceiling, loadWide(ceiling));
            state.module.AddFunction(spv::OpStore, kept[0].finish, finish);
            state.module.AddFunction(spv::OpStore, kept[0].delta, delta);
            state.module.AddFunction(spv::OpStore, kept[0].permissions, permissions);
        });
        if (!check) {
            const auto discarded = Unary(state, spv::OpLogicalNot, boolean, keeps);
            const auto searched = state.currentLabel;
            state.module.AddFunction(spv::OpBranch, found);
            EmitLabel(state, found);
            const auto phi = [&](std::uint32_t type, std::uint32_t kept, std::uint32_t looked) {
                const auto value = state.module.AllocateId();
                state.module.AddFunction(spv::OpPhi, type, value, kept, hit.block, looked, searched);
                return value;
            };
            finish = phi(u64, hit.finish, finish);
            delta = phi(u64, hit.delta, delta);
            permissions = phi(u32, hit.permissions, permissions);
            base = phi(u64, BdaConstant(state, 0u), base);
            unchecked = phi(boolean, ConstantBool(state, false), discarded);
            fail(overflowed, BdaAbi::FaultReason::Overflow);
        }
    }
    fail(binary(spv::OpUGreaterThan, boolean, end, finish), BdaAbi::FaultReason::Unmapped);
    fail(binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, permission), permission), BdaAbi::FaultReason::Permission);
    const auto result = cache ? binary(spv::OpIAdd, u64, address, delta) : binary(spv::OpIAdd, u64, base, binary(spv::OpISub, u64, address, begin));
    const auto checkDevice = [&](std::uint32_t deviceBase) {
        fail(binary(spv::OpIEqual, boolean, deviceBase, BdaConstant(state, 0)), BdaAbi::FaultReason::InvalidTable);
        const auto deviceEnd = binary(spv::OpIAdd, u64, result, Unary(state, spv::OpUConvert, u64, bytes));
        fail(either(binary(spv::OpULessThan, boolean, result, deviceBase), binary(spv::OpULessThanEqual, boolean, deviceEnd, result)), BdaAbi::FaultReason::Overflow);
    };
    if (cache && !check) {
        EmitIfCondition(state, unchecked, [&] { checkDevice(base); });
    } else {
        checkDevice(base);
    }
    const auto looked = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, pair, looked, result, BdaConstant(state, 0u));
    state.module.AddFunction(spv::OpReturnValue, looked);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

// (u64 address, u32 bytes, u32 instruction) -> u64 device address of the range holding all the bytes,
std::uint32_t DefineBdaLookup(SpirvEmitterState& state, const char* name, bool recordFaults, std::uint32_t permission = BdaAbi::Read) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
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
    const auto looked = BdaLookup(state, address, bytes, permission);
    if (recordFaults) {
        EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), looked.reason, ConstantU32(state, 0u)), [&] {
            state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaFaultFunction, looked.reason, address, bytes, instruction);
        });
    }
    state.module.AddFunction(spv::OpReturnValue, looked.address);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

BdaLookupResult BdaLookup(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t permission) {
    const auto u64 = TypeScalarU64(state);
    const auto looked = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVector, u64, 2u), looked, state.bdaLookupFunction, address, bytes, ConstantU32(state, permission));
    const auto device = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, u64, device, looked, 0u);
    const auto reason = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, u64, reason, looked, 1u);
    return {device, Unary(state, spv::OpUConvert, TypeU32(state), reason)};
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
    state.module.AddFunction(spv::OpStore, done, ConstantBool(state, false));
    const auto page = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, TypeScalarU64(state), address, BdaConstant(state, BdaAbi::WrittenPageShift))), constant(1));
    const auto hash = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, page, constant(0x9e3779b1u)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::WrittenPageSlots))));
    const auto scope = constant(spv::ScopeDevice);
    const auto relaxed = constant(spv::MemorySemanticsMaskNone);
    for (std::uint32_t probe = 0; probe < BdaAbi::WrittenPageProbes; probe++) {
        const auto pending = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, boolean, pending, done);
        EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, pending), [&] {
            const auto slot = binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenSlotsWord), binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, hash, constant(probe)), constant(BdaAbi::WrittenPageSlots - 1u)));
            const auto previous = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, BdaWord(state, state.faultBufferVariable, slot), scope, relaxed, relaxed, page, constant(0u));
            const auto taken = binary(spv::OpLogicalOr, boolean, binary(spv::OpIEqual, boolean, previous, constant(0u)), binary(spv::OpIEqual, boolean, previous, page));
            state.module.AddFunction(spv::OpStore, done, taken);
        });
    }
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, noted, done);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, noted), [&] {
        state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenOverflowWord)), scope, relaxed, constant(1u));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void ResetBdaRangeCache(SpirvEmitterState& state) {
    // No address is below a ceiling of zero.
    for (const auto variable : state.bdaRangeCache64) state.module.AddFunction(spv::OpStore, variable, BdaConstant(state, 0u));
    for (const auto variable : state.bdaRangeCache32) state.module.AddFunction(spv::OpStore, variable, ConstantU32(state, 0u));
}

void DefineGetBdaPointer(SpirvEmitterState& state) {
    if (!state.program.Info().usesDma) return;
    DefineBdaFaultFunction(state);
    state.bdaLookupFunction = DefineBdaLookupCore(state);
    state.bdaPointerFunction = DefineBdaLookup(state, "get_bda_pointer", true);
    if (!BdaByteReadsForced()) state.bdaProbeFunction = DefineBdaLookup(state, "probe_bda_pointer", false);
    DefineBdaDwordReadFunctions(state);
    if (state.program.Info().bdaWrites) {
        state.bdaWritePointerFunction = DefineBdaLookup(state, "get_bda_write_pointer", true, BdaAbi::Write);
        state.bdaAtomicPointerFunction = DefineBdaLookup(state, "get_bda_atomic_pointer", true, BdaAbi::Read | BdaAbi::Write);
        state.bdaNoteWriteFunction = DefineBdaNoteWrite(state);
    }
}

}
