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
//
// The function has one return and two selections (the search, and the selected range's words):
// Vulkan drivers inline it at every access, a return in the middle of an inlined function becomes a
// flag that every later block tests, and RADV's compile time and memory grow faster than the code.
// Every check is therefore evaluated, on zeros where an earlier one failed, and the reason is the
// first failing check's, in the order the checks had when each returned: the header (validated once
// per invocation, see PrepareBdaLookups), overflow of the access, no range at or below the address,
// the range's own words, its end, the permission, a zero device address, overflow of the device
// address.
//
// With `span` the function is lookup_bda_span, for the reads that try a shared base's hull: a lookup
// may ask it (BdaLookupSpan) whether the whole access takes the search its first byte takes. It then
// searches whatever is kept, and fails unless the access ends at or below every begin the search
// stopped short of: every byte of such an access, looked up by itself, is found in the same range,
// sorted table or not. It shares the per-invocation variables of lookup_bda, defined first.
std::uint32_t DefineBdaLookupCore(SpirvEmitterState& state, bool span = false) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto pair = state.module.Type(spv::OpTypeVector, u64, 2u);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto either = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalOr, boolean, left, right); };
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalAnd, boolean, left, right); };
    const auto negate = [&](std::uint32_t value) { return Unary(state, spv::OpLogicalNot, boolean, value); };
    const auto load = [&](std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, value, pointer);
        return value;
    };
    const auto loadWide = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u64, value, variable);
        return value;
    };
    const auto wide = [&] { return state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, u64), spv::StorageClassPrivate); };
    const auto word = [&] { return state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, u32), spv::StorageClassPrivate); };
    const auto zero = BdaConstant(state, 0u);
    const auto no = ConstantBool(state, false);
    const bool first = state.bdaTableInvalid == 0u;
    if (first) state.bdaTableInvalid = word();
    // The range cache. The page table does not change while a draw or dispatch runs (BdaResources
    // fills it once, before it is bound), so the outcome of a search depends only on how the address
    // compares with the begins the search visits: an address at or above every begin it stepped past
    // and below every begin it stopped short of takes the same steps and selects the same range,
    // sorted table or not. Those two bounds and the selected range are kept in Private variables,
    // per invocation, and an address between the bounds skips the search.
    // The selected range's begin is one of the begins stepped past, so such an address is not below
    // it. A range is kept only when its own words are valid, its device address is not zero and the
    // device addresses of its bytes do not wrap: the two device checks then pass for any access the
    // earlier checks let through, so a kept range skips them and keeps, instead of its begin and
    // device address, their difference. The checks that depend on the address, bytes and permission
    // (overflow, the range's end, the permission) run on the kept words, so every result and fault
    // reason is the one the search gives. The entry point empties the cache (PrepareBdaLookups) by
    // storing to every one of these variables: a variable only this function used would be made a
    // local of it by the optimizer's private-to-local pass and lose its value between calls.
    // Several ranges are kept (bdaRangeCacheEntries), tried together and replaced oldest first.
    // The check (bdaRangeCacheCheck) searches at every lookup, uses what the search selected, and
    // returns RangeCacheMismatch when a kept range holds the address and the search finds no range,
    // an invalid one, or one with another begin, end, device address or permissions.
    const bool cache = state.bdaRangeCacheEntries != 0;
    const bool check = cache && state.bdaRangeCacheCheck;
    const bool shared = span;
    // `begin` only with the check.
    struct Kept {
        std::uint32_t floor, ceiling, finish, delta, permissions, begin;
    };
    std::vector<Kept> kept;
    for (std::uint32_t entry = 0; cache && entry < state.bdaRangeCacheEntries; ++entry) {
        if (!first) {
            const auto* words = &state.bdaRangeCache64[entry * (check ? 5u : 4u)];
            kept.push_back({words[0], words[1], words[2], words[3], state.bdaRangeCache32[entry], check ? words[4] : 0u});
            continue;
        }
        kept.push_back({wide(), wide(), wide(), wide(), word(), check ? wide() : 0u});
        state.bdaRangeCache64.insert(state.bdaRangeCache64.end(), {kept.back().floor, kept.back().ceiling, kept.back().finish, kept.back().delta});
        if (check) state.bdaRangeCache64.push_back(kept.back().begin);
        state.bdaRangeCache32.push_back(kept.back().permissions);
    }
    const auto function = state.module.AllocateId();
    state.module.AddName(function, span ? "lookup_bda_span" : "lookup_bda");
    state.module.AddFunction(spv::OpFunction, pair, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, pair, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto permission = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, permission);
    EmitLabel(state, state.module.AllocateId());
    const auto local = [&](std::uint32_t type) {
        const auto variable = state.module.AllocateId();
        state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, type), variable, spv::StorageClassFunction);
        return variable;
    };
    const auto low = local(u32);
    const auto high = local(u32);
    const auto floor = cache ? local(u64) : 0u;
    const auto ceiling = cache || shared ? local(u64) : 0u;
    const auto spanning = shared ? binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permission, constant(BdaLookupSpan)), constant(0)) : 0u;
    const auto wanted = shared ? binary(spv::OpBitwiseAnd, u32, permission, constant(BdaAbi::Read | BdaAbi::Write)) : permission;
    if (shared) state.module.AddFunction(spv::OpStore, ceiling, BdaConstant(state, 0u));
    const auto invalidTable = binary(spv::OpINotEqual, boolean, load(state.bdaTableInvalid), constant(0));
    const auto end = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
    const auto overflowed = either(binary(spv::OpIEqual, boolean, bytes, constant(0)), binary(spv::OpULessThanEqual, boolean, end, address));
    // The range the lookup uses: the kept one that holds the address, then the one the search selected.
    struct Taken {
        std::uint32_t finish, delta, base, permissions, begin;
    };
    Taken taken{zero, zero, zero, constant(0), zero};
    std::uint32_t within = no;
    for (const auto& entry : kept) {
        const auto holds = both(binary(spv::OpULessThanEqual, boolean, loadWide(entry.floor), address), binary(spv::OpULessThan, boolean, address, loadWide(entry.ceiling)));
        const Taken values{loadWide(entry.finish), loadWide(entry.delta), zero, load(entry.permissions), check ? loadWide(entry.begin) : zero};
        if (within == no) {
            within = holds;
            taken = values;
        } else {
            taken = {Select(state, u64, holds, values.finish, taken.finish), Select(state, u64, holds, values.delta, taken.delta), zero, Select(state, u32, holds, values.permissions, taken.permissions), check ? Select(state, u64, holds, values.begin, taken.begin) : zero};
            within = either(within, holds);
        }
    }
    const Taken held = taken;
    auto searches = negate(either(invalidTable, overflowed));
    if (cache && !check) searches = both(searches, shared ? either(negate(within), spanning) : negate(within));
    const auto phi = [&](std::uint32_t type, std::uint32_t value, std::uint32_t from, std::uint32_t otherwise, std::uint32_t before) {
        const auto merged = state.module.AllocateId();
        state.module.AddFunction(spv::OpPhi, type, merged, value, from, otherwise, before);
        return merged;
    };
    const auto beforeSearch = state.currentLabel;
    const auto searchLabel = state.module.AllocateId();
    const auto searched = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, searched, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, searches, searchLabel, searched);
    EmitLabel(state, searchLabel);
    state.module.AddFunction(spv::OpStore, low, constant(0));
    state.module.AddFunction(spv::OpStore, high, BdaLoadWord(state, constant(1)));
    if (cache) state.module.AddFunction(spv::OpStore, floor, zero);
    if (cache || shared) state.module.AddFunction(spv::OpStore, ceiling, BdaConstant(state, ~std::uint64_t{0}));
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
    const auto visited = BdaLoadAddress(state, binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, midpoint, constant(8))));
    const auto before = binary(spv::OpULessThan, boolean, address, visited);
    state.module.AddFunction(spv::OpStore, low, Select(state, u32, before, lower, binary(spv::OpIAdd, u32, midpoint, constant(1))));
    state.module.AddFunction(spv::OpStore, high, Select(state, u32, before, midpoint, upper));
    if (cache) {
        const auto oldFloor = loadWide(floor);
        state.module.AddFunction(spv::OpStore, floor, Select(state, u64, both(negate(before), binary(spv::OpUGreaterThan, boolean, visited, oldFloor)), visited, oldFloor));
    }
    if (cache || shared) {
        const auto oldCeiling = loadWide(ceiling);
        state.module.AddFunction(spv::OpStore, ceiling, Select(state, u64, both(before, binary(spv::OpULessThan, boolean, visited, oldCeiling)), visited, oldCeiling));
    }
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto index = load(low);
    const auto mapped = binary(spv::OpINotEqual, boolean, index, constant(0));
    const auto unmappedThere = negate(mapped);
    const auto rangeLabel = state.module.AllocateId();
    const auto ranged = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, ranged, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, mapped, rangeLabel, ranged);
    EmitLabel(state, rangeLabel);
    const auto selected = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(8)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    const auto begin = BdaLoadAddress(state, selected);
    const auto finish = BdaLoadAddress(state, at(2));
    const auto base = BdaLoadAddress(state, at(4));
    const auto permissions = BdaLoadWord(state, at(6));
    const auto badRange = either(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, at(7)), constant(0)), binary(spv::OpUGreaterThanEqual, boolean, begin, finish));
    const auto delta = binary(spv::OpISub, u64, base, begin);
    std::uint32_t loose = ConstantBool(state, true);
    std::uint32_t differs = no;
    if (cache) {
        const auto last = binary(spv::OpIAdd, u64, base, binary(spv::OpISub, u64, finish, begin));
        auto keeps = both(negate(badRange), both(binary(spv::OpINotEqual, boolean, base, zero), binary(spv::OpUGreaterThan, boolean, last, base)));
        if (check) {
            const auto unequal = [&](std::uint32_t looked, std::uint32_t was) { return binary(spv::OpINotEqual, boolean, looked, was); };
            differs = either(either(unequal(begin, held.begin), unequal(finish, held.finish)), either(unequal(delta, held.delta), unequal(permissions, held.permissions)));
            keeps = both(keeps, negate(within));
        } else {
            loose = negate(keeps);
        }
        EmitIfCondition(state, keeps, [&] {
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
    }
    // Both merges take the searched range's words and flags, or zeros (the kept range's, round the search).
    struct Flags {
        std::uint32_t bad, loose, differs;
    };
    const auto mergeFrom = [&](std::uint32_t from, std::uint32_t otherwiseFrom, const Taken& otherwise, Flags& flags) {
        taken = {phi(u64, taken.finish, from, otherwise.finish, otherwiseFrom), phi(u64, taken.delta, from, otherwise.delta, otherwiseFrom), phi(u64, taken.base, from, zero, otherwiseFrom), phi(u32, taken.permissions, from, otherwise.permissions, otherwiseFrom), zero};
        flags.bad = phi(boolean, flags.bad, from, no, otherwiseFrom);
        if (cache && !check) flags.loose = phi(boolean, flags.loose, from, no, otherwiseFrom);
        if (check) flags.differs = phi(boolean, flags.differs, from, no, otherwiseFrom);
    };
    taken = {finish, delta, base, permissions, zero};
    Flags flags{badRange, loose, differs};
    const auto rangeEnd = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, ranged);
    EmitLabel(state, ranged);
    mergeFrom(rangeEnd, merge, Taken{zero, zero, zero, constant(0), zero}, flags);
    state.module.AddFunction(spv::OpBranch, searched);
    EmitLabel(state, searched);
    mergeFrom(ranged, beforeSearch, held, flags);
    const auto unmapped = phi(boolean, unmappedThere, ranged, no, beforeSearch);
    const auto bad = flags.bad;
    loose = flags.loose;
    differs = flags.differs;
    const auto result = binary(spv::OpIAdd, u64, address, taken.delta);
    const auto deviceEnd = binary(spv::OpIAdd, u64, result, Unary(state, spv::OpUConvert, u64, bytes));
    const auto reasons = [&](std::uint32_t condition, BdaAbi::FaultReason reason, std::uint32_t otherwise) { return Select(state, u64, condition, BdaConstant(state, static_cast<std::uint32_t>(reason)), otherwise); };
    const auto due = [&](std::uint32_t condition) { return cache && !check ? both(loose, condition) : condition; };
    // A span that fails only here is in one range by its first byte's search and not by a later byte's.
    auto reason = shared ? reasons(both(spanning, binary(spv::OpUGreaterThan, boolean, end, loadWide(ceiling))), BdaAbi::FaultReason::Unmapped, zero) : zero;
    reason = reasons(due(either(binary(spv::OpULessThan, boolean, result, taken.base), binary(spv::OpULessThanEqual, boolean, deviceEnd, result))), BdaAbi::FaultReason::Overflow, reason);
    reason = reasons(due(binary(spv::OpIEqual, boolean, taken.base, zero)), BdaAbi::FaultReason::InvalidTable, reason);
    reason = reasons(binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, taken.permissions, wanted), wanted), BdaAbi::FaultReason::Permission, reason);
    reason = reasons(binary(spv::OpUGreaterThan, boolean, end, taken.finish), BdaAbi::FaultReason::Unmapped, reason);
    reason = reasons(bad, BdaAbi::FaultReason::InvalidTable, reason);
    reason = reasons(unmapped, BdaAbi::FaultReason::Unmapped, reason);
    if (check) reason = reasons(both(within, either(either(unmapped, bad), differs)), BdaAbi::FaultReason::RangeCacheMismatch, reason);
    reason = reasons(overflowed, BdaAbi::FaultReason::Overflow, reason);
    reason = reasons(invalidTable, BdaAbi::FaultReason::InvalidTable, reason);
    const auto looked = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, pair, looked, Select(state, u64, binary(spv::OpIEqual, boolean, reason, zero), result, zero), reason);
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
    auto found = looked.address;
    if (state.bdaStopped != 0u && (permission & BdaAbi::Write) != 0u) {
        // An invocation with a pending stop stores nothing (see BdaStopCanBeDeferred).
        const auto stopped = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeBool(state), stopped, state.bdaStopped);
        found = Select(state, u64, stopped, BdaConstant(state, 0u), found);
    }
    state.module.AddFunction(spv::OpReturnValue, found);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

BdaLookupResult BdaLookup(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t permission) {
    return BdaSpanLookup(state, address, bytes, ConstantU32(state, permission), false);
}

BdaLookupResult BdaSpanLookup(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t permission, bool span) {
    const auto u64 = TypeScalarU64(state);
    const auto looked = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVector, u64, 2u), looked, span ? state.bdaSpanLookupFunction : state.bdaLookupFunction, address, bytes, permission);
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

void PrepareBdaLookups(SpirvEmitterState& state) {
    if (state.bdaTableInvalid == 0) return;
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto differs = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpINotEqual, boolean, left, right); };
    const auto either = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpLogicalOr, boolean, left, right); };
    // The header words are read only when the buffer holds them.
    if (state.bdaStopped != 0u) state.module.AddFunction(spv::OpStore, state.bdaStopped, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, state.bdaTableInvalid, constant(1));
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    EmitIfCondition(state, Binary(state, spv::OpUGreaterThanEqual, boolean, length, constant(4)), [&] {
        const auto available = Binary(state, spv::OpISub, u32, length, constant(4));
        auto invalid = differs(BdaLoadWord(state, constant(0)), constant(BdaAbi::Version));
        invalid = either(invalid, differs(BdaLoadWord(state, constant(2)), constant(sizeof(BdaAbi::Range))));
        invalid = either(invalid, differs(BdaLoadWord(state, constant(3)), constant(0)));
        invalid = either(invalid, differs(Binary(state, spv::OpBitwiseAnd, u32, available, constant(7)), constant(0)));
        invalid = either(invalid, differs(BdaLoadWord(state, constant(1)), Binary(state, spv::OpShiftRightLogical, u32, available, constant(3))));
        state.module.AddFunction(spv::OpStore, state.bdaTableInvalid, Select(state, u32, invalid, constant(1), constant(0)));
    });
    // No address is below a ceiling of zero.
    for (const auto variable : state.bdaRangeCache64) state.module.AddFunction(spv::OpStore, variable, BdaConstant(state, 0u));
    for (const auto variable : state.bdaRangeCache32) state.module.AddFunction(spv::OpStore, variable, ConstantU32(state, 0u));
    for (const auto& base : state.bdaSharedBases) state.module.AddFunction(spv::OpStore, base.variable, BdaConstant(state, 0u));
}

void DefineGetBdaPointer(SpirvEmitterState& state) {
    if (!state.program.Info().usesDma) return;
    DefineBdaFaultFunction(state);
    if (state.bdaStopDeferred) state.bdaStopped = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeBool(state)), spv::StorageClassPrivate);
    state.bdaLookupFunction = DefineBdaLookupCore(state);
    if (state.bdaSharedBase) state.bdaSpanLookupFunction = DefineBdaLookupCore(state, true);
    state.bdaPointerFunction = DefineBdaLookup(state, "get_bda_pointer", true);
    if (!BdaByteReadsForced()) state.bdaProbeFunction = DefineBdaLookup(state, "probe_bda_pointer", false);
    DefineBdaDwordReadFunctions(state);
    for (auto& base : state.bdaSharedBases) base.variable = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeScalarU64(state)), spv::StorageClassPrivate);
    if (state.program.Info().bdaWrites) {
        state.bdaWritePointerFunction = DefineBdaLookup(state, "get_bda_write_pointer", true, BdaAbi::Write);
        state.bdaAtomicPointerFunction = DefineBdaLookup(state, "get_bda_atomic_pointer", true, BdaAbi::Read | BdaAbi::Write);
        state.bdaNoteWriteFunction = DefineBdaNoteWrite(state);
    }
}

}
