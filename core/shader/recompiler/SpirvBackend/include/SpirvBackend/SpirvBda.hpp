#ifndef CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBDA_HPP
#define CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBDA_HPP

#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "SpirvBackend/SpirvEmitter.hpp"
#include "BdaAbi.hpp"
#include <array>
#include <functional>

namespace ShaderRecompiler {

std::uint32_t BdaConstant(SpirvEmitterState& state, std::uint64_t value);
std::uint32_t BdaWord(SpirvEmitterState& state, std::uint32_t variable, std::uint32_t index);
std::uint32_t BdaLoadWord(SpirvEmitterState& state, std::uint32_t index);
std::uint32_t BdaLoadAddress(SpirvEmitterState& state, std::uint32_t index);
struct BdaLookupResult {
    std::uint32_t address;
    std::uint32_t reason;
};
// Calls lookup_bda, which searches the page table unless one of the ranges the invocation found last
// holds the address (APS5_NO_BDA_RANGE_CACHE=1 always searches); see DefineBdaLookupCore.
BdaLookupResult BdaLookup(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t permission);
// A permission bit for lookup_bda_span, beside BdaAbi's: the lookup then fails unless every byte of
// the access also takes the search the first one took (and it always searches).
inline constexpr std::uint32_t BdaLookupSpan = 4u;
// BdaLookup with the permission as a value, through lookup_bda_span with `span` (modules that share bases).
BdaLookupResult BdaSpanLookup(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t permission, bool span = true);
// Validates the page table's header for lookup_bda and empties its range cache; emitted at the start
// of the program's function.
void PrepareBdaLookups(SpirvEmitterState& state);
void DefineBdaFaultFunction(SpirvEmitterState& state);
void DefineBdaDwordReadFunctions(SpirvEmitterState& state);
void RecordBdaFault(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void ReturnBdaFailureIf(SpirvEmitterState& state, std::uint32_t condition, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void ValidateBdaTarget(const IrProgram& program, const SpirvTargetOptions& target);
// Whether a faulting BDA access may end its invocation. Programs with workgroup barriers must keep
// every invocation running, so their faulting reads return zero instead.
bool BdaInvocationsMayStop(const IrProgram& program);
void StopBdaInvocationIf(SpirvEmitterState& state, std::uint32_t condition);
// Whether the program's faulting reads may set a flag instead of ending the invocation where they
// are: the driver's compile time grows much faster than the program with one return per read site.
// A flagged invocation must leave no trace, so this holds only for one-lane pixel and compute
// programs whose effects are all suppressed for it (address-based stores and atomics, void buffer,
// image, LDS and GDS stores) or discarded with it (pixel exports), and whose only views of other invocations
// are a ballot, where a flagged invocation counts as absent, v_readfirstlane_b32, which takes its
// lane from such a ballot, and a DPP operand: one that skips
// inactive lanes reads zero from a flagged lane, as it did from one that had ended, and a
// fetch-inactive one reads what the flagged lane computed, where a lane that had ended gave nothing
// defined.
// APS5_BDA_STOP_PER_SITE=1 ends every faulting invocation at the read, as before.
bool BdaStopCanBeDeferred(const SpirvEmitterState& state);
// Whether `instruction` is a store that the flow emitter skips for an invocation with a pending stop:
// a bound-buffer, image, LDS or GDS store. Address-based stores are refused by their lookup instead.
bool BdaStoppedInvocationSkips(const IrValue& instruction);
// Ends the invocation here when a deferred stop is pending (before a guest loop, which a flagged
// invocation could spin in, and where a pixel program returns).
void EndStoppedBdaInvocation(SpirvEmitterState& state);
// `condition`, false for an invocation with a pending stop.
std::uint32_t UnlessBdaStopped(SpirvEmitterState& state, std::uint32_t condition);
// Emits an effect that an invocation with a pending stop must not have.
void EmitUnlessBdaStopped(SpirvEmitterState& state, const std::function<void()>& emit);
std::uint32_t EmitBdaRead(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bits);
void EmitBdaWrite(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits = 32u);
void EmitBdaStore(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits);
std::uint32_t EmitBdaAtomic(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bytes, const std::function<std::uint32_t(std::uint32_t)>& operation);
// Reads the dwords of a 1-4 dword load at address + offset that the program extracts, with one
// table lookup for the whole span; the per-byte lookups of EmitBdaRead remain the fallback (and
// the only path under APS5_BDA_BYTE_READS=1). Dwords the program never extracts are neither read
// With `shared` the read's own lookup also tries the hull of that base when no member has yet, and
// leaves the outcome in the base's variable (see EmitBdaSharedDwordReads).
struct BdaSharedTry {
    std::uint32_t variable;
    // The variable's value where the read starts.
    std::uint32_t held;
    // Bytes from the hull's first byte to the read's first extracted dword.
    std::uint32_t back;
    std::uint32_t bytes;
};
std::array<std::uint32_t, 4> EmitBdaDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword = false, const BdaSharedTry* shared = nullptr);
// Groups the program's address-based dword reads by base address (the values it is computed from,
// the kind of address and the immediates' residue modulo 4); a base with two reads or more is shared.
// Nothing is emitted. APS5_NO_BDA_SHARED_BASE=1 finds none, so that every read looks its own bytes up.
void FindBdaSharedBases(SpirvEmitterState& state);
// Adds a shared base for the bytes [first, first + bytes) from some address; its variable is defined
// with the lookup functions (DefineGetBdaPointer) or, after them, here.
std::uint32_t AddBdaSharedBase(SpirvEmitterState& state, std::int64_t first, std::uint32_t bytes);
// The shared read that `inst` is, or null.
const BdaSharedRead* SharedBdaRead(const SpirvEmitterState& state, const IrValue& inst);
// EmitBdaDwordReads for a member of `base` whose first extracted dword lies `begin` bytes from the
// base address. Once a member has found the hull in one readable range, a member loads its dwords
// at the hull's device address plus a constant, with no lookup; until then, and when the hull is
// not in one range, it reads as EmitBdaDwordReads does, and a member that `tries` has its own
// lookup try the hull first.
std::array<std::uint32_t, 4> EmitBdaSharedDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, const BdaSharedBase& base, std::int64_t begin, bool tries, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword = false);
// Compares, once per block, the slot the CPU filled for a shared base with the base address the
// program computes: the slot's hull may be used when the draw has slots for this program, the slot's
// device address is not 0 and its guest base address is the program's (computed without a wrap).
// Emitted where it is first asked for in a block, outside any activity condition; null for a base
// the CPU does not translate.
const BdaCpuBaseCheck* EmitBdaCpuBaseCheck(SpirvValueEmitContext& ctx, std::uint32_t base);
// The dwords of a member of `base` under `check`: loaded at the slot's device address plus a constant
// where the slot may be used, and read by `own` (the read without the CPU's slot, with its address
// arithmetic and every fault) where it may not. The CPU translated the hull by the lookup's own rule
// on the page table bound for the draw, so the bytes are the ones `own` would load, and no address
// in a hull the CPU accepted wraps.
std::array<std::uint32_t, 4> EmitBdaCpuBaseDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, const BdaSharedBase& base, std::int64_t begin, std::uint32_t dwords, const BdaCpuBaseCheck& check, const std::function<std::array<std::uint32_t, 4>()>& own);
// Forgets, where `block` starts, the translations of the bases computed from a value it defines.
void ResetBdaSharedBases(SpirvEmitterState& state, const IrBlock* block);
bool BdaByteReadsForced();
std::uint32_t AddBdaAddress(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, bool subtract);
std::uint32_t AddBdaImmediate(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::int32_t immediate);

}

#endif
