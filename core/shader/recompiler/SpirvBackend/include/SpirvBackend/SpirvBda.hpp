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
void DefineBdaFaultFunction(SpirvEmitterState& state);
void DefineBdaDwordReadFunctions(SpirvEmitterState& state);
void DefineBdaSpanReadFunctions(SpirvEmitterState& state);
void RecordBdaFault(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void RecordBdaFaultWords(SpirvEmitterState& state, std::uint32_t addressLow, std::uint32_t addressHigh, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void ReturnBdaFailureIf(SpirvEmitterState& state, std::uint32_t condition, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void ValidateBdaTarget(const IrProgram& program, const SpirvTargetOptions& target);
// Whether a faulting BDA access may end its invocation. Programs with workgroup barriers must keep
// every invocation running, so their faulting reads return zero instead.
bool BdaInvocationsMayStop(const IrProgram& program);
void StopBdaInvocationIf(SpirvEmitterState& state, std::uint32_t condition);
std::uint32_t BdaInstructionPc(SpirvEmitterState& state, const IrValue& inst);
// Whether the program's faulting reads may set a flag instead of ending the invocation where they
// are: the driver's compile time grows much faster than the program with one return per read site.
// A flagged invocation must leave no trace, so this holds only for one-lane pixel and compute
// programs whose effects are all suppressed for it (address-based stores and atomics, void buffer
// and image stores) or discarded with it (pixel exports), and whose only views of other invocations
// are a ballot, where a flagged invocation counts as absent, v_readfirstlane_b32, which takes its
// lane from such a ballot, and a DPP operand: one that skips
// inactive lanes reads zero from a flagged lane, as it did from one that had ended, and a
// fetch-inactive one reads what the flagged lane computed, where a lane that had ended gave nothing
// defined.
// APS5_BDA_STOP_PER_SITE=1 ends every faulting invocation at the read, as before.
bool BdaStopCanBeDeferred(const SpirvEmitterState& state);
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
void DefineBdaByteWriteFunctions(SpirvEmitterState& state);
void EmitBdaDwordWrites(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, std::uint32_t value);
std::uint32_t EmitBdaAtomic(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bytes, const std::function<std::uint32_t(std::uint32_t)>& operation);
// Reads the dwords of a 1-4 dword load at address + offset that the program extracts, with one
// table lookup for the whole span; the per-byte lookups of EmitBdaRead remain the fallback (and
// the only path under APS5_BDA_BYTE_READS=1). Dwords the program never extracts are neither read
std::array<std::uint32_t, 4> EmitBdaDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword = false);
bool BdaByteReadsForced();
std::uint32_t AddBdaAddress(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, bool subtract);
std::uint32_t AddBdaImmediate(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::int32_t immediate);

}

#endif
