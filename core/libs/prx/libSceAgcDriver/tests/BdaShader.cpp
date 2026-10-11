#include "BdaShader.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <algorithm>
#include <functional>
#include <map>
#include <utility>

namespace {

template<typename TEmit>
std::vector<std::uint32_t> MakeShader(std::uint64_t address, bool coherent, bool stops, const TEmit& emit, bool deferred, std::uint32_t rangeCacheEntries = ShaderRecompiler::BdaRangeCacheDefaultEntries, bool writes = false, bool rangeCacheCheck = false, const std::function<void(ShaderRecompiler::SpirvEmitterState&)>& shareBases = {}) {
    using namespace ShaderRecompiler;
    IrProgram program;
    program.Resources().stage = IrShaderStage::Compute;
    program.Info().usesDma = true;
    program.Info().bdaWrites = writes;
    if (coherent) {
        MemoryInfo memory;
        memory.coherent = true;
        program.Resources().memoryInfo = {memory};
    }
    SpirvEmitterState state(program, {});
    EmitBaseHeader(state.module, program);
    const auto define = [&](std::uint32_t binding) {
        const auto variable = state.module.DefineGlobalVariable(TypeStorageBufferPointer(state), spv::StorageClassStorageBuffer);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, 0u);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
        return variable;
    };
    state.bdaPagetableVariable = define(0);
    state.faultBufferVariable = define(1);
    const auto output = define(2);
    std::vector<std::uint32_t> interfaceVariables;
    if (!stops) {
        interfaceVariables.push_back(state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassInput, TypeU32(state)), spv::StorageClassInput));
        state.module.AddAnnotation(spv::OpDecorate, interfaceVariables[0], spv::DecorationBuiltIn, spv::BuiltInLocalInvocationIndex);
    }
    state.bdaStopsInvocations = stops;
    state.bdaStopDeferred = stops && deferred;
    state.bdaRangeCacheEntries = rangeCacheEntries;
    state.bdaRangeCacheCheck = rangeCacheCheck;
    state.bdaSharedBase = static_cast<bool>(shareBases);
    DefineGetBdaPointer(state);
    if (shareBases) shareBases(state);
    const auto main = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), main, spv::FunctionControlMaskNone, TypeFunction(state));
    EmitLabel(state, state.module.AllocateId());
    PrepareBdaLookups(state);
    SpirvValueEmitContext ctx(state);
    auto& instruction = program.CreateValue(IrOpcode::LoadAddressU32, IrType::U32);
    MemoryFlags flags{};
    flags.pc = 0x1234;
    instruction.SetFlags(flags);
    auto base = BdaConstant(state, address);
    if (!stops) {
        const auto index = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), index, interfaceVariables[0]);
        base = Binary(state, spv::OpIAdd, TypeScalarU64(state), base, Unary(state, spv::OpUConvert, TypeScalarU64(state), index));
    }
    const auto values = emit(ctx, instruction, base);
    EmitUnlessBdaStopped(state, [&] {
        for (std::uint32_t index = 0; index < values.size(); ++index) state.module.AddFunction(spv::OpStore, BdaWord(state, output, ConstantU32(state, index)), values[index]);
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.module.AddExecutionMode(main, spv::ExecutionModeLocalSize, 1u, 1u, 1u);
    state.module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", interfaceVariables);
    return state.module.Finalize();
}

}

std::vector<std::uint32_t> MakeBdaTestShader(std::uint64_t address, std::uint32_t bits, std::int64_t offset, bool deferred) {
    using namespace ShaderRecompiler;
    return MakeShader(address, false, true, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t base) {
        if (offset != 0) base = AddBdaAddress(ctx, instruction, base, BdaConstant(ctx.state, offset < 0 ? std::uint64_t{0} - static_cast<std::uint64_t>(offset) : static_cast<std::uint64_t>(offset)), offset < 0);
        return std::vector<std::uint32_t>{EmitBdaRead(ctx, instruction, base, bits)};
    }, deferred);
}

std::vector<std::uint32_t> MakeBdaDwordReadTestShader(std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops, bool deferred) {
    using namespace ShaderRecompiler;
    return MakeShader(address, coherent, stops, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t base) {
        const auto values = EmitBdaDwordReads(ctx, instruction, base, 0u, dwords, true);
        return std::vector<std::uint32_t>(values.begin(), values.begin() + dwords);
    }, deferred);
}

std::vector<std::uint32_t> MakeBdaSequenceTestShader(std::span<const BdaTestStep> steps, std::uint32_t rangeCacheEntries, bool stops, bool rangeCacheCheck, bool deferred, std::uint64_t sharedBytes) {
    using namespace ShaderRecompiler;
    // The hull of each base, by block and residue: [first address, end), and the reads in it.
    struct Hull {
        std::uint64_t first = ~std::uint64_t{0};
        std::uint64_t end = 0;
        std::uint32_t reads = 0;
        std::uint32_t base = 0;
        std::uint32_t emitted = 0;
    };
    std::map<std::pair<std::uint64_t, std::uint64_t>, Hull> hulls;
    const auto hullOf = [&](const BdaTestStep& step) -> Hull* {
        if (sharedBytes == 0u || step.kind != BdaTestStep::Read) return nullptr;
        return &hulls[{step.address & ~(sharedBytes - 1u), step.address & 3u}];
    };
    for (const auto& step : steps) {
        auto* hull = hullOf(step);
        if (hull == nullptr) continue;
        hull->first = std::min(hull->first, step.address);
        hull->end = std::max(hull->end, step.address + step.dwords * 4u);
        ++hull->reads;
    }
    const auto shareBases = [&](SpirvEmitterState& state) {
        for (auto& [block, hull] : hulls) {
            if (hull.reads >= 2u) hull.base = AddBdaSharedBase(state, 0, static_cast<std::uint32_t>(hull.end - hull.first));
        }
    };
    return MakeShader(0, false, stops, [&](SpirvValueEmitContext& ctx, IrValue& instruction, std::uint32_t) {
        auto& state = ctx.state;
        std::vector<std::uint32_t> values;
        for (std::uint32_t index = 0; index < steps.size(); ++index) {
            MemoryFlags flags{};
            flags.pc = BdaTestStepPc + index;
            instruction.SetFlags(flags);
            const auto address = BdaConstant(state, steps[index].address);
            const auto operand = ConstantU32(state, steps[index].value);
            switch (steps[index].kind) {
                case BdaTestStep::Read: {
                    auto* hull = hullOf(steps[index]);
                    const auto dwords = steps[index].dwords;
                    std::array<std::uint32_t, 4> read{};
                    // The first read of a base tries its hull, and every third one after it (as a read under another condition would).
                    if (hull != nullptr && hull->reads >= 2u) read = EmitBdaSharedDwordReads(ctx, instruction, state.bdaSharedBases[hull->base], static_cast<std::int64_t>(steps[index].address - hull->first), hull->emitted++ % 3u == 0u, address, 0u, dwords, true);
                    else if (dwords == 1u) read[0] = EmitBdaRead(ctx, instruction, address, 32u);
                    else read = EmitBdaDwordReads(ctx, instruction, address, 0u, dwords, true);
                    for (std::uint32_t dword = 1; dword < dwords; ++dword) read[0] = Binary(state, spv::OpIAdd, TypeU32(state), read[0], Binary(state, spv::OpIMul, TypeU32(state), read[dword], ConstantU32(state, 2u * dword + 1u)));
                    values.push_back(read[0]);
                    break;
                }
                case BdaTestStep::Store:
                    EmitBdaStore(ctx, instruction, address, operand, 32u);
                    values.push_back(operand);
                    break;
                case BdaTestStep::AtomicAdd:
                    values.push_back(EmitBdaAtomic(ctx, instruction, address, 4u, [&](std::uint32_t pointer) {
                        const auto old = state.module.AllocateId();
                        state.module.AddFunction(spv::OpAtomicIAdd, TypeU32(state), old, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), operand);
                        return old;
                    }));
                    break;
            }
        }
        return values;
    }, deferred, rangeCacheEntries, true, rangeCacheCheck, sharedBytes != 0u ? std::function<void(SpirvEmitterState&)>(shareBases) : std::function<void(SpirvEmitterState&)>{});
}
