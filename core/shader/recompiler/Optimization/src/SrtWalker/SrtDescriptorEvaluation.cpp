#include "Optimization/SrtWalker/SrtDescriptorEvaluation.hpp"
#include "Optimization/SrtWalker/SrtEvaluator.hpp"
#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"
#include "Optimization/SrtWalker/SrtRuntimeValidator.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <unordered_set>

namespace ShaderRecompiler::Detail {

namespace {

std::string& failureReason() {
    struct FailureReasonStorage {};
    return HostThreadLocal<std::string, FailureReasonStorage>();
}

std::string DescribeValue(const IrValue* value, std::uint32_t depth) {
    if (value == nullptr) return "null";
    value = value->Resolve();
    std::string text(IrOpcodeName(value->Opcode()));
    if (value->HasImmediate() && value->Type() == IrType::U32) return text + "(" + std::to_string(value->ImmediateU32()) + ")";
    if (depth == 0 || value->ArgumentCount() == 0) return text;
    text += "(";
    for (std::size_t index = 0; index < value->ArgumentCount(); ++index) {
        if (index != 0) text += ", ";
        text += DescribeValue(value->Argument(index), depth - 1);
    }
    return text + ")";
}

bool Fail(std::string reason) {
    failureReason() = std::move(reason);
    return false;
}

struct FlatBackedCondition {
    bool valid = false;
    std::vector<IrValue*> laneMasks;
};

FlatBackedCondition analyzeFlatBackedCondition(const IrResourcePlan& program, IrValue* condition) {
    RuntimeValidator validator(program, RuntimeValueType::Any);
    std::unordered_set<IrValue*> visiting;
    std::unordered_set<IrValue*> validated;
    FlatBackedCondition result;
    const auto validate = [&](auto&& self, IrValue* raw, bool flatTarget = false) -> bool {
        auto* value = raw->Resolve();
        const auto op = value->Opcode();
        if (op == IrOpcode::Void) return value->HasImmediate() && validator.Run(value);
        if ((op == IrOpcode::LoadAddressU32 || op == IrOpcode::ReadConstBuffer) && (!flatTarget || !IsRawRead(program, *value))) return false;
        if (validated.contains(value)) return true;
        if (!validator.Run(value)) return false;
        if (op == IrOpcode::GetUserData || op == IrOpcode::GetShaderBase) return true;
        if (!visiting.insert(value).second) return false;
        bool valid = true;
        if (op == IrOpcode::ReadConst) {
            auto* slot = value->ArgumentCount() == 2u ? value->Argument(1)->Resolve() : nullptr;
            valid = slot != nullptr && value->Argument(0)->Resolve()->Opcode() == IrOpcode::GetSrtResource && value->Argument(0)->Resolve()->ArgumentCount() == 0u && slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < program.srtReads.size();
            if (valid) {
                const auto index = slot->ImmediateU32();
                const auto& read = program.srtReads[index];
                valid = read.value != nullptr && read.flatOffset == index && IsRawRead(program, *read.value->Resolve()) && self(self, read.value, true);
            }
        } else if (op == IrOpcode::Phi) {
            auto* invariant = ResolveInvariantPhi(program, value);
            valid = invariant != nullptr && self(self, invariant);
        } else {
            for (std::size_t index = 0; valid && index < value->ArgumentCount(); ++index) valid = self(self, value->Argument(index));
        }
        if (valid && op == IrOpcode::ReadFirstLane) result.laneMasks.push_back(value->Argument(1));
        visiting.erase(value);
        if (valid) validated.insert(value);
        return valid;
    };
    result.valid = validate(validate, condition);
    return result;
}

bool isFlatBackedCondition(const IrResourcePlan& program, const ResourceBlock& block, Evaluator& memoryFreeEvaluator) {
    FlatBackedCondition analyzed;
    if (!block.conditionAnalyzed) analyzed = analyzeFlatBackedCondition(program, block.condition);
    if (!(block.conditionAnalyzed ? block.conditionFlatBacked : analyzed.valid)) return false;
    for (auto* laneMask : block.conditionAnalyzed ? block.conditionLaneMasks : analyzed.laneMasks) {
        std::uint32_t mask = 0u;
        if (!memoryFreeEvaluator.Evaluate(laneMask, mask) || mask == 0u) return false;
    }
    return true;
}

const DescriptorSource* Source(const IrResourcePlan& program, std::uint32_t source) {
    if (source >= program.descriptorSources.size()) {
        return nullptr;
    }
    return &program.descriptorSources[source];
}

}

void AnalyzeFlatBackedConditions(IrResourcePlan& plan) {
    for (auto& block : plan.controlFlow) {
        block.conditionAnalyzed = false;
        if (block.condition == nullptr || block.successors.size() != 2u) continue;
        auto analyzed = analyzeFlatBackedCondition(plan, block.condition);
        block.conditionFlatBacked = analyzed.valid;
        block.conditionLaneMasks = std::move(analyzed.laneMasks);
        block.conditionAnalyzed = true;
    }
}

bool EvaluateRuntimeSourcesImpl(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, bool evaluateFlat, std::span<const std::uint8_t> cleanFlatSlots, std::vector<std::uint8_t>& activeSources, std::vector<BdaAbi::BaseRequest>* bdaBases) {
    failureReason().clear();
    static const bool debug = std::getenv("APS5_SRT_DEBUG") != nullptr;
    if (debug) {
        for (std::size_t slot = 0; slot < program.srtReads.size(); ++slot) std::fprintf(stderr, "[srt] slot %zu = %s"  "\n", slot, DescribeValue(program.srtReads[slot].value, 6).c_str());
    }
    if (!program.srtPlanComplete) {
        return Fail("SRT plan is incomplete");
    }
    if (std::any_of(cleanFlatSlots.begin(), cleanFlatSlots.end(), [](std::uint8_t clean) { return clean != 0u; }) && runtime.readSpecializationMemory == nullptr) {
        return Fail("clean flat slots need specialization memory");
    }
    SrtRuntime cleanRuntime = runtime;
    cleanRuntime.readMemory = runtime.readSpecializationMemory;
    std::vector<std::uint32_t> flattened(evaluateFlat ? program.srtReads.size() : 0u);
    std::vector<std::uint8_t> recordedClean(flattened.size());
    Evaluator cleanEvaluator(program, cleanRuntime, cleanFlatSlots, nullptr, nullptr, flattened, recordedClean);
    Evaluator evaluator(program, runtime, cleanFlatSlots, &cleanEvaluator, nullptr, flattened, recordedClean);
    std::vector<std::uint8_t> active;
    std::vector<std::uint8_t> activeReads;
    if (evaluateFlat) {
        active.assign(program.descriptorSources.size(), 1u);
        activeReads.assign(program.srtReads.size(), 1u);
    }
    if (evaluateFlat && !program.controlFlow.empty()) {
        for (const auto& block : program.controlFlow) {
            for (const auto source : block.sources) {
                active.at(source) = 0u;
            }
            for (const auto read : block.reads) activeReads.at(read) = 0u;
        }
        auto conditionRuntime = runtime;
        conditionRuntime.readMemory = +[](void*, std::uint64_t, std::uint32_t*) { return false; };
        conditionRuntime.readSpecializationMemory = conditionRuntime.readMemory;
        conditionRuntime.readTrace = nullptr;
        Evaluator conditionEvaluator(program, conditionRuntime);
        std::vector<std::uint8_t> visited(program.controlFlow.size());
        std::vector<std::uint32_t> pending {0};
        while (!pending.empty()) {
            const auto index = pending.back();
            pending.pop_back();
            if (visited.at(index)) {
                continue;
            }
            visited[index] = 1u;
            const auto& block = program.controlFlow[index];
            for (const auto source : block.sources) {
                active[source] = 1u;
            }
            for (const auto read : block.reads) activeReads[read] = 1u;
            std::uint32_t condition = 0;
            const bool evaluable = block.condition != nullptr && block.successors.size() == 2u && isFlatBackedCondition(program, block, conditionEvaluator);
            if (evaluable) {
                if (!evaluator.Evaluate(block.condition, condition)) return Fail("SRT branch condition in block " + std::to_string(index) + ": " + DescribeValue(block.condition, 4));
                pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
            } else {
                pending.insert(pending.end(), block.successors.begin(), block.successors.end());
            }
        }
    }
    std::vector<DescriptorValue> evaluated;
    evaluated.reserve(sources.size());
    for (const auto sourceIndex : sources) {
        const auto* source = Source(program, sourceIndex);
        if (source == nullptr) {
            return Fail("descriptor source " + std::to_string(sourceIndex) + " does not exist");
        }
        DescriptorValue value;
        value.dwordCount = source->dwordCount;
        if (!evaluateFlat || active[sourceIndex]) {
            for (std::uint32_t index = 0; index < source->dwordCount; index++) {
                if (!evaluator.Evaluate(source->dwords[index], value.dwords[index])) {
                    std::string detail = DescribeValue(source->dwords[index], 4);
                    const IrValue* dword = source->dwords[index]->Resolve();
                    if (dword->Opcode() == IrOpcode::ReadConst && dword->ArgumentCount() == 2 && dword->Argument(1)->Resolve()->HasImmediate()) {
                        const auto slot = dword->Argument(1)->Resolve()->ImmediateU32();
                        if (slot < program.srtReads.size()) detail += " where slot " + std::to_string(slot) + " = " + DescribeValue(program.srtReads[slot].value, 8);
                    }
                    return Fail("descriptor source " + std::to_string(sourceIndex) + " dword " + std::to_string(index) + ": " + detail);
                }
            }
        }
        evaluated.push_back(value);
    }
    if (evaluateFlat) {
        for (const auto& read : program.srtReads) {
            if (read.flatOffset < activeReads.size() && activeReads[read.flatOffset] == 0u) continue;
            const bool clean = read.flatOffset < cleanFlatSlots.size() && cleanFlatSlots[read.flatOffset] != 0u;
            auto& selected = clean ? cleanEvaluator : evaluator;
            // A pure slot's raw read is reachable from no root, so it was not evaluated (nor
            // cached) before this loop: its dereference happens here, once, and is recorded as
            // the slot's leaf; reads nested in its address cone land among the other reads.
            auto* trace = runtime.readTrace;
            const bool pure = trace != nullptr && read.flatOffset < program.pureFlatSlots.size() && program.pureFlatSlots[read.flatOffset] != 0u;
            if (pure) {
                trace->leaf = read.value->Resolve();
                trace->leafSlot = read.flatOffset;
            }
            const bool evaluated = read.flatOffset < flattened.size() && selected.Evaluate(read.value, flattened[read.flatOffset]);
            if (pure) trace->leaf = nullptr;
            if (!evaluated) {
                return Fail(std::string(clean ? "clean " : "") + "SRT read at flat offset " + std::to_string(read.flatOffset) + ": " + DescribeValue(read.value, 4));
            }
        }
    }
    if (bdaBases != nullptr) {
        // After the flat loop, so that every SRT read a base is computed from is already evaluated:
        // a base adds no guest memory read of its own (see IrBdaBase). A base that cannot be
        // evaluated, or whose address or hull wraps, is left without bytes and is not translated.
        bdaBases->assign(program.bdaBases.size(), {});
        // shortcut: a base may be computed from a read this draw's control flow left inactive, and evaluating it here would perform that read; with any read inactive no base is translated (the shader's own lookup serves them). Upgrade to per-base read sets if the profile shows many draws losing their slots.
        const bool everyReadActive = std::find(activeReads.begin(), activeReads.end(), std::uint8_t{0}) == activeReads.end();
        for (std::size_t index = 0; evaluateFlat && everyReadActive && index < program.bdaBases.size(); ++index) {
            const auto& base = program.bdaBases[index];
            std::uint32_t low = 0;
            std::uint32_t pairLow = 0;
            std::uint32_t pairHigh = 0;
            if (base.low == nullptr || base.pairHigh == nullptr || !evaluator.Evaluate(base.low, low) || !evaluator.Evaluate(base.pairHigh, pairHigh) || (base.pairLow != nullptr && !evaluator.Evaluate(base.pairLow, pairLow))) continue;
            // As the emitter's GuestAddressBase computes it.
            if (base.scalar) low &= ~3u;
            const auto pair = (static_cast<std::uint64_t>(pairHigh) << 32u) | (base.pairLow != nullptr ? pairLow : low);
            const auto address = base.pairLow != nullptr ? pair + low : pair;
            const auto hull = address + static_cast<std::uint64_t>(base.first);
            if (address < pair || (base.first < 0 ? hull > address : hull < address) || base.bytes == 0u || hull + base.bytes <= hull) continue;
            (*bdaBases)[index] = {address, hull, base.bytes};
        }
    }
    results = std::move(evaluated);
    activeSources = std::move(active);
    if (evaluateFlat) {
        flat = std::move(flattened);
    }
    return true;
}

const std::string& RuntimeSourceFailureReason() {
    return failureReason();
}

}
