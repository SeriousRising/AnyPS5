#include "ControlFlow/Structurizer.hpp"
#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ShaderRecompiler;

namespace {

ControlFlowGraph makeGraph(const std::vector<std::vector<std::uint32_t>>& successors) {
    ControlFlowGraph graph;
    graph.entryBlock = 0;
    for (std::uint32_t id = 0; id < successors.size(); ++id) {
        BasicBlock block;
        block.id = id;
        block.startProgramCounter = id * 8;
        block.endProgramCounter = id * 8 + 8;
        block.instructionBegin = id * 2;
        block.instructionEnd = id * 2 + 2;
        block.successors = successors[id];
        auto& terminator = block.terminator;
        if (successors[id].empty()) {
            terminator.kind = TerminatorKind::Return;
        } else if (successors[id].size() == 1) {
            terminator.kind = TerminatorKind::Branch;
            terminator.trueBlock = successors[id][0];
        } else {
            terminator.kind = TerminatorKind::ConditionalBranch;
            terminator.condition = BranchCondition::SccNonZero;
            terminator.trueBlock = successors[id][0];
            terminator.falseBlock = successors[id][1];
        }
        graph.blocks.push_back(std::move(block));
    }
    for (const auto& block : graph.blocks) {
        for (const auto successor : block.successors) graph.blocks[successor].predecessors.push_back(block.id);
    }
    return graph;
}

const BasicBlock* innermostLoopHeader(const ControlFlowGraph& graph, std::uint32_t blockId) {
    const BasicBlock* innermost = nullptr;
    std::size_t innermostSize = 0;
    for (const auto& header : graph.blocks) {
        if (!header.terminator.loopHeader || !graph.Dominates(header.id, blockId) || graph.Dominates(header.terminator.mergeBlock, blockId) || blockId == header.terminator.mergeBlock) continue;
        const auto size = static_cast<std::size_t>(std::count_if(graph.blocks.begin(), graph.blocks.end(), [&](const BasicBlock& block) { return graph.Dominates(header.id, block.id) && !graph.Dominates(header.terminator.mergeBlock, block.id); }));
        if (innermost == nullptr || size < innermostSize) {
            innermost = &header;
            innermostSize = size;
        }
    }
    return innermost;
}

void requireStructuredBranches(const ControlFlowGraph& graph, const char* name) {
    for (const auto& block : graph.blocks) {
        const auto& terminator = block.terminator;
        if (terminator.kind != TerminatorKind::ConditionalBranch || terminator.loopHeader || terminator.trueBlock == terminator.falseBlock) continue;
        const auto* loop = innermostLoopHeader(graph, block.id);
        if (terminator.mergeBlock != InvalidControlFlowId) {
            if (loop != nullptr && terminator.mergeBlock == loop->terminator.continueBlock) {
                throw std::runtime_error(std::string(name) + ": block " + std::to_string(block.id) + " in the loop at block " + std::to_string(loop->id) + " merges at the loop's continue block " + std::to_string(terminator.mergeBlock));
            }
            continue;
        }
        const auto exits = [&](std::uint32_t target) { return loop != nullptr && (target == loop->terminator.mergeBlock || target == loop->terminator.continueBlock); };
        if (!exits(terminator.trueBlock) && !exits(terminator.falseBlock)) {
            throw std::runtime_error(std::string(name) + ": block " + std::to_string(block.id) + " branches to " + std::to_string(terminator.trueBlock) + "/" + std::to_string(terminator.falseBlock) + " without a merge, and neither is its loop's merge or continue");
        }
    }
}

void requireMergeReachesRange(const ControlFlowGraph& graph, std::uint32_t source, std::uint32_t target) {
    const auto header = std::find_if(graph.blocks.begin(), graph.blocks.end(), [&](const BasicBlock& block) {
        return block.instructionBegin == source * 2u && block.instructionEnd == source * 2u + 2u;
    });
    if (header == graph.blocks.end()) throw std::runtime_error("selection instruction range was not preserved");
    auto merge = header->terminator.mergeBlock;
    for (std::size_t steps = 0; merge != InvalidControlFlowId && steps < graph.blocks.size(); ++steps) {
        const auto& block = graph.FindBlock(merge);
        if (block.instructionBegin == target * 2u && block.instructionEnd == target * 2u + 2u) return;
        if (block.instructionBegin != block.instructionEnd || block.terminator.kind != TerminatorKind::Branch || block.successors.size() != 1u) break;
        merge = block.successors.front();
    }
    throw std::runtime_error("selection merge does not reach the expected original instruction range through empty blocks");
}

void requireEmptyContinuations(const ControlFlowGraph& graph) {
    for (const auto& header : graph.blocks) {
        if (!header.terminator.loopHeader) continue;
        const auto& continuation = graph.FindBlock(header.terminator.continueBlock);
        if (continuation.instructionBegin != continuation.instructionEnd || continuation.terminator.kind != TerminatorKind::Branch ||
            continuation.terminator.trueBlock != header.id || continuation.successors != std::vector<std::uint32_t>{header.id}) {
            throw std::runtime_error("loop continuation is not an empty unconditional backedge");
        }
    }
}

}

int main() {
    try {
        for (const auto& edges : {std::vector<std::vector<std::uint32_t>>{{1, 3}, {2}, {0, 5}, {4}, {5, 0}, {}},
                std::vector<std::vector<std::uint32_t>>{{1, 6}, {2, 5}, {3}, {5, 4}, {5, 0}, {0}, {}},
                std::vector<std::vector<std::uint32_t>>{{1, 7}, {2, 6}, {3}, {4}, {6, 5}, {6, 0}, {0}, {}}}) {
            const auto conditionCount = std::count_if(edges.begin(), edges.end(), [](const auto& targets) { return targets.size() == 2u; });
            for (unsigned mask = 0; mask < (1u << conditionCount); ++mask) {
                auto oriented = edges;
                unsigned condition = 0;
                for (auto& targets : oriented) {
                    if (targets.size() == 2u && (mask & (1u << condition++))) std::swap(targets[0], targets[1]);
                }
                auto graph = makeGraph(oriented);
                const auto original = graph;
                Structurizer{}.Structurize(graph);
                if (graph.blocks.size() > original.blocks.size() + 3u || graph.naturalLoops.size() != 1u) throw std::runtime_error("loop-local selection merges did not bound graph growth");
                for (const auto& source : original.blocks) {
                    const auto expected = original.blocks.size() > 6u && source.id + 2u == original.blocks.size() ? 2 : 1;
                    if (std::count_if(graph.blocks.begin(), graph.blocks.end(), [&](const BasicBlock& block) { return block.instructionBegin == source.instructionBegin && block.instructionEnd == source.instructionEnd; }) != expected) throw std::runtime_error("loop-local selection merges copied or lost an unexpected instruction range");
                }
                for (const auto& block : graph.blocks) {
                    if (block.terminator.condition == BranchCondition::GotoVariable || block.terminator.gotoValue >= 0) throw std::runtime_error("loop-local selection merges introduced route variables");
                }
                requireEmptyContinuations(graph);
                requireStructuredBranches(graph, "loop-local selection merge through an unconditional arm");
                const auto normalized = GraphToString(graph);
                Structurizer{}.Structurize(graph);
                if (GraphToString(graph) != normalized) throw std::runtime_error("loop-local selection merges are not idempotent");
            }
        }
        for (unsigned control = 0; control < 8u; ++control) {
            auto graph = makeGraph({{1, 3}, {2}, {0, 5}, {4}, {5, 0}, {}});
            if (control == 0u) graph.FindBlock(3).terminator.kind = TerminatorKind::Unsupported;
            if (control == 1u) graph.FindBlock(3).terminator.kind = TerminatorKind::IndirectBranch;
            if (control == 2u) graph.FindBlock(4).terminator.condition = BranchCondition::GotoVariable;
            if (control == 3u) {
                graph.FindBlock(2).successors = {0, 3};
                graph.FindBlock(2).terminator.falseBlock = 3;
                std::erase(graph.FindBlock(5).predecessors, 2u);
                graph.FindBlock(3).predecessors.push_back(2);
            }
            if (control == 4u) graph.FindBlock(4).terminator.kind = TerminatorKind::Unsupported;
            if (control == 5u) graph.FindBlock(3).terminator.condition = BranchCondition::GotoVariable;
            if (control == 6u || control == 7u) {
                auto& terminator = graph.FindBlock(control == 6u ? 3u : 4u).terminator;
                terminator.gotoVariable = 0;
                terminator.gotoValue = 1;
            }
            Structurizer{}.Structurize(graph);
            requireMergeReachesRange(graph, 0u, 3u);
            requireEmptyContinuations(graph);
            requireStructuredBranches(graph, "unconditional arm retains unsupported and external-entry guards");
        }
        auto endpointOnlyArm = makeGraph({{1}, {2, 5}, {3, 4}, {4}, {1}, {}});
        Structurizer{}.Structurize(endpointOnlyArm);
        requireMergeReachesRange(endpointOnlyArm, 2u, 4u);
        requireEmptyContinuations(endpointOnlyArm);
        requireStructuredBranches(endpointOnlyArm, "unconditional arm ends without a conditional tail");
        for (const auto& edges : {std::vector<std::vector<std::uint32_t>>{{1}, {2, 3}, {1}, {}},
                std::vector<std::vector<std::uint32_t>>{{1}, {2}, {1, 3}, {}},
                std::vector<std::vector<std::uint32_t>>{{1}, {2, 5}, {4, 3}, {4}, {1}, {}},
                std::vector<std::vector<std::uint32_t>>{{1}, {2, 8}, {3, 6}, {5, 4}, {5}, {2}, {7}, {1}, {}}}) {
            auto graph = makeGraph(edges);
            const auto original = graph;
            Structurizer{}.Structurize(graph);
            for (const auto& source : original.blocks) {
                const auto copies = std::count_if(graph.blocks.begin(), graph.blocks.end(), [&](const BasicBlock& block) {
                    return block.instructionBegin == source.instructionBegin && block.instructionEnd == source.instructionEnd;
                });
                if (copies != 1) throw std::runtime_error("loop normalization changed a guest instruction range");
            }
            requireEmptyContinuations(graph);
            requireStructuredBranches(graph, "dedicated loop continuation");
            const auto normalized = GraphToString(graph);
            Structurizer{}.Structurize(graph);
            if (GraphToString(graph) != normalized) throw std::runtime_error("loop normalization is not idempotent");
        }
        auto conditionalLatch = makeGraph({{1}, {2, 5}, {3, 4}, {4, 6}, {5}, {6, 7}, {1, 7}, {}});
        Structurizer{}.Structurize(conditionalLatch);
        requireEmptyContinuations(conditionalLatch);
        requireStructuredBranches(conditionalLatch, "nested selection beside a conditional latch");
        auto sharedTail = makeGraph({{1}, {2, 9}, {3, 4}, {5, 6}, {8}, {7}, {7}, {4, 8}, {1}, {}});
        Structurizer{}.Structurize(sharedTail);
        requireEmptyContinuations(sharedTail);
        requireStructuredBranches(sharedTail, "conditional tails sharing an acyclic join");
        requireMergeReachesRange(sharedTail, 2u, 4u);
        auto cyclicTail = makeGraph({{1}, {2, 8}, {3, 4}, {5, 6}, {7}, {5, 4}, {7}, {1}, {}});
        Structurizer{}.Structurize(cyclicTail);
        requireEmptyContinuations(cyclicTail);
        requireStructuredBranches(cyclicTail, "a conditional tail containing a nested cycle");
        requireMergeReachesRange(cyclicTail, 2u, 7u);
        auto sequentialContinues = makeGraph({{1}, {2, 6}, {4, 3}, {4, 5}, {1}, {4}, {}});
        Structurizer{}.Structurize(sequentialContinues);
        requireEmptyContinuations(sequentialContinues);
        requireStructuredBranches(sequentialContinues, "sequential conditional continues");
        auto nested = makeGraph({{1}, {2}, {5, 3}, {5, 4}, {7}, {6, 7}, {}, {1}});
        Structurizer{}.Structurize(nested);
        if (nested.FindBlock(2).terminator.mergeBlock == nested.FindBlock(3).terminator.mergeBlock) {
            std::fprintf(stderr, "the nested selections share merge block %u\n", nested.FindBlock(2).terminator.mergeBlock);
            return 1;
        }
        requireStructuredBranches(nested, "nested selections");
        auto exitTail = makeGraph({{1}, {2}, {3, 4}, {6}, {6, 5}, {1}, {}});
        Structurizer{}.Structurize(exitTail);
        requireStructuredBranches(exitTail, "a loop exit through a tail block");
        auto exitTails = makeGraph({{1}, {2}, {3, 4}, {7}, {5, 6}, {7, 1}, {7}, {}});
        Structurizer{}.Structurize(exitTails);
        requireStructuredBranches(exitTails, "a loop with two exit tails");
        auto endingExits = makeGraph({{6, 1}, {2}, {3}, {6, 4}, {2, 5}, {}, {}});
        Structurizer{}.Structurize(endingExits);
        requireStructuredBranches(endingExits, "a loop whose two exits end the program");
        auto threeEndingExits = makeGraph({{1}, {2}, {5, 3}, {6, 4}, {1, 7}, {}, {}, {}});
        Structurizer{}.Structurize(threeEndingExits);
        requireStructuredBranches(threeEndingExits, "a loop whose three exits end the program");
        auto innerEndingExit = makeGraph({{1}, {2}, {3}, {7, 4}, {2, 5}, {1, 6}, {}, {}});
        Structurizer{}.Structurize(innerEndingExit);
        requireStructuredBranches(innerEndingExit, "an inner loop exit that ends the program");
        auto earlyReturn = makeGraph({{2, 1}, {4, 2}, {3, 5}, {5}, {}, {}});
        Structurizer{}.Structurize(earlyReturn);
        requireStructuredBranches(earlyReturn, "an early return inside a selection that joins its parent's merge");
        for (const auto& block : earlyReturn.blocks) {
            const auto merge = block.terminator.mergeBlock;
            if (merge != InvalidControlFlowId && !earlyReturn.Dominates(block.id, merge)) {
                std::fprintf(stderr, "an early return inside a selection: header %u does not dominate its merge %u\n", block.id, merge);
                return 1;
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
