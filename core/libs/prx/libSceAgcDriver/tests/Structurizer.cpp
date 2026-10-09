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

<<<<<<< HEAD
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
=======
void requireExactPostDominators(const ControlFlowGraph& graph, const char* name) {
    const auto count = static_cast<std::uint32_t>(graph.blocks.size());
    const auto reachesExitAvoiding = [&](std::uint32_t from, std::uint32_t avoided) {
        std::vector<bool> seen(count, false);
        std::vector<std::uint32_t> stack{from};
        seen[from] = true;
        while (!stack.empty()) {
            const auto id = stack.back();
            stack.pop_back();
            if (graph.blocks[id].successors.empty()) return true;
            for (const auto successor : graph.blocks[id].successors) {
                if (successor == avoided || seen[successor]) continue;
                seen[successor] = true;
                stack.push_back(successor);
            }
        }
        return false;
    };
    for (const auto& block : graph.blocks) {
        std::vector<std::uint32_t> expected;
        for (std::uint32_t candidate = 0; candidate < count; ++candidate) {
            if (candidate == block.id || !reachesExitAvoiding(block.id, candidate)) expected.push_back(candidate);
        }
        if (block.postDominators != expected) {
            throw std::runtime_error(std::string(name) + ": block " + std::to_string(block.id) + " has " + std::to_string(block.postDominators.size()) + " post-dominators, expected " + std::to_string(expected.size()));
>>>>>>> b29ef5af (perf(recompiler): compute post-dominators visiting successors first)
        }
    }
}

}

int main() {
    try {
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
        for (const auto& edges : {std::vector<std::vector<std::uint32_t>>{{1}, {2}, {3, 1}, {4, 2}, {}},
                std::vector<std::vector<std::uint32_t>>{{1}, {2}, {1, 3}, {2, 4}, {}}}) {
            auto graph = makeGraph(edges);
            const auto original = graph;
            Structurizer{}.Structurize(graph);
            if (graph.naturalLoops.size() != 2u || graph.blocks.size() > original.blocks.size() + 8u) throw std::runtime_error("nested loop exit routing did not retain two bounded loops");
            const auto firstMerge = graph.naturalLoops.front().mergeBlock;
            if (firstMerge == graph.naturalLoops.back().mergeBlock) throw std::runtime_error("nested loops share a merge after exit routing");
            for (const auto& source : original.blocks) {
                const auto copies = std::count_if(graph.blocks.begin(), graph.blocks.end(), [&](const BasicBlock& block) {
                    return block.instructionBegin == source.instructionBegin && block.instructionEnd == source.instructionEnd;
                });
                if (copies != 1) throw std::runtime_error("nested loop exit routing changed an instruction range");
            }
            requireEmptyContinuations(graph);
            requireStructuredBranches(graph, "nested loop exits to the parent's merge and continuation");
            const auto normalized = GraphToString(graph);
            Structurizer{}.Structurize(graph);
            if (GraphToString(graph) != normalized) throw std::runtime_error("nested loop exit routing is not idempotent");
        }
        for (const auto& edges : {std::vector<std::vector<std::uint32_t>>{{1}, {2, 4}, {3, 5}, {1}, {6}, {6}, {}},
                std::vector<std::vector<std::uint32_t>>{{1}, {2, 8}, {3}, {4, 5}, {3, 6}, {6}, {7}, {1}, {}}}) {
            auto graph = makeGraph(edges);
            Structurizer{}.Structurize(graph);
            for (const auto& block : graph.blocks) {
                if (block.terminator.condition == BranchCondition::GotoVariable || block.terminator.gotoValue >= 0) throw std::runtime_error("a loop with a local merge unexpectedly requires exit routing");
            }
            requireEmptyContinuations(graph);
            requireStructuredBranches(graph, "a loop whose merge stays in its enclosing construct");
            const auto normalized = GraphToString(graph);
            Structurizer{}.Structurize(graph);
            if (GraphToString(graph) != normalized) throw std::runtime_error("local loop merge normalization is not idempotent");
        }
        for (const auto kind : {TerminatorKind::Unsupported, TerminatorKind::IndirectBranch}) {
            auto graph = makeGraph({{1}, {2}, {3, 1}, {4, 2}, {}});
            graph.FindBlock(2).terminator.kind = kind;
            bool rejected = false;
            try {
                Structurizer{}.Structurize(graph);
            } catch (const std::exception& error) {
                if (std::string(error.what()).find("indirect or unsupported branch") == std::string::npos) throw;
                rejected = true;
            }
            if (!rejected) throw std::runtime_error("loop exit routing accepted an unsupported terminator");
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
        requireExactPostDominators(nested, "nested selections");
        if (nested.FindBlock(2).terminator.mergeBlock == nested.FindBlock(3).terminator.mergeBlock) {
            std::fprintf(stderr, "the nested selections share merge block %u\n", nested.FindBlock(2).terminator.mergeBlock);
            return 1;
        }
        requireStructuredBranches(nested, "nested selections");
        auto exitTail = makeGraph({{1}, {2}, {3, 4}, {6}, {6, 5}, {1}, {}});
        Structurizer{}.Structurize(exitTail);
        requireExactPostDominators(exitTail, "a loop exit through a tail block");
        requireStructuredBranches(exitTail, "a loop exit through a tail block");
        auto exitTails = makeGraph({{1}, {2}, {3, 4}, {7}, {5, 6}, {7, 1}, {7}, {}});
        Structurizer{}.Structurize(exitTails);
        requireExactPostDominators(exitTails, "a loop with two exit tails");
        requireStructuredBranches(exitTails, "a loop with two exit tails");
        auto endingExits = makeGraph({{6, 1}, {2}, {3}, {6, 4}, {2, 5}, {}, {}});
        Structurizer{}.Structurize(endingExits);
        requireExactPostDominators(endingExits, "a loop whose two exits end the program");
        requireStructuredBranches(endingExits, "a loop whose two exits end the program");
        auto threeEndingExits = makeGraph({{1}, {2}, {5, 3}, {6, 4}, {1, 7}, {}, {}, {}});
        Structurizer{}.Structurize(threeEndingExits);
        requireExactPostDominators(threeEndingExits, "a loop whose three exits end the program");
        requireStructuredBranches(threeEndingExits, "a loop whose three exits end the program");
        auto innerEndingExit = makeGraph({{1}, {2}, {3}, {7, 4}, {2, 5}, {1, 6}, {}, {}});
        Structurizer{}.Structurize(innerEndingExit);
        requireExactPostDominators(innerEndingExit, "an inner loop exit that ends the program");
        requireStructuredBranches(innerEndingExit, "an inner loop exit that ends the program");
        auto earlyReturn = makeGraph({{2, 1}, {4, 2}, {3, 5}, {5}, {}, {}});
        Structurizer{}.Structurize(earlyReturn);
        requireExactPostDominators(earlyReturn, "an early return inside a selection that joins its parent's merge");
        requireStructuredBranches(earlyReturn, "an early return inside a selection that joins its parent's merge");
        for (const auto& block : earlyReturn.blocks) {
            const auto merge = block.terminator.mergeBlock;
            if (merge != InvalidControlFlowId && !earlyReturn.Dominates(block.id, merge)) {
                std::fprintf(stderr, "an early return inside a selection: header %u does not dominate its merge %u\n", block.id, merge);
                return 1;
            }
        }
        auto reorderedChain = makeGraph({{2}, {4}, {}, {1}, {0}});
        reorderedChain.entryBlock = 3;
        Structurizer{}.Structurize(reorderedChain);
        requireExactPostDominators(reorderedChain, "a chain whose block ids do not follow control flow");
        auto disconnected = makeGraph({{1}, {}, {3}, {4}, {}});
        Structurizer{}.Structurize(disconnected);
        requireExactPostDominators(disconnected, "a disconnected forward chain");
        auto nonterminating = makeGraph({{1}, {2}, {1}});
        try {
            Structurizer{}.Structurize(nonterminating);
            throw std::runtime_error("a nonterminating loop unexpectedly structurized");
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()).find("has no structured merge/continue") == std::string::npos) throw;
        }
        requireExactPostDominators(nonterminating, "a nonterminating loop");
        std::vector<std::vector<std::uint32_t>> diamonds;
        for (std::uint32_t diamond = 0; diamond < 50u; ++diamond) {
            const auto head = diamond * 3u;
            diamonds.push_back({head + 1u, head + 2u});
            diamonds.push_back({head + 3u});
            diamonds.push_back({head + 3u});
        }
        diamonds.push_back({});
        auto chain = makeGraph(diamonds);
        Structurizer{}.Structurize(chain);
        requireExactPostDominators(chain, "a chain of fifty selections");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
