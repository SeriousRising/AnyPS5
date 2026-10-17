#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_SPLITRECORDER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_SPLITRECORDER_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace AgcDriver::DriverDetail {

// Split recording (default, APS5_NO_SPLIT_DRAWS=1 off): one queue-0 draw whose front half the worker finished,
// handed to the recorder thread to record in packet order. It owns everything the record reads:
// the stages point into `results` or the matched variants, the snapshots into the decode's
// programs, the matched variants' regions or `decodeReads`.
struct SplitDraw {
    std::uint64_t sequence = 0;
    std::shared_ptr<VulkanDevice> device;
    std::shared_ptr<const DrawDecode> decode;
    std::vector<DrawProgram> programs;
    Pm4::DrawParameters parameters{};
    std::vector<ShaderRecompiler::RecompileResult> results;
    std::vector<Graphics::CompiledShader> stages;
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads;
    std::vector<std::shared_ptr<DispatchVariant>> matched;
    std::vector<std::vector<ShaderRecompiler::MemoryRegion>> matchedRegions;
    std::vector<std::shared_ptr<DispatchVariant>> recipeStages;
    std::shared_ptr<const DrawRecipe> recipe;
    std::uint64_t drawKey = 0;
    // The labels the worker queued before this draw: recorded first, as lockForDraw would.
    std::vector<DeferredLabel> labels;
    std::uint32_t header = 0;
    // The worker's collect epoch at the hand-over: the recorder starts an epoch of its own wherever
    // the worker passed an ordering point between two draws (GuestMemory::BumpCollectEpoch).
    std::uint64_t collectEpoch = 0;
    // Instead of a draw: an end-of-pipe RELEASE_MEM the recorder executes as the worker would
    // (its label recorded in order, the interrupt delivered after the work before it).
    std::vector<std::uint32_t> packet;
};

// A label handed over inside a split draw and not recorded yet (sequence: that draw's).
struct SplitTransitLabel {
    std::uint64_t sequence;
    DeferredLabel label;
};

// Guest memory [begin, end) that draws handed over and not recorded yet may write (sequence: the
// newest of them), which the worker's front halves must not read before those draws are recorded.
struct SplitTransitWrite {
    std::uint64_t sequence;
    std::uint64_t begin;
    std::uint64_t end;
};

}

#endif
