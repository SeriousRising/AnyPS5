#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PARALLELCOMPARE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PARALLELCOMPARE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace AgcDriver::Graphics {

struct CompareSpan {
    const void* first;
    const void* second;
    std::size_t bytes;
};

void CompareSpans(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal);
void CompareSpansFrom(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal, std::size_t parallelBytes);
unsigned CompareHelpers();

// Copies run by helper threads: DeferCopy queues one (keep holds its destination alive), WaitCopies
// returns once every copy queued so far finished, CopiesPending says whether any is unfinished.
void DeferCopy(void* destination, const void* source, std::size_t bytes, std::shared_ptr<const void> keep);
void WaitCopies();
bool CopiesPending();

}

#endif
