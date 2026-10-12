#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"

namespace AgcDriver::DriverDetail {

void Driver::reportDrawCache(DrawEntryCounters& counters) {
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto validated = counters.lookups - counters.absent;
    const auto miss = [&](DrawMiss reason) { return count(counters.misses[static_cast<std::size_t>(reason)]); };
    std::string ranks;
    for (std::size_t rank = 0; rank < dispatchVariants(); ++rank) {
        char text[32];
        std::snprintf(text, sizeof(text), "%s%llu", rank == 0 ? "" : " / ", count(counters.variantHitsByRank[rank]));
        ranks += text;
    }
    AgcDriver::ProfilePrint_nid_no_patch( "[draw-cache] %llu lookups (10 s): %llu no entry, %llu validated in %.1f us each (key, lookup and every stage): %llu hits (every stage equal; stage hits by variant rank 1..k %s), misses by reason: front stage differing %llu, fragment differing %llu, other stage differing %llu, layout (push offset) %llu, gate %llu, stage count %llu; %llu stage validations (%llu equal, %.2f variants compared each); %llu inserts (%llu variants inserted, %llu evicted beyond k, %llu already present, %llu unstable), %zu entries (%llu variants, ~%.1f MiB), %llu evictions in total, %llu LRU moves; verify: %llu hits captured again, %llu stages differed\n", count(counters.lookups), count(counters.absent), count(validated), validated != 0 ? counters.validateUs / static_cast<double>(validated) : 0.0, count(counters.hits), ranks.c_str(), miss(DrawMiss::FrontDiffering), miss(DrawMiss::FragmentDiffering), miss(DrawMiss::OtherDiffering), miss(DrawMiss::Layout), miss(DrawMiss::Gate), miss(DrawMiss::Stages), count(counters.stageValidations), count(counters.stageEqual), counters.stageValidations != 0 ? static_cast<double>(counters.variantsCompared) / static_cast<double>(counters.stageValidations) : 0.0, count(counters.inserts), count(counters.variantsInserted), count(counters.variantsEvicted), count(counters.present), count(counters.unstable), drawCache.size(), count(drawCacheVariants), static_cast<double>(drawCacheVariantBytes) / (1024.0 * 1024.0), count(drawCacheEvictions), count(counters.touches), count(counters.verifyHits), count(counters.verifyMismatches));
    AgcDriver::ProfilePrint_nid_no_patch( "[draw-cache] register key (10 s): %llu lookups, %llu hits, key build %.1f us each; decode skipped %llu, partial (state/pixel/programs from the entry) %llu; facade log != table %llu; verify: %llu decodes compared, %llu differed\n", count(counters.registerKeyLookups), count(counters.registerKeyHits), counters.registerKeyLookups != 0 ? counters.keyUs / static_cast<double>(counters.registerKeyLookups) : 0.0, count(counters.decodeSkipped), count(counters.decodePartial), count(counters.facadeMismatches), count(counters.verifyDecodes), count(counters.verifyDecodeMismatches));
    std::array<unsigned long long, DrawAheadCountKinds> ahead{};
    for (std::size_t kind = 0; kind < DrawAheadCountKinds; ++kind) {
        const auto total = drawAhead.counts[kind].load(std::memory_order_relaxed);
        ahead[kind] = total - drawAhead.reported[kind];
        drawAhead.reported[kind] = total;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[draw-ahead] %s (10 s): %llu scans started, %llu draws prepared ahead, %llu draws hit with every stage prepared ahead; of the prepared, reads over recorded GPU work: %llu served from known values, %llu read raw on evidence, %llu read without the wait they need in order, %llu under a pending or shadowed image (%llu of all these in vertex tables); not prepared: %llu already cached, %llu the queue had reached, %llu over recorded GPU work with that switched off, %llu indirect, mesh or rejected, %llu failed; %llu scans stopped at a packet they cannot follow\n", drawAheadEnabled() ? "on" : "off", ahead[AheadScans], ahead[AheadPrepared], ahead[AheadHits], ahead[AheadOverKnown], ahead[AheadOverRaw], ahead[AheadOverSync], ahead[AheadOverImage], ahead[AheadOverVertex], ahead[AheadWarm], ahead[AheadPassed], ahead[AheadPending], ahead[AheadUnsupported], ahead[AheadErrors], ahead[AheadStops]);
    std::array<unsigned long long, 9> refused{};
    for (std::size_t kind = 0; kind < refused.size(); ++kind) {
        const auto total = drawAhead.refused[kind].load(std::memory_order_relaxed);
        refused[kind] = total - drawAhead.refusedReported[kind];
        drawAhead.refusedReported[kind] = total;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[draw-ahead] stages prepared ahead that the validation refused (10 s): %llu differing, %llu unmapped, %llu under a queued label, %llu under an image still pending or shadowed, %llu while recorded writes were published, %llu while the pending images changed, %llu while memory was unmapped or mapped; %llu stage validations of either cache read under such an image through the flush hook, as a capture does\n", refused[static_cast<std::size_t>(EntryOutcome::Differing)], refused[static_cast<std::size_t>(EntryOutcome::Inaccessible)], refused[static_cast<std::size_t>(EntryOutcome::QueuedLabel)], refused[static_cast<std::size_t>(EntryOutcome::FlushingImage)], refused[static_cast<std::size_t>(EntryOutcome::PublishMoved)], refused[static_cast<std::size_t>(EntryOutcome::PendingMoved)], refused[static_cast<std::size_t>(EntryOutcome::ForgetMoved)], static_cast<unsigned long long>(validatedThroughHook.load(std::memory_order_relaxed) - std::exchange(validatedThroughHookReported, validatedThroughHook.load(std::memory_order_relaxed))));
    std::array<unsigned long long, DrawLockRunCountKinds> runs{};
    for (std::size_t kind = 0; kind < DrawLockRunCountKinds; ++kind) {
        const auto total = drawLockRunCounts[kind].load(std::memory_order_relaxed);
        runs[kind] = total - drawLockRunReported[kind];
        drawLockRunReported[kind] = total;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[draw-run] %s (10 s): the GPU mutex was kept after %llu draws and %llu draws after them found it held; given up at a draw that missed the cache %llu, before a validation's GPU wait %llu, at a packet that is neither a draw nor a register write or at the end %llu, after %u draws or %lld us %llu\n", drawLockRunEnabled() ? "on" : "off", runs[RunsStarted], runs[RunDrawsHeld], runs[RunEndedByMiss], runs[RunEndedBySync], runs[RunEndedByPacket], DrawLockRun::MaxDraws, static_cast<long long>(DrawLockRun::MaxHeld.count()), runs[RunEndedByBound]);
    counters = DrawEntryCounters{};
}

DrawEntryCounters Driver::DrawCacheCounters() {
    std::lock_guard cacheLock(drawCacheMutex);
    return drawEntryCounters;
}

}
