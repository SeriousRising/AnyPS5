#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>

// Split recording (APS5_SPLIT_DRAWS=1; design in NEXT.md "Two-stage worker design"): the queue-0
// worker keeps packet decode, register state and every draw's front half, and hands each cache-hit
// draw to a recorder thread that records it in packet order with the labels queued before it.
// While draws are in flight the worker records nothing: a gate on its GPU-mutex acquisitions waits
// for the recorder first, and packets that read recorded state without the mutex drain it.

namespace AgcDriver::DriverDetail {

namespace {

enum SplitCount : std::size_t { SplitHanded, SplitDrains, SplitDrainUs, SplitTriesRefused, SplitVerified, SplitVerifyMismatches, SplitWaitsInOrder };

// Drain sites that are not a packet's own opcode (the [split] line names them).
enum SplitSite : std::uint32_t { SiteEnd = 0x100, SiteMiss, SiteIndirect, SiteWaits, SiteGate, SiteFlip, SiteRenderingWait, SitePredicated, SiteLabel };

// In-flight draws before the worker waits for room.
constexpr std::size_t SplitWindow = 64;

}

bool Driver::splitDraws() {
    static const bool enabled = std::getenv("APS5_SPLIT_DRAWS") != nullptr && std::getenv("APS5_LOCKED_DRAW_PREPARE") == nullptr && drawEntries() && registerKeyEnabled();
    return enabled;
}

// Debug aid: the recorder re-validates each draw's matched stages at its place in packet order,
// counting the draws whose worker-side validation it would not have passed (APS5_SPLIT_VERIFY=1).
bool Driver::splitVerify() {
    static const bool verify = std::getenv("APS5_SPLIT_VERIFY") != nullptr;
    return verify;
}

bool& Driver::splitLeader() {
    static thread_local bool leader = false;
    return leader;
}

std::deque<SplitTransitLabel>& Driver::splitTransitLabels() {
    static thread_local std::deque<SplitTransitLabel> labels;
    return labels;
}

// Set by splitDrainSite for a memory wait that this queue's own newest label over it satisfies:
// the label precedes the wait in queue order, so the wait holds when the recorder records it (the
// "from the recorder, same queue" rule of waitMemory), and the worker skips the wait.
bool& Driver::splitWaitInOrder() {
    static thread_local bool inOrder = false;
    return inOrder;
}

bool Driver::splitBusy() {
    return !splitIdle.load(std::memory_order_acquire);
}

void Driver::splitDrain(std::uint32_t site) {
    if (!splitLeader() || (splitIdle.load(std::memory_order_acquire) && !splitFailed.load(std::memory_order_acquire))) return;
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock lock(splitMutex);
    if (!splitIdle.load(std::memory_order_acquire)) {
        splitLeaderWaiting = true;
        splitIdled.wait(lock, [&] { return splitIdle.load(std::memory_order_acquire); });
        splitLeaderWaiting = false;
        splitCounts[SplitDrains].fetch_add(1, std::memory_order_relaxed);
        splitDrainSites[std::min<std::size_t>(site, splitDrainSites.size() - 1)].fetch_add(1, std::memory_order_relaxed);
        splitCounts[SplitDrainUs].fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
    }
    splitTransitLabels().clear();
    if (splitError != nullptr) {
        splitFailed.store(false, std::memory_order_release);
        std::rethrow_exception(std::exchange(splitError, nullptr));
    }
}

// The drain site of a packet that must not run while the recorder holds draws (nullopt: it may):
// everything but register writes, draws, packets that read no recorded state (NeedsRecordedLabels)
// label writes without an interrupt, which stay visible to validation as queued labels, and memory
// waits on anything but this queue's own unrecorded labels (a draw in flight writes nothing else
// the wait can see).
std::optional<std::uint32_t> Driver::splitDrainSite(std::uint32_t header, std::span<const std::uint32_t> packet, const QueueState& queue) {
    if (header == FlipPacketHeader) return SiteFlip;
    if (header == RenderingWaitPacketHeader) return SiteRenderingWait;
    if (Pm4::Predicated(header) && queue.predication.operation != 0) return SitePredicated;
    const auto opcode = (header >> 8u) & 0xffu;
    if (Pm4::DrawOpcode(opcode) || opcode == 0x63 || opcode == 0x64 || opcode == 0x9f || !NeedsRecordedLabels(header)) return std::nullopt;
    if (opcode == 0x37 && Pm4::DecodeLabelWrite(packet).has_value()) return std::nullopt;
    if ((opcode == 0x93 || (opcode == 0x3c && ((packet[1] >> 4u) & 1u) != 0)) && packet.size() > 5) {
        const auto awaited = (packet[2] & ~3u) | (static_cast<std::uint64_t>(packet[3]) << 32u);
        const auto bytes = Pm4::WaitAwaitedBytes(packet);
        const std::pair<std::uint64_t, std::uint64_t> run{awaited, awaited + bytes};
        if (!splitLabelsOver(std::span(&run, 1))) return std::nullopt;
        // The newest label over it decides: held ones are newer than those in transit.
        const auto newest = [&]() -> const DeferredLabel* {
            const auto& held = deferredLabels().labels;
            for (auto it = held.rbegin(); it != held.rend(); ++it) {
                if (it->address < run.second && run.first < it->address + it->size) return &*it;
            }
            const auto& transit = splitTransitLabels();
            for (auto it = transit.rbegin(); it != transit.rend(); ++it) {
                if (it->label.address < run.second && run.first < it->label.address + it->label.size) return &it->label;
            }
            return nullptr;
        }();
        if (newest != nullptr && newest->address <= awaited && awaited + bytes <= newest->address + newest->size) {
            std::uint64_t value = 0;
            std::memcpy(&value, newest->bytes.data() + (awaited - newest->address), bytes);
            if (Pm4::WaitComparesValue(packet, value)) {
                splitWaitInOrder() = true;
                splitCounts[SplitWaitsInOrder].fetch_add(1, std::memory_order_relaxed);
                return std::nullopt;
            }
        }
    }
    if (opcode == 0x49 && packet.size() > 4) {
        const auto interrupt = (packet[2] >> 24u) & 7u;
        const bool storesNothing = (packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0;
        if ((interrupt == 0 || interrupt == 3) && (storesNothing || Pm4::DecodeLabelWrite(packet).has_value())) return std::nullopt;
    }
    return opcode;
}

bool Driver::splitGate(bool wait) {
    auto& driver = Get();
    if (driver.splitIdle.load(std::memory_order_acquire) && !driver.splitFailed.load(std::memory_order_acquire)) return true;
    if (!wait) {
        driver.splitCounts[SplitTriesRefused].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    driver.splitDrain(SiteGate);
    return true;
}

bool Driver::splitLabelsOver(std::span<const std::pair<std::uint64_t, std::uint64_t>> runs) {
    auto& transit = splitTransitLabels();
    const auto recorded = splitRecorded.load(std::memory_order_acquire);
    while (!transit.empty() && transit.front().sequence <= recorded) transit.pop_front();
    const auto& held = deferredLabels().labels;
    if (transit.empty() && held.empty()) return false;
    for (const auto& [begin, end] : runs) {
        for (const auto& [sequence, label] : transit) {
            if (label.address < end && begin < label.address + label.size) return true;
        }
        for (const auto& label : held) {
            if (label.address < end && begin < label.address + label.size) return true;
        }
    }
    return false;
}

void Driver::splitHandOver(std::unique_ptr<SplitDraw> draw) {
    auto& deferred = deferredLabels();
    draw->labels.swap(deferred.labels);
    deferred.labels.clear();
    auto& transit = splitTransitLabels();
    const auto recorded = splitRecorded.load(std::memory_order_acquire);
    while (!transit.empty() && transit.front().sequence <= recorded) transit.pop_front();
    std::unique_lock lock(splitMutex);
    if (splitOps.size() >= SplitWindow) {
        splitLeaderWaiting = true;
        splitIdled.wait(lock, [&] { return splitOps.size() < SplitWindow || splitError != nullptr; });
        splitLeaderWaiting = false;
    }
    if (splitError != nullptr) {
        splitFailed.store(false, std::memory_order_release);
        transit.clear();
        std::rethrow_exception(std::exchange(splitError, nullptr));
    }
    if (!splitThread.joinable()) splitThread = std::thread([this] { runSplitRecorder(); });
    draw->sequence = ++splitHanded;
    for (const auto& label : draw->labels) transit.push_back({draw->sequence, label});
    splitOps.push_back(std::move(draw));
    splitIdle.store(false, std::memory_order_release);
    const bool wake = splitSleeping;
    lock.unlock();
    if (wake) splitWake.notify_one();
    splitCounts[SplitHanded].fetch_add(1, std::memory_order_relaxed);
}

void Driver::recordSplitDraw(SplitDraw& draw) {
    GuestMemory::SetCurrentPacket((draw.header >> 8u) & 0xffu, 0);
    if (auto current = device.Load(); current != nullptr && current != draw.device) draw.device = std::move(current);
    if (!draw.labels.empty()) {
        auto& deferred = deferredLabels();
        if (deferred.labels.empty()) deferred.since = std::chrono::steady_clock::now();
        deferred.labels.insert(deferred.labels.end(), draw.labels.begin(), draw.labels.end());
    }
    // The duties of the packets between the previous draw and this one (batch cap, label deadline,
    // boundary reap), then the labels, as the worker's own packet loop and lockForDraw do.
    flushBetweenPackets(0, draw.header, false);
    recordLabelsForPacket(draw.device.get(), 0);
    if (splitVerify()) {
        const auto& programs = draw.programs.empty() ? draw.decode->programs : draw.programs;
        std::uint64_t imagesFlushed = 0, runsSynced = 0;
        std::optional<SampledReadScope> sampling;
        for (std::size_t i = 0; i < draw.matched.size(); ++i) {
            if (draw.matched[i] == nullptr) continue;
            const auto outcome = validateVariant(programs[i].binary.codeAddress, 0, *draw.matched[i], draw.matchedRegions[i], imagesFlushed, runsSynced, sampling);
            splitCounts[SplitVerified].fetch_add(1, std::memory_order_relaxed);
            if (outcome == EntryOutcome::Equal) continue;
            if (splitCounts[SplitVerifyMismatches].fetch_add(1, std::memory_order_relaxed) < 20) std::fprintf(stderr, "[split] verify: draw %llu stage %zu (program 0x%llx) validates as %d at its place in packet order\n", static_cast<unsigned long long>(draw.sequence), i, static_cast<unsigned long long>(programs[i].binary.codeAddress), static_cast<int>(outcome));
        }
    }
    const auto& graphics = draw.decode->state;
    noteDrawWriters(draw.stages, 0);
    if (draw.recipe != nullptr) {
        if (draw.device->DrawFromRecipe(graphics, draw.parameters, draw.stages, draw.snapshots, draw.recipe) == RecipeOutcome::Recorded) return;
        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
    }
    std::shared_ptr<const DrawRecipe> built;
    draw.device->Draw(graphics, draw.parameters, draw.stages, draw.snapshots, draw.recipeStages.empty() ? nullptr : &built);
    if (built != nullptr) attachDrawRecipe(draw.drawKey, draw.recipeStages, std::move(built));
}

void Driver::runSplitRecorder() noexcept {
    PinWorkerThread("queue 0 recorder");
    GuestMemory::TagGpuLockThread(0);
    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);
    std::uint32_t held = 0;
    std::uint64_t collectEpoch = ~0ull;
    auto heldSince = std::chrono::steady_clock::now();
    std::unique_lock lock(splitMutex);
    for (;;) {
        if (splitOps.empty()) {
            if (gpuLock.owns_lock()) {
                lock.unlock();
                gpuLock.unlock();
                lock.lock();
                continue;
            }
            splitIdle.store(true, std::memory_order_release);
            if (splitLeaderWaiting) splitIdled.notify_all();
            if (splitStop) return;
            // The worker hands the next draw over within microseconds in a run of draws: spin a
            // little before sleeping, so a hand-over rarely costs a wake-up.
            lock.unlock();
            for (int spin = 0; spin < 4000 && splitIdle.load(std::memory_order_acquire); ++spin) _mm_pause();
            lock.lock();
            if (splitOps.empty() && !splitStop) {
                splitSleeping = true;
                splitWake.wait(lock, [&] { return splitStop || !splitOps.empty(); });
                splitSleeping = false;
            }
            continue;
        }
        auto draw = std::move(splitOps.front());
        splitOps.pop_front();
        if (splitLeaderWaiting) splitIdled.notify_all();
        lock.unlock();
        try {
            if (!gpuLock.owns_lock()) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
                gpuLock.lock();
                held = 0;
                heldSince = std::chrono::steady_clock::now();
            }
            if (draw->collectEpoch != collectEpoch) {
                collectEpoch = draw->collectEpoch;
                GuestMemory::BumpCollectEpoch();
            }
            recordSplitDraw(*draw);
            splitRecorded.store(draw->sequence, std::memory_order_release);
            draw.reset();
            if (++held >= DrawLockRun::MaxDraws || std::chrono::steady_clock::now() - heldSince >= DrawLockRun::MaxHeld) gpuLock.unlock();
            lock.lock();
        } catch (...) {
            const auto error = std::current_exception();
            draw.reset();
            if (gpuLock.owns_lock()) gpuLock.unlock();
            lock.lock();
            if (splitError == nullptr) splitError = error;
            splitFailed.store(true, std::memory_order_release);
            splitOps.clear();
            splitRecorded.store(splitHanded, std::memory_order_release);
        }
    }
}

std::uint64_t Driver::SplitDrawsHanded() {
    return splitCounts[SplitHanded].load(std::memory_order_relaxed);
}

void Driver::stopSplitRecorder() {
    {
        std::lock_guard lock(splitMutex);
        splitStop = true;
    }
    splitWake.notify_all();
    if (splitThread.joinable()) splitThread.join();
}

void Driver::reportSplit() {
    static std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(10)) return;
    last = now;
    std::array<unsigned long long, 7> counts{};
    for (std::size_t kind = 0; kind < counts.size(); ++kind) {
        const auto total = splitCounts[kind].load(std::memory_order_relaxed);
        counts[kind] = total - splitReported[kind];
        splitReported[kind] = total;
    }
    std::vector<std::pair<std::uint64_t, std::size_t>> sites;
    for (std::size_t site = 0; site < splitDrainSites.size(); ++site) {
        const auto total = splitDrainSites[site].load(std::memory_order_relaxed);
        if (total != splitDrainSitesReported[site]) sites.emplace_back(total - splitDrainSitesReported[site], site);
        splitDrainSitesReported[site] = total;
    }
    std::sort(sites.rbegin(), sites.rend());
    std::string bySite;
    for (std::size_t i = 0; i < sites.size() && i < 8; ++i) {
        static constexpr std::array<const char*, 9> names{"end", "miss", "indirect", "waits", "gate", "flip", "rendering wait", "predicated", "label"};
        char item[48];
        if (sites[i].second >= SiteEnd) std::snprintf(item, sizeof item, " %s %llu", names[sites[i].second - SiteEnd], static_cast<unsigned long long>(sites[i].first));
        else std::snprintf(item, sizeof item, " 0x%02zx %llu", sites[i].second, static_cast<unsigned long long>(sites[i].first));
        bySite += item;
    }
    static DrawEntryCounters previous;
    DrawEntryCounters cache;
    {
        std::lock_guard cacheLock(drawCacheMutex);
        cache = drawEntryCounters;
    }
    const auto delta = [&](std::uint64_t now, std::uint64_t before) { return static_cast<unsigned long long>(now - before); };
    std::fprintf(stderr, "[split] draw cache (10 s): %llu lookups, %llu no entry, %llu hits; misses: front %llu, fragment %llu, other %llu, layout %llu, gate %llu, stages %llu\n", delta(cache.lookups, previous.lookups), delta(cache.absent, previous.absent), delta(cache.hits, previous.hits),
                 delta(cache.misses[0], previous.misses[0]), delta(cache.misses[1], previous.misses[1]), delta(cache.misses[2], previous.misses[2]), delta(cache.misses[3], previous.misses[3]), delta(cache.misses[4], previous.misses[4]), delta(cache.misses[5], previous.misses[5]));
    previous = cache;
    static std::array<std::uint64_t, DrawAheadCountKinds> aheadBefore{};
    std::array<unsigned long long, DrawAheadCountKinds> ahead{};
    for (std::size_t kind = 0; kind < ahead.size(); ++kind) {
        const auto total = drawAhead.counts[kind].load(std::memory_order_relaxed);
        ahead[kind] = total - aheadBefore[kind];
        aheadBefore[kind] = total;
    }
    std::string stops;
    {
        std::vector<std::pair<std::uint64_t, std::size_t>> at;
        for (std::size_t op = 0; op < drawAhead.stoppedAt.size(); ++op) {
            const auto total = drawAhead.stoppedAt[op].load(std::memory_order_relaxed);
            if (total != drawAhead.stoppedAtReported[op]) at.emplace_back(total - drawAhead.stoppedAtReported[op], op);
            drawAhead.stoppedAtReported[op] = total;
        }
        std::sort(at.rbegin(), at.rend());
        for (std::size_t i = 0; i < at.size() && i < 6; ++i) {
            char item[40];
            std::snprintf(item, sizeof item, " 0x%zx %llu", at[i].second, static_cast<unsigned long long>(at[i].first));
            stops += item;
        }
    }
    std::fprintf(stderr, "[split] draw-ahead (10 s): %llu scans, stopped at:%s\n", ahead[AheadScans], stops.c_str());
    std::fprintf(stderr, "[split] draw-ahead (10 s): %llu prepared, %llu already warm, %llu passed by the worker first, %llu hits on prepared stages, %llu unsupported, %llu entries the worker waited for\n", ahead[AheadPrepared], ahead[AheadWarm], ahead[AheadPassed], ahead[AheadHits], ahead[AheadUnsupported], ahead[AheadAwaited]);
    std::fprintf(stderr, "[split] 10 s: %llu draws handed to the recorder, %llu drains waited %.1f ms in all (by site:%s), %llu waits held in queue order, %llu GPU-mutex tries refused while draws were in flight; verify: %llu stages, %llu mismatches\n", counts[SplitHanded], counts[SplitDrains], counts[SplitDrainUs] / 1000.0, bySite.c_str(), counts[SplitWaitsInOrder], counts[SplitTriesRefused], counts[SplitVerified], counts[SplitVerifyMismatches]);
}

}

std::uint64_t AgcDriver::SplitDrawsHanded() {
    return DriverDetail::Driver::Get().SplitDrawsHanded();
}
