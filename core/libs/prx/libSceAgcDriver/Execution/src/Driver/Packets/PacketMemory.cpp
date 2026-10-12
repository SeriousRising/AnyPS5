#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <string>

namespace AgcDriver::DriverDetail {

namespace {

// APS5_PROFILE_DRAW: what each DMA_DATA cost in preparePacketMemory and how it was served, as a
// [dma] line every 10 s (the first packet reports at once): by form, the packets, their bytes and
// time, the time's parts (reading the source, flushing pending image results over the destination
// or syncing a copy, recording the store) and the outcomes; then the slowest packets of the window
// with what their destination and source memory is to the storage images (ClassifyFill's cover).
struct DmaTimes {
    double resolveMs = 0;
    double flushMs = 0;
    double storeMs = 0;
    std::size_t outcome = 5;
    int reason = -1;
    int syncCause = 0;
};

constexpr std::array<const char*, 7> DmaOutcomeNames{"gpu", "completion", "cpu", "gpu copy", "empty", "drained", "settled copy"};

std::chrono::steady_clock::time_point dmaClock(bool timed) {
    return timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
}

double dmaSince(bool timed, std::chrono::steady_clock::time_point start) {
    return timed ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() : 0.0;
}

void noteDma(std::span<const std::uint32_t> packet, const DmaTimes& times, double totalMs) {
    struct Form {
        std::uint64_t packets = 0;
        std::uint64_t bytes = 0;
        double ms = 0, resolveMs = 0, flushMs = 0, storeMs = 0;
        std::array<std::uint64_t, 7> outcomes{};
        std::array<std::uint64_t, 4> sizes{};
        std::array<std::uint64_t, 3> sources{};
        std::uint64_t waits = 0;
        std::array<std::uint64_t, 8> waitCauses{};
    };
    struct Slow {
        double ms = 0;
        std::string text;
    };
    static constexpr std::array<const char*, 6> formNames{"fill <=64K", "fill >64K", "copy <=64K", "copy >64K", "to GDS", "from GDS"};
    static constexpr std::array<const char*, 8> coverNames{"none", "exact", "inside", "around", "straddle", "several", "keys", "layer"};
    static std::mutex mutex;
    static std::array<Form, 6> forms{};
    static std::array<Slow, 6> slowest{};
    static std::chrono::steady_clock::time_point lastReport{};
    if (packet.size() < 7) return;
    const auto source = ((packet[1] >> 29u) & 3u) | ((packet[6] >> 24u) & 4u) | ((packet[6] >> 25u) & 8u);
    const auto destination = ((packet[1] >> 20u) & 3u) | ((packet[6] >> 25u) & 4u) | ((packet[6] >> 26u) & 8u);
    const std::size_t bytes = packet[6] & 0x3ffffffu;
    const bool gds = source == 1 || destination == 1;
    const bool fill = source == 2;
    const std::size_t form = destination == 1 ? 4u : source == 1 ? 5u : (fill ? 0u : 2u) + (bytes > 65536 ? 1u : 0u);
    const auto from = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
    const auto to = packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
    std::string slowText;
    if (totalMs >= 1.0 && gds) {
        char text[320];
        char origin[48];
        if (fill) std::snprintf(origin, sizeof(origin), "fill %08x", packet[2]);
        else if (source == 1) std::snprintf(origin, sizeof(origin), "GDS+0x%x", packet[2]);
        else std::snprintf(origin, sizeof(origin), "0x%llx", static_cast<unsigned long long>(from));
        char target[48];
        if (destination == 1) std::snprintf(target, sizeof(target), "GDS+0x%x", packet[4]);
        else std::snprintf(target, sizeof(target), "0x%llx", static_cast<unsigned long long>(to));
        std::snprintf(text, sizeof(text), " [%.1f ms: 0x%zx bytes %s -> %s, resolve %.1f flush %.1f store %.1f, %s reason %d]", totalMs, bytes, origin, target, times.resolveMs, times.flushMs, times.storeMs, DmaOutcomeNames[times.outcome], times.reason);
        slowText = text;
    } else if (totalMs >= 1.0 && bytes != 0) {
        const auto cover = [](std::uint64_t address, std::size_t size) {
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            return coverNames[static_cast<std::size_t>(Graphics::StorageTexture::ClassifyFill(address, size).cover) % coverNames.size()];
        };
        char text[320];
        if (fill) std::snprintf(text, sizeof(text), " [%.1f ms: fill 0x%zx bytes of %08x at 0x%llx (images: %s), resolve %.1f flush %.1f store %.1f, %s reason %d]", totalMs, bytes, packet[2], static_cast<unsigned long long>(to), cover(to, bytes), times.resolveMs, times.flushMs, times.storeMs, DmaOutcomeNames[times.outcome], times.reason);
        else std::snprintf(text, sizeof(text), " [%.1f ms: copy 0x%zx bytes 0x%llx (images: %s) -> 0x%llx (images: %s), resolve %.1f flush %.1f store %.1f, %s reason %d]", totalMs, bytes, static_cast<unsigned long long>(from), cover(from, bytes), static_cast<unsigned long long>(to), cover(to, bytes), times.resolveMs, times.flushMs, times.storeMs, DmaOutcomeNames[times.outcome], times.reason);
        slowText = text;
    }
    std::string line;
    {
        std::lock_guard lock(mutex);
        auto& entry = forms[form];
        ++entry.packets;
        entry.bytes += bytes;
        entry.ms += totalMs;
        entry.resolveMs += times.resolveMs;
        entry.flushMs += times.flushMs;
        entry.storeMs += times.storeMs;
        ++entry.outcomes[times.outcome];
        ++entry.sizes[bytes <= 4 ? 0u : bytes <= 8 ? 1u : bytes <= 64 ? 2u : 3u];
        ++entry.sources[fill ? 0u : source == 1 ? 2u : 1u];
        if (times.syncCause != 0) ++entry.waits;
        for (std::size_t bit = 0; bit < entry.waitCauses.size(); ++bit) entry.waitCauses[bit] += (times.syncCause >> bit) & 1;
        if (!slowText.empty()) {
            const auto least = std::min_element(slowest.begin(), slowest.end(), [](const Slow& left, const Slow& right) { return left.ms < right.ms; });
            if (totalMs > least->ms) *least = {totalMs, std::move(slowText)};
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - lastReport < std::chrono::seconds(10)) return;
        lastReport = now;
        char text[320];
        for (std::size_t i = 0; i < forms.size(); ++i) {
            const auto& each = forms[i];
            if (each.packets == 0) continue;
            std::snprintf(text, sizeof(text), " %s x%llu %.1f MiB %.0f ms (resolve %.0f, flush %.0f, store %.0f; gpu %llu, completion %llu, cpu %llu, gpu copy %llu, settled copy %llu, empty %llu, drained %llu);", formNames[i], static_cast<unsigned long long>(each.packets), static_cast<double>(each.bytes) / 1048576.0, each.ms, each.resolveMs, each.flushMs, each.storeMs,
                          static_cast<unsigned long long>(each.outcomes[0]), static_cast<unsigned long long>(each.outcomes[1]), static_cast<unsigned long long>(each.outcomes[2]), static_cast<unsigned long long>(each.outcomes[3]), static_cast<unsigned long long>(each.outcomes[6]), static_cast<unsigned long long>(each.outcomes[4]), static_cast<unsigned long long>(each.outcomes[5]));
            line += text;
            if (i >= 4) {
                std::snprintf(text, sizeof(text), " (bytes <=4 %llu, <=8 %llu, <=64 %llu, more %llu; source fill %llu, memory %llu, GDS %llu);", static_cast<unsigned long long>(each.sizes[0]), static_cast<unsigned long long>(each.sizes[1]), static_cast<unsigned long long>(each.sizes[2]), static_cast<unsigned long long>(each.sizes[3]),
                              static_cast<unsigned long long>(each.sources[0]), static_cast<unsigned long long>(each.sources[1]), static_cast<unsigned long long>(each.sources[2]));
                line += text;
            }
            if (each.waits != 0) {
                std::snprintf(text, sizeof(text), " (%llu copies waited: source for a dispatch's copied buffer %llu, a draw's %llu, a completion label %llu; destination for %llu, %llu, %llu);", static_cast<unsigned long long>(each.waits), static_cast<unsigned long long>(each.waitCauses[0]), static_cast<unsigned long long>(each.waitCauses[1]), static_cast<unsigned long long>(each.waitCauses[2]),
                              static_cast<unsigned long long>(each.waitCauses[4]), static_cast<unsigned long long>(each.waitCauses[5]), static_cast<unsigned long long>(each.waitCauses[6]));
                line += text;
            }
        }
        line += " slowest:";
        std::sort(slowest.begin(), slowest.end(), [](const Slow& left, const Slow& right) { return left.ms > right.ms; });
        for (const auto& slow : slowest) line += slow.text;
        forms = {};
        slowest = {};
    }
    AgcDriver::ProfilePrint_nid_no_patch("[dma] DMA_DATA in the last window:%s\n", line.c_str());
}

}

bool Driver::preparePacketMemory(const Submission& submission, QueueState& queue, std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode, bool& wroteOnGpu, bool& endOfPipeInterrupt, bool& interruptDeferred, bool& drawPacket, bool& sampleDump) {
    static const bool drainAll = std::getenv("APS5_DRAIN_ALL") != nullptr;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const bool dmaTimed = profile && opcode == 0x50;
    const auto dmaStart = dmaClock(dmaTimed);
    DmaTimes dmaTimes;
    wroteOnGpu = false;
    LabelStepTimer steps(profile);

    bool orderedAlready = false;

    const auto interruptSelect = (packet[2] >> 24u) & 7u;
    endOfPipeInterrupt = opcode == 0x49 && interruptSelect != 0 && interruptSelect != 3;
    interruptDeferred = false;
    if (!drainAll && endOfPipeInterrupt) {
        const auto label = Pm4::DecodeLabelWrite(packet);
        const bool storesNothing = (packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0;
        if (label.has_value() || storesNothing) {
            bumpEpoch(&EpochBumps::drains);
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            steps.Mark(InterruptLockWait);
            const auto localDevice = device.Load();
            recordDeferredLabels(localDevice.get(), submission.queue);
            steps.Mark(InterruptRecord);
            const bool workOpen = Graphics::Recorder::RecordedWorkSinceSubmit() != 0;
            int reason = localDevice != nullptr ? 0 : 1;
            if (label.has_value()) {
                const auto bytes = label->Bytes();
                const auto stamp = ++eventSerial;
                reason = localDevice != nullptr ? localDevice->WriteLabelOnGpu(label->address, bytes, stamp, submission.queue) : 4;
                if (reason == 1) GuestMemory::Write(label->address, bytes, 4);
                if (reason == 0 || reason == 1 || reason == 5 || reason == 6) {
                    noteLabelStore(label->address, bytes, stamp);
                    Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
                }
                countLabelOutcome(reason);
                ++immediateLabels;
            } else {
                ++noOpLabels;
            }
            steps.Mark(InterruptStore);
            if (reason == 0 || reason == 5 || reason == 6) {
                const auto queueId = submission.queue;
                interruptDeferred = localDevice->AfterRecordedWork([queueId] { AgcDriverDeliverEopInterrupt(queueId); }, submission.queue == 0);
                steps.Mark(InterruptCompletion);
                if (interruptDeferred && workOpen) {
                    localDevice->SubmitRecorded(submission.queue == 0);
                    steps.Mark(InterruptSubmit);
                }
                wroteOnGpu = true;
            } else if (reason == 1) {
                wroteOnGpu = true;
            }
        }
    }
    if (!drainAll && !endOfPipeInterrupt && (opcode == 0x49 || opcode == 0x37)) {
        if (const auto label = Pm4::DecodeLabelWrite(packet)) {
            const auto bytes = label->Bytes();
            if (DeferLabels() && bytes.size() <= DeferredLabel::Capacity && bytes.size() % 4 == 0 && label->address % 4 == 0) {

                auto& deferred = deferredLabels();
                if (deferred.labels.empty()) deferred.since = std::chrono::steady_clock::now();
                auto& entry = deferred.labels.emplace_back();
                entry.address = label->address;
                entry.size = bytes.size();
                std::memcpy(entry.bytes.data(), bytes.data(), bytes.size());
                ++queuedLabels;
                wroteOnGpu = true;
                steps.Mark(LabelQueued);
            } else {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                const auto localDevice = device.Load();

                recordDeferredLabels(localDevice.get(), submission.queue);
                const auto stamp = ++eventSerial;
                const auto reason = localDevice != nullptr ? localDevice->WriteLabelOnGpu(label->address, bytes, stamp, submission.queue) : 4;
                wroteOnGpu = reason == 0 || reason == 5 || reason == 6;
                if (reason == 1) {

                    GuestMemory::Write(label->address, bytes, 4);
                    wroteOnGpu = true;
                }
                if (wroteOnGpu) noteLabelStore(label->address, bytes, stamp);
                Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
                countLabelOutcome(reason);
                ++immediateLabels;
                steps.Mark(LabelLocked);
            }
        } else if (opcode == 0x49 ? ((packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0) : (packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u)) == 0) {
            orderedAlready = true;
            ++noOpLabels;
            steps.Mark(LabelNoOp);
        } else {
            ++labelFallbacks[4];
        }
    }

    if (const auto fill = Pm4::DecodeMemoryFill(packet); fill.has_value() && fill->bytes != 0) Graphics::NoteDepthMetadataFill(fill->destination, fill->bytes, std::array<std::uint32_t, 4>{fill->word, fill->word, fill->word, fill->word});

    static const bool cpuStores = std::getenv("APS5_CPU_STORES") != nullptr;
    if (!drainAll && !cpuStores && (opcode == 0x40 || opcode == 0x50 || opcode == 0x83)) {
        constexpr std::size_t gpuStoreLimit = 65536;
        bool drained = true;
        // A copy out of the GDS is recorded as a copy in queue order (or made at once when its
        // source and destination are settled) instead of reading the GDS here, which waits for
        // the recorded work that writes it. A refused copy takes the store path below.
        auto gdsCopy = opcode == 0x50 ? Pm4::DecodeGdsSourceCopy(packet) : std::nullopt;
        // So is a copy out of memory that recorded work writes: reading that source here waits
        // for the work too. One whose source nothing recorded writes stays a store of its bytes.
        if (opcode == 0x50 && !gdsCopy.has_value()) {
            gdsCopy = Pm4::DecodeMemorySourceCopy(packet);
            if (gdsCopy.has_value() && !(GuestMemory::Accessible(reinterpret_cast<const void*>(gdsCopy->source), gdsCopy->bytes) && Graphics::Recorder::SnapshotWriteOverlaps(gdsCopy->source, gdsCopy->bytes))) gdsCopy.reset();
        }
        if (gdsCopy.has_value() && gdsCopy->bytes <= gpuStoreLimit && GuestMemory::Accessible(reinterpret_cast<const void*>(gdsCopy->destination), gdsCopy->bytes, true)) {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Copy);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            if (const auto localDevice = device.Load()) {
                recordDeferredLabels(localDevice.get(), submission.queue);
                const auto storeStart = dmaClock(dmaTimed);
                const auto outcome = localDevice->CopyBuffer(gdsCopy->destination, gdsCopy->source, gdsCopy->bytes, gdsCopy->bytes, std::numeric_limits<std::size_t>::max(), 0, 0, submission.queue, [](std::span<const std::byte>, std::uint64_t) {});
                if (outcome.path == 0 || outcome.path == 1 || outcome.path == 3) {
                    if (outcome.path == 0) noteForeignWriter(gdsCopy->destination, gdsCopy->destination + gdsCopy->bytes, submission.queue);
                    dmaTimes.storeMs = dmaSince(dmaTimed, storeStart) - outcome.syncMs;
                    dmaTimes.flushMs = outcome.syncMs;
                    dmaTimes.reason = outcome.reason;
                    dmaTimes.syncCause = outcome.syncCause;
                    dmaTimes.outcome = outcome.path == 0 ? 6u : 3u;
                    wroteOnGpu = true;
                    drained = false;
                }
            }
        }
        const auto resolveStart = dmaClock(dmaTimed);
        if (const auto store = drained ? Pm4::ResolveStore(packet, queue, gpuStoreLimit) : std::optional<Pm4::StoreWrite>{}) {
            dmaTimes.resolveMs = dmaSince(dmaTimed, resolveStart);
            const auto bytes = store->Bytes();
            if (bytes.empty()) {

                orderedAlready = true;
                drained = false;
                dmaTimes.outcome = 4;
            } else {

                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                const auto localDevice = device.Load();

                const auto flushStart = dmaClock(dmaTimed);
                Graphics::StorageTexture::FlushPending(store->address, bytes.size(), nullptr, "packet store", Graphics::PublishScope::PartialUnits);
                dmaTimes.flushMs = dmaSince(dmaTimed, flushStart);

                recordDeferredLabels(localDevice.get(), submission.queue);
                const auto stamp = ++eventSerial;
                const auto storeStart = dmaClock(dmaTimed);
                const auto reason = localDevice != nullptr ? localDevice->WriteLabelOnGpu(store->address, bytes, stamp, submission.queue) : 4;
                dmaTimes.storeMs = dmaSince(dmaTimed, storeStart);
                dmaTimes.reason = reason;
                dmaTimes.outcome = reason == 0 ? 0u : reason == 5 || reason == 6 ? 1u : reason == 1 ? 2u : 5u;
                if (reason == 0 || reason == 1 || reason == 5 || reason == 6) noteLabelStore(store->address, bytes, stamp);
                if (reason == 0 || reason == 5 || reason == 6) {
                    if (reason == 0) ++storesOnGpu;
                    else ++storesBehindCompletions;
                    wroteOnGpu = true;
                    drained = false;
                } else if (reason == 1) {
                    GuestMemory::Write(store->address, bytes, 1);
                    ++storesOnCpu;
                    wroteOnGpu = true;
                    drained = false;
                }
                Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
            }
        }
        if (drained && opcode == 0x50) {
            const auto copy = Pm4::DecodeMemoryCopy(packet);
            if (copy.has_value() && copy->bytes > gpuStoreLimit && GuestMemory::Accessible(reinterpret_cast<const void*>(copy->source), copy->bytes) && GuestMemory::Accessible(reinterpret_cast<const void*>(copy->destination), copy->bytes, true)) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Copy);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                if (const auto localDevice = device.Load()) {
                    recordDeferredLabels(localDevice.get(), submission.queue);
                    const auto storeStart = dmaClock(dmaTimed);
                    const auto outcome = localDevice->CopyBuffer(copy->destination, copy->source, copy->bytes, 0, std::numeric_limits<std::size_t>::max(), 0, 0, submission.queue, [](std::span<const std::byte>, std::uint64_t) {});
                    dmaTimes.storeMs = dmaSince(dmaTimed, storeStart) - outcome.syncMs;
                    dmaTimes.flushMs = outcome.syncMs;
                    dmaTimes.reason = outcome.path;
                    if (outcome.path == 1 || outcome.path == 3) {
                        dmaTimes.outcome = 3;
                        wroteOnGpu = true;
                        drained = false;
                    }
                }
            }
        }
        if (drained) ++storesDrained;
    }

    static const bool drawDrain = std::getenv("APS5_DRAW_DRAIN") != nullptr;
    drawPacket = Pm4::DrawOpcode(opcode);
    sampleDump = opcode == 0x46 && (packet[1] & 0x3fu) == 0x39u;
    if (sampleDump && !drainAll) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (const auto localDevice = device.Load()) {
            recordDeferredLabels(localDevice.get(), submission.queue);
            wroteOnGpu = localDevice->DumpSamplesOnGpu(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u));
        }
    }

    static const bool syncFlip = std::getenv("APS5_SYNC_FLIP") != nullptr;
    const bool drains = drainAll ? ((Pm4::AccessesMemory(header) && opcode != 0x16) || opcode == 0x42 || opcode == 0x46 || opcode == 0x58 || header == FlipPacketHeader)
                                 : (!wroteOnGpu && !orderedAlready && (opcode == 0x49 || opcode == 0x37 || opcode == 0x40 || opcode == 0x45 || opcode == 0x50 || opcode == 0x83 || sampleDump || (drawPacket && drawDrain) || (header == FlipPacketHeader && syncFlip)));
    if (drains) {

        static const bool unlockedDrain = std::getenv("APS5_NO_UNLOCKED_DRAIN") == nullptr && !drainAll;

        bumpEpoch(&EpochBumps::drains);
        std::shared_ptr<VulkanDevice> draining;
        std::uint64_t epoch = 0;
        {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            if (profile) {
                ++drainCounts[header == FlipPacketHeader ? 0xffffu : opcode];
                ++drainTotal;
                if (std::chrono::steady_clock::now() - lastSyncReport > std::chrono::seconds(10)) {
                    lastSyncReport = std::chrono::steady_clock::now();
                    reportSync();
                }
            }
            draining = device.Load();

            recordDeferredLabels(draining.get(), submission.queue);
            if (draining != nullptr) {
                if (unlockedDrain && draining->CanWaitUnlocked()) epoch = draining->SubmitAndEpoch();
                else draining->WaitIdle();
            }
            if (epoch == 0) draining.reset();
        }
        if (epoch != 0) {
            ++unlockedDrains;
            try {
                draining->WaitRecorded(epoch);
            } catch (...) {

                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                draining.reset();
                throw;
            }
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            draining->ReapRecorded(epoch);
            draining.reset();
        }
        steps.Mark(PacketDrain);
    }
    if (dmaTimed) noteDma(packet, dmaTimes, dmaSince(true, dmaStart));
    return drains;
}

void Driver::dumpSampleCounters(std::uint64_t address) {
    std::uint64_t samples = 0;
    {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        auto* recorder = Graphics::Recorder::Active();
        require(recorder != nullptr, "occlusion counters without the command recorder are not implemented");
        recorder->CountSamples();
        samples = recorder->SamplesTotal();
    }
    constexpr std::uint64_t ready = 1ull << 63u;
    for (std::uint64_t db = 0; db < 16; ++db) {
        const std::uint64_t value = ready | (db == 0 ? samples : 0u);
        GuestMemory::Write(address + db * 16u, std::as_bytes(std::span(&value, 1)), 8);
    }
}

}
