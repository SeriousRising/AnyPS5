#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SYNCHRONIZATIONSTATISTICS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SYNCHRONIZATIONSTATISTICS_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>

namespace AgcDriver::DriverDetail {

struct SubmissionCosts {
    std::atomic<std::uint64_t> submissions{0}, validateNs{0}, copyNs{0}, dequeueNs{0}, completeNs{0}, suspendNs{0}, suspends{0}, suspendsSkipped{0}, endSubmits{0}, endSkipped{0}, notifiesSkipped{0};
};

extern std::atomic<std::uint64_t> gpuLabels, completionLabels, noOpLabels, unlockedDrains;

extern std::atomic<std::uint64_t> labelFallbacks[5];

extern std::atomic<std::uint64_t> queuedLabels, labelGroups, immediateLabels;

extern std::atomic<std::uint64_t> packetLockRecords, packetLockSubmits, packetLockDeferred, packetSubmitDeferred, captureTryRecords, captureRetries, suspendPoints;

extern std::atomic<std::uint64_t> recordTriesSkipped;

extern std::atomic<bool> queue0Dormant;

extern std::atomic<std::uint64_t> packetLockRecordUs, packetLockSubmitUs;

enum TrySite { TryLabel = 0, TryFlush, TryReap, TryPoll, TryIdle, TryCapture, TryPollRecord, TrySites };

extern std::atomic<std::uint64_t> triesFailed[TrySites];

extern std::atomic<std::uint64_t> storesOnGpu, storesBehindCompletions, storesOnCpu, storesDrained, copiesToHostMemory;

extern std::map<std::uint32_t, std::uint64_t> drainCounts;

extern std::uint64_t drainTotal;

extern std::chrono::steady_clock::time_point lastSyncReport;

extern std::atomic<std::uint64_t> pollLabelRecords, hookLabelRecords;

enum LabelStep { FlushIdle = 0, FlushLockWait, FlushTryFailed, FlushRecord, FlushSubmit, FlushReap, LabelQueued, LabelNoOp, LabelLocked, InterruptLockWait, InterruptRecord, InterruptStore, InterruptCompletion, InterruptSubmit, PacketDrain, LabelSteps };

inline constexpr std::array<const char*, LabelSteps> LabelStepNames{"flush nothing due", "flush lock wait", "flush try failed", "flush record", "flush submit", "flush reap", "label queued", "label no-op", "label under the lock", "interrupt lock wait", "interrupt record queued", "interrupt store", "interrupt completion", "interrupt submit", "drain"};

struct LabelStepProfile {
    std::array<std::uint64_t, LabelSteps> count{}, ns{};
};

LabelStepProfile& labelSteps();

struct LabelStepTimer {
    bool enabled;
    std::chrono::steady_clock::time_point at;
    explicit LabelStepTimer(bool enabled) : enabled(enabled), at(enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    void Mark(LabelStep step) {
        if (!enabled) return;
        const auto now = std::chrono::steady_clock::now();
        auto& profile = labelSteps();
        ++profile.count[step];
        profile.ns[step] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - at).count());
        at = now;
    }
};

SubmissionCosts& submissionCosts(std::uint32_t queue);

void reportSync();

void countLabelOutcome(int reason);

}

#endif
