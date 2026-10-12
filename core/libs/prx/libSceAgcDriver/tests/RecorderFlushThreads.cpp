#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace AgcDriver::Graphics;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

// A device whose queue runs a batch the moment it is submitted, and whose submit can be held.
struct MockDevice {
    std::mutex mutex;
    std::condition_variable changed;
    std::uintptr_t next = 1;
    std::map<VkFence, bool> signaled;
    std::vector<std::uint64_t> submitted;
    std::uint64_t timeline = 0;
    bool held = false;
    bool entered = false;
};

MockDevice mock;

VKAPI_ATTR VkResult VKAPI_CALL mockAllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer* commands) {
    std::lock_guard lock(mock.mutex);
    *commands = reinterpret_cast<VkCommandBuffer>(mock.next++);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mockFreeCommandBuffers(VkDevice, VkCommandPool, std::uint32_t, const VkCommandBuffer*) {}

VKAPI_ATTR VkResult VKAPI_CALL mockCreateFence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence* fence) {
    std::lock_guard lock(mock.mutex);
    *fence = reinterpret_cast<VkFence>(mock.next++);
    mock.signaled[*fence] = false;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mockDestroyFence(VkDevice, VkFence fence, const VkAllocationCallbacks*) {
    std::lock_guard lock(mock.mutex);
    mock.signaled.erase(fence);
}

VKAPI_ATTR VkResult VKAPI_CALL mockGetFenceStatus(VkDevice, VkFence fence) {
    std::lock_guard lock(mock.mutex);
    return mock.signaled.at(fence) ? VK_SUCCESS : VK_NOT_READY;
}

VKAPI_ATTR VkResult VKAPI_CALL mockWaitForFences(VkDevice, std::uint32_t, const VkFence* fences, VkBool32, std::uint64_t timeout) {
    std::unique_lock lock(mock.mutex);
    return mock.changed.wait_for(lock, std::chrono::nanoseconds(timeout), [&] { return mock.signaled.at(fences[0]); }) ? VK_SUCCESS : VK_TIMEOUT;
}

VKAPI_ATTR VkResult VKAPI_CALL mockResetFences(VkDevice, std::uint32_t count, const VkFence* fences) {
    std::lock_guard lock(mock.mutex);
    for (std::uint32_t i = 0; i < count; ++i) mock.signaled.at(fences[i]) = false;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mockBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mockEndCommandBuffer(VkCommandBuffer) {
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mockQueueSubmit(VkQueue, std::uint32_t, const VkSubmitInfo* submit, VkFence fence) {
    std::unique_lock lock(mock.mutex);
    mock.entered = true;
    mock.changed.notify_all();
    mock.changed.wait(lock, [&] { return !mock.held; });
    const auto* timeline = static_cast<const VkTimelineSemaphoreSubmitInfoKHR*>(submit->pNext);
    mock.submitted.push_back(timeline != nullptr ? timeline->pSignalSemaphoreValues[0] : 0);
    if (timeline != nullptr) mock.timeline = timeline->pSignalSemaphoreValues[0];
    mock.signaled.at(fence) = true;
    mock.changed.notify_all();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mockCreateSemaphore(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* semaphore) {
    std::lock_guard lock(mock.mutex);
    *semaphore = reinterpret_cast<VkSemaphore>(mock.next++);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mockDestroySemaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL mockWaitSemaphores(VkDevice, const VkSemaphoreWaitInfoKHR* wait, std::uint64_t timeout) {
    std::unique_lock lock(mock.mutex);
    return mock.changed.wait_for(lock, std::chrono::nanoseconds(timeout), [&] { return mock.timeline >= wait->pValues[0]; }) ? VK_SUCCESS : VK_TIMEOUT;
}

VKAPI_ATTR void VKAPI_CALL mockCmdUpdateBuffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*) {}
VKAPI_ATTR void VKAPI_CALL mockCmdPipelineBarrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL mockCmdBeginQuery(VkCommandBuffer, VkQueryPool, std::uint32_t, VkQueryControlFlags) {}
VKAPI_ATTR void VKAPI_CALL mockCmdEndQuery(VkCommandBuffer, VkQueryPool, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mockCmdResetQueryPool(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mockCmdCopyQueryPoolResults(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer, VkDeviceSize, VkDeviceSize, VkQueryResultFlags) {}
VKAPI_ATTR void VKAPI_CALL mockCmdBindPipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL mockCmdPushConstants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL mockCmdDispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}

PFN_vkVoidFunction VKAPI_CALL mockProc(VkDevice, const char* name) {
    static const std::map<std::string_view, PFN_vkVoidFunction> table{
        {"vkAllocateCommandBuffers", reinterpret_cast<PFN_vkVoidFunction>(mockAllocateCommandBuffers)},
        {"vkFreeCommandBuffers", reinterpret_cast<PFN_vkVoidFunction>(mockFreeCommandBuffers)},
        {"vkCreateFence", reinterpret_cast<PFN_vkVoidFunction>(mockCreateFence)},
        {"vkDestroyFence", reinterpret_cast<PFN_vkVoidFunction>(mockDestroyFence)},
        {"vkGetFenceStatus", reinterpret_cast<PFN_vkVoidFunction>(mockGetFenceStatus)},
        {"vkWaitForFences", reinterpret_cast<PFN_vkVoidFunction>(mockWaitForFences)},
        {"vkResetFences", reinterpret_cast<PFN_vkVoidFunction>(mockResetFences)},
        {"vkBeginCommandBuffer", reinterpret_cast<PFN_vkVoidFunction>(mockBeginCommandBuffer)},
        {"vkEndCommandBuffer", reinterpret_cast<PFN_vkVoidFunction>(mockEndCommandBuffer)},
        {"vkQueueSubmit", reinterpret_cast<PFN_vkVoidFunction>(mockQueueSubmit)},
        {"vkCreateSemaphore", reinterpret_cast<PFN_vkVoidFunction>(mockCreateSemaphore)},
        {"vkDestroySemaphore", reinterpret_cast<PFN_vkVoidFunction>(mockDestroySemaphore)},
        {"vkWaitSemaphoresKHR", reinterpret_cast<PFN_vkVoidFunction>(mockWaitSemaphores)},
        {"vkCmdUpdateBuffer", reinterpret_cast<PFN_vkVoidFunction>(mockCmdUpdateBuffer)},
        {"vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction>(mockCmdPipelineBarrier)},
        {"vkCmdBeginQuery", reinterpret_cast<PFN_vkVoidFunction>(mockCmdBeginQuery)},
        {"vkCmdEndQuery", reinterpret_cast<PFN_vkVoidFunction>(mockCmdEndQuery)},
        {"vkCmdResetQueryPool", reinterpret_cast<PFN_vkVoidFunction>(mockCmdResetQueryPool)},
        {"vkCmdCopyQueryPoolResults", reinterpret_cast<PFN_vkVoidFunction>(mockCmdCopyQueryPoolResults)},
        {"vkCmdBindPipeline", reinterpret_cast<PFN_vkVoidFunction>(mockCmdBindPipeline)},
        {"vkCmdPushConstants", reinterpret_cast<PFN_vkVoidFunction>(mockCmdPushConstants)},
        {"vkCmdDispatch", reinterpret_cast<PFN_vkVoidFunction>(mockCmdDispatch)},
    };
    const auto it = table.find(name);
    return it == table.end() ? nullptr : it->second;
}

Context mockContext() {
    Context context{};
    std::lock_guard lock(mock.mutex);
    context.device = reinterpret_cast<VkDevice>(mock.next++);
    context.queue = reinterpret_cast<VkQueue>(mock.next++);
    context.deviceProc = mockProc;
    return context;
}

constexpr std::uint64_t Batches = 8;
const bool threads = std::getenv("APS5_NO_FLUSH_THREADS") == nullptr && std::getenv("APS5_NO_SUBMIT_THREAD") == nullptr;
const bool reaps = std::getenv("APS5_NO_FLUSH_THREADS") == nullptr && std::getenv("APS5_REAP_THREAD") != nullptr;

void Hold(bool held) {
    std::lock_guard lock(mock.mutex);
    mock.held = held;
    mock.entered = false;
    mock.submitted.clear();
    mock.timeline = 0;
    mock.changed.notify_all();
}

bool WaitFor(const std::function<bool()>& reached) {
    std::unique_lock lock(mock.mutex);
    return mock.changed.wait_for(lock, std::chrono::seconds(20), reached);
}

struct Published {
    std::mutex mutex;
    std::vector<std::uint64_t> order;
    bool early = false;
    bool unlocked = false;
    std::atomic<std::uint64_t> count{0};
};

// One flush: a batch whose completion publishes `label`, as a label or an interrupt behind
// recorded work is published.
void Flush(Recorder& recorder, Published& published, std::uint64_t label) {
    recorder.OnComplete([&published, label] {
        bool early = false;
        {
            std::lock_guard device(mock.mutex);
            early = mock.submitted.size() < label;
        }
        {
            std::lock_guard lock(published.mutex);
            if (early) published.early = true;
            if (!AgcDriver::GuestMemory::GpuMutex().HeldByThisThread()) published.unlocked = true;
            published.order.push_back(label);
        }
        published.count.fetch_add(1);
        std::lock_guard device(mock.mutex);
        mock.changed.notify_all();
    });
    recorder.Submit();
}

bool InOrder(const std::vector<std::uint64_t>& values) {
    if (values.size() != Batches) return false;
    for (std::uint64_t i = 0; i < Batches; ++i) {
        if (values[i] != i + 1) return false;
    }
    return true;
}

// Labels of several flushes are published in the order of the flushes, none before its batch was
// submitted, and a thread waiting for a batch that is still queued for its submit sees it complete.
void OrderAndWaiter(bool reapThread) {
    Hold(threads);
    Published published;
    std::atomic<bool> waited{false};
    std::thread waiter;
    {
        auto recorder = std::make_unique<Recorder>(mockContext(), true);
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            recorder->Activate();
            for (std::uint64_t label = 1; label <= Batches; ++label) Flush(*recorder, published, label);
            Expect(recorder->Submissions() == Batches && recorder->InFlightBatches() == Batches, "the flushed batches are not all in flight");
            if (threads) {
                Expect(WaitFor([] { return mock.entered; }), "the submit thread never reached the device queue");
                std::lock_guard lock(mock.mutex);
                Expect(mock.submitted.empty(), "a held submit was made");
            }
            if (threads) Expect(!recorder->Reap(), "batches queued for their submit were retired");
            waiter = std::thread([&, serial = Batches - 2] {
                recorder->WaitSerial(serial);
                bool early = false;
                {
                    std::lock_guard lock(mock.mutex);
                    early = mock.timeline < serial;
                }
                std::lock_guard lock(published.mutex);
                if (early) published.early = true;
                waited = true;
            });
            if (threads) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                Expect(!waited, "a wait for a batch queued for its submit returned before the submit");
                std::lock_guard lock(published.mutex);
                Expect(published.order.empty(), "a label was published before its batch was submitted");
            }
        }
        {
            std::lock_guard lock(mock.mutex);
            mock.held = false;
            mock.changed.notify_all();
        }
        waiter.join();
        Expect(waited, "the wait for a queued batch did not return");
        if (reapThread && reaps) {
            Expect(WaitFor([&] { return published.count.load() == Batches; }), "the reap thread did not publish every label");
        }
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder->Sync();
        {
            std::lock_guard lock(mock.mutex);
            Expect(InOrder(mock.submitted), "the batches were not submitted once each in the order of their flushes");
        }
        recorder.reset();
    }
    std::lock_guard lock(published.mutex);
    Expect(InOrder(published.order), "the labels were not published once each in the order of their flushes");
    Expect(!published.early, "a label was published, or a wait returned, before the batch was submitted");
    Expect(!published.unlocked, "a label was published without the GPU mutex");
}

// A recorder torn down with batches queued for their submit and in flight publishes every label
// in order before it is gone, under the GPU mutex (as the driver tears a device down) or not.
void Shutdown(bool locked) {
    Hold(threads);
    Published published;
    auto recorder = std::make_unique<Recorder>(mockContext(), true);
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder->Activate();
        for (std::uint64_t label = 1; label <= Batches; ++label) Flush(*recorder, published, label);
    }
    std::thread release([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::lock_guard lock(mock.mutex);
        mock.held = false;
        mock.changed.notify_all();
    });
    if (locked) {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder.reset();
    } else {
        recorder.reset();
    }
    release.join();
    std::lock_guard lock(published.mutex);
    Expect(InOrder(published.order), "a recorder torn down with batches pending did not publish every label in order");
    Expect(!published.early, "a recorder torn down with batches pending published a label before its batch was submitted");
    std::lock_guard device(mock.mutex);
    Expect(InOrder(mock.submitted), "a recorder torn down with batches pending did not submit every batch in order");
}

// A thread that wants the GPU mutex while the reap thread runs a batch's completions waits for
// the one in progress and no more, and the rest run afterwards, once each and in order, on
// whichever thread finishes the batch.
void WorkerBehindAReap() {
    if (!reaps) return;
    Hold(false);
    constexpr std::uint64_t Steps = 100, Stop = 10;
    struct Run {
        std::mutex mutex;
        std::condition_variable changed;
        std::vector<std::uint64_t> order;
        bool reached = false;
        bool contended = false;
    } run;
    auto recorder = std::make_unique<Recorder>(mockContext(), true);
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder->Activate();
        for (std::uint64_t step = 1; step <= Steps; ++step) {
            recorder->OnComplete([&run, step] {
                std::unique_lock lock(run.mutex);
                run.order.push_back(step);
                if (step != Stop) return;
                run.reached = true;
                run.changed.notify_all();
                lock.unlock();
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                while (!AgcDriver::GuestMemory::GpuMutex().Contended() && std::chrono::steady_clock::now() < until) std::this_thread::yield();
                lock.lock();
                run.contended = AgcDriver::GuestMemory::GpuMutex().Contended();
            });
        }
        recorder->Submit();
    }
    {
        std::unique_lock lock(run.mutex);
        Expect(run.changed.wait_for(lock, std::chrono::seconds(20), [&] { return run.reached; }), "the reap thread did not run the batch's completions");
    }
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        {
            std::lock_guard lock(run.mutex);
            Expect(run.contended, "the completion in progress did not see the waiting thread");
            Expect(run.order.size() == Stop, "a thread waiting for the GPU mutex waited behind " + std::to_string(run.order.size() - std::min<std::size_t>(run.order.size(), Stop)) + " more completions than the one in progress");
        }
        Expect(recorder->InFlightBatches() == 1, "the batch the reap thread gave up is not in flight");
        recorder->Sync();
        std::lock_guard lock(run.mutex);
        bool ordered = run.order.size() == Steps;
        for (std::uint64_t step = 0; ordered && step < Steps; ++step) ordered = run.order[step] == step + 1;
        Expect(ordered, "the completions of a batch given up by the reap thread did not run once each in order");
        recorder.reset();
    }
}

// With the reap thread kept from the mutex, the worker leaves its own reaps to it only while
// fewer than eight batches are in flight, and reaps them itself in order after that.
void StarvedReapThread() {
    Hold(false);
    Published published;
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    auto recorder = std::make_unique<Recorder>(mockContext(), true);
    recorder->Activate();
    for (std::uint64_t label = 1; label < Batches; ++label) Flush(*recorder, published, label);
    Expect(recorder->ReapLeftToThread() == reaps, reaps ? "seven batches in flight were not left to the reap thread" : "reaps were left to a reap thread that is switched off");
    Flush(*recorder, published, Batches);
    Expect(!recorder->ReapLeftToThread(), "eight batches in flight were still left to the reap thread");
    Expect(WaitFor([] { return mock.submitted.size() == Batches; }), "the batches were not submitted");
    Expect(recorder->Reap(), "the worker's own reap did not retire the finished batches");
    {
        std::lock_guard lock(published.mutex);
        Expect(InOrder(published.order), "the worker's own reap did not publish every label in order");
    }
    recorder.reset();
}

}

int main() {
    try {
        OrderAndWaiter(false);
        Shutdown(true);
        Shutdown(false);
        Recorder::AllowReapThread();
        for (int round = 0; round < 8; ++round) {
            OrderAndWaiter(true);
            Shutdown(round % 2 == 0);
            WorkerBehindAReap();
            StarvedReapThread();
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d recorder flush thread checks failed\n", failures);
        return 1;
    }
    std::puts("recorder flush thread tests passed");
    return 0;
}
