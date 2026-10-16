#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

// The split-recording gate (GuestMemory::SetGpuLockGate): it runs before the installing thread's
// outermost GPU-mutex acquisitions only, makes a refused try fail without taking the mutex, and is
// per thread.

namespace {

using namespace AgcDriver::GuestMemory;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

int waits = 0;
int tries = 0;
bool allow = true;

bool Gate(bool wait) {
    if (wait) ++waits;
    else ++tries;
    return wait || allow;
}

void CheckOutermostOnly() {
    SetGpuLockGate(&Gate);
    waits = tries = 0;
    {
        std::lock_guard outer(GpuMutex());
        Require(waits == 1, "lock() did not wait in the gate first");
        std::lock_guard nested(GpuMutex());
        Require(GpuMutex().try_lock(), "a nested try failed");
        GpuMutex().unlock();
        Require(waits == 1 && tries == 0, "a nested acquisition ran the gate");
    }
    SetGpuLockGate(nullptr);
}

void CheckRefusedTry() {
    SetGpuLockGate(&Gate);
    waits = tries = 0;
    allow = false;
    Require(!GpuMutex().try_lock(), "a try the gate refused took the mutex");
    Require(tries == 1 && !GpuMutex().HeldByThisThread(), "a refused try left the mutex held or skipped the gate");
    allow = true;
    Require(GpuMutex().try_lock(), "a try the gate allowed failed");
    Require(tries == 2 && GpuMutex().HeldByThisThread(), "an allowed try did not hold the mutex");
    GpuMutex().unlock();
    SetGpuLockGate(nullptr);
}

void CheckPerThread() {
    SetGpuLockGate(&Gate);
    waits = tries = 0;
    std::thread other([] {
        std::lock_guard lock(GpuMutex());
        Require(GpuMutex().try_lock(), "another thread's nested try failed");
        GpuMutex().unlock();
    });
    other.join();
    Require(waits == 0 && tries == 0, "another thread's acquisitions ran this thread's gate");
    SetGpuLockGate(nullptr);
    std::lock_guard lock(GpuMutex());
    Require(waits == 0, "a removed gate still ran");
}

}

int main() {
    try {
        CheckOutermostOnly();
        CheckRefusedTry();
        CheckPerThread();
    } catch (const std::exception& error) {
        std::cerr << "GPU lock gate test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "GPU lock gate tests passed\n";
    return 0;
}
