#include "ResidentTargetScene.hpp"
#include <chrono>
#include <cstdlib>
#include <functional>
#include <string>

namespace {

using namespace ResidentTargetScene;
using AgcDriver::Graphics::DccKeys;

constexpr std::uint32_t Frames = 8;
constexpr std::uint32_t DrawsPerFrame = 16;

std::size_t Blocks(std::size_t bytes) {
    return (bytes + BlockBytes - 1) / BlockBytes * BlockBytes;
}

struct Cost {
    double firstMs = 0;
    double restMs = 0;
    std::uint64_t waits = 0;
    std::uint64_t pages = 0;
};

void Print(const char* shape, const Cost& cost) {
    std::printf("%s: first draw of a frame %.3f ms, each later draw %.4f ms, %.2f batches waited for a frame, %llu pages walked a frame\n", shape, cost.firstMs / Frames, cost.restMs / (Frames * (DrawsPerFrame - 1)), static_cast<double>(cost.waits) / Frames, static_cast<unsigned long long>(cost.pages / Frames));
}

}

int main(int count, char** arguments) {
    try {
        const VkExtent2D Extent{count > 2 ? static_cast<std::uint32_t>(std::atoi(arguments[1])) : 1920u, count > 2 ? static_cast<std::uint32_t>(std::atoi(arguments[2])) : 1080u};
        using namespace AgcDriver::GuestMemory;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        std::lock_guard gpu(GpuMutex());
        if (!WriteWatched()) {
            std::puts("no write watching: skipped");
            return VulkanTestSkipped;
        }
        const AgcDriver::Graphics::ColorTargetLayout layout(Extent.width, Extent.height, AgcDriver::Graphics::ColorTileMode::RenderTarget);
        const auto keyBytes = AgcDriver::Graphics::DccKeyBytes(layout.Bytes());
        const auto cmaskBytes = AgcDriver::Graphics::CmaskBytes(Extent.width, Extent.height);
        WatchedBlock values(BlockBytes);
        const float one = 1.0f;
        std::memcpy(values.Data(), &one, sizeof(one));
        WatchedBlock compressed(Blocks(layout.Bytes()));
        WatchedBlock keys(Blocks(keyBytes));
        WatchedBlock masked(Blocks(layout.Bytes()));
        WatchedBlock cmask(Blocks(cmaskBytes));
        struct Idle {
            AgcDriver::VulkanDevice& device;
            ~Idle() { device.WaitIdle(); }
        } idle{*device};
        Scene scene(*device, compressed.Address(), values.Address(), Extent);
        const auto left = Extent.width / 2;

        auto dcc = scene.color;
        dcc.dccAddress = keys.Address();
        std::memset(keys.Data(), 0xff, keyBytes);
        auto cleared = scene.color;
        cleared.address = masked.Address();
        cleared.cmaskAddress = cmask.Address();
        cleared.cmaskBytes = cmaskBytes;
        std::memset(cmask.Data(), 0xff, cmaskBytes);

        BumpCollectEpoch();
        if (!scene.Draw(dcc, true, 0) || !scene.Draw(cleared, true, 0)) {
            std::puts("draws are not recorded on this device: waits of a resident target's lookups not tested");
            return 0;
        }
        auto* recorder = AgcDriver::Graphics::Recorder::Active();
        const std::array<std::uint32_t, 4> zeros{};
        const std::array<std::uint32_t, 4> registerKeys{0x20202020u, 0x20202020u, 0x20202020u, 0x20202020u};
        if (recorder == nullptr || !device->FillBuffer(cmask.Address(), cmaskBytes, zeros)) {
            std::puts("fills are not recorded on this device: waits of a resident target's lookups not tested");
            return 0;
        }
        Require(scene.Draw(cleared, false, left) && scene.Holds(cleared, left, 0u, 0u), "resident target waits: the first CMASK clear did not clear its target");
        Require(scene.Holds(dcc, left, 0xffffffffu, 0xffffffffu), "resident target waits: the compressed target did not keep its first draw");

        // What a frame costs its queue worker: the clear the title records, the draw that turns
        // it into texels, and the draws behind it.
        const auto frames = [&](const AgcDriver::Graphics::ColorTarget& target, const std::function<void(std::uint32_t)>& clear, bool epochs) {
            Cost cost;
            Require(scene.Draw(target, false, left), "resident target waits: a draw was not recorded");
            device->WaitIdle();
            for (std::uint32_t frame = 0; frame < Frames; ++frame) {
                BumpCollectEpoch();
                auto color = target;
                color.clearWords = {0x01010101u * (frame + 1), 0u};
                clear(frame);
                const auto submissions = recorder->Submissions();
                const auto walked = ThreadCollectedBytes();
                const auto start = std::chrono::steady_clock::now();
                Require(scene.Draw(color, false, left), "resident target waits: a draw was not recorded");
                const auto first = std::chrono::steady_clock::now();
                cost.waits += recorder->Submissions() - submissions;
                for (std::uint32_t draw = 1; draw < DrawsPerFrame; ++draw) {
                    if (epochs) BumpCollectEpoch();
                    Require(scene.Draw(color, false, left), "resident target waits: a draw was not recorded");
                }
                cost.firstMs += std::chrono::duration<double, std::milli>(first - start).count();
                cost.restMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - first).count();
                cost.pages += (ThreadCollectedBytes() - walked) / PageBytes;
                device->WaitIdle();
            }
            return cost;
        };
        const auto noClear = [](std::uint32_t) {};
        const auto registerClear = [&](std::uint32_t) {
            StorageTexture::NoteKeysFill(keys.Address(), keyBytes, 0x20);
            Require(device->FillBuffer(keys.Address(), keyBytes, registerKeys), "resident target waits: the key fill was not recorded");
            AgcDriver::Graphics::NoteKeysFillOnGpu(keys.Address(), keyBytes, DccKeys::ClearRegister);
        };
        const auto cmaskClear = [&](std::uint32_t) { Require(device->FillBuffer(cmask.Address(), cmaskBytes, zeros), "resident target waits: the CMASK fill was not recorded"); };
        const auto halves = [&](std::uint32_t) {
            const auto half = cmaskBytes / 2 & ~std::size_t{15};
            Require(device->FillBuffer(cmask.Address(), half, zeros) && device->FillBuffer(cmask.Address() + half, cmaskBytes - half, zeros), "resident target waits: the CMASK fills were not recorded");
        };
        const auto expanded = [](WatchedBlock& block, std::size_t bytes) { return std::all_of(block.Data(), block.Data() + bytes, [](std::uint8_t byte) { return byte == 0xff; }); };
        const std::uint32_t last = 0x01010101u * Frames;
        // Every shape is measured before any of them fails the run.
        const bool clearBehindKeys = std::getenv("APS5_NO_CLEAR_BEHIND_KEYS") == nullptr;
        const bool joinedFills = std::getenv("APS5_NO_JOINED_KEY_FILLS") == nullptr;
        std::vector<std::string> failures;
        const auto Check = [&](bool held, const char* message) {
            if (!held) failures.emplace_back(message);
        };

        Print("no clear, one epoch a frame", frames(dcc, noClear, false));
        const auto perDraw = frames(dcc, noClear, true);
        Print("no clear, an epoch a draw", perDraw);

        auto cost = frames(dcc, registerClear, false);
        Print("register clear behind a recorded key fill", cost);
        Check(scene.Holds(dcc, left, last, 0u) && expanded(keys, keyBytes), "resident target waits: a register clear behind a recorded key fill did not leave the clear words of its target, or left the keys cleared");
        Check(cost.waits == (clearBehindKeys ? 0u : Frames), clearBehindKeys ? "resident target waits: a draw waited for the recorded fill of its target's DCC keys" : "resident target waits: the switch did not make the draws wait for the recorded fill of their target's DCC keys");

        cost = frames(cleared, cmaskClear, false);
        Print("CMASK clear behind a recorded fill", cost);
        Check(scene.Holds(cleared, left, last, 0u) && expanded(cmask, cmaskBytes), "resident target waits: a draw behind a recorded CMASK clear did not leave the clear words of its target, or left the CMASK cleared");
        Check(cost.waits == 0, "resident target waits: a draw waited for the recorded fill that cleared its CMASK");

        // A write of unknown content recorded before the fills is overwritten by them.
        cost = frames(cleared, [&](std::uint32_t frame) {
            recorder->NotePendingWrite(cmask.Address(), cmaskBytes);
            halves(frame);
        }, false);
        Print("CMASK clear behind two recorded fills", cost);
        Check(scene.Holds(cleared, left, last, 0u) && expanded(cmask, cmaskBytes), "resident target waits: a draw behind two recorded CMASK fills did not leave the clear words of its target, or left the CMASK cleared");
        Check(cost.waits == (joinedFills ? 0u : Frames), joinedFills ? "resident target waits: a draw waited for the two recorded fills that cleared its CMASK" : "resident target waits: the switch did not make the draws wait for the two recorded fills of their CMASK");

        // What the target holds around a register clear is the same whether or not the draw
        // waited for the key fill.
        const std::array<std::uint32_t, 4> stored{0x44444444u, 0x44444444u, 0x44444444u, 0x44444444u};
        dcc.clearWords = {0x0badf00du, 0u};
        Check(device->FillBuffer(dcc.address, dcc.bytes, stored), "resident target waits: the store into the target was not recorded");
        registerClear(0);
        Check(scene.Draw(dcc, false, left) && scene.Holds(dcc, left, 0x0badf00du, 0u), "resident target waits: a register clear did not replace what the GPU stored into its target before it");
        registerClear(0);
        Check(scene.Draw(dcc, false, left) && device->FillBuffer(dcc.address, dcc.bytes, stored), "resident target waits: the store behind the clear was not recorded");
        Check(scene.Draw(dcc, false, left) && scene.Holds(dcc, left, 0x44444444u, 0u), "resident target waits: a draw did not keep what the GPU stored into its target behind a register clear");
        registerClear(0);
        Check(scene.Draw(dcc, false, left), "resident target waits: a draw was not recorded");
        std::memset(compressed.Data(), 0x5a, dcc.bytes);
        BumpCollectEpoch();
        Check(scene.Draw(dcc, false, left) && scene.Holds(dcc, left, 0x5a5a5a5au, 0u), "resident target waits: a draw did not keep what the CPU stored into its target behind a register clear");
        // Keys written behind the fill by something that does not say what it stores are waited
        // for and read, as are a CMASK's.
        registerClear(0);
        recorder->NotePendingWrite(keys.Address(), keyBytes);
        auto submissions = recorder->Submissions();
        Check(scene.Draw(dcc, false, left) && recorder->Submissions() != submissions, "resident target waits: a draw did not wait for a write of unknown content into its target's DCC keys");
        Check(scene.Holds(dcc, left, 0x0badf00du, 0u) && expanded(keys, keyBytes), "resident target waits: a register clear behind a write of unknown content did not leave the clear words of its target, or left the keys cleared");
        halves(0);
        recorder->NotePendingWrite(cmask.Address(), 16);
        submissions = recorder->Submissions();
        cleared.clearWords = {0x0badf00du, 0u};
        Check(scene.Draw(cleared, false, left) && recorder->Submissions() != submissions, "resident target waits: a draw did not wait for a write of unknown content into its CMASK");
        Check(scene.Holds(cleared, left, 0x0badf00du, 0u) && expanded(cmask, cmaskBytes), "resident target waits: a CMASK clear behind a write of unknown content did not leave the clear words of its target, or left the CMASK cleared");
        for (const auto& failure : failures) std::cerr << failure << '\n';
        if (!failures.empty()) return 1;
        std::puts("resident target waits tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
