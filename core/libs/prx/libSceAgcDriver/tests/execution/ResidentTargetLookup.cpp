#include "ResidentTargetScene.hpp"

namespace {

using namespace ResidentTargetScene;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 64;
constexpr std::uint32_t White = 0xffffffffu;

}

int main() {
    try {
        using namespace AgcDriver::GuestMemory;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        std::lock_guard gpu(GpuMutex());
        if (!WriteWatched()) {
            std::puts("no write watching: skipped");
            return VulkanTestSkipped;
        }
        WatchedBlock values(BlockBytes);
        const float one = 1.0f;
        std::memcpy(values.Data(), &one, sizeof(one));
        WatchedBlock target(BlockBytes);
        // The CMASK in the first block; 64 pages of other ranges behind it.
        WatchedBlock other(5 * BlockBytes);
        // The blocks are released with nothing recorded over them, also when a check fails.
        struct Idle {
            AgcDriver::VulkanDevice& device;
            ~Idle() { device.WaitIdle(); }
        } idle{*device};
        Scene scene(*device, target.Address(), values.Address(), {Width, Height});
        auto& color = scene.color;
        Require(color.bytes == BlockBytes, "resident target lookup: the target is not one block");
        const auto walkOthers = [&] {
            for (std::size_t page = 0; page < 64; ++page) CollectWrites(other.Address() + BlockBytes + page * PageBytes, PageBytes);
        };

        BumpCollectEpoch();
        if (!scene.Draw(color, true, 0)) {
            std::puts("draws are not recorded on this device: lookups of a resident target not tested");
            return 0;
        }
        // An import that does not keep write protection is compared instead: nothing is walked.
        const bool watched = Watched(color.address, color.bytes);
        if (!watched) std::puts("the imported target is not write-watched on this device: its page walks not tested");
        const auto image = StorageTexture::FindPending(color.address, color.bytes);
        Require(image != nullptr, "resident target lookup: the drawn target has no resident image");
        scene.Settle(*image);
        auto lookup = scene.Refresh(*image);
        Require(lookup.current && !lookup.walked, "resident target lookup: a lookup repeated in one submission walked the target's pages");
        // A frame's draws collect far more ranges between two lookups of a target than the
        // per-thread memo holds.
        walkOthers();
        lookup = scene.Refresh(*image);
        Require(lookup.current && !lookup.walked, "resident target lookup: walks of 64 other ranges made the next lookup walk the target's pages again");
        walkOthers();
        const auto beforeDraw = TrackerGeneration();
        Require(scene.Draw(color, false, Width / 2) && TrackerGeneration() == beforeDraw, "resident target lookup: a draw into a target looked up before in its submission walked pages");
        Require(scene.Holds(color, Width / 2, White, 0u), "resident target lookup: draws after remembered walks did not keep what the draws before them drew");

        // A CPU store is collected at the next ordering point, and wins over nothing pending.
        scene.Settle(*image);
        std::memset(target.Data(), 0x5a, BlockBytes);
        BumpCollectEpoch();
        lookup = scene.Refresh(*image);
        Require(!lookup.current && lookup.walked == watched, "resident target lookup: a CPU store into the target was not collected at the next ordering point");
        Require(scene.Draw(color, true, Width / 2) && scene.Holds(color, Width / 2, 0x5a5a5a5au, White), "resident target lookup: a draw after a CPU store did not keep the stored texels");

        // The driver's own store stamps its blocks: it is seen inside the epoch, without a walk.
        scene.Settle(*image);
        walkOthers();
        StoreOwnBytes(color.address, BlockBytes, [&] { std::memset(target.Data(), 0x22, BlockBytes); });
        Require(!scene.Refresh(*image).current, "resident target lookup: a driver store into the target was missed inside its epoch");
        Require(scene.Draw(color, false, Width / 2) && scene.Holds(color, Width / 2, 0x22222222u, 0u), "resident target lookup: a draw after a driver store did not keep the stored texels");

        // A store the GPU makes into the target's memory, where the device records one.
        const std::array<std::uint32_t, 4> pattern{0x44444444u, 0x44444444u, 0x44444444u, 0x44444444u};
        scene.Settle(*image);
        walkOthers();
        if (device->FillBuffer(color.address, BlockBytes, pattern)) Require(scene.Draw(color, true, Width / 2) && scene.Holds(color, Width / 2, 0x44444444u, White), "resident target lookup: a draw after a GPU store did not keep the stored texels");
        else std::puts("fills are not recorded on this device: a GPU store into a resident target not tested");

        // A fast clear through the CMASK takes the clear words of the draw that eliminates it.
        auto cleared = color;
        cleared.cmaskAddress = other.Address();
        cleared.cmaskBytes = AgcDriver::Graphics::CmaskBytes(Width, Height);
        const auto expanded = [&] { return std::all_of(other.Data(), other.Data() + cleared.cmaskBytes, [](std::uint8_t byte) { return byte == 0xff; }); };
        for (const auto word : {0x11223344u, 0x55667788u}) {
            cleared.clearWords = {word, 0u};
            scene.Settle(*image);
            walkOthers();
            std::memset(other.Data(), 0, cleared.cmaskBytes);
            Require(scene.Draw(cleared, false, Width / 2) && scene.Holds(cleared, Width / 2, word, 0u) && expanded(), "resident target lookup: a draw over a fast-cleared CMASK did not leave the clear words of its target, or left the CMASK cleared");
        }

        // The fill a title clears its CMASK with is recorded: the draw behind it knows what the
        // fill stores and waits for nothing. Only the newest write over all of the CMASK says so.
        auto* recorder = AgcDriver::Graphics::Recorder::Active();
        const std::array<std::uint32_t, 4> zeros{};
        const std::array<std::uint32_t, 4> ones{~0u, ~0u, ~0u, ~0u};
        const auto cmaskKeys = [&] { return AgcDriver::Graphics::CurrentDccKeys(cleared.cmaskAddress, cleared.cmaskBytes * 256u); };
        cleared.clearWords = {0x99aabbccu, 0u};
        scene.Settle(*image);
        if (recorder != nullptr && device->FillBuffer(cleared.cmaskAddress, cleared.cmaskBytes, zeros)) {
            auto submissions = recorder->Submissions();
            Require(scene.Draw(cleared, false, Width / 2) && recorder->Submissions() == submissions, "resident target lookup: a draw waited for the recorded fill that cleared its CMASK");
            Require(scene.Holds(cleared, Width / 2, 0x99aabbccu, 0u) && expanded(), "resident target lookup: a draw behind a recorded CMASK clear did not leave the clear words of its target, or left the CMASK cleared");
            Require(device->FillBuffer(cleared.cmaskAddress, cleared.cmaskBytes, zeros) && device->FillBuffer(cleared.cmaskAddress, cleared.cmaskBytes, ones), "resident target lookup: the CMASK fills were not recorded");
            submissions = recorder->Submissions();
            Require(cmaskKeys() == AgcDriver::Graphics::DccKeys::Uncompressed && recorder->Submissions() == submissions, "resident target lookup: the newest of two recorded fills did not answer for the CMASK");
            Require(device->FillBuffer(cleared.cmaskAddress, 16, zeros), "resident target lookup: the partial CMASK fill was not recorded");
            Require(cmaskKeys() == AgcDriver::Graphics::DccKeys::Mixed && recorder->Submissions() == submissions, "resident target lookup: a fill of part of the CMASK over a fill of all of it did not answer for the CMASK");
            // Nothing recorded says what the rest of the CMASK holds: the bytes do, once the fill landed.
            device->WaitIdle();
            Require(device->FillBuffer(cleared.cmaskAddress, 16, zeros), "resident target lookup: the partial CMASK fill was not recorded");
            submissions = recorder->Submissions();
            Require(cmaskKeys() == AgcDriver::Graphics::DccKeys::Mixed && recorder->Submissions() != submissions, "resident target lookup: a fill of part of the CMASK was not waited for and read");
            device->WaitIdle();
        } else {
            std::puts("fills are not recorded on this device: a recorded CMASK clear not tested");
        }

        // The same memory as a target of another size is another image, which reads what the
        // first one drew.
        Require(scene.Draw(color, true, 0), "resident target lookup: the draw before the resize was not recorded");
        auto resized = color;
        resized.extent = {Width, Height / 2};
        resized.bytes = AgcDriver::Graphics::ColorTargetLayout(Width, Height / 2, AgcDriver::Graphics::ColorTileMode::RenderTarget).Bytes();
        Require(scene.Draw(resized, false, Width / 2), "resident target lookup: the draw into the resized target was not recorded");
        const auto second = StorageTexture::FindPending(resized.address, resized.bytes);
        Require(second != nullptr && second != image && second->Descriptor().height == Height / 2, "resident target lookup: the resized target has no image of its own");
        Require(scene.Holds(resized, Width / 2, White, 0u), "resident target lookup: the resized target did not keep what the first one drew");
        std::puts("resident target lookup tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
