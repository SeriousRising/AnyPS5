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
        // 64 blocks of other ranges: a collect of a page is widened to its block, so that many
        // overrun the per-thread memo.
        WatchedBlock other(64 * BlockBytes);
        // The blocks are released with nothing recorded over them, also when a check fails.
        struct Idle {
            AgcDriver::VulkanDevice& device;
            ~Idle() { device.WaitIdle(); }
        } idle{*device};
        Scene scene(*device, target.Address(), values.Address(), {Width, Height});
        auto& color = scene.color;
        Require(color.bytes == BlockBytes, "resident target lookup: the target is not one block");
        const auto walkOthers = [&] {
            for (std::size_t block = 0; block < 64; ++block) CollectWrites(other.Address() + block * BlockBytes, PageBytes);
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
