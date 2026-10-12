#include "ResidentTargetScene.hpp"
#include <cstdlib>
#include <initializer_list>
#include <string>

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
        WatchedBlock second(BlockBytes);
        WatchedBlock compressed(BlockBytes);
        WatchedBlock keys(BlockBytes);
        WatchedBlock spare(BlockBytes);
        struct Idle {
            AgcDriver::VulkanDevice& device;
            ~Idle() { device.WaitIdle(); }
        } idle{*device};
        Scene scene(*device, target.Address(), values.Address(), {Width, Height});
        auto& color = scene.color;
        Require(color.bytes == BlockBytes, "resident target early return: the target is not one block");

        BumpCollectEpoch();
        if (!scene.Draw(color, true, 0)) {
            std::puts("draws are not recorded on this device: the early return of a resident target's lookup not tested");
            return 0;
        }
        if (!Watched(color.address, color.bytes)) {
            std::puts("the imported target is not write-watched on this device: the early return of its lookup not tested");
            return 0;
        }
        const auto image = StorageTexture::FindPending(color.address, color.bytes);
        Require(image != nullptr, "resident target early return: the drawn target has no resident image");
        const auto expect = [&](StorageTexture& looked, bool proved, bool current, const char* what) {
            const auto before = StorageTexture::RefreshesProved();
            const bool held = scene.Refresh(looked).current;
            const bool answered = StorageTexture::RefreshesProved() != before;
            if (held != current) throw std::runtime_error(std::string("resident target early return: ") + what + ": the lookup found the image " + (held ? "current" : "changed"));
            if (answered != proved) throw std::runtime_error(std::string("resident target early return: ") + what + ": the lookup was " + (answered ? "answered from the kept proof" : "refreshed in full"));
        };

        scene.Settle(*image);
        expect(*image, true, true, "an unchanged target");
        expect(*image, true, true, "an unchanged target looked up again");
        const auto walked = ThreadCollectedBytes();
        Require(scene.Draw(color, false, Width / 2) && ThreadCollectedBytes() == walked, "resident target early return: a draw into an unchanged target walked pages");
        Require(scene.Holds(color, Width / 2, White, 0u), "resident target early return: draws into an unchanged target lost the draws before them");

        scene.Settle(*image);
        BumpCollectEpoch();
        expect(*image, true, true, "a new collect epoch");
        expect(*image, true, true, "the lookup after a new collect epoch");

        target.Data()[5 * PageBytes + 3] ^= 0xffu;
        expect(*image, true, true, "a CPU store inside the epoch");
        BumpCollectEpoch();
        expect(*image, false, false, "a CPU store into one page, at the next epoch");
        scene.Settle(*image);
        std::memset(target.Data(), 0x5a, BlockBytes);
        BumpCollectEpoch();
        Require(scene.Draw(color, true, Width / 2) && scene.Holds(color, Width / 2, 0x5a5a5a5au, White), "resident target early return: a draw after a CPU store did not keep the stored texels");

        scene.Settle(*image);
        expect(*image, true, true, "the target settled after a CPU store");
        StoreOwnBytes(color.address + 7 * PageBytes, PageBytes, [&] { std::memset(target.Data() + 7 * PageBytes, 0x11, PageBytes); });
        expect(*image, false, false, "a driver store into one page");
        scene.Settle(*image);
        StoreOwnBytes(color.address, BlockBytes, [&] { std::memset(target.Data(), 0x22, BlockBytes); });
        Require(scene.Draw(color, false, Width / 2) && scene.Holds(color, Width / 2, 0x22222222u, 0u), "resident target early return: a draw after a driver store did not keep the stored texels");

        const std::array<std::uint32_t, 4> pattern{0x44444444u, 0x44444444u, 0x44444444u, 0x44444444u};
        scene.Settle(*image);
        expect(*image, true, true, "the target settled after a driver store");
        if (device->FillBuffer(color.address, BlockBytes, pattern)) {
            expect(*image, false, false, "a recorded GPU store into the target");
            Require(scene.Draw(color, true, Width / 2) && scene.Holds(color, Width / 2, 0x44444444u, White), "resident target early return: a draw after a GPU store did not keep the stored texels");
        } else {
            std::puts("fills are not recorded on this device: a GPU store into a resident target not tested");
        }

        scene.Settle(*image);
        expect(*image, true, true, "the target settled after a GPU store");
        auto other = color;
        other.address = second.Address();
        Require(scene.Draw(other, true, 0), "resident target early return: the draw into a second target was not recorded");
        expect(*image, true, true, "another image with results pending since");
        expect(*image, true, true, "the lookup after another image's results");

        Unwatch(spare.Address(), BlockBytes);
        expect(*image, true, true, "memory unwatched since");
        expect(*image, true, true, "the lookup after memory was unwatched");

        auto dcc = color;
        dcc.address = compressed.Address();
        dcc.dccAddress = keys.Address();
        const auto keyBytes = AgcDriver::Graphics::DccKeyBytes(dcc.bytes);
        std::memset(keys.Data(), 0xff, keyBytes);
        BumpCollectEpoch();
        Require(scene.Draw(dcc, true, 0), "resident target early return: the draw into a compressed target was not recorded");
        const auto keyed = StorageTexture::FindPending(dcc.address, dcc.bytes);
        Require(keyed != nullptr, "resident target early return: the compressed target has no resident image");
        scene.Settle(*keyed);
        scene.Refresh(*keyed);
        expect(*keyed, true, true, "an unchanged compressed target");
        StoreOwnBytes(keys.Address(), keyBytes, [&] { std::memset(keys.Data(), 0xff, keyBytes); });
        expect(*keyed, false, true, "a driver store of the same keys");
        scene.Refresh(*keyed);
        expect(*keyed, true, true, "the compressed target after its keys were proved again");
        const std::array<std::uint32_t, 4> uncompressed{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
        if (device->FillBuffer(keys.Address(), keyBytes, uncompressed)) expect(*keyed, false, true, "a recorded fill of the keys");
        else std::puts("fills are not recorded on this device: a recorded fill of a target's keys not tested");
        Require(scene.Holds(dcc, Width / 2, White, White), "resident target early return: the compressed target lost its draw");
        std::puts("resident target early return tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
