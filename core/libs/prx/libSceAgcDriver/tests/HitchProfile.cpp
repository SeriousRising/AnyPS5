#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Packets/PacketTiming.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void timestampPacketTests() {
    using AgcDriver::DriverDetail::TimestampPacketCounts;
    TimestampPacketCounts counts;
    std::array<std::uint32_t, 8> packet{};
    counts.Note(0x49, std::span(packet).first(6));
    counts.Note(0x49, std::span(packet).first(7));
    counts.Note(0x40, std::span(packet).first(5));
    counts.Note(0x15, packet);
    check(counts.releaseMem == 0 && counts.copyData == 0, "unrelated or short packets counted as timestamp packets");
    packet[2] = 2u << 29u;
    counts.Note(0x49, packet);
    packet[2] = (3u << 29u) | (3u << 24u);
    counts.Note(0x49, packet);
    packet[4] = 1;
    counts.Note(0x49, packet);
    packet[3] = 4;
    packet[4] = 0;
    counts.Note(0x49, packet);
    check(counts.releaseMem == 4 && counts.releaseClock == 3 && counts.releaseClockWrites == 2, "release timestamps did not distinguish immediate data and null destinations");
    packet[1] = 9u | (1u << 8u);
    counts.Note(0x40, packet);
    packet[1] |= 1u << 16u;
    counts.Note(0x40, packet);
    packet[1] |= 1u << 30u;
    counts.Note(0x40, packet);
    packet[1] = 5u | (1u << 8u);
    counts.Note(0x40, packet);
    check(counts.copyData == 4 && counts.copyClock == 2, "copy timestamps did not distinguish both widths, source extension and immediate data");
    counts = {};
    packet[1] = 0x80000514;
    counts.Note(0x49, packet);
    packet[1] = 0x0400062f;
    counts.Note(0x49, packet);
    packet[1] = 0x02000630;
    counts.Note(0x49, packet);
    packet[1] = 0x80000631;
    counts.Note(0x49, packet);
    packet[1] = 0x04000714;
    counts.Note(0x49, packet);
    packet[3] = 0;
    counts.Note(0x49, packet);
    check(counts.releaseEop == 1 && counts.releaseCs == 1 && counts.releasePs == 1 && counts.releaseOther == 2 && counts.firstOtherEvent == 0x631, "timestamp stages or first unsupported event were classified incorrectly");
    check(counts.releaseClock == 6 && counts.releaseClockWrites == counts.releaseEop + counts.releaseCs + counts.releasePs + counts.releaseOther, "timestamp stages included a null destination or missed a write");
    counts = {};
    check(counts.releaseMem == 0 && counts.releaseClock == 0 && counts.releaseClockWrites == 0 && counts.copyData == 0 && counts.copyClock == 0, "timestamp packet counts survived a window reset");
    check(counts.releaseEop == 0 && counts.releaseCs == 0 && counts.releasePs == 0 && counts.releaseOther == 0 && counts.firstOtherEvent == 0, "timestamp stage counts survived a window reset");
}

}

int main() {
    try {
        timestampPacketTests();
#ifdef _WIN32
        _putenv_s("APS5_PROFILE_DRAW", "1");
#else
        setenv("APS5_PROFILE_DRAW", "1", 1);
#endif
        using AgcDriver::DriverDetail::HitchLines;
        alignas(64) static volatile std::uint32_t gate = 0, done = 0;
        const auto gateAddress = reinterpret_cast<std::uintptr_t>(&gate), doneAddress = reinterpret_cast<std::uintptr_t>(&done);
        // WAIT_REG_MEM until the gate is 1, then WRITE_DATA of 1 to `done`.
        std::vector<std::uint32_t> words{0xc0053c00, 0x13, static_cast<std::uint32_t>(gateAddress), static_cast<std::uint32_t>(static_cast<std::uint64_t>(gateAddress) >> 32u), 1, 0xffffffffu, 0x19,
                                         0xc0033700, 0x00100200, static_cast<std::uint32_t>(doneAddress), static_cast<std::uint32_t>(static_cast<std::uint64_t>(doneAddress) >> 32u), 1};
        const auto start = std::chrono::steady_clock::now();
        check(HitchLines(start, 0).empty(), "the first present reported an interval");
        Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
        check(sceAgcDriverSubmitDcb(&packet) == 0, "the submission was not accepted");
        // The one wait the test needs: the worker sits in the wait packet for over 100 ms.
        std::this_thread::sleep_for(std::chrono::milliseconds(130));
        gate = 1;
        AgcDriverWaitIdle_nid_postfix();
        check(done == 1, "the queue did not pass its wait");
        const auto lines = HitchLines(std::chrono::steady_clock::now(), 2.5);
        check(lines.size() == 1, "an interval over 100 ms produced " + std::to_string(lines.size()) + " lines, not one");
        const auto& line = lines.front();
        std::puts(line.c_str());
        check(line.rfind("[hitch] ", 0) == 0 && line.find("in-flight wait 2.5 ms") != std::string::npos, "the hitch line does not name its interval and in-flight wait");
        check(line.find("longest packet WAIT_REG_MEM") != std::string::npos, "the hitch line does not name the wait packet as the longest");
        const auto kinds = line.find("by packet kind: WAIT_REG_MEM");
        check(kinds != std::string::npos, "the hitch line does not put the wait packet first among the packet kinds");
        check(line.find("the stall is on the guest side") == std::string::npos, "a worker held in a packet was reported as idle");
        // An interval with the worker idle, and the 10 s line, with times given instead of waited for.
        const auto later = std::chrono::steady_clock::now();
        check(HitchLines(later + std::chrono::milliseconds(20), 0).empty(), "a short interval was listed");
        const auto idle = HitchLines(later + std::chrono::milliseconds(320), 0);
        check(idle.size() == 1 && idle.front().find("the stall is on the guest side or the presenter") != std::string::npos, "an interval with an idle worker was not reported as one");
        const auto summary = HitchLines(start + std::chrono::seconds(11), 0);
        check(!summary.empty(), "no 10 s line");
        std::puts(summary.back().c_str());
        check(summary.back().find("over 100 ms 3, over 250 ms 2") != std::string::npos || summary.back().find("over 100 ms 2, over 250 ms 1") != std::string::npos, "the 10 s line does not count the long intervals");
        std::puts("hitch profile tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
