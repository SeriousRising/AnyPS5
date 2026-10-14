#include "prx/libSceVideoOut/include/PresentProfile.hpp"
#include <cstdio>
#include <stdexcept>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void thresholds() {
    using Clock = PresentProfile::Clock;
    PresentProfile profile;
    auto now = Clock::time_point{};
    check(!profile.Record(now, true, false), "first presentation produced an interval");
    const auto sixty = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<int, std::ratio<1, 60>>(1));
    const auto thirty = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<int, std::ratio<1, 30>>(1));
    const auto fifty = std::chrono::duration_cast<Clock::duration>(std::chrono::milliseconds(50));
    const auto hundred = std::chrono::duration_cast<Clock::duration>(std::chrono::milliseconds(100));
    const Clock::duration tick{1};
    const Clock::duration intervals[]{sixty, sixty + tick, thirty, thirty + tick, fifty, fifty + tick, hundred, hundred + tick};
    for (auto interval : intervals) {
        now += interval;
        check(!profile.Record(now, true, false), "short window emitted a summary");
    }
    const auto summary = profile.Finish();
    check(summary.has_value(), "nonempty window lost its summary");
    check(summary->intervals == 8 && summary->elapsed == now.time_since_epoch(), "interval count or elapsed duration changed");
    check(summary->maximum == hundred + tick, "maximum interval changed");
    check(summary->over60Hz == 7 && summary->over30Hz == 5, "fractional-second budget boundary is wrong");
    check(summary->over50Ms == 3 && summary->over100Ms == 1, "exact millisecond boundary is wrong");
    check(!profile.Finish(), "empty finish duplicated a summary");
}

void windowsAndFiltering() {
    using namespace std::chrono;
    const PresentProfile::Clock::time_point start{};
    PresentProfile profile;
    check(!profile.Record(start, false, false), "canceled flip produced a summary");
    check(!profile.Record(start + seconds(1), true, true), "overlay flip produced a summary");
    check(!profile.Finish(), "filtered flips initialized a window");
    check(!profile.Record(start, true, false), "first sample was counted");
    check(!profile.Record(start + seconds(4), true, false), "window emitted before ten seconds");
    check(!profile.Record(start + seconds(5), true, true), "overlay emitted a summary");
    check(!profile.Record(start + seconds(7), false, false), "cancellation emitted a summary");
    const auto first = profile.Record(start + seconds(10), true, false);
    check(first && first->intervals == 2 && first->elapsed == seconds(10), "ten-second boundary lost or added an interval");
    check(first->maximum == seconds(6), "filtered flips changed the previous accepted presentation");
    check(first->over30Hz == 2 && first->over100Ms == 2, "filtered flips changed threshold counts");
    check(!profile.Record(start + seconds(10) + milliseconds(30), true, false), "new window emitted immediately");
    const auto second = profile.Record(start + seconds(20), true, false);
    check(second && second->intervals == 2 && second->elapsed == seconds(10), "report boundary dropped the next interval");
    check(second->maximum == milliseconds(9970) && second->over30Hz == 1, "window counters were not reset");
    check(!profile.Record(start + seconds(20) + milliseconds(20), true, false), "partial window emitted early");
    const auto partial = profile.Finish();
    check(partial && partial->intervals == 1 && partial->elapsed == milliseconds(20), "partial summary lost its interval");
    check(!profile.Record(start + seconds(200), true, false), "finished output retained a previous timestamp");
    check(!profile.Finish(), "first sample after reset produced a summary");
    PresentProfile reopened;
    check(!reopened.Record(start + seconds(300), true, false) && !reopened.Finish(), "new output inherited old samples");
    check(!profile.Record(start, true, false) && !profile.Record(start, true, false), "zero-duration samples emitted early");
    const auto zero = profile.Finish();
    check(zero && zero->intervals == 1 && zero->elapsed == PresentProfile::Clock::duration{} && zero->maximum == PresentProfile::Clock::duration{}, "zero-duration interval was not preserved safely");
}

}

int main() {
    try {
        thresholds();
        windowsAndFiltering();
        std::puts("present profile tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
