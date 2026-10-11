#include "prx/libScePad/include/PadMotionEmulation.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

using Pad::MotionEmulation;

constexpr float G = MotionEmulation::Gravity;

void Require(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error(what);
}

bool Near(float a, float b) {
    return std::fabs(a - b) < 1e-3f;
}

std::string Text(const PadInputState& state) {
    char text[160];
    std::snprintf(text, sizeof(text), " (accel %.3f %.3f %.3f, gyro %.3f %.3f %.3f)", state.accel[0], state.accel[1], state.accel[2], state.gyro[0], state.gyro[1], state.gyro[2]);
    return text;
}

PadInputState Input(std::uint32_t buttons, std::uint8_t rightX, std::uint8_t rightY) {
    PadInputState state;
    state.buttons = buttons;
    state.sticks = {128, 128, rightX, rightY};
    return state;
}

PadInputState Run(MotionEmulation& motion, std::uint32_t buttons, std::uint8_t rightX, std::uint8_t rightY, int steps, float dt) {
    PadInputState state;
    for (int step = 0; step < steps; ++step) {
        state = Input(buttons, rightX, rightY);
        motion.Step(state, dt);
    }
    return state;
}

}

int main() {
    try {
        constexpr std::uint32_t cross = 0x4000u;
        MotionEmulation motion;
        auto state = Input(cross | MotionEmulation::Lift, 255, 0);
        motion.Step(state, 0.016f);
        Require(state.hasMotion && state.accel == std::array<float, 3>{0.0f, G, 0.0f} && state.gyro == std::array<float, 3>{0.0f, 0.0f, 0.0f}, "at rest the controller does not report gravity on +y and no rotation" + Text(state));
        Require(state.buttons == (cross | MotionEmulation::Lift) && state.sticks == std::array<std::uint8_t, 4>{128, 128, 255, 0}, "inputs changed without the modifier");

        state = Run(motion, cross | MotionEmulation::Modifier, 128, 255, 1, 0.1f);
        Require(Near(state.gyro[0], MotionEmulation::MaxRate) && state.gyro[1] == 0.0f && state.gyro[2] == 0.0f, "a full stick pull does not pitch at the maximum rate" + Text(state));
        Require(Near(state.accel[1], G * std::cos(0.31415927f)) && Near(state.accel[2], -G * std::sin(0.31415927f)) && state.accel[0] == 0.0f, "gravity did not follow a pitch of 0.1 s at the maximum rate" + Text(state));
        Require(state.buttons == cross && state.sticks == std::array<std::uint8_t, 4>{128, 128, 128, 128}, "the modifier and the right stick reached the title");
        state = Run(motion, MotionEmulation::Modifier, 128, 255, 5, 0.1f);
        Require(state.gyro[0] == 0.0f && Near(state.accel[1], G * 0.5f) && Near(state.accel[2], -G * std::sin(MotionEmulation::MaxTilt)), "a held stick pull does not rest at a pitch of 60 degrees" + Text(state));
        state = Run(motion, 0u, 128, 128, 1, 0.1f);
        Require(Near(state.gyro[0], -MotionEmulation::MaxRate), "releasing the modifier does not turn the controller back" + Text(state));
        state = Run(motion, 0u, 128, 128, 5, 0.1f);
        Require(state.accel == std::array<float, 3>{0.0f, G, 0.0f} && state.gyro == std::array<float, 3>{0.0f, 0.0f, 0.0f}, "the controller did not return to rest" + Text(state));

        state = Run(motion, MotionEmulation::Modifier, 255, 128, 6, 0.1f);
        Require(Near(state.accel[0], -G * std::sin(MotionEmulation::MaxTilt)) && Near(state.accel[1], G * 0.5f) && Near(state.accel[2], 0.0f) && state.gyro[2] == 0.0f, "a held stick push to the right does not rest at a roll of 60 degrees to the right" + Text(state));
        state = Run(motion, MotionEmulation::Modifier, 128, 128, 1, 0.1f);
        Require(Near(state.gyro[2], MotionEmulation::MaxRate), "centering the stick does not roll the controller back" + Text(state));
        Run(motion, 0u, 128, 128, 5, 0.1f);

        const float expected[10] = {2.0f * G, 2.0f * G, 2.0f * G, 2.0f * G, 0.0f, 0.0f, 0.0f, 0.0f, G, G};
        for (int step = 0; step < 10; ++step) {
            state = Run(motion, cross | MotionEmulation::Modifier | MotionEmulation::Lift, 128, 128, 1, 0.04f);
            Require(Near(state.accel[1], expected[step]) && state.accel[0] == 0.0f && Near(state.accel[2], 0.0f) && state.gyro == std::array<float, 3>{0.0f, 0.0f, 0.0f}, "step " + std::to_string(step) + " of a lift is not " + std::to_string(expected[step]) + " on +y" + Text(state));
            Require(state.buttons == cross, "the lift control reached the title");
        }
        Run(motion, MotionEmulation::Modifier, 128, 128, 1, 0.05f);
        state = Run(motion, MotionEmulation::Modifier | MotionEmulation::Lift, 128, 128, 1, 0.05f);
        Require(Near(state.accel[1], 2.0f * G), "a second press of the lift control does not start a second pulse" + Text(state));
        std::puts("pad motion emulation tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
