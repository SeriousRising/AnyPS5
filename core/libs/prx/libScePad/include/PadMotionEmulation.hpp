#ifndef CORE_LIBS_PRX_LIBSCEPAD_PADMOTIONEMULATION_HPP
#define CORE_LIBS_PRX_LIBSCEPAD_PADMOTIONEMULATION_HPP

#include "prx/libScePad/include/PadState.hpp"
#include <algorithm>
#include <cmath>

namespace Pad {

// APS5_PAD_MOTION_EMULATION: the motion of a controller without sensors, made from its right stick
// and stick clicks, in SDL's sensor frame and units (x to the right, y up out of the face of a
// controller lying flat, z toward the player; m/s^2 and rad/s).
struct MotionEmulation {
    static constexpr std::uint32_t Modifier = 0x00000004u;
    static constexpr std::uint32_t Lift = 0x00000002u;
    static constexpr float Gravity = 9.80665f;
    static constexpr float MaxTilt = 1.0471976f;
    static constexpr float MaxRate = 3.1415927f;
    static constexpr float LiftSeconds = 0.15f;

    float pitch = 0.0f;
    float roll = 0.0f;
    float liftElapsed = 2.0f * LiftSeconds;
    bool liftHeld = false;

    // Advances the emulated controller by `dt` seconds from the inputs in `state` and writes its
    // motion there. While the modifier is held the right stick and both stick clicks drive the
    // motion and are hidden from the title.
    void Step(PadInputState& state, float dt) {
        const bool held = (state.buttons & Modifier) != 0;
        float targetPitch = 0.0f;
        float targetRoll = 0.0f;
        bool lift = false;
        if (held) {
            const auto deflection = [](std::uint8_t value) { return std::clamp((static_cast<float>(value) - 128.0f) / 127.0f, -1.0f, 1.0f); };
            targetPitch = deflection(state.sticks[3]) * MaxTilt;
            targetRoll = -deflection(state.sticks[2]) * MaxTilt;
            lift = (state.buttons & Lift) != 0;
            state.buttons &= ~(Modifier | Lift);
            state.sticks[2] = 128;
            state.sticks[3] = 128;
        }
        const auto turn = [dt](float& angle, float target) {
            if (dt <= 0.0f) return 0.0f;
            const float step = std::clamp(target - angle, -MaxRate * dt, MaxRate * dt);
            angle += step;
            return step / dt;
        };
        const float pitchRate = turn(pitch, targetPitch);
        const float rollRate = turn(roll, targetRoll);
        if (lift && !liftHeld) liftElapsed = 0.0f;
        else if (liftElapsed < 2.0f * LiftSeconds) liftElapsed += dt;
        liftHeld = lift;
        const float pulse = liftElapsed < LiftSeconds ? Gravity : liftElapsed < 2.0f * LiftSeconds ? -Gravity : 0.0f;
        state.hasMotion = true;
        state.accel = {Gravity * std::cos(pitch) * std::sin(roll), Gravity * std::cos(pitch) * std::cos(roll) + pulse, -Gravity * std::sin(pitch)};
        state.gyro = {pitchRate, 0.0f, rollRate};
    }
};

}

#endif
