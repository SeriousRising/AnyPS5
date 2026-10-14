#include "VulkanTestDevice.hpp"
#include "SceShaders.hpp"
#include "ShaderDiskCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void SetEnvironment(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

std::uint64_t Compiles() {
    const auto totals = ShaderRecompiler::ShaderDiskCache::Totals();
    return totals.hits + totals.misses + totals.loadFailures - totals.specializedLookups;
}

void DrawsReuseBothEarlyTestVariants() {
    constexpr std::uint32_t extent = 16;
    constexpr std::uint32_t pitch = 64;
    alignas(256) std::array<std::uint32_t, pitch * extent> pixels{};
    alignas(256) std::array<std::uint16_t, extent * extent> depth{};
    alignas(256) const std::array<std::array<float, 4>, 3> positions{{{-1, -1, 0.5f, 1}, {3, -1, 0.5f, 1}, {-1, 3, 0.5f, 1}}};
    alignas(256) const std::array<std::uint32_t, 6> vertexCode{0xe0382000u, 0x80020005u, 0xbf8c3f70u, 0xf80008cfu, 0x03020100u, 0xbf810000u};
    alignas(256) const std::array<std::uint32_t, 11> pixelCode{0x7e0002ffu, 0, 0x7e0202ffu, 0x3e800000u, 0x7e0402ffu, 0x3f000000u, 0x7e0602ffu, 0x3f800000u, 0xf800100fu, 0x03020100u, 0xbf810000u};
    struct Header {
        Shader shader{};
        ShaderUserData users{};
        std::array<ShaderRegister, 3> registers{};
        std::array<ShaderRegister, 5> context{};
    };
    std::array<Header, 2> headers{};
    for (std::size_t index = 0; index < headers.size(); ++index) {
        auto& header = headers[index];
        const auto* code = index == 0 ? vertexCode.data() : pixelCode.data();
        const auto address = reinterpret_cast<std::uintptr_t>(code);
        const auto program = index == 0 ? 0xc8u : 0x8u;
        header.shader.file_header = 0x34333231u;
        header.shader.version = 0x18;
        header.shader.type = index == 0 ? 2u : 1u;
        header.shader.header_size = sizeof(header);
        header.shader.shader_size = index == 0 ? sizeof(vertexCode) : sizeof(pixelCode);
        header.shader.code = code;
        header.shader.user_data = &header.users;
        header.shader.sh_registers = header.registers.data();
        header.shader.num_sh_registers = header.registers.size();
        header.registers = {{{program, static_cast<std::uint32_t>(address >> 8u)}, {program + 1u, static_cast<std::uint32_t>(address >> 40u)}, {index == 0 ? 0x8bu : 0xbu, index == 0 ? 4u << 1u : 0u}}};
        header.context = index == 0 ? std::array<ShaderRegister, 5>{{{0x2d5, 0x00402000}, {}, {}, {}, {}}} : std::array<ShaderRegister, 5>{{{0x1b6, 0x8000}, {0x1b3, 0}, {0x1b4, 0}, {0x203, 0x50}, {0x1c5, 9}}};
        header.shader.cx_registers = header.context.data();
        header.shader.num_cx_registers = index == 0 ? 1u : 5u;
        AgcDriverRegisterShader_nid_postfix(&header.shader);
    }
    AgcDriverResolveShaderAbi_nid_postfix(&headers[1].shader, {}, {});
    AgcDriver::Registers context{
        {0x2d5, 0x00402000}, {0x1b6, 0x8000}, {0x1b3, 0}, {0x1b4, 0}, {0x207, 0}, {0x203, 0x50},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (15u << 14u) | 15u}, {0x3b8, 0x9000000}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x100010}, {0x81, 0x80000000}, {0x82, 0x100010},
        {0x90, 0x80000000}, {0x91, 0x100010}, {0x94, 0x80000000}, {0x95, 0x100010},
        {0x000, 0}, {0x002, 0}, {0x010, 0x80000181}, {0x011, 0x20000180}, {0x007, 0x000f000f},
        {0x00a, 0}, {0x00b, std::bit_cast<std::uint32_t>(1.0f)}
    };
    const auto colorAddress = reinterpret_cast<std::uintptr_t>(pixels.data());
    context[0x318] = static_cast<std::uint32_t>(colorAddress >> 8u);
    context[0x390] = static_cast<std::uint32_t>(colorAddress >> 40u);
    const auto depthAddress = reinterpret_cast<std::uintptr_t>(depth.data());
    context[0x012] = context[0x014] = static_cast<std::uint32_t>(depthAddress >> 8u);
    context[0x01a] = context[0x01c] = static_cast<std::uint32_t>(depthAddress >> 40u);
    for (const auto offset : {0x10fu, 0x110u, 0x112u}) context[offset] = std::bit_cast<std::uint32_t>(extent / 2.0f);
    context[0x111] = std::bit_cast<std::uint32_t>(-static_cast<float>(extent) / 2.0f);
    context[0x113] = context[0xb5] = std::bit_cast<std::uint32_t>(1.0f);
    context[0x114] = context[0xb4] = 0;
    const auto positionAddress = reinterpret_cast<std::uintptr_t>(positions.data());
    const std::array<std::uint32_t, 4> users{static_cast<std::uint32_t>(positionAddress), static_cast<std::uint32_t>(positionAddress >> 32u) | (16u << 16u), 3, 0x01016facu};
    const auto draw = [&](std::uint32_t depthControl) {
        context[0x200] = depthControl;
        std::vector<std::uint32_t> commands;
        for (const auto& [offset, value] : context) commands.insert(commands.end(), {0xc0016900u, offset, value});
        for (const auto& header : headers) for (const auto& reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
        for (std::uint32_t index = 0; index < users.size(); ++index) commands.insert(commands.end(), {0xc0017600u, 0x8cu + index, users[index]});
        commands.insert(commands.end(), {0xc0017900u, 0x242, 4, 0xc0002f00u, 1, 0xc0012d00u, 3, 2});
        pixels.fill(0x12345678u);
        const auto before = Compiles();
        const Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
        Require(sceAgcDriverSubmitDcb(&packet) == 0, "killing pixel test submission failed");
        AgcDriverWaitIdle_nid_postfix();
        for (std::uint32_t y = 0; y < extent; ++y) {
            for (std::uint32_t x = 0; x < extent; ++x) Require(pixels[y * pitch + x] == 0xff804000u, "the killing pixel program did not write its color");
        }
        return Compiles() - before;
    };
    const auto writing = draw(0x76);
    Require(writing <= 1, "a draw that writes depth compiled the pre-warmed killing pixel program again (" + std::to_string(writing) + " compiles)");
    const auto testing = draw(0x72);
    Require(testing == 0, "a draw that only tests depth compiled the killing pixel program instead of reusing the pre-warmed one (" + std::to_string(testing) + " compiles)");
    AgcDriverShutdown_nid_postfix();
}

}

int main() {
    const auto cache = std::filesystem::temp_directory_path() / "anyps5-prepared-killing-pixel";
    try {
        std::filesystem::remove_all(cache);
        SetEnvironment("ANYPS5_NO_SHADER_CACHE", "0");
        SetEnvironment("ANYPS5_SHADER_CACHE_DIR", cache.string());
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        device.reset();
        Require(ShaderRecompiler::ShaderDiskCache::Enabled(), "the shader disk cache that counts compiles is disabled");
        DrawsReuseBothEarlyTestVariants();
        ShaderRecompiler::ShaderDiskCache::Flush();
        std::filesystem::remove_all(cache);
        std::cout << "prepared killing pixel tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { AgcDriverShutdown_nid_postfix(); }
        catch (const std::exception& shutdownError) { std::cerr << shutdownError.what() << '\n'; }
        std::error_code ignored;
        std::filesystem::remove_all(cache, ignored);
        return 1;
    }
}
