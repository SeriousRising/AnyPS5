#include "VulkanTestDevice.hpp"
#include "SceShaders.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/Shutdown.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <initializer_list>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t TableBytes = 16;
constexpr std::uint32_t Tables = 2048;
constexpr std::uint32_t ValuesOffset = Tables * TableBytes;
constexpr std::uint32_t Values = 4096;
constexpr std::uint32_t ResultsOffset = ValuesOffset + Values * 4u;
constexpr std::uint32_t Results = 4096;
constexpr std::uint32_t ConditionsOffset = BlockBytes;
constexpr std::uint32_t DoneOffset = BlockBytes + 0x8000u;

class GuestBlock {
public:
    explicit GuestBlock(std::size_t bytes) : bytes(bytes) {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, bytes));
#endif
        Require(block != nullptr, "draw ahead: cannot allocate a guest block");
        std::memset(block, 0, bytes);
        GuestAllocations::Mutation().Add(block, bytes, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint64_t Address(std::uint32_t offset = 0) const { return reinterpret_cast<std::uintptr_t>(block) + offset; }
    std::size_t Bytes() const { return bytes; }
    std::uint32_t Word(std::uint32_t offset) const {
        std::uint32_t value = 0;
        std::memcpy(&value, block + offset, 4u);
        return value;
    }
    void SetWord(std::uint32_t offset, std::uint32_t value) { std::memcpy(block + offset, &value, 4u); }
    void SetAddress(std::uint32_t offset, std::uint64_t value) { std::memcpy(block + offset, &value, 8u); }

private:
    std::uint8_t* block = nullptr;
    std::size_t bytes = 0;
};

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{0xe0382000, 0x80020005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000};
alignas(256) constexpr std::array<std::uint32_t, 9> ReadCode{0xf4080304, 0xfa000000, 0x7e1e0280, 0xdc308000, 0x0a0c000f, 0xbf8c3f70, 0xf800180f, 0x0a0a0a0a, 0xbf810000};
alignas(256) constexpr std::array<std::uint32_t, 13> CopyCode{0xf4080304, 0xfa000000, 0x7e1e0280, 0xdc308000, 0x0a0c000f, 0xbf8c3f70, 0x7e02020e, 0x7e04020f, 0xdc708000, 0x007d0a01, 0xf800180f, 0x0a0a0a0a, 0xbf810000};
alignas(256) std::array<std::array<float, 4>, 3> Triangle{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};

struct Program {
    struct Header {
        Shader shader{};
        ShaderUserData users{};
        std::array<ShaderRegister, 4> registers{};
    } header;

    Program(std::uint8_t type, std::span<const std::uint32_t> code, std::uint32_t users) {
        const auto address = reinterpret_cast<std::uintptr_t>(code.data());
        const std::uint32_t program = type == 1 ? 0x008u : 0x0c8u;
        const std::uint32_t resources = type == 1 ? 0x00bu : 0x08bu;
        header.shader.file_header = 0x34333231u;
        header.shader.version = 0x18;
        header.shader.header_size = sizeof(header);
        header.shader.shader_size = static_cast<std::uint32_t>(code.size_bytes());
        header.shader.code = code.data();
        header.shader.type = type;
        header.shader.user_data = &header.users;
        header.shader.sh_registers = header.registers.data();
        header.shader.num_sh_registers = static_cast<std::uint8_t>(header.registers.size());
        header.registers = {{{program, static_cast<std::uint32_t>(address >> 8u)}, {program + 1u, static_cast<std::uint32_t>(address >> 40u)}, {resources - 1u, 0}, {resources, users << 1u}}};
        AgcDriverRegisterShader_nid_postfix(&header.shader);
    }

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
};

std::vector<ShaderRegister> ContextRegisters(std::uint64_t target) {
    return {
        {0x2d5, 0x2000}, {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800}, {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa}, {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240}, {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010}, {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4},
        {0x31c, 0x28028}, {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 63u}, {0x3b8, 0x906c000}, {0x1e0, 0}, {0xc, 0}, {0xd, 0x400040},
        {0x81, 0x80000000}, {0x82, 0x400040}, {0x90, 0x80000000}, {0x91, 0x400040}, {0x94, 0x80000000}, {0x95, 0x400040},
        {0x318, static_cast<std::uint32_t>(target >> 8u)}, {0x390, static_cast<std::uint32_t>(target >> 40u)},
        {0x10f, std::bit_cast<std::uint32_t>(32.0f)}, {0x110, std::bit_cast<std::uint32_t>(32.0f)}, {0x111, std::bit_cast<std::uint32_t>(-32.0f)}, {0x112, std::bit_cast<std::uint32_t>(32.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0}, {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)}
    };
}

struct Commands {
    std::vector<std::uint32_t> words;

    void Add(std::initializer_list<std::uint32_t> packet) { words.insert(words.end(), packet); }
    void Context(std::uint32_t offset, std::uint32_t value) { Add({0xc0016900u, offset, value}); }
    void ShaderRegister(std::uint32_t offset, std::uint32_t value) { Add({0xc0017600u, offset, value}); }
    void Program(const ::Program& program) {
        for (const auto reg : program.header.registers) ShaderRegister(reg.offset, reg.value);
    }
    void Table(std::uint64_t address) {
        ShaderRegister(0x00cu + 8u, static_cast<std::uint32_t>(address));
        ShaderRegister(0x00cu + 9u, static_cast<std::uint32_t>(address >> 32u));
    }
    void Write(std::uint64_t address, std::uint32_t value) { Add({0xc0033700u, 0x00100200u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), value}); }
    void Target(std::uint64_t address) {
        Context(0x318, static_cast<std::uint32_t>(address >> 8u));
        Context(0x390, static_cast<std::uint32_t>(address >> 40u));
    }
    void Condition(std::uint64_t address, std::uint32_t words) { Add({0xc0032200u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), 0u, words}); }
    void Draw() { Add({0xc0012d00u, static_cast<std::uint32_t>(Triangle.size()), 2u}); }
    void Submit() {
        Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
        Require(sceAgcDriverSubmitDcb(&packet) == 0, "draw ahead: a submission was not accepted");
        words.clear();
    }
};

struct Scene {
    GuestBlock memory{BlockBytes * 2u};
    GuestBlock target{BlockBytes};
    GuestBlock second{BlockBytes};
    Program vertex{2, VertexCode, 4};
    Program read{1, ReadCode, 10};
    Program copy{1, CopyCode, 10};
    std::vector<ShaderRegister> context = ContextRegisters(target.Address());
    std::uint32_t serial = 0;

    Scene() {
        const std::array<ShaderRegister, 1> primitive{{{0x242, 4}}};
        for (const auto* program : {&vertex, &read, &copy}) AgcDriverResolveShaderAbi_nid_postfix(&program->header.shader, context, primitive);
    }

    std::uint64_t Table(std::uint32_t index) const { return memory.Address(index * TableBytes); }
    std::uint64_t Value(std::uint32_t index) const { return memory.Address(ValuesOffset + index * 4u); }
    std::uint64_t Result(std::uint32_t index) const { return memory.Address(ResultsOffset + index * 4u); }
    std::uint32_t ResultWord(std::uint32_t index) const { return memory.Word(ResultsOffset + index * 4u); }
    void SetTable(std::uint32_t index, std::uint64_t source, std::uint64_t destination) {
        memory.SetAddress(index * TableBytes, source);
        memory.SetAddress(index * TableBytes + 8u, destination);
    }

    void Finish(Commands& commands) {
        commands.Write(memory.Address(DoneOffset), ++serial);
        commands.Submit();
        AgcDriverWaitIdle_nid_postfix();
        const auto start = std::chrono::steady_clock::now();
        while (memory.Word(DoneOffset) != serial) {
            Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(20), "draw ahead: the label behind a submission never landed");
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        AgcDriverWaitIdle_nid_postfix();
    }

    bool Holds(std::uint64_t address, std::uint32_t texel) {
        AgcDriver::QueueState queue;
        for (const auto reg : ContextRegisters(address)) queue.context[reg.offset] = reg.value;
        queue.userConfig[0x242] = 4;
        const auto color = AgcDriver::Graphics::DecodeState(queue).color;
        AgcDriver::Graphics::StorageTexture::FlushPending(color.address, color.bytes, nullptr, "test");
        Commands commands;
        Finish(commands);
        std::vector<std::uint32_t> pixels(static_cast<std::size_t>(color.extent.width) * color.extent.height);
        AgcDriver::Graphics::ReadColorTarget(color, std::as_writable_bytes(std::span(pixels)));
        return std::all_of(pixels.begin(), pixels.end(), [texel](std::uint32_t pixel) { return pixel == texel; });
    }

    void State(Commands& commands, const Program& fragment) const {
        for (const auto reg : context) commands.Context(reg.offset, reg.value);
        commands.Add({0xc0017900u, 0x242, 4});
        commands.Program(vertex);
        const auto triangle = reinterpret_cast<std::uintptr_t>(Triangle.data());
        commands.ShaderRegister(0x08c, static_cast<std::uint32_t>(triangle));
        commands.ShaderRegister(0x08d, (static_cast<std::uint32_t>(triangle >> 32u) & 0xffffu) | (16u << 16u));
        commands.ShaderRegister(0x08e, static_cast<std::uint32_t>(Triangle.size()));
        commands.ShaderRegister(0x08f, 0x01016facu);
        commands.Program(fragment);
        for (std::uint32_t user = 0; user < 10; ++user) commands.ShaderRegister(0x00cu + user, 0);
    }
};

void Chain(Scene& scene) {
    for (std::uint32_t i = 0; i < 8; ++i) scene.memory.SetWord(ValuesOffset + i * 4u, 0x41000000u + i);
    scene.SetTable(0, scene.Value(3), scene.Result(0));
    for (std::uint32_t i = 1; i < 8; ++i) scene.SetTable(i, scene.Result(i - 1), scene.Result(i));
    Commands commands;
    scene.State(commands, scene.copy);
    for (std::uint32_t i = 0; i < 8; ++i) {
        commands.Table(scene.Table(i));
        commands.Draw();
    }
    scene.Finish(commands);
    for (std::uint32_t i = 0; i < 8; ++i) Require(scene.ResultWord(i) == 0x41000003u, "draw ahead: a draw did not read the word the draw before it stored");
}

void Registers(Scene& scene) {
    scene.memory.SetWord(ValuesOffset, 0x3f800000u);
    scene.memory.SetWord(ValuesOffset + 4u, 0u);
    scene.memory.SetWord(ValuesOffset + 8u, 0x40400000u);
    scene.SetTable(0, scene.Value(0), scene.Result(16));
    scene.SetTable(1, scene.Value(0), scene.Result(17));
    scene.SetTable(2, scene.Value(1), scene.Result(18));
    scene.SetTable(3, scene.Value(2), scene.Result(19));
    Commands commands;
    scene.State(commands, scene.copy);
    commands.Table(scene.Table(0));
    commands.Draw();
    commands.Target(scene.second.Address());
    commands.Table(scene.Table(1));
    commands.Draw();
    commands.Target(scene.target.Address());
    commands.Program(scene.read);
    commands.Table(scene.Table(2));
    commands.Draw();
    commands.Program(scene.copy);
    commands.Target(scene.second.Address());
    commands.Context(0x82, 0x200020);
    commands.Table(scene.Table(3));
    commands.Draw();
    commands.Context(0x82, 0x400040);
    commands.Target(scene.target.Address());
    scene.Finish(commands);
    Require(scene.ResultWord(16) == 0x3f800000u && scene.ResultWord(17) == 0x3f800000u && scene.ResultWord(19) == 0x40400000u, "draw ahead: draws between register changes did not each copy the word of their own table");
    Require(scene.ResultWord(18) == 0u, "draw ahead: a draw of the reading program stored a word");
    Require(scene.Holds(scene.target.Address(), 0u) && scene.Holds(scene.second.Address(), 0xffffffffu), "draw ahead: draws between target changes did not each draw to their own target");
}

void Labels(Scene& scene) {
    for (std::uint32_t i = 0; i < 4; ++i) scene.memory.SetWord(ValuesOffset + i * 4u, 0x42000000u + i);
    scene.SetTable(0, scene.Value(0), scene.Result(32));
    scene.SetTable(1, scene.Value(0), scene.Result(33));
    scene.SetTable(2, scene.Value(0), scene.Result(34));
    Commands commands;
    scene.State(commands, scene.copy);
    commands.Table(scene.Table(0));
    commands.Draw();
    commands.Write(scene.Table(1), static_cast<std::uint32_t>(scene.Value(1)));
    commands.Table(scene.Table(1));
    commands.Draw();
    commands.Write(scene.Table(1), static_cast<std::uint32_t>(scene.Value(2)));
    commands.Write(scene.Table(1) + 8u, static_cast<std::uint32_t>(scene.Result(35)));
    commands.Draw();
    commands.Table(scene.Table(2));
    commands.Draw();
    commands.Write(scene.Table(2), static_cast<std::uint32_t>(scene.Value(3)));
    commands.Draw();
    scene.Finish(commands);
    Require(scene.ResultWord(32) == 0x42000000u, "draw ahead: a draw before a label store did not read its table");
    Require(scene.ResultWord(33) == 0x42000001u && scene.ResultWord(35) == 0x42000002u, "draw ahead: a draw did not read the table words a label stored before it");
    Require(scene.ResultWord(34) == 0x42000003u, "draw ahead: the second draw of a table did not read the word a label stored between the two");
}

void Conditions(Scene& scene) {
    for (std::uint32_t i = 0; i < 4; ++i) scene.memory.SetWord(ValuesOffset + i * 4u, 0x43000000u + i);
    for (std::uint32_t i = 0; i < 4; ++i) scene.SetTable(i, scene.Value(i), scene.Result(48 + i));
    scene.memory.SetWord(ConditionsOffset, 0);
    scene.memory.SetWord(ConditionsOffset + 4u, 1);
    Commands commands;
    scene.State(commands, scene.copy);
    commands.Table(scene.Table(0));
    commands.Draw();
    commands.Condition(scene.memory.Address(ConditionsOffset), 9);
    commands.Table(scene.Table(1));
    commands.Draw();
    commands.Draw();
    commands.Condition(scene.memory.Address(ConditionsOffset + 4u), 9);
    commands.Table(scene.Table(2));
    commands.Draw();
    commands.Table(scene.Table(3));
    commands.Draw();
    scene.Finish(commands);
    Require(scene.ResultWord(48) == 0x43000000u && scene.ResultWord(49) == 0u, "draw ahead: a skipped conditional block changed the table of the draw after it");
    Require(scene.ResultWord(50) == 0x43000002u && scene.ResultWord(51) == 0x43000003u, "draw ahead: an executed conditional block did not draw");
}

void Stress(Scene& scene, std::uint32_t rounds) {
    constexpr std::uint32_t pool = 24;
    std::mt19937 random(0x5eed);
    const auto pick = [&](std::uint32_t count) { return static_cast<std::uint32_t>(random() % count); };
    for (std::uint32_t i = 0; i < Values; ++i) scene.memory.SetWord(ValuesOffset + i * 4u, 0x44000000u + i);
    std::vector<std::uint32_t> expected(Results);
    for (std::uint32_t i = 0; i < Results; ++i) expected[i] = scene.ResultWord(i);
    std::uint32_t nextResult = 64;
    for (std::uint32_t round = 0; round < rounds; ++round) {
        std::vector<std::uint64_t> source(pool), destination(pool);
        const auto word = [&](std::uint64_t address) {
            const auto offset = static_cast<std::uint32_t>(address - scene.memory.Address());
            return offset >= ResultsOffset ? expected[(offset - ResultsOffset) / 4u] : scene.memory.Word(offset);
        };
        const auto fresh = [&] {
            nextResult = nextResult + 1 == Results ? 64 : nextResult + 1;
            return nextResult;
        };
        for (std::uint32_t table = 0; table < pool; ++table) {
            source[table] = pick(4) == 0 ? scene.Result(64 + pick(Results - 64)) : scene.Value(pick(Values));
            destination[table] = scene.Result(fresh());
            scene.SetTable(table, source[table], destination[table]);
        }
        Commands commands;
        scene.State(commands, scene.copy);
        bool copies = true, first = true;
        std::uint32_t table = 0;
        commands.Table(scene.Table(table));
        const auto draw = [&] {
            commands.Draw();
            if (copies) expected[static_cast<std::uint32_t>(destination[table] - scene.Result(0)) / 4u] = word(source[table]);
        };
        for (std::uint32_t step = 0; step < 160; ++step) {
            switch (pick(12)) {
                case 0:
                    source[table] = scene.Value(pick(Values));
                    commands.Write(scene.Table(table), static_cast<std::uint32_t>(source[table]));
                    break;
                case 1:
                    copies = !copies;
                    commands.Program(copies ? scene.copy : scene.read);
                    break;
                case 2:
                    first = !first;
                    commands.Target(first ? scene.target.Address() : scene.second.Address());
                    commands.Context(0x82, first ? 0x400040u : 0x200020u);
                    break;
                case 3: {
                    const bool taken = pick(2) != 0;
                    const auto other = pick(pool);
                    scene.memory.SetWord(ConditionsOffset + step * 4u, taken ? 1u : 0u);
                    commands.Condition(scene.memory.Address(ConditionsOffset + step * 4u), 9);
                    commands.Table(scene.Table(other));
                    if (taken) table = other;
                    if (taken) draw();
                    else commands.Draw();
                    break;
                }
                default:
                    table = pick(pool);
                    commands.Table(scene.Table(table));
                    draw();
                    break;
            }
        }
        commands.Target(scene.target.Address());
        commands.Context(0x82, 0x400040);
        scene.Finish(commands);
        for (std::uint32_t i = 0; i < Results; ++i) {
            if (scene.ResultWord(i) != expected[i]) throw std::runtime_error("draw ahead: round " + std::to_string(round) + " left result " + std::to_string(i) + " unlike the draws executed in order");
        }
    }
}

void Benchmark(Scene& scene, std::uint32_t batches) {
    constexpr std::uint32_t frames = 3, draws = 650;
    for (std::uint32_t i = 0; i < Values; ++i) scene.memory.SetWord(ValuesOffset + i * 4u, i % 2 == 0 ? 0x3f800000u : 0u);
    std::uint64_t issued = 0;
    const auto batch = [&](std::uint32_t serial) {
        for (std::uint32_t i = 0; i < frames * draws; ++i) scene.SetTable(i, scene.Value((serial * 1031u + i * 7u) % Values), scene.Result(0));
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            Commands commands;
            scene.State(commands, scene.read);
            for (std::uint32_t i = 0; i < draws; ++i) {
                commands.Table(scene.Table(frame * draws + i));
                commands.Draw();
            }
            if (frame + 1 == frames) scene.Finish(commands);
            else commands.Submit();
        }
        issued += frames * draws;
    };
    for (std::uint32_t serial = 0; serial < 4; ++serial) batch(serial);
    issued = 0;
    const auto before = AgcDriver::DrawAheadCounts();
    const auto cpu = std::clock();
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t serial = 4; serial < 4 + batches; ++serial) batch(serial);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double busy = static_cast<double>(std::clock() - cpu) / CLOCKS_PER_SEC;
    const auto after = AgcDriver::DrawAheadCounts();
    std::printf("draw ahead %s: %llu draws in %.2f s, %.1f us a draw, %.2f cores busy, %llu prepared ahead, %llu hit\n", std::getenv("APS5_NO_DRAW_AHEAD") != nullptr ? "off" : "on", static_cast<unsigned long long>(issued), seconds, seconds * 1e6 / static_cast<double>(issued), busy / seconds, static_cast<unsigned long long>(after.prepared - before.prepared), static_cast<unsigned long long>(after.hits - before.hits));
}

}

int main(int argc, char** argv) {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Require(argc == 1 || (argc == 3 && std::string(argv[1]) == "bench"), "draw ahead: expected no arguments, or bench and a number of batches");
        {
            Scene scene;
            if (argc == 3) {
                Benchmark(scene, static_cast<std::uint32_t>(std::atoi(argv[2])));
            } else {
                Chain(scene);
                Registers(scene);
                Labels(scene);
                Conditions(scene);
                Stress(scene, 48);
                const auto counts = AgcDriver::DrawAheadCounts();
                const bool off = std::getenv("APS5_NO_DRAW_AHEAD") != nullptr;
                std::printf("draw ahead: %llu draws prepared ahead, %llu hit\n", static_cast<unsigned long long>(counts.prepared), static_cast<unsigned long long>(counts.hits));
                Require(off ? counts.prepared == 0 && counts.hits == 0 : counts.prepared != 0 && counts.hits != 0, off ? "draw ahead: draws were prepared ahead with the look-ahead switched off" : "draw ahead: no draw was prepared ahead and then hit");
            }
        }
        LibcRunShutdown_nid_postfix();
        std::puts("draw ahead tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        try {
            LibcRunShutdown_nid_postfix();
        } catch (const std::exception& shutdown) {
            std::fprintf(stderr, "shutdown: %s\n", shutdown.what());
        }
        return 1;
    }
}
