#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparation.hpp"
#include "SceShaders.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <bit>
#include <vector>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

extern "C" int APS5_VABI sceAgcCreatePrimState(ShaderRegister*, ShaderRegister*, const Shader*, const Shader*, std::uint32_t);
extern "C" int APS5_VABI sceAgcCreateInterpolantMapping(ShaderRegister*, const Shader*, const Shader*);

namespace {

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename TAction>
void Reject(TAction action, const char* expected) {
    try { action(); }
    catch (const std::exception& error) {
        Require(std::string(error.what()).find(expected) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error(std::string("invalid graphics ABI was accepted; expected: ") + expected);
}

struct Fixture {
    struct Header {
        Shader shader{};
        ShaderUserData users{};
    } header;
    alignas(256) std::array<std::uint32_t, 65> code{};
    std::shared_ptr<AgcDriver::DriverDetail::ShaderSnapshot> snapshot;

    void Initialize(std::uint8_t type) {
        header.shader.type = type;
        header.shader.user_data = &header.users;
        code.fill(0xffffffffu);
        code.back() = 0xbf810000u;
        const auto address = reinterpret_cast<std::uintptr_t>(code.data());
        const auto headerAddress = reinterpret_cast<std::uintptr_t>(&header);
        snapshot = std::make_shared<AgcDriver::DriverDetail::ShaderSnapshot>();
        snapshot->type = type;
        snapshot->codeAddress = address;
        snapshot->headerAddress = headerAddress;
        snapshot->code.assign(code.begin(), code.end());
        snapshot->header.resize(sizeof(header));
        std::memcpy(snapshot->header.data(), &header, sizeof(header));
    }

    void Bind(AgcDriver::QueueState& queue, std::uint32_t program, std::uint32_t resources) const {
        const auto address = snapshot->codeAddress + 256u;
        queue.shader[program] = static_cast<std::uint32_t>(address >> 8u);
        queue.shader[program + 1u] = static_cast<std::uint32_t>(address >> 40u);
        queue.shader[resources] = 16u << 1u;
    }
};

void DrawPreparedRectangle(AgcDriver::VulkanDevice& device, std::uint32_t waveSize) {
    using namespace AgcDriver::DriverDetail;
    using namespace ShaderRecompiler;
    constexpr std::uint32_t extent = 16;
    alignas(256) std::array<std::uint32_t, extent * extent * 4> pixels{};
    constexpr std::array<std::array<float, 4>, 3> positions{{{-1, -1, 0, 1}, {-1, 1, 0, 1}, {1, -1, 0, 1}}};
    constexpr std::array<std::uint32_t, 6> vertexCode{0xe0382000u, 0x80000005u, 0xbf8c3f70u, 0xf80008cfu, 0x03020100u, 0xbf810000u};
    constexpr std::array<std::uint32_t, 4> values{0, 0x3e800000u, 0x3f000000u, 0x3f800000u};
    const std::array<std::uint32_t, 11> pixelCode{0x7e0002ffu, values[0], 0x7e0202ffu, values[1], 0x7e0402ffu, values[2], 0x7e0602ffu, values[3], 0xf800180fu, 0x03020100u, 0xbf810000u};
    const auto address = reinterpret_cast<std::uintptr_t>(positions.data());
    const std::array<std::uint32_t, 4> users{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) | (16u << 16u), 3, 0x01016facu};
    RecompileRequest vertexRequest{{ShaderStage::Vertex, 0x10000, vertexCode, 0, {}}, {waveSize, 0, users, {}, {}, ShaderVertexStageInfo{}, {}}, device.Target(), {0, 0, 0, 128}};
    const auto vertex = Recompile(vertexRequest);
    const auto pushOffset = static_cast<std::uint32_t>(vertex.pushConstants.size());
    AgcDriver::Registers context{{0x1b6, 0x8000}, {0x1b3, 0}, {0x1b4, 0}, {0x203, 0}, {0x1c5, 9}};
    auto pixelInfo = AgcDriver::Graphics::DecodePixelStageInfo(context, {});
    pixelInfo.wave32 = waveSize == 32;
    pixelInfo.targetExportMapping.fill(0xe4u);
    RecompileRequest pixelRequest{{ShaderStage::Fragment, 0x20000, pixelCode, 0, {}}, {waveSize, 0, {}, {}, pixelInfo, {}, {}}, device.Target(), {0, 0, pushOffset, 128 - pushOffset}};
    const auto pixel = Recompile(pixelRequest);
    ShaderSnapshot front{0x10000, 0, 2, {vertexCode.begin(), vertexCode.end()}, {}};
    auto fragment = std::make_shared<ShaderSnapshot>(ShaderSnapshot{0x20000, 0, 1, {pixelCode.begin(), pixelCode.end()}, {}});
    front.prepared->entries.push_back({0, PrepareShader(vertexRequest)});
    fragment->prepared->entries.push_back({0, PrepareShader(pixelRequest)});
    ResolvePreparedGraphics(front, fragment, 17, device.Target());
    const auto rectangle = PreparedRectangle(front, vertex, pixel, device.Target());
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0, waveSize, waveSize, {}, {}};
    state.color = {reinterpret_cast<std::uintptr_t>(pixels.data()), {extent, extent}, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(pixels), 0xe4u};
    state.color.elementBytes = 16;
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {extent, extent};
    state.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    state.rectList = true;
    state.viewport = {0, static_cast<float>(extent), static_cast<float>(extent), -static_cast<float>(extent), 0, 1};
    state.scissor = {{0, 0}, {extent, extent}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    const AgcDriver::Pm4::DrawParameters draw{0, 3, 0, 1, 0, false};
    const auto materializedRectangle = BuildRectListShaders(vertex, pixel, device.Target());
    for (const auto* auxiliary : {&rectangle, &materializedRectangle}) {
        const std::array<AgcDriver::Graphics::CompiledShader, 4> shaders{{{ShaderStage::Vertex, &vertex, 0}, {ShaderStage::TessellationControl, &auxiliary->control, 0}, {ShaderStage::TessellationEvaluation, &auxiliary->evaluation, 0}, {ShaderStage::Fragment, &pixel, pushOffset}}};
        pixels.fill(0xdeadbeefu);
        device.Draw(state, draw, shaders);
        device.WaitIdle();
        for (std::size_t index = 0; index < pixels.size(); ++index) Require(pixels[index] == values[index % 4], "prepared rectangle draw produced incorrect pixels");
    }
}

void DrawLazyRectangle(AgcDriver::VulkanDevice& device) {
    using namespace AgcDriver::DriverDetail;
    using namespace ShaderRecompiler;
    constexpr std::uint32_t extent = 16;
    alignas(256) std::array<std::uint32_t, extent * extent * 4> pixels{};
    constexpr std::array<std::array<float, 4>, 3> positions{{{-1, -1, 0, 1}, {-1, 1, 0, 1}, {1, -1, 0, 1}}};
    constexpr std::array<std::uint32_t, 4> values{0, 0x3e800000u, 0x3f000000u, 0x3f800000u};
    alignas(256) std::array<std::uint32_t, 16> vertexCode{0xe0382000u, 0x80000005u, 0xbf8c3f70u, 0xf80008cfu, 0x03020100u, 0x7e0802ffu, values[0], 0x7e0a02ffu, values[1], 0x7e0c02ffu, values[2], 0x7e0e02ffu, values[3], 0xf800023fu, 0x07060504u, 0xbf810000u};
    alignas(256) const std::array<std::uint32_t, 7> pixelCode{0xc8020002u, 0xc8060102u, 0xc80a0202u, 0xc80e0302u, 0xf800180fu, 0x03020100u, 0xbf810000u};
    ShaderSnapshot front{0x10000, 0, 2, {vertexCode.begin(), vertexCode.end()}, {}};
    ShaderSnapshot fragment{0x20000, 0, 1, {pixelCode.begin(), pixelCode.end()}, {}};
    for (auto* snapshot : {&front, &fragment}) {
        snapshot->header.resize(sizeof(Shader));
        snapshot->prepared->deferred = true;
    }
    const auto address = reinterpret_cast<std::uintptr_t>(positions.data());
    const std::array<std::uint32_t, 4> users{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) | (16u << 16u), 3, 0x01016facu};
    RecompileRequest vertexRequest{{ShaderStage::Vertex, front.codeAddress, front.code, 0, {}}, {32, 0, users, {}, {}, ShaderVertexStageInfo{}, {}}, device.Target(), {0, 0, 0, 128}};
    const auto vertexInvocation = InvocationFor(front, 0, vertexRequest);
    SrtRuntime vertexRuntime{};
    vertexRuntime.userData = users;
    const auto vertexCapture = vertexInvocation.Capture(vertexRuntime);
    const auto vertex = vertexInvocation.Materialize(*vertexCapture);
    const auto pushOffset = static_cast<std::uint32_t>(vertex->pushConstants.size());
    AgcDriver::Registers context{{0x191, 0x403}, {0x1b6, 0x8001}, {0x1b3, 0}, {0x1b4, 0}, {0x203, 0}, {0x1c5, 9}};
    RecompileRequest pixelRequest{{ShaderStage::Fragment, fragment.codeAddress, fragment.code, 0, {}}, {32, 0, {}, {}, AgcDriver::Graphics::DecodePixelStageInfo(context, {}), {}, {}}, device.Target(), {0, 0, pushOffset, 128 - pushOffset}};
    pixelRequest.context.pixel->targetExportMapping.fill(0xe4u);
    const auto pixelInvocation = InvocationFor(fragment, 0, pixelRequest);
    const auto pixelCapture = pixelInvocation.Capture({});
    const auto pixel = pixelInvocation.Materialize(*pixelCapture);
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0, 32, 32, {}, {}};
    state.color = {reinterpret_cast<std::uintptr_t>(pixels.data()), {extent, extent}, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(pixels), 0xe4u};
    state.color.elementBytes = 16;
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {extent, extent};
    state.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    state.rectList = true;
    state.viewport = {0, static_cast<float>(extent), static_cast<float>(extent), -static_cast<float>(extent), 0, 1};
    state.scissor = {{0, 0}, {extent, extent}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    const AgcDriver::Pm4::DrawParameters draw{0, 3, 0, 1, 0, false};
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
        const auto rectangle = PreparedRectangle(front, *vertex, *pixel, device.Target());
        const std::array<AgcDriver::Graphics::CompiledShader, 4> shaders{{{ShaderStage::Vertex, vertex.get(), 0}, {ShaderStage::TessellationControl, &rectangle.control, 0}, {ShaderStage::TessellationEvaluation, &rectangle.evaluation, 0}, {ShaderStage::Fragment, pixel.get(), pushOffset}}};
        pixels.fill(0xdeadbeefu);
        device.Draw(state, draw, shaders);
        device.WaitIdle();
        for (std::size_t index = 0; index < pixels.size(); ++index) Require(pixels[index] == values[index % 4], "first lazy rectangle draw produced incorrect pixels");
        Require(front.prepared->entries.size() == 1 && fragment.prepared->entries.size() == 1 && front.prepared->rectangles.size() == 1, "repeated lazy rectangle draw replaced its prepared artifacts");
    }
    vertexCode[1] = 0x80020005u;
    struct Header {
        Shader shader{};
        ShaderUserData users{};
        std::array<ShaderRegister, 3> registers{};
        std::array<ShaderRegister, 80> context{};
        ShaderSpecialRegs specials{};
        ShaderSemantic semantic{};
    };
    std::array<Header, 2> headers{};
    for (std::size_t index = 0; index < headers.size(); ++index) {
        auto& header = headers[index];
        const auto* code = index == 0 ? vertexCode.data() : pixelCode.data();
        const auto codeAddress = reinterpret_cast<std::uintptr_t>(code);
        const auto program = index == 0 ? 0xc8u : 0x8u;
        header.shader.file_header = 0x34333231u;
        header.shader.version = 0x18;
        header.shader.type = index == 0 ? 2u : 1u;
        header.shader.header_size = sizeof(header);
        header.shader.shader_size = index == 0 ? sizeof(vertexCode) : sizeof(pixelCode);
        header.shader.code = code;
        header.shader.user_data = &header.users;
        header.semantic.semantic = 9;
        if (index == 0) {
            header.semantic.hardware_mapping = 3;
            header.shader.output_semantics = &header.semantic;
            header.shader.num_output_semantics = 1;
            header.specials.vgt_shader_stages_en = {ShaderRegs::VGT_SHADER_STAGES_EN, 0x00402000};
            header.specials.vgt_gs_out_prim_type = {ShaderRegs::VGT_GS_OUT_PRIM_TYPE, 0};
            header.specials.ge_cntl = {ShaderRegs::GE_CNTL, 0};
            header.specials.ge_user_vgpr_en = {ShaderRegs::GE_USER_VGPR_EN, 0};
            header.shader.specials = &header.specials;
        } else {
            header.semantic.is_flat_shaded = 1;
            header.shader.input_semantics = &header.semantic;
            header.shader.num_input_semantics = 1;
        }
        header.shader.sh_registers = header.registers.data();
        header.shader.num_sh_registers = header.registers.size();
        header.registers = {{{program, static_cast<std::uint32_t>(codeAddress >> 8u)}, {program + 1u, static_cast<std::uint32_t>(codeAddress >> 40u)}, {index == 0 ? 0x8bu : 0xbu, index == 0 ? 4u << 1u : 0u}}};
        AgcDriverRegisterShader_nid_postfix(&header.shader);
    }
    AgcDriver::Registers drawContext{
        {0x2d5, 0x00402000}, {0x191, 0x403}, {0x1b6, 0x8001}, {0x1b3, 0}, {0x1b4, 0}, {0x1c3, 4}, {0x1c5, 9},
        {0x207, 0}, {0x200, 0}, {0x203, 0x800}, {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa}, {0x2f9, 0x2d}, {0x313, 0x6000},
        {0x30e, 0xffffffff}, {0x30f, 0xffffffff}, {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 3},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010}, {0x1c4, 0}, {0x31c, (7u << 8u) | (14u << 2u)},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, ((extent - 1u) << 14u) | (extent - 1u)}, {0x3b8, 0x9000000}, {0x1e0, 0},
        {0xc, 0}, {0xd, (extent << 16u) | extent}, {0x81, 0x80000000}, {0x82, (extent << 16u) | extent},
        {0x90, 0x80000000}, {0x91, (extent << 16u) | extent}, {0x94, 0x80000000}, {0x95, (extent << 16u) | extent}
    };
    const auto colorAddress = reinterpret_cast<std::uintptr_t>(pixels.data());
    drawContext[0x318] = static_cast<std::uint32_t>(colorAddress >> 8u);
    drawContext[0x390] = static_cast<std::uint32_t>(colorAddress >> 40u);
    for (const auto reg : {0x10fu, 0x110u, 0x112u}) drawContext[reg] = std::bit_cast<std::uint32_t>(extent / 2.0f);
    drawContext[0x111] = std::bit_cast<std::uint32_t>(-static_cast<float>(extent) / 2.0f);
    drawContext[0x113] = std::bit_cast<std::uint32_t>(1.0f);
    drawContext[0x114] = 0;
    drawContext[0xb4] = 0;
    drawContext[0xb5] = std::bit_cast<std::uint32_t>(1.0f);
    for (unsigned helperOrder = 0; helperOrder < 3; ++helperOrder) {
        auto submittedContext = drawContext;
        std::array<ShaderRegister, 4> primitive{};
        if (helperOrder != 0) {
            for (std::size_t stage = 0; stage < headers.size(); ++stage) {
                auto& header = headers[stage];
                header.shader.target = helperOrder;
                header.shader.cx_registers = header.context.data();
                std::size_t index = 0;
                for (const auto& [offset, value] : drawContext) {
                    if (stage == 1 && offset == ShaderRegs::VGT_SHADER_STAGES_EN) continue;
                    header.context[index++] = {offset, stage == 1 && offset == ShaderRegs::SPI_PS_INPUT_CNTL_0 ? 0x20u : value};
                }
                header.shader.num_cx_registers = index;
                AgcDriverRegisterShader_nid_postfix(&header.shader);
            }
            std::array<ShaderRegister, 3> primitiveContext{};
            std::array<ShaderRegister, 33> interpolants{};
            primitiveContext.back() = primitive.back() = interpolants.back() = {0xdeadbeefu, 0xcafebabeu};
            const auto prepare = [&] { Require(sceAgcCreatePrimState(primitiveContext.data(), primitive.data(), nullptr, &headers[0].shader, 7) == 0, "primitive helper rejected a lazy rectangle"); };
            if (helperOrder == 1) prepare();
            Require(sceAgcCreateInterpolantMapping(interpolants.data(), &headers[0].shader, &headers[1].shader) == 0, "mapping helper rejected a lazy rectangle");
            if (helperOrder == 2) prepare();
            Require(primitiveContext[0].offset == ShaderRegs::VGT_SHADER_STAGES_EN && primitiveContext[0].value == 0x00402000 && primitive[2].value == 7, "primitive helper lost its vertex-owned routing");
            Require(interpolants[0].offset == ShaderRegs::SPI_PS_INPUT_CNTL_0 && interpolants[0].value == 0x403u, "mapping helper lost its nonzero flat interpolant");
            for (const auto reg : {primitiveContext.back(), primitive.back(), interpolants.back()}) Require(reg.offset == 0xdeadbeefu && reg.value == 0xcafebabeu, "shader helper overwrote output padding");
            for (std::size_t index = 0; index < primitiveContext.size() - 1; ++index) submittedContext[primitiveContext[index].offset] = primitiveContext[index].value;
            for (std::size_t index = 0; index < interpolants.size() - 1; ++index) submittedContext[interpolants[index].offset] = interpolants[index].value;
        }
        std::vector<std::uint32_t> commands;
        for (const auto& [offset, value] : submittedContext) commands.insert(commands.end(), {0xc0016900u, offset, value});
        for (const auto& header : headers) for (const auto& reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
        for (std::size_t index = 0; index < users.size(); ++index) commands.insert(commands.end(), {0xc0017600u, 0x8cu + static_cast<std::uint32_t>(index), users[index]});
        if (helperOrder == 0) commands.insert(commands.end(), {0xc0017900u, 0x242, 7});
        else for (std::size_t index = 0; index < primitive.size() - 1; ++index) commands.insert(commands.end(), {0xc0017900u, primitive[index].offset, primitive[index].value});
        commands.insert(commands.end(), {0xc0002f00u, 1, 0xc0012d00u, 3, 2});
        const Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
        for (unsigned iteration = 0; iteration < 2; ++iteration) {
            pixels.fill(0xdeadbeefu);
            sceAgcDriverSubmitDcb(&packet);
            AgcDriverWaitIdle_nid_postfix();
            for (std::size_t index = 0; index < pixels.size(); ++index) Require(pixels[index] == values[index % 4], "first driver rectangle draw produced incorrect pixels");
        }
    }
}

void CheckRegisteredNullPixel() {
    constexpr std::uint32_t extent = 16;
    constexpr std::uint32_t pitch = 64;
    alignas(256) std::array<std::uint32_t, pitch * extent> pixels{};
    alignas(256) std::array<std::uint16_t, extent * extent> depth{};
    alignas(256) std::array<std::array<float, 4>, 3> positions{{{-1, -1, 0.5f, 1}, {3, -1, 0.5f, 1}, {-1, 3, 0.5f, 1}}};
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
        header.registers = {{{program, static_cast<std::uint32_t>(address >> 8u)}, {program + 1u, static_cast<std::uint32_t>(address >> 40u)}, {index == 0 ? 0x8bu : 0xbu, 4u << 1u}}};
        header.context = index == 0 ? std::array<ShaderRegister, 5>{{{0x2d5, 0x00402000}, {}, {}, {}, {}}} : std::array<ShaderRegister, 5>{{{0x1b6, 0x8000}, {0x1b3, 0}, {0x1b4, 0}, {0x203, 0}, {0x1c5, 9}}};
        header.shader.cx_registers = header.context.data();
        header.shader.num_cx_registers = index == 0 ? 1u : 5u;
        AgcDriverRegisterShader_nid_postfix(&header.shader);
    }
    AgcDriver::Registers context{
        {0x2d5, 0x00402000}, {0x1b6, 0}, {0x207, 0}, {0x200, 0x16}, {0x203, 0},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0}, {0x8f, 0xf}, {0x202, 0xcc0010},
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
    std::vector<std::uint32_t> commands;
    for (const auto& [offset, value] : context) commands.insert(commands.end(), {0xc0016900u, offset, value});
    for (const auto& reg : headers[0].registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
    for (std::uint32_t index = 0; index < users.size(); ++index) commands.insert(commands.end(), {0xc0017600u, 0x8cu + index, users[index]});
    commands.insert(commands.end(), {0xc0017600u, 0x008, 0, 0xc0017600u, 0x009, 0, 0xc0017600u, 0x00b, 0, 0xc0017900u, 0x242, 4, 0xc0002f00u, 1, 0xc0012d00u, 3, 2});
    const auto submit = [&] {
        const Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
        Require(sceAgcDriverSubmitDcb(&packet) == 0, "null pixel test submission failed");
        AgcDriverWaitIdle_nid_postfix();
        commands.clear();
    };
    pixels.fill(0x12345678u);
    submit();
    for (const auto pixel : pixels) Require(pixel == 0x12345678u, "absent pixel program wrote color");
    for (auto& position : positions) position[2] = 0.25f;
    for (const auto& reg : headers[1].registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
    for (std::uint32_t index = 0; index < 4; ++index) commands.insert(commands.end(), {0xc0017600u, 0x00cu + index, 0xdead0000u + index});
    commands.insert(commands.end(), {0xc0016900u, 0x1b6, 0x8000, 0xc0002f00u, 1, 0xc0012d00u, 3, 2});
    submit();
    for (const auto pixel : pixels) Require(pixel == 0x12345678u, "skipped pixel program wrote color");
    for (auto& position : positions) position[2] = 0.75f;
    commands.insert(commands.end(), {0xc0016900u, 0x8e, 0xf, 0xc0002f00u, 1, 0xc0012d00u, 3, 2});
    submit();
    for (const auto pixel : pixels) Require(pixel == 0x12345678u, "null pixel draw did not write depth");
    for (auto& position : positions) position[2] = 0.125f;
    commands.insert(commands.end(), {0xc0002f00u, 1, 0xc0012d00u, 3, 2});
    submit();
    for (std::uint32_t y = 0; y < extent; ++y) {
        for (std::uint32_t x = 0; x < pitch; ++x) Require(pixels[y * pitch + x] == (x < extent ? 0xff804000u : 0x12345678u), "restored pixel program did not preserve its color interface or target padding");
    }
    AgcDriverShutdown_nid_postfix();
}

void CheckNullPixel(AgcDriver::VulkanDevice& device) {
    using namespace AgcDriver::DriverDetail;
    using namespace ShaderRecompiler;
    Fixture vertex;
    vertex.Initialize(2);
    Shader header{};
    header.type = 1;
    auto null = std::make_shared<ShaderSnapshot>();
    null->type = 1;
    null->codeAddress = NullPixelProgramAddress();
    null->headerAddress = reinterpret_cast<std::uintptr_t>(&header);
    null->code = {0xbf810000u};
    null->header.resize(sizeof(header));
    std::memcpy(null->header.data(), &header, sizeof(header));
    ShaderPixelStageInfo canonical{};
    canonical.inputAddr = 0x2u;
    canonical.hasPerspectiveCenterVgpr = true;
    RecompileRequest registered{{ShaderStage::Fragment, null->codeAddress, null->code, null->headerAddress, null->header}, {64, 0, {}, {}, canonical, {}, {}}, device.Target(), {0, 0, 0, 128}};
    const auto handle = PrepareShader(registered);
    null->prepared->entries.push_back({0, handle});
    const auto& artifact = GetPreparedArtifact(*handle);
    ShaderRegistry registry{{vertex.snapshot->codeAddress, vertex.snapshot}, {null->codeAddress, null}};
    for (const bool absent : {true, false}) {
        for (const auto wave : {64u, 32u}) {
            for (const auto users : {0u, 4u}) {
                AgcDriver::QueueState queue{};
                vertex.Bind(queue, 0xc8, 0x8b);
                queue.shader[0x008] = absent ? 0u : 0x100u;
                queue.shader[0x009] = 0;
                queue.shader[0x00b] = users << 1u;
                for (std::uint32_t i = 0; i < 4; ++i) queue.shader[0x00c + i] = 0xdead0000u + i;
                queue.context[0x8e] = 0;
                queue.context[0x8f] = 0xf;
                queue.context[0x1c4] = 0;
                queue.context[0x203] = 0;
                queue.context[0x1b3] = 0xffffffffu;
                queue.context[0x1b4] = 0xffffffffu;
                queue.context[0x1b6] = (wave == 32 ? 0x8000u : 0u) | 1u;
                const auto shader = queue.shader;
                const auto context = queue.context;
                DrawDecode draw{};
                draw.state.stages.path = AgcDriver::Graphics::ShaderPath::Vertex;
                draw.state.stages.vertexWaveSize = 32;
                draw.state.stages.fragmentWaveSize = wave;
                DecodeGraphicsPrograms(draw, queue, registry, false, true);
                draw.pixel = AgcDriver::Graphics::DecodePixelStageInfo(queue.context, {}, AgcDriver::Graphics::PixelProgramSkipped(queue));
                const auto& program = draw.programs.back();
                RecompileRequest request{program.binary, {draw.state.stages.fragmentWaveSize, program.firstUserSgpr, program.userData, {}, draw.pixel, {}, program.memory}, device.Target(), {0, 0, 0, 128}};
                Require(SourceHandleFor(*program.snapshot, program.codeOffset, request) == handle, "null pixel draw did not reuse its canonical prepared source");
                const auto invocation = InvocationFor(*program.snapshot, program.codeOffset, request);
                SrtRuntime runtime{};
                runtime.userData = program.userData;
                const auto capture = invocation.Capture(runtime);
                const auto result = invocation.Materialize(*capture);
                Require(result->variantId == artifact.variantId && null->prepared->entries.size() == 1, "null pixel draw did not reuse its canonical prepared artifact");
                Require(program.snapshot == null && program.firstUserSgpr == 0 && program.userDataBase == 0x00c && program.userData.empty() && request.context.waveSize == 64 && !draw.pixel.wave32, "null pixel draw inherited the previous program ABI");
                Require(queue.shader == shader && queue.context == context, "null pixel draw changed the queue registers");
            }
        }
    }
    auto invalid = registered;
    invalid.context.waveSize = 32;
    Reject([&] { static_cast<void>(InvocationFor(*null, 0, invalid)); }, "artifact is missing");
    invalid = registered;
    const std::array<std::uint32_t, 4> users{};
    invalid.context.userData = users;
    Reject([&] { static_cast<void>(InvocationFor(*null, 0, invalid)); }, "artifact is missing");
    invalid = registered;
    invalid.context.pixel->inputAddr = 0;
    Reject([&] { static_cast<void>(InvocationFor(*null, 0, invalid)); }, "artifact is missing");
}

void Check(AgcDriver::VulkanDevice& device, AgcDriver::Graphics::ShaderPath path, const std::filesystem::path& dump) {
    using namespace AgcDriver::DriverDetail;
    using namespace ShaderRecompiler;
    Fixture front;
    Fixture back;
    Fixture domain;
    Fixture fragment;
    const bool tessellation = path == AgcDriver::Graphics::ShaderPath::Tessellation;
    const bool mesh = path == AgcDriver::Graphics::ShaderPath::Geometry;
    front.Initialize(tessellation ? 5u : mesh ? 4u : 2u);
    back.Initialize(tessellation ? 7u : 6u);
    domain.Initialize(2u);
    fragment.Initialize(1u);
    const std::array<std::uint32_t, 7> pixelCode{0xc8020002u, 0xc8060102u, 0xc80a0202u, 0xc80e0302u, 0xf800180fu, 0x03020100u, 0xbf810000u};
    fragment.snapshot->code.resize(64);
    fragment.snapshot->code.insert(fragment.snapshot->code.end(), pixelCode.begin(), pixelCode.end());
    ShaderRegistry registry;
    for (const auto* fixture : {&front, &back, &domain, &fragment}) registry.emplace(fixture->snapshot->codeAddress, fixture->snapshot);
    AgcDriver::QueueState queue{};
    queue.context[0x8e] = 0xfu;
    queue.context[0x8f] = 0xfu;
    front.Bind(queue, tessellation ? 0x148u : 0xc8u, tessellation ? 0x10bu : 0x8bu);
    if (tessellation || mesh) back.Bind(queue, tessellation ? 0x108u : 0x88u, tessellation ? 0x10bu : 0x8bu);
    if (tessellation) domain.Bind(queue, 0xc8u, 0x8bu);
    fragment.Bind(queue, 0x8u, 0xbu);
    alignas(8) const std::array<std::uint32_t, 2> merged{};
    const auto mergedAddress = reinterpret_cast<std::uintptr_t>(merged.data());
    const auto pointerBase = tessellation ? 0x102u : 0x82u;
    queue.shader[pointerBase] = static_cast<std::uint32_t>(mergedAddress);
    queue.shader[pointerBase + 1u] = static_cast<std::uint32_t>(mergedAddress >> 32u);
    DrawDecode prepared{};
    prepared.state.stages.path = path;
    prepared.state.stages.vertexWaveSize = tessellation || mesh ? 64u : 32u;
    prepared.state.stages.fragmentWaveSize = 32u;
    if (mesh) prepared.state.stages.mesh = MeshConfiguration{4, 1, 3, 3, 1, 64, 128, 0, 4};
    if (tessellation) prepared.state.stages.tessellation = TessellationConfiguration{3, 4, 1, 2, 2};
    prepared.pixel.wave32 = true;
    prepared.pixel.interpolatorCount = 2;
    prepared.pixel.interpolatorSettings[0] = 0x403u;
    prepared.pixel.interpolatorSettings[1] = 0x220u;
    prepared.pixel.targetOutputMode[0] = 9;
    prepared.pixel.targetExportMapping.fill(0xe4u);
    DecodeGraphicsPrograms(prepared, queue, registry, true, true);
    DrawDecode draw{};
    draw.state = prepared.state;
    draw.pixel = prepared.pixel;
    DecodeGraphicsPrograms(draw, queue, registry, false, true);
    Require(prepared.programs.size() == draw.programs.size() && prepared.roles == draw.roles, "prepared graphics programs differ from draw programs");
    for (std::size_t index = 0; index < prepared.programs.size(); ++index) {
        const auto& expected = prepared.programs[index];
        const auto& actual = draw.programs[index];
        Require(expected.codeOffset == 64u && expected.binary.code.size() == (prepared.roles[index] == ProgramRole::Fragment ? pixelCode.size() : 1u) && expected.binary.code[0] == (prepared.roles[index] == ProgramRole::Fragment ? pixelCode[0] : 0xbf810000u), "graphics entry point did not trim the code prefix");
        Require(expected.binary.stage == actual.binary.stage && expected.firstUserSgpr == actual.firstUserSgpr && expected.userData.size() == actual.userData.size(), "graphics preparation changed the user SGPR ABI");
    }
    auto target = device.Target();
    std::vector<std::uint32_t> capabilities(target.supportedCapabilities.begin(), target.supportedCapabilities.end());
    std::vector<std::string_view> extensions(target.supportedExtensions.begin(), target.supportedExtensions.end());
    if (mesh) {
        capabilities.push_back(spv::CapabilityMeshShadingEXT);
        extensions.push_back("SPV_EXT_mesh_shader");
        target.supportedCapabilities = capabilities;
        target.supportedExtensions = extensions;
        target.mesh = MeshTargetLimits{{128, 1, 1}, 128, 32768, 256, 256, 128, 32768, 1, 1};
    }
    const auto stages = PrepareGraphicsStages(prepared, target);
    Require(stages.size() == (tessellation ? 4u : 2u), "graphics preparation compiled the wrong stages");
    RecompileResult vertexResult;
    RecompileResult fragmentResult;
    static_cast<CompiledShaderArtifact&>(vertexResult) = GetPreparedArtifact(*stages.front().entry.handle);
    static_cast<CompiledShaderArtifact&>(fragmentResult) = GetPreparedArtifact(*stages.back().entry.handle);
    if (!mesh && !tessellation) {
        front.snapshot->prepared->deferred = true;
        fragment.snapshot->prepared->deferred = true;
        for (const bool primitiveFirst : {false, true}) {
            front.snapshot->prepared->fragments.clear();
            front.snapshot->prepared->rectangleRequested = false;
            if (primitiveFirst) ResolvePreparedGraphics(*front.snapshot, {}, 7, target);
            ResolvePreparedGraphics(*front.snapshot, fragment.snapshot, 0, target);
            if (!primitiveFirst) ResolvePreparedGraphics(*front.snapshot, {}, 17, target);
            Require(front.snapshot->prepared->entries.empty() && fragment.snapshot->prepared->entries.empty() && front.snapshot->prepared->rectangles.empty(), "rectangle helpers fabricated missing lazy stage artifacts");
            Require(front.snapshot->prepared->rectangleRequested && front.snapshot->prepared->fragments.size() == 1, "rectangle helpers lost a pending lazy stage pair");
        }
        const auto firstRectangle = PreparedRectangle(*front.snapshot, vertexResult, fragmentResult, target);
        const auto repeatedRectangle = PreparedRectangle(*front.snapshot, vertexResult, fragmentResult, target);
        Require(front.snapshot->prepared->entries.empty() && front.snapshot->prepared->rectangles.size() == 1 && firstRectangle.control.spirv.data() == repeatedRectangle.control.spirv.data(), "first lazy rectangle request did not reuse its generated artifacts");
        auto unsupported = target;
        unsupported.tessellation.reset();
        Reject([&] { static_cast<void>(PreparedRectangle(*front.snapshot, vertexResult, fragmentResult, unsupported)); }, "tessellation shaders are unavailable");
        Require(front.snapshot->prepared->rectangles.size() == 1, "failed lazy rectangle preparation published an artifact");
        auto changedTarget = target;
        ++changedTarget.tessellation->maxPatchSize;
        static_cast<void>(PreparedRectangle(*front.snapshot, vertexResult, fragmentResult, changedTarget));
        Require(front.snapshot->prepared->rectangles.size() == 2, "lazy rectangle cache ignored a changed target");
        ResolvePreparedGraphics(*front.snapshot, fragment.snapshot, 7, unsupported);
        front.snapshot->prepared->rectangles.clear();
        front.snapshot->prepared->fragments.clear();
        front.snapshot->prepared->rectangleRequested = false;
        front.snapshot->prepared->deferred = false;
        fragment.snapshot->prepared->deferred = false;
    }
    for (const auto& stage : stages) stage.snapshot->prepared->entries.push_back(stage.entry);
    const auto& pixelArtifact = GetPreparedArtifact(*stages.back().entry.handle);
    Require(!pixelArtifact.fragmentParameters.empty() && pixelArtifact.fragmentParameters.front().sourceLocation == 3u, "prepared fragment lost interpolant mapping");
    if (!mesh && !tessellation) {
        const auto frontCount = front.snapshot->prepared->entries.size();
        const auto pixelCount = fragment.snapshot->prepared->entries.size();
        Reject([&] {
            ShaderPreparationTransaction transaction;
            transaction.Edit(*front.snapshot).entries.push_back(stages.front().entry);
            transaction.Edit(*fragment.snapshot).entries.push_back(stages.back().entry);
            auto unsupported = target;
            unsupported.tessellation.reset();
            ResolvePreparedGraphics(*front.snapshot, fragment.snapshot, 7, unsupported);
            transaction.Commit();
        }, "tessellation shaders are unavailable");
        Require(front.snapshot->prepared->entries.size() == frontCount && fragment.snapshot->prepared->entries.size() == pixelCount && front.snapshot->prepared->rectangles.empty() && front.snapshot->prepared->fragments.empty() && !front.snapshot->prepared->rectangleRequested, "failed rectangle compilation published part of the stage group");
        for (const bool primitiveFirst : {false, true}) {
            front.snapshot->prepared->rectangles.clear();
            front.snapshot->prepared->rectangleProgress.clear();
            front.snapshot->prepared->fragments.clear();
            front.snapshot->prepared->rectangleRequested = false;
            if (primitiveFirst) ResolvePreparedGraphics(*front.snapshot, {}, 7, target);
            ResolvePreparedGraphics(*front.snapshot, fragment.snapshot, 0, target);
            if (!primitiveFirst) ResolvePreparedGraphics(*front.snapshot, {}, 17, target);
            const auto rectangle = PreparedRectangle(*front.snapshot, vertexResult, fragmentResult, target);
            Require(!rectangle.control.spirv.empty() && !rectangle.evaluation.spirv.empty(), "separate helpers did not prepare rectangle shaders");
            const auto count = front.snapshot->prepared->rectangles.size();
            ResolvePreparedGraphics(*front.snapshot, fragment.snapshot, 7, target);
            Require(front.snapshot->prepared->rectangles.size() == count, "repeated helpers duplicated rectangle shaders");
        }
        auto changed = prepared;
        changed.state.stages.vertexWaveSize = 64;
        changed.pixel.interpolatorSettings[0] = 0x404u;
        const auto newStages = PrepareGraphicsStages(changed, target);
        front.snapshot->prepared->entries.push_back(newStages.front().entry);
        ResolvePreparedGraphics(*front.snapshot, {}, 7, target);
        const auto newVertexId = GetPreparedArtifact(*newStages.front().entry.handle).variantId;
        Require(newVertexId != GetPreparedArtifact(*stages.front().entry.handle).variantId, "rectangle test did not change the vertex variant");
        RecompileResult newVertexResult;
        static_cast<CompiledShaderArtifact&>(newVertexResult) = GetPreparedArtifact(*newStages.front().entry.handle);
        static_cast<void>(PreparedRectangle(*front.snapshot, newVertexResult, fragmentResult, target));
        fragment.snapshot->prepared->entries.push_back(newStages.back().entry);
        ResolvePreparedGraphics(*front.snapshot, fragment.snapshot, 0, target);
        const auto newPixelId = GetPreparedArtifact(*newStages.back().entry.handle).variantId;
        Require(newPixelId != pixelArtifact.variantId, "rectangle test did not change the fragment variant");
        RecompileResult newFragmentResult;
        static_cast<CompiledShaderArtifact&>(newFragmentResult) = GetPreparedArtifact(*newStages.back().entry.handle);
        static_cast<void>(PreparedRectangle(*front.snapshot, newVertexResult, newFragmentResult, target));
        auto temporary = std::make_shared<ShaderSnapshot>();
        temporary->prepared->entries = fragment.snapshot->prepared->entries;
        ResolvePreparedGraphics(*front.snapshot, temporary, 0, target);
        std::weak_ptr<const ShaderSnapshot> expired = temporary;
        temporary.reset();
        Require(expired.expired(), "helper link retained a shader snapshot");
        ResolvePreparedGraphics(*front.snapshot, {}, 7, target);
        Require(front.snapshot->prepared->fragments.size() == 1, "expired helper link was not removed");
        Require(front.snapshot->prepared->rectangleProgress.size() == 1 && !front.snapshot->prepared->rectangleProgress.front().fragment.expired(), "expired rectangle progress was not removed");
    }
    if (!dump.empty()) {
        for (std::size_t index = 0; index < stages.size(); ++index) {
            const auto& words = GetPreparedArtifact(*stages[index].entry.handle).spirv;
            const auto name = dump / (std::to_string(static_cast<unsigned>(path)) + "-" + std::to_string(index) + ".spv");
            std::ofstream output(name, std::ios::binary);
            output.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(std::uint32_t)));
            Require(static_cast<bool>(output), "cannot write prepared graphics SPIR-V");
        }
    }
    auto incomplete = prepared;
    incomplete.roles.pop_back();
    Reject([&] { static_cast<void>(PrepareGraphicsStages(incomplete, target)); }, "program roles are incomplete");
    auto wrongEntry = prepared;
    wrongEntry.programs.front().binary.codeAddress += 4;
    Reject([&] { static_cast<void>(PrepareGraphicsStages(wrongEntry, target)); }, "entry point differs");
    std::vector<LinkedProgram> linked;
    std::vector<MemoryRegion> memory;
    for (std::size_t index = 0; index < draw.programs.size(); ++index) {
        const auto& program = draw.programs[index];
        linked.push_back({draw.roles[index], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
        memory.insert(memory.end(), program.memory.begin(), program.memory.end());
    }
    for (std::size_t index = 0; index < draw.programs.size(); ++index) {
        if (draw.roles[index] == ProgramRole::GeometryBack) continue;
        const auto& program = draw.programs[index];
        const bool pixel = program.binary.stage == ShaderStage::Fragment;
        std::optional<ShaderVertexStageInfo> vertex;
        if (!pixel) vertex = AgcDriver::Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, nullptr, true);
        RecompileRequest request{program.binary, {pixel ? 32u : prepared.state.stages.vertexWaveSize, program.firstUserSgpr, program.userData, {}, pixel ? std::optional(draw.pixel) : std::nullopt, vertex, memory}, target, {0, 0, 0, mesh ? MeshDrawPushOffsetBytes : 128u}, GraphicsCompileContext{program.firstUserSgpr, linked, prepared.state.stages.mesh, prepared.state.stages.tessellation, {0, 3, 4, 1}}};
        static_cast<void>(InvocationFor(*program.snapshot, program.codeOffset, request));
        request.layout.pushConstantOffsetBytes = 4;
        request.layout.pushConstantSizeBytes -= 4;
        static_cast<void>(InvocationFor(*program.snapshot, program.codeOffset, request));
        request.layout.pushConstantOffsetBytes = 2;
        Reject([&] { static_cast<void>(InvocationFor(*program.snapshot, program.codeOffset, request)); }, "artifact is missing");
        request.layout = {0, 0, 0, mesh ? MeshDrawPushOffsetBytes : 128u};
        if (pixel) request.context.pixel->interpolatorSettings[0] ^= 1u;
        else if (tessellation) ++request.graphics->tessellation->outputControlPoints;
        else if (mesh) ++request.graphics->mesh->maxVertices;
        else ++request.context.userDataBaseRegister;
        Reject([&] { static_cast<void>(InvocationFor(*program.snapshot, program.codeOffset, request)); }, "artifact is missing");
    }
    ShaderRegistry invalidRegistry;
    DrawDecode invalid{};
    invalid.state = prepared.state;
    Reject([&] { DecodeGraphicsPrograms(invalid, queue, invalidRegistry, true, true); }, "registered");
    const auto frontBase = tessellation ? 0x148u : 0xc8u;
    queue.shader[frontBase + 1u] |= 0x100u;
    Reject([&] { DecodeGraphicsPrograms(invalid, queue, registry, true, true); }, "reserved graphics program address");
}

}

int main(int argc, char** argv) {
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Require(argc <= 2, "invalid test arguments");
        const auto dump = argc == 2 ? std::filesystem::path(argv[1]) : std::filesystem::path{};
        for (const auto waveSize : {32u, 64u}) DrawPreparedRectangle(*device, waveSize);
        DrawLazyRectangle(*device);
        CheckRegisteredNullPixel();
        CheckNullPixel(*device);
        Check(*device, AgcDriver::Graphics::ShaderPath::Vertex, dump);
        Check(*device, AgcDriver::Graphics::ShaderPath::Geometry, dump);
        Check(*device, AgcDriver::Graphics::ShaderPath::Tessellation, dump);
        std::cout << "prepared graphics ABI tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { AgcDriverShutdown_nid_postfix(); }
        catch (const std::exception& shutdownError) { std::cerr << shutdownError.what() << '\n'; }
        return 1;
    }
}
