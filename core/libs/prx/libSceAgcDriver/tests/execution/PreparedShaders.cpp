#include "VulkanTestDevice.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparation.hpp"
#include <chrono>
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <future>
#include <barrier>

namespace {

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename TAction>
void ExpectFailure(TAction action, const char* expected) {
    try { action(); }
    catch (const std::exception& error) {
        Require(std::string(error.what()).find(expected) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("expected preparation failure");
}

void RegisterWhileGpuLocked(const Shader* shader) {
    std::promise<void> registering;
    auto started = registering.get_future();
    std::future<void> registration;
    std::future<void> preparation;
    bool available;
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        registration = std::async(std::launch::async, [&] {
            registering.set_value();
            AgcDriverRegisterShader_nid_postfix(shader);
        });
        started.get();
        const auto waiting = registration.wait_for(std::chrono::milliseconds(10));
        preparation = std::async(std::launch::async, [] {
            AgcDriver::DriverDetail::ShaderPreparationTransaction transaction;
            transaction.Commit();
        });
        available = waiting == std::future_status::timeout && preparation.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready;
    }
    registration.get();
    preparation.get();
    Require(available, "shader registration held the preparation writer while waiting for the GPU mutex");
}

void RunLazyPreparation(AgcDriver::VulkanDevice& device, ShaderRecompiler::RecompileRequest request, bool deferred = false) {
    const auto code = request.shader.code;
    for (const bool sourceFirst : {false, true}) {
        AgcDriver::DriverDetail::ShaderSnapshot snapshot{request.shader.codeAddress, 0, 0, {code.begin(), code.end()}, {}};
        if (deferred) {
            snapshot.header.resize(sizeof(Shader));
            snapshot.prepared->deferred = true;
        }
        request.shader.code = snapshot.code;
        if (deferred) {
            ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, snapshot.code.size(), request)); }, "outside the snapshot");
            auto truncated = request;
            truncated.shader.code = std::span(snapshot.code).first(snapshot.code.size() - 1);
            ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, truncated)); }, "does not refer to registered code");
            Require(snapshot.prepared->entries.empty(), "invalid lazy source published an artifact");
        }
        if (sourceFirst) {
            const auto handle = AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request);
            Require(snapshot.prepared->entries.size() == 1 && snapshot.prepared->entries.front().handle == handle, "unregistered compute source was not cached");
        }
        static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request));
        Require(snapshot.prepared->entries.size() == 1, "unregistered compute invocation did not cache exactly one artifact");
        const auto handle = snapshot.prepared->entries.front().handle;
        const auto& artifact = ShaderRecompiler::GetPreparedArtifact(*handle);
        std::shared_ptr<const ShaderRecompiler::RecompileResult> firstResult;
        for (const bool useCache : {false, true}) {
            request.useCache = useCache;
            Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request) == handle, "unregistered compute source was prepared again");
            const auto repeated = AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request);
            ShaderRecompiler::SrtRuntime runtime{};
            runtime.userData = request.context.userData;
            const auto capture = repeated.Capture(runtime);
            const auto result = repeated.Materialize(*capture);
            if (!firstResult) firstResult = result;
            Require(snapshot.prepared->entries.size() == 1 && result->variantId == artifact.variantId && result->spirv.data() == firstResult->spirv.data() && result->PipelineVariantId() == firstResult->PipelineVariantId(), "lazy compute invocation replaced its cached artifact or specialization");
            const auto repeatedCapture = repeated.Capture(runtime);
            const auto repeatedResult = repeated.Materialize(*repeatedCapture);
            Require(repeatedResult->variantId == result->variantId && repeatedResult->spirv.data() == result->spirv.data() && repeatedResult->PipelineVariantId() == result->PipelineVariantId(), "lazy compute invocation rebuilt an unchanged specialization");
            device.Dispatch(*result, 1, 1, 1);
        }
        device.WaitIdle();
        if (deferred) {
            request.context.compute->numThreads[0] += 1;
            const auto changed = AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request);
            Require(snapshot.prepared->entries.size() == 2 && snapshot.prepared->entries.back().handle != handle, "deferred preparation reused an artifact with a changed static ABI");
            const auto changedHandle = snapshot.prepared->entries.back().handle;
            Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request) == changedHandle, "changed deferred source was prepared again");
            ShaderRecompiler::SrtRuntime runtime{};
            runtime.userData = request.context.userData;
            const auto capture = changed.Capture(runtime);
            device.Dispatch(*changed.Materialize(*capture), 1, 1, 1);
            device.WaitIdle();
            request.context.compute->numThreads[0] -= 1;
            Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request) == handle && snapshot.prepared->entries.size() == 2, "deferred preparation lost the previous static ABI");
        }
    }
}

void Run(AgcDriver::VulkanDevice& device) {
    alignas(256) std::array<std::uint32_t, 1> code{0xbf810000u};
    std::array<std::uint32_t, 4> users{};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{1, 1, 1}, 0, {false, false, false}, false, 1, {}};
    ShaderRecompiler::RecompileRequest request{{ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}}, {32, 0, users, compute, {}, {}, {}}, device.ComputeTarget(32), {0, 0, 0, 128}};
    RunLazyPreparation(device, request);
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{request.shader.codeAddress, 0, 0, {code.begin(), code.end()}, {}};
    snapshot.header.resize(sizeof(Shader));
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request)); }, "artifact is missing");
    const auto handle = ShaderRecompiler::PrepareShader(request);
    snapshot.prepared->entries.push_back({0, handle});
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 1, request)); }, "artifact is missing");
    const auto& artifact = ShaderRecompiler::GetPreparedArtifact(*handle);
    for (std::uint32_t value = 0; value < 8; ++value) {
        users.fill(value);
        request.useCache = value % 2 == 0;
        const auto ready = AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request);
        ShaderRecompiler::SrtRuntime runtime{};
        runtime.userData = users;
        const auto capture = ShaderRecompiler::CaptureResources(request, runtime, *ready);
        const auto result = ShaderRecompiler::MaterializeShader(request, *capture, *ready);
        Require(result->variantId == artifact.variantId && result->spirv.data() == artifact.spirv.data(), "invocation replaced the prepared artifact");
        device.Dispatch(*result, 1, 1, 1);
    }
    device.WaitIdle();
    auto registeredRequest = request;
    registeredRequest.shader.code = snapshot.code;
    registeredRequest.useCache = true;
    const auto invocation = AgcDriver::DriverDetail::InvocationFor(snapshot, 0, registeredRequest);
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, request)); }, "does not refer to registered code");
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 1, registeredRequest)); }, "outside the snapshot");
    registeredRequest.context.compute->numThreads[0] = 2;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, registeredRequest)); }, "artifact is missing");
    ShaderRecompiler::SrtRuntime preparedRuntime{};
    preparedRuntime.userData = users;
    const auto preparedCapture = invocation.Capture(preparedRuntime);
    const auto firstResult = invocation.Materialize(*preparedCapture);
    const auto repeatedCapture = invocation.Capture(preparedRuntime);
    Require(invocation.Materialize(*repeatedCapture) == firstResult, "prepared invocation rebuilt an unchanged materialized result");
    users[0] += 1;
    const auto changedCapture = invocation.Capture(preparedRuntime);
    const auto changedResult = invocation.Materialize(*changedCapture);
    Require(changedResult != firstResult && changedResult->spirv.data() == firstResult->spirv.data(), "prepared invocation did not distinguish changed runtime data");
    users[0] -= 1;
    Require(invocation.Materialize(*preparedCapture) == firstResult, "prepared invocation evicted the previous runtime data");
    for (std::uint32_t other = 1; other <= 600; ++other) {
        users[1] = other;
        static_cast<void>(invocation.Materialize(*invocation.Capture(preparedRuntime)));
    }
    users[1] = users[0];
    const char* memoEntries = std::getenv("APS5_RESULT_MEMO_ENTRIES");
    const bool kept = memoEntries == nullptr || std::atoi(memoEntries) > 601;
    Require((invocation.Materialize(*preparedCapture) == firstResult) == kept, kept ? "the result memo lost a result behind 600 others of its program" : "the result memo of the size asked for kept more results than that");
    auto replacement = OpenVulkanTestDevice();
    Require(replacement != nullptr && replacement->Serial() != device.Serial(), "replacement device was not created");
    auto replacementRequest = request;
    replacementRequest.target = replacement->ComputeTarget(32);
    Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, replacementRequest) == handle, "device replacement discarded a compatible prepared artifact");
    replacementRequest.shader.code = snapshot.code;
    const auto replacementInvocation = AgcDriver::DriverDetail::InvocationFor(snapshot, 0, replacementRequest);
    const auto replacementCapture = replacementInvocation.Capture(preparedRuntime);
    const auto replacementResult = replacementInvocation.Materialize(*replacementCapture);
    Require(replacementResult->spirv.data() == artifact.spirv.data(), "device replacement recompiled the prepared shader");
    replacement->Dispatch(*replacementResult, 1, 1, 1);
    replacement->WaitIdle();
    replacementRequest.target.nonConstantImageOffsets = !replacementRequest.target.nonConstantImageOffsets;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::InvocationFor(snapshot, 0, replacementRequest)); }, "artifact is missing");
    ShaderRecompiler::SrtRuntime runtime{};
    runtime.userData = users;
    const auto capture = ShaderRecompiler::CaptureResources(request, runtime, *handle);
    request.context.compute->numThreads[0] = 2;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request)); }, "artifact is missing");
    ExpectFailure([&] { static_cast<void>(ShaderRecompiler::CaptureResources(request, runtime, *handle)); }, "does not match the static ABI");
    ExpectFailure([&] { static_cast<void>(ShaderRecompiler::MaterializeShader(request, *capture, *handle)); }, "does not match the static ABI");
    const auto otherHandle = ShaderRecompiler::PrepareShader(request);
    const auto otherCapture = ShaderRecompiler::CaptureResources(request, runtime, *otherHandle);
    ExpectFailure([&] { static_cast<void>(invocation.Materialize(*otherCapture)); }, "another prepared shader");
    ExpectFailure([&] { static_cast<void>(ShaderRecompiler::MaterializeShader(request, *capture, *otherHandle)); }, "another prepared shader");
    request.context.compute->numThreads[0] = 1;
    request.layout.pushConstantSizeBytes = 124;
    Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request) == handle, "compatible push constant capacity discarded the prepared artifact");
    request.layout.pushConstantSizeBytes = 126;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request)); }, "artifact is missing");
    request.layout.pushConstantSizeBytes = 128;
    code[0] = 0xffffffffu;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, request)); }, "artifact is missing");
}

void RunInactiveSrtReads(AgcDriver::VulkanDevice& device, bool memorySelector) {
    using namespace ShaderRecompiler;
    alignas(256) std::array<std::uint32_t, 64> output{};
    alignas(4096) std::uint32_t payload = 0u;
    alignas(4096) std::uint32_t selectorWord = 0u;
    const auto outputAddress = reinterpret_cast<std::uintptr_t>(output.data() + 16u);
    const auto payloadAddress = reinterpret_cast<std::uintptr_t>(&payload);
    const auto selectorAddress = reinterpret_cast<std::uintptr_t>(&selectorWord);
    std::array<std::uint32_t, 13> users{};
    users[0] = static_cast<std::uint32_t>(outputAddress);
    users[1] = static_cast<std::uint32_t>((outputAddress >> 32u) & 0xffffu);
    users[2] = 32u;
    users[3] = 0x31016facu;
    users[6] = static_cast<std::uint32_t>(selectorAddress);
    users[7] = static_cast<std::uint32_t>(selectorAddress >> 32u);
    const auto dispatch = [&](const RecompileResult& result, std::uint32_t expected, const char* message) {
        output.fill(0xdeadbeefu);
        auto expectedOutput = output;
        expectedOutput[16] = expected;
        device.Dispatch(result, 1u, 1u, 1u);
        device.WaitIdle();
        const auto actualBytes = std::as_bytes(std::span(output));
        const auto expectedBytes = std::as_bytes(std::span(expectedOutput));
        Require(std::equal(actualBytes.begin(), actualBytes.end(), expectedBytes.begin()), message);
    };
    for (const std::uint32_t waveSize : {32u, 64u}) {
        for (const bool inverted : {false, true}) {
            alignas(256) std::array<std::uint32_t, 17> code{
                0xbf06800cu, 0xbf850005u, 0xf4000404u, 0xfa000000u, 0xbf8cc07fu, 0x7e020210u, 0xbf820004u,
                0xf4000485u, 0xfa000000u, 0xbf8cc07fu, 0x7e020212u, 0xe0700000u, 0x80000100u, 0xbf810000u
            };
            code[1] = inverted ? 0xbf840005u : 0xbf850005u;
            if (memorySelector) {
                std::move_backward(code.begin(), code.begin() + 14u, code.end());
                code[0] = 0xf4000303u;
                code[1] = 0xfa000000u;
                code[2] = 0xbf8cc07fu;
            }
            const auto shaderCode = std::span(code).first(memorySelector ? 17u : 14u);
            const auto codeAddress = reinterpret_cast<std::uintptr_t>(code.data());
            AgcDriver::DriverDetail::ShaderSnapshot snapshot{codeAddress, 0u, 0u, {shaderCode.begin(), shaderCode.end()}, {}};
            const ShaderComputeStageInfo compute{{1u, 1u, 1u}, 0u, {false, false, false}, false, 1u, {}};
            RecompileRequest request{{ShaderStage::Compute, codeAddress, snapshot.code, 0u, {}}, {waveSize, 0u, users, compute, {}, {}, {}}, device.ComputeTarget(waveSize), {0u, 0u, 0u, 128u}};
            const auto handle = AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0u, request);
            for (const auto selector : {0u, 1u, 0u}) {
                for (const bool useCache : {false, true}) {
                    request.useCache = useCache;
                    request.context.memory = {};
                    const bool taken = (selector == 0u) != inverted;
                    const auto pointerSlot = taken ? 10u : 8u;
                    users[8] = users[9] = users[10] = users[11] = 0u;
                    users[pointerSlot] = static_cast<std::uint32_t>(payloadAddress);
                    users[pointerSlot + 1u] = static_cast<std::uint32_t>(payloadAddress >> 32u);
                    users[12] = memorySelector ? selector ^ 1u : selector;
                    selectorWord = selector;
                    payload = 0x12345678u + selector + (useCache ? 16u : 0u);
                    const auto expected = payload;
                    AgcDriver::ShaderMemory memory({});
                    static_cast<void>(memory.Capture(request));
                    const auto regions = memory.Regions();
                    Require(regions.size() == (memorySelector ? 2u : 1u), "inactive SRT capture recorded an unused pointer");
                    for (const auto& region : regions) {
                        Require((region.guestAddress == payloadAddress || (memorySelector && region.guestAddress == selectorAddress)) && region.bytes.size() == sizeof(payload), "inactive SRT capture recorded an unexpected range");
                    }
                    request.context.memory = regions;
                    const auto ordinary = Recompile(request);
                    Require(useCache || !ordinary.cacheHit, "inactive SRT uncached compilation reused a variant");
                    dispatch(ordinary, expected, "inactive SRT ordinary dispatch changed the selected payload or output padding");
                    if (useCache) {
                        const auto cached = Recompile(request);
                        Require(cached.cacheHit, "inactive SRT unchanged request missed the shader cache");
                        dispatch(cached, expected, "inactive SRT cached dispatch changed the selected payload or output padding");
                    }
                    Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0u, request) == handle, "inactive SRT runtime data replaced the prepared source");
                    const auto invocation = AgcDriver::DriverDetail::InvocationFor(snapshot, 0u, request);
                    AgcDriver::ShaderMemory preparedMemory({});
                    const auto capture = preparedMemory.Capture(invocation);
                    const auto serialized = RequestSerializer{}.Serialize(request);
                    payload = 0xbad0bad0u;
                    users[8] = users[9] = users[10] = users[11] = 0u;
                    users[12] ^= 1u;
                    selectorWord ^= 1u;
                    const auto materialized = invocation.Materialize(*capture);
                    dispatch(*materialized, expected, "inactive SRT prepared dispatch changed after live memory changed following capture");
                    payload = 0xfeedabcdu;
                    selectorWord ^= 1u;
                    dispatch(*materialized, expected, "inactive SRT materialized dispatch changed after live memory changed");
                    auto replay = RequestSerializer{}.Deserialize(serialized);
                    dispatch(Recompile(replay.request), expected, "inactive SRT replay changed the selected payload or output padding after live memory changed");
                    dispatch(*invocation.Materialize(*capture), expected, "inactive SRT prepared capture changed after live memory changed");
                    Require(snapshot.prepared->entries.size() == 1u, "inactive SRT toggling prepared the source repeatedly");
                }
            }
        }
    }
    std::cout << (memorySelector ? "flat-backed SRT conditions" : "inactive SRT") << " wave32/64 branch, prepared, cache and snapshot replay tests passed\n";
}

class GuestScalarData {
public:
    explicit GuestScalarData(std::uint32_t word) : data(static_cast<std::uint32_t*>(std::aligned_alloc(65536u, 65536u))) {
        Require(data != nullptr, "GPU-controlled scalar data allocation failed");
        std::fill_n(data, 65536u / sizeof(std::uint32_t), 0u);
        data[0] = word;
        GuestAllocations::Mutation().Add(data, 65536u, true, false);
    }
    ~GuestScalarData() {
        GuestAllocations::Mutation().Remove(data);
        std::free(data);
    }
    GuestScalarData(const GuestScalarData&) = delete;
    GuestScalarData& operator=(const GuestScalarData&) = delete;
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(data); }
    void Set(std::size_t index, std::uint32_t word) { data[index] = word; }
private:
    std::uint32_t* data;
};

void RunGpuControlledScalarCase(AgcDriver::VulkanDevice& device, std::uint32_t waveSize, bool inverted, bool buffer, std::uint32_t value, bool useCache, bool invalidActive) {
    using namespace ShaderRecompiler;
    alignas(256) std::array<std::uint32_t, 64> output{};
    GuestScalarData payload(value);
    GuestScalarData table(static_cast<std::uint32_t>(payload.Address()));
    table.Set(1u, static_cast<std::uint32_t>((payload.Address() >> 32u) & 0xffffu));
    table.Set(2u, 4u);
    table.Set(3u, 0x31016facu);
    const auto outputAddress = reinterpret_cast<std::uintptr_t>(output.data() + 16u);
    std::array<std::uint32_t, 12> users{};
    users[0] = static_cast<std::uint32_t>(outputAddress);
    users[1] = static_cast<std::uint32_t>((outputAddress >> 32u) & 0xffffu);
    users[2] = 32u;
    users[3] = 0x31016facu;
    users[8] = static_cast<std::uint32_t>(table.Address());
    users[9] = static_cast<std::uint32_t>(table.Address() >> 32u);
    alignas(256) std::array<std::uint32_t, 22> code{
        0xbf06800cu, 0xbf850004u, 0xf4040405u, 0xfa000030u, 0xbf8cc07fu, 0xbf820003u,
        0xf4040404u, 0xfa000000u, 0xbf8cc07fu, 0xf4000488u, 0xfa000000u, 0xbf8cc07fu,
        0xbf068012u, 0xbf850003u, 0x7e0202ffu, 0x13579bdfu, 0xbf820002u,
        0x7e0202ffu, 0x2468ace0u, 0xe0700000u, 0x80000100u, 0xbf810000u
    };
    if (buffer) {
        code[2] = 0xf4080405u;
        code[6] = 0xf4080404u;
        code[9] = 0xf4200508u;
        code[12] = 0xbf068014u;
    }
    if (inverted) {
        code[1] = 0xbf840004u;
        std::swap(code[2], code[6]);
        std::swap(code[3], code[7]);
    }
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(code.data());
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{codeAddress, 0u, 0u, {code.begin(), code.end()}, {}};
    const ShaderComputeStageInfo compute{{1u, 1u, 1u}, 0u, {true, false, false}, false, 1u, {}};
    RecompileRequest request{{ShaderStage::Compute, codeAddress, snapshot.code, 0u, {}}, {waveSize, 0u, users, compute, {}, {}, {}}, device.ComputeTarget(waveSize), {0u, 0u, 0u, 128u}};
    request.useCache = useCache;
    const auto handle = AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0u, request);
    AgcDriver::ShaderMemory memory({});
    static_cast<void>(memory.Capture(request));
    Require(memory.Regions().empty(), "GPU-controlled scalar payload was captured on the CPU");
    const auto ordinary = Recompile(request);
    Require(useCache || !ordinary.cacheHit, "GPU-controlled uncached compilation reused a variant");
    const auto dispatch = [&](const RecompileResult& result, std::uint32_t expectedValue) {
        output.fill(0xdeadbeefu);
        auto expected = output;
        expected[16] = expectedValue == 0u ? 0x2468ace0u : 0x13579bdfu;
        device.Dispatch(result, invalidActive ? 2u : 1u, 1u, 1u);
        device.WaitIdle();
        if (invalidActive) {
            std::cout << "GPU-controlled active-invalid dispatch completed; verify deferred BDA fault\n";
            return;
        }
        const auto actualBytes = std::as_bytes(std::span(output));
        const auto expectedBytes = std::as_bytes(std::span(expected));
        Require(std::equal(actualBytes.begin(), actualBytes.end(), expectedBytes.begin()), "GPU-controlled scalar pointer/V# changed its payload or output padding");
        std::cout << "GPU-controlled scalar pointer/V# full 256-byte output verified\n";
    };
    dispatch(ordinary, value);
    if (invalidActive) return;
    if (useCache) {
        const auto cached = Recompile(request);
        Require(cached.cacheHit, "GPU-controlled unchanged request missed the shader cache");
        dispatch(cached, value);
    }
    const auto invocation = AgcDriver::DriverDetail::InvocationFor(snapshot, 0u, request);
    AgcDriver::ShaderMemory preparedMemory({});
    const auto capture = preparedMemory.Capture(invocation);
    Require(preparedMemory.Regions().empty(), "GPU-controlled prepared source captured a native payload");
    const auto serialized = RequestSerializer{}.Serialize(request);
    payload.Set(0u, 1u - value);
    auto replay = RequestSerializer{}.Deserialize(serialized);
    dispatch(Recompile(replay.request), 1u - value);
    dispatch(*invocation.Materialize(*capture), 1u - value);
    Require(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0u, request) == handle && snapshot.prepared->entries.size() == 1u, "GPU-controlled live data replaced its prepared source");
}

void RunGpuControlledScalarReads(AgcDriver::VulkanDevice& device, bool invalidActive = false) {
    for (const auto wave : {32u, 64u}) {
        for (const bool inverted : {false, true}) {
            for (const bool buffer : {false, true}) {
                for (const auto value : {0u, 1u}) {
                    for (const bool useCache : {false, true}) {
                        if (invalidActive) {
                            if (useCache) continue;
                            auto isolated = OpenVulkanTestDevice();
                            Require(isolated != nullptr, "GPU-controlled fault control requires Vulkan");
                            RunGpuControlledScalarCase(*isolated, wave, inverted, buffer, value, useCache, true);
                        } else {
                            RunGpuControlledScalarCase(device, wave, inverted, buffer, value, useCache, false);
                        }
                    }
                }
            }
        }
    }
}

void PrepareMultisampledStorage(AgcDriver::VulkanDevice& device) {
    using namespace ShaderRecompiler;
    std::array<std::uint32_t, 13> code{0xd7460000u, 0x0401060cu, 0xd7460001u, 0x0405060du, 0x7e04020eu, 0x7e060280u, 0x7e080208u, 0x7e0a0209u, 0x7e0c020au, 0x7e0e020bu, 0xf0200f38u, 0x00000400u, 0xbf810000u};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x30000u, code, 0, {}};
    const std::array<std::uint32_t, 16> users{};
    request.context.userData = users;
    request.context.waveSize = 32;
    request.context.compute = ShaderComputeStageInfo{{8, 8, 1}, 0, {true, true, false}, false, 2, {}};
    request.target = device.ComputeTarget(32);
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    for (std::uint32_t variant = 0; variant < 5; ++variant) {
        if (variant != 0u) {
            code[10] = variant % 2u != 0u ? 0xf0200f38u : 0xf0200f30u;
            if (variant >= 3u) code[10] = variant == 3u ? 0xf03c0138u : 0xf03c0130u;
            code[5] = 0x7e060283u;
            code[4] = 0x7e040282u;
        }
        const auto handle = PrepareShader(request);
        const auto& artifact = GetPreparedArtifact(*handle);
        std::vector<std::uint32_t> samples;
        for (std::size_t offset = 5; offset < artifact.spirv.size(); offset += artifact.spirv[offset] >> 16u) {
            const auto instruction = artifact.spirv[offset];
            if ((instruction & 0xffffu) == spv::OpImageWrite) {
                Require(variant < 3u && (instruction >> 16u) == 6u && artifact.spirv[offset + 4] == spv::ImageOperandsSampleMask, "MSAA storage write lost its sample operand");
                samples.push_back(artifact.spirv[offset + 5]);
            }
            if ((instruction & 0xffffu) == spv::OpImageTexelPointer) {
                Require(variant >= 3u && (instruction >> 16u) == 6u, "MSAA atomic lost its sample operand");
                samples.push_back(artifact.spirv[offset + 5]);
            }
        }
        Require(!samples.empty(), "MSAA storage shader has no image writes or atomics");
        const auto expectedSample = variant == 0u ? 0u : variant % 2u != 0u ? 3u : 2u;
        for (const auto sample : samples) {
            bool found = false;
            for (std::size_t offset = 5; offset < artifact.spirv.size(); offset += artifact.spirv[offset] >> 16u) {
                if ((artifact.spirv[offset] & 0xffffu) != spv::OpConstant || artifact.spirv[offset + 2] != sample) continue;
                Require(artifact.spirv[offset + 3] == expectedSample, "MSAA storage operation addresses the wrong sample");
                found = true;
            }
            Require(found, "MSAA sample constant is missing");
        }
    }
    std::vector<std::uint32_t> capabilities(request.target.supportedCapabilities.begin(), request.target.supportedCapabilities.end());
    std::erase(capabilities, spv::CapabilityStorageImageMultisample);
    request.target.supportedCapabilities = capabilities;
    ExpectFailure([&] { static_cast<void>(PrepareShader(request)); }, "storage image multisampling is unavailable on the target device");
}

void GraphicsHeaderCopies(bool stagesFirst) {
    alignas(256) std::array<std::uint32_t, 1> vertexCode{0xbf810000u};
    alignas(256) std::array<std::uint32_t, 1> pixelCode{0xbf810000u};
    struct Header {
        Shader shader{};
        ShaderUserData users{};
        std::array<ShaderRegister, 3> registers{};
        std::array<ShaderRegister, 6> context{{{0x2d5, 0x00402000u}, {0x1b6, 0x8000u}, {0x1b3, 0}, {0x1b4, 0}, {0x203, 0}, {0x1c5, 0}}};
        std::array<std::uint16_t, ShaderRegs::AGC_DIRECT_RESOURCE_TYPE_COUNT> offsets{};
        ShaderSemantic semantic{};
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
        header.shader.target = stagesFirst ? 1u : 0u;
        header.shader.header_size = sizeof(header);
        header.shader.shader_size = sizeof(vertexCode);
        header.shader.code = code;
        header.shader.user_data = &header.users;
        header.shader.sh_registers = header.registers.data();
        header.shader.num_sh_registers = header.registers.size();
        header.shader.cx_registers = header.context.data();
        header.shader.num_cx_registers = header.context.size();
        header.registers = {{{program, static_cast<std::uint32_t>(address >> 8u)}, {program + 1u, static_cast<std::uint32_t>(address >> 40u)}, {index == 0 ? 0x8bu : 0xbu, index == 0 ? 4u << 1u : 0u}}};
    }
    auto& vertex = headers.front();
    vertex.offsets.fill(ShaderRegs::AGC_ILLEGAL_DIRECT_OFFSET);
    vertex.offsets[static_cast<std::size_t>(ShaderRegs::AgcDirectResourceType::PtrVertexBufferTable)] = 0;
    vertex.offsets[static_cast<std::size_t>(ShaderRegs::AgcDirectResourceType::PtrVertexAttribDescTable)] = 2;
    vertex.users.direct_resource_offset = vertex.offsets.data();
    vertex.users.direct_resource_count = vertex.offsets.size();
    vertex.shader.input_semantics = &vertex.semantic;
    vertex.shader.num_input_semantics = 1;
    vertex.semantic.hardware_mapping = 8;
    vertex.semantic.size_in_elements = 4;
    const std::array<std::uint32_t, 4> users{};
    const auto info = AgcDriver::Graphics::DecodeVertexStageInfo(std::as_bytes(std::span(&vertex, 1)), reinterpret_cast<std::uintptr_t>(&vertex), users, nullptr, true);
    Require(info.fetchEmbedded && info.fetchBufferReg == 0 && info.fetchAttribReg == 2 && info.resourcesNum == 1 && info.resourcesDst[0].registerStart == 8 && info.resourcesDst[0].registersNum == 4, "registered header lost its vertex fetch ABI");
    for (const auto& header : headers) AgcDriverRegisterShader_nid_postfix(&header.shader);
    std::array<Shader, 2> copies{headers[0].shader, headers[1].shader};
    static_assert(offsetof(Shader, num_sh_registers) + sizeof(Shader::num_sh_registers) < sizeof(Shader));
    for (auto& copy : copies) {
        copy.user_data = nullptr;
        reinterpret_cast<std::byte*>(&copy)[sizeof(Shader) - 1] = std::byte{0x7d};
    }
    const std::array<const Shader*, 2> stages{&copies[0], &copies[1]};
    const std::array<ShaderRegister, 1> primitive{{{0x242, 7}}};
    const auto resolve = [&] {
        if (stagesFirst) AgcDriverResolveGraphicsStagesAbi_nid_postfix(stages, {}, primitive);
        AgcDriverResolveGraphicsAbi_nid_postfix(&copies[0], &copies[1], 7);
        if (!stagesFirst) AgcDriverResolveGraphicsStagesAbi_nid_postfix(stages, {}, primitive);
    };
    resolve();
    for (auto& copy : copies) {
        copy.target ^= 1u;
        ExpectFailure([&] { AgcDriverResolveGraphicsAbi_nid_postfix(&copies[0], &copies[1], 7); }, "rectangle ABI refers to a replaced shader header");
        ExpectFailure([&] { AgcDriverResolveGraphicsStagesAbi_nid_postfix(stages, {}, primitive); }, "graphics ABI refers to a replaced shader header");
        copy.target ^= 1u;
        copy.shader_size += 4;
        ExpectFailure([&] { AgcDriverResolveGraphicsAbi_nid_postfix(&copies[0], &copies[1], 7); }, "rectangle ABI refers to a replaced shader header");
        ExpectFailure([&] { AgcDriverResolveGraphicsStagesAbi_nid_postfix(stages, {}, primitive); }, "graphics ABI refers to a replaced shader header");
        copy.shader_size -= 4;
    }
    resolve();
}

void Registration(bool indirect) {
    alignas(256) std::array<std::uint32_t, 1> code{0xbf810000u};
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 9> registers{};
        std::array<ShaderRegister, 2> context{{{0x1b6, 0}, {0x1b6, 1}}};
        ShaderSpecialRegs specials{};
    } header;
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = sizeof(code);
    header.shader.code = code.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.cx_registers = header.context.data();
    header.shader.num_cx_registers = header.context.size();
    header.shader.specials = &header.specials;
    header.specials.dispatch_modifier = 0x8000;
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, 2}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 0}, {0x207, 1}, {0x207, 1}}};
    const auto threadRegisterIndex = header.registers.size() - 1;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    std::barrier start(4);
    std::vector<std::future<void>> registrations;
    for (unsigned worker = 0; worker < 4; ++worker) {
        registrations.push_back(std::async(std::launch::async, [&] {
            start.arrive_and_wait();
            for (unsigned iteration = 0; iteration < 8; ++iteration) {
                AgcDriverRegisterShader_nid_postfix(&header.shader);
                AgcDriverResolveShaderAbi_nid_postfix(&header.shader, {}, {});
            }
        }));
    }
    for (auto& registration : registrations) registration.get();
    Shader copy = header.shader;
    copy.user_data = reinterpret_cast<ShaderUserData*>(&copy);
    AgcDriverResolveShaderAbi_nid_postfix(&copy, {}, {});
    alignas(Shader) std::array<std::byte, sizeof(Shader)> padded{};
    std::memcpy(padded.data(), &copy, sizeof(Shader));
    padded.back() = std::byte{0x7d};
    AgcDriverResolveShaderAbi_nid_postfix(reinterpret_cast<Shader*>(padded.data()), {}, {});
    copy.target ^= 1u;
    ExpectFailure([&] { AgcDriverResolveShaderAbi_nid_postfix(&copy, {}, {}); }, "replaced shader header");
    header.registers[1].value |= 0x100u;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "invalid registered program address");
    header.registers[1].value &= 0xffu;
    ++header.registers[0].value;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "entry point is outside shader code");
    --header.registers[0].value;
    header.shader.num_sh_registers = 255;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "truncated shader metadata");
    header.shader.num_sh_registers = header.registers.size();
    header.registers[threadRegisterIndex].value = 0;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "must be nonzero");
    header.registers[threadRegisterIndex].value = 1;
    code[0] = 0xffffffffu;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "");
    std::vector<std::uint32_t> commands;
    for (const auto reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
    const std::array<std::uint32_t, 3> arguments{1, 1, 1};
    const auto argumentAddress = reinterpret_cast<std::uintptr_t>(arguments.data());
    if (indirect) commands.insert(commands.end(), {0xc0021600u, static_cast<std::uint32_t>(argumentAddress), static_cast<std::uint32_t>(argumentAddress >> 32u), 0x8041});
    else commands.insert(commands.end(), {0xc0031500u, 1, 1, 1, 0x8041});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    for (std::uint32_t iteration = 0; iteration < 3; ++iteration) {
        sceAgcDriverSubmitAcb(0x20, &packet);
        AgcDriverWaitIdle_nid_postfix();
    }
    std::uint32_t destination = 0;
    const auto destinationAddress = reinterpret_cast<std::uintptr_t>(&destination);
    commands[threadRegisterIndex * 3 + 2] = 0;
    commands.insert(commands.end(), {0xc0033700u, 0x00100200u, static_cast<std::uint32_t>(destinationAddress), static_cast<std::uint32_t>(destinationAddress >> 32u), 7});
    packet = Packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    sceAgcDriverSubmitAcb(0x20, &packet);
    ExpectFailure([] { AgcDriverWaitIdle_nid_postfix(); }, "must be nonzero");
    ExpectFailure([] { AgcDriverWaitIdle_nid_postfix(); }, "must be nonzero");
    ExpectFailure([] { AgcDriverShutdown_nid_postfix(); }, "must be nonzero");
    Require(destination == 0, "failed dispatch executed a subsequent memory write");
}

void MissingProgramRegistration(AgcDriver::VulkanDevice& device, bool complete = false) {
    alignas(256) std::array<std::uint32_t, 5> code{0x34020082u, 0x7e040204u, 0xe0701000u, 0x80000201u, 0xbf810000u};
    alignas(256) std::array<std::uint32_t, 2> output{};
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    const auto outputAddress = reinterpret_cast<std::uintptr_t>(output.data());
    std::array<std::uint32_t, 5> users{static_cast<std::uint32_t>(outputAddress), static_cast<std::uint32_t>((outputAddress >> 32u) & 0xffffu), sizeof(output), 0x31016facu, 7u};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{1, 1, 1}, 0, {false, false, false}, false, 1, {}};
    ShaderRecompiler::RecompileRequest request{{ShaderRecompiler::ShaderStage::Compute, address, code, 0, {}}, {32, 0, users, compute, {}, {}, {}}, device.ComputeTarget(32), {0, 0, 0, 128}};
    RunLazyPreparation(device, request, true);
    Require(output[0] == 7 && output[1] == 7, "deferred invocation did not execute both static ABIs");
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 7> registers{{{0x20c, 0}, {0x20d, 0}, {0x207, 1}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 10}}};
        ShaderSpecialRegs specials{};
    } header;
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = sizeof(code);
    header.shader.code = code.data();
    header.shader.sh_registers = header.registers.data() + (complete ? 0u : 2u);
    header.shader.num_sh_registers = header.registers.size() - (complete ? 0u : 2u);
    header.shader.specials = &header.specials;
    header.specials.dispatch_modifier = 0x8000;
    header.registers[0].value = static_cast<std::uint32_t>(address >> 8u);
    header.registers[1].value = static_cast<std::uint32_t>(address >> 40u);
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    for (const std::uint32_t threads : {1u, 2u, 1u, 2u}) {
        output.fill(0);
        users[4] += 1;
        std::vector<std::uint32_t> commands;
        for (const auto reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.offset == 0x207 ? threads : reg.value});
        commands.insert(commands.end(), {0xc0017600u, 0x20c, static_cast<std::uint32_t>(address >> 8u), 0xc0017600u, 0x20d, static_cast<std::uint32_t>(address >> 40u)});
        for (std::uint32_t index = 0; index < users.size(); ++index) commands.insert(commands.end(), {0xc0017600u, 0x240 + index, users[index]});
        commands.insert(commands.end(), {0xc0031500u, 1, 1, 1, 0x8041});
        Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
        sceAgcDriverSubmitAcb(0x20, &packet);
        AgcDriverWaitIdle_nid_postfix();
        AgcDriver::GuestMemory::FlushGpuWrites(outputAddress, sizeof(output));
        if (output[0] != users[4] || output[1] != (threads == 2 ? users[4] : 0)) {
            throw std::runtime_error("shader registered without a program address: threads=" + std::to_string(threads) + " value=" + std::to_string(users[4]) + " output=" + std::to_string(output[0]) + "," + std::to_string(output[1]));
        }
    }
}

void DeferredRegistration(bool completeUnsupported = false, bool submit = true) {
    alignas(256) std::array<std::uint32_t, 2> code{0xbe842104u, 0xbf810000u};
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 8> registers{};
        ShaderSpecialRegs specials{};
    } header;
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = sizeof(code);
    header.shader.code = code.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.specials = &header.specials;
    header.specials.dispatch_modifier = 0x8000;
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, 1}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 12}, {0x207, 1}}};
    if (completeUnsupported) code[0] = 0xbe802204u;
    else {
        code[0] = 0xbe852104u;
        ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "unsupported scalar call link register");
        code[0] = 0xbe842104u;
    }
    if (completeUnsupported && !submit) RegisterWhileGpuLocked(&header.shader);
    else AgcDriverRegisterShader_nid_postfix(&header.shader);
    const auto expected = completeUnsupported ? "s_rfe_b64 at pc 0 returns from a trap handler" : "not statically resolvable";
    if (completeUnsupported) ExpectFailure([&] { AgcDriverResolveShaderAbi_nid_postfix(&header.shader, {}, {}); }, expected);
    if (!submit) return;
    std::vector<std::uint32_t> commands;
    for (const auto reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
    commands.insert(commands.end(), {0xc0031500u, 1, 1, 1, 0x8041});
    std::uint32_t destination = 0;
    const auto destinationAddress = reinterpret_cast<std::uintptr_t>(&destination);
    commands.insert(commands.end(), {0xc0033700u, 0x00100200u, static_cast<std::uint32_t>(destinationAddress), static_cast<std::uint32_t>(destinationAddress >> 32u), 7});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    sceAgcDriverSubmitAcb(0x20, &packet);
    ExpectFailure([&] { AgcDriverWaitIdle_nid_postfix(); }, expected);
    ExpectFailure([&] { AgcDriverShutdown_nid_postfix(); }, expected);
    Require(destination == 0, "unsupported shader dispatch executed a subsequent memory write");
}

}

int main(int argc, char** argv) {
    try {
        Require(argc == 1 || (argc == 2 && (std::string_view(argv[1]) == "--indirect" || std::string_view(argv[1]) == "--fail-before-registration" || std::string_view(argv[1]) == "--deferred" || std::string_view(argv[1]) == "--lazy-static" || std::string_view(argv[1]) == "--gpu-controlled-srt" || std::string_view(argv[1]) == "--gpu-controlled-invalid")), "invalid test arguments");
        if (argc == 2 && (std::string_view(argv[1]) == "--deferred" || std::string_view(argv[1]) == "--lazy-static")) {
            const bool complete = std::string_view(argv[1]) == "--lazy-static";
            auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
            if (complete) DeferredRegistration(true, false);
            MissingProgramRegistration(*device, complete);
            device.reset();
            DeferredRegistration(complete);
            std::cout << "deferred shader preparation tests passed\n";
            return 0;
        }
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (argc == 2 && (std::string_view(argv[1]) == "--gpu-controlled-srt" || std::string_view(argv[1]) == "--gpu-controlled-invalid")) {
            RunGpuControlledScalarReads(*device, std::string_view(argv[1]) == "--gpu-controlled-invalid");
            return 0;
        }
        RunGpuControlledScalarReads(*device);
        RunInactiveSrtReads(*device, false);
        RunInactiveSrtReads(*device, true);
        Run(*device);
        Require(argc != 2 || std::string_view(argv[1]) != "--fail-before-registration", "injected failure before registration");
        PrepareMultisampledStorage(*device);
        device.reset();
        GraphicsHeaderCopies(false);
        GraphicsHeaderCopies(true);
        Registration(argc == 2);
        std::cout << "prepared shader and transactional registration tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception& shutdownError) {
            std::cerr << shutdownError.what() << '\n';
        }
        return 1;
    }
}
