#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include <cstdlib>

namespace AgcDriver::DriverDetail {

bool Driver::drawAheadEnabled() {
    static const bool enabled = std::getenv("APS5_NO_DRAW_AHEAD") == nullptr && std::getenv("APS5_LOCKED_DRAW_PREPARE") == nullptr && std::getenv("APS5_DUMP_DRAW_SHADERS") == nullptr && std::getenv("APS5_DUMP_DRAW_SLOT1") == nullptr && drawEntries() && registerKeyEnabled() && !validateLegacy();
    return enabled && !ShaderRecompiler::DebugProbeActive();
}

bool Driver::drawLockRunEnabled() {
    static const bool enabled = std::getenv("APS5_NO_DRAW_LOCK_RUN") == nullptr && std::getenv("APS5_LOCKED_DRAW_PREPARE") == nullptr && std::getenv("APS5_DRAIN_ALL") == nullptr;
    return enabled;
}

DrawLockRun& Driver::drawLockRun() {
    static thread_local DrawLockRun run;
    return run;
}

void Driver::endDrawLockRun(DrawLockRunCount why) {
    auto& run = drawLockRun();
    if (!run.lock.owns_lock()) return;
    run.lock.unlock();
    drawLockRunCounts[why].fetch_add(1, std::memory_order_relaxed);
}

void Driver::keepDrawLockRun() {
    auto& run = drawLockRun();
    if (!run.enabled) return;
    if (run.lock.owns_lock()) {
        ++run.draws;
        drawLockRunCounts[RunDrawsHeld].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    run.lock.lock();
    run.draws = 1;
    run.since = std::chrono::steady_clock::now();
    drawLockRunCounts[RunsStarted].fetch_add(1, std::memory_order_relaxed);
}

DrawAheadTotals Driver::DrawAheadCounts() {
    return {drawAhead.counts[AheadPrepared].load(std::memory_order_relaxed), drawAhead.counts[AheadHits].load(std::memory_order_relaxed)};
}

void Driver::beginDrawAhead() {
    cancelDrawAhead();
    drawAhead.stopped.store(0, std::memory_order_release);
}

void Driver::cancelDrawAhead() {
    auto& ahead = drawAhead;
    std::unique_lock lock(ahead.mutex);
    ahead.cancel.store(true, std::memory_order_release);
    ahead.wake.notify_all();
    ahead.idle.wait(lock, [&] { return !ahead.busy; });
    ahead.cancel.store(false, std::memory_order_release);
    ahead.stopped.store(DrawAhead::Running, std::memory_order_release);
}

void Driver::stopDrawAhead() {
    auto& ahead = drawAhead;
    {
        std::lock_guard lock(ahead.mutex);
        ahead.stop = true;
        ahead.cancel.store(true, std::memory_order_release);
    }
    ahead.wake.notify_all();
    if (ahead.thread.joinable()) ahead.thread.join();
}

void Driver::drawAheadAt(const Submission& submission, const QueueState& queue, std::size_t cursor) {
    auto& ahead = drawAhead;
    const auto draws = ahead.draws.fetch_add(1, std::memory_order_relaxed) + 1;
    if (ahead.stopped.load(std::memory_order_acquire) > cursor) return;
    std::unique_lock lock(ahead.mutex);
    ahead.idle.wait(lock, [&] { return !ahead.busy; });
    if (ahead.stop) return;
    if (!ahead.thread.joinable()) ahead.thread = std::thread([this] { runDrawAhead(); });
    ahead.state.shader = queue.shader;
    ahead.state.context = queue.context;
    ahead.state.userConfig = queue.userConfig;
    ahead.state.savedContext = queue.savedContext;
    ahead.state.indexBase = queue.indexBase;
    ahead.state.drawIndirectBase = queue.drawIndirectBase;
    ahead.state.indexBufferSize = queue.indexBufferSize;
    ahead.state.indexType = queue.indexType;
    ahead.state.instanceCount = queue.instanceCount;
    ahead.state.predication = queue.predication;
    ahead.submission = &submission;
    ahead.start = cursor;
    ahead.firstDraw = draws;
    ahead.stopped.store(DrawAhead::Running, std::memory_order_release);
    ahead.busy = true;
    ahead.Count(AheadScans);
    lock.unlock();
    ahead.wake.notify_one();
}

void Driver::runDrawAhead() noexcept {
    PinWorkerThread("draw look-ahead");
    GuestMemory::LeaveGpuWritesPending();
    auto& ahead = drawAhead;
    std::unique_lock lock(ahead.mutex);
    for (;;) {
        ahead.wake.wait(lock, [&] { return ahead.stop || ahead.busy; });
        if (ahead.stop) {
            ahead.busy = false;
            ahead.idle.notify_all();
            return;
        }
        const auto& submission = *ahead.submission;
        const auto start = ahead.start;
        const auto firstDraw = ahead.firstDraw;
        lock.unlock();
        auto stopped = submission.commands.size();
        try {
            stopped = scanDrawsAhead(submission, start, firstDraw);
        } catch (...) {
            ahead.Count(AheadErrors);
        }
        lock.lock();
        ahead.stopped.store(stopped, std::memory_order_release);
        ahead.busy = false;
        ahead.idle.notify_all();
    }
}

std::size_t Driver::scanDrawsAhead(const Submission& submission, std::size_t start, std::uint64_t firstDraw) {
    auto& ahead = drawAhead;
    auto& queue = ahead.state;
    const auto& commands = submission.commands;
    std::uint64_t seen = firstDraw;
    for (std::size_t cursor = start + Pm4::PacketWords(commands[start]); cursor < commands.size();) {
        if (ahead.cancel.load(std::memory_order_acquire)) return cursor;
        const auto header = commands[cursor];
        if (Pm4::FillerPacket(header)) {
            ++cursor;
            continue;
        }
        const auto count = Pm4::PacketWords(header);
        const auto packet = std::span(commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        if (Pm4::Predicated(header) && queue.predication.operation != 0) {
            ahead.Count(AheadStops);
            return cursor;
        }
        if (Pm4::DrawOpcode(opcode)) {
            ++seen;
            while (seen > ahead.draws.load(std::memory_order_relaxed) + DrawAhead::Window) {
                std::unique_lock lock(ahead.mutex);
                if (ahead.wake.wait_for(lock, std::chrono::microseconds(50), [&] { return ahead.cancel.load(std::memory_order_acquire); })) return cursor;
            }
            if (seen > ahead.draws.load(std::memory_order_relaxed)) {
                try {
                    prepareDrawAhead(queue, packet, submission);
                } catch (const std::exception&) {
                    ahead.Count(AheadErrors);
                }
            } else {
                ahead.Count(AheadPassed);
            }
            cursor += count;
            continue;
        }
        switch (opcode) {
            case 0x11: case 0x12: case 0x13: case 0x20: case 0x26: case 0x2a: case 0x2f: case 0x69: case 0x76: case 0x79: case 0x7a:
                Pm4::Execute(packet, queue);
                break;
            case 0x63: case 0x64: case 0x9f:
                Pm4::ExecuteIndirectRegisters(packet, submission.registerLists.at(cursor), queue);
                break;
            case 0x10: {
                const auto kind = (header >> 2u) & 0x3fu;
                if (kind == 0x1a) Pm4::Execute(packet, queue);
                else if (kind != 0 && kind != 0x0b && kind != 0x0c) {
                    ahead.Count(AheadStops);
                    return cursor;
                }
                break;
            }
            case 0x15: case 0x16: case 0x37: case 0x3c: case 0x3f: case 0x40: case 0x42: case 0x45: case 0x46: case 0x49: case 0x50: case 0x58: case 0x81: case 0x83: case 0x93:
                break;
            default:
                ahead.Count(AheadStops);
                return cursor;
        }
        cursor += count;
    }
    return commands.size();
}

void Driver::prepareDrawAhead(const QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission) {
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;
    const auto skip = [&](DrawAheadCount kind) { drawAhead.Count(kind); };
    const auto drawParameters = Pm4::ResolveDraw(packet, queue);
    std::string rejected;
    bool traceIndirect = false;
    const auto localDevice = device.Load();
    const auto unsupported = [&](DrawAheadCount why) {
        skip(why);
        skip(AheadUnsupported);
    };
    if (drawParameters.indirect) return unsupported(AheadIndirect);
    if (localDevice == nullptr) return unsupported(AheadNoDevice);
    if (!drawParameters.indexed && (drawParameters.indexCount == 0 || drawParameters.instanceCount == 0)) return unsupported(AheadEmpty);
    if (Graphics::DecodeColorMetadataPass(queue)) return unsupported(AheadMetadataPass);
    if (const auto verdict = precheckDraw(queue, submission, packet, drawParameters, rejected, traceIndirect, true)) return unsupported(*verdict == DrawVerdict::Rejected ? AheadRejected : AheadNothing);
    const auto drawKey = drawKeyFor(queue, *submission.shaders, localDevice->Serial(), drawParameters);
    std::shared_ptr<DrawEntry> entry;
    {
        std::lock_guard cacheLock(drawCacheMutex);
        const auto found = drawCache.find(drawKey);
        if (found != drawCache.end()) entry = found->second;
    }
    std::shared_ptr<const DrawDecode> decode = entry != nullptr ? entry->decode : nullptr;
    if (decode != nullptr && !DecodeProgramsCurrent(*decode, *submission.shaders)) {
        entry = nullptr;
        decode = nullptr;
    }
    if (decode == nullptr) decode = decodeDraw(queue, submission);
    const auto& graphics = decode->state;
    const auto& programs = decode->programs;
    const auto& roles = decode->roles;
    if (graphics.stages.mesh) return unsupported(AheadMesh);
    if (entry != nullptr && entry->stages.size() == programs.size()) {
        bool warm = true;
        std::vector<ShaderRecompiler::MemoryRegion> regions;
        for (std::size_t i = 0; warm && i < programs.size(); ++i) {
            if (roles[i] == Role::GeometryBack) continue;
            warm = std::any_of(entry->stages[i].begin(), entry->stages[i].end(), [&](const std::shared_ptr<DispatchVariant>& variant) {
                regions.clear();
                appendEntryRegions(*variant, regions);
                return std::all_of(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& region) { return GuestMemory::CompareMapped(region.guestAddress, region.bytes) == GuestMemory::Compare::Equal; });
            });
        }
        if (warm) return skip(AheadWarm);
    }
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>> vertexInfos(programs.size());
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads(programs.size());
    for (std::size_t i = 0; i < programs.size(); ++i) {
        const auto& program = programs[i];
        for (const auto& region : program.memory) {
            if (!region.bytes.empty()) memory.push_back(region);
        }
        linked.push_back({roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
        if (program.binary.stage != Stage::Fragment && roles[i] != Role::GeometryBack) vertexInfos[i] = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, &decodeReads[i], false, 8u - program.firstUserSgpr);
    }
    ShaderMemory shaderMemory(memory, &queryPendingWrite);
    std::vector<StageCapture> stageCaptures(programs.size());
    std::vector<std::shared_ptr<DispatchVariant>> matched(programs.size());
    std::vector<std::shared_ptr<DispatchVariant>> fresh(programs.size());
    std::vector<std::vector<ShaderRecompiler::MemoryRegion>> matchedRegions(programs.size());
    std::vector<bool> recompiled(programs.size(), false);
    std::array<double, DrawDriverPhaseCount> phaseMs{};
    std::chrono::steady_clock::time_point phaseLap{};
    DrawPhaseTiming phaseTiming{false, phaseMs, phaseLap};
    std::uint64_t captures = 0;
    std::uint32_t pushCursorBytes = 0;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (roles[i] == Role::GeometryBack) continue;
        pushCursorBytes = Graphics::StagePushOffset(pushCursorBytes, programs[i].binary.stage, localDevice->GraphicsPipelineLibraries());
        auto pixel = decode->pixel;
        if (programs[i].binary.stage == Stage::Fragment) {
            for (auto previous = i; previous-- > 0;) {
                if (stageCaptures[previous].compiled == nullptr) continue;
                pixel = Graphics::DefaultUnexportedInputs(decode->pixel, stageCaptures[previous].compiled->parameterExports);
                break;
            }
        }
        static_cast<void>(materializeDrawStage(i, pushCursorBytes, queue, submission, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, shaderMemory, stageCaptures, recompiled, false, matched, matchedRegions, false, 0, 0, captures, phaseTiming, phaseMs, rejected));
        if (!rejected.empty()) return unsupported(AheadStageRejected);
        const auto pushBytes = stageCaptures[i].compiled->pushConstants.size();
        const auto slotEnd = (pushCursorBytes / Graphics::PipelinePushSlotBytes + 1u) * Graphics::PipelinePushSlotBytes;
        require(pushBytes <= slotEnd - pushCursorBytes, "stage push constants exceed the pipeline push constant block");
        pushCursorBytes += static_cast<std::uint32_t>(pushBytes);
    }
    static const bool overPending = std::getenv("APS5_NO_DRAW_AHEAD_PENDING") == nullptr;
    PendingView pending;
    pending.Load();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    auto worst = ShaderMemory::PendingWrite::None;
    bool vertexTables = false;
    const auto read = [&](std::uint64_t address, std::size_t bytes, bool vertex) {
        using Policy = ShaderMemory::PendingWrite;
        runs.emplace_back(address, address + bytes);
        if (!pending.Overlaps(address, bytes)) return;
        std::uint64_t ValidateCounters::*reason = nullptr;
        const auto policy = classifyPendingWrite(address, bytes, reason, pending);
        if (policy == Policy::Sync || policy == Policy::None) worst = Policy::Sync;
        else if (worst != Policy::Sync && (policy == Policy::Raw || policy == Policy::VerifyRaw)) worst = Policy::Raw;
        else if (worst == Policy::None) worst = Policy::KnownValue;
        vertexTables = vertexTables || vertex;
    };
    for (std::size_t i = 0; i < programs.size(); ++i) {
        for (const auto& region : stageCaptures[i].regions) read(region.guestAddress, region.bytes.size(), false);
        for (const auto& decoded : decodeReads[i]) read(decoded.address, decoded.bytes.size(), true);
    }
    const bool images = Graphics::StorageTexture::AnyPendingOverlaps(runs) || Graphics::AnyShadowedOverlaps(runs);
    if (worst != ShaderMemory::PendingWrite::None || images) {
        if (!overPending) return skip(AheadPending);
        skip(images ? AheadOverImage : worst == ShaderMemory::PendingWrite::Sync ? AheadOverSync : worst == ShaderMemory::PendingWrite::Raw ? AheadOverRaw : AheadOverKnown);
        if (vertexTables) skip(AheadOverVertex);
    }
    cacheDrawStages(true, false, drawParameters, std::nullopt, programs, stageCaptures, vertexInfos, decodeReads, false, matched, fresh, drawKey, true, decode, phaseTiming);
    for (const auto& variant : fresh) {
        if (variant != nullptr) variant->ahead.fetch_add(1, std::memory_order_relaxed);
    }
    skip(AheadPrepared);
}

}

AgcDriver::DrawAheadTotals AgcDriver::DrawAheadCounts() {
    return DriverDetail::Driver::Get().DrawAheadCounts();
}
