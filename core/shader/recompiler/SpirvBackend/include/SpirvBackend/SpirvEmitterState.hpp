#ifndef CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVEMITTERSTATE_HPP
#define CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVEMITTERSTATE_HPP

#include "SpirvBackend/SpirvModule.hpp"
#include "SpirvBackend/SpirvAnalysis.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

inline constexpr std::uint32_t NoImageComponent = 0xffffffffu;

struct SpirvInputBinding : StageInput {
    std::uint32_t variableId = 0;
};

struct SpirvOutputBinding : StageOutput {
    std::uint32_t variableId = 0;
    std::uint32_t meshDataVariable = 0;
};

struct RdnaImageDimensionInfo {
    RdnaImageDimension dimension;
    std::uint32_t spirvDimension;
    std::uint32_t coordinateComponents;
    std::uint32_t spatialComponents;
    std::uint32_t arrayed;
    std::uint32_t multisampled;
};

enum class VertexInputScalarKind { Float, Sint, Uint };

struct DppTargetLane {
    std::uint32_t lane = 0;
    std::uint32_t valid = 0;
};

struct ImageSampleLayout {
    std::uint32_t offset = NoImageComponent;
    std::uint32_t dref = NoImageComponent;
    std::uint32_t bias = NoImageComponent;
    std::uint32_t coord = 0;
    std::uint32_t lod = NoImageComponent;
    std::uint32_t clamp = NoImageComponent;
    std::uint32_t gradX = NoImageComponent;
    std::uint32_t gradY = NoImageComponent;
};

struct F32Class {
    std::uint32_t bits = 0;
    std::uint32_t nan = 0;
    std::uint32_t zero = 0;
};

struct MemoryResourceAccess {
    ResourceKind kind = ResourceKind::None;
    std::uint32_t objectPointer = 0;
    std::uint32_t length = 0;
    std::uint32_t indexOffset = 0;
    std::uint32_t byteOffset = 0;
    bool addIndexOffset = false;
    std::uint32_t memoryAccess = 0;
    std::uint32_t misalignment = 0;
};

// The ranges lookup_bda keeps unless a switch says otherwise: none, because RADV takes more memory to
// compile a program whose lookups keep ranges (see the range cache's technical debt entry).
inline constexpr std::uint32_t BdaRangeCacheDefaultEntries = 0;

// A set of address-based dword reads whose addresses differ only by their immediates (see
// FindBdaSharedBases). `variable` is a Private u64: 0 until a member tries to translate the hull, the
// `bytes` bytes from the base address plus `first`; then the hull's device address, or 1 when the
// hull does not lie in one readable range at a 4-aligned device address.
// With `cpuSlot` the CPU translates the base for each draw (IrResourcePlan::bdaBases[cpuSlot], whose
// hull holds this one `cpuBack` bytes from its start), and the key's values give the base address.
inline constexpr std::uint32_t NoBdaCpuSlot = 0xffffffffu;

struct BdaSharedBase {
    std::uint32_t variable = 0;
    std::int64_t first = 0;
    std::uint32_t bytes = 0;
    std::uint32_t cpuSlot = NoBdaCpuSlot;
    std::uint32_t cpuBack = 0;
    const IrValue* pairLow = nullptr;
    const IrValue* pairHigh = nullptr;
    const IrValue* low = nullptr;
    bool scalar = false;
};

// What a block knows of a base the CPU translated: whether the draw's slot may be used (a bool) and
// the device address of the slot's hull.
struct BdaCpuBaseCheck {
    std::uint32_t usable = 0;
    std::uint32_t device = 0;
};

// A member of bdaSharedBases[base]: its first extracted dword lies `begin` bytes from the base address.
// The members that try the hull are the first of their base in each block for each activity
// condition: every helper call is inlined, and the call that tries a hull is the larger one.
struct BdaSharedRead {
    std::uint32_t base = 0;
    std::int64_t begin = 0;
    bool tries = false;
};

// A leaf of the direct table walked where a block first reads a base address that it reads more
// than once: the leaf of that base plus `immediate`, the address's `offset` in the leaf's unit and
// `limit`, the leaf's bytes less 3 (0 for fewer). A dword `distance` bytes from the walked address
// lies in the leaf's mapping when offset + distance, unsigned, is below the limit, and is then read
// with no walk of its own. `reads` counts the block's dword reads of the base; no leaf is walked
// ahead for one.
struct BdaDirectHint {
    std::uint32_t reads = 0;
    bool walked = false;
    std::uint32_t delta = 0;
    std::uint32_t bytes = 0;
    std::uint32_t flags = 0;
    std::uint32_t limit = 0;
    std::uint32_t offset = 0;
    std::int32_t immediate = 0;
};

// The values a read's base address is computed from (the address pair or the handle's, and the low
// dword), and whether the low dword is rounded down to a multiple of 4.
using BdaDirectKey = std::tuple<const IrValue*, const IrValue*, const IrValue*, bool>;

struct SpirvEmitterState {
    const ImageResource* runtimeImage = nullptr;
    std::uint32_t runtimeImageMetadata = 0u;
    SpirvEmitterState(const IrProgram& program, const ShaderStageInputInfo& inputInfo);

    SpirvModule module;
    const IrProgram& program;
    ShaderStageInputInfo inputInfo;
    std::array<std::uint32_t, 6> tessVariables {};
    std::uint32_t tessInnerVariable = 0;
    std::uint32_t tessPatchBase = 0;

    SpirvRequirements requirements;
    std::uint32_t laneCount = 1;
    std::uint32_t hostSubgroupSize = 0;
    bool splitSubgroup = false;
    bool narrowSubgroupClock = false;
    std::unordered_set<const IrValue*> sharedLaneValues;
    std::uint32_t laneHalf = 0;
    // The target's SPIR-V version and what the device accepts, for capabilities an emitter adds
    // only when needed (bindless image tables: see TableImageIndex).
    std::uint32_t spirvVersion = 0x00010300u;
    std::span<const std::uint32_t> supportedCapabilities;
    std::span<const std::string_view> supportedExtensions;
    bool nonConstantImageOffsets = false;
    // A bindless table's runtime slot is wave-uniform; it is uniform over the invocation group
    // only for a single-wave compute workgroup, elsewhere it needs the NonUniform decoration.
    bool tableIndexNonUniform = true;
    std::uint32_t storageBufferVariable = 0;
    std::uint32_t storageBufferU64Variable = 0;
    std::vector<std::uint32_t> memoryByteOffsets;
    std::uint32_t bdaPagetableVariable = 0;
    std::uint32_t faultBufferVariable = 0;
    std::uint32_t bdaLookupFunction = 0;
    // Private: not zero when the page table's header is invalid (set by PrepareBdaLookups).
    std::uint32_t bdaTableInvalid = 0;
    // The program is made for the direct table (BdaAbi::DirectVersion): a lookup walks it instead of
    // searching the ranges, and neither keeps ranges nor shares base addresses.
    bool bdaDirect = false;
    // Private, set by PrepareBdaLookups from the direct table's header: its base (u64), root shift,
    // (1 << root shift) - 1 and root count, the count 0 when the header is invalid.
    std::uint32_t bdaDirectBase = 0;
    std::uint32_t bdaDirectShift = 0;
    std::uint32_t bdaDirectMask = 0;
    std::uint32_t bdaDirectCount = 0;
    // How many ranges lookup_bda keeps for its invocation (see DefineBdaLookupCore); 0 searches at
    // every lookup. APS5_BDA_RANGE_CACHE_ENTRIES=<1 to 4> sets it, APS5_NO_BDA_RANGE_CACHE=1 clears it.
    std::uint32_t bdaRangeCacheEntries = BdaRangeCacheDefaultEntries;
    // Debug aid (APS5_BDA_RANGE_CACHE_CHECK=1): lookup_bda searches at every lookup as well and reports
    // a kept range that differs from the one the search selects (BdaAbi::FaultReason::RangeCacheMismatch).
    bool bdaRangeCacheCheck = false;
    // The cache's Private variables (64-bit, then 32-bit); empty without the cache.
    std::vector<std::uint32_t> bdaRangeCache64;
    std::vector<std::uint32_t> bdaRangeCache32;
    std::uint32_t bdaPointerFunction = 0;
    // The lookup without fault recording that wide reads try first (see EmitBdaDwordReads); 0 when
    // every read takes the byte path.
    std::uint32_t bdaProbeFunction = 0;
    std::uint32_t bdaWritePointerFunction = 0;
    std::uint32_t bdaAtomicPointerFunction = 0;
    std::uint32_t bdaNoteWriteFunction = 0;
    std::uint32_t bdaFaultFunction = 0;
    // In a program for the direct table record_bda_fault keeps the invocation's first fault in these
    // Private variables (reason, address, bytes, instruction; no fault: reason 0), and
    // publish_bda_fault writes it to the fault buffer where the invocation ends.
    std::array<std::uint32_t, 4> bdaPendingFault {};
    std::uint32_t bdaPublishFunction = 0;
    std::array<std::array<std::uint32_t, 2>, 2> bdaDwordReadFunctions {};
    std::array<std::array<std::uint32_t, 2>, 2> bdaProbedDwordReadFunctions {};
    // The direct table's dword reads, by what a faulting one does (nothing, returns a flag, sets the
    // pending stop) and by coherence; see DefineBdaDirectDwordRead.
    std::array<std::array<std::uint32_t, 2>, 3> bdaDirectDwordReadFunctions {};
    // The leaves walked ahead in the block being emitted, by base (see BdaDirectHint).
    std::map<BdaDirectKey, BdaDirectHint> bdaDirectHints;
    // Set when reads share translations (FindBdaSharedBases, off with APS5_NO_BDA_SHARED_BASE=1):
    // lookup_bda_span and the two functions below, which try a hull before the read's own bytes,
    // are then defined.
    bool bdaSharedBase = false;
    std::uint32_t bdaSpanLookupFunction = 0;
    std::vector<BdaSharedBase> bdaSharedBases;
    std::unordered_map<const IrValue*, BdaSharedRead> bdaSharedReads;
    // The bases to forget where a block starts: those whose address is computed from a value the block defines.
    std::unordered_map<const IrBlock*, std::vector<std::uint32_t>> bdaSharedBaseResets;
    std::array<std::array<std::uint32_t, 2>, 2> bdaSharedDwordReadFunctions {};
    std::uint32_t bdaSharedProbeFunction = 0;
    // The directory word of the program's stage (BdaAbi::BaseDirectoryWord), loaded where the program
    // starts; 0 in a program with no base the CPU translates.
    std::uint32_t bdaBaseDirectory = 0;
    // The slots the block being emitted has already compared, by shared base.
    std::unordered_map<std::uint32_t, BdaCpuBaseCheck> bdaCpuBaseChecks;
    std::uint32_t bdaStopValue = 0;
    // False for programs with workgroup barriers: faulting BDA accesses then continue (see BdaInvocationsMayStop).
    bool bdaStopsInvocations = true;
    // A faulting read sets bdaStopped and its invocation runs on without effects instead of ending at
    // the read (see BdaStopCanBeDeferred). The flow emitter clears it for the blocks of guest loops,
    // where a faulting read still ends the invocation at once.
    bool bdaStopDeferred = false;
    // Private bool, true once the invocation met a fault that ends it; 0 in programs that defer no stop.
    std::uint32_t bdaStopped = 0;
    // Execution scope of the barriers that keep one guest wave's LDS accesses in program order across
    // host invocations (see WaveLdsScope); 0 when none are emitted.
    std::uint32_t waveLdsScope = 0;
    // Debug aid (APS5_LOOP_GUARD=<exits>): after that many evaluated loop exits an invocation leaves its
    // loops, and the first loop that ran out is reported as a LoopLimit fault instead of hanging the GPU.
    std::uint32_t loopGuardLimit = 0;
    std::uint32_t loopGuardVisits = 0;
    std::uint32_t loopGuardPc = 0;
    std::uint32_t srgbTableVariable = 0;
    std::uint32_t gdsVariable = 0;
    std::uint32_t gdsLength = 0;
    std::uint32_t pushConstantVariable = 0;
    std::uint32_t shaderDataStorageVariable = 0;
    std::uint32_t flattenedSrtVariable = 0;
    std::uint32_t ldsVariable = 0;
    std::array<std::uint32_t, 2> scratchVariable {};
    std::array<std::uint32_t, ImageBindingCount> imageVariables {};
    std::uint32_t samplerVariable = 0;
    std::uint32_t mainFunc = 0;
    std::uint32_t meshGuestFunc = 0;
    std::uint32_t meshAllocation = 0;
    std::uint32_t meshPrimitiveData = 0;
    std::uint32_t meshPrimitives = 0;
    std::uint32_t meshCull = 0;
    std::uint32_t entryLabel = 0;
    std::uint32_t currentLabel = 0;
    const IrBlock* currentBlock = nullptr;
    std::uint32_t pixelValidMaskVariable = 0;
    std::uint32_t subgroupLocalInvocationIdVariable = 0;
    std::uint32_t perVertexVariable = 0;
    std::uint32_t pointSizeVariable = 0;
    std::uint32_t clipDistanceVariable = 0;
    std::uint32_t cullDistanceVariable = 0;
    std::uint32_t layerVariable = 0;
    std::uint32_t viewportIndexVariable = 0;
    std::uint32_t depthVariable = 0;
    std::uint32_t sampleMaskVariable = 0;
    std::vector<SpirvInputBinding> inputs;
    std::vector<SpirvOutputBinding> outputs;
    std::vector<std::uint32_t> interfaceVariables;
    std::unordered_map<const IrBlock*, std::uint32_t> labels;
};

struct SpirvValueEmitContext {
    explicit SpirvValueEmitContext(SpirvEmitterState& state);

    std::uint32_t Def(const IrValue* value);
    std::uint32_t Arg(const IrValue& inst, std::size_t index);
    std::uint32_t HalfArg(const IrValue& inst, std::size_t index, std::uint32_t half);
    std::uint32_t Ballot(const IrValue* predicate);
    std::uint32_t FirstLane(std::uint32_t ballot);
    std::uint32_t Shuffle(const IrValue& inst, std::size_t index, std::uint32_t lane);
    std::uint32_t Result(const IrValue& inst);
    std::uint32_t Define(const IrValue& inst, std::uint32_t value);
    std::uint32_t ResourceIndex(const IrValue* value, IrOpcode opcode);
    const IrValue* ImageAddress(const IrValue* value);
    const MemoryInfo& Memory(const IrValue& inst) const;
    const ExportInfo& Export(const IrValue& inst) const;
    std::uint32_t Label(const IrBlock* block) const;
    [[noreturn]] void Fail(const char* reason) const;
    [[noreturn]] void Fail(const IrValue& inst, const char* reason) const;

    SpirvEmitterState& state;
    std::unordered_map<const IrValue*, std::uint32_t> definitions;
    std::uint32_t scratchU32Variable = 0;
    SpirvValueEmitContext* otherHalf = nullptr;
    std::uint32_t half = 0;
};

struct SpirvDeferredPhiPatch {
    SpirvDeferredPhi phi;
    const IrValue* instruction = nullptr;
    std::uint32_t half = 0;
};

struct StructuredFunctionState {
    std::unordered_map<const IrBlock*, std::uint32_t> blockExitLabels;
    std::vector<SpirvDeferredPhiPatch> deferredPhis;
};

enum class SpirvFormatComponentType {
    Unknown,
    Uint,
    Sint,
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Float,
};

struct SpirvBufferFormatInfo {
    IrBufferFormat format = IrBufferFormat::Invalid;
    SpirvFormatComponentType type = SpirvFormatComponentType::Unknown;
    std::uint32_t componentCount = 0;
    std::uint32_t byteSize = 0;
    std::uint32_t componentBits[4] = {};
    std::uint32_t componentBitOffset[4] = {};
    bool packedBitfield = false;
};

}

#endif
