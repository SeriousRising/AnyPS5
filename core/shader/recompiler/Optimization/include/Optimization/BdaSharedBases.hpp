#ifndef CORE_SHADER_RECOMPILER_OPTIMIZATION_INCLUDE_OPTIMIZATION_BDASHAREDBASES_HPP
#define CORE_SHADER_RECOMPILER_OPTIMIZATION_INCLUDE_OPTIMIZATION_BDASHAREDBASES_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <compare>
#include <cstdint>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

// What the base address of an address-based dword read is computed from: the address pair's dwords
// (pairLow is null for a full address, whose high dword is pairHigh) and the low dword, and whether
// the low dword and the immediate lose their low bits; `residue` is the immediates' value modulo 4.
struct BdaBaseKey {
    const IrValue* pairLow = nullptr;
    const IrValue* pairHigh = nullptr;
    const IrValue* low = nullptr;
    bool scalar = false;
    std::uint32_t residue = 0;
    auto operator<=>(const BdaBaseKey&) const = default;
};

// The address-based dword reads (one to four dwords) of one base, in program order. A read's number
// is the distance in bytes from the base address to its first extracted dword; `tries` marks the
// first read of the base in each block under each activity condition. The hull is [first, end).
struct BdaBaseGroup {
    BdaBaseKey key;
    std::vector<std::pair<const IrValue*, std::int64_t>> reads;
    std::vector<bool> tries;
    std::int64_t first = 0;
    std::int64_t end = 0;
};

// The dwords of a wide load that the program extracts (all of them when the composite is used whole).
std::uint32_t UsedBdaDwords(const IrValue& inst, std::uint32_t dwords);
// The first and last extracted dword of a read; first > last when none is.
std::pair<std::uint32_t, std::uint32_t> ExtractedBdaDwords(const IrValue& inst, std::uint32_t dwords, bool everyDword);
// Every base of the program with its reads, in the order of each base's first read.
std::vector<BdaBaseGroup> CollectBdaBaseGroups(const IrProgram& program);
// Whether the CPU translates base addresses that are the same for a whole draw or dispatch
// (APS5_NO_BDA_CPU_BASE=1: no).
bool BdaCpuBases();
// Records in the program's resource plan the shared bases (two reads or more) that the resource walk
// can evaluate for a draw without reading guest memory it would not read anyway: see IrBdaBase.
void PlanBdaBases(IrProgram& program);

}

#endif
