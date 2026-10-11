#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDADIRECTTABLE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDADIRECTTABLE_HPP

#include "BdaAbi.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

// The direct table of `ranges` (BdaAbi::DirectVersion), which must be sorted by begin: header,
// entries and the ranges, as dwords. A range answers from its begin to the next one's, as the
// search of the sorted ranges selects.
std::vector<std::uint32_t> BdaDirectTable(std::span<const ShaderRecompiler::BdaAbi::Range> ranges);

}

#endif
