#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_DENORMALFLUSHELIMINATOR_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_DENORMALFLUSHELIMINATOR_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <cstdint>

namespace ShaderRecompiler {

struct DenormalFlushEliminationStats {
    std::uint32_t removedFlushes = 0;
};

class DenormalFlushEliminator {
public:
    [[nodiscard]] DenormalFlushEliminationStats Eliminate(IrProgram& program) const;
    // Fast float mode (SpirvTarget::fastFloat32, an accuracy trade): every exact f32 flush select and
    // every select a NaN test guards (the guest's NaN choice) gives way to its plain operand; the
    // module flushes f32 denormals in hardware instead.
    [[nodiscard]] DenormalFlushEliminationStats EliminateAll(IrProgram& program) const;
};

}

#endif
