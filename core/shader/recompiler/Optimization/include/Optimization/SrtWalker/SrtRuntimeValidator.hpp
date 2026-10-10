#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTRUNTIMEVALIDATOR_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTRUNTIMEVALIDATOR_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/SrtWalker.hpp"

#include <unordered_map>
#include <unordered_set>

namespace ShaderRecompiler::Detail {

class RuntimeValidator {
public:
    RuntimeValidator(const IrResourcePlan& program, RuntimeValueType type) : _program(program), _type(type) {}

    bool Run(IrValue* value) { analyzeCycles(value); return Validate(value); }

private:
    void analyzeCycles(IrValue* value);
    bool ValidateArguments(IrValue& inst, bool requireUniform);
    bool Validate(IrValue* raw, bool requireUniform = true);

    const IrResourcePlan& _program;
    RuntimeValueType _type;
    IrValue* _activeMask = nullptr;
    std::unordered_set<IrValue*> _analyzedValues;
    std::unordered_set<IrValue*> _cyclicValues;
    std::unordered_set<IrValue*> _visiting;
    std::unordered_set<IrValue*> _validatedDependencies;
    std::unordered_map<IrValue*, std::unordered_set<IrValue*>> _validatedUniform;
};

}

#endif
