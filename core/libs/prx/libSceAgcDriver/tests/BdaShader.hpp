#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BDASHADER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BDASHADER_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstdint>
#include <span>
#include <vector>

std::vector<std::uint32_t> MakeBdaTestShader(std::uint64_t address, std::uint32_t bits, std::int64_t offset = 0);
std::vector<std::uint32_t> MakeBdaDwordReadTestShader(std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops);
struct BdaTestStep {
    enum Kind { Read, Store, AtomicAdd } kind;
    std::uint64_t address;
    std::uint32_t value = 0;
};
inline constexpr std::uint32_t BdaTestStepPc = 0x2000;
// One invocation runs the 32-bit steps in order; step i reports faults as instruction BdaTestStepPc + i
// and puts its value (the store's operand, the atomic's old value) in output word i.
std::vector<std::uint32_t> MakeBdaSequenceTestShader(std::span<const BdaTestStep> steps, bool rangeCache, bool stops);
void RunBdaExecutionTests(const AgcDriver::Graphics::Context& context);
void RunBdaContractTests();

#endif
