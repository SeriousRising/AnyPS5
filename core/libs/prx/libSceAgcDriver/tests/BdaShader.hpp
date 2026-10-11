#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BDASHADER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BDASHADER_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstdint>
#include <span>
#include <vector>

// A stopping shader made with `deferred` flags a faulting read and runs on without effects (see
// BdaStopCanBeDeferred); without it the invocation ends at the read. Both must give the same results.
std::vector<std::uint32_t> MakeBdaTestShader(std::uint64_t address, std::uint32_t bits, std::int64_t offset = 0, bool deferred = true);
std::vector<std::uint32_t> MakeBdaDwordReadTestShader(std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops, bool deferred = true);
struct BdaTestStep {
    enum Kind { Read, Store, AtomicAdd } kind;
    std::uint64_t address;
    std::uint32_t value = 0;
};
inline constexpr std::uint32_t BdaTestStepPc = 0x2000;
// One invocation runs the 32-bit steps in order; step i reports faults as instruction BdaTestStepPc + i
// and puts its value (the store's operand, the atomic's old value) in output word i. The lookup
// keeps `rangeCacheEntries` ranges (0 searches every time) and, with `rangeCacheCheck`, compares them
// with a search at every lookup.
std::vector<std::uint32_t> MakeBdaSequenceTestShader(std::span<const BdaTestStep> steps, std::uint32_t rangeCacheEntries, bool stops, bool rangeCacheCheck = false, bool deferred = true);
void RunBdaExecutionTests(const AgcDriver::Graphics::Context& context);
void RunBdaContractTests();

#endif
