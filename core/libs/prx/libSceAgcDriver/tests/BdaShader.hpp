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
    // A read of this many dwords puts dword 0 + 3 * dword 1 + 5 * dword 2 + 7 * dword 3 in its output word.
    std::uint32_t dwords = 1;
};
inline constexpr std::uint32_t BdaTestStepPc = 0x2000;
// One invocation runs the 32-bit steps in order; step i reports faults as instruction BdaTestStepPc + i
// and puts its value (the store's operand, the atomic's old value) in output word i. The lookup
// keeps `rangeCacheEntries` ranges (0 searches every time) and, with `rangeCacheCheck`, compares them
// with a search at every lookup. With `sharedBytes` (a power of two, 0 for none) the reads whose
// addresses lie in the same block of that many bytes and agree modulo 4 share one base address when
// they are two or more (see EmitBdaSharedDwordReads); every other read looks its own bytes up.
std::vector<std::uint32_t> MakeBdaSequenceTestShader(std::span<const BdaTestStep> steps, std::uint32_t rangeCacheEntries, bool stops, bool rangeCacheCheck = false, bool deferred = true, std::uint64_t sharedBytes = 0);
void RunBdaExecutionTests(const AgcDriver::Graphics::Context& context);
void RunBdaContractTests();

#endif
