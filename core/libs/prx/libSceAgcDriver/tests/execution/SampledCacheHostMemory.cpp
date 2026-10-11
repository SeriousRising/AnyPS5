#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Side = 1024;
constexpr std::uint32_t Format32Float = 22;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint64_t TextureBytes = std::uint64_t{Side} * Side * 4u;
constexpr std::uint32_t Textures = 40;
constexpr std::uint64_t BudgetMiB = 64;
constexpr std::uint64_t Alignment = 0x10000;

alignas(256) std::array<std::uint32_t, Threads * 4> Buffer{};

alignas(256) constexpr std::array<std::uint32_t, 9> LoadCode{
    0x34020084, 0x7e3c0300, 0x7e3e0280, 0xf0001f08, 0x0001141e, 0xbf8c3f70, 0xe0781000, 0x80001401,
    0xbf810000,
};

std::array<std::uint32_t, 8> TextureDescriptor(std::uint64_t address) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32Float << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        0xfacu | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

void Sample(AgcDriver::VulkanDevice& device, std::uint64_t address, float expected, const std::string& what) {
    Buffer.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(16, 0u);
    const auto output = reinterpret_cast<std::uintptr_t>(Buffer.data());
    const std::array<std::uint32_t, 4> buffer{static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), static_cast<std::uint32_t>(Buffer.size() * 4u), 0x31016facu};
    const auto texture = TextureDescriptor(address);
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(LoadCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto actual = std::bit_cast<float>(Buffer[tid * 4u]);
        Require(actual == expected, what + ": thread " + std::to_string(tid) + " loaded " + std::to_string(actual) + ", expected " + std::to_string(expected));
    }
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_TEXTURE_CACHE_MIB", "64");
#else
    setenv("APS5_TEXTURE_CACHE_MIB", "64", 1);
#endif
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        std::vector<float> storage((Textures * TextureBytes + Alignment) / 4u, 0.0f);
        auto* texels = reinterpret_cast<float*>((reinterpret_cast<std::uintptr_t>(storage.data()) + Alignment - 1u) & ~std::uintptr_t{Alignment - 1u});
        for (std::uint32_t texture = 0; texture < Textures; ++texture) std::fill_n(texels + texture * (TextureBytes / 4u), Side, static_cast<float>(texture + 1u));
        const auto base = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texels));
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(texels, static_cast<std::size_t>(Textures * TextureBytes), true, true, true);
        }
        std::string failure;
        try {
            const auto budget = BudgetMiB << 20u;
            std::uint64_t mostHost = 0;
            for (std::uint32_t pass = 0; pass < 2u; ++pass) {
                for (std::uint32_t texture = 0; texture < Textures; ++texture) {
                    Sample(*device, base + texture * TextureBytes, static_cast<float>(texture + 1u), "texture " + std::to_string(texture) + " of pass " + std::to_string(pass));
                    const auto use = AgcDriver::Graphics::TextureCacheUsage();
                    mostHost = std::max(mostHost, use.sampledHostBytes);
                    Require(use.sampledBytes + use.sampledHostBytes <= budget, "after " + std::to_string(pass * Textures + texture + 1u) + " sampled textures the cache holds " + std::to_string(use.sampledBytes >> 20u) + " MiB of device memory and " + std::to_string(use.sampledHostBytes >> 20u) + " MiB of host snapshots in " + std::to_string(use.sampledEntries) + " entries, over its budget of " + std::to_string(BudgetMiB) + " MiB");
                }
            }
            Require(mostHost >= TextureBytes, "the sampled textures kept no host snapshot: the test measures nothing");
            const auto mapped = AgcDriver::Graphics::MappedBufferUsage();
            std::printf("sampled cache host snapshots peaked at %llu MiB under a %llu MiB budget; %llu mapped buffers (%llu MiB) live\n", static_cast<unsigned long long>(mostHost >> 20u), static_cast<unsigned long long>(BudgetMiB), static_cast<unsigned long long>(mapped.count), static_cast<unsigned long long>(mapped.bytes >> 20u));
        } catch (const std::exception& error) {
            failure = error.what();
        }
        AgcDriver::Graphics::ClearCachedTextures(device->Device());
        Require(AgcDriver::Graphics::TextureCacheUsage().sampledHostBytes == 0, "host snapshots remain after the cached textures were cleared");
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(texels);
        }
        if (!failure.empty()) throw std::runtime_error(failure);
        std::puts("sampled cache host memory tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
