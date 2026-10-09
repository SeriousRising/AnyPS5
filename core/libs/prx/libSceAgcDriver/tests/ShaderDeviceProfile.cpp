#include "prx/libSceAgcDriver/Execution/include/ShaderDeviceProfile.hpp"
#include "BdaAbi.hpp"
#include "Optimization/BindingAllocator.hpp"
#include "Optimization/DescriptorBindingBuilder.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using AgcDriver::Graphics::Require;

struct ProfileInput {
    std::vector<std::uint32_t> capabilities{spv::CapabilityShader, spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess};
    std::array<std::string, 2> extensionStrings{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    std::array<std::string_view, 2> extensions{extensionStrings[0], extensionStrings[1]};
    VkPhysicalDeviceRobustness2FeaturesEXT robustness{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, nullptr, VK_FALSE, VK_FALSE, VK_TRUE};
    VkPhysicalDeviceDescriptorIndexingFeatures indexing{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, &robustness};
    VkPhysicalDevice8BitStorageFeatures bytes{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, &indexing, VK_TRUE};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, &bytes, VK_TRUE};
    VkPhysicalDeviceFeatures core{};
    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    VkPhysicalDeviceLimits limits{};

    ProfileInput() {
        core.shaderInt64 = VK_TRUE;
        core.vertexPipelineStoresAndAtomics = VK_TRUE;
        core.fragmentStoresAndAtomics = VK_TRUE;
        device.pEnabledFeatures = &core;
        device.pNext = &bda;
        limits.maxPushConstantsSize = 128u;
        limits.maxBoundDescriptorSets = 1u;
        limits.maxStorageBufferRange = sizeof(ShaderRecompiler::RuntimeAbi::ShaderData);
    }

    ShaderRecompiler::SpirvTarget Target() const {
        ShaderRecompiler::SpirvTarget target{};
        target.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
        target.supportedCapabilities = capabilities;
        target.supportedExtensions = extensions;
        return target;
    }
};

template<typename TAction>
void Reject(TAction action, const char* expected) {
    try {
        action();
    } catch (const std::runtime_error& error) {
        Require(std::string(error.what()).find(expected) != std::string::npos, "unexpected profile validation error");
        return;
    }
    throw std::runtime_error("invalid shader device profile was accepted");
}

void CheckProfile() {
    ProfileInput input;
    const AgcDriver::ShaderDeviceProfile profile(input.Target(), input.device, input.limits);
    input.capabilities.clear();
    input.extensionStrings[0] = "changed";
    input.core.shaderInt64 = VK_FALSE;
    input.robustness.nullDescriptor = VK_FALSE;
    input.limits.maxPushConstantsSize = 0u;
    const auto target = profile.Target();
    Require(target.supportedCapabilities.size() == 4u && std::ranges::is_sorted(target.supportedCapabilities), "profile capabilities were not frozen");
    Require(std::ranges::find(target.supportedExtensions, "SPV_KHR_physical_storage_buffer") != target.supportedExtensions.end(), "profile extension storage is borrowed");
    Require(profile.Limits().maxPushConstantsSize == 128u, "profile limits were not frozen");
    Require(profile.NullDescriptors(), "profile null descriptor support was not frozen");
    static_assert(!std::is_copy_constructible_v<AgcDriver::ShaderDeviceProfile> && !std::is_move_constructible_v<AgcDriver::ShaderDeviceProfile>);
    const auto invalid = [](auto mutate, const char* expected) {
        ProfileInput candidate;
        mutate(candidate);
        Reject([&] { AgcDriver::ShaderDeviceProfile rejected(candidate.Target(), candidate.device, candidate.limits); }, expected);
    };
    invalid([](auto& value) { value.device.pEnabledFeatures = nullptr; }, "enabled core features");
    invalid([](auto& value) { value.bda.bufferDeviceAddress = VK_FALSE; }, "bufferDeviceAddress");
    invalid([](auto& value) { value.device.pNext = nullptr; }, "bufferDeviceAddress");
    invalid([](auto& value) { value.core.shaderInt64 = VK_FALSE; }, "shaderInt64");
    invalid([](auto& value) { value.bytes.storageBuffer8BitAccess = VK_FALSE; }, "storageBuffer8BitAccess");
    ProfileInput withoutNull;
    withoutNull.robustness.nullDescriptor = VK_FALSE;
    Require(!AgcDriver::ShaderDeviceProfile(withoutNull.Target(), withoutNull.device, withoutNull.limits).NullDescriptors(), "a device without nullDescriptor was profiled with it");
    ProfileInput withoutRobustness;
    withoutRobustness.indexing.pNext = nullptr;
    Require(!AgcDriver::ShaderDeviceProfile(withoutRobustness.Target(), withoutRobustness.device, withoutRobustness.limits).NullDescriptors(), "a device without VK_EXT_robustness2 was profiled with nullDescriptor");
    invalid([](auto& value) { value.core.fragmentStoresAndAtomics = VK_FALSE; }, "graphics stores and atomics");
    invalid([](auto& value) { value.limits.maxPushConstantsSize = 127u; }, "exceeds device limits");
    invalid([](auto& value) { value.limits.maxBoundDescriptorSets = 0u; }, "exceeds device limits");
    invalid([](auto& value) { --value.limits.maxStorageBufferRange; }, "ShaderData exceeds device limits");
    invalid([](auto& value) { value.capabilities.clear(); }, "missing runtime capabilities");
    invalid([](auto& value) { value.extensions[0] = "missing"; }, "missing runtime extensions");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilitySampledImageArrayDynamicIndexing); }, "shaderSampledImageArrayDynamicIndexing");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityStorageImageArrayNonUniformIndexing); }, "shaderStorageImageArrayNonUniformIndexing");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityStorageImageReadWithoutFormat); }, "shaderStorageImageReadWithoutFormat");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityStorageImageMultisample); }, "shaderStorageImageMultisample");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityFragmentBarycentricKHR); }, "fragmentShaderBarycentric");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityMeshShadingEXT); }, "meshShader");
    ProfileInput enabled;
    enabled.capabilities.push_back(spv::CapabilitySampledImageArrayDynamicIndexing);
    enabled.capabilities.push_back(spv::CapabilityStorageImageArrayNonUniformIndexing);
    enabled.core.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    enabled.indexing.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    const AgcDriver::ShaderDeviceProfile accepted(enabled.Target(), enabled.device, enabled.limits);
    Require(accepted.Target().supportedCapabilities.size() == 6u, "enabled descriptor indexing was rejected");
}

void CheckAbi() {
    using ShaderRecompiler::RuntimeAbi::Binding;
    using ShaderRecompiler::RuntimeAbi::BindingNumber;
    using ShaderRecompiler::RuntimeAbi::Stage;
    std::set<std::uint32_t> bindings;
    for (std::uint32_t stage = 0u; stage < ShaderRecompiler::RuntimeAbi::StageCount; ++stage) {
        for (std::uint32_t binding = 0u; binding < static_cast<std::uint32_t>(Binding::Count); ++binding) {
            Require(bindings.insert(BindingNumber(static_cast<Stage>(stage), static_cast<Binding>(binding))).second, "runtime ABI bindings overlap");
        }
    }
    Require(BindingNumber(Stage::Main, Binding::ShaderData) == 62u && BindingNumber(Stage::Fragment, Binding::ShaderData) == 125u, "runtime ABI binding numbers changed");
    Reject([] { BindingNumber(static_cast<Stage>(4u), Binding::Buffers); }, "invalid stage or binding");
    Reject([] { BindingNumber(Stage::Main, Binding::Count); }, "invalid stage or binding");
    Reject([] { ShaderRecompiler::RuntimeAbi::RequireVersion(0u); }, "incompatible version");
    ShaderRecompiler::RuntimeAbi::RequireVersion(ShaderRecompiler::RuntimeAbi::Version);
}

void CheckHeaps() {
    using namespace ShaderRecompiler;
    const auto allocate = [](const ImageResource& image, std::uint32_t count, std::uint32_t samplers = 0u) {
        IrProgram program;
        program.Metadata().shaderInfoComplete = true;
        program.Resources().info.images.assign(count, image);
        program.Resources().info.samplers.resize(samplers);
        return BindingAllocator{}.Allocate(program, {0u, 0u, 0u, 128u});
    };
    ImageResource image;
    image.resourceClass = ImageResourceClass::Sampled;
    image.numericClass = IrTextureNumericClass::Float;
    image.dimension = RdnaImageDimension::Dim2D;
    const auto single = allocate(image, 1u);
    const auto sixteenSamplers = allocate(image, 1u, 16u);
    Require(BindingAllocator{}.FindBinding(sixteenSamplers.layout, DescriptorBindingKind::Samplers).resources.size() == 32u, "sixteen logical samplers must fit with both descriptor variants");
    const auto full = allocate(image, RuntimeAbi::SampledHeapCapacity, RuntimeAbi::SamplerHeapCapacity / 2u);
    Require(single.layout.ShaderDataDwords() == full.layout.ShaderDataDwords() && single.layout.memoryOffsetDword == full.layout.memoryOffsetDword && !full.layout.UsesPushData(), "runtime layout depends on resource count");
    Require(full.layout.memoryOffsetDword == 0u && full.layout.DispatchThreadLimitDword() == 0u && full.layout.ShaderDataDwords() == 0u, "direct image resources allocated runtime metadata");
    Reject([&] { allocate(image, RuntimeAbi::SampledHeapCapacity + 1u); }, "heap capacity exceeded");
    Require(BindingAllocator{}.FindBinding(allocate(image, 40u).layout, DescriptorBindingForImage(image)).resources.size() == 40u, "a sampled heap does not hold 40 images of one class");
    Require(BindingAllocator{}.FindBinding(allocate(image, RuntimeAbi::SampledHeapCapacity).layout, DescriptorBindingForImage(image)).resources.size() == RuntimeAbi::SampledHeapCapacity, "a sampled heap does not hold a full class of images");
    Require(ResourceMaterializer::BindlessSlots() == 16u, "bindless image tables changed size with the sampled heap");
    Reject([&] { allocate(image, 1u, RuntimeAbi::SamplerHeapCapacity + 1u); }, "metadata capacity");
    image.resourceClass = ImageResourceClass::Storage;
    const auto directStorage = image;
    for (const auto count : {5u, 16u}) {
        const auto direct = allocate(image, count);
        const auto& resources = BindingAllocator{}.FindBinding(direct.layout, DescriptorBindingForImage(image)).resources;
        Require(resources.size() == count, "direct storage images did not retain their heap elements");
        for (std::uint32_t index = 0u; index < count; ++index) Require(resources[index] == index, "direct storage image resource indices changed");
    }
    Reject([&] { allocate(image, 17u); }, "heap capacity exceeded");
    image.mipMode = ImageMipMode::DynamicStorage;
    image.mipCount = RuntimeAbi::StorageMipSlots;
    const auto storage = allocate(image, 1u);
    Require(BindingAllocator{}.FindBinding(storage.layout, DescriptorBindingForImage(image)).resources.size() == image.mipCount, "storage heap did not reserve each mip");
    const auto shared = allocate(image, 2u);
    Require(BindingAllocator{}.FindBinding(shared.layout, DescriptorBindingForImage(image)).resources.size() == 2u * image.mipCount, "a storage heap did not hold two dynamic-mip images");
    const auto capacity = RuntimeAbi::StorageHeapCapacity / image.mipCount;
    const auto filled = allocate(image, capacity);
    Require(BindingAllocator{}.FindBinding(filled.layout, DescriptorBindingForImage(image)).resources.size() == RuntimeAbi::StorageHeapCapacity, "a full storage heap was not allocated");
    Reject([&] { allocate(image, capacity + 1u); }, "heap capacity exceeded");
    const auto allocateImages = [](std::vector<ImageResource> images) {
        IrProgram program;
        program.Metadata().shaderInfoComplete = true;
        program.Resources().info.images = std::move(images);
        return BindingAllocator{}.Allocate(program, {0u, 0u, 0u, 128u});
    };
    const auto interleaved = allocateImages({directStorage, image, directStorage, image});
    Require(BindingAllocator{}.FindBinding(interleaved.layout, DescriptorBindingForImage(image)).resources == std::vector<std::uint32_t>{0u, 1u, 1u, 1u, 1u, 2u, 3u, 3u, 3u, 3u}, "interleaved storage images overlap or changed their mip slots");
    const auto mixedFull = allocateImages({image, directStorage, image, directStorage, image, directStorage, directStorage});
    Require(BindingAllocator{}.FindBinding(mixedFull.layout, DescriptorBindingForImage(image)).resources == std::vector<std::uint32_t>{0u, 0u, 0u, 0u, 1u, 2u, 2u, 2u, 2u, 3u, 4u, 4u, 4u, 4u, 5u, 6u}, "a full mixed storage heap changed its resource or mip slots");
    Reject([&] { allocateImages({image, directStorage, image, directStorage, image, directStorage, directStorage, directStorage}); }, "heap capacity exceeded");
    Reject([&] { allocate(image, 0u, RuntimeAbi::SamplerHeapCapacity / 2u + 1u); }, "sampler pairs");
    const std::array dimensions{RdnaImageDimension::Dim1D, RdnaImageDimension::Dim1DArray, RdnaImageDimension::Dim2D, RdnaImageDimension::Dim2DArray, RdnaImageDimension::Dim3D, RdnaImageDimension::Dim2DMsaa, RdnaImageDimension::Dim2DMsaaArray};
    std::set<std::uint32_t> classes;
    for (std::uint32_t group = 0u; group < 8u; ++group) {
        for (const auto dimension : dimensions) {
            image = {};
            image.resourceClass = group < 4u ? ImageResourceClass::Sampled : ImageResourceClass::Storage;
            image.numericClass = group == 1u || group >= 5u ? IrTextureNumericClass::Uint : group == 2u ? IrTextureNumericClass::Sint : IrTextureNumericClass::Float;
            image.depthCompare = group == 3u;
            image.atomic = group >= 6u;
            image.atomic64 = group == 7u;
            image.dimension = dimension;
            const auto binding = DescriptorBindingForImage(image);
            Require(classes.insert(static_cast<std::uint32_t>(binding)).second, "typed image classes overlap");
            Require(RuntimeAbi::HeapCapacity(binding) == (group < 4u ? RuntimeAbi::SampledHeapCapacity : RuntimeAbi::StorageHeapCapacity), "typed image class has an invalid capacity");
        }
    }
    Require(classes.size() == RuntimeAbi::ImageBindingCount && *classes.begin() == 1u && *classes.rbegin() == RuntimeAbi::ImageBindingCount, "typed image class mapping is incomplete");
    Reject([] { RuntimeAbi::HeapCapacity(RuntimeAbi::Binding::ShaderData); }, "not a typed heap");
}

void CheckStorageMips() {
    using namespace ShaderRecompiler;
    ImageResource image;
    image.resourceClass = ImageResourceClass::Storage;
    image.numericClass = IrTextureNumericClass::Float;
    image.dimension = RdnaImageDimension::Dim2D;
    image.mipMode = ImageMipMode::DynamicStorage;
    image.mipCount = RuntimeAbi::StorageMipSlots;
    const auto modes = ResourceMaterializer::RuntimeImageModes(image);
    Require(!modes.empty() && std::ranges::all_of(modes, [](const auto& mode) { return mode.mipCount == 4u; }), "runtime storage image modes changed the per-image mip span");
    DescriptorValue descriptor{{1u, (20u << 20u) | (3u << 30u), 15u | (15u << 14u), 0x90000facu | (3u << 16u), 0u, 0u, 0u, 0u}, 8u};
    Require(ResourceMaterializer::RuntimeImageMode(image, descriptor, modes) < modes.size(), "four storage mip levels were rejected");
    IrProgram program;
    program.Metadata().shaderInfoComplete = true;
    program.Resources().info.images = {image};
    ResourceMaterializer::PrepareImageModes(program.Resources().info);
    const auto allocation = BindingAllocator{}.Allocate(program, {0u, 0u, 0u, 128u});
    ResourceSnapshot snapshot;
    snapshot.images = {descriptor};
    const auto prepare = [&] { return DescriptorBindingBuilder{}.Prepare(allocation.layout, program.Resources().info, IrShaderStage::Compute, snapshot); };
    const auto plan = prepare();
    Require(plan.bindings.size() == 1u && plan.bindings.front().descriptor.count == 4u && plan.bindings.front().resources == std::vector<std::uint32_t>{0u, 0u, 0u, 0u}, "four storage mip levels did not retain their descriptor elements");
    Require(std::ranges::find(plan.specialization, PipelineSpecializationConstant{PipelineSpecialization::MipCountBase, 4u}) != plan.specialization.end(), "four storage mip levels changed their specialization count");
    descriptor.dwords[3] = 0x90000facu | (4u << 16u);
    Reject([&] { ResourceMaterializer::RuntimeImageMode(image, descriptor, modes); }, "runtime storage image mip capacity exceeded");
    snapshot.images = {descriptor};
    Reject(prepare, "runtime storage image mip capacity exceeded");
    snapshot.images.front().dwords[0] = 0u;
    Reject(prepare, "invalid dynamic storage mip range");
}

}

int main() {
    try {
        CheckAbi();
        CheckHeaps();
        CheckStorageMips();
        CheckProfile();
        std::cout << "shader runtime ABI and device profile tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
