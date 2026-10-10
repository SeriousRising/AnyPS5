#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaDescriptorFormat.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

class DepthSurface {
public:
    DepthSurface(const Context& context, const DepthTarget& target) : context(context), target(target) {
        this->context.bufferPool.reset();
        VkFormatProperties properties{};
        context.formatProperties(context.physical, target.format, &properties);
        Require((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0, "depth/stencil format " + std::to_string(target.format) + " cannot be an attachment on this device");
        Require((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0, "depth/stencil format " + std::to_string(target.format) + " cannot be sampled on this device");
        Require(target.extent.width <= context.limits.maxFramebufferWidth && target.extent.height <= context.limits.maxFramebufferHeight, "depth target exceeds framebuffer limits");
        const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (target.stencilAddress != 0 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
        try {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = target.format;
            info.extent = {target.extent.width, target.extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage depth");
            VkMemoryRequirements requirements{};
            context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory depth target");
            Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory depth");
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = target.format;
            viewInfo.subresourceRange = {aspects, 0, 1, 0, 1};
            Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView depth");
            auto* recorder = Recorder::Active();
            std::unique_ptr<CommandBatch> batch;
            if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
            const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
            VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            toGeneral.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.image = image;
            toGeneral.subresourceRange = viewInfo.subresourceRange;
            const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
            barrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
            const VkClearDepthStencilValue clear{target.clearDepth, target.clearStencil};
            context.Function<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &toGeneral.subresourceRange);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            if (batch) batch->SubmitAndWait();
            else Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
        } catch (...) {
            release();
            throw;
        }
    }
    ~DepthSurface() { release(); }
    DepthSurface(const DepthSurface&) = delete;
    DepthSurface& operator=(const DepthSurface&) = delete;

    std::shared_ptr<Texture> Sampled(std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components) {
        std::array<std::uint32_t, 12> key{};
        std::copy_n(words.begin(), std::min<std::size_t>(words.size(), 8), key.begin());
        key[8] = components.r;
        key[9] = components.g;
        key[10] = components.b;
        key[11] = components.a;
        if (const auto found = textures.find(key); found != textures.end()) return found->second;
        const bool stencil = Stencil(resource);
        const auto format = ResolveTextureFormat(resource.format);
        if (!Describes(words, resource)) {
            char text[448];
            std::snprintf(text, sizeof(text), "AGC graphics: sampling the %s plane of depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u texture of guest format %u (vk %d), tile mode %u, dimension %d, levels %u-%u, slice %u is not implemented (T# %08x %08x %08x %08x %08x %08x %08x %08x)",
                          stencil ? "stencil" : "depth", static_cast<unsigned long long>(target.address), target.extent.width, target.extent.height, static_cast<int>(target.format), resource.width, resource.height, resource.format, static_cast<int>(format),
                          static_cast<unsigned>(resource.tileMode), static_cast<int>(resource.dimension), resource.baseLevel, resource.lastLevel, resource.baseArray, key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7]);
            throw std::runtime_error(text);
        }
        auto texture = std::make_shared<Texture>(context, image, target.format, stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, components);
        textures.emplace(key, texture);
        return texture;
    }

    // Moves the depth plane between this surface and the first level of a storage image of the
    // plane's format, through a buffer: Vulkan copies between a depth and a color image only with
    // VK_KHR_maintenance8.
    void CopyDepth(const std::shared_ptr<StorageTexture>& storage, bool toStorage) const {
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto bytes = static_cast<std::size_t>(target.extent.width) * target.extent.height * (d16 ? 2u : 4u);
        auto staging = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        auto* recorder = Recorder::Active();
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        constexpr VkAccessFlags every = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkBufferImageCopy depthRegion{};
        depthRegion.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        depthRegion.imageExtent = {target.extent.width, target.extent.height, 1};
        auto colorRegion = depthRegion;
        colorRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        const auto toBuffer = context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer");
        const auto toImage = context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, every, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        toBuffer(commands, toStorage ? image : storage->Image(), VK_IMAGE_LAYOUT_GENERAL, staging->Handle(), 1, toStorage ? &depthRegion : &colorRegion);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        toImage(commands, staging->Handle(), toStorage ? storage->Image() : image, VK_IMAGE_LAYOUT_GENERAL, 1, toStorage ? &colorRegion : &depthRegion);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, every);
        if (batch) {
            batch->SubmitAndWait();
            return;
        }
        Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
        recorder->Keep(std::move(staging));
        recorder->Keep(storage);
    }

    bool Stencil(const GuestTextureResource& resource) const {
        return target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
    }

    bool Describes(std::span<const std::uint32_t> words, const GuestTextureResource& resource) const {
        const bool stencil = Stencil(resource);
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
        const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
        return (ResolveTextureFormat(resource.format) == expected || depthBits) && resource.dimension == TextureDimension::k2D && resource.width == target.extent.width && resource.height == target.extent.height && resource.baseLevel == 0 && resource.lastLevel == 0 && resource.baseArray == 0;
    }

    const Context context;
    const DepthTarget target;
    // The depth plane's contents are in the storage image of `storage` (a shader stores to it)
    // until a draw or a sampled read of the surface takes them back.
    bool handedOff = false;
    GuestTextureResource storage{};
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;

private:
    std::map<std::array<std::uint32_t, 12>, std::shared_ptr<Texture>> textures;

    void release() noexcept {
        textures.clear();
        if (view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
        if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
        if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        view = VK_NULL_HANDLE;
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
    }
};

std::mutex& surfacesMutex();

void takeBack(const std::shared_ptr<DepthSurface>& surface) {
    surface->CopyDepth(CachedStorageSurface(surface->context, surface->storage), false);
    std::lock_guard lock(surfacesMutex());
    surface->handedOff = false;
}

bool sameSurface(const DepthTarget& a, const DepthTarget& b) {
    return a.address == b.address && a.stencilAddress == b.stencilAddress && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format;
}

std::mutex& surfacesMutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::shared_ptr<DepthSurface>>& surfaces() {
    static auto* list = new std::vector<std::shared_ptr<DepthSurface>>();
    return *list;
}

std::vector<std::shared_ptr<DepthSurface>>& retired() {
    static auto* list = new std::vector<std::shared_ptr<DepthSurface>>();
    return *list;
}

}

std::uint64_t DepthSliceBytes(VkExtent2D extent, std::uint32_t bytesPerTexel) {
    const std::uint32_t blockWidth = bytesPerTexel == 4 ? 128u : 256u;
    const std::uint32_t blockHeight = bytesPerTexel == 1 ? 256u : 128u;
    const auto width = static_cast<std::uint64_t>((extent.width + blockWidth - 1) / blockWidth * blockWidth);
    const auto height = static_cast<std::uint64_t>((extent.height + blockHeight - 1) / blockHeight * blockHeight);
    return width * height * bytesPerTexel;
}

VkImageView DepthSurfaceView(const Context& context, const DepthTarget& target) {
    std::shared_ptr<DepthSurface> found;
    {
        std::lock_guard lock(surfacesMutex());
        for (const auto& surface : surfaces()) {
            if (surface->context.device == context.device && sameSurface(surface->target, target)) found = surface;
        }
        if (found == nullptr) {
            surfaces().push_back(std::make_shared<DepthSurface>(context, target));
            return surfaces().back()->view;
        }
        if (!found->handedOff) return found->view;
    }
    takeBack(found);
    return found->view;
}

void ClearDepthSurfaces(VkDevice device) {
    std::lock_guard lock(surfacesMutex());
    std::erase_if(surfaces(), [&](const auto& surface) { return surface->context.device == device; });
    std::erase_if(retired(), [&](const auto& surface) { return surface->context.device == device; });
}

std::shared_ptr<Texture> DepthSurfaceTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components) {
    std::shared_ptr<DepthSurface> surface;
    {
        std::lock_guard lock(surfacesMutex());
        const auto& list = surfaces();
        const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& candidate) {
            return candidate->context.device == context.device && (candidate->target.address == resource.baseAddress || (candidate->target.stencilAddress != 0 && candidate->target.stencilAddress == resource.baseAddress));
        });
        if (found == list.rend()) return nullptr;
        surface = *found;
        if (!surface->handedOff || surface->Stencil(resource)) return surface->Sampled(words, resource, components);
    }
    takeBack(surface);
    std::lock_guard lock(surfacesMutex());
    return surface->Sampled(words, resource, components);
}

void GiveDepthSurfacesTo(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, const std::shared_ptr<StorageTexture>& storage) {
    std::vector<std::shared_ptr<DepthSurface>> reused;
    std::shared_ptr<DepthSurface> live;
    {
        std::lock_guard lock(surfacesMutex());
        auto& list = surfaces();
        const auto at = [&](const auto& surface) { return surface->context.device == context.device && (surface->target.address == resource.baseAddress || surface->Stencil(resource)); };
        for (const auto& surface : list) {
            if (!at(surface) || !surface->Stencil(resource) || !surface->Describes(words, resource)) continue;
            char text[256];
            std::snprintf(text, sizeof(text), "AGC graphics: storage image access to the stencil plane 0x%llx of the %ux%u depth target (vk format %d) that is live at that address is not implemented",
                          static_cast<unsigned long long>(resource.baseAddress), surface->target.extent.width, surface->target.extent.height, static_cast<int>(surface->target.format));
            throw std::runtime_error(text);
        }
        for (auto it = list.begin(); it != list.end();) {
            if (!at(*it)) {
                ++it;
                continue;
            }
            if ((*it)->Describes(words, resource)) {
                if (!(*it)->handedOff) live = *it;
                ++it;
                continue;
            }
            reused.push_back(std::move(*it));
            it = list.erase(it);
        }
        if (Recorder::Active() == nullptr) retired().insert(retired().end(), reused.begin(), reused.end());
    }
    if (live != nullptr) {
        live->CopyDepth(storage, true);
        std::lock_guard lock(surfacesMutex());
        live->handedOff = true;
        live->storage = resource;
        static unsigned reported = 0;
        if (reported < 8) {
            ++reported;
            std::fprintf(stderr, "[gpu] depth surface 0x%llx (%ux%u, VkFormat %d) is written through a %ux%u storage image of guest format %u: its depth moves to that image and back\n", static_cast<unsigned long long>(live->target.address), live->target.extent.width, live->target.extent.height, static_cast<int>(live->target.format), resource.width, resource.height, resource.format);
        }
    }
    for (auto& surface : reused) {
        static unsigned reported = 0;
        if (reported < 32) {
            ++reported;
            std::fprintf(stderr, "[gpu] depth surface 0x%llx (%ux%u, VkFormat %d) retired: its memory is now a %ux%u storage image of guest format %u\n", static_cast<unsigned long long>(surface->target.address), surface->target.extent.width, surface->target.extent.height, static_cast<int>(surface->target.format), resource.width, resource.height, resource.format);
        }
        if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->Keep(std::move(surface));
    }
}

}
