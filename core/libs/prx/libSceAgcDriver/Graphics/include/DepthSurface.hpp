#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHSURFACE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHSURFACE_HPP

#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace AgcDriver::Graphics {

class Texture;

VkImageView DepthSurfaceView(const Context& context, const DepthTarget& target);
std::uint64_t DepthSliceBytes(VkExtent2D extent, std::uint32_t bytesPerTexel);
void ClearDepthSurfaces(VkDevice device);
std::uint64_t DepthHtileSliceBytes(VkExtent2D extent);
// A fill of guest memory in packet order: a live depth surface whose HTILE slice it covers with
// words that say "cleared" is cleared when it is next used.
void NoteDepthMetadataFill(std::uint64_t address, std::size_t bytes, std::span<const std::uint32_t, 4> pattern);
class StorageTexture;
void GiveDepthSurfacesTo(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, const std::shared_ptr<StorageTexture>& storage);
std::shared_ptr<Texture> DepthSurfaceTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components);

}

#endif
