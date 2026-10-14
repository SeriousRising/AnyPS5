#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWSNAPSHOTREUSE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWSNAPSHOTREUSE_HPP

#include <array>
#include <cstddef>
#include <cstdint>

namespace AgcDriver::Graphics {

enum class DrawSnapshotReuseKind : std::uint8_t { Cold, Invalidated, Held, Unchanged, Equal, Patched, Count };

struct DrawSnapshotReuse {
    DrawSnapshotReuseKind kind = DrawSnapshotReuseKind::Cold;
    std::uint64_t patchedBytes = 0;
};

struct DrawSnapshotReuseCensus {
    static constexpr auto Kinds = static_cast<std::size_t>(DrawSnapshotReuseKind::Count);
    std::array<std::uint64_t, Kinds> views{};
    std::array<std::uint64_t, Kinds> bytes{};
    std::uint64_t oddViews = 0;
    std::uint64_t oddBytes = 0;
    std::uint64_t patchedBytes = 0;

    bool operator==(const DrawSnapshotReuseCensus&) const = default;

    void Add(std::size_t extent, const DrawSnapshotReuse& reuse) {
        if (extent % 4 != 0) {
            ++oddViews;
            oddBytes += extent;
            return;
        }
        const auto index = static_cast<std::size_t>(reuse.kind);
        ++views[index];
        bytes[index] += extent;
        patchedBytes += reuse.patchedBytes;
    }

    void Add(const DrawSnapshotReuseCensus& other) {
        for (std::size_t i = 0; i < Kinds; ++i) {
            views[i] += other.views[i];
            bytes[i] += other.bytes[i];
        }
        oddViews += other.oddViews;
        oddBytes += other.oddBytes;
        patchedBytes += other.patchedBytes;
    }
};

}

#endif
