#pragma once

#include "array.hpp"

#include <limits>
#include <mdspan>
#include <optional>

namespace pyb26 {

template <class T, class Extents, class Layout, class Accessor>
struct type_caster<std::mdspan<T, Extents, Layout, Accessor>> {
  static constexpr caster_effects effects{true, true, true};
  static_assert(std::is_same_v<Accessor, std::default_accessor<T>>,
                "mdspan requires the default accessor");
  static_assert(std::is_same_v<Layout, std::layout_right> ||
                    std::is_same_v<Layout, std::layout_left> ||
                    std::is_same_v<Layout, std::layout_stride>,
                "unsupported mdspan layout");
  using view_type = std::mdspan<T, Extents, Layout, Accessor>;
  using index_type = typename Extents::index_type;
  using source_layout = std::conditional_t<
      std::is_same_v<Layout, std::layout_right>, layout::c_contiguous,
      std::conditional_t<std::is_same_v<Layout, std::layout_left>,
                         layout::f_contiguous, layout::strided>>;
  using source_type = array_view<T, Extents::rank(), source_layout>;
  static constexpr std::size_t rank = Extents::rank();
  static constexpr auto limit =
      static_cast<std::size_t>(std::numeric_limits<index_type>::max());
  type_caster<source_type> source_;
  std::optional<view_type> view_;

  bool load(PyObject *source) noexcept {
    view_.reset();
    if (!source_.load(source))
      return false;
    const auto value = source_.template as<source_type>();
    std::array<index_type, rank> shape{};
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const auto extent = value.extent(axis);
      if (extent > limit ||
          (Extents::static_extent(axis) != std::dynamic_extent &&
           extent != Extents::static_extent(axis)))
        return false;
      shape[axis] = static_cast<index_type>(extent);
    }
    if constexpr (std::is_same_v<Layout, std::layout_stride>) {
      std::array<index_type, rank> strides{};
      if (!validate_strides(value, strides))
        return false;
      view_.emplace(value.data(),
                    typename view_type::mapping_type(Extents(shape), strides));
    } else {
      std::size_t product = 1;
      for (const auto extent : value.shape()) {
        if (extent == 0)
          continue;
        if (extent > limit / product)
          return false;
        product *= extent;
      }
      view_.emplace(value.data(), Extents(shape));
    }
    return true;
  }

  template <class A> decltype(auto) as() noexcept {
    return static_cast<A>(*view_);
  }

  template <class R> static PyObject *cast(R &&) noexcept {
    static_assert(!std::is_same_v<R, R>,
                  "borrowed view cannot be returned; use an owning result or "
                  "explicit copy");
    return nullptr;
  }

private:
  static bool validate_strides(source_type value,
                               std::array<index_type, rank> &strides) noexcept {
    std::array<std::size_t, rank> axes{};
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const auto bytes = value.stride_bytes(axis);
      if (bytes <= 0 || bytes % static_cast<std::ptrdiff_t>(sizeof(T)))
        return false;
      const auto stride = static_cast<std::size_t>(bytes) / sizeof(T);
      if (stride > limit)
        return false;
      strides[axis] = static_cast<index_type>(stride);
      std::size_t position = axis;
      while (position &&
             (strides[axes[position - 1]] > strides[axis] ||
              (strides[axes[position - 1]] == strides[axis] &&
               value.extent(axes[position - 1]) > value.extent(axis)))) {
        axes[position] = axes[position - 1];
        --position;
      }
      axes[position] = axis;
    }
    for (std::size_t i = 1; i < rank; ++i) {
      const auto previous = axes[i - 1];
      if (value.extent(previous) >
          static_cast<std::size_t>(strides[axes[i]]) /
              static_cast<std::size_t>(strides[previous]))
        return false;
    }
    if (!value.empty()) {
      std::size_t span = 1;
      for (std::size_t axis = 0; axis < rank; ++axis) {
        const auto count = value.extent(axis) - 1;
        const auto stride = static_cast<std::size_t>(strides[axis]);
        if (count > (limit - span) / stride)
          return false;
        span += count * stride;
      }
    }
    return true;
  }
};

} // namespace pyb26
