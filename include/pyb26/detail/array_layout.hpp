#pragma once

#include "../numeric/view.hpp"
#include "buffer.hpp"

namespace pyb26::detail {

template <class T, std::size_t Rank>
inline bool validate_array_strides(
    std::uintptr_t address, const std::array<std::size_t, Rank> &shape,
    const std::array<std::ptrdiff_t, Rank> &strides) noexcept {
  constexpr auto limit =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  std::size_t lower = 0, upper = 0;
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    if (shape[axis] <= 1)
      continue;
    if (strides[axis] % static_cast<std::ptrdiff_t>(alignof(T)))
      return false;
    const auto magnitude =
        strides[axis] < 0
            ? std::size_t{0} - static_cast<std::size_t>(strides[axis])
            : static_cast<std::size_t>(strides[axis]);
    auto &extent = strides[axis] < 0 ? lower : upper;
    if (magnitude > (limit - extent) / (shape[axis] - 1))
      return detail::buffer_overflow();
    extent += magnitude * (shape[axis] - 1);
  }
  if (lower > limit - upper || sizeof(T) > limit - lower - upper ||
      lower > address ||
      upper + sizeof(T) - 1 >
          std::numeric_limits<std::uintptr_t>::max() - address)
    return detail::buffer_overflow();
  if constexpr (!std::is_const_v<T>) {
    std::array<std::size_t, Rank> axes{};
    std::array<std::size_t, Rank> magnitudes{};
    std::size_t count = 0;
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      if (shape[axis] <= 1)
        continue;
      const auto magnitude =
          strides[axis] < 0
              ? std::size_t{0} - static_cast<std::size_t>(strides[axis])
              : static_cast<std::size_t>(strides[axis]);
      std::size_t i = count++;
      while (i && magnitudes[i - 1] > magnitude) {
        axes[i] = axes[i - 1];
        magnitudes[i] = magnitudes[i - 1];
        --i;
      }
      axes[i] = axis;
      magnitudes[i] = magnitude;
    }
    std::size_t span = sizeof(T);
    for (std::size_t i = 0; i < count; ++i) {
      if (magnitudes[i] < span)
        return false;
      span += (shape[axes[i]] - 1) * magnitudes[i];
    }
  }
  return true;
}

template <class T, std::size_t Rank, class Layout, class Strides>
inline bool
validate_array_layout(void *data, const std::array<std::size_t, Rank> &shape,
                      std::size_t bytes, const Strides &source_strides,
                      array_view<T, Rank, Layout> &result) noexcept {
  using view_type = array_view<T, Rank, Layout>;
  std::array<std::ptrdiff_t, Rank> strides{};
  if constexpr (std::is_same_v<Layout, layout::strided>) {
    for (std::size_t axis = 0; axis < Rank; ++axis)
      strides[axis] = source_strides[axis];
  }
  if (bytes) {
    if (!data)
      return detail::invalid_buffer("nonempty buffer has a null pointer");
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    if constexpr (!std::is_same_v<Layout, layout::strided>) {
      if (bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - address)
        return detail::buffer_overflow();
    }
    if (address % alignof(T))
      return false;
    if constexpr (std::is_same_v<Layout, layout::strided>) {
      if (!validate_array_strides<T>(address, shape, strides))
        return false;
    } else {
      std::size_t stride = sizeof(T);
      for (std::size_t i = 0; i < Rank; ++i) {
        const auto axis =
            std::is_same_v<Layout, layout::c_contiguous> ? Rank - 1 - i : i;
        if (shape[axis] > 1 &&
            source_strides[axis] != static_cast<Py_ssize_t>(stride))
          return false;
        stride *= shape[axis];
      }
    }
  }
  if constexpr (std::is_same_v<Layout, layout::strided>)
    result = view_type(static_cast<T *>(data), shape, strides);
  else
    result = view_type(static_cast<T *>(data), shape);
  return true;
}

} // namespace pyb26::detail
