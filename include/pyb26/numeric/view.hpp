#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

namespace pyb26 {

namespace layout {
struct c_contiguous {};
struct f_contiguous {};
struct strided {};
} // namespace layout

template <class T, std::size_t Rank, class Layout = layout::c_contiguous>
class array_view {
  static constexpr bool is_strided = std::is_same_v<Layout, layout::strided>;
  static constexpr bool is_c = std::is_same_v<Layout, layout::c_contiguous>;
  static_assert(is_strided || is_c ||
                    std::is_same_v<Layout, layout::f_contiguous>,
                "unsupported array layout");
  static_assert(!std::is_volatile_v<T>,
                "volatile array elements are not supported");
  struct no_strides {};
  using stride_storage =
      std::conditional_t<is_strided, std::array<std::ptrdiff_t, Rank>,
                         no_strides>;
  T *data_ = nullptr;
  [[no_unique_address]] std::array<std::size_t, Rank> shape_{};
  [[no_unique_address]] stride_storage strides_{};

  template <class, std::size_t, class> friend class array_view;

public:
  using element_type = T;
  using layout_type = Layout;
  static constexpr std::size_t rank = Rank;

  constexpr array_view() noexcept = default;
  constexpr array_view(T *data, std::array<std::size_t, Rank> shape) noexcept
    requires(!is_strided)
      : data_(data), shape_(shape) {}
  constexpr array_view(T *data, std::array<std::size_t, Rank> shape,
                       std::array<std::ptrdiff_t, Rank> strides) noexcept
    requires(is_strided)
      : data_(data), shape_(shape), strides_(strides) {}

  template <class U>
    requires(std::is_same_v<std::remove_cv_t<U>, std::remove_cv_t<T>> &&
             std::is_convertible_v<U *, T *>)
  constexpr array_view(array_view<U, Rank, Layout> other) noexcept
      : data_(other.data()), shape_(other.shape()) {
    if constexpr (is_strided)
      strides_ = other.strides_;
  }

  constexpr T *data() const noexcept { return data_; }
  constexpr const auto &shape() const noexcept { return shape_; }
  constexpr std::size_t extent(std::size_t axis) const noexcept {
    return shape_[axis];
  }
  constexpr std::size_t size() const noexcept {
    std::size_t count = 1;
    for (auto extent : shape_)
      count *= extent;
    return count;
  }
  constexpr bool empty() const noexcept { return size() == 0; }
  constexpr std::ptrdiff_t stride_bytes(std::size_t axis) const noexcept {
    if constexpr (is_strided) {
      return strides_[axis];
    } else {
      if (empty())
        return 0;
      std::size_t stride = sizeof(T);
      if constexpr (is_c) {
        for (std::size_t i = axis + 1; i < Rank; ++i)
          stride *= shape_[i];
      } else {
        for (std::size_t i = 0; i < axis; ++i)
          stride *= shape_[i];
      }
      return static_cast<std::ptrdiff_t>(stride);
    }
  }

  constexpr T &operator[](std::size_t index) const noexcept
    requires(!is_strided || Rank == 1)
  {
    if constexpr (is_strided)
      return (*this)(index);
    else
      return data_[index];
  }

  constexpr T &
  operator()(const std::array<std::size_t, Rank> &index) const noexcept {
    if constexpr (is_strided) {
      std::ptrdiff_t offset = 0;
      for (std::size_t axis = 0; axis < Rank; ++axis)
        offset += static_cast<std::ptrdiff_t>(index[axis]) * strides_[axis];
      using byte =
          std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>;
      return *reinterpret_cast<T *>(reinterpret_cast<byte *>(data_) + offset);
    } else {
      std::size_t offset = 0;
      if constexpr (is_c) {
        for (std::size_t axis = 0; axis < Rank; ++axis)
          offset = offset * shape_[axis] + index[axis];
      } else {
        for (std::size_t axis = Rank; axis-- > 0;)
          offset = offset * shape_[axis] + index[axis];
      }
      return data_[offset];
    }
  }

  template <class... I>
    requires(sizeof...(I) == Rank && (std::is_integral_v<I> && ...))
  constexpr T &operator()(I... indices) const noexcept {
    const std::array<std::size_t, Rank> index{
        static_cast<std::size_t>(indices)...};
    return (*this)(index);
  }
};

} // namespace pyb26
