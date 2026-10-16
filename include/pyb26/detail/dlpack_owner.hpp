#pragma once

#include "../numeric/shared_array.hpp"
#include "vendor/dlpack/dlpack.h"

#include <complex>
#include <cstring>

namespace pyb26::detail {

template <class T> consteval DLDataType dlpack_dtype() {
  using U = std::remove_const_t<T>;
  static_assert(
      std::is_same_v<U, float> || std::is_same_v<U, double> ||
          std::is_same_v<U, std::int8_t> || std::is_same_v<U, std::uint8_t> ||
          std::is_same_v<U, std::int16_t> || std::is_same_v<U, std::uint16_t> ||
          std::is_same_v<U, std::int32_t> || std::is_same_v<U, std::uint32_t> ||
          std::is_same_v<U, std::int64_t> || std::is_same_v<U, std::uint64_t> ||
          std::is_same_v<U, std::complex<float>> ||
          std::is_same_v<U, std::complex<double>>,
      "unsupported DLPack element type");
  constexpr auto code = std::is_same_v<U, std::complex<float>> ||
                                std::is_same_v<U, std::complex<double>>
                            ? kDLComplex
                        : std::is_floating_point_v<U> ? kDLFloat
                        : std::is_signed_v<U>         ? kDLInt
                                                      : kDLUInt;
  return {static_cast<std::uint8_t>(code), sizeof(U) * 8, 1};
}

template <class Managed, std::size_t Rank> class native_dlpack_owner {
  static_assert(Rank <= 32, "DLPack export rank exceeds the supported limit");
  static_assert(sizeof(std::ptrdiff_t) <= sizeof(std::int64_t));
  Managed managed_{};
  shared_native_storage *storage_ = nullptr;
  std::array<std::int64_t, Rank> shape_{};
  std::array<std::int64_t, Rank> strides_{};

  template <class T, class Layout>
  explicit native_dlpack_owner(shared_array<T, Rank, Layout> &source,
                               std::uint64_t flags) noexcept {
    const auto view = source.view();
    auto &tensor = managed_.dl_tensor;
    tensor.data = view.empty()
                      ? nullptr
                      : const_cast<std::remove_const_t<T> *>(view.data());
    tensor.device = {kDLCPU, 0};
    tensor.ndim = Rank;
    tensor.dtype = dlpack_dtype<T>();
    for (std::size_t i = 0; i < Rank; ++i) {
      shape_[i] = static_cast<std::int64_t>(view.extent(i));
      strides_[i] = static_cast<std::int64_t>(
          view.stride_bytes(i) / static_cast<std::ptrdiff_t>(sizeof(T)));
    }
    if constexpr (Rank) {
      tensor.shape = shape_.data();
      tensor.strides = strides_.data();
    }
    managed_.manager_ctx = this;
    managed_.deleter = &destroy;
    if constexpr (std::is_same_v<Managed, DLManagedTensorVersioned>) {
      managed_.version = {1, 3};
      managed_.flags = flags;
    }
    storage_ = shared_array_access::release(source);
  }

  static void destroy(Managed *tensor) noexcept {
    delete static_cast<native_dlpack_owner *>(tensor->manager_ctx);
  }

public:
  native_dlpack_owner(const native_dlpack_owner &) = delete;
  native_dlpack_owner &operator=(const native_dlpack_owner &) = delete;
  ~native_dlpack_owner() noexcept { release_shared_storage(storage_); }

  template <class T, class Layout>
  static Managed *create(shared_array<T, Rank, Layout> source,
                         std::uint64_t flags) noexcept {
    if (!source)
      return nullptr;
    auto *record = new (std::nothrow) native_dlpack_owner(source, flags);
    return record ? &record->managed_ : nullptr;
  }
};

template <class T, std::size_t Rank, class Layout>
auto copy_dlpack_storage(array_view<T, Rank, Layout> source) noexcept {
  using element = std::remove_const_t<T>;
  auto result = owned_array<element, Rank>::allocate(source.shape());
  if (!result)
    return share(std::move(result));
  const auto size = result.view().size();
  if (!size)
    return share(std::move(result));
  const bool contiguous = [&] {
    if constexpr (std::is_same_v<Layout, layout::c_contiguous>) {
      return true;
    } else {
      std::size_t stride = sizeof(T);
      for (std::size_t axis = Rank; axis-- > 0;) {
        if (source.extent(axis) > 1 &&
            source.stride_bytes(axis) != static_cast<std::ptrdiff_t>(stride))
          return false;
        stride *= source.extent(axis);
      }
      return true;
    }
  }();
  if (contiguous) {
    std::memcpy(result.view().data(), source.data(), size * sizeof(T));
  } else {
    std::array<std::size_t, Rank> index{};
    for (std::size_t i = 0; i < size; ++i) {
      result.view()[i] = source(index);
      for (std::size_t axis = Rank; axis-- > 0;) {
        if (++index[axis] < source.extent(axis))
          break;
        index[axis] = 0;
      }
    }
  }
  return share(std::move(result));
}

} // namespace pyb26::detail
