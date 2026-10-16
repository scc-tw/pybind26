#pragma once

#include "owned_array.hpp"

#include <atomic>

namespace pyb26 {

struct array_slice {
  std::size_t start;
  std::size_t count;
  std::ptrdiff_t step = 1;
};

namespace detail {

struct shared_native_storage {
  std::atomic<std::size_t> references{1};
  native_storage *storage;
};

inline void *retain_storage_export(shared_native_storage *storage) noexcept {
  return storage && storage->storage->retain_export
             ? storage->storage->retain_export(storage->storage)
             : nullptr;
}

inline void retain_shared_storage(shared_native_storage *storage) noexcept {
  if (storage)
    storage->references.fetch_add(1, std::memory_order_relaxed);
}

inline void release_shared_storage(shared_native_storage *storage) noexcept {
  if (storage &&
      storage->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    auto *allocation = storage->storage;
    delete storage;
    allocation->destroy(allocation);
  }
}

constexpr std::size_t stride_magnitude(std::ptrdiff_t stride) noexcept {
  return stride < 0 ? static_cast<std::size_t>(-(stride + 1)) + 1
                    : static_cast<std::size_t>(stride);
}

struct shared_array_access;

} // namespace detail

template <class T, std::size_t Rank, class Layout = layout::c_contiguous>
class shared_array;

template <class T, std::size_t Rank>
shared_array<T, Rank> share(owned_array<T, Rank> &&source) noexcept;

template <class T, std::size_t Rank, class Layout> class shared_array {
  static_assert(!std::is_volatile_v<T> && std::is_implicit_lifetime_v<T> &&
                    std::is_trivially_copyable_v<T>,
                "shared array elements must be nonvolatile implicit-lifetime "
                "and trivially copyable");
  detail::shared_native_storage *storage_ = nullptr;
  array_view<T, Rank, Layout> view_{};
  array_error error_ = array_error::uninitialized;

  shared_array(detail::shared_native_storage *storage,
               array_view<T, Rank, Layout> view) noexcept
      : storage_(storage), view_(view), error_(array_error::none) {}
  template <class, std::size_t, class> friend class shared_array;
  template <class U, std::size_t N>
  friend shared_array<U, N> share(owned_array<U, N> &&) noexcept;
  friend struct detail::shared_array_access;

public:
  shared_array() noexcept = default;
  shared_array(const shared_array &other) noexcept
      : storage_(other.storage_), view_(other.view_), error_(other.error_) {
    detail::retain_shared_storage(storage_);
  }
  template <class U>
    requires(!std::is_same_v<T, U> &&
             std::is_same_v<std::remove_const_t<T>, U> && std::is_const_v<T>)
  shared_array(const shared_array<U, Rank, Layout> &other) noexcept
      : storage_(other.storage_), view_(other.view_), error_(other.error_) {
    detail::retain_shared_storage(storage_);
  }
  shared_array(shared_array &&other) noexcept
      : storage_(std::exchange(other.storage_, nullptr)),
        view_(std::exchange(other.view_, {})),
        error_(std::exchange(other.error_, array_error::uninitialized)) {}
  shared_array &operator=(const shared_array &other) noexcept {
    if (this != &other) {
      shared_array replacement(other);
      swap(replacement);
    }
    return *this;
  }
  shared_array &operator=(shared_array &&other) noexcept {
    if (this != &other) {
      shared_array replacement(std::move(other));
      swap(replacement);
    }
    return *this;
  }
  ~shared_array() noexcept { reset(); }

  void swap(shared_array &other) noexcept {
    std::swap(storage_, other.storage_);
    std::swap(view_, other.view_);
    std::swap(error_, other.error_);
  }
  void reset() noexcept {
    auto *storage = std::exchange(storage_, nullptr);
    view_ = {};
    error_ = array_error::uninitialized;
    detail::release_shared_storage(storage);
  }
  explicit operator bool() const noexcept { return storage_ != nullptr; }
  array_error error() const noexcept { return error_; }
  array_view<T, Rank, Layout> view() const noexcept { return view_; }
  static shared_array failure(array_error error) noexcept {
    shared_array result;
    result.error_ =
        error == array_error::none ? array_error::uninitialized : error;
    return result;
  }
  shared_array<T, Rank, layout::strided>
  transpose(const std::array<std::size_t, Rank> &axes) const noexcept {
    using result_type = shared_array<T, Rank, layout::strided>;
    if (!storage_)
      return result_type::failure(error_);
    std::array<bool, Rank> seen{};
    std::array<std::size_t, Rank> shape{};
    std::array<std::ptrdiff_t, Rank> strides{};
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      const auto source = axes[axis];
      if (source >= Rank || seen[source])
        return result_type::failure(array_error::invalid_axes);
      seen[source] = true;
      shape[axis] = view_.extent(source);
      strides[axis] = view_.stride_bytes(source);
    }
    detail::retain_shared_storage(storage_);
    return result_type(storage_, {view_.data(), shape, strides});
  }
  shared_array<T, Rank, layout::strided>
  slice(const std::array<array_slice, Rank> &axes) const noexcept {
    using result_type = shared_array<T, Rank, layout::strided>;
    if (!storage_)
      return result_type::failure(error_);
    std::array<std::size_t, Rank> shape{};
    bool empty = false;
    for (std::size_t i = 0; i < Rank; ++i) {
      const auto [start, count, step] = axes[i];
      const auto extent = view_.extent(i);
      if (!step || start > extent || (count && start == extent))
        return result_type::failure(array_error::invalid_slice);
      if (count > 1) {
        const auto available = step < 0 ? start : extent - 1 - start;
        if (count - 1 > available / detail::stride_magnitude(step))
          return result_type::failure(array_error::invalid_slice);
      }
      shape[i] = count;
      empty = empty || count == 0;
    }
    std::array<std::ptrdiff_t, Rank> strides{};
    auto *data = view_.data();
    if (!empty) {
      std::ptrdiff_t offset = 0;
      constexpr auto limit =
          static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
      for (std::size_t i = 0; i < Rank; ++i) {
        const auto stride = view_.stride_bytes(i);
        offset += stride * static_cast<std::ptrdiff_t>(axes[i].start);
        if (shape[i] <= 1)
          continue;
        const auto magnitude = detail::stride_magnitude(stride);
        const auto step = detail::stride_magnitude(axes[i].step);
        if (magnitude && step > limit / magnitude)
          return result_type::failure(array_error::overflow);
        const auto product = static_cast<std::ptrdiff_t>(magnitude * step);
        strides[i] = (stride < 0) != (axes[i].step < 0) ? -product : product;
      }
      using byte =
          std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>;
      data = reinterpret_cast<T *>(reinterpret_cast<byte *>(data) + offset);
    }
    detail::retain_shared_storage(storage_);
    return result_type(storage_, {data, shape, strides});
  }
};

template <class T, std::size_t Rank>
shared_array<T, Rank> share(owned_array<T, Rank> &&source) noexcept {
  auto owner = std::move(source);
  using result_type = shared_array<T, Rank>;
  if (!owner)
    return result_type::failure(owner.error());
  auto *storage = new (std::nothrow) detail::shared_native_storage;
  if (!storage)
    return result_type::failure(array_error::out_of_memory);
  const auto view = owner.view();
  storage->storage = detail::owned_array_access::release(owner);
  return result_type(storage, view);
}

namespace detail {

struct shared_array_access {
  template <class T, std::size_t Rank, class Layout>
  static shared_native_storage *
  release(shared_array<T, Rank, Layout> &source) noexcept {
    auto *storage = std::exchange(source.storage_, nullptr);
    source.reset();
    return storage;
  }
};

} // namespace detail
} // namespace pyb26
