#pragma once

#include "shared_array.hpp"

namespace pyb26 {

class array_owner {
  detail::shared_native_storage *storage_ = nullptr;
  explicit array_owner(detail::shared_native_storage *storage) noexcept
      : storage_(storage) {}

public:
  array_owner() noexcept = default;
  array_owner(const array_owner &other) noexcept : storage_(other.storage_) {
    detail::retain_shared_storage(storage_);
  }
  array_owner(array_owner &&other) noexcept
      : storage_(std::exchange(other.storage_, nullptr)) {}
  array_owner &operator=(array_owner other) noexcept {
    std::swap(storage_, other.storage_);
    return *this;
  }
  ~array_owner() noexcept { detail::release_shared_storage(storage_); }
  explicit operator bool() const noexcept {
    return storage_ && (!storage_->storage->valid ||
                        storage_->storage->valid(storage_->storage));
  }
  void reset() noexcept { array_owner{}.swap(*this); }
  void swap(array_owner &other) noexcept {
    std::swap(storage_, other.storage_);
  }
  void *retain_export() const noexcept {
    return detail::retain_storage_export(storage_);
  }
  detail::shared_native_storage *release() noexcept {
    return std::exchange(storage_, nullptr);
  }
  static array_owner adopt(detail::native_storage *storage) noexcept {
    if (!storage)
      return {};
    auto *shared = new (std::nothrow) detail::shared_native_storage;
    if (!shared) {
      storage->destroy(storage);
      return {};
    }
    shared->storage = storage;
    return array_owner(shared);
  }
  template <class Owner>
  static array_owner retain(std::shared_ptr<Owner> source) noexcept {
    struct retained : detail::native_storage {
      std::shared_ptr<Owner> owner;
      explicit retained(std::shared_ptr<Owner> value) noexcept
          : detail::native_storage{nullptr, 0, &destroy_owner},
            owner(std::move(value)) {}
      static void destroy_owner(detail::native_storage *storage) noexcept {
        delete static_cast<retained *>(storage);
      }
    };
    if (!source)
      return {};
    return adopt(new (std::nothrow) retained(std::move(source)));
  }
  template <class T, std::size_t Rank, class Layout>
  static array_owner retain(shared_array<T, Rank, Layout> source) noexcept {
    if (!source)
      return {};
    return array_owner(detail::shared_array_access::release(source));
  }
};

template <class T, std::size_t Rank, class Layout = layout::strided>
class external_array {
  array_owner owner_;
  array_view<T, Rank, Layout> view_{};
  array_error error_ = array_error::uninitialized;

  static array_error validate(array_view<T, Rank, Layout> view) noexcept {
    std::size_t bytes;
    if (!detail::native_array_bytes(view.shape(), sizeof(T), bytes))
      return array_error::overflow;
    if (!bytes)
      return array_error::none;
    const auto address = reinterpret_cast<std::uintptr_t>(view.data());
    if (!address || address % alignof(T))
      return array_error::invalid_storage;
    constexpr auto limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    std::size_t lower = 0, upper = 0;
    std::array<std::size_t, Rank> axes{}, magnitudes{};
    std::size_t count = 0;
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      if (view.extent(axis) <= 1)
        continue;
      const auto stride = view.stride_bytes(axis);
      if (stride % static_cast<std::ptrdiff_t>(alignof(T)))
        return array_error::invalid_storage;
      const auto magnitude = detail::stride_magnitude(stride);
      auto &span = stride < 0 ? lower : upper;
      if (magnitude > (limit - span) / (view.extent(axis) - 1))
        return array_error::overflow;
      span += magnitude * (view.extent(axis) - 1);
      if constexpr (!std::is_const_v<T>) {
        std::size_t index = count++;
        while (index && magnitudes[index - 1] > magnitude) {
          axes[index] = axes[index - 1];
          magnitudes[index] = magnitudes[index - 1];
          --index;
        }
        axes[index] = axis;
        magnitudes[index] = magnitude;
      }
    }
    if (lower > limit - upper || sizeof(T) > limit - lower - upper ||
        lower > address ||
        upper + sizeof(T) - 1 >
            std::numeric_limits<std::uintptr_t>::max() - address)
      return array_error::overflow;
    if constexpr (!std::is_const_v<T>) {
      std::size_t span = sizeof(T);
      for (std::size_t index = 0; index < count; ++index) {
        if (magnitudes[index] < span)
          return array_error::invalid_storage;
        span += (view.extent(axes[index]) - 1) * magnitudes[index];
      }
    }
    return array_error::none;
  }

public:
  using element_type = T;
  external_array() noexcept = default;
  external_array(array_view<T, Rank, Layout> view, array_owner owner) noexcept
      : owner_(std::move(owner)), view_(view),
        error_(owner_ ? validate(view) : array_error::invalid_storage) {
    if (error_ != array_error::none) {
      view_ = {};
      owner_.reset();
    }
  }
  explicit operator bool() const noexcept { return static_cast<bool>(owner_); }
  array_error error() const noexcept {
    return error_ == array_error::none && !owner_ ? array_error::invalid_storage
                                                  : error_;
  }
  array_view<T, Rank, Layout> view() const noexcept {
    return owner_ ? view_ : array_view<T, Rank, Layout>{};
  }
  array_owner owner() const noexcept { return owner_; }
  array_owner release_owner() noexcept { return std::move(owner_); }
  static external_array unchecked_from_view(array_view<T, Rank, Layout> view,
                                            array_owner owner) noexcept {
    external_array result;
    result.owner_ = std::move(owner);
    result.view_ = view;
    result.error_ =
        result.owner_ ? array_error::none : array_error::invalid_storage;
    return result;
  }
  static external_array failure(array_error error) noexcept {
    external_array result;
    result.error_ = error;
    return result;
  }
};

} // namespace pyb26
