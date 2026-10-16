#pragma once

#include "view.hpp"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace pyb26 {

enum class array_error {
  none,
  uninitialized,
  overflow,
  out_of_memory,
  invalid_storage,
  invalid_slice,
  invalid_axes
};

namespace detail {

struct alignas(std::max_align_t) native_storage {
  void *data;
  std::size_t bytes;
  void (*destroy)(native_storage *) noexcept;
  bool (*valid)(native_storage *) noexcept = nullptr;
  void *(*retain_export)(native_storage *) noexcept = nullptr;
};

inline void free_native_storage(native_storage *storage) noexcept {
  std::free(storage);
}

template <class T, class Deleter> struct adopted_storage : native_storage {
  std::unique_ptr<T[], Deleter> allocation;
  adopted_storage(std::unique_ptr<T[], Deleter> &&source,
                  std::size_t bytes) noexcept
      : native_storage{source.get(), bytes, &dispose},
        allocation(std::move(source)) {}
  static void dispose(native_storage *storage) noexcept {
    delete static_cast<adopted_storage *>(storage);
  }
};

template <std::size_t Rank>
bool native_array_bytes(const std::array<std::size_t, Rank> &shape,
                        std::size_t itemsize, std::size_t &bytes) noexcept {
  constexpr auto limit =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  bool empty = false;
  for (auto extent : shape) {
    if (extent > limit)
      return false;
    empty = empty || extent == 0;
  }
  bytes = empty ? 0 : itemsize;
  if (!empty) {
    for (auto extent : shape) {
      if (extent > limit / bytes)
        return false;
      bytes *= extent;
    }
  }
  return true;
}

struct owned_array_access;

} // namespace detail

template <class T, std::size_t Rank> class owned_array {
  static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>,
                "owned array elements must be mutable");
  static_assert(
      std::is_implicit_lifetime_v<T> && std::is_trivially_copyable_v<T>,
      "owned array elements must be implicit-lifetime and trivially copyable");
  static_assert(alignof(T) <= alignof(std::max_align_t),
                "over-aligned owned array elements are not supported");
  detail::native_storage *storage_ = nullptr;
  std::array<std::size_t, Rank> shape_{};
  array_error error_ = array_error::uninitialized;

  owned_array(detail::native_storage *storage,
              const std::array<std::size_t, Rank> &shape) noexcept
      : storage_(storage), shape_(shape), error_(array_error::none) {}
  friend struct detail::owned_array_access;

public:
  owned_array() noexcept = default;
  owned_array(const owned_array &) = delete;
  owned_array &operator=(const owned_array &) = delete;
  owned_array(owned_array &&other) noexcept
      : storage_(std::exchange(other.storage_, nullptr)),
        shape_(std::exchange(other.shape_, {})),
        error_(std::exchange(other.error_, array_error::uninitialized)) {}
  owned_array &operator=(owned_array &&other) noexcept {
    if (this != &other) {
      owned_array previous(std::move(other));
      std::swap(storage_, previous.storage_);
      std::swap(shape_, previous.shape_);
      std::swap(error_, previous.error_);
    }
    return *this;
  }
  ~owned_array() noexcept { reset(); }

  void reset() noexcept {
    auto *storage = std::exchange(storage_, nullptr);
    shape_ = {};
    error_ = array_error::uninitialized;
    if (storage)
      storage->destroy(storage);
  }
  explicit operator bool() const noexcept { return storage_ != nullptr; }
  array_error error() const noexcept { return error_; }
  array_view<T, Rank> view() const noexcept {
    return {storage_ ? static_cast<T *>(storage_->data) : nullptr, shape_};
  }
  static owned_array failure(array_error error) noexcept {
    owned_array result;
    result.error_ =
        error == array_error::none ? array_error::uninitialized : error;
    return result;
  }
  static owned_array
  allocate(const std::array<std::size_t, Rank> &shape) noexcept {
    std::size_t bytes;
    if (!detail::native_array_bytes(shape, sizeof(T), bytes))
      return failure(array_error::overflow);
    const auto payload = bytes ? bytes : sizeof(T);
    if (payload > std::numeric_limits<std::size_t>::max() -
                      sizeof(detail::native_storage))
      return failure(array_error::overflow);
    void *memory = std::malloc(sizeof(detail::native_storage) + payload);
    if (!memory)
      return failure(array_error::out_of_memory);
    void *data =
        static_cast<std::byte *>(memory) + sizeof(detail::native_storage);
    auto *storage = ::new (memory)
        detail::native_storage{data, bytes, &detail::free_native_storage};
    return owned_array(storage, shape);
  }
  template <class Deleter>
  static owned_array
  adopt(std::unique_ptr<T[], Deleter> allocation,
        const std::array<std::size_t, Rank> &shape) noexcept {
    static_assert(
        std::is_same_v<typename std::unique_ptr<T[], Deleter>::pointer, T *> &&
            !std::is_reference_v<Deleter> &&
            std::is_nothrow_move_constructible_v<Deleter> &&
            std::is_nothrow_destructible_v<Deleter>,
        "native adoption requires a raw pointer and an owned nothrow-movable "
        "deleter");
    std::size_t bytes;
    if (!detail::native_array_bytes(shape, sizeof(T), bytes))
      return failure(array_error::overflow);
    const auto address = reinterpret_cast<std::uintptr_t>(allocation.get());
    if (!address || address % alignof(T) ||
        (bytes &&
         address > std::numeric_limits<std::uintptr_t>::max() - (bytes - 1)))
      return failure(array_error::invalid_storage);
    auto *storage = new (std::nothrow)
        detail::adopted_storage<T, Deleter>(std::move(allocation), bytes);
    if (!storage)
      return failure(array_error::out_of_memory);
    return owned_array(storage, shape);
  }
};

namespace detail {

struct owned_array_access {
  template <class T, std::size_t Rank>
  static native_storage *release(owned_array<T, Rank> &source) noexcept {
    auto *storage = std::exchange(source.storage_, nullptr);
    source.reset();
    return storage;
  }
};

} // namespace detail
} // namespace pyb26
