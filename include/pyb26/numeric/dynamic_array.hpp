#pragma once

#include "array_owner.hpp"
#include "buffer_format.hpp"
#include "element_traits.hpp"
#include <span>
#include <vector>

namespace pyb26 {

struct array_metadata {
  std::vector<std::ptrdiff_t> shape;
  std::vector<std::ptrdiff_t> byte_strides;
};

struct array_descriptor {
  void *data = nullptr;
  std::size_t itemsize = 0;
  scalar_dtype dtype = scalar_dtype::u8;
  const char *format = nullptr;
  bool readonly = true;
  std::size_t bytes = 0;
  std::shared_ptr<const array_metadata> metadata;
  array_owner owner;
  std::size_t rank() const noexcept {
    return metadata ? metadata->shape.size() : 0;
  }
  std::span<const std::ptrdiff_t> shape() const noexcept {
    return metadata ? std::span(metadata->shape)
                    : std::span<const std::ptrdiff_t>{};
  }
  std::span<const std::ptrdiff_t> byte_strides() const noexcept {
    return metadata ? std::span(metadata->byte_strides)
                    : std::span<const std::ptrdiff_t>{};
  }
  explicit operator bool() const noexcept {
    return static_cast<bool>(owner) && static_cast<bool>(metadata);
  }
  array_error error() const noexcept {
    return *this ? array_error::none : array_error::invalid_storage;
  }
  bool element_strides(std::span<std::ptrdiff_t> output) const noexcept {
    if (!metadata || byte_strides().size() != rank() || !itemsize ||
        itemsize > static_cast<std::size_t>(
                       std::numeric_limits<std::ptrdiff_t>::max()) ||
        output.size() != rank())
      return false;
    for (std::size_t axis = 0; axis < rank(); ++axis) {
      const auto value = metadata->byte_strides[axis];
      if (value % static_cast<std::ptrdiff_t>(itemsize))
        return false;
      output[axis] = value / static_cast<std::ptrdiff_t>(itemsize);
    }
    return true;
  }
};

template <class T> class dynamic_array {
  static_assert(!std::is_volatile_v<T> && std::is_implicit_lifetime_v<T> &&
                std::is_trivially_copyable_v<T>);
  static_assert(element_traits<std::remove_const_t<T>>::enabled);
  array_descriptor descriptor_;
  array_error error_ = array_error::uninitialized;

public:
  using element_type = T;
  dynamic_array() noexcept = default;
  explicit dynamic_array(array_descriptor descriptor) noexcept
      : descriptor_(std::move(descriptor)), error_(array_error::none) {
    const auto invalid = [&] {
      if (!descriptor_ || descriptor_.itemsize != sizeof(T) ||
          descriptor_.dtype != element_traits<T>::dtype ||
          descriptor_.shape().size() != descriptor_.byte_strides().size() ||
          descriptor_.rank() > 64 ||
          (!std::is_const_v<T> && descriptor_.readonly))
        return array_error::invalid_storage;
      const auto format = detail::parse_buffer_dtype(descriptor_.format);
      if (!format.native_endian || format.kind != element_traits<T>::kind ||
          format.size != sizeof(T))
        return array_error::invalid_storage;
      constexpr auto limit =
          static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
      bool empty = false;
      for (const auto extent : descriptor_.shape()) {
        if (extent < 0)
          return array_error::invalid_storage;
        empty = empty || !extent;
      }
      std::size_t bytes = empty ? 0 : sizeof(T);
      if (!empty)
        for (const auto extent : descriptor_.shape()) {
          if (static_cast<std::size_t>(extent) > limit / bytes)
            return array_error::overflow;
          bytes *= static_cast<std::size_t>(extent);
        }
      if (bytes != descriptor_.bytes)
        return array_error::invalid_storage;
      if (bytes) {
        const auto address = reinterpret_cast<std::uintptr_t>(descriptor_.data);
        if (!address || address % alignof(T))
          return array_error::invalid_storage;
        std::size_t lower = 0, upper = 0;
        for (std::size_t axis = 0; axis < descriptor_.rank(); ++axis) {
          if (descriptor_.shape()[axis] <= 1)
            continue;
          const auto stride = descriptor_.byte_strides()[axis];
          if (stride % static_cast<std::ptrdiff_t>(alignof(T)))
            return array_error::invalid_storage;
          const auto magnitude = detail::stride_magnitude(stride);
          auto &span = stride < 0 ? lower : upper;
          const auto steps =
              static_cast<std::size_t>(descriptor_.shape()[axis] - 1);
          if (magnitude > (limit - span) / steps)
            return array_error::overflow;
          span += magnitude * steps;
        }
        if (lower > limit - upper || sizeof(T) > limit - lower - upper ||
            lower > address ||
            upper + sizeof(T) - 1 >
                std::numeric_limits<std::uintptr_t>::max() - address)
          return array_error::overflow;
      }
      return array_error::none;
    }();
    if (invalid != array_error::none) {
      descriptor_ = {};
      error_ = invalid;
    } else {
      descriptor_.metadata =
          std::make_shared<const array_metadata>(*descriptor_.metadata);
      descriptor_.format = element_traits<T>::format;
      if constexpr (std::is_const_v<T>)
        descriptor_.readonly = true;
    }
  }
  explicit operator bool() const noexcept {
    return static_cast<bool>(descriptor_);
  }
  array_error error() const noexcept {
    return error_ == array_error::none && !descriptor_
               ? array_error::invalid_storage
               : error_;
  }
  T *data() const noexcept {
    return descriptor_ ? static_cast<T *>(descriptor_.data) : nullptr;
  }
  std::size_t rank() const noexcept { return descriptor_.rank(); }
  auto shape() const noexcept { return descriptor_.shape(); }
  auto byte_strides() const noexcept { return descriptor_.byte_strides(); }
  std::size_t size() const noexcept {
    return descriptor_ ? descriptor_.bytes / sizeof(T) : 0;
  }
  bool readonly() const noexcept { return descriptor_.readonly; }
  const array_descriptor &descriptor() const noexcept { return descriptor_; }
  array_descriptor release_descriptor() noexcept {
    return std::move(descriptor_);
  }
  static dynamic_array
  unchecked_from_descriptor(array_descriptor descriptor) noexcept {
    dynamic_array result;
    result.descriptor_ = std::move(descriptor);
    result.error_ = array_error::none;
    return result;
  }
  static dynamic_array failure(array_error error) noexcept {
    dynamic_array result;
    result.error_ = error;
    return result;
  }
};

} // namespace pyb26
