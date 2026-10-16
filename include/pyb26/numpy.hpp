#pragma once

#include "array.hpp"
#include "detail/argument_admission.hpp"
#include "detail/array_descriptor.hpp"
#include "detail/array_error.hpp"
#include "detail/dynamic_array.hpp"
#include "detail/object.hpp"
#include "numeric/shared_array.hpp"

#include <cstring>
#include <optional>
#include <utility>

namespace pyb26::numpy {

int initialize() noexcept;

namespace detail {

using dtype = scalar_dtype;

template <class T> consteval dtype dtype_of() {
  return element_traits<std::remove_cv_t<T>>::dtype;
}

bool numpy_buffer_metadata(PyObject *source, Py_buffer &metadata) noexcept;

PyObject *allocate(dtype type, std::size_t rank, const std::size_t *shape,
                   void **data) noexcept;
PyObject *wrap_native(dtype type, std::size_t rank, const std::size_t *shape,
                      pyb26::detail::native_storage *storage) noexcept;
PyObject *wrap_shared(dtype type, std::size_t rank, const std::size_t *shape,
                      const std::ptrdiff_t *strides, const void *data,
                      bool writable,
                      pyb26::detail::shared_native_storage *storage) noexcept;

} // namespace detail

template <class T, std::size_t Rank, class Layout = layout::strided>
external_array<T, Rank, Layout> borrow(handle source) noexcept {
  if (PyInterpreterState_Get() != PyInterpreterState_Main()) {
    PyErr_SetString(PyExc_RuntimeError,
                    "persistent NumPy imports require the main interpreter");
    return {};
  }
  using checked_type = array_view<T, Rank, Layout>;
  type_caster<checked_type> caster;
  if (!caster.load(source.ptr()))
    return {};
  auto checked = caster.template as<checked_type>();
  auto lease = caster.release_lease();
  auto metadata = lease;
  if (detail::numpy_buffer_metadata(source.ptr(), metadata)) {
    std::array<std::size_t, Rank> shape{};
    for (std::size_t axis = 0; axis < Rank; ++axis)
      shape[axis] = static_cast<std::size_t>(metadata.shape[axis]);
    if (!pyb26::detail::validate_array_layout<T, Rank, Layout>(
            metadata.buf, shape, static_cast<std::size_t>(metadata.len),
            metadata.strides, checked)) {
      PyBuffer_Release(&lease);
      return {};
    }
  }
  if (!std::is_const_v<T> && lease.readonly) {
    PyBuffer_Release(&lease);
    return {};
  }
  array_view<T, Rank, Layout> view;
  if constexpr (std::is_same_v<Layout, layout::strided>) {
    std::array<std::ptrdiff_t, Rank> strides{};
    for (std::size_t axis = 0; axis < Rank; ++axis)
      strides[axis] = checked.stride_bytes(axis);
    view = {const_cast<T *>(checked.data()), checked.shape(), strides};
  } else
    view = {const_cast<T *>(checked.data()), checked.shape()};
  auto owner = pyb26::detail::retain_python_buffer(lease, source.ptr());
  if (!owner)
    return {};
  return external_array<T, Rank, Layout>::unchecked_from_view(view,
                                                              std::move(owner));
}

template <class T> dynamic_array<T> borrow(handle source) noexcept {
  if (PyInterpreterState_Get() != PyInterpreterState_Main()) {
    PyErr_SetString(PyExc_RuntimeError,
                    "persistent NumPy imports require the main interpreter");
    return {};
  }
  pyb26::detail::buffer_lease lease;
  if (!lease.acquire(source.ptr()))
    return {};
  array_descriptor descriptor;
  auto metadata = lease.get();
  detail::numpy_buffer_metadata(source.ptr(), metadata);
  if (!pyb26::detail::validate_dynamic_buffer<T>(metadata, descriptor))
    return {};
  descriptor.owner =
      pyb26::detail::retain_python_buffer(lease.release(), source.ptr());
  if (!descriptor.owner)
    return {};
  return dynamic_array<T>::unchecked_from_descriptor(std::move(descriptor));
}

inline array_descriptor describe(handle source,
                                 bool writable = false) noexcept {
  if (PyInterpreterState_Get() != PyInterpreterState_Main()) {
    PyErr_SetString(PyExc_RuntimeError,
                    "persistent NumPy imports require the main interpreter");
    return {};
  }
  pyb26::detail::buffer_lease lease;
  if (!lease.acquire(source.ptr()))
    return {};
  auto buffer = lease.get();
  detail::numpy_buffer_metadata(source.ptr(), buffer);
  auto parsed = pyb26::detail::parse_buffer_dtype(buffer.format);
  if (!parsed.native_endian)
    return {};
  scalar_dtype dtype;
  const char *format;
  std::size_t alignment;
  using enum scalar_kind;
  if (parsed.kind == boolean && parsed.size == 1) {
    dtype = scalar_dtype::boolean;
    format = "?";
    alignment = alignof(bool);
  } else if (parsed.kind == floating && parsed.size == 2) {
    dtype = scalar_dtype::f16;
    format = "e";
    alignment = 2;
  } else if (parsed.kind == floating && parsed.size == 4) {
    dtype = scalar_dtype::f32;
    format = "f";
    alignment = alignof(float);
  } else if (parsed.kind == floating && parsed.size == 8) {
    dtype = scalar_dtype::f64;
    format = "d";
    alignment = alignof(double);
  } else if (parsed.kind == complex && parsed.size == 8) {
    dtype = scalar_dtype::c64;
    format = "Zf";
    alignment = alignof(float);
  } else if (parsed.kind == complex && parsed.size == 16) {
    dtype = scalar_dtype::c128;
    format = "Zd";
    alignment = alignof(double);
  } else if (parsed.kind == signed_integer || parsed.kind == unsigned_integer) {
    const bool sign = parsed.kind == signed_integer;
    switch (parsed.size) {
    case 1:
      dtype = sign ? scalar_dtype::i8 : scalar_dtype::u8;
      format = sign ? "b" : "B";
      alignment = 1;
      break;
    case 2:
      dtype = sign ? scalar_dtype::i16 : scalar_dtype::u16;
      format = sign ? "h" : "H";
      alignment = alignof(std::int16_t);
      break;
    case 4:
      dtype = sign ? scalar_dtype::i32 : scalar_dtype::u32;
      format = sign ? "i" : "I";
      alignment = alignof(std::int32_t);
      break;
    case 8:
      dtype = sign ? scalar_dtype::i64 : scalar_dtype::u64;
      format = sign ? "q" : "Q";
      alignment = alignof(std::int64_t);
      break;
    default:
      return {};
    }
  } else
    return {};
  array_descriptor descriptor;
  if (!pyb26::detail::validate_dynamic_buffer_metadata(
          buffer, parsed.size, alignment, parsed.kind, dtype, format, writable,
          descriptor))
    return {};
  descriptor.owner =
      pyb26::detail::retain_python_buffer(lease.release(), source.ptr());
  return descriptor;
}

class descriptor_result {
  array_descriptor descriptor_;

public:
  explicit descriptor_result(array_descriptor descriptor) noexcept
      : descriptor_(std::move(descriptor)) {}
  PyObject *release() && noexcept {
    if (!descriptor_)
      return pyb26::detail::array_failure(array_error::invalid_storage);
    if (!pyb26::detail::validate_array_descriptor(descriptor_))
      return nullptr;
    if (descriptor_.rank() > PyBUF_MAX_NDIM)
      return pyb26::detail::invalid_buffer(
                 "array rank exceeds the supported limit"),
             nullptr;
    std::array<std::size_t, PyBUF_MAX_NDIM> shape{};
    for (std::size_t axis = 0; axis < descriptor_.rank(); ++axis) {
      if (descriptor_.shape()[axis] < 0)
        return pyb26::detail::invalid_buffer("negative array extent"), nullptr;
      shape[axis] = static_cast<std::size_t>(descriptor_.shape()[axis]);
    }
    return detail::wrap_shared(descriptor_.dtype, descriptor_.rank(),
                               shape.data(), descriptor_.byte_strides().data(),
                               descriptor_.data, !descriptor_.readonly,
                               descriptor_.owner.release());
  }
};
inline descriptor_result transfer(array_descriptor descriptor) noexcept {
  return descriptor_result(std::move(descriptor));
}

template <class T, std::size_t Rank, class Layout> class external_result {
  external_array<T, Rank, Layout> owner_;

public:
  explicit external_result(external_array<T, Rank, Layout> owner) noexcept
      : owner_(std::move(owner)) {}
  PyObject *release() && noexcept {
    if (!owner_)
      return pyb26::detail::array_failure(owner_.error());
    const auto view = owner_.view();
    std::array<std::ptrdiff_t, Rank> strides{};
    for (std::size_t axis = 0; axis < Rank; ++axis)
      strides[axis] = view.stride_bytes(axis);
    return detail::wrap_shared(detail::dtype_of<T>(), Rank, view.shape().data(),
                               strides.data(), view.data(), !std::is_const_v<T>,
                               owner_.release_owner().release());
  }
};

template <class T, std::size_t Rank, class Layout>
external_result<T, Rank, Layout>
transfer(external_array<T, Rank, Layout> owner) noexcept {
  return external_result<T, Rank, Layout>(std::move(owner));
}

template <class T> class dynamic_result {
  dynamic_array<T> owner_;

public:
  explicit dynamic_result(dynamic_array<T> owner) noexcept
      : owner_(std::move(owner)) {}
  PyObject *release() && noexcept {
    if (!owner_)
      return pyb26::detail::array_failure(owner_.error());
    auto descriptor = owner_.release_descriptor();
    if (!pyb26::detail::validate_array_descriptor(descriptor))
      return nullptr;
    if (descriptor.rank() > PyBUF_MAX_NDIM)
      return pyb26::detail::invalid_buffer(
                 "array rank exceeds the supported limit"),
             nullptr;
    std::array<std::size_t, PyBUF_MAX_NDIM> shape{};
    for (std::size_t axis = 0; axis < descriptor.rank(); ++axis) {
      if (descriptor.shape()[axis] < 0)
        return pyb26::detail::invalid_buffer("negative array extent"), nullptr;
      shape[axis] = static_cast<std::size_t>(descriptor.shape()[axis]);
    }
    return detail::wrap_shared(descriptor.dtype, descriptor.rank(),
                               shape.data(), descriptor.byte_strides().data(),
                               descriptor.data, !descriptor.readonly,
                               descriptor.owner.release());
  }
};

template <class T> dynamic_result<T> transfer(dynamic_array<T> owner) noexcept {
  return dynamic_result<T>(std::move(owner));
}

template <class T, std::size_t Rank> class owned_result {
  static_assert(Rank <= PyBUF_MAX_NDIM,
                "NumPy output rank exceeds the supported limit");
  static constexpr auto dtype_ = detail::dtype_of<T>();
  owned_array<T, Rank> owner_;

public:
  explicit owned_result(owned_array<T, Rank> &&owner) noexcept
      : owner_(std::move(owner)) {}
  owned_result(const owned_result &) = delete;
  owned_result &operator=(const owned_result &) = delete;
  owned_result(owned_result &&) noexcept = default;
  owned_result &operator=(owned_result &&) noexcept = default;
  PyObject *release() && noexcept {
    if (!owner_)
      return pyb26::detail::array_failure(owner_.error());
    const auto shape = owner_.view().shape();
    return detail::wrap_native(
        dtype_, Rank, shape.data(),
        pyb26::detail::owned_array_access::release(owner_));
  }
};

template <class T, std::size_t Rank>
owned_result<T, Rank> transfer(owned_array<T, Rank> &&owner) noexcept {
  return owned_result<T, Rank>(std::move(owner));
}

template <class T, std::size_t Rank, class Layout = layout::c_contiguous>
class shared_result {
  static_assert(Rank <= PyBUF_MAX_NDIM,
                "NumPy output rank exceeds the supported limit");
  static constexpr auto dtype_ = detail::dtype_of<std::remove_const_t<T>>();
  shared_array<T, Rank, Layout> owner_;

public:
  explicit shared_result(shared_array<T, Rank, Layout> &&owner) noexcept
      : owner_(std::move(owner)) {}
  shared_result(const shared_result &) = delete;
  shared_result &operator=(const shared_result &) = delete;
  shared_result(shared_result &&) noexcept = default;
  shared_result &operator=(shared_result &&) noexcept = default;
  PyObject *release() && noexcept {
    if (!owner_)
      return pyb26::detail::array_failure(owner_.error());
    const auto view = owner_.view();
    std::array<std::ptrdiff_t, Rank> strides{};
    if constexpr (!std::is_same_v<Layout, layout::c_contiguous>) {
      for (std::size_t i = 0; i < Rank; ++i)
        strides[i] = view.stride_bytes(i);
    }
    return detail::wrap_shared(
        dtype_, Rank, view.shape().data(),
        std::is_same_v<Layout, layout::c_contiguous> ? nullptr : strides.data(),
        view.data(), !std::is_const_v<T>,
        pyb26::detail::shared_array_access::release(owner_));
  }
};

template <class T, std::size_t Rank, class Layout>
shared_result<T, Rank, Layout>
transfer(shared_array<T, Rank, Layout> &&owner) noexcept {
  return shared_result<T, Rank, Layout>(std::move(owner));
}

template <class T, std::size_t Rank> class array;
template <class T, std::size_t Rank>
array<T, Rank> empty(const std::array<std::size_t, Rank> &shape) noexcept;

template <class T, std::size_t Rank> class array {
  static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>,
                "NumPy output elements must be mutable");
  static_assert(Rank <= PyBUF_MAX_NDIM,
                "NumPy output rank exceeds the supported limit");
  static constexpr auto dtype_ = detail::dtype_of<T>();
  object owner_;
  array_view<T, Rank> view_;

  array(PyObject *owner, T *data,
        const std::array<std::size_t, Rank> &shape) noexcept
      : owner_(object::steal(owner)), view_(data, shape) {}
  friend array empty<T, Rank>(const std::array<std::size_t, Rank> &) noexcept;

public:
  array() noexcept = default;
  array(const array &) = delete;
  array &operator=(const array &) = delete;
  array(array &&other) noexcept
      : owner_(std::move(other.owner_)), view_(std::exchange(other.view_, {})) {
  }
  array &operator=(array &&other) noexcept {
    if (this != &other) {
      owner_ = std::move(other.owner_);
      view_ = std::exchange(other.view_, {});
    }
    return *this;
  }

  explicit operator bool() const noexcept { return bool(owner_); }
  array_view<T, Rank> view() const noexcept { return view_; }
  PyObject *release() noexcept {
    view_ = {};
    return owner_.release();
  }
};

template <class T, std::size_t Rank>
array<T, Rank> empty(const std::array<std::size_t, Rank> &shape) noexcept {
  void *data = nullptr;
  PyObject *owner =
      detail::allocate(detail::dtype_of<T>(), Rank, shape.data(), &data);
  if (!owner)
    return {};
  return array<T, Rank>(owner, static_cast<T *>(data), shape);
}

template <class T, std::size_t Rank, class Layout>
array<std::remove_const_t<T>, Rank>
copy(array_view<T, Rank, Layout> source) noexcept {
  using element = std::remove_const_t<T>;
  auto result = empty<element>(source.shape());
  if (!result)
    return {};
  const auto bytes = result.view().size() * sizeof(element);
  if (bytes) {
    if (!source.data()) {
      PyErr_SetString(PyExc_BufferError,
                      "nonempty copy source has a null pointer");
      return {};
    }
    if constexpr (std::is_same_v<Layout, layout::c_contiguous> ||
                  (Rank <= 1 && std::is_same_v<Layout, layout::f_contiguous>)) {
      std::memcpy(result.view().data(), source.data(), bytes);
    } else {
      std::array<std::size_t, Rank> index{};
      auto *destination = result.view().data();
      for (std::size_t i = 0; i < result.view().size(); ++i) {
        destination[i] = source(index);
        for (std::size_t axis = Rank; axis-- > 0;) {
          if (++index[axis] < source.extent(axis))
            break;
          index[axis] = 0;
        }
      }
    }
  }
  return result;
}

} // namespace pyb26::numpy

namespace pyb26 {

template <class T, std::size_t Rank, class Layout>
struct type_caster<numpy::shared_result<T, Rank, Layout>> {
  static constexpr caster_effects effects{false, false, false};
  static PyObject *cast(numpy::shared_result<T, Rank, Layout> result) noexcept {
    return std::move(result).release();
  }
};

template <class T, std::size_t Rank>
struct type_caster<numpy::owned_result<T, Rank>> {
  static constexpr caster_effects effects{false, false, false};
  static PyObject *cast(numpy::owned_result<T, Rank> result) noexcept {
    return std::move(result).release();
  }
};

template <class T, std::size_t Rank> struct type_caster<numpy::array<T, Rank>> {
  static constexpr caster_effects effects{false, false, false, true};
  static PyObject *cast(numpy::array<T, Rank> result) noexcept {
    if (!result && !PyErr_Occurred())
      PyErr_SetString(PyExc_RuntimeError, "NumPy result has no allocation");
    return result.release();
  }
};

} // namespace pyb26

namespace pyb26 {
template <class T, std::size_t Rank, class Layout>
struct type_caster<numpy::external_result<T, Rank, Layout>> {
  static constexpr caster_effects effects{false, false, false};
  static PyObject *
  cast(numpy::external_result<T, Rank, Layout> result) noexcept {
    return std::move(result).release();
  }
};
template <class T> struct type_caster<numpy::dynamic_result<T>> {
  static constexpr caster_effects effects{false, false, false};
  static PyObject *cast(numpy::dynamic_result<T> result) noexcept {
    return std::move(result).release();
  }
};
} // namespace pyb26

namespace pyb26 {
template <class T> struct type_caster<dynamic_array<T>> {
  static constexpr caster_effects effects{true, false, false, true, true};
  detail::argument_owner_admission<true> admission_;
  dynamic_array<T> value_;
  bool load(PyObject *source) noexcept {
    if (!admission_.acquire())
      return false;
    value_ = numpy::borrow<T>(handle(source));
    return static_cast<bool>(value_);
  }
  template <class A> decltype(auto) as() noexcept {
    return static_cast<A>(value_);
  }
};
template <class T, std::size_t Rank, class Layout>
struct type_caster<external_array<T, Rank, Layout>> {
  static constexpr caster_effects effects{true, false, false, true, true};
  detail::argument_owner_admission<true> admission_;
  external_array<T, Rank, Layout> value_;
  bool load(PyObject *source) noexcept {
    if (!admission_.acquire())
      return false;
    value_ = numpy::borrow<T, Rank, Layout>(handle(source));
    return static_cast<bool>(value_);
  }
  template <class A> decltype(auto) as() noexcept {
    return static_cast<A>(value_);
  }
};
} // namespace pyb26

namespace pyb26 {
template <> struct type_caster<numpy::descriptor_result> {
  static constexpr caster_effects effects{false, false, false};
  static PyObject *cast(numpy::descriptor_result value) noexcept {
    return std::move(value).release();
  }
};
} // namespace pyb26
