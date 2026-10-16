#pragma once

#include "complex.hpp"
#include "detail/array_layout.hpp"
#include "numeric/view.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace pyb26 {

template <class T, std::size_t Rank, class Layout>
struct type_caster<array_view<T, Rank, Layout>> {
  static constexpr caster_effects effects{true, true, true};
  using view_type = array_view<T, Rank, Layout>;
  using element_type = std::remove_const_t<T>;
  static_assert(Rank <= PyBUF_MAX_NDIM,
                "array rank exceeds the buffer protocol limit");
  static_assert(std::is_implicit_lifetime_v<element_type> &&
                    std::is_trivially_copyable_v<element_type>,
                "numeric storage requires trivial implicit-lifetime elements");
  static_assert(element_traits<element_type>::enabled,
                "unsupported array element type");

  detail::buffer_lease lease_;
  view_type view_;

  bool load(PyObject *source) noexcept {
    view_ = {};
    if (!lease_.acquire(source))
      return false;
    return validate();
  }

  Py_buffer release_lease() noexcept { return lease_.release(); }

  template <class A> decltype(auto) as() noexcept {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(view_);
    else
      return view_;
  }

  template <class R> static PyObject *cast(R &&) noexcept {
    static_assert(!std::is_same_v<R, R>,
                  "borrowed view cannot be returned; use an owning result or "
                  "explicit copy");
    return nullptr;
  }

private:
  bool validate() noexcept {
    const Py_buffer &buffer = lease_.get();
    if (!buffer.obj || buffer.ndim < 0 || buffer.ndim > PyBUF_MAX_NDIM ||
        buffer.itemsize <= 0 || buffer.len < 0 || !buffer.format)
      return detail::invalid_buffer("invalid buffer metadata");
    if (buffer.ndim != static_cast<int>(Rank))
      return false;
    if constexpr (Rank == 0) {
      if (buffer.shape || buffer.strides || buffer.suboffsets)
        return detail::invalid_buffer(
            "scalar buffer must not have shape or strides");
    } else {
      if (!buffer.shape || !buffer.strides)
        return detail::invalid_buffer(
            "buffer did not provide requested shape and strides");
    }
    const auto dtype =
        detail::parse_buffer_dtype_for<element_type>(buffer.format);
    if (dtype.kind == detail::numeric_kind::unsupported)
      return false;
    if (dtype.size != static_cast<std::size_t>(buffer.itemsize))
      return detail::invalid_buffer("buffer format and itemsize disagree");
    constexpr auto expected_kind = element_traits<element_type>::kind;
    if (!dtype.native_endian || dtype.kind != expected_kind ||
        dtype.size != sizeof(T) || buffer.suboffsets ||
        (!std::is_const_v<T> && buffer.readonly))
      return false;

    std::array<std::size_t, Rank> shape{};
    bool empty = false;
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      if (buffer.shape[axis] < 0)
        return detail::invalid_buffer("negative array extent");
      shape[axis] = static_cast<std::size_t>(buffer.shape[axis]);
      empty = empty || shape[axis] == 0;
    }
    std::size_t bytes = empty ? 0 : sizeof(T);
    if (!empty) {
      for (auto extent : shape) {
        if (extent > static_cast<std::size_t>(PY_SSIZE_T_MAX) / bytes)
          return detail::buffer_overflow();
        bytes *= extent;
      }
    }
    if (bytes != static_cast<std::size_t>(buffer.len))
      return detail::invalid_buffer("buffer length and shape disagree");
    return detail::validate_array_layout<T, Rank, Layout>(
        buffer.buf, shape, bytes, buffer.strides, view_);
  }
};

} // namespace pyb26
