#pragma once

#include "../numeric/buffer_format.hpp"
#include "../numeric/element_traits.hpp"
#include "common.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace pyb26::detail {

class buffer_lease {
  Py_buffer buffer_{};

  PYB26_COLD void release_with_error() noexcept {
    PyObject *error = PyErr_GetRaisedException();
    PyBuffer_Release(&buffer_);
    PyErr_SetRaisedException(error);
  }

public:
  buffer_lease() noexcept = default;
  buffer_lease(const buffer_lease &) = delete;
  buffer_lease &operator=(const buffer_lease &) = delete;
  ~buffer_lease() noexcept { reset(); }

  void reset() noexcept {
    if (buffer_.obj) {
      if (PyErr_Occurred()) [[unlikely]]
        release_with_error();
      else
        PyBuffer_Release(&buffer_);
    }
  }

  bool acquire(PyObject *source) noexcept {
    reset();
    buffer_ = {};
    if (!PyObject_CheckBuffer(source)) [[unlikely]]
      return false;
    return PyObject_GetBuffer(source, &buffer_, PyBUF_FULL_RO) == 0;
  }

  const Py_buffer &get() const noexcept { return buffer_; }
  Py_buffer release() noexcept { return std::exchange(buffer_, Py_buffer{}); }
};

template <class T>
inline buffer_dtype parse_buffer_dtype_for(const char *format) noexcept {
  if constexpr (requires {
                  typename std::integral_constant<
                      numeric_kind,
                      parse_buffer_dtype(element_traits<T>::format).kind>;
                }) {
    constexpr const char *canonical = element_traits<T>::format;
    constexpr auto dtype = parse_buffer_dtype(canonical);
    if constexpr (dtype.kind != numeric_kind::unsupported) {
      constexpr auto length = [canonical] {
        std::size_t result = 0;
        while (canonical[result])
          ++result;
        return result;
      }();
      if (format && [&]<std::size_t... I>(std::index_sequence<I...>) {
            return ((format[I] == canonical[I]) && ...);
          }(std::make_index_sequence<length + 1>{}))
        return dtype;
    }
  }
  return parse_buffer_dtype(format);
}

PYB26_COLD inline bool invalid_buffer(const char *message) noexcept {
  PyErr_SetString(PyExc_BufferError, message);
  return false;
}

PYB26_COLD inline bool buffer_overflow() noexcept {
  PyErr_SetString(PyExc_OverflowError,
                  "array metadata exceeds the addressable range");
  return false;
}

} // namespace pyb26::detail
