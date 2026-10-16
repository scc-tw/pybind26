#pragma once

#include "detail/common.hpp"

#include <memory>
#include <optional>
#include <type_traits>

namespace pyb26 {

class object;

namespace detail {

template <class T>
struct python_error_result
    : std::bool_constant<std::is_pointer_v<T> || std::is_base_of_v<object, T>> {
};
template <class T>
struct python_error_result<std::optional<T>> : std::true_type {};
template <class T>
struct python_error_result<std::shared_ptr<T>> : std::true_type {};
template <class T, class D>
struct python_error_result<std::unique_ptr<T, D>> : std::true_type {};
template <class T>
inline constexpr bool python_error_result_v =
    python_error_result<std::remove_cvref_t<T>>::value;

} // namespace detail

PYB26_COLD inline int set_error(PyObject *exception,
                                const char *message) noexcept {
  if (!PyErr_Occurred())
    PyErr_SetString(exception, message);
  return -1;
}

PYB26_COLD inline int value_error(const char *message) noexcept {
  return set_error(PyExc_ValueError, message);
}

PYB26_COLD inline int timeout_error(const char *message) noexcept {
  return set_error(PyExc_TimeoutError, message);
}

PYB26_COLD inline PyObject *error_return(PyObject *exception,
                                         const char *message) noexcept {
  set_error(exception, message);
  return nullptr;
}

} // namespace pyb26
