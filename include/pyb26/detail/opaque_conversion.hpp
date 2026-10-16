#pragma once

#include "stl.hpp"

#include <optional>
#include <utility>

namespace pyb26::detail {

template <class T>
concept opaque_bound_value = requires(caster_for<T> caster) {
  { caster.ptr } -> std::convertible_to<T *>;
};

template <class T>
bool opaque_load_item(PyObject *object, std::optional<T> &value,
                      callback_error_scope &error) noexcept {
  const bool loaded = stl_convert_item<T>(object, &error, [&](auto &&item) {
    value.emplace(std::forward<decltype(item)>(item));
    return true;
  });
  return !error.capture() && loaded;
}

inline PyObject *opaque_finish_output(PyObject *result,
                                      callback_error_scope &error) noexcept {
  if (error.capture()) {
    error.discard(result);
    return nullptr;
  }
  return result;
}

template <class T, class F>
bool opaque_with_copy(const T &source, callback_error_scope &error,
                      F &&use) noexcept {
  if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T, std::string>) {
    return use(T(source));
  } else {
    T copy(source);
    callback_error_checkpoint checkpoint{error};
    return !error.capture() && use(std::move(copy));
  }
}

template <class T> PyObject *opaque_cast_value(T &&value) noexcept {
  if constexpr (!builtin_output_cast<T>)
    if (PyErr_Occurred())
      return nullptr;
  PyObject *result = caster_for<T>::cast(std::forward<T>(value));
  if constexpr (!builtin_output_cast<T>)
    if (!result || PyErr_Occurred())
      return stl_output_failure(result);
  return result;
}

template <class T> PyObject *opaque_copy_output(T &source) noexcept {
  if constexpr (!opaque_bound_value<T>) {
    return opaque_cast_value(source);
  } else {
    callback_error_scope error;
    if (error.capture())
      return nullptr;
    PyObject *result = nullptr;
    {
      T copy(source);
      callback_error_checkpoint checkpoint{error};
      if (!error.capture())
        result = opaque_cast_value(std::move(copy));
    }
    return opaque_finish_output(result, error);
  }
}

} // namespace pyb26::detail
