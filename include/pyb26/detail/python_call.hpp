#pragma once

#include "argument_plan.hpp"
#include "lifetime.hpp"
#include "python_failure.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pyb26::detail {

template <class A>
PYB26_ALWAYS_INLINE PyObject *python_call_argument(A &&value) {
  using D = std::remove_cvref_t<A>;
  if constexpr (requires { type_caster<D>::cast(static_cast<A &&>(value)); })
    return type_caster<D>::cast(static_cast<A &&>(value));
  else if constexpr (std::is_pointer_v<D>)
    return type_caster<std::remove_cv_t<std::remove_pointer_t<D>>>::cast(value);
  else
    return type_caster<D>::cast(static_cast<A &&>(value));
}

template <class... A, std::size_t... I>
PYB26_ALWAYS_INLINE bool
build_python_call_arguments(PyThreadState *state, PyObject **values,
                            std::index_sequence<I...>, A &&...args) {
  return (... && ((values[I] = python_call_argument(static_cast<A &&>(args))) &&
                  !cpython::current_exception(state)));
}

template <std::size_t N>
PYB26_ALWAYS_INLINE void
clear_python_call_arguments(PyObject **values,
                            callback_error_scope &error) noexcept {
  for (std::size_t i = 0; i < N; ++i)
    error.discard(values[i]);
}

struct python_result_no_cleanup {
  explicit python_result_no_cleanup(callback_error_scope &) noexcept {}
};

template <class R>
PYB26_ALWAYS_INLINE R load_python_call_result(PyThreadState *state,
                                              PyObject *result,
                                              callback_error_scope &error,
                                              const char *message) {
  using caster = type_caster<std::remove_cvref_t<R>>;
  caster converted;
  using checkpoint =
      std::conditional_t<std::is_trivially_destructible_v<caster> ||
                             input_caster_cleanup_inert_v<R>,
                         python_result_no_cleanup, callback_error_checkpoint>;
  checkpoint cleanup{error};
  if (!converted.load(result) || cpython::current_exception(state))
      [[unlikely]] {
    if (!cpython::current_exception(state))
      PyErr_SetString(PyExc_TypeError, message);
    return python_failure_return<R>();
  }
  return project_input<R>(converted);
}

template <class R>
PYB26_ALWAYS_INLINE R finish_python_call(PyThreadState *state, PyObject *result,
                                         callback_error_scope &error,
                                         const char *message) {
  static_assert(
      !std::is_reference_v<R>,
      "pyb26: a Python callback or override cannot return a reference; "
      "return an owning value");
  static_assert(
      !caster_yields_view_v<std::remove_cvref_t<R>>,
      "pyb26: a Python callback or override cannot return a view type; "
      "return an owning value");
  static_assert(python_failure_sentinel_v<R>,
                "pyb26: Python callbacks and overrides require void, "
                "retained_object, or std::optional<T> to report failure "
                "without C++ exceptions");
  if (!result || cpython::current_exception(state)) [[unlikely]] {
    error.discard(result);
    return python_failure_return<R>();
  }
  if constexpr (std::is_void_v<R>) {
    error.discard(result);
    if (cpython::current_exception(state)) [[unlikely]]
      return python_failure_return<R>();
    return;
  } else {
    R value = load_python_call_result<R>(state, result, error, message);
    error.restore();
    error.discard(result);
    if (cpython::current_exception(state)) [[unlikely]] {
      error.capture();
      return python_failure_return<R>();
    }
    return value;
  }
}

} // namespace pyb26::detail
