#pragma once

#include "lifetime.hpp"

#include <concepts>
#include <type_traits>

namespace pyb26::detail {

template <class C>
inline constexpr bool borrowed_return_owner_v = [] {
  using D = std::remove_cvref_t<C>;
  if constexpr (requires { typename D::borrowed_owner_implementation; })
    return std::same_as<D, typename D::borrowed_owner_implementation>;
  return false;
}();

template <bool Pin> class return_owner_pin {
public:
  explicit return_owner_pin(PyObject *) noexcept {}
  return_owner_pin(const return_owner_pin &) = delete;
};

template <> class return_owner_pin<true> {
  PyThreadState *state_;
  PyObject *owner_;

public:
  explicit return_owner_pin(
      PyObject *owner, PyThreadState *state = cpython::thread_state()) noexcept
      : state_(state), owner_(Py_XNewRef(owner)) {}
  return_owner_pin(const return_owner_pin &) = delete;
  ~return_owner_pin() {
    callback_error_scope error{state_};
    error.discard(owner_);
  }
};

} // namespace pyb26::detail
