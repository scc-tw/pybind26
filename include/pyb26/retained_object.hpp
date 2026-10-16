#pragma once

#include "detail/lifetime.hpp"
#include "detail/object.hpp"

namespace pyb26 {

class retained_object {
  detail::python_owner_ref owner_;

  explicit retained_object(detail::python_owner_ref owner) noexcept
      : owner_(std::move(owner)) {}

public:
  static constexpr bool interpreter_retained = true;
  retained_object() noexcept = default;
  retained_object(const retained_object &) = delete;
  retained_object &operator=(const retained_object &) = delete;
  retained_object(retained_object &&) noexcept = default;
  retained_object &operator=(retained_object &&) noexcept = default;

  static retained_object borrow(PyObject *object) noexcept {
    return retained_object(detail::python_owner_ref(object));
  }
  static retained_object steal(PyObject *object) noexcept {
    return retained_object(detail::python_owner_ref::steal(object));
  }
  PyObject *ptr() const noexcept { return owner_.get(); }
  PyObject *release() noexcept { return owner_.release(); }
  void reset() noexcept { owner_.reset(); }
  const detail::python_owner_context &context() const noexcept {
    return owner_.context();
  }
  explicit operator bool() const noexcept { return ptr() != nullptr; }
};

template <> struct type_caster<retained_object> {
  static constexpr caster_effects effects{true, false, false, true};
  retained_object value;
  bool load(PyObject *object) noexcept {
    value = retained_object::borrow(object);
    return static_cast<bool>(value);
  }
  template <class A> decltype(auto) as() noexcept {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(value);
    else
      return retained_object(std::move(value));
  }
  static PyObject *cast(retained_object value) noexcept {
    auto *state = detail::cpython::thread_state();
    if (value.context().interpreter &&
        state->interp != value.context().interpreter) {
      if (!PyErr_Occurred())
        PyErr_SetString(
            PyExc_RuntimeError,
            value ? "retained Python owner belongs to another interpreter"
                  : "retained Python owner is empty or closed");
      return nullptr;
    }
    PyObject *result = value.release();
    if (!result && !PyErr_Occurred())
      PyErr_SetString(PyExc_RuntimeError,
                      "retained Python owner is empty or closed");
    return result;
  }
};

namespace detail {
template <> struct py_hint<retained_object> {
  static void append(std::string &result) { result += "typing.Any"; }
};
} // namespace detail
} // namespace pyb26
