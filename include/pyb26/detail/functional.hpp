#pragma once

#include "callback_owner.hpp"
#include "caster.hpp"
#include "common.hpp"
#include "gil.hpp"
#include "lifetime.hpp"
#include "python_call.hpp"
#include "python_failure.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

namespace pyb26 {
namespace detail {

template <class R, class... Args> struct py_function_call {
  callback_owner_ref owner;

  static constexpr std::size_t N = sizeof...(Args);

  R operator()(Args &&...args) const {
    auto *callable_owner = owner.get();
    auto invocation_owner = callback_owner_ref::retain(callable_owner);
    callback_invocation_scope invocation(callable_owner->callable.context(),
                                         &py_callable_owner::store_error,
                                         callable_owner);
    if (!invocation) {
      if (invocation.thread_state())
        return python_failure_return<R>();
      callable_owner->native_failure.fetch_or(1, std::memory_order_release);
      return python_failure_value<R>();
    }
    auto *state = invocation.thread_state();
    callback_error_scope error{state};
    if (cpython::current_exception(state)) [[unlikely]]
      return python_failure_return<R>();

    PyObject *storage[N + 1] = {};
    PyObject **vargs = storage + 1;

    if (!build_python_call_arguments(state, vargs,
                                     std::index_sequence_for<Args...>{},
                                     std::forward<Args>(args)...)) {
      clear_python_call_arguments<N>(vargs, error);
      if (!cpython::current_exception(state))
        PyErr_SetString(PyExc_TypeError,
                        "pyb26: could not convert C++ argument to Python "
                        "for std::function callback");
      return python_failure_return<R>();
    }

    PyObject *result =
        PyObject_Vectorcall(callable_owner->callable.get(), vargs,
                            N | PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr);

    clear_python_call_arguments<N>(vargs, error);
    return finish_python_call<R>(
        state, result, error,
        "pyb26: could not convert Python result of std::function callback "
        "to the C++ return type");
  }
};

} // namespace detail
} // namespace pyb26

#if defined(__GLIBCXX__)
namespace std {
// The owner stays fixed; this target must never contain address-dependent
// state. libstdc++ transfers the handle without destroying the relocated
// source.
// https://gcc.gnu.org/onlinedocs/libstdc++/latest-doxygen/a00710_source.html
template <class R, class... Args>
struct __is_location_invariant<pyb26::detail::py_function_call<R, Args...>>
    : true_type {
  using target = pyb26::detail::py_function_call<R, Args...>;
  using owner_handle = pyb26::detail::callback_owner_ref;
  static_assert(sizeof(target) == sizeof(owner_handle));
  static_assert(alignof(target) == alignof(owner_handle));
  static_assert(is_nothrow_copy_constructible_v<target>);
  static_assert(is_nothrow_move_constructible_v<target>);
};
} // namespace std
#endif

namespace pyb26 {

template <class R, class... Args>
struct type_caster<std::function<R(Args...)>> {
  static constexpr caster_effects effects{true, false, false, true};
  static_assert(
      !std::is_reference_v<R>,
      "pyb26: a std::function callback cannot return a reference: the "
      "value converted from the Python result lives only inside the "
      "call, so the reference would dangle. Return by value.");
  static_assert(
      !caster_yields_view_v<std::remove_cvref_t<R>>,
      "pyb26: a std::function callback cannot return a view type "
      "(std::string_view / const char* / PyObject*): the Python result "
      "it points into is released when the callback returns. Return an "
      "owning Python object or std::optional<std::string>.");
  static_assert(detail::python_failure_sentinel_v<R>,
                "pyb26: Python callbacks require void, retained_object, or "
                "std::optional<T> to report failure without C++ exceptions");
  std::optional<detail::owner_admission> binding_admission_;
  detail::callback_binding_owner owner_;
  std::function<R(Args...)> value_;

  bool restore_error() noexcept {
    bool local_target = false;
    if (owner_) {
      const auto *local =
          value_
              ? value_.template target<detail::py_function_call<R, Args...>>()
              : nullptr;
      local_target = local && local->owner.get() == owner_.get();
      if (owner_->has_other_targets(local_target))
        owner_->callable.register_escape(*binding_admission_);
      owner_->restore_error();
    }
    return local_target;
  }
  void finish_binding() noexcept {
    const bool local_target = restore_error();
    if (!owner_)
      return;
    auto *state = detail::cpython::attached_thread_state();
    auto finish_owner = [&] {
      if (state && state->interp == owner_->callable.context().interpreter)
        owner_.finish_admitted(*binding_admission_, state);
      else
        owner_.reset();
    };
    if (local_target || !value_ || !state) {
      value_ = {};
      finish_owner();
    } else {
      detail::callback_error_scope error{state};
      error.capture();
      value_ = {};
      error.capture();
      finish_owner();
    }
  }
  ~type_caster() { finish_binding(); }

  PYB26_HOT bool load(PyObject *o) noexcept {
    if (owner_) {
      finish_binding();
      binding_admission_.reset();
    }
    if (!o) [[unlikely]]
      return false;
    auto *state = detail::cpython::thread_state();
    if (detail::cpython::current_exception(state)) [[unlikely]]
      return false;
    if (o == Py_None) [[unlikely]]
      return false;
    if (!PyCallable_Check(o)) [[unlikely]]
      return false;
    auto context = detail::capture_attached_python_owner_context(state);
    binding_admission_.emplace(context);
    if (!*binding_admission_) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_RuntimeError,
                        "callback interpreter is closing or closed");
      return false;
    }
    owner_.reset(new detail::py_callable_owner(o, std::move(context),
                                               *binding_admission_));
    if (!owner_->callable || detail::cpython::current_exception(state))
      return false;
    value_ = detail::py_function_call<R, Args...>{
        detail::callback_owner_ref::initial(owner_.get())};
    return true;
  }

  template <class A> decltype(auto) as() {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(value_);
    else
      return A(std::move(value_));
  }
};

template <class R, class... A>
invocation_context
callback_context(const std::function<R(A...)> &callback) noexcept {
  const auto *target =
      callback.template target<detail::py_function_call<R, A...>>();
  return target ? target->owner->callable.context() : invocation_context{};
}

template <class R, class... A>
bool restore_callback_error(const std::function<R(A...)> &callback) noexcept {
  const auto *target =
      callback.template target<detail::py_function_call<R, A...>>();
  if (!target)
    return true;
  auto *current = detail::cpython::attached_thread_state();
  if (!current)
    return false;
  if (current->interp != target->owner->callable.context().interpreter) {
    PyErr_SetString(PyExc_RuntimeError,
                    "callback error belongs to another interpreter");
    return false;
  }
  target->owner->restore_error();
  return !PyErr_Occurred();
}

} // namespace pyb26
