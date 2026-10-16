#pragma once

#include "dispatch.hpp"

#include <functional>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pyb26::detail {

int add_owned_getset(PyTypeObject *, const char *, const char *, getter, setter,
                     void *, void (*)(void *) noexcept) noexcept;

template <class F> struct property_signature {
  using type = functor_sig_t<F>;
};
template <class R, class... A> struct property_signature<R (*)(A...) noexcept> {
  using type = R(A...);
};
template <class Sig> struct property_args;
template <class R, class... A> struct property_args<R(A...)> {
  using result = R;
  using args = std::tuple<A...>;
};

template <class C, class A> decltype(auto) property_self(C *self) noexcept {
  static_assert(
      (std::is_pointer_v<A> || std::is_lvalue_reference_v<A>) &&
          std::is_same_v<std::remove_cv_t<
                             std::remove_pointer_t<std::remove_reference_t<A>>>,
                         C>,
      "property callable first argument must be C&, const C&, C* or const C*");
  if constexpr (std::is_pointer_v<A>)
    return static_cast<A>(self);
  else
    return static_cast<A>(*self);
}

template <class T>
inline constexpr bool accessor_inert_value =
    std::is_reference_v<T> || std::is_trivially_destructible_v<T> ||
    std::is_same_v<std::remove_cvref_t<T>, std::string>;

template <class T, class... Extras>
using accessor_get_error =
    std::conditional_t<accessor_inert_value<T> &&
                           !has_profile_hook_v<Extras...> &&
                           std::is_void_v<find_call_guard_t<Extras...>>,
                       python_error_probe, python_error_latch>;

template <class T, class... Extras>
using accessor_set_error = std::conditional_t<
    accessor_inert_value<T> &&
        (std::is_trivially_destructible_v<caster_for<T>> ||
         std::is_same_v<std::remove_cvref_t<T>, std::string>) &&
        accessor_inert_value<
            decltype(std::declval<caster_for<T> &>().template as<T>())> &&
        !has_profile_hook_v<Extras...> &&
        std::is_void_v<find_call_guard_t<Extras...>>,
    python_error_probe, python_error_latch>;

template <class Error> struct accessor_checkpoint {
  Error &error;
  ~accessor_checkpoint() {
    if constexpr (std::is_same_v<Error, python_error_latch>)
      error.capture();
  }
};

template <class Error, class F>
PYB26_ALWAYS_INLINE PyObject *accessor_get_transaction(F &&invoke) noexcept {
  Error error;
  PyObject *result = invoke(error);
  if (!result && !error.capture())
    PyErr_SetString(PyExc_SystemError,
                    "accessor failed without a Python error");
  if (error.capture()) {
    error.discard(result);
    if constexpr (requires { error.restore(); })
      error.restore();
    return nullptr;
  }
  return result;
}

template <class Error, class F>
PYB26_ALWAYS_INLINE int accessor_set_transaction(F &&invoke) noexcept {
  Error error;
  const int status = invoke(error);
  if (status != 0 && !error.capture())
    PyErr_SetString(PyExc_RuntimeError,
                    "property setter failed without a Python error");
  if (error.capture()) {
    if constexpr (requires { error.restore(); })
      error.restore();
    return -1;
  }
  return status;
}

template <class Guard, class Error, class F>
PYB26_ALWAYS_INLINE bool accessor_guarded(Error &error, F &&invoke) noexcept {
  invocation_guarded<Guard>(error, std::forward<F>(invoke));
  return !error.capture();
}

template <rv_policy Policy, class Guard, class Error, class F>
PYB26_ALWAYS_INLINE PyObject *accessor_get_native(Error &error, PyObject *self,
                                                  F &&invoke) noexcept {
  if constexpr (std::is_void_v<Guard>) {
    decltype(auto) value = invoke();
    accessor_checkpoint<Error> checkpoint{error};
    return checked_cast_return_rvp<Policy>(std::forward<decltype(value)>(value),
                                           self, error);
  } else {
    using R = decltype(invoke());
    static_assert(!(guard_releases_python_v<Guard> &&
                    guard_argument_python_owner_v<R> &&
                    !std::is_reference_v<R>),
                  "release guards must return native values or references");
    invocation_return_storage<R> value;
    accessor_checkpoint<Error> checkpoint{error};
    accessor_guarded<Guard>(error, [&] { value.construct(invoke); });
    if (!value.has_value())
      return nullptr;
    return checked_cast_return_rvp<Policy>(value.get(), self, error);
  }
}

template <class Guard, class Error, class F>
PYB26_ALWAYS_INLINE bool accessor_set_native(Error &error,
                                             F &&invoke) noexcept {
  using R = decltype(invoke());
  static_assert(std::is_void_v<R> || std::is_same_v<R, int>,
                "property setters return void or checked int status");
  if constexpr (std::is_void_v<R>) {
    return accessor_guarded<Guard>(error, std::forward<F>(invoke));
  } else {
    int status = 0;
    if (!accessor_guarded<Guard>(error, [&] { status = invoke(); }))
      return false;
    if (status != 0) {
      PyErr_SetString(PyExc_RuntimeError,
                      "property setter failed without a Python error");
      error.capture();
      return false;
    }
    return true;
  }
}

template <class C, class F, rv_policy Policy> struct callable_property {
  using traits = property_args<typename property_signature<F>::type>;
  static_assert(std::tuple_size_v<typename traits::args> == 1,
                "property getter must accept only self");
  using Self = std::tuple_element_t<0, typename traits::args>;

  static PyObject *get_with(PyObject *self, F &fn) noexcept {
    C *native = cast_to<C>(self);
    if (!native) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError, "not a valid instance");
      return nullptr;
    }
    using R = typename traits::result;
    return accessor_get_transaction<accessor_get_error<R>>([&](auto &error) {
      return accessor_get_native<Policy, void>(
          error, self, [&]() -> decltype(auto) {
            return std::invoke(fn, property_self<C, Self>(native));
          });
    });
  }
  static PyObject *get(PyObject *self, void *closure) noexcept {
    return get_with(self, *static_cast<F *>(closure));
  }
  static void free(void *closure) noexcept { delete static_cast<F *>(closure); }
};

template <class C, class F, class S, rv_policy Policy> struct property_pair {
  F getter;
  S setter;
  using traits = property_args<typename property_signature<S>::type>;
  static_assert(std::tuple_size_v<typename traits::args> == 2,
                "property setter must accept self and value");
  using Self = std::tuple_element_t<0, typename traits::args>;
  using Value = std::tuple_element_t<1, typename traits::args>;
  using Result = typename traits::result;
  static_assert(std::is_void_v<Result> || std::is_same_v<Result, int>,
                "property setter returns void or checked int status (0/-1)");

  static PyObject *get(PyObject *self, void *closure) noexcept {
    return callable_property<C, F, Policy>::get_with(
        self, static_cast<property_pair *>(closure)->getter);
  }
  static int set(PyObject *self, PyObject *value, void *closure) noexcept {
    if (!value) {
      PyErr_SetString(PyExc_TypeError, "cannot delete a property");
      return -1;
    }
    C *native = cast_to<C>(self);
    if (!native) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError, "not a valid instance");
      return -1;
    }
    reject_lease_escape<Value>();
    return accessor_set_transaction<accessor_set_error<Value>>(
        [&](auto &error) {
          using Error = std::remove_reference_t<decltype(error)>;
          caster_for<Value> c;
          accessor_checkpoint<Error> caster_checkpoint{error};
          if constexpr (!native_dispatch_argument<Value>)
            if (error.capture())
              return -1;
          if (!c.load(value)) {
            if (!PyErr_Occurred())
              PyErr_SetString(PyExc_TypeError, "incompatible attribute type");
            return -1;
          }
          if (error.capture())
            return -1;
          auto prepared = prepare_input_argument<Value>(c);
          accessor_checkpoint<Error> prepared_checkpoint{error};
          if (error.capture())
            return -1;
          S &fn = static_cast<property_pair *>(closure)->setter;
          return accessor_set_native<void>(
                     error,
                     [&]() -> decltype(auto) {
                       return std::invoke(fn, property_self<C, Self>(native),
                                          prepared.template as<Value>());
                     })
                     ? 0
                     : -1;
        });
  }
  static void free(void *closure) noexcept {
    delete static_cast<property_pair *>(closure);
  }
};

template <class C, rv_policy Policy, class F>
int register_callable_property_readonly(PyTypeObject *type, const char *name,
                                        const char *doc, F &&fn) noexcept {
  using D = std::decay_t<F>;
  using Adapter = callable_property<C, D, Policy>;
  D *capture = new (std::nothrow) D(std::forward<F>(fn));
  if (!capture) {
    PyErr_NoMemory();
    return -1;
  }
  return add_owned_getset(type, name, doc, Adapter::get, nullptr, capture,
                          Adapter::free);
}

template <class C, rv_policy Policy, class F, class S>
int register_callable_property(PyTypeObject *type, const char *name,
                               const char *doc, F &&fn, S &&setter) noexcept {
  using Pair = property_pair<C, std::decay_t<F>, std::decay_t<S>, Policy>;
  Pair *capture =
      new (std::nothrow) Pair{std::forward<F>(fn), std::forward<S>(setter)};
  if (!capture) {
    PyErr_NoMemory();
    return -1;
  }
  return add_owned_getset(type, name, doc, Pair::get, Pair::set, capture,
                          Pair::free);
}

} // namespace pyb26::detail
