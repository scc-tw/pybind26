#pragma once

#include "caster.hpp"
#include "dispatch.hpp"
#include "holders.hpp"
#include "instance.hpp"
#include "module.hpp"
#include "property.hpp"
#include "trampoline.hpp"

#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace pyb26 {

namespace detail {

template <class M> struct field_traits;
template <class C, class FT> struct field_traits<FT C::*> {
  using cls = C;
  using type = FT;
};

template <class F> struct frontend_signature {
  using type =
      typename member_sig<decltype(&std::remove_cvref_t<F>::operator())>::sig;
};
template <class R, class... A> struct frontend_signature<R (*)(A...)> {
  using type = R(A...);
};
template <class R, class... A> struct frontend_signature<R (*)(A...) noexcept> {
  using type = R(A...);
};
template <class F>
using frontend_signature_t =
    typename frontend_signature<std::remove_cvref_t<F>>::type;

template <class Sig> struct self_first_signature;
template <class R, class Self, class... A>
struct self_first_signature<R(Self, A...)> {
  using self = Self;
  using cls =
      std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<Self>>>;
  using sig = R(A...);
  using ret = R;
};

template <auto F, bool Member = std::is_member_function_pointer_v<decltype(F)>>
struct property_traits;
template <auto F> struct property_traits<F, true> : member_sig<decltype(F)> {};
template <auto F>
struct property_traits<F, false>
    : self_first_signature<frontend_signature_t<decltype(F)>> {};

template <class Sig> struct unary_signature;
template <class R, class V> struct unary_signature<R(V)> {
  using arg = V;
  using ret = R;
};

template <class Self, class C>
PYB26_ALWAYS_INLINE decltype(auto) frontend_self(C *obj) noexcept {
  if constexpr (std::is_pointer_v<std::remove_reference_t<Self>>)
    return static_cast<Self>(obj);
  else {
    static_assert(
        std::is_reference_v<Self>,
        "pyb26: self-first callables must accept self by reference or pointer");
    return static_cast<Self>(*obj);
  }
}

template <auto F, class C, class... A>
PYB26_ALWAYS_INLINE decltype(auto) invoke_property(C *obj, A &&...a) {
  if constexpr (std::is_member_function_pointer_v<decltype(F)>)
    return std::invoke(F, obj, std::forward<A>(a)...);
  else {
    using Self = typename property_traits<F>::self;
    return std::invoke(F, frontend_self<Self>(obj), std::forward<A>(a)...);
  }
}

inline const char *property_label(void *closure) noexcept {
  return closure ? static_cast<property_context *>(closure)->profile_label
                 : nullptr;
}

template <class Extra>
bool apply_getter_keep_alive(PyObject *result, PyObject *self) noexcept {
  using E = std::remove_cvref_t<Extra>;
  if constexpr (!is_keep_alive_v<E>)
    return true;
  else if constexpr (E::nurse > 1 || E::patient > 1)
    return true;
  else {
    PyObject *slots[] = {self};
    return apply_one_keep_alive<E>(result, slots, 1);
  }
}

template <class Extra>
bool apply_setter_keep_alive(PyObject *self, PyObject *value) noexcept {
  using E = std::remove_cvref_t<Extra>;
  if constexpr (!is_keep_alive_v<E>)
    return true;
  else if constexpr (E::nurse == 0 || E::patient == 0)
    return true;
  else {
    PyObject *slots[] = {self, value};
    return apply_one_keep_alive<E>(nullptr, slots, 2);
  }
}

template <class Extra>
bool validate_getter_keep_alive(PyObject *self) noexcept {
  using E = std::remove_cvref_t<Extra>;
  if constexpr (!is_keep_alive_v<E>)
    return true;
  else if constexpr (E::nurse > 1 || E::patient > 1)
    return true;
  else {
    PyObject *slots[] = {self};
    return validate_keep_alive<E>(slots, 1);
  }
}

template <class Extra>
bool validate_setter_keep_alive(PyObject *self, PyObject *value) noexcept {
  using E = std::remove_cvref_t<Extra>;
  if constexpr (!is_keep_alive_v<E>)
    return true;
  else if constexpr (E::nurse == 0 || E::patient == 0)
    return true;
  else {
    PyObject *slots[] = {self, value};
    return validate_keep_alive<E>(slots, 2);
  }
}

template <bool Writable, class Extra>
consteval bool valid_property_keep_alive() {
  using E = std::remove_cvref_t<Extra>;
  if constexpr (!is_keep_alive_v<E>)
    return true;
  else
    return E::nurse <= (Writable ? 2 : 1) && E::patient <= (Writable ? 2 : 1) &&
           !(E::nurse == 0 && E::patient == 2) &&
           !(E::nurse == 2 && E::patient == 0);
}

template <bool Writable, class... Extras>
consteval void validate_property_extras() {
  static_assert((!is_return_owner_v<std::remove_cvref_t<Extras>> && ...),
                "pyb26: property aliases use their receiver as owner");
  static_assert((valid_property_keep_alive<Writable, Extras>() && ...),
                "pyb26: property keep_alive slots are getter return=0, self=1, "
                "setter value=2; return and value belong to different calls");
  static_assert(((!std::is_same_v<std::remove_cvref_t<Extras>, arg> &&
                  !std::is_same_v<std::remove_cvref_t<Extras>, kw_only> &&
                  !std::is_same_v<std::remove_cvref_t<Extras>, pos_only>) &&
                 ...),
                "pyb26: properties do not accept function argument metadata");
}

template <class... Extras>
getset_spec make_getset_spec(PyTypeObject *type, const char *name, getter get,
                             setter set, Extras &&...extras) {
  getset_spec spec{name, doc_of(extras...), get, set};
  const std::size_t name_size = std::strlen(name) + 1;
  auto *owned_name = new (std::nothrow) char[name_size];
  if (!owned_name) {
    PyErr_NoMemory();
    return spec;
  }
  std::memcpy(owned_name, name, name_size);
  auto *context = new (std::nothrow) property_context{owned_name, nullptr};
  if (!context) {
    delete[] owned_name;
    PyErr_NoMemory();
    return spec;
  }
  callable_spec callable{};
  callable.name = name;
  fill_profile_label<Extras...>(callable, reinterpret_cast<PyObject *>(type));
  context->profile_label = callable.profile_label;
  spec.closure = context;
  spec.free_closure = [](void *p) noexcept {
    auto *c = static_cast<property_context *>(p);
    delete[] c->name;
    delete[] c->profile_label;
    delete c;
  };
  return spec;
}

template <class Error, class... Extras, class F>
PYB26_ALWAYS_INLINE PyObject *
accessor_get_execute(PyObject *self, void *closure, F &&invoke) noexcept {
  return accessor_get_transaction<Error>([&](auto &error) -> PyObject * {
    if (!(validate_getter_keep_alive<Extras>(self) && ...))
      return nullptr;
    profile_scope_t<Extras...> profile(property_label(closure));
    accessor_checkpoint<Error> checkpoint{error};
    if (error.capture())
      return nullptr;
    PyObject *result = invoke(error);
    if (error.capture()) {
      error.discard(result);
      return nullptr;
    }
    if (!result)
      return nullptr;
    if (!(apply_getter_keep_alive<Extras>(result, self) && ...)) {
      error.capture();
      error.discard(result);
      return nullptr;
    }
    return result;
  });
}

template <class V, class Error, class... Extras, class F>
PYB26_ALWAYS_INLINE int accessor_set_execute(PyObject *self, PyObject *value,
                                             void *closure,
                                             F &&invoke) noexcept {
  return accessor_set_transaction<Error>([&](auto &error) {
    caster_for<V> c;
    accessor_checkpoint<Error> caster_checkpoint{error};
    if constexpr (!native_dispatch_argument<V>)
      if (error.capture())
        return -1;
    if (!c.load(value)) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError, "incompatible attribute type");
      return -1;
    }
    if (error.capture())
      return -1;
    if (!(validate_setter_keep_alive<Extras>(self, value) && ...))
      return -1;
    profile_scope_t<Extras...> profile(property_label(closure));
    accessor_checkpoint<Error> profile_checkpoint{error};
    if (error.capture())
      return -1;
    auto prepared = prepare_input_argument<V>(c);
    accessor_checkpoint<Error> prepared_checkpoint{error};
    if (error.capture() || !invoke(error, prepared) || error.capture())
      return -1;
    return (apply_setter_keep_alive<Extras>(self, value) && ...) ? 0 : -1;
  });
}

template <auto FP, rv_policy Policy = rv_policy::automatic, class... Extras>
PyObject *field_get(PyObject *self, void *closure) noexcept {
  using C = typename field_traits<decltype(FP)>::cls;
  using FT = typename field_traits<decltype(FP)>::type;
  C *obj = cast_to<C>(self);
  if (!obj) [[unlikely]]
    return PyErr_SetString(PyExc_TypeError, "not a valid instance"), nullptr;
  static_assert(Policy != rv_policy::take_ownership,
                "pyb26: an embedded field cannot transfer ownership; use a "
                "pointer-returning property");
  using Guard = find_call_guard_t<Extras...>;
  using Error = accessor_get_error<FT &, Extras...>;
  return accessor_get_execute<Error, Extras...>(
      self, closure, [&](auto &error) {
        return accessor_get_native<Policy, Guard>(
            error, self, [&]() -> decltype(auto) { return (obj->*FP); });
      });
}

template <auto FP, class... Extras>
int field_set(PyObject *self, PyObject *val, void *closure) noexcept {
  using C = typename field_traits<decltype(FP)>::cls;
  using FT = typename field_traits<decltype(FP)>::type;
  reject_lease_escape<FT>();
  C *obj = cast_to<C>(self);
  if (!obj) [[unlikely]] {
    PyErr_SetString(PyExc_TypeError, "not a valid instance");
    return -1;
  }
  if (!val) [[unlikely]] {
    PyErr_SetString(PyExc_TypeError, "cannot delete a field");
    return -1;
  }
  using Guard = find_call_guard_t<Extras...>;
  static_assert(
      !(guard_releases_python_v<Guard> && guard_argument_python_owner_v<FT>));
  return accessor_set_execute<FT, accessor_set_error<FT, Extras...>, Extras...>(
      self, val, closure, [&](auto &error, auto &prepared) {
        return accessor_set_native<Guard>(
            error, [&] { obj->*FP = prepared.template as<FT>(); });
      });
}

template <class M> struct getter_traits;
template <class R, class C> struct getter_traits<R (C::*)()> {
  using cls = C;
  using ret = R;
};
template <class R, class C> struct getter_traits<R (C::*)() const> {
  using cls = C;
  using ret = R;
};
template <class R, class C> struct getter_traits<R (C::*)() noexcept> {
  using cls = C;
  using ret = R;
};
template <class R, class C> struct getter_traits<R (C::*)() const noexcept> {
  using cls = C;
  using ret = R;
};

template <auto Get, rv_policy Policy = rv_policy::automatic, class... Extras>
PyObject *prop_get(PyObject *self, void *closure) noexcept {
  using C = typename property_traits<Get>::cls;
  C *obj = cast_to<C>(self);
  if (!obj) [[unlikely]]
    return PyErr_SetString(PyExc_TypeError, "not a valid instance"), nullptr;
  using Guard = find_call_guard_t<Extras...>;
  using R = decltype(invoke_property<Get>(obj));
  return accessor_get_execute<accessor_get_error<R, Extras...>, Extras...>(
      self, closure, [&](auto &error) {
        return accessor_get_native<Policy, Guard>(
            error, self,
            [&]() -> decltype(auto) { return invoke_property<Get>(obj); });
      });
}

template <auto Set, class... Extras>
int prop_set(PyObject *self, PyObject *val, void *closure) noexcept {
  using C = typename property_traits<Set>::cls;
  using V = typename unary_signature<typename property_traits<Set>::sig>::arg;
  C *obj = cast_to<C>(self);
  if (!obj) [[unlikely]] {
    PyErr_SetString(PyExc_TypeError, "not a valid instance");
    return -1;
  }
  if (!val) [[unlikely]] {
    PyErr_SetString(PyExc_TypeError, "cannot delete a property");
    return -1;
  }
  using Guard = find_call_guard_t<Extras...>;
  static_assert(
      !(guard_releases_python_v<Guard> && guard_argument_python_owner_v<V>));
  return accessor_set_execute<V, accessor_set_error<V, Extras...>, Extras...>(
      self, val, closure, [&](auto &error, auto &prepared) {
        return accessor_set_native<Guard>(error, [&]() -> decltype(auto) {
          return invoke_property<Set>(obj, prepared.template as<V>());
        });
      });
}

template <class T> inline bool static_self_ok(PyObject *self) noexcept {
  if (!PyType_Check(self))
    return true;
  type_record *tr = resolve<T>();
  return tr &&
         PyType_IsSubtype(reinterpret_cast<PyTypeObject *>(self), tr->py_type);
}

PYB26_COLD inline PyObject *static_access_error(PyObject *self,
                                                void *closure) noexcept {
  const char *tn = PyType_Check(self)
                       ? reinterpret_cast<PyTypeObject *>(self)->tp_name
                       : Py_TYPE(self)->tp_name;
  PyErr_Format(PyExc_AttributeError,
               "type object '%.200s' has no attribute '%s'", tn,
               closure ? static_cast<property_context *>(closure)->name
                       : "<static property>");
  return nullptr;
}

template <class T, auto SP, rv_policy Policy = rv_policy::automatic,
          class... Extras>
PyObject *static_field_get(PyObject *self, void *closure) noexcept {
  if (!static_self_ok<T>(self)) [[unlikely]]
    return static_access_error(self, closure);
  using Guard = find_call_guard_t<Extras...>;
  using R = decltype((*SP));
  return accessor_get_execute<accessor_get_error<R, Extras...>, Extras...>(
      self, closure, [&](auto &error) {
        return accessor_get_native<Policy, Guard>(
            error, self, []() -> decltype(auto) { return (*SP); });
      });
}

template <class T, auto SP, class... Extras>
int static_field_set(PyObject *self, PyObject *val, void *closure) noexcept {
  using VT = std::remove_cvref_t<std::remove_pointer_t<decltype(SP)>>;
  reject_lease_escape<VT>();
  if (!static_self_ok<T>(self)) [[unlikely]] {
    static_access_error(self, closure);
    return -1;
  }
  if (!val) [[unlikely]] {
    PyErr_SetString(PyExc_TypeError, "cannot delete a static member");
    return -1;
  }
  using Guard = find_call_guard_t<Extras...>;
  static_assert(
      !(guard_releases_python_v<Guard> && guard_argument_python_owner_v<VT>));
  return accessor_set_execute<VT, accessor_set_error<VT, Extras...>, Extras...>(
      self, val, closure, [&](auto &error, auto &prepared) {
        return accessor_set_native<Guard>(
            error, [&] { *SP = prepared.template as<VT>(); });
      });
}

template <class Sig> struct static_property_signature;
template <class R, class... A> struct static_property_signature<R(A...)> {
  static constexpr std::size_t arity = sizeof...(A);
  using args = std::tuple<A...>;
  using result = R;
};

template <class T, auto Get, rv_policy Policy = rv_policy::automatic,
          class... Extras>
PyObject *static_prop_get(PyObject *self, void *closure) noexcept {
  if (!static_self_ok<T>(self)) [[unlikely]]
    return static_access_error(self, closure);
  using S = static_property_signature<frontend_signature_t<decltype(Get)>>;
  static_assert(S::arity <= 1, "pyb26: a static getter accepts zero args or "
                               "one Python owner/type argument");
  using Guard = find_call_guard_t<Extras...>;
  if constexpr (S::arity == 0) {
    using Error = accessor_get_error<typename S::result, Extras...>;
    return accessor_get_execute<Error, Extras...>(
        self, closure, [&](auto &error) {
          return accessor_get_native<Policy, Guard>(
              error, self, []() -> decltype(auto) { return std::invoke(Get); });
        });
  } else {
    using Owner = std::tuple_element_t<0, typename S::args>;
    static_assert(!guard_releases_python_v<Guard>,
                  "pyb26: static Python owner callbacks must remain attached");
    return accessor_get_execute<python_error_latch, Extras...>(
        self, closure, [&](auto &error) -> PyObject * {
          caster_for<Owner> c;
          accessor_checkpoint<python_error_latch> caster_checkpoint{error};
          if constexpr (!native_dispatch_argument<Owner>)
            if (error.capture())
              return nullptr;
          PyObject *owner = PyType_Check(self)
                                ? self
                                : reinterpret_cast<PyObject *>(Py_TYPE(self));
          if (!c.load(owner)) {
            if (!PyErr_Occurred())
              PyErr_SetString(PyExc_TypeError,
                              "incompatible static property owner");
            return nullptr;
          }
          if (error.capture())
            return nullptr;
          auto prepared = prepare_input_argument<Owner>(c);
          accessor_checkpoint<python_error_latch> prepared_checkpoint{error};
          if (error.capture())
            return nullptr;
          return accessor_get_native<Policy, Guard>(
              error, self, [&]() -> decltype(auto) {
                return std::invoke(Get, prepared.template as<Owner>());
              });
        });
  }
}

template <class T, auto Set, class... Extras>
int static_prop_set(PyObject *self, PyObject *val, void *closure) noexcept {
  using S = static_property_signature<frontend_signature_t<decltype(Set)>>;
  static_assert(
      S::arity == 1 || S::arity == 2,
      "pyb26: a static setter accepts value or (Python owner/type, value)");
  using V = std::tuple_element_t<S::arity - 1, typename S::args>;
  if (!static_self_ok<T>(self)) [[unlikely]] {
    static_access_error(self, closure);
    return -1;
  }
  if (!val) [[unlikely]] {
    PyErr_SetString(PyExc_TypeError, "cannot delete a static property");
    return -1;
  }
  using Guard = find_call_guard_t<Extras...>;
  static_assert(
      !(guard_releases_python_v<Guard> && guard_argument_python_owner_v<V>));
  if constexpr (S::arity == 1) {
    return accessor_set_execute<V, accessor_set_error<V, Extras...>, Extras...>(
        self, val, closure, [&](auto &error, auto &prepared) {
          return accessor_set_native<Guard>(error, [&]() -> decltype(auto) {
            return std::invoke(Set, prepared.template as<V>());
          });
        });
  } else {
    using Owner = std::tuple_element_t<0, typename S::args>;
    static_assert(!guard_releases_python_v<Guard>,
                  "pyb26: static Python owner callbacks must remain attached");
    return accessor_set_execute<V, python_error_latch, Extras...>(
        self, val, closure, [&](auto &error, auto &prepared) {
          caster_for<Owner> owner_caster;
          accessor_checkpoint<python_error_latch> caster_checkpoint{error};
          if constexpr (!native_dispatch_argument<Owner>)
            if (error.capture())
              return false;
          PyObject *owner = PyType_Check(self)
                                ? self
                                : reinterpret_cast<PyObject *>(Py_TYPE(self));
          if (!owner_caster.load(owner)) {
            if (!PyErr_Occurred())
              PyErr_SetString(PyExc_TypeError,
                              "incompatible static property owner");
            return false;
          }
          if (error.capture())
            return false;
          auto owner_prepared = prepare_input_argument<Owner>(owner_caster);
          accessor_checkpoint<python_error_latch> prepared_checkpoint{error};
          if (error.capture())
            return false;
          return accessor_set_native<Guard>(error, [&]() -> decltype(auto) {
            return std::invoke(Set, owner_prepared.template as<Owner>(),
                               prepared.template as<V>());
          });
        });
  }
}

enum class constructor_outcome {
  empty,
  constructed,
  no_memory,
  null_owner,
  unavailable_native,
  unavailable_alias
};

template <class T, class Native, class Signature, class... Extras>
struct constructor_pipeline;

template <class T, class Native, class... A, class... Extras>
struct constructor_pipeline<T, Native, void(A...), Extras...> {
  static_assert(return_owner_slot_v<Extras...> == 0,
                "pyb26: constructors do not accept return_owner");
  static constexpr int arity = sizeof...(A);
  using Arguments = invocation_arguments<A...>;
  using Guard = find_call_guard_t<Extras...>;
  using Error =
      std::conditional_t<Arguments::cleanup || !std::is_void_v<Guard> ||
                             has_profile_hook_v<Extras...>,
                         python_error_latch, python_error_probe>;
  static_assert(!(guard_releases_python_v<Guard> &&
                  ((guard_argument_python_owner_v<A> || is_args_v<A>) || ...)));
  static_assert(
      !guard_releases_python_v<Guard> ||
          (input_argument_plan<A, invocation_caster_t<A>>::parameter_inert &&
           ...),
      "pyb26: release guard requires inert native parameter construction and "
      "destruction");

  template <class Construct, class Finalize>
  PYB26_ALWAYS_INLINE static int invoke(PyObject *self, PyObject *const *args,
                                        Py_ssize_t off, const ctor_spec &spec,
                                        Construct &&construct,
                                        Finalize &&finalize) noexcept {
    argument_owner_admission<(caster_contract<A>.requires_admission || ...)>
        admission;
    if (!admission.acquire())
      return -1;
    Error error;
    constructor_outcome outcome = constructor_outcome::empty;
    PyObject *slots[arity + 1] = {self};
    if constexpr (has_keep_alive_v<Extras...>)
      for (int i = 0; i < arity; ++i)
        slots[i + 1] = args[i];
    auto execute = [&](auto &&...values) -> int {
      if (!validate_keep_alive<Extras...>(slots, arity + 1)) {
        error.capture();
        return -1;
      }
      profile_scope_t<Extras...> profile(spec.profile_label);
      invocation_checkpoint<Error, has_profile_hook_v<Extras...>> checkpoint{
          error};
      if constexpr (has_profile_hook_v<Extras...>)
        if (error.capture()) [[unlikely]]
          return -1;
      invocation_guarded<Guard>(error, [&] {
        outcome = construct(std::forward<decltype(values)>(values)...);
      });
      if (error.capture()) [[unlikely]]
        return -1;
      if constexpr (!std::is_same_v<T, Native>) {
        finalize();
        if (error.capture()) [[unlikely]]
          return -1;
      }
      if (outcome == constructor_outcome::no_memory) [[unlikely]] {
        PyErr_NoMemory();
        error.capture();
        return -1;
      }
      if (outcome == constructor_outcome::null_owner) [[unlikely]] {
        PyErr_SetString(PyExc_TypeError,
                        "constructor factory returned a null owning pointer");
        error.capture();
        return -1;
      }
      if (outcome == constructor_outcome::unavailable_native ||
          outcome == constructor_outcome::unavailable_alias) [[unlikely]] {
        PyErr_SetString(
            PyExc_TypeError,
            outcome == constructor_outcome::unavailable_native
                ? "native constructor is unavailable; use a Python subclass or "
                  "alias_init"
                : "trampoline constructor is unavailable for this signature");
        error.capture();
        return -1;
      }
      return 0;
    };
    auto load = [&]<std::size_t I>(auto &cast) { return cast.load(args[I]); };
    auto prepare = [&](auto &cast) {
      return Arguments::prepare(cast, error, execute);
    };
    const int status = Arguments::load(load, error, prepare);
    if (error.capture()) [[unlikely]] {
      if (outcome == constructor_outcome::constructed)
        rollback_constructor<T, Native>(self, off);
      if constexpr (requires { error.restore(); })
        error.restore();
      return -1;
    }
    if constexpr (has_keep_alive_v<Extras...>)
      if (status == 0 &&
          !commit_constructor_keep_alive<Extras...>(slots, arity + 1))
        return rollback_constructor<T, Native>(self, off);
    return status;
  }
};

template <class T, class Native, class Holder, class Signature, class... Extras>
struct constructor_binding;

template <class T, class Native, class Holder, class... A, class... Extras>
struct constructor_binding<T, Native, Holder, void(A...), Extras...> {
  static constexpr int arity = sizeof...(A);
  static constexpr bool has_alias = !std::is_same_v<T, Native>;
  static constexpr bool force_alias =
      (std::is_same_v<std::remove_cvref_t<Extras>, alias_init> || ...);
  static_assert(!force_alias || has_alias,
                "pyb26: alias_init requires a registered trampoline");
  static_assert(std::is_constructible_v<T, A...> ||
                    std::is_constructible_v<Native, A...>,
                "pyb26: neither native nor trampoline constructor accepts this "
                "signature");
  static_assert(policy_of<Extras...>() == rv_policy::automatic,
                "pyb26: direct constructors own their native value/holder; "
                "return-value policies do not apply");

  static int invoke(PyObject *self, PyObject *const *args, Py_ssize_t off,
                    const ctor_spec &spec) noexcept {
    Native *native = nullptr;
    bool alias = force_alias;
    if constexpr (has_alias && !force_alias) {
      auto *record = resolve_for_instance<T>(self);
      if (!record) [[unlikely]] {
        error_unregistered(typeid(T).name());
        return -1;
      }
      alias = Py_TYPE(self) != record->py_type;
    }
    auto construct = [&]<class C>(auto &&...values) {
      if constexpr (!std::is_constructible_v<C, A...>) {
        return std::is_same_v<C, T> ? constructor_outcome::unavailable_native
                                    : constructor_outcome::unavailable_alias;
      } else {
        auto *in = reinterpret_cast<instance *>(self);
        void *mem = reinterpret_cast<char *>(self) + off;
        constexpr unsigned char identity =
            std::is_same_v<C, T> ? 0 : inst_alias;
        C *value;
        if constexpr (std::is_same_v<Holder, std::shared_ptr<T>>) {
          auto owner =
              std::make_shared<C>(std::forward<decltype(values)>(values)...);
          value = owner.get();
          arm_block<T, st_shared>(in, mem, std::move(owner), identity);
        } else if constexpr (std::is_same_v<Holder, std::unique_ptr<T>>) {
          auto owner =
              std::make_unique<C>(std::forward<decltype(values)>(values)...);
          value = owner.get();
          arm_block<T, st_unique>(in, mem, std::move(owner), identity);
        } else if constexpr (!native_inline_v<C>) {
          void *allocation = allocate_native_storage<C>();
          if (!allocation)
            return constructor_outcome::no_memory;
          value =
              ::new (allocation) C(std::forward<decltype(values)>(values)...);
          arm_boxed<T>(in, mem, value, identity);
        } else {
          value = ::new (mem) C(std::forward<decltype(values)>(values)...);
          in->flags.set_lifecycle(inst_constructed | inst_owned |
                                  inst_initializing | identity);
        }
        if constexpr (has_alias && std::is_same_v<C, Native>)
          native = value;
        return constructor_outcome::constructed;
      }
    };
    return constructor_pipeline<T, Native, void(A...), Extras...>::invoke(
        self, args, off, spec,
        [&](auto &&...values) {
          if constexpr (has_alias)
            if (alias)
              return construct.template operator()<Native>(
                  std::forward<decltype(values)>(values)...);
          return construct.template operator()<T>(
              std::forward<decltype(values)>(values)...);
        },
        [&] {
          if constexpr (has_alias)
            if (native)
              set_trampoline_self<Native>(native, self);
        });
  }
};

template <class P> struct factory_holder {
  static constexpr storage_kind kind = st_inline;
};
template <class T> struct factory_holder<T *> {
  static constexpr storage_kind kind = st_pointer;
};
template <class T> struct factory_holder<std::shared_ptr<T>> {
  static constexpr storage_kind kind = st_shared;
};
template <class T> struct factory_holder<std::unique_ptr<T>> {
  static constexpr storage_kind kind = st_unique;
};

template <class T, auto Factory, class Signature, class... Extras>
struct factory_constructor_binding;

template <class T, auto Factory, class R, class... A, class... Extras>
struct factory_constructor_binding<T, Factory, R(A...), Extras...> {
  static constexpr int arity = sizeof...(A);
  static_assert(!std::is_reference_v<R>,
                "pyb26: constructor factories must return a value or an owning "
                "holder/pointer");
  static constexpr storage_kind kind =
      factory_holder<std::remove_cvref_t<R>>::kind;
  static_assert(policy_of<Extras...>() == rv_policy::automatic ||
                    (kind == st_pointer &&
                     policy_of<Extras...>() == rv_policy::take_ownership),
                "pyb26: factories materialize an owning value/holder; only raw "
                "owning pointers accept take_ownership");

  static int invoke(PyObject *self, PyObject *const *args, Py_ssize_t off,
                    const ctor_spec &spec) noexcept {
    return constructor_pipeline<T, T, void(A...), Extras...>::invoke(
        self, args, off, spec,
        [&](auto &&...values) {
          auto *in = reinterpret_cast<instance *>(self);
          void *mem = reinterpret_cast<char *>(self) + off;
          if constexpr (kind == st_inline) {
            static_assert(
                std::is_same_v<std::remove_cvref_t<R>, T>,
                "pyb26: a value-returning constructor factory must return "
                "the bound type");
            if constexpr (native_inline_v<T>) {
              ::new (mem) T(std::invoke(
                  Factory, std::forward<decltype(values)>(values)...));
              in->flags.set_lifecycle(inst_constructed | inst_owned |
                                      inst_initializing);
            } else {
              void *allocation = allocate_native_storage<T>();
              if (!allocation)
                return constructor_outcome::no_memory;
              auto *value = ::new (allocation) T(std::invoke(
                  Factory, std::forward<decltype(values)>(values)...));
              arm_boxed<T>(in, mem, value);
            }
          } else {
            auto value =
                std::invoke(Factory, std::forward<decltype(values)>(values)...);
            if (!value)
              return constructor_outcome::null_owner;
            arm_block<T, kind>(in, mem, std::move(value));
          }
          return constructor_outcome::constructed;
        },
        [] {});
  }
};

template <class Sig>
PyObject *constructor_signature(const callable_spec &spec) noexcept {
  if (PyErr_Occurred())
    return nullptr;
  function_object view{};
  view.argsinfo = spec.argsinfo;
  view.arity = spec.arity;
  return sig_builder<Sig, false>::build(&view);
}

template <class Sig, int Arity, class... Extras>
int register_ctor(type_record *record, ctor_vector_fn invoke,
                  Extras &&...extras) {

  static_assert(([] {
                  using E = std::remove_cvref_t<Extras>;
                  if constexpr (is_keep_alive_v<E>)
                    return E::nurse > 0 && E::patient > 0 &&
                           E::nurse <= Arity + 1 && E::patient <= Arity + 1;
                  else
                    return true;
                }() &&
                 ...),
                "pyb26: constructor keep_alive slots are self=1, explicit "
                "args=2...; there is no return slot");
  callable_spec callable{};
  callable.name = "__init__";
  callable.arity = Arity;
  fill_spec<Arity>(callable, std::forward<Extras>(extras)...);
  fill_profile_label<Extras...>(callable,
                                reinterpret_cast<PyObject *>(record->py_type));
  ctor_spec spec{};
  spec.arity = Arity;
  spec.argsinfo = callable.argsinfo;
  spec.invoke = invoke;
  spec.profile_label = callable.profile_label;
  if constexpr (Arity > 16)
    spec.normalize = &normalize_ctor<Arity>;
  spec.complete_positional = !has_kw_only_v<Extras...>;
  return add_ctor(record, spec, constructor_signature<Sig>(callable));
}

template <auto M, class... Extras>
void register_member_method(type_record *record, const char *name,
                            Extras &&...extras) {
  type_mutation mutation(record);
  if (!mutation)
    return;

  if (!record || !record->py_type)
    return;
  using Tr = for_method<M, policy_of<Extras...>(), find_call_guard_t<Extras...>,
                        policy_pack_t<Extras...>>;
  callable_spec spec{};
  spec.name = name;
  spec.arity = Tr::arity;
  fill_spec<Tr::arity>(spec, std::forward<Extras>(extras)...);
  fill_profile_label<Extras...>(spec,
                                reinterpret_cast<PyObject *>(record->py_type));
  spec.make_sig =
      &sig_builder<typename member_sig<decltype(M)>::sig, true>::build;
  set_entry<Tr, has_kw_only_v<Extras...>>(spec);
  add_method(record->py_type, spec);
}

template <class C, auto F, class... Extras>
void register_self_first_method(type_record *record, const char *name,
                                Extras &&...extras) {
  type_mutation mutation(record);
  if (!mutation)
    return;

  if (!record || !record->py_type)
    return;
  using Tr =
      for_fn_method<F, policy_of<Extras...>(), find_call_guard_t<Extras...>, C,
                    policy_pack_t<Extras...>>;
  callable_spec spec{};
  spec.name = name;
  spec.arity = Tr::arity;
  fill_spec<Tr::arity>(spec, std::forward<Extras>(extras)...);
  fill_profile_label<Extras...>(spec,
                                reinterpret_cast<PyObject *>(record->py_type));
  spec.make_sig =
      &sig_builder<typename self_first_sig<nttp_sig_t<F>>::sig, true>::build;
  set_entry<Tr, has_kw_only_v<Extras...>>(spec);
  add_method(record->py_type, spec);
}

template <auto F, class... Extras>
void register_static_method(type_record *record, const char *name,
                            Extras &&...extras) {
  type_mutation mutation(record);
  if (!mutation)
    return;

  if (!record || !record->py_type)
    return;
  using Tr = for_fn<F, policy_of<Extras...>(), find_call_guard_t<Extras...>,
                    policy_pack_t<Extras...>>;
  callable_spec spec{};
  spec.name = name;
  spec.arity = Tr::arity;
  fill_spec<Tr::arity>(spec, std::forward<Extras>(extras)...);
  fill_profile_label<Extras...>(spec,
                                reinterpret_cast<PyObject *>(record->py_type));
  spec.make_sig = &sig_builder<nttp_sig_t<F>, false>::build;
  set_entry<Tr, has_kw_only_v<Extras...>>(spec);
  add_static(record->py_type, spec);
}

template <auto FP, bool Writable, class... Extras>
void register_field(type_record *record, const char *name, Extras &&...extras) {
  type_mutation mutation(record);
  if (!mutation)
    return;

  validate_property_extras<Writable, Extras...>();
  if (!record || !record->py_type)
    return;
  setter set = nullptr;
  if constexpr (Writable)
    set = &field_set<FP, Extras...>;
  auto spec = make_getset_spec(
      record->py_type, name, &field_get<FP, policy_of<Extras...>(), Extras...>,
      set, std::forward<Extras>(extras)...);
  add_getset(record->py_type, spec);
}

template <auto Get, auto Set, class... Extras>
void register_property(type_record *record, const char *name,
                       Extras &&...extras) {
  type_mutation mutation(record);
  if (!mutation)
    return;

  validate_property_extras<true, Extras...>();
  if (!record || !record->py_type)
    return;
  static_assert(
      std::is_same_v<typename property_traits<Get>::cls,
                     typename property_traits<Set>::cls>,
      "pyb26: property getter and setter must have the same native self type");
  auto spec = make_getset_spec(
      record->py_type, name, &prop_get<Get, policy_of<Extras...>(), Extras...>,
      &prop_set<Set, Extras...>, std::forward<Extras>(extras)...);
  add_getset(record->py_type, spec);
}

template <auto Get, class... Extras>
void register_readonly_property(type_record *record, const char *name,
                                Extras &&...extras) {
  type_mutation mutation(record);
  if (!mutation)
    return;

  validate_property_extras<false, Extras...>();
  if (!record || !record->py_type)
    return;
  auto spec = make_getset_spec(
      record->py_type, name, &prop_get<Get, policy_of<Extras...>(), Extras...>,
      nullptr, std::forward<Extras>(extras)...);
  add_getset(record->py_type, spec);
}

template <class T, auto SP, bool Writable, class... Extras>
void register_static_field(PyObject *module, type_record *record,
                           const char *name, Extras &&...extras) {
  type_mutation mutation(record);
  if (!mutation)
    return;

  validate_property_extras<Writable, Extras...>();
  static_assert(
      std::is_pointer_v<decltype(SP)>,
      "pyb26: static field binding expects a pointer to a static data member");
  if (!record || !record->py_type)
    return;
  setter set = nullptr;
  if constexpr (Writable)
    set = &static_field_set<T, SP, Extras...>;
  auto spec = make_getset_spec(
      record->py_type, name,
      &static_field_get<T, SP, policy_of<Extras...>(), Extras...>, set,
      std::forward<Extras>(extras)...);
  add_static_getset(module, record->py_type, spec);
}

template <class T, auto Get, auto Set, class... Extras>
void register_static_property(PyObject *module, type_record *record,
                              const char *name, Extras &&...extras) {
  type_mutation mutation(record);
  if (!mutation)
    return;

  validate_property_extras<true, Extras...>();
  if (!record || !record->py_type)
    return;
  auto spec = make_getset_spec(
      record->py_type, name,
      &static_prop_get<T, Get, policy_of<Extras...>(), Extras...>,
      &static_prop_set<T, Set, Extras...>, std::forward<Extras>(extras)...);
  add_static_getset(module, record->py_type, spec);
}

template <class T, auto Get, class... Extras>
void register_readonly_static_property(PyObject *module, type_record *record,
                                       const char *name, Extras &&...extras) {
  type_mutation mutation(record);
  if (!mutation)
    return;

  validate_property_extras<false, Extras...>();
  if (!record || !record->py_type)
    return;
  auto spec = make_getset_spec(
      record->py_type, name,
      &static_prop_get<T, Get, policy_of<Extras...>(), Extras...>, nullptr,
      std::forward<Extras>(extras)...);
  add_static_getset(module, record->py_type, spec);
}

template <bool Enabled> struct implicit_slot {};

template <> struct implicit_slot<true> {
  PyObject *temp = nullptr;
  implicit_slot() = default;
  implicit_slot(const implicit_slot &) = delete;
  implicit_slot &operator=(const implicit_slot &) = delete;
  ~implicit_slot() { Py_XDECREF(temp); }
};

} // namespace detail

template <typename T>
struct type_caster : detail::implicit_slot<enable_implicit_v<T>>,
                     detail::wrapper_input_projection<type_caster<T>, T> {
  static constexpr caster_effects effects{enable_implicit_v<T>, false, false};
  using input_implementation = type_caster<T>;
  using native_lifetime_type = T;
  using native_lifetime_caster = type_caster<T>;
  using owner_implementation = type_caster<T>;
  using borrowed_owner_implementation = type_caster<T>;
  static_assert(std::is_class_v<T>,
                "pyb26: no type_caster for this type. Register the class with "
                "pyb26::class_<T>, add an STL include, or it is unsupported.");
  T *ptr = nullptr;

  PyObject *alias_owner(PyObject *input) const noexcept {
    if constexpr (enable_implicit_v<T>)
      return this->temp ? this->temp : input;
    else
      return input;
  }

  PYB26_HOT bool load(PyObject *o) noexcept {
    ptr = detail::cast_to<T>(o);
    if constexpr (enable_implicit_v<T>) {
      if (ptr) [[likely]]
        return true;
      return load_converted(o);
    } else {
      return ptr != nullptr;
    }
  }

  PYB26_HOT bool load_exact(PyObject *o) noexcept {
    ptr = detail::cast_to<T>(o);
    return ptr != nullptr;
  }

  PYB26_COLD PYB26_NOINLINE bool load_converted(PyObject *o) noexcept
    requires(enable_implicit_v<T>)
  {
    detail::type_record *tr = detail::resolve<T>();
    if (!tr) [[unlikely]]
      return false;
    for (detail::implicit_conv *c = tr->convs; c; c = c->next) {
      if (!c->gate(o)) {
        if (PyErr_Occurred())
          return false;
        continue;
      }
      PyObject *built =
          PyObject_CallOneArg(reinterpret_cast<PyObject *>(tr->py_type), o);
      if (!built)
        return false;
      T *p = detail::cast_to<T>(built);
      if (!p) [[unlikely]] {
        Py_DECREF(built);
        if (!PyErr_Occurred())
          PyErr_SetString(PyExc_TypeError,
                          "implicit conversion returned an invalid instance");
        return false;
      }
      this->temp = built;
      ptr = p;
      return true;
    }
    return false;
  }

  template <class A> decltype(auto) as() {
    using R = std::remove_reference_t<A>;
    if constexpr (std::is_pointer_v<R>)
      return ptr;
    else
      return static_cast<A>(*ptr);
  }
  static PyObject *cast(const T &v) {
    return detail::make_instance<T>([&](void *m) { ::new (m) T(v); });
  }
  static PyObject *cast(T &&v) {
    return detail::make_instance<T>(
        [&](void *m) { ::new (m) T(static_cast<T &&>(v)); });
  }
  static PyObject *cast(T *p) {
    return p ? cast(*p) : (Py_INCREF(Py_None), Py_None);
  }
  static PyObject *cast(const T *p) {
    return p ? cast(*p) : (Py_INCREF(Py_None), Py_None);
  }
};

template <typename T>
  requires(std::is_class_v<T> && detail::exact_loadable<std::remove_cv_t<T>>)
struct type_caster<T *>
    : detail::scalar_input_projection<type_caster<T *>,
                                      argument_source::wrapper_payload> {
  static constexpr caster_effects effects{false, true, false};
  using input_implementation = type_caster;
  using owner_implementation = type_caster;
  using borrowed_owner_implementation = type_caster;
  PyObject *alias_owner(PyObject *input) const noexcept { return input; }
  using V = std::remove_cv_t<T>;
  T *ptr = nullptr;

  PYB26_HOT bool load(PyObject *o) noexcept {
    if (Py_IsNone(o)) {
      ptr = nullptr;
      return true;
    }
    ptr = detail::cast_to<V>(o);
    return ptr != nullptr;
  }
  template <class A> A as() const noexcept { return ptr; }
  static PyObject *cast(T *p) { return type_caster<V>::cast(p); }
};

template <class A, class B> inline void implicitly_convertible(module_ m) {
  detail::reject_lease_escape<A>();
  static_assert(
      enable_implicit_v<B>,
      "pyb26: implicitly_convertible<A, B> requires the opt-in trait "
      "`template <> inline constexpr bool pyb26::enable_implicit_v<B> "
      "= true;` declared BEFORE B is bound or used in a binding — the "
      "trait changes B's caster shape, so forgetting it must be a "
      "build error, not a dead registration (bind_vector precedent)");
  static_assert(
      std::is_class_v<B>,
      "pyb26: implicitly_convertible<A, B> — B must be a bound class");
  (void)m;
  detail::type_record *tr = detail::resolve<B>();
  if (!tr) [[unlikely]]
    return;
  detail::type_mutation mutation(tr);
  if (!mutation)
    return;
  auto *node =
      new (std::nothrow) detail::implicit_conv{[](PyObject *o) noexcept {
                                                 type_caster<A> c;
                                                 return c.load(o);
                                               },
                                               nullptr};
  if (!node) [[unlikely]]
    return;
  detail::implicit_conv **pp = &tr->convs;
  while (*pp)
    pp = &(*pp)->next;
  *pp = node;
}

template <class T> struct holder_traits {
  static constexpr bool is_holder = false;
  using value_type = T;
};
template <class T> struct holder_traits<std::shared_ptr<T>> {
  static constexpr bool is_holder = true;
  using value_type = T;
};
template <class T> struct holder_traits<std::unique_ptr<T>> {
  static constexpr bool is_holder = true;
  using value_type = T;
};

struct gc_t {};
constexpr gc_t gc() noexcept { return {}; }

template <class T = void> struct class_options {
  bool gc = false;
  const PyType_Slot *slots = nullptr;
  int count = 0;
  int (*prepare)(module_) noexcept = nullptr;
};

template <class Native> struct trampoline {
  using type = Native;
};

namespace detail {
template <class Option> struct trampoline_option {
  static constexpr bool value = false;
  using type = void;
};
template <class Native> struct trampoline_option<::pyb26::trampoline<Native>> {
  static constexpr bool value = true;
  using type = Native;
};
template <class... Types> struct first_class_option {
  using type = void;
};
template <class First, class... Rest>
struct first_class_option<First, Rest...> {
  using type =
      std::conditional_t<std::is_void_v<First>,
                         typename first_class_option<Rest...>::type, First>;
};
} // namespace detail

template <class T, class... Options> class class_ {
  static_assert(
      !caster_requires_lease_v<std::remove_cv_t<T>>,
      "borrowed array view requires a live lease and cannot be stored here");
  static constexpr unsigned holders =
      (0u + ... + holder_traits<Options>::is_holder);
  static constexpr unsigned trampolines =
      (0u + ... + detail::trampoline_option<Options>::value);
  static constexpr unsigned bases =
      (0u + ... +
       (!std::is_void_v<Options> && !holder_traits<Options>::is_holder &&
        !detail::trampoline_option<Options>::value));
  static_assert(
      holders <= 1 && trampolines <= 1 && bases <= 1,
      "pyb26: class options permit one native base, holder, and trampoline");
  using Base = typename detail::first_class_option<
      std::conditional_t<!holder_traits<Options>::is_holder &&
                             !detail::trampoline_option<Options>::value,
                         Options, void>...>::type;
  using Holder = typename detail::first_class_option<std::conditional_t<
      holder_traits<Options>::is_holder, Options, void>...>::type;
  using SelectedTrampoline = typename detail::first_class_option<
      typename detail::trampoline_option<Options>::type...>::type;
  using Native = std::conditional_t<std::is_void_v<SelectedTrampoline>, T,
                                    SelectedTrampoline>;
  static constexpr bool has_trampoline = !std::is_void_v<SelectedTrampoline>;
  static_assert(
      std::is_void_v<Holder> ||
          std::is_same_v<typename holder_traits<Holder>::value_type, T>,
      "pyb26: class holder must own the bound native type");
  static_assert(!has_trampoline ||
                    (std::is_base_of_v<T, Native> &&
                     std::is_base_of_v<trampoline_self, Native> &&
                     std::has_virtual_destructor_v<T>),
                "pyb26: trampoline must derive from the bound type and "
                "trampoline_self; bound type needs a virtual destructor");

  detail::type_record *tr_ = nullptr;
  PyObject *mod_;

public:
  class_(module_ m, const char *name, const char *doc = nullptr)
      : class_(m, name, doc, false) {}
  class_(module_ m, const char *name, gc_t) : class_(m, name, nullptr, true) {}
  class_(module_ m, const char *name, const char *doc, gc_t)
      : class_(m, name, doc, true) {}
  class_(module_ m, const char *name, const char *doc,
         const PyType_Slot *extra_slots, int n_extra)
      : class_(m, name, doc, false, extra_slots, n_extra) {}
  class_(module_ m, const char *name, detail::factory_only_t,
         const PyType_Slot *extra_slots, int n_extra)
      : class_(m, name, nullptr, false, extra_slots, n_extra, nullptr,
               detail::class_creation_policy::factory_only) {
    static_assert(detail::native_inline_v<T> && std::is_void_v<Base> &&
                      std::is_void_v<Holder> && !has_trampoline,
                  "pyb26: factory-only classes require inline storage without "
                  "a base, holder, or trampoline");
  }
  template <class Owner>
    requires(std::is_void_v<Owner> || std::is_base_of_v<Owner, T>)
  class_(module_ m, const char *name, class_options<Owner> options,
         const char *doc = nullptr)
      : class_(m, name, doc, options.gc, options.slots, options.count,
               options.prepare) {}

private:
  class_(module_ m, const char *name, const char *doc, bool use_gc,
         const PyType_Slot *extra_slots = nullptr, int n_extra = 0,
         int (*prepare)(module_) noexcept = nullptr,
         detail::class_creation_policy policy =
             detail::class_creation_policy::regular)
      : mod_(m.state_ptr()) {
    static_assert(std::is_void_v<Base> || std::is_base_of_v<Base, T>,
                  "pyb26: Base must be a base class of T");
    if (PyErr_Occurred()) [[unlikely]]
      return;
    if (!name) {
      PyErr_SetString(PyExc_ValueError, "invalid class name or instance size");
      return;
    }
    PyTypeObject *base_py = nullptr;
    if constexpr (!std::is_void_v<Base>) {
      detail::type_record *btr = detail::resolve<Base>();
      base_py = btr ? btr->py_type : nullptr;
      if (!base_py) {
        detail::error_unregistered(typeid(Base).name());
        return;
      }
      use_gc = use_gc || PyType_IS_GC(base_py);
    }
    if (prepare) {
      const int status = prepare(m);
      if (status < 0 || PyErr_Occurred()) {
        if (!PyErr_Occurred())
          PyErr_SetString(PyExc_RuntimeError,
                          "class preparation failed without an exception");
        return;
      }
    }
    std::unique_ptr<detail::type_record, decltype(&detail::destroy_type_record)>
        record(detail::binding_new<detail::allocation_site::type_metadata,
                                   detail::type_record>(),
               &detail::destroy_type_record);
    if (!record) {
      PyErr_NoMemory();
      return;
    }
    record->cpp_type = &typeid(T);
    record->cpp_size = static_cast<Py_ssize_t>(sizeof(Native));
    record->data_offset = has_trampoline
                              ? detail::tramp_data_offset<T, Native>()
                              : detail::holder_data_offset<T>();
    record->requires_wrapper_owner = has_trampoline;
    record->name = name;
    record->drain = &detail::drain_freelist<T>;
    record->traverse_freelists = &detail::traverse_freelists<T>;
    record->native_pointer = [](PyObject *self,
                                Py_ssize_t offset) noexcept -> void * {
      if constexpr (has_trampoline)
        if (auto *in = reinterpret_cast<detail::instance *>(self);
            in->storage == detail::st_inline &&
            (in->flags.lifecycle() & detail::inst_alias))
          return static_cast<T *>(reinterpret_cast<Native *>(
              reinterpret_cast<char *>(self) + offset));
      return detail::object_ptr<T>(
          self, reinterpret_cast<detail::instance *>(self)->storage, offset);
    };
    if constexpr (!std::is_void_v<Base>) {
      record->base = detail::resolve<Base>();
      record->upcast = [](void *pointer) noexcept -> void * {
        return static_cast<Base *>(static_cast<T *>(pointer));
      };
    }
    Py_ssize_t basicsize =
        record->data_offset + (has_trampoline
                                   ? detail::tramp_payload_size<T, Native>()
                                   : detail::storage_payload_size<T>());
    if (base_py && basicsize < base_py->tp_basicsize)
      basicsize = base_py->tp_basicsize;
    if (use_gc) {
      record->lifetime_offset =
          detail::align_up(basicsize, alignof(PyObject *));
      basicsize = record->lifetime_offset + sizeof(PyObject *);
    }
    const auto dealloc = [] {
      if constexpr (has_trampoline)
        return &detail::tramp_dealloc<T, Native>;
      else
        return &detail::inst_dealloc<T>;
    }();
    const auto dealloc_gc = [] {
      if constexpr (has_trampoline)
        return &detail::tramp_dealloc_gc<T, Native>;
      else
        return &detail::inst_dealloc_gc<T>;
    }();
    record->py_type =
        use_gc
            ? detail::create_class(
                  m.ptr(), m.state_ptr(), name, doc, basicsize, dealloc_gc,
                  &detail::inst_init<T>, &detail::inst_new_gc<T>, base_py,
                  &detail::inst_traverse<T, Native>,
                  &detail::inst_clear<T, Native>, extra_slots, n_extra, policy)
            : detail::create_class(m.ptr(), m.state_ptr(), name, doc, basicsize,
                                   dealloc, &detail::inst_init<T>,
                                   &detail::inst_new<T>, base_py, nullptr,
                                   nullptr, extra_slots, n_extra, policy);
    if (!record->py_type) [[unlikely]] {
      return;
    }
    if (policy == detail::class_creation_policy::factory_only &&
        !detail::seal_type(record.get()))
      return;
    if (detail::register_class(m.ptr(), m.state_ptr(), record.get(),
                               detail::type_chain_of<T>) < 0)
      return;
    tr_ = record.release();
  }

public:
  template <auto Getter> class_ &def_dlpack();

  class_ &ctor_raw(detail::ctor_fn f) {
    static_assert(!has_trampoline,
                  "pyb26: trampoline classes require typed constructors");
    static_assert(detail::native_inline_v<T>,
                  "pyb26: over-aligned classes require typed constructors");
    if (!tr_ || !tr_->py_type) [[unlikely]]
      return *this;
    detail::ctor_spec spec{};
    spec.legacy = f;
    detail::add_ctor(tr_, spec);
    return *this;
  }

  template <class... Args, class... Extras> class_ &ctor(Extras &&...extras) {
    if (!tr_ || !tr_->py_type) [[unlikely]]
      return *this;
    using Ctor = detail::constructor_binding<T, Native, Holder, void(Args...),
                                             Extras...>;
    detail::register_ctor<void(Args...), Ctor::arity>(
        tr_, &Ctor::invoke, std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto Fn, class... Extras> class_ &factory(Extras &&...extras) {
    return ctor_factory<Fn>(std::forward<Extras>(extras)...);
  }

  template <auto Factory, class... Extras>
  class_ &ctor_factory(Extras &&...extras) {
    static_assert(!has_trampoline,
                  "pyb26: trampoline classes require typed constructors");
    if (!tr_ || !tr_->py_type)
      return *this;
    using Ctor = detail::factory_constructor_binding<
        T, Factory, detail::frontend_signature_t<decltype(Factory)>, Extras...>;
    detail::register_ctor<detail::frontend_signature_t<decltype(Factory)>,
                          Ctor::arity>(tr_, &Ctor::invoke,
                                       std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto M, class... Extras>
  class_ &def(const char *name, Extras &&...extras) {
    detail::register_member_method<M>(tr_, name,
                                      std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto F, class... Extras>
  class_ &def_fn(const char *name, Extras &&...extras) {
    detail::register_self_first_method<T, F>(tr_, name,
                                             std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto F, class... Extras>
  class_ &def_static(const char *name, Extras &&...extras) {
    detail::register_static_method<F>(tr_, name,
                                      std::forward<Extras>(extras)...);
    return *this;
  }

  template <class F, class... Extras>
  class_ &def_property_readonly(const char *name, F &&getter,
                                Extras &&...extras) {
    detail::type_mutation mutation(tr_);
    if (!mutation)
      return *this;

    static_assert(
        ((detail::is_rv_tag<Extras> ||
          std::is_convertible_v<Extras, const char *>) &&
         ...),
        "pyb26: properties accept return policy and documentation extras");
    if (tr_ && tr_->py_type)
      detail::register_callable_property_readonly<
          T, detail::policy_of<Extras...>()>(tr_->py_type, name,
                                             detail::doc_of(extras...),
                                             std::forward<F>(getter));
    return *this;
  }

  template <class F, class S, class... Extras>
  class_ &def_property(const char *name, F &&getter, S &&setter,
                       Extras &&...extras) {
    detail::type_mutation mutation(tr_);
    if (!mutation)
      return *this;

    static_assert(
        ((detail::is_rv_tag<Extras> ||
          std::is_convertible_v<Extras, const char *>) &&
         ...),
        "pyb26: properties accept return policy and documentation extras");
    if (tr_ && tr_->py_type)
      detail::register_callable_property<T, detail::policy_of<Extras...>()>(
          tr_->py_type, name, detail::doc_of(extras...),
          std::forward<F>(getter), std::forward<S>(setter));
    return *this;
  }

  template <auto FP, class... Extras>
  class_ &def_readwrite(const char *name, Extras &&...extras) {
    detail::register_field<FP, true>(tr_, name,
                                     std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto FP, class... Extras>
  class_ &def_readonly(const char *name, Extras &&...extras) {
    detail::register_field<FP, false>(tr_, name,
                                      std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto Get, auto Set, class... Extras>
  class_ &def_property(const char *name, Extras &&...extras) {
    detail::register_property<Get, Set>(tr_, name,
                                        std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto Get, class... Extras>
  class_ &def_property_readonly(const char *name, Extras &&...extras) {
    detail::register_readonly_property<Get>(tr_, name,
                                            std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto SP, class... Extras>
  class_ &def_readwrite_static(const char *name, Extras &&...extras) {
    detail::register_static_field<T, SP, true>(mod_, tr_, name,
                                               std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto SP, class... Extras>
  class_ &def_readonly_static(const char *name, Extras &&...extras) {
    detail::register_static_field<T, SP, false>(
        mod_, tr_, name, std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto Get, auto Set, class... Extras>
  class_ &def_property_static(const char *name, Extras &&...extras) {
    detail::register_static_property<T, Get, Set>(
        mod_, tr_, name, std::forward<Extras>(extras)...);
    return *this;
  }

  template <auto Get, class... Extras>
  class_ &def_property_readonly_static(const char *name, Extras &&...extras) {
    detail::register_readonly_static_property<T, Get>(
        mod_, tr_, name, std::forward<Extras>(extras)...);
    return *this;
  }
};

} // namespace pyb26
