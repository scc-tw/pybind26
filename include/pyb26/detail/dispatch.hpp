#pragma once

#include "../args.hpp"
#include "argument_admission.hpp"
#include "call_policy.hpp"
#include "caster.hpp"
#include "gil.hpp"
#include "holders.hpp"
#include "instance.hpp"
#include "invocation.hpp"
#include "return_owner.hpp"

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pyb26::detail {

template <class Caster>
PYB26_ALWAYS_INLINE void restore_caster_error(Caster &caster) noexcept {
  if constexpr (requires { caster.restore_error(); })
    caster.restore_error();
}

template <class R> PYB26_ALWAYS_INLINE PyObject *cast_return(R &&r) {
  using D = std::remove_cvref_t<R>;
  static_assert(!caster_requires_lease_v<D>,
                "borrowed view cannot be returned; use an owning result or "
                "explicit copy");
  if constexpr (std::is_same_v<D, PyObject *>) {
    if (r) [[likely]]
      return r;
    return PyErr_Occurred() ? nullptr : Py_NewRef(Py_None);
  } else if constexpr (std::is_pointer_v<D>)
    return type_caster<std::remove_cv_t<std::remove_pointer_t<D>>>::cast(r);
  else
    return type_caster<D>::cast(static_cast<R &&>(r));
}

template <rv_policy Policy, class R>
PYB26_ALWAYS_INLINE PyObject *cast_return_rvp(R &&r, PyObject *parent) {
  using D = std::remove_cvref_t<R>;
  static_assert(!caster_requires_lease_v<D>,
                "borrowed view cannot be returned; use an owning result or "
                "explicit copy");
  if constexpr (Policy == rv_policy::automatic || Policy == rv_policy::copy ||
                Policy == rv_policy::move) {
    (void)parent;
    return cast_return(static_cast<R &&>(r));
  } else {
    using Obj = std::remove_pointer_t<std::remove_reference_t<R>>;
    using Bare = std::remove_cv_t<Obj>;
    static_assert(
        std::is_pointer_v<std::remove_reference_t<R>> || std::is_reference_v<R>,
        "pyb26: reference/take_ownership/reference_internal require a "
        "pointer- or reference-returning target");
    Bare *p;
    if constexpr (std::is_pointer_v<std::remove_reference_t<R>>)
      p = const_cast<Bare *>(r);
    else
      p = const_cast<Bare *>(std::addressof(r));
    if constexpr (Policy == rv_policy::reference)
      return make_reference<Bare>(p, nullptr, false);
    else if constexpr (Policy == rv_policy::reference_internal)
      return make_reference<Bare>(p, parent, true);
    else
      return make_owned_pointer<Bare>(p);
  }
}

template <rv_policy Policy, class R, class CallPolicy>
consteval auto postcall_native_type() {
  if constexpr (std::is_void_v<R> ||
                !native_postcall_policy<CallPolicy>::value) {
    return std::type_identity<void>{};
  } else if constexpr (Policy == rv_policy::automatic ||
                       Policy == rv_policy::copy || Policy == rv_policy::move) {
    using D = std::remove_cvref_t<R>;
    if constexpr (std::is_same_v<D, PyObject *>) {
      return std::type_identity<void>{};
    } else {
      using CastT =
          std::conditional_t<std::is_pointer_v<D>,
                             std::remove_cv_t<std::remove_pointer_t<D>>, D>;
      using Caster = type_caster<CastT>;
      if constexpr (requires {
                      typename Caster::native_lifetime_type;
                      typename Caster::native_lifetime_caster;
                    }) {
        if constexpr (std::is_same_v<typename Caster::native_lifetime_caster,
                                     Caster>)
          return std::type_identity<typename Caster::native_lifetime_type>{};
        else
          return std::type_identity<void>{};
      } else {
        return std::type_identity<void>{};
      }
    }
  } else {
    using Bare =
        std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<R>>>;
    return std::type_identity<Bare>{};
  }
}

class python_error_latch {
  PyThreadState *state_;
  PyObject *error_ = nullptr;

public:
  python_error_latch() noexcept : state_(cpython::thread_state()) {}
  python_error_latch(const python_error_latch &) = delete;
  python_error_latch &operator=(const python_error_latch &) = delete;
  ~python_error_latch() {
    if (error_) [[unlikely]]
      discard_python_result(error_);
  }
  bool capture() noexcept {
    if (cpython::current_exception(state_)) [[unlikely]] {
      if (!error_)
        error_ = retain_python_error(cpython::current_exception(state_));
    }
    return error_ != nullptr;
  }
  void restore() noexcept {
    if (error_) [[unlikely]]
      finish_python_error(state_, std::exchange(error_, nullptr), nullptr);
  }
  void discard(PyObject *result) noexcept {
    if (result)
      discard_python_result(result);
  }
  PyObject *finish(PyObject *result) noexcept {
    if (capture()) [[unlikely]] {
      return finish_python_error(state_, std::exchange(error_, nullptr),
                                 result);
    }
    return result;
  }
};

class python_error_probe {
  PyThreadState *state_ = cpython::thread_state();

public:
  bool capture() const noexcept {
    return cpython::current_exception(state_) != nullptr;
  }
  void discard(PyObject *result) const noexcept {
    if (result)
      discard_preserving_python_error(state_, result);
  }
  PyObject *finish(PyObject *result) const noexcept { return result; }
};

template <rv_policy Policy, class R, class Error>
PYB26_ALWAYS_INLINE PyObject *
checked_cast_return_rvp(R &&value, PyObject *parent, Error &error) {
  if (error.capture()) [[unlikely]] {
    if constexpr (std::is_same_v<std::remove_cvref_t<R>, PyObject *>) {
      error.discard(value);
    } else if constexpr (Policy == rv_policy::take_ownership) {
      using Bare =
          std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<R>>>;
      if constexpr (std::is_class_v<Bare>) {
        if constexpr (std::is_pointer_v<std::remove_reference_t<R>>)
          delete const_cast<Bare *>(value);
        else if constexpr (std::is_reference_v<R>)
          delete const_cast<Bare *>(std::addressof(value));
      }
    }
    return nullptr;
  }
  PyObject *result = cast_return_rvp<Policy>(static_cast<R &&>(value), parent);
  if (!result && !error.capture()) [[unlikely]]
    PyErr_SetString(PyExc_SystemError,
                    "return caster failed without a Python error");
  if (error.capture()) [[unlikely]] {
    error.discard(result);
    return nullptr;
  }
  return result;
}

template <rv_policy Policy, class R>
PYB26_ALWAYS_INLINE PyObject *checked_cast_return_rvp(R &&value,
                                                      PyObject *parent) {
  python_error_latch error;
  PyObject *result =
      checked_cast_return_rvp<Policy>(static_cast<R &&>(value), parent, error);
  return error.finish(result);
}

template <rv_policy Policy = rv_policy::automatic, class F>
PYB26_ALWAYS_INLINE PyObject *invoke_getter(F &&getter, PyObject *parent) {
  using R = decltype(getter());
  if constexpr (python_error_result_v<R> || guard_argument_python_owner_v<R>) {
    decltype(auto) value = getter();
    return checked_cast_return_rvp<Policy>(static_cast<R &&>(value), parent);
  } else {
    return cast_return_rvp<Policy>(getter(), parent);
  }
}

template <auto Fn> struct nttp_invoker {
  static constexpr bool has_self = false;
  static constexpr bool self_parented = false;
  static constexpr std::size_t arg_base = 0;
  template <class... A, class Cast, std::size_t... I>
  PYB26_ALWAYS_INLINE static decltype(auto) invoke(Cast &c,
                                                   std::index_sequence<I...>) {
    return Fn(project_input<A>(get_argument<I>(c))...);
  }
};

template <class F> struct functor_invoker {
  static constexpr bool has_self = false;
  static constexpr bool self_parented = false;
  static constexpr std::size_t arg_base = 0;
  template <class... A, class Cast, std::size_t... I>
  PYB26_ALWAYS_INLINE static decltype(auto) invoke(F *fn, Cast &c,
                                                   std::index_sequence<I...>) {
    return (*fn)(project_input<A>(get_argument<I>(c))...);
  }
  static void free_capture(void *p) noexcept { delete static_cast<F *>(p); }
};

template <auto M, class C> struct method_invoker {
  static constexpr bool has_self = true;
  static constexpr bool self_parented = false;
  static constexpr std::size_t arg_base = 1;
  static constexpr bool python_owner_self = guard_argument_python_owner_v<C>;
  template <class... A, class Cast, std::size_t... I>
  PYB26_ALWAYS_INLINE static decltype(auto) invoke(C *self, Cast &c,
                                                   std::index_sequence<I...>) {
    return (self->*M)(project_input<A>(get_argument<I>(c))...);
  }
};

PYB26_ALWAYS_INLINE bool no_kw(PyObject *kwnames) noexcept {
  return kwnames == nullptr || PyTuple_GET_SIZE(kwnames) == 0;
}

template <auto Fn, class Self, class C> struct nttp_fnm_invoker {
  using self_type = std::remove_cvref_t<Self>;
  static constexpr bool python_self = std::is_same_v<self_type, PyObject *> ||
                                      std::is_same_v<self_type, handle>;
  using cls = std::conditional_t<python_self, PyObject, C>;
  using native_self = std::remove_reference_t<std::conditional_t<
      std::is_pointer_v<Self>, std::remove_pointer_t<Self>, Self>>;
  static_assert(python_self || std::is_convertible_v<C *, native_self *>,
                "pyb26: native self must accept the bound class or its base");
  static constexpr bool python_owner_self = guard_argument_python_owner_v<Self>;
  static constexpr bool python_wrapper_self = python_self;
  static constexpr bool alias_self_stable =
      python_self || std::is_pointer_v<Self> || std::is_reference_v<Self>;
  static constexpr bool has_self = true;
  static constexpr bool self_parented = false;
  static constexpr std::size_t arg_base = 1;
  [[gnu::always_inline]] static inline cls *
  receiver_of(PyObject *self) noexcept {
    if constexpr (python_self)
      return self;
    else
      return cast_to<C>(self);
  }
  template <class... A, class Cast, std::size_t... I>
  PYB26_ALWAYS_INLINE static decltype(auto) invoke(cls *self, Cast &c,
                                                   std::index_sequence<I...>) {
    if constexpr (std::is_same_v<self_type, PyObject *>)
      return Fn(self, project_input<A>(get_argument<I>(c))...);
    else if constexpr (std::is_same_v<self_type, handle>)
      return Fn(self_type{self}, project_input<A>(get_argument<I>(c))...);
    else if constexpr (std::is_pointer_v<Self>)
      return Fn(static_cast<Self>(self),
                project_input<A>(get_argument<I>(c))...);
    else
      return Fn(static_cast<Self>(*self),
                project_input<A>(get_argument<I>(c))...);
  }
};

template <class Sig, bool WithSelf> struct sig_builder;
template <class R, class... A, bool WithSelf>
struct sig_builder<R(A...), WithSelf> {
  static PyObject *build(const function_object *f) noexcept {
    static constexpr std::size_t N = sizeof...(A);
    static constexpr bool captures[N == 0 ? 1 : N] = {is_args_v<A>...};
    std::string hints[N == 0 ? 1 : N];
    std::string fallback_names[N == 0 ? 1 : N];
    std::string rhint;
    std::size_t i = 0;
    auto append_argument = [&]<class T>() {
      append_py_hint<T>(hints[i]);
      if (PyErr_Occurred())
        return false;
      fallback_names[i] = "arg" + std::to_string(i);
      ++i;
      return true;
    };
    if (!(true && ... && append_argument.template operator()<A>()))
      return nullptr;
    append_py_hint<R>(rhint);
    if (PyErr_Occurred())
      return nullptr;

    PyObject *inspect = PyImport_ImportModule("inspect");
    if (!inspect)
      return nullptr;
    PyObject *Parameter = PyObject_GetAttrString(inspect, "Parameter");
    PyObject *Signature = PyObject_GetAttrString(inspect, "Signature");
    Py_DECREF(inspect);
    if (!Parameter || !Signature) {
      Py_XDECREF(Parameter);
      Py_XDECREF(Signature);
      return nullptr;
    }
    PyObject *K_POK =
        PyObject_GetAttrString(Parameter, "POSITIONAL_OR_KEYWORD");
    PyObject *K_PO = PyObject_GetAttrString(Parameter, "POSITIONAL_ONLY");
    PyObject *K_KO = PyObject_GetAttrString(Parameter, "KEYWORD_ONLY");
    PyObject *K_VP = PyObject_GetAttrString(Parameter, "VAR_POSITIONAL");
    PyObject *EMPTY = PyObject_GetAttrString(Parameter, "empty");
    PyObject *params = PyList_New(0);
    PyObject *result = nullptr;
    if (!K_POK || !K_PO || !K_KO || !K_VP || !EMPTY || !params)
      goto done;

    if constexpr (WithSelf) {
      PyObject *p = PyObject_CallFunction(Parameter, "sO", "self", K_PO);
      if (!p || PyList_Append(params, p) < 0) {
        Py_XDECREF(p);
        goto done;
      }
      Py_DECREF(p);
    }
    {
      std::size_t order[N == 0 ? 1 : N];
      std::size_t count = 0;
      bool capture_emitted = false;
      for (std::size_t i = 0; i < N; ++i) {
        if (captures[i]) {
          if (!capture_emitted)
            order[count++] = i;
          continue;
        }
        if constexpr (N > 0) {
          if (captures[N - 1] && !capture_emitted && f->argsinfo &&
              (f->argsinfo[i].flags & arg_kw_only)) {
            order[count++] = N - 1;
            capture_emitted = true;
          }
        }
        order[count++] = i;
      }
      for (std::size_t position = 0; position < count; ++position) {
        const std::size_t i = order[position];
        arg_info *ai = f->argsinfo && i < static_cast<std::size_t>(f->arity)
                           ? &f->argsinfo[i]
                           : nullptr;
        const char *nm = nullptr;
        if (ai && ai->name) {
          nm = PyUnicode_AsUTF8(ai->name);
          if (!nm)
            goto done;
        } else {
          nm = fallback_names[i].c_str();
        }
        PyObject *kind = captures[i] ? K_VP : K_POK;
        if (ai && (ai->flags & arg_kw_only))
          kind = K_KO;
        else if (ai && (ai->flags & arg_pos_only))
          kind = K_PO;
        PyObject *dflt = (ai && ai->deflt) ? ai->deflt : EMPTY;
        PyObject *kw = Py_BuildValue("{s:O,s:s}", "default", dflt, "annotation",
                                     hints[i].c_str());
        PyObject *pargs = Py_BuildValue("(sO)", nm, kind);
        PyObject *p =
            (kw && pargs) ? PyObject_Call(Parameter, pargs, kw) : nullptr;
        Py_XDECREF(kw);
        Py_XDECREF(pargs);
        if (!p || PyList_Append(params, p) < 0) {
          Py_XDECREF(p);
          goto done;
        }
        Py_DECREF(p);
      }
    }
    {
      PyObject *kw = Py_BuildValue("{s:s}", "return_annotation", rhint.c_str());
      PyObject *args = Py_BuildValue("(O)", params);
      result = (kw && args) ? PyObject_Call(Signature, args, kw) : nullptr;
      Py_XDECREF(kw);
      Py_XDECREF(args);
    }
  done:
    Py_XDECREF(K_POK);
    Py_XDECREF(K_PO);
    Py_XDECREF(K_KO);
    Py_XDECREF(K_VP);
    Py_XDECREF(EMPTY);
    Py_XDECREF(params);
    Py_DECREF(Parameter);
    Py_DECREF(Signature);
    return result;
  }
};

template <auto Fn, class Sig> struct nttp_sig;
template <class Sig> struct self_first_sig;
template <class R, class Self, class... A>
struct self_first_sig<R(Self, A...)> {
  using self = Self;
  using cls =
      std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<Self>>>;
  using sig = R(A...);
};
template <class Invoker> struct invoker_receiver {
  using type = void;
};
template <auto M, class C> struct invoker_receiver<method_invoker<M, C>> {
  using type = C;
};
template <auto Fn, class Self, class C>
struct invoker_receiver<nttp_fnm_invoker<Fn, Self, C>> {
  using type = typename nttp_fnm_invoker<Fn, Self, C>::cls;
};
template <class F> struct invoker_receiver<functor_invoker<F>> {
  using type = F;
};

template <class Invoker> struct trivial_invoker_cleanup : std::false_type {};
template <auto Fn>
struct trivial_invoker_cleanup<nttp_invoker<Fn>> : std::true_type {};
template <class F>
struct trivial_invoker_cleanup<functor_invoker<F>> : std::true_type {};
template <auto M, class C>
struct trivial_invoker_cleanup<method_invoker<M, C>> : std::true_type {};
template <auto Fn, class Self, class C>
struct trivial_invoker_cleanup<nttp_fnm_invoker<Fn, Self, C>>
    : std::bool_constant<std::is_trivially_destructible_v<Self>> {};

template <class Policy> struct trivial_checked_policy : std::false_type {};
template <> struct trivial_checked_policy<void> : std::true_type {};
template <class... Extras>
struct trivial_checked_policy<call_policies<Extras...>>
    : std::bool_constant<std::is_same_v<
          typename call_policies<Extras...>::scope, profile_scope<void>>> {};

template <class Invoker> inline constexpr bool invoker_is_functor_v = false;
template <class F>
inline constexpr bool invoker_is_functor_v<functor_invoker<F>> = true;

template <class T>
inline constexpr bool native_dispatch_argument = [] {
  using Plan = input_argument_plan<T, invocation_caster_t<T>>;
  return Plan::load_inert && Plan::extraction_inert && Plan::parameter_inert &&
         Plan::cleanup_inert;
}();

template <class Invoker, class Sig, rv_policy Policy = rv_policy::automatic,
          class Guard = void, class CallPolicy = void>
struct trampoline;

template <class Invoker, class R, class... A, rv_policy Policy, class Guard,
          class CallPolicy>
struct trampoline<Invoker, R(A...), Policy, Guard, CallPolicy> {
  using signature = R(A...);
  using native_postcall_type =
      typename decltype(postcall_native_type<Policy, R, CallPolicy>())::type;
  static constexpr int parameter_count = static_cast<int>(sizeof...(A));
  static constexpr int args_count = (0 + ... + (is_args_v<A> ? 1 : 0));
  static constexpr bool variadic = args_count != 0;
  static constexpr int arity = parameter_count - (variadic ? 1 : 0);
  static_assert(args_count <= 1,
                "only one positional args capture is supported");
  static_assert(
      !variadic ||
          is_args_v<
              std::tuple_element_t<parameter_count ? parameter_count - 1 : 0,
                                   std::tuple<A..., void>>>,
      "args must be the final C++ parameter");
  static constexpr bool has_self = Invoker::has_self;
  static constexpr bool has_guard = !std::is_void_v<Guard>;
  static constexpr std::size_t base = Invoker::arg_base;
  static constexpr std::size_t return_owner_slot = [] {
    if constexpr (std::is_void_v<CallPolicy>)
      return std::size_t{0};
    else if constexpr (requires { CallPolicy::return_owner_slot; })
      return CallPolicy::return_owner_slot;
    else
      return std::size_t{0};
  }();
  static_assert(!return_owner_slot || Policy == rv_policy::reference_internal,
                "pyb26: return_owner requires rv_policy::reference_internal");
  static_assert(return_owner_slot <= arity + base,
                "pyb26: return_owner is outside this call's argument slots");
  static_assert(
      Policy != rv_policy::reference_internal || has_self ||
          Invoker::self_parented || return_owner_slot,
      "pyb26: free-function reference_internal needs return_owner<N>");
  static_assert(
      [] {
        if constexpr (Policy == rv_policy::reference_internal &&
                      return_owner_slot <= base &&
                      requires { Invoker::alias_self_stable; })
          return Invoker::alias_self_stable;
        return true;
      }(),
      "pyb26: a class value receiver cannot anchor a returned alias");
  static constexpr bool pins_return_owner = [] {
    if constexpr (return_owner_slot > base &&
                  return_owner_slot <= arity + base) {
      using Arg =
          std::tuple_element_t<return_owner_slot - base - 1, std::tuple<A...>>;
      return !borrowed_return_owner_v<invocation_caster_t<Arg>>;
    }
    return false;
  }();
  static constexpr bool python_owner_self = [] {
    if constexpr (requires { Invoker::python_owner_self; })
      return Invoker::python_owner_self;
    else
      return false;
  }();
  static constexpr bool python_wrapper_self = [] {
    if constexpr (requires { Invoker::python_wrapper_self; })
      return Invoker::python_wrapper_self;
    else
      return false;
  }();
  static constexpr bool checked_invocation =
      [] {
        if constexpr (std::is_void_v<CallPolicy>)
          return false;
        else if constexpr (requires { CallPolicy::checks_target; })
          return CallPolicy::checks_target;
        else
          return false;
      }() ||
      python_wrapper_self || python_owner_self ||
      (caster_contract<A>.requires_admission || ...) ||
      ((guard_argument_python_owner_v<A> || is_args_v<A> ||
        std::is_same_v<std::remove_cvref_t<A>, handle> ||
        std::is_same_v<std::remove_cvref_t<A>, PyObject *>) ||
       ...) ||
      python_error_result_v<R> || guard_argument_python_owner_v<R>;
  using Arguments = invocation_arguments<A...>;
  static constexpr bool native_guard =
      std::is_void_v<Guard> ||
      std::is_same_v<Guard, call_guard<gil_scoped_release>>;
  static constexpr bool checked_cleanup =
      checked_invocation || pins_return_owner || !native_guard ||
      ((!native_dispatch_argument<A>) || ...) ||
      [] {
        if constexpr (std::is_void_v<R>)
          return false;
        else
          return !builtin_output_cast<R>;
      }() ||
      [] {
        if constexpr (std::is_void_v<CallPolicy>)
          return false;
        else if constexpr (requires { CallPolicy::error_capable; })
          return CallPolicy::error_capable;
        else
          return true;
      }();
  static constexpr bool trivial_checked_cleanup = [] {
    if constexpr (!checked_cleanup || has_guard || pins_return_owner ||
                  Policy == rv_policy::take_ownership ||
                  !trivial_invoker_cleanup<Invoker>::value ||
                  !trivial_checked_policy<CallPolicy>::value) {
      return false;
    } else {
      return (std::is_void_v<R> || native_cleanup_inert_v<R>) &&
             !Arguments::cleanup;
    }
  }();
  using checked_error_context =
      std::conditional_t<trivial_checked_cleanup, python_error_probe,
                         python_error_latch>;
  static_assert(!(guard_releases_python_v<Guard> && python_wrapper_self),
                "pyb26: a release guard requires native self; raw Python self "
                "callbacks must remain attached");
  static_assert(
      !guard_releases_python_v<Guard> ||
          (input_argument_plan<A, invocation_caster_t<A>>::parameter_inert &&
           ...),
      "pyb26: release guard requires inert native parameter construction and "
      "destruction");
  static_assert(!(guard_releases_python_v<Guard> &&
                  (((guard_argument_python_owner_v<A> || is_args_v<A>) ||
                    ...) ||
                   python_owner_self)),
                "pyb26: a release guard cannot receive owning Python objects; "
                "convert inputs to native values first");
  static_assert(!(guard_releases_python_v<Guard> &&
                  guard_argument_python_owner_v<R> && !std::is_reference_v<R>),
                "pyb26: a release guard must return native values or "
                "references, not owning Python objects by value");

  PYB26_ALWAYS_INLINE static auto receiver_of(PyObject *self) noexcept {
    if constexpr (requires { Invoker::receiver_of(self); })
      return Invoker::receiver_of(self);
    else
      return cast_to<receiver_t>(self);
  }

  template <class Cast>
  PYB26_ALWAYS_INLINE static PyObject *parent_of(PyObject *const *args,
                                                 Cast &cast) noexcept {
    if constexpr (return_owner_slot > base) {
      constexpr std::size_t index = return_owner_slot - base - 1;
      using Arg = std::tuple_element_t<index, std::tuple<A...>>;
      auto &caster = get_argument<index>(cast);
      using C = std::remove_cvref_t<decltype(caster)>;
      static_assert(
          requires { typename C::owner_implementation; },
          "pyb26: return_owner requires a caster with an explicit owner "
          "contract");
      static_assert(std::is_same_v<C, typename C::owner_implementation>,
                    "pyb26: custom caster must declare its own owner contract");
      static_assert(
          !requires { typename C::native_lifetime_type; } ||
              std::is_reference_v<Arg> || std::is_pointer_v<Arg> ||
              requires { typename C::template invocation_caster<Arg>; },
          "pyb26: a class value parameter cannot anchor a returned alias");
      PyObject *owner = caster.alias_owner(args[index + base]);
      if (!owner && !PyErr_Occurred()) [[unlikely]]
        PyErr_SetString(PyExc_RuntimeError,
                        "return alias owner is unavailable");
      return owner;
    } else if constexpr (has_self || Invoker::self_parented)
      return args[0];
    else
      return nullptr;
  }

  template <class Cast, std::size_t... I, class... Recv>
  PYB26_ALWAYS_INLINE static PyObject *
  invoke_tail(PyObject *, PyObject *const *args, Cast &cast,
              std::index_sequence<I...> seq, Recv... recv) {
    static_assert(!checked_cleanup);
    if constexpr (has_guard) {
      auto prepared = make_prepared_arguments<A...>(cast, seq);
      if constexpr (std::is_void_v<R>) {
        {
          gil_scoped_release guard;
          Invoker::template invoke<A...>(recv..., prepared, seq);
        }
        Py_RETURN_NONE;
      } else {
        R value = [&]() -> R {
          gil_scoped_release guard;
          return Invoker::template invoke<A...>(recv..., prepared, seq);
        }();
        return cast_return_rvp<Policy>(static_cast<R &&>(value),
                                       parent_of(args, cast));
      }
    } else if constexpr (std::is_void_v<R>) {
      Invoker::template invoke<A...>(recv..., cast, seq);
      Py_RETURN_NONE;
    } else {
      return cast_return_rvp<Policy>(
          Invoker::template invoke<A...>(recv..., cast, seq),
          parent_of(args, cast));
    }
  }

  template <class Cast, std::size_t... I, class... Recv>
  PYB26_ALWAYS_INLINE static PyObject *
  invoke_result_checked(checked_error_context &error, PyObject *parent,
                        Cast &cast, std::index_sequence<I...> seq,
                        Recv... recv) {
    PyObject *result = nullptr;
    auto execute = [&](auto &&...values) -> int {
      prepared_arguments<std::index_sequence<I...>, decltype(values)...>
          prepared{{std::forward<decltype(values)>(values)}...};
      auto restore = [&] {
        ((restore_caster_error(get_argument<I>(cast)), error.capture()), ...);
      };
      if constexpr (std::is_void_v<R>) {
        invocation_guarded<Guard>(error, [&] {
          Invoker::template invoke<A...>(recv..., prepared, seq);
        });
        error.capture();
        restore();
        if (!error.capture()) [[likely]]
          result = Py_NewRef(Py_None);
      } else {
        invocation_return_storage<R> value;
        invocation_checkpoint<checked_error_context, !native_cleanup_inert_v<R>>
            checkpoint{error};
        invocation_guarded<Guard>(error, [&] {
          value.construct([&]() -> R {
            return Invoker::template invoke<A...>(recv..., prepared, seq);
          });
        });
        error.capture();
        restore();
        if (value.has_value()) [[likely]]
          result = checked_cast_return_rvp<Policy>(value.get(), parent, error);
      }
      return result ? 0 : -1;
    };
    Arguments::prepare(cast, error, execute);
    return result;
  }

  template <class Cast, std::size_t... I, class... Recv>
  PYB26_ALWAYS_INLINE static PyObject *
  invoke_tail_checked(checked_error_context &error, PyObject *self_fn,
                      PyObject *const *args, Cast &cast,
                      std::index_sequence<I...> seq, Recv... recv) {
    PyObject *parent = parent_of(args, cast);
    if constexpr (Policy == rv_policy::reference_internal)
      if (!parent || error.capture()) [[unlikely]]
        return nullptr;
    return_owner_pin<pins_return_owner> owner{parent};
    if constexpr (std::is_void_v<CallPolicy>) {
      return invoke_result_checked(error, parent, cast, seq, recv...);
    } else {
      if (!CallPolicy::precall(args, arity + base) || error.capture()) {
        error.capture();
        return nullptr;
      }
      typename CallPolicy::scope profile(
          reinterpret_cast<function_object *>(self_fn)->profile_label);
      invocation_checkpoint<checked_error_context,
                            !trivial_checked_policy<CallPolicy>::value>
          checkpoint{error};
      if (error.capture()) [[unlikely]]
        return nullptr;
      return invoke_result_checked(error, parent, cast, seq, recv...);
    }
  }

  template <class Load, std::size_t... I, class... Recv>
  PYB26_ALWAYS_INLINE static PyObject *
  run_checked(PyObject *self_fn, PyObject *const *args, bool attempt,
              Load &load, std::index_sequence<I...> seq, Recv... recv) {
    checked_error_context error;
    PyObject *result = nullptr;
    auto invoke = [&](auto &cast) -> int {
      result = invoke_tail_checked(error, self_fn, args, cast, seq, recv...);
      return result ? 0 : -1;
    };
    const int status = Arguments::load(load, error, invoke);
    if (error.capture()) [[unlikely]] {
      error.discard(result);
      return error.finish(nullptr);
    }
    if (status == 1) {
      if (attempt)
        return overload_miss;
      return error_arg_convert(self_fn);
    }
    if (!result) [[unlikely]] {
      PyErr_SetString(PyExc_SystemError,
                      "invocation failed without a Python error");
      return nullptr;
    }
    if constexpr (!std::is_void_v<CallPolicy>) {
      if (!apply_policy_postcall<CallPolicy, native_postcall_type>(
              result, args, arity + base)) {
        if (!error.capture())
          PyErr_SetString(PyExc_SystemError,
                          "call policy failed without a Python error");
        error.capture();
        error.discard(result);
        return error.finish(nullptr);
      }
    }
    if (error.capture()) [[unlikely]] {
      error.discard(result);
      return error.finish(nullptr);
    }
    return result;
  }

  template <std::size_t... I, class... Recv>
  PYB26_NO_SSP PYB26_FLATTEN static PyObject *
  run(PyObject *self_fn, PyObject *const *args, std::index_sequence<I...> seq,
      Recv... recv) noexcept {
    argument_owner_admission<(caster_contract<A>.requires_admission || ...)>
        admission;
    if (!admission.acquire())
      return nullptr;
    if constexpr (checked_cleanup) {
      auto load = [&]<std::size_t N>(auto &cast) {
        return cast.load(args[N + base]);
      };
      return run_checked(self_fn, args, false, load, seq, recv...);
    } else {
      std::tuple<caster_for<A>...> cast;
      template for (constexpr std::size_t i : {I...}) {
        if (!std::get<i>(cast).load(args[i + base])) [[unlikely]]
          return error_arg_convert(self_fn);
      }
      return invoke_tail(self_fn, args, cast, seq, recv...);
    }
  }

  template <std::size_t... I, class... Recv>
  static PyObject *run_try(PyObject *self_fn, PyObject *const *args,
                           std::index_sequence<I...> seq,
                           Recv... recv) noexcept {
    argument_owner_admission<(caster_contract<A>.requires_admission || ...)>
        admission;
    if (!admission.acquire())
      return nullptr;
    if constexpr (checked_cleanup) {
      auto load = [&]<std::size_t N>(auto &cast) {
        return cast.load(args[N + base]);
      };
      return run_checked(self_fn, args, true, load, seq, recv...);
    } else {
      std::tuple<caster_for<A>...> cast;
      template for (constexpr std::size_t i : {I...}) {
        if (!std::get<i>(cast).load(args[i + base]))
          return PyErr_Occurred() ? nullptr : overload_miss;
      }
      return invoke_tail(self_fn, args, cast, seq, recv...);
    }
  }

  PYB26_NO_SSP PYB26_HOT static PyObject *call(PyObject *self_fn,
                                               PyObject *const *args,
                                               size_t nargsf,
                                               PyObject *kwnames) noexcept {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if constexpr (variadic)
      return variadic_call(self_fn, args, nargs, kwnames, false);
    if constexpr (has_self) {
      using C = receiver_t;
      if (nargs == arity + 1 && no_kw(kwnames)) [[likely]] {
        C *self = receiver_of(args[0]);
        if (!self) [[unlikely]]
          return error_arg_convert(self_fn);
        return run(self_fn, args, std::index_sequence_for<A...>{}, self);
      }
      return method_call_slow(self_fn, args, nargs, kwnames);
    } else if constexpr (is_functor) {
      using F = receiver_t;
      F *fn = static_cast<F *>(
          reinterpret_cast<function_object *>(self_fn)->capture);
      if (nargs == arity && no_kw(kwnames)) [[likely]]
        return run(self_fn, args, std::index_sequence_for<A...>{}, fn);
      return call_slow(self_fn, args, nargs, kwnames, fn);
    } else {
      if (nargs == arity && no_kw(kwnames)) [[likely]]
        return run(self_fn, args, std::index_sequence_for<A...>{});
      return call_slow(self_fn, args, nargs, kwnames);
    }
  }

  PYB26_COLD static PyObject *call_keyword_only(PyObject *self_fn,
                                                PyObject *const *args,
                                                Py_ssize_t nargs,
                                                PyObject *kwnames) noexcept {
    if constexpr (has_self)
      return method_call_slow(self_fn, args, nargs, kwnames);
    else if constexpr (is_functor)
      return call_slow(
          self_fn, args, nargs, kwnames,
          static_cast<receiver_t *>(
              reinterpret_cast<function_object *>(self_fn)->capture));
    else
      return call_slow(self_fn, args, nargs, kwnames);
  }

  template <class... Recv>
  PYB26_COLD static PyObject *
  call_slow(PyObject *self_fn, PyObject *const *args, Py_ssize_t nargs,
            PyObject *kwnames, Recv... recv) noexcept {
    if constexpr (arity == 0) {
      return error_arity(self_fn, nargs, kwnames);
    } else {
      PyObject *slots[arity];
      if (assemble_args(self_fn, args, nargs, kwnames, slots) != 0) [[unlikely]]
        return nullptr;
      return run(self_fn, slots, std::index_sequence_for<A...>{}, recv...);
    }
  }

  PYB26_COLD PYB26_NOINLINE static PyObject *
  method_call_slow(PyObject *self_fn, PyObject *const *args, Py_ssize_t nargs,
                   PyObject *kwnames) noexcept {
    using C = receiver_t;
    if (nargs < 1) [[unlikely]]
      return error_arity(self_fn, nargs, kwnames);
    C *self = receiver_of(args[0]);
    if (!self) [[unlikely]]
      return error_arg_convert(self_fn);
    PyObject *slots[arity + 1];
    slots[0] = args[0];
    if (assemble_args(self_fn, args + 1, nargs - 1, kwnames, slots + 1) != 0)
        [[unlikely]]
      return nullptr;
    return run(self_fn, slots, std::index_sequence_for<A...>{}, self);
  }

  template <std::size_t... I, class... Recv>
  static PyObject *run_variadic(PyObject *self_fn, PyObject *const *slots,
                                PyObject *const *tail, Py_ssize_t tail_size,
                                bool attempt, std::index_sequence<I...> seq,
                                Recv... recv) noexcept {
    argument_owner_admission<(caster_contract<A>.requires_admission || ...)>
        admission;
    if (!admission.acquire())
      return nullptr;
    static_assert(checked_cleanup);
    auto load = [&]<std::size_t N>(auto &cast) {
      if constexpr (N == static_cast<std::size_t>(arity))
        return cast.load_range(tail, tail_size);
      else
        return cast.load(slots[N + base]);
    };
    return run_checked(self_fn, slots, attempt, load, seq, recv...);
  }

  static PyObject *variadic_call(PyObject *self_fn, PyObject *const *args,
                                 Py_ssize_t nargs, PyObject *kwnames,
                                 bool attempt,
                                 bool force_parse = false) noexcept {
    static_assert(variadic);
    if (nargs < static_cast<Py_ssize_t>(base)) {
      if (attempt)
        return overload_miss;
      return error_arity(self_fn, nargs, kwnames);
    }
    const Py_ssize_t supplied = nargs - base;
    Py_ssize_t positional_limit = arity;
    if (force_parse) {
      auto *function = reinterpret_cast<function_object *>(self_fn);
      for (int i = 0; i < arity; ++i) {
        if (function->argsinfo && (function->argsinfo[i].flags & arg_kw_only)) {
          positional_limit = i;
          break;
        }
      }
    }
    const Py_ssize_t prefix =
        supplied < positional_limit ? supplied : positional_limit;
    const Py_ssize_t tail_size = supplied - prefix;
    PyObject *slots[arity + base ? arity + base : 1];
    if constexpr (has_self)
      slots[0] = args[0];
    if (prefix == arity && no_kw(kwnames) && !force_parse) {
      for (int i = 0; i < arity; ++i)
        slots[i + base] = args[i + base];
    } else {
      const int status = assemble_variadic_args(
          self_fn, args + base, prefix, kwnames, args + nargs, slots + base);
      if (status != 0) {
        if (attempt && status > 0) {
          PyErr_Clear();
          return overload_miss;
        }
        return nullptr;
      }
    }
    PyObject *const *tail = tail_size ? args + base + prefix : nullptr;
    if constexpr (has_self) {
      receiver_t *receiver = receiver_of(slots[0]);
      if (!receiver)
        return attempt && !PyErr_Occurred() ? overload_miss
                                            : error_arg_convert(self_fn);
      return run_variadic(self_fn, slots, tail, tail_size, attempt,
                          std::index_sequence_for<A...>{}, receiver);
    } else if constexpr (is_functor) {
      auto *receiver = static_cast<receiver_t *>(
          reinterpret_cast<function_object *>(self_fn)->capture);
      return run_variadic(self_fn, slots, tail, tail_size, attempt,
                          std::index_sequence_for<A...>{}, receiver);
    } else {
      return run_variadic(self_fn, slots, tail, tail_size, attempt,
                          std::index_sequence_for<A...>{});
    }
  }

  static void free_capture(void *p) noexcept { Invoker::free_capture(p); }

  static PyObject *try_call(PyObject *self_fn, PyObject *const *args,
                            size_t nargsf, PyObject *kwnames) noexcept {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if constexpr (variadic)
      return variadic_call(self_fn, args, nargs, kwnames, true);
    if constexpr (has_self) {
      using C = receiver_t;
      if (nargs == arity + 1 && no_kw(kwnames)) {
        C *self = receiver_of(args[0]);
        if (!self)
          return PyErr_Occurred() ? nullptr : overload_miss;
        return run_try(self_fn, args, std::index_sequence_for<A...>{}, self);
      }
      return try_call_slow(self_fn, args, nargs, kwnames);
    } else if constexpr (is_functor) {
      using F = receiver_t;
      if (nargs == arity && no_kw(kwnames)) {
        F *fn = static_cast<F *>(
            reinterpret_cast<function_object *>(self_fn)->capture);
        return run_try(self_fn, args, std::index_sequence_for<A...>{}, fn);
      }
      return try_call_slow(self_fn, args, nargs, kwnames);
    } else {
      if (nargs == arity && no_kw(kwnames))
        return run_try(self_fn, args, std::index_sequence_for<A...>{});
      return try_call_slow(self_fn, args, nargs, kwnames);
    }
  }

  static PyObject *try_call_normalized(PyObject *self_fn, PyObject *const *args,
                                       PyObject *const *slots,
                                       int status) noexcept {
    if constexpr (has_self) {
      auto *self = receiver_of(args[0]);
      if (!self)
        return PyErr_Occurred() ? nullptr : overload_miss;
      if (status)
        return overload_miss;
      return run_try(self_fn, slots, std::index_sequence_for<A...>{}, self);
    } else {
      if (status)
        return overload_miss;
      if constexpr (is_functor) {
        auto *fn = static_cast<receiver_t *>(
            reinterpret_cast<function_object *>(self_fn)->capture);
        return run_try(self_fn, slots, std::index_sequence_for<A...>{}, fn);
      } else {
        return run_try(self_fn, slots, std::index_sequence_for<A...>{});
      }
    }
  }

  template <class... Recv>
  static PyObject *keyword_frame(PyObject *entry, PyObject *owner, int index,
                                 PyObject *const *args, Py_ssize_t nargs,
                                 size_t nargsf, PyObject *kwnames, int *next,
                                 Recv... recv) noexcept {
    [[indeterminate]] PyObject *slots[arity + base == 0 ? 1 : arity + base];
    if constexpr (has_self)
      slots[0] = args[0];
    const int status =
        assemble_args(entry, args + base, nargs - base, kwnames, slots + base);
    if (status < 0)
      return nullptr;
    if (!status) {
      if constexpr (checked_cleanup) {
        PyObject *result =
            run_try(entry, slots, std::index_sequence_for<A...>{}, recv...);
        if (result != overload_miss)
          return result;
      } else {
        std::tuple<type_caster<std::remove_cvref_t<A>>...> casters;
        const int match = [&]<std::size_t... I>(std::index_sequence<I...>) {
          template for (constexpr std::size_t i : {I...}) {
            if (!std::get<i>(casters).load(slots[i + base]))
              return PyErr_Occurred() ? -1 : 1;
          }
          return 0;
        }(std::index_sequence_for<A...>{});
        if (match < 0)
          return nullptr;
        if (!match)
          return invoke_tail(entry, slots, casters,
                             std::index_sequence_for<A...>{}, recv...);
      }
    } else {
      PyErr_Clear();
    }
    return overload_keyword_miss(owner, index, args, nargsf, kwnames, slots,
                                 status, next);
  }

  static PyObject *try_call_keywords(PyObject *entry, PyObject *owner,
                                     int index, PyObject *const *args,
                                     size_t nargsf, PyObject *kwnames,
                                     int *next) noexcept {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if constexpr (!has_self && arity == 0) {
      if (next) {
        *next = index + 1;
        return overload_miss;
      }
      return overload_continue(owner, args, nargsf, kwnames, index + 1);
    } else if constexpr (has_self) {
      if (nargs < 1) {
        if (next) {
          *next = index + 1;
          return overload_miss;
        }
        return overload_continue(owner, args, nargsf, kwnames, index + 1);
      }
      auto *self = receiver_of(args[0]);
      if (!self) {
        if (PyErr_Occurred())
          return nullptr;
        if (next) {
          *next = index + 1;
          return overload_miss;
        }
        return overload_continue(owner, args, nargsf, kwnames, index + 1);
      }
      return keyword_frame(entry, owner, index, args, nargs, nargsf, kwnames,
                           next, self);
    } else if constexpr (is_functor) {
      auto *fn = static_cast<receiver_t *>(
          reinterpret_cast<function_object *>(entry)->capture);
      return keyword_frame(entry, owner, index, args, nargs, nargsf, kwnames,
                           next, fn);
    } else {
      return keyword_frame(entry, owner, index, args, nargs, nargsf, kwnames,
                           next);
    }
  }

  static PyObject *overload_call(PyObject *owner, PyObject *const *args,
                                 size_t nargsf, PyObject *kwnames) noexcept {
    if (!kwnames || !PyTuple_GET_SIZE(kwnames))
      return overload_positional(owner, args, nargsf, kwnames);
    auto *set = static_cast<overload_set *>(
        reinterpret_cast<function_object *>(owner)->capture);
    return try_call_keywords(reinterpret_cast<PyObject *>(set->entries[0]),
                             owner, 0, args, nargsf, kwnames, nullptr);
  }

  PYB26_COLD PYB26_NOINLINE static PyObject *
  try_call_slow(PyObject *self_fn, PyObject *const *args, Py_ssize_t nargs,
                PyObject *kwnames) noexcept {
    if constexpr (has_self) {
      using C = receiver_t;
      if (nargs < 1)
        return overload_miss;
      C *self = receiver_of(args[0]);
      if (!self)
        return PyErr_Occurred() ? nullptr : overload_miss;
      PyObject *slots[arity + 1];
      slots[0] = args[0];
      const int status =
          assemble_args(self_fn, args + 1, nargs - 1, kwnames, slots + 1);
      if (status < 0)
        return nullptr;
      if (status > 0) {
        PyErr_Clear();
        return overload_miss;
      }
      return run_try(self_fn, slots, std::index_sequence_for<A...>{}, self);
    } else if constexpr (arity == 0) {
      (void)args;
      (void)nargs;
      (void)kwnames;
      return overload_miss;
    } else {
      PyObject *slots[arity];
      const int status = assemble_args(self_fn, args, nargs, kwnames, slots);
      if (status < 0)
        return nullptr;
      if (status > 0) {
        PyErr_Clear();
        return overload_miss;
      }
      if constexpr (is_functor) {
        using F = receiver_t;
        F *fn = static_cast<F *>(
            reinterpret_cast<function_object *>(self_fn)->capture);
        return run_try(self_fn, slots, std::index_sequence_for<A...>{}, fn);
      } else {
        return run_try(self_fn, slots, std::index_sequence_for<A...>{});
      }
    }
  }

private:
  static constexpr bool is_functor = invoker_is_functor_v<Invoker>;
  using receiver_t = typename invoker_receiver<Invoker>::type;

public:
  inline static constexpr callable_ops callbacks = [] {
    callable_ops result{};
    result.make_sig = &sig_builder<signature, has_self>::build;
    if constexpr (!variadic) {
      result.overload_vc = &overload_call;
      result.keywords = &try_call_keywords;
      result.normalized = &try_call_normalized;
      result.arg_base = base;
    }
    return result;
  }();
};

template <class M> struct member_sig;
template <auto Fn, class F> struct nttp_sig {
  using sig =
      typename member_sig<decltype(&std::remove_cvref_t<F>::operator())>::sig;
};
template <auto Fn, class R, class... A> struct nttp_sig<Fn, R (*)(A...)> {
  using sig = R(A...);
};
template <auto Fn, class R, class... A>
struct nttp_sig<Fn, R (*)(A...) noexcept> {
  using sig = R(A...);
};
template <auto Fn> using nttp_sig_t = typename nttp_sig<Fn, decltype(Fn)>::sig;

template <class M> struct member_sig;
template <class R, class C, class... A> struct member_sig<R (C::*)(A...)> {
  using cls = C;
  using sig = R(A...);
};
template <class R, class C, class... A>
struct member_sig<R (C::*)(A...) const> {
  using cls = C;
  using sig = R(A...);
};
template <class R, class C, class... A>
struct member_sig<R (C::*)(A...) noexcept> {
  using cls = C;
  using sig = R(A...);
};
template <class R, class C, class... A>
struct member_sig<R (C::*)(A...) const noexcept> {
  using cls = C;
  using sig = R(A...);
};

template <class F> struct functor_sig {
  using sig = typename member_sig<decltype(&F::operator())>::sig;
};
template <class R, class... A> struct functor_sig<R (*)(A...)> {
  using sig = R(A...);
};
template <class R, class... A> struct functor_sig<R (*)(A...) noexcept> {
  using sig = R(A...);
};
template <class F>
using functor_sig_t = typename functor_sig<std::remove_cvref_t<F>>::sig;

template <class Invoker> struct invoker_sig;
template <auto Fn> struct invoker_sig<nttp_invoker<Fn>> {
  using type = nttp_sig_t<Fn>;
};
template <auto Fn, class Self, class C>
struct invoker_sig<nttp_fnm_invoker<Fn, Self, C>> {
  using type = typename self_first_sig<nttp_sig_t<Fn>>::sig;
};
template <auto M, class C> struct invoker_sig<method_invoker<M, C>> {
  using type = typename member_sig<decltype(M)>::sig;
};
template <class F> struct invoker_sig<functor_invoker<F>> {
  using type = functor_sig_t<F>;
};
template <class Invoker>
using invoker_sig_t = typename invoker_sig<Invoker>::type;

template <class Invoker, rv_policy Policy = rv_policy::automatic,
          class Guard = void, class CallPolicy = void>
using trampoline_for =
    trampoline<Invoker, invoker_sig_t<Invoker>, Policy, Guard, CallPolicy>;

template <auto Fn, rv_policy Policy = rv_policy::automatic, class Guard = void,
          class CallPolicy = void>
using for_fn = trampoline_for<nttp_invoker<Fn>, Policy, Guard, CallPolicy>;
template <auto M, rv_policy Policy = rv_policy::automatic, class Guard = void,
          class CallPolicy = void>
using for_method =
    trampoline_for<method_invoker<M, typename member_sig<decltype(M)>::cls>,
                   Policy, Guard, CallPolicy>;
template <class F, rv_policy Policy = rv_policy::automatic, class Guard = void,
          class CallPolicy = void>
using for_functor = trampoline_for<functor_invoker<std::remove_cvref_t<F>>,
                                   Policy, Guard, CallPolicy>;
template <
    auto F, rv_policy Policy = rv_policy::automatic, class Guard = void,
    class C = std::remove_cv_t<std::remove_pointer_t<
        std::remove_reference_t<typename self_first_sig<nttp_sig_t<F>>::self>>>,
    class CallPolicy = void>
using for_fn_method = trampoline_for<
    nttp_fnm_invoker<F, typename self_first_sig<nttp_sig_t<F>>::self, C>,
    Policy, Guard, CallPolicy>;

} // namespace pyb26::detail
