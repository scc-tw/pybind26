#pragma once

#include "dispatch.hpp"

namespace pyb26 {

struct arg {
  const char *name_;
  PyObject *default_ = nullptr;
  bool default_failed_ = false;
  explicit arg(const char *n) noexcept : name_(n) {}
  arg(const arg &a) noexcept
      : name_(a.name_), default_(Py_XNewRef(a.default_)),
        default_failed_(a.default_failed_) {}
  arg(arg &&a) noexcept
      : name_(a.name_), default_(std::exchange(a.default_, nullptr)),
        default_failed_(a.default_failed_) {}
  arg &operator=(const arg &other) noexcept {
    if (this != &other) {
      PyObject *value = Py_XNewRef(other.default_);
      Py_XDECREF(default_);
      name_ = other.name_;
      default_ = value;
      default_failed_ = other.default_failed_;
    }
    return *this;
  }
  arg &operator=(arg &&other) noexcept {
    if (this != &other) {
      Py_XDECREF(default_);
      name_ = other.name_;
      default_ = std::exchange(other.default_, nullptr);
      default_failed_ = other.default_failed_;
    }
    return *this;
  }
  ~arg() { Py_XDECREF(default_); }
  template <class T> arg operator=(T &&v) const {
    arg a(name_);
    using D = std::remove_cvref_t<T>;
    if constexpr (std::is_array_v<D> &&
                  std::is_same_v<std::remove_cv_t<std::remove_extent_t<D>>,
                                 char>)
      a.default_ = PyUnicode_FromStringAndSize(
          v, static_cast<Py_ssize_t>(
                 std::extent_v<D> - (v[std::extent_v<D> - 1] == '\0' ? 1 : 0)));
    else
      a.default_ = type_caster<D>::cast(std::forward<T>(v));
    a.default_failed_ = a.default_ == nullptr;
    return a;
  }
};

struct kw_only {};
struct alias_init {};
struct pos_only {};

namespace detail {

template <int Arity> struct extra_processor {
  callable_spec &s;
  arg_info *infos = nullptr;
  int argc = 0;
  bool kw_zone = false;
  bool positional_boundary = false;

  template <class E> void apply(E &&e) {
    if (PyErr_Occurred())
      return;
    if constexpr (std::is_same_v<std::remove_cvref_t<E>, arg>) {
      if (!infos) {
        infos = binding_new_array<allocation_site::argument_metadata, arg_info>(
            Arity);
        s.argsinfo = infos;
        if (!infos) {
          PyErr_NoMemory();
          return;
        }
      }
      if (argc < Arity) {
        if (!e.name_ || !*e.name_) {
          PyErr_SetString(PyExc_TypeError,
                          "pyb26: argument name must not be empty");
          ++argc;
          return;
        }
        infos[argc].name = PyUnicode_InternFromString(e.name_);
        for (int i = 0; i < argc && infos[argc].name; ++i) {
          if (infos[i].name == infos[argc].name)
            PyErr_SetString(PyExc_TypeError, "pyb26: duplicate argument name");
        }
        infos[argc].deflt = Py_XNewRef(e.default_);
        if (e.default_failed_ && !PyErr_Occurred())
          PyErr_SetString(PyExc_TypeError,
                          "pyb26: argument default conversion failed");
        if (kw_zone)
          infos[argc].flags |= arg_kw_only;
      }
      ++argc;
    } else if constexpr (std::is_same_v<std::remove_cvref_t<E>, kw_only>) {
      if (kw_zone)
        PyErr_SetString(PyExc_TypeError,
                        "pyb26: duplicate keyword-only boundary");
      kw_zone = true;
    } else if constexpr (std::is_same_v<std::remove_cvref_t<E>, pos_only>) {
      if (positional_boundary || kw_zone || !argc)
        PyErr_SetString(PyExc_TypeError,
                        "pyb26: invalid positional-only boundary");
      positional_boundary = true;
      for (int i = 0; i < argc && i < Arity; ++i)
        infos[i].flags |= arg_pos_only;
    } else if constexpr (is_call_guard_v<E>) {
      (void)e;
    } else if constexpr (is_rv_tag<E> ||
                         std::is_same_v<std::remove_cvref_t<E>, alias_init> ||
                         is_keep_alive_v<std::remove_cvref_t<E>> ||
                         is_return_owner_v<std::remove_cvref_t<E>> ||
                         is_profile_hook_v<E> ||
                         std::is_same_v<std::remove_cvref_t<E>,
                                        checked_target>) {
    } else {
      s.doc = e;
    }
  }
};

template <int Arity, class... Extras>
void fill_spec(callable_spec &s, Extras &&...extras) {
  constexpr int nargs_given =
      (0 + ... + (std::is_same_v<std::remove_cvref_t<Extras>, arg> ? 1 : 0));
  static_assert(
      nargs_given <= Arity,
      "pyb26: more arg() specs than the bound function has parameters");
  extra_processor<Arity> p{s};
  (p.apply(std::forward<Extras>(extras)), ...);
}

template <class... Extras>
inline constexpr bool has_kw_only_v =
    (std::is_same_v<std::remove_cvref_t<Extras>, kw_only> || ...);

template <class Tr> struct kw_only_entry {
  static PyObject *call(PyObject *self_fn, PyObject *const *args, size_t nargsf,
                        PyObject *kwnames) noexcept {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if constexpr (Tr::variadic)
      return Tr::variadic_call(self_fn, args, nargs, kwnames, false, true);
    else
      return Tr::call_keyword_only(self_fn, args, nargs, kwnames);
  }
  static PyObject *try_call(PyObject *self_fn, PyObject *const *args,
                            size_t nargsf, PyObject *kwnames) noexcept {
    if constexpr (Tr::variadic)
      return Tr::variadic_call(self_fn, args, PyVectorcall_NARGS(nargsf),
                               kwnames, true, true);
    else
      return Tr::try_call_slow(self_fn, args, PyVectorcall_NARGS(nargsf),
                               kwnames);
  }
};

template <class Tr, bool KwOnly> void set_entry(callable_spec &s) noexcept {
  s.variadic = Tr::variadic;
  s.callbacks = &Tr::callbacks;
  if (!s.make_sig)
    s.make_sig = Tr::callbacks.make_sig;
  if constexpr (KwOnly) {
    s.vc = &kw_only_entry<Tr>::call;
    s.try_vc = &kw_only_entry<Tr>::try_call;
  } else {
    s.vc = &Tr::call;
    s.try_vc = &Tr::try_call;
  }
}

} // namespace detail

enum class module_gil { not_used, required };
namespace detail {
constexpr void *
module_gil_slot(module_gil mode = module_gil::not_used) noexcept {
  return mode == module_gil::required ? Py_MOD_GIL_USED : Py_MOD_GIL_NOT_USED;
}
} // namespace detail

class module_ {
  PyObject *m_;
  PyObject *state_;

public:
  explicit module_(PyObject *m) noexcept : m_(m), state_(m) {}
  module_(PyObject *m, PyObject *state) noexcept : m_(m), state_(state) {}
  PyObject *ptr() const noexcept { return m_; }
  PyObject *state_ptr() const noexcept { return state_; }

  template <auto Fn, class... Extras>
  module_ &def(const char *name, Extras &&...extras) {
    constexpr detail::rv_policy P = detail::policy_of<Extras...>();
    detail::callable_spec s{};
    s.name = name;
    s.arity = detail::for_fn<Fn>::arity;
    if constexpr (detail::has_call_guard_v<Extras...>) {
      using Tr = detail::for_fn<Fn, P, detail::find_call_guard_t<Extras...>,
                                detail::policy_pack_t<Extras...>>;
      detail::set_entry<Tr, detail::has_kw_only_v<Extras...>>(s);
    } else {
      using Tr = detail::for_fn<Fn, P, void, detail::policy_pack_t<Extras...>>;
      detail::set_entry<Tr, detail::has_kw_only_v<Extras...>>(s);
    }
    detail::fill_spec<detail::for_fn<Fn>::arity>(
        s, std::forward<Extras>(extras)...);
    s.make_sig = &detail::sig_builder<detail::nttp_sig_t<Fn>, false>::build;
    detail::fill_profile_label<Extras...>(s, m_);
    detail::add_function(m_, state_, s);
    return *this;
  }

  template <class F, class... Extras>
    requires(!std::is_convertible_v<F, const char *>)
  module_ &def(const char *name, F &&f, Extras &&...extras) {
    using FT = std::remove_cvref_t<F>;
    using Guard = detail::find_call_guard_t<Extras...>;
    constexpr auto P = detail::policy_of<Extras...>();
    using Tr = detail::trampoline_for<detail::functor_invoker<FT>, P, Guard,
                                      detail::policy_pack_t<Extras...>>;
    detail::callable_spec s{};
    s.name = name;
    s.arity = Tr::arity;
    s.capture =
        detail::binding_new<detail::allocation_site::callable_capture, FT>(
            std::forward<F>(f));
    if (!s.capture)
      return *this;
    s.free_capture = &Tr::free_capture;
    detail::set_entry<Tr, detail::has_kw_only_v<Extras...>>(s);
    detail::fill_spec<Tr::arity>(s, std::forward<Extras>(extras)...);
    s.make_sig = &detail::sig_builder<detail::functor_sig_t<F>, false>::build;
    detail::fill_profile_label<Extras...>(s, m_);
    detail::add_function(m_, state_, s);
    return *this;
  }

  module_ &doc(const char *d) {
    PyModule_SetDocString(m_, d);
    return *this;
  }

  template <class T> module_ &attr(const char *name, T &&value) {
    PyObject *o =
        type_caster<std::remove_cvref_t<T>>::cast(std::forward<T>(value));
    if (o) {
      PyModule_AddObjectRef(m_, name, o);
      Py_DECREF(o);
    }
    return *this;
  }

  module_ def_submodule(const char *name, const char *doc = nullptr) {
    PyObject *sub = nullptr;
    if (const char *pname = PyModule_GetName(m_)) {
      if (PyObject *full = PyUnicode_FromFormat("%s.%s", pname, name)) {
        sub = PyModule_NewObject(full);
        Py_DECREF(full);
      }
    }
    if (!sub) {
      PyErr_Clear();
      sub = PyModule_New(name);
    }
    if (!sub)
      return *this;
    if (doc)
      PyModule_SetDocString(sub, doc);
    int rc = PyModule_AddObjectRef(m_, name, sub);
    Py_DECREF(sub);
    if (rc < 0)
      return *this;
    return module_{sub, state_};
  }
};

namespace detail {
inline int execute_module(void (*initialize)(module_),
                          PyObject *module) noexcept {
  initialize(module_{module});
  return PyErr_Occurred() ? -1 : finish_module(module);
}
} // namespace detail

} // namespace pyb26

#define PYB26_MODULE(NAME, VAR, ...)                                           \
  static void pyb26_user_init_##NAME(::pyb26::module_ VAR);                    \
  static int pyb26_exec_##NAME(PyObject *m) {                                  \
    if (::pyb26::detail::module_init(m) != 0)                                  \
      return -1;                                                               \
    return ::pyb26::detail::execute_module(&pyb26_user_init_##NAME, m);        \
  }                                                                            \
  static PyModuleDef_Slot pyb26_slots_##NAME[] = {                             \
      {Py_mod_exec, reinterpret_cast<void *>(&pyb26_exec_##NAME)},             \
      {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},    \
      {Py_mod_gil, ::pyb26::detail::module_gil_slot(__VA_ARGS__)},             \
      {0, nullptr},                                                            \
  };                                                                           \
  static PyModuleDef pyb26_moduledef_##NAME = {                                \
      PyModuleDef_HEAD_INIT,                                                   \
      #NAME,                                                                   \
      nullptr,                                                                 \
      sizeof(::pyb26::detail::module_state),                                   \
      nullptr,                                                                 \
      pyb26_slots_##NAME,                                                      \
      &::pyb26::detail::module_traverse,                                       \
      &::pyb26::detail::module_clear,                                          \
      &::pyb26::detail::module_free,                                           \
  };                                                                           \
  PyMODINIT_FUNC PyInit_##NAME(void) {                                         \
    return PyModuleDef_Init(&pyb26_moduledef_##NAME);                          \
  }                                                                            \
  PYB26_MODEXPORT(NAME)                                                        \
  static void pyb26_user_init_##NAME(::pyb26::module_ VAR)

#if defined(Py_mod_state_size) && PY_VERSION_HEX >= 0x030F0000
#define PYB26_MODEXPORT(NAME)                                                  \
  static PyModuleDef_Slot pyb26_state_slots_##NAME[] = {                       \
      {Py_mod_state_size,                                                      \
       reinterpret_cast<void *>(sizeof(::pyb26::detail::module_state))},       \
      {Py_mod_state_traverse,                                                  \
       reinterpret_cast<void *>(&::pyb26::detail::module_traverse)},           \
      {Py_mod_state_clear,                                                     \
       reinterpret_cast<void *>(&::pyb26::detail::module_clear)},              \
      {Py_mod_state_free,                                                      \
       reinterpret_cast<void *>(&::pyb26::detail::module_free)},               \
  };                                                                           \
  extern "C" Py_EXPORTED_SYMBOL PyModuleDef_Slot *PyModExport_##NAME(          \
      PyObject *) {                                                            \
    static PyModuleDef_Slot                                                    \
        export_slots_##NAME[(sizeof(pyb26_slots_##NAME) +                      \
                             sizeof(pyb26_state_slots_##NAME)) /               \
                            sizeof(PyModuleDef_Slot)];                         \
    int n = 0;                                                                 \
    for (auto &s : pyb26_state_slots_##NAME)                                   \
      export_slots_##NAME[n++] = s;                                            \
    for (auto &s : pyb26_slots_##NAME)                                         \
      export_slots_##NAME[n++] = s;                                            \
    return export_slots_##NAME;                                                \
  }
#else
#define PYB26_MODEXPORT(NAME)
#endif
