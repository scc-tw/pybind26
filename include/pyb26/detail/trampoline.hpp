#pragma once

#include "caster.hpp"
#include "common.hpp"
#include "holders.hpp"
#include "instance.hpp"
#include "lifetime.hpp"
#include "python_call.hpp"
#include "python_failure.hpp"

#include <cstring>
#include <type_traits>
#include <utility>

namespace pyb26 {

struct trampoline_self {
  PyObject *py_self = nullptr;
  detail::python_owner_context py_context{};
};

namespace detail {

template <class Tr>
PYB26_ALWAYS_INLINE void set_trampoline_self(void *obj,
                                             PyObject *self) noexcept {
  if constexpr (std::is_base_of_v<trampoline_self, Tr>) {
    auto *trampoline = static_cast<trampoline_self *>(static_cast<Tr *>(obj));
    trampoline->py_self = self;
    trampoline->py_context = capture_python_owner_context();
  }
}

template <class T, class Tr> constexpr Py_ssize_t tramp_data_offset() noexcept {
  return native_data_offset<T, Tr>();
}
template <class T, class Tr>
constexpr Py_ssize_t tramp_payload_size() noexcept {
  if constexpr (!native_inline_v<Tr>)
    return storage_payload_size<T>();
  const Py_ssize_t s = static_cast<Py_ssize_t>(sizeof(Tr));
  const Py_ssize_t h = storage_payload_size<T>();
  return s > h ? s : h;
}

template <class T, class Tr> void tramp_dealloc(PyObject *self) noexcept {
  static_assert(std::has_virtual_destructor_v<T>,
                "pyb26: a trampoline base must have a virtual destructor");
  auto *in = reinterpret_cast<instance *>(self);
  const Py_ssize_t off = tramp_data_offset<T, Tr>();
  if (in->storage != st_inline) [[unlikely]] {
    destroy_holder<T>(self, in->storage, off);
  } else if (in->flags & inst_constructed) {
    destroy_inline_native<T, Tr>(self, off, in->flags.lifecycle());
  }
  PyTypeObject *t = Py_TYPE(self);
  t->tp_free(self);
  Py_DECREF(t);
}

template <class T, class Tr> void tramp_dealloc_gc(PyObject *self) noexcept {
  static_assert(std::has_virtual_destructor_v<T>,
                "pyb26: a trampoline base must have a virtual destructor");
  PyObject_GC_UnTrack(self);
  auto *in = reinterpret_cast<instance *>(self);
  const Py_ssize_t off = tramp_data_offset<T, Tr>();
  if (in->storage != st_inline) [[unlikely]] {
    destroy_holder<T>(self, in->storage, off);
  } else if (in->flags & inst_constructed) {
    destroy_inline_native<T, Tr>(self, off, in->flags.lifecycle());
  }
  clear_lifetime(self);
  PyTypeObject *type = Py_TYPE(self);
  type->tp_free(self);
  Py_DECREF(type);
}

PYB26_NOINLINE inline bool
override_is_calling_itself(PyObject *bound, PyObject *self,
                           PyThreadState *state) noexcept {
  PyObject *fn = bound;
  if (PyMethod_Check(fn))
    fn = PyMethod_GET_FUNCTION(fn);
  if (!PyFunction_Check(fn))
    return false;
  if (!current_frame_matches_function(state, fn))
    return false;
  callback_error_scope error{state};
  PyFrameObject *frame = PyEval_GetFrame();
  if (!frame)
    return false;
  PyCodeObject *fcode = PyFrame_GetCode(frame);
  const bool same_code =
      reinterpret_cast<PyObject *>(fcode) == PyFunction_GET_CODE(fn);
  if (!same_code) {
    error.discard(reinterpret_cast<PyObject *>(fcode));
    return false;
  }
  bool same_self = false;
  PyObject *varnames = PyCode_GetVarnames(fcode);
  error.discard(reinterpret_cast<PyObject *>(fcode));
  if (varnames) {
    if (PyTuple_GET_SIZE(varnames) > 0) {
      PyObject *locals = PyFrame_GetLocals(frame);
      if (locals) {
        PyObject *first =
            PyObject_GetItem(locals, PyTuple_GET_ITEM(varnames, 0));
        same_self = (first == self);
        error.discard(first);
        error.discard(locals);
      }
    }
    error.discard(varnames);
  }
  return same_self;
}

struct override_target {
  PyObject *function = nullptr;
  PyObject *receiver = nullptr;

  override_target() noexcept = default;
  explicit override_target(PyObject *fn, PyObject *self = nullptr) noexcept
      : function(fn), receiver(self) {}
  override_target(const override_target &) = delete;
  override_target(override_target &&other) noexcept
      : function(std::exchange(other.function, nullptr)),
        receiver(std::exchange(other.receiver, nullptr)) {}
  ~override_target() {
    if (function || receiver) {
      callback_error_scope error;
      clear(error);
    }
  }
  explicit operator bool() const noexcept { return function != nullptr; }
  void clear(callback_error_scope &error) noexcept {
    error.discard(std::exchange(function, nullptr));
    error.discard(std::exchange(receiver, nullptr));
  }
};

inline PyObject *override_mro_snapshot(PyTypeObject *type) noexcept {
#ifdef Py_GIL_DISABLED
  static const PyGetSetDef *const mro_getter = [] {
    for (auto *entry = PyType_Type.tp_getset; entry->name; ++entry)
      if (std::strcmp(entry->name, "__mro__") == 0)
        return entry;
    return static_cast<PyGetSetDef *>(nullptr);
  }();
  if (!mro_getter) [[unlikely]] {
    PyErr_SetString(PyExc_SystemError,
                    "CPython type MRO getter is unavailable");
    return nullptr;
  }
  return mro_getter->get(reinterpret_cast<PyObject *>(type),
                         mro_getter->closure);
#else
  return Py_XNewRef(type->tp_mro);
#endif
}

struct override_cache_entry {
  PyTypeObject *type = nullptr;
  PyTypeObject *base = nullptr;
  _Py_Identifier *name = nullptr;
  std::uint64_t state = 0;
  unsigned version = 0;
  int present = 0;
};
PYB26_HIDDEN PYB26_TLS_MODEL inline thread_local override_cache_entry
    override_cache[16];

PYB26_ALWAYS_INLINE std::size_t
override_cache_index(PyTypeObject *type, _Py_Identifier &identifier) noexcept {
  return ((reinterpret_cast<std::uintptr_t>(type) >> 4) ^
          (reinterpret_cast<std::uintptr_t>(&identifier) >> 4)) &
         15;
}

PYB26_ALWAYS_INLINE int
cached_override_present(PyTypeObject *type, PyTypeObject *base,
                        _Py_Identifier &identifier) noexcept {
  const auto state = type_state();
  const auto version = cpython::type_version(type);
  const auto &cached = override_cache[override_cache_index(type, identifier)];
  if (version && cached.version == version && cached.type == type &&
      cached.base == base && cached.name == &identifier &&
      cached.state == state)
    return cached.present;
  return -1;
}

PYB26_NOINLINE inline int override_present(PyTypeObject *type,
                                           PyTypeObject *base,
                                           _Py_Identifier &identifier,
                                           PyObject *&name,
                                           PyThreadState *thread) noexcept {
  const auto state = type_state();
  auto version = cpython::type_version(type);
  if (!version && PyUnstable_Type_AssignVersionTag(type))
    version = cpython::type_version(type);
  name = _PyUnicode_FromId(&identifier);
  if (!name)
    return -1;
  callback_error_scope error{thread};
  PyObject *mro = override_mro_snapshot(type);
  if (!mro)
    return cpython::current_exception(thread) ? -1 : 0;
  if (!PyTuple_Check(mro)) [[unlikely]] {
    error.discard(mro);
    return 0;
  }
  int present = 0;
  const Py_ssize_t n = PyTuple_GET_SIZE(mro);
  for (Py_ssize_t i = 0; i < n; ++i) {
    auto *cls = reinterpret_cast<PyTypeObject *>(PyTuple_GET_ITEM(mro, i));
    if (cls == base)
      break;
    PyObject *dict = PyType_GetDict(cls);
    if (!dict)
      continue;
    present = PyDict_Contains(dict, name);
    error.discard(dict);
    if (present != 0)
      break;
  }
  error.discard(mro);
  if (cpython::current_exception(thread))
    return -1;
  if (version && version == cpython::type_version(type) &&
      state == type_state())
    override_cache[override_cache_index(type, identifier)] = {
        type, base, &identifier, state, version, present};
  return present;
}

PYB26_NOINLINE inline override_target
find_override(PyObject *self, PyTypeObject *cpp_type,
              _Py_Identifier &identifier,
              PyThreadState *state = cpython::thread_state()) noexcept {
  callback_error_scope error{state};
  if (!self || !cpp_type) [[unlikely]]
    return {};
  auto *type = Py_TYPE(self);
  PyObject *name = nullptr;
  int present = type == cpp_type
                    ? 0
                    : cached_override_present(type, cpp_type, identifier);
  if (present < 0) [[unlikely]] {
    Py_INCREF(type);
    present = override_present(type, cpp_type, identifier, name, state);
    error.discard(reinterpret_cast<PyObject *>(type));
  }
  if (present != 1 || cpython::current_exception(state))
    return {};
  if (!name)
    name = _PyUnicode_FromId(&identifier);
  if (!name)
    return {};
  PyObject *fn = nullptr;
  const bool method = cpython::get_method(self, name, &fn);
  if (!fn)
    return {};
  override_target target(fn, method ? Py_NewRef(self) : nullptr);
  if (override_is_calling_itself(fn, self, state) ||
      cpython::current_exception(state)) [[unlikely]]
    return {};
  return target;
}

PYB26_COLD inline void set_pure_not_overridden(const char *base,
                                               const char *method) noexcept {
  if (!PyErr_Occurred())
    PyErr_Format(
        PyExc_RuntimeError,
        "pyb26: pure virtual method \"%s::%s\" is not overridden in Python",
        base, method);
}

template <class R> PYB26_NOINLINE R fail_override() {
  return python_failure_return<R>();
}

template <class R>
PYB26_ALWAYS_INLINE R finish_override(PyObject *res, PyThreadState *state,
                                      callback_error_scope &error) {
  return finish_python_call<R>(state, res, error,
                               "pyb26: override returned an incompatible type");
}

template <class R> PYB26_ALWAYS_INLINE R finish_override(PyObject *res) {
  auto *state = cpython::thread_state();
  callback_error_scope error{state};
  return finish_override<R>(res, state, error);
}

template <class R, class... A>
PYB26_NOINLINE R call_override_in_state(PyThreadState *state,
                                        override_target target, A &&...args) {
  callback_error_scope error{state};
  if (cpython::current_exception(state)) [[unlikely]] {
    target.clear(error);
    return fail_override<R>();
  }
  constexpr std::size_t N = sizeof...(A);
  PyObject *storage[N + 1] = {target.receiver};
  PyObject **tmp = storage + 1;
  if (!build_python_call_arguments(state, tmp, std::index_sequence_for<A...>{},
                                   std::forward<A>(args)...)) [[unlikely]] {
    clear_python_call_arguments<N>(tmp, error);
    target.clear(error);
    if (!cpython::current_exception(state))
      PyErr_SetString(
          PyExc_TypeError,
          "pyb26: failed to convert an override argument to Python");
    return fail_override<R>();
  }
  PyObject *res =
      PyObject_Vectorcall(target.function, target.receiver ? storage : tmp,
                          N + (target.receiver != nullptr), nullptr);
  clear_python_call_arguments<N>(tmp, error);
  target.clear(error);
  return finish_override<R>(res, state, error);
}

template <class R, class... A>
PYB26_ALWAYS_INLINE R call_override(override_target target, A &&...args) {
  return call_override_in_state<R>(cpython::thread_state(), std::move(target),
                                   std::forward<A>(args)...);
}

} // namespace detail
} // namespace pyb26

#define PYB26_OVERRIDE(ret, Base, method, ...)                                 \
  do {                                                                         \
    const auto *pyb26_ovr_owner_ =                                             \
        static_cast<const ::pyb26::trampoline_self *>(this);                   \
    ::pyb26::detail::callback_invocation_scope pyb26_ovr_scope_(               \
        pyb26_ovr_owner_->py_context);                                         \
    if (!pyb26_ovr_scope_)                                                     \
      return ::pyb26::detail::fail_override<ret>();                            \
    auto *pyb26_ovr_state_ = pyb26_ovr_scope_.thread_state();                  \
    if (::pyb26::detail::cpython::current_exception(pyb26_ovr_state_))         \
      return ::pyb26::detail::fail_override<ret>();                            \
    static _Py_Identifier pyb26_ovr_name_{.string = #method, .index = -1};     \
    PyObject *pyb26_ovr_self_ =                                                \
        static_cast<const ::pyb26::trampoline_self *>(this)->py_self;          \
    ::pyb26::detail::type_record *pyb26_ovr_tr_ =                              \
        ::pyb26::detail::resolve<Base>(pyb26_ovr_state_->interp);              \
    auto pyb26_ovr_fn_ = ::pyb26::detail::find_override(                       \
        pyb26_ovr_self_, pyb26_ovr_tr_ ? pyb26_ovr_tr_->py_type : nullptr,     \
        pyb26_ovr_name_, pyb26_ovr_state_);                                    \
    if (::pyb26::detail::cpython::current_exception(pyb26_ovr_state_))         \
      return ::pyb26::detail::fail_override<ret>();                            \
    if (pyb26_ovr_fn_)                                                         \
      return ::pyb26::detail::call_override_in_state<ret>(                     \
          pyb26_ovr_state_,                                                    \
          std::move(pyb26_ovr_fn_) __VA_OPT__(, ) __VA_ARGS__);                \
    return this->Base::method(__VA_ARGS__);                                    \
  } while (0)

#define PYB26_OVERRIDE_PURE(ret, Base, method, ...)                            \
  do {                                                                         \
    const auto *pyb26_ovr_owner_ =                                             \
        static_cast<const ::pyb26::trampoline_self *>(this);                   \
    ::pyb26::detail::callback_invocation_scope pyb26_ovr_scope_(               \
        pyb26_ovr_owner_->py_context);                                         \
    if (!pyb26_ovr_scope_)                                                     \
      return ::pyb26::detail::fail_override<ret>();                            \
    auto *pyb26_ovr_state_ = pyb26_ovr_scope_.thread_state();                  \
    if (::pyb26::detail::cpython::current_exception(pyb26_ovr_state_))         \
      return ::pyb26::detail::fail_override<ret>();                            \
    static _Py_Identifier pyb26_ovr_name_{.string = #method, .index = -1};     \
    PyObject *pyb26_ovr_self_ =                                                \
        static_cast<const ::pyb26::trampoline_self *>(this)->py_self;          \
    ::pyb26::detail::type_record *pyb26_ovr_tr_ =                              \
        ::pyb26::detail::resolve<Base>(pyb26_ovr_state_->interp);              \
    auto pyb26_ovr_fn_ = ::pyb26::detail::find_override(                       \
        pyb26_ovr_self_, pyb26_ovr_tr_ ? pyb26_ovr_tr_->py_type : nullptr,     \
        pyb26_ovr_name_, pyb26_ovr_state_);                                    \
    if (::pyb26::detail::cpython::current_exception(pyb26_ovr_state_))         \
      return ::pyb26::detail::fail_override<ret>();                            \
    if (pyb26_ovr_fn_)                                                         \
      return ::pyb26::detail::call_override_in_state<ret>(                     \
          pyb26_ovr_state_,                                                    \
          std::move(pyb26_ovr_fn_) __VA_OPT__(, ) __VA_ARGS__);                \
    ::pyb26::detail::set_pure_not_overridden(#Base, #method);                  \
    return ::pyb26::detail::fail_override<ret>();                              \
  } while (0)
