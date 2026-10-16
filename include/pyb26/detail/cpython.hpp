#pragma once

#include <Python.h>
#include <atomic>
#include <cpython/longintrepr.h>
#include <cstddef>

#ifdef Py_LIMITED_API
#error "pyb26 requires the version-specific CPython API."
#endif

extern "C" PyAPI_FUNC(int)
    _PyObject_GetMethod(PyObject *, PyObject *, PyObject **);

namespace pyb26::detail::cpython {

inline bool get_method(PyObject *self, PyObject *name,
                       PyObject **callable) noexcept {
  return _PyObject_GetMethod(self, name, callable) != 0;
}

inline unsigned type_version(PyTypeObject *type) noexcept {
#ifdef Py_GIL_DISABLED
  return std::atomic_ref(type->tp_version_tag).load(std::memory_order_acquire);
#else
  return type->tp_version_tag;
#endif
}

inline PyObject *dict_new_presized(std::size_t count) noexcept {
  if (count > static_cast<std::size_t>(PY_SSIZE_T_MAX))
    return PyErr_NoMemory();
  return _PyDict_NewPresized(static_cast<Py_ssize_t>(count));
}

[[gnu::always_inline]] inline PyThreadState *thread_state() noexcept {
  return PyThreadState_Get();
}
[[gnu::always_inline]] inline PyThreadState *attached_thread_state() noexcept {
  return PyThreadState_GetUnchecked();
}
[[gnu::always_inline]] inline PyInterpreterState *
current_interpreter() noexcept {
  PyThreadState *state = attached_thread_state();
  return state ? state->interp : nullptr;
}
[[gnu::always_inline]] inline PyObject *
current_exception(PyThreadState *state) noexcept {
  return state->current_exception;
}
[[gnu::always_inline]] inline void revive_object(PyObject *object) noexcept {
  _Py_NewReference(object);
}
[[gnu::always_inline]] inline PyObject **tuple_items(PyObject *tuple) noexcept {
  return reinterpret_cast<PyTupleObject *>(tuple)->ob_item;
}
[[gnu::always_inline]] inline PyObject *const *
list_items(PyObject *list) noexcept {
  return reinterpret_cast<PyListObject *>(list)->ob_item;
}
[[gnu::always_inline]] inline bool
long_is_compact(PyLongObject *value) noexcept {
  return PyUnstable_Long_IsCompact(value);
}
[[gnu::always_inline]] inline Py_ssize_t
compact_long_value(PyLongObject *value) noexcept {
  return PyUnstable_Long_CompactValue(value);
}
[[gnu::always_inline]] inline bool exact_float_value(PyObject *object,
                                                     double &value) noexcept {
  if (Py_TYPE(object) != &PyFloat_Type)
    return false;
  value = reinterpret_cast<PyFloatObject *>(object)->ob_fval;
  return true;
}
[[gnu::always_inline]] inline bool
exact_compact_ascii(PyObject *object, const char *&data,
                    std::size_t &size) noexcept {
  if (Py_TYPE(object) != &PyUnicode_Type)
    return false;
  const auto *value = reinterpret_cast<PyASCIIObject *>(object);
  if (!value->state.compact || !value->state.ascii)
    return false;
  data = reinterpret_cast<const char *>(value + 1);
  size = static_cast<std::size_t>(value->length);
  return true;
}
[[gnu::always_inline]] inline void
prepare_shared_object(PyObject *object) noexcept {
#ifdef Py_GIL_DISABLED
  if (object)
    PyUnstable_Object_EnableDeferredRefcount(object);
#else
  (void)object;
#endif
}
[[gnu::always_inline]] inline int
at_interpreter_exit(PyInterpreterState *interpreter, void (*callback)(void *),
                    void *data) noexcept {
  return PyUnstable_AtExit(interpreter, callback, data);
}

} // namespace pyb26::detail::cpython
