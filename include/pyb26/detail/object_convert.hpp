#pragma once

#include "caster.hpp"

#include <string_view>
#include <type_traits>
#include <utility>

namespace pyb26 {

template <class T> T cast(handle) = delete;

template <class T> bool try_cast(handle h, T &out) noexcept {
  detail::reject_lease_escape<T>();
  if (PyErr_Occurred()) [[unlikely]]
    return false;
  type_caster<std::remove_cvref_t<T>> c;
  if (h && c.load(h.ptr()) && !PyErr_Occurred()) {
    out = c.template as<T>();
    return !PyErr_Occurred();
  }
  if (!PyErr_Occurred())
    PyErr_SetString(PyExc_TypeError, "pyb26::cast: incompatible Python object");
  return false;
}

template <class T>
  requires requires(PyObject *p) { T::check_(p); }
bool isinstance(handle h) noexcept {
  return T::check_(h.ptr());
}

template <class T> object cast(T &&v) {
  using D = std::remove_cvref_t<T>;
  if constexpr (std::is_base_of_v<handle, D>)
    return object::borrow(static_cast<const handle &>(v).ptr());
  else if constexpr (std::is_same_v<D, PyObject *>)
    return object::borrow(v);
  else
    return object::steal(type_caster<D>::cast(std::forward<T>(v)));
}

inline object import(const char *name) {
  return object::steal(PyImport_ImportModule(name));
}

namespace detail {
inline PyObject *run_string(const char *src, int start) noexcept {
  PyObject *main_mod = PyImport_AddModule("__main__");
  if (!main_mod)
    return nullptr;
  PyObject *globals = PyModule_GetDict(main_mod);
  if (!globals)
    return nullptr;
  return PyRun_String(src, start, globals, globals);
}
} // namespace detail

inline object eval(const char *expr) {
  return object::steal(detail::run_string(expr, Py_eval_input));
}
inline object eval(std::string_view expr) {
  return eval(std::string(expr).c_str());
}

inline object exec(const char *code) {
  return object::steal(detail::run_string(code, Py_file_input));
}
inline object exec(std::string_view code) {
  return exec(std::string(code).c_str());
}

} // namespace pyb26
