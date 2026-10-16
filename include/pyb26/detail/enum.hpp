#pragma once

#include "module.hpp"

#if defined(__cpp_reflection) || defined(__cpp_impl_reflection)

#include <meta>
#include <type_traits>

namespace pyb26 {

namespace detail {

template <class E> PyObject *enum_members_dict() {
  PyObject *members = PyDict_New();
  if (!members)
    return nullptr;
  bool ok = true;
  template for (constexpr auto e :
                std::define_static_array(std::meta::enumerators_of(^^E))) {
    if (ok) {
      constexpr const char *nm =
          std::define_static_string(std::meta::identifier_of(e));
      constexpr long long val =
          static_cast<long long>(static_cast<std::underlying_type_t<E>>([:e:]));
      PyObject *py_val = PyLong_FromLongLong(val);
      if (!py_val || PyDict_SetItemString(members, nm, py_val) != 0)
        ok = false;
      Py_XDECREF(py_val);
    }
  }
  if (!ok) {
    Py_DECREF(members);
    return nullptr;
  }
  return members;
}

inline PyObject *make_int_enum(PyObject *module, const char *name,
                               PyObject *members) {
  PyObject *enum_mod = PyImport_ImportModule("enum");
  if (!enum_mod)
    return nullptr;
  PyObject *IntEnum = PyObject_GetAttrString(enum_mod, "IntEnum");
  Py_DECREF(enum_mod);
  if (!IntEnum)
    return nullptr;
  PyObject *py_name = PyUnicode_FromString(name);
  if (!py_name) {
    Py_DECREF(IntEnum);
    return nullptr;
  }
  PyObject *enum_type =
      PyObject_CallFunctionObjArgs(IntEnum, py_name, members, nullptr);
  Py_DECREF(py_name);
  Py_DECREF(IntEnum);
  if (!enum_type)
    return nullptr;
  if (PyObject *modname = PyModule_GetNameObject(module)) {
    if (PyObject_SetAttrString(enum_type, "__module__", modname) != 0)
      PyErr_Clear();
    Py_DECREF(modname);
  } else {
    PyErr_Clear();
  }
  if (PyModule_AddObjectRef(module, name, enum_type) != 0) {
    Py_DECREF(enum_type);
    return nullptr;
  }
  return enum_type;
}

} // namespace detail

template <class E>
  requires std::is_enum_v<E>
struct enum_ {
  PyObject *type_ = nullptr;

  enum_(module_ m, const char *name = nullptr, const char * = nullptr) {
    static constexpr const char *reflected =
        std::define_static_string(std::meta::identifier_of(^^E));
    const char *the_name = name ? name : reflected;
    PyObject *members = detail::enum_members_dict<E>();
    if (!members)
      return;
    PyObject *t = detail::make_int_enum(m.ptr(), the_name, members);
    Py_DECREF(members);
    if (t) {
      type_ = t;
      detail::register_enum<E>(t);
      Py_DECREF(t);
    }
  }

  PyObject *ptr() const noexcept { return type_; }
};

} // namespace pyb26

#else

#include <type_traits>

namespace pyb26 {
template <class E>
  requires std::is_enum_v<E>
struct enum_ {
  enum_(module_, const char * = nullptr, const char * = nullptr) {
    static_assert(sizeof(E) == 0,
                  "pyb26::enum_<E> requires C++26 static reflection. Compile "
                  "this translation unit with `-std=c++26 -freflection` "
                  "(GCC 16+).");
  }
  PyObject *ptr() const noexcept { return nullptr; }
};
} // namespace pyb26

#endif
