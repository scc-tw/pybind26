#pragma once

#include "detail/struct_cast.hpp"
#include "pyb26.hpp"

#if defined(__cpp_reflection) || defined(__cpp_impl_reflection)

#include <meta>

#include <array>
#include <concepts>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

namespace pyb26 {

namespace detail {

template <class T> consteval Py_ssize_t auto_member_count() {
  Py_ssize_t n = 0;
  for (std::meta::info m : std::meta::nonstatic_data_members_of(
           ^^T, std::meta::access_context::unchecked()))
    if (struct_cast_member(m))
      ++n;
  return n;
}

template <class T> inline T *auto_cast_self(PyObject *self) noexcept {
  T *p = cast_to<T>(self);
  if (!p) [[unlikely]]
    PyErr_SetString(PyExc_TypeError, "pyb26: invalid bound-class instance");
  return p;
}

template <class T>
consteval std::string auto_repr_label(std::meta::info member) {
  bool first = true;
  for (std::meta::info m : std::meta::nonstatic_data_members_of(
           ^^T, std::meta::access_context::unchecked())) {
    if (struct_cast_member(m)) {
      if (m == member)
        return std::string(first ? "(" : ", ") +
               std::string(std::meta::identifier_of(m)) + "=";
      first = false;
    }
  }
  return {};
}

consteval bool auto_repr_ascii(std::string_view text) {
  for (unsigned char ch : text)
    if (ch > 127)
      return false;
  return true;
}

template <class T> PyObject *auto_repr_impl(PyObject *self) noexcept {
  T *obj = auto_cast_self<T>(self);
  if (!obj) [[unlikely]]
    return nullptr;
  auto name = object::steal(PyType_GetName(Py_TYPE(self)));
  if (!name) [[unlikely]]
    return nullptr;
  constexpr std::size_t count =
      static_cast<std::size_t>(auto_member_count<T>());
  std::array<object, count> parts;
  Py_ssize_t length = PyUnicode_GET_LENGTH(name.ptr());
  if (length > PY_SSIZE_T_MAX - 2) [[unlikely]]
    return PyErr_NoMemory();
  length += count ? 1 : 2;
  std::size_t index = 0;
  template for (constexpr auto m :
                std::define_static_array(std::meta::nonstatic_data_members_of(
                    ^^T, std::meta::access_context::unchecked()))) {
    if constexpr (struct_cast_member(m)) {
      constexpr std::string_view label =
          std::define_static_string(auto_repr_label<T>(m));
      using MT = std::remove_cvref_t<decltype((*obj).[:m:])>;
      auto value = object::steal(type_caster<MT>::cast((*obj).[:m:]));
      if (!value) [[unlikely]]
        return nullptr;
      parts[index] = object::steal(PyObject_Repr(value.ptr()));
      if (!parts[index]) [[unlikely]]
        return nullptr;
      const Py_ssize_t size = PyUnicode_GET_LENGTH(parts[index].ptr());
      if (static_cast<Py_ssize_t>(label.size()) > PY_SSIZE_T_MAX - length)
          [[unlikely]]
        return PyErr_NoMemory();
      length += static_cast<Py_ssize_t>(label.size());
      if (size > PY_SSIZE_T_MAX - length) [[unlikely]]
        return PyErr_NoMemory();
      length += size;
      ++index;
    }
  }
  std::unique_ptr<PyUnicodeWriter, decltype(&PyUnicodeWriter_Discard)> writer(
      PyUnicodeWriter_Create(length), &PyUnicodeWriter_Discard);
  if (!writer) [[unlikely]]
    return nullptr;
  if (PyUnicodeWriter_WriteSubstring(writer.get(), name.ptr(), 0,
                                     PyUnicode_GET_LENGTH(name.ptr())) < 0)
      [[unlikely]]
    return nullptr;
  index = 0;
  template for (constexpr auto m :
                std::define_static_array(std::meta::nonstatic_data_members_of(
                    ^^T, std::meta::access_context::unchecked()))) {
    if constexpr (struct_cast_member(m)) {
      constexpr std::string_view label =
          std::define_static_string(auto_repr_label<T>(m));
      const int status = auto_repr_ascii(label)
                             ? PyUnicodeWriter_WriteASCII(
                                   writer.get(), label.data(),
                                   static_cast<Py_ssize_t>(label.size()))
                             : PyUnicodeWriter_WriteUTF8(
                                   writer.get(), label.data(),
                                   static_cast<Py_ssize_t>(label.size()));
      if (status < 0) [[unlikely]]
        return nullptr;
      PyObject *part = parts[index++].ptr();
      if (PyUnicodeWriter_WriteSubstring(writer.get(), part, 0,
                                         PyUnicode_GET_LENGTH(part)) < 0)
          [[unlikely]]
        return nullptr;
    }
  }
  if constexpr (count == 0) {
    if (PyUnicodeWriter_WriteChar(writer.get(), '(') < 0) [[unlikely]]
      return nullptr;
  }
  if (PyUnicodeWriter_WriteChar(writer.get(), ')') < 0) [[unlikely]]
    return nullptr;
  return PyUnicodeWriter_Finish(writer.release());
}

template <class T>
PyObject *auto_eq_impl(PyObject *self, PyObject *other) noexcept {
  T *a = auto_cast_self<T>(self);
  if (!a) [[unlikely]]
    return nullptr;
  T *b = cast_to<T>(other);
  if (!b)
    Py_RETURN_NOTIMPLEMENTED;
  bool eq = true;
  template for (constexpr auto m :
                std::define_static_array(std::meta::nonstatic_data_members_of(
                    ^^T, std::meta::access_context::unchecked()))) {
    if constexpr (struct_cast_member(m)) {
      using MT = std::remove_cvref_t<decltype((*a).[:m:])>;
      static_assert(std::equality_comparable<MT>,
                    "pyb26::auto_eq<T>: every participating (public, named, "
                    "non-bit-field) data member must be equality-comparable — "
                    "give the member's type an operator==, or do not install "
                    "auto_eq on this class.");
      if constexpr (std::equality_comparable<MT>) {
        if (eq)
          eq = ((*a).[:m:] == (*b).[:m:]);
      }
    }
  }
  return PyBool_FromLong(eq);
}

template <class T>
PyObject *auto_ne_impl(PyObject *self, PyObject *other) noexcept {
  PyObject *r = auto_eq_impl<T>(self, other);
  if (!r || r == Py_NotImplemented)
    return r;
  const bool eq = (r == Py_True);
  Py_DECREF(r);
  return PyBool_FromLong(!eq);
}

template <class T> inline void auto_default_unhashable() noexcept {
  type_record *tr = resolve<T>();
  if (!tr || !tr->py_type) [[unlikely]]
    return;
  PyObject *d = PyType_GetDict(tr->py_type);
  const int has = d ? PyDict_ContainsString(d, "__hash__") : -1;
  Py_XDECREF(d);
  if (has != 0) {
    if (has < 0)
      PyErr_Clear();
    return;
  }
  (void)PyObject_SetAttrString(reinterpret_cast<PyObject *>(tr->py_type),
                               "__hash__", Py_None);
}

template <class T> PyObject *auto_hash_impl(PyObject *self) noexcept {
  T *obj = auto_cast_self<T>(self);
  if (!obj) [[unlikely]]
    return nullptr;
  Py_uhash_t x = static_cast<Py_uhash_t>(0x345678UL);
  bool ok = true;
  template for (constexpr auto m :
                std::define_static_array(std::meta::nonstatic_data_members_of(
                    ^^T, std::meta::access_context::unchecked()))) {
    if constexpr (struct_cast_member(m)) {
      if (ok) {
        using MT = std::remove_cvref_t<decltype((*obj).[:m:])>;
        PyObject *v = type_caster<MT>::cast((*obj).[:m:]);
        if (!v) {
          ok = false;
        } else {
          const Py_hash_t h = PyObject_Hash(v);
          Py_DECREF(v);
          if (h == -1) {
            ok = false;
          } else {
            x = (x * static_cast<Py_uhash_t>(1000003UL)) ^
                static_cast<Py_uhash_t>(h);
          }
        }
      }
    }
  }
  if (!ok) [[unlikely]]
    return nullptr;
  Py_hash_t r = static_cast<Py_hash_t>(x);
  if (r == -1)
    r = -2;
  return PyLong_FromSsize_t(r);
}

template <class T> PyObject *auto_getstate_impl(PyObject *self) noexcept {
  T *obj = auto_cast_self<T>(self);
  if (!obj) [[unlikely]]
    return nullptr;
  constexpr Py_ssize_t N = auto_member_count<T>();
  PyObject *t = PyTuple_New(N);
  if (!t) [[unlikely]]
    return nullptr;
  Py_ssize_t idx = 0;
  bool ok = true;
  template for (constexpr auto m :
                std::define_static_array(std::meta::nonstatic_data_members_of(
                    ^^T, std::meta::access_context::unchecked()))) {
    if constexpr (struct_cast_member(m)) {
      if (ok) {
        using MT = std::remove_cvref_t<decltype((*obj).[:m:])>;
        PyObject *v = type_caster<MT>::cast((*obj).[:m:]);
        if (!v)
          ok = false;
        else
          PyTuple_SET_ITEM(t, idx++, v);
      }
    }
  }
  if (!ok) [[unlikely]] {
    Py_DECREF(t);
    return nullptr;
  }
  return t;
}

template <class T>
PyObject *auto_setstate_impl(PyObject *self, PyObject *state) noexcept {
  T *obj = auto_cast_self<T>(self);
  if (!obj) [[unlikely]]
    return nullptr;
  constexpr Py_ssize_t N = auto_member_count<T>();
  if (!PyTuple_Check(state)) {
    PyErr_Format(
        PyExc_TypeError,
        "%.200s.__setstate__() expects a tuple of %zd items, got %.80s",
        Py_TYPE(self)->tp_name, N, Py_TYPE(state)->tp_name);
    return nullptr;
  }
  if (PyTuple_GET_SIZE(state) != N) {
    PyErr_Format(PyExc_ValueError,
                 "%.200s.__setstate__() expects a tuple of %zd items, got %zd",
                 Py_TYPE(self)->tp_name, N, PyTuple_GET_SIZE(state));
    return nullptr;
  }
  Py_ssize_t idx = 0;
  bool ok = true;
  template for (constexpr auto m :
                std::define_static_array(std::meta::nonstatic_data_members_of(
                    ^^T, std::meta::access_context::unchecked()))) {
    if constexpr (struct_cast_member(m)) {
      using MDecl = decltype((*obj).[:m:]);
      using MT = std::remove_cvref_t<MDecl>;
      static_assert(std::is_assignable_v<MDecl &, MT &&>,
                    "pyb26::auto_pickle<T>: every participating (public, "
                    "named, non-bit-field) data member must be assignable — "
                    "a const member cannot be restored by __setstate__.");
      if constexpr (std::is_assignable_v<MDecl &, MT &&>) {
        if (ok) {
          PyObject *item = PyTuple_GET_ITEM(state, idx);
          ++idx;
          type_caster<MT> c;
          reject_lease_escape<MT>();
          if (!c.load(item)) {
            if (!PyErr_Occurred())
              PyErr_Format(PyExc_TypeError,
                           "%.200s.__setstate__(): item %zd has an "
                           "incompatible type (%.80s)",
                           Py_TYPE(self)->tp_name, idx - 1,
                           Py_TYPE(item)->tp_name);
            ok = false;
          } else {
            (*obj).[:m:] = c.template as<MT>();
          }
        }
      }
    }
  }
  if (!ok) [[unlikely]]
    return nullptr;
  Py_RETURN_NONE;
}

template <class T> PyObject *auto_reduce_impl(PyObject *self) noexcept {
  PyObject *state = auto_getstate_impl<T>(self);
  if (!state) [[unlikely]]
    return nullptr;
  PyObject *noargs = PyTuple_New(0);
  if (!noargs) [[unlikely]] {
    Py_DECREF(state);
    return nullptr;
  }
  PyObject *r = PyTuple_Pack(3, reinterpret_cast<PyObject *>(Py_TYPE(self)),
                             noargs, state);
  Py_DECREF(noargs);
  Py_DECREF(state);
  return r;
}

template <class T> inline void auto_fill_module() noexcept {
  type_record *tr = resolve<T>();
  if (!tr || !tr->py_type) [[unlikely]]
    return;
  PyObject *d = PyType_GetDict(tr->py_type);
  const int has = d ? PyDict_ContainsString(d, "__module__") : -1;
  Py_XDECREF(d);
  if (has != 0) {
    if (has < 0)
      PyErr_Clear();
    return;
  }
  PyObject *mod = PyType_GetModule(tr->py_type);
  if (!mod) {
    PyErr_Clear();
    return;
  }
  PyObject *name = PyModule_GetNameObject(mod);
  if (!name) {
    PyErr_Clear();
    return;
  }
  if (PyObject_SetAttrString(reinterpret_cast<PyObject *>(tr->py_type),
                             "__module__", name) != 0)
    PyErr_Clear();
  Py_DECREF(name);
}

} // namespace detail

template <class T, class... Options>
class_<T, Options...> &auto_repr(class_<T, Options...> &c) {
  c.template def_fn<&detail::auto_repr_impl<T>>("__repr__");
  return c;
}

template <class T, class... Options>
class_<T, Options...> &auto_eq(class_<T, Options...> &c) {
  c.template def_fn<&detail::auto_eq_impl<T>>("__eq__");
  c.template def_fn<&detail::auto_ne_impl<T>>("__ne__");
  detail::auto_default_unhashable<T>();
  return c;
}

template <class T, class... Options>
class_<T, Options...> &auto_hash(class_<T, Options...> &c) {
  c.template def_fn<&detail::auto_hash_impl<T>>("__hash__");
  return c;
}

template <class T, class... Options>
class_<T, Options...> &auto_pickle(class_<T, Options...> &c) {
  c.template def_fn<&detail::auto_getstate_impl<T>>("__getstate__");
  c.template def_fn<&detail::auto_setstate_impl<T>>("__setstate__");
  c.template def_fn<&detail::auto_reduce_impl<T>>("__reduce__");
  detail::auto_fill_module<T>();
  return c;
}

} // namespace pyb26

#else

namespace pyb26 {

#define PYB26_AUTO_REQUIRES_REFLECTION                                         \
  static_assert(sizeof(T) == 0,                                                \
                "pyb26/auto.hpp (auto_repr/auto_eq/auto_hash/auto_pickle) "    \
                "requires C++26 static reflection. Compile this "              \
                "translation unit with `-std=c++26 -freflection` (GCC 16+).")

template <class T, class... Options>
class_<T, Options...> &auto_repr(class_<T, Options...> &c) {
  PYB26_AUTO_REQUIRES_REFLECTION;
  return c;
}
template <class T, class... Options>
class_<T, Options...> &auto_eq(class_<T, Options...> &c) {
  PYB26_AUTO_REQUIRES_REFLECTION;
  return c;
}
template <class T, class... Options>
class_<T, Options...> &auto_hash(class_<T, Options...> &c) {
  PYB26_AUTO_REQUIRES_REFLECTION;
  return c;
}
template <class T, class... Options>
class_<T, Options...> &auto_pickle(class_<T, Options...> &c) {
  PYB26_AUTO_REQUIRES_REFLECTION;
  return c;
}

#undef PYB26_AUTO_REQUIRES_REFLECTION

} // namespace pyb26

#endif
