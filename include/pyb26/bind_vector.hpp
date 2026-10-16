#pragma once

#include "detail/opaque_conversion.hpp"
#include "pyb26.hpp"

#include <concepts>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pyb26 {
namespace detail {

template <class V> PyObject *vec_elem_out(typename V::value_type &e) noexcept {
  using T = typename V::value_type;
  return opaque_copy_output<T>(e);
}

template <class V> PYB26_ALWAYS_INLINE V *vec_self(PyObject *self) noexcept {
  V *v = cast_to<V>(self);
  if (!v) [[unlikely]]
    PyErr_SetString(PyExc_TypeError, "pyb26: invalid bound-vector instance");
  return v;
}

template <class V> Py_ssize_t vec_len(PyObject *self) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return -1;
  Py_ssize_t n;
  Py_BEGIN_CRITICAL_SECTION(self);
  n = static_cast<Py_ssize_t>(v->size());
  Py_END_CRITICAL_SECTION();
  return n;
}

template <class V>
PyObject *vec_sq_item(PyObject *self, Py_ssize_t i) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  PyObject *r = nullptr;
  Py_BEGIN_CRITICAL_SECTION(self);
  if (i < 0 || i >= static_cast<Py_ssize_t>(v->size()))
    PyErr_SetString(PyExc_IndexError, "vector index out of range");
  else
    r = vec_elem_out<V>((*v)[static_cast<std::size_t>(i)]);
  Py_END_CRITICAL_SECTION();
  return r;
}

template <class V>
PyObject *vec_subscript(PyObject *self, PyObject *key) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  if (PyIndex_Check(key)) {
    Py_ssize_t i = PyNumber_AsSsize_t(key, PyExc_IndexError);
    if (i == -1 && PyErr_Occurred()) [[unlikely]]
      return nullptr;
    PyObject *r = nullptr;
    Py_BEGIN_CRITICAL_SECTION(self);
    const Py_ssize_t n = static_cast<Py_ssize_t>(v->size());
    if (i < 0)
      i += n;
    if (i < 0 || i >= n)
      PyErr_SetString(PyExc_IndexError, "vector index out of range");
    else
      r = vec_elem_out<V>((*v)[static_cast<std::size_t>(i)]);
    Py_END_CRITICAL_SECTION();
    return r;
  }
  if (PySlice_Check(key)) {
    callback_error_scope error;
    V out;
    callback_error_checkpoint checkpoint{error};
    bool ok = true;
    Py_BEGIN_CRITICAL_SECTION(self);
    Py_ssize_t start, stop, step;
    if (PySlice_Unpack(key, &start, &stop, &step) < 0) {
      ok = false;
    } else {
      const Py_ssize_t n = PySlice_AdjustIndices(
          static_cast<Py_ssize_t>(v->size()), &start, &stop, step);
      out.reserve(static_cast<std::size_t>(n));
      for (Py_ssize_t k = 0, i2 = start; k < n && !error.capture();
           ++k, i2 += step)
        out.push_back((*v)[static_cast<std::size_t>(i2)]);
    }
    Py_END_CRITICAL_SECTION();
    if (!ok || error.capture()) [[unlikely]]
      return nullptr;
    return make_instance<V>([&](void *m) { ::new (m) V(std::move(out)); });
  }
  PyErr_Format(PyExc_TypeError,
               "vector indices must be integers or slices, not %.80s",
               Py_TYPE(key)->tp_name);
  return nullptr;
}

template <class V>
int vec_ass_subscript(PyObject *self, PyObject *key, PyObject *value) noexcept {
  using T = typename V::value_type;
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return -1;
  if (!PyIndex_Check(key)) [[unlikely]] {
    PyErr_SetString(
        PyExc_TypeError,
        PySlice_Check(key)
            ? "pyb26: slice assignment/deletion is not supported (yet)"
            : "vector indices must be integers");
    return -1;
  }
  Py_ssize_t i = PyNumber_AsSsize_t(key, PyExc_IndexError);
  if (i == -1 && PyErr_Occurred()) [[unlikely]]
    return -1;

  if (value) {
    callback_error_scope error;
    std::optional<T> item;
    callback_error_checkpoint checkpoint{error};
    if (!opaque_load_item(value, item, error)) [[unlikely]] {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError,
                        "pyb26: incompatible element type for this vector");
      return -1;
    }
    int rc = 0;
    Py_BEGIN_CRITICAL_SECTION(self);
    const Py_ssize_t n = static_cast<Py_ssize_t>(v->size());
    if (i < 0)
      i += n;
    if (i < 0 || i >= n) {
      PyErr_SetString(PyExc_IndexError, "vector assignment index out of range");
      rc = -1;
    } else {
      (*v)[static_cast<std::size_t>(i)] = std::move(*item);
    }
    Py_END_CRITICAL_SECTION();
    error.capture();
    item.reset();
    return error.capture() ? -1 : rc;
  }
  int rc = 0;
  Py_BEGIN_CRITICAL_SECTION(self);
  const Py_ssize_t n = static_cast<Py_ssize_t>(v->size());
  if (i < 0)
    i += n;
  if (i < 0 || i >= n) {
    PyErr_SetString(PyExc_IndexError, "vector deletion index out of range");
    rc = -1;
  } else {
    v->erase(v->begin() + i);
  }
  Py_END_CRITICAL_SECTION();
  return PyErr_Occurred() ? -1 : rc;
}

template <class V> int vec_contains(PyObject *self, PyObject *probe) noexcept {
  using T = typename V::value_type;
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return -1;
  if constexpr (std::equality_comparable<T>) {
    callback_error_scope error;
    std::optional<T> needle;
    callback_error_checkpoint checkpoint{error};
    if (!opaque_load_item(probe, needle, error))
      return error.capture() ? -1 : 0;
    int found = 0;
    Py_BEGIN_CRITICAL_SECTION(self);
    for (const T &e : *v)
      if (e == *needle) {
        found = 1;
        break;
      }
    Py_END_CRITICAL_SECTION();
    error.capture();
    needle.reset();
    return error.capture() ? -1 : found;
  } else {
    (void)probe;
    PyErr_SetString(
        PyExc_TypeError,
        "pyb26: this vector's elements are not equality-comparable");
    return -1;
  }
}

template <class V>
PyObject *vec_append(PyObject *self, const typename V::value_type &x) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  callback_error_scope error;
  opaque_with_copy(x, error, [&](auto &&value) {
    Py_BEGIN_CRITICAL_SECTION(self);
    v->push_back(std::forward<decltype(value)>(value));
    Py_END_CRITICAL_SECTION();
    return true;
  });
  if (error.capture())
    return nullptr;
  Py_RETURN_NONE;
}

template <class V> bool vec_stage(PyObject *iterable, V &out) noexcept {
  using T = typename V::value_type;
  if (PyList_Check(iterable) || PyTuple_Check(iterable)) {
    const Py_ssize_t n = PyList_Check(iterable) ? PyList_GET_SIZE(iterable)
                                                : PyTuple_GET_SIZE(iterable);
    out.reserve(static_cast<std::size_t>(n));
  }
  return stl_for_each<T>(iterable, [&out](PyObject *item, auto *error) {
    return stl_convert_item<T>(item, error, [&out](auto &&value) {
      out.push_back(std::forward<decltype(value)>(value));
      return true;
    });
  });
}

template <class V>
PyObject *vec_extend(PyObject *self, PyObject *iterable) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  callback_error_scope error;
  if (V *src = cast_to<V>(iterable)) {
    Py_BEGIN_CRITICAL_SECTION2(self, iterable);
    if (v == src) {
      V staged(*src);
      callback_error_checkpoint checkpoint{error};
      if (!error.capture())
        v->insert(v->end(), std::make_move_iterator(staged.begin()),
                  std::make_move_iterator(staged.end()));
    } else {
      v->insert(v->end(), src->begin(), src->end());
    }
    Py_END_CRITICAL_SECTION2();
    if (error.capture())
      return nullptr;
    Py_RETURN_NONE;
  }
  V staged;
  callback_error_checkpoint checkpoint{error};
  if (!vec_stage<V>(iterable, staged)) [[unlikely]] {
    if (!PyErr_Occurred())
      PyErr_SetString(
          PyExc_TypeError,
          "pyb26: extend() expects an iterable of compatible elements");
    return nullptr;
  }
  Py_BEGIN_CRITICAL_SECTION(self);
  v->insert(v->end(), std::make_move_iterator(staged.begin()),
            std::make_move_iterator(staged.end()));
  Py_END_CRITICAL_SECTION();
  if (error.capture())
    return nullptr;
  Py_RETURN_NONE;
}

template <class V>
PyObject *vec_insert(PyObject *self, Py_ssize_t i,
                     const typename V::value_type &x) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  callback_error_scope error;
  opaque_with_copy(x, error, [&](auto &&value) {
    Py_BEGIN_CRITICAL_SECTION(self);
    const Py_ssize_t n = static_cast<Py_ssize_t>(v->size());
    if (i < 0)
      i += n;
    if (i < 0)
      i = 0;
    if (i > n)
      i = n;
    v->insert(v->begin() + i, std::forward<decltype(value)>(value));
    Py_END_CRITICAL_SECTION();
    return true;
  });
  if (error.capture())
    return nullptr;
  Py_RETURN_NONE;
}

template <class V> PyObject *vec_pop_at(PyObject *self, Py_ssize_t i) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  PyObject *r = nullptr;
  callback_error_scope error;
  Py_BEGIN_CRITICAL_SECTION(self);
  const Py_ssize_t n = static_cast<Py_ssize_t>(v->size());
  Py_ssize_t k = i < 0 ? i + n : i;
  if (n == 0)
    PyErr_SetString(PyExc_IndexError, "pop from empty vector");
  else if (k < 0 || k >= n)
    PyErr_SetString(PyExc_IndexError, "pop index out of range");
  else {
    typename V::value_type out = std::move((*v)[static_cast<std::size_t>(k)]);
    callback_error_checkpoint checkpoint{error};
    v->erase(v->begin() + k);
    if (!error.capture())
      r = opaque_cast_value(std::move(out));
  }
  Py_END_CRITICAL_SECTION();
  return opaque_finish_output(r, error);
}
template <class V> PyObject *vec_pop_last(PyObject *self) noexcept {
  return vec_pop_at<V>(self, -1);
}

template <class V> PyObject *vec_clear(PyObject *self) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  Py_BEGIN_CRITICAL_SECTION(self);
  v->clear();
  Py_END_CRITICAL_SECTION();
  Py_RETURN_NONE;
}

template <class V>
PyObject *vec_count(PyObject *self, const typename V::value_type &x) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  if constexpr (std::equality_comparable<typename V::value_type>) {
    long n = 0;
    Py_BEGIN_CRITICAL_SECTION(self);
    for (const auto &e : *v)
      if (e == x)
        ++n;
    Py_END_CRITICAL_SECTION();
    return PyLong_FromLong(n);
  } else {
    (void)x;
    PyErr_SetString(
        PyExc_TypeError,
        "pyb26: this vector's elements are not equality-comparable");
    return nullptr;
  }
}

template <class V>
PyObject *vec_remove(PyObject *self, const typename V::value_type &x) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  if constexpr (std::equality_comparable<typename V::value_type>) {
    bool removed = false;
    Py_BEGIN_CRITICAL_SECTION(self);
    for (auto it = v->begin(); it != v->end(); ++it)
      if (*it == x) {
        v->erase(it);
        removed = true;
        break;
      }
    Py_END_CRITICAL_SECTION();
    if (!removed) [[unlikely]] {
      PyErr_SetString(PyExc_ValueError, "remove(x): x not in vector");
      return nullptr;
    }
    Py_RETURN_NONE;
  } else {
    (void)x;
    PyErr_SetString(
        PyExc_TypeError,
        "pyb26: this vector's elements are not equality-comparable");
    return nullptr;
  }
}

template <class V> PyObject *vec_eq(PyObject *self, PyObject *other) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  if constexpr (std::equality_comparable<typename V::value_type>) {
    V *o = cast_to<V>(other);
    if (!o)
      Py_RETURN_NOTIMPLEMENTED;
    bool eq;
    Py_BEGIN_CRITICAL_SECTION2(self, other);
    eq = (*v == *o);
    Py_END_CRITICAL_SECTION2();
    return PyBool_FromLong(eq);
  } else {
    (void)other;
    Py_RETURN_NOTIMPLEMENTED;
  }
}
template <class V> PyObject *vec_ne(PyObject *self, PyObject *other) noexcept {
  PyObject *r = vec_eq<V>(self, other);
  if (!r || r == Py_NotImplemented)
    return r;
  const bool eq = (r == Py_True);
  Py_DECREF(r);
  return PyBool_FromLong(!eq);
}

template <class V> PyObject *vec_repr(PyObject *self) noexcept {
  V *v = vec_self<V>(self);
  if (!v) [[unlikely]]
    return nullptr;
  std::string s = Py_TYPE(self)->tp_name;
  callback_error_scope error;
  s += "([";
  const Py_ssize_t n = vec_len<V>(self);
  for (Py_ssize_t i = 0; i < n && i < 64; ++i) {
    PyObject *e = vec_sq_item<V>(self, i);
    if (!e)
      return nullptr;
    PyObject *r = PyObject_Repr(e);
    error.discard(e);
    if (!r || error.capture()) {
      error.discard(r);
      return nullptr;
    }
    if (const char *c = PyUnicode_AsUTF8(r)) {
      s += c;
    } else {
      error.discard(r);
      return nullptr;
    }
    error.discard(r);
    if (error.capture())
      return nullptr;
    if (i + 1 < n)
      s += ", ";
  }
  if (n > 64)
    s += "...";
  s += "])";
  return PyUnicode_FromStringAndSize(s.data(),
                                     static_cast<Py_ssize_t>(s.size()));
}

template <class V>
int vec_ctor_iterable(PyObject *self, PyObject *args,
                      Py_ssize_t data_offset) noexcept {
  if (PyTuple_GET_SIZE(args) != 1)
    return 1;
  callback_error_scope error;
  V staged;
  callback_error_checkpoint checkpoint{error};
  if (!vec_stage<V>(PyTuple_GET_ITEM(args, 0), staged)) {
    if (PyErr_Occurred())
      return -1;
    return 1;
  }
  ::new (reinterpret_cast<char *>(self) + data_offset) V(std::move(staged));
  reinterpret_cast<instance *>(self)->flags |= inst_constructed | inst_owned;
  return 0;
}

} // namespace detail

template <class V, class... Extras>
class_<V> bind_vector(module_ m, const char *name, const char *doc = nullptr) {
  static_assert(
      opaque_v<V>,
      "pyb26: bind_vector<V> requires V to be OPAQUE — specialize the "
      "trait first (before any def mentioning V):  template <> inline "
      "constexpr bool pyb26::opaque_v<V> = true;  (or PYB26_OPAQUE(V)). "
      "Without it the STL caster would keep copying at every boundary.");
  using T = typename V::value_type;
  static_assert(sizeof...(Extras) == 0,
                "pyb26: bound vectors use copied elements; alias policies are "
                "unsupported");
  static_assert(std::is_copy_constructible_v<T>,
                "pyb26: bound vector elements must support independent copies");
  static_assert(
      !caster_yields_view_v<T>,
      "pyb26: a bound vector cannot hold view elements (string_view / "
      "const char* / PyObject*): appended views would outlive the "
      "Python objects they point into. Use owned elements.");

  static PyType_Slot slots[] = {
      {Py_sq_length, reinterpret_cast<void *>(&detail::vec_len<V>)},
      {Py_sq_item, reinterpret_cast<void *>(&detail::vec_sq_item<V>)},
      {Py_sq_contains, reinterpret_cast<void *>(&detail::vec_contains<V>)},
      {Py_mp_length, reinterpret_cast<void *>(&detail::vec_len<V>)},
      {Py_mp_subscript, reinterpret_cast<void *>(&detail::vec_subscript<V>)},
      {Py_mp_ass_subscript,
       reinterpret_cast<void *>(&detail::vec_ass_subscript<V>)},
  };
  class_<V> c(m, name, doc, slots,
              static_cast<int>(sizeof(slots) / sizeof(slots[0])));
  c.template ctor<>();
  c.ctor_raw(&detail::vec_ctor_iterable<V>);
  c.template def_fn<&detail::vec_append<V>>("append");
  c.template def_fn<&detail::vec_extend<V>>("extend");
  c.template def_fn<&detail::vec_insert<V>>("insert");
  c.template def_fn<&detail::vec_pop_last<V>>("pop");
  c.template def_fn<&detail::vec_pop_at<V>>("pop");
  c.template def_fn<&detail::vec_clear<V>>("clear");
  c.template def_fn<&detail::vec_count<V>>("count");
  c.template def_fn<&detail::vec_remove<V>>("remove");
  c.template def_fn<&detail::vec_eq<V>>("__eq__");
  c.template def_fn<&detail::vec_ne<V>>("__ne__");
  c.template def_fn<&detail::vec_repr<V>>("__repr__");
  return c;
}

} // namespace pyb26
