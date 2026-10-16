#pragma once

#include "caster.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pyb26 {

namespace detail {
template <class T> PyObject *arg_to_pyobject(T &&v) noexcept;
}

class dict : public object {
public:
  dict() : object(PyDict_New(), adopt_t{}) {}
  static bool check_(PyObject *p) noexcept { return p && PyDict_Check(p); }
  static dict borrow(PyObject *p) { return dict{Py_XNewRef(p)}; }
  static dict steal(PyObject *p) { return dict{p}; }

  Py_ssize_t size() const noexcept { return p_ ? PyDict_Size(p_) : -1; }
  int contains(handle key) const noexcept {
    if (!p_ || !key)
      return invalid();
    return PyDict_Contains(p_, key.ptr());
  }
  template <class K>
    requires(!std::is_base_of_v<handle, std::remove_cvref_t<K>>)
  int contains(K &&key) const noexcept {
    object k = object::steal(detail::arg_to_pyobject(std::forward<K>(key)));
    return k ? contains(handle{k.ptr()}) : -1;
  }
  object get(handle key) const noexcept {
    if (!p_ || !key) {
      invalid();
      return {};
    }
    PyObject *value = nullptr;
    int rc = PyDict_GetItemRef(p_, key.ptr(), &value);
    if (rc == 0)
      PyErr_SetObject(PyExc_KeyError, key.ptr());
    return object::steal(value);
  }
  template <class K>
    requires(!std::is_base_of_v<handle, std::remove_cvref_t<K>>)
  object get(K &&key) const noexcept {
    object k = object::steal(detail::arg_to_pyobject(std::forward<K>(key)));
    return k ? get(handle{k.ptr()}) : object{};
  }
  int lookup(handle key, object &value) const noexcept {
    if (!p_ || !key) {
      value = {};
      return invalid();
    }
    PyObject *found = nullptr;
    int rc = PyDict_GetItemRef(p_, key.ptr(), &found);
    value = object::steal(found);
    return rc;
  }
  template <class K>
    requires(!std::is_base_of_v<handle, std::remove_cvref_t<K>>)
  int lookup(K &&key, object &value) const noexcept {
    object k = object::steal(detail::arg_to_pyobject(std::forward<K>(key)));
    if (!k) {
      value = {};
      return -1;
    }
    return lookup(handle{k.ptr()}, value);
  }
  int set(handle key, handle value) noexcept {
    if (!p_ || !key || !value)
      return invalid();
    return PyDict_SetItem(p_, key.ptr(), value.ptr());
  }
  template <class K, class V> int set(K &&key, V &&value) noexcept {
    object k = object::steal(detail::arg_to_pyobject(std::forward<K>(key)));
    if (!k)
      return -1;
    object v = object::steal(detail::arg_to_pyobject(std::forward<V>(value)));
    if (!v)
      return -1;
    return set(handle{k.ptr()}, handle{v.ptr()});
  }

private:
  static int invalid() noexcept {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "invalid dictionary operation");
    return -1;
  }
  explicit dict(PyObject *p) : object(p, adopt_t{}) {}
};

class list : public object {
public:
  explicit list(Py_ssize_t n = 0) : object(PyList_New(n), adopt_t{}) {}
  static bool check_(PyObject *p) noexcept { return p && PyList_Check(p); }
  static list borrow(PyObject *p) { return list{Py_XNewRef(p)}; }
  static list steal(PyObject *p) { return list{p}; }

  Py_ssize_t size() const noexcept { return p_ ? PyList_Size(p_) : -1; }

  int append(handle item) noexcept {
    if (!p_ || !item)
      return invalid();
    return PyList_Append(p_, item.ptr());
  }
  template <class T>
    requires(!std::is_base_of_v<handle, std::remove_cvref_t<T>>)
  int append(T &&v) noexcept {
    PyObject *o = detail::arg_to_pyobject(std::forward<T>(v));
    if (!p_ || !o) {
      Py_XDECREF(o);
      return invalid();
    }
    int r = PyList_Append(p_, o);
    Py_DECREF(o);
    return r;
  }
  object get(Py_ssize_t index) const noexcept {
    if (!p_) {
      invalid();
      return {};
    }
    return object::steal(PyList_GetItemRef(p_, index));
  }
  int set(Py_ssize_t index, handle value) noexcept {
    if (!p_ || !value)
      return invalid();
    return PyList_SetItem(p_, index, Py_NewRef(value.ptr()));
  }
  template <class V>
    requires(!std::is_base_of_v<handle, std::remove_cvref_t<V>>)
  int set(Py_ssize_t index, V &&value) noexcept {
    object v = object::steal(detail::arg_to_pyobject(std::forward<V>(value)));
    return v ? set(index, handle{v.ptr()}) : -1;
  }

private:
  static int invalid() noexcept {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "invalid list operation");
    return -1;
  }
  explicit list(PyObject *p) : object(p, adopt_t{}) {}
};

class tuple : public object {
public:
  tuple() : object(PyTuple_New(0), adopt_t{}) {}
  static bool check_(PyObject *p) noexcept { return p && PyTuple_Check(p); }
  static tuple borrow(PyObject *p) { return tuple{Py_XNewRef(p)}; }
  static tuple steal(PyObject *p) { return tuple{p}; }

  Py_ssize_t size() const noexcept { return p_ ? PyTuple_Size(p_) : -1; }

  template <class... Args> static tuple make(Args &&...args) {
    constexpr Py_ssize_t N = static_cast<Py_ssize_t>(sizeof...(Args));
    PyObject *t = PyTuple_New(N);
    if (!t)
      return tuple{nullptr};
    Py_ssize_t i = 0;
    auto put = [&](auto &&value) {
      PyObject *o =
          detail::arg_to_pyobject(std::forward<decltype(value)>(value));
      if (!o)
        return false;
      PyTuple_SET_ITEM(t, i++, o);
      return true;
    };
    if (!(put(std::forward<Args>(args)) && ...)) {
      Py_DECREF(t);
      return tuple{nullptr};
    }
    return tuple{t};
  }
  object get(Py_ssize_t index) const noexcept {
    if (!p_) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError, "invalid tuple operation");
      return {};
    }
    PyObject *value = PyTuple_GetItem(p_, index);
    return object::borrow(value);
  }

private:
  explicit tuple(PyObject *p) : object(p, adopt_t{}) {}
};

class str : public object {
public:
  str() : object(PyUnicode_FromStringAndSize("", 0), adopt_t{}) {}
  str(const char *s) : object(PyUnicode_FromString(s ? s : ""), adopt_t{}) {}
  str(std::string_view s)
      : object(PyUnicode_FromStringAndSize(s.data(),
                                           static_cast<Py_ssize_t>(s.size())),
               adopt_t{}) {}
  str(const std::string &s)
      : object(PyUnicode_FromStringAndSize(s.data(),
                                           static_cast<Py_ssize_t>(s.size())),
               adopt_t{}) {}

  static bool check_(PyObject *p) noexcept { return p && PyUnicode_Check(p); }
  static str borrow(PyObject *p) { return str{Py_XNewRef(p)}; }
  static str steal(PyObject *p) { return str{p}; }

  std::string_view view() const noexcept {
    if (!p_)
      return {};
    Py_ssize_t n = 0;
    const char *s = PyUnicode_AsUTF8AndSize(p_, &n);
    return s ? std::string_view(s, static_cast<std::size_t>(n))
             : std::string_view{};
  }

private:
  explicit str(PyObject *p) : object(p, adopt_t{}) {}
};

class bytes : public object {
public:
  bytes() : object(PyBytes_FromStringAndSize("", 0), adopt_t{}) {}
  bytes(const char *data, Py_ssize_t n)
      : object(PyBytes_FromStringAndSize(data, n), adopt_t{}) {}
  bytes(std::string_view s)
      : object(PyBytes_FromStringAndSize(s.data(),
                                         static_cast<Py_ssize_t>(s.size())),
               adopt_t{}) {}

  static bool check_(PyObject *p) noexcept { return p && PyBytes_Check(p); }
  static bytes borrow(PyObject *p) { return bytes{Py_XNewRef(p)}; }
  static bytes steal(PyObject *p) { return bytes{p}; }

  Py_ssize_t size() const noexcept { return p_ ? PyBytes_Size(p_) : -1; }
  bool try_view(std::string_view &out) const noexcept {
    out = {};
    if (PyErr_Occurred()) [[unlikely]]
      return false;
    if (!p_) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError, "invalid bytes object");
      return false;
    }
    char *data = nullptr;
    Py_ssize_t n = 0;
    if (PyBytes_AsStringAndSize(p_, &data, &n) < 0)
      return false;
    out = std::string_view(data, static_cast<std::size_t>(n));
    return true;
  }

private:
  explicit bytes(PyObject *p) : object(p, adopt_t{}) {}
};

class int_ : public object {
public:
  int_() : object(PyLong_FromLong(0), adopt_t{}) {}
  template <class T>
    requires std::integral<T>
  int_(T v) : object(make(v), adopt_t{}) {}

  static bool check_(PyObject *p) noexcept { return p && PyLong_Check(p); }
  static int_ borrow(PyObject *p) { return int_{Py_XNewRef(p)}; }
  static int_ steal(PyObject *p) { return int_{p}; }

private:
  template <class T> static PyObject *make(T v) noexcept {
    if constexpr (std::is_signed_v<T>) {
      if constexpr (sizeof(T) <= sizeof(long))
        return PyLong_FromLong(static_cast<long>(v));
      else
        return PyLong_FromLongLong(static_cast<long long>(v));
    } else {
      if constexpr (sizeof(T) <= sizeof(unsigned long))
        return PyLong_FromUnsignedLong(static_cast<unsigned long>(v));
      else
        return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(v));
    }
  }
  explicit int_(PyObject *p) : object(p, adopt_t{}) {}
};

class float_ : public object {
public:
  explicit float_(double v = 0.0) : object(PyFloat_FromDouble(v), adopt_t{}) {}

  static bool check_(PyObject *p) noexcept { return p && PyFloat_Check(p); }
  static float_ borrow(PyObject *p) { return float_{Py_XNewRef(p)}; }
  static float_ steal(PyObject *p) { return float_{p}; }

private:
  explicit float_(PyObject *p) : object(p, adopt_t{}) {}
};

class bool_ : public object {
public:
  explicit bool_(bool v = false) : object(PyBool_FromLong(v), adopt_t{}) {}

  static bool check_(PyObject *p) noexcept { return p && PyBool_Check(p); }
  static bool_ borrow(PyObject *p) { return bool_{Py_XNewRef(p)}; }
  static bool_ steal(PyObject *p) { return bool_{p}; }

private:
  explicit bool_(PyObject *p) : object(p, adopt_t{}) {}
};

} // namespace pyb26
