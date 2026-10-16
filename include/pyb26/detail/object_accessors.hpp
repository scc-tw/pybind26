#pragma once

#include "caster.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pyb26 {

class handle;
class object;

namespace detail {

template <class T> PyObject *arg_to_pyobject(T &&v) noexcept {
  using D = std::remove_cvref_t<T>;
  if constexpr (std::is_base_of_v<handle, D>) {
    PyObject *p = static_cast<const handle &>(v).ptr();
    if (!p && !PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "invalid Python object");
    return Py_XNewRef(p);
  } else if constexpr (std::is_same_v<D, PyObject *>) {
    if (!v && !PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "invalid Python object");
    return Py_XNewRef(v);
  } else {
    using C = std::decay_t<T>;
    if constexpr (std::is_lvalue_reference_v<T> &&
                  std::is_constructible_v<C, T &&> &&
                  std::is_copy_constructible_v<C> && requires {
                    type_caster<C>::cast(static_cast<const C &>(v));
                  }) {
      return type_caster<C>::cast(static_cast<const C &>(v));
    } else if constexpr (!std::is_lvalue_reference_v<T> && requires {
                           type_caster<C>::cast(std::forward<T>(v));
                         }) {
      return type_caster<C>::cast(std::forward<T>(v));
    } else {
      return type_caster<C>::cast(static_cast<C>(std::forward<T>(v)));
    }
  }
}

} // namespace detail

class attr_proxy {
  PyObject *obj_;
  const char *name_;

public:
  attr_proxy(PyObject *obj, const char *name) noexcept
      : obj_(obj), name_(name) {}

  operator object() const { return get(); }
  object get() const noexcept { return handle{obj_}.get_attr(name_); }
  int set(handle value) const noexcept {
    return handle{obj_}.set_attr(name_, value);
  }

  template <class... Args> object operator()(Args &&...args) const {
    object fn = get();
    return fn(std::forward<Args>(args)...);
  }

  template <class T>
    requires(!std::is_base_of_v<handle, std::remove_cvref_t<T>>)
  int set(T &&v) const noexcept {
    PyObject *o = detail::arg_to_pyobject(std::forward<T>(v));
    int rc = o ? set(handle{o}) : -1;
    Py_XDECREF(o);
    return rc;
  }
};

inline attr_proxy handle::attr(const char *name) const {
  return attr_proxy{p_, name};
}

inline object handle::get_attr(const char *name) const noexcept {
  if (!p_) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "invalid Python object");
    return {};
  }
  return object::steal(PyObject_GetAttrString(p_, name));
}
inline object handle::get_item(handle key) const noexcept {
  if (!p_ || !key) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "invalid item operation");
    return {};
  }
  return object::steal(PyObject_GetItem(p_, key.ptr()));
}
inline int handle::set_item(handle key, handle value) const noexcept {
  if (!p_ || !key || !value) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "invalid item operation");
    return -1;
  }
  return PyObject_SetItem(p_, key.ptr(), value.ptr());
}

class item_proxy {
  PyObject *obj_;
  object key_;

public:
  item_proxy(PyObject *obj, object key) noexcept
      : obj_(obj), key_(std::move(key)) {}

  operator object() const { return get(); }
  object get() const noexcept { return handle{obj_}.get_item(key_); }
  int set(handle value) const noexcept {
    return handle{obj_}.set_item(key_, value);
  }

  template <class T>
    requires(!std::is_base_of_v<handle, std::remove_cvref_t<T>>)
  int set(T &&v) const noexcept {
    PyObject *o = detail::arg_to_pyobject(std::forward<T>(v));
    int rc = o ? set(handle{o}) : -1;
    Py_XDECREF(o);
    return rc;
  }
};

template <class K> inline auto handle::operator[](K &&key) const {
  return item_proxy{
      p_, object::steal(detail::arg_to_pyobject(std::forward<K>(key)))};
}

template <class... Args>
inline object handle::operator()(Args &&...args) const {
  if (!p_) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "invalid Python callable");
    return {};
  }
  constexpr std::size_t N = sizeof...(Args);
  if constexpr (N == 0) {
    return object::steal(PyObject_CallNoArgs(p_));
  } else {
    PyObject *argv[N] = {};
    std::size_t i = 0;
    auto put = [&](auto &&value) {
      PyObject *o =
          detail::arg_to_pyobject(std::forward<decltype(value)>(value));
      if (!o)
        return false;
      argv[i++] = o;
      return true;
    };
    bool ok = (put(std::forward<Args>(args)) && ...);
    PyObject *res = nullptr;
    if (ok) {
      res = PyObject_Vectorcall(p_, argv, N, nullptr);
    }
    for (std::size_t i = 0; i < N; ++i)
      Py_XDECREF(argv[i]);
    return object::steal(res);
  }
}

} // namespace pyb26
