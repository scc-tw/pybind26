#pragma once

#include "object_types.hpp"

#include <cstddef>
#include <utility>

namespace pyb26 {

class iterator {
  object owner_;

public:
  iterator() = default;
  explicit iterator(handle source) noexcept {
    if (source)
      owner_ = object::steal(PyObject_GetIter(source.ptr()));
    else if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "invalid iterable");
  }
  explicit operator bool() const noexcept { return bool(owner_); }
  int next(object &item) noexcept {
    item = {};
    if (PyErr_Occurred()) [[unlikely]]
      return -1;
    if (!owner_) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError, "invalid iterator");
      return -1;
    }
    PyObject *value = nullptr;
    int rc = PyIter_NextItem(owner_.ptr(), &value);
    item = object::steal(value);
    return rc;
  }
};

inline iterator handle::iterate() const noexcept { return iterator{*this}; }

class tuple_builder {
  object tuple_;
  Py_ssize_t size_ = 0;
  Py_ssize_t filled_ = 0;

public:
  explicit tuple_builder(Py_ssize_t size) noexcept
      : tuple_(object::steal(PyTuple_New(size))), size_(size) {}
  tuple_builder(const tuple_builder &) = delete;
  tuple_builder &operator=(const tuple_builder &) = delete;
  tuple_builder(tuple_builder &&) = default;
  tuple_builder &operator=(tuple_builder &&) = default;
  explicit operator bool() const noexcept { return bool(tuple_); }

  template <class T> int append(T &&value) noexcept {
    if (!tuple_)
      return set_error(PyExc_RuntimeError, "invalid tuple builder");
    if (filled_ >= size_)
      return value_error("tuple builder is full");
    PyObject *item = detail::arg_to_pyobject(std::forward<T>(value));
    if (!item)
      return -1;
    PyTuple_SET_ITEM(tuple_.ptr(), filled_++, item);
    return 0;
  }
  tuple finish() noexcept {
    if (!tuple_) {
      set_error(PyExc_RuntimeError, "invalid tuple builder");
      return tuple::steal(nullptr);
    }
    if (filled_ != size_) {
      value_error("tuple builder is incomplete");
      return tuple::steal(nullptr);
    }
    if (PyErr_Occurred())
      return tuple::steal(nullptr);
    return tuple::steal(tuple_.release());
  }
};

} // namespace pyb26
