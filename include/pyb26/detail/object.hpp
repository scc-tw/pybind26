#pragma once

#include "caster.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pyb26 {

class object;
class attr_proxy;
class item_proxy;
class iterator;

class handle {
protected:
  PyObject *p_ = nullptr;

public:
  handle() = default;
  handle(PyObject *p) noexcept : p_(p) {}

  PyObject *ptr() const noexcept { return p_; }
  explicit operator bool() const noexcept { return p_ != nullptr; }

  bool is_none() const noexcept { return p_ == Py_None; }
  bool is(handle other) const noexcept { return p_ == other.p_; }
  bool is_valid() const noexcept { return p_ != nullptr; }

  object inc_ref_object() const;

  int truthy() const noexcept {
    return p_ ? PyObject_IsTrue(p_)
              : set_error(PyExc_TypeError, "invalid Python object");
  }
  Py_ssize_t length() const noexcept {
    return p_ ? PyObject_Length(p_)
              : set_error(PyExc_TypeError, "invalid Python object");
  }

  inline attr_proxy attr(const char *name) const;
  object get_attr(const char *name) const noexcept;
  object get_item(handle key) const noexcept;
  int set_item(handle key, handle value) const noexcept;
  iterator iterate() const noexcept;
  int set_attr(const char *name, handle value) const noexcept {
    return (p_ && value)
               ? PyObject_SetAttrString(p_, name, value.ptr())
               : set_error(PyExc_TypeError, "invalid attribute operation");
  }
  int has_attr_checked(const char *name) const noexcept {
    return p_ ? PyObject_HasAttrStringWithError(p_, name)
              : set_error(PyExc_TypeError, "invalid Python object");
  }
  bool has_attr(const char *name) const noexcept {
    return p_ && PyObject_HasAttrString(p_, name) == 1;
  }

  template <class K> inline auto operator[](K &&key) const;

  template <class... Args> inline object operator()(Args &&...args) const;
};

class object : public handle {
public:
  object() = default;

  object(const object &o) noexcept : handle(Py_XNewRef(o.p_)) {}
  object(object &&o) noexcept : handle(o.p_) { o.p_ = nullptr; }

  object &operator=(const object &o) noexcept {
    if (this != &o) {
      PyObject *old = p_;
      p_ = Py_XNewRef(o.p_);
      Py_XDECREF(old);
    }
    return *this;
  }
  object &operator=(object &&o) noexcept {
    if (this != &o) {
      PyObject *old = p_;
      p_ = o.p_;
      o.p_ = nullptr;
      Py_XDECREF(old);
    }
    return *this;
  }

  ~object() { Py_XDECREF(p_); }

  static object borrow(PyObject *p) noexcept {
    object o;
    o.p_ = Py_XNewRef(p);
    return o;
  }
  static object steal(PyObject *p) noexcept {
    object o;
    o.p_ = p;
    return o;
  }

  PyObject *release() noexcept {
    PyObject *t = p_;
    p_ = nullptr;
    return t;
  }

protected:
  struct adopt_t {};
  object(PyObject *p, adopt_t) noexcept : handle(p) {}
};

inline object handle::inc_ref_object() const { return object::borrow(p_); }

class none : public object {
public:
  none() : object(Py_NewRef(Py_None), adopt_t{}) {}
};

} // namespace pyb26

#include "object_accessors.hpp"
#include "object_convert.hpp"
#include "object_iteration.hpp"
#include "object_types.hpp"
