#pragma once

#include "object.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace pyb26 {

template <> struct type_caster<handle> {
  static constexpr caster_effects effects{false, true, false};
  PyObject *p = nullptr;
  PYB26_HOT bool load(PyObject *o) noexcept {
    p = o;
    return true;
  }
  template <class A> handle as() const noexcept { return handle{p}; }
  static PyObject *cast(handle h) noexcept {
    return h.ptr() ? Py_NewRef(h.ptr()) : Py_NewRef(Py_None);
  }
};

template <> struct type_caster<object> {
  static constexpr caster_effects effects{false, false, false, true};
  object value;
  PYB26_HOT bool load(PyObject *o) noexcept {
    value = object::borrow(o);
    return true;
  }
  template <class A> decltype(auto) as() {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(value);
    else
      return object(std::move(value));
  }
  static PyObject *cast(object o) noexcept { return o.release(); }
};

template <class W>
  requires(std::derived_from<W, object> && !std::same_as<W, object>)
struct type_caster<W> {
  static constexpr caster_effects effects{false, false, false, true};
  union {
    W w_;
  };
  bool has_ = false;
  type_caster() noexcept {}
  ~type_caster() {
    if (has_)
      w_.~W();
  }
  type_caster(const type_caster &) = delete;
  type_caster &operator=(const type_caster &) = delete;

  PYB26_HOT bool load(PyObject *o) noexcept {
    if constexpr (requires(PyObject *q) {
                    { W::check_(q) } -> std::convertible_to<bool>;
                  }) {
      if (!W::check_(o)) [[unlikely]]
        return false;
    }
    ::new (&w_) W(W::borrow(o));
    has_ = true;
    return true;
  }
  template <class A> decltype(auto) as() {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(w_);
    else
      return W(std::move(w_));
  }
  static PyObject *cast(const W &w) noexcept {
    return w.ptr() ? Py_NewRef(w.ptr()) : nullptr;
  }
  static PyObject *cast(W &&w) noexcept {
    return static_cast<object &>(w).release();
  }
};

} // namespace pyb26
