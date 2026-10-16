#pragma once

#include "pyb26.hpp"

#include <new>
#include <utility>

namespace pyb26 {
namespace detail {

template <class It, class Sentinel = It> struct cpp_iter_state {
  It cur;
  Sentinel end;
  object parent;
};

template <class It, class Sentinel>
PyObject *cppiter_next(PyObject *self) noexcept {
  using State = cpp_iter_state<It, Sentinel>;
  State *st = cast_to<State>(self);
  if (!st) [[unlikely]] {
    PyErr_SetString(PyExc_TypeError, "pyb26: invalid iterator instance");
    return nullptr;
  }
  if (st->cur == st->end)
    return nullptr;
  using E = std::remove_cvref_t<decltype(*st->cur)>;
  PyObject *r = type_caster<E>::cast(*st->cur);
  if (r) [[likely]]
    ++st->cur;
  return r;
}

} // namespace detail

template <class It, class Sentinel = It>
class_<detail::cpp_iter_state<It, Sentinel>> bind_iterator(module_ m,
                                                           const char *name) {
  using State = detail::cpp_iter_state<It, Sentinel>;
  static PyType_Slot slots[] = {
      {Py_tp_iter, reinterpret_cast<void *>(&PyObject_SelfIter)},
      {Py_tp_iternext,
       reinterpret_cast<void *>(&detail::cppiter_next<It, Sentinel>)},
  };
  return class_<State>(m, name, nullptr, slots,
                       static_cast<int>(sizeof(slots) / sizeof(slots[0])));
}

template <class It, class Sentinel = It>
PyObject *make_iterator(It first, Sentinel last,
                        PyObject *parent = nullptr) noexcept {
  using State = detail::cpp_iter_state<It, Sentinel>;
  return detail::make_instance<State>([&](void *mem) {
    ::new (mem)
        State{std::move(first), std::move(last), object::borrow(parent)};
  });
}

} // namespace pyb26
