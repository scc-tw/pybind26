#pragma once

#include "detail/caster.hpp"

#include <cstddef>

namespace pyb26 {

class args {
  PyObject *const *values_ = nullptr;
  Py_ssize_t size_ = 0;

public:
  args() noexcept = default;
  args(PyObject *const *values, Py_ssize_t size) noexcept
      : values_(values), size_(size) {}
  Py_ssize_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  PyObject *operator[](Py_ssize_t index) const noexcept {
    return values_[index];
  }
  PyObject *const *begin() const noexcept { return values_; }
  PyObject *const *end() const noexcept {
    return size_ ? values_ + size_ : values_;
  }
};

template <> struct type_caster<args> {
  static constexpr caster_effects effects{false, true, true};
  args value;
  bool load(PyObject *object) noexcept {
    if (!PyTuple_Check(object))
      return false;
    value = args(reinterpret_cast<PyTupleObject *>(object)->ob_item,
                 PyTuple_GET_SIZE(object));
    return true;
  }
  bool load_range(PyObject *const *values, Py_ssize_t count) noexcept {
    value = args(values, count);
    return true;
  }
  template <class A> decltype(auto) as() noexcept {
    return static_cast<A>(value);
  }
};

namespace detail {
template <class T>
inline constexpr bool is_args_v = std::is_same_v<std::remove_cvref_t<T>, args>;
}
} // namespace pyb26
