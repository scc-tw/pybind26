#pragma once
#include <pyb26/detail/caster.hpp>
#include <pyb26/errors.hpp>

namespace pyb26 {

template <std::integral T>
  requires(!std::same_as<T, bool> && std::same_as<T, std::remove_cv_t<T>> &&
           sizeof(T) <= sizeof(long long))
struct compatible_integer {
  T value;
};

namespace detail {

template <std::integral T>
  requires(!std::same_as<T, bool> && std::same_as<T, std::remove_cv_t<T>> &&
           sizeof(T) <= sizeof(long long))
struct py_hint<compatible_integer<T>> : py_hint<T> {};

PYB26_COLD inline bool consumer_integer_out_of_range() noexcept {
  (void)set_error(PyExc_TypeError,
                  "integer out of range for the C++ parameter type");
  return false;
}

template <std::integral T>
PYB26_NOINLINE bool load_consumer_integer_slow(PyObject *o, T &out) noexcept {
  if constexpr (std::is_signed_v<T>) {
    int overflow = 0;
    const long long value = PyLong_AsLongLongAndOverflow(o, &overflow);
    if (overflow)
      return consumer_integer_out_of_range();
    if (value == -1 && PyErr_Occurred())
      return false;
    out = static_cast<T>(value);
    if constexpr (sizeof(T) < sizeof(long long)) {
      if (value != static_cast<long long>(out))
        return consumer_integer_out_of_range();
    }
  } else {
    int sign = 0;
    if (PyLong_GetSign(o, &sign) < 0)
      return false;
    if (sign < 0)
      return consumer_integer_out_of_range();
    const Py_ssize_t size = PyLong_AsNativeBytes(
        o, &out, sizeof(T),
        Py_ASNATIVEBYTES_NATIVE_ENDIAN | Py_ASNATIVEBYTES_UNSIGNED_BUFFER);
    if (size < 0)
      return false;
    if (size > static_cast<Py_ssize_t>(sizeof(T)))
      return consumer_integer_out_of_range();
  }
  return true;
}

template <std::integral T>
PYB26_HOT bool load_consumer_integer(PyObject *o, T &out) noexcept {
#if !defined(Py_LIMITED_API)
  if (PyLong_CheckExact(o)) [[likely]] {
    auto *value = reinterpret_cast<PyLongObject *>(o);
    if (cpython::long_is_compact(value)) [[likely]] {
      const Py_ssize_t native = cpython::compact_long_value(value);
      out = static_cast<T>(native);
      if constexpr (std::is_unsigned_v<T> || sizeof(T) != sizeof(Py_ssize_t)) {
        if ((std::is_unsigned_v<T> && native < 0) ||
            (sizeof(T) != sizeof(Py_ssize_t) &&
             native != static_cast<Py_ssize_t>(out))) [[unlikely]]
          return consumer_integer_out_of_range();
      }
      return true;
    }
  }
#endif
  return load_consumer_integer_slow(o, out);
}

template <std::integral T>
PYB26_NOINLINE bool load_consumer_index(PyObject *o, T &out) noexcept {
  if (PyErr_Occurred() || !PyIndex_Check(o))
    return false;
  PyObject *index = PyNumber_Index(o);
  if (!index)
    return false;
  const bool success = !PyErr_Occurred() && load_consumer_integer(index, out);
  Py_DECREF(index);
  return success;
}

} // namespace detail

template <std::integral T>
  requires(!std::same_as<T, bool> && std::same_as<T, std::remove_cv_t<T>> &&
           sizeof(T) <= sizeof(long long))
struct type_caster<compatible_integer<T>> {
  static constexpr caster_effects effects{true, false, false};
  T value;

  PYB26_HOT bool load(PyObject *o) noexcept {
    if (PyLong_Check(o)) [[likely]]
      return detail::load_consumer_integer(o, value);
    return detail::load_consumer_index(o, value);
  }

  template <class A> compatible_integer<T> as() const noexcept {
    return {value};
  }

  static PyObject *cast(const compatible_integer<T> &result) noexcept {
    return caster_for<T>::cast(result.value);
  }
};

} // namespace pyb26
