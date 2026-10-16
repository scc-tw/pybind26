#pragma once

#include "detail/caster.hpp"

#include <cmath>
#include <complex>
#include <limits>

namespace pyb26::detail {

static_assert(
    sizeof(std::complex<float>) == 8 && sizeof(std::complex<double>) == 16 &&
        std::is_trivially_copyable_v<std::complex<float>> &&
        std::is_trivially_copyable_v<std::complex<double>> &&
        std::numeric_limits<float>::is_iec559 &&
        std::numeric_limits<double>::is_iec559,
    "complex buffers require IEEE complex64/complex128 representations");

template <class T>
inline constexpr bool is_complex_v =
    std::is_same_v<std::remove_cv_t<T>, std::complex<float>> ||
    std::is_same_v<std::remove_cv_t<T>, std::complex<double>>;

template <class T> struct py_hint<std::complex<T>> {
  static void append(std::string &value) { value += "complex"; }
};

struct complex_load_result {
  Py_complex value;
  bool success;
};

PYB26_NOINLINE inline complex_load_result
complex_from_protocol(PyObject *source) noexcept {
  PyObject *name = PyUnicode_FromString("__complex__");
  if (!name)
    return {{}, false};
  PyObject *method = _PyType_LookupRef(Py_TYPE(source), name);
  Py_DECREF(name);
  if (!method) {
    if (PyErr_Occurred())
      return {{}, false};
    auto *number = Py_TYPE(source)->tp_as_number;
    if (!number || (!number->nb_float && !number->nb_index))
      return {{}, false};
  }
  Py_XDECREF(method);
  const auto value = PyComplex_AsCComplex(source);
  return {value, value.real != -1.0 || !PyErr_Occurred()};
}

PYB26_COLD inline bool complex_component_overflow() noexcept {
  PyErr_SetString(PyExc_OverflowError,
                  "complex component is out of range for float32");
  return false;
}

} // namespace pyb26::detail

namespace pyb26 {

template <class T> struct type_caster<std::complex<T>> {
  static constexpr caster_effects effects{true, false, false};
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "only complex<float> and complex<double> are supported");
  static_assert(
      sizeof(std::complex<T>) == 2 * sizeof(T) &&
          std::is_trivially_copyable_v<std::complex<T>> &&
          std::numeric_limits<T>::is_iec559,
      "complex elements require an IEEE floating-point pair representation");
  std::complex<T> value_{};

  PYB26_HOT bool load(PyObject *source) noexcept {
    Py_complex value;
    if (PyComplex_Check(source)) [[likely]] {
      value = reinterpret_cast<PyComplexObject *>(source)->cval;
    } else if (PyFloat_CheckExact(source)) {
      value = {PyFloat_AS_DOUBLE(source), 0};
    } else if (PyLong_CheckExact(source)) {
      const double real = PyLong_AsDouble(source);
      if (real == -1.0 && PyErr_Occurred())
        return false;
      value = {real, 0};
    } else {
      const auto converted = detail::complex_from_protocol(source);
      if (!converted.success)
        return false;
      value = converted.value;
    }
    if constexpr (std::is_same_v<T, float>) {
      constexpr double limit = std::numeric_limits<float>::max();
      if ((std::isfinite(value.real) && std::abs(value.real) > limit) ||
          (std::isfinite(value.imag) && std::abs(value.imag) > limit))
        return detail::complex_component_overflow();
    }
    value_ = {static_cast<T>(value.real), static_cast<T>(value.imag)};
    return true;
  }

  template <class A> decltype(auto) as() noexcept {
    return static_cast<A>(value_);
  }
  static PyObject *cast(const std::complex<T> &value) noexcept {
    return PyComplex_FromDoubles(static_cast<double>(value.real()),
                                 static_cast<double>(value.imag()));
  }
};

} // namespace pyb26
