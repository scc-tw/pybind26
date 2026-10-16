#pragma once

#include "../numpy.hpp"
#include <cstring>

namespace pyb26::numpy::detail {
bool half_scalar_bits(PyObject *source, std::uint16_t &bits) noexcept;
}

namespace pyb26 {

template <class T, bool Convert = true> struct half_scalar {
  T value;
};

template <class T> struct custom_complex_scalar {
  T value;
};

template <class T> struct structured_complex_scalar {
  T value;
};

namespace detail {

template <class T>
T custom_complex_from_values(double real, double imag) noexcept {
  using traits = custom_complex_traits<T>;
  using C = typename traits::component_type;
  static_assert(element_traits<T>::dtype == scalar_dtype::c64 ||
                element_traits<T>::dtype == scalar_dtype::c128);
  T result{};
  C re = static_cast<C>(real), im = static_cast<C>(imag);
  std::memcpy(reinterpret_cast<std::byte *>(&result) + traits::real_offset, &re,
              sizeof(C));
  std::memcpy(reinterpret_cast<std::byte *>(&result) + traits::imag_offset, &im,
              sizeof(C));
  return result;
}
template <class T> PyObject *custom_complex_to_python(const T &value) noexcept {
  using traits = custom_complex_traits<T>;
  using C = typename traits::component_type;
  C real, imag;
  std::memcpy(&real,
              reinterpret_cast<const std::byte *>(&value) + traits::real_offset,
              sizeof(C));
  std::memcpy(&imag,
              reinterpret_cast<const std::byte *>(&value) + traits::imag_offset,
              sizeof(C));
  return PyComplex_FromDoubles(real, imag);
}
} // namespace detail

template <class T, bool Convert> struct type_caster<half_scalar<T, Convert>> {
  static constexpr caster_effects effects{true, false, false};
  static_assert(half_traits<T>::enabled &&
                element_traits<T>::dtype == scalar_dtype::f16);
  half_scalar<T, Convert> value_{};
  bool load(PyObject *source) noexcept {
    std::uint16_t bits;
    if (numpy::detail::half_scalar_bits(source, bits)) {
      value_.value = half_traits<T>::from_bits(bits);
      return true;
    }
    if (PyErr_Occurred() || PyComplex_Check(source))
      return false;
    double value;
    if (PyFloat_Check(source))
      value = PyFloat_AS_DOUBLE(source);
    else if constexpr (!Convert)
      return false;
    else {
      value = PyFloat_AsDouble(source);
      if (value == -1.0 && PyErr_Occurred())
        return false;
    }
    value_.value = half_traits<T>::from_double(value);
    return !PyErr_Occurred();
  }
  bool load_exact(PyObject *source) noexcept {
    type_caster<half_scalar<T, false>> caster;
    if (!caster.load(source))
      return false;
    value_.value = caster.value_.value;
    return true;
  }
  template <class A> half_scalar<T, Convert> as() const noexcept {
    return value_;
  }
  static PyObject *cast(const half_scalar<T, Convert> &source) noexcept {
    return PyFloat_FromDouble(half_traits<T>::to_double(source.value));
  }
};

template <class T> struct type_caster<custom_complex_scalar<T>> {
  static constexpr caster_effects effects{true, false, false};
  type_caster<std::complex<typename custom_complex_traits<T>::component_type>>
      inner_;
  template <class A> custom_complex_scalar<T> as() noexcept {
    auto value = inner_.template as<
        std::complex<typename custom_complex_traits<T>::component_type>>();
    return {detail::custom_complex_from_values<T>(value.real(), value.imag())};
  }
  bool load(PyObject *source) noexcept { return inner_.load(source); }
  static PyObject *cast(const custom_complex_scalar<T> &source) noexcept {
    return detail::custom_complex_to_python(source.value);
  }
};

template <class T> struct type_caster<structured_complex_scalar<T>> {
  static constexpr caster_effects effects{true, false, false};
  structured_complex_scalar<T> value_{};
  bool load(PyObject *source) noexcept {
    using traits = custom_complex_traits<T>;
    object real_key = object::steal(PyUnicode_FromString(traits::real_name));
    object imag_key = object::steal(PyUnicode_FromString(traits::imag_name));
    if (!real_key || !imag_key)
      return false;
    object real = object::steal(PyObject_GetItem(source, real_key.ptr()));
    object imag = object::steal(PyObject_GetItem(source, imag_key.ptr()));
    if (!real || !imag)
      return false;
    const double re = PyFloat_AsDouble(real.ptr()),
                 im = PyFloat_AsDouble(imag.ptr());
    if (PyErr_Occurred())
      return false;
    value_.value = detail::custom_complex_from_values<T>(re, im);
    return true;
  }
  template <class A> structured_complex_scalar<T> as() const noexcept {
    return value_;
  }
  static PyObject *cast(const structured_complex_scalar<T> &source) noexcept {
    return detail::custom_complex_to_python(source.value);
  }
};

} // namespace pyb26
