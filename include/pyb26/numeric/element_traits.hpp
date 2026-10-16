#pragma once

#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace pyb26 {

enum class scalar_dtype {
  f32,
  f64,
  i8,
  u8,
  i16,
  u16,
  i32,
  u32,
  i64,
  u64,
  c64,
  c128,
  boolean,
  f16
};
enum class scalar_kind {
  unsupported,
  signed_integer,
  unsigned_integer,
  floating,
  complex,
  boolean
};

template <class T> struct half_traits {
  static constexpr bool enabled = false;
};
template <class T> struct custom_complex_traits {
  static constexpr bool enabled = false;
};

template <class T> struct element_traits {
  using value_type = std::remove_cv_t<T>;
  static constexpr bool enabled = half_traits<value_type>::enabled ||
                                  custom_complex_traits<value_type>::enabled;
  static consteval scalar_dtype dtype_value() {
    static_assert(enabled,
                  "unsupported numeric element; specialize element_traits, "
                  "half_traits, or custom_complex_traits");
    if constexpr (half_traits<value_type>::enabled) {
      static_assert(sizeof(value_type) == 2 &&
                        alignof(value_type) == alignof(std::uint16_t) &&
                        std::is_trivially_copyable_v<value_type> &&
                        std::is_standard_layout_v<value_type>,
                    "half storage must be a standard-layout, "
                    "trivially-copyable binary16 value");
      return scalar_dtype::f16;
    } else {
      using C = typename custom_complex_traits<value_type>::component_type;
      static_assert(std::is_same_v<C, float> || std::is_same_v<C, double>);
      static_assert(
          std::numeric_limits<C>::is_iec559 &&
              std::is_standard_layout_v<value_type> &&
              std::is_trivially_copyable_v<value_type> &&
              sizeof(value_type) == 2 * sizeof(C) &&
              alignof(value_type) == alignof(C) &&
              custom_complex_traits<value_type>::real_offset == 0 &&
              custom_complex_traits<value_type>::imag_offset == sizeof(C),
          "custom complex storage must be an aligned real/imaginary IEEE pair");
      return std::is_same_v<C, float> ? scalar_dtype::c64 : scalar_dtype::c128;
    }
  }
  static constexpr scalar_dtype dtype = dtype_value();
  static constexpr scalar_kind kind =
      dtype == scalar_dtype::f16 ? scalar_kind::floating : scalar_kind::complex;
  static constexpr const char *format = [] {
    if constexpr (dtype == scalar_dtype::f16)
      return "e";
    else if constexpr (dtype == scalar_dtype::c64)
      return "=Zf";
    else
      return "=Zd";
  }();
};

#define PYB26_ELEMENT(Type, Dtype, Kind, Format)                               \
  template <> struct element_traits<Type> {                                    \
    static constexpr bool enabled = true;                                      \
    static constexpr scalar_dtype dtype = scalar_dtype::Dtype;                 \
    static constexpr scalar_kind kind = scalar_kind::Kind;                     \
    static constexpr const char *format = Format;                              \
  }
PYB26_ELEMENT(float, f32, floating, "f");
PYB26_ELEMENT(double, f64, floating, "d");
PYB26_ELEMENT(std::int8_t, i8, signed_integer, "b");
PYB26_ELEMENT(std::uint8_t, u8, unsigned_integer, "B");
PYB26_ELEMENT(std::int16_t, i16, signed_integer, "h");
PYB26_ELEMENT(std::uint16_t, u16, unsigned_integer, "H");
PYB26_ELEMENT(std::int32_t, i32, signed_integer, "i");
PYB26_ELEMENT(std::uint32_t, u32, unsigned_integer, "I");
PYB26_ELEMENT(std::int64_t, i64, signed_integer, "q");
PYB26_ELEMENT(std::uint64_t, u64, unsigned_integer, "Q");
PYB26_ELEMENT(std::complex<float>, c64, complex, "Zf");
PYB26_ELEMENT(std::complex<double>, c128, complex, "Zd");
PYB26_ELEMENT(bool, boolean, boolean, "?");
#undef PYB26_ELEMENT

static_assert(sizeof(bool) == 1 && alignof(bool) == 1 &&
                  std::bit_cast<std::uint8_t>(false) == 0 &&
                  std::bit_cast<std::uint8_t>(true) == 1,
              "NumPy boolean storage requires a one-byte bool representation");

template <class T>
  requires(!std::is_same_v<T, std::remove_cv_t<T>>)
struct element_traits<T> : element_traits<std::remove_cv_t<T>> {};

} // namespace pyb26
