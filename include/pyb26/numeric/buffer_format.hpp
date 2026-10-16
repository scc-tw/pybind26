#pragma once

#include "element_traits.hpp"

namespace pyb26::detail {

using numeric_kind = scalar_kind;

struct buffer_dtype {
  numeric_kind kind = numeric_kind::unsupported;
  std::size_t size = 0;
  bool native_endian = true;
};

constexpr buffer_dtype parse_buffer_dtype(const char *format) noexcept {
  if (!format)
    return {};
  bool native_size = true;
  bool native_endian = true;
  switch (*format) {
  case '@':
    ++format;
    break;
  case '=':
    native_size = false;
    ++format;
    break;
  case '<':
    native_size = false;
    native_endian = std::endian::native == std::endian::little;
    ++format;
    break;
  case '>':
  case '!':
    native_size = false;
    native_endian = std::endian::native == std::endian::big;
    ++format;
    break;
  }
  if (!*format)
    return {};
  using enum numeric_kind;
  if (format[1] != '\0') {
    if (*format == 'Z' && format[2] == '\0') {
      if (format[1] == 'f')
        return {complex, 8, native_endian};
      if (format[1] == 'd')
        return {complex, 16, native_endian};
    }
    return {};
  }
  switch (*format) {
  case 'b':
    return {signed_integer, 1, true};
  case 'B':
    return {unsigned_integer, 1, true};
  case 'h':
    return {signed_integer, native_size ? sizeof(short) : 2, native_endian};
  case 'H':
    return {unsigned_integer, native_size ? sizeof(unsigned short) : 2,
            native_endian};
  case 'i':
    return {signed_integer, native_size ? sizeof(int) : 4, native_endian};
  case 'I':
    return {unsigned_integer, native_size ? sizeof(unsigned) : 4,
            native_endian};
  case 'l':
    return {signed_integer, native_size ? sizeof(long) : 4, native_endian};
  case 'L':
    return {unsigned_integer, native_size ? sizeof(unsigned long) : 4,
            native_endian};
  case 'q':
    return {signed_integer, native_size ? sizeof(long long) : 8, native_endian};
  case 'Q':
    return {unsigned_integer, native_size ? sizeof(unsigned long long) : 8,
            native_endian};
  case 'n':
    return native_size
               ? buffer_dtype{signed_integer, sizeof(std::ptrdiff_t), true}
               : buffer_dtype{};
  case 'N':
    return native_size
               ? buffer_dtype{unsigned_integer, sizeof(std::size_t), true}
               : buffer_dtype{};
  case '?':
    return {boolean, 1, true};
  case 'e':
    return {floating, 2, native_endian};
  case 'f':
    return {floating, 4, native_endian};
  case 'd':
    return {floating, 8, native_endian};
  default:
    return {};
  }
}

} // namespace pyb26::detail
