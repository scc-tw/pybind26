#pragma once

#if defined(__cpp_reflection) || defined(__cpp_impl_reflection)

#include <meta>
#include <string_view>

namespace pyb26 {

namespace detail {

template <class T> consteval bool type_has_identifier() {
  return std::meta::has_identifier(^^T);
}

} // namespace detail

template <class T> consteval std::string_view type_name() {
  return std::string_view(
      std::define_static_string(std::meta::display_string_of(^^T)));
}

template <class T> consteval std::string_view type_name_short() {
  if constexpr (detail::type_has_identifier<T>())
    return std::string_view(
        std::define_static_string(std::meta::identifier_of(^^T)));
  else
    return type_name<T>();
}

} // namespace pyb26

#else

#include <string_view>

namespace pyb26 {
template <class T> consteval std::string_view type_name() {
  static_assert(sizeof(T) == 0,
                "pyb26::type_name<T>() requires C++26 static reflection "
                "(-std=c++26 -freflection, GCC 16+).");
  return {};
}
template <class T> consteval std::string_view type_name_short() {
  static_assert(sizeof(T) == 0,
                "pyb26::type_name_short<T>() requires C++26 static reflection "
                "(-std=c++26 -freflection, GCC 16+).");
  return {};
}
} // namespace pyb26

#endif
