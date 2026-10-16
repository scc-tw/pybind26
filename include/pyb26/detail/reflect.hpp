#pragma once

#include "class.hpp"

#if defined(__cpp_reflection) || defined(__cpp_impl_reflection)

#include <meta>

namespace pyb26 {

namespace detail {

consteval bool reflect_bindable_field(std::meta::info m) {
  return std::meta::is_public(m) && std::meta::has_identifier(m) &&
         !std::meta::is_bit_field(m);
}

consteval bool reflect_bindable_method(std::meta::info m) {
  return std::meta::is_function(m) && !std::meta::is_static_member(m) &&
         !std::meta::is_special_member_function(m) &&
         !std::meta::is_constructor(m) && !std::meta::is_destructor(m) &&
         !std::meta::is_operator_function(m) &&
         !std::meta::is_conversion_function(m) && std::meta::is_public(m) &&
         std::meta::has_identifier(m);
}

} // namespace detail

template <class T>
class_<T> bind(module_ m, const char *name, const char *doc = nullptr) {
  static_assert(std::is_class_v<T>, "pyb26::bind<T>: T must be a class/struct");
  class_<T> c(m, name, doc);

  template for (constexpr auto member :
                define_static_array(std::meta::nonstatic_data_members_of(
                    ^^T, std::meta::access_context::unchecked()))) {
    if constexpr (detail::reflect_bindable_field(member)) {
      c.template def_readwrite<&[:member:]>(
          std::define_static_string(std::meta::identifier_of(member)));
    }
  }

  template for (constexpr auto member :
                define_static_array(std::meta::members_of(
                    ^^T, std::meta::access_context::unchecked()))) {
    if constexpr (detail::reflect_bindable_method(member)) {
      c.template def<&[:member:]>(
          std::define_static_string(std::meta::identifier_of(member)));
    }
  }

  return c;
}

} // namespace pyb26

#else

namespace pyb26 {
template <class T>
class_<T> bind(module_, const char *, const char * = nullptr) {
  static_assert(sizeof(T) == 0,
                "pyb26::bind<T> requires C++26 static reflection. Compile this "
                "translation unit with `-std=c++26 -freflection` (GCC 16+), or "
                "use the explicit class_<T> builder instead.");
}
} // namespace pyb26

#endif
