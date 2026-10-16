#pragma once

#include "caster.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace pyb26::detail {

template <class C>
inline constexpr bool verified_input_projection = [] {
  if constexpr (requires { typename C::projection_implementation; })
    return std::same_as<C, typename C::projection_implementation>;
  return false;
}();

template <class A, class C = caster_for<A>>
inline constexpr bool input_caster_available = [] {
  if constexpr (!requires(C &caster, PyObject *object) {
                  caster.load(object);
                }) {
    return false;
  } else if constexpr (verified_input_projection<C> &&
                       requires { C::template projection_inert<A>; }) {
    if constexpr (C::template projection_inert<A>)
      return requires(C &caster) { caster.template project<A>(); };
    else
      return requires(C &caster) { caster.template as<A>(); };
  } else {
    return requires(C &caster) { caster.template as<A>(); };
  }
}();

template <class A, class V>
inline constexpr bool native_argument_binding = [] {
  using Target = std::remove_cvref_t<A>;
  using Source = std::remove_cvref_t<V>;
  if constexpr (!std::is_convertible_v<V, A>) {
    return false;
  } else if constexpr (std::same_as<Target, Source>) {
    return !std::is_class_v<Target> || std::is_reference_v<A> ||
           std::same_as<V, Target &&> || std::same_as<V, const Target &> ||
           std::is_trivially_constructible_v<Target, V>;
  } else if constexpr (std::is_class_v<Target> && std::is_class_v<Source>) {
    return std::is_base_of_v<Target, Source> &&
           (std::is_reference_v<A> ||
            std::is_trivially_constructible_v<Target, V>);
  } else if constexpr (std::is_pointer_v<Target>) {
    return std::is_pointer_v<Source> || std::is_null_pointer_v<Source>;
  } else {
    return (std::is_arithmetic_v<Target> || std::is_enum_v<Target>) &&
           (std::is_arithmetic_v<Source> || std::is_enum_v<Source>);
  }
}();

template <class A, class C = caster_for<A>> struct input_argument_plan {
  static constexpr bool direct_projection = [] {
    if constexpr (verified_input_projection<C> &&
                  requires { C::template projection_inert<A>; })
      return C::template projection_inert<A>;
    return false;
  }();
  static constexpr argument_source source = [] {
    if constexpr (verified_input_projection<C>)
      return C::storage_source;
    return argument_source::opaque;
  }();
  static constexpr bool construction_inert = [] {
    if constexpr (verified_input_projection<C> &&
                  requires { C::input_construction_inert; })
      return C::input_construction_inert;
    return false;
  }();
  static constexpr bool load_runs_python = caster_contract<A>.runs_python;
  static constexpr bool extraction_inert = direct_projection;
  static decltype(auto) project(C &caster) {
    if constexpr (direct_projection)
      return caster.template project<A>();
    else
      return caster.template as<A>();
  }

  using projected_type = decltype(project(std::declval<C &>()));
  using stored_type =
      std::conditional_t<std::is_lvalue_reference_v<projected_type> ||
                             (direct_projection &&
                              std::is_reference_v<projected_type>),
                         projected_type, std::remove_cvref_t<projected_type>>;
  using forwarded_type = stored_type &&;
  static constexpr bool parameter_inert =
      native_argument_binding<A, forwarded_type> &&
      (std::is_reference_v<A> ||
       (native_cleanup_inert_v<A> &&
        ((std::is_lvalue_reference_v<forwarded_type> ||
          std::is_const_v<std::remove_reference_t<forwarded_type>>)
             ? native_copy_inert_v<A>
             : native_move_inert_v<A>)));
  static constexpr bool cleanup_inert =
      input_caster_cleanup_inert_v<A> && native_cleanup_inert_v<stored_type>;
  static constexpr bool load_inert = construction_inert && !load_runs_python &&
                                     input_caster_cleanup_inert_v<A>;
};

template <class A, class C> decltype(auto) project_input(C &caster) {
  return input_argument_plan<A, C>::project(caster);
}

} // namespace pyb26::detail
