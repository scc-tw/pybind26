#pragma once

#include "caster.hpp"
#include "guard_arguments.hpp"

#include <concepts>
#include <optional>
#include <type_traits>

namespace pyb26 {

class object;
class handle;

namespace detail {

template <class T>
inline constexpr bool interpreter_retained_v =
    requires { requires T::interpreter_retained; };

template <class T>
struct callback_owns_value
    : std::bool_constant<!std::is_pointer_v<T> && !caster_yields_view_v<T> &&
                         !caster_requires_lease_v<T> &&
                         (!std::is_base_of_v<handle, T> ||
                          interpreter_retained_v<T>)> {};

template <class T>
inline constexpr bool callback_owns_value_v =
    !std::is_reference_v<T> && callback_owns_value<std::remove_cv_t<T>>::value;

template <class T>
struct callback_owns_value<std::optional<T>>
    : std::bool_constant<callback_owns_value_v<T>> {};
template <class T>
struct callback_owns_value<std::reference_wrapper<T>> : std::false_type {};
template <class T, std::size_t N>
struct callback_owns_value<std::array<T, N>>
    : std::bool_constant<callback_owns_value_v<T>> {};
template <class A, class B>
struct callback_owns_value<std::pair<A, B>>
    : std::bool_constant<callback_owns_value_v<A> && callback_owns_value_v<B>> {
};
template <class... T>
struct callback_owns_value<std::tuple<T...>>
    : std::bool_constant<(callback_owns_value_v<T> && ...)> {};
template <class... T>
struct callback_owns_value<std::variant<T...>>
    : std::bool_constant<(callback_owns_value_v<T> && ...)> {};

#define PYB26_CALLBACK_SEQUENCE(Container)                                     \
  template <class T, class Alloc>                                              \
  struct callback_owns_value<std::Container<T, Alloc>>                         \
      : std::bool_constant<callback_owns_value_v<T>> {};
PYB26_CALLBACK_SEQUENCE(vector)
PYB26_CALLBACK_SEQUENCE(list)
PYB26_CALLBACK_SEQUENCE(deque)
PYB26_CALLBACK_SEQUENCE(forward_list)
#undef PYB26_CALLBACK_SEQUENCE

#define PYB26_CALLBACK_SET(Container)                                          \
  template <class T, class Compare, class Alloc>                               \
  struct callback_owns_value<std::Container<T, Compare, Alloc>>                \
      : std::bool_constant<callback_owns_value_v<T>> {};
PYB26_CALLBACK_SET(set)
PYB26_CALLBACK_SET(multiset)
#undef PYB26_CALLBACK_SET

#define PYB26_CALLBACK_UNORDERED_SET(Container)                                \
  template <class T, class Hash, class Equal, class Alloc>                     \
  struct callback_owns_value<std::Container<T, Hash, Equal, Alloc>>            \
      : std::bool_constant<callback_owns_value_v<T>> {};
PYB26_CALLBACK_UNORDERED_SET(unordered_set)
PYB26_CALLBACK_UNORDERED_SET(unordered_multiset)
#undef PYB26_CALLBACK_UNORDERED_SET

#define PYB26_CALLBACK_MAP(Container)                                          \
  template <class K, class V, class Compare, class Alloc>                      \
  struct callback_owns_value<std::Container<K, V, Compare, Alloc>>             \
      : std::bool_constant<callback_owns_value_v<K> &&                         \
                           callback_owns_value_v<V>> {};
PYB26_CALLBACK_MAP(map)
PYB26_CALLBACK_MAP(multimap)
#undef PYB26_CALLBACK_MAP

#define PYB26_CALLBACK_UNORDERED_MAP(Container)                                \
  template <class K, class V, class Hash, class Equal, class Alloc>            \
  struct callback_owns_value<std::Container<K, V, Hash, Equal, Alloc>>         \
      : std::bool_constant<callback_owns_value_v<K> &&                         \
                           callback_owns_value_v<V>> {};
PYB26_CALLBACK_UNORDERED_MAP(unordered_map)
PYB26_CALLBACK_UNORDERED_MAP(unordered_multimap)
#undef PYB26_CALLBACK_UNORDERED_MAP

template <class R>
inline constexpr bool python_failure_sentinel_v =
    std::is_void_v<R> || (interpreter_retained_v<R> && requires {
      { R::steal(nullptr) } -> std::same_as<R>;
    });

template <class T>
inline constexpr bool python_failure_sentinel_v<std::optional<T>> =
    callback_owns_value_v<T>;

template <class R> R python_failure_value() noexcept {
  if constexpr (std::is_void_v<R>)
    return;
  else if constexpr (interpreter_retained_v<R>)
    return R::steal(nullptr);
  else
    return std::nullopt;
}

template <class R> R python_failure_return() noexcept {
  static_assert(
      python_failure_sentinel_v<R>,
      "pyb26: Python callbacks and overrides require void, retained_object, "
      "or std::optional<T> to report failure without C++ exceptions");
  if (!PyErr_Occurred())
    PyErr_SetString(PyExc_SystemError,
                    "Python callback failed without an exception");
  return python_failure_value<R>();
}

} // namespace detail
} // namespace pyb26
