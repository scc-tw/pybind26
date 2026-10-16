#pragma once

#include <array>
#include <functional>
#include <list>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace pyb26::detail {

template <class T> struct native_cleanup_inert;
template <class T>
inline constexpr bool native_cleanup_inert_v = native_cleanup_inert<T>::value;

template <class T>
struct native_cleanup_shape
    : std::bool_constant<std::is_trivially_destructible_v<T>> {};
template <> struct native_cleanup_shape<std::string> : std::true_type {};
template <class T>
struct native_cleanup_shape<std::vector<T>>
    : std::bool_constant<native_cleanup_inert_v<T>> {};
template <class T>
struct native_cleanup_shape<std::list<T>>
    : std::bool_constant<native_cleanup_inert_v<T>> {};
template <class K, class V>
struct native_cleanup_shape<std::map<K, V>>
    : std::bool_constant<
          native_cleanup_inert_v<K> && native_cleanup_inert_v<V> &&
          native_cleanup_inert_v<typename std::map<K, V>::key_compare>> {};
template <class K, class V>
struct native_cleanup_shape<std::unordered_map<K, V>>
    : std::bool_constant<
          native_cleanup_inert_v<K> && native_cleanup_inert_v<V> &&
          native_cleanup_inert_v<typename std::unordered_map<K, V>::hasher> &&
          native_cleanup_inert_v<
              typename std::unordered_map<K, V>::key_equal>> {};
template <class T, std::size_t N>
struct native_cleanup_shape<std::array<T, N>>
    : std::bool_constant<native_cleanup_inert_v<T>> {};
template <class A, class B>
struct native_cleanup_shape<std::pair<A, B>>
    : std::bool_constant<native_cleanup_inert_v<A> &&
                         native_cleanup_inert_v<B>> {};
template <class... T>
struct native_cleanup_shape<std::tuple<T...>>
    : std::bool_constant<(native_cleanup_inert_v<T> && ...)> {};
template <class T>
struct native_cleanup_shape<std::optional<T>>
    : std::bool_constant<native_cleanup_inert_v<T>> {};
template <class... T>
struct native_cleanup_shape<std::variant<T...>>
    : std::bool_constant<(native_cleanup_inert_v<T> && ...)> {};
template <class T>
struct native_cleanup_inert
    : std::bool_constant<std::is_reference_v<T> ||
                         native_cleanup_shape<std::remove_cv_t<T>>::value> {};

template <class T> struct native_copy_inert;
template <class T>
inline constexpr bool native_copy_inert_v =
    native_copy_inert<std::remove_cvref_t<T>>::value;
template <class T>
struct native_copy_inert
    : std::bool_constant<std::is_trivially_copy_constructible_v<T> &&
                         std::is_trivially_destructible_v<T>> {};
template <> struct native_copy_inert<std::string> : std::true_type {};
template <> struct native_copy_inert<std::string_view> : std::true_type {};
template <class T>
struct native_copy_inert<std::vector<T>>
    : std::bool_constant<native_copy_inert_v<T>> {};
template <class T>
struct native_copy_inert<std::list<T>>
    : std::bool_constant<native_copy_inert_v<T>> {};
template <class T, std::size_t N>
struct native_copy_inert<std::array<T, N>>
    : std::bool_constant<native_copy_inert_v<T>> {};
template <class A, class B>
struct native_copy_inert<std::pair<A, B>>
    : std::bool_constant<native_copy_inert_v<A> && native_copy_inert_v<B>> {};
template <class... T>
struct native_copy_inert<std::tuple<T...>>
    : std::bool_constant<(native_copy_inert_v<T> && ...)> {};
template <class T>
struct native_copy_inert<std::optional<T>>
    : std::bool_constant<native_copy_inert_v<T>> {};
template <class... T>
struct native_copy_inert<std::variant<T...>>
    : std::bool_constant<(native_copy_inert_v<T> && ...)> {};

template <class T> struct native_move_inert;
template <class T>
inline constexpr bool native_move_inert_v =
    native_move_inert<std::remove_cvref_t<T>>::value;
template <class T>
struct native_move_inert
    : std::bool_constant<std::is_trivially_move_constructible_v<T> &&
                         std::is_trivially_destructible_v<T>> {};
template <> struct native_move_inert<std::string> : std::true_type {};
template <> struct native_move_inert<std::string_view> : std::true_type {};
template <class T> struct native_move_inert<std::vector<T>> : std::true_type {};
template <class T> struct native_move_inert<std::list<T>> : std::true_type {};
template <class K, class V>
struct native_move_inert<std::map<K, V>>
    : std::bool_constant<
          native_move_inert_v<typename std::map<K, V>::key_compare>> {};
template <class K, class V>
struct native_move_inert<std::unordered_map<K, V>>
    : std::bool_constant<
          native_move_inert_v<typename std::unordered_map<K, V>::hasher> &&
          native_move_inert_v<typename std::unordered_map<K, V>::key_equal>> {};
template <class T, std::size_t N>
struct native_move_inert<std::array<T, N>>
    : std::bool_constant<native_move_inert_v<T>> {};
template <class A, class B>
struct native_move_inert<std::pair<A, B>>
    : std::bool_constant<native_move_inert_v<A> && native_move_inert_v<B>> {};
template <class... T>
struct native_move_inert<std::tuple<T...>>
    : std::bool_constant<(native_move_inert_v<T> && ...)> {};
template <class T>
struct native_move_inert<std::optional<T>>
    : std::bool_constant<native_move_inert_v<T>> {};
template <class... T>
struct native_move_inert<std::variant<T...>>
    : std::bool_constant<(native_move_inert_v<T> && ...)> {};

template <class T> struct native_default_inert;
template <class T>
inline constexpr bool native_default_inert_v =
    native_default_inert<std::remove_cvref_t<T>>::value;
template <class T>
struct native_default_inert
    : std::bool_constant<std::is_trivially_default_constructible_v<T>> {};
template <> struct native_default_inert<std::string> : std::true_type {};
template <> struct native_default_inert<std::string_view> : std::true_type {};
template <class T>
struct native_default_inert<std::vector<T>> : std::true_type {};
template <class T>
struct native_default_inert<std::list<T>> : std::true_type {};
template <class K, class V>
struct native_default_inert<std::map<K, V>>
    : std::bool_constant<
          native_default_inert_v<typename std::map<K, V>::key_compare>> {};
template <class K, class V>
struct native_default_inert<std::unordered_map<K, V>>
    : std::bool_constant<
          native_default_inert_v<typename std::unordered_map<K, V>::hasher> &&
          native_default_inert_v<
              typename std::unordered_map<K, V>::key_equal>> {};
template <class T, std::size_t N>
struct native_default_inert<std::array<T, N>>
    : std::bool_constant<native_default_inert_v<T>> {};
template <class A, class B>
struct native_default_inert<std::pair<A, B>>
    : std::bool_constant<native_default_inert_v<A> &&
                         native_default_inert_v<B>> {};
template <class... T>
struct native_default_inert<std::tuple<T...>>
    : std::bool_constant<(native_default_inert_v<T> && ...)> {};
template <class T>
struct native_default_inert<std::optional<T>> : std::true_type {};
template <class First, class... Rest>
struct native_default_inert<std::variant<First, Rest...>>
    : std::bool_constant<native_default_inert_v<First>> {};

template <class T, class From>
struct native_assignment_inert
    : std::bool_constant<std::is_trivially_assignable_v<T &, From>> {};
template <class From>
struct native_assignment_inert<std::string, From>
    : std::bool_constant<
          std::is_same_v<std::remove_cvref_t<From>, std::string> &&
          std::is_assignable_v<std::string &, From>> {};
template <class T, class From>
inline constexpr bool native_assignment_inert_v =
    native_assignment_inert<T, From>::value;

template <class T>
inline constexpr bool native_key_operations_inert_v =
    std::is_arithmetic_v<T> || std::is_same_v<T, std::string>;

template <class Operation, class T>
struct native_key_operation_inert : std::false_type {};
template <class T>
struct native_key_operation_inert<std::less<T>, T>
    : std::bool_constant<native_key_operations_inert_v<T>> {};
template <class T>
struct native_key_operation_inert<std::equal_to<T>, T>
    : std::bool_constant<native_key_operations_inert_v<T>> {};
template <class T>
struct native_key_operation_inert<std::hash<T>, T>
    : std::bool_constant<native_key_operations_inert_v<T>> {};

} // namespace pyb26::detail
