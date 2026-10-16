#pragma once

#include "argument_plan.hpp"

#include <array>
#include <deque>
#include <forward_list>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace pyb26 {
class object;
class handle;

namespace detail {
template <class Stored> struct prepared_argument {
  Stored value;
  template <class A> decltype(auto) as() noexcept {
    return static_cast<Stored &&>(value);
  }
};
template <class A, class C> auto prepare_input_argument(C &caster) {
  using Plan = input_argument_plan<A, C>;
  return prepared_argument<typename Plan::stored_type>{
      project_input<A>(caster)};
}

template <std::size_t I, class Stored> struct prepared_argument_slot {
  Stored value;
  template <class A> decltype(auto) as() noexcept {
    return static_cast<Stored &&>(value);
  }
};

template <class Sequence, class... Stored> struct prepared_arguments;

template <std::size_t... I, class... Stored>
struct prepared_arguments<std::index_sequence<I...>, Stored...>
    : prepared_argument_slot<I, Stored>... {
  template <std::size_t N> decltype(auto) get() noexcept {
    using Slot =
        prepared_argument_slot<N,
                               std::tuple_element_t<N, std::tuple<Stored...>>>;
    return static_cast<Slot &>(*this);
  }
  template <std::size_t N> decltype(auto) get() const noexcept {
    using Slot =
        prepared_argument_slot<N,
                               std::tuple_element_t<N, std::tuple<Stored...>>>;
    return static_cast<const Slot &>(*this);
  }
};

template <std::size_t I, class Cast>
decltype(auto) get_argument(Cast &cast) noexcept {
  if constexpr (requires { cast.template get<I>(); })
    return cast.template get<I>();
  else
    return std::get<I>(cast);
}

template <class... A, class Cast, std::size_t... I>
auto make_prepared_arguments(Cast &cast, std::index_sequence<I...>) {
  static_assert(sizeof...(A) == sizeof...(I));
  using Pack =
      prepared_arguments<std::index_sequence<I...>,
                         typename input_argument_plan<
                             A, std::remove_cvref_t<decltype(get_argument<I>(
                                    cast))>>::stored_type...>;
  return Pack{{project_input<A>(get_argument<I>(cast))}...};
}

template <class T>
struct guard_argument_python_owner
    : std::bool_constant<caster_contract<T>.owns_python ||
                         std::is_base_of_v<object, T>> {};
template <class T>
inline constexpr bool guard_argument_python_owner_v =
    guard_argument_python_owner<std::remove_cvref_t<T>>::value;

template <class T, class Alloc>
struct guard_argument_python_owner<std::vector<T, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class T, class Alloc>
struct guard_argument_python_owner<std::list<T, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class T, class Alloc>
struct guard_argument_python_owner<std::forward_list<T, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class T, class Alloc>
struct guard_argument_python_owner<std::deque<T, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class T, std::size_t N>
struct guard_argument_python_owner<std::array<T, N>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class T>
struct guard_argument_python_owner<std::optional<T>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class... T>
struct guard_argument_python_owner<std::tuple<T...>>
    : std::bool_constant<(guard_argument_python_owner_v<T> || ...)> {};
template <class... T>
struct guard_argument_python_owner<std::variant<T...>>
    : std::bool_constant<(guard_argument_python_owner_v<T> || ...)> {};
template <class A, class B>
struct guard_argument_python_owner<std::pair<A, B>>
    : std::bool_constant<guard_argument_python_owner_v<A> ||
                         guard_argument_python_owner_v<B>> {};
template <class T, class Compare, class Alloc>
struct guard_argument_python_owner<std::set<T, Compare, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class T, class Compare, class Alloc>
struct guard_argument_python_owner<std::multiset<T, Compare, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class T, class Hash, class Equal, class Alloc>
struct guard_argument_python_owner<std::unordered_set<T, Hash, Equal, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class T, class Hash, class Equal, class Alloc>
struct guard_argument_python_owner<
    std::unordered_multiset<T, Hash, Equal, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class K, class V, class Compare, class Alloc>
struct guard_argument_python_owner<std::map<K, V, Compare, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<K> ||
                         guard_argument_python_owner_v<V>> {};
template <class K, class V, class Compare, class Alloc>
struct guard_argument_python_owner<std::multimap<K, V, Compare, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<K> ||
                         guard_argument_python_owner_v<V>> {};
template <class K, class V, class Hash, class Equal, class Alloc>
struct guard_argument_python_owner<std::unordered_map<K, V, Hash, Equal, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<K> ||
                         guard_argument_python_owner_v<V>> {};
template <class K, class V, class Hash, class Equal, class Alloc>
struct guard_argument_python_owner<
    std::unordered_multimap<K, V, Hash, Equal, Alloc>>
    : std::bool_constant<guard_argument_python_owner_v<K> ||
                         guard_argument_python_owner_v<V>> {};
template <class T>
struct guard_argument_python_owner<T *>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class Sig>
struct guard_argument_python_owner<std::function<Sig>> : std::true_type {};
template <class T>
struct guard_argument_python_owner<std::shared_ptr<T>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class T>
struct guard_argument_python_owner<std::weak_ptr<T>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
template <class T, class D>
struct guard_argument_python_owner<std::unique_ptr<T, D>>
    : std::bool_constant<guard_argument_python_owner_v<T>> {};
} // namespace detail
} // namespace pyb26
