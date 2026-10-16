#pragma once

#include "detail/common.hpp"

#include <type_traits>

namespace pyb26 {

template <class Hook> struct profile_hook {};

namespace detail {

template <class T> struct profile_hook_traits {
  static constexpr bool enabled = false;
  using type = void;
};
template <class Hook> struct profile_hook_traits<profile_hook<Hook>> {
  static constexpr bool enabled = true;
  using type = Hook;
};
template <class T>
inline constexpr bool is_profile_hook_v =
    profile_hook_traits<std::remove_cvref_t<T>>::enabled;

template <class... Extras> struct find_profile_hook {
  using type = void;
};
template <class E, class... Rest> struct find_profile_hook<E, Rest...> {
  using traits = profile_hook_traits<std::remove_cvref_t<E>>;
  using type = std::conditional_t<traits::enabled, typename traits::type,
                                  typename find_profile_hook<Rest...>::type>;
};

template <class Hook> class profile_scope {
  const char *label_;
  bool started_;

public:
  explicit profile_scope(const char *label) noexcept
      : label_(label), started_(Hook::start(label)) {
    static_assert(noexcept(Hook::start(label)),
                  "profile start must be noexcept");
    static_assert(noexcept(Hook::end(label)), "profile end must be noexcept");
  }
  profile_scope(const profile_scope &) = delete;
  profile_scope &operator=(const profile_scope &) = delete;
  ~profile_scope() {
    if (started_)
      Hook::end(label_);
  }
};
template <> class profile_scope<void> {
public:
  explicit profile_scope(const char * = nullptr) noexcept {}
};

template <class... Extras>
using profile_scope_t =
    profile_scope<typename find_profile_hook<Extras...>::type>;

template <class... Extras>
inline constexpr bool has_profile_hook_v = (is_profile_hook_v<Extras> || ...);

template <class... Extras>
void fill_profile_label(callable_spec &spec, PyObject *scope) noexcept {
  static_assert((0 + ... + (is_profile_hook_v<Extras> ? 1 : 0)) <= 1,
                "a callable can have only one profile hook");
  if constexpr (has_profile_hook_v<Extras...>)
    spec.profile_label = make_profile_label(scope, spec.name);
}

} // namespace detail
} // namespace pyb26
