#pragma once

#include "../lifetime.hpp"
#include "../profile.hpp"

namespace pyb26 {

struct checked_target {};

namespace detail {

template <class... Extras> struct call_policies {
  using scope = profile_scope_t<Extras...>;
  static constexpr std::size_t return_owner_slot =
      return_owner_slot_v<Extras...>;
  static constexpr bool checks_target =
      (std::is_same_v<std::remove_cvref_t<Extras>, checked_target> || ...);
  static constexpr bool error_capable = has_keep_alive_v<Extras...> ||
                                        has_profile_hook_v<Extras...> ||
                                        checks_target;
  static bool precall(PyObject *const *slots, std::size_t count) noexcept {
    return validate_keep_alive<Extras...>(slots, count);
  }
  static bool postcall(PyObject *result, PyObject *const *slots,
                       std::size_t count) noexcept {
    return apply_keep_alive<Extras...>(result, slots, count);
  }
  template <class NativeT>
  static bool postcall_native(PyObject *result, PyObject *const *slots,
                              std::size_t count) noexcept {
    return apply_keep_alive_native<NativeT, Extras...>(result, slots, count);
  }
};

template <class Policy> struct native_postcall_policy : std::false_type {};

template <class... Extras>
struct native_postcall_policy<call_policies<Extras...>>
    : std::bool_constant<([] {
      using Extra = std::remove_cvref_t<Extras>;
      if constexpr (is_keep_alive_v<Extra>)
        return Extra::nurse == 0;
      else
        return false;
    }() || ...)> {};

template <class Policy, class NativeT>
PYB26_ALWAYS_INLINE decltype(auto) apply_policy_postcall(PyObject *&result,
                                                         PyObject *const *slots,
                                                         std::size_t count) {
  if constexpr (native_postcall_policy<Policy>::value &&
                !std::is_void_v<NativeT>)
    return Policy::template postcall_native<NativeT>(result, slots, count);
  else
    return Policy::postcall(result, slots, count);
}

template <class... Extras>
using policy_pack_t = std::conditional_t<
    has_profile_hook_v<Extras...> || has_keep_alive_v<Extras...> ||
        return_owner_slot_v<Extras...> != 0 ||
        (std::is_same_v<std::remove_cvref_t<Extras>, checked_target> || ...),
    call_policies<Extras...>, void>;

} // namespace detail
} // namespace pyb26
