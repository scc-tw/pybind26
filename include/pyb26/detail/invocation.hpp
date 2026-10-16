#pragma once

#include "gil.hpp"
#include "guard_arguments.hpp"

#include <memory>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pyb26::detail {

template <class Error, bool Enabled> struct invocation_checkpoint {
  Error &error;
  ~invocation_checkpoint() { error.capture(); }
};
template <class Error> struct invocation_checkpoint<Error, false> {
  explicit invocation_checkpoint(Error &) noexcept {}
};

template <class A>
using invocation_caster_t = typename decltype([] {
  using C = caster_for<A>;
  if constexpr (requires {
                  typename C::native_lifetime_caster;
                  typename C::template invocation_caster<A>;
                }) {
    if constexpr (std::is_same_v<C, typename C::native_lifetime_caster>)
      return std::type_identity<typename C::template invocation_caster<A>>{};
    else
      return std::type_identity<C>{};
  } else {
    return std::type_identity<C>{};
  }
}())::type;

template <class... A> struct invocation_arguments {
  using types = std::tuple<A...>;
  static constexpr bool cleanup =
      ((!input_argument_plan<A, invocation_caster_t<A>>::cleanup_inert) || ...);

  template <std::size_t I = 0, class Cast, class Error, class Invoke,
            class... V>
  PYB26_ALWAYS_INLINE static int prepare(Cast &cast, Error &error,
                                         Invoke &invoke, V &&...values) {
    if constexpr (I == sizeof...(A)) {
      return invoke(std::forward<V>(values)...);
    } else {
      using Arg = std::tuple_element_t<I, types>;
      using Plan = input_argument_plan<Arg, invocation_caster_t<Arg>>;
      typename Plan::stored_type value =
          project_input<Arg>(get_argument<I>(cast));
      invocation_checkpoint<Error, !native_cleanup_inert_v<decltype(value)>>
          checkpoint{error};
      if constexpr (!Plan::extraction_inert)
        if (error.capture()) [[unlikely]]
          return -1;
      return prepare<I + 1>(cast, error, invoke, std::forward<V>(values)...,
                            static_cast<typename Plan::stored_type &&>(value));
    }
  }

  template <std::size_t I = 0, class Load, class Error, class Invoke,
            class... Cast>
  PYB26_ALWAYS_INLINE static int load(Load &loader, Error &error,
                                      Invoke &invoke, Cast &...loaded) {
    if constexpr (I == sizeof...(A)) {
      auto cast = std::tie(loaded...);
      return invoke(cast);
    } else {
      using Arg = std::tuple_element_t<I, types>;
      using Plan = input_argument_plan<Arg, invocation_caster_t<Arg>>;
      invocation_caster_t<Arg> cast;
      invocation_checkpoint<Error, !input_caster_cleanup_inert_v<Arg>>
          checkpoint{error};
      if constexpr (!Plan::load_inert)
        if (error.capture()) [[unlikely]]
          return -1;
      if (!loader.template operator()<I>(cast))
        return error.capture() ? -1 : 1;
      if constexpr (!Plan::load_inert)
        if (error.capture()) [[unlikely]]
          return -1;
      return load<I + 1>(loader, error, invoke, loaded..., cast);
    }
  }
};

template <class Guards> struct invocation_guard_sequence;
template <class... G> struct invocation_guard_sequence<call_guard<G...>> {
  using guards = std::tuple<G...>;
  template <std::size_t I = 0, class Error, class Invoke>
  PYB26_ALWAYS_INLINE static void run(Error &error, Invoke &invoke) {
    if constexpr (I == sizeof...(G)) {
      invoke();
    } else {
      using Guard = std::tuple_element_t<I, guards>;
      Guard guard;
      if constexpr (std::is_base_of_v<gil_scoped_release, Guard>) {
        static_assert(I + 1 == sizeof...(G),
                      "pyb26: a release guard must be the final guard");
        invoke();
      } else {
        invocation_checkpoint<Error, true> checkpoint{error};
        if (!error.capture()) [[likely]]
          run<I + 1>(error, invoke);
      }
    }
  }
};

template <class Guard, class Error, class Invoke>
PYB26_ALWAYS_INLINE void invocation_guarded(Error &error, Invoke &&invoke) {
  if constexpr (std::is_void_v<Guard>)
    invoke();
  else
    invocation_guard_sequence<Guard>::run(error, invoke);
}

template <class R, bool Reference = std::is_reference_v<R>>
class invocation_return_storage {
  union {
    unsigned char empty_;
    R value_;
  };
  bool constructed_ = false;

public:
  invocation_return_storage() noexcept : empty_(0) {}
  invocation_return_storage(const invocation_return_storage &) = delete;
  ~invocation_return_storage() {
    if (constructed_)
      std::destroy_at(std::addressof(value_));
  }
  template <class F> void construct(F &&invoke) {
    ::new (static_cast<void *>(
        const_cast<std::remove_cv_t<R> *>(std::addressof(value_)))) R(invoke());
    constructed_ = true;
  }
  bool has_value() const noexcept { return constructed_; }
  R &&get() noexcept { return std::move(value_); }
};

template <class R> class invocation_return_storage<R, true> {
  std::remove_reference_t<R> *value_ = nullptr;

public:
  template <class F> void construct(F &&invoke) {
    auto &&value = invoke();
    value_ = std::addressof(value);
  }
  bool has_value() const noexcept { return value_ != nullptr; }
  R get() noexcept { return static_cast<R>(*value_); }
};

} // namespace pyb26::detail
