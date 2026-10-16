#pragma once

#include "common.hpp"

#include <tuple>
#include <utility>

namespace pyb26 {

namespace detail {
inline thread_local PyThreadState *released_python_state = nullptr;
}

class gil_scoped_acquire {
  PyGILState_STATE state_ = PyGILState_LOCKED;
  bool acquired_ = false;
  PyThreadState *restored_ = nullptr;

public:
  gil_scoped_acquire() noexcept {
    if (!::pyb26::detail::cpython::attached_thread_state()) [[unlikely]] {
      if ((restored_ = detail::released_python_state)) {
        PyEval_RestoreThread(restored_);
      } else {
        state_ = PyGILState_Ensure();
        acquired_ = true;
      }
    }
  }
  ~gil_scoped_acquire() noexcept {
    if (restored_) [[unlikely]] {
      PyEval_SaveThread();
    } else if (acquired_) [[unlikely]]
      PyGILState_Release(state_);
  }
  gil_scoped_acquire(const gil_scoped_acquire &) = delete;
  gil_scoped_acquire &operator=(const gil_scoped_acquire &) = delete;
};

class gil_scoped_release {
  PyThreadState *state_;
  PyThreadState *previous_;

public:
  gil_scoped_release() noexcept
      : state_(PyEval_SaveThread()),
        previous_(std::exchange(detail::released_python_state, state_)) {}
  ~gil_scoped_release() noexcept {
    detail::released_python_state = previous_;
    PyEval_RestoreThread(state_);
  }
  gil_scoped_release(const gil_scoped_release &) = delete;
  gil_scoped_release &operator=(const gil_scoped_release &) = delete;
};

template <class... Guards> struct call_guard {
  using guard_tuple = std::tuple<Guards...>;
};

namespace detail {

template <class T> struct is_call_guard : std::false_type {};
template <class... G>
struct is_call_guard<call_guard<G...>> : std::true_type {};
template <class T>
inline constexpr bool is_call_guard_v =
    is_call_guard<std::remove_cvref_t<T>>::value;

template <class... Extras>
inline constexpr bool has_call_guard_v = (is_call_guard_v<Extras> || ...);

template <class... Extras> struct find_call_guard {
  using type = void;
};
template <class E, class... Rest> struct find_call_guard<E, Rest...> {
  using type = std::conditional_t<is_call_guard_v<E>, std::remove_cvref_t<E>,
                                  typename find_call_guard<Rest...>::type>;
};
template <class... Extras>
using find_call_guard_t = typename find_call_guard<Extras...>::type;

template <class CG> inline constexpr bool guard_releases_python_v = false;
template <class... G>
inline constexpr bool guard_releases_python_v<call_guard<G...>> =
    (std::is_base_of_v<gil_scoped_release, G> || ...);

template <class CG> struct guard_scope {
  typename CG::guard_tuple guards_{};
};

} // namespace detail
} // namespace pyb26
