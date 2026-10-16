#pragma once

#include "lifetime.hpp"
#include <optional>

namespace pyb26::detail {

template <bool Required> class argument_owner_admission {
public:
  constexpr bool acquire() const noexcept { return true; }
};

template <> class argument_owner_admission<true> {
  std::optional<owner_admission> owned_;

public:
  bool acquire() noexcept {
    if (owned_)
      return static_cast<bool>(*owned_);
    auto *current = cpython::attached_thread_state();
    if (!current)
      return false;
    for (auto *entry = owner_admissions; entry; entry = entry->previous) {
      if (entry->owner->interpreter != current->interp)
        continue;
      if (entry->owner->state.load(std::memory_order_acquire) &
          python_interpreter_owner::closing) {
        PyErr_SetString(PyExc_RuntimeError,
                        "array interpreter is closing or closed");
        return false;
      }
      owned_.emplace(*entry, owner_admission::borrowed{});
      return static_cast<bool>(*owned_);
    }
    owned_.emplace(capture_python_owner_context());
    if (*owned_)
      return true;
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_RuntimeError,
                      "array interpreter is closing or closed");
    return false;
  }
};

} // namespace pyb26::detail
