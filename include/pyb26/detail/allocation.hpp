#pragma once

#include <Python.h>
#include <new>
#include <utility>

namespace pyb26::detail {

enum class allocation_site {
  none,
  callable_capture,
  argument_metadata,
  type_metadata,
  constructor_entries,
  callable_callbacks,
  callback_error_channel,
  lifetime_owners
};

inline thread_local allocation_site failed_allocation_site =
    allocation_site::none;

class allocation_failure_scope {
  allocation_site previous_;

public:
  explicit allocation_failure_scope(allocation_site site) noexcept
      : previous_(std::exchange(failed_allocation_site, site)) {}
  ~allocation_failure_scope() { failed_allocation_site = previous_; }
  allocation_failure_scope(const allocation_failure_scope &) = delete;
};

template <allocation_site Site> bool allocation_admitted() noexcept {
  if (failed_allocation_site != Site) [[likely]]
    return true;
  failed_allocation_site = allocation_site::none;
  PyErr_NoMemory();
  return false;
}

template <allocation_site Site, class T, class... A>
T *binding_new(A &&...args) noexcept {
  if (!allocation_admitted<Site>())
    return nullptr;
  auto *value = new (std::nothrow) T(std::forward<A>(args)...);
  if (!value)
    PyErr_NoMemory();
  return value;
}

template <allocation_site Site, class T>
T *binding_new_array(std::size_t count) noexcept {
  if (!allocation_admitted<Site>())
    return nullptr;
  auto *value = new (std::nothrow) T[count]();
  if (!value)
    PyErr_NoMemory();
  return value;
}

} // namespace pyb26::detail
