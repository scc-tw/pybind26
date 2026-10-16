#pragma once

#include "array.hpp"

#include <optional>
#include <span>

namespace pyb26 {

template <class T, std::size_t Extent>
struct type_caster<std::span<T, Extent>> {
  static constexpr caster_effects effects{true, true, true};
  using view_type = std::span<T, Extent>;
  using source_type = array_view<T, 1>;
  type_caster<source_type> source_;
  std::optional<view_type> view_;

  bool load(PyObject *source) noexcept {
    view_.reset();
    if (!source_.load(source))
      return false;
    const auto value = source_.template as<source_type>();
    if constexpr (Extent != std::dynamic_extent) {
      if (value.size() != Extent)
        return false;
    }
    view_.emplace(value.data(), value.size());
    return true;
  }

  template <class A> decltype(auto) as() noexcept {
    return static_cast<A>(*view_);
  }

  template <class R> static PyObject *cast(R &&) noexcept {
    static_assert(!std::is_same_v<R, R>,
                  "borrowed view cannot be returned; use an owning result or "
                  "explicit copy");
    return nullptr;
  }
};

} // namespace pyb26
