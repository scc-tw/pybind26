#pragma once

#include "array.hpp"
#include "detail/dlpack.hpp"
#include "detail/dlpack_export.hpp"

namespace pyb26::dlpack {

enum class negotiation { versioned, legacy, legacy_writable };

template <class T, std::size_t Rank, class Layout = layout::c_contiguous,
          negotiation Protocol = negotiation::versioned>
class view : public array_view<T, Rank, Layout> {
  static_assert(Protocol != negotiation::legacy || std::is_const_v<T>,
                "mutable legacy DLPack input requires explicit legacy_writable "
                "negotiation");

public:
  using array_view<T, Rank, Layout>::array_view;
  constexpr view() noexcept = default;
  constexpr view(array_view<T, Rank, Layout> value) noexcept
      : array_view<T, Rank, Layout>(value) {}
};

} // namespace pyb26::dlpack

namespace pyb26 {

template <class T, std::size_t Rank, class Layout, dlpack::negotiation Protocol>
struct type_caster<dlpack::view<T, Rank, Layout, Protocol>> {
  static constexpr caster_effects effects{true, true, true};
  using view_type = dlpack::view<T, Rank, Layout, Protocol>;
  using element_type = std::remove_const_t<T>;
  static_assert(Rank <= PyBUF_MAX_NDIM,
                "DLPack array rank exceeds the supported limit");
  static_assert(std::is_same_v<element_type, float> ||
                    std::is_same_v<element_type, double> ||
                    std::is_same_v<element_type, std::int8_t> ||
                    std::is_same_v<element_type, std::uint8_t> ||
                    std::is_same_v<element_type, std::int16_t> ||
                    std::is_same_v<element_type, std::uint16_t> ||
                    std::is_same_v<element_type, std::int32_t> ||
                    std::is_same_v<element_type, std::uint32_t> ||
                    std::is_same_v<element_type, std::int64_t> ||
                    std::is_same_v<element_type, std::uint64_t> ||
                    detail::is_complex_v<element_type>,
                "unsupported DLPack element type");
  detail::dlpack_lease lease_;
  view_type view_;

  bool load(PyObject *source) noexcept {
    view_ = {};
    if (!lease_.acquire(source, Protocol != dlpack::negotiation::versioned))
      return false;
    return validate();
  }

  template <class A> decltype(auto) as() noexcept {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(view_);
    else
      return view_;
  }

  template <class R> static PyObject *cast(R &&) noexcept {
    static_assert(!std::is_same_v<R, R>,
                  "borrowed view cannot be returned; use an owning result or "
                  "explicit copy");
    return nullptr;
  }

private:
  bool validate() noexcept {
    const auto &tensor = lease_.tensor();
    if (tensor.device.device_type != kDLCPU || tensor.device.device_id != 0)
      return detail::dlpack_error(
          "DLPack device query and exported tensor disagree");
    if (tensor.ndim != static_cast<std::int32_t>(Rank))
      return detail::dlpack_error("DLPack rank does not match the binding");
    constexpr auto code = detail::is_complex_v<T>       ? kDLComplex
                          : std::is_floating_point_v<T> ? kDLFloat
                          : std::is_signed_v<T>         ? kDLInt
                                                        : kDLUInt;
    if (tensor.dtype.code != code || tensor.dtype.bits != sizeof(T) * 8 ||
        tensor.dtype.lanes != 1)
      return detail::dlpack_error("DLPack dtype does not match the binding");
    if (!std::is_const_v<T> && lease_.readonly())
      return detail::dlpack_error(
          "mutable DLPack input cannot use readonly storage");
    if constexpr (Rank > 0) {
      if (!tensor.shape || (!tensor.strides && lease_.explicit_strides()))
        return detail::dlpack_error(
            "DLPack tensor is missing shape or required strides");
    }
    std::array<std::size_t, Rank> shape{};
    bool empty = false;
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      if (tensor.shape[axis] < 0)
        return detail::dlpack_error("negative DLPack extent");
      shape[axis] = static_cast<std::size_t>(tensor.shape[axis]);
      empty = empty || shape[axis] == 0;
    }
    std::size_t bytes = empty ? 0 : sizeof(T);
    if (!empty) {
      for (auto extent : shape) {
        if (extent > static_cast<std::size_t>(PY_SSIZE_T_MAX) / bytes)
          return detail::buffer_overflow();
        bytes *= extent;
      }
    }
    std::array<std::ptrdiff_t, Rank> strides{};
    constexpr auto maximum = std::numeric_limits<std::ptrdiff_t>::max();
    constexpr auto minimum = std::numeric_limits<std::ptrdiff_t>::min();
    if (tensor.strides) {
      for (std::size_t axis = 0; axis < Rank; ++axis) {
        const auto stride = tensor.strides[axis];
        if (stride > maximum / static_cast<std::ptrdiff_t>(sizeof(T)) ||
            stride < minimum / static_cast<std::ptrdiff_t>(sizeof(T)))
          return detail::buffer_overflow();
        strides[axis] = static_cast<std::ptrdiff_t>(stride) *
                        static_cast<std::ptrdiff_t>(sizeof(T));
      }
    } else if (!empty) {
      std::size_t stride = sizeof(T);
      for (std::size_t axis = Rank; axis-- > 0;) {
        strides[axis] = static_cast<std::ptrdiff_t>(stride);
        stride *= shape[axis];
      }
    }
    const auto base = reinterpret_cast<std::uintptr_t>(tensor.data);
    if (tensor.byte_offset > std::numeric_limits<std::uintptr_t>::max() - base)
      return detail::buffer_overflow();
    if (bytes && !tensor.data)
      return detail::dlpack_error("nonempty DLPack tensor has a null pointer");
    void *data = reinterpret_cast<void *>(
        base + static_cast<std::uintptr_t>(tensor.byte_offset));
    array_view<T, Rank, Layout> value;
    if (!detail::validate_array_layout<T, Rank, Layout>(data, shape, bytes,
                                                        strides, value)) {
      if (!PyErr_Occurred())
        return detail::dlpack_error(
            "DLPack alignment or layout does not match the binding");
      return false;
    }
    view_ = view_type(value);
    return true;
  }
};

} // namespace pyb26
