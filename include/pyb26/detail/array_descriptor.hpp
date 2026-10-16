#pragma once

#include "../numeric/dynamic_array.hpp"
#include "buffer.hpp"

namespace pyb26::detail {

inline bool
validate_array_descriptor(const array_descriptor &descriptor) noexcept {
  if (!descriptor || descriptor.rank() > PyBUF_MAX_NDIM ||
      descriptor.byte_strides().size() != descriptor.rank() ||
      !descriptor.format || !descriptor.itemsize ||
      descriptor.bytes > static_cast<std::size_t>(PY_SSIZE_T_MAX))
    return invalid_buffer("invalid buffer metadata");
  static constexpr std::size_t sizes[]{4, 8, 1, 1, 2,  2, 4,
                                       4, 8, 8, 8, 16, 1, 2};
  static constexpr std::size_t alignments[]{
      alignof(float),         alignof(double),        alignof(std::int8_t),
      alignof(std::uint8_t),  alignof(std::int16_t),  alignof(std::uint16_t),
      alignof(std::int32_t),  alignof(std::uint32_t), alignof(std::int64_t),
      alignof(std::uint64_t), alignof(float),         alignof(double),
      alignof(bool),          alignof(std::uint16_t)};
  static constexpr scalar_kind kinds[]{
      scalar_kind::floating,       scalar_kind::floating,
      scalar_kind::signed_integer, scalar_kind::unsigned_integer,
      scalar_kind::signed_integer, scalar_kind::unsigned_integer,
      scalar_kind::signed_integer, scalar_kind::unsigned_integer,
      scalar_kind::signed_integer, scalar_kind::unsigned_integer,
      scalar_kind::complex,        scalar_kind::complex,
      scalar_kind::boolean,        scalar_kind::floating};
  const auto index = static_cast<unsigned>(descriptor.dtype);
  if (index >= std::size(sizes))
    return invalid_buffer("unsupported scalar dtype");
  const auto parsed = parse_buffer_dtype(descriptor.format);
  if (!parsed.native_endian || parsed.kind != kinds[index] ||
      parsed.size != sizes[index] || descriptor.itemsize != sizes[index])
    return invalid_buffer("buffer format and itemsize disagree");
  bool empty = false;
  for (auto extent : descriptor.shape()) {
    if (extent < 0)
      return invalid_buffer("negative array extent");
    empty = empty || !extent;
  }
  constexpr auto limit = static_cast<std::size_t>(PY_SSIZE_T_MAX);
  std::size_t bytes = empty ? 0 : descriptor.itemsize;
  if (!empty)
    for (auto extent : descriptor.shape()) {
      if (static_cast<std::size_t>(extent) > limit / bytes)
        return buffer_overflow();
      bytes *= static_cast<std::size_t>(extent);
    }
  if (bytes != descriptor.bytes)
    return invalid_buffer("buffer length and shape disagree");
  if (!bytes)
    return true;
  if (!descriptor.data)
    return invalid_buffer("nonempty buffer has a null pointer");
  const auto address = reinterpret_cast<std::uintptr_t>(descriptor.data);
  const auto alignment = alignments[index];
  if (address % alignment)
    return invalid_buffer("buffer origin is not aligned");
  std::size_t lower = 0, upper = 0;
  for (std::size_t axis = 0; axis < descriptor.rank(); ++axis) {
    if (descriptor.shape()[axis] <= 1)
      continue;
    const auto stride = descriptor.byte_strides()[axis];
    if (stride % static_cast<std::ptrdiff_t>(alignment))
      return invalid_buffer("buffer stride is not aligned");
    const auto magnitude =
        stride < 0 ? std::size_t{0} - static_cast<std::size_t>(stride)
                   : static_cast<std::size_t>(stride);
    auto &span = stride < 0 ? lower : upper;
    if (magnitude >
        (limit - span) / static_cast<std::size_t>(descriptor.shape()[axis] - 1))
      return buffer_overflow();
    span += magnitude * static_cast<std::size_t>(descriptor.shape()[axis] - 1);
  }
  if (lower > limit - upper || descriptor.itemsize > limit - lower - upper ||
      lower > address ||
      upper + descriptor.itemsize - 1 >
          std::numeric_limits<std::uintptr_t>::max() - address)
    return buffer_overflow();
  return true;
}

} // namespace pyb26::detail
