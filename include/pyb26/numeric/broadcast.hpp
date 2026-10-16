#pragma once

#include "owned_array.hpp"

#include <algorithm>
#include <tuple>

namespace pyb26::detail {

template <class T> struct broadcast_operand {
  const T *data = nullptr;
  std::size_t rank = 0;
  const std::ptrdiff_t *shape = nullptr;
  const std::ptrdiff_t *strides = nullptr;
};

enum class broadcast_error { none, shape, rank, overflow };

template <std::size_t MaxRank, std::size_t Arity> struct broadcast_plan {
  std::array<std::size_t, MaxRank> shape{};
  std::array<std::size_t, MaxRank> loops{};
  std::array<std::array<std::ptrdiff_t, MaxRank>, Arity> strides{};
  std::size_t rank = 0, loop_rank = 0, size = 1;
  broadcast_error error = broadcast_error::none;
};

template <std::size_t MaxRank, class... T>
auto plan_broadcast(const broadcast_operand<T> &...input) noexcept {
  broadcast_plan<MaxRank, sizeof...(T)> plan;
  ((plan.rank = std::max(plan.rank, input.rank)), ...);
  if (plan.rank > MaxRank) {
    plan.error = broadcast_error::rank;
    return plan;
  }
  for (std::size_t axis = 0; axis < plan.rank; ++axis)
    plan.shape[axis] = 1;
  auto merge = [&](const auto &operand) {
    for (std::size_t axis = 0; axis < operand.rank; ++axis) {
      if (operand.shape[axis] < 0) {
        plan.error = broadcast_error::shape;
        return;
      }
      auto extent = static_cast<std::size_t>(operand.shape[axis]);
      auto &dimension = plan.shape[plan.rank - operand.rank + axis];
      if (dimension == 1)
        dimension = extent;
      else if (extent != 1 && extent != dimension)
        plan.error = broadcast_error::shape;
    }
  };
  (merge(input), ...);
  if (plan.error != broadcast_error::none)
    return plan;
  for (std::size_t axis = 0; axis < plan.rank; ++axis)
    if (plan.shape[axis] == 0)
      plan.size = 0;
  constexpr auto limit =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (!plan.size)
    return plan;
  for (std::size_t axis = 0; axis < plan.rank; ++axis) {
    if (plan.shape[axis] > limit / plan.size) {
      plan.error = broadcast_error::overflow;
      return plan;
    }
    plan.size *= plan.shape[axis];
    if (plan.shape[axis] == 1)
      continue;
    std::array<std::ptrdiff_t, sizeof...(T)> steps{};
    std::size_t argument = 0;
    auto stride = [&](const auto &operand) {
      auto leading = plan.rank - operand.rank;
      if (axis >= leading && operand.shape[axis - leading] != 1)
        steps[argument] = operand.strides[axis - leading];
      ++argument;
    };
    (stride(input), ...);
    bool collapse = plan.loop_rank != 0;
    auto extent = static_cast<std::ptrdiff_t>(plan.shape[axis]);
    if (collapse) {
      for (std::size_t arg = 0; arg < sizeof...(T); ++arg) {
        auto previous = plan.strides[arg][plan.loop_rank - 1];
        collapse = collapse && previous % extent == 0 &&
                   previous / extent == steps[arg];
      }
    }
    auto index = collapse ? plan.loop_rank - 1 : plan.loop_rank++;
    plan.loops[index] =
        collapse ? plan.loops[index] * plan.shape[axis] : plan.shape[axis];
    for (std::size_t arg = 0; arg < sizeof...(T); ++arg)
      plan.strides[arg][index] = steps[arg];
  }
  return plan;
}

template <auto Fn, class... T> struct broadcast_loop {
  using result_type = std::invoke_result_t<decltype(Fn), T...>;
  static constexpr auto arity = sizeof...(T);
  using pointers = std::array<const std::byte *, arity>;
  using steps = std::array<std::ptrdiff_t, arity>;

  template <std::size_t Mask, std::size_t... I>
  static void contiguous(result_type *__restrict output, const pointers &data,
                         std::size_t size, std::index_sequence<I...>) {
    for (std::size_t i = 0; i < size; ++i)
      output[i] = Fn(reinterpret_cast<const T *>(
          data[I])[(Mask & (std::size_t{1} << I)) ? 0 : i]...);
  }

  template <std::size_t Index = 0, std::size_t Mask = 0>
  static void select(result_type *output, const pointers &data,
                     const steps &strides, std::size_t size) {
    if constexpr (Index == arity)
      contiguous<Mask>(output, data, size, std::index_sequence_for<T...>{});
    else if (strides[Index] == 0)
      select<Index + 1, Mask | (std::size_t{1} << Index)>(output, data, strides,
                                                          size);
    else
      select<Index + 1, Mask>(output, data, strides, size);
  }

  template <std::size_t... I>
  static void strided(result_type *__restrict output, const pointers &data,
                      const steps &strides, std::size_t size,
                      std::index_sequence<I...>) {
    for (std::size_t i = 0; i < size; ++i)
      output[i] = Fn(*reinterpret_cast<const T *>(
          data[I] + static_cast<std::ptrdiff_t>(i) * strides[I])...);
  }

  template <std::size_t MaxRank>
  [[gnu::noinline]]
  static void run(result_type *output,
                  const broadcast_plan<MaxRank, arity> &plan,
                  const broadcast_operand<T> &...input) {
    if (!plan.size)
      return;
    pointers data{reinterpret_cast<const std::byte *>(input.data)...};
    steps strides{};
    const auto inner = plan.loop_rank ? plan.loops[plan.loop_rank - 1] : 1;
    if (plan.loop_rank)
      for (std::size_t arg = 0; arg < arity; ++arg)
        strides[arg] = plan.strides[arg][plan.loop_rank - 1];
    constexpr std::array<std::size_t, arity> widths{sizeof(T)...};
    bool fast = true;
    for (std::size_t arg = 0; arg < arity; ++arg)
      fast = fast && (strides[arg] == 0 ||
                      strides[arg] == static_cast<std::ptrdiff_t>(widths[arg]));
    std::array<std::size_t, MaxRank> index{};
    for (std::size_t offset = 0; offset < plan.size; offset += inner) {
      if constexpr (arity <= 4) {
        if (fast)
          select(output + offset, data, strides, inner);
        else
          strided(output + offset, data, strides, inner,
                  std::index_sequence_for<T...>{});
      } else {
        strided(output + offset, data, strides, inner,
                std::index_sequence_for<T...>{});
      }
      if (offset + inner == plan.size)
        break;
      for (std::size_t axis = plan.loop_rank - 1; axis-- > 0;) {
        if (++index[axis] < plan.loops[axis]) {
          for (std::size_t arg = 0; arg < arity; ++arg)
            data[arg] += plan.strides[arg][axis];
          break;
        }
        index[axis] = 0;
        for (std::size_t arg = 0; arg < arity; ++arg)
          data[arg] -= static_cast<std::ptrdiff_t>(plan.loops[axis] - 1) *
                       plan.strides[arg][axis];
      }
    }
  }
};

} // namespace pyb26::detail
