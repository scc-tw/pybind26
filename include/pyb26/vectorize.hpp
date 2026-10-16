#pragma once

#include "numeric/broadcast.hpp"
#include "numpy.hpp"

namespace pyb26::numpy::detail {

template <class T, std::size_t MaxRank> struct vector_argument {
  pyb26::detail::broadcast_operand<T> view;
};

template <class T, std::size_t MaxRank> struct vector_result {
  owned_array<T, 1> owner;
  std::array<std::size_t, MaxRank> shape{};
  std::size_t rank = 0;
  pyb26::detail::broadcast_error error = pyb26::detail::broadcast_error::none;
  PyObject *release() && noexcept {
    using E = pyb26::detail::broadcast_error;
    if (error != E::none) {
      PyErr_SetString(
          error == E::overflow ? PyExc_OverflowError : PyExc_ValueError,
          error == E::shape  ? "input shapes cannot be broadcast together"
          : error == E::rank ? "broadcast rank exceeds the supported limit"
                             : "broadcast size overflow");
      return nullptr;
    }
    if (!owner)
      return pyb26::detail::array_failure(owner.error());
    return wrap_native(dtype_of<T>(), rank, shape.data(),
                       pyb26::detail::owned_array_access::release(owner));
  }
};

template <auto Fn, std::size_t MaxRank, class Signature = decltype(Fn)>
struct vectorized;

template <auto Fn, std::size_t MaxRank, class R, class... A>
struct vectorized<Fn, MaxRank, R (*)(A...)> {
  static_assert(MaxRank <= PyBUF_MAX_NDIM,
                "vectorize rank exceeds the buffer protocol limit");
  static_assert(sizeof...(A) > 0,
                "vectorize requires at least one scalar argument");
  static_assert(((!std::is_reference_v<A> && !std::is_pointer_v<A>) && ...),
                "vectorize requires scalar arguments passed by value");
  static constexpr auto result_dtype = dtype_of<R>();
  static constexpr std::array argument_dtypes{dtype_of<A>()...};

  template <std::size_t Capacity>
  static vector_result<R, MaxRank>
  evaluate(const vector_argument<A, MaxRank> &...input) {
    static_assert(argument_dtypes.size() == sizeof...(A));
    static_assert(result_dtype == dtype_of<R>());
    auto plan = pyb26::detail::plan_broadcast<Capacity>(input.view...);
    vector_result<R, MaxRank> result;
    result.error = plan.error;
    if (result.error != pyb26::detail::broadcast_error::none)
      return result;
    result.rank = plan.rank;
    for (std::size_t axis = 0; axis < plan.rank; ++axis)
      result.shape[axis] = plan.shape[axis];
    result.owner = owned_array<R, 1>::allocate({plan.size});
    if (result.owner)
      pyb26::detail::broadcast_loop<Fn, A...>::run(result.owner.view().data(),
                                                   plan, input.view...);
    return result;
  }

  static vector_result<R, MaxRank>
  call(const vector_argument<A, MaxRank> &...input) {
    if constexpr (MaxRank > 4) {
      if (((input.view.rank <= 4) && ...))
        return evaluate<4>(input...);
    }
    return evaluate<MaxRank>(input...);
  }
};

template <auto Fn, std::size_t MaxRank, class R, class... A>
struct vectorized<Fn, MaxRank, R (*)(A...) noexcept>
    : vectorized<Fn, MaxRank, R (*)(A...)> {};

} // namespace pyb26::numpy::detail

namespace pyb26::numpy {

template <auto Fn, std::size_t MaxRank = PyBUF_MAX_NDIM>
inline constexpr auto vectorize = &detail::vectorized<Fn, MaxRank>::call;

}

namespace pyb26 {

template <class T, std::size_t MaxRank>
struct type_caster<numpy::detail::vector_argument<T, MaxRank>> {
  static constexpr caster_effects effects{true, true, true};
  using argument = numpy::detail::vector_argument<T, MaxRank>;
  detail::buffer_lease lease_;
  type_caster<T> scalar_;
  T value_{};
  argument argument_;

  bool load(PyObject *source) noexcept {
    argument_ = {};
    if (!PyObject_CheckBuffer(source)) {
      lease_.reset();
      if (!scalar_.load(source))
        return false;
      value_ = scalar_.template as<T>();
      argument_.view.data = &value_;
      return true;
    }
    if (!lease_.acquire(source))
      return false;
    const auto &buffer = lease_.get();
    if (!buffer.obj || buffer.ndim < 0 || buffer.ndim > PyBUF_MAX_NDIM ||
        buffer.itemsize <= 0 || buffer.len < 0 || !buffer.format)
      return detail::invalid_buffer("invalid buffer metadata");
    if (buffer.ndim > static_cast<int>(MaxRank))
      return false;
    if (buffer.ndim == 0) {
      if (buffer.shape || buffer.strides || buffer.suboffsets)
        return detail::invalid_buffer(
            "scalar buffer must not have shape or strides");
    } else if (!buffer.shape || !buffer.strides) {
      return detail::invalid_buffer(
          "buffer did not provide requested shape and strides");
    }
    const auto dtype = detail::parse_buffer_dtype_for<T>(buffer.format);
    if (dtype.kind == detail::numeric_kind::unsupported)
      return false;
    if (dtype.size != static_cast<std::size_t>(buffer.itemsize))
      return detail::invalid_buffer("buffer format and itemsize disagree");
    constexpr auto kind = element_traits<T>::kind;
    if (!dtype.native_endian || dtype.kind != kind || dtype.size != sizeof(T) ||
        buffer.suboffsets)
      return false;
    bool empty = false;
    for (int axis = 0; axis < buffer.ndim; ++axis) {
      if (buffer.shape[axis] < 0)
        return detail::invalid_buffer("negative array extent");
      empty = empty || buffer.shape[axis] == 0;
    }
    std::size_t bytes = empty ? 0 : sizeof(T);
    constexpr auto limit = static_cast<std::size_t>(PY_SSIZE_T_MAX);
    if (!empty) {
      for (int axis = 0; axis < buffer.ndim; ++axis) {
        if (static_cast<std::size_t>(buffer.shape[axis]) > limit / bytes)
          return detail::buffer_overflow();
        bytes *= static_cast<std::size_t>(buffer.shape[axis]);
      }
    }
    if (bytes != static_cast<std::size_t>(buffer.len))
      return detail::invalid_buffer("buffer length and shape disagree");
    if (bytes) {
      if (!buffer.buf)
        return detail::invalid_buffer("nonempty buffer has a null pointer");
      const auto address = reinterpret_cast<std::uintptr_t>(buffer.buf);
      if (address % alignof(T))
        return false;
      std::size_t lower = 0, upper = 0;
      for (int axis = 0; axis < buffer.ndim; ++axis) {
        if (buffer.shape[axis] <= 1)
          continue;
        auto stride = buffer.strides[axis];
        if (stride % alignof(T))
          return false;
        auto magnitude = stride < 0
                             ? std::size_t{0} - static_cast<std::size_t>(stride)
                             : static_cast<std::size_t>(stride);
        auto &extent = stride < 0 ? lower : upper;
        if (magnitude >
            (limit - extent) / static_cast<std::size_t>(buffer.shape[axis] - 1))
          return detail::buffer_overflow();
        extent += magnitude * static_cast<std::size_t>(buffer.shape[axis] - 1);
      }
      if (lower > limit - upper || sizeof(T) > limit - lower - upper ||
          lower > address ||
          upper + sizeof(T) - 1 >
              std::numeric_limits<std::uintptr_t>::max() - address)
        return detail::buffer_overflow();
    }
    static_assert(std::is_same_v<Py_ssize_t, std::ptrdiff_t>);
    argument_.view = {static_cast<const T *>(buffer.buf),
                      static_cast<std::size_t>(buffer.ndim), buffer.shape,
                      buffer.strides};
    return true;
  }

  template <class A> decltype(auto) as() noexcept {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(argument_);
    else
      return argument_;
  }

  template <class R> static PyObject *cast(R &&) noexcept {
    static_assert(!std::is_same_v<R, R>,
                  "borrowed vectorize argument cannot be returned");
    return nullptr;
  }
};

template <class T, std::size_t MaxRank>
struct type_caster<numpy::detail::vector_result<T, MaxRank>> {
  static constexpr caster_effects effects{false, false, false};
  static PyObject *
  cast(numpy::detail::vector_result<T, MaxRank> result) noexcept {
    return std::move(result).release();
  }
};

} // namespace pyb26
