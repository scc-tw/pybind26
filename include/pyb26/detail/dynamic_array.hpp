#pragma once

#include "../numeric/dynamic_array.hpp"
#include "array_layout.hpp"
#include "lifetime.hpp"

namespace pyb26::detail {

struct python_buffer_payload {
  Py_buffer buffer;
  PyObject *input;
  static void release(PyObject *capsule) noexcept {
    auto *payload = static_cast<python_buffer_payload *>(
        PyCapsule_GetPointer(capsule, "pyb26.buffer_owner"));
    if (!payload)
      return;
    PyBuffer_Release(&payload->buffer);
    Py_DECREF(payload->input);
    delete payload;
  }
};

struct python_array_storage : native_storage {
  python_owner_ref owner;
  python_array_storage(void *pointer, std::size_t length,
                       python_owner_ref retained) noexcept
      : native_storage{pointer, length, &dispose, &alive, &retain_export},
        owner(std::move(retained)) {}
  static bool alive(native_storage *pointer) noexcept {
    return static_cast<bool>(
        static_cast<python_array_storage *>(pointer)->owner);
  }
  static void *retain_export(native_storage *pointer) noexcept {
    auto &owner = static_cast<python_array_storage *>(pointer)->owner;
    auto *current = cpython::attached_thread_state();
    if (!current || current->interp != owner.context().interpreter) {
      if (current)
        PyErr_SetString(PyExc_RuntimeError,
                        "buffer export requires its owner interpreter");
      return nullptr;
    }
    owner_admission admission(owner.context());
    if (!admission || !owner) {
      PyErr_SetString(PyExc_RuntimeError, "buffer owner is closed");
      return nullptr;
    }
    return Py_NewRef(owner.get());
  }
  static void dispose(native_storage *pointer) noexcept {
    delete static_cast<python_array_storage *>(pointer);
  }
};

inline array_owner retain_python_buffer(Py_buffer buffer,
                                        PyObject *source) noexcept {
  auto *payload =
      new (std::nothrow) python_buffer_payload{buffer, Py_NewRef(source)};
  if (!payload) {
    PyBuffer_Release(&buffer);
    PyErr_NoMemory();
    return {};
  }
  PyObject *capsule = PyCapsule_New(payload, "pyb26.buffer_owner",
                                    &python_buffer_payload::release);
  if (!capsule) {
    PyBuffer_Release(&payload->buffer);
    Py_DECREF(payload->input);
    delete payload;
    return {};
  }
  auto retained = python_owner_ref::steal(capsule);
  if (!retained)
    return {};
  auto *storage = new (std::nothrow) python_array_storage(
      buffer.buf, static_cast<std::size_t>(buffer.len), std::move(retained));
  if (!storage) {
    PyErr_NoMemory();
    return {};
  }
  auto result = array_owner::adopt(storage);
  if (!result)
    PyErr_NoMemory();
  return result;
}

inline bool validate_dynamic_buffer_metadata(
    const Py_buffer &buffer, std::size_t itemsize, std::size_t alignment,
    scalar_kind expected_kind, scalar_dtype expected_dtype, const char *format,
    bool writable, array_descriptor &result) noexcept {
  if (!buffer.obj || buffer.ndim < 0 || buffer.ndim > PyBUF_MAX_NDIM ||
      buffer.itemsize <= 0 || buffer.len < 0 || !buffer.format)
    return invalid_buffer("invalid buffer metadata");
  if ((buffer.ndim && (!buffer.shape || !buffer.strides)) ||
      (!buffer.ndim && (buffer.shape || buffer.strides || buffer.suboffsets)))
    return invalid_buffer("buffer did not provide valid shape and strides");
  const auto dtype = parse_buffer_dtype(buffer.format);
  if (dtype.size != static_cast<std::size_t>(buffer.itemsize))
    return dtype.kind == scalar_kind::unsupported
               ? false
               : invalid_buffer("buffer format and itemsize disagree");
  if (!dtype.native_endian || dtype.kind != expected_kind ||
      dtype.size != itemsize || buffer.suboffsets ||
      (writable && buffer.readonly))
    return false;
  bool empty = false;
  for (int axis = 0; axis < buffer.ndim; ++axis) {
    if (buffer.shape[axis] < 0)
      return invalid_buffer("negative array extent");
    empty = empty || !buffer.shape[axis];
  }
  constexpr auto limit = static_cast<std::size_t>(PY_SSIZE_T_MAX);
  std::size_t bytes = empty ? 0 : itemsize;
  if (!empty)
    for (int axis = 0; axis < buffer.ndim; ++axis) {
      if (static_cast<std::size_t>(buffer.shape[axis]) > limit / bytes)
        return buffer_overflow();
      bytes *= static_cast<std::size_t>(buffer.shape[axis]);
    }
  if (bytes != static_cast<std::size_t>(buffer.len))
    return invalid_buffer("buffer length and shape disagree");
  if (bytes) {
    if (!buffer.buf)
      return invalid_buffer("nonempty buffer has a null pointer");
    const auto address = reinterpret_cast<std::uintptr_t>(buffer.buf);
    if (address % alignment)
      return false;
    std::size_t lower = 0, upper = 0;
    for (int axis = 0; axis < buffer.ndim; ++axis) {
      if (buffer.shape[axis] <= 1)
        continue;
      const auto stride = buffer.strides[axis];
      if (stride % alignment)
        return false;
      const auto magnitude =
          stride < 0 ? std::size_t{0} - static_cast<std::size_t>(stride)
                     : static_cast<std::size_t>(stride);
      auto &span = stride < 0 ? lower : upper;
      if (magnitude >
          (limit - span) / static_cast<std::size_t>(buffer.shape[axis] - 1))
        return buffer_overflow();
      span += magnitude * static_cast<std::size_t>(buffer.shape[axis] - 1);
    }
    if (lower > limit - upper || itemsize > limit - lower - upper ||
        lower > address ||
        upper + itemsize - 1 >
            std::numeric_limits<std::uintptr_t>::max() - address)
      return buffer_overflow();
  }
  auto metadata = std::make_shared<array_metadata>();
  if (buffer.ndim) {
    metadata->shape.assign(buffer.shape, buffer.shape + buffer.ndim);
    metadata->byte_strides.assign(buffer.strides, buffer.strides + buffer.ndim);
  }
  result = {buffer.buf,
            itemsize,
            expected_dtype,
            format,
            !writable || static_cast<bool>(buffer.readonly),
            bytes,
            std::move(metadata),
            {}};
  return true;
}

template <class T>
bool validate_dynamic_buffer(const Py_buffer &buffer,
                             array_descriptor &result) noexcept {
  static_assert(element_traits<std::remove_const_t<T>>::enabled);
  return validate_dynamic_buffer_metadata(
      buffer, sizeof(T), alignof(T), element_traits<T>::kind,
      element_traits<T>::dtype, element_traits<T>::format, !std::is_const_v<T>,
      result);
}

} // namespace pyb26::detail
