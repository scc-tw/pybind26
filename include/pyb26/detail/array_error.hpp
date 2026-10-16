#pragma once

#include "../numeric/owned_array.hpp"
#include "buffer.hpp"

namespace pyb26::detail {

PYB26_COLD inline PyObject *array_failure(array_error error) noexcept {
  if (PyErr_Occurred())
    return nullptr;
  switch (error) {
  case array_error::out_of_memory:
    return PyErr_NoMemory();
  case array_error::overflow:
    buffer_overflow();
    return nullptr;
  case array_error::invalid_storage:
    PyErr_SetString(PyExc_BufferError, "native allocation has invalid storage");
    return nullptr;
  case array_error::invalid_slice:
    PyErr_SetString(PyExc_IndexError,
                    "native slice has invalid bounds or a zero step");
    return nullptr;
  case array_error::invalid_axes:
    PyErr_SetString(PyExc_ValueError,
                    "native transpose requires a permutation of the axes");
    return nullptr;
  default:
    PyErr_SetString(PyExc_RuntimeError, "native result has no allocation");
    return nullptr;
  }
}

} // namespace pyb26::detail
