#pragma once

#include "common.hpp"
#include "vendor/dlpack/dlpack.h"

#include <cstring>
#include <utility>

static_assert(DLPACK_MAJOR_VERSION == 1 && DLPACK_MINOR_VERSION >= 3,
              "pyb26 requires the DLPack 1.3 or newer 1.x header");

namespace pyb26::detail {

PYB26_COLD inline bool dlpack_error(const char *message) noexcept {
  PyErr_SetString(PyExc_BufferError, message);
  return false;
}

class dlpack_lease {
  DLManagedTensor *legacy_ = nullptr;
  DLManagedTensorVersioned *versioned_ = nullptr;

  void release() noexcept {
    auto *legacy = std::exchange(legacy_, nullptr);
    auto *versioned = std::exchange(versioned_, nullptr);
    if (legacy && legacy->deleter)
      legacy->deleter(legacy);
    if (versioned && versioned->deleter)
      versioned->deleter(versioned);
  }

  bool consume(PyObject *capsule) noexcept {
    if (!PyCapsule_CheckExact(capsule))
      return dlpack_error("DLPack producer must return a capsule");
    bool consumed = false;
    Py_BEGIN_CRITICAL_SECTION(capsule);
    const char *name = PyCapsule_GetName(capsule);
    if (name && std::strcmp(name, "dltensor") == 0) {
      auto *tensor =
          static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, name));
      if (tensor && PyCapsule_SetName(capsule, "used_dltensor") == 0) {
        legacy_ = tensor;
        consumed = true;
      }
    } else if (name && std::strcmp(name, "dltensor_versioned") == 0) {
      auto *tensor = static_cast<DLManagedTensorVersioned *>(
          PyCapsule_GetPointer(capsule, name));
      if (tensor &&
          PyCapsule_SetName(capsule, "used_dltensor_versioned") == 0) {
        versioned_ = tensor;
        consumed = true;
      }
    }
    Py_END_CRITICAL_SECTION();
    if (!consumed && !PyErr_Occurred())
      return dlpack_error("DLPack capsule is invalid or already consumed");
    return consumed;
  }

public:
  dlpack_lease() noexcept = default;
  dlpack_lease(const dlpack_lease &) = delete;
  dlpack_lease &operator=(const dlpack_lease &) = delete;
  ~dlpack_lease() noexcept { reset(); }

  void reset() noexcept {
    if (!legacy_ && !versioned_)
      return;
    if (PyErr_Occurred()) [[unlikely]] {
      PyObject *error = PyErr_GetRaisedException();
      release();
      PyErr_SetRaisedException(error);
    } else {
      release();
      if (PyErr_Occurred()) [[unlikely]]
        PyErr_WriteUnraisable(nullptr);
    }
  }

  bool acquire(PyObject *source, bool legacy) noexcept {
    reset();
    PyObject *method = PyObject_GetAttrString(source, "__dlpack__");
    if (!method)
      return false;
    PyObject *device =
        PyObject_CallMethod(source, "__dlpack_device__", nullptr);
    if (!device) {
      Py_DECREF(method);
      return false;
    }
    bool cpu = false;
    if (PyTuple_Check(device) && PyTuple_GET_SIZE(device) == 2) {
      auto type = PyLong_AsLong(PyTuple_GET_ITEM(device, 0));
      if (!PyErr_Occurred()) {
        auto index = PyLong_AsLong(PyTuple_GET_ITEM(device, 1));
        cpu = !PyErr_Occurred() && type == kDLCPU && index == 0;
      }
    }
    Py_DECREF(device);
    if (!cpu) {
      Py_DECREF(method);
      if (!PyErr_Occurred())
        return dlpack_error("DLPack input requires CPU device zero");
      return false;
    }
    PyObject *capsule = nullptr;
    if (legacy) {
      capsule = PyObject_CallNoArgs(method);
    } else {
      PyObject *keywords =
          Py_BuildValue("{s:(ii),s:O,s:O}", "max_version", 1, 3, "copy",
                        Py_False, "stream", Py_None);
      if (keywords) {
        PyObject *args = PyTuple_New(0);
        if (args) {
          capsule = PyObject_Call(method, args, keywords);
          Py_DECREF(args);
        }
        Py_DECREF(keywords);
      }
    }
    Py_DECREF(method);
    if (!capsule)
      return false;
    const bool consumed = consume(capsule);
    Py_DECREF(capsule);
    if (!consumed || PyErr_Occurred())
      return false;
    if (legacy != (legacy_ != nullptr))
      return dlpack_error(
          "DLPack capsule does not match the explicit negotiation mode");
    if (versioned_) {
      if (versioned_->version.major != DLPACK_MAJOR_VERSION)
        return dlpack_error("unsupported DLPack major version");
      if (versioned_->flags & ~std::uint64_t{DLPACK_FLAG_BITMASK_READ_ONLY})
        return dlpack_error(
            "DLPack no-copy input has copied or unsupported flags");
    }
    return true;
  }

  const DLTensor &tensor() const noexcept {
    return legacy_ ? legacy_->dl_tensor : versioned_->dl_tensor;
  }
  bool readonly() const noexcept {
    return versioned_ && (versioned_->flags & DLPACK_FLAG_BITMASK_READ_ONLY);
  }
  bool explicit_strides() const noexcept {
    return versioned_ && versioned_->version.minor >= 2;
  }
};

} // namespace pyb26::detail
