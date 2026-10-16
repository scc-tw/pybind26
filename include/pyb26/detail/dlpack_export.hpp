#pragma once

#include "array_error.hpp"
#include "class.hpp"
#include "dlpack.hpp"
#include "dlpack_owner.hpp"

namespace pyb26::detail {

struct dlpack_request {
  bool versioned = false;
  bool copy = false;

  static bool pair(PyObject *value, const char *name, std::uint32_t &first,
                   std::uint32_t &second) noexcept {
    if (!PyTuple_Check(value) || PyTuple_GET_SIZE(value) != 2) {
      PyErr_Format(PyExc_TypeError, "%s must be a pair of nonnegative integers",
                   name);
      return false;
    }
    std::uint32_t values[2];
    for (int i = 0; i < 2; ++i) {
      auto *item = PyTuple_GET_ITEM(value, i);
      if (PyBool_Check(item)) {
        PyErr_Format(PyExc_TypeError, "%s does not accept bool entries", name);
        return false;
      }
      PyObject *converted = nullptr;
      if (!PyLong_Check(item)) {
        converted = PyNumber_Index(item);
        if (!converted)
          return false;
        item = converted;
      }
      auto number = PyLong_AsUnsignedLong(item);
      Py_XDECREF(converted);
      if (PyErr_Occurred())
        return false;
      if (number > std::numeric_limits<std::uint32_t>::max()) {
        PyErr_Format(PyExc_OverflowError, "%s entries must fit uint32", name);
        return false;
      }
      values[i] = static_cast<std::uint32_t>(number);
    }
    first = values[0];
    second = values[1];
    return true;
  }

  bool parse(PyObject *const *args, Py_ssize_t nargs,
             PyObject *names) noexcept {
    if (nargs > 1) {
      PyErr_SetString(
          PyExc_TypeError,
          "__dlpack__ accepts at most one positional argument (stream)");
      return false;
    }
    PyObject *values[] = {nargs ? args[0] : Py_None, Py_None, Py_None, Py_None};
    unsigned seen = nargs ? 1u : 0u;
    const auto count = names ? PyTuple_GET_SIZE(names) : 0;
    constexpr const char *keys[] = {"stream", "max_version", "dl_device",
                                    "copy"};
    for (Py_ssize_t i = 0; i < count; ++i) {
      auto *name = PyTuple_GET_ITEM(names, i);
      int key = 0;
      for (; key < 4; ++key) {
        const int match = PyUnicode_CompareWithASCIIString(name, keys[key]);
        if (PyErr_Occurred())
          return false;
        if (!match)
          break;
      }
      if (key == 4) {
        PyErr_Format(PyExc_TypeError, "__dlpack__ got an unexpected keyword %R",
                     name);
        return false;
      }
      if (seen & (1u << key)) {
        PyErr_Format(PyExc_TypeError, "__dlpack__ got multiple values for %s",
                     keys[key]);
        return false;
      }
      seen |= 1u << key;
      values[key] = args[nargs + i];
    }
    if (values[0] != Py_None)
      return dlpack_error("CPU DLPack export requires stream=None");
    if (values[1] != Py_None) {
      std::uint32_t major, minor;
      if (!pair(values[1], "max_version", major, minor))
        return false;
      versioned = major >= 1;
    }
    if (values[2] != Py_None) {
      std::uint32_t device, index;
      if (!pair(values[2], "dl_device", device, index))
        return false;
      if (device != kDLCPU || index != 0)
        return dlpack_error("DLPack export supports CPU device zero only");
    }
    if (values[3] != Py_None && !PyBool_Check(values[3])) {
      PyErr_SetString(PyExc_TypeError, "copy must be True, False or None");
      return false;
    }
    copy = values[3] == Py_True;
    if (copy && !versioned)
      return dlpack_error(
          "explicit DLPack copy requires versioned negotiation");
    return true;
  }
};

template <class Managed>
inline constexpr const char *dlpack_unused_name =
    std::is_same_v<Managed, DLManagedTensor> ? "dltensor"
                                             : "dltensor_versioned";
template <class Managed>
inline constexpr const char *dlpack_used_name =
    std::is_same_v<Managed, DLManagedTensor> ? "used_dltensor"
                                             : "used_dltensor_versioned";

template <class Managed>
void native_dlpack_capsule_destroy(PyObject *capsule) noexcept {
  if (PyCapsule_IsValid(capsule, dlpack_used_name<Managed>))
    return;
  auto *tensor = static_cast<Managed *>(
      PyCapsule_GetPointer(capsule, dlpack_unused_name<Managed>));
  if (!tensor) {
    PyErr_WriteUnraisable(capsule);
    return;
  }
  tensor->deleter(tensor);
}

template <bool CheckError, class Managed, class T, std::size_t Rank,
          class Layout>
PyObject *wrap_native_dlpack_impl(shared_array<T, Rank, Layout> source,
                                  std::uint64_t flags) noexcept {
  if constexpr (CheckError) {
    if (PyErr_Occurred())
      return nullptr;
  }
  if (!source)
    return array_failure(source.error());
  auto *tensor =
      native_dlpack_owner<Managed, Rank>::create(std::move(source), flags);
  if (!tensor)
    return PyErr_NoMemory();
  auto *capsule = PyCapsule_New(tensor, dlpack_unused_name<Managed>,
                                &native_dlpack_capsule_destroy<Managed>);
  if (!capsule)
    tensor->deleter(tensor);
  return capsule;
}

template <class Managed, class T, std::size_t Rank, class Layout>
PyObject *wrap_native_dlpack(shared_array<T, Rank, Layout> source,
                             std::uint64_t flags) noexcept {
  return wrap_native_dlpack_impl<true, Managed>(std::move(source), flags);
}

template <class Source> struct dlpack_export_source {
  static_assert(!std::is_same_v<Source, Source>,
                "def_dlpack requires a shared_array returned by value");
};

template <class T, std::size_t Rank, class Layout>
struct dlpack_export_source<shared_array<T, Rank, Layout>> {
  static constexpr bool readonly = std::is_const_v<T>;
  static_assert(Rank <= 32, "DLPack export rank exceeds the supported limit");
  static constexpr auto dtype = dlpack_dtype<T>();

  static PyObject *export_capsule(shared_array<T, Rank, Layout> source,
                                  dlpack_request request) noexcept {
    return export_impl<true>(std::move(source), request);
  }

  template <bool CheckError>
  static PyObject *export_impl(shared_array<T, Rank, Layout> source,
                               dlpack_request request) noexcept {
    if constexpr (CheckError) {
      if (PyErr_Occurred())
        return nullptr;
    }
    if (!source)
      return array_failure(source.error());
    constexpr auto flags = readonly ? DLPACK_FLAG_BITMASK_READ_ONLY : 0;
    if (request.copy) {
      auto copy = copy_dlpack_storage(source.view());
      return wrap_native_dlpack<DLManagedTensorVersioned>(
          std::move(copy), flags | DLPACK_FLAG_BITMASK_IS_COPIED);
    }
    if (request.versioned)
      return wrap_native_dlpack_impl<false, DLManagedTensorVersioned>(
          std::move(source), flags);
    return wrap_native_dlpack_impl<false, DLManagedTensor>(std::move(source),
                                                           0);
  }
};

template <auto Getter> struct class_dlpack {
  using parent_type = typename getter_traits<decltype(Getter)>::cls;
  using result_type = typename getter_traits<decltype(Getter)>::ret;
  using source_type = dlpack_export_source<std::remove_cv_t<result_type>>;

  static PyObject *device(PyObject *self, PyObject *) noexcept {
    if (PyErr_Occurred())
      return nullptr;
    if (!cast_to<parent_type>(self)) {
      PyErr_SetString(PyExc_TypeError, "invalid DLPack exporter instance");
      return nullptr;
    }
    return Py_BuildValue("(ii)", kDLCPU, 0);
  }

  static PyObject *export_capsule(PyObject *self, PyObject *const *args,
                                  Py_ssize_t nargs, PyObject *names) noexcept {
    if (PyErr_Occurred())
      return nullptr;
    dlpack_request request;
    if (!request.parse(args, nargs, names))
      return nullptr;
    if constexpr (source_type::readonly) {
      if (!request.versioned)
        return dlpack_error(
                   "readonly DLPack export requires versioned negotiation"),
               nullptr;
    }
    auto *parent = cast_to<parent_type>(self);
    if (!parent) {
      PyErr_SetString(PyExc_TypeError, "invalid DLPack exporter instance");
      return nullptr;
    }
    auto snapshot = (parent->*Getter)();
    if (PyErr_Occurred())
      return nullptr;
    return source_type::template export_impl<false>(std::move(snapshot),
                                                    request);
  }

  static PyMethodDef *methods() noexcept {
    static PyMethodDef value[] = {
        {"__dlpack__", reinterpret_cast<PyCFunction>(&export_capsule),
         METH_FASTCALL | METH_KEYWORDS,
         "__dlpack__($self, /, stream=None, *, max_version=None, "
         "dl_device=None, copy=None)\n--\n\nExport a CPU DLPack capsule."},
        {"__dlpack_device__", &device, METH_NOARGS,
         "__dlpack_device__($self, /)\n--\n\nReturn the CPU device."},
        {nullptr, nullptr, 0, nullptr}};
    return value;
  }
};

} // namespace pyb26::detail

namespace pyb26 {

template <class T, class... Options>
template <auto Getter>
class_<T, Options...> &class_<T, Options...>::def_dlpack() {
  detail::type_mutation mutation(tr_);
  if (!mutation)
    return *this;
  using exporter = detail::class_dlpack<Getter>;
  static_assert(std::is_base_of_v<typename exporter::parent_type, T>,
                "DLPack getter must belong to the bound class or its base");
  if (PyErr_Occurred() || !tr_ || !tr_->py_type)
    return *this;
  for (auto *method = exporter::methods(); method->ml_name; ++method) {
    const int present =
        PyDict_ContainsString(tr_->py_type->tp_dict, method->ml_name);
    if (present < 0)
      return *this;
    if (present) {
      PyErr_SetString(PyExc_ValueError,
                      "DLPack methods are already registered on this class");
      return *this;
    }
  }
  for (auto *method = exporter::methods(); method->ml_name; ++method) {
    auto *descriptor = PyDescr_NewMethod(tr_->py_type, method);
    if (!descriptor)
      return *this;
    const int status =
        PyObject_SetAttrString(reinterpret_cast<PyObject *>(tr_->py_type),
                               method->ml_name, descriptor);
    Py_DECREF(descriptor);
    if (status < 0)
      return *this;
  }
  return *this;
}

} // namespace pyb26
