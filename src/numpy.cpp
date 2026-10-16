#include <cstring>
#include <pyb26/numpy.hpp>

#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

namespace pyb26::numpy {

static_assert(sizeof(npy_bool) == sizeof(bool) &&
              alignof(npy_bool) == alignof(bool) && NPY_FALSE == 0 &&
              NPY_TRUE == 1);

int initialize() noexcept {
  if (PyInterpreterState_Get() != PyInterpreterState_Main()) {
    PyErr_SetString(PyExc_ImportError,
                    "pyb26 NumPy adapter requires the main interpreter");
    return -1;
  }
  if (PyArray_API)
    return 0;
  if (_import_array() < 0) {
    PyArray_API = nullptr;
    PyArray_RUNTIME_VERSION = 0;
    return -1;
  }
  return 0;
}

namespace detail {

bool numpy_buffer_metadata(PyObject *source, Py_buffer &metadata) noexcept {
  if (!PyArray_API || !PyArray_Check(source))
    return false;
  static_assert(sizeof(npy_intp) == sizeof(Py_ssize_t) &&
                alignof(npy_intp) == alignof(Py_ssize_t));
  auto *array = reinterpret_cast<PyArrayObject *>(source);
  metadata.buf = PyArray_DATA(array);
  metadata.ndim = PyArray_NDIM(array);
  metadata.shape = metadata.ndim
                       ? reinterpret_cast<Py_ssize_t *>(PyArray_DIMS(array))
                       : nullptr;
  metadata.strides =
      metadata.ndim ? reinterpret_cast<Py_ssize_t *>(PyArray_STRIDES(array))
                    : nullptr;
  metadata.readonly = !PyArray_ISWRITEABLE(array);
  return true;
}

static void destroy_native_capsule(PyObject *capsule) noexcept {
  auto *storage = static_cast<pyb26::detail::native_storage *>(
      PyCapsule_GetPointer(capsule, "pyb26.native_storage"));
  if (storage)
    storage->destroy(storage);
}

static void destroy_shared_capsule(PyObject *capsule) noexcept {
  auto *storage = static_cast<pyb26::detail::shared_native_storage *>(
      PyCapsule_GetPointer(capsule, "pyb26.shared_storage"));
  pyb26::detail::release_shared_storage(storage);
  Py_XDECREF(static_cast<PyObject *>(PyCapsule_GetContext(capsule)));
}

PyObject *wrap_shared(dtype type, std::size_t rank, const std::size_t *shape,
                      const std::ptrdiff_t *strides, const void *data,
                      bool writable,
                      pyb26::detail::shared_native_storage *storage) noexcept {
  std::unique_ptr<pyb26::detail::shared_native_storage,
                  decltype(&pyb26::detail::release_shared_storage)>
      owner(storage, &pyb26::detail::release_shared_storage);
  if (PyErr_Occurred())
    return nullptr;
  object anchor = object::steal(
      static_cast<PyObject *>(pyb26::detail::retain_storage_export(storage)));
  if (PyErr_Occurred())
    return nullptr;
  if (!PyArray_API) [[unlikely]] {
    PyErr_SetString(
        PyExc_RuntimeError,
        "call pyb26::numpy::initialize during module initialization");
    return nullptr;
  }
  if (rank > NPY_MAXDIMS) [[unlikely]] {
    PyErr_SetString(PyExc_ValueError,
                    "NumPy output rank exceeds the supported limit");
    return nullptr;
  }
  static constexpr int types[]{
      NPY_FLOAT32,   NPY_FLOAT64,    NPY_INT8,   NPY_UINT8,  NPY_INT16,
      NPY_UINT16,    NPY_INT32,      NPY_UINT32, NPY_INT64,  NPY_UINT64,
      NPY_COMPLEX64, NPY_COMPLEX128, NPY_BOOL,   NPY_FLOAT16};
  const auto index = static_cast<unsigned>(type);
  if (index >= std::size(types)) [[unlikely]] {
    PyErr_SetString(PyExc_TypeError, "unsupported NumPy output dtype");
    return nullptr;
  }
  npy_intp dims[NPY_MAXDIMS], steps[NPY_MAXDIMS];
  for (std::size_t i = 0; i < rank; ++i) {
    if (shape[i] > static_cast<std::size_t>(NPY_MAX_INTP)) [[unlikely]] {
      pyb26::detail::buffer_overflow();
      return nullptr;
    }
    dims[i] = static_cast<npy_intp>(shape[i]);
    if (strides)
      steps[i] = static_cast<npy_intp>(strides[i]);
  }
  object capsule = object::steal(
      PyCapsule_New(storage, "pyb26.shared_storage", &destroy_shared_capsule));
  if (!capsule)
    return nullptr;
  owner.release();
  if (anchor) {
    if (PyCapsule_SetContext(capsule.ptr(), anchor.ptr()) < 0)
      return nullptr;
    anchor.release();
  }
  auto *descr = PyArray_DescrFromType(types[index]);
  if (!descr)
    return nullptr;
  object result = object::steal(PyArray_NewFromDescr(
      &PyArray_Type, descr, static_cast<int>(rank), dims,
      strides ? steps : nullptr, const_cast<void *>(data),
      NPY_ARRAY_ALIGNED | (writable ? NPY_ARRAY_WRITEABLE : 0), nullptr));
  if (!result)
    return nullptr;
  if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(result.ptr()),
                            capsule.release()) < 0)
    return nullptr;
  return result.release();
}

PyObject *wrap_native(dtype type, std::size_t rank, const std::size_t *shape,
                      pyb26::detail::native_storage *storage) noexcept {
  using storage_ptr =
      std::unique_ptr<pyb26::detail::native_storage,
                      void (*)(pyb26::detail::native_storage *) noexcept>;
  storage_ptr owner(storage, storage->destroy);
  if (PyErr_Occurred())
    return nullptr;
  if (!PyArray_API) [[unlikely]] {
    PyErr_SetString(
        PyExc_RuntimeError,
        "call pyb26::numpy::initialize during module initialization");
    return nullptr;
  }
  if (rank > NPY_MAXDIMS) [[unlikely]] {
    PyErr_SetString(PyExc_ValueError,
                    "NumPy output rank exceeds the supported limit");
    return nullptr;
  }
  static constexpr int types[]{
      NPY_FLOAT32,   NPY_FLOAT64,    NPY_INT8,   NPY_UINT8,  NPY_INT16,
      NPY_UINT16,    NPY_INT32,      NPY_UINT32, NPY_INT64,  NPY_UINT64,
      NPY_COMPLEX64, NPY_COMPLEX128, NPY_BOOL,   NPY_FLOAT16};
  const auto index = static_cast<unsigned>(type);
  if (index >= std::size(types)) [[unlikely]] {
    PyErr_SetString(PyExc_TypeError, "unsupported NumPy output dtype");
    return nullptr;
  }
  npy_intp dims[NPY_MAXDIMS];
  for (std::size_t i = 0; i < rank; ++i) {
    if (shape[i] > static_cast<std::size_t>(NPY_MAX_INTP)) [[unlikely]] {
      pyb26::detail::buffer_overflow();
      return nullptr;
    }
    dims[i] = static_cast<npy_intp>(shape[i]);
  }
  object capsule = object::steal(
      PyCapsule_New(storage, "pyb26.native_storage", &destroy_native_capsule));
  if (!capsule)
    return nullptr;
  owner.release();
  object result = object::steal(PyArray_SimpleNewFromData(
      static_cast<int>(rank), dims, types[index], storage->data));
  if (!result)
    return nullptr;
  if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(result.ptr()),
                            capsule.release()) < 0)
    return nullptr;
  return result.release();
}

PyObject *allocate(dtype type, std::size_t rank, const std::size_t *shape,
                   void **data) noexcept {
  if (!PyArray_API) [[unlikely]] {
    PyErr_SetString(
        PyExc_RuntimeError,
        "call pyb26::numpy::initialize during module initialization");
    return nullptr;
  }
  if (rank > NPY_MAXDIMS) [[unlikely]] {
    PyErr_SetString(PyExc_ValueError,
                    "NumPy output rank exceeds the supported limit");
    return nullptr;
  }
  npy_intp dims[NPY_MAXDIMS];
  bool empty = false;
  for (std::size_t i = 0; i < rank; ++i) {
    if (shape[i] > static_cast<std::size_t>(NPY_MAX_INTP)) [[unlikely]] {
      pyb26::detail::buffer_overflow();
      return nullptr;
    }
    dims[i] = static_cast<npy_intp>(shape[i]);
    empty = empty || shape[i] == 0;
  }
  int typenum;
  std::size_t itemsize;
  switch (type) {
  case dtype::f32:
    typenum = NPY_FLOAT32;
    itemsize = 4;
    break;
  case dtype::f64:
    typenum = NPY_FLOAT64;
    itemsize = 8;
    break;
  case dtype::i8:
    typenum = NPY_INT8;
    itemsize = 1;
    break;
  case dtype::u8:
    typenum = NPY_UINT8;
    itemsize = 1;
    break;
  case dtype::i16:
    typenum = NPY_INT16;
    itemsize = 2;
    break;
  case dtype::u16:
    typenum = NPY_UINT16;
    itemsize = 2;
    break;
  case dtype::i32:
    typenum = NPY_INT32;
    itemsize = 4;
    break;
  case dtype::u32:
    typenum = NPY_UINT32;
    itemsize = 4;
    break;
  case dtype::i64:
    typenum = NPY_INT64;
    itemsize = 8;
    break;
  case dtype::u64:
    typenum = NPY_UINT64;
    itemsize = 8;
    break;
  case dtype::c64:
    typenum = NPY_COMPLEX64;
    itemsize = 8;
    break;
  case dtype::c128:
    typenum = NPY_COMPLEX128;
    itemsize = 16;
    break;
  case dtype::boolean:
    typenum = NPY_BOOL;
    itemsize = 1;
    break;
  case dtype::f16:
    typenum = NPY_FLOAT16;
    itemsize = 2;
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "unsupported NumPy output dtype");
    return nullptr;
  }
  if (!empty) {
    std::size_t bytes = itemsize;
    for (std::size_t i = 0; i < rank; ++i) {
      if (shape[i] > static_cast<std::size_t>(NPY_MAX_INTP) / bytes)
          [[unlikely]] {
        pyb26::detail::buffer_overflow();
        return nullptr;
      }
      bytes *= shape[i];
    }
  }
  PyObject *result = PyArray_SimpleNew(static_cast<int>(rank), dims, typenum);
  if (result)
    *data = PyArray_DATA(reinterpret_cast<PyArrayObject *>(result));
  return result;
}

} // namespace detail
} // namespace pyb26::numpy

namespace pyb26::numpy::detail {
bool half_scalar_bits(PyObject *source, std::uint16_t &bits) noexcept {
  if (!PyArray_API || Py_TYPE(source) != &PyHalfArrType_Type)
    return false;
  pyb26::detail::buffer_lease lease;
  if (!lease.acquire(source))
    return false;
  const auto &buffer = lease.get();
  const auto dtype = pyb26::detail::parse_buffer_dtype(buffer.format);
  if (buffer.ndim || buffer.itemsize != 2 || buffer.len != 2 || !buffer.buf ||
      dtype.kind != scalar_kind::floating || dtype.size != 2 ||
      !dtype.native_endian)
    return false;
  std::memcpy(&bits, buffer.buf, sizeof(bits));
  return true;
}
} // namespace pyb26::numpy::detail
