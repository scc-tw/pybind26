#pragma once

#include "array.hpp"
#include "detail/array_descriptor.hpp"
#include "detail/array_error.hpp"
#include "numeric/dynamic_array.hpp"
#include "numeric/shared_array.hpp"
#include "pyb26.hpp"

namespace pyb26::detail {

template <class T> consteval const char *native_buffer_format() {
  return element_traits<std::remove_const_t<T>>::format;
}

inline bool valid_buffer_request(int flags) noexcept {
  constexpr int strides = PyBUF_STRIDES & ~PyBUF_ND;
  constexpr int layouts = (PyBUF_C_CONTIGUOUS | PyBUF_F_CONTIGUOUS |
                           PyBUF_ANY_CONTIGUOUS | PyBUF_INDIRECT) &
                          ~PyBUF_STRIDES;
  constexpr int supported = PyBUF_FULL | PyBUF_C_CONTIGUOUS |
                            PyBUF_F_CONTIGUOUS | PyBUF_ANY_CONTIGUOUS;
  if ((flags & ~supported) ||
      ((flags & (PyBUF_FORMAT | strides)) && !(flags & PyBUF_ND)) ||
      ((flags & layouts) && (flags & PyBUF_STRIDES) != PyBUF_STRIDES))
    return invalid_buffer("invalid buffer request flags");
  return true;
}

template <class T, std::size_t Rank> class native_buffer_owner {
  static_assert(Rank <= PyBUF_MAX_NDIM,
                "buffer export rank exceeds the protocol limit");
  static constexpr const char *format = native_buffer_format<T>();
  shared_native_storage *storage_;
  object anchor_;
  void *data_;
  Py_ssize_t bytes_;
  std::array<Py_ssize_t, Rank> shape_;
  std::array<Py_ssize_t, Rank> strides_;

public:
  template <class Layout>
  explicit native_buffer_owner(
      shared_array<T, Rank, Layout> &&source) noexcept {
    const auto view = source.view();
    data_ = const_cast<std::remove_const_t<T> *>(view.data());
    bytes_ = static_cast<Py_ssize_t>(view.size() * sizeof(T));
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      shape_[axis] = static_cast<Py_ssize_t>(view.extent(axis));
      strides_[axis] = static_cast<Py_ssize_t>(view.stride_bytes(axis));
    }
    storage_ = shared_array_access::release(source);
    anchor_ =
        object::steal(static_cast<PyObject *>(retain_storage_export(storage_)));
  }
  template <class Layout>
  explicit native_buffer_owner(
      external_array<T, Rank, Layout> &&source) noexcept {
    const auto view = source.view();
    data_ = const_cast<std::remove_const_t<T> *>(view.data());
    bytes_ = static_cast<Py_ssize_t>(view.size() * sizeof(T));
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      shape_[axis] = static_cast<Py_ssize_t>(view.extent(axis));
      strides_[axis] = static_cast<Py_ssize_t>(view.stride_bytes(axis));
    }
    storage_ = source.release_owner().release();
    anchor_ =
        object::steal(static_cast<PyObject *>(retain_storage_export(storage_)));
  }
  native_buffer_owner(const native_buffer_owner &) = delete;
  native_buffer_owner &operator=(const native_buffer_owner &) = delete;
  native_buffer_owner(native_buffer_owner &&other) noexcept
      : storage_(std::exchange(other.storage_, nullptr)),
        anchor_(std::move(other.anchor_)), data_(other.data_),
        bytes_(other.bytes_), shape_(other.shape_), strides_(other.strides_) {}
  ~native_buffer_owner() noexcept { release_shared_storage(storage_); }

  void describe(Py_buffer *view, int flags) const noexcept {
    *view = {};
    view->buf = data_;
    view->len = bytes_;
    view->itemsize = sizeof(T);
    view->readonly = std::is_const_v<T>;
    view->ndim = Rank;
    if (flags & PyBUF_FORMAT)
      view->format = const_cast<char *>(format);
    if constexpr (Rank != 0) {
      if (flags & PyBUF_ND)
        view->shape = const_cast<Py_ssize_t *>(shape_.data());
      if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES)
        view->strides = const_cast<Py_ssize_t *>(strides_.data());
    }
  }

  bool supports(int flags) const noexcept {
    if (PyErr_Occurred())
      return false;
    if (!storage_ || (!anchor_ && storage_->storage->valid &&
                      !storage_->storage->valid(storage_->storage)))
      return invalid_buffer("buffer storage is closed");
    if constexpr (std::is_const_v<T>) {
      if (flags & PyBUF_WRITABLE)
        return invalid_buffer("buffer is read-only");
    }
    Py_buffer view;
    describe(&view, PyBUF_FULL_RO);
    if (((flags & PyBUF_STRIDES) != PyBUF_STRIDES ||
         (flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS) &&
        !PyBuffer_IsContiguous(&view, 'C'))
      return invalid_buffer("buffer is not C-contiguous");
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
        !PyBuffer_IsContiguous(&view, 'F'))
      return invalid_buffer("buffer is not F-contiguous");
    if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS &&
        !PyBuffer_IsContiguous(&view, 'A'))
      return invalid_buffer("buffer is not contiguous");
    return true;
  }

  static int getbuffer(PyObject *self, Py_buffer *view, int flags) noexcept {
    if (view)
      *view = {};
    if (PyErr_Occurred())
      return -1;
    if (!view)
      return invalid_buffer("null buffer descriptor"), -1;
    if (!valid_buffer_request(flags))
      return -1;
    auto *owner = cast_to<native_buffer_owner>(self);
    if (!owner) {
      PyErr_SetString(PyExc_TypeError, "invalid native buffer owner");
      return -1;
    }
    if (!owner->supports(flags))
      return -1;
    owner->describe(view, flags);
    view->obj = Py_NewRef(self);
    return 0;
  }
};

class dynamic_buffer_owner {
  array_descriptor descriptor_;
  object anchor_;
  std::string format_;
  bool valid_ = false;

public:
  explicit dynamic_buffer_owner(array_descriptor source) noexcept
      : descriptor_(std::move(source)) {
    if (!validate_array_descriptor(descriptor_))
      return;
    anchor_ = object::steal(
        static_cast<PyObject *>(descriptor_.owner.retain_export()));
    if (PyErr_Occurred())
      return;
    format_ = descriptor_.format;
    descriptor_.metadata =
        std::make_shared<const array_metadata>(*descriptor_.metadata);
    valid_ = true;
  }
  template <class T>
  explicit dynamic_buffer_owner(dynamic_array<T> &&source) noexcept
      : dynamic_buffer_owner(source.release_descriptor()) {}
  dynamic_buffer_owner(const dynamic_buffer_owner &) = delete;
  dynamic_buffer_owner(dynamic_buffer_owner &&) noexcept = default;
  void describe(Py_buffer *view, int flags) const noexcept {
    static_assert(std::is_same_v<Py_ssize_t, std::ptrdiff_t>);
    *view = {};
    view->buf = descriptor_.data;
    view->len = static_cast<Py_ssize_t>(descriptor_.bytes);
    view->itemsize = static_cast<Py_ssize_t>(descriptor_.itemsize);
    view->readonly = descriptor_.readonly;
    view->ndim = static_cast<int>(descriptor_.rank());
    if (flags & PyBUF_FORMAT)
      view->format = const_cast<char *>(format_.c_str());
    if (descriptor_.rank()) {
      if (flags & PyBUF_ND)
        view->shape = const_cast<Py_ssize_t *>(descriptor_.shape().data());
      if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES)
        view->strides =
            const_cast<Py_ssize_t *>(descriptor_.byte_strides().data());
    }
  }
  bool supports(int flags) const noexcept {
    if (PyErr_Occurred())
      return false;
    if (!valid_ || (!anchor_ && !descriptor_))
      return invalid_buffer("buffer storage is closed or invalid");
    if ((flags & PyBUF_WRITABLE) && descriptor_.readonly)
      return invalid_buffer("buffer is read-only");
    Py_buffer view;
    describe(&view, PyBUF_FULL_RO);
    if (((flags & PyBUF_STRIDES) != PyBUF_STRIDES ||
         (flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS) &&
        !PyBuffer_IsContiguous(&view, 'C'))
      return invalid_buffer("buffer is not C-contiguous");
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
        !PyBuffer_IsContiguous(&view, 'F'))
      return invalid_buffer("buffer is not F-contiguous");
    if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS &&
        !PyBuffer_IsContiguous(&view, 'A'))
      return invalid_buffer("buffer is not contiguous");
    return true;
  }
  static int getbuffer(PyObject *self, Py_buffer *view, int flags) noexcept {
    if (view)
      *view = {};
    if (PyErr_Occurred())
      return -1;
    if (!view)
      return invalid_buffer("null buffer descriptor"), -1;
    if (!valid_buffer_request(flags))
      return -1;
    auto *owner = cast_to<dynamic_buffer_owner>(self);
    if (!owner) {
      PyErr_SetString(PyExc_TypeError, "invalid buffer owner");
      return -1;
    }
    if (!owner->supports(flags))
      return -1;
    owner->describe(view, flags);
    view->obj = Py_NewRef(self);
    return 0;
  }
};

template <class T> struct buffer_source {
  static_assert(!std::is_same_v<T, T>,
                "def_buffer requires a shared_array returned by value");
};

template <class T, std::size_t Rank, class Layout>
struct buffer_source<shared_array<T, Rank, Layout>> {
  using owner_type = native_buffer_owner<T, Rank>;
};

template <class T, std::size_t Rank, class Layout>
struct buffer_source<external_array<T, Rank, Layout>> {
  using owner_type = native_buffer_owner<T, Rank>;
};

template <> struct buffer_source<array_descriptor> {
  using owner_type = dynamic_buffer_owner;
};

template <class T> struct buffer_source<dynamic_array<T>> {
  using owner_type = dynamic_buffer_owner;
};

template <class Owner> int prepare_buffer_owner(module_ module) noexcept {
  if (!module.state_ptr() || !PyModule_Check(module.state_ptr())) {
    PyErr_SetString(PyExc_TypeError, "buffer registration requires a module");
    return -1;
  }
  auto *dict = PyModule_GetDict(module.state_ptr());
  if (!dict)
    return -1;
  PyObject *private_module = nullptr;
  const int found =
      PyDict_GetItemStringRef(dict, "_pyb26_buffers", &private_module);
  if (found < 0)
    return -1;
  auto home = object::steal(private_module);
  if (!found) {
    auto name = object::steal(PyModule_GetNameObject(module.state_ptr()));
    if (!name)
      return -1;
    auto fullname =
        object::steal(PyUnicode_FromFormat("%U._pyb26_buffers", name.ptr()));
    if (!fullname)
      return -1;
    home = object::steal(PyModule_NewObject(fullname.ptr()));
    if (!home || PyModule_AddObjectRef(module.state_ptr(), "_pyb26_buffers",
                                       home.ptr()) < 0)
      return -1;
  } else if (!PyModule_Check(home.ptr())) {
    PyErr_SetString(PyExc_TypeError, "invalid private buffer type registry");
    return -1;
  }
  const char *name = typeid(Owner).name();
  auto *interpreter = PyInterpreterState_Get();
  if (auto *registered = resolve_in_chain(type_chain_of<Owner>, interpreter))
    return PyModule_AddObjectRef(
        home.ptr(), name, reinterpret_cast<PyObject *>(registered->py_type));
  static const PyType_Slot slots[] = {
      {Py_bf_getbuffer, reinterpret_cast<void *>(&Owner::getbuffer)}};
  class_<Owner> binding(module_(home.ptr(), module.state_ptr()), name,
                        class_options<>{.slots = slots, .count = 1});
  if (PyErr_Occurred())
    return -1;
  auto *registered = resolve_in_chain(type_chain_of<Owner>, interpreter);
  if (!registered)
    return error_unregistered(name), -1;
  registered->py_type->tp_flags &= ~Py_TPFLAGS_BASETYPE;
  return PyType_Freeze(registered->py_type);
}

template <auto Getter> struct class_buffer {
  using parent_type = typename member_sig<decltype(Getter)>::cls;
  using result_type =
      decltype(std::invoke(Getter, std::declval<parent_type &>()));
  using owner_type =
      typename buffer_source<std::remove_cv_t<result_type>>::owner_type;

  static int getbuffer(PyObject *self, Py_buffer *view, int flags) noexcept {
    if (view)
      *view = {};
    if (PyErr_Occurred())
      return -1;
    if (!view)
      return invalid_buffer("null buffer descriptor"), -1;
    if (!valid_buffer_request(flags))
      return -1;
    auto *parent = cast_to<parent_type>(self);
    if (!parent) {
      PyErr_SetString(PyExc_TypeError, "invalid buffer exporter instance");
      return -1;
    }
    auto snapshot = (parent->*Getter)();
    if (PyErr_Occurred())
      return -1;
    if (!snapshot)
      return array_failure(snapshot.error()), -1;
    owner_type owner(std::move(snapshot));
    if (!owner.supports(flags))
      return -1;
    PyObject *lease = make_instance<owner_type>(
        [&](void *memory) { ::new (memory) owner_type(std::move(owner)); });
    if (!lease)
      return -1;
    auto *stored = reinterpret_cast<owner_type *>(
        reinterpret_cast<char *>(lease) + holder_data_offset<owner_type>());
    stored->describe(view, flags);
    view->obj = lease;
    return 0;
  }

  static const PyType_Slot *slots() noexcept {
    static const PyType_Slot value[] = {
        {Py_bf_getbuffer, reinterpret_cast<void *>(&getbuffer)}};
    return value;
  }
};

} // namespace pyb26::detail

namespace pyb26 {

template <auto Getter> auto def_buffer() noexcept {
  using exporter = detail::class_buffer<Getter>;
  return class_options<typename exporter::parent_type>{
      .slots = exporter::slots(),
      .count = 1,
      .prepare = &detail::prepare_buffer_owner<typename exporter::owner_type>};
}

} // namespace pyb26
