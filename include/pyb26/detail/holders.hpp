#pragma once

#include "caster.hpp"
#include "lifetime.hpp"

#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace pyb26::detail {

enum class rv_policy : unsigned char {
  automatic = 0,
  copy,
  move,
  reference,
  take_ownership,
  reference_internal,
};

namespace rv {
struct take_ownership_t {
  static constexpr rv_policy value = rv_policy::take_ownership;
};
struct copy_t {
  static constexpr rv_policy value = rv_policy::copy;
};
struct move_t {
  static constexpr rv_policy value = rv_policy::move;
};
struct reference_t {
  static constexpr rv_policy value = rv_policy::reference;
};
struct reference_internal_t {
  static constexpr rv_policy value = rv_policy::reference_internal;
};
struct automatic_t {
  static constexpr rv_policy value = rv_policy::automatic;
};
constexpr take_ownership_t take_ownership() noexcept { return {}; }
constexpr copy_t copy() noexcept { return {}; }
constexpr move_t move() noexcept { return {}; }
constexpr reference_t reference() noexcept { return {}; }
constexpr reference_internal_t reference_internal() noexcept { return {}; }
constexpr automatic_t automatic() noexcept { return {}; }
} // namespace rv

template <class E> struct rv_tag_traits {
  static constexpr bool is = false;
  static constexpr rv_policy value = rv_policy::automatic;
};
template <> struct rv_tag_traits<rv::take_ownership_t> {
  static constexpr bool is = true;
  static constexpr rv_policy value = rv_policy::take_ownership;
};
template <> struct rv_tag_traits<rv::copy_t> {
  static constexpr bool is = true;
  static constexpr rv_policy value = rv_policy::copy;
};
template <> struct rv_tag_traits<rv::move_t> {
  static constexpr bool is = true;
  static constexpr rv_policy value = rv_policy::move;
};
template <> struct rv_tag_traits<rv::reference_t> {
  static constexpr bool is = true;
  static constexpr rv_policy value = rv_policy::reference;
};
template <> struct rv_tag_traits<rv::reference_internal_t> {
  static constexpr bool is = true;
  static constexpr rv_policy value = rv_policy::reference_internal;
};
template <> struct rv_tag_traits<rv::automatic_t> {
  static constexpr bool is = true;
  static constexpr rv_policy value = rv_policy::automatic;
};

template <class E>
constexpr bool is_rv_tag = rv_tag_traits<std::remove_cvref_t<E>>::is;

template <class... Extras> constexpr rv_policy policy_of() noexcept {
  rv_policy p = rv_policy::automatic;
  ((is_rv_tag<Extras>
        ? (p = rv_tag_traits<std::remove_cvref_t<Extras>>::value, 0)
        : 0),
   ...);
  return p;
}

inline const char *doc_of() noexcept { return nullptr; }
template <class E, class... Rest>
const char *doc_of(E &&e, Rest &&...rest) noexcept {
  if constexpr (std::is_convertible_v<E, const char *>)
    return static_cast<const char *>(e);
  else
    return doc_of(std::forward<Rest>(rest)...);
}

struct boxed_native_storage {
  void *allocation;
  void (*destroy)(void *) noexcept;
};

template <class T> struct holder_block {
  T *ptr;
  PyObject *keepalive;
  union {
    std::shared_ptr<T> shared;
    std::unique_ptr<T> unique;
    boxed_native_storage boxed;
  };
  holder_block() noexcept : ptr(nullptr), keepalive(nullptr) {}
  ~holder_block() noexcept {}
};

template <class T>
inline constexpr bool native_inline_v = alignof(T) <= alignof(std::max_align_t);

template <class T> constexpr Py_ssize_t storage_payload_size() noexcept {
  if constexpr (!native_inline_v<T>)
    return sizeof(holder_block<T>);
  return static_cast<Py_ssize_t>(sizeof(T) > sizeof(holder_block<T>)
                                     ? sizeof(T)
                                     : sizeof(holder_block<T>));
}
template <class T> constexpr Py_ssize_t storage_payload_align() noexcept {
  if constexpr (!native_inline_v<T>)
    return alignof(holder_block<T>);
  return static_cast<Py_ssize_t>(alignof(T) > alignof(holder_block<T>)
                                     ? alignof(T)
                                     : alignof(holder_block<T>));
}

template <class T> constexpr Py_ssize_t holder_data_offset() noexcept {
  return align_up(static_cast<Py_ssize_t>(sizeof(instance)),
                  storage_payload_align<T>());
}

template <class T, class Native = T>
constexpr Py_ssize_t native_data_offset() noexcept {
  if constexpr (!native_inline_v<Native>)
    return holder_data_offset<T>();
  constexpr auto alignment = alignof(Native) > storage_payload_align<T>()
                                 ? alignof(Native)
                                 : storage_payload_align<T>();
  return align_up(sizeof(instance), alignment);
}

template <class T>
PYB26_ALWAYS_INLINE holder_block<T> *block_of(PyObject *o,
                                              Py_ssize_t off) noexcept {
  return reinterpret_cast<holder_block<T> *>(reinterpret_cast<char *>(o) + off);
}

template <class T>
PYB26_ALWAYS_INLINE T *object_ptr(PyObject *o, unsigned char storage,
                                  Py_ssize_t off) noexcept {
  if (storage == st_inline) [[likely]]
    return reinterpret_cast<T *>(reinterpret_cast<char *>(o) + off);
  return block_of<T>(o, off)->ptr;
}

PyObject *alloc_holder_instance(type_record *tr) noexcept;

template <class T> PyObject *make_holder_instance(type_record *tr) noexcept;

template <class Native> void *allocate_native_storage() noexcept {
  if constexpr (alignof(Native) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    return ::operator new(sizeof(Native), std::align_val_t(alignof(Native)),
                          std::nothrow);
  else
    return ::operator new(sizeof(Native), std::nothrow);
}

template <class Native> void destroy_native_storage(void *allocation) noexcept {
  std::destroy_at(static_cast<Native *>(allocation));
  if constexpr (alignof(Native) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    ::operator delete(allocation, std::align_val_t(alignof(Native)));
  else
    ::operator delete(allocation);
}

template <class T, class Native>
void arm_boxed(instance *in, void *mem, Native *native,
               unsigned char identity = 0) noexcept {
  auto *block = ::new (mem) holder_block<T>();
  block->ptr = static_cast<T *>(native);
  ::new (&block->boxed)
      boxed_native_storage{native, &destroy_native_storage<Native>};
  in->storage = st_boxed;
  in->flags.set_lifecycle(inst_constructed | inst_owned | identity |
                          (in->flags.lifecycle() & inst_initializing));
}

template <class T, storage_kind K, class Src>
PYB26_ALWAYS_INLINE void arm_block(instance *in, void *mem, Src &&src,
                                   unsigned char identity = 0) noexcept {
  static_assert(K != st_boxed,
                "boxed storage requires its native allocation identity");
  const unsigned char initializing = in->flags.lifecycle() & inst_initializing;
  auto *b = ::new (mem) holder_block<T>();
  if constexpr (K == st_shared) {
    b->ptr = src.get();
    ::new (&b->shared) std::shared_ptr<T>(std::move(src));
  } else if constexpr (K == st_unique) {
    b->ptr = src.get();
    ::new (&b->unique) std::unique_ptr<T>(std::move(src));
  } else if constexpr (K == st_pointer) {
    b->ptr = src;
  } else {
    b->ptr = src;
  }
  in->storage = K;
  in->flags.set_lifecycle(inst_constructed | initializing | identity |
                          (K == st_reference ? 0 : inst_owned));
}

template <class T>
PyObject *make_reference(T *p, PyObject *parent, bool internal) noexcept {
  if (!p)
    return Py_NewRef(Py_None);
  type_record *tr = resolve<T>();
  if (!tr) [[unlikely]]
    return error_unregistered(typeid(T).name());
  PyObject *inst = make_holder_instance<T>(tr);
  if (!inst) [[unlikely]]
    return nullptr;
  auto *in = reinterpret_cast<instance *>(inst);
  void *mem = reinterpret_cast<char *>(inst) + tr->data_offset;
  arm_block<T, st_reference>(in, mem, p);
  if (internal && parent) {
    if (tr->lifetime_offset) {
      if (!attach_lifetime(inst, parent, *tr)) {
        Py_DECREF(inst);
        return nullptr;
      }
    } else {
      Py_INCREF(parent);
      reinterpret_cast<holder_block<T> *>(mem)->keepalive = parent;
    }
  }
  return inst;
}

template <class T> PyObject *make_owned_pointer(T *p) noexcept {
  if (!p)
    return Py_NewRef(Py_None);
  type_record *tr = resolve<T>();
  if (!tr) [[unlikely]] {
    delete p;
    return error_unregistered(typeid(T).name());
  }
  PyObject *inst = make_holder_instance<T>(tr);
  if (!inst) [[unlikely]] {
    delete p;
    return nullptr;
  }
  auto *in = reinterpret_cast<instance *>(inst);
  void *mem = reinterpret_cast<char *>(inst) + tr->data_offset;
  arm_block<T, st_pointer>(in, mem, p);
  return inst;
}

template <class T> PyObject *make_shared(std::shared_ptr<T> sp) noexcept {
  if constexpr (std::is_const_v<T>)
    return make_shared<std::remove_const_t<T>>(
        std::const_pointer_cast<std::remove_const_t<T>>(std::move(sp)));
  if (!sp)
    return Py_NewRef(Py_None);
  type_record *tr = resolve<T>();
  if (!tr) [[unlikely]]
    return error_unregistered(typeid(T).name());
  PyObject *inst = make_holder_instance<T>(tr);
  if (!inst) [[unlikely]]
    return nullptr;
  auto *in = reinterpret_cast<instance *>(inst);
  void *mem = reinterpret_cast<char *>(inst) + tr->data_offset;
  arm_block<T, st_shared>(in, mem, std::move(sp));
  return inst;
}

template <class T> PyObject *make_unique(std::unique_ptr<T> up) noexcept {
  if (!up)
    return Py_NewRef(Py_None);
  type_record *tr = resolve<T>();
  if (!tr) [[unlikely]]
    return error_unregistered(typeid(T).name());
  PyObject *inst = make_holder_instance<T>(tr);
  if (!inst) [[unlikely]]
    return nullptr;
  auto *in = reinterpret_cast<instance *>(inst);
  void *mem = reinterpret_cast<char *>(inst) + tr->data_offset;
  arm_block<T, st_unique>(in, mem, std::move(up));
  return inst;
}

template <class T>
PYB26_NOINLINE void destroy_holder(PyObject *self, unsigned char storage,
                                   Py_ssize_t off) noexcept {
  auto *b = block_of<T>(self, off);
  switch (storage) {
  case st_pointer:
    delete b->ptr;
    break;
  case st_boxed:
    b->boxed.destroy(b->boxed.allocation);
    break;
  case st_shared:
    b->shared.~shared_ptr();
    break;
  case st_unique:
    b->unique.~unique_ptr();
    break;
  case st_reference:
  default:
    break;
  }
  Py_XDECREF(b->keepalive);
  b->~holder_block<T>();
}

template <class T>
std::shared_ptr<T> shared_from_instance(PyObject *o) noexcept {
  using Bare = std::remove_const_t<T>;
  type_record *tr = resolve_for_instance<Bare>(o);
  if (!tr)
    return nullptr;
  PyTypeObject *ot = Py_TYPE(o);
  if (ot != tr->py_type && !PyType_IsSubtype(ot, tr->py_type))
    return nullptr;
  auto *in = reinterpret_cast<instance *>(o);
  const auto lifecycle = in->flags.lifecycle();
  if ((lifecycle & (inst_constructed | inst_initializing)) != inst_constructed)
      [[unlikely]]
    return nullptr;
  if (ot == tr->py_type && in->storage == st_shared &&
      !tr->requires_wrapper_owner) {
    std::shared_ptr<T> result;
    Py_BEGIN_CRITICAL_SECTION(o);
    result = block_of<Bare>(o, tr->data_offset)->shared;
    Py_END_CRITICAL_SECTION();
    return result;
  }
  T *p = ot == tr->py_type && !(lifecycle & inst_alias)
             ? object_ptr<Bare>(o, in->storage, tr->data_offset)
             : static_cast<Bare *>(cast_registered_base(o, tr));
  if (!p)
    return nullptr;
  return std::shared_ptr<T>(shared_python_owner(o), p);
}

template <class T> class const_shared_argument;

} // namespace pyb26::detail

namespace pyb26 {

namespace rv = detail::rv;
using detail::rv_policy;

template <typename T> struct type_caster<std::shared_ptr<T>> {
  static constexpr caster_effects effects{false, false, false, true};
  using native_lifetime_type = std::remove_const_t<T>;
  using native_lifetime_caster = type_caster<std::shared_ptr<T>>;
  using owner_implementation = type_caster;
  using borrowed_owner_implementation = type_caster;
  PyObject *alias_owner(PyObject *input) const noexcept { return input; }
  template <class A>
  using invocation_caster =
      std::conditional_t<std::is_same_v<A, const std::shared_ptr<T> &> &&
                             std::is_same_v<T, std::remove_cv_t<T>>,
                         detail::const_shared_argument<T>, type_caster>;
  static_assert(
      !caster_requires_lease_v<std::remove_cv_t<T>>,
      "borrowed array view requires a live lease and cannot be stored here");
  std::shared_ptr<T> sp;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (o == Py_None) {
      sp = nullptr;
      return true;
    }
    sp = detail::shared_from_instance<T>(o);
    return static_cast<bool>(sp);
  }
  template <class A> decltype(auto) as() {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(sp);
    else
      return std::shared_ptr<T>(std::move(sp));
  }
  static PyObject *cast(std::shared_ptr<T> v) {
    return detail::make_shared<T>(std::move(v));
  }
};

namespace detail {

template <class T> class const_shared_argument {
  type_caster<std::shared_ptr<T>> fallback_;
  const std::shared_ptr<T> *value_ = &fallback_.sp;

public:
  const_shared_argument() noexcept = default;
  using owner_implementation = const_shared_argument;
  using borrowed_owner_implementation = const_shared_argument;
  PyObject *alias_owner(PyObject *input) const noexcept { return input; }
  const_shared_argument(const const_shared_argument &) = delete;
  const_shared_argument &operator=(const const_shared_argument &) = delete;

  bool load(PyObject *object) noexcept {
    value_ = &fallback_.sp;
    if (object != Py_None) {
      auto *record = resolve_for_instance<T>(object);
      if (record && Py_TYPE(object) == record->py_type &&
          !record->requires_wrapper_owner) {
        auto *in = reinterpret_cast<instance *>(object);
        const auto lifecycle = in->flags.lifecycle();
        if ((lifecycle & (inst_constructed | inst_initializing | inst_alias)) ==
                inst_constructed &&
            in->storage == st_shared) {
          value_ = &block_of<T>(object, record->data_offset)->shared;
          return static_cast<bool>(*value_);
        }
      }
    }
    return fallback_.load(object);
  }

  template <class A> const std::shared_ptr<T> &as() noexcept {
    static_assert(std::is_same_v<A, const std::shared_ptr<T> &>);
    return *value_;
  }
};

} // namespace detail

template <typename T> struct type_caster<std::unique_ptr<T>> {
  static constexpr caster_effects effects{false, false, false};
  using native_lifetime_type = T;
  using native_lifetime_caster = type_caster<std::unique_ptr<T>>;
  static_assert(
      !caster_requires_lease_v<std::remove_cv_t<T>>,
      "borrowed array view requires a live lease and cannot be stored here");
  std::unique_ptr<T> up;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (o == Py_None) {
      up = nullptr;
      return true;
    }
    PyErr_SetString(PyExc_TypeError, "consuming a Python-held unique owner is "
                                     "unsupported; use shared ownership");
    return false;
  }
  template <class A> decltype(auto) as() {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(up);
    else
      return std::unique_ptr<T>(std::move(up));
  }
  static PyObject *cast(std::unique_ptr<T> v) {
    return detail::make_unique<T>(std::move(v));
  }
};

} // namespace pyb26
