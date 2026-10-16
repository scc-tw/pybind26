#pragma once

#include "caster.hpp"
#include "holders.hpp"

#include <atomic>
#include <memory>
#include <new>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace pyb26::detail {

struct constructor_error_scope {
  PyObject *error = nullptr;
  constructor_error_scope() noexcept = default;
  constructor_error_scope(const constructor_error_scope &) = delete;
  constructor_error_scope &operator=(const constructor_error_scope &) = delete;
  ~constructor_error_scope() {
    if (error)
      PyErr_SetRaisedException(error);
  }
  int fail() noexcept {
    if (!error)
      error = PyErr_GetRaisedException();
    return -1;
  }
};

template <class T, class Native = T>
PYB26_ALWAYS_INLINE void
destroy_inline_native(PyObject *self, Py_ssize_t off,
                      unsigned char lifecycle) noexcept {
  void *value = reinterpret_cast<char *>(self) + off;
  if constexpr (!std::is_same_v<T, Native>) {
    if (lifecycle & inst_alias) {
      static_cast<Native *>(value)->~Native();
      return;
    }
  }
  static_cast<T *>(value)->~T();
}

template <class T, class Native = T>
PYB26_COLD PYB26_NOINLINE int rollback_constructor(PyObject *self,
                                                   Py_ssize_t off) noexcept {
  constructor_error_scope error;
  error.fail();
  auto *in = reinterpret_cast<instance *>(self);
  const auto lifecycle = in->flags.lifecycle();
  in->flags &=
      static_cast<unsigned char>(~(inst_constructed | inst_owned | inst_alias));
  if (in->storage == st_inline) {
    destroy_inline_native<T, Native>(self, off, lifecycle);
  } else {
    destroy_holder<T>(self, in->storage, off);
    in->storage = st_inline;
  }
  return -1;
}

template <class T>
PYB26_ALWAYS_INLINE T *exact_instance_pointer(PyObject *o,
                                              type_record *tr) noexcept {
  auto *in = reinterpret_cast<instance *>(o);
  const auto lifecycle = in->flags.lifecycle();
  if ((lifecycle & (inst_constructed | inst_initializing)) != inst_constructed)
      [[unlikely]]
    return nullptr;
  if (in->storage == st_inline) [[likely]] {
    if (lifecycle & inst_alias) [[unlikely]]
      return static_cast<T *>(tr->native_pointer(o, tr->data_offset));
    return reinterpret_cast<T *>(reinterpret_cast<char *>(o) + tr->data_offset);
  }
  return block_of<T>(o, tr->data_offset)->ptr;
}

template <class T>
PYB26_HOT PYB26_ALWAYS_INLINE T *cast_to(PyObject *o) noexcept {
  PyTypeObject *type = Py_TYPE(o);
  type_cache<T> &cache = type_cache_of<T>;
  if (cache.state == type_state() && cache.py_type == type) [[likely]]
    return exact_instance_pointer<T>(o, cache.tr);
  type_record *record = resolve<T>();
  if (!record) [[unlikely]]
    return nullptr;
  if (type == record->py_type) [[likely]]
    return exact_instance_pointer<T>(o, record);
  if (!PyType_IsSubtype(type, record->py_type))
    return nullptr;
  auto *in = reinterpret_cast<instance *>(o);
  if ((in->flags.lifecycle() & (inst_constructed | inst_initializing)) !=
      inst_constructed) [[unlikely]]
    return nullptr;
  return static_cast<T *>(cast_registered_base(o, record));
}

struct fl_block_base {
  fl_block_base *thread_next = nullptr;
  bool orphaned = false;
};

template <class T> struct inst_freelist : fl_block_base {
  static constexpr int CAP = 32;
  PyObject *slots[CAP];
  std::atomic<PyTypeObject *> type{nullptr};
  std::atomic<int> n{0};
  inst_freelist *type_next = nullptr;
};

template <class T> PYB26_HIDDEN inline inst_freelist<T> *fl_all_of = nullptr;

template <class T>
PYB26_HIDDEN PYB26_TLS_MODEL inline thread_local inst_freelist<T>
    *inst_freelist_of = nullptr;

struct fl_thread_guard {
  fl_block_base *head = nullptr;
  ~fl_thread_guard() {
    if (!head)
      return;
    fl_lock();
    for (fl_block_base *b = head; b; b = b->thread_next)
      b->orphaned = true;
    fl_unlock();
  }
};
PYB26_HIDDEN PYB26_TLS_MODEL inline thread_local fl_thread_guard
    fl_thread_blocks{};

struct fl_guard {
  fl_guard() noexcept { fl_lock(); }
  ~fl_guard() { fl_unlock(); }
};

template <class T>
PYB26_NOINLINE inst_freelist<T> *fl_create(PyTypeObject *type) noexcept {
  auto *b = new (std::nothrow) inst_freelist<T>{};
  if (!b) [[unlikely]]
    return nullptr;
  b->type.store(type, std::memory_order_relaxed);
  {
    fl_guard g;
    b->type_next = fl_all_of<T>;
    fl_all_of<T> = b;
    b->thread_next = fl_thread_blocks.head;
    fl_thread_blocks.head = b;
  }
  inst_freelist_of<T> = b;
  return b;
}

template <class T> struct fl_retired_slots {
  PyTypeObject *type = nullptr;
  PyObject *slots[inst_freelist<T>::CAP];
  int n = 0;
  void detach(inst_freelist<T> &fl) noexcept {
    type = fl.type.load(std::memory_order_relaxed);
    n = fl.n.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i)
      slots[i] = fl.slots[i];
    fl.n.store(0, std::memory_order_relaxed);
    fl.type.store(nullptr, std::memory_order_relaxed);
  }
  ~fl_retired_slots() {
    for (int i = 0; i < n; ++i) {
      type->tp_free(slots[i]);
      Py_DECREF(type);
    }
  }
};

template <class T>
PYB26_NOINLINE void fl_rebind(inst_freelist<T> &fl,
                              PyTypeObject *type) noexcept {
  for (;;) {
    fl_retired_slots<T> retired;
    {
      fl_guard guard;
      if (fl.type.load(std::memory_order_relaxed) == type)
        return;
      if (!fl.n.load(std::memory_order_relaxed)) {
        fl.type.store(type, std::memory_order_relaxed);
        return;
      }
      retired.detach(fl);
    }
  }
}

template <class T> void drain_freelist(PyTypeObject *dying) noexcept {
  for (;;) {
    fl_retired_slots<T> retired;
    bool found = false;
    {
      fl_guard guard;
      inst_freelist<T> **link = &fl_all_of<T>;
      while (inst_freelist<T> *b = *link) {
        if (b->type.load(std::memory_order_relaxed) == dying) {
          retired.detach(*b);
          found = true;
        }
        if (b->orphaned && !b->type.load(std::memory_order_relaxed)) {
          *link = b->type_next;
          delete b;
        } else {
          link = &b->type_next;
        }
        if (found)
          break;
      }
    }
    if (!found)
      return;
  }
}

template <class T>
int traverse_freelists(PyTypeObject *type, visitproc visit,
                       void *arg) noexcept {
  Py_ssize_t count = 0;
  {
    fl_guard guard;
    for (auto *block = fl_all_of<T>; block; block = block->type_next)
      if (block->type.load(std::memory_order_relaxed) == type)
        count += block->n.load(std::memory_order_relaxed);
  }
  for (Py_ssize_t i = 0; i < count; ++i) {
    const int status = visit(reinterpret_cast<PyObject *>(type), arg);
    if (status)
      return status;
  }
  return 0;
}

template <class T>
PYB26_ALWAYS_INLINE PyObject *fl_alloc(type_record *tr) noexcept {
  if (!state_bypasses_freelist(type_state())) [[likely]] {
    inst_freelist<T> *fl = inst_freelist_of<T>;
    if (fl) [[likely]] {
      if (fl->type.load(std::memory_order_relaxed) == tr->py_type) [[likely]] {
        const int count = fl->n.load(std::memory_order_relaxed);
        if (count > 0) {
          PyObject *o = fl->slots[count - 1];
          fl->n.store(count - 1, std::memory_order_relaxed);
          cpython::revive_object(o);
          reinterpret_cast<instance *>(o)->flags = 0;
          reinterpret_cast<instance *>(o)->storage = st_inline;
          return o;
        }
      } else [[unlikely]] {
        fl_rebind<T>(*fl, tr->py_type);
      }
    } else [[unlikely]] {
      fl_create<T>(tr->py_type);
    }
  }
  PyObject *o = tr->py_type->tp_alloc(tr->py_type, 0);
  if (o) {
    std::construct_at(&reinterpret_cast<instance *>(o)->flags);
    reinterpret_cast<instance *>(o)->storage = st_inline;
  }
  return o;
}

template <class T> void inst_dealloc(PyObject *self) noexcept;

template <class T>
PYB26_ALWAYS_INLINE PyObject *make_holder_instance(type_record *tr) noexcept {
  if (!seal_type(tr))
    return nullptr;
  if (tr->py_type->tp_dealloc == &inst_dealloc<T>) [[likely]]
    return fl_alloc<T>(tr);
  return alloc_holder_instance(tr);
}

template <class T, class Construct>
PYB26_ALWAYS_INLINE void emplace_value(instance *in, void *mem,
                                       Construct &&construct) {
  construct(mem);
  in->flags |= inst_constructed | inst_owned;
}

template <class T, class Construct>
PYB26_COLD PyObject *make_instance_out_of_line(type_record *tr,
                                               Construct &&construct) {
  PyObject *inst = make_holder_instance<T>(tr);
  if (!inst)
    return nullptr;
  void *native = allocate_native_storage<T>();
  if (!native) {
    Py_DECREF(inst);
    return PyErr_NoMemory();
  }
  construct(native);
  arm_boxed<T>(reinterpret_cast<instance *>(inst),
               reinterpret_cast<char *>(inst) + tr->data_offset,
               static_cast<T *>(native));
  if (PyErr_Occurred()) {
    constructor_error_scope error;
    error.fail();
    Py_DECREF(inst);
    return nullptr;
  }
  return inst;
}

template <class T, class Construct>
PYB26_ALWAYS_INLINE PyObject *make_instance(Construct &&construct) {
  type_record *tr = resolve<T>();
  if (!tr) [[unlikely]]
    return error_unregistered(typeid(T).name());
  if constexpr (!native_inline_v<T>) {
    return make_instance_out_of_line<T>(tr, std::forward<Construct>(construct));
  } else {
    PyObject *inst = make_holder_instance<T>(tr);
    if (!inst) [[unlikely]]
      return nullptr;
    void *mem = reinterpret_cast<char *>(inst) + tr->data_offset;
    auto *in = reinterpret_cast<instance *>(inst);
    emplace_value<T>(in, mem, construct);
    if (PyErr_Occurred()) [[unlikely]] {
      constructor_error_scope error;
      error.fail();
      Py_DECREF(inst);
      return nullptr;
    }
    return inst;
  }
}

template <class T>
PyObject *inst_new(PyTypeObject *type, PyObject *, PyObject *) noexcept {
  type_record *tr = resolve_for_type<T>(type);
  if (tr && type != tr->py_type && !matches_registered_type(type, tr)) {
    PyErr_SetString(PyExc_TypeError,
                    "native allocation requires the matching class __new__");
    return nullptr;
  }
  if (tr && !seal_type(tr))
    return nullptr;
  if (tr && type == tr->py_type) [[likely]]
    return fl_alloc<T>(tr);
  PyObject *o = PyType_GenericAlloc(type, 0);
  if (o)
    std::construct_at(&reinterpret_cast<instance *>(o)->flags);
  return o;
}

template <class T> void inst_dealloc(PyObject *self) noexcept {
  auto *in = reinterpret_cast<instance *>(self);
  const Py_ssize_t off = holder_data_offset<T>();
  if (in->storage != st_inline) [[unlikely]] {
    destroy_holder<T>(self, in->storage, off);
  } else if (in->flags.lifecycle() & inst_constructed) {
    reinterpret_cast<T *>(reinterpret_cast<char *>(self) + off)->~T();
  }
  PyTypeObject *t = Py_TYPE(self);
  inst_freelist<T> *fl = inst_freelist_of<T>;
  if (fl && t == fl->type.load(std::memory_order_relaxed) &&
      !state_bypasses_freelist(type_state())) [[likely]] {
    const int count = fl->n.load(std::memory_order_relaxed);
    if (count < inst_freelist<T>::CAP) {
      fl->slots[count] = self;
      fl->n.store(count + 1, std::memory_order_relaxed);
      return;
    }
  }
  t->tp_free(self);
  Py_DECREF(t);
}

template <class T, class Native = T>
int inst_traverse(PyObject *self, visitproc visit, void *arg) noexcept {
  Py_VISIT(reinterpret_cast<PyObject *>(Py_TYPE(self)));
  auto *in = reinterpret_cast<instance *>(self);
  if (in->storage != st_inline)
    Py_VISIT(block_of<T>(self, native_data_offset<T, Native>())->keepalive);
  return traverse_lifetime(self, visit, arg);
}

template <class T, class Native = T> int inst_clear(PyObject *self) noexcept {
  auto *in = reinterpret_cast<instance *>(self);
  if (in->storage != st_inline) {
    auto *b = block_of<T>(self, native_data_offset<T, Native>());
    Py_CLEAR(b->keepalive);
  }
  clear_lifetime(self);
  return 0;
}

template <class T>
PyObject *inst_new_gc(PyTypeObject *type, PyObject *, PyObject *) noexcept {
  if (auto *record = resolve_for_type<T>(type)) {
    if (type != record->py_type && !matches_registered_type(type, record)) {
      PyErr_SetString(PyExc_TypeError,
                      "native allocation requires the matching class __new__");
      return nullptr;
    }
    if (!seal_type(record))
      return nullptr;
  }
  PyObject *o = PyType_GenericAlloc(type, 0);
  if (o)
    std::construct_at(&reinterpret_cast<instance *>(o)->flags);
  return o;
}

template <class T> void inst_dealloc_gc(PyObject *self) noexcept {
  PyObject_GC_UnTrack(self);
  auto *in = reinterpret_cast<instance *>(self);
  const Py_ssize_t off = holder_data_offset<T>();
  if (in->storage != st_inline) [[unlikely]] {
    destroy_holder<T>(self, in->storage, off);
  } else if (in->flags.lifecycle() & inst_constructed) {
    reinterpret_cast<T *>(reinterpret_cast<char *>(self) + off)->~T();
  }
  clear_lifetime(self);
  PyTypeObject *t = Py_TYPE(self);
  t->tp_free(self);
  Py_DECREF(t);
}

template <int Arity>
PYB26_COLD int normalize_ctor(PyObject *self, PyObject *args, PyObject *kwds,
                              Py_ssize_t off, const ctor_spec &spec) noexcept {
  [[indeterminate]] PyObject *slots[Arity];
  const int parsed = assemble_ctor_args(spec, self, args, kwds, slots);
  return parsed ? (parsed < 0 ? -1 : 2) : spec.invoke(self, slots, off, spec);
}

template <class T>
int inst_init(PyObject *self, PyObject *args, PyObject *kwds) noexcept {
  type_record *tr = resolve_for_instance<T>(self);
  if (!tr) [[unlikely]] {
    PyErr_SetString(PyExc_TypeError,
                    "pyb26: type not registered in this interpreter");
    return -1;
  }
  if (Py_TYPE(self) != tr->py_type &&
      !matches_registered_type(Py_TYPE(self), tr)) {
    PyErr_SetString(
        PyExc_TypeError,
        "native initialization requires the matching class __init__");
    return -1;
  }
  auto *in = reinterpret_cast<instance *>(self);
  const bool admitted = in->flags.try_begin(
      inst_constructed | inst_initializing, inst_initializing);
  if (!admitted) {
    PyErr_Format(PyExc_TypeError,
                 "%s.__init__(): an instance can be initialized only once",
                 tr->name.c_str());
    return -1;
  }
  struct init_admission {
    instance *value;
    ~init_admission() {
      value->flags &= static_cast<unsigned char>(~inst_initializing);
    }
  } admission{in};
  if (tr->nctors == 0) [[unlikely]] {
    PyErr_Format(PyExc_TypeError, "%s: no constructor", tr->name.c_str());
    return -1;
  }
  PyObject *mismatch = nullptr;
  const ctor_spec *shape_mismatch = nullptr;
  const bool positional = !kwds || !PyDict_GET_SIZE(kwds);
  const auto nargs = PyTuple_GET_SIZE(args);
  for (int i = 0; i < tr->nctors; ++i) {
    const ctor_spec &spec = tr->ctors[i];
    int r;
    if (spec.legacy) [[unlikely]] {
      r = kwds && PyDict_GET_SIZE(kwds)
              ? 1
              : spec.legacy(self, args, tr->data_offset);
    } else if (spec.complete_positional && nargs == spec.arity && positional) {
      r = spec.invoke(self, cpython::tuple_items(args), tr->data_offset, spec);
    } else if (!spec.argsinfo && positional && nargs != spec.arity) {
      shape_mismatch = &spec;
      Py_CLEAR(mismatch);
      continue;
    } else {
      [[indeterminate]] PyObject *stack[16];
      if (spec.arity > 16) {
        r = spec.normalize(self, args, kwds, tr->data_offset, spec);
      } else {
        const int parsed = assemble_ctor_args(spec, self, args, kwds, stack);
        r = parsed ? (parsed < 0 ? -1 : 2)
                   : spec.invoke(self, stack, tr->data_offset, spec);
      }
    }
    if (r == 0) {
      Py_XDECREF(mismatch);
      return 0;
    }
    if (r == 2) {
      shape_mismatch = nullptr;
      Py_XDECREF(mismatch);
      mismatch = PyErr_GetRaisedException();
      if (spec.errors == ctor_error_style::regular && kwds &&
          PyDict_GET_SIZE(kwds) && i + 1 < tr->nctors)
        return init_ctor_keyword_overloads(tr, i + 1, self, args, kwds,
                                           mismatch);
      continue;
    }
    if (r < 0 || PyErr_Occurred()) {
      Py_XDECREF(mismatch);
      return -1;
    }
  }
  if (shape_mismatch)
    return reject_ctor_shape(*shape_mismatch, self, args, kwds);
  if (mismatch) {
    PyErr_SetRaisedException(mismatch);
    return -1;
  }
  PyErr_Format(PyExc_TypeError, "%s: no constructor matches the arguments",
               tr->name.c_str());
  return -1;
}

} // namespace pyb26::detail
