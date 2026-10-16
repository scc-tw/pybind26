#pragma once

#include "allocation.hpp"
#include "cpython.hpp"
#include <Python.h>

#ifdef Py_LIMITED_API
#error                                                                         \
    "pyb26 requires the version-specific CPython API; abi3/abi3t are unsupported."
#endif

#if PY_VERSION_HEX < 0x030E0000
#error "pyb26 targets CPython 3.14 or newer (no 3.13 compatibility)."
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace pyb26 {

#if defined(__GNUC__) || defined(__clang__)
#define PYB26_HOT [[gnu::hot]]
#define PYB26_COLD [[gnu::cold]]
#define PYB26_FLATTEN [[gnu::flatten]]
#define PYB26_NOINLINE [[gnu::noinline]]
#define PYB26_ALWAYS_INLINE [[gnu::always_inline]] inline
#if defined(__ELF__)
#define PYB26_TLS_MODEL [[gnu::tls_model("global-dynamic")]]
#else
#define PYB26_TLS_MODEL [[gnu::tls_model("initial-exec")]]
#endif
#define PYB26_HIDDEN [[gnu::visibility("hidden")]]
#if defined(__has_attribute)
#if __has_attribute(no_stack_protector)
#define PYB26_NO_SSP [[gnu::no_stack_protector]]
#endif
#endif
#else
#define PYB26_HOT
#define PYB26_COLD
#define PYB26_FLATTEN
#define PYB26_NOINLINE
#define PYB26_ALWAYS_INLINE inline
#define PYB26_TLS_MODEL
#define PYB26_HIDDEN
#endif
#ifndef PYB26_NO_SSP
#define PYB26_NO_SSP
#endif

namespace detail {

struct arg_info {
  PyObject *name;
  PyObject *deflt;
  unsigned char flags;
};
enum arg_flag : unsigned char {
  arg_kw_only = 1,
  arg_pos_only = 2,
};

struct function_object;
struct overload_set {
  int count;
  int cap;
  function_object **entries;
  int *group_end;
};
using normalized_call = PyObject *(*)(PyObject *, PyObject *const *,
                                      PyObject *const *, int) noexcept;
using keyword_call = PyObject *(*)(PyObject *, PyObject *, int,
                                   PyObject *const *, size_t, PyObject *,
                                   int *) noexcept;
struct callable_ops {
  PyObject *(*make_sig)(const function_object *);
  vectorcallfunc overload_vc;
  keyword_call keywords;
  normalized_call normalized;
  unsigned char arg_base;
};

struct function_object {
  PyObject_HEAD vectorcallfunc vectorcall;
  vectorcallfunc try_call;
  void *capture;
  void (*free_capture)(void *);
  const char *name;
  const char *doc;
  arg_info *argsinfo;
  int arity;
  unsigned char kind;
  const callable_ops *callbacks;
  char *profile_label;
  bool variadic;
  bool owns_callbacks;
};
enum function_kind : unsigned char {
  fn_plain = 0,
  fn_functor = 1,
  fn_overload = 2
};

struct callable_spec {
  const char *name;
  const char *doc;
  int arity;
  vectorcallfunc vc;
  vectorcallfunc try_vc;
  void *capture = nullptr;
  void (*free_capture)(void *) = nullptr;
  arg_info *argsinfo = nullptr;
  PyObject *(*make_sig)(const function_object *) = nullptr;
  char *profile_label = nullptr;
  bool variadic = false;
  const callable_ops *callbacks = nullptr;
};

PyObject *overload_positional(PyObject *, PyObject *const *, size_t,
                              PyObject *) noexcept;
PYB26_COLD PyObject *overload_continue(PyObject *, PyObject *const *, size_t,
                                       PyObject *, int) noexcept;
PYB26_COLD PyObject *overload_keyword_miss(PyObject *, int, PyObject *const *,
                                           size_t, PyObject *,
                                           PyObject *const *, int,
                                           int *) noexcept;

extern PyObject *const overload_miss;

PYB26_HIDDEN PYB26_NOINLINE std::shared_ptr<void>
shared_python_owner(PyObject *) noexcept;

PYB26_HIDDEN bool current_frame_matches_function(PyThreadState *,
                                                 PyObject *) noexcept;

PYB26_HIDDEN PYB26_COLD PYB26_NOINLINE PyObject *
retain_python_error(PyObject *pending) noexcept;
PYB26_HIDDEN PYB26_COLD PYB26_NOINLINE void
discard_python_result(PyObject *result) noexcept;
PYB26_HIDDEN PYB26_COLD PYB26_NOINLINE PyObject *
finish_python_error(PyThreadState *state, PyObject *primary,
                    PyObject *result) noexcept;
PYB26_HIDDEN PYB26_COLD PYB26_NOINLINE void
discard_preserving_python_error(PyThreadState *state,
                                PyObject *result) noexcept;

struct type_registry;

struct module_state {
  PyTypeObject *function_type;
  PyTypeObject *method_type;
  PyTypeObject *bound_method_type;
  type_registry *types;
  PyTypeObject *static_metaclass;
  PyTypeObject *owned_getset_type;
};

enum instance_flag : unsigned char {
  inst_constructed = 1,
  inst_owned = 2,
  inst_keep_alive_list = 4,
  inst_has_lifetime_patients = 8,
  inst_initializing = 16,
  inst_alias = 32,
};

struct instance_flags {
  static constexpr unsigned char lifecycle_mask =
      inst_constructed | inst_owned | inst_initializing | inst_alias;
  static constexpr unsigned char patient_mask =
      inst_keep_alive_list | inst_has_lifetime_patients;
  mutable unsigned char lifecycle_value = 0;
  mutable unsigned char patient_value = 0;

  unsigned char lifecycle() const noexcept {
#ifdef Py_GIL_DISABLED
    return std::atomic_ref(lifecycle_value).load(std::memory_order_acquire);
#else
    return lifecycle_value;
#endif
  }
  unsigned char patients() const noexcept {
#ifdef Py_GIL_DISABLED
    return std::atomic_ref(patient_value).load(std::memory_order_acquire);
#else
    return patient_value;
#endif
  }
  void set_lifecycle(unsigned char flags) noexcept {
#ifdef Py_GIL_DISABLED
    std::atomic_ref(lifecycle_value).store(flags, std::memory_order_release);
#else
    lifecycle_value = flags;
#endif
  }
  operator unsigned char() const noexcept { return lifecycle() | patients(); }
  instance_flags &operator=(unsigned char flags) noexcept {
    set_lifecycle(flags & lifecycle_mask);
#ifdef Py_GIL_DISABLED
    std::atomic_ref(patient_value)
        .store(flags & patient_mask, std::memory_order_release);
#else
    patient_value = flags & patient_mask;
#endif
    return *this;
  }
  instance_flags &operator|=(unsigned char flags) noexcept {
    if (flags & lifecycle_mask)
      set_lifecycle(lifecycle() | (flags & lifecycle_mask));
    if (flags & patient_mask) {
#ifdef Py_GIL_DISABLED
      std::atomic_ref(patient_value)
          .fetch_or(flags & patient_mask, std::memory_order_release);
#else
      patient_value |= flags & patient_mask;
#endif
    }
    return *this;
  }
  instance_flags &operator&=(unsigned char flags) noexcept {
    if ((flags & lifecycle_mask) != lifecycle_mask)
      set_lifecycle(lifecycle() & flags);
    if ((flags & patient_mask) != patient_mask) {
#ifdef Py_GIL_DISABLED
      std::atomic_ref(patient_value)
          .fetch_and(flags & patient_mask, std::memory_order_release);
#else
      patient_value &= flags & patient_mask;
#endif
    }
    return *this;
  }
  bool try_begin(unsigned char blocked, unsigned char admission) noexcept {
#ifdef Py_GIL_DISABLED
    auto flags = lifecycle();
    do {
      if (flags & blocked)
        return false;
    } while (!std::atomic_ref(lifecycle_value)
                  .compare_exchange_weak(
                      flags, static_cast<unsigned char>(flags | admission),
                      std::memory_order_acq_rel, std::memory_order_acquire));
#else
    if (lifecycle_value & blocked)
      return false;
    lifecycle_value |= admission;
#endif
    return true;
  }
};
static_assert(sizeof(instance_flags) == 2 &&
              std::is_trivially_copyable_v<instance_flags>);

struct instance {
  PyObject_HEAD instance_flags flags;
  unsigned char storage;
};

enum storage_kind : unsigned char {
  st_inline = 0,
  st_reference = 1,
  st_pointer = 2,
  st_shared = 3,
  st_unique = 4,
  st_boxed = 5,
};

constexpr Py_ssize_t align_up(Py_ssize_t n, Py_ssize_t a) noexcept {
  return (n + a - 1) & ~(a - 1);
}

using ctor_fn = int (*)(PyObject *self, PyObject *args,
                        Py_ssize_t data_offset) noexcept;

enum class ctor_error_style : unsigned char { regular, cx };

struct ctor_spec;
using ctor_normalize_fn = int (*)(PyObject *, PyObject *, PyObject *,
                                  Py_ssize_t, const ctor_spec &) noexcept;
using ctor_vector_fn = int (*)(PyObject *, PyObject *const *, Py_ssize_t,
                               const ctor_spec &) noexcept;
struct ctor_spec {
  ctor_fn legacy = nullptr;
  int arity = 0;
  arg_info *argsinfo = nullptr;
  ctor_vector_fn invoke = nullptr;
  char *profile_label = nullptr;
  ctor_normalize_fn normalize = nullptr;
  bool complete_positional = true;
  ctor_error_style errors = ctor_error_style::regular;
};

struct property_context {
  const char *name = nullptr;
  char *profile_label = nullptr;
};
struct getset_spec {
  const char *name;
  const char *doc;
  getter get;
  setter set;
  void *closure = nullptr;
  void (*free_closure)(void *) noexcept = nullptr;
};

struct implicit_conv {
  bool (*gate)(PyObject *);
  implicit_conv *next;
};

struct type_record {
  const std::type_info *cpp_type = nullptr;
  PyTypeObject *py_type = nullptr;
  Py_ssize_t data_offset = 0;
  Py_ssize_t cpp_size = 0;
  ctor_spec *ctors = nullptr;
  int nctors = 0;
  int ctor_capacity = 0;
  Py_ssize_t lifetime_offset = 0;
  PyTypeObject *static_meta = nullptr;
  void (*drain)(PyTypeObject *) = nullptr;
  int (*traverse_freelists)(PyTypeObject *, visitproc,
                            void *) noexcept = nullptr;
  PyInterpreterState *interp = nullptr;
  type_record *next = nullptr;
  type_record **chain_head = nullptr;
  implicit_conv *convs = nullptr;
  std::atomic<unsigned char> phase{0};
  type_record *base = nullptr;
  void *(*upcast)(void *) noexcept = nullptr;
  void *(*native_pointer)(PyObject *, Py_ssize_t) noexcept = nullptr;
  bool requires_wrapper_owner = false;
  std::string name;
};

struct type_mutation {
  type_record *record = nullptr;
  type_mutation(const type_mutation &) = delete;
  type_mutation &operator=(const type_mutation &) = delete;
  explicit type_mutation(type_record *value) noexcept {
    if (!value || PyErr_Occurred())
      return;
    unsigned char expected = 0;
    if (value->phase.compare_exchange_strong(expected, 1,
                                             std::memory_order_acquire))
      record = value;
    else
      PyErr_SetString(PyExc_RuntimeError,
                      "native class metadata is frozen or being registered");
  }
  ~type_mutation() {
    if (record)
      record->phase.store(0, std::memory_order_release);
  }
  explicit operator bool() const noexcept { return record != nullptr; }
};

inline bool seal_type(type_record *record) noexcept {
  auto phase = record->phase.load(std::memory_order_acquire);
  if (phase == 2) [[likely]]
    return true;
  if (phase == 0 && record->phase.compare_exchange_strong(
                        phase, 2, std::memory_order_acq_rel))
    return true;
  PyErr_SetString(
      PyExc_RuntimeError,
      "native class construction overlaps registration or teardown");
  return false;
}

int register_type(type_record *&chain_head, type_record *tr) noexcept;
void unregister_type(type_record *&chain_head, type_record *tr) noexcept;
void destroy_type_record(type_record *tr) noexcept;
type_record *resolve_in_chain(type_record *const &chain_head,
                              PyInterpreterState *interp) noexcept;
void *cast_registered_base(PyObject *, type_record *) noexcept;
bool matches_registered_type(PyTypeObject *, type_record *) noexcept;
int finish_module(PyObject *) noexcept;

void fl_lock() noexcept;
void fl_unlock() noexcept;

PYB26_HIDDEN inline std::atomic<std::uint64_t> g_type_state{1u << 1};
PYB26_ALWAYS_INLINE std::uint64_t type_state() noexcept {
  return g_type_state.load(std::memory_order_acquire);
}
PYB26_ALWAYS_INLINE bool state_bypasses_freelist(std::uint64_t s) noexcept {
  return s & 1u;
}

template <typename T> PYB26_HIDDEN inline type_record *type_chain_of = nullptr;

template <typename T> struct type_cache {
  std::uint64_t state = 0;
  PyTypeObject *py_type = nullptr;
  type_record *tr = nullptr;
  PyInterpreterState *interp = nullptr;
};
template <typename T>
PYB26_HIDDEN PYB26_TLS_MODEL inline thread_local type_cache<T> type_cache_of{};

PYB26_ALWAYS_INLINE PyInterpreterState *current_interp() noexcept {
  return cpython::current_interpreter();
}

template <typename T>
PYB26_HOT PYB26_ALWAYS_INLINE type_record *
resolve(PyInterpreterState *cur) noexcept {
  type_cache<T> &c = type_cache_of<T>;
  std::uint64_t s = type_state();
  if (c.state == s && c.interp == cur) [[likely]]
    return c.tr;
  type_record *tr = resolve_in_chain(type_chain_of<T>, cur);
  if (tr) {
    c.interp = cur;
    c.py_type = tr->py_type;
    c.tr = tr;
    c.state = s;
  }
  return tr;
}

template <typename T>
PYB26_HOT PYB26_ALWAYS_INLINE type_record *resolve() noexcept {
  return resolve<T>(current_interp());
}

template <typename T>
PYB26_HOT PYB26_ALWAYS_INLINE type_record *
resolve_for_type(PyTypeObject *type) noexcept {
  type_cache<T> &cache = type_cache_of<T>;
  if (cache.state == type_state() && cache.py_type == type) [[likely]]
    return cache.tr;
  return resolve<T>();
}

template <typename T>
PYB26_HOT PYB26_ALWAYS_INLINE type_record *
resolve_for_instance(PyObject *object) noexcept {
  return resolve_for_type<T>(Py_TYPE(object));
}

int module_init(PyObject *module) noexcept;
int add_function(PyObject *module, const callable_spec &spec) noexcept;
int add_function(PyObject *module, PyObject *state_module,
                 const callable_spec &spec) noexcept;
PYB26_COLD int assemble_args(PyObject *self, PyObject *const *args,
                             Py_ssize_t nargs, PyObject *kwnames,
                             PyObject **out) noexcept;
PYB26_COLD int assemble_ctor_args(const ctor_spec &, PyObject *self,
                                  PyObject *args, PyObject *kwds,
                                  PyObject **out) noexcept;
PYB26_COLD int reject_ctor_shape(const ctor_spec &, PyObject *, PyObject *,
                                 PyObject *) noexcept;
PYB26_COLD int init_ctor_keyword_overloads(type_record *, int first,
                                           PyObject *self, PyObject *args,
                                           PyObject *kwds,
                                           PyObject *mismatch) noexcept;
int add_ctor(type_record *, const ctor_spec &,
             PyObject *signature = nullptr) noexcept;
PyObject **lifetime_slot(PyObject *) noexcept;
PYB26_HIDDEN PyObject **typed_lifetime_slot(PyObject *,
                                            type_record *const *) noexcept;
PyObject **lifetime_slot_gc(PyObject *) noexcept;
int set_lifetime_offset(PyTypeObject *, Py_ssize_t offset) noexcept;
PYB26_COLD int assemble_variadic_args(PyObject *self, PyObject *const *args,
                                      Py_ssize_t fixed_nargs, PyObject *kwnames,
                                      PyObject *const *keyword_values,
                                      PyObject **out) noexcept;
PYB26_COLD char *make_profile_label(PyObject *scope, const char *name) noexcept;
void clear_callable_spec(const callable_spec &spec) noexcept;

int module_traverse(PyObject *module, visitproc visit, void *arg) noexcept;
int module_clear(PyObject *module) noexcept;
void module_free(void *module) noexcept;

struct factory_only_t {};
enum class class_creation_policy : unsigned char { regular, factory_only };

PyTypeObject *create_class(
    PyObject *module, const char *name, const char *doc, Py_ssize_t basicsize,
    destructor dealloc, initproc init, newfunc tp_new, PyTypeObject *base,
    class_creation_policy policy = class_creation_policy::regular) noexcept;
PyTypeObject *create_class(
    PyObject *home, PyObject *state_module, const char *name, const char *doc,
    Py_ssize_t basicsize, destructor dealloc, initproc init, newfunc tp_new,
    PyTypeObject *base, traverseproc traverse = nullptr,
    inquiry clear = nullptr, const PyType_Slot *extra_slots = nullptr,
    int n_extra = 0,
    class_creation_policy policy = class_creation_policy::regular) noexcept;
int register_class(PyObject *home, PyObject *module, type_record *tr,
                   type_record *&chain_head) noexcept;
PyObject *alloc_instance(type_record *tr) noexcept;
PyObject *alloc_holder_instance(type_record *tr) noexcept;
int add_method(PyTypeObject *type, const callable_spec &spec) noexcept;
int add_static(PyTypeObject *type, const callable_spec &spec) noexcept;
int add_getset(PyTypeObject *type, const char *name, const char *doc, getter g,
               setter s) noexcept;
int add_getset(PyTypeObject *type, const getset_spec &) noexcept;
int add_static_getset(PyObject *module, PyTypeObject *type, const char *name,
                      getter g, setter s) noexcept;
int add_static_getset(PyObject *module, PyTypeObject *type,
                      const getset_spec &) noexcept;

PYB26_COLD PyObject *error_arity(PyObject *self, Py_ssize_t got_args,
                                 PyObject *kwnames) noexcept;
PYB26_COLD PyObject *error_arg_convert(PyObject *self) noexcept;
PYB26_COLD PyObject *error_unregistered(const char *cpp_name) noexcept;

} // namespace detail
} // namespace pyb26
