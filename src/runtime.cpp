#include "pyb26/detail/common.hpp"

#include <atomic>
#include <climits>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <structmember.h>
#include <unordered_map>

#ifndef Py_TPFLAGS_METHOD_DESCRIPTOR
#define Py_TPFLAGS_METHOD_DESCRIPTOR (1UL << 17)
#endif

namespace pyb26::detail {

static PyObject *make_owned_getset(PyTypeObject *, const char *, const char *,
                                   getter, setter, void *,
                                   void (*)(void *) noexcept, bool) noexcept;

alignas(PyObject) static unsigned char overload_miss_storage[sizeof(PyObject)];
PyObject *const overload_miss =
    reinterpret_cast<PyObject *>(overload_miss_storage);

PYB26_HIDDEN PYB26_COLD PYB26_NOINLINE PyObject *
retain_python_error(PyObject *pending) noexcept {
  return Py_NewRef(pending);
}

PYB26_HIDDEN PYB26_COLD PYB26_NOINLINE void
discard_python_result(PyObject *result) noexcept {
  if (result && result != overload_miss)
    Py_DECREF(result);
}

PYB26_HIDDEN PYB26_COLD PYB26_NOINLINE PyObject *
finish_python_error(PyThreadState *state, PyObject *primary,
                    PyObject *result) noexcept {
  if (result && result != overload_miss)
    Py_DECREF(result);
  if (cpython::current_exception(state) == primary)
    Py_DECREF(primary);
  else
    PyErr_SetRaisedException(primary);
  return nullptr;
}

PYB26_HIDDEN PYB26_COLD PYB26_NOINLINE void
discard_preserving_python_error(PyThreadState *state,
                                PyObject *result) noexcept {
  if (!result || result == overload_miss)
    return;
  PyObject *primary = Py_XNewRef(cpython::current_exception(state));
  Py_DECREF(result);
  if (primary) {
    if (cpython::current_exception(state) == primary)
      Py_DECREF(primary);
    else
      PyErr_SetRaisedException(primary);
  }
}

struct type_registry {
  std::unordered_map<const std::type_info *, type_record *> map;
  struct teardown_hook *hook = nullptr;
};

struct teardown_hook {
  module_state *state;
};

static void clear_type_registry(module_state *st) noexcept;

static std::mutex &chain_mutex() noexcept {
  static std::mutex m;
  return m;
}

static std::unordered_map<PyTypeObject *, type_record *> &
registered_types() noexcept {
  static std::unordered_map<PyTypeObject *, type_record *> records;
  return records;
}

static type_record *record_for_type(PyTypeObject *type) noexcept {
  std::lock_guard<std::mutex> lock(chain_mutex());
  for (; type; type = type->tp_base) {
    const auto it = registered_types().find(type);
    if (it != registered_types().end())
      return it->second;
  }
  return nullptr;
}

namespace {
constexpr char lifetime_attribute[] = "_pyb26_lifetime_layout";
constexpr const char *lifetime_capsule_name = "pyb26.lifetime_layout.v1";
struct lifetime_layout {
  Py_ssize_t offset;
};

struct borrowed_lifetime_layout {
  bool found = false;
  Py_ssize_t offset = 0;
};

borrowed_lifetime_layout read_lifetime_layout(PyObject *dict) noexcept {
  Py_ssize_t position = 0;
  PyObject *key, *value;
  while (PyDict_Next(dict, &position, &key, &value)) {
    if (!PyUnicode_Check(key) || !PyUnicode_IS_ASCII(key) ||
        PyUnicode_GET_LENGTH(key) !=
            static_cast<Py_ssize_t>(sizeof(lifetime_attribute) - 1) ||
        std::memcmp(PyUnicode_1BYTE_DATA(key), lifetime_attribute,
                    sizeof(lifetime_attribute) - 1) != 0)
      continue;
    if (!PyCapsule_IsValid(value, lifetime_capsule_name))
      return {true, 0};
    return {true, static_cast<lifetime_layout *>(
                      PyCapsule_GetPointer(value, lifetime_capsule_name))
                      ->offset};
  }
  return {};
}

PyObject **checked_lifetime_slot(PyObject *object, Py_ssize_t offset) noexcept {
  if (offset < static_cast<Py_ssize_t>(sizeof(instance)) ||
      offset > Py_TYPE(object)->tp_basicsize -
                   static_cast<Py_ssize_t>(sizeof(PyObject *)))
    return nullptr;
  return reinterpret_cast<PyObject **>(reinterpret_cast<char *>(object) +
                                       offset);
}

void destroy_lifetime_layout(PyObject *capsule) noexcept {
  delete static_cast<lifetime_layout *>(
      PyCapsule_GetPointer(capsule, lifetime_capsule_name));
}
} // namespace

int set_lifetime_offset(PyTypeObject *type, Py_ssize_t offset) noexcept {
  if (!PyType_IS_GC(type) ||
      offset < static_cast<Py_ssize_t>(sizeof(instance)) ||
      offset >
          type->tp_basicsize - static_cast<Py_ssize_t>(sizeof(PyObject *))) {
    PyErr_SetString(PyExc_TypeError, "invalid GC lifetime layout");
    return -1;
  }
  auto *layout = new (std::nothrow) lifetime_layout{offset};
  if (!layout) {
    PyErr_NoMemory();
    return -1;
  }
  PyObject *capsule =
      PyCapsule_New(layout, lifetime_capsule_name, &destroy_lifetime_layout);
  if (!capsule) {
    delete layout;
    return -1;
  }
  const int status =
      PyDict_SetItemString(type->tp_dict, lifetime_attribute, capsule);
  Py_DECREF(capsule);
  if (status == 0)
    PyType_Modified(type);
  return status;
}

PyObject **lifetime_slot(PyObject *object) noexcept {
  if (!PyObject_IS_GC(object))
    return nullptr;
  {
    std::lock_guard<std::mutex> lock(chain_mutex());
    for (PyTypeObject *type = Py_TYPE(object); type; type = type->tp_base) {
      const auto found = registered_types().find(type);
      if (found != registered_types().end()) {
        const Py_ssize_t offset = found->second->lifetime_offset;
        return offset ? checked_lifetime_slot(object, offset) : nullptr;
      }
    }
  }
  for (PyTypeObject *type = Py_TYPE(object); type; type = type->tp_base) {
    if (!type->tp_dict)
      continue;
    borrowed_lifetime_layout layout;
    Py_BEGIN_CRITICAL_SECTION(type->tp_dict);
    layout = read_lifetime_layout(type->tp_dict);
    Py_END_CRITICAL_SECTION();
    if (layout.found)
      return checked_lifetime_slot(object, layout.offset);
  }
  return nullptr;
}

PyObject **typed_lifetime_slot(PyObject *object,
                               type_record *const *chain) noexcept {
  if (!PyObject_IS_GC(object))
    return nullptr;
  {
    std::lock_guard<std::mutex> lock(chain_mutex());
    const PyTypeObject *actual_type = Py_TYPE(object);
    for (type_record *record = *chain; record; record = record->next) {
      if (record->py_type == actual_type) {
        const Py_ssize_t offset = record->lifetime_offset;
        return offset ? checked_lifetime_slot(object, offset) : nullptr;
      }
    }
    for (PyTypeObject *type = Py_TYPE(object); type; type = type->tp_base) {
      const auto found = registered_types().find(type);
      if (found != registered_types().end()) {
        const Py_ssize_t offset = found->second->lifetime_offset;
        return offset ? checked_lifetime_slot(object, offset) : nullptr;
      }
    }
  }
  for (PyTypeObject *type = Py_TYPE(object); type; type = type->tp_base) {
    if (!type->tp_dict)
      continue;
    borrowed_lifetime_layout layout;
    Py_BEGIN_CRITICAL_SECTION(type->tp_dict);
    layout = read_lifetime_layout(type->tp_dict);
    Py_END_CRITICAL_SECTION();
    if (layout.found)
      return checked_lifetime_slot(object, layout.offset);
  }
  return nullptr;
}

PyObject **lifetime_slot_gc(PyObject *object) noexcept {
  if (!PyObject_IS_GC(object))
    return nullptr;
  for (PyTypeObject *type = Py_TYPE(object); type; type = type->tp_base) {
    if (!type->tp_dict)
      continue;
    const auto layout = read_lifetime_layout(type->tp_dict);
    if (layout.found)
      return checked_lifetime_slot(object, layout.offset);
  }
  return nullptr;
}

static void free_arg_info(arg_info *info, int arity) noexcept {
  if (!info)
    return;
  for (int i = 0; i < arity; ++i) {
    Py_XDECREF(info[i].name);
    Py_XDECREF(info[i].deflt);
  }
  delete[] info;
}

static void clear_ctor_spec(const ctor_spec &spec,
                            PyObject *signature = nullptr) noexcept {
  PyObject *error = PyErr_GetRaisedException();
  free_arg_info(spec.argsinfo, spec.arity);
  delete[] spec.profile_label;
  Py_XDECREF(signature);
  PyErr_SetRaisedException(error);
}

int add_ctor(type_record *record, const ctor_spec &spec,
             PyObject *signature) noexcept {
  type_mutation mutation(record);
  if (!mutation) {
    clear_ctor_spec(spec, signature);
    return -1;
  }
  if (PyErr_Occurred()) {
    clear_ctor_spec(spec, signature);
    return -1;
  }
  if (record->nctors == record->ctor_capacity) {
    const int capacity = record->ctor_capacity ? record->ctor_capacity * 2 : 8;
    auto *entries =
        binding_new_array<allocation_site::constructor_entries, ctor_spec>(
            capacity);
    if (!entries) {
      PyErr_NoMemory();
      clear_ctor_spec(spec, signature);
      return -1;
    }
    for (int i = 0; i < record->nctors; ++i)
      entries[i] = record->ctors[i];
    delete[] record->ctors;
    record->ctors = entries;
    record->ctor_capacity = capacity;
  }
  if (record->nctors == 0 && signature &&
      PyObject_SetAttrString(reinterpret_cast<PyObject *>(record->py_type),
                             "__signature__", signature) < 0) {
    clear_ctor_spec(spec, signature);
    return -1;
  }
  Py_XDECREF(signature);
  record->ctors[record->nctors] = spec;
  auto &stored = record->ctors[record->nctors++];
  if (stored.complete_positional && stored.argsinfo) {
    for (int i = 0; i < stored.arity; ++i)
      if (stored.argsinfo[i].flags & arg_kw_only) {
        stored.complete_positional = false;
        break;
      }
  }
  return 0;
}

char *make_profile_label(PyObject *scope, const char *name) noexcept {
  PyObject *scope_name = PyObject_GetAttrString(scope, "__name__");
  if (!scope_name)
    return nullptr;
  const char *prefix = PyUnicode_AsUTF8(scope_name);
  char *label = nullptr;
  if (prefix) {
    const std::size_t prefix_size = std::strlen(prefix);
    const std::size_t name_size = std::strlen(name);
    label = new (std::nothrow) char[prefix_size + name_size + 2];
    if (label) {
      std::memcpy(label, prefix, prefix_size);
      label[prefix_size] = '.';
      std::memcpy(label + prefix_size + 1, name, name_size + 1);
    } else {
      PyErr_NoMemory();
    }
  }
  Py_DECREF(scope_name);
  return label;
}

static std::unordered_map<PyInterpreterState *, int> &interp_refs() noexcept {
  static std::unordered_map<PyInterpreterState *, int> m;
  return m;
}
static std::size_t active_registry_retirements = 0;
static void publish_state() noexcept {
  const bool bypass = interp_refs().size() > 1 || active_registry_retirements;
  std::uint64_t cur = g_type_state.load(std::memory_order_relaxed);
  std::uint64_t gen = (cur >> 1) + 1;
  g_type_state.store((gen << 1) | (bypass ? 1u : 0u),
                     std::memory_order_release);
}

int register_type(type_record *&chain_head, type_record *tr) noexcept {
  {
    std::lock_guard<std::mutex> g(chain_mutex());
    bool duplicate = false;
    for (auto *current = chain_head; current; current = current->next)
      duplicate = duplicate || current->interp == tr->interp;
    if (!duplicate) {
      auto [interpreter, inserted] = interp_refs().try_emplace(tr->interp, 0);
      if (!registered_types().emplace(tr->py_type, tr).second) {
        if (inserted)
          interp_refs().erase(interpreter);
        PyErr_SetString(PyExc_TypeError,
                        "Python type already has registration metadata");
        return -1;
      }
      ++interpreter->second;
      tr->chain_head = &chain_head;
      tr->next = chain_head;
      chain_head = tr;
      publish_state();
      return 0;
    }
  }
  PyErr_Format(PyExc_TypeError,
               "C++ type for '%s' is already registered in this interpreter",
               tr->name.c_str());
  return -1;
}

void unregister_type(type_record *&chain_head, type_record *tr) noexcept {
  std::lock_guard<std::mutex> g(chain_mutex());
  type_record **pp = &chain_head;
  bool found = false;
  while (*pp) {
    if (*pp == tr) {
      *pp = tr->next;
      tr->next = nullptr;
      found = true;
      break;
    }
    pp = &(*pp)->next;
  }
  if (found) {
    registered_types().erase(tr->py_type);
    tr->chain_head = nullptr;
    auto &m = interp_refs();
    auto it = m.find(tr->interp);
    if (it != m.end() && --it->second <= 0)
      m.erase(it);
  }
  publish_state();
}

void destroy_type_record(type_record *tr) noexcept {
  PyObject *error = PyErr_GetRaisedException();
  if (tr->chain_head)
    unregister_type(*tr->chain_head, tr);
  for (implicit_conv *c = tr->convs; c;) {
    implicit_conv *next = c->next;
    delete c;
    c = next;
  }
  for (int i = 0; i < tr->nctors; ++i)
    clear_ctor_spec(tr->ctors[i]);
  delete[] tr->ctors;
  Py_XDECREF(tr->py_type);
  delete tr;
  PyErr_SetRaisedException(error);
}

static void clear_types_at_exit(void *data) noexcept {
  auto *hk = static_cast<teardown_hook *>(data);
  module_state *st;
  {
    std::lock_guard<std::mutex> g(chain_mutex());
    st = hk->state;
    if (st)
      st->types->hook = nullptr;
  }
  if (st)
    clear_type_registry(st);
  delete hk;
}

static std::mutex &fl_registry_mutex() noexcept {
  static std::mutex m;
  return m;
}
void fl_lock() noexcept { fl_registry_mutex().lock(); }
void fl_unlock() noexcept { fl_registry_mutex().unlock(); }

type_record *resolve_in_chain(type_record *const &chain_head,
                              PyInterpreterState *interp) noexcept {
  std::lock_guard<std::mutex> g(chain_mutex());
  for (type_record *tr = chain_head; tr; tr = tr->next)
    if (tr->interp == interp)
      return tr;
  return nullptr;
}

bool matches_registered_type(PyTypeObject *type, type_record *wanted) noexcept {
  std::lock_guard<std::mutex> lock(chain_mutex());
  for (; type; type = type->tp_base) {
    const auto found = registered_types().find(type);
    if (found != registered_types().end())
      return found->second == wanted;
  }
  return false;
}

void *cast_registered_base(PyObject *object, type_record *wanted) noexcept {
  std::lock_guard<std::mutex> lock(chain_mutex());
  type_record *record = nullptr;
  for (auto *type = Py_TYPE(object); type; type = type->tp_base) {
    const auto found = registered_types().find(type);
    if (found != registered_types().end()) {
      record = found->second;
      break;
    }
  }
  if (!record || !record->native_pointer)
    return nullptr;
  void *pointer = record->native_pointer(object, record->data_offset);
  while (record != wanted) {
    if (!record->base || !record->upcast)
      return nullptr;
    pointer = record->upcast(pointer);
    record = record->base;
  }
  return pointer;
}

static bool matching_layout(const function_object *a,
                            const function_object *b) noexcept {
  if (!a->callbacks || !b->callbacks || !a->callbacks->keywords ||
      !b->callbacks->keywords || a->variadic || b->variadic ||
      a->arity != b->arity ||
      a->callbacks->arg_base != b->callbacks->arg_base ||
      (a->name != b->name &&
       (!a->name || !b->name || std::strcmp(a->name, b->name) != 0)))
    return false;
  if (!a->argsinfo || !b->argsinfo)
    return a->argsinfo == b->argsinfo;
  for (int i = 0; i < a->arity; ++i) {
    const auto &x = a->argsinfo[i];
    const auto &y = b->argsinfo[i];
    if (x.flags != y.flags || x.deflt != y.deflt)
      return false;
    if (x.name && !PyUnicode_CheckExact(x.name))
      return false;
    if (y.name && !PyUnicode_CheckExact(y.name))
      return false;
    if (x.name != y.name &&
        (!x.name || !y.name || PyUnicode_Compare(x.name, y.name) != 0))
      return false;
  }
  return true;
}

static void rebuild_layout_groups(overload_set *set) noexcept {
  for (int first = 0; first < set->count;) {
    int end = first + 1;
    while (end < set->count &&
           matching_layout(set->entries[first], set->entries[end]))
      ++end;
    for (int i = first; i < end; ++i)
      set->group_end[i] = end;
    first = end;
  }
}

PYB26_COLD PyObject *overload_keyword_miss(PyObject *owner, int index,
                                           PyObject *const *args, size_t nargsf,
                                           PyObject *kwnames,
                                           PyObject *const *slots, int status,
                                           int *next) noexcept {
  auto *set = static_cast<overload_set *>(
      reinterpret_cast<function_object *>(owner)->capture);
  const int end = set->group_end[index];
  int resume = index + 1;
  bool compatible = end > resume;
  if (compatible)
    for (Py_ssize_t k = 0; k < PyTuple_GET_SIZE(kwnames); ++k)
      if (!PyUnicode_CheckExact(PyTuple_GET_ITEM(kwnames, k))) {
        compatible = false;
        break;
      }
  if (compatible) {
    for (int i = resume; i < end; ++i) {
      auto *entry = set->entries[i];
      PyObject *result = entry->callbacks->normalized(
          reinterpret_cast<PyObject *>(entry), args, slots, status);
      if (result != overload_miss)
        return result;
    }
    resume = end;
  }
  if (next) {
    *next = resume;
    return overload_miss;
  }
  return overload_continue(owner, args, nargsf, kwnames, resume);
}

PYB26_COLD PyObject *overload_continue(PyObject *owner, PyObject *const *args,
                                       size_t nargsf, PyObject *kwnames,
                                       int first) noexcept {
  auto *function = reinterpret_cast<function_object *>(owner);
  auto *set = static_cast<overload_set *>(function->capture);
  for (int i = first; i < set->count;) {
    auto *entry = set->entries[i];
    PyObject *result;
    if (entry->callbacks && entry->callbacks->keywords) {
      int next;
      result =
          entry->callbacks->keywords(reinterpret_cast<PyObject *>(entry), owner,
                                     i, args, nargsf, kwnames, &next);
      if (result != overload_miss)
        return result;
      i = next;
    } else {
      result = entry->try_call(reinterpret_cast<PyObject *>(entry), args,
                               nargsf, kwnames);
      if (result != overload_miss)
        return result;
      ++i;
    }
  }
  PyErr_Format(PyExc_TypeError, "%s(): no overload matches the given arguments",
               function->name);
  return nullptr;
}

PyObject *overload_positional(PyObject *self, PyObject *const *args,
                              size_t nargsf, PyObject *kwnames) noexcept {
  auto *f = reinterpret_cast<function_object *>(self);
  auto *set = static_cast<overload_set *>(f->capture);
  for (int i = 0; i < set->count; ++i) {
    function_object *e = set->entries[i];
    PyObject *r =
        e->try_call(reinterpret_cast<PyObject *>(e), args, nargsf, kwnames);
    if (r != overload_miss)
      return r;
  }
  PyErr_Format(PyExc_TypeError, "%s(): no overload matches the given arguments",
               f->name);
  return nullptr;
}

static PyObject *overload_vectorcall(PyObject *self, PyObject *const *args,
                                     size_t nargsf,
                                     PyObject *kwnames) noexcept {
  if (kwnames && PyTuple_GET_SIZE(kwnames))
    return overload_continue(self, args, nargsf, kwnames, 0);
  return overload_positional(self, args, nargsf, kwnames);
}

static int function_traverse(PyObject *self, visitproc visit,
                             void *arg) noexcept {
  auto *f = reinterpret_cast<function_object *>(self);
  Py_VISIT(reinterpret_cast<PyObject *>(Py_TYPE(self)));
  if (f->kind == fn_overload && f->capture) {
    auto *set = static_cast<overload_set *>(f->capture);
    for (int i = 0; i < set->count; ++i)
      Py_VISIT(reinterpret_cast<PyObject *>(set->entries[i]));
  }
  if (f->argsinfo) {
    for (int i = 0; i < f->arity; ++i) {
      Py_VISIT(f->argsinfo[i].name);
      Py_VISIT(f->argsinfo[i].deflt);
    }
  }
  return 0;
}

static int function_clear(PyObject *self) noexcept {
  auto *f = reinterpret_cast<function_object *>(self);
  if (f->kind == fn_overload && f->capture) {
    auto *set = static_cast<overload_set *>(f->capture);
    f->capture = nullptr;
    for (int i = 0; i < set->count; ++i)
      Py_XDECREF(reinterpret_cast<PyObject *>(set->entries[i]));
    delete[] set->entries;
    delete[] set->group_end;
    delete set;
  }
  if (f->argsinfo)
    for (int i = 0; i < f->arity; ++i)
      Py_CLEAR(f->argsinfo[i].deflt);
  return 0;
}

static void function_dealloc(PyObject *self) noexcept {
  PyObject_GC_UnTrack(self);
  auto *f = reinterpret_cast<function_object *>(self);
  if (f->kind == fn_overload && f->capture) {
    auto *set = static_cast<overload_set *>(f->capture);
    for (int i = 0; i < set->count; ++i)
      Py_XDECREF(reinterpret_cast<PyObject *>(set->entries[i]));
    delete[] set->entries;
    delete[] set->group_end;
    delete set;
  } else if (f->free_capture && f->capture) {
    f->free_capture(f->capture);
  }
  if (f->argsinfo) {
    for (int i = 0; i < f->arity; ++i) {
      Py_XDECREF(f->argsinfo[i].name);
      Py_XDECREF(f->argsinfo[i].deflt);
    }
    delete[] f->argsinfo;
  }
  delete[] const_cast<char *>(f->name);
  delete[] f->profile_label;
  if (f->owns_callbacks)
    delete f->callbacks;
  PyTypeObject *t = Py_TYPE(self);
  t->tp_free(self);
  Py_DECREF(t);
}

static PyObject *function_repr(PyObject *self) noexcept {
  auto *f = reinterpret_cast<function_object *>(self);
  return PyUnicode_FromFormat("<pyb26 function %s at %p>",
                              f->name ? f->name : "?", static_cast<void *>(f));
}

static PyObject *function_get_doc(PyObject *self, void *) noexcept {
  auto *f = reinterpret_cast<function_object *>(self);
  if (f->doc)
    return PyUnicode_FromString(f->doc);
  Py_RETURN_NONE;
}

static PyObject *function_get_name(PyObject *self, void *) noexcept {
  auto *f = reinterpret_cast<function_object *>(self);
  return PyUnicode_FromString(f->name ? f->name : "");
}

static PyObject *function_get_text_signature(PyObject *self, void *) noexcept {
  auto *f = reinterpret_cast<function_object *>(self);
  std::string out = "(";
  bool kw_marker_emitted = false;
  bool capture_emitted = false;
  for (int i = 0; i < f->arity; ++i) {
    arg_info *ai = f->argsinfo ? &f->argsinfo[i] : nullptr;
    if (ai && (ai->flags & arg_kw_only) && !kw_marker_emitted) {
      out += f->variadic ? "*arg" + std::to_string(f->arity) + ", " : "*, ";
      capture_emitted = f->variadic;
      kw_marker_emitted = true;
    }
    if (ai && ai->name) {
      if (const char *n = PyUnicode_AsUTF8(ai->name))
        out += n;
      else {
        PyErr_Clear();
        out += "arg" + std::to_string(i);
      }
    } else {
      out += "arg" + std::to_string(i);
    }
    if (ai && ai->deflt) {
      out += "=";
      PyObject *r = PyObject_Repr(ai->deflt);
      const char *rs = r ? PyUnicode_AsUTF8(r) : nullptr;
      out += rs ? rs : "...";
      Py_XDECREF(r);
      PyErr_Clear();
    }
    if (i + 1 < f->arity)
      out += ", ";
  }
  if (f->variadic && !capture_emitted) {
    if (f->arity)
      out += ", ";
    out += "*arg" + std::to_string(f->arity);
  }
  out += ")";
  return PyUnicode_FromStringAndSize(out.data(),
                                     static_cast<Py_ssize_t>(out.size()));
}

static PyObject *function_get_signature(PyObject *self, void *) noexcept {
  auto *f = reinterpret_cast<function_object *>(self);
  if (!f->callbacks || !f->callbacks->make_sig) {
    PyErr_SetString(PyExc_AttributeError, "__signature__");
    return nullptr;
  }
  return f->callbacks->make_sig(f);
}

static PyMemberDef function_members[] = {
    {"__vectorcalloffset__", Py_T_PYSSIZET,
     static_cast<Py_ssize_t>(offsetof(function_object, vectorcall)),
     Py_READONLY, nullptr},
    {nullptr, 0, 0, 0, nullptr},
};

static PyGetSetDef function_getset[] = {
    {"__doc__", function_get_doc, nullptr, nullptr, nullptr},
    {"__text_signature__", function_get_text_signature, nullptr, nullptr,
     nullptr},
    {"__signature__", function_get_signature, nullptr, nullptr, nullptr},
    {"__name__", function_get_name, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

static PyType_Slot function_type_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(&function_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(&function_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(&function_clear)},
    {Py_tp_repr, reinterpret_cast<void *>(&function_repr)},
    {Py_tp_call, reinterpret_cast<void *>(&PyVectorcall_Call)},
    {Py_tp_members, reinterpret_cast<void *>(function_members)},
    {Py_tp_getset, reinterpret_cast<void *>(function_getset)},
    {0, nullptr},
};

static PyType_Spec function_type_spec = {
    "pyb26.function",
    static_cast<int>(sizeof(function_object)),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL | Py_TPFLAGS_HAVE_GC,
    function_type_slots,
};

struct bound_method_object {
  PyObject_HEAD vectorcallfunc vectorcall;
  PyObject *function;
  PyObject *receiver;
};

static int bound_method_traverse(PyObject *self, visitproc visit,
                                 void *arg) noexcept {
  auto *method = reinterpret_cast<bound_method_object *>(self);
  Py_VISIT(Py_TYPE(self));
  Py_VISIT(method->function);
  Py_VISIT(method->receiver);
  return 0;
}

static int bound_method_clear(PyObject *self) noexcept {
  auto *method = reinterpret_cast<bound_method_object *>(self);
  auto *function = std::exchange(method->function, nullptr);
  auto *receiver = std::exchange(method->receiver, nullptr);
  discard_preserving_python_error(cpython::thread_state(), function);
  discard_preserving_python_error(cpython::thread_state(), receiver);
  return 0;
}

static void bound_method_dealloc(PyObject *self) noexcept {
  PyObject_GC_UnTrack(self);
  PyObject_ClearWeakRefs(self);
  bound_method_clear(self);
  auto *type = Py_TYPE(self);
  type->tp_free(self);
  Py_DECREF(type);
}

PYB26_ALWAYS_INLINE PyObject *bound_method_invoke(PyObject *function,
                                                  PyObject *const *args,
                                                  size_t nargs,
                                                  PyObject *kwnames) noexcept {
  if (PyType_HasFeature(Py_TYPE(function), Py_TPFLAGS_HAVE_VECTORCALL))
      [[likely]]
    return reinterpret_cast<function_object *>(function)->vectorcall(
        function, args, nargs, kwnames);
  return PyObject_Vectorcall(function, args, nargs, kwnames);
}

PYB26_NOINLINE static PyObject *
bound_method_copy_call(bound_method_object *method, PyObject *const *args,
                       Py_ssize_t count, PyObject *kwnames) noexcept {
  const auto keywords = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  if (count > PY_SSIZE_T_MAX - keywords - 1) [[unlikely]]
    return PyErr_NoMemory();
  const auto total = static_cast<size_t>(count + keywords + 1);
  [[indeterminate]] PyObject *local[8];
  PyObject **prepended = local;
  if (total > std::size(local)) [[unlikely]] {
    if (total > SIZE_MAX / sizeof(PyObject *))
      return PyErr_NoMemory();
    prepended =
        static_cast<PyObject **>(PyMem_Malloc(total * sizeof(PyObject *)));
    if (!prepended)
      return PyErr_NoMemory();
  }
  prepended[0] = method->receiver;
  if (total > 1)
    std::memcpy(prepended + 1, args, (total - 1) * sizeof(PyObject *));
  auto *result = bound_method_invoke(method->function, prepended,
                                     static_cast<size_t>(count + 1), kwnames);
  if (prepended != local)
    PyMem_Free(prepended);
  return result;
}

static PyObject *bound_method_call(PyObject *self, PyObject *const *args,
                                   size_t nargsf, PyObject *kwnames) noexcept {
  auto *method = reinterpret_cast<bound_method_object *>(self);
  if (!method->function || !method->receiver) [[unlikely]] {
    PyErr_SetString(PyExc_RuntimeError, "bound method has been cleared");
    return nullptr;
  }
  const auto count = PyVectorcall_NARGS(nargsf);
  if (nargsf & PY_VECTORCALL_ARGUMENTS_OFFSET) [[likely]] {
    auto **prepended = const_cast<PyObject **>(args) - 1;
    auto *saved = prepended[0];
    prepended[0] = method->receiver;
    auto *result = bound_method_invoke(method->function, prepended,
                                       static_cast<size_t>(count + 1), kwnames);
    prepended[0] = saved;
    return result;
  }
  return bound_method_copy_call(method, args, count, kwnames);
}

static PyObject *bound_method_getattro(PyObject *self,
                                       PyObject *name) noexcept {
  auto *descriptor = _PyType_LookupRef(Py_TYPE(self), name);
  if (descriptor) {
    if (auto get = Py_TYPE(descriptor)->tp_descr_get) {
      auto *result =
          get(descriptor, self, reinterpret_cast<PyObject *>(Py_TYPE(self)));
      discard_preserving_python_error(cpython::thread_state(), descriptor);
      return result;
    }
    return descriptor;
  }
  auto *function = reinterpret_cast<bound_method_object *>(self)->function;
  if (function)
    return PyObject_GetAttr(function, name);
  PyErr_SetObject(PyExc_AttributeError, name);
  return nullptr;
}

static PyObject *bound_method_doc(PyObject *self, void *) noexcept {
  auto *function = reinterpret_cast<bound_method_object *>(self)->function;
  return function ? PyObject_GetAttrString(function, "__doc__")
                  : Py_NewRef(Py_None);
}

static PyObject *bound_method_signature(PyObject *self, void *) noexcept {
  auto *function = reinterpret_cast<bound_method_object *>(self)->function;
  if (!function) {
    PyErr_SetString(PyExc_AttributeError, "__signature__");
    return nullptr;
  }
  auto *signature = PyObject_GetAttrString(function, "__signature__");
  if (!signature)
    return nullptr;
  auto *parameters = PyObject_GetAttrString(signature, "parameters");
  auto *values = parameters ? PyMapping_Values(parameters) : nullptr;
  auto *remaining =
      values ? PyList_GetSlice(values, 1, PyList_GET_SIZE(values)) : nullptr;
  auto *replace =
      remaining ? PyObject_GetAttrString(signature, "replace") : nullptr;
  auto *names = replace ? Py_BuildValue("(s)", "parameters") : nullptr;
  PyObject *result = nullptr;
  if (names) {
    PyObject *arguments[] = {remaining};
    result = PyObject_Vectorcall(replace, arguments, 0, names);
  }
  auto *state = cpython::thread_state();
  for (auto *value : {names, replace, remaining, values, parameters, signature})
    discard_preserving_python_error(state, value);
  if (cpython::current_exception(state)) {
    discard_preserving_python_error(state, result);
    return nullptr;
  }
  return result;
}

static Py_hash_t bound_method_hash(PyObject *self) noexcept {
  auto *method = reinterpret_cast<bound_method_object *>(self);
  if (!method->function || !method->receiver) {
    PyErr_SetString(PyExc_RuntimeError, "bound method has been cleared");
    return -1;
  }
  const auto function_hash = PyObject_Hash(method->function);
  if (function_hash == -1)
    return -1;
  const auto result = function_hash ^ Py_HashPointer(method->receiver);
  return result == -1 ? -2 : result;
}

static PyObject *bound_method_compare(PyObject *left, PyObject *right,
                                      int operation) noexcept {
  if (Py_TYPE(left) != Py_TYPE(right) ||
      (operation != Py_EQ && operation != Py_NE))
    Py_RETURN_NOTIMPLEMENTED;
  auto *a = reinterpret_cast<bound_method_object *>(left);
  auto *b = reinterpret_cast<bound_method_object *>(right);
  if (!a->function || !b->function) {
    const bool equal = left == right;
    return PyBool_FromLong(operation == Py_EQ ? equal : !equal);
  }
  const int functions_equal =
      PyObject_RichCompareBool(a->function, b->function, Py_EQ);
  if (functions_equal < 0)
    return nullptr;
  const bool equal = functions_equal && a->receiver == b->receiver;
  return PyBool_FromLong(operation == Py_EQ ? equal : !equal);
}

static PyObject *bound_method_repr(PyObject *self) noexcept {
  auto *method = reinterpret_cast<bound_method_object *>(self);
  if (!method->function || !method->receiver)
    return PyUnicode_FromString("<cleared pyb26.bound_method>");
  auto *name = PyObject_GetAttrString(method->function, "__name__");
  if (!name)
    return nullptr;
  auto *result =
      PyUnicode_FromFormat("<bound method %S of %R>", name, method->receiver);
  discard_preserving_python_error(cpython::thread_state(), name);
  return result;
}

static PyObject *bound_method_reduce(PyObject *self, PyObject *) noexcept {
  auto *method = reinterpret_cast<bound_method_object *>(self);
  if (!method->function || !method->receiver) {
    PyErr_SetString(PyExc_RuntimeError, "bound method has been cleared");
    return nullptr;
  }
  auto *name = PyObject_GetAttrString(method->function, "__name__");
  if (!name)
    return nullptr;
  auto *getattr = PyDict_GetItemString(PyEval_GetBuiltins(), "getattr");
  PyObject *result = nullptr;
  if (getattr)
    result = Py_BuildValue("O(OO)", getattr, method->receiver, name);
  else
    PyErr_SetString(PyExc_RuntimeError, "builtins.getattr is unavailable");
  discard_preserving_python_error(cpython::thread_state(), name);
  return result;
}

static PyMethodDef bound_method_methods[] = {
    {"__reduce__", bound_method_reduce, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

static PyObject *bound_method_descr_get(PyObject *self, PyObject *,
                                        PyObject *) noexcept {
  return Py_NewRef(self);
}

static PyMemberDef bound_method_members[] = {
    {"__vectorcalloffset__", Py_T_PYSSIZET,
     offsetof(bound_method_object, vectorcall), Py_READONLY, nullptr},
    {"__func__", Py_T_OBJECT_EX, offsetof(bound_method_object, function),
     Py_READONLY, nullptr},
    {"__self__", Py_T_OBJECT_EX, offsetof(bound_method_object, receiver),
     Py_READONLY, nullptr},
    {nullptr, 0, 0, 0, nullptr},
};
static PyGetSetDef bound_method_getset[] = {
    {"__doc__", bound_method_doc, nullptr, nullptr, nullptr},
    {"__signature__", bound_method_signature, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};
static PyType_Slot bound_method_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(&bound_method_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(&bound_method_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(&bound_method_clear)},
    {Py_tp_call, reinterpret_cast<void *>(&PyVectorcall_Call)},
    {Py_tp_getattro, reinterpret_cast<void *>(&bound_method_getattro)},
    {Py_tp_descr_get, reinterpret_cast<void *>(&bound_method_descr_get)},
    {Py_tp_repr, reinterpret_cast<void *>(&bound_method_repr)},
    {Py_tp_hash, reinterpret_cast<void *>(&bound_method_hash)},
    {Py_tp_richcompare, reinterpret_cast<void *>(&bound_method_compare)},
    {Py_tp_members, reinterpret_cast<void *>(bound_method_members)},
    {Py_tp_methods, reinterpret_cast<void *>(bound_method_methods)},
    {Py_tp_getset, reinterpret_cast<void *>(bound_method_getset)},
    {0, nullptr},
};
static PyType_Spec bound_method_spec = {
    "pyb26.bound_method",
    static_cast<int>(sizeof(bound_method_object)),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL | Py_TPFLAGS_HAVE_GC |
        Py_TPFLAGS_MANAGED_WEAKREF | Py_TPFLAGS_IMMUTABLETYPE |
        Py_TPFLAGS_DISALLOW_INSTANTIATION,
    bound_method_slots,
};

static PyObject *method_descr_get(PyObject *func, PyObject *obj,
                                  PyObject *) noexcept {
  if (obj == nullptr || obj == Py_None)
    return Py_NewRef(func);
  auto *state =
      static_cast<module_state *>(PyType_GetModuleState(Py_TYPE(func)));
  if (!state)
    return nullptr;
  auto *type = state->bound_method_type;
  if (!type) {
    PyErr_SetString(PyExc_RuntimeError, "method module has been cleared");
    return nullptr;
  }
  auto *method =
      reinterpret_cast<bound_method_object *>(type->tp_alloc(type, 0));
  if (!method)
    return nullptr;
  method->vectorcall = bound_method_call;
  method->function = Py_NewRef(func);
  method->receiver = Py_NewRef(obj);
  return reinterpret_cast<PyObject *>(method);
}

static PyType_Slot method_type_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(&function_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(&function_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(&function_clear)},
    {Py_tp_repr, reinterpret_cast<void *>(&function_repr)},
    {Py_tp_call, reinterpret_cast<void *>(&PyVectorcall_Call)},
    {Py_tp_descr_get, reinterpret_cast<void *>(&method_descr_get)},
    {Py_tp_members, reinterpret_cast<void *>(function_members)},
    {Py_tp_getset, reinterpret_cast<void *>(function_getset)},
    {0, nullptr},
};

static PyType_Spec method_type_spec = {
    "pyb26.method",
    static_cast<int>(sizeof(function_object)),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL |
        Py_TPFLAGS_METHOD_DESCRIPTOR | Py_TPFLAGS_HAVE_GC,
    method_type_slots,
};

int module_init(PyObject *module) noexcept {
  auto *st = static_cast<module_state *>(PyModule_GetState(module));
  if (!st)
    return -1;
  PyObject *f = PyType_FromModuleAndSpec(module, &function_type_spec, nullptr);
  if (!f)
    return -1;
  st->function_type = reinterpret_cast<PyTypeObject *>(f);
  PyObject *m = PyType_FromModuleAndSpec(module, &method_type_spec, nullptr);
  if (!m)
    return -1;
  st->method_type = reinterpret_cast<PyTypeObject *>(m);
  auto *bound = PyType_FromModuleAndSpec(module, &bound_method_spec, nullptr);
  if (!bound)
    return -1;
  st->bound_method_type = reinterpret_cast<PyTypeObject *>(bound);
  cpython::prepare_shared_object(f);
  cpython::prepare_shared_object(m);
  return 0;
}

int module_traverse(PyObject *module, visitproc visit, void *arg) noexcept {
  auto *st = static_cast<module_state *>(PyModule_GetState(module));
  if (st) {
    Py_VISIT(st->function_type);
    Py_VISIT(st->method_type);
    Py_VISIT(st->bound_method_type);
    Py_VISIT(st->static_metaclass);
    Py_VISIT(st->owned_getset_type);
    if (st->types) {
      for (auto &[ti, tr] : st->types->map) {
        Py_VISIT(tr->py_type);
        if (tr->traverse_freelists) {
          const int status = tr->traverse_freelists(tr->py_type, visit, arg);
          if (status)
            return status;
        }
        for (int i = 0; i < tr->nctors; ++i) {
          const auto &ctor = tr->ctors[i];
          if (!ctor.argsinfo)
            continue;
          for (int j = 0; j < ctor.arity; ++j) {
            Py_VISIT(ctor.argsinfo[j].name);
            Py_VISIT(ctor.argsinfo[j].deflt);
          }
        }
      }
    }
  }
  return 0;
}
static void clear_type_registry(module_state *st) noexcept {
  if (!st || !st->types)
    return;
  type_registry *reg = st->types;
  st->types = nullptr;
  {
    std::lock_guard<std::mutex> g(chain_mutex());
    ++active_registry_retirements;
    publish_state();
    if (reg->hook) {
      reg->hook->state = nullptr;
      reg->hook = nullptr;
    }
  }
  for (auto &[ti, tr] : reg->map) {
    tr->phase.store(3, std::memory_order_release);
    if (tr->chain_head)
      unregister_type(*tr->chain_head, tr);
  }
  for (auto &[ti, tr] : reg->map) {
    if (tr->drain)
      tr->drain(tr->py_type);
    destroy_type_record(tr);
  }
  delete reg;
  {
    std::lock_guard<std::mutex> guard(chain_mutex());
    --active_registry_retirements;
    publish_state();
  }
}

int finish_module(PyObject *module) noexcept {
  auto *state = static_cast<module_state *>(PyModule_GetState(module));
  if (state && state->types)
    for (auto &[key, record] : state->types->map)
      if (!seal_type(record))
        return -1;
  return 0;
}
int module_clear(PyObject *module) noexcept {
  auto *st = static_cast<module_state *>(PyModule_GetState(module));
  if (st) {
    clear_type_registry(st);
    Py_CLEAR(st->function_type);
    Py_CLEAR(st->method_type);
    Py_CLEAR(st->bound_method_type);
    Py_CLEAR(st->static_metaclass);
    Py_CLEAR(st->owned_getset_type);
  }
  return 0;
}
void module_free(void *module) noexcept {
  module_clear(static_cast<PyObject *>(module));
}

void clear_callable_spec(const callable_spec &spec) noexcept {
  PyObject *error = PyErr_GetRaisedException();
  if (spec.free_capture && spec.capture)
    spec.free_capture(spec.capture);
  free_arg_info(spec.argsinfo, spec.arity);
  delete[] spec.profile_label;
  PyErr_SetRaisedException(error);
}

static function_object *make_callable(PyTypeObject *ctype,
                                      const callable_spec &s) noexcept {
  auto *f = reinterpret_cast<function_object *>(ctype->tp_alloc(ctype, 0));
  if (!f) {
    clear_callable_spec(s);
    return nullptr;
  }
  f->vectorcall = s.vc;
  f->try_call = s.try_vc;
  f->capture = s.capture;
  f->free_capture = s.free_capture;
  f->name = nullptr;
  f->doc = s.doc;
  f->argsinfo = s.argsinfo;
  f->arity = s.arity;
  f->callbacks = s.callbacks;
  f->owns_callbacks = false;
  f->profile_label = s.profile_label;
  f->variadic = s.variadic;
  f->kind = s.capture ? fn_functor : fn_plain;
  if ((!s.callbacks && s.make_sig) ||
      (s.callbacks && s.make_sig != s.callbacks->make_sig)) {
    auto *callbacks =
        binding_new<allocation_site::callable_callbacks, callable_ops>();
    if (!callbacks) {
      Py_DECREF(f);
      PyErr_NoMemory();
      return nullptr;
    }
    if (s.callbacks)
      *callbacks = *s.callbacks;
    callbacks->make_sig = s.make_sig;
    f->callbacks = callbacks;
    f->owns_callbacks = true;
  }
  if (s.name) {
    const std::size_t length = std::strlen(s.name) + 1;
    auto *owned_name = new (std::nothrow) char[length];
    if (!owned_name) {
      Py_DECREF(f);
      PyErr_NoMemory();
      return nullptr;
    }
    std::memcpy(owned_name, s.name, length);
    f->name = owned_name;
  }
  cpython::prepare_shared_object(reinterpret_cast<PyObject *>(f));
  return f;
}

static PyObject *combine_overload(function_object *existing,
                                  function_object *entry,
                                  PyTypeObject *otype) noexcept {
  if (existing->kind == fn_overload) {
    auto *set = static_cast<overload_set *>(existing->capture);
    if (set->count == set->cap) {
      int ncap = set->cap * 2;
      auto **ne = new (std::nothrow) function_object *[ncap];
      auto *ng = new (std::nothrow) int[ncap];
      if (!ne || !ng) {
        delete[] ne;
        delete[] ng;
        Py_DECREF(entry);
        PyErr_NoMemory();
        return nullptr;
      }
      for (int i = 0; i < set->count; ++i)
        ne[i] = set->entries[i];
      delete[] set->entries;
      delete[] set->group_end;
      set->entries = ne;
      set->group_end = ng;
      set->cap = ncap;
    }
    set->entries[set->count++] = entry;
    rebuild_layout_groups(set);
    return Py_NewRef(reinterpret_cast<PyObject *>(existing));
  }
  auto *set = new (std::nothrow) overload_set{};
  if (!set) {
    Py_DECREF(entry);
    return nullptr;
  }
  set->cap = 4;
  set->entries = new (std::nothrow) function_object *[set->cap];
  set->group_end = new (std::nothrow) int[set->cap];
  if (!set->entries || !set->group_end) {
    delete[] set->entries;
    delete[] set->group_end;
    delete set;
    Py_DECREF(entry);
    PyErr_NoMemory();
    return nullptr;
  }
  set->count = 0;
  set->entries[set->count++] = reinterpret_cast<function_object *>(
      Py_NewRef(reinterpret_cast<PyObject *>(existing)));
  set->entries[set->count++] = entry;
  rebuild_layout_groups(set);
  const vectorcallfunc vc =
      existing->callbacks && existing->callbacks->overload_vc
          ? existing->callbacks->overload_vc
          : &overload_vectorcall;
  callable_spec os{existing->name, existing->doc, existing->arity, vc, vc, set,
                   nullptr,        nullptr};
  function_object *ov = make_callable(otype, os);
  if (ov)
    ov->kind = fn_overload;
  if (!ov) {
    Py_DECREF(reinterpret_cast<PyObject *>(set->entries[0]));
    Py_DECREF(entry);
    delete[] set->entries;
    delete[] set->group_end;
    delete set;
    return nullptr;
  }
  return reinterpret_cast<PyObject *>(ov);
}

int add_function(PyObject *module, const callable_spec &spec) noexcept {
  return add_function(module, module, spec);
}

int add_function(PyObject *module, PyObject *state_module,
                 const callable_spec &spec) noexcept {
  if (PyErr_Occurred()) {
    clear_callable_spec(spec);
    return -1;
  }
  auto *st = static_cast<module_state *>(PyModule_GetState(state_module));
  if (!st || !st->function_type) {
    clear_callable_spec(spec);
    return -1;
  }
  auto *entry = make_callable(st->function_type, spec);
  if (!entry)
    return -1;

  PyObject *dict = PyModule_GetDict(module);
  PyObject *existing = nullptr;
  if (dict)
    PyDict_GetItemStringRef(dict, spec.name, &existing);
  if (existing && Py_IS_TYPE(existing, st->function_type)) {
    PyObject *bound =
        combine_overload(reinterpret_cast<function_object *>(existing), entry,
                         st->function_type);
    Py_DECREF(existing);
    if (!bound)
      return -1;
    int rc = PyModule_AddObjectRef(module, spec.name, bound);
    Py_DECREF(bound);
    return rc;
  }
  Py_XDECREF(existing);
  int rc = PyModule_AddObjectRef(module, spec.name,
                                 reinterpret_cast<PyObject *>(entry));
  Py_DECREF(entry);
  return rc;
}

namespace {

inline int argument_mismatch() noexcept {
  return PyErr_ExceptionMatches(PyExc_TypeError) ? 1 : -1;
}

template <bool ClearUnbound = true>
PYB26_ALWAYS_INLINE int bind_positional(const char *name, int arity,
                                        arg_info *info, PyObject *const *args,
                                        Py_ssize_t nargs,
                                        PyObject **out) noexcept {
  if (nargs > arity) {
    PyErr_Format(PyExc_TypeError,
                 "%s() takes at most %d positional argument%s (%zd given)",
                 name, arity, arity == 1 ? "" : "s", nargs);
    return argument_mismatch();
  }
  if constexpr (ClearUnbound) {
    for (int i = 0; i < arity; ++i)
      out[i] = nullptr;
  }
  for (Py_ssize_t i = 0; i < nargs; ++i) {
    if (info && (info[i].flags & arg_kw_only)) {
      if constexpr (!ClearUnbound) {
        for (Py_ssize_t j = i; j < arity; ++j)
          out[j] = nullptr;
      }
      if (info[i].name)
        PyErr_Format(PyExc_TypeError, "%s(): argument '%U' is keyword-only",
                     name, info[i].name);
      else
        PyErr_Format(PyExc_TypeError, "%s(): argument %zd is keyword-only",
                     name, i);
      return argument_mismatch();
    }
    out[i] = args[i];
  }
  return 0;
}

PYB26_ALWAYS_INLINE int bind_keyword(const char *name, int arity,
                                     arg_info *info, PyObject *key,
                                     PyObject *value, PyObject **out) noexcept {
  int index = -1;
  for (int i = 0; i < arity; ++i) {
    if (info[i].name == key) {
      index = i;
      break;
    }
  }
  if (index < 0) {
    for (int i = 0; i < arity; ++i) {
      if (!info[i].name)
        continue;
      const int equal = PyObject_RichCompareBool(info[i].name, key, Py_EQ);
      if (equal < 0)
        return -1;
      if (equal) {
        index = i;
        break;
      }
    }
  }
  if (index < 0) {
    PyErr_Format(PyExc_TypeError,
                 "%s() got an unexpected keyword argument '%U'", name, key);
    return argument_mismatch();
  }
  if (info[index].flags & arg_pos_only) {
    PyErr_Format(PyExc_TypeError, "%s(): argument '%U' is positional-only",
                 name, key);
    return argument_mismatch();
  }
  if (out[index]) {
    PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument '%U'",
                 name, key);
    return argument_mismatch();
  }
  out[index] = value;
  return 0;
}

template <bool AllMissing = false>
PYB26_ALWAYS_INLINE int bind_defaults(const char *name, int arity,
                                      arg_info *info, PyObject **out,
                                      int first = 0) noexcept {
  for (int i = first; i < arity; ++i) {
    if constexpr (!AllMissing) {
      if (out[i])
        continue;
    }
    if (info && info[i].deflt) {
      out[i] = info[i].deflt;
    } else {
      if constexpr (AllMissing) {
        for (int j = i; j < arity; ++j)
          out[j] = nullptr;
      }
      if (info && info[i].name)
        PyErr_Format(PyExc_TypeError, "%s() missing required argument '%U'",
                     name, info[i].name);
      else
        PyErr_Format(PyExc_TypeError,
                     "%s() missing required argument at position %d", name, i);
      return argument_mismatch();
    }
  }
  return 0;
}

PYB26_ALWAYS_INLINE int bind_without_keywords(function_object *function,
                                              PyObject *const *args,
                                              Py_ssize_t nargs,
                                              PyObject **out) noexcept {
  const int status = bind_positional<false>(
      function->name, function->arity, function->argsinfo, args, nargs, out);
  if (status)
    return status;
  const int first_missing = nargs > 0 ? static_cast<int>(nargs) : 0;
  return bind_defaults<true>(function->name, function->arity,
                             function->argsinfo, out, first_missing);
}

PYB26_ALWAYS_INLINE bool bind_identity_keyword(function_object *function,
                                               Py_ssize_t nargs, Py_ssize_t nkw,
                                               PyObject *kwnames,
                                               PyObject *const *values,
                                               PyObject **out) noexcept {
  if (function->arity != 1 || nargs != 0 || nkw != 1)
    return false;
  const auto *info = function->argsinfo;
  if (!info || (info[0].flags & arg_pos_only) ||
      info[0].name != PyTuple_GET_ITEM(kwnames, 0))
    return false;
  out[0] = values[0];
  return true;
}

} // namespace

PYB26_COLD int assemble_args(PyObject *self, PyObject *const *args,
                             Py_ssize_t nargs, PyObject *kwnames,
                             PyObject **out) noexcept {
  auto *function = reinterpret_cast<function_object *>(self);
  const Py_ssize_t nkw = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  if (!nkw)
    return bind_without_keywords(function, args, nargs, out);
  if (bind_identity_keyword(function, nargs, nkw, kwnames, args + nargs, out))
    return 0;
  int status = bind_positional(function->name, function->arity,
                               function->argsinfo, args, nargs, out);
  if (status)
    return status;
  if (nkw && !function->argsinfo) {
    PyErr_Format(PyExc_TypeError, "%s() takes no keyword arguments",
                 function->name);
    return argument_mismatch();
  }
  for (Py_ssize_t k = 0; k < nkw; ++k) {
    status = bind_keyword(function->name, function->arity, function->argsinfo,
                          PyTuple_GET_ITEM(kwnames, k), args[nargs + k], out);
    if (status)
      return status;
  }
  return bind_defaults(function->name, function->arity, function->argsinfo,
                       out);
}

PYB26_COLD int assemble_variadic_args(PyObject *self, PyObject *const *args,
                                      Py_ssize_t fixed_nargs, PyObject *kwnames,
                                      PyObject *const *keyword_values,
                                      PyObject **out) noexcept {
  auto *function = reinterpret_cast<function_object *>(self);
  const Py_ssize_t nkw = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  if (!nkw)
    return bind_without_keywords(function, args, fixed_nargs, out);
  if (bind_identity_keyword(function, fixed_nargs, nkw, kwnames, keyword_values,
                            out))
    return 0;
  int status = bind_positional(function->name, function->arity,
                               function->argsinfo, args, fixed_nargs, out);
  if (status)
    return status;
  if (nkw && !function->argsinfo) {
    PyErr_Format(PyExc_TypeError, "%s() takes no keyword arguments",
                 function->name);
    return argument_mismatch();
  }
  for (Py_ssize_t k = 0; k < nkw; ++k) {
    status = bind_keyword(function->name, function->arity, function->argsinfo,
                          PyTuple_GET_ITEM(kwnames, k), keyword_values[k], out);
    if (status)
      return status;
  }
  return bind_defaults(function->name, function->arity, function->argsinfo,
                       out);
}

namespace {

enum class ctor_mismatch_kind : unsigned char {
  none,
  too_many,
  keyword_only_name,
  keyword_only_position,
  unexpected_keyword,
  positional_only,
  multiple_values,
  missing_name,
  missing_position
};

struct ctor_diagnostic {
  ctor_mismatch_kind kind = ctor_mismatch_kind::none;
  int position = 0;
  Py_ssize_t count = 0;
  PyObject *text = nullptr;
  PyObject *raised = nullptr;
  PyThreadState *state;

  explicit ctor_diagnostic(PyObject *mismatch) noexcept
      : raised(mismatch), state(cpython::thread_state()) {}
  ~ctor_diagnostic() {
    if (text || raised)
      release();
  }

  void release() noexcept {
    PyObject *primary = Py_XNewRef(cpython::current_exception(state));
    reset_pending();
    Py_CLEAR(raised);
    if (primary)
      finish_python_error(state, primary, nullptr);
  }
  int finish(int status) noexcept {
    release();
    return cpython::current_exception(state) ? -1 : status;
  }

  void reset_pending() noexcept {
    kind = ctor_mismatch_kind::none;
    Py_CLEAR(text);
  }
  int defer(ctor_mismatch_kind next, PyObject *label = nullptr, int index = 0,
            Py_ssize_t amount = 0) noexcept {
    if (text != label) {
      PyObject *owned = Py_XNewRef(label);
      Py_XSETREF(text, owned);
    }
    kind = next;
    position = index;
    count = amount;
    return 1;
  }
  void retain_current() noexcept {
    release();
    raised = PyErr_GetRaisedException();
  }
  bool flush() noexcept {
    if (kind == ctor_mismatch_kind::none)
      return true;
    switch (kind) {
    case ctor_mismatch_kind::too_many:
      PyErr_Format(
          PyExc_TypeError,
          "__init__() takes at most %d positional argument%s (%zd given)",
          position, position == 1 ? "" : "s", count);
      break;
    case ctor_mismatch_kind::keyword_only_name:
      PyErr_Format(PyExc_TypeError, "__init__(): argument '%U' is keyword-only",
                   text);
      break;
    case ctor_mismatch_kind::keyword_only_position:
      PyErr_Format(PyExc_TypeError, "__init__(): argument %zd is keyword-only",
                   count);
      break;
    case ctor_mismatch_kind::unexpected_keyword:
      PyErr_Format(PyExc_TypeError,
                   "__init__() got an unexpected keyword argument '%U'", text);
      break;
    case ctor_mismatch_kind::positional_only:
      PyErr_Format(PyExc_TypeError,
                   "__init__(): argument '%U' is positional-only", text);
      break;
    case ctor_mismatch_kind::multiple_values:
      PyErr_Format(PyExc_TypeError,
                   "__init__() got multiple values for argument '%U'", text);
      break;
    case ctor_mismatch_kind::missing_name:
      PyErr_Format(PyExc_TypeError, "__init__() missing required argument '%U'",
                   text);
      break;
    case ctor_mismatch_kind::missing_position:
      PyErr_Format(PyExc_TypeError,
                   "__init__() missing required argument at position %d",
                   position);
      break;
    case ctor_mismatch_kind::none:
      return true;
    }
    if (argument_mismatch() < 0) {
      reset_pending();
      return false;
    }
    retain_current();
    return true;
  }
};

template <bool Deferred>
int assemble_ctor_args_impl(const ctor_spec &spec, PyObject *self,
                            PyObject *args, PyObject *kwds, PyObject **out,
                            ctor_diagnostic *diagnostic) noexcept {
  const bool cx = !Deferred && spec.errors == ctor_error_style::cx;
  const auto name = [&]() noexcept {
    return cx ? Py_TYPE(self)->tp_name : "__init__";
  };
  const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
  const int arity = spec.arity;
  arg_info *info = spec.argsinfo;
  if (nargs > arity) {
    if constexpr (Deferred)
      return diagnostic->defer(ctor_mismatch_kind::too_many, nullptr, arity,
                               nargs);
    if (cx)
      PyErr_Format(PyExc_TypeError, "%s(): too many positional arguments",
                   name());
    else
      PyErr_Format(PyExc_TypeError,
                   "%s() takes at most %d positional argument%s (%zd given)",
                   name(), arity, arity == 1 ? "" : "s", nargs);
    return argument_mismatch();
  }
  for (int i = 0; i < arity; ++i)
    out[i] = nullptr;
  for (Py_ssize_t i = 0; i < nargs; ++i) {
    if (info && (info[i].flags & arg_kw_only)) {
      if constexpr (Deferred) {
        if (!info[i].name)
          return diagnostic->defer(ctor_mismatch_kind::keyword_only_position,
                                   nullptr, 0, i);
        if (PyUnicode_CheckExact(info[i].name))
          return diagnostic->defer(ctor_mismatch_kind::keyword_only_name,
                                   info[i].name);
        if (!diagnostic->flush())
          return -1;
      }
      if (info[i].name)
        PyErr_Format(PyExc_TypeError, "%s(): argument '%U' is keyword-only",
                     name(), info[i].name);
      else
        PyErr_Format(PyExc_TypeError, "%s(): argument %zd is keyword-only",
                     name(), i);
      return argument_mismatch();
    }
    out[i] = PyTuple_GET_ITEM(args, i);
  }
  if (kwds && PyDict_GET_SIZE(kwds)) {
    if (!info) {
      if constexpr (Deferred)
        if (!diagnostic->flush())
          return -1;
      if (cx) {
        Py_ssize_t position = 0;
        PyObject *key, *value;
        PyDict_Next(kwds, &position, &key, &value);
        if (!PyUnicode_Check(key))
          PyErr_SetString(PyExc_TypeError, "keywords must be strings");
        else
          PyErr_Format(PyExc_TypeError,
                       "%s(): unexpected keyword argument '%U'", name(), key);
      } else {
        PyErr_SetString(PyExc_TypeError,
                        "__init__() takes no keyword arguments");
      }
      return argument_mismatch();
    }
    Py_ssize_t position = 0;
    PyObject *key, *value;
    while (PyDict_Next(kwds, &position, &key, &value)) {
      if (!PyUnicode_Check(key)) {
        if constexpr (Deferred)
          if (!diagnostic->flush())
            return -1;
        PyErr_SetString(PyExc_TypeError, "keywords must be strings");
        return argument_mismatch();
      }
      int index = -1;
      for (int i = 0; i < arity; ++i)
        if (info[i].name == key) {
          index = i;
          break;
        }
      if (index < 0) {
        for (int i = 0; i < arity; ++i) {
          if (!info[i].name)
            continue;
          if constexpr (Deferred) {
            if (!(PyUnicode_CheckExact(info[i].name) &&
                  PyUnicode_CheckExact(key)) &&
                !diagnostic->flush())
              return -1;
          }
          const int equal = PyObject_RichCompareBool(info[i].name, key, Py_EQ);
          if (equal < 0)
            return -1;
          if (equal) {
            index = i;
            break;
          }
        }
      }
      if (index < 0) {
        if constexpr (Deferred) {
          if (PyUnicode_CheckExact(key))
            return diagnostic->defer(ctor_mismatch_kind::unexpected_keyword,
                                     key);
          if (!diagnostic->flush())
            return -1;
        }
        PyErr_Format(PyExc_TypeError,
                     cx ? "%s(): unexpected keyword argument '%U'"
                        : "%s() got an unexpected keyword argument '%U'",
                     name(), key);
        return argument_mismatch();
      }
      if (info[index].flags & arg_pos_only) {
        if constexpr (Deferred) {
          if (PyUnicode_CheckExact(key))
            return diagnostic->defer(ctor_mismatch_kind::positional_only, key);
          if (!diagnostic->flush())
            return -1;
        }
        PyErr_Format(PyExc_TypeError, "%s(): argument '%U' is positional-only",
                     name(), key);
        return argument_mismatch();
      }
      if (out[index]) {
        if constexpr (Deferred) {
          if (PyUnicode_CheckExact(key))
            return diagnostic->defer(ctor_mismatch_kind::multiple_values, key);
          if (!diagnostic->flush())
            return -1;
        }
        PyErr_Format(PyExc_TypeError,
                     cx ? "%s(): multiple values for argument '%U'"
                        : "%s() got multiple values for argument '%U'",
                     name(), key);
        return argument_mismatch();
      }
      out[index] = value;
    }
  }
  for (int i = 0; i < arity; ++i) {
    if (out[i])
      continue;
    if (info && info[i].deflt)
      out[i] = info[i].deflt;
    else {
      if constexpr (Deferred) {
        if (!info || !info[i].name)
          return diagnostic->defer(ctor_mismatch_kind::missing_position,
                                   nullptr, i);
        if (PyUnicode_CheckExact(info[i].name))
          return diagnostic->defer(ctor_mismatch_kind::missing_name,
                                   info[i].name);
        if (!diagnostic->flush())
          return -1;
      }
      if (info && info[i].name)
        PyErr_Format(PyExc_TypeError,
                     cx ? "%s(): missing required argument '%U'"
                        : "%s() missing required argument '%U'",
                     name(), info[i].name);
      else
        PyErr_Format(PyExc_TypeError,
                     cx ? "%s(): missing required argument at position %d"
                        : "%s() missing required argument at position %d",
                     name(), i);
      return argument_mismatch();
    }
  }
  if (cx && (reinterpret_cast<instance *>(self)->flags & inst_constructed)) {
    for (int i = 0; i < arity; ++i) {
      if (out[i] == self) {
        PyErr_SetString(PyExc_TypeError,
                        "cannot reinitialize an instance from itself");
        return -1;
      }
    }
  }
  return 0;
}

} // namespace

PYB26_COLD int assemble_ctor_args(const ctor_spec &spec, PyObject *self,
                                  PyObject *args, PyObject *kwds,
                                  PyObject **out) noexcept {
  return assemble_ctor_args_impl<false>(spec, self, args, kwds, out, nullptr);
}

PYB26_COLD int reject_ctor_shape(const ctor_spec &spec, PyObject *self,
                                 PyObject *args, PyObject *kwds) noexcept {
  [[indeterminate]] PyObject *stack[16];
  if (spec.arity <= 16) {
    assemble_ctor_args(spec, self, args, kwds, stack);
  } else {
    auto slots = std::unique_ptr<PyObject *[]>(new (std::nothrow)
                                                   PyObject *[spec.arity]);
    if (!slots) {
      PyErr_NoMemory();
      return -1;
    }
    assemble_ctor_args(spec, self, args, kwds, slots.get());
  }
  return -1;
}

PYB26_COLD int init_ctor_keyword_overloads(type_record *tr, int first,
                                           PyObject *self, PyObject *args,
                                           PyObject *kwds,
                                           PyObject *mismatch) noexcept {
  ctor_diagnostic diagnostic(mismatch);
  const int status = [&]() -> int {
    for (int i = first; i < tr->nctors; ++i) {
      const ctor_spec &spec = tr->ctors[i];
      int r;
      if (spec.legacy) {
        if (!diagnostic.flush())
          return -1;
        r = kwds && PyDict_GET_SIZE(kwds)
                ? 1
                : spec.legacy(self, args, tr->data_offset);
      } else if (spec.complete_positional &&
                 PyTuple_GET_SIZE(args) == spec.arity &&
                 (!kwds || !PyDict_GET_SIZE(kwds))) {
        if (!diagnostic.flush())
          return -1;
        r = spec.invoke(self, cpython::tuple_items(args), tr->data_offset,
                        spec);
      } else {
        [[indeterminate]] PyObject *stack[16];
        if (spec.arity > 16) {
          if (!diagnostic.flush())
            return -1;
          r = spec.normalize(self, args, kwds, tr->data_offset, spec);
        } else {
          int parsed;
          if (spec.errors == ctor_error_style::regular) {
            bool scalar = false;
            if (spec.arity == 1 && spec.argsinfo && spec.argsinfo[0].name &&
                PyUnicode_CheckExact(spec.argsinfo[0].name) &&
                PyTuple_GET_SIZE(args) == 0 && kwds &&
                PyDict_GET_SIZE(kwds) == 1) {
              Py_ssize_t position = 0;
              PyObject *key, *value;
              if (PyDict_Next(kwds, &position, &key, &value) &&
                  PyUnicode_CheckExact(key)) {
                const int equal =
                    spec.argsinfo[0].name == key
                        ? 1
                        : PyUnicode_Equal(spec.argsinfo[0].name, key);
                if (equal < 0)
                  return -1;
                if (!equal) {
                  diagnostic.defer(ctor_mismatch_kind::unexpected_keyword, key);
                  continue;
                }
                if (spec.argsinfo[0].flags & arg_pos_only) {
                  diagnostic.defer(ctor_mismatch_kind::positional_only, key);
                  continue;
                }
                stack[0] = value;
                scalar = true;
              }
            }
            parsed = scalar ? 0
                            : assemble_ctor_args_impl<true>(
                                  spec, self, args, kwds, stack, &diagnostic);
          } else {
            if (!diagnostic.flush())
              return -1;
            parsed = assemble_ctor_args(spec, self, args, kwds, stack);
          }
          if (parsed < 0)
            return -1;
          if (parsed) {
            if (diagnostic.kind == ctor_mismatch_kind::none)
              diagnostic.retain_current();
            continue;
          }
          if (!diagnostic.flush())
            return -1;
          r = spec.invoke(self, stack, tr->data_offset, spec);
        }
      }
      if (r == 0)
        return 0;
      if (r == 2) {
        diagnostic.retain_current();
        continue;
      }
      if (r < 0 || PyErr_Occurred())
        return -1;
    }
    if (!diagnostic.flush())
      return -1;
    if (diagnostic.raised) {
      PyErr_SetRaisedException(diagnostic.raised);
      diagnostic.raised = nullptr;
      return -1;
    }
    PyErr_Format(PyExc_TypeError, "%s: no constructor matches the arguments",
                 tr->name.c_str());
    return -1;
  }();
  return diagnostic.finish(status);
}

PyTypeObject *create_class(PyObject *module, const char *name, const char *doc,
                           Py_ssize_t basicsize, destructor dealloc,
                           initproc init, newfunc tp_new, PyTypeObject *base,
                           class_creation_policy policy) noexcept {
  return create_class(module, module, name, doc, basicsize, dealloc, init,
                      tp_new, base, nullptr, nullptr, nullptr, 0, policy);
}

PyTypeObject *create_class(PyObject *home, PyObject *state_module,
                           const char *name, const char *doc,
                           Py_ssize_t basicsize, destructor dealloc,
                           initproc init, newfunc tp_new, PyTypeObject *base,
                           traverseproc traverse, inquiry clear,
                           const PyType_Slot *extra, int n_extra,
                           class_creation_policy policy) noexcept {
  if (PyErr_Occurred())
    return nullptr;
  if (!name || basicsize < static_cast<Py_ssize_t>(sizeof(instance)) ||
      basicsize > INT_MAX) {
    PyErr_SetString(PyExc_ValueError, "invalid class name or instance size");
    return nullptr;
  }
  if (n_extra < 0 || (n_extra && !extra) || (!!traverse != !!clear)) {
    PyErr_SetString(PyExc_ValueError,
                    "invalid class slots or incomplete GC callbacks");
    return nullptr;
  }
  const bool factory_only = policy == class_creation_policy::factory_only;
  if (factory_only && (base || traverse || clear)) {
    PyErr_SetString(PyExc_ValueError,
                    "factory-only classes cannot have a base or GC callbacks");
    return nullptr;
  }
  PyType_Slot slots[24];
  int i = 0;
  slots[i++] = {Py_tp_dealloc, reinterpret_cast<void *>(dealloc)};
  if (!factory_only) {
    slots[i++] = {Py_tp_init, reinterpret_cast<void *>(init)};
    slots[i++] = {Py_tp_new, reinterpret_cast<void *>(
                                 tp_new ? tp_new : &PyType_GenericNew)};
  }
  if (doc)
    slots[i++] = {Py_tp_doc, const_cast<char *>(doc)};
  if (traverse) {
    slots[i++] = {Py_tp_traverse, reinterpret_cast<void *>(traverse)};
    slots[i++] = {Py_tp_clear, reinterpret_cast<void *>(clear)};
  }
  if (n_extra > static_cast<int>(std::size(slots)) - i - 1) {
    PyErr_SetString(PyExc_OverflowError, "too many class slots");
    return nullptr;
  }
  for (int k = 0; k < n_extra; ++k) {
    const int slot = extra[k].slot;
    if (slot <= 0 || !extra[k].pfunc || slot == Py_tp_dealloc ||
        slot == Py_tp_init || slot == Py_tp_new || slot == Py_tp_alloc ||
        slot == Py_tp_free || slot == Py_tp_traverse || slot == Py_tp_clear ||
        slot == Py_tp_base || slot == Py_tp_bases) {
      PyErr_SetString(PyExc_ValueError, "invalid or reserved class slot");
      return nullptr;
    }
    for (int j = 0; j < i; ++j) {
      if (slots[j].slot == slot) {
        PyErr_SetString(PyExc_ValueError, "duplicate class slot");
        return nullptr;
      }
    }
    slots[i++] = extra[k];
  }
  slots[i++] = {0, nullptr};

  PyObject *mod_name = PyModule_GetNameObject(home);
  if (!mod_name)
    return nullptr;
  PyObject *qualified = PyUnicode_FromFormat("%U.%s", mod_name, name);
  if (!qualified) {
    Py_DECREF(mod_name);
    return nullptr;
  }
  const char *qualified_name = PyUnicode_AsUTF8(qualified);
  if (!qualified_name) {
    Py_DECREF(qualified);
    Py_DECREF(mod_name);
    return nullptr;
  }
  PyType_Spec spec{};
  spec.name = qualified_name;
  spec.basicsize = static_cast<int>(basicsize);
  spec.itemsize = 0;
  spec.flags = Py_TPFLAGS_DEFAULT |
               (factory_only ? Py_TPFLAGS_DISALLOW_INSTANTIATION |
                                   Py_TPFLAGS_IMMUTABLETYPE
                             : Py_TPFLAGS_BASETYPE) |
               (traverse ? Py_TPFLAGS_HAVE_GC : 0);
  spec.slots = slots;

  PyObject *bases = nullptr;
  if (base) {
    bases = PyTuple_Pack(1, reinterpret_cast<PyObject *>(base));
    if (!bases) {
      Py_DECREF(qualified);
      Py_DECREF(mod_name);
      return nullptr;
    }
  }
  PyObject *t = PyType_FromModuleAndSpec(state_module, &spec, bases);
  Py_XDECREF(bases);
  Py_DECREF(qualified);
  if (!t) {
    Py_DECREF(mod_name);
    return nullptr;
  }
  const int status =
      factory_only
          ? PyDict_SetItemString(reinterpret_cast<PyTypeObject *>(t)->tp_dict,
                                 "__module__", mod_name)
          : PyObject_SetAttrString(t, "__module__", mod_name);
  if (factory_only && status == 0)
    PyType_Modified(reinterpret_cast<PyTypeObject *>(t));
  Py_DECREF(mod_name);
  if (status < 0) {
    Py_DECREF(t);
    return nullptr;
  }
  return reinterpret_cast<PyTypeObject *>(t);
}

int register_class(PyObject *home, PyObject *module, type_record *tr,
                   type_record *&chain_head) noexcept {
  if (PyErr_Occurred())
    return -1;
  PyObject *dict = PyModule_GetDict(home);
  if (!dict)
    return -1;
  PyObject *existing = nullptr;
  const int found = PyDict_GetItemStringRef(dict, tr->name.c_str(), &existing);
  if (found < 0)
    return -1;
  if (found) {
    Py_DECREF(existing);
    PyErr_Format(PyExc_TypeError, "module attribute '%s' already exists",
                 tr->name.c_str());
    return -1;
  }
  auto *st = static_cast<module_state *>(PyModule_GetState(module));
  if (!st) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_RuntimeError,
                      "class registration requires extension module state");
    return -1;
  }
  if (!st->types) {
    auto *reg = new (std::nothrow) type_registry{};
    if (!reg) {
      PyErr_NoMemory();
      return -1;
    }
    auto *hk = new (std::nothrow) teardown_hook{st};
    if (!hk) {
      delete reg;
      PyErr_NoMemory();
      return -1;
    }
    if (cpython::at_interpreter_exit(PyInterpreterState_Get(),
                                     &clear_types_at_exit, hk) < 0) {
      delete hk;
      delete reg;
      return -1;
    }
    reg->hook = hk;
    st->types = reg;
  }
  tr->interp = PyInterpreterState_Get();
  if (!st->types->map.emplace(tr->cpp_type, tr).second) {
    PyErr_Format(PyExc_TypeError,
                 "C++ type for '%s' is already registered in this module",
                 tr->name.c_str());
    return -1;
  }
  if (register_type(chain_head, tr) < 0) {
    st->types->map.erase(tr->cpp_type);
    return -1;
  }
  if (tr->lifetime_offset &&
      set_lifetime_offset(tr->py_type, tr->lifetime_offset) < 0) {
    unregister_type(chain_head, tr);
    st->types->map.erase(tr->cpp_type);
    return -1;
  }
  if (PyModule_AddObjectRef(home, tr->name.c_str(),
                            reinterpret_cast<PyObject *>(tr->py_type)) < 0) {
    unregister_type(chain_head, tr);
    st->types->map.erase(tr->cpp_type);
    return -1;
  }
  cpython::prepare_shared_object(reinterpret_cast<PyObject *>(tr->py_type));
  return 0;
}

PyObject *alloc_instance(type_record *tr) noexcept {
  PyObject *o = tr->py_type->tp_alloc(tr->py_type, 0);
  if (o)
    std::construct_at(&reinterpret_cast<instance *>(o)->flags);
  return o;
}

PyObject *alloc_holder_instance(type_record *tr) noexcept {
  PyObject *o = tr->py_type->tp_alloc(tr->py_type, 0);
  if (o) {
    auto *in = reinterpret_cast<instance *>(o);
    std::construct_at(&in->flags);
    in->storage = st_inline;
  }
  return o;
}

static int bind_on_type(PyTypeObject *type, const char *name,
                        function_object *entry, PyTypeObject *otype) noexcept {
  PyObject *tdict = type->tp_dict;
  PyObject *existing = nullptr;
  if (tdict)
    PyDict_GetItemStringRef(tdict, name, &existing);
  PyObject *to_bind;
  if (existing && Py_IS_TYPE(existing, otype)) {
    to_bind = combine_overload(reinterpret_cast<function_object *>(existing),
                               entry, otype);
    Py_DECREF(existing);
    if (!to_bind)
      return -1;
  } else {
    Py_XDECREF(existing);
    to_bind = reinterpret_cast<PyObject *>(entry);
  }
  int rc =
      PyObject_SetAttrString(reinterpret_cast<PyObject *>(type), name, to_bind);
  Py_DECREF(to_bind);
  return rc;
}

int add_method(PyTypeObject *type, const callable_spec &spec) noexcept {
  if (PyErr_Occurred()) {
    clear_callable_spec(spec);
    return -1;
  }
  auto *st = static_cast<module_state *>(PyType_GetModuleState(type));
  if (!st || !st->method_type) {
    clear_callable_spec(spec);
    return -1;
  }
  auto *f = make_callable(st->method_type, spec);
  if (!f)
    return -1;
  return bind_on_type(type, spec.name, f, st->method_type);
}

int add_static(PyTypeObject *type, const callable_spec &spec) noexcept {
  if (PyErr_Occurred()) {
    clear_callable_spec(spec);
    return -1;
  }
  auto *st = static_cast<module_state *>(PyType_GetModuleState(type));
  if (!st || !st->function_type) {
    clear_callable_spec(spec);
    return -1;
  }
  auto *f = make_callable(st->function_type, spec);
  if (!f)
    return -1;
  return bind_on_type(type, spec.name, f, st->function_type);
}

int add_getset(PyTypeObject *type, const getset_spec &spec) noexcept {
  PyObject *descriptor =
      make_owned_getset(type, spec.name, spec.doc, spec.get, spec.set,
                        spec.closure, spec.free_closure, false);
  if (!descriptor)
    return -1;
  int status = PyObject_SetAttrString(reinterpret_cast<PyObject *>(type),
                                      spec.name, descriptor);
  PyObject *error = PyErr_GetRaisedException();
  Py_DECREF(descriptor);
  PyErr_SetRaisedException(error);
  return status;
}

int add_getset(PyTypeObject *type, const char *name, const char *doc, getter g,
               setter s) noexcept {
  return add_getset(type, getset_spec{name, doc, g, s});
}

static int static_meta_traverse(PyObject *self, visitproc visit,
                                void *arg) noexcept {
  Py_VISIT(reinterpret_cast<PyObject *>(Py_TYPE(self)));
  return PyType_Type.tp_traverse(self, visit, arg);
}
static int static_meta_clear(PyObject *self) noexcept {
  return PyType_Type.tp_clear(self);
}

static PyTypeObject *get_static_metaclass(PyObject *module,
                                          module_state *st) noexcept {
  if (st->static_metaclass)
    return st->static_metaclass;
  static PyType_Slot meta_slots[] = {
      {Py_tp_base, &PyType_Type},
      {Py_tp_traverse, reinterpret_cast<void *>(&static_meta_traverse)},
      {Py_tp_clear, reinterpret_cast<void *>(&static_meta_clear)},
      {0, nullptr},
  };
  static PyType_Spec meta_spec = {
      "pyb26.static_meta",
      0,
      0,
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
      meta_slots,
  };
  PyObject *mt = PyType_FromModuleAndSpec(module, &meta_spec, nullptr);
  if (!mt)
    return nullptr;
  st->static_metaclass = reinterpret_cast<PyTypeObject *>(mt);
  cpython::prepare_shared_object(mt);
  return st->static_metaclass;
}

int add_static_getset(PyObject *module, PyTypeObject *type,
                      const getset_spec &spec) noexcept {
  PyObject *descriptor =
      make_owned_getset(type, spec.name, spec.doc, spec.get, spec.set,
                        spec.closure, spec.free_closure, true);
  if (!descriptor)
    return -1;
  const int status = [&]() noexcept -> int {
    auto *st = static_cast<module_state *>(PyModule_GetState(module));
    if (!st)
      return -1;
    PyTypeObject *common_meta = get_static_metaclass(module, st);
    if (!common_meta)
      return -1;
    type_record *owner = record_for_type(type);
    if (!owner) {
      PyErr_SetString(PyExc_TypeError,
                      "static property owner is not registered");
      return -1;
    }
    PyTypeObject *meta = owner->static_meta;
    if (!meta) {
      PyTypeObject *previous = Py_TYPE(type);
      if (previous != &PyType_Type &&
          !PyType_IsSubtype(previous, common_meta)) {
        PyErr_Format(
            PyExc_SystemError,
            "pyb26: cannot add a static member to '%s' (unexpected metaclass)",
            type->tp_name);
        return -1;
      }
      PyTypeObject *base = previous == &PyType_Type ? common_meta : previous;
      PyObject *bases = PyTuple_Pack(1, reinterpret_cast<PyObject *>(base));
      if (!bases)
        return -1;
      PyType_Slot slots[] = {{0, nullptr}};
      PyType_Spec meta_spec{"pyb26.static_meta", 0, 0,
                            Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, slots};
      PyObject *created = PyType_FromModuleAndSpec(module, &meta_spec, bases);
      Py_DECREF(bases);
      if (!created)
        return -1;
      meta = reinterpret_cast<PyTypeObject *>(created);
      Py_SET_TYPE(reinterpret_cast<PyObject *>(type),
                  reinterpret_cast<PyTypeObject *>(meta));
      Py_DECREF(previous);
      owner->static_meta = meta;
      PyType_Modified(type);
    }
    if (PyDict_SetItemString(meta->tp_dict, spec.name, descriptor) < 0)
      return -1;
    PyType_Modified(meta);
    if (PyDict_SetItemString(type->tp_dict, spec.name, descriptor) < 0)
      return -1;
    PyType_Modified(type);
    return 0;
  }();
  PyObject *error = PyErr_GetRaisedException();
  Py_DECREF(descriptor);
  PyErr_SetRaisedException(error);
  return status;
}

int add_static_getset(PyObject *module, PyTypeObject *type, const char *name,
                      getter g, setter s) noexcept {
  return add_static_getset(
      module, type, getset_spec{name, nullptr, g, s, const_cast<char *>(name)});
}

PYB26_COLD PyObject *error_arity(PyObject *self, Py_ssize_t got_args,
                                 PyObject *kwnames) noexcept {
  auto *f = reinterpret_cast<function_object *>(self);
  if (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0) {
    PyErr_Format(PyExc_TypeError, "%s() takes no keyword arguments", f->name);
    return nullptr;
  }
  PyErr_Format(PyExc_TypeError,
               "%s() takes exactly %d positional argument%s (%zd given)",
               f->name, f->arity, f->arity == 1 ? "" : "s", got_args);
  return nullptr;
}

PYB26_COLD PyObject *error_arg_convert(PyObject *self) noexcept {
  if (!PyErr_Occurred()) {
    auto *f = reinterpret_cast<function_object *>(self);
    PyErr_Format(PyExc_TypeError, "%s(): incompatible argument type(s)",
                 f->name);
  }
  return nullptr;
}

PYB26_COLD PyObject *error_unregistered(const char *cpp_name) noexcept {
  PyErr_Format(PyExc_TypeError,
               "pyb26: C++ type '%s' has no registered Python class", cpp_name);
  return nullptr;
}

} // namespace pyb26::detail

#include "pyb26/detail/owned_property_runtime.hpp"
