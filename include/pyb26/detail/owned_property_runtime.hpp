#pragma once

#include "common.hpp"

namespace pyb26::detail {

struct owned_getset_object {
  PyObject_HEAD PyTypeObject *owner;
  PyObject *name;
  PyObject *doc;
  getter get;
  setter set;
  bool is_static;
  void *capture;
  void (*free_capture)(void *) noexcept;
};

static int owned_getset_traverse(PyObject *self, visitproc visit,
                                 void *arg) noexcept {
  auto *d = reinterpret_cast<owned_getset_object *>(self);
  Py_VISIT(Py_TYPE(self));
  Py_VISIT(d->owner);
  Py_VISIT(d->name);
  Py_VISIT(d->doc);
  return 0;
}

static int owned_getset_clear(PyObject *self) noexcept {
  auto *d = reinterpret_cast<owned_getset_object *>(self);
  PyObject *error = PyErr_GetRaisedException();
  void *capture = d->capture;
  d->capture = nullptr;
  d->get = nullptr;
  d->set = nullptr;
  if (capture && d->free_capture)
    d->free_capture(capture);
  Py_CLEAR(d->owner);
  Py_CLEAR(d->name);
  Py_CLEAR(d->doc);
  PyErr_SetRaisedException(error);
  return 0;
}

static void owned_getset_dealloc(PyObject *self) noexcept {
  PyTypeObject *type = Py_TYPE(self);
  PyObject_GC_UnTrack(self);
  owned_getset_clear(self);
  type->tp_free(self);
  Py_DECREF(type);
}

static PyObject *owned_getset_new(PyTypeObject *, PyObject *,
                                  PyObject *) noexcept {
  PyErr_SetString(PyExc_TypeError,
                  "pyb26 properties cannot be constructed directly");
  return nullptr;
}

static bool owned_getset_accepts(owned_getset_object *d,
                                 PyObject *instance) noexcept {
  if (d->owner && (PyObject_TypeCheck(instance, d->owner) ||
                   (d->is_static && PyType_Check(instance) &&
                    PyType_IsSubtype(reinterpret_cast<PyTypeObject *>(instance),
                                     d->owner))))
    return true;
  PyErr_SetString(PyExc_TypeError,
                  "property requires its registered instance type");
  return false;
}

static PyObject *owned_getset_get(PyObject *self, PyObject *instance,
                                  PyObject *) noexcept {
  auto *d = reinterpret_cast<owned_getset_object *>(self);
  if (!instance || instance == Py_None) {
    if (!d->is_static)
      return Py_NewRef(self);
    instance = reinterpret_cast<PyObject *>(d->owner);
    if (!instance) {
      PyErr_SetString(PyExc_AttributeError, "property owner was cleared");
      return nullptr;
    }
  }
  if (!owned_getset_accepts(d, instance))
    return nullptr;
  if (!d->get) {
    PyErr_SetString(PyExc_AttributeError, "property has no getter");
    return nullptr;
  }
  return d->get(instance, d->capture);
}

static int owned_getset_set(PyObject *self, PyObject *instance,
                            PyObject *value) noexcept {
  auto *d = reinterpret_cast<owned_getset_object *>(self);
  if (!owned_getset_accepts(d, instance))
    return -1;
  if (!d->set) {
    PyErr_SetString(PyExc_AttributeError, "property is read-only");
    return -1;
  }
  return d->set(instance, value, d->capture);
}

static PyObject *owned_getset_name(PyObject *self, void *) noexcept {
  auto *d = reinterpret_cast<owned_getset_object *>(self);
  return Py_NewRef(d->name ? d->name : Py_None);
}

static PyObject *owned_getset_doc(PyObject *self, void *) noexcept {
  auto *d = reinterpret_cast<owned_getset_object *>(self);
  return Py_NewRef(d->doc ? d->doc : Py_None);
}

static PyObject *owned_getset_owner(PyObject *self, void *) noexcept {
  auto *d = reinterpret_cast<owned_getset_object *>(self);
  return Py_NewRef(d->owner ? reinterpret_cast<PyObject *>(d->owner) : Py_None);
}

static PyGetSetDef owned_getset_metadata[] = {
    {"__name__", owned_getset_name, nullptr, nullptr, nullptr},
    {"__doc__", owned_getset_doc, nullptr, nullptr, nullptr},
    {"__objclass__", owned_getset_owner, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

static PyType_Slot owned_getset_slots[] = {
    {Py_tp_new, reinterpret_cast<void *>(&owned_getset_new)},
    {Py_tp_dealloc, reinterpret_cast<void *>(&owned_getset_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(&owned_getset_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(&owned_getset_clear)},
    {Py_tp_descr_get, reinterpret_cast<void *>(&owned_getset_get)},
    {Py_tp_descr_set, reinterpret_cast<void *>(&owned_getset_set)},
    {Py_tp_getset, owned_getset_metadata},
    {0, nullptr},
};

static PyType_Spec owned_getset_spec = {
    "pyb26.owned_property",
    sizeof(owned_getset_object),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    owned_getset_slots,
};

static PyTypeObject *owned_getset_type(PyTypeObject *owner) noexcept {
  PyObject *module = PyType_GetModule(owner);
  if (!module)
    return nullptr;
  auto *st = static_cast<module_state *>(PyModule_GetState(module));
  if (!st)
    return nullptr;
  PyTypeObject *result = nullptr;
  Py_BEGIN_CRITICAL_SECTION(module);
  if (!st->owned_getset_type) {
    PyObject *made =
        PyType_FromModuleAndSpec(module, &owned_getset_spec, nullptr);
    if (made) {
      if (!st->owned_getset_type)
        st->owned_getset_type = reinterpret_cast<PyTypeObject *>(made);
      else
        Py_DECREF(made);
    }
  }
  if (!PyErr_Occurred() && st->owned_getset_type)
    result = reinterpret_cast<PyTypeObject *>(
        Py_NewRef(reinterpret_cast<PyObject *>(st->owned_getset_type)));
  Py_END_CRITICAL_SECTION();
  return result;
}

static PyObject *make_owned_getset(PyTypeObject *owner, const char *name,
                                   const char *doc, getter get, setter set,
                                   void *capture,
                                   void (*free_capture)(void *) noexcept,
                                   bool is_static) noexcept {
  if (PyErr_Occurred()) {
    PyObject *error = PyErr_GetRaisedException();
    if (capture && free_capture)
      free_capture(capture);
    PyErr_SetRaisedException(error);
    return nullptr;
  }
  PyTypeObject *dtype = owned_getset_type(owner);
  if (!dtype) {
    PyObject *error = PyErr_GetRaisedException();
    if (capture && free_capture)
      free_capture(capture);
    PyErr_SetRaisedException(error);
    return nullptr;
  }
  auto *d = reinterpret_cast<owned_getset_object *>(dtype->tp_alloc(dtype, 0));
  Py_DECREF(dtype);
  if (!d) {
    PyObject *error = PyErr_GetRaisedException();
    if (capture && free_capture)
      free_capture(capture);
    PyErr_SetRaisedException(error);
    return nullptr;
  }
  d->owner = reinterpret_cast<PyTypeObject *>(
      Py_NewRef(reinterpret_cast<PyObject *>(owner)));
  d->capture = capture;
  d->free_capture = free_capture;
  d->get = get;
  d->set = set;
  d->is_static = is_static;
  d->name = PyUnicode_FromString(name);
  if (d->name)
    d->doc = doc ? PyUnicode_FromString(doc) : Py_NewRef(Py_None);
  if (!d->name || !d->doc) {
    PyObject *error = PyErr_GetRaisedException();
    Py_DECREF(d);
    PyErr_SetRaisedException(error);
    return nullptr;
  }
  return reinterpret_cast<PyObject *>(d);
}

int add_owned_getset(PyTypeObject *owner, const char *name, const char *doc,
                     getter get, setter set, void *capture,
                     void (*free_capture)(void *) noexcept) noexcept {
  PyObject *descriptor = make_owned_getset(owner, name, doc, get, set, capture,
                                           free_capture, false);
  if (!descriptor)
    return -1;
  int status = PyObject_SetAttrString(reinterpret_cast<PyObject *>(owner), name,
                                      descriptor);
  PyObject *error = PyErr_GetRaisedException();
  Py_DECREF(descriptor);
  PyErr_SetRaisedException(error);
  return status;
}

} // namespace pyb26::detail
