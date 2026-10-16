#pragma once

#include "detail/map_output_snapshot.hpp"
#include "detail/opaque_conversion.hpp"
#include "pyb26.hpp"

#include <atomic>
#include <concepts>
#include <map>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pyb26 {
namespace detail {

template <class M> PYB26_ALWAYS_INLINE M *map_self(PyObject *self) noexcept {
  M *m = cast_to<M>(self);
  if (!m) [[unlikely]]
    PyErr_SetString(PyExc_TypeError, "pyb26: invalid bound-map instance");
  return m;
}

template <class M>
PyObject *map_value_out(typename M::mapped_type &v) noexcept {
  using T = typename M::mapped_type;
  return opaque_copy_output<T>(v);
}

template <class M> inline constexpr bool native_map_iteration = false;
template <class K, class V, class Compare, class Allocator>
inline constexpr bool native_map_iteration<std::map<K, V, Compare, Allocator>> =
    true;
template <class K, class V, class Hash, class Equal, class Allocator>
inline constexpr bool
    native_map_iteration<std::unordered_map<K, V, Hash, Equal, Allocator>> =
        true;

template <class M, map_iter_kind Kind>
inline constexpr bool flat_map_output_snapshot =
    native_map_iteration<M> && map_output_fields<M, Kind>::enabled;

template <class M, map_iter_kind Kind> struct map_iter_state {
  using value_type = std::conditional_t<
      Kind == map_iter_kind::keys, typename M::key_type,
      std::conditional_t<
          Kind == map_iter_kind::values, typename M::mapped_type,
          std::pair<typename M::key_type, typename M::mapped_type>>>;
  std::conditional_t<flat_map_output_snapshot<M, Kind>,
                     map_output_snapshot<M, Kind>, std::vector<value_type>>
      snapshot;
#ifdef Py_GIL_DISABLED
  std::conditional_t<flat_map_output_snapshot<M, Kind>,
                     std::atomic<std::size_t>, std::size_t>
      next{0};
#else
  std::size_t next = 0;
#endif
};

template <class M, map_iter_kind Kind>
PYB26_ALWAYS_INLINE map_iter_state<M, Kind> *
map_iter_self(PyObject *self) noexcept {
  using State = map_iter_state<M, Kind>;
  auto *in = reinterpret_cast<instance *>(self);
  State *st = nullptr;
  if ((in->flags.lifecycle() & (inst_constructed | inst_initializing)) ==
      inst_constructed)
    st = object_ptr<State>(self, in->storage, holder_data_offset<State>());
  if (!st) [[unlikely]]
    PyErr_SetString(PyExc_TypeError, "pyb26: invalid map-iterator instance");
  return st;
}

template <class M, map_iter_kind Kind>
PyObject *map_iter_length_hint(PyObject *self, PyObject *) noexcept {
  auto *st = map_iter_self<M, Kind>(self);
  if (!st) [[unlikely]]
    return nullptr;
  std::size_t remaining;
#ifdef Py_GIL_DISABLED
  if constexpr (flat_map_output_snapshot<M, Kind>) {
    remaining = st->snapshot.size() - st->next.load(std::memory_order_relaxed);
  } else
#endif
  {
    Py_BEGIN_CRITICAL_SECTION(self);
    remaining = st->snapshot.size() - st->next;
    Py_END_CRITICAL_SECTION();
  }
  return PyLong_FromSize_t(remaining);
}

template <class M, map_iter_kind Kind>
PyObject *map_iter_next(PyObject *self) noexcept {
  using State = map_iter_state<M, Kind>;
  auto *st = map_iter_self<M, Kind>(self);
  if (!st) [[unlikely]]
    return nullptr;
  if constexpr (flat_map_output_snapshot<M, Kind>) {
#ifdef Py_GIL_DISABLED
    auto index = st->next.load(std::memory_order_relaxed);
    const auto limit = st->snapshot.size();
    while (index < limit) {
      if (st->next.compare_exchange_weak(index, index + 1,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed))
        return st->snapshot.cast(index);
    }
    return nullptr;
#else
    std::size_t index = 0;
    bool found = false;
    Py_BEGIN_CRITICAL_SECTION(self);
    if (st->next < st->snapshot.size()) {
      index = st->next++;
      found = true;
    }
    Py_END_CRITICAL_SECTION();
    return found ? st->snapshot.cast(index) : nullptr;
#endif
  } else {
    using Value = typename State::value_type;
    constexpr bool native_output = [] {
      if constexpr (Kind == map_iter_kind::items)
        return builtin_output_cast<typename M::key_type> &&
               builtin_output_cast<typename M::mapped_type>;
      else
        return builtin_output_cast<Value>;
    }();
    if constexpr (native_move_inert_v<Value> && native_cleanup_inert_v<Value> &&
                  native_output) {
      const Value *value = nullptr;
      Py_BEGIN_CRITICAL_SECTION(self);
      if (st->next < st->snapshot.size())
        value = &st->snapshot[st->next++];
      Py_END_CRITICAL_SECTION();
      if (!value)
        return nullptr;
      if constexpr (Kind == map_iter_kind::items)
        return stl_cast_tuple(*value, std::index_sequence<0, 1>{});
      else
        return opaque_cast_value(*value);
    } else {
      std::optional<Value> value;
      auto take_next = [&] {
        Py_BEGIN_CRITICAL_SECTION(self);
        if (st->next < st->snapshot.size())
          value.emplace(std::move(st->snapshot[st->next++]));
        Py_END_CRITICAL_SECTION();
      };
      auto cast_value = [&] {
        if constexpr (Kind == map_iter_kind::items)
          return stl_cast_tuple(*value, std::index_sequence<0, 1>{});
        else
          return opaque_cast_value(std::move(*value));
      };
      callback_error_scope error;
      take_next();
      if (error.capture()) {
        value.reset();
        return nullptr;
      }
      if (!value)
        return nullptr;
      PyObject *result = cast_value();
      error.capture();
      value.reset();
      return opaque_finish_output(result, error);
    }
  }
}

template <class M, map_iter_kind Kind>
void register_map_iterator(module_ m, const char *name) {
  static PyMethodDef methods[] = {
      {"__length_hint__", &map_iter_length_hint<M, Kind>, METH_NOARGS,
       "__length_hint__($self, /)\n--\n\n"
       "Return the number of unclaimed snapshot entries."},
      {nullptr, nullptr, 0, nullptr},
  };
  static PyType_Slot slots[] = {
      {Py_tp_iter, reinterpret_cast<void *>(&PyObject_SelfIter)},
      {Py_tp_iternext, reinterpret_cast<void *>(&map_iter_next<M, Kind>)},
      {Py_tp_methods, methods},
  };
  class_<map_iter_state<M, Kind>>(m, name, factory_only_t{}, slots,
                                  static_cast<int>(std::size(slots)));
}

template <class M, map_iter_kind Kind>
PyObject *map_iter_new(PyObject *map_wrapper) noexcept {
  using State = map_iter_state<M, Kind>;
  M *m = map_self<M>(map_wrapper);
  if (!m) [[unlikely]]
    return nullptr;
  if constexpr (flat_map_output_snapshot<M, Kind>) {
    map_output_snapshot<M, Kind> snapshot;
    if (!snapshot.acquire(map_wrapper, *m))
      return nullptr;
    return make_instance<State>(
        [&](void *mem) { ::new (mem) State{std::move(snapshot)}; });
  } else {
    auto build = [&]<bool Checked, class Error>(Error &error) -> PyObject * {
      std::vector<typename State::value_type> snapshot;
      invocation_checkpoint<Error, Checked> checkpoint{error};
      Py_BEGIN_CRITICAL_SECTION(map_wrapper);
      snapshot.reserve(m->size());
      for (const auto &[key, value] : *m) {
        if constexpr (Checked)
          if (error.capture())
            break;
        if constexpr (Kind == map_iter_kind::keys)
          snapshot.push_back(key);
        else if constexpr (Kind == map_iter_kind::values)
          snapshot.push_back(value);
        else
          snapshot.emplace_back(key, value);
      }
      Py_END_CRITICAL_SECTION();
      if constexpr (Checked)
        if (error.capture())
          return nullptr;
      return make_instance<State>(
          [&](void *mem) { ::new (mem) State{std::move(snapshot)}; });
    };
    if constexpr (native_map_iteration<M> &&
                  native_copy_inert_v<typename State::value_type> &&
                  native_cleanup_inert_v<typename State::value_type>) {
      std::nullptr_t error = nullptr;
      return build.template operator()<false>(error);
    } else {
      callback_error_scope error;
      return build.template operator()<true>(error);
    }
  }
}

template <class M> Py_ssize_t map_len(PyObject *self) noexcept {
  M *m = map_self<M>(self);
  if (!m) [[unlikely]]
    return -1;
  Py_ssize_t n;
  Py_BEGIN_CRITICAL_SECTION(self);
  n = static_cast<Py_ssize_t>(m->size());
  Py_END_CRITICAL_SECTION();
  return n;
}

template <class M>
PyObject *map_subscript(PyObject *self, PyObject *key) noexcept {
  using K = typename M::key_type;
  M *m = map_self<M>(self);
  if (!m) [[unlikely]]
    return nullptr;
  callback_error_scope error;
  std::optional<K> k;
  callback_error_checkpoint checkpoint{error};
  if (!opaque_load_item(key, k, error)) [[unlikely]] {
    if (!PyErr_Occurred())
      PyErr_SetObject(PyExc_KeyError, key);
    return nullptr;
  }
  PyObject *r = nullptr;
  bool miss = false;
  Py_BEGIN_CRITICAL_SECTION(self);
  auto it = m->find(*k);
  if (!error.capture()) {
    if (it == m->end())
      miss = true;
    else
      r = map_value_out<M>(it->second);
  }
  Py_END_CRITICAL_SECTION();
  if (miss && !error.capture()) [[unlikely]] {
    PyErr_SetObject(PyExc_KeyError, key);
    return nullptr;
  }
  error.capture();
  k.reset();
  return opaque_finish_output(r, error);
}

template <class M>
int map_ass_subscript(PyObject *self, PyObject *key, PyObject *value) noexcept {
  using K = typename M::key_type;
  using V = typename M::mapped_type;
  M *m = map_self<M>(self);
  if (!m) [[unlikely]]
    return -1;
  callback_error_scope error;
  std::optional<K> k;
  std::optional<V> v;
  callback_error_checkpoint checkpoint{error};
  if (!opaque_load_item(key, k, error)) [[unlikely]] {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError,
                      "pyb26: incompatible key type for this map");
    return -1;
  }
  if (value) {
    if (!opaque_load_item(value, v, error)) [[unlikely]] {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError,
                        "pyb26: incompatible value type for this map");
      return -1;
    }
    Py_BEGIN_CRITICAL_SECTION(self);
    m->insert_or_assign(std::move(*k), std::move(*v));
    Py_END_CRITICAL_SECTION();
    error.capture();
    v.reset();
    error.capture();
    k.reset();
    return error.capture() ? -1 : 0;
  }
  std::size_t erased;
  Py_BEGIN_CRITICAL_SECTION(self);
  erased = m->erase(*k);
  Py_END_CRITICAL_SECTION();
  if (!erased && !error.capture()) [[unlikely]] {
    PyErr_SetObject(PyExc_KeyError, key);
    return -1;
  }
  error.capture();
  k.reset();
  return error.capture() ? -1 : 0;
}

template <class M> int map_contains(PyObject *self, PyObject *key) noexcept {
  using K = typename M::key_type;
  M *m = map_self<M>(self);
  if (!m) [[unlikely]]
    return -1;
  callback_error_scope error;
  std::optional<K> k;
  callback_error_checkpoint checkpoint{error};
  if (!opaque_load_item(key, k, error))
    return error.capture() ? -1 : 0;
  int found;
  Py_BEGIN_CRITICAL_SECTION(self);
  found = m->find(*k) != m->end() ? 1 : 0;
  Py_END_CRITICAL_SECTION();
  error.capture();
  k.reset();
  return error.capture() ? -1 : found;
}

template <class M> PyObject *map_tp_iter(PyObject *self) noexcept {
  return map_iter_new<M, map_iter_kind::keys>(self);
}

template <class M> PyObject *map_keys(PyObject *self) noexcept {
  return map_iter_new<M, map_iter_kind::keys>(self);
}
template <class M> PyObject *map_values(PyObject *self) noexcept {
  return map_iter_new<M, map_iter_kind::values>(self);
}
template <class M> PyObject *map_items(PyObject *self) noexcept {
  return map_iter_new<M, map_iter_kind::items>(self);
}

template <class M>
PyObject *map_get_default(PyObject *self, const typename M::key_type &k,
                          PyObject *dflt) noexcept {
  M *m = map_self<M>(self);
  if (!m) [[unlikely]]
    return nullptr;
  PyObject *r = nullptr;
  callback_error_scope error;
  Py_BEGIN_CRITICAL_SECTION(self);
  auto it = m->find(k);
  if (!error.capture() && it != m->end())
    r = map_value_out<M>(it->second);
  Py_END_CRITICAL_SECTION();
  if (r || error.capture())
    return opaque_finish_output(r, error);
  return Py_NewRef(dflt ? dflt : Py_None);
}
template <class M>
PyObject *map_get(PyObject *self, const typename M::key_type &k) noexcept {
  return map_get_default<M>(self, k, nullptr);
}

template <class M> PyObject *map_clear(PyObject *self) noexcept {
  M *m = map_self<M>(self);
  if (!m) [[unlikely]]
    return nullptr;
  Py_BEGIN_CRITICAL_SECTION(self);
  m->clear();
  Py_END_CRITICAL_SECTION();
  Py_RETURN_NONE;
}

template <class M> PyObject *map_eq(PyObject *self, PyObject *other) noexcept {
  M *m = map_self<M>(self);
  if (!m) [[unlikely]]
    return nullptr;
  if constexpr (std::equality_comparable<typename M::mapped_type>) {
    M *o = cast_to<M>(other);
    if (!o)
      Py_RETURN_NOTIMPLEMENTED;
    bool eq;
    Py_BEGIN_CRITICAL_SECTION2(self, other);
    eq = (*m == *o);
    Py_END_CRITICAL_SECTION2();
    return PyBool_FromLong(eq);
  } else {
    (void)other;
    Py_RETURN_NOTIMPLEMENTED;
  }
}

template <class M>
int map_ctor_from(PyObject *self, PyObject *args,
                  Py_ssize_t data_offset) noexcept {
  using K = typename M::key_type;
  using V = typename M::mapped_type;
  if (PyTuple_GET_SIZE(args) != 1)
    return 1;
  PyObject *src = PyTuple_GET_ITEM(args, 0);
  callback_error_scope error;
  M staged;
  callback_error_checkpoint checkpoint{error};
  if (!PyDict_Check(src))
    return 1;
  if (!stl_load_dict<K, V>(src, staged))
    return error.capture() ? -1 : 1;
  ::new (reinterpret_cast<char *>(self) + data_offset) M(std::move(staged));
  reinterpret_cast<instance *>(self)->flags |= inst_constructed | inst_owned;
  return 0;
}

} // namespace detail

template <class M, class... Extras>
class_<M> bind_map(module_ m, const char *name, const char *doc = nullptr) {
  static_assert(opaque_v<M>,
                "pyb26: bind_map<M> requires M to be OPAQUE — specialize "
                "pyb26::opaque_v<M> (or PYB26_OPAQUE(M)) first; without it the "
                "STL caster would keep copying at every boundary.");
  using K = typename M::key_type;
  using V = typename M::mapped_type;
  static_assert(
      sizeof...(Extras) == 0,
      "pyb26: bound maps use copied elements; alias policies are unsupported");
  static_assert(
      std::is_copy_constructible_v<K> && std::is_copy_constructible_v<V>,
      "pyb26: bound map keys and values must support independent copies");
  static_assert(
      !caster_yields_view_v<K> && !caster_yields_view_v<V>,
      "pyb26: a bound map cannot hold view keys/values (string_view / "
      "const char* / PyObject*): stored views would outlive the Python "
      "objects they point into. Use owned types.");

  static PyType_Slot slots[] = {
      {Py_mp_length, reinterpret_cast<void *>(&detail::map_len<M>)},
      {Py_mp_subscript, reinterpret_cast<void *>(&detail::map_subscript<M>)},
      {Py_mp_ass_subscript,
       reinterpret_cast<void *>(&detail::map_ass_subscript<M>)},
      {Py_sq_contains, reinterpret_cast<void *>(&detail::map_contains<M>)},
      {Py_tp_iter, reinterpret_cast<void *>(&detail::map_tp_iter<M>)},
  };
  class_<M> c(m, name, doc, slots,
              static_cast<int>(sizeof(slots) / sizeof(slots[0])));
  if (PyErr_Occurred())
    return c;
  c.template ctor<>();
  c.ctor_raw(&detail::map_ctor_from<M>);
  c.template def_fn<&detail::map_keys<M>>("keys");
  c.template def_fn<&detail::map_values<M>>("values");
  c.template def_fn<&detail::map_items<M>>("items");
  c.template def_fn<&detail::map_get<M>>("get");
  c.template def_fn<&detail::map_get_default<M>>("get");
  c.template def_fn<&detail::map_clear<M>>("clear");
  c.template def_fn<&detail::map_eq<M>>("__eq__");

  {
    std::string base = name;
    const std::string keys_name = base + "KeysIter";
    const std::string values_name = base + "ValuesIter";
    const std::string items_name = base + "ItemsIter";
    detail::register_map_iterator<M, detail::map_iter_kind::keys>(
        m, keys_name.c_str());
    detail::register_map_iterator<M, detail::map_iter_kind::values>(
        m, values_name.c_str());
    detail::register_map_iterator<M, detail::map_iter_kind::items>(
        m, items_name.c_str());
  }
  return c;
}

} // namespace pyb26
