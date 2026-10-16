#pragma once

#include "argument_admission.hpp"
#include "argument_plan.hpp"
#include "dict_snapshot.hpp"

#include <array>
#include <cassert>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace pyb26 {

#define PYB26_OWNING_AS(member)                                                \
  using projection_implementation = type_caster;                               \
  static constexpr argument_source storage_source =                            \
      argument_source::caster_value;                                           \
  static constexpr bool input_construction_inert =                             \
      detail::native_default_inert_v<decltype(member)>;                        \
  template <class Arg>                                                         \
  static constexpr bool projection_inert =                                     \
      std::is_reference_v<Arg> || detail::native_move_inert_v<Arg>;            \
  template <class Arg> decltype(auto) project() noexcept {                     \
    if constexpr (std::is_reference_v<Arg>)                                    \
      return static_cast<Arg>(member);                                         \
    else                                                                       \
      return std::move(member);                                                \
  }                                                                            \
  template <class Arg> decltype(auto) as() {                                   \
    if constexpr (std::is_reference_v<Arg>)                                    \
      return static_cast<Arg>(member);                                         \
    else                                                                       \
      return Arg(std::move(member));                                           \
  }

#define PYB26_NO_VIEW_ELEMENTS(E)                                              \
  static_assert(                                                               \
      !caster_requires_lease_v<std::remove_cvref_t<E>>,                        \
      "borrowed array view requires a live lease and cannot be stored here");  \
  static_assert(!caster_yields_view_v<std::remove_cvref_t<E>>,                 \
                "pyb26: an owning container of views (std::string_view / "     \
                "const char* / PyObject*) would dangle: the container "        \
                "outlives the temporaries the views point into. Use owned "    \
                "elements (e.g. std::string) instead.")

namespace detail {

template <class T>
struct input_caster_cleanup_shape<std::vector<T>>
    : std::bool_constant<input_caster_cleanup_inert_v<T> &&
                         native_cleanup_inert_v<T>> {};
template <class T>
struct input_caster_cleanup_shape<std::list<T>>
    : std::bool_constant<input_caster_cleanup_inert_v<T> &&
                         native_cleanup_inert_v<T>> {};
template <class K, class V>
struct input_caster_cleanup_shape<std::map<K, V>>
    : std::bool_constant<input_caster_cleanup_inert_v<K> &&
                         input_caster_cleanup_inert_v<V> &&
                         native_cleanup_inert_v<std::map<K, V>>> {};
template <class K, class V>
struct input_caster_cleanup_shape<std::unordered_map<K, V>>
    : std::bool_constant<input_caster_cleanup_inert_v<K> &&
                         input_caster_cleanup_inert_v<V> &&
                         native_cleanup_inert_v<std::unordered_map<K, V>>> {};
template <class T, std::size_t N>
struct input_caster_cleanup_shape<std::array<T, N>>
    : std::bool_constant<input_caster_cleanup_inert_v<T> &&
                         native_cleanup_inert_v<T>> {};
template <class A, class B>
struct input_caster_cleanup_shape<std::pair<A, B>>
    : std::bool_constant<
          input_caster_cleanup_inert_v<A> && input_caster_cleanup_inert_v<B> &&
          native_cleanup_inert_v<A> && native_cleanup_inert_v<B>> {};
template <class... T>
struct input_caster_cleanup_shape<std::tuple<T...>>
    : std::bool_constant<(
          (input_caster_cleanup_inert_v<T> && native_cleanup_inert_v<T>) &&
          ...)> {};
template <class T>
struct input_caster_cleanup_shape<std::optional<T>>
    : std::bool_constant<input_caster_cleanup_inert_v<T> &&
                         native_cleanup_inert_v<T>> {};
template <class... T>
struct input_caster_cleanup_shape<std::variant<T...>>
    : std::bool_constant<(
          (input_caster_cleanup_inert_v<T> && native_cleanup_inert_v<T>) &&
          ...)> {};

template <class T>
inline constexpr bool stl_caster_cleanup_needs_checkpoint =
    !input_caster_cleanup_inert_v<T>;

template <class T>
inline constexpr bool stl_inert_builtin_conversion = [] {
  if constexpr (input_caster_available<T>)
    return input_argument_plan<T>::direct_projection &&
           input_argument_plan<T>::construction_inert &&
           input_argument_plan<T>::parameter_inert &&
           !stl_caster_cleanup_needs_checkpoint<T>;
  return false;
}();

template <class T>
inline constexpr bool stl_element_load_inert = [] {
  if constexpr (input_caster_available<T>)
    return input_argument_plan<T>::construction_inert &&
           !input_argument_plan<T>::load_runs_python &&
           input_argument_plan<T>::extraction_inert &&
           input_argument_plan<T>::parameter_inert &&
           input_argument_plan<T>::cleanup_inert;
  return false;
}();

template <class T>
inline constexpr bool stl_element_assignment_inert = [] {
  if constexpr (input_caster_available<T>)
    return native_assignment_inert_v<
        T, typename input_argument_plan<T>::forwarded_type>;
  return false;
}();

template <class Map>
inline constexpr bool stl_key_operations_inert = [] {
  using Key = typename Map::key_type;
  if constexpr (requires { typename Map::key_compare; })
    return native_key_operation_inert<typename Map::key_compare, Key>::value;
  else
    return native_key_operation_inert<typename Map::hasher, Key>::value &&
           native_key_operation_inert<typename Map::key_equal, Key>::value;
}();

template <class T>
  requires stl_inert_builtin_conversion<T>
decltype(auto) stl_builtin_argument(caster_for<T> &caster) noexcept {
  return project_input<T>(caster);
}

template <class T> struct stl_dict_node_argument {
  caster_for<T> caster;
  bool load(PyObject *object) noexcept { return caster.load(object); }
  auto constructor_args() noexcept {
    return std::tuple<T>{project_input<T>(caster)};
  }
};
template <> struct stl_dict_node_argument<std::string> {
  const char *data;
  Py_ssize_t size;
  bool load(PyObject *object) noexcept {
    data = PyUnicode_AsUTF8AndSize(object, &size);
    if (!data) {
      if (!PyUnicode_Check(object))
        PyErr_Clear();
      return false;
    }
    return true;
  }
  auto constructor_args() const noexcept {
    return std::tuple<const char *, std::size_t>{
        data, static_cast<std::size_t>(size)};
  }
};

template <class Map, class Key>
void stl_try_emplace_string(
    Map &output, Key &&key,
    const std::tuple<const char *, std::size_t> &value_args) noexcept {
  const auto &[data, size] = value_args;
  output.try_emplace(std::forward<Key>(key), data, size);
}

template <class T, class Push>
PYB26_HOT bool stl_convert_item(PyObject *item, callback_error_scope *error,
                                Push &&push) noexcept {
  if constexpr (stl_inert_builtin_conversion<T>) {
    caster_for<T> ec;
    return ec.load(item) && push(project_input<T>(ec));
  } else {
    caster_for<T> ec;
    callback_error_checkpoint caster_checkpoint{*error};
    if (error->capture() || !ec.load(item) || error->capture())
      return false;
    using Plan = input_argument_plan<T>;
    typename Plan::stored_type value = project_input<T>(ec);
    callback_error_checkpoint value_checkpoint{*error};
    return !error->capture() &&
           push(static_cast<typename Plan::stored_type &&>(value));
  }
}

PYB26_HOT inline bool stl_finish_item(bool loaded, PyObject *item,
                                      callback_error_scope &error) noexcept {
  if (!loaded || error.capture()) [[unlikely]] {
    error.discard(item);
    return false;
  }
  Py_DECREF(item);
  return !error.capture();
}

PYB26_COLD inline PyObject *
stl_output_failure(PyObject *result = nullptr, PyObject *first = nullptr,
                   PyObject *second = nullptr) noexcept {
  if (!PyErr_Occurred())
    PyErr_SetString(PyExc_SystemError,
                    "pyb26: STL caster failed without a Python error");
  callback_error_scope error;
  error.discard(first);
  error.discard(second);
  error.discard(result);
  return nullptr;
}

class stl_output_buffer {
  static constexpr std::size_t inline_capacity = 32;
  callback_error_scope error_;
  PyObject *inline_[inline_capacity];
  std::unique_ptr<PyObject *[]> heap_;
  PyObject **items_ = inline_;
  std::size_t size_ = 0;

public:
  stl_output_buffer() noexcept = default;
  stl_output_buffer(const stl_output_buffer &) = delete;
  ~stl_output_buffer() { clear(); }
  bool prepare(std::size_t count) noexcept {
    if (count > inline_capacity) {
      if (count >
          std::numeric_limits<std::size_t>::max() / sizeof(PyObject *)) {
        PyErr_NoMemory();
        return false;
      }
      heap_.reset(new (std::nothrow) PyObject *[count]);
      if (!heap_) {
        PyErr_NoMemory();
        return false;
      }
      items_ = heap_.get();
    }
    return !error_.capture();
  }
  template <class T> bool append(const T &value) noexcept {
    PyObject *item = caster_for<T>::cast(value);
    if (!item) {
      stl_output_failure();
      error_.capture();
      return false;
    }
    items_[size_++] = item;
    if constexpr (!builtin_output_cast<T>)
      return !error_.capture();
    return true;
  }
  PyObject *operator[](std::size_t index) const noexcept {
    return items_[index];
  }
  std::size_t size() const noexcept { return size_; }
  void clear() noexcept {
    const auto size = std::exchange(size_, 0);
    for (std::size_t i = 0; i < size; ++i)
      error_.discard(items_[i]);
  }
  bool finish() noexcept {
    clear();
    return !error_.capture();
  }
  template <bool Tuple> PyObject *sequence() noexcept {
    PyObject *result = Tuple ? PyTuple_New(static_cast<Py_ssize_t>(size_))
                             : PyList_New(static_cast<Py_ssize_t>(size_));
    if (!result)
      return stl_output_failure();
    for (std::size_t i = 0; i < size_; ++i) {
      if constexpr (Tuple)
        PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(i), items_[i]);
      else
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), items_[i]);
    }
    size_ = 0;
    return result;
  }
};

template <class T> PyObject *stl_cast_scalar(const T &value) noexcept {
  PyObject *result = caster_for<T>::cast(value);
  if constexpr (!builtin_output_cast<T>)
    if (!result || PyErr_Occurred())
      return stl_output_failure(result);
  return result;
}

template <class T, class Slot>
PYB26_HOT bool stl_load_slot(PyObject *object, Slot &slot,
                             callback_error_scope &error) noexcept {
  const bool loaded = stl_convert_item<T>(object, &error, [&](auto &&value) {
    slot = std::forward<decltype(value)>(value);
    return true;
  });
  return !error.capture() && loaded;
}

template <class Inner, class It>
PYB26_HOT bool fill_list_from_range(PyObject *lst, It first, It last) noexcept {
#if !defined(Py_LIMITED_API)
  PyObject **slot = reinterpret_cast<PyListObject *>(lst)->ob_item;
  for (; first != last; ++first, ++slot) {
    PyObject *p = Inner::cast(*first);
    if (!p) [[unlikely]]
      return false;
    *slot = p;
  }
#else
  Py_ssize_t i = 0;
  for (; first != last; ++first, ++i) {
    PyObject *p = Inner::cast(*first);
    if (!p) [[unlikely]]
      return false;
    PyList_SET_ITEM(lst, i, p);
  }
#endif
  return true;
}

template <class T, class Range>
PYB26_HOT PyObject *stl_cast_list(const Range &value) noexcept {
  if constexpr (builtin_output_cast<T>) {
    PyObject *result = PyList_New(static_cast<Py_ssize_t>(value.size()));
    if (!result)
      return nullptr;
    if (!fill_list_from_range<caster_for<T>>(result, value.begin(),
                                             value.end()))
      return stl_output_failure(result);
    return result;
  } else {
    stl_output_buffer buffer;
    if (!buffer.prepare(value.size()))
      return nullptr;
    for (const auto &item : value)
      if (!buffer.append(item))
        return nullptr;
    return buffer.template sequence<false>();
  }
}

template <class Tuple, std::size_t... I>
PyObject *stl_cast_tuple(const Tuple &value,
                         std::index_sequence<I...>) noexcept {
  if constexpr ((builtin_output_cast<std::tuple_element_t<I, Tuple>> && ...)) {
    PyObject *result = PyTuple_New(sizeof...(I));
    if (!result)
      return nullptr;
    const auto one = [&]<std::size_t Index>() {
      using T = std::tuple_element_t<Index, Tuple>;
      PyObject *item = caster_for<T>::cast(std::get<Index>(value));
      if (!item)
        return false;
      PyTuple_SET_ITEM(result, Index, item);
      return true;
    };
    if (!(one.template operator()<I>() && ...))
      return stl_output_failure(result);
    return result;
  } else {
    stl_output_buffer buffer;
    if (!buffer.prepare(sizeof...(I)) ||
        !(buffer.append(std::get<I>(value)) && ...))
      return nullptr;
    return buffer.template sequence<true>();
  }
}

struct stl_no_reserve {
  constexpr bool operator()(std::size_t) const noexcept { return true; }
};

template <class Push, class Reserve>
PYB26_HOT bool stl_for_each_owned_list(PyObject *o, Push &&push,
                                       Reserve &&reserve) noexcept {
  PyObject *snapshot = PyList_AsTuple(o);
  if (!snapshot)
    return false;
  callback_error_scope error;
  const auto size = PyTuple_GET_SIZE(snapshot);
  bool ok = reserve(static_cast<std::size_t>(size)) && !error.capture();
  for (Py_ssize_t i = 0; ok && i < size; ++i) {
    ok = push(PyTuple_GET_ITEM(snapshot, i), &error);
    if (error.capture())
      ok = false;
  }
  error.discard(snapshot);
  return !error.capture() && ok;
}

template <class T>
inline constexpr bool stl_native_list_element =
    stl_inert_builtin_conversion<T> && snapshot_value<T>::enabled &&
    (std::is_integral_v<T> || std::is_same_v<T, double>);

template <class T> struct stl_list_load_result {
  std::vector<T> value;
  bool loaded;
};

class stl_list_owned_tail {
  static constexpr std::size_t inline_capacity = 16;
  std::unique_ptr<PyObject *[]> heap_;
  std::size_t capacity_ = inline_capacity;
  std::size_t size_ = 0;
  PyObject *inline_[inline_capacity];

  PyObject **backing() noexcept { return heap_ ? heap_.get() : inline_; }

  void clear(callback_error_scope &error) noexcept {
    auto size = std::exchange(size_, 0);
    auto **items = backing();
    while (size)
      error.discard(items[--size]);
  }

public:
  stl_list_owned_tail() noexcept {}
  stl_list_owned_tail(const stl_list_owned_tail &) = delete;
  ~stl_list_owned_tail() {
    if (size_) {
      callback_error_scope error;
      clear(error);
    }
  }
  std::size_t capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return size_ == 0; }
  bool reserve(std::size_t count) noexcept {
    assert(empty());
    if (count <= capacity_)
      return true;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(PyObject *)) {
      PyErr_NoMemory();
      return false;
    }
    auto next =
        std::unique_ptr<PyObject *[]>(new (std::nothrow) PyObject *[count]);
    if (!next) {
      PyErr_NoMemory();
      return false;
    }
    heap_ = std::move(next);
    capacity_ = count;
    return true;
  }
  void capture(PyObject *const *first, std::size_t count) noexcept {
    assert(empty() && count <= capacity_);
    auto **items = backing();
    for (std::size_t i = 0; i < count; ++i)
      items[i] = Py_NewRef(first[i]);
    size_ = count;
  }
  template <class T>
    requires stl_native_list_element<T>
  PYB26_NOINLINE stl_list_load_result<T>
  convert(std::vector<T> value) noexcept {
    callback_error_scope error;
    bool loaded = !error.capture();
    auto **items = backing();
    for (std::size_t i = 0; loaded && i < size_; ++i) {
      loaded = stl_convert_item<T>(items[i], &error, [&](auto &&item) {
        value.push_back(std::forward<decltype(item)>(item));
        return true;
      });
      if (error.capture())
        loaded = false;
    }
    clear(error);
    return {std::move(value), !error.capture() && loaded};
  }
};

class stl_list_snapshot_lock {
#ifdef Py_GIL_DISABLED
  PyCriticalSection section_;
  bool held_ = false;
#endif

public:
  explicit stl_list_snapshot_lock(PyObject *source) noexcept {
    acquire(source);
  }
  stl_list_snapshot_lock(const stl_list_snapshot_lock &) = delete;
  ~stl_list_snapshot_lock() { release(); }
  void acquire([[maybe_unused]] PyObject *source) noexcept {
#ifdef Py_GIL_DISABLED
    assert(!held_);
    PyCriticalSection_Begin(&section_, source);
    held_ = true;
#endif
  }
  void release() noexcept {
#ifdef Py_GIL_DISABLED
    if (std::exchange(held_, false))
      PyCriticalSection_End(&section_);
#endif
  }
};

struct stl_list_native_prefix {
  PyObject *const *next;
  std::size_t remaining;
};

template <class T>
  requires stl_native_list_element<T>
PYB26_ALWAYS_INLINE stl_list_native_prefix stl_capture_native_list_prefix(
    PyObject *source, std::size_t count, std::vector<T> &output) noexcept {
  using codec = snapshot_value<T>;
  auto *cursor = cpython::list_items(source);
  while (count) {
    typename codec::probe_type value;
    if (!codec::probe(*cursor, value))
      break;
    output.push_back(codec::materialize(value, nullptr));
    ++cursor;
    --count;
  }
  return {cursor, count};
}

template <class T>
  requires stl_native_list_element<T>
PYB26_NOINLINE stl_list_load_result<T>
stl_load_coercing_list(PyObject *source, stl_list_native_prefix suffix,
                       std::vector<T> output,
                       stl_list_snapshot_lock &lock) noexcept {
  [[indeterminate]] stl_list_owned_tail tail;
  if (suffix.remaining <= tail.capacity()) {
    tail.capture(suffix.next, suffix.remaining);
    lock.release();
    return tail.convert<T>(std::move(output));
  }
  lock.release();
  auto required_tail = suffix.remaining;
  auto required = static_cast<std::size_t>(PyList_GET_SIZE(source));
  for (;;) {
    output.clear();
    if (!tail.reserve(required_tail))
      return {std::move(output), false};
    if (required > output.capacity())
      output.reserve(required);
    lock.acquire(source);
    required = static_cast<std::size_t>(PyList_GET_SIZE(source));
    if (required > output.capacity()) {
      lock.release();
      continue;
    }
    suffix = stl_capture_native_list_prefix(source, required, output);
    if (suffix.remaining <= tail.capacity()) {
      tail.capture(suffix.next, suffix.remaining);
      lock.release();
      if (tail.empty())
        return {std::move(output), true};
      return tail.convert<T>(std::move(output));
    }
    required_tail = suffix.remaining;
    lock.release();
  }
}

template <class T>
  requires stl_native_list_element<T>
PYB26_HOT stl_list_load_result<T>
stl_load_native_list(PyObject *o, std::vector<T> &&incoming) noexcept {
  std::vector<T> output = std::move(incoming);
  assert(PyList_Check(o) && output.empty());
  auto required = static_cast<std::size_t>(PyList_GET_SIZE(o));
  for (;;) {
    if (required > output.capacity())
      output.reserve(required);
    stl_list_snapshot_lock lock(o);
    required = static_cast<std::size_t>(PyList_GET_SIZE(o));
    if (required > output.capacity()) {
      lock.release();
      continue;
    }
    const auto suffix = stl_capture_native_list_prefix(o, required, output);
    if (suffix.remaining)
      return stl_load_coercing_list(o, suffix, std::move(output), lock);
    lock.release();
    return {std::move(output), true};
  }
}

template <class Elem, class Push, class Reserve = stl_no_reserve>
PYB26_HOT bool stl_for_each(PyObject *o, Push &&push,
                            Reserve reserve = {}) noexcept {
  constexpr bool quiet =
      stl_inert_builtin_conversion<Elem> &&
      !caster_load_runs_python<std::remove_cvref_t<Elem>> &&
      (std::is_arithmetic_v<Elem> || std::is_same_v<Elem, std::string>);
  if (PyTuple_CheckExact(o)) {
    callback_error_scope error;
    const auto size = PyTuple_GET_SIZE(o);
    if (!reserve(static_cast<std::size_t>(size)) || error.capture())
      return false;
    for (Py_ssize_t i = 0; i < size; ++i)
      if (!push(PyTuple_GET_ITEM(o, i), &error) || error.capture())
        return false;
    return true;
  }
  if (PyList_Check(o)) {
    if constexpr (quiet) {
      const Py_ssize_t n = PyList_GET_SIZE(o);
      if constexpr (!std::is_same_v<Reserve, stl_no_reserve>)
        if (!reserve(static_cast<std::size_t>(n)) || PyErr_Occurred())
          return false;
#ifdef Py_GIL_DISABLED
      callback_error_scope error;
#endif
      for (Py_ssize_t i = 0; i < n; ++i) {
#ifdef Py_GIL_DISABLED
        PyObject *item = PyList_GetItemRef(o, i);
        if (!item) [[unlikely]]
          return false;
        const bool ok = push(item, &error);
        if (!stl_finish_item(ok, item, error)) [[unlikely]]
          return false;
#else
        if (!push(PyList_GET_ITEM(o, i),
                  static_cast<callback_error_scope *>(nullptr))) [[unlikely]]
          return false;
#endif
      }
      return true;
    } else {
      return stl_for_each_owned_list(o, push, reserve);
    }
  }
  if (PyUnicode_Check(o) || PyBytes_Check(o) || PyByteArray_Check(o) ||
      PyDict_Check(o)) [[unlikely]]
    return false;
  if (!PyType_GetSlot(Py_TYPE(o), Py_tp_iter) && !PySequence_Check(o))
      [[unlikely]]
    return false;
  PyObject *it = PyObject_GetIter(o);
  if (!it) [[unlikely]]
    return false;
  callback_error_scope error;
  PyObject *item;
  while ((item = PyIter_Next(it)) != nullptr) {
    const bool ok = push(item, &error);
    if (!stl_finish_item(ok, item, error)) [[unlikely]] {
      error.discard(it);
      return false;
    }
  }
  error.discard(it);
  return !error.capture();
}

template <class T> struct py_hint<std::vector<T>> {
  static void append(std::string &o) {
    o += "list[";
    append_py_hint<T>(o);
    o += "]";
  }
};
template <class T> struct py_hint<std::list<T>> {
  static void append(std::string &o) {
    o += "list[";
    append_py_hint<T>(o);
    o += "]";
  }
};
template <class T> struct py_hint<std::set<T>> {
  static void append(std::string &o) {
    o += "set[";
    append_py_hint<T>(o);
    o += "]";
  }
};
template <class T, std::size_t N> struct py_hint<std::array<T, N>> {
  static void append(std::string &o) {
    o += "list[";
    append_py_hint<T>(o);
    o += "]";
  }
};
template <class K, class V> struct py_hint<std::map<K, V>> {
  static void append(std::string &o) {
    o += "dict[";
    append_py_hint<K>(o);
    o += ", ";
    append_py_hint<V>(o);
    o += "]";
  }
};
template <class K, class V> struct py_hint<std::unordered_map<K, V>> {
  static void append(std::string &o) {
    o += "dict[";
    append_py_hint<K>(o);
    o += ", ";
    append_py_hint<V>(o);
    o += "]";
  }
};
template <class T> struct py_hint<std::optional<T>> {
  static void append(std::string &o) {
    append_py_hint<T>(o);
    o += " | None";
  }
};
template <class A, class B> struct py_hint<std::pair<A, B>> {
  static void append(std::string &o) {
    o += "tuple[";
    append_py_hint<A>(o);
    o += ", ";
    append_py_hint<B>(o);
    o += "]";
  }
};
template <class... Ts> struct py_hint<std::tuple<Ts...>> {
  static void append(std::string &o) {
    o += "tuple[";
    bool first = true;
    ((o += (first ? (first = false, "") : ", "), append_py_hint<Ts>(o)), ...);
    o += "]";
  }
};
template <class... Ts> struct py_hint<std::variant<Ts...>> {
  static void append(std::string &o) {
    bool first = true;
    ((o += (first ? (first = false, "") : " | "), append_py_hint<Ts>(o)), ...);
  }
};

} // namespace detail

template <typename T> struct type_caster<std::optional<T>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects = [] {
    auto result = caster_contract<T>;
    result.runs_python = !detail::stl_element_load_inert<T>;
    return result;
  }();
  [[no_unique_address]] detail::argument_owner_admission<
      effects.requires_admission> admission_;
  static_assert(
      !effects.requires_lease,
      "borrowed array view requires a live lease and cannot be stored here");
  using inner = type_caster<std::remove_cvref_t<T>>;
  std::optional<T> value_;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (!admission_.acquire())
      return false;
    if constexpr (detail::stl_inert_builtin_conversion<T>) {
      if (o == Py_None) {
        value_ = std::nullopt;
        return true;
      }
      inner ec;
      if (!ec.load(o)) [[unlikely]]
        return false;
      value_.emplace(detail::project_input<T>(ec));
      return true;
    } else {
      detail::callback_error_scope error;
      if (o == Py_None) {
        value_.reset();
        return !error.capture();
      }
      return detail::stl_load_slot<T>(o, value_, error);
    }
  }
  PYB26_OWNING_AS(value_)
  static PyObject *cast(const std::optional<T> &v) {
    if (!v)
      return Py_NewRef(Py_None);
    return detail::stl_cast_scalar(*v);
  }
};

template <typename T>
  requires(!opaque_v<std::vector<T>>)
struct type_caster<std::vector<T>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{
      true, false, caster_requires_lease_v<std::remove_cvref_t<T>>,
      caster_contract<T>.owns_python, caster_contract<T>.requires_admission};
  [[no_unique_address]] detail::argument_owner_admission<
      effects.requires_admission> admission_;
  PYB26_NO_VIEW_ELEMENTS(T);
  using inner = type_caster<std::remove_cvref_t<T>>;
  std::vector<T> value_;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (!admission_.acquire())
      return false;
    value_.clear();
    if constexpr (detail::stl_native_list_element<T>)
      if (PyList_Check(o)) [[likely]] {
        auto result = detail::stl_load_native_list(o, std::move(value_));
        value_ = std::move(result.value);
        return result.loaded;
      }
    auto push = [this](PyObject *item, auto *error) {
      return detail::stl_convert_item<T>(item, error, [this](auto &&value) {
        value_.push_back(std::forward<decltype(value)>(value));
        return true;
      });
    };
    auto reserve = [this](std::size_t size) {
      value_.reserve(size);
      return true;
    };
    if constexpr (!detail::caster_load_runs_python<std::remove_cvref_t<T>> &&
                  (std::is_trivially_copyable_v<T> ||
                   std::is_same_v<T, std::string>)
#ifndef Py_GIL_DISABLED
                  && detail::stl_inert_builtin_conversion<T>
#endif
    )
      if (PyList_Check(o)) [[likely]] {
        const Py_ssize_t n = PyList_GET_SIZE(o);
        value_.reserve(static_cast<size_t>(n));
#ifdef Py_GIL_DISABLED
        detail::callback_error_scope error;
#endif
        for (Py_ssize_t i = 0; i < n; ++i) {
#ifdef Py_GIL_DISABLED
          PyObject *item = PyList_GetItemRef(o, i);
          if (!item) [[unlikely]]
            return false;
          const bool loaded =
              detail::stl_convert_item<T>(item, &error, [this](auto &&value) {
                value_.push_back(std::forward<decltype(value)>(value));
                return true;
              });
          if (!detail::stl_finish_item(loaded, item, error)) [[unlikely]]
            return false;
#else
          inner ec;
          if (!ec.load(PyList_GET_ITEM(o, i))) [[unlikely]]
            return false;
          value_.push_back(detail::project_input<T>(ec));
#endif
        }
        return true;
      }
    return detail::stl_for_each<T>(o, push, reserve);
  }
  PYB26_OWNING_AS(value_)
  static PyObject *cast(const std::vector<T> &v) {
    return detail::stl_cast_list<T>(v);
  }
};

template <typename T>
  requires(!opaque_v<std::list<T>>)
struct type_caster<std::list<T>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{
      true, false, caster_requires_lease_v<std::remove_cvref_t<T>>,
      caster_contract<T>.owns_python, caster_contract<T>.requires_admission};
  [[no_unique_address]] detail::argument_owner_admission<
      effects.requires_admission> admission_;
  PYB26_NO_VIEW_ELEMENTS(T);
  using inner = type_caster<std::remove_cvref_t<T>>;
  std::list<T> value_;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (!admission_.acquire())
      return false;
    value_.clear();
    return detail::stl_for_each<T>(o, [this](PyObject *item, auto *error) {
      return detail::stl_convert_item<T>(item, error, [this](auto &&value) {
        value_.push_back(std::forward<decltype(value)>(value));
        return true;
      });
    });
  }
  PYB26_OWNING_AS(value_)
  static PyObject *cast(const std::list<T> &v) {
    return detail::stl_cast_list<T>(v);
  }
};

template <typename T>
  requires(!opaque_v<std::set<T>>)
struct type_caster<std::set<T>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{
      true, false, caster_requires_lease_v<std::remove_cvref_t<T>>,
      caster_contract<T>.owns_python, caster_contract<T>.requires_admission};
  [[no_unique_address]] detail::argument_owner_admission<
      effects.requires_admission> admission_;
  PYB26_NO_VIEW_ELEMENTS(T);
  using inner = type_caster<std::remove_cvref_t<T>>;
  std::set<T> value_;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (!admission_.acquire())
      return false;
    value_.clear();
    return detail::stl_for_each<T>(o, [this](PyObject *item, auto *error) {
      return detail::stl_convert_item<T>(item, error, [this](auto &&value) {
        value_.insert(std::forward<decltype(value)>(value));
        return true;
      });
    });
  }
  PYB26_OWNING_AS(value_)
  static PyObject *cast(const std::set<T> &v) {
    if constexpr (!detail::builtin_output_cast<T>) {
      detail::stl_output_buffer buffer;
      if (!buffer.prepare(v.size()))
        return nullptr;
      for (const auto &item : v)
        if (!buffer.append(item))
          return nullptr;
      PyObject *result = PySet_New(nullptr);
      if (!result)
        return detail::stl_output_failure();
      for (std::size_t i = 0; i < buffer.size(); ++i) {
        const int rc = PySet_Add(result, buffer[i]);
        if (rc < 0 || PyErr_Occurred())
          return detail::stl_output_failure(result);
      }
      if (!buffer.finish())
        return detail::stl_output_failure(result);
      return result;
    }
    PyObject *s = PySet_New(nullptr);
    if (!s) [[unlikely]]
      return nullptr;
    for (const auto &e : v) {
      PyObject *p = inner::cast(e);
      if (!p) [[unlikely]]
        return detail::stl_output_failure(s);
      const int rc = PySet_Add(s, p);
      if (rc < 0) [[unlikely]]
        return detail::stl_output_failure(s, p);
      Py_DECREF(p);
    }
    return s;
  }
};

template <typename T, std::size_t N>
  requires(!opaque_v<std::array<T, N>>)
struct type_caster<std::array<T, N>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{
      true, false, caster_requires_lease_v<std::remove_cvref_t<T>>,
      caster_contract<T>.owns_python, caster_contract<T>.requires_admission};
  [[no_unique_address]] detail::argument_owner_admission<
      effects.requires_admission> admission_;
  PYB26_NO_VIEW_ELEMENTS(T);
  using inner = type_caster<std::remove_cvref_t<T>>;
  std::array<T, N> value_;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (!admission_.acquire())
      return false;
    std::size_t i = 0;
    const bool ok = detail::stl_for_each<T>(o, [this, &i](PyObject *item,
                                                          auto *error) {
      if (i >= N) [[unlikely]]
        return false;
      return detail::stl_convert_item<T>(item, error, [this, &i](auto &&value) {
        value_[i++] = std::forward<decltype(value)>(value);
        return true;
      });
    });
    if (!ok) [[unlikely]]
      return false;
    return i == N;
  }
  PYB26_OWNING_AS(value_)
  static PyObject *cast(const std::array<T, N> &v) {
    return detail::stl_cast_list<T>(v);
  }
};

template <typename A, typename B> struct type_caster<std::pair<A, B>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{
      !detail::stl_element_load_inert<A> ||
          !detail::stl_element_load_inert<B> ||
          !detail::stl_element_assignment_inert<A> ||
          !detail::stl_element_assignment_inert<B>,
      caster_contract<A>.yields_view || caster_contract<B>.yields_view,
      caster_contract<A>.requires_lease || caster_contract<B>.requires_lease,
      caster_contract<A>.owns_python || caster_contract<B>.owns_python,
      caster_contract<A>.requires_admission ||
          caster_contract<B>.requires_admission};
  [[no_unique_address]] detail::argument_owner_admission<
      effects.requires_admission> admission_;
  static_assert(
      !effects.requires_lease,
      "borrowed array view requires a live lease and cannot be stored here");
  using ca = type_caster<std::remove_cvref_t<A>>;
  using cb = type_caster<std::remove_cvref_t<B>>;
  std::pair<A, B> value_;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (!admission_.acquire())
      return false;
    if (!PyTuple_Check(o) || PyTuple_GET_SIZE(o) != 2) [[unlikely]]
      return false;
    if constexpr (detail::stl_inert_builtin_conversion<A> &&
                  detail::stl_inert_builtin_conversion<B> &&
                  detail::stl_element_assignment_inert<A> &&
                  detail::stl_element_assignment_inert<B>) {
      ca ka;
      cb kb;
      if (!ka.load(PyTuple_GET_ITEM(o, 0))) [[unlikely]]
        return false;
      if (!kb.load(PyTuple_GET_ITEM(o, 1))) [[unlikely]]
        return false;
      value_.first = detail::project_input<A>(ka);
      value_.second = detail::project_input<B>(kb);
      return true;
    } else {
      detail::callback_error_scope error;
      return detail::stl_load_slot<A>(PyTuple_GET_ITEM(o, 0), value_.first,
                                      error) &&
             detail::stl_load_slot<B>(PyTuple_GET_ITEM(o, 1), value_.second,
                                      error);
    }
  }
  PYB26_OWNING_AS(value_)
  static PyObject *cast(const std::pair<A, B> &v) {
    return detail::stl_cast_tuple(v, std::index_sequence<0, 1>{});
  }
};

template <typename... Ts> struct type_caster<std::tuple<Ts...>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{
      ((!detail::stl_element_load_inert<Ts> ||
        !detail::stl_element_assignment_inert<Ts>) ||
       ...),
      (caster_contract<Ts>.yields_view || ...),
      (caster_contract<Ts>.requires_lease || ...),
      (caster_contract<Ts>.owns_python || ...),
      (caster_contract<Ts>.requires_admission || ...)};
  [[no_unique_address]] detail::argument_owner_admission<
      effects.requires_admission> admission_;
  static_assert(
      !effects.requires_lease,
      "borrowed array view requires a live lease and cannot be stored here");
  static constexpr std::size_t N = sizeof...(Ts);
  std::tuple<Ts...> value_;

  PYB26_HOT bool load(PyObject *o) noexcept {
    if (!admission_.acquire())
      return false;
    if (!PyTuple_Check(o) || static_cast<std::size_t>(PyTuple_GET_SIZE(o)) != N)
        [[unlikely]]
      return false;
    return load_impl(o, std::index_sequence_for<Ts...>{});
  }
  PYB26_OWNING_AS(value_)
  static PyObject *cast(const std::tuple<Ts...> &v) {
    return detail::stl_cast_tuple(v, std::index_sequence_for<Ts...>{});
  }

private:
  template <std::size_t... I>
  PYB26_HOT bool load_impl([[maybe_unused]] PyObject *o,
                           std::index_sequence<I...>) noexcept {
    if constexpr (((detail::stl_inert_builtin_conversion<Ts> &&
                    detail::stl_element_assignment_inert<Ts>) &&
                   ...)) {
      return (load_one<I, std::tuple_element_t<I, std::tuple<Ts...>>>(o) &&
              ...);
    } else {
      detail::callback_error_scope error;
      return (detail::stl_load_slot<std::tuple_element_t<I, std::tuple<Ts...>>>(
                  PyTuple_GET_ITEM(o, I), std::get<I>(value_), error) &&
              ...);
    }
  }
  template <std::size_t I, typename E>
  PYB26_HOT bool load_one(PyObject *o) noexcept {
    type_caster<std::remove_cvref_t<E>> ec;
    if (!ec.load(PyTuple_GET_ITEM(o, static_cast<Py_ssize_t>(I)))) [[unlikely]]
      return false;
    std::get<I>(value_) = detail::project_input<E>(ec);
    return true;
  }
};

namespace detail {

class stl_dict_snapshot {
  static constexpr Py_ssize_t inline_capacity = 16;
  PyObject *inline_[2 * inline_capacity];
  std::unique_ptr<PyObject *[]> heap_;
  Py_ssize_t count_ = 0;

  PyObject **items() noexcept { return heap_ ? heap_.get() : inline_; }
  PyObject *const *items() const noexcept {
    return heap_ ? heap_.get() : inline_;
  }

public:
  class reader {
    PyObject *const *const records_;
    PyObject *const *const end_;
    friend class stl_dict_snapshot;
    reader(PyObject *const *records, Py_ssize_t count) noexcept
        : records_(records), end_(records + 2 * count) {}

  public:
    PyObject *const *first() const noexcept { return records_; }
    PyObject *const *last() const noexcept { return end_; }
    static PyObject *const *next(PyObject *const *row) noexcept {
      return row + 2;
    }
    PyObject *key(PyObject *const *row) const noexcept { return row[0]; }
    PyObject *value(PyObject *const *row) const noexcept { return row[1]; }
  };

  stl_dict_snapshot() noexcept = default;
  stl_dict_snapshot(const stl_dict_snapshot &) = delete;
  ~stl_dict_snapshot() { clear(); }
  void clear() noexcept {
    if (!count_)
      return;
    callback_error_scope error;
    const auto count = std::exchange(count_, 0);
    auto *records = items();
    Py_ssize_t i = 0;
    if (!error.capture()) {
      while (i < 2 * count) {
        Py_DECREF(records[i++]);
        if (error.capture()) [[unlikely]]
          break;
      }
    }
    for (; i < 2 * count; ++i)
      error.discard(records[i]);
  }
  bool acquire(PyObject *source) noexcept {
    Py_ssize_t capacity = inline_capacity;
    for (;;) {
      Py_ssize_t required;
      Py_BEGIN_CRITICAL_SECTION(source);
      required = PyDict_GET_SIZE(source);
      if (required <= capacity) {
        auto *records = items();
        Py_ssize_t position = 0;
        PyObject *key, *value;
        while (PyDict_Next(source, &position, &key, &value)) {
          records[2 * count_] = Py_NewRef(key);
          records[2 * count_ + 1] = Py_NewRef(value);
          ++count_;
        }
      }
      Py_END_CRITICAL_SECTION();
      if (required <= capacity)
        return true;
      constexpr auto limit =
          std::numeric_limits<std::size_t>::max() / (2 * sizeof(PyObject *));
      if (static_cast<std::size_t>(required) > limit) {
        PyErr_NoMemory();
        return false;
      }
      auto next = std::unique_ptr<PyObject *[]>(new (
          std::nothrow) PyObject *[2 * static_cast<std::size_t>(required)]);
      if (!next) {
        PyErr_NoMemory();
        return false;
      }
      heap_ = std::move(next);
      capacity = required;
    }
  }
  Py_ssize_t size() const noexcept { return count_; }
  reader read() const noexcept { return {items(), count_}; }
};

template <typename K, typename V, class Map>
PYB26_HOT bool stl_load_dict(PyObject *o, Map &output) noexcept {
  assert(output.empty());
  if (!PyDict_Check(o)) [[unlikely]]
    return false;
  callback_error_scope error;
  constexpr bool native_insert =
      stl_inert_builtin_conversion<K> && stl_inert_builtin_conversion<V> &&
      stl_key_operations_inert<Map> &&
      (std::is_same_v<Map, std::map<K, V>> ||
       std::is_same_v<Map, std::unordered_map<K, V>>);
  constexpr bool native_snapshot =
      native_insert && snapshot_value<K>::enabled && snapshot_value<V>::enabled;
  constexpr bool prepared_mapped_string =
      native_insert && std::is_same_v<std::remove_cvref_t<V>, std::string>;
  [[indeterminate]] std::conditional_t<
      native_snapshot, native_dict_snapshot<K, V>, stl_dict_snapshot> snapshot;
  if (!snapshot.acquire(o))
    return false;
  if constexpr (requires { output.reserve(std::size_t{}); }) {
    output.reserve(static_cast<std::size_t>(snapshot.size()));
    if constexpr (!native_insert)
      if (error.capture())
        return false;
  }
  if constexpr (native_snapshot) {
    if (snapshot.all_native()) {
      const auto input = snapshot.read();
      for (auto row = input.first(); row != input.last();
           row = input.next(row)) {
        if constexpr (std::is_same_v<K, std::string>) {
          auto key_args = input.native_key_args(row);
          auto value_args = input.native_value_args(row);
          output.emplace(std::piecewise_construct, std::move(key_args),
                         std::move(value_args));
        } else {
          auto key = input.native_key(row);
          if constexpr (prepared_mapped_string) {
            auto value_args = input.native_value_args(row);
            stl_try_emplace_string(output, std::move(key), value_args);
          } else {
            auto value = input.native_value(row);
            output.emplace(std::move(key), std::move(value));
          }
        }
      }
      return !error.capture();
    }
  }
  const auto load_pairs = [&]<bool check_native, bool direct_node>(
                              const auto input) {
    bool loaded = true;
    [[maybe_unused]] Py_ssize_t index = 0;
    for (auto row = input.first(); loaded && row != input.last();
         row = input.next(row)) {
      if constexpr (check_native) {
        if (input.is_native(index++)) {
          if constexpr (direct_node) {
            auto key_args = input.native_key_args(row);
            auto value_args = input.native_value_args(row);
            output.emplace(std::piecewise_construct, std::move(key_args),
                           std::move(value_args));
          } else {
            auto key = input.native_key(row);
            if constexpr (prepared_mapped_string) {
              auto value_args = input.native_value_args(row);
              stl_try_emplace_string(output, std::move(key), value_args);
            } else {
              auto value = input.native_value(row);
              output.emplace(std::move(key), std::move(value));
            }
          }
          continue;
        }
      }
      if constexpr (prepared_mapped_string && !direct_node) {
        caster_for<K> kc;
        stl_dict_node_argument<std::remove_cvref_t<V>> vc;
        loaded = kc.load(input.key(row));
        if (loaded) {
          decltype(auto) key = stl_builtin_argument<K>(kc);
          loaded = vc.load(input.value(row));
          if (loaded)
            stl_try_emplace_string(output, std::forward<decltype(key)>(key),
                                   vc.constructor_args());
        }
      } else if constexpr (stl_inert_builtin_conversion<K> &&
                           stl_inert_builtin_conversion<V>) {
        std::conditional_t<direct_node, stl_dict_node_argument<K>,
                           caster_for<K>>
            kc;
        std::conditional_t<direct_node, stl_dict_node_argument<V>,
                           caster_for<V>>
            vc;
        loaded = kc.load(input.key(row)) && vc.load(input.value(row));
        if (loaded) {
          if constexpr (direct_node) {
            auto key_args = kc.constructor_args();
            auto value_args = vc.constructor_args();
            output.emplace(std::piecewise_construct, std::move(key_args),
                           std::move(value_args));
          } else {
            output.emplace(stl_builtin_argument<K>(kc),
                           stl_builtin_argument<V>(vc));
          }
        }
      } else {
        loaded = stl_convert_item<K>(input.key(row), &error, [&](auto &&key) {
          return stl_convert_item<V>(
              input.value(row), &error, [&](auto &&value) {
                output.emplace(std::forward<decltype(key)>(key),
                               std::forward<decltype(value)>(value));
                return true;
              });
        });
      }
      if constexpr (!native_insert)
        if (error.capture())
          loaded = false;
    }
    return loaded;
  };
  const auto load_mode = [&]<bool direct_node>() {
    if constexpr (native_snapshot)
      if (!snapshot.all_python())
        return load_pairs.template operator()<true, direct_node>(
            snapshot.read());
    return load_pairs.template operator()<false, direct_node>(snapshot.read());
  };
  const bool loaded = [&] {
    if constexpr (native_snapshot && std::is_same_v<K, std::string>)
      if (snapshot.all_exact_string_keys())
        return load_mode.template operator()<true>();
    return load_mode.template operator()<false>();
  }();
  error.capture();
  snapshot.clear();
  return !error.capture() && loaded;
}

template <class K> PyObject *stl_new_dict(std::size_t count) noexcept {
  if constexpr (builtin_output_cast<K> &&
                std::is_arithmetic_v<std::remove_cvref_t<K>>)
    return cpython::dict_new_presized(count);
  else
    return PyDict_New();
}

template <typename K, typename V, class Map>
PyObject *stl_cast_dict(const Map &m) {
  if constexpr (!builtin_output_cast<K> || !builtin_output_cast<V>) {
    stl_output_buffer buffer;
    if (m.size() > std::numeric_limits<std::size_t>::max() / 2) {
      PyErr_NoMemory();
      return nullptr;
    }
    if (!buffer.prepare(2 * m.size()))
      return nullptr;
    for (const auto &[key, value] : m)
      if (!buffer.append(key) || !buffer.append(value))
        return nullptr;
    const auto count = buffer.size() / 2;
    PyObject *result = stl_new_dict<K>(count);
    if (!result)
      return stl_output_failure();
    for (std::size_t i = 0; i < count; ++i) {
      const int rc = PyDict_SetItem(result, buffer[2 * i], buffer[2 * i + 1]);
      if (rc < 0 || PyErr_Occurred())
        return stl_output_failure(result);
    }
    if (!buffer.finish())
      return stl_output_failure(result);
    return result;
  }
  PyObject *d = stl_new_dict<K>(m.size());
  if (!d) [[unlikely]]
    return nullptr;
  for (const auto &[k, v] : m) {
    PyObject *pk = type_caster<std::remove_cvref_t<K>>::cast(k);
    if (!pk) [[unlikely]]
      return stl_output_failure(d);
    PyObject *pv = type_caster<std::remove_cvref_t<V>>::cast(v);
    if (!pv) [[unlikely]]
      return stl_output_failure(d, pk);
    const int rc = PyDict_SetItem(d, pk, pv);
    if (rc < 0) [[unlikely]]
      return stl_output_failure(d, pk, pv);
    Py_DECREF(pk);
    Py_DECREF(pv);
  }
  return d;
}

} // namespace detail

template <typename K, typename V>
  requires(!opaque_v<std::map<K, V>>)
struct type_caster<std::map<K, V>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{
      true, false,
      caster_requires_lease_v<std::remove_cvref_t<K>> ||
          caster_requires_lease_v<std::remove_cvref_t<V>>,
      caster_contract<K>.owns_python || caster_contract<V>.owns_python,
      caster_contract<K>.requires_admission ||
          caster_contract<V>.requires_admission};
  [[no_unique_address]] detail::argument_owner_admission<
      effects.requires_admission> admission_;
  PYB26_NO_VIEW_ELEMENTS(K);
  PYB26_NO_VIEW_ELEMENTS(V);
  std::map<K, V> value_;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (!admission_.acquire())
      return false;
    value_.clear();
    return detail::stl_load_dict<K, V>(o, value_);
  }
  PYB26_OWNING_AS(value_)
  static PyObject *cast(const std::map<K, V> &m) {
    return detail::stl_cast_dict<K, V>(m);
  }
};

template <typename K, typename V>
  requires(!opaque_v<std::unordered_map<K, V>>)
struct type_caster<std::unordered_map<K, V>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{
      true, false,
      caster_requires_lease_v<std::remove_cvref_t<K>> ||
          caster_requires_lease_v<std::remove_cvref_t<V>>,
      caster_contract<K>.owns_python || caster_contract<V>.owns_python,
      caster_contract<K>.requires_admission ||
          caster_contract<V>.requires_admission};
  [[no_unique_address]] detail::argument_owner_admission<
      effects.requires_admission> admission_;
  PYB26_NO_VIEW_ELEMENTS(K);
  PYB26_NO_VIEW_ELEMENTS(V);
  std::unordered_map<K, V> value_;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (!admission_.acquire())
      return false;
    value_.clear();
    return detail::stl_load_dict<K, V>(o, value_);
  }
  PYB26_OWNING_AS(value_)
  static PyObject *cast(const std::unordered_map<K, V> &m) {
    return detail::stl_cast_dict<K, V>(m);
  }
};

template <typename... Ts> struct type_caster<std::variant<Ts...>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{
      ((!detail::stl_element_load_inert<Ts>) || ...),
      (caster_contract<Ts>.yields_view || ...),
      (caster_contract<Ts>.requires_lease || ...),
      (caster_contract<Ts>.owns_python || ...),
      (caster_contract<Ts>.requires_admission || ...)};
  [[no_unique_address]] detail::argument_owner_admission<
      effects.requires_admission> admission_;
  static_assert(
      !effects.requires_lease,
      "borrowed array view requires a live lease and cannot be stored here");
  std::variant<Ts...> value_;

  PYB26_HOT bool load(PyObject *o) noexcept {
    return admission_.acquire() && try_alts<Ts...>(o);
  }
  PYB26_OWNING_AS(value_)
  static PyObject *cast(const std::variant<Ts...> &v) {
    return std::visit(
        [](const auto &alt) -> PyObject * {
          return detail::stl_cast_scalar(alt);
        },
        v);
  }

private:
  template <typename First, typename... Rest>
  PYB26_HOT bool try_alts(PyObject *o) noexcept {
    if constexpr ((detail::stl_inert_builtin_conversion<Ts> && ...)) {
      type_caster<std::remove_cvref_t<First>> ec;
      if (ec.load(o)) {
        value_.template emplace<First>(detail::project_input<First>(ec));
        return true;
      }
    } else {
      detail::callback_error_scope error;
      const bool loaded =
          detail::stl_convert_item<First>(o, &error, [&](auto &&value) {
            value_.template emplace<First>(
                std::forward<decltype(value)>(value));
            return true;
          });
      if (error.capture())
        return false;
      if (loaded)
        return true;
    }
    if constexpr (sizeof...(Rest) > 0) {
      if (PyErr_Occurred())
        return false;
      return try_alts<Rest...>(o);
    } else {
      return false;
    }
  }
};

#undef PYB26_OWNING_AS
#undef PYB26_NO_VIEW_ELEMENTS

} // namespace pyb26
