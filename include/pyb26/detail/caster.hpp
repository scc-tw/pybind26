#pragma once

#include "../errors.hpp"
#include "common.hpp"
#include "native_cleanup.hpp"

#include <atomic>
#include <concepts>
#include <new>
#include <string>
#include <string_view>
#include <type_traits>

namespace pyb26 {

template <typename T> struct type_caster;

template <typename T> using caster_for = type_caster<std::remove_cvref_t<T>>;

struct caster_effects {
  bool runs_python;
  bool yields_view;
  bool requires_lease;
  bool owns_python = false;
  bool requires_admission = false;
};

enum class argument_source {
  opaque,
  caster_value,
  wrapper_payload,
  input_view,
  leased_view
};

template <class T>
inline constexpr caster_effects caster_contract = [] {
  if constexpr (std::is_void_v<T>) {
    return caster_effects{false, false, false, false};
  } else if constexpr (requires { caster_for<T>::effects; }) {
    static_assert(
        std::same_as<std::remove_cv_t<decltype(caster_for<T>::effects)>,
                     caster_effects>,
        "caster effects must have type pyb26::caster_effects");
    constexpr auto effects = caster_for<T>::effects;
    static_assert(!effects.requires_lease || effects.yields_view,
                  "a caster lease requires a borrowed view");
    return effects;
  } else {
    return caster_effects{true, true, true, true, true};
  }
}();

template <typename C>
concept Caster = requires(C c, PyObject *o) {
  { c.load(o) } noexcept -> std::same_as<bool>;
};

template <class T>
inline constexpr bool caster_yields_view_v = caster_contract<T>.yields_view;
template <class T>
inline constexpr bool caster_requires_lease_v =
    caster_contract<T>.requires_lease;

template <class V> inline constexpr bool opaque_v = false;

namespace detail {

template <class C, argument_source Source = argument_source::caster_value>
struct scalar_input_projection {
  using projection_implementation = C;
  static constexpr argument_source storage_source = Source;
  static constexpr bool input_construction_inert = true;
  template <class A> static constexpr bool projection_inert = true;
  template <class A> decltype(auto) project() noexcept {
    return static_cast<C &>(*this).template as<A>();
  }
};

template <class C> struct owning_input_projection {
  using projection_implementation = C;
  static constexpr argument_source storage_source =
      argument_source::caster_value;
  static constexpr bool input_construction_inert = true;
  template <class A>
  static constexpr bool projection_inert =
      std::is_reference_v<A> || native_move_inert_v<A>;
  template <class A> decltype(auto) project() noexcept {
    using Projection = std::conditional_t<std::is_reference_v<A>, A, A &&>;
    return static_cast<C &>(*this).template as<Projection>();
  }
};

template <class C, class T> struct wrapper_input_projection {
  using projection_implementation = C;
  static constexpr argument_source storage_source =
      argument_source::wrapper_payload;
  static constexpr bool input_construction_inert = true;
  template <class A>
  static constexpr bool projection_inert =
      std::is_reference_v<A> || std::is_pointer_v<A> || native_copy_inert_v<T>;
  template <class A> decltype(auto) project() noexcept {
    if constexpr (std::is_reference_v<A> || std::is_pointer_v<A>)
      return static_cast<C &>(*this).template as<A>();
    else
      return static_cast<C &>(*this).template as<const T &>();
  }
};

template <class T>
inline constexpr bool first_party_input_caster_v = [] {
  if constexpr (requires { typename caster_for<T>::input_implementation; })
    return std::same_as<caster_for<T>,
                        typename caster_for<T>::input_implementation>;
  return false;
}();

template <class T>
struct input_caster_cleanup_shape
    : std::bool_constant<first_party_input_caster_v<T> &&
                         native_cleanup_inert_v<caster_for<T>>> {};
template <class T>
inline constexpr bool input_caster_cleanup_inert_v =
    first_party_input_caster_v<T> && !caster_contract<T>.owns_python &&
    !caster_contract<T>.requires_admission &&
    input_caster_cleanup_shape<std::remove_cvref_t<T>>::value;
template <> struct input_caster_cleanup_shape<std::string> : std::true_type {};

template <class T> struct builtin_output_caster {};

template <> struct builtin_output_caster<bool> {
  static PyObject *cast(bool x) noexcept { return PyBool_FromLong(x); }
};

template <std::integral T>
  requires(!std::same_as<T, bool>)
struct builtin_output_caster<T> {
  static PyObject *cast(T x) noexcept {
    if constexpr (std::is_signed_v<T>) {
      if constexpr (sizeof(T) <= sizeof(long))
        return PyLong_FromLong(static_cast<long>(x));
      else
        return PyLong_FromLongLong(static_cast<long long>(x));
    } else {
      if constexpr (sizeof(T) <= sizeof(unsigned long))
        return PyLong_FromUnsignedLong(static_cast<unsigned long>(x));
      else
        return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(x));
    }
  }
};

template <std::floating_point T> struct builtin_output_caster<T> {
  static PyObject *cast(T x) noexcept {
    return PyFloat_FromDouble(static_cast<double>(x));
  }
};

template <> struct builtin_output_caster<std::string> {
  static PyObject *cast(const std::string &s) noexcept {
    return PyUnicode_FromStringAndSize(s.data(),
                                       static_cast<Py_ssize_t>(s.size()));
  }
};

template <> struct builtin_output_caster<std::string_view> {
  static PyObject *cast(std::string_view value) noexcept {
    return PyUnicode_FromStringAndSize(value.data(),
                                       static_cast<Py_ssize_t>(value.size()));
  }
};

template <class T>
inline constexpr bool builtin_output_cast = [] {
  using Actual = caster_for<T>;
  using Builtin = builtin_output_caster<std::remove_cvref_t<T>>;
  if constexpr (requires {
                  &Actual::cast;
                  &Builtin::cast;
                }) {
    if constexpr (std::is_same_v<decltype(&Actual::cast),
                                 decltype(&Builtin::cast)>)
      return &Actual::cast == &Builtin::cast;
  }
  return false;
}();

template <class T>
inline constexpr bool caster_load_runs_python = caster_contract<T>.runs_python;

template <class T> consteval void reject_lease_escape() {
  static_assert(
      !caster_requires_lease_v<std::remove_cvref_t<T>>,
      "borrowed array view requires a live lease and cannot be stored here");
}

template <class T> struct py_hint {
  static void append(std::string &out) {
    if constexpr (std::is_class_v<T>) {
      if (type_record *tr = resolve<T>()) {
        out += tr->name;
        return;
      }
    }
    out += "typing.Any";
  }
};
template <> struct py_hint<bool> {
  static void append(std::string &o) { o += "bool"; }
};
template <> struct py_hint<void> {
  static void append(std::string &o) { o += "None"; }
};
template <std::integral T>
  requires(!std::same_as<T, bool>)
struct py_hint<T> {
  static void append(std::string &o) { o += "int"; }
};
template <std::floating_point T> struct py_hint<T> {
  static void append(std::string &o) { o += "float"; }
};
template <> struct py_hint<std::string> {
  static void append(std::string &o) { o += "str"; }
};
template <> struct py_hint<std::string_view> {
  static void append(std::string &o) { o += "str"; }
};
template <> struct py_hint<const char *> {
  static void append(std::string &o) { o += "str"; }
};
template <> struct py_hint<PyObject *> {
  static void append(std::string &o) { o += "typing.Any"; }
};
template <class T>
  requires std::is_enum_v<T>
struct py_hint<T> {
  static void append(std::string &o) { o += "int"; }
};

template <class T> void append_py_hint(std::string &out) {
  using D = std::remove_cvref_t<T>;
  if constexpr (std::is_pointer_v<D> && !std::is_same_v<D, PyObject *> &&
                !std::is_same_v<D, const char *>) {
    py_hint<std::remove_cv_t<std::remove_pointer_t<D>>>::append(out);
    out += " | None";
  } else {
    py_hint<D>::append(out);
  }
}

} // namespace detail

template <class B> inline constexpr bool enable_implicit_v = false;

template <>
struct type_caster<bool> : detail::builtin_output_caster<bool>,
                           detail::scalar_input_projection<type_caster<bool>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{false, false, false};
  bool v;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (o == Py_True) {
      v = true;
      return true;
    }
    if (o == Py_False) {
      v = false;
      return true;
    }
    return false;
  }
  template <class A> A as() const noexcept { return static_cast<A>(v); }
};

namespace detail {

PYB26_COLD inline bool integer_out_of_range() noexcept {
  PyErr_SetString(PyExc_OverflowError,
                  "integer out of range for the C++ parameter type");
  return false;
}

struct float_load_result {
  double value;
  bool success;
};

PYB26_NOINLINE inline float_load_result
float_from_protocol(PyObject *o) noexcept {
  if (!PyType_GetSlot(Py_TYPE(o), Py_nb_float) && !PyIndex_Check(o))
    return {0.0, false};
  const double value = PyFloat_AsDouble(o);
  return {value, value != -1.0 || !PyErr_Occurred()};
}

template <class T>
PYB26_HOT bool load_int_checked(PyObject *o, T &out) noexcept {
  if constexpr (std::is_signed_v<T>) {
    const long long x = PyLong_AsLongLong(o);
    if (x == -1 && PyErr_Occurred()) [[unlikely]]
      return false;
    out = static_cast<T>(x);
    if constexpr (sizeof(T) < sizeof(long long)) {
      if (x != static_cast<long long>(out)) [[unlikely]]
        return integer_out_of_range();
    }
  } else {
    const unsigned long long x = PyLong_AsUnsignedLongLong(o);
    if (x == static_cast<unsigned long long>(-1) && PyErr_Occurred())
        [[unlikely]]
      return false;
    out = static_cast<T>(x);
    if constexpr (sizeof(T) < sizeof(unsigned long long)) {
      if (x != static_cast<unsigned long long>(out)) [[unlikely]]
        return integer_out_of_range();
    }
  }
  return true;
}

} // namespace detail

template <std::integral T>
  requires(!std::same_as<T, bool>)
struct type_caster<T> : detail::builtin_output_caster<T>,
                        detail::scalar_input_projection<type_caster<T>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{false, false, false};
  T v;
  PYB26_HOT bool load(PyObject *o) noexcept {
#if !defined(Py_LIMITED_API)
    if (PyLong_CheckExact(o)) [[likely]] {
      auto *l = reinterpret_cast<PyLongObject *>(o);
      if (detail::cpython::long_is_compact(l)) [[likely]] {
        const Py_ssize_t x = detail::cpython::compact_long_value(l);
        v = static_cast<T>(x);
        if constexpr (std::is_unsigned_v<T> ||
                      sizeof(T) != sizeof(Py_ssize_t)) {
          if ((std::is_unsigned_v<T> && x < 0) ||
              (sizeof(T) != sizeof(Py_ssize_t) &&
               x != static_cast<Py_ssize_t>(v))) [[unlikely]]
            return detail::integer_out_of_range();
        }
        return true;
      }
    } else if (!PyLong_Check(o)) [[unlikely]] {
      return false;
    }
#else
    if (!PyLong_Check(o)) [[unlikely]]
      return false;
#endif
    return detail::load_int_checked<T>(o, v);
  }
  template <class A> A as() const noexcept { return static_cast<A>(v); }
};

template <std::floating_point T>
struct type_caster<T> : detail::builtin_output_caster<T>,
                        detail::scalar_input_projection<type_caster<T>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{true, false, false};
  T v;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (PyFloat_CheckExact(o)) [[likely]] {
      v = static_cast<T>(PyFloat_AS_DOUBLE(o));
      return true;
    }
    const auto converted = detail::float_from_protocol(o);
    if (!converted.success) [[unlikely]]
      return false;
    v = static_cast<T>(converted.value);
    return true;
  }
  template <class A> A as() const noexcept { return static_cast<A>(v); }
};

namespace detail {

struct enum_record {
  PyInterpreterState *interp;
  PyObject *pyenum;
  enum_record *next;
};

template <class E>
PYB26_HIDDEN inline std::atomic<enum_record *> enum_chain_of{nullptr};

template <class E> inline void register_enum(PyObject *pyenum) noexcept {
  auto *rec = new (std::nothrow) enum_record;
  if (!rec)
    return;
  rec->interp = current_interp();
  rec->pyenum = Py_NewRef(pyenum);
  std::atomic<enum_record *> &head = enum_chain_of<E>;
  rec->next = head.load(std::memory_order_relaxed);
  while (!head.compare_exchange_weak(rec->next, rec, std::memory_order_release,
                                     std::memory_order_relaxed)) {
  }
}

template <class E> inline PyObject *lookup_enum() noexcept {
  enum_record *r = enum_chain_of<E>.load(std::memory_order_acquire);
  if (!r)
    return nullptr;
  PyInterpreterState *cur = current_interp();
  for (; r; r = r->next)
    if (r->interp == cur)
      return r->pyenum;
  return nullptr;
}

} // namespace detail

template <typename E>
  requires std::is_enum_v<E>
struct type_caster<E> : detail::scalar_input_projection<type_caster<E>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{false, false, false};
  using U = std::underlying_type_t<E>;
  E v;
  PYB26_HOT bool load(PyObject *o) noexcept {
    if (!PyLong_Check(o)) [[unlikely]]
      return false;
    U u;
    if (!detail::load_int_checked<U>(o, u)) [[unlikely]]
      return false;
    v = static_cast<E>(u);
    return true;
  }
  template <class A> A as() const noexcept { return static_cast<A>(v); }
  static PyObject *cast(E x) noexcept {
    PyObject *i =
        PyLong_FromLongLong(static_cast<long long>(static_cast<U>(x)));
    if (!i)
      return nullptr;
    PyObject *et = detail::lookup_enum<E>();
    if (!et)
      return i;
    PyObject *member = PyObject_CallOneArg(et, i);
    if (member) {
      Py_DECREF(i);
      return member;
    }
    if (PyErr_ExceptionMatches(PyExc_ValueError)) {
      PyErr_Clear();
      return i;
    }
    Py_DECREF(i);
    return nullptr;
  }
};

template <>
struct type_caster<std::string>
    : detail::builtin_output_caster<std::string>,
      detail::owning_input_projection<type_caster<std::string>> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{false, false, false};
  std::string s;
  PYB26_HOT bool load(PyObject *o) noexcept {
    Py_ssize_t n = 0;
    const char *p = PyUnicode_AsUTF8AndSize(o, &n);
    if (!p) [[unlikely]] {
      if (!PyUnicode_Check(o))
        PyErr_Clear();
      return false;
    }
    s.assign(p, static_cast<size_t>(n));
    return true;
  }
  template <class A> decltype(auto) as() {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(s);
    else
      return A(std::move(s));
  }
};

template <>
struct type_caster<std::string_view>
    : detail::builtin_output_caster<std::string_view>,
      detail::scalar_input_projection<type_caster<std::string_view>,
                                      argument_source::input_view> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{false, true, false};
  std::string_view sv;
  PYB26_HOT bool load(PyObject *o) noexcept {
    Py_ssize_t n = 0;
    const char *p = PyUnicode_AsUTF8AndSize(o, &n);
    if (!p) [[unlikely]] {
      if (!PyUnicode_Check(o))
        PyErr_Clear();
      return false;
    }
    sv = std::string_view(p, static_cast<size_t>(n));
    return true;
  }
  template <class A> A as() const noexcept { return static_cast<A>(sv); }
};

template <>
struct type_caster<const char *>
    : detail::scalar_input_projection<type_caster<const char *>,
                                      argument_source::input_view> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{false, true, false};
  const char *p = nullptr;
  PYB26_HOT bool load(PyObject *o) noexcept {
    Py_ssize_t n = 0;
    p = PyUnicode_AsUTF8AndSize(o, &n);
    if (!p) [[unlikely]] {
      if (!PyUnicode_Check(o))
        PyErr_Clear();
      return false;
    }
    if (std::char_traits<char>::length(p) != static_cast<size_t>(n))
        [[unlikely]]
      return false;
    return true;
  }
  template <class A> A as() const noexcept { return p; }
  static PyObject *cast(const char *x) noexcept {
    return x ? PyUnicode_FromString(x) : Py_NewRef(Py_None);
  }
};

template <>
struct type_caster<PyObject *>
    : detail::scalar_input_projection<type_caster<PyObject *>,
                                      argument_source::input_view> {
  using input_implementation = type_caster;
  static constexpr caster_effects effects{false, true, false};
  PyObject *p = nullptr;
  PYB26_HOT bool load(PyObject *o) noexcept {
    p = o;
    return true;
  }
  template <class A> A as() const noexcept { return p; }
  static PyObject *cast(PyObject *x) noexcept {
    return x ? Py_NewRef(x) : Py_NewRef(Py_None);
  }
};

template <class T> struct compatible {
  static_assert(
      std::is_arithmetic_v<T> || std::same_as<T, std::string>,
      "pyb26: compatible<T> supports owned arithmetic and std::string values");
  T value;
};

namespace detail {

template <class T> struct py_hint<compatible<T>> : py_hint<T> {};

PYB26_NOINLINE inline bool compatible_bool(PyObject *o, bool &value) noexcept {
  if (o == Py_None) {
    value = false;
    return true;
  }
  if (!PyType_GetSlot(Py_TYPE(o), Py_nb_bool))
    return false;
  const int result = PyObject_IsTrue(o);
  if (result < 0)
    return false;
  value = result != 0;
  return true;
}

PYB26_HOT inline bool assign_compatible_string(std::string &out,
                                               const char *data,
                                               Py_ssize_t size) noexcept {
  out.assign(data, static_cast<std::size_t>(size));
  return true;
}

PYB26_NOINLINE inline bool compatible_binary_string(PyObject *o,
                                                    std::string &out) noexcept {
  if (PyBytes_Check(o))
    return assign_compatible_string(out, PyBytes_AS_STRING(o),
                                    PyBytes_GET_SIZE(o));
  if (!PyByteArray_Check(o))
    return false;
  bool success;
  Py_BEGIN_CRITICAL_SECTION(o);
  success = assign_compatible_string(out, PyByteArray_AS_STRING(o),
                                     PyByteArray_GET_SIZE(o));
  Py_END_CRITICAL_SECTION();
  return success;
}

} // namespace detail

template <class T> struct type_caster<compatible<T>> {
  static constexpr caster_effects effects{true, false, false};
  caster_for<T> inner;

  PYB26_HOT bool load(PyObject *o) noexcept {
    if constexpr (std::same_as<T, bool>) {
      if (inner.load(o)) [[likely]]
        return true;
      return detail::compatible_bool(o, inner.v);
    } else if constexpr (std::integral<T>) {
      if (PyLong_Check(o)) [[likely]]
        return inner.load(o);
      if (!PyIndex_Check(o))
        return false;
      PyObject *index = PyNumber_Index(o);
      if (!index)
        return false;
      const bool success = inner.load(index);
      Py_DECREF(index);
      return success;
    } else if constexpr (std::floating_point<T>) {
      return inner.load(o);
    } else {
      if (!PyUnicode_Check(o)) [[unlikely]]
        return detail::compatible_binary_string(o, inner.s);
      Py_ssize_t size;
      const char *data = PyUnicode_AsUTF8AndSize(o, &size);
      if (!data)
        return false;
      return detail::assign_compatible_string(inner.s, data, size);
    }
  }

  template <class A> compatible<T> as() { return {inner.template as<T>()}; }

  static PyObject *cast(const compatible<T> &value) noexcept {
    return caster_for<T>::cast(value.value);
  }
};

template <class T> struct exact;

namespace detail {

template <class X> inline constexpr bool is_exact_v = false;
template <class X> inline constexpr bool is_exact_v<exact<X>> = true;

template <class T>
concept exact_loadable = requires(caster_for<T> &c, PyObject *o) {
  { c.load_exact(o) } noexcept -> std::same_as<bool>;
};

} // namespace detail

template <class T> struct exact {
  static_assert(!detail::is_exact_v<T>,
                "pyb26: exact<exact<T>> is redundant — use exact<T>");
  static_assert(!std::is_reference_v<T>,
                "pyb26: exact<T&> is not a thing — exact<T> is taken by value");
  static_assert(!caster_yields_view_v<T>,
                "pyb26: exact<T> cannot wrap a view type (std::string_view / "
                "const char* / PyObject* / a bound-class pointer): a view "
                "borrows the argument's own storage and is already "
                "type-strict — take the view directly, or use "
                "exact<std::string> for an owned exact str");
  T value;
};

template <class T> struct type_caster<exact<T>> {
  static constexpr caster_effects effects{
      !(std::is_arithmetic_v<T> || std::is_same_v<T, std::string>) &&
          caster_contract<T>.runs_python,
      false, false, caster_contract<T>.owns_python,
      caster_contract<T>.requires_admission};
  caster_for<T> inner;

  PYB26_HOT bool load(PyObject *o) noexcept {
    if constexpr (std::same_as<T, bool>) {
      return inner.load(o);
    } else if constexpr (std::integral<T>) {
      if (!PyLong_CheckExact(o)) [[unlikely]]
        return false;
      return inner.load(o);
    } else if constexpr (std::floating_point<T>) {
      if (!PyFloat_CheckExact(o)) [[unlikely]]
        return false;
      return inner.load(o);
    } else if constexpr (std::same_as<T, std::string>) {
      return inner.load(o);
    } else {
      static_assert(detail::exact_loadable<T>,
                    "pyb26: exact<T> supports arithmetic types, bool, "
                    "std::string, and bound classes only");
      return inner.load_exact(o);
    }
  }
  template <class A> exact<T> as() { return exact<T>{inner.template as<T>()}; }
  static PyObject *cast(const exact<T> &e) {
    return caster_for<T>::cast(e.value);
  }
};

} // namespace pyb26

#define PYB26_OPAQUE(...)                                                      \
  template <> inline constexpr bool ::pyb26::opaque_v<__VA_ARGS__> = true
