#pragma once

#include "caster.hpp"

#if defined(__cpp_reflection) || defined(__cpp_impl_reflection)

#include <meta>
#include <type_traits>

namespace pyb26 {

struct py_dict {
  PyObject *obj = nullptr;
  bool owned = false;

  py_dict() = default;
  explicit py_dict(PyObject *o) noexcept : obj(o), owned(true) {}
  static py_dict borrow(PyObject *o) noexcept {
    py_dict d;
    d.obj = o;
    return d;
  }
};

template <> struct type_caster<py_dict> {
  static constexpr caster_effects effects{false, true, false};
  py_dict h;
  bool load(PyObject *o) noexcept {
    if (!PyDict_Check(o))
      return false;
    h = py_dict::borrow(o);
    return true;
  }
  template <class A> py_dict as() const noexcept { return h; }
  static PyObject *cast(py_dict d) noexcept {
    if (d.obj)
      return d.owned ? d.obj : Py_NewRef(d.obj);
    return PyErr_Occurred() ? nullptr : Py_NewRef(Py_None);
  }
};

template <class T> struct struct_cast;

namespace detail {

consteval bool struct_cast_member(std::meta::info m) {
  return std::meta::is_public(m) && std::meta::has_identifier(m) &&
         !std::meta::is_bit_field(m);
}

template <class MT>
constexpr bool member_is_aggregate =
    std::is_aggregate_v<MT> && !std::is_array_v<MT>;

template <class MT> bool member_load(PyObject *item, MT &out) noexcept {
  reject_lease_escape<MT>();
  if constexpr (member_is_aggregate<MT>) {
    struct_cast<MT> c;
    if (!c.load(item))
      return false;
    out = c.template as<MT>();
    return true;
  } else {
    type_caster<MT> c;
    if (!c.load(item))
      return false;
    out = c.template as<MT>();
    return true;
  }
}

template <class MT> PyObject *member_cast(const MT &in) {
  if constexpr (member_is_aggregate<MT>)
    return struct_cast<MT>::cast(in);
  else
    return type_caster<MT>::cast(in);
}

} // namespace detail

template <class T> struct struct_cast {
  static_assert(std::is_aggregate_v<T>,
                "pyb26::struct_cast<T>: T must be an aggregate (a plain "
                "struct of public data members).");

  T value{};

  bool load(PyObject *o) noexcept {
    if (!PyDict_Check(o))
      return false;
    bool ok = true;
    template for (constexpr auto m :
                  std::define_static_array(std::meta::nonstatic_data_members_of(
                      ^^T, std::meta::access_context::unchecked()))) {
      if constexpr (detail::struct_cast_member(m)) {
        if (ok) {
          constexpr const char *key =
              std::define_static_string(std::meta::identifier_of(m));
          PyObject *item = PyDict_GetItemString(o, key);
          if (!item) {
            ok = false;
          } else {
            using MT = std::remove_cvref_t<decltype(value.[:m:])>;
            MT tmp{};
            if (!detail::member_load<MT>(item, tmp))
              ok = false;
            else
              value.[:m:] = tmp;
          }
        }
      }
    }
    return ok;
  }

  template <class A> decltype(auto) as() {
    if constexpr (std::is_reference_v<A>)
      return static_cast<A>(value);
    else
      return A(value);
  }

  static PyObject *cast(const T &v) {
    PyObject *d = PyDict_New();
    if (!d)
      return nullptr;
    bool ok = true;
    template for (constexpr auto m :
                  std::define_static_array(std::meta::nonstatic_data_members_of(
                      ^^T, std::meta::access_context::unchecked()))) {
      if constexpr (detail::struct_cast_member(m)) {
        if (ok) {
          constexpr const char *key =
              std::define_static_string(std::meta::identifier_of(m));
          using MT = std::remove_cvref_t<decltype(v.[:m:])>;
          PyObject *item = detail::member_cast<MT>(v.[:m:]);
          if (!item || PyDict_SetItemString(d, key, item) != 0)
            ok = false;
          Py_XDECREF(item);
        }
      }
    }
    if (!ok) {
      Py_DECREF(d);
      return nullptr;
    }
    return d;
  }
};

} // namespace pyb26

#else

namespace pyb26 {
template <class T> struct struct_cast {
  static_assert(sizeof(T) == 0,
                "pyb26::struct_cast<T> requires C++26 static reflection "
                "(-std=c++26 -freflection, GCC 16+).");
};
} // namespace pyb26

#endif
