#pragma once

#include "caster.hpp"
#include "cpython.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>

namespace pyb26::detail {

template <class T> struct snapshot_value {
  static constexpr bool enabled = false;
};

template <class T>
  requires(first_party_input_caster_v<T> &&
           std::is_integral_v<std::remove_cvref_t<T>> &&
           !std::is_same_v<std::remove_cvref_t<T>, bool>)
struct snapshot_value<T> {
  using value_type = std::remove_cvref_t<T>;
  using probe_type = Py_ssize_t;
  using storage_type = Py_ssize_t;
  static constexpr bool enabled = true;
  static constexpr bool uses_arena = false;
  static bool probe(PyObject *object, probe_type &value) noexcept {
    if (Py_TYPE(object) != &PyLong_Type)
      return false;
    auto *integer = reinterpret_cast<PyLongObject *>(object);
    if (!cpython::long_is_compact(integer))
      return false;
    const Py_ssize_t number = cpython::compact_long_value(integer);
    if constexpr (std::is_unsigned_v<value_type>) {
      if (number < 0)
        return false;
      if constexpr (std::numeric_limits<value_type>::digits <
                    std::numeric_limits<Py_ssize_t>::digits)
        if (number >
            static_cast<Py_ssize_t>(std::numeric_limits<value_type>::max()))
          return false;
    } else if constexpr (std::numeric_limits<value_type>::digits <
                         std::numeric_limits<Py_ssize_t>::digits) {
      if (number <
              static_cast<Py_ssize_t>(std::numeric_limits<value_type>::min()) ||
          number >
              static_cast<Py_ssize_t>(std::numeric_limits<value_type>::max()))
        return false;
    }
    value = number;
    return true;
  }
  static constexpr std::size_t bytes(probe_type) noexcept { return 0; }
  static storage_type store(probe_type value, char *, std::size_t &) noexcept {
    return value;
  }
  static value_type materialize(storage_type value, const char *) noexcept {
    return static_cast<value_type>(value);
  }
  static auto constructor_args(storage_type value, const char *arena) noexcept {
    return std::tuple<value_type>{materialize(value, arena)};
  }
};

template <class T>
  requires(first_party_input_caster_v<T> &&
           std::is_same_v<std::remove_cvref_t<T>, bool>)
struct snapshot_value<T> {
  using probe_type = bool;
  using storage_type = bool;
  static constexpr bool enabled = true;
  static constexpr bool uses_arena = false;
  static bool probe(PyObject *object, probe_type &value) noexcept {
    if (object != Py_True && object != Py_False)
      return false;
    value = object == Py_True;
    return true;
  }
  static constexpr std::size_t bytes(probe_type) noexcept { return 0; }
  static storage_type store(probe_type value, char *, std::size_t &) noexcept {
    return value;
  }
  static bool materialize(storage_type value, const char *) noexcept {
    return value;
  }
  static auto constructor_args(storage_type value, const char *arena) noexcept {
    return std::tuple<bool>{materialize(value, arena)};
  }
};

template <class T>
  requires(first_party_input_caster_v<T> &&
           std::is_floating_point_v<std::remove_cvref_t<T>>)
struct snapshot_value<T> {
  using value_type = std::remove_cvref_t<T>;
  using probe_type = double;
  using storage_type = double;
  static constexpr bool enabled = true;
  static constexpr bool uses_arena = false;
  static bool probe(PyObject *object, probe_type &value) noexcept {
    return cpython::exact_float_value(object, value);
  }
  static constexpr std::size_t bytes(probe_type) noexcept { return 0; }
  static storage_type store(probe_type value, char *, std::size_t &) noexcept {
    return value;
  }
  static value_type materialize(storage_type value, const char *) noexcept {
    return static_cast<value_type>(value);
  }
  static auto constructor_args(storage_type value, const char *arena) noexcept {
    return std::tuple<value_type>{materialize(value, arena)};
  }
};

template <class T>
  requires(first_party_input_caster_v<T> &&
           std::is_same_v<std::remove_cvref_t<T>, std::string>)
struct snapshot_value<T> {
  struct probe_type {
    const char *data;
    std::size_t size;
  };
  struct storage_type {
    std::uint16_t offset, length;
  };
  static constexpr bool enabled = true;
  static constexpr bool uses_arena = true;
  static bool probe(PyObject *object, probe_type &value) noexcept {
    return cpython::exact_compact_ascii(object, value.data, value.size);
  }
  static constexpr std::size_t bytes(const probe_type &value) noexcept {
    return value.size;
  }
  static storage_type store(const probe_type &value, char *arena,
                            std::size_t &used) noexcept {
    storage_type result{static_cast<std::uint16_t>(used),
                        static_cast<std::uint16_t>(value.size)};
    std::memcpy(arena + used, value.data, value.size);
    used += value.size;
    return result;
  }
  static std::string materialize(storage_type value, const char *arena) {
    return std::string(arena + value.offset, value.length);
  }
  static auto constructor_args(storage_type value, const char *arena) noexcept {
    return std::tuple<const char *, std::size_t>{arena + value.offset,
                                                 value.length};
  }
};

} // namespace pyb26::detail
