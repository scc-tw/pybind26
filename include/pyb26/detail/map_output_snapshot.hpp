#pragma once

#include "stl.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>

namespace pyb26::detail {

enum class map_iter_kind : unsigned char { keys, values, items };

template <class T> struct map_output_field {
  using record_type = std::remove_cv_t<T>;
  static constexpr bool enabled = std::is_arithmetic_v<T>;
  static constexpr bool has_bytes = false;
  static constexpr std::size_t bytes(const T &) noexcept { return 0; }
  static void store(record_type &record, const T &value, char *,
                    std::size_t) noexcept {
    record = value;
  }
  static PyObject *cast(const record_type &value, const char *) noexcept {
    return caster_for<T>::cast(value);
  }
};

template <> struct map_output_field<std::string> {
  struct record_type {
    std::size_t offset, size;
  };
  static constexpr bool enabled = true;
  static constexpr bool has_bytes = true;
  static std::size_t bytes(const std::string &value) noexcept {
    return value.size();
  }
  static void store(record_type &record, const std::string &value, char *arena,
                    std::size_t offset) noexcept {
    record = {offset, value.size()};
    if (record.size)
      std::memcpy(arena + record.offset, value.data(), record.size);
  }
  static PyObject *cast(const record_type &value, const char *arena) noexcept {
    if (value.size > static_cast<std::size_t>(PY_SSIZE_T_MAX)) {
      PyErr_SetString(PyExc_OverflowError, "map snapshot string is too large");
      return nullptr;
    }
    return PyUnicode_FromStringAndSize(arena + value.offset,
                                       static_cast<Py_ssize_t>(value.size));
  }
};

template <class M, map_iter_kind Kind> struct map_output_fields;

template <class M> struct map_output_fields<M, map_iter_kind::keys> {
  using codec = map_output_field<std::remove_cv_t<typename M::key_type>>;
  using record_type = typename codec::record_type;
  static constexpr bool enabled =
      codec::enabled && builtin_output_cast<typename M::key_type>;
  static constexpr bool has_bytes = codec::has_bytes;
  static std::size_t bytes(const auto &entry) noexcept {
    return codec::bytes(entry.first);
  }
  static void store(record_type &record, const auto &entry, char *arena,
                    std::size_t offset) noexcept {
    codec::store(record, entry.first, arena, offset);
  }
  static PyObject *cast(const record_type &record, const char *arena) noexcept {
    return codec::cast(record, arena);
  }
};

template <class M> struct map_output_fields<M, map_iter_kind::values> {
  using codec = map_output_field<std::remove_cv_t<typename M::mapped_type>>;
  using record_type = typename codec::record_type;
  static constexpr bool enabled =
      codec::enabled && builtin_output_cast<typename M::mapped_type>;
  static constexpr bool has_bytes = codec::has_bytes;
  static std::size_t bytes(const auto &entry) noexcept {
    return codec::bytes(entry.second);
  }
  static void store(record_type &record, const auto &entry, char *arena,
                    std::size_t offset) noexcept {
    codec::store(record, entry.second, arena, offset);
  }
  static PyObject *cast(const record_type &record, const char *arena) noexcept {
    return codec::cast(record, arena);
  }
};

template <class M> struct map_output_fields<M, map_iter_kind::items> {
  using key_codec = map_output_field<std::remove_cv_t<typename M::key_type>>;
  using value_codec =
      map_output_field<std::remove_cv_t<typename M::mapped_type>>;
  struct record_type {
    typename key_codec::record_type key;
    typename value_codec::record_type value;
  };
  static constexpr bool enabled = key_codec::enabled && value_codec::enabled &&
                                  builtin_output_cast<typename M::key_type> &&
                                  builtin_output_cast<typename M::mapped_type>;
  static constexpr bool has_bytes =
      key_codec::has_bytes || value_codec::has_bytes;
  static std::size_t key_bytes(const auto &entry) noexcept {
    return key_codec::bytes(entry.first);
  }
  static std::size_t value_bytes(const auto &entry) noexcept {
    return value_codec::bytes(entry.second);
  }
  static void store(record_type &record, const auto &entry, char *arena,
                    std::size_t offset) noexcept {
    const auto value_offset = offset + key_codec::bytes(entry.first);
    key_codec::store(record.key, entry.first, arena, offset);
    value_codec::store(record.value, entry.second, arena, value_offset);
  }
  static PyObject *cast(const record_type &record, const char *arena) noexcept {
    PyObject *result = PyTuple_New(2);
    if (!result)
      return nullptr;
    PyObject *key = key_codec::cast(record.key, arena);
    if (!key)
      return stl_output_failure(result);
    PyTuple_SET_ITEM(result, 0, key);
    PyObject *value = value_codec::cast(record.value, arena);
    if (!value)
      return stl_output_failure(result);
    PyTuple_SET_ITEM(result, 1, value);
    return result;
  }
};

template <class M, map_iter_kind Kind> class map_output_snapshot {
  using fields = map_output_fields<M, Kind>;
  using record_type = typename fields::record_type;
  static_assert(fields::enabled && std::is_trivially_copyable_v<record_type>);
  static_assert(alignof(record_type) <= alignof(std::max_align_t));
  std::unique_ptr<std::byte[]> storage_;
  std::size_t byte_offset_ = 0;
  std::size_t size_ = 0;

  static bool add_bytes(std::size_t &total, std::size_t count) noexcept {
    if (count > std::numeric_limits<std::size_t>::max() - total)
      return false;
    total += count;
    return true;
  }
  static bool add_entry_bytes(std::size_t &total,
                              const typename M::value_type &entry) noexcept {
    if constexpr (Kind == map_iter_kind::items)
      return add_bytes(total, fields::key_bytes(entry)) &&
             add_bytes(total, fields::value_bytes(entry));
    else
      return add_bytes(total, fields::bytes(entry));
  }
  char *arena() noexcept {
    return reinterpret_cast<char *>(storage_.get() + byte_offset_);
  }
  const char *arena() const noexcept {
    return reinterpret_cast<const char *>(storage_.get() + byte_offset_);
  }

public:
  map_output_snapshot() noexcept = default;
  map_output_snapshot(map_output_snapshot &&) noexcept = default;
  bool acquire(PyObject *wrapper, const M &source) noexcept {
    size_ = 0;
    std::size_t capacity = 0, byte_capacity = 0;
    for (;;) {
      std::size_t count = 0, bytes = 0;
      bool overflow = false, captured = false;
      Py_BEGIN_CRITICAL_SECTION(wrapper);
      count = source.size();
      if (count <= capacity) {
        std::size_t index = 0;
        for (const auto &entry : source) {
          const auto offset = bytes;
          if constexpr (fields::has_bytes) {
            if (!add_entry_bytes(bytes, entry)) {
              overflow = true;
              break;
            }
            if (bytes > byte_capacity)
              continue;
          }
          auto *slot = reinterpret_cast<record_type *>(
              storage_.get() + index * sizeof(record_type));
          ::new (static_cast<void *>(slot)) record_type;
          fields::store(*slot, entry, arena(), offset);
          ++index;
        }
        if constexpr (!fields::has_bytes) {
          size_ = count;
          captured = true;
        } else if (!overflow && bytes <= byte_capacity) {
          size_ = count;
          captured = true;
        }
      } else if constexpr (fields::has_bytes) {
        for (const auto &entry : source)
          if (!add_entry_bytes(bytes, entry)) {
            overflow = true;
            break;
          }
      }
      Py_END_CRITICAL_SECTION();
      if (captured)
        return true;
      constexpr auto limit = std::numeric_limits<std::size_t>::max();
      if (overflow || count > limit / sizeof(record_type)) {
        PyErr_NoMemory();
        return false;
      }
      const auto offset = count * sizeof(record_type);
      if (bytes > limit - offset) {
        PyErr_NoMemory();
        return false;
      }
      auto storage = std::unique_ptr<std::byte[]>(
          new (std::nothrow) std::byte[offset + bytes]);
      if (!storage) {
        PyErr_NoMemory();
        return false;
      }
      storage_ = std::move(storage);
      capacity = count;
      byte_capacity = bytes;
      byte_offset_ = offset;
    }
  }
  std::size_t size() const noexcept { return size_; }
  PyObject *cast(std::size_t index) const noexcept {
    const auto *record = std::launder(reinterpret_cast<const record_type *>(
        storage_.get() + index * sizeof(record_type)));
    return fields::cast(*record, arena());
  }
};

} // namespace pyb26::detail
