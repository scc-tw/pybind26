#pragma once

#include "lifetime.hpp"
#include "snapshot_value.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace pyb26::detail {

template <class K> struct dict_string_key_proof {
  static constexpr bool all_exact = false;
};
template <> struct dict_string_key_proof<std::string> {
  bool all_exact = false;
};

template <class K, class V> class native_dict_snapshot {
  using key_codec = snapshot_value<K>;
  using value_codec = snapshot_value<V>;
  static_assert(key_codec::enabled && value_codec::enabled);
  using key_storage = typename key_codec::storage_type;
  using value_storage = typename value_codec::storage_type;
  static_assert(std::is_trivially_copyable_v<key_storage> &&
                std::is_trivially_destructible_v<key_storage> &&
                sizeof(key_storage) <= sizeof(PyObject *) &&
                alignof(key_storage) <= alignof(PyObject *));
  static_assert(std::is_trivially_copyable_v<value_storage> &&
                std::is_trivially_destructible_v<value_storage> &&
                sizeof(value_storage) <= sizeof(PyObject *) &&
                alignof(value_storage) <= alignof(PyObject *));

  template <class Storage> union slot {
    PyObject *python;
    Storage native;
    slot() noexcept {}
  };
  struct pair {
    slot<key_storage> key;
    slot<value_storage> value;
  };
  static_assert(sizeof(pair) == 2 * sizeof(PyObject *));
  static_assert(std::is_trivially_destructible_v<pair> &&
                alignof(pair) <= alignof(std::max_align_t) &&
                alignof(std::uint64_t) <= alignof(std::max_align_t) &&
                sizeof(pair) % alignof(std::uint64_t) == 0);
  static constexpr Py_ssize_t inline_capacity = 16;
  static constexpr std::size_t max_arena_capacity = 256;
  static constexpr std::size_t arena_capacity =
      key_codec::uses_arena || value_codec::uses_arena ? max_arena_capacity : 0;
  struct empty_arena {};
  pair inline_[inline_capacity];
  [[no_unique_address]] std::conditional_t<
      arena_capacity != 0, char[max_arena_capacity], empty_arena> arena_;
  std::unique_ptr<std::byte[]> heap_;
  Py_ssize_t capacity_ = inline_capacity;
  Py_ssize_t count_ = 0;
  Py_ssize_t python_count_ = 0;
  std::uint64_t inline_mask_ = 0;
  [[no_unique_address]] dict_string_key_proof<K> key_proof_;

  char *arena_data() noexcept {
    if constexpr (arena_capacity != 0)
      return arena_;
    else
      return nullptr;
  }
  const char *arena_data() const noexcept {
    if constexpr (arena_capacity != 0)
      return arena_;
    else
      return nullptr;
  }
  pair *items() noexcept {
    return heap_ ? std::launder(reinterpret_cast<pair *>(heap_.get()))
                 : inline_;
  }
  const pair *items() const noexcept {
    return heap_ ? std::launder(reinterpret_cast<const pair *>(heap_.get()))
                 : inline_;
  }
  std::uint64_t *mask() noexcept {
    return capacity_ > 64
               ? std::launder(reinterpret_cast<std::uint64_t *>(
                     heap_.get() +
                     static_cast<std::size_t>(capacity_) * sizeof(pair)))
               : &inline_mask_;
  }
  const std::uint64_t *mask() const noexcept {
    return capacity_ > 64
               ? std::launder(reinterpret_cast<const std::uint64_t *>(
                     heap_.get() +
                     static_cast<std::size_t>(capacity_) * sizeof(pair)))
               : &inline_mask_;
  }
  static std::size_t mask_words(Py_ssize_t count) noexcept {
    return (static_cast<std::size_t>(count) - 1) / 64 + 1;
  }
  bool reserve(Py_ssize_t required) noexcept {
    constexpr auto limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    const auto count = static_cast<std::size_t>(required);
    if (count > limit / sizeof(pair)) {
      PyErr_NoMemory();
      return false;
    }
    const auto record_bytes = count * sizeof(pair);
    const auto words = required > 64 ? mask_words(required) : 0;
    if (words > (limit - record_bytes) / sizeof(std::uint64_t)) {
      PyErr_NoMemory();
      return false;
    }
    auto next = std::unique_ptr<std::byte[]>(new (
        std::nothrow) std::byte[record_bytes + words * sizeof(std::uint64_t)]);
    if (!next) {
      PyErr_NoMemory();
      return false;
    }
    ::new (static_cast<void *>(next.get())) pair[count];
    if (words)
      ::new (static_cast<void *>(next.get() + record_bytes))
          std::uint64_t[words];
    heap_ = std::move(next);
    capacity_ = required;
    return true;
  }
  static bool probe_pair(PyObject *key, PyObject *value,
                         typename key_codec::probe_type &key_probe,
                         typename value_codec::probe_type &value_probe,
                         std::size_t used) noexcept {
    const bool key_ready = key_codec::probe(key, key_probe);
    const bool value_ready = value_codec::probe(value, value_probe);
    if (!key_ready || !value_ready)
      return false;
    const auto key_bytes = key_codec::bytes(key_probe);
    const auto value_bytes = value_codec::bytes(value_probe);
    const auto remaining = arena_capacity - used;
    return key_bytes <= remaining && value_bytes <= remaining - key_bytes;
  }
  void store_native(pair &target,
                    const typename key_codec::probe_type &key_probe,
                    const typename value_codec::probe_type &value_probe,
                    std::size_t &used) noexcept {
    std::construct_at(&target.key.native,
                      key_codec::store(key_probe, arena_data(), used));
    std::construct_at(&target.value.native,
                      value_codec::store(value_probe, arena_data(), used));
  }
  void store_python(pair &target, PyObject *key, PyObject *value) noexcept {
    if constexpr (std::is_same_v<K, std::string>)
      key_proof_.all_exact &= PyUnicode_CheckExact(key);
    target.key.python = Py_NewRef(key);
    target.value.python = Py_NewRef(value);
  }
  static bool is_native(const std::uint64_t *fallback,
                        Py_ssize_t index) noexcept {
    return !(fallback[static_cast<std::size_t>(index) / 64] &
             (std::uint64_t{1} << (static_cast<std::size_t>(index) % 64)));
  }
  template <bool CheckNative>
  PYB26_COLD PYB26_NOINLINE static void
  clear_failed_pairs(const pair *records, const std::uint64_t *fallback,
                     PyThreadState *state, Py_ssize_t count, Py_ssize_t next,
                     bool value_pending) noexcept {
    callback_error_scope error{state};
    error.capture();
    if (value_pending)
      error.discard(records[next++].value.python);
    for (Py_ssize_t i = next; i < count; ++i) {
      if constexpr (CheckNative)
        if (is_native(fallback, i))
          continue;
      error.discard(records[i].key.python);
      error.discard(records[i].value.python);
    }
  }
  template <bool CheckNative>
  PYB26_NOINLINE static void clear_pairs(const pair *records,
                                         const std::uint64_t *fallback,
                                         Py_ssize_t count) noexcept {
    auto *state = cpython::thread_state();
    if (cpython::current_exception(state)) [[unlikely]] {
      clear_failed_pairs<CheckNative>(records, fallback, state, count, 0,
                                      false);
      return;
    }
    for (Py_ssize_t i = 0; i < count; ++i) {
      if constexpr (CheckNative)
        if (is_native(fallback, i))
          continue;
      Py_DECREF(records[i].key.python);
      if (cpython::current_exception(state)) [[unlikely]] {
        clear_failed_pairs<CheckNative>(records, fallback, state, count, i,
                                        true);
        return;
      }
      Py_DECREF(records[i].value.python);
      if (cpython::current_exception(state)) [[unlikely]] {
        clear_failed_pairs<CheckNative>(records, fallback, state, count, i + 1,
                                        false);
        return;
      }
    }
  }

public:
  class reader {
    const pair *const records_;
    const char *const arena_;
    const std::uint64_t *const fallback_;
    const pair *const end_;

    friend class native_dict_snapshot;
    reader(const pair *records, const char *arena,
           const std::uint64_t *fallback, Py_ssize_t count) noexcept
        : records_(records), arena_(arena), fallback_(fallback),
          end_(records + count) {}

  public:
    const pair *first() const noexcept { return records_; }
    const pair *last() const noexcept { return end_; }
    static const pair *next(const pair *row) noexcept { return row + 1; }
    bool is_native(Py_ssize_t index) const noexcept {
      return native_dict_snapshot::is_native(fallback_, index);
    }
    K native_key(const pair *row) const {
      return key_codec::materialize(row->key.native, arena_);
    }
    V native_value(const pair *row) const {
      return value_codec::materialize(row->value.native, arena_);
    }
    auto native_key_args(const pair *row) const noexcept {
      return key_codec::constructor_args(row->key.native, arena_);
    }
    auto native_value_args(const pair *row) const noexcept {
      return value_codec::constructor_args(row->value.native, arena_);
    }
    PyObject *key(const pair *row) const noexcept { return row->key.python; }
    PyObject *value(const pair *row) const noexcept {
      return row->value.python;
    }
  };

  native_dict_snapshot() noexcept {}
  native_dict_snapshot(const native_dict_snapshot &) = delete;
  native_dict_snapshot &operator=(const native_dict_snapshot &) = delete;
  ~native_dict_snapshot() { clear(); }

  bool acquire(PyObject *source) noexcept {
    if (python_count_) {
      clear();
      if (PyErr_Occurred())
        return false;
    } else {
      clear();
    }
    if (!PyDict_Check(source))
      return false;
    for (;;) {
      Py_ssize_t required;
      bool captured = false;
      Py_BEGIN_CRITICAL_SECTION(source);
      required = PyDict_GET_SIZE(source);
      if (required <= capacity_) {
        auto *records = items();
        if constexpr (std::is_same_v<K, std::string>)
          key_proof_.all_exact = true;
        Py_ssize_t position = 0, count = 0, python_count = 0;
        std::size_t used = 0;
        PyObject *key, *value;
        if (PyDict_Next(source, &position, &key, &value)) {
          typename key_codec::probe_type key_probe;
          typename value_codec::probe_type value_probe;
          if (!probe_pair(key, value, key_probe, value_probe, used)) {
            do {
              store_python(records[count++], key, value);
            } while (PyDict_Next(source, &position, &key, &value));
            python_count = count;
          } else {
            const auto words = capacity_ > 64 ? mask_words(required) : 1;
            auto *fallback = mask();
            for (std::size_t i = 0; i < words; ++i)
              fallback[i] = 0;
            store_native(records[count++], key_probe, value_probe, used);
            while (PyDict_Next(source, &position, &key, &value)) {
              if (probe_pair(key, value, key_probe, value_probe, used)) {
                store_native(records[count], key_probe, value_probe, used);
              } else {
                store_python(records[count], key, value);
                fallback[static_cast<std::size_t>(count) / 64] |=
                    std::uint64_t{1} << (static_cast<std::size_t>(count) % 64);
                ++python_count;
              }
              ++count;
            }
          }
        }
        count_ = count;
        python_count_ = python_count;
        captured = true;
      }
      Py_END_CRITICAL_SECTION();
      if (captured)
        return true;
      if (!reserve(required))
        return false;
    }
  }

  Py_ssize_t size() const noexcept { return count_; }
  bool all_native() const noexcept { return python_count_ == 0; }
  bool all_python() const noexcept {
    return count_ != 0 && python_count_ == count_;
  }
  bool all_exact_string_keys() const noexcept { return key_proof_.all_exact; }
  reader read() const noexcept {
    return {items(), arena_data(), mask(), count_};
  }
  PYB26_ALWAYS_INLINE void clear() noexcept {
    if constexpr (std::is_same_v<K, std::string>)
      key_proof_.all_exact = false;
    const auto count = std::exchange(count_, 0);
    const auto python_count = std::exchange(python_count_, 0);
    if (!python_count)
      return;
    if (python_count == count)
      clear_pairs<false>(items(), nullptr, count);
    else
      clear_pairs<true>(items(), mask(), count);
  }
};

} // namespace pyb26::detail
