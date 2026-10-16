#pragma once

#include "lifetime.hpp"

#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <utility>

namespace pyb26::detail {

class callback_owner_ref;
class callback_binding_owner;

struct py_callable_owner {
private:
  static constexpr std::size_t binding_bit =
      std::size_t{1} << (std::numeric_limits<std::size_t>::digits - 1);
  std::atomic<std::size_t> ownership_{binding_bit};
  friend class callback_owner_ref;
  friend class callback_binding_owner;

  void reserve_initial_target() noexcept {
    ownership_.store(binding_bit | 1, std::memory_order_relaxed);
  }
  void retain_target() noexcept {
    ownership_.fetch_add(1, std::memory_order_relaxed);
  }
  PYB26_HIDDEN void destroy(const owner_admission *binding,
                            PyThreadState *state) noexcept;
  void release_target() noexcept {
    if (ownership_.load(std::memory_order_acquire) == 1) {
      destroy(nullptr, nullptr);
      return;
    }
    if (ownership_.fetch_sub(1, std::memory_order_acq_rel) == 1)
      destroy(nullptr, nullptr);
  }
  void release_binding(const owner_admission *binding = nullptr,
                       PyThreadState *state = nullptr) noexcept {
    if (ownership_.load(std::memory_order_acquire) == binding_bit) {
      destroy(binding, state);
      return;
    }
    if (ownership_.fetch_and(~binding_bit, std::memory_order_acq_rel) ==
        binding_bit)
      destroy(binding, state);
  }

public:
  long use_count() const noexcept {
    const auto state = ownership_.load(std::memory_order_acquire);
    return static_cast<long>((state & ~binding_bit) +
                             ((state & binding_bit) != 0));
  }
  bool has_other_targets(bool binding_target) const noexcept {
    return ownership_.load(std::memory_order_acquire) >
           binding_bit + static_cast<std::size_t>(binding_target);
  }
  struct error_channel {
    std::mutex mutex;
    python_owner_ref pending;
  };
  python_owner_ref callable;
  std::atomic<error_channel *> errors{nullptr};
  std::atomic<unsigned> native_failure{0};
  explicit py_callable_owner(
      PyObject *f,
      python_owner_context context = capture_python_owner_context()) noexcept
      : callable(f, std::move(context)) {}
  py_callable_owner(PyObject *f, python_owner_context context,
                    const owner_admission &binding) noexcept
      : callable(f, std::move(context), binding) {}
  py_callable_owner(const py_callable_owner &) = delete;
  py_callable_owner &operator=(const py_callable_owner &) = delete;
  PYB26_HIDDEN ~py_callable_owner() noexcept;
  static void store_error(void *input, PyObject *error) noexcept {
    auto &owner = *static_cast<py_callable_owner *>(input);
    auto *channel = owner.errors.load(std::memory_order_acquire);
    if (!channel) {
      auto *candidate =
          binding_new<allocation_site::callback_error_channel, error_channel>();
      if (!candidate) {
        owner.native_failure.fetch_or(2, std::memory_order_release);
        cleanup_preserving_exception(
            [](void *value) noexcept {
              Py_DECREF(static_cast<PyObject *>(value));
            },
            error);
        PyErr_Clear();
        return;
      }
      if (owner.errors.compare_exchange_strong(channel, candidate,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire))
        channel = candidate;
      else
        delete candidate;
    }
    auto retained = python_owner_ref::steal(error, owner.callable.context());
    {
      std::lock_guard lock(channel->mutex);
      if (!channel->pending)
        channel->pending = std::move(retained);
    }
  }
  void restore_error() noexcept {
    PyObject *error = nullptr;
    if (auto *channel = errors.load(std::memory_order_acquire)) {
      std::lock_guard lock(channel->mutex);
      error = channel->pending.release();
    }
    if (error) {
      if (!PyErr_Occurred())
        PyErr_SetRaisedException(error);
      else
        cleanup_preserving_exception(
            [](void *value) noexcept {
              Py_DECREF(static_cast<PyObject *>(value));
            },
            error);
    }
    if (native_failure.load(std::memory_order_acquire)) {
      const auto failure =
          native_failure.exchange(0, std::memory_order_acq_rel);
      if (!PyErr_Occurred()) {
        if (failure & 2)
          PyErr_NoMemory();
        else if (failure)
          PyErr_SetString(PyExc_RuntimeError,
                          "callback interpreter is closed or cannot attach");
      }
    }
  }
};

class callback_owner_ref {
  py_callable_owner *pointer_ = nullptr;
  struct adopted {};
  callback_owner_ref(py_callable_owner *owner, adopted) noexcept
      : pointer_(owner) {}

public:
  callback_owner_ref() noexcept = default;
  static callback_owner_ref initial(py_callable_owner *owner) noexcept {
    owner->reserve_initial_target();
    return callback_owner_ref(owner, adopted{});
  }
  static callback_owner_ref retain(py_callable_owner *owner) noexcept {
    owner->retain_target();
    return callback_owner_ref(owner, adopted{});
  }
  callback_owner_ref(const callback_owner_ref &other) noexcept
      : pointer_(other.pointer_) {
    if (pointer_)
      pointer_->retain_target();
  }
  callback_owner_ref(callback_owner_ref &&other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)) {}
  callback_owner_ref &operator=(const callback_owner_ref &other) noexcept {
    if (pointer_ != other.pointer_) {
      if (other.pointer_)
        other.pointer_->retain_target();
      reset(other.pointer_);
    }
    return *this;
  }
  callback_owner_ref &operator=(callback_owner_ref &&other) noexcept {
    if (this != &other)
      reset(std::exchange(other.pointer_, nullptr));
    return *this;
  }
  ~callback_owner_ref() { reset(); }
  void reset() noexcept { reset(nullptr); }
  py_callable_owner *get() const noexcept { return pointer_; }
  py_callable_owner *operator->() const noexcept { return pointer_; }
  explicit operator bool() const noexcept { return pointer_ != nullptr; }
  long use_count() const noexcept {
    return pointer_ ? pointer_->use_count() : 0;
  }

private:
  void reset(py_callable_owner *owner) noexcept {
    if (auto *previous = std::exchange(pointer_, owner))
      previous->release_target();
  }
};

class callback_binding_owner {
  py_callable_owner *pointer_ = nullptr;

public:
  callback_binding_owner() noexcept = default;
  explicit callback_binding_owner(py_callable_owner *owner) noexcept
      : pointer_(owner) {}
  callback_binding_owner(const callback_binding_owner &) = delete;
  ~callback_binding_owner() { reset(); }
  void reset(py_callable_owner *owner = nullptr) noexcept {
    if (auto *previous = std::exchange(pointer_, owner))
      previous->release_binding();
  }
  void finish_admitted(const owner_admission &binding,
                       PyThreadState *state) noexcept {
    if (auto *previous = std::exchange(pointer_, nullptr))
      previous->release_binding(&binding, state);
  }
  py_callable_owner *get() const noexcept { return pointer_; }
  py_callable_owner *operator->() const noexcept { return pointer_; }
  explicit operator bool() const noexcept { return pointer_ != nullptr; }
  long use_count() const noexcept {
    return pointer_ ? pointer_->use_count() : 0;
  }
};

} // namespace pyb26::detail
