#pragma once

#include "allocation.hpp"
#include "common.hpp"
#include "gil.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace pyb26::detail {

class python_owner_ref;
struct owner_admission;
inline thread_local owner_admission *owner_admissions = nullptr;

struct python_interpreter_owner
    : std::enable_shared_from_this<python_interpreter_owner> {
private:
  class creation_key {
    friend struct python_interpreter_owner;
    explicit creation_key() = default;
  };

public:
  static constexpr std::uint64_t closing = std::uint64_t{1} << 63;
  static constexpr std::uint64_t dead = std::uint64_t{1} << 62;
  static constexpr std::uint64_t count_mask = dead - 1;
  PyInterpreterState *interpreter;
  std::atomic<std::uint64_t> state{0};
  std::mutex mutex;
  python_owner_ref *owners = nullptr;
  std::vector<PyObject *> deferred_cleanup;
  std::mutex error_mutex;
  std::unique_ptr<python_owner_ref> pending_error;
  explicit python_interpreter_owner(creation_key,
                                    PyInterpreterState *value) noexcept
      : interpreter(value) {}
  static std::shared_ptr<python_interpreter_owner>
  create(PyInterpreterState *interpreter) noexcept;
  bool close() noexcept;
  void store_error(PyObject *) noexcept;
  PyObject *take_error() noexcept;
  PYB26_COLD PYB26_NOINLINE void release_closing_admission() noexcept {
    auto pin = shared_from_this();
    const auto prior = state.fetch_sub(1, std::memory_order_release);
    if ((prior & count_mask) == 1)
      state.notify_all();
  }
};

struct python_owner_context {
  PyInterpreterState *interpreter = nullptr;
  std::shared_ptr<python_interpreter_owner> lifetime;
  bool alive() const noexcept {
    return lifetime && !(lifetime->state.load(std::memory_order_acquire) &
                         python_interpreter_owner::closing);
  }
  bool close() const noexcept { return lifetime && lifetime->close(); }
  PyObject *take_error() const noexcept {
    return lifetime ? lifetime->take_error() : nullptr;
  }
};

enum class owner_admission_kind { invocation, retention, cleanup };

struct owner_admission {
  python_interpreter_owner *owner = nullptr;
  owner_admission *previous = nullptr;
  owner_admission *next = nullptr;
  bool counted = false;
  void link() noexcept {
    previous = std::exchange(owner_admissions, this);
    if (previous)
      previous->next = this;
  }
  struct borrowed {};
  owner_admission(owner_admission &peer, borrowed) noexcept
      : owner(peer.owner) {
    if (owner)
      link();
  }
  explicit owner_admission(
      const python_owner_context &context,
      owner_admission_kind kind = owner_admission_kind::invocation) noexcept {
    auto *control = context.lifetime.get();
    if (!control)
      return;
    auto state = control->state.load(std::memory_order_relaxed);
    if ((state & python_interpreter_owner::dead) ||
        (kind == owner_admission_kind::invocation &&
         (state & python_interpreter_owner::closing)))
      return;
    for (auto *entry = owner_admissions; entry; entry = entry->previous)
      if (entry->owner == control) {
        owner = control;
        link();
        return;
      }
    do {
      if ((state & python_interpreter_owner::closing) &&
          (kind != owner_admission_kind::cleanup ||
           !(state & python_interpreter_owner::count_mask)))
        return;
    } while (!control->state.compare_exchange_weak(state, state + 1,
                                                   std::memory_order_acquire,
                                                   std::memory_order_relaxed));
    owner = control;
    counted = true;
    link();
  }
  ~owner_admission() {
    if (owner) {
      if (next)
        next->previous = previous;
      else
        owner_admissions = previous;
      if (previous)
        previous->next = next;
      if (!counted)
        return;
      for (auto *entry = owner_admissions; entry; entry = entry->previous)
        if (entry->owner == owner) {
          entry->counted = true;
          return;
        }
      auto state = owner->state.load(std::memory_order_relaxed);
      while (!(state & python_interpreter_owner::closing)) {
        if (owner->state.compare_exchange_weak(state, state - 1,
                                               std::memory_order_release,
                                               std::memory_order_relaxed))
          return;
      }
      owner->release_closing_admission();
    }
  }
  explicit operator bool() const noexcept { return owner != nullptr; }
  owner_admission(const owner_admission &) = delete;
};

inline python_owner_context capture_python_owner_context() noexcept;
inline thread_local python_interpreter_owner *closing_python_owner = nullptr;

class callback_error_scope {
  PyThreadState *state_;
  PyObject *primary_ = nullptr;

public:
  explicit callback_error_scope(
      PyThreadState *state = cpython::thread_state()) noexcept
      : state_(state) {}
  callback_error_scope(const callback_error_scope &) = delete;
  ~callback_error_scope() { restore(); }
  bool capture() noexcept {
    if (!primary_ && ::pyb26::detail::cpython::current_exception(state_))
        [[unlikely]]
      primary_ = retain_python_error(
          ::pyb26::detail::cpython::current_exception(state_));
    return primary_ != nullptr;
  }
  void restore() noexcept {
    if (primary_) [[unlikely]]
      finish_python_error(state_, std::exchange(primary_, nullptr), nullptr);
  }
  void discard(PyObject *value) noexcept {
    capture();
    Py_XDECREF(value);
    capture();
    restore();
  }
};

struct callback_error_checkpoint {
  callback_error_scope &error;
  ~callback_error_checkpoint() { error.capture(); }
};

inline void cleanup_preserving_exception(PyThreadState *state,
                                         void (*cleanup)(void *) noexcept,
                                         void *value) noexcept {
  PyObject *error =
      cpython::current_exception(state) ? PyErr_GetRaisedException() : nullptr;
  cleanup(value);
  if (cpython::current_exception(state)) [[unlikely]]
    PyErr_Clear();
  if (error) [[unlikely]]
    PyErr_SetRaisedException(error);
}

inline void cleanup_preserving_exception(void (*cleanup)(void *) noexcept,
                                         void *value) noexcept {
  cleanup_preserving_exception(cpython::thread_state(), cleanup, value);
}

inline bool run_attached_admitted(const python_owner_context &context,
                                  PyThreadState *current,
                                  void (*cleanup)(void *) noexcept,
                                  void *value) noexcept {
  if (!cleanup || !context.interpreter)
    return false;
  if (current && current->interp == context.interpreter) [[likely]] {
    cleanup_preserving_exception(current, cleanup, value);
    return true;
  }
  if (!Py_IsInitialized() || Py_IsFinalizing())
    return false;

  if (!current && released_python_state &&
      released_python_state->interp == context.interpreter) {
    PyEval_RestoreThread(released_python_state);
    cleanup_preserving_exception(released_python_state, cleanup, value);
    PyEval_SaveThread();
    return true;
  }

  PyThreadState *previous = current ? PyEval_SaveThread() : nullptr;
  PyThreadState *attached = PyThreadState_New(context.interpreter);
  if (!attached) {
    if (previous)
      PyEval_RestoreThread(previous);
    return false;
  }
  PyEval_AcquireThread(attached);
  cleanup_preserving_exception(attached, cleanup, value);
  PyThreadState_Clear(attached);
  PyThreadState_DeleteCurrent();
  if (previous)
    PyEval_RestoreThread(previous);
  return true;
}

inline bool run_attached_admitted(const python_owner_context &context,
                                  void (*cleanup)(void *) noexcept,
                                  void *value) noexcept {
  return run_attached_admitted(context, cpython::attached_thread_state(),
                               cleanup, value);
}

inline bool run_attached(const python_owner_context &context,
                         void (*cleanup)(void *) noexcept,
                         void *value) noexcept {
  owner_admission admission(context, owner_admission_kind::cleanup);
  return admission && run_attached_admitted(context, cleanup, value);
}

class python_owner_ref {
  std::atomic<PyObject *> owner_{nullptr};
  python_owner_context context_{};
  python_owner_ref *previous_ = nullptr;
  python_owner_ref *next_ = nullptr;
  bool linked_ = false;
  bool registered_ = false;
  friend struct python_interpreter_owner;

  void unlink_locked() noexcept {
    if (!linked_)
      return;
    if (previous_)
      previous_->next_ = next_;
    else
      context_.lifetime->owners = next_;
    if (next_)
      next_->previous_ = previous_;
    previous_ = next_ = nullptr;
    linked_ = false;
  }
  void install(PyObject *owner, python_owner_context context) noexcept {
    context_ = std::move(context);
    if (!owner || !context_.lifetime)
      return;
    std::lock_guard lock(context_.lifetime->mutex);
    if (context_.lifetime->state.load(std::memory_order_acquire) &
        python_interpreter_owner::dead)
      return;
    owner_.store(owner, std::memory_order_release);
    next_ = context_.lifetime->owners;
    if (next_)
      next_->previous_ = this;
    context_.lifetime->owners = this;
    linked_ = true;
    registered_ = true;
  }
  void move_from(python_owner_ref &other) noexcept {
    context_ = other.context_;
    if (!context_.lifetime)
      return;
    std::lock_guard lock(context_.lifetime->mutex);
    registered_ = std::exchange(other.registered_, false);
    owner_.store(other.owner_.exchange(nullptr, std::memory_order_acq_rel),
                 std::memory_order_release);
    if (other.linked_) {
      previous_ = other.previous_;
      next_ = other.next_;
      linked_ = true;
      if (previous_)
        previous_->next_ = this;
      else
        context_.lifetime->owners = this;
      if (next_)
        next_->previous_ = this;
      other.previous_ = other.next_ = nullptr;
      other.linked_ = false;
    }
  }
  static void decref(void *value) noexcept {
    Py_DECREF(static_cast<PyObject *>(value));
  }

public:
  python_owner_ref() noexcept = default;
  python_owner_ref(PyObject *owner, python_owner_context context,
                   const owner_admission &binding) noexcept
      : context_(std::move(context)) {
    if (owner && binding && binding.owner == context_.lifetime.get())
      owner_.store(Py_NewRef(owner), std::memory_order_release);
    else
      PyErr_SetString(PyExc_RuntimeError,
                      "callback owner requires binding admission");
  }
  void register_escape(const owner_admission &binding) noexcept {
    if (registered_ || !get())
      return;
    if (!binding || binding.owner != context_.lifetime.get()) {
      PyErr_SetString(PyExc_RuntimeError,
                      "callback owner escaped its binding admission");
      return;
    }
    std::lock_guard lock(context_.lifetime->mutex);
    next_ = context_.lifetime->owners;
    if (next_)
      next_->previous_ = this;
    context_.lifetime->owners = this;
    linked_ = registered_ = true;
  }
  python_owner_ref(PyObject *owner, python_owner_context context) noexcept {
    if (!owner)
      return;
    owner_admission admission(context, owner_admission_kind::retention);
    if (admission)
      install(Py_NewRef(owner), std::move(context));
    else if (!PyErr_Occurred())
      PyErr_SetString(PyExc_RuntimeError,
                      "Python owner interpreter is closing or closed");
  }
  explicit python_owner_ref(PyObject *owner) noexcept
      : python_owner_ref(owner, owner ? capture_python_owner_context()
                                      : python_owner_context{}) {}
  static python_owner_ref steal(PyObject *owner) noexcept {
    if (!owner)
      return {};
    return steal(owner, capture_python_owner_context());
  }
  static python_owner_ref steal(PyObject *owner,
                                python_owner_context context) noexcept {
    python_owner_ref result;
    if (!owner)
      return result;
    owner_admission admission(context, owner_admission_kind::retention);
    if (admission)
      result.install(owner, std::move(context));
    else {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_RuntimeError,
                        "Python owner interpreter is closing or closed");
      cleanup_preserving_exception(&decref, owner);
    }
    return result;
  }
  ~python_owner_ref() noexcept { reset(); }
  python_owner_ref(const python_owner_ref &) = delete;
  python_owner_ref &operator=(const python_owner_ref &) = delete;
  python_owner_ref(python_owner_ref &&other) noexcept { move_from(other); }
  python_owner_ref &operator=(python_owner_ref &&other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }
  PyObject *get() const noexcept {
    return owner_.load(std::memory_order_acquire);
  }
  explicit operator bool() const noexcept { return get() != nullptr; }
  const python_owner_context &context() const noexcept { return context_; }
  PyObject *release() noexcept {
    if (!context_.lifetime)
      return nullptr;
    if (!registered_)
      return owner_.exchange(nullptr, std::memory_order_acq_rel);
    std::lock_guard lock(context_.lifetime->mutex);
    unlink_locked();
    PyObject *result = owner_.exchange(nullptr, std::memory_order_acq_rel);
    return result;
  }
  void reset_admitted(const owner_admission &admission,
                      PyThreadState *state) noexcept {
    assert(admission && admission.owner == context_.lifetime.get());
    assert(state && state->interp == context_.interpreter);
    PyObject *owner = release();
    if (owner)
      cleanup_preserving_exception(state, &decref, owner);
    context_ = {};
    registered_ = false;
  }
  void reset() noexcept {
    if (!context_.lifetime)
      return;
    owner_admission admission(context_, owner_admission_kind::cleanup);
    auto *current = ::pyb26::detail::cpython::attached_thread_state();
    if (!admission && closing_python_owner != context_.lifetime.get()) {
      PyThreadState *saved = current ? PyEval_SaveThread() : nullptr;
      auto &state = context_.lifetime->state;
      for (auto value = state.load(std::memory_order_acquire);
           !(value & python_interpreter_owner::dead);
           value = state.load(std::memory_order_acquire))
        state.wait(value, std::memory_order_acquire);
      if (saved)
        PyEval_RestoreThread(saved);
      std::lock_guard lock(context_.lifetime->mutex);
      unlink_locked();
      return;
    }
    PyObject *owner = release();
    if (owner && !run_attached_admitted(context_, current, &decref, owner)) {
      std::lock_guard lock(context_.lifetime->mutex);
      context_.lifetime->deferred_cleanup.push_back(owner);
    }
  }
};

inline bool python_interpreter_owner::close() noexcept {
  if (closing_python_owner == this) {
    PyErr_SetString(PyExc_RuntimeError,
                    "interpreter owner close is already running");
    return false;
  }
  for (auto *entry = owner_admissions; entry; entry = entry->previous)
    if (entry->owner == this) {
      PyErr_SetString(
          PyExc_RuntimeError,
          "cannot close an interpreter owner during its active invocation");
      return false;
    }
  auto *current = ::pyb26::detail::cpython::attached_thread_state();
  if (!current || current->interp != interpreter) {
    if (current)
      PyErr_SetString(PyExc_RuntimeError,
                      "interpreter owner must be closed in its interpreter");
    return false;
  }
  const auto prior = state.fetch_or(closing, std::memory_order_acq_rel);
  if (prior & dead)
    return true;
  auto wait = [&](bool for_dead) {
    PyThreadState *saved = PyEval_SaveThread();
    for (auto value = state.load(std::memory_order_acquire);
         for_dead ? !(value & dead) : (value & count_mask) != 0;
         value = state.load(std::memory_order_acquire))
      state.wait(value, std::memory_order_acquire);
    PyEval_RestoreThread(saved);
  };
  if (prior & closing) {
    wait(true);
    return true;
  }
  auto *previous_closing = std::exchange(closing_python_owner, this);
  wait(false);
  for (;;) {
    PyObject *value;
    {
      std::lock_guard lock(mutex);
      auto *entry = owners;
      if (entry) {
        value = entry->owner_.exchange(nullptr, std::memory_order_acq_rel);
        entry->unlink_locked();
        entry->owner_.notify_all();
      } else if (!deferred_cleanup.empty()) {
        value = deferred_cleanup.back();
        deferred_cleanup.pop_back();
      } else {
        break;
      }
    }
    cleanup_preserving_exception(
        [](void *value) noexcept {
          Py_XDECREF(static_cast<PyObject *>(value));
        },
        value);
  }
  {
    std::lock_guard lock(error_mutex);
    pending_error.reset();
  }
  state.fetch_or(dead, std::memory_order_release);
  state.notify_all();
  closing_python_owner = previous_closing;
  return true;
}

inline void python_interpreter_owner::store_error(PyObject *error) noexcept {
  if (!error)
    return;
  std::lock_guard lock(error_mutex);
  if (!pending_error || !pending_error->get())
    pending_error = std::make_unique<python_owner_ref>(python_owner_ref::steal(
        error, python_owner_context{interpreter, shared_from_this()}));
  else
    cleanup_preserving_exception(
        [](void *value) noexcept { Py_DECREF(static_cast<PyObject *>(value)); },
        error);
}

inline PyObject *python_interpreter_owner::take_error() noexcept {
  auto *current = ::pyb26::detail::cpython::attached_thread_state();
  if (!current || current->interp != interpreter)
    return nullptr;
  std::lock_guard lock(error_mutex);
  return pending_error ? pending_error->release() : nullptr;
}

inline std::shared_ptr<python_interpreter_owner>
python_interpreter_owner::create(PyInterpreterState *interpreter) noexcept {
  auto control =
      std::make_shared<python_interpreter_owner>(creation_key{}, interpreter);
  auto *fallback =
      new (std::nothrow) std::shared_ptr<python_interpreter_owner>(control);
  if (!fallback ||
      cpython::at_interpreter_exit(
          interpreter,
          [](void *input) noexcept {
            auto *owner =
                static_cast<std::shared_ptr<python_interpreter_owner> *>(input);
            if ((*owner)->close())
              delete owner;
          },
          fallback) < 0) {
    delete fallback;
    if (!PyErr_Occurred())
      PyErr_NoMemory();
    return {};
  }
  return control;
}

inline PyObject *close_python_owner(PyObject *capsule, PyObject *) noexcept {
  auto *owner = static_cast<std::shared_ptr<python_interpreter_owner> *>(
      PyCapsule_GetPointer(capsule, "pyb26.interpreter_owner"));
  if (!owner || !(*owner)->close())
    return nullptr;
  return Py_NewRef(Py_None);
}

inline python_owner_context
capture_attached_python_owner_context(PyThreadState *state) noexcept {
  if (!state)
    return {};
  static thread_local python_owner_context cache;
  if (cache.interpreter == state->interp && cache.alive())
    return cache;
  static const auto key = [] {
    std::array<char, 96> name{};
    std::snprintf(name.data(), name.size(), "_pyb26_owner_%p",
                  reinterpret_cast<void *>(&close_python_owner));
    return name;
  }();
  PyObject *dict = PyInterpreterState_GetDict(state->interp);
  if (!dict)
    return {};
  PyObject *capsule = nullptr;
  const int found = PyDict_GetItemStringRef(dict, key.data(), &capsule);
  if (found < 0)
    return {};
  if (found) {
    auto *owner = static_cast<std::shared_ptr<python_interpreter_owner> *>(
        PyCapsule_GetPointer(capsule, "pyb26.interpreter_owner"));
    if (!owner) {
      Py_DECREF(capsule);
      return {};
    }
    cache = {state->interp, *owner};
    Py_DECREF(capsule);
    return cache;
  }
  PyObject *atexit = PyImport_ImportModule("atexit");
  if (!atexit)
    return {};
  PyObject *registration = PyObject_GetAttrString(atexit, "register");
  Py_DECREF(atexit);
  if (!registration)
    return {};
  auto control = python_interpreter_owner::create(state->interp);
  if (!control) {
    Py_DECREF(registration);
    return {};
  }
  auto *payload =
      new (std::nothrow) std::shared_ptr<python_interpreter_owner>(control);
  if (!payload) {
    Py_DECREF(registration);
    PyErr_NoMemory();
    return {};
  }
  capsule =
      PyCapsule_New(payload, "pyb26.interpreter_owner", [](PyObject *value) {
        delete static_cast<std::shared_ptr<python_interpreter_owner> *>(
            PyCapsule_GetPointer(value, "pyb26.interpreter_owner"));
      });
  if (!capsule) {
    delete payload;
    Py_DECREF(registration);
    return {};
  }
  static PyMethodDef method{"_close_pyb26_owners", &close_python_owner,
                            METH_NOARGS, nullptr};
  PyObject *function = PyCFunction_New(&method, capsule);
  PyObject *registered =
      function ? PyObject_CallOneArg(registration, function) : nullptr;
  Py_DECREF(registration);
  Py_XDECREF(function);
  PyObject *key_object = PyUnicode_FromString(key.data());
  PyObject *winner = nullptr;
  const int inserted =
      registered && key_object
          ? PyDict_SetDefaultRef(dict, key_object, capsule, &winner)
          : -1;
  Py_XDECREF(key_object);
  if (inserted < 0) {
    Py_XDECREF(registered);
    Py_DECREF(capsule);
    return {};
  }
  Py_DECREF(registered);
  Py_DECREF(capsule);
  auto *chosen = static_cast<std::shared_ptr<python_interpreter_owner> *>(
      PyCapsule_GetPointer(winner, "pyb26.interpreter_owner"));
  if (!chosen) {
    Py_DECREF(winner);
    return {};
  }
  cache = {state->interp, *chosen};
  Py_DECREF(winner);
  return cache;
}

inline python_owner_context capture_python_owner_context() noexcept {
  return capture_attached_python_owner_context(
      cpython::attached_thread_state());
}

class callback_invocation_scope {
  enum class attachment { borrowed, restored, created };
  owner_admission admission_;
  python_interpreter_owner *owner_ = nullptr;
  PyInterpreterState *interpreter_ = nullptr;
  PyThreadState *state_ = nullptr;
  attachment attachment_ = attachment::borrowed;
  bool valid_ = false;
  void (*error_sink_)(void *, PyObject *) noexcept = nullptr;
  void *sink_ = nullptr;

public:
  explicit callback_invocation_scope(
      const python_owner_context &owner,
      void (*error_sink)(void *, PyObject *) noexcept = nullptr,
      void *sink = nullptr) noexcept
      : admission_(owner), owner_(admission_ ? owner.lifetime.get() : nullptr),
        interpreter_(admission_ ? owner.interpreter : nullptr),
        error_sink_(error_sink), sink_(sink) {
    auto *current = state_ = cpython::attached_thread_state();
    if (!admission_) {
      if (current)
        PyErr_SetString(PyExc_RuntimeError,
                        "callback interpreter is closing or closed");
      return;
    }
    if (current) {
      valid_ = current->interp == interpreter_;
      if (!valid_)
        PyErr_SetString(PyExc_RuntimeError,
                        "pyb26: callback belongs to another interpreter");
      return;
    }
    if (!Py_IsInitialized() || Py_IsFinalizing())
      return;
    if (auto *released = released_python_state;
        released && released->interp == interpreter_) {
      state_ = released;
      PyEval_RestoreThread(state_);
      attachment_ = attachment::restored;
    } else {
      state_ = PyThreadState_New(interpreter_);
      if (!state_)
        return;
      PyEval_AcquireThread(state_);
      attachment_ = attachment::created;
    }
    valid_ = true;
  }
  ~callback_invocation_scope() {
    if (attachment_ == attachment::borrowed)
      return;
    if (attachment_ == attachment::created) {
      if (PyObject *error = PyErr_GetRaisedException()) {
        if (error_sink_)
          error_sink_(sink_, error);
        else
          owner_->store_error(error);
      }
      PyThreadState_Clear(state_);
      if (PyObject *error = PyErr_GetRaisedException()) {
        if (error_sink_)
          error_sink_(sink_, error);
        else
          owner_->store_error(error);
      }
      PyThreadState_DeleteCurrent();
    } else {
      PyEval_SaveThread();
    }
  }
  explicit operator bool() const noexcept { return valid_; }
  PyThreadState *thread_state() const noexcept { return state_; }
  bool created_thread_state() const noexcept {
    return attachment_ == attachment::created;
  }
  callback_invocation_scope(const callback_invocation_scope &) = delete;
};

PyObject **lifetime_slot(PyObject *object) noexcept;

PYB26_NOINLINE inline bool
attach_lifetime_slow(PyObject *nurse, PyObject *patient, PyObject **slot,
                     PyObject *existing, bool list) noexcept {
  auto *instance = reinterpret_cast<detail::instance *>(nurse);
  for (;;) {
    {
      callback_error_scope error;
      if (list) {
        bool matched = false, success = true;
        PyObject *retired = nullptr;
        Py_BEGIN_CRITICAL_SECTION2(nurse, existing);
        if (*slot == existing && (instance->flags & inst_keep_alive_list)) {
          matched = true;
          bool found = false;
          Py_ssize_t hole = -1;
          const auto count = PyList_GET_SIZE(existing);
          for (Py_ssize_t index = 0; index < count; ++index) {
            PyObject *value = PyList_GET_ITEM(existing, index);
            if (value == patient) {
              found = true;
              break;
            }
            if (value == Py_None && hole < 0)
              hole = index;
          }
          if (!found) {
            if (hole >= 0) {
              retired = PyList_GET_ITEM(existing, hole);
              PyList_SET_ITEM(existing, hole, Py_NewRef(patient));
            } else if (count >= reinterpret_cast<PyListObject *>(existing)
                                    ->allocated &&
                       !allocation_admitted<
                           allocation_site::lifetime_owners>()) {
              success = false;
            } else {
              success = PyList_Append(existing, patient) == 0;
              if (success && PyErr_Occurred()) {
                retired = PyList_GET_ITEM(existing, count);
                PyList_SET_ITEM(existing, count, Py_NewRef(Py_None));
                success = false;
              }
            }
          }
        }
        Py_END_CRITICAL_SECTION2();
        error.discard(retired);
        error.discard(existing);
        if (!success || error.capture())
          return false;
        if (matched)
          return true;
      } else {
        PyObject *owners =
            allocation_admitted<allocation_site::lifetime_owners>()
                ? PyList_New(2)
                : nullptr;
        if (!owners || error.capture()) {
          error.discard(owners);
          error.discard(existing);
          return false;
        }
        PyList_SET_ITEM(owners, 0, existing);
        PyList_SET_ITEM(owners, 1, Py_NewRef(patient));
        PyObject *retired = nullptr;
        bool matched = false;
        Py_BEGIN_CRITICAL_SECTION(nurse);
        if (*slot == existing && !(instance->flags & inst_keep_alive_list)) {
          retired = *slot;
          *slot = owners;
          owners = nullptr;
          instance->flags |= inst_keep_alive_list | inst_has_lifetime_patients;
          matched = true;
        }
        Py_END_CRITICAL_SECTION();
        error.discard(retired);
        error.discard(owners);
        if (error.capture())
          return false;
        if (matched)
          return true;
      }
    }
    existing = nullptr;
    list = false;
    bool done = false;
    Py_BEGIN_CRITICAL_SECTION(nurse);
    if (!*slot) {
      *slot = Py_NewRef(patient);
      instance->flags |= inst_has_lifetime_patients;
      done = true;
    } else if (!(instance->flags & inst_keep_alive_list) && *slot == patient) {
      done = true;
    } else {
      existing = Py_NewRef(*slot);
      list = instance->flags & inst_keep_alive_list;
    }
    Py_END_CRITICAL_SECTION();
    if (done)
      return true;
  }
}

template <class SlotResolver>
PYB26_ALWAYS_INLINE bool
attach_lifetime_impl(PyObject *nurse, PyObject *patient,
                     SlotResolver resolve_slot) noexcept {
  if (PyErr_Occurred())
    return false;
  if (!nurse || !patient) {
    PyErr_SetString(PyExc_TypeError, "keep_alive requires two Python objects");
    return false;
  }
  if (nurse == Py_None || patient == Py_None || nurse == patient)
    return true;
  PyObject **slot = resolve_slot(nurse);
  if (!slot) {
    PyErr_SetString(
        PyExc_TypeError,
        "keep_alive nurse must be a pyb26 class registered with gc()");
    return false;
  }
  auto *instance = reinterpret_cast<detail::instance *>(nurse);
  PyObject *existing = nullptr;
  bool list = false, done = false;
  Py_BEGIN_CRITICAL_SECTION(nurse);
  if (!*slot) {
    *slot = Py_NewRef(patient);
    instance->flags |= inst_has_lifetime_patients;
    done = true;
  } else if (!(instance->flags & inst_keep_alive_list) && *slot == patient) {
    done = true;
  } else {
    existing = Py_NewRef(*slot);
    list = instance->flags & inst_keep_alive_list;
  }
  Py_END_CRITICAL_SECTION();
  return done || attach_lifetime_slow(nurse, patient, slot, existing, list);
}

inline void detach_lifetime_token(PyObject *nurse, PyObject *token) noexcept {
  PyObject **slot = nurse && token ? lifetime_slot(nurse) : nullptr;
  if (!slot)
    return;
  auto *instance = reinterpret_cast<detail::instance *>(nurse);
  callback_error_scope error;
  for (;;) {
    PyObject *existing = nullptr, *retired = nullptr;
    bool done = false;
    Py_BEGIN_CRITICAL_SECTION(nurse);
    if (!*slot) {
      done = true;
    } else if (!(instance->flags & inst_keep_alive_list)) {
      if (*slot == token) {
        retired = std::exchange(*slot, nullptr);
        instance->flags &= static_cast<unsigned char>(
            ~(inst_keep_alive_list | inst_has_lifetime_patients));
      }
      done = true;
    } else {
      existing = Py_NewRef(*slot);
    }
    Py_END_CRITICAL_SECTION();
    error.discard(retired);
    if (done)
      return;
    Py_BEGIN_CRITICAL_SECTION2(nurse, existing);
    if (*slot == existing && (instance->flags & inst_keep_alive_list)) {
      const auto count = PyList_GET_SIZE(existing);
      for (Py_ssize_t index = 0; index < count; ++index)
        if (PyList_GET_ITEM(existing, index) == token) {
          retired = token;
          PyList_SET_ITEM(existing, index, Py_NewRef(Py_None));
          break;
        }
      done = true;
    }
    Py_END_CRITICAL_SECTION2();
    error.discard(retired);
    error.discard(existing);
    if (done)
      return;
  }
}

inline bool attach_lifetime(PyObject *nurse, PyObject *patient) noexcept {
  return attach_lifetime_impl(nurse, patient, [](PyObject *object) noexcept {
    return lifetime_slot(object);
  });
}

template <class NativeT>
inline bool attach_lifetime_for(PyObject *nurse, PyObject *patient) noexcept {
  return attach_lifetime_impl(nurse, patient, [](PyObject *object) noexcept {
    return typed_lifetime_slot(object, &type_chain_of<NativeT>);
  });
}

inline bool attach_lifetime(PyObject *nurse, PyObject *patient,
                            const type_record &record) noexcept {
  return attach_lifetime_impl(
      nurse, patient, [&record](PyObject *object) noexcept -> PyObject ** {
        PyTypeObject *type = Py_TYPE(object);
        if (type != record.py_type || type->tp_is_gc) [[unlikely]]
          return lifetime_slot(object);
        if (!PyObject_IS_GC(object)) [[unlikely]]
          return nullptr;
        const Py_ssize_t offset = record.lifetime_offset;
        if (offset < static_cast<Py_ssize_t>(sizeof(instance)) ||
            offset > type->tp_basicsize - static_cast<Py_ssize_t>(
                                              sizeof(PyObject *))) [[unlikely]]
          return nullptr;
        return reinterpret_cast<PyObject **>(reinterpret_cast<char *>(object) +
                                             offset);
      });
}

inline int traverse_lifetime(PyObject *object, visitproc visit,
                             void *arg) noexcept {
  if (!(reinterpret_cast<instance *>(object)->flags &
        inst_has_lifetime_patients))
    return 0;
  if (PyObject **slot = lifetime_slot_gc(object))
    Py_VISIT(*slot);
  return 0;
}

inline void clear_lifetime(PyObject *object) noexcept {
  if (!(reinterpret_cast<instance *>(object)->flags &
        inst_has_lifetime_patients))
    return;
  PyObject **slot = lifetime_slot(object);
  if (!slot)
    return;
  PyObject *owners;
  Py_BEGIN_CRITICAL_SECTION(object);
  owners = std::exchange(*slot, nullptr);
  reinterpret_cast<instance *>(object)->flags &= static_cast<unsigned char>(
      ~(inst_keep_alive_list | inst_has_lifetime_patients));
  Py_END_CRITICAL_SECTION();
  callback_error_scope error;
  error.discard(owners);
}

} // namespace pyb26::detail

namespace pyb26 {

using invocation_context = detail::python_owner_context;

inline invocation_context capture_invocation_context() noexcept {
  return detail::capture_python_owner_context();
}

class invocation_scope {
  detail::callback_invocation_scope scope_;

public:
  explicit invocation_scope(invocation_context owner) noexcept
      : scope_(std::move(owner)) {}
  explicit operator bool() const noexcept { return static_cast<bool>(scope_); }
  invocation_scope(const invocation_scope &) = delete;
};

} // namespace pyb26
