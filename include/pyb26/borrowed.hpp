#pragma once

#include "detail/instance.hpp"
#include "detail/lifetime.hpp"

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace pyb26 {

template <class T> class borrowed_alias {
  T *pointer_ = nullptr;
  detail::python_owner_ref owner_{};

public:
  borrowed_alias() noexcept = default;
  borrowed_alias(T *pointer, detail::python_owner_ref &&owner) noexcept
      : pointer_(pointer), owner_(std::move(owner)) {}
  borrowed_alias(borrowed_alias &&other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)),
        owner_(std::move(other.owner_)) {}
  borrowed_alias &operator=(borrowed_alias &&other) noexcept {
    if (this != &other) {
      owner_ = std::move(other.owner_);
      pointer_ = std::exchange(other.pointer_, nullptr);
    }
    return *this;
  }
  borrowed_alias(const borrowed_alias &) = delete;
  borrowed_alias &operator=(const borrowed_alias &) = delete;
  T *get() const noexcept { return pointer_; }
  PyObject *owner() const noexcept { return owner_.get(); }
  T &operator*() const noexcept { return *pointer_; }
  T *operator->() const noexcept { return pointer_; }
};

template <class T> class borrowed_caster {
  T *pointer_ = nullptr;
  detail::python_owner_ref owner_{};

public:
  static constexpr caster_effects effects{false, true, false, true};
  bool retain_alias(PyObject *owner, T *pointer) noexcept {
    if (!owner || !pointer) {
      owner_ = {};
      pointer_ = nullptr;
      return false;
    }
    detail::python_owner_ref retained(owner);
    owner_ = std::move(retained);
    pointer_ = pointer;
    return true;
  }
  bool load(PyObject *object) noexcept {
    return retain_alias(object, detail::cast_to<std::remove_cv_t<T>>(object));
  }
  template <class A> decltype(auto) as() const noexcept {
    static_assert(std::is_pointer_v<A> ||
                      (std::is_lvalue_reference_v<A> &&
                       !std::is_pointer_v<std::remove_reference_t<A>>),
                  "borrowed caster arguments must be native lvalue references "
                  "or pointers, not value copies or moves");
    using Requested =
        std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<A>>>;
    static_assert(std::is_same_v<Requested, std::remove_cv_t<T>>,
                  "borrowed caster alias has a different native type");
    if constexpr (std::is_pointer_v<A>)
      return static_cast<A>(pointer_);
    else
      return static_cast<A>(*pointer_);
  }
  T *get() const noexcept { return pointer_; }
  PyObject *owner() const noexcept { return owner_.get(); }
  borrowed_alias<T> release_alias() noexcept {
    T *pointer = std::exchange(pointer_, nullptr);
    return {pointer, std::move(owner_)};
  }
};

template <class T> class borrowed_pointer_caster {
  caster_for<std::remove_cv_t<T>> caster_;
  bool none_ = false;

public:
  static constexpr caster_effects effects{false, true, false, true};
  bool load(PyObject *object) noexcept {
    none_ = object == Py_None;
    if (none_) {
      (void)caster_.retain_alias(nullptr, nullptr);
      return true;
    }
    return caster_.load(object);
  }
  template <class A> A as() const noexcept {
    static_assert(std::is_pointer_v<A> &&
                      std::is_same_v<std::remove_cv_t<std::remove_pointer_t<A>>,
                                     std::remove_cv_t<T>>,
                  "borrowed pointer caster requires a matching native pointer");
    return none_ ? nullptr : caster_.template as<A>();
  }
};

template <class T> class borrowed_argument_storage {
  std::vector<borrowed_alias<T>> aliases_;

public:
  bool load_and_retain(PyObject *object) noexcept {
    caster_for<T> caster;
    if (!caster.load(object))
      return false;
    aliases_.push_back(caster.release_alias());
    return true;
  }
  std::size_t size() const noexcept { return aliases_.size(); }
  T *operator[](std::size_t index) const noexcept {
    return aliases_[index].get();
  }
};

} // namespace pyb26
