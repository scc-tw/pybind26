#pragma once

#include "detail/lifetime.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace pyb26 {

template <std::size_t Nurse, std::size_t Patient> struct keep_alive {
  static constexpr std::size_t nurse = Nurse;
  static constexpr std::size_t patient = Patient;
};

template <std::size_t Slot> struct return_owner {
  static_assert(Slot != 0, "pyb26: return_owner uses one-based argument slots");
  static constexpr std::size_t slot = Slot;
};

inline bool keep_alive_attach(PyObject *nurse, PyObject *patient) noexcept {
  return detail::attach_lifetime(nurse, patient);
}

template <class Nurse, class Patient>
  requires requires(const Nurse &nurse, const Patient &patient) {
    { nurse.ptr() } -> std::convertible_to<PyObject *>;
    { patient.ptr() } -> std::convertible_to<PyObject *>;
  }
inline bool keep_alive_attach(const Nurse &nurse,
                              const Patient &patient) noexcept {
  return detail::attach_lifetime(nurse.ptr(), patient.ptr());
}

namespace detail {

template <class Extra> inline constexpr bool is_return_owner_v = false;
template <std::size_t Slot>
inline constexpr bool is_return_owner_v<return_owner<Slot>> = true;
template <class... Extras>
inline constexpr std::size_t return_owner_slot_v = [] {
  static_assert(
      (0 + ... + (is_return_owner_v<std::remove_cvref_t<Extras>> ? 1 : 0)) <= 1,
      "pyb26: specify only one return_owner");
  std::size_t slot = 0;
  (
      [&] {
        if constexpr (is_return_owner_v<std::remove_cvref_t<Extras>>)
          slot = std::remove_cvref_t<Extras>::slot;
      }(),
      ...);
  return slot;
}();

template <class Extra> inline constexpr bool is_keep_alive_v = false;
template <std::size_t Nurse, std::size_t Patient>
inline constexpr bool is_keep_alive_v<keep_alive<Nurse, Patient>> = true;
template <class... Extras>
inline constexpr bool has_keep_alive_v =
    (is_keep_alive_v<std::remove_cvref_t<Extras>> || ...);

template <class Extra>
inline bool validate_one_keep_alive(PyObject *const *slots,
                                    std::size_t count) noexcept {
  using Policy = std::remove_cvref_t<Extra>;
  if constexpr (!is_keep_alive_v<Policy>) {
    (void)slots;
    (void)count;
    return true;
  } else {
    constexpr std::size_t nurse = Policy::nurse;
    constexpr std::size_t patient = Policy::patient;
    if ((nurse && nurse > count) || (patient && patient > count)) {
      PyErr_SetString(
          PyExc_TypeError,
          "keep_alive index outside this call's argument/return slots");
      return false;
    }
    if constexpr (nurse == 0) {
      return true;
    } else {
      PyObject *object = slots[nurse - 1];
      if (object == Py_None)
        return true;
      if constexpr (patient != 0) {
        PyObject *owner = slots[patient - 1];
        if (owner == Py_None || object == owner)
          return true;
      }
      if (!object || !lifetime_slot(object)) {
        PyErr_SetString(
            PyExc_TypeError,
            "keep_alive nurse must be a pyb26 class registered with gc()");
        return false;
      }
      return true;
    }
  }
}

template <class... Extras>
inline bool validate_keep_alive(PyObject *const *slots,
                                std::size_t count) noexcept {
  return (validate_one_keep_alive<Extras>(slots, count) && ...);
}

template <class Extra>
inline bool apply_one_keep_alive(PyObject *result, PyObject *const *slots,
                                 std::size_t count) noexcept {
  using Policy = std::remove_cvref_t<Extra>;
  if constexpr (!is_keep_alive_v<Policy>) {
    (void)result;
    (void)slots;
    (void)count;
    return true;
  } else {
    constexpr std::size_t nurse = Policy::nurse;
    constexpr std::size_t patient = Policy::patient;
    if ((nurse && nurse > count) || (patient && patient > count) ||
        ((!nurse || !patient) && !result)) {
      PyErr_SetString(
          PyExc_TypeError,
          "keep_alive index outside this call's argument/return slots");
      return false;
    }
    return attach_lifetime(nurse ? slots[nurse - 1] : result,
                           patient ? slots[patient - 1] : result);
  }
}

template <class NativeT, class Extra>
inline bool apply_one_keep_alive_native(PyObject *result,
                                        PyObject *const *slots,
                                        std::size_t count) noexcept {
  using Policy = std::remove_cvref_t<Extra>;
  if constexpr (is_keep_alive_v<Policy>) {
    if constexpr (Policy::nurse == 0) {
      constexpr std::size_t patient = Policy::patient;
      if ((patient && patient > count) || !result) {
        PyErr_SetString(
            PyExc_TypeError,
            "keep_alive index outside this call's argument/return slots");
        return false;
      }
      return attach_lifetime_for<NativeT>(result, patient ? slots[patient - 1]
                                                          : result);
    } else {
      return apply_one_keep_alive<Extra>(result, slots, count);
    }
  } else {
    return apply_one_keep_alive<Extra>(result, slots, count);
  }
}

template <class NativeT, class... Extras>
inline bool apply_keep_alive_native(PyObject *result, PyObject *const *slots,
                                    std::size_t count) noexcept {
  return (apply_one_keep_alive_native<NativeT, Extras>(result, slots, count) &&
          ...);
}

template <class... Extras>
inline bool apply_keep_alive(PyObject *result, PyObject *const *slots,
                             std::size_t count) noexcept {
  return (apply_one_keep_alive<Extras>(result, slots, count) && ...);
}

template <std::size_t Capacity> class constructor_lifetime_transaction {
  struct edge {
    PyObject *nurse;
    PyObject *patient;
  };
  struct bundle {
    PyObject *nurse = nullptr;
    PyObject *token = nullptr;
    std::size_t size = 0;
    bool attached = false;
  };
  std::array<edge, Capacity> edges_;
  std::array<bundle, Capacity> bundles_{};
  std::size_t edges_size_ = 0;
  std::size_t bundles_size_ = 0;
  bool committed_ = false;

public:
  ~constructor_lifetime_transaction() {
    callback_error_scope error;
    error.capture();
    if (!committed_)
      for (std::size_t i = bundles_size_; i > 0; --i)
        if (bundles_[i - 1].attached)
          detach_lifetime_token(bundles_[i - 1].nurse, bundles_[i - 1].token);
    for (std::size_t i = 0; i < bundles_size_; ++i)
      error.discard(bundles_[i].token);
  }
  template <class Extra> void stage(PyObject *const *slots) noexcept {
    using Policy = std::remove_cvref_t<Extra>;
    if constexpr (is_keep_alive_v<Policy>) {
      static_assert(Policy::nurse && Policy::patient);
      PyObject *nurse = slots[Policy::nurse - 1];
      PyObject *patient = slots[Policy::patient - 1];
      if (nurse != Py_None && patient != Py_None && nurse != patient)
        edges_[edges_size_++] = {nurse, patient};
    }
  }
  bool commit() noexcept {
    for (std::size_t i = 0; i < edges_size_; ++i) {
      std::size_t index = 0;
      while (index < bundles_size_ && bundles_[index].nurse != edges_[i].nurse)
        ++index;
      if (index == bundles_size_)
        bundles_[bundles_size_++].nurse = edges_[i].nurse;
      ++bundles_[index].size;
    }
    for (std::size_t i = 0; i < bundles_size_; ++i) {
      auto &group = bundles_[i];
      group.token = PyTuple_New(static_cast<Py_ssize_t>(group.size));
      if (!group.token || PyErr_Occurred())
        return false;
      Py_ssize_t index = 0;
      for (std::size_t j = 0; j < edges_size_; ++j)
        if (edges_[j].nurse == group.nurse)
          PyTuple_SET_ITEM(group.token, index++, Py_NewRef(edges_[j].patient));
    }
    for (std::size_t i = 0; i < bundles_size_; ++i) {
      auto &group = bundles_[i];
      if (!attach_lifetime(group.nurse, group.token))
        return false;
      group.attached = true;
    }
    return committed_ = true;
  }
};

template <class... Extras>
bool commit_constructor_keep_alive(PyObject *const *slots,
                                   std::size_t count) noexcept {
  constexpr auto policies =
      (std::size_t{0} + ... + is_keep_alive_v<std::remove_cvref_t<Extras>>);
  if constexpr (policies <= 1) {
    return apply_keep_alive<Extras...>(nullptr, slots, count);
  } else {
    constructor_lifetime_transaction<policies> transaction;
    (transaction.template stage<Extras>(slots), ...);
    return transaction.commit();
  }
}

} // namespace detail
} // namespace pyb26
