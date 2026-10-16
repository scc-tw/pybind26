#include "pyb26/detail/callback_owner.hpp"

namespace pyb26::detail {

py_callable_owner::~py_callable_owner() noexcept {
  delete errors.load(std::memory_order_relaxed);
}

void py_callable_owner::destroy(const owner_admission *binding,
                                PyThreadState *state) noexcept {
  if (binding)
    callable.reset_admitted(*binding, state);
  delete this;
}

std::shared_ptr<void> shared_python_owner(PyObject *object) noexcept {
  return std::make_shared<python_owner_ref>(object);
}

} // namespace pyb26::detail
