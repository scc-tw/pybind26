#include "pyb26/detail/common.hpp"

#pragma push_macro("Py_BUILD_CORE")
#undef Py_BUILD_CORE
#define Py_BUILD_CORE
#include <internal/pycore_interpframe.h>
#pragma pop_macro("Py_BUILD_CORE")

namespace pyb26::detail {

bool current_frame_matches_function(PyThreadState *state,
                                    PyObject *function) noexcept {
  auto *frame = _PyThreadState_GetFrame(state);
  if (!frame)
    return false;
  PyObject *code = reinterpret_cast<PyObject *>(_PyFrame_GetCode(frame));
  return code == PyFunction_GET_CODE(function);
}

} // namespace pyb26::detail
