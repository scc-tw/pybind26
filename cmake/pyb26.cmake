include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/pyb26Toolchain.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/pyb26Runtime.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/pyb26Module.cmake")

function(_pyb26_set_resources include_dir source_dir)
  get_target_property(target pyb26::headers ALIASED_TARGET)
  if(NOT target)
    set(target pyb26::headers)
  endif()
  set_target_properties(${target} PROPERTIES
    _PYB26_INCLUDE_DIR "${include_dir}"
    _PYB26_RUNTIME_SOURCES "${source_dir}/runtime.cpp;${source_dir}/runtime_owner.cpp;${source_dir}/runtime_override.cpp"
    _PYB26_NUMPY_SOURCE "${source_dir}/numpy.cpp"
  )
endfunction()
