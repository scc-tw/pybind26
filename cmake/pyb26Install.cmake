include(CMakePackageConfigHelpers)

set(pyb26_install_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/pyb26")
set(pyb26_install_sourcedir "${CMAKE_INSTALL_DATADIR}/pyb26/src")

install(TARGETS pyb26_headers EXPORT pyb26Targets
  FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(FILES include/pyb26/detail/vendor/dlpack/LICENSE
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/pyb26/detail/vendor/dlpack"
)
install(FILES src/runtime.cpp src/runtime_owner.cpp src/runtime_override.cpp src/numpy.cpp
  DESTINATION "${pyb26_install_sourcedir}"
)
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/pyb26")
install(EXPORT pyb26Targets NAMESPACE pyb26:: DESTINATION "${pyb26_install_cmakedir}")

configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/pyb26Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/pyb26Config.cmake"
  INSTALL_DESTINATION "${pyb26_install_cmakedir}"
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR pyb26_install_sourcedir
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/pyb26ConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY ExactVersion
  ARCH_INDEPENDENT
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/pyb26Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/pyb26ConfigVersion.cmake"
  "${CMAKE_CURRENT_LIST_DIR}/pyb26.cmake"
  "${CMAKE_CURRENT_LIST_DIR}/pyb26Module.cmake"
  "${CMAKE_CURRENT_LIST_DIR}/pyb26Runtime.cmake"
  "${CMAKE_CURRENT_LIST_DIR}/pyb26Toolchain.cmake"
  DESTINATION "${pyb26_install_cmakedir}"
)
