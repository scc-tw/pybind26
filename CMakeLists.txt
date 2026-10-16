cmake_minimum_required(VERSION 4.4)

project(pybind26 VERSION 0.1.0 DESCRIPTION "C++26 bindings for CPython" LANGUAGES CXX)

include(GNUInstallDirs)

add_library(pyb26_headers INTERFACE)
add_library(pyb26::headers ALIAS pyb26_headers)
set_target_properties(pyb26_headers PROPERTIES EXPORT_NAME headers)
target_compile_features(pyb26_headers INTERFACE cxx_std_26)
include(cmake/pyb26Headers.cmake)

include(cmake/pyb26.cmake)
_pyb26_set_resources(
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

option(PYB26_INSTALL "Generate pybind26 install rules" ${PROJECT_IS_TOP_LEVEL})
if(PYB26_INSTALL)
  include(cmake/pyb26Install.cmake)
endif()
