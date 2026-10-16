include_guard(GLOBAL)

function(pyb26_check_toolchain)
  include(CheckCXXSourceCompiles)
  set(CMAKE_CXX_STANDARD 26)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  check_cxx_source_compiles([=[
    #include <concepts>
    #include <expected>
    #if __cplusplus <= 202302L
    #error C++26 required
    #endif
    int main() {
      std::expected<int, int> result(1);
      return *result - 1;
    }
  ]=] PYB26_HAS_CXX26)
  if(NOT PYB26_HAS_CXX26)
    message(FATAL_ERROR "pyb26 requires a C++26 compiler and standard library with concepts and std::expected")
  endif()
endfunction()

function(_pyb26_configure_compile target)
  if(MSVC)
    message(FATAL_ERROR "pyb26 requires a supported C++26 GCC or Clang toolchain")
  endif()
  pyb26_check_toolchain()
  target_compile_features(${target} PRIVATE cxx_std_26)
  set_target_properties(${target} PROPERTIES
    CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  target_compile_options(${target} PRIVATE "$<$<CONFIG:Release>:-O3>")
  if(NOT "REFERENCE_EXCEPTIONS" IN_LIST ARGN)
    target_compile_options(${target} PRIVATE -fno-exceptions)
    if(CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
      include(CheckCXXCompilerFlag)
      check_cxx_compiler_flag("-mtls-dialect=gnu2" PYB26_HAS_GNU2_TLS)
      if(NOT PYB26_HAS_GNU2_TLS)
        message(FATAL_ERROR "GNU ELF x86 builds require -mtls-dialect=gnu2")
      endif()
      target_compile_options(${target} PRIVATE -mtls-dialect=gnu2)
    endif()
  endif()
endfunction()

