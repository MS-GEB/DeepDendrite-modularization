# For now just do nothing in here

if(CMAKE_ISPC_COMPILER_FORCED)
  # The compiler configuration was forced by the user.
  # Assume the user has configured all compiler information.
  set(CMAKE_ISPC_COMPILER_WORKS TRUE)
  return()
endif()

set(CMAKE_ISPC_COMPILER_WORKS 1 CACHE INTERNAL "")


# Remove any cached result from an older CMake version.
# We now store this in CMakeISPCCompiler.cmake.
unset(CMAKE_ISPC_COMPILER_WORKS CACHE)

# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that the selected ispc compiler can actually compile
# and the most basic program. If not, a fatal error
# is set and cmake stops processing commands and will not generate
# any makefiles or projects.
if(NOT CMAKE_ISPC_COMPILER_WORKS)
  PrintTestCompilerStatus("ISPC" "")
  file(WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/kernel.ispc
    "export uniform int kernel(uniform int){return 0;}\n")

  execute_process(COMMAND ${CMAKE_ISPC_COMPILER} "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/kernel.ispc"
     OUTPUT_VARIABLE __CMAKE_ISPC_COMPILER_OUTPUT
     ERROR_VARIABLE __CMAKE_ISPC_COMPILER_ERROR
     RESULT_VARIABLE ISPC_RESULT)

  if(${ISPC_RESULT})
    set(CMAKE_ISPC_COMPILER_WORKS 0)
  else()
    set(CMAKE_ISPC_COMPILER_WORKS 1)
  endif()

  # Move result from cache to normal variable.
  unset(CMAKE_ISPC_COMPILER_WORKS CACHE)
  set(ISPC_TEST_WAS_RUN 1)
endif()

if(NOT CMAKE_ISPC_COMPILER_WORKS)
  PrintTestCompilerStatus("ISPC" " -- broken")
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
    "Determining if the ISPC compiler works failed with "
    "the following output:\n${__CMAKE_ISPC_COMPILER_OUTPUT}\n\n")
  string(REPLACE "\n" "\n  " _output "${__CMAKE_ISPC_COMPILER_OUTPUT} ${__CMAKE_ISPC_COMPILER_ERROR}")
  message(FATAL_ERROR "The ISPC compiler\n  \"${CMAKE_ISPC_COMPILER}\"\n"
    "is not able to compile a simple test program.\nIt fails "
    "with the following output:\n  ${_output}\n\n"
    "CMake will not be able to correctly generate this project.")
else()
  if(ISPC_TEST_WAS_RUN)
    PrintTestCompilerStatus("ISPC" " -- works")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Determining if the ISPC compiler works passed with "
      "the following output:\n${__CMAKE_ISPC_COMPILER_OUTPUT}\n\n")
  endif()
endif()


unset(__CMAKE_ISPC_COMPILER_OUTPUT)
unset(__CMAKE_ISPC_COMPILER_ERROR)
