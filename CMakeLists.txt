cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX C)

set(RAW_COVER_FLOAT_FLAGS "${CMAKE_C_FLAGS} -Wextra -Wall -Wno-format -Werror")
separate_arguments(COVER_FLOAT_FLAGS NATIVE_COMMAND "${RAW_COVER_FLOAT_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall -Wno-format -Werror")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(src/submodules)

include_directories(src/c)
include_directories(src/submodules/spike/)

file(GLOB SOFTFLOAT_SRC submodules/spike/softfloat/*.c)

add_library(coverfloat STATIC src/c/coverfloat.c)
target_link_libraries(coverfloat PRIVATE softfloat)
set_target_properties(coverfloat PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(coverfloat PRIVATE ${COVER_FLOAT_FLAGS})


pybind11_add_module(_reference MODULE src/c/main.cpp)
target_link_libraries(_reference PRIVATE coverfloat)
target_compile_options(_reference PRIVATE ${COVER_FLOAT_FLAGS})

install(TARGETS _reference DESTINATION ${SKBUILD_PROJECT_NAME})
