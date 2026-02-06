cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall")

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


pybind11_add_module(_reference MODULE src/c/main.cpp)
target_link_libraries(_reference PRIVATE coverfloat)

install(TARGETS _reference DESTINATION ${SKBUILD_PROJECT_NAME})
