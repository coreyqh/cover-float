project(softfloat LANGUAGES C)

include_directories(spike/softfloat)
file(GLOB SOFTFLOAT_SRC spike/softfloat/*.c)

add_library(softfloat STATIC ${SOFTFLOAT_SRC})
set_target_properties(softfloat PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(softfloat PRIVATE -Wall -Wextra -Werror -Wno-sign-compare -Wno-implicit-fallthrough)
