# ggml-rocket: RK3588 NPU backend via mainline "rocket" DRM-accel driver.
# No vendor library — raw DRM ioctls against /dev/accel/accel0.

ggml_add_backend_library(ggml-rocket
    ggml-rocket.cpp
    librocket.c
    rkt_npu_matmul.c
    rkt_operands.c
    rkt_matmul.c
    rkt_matmul_cna.c
    rkt_matmul_coredpu.c
    rkt_gemm.c
)

target_include_directories(ggml-rocket
    PRIVATE
        /usr/include/drm
        /usr/include/libdrm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(Threads REQUIRED)
target_link_libraries(ggml-rocket PRIVATE Threads::Threads)
