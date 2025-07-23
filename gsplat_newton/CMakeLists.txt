cmake_minimum_required(VERSION 3.24...3.30)

# Manually list all C++ and CUDA source files
set(GSN_SOURCES
    # C++ files from core/
    core/Newton.cpp
    core/camera_knn.cpp
    core/kernels.cpp
    core/local_newton_trainer.cpp
    core/newton_functions.cpp
    core/rasterizer_newton.cpp
    #core/rasterizer_newton_old.cpp

    # CUDA files from cuda/
    cuda/NewtonCUDA.cu
    cuda/assembly_kernels.cu
    cuda/loss_kernels.cu
    cuda/projection_kernels.cu
    cuda/rasterization_kernels.cu
    cuda/sh_kernels.cu
    # cuda/sh_kernels_old.cu
    cuda/update_kernels.cu
)

# Create a static library named gsplat_newton_backend
add_library(gsplat_newton_backend STATIC ${GSN_SOURCES})

# Set standard properties for the CUDA compiler
set_target_properties(gsplat_newton_backend PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Tell other targets where to find this library's public headers
target_include_directories(gsplat_newton_backend
    PUBLIC
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/gsplat
        ${TORCH_INCLUDE_DIRS}
    PRIVATE
        ${Python3_INCLUDE_DIRS}
)

# Link against required libraries
target_link_libraries(gsplat_newton_backend
    PUBLIC
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        glm::glm
)

# Set compile options for debug and release builds
target_compile_options(gsplat_newton_backend PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 --use_fast_math --expt-relaxed-constexpr -diag-suppress=20012,186>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
)