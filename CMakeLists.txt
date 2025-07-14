cmake_minimum_required(VERSION 3.19)

project(brusselator)
find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED PATHS /home/qg7/indir/)
find_package(mgard REQUIRED)

find_package(zstd REQUIRED /home/qg7/Software/MGARD/install-cuda-ampere/lib/cmake/zstd/)
set(ZSTD_INCLUDES "/home/qg7/Software/MGARD/install-cuda-ampere/include/")
find_library(ZSTD_LIB zstd HINTS "/home/qg7/Software/MGARD/install-cuda-ampere/lib/")

find_package(CUDAToolkit REQUIRED)
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")

add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")

find_package(SZ3 REQUIRED PATHS /usr/local/lib/cmake/SZ3/) 
set(SZ3_INCLUDES "/usr/local/include/SZ3/")
find_library(SZ3_LIB SZ3c REQUIRED HINTS "/usr/local/lib/")

add_executable(brusselator_mpi brusselator_mpi.cpp io.tpp dualSystemEq.tpp funInit.tpp)
target_include_directories(brusselator_mpi PRIVATE ${SZ3_INCLUDES} ${ZSTD_INCLUDES})
target_link_libraries(brusselator_mpi adios2::adios2 MPI::MPI_C mgard::mgard ${ZSTD_LIB} ${SZ3_LIB})

add_executable(errCalc errCalc.cpp)
target_link_libraries(errCalc adios2::adios2 MPI::MPI_C)


add_executable(brusselator_mpi_extended brusselator_mpi_extended.cpp io.tpp dualSystemEq.tpp funInit.tpp)
target_include_directories(brusselator_mpi_extended PRIVATE ${SZ3_INCLUDES} ${ZSTD_INCLUDES})
target_link_libraries(brusselator_mpi_extended adios2::adios2 MPI::MPI_C mgard::mgard ${ZSTD_LIB} ${SZ3_LIB})
