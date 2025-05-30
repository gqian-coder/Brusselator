cmake_minimum_required(VERSION 3.19)

project(brusselator)
find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED PATHS /home/qian/Software/ADIOS2/install-adios/)
find_package(mgard REQUIRED)

find_package(CUDAToolkit REQUIRED)
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")

add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")

add_executable(brusselator_mpi brusselator_mpi.cpp io.tpp dualSystemEq.tpp funInit.tpp)
target_link_libraries(brusselator_mpi adios2::adios2 MPI::MPI_C mgard::mgard)
