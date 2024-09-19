cmake_minimum_required(VERSION 3.23)
project(parallel_tree_tensor_network)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)

FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen)

FetchContent_Declare(
        xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
        GIT_TAG 0.7.7
)
FetchContent_MakeAvailable(xtl)

FetchContent_Declare(
        xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
        GIT_TAG 0.25.0
)
FetchContent_MakeAvailable(xtensor)

FetchContent_Declare(
        xtensor-blas
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
        GIT_TAG 0.21.0
)
FetchContent_MakeAvailable(xtensor-blas)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)

#find_package(OpenMP REQUIRED)
#if(OPENMP_FOUND)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

# Include directories for Eigen, xtl, and xtensor
include_directories(${eigen_SOURCE_DIR})
include_directories(${xtl_SOURCE_DIR}/include)
include_directories(${xtensor_SOURCE_DIR}/include)
include_directories(${xtensor-blas_SOURCE_DIR}/include)
include_directories(${googletest_SOURCE_DIR})

add_executable(parallel_tree_tensor_network src/main.cpp
        src/Circuits/QuantumGate/QuantumGate.cpp
        src/Circuits/CircuitGate/CircuitGate.cpp
        src/Circuits/Circuit/Circuit.cpp
        src/Circuits/SingleStateToTrees/SingleStateToTrees.cpp
        src/TTNCircuitSim/PseudoTNode/PseudoTNode.cpp
        src/TTNCircuitSim/TNode/TNode.cpp
        src/TTNCircuitSim/TTN/TTN.cpp
        src/Structure/SNode/SNode.cpp
        src/Structure/QPair/QPair.cpp
        src/Structure/FindTreeStructure/FindTreeStructure.cpp
        src/Operations/TTNContract/TTNContract.cpp
        src/Operations/TTNSampling/TTNSampling.cpp
        src/Operations/Walker/Walker.cpp
        src/Operations/Orthonormalization/Orthonormalization.cpp
        src/TTNCircuitSim/TreeStructure/TreeStructure.cpp
        src/Operations/GateOperations/GateOperations.cpp
        src/TTNCircuitSim/TestTTN/TestTTN.cpp
)

target_link_libraries(parallel_tree_tensor_network gtest gtest_main)
