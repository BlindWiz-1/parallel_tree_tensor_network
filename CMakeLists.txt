cmake_minimum_required(VERSION 3.14)
project(parallel_tree_tensor_network)

set(CMAKE_CXX_STANDARD 17)

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

# Include directories for Eigen, xtl, and xtensor
include_directories(${eigen_SOURCE_DIR})
include_directories(${xtl_SOURCE_DIR}/include)
include_directories(${xtensor_SOURCE_DIR}/include)

add_executable(parallel_tree_tensor_network src/main.cpp
        src/Circuits/QuantumGate/QuantumGate.cpp
        src/Circuits/CircuitGate/CircuitGate.cpp
        src/Circuits/Circuit/Circuit.cpp
        src/PseudoTNode/PseudoTNode.cpp
        src/TNode/TNode.cpp
        src/TTN/TTN.cpp
        src/Structure/SNode/SNode.cpp
        src/Structure/QPair/QPair.cpp
        src/Circuits/QuantumGate/SingleStateToTrees/SingleStateToTrees.cpp
        src/Operations/TTNContract/TTNContract.cpp
)

target_link_libraries(parallel_tree_tensor_network)
