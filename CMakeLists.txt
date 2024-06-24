cmake_minimum_required(VERSION 3.14)
project(parallel_tree_tensor_network)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# Fetch Eigen
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen)

# Fetch xtl
FetchContent_Declare(
        xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
        GIT_TAG 0.7.7
)
FetchContent_MakeAvailable(xtl)

# Fetch Xtensor
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

# Add executable
add_executable(parallel_tree_tensor_network src/main.cpp)

# Link libraries (Eigen, xtl, and xtensor are header-only, so no linking necessary)
target_link_libraries(parallel_tree_tensor_network)
