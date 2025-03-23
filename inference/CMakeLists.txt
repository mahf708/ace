cmake_minimum_required(VERSION 3.10)
project(inference_app)

set(CMAKE_PREFIX_PATH "/global/cfs/cdirs/e3sm/software/libtorch")

find_package(Torch REQUIRED)

add_library(inference inference.cpp)

target_link_libraries(inference ${TORCH_LIBRARIES})
target_include_directories(inference PUBLIC ${TORCH_INCLUDE_DIRS})

set_property(TARGET inference PROPERTY CXX_STANDARD 17)
