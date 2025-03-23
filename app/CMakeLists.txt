cmake_minimum_required(VERSION 3.10)
project(new_app)

set(CMAKE_PREFIX_PATH "/global/cfs/cdirs/e3sm/software/libtorch")

find_package(Torch REQUIRED)

# Add the directory containing the inference library
add_subdirectory(/pscratch/sd/m/mahf708/ace/inference ${CMAKE_BINARY_DIR}/inference_build)

add_executable(new_app main.cpp)

# Link the inference library to your new application
target_link_libraries(new_app inference ${TORCH_LIBRARIES})
target_include_directories(new_app PUBLIC ${TORCH_INCLUDE_DIRS})

set_property(TARGET new_app PROPERTY CXX_STANDARD 17)
