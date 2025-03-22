cmake_minimum_required(VERSION 3.10)
project(inference_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/global/u1/m/mahf708/.micromamba/envs/fme/lib/python3.10/site-packages/torch/share/cmake/Torch")

find_package(Torch REQUIRED)

add_executable(inference inference/inference.cpp)

target_link_libraries(inference ${TORCH_LIBRARIES})
target_include_directories(inference PRIVATE ${TORCH_INCLUDE_DIRS})

set_property(TARGET inference PROPERTY CXX_STANDARD 17)
