cmake_minimum_required (VERSION 3.8)
project(neural_networks)

find_package(Torch REQUIRED PATHS /home/luis/libtorch)

add_executable (neural_networks main.cpp Perceptron.h Perceptron.cpp)
target_link_libraries(neural_networks "${TORCH_LIBRARIES}")
set_property(TARGET neural_networks PROPERTY CXX_STANDARD 11)