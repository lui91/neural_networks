cmake_minimum_required (VERSION 3.8)
project(neural_networks)

# find_package(Torch REQUIRED PATHS /home/luis/libtorch)
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable (neural_networks main.cpp Perceptron.h Perceptron.cpp)
# target_link_libraries(neural_networks "${TORCH_LIBRARIES}")
target_link_libraries(neural_networks ${ARMADILLO_LIBRARIES})
set_property(TARGET neural_networks PROPERTY CXX_STANDARD 11)