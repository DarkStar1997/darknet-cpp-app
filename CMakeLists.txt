cmake_minimum_required(VERSION 3.5)

project(darknet-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -march=native")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DARKNET_PATH /home/rohan/Downloads/darknet)
set(DARKNET_LIB_NAME libdarknet)

function(add_code name)
    find_package(OpenCV REQUIRED)
    add_executable(${name} ${name}.cpp)
    target_compile_definitions(${name} PRIVATE GPU=1 CUDNN=1 OPENCV=1)
    target_include_directories(${name} PRIVATE ${DARKNET_PATH}/include)
    target_link_libraries(${name} ${OpenCV_LIBS} ${DARKNET_PATH}/${DARKNET_LIB_NAME}.so)
endfunction()

add_code(main)
