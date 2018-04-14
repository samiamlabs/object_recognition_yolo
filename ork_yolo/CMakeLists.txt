cmake_minimum_required(VERSION 2.8.12)
project(object_recognition_yolo)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# Define path of darknet folder here.
find_path(DARKNET_PATH
  NAMES "README.md"
  HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../darknet/")
message("Darknet path dir = ${DARKNET_PATH}")
add_definitions(-DDARKNET_FILE_PATH="${DARKNET_PATH}")

# Find rquired packeges
find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED object_recognition_core object_recognition_renderer)

catkin_package(CATKIN_DEPENDS ecto object_recognition_core object_recognition_renderer
               DEPENDS OpenCV
)

#install targets for all things python
catkin_python_setup()

include_directories(
  include
  ${DARKNET_PATH}/src
  ${DARKNET_PATH}/include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

include_directories(
  SYSTEM
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib

  ${DARKNET_PATH}/src/activation_layer.c
  ${DARKNET_PATH}/src/activations.c
  ${DARKNET_PATH}/src/avgpool_layer.c
  ${DARKNET_PATH}/src/batchnorm_layer.c
  ${DARKNET_PATH}/src/blas.c
  ${DARKNET_PATH}/src/box.c
  ${DARKNET_PATH}/src/col2im.c
  ${DARKNET_PATH}/src/connected_layer.c
  ${DARKNET_PATH}/src/convolutional_layer.c
  ${DARKNET_PATH}/src/cost_layer.c
  ${DARKNET_PATH}/src/crnn_layer.c
  ${DARKNET_PATH}/src/crop_layer.c
  ${DARKNET_PATH}/src/cuda.c
  ${DARKNET_PATH}/src/data.c
  ${DARKNET_PATH}/src/deconvolutional_layer.c
  ${DARKNET_PATH}/src/demo.c
  ${DARKNET_PATH}/src/detection_layer.c
  ${DARKNET_PATH}/src/dropout_layer.c
  ${DARKNET_PATH}/src/gemm.c
  ${DARKNET_PATH}/src/gru_layer.c
  ${DARKNET_PATH}/src/im2col.c
  ${DARKNET_PATH}/src/image.c
  ${DARKNET_PATH}/src/l2norm_layer.c
  ${DARKNET_PATH}/src/layer.c
  ${DARKNET_PATH}/src/list.c
  ${DARKNET_PATH}/src/local_layer.c
  ${DARKNET_PATH}/src/logistic_layer.c
  ${DARKNET_PATH}/src/lstm_layer.c
  ${DARKNET_PATH}/src/matrix.c
  ${DARKNET_PATH}/src/maxpool_layer.c
  ${DARKNET_PATH}/src/network.c
  ${DARKNET_PATH}/src/normalization_layer.c
  ${DARKNET_PATH}/src/option_list.c
  ${DARKNET_PATH}/src/parser.c
  ${DARKNET_PATH}/src/region_layer.c
  ${DARKNET_PATH}/src/reorg_layer.c
  ${DARKNET_PATH}/src/rnn_layer.c
  ${DARKNET_PATH}/src/route_layer.c
  ${DARKNET_PATH}/src/shortcut_layer.c
  ${DARKNET_PATH}/src/softmax_layer.c
  ${DARKNET_PATH}/src/tree.c
  ${DARKNET_PATH}/src/upsample_layer.c
  ${DARKNET_PATH}/src/utils.c
  ${DARKNET_PATH}/src/yolo_layer.c

)

target_link_libraries(${PROJECT_NAME}_lib
  m
  pthread
  stdc++
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

# Compile the module
add_subdirectory(python)
add_subdirectory(src)

# add the tests
if(CATKIN_ENABLE_TESTING)
  add_subdirectory(test)
endif()

# build the docs
# ecto_sphinx(${CMAKE_CURRENT_SOURCE_DIR}/doc/source doc)

# install the conf files
install(DIRECTORY conf
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
