# Compile the ecto

find_package(Eigen3 REQUIRED)

include_directories(${catkin_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIRS}
)

set(SRC_FILES yolo_train.cpp
              module.cpp
              save_training_data.cpp
              image_mux.cpp
              yolo_detect.cpp
              observation_renderer.cpp
)

# ModelFiller.cpp
# db_yolo.cpp


set(INCLUDE_DIRS ${catkin_INCLUDE_DIRS}
                 ${OpenCV_INCLUDE_DIRS}
                 ${DARKNET_PATH}/src
                 ${DARKNET_PATH}/include
)

set(LIBRARIES ${catkin_LIBRARIES}
              ${OpenCV_LIBRARIES}
              ${PROJECT_NAME}_lib
)


ectomodule(ecto_yolo DESTINATION ${PROJECT_NAME}/ecto_cells
                        INSTALL
                        ${SRC_FILES}
)

link_ecto(ecto_yolo ${LIBRARIES})
