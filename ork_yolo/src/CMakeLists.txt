# Compile the ecto

set(SRC_FILES yolo_train.cpp
              module.cpp
              save_training_data.cpp
              image_mux.cpp
              yolo_detect.cpp
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

include_directories(${INCLUDE_DIRS})
ectomodule(ecto_yolo DESTINATION ${PROJECT_NAME}/ecto_cells
                        INSTALL
                        ${SRC_FILES}
)

link_ecto(ecto_yolo ${LIBRARIES})
