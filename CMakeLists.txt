CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(depth2normal)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/opt/eigen/share/cmake/Modules)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(PCL 1.8 REQUIRED QUIET)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS} ${PCL_LIBRARY_DIRS})
LINK_LIBRARIES(${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

ADD_EXECUTABLE(depth2normal main.cpp)
