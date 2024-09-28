cmake_minimum_required(VERSION 3.10)
project(FaceRecognition)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find Dlib package
find_package(dlib REQUIRED)

# Include OpenCV and Dlib directories
include_directories(${OpenCV_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS})

# Add executable
add_executable(FaceRecognition src/main.cpp)

# Link libraries
target_link_libraries(FaceRecognition ${OpenCV_LIBS} dlib::dlib)
