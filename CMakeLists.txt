cmake_minimum_required(VERSION 2.8)
project( gms_match_demo)

# OpenCV 
find_package( OpenCV 2.4.9 REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")

# 
include_directories( 
 include
${OpenCV_INCLUDE_DIRS} 
)


add_executable( gms_match_demo ./src/demo.cpp )
target_link_libraries( gms_match_demo ${OpenCV_LIBS} )
