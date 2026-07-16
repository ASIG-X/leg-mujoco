cmake_minimum_required(VERSION 3.10)
project(go2_joystick_deploy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(yaml-cpp REQUIRED)

if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp)
else()
  set(YAML_CPP_TARGET ${YAML_CPP_LIBRARIES})
endif()

find_package(onnxruntime CONFIG QUIET)
if(onnxruntime_FOUND)
  if(TARGET onnxruntime::onnxruntime)
    set(ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
  else()
    set(ONNXRUNTIME_TARGET onnxruntime)
  endif()
else()
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
      $ENV{ONNXRUNTIME_ROOT}/include
      /usr/local/include
      /usr/include
    PATH_SUFFIXES onnxruntime
  )
  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
      $ENV{ONNXRUNTIME_ROOT}/lib
      /usr/local/lib
      /usr/lib
      /usr/lib/x86_64-linux-gnu
  )
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
      "ONNX Runtime was not found. Set ONNXRUNTIME_ROOT to a directory containing "
      "include/onnxruntime_cxx_api.h and lib/libonnxruntime.so."
    )
  endif()
  add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
  )
  set(ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
endif()

add_executable(locomotion_node locomotion_node.cpp)

ament_target_dependencies(locomotion_node
  rclcpp
  geometry_msgs
  std_msgs
)

target_link_libraries(locomotion_node
  Eigen3::Eigen
  fmt::fmt
  unitree_sdk2
  ${YAML_CPP_TARGET}
  ${ONNXRUNTIME_TARGET}
)

install(TARGETS locomotion_node DESTINATION lib/${PROJECT_NAME})
install(FILES config.yaml DESTINATION share/${PROJECT_NAME})

ament_package()
