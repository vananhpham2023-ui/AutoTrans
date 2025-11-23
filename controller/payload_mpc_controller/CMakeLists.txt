cmake_minimum_required(VERSION 3.16)
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
project(payload_mpc_controller VERSION 1.0.0)

# Allow catkin / command line to control build type.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ARM NEON flags
if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations")
  message("enabling ARM neon optimizations")
endif()

# Flags for speed and diagnostics.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
# Use a more moderate optimization level to reduce compiler memory usage,
# especially when building LibTorch-heavy translation units.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  quadrotor_msgs
  uav_utils
  roslib

)
message(STATUS "Torch_DIR hint: ${Torch_DIR}")
find_package(Torch QUIET)
message(STATUS "Torch_FOUND: ${Torch_FOUND}")
if(Torch_FOUND AND CMAKE_VERSION VERSION_LESS 3.18)
  message(WARNING "LibTorch detected with CMake ${CMAKE_VERSION} (< 3.18). "
                  "This configuration is not officially supported; consider upgrading CMake "
                  "if you encounter build issues.")
endif()

catkin_package(
  CATKIN_DEPENDS
    roscpp
    std_msgs
    geometry_msgs
    quadrotor_msgs
    uav_utils
    roslib
)

#include dir
# include_directories(
#   include
#   include/payload_mpc_controller
# )

# add_library(mpc_solver
#   externals/qpoases/SRC/Bounds.cpp
#   externals/qpoases/SRC/Constraints.cpp
#   externals/qpoases/SRC/CyclingManager.cpp
#   externals/qpoases/SRC/Indexlist.cpp
#   externals/qpoases/SRC/MessageHandling.cpp
#   externals/qpoases/SRC/QProblem.cpp
#   externals/qpoases/SRC/QProblemB.cpp
#   externals/qpoases/SRC/SubjectTo.cpp
#   externals/qpoases/SRC/Utils.cpp
#   externals/qpoases/SRC/EXTRAS/SolutionAnalysis.cpp
#   model/quadrotor_payload_mpc/acado_qpoases_interface.cpp
#   model/quadrotor_payload_mpc/acado_integrator.c
#   model/quadrotor_payload_mpc/acado_solver.c
#   model/quadrotor_payload_mpc/acado_auxiliary_functions.c)

# target_include_directories(mpc_solver PUBLIC
#   model/quadrotor_payload_mpc/
#   externals/qpoases
#   externals/qpoases/INCLUDE
#   externals/qpoases/SRC
#   )

add_executable(mpc_controller_node 
  src/mpc_controller_node.cpp
  src/mpc_controller.cpp
  src/mpc_input.cpp
  src/mpc_fsm.cpp
  src/reference_generator.cpp
  src/mpc_wrapper.cpp
  # MPC solver
  externals/qpoases/SRC/Bounds.cpp
  externals/qpoases/SRC/Constraints.cpp
  externals/qpoases/SRC/CyclingManager.cpp
  externals/qpoases/SRC/Indexlist.cpp
  externals/qpoases/SRC/MessageHandling.cpp
  externals/qpoases/SRC/QProblem.cpp
  externals/qpoases/SRC/QProblemB.cpp
  externals/qpoases/SRC/SubjectTo.cpp
  externals/qpoases/SRC/Utils.cpp
  externals/qpoases/SRC/EXTRAS/SolutionAnalysis.cpp
  model/quadrotor_payload_mpc/acado_qpoases_interface.cpp
  model/quadrotor_payload_mpc/acado_integrator.c
  model/quadrotor_payload_mpc/acado_solver.c
  model/quadrotor_payload_mpc/acado_auxiliary_functions.c
  )
target_include_directories(mpc_controller_node PUBLIC
  include/payload_mpc_controller
  include 
  ${EIGEN3_INCLUDE_DIR}
  SYSTEM
  ${catkin_INCLUDE_DIRS} 
  # MPC solver
  model/quadrotor_payload_mpc/
  externals/qpoases
  externals/qpoases/INCLUDE
  externals/qpoases/SRC
  )
target_link_libraries(mpc_controller_node
  ${catkin_LIBRARIES}
)
add_executable(pinn_estimator_cli
  src/pinn_estimator_cli.cpp
)
target_include_directories(pinn_estimator_cli PRIVATE
  include/payload_mpc_controller
  include
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)
target_link_libraries(pinn_estimator_cli
  ${catkin_LIBRARIES}
)
install(TARGETS pinn_estimator_cli RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

if(Torch_FOUND)
  message(STATUS "LibTorch found, enabling PINN force estimator.")
  target_sources(mpc_controller_node PRIVATE
    src/pinn_force_estimator_libtorch.cpp
  )
  target_include_directories(mpc_controller_node PUBLIC
    ${TORCH_INCLUDE_DIRS}
  )
  target_link_libraries(mpc_controller_node
    ${TORCH_LIBRARIES}
  )
  target_compile_definitions(mpc_controller_node PUBLIC AUTOTRANS_WITH_TORCH)

  target_include_directories(pinn_estimator_cli PRIVATE
    ${TORCH_INCLUDE_DIRS}
  )
  target_link_libraries(pinn_estimator_cli
    ${TORCH_LIBRARIES}
  )
  target_compile_definitions(pinn_estimator_cli PRIVATE AUTOTRANS_WITH_TORCH)
else()
  message(WARNING "LibTorch not found, building without PINN force estimator (L-BFGS only).")
  target_sources(mpc_controller_node PRIVATE
    src/pinn_force_estimator_stub.cpp
  )
endif()

# Install pre-packaged model assets when available.
set(_pinn_model_candidates
  \"${CMAKE_CURRENT_SOURCE_DIR}/models/force_estimation_script3.pt\"
  \"${CMAKE_CURRENT_SOURCE_DIR}/../../force_estimation_script3.pt\"
)
foreach(candidate IN LISTS _pinn_model_candidates)
  if(NOT PINN_MODEL_SOURCE AND EXISTS ${candidate})
    set(PINN_MODEL_SOURCE ${candidate})
  endif()
endforeach()
if(PINN_MODEL_SOURCE)
  install(FILES ${PINN_MODEL_SOURCE} DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models)
endif()
set(PINN_NORMALIZER_SOURCE \"${CMAKE_CURRENT_SOURCE_DIR}/models/normalizer_stats.json\")
if(EXISTS ${PINN_NORMALIZER_SOURCE})
  install(FILES ${PINN_NORMALIZER_SOURCE} DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models)
endif()
if(EXISTS \"${CMAKE_CURRENT_SOURCE_DIR}/models/README.md\")
  install(FILES \"${CMAKE_CURRENT_SOURCE_DIR}/models/README.md\" DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models)
endif()

catkin_install_python(PROGRAMS
  scripts/force_data_recorder.py
  scripts/plot_force_comparison.py
  scripts/plot_state_tracking.py
  scripts/compare_nmpc_schemes.py
  scripts/pinn_data_recorder.py
  scripts/compute_normalizer_stats.py
  scripts/wait_for_completion.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(reference_generator_test
    test/reference_generator_test.cpp
    src/reference_generator.cpp
  )
  if(TARGET reference_generator_test)
    target_include_directories(reference_generator_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/include/payload_mpc_controller
      ${EIGEN3_INCLUDE_DIR}
    )
    target_link_libraries(reference_generator_test
      gtest_main
      gtest
      ${catkin_LIBRARIES}
    )
  endif()
endif()
