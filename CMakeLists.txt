cmake_minimum_required(VERSION 3.28)
project(VulkanTutorial LANGUAGES CXX)

# C++20 modules configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_BUILD_TYPE Debug)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# Main executable
add_executable(Test main.cpp)

# Link libraries
target_link_libraries(Test 
    PRIVATE 
    Vulkan::Vulkan
    glfw
    glm::glm
)

# Include directories
target_include_directories(Test 
    PRIVATE 
    ${Vulkan_INCLUDE_DIRS}
)

# Compile definitions for Vulkan HPP
target_compile_definitions(Test 
    PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)

# Ensure proper C++20 support
set_target_properties(Test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# find slangc executable
find_program(SLANGC_EXECUTABLE
    NAMES slangc slangc.exe
    PATHS 
        ${SLANG_ROOT}/bin
        $ENV{SLANG_ROOT}/bin
    DOC "Slang compiler executable"
)

if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "Could not find slangc executable")
endif()

function (add_slang_shader_target TARGET)
  cmake_parse_arguments ("SHADER" "" "" "SOURCES" ${ARGN})
  set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
  set (ENTRY_POINTS -entry vertMain -entry fragMain)
  add_custom_command (
          OUTPUT ${SHADERS_DIR}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )
  add_custom_command (
          OUTPUT  ${SHADERS_DIR}/slang.spv
          COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
          WORKING_DIRECTORY ${SHADERS_DIR}
          DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
          COMMENT "Compiling Slang Shaders"
          VERBATIM
  )
  add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

add_slang_shader_target(shaders SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang)
add_dependencies(Test shaders)