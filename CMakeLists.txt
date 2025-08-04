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