#include "HelloTriangleApplication.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdlib>

int main()
{
    HelloTriangleApplication app;

    try
    {
        app.run();
    }
    catch (const vk::SystemError& err)
    {
        std::cerr << "Vulkan error: " << err.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
    
}