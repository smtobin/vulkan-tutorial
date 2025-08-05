#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector validation_layers = {
    "VK_LAYER_KHRONOS_validation"
};

std::vector<const char*> device_extensions = {
    vk::KHRSwapchainExtensionName,
    vk::KHRSpirv14ExtensionName,
    vk::KHRSynchronization2ExtensionName,
    vk::KHRCreateRenderpass2ExtensionName
};

#ifdef NDEBUG
constexpr bool enable_validation_layers = false;
#else
constexpr bool enable_validation_layers = true;
#endif

class HelloTriangleApplication
{
public:
    static std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary); // start reading at the end of the file and in binary

        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file!");
        }

        std::vector<char> buffer(file.tellg()); // use the end of file position to determine the size of the file and allocate a buffer

        // go back to beginning of file and read all at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();

        return buffer;

        
    }

    void run()
    {
        _initWindow();
        _initVulkan();
        _mainLoop();
        _cleanup();
    }
private:
    void _initWindow()
    {
        glfwInit(); // initialize GLFW
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // tells GLFW to not create an OpenGL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);     // disable resizable windows

        _window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);    // (width, height, title, (optional) monitor, (optional) OpenGL-specific)
    }

    void _initVulkan()
    {
        _createInstance();
        _createSurface();
        _pickPhysicalDevice();
        _createLogicalDevice();
        _createSwapChain();
        _createImageViews();
        _createGraphicsPipeline();
    }

    void _createInstance()
    {
        constexpr vk::ApplicationInfo app_info{
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = vk::ApiVersion14
        };  

        // get the required layers
        std::vector<char const*> required_layers;
        if (enable_validation_layers)
        {
            required_layers.assign(validation_layers.begin(), validation_layers.end());
        }

        // check if the required layers are supported by the Vulkan implementation
        auto layer_properties = _context.enumerateInstanceLayerProperties();
        if (std::ranges::any_of(required_layers, [&layer_properties](auto const& required_layer){
            return std::ranges::none_of(layer_properties, [required_layer](auto const& layer_property){
                return strcmp(layer_property.layerName, required_layer) == 0;
            });
        }))
        {
            throw std::runtime_error("One or more required layers are note supported!");
        }

        // get the required instance extensions from GLFW
        uint32_t glfw_extension_count = 0;
        auto glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

        // check if the required GLFW extensions are supported by the Vulkan implementation
        auto extension_properties = _context.enumerateInstanceExtensionProperties();
        for (uint32_t i = 0; i < glfw_extension_count; i++)
        {
            if (std::ranges::none_of(
                extension_properties,
                [glfw_extension = glfw_extensions[i]](auto const& extension_property)
                { return strcmp(extension_property.extensionName, glfw_extension) == 0; }))
            {
                throw std::runtime_error("Required GLFW extensions not supported: " + std::string(glfw_extensions[i]));
            }
        }

        vk::InstanceCreateInfo create_info{
            .pApplicationInfo = &app_info,
            .enabledLayerCount = static_cast<uint32_t>(required_layers.size()),
            .ppEnabledLayerNames = required_layers.data(),
            .enabledExtensionCount = glfw_extension_count,
            .ppEnabledExtensionNames = glfw_extensions
        };

        _instance = vk::raii::Instance(_context, create_info);  // create the Vulkan instance
    }

    void _createSurface()
    {
        VkSurfaceKHR surface;
        if (glfwCreateWindowSurface(*_instance, _window, nullptr, &surface) != 0)   // glfw only deals with the Vulkan C API
        {
            throw std::runtime_error("Failed to create window surface!");
        }
        _surface = vk::raii::SurfaceKHR(_instance, surface);
    }

    void _pickPhysicalDevice()
    {

        auto devices = _instance.enumeratePhysicalDevices();    // list the graphics cards
        if (devices.empty())
        {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }

        // evaluate each GPU and check if they meet the requirements necessary
        bool device_found = false;
        for (const auto& device : devices)
        {
            auto queue_families = device.getQueueFamilyProperties();
            bool is_suitable = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

            const auto qfp_iter = std::ranges::find_if(queue_families,
                [](vk::QueueFamilyProperties const& qfp) {
                    return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
                });
            is_suitable = is_suitable && (qfp_iter != queue_families.end());

            auto extensions = device.enumerateDeviceExtensionProperties();
            bool found = true;
            for (auto const& extension : device_extensions)
            {
                auto extension_iter = std::ranges::find_if(extensions, [extension](auto const& ext) {
                    return strcmp(ext.extensionName, extension) == 0;
                });
                found = found && extension_iter != extensions.end();
            }
            is_suitable = is_suitable && found;
            
            if (is_suitable)
            {
                _physical_device = device;
                device_found = true;
                break;
            }
        }

        if (!device_found)
        {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
    }

    void _createLogicalDevice()
    {
        std::vector<vk::QueueFamilyProperties> queue_family_properties = _physical_device.getQueueFamilyProperties();
        
        // get the first index into queueFamilyProperties which supports both graphics and present
        uint32_t queue_index = ~0u;
        for (uint32_t qfpIndex = 0; qfpIndex < queue_family_properties.size(); qfpIndex++)
        {
            if ((queue_family_properties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
                _physical_device.getSurfaceSupportKHR(qfpIndex, *_surface))
            {
            // found a queue family that supports both graphics and present
            queue_index = qfpIndex;
            break;
            }
        }
        if (queue_index == ~0u)
        {
            throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
        }

        float queue_priority = 0.0f;
        vk::DeviceQueueCreateInfo device_queue_create_info { 
            .queueFamilyIndex = queue_index, 
            .queueCount = 1, 
            .pQueuePriorities = &queue_priority 
        };

        vk::PhysicalDeviceFeatures device_features;
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> feature_chain = {
            {},     // empty for now
            {.dynamicRendering = true},         // enable dynamic rendering from Vulkan 1.3
            {.extendedDynamicState = true}      // enable extended dynamic state from the extension
        };

        // create the logical device
        vk::DeviceCreateInfo device_create_info {
            .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &device_queue_create_info,
            .enabledExtensionCount = static_cast<uint32_t>(device_extensions.size()),
            .ppEnabledExtensionNames = device_extensions.data()
        };
        _device = vk::raii::Device(_physical_device, device_create_info);

        // get a handle to the graphics queue
        _queue = vk::raii::Queue(_device, queue_index, 0);
    }


    vk::SurfaceFormatKHR _chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& available_formats)
    {
        for (const auto& available_format : available_formats)
        {
            // prefer 8-bit sRGB surface format
            if (available_format.format == vk::Format::eB8G8R8A8Srgb && available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return available_format;
        }

        // fall back to whatever else
        return available_formats[0];
    }

    vk::PresentModeKHR _chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& available_present_modes)
    {
        for (const auto& available_present_mode : available_present_modes)
        {
            // prefer "Mailbox" present mode
            // the application is not blocked when the image queue is full, but the images that are already queued are replaced with newer ones
            // this results in fewer latency issues than standard vertical sync while avoiding tearing
            // commonly known as "triple buffering"
            if (available_present_mode == vk::PresentModeKHR::eMailbox)
                return available_present_mode;
        }

        // FIFO queue mode (vertical sync) is guaranteed to exist, so use it as the default
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D _chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
    {
        // swap extent = resolution of the swap chain images, usually the resolution of the window that we're drawing to
        // mesaured in pixels
        if (capabilities.currentExtent.width !=- std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }

        int width, height;
        glfwGetFramebufferSize(_window, &width, &height);   // get the frame buffer size to get the window size in pixels

        return {
            std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    void _createSwapChain()
    {
        auto surface_capabilities = _physical_device.getSurfaceCapabilitiesKHR(_surface);
        auto swap_chain_surface_format = _chooseSwapSurfaceFormat(_physical_device.getSurfaceFormatsKHR(_surface));
        _swap_chain_image_format = swap_chain_surface_format.format;
        _swap_chain_extent = _chooseSwapExtent(surface_capabilities);
        uint32_t image_count = surface_capabilities.minImageCount + 1; // recommended to request at least one more image than the minimum

        // make sure we do not exceed the max number of images (0 is a special value that means there is no maximum)
        if (surface_capabilities.maxImageCount > 0 && image_count > surface_capabilities.maxImageCount)
        {
            image_count = surface_capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR swap_chain_create_info {
            .flags = vk::SwapchainCreateFlagsKHR(),
            .surface = _surface,
            .minImageCount = image_count,
            .imageFormat = _swap_chain_image_format,
            .imageColorSpace = swap_chain_surface_format.colorSpace,
            .imageExtent = _swap_chain_extent,
            .imageArrayLayers = 1,  // always 1 unless developing a stereoscopioc 3D application
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment, // what kind of operations we'll use the images in the swap chain for
            .imageSharingMode = vk::SharingMode::eExclusive,    // an image is owned by one queue family at a time (best performance)
            .preTransform = surface_capabilities.currentTransform,  // can apply transforms (90 deg rotation, flip, etc.) - currentTransform = no transform
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,   // opaque = ignore the alpha channel
            .presentMode = _chooseSwapPresentMode(_physical_device.getSurfacePresentModesKHR(_surface)),
            .clipped = true,    // clip pixels if covered by another window
            .oldSwapchain = nullptr 
        };

        _swap_chain = vk::raii::SwapchainKHR(_device, swap_chain_create_info);
        _swap_chain_images = _swap_chain.getImages();
    }
    void _createImageViews()
    {
        _swap_chain_image_views.clear();

        vk::ImageViewCreateInfo image_view_create_info {
            .viewType = vk::ImageViewType::e2D,
            .format = _swap_chain_image_format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}   // describes the image's purpose and which part of the image should be accessed
        };

        // create an image view for each swap chain image
        for (auto image : _swap_chain_images)
        {
            image_view_create_info.image = image;
            _swap_chain_image_views.emplace_back(_device, image_view_create_info);
        }


    }

    [[nodiscard]] vk::raii::ShaderModule _createShaderModule(const std::vector<char>& code) const
    {
        vk::ShaderModuleCreateInfo create_info{
            .codeSize = code.size() * sizeof(char),
            // bytecode pointer is uint32_t, not char, so we cast the pointer with reinterpret_cast
            // alignment requirements are met implicitly because data is stored in std::vector where default allocator ensures the data satisfies worst case alignemnt requirements
            .pCode = reinterpret_cast<const uint32_t*>(code.data()) 
        };

        // shader module is just a thin wrapper around the shader bytecode that we've loaded from file
        vk::raii::ShaderModule shader_module{ _device, create_info };

        return shader_module;
    }

    void _createGraphicsPipeline()
    {
        auto shader_code = readFile("shaders/slang.spv");
        vk::raii::ShaderModule shader_module = _createShaderModule(shader_code);

        vk::PipelineShaderStageCreateInfo vert_shader_stage_info{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = shader_module,
            .pName = "vertMain" // the entrypoint
        };

        vk::PipelineShaderStageCreateInfo frag_shader_stage_info{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = shader_module,
            .pName = "fragMain" // the entrypoint
        };

        vk::PipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};
    }

    void _mainLoop()
    {
        // keep application running until error occurs or window is closed
        while (!glfwWindowShouldClose(_window))
        {
            glfwPollEvents();
        }
    }

    void _cleanup()
    {
        // device queues are implicitly cleaned up when the device is destroyed, so we don't need to do anything in cleanup

        glfwDestroyWindow(_window); // destroy the window

        glfwTerminate();
    }

private:
    GLFWwindow* _window;    // the GLFW window

    vk::raii::Context _context; // the Vulkan RAII context
    vk::raii::Instance _instance = nullptr; // the Vulkan instance - the connection between this application and the Vulkan library

    vk::raii::PhysicalDevice _physical_device = nullptr; // the graphics card

    vk::raii::Device _device = nullptr; // the logical device that interfaces with the graphics card

    vk::raii::Queue _queue = nullptr;    // the graphics and presentation queue

    vk::raii::SurfaceKHR _surface = nullptr;    // the surface to draw to

    vk::raii::SwapchainKHR _swap_chain = nullptr;
    std::vector<vk::Image> _swap_chain_images;

    vk::Format _swap_chain_image_format = vk::Format::eUndefined;
    vk::Extent2D _swap_chain_extent;
    std::vector<vk::raii::ImageView> _swap_chain_image_views;
};