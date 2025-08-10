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

std::vector<vk::DynamicState> dynamic_states = {
    vk::DynamicState::eViewport,
    vk::DynamicState::eScissor
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
        _createCommandPool();
        _createCommandBuffer();
        _createSyncObjects();
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
        _graphics_index = ~0u;
        for (uint32_t qfpIndex = 0; qfpIndex < queue_family_properties.size(); qfpIndex++)
        {
            if ((queue_family_properties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
                _physical_device.getSurfaceSupportKHR(qfpIndex, *_surface))
            {
            // found a queue family that supports both graphics and present
            _graphics_index = qfpIndex;
            break;
            }
        }
        if (_graphics_index == ~0u)
        {
            throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
        }

        float queue_priority = 0.0f;
        vk::DeviceQueueCreateInfo device_queue_create_info { 
            .queueFamilyIndex = _graphics_index, 
            .queueCount = 1, 
            .pQueuePriorities = &queue_priority 
        };

        vk::PhysicalDeviceFeatures device_features;
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
             vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> feature_chain = {
            {},     // empty for now
            {.shaderDrawParameters = true},
            {.synchronization2 = true, .dynamicRendering = true},         // enable dynamic rendering from Vulkan 1.3
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
        _queue = vk::raii::Queue(_device, _graphics_index, 0);
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
        auto shader_code = readFile("../shaders/slang.spv");
        vk::raii::ShaderModule shader_module = _createShaderModule(shader_code);

        // vertex shader info
        vk::PipelineShaderStageCreateInfo vert_shader_stage_info{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = shader_module,
            .pName = "vertMain" // the entrypoint
        };

        // fragment shader info
        vk::PipelineShaderStageCreateInfo frag_shader_stage_info{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = shader_module,
            .pName = "fragMain" // the entrypoint
        };

        vk::PipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};


        // dynamic parts of the pipeline - determined by the dynamic_states vector
        vk::PipelineDynamicStateCreateInfo dynamic_state { 
            .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data()
        };

        // describes the format of the vertex data that will be passed to the vertex shader
        vk::PipelineVertexInputStateCreateInfo vertex_input_info;   // nothing for now since we've hardcoded the vertices

        // describes what kind of geometry will be drawn from the vertices
        // if primitive restart is enabled, then we can break up lines and triangles in strips by using the special index 0xFFFF
        vk::PipelineInputAssemblyStateCreateInfo input_assembly{
            .topology = vk::PrimitiveTopology::eTriangleList
        };

        // viewport describes the region of the framebuffer tthe output will be rendered to
        // scissor rectangle defines in which region pixels will actually be stored (a filter rather than a transformation)
        // the rasterizer discards any pixels outside the scissored rectangle
        vk::PipelineViewportStateCreateInfo viewport_state {
            .viewportCount = 1,
            .scissorCount = 1
        };

        // rasterizer takes geometry shaped by the vertices from the vertex shader and turns it into fragments to be colored by the fragment shader
        vk::PipelineRasterizationStateCreateInfo rasterizer {
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,   // if true, then fragments beyond the near and far planes are clamped to them (instead of being discarded)
            .polygonMode = vk::PolygonMode::eFill,  // fill the area of the polygon with fragments
            .cullMode = vk::CullModeFlagBits::eBack,    // the type of face culling to use
            .frontFace = vk::FrontFace::eClockwise,
            .depthBiasEnable = vk::False,
            .depthBiasSlopeFactor = 1.0f,
            .lineWidth = 1.0f
        };

        // configure multisampling (one of the ways to perform antialiasing)
        // combines the fragment shader results of multiple polygons that rasterize to the same pixel
        vk::PipelineMultisampleStateCreateInfo multisampling {
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False
        };

        // // after a fragment shader has returned a color, it needs to be combined with the color that is already in the framebuffer
        // // this transformation is known as color blending - we can either mix the two values or combine the two values with a bitwise operation
        // vk::PipelineColorBlendAttachmentState color_blend_attachment;
        // // alpha blending - new color blended with the old color based on its opacity
        // color_blend_attachment.blendEnable = vk::True;
        // color_blend_attachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        // color_blend_attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        // color_blend_attachment.colorBlendOp = vk::BlendOp::eAdd;
        // color_blend_attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        // color_blend_attachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
        // color_blend_attachment.alphaBlendOp = vk::BlendOp::eAdd;

        vk::PipelineColorBlendAttachmentState color_blend_attachment{ .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo color_blending {
            .logicOpEnable = vk::False,  // not doing bitwise combination
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment
        };

        vk::PipelineLayoutCreateInfo pipeline_layout_info {
            .setLayoutCount = 0,
            .pushConstantRangeCount = 0
        };

        _pipeline_layout = vk::raii::PipelineLayout(_device, pipeline_layout_info); // also can define "push constants"
        

        // specify the formats of the attachments that will be used during rendering (to use dynamic rendering)
        vk::PipelineRenderingCreateInfo pipeline_rendering_create_info {
            .colorAttachmentCount = 1,  // we will use one color attachment with the format of our swap chain images
            .pColorAttachmentFormats = &_swap_chain_image_format
        };

        vk::GraphicsPipelineCreateInfo pipeline_info {
            .pNext = &pipeline_rendering_create_info,
            .stageCount = 2,
            .pStages = shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_state,
            .layout = _pipeline_layout,
            .renderPass = nullptr   // this is set to nullptr since we are using dynamic rendering instead of a traditional render pass
        };

        _graphics_pipeline = vk::raii::Pipeline(_device, nullptr, pipeline_info);

    }

    void _createCommandPool()
    {
        vk::CommandPoolCreateInfo pool_info {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,    // we are recording a command buffer every frame so we want to be able to reset and rerecord over it
            .queueFamilyIndex = _graphics_index // command ubffers are executed by submitting them on one of the device queues
        };

        _command_pool = vk::raii::CommandPool(_device, pool_info);
    }

    void _createCommandBuffer()
    {
        vk::CommandBufferAllocateInfo alloc_info {
            .commandPool = _command_pool,
            .level = vk::CommandBufferLevel::ePrimary,  // primary = submitted to a queue for execution, but cannot be called from other command buffers
            // secondary = cannot be submitted directly, but can be called from primary command buffers
            .commandBufferCount = 1
        };

        _command_buffer = std::move(vk::raii::CommandBuffers(_device, alloc_info).front());
    }

    void _recordCommandBuffer(uint32_t image_index)
    {
        _command_buffer.begin( {} );    // this will reset _command_buffer if it has already been recorded - not possible to append commands

        // before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
        _transitionImageLayout(
            image_index,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {}, // srcAccessMask (no need to wait for previous operations)
            vk::AccessFlagBits2::eColorAttachmentWrite, // dstAccessMask
            vk::PipelineStageFlagBits2::eTopOfPipe, // srcStage
            vk::PipelineStageFlagBits2::eColorAttachmentOutput  // dstStage
        );

        // set up the color attachment
        vk::ClearValue clear_color = vk::ClearColorValue(0.f, 0.f, 0.f, 1.0f);
        vk::RenderingAttachmentInfo attachment_info = {
            .imageView = _swap_chain_image_views[image_index],  // which image view to render to
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,    // the layout the image will be in during rendering
            .loadOp = vk::AttachmentLoadOp::eClear, // what to do with the image before rendering
            .storeOp = vk::AttachmentStoreOp::eStore,   // what to do with the image after rendering
            .clearValue = clear_color
        };

        // set up rendering info
        vk::RenderingInfo rendering_info = {
            .renderArea = { .offset = {0,0}, .extent = _swap_chain_extent}, // defines the size of the render area
            .layerCount = 1,    // number of layers to render to (1 for a non-layered image)
            .colorAttachmentCount = 1,  // specify color attachments to render to
            .pColorAttachments = &attachment_info
        };

        // now we can begin rendering
        _command_buffer.beginRendering(rendering_info);

        // bind the graphics pipeline
        _command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _graphics_pipeline);

        // since we specified viewport and scissor state for the pipeline as dynamic, we need to set them here before issuing the draw command
        _command_buffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(_swap_chain_extent.width), static_cast<float>(_swap_chain_extent.height), 0.0f, 1.0f));
        _command_buffer.setScissor(0, vk::Rect2D(vk::Offset2D(0,0), _swap_chain_extent));

        // issue the draw command for the triangle
        _command_buffer.draw(
            3,  // vertexCount - 3 vertices to draw
            1,  // instanceCount - use 1 if not using
            0,  // firstVertex - used as an offset into the vertex buffer, defines the lowest value of SV_VertexId
            0   // firstInstance - used as an offset for isntanced rendering, defines the lowest value of SV_InstanceId
        );

        _command_buffer.endRendering();

        // transition the image layout back to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR so it can be presented to the screen
        _transitionImageLayout(
            image_index,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe
        );

        _command_buffer.end();
    }

    void _transitionImageLayout(
        uint32_t image_index,
        vk::ImageLayout old_layout,
        vk::ImageLayout new_layout,
        vk::AccessFlags2 src_access_mask,
        vk::AccessFlags2 dst_access_mask,
        vk::PipelineStageFlags2 src_stage_mask,
        vk::PipelineStageFlags2 dst_stage_mask
    ) {
        // transition the image layout from old to new
        // used for transitioning an image layout to one that is suitable for rendering
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = src_stage_mask,
            .srcAccessMask = src_access_mask,
            .dstStageMask = dst_stage_mask,
            .dstAccessMask = dst_access_mask,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = _swap_chain_images[image_index],
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        vk::DependencyInfo dependency_info = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier
        };

        _command_buffer.pipelineBarrier2(dependency_info);
    }

    void _createSyncObjects()
    {
        _present_complete_semaphore = vk::raii::Semaphore(_device, vk::SemaphoreCreateInfo());
        _render_finished_semaphore = vk::raii::Semaphore(_device, vk::SemaphoreCreateInfo());
        _draw_fence = vk::raii::Fence(_device, {.flags = vk::FenceCreateFlagBits::eSignaled});
    }

    void _mainLoop()
    {
        // keep application running until error occurs or window is closed
        while (!glfwWindowShouldClose(_window))
        {
            glfwPollEvents();
            _drawFrame();
        }

        _device.waitIdle();
    }

    void _drawFrame()
    {
        _queue.waitIdle();  // essentially waits for all fences to signal
        // vkWaitForFences();

        auto [result, image_index] = _swap_chain.acquireNextImage(
            UINT64_MAX, // no timeout
            *_present_complete_semaphore,   // signal the present complete semphore that the next image has been acquired
            nullptr // not dependent on other sync objects since we already waited for fences
        );

        // record the command buffer to the swap chain image
        _recordCommandBuffer(image_index);

        // make sure the fence is reset
        _device.resetFences(*_draw_fence);
        
        vk::PipelineStageFlags wait_destination_stage_mask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        vk::SubmitInfo submit_info {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*_present_complete_semaphore,  // which semaphore(s) to wait on before execution begins
            .pWaitDstStageMask = &wait_destination_stage_mask,// which stage(s) of the pipieline to wait - want to wait for writing colors to the image until it's available
            .commandBufferCount = 1,
            .pCommandBuffers = &*_command_buffer,  // which command buffer(s) to submit for execution
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*_render_finished_semaphore    // which semaphores to signal once the command buffer(s) have finished execution
        };
        _queue.submit(submit_info, *_draw_fence);

        // the CPU needs to wait while the GPU finishes rendering the frame we just submitted
        while ( vk::Result::eTimeout == _device.waitForFences( *_draw_fence, vk::True, UINT64_MAX ) )
            ;
        
        vk::PresentInfoKHR present_info_KHR {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*_render_finished_semaphore,   // which semaphore(s) to wait on before presentation can happen
            .swapchainCount = 1,
            .pSwapchains = &*_swap_chain,  // the swap chain(s) to present images to
            .pImageIndices = &image_index // the index of the image for each swap chain
        };
        result = _queue.presentKHR(present_info_KHR);
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
    uint32_t _graphics_index;   // the index of the queue family for the device queue of our graphics card

    vk::raii::Device _device = nullptr; // the logical device that interfaces with the graphics card

    vk::raii::Queue _queue = nullptr;    // the graphics and presentation queue

    vk::raii::SurfaceKHR _surface = nullptr;    // the surface to draw to

    vk::raii::SwapchainKHR _swap_chain = nullptr;
    std::vector<vk::Image> _swap_chain_images;

    vk::Format _swap_chain_image_format = vk::Format::eUndefined;
    vk::Extent2D _swap_chain_extent;
    std::vector<vk::raii::ImageView> _swap_chain_image_views;

    vk::raii::PipelineLayout _pipeline_layout = nullptr;   // specifies "uniform" values which can pass dynamic values to shaders

    vk::raii::Pipeline _graphics_pipeline = nullptr;    // the graphics pipeline

    vk::raii::CommandPool _command_pool = nullptr;  // manages the memory used to store the buffers
    vk::raii::CommandBuffer _command_buffer = nullptr; // the command buffer

    vk::raii::Semaphore _present_complete_semaphore = nullptr;  // indicates that an image has been acquired form the swapchain and is ready for rendering
    vk::raii::Semaphore _render_finished_semaphore = nullptr;   // indicates that the rendering has finished and presentation can happen
    vk::raii::Fence _draw_fence = nullptr;  // to make usre only one frame is rendered at a time
};