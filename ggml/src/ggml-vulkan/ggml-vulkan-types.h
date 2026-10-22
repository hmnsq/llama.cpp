#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-vulkan.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <vulkan/vulkan_core.h>

#ਜੇ defined(GGML_VULKAN_ਲੜੀਵਾਰ_TESTS) || defined(GGML_VULKAN_CHECK_RESULTS)
#ਅਨ੍ਤਰ੍ਭਾਵ <chrono>
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#ਜੇ VK_ਸਿਰਲੇਖ_ਸੰਸਕਰਣ >= 301
namespace vk::detail { class DispatchLoaderDynamic; }
using vk::detail::DispatchLoaderDynamic;
#ਨਹੀਂਤਾਂ
namespace vk { class DispatchLoaderDynamic; }
using vk::DispatchLoaderDynamic;
#ਜੇ_ਅੰਤ

DispatchLoaderDynamic & ggml_vk_default_dispatcher();

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VULKAN_HPP_DEFAULT_DISPATCHER ggml_vk_default_dispatcher()

#ਅਨ੍ਤਰ੍ਭਾਵ <vulkan/vulkan.hpp>

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ VK_NV_cooperative_ਮੈਟ੍ਰਿਕਸ_decode_ਸਦਿਸ਼
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_NV_cooperative_ਮੈਟ੍ਰਿਕਸ_decode_ਸਦਿਸ਼ 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_NV_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_DECODE_ਸਦਿਸ਼_EXTENSION_NAME "VK_NV_cooperative_matrix_decode_vector"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_DECODE_ਸਦਿਸ਼_FEATURES_NV ((VkStructureType)1000689000)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ Vk_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Decode_ਸਦਿਸ਼_Features_NV {
    VkStructureType    sType;
    ਖਾਲੀ*              p_ਅਗਲਾ;
    VkBool32           cooperative_ਮੈਟ੍ਰਿਕਸ_Decode_ਸਦਿਸ਼;
} Vk_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Decode_ਸਦਿਸ਼_Features_NV;
#ਜੇ_ਅੰਤ

#ਜੇ __has_include(<spirv/unified1/spirv.hpp>)
#    include <spirv/unified1/spirv.hpp>
#ਜਾਂਜੇ __has_include(<spirv-ਸਿਰਲੇਖਾਂ/spirv.hpp>)
#    include <spirv-headers/spirv.hpp>
#ਜਾਂਜੇ __has_include(<spirv.hpp>)
#    include <spirv.hpp>
#ਨਹੀਂਤਾਂ
     // Fallback to let the compiler throw a standard "file not found" error
#    include <spirv/unified1/spirv.hpp>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <functional>
#ਅਨ੍ਤਰ੍ਭਾਵ <iomanip>
#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <tuple>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <deque>
#ਅਨ੍ਤਰ੍ਭਾਵ <sstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <utility>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <set>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <shared_mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <future>
#ਅਨ੍ਤਰ੍ਭਾਵ <condition_variable>
#ਅਨ੍ਤਰ੍ਭਾਵ <thread>

#ਜੇ defined(_MSC_VER)
# define NOMINMAX 1
# include <windows.h>
# define YIELD() YieldProcessor()
#ਜਾਂਜੇ defined(__clang__) || defined(__GNUC__)
# ਜੇ defined(__x86_64__) ||defined(__i386__)
#  include <immintrin.h>
#  define YIELD() _mm_pause()
# elif defined(__arm__) || defined(__aarch64__)
#  ਜੇ defined(__clang__)
#   include <arm_acle.h>
#   define YIELD() __yield()
#  ਨਹੀਂਤਾਂ
#   define YIELD() asm ਅਸਥਿਰ("yield")
#  endif
# endif
#ਜੇ_ਅੰਤ

#ਜੇ !defined(YIELD)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ YIELD()
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-vulkan-shaders.hpp"

#ਜੇ !defined(VK_KHR_shader_bfloat16)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_KHR_shader_bfloat16 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_KHR_SHADER_BFLOAT16_SPEC_ਸੰਸਕਰਣ                          1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME                        "VK_KHR_shader_bfloat16"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR ((VkStructureType)1000141000)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_COMPONENT_TYPE_BFLOAT16_KHR                               ((VkComponentTypeKHR)1000141000)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType                       sType;
    ਖਾਲੀ*                                 p_ਅਗਲਾ;
    VkBool32                              shaderBFloat16Type;
    VkBool32                              shader_BFloat16_Dot_ਗੁਣਨਫਲ;
    VkBool32                              shader_BFloat16_Cooperative_ਮੈਟ੍ਰਿਕਸ;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;
#ਜੇ_ਅੰਤ

#ਜੇ !defined(VK_VALVE_shader_mixed_float_dot_ਗੁਣਨਫਲ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_VALVE_shader_mixed_float_dot_ਗੁਣਨਫਲ 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_VALVE_SHADER_MIXED_FLOAT_DOT_ਗੁਣਨਫਲ_SPEC_ਸੰਸਕਰਣ 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_VALVE_SHADER_MIXED_FLOAT_DOT_ਗੁਣਨਫਲ_EXTENSION_NAME "VK_VALVE_shader_mixed_float_dot_product"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_ਗੁਣਨਫਲ_FEATURES_VALVE ((VkStructureType)1000673000)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ Vk_Physical_Device_Shader_Mixed_Float_Dot_ਗੁਣਨਫਲ_Features_VALVE {
    VkStructureType    sType;
    ਖਾਲੀ*              p_ਅਗਲਾ;
    VkBool32           shader_Mixed_Float_Dot_ਗੁਣਨਫਲ_Float16_Acc_Float32;
    VkBool32           shader_Mixed_Float_Dot_ਗੁਣਨਫਲ_Float16_Acc_Float16;
    VkBool32           shader_Mixed_Float_Dot_ਗੁਣਨਫਲ_BFloat16_Acc;
    VkBool32           shader_Mixed_Float_Dot_ਗੁਣਨਫਲ_Float8_Acc_Float32;
} Vk_Physical_Device_Shader_Mixed_Float_Dot_ਗੁਣਨਫਲ_Features_VALVE;
#ਜੇ_ਅੰਤ

#ਜੇ !defined(VK_EXT_shader_ocp_microscaling_types)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_EXT_shader_ocp_microscaling_types 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_EXT_SHADER_OCP_MICROSCALING_TYPES_SPEC_ਸੰਸਕਰਣ 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME "VK_EXT_shader_ocp_microscaling_types"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT ((VkStructureType)1000672000)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT {
    VkStructureType    sType;
    ਖਾਲੀ*              p_ਅਗਲਾ;
    VkBool32           shaderFloat4;
    VkBool32           shaderFloat6;
    VkBool32           shaderFloat8UnsignedE8M0;
    VkBool32           shaderMXInt8;
} VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT;
#ਜੇ_ਅੰਤ

#ਜੇ !defined(VK_EXT_shader_float8)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_EXT_shader_float8 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_EXT_SHADER_FLOAT8_SPEC_ਸੰਸਕਰਣ 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_EXT_SHADER_FLOAT8_EXTENSION_NAME "VK_EXT_shader_float8"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT ((VkStructureType)1000567000)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ VkPhysicalDeviceShaderFloat8FeaturesEXT {
    VkStructureType    sType;
    ਖਾਲੀ*              p_ਅਗਲਾ;
    VkBool32           shaderFloat8;
    VkBool32           shader_Float8_Cooperative_ਮੈਟ੍ਰਿਕਸ;
} VkPhysicalDeviceShaderFloat8FeaturesEXT;
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME "VK_KHR_internally_synchronized_queues"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR ((VkStructureType)1000504000)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_DEVICE_ਕਤਾਰ_CREATE_INTERNALLY_SYNCHRONIZED_ਦੋਆਧਾਰੀ_ਅੰਕ_KHR ((Vk_Device_ਕਤਾਰ_Create_Flag_ਬਿੱਟ)0x00000004)

// Compile-time constant guaranteed; no runtime initialization overhead
ਸਥਿਰ constexpr vk::Device_ਕਤਾਰ_Create_Flag_ਬਿੱਟ eInternallySynchronizedKHR =
    static_cast<vk::Device_ਕਤਾਰ_Create_Flag_ਬਿੱਟ>(0x00000004);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR {
    VkStructureType    sType;
    ਖਾਲੀ* p_ਅਗਲਾ;
    VkBool32           internallySynchronizedQueues;
} VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR;
#ਨਹੀਂਤਾਂ
ਸਥਿਰ constexpr vk::Device_ਕਤਾਰ_Create_Flag_ਬਿੱਟ eInternallySynchronizedKHR = vk::Device_ਕਤਾਰ_Create_Flag_ਬਿੱਟ::eInternallySynchronizedKHR;
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CEIL_DIV(M, N) (((M) / (N)) + (((M) % (N)) != 0))

ਸਥਿਰ bool is_pow2(uint32_t x) { ਵਾਪਸ x > 1 && (x & (x-1)) == 0; }

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_VENDOR_ID_AMD 0x1002

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_VENDOR_ID_APPLE 0x106b

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_VENDOR_ID_INTEL 0x8086

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_VENDOR_ID_NVIDIA 0x10de

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_VENDOR_ID_QUALCOMM 0x5143

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_VENDOR_ID_SAMSUNG 0x144d

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_DEVICE_DESCRIPTOR_POOL_ਮਾਪ 256

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_CHECK(err, ਸੁਨੇਹਾ_2, dev)                                     \
    ਕਰੋ {                                                            \
        vk::ਨਤੀਜਾ_2 err_;                                            \
        try {                                                       \
            err_ = (err);                                           \
        } catch (vk::Device_ਗੁਆਚਾ_Error &) {                           \
            ggml_vk_print_device_ਗੁਆਚਾ_info(dev);                    \
            GGML_LOG_ERROR("ggml_vulkan: %s at %s:%d\n",            \
                #err, __FILE__, __LINE__);                          \
            throw;                                                  \
        }                                                           \
        ਜੇ (err_ != vk::ਨਤੀਜਾ_2::eSuccess) {                         \
            GGML_LOG_ERROR("ggml_vulkan: %s error %s at %s:%d\n",   \
                #err, ਮੰਜ਼ਿਲ_string(err_).c_str(), __FILE__, __LINE__); \
            throw vk::SystemError(vk::make_error_ਕੋਡ(err_),        \
                "ggml_vulkan: " ਸੁਨੇਹਾ_2);                               \
        }                                                           \
    } ਜਦੋਂ (0)

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_DEBUG
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_LOG_DEBUG(ਸੁਨੇਹਾ_2) std::cerr << ਸੁਨੇਹਾ_2 << std::endl
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_LOG_DEBUG(ਸੁਨੇਹਾ_2) ((ਖਾਲੀ) 0)
#ਜੇ_ਅੰਤ // GGML_VULKAN_DEBUG

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_PARAMETER_ਗਿਣਤੀ 12

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_FUSED_ADDS (MAX_PARAMETER_ਗਿਣਤੀ - 3)

ਬਣਤਰ vk_pipeline_struct;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::shared_ptr<ਬਣਤਰ vk_pipeline_struct> vk_pipeline;

ਬਣਤਰ vk_pipeline_struct {
    std::string name;
    vk::ShaderModule shader_module;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t ਧੱਕੋ_constant_ਮਾਪ;
    uint32_t parameter_ਗਿਣਤੀ;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
    // true if fields have been set by ggml_vk_create_pipeline
    bool initialized {};
    // true while a compile is in flight, used to dedupe concurrent claims.
    // Protected by device->compile_mutex.
    bool compile_pending {};
    // set to true when the shader has been compiled
    std::atomic<bool> compiled {};
    // number of registers used, extracted from pipeline executable properties
    uint32_t ਰਜਿਸਟਰ_ਗਿਣਤੀ {};

#ਜੇ defined(VK_EXT_shader_64bit_indexing)
    bool is_64b_indexing {};
#ਜੇ_ਅੰਤ
    // linked list of pipelines for multiple compilation variants.
    // currently only used to compile a 64-bit indexing variant.
    vk_pipeline ਅਗਲਾ;
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;

ਬਣਤਰ vk_matmul_pipeline_ਕੁੰਜੀ {
    ggml_type type_a;
    ggml_type type_b;
    bool mul_mat_id;
    bool f16acc;

    bool ਸੰਚਾਲਕ<(ਅਚਲ vk_matmul_pipeline_ਕੁੰਜੀ & o) ਅਚਲ {
        ਵਾਪਸ std::tie(type_a, type_b, mul_mat_id, f16acc)
             < std::tie(o.type_a, o.type_b, o.mul_mat_id, o.f16acc);
    }
};

ਬਣਤਰ vk_matmul_pipeline_pair {
    vk_pipeline unaligned;
    vk_pipeline aligned;
    uint32_t align;
};

ਬਣਤਰ vk_tile_config {
    std::ਸਦਿਸ਼<uint32_t> warptile;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
};

using matmul_tile_selector_t = std::function<uint32_t(
    uint32_t m, uint32_t n, uint32_t k, uint32_t shader_core_ਗਿਣਤੀ,
    ਅਚਲ std::ਸਦਿਸ਼<vk_matmul_pipeline_pair>& configs)>;

ਬਣਤਰ vk_device_struct;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::shared_ptr<vk_device_struct> vk_device;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::weak_ptr<vk_device_struct> vk_device_ref;

ਬਣਤਰ vk_ਬਫਰ_struct;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::shared_ptr<vk_ਬਫਰ_struct> vk_ਬਫਰ;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::weak_ptr<vk_ਬਫਰ_struct> vk_ਬਫਰ_ref;

ਬਣਤਰ ggml_backend_vk_ਬਫਰ_type_context {
    std::string name;
    vk_device device;
};

ਬਣਤਰ vk_command_ਬਫਰ {
    vk::Command_ਬਫਰ buf;
    uint64_t use_counter = 0;
    bool ਵਿੱਚ_use = false;
};

ਬਣਤਰ vk_ਕਤਾਰ;

ਬਣਤਰ vk_command_pool {
    ਖਾਲੀ init(vk_device& device, vk_ਕਤਾਰ *q_);
    ਖਾਲੀ destroy(vk::Device& device);

    vk::CommandPool pool;
    // Using deque so the pointers to command buffers
    // remain valid even if we add more
    std::deque<vk_command_ਬਫਰ> cmd_buffers;

    vk_ਕਤਾਰ *q;

    size_t buffers_ਵਿੱਚ_use() ਅਚਲ {
        ਵਾਪਸ std::ਗਿਣਤੀ_if(cmd_buffers.begin(), cmd_buffers.ਸਿਰਾ(),
            [](ਅਚਲ ਸਵੈਚਾਲਿਤ& cb) { ਵਾਪਸ cb.ਵਿੱਚ_use; });
    }
};

ਬਣਤਰ vk_ਕਤਾਰ_handle {
    vk::ਕਤਾਰ_2 ਕਤਾਰ;
    vk_device_ref device;
    std::mutex * device_submit_mutex = nullptr;
    ਆਭਾਸੀ ਖਾਲੀ submit(vk::ArrayProxy<ਅਚਲ vk::SubmitInfo> submits, vk::Fence fence) = 0;
    ਆਭਾਸੀ ਖਾਲੀ ਤਾਲਾ()   {}   // no-op by default (internally synchronized case)
    ਆਭਾਸੀ ਖਾਲੀ unlock() {}
    ਆਭਾਸੀ ~vk_ਕਤਾਰ_handle() = ਮੂਲ;
};

ਬਣਤਰ vk_ਕਤਾਰ_handle_synchronized : vk_ਕਤਾਰ_handle {
    std::mutex mutex;
    ਖਾਲੀ submit(vk::ArrayProxy<ਅਚਲ vk::SubmitInfo> submits, vk::Fence fence) override;

    ਖਾਲੀ ਤਾਲਾ()   override { mutex.ਤਾਲਾ(); }
    ਖਾਲੀ unlock() override { mutex.unlock(); }
};

ਬਣਤਰ vk_ਕਤਾਰ_handle_unsynchronized : vk_ਕਤਾਰ_handle {
    ਖਾਲੀ submit(vk::ArrayProxy<ਅਚਲ vk::SubmitInfo> submits, vk::Fence fence) override;

    // lock()/unlock() inherited no-ops
};

ਬਣਤਰ vk_ਕਤਾਰ {
    uint32_t ਕਤਾਰ_family_ਸੂਚਕਾਂਕ;
    std::shared_ptr<vk_ਕਤਾਰ_handle> handle;

    vk_command_pool cmd_pool;

    vk::PipelineStageFlags stage_flags;

    bool transfer_only;
};

ਸਥਿਰ constexpr uint32_t mul_mat_vec_max_ਖਾਨੇ_ਗਿਣਤੀ = 8;

ਸਥਿਰ constexpr uint32_t p021_max_gqa_ਅਨੁਪਾਤ = 8;

ਗਿਣਤੀ vk_device_architecture {
    OTHER,
    AMD_GCN,
    AMD_RDNA1,
    AMD_RDNA2,
    AMD_RDNA3,
    AMD_RDNA4,
    INTEL_XE1,
    INTEL_XE2,
    NVIDIA_PRE_TURING,
    NVIDIA_TURING,
    QUALCOMM_ADRENO,
};

ਗਿਣਤੀ vk_conv_shapes {
    CONV_SHAPE_128x128,
    CONV_SHAPE_64x32,
    CONV_SHAPE_32x256,
    CONV_SHAPE_64x128,
    CONV_SHAPE_ਗਿਣਤੀ,
};

ਬਣਤਰ vk_conv_ਖੰਡ_ਮਾਪ {
    uint32_t K;
    uint32_t NPQ;
    uint32_t CRS;
};

ਸਤਰਅੰਦਰ vk_conv_ਖੰਡ_ਮਾਪ vk_conv_ਖੰਡ_sizes[CONV_SHAPE_ਗਿਣਤੀ] = {
    // K   NPQ  CRS
    { 128, 128, 16 }, // CONV_SHAPE_128x128
    {  64,  32, 32 }, // CONV_SHAPE_64x32
    {  32, 256, 16 }, // CONV_SHAPE_32x256
    {  64, 128, 16 }, // CONV_SHAPE_64x128
};

ਗਿਣਤੀ dmmv_wg_sizes {
    DMMV_WG_ਮਾਪ_SUBGROUP,
    DMMV_WG_ਮਾਪ_LARGE,
    DMMV_WG_ਮਾਪ_ਗਿਣਤੀ,
};

ਗਿਣਤੀ Fa_ਕੋਡ_ਰਸਤਾ {
    FA_SCALAR,
    FA_COOPMAT1,
    FA_COOPMAT2,
};

ਬਣਤਰ vk_fa_pipeline_state {
    uint32_t HSK, HSV;
    uint32_t Br, Bc;
    uint32_t D_ਵੰਡੋ, ਪੰਕਤੀ_ਵੰਡੋ;
    bool shmem_staging;
    Fa_ਕੋਡ_ਰਸਤਾ ਰਸਤਾ;
    uint32_t workgroup_ਮਾਪ, subgroup_ਮਾਪ;
    bool aligned;
    bool f32acc;
    uint32_t flags;
    uint32_t ਹੱਦਬੰਦੀ_occupancy_shmem;
    ggml_type k_type;
    ggml_type v_type;

    bool ਸੰਚਾਲਕ<(ਅਚਲ vk_fa_pipeline_state &b) ਅਚਲ {
        ਵਾਪਸ std::tie(HSK, HSV, Br, Bc, D_ਵੰਡੋ, ਪੰਕਤੀ_ਵੰਡੋ, shmem_staging, ਰਸਤਾ, workgroup_ਮਾਪ, subgroup_ਮਾਪ, aligned, f32acc, flags, ਹੱਦਬੰਦੀ_occupancy_shmem, k_type, v_type) <
               std::tie(b.HSK, b.HSV, b.Br, b.Bc, b.D_ਵੰਡੋ, b.ਪੰਕਤੀ_ਵੰਡੋ, b.shmem_staging, b.ਰਸਤਾ, b.workgroup_ਮਾਪ, b.subgroup_ਮਾਪ, b.aligned, b.f32acc, b.flags, b.ਹੱਦਬੰਦੀ_occupancy_shmem, b.k_type, b.v_type);
    }
};

ਬਣਤਰ vk_conv2d_pipeline_state {
    vk_conv2d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t p0, uint32_t p1, uint32_t d0, uint32_t d1, uint32_t KW, uint32_t KH, uint32_t aligned)
        : s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), KW(KW), KH(KH), aligned(aligned) {}

    uint32_t s0, s1, p0, p1, d0, d1, KW, KH;
    // when set, shader can skip K/CRS/NPQ bounds checks and address clamps
    uint32_t aligned;

    bool ਸੰਚਾਲਕ<(ਅਚਲ vk_conv2d_pipeline_state &b) ਅਚਲ {
        ਵਾਪਸ std::tie(s0, s1, p0, p1, d0, d1, KW, KH, aligned) <
               std::tie(b.s0, b.s1, b.p0, b.p1, b.d0, b.d1, b.KW, b.KH, b.aligned);
    }
};

ਬਣਤਰ vk_conv3d_pipeline_state {
    vk_conv3d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t p0, uint32_t p1, uint32_t p2,
                             uint32_t d0, uint32_t d1, uint32_t d2, uint32_t KW, uint32_t KH, uint32_t KD, uint32_t aligned)
        : s0(s0), s1(s1), s2(s2), p0(p0), p1(p1), p2(p2), d0(d0), d1(d1), d2(d2), KW(KW), KH(KH), KD(KD), aligned(aligned) {}

    uint32_t s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD;
    uint32_t aligned;

    bool ਸੰਚਾਲਕ<(ਅਚਲ vk_conv3d_pipeline_state &b) ਅਚਲ {
        ਵਾਪਸ std::tie(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned) <
               std::tie(b.s0, b.s1, b.s2, b.p0, b.p1, b.p2, b.d0, b.d1, b.d2, b.KW, b.KH, b.KD, b.aligned);
    }
};

ਬਣਤਰ vk_solve_tri_pipeline_state {
    vk_solve_tri_pipeline_state(uint32_t N, uint32_t K)
        : N(N), K(K) {}

    uint32_t N, K;

    bool ਸੰਚਾਲਕ<(ਅਚਲ vk_solve_tri_pipeline_state &b) ਅਚਲ {
        ਵਾਪਸ std::tie(N, K) <
               std::tie(b.N, b.K);
    }
};

ਗਿਣਤੀ shader_reduction_ਬਹੁਲਕ {
    SHADER_REDUCTION_ਬਹੁਲਕ_SHMEM,
    SHADER_REDUCTION_ਬਹੁਲਕ_HYBRID,
    SHADER_REDUCTION_ਬਹੁਲਕ_SUBGROUP,
    SHADER_REDUCTION_ਬਹੁਲਕ_ਗਿਣਤੀ,
};

ਸਥਿਰ constexpr uint32_t ਅੰਕ_argsort_pipelines = 11;

ਸਥਿਰ constexpr uint32_t ਅੰਕ_topk_moe_pipelines = 10;

ਸਥਿਰ constexpr uint32_t ਅੰਕ_topk_pipelines = 11;

ਸਥਿਰ constexpr std::initializer_list<ggml_op> topk_moe_early_softmax_norm{ GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_ਪੰਕਤੀਆਂ, GGML_OP_RESHAPE,
                                                                             GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ, GGML_OP_CLAMP,    GGML_OP_DIV,
                                                                             GGML_OP_RESHAPE };

ਸਥਿਰ constexpr std::initializer_list<ggml_op> topk_moe_sigmoid_norm_bias{ GGML_OP_UNARY,    GGML_OP_RESHAPE,  GGML_OP_ਜੋੜੋ,
                                                                            GGML_OP_ARGSORT,  GGML_OP_VIEW,     GGML_OP_GET_ਪੰਕਤੀਆਂ,
                                                                            GGML_OP_RESHAPE,  GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ, GGML_OP_CLAMP,
                                                                            GGML_OP_DIV,      GGML_OP_RESHAPE };

ਸਥਿਰ constexpr std::initializer_list<ggml_op> topk_moe_sqrt_softplus_norm_bias{ GGML_OP_UNARY,    GGML_OP_SQRT,
                                                                                  GGML_OP_RESHAPE,  GGML_OP_ਜੋੜੋ,
                                                                                  GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                                  GGML_OP_GET_ਪੰਕਤੀਆਂ, GGML_OP_RESHAPE,
                                                                                  GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ, GGML_OP_CLAMP,
                                                                                  GGML_OP_DIV,      GGML_OP_RESHAPE };

ਸਥਿਰ constexpr std::initializer_list<ggml_op> topk_moe_early_softmax     { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_ਪੰਕਤੀਆਂ };

ਸਥਿਰ constexpr std::initializer_list<ggml_op> topk_moe_late_softmax      { GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                             GGML_OP_GET_ਪੰਕਤੀਆਂ, GGML_OP_RESHAPE,
                                                                             GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

ਸਥਿਰ constexpr std::initializer_list<ggml_op> snake_ਨਮੂਨਾ              { GGML_OP_MUL,      GGML_OP_SIN,
                                                                             GGML_OP_SQR,      GGML_OP_MUL,
                                                                             GGML_OP_ਜੋੜੋ };

ਸਥਿਰ constexpr std::initializer_list<ggml_op> topk_qsa_ਨਮੂਨਾ { GGML_OP_GET_ਪੰਕਤੀਆਂ, GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ,
                                                                   GGML_OP_CONT,     GGML_OP_CPY,
                                                                   GGML_OP_RESHAPE,  GGML_OP_ਜੋੜੋ,
                                                                   GGML_OP_ਸਿਖਰ_K };

ਸਥਿਰ constexpr std::initializer_list<std::array<ਪੂਰਨ, 3>> topk_qsa_ਕਿਨਾਰੇ {
    { 1, 0, 0 }, // permute->src[0] == get_rows
    { 2, 0, 1 }, // cont->src[0]    == permute
    { 4, 0, 3 }, // reshape->src[0] == cpy (mask cast)
    { 5, 0, 2 }, // add->src[0]     == cont
    { 5, 1, 4 }, // add->src[1]     == reshape
    { 6, 0, 5 }, // top_k->src[0]   == add
};

ਸਥਿਰ constexpr std::initializer_list<ggml_op> rms_norm_mul_ਜੋੜੋ_mul_ਨਮੂਨਾ { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ਜੋੜੋ, GGML_OP_MUL };

ਸਥਿਰ constexpr std::initializer_list<ggml_op> rms_norm_mul_ਜੋੜੋ_ਨਮੂਨਾ     { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ਜੋੜੋ };

ਸਥਿਰ constexpr std::initializer_list<ggml_op> rms_norm_mul_rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ };

ਸਥਿਰ constexpr std::initializer_list<ggml_op> rms_norm_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ };

ਸਥਿਰ constexpr std::initializer_list<ggml_op> rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ };

// scale_out*sigmoid(scale_in*x) as the hc_post weights (qwen4exp hc_combine)
ਸਥਿਰ constexpr std::initializer_list<ggml_op> hc_post_gate_ਨਮੂਨਾ { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

ਸਥਿਰ constexpr std::initializer_list<std::array<ਪੂਰਨ, 3>> hc_post_gate_ਕਿਨਾਰੇ {
    { 1, 0, 0 }, // sigmoid->src[0] == scale
    { 2, 0, 1 }, // scale->src[0]   == sigmoid
    { 3, 2, 2 }, // hc_post->src[2] == scale (post)
};

ਸਥਿਰ constexpr std::initializer_list<std::array<ਪੂਰਨ, 3>> topk_moe_early_softmax_norm_ਕਿਨਾਰੇ {
    { 1, 0, 0 }, // reshape->src[0]  == softmax
    { 2, 0, 0 }, // argsort->src[0]  == softmax
    { 3, 0, 2 }, // view->src[0]     == argsort
    { 4, 0, 1 }, // get_rows->src[0] == reshape
    { 4, 1, 3 }, // get_rows->src[1] == view
    { 5, 0, 4 }, // reshape->src[0]  == get_rows
    { 6, 0, 5 }, // sum_rows->src[0] == reshape
    { 7, 0, 6 }, // clamp->src[0]    == sum_rows
    { 8, 0, 5 }, // div->src[0]      == reshape
    { 8, 1, 7 }, // div->src[1]      == clamp
    { 9, 0, 8 }, // reshape->src[0]  == div
};

ਸਥਿਰ constexpr std::initializer_list<std::array<ਪੂਰਨ, 3>> topk_moe_sigmoid_norm_bias_ਕਿਨਾਰੇ {
    { 1, 0, 0 }, // reshape->src[0]  == sigmoid
    { 2, 0, 0 }, // add->src[0]      == sigmoid
    { 3, 0, 2 }, // argsort->src[0]  == add
    { 4, 0, 3 }, // view->src[0]     == argsort
    { 5, 0, 1 }, // get_rows->src[0] == reshape
    { 5, 1, 4 }, // get_rows->src[1] == view
    { 6, 0, 5 }, // reshape->src[0]  == get_rows
    { 7, 0, 6 }, // sum_rows->src[0] == reshape
    { 8, 0, 7 }, // clamp->src[0]    == sum_rows
    { 9, 0, 6 }, // div->src[0]      == reshape
    { 9, 1, 8 }, // div->src[1]      == clamp
    {10, 0, 9 }, // reshape->src[0]  == div
};

ਸਥਿਰ constexpr std::initializer_list<std::array<ਪੂਰਨ, 3>> topk_moe_sqrt_softplus_norm_bias_ਕਿਨਾਰੇ {
    { 1, 0, 0 }, // sqrt->src[0]     == softplus
    { 2, 0, 1 }, // reshape->src[0]  == sqrt
    { 3, 0, 1 }, // add->src[0]      == sqrt
    { 4, 0, 3 }, // argsort->src[0]  == add
    { 5, 0, 4 }, // view->src[0]     == argsort
    { 6, 0, 2 }, // get_rows->src[0] == reshape
    { 6, 1, 5 }, // get_rows->src[1] == view
    { 7, 0, 6 }, // reshape->src[0]  == get_rows
    { 8, 0, 7 }, // sum_rows->src[0] == reshape
    { 9, 0, 8 }, // clamp->src[0]    == sum_rows
    {10, 0, 7 }, // div->src[0]      == reshape
    {10, 1, 9 }, // div->src[1]      == clamp
    {11, 0,10 }, // reshape->src[0]  == div
};

ਸਥਿਰ constexpr std::initializer_list<std::array<ਪੂਰਨ, 3>> topk_moe_early_softmax_ਕਿਨਾਰੇ {
    { 1, 0, 0 }, // reshape->src[0]  == softmax
    { 2, 0, 0 }, // argsort->src[0]  == softmax
    { 3, 0, 2 }, // view->src[0]     == argsort
    { 4, 0, 1 }, // get_rows->src[0] == reshape
    { 4, 1, 3 }, // get_rows->src[1] == view
};

ਸਥਿਰ constexpr std::initializer_list<std::array<ਪੂਰਨ, 3>> topk_moe_late_softmax_ਕਿਨਾਰੇ {
    { 1, 0, 0 }, // view->src[0]     == argsort
    { 2, 1, 1 }, // get_rows->src[1] == view
    { 3, 0, 2 }, // reshape->src[0]  == get_rows
    { 4, 0, 3 }, // soft_max->src[0] == reshape
    { 5, 0, 4 }, // reshape->src[0]  == soft_max
};

ਗਿਣਤੀ topk_moe_ਬਹੁਲਕ {
    TOPK_MOE_EARLY_SOFTMAX,
    TOPK_MOE_EARLY_SOFTMAX_NORM,
    TOPK_MOE_LATE_SOFTMAX,
    TOPK_MOE_SIGMOID_NORM_BIAS,
    TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS,
    TOPK_MOE_ਗਿਣਤੀ,
};

ਗਿਣਤੀ rms_norm_ਬਹੁਲਕ {
    RMS_NORM_MUL,
    RMS_NORM_MUL_ਜੋੜੋ,
    RMS_NORM_MUL_ਜੋੜੋ_MUL,
    RMS_NORM_MUL_ROPE,
    RMS_NORM_MUL_ROPE_VIEW_ਸੈੱਟ_ਪੰਕਤੀਆਂ,
    RMS_NORM_VIEW_ਸੈੱਟ_ਪੰਕਤੀਆਂ,
    RMS_NORM_ਗਿਣਤੀ,
};

ਸਥਿਰ constexpr std::initializer_list<std::array<ਪੂਰਨ, 3>> rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਕਿਨਾਰੇ {
    { 1, 0, 0 }, // view->src[0]     == rope
    { 2, 0, 1 }, // set_rows->src[0] == view
};

ਸਥਿਰ constexpr std::initializer_list<std::array<ਪੂਰਨ, 3>> rms_norm_mul_rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਕਿਨਾਰੇ {
    { 1, 0, 0 }, // mul->src[0]      == rms
    { 2, 0, 1 }, // rope->src[0]     == mul
    { 3, 0, 2 }, // view->src[0]     == rope
    { 4, 0, 3 }, // set_rows->src[0] == view
};

ਸਥਿਰ constexpr std::initializer_list<std::array<ਪੂਰਨ, 3>> rms_norm_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਕਿਨਾਰੇ {
    { 1, 0, 0 }, // view->src[0]     == rms_norm
    { 2, 0, 1 }, // set_rows->src[0] == view
};

ਸਥਿਰ constexpr std::array<ggml_type, 9> lightning_indexer_k_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_0,
    GGML_TYPE_IQ4_NL,
};

class vk_ਸਮ੍ਰਿਤੀ_logger;

ਬਣਤਰ vk_device_struct {
    std::recursive_mutex mutex;
    std::mutex ਕਤਾਰ_submit_mutex;

    // Guards compile_pending, all_pipelines, and the dynamic pipeline maps
    // (flash_attn, fa_mask_opt, solve_tri, conv2d, etc). The actual compile
    // runs with no lock held, so different pipelines can compile in parallel.
    // Lock order is device->mutex -> compile_mutex, never the reverse.
    std::mutex compile_mutex;
    std::condition_variable compile_cv;

    uint32_t debug_cmdbuf_idx {};

    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties properties;
    std::string name;
    uint64_t max_ਸਮ੍ਰਿਤੀ_ਵੰਡ_ਮਾਪ;
    uint64_t max_ਬਫਰ_ਮਾਪ;
    uint64_t suballocation_ਖੰਡ_ਮਾਪ;
    uint64_t min_imported_host_ਸੂਚਕ_alignment;
    bool external_ਸਮ੍ਰਿਤੀ_host {};
    bool fp16;
    bool bf16;
    bool pipeline_robustness;
    bool ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ;
    vk::Device device;
    uint32_t vendor_id;
    vk::DriverId driver_id;
    vk_device_architecture architecture;
    std::unique_ptr<vk_ਕਤਾਰ> compute_ਕਤਾਰ;
    std::unique_ptr<vk_ਕਤਾਰ> transfer_ਕਤਾਰ;
    bool single_ਕਤਾਰ;
    bool support_async;
    bool async_use_transfer_ਕਤਾਰ;
    bool has_internally_synchronized_queues = false;
    uint32_t subgroup_ਮਾਪ;
    uint32_t subgroup_ਮਾਪ_log2;
    uint32_t shader_core_ਗਿਣਤੀ;
    bool uma;
    bool prefer_host_ਸਮ੍ਰਿਤੀ;
    bool float_controls_rte_fp16;
    bool float_controls_denorm_preserve_fp16;
    bool subgroup_basic;
    bool subgroup_arithmetic;
    bool subgroup_shuffle;
    bool subgroup_ballot;
    bool subgroup_clustered;
    bool subgroup_vote;
    bool multi_ਜੋੜੋ;
    bool shader_int64;
    bool ਬਫਰ_device_ਪਤਾ;
    bool vulkan_ਸਮ੍ਰਿਤੀ_model;

    bool ਜੋੜੋ_rms_fusion;
    uint32_t partials_binding_alignment;
    uint32_t max_nodes_ਪ੍ਰਤੀ_submit;

    bool shader_64b_indexing;

    bool integer_dot_ਗੁਣਨਫਲ;
    // 0: default, 1: force mmvq, -1: disable mmvq
    int32_t mmvq_ਬਹੁਲਕ;

    bool subgroup_ਮਾਪ_control;
    uint32_t subgroup_min_ਮਾਪ;
    uint32_t subgroup_max_ਮਾਪ;
    bool subgroup_require_ਭਰਿਆ_support;

    // floor(log2(maxComputeWorkGroupInvocations))
    uint32_t max_workgroup_ਮਾਪ_log2 {};

    bool coopmat_support;
    bool coopmat_acc_f32_support {};
    bool coopmat_acc_f16_support {};
    bool coopmat_bf16_support {};
    bool coopmat_support_16x16x16_f16acc {};
    bool coopmat_support_16x16x16_f32acc {};
    bool coopmat1_fa_support {};
    uint32_t coopmat_m;
    uint32_t coopmat_n;
    uint32_t coopmat_k;

    bool coopmat_int_support;
    uint32_t coopmat_int_m;
    uint32_t coopmat_int_n;
    uint32_t coopmat_int_k;

    bool coopmat2;
    bool coopmat2_bf16_support {};
    bool coopmat2_decode_ਸਦਿਸ਼;

    bool dot2_f16 {};
    bool ocp_fp4 {};

    bool pipeline_executable_properties_support {};

    bool device_fault {};
    PFN_vkGetDeviceFaultInfoEXT pfn_vkGetDeviceFaultInfoEXT {};

    bool serialize_submissions {};

    ਅਚਲ ggml_cgraph * diag_cgraph {};
    ਪੂਰਨ diag_prev_ਸ਼ੁਰੂ = -1;
    ਪੂਰਨ diag_prev_ਸਿਰਾ = -1;

    size_t idx;

    bool mul_mat_l[GGML_TYPE_ਗਿਣਤੀ];
    bool mul_mat_m[GGML_TYPE_ਗਿਣਤੀ];
    bool mul_mat_s[GGML_TYPE_ਗਿਣਤੀ];
    bool mul_mat_id_l[GGML_TYPE_ਗਿਣਤੀ];
    bool mul_mat_id_m[GGML_TYPE_ਗਿਣਤੀ];
    bool mul_mat_id_s[GGML_TYPE_ਗਿਣਤੀ];

    // Separate flags for the q8_1 (integer dot) mmq path, whose shader uses
    // a different shared-memory layout than the float matmul shaders.
    bool mul_mat_l_int[GGML_TYPE_ਗਿਣਤੀ];
    bool mul_mat_m_int[GGML_TYPE_ਗਿਣਤੀ];
    bool mul_mat_s_int[GGML_TYPE_ਗਿਣਤੀ];
    bool mul_mat_id_l_int[GGML_TYPE_ਗਿਣਤੀ];
    bool mul_mat_id_m_int[GGML_TYPE_ਗਿਣਤੀ];
    bool mul_mat_id_s_int[GGML_TYPE_ਗਿਣਤੀ];

    vk::Descriptor_ਸੈੱਟ_Layout dsl;

    std::map<vk_matmul_pipeline_ਕੁੰਜੀ, std::ਸਦਿਸ਼<vk_matmul_pipeline_pair>> pipeline_matmul;
    matmul_tile_selector_t matmul_tile_selector;
    matmul_tile_selector_t matmul_id_tile_selector;

    vk_pipeline pipeline_matmul_ਵੰਡੋ_k_reduce;
    vk_pipeline pipeline_quantize_q8_1_x4;

    vk_pipeline pipeline_dequant[GGML_TYPE_ਗਿਣਤੀ];
    vk_pipeline pipeline_dequant_transpose[GGML_TYPE_ਗਿਣਤੀ]; // fused dequant+transpose for FA quant-KV
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_ਮਾਪ_ਗਿਣਤੀ][GGML_TYPE_ਗਿਣਤੀ][mul_mat_vec_max_ਖਾਨੇ_ਗਿਣਤੀ];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[DMMV_WG_ਮਾਪ_ਗਿਣਤੀ][GGML_TYPE_ਗਿਣਤੀ][mul_mat_vec_max_ਖਾਨੇ_ਗਿਣਤੀ];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[DMMV_WG_ਮਾਪ_ਗਿਣਤੀ][GGML_TYPE_ਗਿਣਤੀ];

    vk_pipeline pipeline_dequant_mul_mat_vec_q8_1_f32[DMMV_WG_ਮਾਪ_ਗਿਣਤੀ][GGML_TYPE_ਗਿਣਤੀ][mul_mat_vec_max_ਖਾਨੇ_ਗਿਣਤੀ];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_q8_1_f32[DMMV_WG_ਮਾਪ_ਗਿਣਤੀ][GGML_TYPE_ਗਿਣਤੀ];

    vk_pipeline pipeline_mul_mat_vec_p021_f16_f32[p021_max_gqa_ਅਨੁਪਾਤ];
    vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
    vk_pipeline pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_ਗਿਣਤੀ];
    vk_pipeline pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_ਗਿਣਤੀ];
    vk_pipeline pipeline_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_f32;
    vk_pipeline pipeline_acc_f32;
    vk_pipeline pipeline_ਸੈੱਟ_f32;

    // [src0 0=fp32,1=fp16][src1 0=fp32,1=fp16][dst 0=fp32,1=fp16]
    vk_pipeline pipeline_ਜੋੜੋ[2][2][2];
    vk_pipeline pipeline_ਜੋੜੋ_norepeat[2][2][2];
    vk_pipeline pipeline_sub[2][2][2];
    vk_pipeline pipeline_sub_norepeat[2][2][2];
    vk_pipeline pipeline_mul[2][2][2];
    vk_pipeline pipeline_mul_norepeat[2][2][2];
    vk_pipeline pipeline_div[2][2][2];
    vk_pipeline pipeline_div_norepeat[2][2][2];
    vk_pipeline pipeline_ਜੋੜੋ_rms[2][2][2];
    vk_pipeline pipeline_ਜੋੜੋ_rms_norepeat[2][2][2];

    // indexed by num_additional_fused_ops == num_adds - 1
    vk_pipeline pipeline_multi_ਜੋੜੋ[MAX_FUSED_ADDS];
    vk_pipeline pipeline_multi_ਜੋੜੋ_rms[MAX_FUSED_ADDS];

    vk_pipeline pipeline_ਜੋੜੋ_id_f32;

    vk_pipeline pipeline_concat_i8, pipeline_concat_i16, pipeline_concat_i32, pipeline_concat_i64;
    vk_pipeline pipeline_upscale_ਸਭ_ਤੋਂ_ਨੇੜੇ_f32, pipeline_upscale_bilinear_f32, pipeline_upscale_bicubic_f32, pipeline_upscale_bilinear_antialias_f32;
    vk_pipeline pipeline_scale_f32;
    vk_pipeline pipeline_log[2];
    vk_pipeline pipeline_tri[2];
    vk_pipeline pipeline_diag[2];
    vk_pipeline pipeline_clamp[2];
    vk_pipeline pipeline_pad_f32;
    vk_pipeline pipeline_pad_reflect_1d_f32;
    vk_pipeline pipeline_roll_f32;
    vk_pipeline pipeline_repeat_i32, pipeline_repeat_ਪਿੱਠ_f32;
    vk_pipeline pipeline_repeat_i16;
    vk_pipeline pipeline_cpy_32, pipeline_cpy_f32_f16, pipeline_cpy_16, pipeline_cpy_f16_f32, pipeline_cpy_f32_bf16, pipeline_cpy_bf16_f32, pipeline_cpy_bf16_f16, pipeline_cpy_f32_i32, pipeline_cpy_i32_f32;
    vk_pipeline pipeline_contig_cpy_32, pipeline_contig_cpy_f32_f16, pipeline_contig_cpy_16, pipeline_contig_cpy_f16_f32, pipeline_contig_cpy_f32_bf16, pipeline_contig_cpy_bf16_f32, pipeline_contig_cpy_bf16_f16, pipeline_contig_cpy_f32_i32, pipeline_contig_cpy_i32_f32;
    vk_pipeline pipeline_cpy_f32_quant[GGML_TYPE_ਗਿਣਤੀ];
    vk_pipeline pipeline_cpy_quant_f32[GGML_TYPE_ਗਿਣਤੀ];
    vk_pipeline pipeline_cpy_transpose_16, pipeline_cpy_transpose_32;
    vk_pipeline pipeline_cpy_transpose_02_16, pipeline_cpy_transpose_02_32;
    // [src0 0=fp32,1=fp16][dst]
    vk_pipeline pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ_i32[2][GGML_TYPE_ਗਿਣਤੀ];
    vk_pipeline pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ_i64[2][GGML_TYPE_ਗਿਣਤੀ];
    vk_pipeline pipeline_norm_f32;
    vk_pipeline pipeline_ਸਮੂਹ_norm_f32;
    vk_pipeline pipeline_rms_norm_f32;
    vk_pipeline pipeline_rms_norm_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_ਜੋੜੋ_f32;
    vk_pipeline pipeline_rms_norm_mul_ਜੋੜੋ_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_ਜੋੜੋ_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_ਜੋੜੋ_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_f32;
    vk_pipeline pipeline_rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_f16;
    vk_pipeline pipeline_rms_norm_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f16;
    vk_pipeline pipeline_rms_norm_ਪਿੱਠ_f32;
    vk_pipeline pipeline_l2_norm_f32;

    // [src/dst 0=fp32,1=fp16]
    vk_pipeline pipeline_exp[2];
    vk_pipeline pipeline_expm1[2];
    vk_pipeline pipeline_elu[2];
    vk_pipeline pipeline_gelu[2];
    vk_pipeline pipeline_gelu_erf[2];
    vk_pipeline pipeline_gelu_ਤੇਜ਼[2];
    vk_pipeline pipeline_silu[2];
    vk_pipeline pipeline_relu[2];
    vk_pipeline pipeline_sqr[2];
    vk_pipeline pipeline_sqrt[2];
    vk_pipeline pipeline_sin[2];
    vk_pipeline pipeline_cos[2];
    vk_pipeline pipeline_xielu[2];
    vk_pipeline pipeline_neg[2];
    vk_pipeline pipeline_tanh[2];
    vk_pipeline pipeline_sigmoid[2];
    vk_pipeline pipeline_hardsigmoid[2];
    vk_pipeline pipeline_hardswish[2];
    vk_pipeline pipeline_abs[2];
    vk_pipeline pipeline_softplus[2];
    vk_pipeline pipeline_ਕਦਮ[2];
    vk_pipeline pipeline_ਗੇੜਾ[2];
    vk_pipeline pipeline_ceil[2];
    vk_pipeline pipeline_floor[2];
    vk_pipeline pipeline_trunc[2];
    vk_pipeline pipeline_sgn[2];

    // fused UNARY+MUL pipelines: [op][f16][norepeat][op_on_b]
    vk_pipeline pipeline_unary_mul[4][2][2][2];

    vk_pipeline pipeline_add1_f16_f16;
    vk_pipeline pipeline_add1_f16_f32;
    vk_pipeline pipeline_add1_f32_f32;

    vk_pipeline pipeline_arange_f32;

    vk_pipeline pipeline_fill_f32;
    vk_pipeline pipeline_fill_f16;

    vk_pipeline pipeline_geglu[2];
    vk_pipeline pipeline_reglu[2];
    vk_pipeline pipeline_swiglu[2];
    vk_pipeline pipeline_swiglu_oai[2];
    vk_pipeline pipeline_swiglu_clamp[2];
    vk_pipeline pipeline_geglu_erf[2];
    vk_pipeline pipeline_geglu_ਤੇਜ਼[2];

    vk_pipeline pipeline_leaky_relu[2];
    vk_pipeline pipeline_silu_ਪਿੱਠ_f32;
    vk_pipeline pipeline_diag_ਨਕਾਬ_inf_f32;
    vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
    vk_pipeline pipeline_soft_max_f32_wg512, pipeline_soft_max_f32_f16_wg512;
    vk_pipeline pipeline_soft_max_ਪਿੱਠ_f32;

    vk_pipeline pipeline_soft_max_large1_f32, pipeline_soft_max_large1_f32_f16;
    vk_pipeline pipeline_soft_max_large2_f32, pipeline_soft_max_large2_f32_f16;
    vk_pipeline pipeline_soft_max_large3_f32, pipeline_soft_max_large3_f32_f16;

    vk_pipeline pipeline_rope_norm_f32, pipeline_rope_norm_f16, pipeline_rope_norm_f32_f16;
    vk_pipeline pipeline_rope_neox_f32, pipeline_rope_neox_f16, pipeline_rope_neox_f32_f16;
    vk_pipeline pipeline_rope_multi_f32, pipeline_rope_multi_f16, pipeline_rope_multi_f32_f16;
    vk_pipeline pipeline_rope_vision_f32, pipeline_rope_vision_f16;
    vk_pipeline pipeline_argsort_f32[ਅੰਕ_argsort_pipelines];
    vk_pipeline pipeline_argsort_large_f32[ਅੰਕ_argsort_pipelines];
    vk_pipeline pipeline_topk_f32[ਅੰਕ_topk_pipelines];
    vk_pipeline pipeline_topk_radix_f32;
    vk_pipeline pipeline_topk_radix_qsa; // qwen4 QSA indexer fusion (f16 mask)
    vk_pipeline pipeline_ਜੋੜ_ਪੰਕਤੀਆਂ_f32;
    vk_pipeline pipeline_cross_entropy_loss_f32, pipeline_cross_entropy_loss_f32_wg512;
    vk_pipeline pipeline_cross_entropy_loss_ਪਿੱਠ_f32, pipeline_cross_entropy_loss_ਪਿੱਠ_f32_wg512;
    vk_pipeline pipeline_fwht_f32[4];
    vk_pipeline pipeline_cumsum_f32;
    vk_pipeline pipeline_cumsum_small_f32;
    vk_pipeline pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline_cumsum_multipass2_f32;
    vk_pipeline pipeline_argmax_f32;
    vk_pipeline pipeline_ਗਿਣਤੀ_equal_i32;
    vk_pipeline pipeline_dsv4_hc_comb_f32;
    vk_pipeline pipeline_dsv4_hc_pre_f32;
    vk_pipeline pipeline_dsv4_hc_pre_gated_f32;
    vk_pipeline pipeline_dsv4_hc_post_f32;
    vk_pipeline pipeline_dsv4_hc_post_nocomb_f32;
    std::map<vk_solve_tri_pipeline_state, vk_pipeline> pipeline_solve_tri_f32;
    vk_pipeline pipeline_im2col_f32, pipeline_im2col_f32_f16;
    vk_pipeline pipeline_im2col_3d_f32, pipeline_im2col_3d_f32_f16;
    vk_pipeline pipeline_timestep_embedding_f32;
    vk_pipeline pipeline_conv_transpose_1d_f32;
    vk_pipeline pipeline_col2im_1d_f32;
    vk_pipeline pipeline_col2im_1d_f16;
    vk_pipeline pipeline_col2im_1d_bf16;
    vk_pipeline pipeline_ਬਾਹਰਲਾ_prod_f32;
    vk_pipeline pipeline_snake_f32;
    vk_pipeline pipeline_snake_f16;
    vk_pipeline pipeline_snake_bf16;
    vk_pipeline pipeline_pool1d_f32;
    vk_pipeline pipeline_pool2d_f32;
    vk_pipeline pipeline_pool2d_subgroup_f32;
    vk_pipeline pipeline_rwkv_wkv6_f32;
    vk_pipeline pipeline_rwkv_wkv7_f32;
    vk_pipeline pipeline_gated_ਰੇਖਿਕ_attn_f32;
    vk_pipeline pipeline_lightning_indexer_f32[GGML_TYPE_ਗਿਣਤੀ];
    // [size_idx][kda] where size_idx: 0=d16, 1=d32, 2=d64, 3=d128
    vk_pipeline pipeline_gated_delta_net[4][2];
    vk_pipeline pipeline_ssm_ਜਾਂਚ_f32_d128;
    vk_pipeline pipeline_ssm_ਜਾਂਚ_f32_d256;
    vk_pipeline pipeline_ssm_conv_f32;
    vk_pipeline pipeline_ssm_conv_silu_f32;
    vk_pipeline pipeline_ssm_conv_bias_silu_f32;
    vk_pipeline pipeline_opt_ਕਦਮ_adamw_f32;
    vk_pipeline pipeline_opt_ਕਦਮ_sgd_f32;
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f32[CONV_SHAPE_ਗਿਣਤੀ];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f16_f32[CONV_SHAPE_ਗਿਣਤੀ];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f32[CONV_SHAPE_ਗਿਣਤੀ];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f16_f32[CONV_SHAPE_ਗਿਣਤੀ];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f32[CONV_SHAPE_ਗਿਣਤੀ];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f16_f32[CONV_SHAPE_ਗਿਣਤੀ];
    vk_pipeline pipeline_conv2d_dw_whcn_f32, pipeline_conv2d_dw_whcn_f16_f32;
    vk_pipeline pipeline_conv2d_dw_cwhn_f32, pipeline_conv2d_dw_cwhn_f16_f32;

    std::map<vk_fa_pipeline_state, vk_pipeline> pipeline_flash_attn_f32_f16;

    std::map<std::pair<uint32_t, uint32_t>, vk_pipeline> pipeline_fa_ਨਕਾਬ_opt;

    vk_pipeline pipeline_fa_sparse_compact;
    vk_pipeline pipeline_fa_sparse_compact_subgroup;
    bool fa_sparse_compact_use_subgroups;

    vk_pipeline pipeline_flash_attn_ਵੰਡੋ_k_reduce;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, std::pair<vk_pipeline, vk_pipeline>> pipeline_xe_fa_decode_dual_phases;
    vk_pipeline pipeline_ਗਿਣਤੀ_experts;

    // [2] is for whether to take n_experts from spec constant (0) or push constant (1)
    vk_pipeline pipeline_topk_moe[ਅੰਕ_topk_moe_pipelines][2];

    std::ਸਦਿਸ਼<vk_pipeline_ref> all_pipelines;

    vk::Fence fence;
    vk_ਬਫਰ sync_staging;

    ggml_backend_ਬਫਰ_type ਬਫਰ_type;

    bool disable_fusion;
    bool disable_descriptor_reuse;
    std::atomic<uint64_t> ਬਫਰ_destroy_ਗਿਣਤੀ {};
    bool disable_host_visible_vidmem;
    bool allow_sysmem_fallback;
    bool disable_ਗ੍ਰਾਫ_optimize;

    std::unique_ptr<vk_ਸਮ੍ਰਿਤੀ_logger> ਸਮ੍ਰਿਤੀ_logger;

    ~vk_device_struct();

};

ਸਤਰਅੰਦਰ ਖਾਲੀ vk_command_pool::init(vk_device& device, vk_ਕਤਾਰ *q_) {
    cmd_buffers.clear();
    q = q_;

    vk::CommandPoolCreateInfo command_pool_create_info(
        vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_ਦੋਆਧਾਰੀ_ਅੰਕ | VK_COMMAND_POOL_CREATE_RESET_COMMAND_ਬਫਰ_ਦੋਆਧਾਰੀ_ਅੰਕ),
        q->ਕਤਾਰ_family_ਸੂਚਕਾਂਕ);
    pool = device->device.createCommandPool(command_pool_create_info);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ vk_command_pool::destroy(vk::Device& device) {
    device.destroyCommandPool(pool);
    pool = nullptr;
    cmd_buffers.clear();
}

ਬਣਤਰ vk_ਬਫਰ_struct {
    vk::ਬਫਰ_2 ਬਫਰ = VK_NULL_HANDLE;
    vk::Device_ਸਮ੍ਰਿਤੀ device_ਸਮ੍ਰਿਤੀ = VK_NULL_HANDLE;
    vk::ਸਮ੍ਰਿਤੀ_Property_Flags ਸਮ੍ਰਿਤੀ_property_flags;
    ਖਾਲੀ * ptr;
    size_t ਮਾਪ = 0;
    vk::Device_ਪਤਾ bda_addr {};

    vk_device device;

    ~vk_ਬਫਰ_struct() {
        ਜੇ (ਮਾਪ == 0) {
            ਵਾਪਸ;
        }
        VK_LOG_DEBUG("~vk_buffer_struct(" << ਬਫਰ << ", " << ਮਾਪ << ")");

        // bump before destroying, so a thread that sees the buffer gone also sees the new count
        device->ਬਫਰ_destroy_ਗਿਣਤੀ.fetch_ਜੋੜੋ(1, std::memory_order_release);
        device->device.free_ਸਮ੍ਰਿਤੀ_2(device_ਸਮ੍ਰਿਤੀ);
        device->device.destroy_ਬਫਰ(ਬਫਰ);
    }
};

ਬਣਤਰ vk_subbuffer {
    vk_ਬਫਰ ਬਫਰ;
    uint64_t ਵਿੱਥ;
    uint64_t ਮਾਪ;

    ਸੰਚਾਲਕ vk::Descriptor_ਬਫਰ_Info() ਅਚਲ {
        ਵਾਪਸ { ਬਫਰ->ਬਫਰ, ਵਿੱਥ, ਮਾਪ };
    }
};

ਬਣਤਰ vk_semaphore {
    vk::Semaphore s;
    uint64_t ਮੁੱਲ;
};

ਬਣਤਰ vk_event {
    std::ਸਦਿਸ਼<vk::Event> events_free; // Events available for reuse
    std::ਸਦਿਸ਼<vk::Event> events_submitted; // Events that are fully submitted and can be reused on next synchronize
    vk::Event event;
    bool has_event;

    vk_semaphore tl_semaphore;
    vk_command_ਬਫਰ* cmd_ਬਫਰ = nullptr;
    uint64_t cmd_ਬਫਰ_use_counter = 0;
};

ਬਣਤਰ vk_submission {
    vk_command_ਬਫਰ* ਬਫਰ = nullptr;
    std::ਸਦਿਸ਼<vk_semaphore> wait_semaphores;
    std::ਸਦਿਸ਼<vk_semaphore> signal_semaphores;
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::ਸਦਿਸ਼<vk_submission> vk_sequence;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAT_VEC_FUSION_FLAGS_BIAS0 0x1

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAT_VEC_FUSION_FLAGS_BIAS1 0x2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAT_VEC_FUSION_FLAGS_SCALE0 0x4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAT_VEC_FUSION_FLAGS_SCALE1 0x8

ਬਣਤਰ vk_staging_memcpy {
    vk_staging_memcpy(ਖਾਲੀ * _dst, ਅਚਲ ਖਾਲੀ * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    ਖਾਲੀ * dst;
    ਅਚਲ ਖਾਲੀ * src;
    size_t n;
};

ਬਣਤਰ vk_staging_memset {
    vk_staging_memset(ਖਾਲੀ * _dst, uint32_t _val, size_t _n) : dst(_dst), val(_val), n(_n) {}

    ਖਾਲੀ * dst;
    uint32_t val;
    size_t n;
};

ਬਣਤਰ vk_context_struct {
    vk_submission * s;
    std::ਸਦਿਸ਼<vk_sequence> seqs;

    ਪੂਰਨ exit_tensor_idx;

    std::ਸਦਿਸ਼<vk_staging_memcpy> ਵਿੱਚ_memcpys;
    std::ਸਦਿਸ਼<vk_staging_memcpy> ਬਾਹਰਲਾ_memcpys;
    std::ਸਦਿਸ਼<vk_staging_memset> memsets;

    std::ਸਦਿਸ਼<std::string> debug_labels;

    vk_command_pool * p {};
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::shared_ptr<vk_context_struct> vk_context;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::weak_ptr<vk_context_struct> vk_context_ref;

ਬਣਤਰ ggml_vk_garbage_collector {
    std::ਸਦਿਸ਼<vk_semaphore> tl_semaphores;
    std::ਸਦਿਸ਼<vk_semaphore> semaphores;
    std::ਸਦਿਸ਼<vk::Event> events;
    std::ਸਦਿਸ਼<vk_context> contexts;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VK_LOG_ਸਮ੍ਰਿਤੀ(ਸੁਨੇਹਾ_2) ਜੇ (vk_ਸਮ੍ਰਿਤੀ_logger_enabled) { std::cerr << "ggml_vulkan memory: " << ਸੁਨੇਹਾ_2 << std::endl; }

ਸਥਿਰ std::string format_ਮਾਪ(size_t ਮਾਪ) {
    ਅਚਲ size_t kib = 1024;
    ਅਚਲ size_t mib = kib * 1024;
    ਅਚਲ size_t gib = mib * 1024;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    ਜੇ (ਮਾਪ >= gib) {
        oss << static_cast<ਦੁੱਗਣਾ>(ਮਾਪ) / gib << " GiB";
    } ਨਹੀਂਤਾਂ ਜੇ (ਮਾਪ >= mib) {
        oss << static_cast<ਦੁੱਗਣਾ>(ਮਾਪ) / mib << " MiB";
    } ਨਹੀਂਤਾਂ ਜੇ (ਮਾਪ >= kib) {
        oss << static_cast<ਦੁੱਗਣਾ>(ਮਾਪ) / kib << " KiB";
    } ਨਹੀਂਤਾਂ {
        oss << ਮਾਪ << " B";
    }

    ਵਾਪਸ oss.str();
}

class vk_ਸਮ੍ਰਿਤੀ_logger {
ਜਨਤਕ:
    vk_ਸਮ੍ਰਿਤੀ_logger(): ਕੁੱਲ_device(0), ਕੁੱਲ_host(0) {}
    ਖਾਲੀ log_ਵੰਡ(vk_ਬਫਰ_ref buf_ref, size_t ਮਾਪ);
    ਖਾਲੀ log_deallocation(vk_ਬਫਰ_ref buf_ref);

ਨਿਜੀ:
    std::map<vk::ਬਫਰ_2, size_t> allocations; // Track allocations
    size_t ਕੁੱਲ_device;
    size_t ਕੁੱਲ_host;
    ਸਥਿਰ std::mutex log_mutex;
};

ਸਤਰਅੰਦਰ std::mutex vk_ਸਮ੍ਰਿਤੀ_logger::log_mutex;

class vk_perf_logger {
  ਜਨਤਕ:
    ਖਾਲੀ print_timings(bool force = false);


    std::string get_ਗੰਢ_fusion_name(ਅਚਲ ggml_tensor * ਗੰਢ, ਅਚਲ ਅੱਖਰ *fusion_name, uint64_t *n_flops);


    ਖਾਲੀ log_timing(ਅਚਲ ggml_tensor * ਗੰਢ, ਅਚਲ ਅੱਖਰ *fusion_name, uint64_t ਸਮਾਂ) {
        uint64_t n_flops;
        std::string name = get_ਗੰਢ_fusion_name(ਗੰਢ, fusion_name, &n_flops);
        ਜੇ (n_flops) {
            flops[name].ਧੱਕੋ_ਪਿੱਠ(n_flops);
        }
        timings[name].ਧੱਕੋ_ਪਿੱਠ(ਸਮਾਂ);
    }

    ਖਾਲੀ log_timing(ਅਚਲ std::ਸਦਿਸ਼<ggml_tensor *> &nodes, ਅਚਲ std::ਸਦਿਸ਼<ਅਚਲ ਅੱਖਰ *> &ਨਾਂ, uint64_t ਸਮਾਂ) {
        uint64_t ਕੁੱਲ_flops = 0;
        std::string name;
        ਲਈ (size_t n = 0; n < nodes.ਮਾਪ(); ++n) {
            uint64_t n_flops = 0;
            name += get_ਗੰਢ_fusion_name(nodes[n], ਨਾਂ[n], &n_flops);
            ਕੁੱਲ_flops += n_flops;

            ਜੇ (n != nodes.ਮਾਪ() - 1) {
                name += ", ";
            }
        }
        ਜੇ (ਕੁੱਲ_flops) {
            flops[name].ਧੱਕੋ_ਪਿੱਠ(ਕੁੱਲ_flops);
        }
        timings[name].ਧੱਕੋ_ਪਿੱਠ(ਸਮਾਂ);
    }

  ਨਿਜੀ:
    std::map<std::string, std::ਸਦਿਸ਼<uint64_t>> timings;
    std::map<std::string, std::ਸਦਿਸ਼<uint64_t>> flops;
    uint32_t print_ਗਿਣਤੀ {};
};

ਬਣਤਰ ggml_backend_vk_context {
    std::string name;

    vk_device device;

    size_t semaphore_idx, event_idx;
    ggml_vk_garbage_collector gc;
    size_t prealloc_ਮਾਪ_x, prealloc_ਮਾਪ_y, prealloc_ਮਾਪ_ਵੰਡੋ_k, prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials, prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials_ਵਿੱਥ;
    vk_ਬਫਰ prealloc_x, prealloc_y, prealloc_ਵੰਡੋ_k, prealloc_ਜੋੜੋ_rms_partials, sync_staging;
    vk::Fence fence, almost_ready_fence;
    bool submit_pending {};
    bool almost_ready_fence_pending {};
    // Set before op_add and unset after op_rms_norm to indicate that the add should
    // write partial sums to accumulate the square of the vector components
    bool do_ਜੋੜੋ_rms_partials_ਵਿੱਥ_calculation;
    bool do_ਜੋੜੋ_rms_partials;

    uint64_t ਆਖਰੀ_ਕੁੱਲ_flops {UINT64_MAX};

    // Cache most recent tensor that was converted into prealloc_y, and what pipeline it used to convert.
    vk_pipeline_struct * prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ {};
    ਅਚਲ ggml_tensor * prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ {};
    // True when the K dimension in prealloc_y is padded.
    bool prealloc_y_ਆਖਰੀ_k_padded {};

    // Track which nodes have been used since the last sync, and whether they were written to
    std::ਸਦਿਸ਼<ਅਚਲ ggml_tensor *> unsynced_nodes_written;
    std::ਸਦਿਸ਼<ਅਚਲ ggml_tensor *> unsynced_nodes_read;
    // Track which prealloc buffers have pending reads that need to be synchronized.
    // These are checked before writing to the buffer (and call ggml_vk_sync_buffers if set),
    // and set to true after the buffer contents are consumed.
    bool prealloc_x_need_sync, prealloc_y_need_sync, prealloc_ਵੰਡੋ_k_need_sync;

    vk_context_ref compute_ctx;

    vk_context_ref transfer_ctx;
    vk_semaphore transfer_semaphore;
    uint64_t transfer_semaphore_ਆਖਰੀ_submitted {};

    std::ਸਦਿਸ਼<vk_context_ref> tensor_ctxs;

    std::ਸਦਿਸ਼<vk::DescriptorPool> descriptor_pools;
    std::ਸਦਿਸ਼<vk::Descriptor_ਸੈੱਟ> descriptor_sets;
    // last bindings written to each set; descriptor_sets is append-only so an index always names the same set
    std::ਸਦਿਸ਼<std::ਸਦਿਸ਼<vk::Descriptor_ਬਫਰ_Info>> descriptor_ਸੈੱਟ_bindings;
    uint64_t descriptor_ਸੈੱਟ_bindings_destroy_ਗਿਣਤੀ {};
    uint32_t descriptor_ਸੈੱਟ_idx {};
    uint32_t pipeline_descriptor_ਸੈੱਟ_requirements {};

    vk_command_pool compute_cmd_pool;
    vk_command_pool transfer_cmd_pool;

    // number of additional consecutive nodes that are being fused with the
    // node currently being processed
    ਪੂਰਨ ਅੰਕ_additional_fused_ops {};
    // Bitmask of which fused ops need to write an intermediate value to memory.
    // Bit 'i' means nodes[start_of_fusion + i] writes to memory.
    // If there's no fusion, bit 0 is still set.
    ਪੂਰਨ fused_ops_write_ਨਕਾਬ {};
    topk_moe_ਬਹੁਲਕ fused_topk_moe_ਬਹੁਲਕ {};
    bool fused_topk_moe_scale {};
    // QSA indexer gather+add+top_k fused into one radix-select
    bool fused_topk_qsa {};
    bool fused_hc_post_gate {};
    rms_norm_ਬਹੁਲਕ fused_rms_norm_ਬਹੁਲਕ {RMS_NORM_ਗਿਣਤੀ};

    // for GGML_VK_PERF_LOGGER
    std::unique_ptr<vk_perf_logger> perf_logger;
    vk::QueryPool query_pool;
    std::ਸਦਿਸ਼<ਅਚਲ ਅੱਖਰ *> query_fusion_ਨਾਂ;
    std::ਸਦਿਸ਼<ਪੂਰਨ> query_fusion_ਗੰਢ_ਗਿਣਤੀ;
    std::ਸਦਿਸ਼<ggml_tensor *> query_nodes;
    std::ਸਦਿਸ਼<ਪੂਰਨ> query_ਗੰਢ_idx;
    int32_t ਅੰਕ_queries {};
    int32_t query_idx {};
};

ਬਣਤਰ ggml_backend_vk_ਬਫਰ_context {
    vk_device_ref device;
    vk_ਬਫਰ dev_ਬਫਰ;
    std::string name;

    ggml_backend_vk_ਬਫਰ_context(vk_device_ref device, vk_ਬਫਰ&& dev_ਬਫਰ, std::string& name) :
        device(device),
        dev_ਬਫਰ(dev_ਬਫਰ),
        name(name) {
    }

    ~ggml_backend_vk_ਬਫਰ_context();

};

ਬਣਤਰ vk_host_ਵੰਡ {
    ਖਾਲੀ *    ਅਧਾਰ = nullptr;
    size_t    ਮਾਪ = 0;
    vk_ਬਫਰ views[GGML_VK_MAX_DEVICES];  // indexed by device->idx; null = not (yet) mapped here
};

ਬਣਤਰ vk_instance_t {
    vk::Instance instance;

    bool debug_utils_support = false;  // VK_EXT_debug_utils enabled
    PFN_vk_ਸੈੱਟ_Debug_Utils_Object_Name_EXT pfn_vk_ਸੈੱਟ_Debug_Utils_Object_Name_EXT = {};
    PFN_vk_ਕਤਾਰ_Begin_Debug_Utils_Label_EXT pfn_vk_ਕਤਾਰ_Begin_Debug_Utils_Label_EXT = {};
    PFN_vk_ਕਤਾਰ_ਸਿਰਾ_Debug_Utils_Label_EXT   pfn_vk_ਕਤਾਰ_ਸਿਰਾ_Debug_Utils_Label_EXT   = {};
    PFN_vkCmdBeginDebugUtilsLabelEXT   pfn_vkCmdBeginDebugUtilsLabelEXT   = {};
    PFN_vk_Cmd_ਸਿਰਾ_Debug_Utils_Label_EXT pfn_vk_Cmd_ਸਿਰਾ_Debug_Utils_Label_EXT = {};
    PFN_vk_Cmd_ਪਾਓ_Debug_Utils_Label_EXT  pfn_vk_Cmd_ਪਾਓ_Debug_Utils_Label_EXT  = {};

    std::ਸਦਿਸ਼<size_t> device_indices;
    std::ਸਦਿਸ਼<bool>   device_supports_membudget;
    std::ਸਦਿਸ਼<bool>   device_host_ਬਫਰ_import;
    vk_device devices[GGML_VK_MAX_DEVICES];

    std::ਸਦਿਸ਼<vk_host_ਵੰਡ> host_allocations;
    std::shared_mutex               host_allocations_mutex;

    size_t host_import_alignment = 0;
    uint32_t host_import_failed = 0;  // bitmask by device->idx, guarded by host_allocations_mutex

    bool multi_device = false;
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*ggml_vk_func_t)(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);

ਸਥਿਰ constexpr uint32_t k_Spv_Op_Cooperative_ਮੈਟ੍ਰਿਕਸ_ਲੱਦੋ_Tensor_NV = 5367;

ਸਥਿਰ constexpr uint32_t k_Spv_Capability_Cooperative_ਮੈਟ੍ਰਿਕਸ_Decode_ਸਦਿਸ਼_NV = 5447;

ਸਥਿਰ constexpr uint32_t k_Spv_Tensor_Addressing_Decode_ਸਦਿਸ਼_Func_ਦੋਆਧਾਰੀ_ਅੰਕ = 0x4;

ਬਣਤਰ vk_fa_tuning_params {
    Fa_ਕੋਡ_ਰਸਤਾ ਰਸਤਾ;
    uint32_t workgroup_ਮਾਪ;
    uint32_t subgroup_ਮਾਪ;
    uint32_t ਖੰਡ_ਪੰਕਤੀਆਂ;
    uint32_t ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ;
    uint32_t d_ਵੰਡੋ;
    uint32_t ਪੰਕਤੀ_ਵੰਡੋ;
    bool shmem_staging;
    bool disable_subgroups;
    uint32_t ਹੱਦਬੰਦੀ_occupancy_shmem;

    ਖਾਲੀ print() ਅਚਲ {
        std::cerr << "path=" << ਰਸਤਾ << " workgroup_size=" << workgroup_ਮਾਪ << " subgroup_size=" << subgroup_ਮਾਪ <<
                     " block_rows=" << ਖੰਡ_ਪੰਕਤੀਆਂ << " block_cols=" << ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ << " d_split=" << d_ਵੰਡੋ <<
                     " row_split=" << ਪੰਕਤੀ_ਵੰਡੋ << " shmem_staging=" << shmem_staging << " disable_subgroups=" << disable_subgroups <<
                     " limit_occupancy_shmem=" << ਹੱਦਬੰਦੀ_occupancy_shmem << std::endl;
    }
};

ਬਣਤਰ GpuPipelineConfig {
    // GPU architecture identifier.
    // Example: vk_device_architecture::AMD_GCN
    vk_device_architecture arch;

    // Mapping of pipeline names to their specific subgroup sizes.
    // Example: {"soft_max_f32", 64}
    std::unordered_map<std::string, uint32_t> pipelines;

    // Default subgroup size for this GPU.
    // Defaults to 0 if not explicitly provided.
    uint32_t default_subgroup_ਮਾਪ = 0;
};

ਸਥਿਰ constexpr uint32_t RDNA_DEFAULT_SUBGROUP_ਮਾਪ = 32;

ਬਣਤਰ CompileTask {
    vk_pipeline pipeline;
    size_t spv_ਮਾਪ;
    ਅਚਲ ਖਾਲੀ * spv_ਅੰਕੜੇ;
    std::string entrypoint;
    uint32_t parameter_ਗਿਣਤੀ;
    std::array<uint32_t, 3> wg_denoms;
    std::ਸਦਿਸ਼<uint32_t> specialization_constants;
    bool disable_robustness;
    bool require_ਭਰਿਆ_subgroups;
    uint32_t required_subgroup_ਮਾਪ;
};

ਬਣਤਰ ggml_vk_debug_label {
    // at most one of these is set, depending on the scope the label was opened in
    vk_context_struct * subctx {};
    vk_ਕਤਾਰ_handle *   qhandle {};

    // one region per dispatch, e.g. "matmul_q4_k_f32_f16acc_aligned_m (192,8,1)".
    // RGP cannot recover the pipeline name on its own, it only has the hash
    ggml_vk_debug_label(vk_context & ctx, ਅਚਲ std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2);


    // one region per graph node
    // fused nodes are joined with '+', e.g. "RMS_NORM+MUL+ROPE Qcur-19"
    ggml_vk_debug_label(vk_context & ctx, ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ਪੂਰਨ n_fused);


    // one region per graph evaluation, opened on the queue instead of a command buffer
    // so it spans every submit the evaluation makes
    ggml_vk_debug_label(vk_ਕਤਾਰ_handle * handle, ਅਚਲ ਅੱਖਰ * name);


    // call before the command buffer can end, the destructor covers the rest
    ਖਾਲੀ close();


    ~ggml_vk_debug_label() {
        close();
    }

    ggml_vk_debug_label(ਅਚਲ ggml_vk_debug_label &) = delete;
    ggml_vk_debug_label & ਸੰਚਾਲਕ=(ਅਚਲ ggml_vk_debug_label &) = delete;

ਨਿਜੀ:
    // the constructors check this too, so the name is not built when markers are off
    ਖਾਲੀ begin(vk_context & ctx, ਅਚਲ std::string & name);

};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅਣਵਰਤਿਆ GGML_ਅਣਵਰਤਿਆ

ਬਣਤਰ ggml_backend_vk_device_context {
    size_t device;
    std::string name;
    std::string description;
    bool is_integrated_gpu;
    std::string pci_bus_id;
    ਪੂਰਨ op_offload_min_batch_ਮਾਪ;
};
