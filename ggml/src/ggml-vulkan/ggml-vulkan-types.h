#సూచన once

#అన్తర్భావ "ggml-vulkan.h"

#అన్తర్భావ <vulkan/vulkan_core.h>

#ఒకవేళ defined(GGML_VULKAN_నిరంతరం_TESTS) || defined(GGML_VULKAN_CHECK_RESULTS)
#అన్తర్భావ <chrono>
#అన్తర్భావ "ggml-cpu.h"
#అయితే_ముగింపు

#నిర్వచించు VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#ఒకవేళ VK_శీర్షిక_సంస్కరణ >= 301
namespace vk::detail { class DispatchLoaderDynamic; }
using vk::detail::DispatchLoaderDynamic;
#లేకపోతే
namespace vk { class DispatchLoaderDynamic; }
using vk::DispatchLoaderDynamic;
#అయితే_ముగింపు

DispatchLoaderDynamic & ggml_vk_default_dispatcher();

#నిర్వచించు VULKAN_HPP_DEFAULT_DISPATCHER ggml_vk_default_dispatcher()

#అన్తర్భావ <vulkan/vulkan.hpp>

#నిర్వచితం_కాకపోతే VK_NV_cooperative_మాతృక_decode_సదిశ
#నిర్వచించు VK_NV_cooperative_మాతృక_decode_సదిశ 1
#నిర్వచించు VK_NV_COOPERATIVE_మాతృక_DECODE_సదిశ_EXTENSION_NAME "VK_NV_cooperative_matrix_decode_vector"
#నిర్వచించు VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_మాతృక_DECODE_సదిశ_FEATURES_NV ((VkStructureType)1000689000)
రకనిర్వచనం నిర్మాణం Vk_Physical_Device_Cooperative_మాతృక_Decode_సదిశ_Features_NV {
    VkStructureType    sType;
    శూన్య*              p_తదుపరి;
    VkBool32           cooperative_మాతృక_Decode_సదిశ;
} Vk_Physical_Device_Cooperative_మాతృక_Decode_సదిశ_Features_NV;
#అయితే_ముగింపు

#ఒకవేళ __has_include(<spirv/unified1/spirv.hpp>)
#    include <spirv/unified1/spirv.hpp>
#లేదావేళ __has_include(<spirv-శీర్షికలు/spirv.hpp>)
#    include <spirv-headers/spirv.hpp>
#లేదావేళ __has_include(<spirv.hpp>)
#    include <spirv.hpp>
#లేకపోతే
     // Fallback to let the compiler throw a standard "file not found" error
#    include <spirv/unified1/spirv.hpp>
#అయితే_ముగింపు

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cmath>
#అన్తర్భావ <functional>
#అన్తర్భావ <iomanip>
#అన్తర్భావ <iostream>
#అన్తర్భావ <tuple>
#అన్తర్భావ <vector>
#అన్తర్భావ <deque>
#అన్తర్భావ <sstream>
#అన్తర్భావ <utility>
#అన్తర్భావ <memory>
#అన్తర్భావ <limits>
#అన్తర్భావ <map>
#అన్తర్భావ <set>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <shared_mutex>
#అన్తర్భావ <mutex>
#అన్తర్భావ <future>
#అన్తర్భావ <condition_variable>
#అన్తర్భావ <thread>

#ఒకవేళ defined(_MSC_VER)
# define NOMINMAX 1
# include <windows.h>
# define YIELD() YieldProcessor()
#లేదావేళ defined(__clang__) || defined(__GNUC__)
# ఒకవేళ defined(__x86_64__) ||defined(__i386__)
#  include <immintrin.h>
#  define YIELD() _mm_pause()
# elif defined(__arm__) || defined(__aarch64__)
#  ఒకవేళ defined(__clang__)
#   include <arm_acle.h>
#   define YIELD() __yield()
#  లేకపోతే
#   define YIELD() asm అస్థిర("yield")
#  endif
# endif
#అయితే_ముగింపు

#ఒకవేళ !defined(YIELD)
#నిర్వచించు YIELD()
#అయితే_ముగింపు

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-vulkan-shaders.hpp"

#ఒకవేళ !defined(VK_KHR_shader_bfloat16)

#నిర్వచించు VK_KHR_shader_bfloat16 1
#నిర్వచించు VK_KHR_SHADER_BFLOAT16_SPEC_సంస్కరణ                          1
#నిర్వచించు VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME                        "VK_KHR_shader_bfloat16"
#నిర్వచించు VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR ((VkStructureType)1000141000)
#నిర్వచించు VK_COMPONENT_TYPE_BFLOAT16_KHR                               ((VkComponentTypeKHR)1000141000)

రకనిర్వచనం నిర్మాణం VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType                       sType;
    శూన్య*                                 p_తదుపరి;
    VkBool32                              shaderBFloat16Type;
    VkBool32                              shader_BFloat16_Dot_లబ్ధం;
    VkBool32                              shader_BFloat16_Cooperative_మాతృక;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;
#అయితే_ముగింపు

#ఒకవేళ !defined(VK_VALVE_shader_mixed_float_dot_లబ్ధం)
#నిర్వచించు VK_VALVE_shader_mixed_float_dot_లబ్ధం 1
#నిర్వచించు VK_VALVE_SHADER_MIXED_FLOAT_DOT_లబ్ధం_SPEC_సంస్కరణ 1
#నిర్వచించు VK_VALVE_SHADER_MIXED_FLOAT_DOT_లబ్ధం_EXTENSION_NAME "VK_VALVE_shader_mixed_float_dot_product"
#నిర్వచించు VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_లబ్ధం_FEATURES_VALVE ((VkStructureType)1000673000)
రకనిర్వచనం నిర్మాణం Vk_Physical_Device_Shader_Mixed_Float_Dot_లబ్ధం_Features_VALVE {
    VkStructureType    sType;
    శూన్య*              p_తదుపరి;
    VkBool32           shader_Mixed_Float_Dot_లబ్ధం_Float16_Acc_Float32;
    VkBool32           shader_Mixed_Float_Dot_లబ్ధం_Float16_Acc_Float16;
    VkBool32           shader_Mixed_Float_Dot_లబ్ధం_BFloat16_Acc;
    VkBool32           shader_Mixed_Float_Dot_లబ్ధం_Float8_Acc_Float32;
} Vk_Physical_Device_Shader_Mixed_Float_Dot_లబ్ధం_Features_VALVE;
#అయితే_ముగింపు

#ఒకవేళ !defined(VK_EXT_shader_ocp_microscaling_types)
#నిర్వచించు VK_EXT_shader_ocp_microscaling_types 1
#నిర్వచించు VK_EXT_SHADER_OCP_MICROSCALING_TYPES_SPEC_సంస్కరణ 1
#నిర్వచించు VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME "VK_EXT_shader_ocp_microscaling_types"
#నిర్వచించు VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT ((VkStructureType)1000672000)
రకనిర్వచనం నిర్మాణం VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT {
    VkStructureType    sType;
    శూన్య*              p_తదుపరి;
    VkBool32           shaderFloat4;
    VkBool32           shaderFloat6;
    VkBool32           shaderFloat8UnsignedE8M0;
    VkBool32           shaderMXInt8;
} VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT;
#అయితే_ముగింపు

#ఒకవేళ !defined(VK_EXT_shader_float8)
#నిర్వచించు VK_EXT_shader_float8 1
#నిర్వచించు VK_EXT_SHADER_FLOAT8_SPEC_సంస్కరణ 1
#నిర్వచించు VK_EXT_SHADER_FLOAT8_EXTENSION_NAME "VK_EXT_shader_float8"
#నిర్వచించు VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT ((VkStructureType)1000567000)
రకనిర్వచనం నిర్మాణం VkPhysicalDeviceShaderFloat8FeaturesEXT {
    VkStructureType    sType;
    శూన్య*              p_తదుపరి;
    VkBool32           shaderFloat8;
    VkBool32           shader_Float8_Cooperative_మాతృక;
} VkPhysicalDeviceShaderFloat8FeaturesEXT;
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME
#నిర్వచించు VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME "VK_KHR_internally_synchronized_queues"
#నిర్వచించు VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR ((VkStructureType)1000504000)
#నిర్వచించు VK_DEVICE_వరుస_CREATE_INTERNALLY_SYNCHRONIZED_బిట్టు_KHR ((Vk_Device_వరుస_Create_Flag_బిట్లు)0x00000004)

// Compile-time constant guaranteed; no runtime initialization overhead
స్థిర constexpr vk::Device_వరుస_Create_Flag_బిట్లు eInternallySynchronizedKHR =
    static_cast<vk::Device_వరుస_Create_Flag_బిట్లు>(0x00000004);

రకనిర్వచనం నిర్మాణం VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR {
    VkStructureType    sType;
    శూన్య* p_తదుపరి;
    VkBool32           internallySynchronizedQueues;
} VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR;
#లేకపోతే
స్థిర constexpr vk::Device_వరుస_Create_Flag_బిట్లు eInternallySynchronizedKHR = vk::Device_వరుస_Create_Flag_బిట్లు::eInternallySynchronizedKHR;
#అయితే_ముగింపు

#నిర్వచించు ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))

#నిర్వచించు CEIL_DIV(M, N) (((M) / (N)) + (((M) % (N)) != 0))

స్థిర bool is_pow2(uint32_t x) { తిరిగి x > 1 && (x & (x-1)) == 0; }

#నిర్వచించు VK_VENDOR_ID_AMD 0x1002

#నిర్వచించు VK_VENDOR_ID_APPLE 0x106b

#నిర్వచించు VK_VENDOR_ID_INTEL 0x8086

#నిర్వచించు VK_VENDOR_ID_NVIDIA 0x10de

#నిర్వచించు VK_VENDOR_ID_QUALCOMM 0x5143

#నిర్వచించు VK_VENDOR_ID_SAMSUNG 0x144d

#నిర్వచించు VK_DEVICE_DESCRIPTOR_POOL_కొలత 256

#నిర్వచించు VK_CHECK(err, సందేశం_2, dev)                                     \
    చేయి {                                                            \
        vk::ఫలితం_2 err_;                                            \
        try {                                                       \
            err_ = (err);                                           \
        } catch (vk::Device_కోల్పోయిన_Error &) {                           \
            ggml_vk_print_device_కోల్పోయిన_info(dev);                    \
            GGML_LOG_ERROR("ggml_vulkan: %s at %s:%d\n",            \
                #err, __FILE__, __LINE__);                          \
            throw;                                                  \
        }                                                           \
        ఒకవేళ (err_ != vk::ఫలితం_2::eSuccess) {                         \
            GGML_LOG_ERROR("ggml_vulkan: %s error %s at %s:%d\n",   \
                #err, గమ్యం_string(err_).c_str(), __FILE__, __LINE__); \
            throw vk::SystemError(vk::make_error_సంకేతం(err_),        \
                "ggml_vulkan: " సందేశం_2);                               \
        }                                                           \
    } అయితే (0)

#నిర్వచితమైతే GGML_VULKAN_DEBUG
#నిర్వచించు VK_LOG_DEBUG(సందేశం_2) std::cerr << సందేశం_2 << std::endl
#లేకపోతే
#నిర్వచించు VK_LOG_DEBUG(సందేశం_2) ((శూన్య) 0)
#అయితే_ముగింపు // GGML_VULKAN_DEBUG

#నిర్వచించు MAX_PARAMETER_లెక్క 12

#నిర్వచించు MAX_FUSED_ADDS (MAX_PARAMETER_లెక్క - 3)

నిర్మాణం vk_pipeline_struct;

రకనిర్వచనం std::shared_ptr<నిర్మాణం vk_pipeline_struct> vk_pipeline;

నిర్మాణం vk_pipeline_struct {
    std::string name;
    vk::ShaderModule shader_module;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t నెట్టు_constant_కొలత;
    uint32_t parameter_లెక్క;
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
    uint32_t రిజిస్టరు_లెక్క {};

#ఒకవేళ defined(VK_EXT_shader_64bit_indexing)
    bool is_64b_indexing {};
#అయితే_ముగింపు
    // linked list of pipelines for multiple compilation variants.
    // currently only used to compile a 64-bit indexing variant.
    vk_pipeline తదుపరి;
};

రకనిర్వచనం std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;

నిర్మాణం vk_matmul_pipeline_కీలకం {
    ggml_type type_a;
    ggml_type type_b;
    bool mul_mat_id;
    bool f16acc;

    bool క్రియాకారకం<(స్థిరరాశి vk_matmul_pipeline_కీలకం & o) స్థిరరాశి {
        తిరిగి std::tie(type_a, type_b, mul_mat_id, f16acc)
             < std::tie(o.type_a, o.type_b, o.mul_mat_id, o.f16acc);
    }
};

నిర్మాణం vk_matmul_pipeline_pair {
    vk_pipeline unaligned;
    vk_pipeline aligned;
    uint32_t align;
};

నిర్మాణం vk_tile_config {
    std::సదిశ<uint32_t> warptile;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
};

using matmul_tile_selector_t = std::function<uint32_t(
    uint32_t m, uint32_t n, uint32_t k, uint32_t shader_core_లెక్క,
    స్థిరరాశి std::సదిశ<vk_matmul_pipeline_pair>& configs)>;

నిర్మాణం vk_device_struct;

రకనిర్వచనం std::shared_ptr<vk_device_struct> vk_device;

రకనిర్వచనం std::weak_ptr<vk_device_struct> vk_device_ref;

నిర్మాణం vk_బఫరు_struct;

రకనిర్వచనం std::shared_ptr<vk_బఫరు_struct> vk_బఫరు;

రకనిర్వచనం std::weak_ptr<vk_బఫరు_struct> vk_బఫరు_ref;

నిర్మాణం ggml_backend_vk_బఫరు_type_context {
    std::string name;
    vk_device device;
};

నిర్మాణం vk_command_బఫరు {
    vk::Command_బఫరు buf;
    uint64_t use_counter = 0;
    bool లోపల_use = false;
};

నిర్మాణం vk_వరుస;

నిర్మాణం vk_command_pool {
    శూన్య init(vk_device& device, vk_వరుస *q_);
    శూన్య destroy(vk::Device& device);

    vk::CommandPool pool;
    // Using deque so the pointers to command buffers
    // remain valid even if we add more
    std::deque<vk_command_బఫరు> cmd_buffers;

    vk_వరుస *q;

    size_t buffers_లోపల_use() స్థిరరాశి {
        తిరిగి std::లెక్క_if(cmd_buffers.begin(), cmd_buffers.చివర(),
            [](స్థిరరాశి స్వయంచాలక& cb) { తిరిగి cb.లోపల_use; });
    }
};

నిర్మాణం vk_వరుస_handle {
    vk::వరుస_2 వరుస;
    vk_device_ref device;
    std::mutex * device_submit_mutex = nullptr;
    మాయా శూన్య submit(vk::ArrayProxy<స్థిరరాశి vk::SubmitInfo> submits, vk::Fence fence) = 0;
    మాయా శూన్య తాళం()   {}   // no-op by default (internally synchronized case)
    మాయా శూన్య unlock() {}
    మాయా ~vk_వరుస_handle() = అప్రమేయ;
};

నిర్మాణం vk_వరుస_handle_synchronized : vk_వరుస_handle {
    std::mutex mutex;
    శూన్య submit(vk::ArrayProxy<స్థిరరాశి vk::SubmitInfo> submits, vk::Fence fence) override;

    శూన్య తాళం()   override { mutex.తాళం(); }
    శూన్య unlock() override { mutex.unlock(); }
};

నిర్మాణం vk_వరుస_handle_unsynchronized : vk_వరుస_handle {
    శూన్య submit(vk::ArrayProxy<స్థిరరాశి vk::SubmitInfo> submits, vk::Fence fence) override;

    // lock()/unlock() inherited no-ops
};

నిర్మాణం vk_వరుస {
    uint32_t వరుస_family_సూచీ;
    std::shared_ptr<vk_వరుస_handle> handle;

    vk_command_pool cmd_pool;

    vk::PipelineStageFlags stage_flags;

    bool transfer_only;
};

స్థిర constexpr uint32_t mul_mat_vec_max_నిలువుల_సంఖ్య = 8;

స్థిర constexpr uint32_t p021_max_gqa_నిష్పత్తి = 8;

గణన vk_device_architecture {
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

గణన vk_conv_shapes {
    CONV_SHAPE_128x128,
    CONV_SHAPE_64x32,
    CONV_SHAPE_32x256,
    CONV_SHAPE_64x128,
    CONV_SHAPE_లెక్క,
};

నిర్మాణం vk_conv_ఖండం_కొలత {
    uint32_t K;
    uint32_t NPQ;
    uint32_t CRS;
};

అంతఃవరుస vk_conv_ఖండం_కొలత vk_conv_ఖండం_sizes[CONV_SHAPE_లెక్క] = {
    // K   NPQ  CRS
    { 128, 128, 16 }, // CONV_SHAPE_128x128
    {  64,  32, 32 }, // CONV_SHAPE_64x32
    {  32, 256, 16 }, // CONV_SHAPE_32x256
    {  64, 128, 16 }, // CONV_SHAPE_64x128
};

గణన dmmv_wg_sizes {
    DMMV_WG_కొలత_SUBGROUP,
    DMMV_WG_కొలత_LARGE,
    DMMV_WG_కొలత_లెక్క,
};

గణన Fa_సంకేతం_మార్గం {
    FA_SCALAR,
    FA_COOPMAT1,
    FA_COOPMAT2,
};

నిర్మాణం vk_fa_pipeline_state {
    uint32_t HSK, HSV;
    uint32_t Br, Bc;
    uint32_t D_విభజించు, అడ్డువరుస_విభజించు;
    bool shmem_staging;
    Fa_సంకేతం_మార్గం మార్గం;
    uint32_t workgroup_కొలత, subgroup_కొలత;
    bool aligned;
    bool f32acc;
    uint32_t flags;
    uint32_t హద్దు_occupancy_shmem;
    ggml_type k_type;
    ggml_type v_type;

    bool క్రియాకారకం<(స్థిరరాశి vk_fa_pipeline_state &b) స్థిరరాశి {
        తిరిగి std::tie(HSK, HSV, Br, Bc, D_విభజించు, అడ్డువరుస_విభజించు, shmem_staging, మార్గం, workgroup_కొలత, subgroup_కొలత, aligned, f32acc, flags, హద్దు_occupancy_shmem, k_type, v_type) <
               std::tie(b.HSK, b.HSV, b.Br, b.Bc, b.D_విభజించు, b.అడ్డువరుస_విభజించు, b.shmem_staging, b.మార్గం, b.workgroup_కొలత, b.subgroup_కొలత, b.aligned, b.f32acc, b.flags, b.హద్దు_occupancy_shmem, b.k_type, b.v_type);
    }
};

నిర్మాణం vk_conv2d_pipeline_state {
    vk_conv2d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t p0, uint32_t p1, uint32_t d0, uint32_t d1, uint32_t KW, uint32_t KH, uint32_t aligned)
        : s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), KW(KW), KH(KH), aligned(aligned) {}

    uint32_t s0, s1, p0, p1, d0, d1, KW, KH;
    // when set, shader can skip K/CRS/NPQ bounds checks and address clamps
    uint32_t aligned;

    bool క్రియాకారకం<(స్థిరరాశి vk_conv2d_pipeline_state &b) స్థిరరాశి {
        తిరిగి std::tie(s0, s1, p0, p1, d0, d1, KW, KH, aligned) <
               std::tie(b.s0, b.s1, b.p0, b.p1, b.d0, b.d1, b.KW, b.KH, b.aligned);
    }
};

నిర్మాణం vk_conv3d_pipeline_state {
    vk_conv3d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t p0, uint32_t p1, uint32_t p2,
                             uint32_t d0, uint32_t d1, uint32_t d2, uint32_t KW, uint32_t KH, uint32_t KD, uint32_t aligned)
        : s0(s0), s1(s1), s2(s2), p0(p0), p1(p1), p2(p2), d0(d0), d1(d1), d2(d2), KW(KW), KH(KH), KD(KD), aligned(aligned) {}

    uint32_t s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD;
    uint32_t aligned;

    bool క్రియాకారకం<(స్థిరరాశి vk_conv3d_pipeline_state &b) స్థిరరాశి {
        తిరిగి std::tie(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned) <
               std::tie(b.s0, b.s1, b.s2, b.p0, b.p1, b.p2, b.d0, b.d1, b.d2, b.KW, b.KH, b.KD, b.aligned);
    }
};

నిర్మాణం vk_solve_tri_pipeline_state {
    vk_solve_tri_pipeline_state(uint32_t N, uint32_t K)
        : N(N), K(K) {}

    uint32_t N, K;

    bool క్రియాకారకం<(స్థిరరాశి vk_solve_tri_pipeline_state &b) స్థిరరాశి {
        తిరిగి std::tie(N, K) <
               std::tie(b.N, b.K);
    }
};

గణన shader_reduction_బహుళకం {
    SHADER_REDUCTION_బహుళకం_SHMEM,
    SHADER_REDUCTION_బహుళకం_HYBRID,
    SHADER_REDUCTION_బహుళకం_SUBGROUP,
    SHADER_REDUCTION_బహుళకం_లెక్క,
};

స్థిర constexpr uint32_t అంకె_argsort_pipelines = 11;

స్థిర constexpr uint32_t అంకె_topk_moe_pipelines = 10;

స్థిర constexpr uint32_t అంకె_topk_pipelines = 11;

స్థిర constexpr std::initializer_list<ggml_op> topk_moe_early_softmax_norm{ GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_అడ్డువరుసలు, GGML_OP_RESHAPE,
                                                                             GGML_OP_కూడిక_అడ్డువరుసలు, GGML_OP_CLAMP,    GGML_OP_DIV,
                                                                             GGML_OP_RESHAPE };

స్థిర constexpr std::initializer_list<ggml_op> topk_moe_sigmoid_norm_bias{ GGML_OP_UNARY,    GGML_OP_RESHAPE,  GGML_OP_కూడు,
                                                                            GGML_OP_ARGSORT,  GGML_OP_VIEW,     GGML_OP_GET_అడ్డువరుసలు,
                                                                            GGML_OP_RESHAPE,  GGML_OP_కూడిక_అడ్డువరుసలు, GGML_OP_CLAMP,
                                                                            GGML_OP_DIV,      GGML_OP_RESHAPE };

స్థిర constexpr std::initializer_list<ggml_op> topk_moe_sqrt_softplus_norm_bias{ GGML_OP_UNARY,    GGML_OP_SQRT,
                                                                                  GGML_OP_RESHAPE,  GGML_OP_కూడు,
                                                                                  GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                                  GGML_OP_GET_అడ్డువరుసలు, GGML_OP_RESHAPE,
                                                                                  GGML_OP_కూడిక_అడ్డువరుసలు, GGML_OP_CLAMP,
                                                                                  GGML_OP_DIV,      GGML_OP_RESHAPE };

స్థిర constexpr std::initializer_list<ggml_op> topk_moe_early_softmax     { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_అడ్డువరుసలు };

స్థిర constexpr std::initializer_list<ggml_op> topk_moe_late_softmax      { GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                             GGML_OP_GET_అడ్డువరుసలు, GGML_OP_RESHAPE,
                                                                             GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

స్థిర constexpr std::initializer_list<ggml_op> snake_నమూనా              { GGML_OP_MUL,      GGML_OP_SIN,
                                                                             GGML_OP_SQR,      GGML_OP_MUL,
                                                                             GGML_OP_కూడు };

స్థిర constexpr std::initializer_list<ggml_op> topk_qsa_నమూనా { GGML_OP_GET_అడ్డువరుసలు, GGML_OP_క్రమమార్పు,
                                                                   GGML_OP_CONT,     GGML_OP_CPY,
                                                                   GGML_OP_RESHAPE,  GGML_OP_కూడు,
                                                                   GGML_OP_శిఖరం_K };

స్థిర constexpr std::initializer_list<std::array<పూర్ణాంకం, 3>> topk_qsa_అంచులు {
    { 1, 0, 0 }, // permute->src[0] == get_rows
    { 2, 0, 1 }, // cont->src[0]    == permute
    { 4, 0, 3 }, // reshape->src[0] == cpy (mask cast)
    { 5, 0, 2 }, // add->src[0]     == cont
    { 5, 1, 4 }, // add->src[1]     == reshape
    { 6, 0, 5 }, // top_k->src[0]   == add
};

స్థిర constexpr std::initializer_list<ggml_op> rms_norm_mul_కూడు_mul_నమూనా { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_కూడు, GGML_OP_MUL };

స్థిర constexpr std::initializer_list<ggml_op> rms_norm_mul_కూడు_నమూనా     { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_కూడు };

స్థిర constexpr std::initializer_list<ggml_op> rms_norm_mul_rope_view_సమితి_అడ్డువరుసలు_నమూనా { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_సమితి_అడ్డువరుసలు };

స్థిర constexpr std::initializer_list<ggml_op> rms_norm_view_సమితి_అడ్డువరుసలు_నమూనా { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_సమితి_అడ్డువరుసలు };

స్థిర constexpr std::initializer_list<ggml_op> rope_view_సమితి_అడ్డువరుసలు_నమూనా { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_సమితి_అడ్డువరుసలు };

// scale_out*sigmoid(scale_in*x) as the hc_post weights (qwen4exp hc_combine)
స్థిర constexpr std::initializer_list<ggml_op> hc_post_gate_నమూనా { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

స్థిర constexpr std::initializer_list<std::array<పూర్ణాంకం, 3>> hc_post_gate_అంచులు {
    { 1, 0, 0 }, // sigmoid->src[0] == scale
    { 2, 0, 1 }, // scale->src[0]   == sigmoid
    { 3, 2, 2 }, // hc_post->src[2] == scale (post)
};

స్థిర constexpr std::initializer_list<std::array<పూర్ణాంకం, 3>> topk_moe_early_softmax_norm_అంచులు {
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

స్థిర constexpr std::initializer_list<std::array<పూర్ణాంకం, 3>> topk_moe_sigmoid_norm_bias_అంచులు {
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

స్థిర constexpr std::initializer_list<std::array<పూర్ణాంకం, 3>> topk_moe_sqrt_softplus_norm_bias_అంచులు {
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

స్థిర constexpr std::initializer_list<std::array<పూర్ణాంకం, 3>> topk_moe_early_softmax_అంచులు {
    { 1, 0, 0 }, // reshape->src[0]  == softmax
    { 2, 0, 0 }, // argsort->src[0]  == softmax
    { 3, 0, 2 }, // view->src[0]     == argsort
    { 4, 0, 1 }, // get_rows->src[0] == reshape
    { 4, 1, 3 }, // get_rows->src[1] == view
};

స్థిర constexpr std::initializer_list<std::array<పూర్ణాంకం, 3>> topk_moe_late_softmax_అంచులు {
    { 1, 0, 0 }, // view->src[0]     == argsort
    { 2, 1, 1 }, // get_rows->src[1] == view
    { 3, 0, 2 }, // reshape->src[0]  == get_rows
    { 4, 0, 3 }, // soft_max->src[0] == reshape
    { 5, 0, 4 }, // reshape->src[0]  == soft_max
};

గణన topk_moe_బహుళకం {
    TOPK_MOE_EARLY_SOFTMAX,
    TOPK_MOE_EARLY_SOFTMAX_NORM,
    TOPK_MOE_LATE_SOFTMAX,
    TOPK_MOE_SIGMOID_NORM_BIAS,
    TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS,
    TOPK_MOE_లెక్క,
};

గణన rms_norm_బహుళకం {
    RMS_NORM_MUL,
    RMS_NORM_MUL_కూడు,
    RMS_NORM_MUL_కూడు_MUL,
    RMS_NORM_MUL_ROPE,
    RMS_NORM_MUL_ROPE_VIEW_సమితి_అడ్డువరుసలు,
    RMS_NORM_VIEW_సమితి_అడ్డువరుసలు,
    RMS_NORM_లెక్క,
};

స్థిర constexpr std::initializer_list<std::array<పూర్ణాంకం, 3>> rope_view_సమితి_అడ్డువరుసలు_అంచులు {
    { 1, 0, 0 }, // view->src[0]     == rope
    { 2, 0, 1 }, // set_rows->src[0] == view
};

స్థిర constexpr std::initializer_list<std::array<పూర్ణాంకం, 3>> rms_norm_mul_rope_view_సమితి_అడ్డువరుసలు_అంచులు {
    { 1, 0, 0 }, // mul->src[0]      == rms
    { 2, 0, 1 }, // rope->src[0]     == mul
    { 3, 0, 2 }, // view->src[0]     == rope
    { 4, 0, 3 }, // set_rows->src[0] == view
};

స్థిర constexpr std::initializer_list<std::array<పూర్ణాంకం, 3>> rms_norm_view_సమితి_అడ్డువరుసలు_అంచులు {
    { 1, 0, 0 }, // view->src[0]     == rms_norm
    { 2, 0, 1 }, // set_rows->src[0] == view
};

స్థిర constexpr std::array<ggml_type, 9> lightning_indexer_k_types = {
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

class vk_జ్ఞాపకం_logger;

నిర్మాణం vk_device_struct {
    std::recursive_mutex mutex;
    std::mutex వరుస_submit_mutex;

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
    uint64_t max_జ్ఞాపకం_కేటాయింపు_కొలత;
    uint64_t max_బఫరు_కొలత;
    uint64_t suballocation_ఖండం_కొలత;
    uint64_t min_imported_host_సూచిక_alignment;
    bool external_జ్ఞాపకం_host {};
    bool fp16;
    bool bf16;
    bool pipeline_robustness;
    bool జ్ఞాపకం_ప్రాధాన్యత;
    vk::Device device;
    uint32_t vendor_id;
    vk::DriverId driver_id;
    vk_device_architecture architecture;
    std::unique_ptr<vk_వరుస> compute_వరుస;
    std::unique_ptr<vk_వరుస> transfer_వరుస;
    bool single_వరుస;
    bool support_async;
    bool async_use_transfer_వరుస;
    bool has_internally_synchronized_queues = false;
    uint32_t subgroup_కొలత;
    uint32_t subgroup_కొలత_log2;
    uint32_t shader_core_లెక్క;
    bool uma;
    bool prefer_host_జ్ఞాపకం;
    bool float_controls_rte_fp16;
    bool float_controls_denorm_preserve_fp16;
    bool subgroup_basic;
    bool subgroup_arithmetic;
    bool subgroup_shuffle;
    bool subgroup_ballot;
    bool subgroup_clustered;
    bool subgroup_vote;
    bool multi_కూడు;
    bool shader_int64;
    bool బఫరు_device_చిరునామా;
    bool vulkan_జ్ఞాపకం_model;

    bool కూడు_rms_fusion;
    uint32_t partials_binding_alignment;
    uint32_t max_nodes_చొప్పున_submit;

    bool shader_64b_indexing;

    bool integer_dot_లబ్ధం;
    // 0: default, 1: force mmvq, -1: disable mmvq
    int32_t mmvq_బహుళకం;

    bool subgroup_కొలత_control;
    uint32_t subgroup_min_కొలత;
    uint32_t subgroup_max_కొలత;
    bool subgroup_require_నిండిన_support;

    // floor(log2(maxComputeWorkGroupInvocations))
    uint32_t max_workgroup_కొలత_log2 {};

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
    bool coopmat2_decode_సదిశ;

    bool dot2_f16 {};
    bool ocp_fp4 {};

    bool pipeline_executable_properties_support {};

    bool device_fault {};
    PFN_vkGetDeviceFaultInfoEXT pfn_vkGetDeviceFaultInfoEXT {};

    bool serialize_submissions {};

    స్థిరరాశి ggml_cgraph * diag_cgraph {};
    పూర్ణాంకం diag_prev_ప్రారంభం = -1;
    పూర్ణాంకం diag_prev_చివర = -1;

    size_t idx;

    bool mul_mat_l[GGML_TYPE_లెక్క];
    bool mul_mat_m[GGML_TYPE_లెక్క];
    bool mul_mat_s[GGML_TYPE_లెక్క];
    bool mul_mat_id_l[GGML_TYPE_లెక్క];
    bool mul_mat_id_m[GGML_TYPE_లెక్క];
    bool mul_mat_id_s[GGML_TYPE_లెక్క];

    // Separate flags for the q8_1 (integer dot) mmq path, whose shader uses
    // a different shared-memory layout than the float matmul shaders.
    bool mul_mat_l_int[GGML_TYPE_లెక్క];
    bool mul_mat_m_int[GGML_TYPE_లెక్క];
    bool mul_mat_s_int[GGML_TYPE_లెక్క];
    bool mul_mat_id_l_int[GGML_TYPE_లెక్క];
    bool mul_mat_id_m_int[GGML_TYPE_లెక్క];
    bool mul_mat_id_s_int[GGML_TYPE_లెక్క];

    vk::Descriptor_సమితి_Layout dsl;

    std::map<vk_matmul_pipeline_కీలకం, std::సదిశ<vk_matmul_pipeline_pair>> pipeline_matmul;
    matmul_tile_selector_t matmul_tile_selector;
    matmul_tile_selector_t matmul_id_tile_selector;

    vk_pipeline pipeline_matmul_విభజించు_k_reduce;
    vk_pipeline pipeline_quantize_q8_1_x4;

    vk_pipeline pipeline_dequant[GGML_TYPE_లెక్క];
    vk_pipeline pipeline_dequant_transpose[GGML_TYPE_లెక్క]; // fused dequant+transpose for FA quant-KV
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_కొలత_లెక్క][GGML_TYPE_లెక్క][mul_mat_vec_max_నిలువుల_సంఖ్య];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[DMMV_WG_కొలత_లెక్క][GGML_TYPE_లెక్క][mul_mat_vec_max_నిలువుల_సంఖ్య];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[DMMV_WG_కొలత_లెక్క][GGML_TYPE_లెక్క];

    vk_pipeline pipeline_dequant_mul_mat_vec_q8_1_f32[DMMV_WG_కొలత_లెక్క][GGML_TYPE_లెక్క][mul_mat_vec_max_నిలువుల_సంఖ్య];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_q8_1_f32[DMMV_WG_కొలత_లెక్క][GGML_TYPE_లెక్క];

    vk_pipeline pipeline_mul_mat_vec_p021_f16_f32[p021_max_gqa_నిష్పత్తి];
    vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
    vk_pipeline pipeline_get_అడ్డువరుసలు[GGML_TYPE_లెక్క];
    vk_pipeline pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_లెక్క];
    vk_pipeline pipeline_get_అడ్డువరుసలు_వెనుక_f32;
    vk_pipeline pipeline_acc_f32;
    vk_pipeline pipeline_సమితి_f32;

    // [src0 0=fp32,1=fp16][src1 0=fp32,1=fp16][dst 0=fp32,1=fp16]
    vk_pipeline pipeline_కూడు[2][2][2];
    vk_pipeline pipeline_కూడు_norepeat[2][2][2];
    vk_pipeline pipeline_sub[2][2][2];
    vk_pipeline pipeline_sub_norepeat[2][2][2];
    vk_pipeline pipeline_mul[2][2][2];
    vk_pipeline pipeline_mul_norepeat[2][2][2];
    vk_pipeline pipeline_div[2][2][2];
    vk_pipeline pipeline_div_norepeat[2][2][2];
    vk_pipeline pipeline_కూడు_rms[2][2][2];
    vk_pipeline pipeline_కూడు_rms_norepeat[2][2][2];

    // indexed by num_additional_fused_ops == num_adds - 1
    vk_pipeline pipeline_multi_కూడు[MAX_FUSED_ADDS];
    vk_pipeline pipeline_multi_కూడు_rms[MAX_FUSED_ADDS];

    vk_pipeline pipeline_కూడు_id_f32;

    vk_pipeline pipeline_concat_i8, pipeline_concat_i16, pipeline_concat_i32, pipeline_concat_i64;
    vk_pipeline pipeline_upscale_సమీప_f32, pipeline_upscale_bilinear_f32, pipeline_upscale_bicubic_f32, pipeline_upscale_bilinear_antialias_f32;
    vk_pipeline pipeline_scale_f32;
    vk_pipeline pipeline_log[2];
    vk_pipeline pipeline_tri[2];
    vk_pipeline pipeline_diag[2];
    vk_pipeline pipeline_clamp[2];
    vk_pipeline pipeline_pad_f32;
    vk_pipeline pipeline_pad_reflect_1d_f32;
    vk_pipeline pipeline_roll_f32;
    vk_pipeline pipeline_repeat_i32, pipeline_repeat_వెనుక_f32;
    vk_pipeline pipeline_repeat_i16;
    vk_pipeline pipeline_cpy_32, pipeline_cpy_f32_f16, pipeline_cpy_16, pipeline_cpy_f16_f32, pipeline_cpy_f32_bf16, pipeline_cpy_bf16_f32, pipeline_cpy_bf16_f16, pipeline_cpy_f32_i32, pipeline_cpy_i32_f32;
    vk_pipeline pipeline_contig_cpy_32, pipeline_contig_cpy_f32_f16, pipeline_contig_cpy_16, pipeline_contig_cpy_f16_f32, pipeline_contig_cpy_f32_bf16, pipeline_contig_cpy_bf16_f32, pipeline_contig_cpy_bf16_f16, pipeline_contig_cpy_f32_i32, pipeline_contig_cpy_i32_f32;
    vk_pipeline pipeline_cpy_f32_quant[GGML_TYPE_లెక్క];
    vk_pipeline pipeline_cpy_quant_f32[GGML_TYPE_లెక్క];
    vk_pipeline pipeline_cpy_transpose_16, pipeline_cpy_transpose_32;
    vk_pipeline pipeline_cpy_transpose_02_16, pipeline_cpy_transpose_02_32;
    // [src0 0=fp32,1=fp16][dst]
    vk_pipeline pipeline_సమితి_అడ్డువరుసలు_i32[2][GGML_TYPE_లెక్క];
    vk_pipeline pipeline_సమితి_అడ్డువరుసలు_i64[2][GGML_TYPE_లెక్క];
    vk_pipeline pipeline_norm_f32;
    vk_pipeline pipeline_సమూహం_norm_f32;
    vk_pipeline pipeline_rms_norm_f32;
    vk_pipeline pipeline_rms_norm_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_కూడు_f32;
    vk_pipeline pipeline_rms_norm_mul_కూడు_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_కూడు_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_కూడు_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_సమితి_అడ్డువరుసలు_f32_f32;
    vk_pipeline pipeline_rms_norm_సమితి_అడ్డువరుసలు_f32_f16;
    vk_pipeline pipeline_rms_norm_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f16;
    vk_pipeline pipeline_rms_norm_వెనుక_f32;
    vk_pipeline pipeline_l2_norm_f32;

    // [src/dst 0=fp32,1=fp16]
    vk_pipeline pipeline_exp[2];
    vk_pipeline pipeline_expm1[2];
    vk_pipeline pipeline_elu[2];
    vk_pipeline pipeline_gelu[2];
    vk_pipeline pipeline_gelu_erf[2];
    vk_pipeline pipeline_gelu_శీఘ్ర[2];
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
    vk_pipeline pipeline_అడుగు[2];
    vk_pipeline pipeline_విడత[2];
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
    vk_pipeline pipeline_geglu_శీఘ్ర[2];

    vk_pipeline pipeline_leaky_relu[2];
    vk_pipeline pipeline_silu_వెనుక_f32;
    vk_pipeline pipeline_diag_ముసుగు_inf_f32;
    vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
    vk_pipeline pipeline_soft_max_f32_wg512, pipeline_soft_max_f32_f16_wg512;
    vk_pipeline pipeline_soft_max_వెనుక_f32;

    vk_pipeline pipeline_soft_max_large1_f32, pipeline_soft_max_large1_f32_f16;
    vk_pipeline pipeline_soft_max_large2_f32, pipeline_soft_max_large2_f32_f16;
    vk_pipeline pipeline_soft_max_large3_f32, pipeline_soft_max_large3_f32_f16;

    vk_pipeline pipeline_rope_norm_f32, pipeline_rope_norm_f16, pipeline_rope_norm_f32_f16;
    vk_pipeline pipeline_rope_neox_f32, pipeline_rope_neox_f16, pipeline_rope_neox_f32_f16;
    vk_pipeline pipeline_rope_multi_f32, pipeline_rope_multi_f16, pipeline_rope_multi_f32_f16;
    vk_pipeline pipeline_rope_vision_f32, pipeline_rope_vision_f16;
    vk_pipeline pipeline_argsort_f32[అంకె_argsort_pipelines];
    vk_pipeline pipeline_argsort_large_f32[అంకె_argsort_pipelines];
    vk_pipeline pipeline_topk_f32[అంకె_topk_pipelines];
    vk_pipeline pipeline_topk_radix_f32;
    vk_pipeline pipeline_topk_radix_qsa; // qwen4 QSA indexer fusion (f16 mask)
    vk_pipeline pipeline_కూడిక_అడ్డువరుసలు_f32;
    vk_pipeline pipeline_cross_entropy_loss_f32, pipeline_cross_entropy_loss_f32_wg512;
    vk_pipeline pipeline_cross_entropy_loss_వెనుక_f32, pipeline_cross_entropy_loss_వెనుక_f32_wg512;
    vk_pipeline pipeline_fwht_f32[4];
    vk_pipeline pipeline_cumsum_f32;
    vk_pipeline pipeline_cumsum_small_f32;
    vk_pipeline pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline_cumsum_multipass2_f32;
    vk_pipeline pipeline_argmax_f32;
    vk_pipeline pipeline_లెక్క_equal_i32;
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
    vk_pipeline pipeline_బహిర్గతం_prod_f32;
    vk_pipeline pipeline_snake_f32;
    vk_pipeline pipeline_snake_f16;
    vk_pipeline pipeline_snake_bf16;
    vk_pipeline pipeline_pool1d_f32;
    vk_pipeline pipeline_pool2d_f32;
    vk_pipeline pipeline_pool2d_subgroup_f32;
    vk_pipeline pipeline_rwkv_wkv6_f32;
    vk_pipeline pipeline_rwkv_wkv7_f32;
    vk_pipeline pipeline_gated_రేఖీయ_attn_f32;
    vk_pipeline pipeline_lightning_indexer_f32[GGML_TYPE_లెక్క];
    // [size_idx][kda] where size_idx: 0=d16, 1=d32, 2=d64, 3=d128
    vk_pipeline pipeline_gated_delta_net[4][2];
    vk_pipeline pipeline_ssm_పరిశీలన_f32_d128;
    vk_pipeline pipeline_ssm_పరిశీలన_f32_d256;
    vk_pipeline pipeline_ssm_conv_f32;
    vk_pipeline pipeline_ssm_conv_silu_f32;
    vk_pipeline pipeline_ssm_conv_bias_silu_f32;
    vk_pipeline pipeline_opt_అడుగు_adamw_f32;
    vk_pipeline pipeline_opt_అడుగు_sgd_f32;
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f32[CONV_SHAPE_లెక్క];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f16_f32[CONV_SHAPE_లెక్క];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f32[CONV_SHAPE_లెక్క];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f16_f32[CONV_SHAPE_లెక్క];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f32[CONV_SHAPE_లెక్క];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f16_f32[CONV_SHAPE_లెక్క];
    vk_pipeline pipeline_conv2d_dw_whcn_f32, pipeline_conv2d_dw_whcn_f16_f32;
    vk_pipeline pipeline_conv2d_dw_cwhn_f32, pipeline_conv2d_dw_cwhn_f16_f32;

    std::map<vk_fa_pipeline_state, vk_pipeline> pipeline_flash_attn_f32_f16;

    std::map<std::pair<uint32_t, uint32_t>, vk_pipeline> pipeline_fa_ముసుగు_opt;

    vk_pipeline pipeline_fa_sparse_compact;
    vk_pipeline pipeline_fa_sparse_compact_subgroup;
    bool fa_sparse_compact_use_subgroups;

    vk_pipeline pipeline_flash_attn_విభజించు_k_reduce;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, std::pair<vk_pipeline, vk_pipeline>> pipeline_xe_fa_decode_dual_phases;
    vk_pipeline pipeline_లెక్క_experts;

    // [2] is for whether to take n_experts from spec constant (0) or push constant (1)
    vk_pipeline pipeline_topk_moe[అంకె_topk_moe_pipelines][2];

    std::సదిశ<vk_pipeline_ref> all_pipelines;

    vk::Fence fence;
    vk_బఫరు sync_staging;

    ggml_backend_బఫరు_type బఫరు_type;

    bool disable_fusion;
    bool disable_descriptor_reuse;
    std::atomic<uint64_t> బఫరు_destroy_లెక్క {};
    bool disable_host_visible_vidmem;
    bool allow_sysmem_fallback;
    bool disable_లేఖాచిత్రం_optimize;

    std::unique_ptr<vk_జ్ఞాపకం_logger> జ్ఞాపకం_logger;

    ~vk_device_struct();

};

అంతఃవరుస శూన్య vk_command_pool::init(vk_device& device, vk_వరుస *q_) {
    cmd_buffers.clear();
    q = q_;

    vk::CommandPoolCreateInfo command_pool_create_info(
        vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_బిట్టు | VK_COMMAND_POOL_CREATE_RESET_COMMAND_బఫరు_బిట్టు),
        q->వరుస_family_సూచీ);
    pool = device->device.createCommandPool(command_pool_create_info);
}

అంతఃవరుస శూన్య vk_command_pool::destroy(vk::Device& device) {
    device.destroyCommandPool(pool);
    pool = nullptr;
    cmd_buffers.clear();
}

నిర్మాణం vk_బఫరు_struct {
    vk::బఫరు_2 బఫరు = VK_NULL_HANDLE;
    vk::Device_జ్ఞాపకం device_జ్ఞాపకం = VK_NULL_HANDLE;
    vk::జ్ఞాపకం_Property_Flags జ్ఞాపకం_property_flags;
    శూన్య * ptr;
    size_t కొలత = 0;
    vk::Device_చిరునామా bda_addr {};

    vk_device device;

    ~vk_బఫరు_struct() {
        ఒకవేళ (కొలత == 0) {
            తిరిగి;
        }
        VK_LOG_DEBUG("~vk_buffer_struct(" << బఫరు << ", " << కొలత << ")");

        // bump before destroying, so a thread that sees the buffer gone also sees the new count
        device->బఫరు_destroy_లెక్క.fetch_కూడు(1, std::memory_order_release);
        device->device.free_జ్ఞాపకం_2(device_జ్ఞాపకం);
        device->device.destroy_బఫరు(బఫరు);
    }
};

నిర్మాణం vk_subbuffer {
    vk_బఫరు బఫరు;
    uint64_t స్థానభ్రంశం;
    uint64_t కొలత;

    క్రియాకారకం vk::Descriptor_బఫరు_Info() స్థిరరాశి {
        తిరిగి { బఫరు->బఫరు, స్థానభ్రంశం, కొలత };
    }
};

నిర్మాణం vk_semaphore {
    vk::Semaphore s;
    uint64_t విలువ;
};

నిర్మాణం vk_event {
    std::సదిశ<vk::Event> events_free; // Events available for reuse
    std::సదిశ<vk::Event> events_submitted; // Events that are fully submitted and can be reused on next synchronize
    vk::Event event;
    bool has_event;

    vk_semaphore tl_semaphore;
    vk_command_బఫరు* cmd_బఫరు = nullptr;
    uint64_t cmd_బఫరు_use_counter = 0;
};

నిర్మాణం vk_submission {
    vk_command_బఫరు* బఫరు = nullptr;
    std::సదిశ<vk_semaphore> wait_semaphores;
    std::సదిశ<vk_semaphore> signal_semaphores;
};

రకనిర్వచనం std::సదిశ<vk_submission> vk_sequence;

#నిర్వచించు MAT_VEC_FUSION_FLAGS_BIAS0 0x1

#నిర్వచించు MAT_VEC_FUSION_FLAGS_BIAS1 0x2

#నిర్వచించు MAT_VEC_FUSION_FLAGS_SCALE0 0x4

#నిర్వచించు MAT_VEC_FUSION_FLAGS_SCALE1 0x8

నిర్మాణం vk_staging_memcpy {
    vk_staging_memcpy(శూన్య * _dst, స్థిరరాశి శూన్య * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    శూన్య * dst;
    స్థిరరాశి శూన్య * src;
    size_t n;
};

నిర్మాణం vk_staging_memset {
    vk_staging_memset(శూన్య * _dst, uint32_t _val, size_t _n) : dst(_dst), val(_val), n(_n) {}

    శూన్య * dst;
    uint32_t val;
    size_t n;
};

నిర్మాణం vk_context_struct {
    vk_submission * s;
    std::సదిశ<vk_sequence> seqs;

    పూర్ణాంకం exit_tensor_idx;

    std::సదిశ<vk_staging_memcpy> లోపల_memcpys;
    std::సదిశ<vk_staging_memcpy> బహిర్గతం_memcpys;
    std::సదిశ<vk_staging_memset> memsets;

    std::సదిశ<std::string> debug_labels;

    vk_command_pool * p {};
};

రకనిర్వచనం std::shared_ptr<vk_context_struct> vk_context;

రకనిర్వచనం std::weak_ptr<vk_context_struct> vk_context_ref;

నిర్మాణం ggml_vk_garbage_collector {
    std::సదిశ<vk_semaphore> tl_semaphores;
    std::సదిశ<vk_semaphore> semaphores;
    std::సదిశ<vk::Event> events;
    std::సదిశ<vk_context> contexts;
};

#నిర్వచించు VK_LOG_జ్ఞాపకం(సందేశం_2) ఒకవేళ (vk_జ్ఞాపకం_logger_enabled) { std::cerr << "ggml_vulkan memory: " << సందేశం_2 << std::endl; }

స్థిర std::string format_కొలత(size_t కొలత) {
    స్థిరరాశి size_t kib = 1024;
    స్థిరరాశి size_t mib = kib * 1024;
    స్థిరరాశి size_t gib = mib * 1024;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    ఒకవేళ (కొలత >= gib) {
        oss << static_cast<ద్విగుణ>(కొలత) / gib << " GiB";
    } లేకపోతే ఒకవేళ (కొలత >= mib) {
        oss << static_cast<ద్విగుణ>(కొలత) / mib << " MiB";
    } లేకపోతే ఒకవేళ (కొలత >= kib) {
        oss << static_cast<ద్విగుణ>(కొలత) / kib << " KiB";
    } లేకపోతే {
        oss << కొలత << " B";
    }

    తిరిగి oss.str();
}

class vk_జ్ఞాపకం_logger {
బహిరంగ:
    vk_జ్ఞాపకం_logger(): మొత్తం_device(0), మొత్తం_host(0) {}
    శూన్య log_కేటాయింపు(vk_బఫరు_ref buf_ref, size_t కొలత);
    శూన్య log_deallocation(vk_బఫరు_ref buf_ref);

అంతరంగ:
    std::map<vk::బఫరు_2, size_t> allocations; // Track allocations
    size_t మొత్తం_device;
    size_t మొత్తం_host;
    స్థిర std::mutex log_mutex;
};

అంతఃవరుస std::mutex vk_జ్ఞాపకం_logger::log_mutex;

class vk_perf_logger {
  బహిరంగ:
    శూన్య print_timings(bool force = false);


    std::string get_కణుపు_fusion_name(స్థిరరాశి ggml_tensor * కణుపు, స్థిరరాశి అక్షరం *fusion_name, uint64_t *n_flops);


    శూన్య log_timing(స్థిరరాశి ggml_tensor * కణుపు, స్థిరరాశి అక్షరం *fusion_name, uint64_t సమయం) {
        uint64_t n_flops;
        std::string name = get_కణుపు_fusion_name(కణుపు, fusion_name, &n_flops);
        ఒకవేళ (n_flops) {
            flops[name].నెట్టు_వెనుక(n_flops);
        }
        timings[name].నెట్టు_వెనుక(సమయం);
    }

    శూన్య log_timing(స్థిరరాశి std::సదిశ<ggml_tensor *> &nodes, స్థిరరాశి std::సదిశ<స్థిరరాశి అక్షరం *> &పేర్లు, uint64_t సమయం) {
        uint64_t మొత్తం_flops = 0;
        std::string name;
        కొరకు (size_t n = 0; n < nodes.కొలత(); ++n) {
            uint64_t n_flops = 0;
            name += get_కణుపు_fusion_name(nodes[n], పేర్లు[n], &n_flops);
            మొత్తం_flops += n_flops;

            ఒకవేళ (n != nodes.కొలత() - 1) {
                name += ", ";
            }
        }
        ఒకవేళ (మొత్తం_flops) {
            flops[name].నెట్టు_వెనుక(మొత్తం_flops);
        }
        timings[name].నెట్టు_వెనుక(సమయం);
    }

  అంతరంగ:
    std::map<std::string, std::సదిశ<uint64_t>> timings;
    std::map<std::string, std::సదిశ<uint64_t>> flops;
    uint32_t print_లెక్క {};
};

నిర్మాణం ggml_backend_vk_context {
    std::string name;

    vk_device device;

    size_t semaphore_idx, event_idx;
    ggml_vk_garbage_collector gc;
    size_t prealloc_కొలత_x, prealloc_కొలత_y, prealloc_కొలత_విభజించు_k, prealloc_కొలత_కూడు_rms_partials, prealloc_కొలత_కూడు_rms_partials_స్థానభ్రంశం;
    vk_బఫరు prealloc_x, prealloc_y, prealloc_విభజించు_k, prealloc_కూడు_rms_partials, sync_staging;
    vk::Fence fence, almost_ready_fence;
    bool submit_pending {};
    bool almost_ready_fence_pending {};
    // Set before op_add and unset after op_rms_norm to indicate that the add should
    // write partial sums to accumulate the square of the vector components
    bool do_కూడు_rms_partials_స్థానభ్రంశం_calculation;
    bool do_కూడు_rms_partials;

    uint64_t చివరి_మొత్తం_flops {UINT64_MAX};

    // Cache most recent tensor that was converted into prealloc_y, and what pipeline it used to convert.
    vk_pipeline_struct * prealloc_y_చివరి_pipeline_వాడిన {};
    స్థిరరాశి ggml_tensor * prealloc_y_చివరి_tensor_వాడిన {};
    // True when the K dimension in prealloc_y is padded.
    bool prealloc_y_చివరి_k_padded {};

    // Track which nodes have been used since the last sync, and whether they were written to
    std::సదిశ<స్థిరరాశి ggml_tensor *> unsynced_nodes_written;
    std::సదిశ<స్థిరరాశి ggml_tensor *> unsynced_nodes_read;
    // Track which prealloc buffers have pending reads that need to be synchronized.
    // These are checked before writing to the buffer (and call ggml_vk_sync_buffers if set),
    // and set to true after the buffer contents are consumed.
    bool prealloc_x_need_sync, prealloc_y_need_sync, prealloc_విభజించు_k_need_sync;

    vk_context_ref compute_ctx;

    vk_context_ref transfer_ctx;
    vk_semaphore transfer_semaphore;
    uint64_t transfer_semaphore_చివరి_submitted {};

    std::సదిశ<vk_context_ref> tensor_ctxs;

    std::సదిశ<vk::DescriptorPool> descriptor_pools;
    std::సదిశ<vk::Descriptor_సమితి> descriptor_sets;
    // last bindings written to each set; descriptor_sets is append-only so an index always names the same set
    std::సదిశ<std::సదిశ<vk::Descriptor_బఫరు_Info>> descriptor_సమితి_bindings;
    uint64_t descriptor_సమితి_bindings_destroy_లెక్క {};
    uint32_t descriptor_సమితి_idx {};
    uint32_t pipeline_descriptor_సమితి_requirements {};

    vk_command_pool compute_cmd_pool;
    vk_command_pool transfer_cmd_pool;

    // number of additional consecutive nodes that are being fused with the
    // node currently being processed
    పూర్ణాంకం అంకె_additional_fused_ops {};
    // Bitmask of which fused ops need to write an intermediate value to memory.
    // Bit 'i' means nodes[start_of_fusion + i] writes to memory.
    // If there's no fusion, bit 0 is still set.
    పూర్ణాంకం fused_ops_write_ముసుగు {};
    topk_moe_బహుళకం fused_topk_moe_బహుళకం {};
    bool fused_topk_moe_scale {};
    // QSA indexer gather+add+top_k fused into one radix-select
    bool fused_topk_qsa {};
    bool fused_hc_post_gate {};
    rms_norm_బహుళకం fused_rms_norm_బహుళకం {RMS_NORM_లెక్క};

    // for GGML_VK_PERF_LOGGER
    std::unique_ptr<vk_perf_logger> perf_logger;
    vk::QueryPool query_pool;
    std::సదిశ<స్థిరరాశి అక్షరం *> query_fusion_పేర్లు;
    std::సదిశ<పూర్ణాంకం> query_fusion_కణుపు_లెక్క;
    std::సదిశ<ggml_tensor *> query_nodes;
    std::సదిశ<పూర్ణాంకం> query_కణుపు_idx;
    int32_t అంకె_queries {};
    int32_t query_idx {};
};

నిర్మాణం ggml_backend_vk_బఫరు_context {
    vk_device_ref device;
    vk_బఫరు dev_బఫరు;
    std::string name;

    ggml_backend_vk_బఫరు_context(vk_device_ref device, vk_బఫరు&& dev_బఫరు, std::string& name) :
        device(device),
        dev_బఫరు(dev_బఫరు),
        name(name) {
    }

    ~ggml_backend_vk_బఫరు_context();

};

నిర్మాణం vk_host_కేటాయింపు {
    శూన్య *    పునాది = nullptr;
    size_t    కొలత = 0;
    vk_బఫరు views[GGML_VK_MAX_DEVICES];  // indexed by device->idx; null = not (yet) mapped here
};

నిర్మాణం vk_instance_t {
    vk::Instance instance;

    bool debug_utils_support = false;  // VK_EXT_debug_utils enabled
    PFN_vk_సమితి_Debug_Utils_Object_Name_EXT pfn_vk_సమితి_Debug_Utils_Object_Name_EXT = {};
    PFN_vk_వరుస_Begin_Debug_Utils_Label_EXT pfn_vk_వరుస_Begin_Debug_Utils_Label_EXT = {};
    PFN_vk_వరుస_చివర_Debug_Utils_Label_EXT   pfn_vk_వరుస_చివర_Debug_Utils_Label_EXT   = {};
    PFN_vkCmdBeginDebugUtilsLabelEXT   pfn_vkCmdBeginDebugUtilsLabelEXT   = {};
    PFN_vk_Cmd_చివర_Debug_Utils_Label_EXT pfn_vk_Cmd_చివర_Debug_Utils_Label_EXT = {};
    PFN_vk_Cmd_చొప్పించు_Debug_Utils_Label_EXT  pfn_vk_Cmd_చొప్పించు_Debug_Utils_Label_EXT  = {};

    std::సదిశ<size_t> device_indices;
    std::సదిశ<bool>   device_supports_membudget;
    std::సదిశ<bool>   device_host_బఫరు_import;
    vk_device devices[GGML_VK_MAX_DEVICES];

    std::సదిశ<vk_host_కేటాయింపు> host_allocations;
    std::shared_mutex               host_allocations_mutex;

    size_t host_import_alignment = 0;
    uint32_t host_import_failed = 0;  // bitmask by device->idx, guarded by host_allocations_mutex

    bool multi_device = false;
};

రకనిర్వచనం శూన్య (*ggml_vk_func_t)(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);

స్థిర constexpr uint32_t k_Spv_Op_Cooperative_మాతృక_ఎక్కించు_Tensor_NV = 5367;

స్థిర constexpr uint32_t k_Spv_Capability_Cooperative_మాతృక_Decode_సదిశ_NV = 5447;

స్థిర constexpr uint32_t k_Spv_Tensor_Addressing_Decode_సదిశ_Func_బిట్టు = 0x4;

నిర్మాణం vk_fa_tuning_params {
    Fa_సంకేతం_మార్గం మార్గం;
    uint32_t workgroup_కొలత;
    uint32_t subgroup_కొలత;
    uint32_t ఖండం_అడ్డువరుసలు;
    uint32_t ఖండం_నిలువుల_సంఖ్య;
    uint32_t d_విభజించు;
    uint32_t అడ్డువరుస_విభజించు;
    bool shmem_staging;
    bool disable_subgroups;
    uint32_t హద్దు_occupancy_shmem;

    శూన్య print() స్థిరరాశి {
        std::cerr << "path=" << మార్గం << " workgroup_size=" << workgroup_కొలత << " subgroup_size=" << subgroup_కొలత <<
                     " block_rows=" << ఖండం_అడ్డువరుసలు << " block_cols=" << ఖండం_నిలువుల_సంఖ్య << " d_split=" << d_విభజించు <<
                     " row_split=" << అడ్డువరుస_విభజించు << " shmem_staging=" << shmem_staging << " disable_subgroups=" << disable_subgroups <<
                     " limit_occupancy_shmem=" << హద్దు_occupancy_shmem << std::endl;
    }
};

నిర్మాణం GpuPipelineConfig {
    // GPU architecture identifier.
    // Example: vk_device_architecture::AMD_GCN
    vk_device_architecture arch;

    // Mapping of pipeline names to their specific subgroup sizes.
    // Example: {"soft_max_f32", 64}
    std::unordered_map<std::string, uint32_t> pipelines;

    // Default subgroup size for this GPU.
    // Defaults to 0 if not explicitly provided.
    uint32_t default_subgroup_కొలత = 0;
};

స్థిర constexpr uint32_t RDNA_DEFAULT_SUBGROUP_కొలత = 32;

నిర్మాణం CompileTask {
    vk_pipeline pipeline;
    size_t spv_కొలత;
    స్థిరరాశి శూన్య * spv_దత్తాంశం;
    std::string entrypoint;
    uint32_t parameter_లెక్క;
    std::array<uint32_t, 3> wg_denoms;
    std::సదిశ<uint32_t> specialization_constants;
    bool disable_robustness;
    bool require_నిండిన_subgroups;
    uint32_t required_subgroup_కొలత;
};

నిర్మాణం ggml_vk_debug_label {
    // at most one of these is set, depending on the scope the label was opened in
    vk_context_struct * subctx {};
    vk_వరుస_handle *   qhandle {};

    // one region per dispatch, e.g. "matmul_q4_k_f32_f16acc_aligned_m (192,8,1)".
    // RGP cannot recover the pipeline name on its own, it only has the hash
    ggml_vk_debug_label(vk_context & ctx, స్థిరరాశి std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2);


    // one region per graph node
    // fused nodes are joined with '+', e.g. "RMS_NORM+MUL+ROPE Qcur-19"
    ggml_vk_debug_label(vk_context & ctx, స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, పూర్ణాంకం n_fused);


    // one region per graph evaluation, opened on the queue instead of a command buffer
    // so it spans every submit the evaluation makes
    ggml_vk_debug_label(vk_వరుస_handle * handle, స్థిరరాశి అక్షరం * name);


    // call before the command buffer can end, the destructor covers the rest
    శూన్య close();


    ~ggml_vk_debug_label() {
        close();
    }

    ggml_vk_debug_label(స్థిరరాశి ggml_vk_debug_label &) = delete;
    ggml_vk_debug_label & క్రియాకారకం=(స్థిరరాశి ggml_vk_debug_label &) = delete;

అంతరంగ:
    // the constructors check this too, so the name is not built when markers are off
    శూన్య begin(vk_context & ctx, స్థిరరాశి std::string & name);

};

#నిర్వచించు వాడని GGML_వాడని

నిర్మాణం ggml_backend_vk_device_context {
    size_t device;
    std::string name;
    std::string description;
    bool is_integrated_gpu;
    std::string pci_bus_id;
    పూర్ణాంకం op_offload_min_batch_కొలత;
};
