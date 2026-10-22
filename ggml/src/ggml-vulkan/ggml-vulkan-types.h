#निर्देशः once

#अन्तर्भाव "ggml-vulkan.h"

#अन्तर्भाव <vulkan/vulkan_core.h>

#यदि defined(GGML_VULKAN_सततम्_TESTS) || defined(GGML_VULKAN_CHECK_RESULTS)
#अन्तर्भाव <chrono>
#अन्तर्भाव "ggml-cpu.h"
#यदि_अन्तः

#परिभाषय VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#यदि VK_शीर्षकम्_संस्करणम् >= 301
namespace vk::detail { class DispatchLoaderDynamic; }
using vk::detail::DispatchLoaderDynamic;
#अन्यथा
namespace vk { class DispatchLoaderDynamic; }
using vk::DispatchLoaderDynamic;
#यदि_अन्तः

DispatchLoaderDynamic & ggml_vk_default_dispatcher();

#परिभाषय VULKAN_HPP_DEFAULT_DISPATCHER ggml_vk_default_dispatcher()

#अन्तर्भाव <vulkan/vulkan.hpp>

#यदि_अपरिभाषितम् VK_NV_cooperative_matrix_decode_सदिशम्
#परिभाषय VK_NV_cooperative_matrix_decode_सदिशम् 1
#परिभाषय VK_NV_COOPERATIVE_MATRIX_DECODE_सदिशम्_EXTENSION_NAME "VK_NV_cooperative_matrix_decode_vector"
#परिभाषय VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_सदिशम्_FEATURES_NV ((VkStructureType)1000689000)
प्रकारनिर्देश संरचना Vk_Physical_Device_Cooperative_Matrix_Decode_सदिशम्_Features_NV {
    VkStructureType    sType;
    शून्य*              p_अग्र;
    VkBool32           cooperative_Matrix_Decode_सदिशम्;
} Vk_Physical_Device_Cooperative_Matrix_Decode_सदिशम्_Features_NV;
#यदि_अन्तः

#यदि __has_include(<spirv/unified1/spirv.hpp>)
#    include <spirv/unified1/spirv.hpp>
#नोचेद्यदि __has_include(<spirv-शीर्षकाणि/spirv.hpp>)
#    include <spirv-headers/spirv.hpp>
#नोचेद्यदि __has_include(<spirv.hpp>)
#    include <spirv.hpp>
#अन्यथा
     // Fallback to let the compiler throw a standard "file not found" error
#    include <spirv/unified1/spirv.hpp>
#यदि_अन्तः

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>
#अन्तर्भाव <functional>
#अन्तर्भाव <iomanip>
#अन्तर्भाव <iostream>
#अन्तर्भाव <tuple>
#अन्तर्भाव <vector>
#अन्तर्भाव <deque>
#अन्तर्भाव <sstream>
#अन्तर्भाव <utility>
#अन्तर्भाव <memory>
#अन्तर्भाव <limits>
#अन्तर्भाव <map>
#अन्तर्भाव <set>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <shared_mutex>
#अन्तर्भाव <mutex>
#अन्तर्भाव <future>
#अन्तर्भाव <condition_variable>
#अन्तर्भाव <thread>

#यदि defined(_MSC_VER)
# define NOMINMAX 1
# include <windows.h>
# define YIELD() YieldProcessor()
#नोचेद्यदि defined(__clang__) || defined(__GNUC__)
# यदि defined(__x86_64__) ||defined(__i386__)
#  include <immintrin.h>
#  define YIELD() _mm_pause()
# elif defined(__arm__) || defined(__aarch64__)
#  यदि defined(__clang__)
#   include <arm_acle.h>
#   define YIELD() __yield()
#  अन्यथा
#   define YIELD() asm चञ्चल("yield")
#  endif
# endif
#यदि_अन्तः

#यदि !defined(YIELD)
#परिभाषय YIELD()
#यदि_अन्तः

#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-vulkan-shaders.hpp"

#यदि !defined(VK_KHR_shader_bfloat16)

#परिभाषय VK_KHR_shader_bfloat16 1
#परिभाषय VK_KHR_SHADER_BFLOAT16_SPEC_संस्करणम्                          1
#परिभाषय VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME                        "VK_KHR_shader_bfloat16"
#परिभाषय VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR ((VkStructureType)1000141000)
#परिभाषय VK_COMPONENT_TYPE_BFLOAT16_KHR                               ((VkComponentTypeKHR)1000141000)

प्रकारनिर्देश संरचना VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType                       sType;
    शून्य*                                 p_अग्र;
    VkBool32                              shaderBFloat16Type;
    VkBool32                              shader_BFloat16_Dot_गुणनफलम्;
    VkBool32                              shaderBFloat16CooperativeMatrix;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;
#यदि_अन्तः

#यदि !defined(VK_VALVE_shader_mixed_float_dot_गुणनफलम्)
#परिभाषय VK_VALVE_shader_mixed_float_dot_गुणनफलम् 1
#परिभाषय VK_VALVE_SHADER_MIXED_FLOAT_DOT_गुणनफलम्_SPEC_संस्करणम् 1
#परिभाषय VK_VALVE_SHADER_MIXED_FLOAT_DOT_गुणनफलम्_EXTENSION_NAME "VK_VALVE_shader_mixed_float_dot_product"
#परिभाषय VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_गुणनफलम्_FEATURES_VALVE ((VkStructureType)1000673000)
प्रकारनिर्देश संरचना Vk_Physical_Device_Shader_Mixed_Float_Dot_गुणनफलम्_Features_VALVE {
    VkStructureType    sType;
    शून्य*              p_अग्र;
    VkBool32           shader_Mixed_Float_Dot_गुणनफलम्_Float16_Acc_Float32;
    VkBool32           shader_Mixed_Float_Dot_गुणनफलम्_Float16_Acc_Float16;
    VkBool32           shader_Mixed_Float_Dot_गुणनफलम्_BFloat16_Acc;
    VkBool32           shader_Mixed_Float_Dot_गुणनफलम्_Float8_Acc_Float32;
} Vk_Physical_Device_Shader_Mixed_Float_Dot_गुणनफलम्_Features_VALVE;
#यदि_अन्तः

#यदि !defined(VK_EXT_shader_ocp_microscaling_types)
#परिभाषय VK_EXT_shader_ocp_microscaling_types 1
#परिभाषय VK_EXT_SHADER_OCP_MICROSCALING_TYPES_SPEC_संस्करणम् 1
#परिभाषय VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME "VK_EXT_shader_ocp_microscaling_types"
#परिभाषय VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT ((VkStructureType)1000672000)
प्रकारनिर्देश संरचना VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT {
    VkStructureType    sType;
    शून्य*              p_अग्र;
    VkBool32           shaderFloat4;
    VkBool32           shaderFloat6;
    VkBool32           shaderFloat8UnsignedE8M0;
    VkBool32           shaderMXInt8;
} VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT;
#यदि_अन्तः

#यदि !defined(VK_EXT_shader_float8)
#परिभाषय VK_EXT_shader_float8 1
#परिभाषय VK_EXT_SHADER_FLOAT8_SPEC_संस्करणम् 1
#परिभाषय VK_EXT_SHADER_FLOAT8_EXTENSION_NAME "VK_EXT_shader_float8"
#परिभाषय VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT ((VkStructureType)1000567000)
प्रकारनिर्देश संरचना VkPhysicalDeviceShaderFloat8FeaturesEXT {
    VkStructureType    sType;
    शून्य*              p_अग्र;
    VkBool32           shaderFloat8;
    VkBool32           shaderFloat8CooperativeMatrix;
} VkPhysicalDeviceShaderFloat8FeaturesEXT;
#यदि_अन्तः

#यदि_अपरिभाषितम् VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME
#परिभाषय VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME "VK_KHR_internally_synchronized_queues"
#परिभाषय VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR ((VkStructureType)1000504000)
#परिभाषय VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR ((Vk_Device_Queue_Create_Flag_द्वयङ्काः)0x00000004)

// Compile-time constant guaranteed; no runtime initialization overhead
स्थिर constexpr vk::Device_Queue_Create_Flag_द्वयङ्काः eInternallySynchronizedKHR =
    static_cast<vk::Device_Queue_Create_Flag_द्वयङ्काः>(0x00000004);

प्रकारनिर्देश संरचना VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR {
    VkStructureType    sType;
    शून्य* p_अग्र;
    VkBool32           internallySynchronizedQueues;
} VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR;
#अन्यथा
स्थिर constexpr vk::Device_Queue_Create_Flag_द्वयङ्काः eInternallySynchronizedKHR = vk::Device_Queue_Create_Flag_द्वयङ्काः::eInternallySynchronizedKHR;
#यदि_अन्तः

#परिभाषय ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))

#परिभाषय CEIL_DIV(M, N) (((M) / (N)) + (((M) % (N)) != 0))

स्थिर bool is_pow2(uint32_t x) { प्रत्यागम x > 1 && (x & (x-1)) == 0; }

#परिभाषय VK_VENDOR_ID_AMD 0x1002

#परिभाषय VK_VENDOR_ID_APPLE 0x106b

#परिभाषय VK_VENDOR_ID_INTEL 0x8086

#परिभाषय VK_VENDOR_ID_NVIDIA 0x10de

#परिभाषय VK_VENDOR_ID_QUALCOMM 0x5143

#परिभाषय VK_VENDOR_ID_SAMSUNG 0x144d

#परिभाषय VK_DEVICE_DESCRIPTOR_POOL_आकार 256

#परिभाषय VK_CHECK(err, सन्देशः_2, dev)                                     \
    कुरु {                                                            \
        vk::परिणाम_2 err_;                                            \
        try {                                                       \
            err_ = (err);                                           \
        } catch (vk::Device_नष्टम्_Error &) {                           \
            ggml_vk_print_device_नष्टम्_info(dev);                    \
            GGML_LOG_ERROR("ggml_vulkan: %s at %s:%d\n",            \
                #err, __FILE__, __LINE__);                          \
            throw;                                                  \
        }                                                           \
        यदि (err_ != vk::परिणाम_2::eSuccess) {                         \
            GGML_LOG_ERROR("ggml_vulkan: %s error %s at %s:%d\n",   \
                #err, गन्तव्यम्_string(err_).c_str(), __FILE__, __LINE__); \
            throw vk::SystemError(vk::make_error_संहिता(err_),        \
                "ggml_vulkan: " सन्देशः_2);                               \
        }                                                           \
    } यावत् (0)

#यदि_परिभाषितम् GGML_VULKAN_DEBUG
#परिभाषय VK_LOG_DEBUG(सन्देशः_2) std::cerr << सन्देशः_2 << std::endl
#अन्यथा
#परिभाषय VK_LOG_DEBUG(सन्देशः_2) ((शून्य) 0)
#यदि_अन्तः // GGML_VULKAN_DEBUG

#परिभाषय MAX_PARAMETER_गणनम् 12

#परिभाषय MAX_FUSED_ADDS (MAX_PARAMETER_गणनम् - 3)

संरचना vk_pipeline_struct;

प्रकारनिर्देश std::shared_ptr<संरचना vk_pipeline_struct> vk_pipeline;

संरचना vk_pipeline_struct {
    std::string name;
    vk::ShaderModule shader_module;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t प्रक्षिप_constant_आकार;
    uint32_t parameter_गणनम्;
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
    uint32_t पञ्जिकाकोशः_गणनम् {};

#यदि defined(VK_EXT_shader_64bit_indexing)
    bool is_64b_indexing {};
#यदि_अन्तः
    // linked list of pipelines for multiple compilation variants.
    // currently only used to compile a 64-bit indexing variant.
    vk_pipeline अग्र;
};

प्रकारनिर्देश std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;

संरचना vk_matmul_pipeline_कुञ्चिका {
    ggml_type type_a;
    ggml_type type_b;
    bool mul_mat_id;
    bool f16acc;

    bool संक्रियकः<(ध्रुव vk_matmul_pipeline_कुञ्चिका & o) ध्रुव {
        प्रत्यागम std::tie(type_a, type_b, mul_mat_id, f16acc)
             < std::tie(o.type_a, o.type_b, o.mul_mat_id, o.f16acc);
    }
};

संरचना vk_matmul_pipeline_pair {
    vk_pipeline unaligned;
    vk_pipeline aligned;
    uint32_t align;
};

संरचना vk_tile_config {
    std::सदिशम्<uint32_t> warptile;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
};

using matmul_tile_selector_t = std::function<uint32_t(
    uint32_t m, uint32_t n, uint32_t k, uint32_t shader_core_गणनम्,
    ध्रुव std::सदिशम्<vk_matmul_pipeline_pair>& configs)>;

संरचना vk_device_struct;

प्रकारनिर्देश std::shared_ptr<vk_device_struct> vk_device;

प्रकारनिर्देश std::weak_ptr<vk_device_struct> vk_device_ref;

संरचना vk_बफर_struct;

प्रकारनिर्देश std::shared_ptr<vk_बफर_struct> vk_बफर;

प्रकारनिर्देश std::weak_ptr<vk_बफर_struct> vk_बफर_ref;

संरचना ggml_backend_vk_बफर_type_context {
    std::string name;
    vk_device device;
};

संरचना vk_command_बफर {
    vk::Command_बफर buf;
    uint64_t use_गणक = 0;
    bool अन्तर्_use = false;
};

संरचना vk_queue;

संरचना vk_command_pool {
    शून्य init(vk_device& device, vk_queue *q_);
    शून्य destroy(vk::Device& device);

    vk::CommandPool pool;
    // Using deque so the pointers to command buffers
    // remain valid even if we add more
    std::deque<vk_command_बफर> cmd_buffers;

    vk_queue *q;

    size_t buffers_अन्तर्_use() ध्रुव {
        प्रत्यागम std::गणनम्_if(cmd_buffers.begin(), cmd_buffers.अन्तः_2(),
            [](ध्रुव स्वचालित& cb) { प्रत्यागम cb.अन्तर्_use; });
    }
};

संरचना vk_queue_handle {
    vk::Queue queue;
    vk_device_ref device;
    std::mutex * device_submit_mutex = nullptr;
    आभास शून्य submit(vk::ArrayProxy<ध्रुव vk::SubmitInfo> submits, vk::Fence fence) = 0;
    आभास शून्य तालकम्()   {}   // no-op by default (internally synchronized case)
    आभास शून्य unlock() {}
    आभास ~vk_queue_handle() = स्वभाव;
};

संरचना vk_queue_handle_synchronized : vk_queue_handle {
    std::mutex mutex;
    शून्य submit(vk::ArrayProxy<ध्रुव vk::SubmitInfo> submits, vk::Fence fence) override;

    शून्य तालकम्()   override { mutex.तालकम्(); }
    शून्य unlock() override { mutex.unlock(); }
};

संरचना vk_queue_handle_unsynchronized : vk_queue_handle {
    शून्य submit(vk::ArrayProxy<ध्रुव vk::SubmitInfo> submits, vk::Fence fence) override;

    // lock()/unlock() inherited no-ops
};

संरचना vk_queue {
    uint32_t queue_family_सूचकांक;
    std::shared_ptr<vk_queue_handle> handle;

    vk_command_pool cmd_pool;

    vk::PipelineStageFlags stage_flags;

    bool transfer_only;
};

स्थिर constexpr uint32_t mul_mat_vec_max_स्तम्भसङ्ख्या = 8;

स्थिर constexpr uint32_t p021_max_gqa_अनुपातः = 8;

गणना vk_device_architecture {
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

गणना vk_conv_shapes {
    CONV_SHAPE_128x128,
    CONV_SHAPE_64x32,
    CONV_SHAPE_32x256,
    CONV_SHAPE_64x128,
    CONV_SHAPE_गणनम्,
};

संरचना vk_conv_खण्डम्_आकार {
    uint32_t K;
    uint32_t NPQ;
    uint32_t CRS;
};

अन्तःस्थ vk_conv_खण्डम्_आकार vk_conv_खण्डम्_sizes[CONV_SHAPE_गणनम्] = {
    // K   NPQ  CRS
    { 128, 128, 16 }, // CONV_SHAPE_128x128
    {  64,  32, 32 }, // CONV_SHAPE_64x32
    {  32, 256, 16 }, // CONV_SHAPE_32x256
    {  64, 128, 16 }, // CONV_SHAPE_64x128
};

गणना dmmv_wg_sizes {
    DMMV_WG_आकार_SUBGROUP,
    DMMV_WG_आकार_LARGE,
    DMMV_WG_आकार_गणनम्,
};

गणना Fa_संहिता_मार्गः {
    FA_SCALAR,
    FA_COOPMAT1,
    FA_COOPMAT2,
};

संरचना vk_fa_pipeline_state {
    uint32_t HSK, HSV;
    uint32_t Br, Bc;
    uint32_t D_विभज, row_विभज;
    bool shmem_staging;
    Fa_संहिता_मार्गः मार्गः;
    uint32_t workgroup_आकार, subgroup_आकार;
    bool aligned;
    bool f32acc;
    uint32_t flags;
    uint32_t सीमा_occupancy_shmem;
    ggml_type k_type;
    ggml_type v_type;

    bool संक्रियकः<(ध्रुव vk_fa_pipeline_state &b) ध्रुव {
        प्रत्यागम std::tie(HSK, HSV, Br, Bc, D_विभज, row_विभज, shmem_staging, मार्गः, workgroup_आकार, subgroup_आकार, aligned, f32acc, flags, सीमा_occupancy_shmem, k_type, v_type) <
               std::tie(b.HSK, b.HSV, b.Br, b.Bc, b.D_विभज, b.row_विभज, b.shmem_staging, b.मार्गः, b.workgroup_आकार, b.subgroup_आकार, b.aligned, b.f32acc, b.flags, b.सीमा_occupancy_shmem, b.k_type, b.v_type);
    }
};

संरचना vk_conv2d_pipeline_state {
    vk_conv2d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t p0, uint32_t p1, uint32_t d0, uint32_t d1, uint32_t KW, uint32_t KH, uint32_t aligned)
        : s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), KW(KW), KH(KH), aligned(aligned) {}

    uint32_t s0, s1, p0, p1, d0, d1, KW, KH;
    // when set, shader can skip K/CRS/NPQ bounds checks and address clamps
    uint32_t aligned;

    bool संक्रियकः<(ध्रुव vk_conv2d_pipeline_state &b) ध्रुव {
        प्रत्यागम std::tie(s0, s1, p0, p1, d0, d1, KW, KH, aligned) <
               std::tie(b.s0, b.s1, b.p0, b.p1, b.d0, b.d1, b.KW, b.KH, b.aligned);
    }
};

संरचना vk_conv3d_pipeline_state {
    vk_conv3d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t p0, uint32_t p1, uint32_t p2,
                             uint32_t d0, uint32_t d1, uint32_t d2, uint32_t KW, uint32_t KH, uint32_t KD, uint32_t aligned)
        : s0(s0), s1(s1), s2(s2), p0(p0), p1(p1), p2(p2), d0(d0), d1(d1), d2(d2), KW(KW), KH(KH), KD(KD), aligned(aligned) {}

    uint32_t s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD;
    uint32_t aligned;

    bool संक्रियकः<(ध्रुव vk_conv3d_pipeline_state &b) ध्रुव {
        प्रत्यागम std::tie(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned) <
               std::tie(b.s0, b.s1, b.s2, b.p0, b.p1, b.p2, b.d0, b.d1, b.d2, b.KW, b.KH, b.KD, b.aligned);
    }
};

संरचना vk_solve_tri_pipeline_state {
    vk_solve_tri_pipeline_state(uint32_t N, uint32_t K)
        : N(N), K(K) {}

    uint32_t N, K;

    bool संक्रियकः<(ध्रुव vk_solve_tri_pipeline_state &b) ध्रुव {
        प्रत्यागम std::tie(N, K) <
               std::tie(b.N, b.K);
    }
};

गणना shader_reduction_बहुलकम् {
    SHADER_REDUCTION_बहुलकम्_SHMEM,
    SHADER_REDUCTION_बहुलकम्_HYBRID,
    SHADER_REDUCTION_बहुलकम्_SUBGROUP,
    SHADER_REDUCTION_बहुलकम्_गणनम्,
};

स्थिर constexpr uint32_t अङ्कः_argsort_pipelines = 11;

स्थिर constexpr uint32_t अङ्कः_topk_moe_pipelines = 10;

स्थिर constexpr uint32_t अङ्कः_topk_pipelines = 11;

स्थिर constexpr std::initializer_list<ggml_op> topk_moe_early_softmax_norm{ GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_पङ्क्तिकाः, GGML_OP_RESHAPE,
                                                                             GGML_OP_योग_पङ्क्तिकाः, GGML_OP_CLAMP,    GGML_OP_DIV,
                                                                             GGML_OP_RESHAPE };

स्थिर constexpr std::initializer_list<ggml_op> topk_moe_sigmoid_norm_bias{ GGML_OP_UNARY,    GGML_OP_RESHAPE,  GGML_OP_सङ्कलय,
                                                                            GGML_OP_ARGSORT,  GGML_OP_VIEW,     GGML_OP_GET_पङ्क्तिकाः,
                                                                            GGML_OP_RESHAPE,  GGML_OP_योग_पङ्क्तिकाः, GGML_OP_CLAMP,
                                                                            GGML_OP_DIV,      GGML_OP_RESHAPE };

स्थिर constexpr std::initializer_list<ggml_op> topk_moe_sqrt_softplus_norm_bias{ GGML_OP_UNARY,    GGML_OP_SQRT,
                                                                                  GGML_OP_RESHAPE,  GGML_OP_सङ्कलय,
                                                                                  GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                                  GGML_OP_GET_पङ्क्तिकाः, GGML_OP_RESHAPE,
                                                                                  GGML_OP_योग_पङ्क्तिकाः, GGML_OP_CLAMP,
                                                                                  GGML_OP_DIV,      GGML_OP_RESHAPE };

स्थिर constexpr std::initializer_list<ggml_op> topk_moe_early_softmax     { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_पङ्क्तिकाः };

स्थिर constexpr std::initializer_list<ggml_op> topk_moe_late_softmax      { GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                             GGML_OP_GET_पङ्क्तिकाः, GGML_OP_RESHAPE,
                                                                             GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

स्थिर constexpr std::initializer_list<ggml_op> snake_आदर्शः              { GGML_OP_MUL,      GGML_OP_SIN,
                                                                             GGML_OP_SQR,      GGML_OP_MUL,
                                                                             GGML_OP_सङ्कलय };

स्थिर constexpr std::initializer_list<ggml_op> topk_qsa_आदर्शः { GGML_OP_GET_पङ्क्तिकाः, GGML_OP_PERMUTE,
                                                                   GGML_OP_CONT,     GGML_OP_CPY,
                                                                   GGML_OP_RESHAPE,  GGML_OP_सङ्कलय,
                                                                   GGML_OP_शिखरम्_K };

स्थिर constexpr std::initializer_list<std::array<पूर्णाङ्क, 3>> topk_qsa_धाराः {
    { 1, 0, 0 }, // permute->src[0] == get_rows
    { 2, 0, 1 }, // cont->src[0]    == permute
    { 4, 0, 3 }, // reshape->src[0] == cpy (mask cast)
    { 5, 0, 2 }, // add->src[0]     == cont
    { 5, 1, 4 }, // add->src[1]     == reshape
    { 6, 0, 5 }, // top_k->src[0]   == add
};

स्थिर constexpr std::initializer_list<ggml_op> rms_norm_mul_सङ्कलय_mul_आदर्शः { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_सङ्कलय, GGML_OP_MUL };

स्थिर constexpr std::initializer_list<ggml_op> rms_norm_mul_सङ्कलय_आदर्शः     { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_सङ्कलय };

स्थिर constexpr std::initializer_list<ggml_op> rms_norm_mul_rope_view_समुच्चयः_पङ्क्तिकाः_आदर्शः { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_समुच्चयः_पङ्क्तिकाः };

स्थिर constexpr std::initializer_list<ggml_op> rms_norm_view_समुच्चयः_पङ्क्तिकाः_आदर्शः { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_समुच्चयः_पङ्क्तिकाः };

स्थिर constexpr std::initializer_list<ggml_op> rope_view_समुच्चयः_पङ्क्तिकाः_आदर्शः { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_समुच्चयः_पङ्क्तिकाः };

// scale_out*sigmoid(scale_in*x) as the hc_post weights (qwen4exp hc_combine)
स्थिर constexpr std::initializer_list<ggml_op> hc_post_gate_आदर्शः { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

स्थिर constexpr std::initializer_list<std::array<पूर्णाङ्क, 3>> hc_post_gate_धाराः {
    { 1, 0, 0 }, // sigmoid->src[0] == scale
    { 2, 0, 1 }, // scale->src[0]   == sigmoid
    { 3, 2, 2 }, // hc_post->src[2] == scale (post)
};

स्थिर constexpr std::initializer_list<std::array<पूर्णाङ्क, 3>> topk_moe_early_softmax_norm_धाराः {
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

स्थिर constexpr std::initializer_list<std::array<पूर्णाङ्क, 3>> topk_moe_sigmoid_norm_bias_धाराः {
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

स्थिर constexpr std::initializer_list<std::array<पूर्णाङ्क, 3>> topk_moe_sqrt_softplus_norm_bias_धाराः {
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

स्थिर constexpr std::initializer_list<std::array<पूर्णाङ्क, 3>> topk_moe_early_softmax_धाराः {
    { 1, 0, 0 }, // reshape->src[0]  == softmax
    { 2, 0, 0 }, // argsort->src[0]  == softmax
    { 3, 0, 2 }, // view->src[0]     == argsort
    { 4, 0, 1 }, // get_rows->src[0] == reshape
    { 4, 1, 3 }, // get_rows->src[1] == view
};

स्थिर constexpr std::initializer_list<std::array<पूर्णाङ्क, 3>> topk_moe_late_softmax_धाराः {
    { 1, 0, 0 }, // view->src[0]     == argsort
    { 2, 1, 1 }, // get_rows->src[1] == view
    { 3, 0, 2 }, // reshape->src[0]  == get_rows
    { 4, 0, 3 }, // soft_max->src[0] == reshape
    { 5, 0, 4 }, // reshape->src[0]  == soft_max
};

गणना topk_moe_बहुलकम् {
    TOPK_MOE_EARLY_SOFTMAX,
    TOPK_MOE_EARLY_SOFTMAX_NORM,
    TOPK_MOE_LATE_SOFTMAX,
    TOPK_MOE_SIGMOID_NORM_BIAS,
    TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS,
    TOPK_MOE_गणनम्,
};

गणना rms_norm_बहुलकम् {
    RMS_NORM_MUL,
    RMS_NORM_MUL_सङ्कलय,
    RMS_NORM_MUL_सङ्कलय_MUL,
    RMS_NORM_MUL_ROPE,
    RMS_NORM_MUL_ROPE_VIEW_समुच्चयः_पङ्क्तिकाः,
    RMS_NORM_VIEW_समुच्चयः_पङ्क्तिकाः,
    RMS_NORM_गणनम्,
};

स्थिर constexpr std::initializer_list<std::array<पूर्णाङ्क, 3>> rope_view_समुच्चयः_पङ्क्तिकाः_धाराः {
    { 1, 0, 0 }, // view->src[0]     == rope
    { 2, 0, 1 }, // set_rows->src[0] == view
};

स्थिर constexpr std::initializer_list<std::array<पूर्णाङ्क, 3>> rms_norm_mul_rope_view_समुच्चयः_पङ्क्तिकाः_धाराः {
    { 1, 0, 0 }, // mul->src[0]      == rms
    { 2, 0, 1 }, // rope->src[0]     == mul
    { 3, 0, 2 }, // view->src[0]     == rope
    { 4, 0, 3 }, // set_rows->src[0] == view
};

स्थिर constexpr std::initializer_list<std::array<पूर्णाङ्क, 3>> rms_norm_view_समुच्चयः_पङ्क्तिकाः_धाराः {
    { 1, 0, 0 }, // view->src[0]     == rms_norm
    { 2, 0, 1 }, // set_rows->src[0] == view
};

स्थिर constexpr std::array<ggml_type, 9> lightning_indexer_k_types = {
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

class vk_स्मृतिः_logger;

संरचना vk_device_struct {
    std::recursive_mutex mutex;
    std::mutex queue_submit_mutex;

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
    uint64_t max_स्मृतिः_आवण्टनम्_आकार;
    uint64_t max_बफर_आकार;
    uint64_t suballocation_खण्डम्_आकार;
    uint64_t min_imported_host_pointer_alignment;
    bool external_स्मृतिः_host {};
    bool fp16;
    bool bf16;
    bool pipeline_robustness;
    bool स्मृतिः_प्राथमिकता;
    vk::Device device;
    uint32_t vendor_id;
    vk::DriverId driver_id;
    vk_device_architecture architecture;
    std::unique_ptr<vk_queue> compute_queue;
    std::unique_ptr<vk_queue> transfer_queue;
    bool single_queue;
    bool support_async;
    bool async_use_transfer_queue;
    bool has_internally_synchronized_queues = false;
    uint32_t subgroup_आकार;
    uint32_t subgroup_आकार_log2;
    uint32_t shader_core_गणनम्;
    bool uma;
    bool prefer_host_स्मृतिः;
    bool float_controls_rte_fp16;
    bool float_controls_denorm_preserve_fp16;
    bool subgroup_basic;
    bool subgroup_arithmetic;
    bool subgroup_shuffle;
    bool subgroup_ballot;
    bool subgroup_clustered;
    bool subgroup_vote;
    bool multi_सङ्कलय;
    bool shader_int64;
    bool बफर_device_स्थानाङ्कः;
    bool vulkan_स्मृतिः_model;

    bool सङ्कलय_rms_fusion;
    uint32_t partials_binding_alignment;
    uint32_t max_nodes_प्रत्येकम्_submit;

    bool shader_64b_indexing;

    bool integer_dot_गुणनफलम्;
    // 0: default, 1: force mmvq, -1: disable mmvq
    int32_t mmvq_बहुलकम्;

    bool subgroup_आकार_control;
    uint32_t subgroup_min_आकार;
    uint32_t subgroup_max_आकार;
    bool subgroup_require_पूर्णम्_support;

    // floor(log2(maxComputeWorkGroupInvocations))
    uint32_t max_workgroup_आकार_log2 {};

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
    bool coopmat2_decode_सदिशम्;

    bool dot2_f16 {};
    bool ocp_fp4 {};

    bool pipeline_executable_properties_support {};

    bool device_fault {};
    PFN_vkGetDeviceFaultInfoEXT pfn_vkGetDeviceFaultInfoEXT {};

    bool serialize_submissions {};

    ध्रुव ggml_cgraph * diag_cgraph {};
    पूर्णाङ्क diag_पूर्व_start = -1;
    पूर्णाङ्क diag_पूर्व_अन्तः = -1;

    size_t idx;

    bool mul_mat_l[GGML_TYPE_गणनम्];
    bool mul_mat_m[GGML_TYPE_गणनम्];
    bool mul_mat_s[GGML_TYPE_गणनम्];
    bool mul_mat_id_l[GGML_TYPE_गणनम्];
    bool mul_mat_id_m[GGML_TYPE_गणनम्];
    bool mul_mat_id_s[GGML_TYPE_गणनम्];

    // Separate flags for the q8_1 (integer dot) mmq path, whose shader uses
    // a different shared-memory layout than the float matmul shaders.
    bool mul_mat_l_int[GGML_TYPE_गणनम्];
    bool mul_mat_m_int[GGML_TYPE_गणनम्];
    bool mul_mat_s_int[GGML_TYPE_गणनम्];
    bool mul_mat_id_l_int[GGML_TYPE_गणनम्];
    bool mul_mat_id_m_int[GGML_TYPE_गणनम्];
    bool mul_mat_id_s_int[GGML_TYPE_गणनम्];

    vk::Descriptor_समुच्चयः_Layout dsl;

    std::map<vk_matmul_pipeline_कुञ्चिका, std::सदिशम्<vk_matmul_pipeline_pair>> pipeline_matmul;
    matmul_tile_selector_t matmul_tile_selector;
    matmul_tile_selector_t matmul_id_tile_selector;

    vk_pipeline pipeline_matmul_विभज_k_reduce;
    vk_pipeline pipeline_quantize_q8_1_x4;

    vk_pipeline pipeline_dequant[GGML_TYPE_गणनम्];
    vk_pipeline pipeline_dequant_transpose[GGML_TYPE_गणनम्]; // fused dequant+transpose for FA quant-KV
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_आकार_गणनम्][GGML_TYPE_गणनम्][mul_mat_vec_max_स्तम्भसङ्ख्या];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[DMMV_WG_आकार_गणनम्][GGML_TYPE_गणनम्][mul_mat_vec_max_स्तम्भसङ्ख्या];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[DMMV_WG_आकार_गणनम्][GGML_TYPE_गणनम्];

    vk_pipeline pipeline_dequant_mul_mat_vec_q8_1_f32[DMMV_WG_आकार_गणनम्][GGML_TYPE_गणनम्][mul_mat_vec_max_स्तम्भसङ्ख्या];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_q8_1_f32[DMMV_WG_आकार_गणनम्][GGML_TYPE_गणनम्];

    vk_pipeline pipeline_mul_mat_vec_p021_f16_f32[p021_max_gqa_अनुपातः];
    vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
    vk_pipeline pipeline_get_पङ्क्तिकाः[GGML_TYPE_गणनम्];
    vk_pipeline pipeline_get_पङ्क्तिकाः_f32[GGML_TYPE_गणनम्];
    vk_pipeline pipeline_get_पङ्क्तिकाः_पृष्ठभागः_f32;
    vk_pipeline pipeline_acc_f32;
    vk_pipeline pipeline_समुच्चयः_f32;

    // [src0 0=fp32,1=fp16][src1 0=fp32,1=fp16][dst 0=fp32,1=fp16]
    vk_pipeline pipeline_सङ्कलय[2][2][2];
    vk_pipeline pipeline_सङ्कलय_norepeat[2][2][2];
    vk_pipeline pipeline_sub[2][2][2];
    vk_pipeline pipeline_sub_norepeat[2][2][2];
    vk_pipeline pipeline_mul[2][2][2];
    vk_pipeline pipeline_mul_norepeat[2][2][2];
    vk_pipeline pipeline_div[2][2][2];
    vk_pipeline pipeline_div_norepeat[2][2][2];
    vk_pipeline pipeline_सङ्कलय_rms[2][2][2];
    vk_pipeline pipeline_सङ्कलय_rms_norepeat[2][2][2];

    // indexed by num_additional_fused_ops == num_adds - 1
    vk_pipeline pipeline_multi_सङ्कलय[MAX_FUSED_ADDS];
    vk_pipeline pipeline_multi_सङ्कलय_rms[MAX_FUSED_ADDS];

    vk_pipeline pipeline_सङ्कलय_id_f32;

    vk_pipeline pipeline_concat_i8, pipeline_concat_i16, pipeline_concat_i32, pipeline_concat_i64;
    vk_pipeline pipeline_upscale_निकटतमः_f32, pipeline_upscale_bilinear_f32, pipeline_upscale_bicubic_f32, pipeline_upscale_bilinear_antialias_f32;
    vk_pipeline pipeline_scale_f32;
    vk_pipeline pipeline_log[2];
    vk_pipeline pipeline_tri[2];
    vk_pipeline pipeline_diag[2];
    vk_pipeline pipeline_clamp[2];
    vk_pipeline pipeline_pad_f32;
    vk_pipeline pipeline_pad_reflect_1d_f32;
    vk_pipeline pipeline_roll_f32;
    vk_pipeline pipeline_repeat_i32, pipeline_repeat_पृष्ठभागः_f32;
    vk_pipeline pipeline_repeat_i16;
    vk_pipeline pipeline_cpy_32, pipeline_cpy_f32_f16, pipeline_cpy_16, pipeline_cpy_f16_f32, pipeline_cpy_f32_bf16, pipeline_cpy_bf16_f32, pipeline_cpy_bf16_f16, pipeline_cpy_f32_i32, pipeline_cpy_i32_f32;
    vk_pipeline pipeline_contig_cpy_32, pipeline_contig_cpy_f32_f16, pipeline_contig_cpy_16, pipeline_contig_cpy_f16_f32, pipeline_contig_cpy_f32_bf16, pipeline_contig_cpy_bf16_f32, pipeline_contig_cpy_bf16_f16, pipeline_contig_cpy_f32_i32, pipeline_contig_cpy_i32_f32;
    vk_pipeline pipeline_cpy_f32_quant[GGML_TYPE_गणनम्];
    vk_pipeline pipeline_cpy_quant_f32[GGML_TYPE_गणनम्];
    vk_pipeline pipeline_cpy_transpose_16, pipeline_cpy_transpose_32;
    vk_pipeline pipeline_cpy_transpose_02_16, pipeline_cpy_transpose_02_32;
    // [src0 0=fp32,1=fp16][dst]
    vk_pipeline pipeline_समुच्चयः_पङ्क्तिकाः_i32[2][GGML_TYPE_गणनम्];
    vk_pipeline pipeline_समुच्चयः_पङ्क्तिकाः_i64[2][GGML_TYPE_गणनम्];
    vk_pipeline pipeline_norm_f32;
    vk_pipeline pipeline_गणः_norm_f32;
    vk_pipeline pipeline_rms_norm_f32;
    vk_pipeline pipeline_rms_norm_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_सङ्कलय_f32;
    vk_pipeline pipeline_rms_norm_mul_सङ्कलय_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_सङ्कलय_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_सङ्कलय_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_समुच्चयः_पङ्क्तिकाः_f32_f32;
    vk_pipeline pipeline_rms_norm_समुच्चयः_पङ्क्तिकाः_f32_f16;
    vk_pipeline pipeline_rms_norm_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f16;
    vk_pipeline pipeline_rms_norm_पृष्ठभागः_f32;
    vk_pipeline pipeline_l2_norm_f32;

    // [src/dst 0=fp32,1=fp16]
    vk_pipeline pipeline_exp[2];
    vk_pipeline pipeline_expm1[2];
    vk_pipeline pipeline_elu[2];
    vk_pipeline pipeline_gelu[2];
    vk_pipeline pipeline_gelu_erf[2];
    vk_pipeline pipeline_gelu_शीघ्र[2];
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
    vk_pipeline pipeline_चरणः[2];
    vk_pipeline pipeline_आवर्तनम्[2];
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
    vk_pipeline pipeline_geglu_शीघ्र[2];

    vk_pipeline pipeline_leaky_relu[2];
    vk_pipeline pipeline_silu_पृष्ठभागः_f32;
    vk_pipeline pipeline_diag_आवरणम्_inf_f32;
    vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
    vk_pipeline pipeline_soft_max_f32_wg512, pipeline_soft_max_f32_f16_wg512;
    vk_pipeline pipeline_soft_max_पृष्ठभागः_f32;

    vk_pipeline pipeline_soft_max_large1_f32, pipeline_soft_max_large1_f32_f16;
    vk_pipeline pipeline_soft_max_large2_f32, pipeline_soft_max_large2_f32_f16;
    vk_pipeline pipeline_soft_max_large3_f32, pipeline_soft_max_large3_f32_f16;

    vk_pipeline pipeline_rope_norm_f32, pipeline_rope_norm_f16, pipeline_rope_norm_f32_f16;
    vk_pipeline pipeline_rope_neox_f32, pipeline_rope_neox_f16, pipeline_rope_neox_f32_f16;
    vk_pipeline pipeline_rope_multi_f32, pipeline_rope_multi_f16, pipeline_rope_multi_f32_f16;
    vk_pipeline pipeline_rope_vision_f32, pipeline_rope_vision_f16;
    vk_pipeline pipeline_argsort_f32[अङ्कः_argsort_pipelines];
    vk_pipeline pipeline_argsort_large_f32[अङ्कः_argsort_pipelines];
    vk_pipeline pipeline_topk_f32[अङ्कः_topk_pipelines];
    vk_pipeline pipeline_topk_radix_f32;
    vk_pipeline pipeline_topk_radix_qsa; // qwen4 QSA indexer fusion (f16 mask)
    vk_pipeline pipeline_योग_पङ्क्तिकाः_f32;
    vk_pipeline pipeline_cross_entropy_loss_f32, pipeline_cross_entropy_loss_f32_wg512;
    vk_pipeline pipeline_cross_entropy_loss_पृष्ठभागः_f32, pipeline_cross_entropy_loss_पृष्ठभागः_f32_wg512;
    vk_pipeline pipeline_fwht_f32[4];
    vk_pipeline pipeline_cumsum_f32;
    vk_pipeline pipeline_cumsum_small_f32;
    vk_pipeline pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline_cumsum_multipass2_f32;
    vk_pipeline pipeline_argmax_f32;
    vk_pipeline pipeline_गणनम्_equal_i32;
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
    vk_pipeline pipeline_बहिर्गमनम्_prod_f32;
    vk_pipeline pipeline_snake_f32;
    vk_pipeline pipeline_snake_f16;
    vk_pipeline pipeline_snake_bf16;
    vk_pipeline pipeline_pool1d_f32;
    vk_pipeline pipeline_pool2d_f32;
    vk_pipeline pipeline_pool2d_subgroup_f32;
    vk_pipeline pipeline_rwkv_wkv6_f32;
    vk_pipeline pipeline_rwkv_wkv7_f32;
    vk_pipeline pipeline_gated_रैखिकम्_attn_f32;
    vk_pipeline pipeline_lightning_indexer_f32[GGML_TYPE_गणनम्];
    // [size_idx][kda] where size_idx: 0=d16, 1=d32, 2=d64, 3=d128
    vk_pipeline pipeline_gated_delta_net[4][2];
    vk_pipeline pipeline_ssm_अवलोकनम्_f32_d128;
    vk_pipeline pipeline_ssm_अवलोकनम्_f32_d256;
    vk_pipeline pipeline_ssm_conv_f32;
    vk_pipeline pipeline_ssm_conv_silu_f32;
    vk_pipeline pipeline_ssm_conv_bias_silu_f32;
    vk_pipeline pipeline_opt_चरणः_adamw_f32;
    vk_pipeline pipeline_opt_चरणः_sgd_f32;
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f32[CONV_SHAPE_गणनम्];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f16_f32[CONV_SHAPE_गणनम्];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f32[CONV_SHAPE_गणनम्];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f16_f32[CONV_SHAPE_गणनम्];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f32[CONV_SHAPE_गणनम्];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f16_f32[CONV_SHAPE_गणनम्];
    vk_pipeline pipeline_conv2d_dw_whcn_f32, pipeline_conv2d_dw_whcn_f16_f32;
    vk_pipeline pipeline_conv2d_dw_cwhn_f32, pipeline_conv2d_dw_cwhn_f16_f32;

    std::map<vk_fa_pipeline_state, vk_pipeline> pipeline_flash_attn_f32_f16;

    std::map<std::pair<uint32_t, uint32_t>, vk_pipeline> pipeline_fa_आवरणम्_opt;

    vk_pipeline pipeline_fa_sparse_compact;
    vk_pipeline pipeline_fa_sparse_compact_subgroup;
    bool fa_sparse_compact_use_subgroups;

    vk_pipeline pipeline_flash_attn_विभज_k_reduce;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, std::pair<vk_pipeline, vk_pipeline>> pipeline_xe_fa_decode_dual_phases;
    vk_pipeline pipeline_गणनम्_experts;

    // [2] is for whether to take n_experts from spec constant (0) or push constant (1)
    vk_pipeline pipeline_topk_moe[अङ्कः_topk_moe_pipelines][2];

    std::सदिशम्<vk_pipeline_ref> all_pipelines;

    vk::Fence fence;
    vk_बफर sync_staging;

    ggml_backend_बफर_type बफर_type;

    bool disable_fusion;
    bool disable_descriptor_reuse;
    std::atomic<uint64_t> बफर_destroy_गणनम् {};
    bool disable_host_visible_vidmem;
    bool allow_sysmem_fallback;
    bool disable_graph_optimize;

    std::unique_ptr<vk_स्मृतिः_logger> स्मृतिः_logger;

    ~vk_device_struct();

};

अन्तःस्थ शून्य vk_command_pool::init(vk_device& device, vk_queue *q_) {
    cmd_buffers.clear();
    q = q_;

    vk::CommandPoolCreateInfo command_pool_create_info(
        vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_बफर_BIT),
        q->queue_family_सूचकांक);
    pool = device->device.createCommandPool(command_pool_create_info);
}

अन्तःस्थ शून्य vk_command_pool::destroy(vk::Device& device) {
    device.destroyCommandPool(pool);
    pool = nullptr;
    cmd_buffers.clear();
}

संरचना vk_बफर_struct {
    vk::बफर_2 बफर = VK_NULL_HANDLE;
    vk::Device_स्मृतिः device_स्मृतिः = VK_NULL_HANDLE;
    vk::स्मृतिः_Property_Flags स्मृतिः_property_flags;
    शून्य * ptr;
    size_t आकार = 0;
    vk::Device_स्थानाङ्कः bda_addr {};

    vk_device device;

    ~vk_बफर_struct() {
        यदि (आकार == 0) {
            प्रत्यागम;
        }
        VK_LOG_DEBUG("~vk_buffer_struct(" << बफर << ", " << आकार << ")");

        // bump before destroying, so a thread that sees the buffer gone also sees the new count
        device->बफर_destroy_गणनम्.fetch_सङ्कलय(1, std::memory_order_release);
        device->device.free_स्मृतिः_2(device_स्मृतिः);
        device->device.destroy_बफर(बफर);
    }
};

संरचना vk_subbuffer {
    vk_बफर बफर;
    uint64_t विस्थापनम्;
    uint64_t आकार;

    संक्रियकः vk::Descriptor_बफर_Info() ध्रुव {
        प्रत्यागम { बफर->बफर, विस्थापनम्, आकार };
    }
};

संरचना vk_semaphore {
    vk::Semaphore s;
    uint64_t मानम्;
};

संरचना vk_event {
    std::सदिशम्<vk::Event> events_free; // Events available for reuse
    std::सदिशम्<vk::Event> events_submitted; // Events that are fully submitted and can be reused on next synchronize
    vk::Event event;
    bool has_event;

    vk_semaphore tl_semaphore;
    vk_command_बफर* cmd_बफर = nullptr;
    uint64_t cmd_बफर_use_गणक = 0;
};

संरचना vk_submission {
    vk_command_बफर* बफर = nullptr;
    std::सदिशम्<vk_semaphore> wait_semaphores;
    std::सदिशम्<vk_semaphore> signal_semaphores;
};

प्रकारनिर्देश std::सदिशम्<vk_submission> vk_sequence;

#परिभाषय MAT_VEC_FUSION_FLAGS_BIAS0 0x1

#परिभाषय MAT_VEC_FUSION_FLAGS_BIAS1 0x2

#परिभाषय MAT_VEC_FUSION_FLAGS_SCALE0 0x4

#परिभाषय MAT_VEC_FUSION_FLAGS_SCALE1 0x8

संरचना vk_staging_memcpy {
    vk_staging_memcpy(शून्य * _dst, ध्रुव शून्य * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    शून्य * dst;
    ध्रुव शून्य * src;
    size_t n;
};

संरचना vk_staging_memset {
    vk_staging_memset(शून्य * _dst, uint32_t _val, size_t _n) : dst(_dst), val(_val), n(_n) {}

    शून्य * dst;
    uint32_t val;
    size_t n;
};

संरचना vk_context_struct {
    vk_submission * s;
    std::सदिशम्<vk_sequence> seqs;

    पूर्णाङ्क exit_tensor_idx;

    std::सदिशम्<vk_staging_memcpy> अन्तर्_memcpys;
    std::सदिशम्<vk_staging_memcpy> बहिर्गमनम्_memcpys;
    std::सदिशम्<vk_staging_memset> memsets;

    std::सदिशम्<std::string> debug_labels;

    vk_command_pool * p {};
};

प्रकारनिर्देश std::shared_ptr<vk_context_struct> vk_context;

प्रकारनिर्देश std::weak_ptr<vk_context_struct> vk_context_ref;

संरचना ggml_vk_garbage_collector {
    std::सदिशम्<vk_semaphore> tl_semaphores;
    std::सदिशम्<vk_semaphore> semaphores;
    std::सदिशम्<vk::Event> events;
    std::सदिशम्<vk_context> contexts;
};

#परिभाषय VK_LOG_स्मृतिः(सन्देशः_2) यदि (vk_स्मृतिः_logger_enabled) { std::cerr << "ggml_vulkan memory: " << सन्देशः_2 << std::endl; }

स्थिर std::string format_आकार(size_t आकार) {
    ध्रुव size_t kib = 1024;
    ध्रुव size_t mib = kib * 1024;
    ध्रुव size_t gib = mib * 1024;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    यदि (आकार >= gib) {
        oss << static_cast<द्विगुण>(आकार) / gib << " GiB";
    } अन्यथा यदि (आकार >= mib) {
        oss << static_cast<द्विगुण>(आकार) / mib << " MiB";
    } अन्यथा यदि (आकार >= kib) {
        oss << static_cast<द्विगुण>(आकार) / kib << " KiB";
    } अन्यथा {
        oss << आकार << " B";
    }

    प्रत्यागम oss.str();
}

class vk_स्मृतिः_logger {
सार्वजनिक:
    vk_स्मृतिः_logger(): योग_device(0), योग_host(0) {}
    शून्य log_आवण्टनम्(vk_बफर_ref buf_ref, size_t आकार);
    शून्य log_deallocation(vk_बफर_ref buf_ref);

गुप्त:
    std::map<vk::बफर_2, size_t> allocations; // Track allocations
    size_t योग_device;
    size_t योग_host;
    स्थिर std::mutex log_mutex;
};

अन्तःस्थ std::mutex vk_स्मृतिः_logger::log_mutex;

class vk_perf_logger {
  सार्वजनिक:
    शून्य print_timings(bool force = false);


    std::string get_ग्रन्थिः_fusion_name(ध्रुव ggml_tensor * ग्रन्थिः, ध्रुव वर्ण *fusion_name, uint64_t *n_flops);


    शून्य log_timing(ध्रुव ggml_tensor * ग्रन्थिः, ध्रुव वर्ण *fusion_name, uint64_t time) {
        uint64_t n_flops;
        std::string name = get_ग्रन्थिः_fusion_name(ग्रन्थिः, fusion_name, &n_flops);
        यदि (n_flops) {
            flops[name].प्रक्षिप_पृष्ठभागः(n_flops);
        }
        timings[name].प्रक्षिप_पृष्ठभागः(time);
    }

    शून्य log_timing(ध्रुव std::सदिशम्<ggml_tensor *> &nodes, ध्रुव std::सदिशम्<ध्रुव वर्ण *> &नामानि, uint64_t time) {
        uint64_t योग_flops = 0;
        std::string name;
        प्रति (size_t n = 0; n < nodes.आकार(); ++n) {
            uint64_t n_flops = 0;
            name += get_ग्रन्थिः_fusion_name(nodes[n], नामानि[n], &n_flops);
            योग_flops += n_flops;

            यदि (n != nodes.आकार() - 1) {
                name += ", ";
            }
        }
        यदि (योग_flops) {
            flops[name].प्रक्षिप_पृष्ठभागः(योग_flops);
        }
        timings[name].प्रक्षिप_पृष्ठभागः(time);
    }

  गुप्त:
    std::map<std::string, std::सदिशम्<uint64_t>> timings;
    std::map<std::string, std::सदिशम्<uint64_t>> flops;
    uint32_t print_गणनम् {};
};

संरचना ggml_backend_vk_context {
    std::string name;

    vk_device device;

    size_t semaphore_idx, event_idx;
    ggml_vk_garbage_collector gc;
    size_t prealloc_आकार_x, prealloc_आकार_y, prealloc_आकार_विभज_k, prealloc_आकार_सङ्कलय_rms_partials, prealloc_आकार_सङ्कलय_rms_partials_विस्थापनम्;
    vk_बफर prealloc_x, prealloc_y, prealloc_विभज_k, prealloc_सङ्कलय_rms_partials, sync_staging;
    vk::Fence fence, almost_ready_fence;
    bool submit_pending {};
    bool almost_ready_fence_pending {};
    // Set before op_add and unset after op_rms_norm to indicate that the add should
    // write partial sums to accumulate the square of the vector components
    bool do_सङ्कलय_rms_partials_विस्थापनम्_calculation;
    bool do_सङ्कलय_rms_partials;

    uint64_t चरमः_योग_flops {UINT64_MAX};

    // Cache most recent tensor that was converted into prealloc_y, and what pipeline it used to convert.
    vk_pipeline_struct * prealloc_y_चरमः_pipeline_प्रयुक्तम् {};
    ध्रुव ggml_tensor * prealloc_y_चरमः_tensor_प्रयुक्तम् {};
    // True when the K dimension in prealloc_y is padded.
    bool prealloc_y_चरमः_k_padded {};

    // Track which nodes have been used since the last sync, and whether they were written to
    std::सदिशम्<ध्रुव ggml_tensor *> unsynced_nodes_written;
    std::सदिशम्<ध्रुव ggml_tensor *> unsynced_nodes_read;
    // Track which prealloc buffers have pending reads that need to be synchronized.
    // These are checked before writing to the buffer (and call ggml_vk_sync_buffers if set),
    // and set to true after the buffer contents are consumed.
    bool prealloc_x_need_sync, prealloc_y_need_sync, prealloc_विभज_k_need_sync;

    vk_context_ref compute_ctx;

    vk_context_ref transfer_ctx;
    vk_semaphore transfer_semaphore;
    uint64_t transfer_semaphore_चरमः_submitted {};

    std::सदिशम्<vk_context_ref> tensor_ctxs;

    std::सदिशम्<vk::DescriptorPool> descriptor_pools;
    std::सदिशम्<vk::Descriptor_समुच्चयः> descriptor_sets;
    // last bindings written to each set; descriptor_sets is append-only so an index always names the same set
    std::सदिशम्<std::सदिशम्<vk::Descriptor_बफर_Info>> descriptor_समुच्चयः_bindings;
    uint64_t descriptor_समुच्चयः_bindings_destroy_गणनम् {};
    uint32_t descriptor_समुच्चयः_idx {};
    uint32_t pipeline_descriptor_समुच्चयः_requirements {};

    vk_command_pool compute_cmd_pool;
    vk_command_pool transfer_cmd_pool;

    // number of additional consecutive nodes that are being fused with the
    // node currently being processed
    पूर्णाङ्क अङ्कः_additional_fused_ops {};
    // Bitmask of which fused ops need to write an intermediate value to memory.
    // Bit 'i' means nodes[start_of_fusion + i] writes to memory.
    // If there's no fusion, bit 0 is still set.
    पूर्णाङ्क fused_ops_write_आवरणम् {};
    topk_moe_बहुलकम् fused_topk_moe_बहुलकम् {};
    bool fused_topk_moe_scale {};
    // QSA indexer gather+add+top_k fused into one radix-select
    bool fused_topk_qsa {};
    bool fused_hc_post_gate {};
    rms_norm_बहुलकम् fused_rms_norm_बहुलकम् {RMS_NORM_गणनम्};

    // for GGML_VK_PERF_LOGGER
    std::unique_ptr<vk_perf_logger> perf_logger;
    vk::QueryPool query_pool;
    std::सदिशम्<ध्रुव वर्ण *> query_fusion_नामानि;
    std::सदिशम्<पूर्णाङ्क> query_fusion_ग्रन्थिः_गणनम्;
    std::सदिशम्<ggml_tensor *> query_nodes;
    std::सदिशम्<पूर्णाङ्क> query_ग्रन्थिः_idx;
    int32_t अङ्कः_queries {};
    int32_t query_idx {};
};

संरचना ggml_backend_vk_बफर_context {
    vk_device_ref device;
    vk_बफर dev_बफर;
    std::string name;

    ggml_backend_vk_बफर_context(vk_device_ref device, vk_बफर&& dev_बफर, std::string& name) :
        device(device),
        dev_बफर(dev_बफर),
        name(name) {
    }

    ~ggml_backend_vk_बफर_context();

};

संरचना vk_host_आवण्टनम् {
    शून्य *    आधारः = nullptr;
    size_t    आकार = 0;
    vk_बफर views[GGML_VK_MAX_DEVICES];  // indexed by device->idx; null = not (yet) mapped here
};

संरचना vk_instance_t {
    vk::Instance instance;

    bool debug_utils_support = false;  // VK_EXT_debug_utils enabled
    PFN_vk_समुच्चयः_Debug_Utils_Object_Name_EXT pfn_vk_समुच्चयः_Debug_Utils_Object_Name_EXT = {};
    PFN_vkQueueBeginDebugUtilsLabelEXT pfn_vkQueueBeginDebugUtilsLabelEXT = {};
    PFN_vk_Queue_अन्तः_Debug_Utils_Label_EXT   pfn_vk_Queue_अन्तः_Debug_Utils_Label_EXT   = {};
    PFN_vkCmdBeginDebugUtilsLabelEXT   pfn_vkCmdBeginDebugUtilsLabelEXT   = {};
    PFN_vk_Cmd_अन्तः_Debug_Utils_Label_EXT pfn_vk_Cmd_अन्तः_Debug_Utils_Label_EXT = {};
    PFN_vk_Cmd_निवेशय_Debug_Utils_Label_EXT  pfn_vk_Cmd_निवेशय_Debug_Utils_Label_EXT  = {};

    std::सदिशम्<size_t> device_indices;
    std::सदिशम्<bool>   device_supports_membudget;
    std::सदिशम्<bool>   device_host_बफर_import;
    vk_device devices[GGML_VK_MAX_DEVICES];

    std::सदिशम्<vk_host_आवण्टनम्> host_allocations;
    std::shared_mutex               host_allocations_mutex;

    size_t host_import_alignment = 0;
    uint32_t host_import_failed = 0;  // bitmask by device->idx, guarded by host_allocations_mutex

    bool multi_device = false;
};

प्रकारनिर्देश शून्य (*ggml_vk_func_t)(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);

स्थिर constexpr uint32_t k_Spv_Op_Cooperative_Matrix_आरोपय_Tensor_NV = 5367;

स्थिर constexpr uint32_t k_Spv_Capability_Cooperative_Matrix_Decode_सदिशम्_NV = 5447;

स्थिर constexpr uint32_t k_Spv_Tensor_Addressing_Decode_सदिशम्_Func_Bit = 0x4;

संरचना vk_fa_tuning_params {
    Fa_संहिता_मार्गः मार्गः;
    uint32_t workgroup_आकार;
    uint32_t subgroup_आकार;
    uint32_t खण्डम्_पङ्क्तिकाः;
    uint32_t खण्डम्_स्तम्भसङ्ख्या;
    uint32_t d_विभज;
    uint32_t row_विभज;
    bool shmem_staging;
    bool disable_subgroups;
    uint32_t सीमा_occupancy_shmem;

    शून्य print() ध्रुव {
        std::cerr << "path=" << मार्गः << " workgroup_size=" << workgroup_आकार << " subgroup_size=" << subgroup_आकार <<
                     " block_rows=" << खण्डम्_पङ्क्तिकाः << " block_cols=" << खण्डम्_स्तम्भसङ्ख्या << " d_split=" << d_विभज <<
                     " row_split=" << row_विभज << " shmem_staging=" << shmem_staging << " disable_subgroups=" << disable_subgroups <<
                     " limit_occupancy_shmem=" << सीमा_occupancy_shmem << std::endl;
    }
};

संरचना GpuPipelineConfig {
    // GPU architecture identifier.
    // Example: vk_device_architecture::AMD_GCN
    vk_device_architecture arch;

    // Mapping of pipeline names to their specific subgroup sizes.
    // Example: {"soft_max_f32", 64}
    std::unordered_map<std::string, uint32_t> pipelines;

    // Default subgroup size for this GPU.
    // Defaults to 0 if not explicitly provided.
    uint32_t default_subgroup_आकार = 0;
};

स्थिर constexpr uint32_t RDNA_DEFAULT_SUBGROUP_आकार = 32;

संरचना CompileTask {
    vk_pipeline pipeline;
    size_t spv_आकार;
    ध्रुव शून्य * spv_दत्त;
    std::string entrypoint;
    uint32_t parameter_गणनम्;
    std::array<uint32_t, 3> wg_denoms;
    std::सदिशम्<uint32_t> specialization_constants;
    bool disable_robustness;
    bool require_पूर्णम्_subgroups;
    uint32_t required_subgroup_आकार;
};

संरचना ggml_vk_debug_label {
    // at most one of these is set, depending on the scope the label was opened in
    vk_context_struct * subctx {};
    vk_queue_handle *   qhandle {};

    // one region per dispatch, e.g. "matmul_q4_k_f32_f16acc_aligned_m (192,8,1)".
    // RGP cannot recover the pipeline name on its own, it only has the hash
    ggml_vk_debug_label(vk_context & ctx, ध्रुव std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2);


    // one region per graph node
    // fused nodes are joined with '+', e.g. "RMS_NORM+MUL+ROPE Qcur-19"
    ggml_vk_debug_label(vk_context & ctx, ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, पूर्णाङ्क n_fused);


    // one region per graph evaluation, opened on the queue instead of a command buffer
    // so it spans every submit the evaluation makes
    ggml_vk_debug_label(vk_queue_handle * handle, ध्रुव वर्ण * name);


    // call before the command buffer can end, the destructor covers the rest
    शून्य close();


    ~ggml_vk_debug_label() {
        close();
    }

    ggml_vk_debug_label(ध्रुव ggml_vk_debug_label &) = delete;
    ggml_vk_debug_label & संक्रियकः=(ध्रुव ggml_vk_debug_label &) = delete;

गुप्त:
    // the constructors check this too, so the name is not built when markers are off
    शून्य begin(vk_context & ctx, ध्रुव std::string & name);

};

#परिभाषय अप्रयुक्तम् GGML_अप्रयुक्तम्

संरचना ggml_backend_vk_device_context {
    size_t device;
    std::string name;
    std::string description;
    bool is_integrated_gpu;
    std::string pci_bus_id;
    पूर्णाङ्क op_offload_min_batch_आकार;
};
