#لارښوونه once

#ضمن "ggml-vulkan.h"

#ضمن <vulkan/vulkan_core.h>

#إذا defined(GGML_VULKAN_پرلپسې_TESTS) || defined(GGML_VULKAN_CHECK_RESULTS)
#ضمن <chrono>
#ضمن "ggml-cpu.h"
#د_که_پای

#تعریف_کړه VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#إذا VK_سرلیک_نسخه >= 301
namespace vk::detail { class DispatchLoaderDynamic; }
using vk::detail::DispatchLoaderDynamic;
#وإلا
namespace vk { class DispatchLoaderDynamic; }
using vk::DispatchLoaderDynamic;
#د_که_پای

DispatchLoaderDynamic & ggml_vk_default_dispatcher();

#تعریف_کړه VULKAN_HPP_DEFAULT_DISPATCHER ggml_vk_default_dispatcher()

#ضمن <vulkan/vulkan.hpp>

#که_نه_تعریف_شوی VK_NV_cooperative_ماتریکس_decode_ویکټور
#تعریف_کړه VK_NV_cooperative_ماتریکس_decode_ویکټور 1
#تعریف_کړه VK_NV_COOPERATIVE_ماتریکس_DECODE_ویکټور_EXTENSION_NAME "VK_NV_cooperative_matrix_decode_vector"
#تعریف_کړه VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ماتریکس_DECODE_ویکټور_FEATURES_NV ((VkStructureType)1000689000)
تعريفنوع بنية Vk_Physical_Device_Cooperative_ماتریکس_Decode_ویکټور_Features_NV {
    VkStructureType    sType;
    فراغ*              p_بل;
    VkBool32           cooperative_ماتریکس_Decode_ویکټور;
} Vk_Physical_Device_Cooperative_ماتریکس_Decode_ویکټور_Features_NV;
#د_که_پای

#إذا __has_include(<spirv/unified1/spirv.hpp>)
#    include <spirv/unified1/spirv.hpp>
#وإلاإذا __has_include(<spirv-سرلیکونه/spirv.hpp>)
#    include <spirv-headers/spirv.hpp>
#وإلاإذا __has_include(<spirv.hpp>)
#    include <spirv.hpp>
#وإلا
     // Fallback to let the compiler throw a standard "file not found" error
#    include <spirv/unified1/spirv.hpp>
#د_که_پای

#ضمن <algorithm>
#ضمن <cmath>
#ضمن <functional>
#ضمن <iomanip>
#ضمن <iostream>
#ضمن <tuple>
#ضمن <vector>
#ضمن <deque>
#ضمن <sstream>
#ضمن <utility>
#ضمن <memory>
#ضمن <limits>
#ضمن <map>
#ضمن <set>
#ضمن <unordered_map>
#ضمن <shared_mutex>
#ضمن <mutex>
#ضمن <future>
#ضمن <condition_variable>
#ضمن <thread>

#إذا defined(_MSC_VER)
# define NOMINMAX 1
# include <windows.h>
# define YIELD() YieldProcessor()
#وإلاإذا defined(__clang__) || defined(__GNUC__)
# إذا defined(__x86_64__) ||defined(__i386__)
#  include <immintrin.h>
#  define YIELD() _mm_pause()
# elif defined(__arm__) || defined(__aarch64__)
#  إذا defined(__clang__)
#   include <arm_acle.h>
#   define YIELD() __yield()
#  وإلا
#   define YIELD() asm متطاير("yield")
#  endif
# endif
#د_که_پای

#إذا !defined(YIELD)
#تعریف_کړه YIELD()
#د_که_پای

#ضمن "ggml-impl.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-vulkan-shaders.hpp"

#إذا !defined(VK_KHR_shader_bfloat16)

#تعریف_کړه VK_KHR_shader_bfloat16 1
#تعریف_کړه VK_KHR_SHADER_BFLOAT16_SPEC_نسخه                          1
#تعریف_کړه VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME                        "VK_KHR_shader_bfloat16"
#تعریف_کړه VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR ((VkStructureType)1000141000)
#تعریف_کړه VK_COMPONENT_TYPE_BFLOAT16_KHR                               ((VkComponentTypeKHR)1000141000)

تعريفنوع بنية VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType                       sType;
    فراغ*                                 p_بل;
    VkBool32                              shaderBFloat16Type;
    VkBool32                              shader_BFloat16_Dot_حاصل_ضرب;
    VkBool32                              shader_BFloat16_Cooperative_ماتریکس;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;
#د_که_پای

#إذا !defined(VK_VALVE_shader_mixed_float_dot_حاصل_ضرب)
#تعریف_کړه VK_VALVE_shader_mixed_float_dot_حاصل_ضرب 1
#تعریف_کړه VK_VALVE_SHADER_MIXED_FLOAT_DOT_حاصل_ضرب_SPEC_نسخه 1
#تعریف_کړه VK_VALVE_SHADER_MIXED_FLOAT_DOT_حاصل_ضرب_EXTENSION_NAME "VK_VALVE_shader_mixed_float_dot_product"
#تعریف_کړه VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_حاصل_ضرب_FEATURES_VALVE ((VkStructureType)1000673000)
تعريفنوع بنية Vk_Physical_Device_Shader_Mixed_Float_Dot_حاصل_ضرب_Features_VALVE {
    VkStructureType    sType;
    فراغ*              p_بل;
    VkBool32           shader_Mixed_Float_Dot_حاصل_ضرب_Float16_Acc_Float32;
    VkBool32           shader_Mixed_Float_Dot_حاصل_ضرب_Float16_Acc_Float16;
    VkBool32           shader_Mixed_Float_Dot_حاصل_ضرب_BFloat16_Acc;
    VkBool32           shader_Mixed_Float_Dot_حاصل_ضرب_Float8_Acc_Float32;
} Vk_Physical_Device_Shader_Mixed_Float_Dot_حاصل_ضرب_Features_VALVE;
#د_که_پای

#إذا !defined(VK_EXT_shader_ocp_microscaling_types)
#تعریف_کړه VK_EXT_shader_ocp_microscaling_types 1
#تعریف_کړه VK_EXT_SHADER_OCP_MICROSCALING_TYPES_SPEC_نسخه 1
#تعریف_کړه VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME "VK_EXT_shader_ocp_microscaling_types"
#تعریف_کړه VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT ((VkStructureType)1000672000)
تعريفنوع بنية VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT {
    VkStructureType    sType;
    فراغ*              p_بل;
    VkBool32           shaderFloat4;
    VkBool32           shaderFloat6;
    VkBool32           shaderFloat8UnsignedE8M0;
    VkBool32           shaderMXInt8;
} VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT;
#د_که_پای

#إذا !defined(VK_EXT_shader_float8)
#تعریف_کړه VK_EXT_shader_float8 1
#تعریف_کړه VK_EXT_SHADER_FLOAT8_SPEC_نسخه 1
#تعریف_کړه VK_EXT_SHADER_FLOAT8_EXTENSION_NAME "VK_EXT_shader_float8"
#تعریف_کړه VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT ((VkStructureType)1000567000)
تعريفنوع بنية VkPhysicalDeviceShaderFloat8FeaturesEXT {
    VkStructureType    sType;
    فراغ*              p_بل;
    VkBool32           shaderFloat8;
    VkBool32           shader_Float8_Cooperative_ماتریکس;
} VkPhysicalDeviceShaderFloat8FeaturesEXT;
#د_که_پای

#که_نه_تعریف_شوی VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME
#تعریف_کړه VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME "VK_KHR_internally_synchronized_queues"
#تعریف_کړه VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR ((VkStructureType)1000504000)
#تعریف_کړه VK_DEVICE_کتار_CREATE_INTERNALLY_SYNCHRONIZED_بټ_KHR ((Vk_Device_کتار_Create_Flag_بټونه)0x00000004)

// Compile-time constant guaranteed; no runtime initialization overhead
ساكن constexpr vk::Device_کتار_Create_Flag_بټونه eInternallySynchronizedKHR =
    static_cast<vk::Device_کتار_Create_Flag_بټونه>(0x00000004);

تعريفنوع بنية VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR {
    VkStructureType    sType;
    فراغ* p_بل;
    VkBool32           internallySynchronizedQueues;
} VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR;
#وإلا
ساكن constexpr vk::Device_کتار_Create_Flag_بټونه eInternallySynchronizedKHR = vk::Device_کتار_Create_Flag_بټونه::eInternallySynchronizedKHR;
#د_که_پای

#تعریف_کړه ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))

#تعریف_کړه CEIL_DIV(M, N) (((M) / (N)) + (((M) % (N)) != 0))

ساكن bool is_pow2(uint32_t x) { أرجع x > 1 && (x & (x-1)) == 0; }

#تعریف_کړه VK_VENDOR_ID_AMD 0x1002

#تعریف_کړه VK_VENDOR_ID_APPLE 0x106b

#تعریف_کړه VK_VENDOR_ID_INTEL 0x8086

#تعریف_کړه VK_VENDOR_ID_NVIDIA 0x10de

#تعریف_کړه VK_VENDOR_ID_QUALCOMM 0x5143

#تعریف_کړه VK_VENDOR_ID_SAMSUNG 0x144d

#تعریف_کړه VK_DEVICE_DESCRIPTOR_POOL_اندازه 256

#تعریف_کړه VK_CHECK(err, پیغام_2, dev)                                     \
    افعل {                                                            \
        vk::پایله_2 err_;                                            \
        try {                                                       \
            err_ = (err);                                           \
        } catch (vk::Device_ورک_Error &) {                           \
            ggml_vk_print_device_ورک_info(dev);                    \
            GGML_LOG_ERROR("ggml_vulkan: %s at %s:%d\n",            \
                #err, __FILE__, __LINE__);                          \
            throw;                                                  \
        }                                                           \
        إذا (err_ != vk::پایله_2::eSuccess) {                         \
            GGML_LOG_ERROR("ggml_vulkan: %s error %s at %s:%d\n",   \
                #err, منزل_string(err_).c_str(), __FILE__, __LINE__); \
            throw vk::SystemError(vk::make_error_کوډ(err_),        \
                "ggml_vulkan: " پیغام_2);                               \
        }                                                           \
    } طالما (0)

#که_تعریف_شوی GGML_VULKAN_DEBUG
#تعریف_کړه VK_LOG_DEBUG(پیغام_2) std::cerr << پیغام_2 << std::endl
#وإلا
#تعریف_کړه VK_LOG_DEBUG(پیغام_2) ((فراغ) 0)
#د_که_پای // GGML_VULKAN_DEBUG

#تعریف_کړه MAX_PARAMETER_شمېر 12

#تعریف_کړه MAX_FUSED_ADDS (MAX_PARAMETER_شمېر - 3)

بنية vk_pipeline_struct;

تعريفنوع std::shared_ptr<بنية vk_pipeline_struct> vk_pipeline;

بنية vk_pipeline_struct {
    std::string name;
    vk::ShaderModule shader_module;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t ټېل_constant_اندازه;
    uint32_t parameter_شمېر;
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
    uint32_t ثبتوونکی_شمېر {};

#إذا defined(VK_EXT_shader_64bit_indexing)
    bool is_64b_indexing {};
#د_که_پای
    // linked list of pipelines for multiple compilation variants.
    // currently only used to compile a 64-bit indexing variant.
    vk_pipeline بل;
};

تعريفنوع std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;

بنية vk_matmul_pipeline_کلی {
    ggml_type type_a;
    ggml_type type_b;
    bool mul_mat_id;
    bool f16acc;

    bool چلوونکی<(ثابت vk_matmul_pipeline_کلی & o) ثابت {
        أرجع std::tie(type_a, type_b, mul_mat_id, f16acc)
             < std::tie(o.type_a, o.type_b, o.mul_mat_id, o.f16acc);
    }
};

بنية vk_matmul_pipeline_pair {
    vk_pipeline unaligned;
    vk_pipeline aligned;
    uint32_t align;
};

بنية vk_tile_config {
    std::ویکټور<uint32_t> warptile;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
};

using matmul_tile_selector_t = std::function<uint32_t(
    uint32_t m, uint32_t n, uint32_t k, uint32_t shader_core_شمېر,
    ثابت std::ویکټور<vk_matmul_pipeline_pair>& configs)>;

بنية vk_device_struct;

تعريفنوع std::shared_ptr<vk_device_struct> vk_device;

تعريفنوع std::weak_ptr<vk_device_struct> vk_device_ref;

بنية vk_بفر_struct;

تعريفنوع std::shared_ptr<vk_بفر_struct> vk_بفر;

تعريفنوع std::weak_ptr<vk_بفر_struct> vk_بفر_ref;

بنية ggml_backend_vk_بفر_type_context {
    std::string name;
    vk_device device;
};

بنية vk_command_بفر {
    vk::Command_بفر buf;
    uint64_t use_counter = 0;
    bool دننه_use = false;
};

بنية vk_کتار;

بنية vk_command_pool {
    فراغ init(vk_device& device, vk_کتار *q_);
    فراغ destroy(vk::Device& device);

    vk::CommandPool pool;
    // Using deque so the pointers to command buffers
    // remain valid even if we add more
    std::deque<vk_command_بفر> cmd_buffers;

    vk_کتار *q;

    size_t buffers_دننه_use() ثابت {
        أرجع std::شمېر_if(cmd_buffers.begin(), cmd_buffers.پای_ټکی(),
            [](ثابت خپلکاری& cb) { أرجع cb.دننه_use; });
    }
};

بنية vk_کتار_handle {
    vk::کتار_2 کتار;
    vk_device_ref device;
    std::mutex * device_submit_mutex = nullptr;
    مجازي فراغ submit(vk::ArrayProxy<ثابت vk::SubmitInfo> submits, vk::Fence fence) = 0;
    مجازي فراغ قلف()   {}   // no-op by default (internally synchronized case)
    مجازي فراغ unlock() {}
    مجازي ~vk_کتار_handle() = افتراضي;
};

بنية vk_کتار_handle_synchronized : vk_کتار_handle {
    std::mutex mutex;
    فراغ submit(vk::ArrayProxy<ثابت vk::SubmitInfo> submits, vk::Fence fence) override;

    فراغ قلف()   override { mutex.قلف(); }
    فراغ unlock() override { mutex.unlock(); }
};

بنية vk_کتار_handle_unsynchronized : vk_کتار_handle {
    فراغ submit(vk::ArrayProxy<ثابت vk::SubmitInfo> submits, vk::Fence fence) override;

    // lock()/unlock() inherited no-ops
};

بنية vk_کتار {
    uint32_t کتار_family_شاخص;
    std::shared_ptr<vk_کتار_handle> handle;

    vk_command_pool cmd_pool;

    vk::PipelineStageFlags stage_flags;

    bool transfer_only;
};

ساكن constexpr uint32_t mul_mat_vec_max_ستون_شمېر = 8;

ساكن constexpr uint32_t p021_max_gqa_نسبت = 8;

تعداد vk_device_architecture {
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

تعداد vk_conv_shapes {
    CONV_SHAPE_128x128,
    CONV_SHAPE_64x32,
    CONV_SHAPE_32x256,
    CONV_SHAPE_64x128,
    CONV_SHAPE_شمېر,
};

بنية vk_conv_بلاک_اندازه {
    uint32_t K;
    uint32_t NPQ;
    uint32_t CRS;
};

مضمن vk_conv_بلاک_اندازه vk_conv_بلاک_sizes[CONV_SHAPE_شمېر] = {
    // K   NPQ  CRS
    { 128, 128, 16 }, // CONV_SHAPE_128x128
    {  64,  32, 32 }, // CONV_SHAPE_64x32
    {  32, 256, 16 }, // CONV_SHAPE_32x256
    {  64, 128, 16 }, // CONV_SHAPE_64x128
};

تعداد dmmv_wg_sizes {
    DMMV_WG_اندازه_SUBGROUP,
    DMMV_WG_اندازه_LARGE,
    DMMV_WG_اندازه_شمېر,
};

تعداد Fa_کوډ_لار {
    FA_SCALAR,
    FA_COOPMAT1,
    FA_COOPMAT2,
};

بنية vk_fa_pipeline_state {
    uint32_t HSK, HSV;
    uint32_t Br, Bc;
    uint32_t D_وېشل, لیکه_وېشل;
    bool shmem_staging;
    Fa_کوډ_لار لار;
    uint32_t workgroup_اندازه, subgroup_اندازه;
    bool aligned;
    bool f32acc;
    uint32_t flags;
    uint32_t حد_occupancy_shmem;
    ggml_type k_type;
    ggml_type v_type;

    bool چلوونکی<(ثابت vk_fa_pipeline_state &b) ثابت {
        أرجع std::tie(HSK, HSV, Br, Bc, D_وېشل, لیکه_وېشل, shmem_staging, لار, workgroup_اندازه, subgroup_اندازه, aligned, f32acc, flags, حد_occupancy_shmem, k_type, v_type) <
               std::tie(b.HSK, b.HSV, b.Br, b.Bc, b.D_وېشل, b.لیکه_وېشل, b.shmem_staging, b.لار, b.workgroup_اندازه, b.subgroup_اندازه, b.aligned, b.f32acc, b.flags, b.حد_occupancy_shmem, b.k_type, b.v_type);
    }
};

بنية vk_conv2d_pipeline_state {
    vk_conv2d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t p0, uint32_t p1, uint32_t d0, uint32_t d1, uint32_t KW, uint32_t KH, uint32_t aligned)
        : s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), KW(KW), KH(KH), aligned(aligned) {}

    uint32_t s0, s1, p0, p1, d0, d1, KW, KH;
    // when set, shader can skip K/CRS/NPQ bounds checks and address clamps
    uint32_t aligned;

    bool چلوونکی<(ثابت vk_conv2d_pipeline_state &b) ثابت {
        أرجع std::tie(s0, s1, p0, p1, d0, d1, KW, KH, aligned) <
               std::tie(b.s0, b.s1, b.p0, b.p1, b.d0, b.d1, b.KW, b.KH, b.aligned);
    }
};

بنية vk_conv3d_pipeline_state {
    vk_conv3d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t p0, uint32_t p1, uint32_t p2,
                             uint32_t d0, uint32_t d1, uint32_t d2, uint32_t KW, uint32_t KH, uint32_t KD, uint32_t aligned)
        : s0(s0), s1(s1), s2(s2), p0(p0), p1(p1), p2(p2), d0(d0), d1(d1), d2(d2), KW(KW), KH(KH), KD(KD), aligned(aligned) {}

    uint32_t s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD;
    uint32_t aligned;

    bool چلوونکی<(ثابت vk_conv3d_pipeline_state &b) ثابت {
        أرجع std::tie(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned) <
               std::tie(b.s0, b.s1, b.s2, b.p0, b.p1, b.p2, b.d0, b.d1, b.d2, b.KW, b.KH, b.KD, b.aligned);
    }
};

بنية vk_solve_tri_pipeline_state {
    vk_solve_tri_pipeline_state(uint32_t N, uint32_t K)
        : N(N), K(K) {}

    uint32_t N, K;

    bool چلوونکی<(ثابت vk_solve_tri_pipeline_state &b) ثابت {
        أرجع std::tie(N, K) <
               std::tie(b.N, b.K);
    }
};

تعداد shader_reduction_ډېرپېښ {
    SHADER_REDUCTION_ډېرپېښ_SHMEM,
    SHADER_REDUCTION_ډېرپېښ_HYBRID,
    SHADER_REDUCTION_ډېرپېښ_SUBGROUP,
    SHADER_REDUCTION_ډېرپېښ_شمېر,
};

ساكن constexpr uint32_t عدد_argsort_pipelines = 11;

ساكن constexpr uint32_t عدد_topk_moe_pipelines = 10;

ساكن constexpr uint32_t عدد_topk_pipelines = 11;

ساكن constexpr std::initializer_list<ggml_op> topk_moe_early_softmax_norm{ GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_لیکې, GGML_OP_RESHAPE,
                                                                             GGML_OP_مجموعه_لیکې, GGML_OP_CLAMP,    GGML_OP_DIV,
                                                                             GGML_OP_RESHAPE };

ساكن constexpr std::initializer_list<ggml_op> topk_moe_sigmoid_norm_bias{ GGML_OP_UNARY,    GGML_OP_RESHAPE,  GGML_OP_جمع_کړه,
                                                                            GGML_OP_ARGSORT,  GGML_OP_VIEW,     GGML_OP_GET_لیکې,
                                                                            GGML_OP_RESHAPE,  GGML_OP_مجموعه_لیکې, GGML_OP_CLAMP,
                                                                            GGML_OP_DIV,      GGML_OP_RESHAPE };

ساكن constexpr std::initializer_list<ggml_op> topk_moe_sqrt_softplus_norm_bias{ GGML_OP_UNARY,    GGML_OP_SQRT,
                                                                                  GGML_OP_RESHAPE,  GGML_OP_جمع_کړه,
                                                                                  GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                                  GGML_OP_GET_لیکې, GGML_OP_RESHAPE,
                                                                                  GGML_OP_مجموعه_لیکې, GGML_OP_CLAMP,
                                                                                  GGML_OP_DIV,      GGML_OP_RESHAPE };

ساكن constexpr std::initializer_list<ggml_op> topk_moe_early_softmax     { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_لیکې };

ساكن constexpr std::initializer_list<ggml_op> topk_moe_late_softmax      { GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                             GGML_OP_GET_لیکې, GGML_OP_RESHAPE,
                                                                             GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

ساكن constexpr std::initializer_list<ggml_op> snake_نمونه              { GGML_OP_MUL,      GGML_OP_SIN,
                                                                             GGML_OP_SQR,      GGML_OP_MUL,
                                                                             GGML_OP_جمع_کړه };

ساكن constexpr std::initializer_list<ggml_op> topk_qsa_نمونه { GGML_OP_GET_لیکې, GGML_OP_ترتیب_بدلول,
                                                                   GGML_OP_CONT,     GGML_OP_CPY,
                                                                   GGML_OP_RESHAPE,  GGML_OP_جمع_کړه,
                                                                   GGML_OP_سر_K };

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_qsa_څنډې {
    { 1, 0, 0 }, // permute->src[0] == get_rows
    { 2, 0, 1 }, // cont->src[0]    == permute
    { 4, 0, 3 }, // reshape->src[0] == cpy (mask cast)
    { 5, 0, 2 }, // add->src[0]     == cont
    { 5, 1, 4 }, // add->src[1]     == reshape
    { 6, 0, 5 }, // top_k->src[0]   == add
};

ساكن constexpr std::initializer_list<ggml_op> rms_norm_mul_جمع_کړه_mul_نمونه { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_جمع_کړه, GGML_OP_MUL };

ساكن constexpr std::initializer_list<ggml_op> rms_norm_mul_جمع_کړه_نمونه     { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_جمع_کړه };

ساكن constexpr std::initializer_list<ggml_op> rms_norm_mul_rope_view_سېټ_لیکې_نمونه { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_سېټ_لیکې };

ساكن constexpr std::initializer_list<ggml_op> rms_norm_view_سېټ_لیکې_نمونه { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_سېټ_لیکې };

ساكن constexpr std::initializer_list<ggml_op> rope_view_سېټ_لیکې_نمونه { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_سېټ_لیکې };

// scale_out*sigmoid(scale_in*x) as the hc_post weights (qwen4exp hc_combine)
ساكن constexpr std::initializer_list<ggml_op> hc_post_gate_نمونه { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> hc_post_gate_څنډې {
    { 1, 0, 0 }, // sigmoid->src[0] == scale
    { 2, 0, 1 }, // scale->src[0]   == sigmoid
    { 3, 2, 2 }, // hc_post->src[2] == scale (post)
};

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_moe_early_softmax_norm_څنډې {
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

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_moe_sigmoid_norm_bias_څنډې {
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

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_moe_sqrt_softplus_norm_bias_څنډې {
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

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_moe_early_softmax_څنډې {
    { 1, 0, 0 }, // reshape->src[0]  == softmax
    { 2, 0, 0 }, // argsort->src[0]  == softmax
    { 3, 0, 2 }, // view->src[0]     == argsort
    { 4, 0, 1 }, // get_rows->src[0] == reshape
    { 4, 1, 3 }, // get_rows->src[1] == view
};

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_moe_late_softmax_څنډې {
    { 1, 0, 0 }, // view->src[0]     == argsort
    { 2, 1, 1 }, // get_rows->src[1] == view
    { 3, 0, 2 }, // reshape->src[0]  == get_rows
    { 4, 0, 3 }, // soft_max->src[0] == reshape
    { 5, 0, 4 }, // reshape->src[0]  == soft_max
};

تعداد topk_moe_ډېرپېښ {
    TOPK_MOE_EARLY_SOFTMAX,
    TOPK_MOE_EARLY_SOFTMAX_NORM,
    TOPK_MOE_LATE_SOFTMAX,
    TOPK_MOE_SIGMOID_NORM_BIAS,
    TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS,
    TOPK_MOE_شمېر,
};

تعداد rms_norm_ډېرپېښ {
    RMS_NORM_MUL,
    RMS_NORM_MUL_جمع_کړه,
    RMS_NORM_MUL_جمع_کړه_MUL,
    RMS_NORM_MUL_ROPE,
    RMS_NORM_MUL_ROPE_VIEW_سېټ_لیکې,
    RMS_NORM_VIEW_سېټ_لیکې,
    RMS_NORM_شمېر,
};

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> rope_view_سېټ_لیکې_څنډې {
    { 1, 0, 0 }, // view->src[0]     == rope
    { 2, 0, 1 }, // set_rows->src[0] == view
};

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> rms_norm_mul_rope_view_سېټ_لیکې_څنډې {
    { 1, 0, 0 }, // mul->src[0]      == rms
    { 2, 0, 1 }, // rope->src[0]     == mul
    { 3, 0, 2 }, // view->src[0]     == rope
    { 4, 0, 3 }, // set_rows->src[0] == view
};

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> rms_norm_view_سېټ_لیکې_څنډې {
    { 1, 0, 0 }, // view->src[0]     == rms_norm
    { 2, 0, 1 }, // set_rows->src[0] == view
};

ساكن constexpr std::array<ggml_type, 9> lightning_indexer_k_types = {
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

class vk_حافظه_logger;

بنية vk_device_struct {
    std::recursive_mutex mutex;
    std::mutex کتار_submit_mutex;

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
    uint64_t max_حافظه_ویش_اندازه;
    uint64_t max_بفر_اندازه;
    uint64_t suballocation_بلاک_اندازه;
    uint64_t min_imported_host_اشاره_ګر_alignment;
    bool external_حافظه_host {};
    bool fp16;
    bool bf16;
    bool pipeline_robustness;
    bool حافظه_لومړیتوب;
    vk::Device device;
    uint32_t vendor_id;
    vk::DriverId driver_id;
    vk_device_architecture architecture;
    std::unique_ptr<vk_کتار> compute_کتار;
    std::unique_ptr<vk_کتار> transfer_کتار;
    bool single_کتار;
    bool support_async;
    bool async_use_transfer_کتار;
    bool has_internally_synchronized_queues = false;
    uint32_t subgroup_اندازه;
    uint32_t subgroup_اندازه_log2;
    uint32_t shader_core_شمېر;
    bool uma;
    bool prefer_host_حافظه;
    bool float_controls_rte_fp16;
    bool float_controls_denorm_preserve_fp16;
    bool subgroup_basic;
    bool subgroup_arithmetic;
    bool subgroup_shuffle;
    bool subgroup_ballot;
    bool subgroup_clustered;
    bool subgroup_vote;
    bool multi_جمع_کړه;
    bool shader_int64;
    bool بفر_device_پته;
    bool vulkan_حافظه_model;

    bool جمع_کړه_rms_fusion;
    uint32_t partials_binding_alignment;
    uint32_t max_nodes_فی_submit;

    bool shader_64b_indexing;

    bool integer_dot_حاصل_ضرب;
    // 0: default, 1: force mmvq, -1: disable mmvq
    int32_t mmvq_ډېرپېښ;

    bool subgroup_اندازه_control;
    uint32_t subgroup_min_اندازه;
    uint32_t subgroup_max_اندازه;
    bool subgroup_require_ډک_support;

    // floor(log2(maxComputeWorkGroupInvocations))
    uint32_t max_workgroup_اندازه_log2 {};

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
    bool coopmat2_decode_ویکټور;

    bool dot2_f16 {};
    bool ocp_fp4 {};

    bool pipeline_executable_properties_support {};

    bool device_fault {};
    PFN_vkGetDeviceFaultInfoEXT pfn_vkGetDeviceFaultInfoEXT {};

    bool serialize_submissions {};

    ثابت ggml_cgraph * diag_cgraph {};
    صحيح diag_prev_پیل = -1;
    صحيح diag_prev_پای_ټکی = -1;

    size_t idx;

    bool mul_mat_l[GGML_TYPE_شمېر];
    bool mul_mat_m[GGML_TYPE_شمېر];
    bool mul_mat_s[GGML_TYPE_شمېر];
    bool mul_mat_id_l[GGML_TYPE_شمېر];
    bool mul_mat_id_m[GGML_TYPE_شمېر];
    bool mul_mat_id_s[GGML_TYPE_شمېر];

    // Separate flags for the q8_1 (integer dot) mmq path, whose shader uses
    // a different shared-memory layout than the float matmul shaders.
    bool mul_mat_l_int[GGML_TYPE_شمېر];
    bool mul_mat_m_int[GGML_TYPE_شمېر];
    bool mul_mat_s_int[GGML_TYPE_شمېر];
    bool mul_mat_id_l_int[GGML_TYPE_شمېر];
    bool mul_mat_id_m_int[GGML_TYPE_شمېر];
    bool mul_mat_id_s_int[GGML_TYPE_شمېر];

    vk::Descriptor_سېټ_Layout dsl;

    std::map<vk_matmul_pipeline_کلی, std::ویکټور<vk_matmul_pipeline_pair>> pipeline_matmul;
    matmul_tile_selector_t matmul_tile_selector;
    matmul_tile_selector_t matmul_id_tile_selector;

    vk_pipeline pipeline_matmul_وېشل_k_reduce;
    vk_pipeline pipeline_quantize_q8_1_x4;

    vk_pipeline pipeline_dequant[GGML_TYPE_شمېر];
    vk_pipeline pipeline_dequant_transpose[GGML_TYPE_شمېر]; // fused dequant+transpose for FA quant-KV
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_اندازه_شمېر][GGML_TYPE_شمېر][mul_mat_vec_max_ستون_شمېر];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[DMMV_WG_اندازه_شمېر][GGML_TYPE_شمېر][mul_mat_vec_max_ستون_شمېر];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[DMMV_WG_اندازه_شمېر][GGML_TYPE_شمېر];

    vk_pipeline pipeline_dequant_mul_mat_vec_q8_1_f32[DMMV_WG_اندازه_شمېر][GGML_TYPE_شمېر][mul_mat_vec_max_ستون_شمېر];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_q8_1_f32[DMMV_WG_اندازه_شمېر][GGML_TYPE_شمېر];

    vk_pipeline pipeline_mul_mat_vec_p021_f16_f32[p021_max_gqa_نسبت];
    vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
    vk_pipeline pipeline_get_لیکې[GGML_TYPE_شمېر];
    vk_pipeline pipeline_get_لیکې_f32[GGML_TYPE_شمېر];
    vk_pipeline pipeline_get_لیکې_شا_f32;
    vk_pipeline pipeline_acc_f32;
    vk_pipeline pipeline_سېټ_f32;

    // [src0 0=fp32,1=fp16][src1 0=fp32,1=fp16][dst 0=fp32,1=fp16]
    vk_pipeline pipeline_جمع_کړه[2][2][2];
    vk_pipeline pipeline_جمع_کړه_norepeat[2][2][2];
    vk_pipeline pipeline_sub[2][2][2];
    vk_pipeline pipeline_sub_norepeat[2][2][2];
    vk_pipeline pipeline_mul[2][2][2];
    vk_pipeline pipeline_mul_norepeat[2][2][2];
    vk_pipeline pipeline_div[2][2][2];
    vk_pipeline pipeline_div_norepeat[2][2][2];
    vk_pipeline pipeline_جمع_کړه_rms[2][2][2];
    vk_pipeline pipeline_جمع_کړه_rms_norepeat[2][2][2];

    // indexed by num_additional_fused_ops == num_adds - 1
    vk_pipeline pipeline_multi_جمع_کړه[MAX_FUSED_ADDS];
    vk_pipeline pipeline_multi_جمع_کړه_rms[MAX_FUSED_ADDS];

    vk_pipeline pipeline_جمع_کړه_id_f32;

    vk_pipeline pipeline_concat_i8, pipeline_concat_i16, pipeline_concat_i32, pipeline_concat_i64;
    vk_pipeline pipeline_upscale_تر_ټولو_نږدې_f32, pipeline_upscale_bilinear_f32, pipeline_upscale_bicubic_f32, pipeline_upscale_bilinear_antialias_f32;
    vk_pipeline pipeline_scale_f32;
    vk_pipeline pipeline_log[2];
    vk_pipeline pipeline_tri[2];
    vk_pipeline pipeline_diag[2];
    vk_pipeline pipeline_clamp[2];
    vk_pipeline pipeline_pad_f32;
    vk_pipeline pipeline_pad_reflect_1d_f32;
    vk_pipeline pipeline_roll_f32;
    vk_pipeline pipeline_repeat_i32, pipeline_repeat_شا_f32;
    vk_pipeline pipeline_repeat_i16;
    vk_pipeline pipeline_cpy_32, pipeline_cpy_f32_f16, pipeline_cpy_16, pipeline_cpy_f16_f32, pipeline_cpy_f32_bf16, pipeline_cpy_bf16_f32, pipeline_cpy_bf16_f16, pipeline_cpy_f32_i32, pipeline_cpy_i32_f32;
    vk_pipeline pipeline_contig_cpy_32, pipeline_contig_cpy_f32_f16, pipeline_contig_cpy_16, pipeline_contig_cpy_f16_f32, pipeline_contig_cpy_f32_bf16, pipeline_contig_cpy_bf16_f32, pipeline_contig_cpy_bf16_f16, pipeline_contig_cpy_f32_i32, pipeline_contig_cpy_i32_f32;
    vk_pipeline pipeline_cpy_f32_quant[GGML_TYPE_شمېر];
    vk_pipeline pipeline_cpy_quant_f32[GGML_TYPE_شمېر];
    vk_pipeline pipeline_cpy_transpose_16, pipeline_cpy_transpose_32;
    vk_pipeline pipeline_cpy_transpose_02_16, pipeline_cpy_transpose_02_32;
    // [src0 0=fp32,1=fp16][dst]
    vk_pipeline pipeline_سېټ_لیکې_i32[2][GGML_TYPE_شمېر];
    vk_pipeline pipeline_سېټ_لیکې_i64[2][GGML_TYPE_شمېر];
    vk_pipeline pipeline_norm_f32;
    vk_pipeline pipeline_ډله_norm_f32;
    vk_pipeline pipeline_rms_norm_f32;
    vk_pipeline pipeline_rms_norm_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_جمع_کړه_f32;
    vk_pipeline pipeline_rms_norm_mul_جمع_کړه_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_جمع_کړه_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_جمع_کړه_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_سېټ_لیکې_f32_f32;
    vk_pipeline pipeline_rms_norm_سېټ_لیکې_f32_f16;
    vk_pipeline pipeline_rms_norm_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f16;
    vk_pipeline pipeline_rms_norm_شا_f32;
    vk_pipeline pipeline_l2_norm_f32;

    // [src/dst 0=fp32,1=fp16]
    vk_pipeline pipeline_exp[2];
    vk_pipeline pipeline_expm1[2];
    vk_pipeline pipeline_elu[2];
    vk_pipeline pipeline_gelu[2];
    vk_pipeline pipeline_gelu_erf[2];
    vk_pipeline pipeline_gelu_چټک[2];
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
    vk_pipeline pipeline_ګام[2];
    vk_pipeline pipeline_پړاو[2];
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
    vk_pipeline pipeline_geglu_چټک[2];

    vk_pipeline pipeline_leaky_relu[2];
    vk_pipeline pipeline_silu_شا_f32;
    vk_pipeline pipeline_diag_ماسک_inf_f32;
    vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
    vk_pipeline pipeline_soft_max_f32_wg512, pipeline_soft_max_f32_f16_wg512;
    vk_pipeline pipeline_soft_max_شا_f32;

    vk_pipeline pipeline_soft_max_large1_f32, pipeline_soft_max_large1_f32_f16;
    vk_pipeline pipeline_soft_max_large2_f32, pipeline_soft_max_large2_f32_f16;
    vk_pipeline pipeline_soft_max_large3_f32, pipeline_soft_max_large3_f32_f16;

    vk_pipeline pipeline_rope_norm_f32, pipeline_rope_norm_f16, pipeline_rope_norm_f32_f16;
    vk_pipeline pipeline_rope_neox_f32, pipeline_rope_neox_f16, pipeline_rope_neox_f32_f16;
    vk_pipeline pipeline_rope_multi_f32, pipeline_rope_multi_f16, pipeline_rope_multi_f32_f16;
    vk_pipeline pipeline_rope_vision_f32, pipeline_rope_vision_f16;
    vk_pipeline pipeline_argsort_f32[عدد_argsort_pipelines];
    vk_pipeline pipeline_argsort_large_f32[عدد_argsort_pipelines];
    vk_pipeline pipeline_topk_f32[عدد_topk_pipelines];
    vk_pipeline pipeline_topk_radix_f32;
    vk_pipeline pipeline_topk_radix_qsa; // qwen4 QSA indexer fusion (f16 mask)
    vk_pipeline pipeline_مجموعه_لیکې_f32;
    vk_pipeline pipeline_cross_entropy_loss_f32, pipeline_cross_entropy_loss_f32_wg512;
    vk_pipeline pipeline_cross_entropy_loss_شا_f32, pipeline_cross_entropy_loss_شا_f32_wg512;
    vk_pipeline pipeline_fwht_f32[4];
    vk_pipeline pipeline_cumsum_f32;
    vk_pipeline pipeline_cumsum_small_f32;
    vk_pipeline pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline_cumsum_multipass2_f32;
    vk_pipeline pipeline_argmax_f32;
    vk_pipeline pipeline_شمېر_equal_i32;
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
    vk_pipeline pipeline_بهر_prod_f32;
    vk_pipeline pipeline_snake_f32;
    vk_pipeline pipeline_snake_f16;
    vk_pipeline pipeline_snake_bf16;
    vk_pipeline pipeline_pool1d_f32;
    vk_pipeline pipeline_pool2d_f32;
    vk_pipeline pipeline_pool2d_subgroup_f32;
    vk_pipeline pipeline_rwkv_wkv6_f32;
    vk_pipeline pipeline_rwkv_wkv7_f32;
    vk_pipeline pipeline_gated_خطي_attn_f32;
    vk_pipeline pipeline_lightning_indexer_f32[GGML_TYPE_شمېر];
    // [size_idx][kda] where size_idx: 0=d16, 1=d32, 2=d64, 3=d128
    vk_pipeline pipeline_gated_delta_net[4][2];
    vk_pipeline pipeline_ssm_لټون_f32_d128;
    vk_pipeline pipeline_ssm_لټون_f32_d256;
    vk_pipeline pipeline_ssm_conv_f32;
    vk_pipeline pipeline_ssm_conv_silu_f32;
    vk_pipeline pipeline_ssm_conv_bias_silu_f32;
    vk_pipeline pipeline_opt_ګام_adamw_f32;
    vk_pipeline pipeline_opt_ګام_sgd_f32;
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f32[CONV_SHAPE_شمېر];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f16_f32[CONV_SHAPE_شمېر];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f32[CONV_SHAPE_شمېر];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f16_f32[CONV_SHAPE_شمېر];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f32[CONV_SHAPE_شمېر];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f16_f32[CONV_SHAPE_شمېر];
    vk_pipeline pipeline_conv2d_dw_whcn_f32, pipeline_conv2d_dw_whcn_f16_f32;
    vk_pipeline pipeline_conv2d_dw_cwhn_f32, pipeline_conv2d_dw_cwhn_f16_f32;

    std::map<vk_fa_pipeline_state, vk_pipeline> pipeline_flash_attn_f32_f16;

    std::map<std::pair<uint32_t, uint32_t>, vk_pipeline> pipeline_fa_ماسک_opt;

    vk_pipeline pipeline_fa_sparse_compact;
    vk_pipeline pipeline_fa_sparse_compact_subgroup;
    bool fa_sparse_compact_use_subgroups;

    vk_pipeline pipeline_flash_attn_وېشل_k_reduce;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, std::pair<vk_pipeline, vk_pipeline>> pipeline_xe_fa_decode_dual_phases;
    vk_pipeline pipeline_شمېر_experts;

    // [2] is for whether to take n_experts from spec constant (0) or push constant (1)
    vk_pipeline pipeline_topk_moe[عدد_topk_moe_pipelines][2];

    std::ویکټور<vk_pipeline_ref> all_pipelines;

    vk::Fence fence;
    vk_بفر sync_staging;

    ggml_backend_بفر_type بفر_type;

    bool disable_fusion;
    bool disable_descriptor_reuse;
    std::atomic<uint64_t> بفر_destroy_شمېر {};
    bool disable_host_visible_vidmem;
    bool allow_sysmem_fallback;
    bool disable_ګراف_optimize;

    std::unique_ptr<vk_حافظه_logger> حافظه_logger;

    ~vk_device_struct();

};

مضمن فراغ vk_command_pool::init(vk_device& device, vk_کتار *q_) {
    cmd_buffers.clear();
    q = q_;

    vk::CommandPoolCreateInfo command_pool_create_info(
        vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_بټ | VK_COMMAND_POOL_CREATE_RESET_COMMAND_بفر_بټ),
        q->کتار_family_شاخص);
    pool = device->device.createCommandPool(command_pool_create_info);
}

مضمن فراغ vk_command_pool::destroy(vk::Device& device) {
    device.destroyCommandPool(pool);
    pool = nullptr;
    cmd_buffers.clear();
}

بنية vk_بفر_struct {
    vk::بفر_2 بفر = VK_NULL_HANDLE;
    vk::Device_حافظه device_حافظه = VK_NULL_HANDLE;
    vk::حافظه_Property_Flags حافظه_property_flags;
    فراغ * ptr;
    size_t اندازه = 0;
    vk::Device_پته bda_addr {};

    vk_device device;

    ~vk_بفر_struct() {
        إذا (اندازه == 0) {
            أرجع;
        }
        VK_LOG_DEBUG("~vk_buffer_struct(" << بفر << ", " << اندازه << ")");

        // bump before destroying, so a thread that sees the buffer gone also sees the new count
        device->بفر_destroy_شمېر.fetch_جمع_کړه(1, std::memory_order_release);
        device->device.free_حافظه_2(device_حافظه);
        device->device.destroy_بفر(بفر);
    }
};

بنية vk_subbuffer {
    vk_بفر بفر;
    uint64_t دننه_واټن;
    uint64_t اندازه;

    چلوونکی vk::Descriptor_بفر_Info() ثابت {
        أرجع { بفر->بفر, دننه_واټن, اندازه };
    }
};

بنية vk_semaphore {
    vk::Semaphore s;
    uint64_t ارزښت;
};

بنية vk_event {
    std::ویکټور<vk::Event> events_free; // Events available for reuse
    std::ویکټور<vk::Event> events_submitted; // Events that are fully submitted and can be reused on next synchronize
    vk::Event event;
    bool has_event;

    vk_semaphore tl_semaphore;
    vk_command_بفر* cmd_بفر = nullptr;
    uint64_t cmd_بفر_use_counter = 0;
};

بنية vk_submission {
    vk_command_بفر* بفر = nullptr;
    std::ویکټور<vk_semaphore> wait_semaphores;
    std::ویکټور<vk_semaphore> signal_semaphores;
};

تعريفنوع std::ویکټور<vk_submission> vk_sequence;

#تعریف_کړه MAT_VEC_FUSION_FLAGS_BIAS0 0x1

#تعریف_کړه MAT_VEC_FUSION_FLAGS_BIAS1 0x2

#تعریف_کړه MAT_VEC_FUSION_FLAGS_SCALE0 0x4

#تعریف_کړه MAT_VEC_FUSION_FLAGS_SCALE1 0x8

بنية vk_staging_memcpy {
    vk_staging_memcpy(فراغ * _dst, ثابت فراغ * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    فراغ * dst;
    ثابت فراغ * src;
    size_t n;
};

بنية vk_staging_memset {
    vk_staging_memset(فراغ * _dst, uint32_t _val, size_t _n) : dst(_dst), val(_val), n(_n) {}

    فراغ * dst;
    uint32_t val;
    size_t n;
};

بنية vk_context_struct {
    vk_submission * s;
    std::ویکټور<vk_sequence> seqs;

    صحيح exit_tensor_idx;

    std::ویکټور<vk_staging_memcpy> دننه_memcpys;
    std::ویکټور<vk_staging_memcpy> بهر_memcpys;
    std::ویکټور<vk_staging_memset> memsets;

    std::ویکټور<std::string> debug_labels;

    vk_command_pool * p {};
};

تعريفنوع std::shared_ptr<vk_context_struct> vk_context;

تعريفنوع std::weak_ptr<vk_context_struct> vk_context_ref;

بنية ggml_vk_garbage_collector {
    std::ویکټور<vk_semaphore> tl_semaphores;
    std::ویکټور<vk_semaphore> semaphores;
    std::ویکټور<vk::Event> events;
    std::ویکټور<vk_context> contexts;
};

#تعریف_کړه VK_LOG_حافظه(پیغام_2) إذا (vk_حافظه_logger_enabled) { std::cerr << "ggml_vulkan memory: " << پیغام_2 << std::endl; }

ساكن std::string format_اندازه(size_t اندازه) {
    ثابت size_t kib = 1024;
    ثابت size_t mib = kib * 1024;
    ثابت size_t gib = mib * 1024;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    إذا (اندازه >= gib) {
        oss << static_cast<مزدوج>(اندازه) / gib << " GiB";
    } وإلا إذا (اندازه >= mib) {
        oss << static_cast<مزدوج>(اندازه) / mib << " MiB";
    } وإلا إذا (اندازه >= kib) {
        oss << static_cast<مزدوج>(اندازه) / kib << " KiB";
    } وإلا {
        oss << اندازه << " B";
    }

    أرجع oss.str();
}

class vk_حافظه_logger {
عام:
    vk_حافظه_logger(): ټول_device(0), ټول_host(0) {}
    فراغ log_ویش(vk_بفر_ref buf_ref, size_t اندازه);
    فراغ log_deallocation(vk_بفر_ref buf_ref);

خاص:
    std::map<vk::بفر_2, size_t> allocations; // Track allocations
    size_t ټول_device;
    size_t ټول_host;
    ساكن std::mutex log_mutex;
};

مضمن std::mutex vk_حافظه_logger::log_mutex;

class vk_perf_logger {
  عام:
    فراغ print_timings(bool force = false);


    std::string get_غوټه_fusion_name(ثابت ggml_tensor * غوټه, ثابت حرف *fusion_name, uint64_t *n_flops);


    فراغ log_timing(ثابت ggml_tensor * غوټه, ثابت حرف *fusion_name, uint64_t وخت) {
        uint64_t n_flops;
        std::string name = get_غوټه_fusion_name(غوټه, fusion_name, &n_flops);
        إذا (n_flops) {
            flops[name].ټېل_شا(n_flops);
        }
        timings[name].ټېل_شا(وخت);
    }

    فراغ log_timing(ثابت std::ویکټور<ggml_tensor *> &nodes, ثابت std::ویکټور<ثابت حرف *> &نومونه, uint64_t وخت) {
        uint64_t ټول_flops = 0;
        std::string name;
        لكل (size_t n = 0; n < nodes.اندازه(); ++n) {
            uint64_t n_flops = 0;
            name += get_غوټه_fusion_name(nodes[n], نومونه[n], &n_flops);
            ټول_flops += n_flops;

            إذا (n != nodes.اندازه() - 1) {
                name += ", ";
            }
        }
        إذا (ټول_flops) {
            flops[name].ټېل_شا(ټول_flops);
        }
        timings[name].ټېل_شا(وخت);
    }

  خاص:
    std::map<std::string, std::ویکټور<uint64_t>> timings;
    std::map<std::string, std::ویکټور<uint64_t>> flops;
    uint32_t print_شمېر {};
};

بنية ggml_backend_vk_context {
    std::string name;

    vk_device device;

    size_t semaphore_idx, event_idx;
    ggml_vk_garbage_collector gc;
    size_t prealloc_اندازه_x, prealloc_اندازه_y, prealloc_اندازه_وېشل_k, prealloc_اندازه_جمع_کړه_rms_partials, prealloc_اندازه_جمع_کړه_rms_partials_دننه_واټن;
    vk_بفر prealloc_x, prealloc_y, prealloc_وېشل_k, prealloc_جمع_کړه_rms_partials, sync_staging;
    vk::Fence fence, almost_ready_fence;
    bool submit_pending {};
    bool almost_ready_fence_pending {};
    // Set before op_add and unset after op_rms_norm to indicate that the add should
    // write partial sums to accumulate the square of the vector components
    bool do_جمع_کړه_rms_partials_دننه_واټن_calculation;
    bool do_جمع_کړه_rms_partials;

    uint64_t وروستی_ټول_flops {UINT64_MAX};

    // Cache most recent tensor that was converted into prealloc_y, and what pipeline it used to convert.
    vk_pipeline_struct * prealloc_y_وروستی_pipeline_کارول_شوی {};
    ثابت ggml_tensor * prealloc_y_وروستی_tensor_کارول_شوی {};
    // True when the K dimension in prealloc_y is padded.
    bool prealloc_y_وروستی_k_padded {};

    // Track which nodes have been used since the last sync, and whether they were written to
    std::ویکټور<ثابت ggml_tensor *> unsynced_nodes_written;
    std::ویکټور<ثابت ggml_tensor *> unsynced_nodes_read;
    // Track which prealloc buffers have pending reads that need to be synchronized.
    // These are checked before writing to the buffer (and call ggml_vk_sync_buffers if set),
    // and set to true after the buffer contents are consumed.
    bool prealloc_x_need_sync, prealloc_y_need_sync, prealloc_وېشل_k_need_sync;

    vk_context_ref compute_ctx;

    vk_context_ref transfer_ctx;
    vk_semaphore transfer_semaphore;
    uint64_t transfer_semaphore_وروستی_submitted {};

    std::ویکټور<vk_context_ref> tensor_ctxs;

    std::ویکټور<vk::DescriptorPool> descriptor_pools;
    std::ویکټور<vk::Descriptor_سېټ> descriptor_sets;
    // last bindings written to each set; descriptor_sets is append-only so an index always names the same set
    std::ویکټور<std::ویکټور<vk::Descriptor_بفر_Info>> descriptor_سېټ_bindings;
    uint64_t descriptor_سېټ_bindings_destroy_شمېر {};
    uint32_t descriptor_سېټ_idx {};
    uint32_t pipeline_descriptor_سېټ_requirements {};

    vk_command_pool compute_cmd_pool;
    vk_command_pool transfer_cmd_pool;

    // number of additional consecutive nodes that are being fused with the
    // node currently being processed
    صحيح عدد_additional_fused_ops {};
    // Bitmask of which fused ops need to write an intermediate value to memory.
    // Bit 'i' means nodes[start_of_fusion + i] writes to memory.
    // If there's no fusion, bit 0 is still set.
    صحيح fused_ops_write_ماسک {};
    topk_moe_ډېرپېښ fused_topk_moe_ډېرپېښ {};
    bool fused_topk_moe_scale {};
    // QSA indexer gather+add+top_k fused into one radix-select
    bool fused_topk_qsa {};
    bool fused_hc_post_gate {};
    rms_norm_ډېرپېښ fused_rms_norm_ډېرپېښ {RMS_NORM_شمېر};

    // for GGML_VK_PERF_LOGGER
    std::unique_ptr<vk_perf_logger> perf_logger;
    vk::QueryPool query_pool;
    std::ویکټور<ثابت حرف *> query_fusion_نومونه;
    std::ویکټور<صحيح> query_fusion_غوټه_شمېر;
    std::ویکټور<ggml_tensor *> query_nodes;
    std::ویکټور<صحيح> query_غوټه_idx;
    int32_t عدد_queries {};
    int32_t query_idx {};
};

بنية ggml_backend_vk_بفر_context {
    vk_device_ref device;
    vk_بفر dev_بفر;
    std::string name;

    ggml_backend_vk_بفر_context(vk_device_ref device, vk_بفر&& dev_بفر, std::string& name) :
        device(device),
        dev_بفر(dev_بفر),
        name(name) {
    }

    ~ggml_backend_vk_بفر_context();

};

بنية vk_host_ویش {
    فراغ *    بنسټ = nullptr;
    size_t    اندازه = 0;
    vk_بفر views[GGML_VK_MAX_DEVICES];  // indexed by device->idx; null = not (yet) mapped here
};

بنية vk_instance_t {
    vk::Instance instance;

    bool debug_utils_support = false;  // VK_EXT_debug_utils enabled
    PFN_vk_سېټ_Debug_Utils_Object_Name_EXT pfn_vk_سېټ_Debug_Utils_Object_Name_EXT = {};
    PFN_vk_کتار_Begin_Debug_Utils_Label_EXT pfn_vk_کتار_Begin_Debug_Utils_Label_EXT = {};
    PFN_vk_کتار_پای_ټکی_Debug_Utils_Label_EXT   pfn_vk_کتار_پای_ټکی_Debug_Utils_Label_EXT   = {};
    PFN_vkCmdBeginDebugUtilsLabelEXT   pfn_vkCmdBeginDebugUtilsLabelEXT   = {};
    PFN_vk_Cmd_پای_ټکی_Debug_Utils_Label_EXT pfn_vk_Cmd_پای_ټکی_Debug_Utils_Label_EXT = {};
    PFN_vk_Cmd_ورننباسه_Debug_Utils_Label_EXT  pfn_vk_Cmd_ورننباسه_Debug_Utils_Label_EXT  = {};

    std::ویکټور<size_t> device_indices;
    std::ویکټور<bool>   device_supports_membudget;
    std::ویکټور<bool>   device_host_بفر_import;
    vk_device devices[GGML_VK_MAX_DEVICES];

    std::ویکټور<vk_host_ویش> host_allocations;
    std::shared_mutex               host_allocations_mutex;

    size_t host_import_alignment = 0;
    uint32_t host_import_failed = 0;  // bitmask by device->idx, guarded by host_allocations_mutex

    bool multi_device = false;
};

تعريفنوع فراغ (*ggml_vk_func_t)(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);

ساكن constexpr uint32_t k_Spv_Op_Cooperative_ماتریکس_بارول_Tensor_NV = 5367;

ساكن constexpr uint32_t k_Spv_Capability_Cooperative_ماتریکس_Decode_ویکټور_NV = 5447;

ساكن constexpr uint32_t k_Spv_Tensor_Addressing_Decode_ویکټور_Func_بټ = 0x4;

بنية vk_fa_tuning_params {
    Fa_کوډ_لار لار;
    uint32_t workgroup_اندازه;
    uint32_t subgroup_اندازه;
    uint32_t بلاک_لیکې;
    uint32_t بلاک_ستون_شمېر;
    uint32_t d_وېشل;
    uint32_t لیکه_وېشل;
    bool shmem_staging;
    bool disable_subgroups;
    uint32_t حد_occupancy_shmem;

    فراغ print() ثابت {
        std::cerr << "path=" << لار << " workgroup_size=" << workgroup_اندازه << " subgroup_size=" << subgroup_اندازه <<
                     " block_rows=" << بلاک_لیکې << " block_cols=" << بلاک_ستون_شمېر << " d_split=" << d_وېشل <<
                     " row_split=" << لیکه_وېشل << " shmem_staging=" << shmem_staging << " disable_subgroups=" << disable_subgroups <<
                     " limit_occupancy_shmem=" << حد_occupancy_shmem << std::endl;
    }
};

بنية GpuPipelineConfig {
    // GPU architecture identifier.
    // Example: vk_device_architecture::AMD_GCN
    vk_device_architecture arch;

    // Mapping of pipeline names to their specific subgroup sizes.
    // Example: {"soft_max_f32", 64}
    std::unordered_map<std::string, uint32_t> pipelines;

    // Default subgroup size for this GPU.
    // Defaults to 0 if not explicitly provided.
    uint32_t default_subgroup_اندازه = 0;
};

ساكن constexpr uint32_t RDNA_DEFAULT_SUBGROUP_اندازه = 32;

بنية CompileTask {
    vk_pipeline pipeline;
    size_t spv_اندازه;
    ثابت فراغ * spv_معلومات;
    std::string entrypoint;
    uint32_t parameter_شمېر;
    std::array<uint32_t, 3> wg_denoms;
    std::ویکټور<uint32_t> specialization_constants;
    bool disable_robustness;
    bool require_ډک_subgroups;
    uint32_t required_subgroup_اندازه;
};

بنية ggml_vk_debug_label {
    // at most one of these is set, depending on the scope the label was opened in
    vk_context_struct * subctx {};
    vk_کتار_handle *   qhandle {};

    // one region per dispatch, e.g. "matmul_q4_k_f32_f16acc_aligned_m (192,8,1)".
    // RGP cannot recover the pipeline name on its own, it only has the hash
    ggml_vk_debug_label(vk_context & ctx, ثابت std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2);


    // one region per graph node
    // fused nodes are joined with '+', e.g. "RMS_NORM+MUL+ROPE Qcur-19"
    ggml_vk_debug_label(vk_context & ctx, ثابت ggml_cgraph * cgraph, صحيح غوټه_idx, صحيح n_fused);


    // one region per graph evaluation, opened on the queue instead of a command buffer
    // so it spans every submit the evaluation makes
    ggml_vk_debug_label(vk_کتار_handle * handle, ثابت حرف * name);


    // call before the command buffer can end, the destructor covers the rest
    فراغ close();


    ~ggml_vk_debug_label() {
        close();
    }

    ggml_vk_debug_label(ثابت ggml_vk_debug_label &) = delete;
    ggml_vk_debug_label & چلوونکی=(ثابت ggml_vk_debug_label &) = delete;

خاص:
    // the constructors check this too, so the name is not built when markers are off
    فراغ begin(vk_context & ctx, ثابت std::string & name);

};

#تعریف_کړه ناکارول_شوی GGML_ناکارول_شوی

بنية ggml_backend_vk_device_context {
    size_t device;
    std::string name;
    std::string description;
    bool is_integrated_gpu;
    std::string pci_bus_id;
    صحيح op_offload_min_batch_اندازه;
};
