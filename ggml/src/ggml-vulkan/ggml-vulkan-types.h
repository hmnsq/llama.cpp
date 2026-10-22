#ہدایت once

#ضمن "ggml-vulkan.h"

#ضمن <vulkan/vulkan_core.h>

#إذا defined(GGML_VULKAN_لگاتار_TESTS) || defined(GGML_VULKAN_CHECK_RESULTS)
#ضمن <chrono>
#ضمن "ggml-cpu.h"
#جے_انت

#تعریف_کرو VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#إذا VK_سرخی_نسخہ >= 301
namespace vk::detail { class DispatchLoaderDynamic; }
using vk::detail::DispatchLoaderDynamic;
#وإلا
namespace vk { class DispatchLoaderDynamic; }
using vk::DispatchLoaderDynamic;
#جے_انت

DispatchLoaderDynamic & ggml_vk_default_dispatcher();

#تعریف_کرو VULKAN_HPP_DEFAULT_DISPATCHER ggml_vk_default_dispatcher()

#ضمن <vulkan/vulkan.hpp>

#جے_بنا_تعریف VK_NV_cooperative_میٹرکس_decode_سمتیہ
#تعریف_کرو VK_NV_cooperative_میٹرکس_decode_سمتیہ 1
#تعریف_کرو VK_NV_COOPERATIVE_میٹرکس_DECODE_سمتیہ_EXTENSION_NAME "VK_NV_cooperative_matrix_decode_vector"
#تعریف_کرو VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_میٹرکس_DECODE_سمتیہ_FEATURES_NV ((VkStructureType)1000689000)
تعريفنوع بنية Vk_Physical_Device_Cooperative_میٹرکس_Decode_سمتیہ_Features_NV {
    VkStructureType    sType;
    فراغ*              p_اگلا;
    VkBool32           cooperative_میٹرکس_Decode_سمتیہ;
} Vk_Physical_Device_Cooperative_میٹرکس_Decode_سمتیہ_Features_NV;
#جے_انت

#إذا __has_include(<spirv/unified1/spirv.hpp>)
#    include <spirv/unified1/spirv.hpp>
#وإلاإذا __has_include(<spirv-سرخیاں/spirv.hpp>)
#    include <spirv-headers/spirv.hpp>
#وإلاإذا __has_include(<spirv.hpp>)
#    include <spirv.hpp>
#وإلا
     // Fallback to let the compiler throw a standard "file not found" error
#    include <spirv/unified1/spirv.hpp>
#جے_انت

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
#جے_انت

#إذا !defined(YIELD)
#تعریف_کرو YIELD()
#جے_انت

#ضمن "ggml-impl.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-vulkan-shaders.hpp"

#إذا !defined(VK_KHR_shader_bfloat16)

#تعریف_کرو VK_KHR_shader_bfloat16 1
#تعریف_کرو VK_KHR_SHADER_BFLOAT16_SPEC_نسخہ                          1
#تعریف_کرو VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME                        "VK_KHR_shader_bfloat16"
#تعریف_کرو VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR ((VkStructureType)1000141000)
#تعریف_کرو VK_COMPONENT_TYPE_BFLOAT16_KHR                               ((VkComponentTypeKHR)1000141000)

تعريفنوع بنية VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType                       sType;
    فراغ*                                 p_اگلا;
    VkBool32                              shaderBFloat16Type;
    VkBool32                              shader_BFloat16_Dot_ضرب_حاصل;
    VkBool32                              shader_BFloat16_Cooperative_میٹرکس;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;
#جے_انت

#إذا !defined(VK_VALVE_shader_mixed_float_dot_ضرب_حاصل)
#تعریف_کرو VK_VALVE_shader_mixed_float_dot_ضرب_حاصل 1
#تعریف_کرو VK_VALVE_SHADER_MIXED_FLOAT_DOT_ضرب_حاصل_SPEC_نسخہ 1
#تعریف_کرو VK_VALVE_SHADER_MIXED_FLOAT_DOT_ضرب_حاصل_EXTENSION_NAME "VK_VALVE_shader_mixed_float_dot_product"
#تعریف_کرو VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_ضرب_حاصل_FEATURES_VALVE ((VkStructureType)1000673000)
تعريفنوع بنية Vk_Physical_Device_Shader_Mixed_Float_Dot_ضرب_حاصل_Features_VALVE {
    VkStructureType    sType;
    فراغ*              p_اگلا;
    VkBool32           shader_Mixed_Float_Dot_ضرب_حاصل_Float16_Acc_Float32;
    VkBool32           shader_Mixed_Float_Dot_ضرب_حاصل_Float16_Acc_Float16;
    VkBool32           shader_Mixed_Float_Dot_ضرب_حاصل_BFloat16_Acc;
    VkBool32           shader_Mixed_Float_Dot_ضرب_حاصل_Float8_Acc_Float32;
} Vk_Physical_Device_Shader_Mixed_Float_Dot_ضرب_حاصل_Features_VALVE;
#جے_انت

#إذا !defined(VK_EXT_shader_ocp_microscaling_types)
#تعریف_کرو VK_EXT_shader_ocp_microscaling_types 1
#تعریف_کرو VK_EXT_SHADER_OCP_MICROSCALING_TYPES_SPEC_نسخہ 1
#تعریف_کرو VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME "VK_EXT_shader_ocp_microscaling_types"
#تعریف_کرو VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT ((VkStructureType)1000672000)
تعريفنوع بنية VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT {
    VkStructureType    sType;
    فراغ*              p_اگلا;
    VkBool32           shaderFloat4;
    VkBool32           shaderFloat6;
    VkBool32           shaderFloat8UnsignedE8M0;
    VkBool32           shaderMXInt8;
} VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT;
#جے_انت

#إذا !defined(VK_EXT_shader_float8)
#تعریف_کرو VK_EXT_shader_float8 1
#تعریف_کرو VK_EXT_SHADER_FLOAT8_SPEC_نسخہ 1
#تعریف_کرو VK_EXT_SHADER_FLOAT8_EXTENSION_NAME "VK_EXT_shader_float8"
#تعریف_کرو VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT ((VkStructureType)1000567000)
تعريفنوع بنية VkPhysicalDeviceShaderFloat8FeaturesEXT {
    VkStructureType    sType;
    فراغ*              p_اگلا;
    VkBool32           shaderFloat8;
    VkBool32           shader_Float8_Cooperative_میٹرکس;
} VkPhysicalDeviceShaderFloat8FeaturesEXT;
#جے_انت

#جے_بنا_تعریف VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME
#تعریف_کرو VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME "VK_KHR_internally_synchronized_queues"
#تعریف_کرو VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR ((VkStructureType)1000504000)
#تعریف_کرو VK_DEVICE_قطار_CREATE_INTERNALLY_SYNCHRONIZED_ثنائی_ہندسہ_KHR ((Vk_Device_قطار_Create_Flag_بٹ)0x00000004)

// Compile-time constant guaranteed; no runtime initialization overhead
ساكن constexpr vk::Device_قطار_Create_Flag_بٹ eInternallySynchronizedKHR =
    static_cast<vk::Device_قطار_Create_Flag_بٹ>(0x00000004);

تعريفنوع بنية VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR {
    VkStructureType    sType;
    فراغ* p_اگلا;
    VkBool32           internallySynchronizedQueues;
} VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR;
#وإلا
ساكن constexpr vk::Device_قطار_Create_Flag_بٹ eInternallySynchronizedKHR = vk::Device_قطار_Create_Flag_بٹ::eInternallySynchronizedKHR;
#جے_انت

#تعریف_کرو ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))

#تعریف_کرو CEIL_DIV(M, N) (((M) / (N)) + (((M) % (N)) != 0))

ساكن bool is_pow2(uint32_t x) { أرجع x > 1 && (x & (x-1)) == 0; }

#تعریف_کرو VK_VENDOR_ID_AMD 0x1002

#تعریف_کرو VK_VENDOR_ID_APPLE 0x106b

#تعریف_کرو VK_VENDOR_ID_INTEL 0x8086

#تعریف_کرو VK_VENDOR_ID_NVIDIA 0x10de

#تعریف_کرو VK_VENDOR_ID_QUALCOMM 0x5143

#تعریف_کرو VK_VENDOR_ID_SAMSUNG 0x144d

#تعریف_کرو VK_DEVICE_DESCRIPTOR_POOL_ناپ 256

#تعریف_کرو VK_CHECK(err, پیغام, dev)                                     \
    افعل {                                                            \
        vk::نتیجہ_2 err_;                                            \
        try {                                                       \
            err_ = (err);                                           \
        } catch (vk::Device_گواچا_Error &) {                           \
            ggml_vk_print_device_گواچا_info(dev);                    \
            GGML_LOG_ERROR("ggml_vulkan: %s at %s:%d\n",            \
                #err, __FILE__, __LINE__);                          \
            throw;                                                  \
        }                                                           \
        إذا (err_ != vk::نتیجہ_2::eSuccess) {                         \
            GGML_LOG_ERROR("ggml_vulkan: %s error %s at %s:%d\n",   \
                #err, منزل_string(err_).c_str(), __FILE__, __LINE__); \
            throw vk::SystemError(vk::make_error_کوڈ(err_),        \
                "ggml_vulkan: " پیغام);                               \
        }                                                           \
    } طالما (0)

#جے_تعریف_شدہ GGML_VULKAN_DEBUG
#تعریف_کرو VK_LOG_DEBUG(پیغام) std::cerr << پیغام << std::endl
#وإلا
#تعریف_کرو VK_LOG_DEBUG(پیغام) ((فراغ) 0)
#جے_انت // GGML_VULKAN_DEBUG

#تعریف_کرو MAX_PARAMETER_گنتی 12

#تعریف_کرو MAX_FUSED_ADDS (MAX_PARAMETER_گنتی - 3)

بنية vk_pipeline_struct;

تعريفنوع std::shared_ptr<بنية vk_pipeline_struct> vk_pipeline;

بنية vk_pipeline_struct {
    std::string name;
    vk::ShaderModule shader_module;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t ڈالو_constant_ناپ;
    uint32_t parameter_گنتی;
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
    uint32_t رجسٹر_گنتی {};

#إذا defined(VK_EXT_shader_64bit_indexing)
    bool is_64b_indexing {};
#جے_انت
    // linked list of pipelines for multiple compilation variants.
    // currently only used to compile a 64-bit indexing variant.
    vk_pipeline اگلا;
};

تعريفنوع std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;

بنية vk_matmul_pipeline_کلید {
    ggml_type type_a;
    ggml_type type_b;
    bool mul_mat_id;
    bool f16acc;

    bool عامل<(ثابت vk_matmul_pipeline_کلید & o) ثابت {
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
    std::سمتیہ<uint32_t> warptile;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
};

using matmul_tile_selector_t = std::function<uint32_t(
    uint32_t m, uint32_t n, uint32_t k, uint32_t shader_core_گنتی,
    ثابت std::سمتیہ<vk_matmul_pipeline_pair>& configs)>;

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
    bool وچ_use = false;
};

بنية vk_قطار;

بنية vk_command_pool {
    فراغ init(vk_device& device, vk_قطار *q_);
    فراغ destroy(vk::Device& device);

    vk::CommandPool pool;
    // Using deque so the pointers to command buffers
    // remain valid even if we add more
    std::deque<vk_command_بفر> cmd_buffers;

    vk_قطار *q;

    size_t buffers_وچ_use() ثابت {
        أرجع std::گنتی_if(cmd_buffers.begin(), cmd_buffers.سرا(),
            [](ثابت خودکار& cb) { أرجع cb.وچ_use; });
    }
};

بنية vk_قطار_handle {
    vk::قطار_2 قطار;
    vk_device_ref device;
    std::mutex * device_submit_mutex = nullptr;
    مجازی فراغ submit(vk::ArrayProxy<ثابت vk::SubmitInfo> submits, vk::Fence fence) = 0;
    مجازی فراغ جندرا()   {}   // no-op by default (internally synchronized case)
    مجازی فراغ unlock() {}
    مجازی ~vk_قطار_handle() = افتراضي;
};

بنية vk_قطار_handle_synchronized : vk_قطار_handle {
    std::mutex mutex;
    فراغ submit(vk::ArrayProxy<ثابت vk::SubmitInfo> submits, vk::Fence fence) override;

    فراغ جندرا()   override { mutex.جندرا(); }
    فراغ unlock() override { mutex.unlock(); }
};

بنية vk_قطار_handle_unsynchronized : vk_قطار_handle {
    فراغ submit(vk::ArrayProxy<ثابت vk::SubmitInfo> submits, vk::Fence fence) override;

    // lock()/unlock() inherited no-ops
};

بنية vk_قطار {
    uint32_t قطار_family_اشاریہ;
    std::shared_ptr<vk_قطار_handle> handle;

    vk_command_pool cmd_pool;

    vk::PipelineStageFlags stage_flags;

    bool transfer_only;
};

ساكن constexpr uint32_t mul_mat_vec_max_خانے_گنتی = 8;

ساكن constexpr uint32_t p021_max_gqa_تناسب = 8;

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
    CONV_SHAPE_گنتی,
};

بنية vk_conv_بلاک_ناپ {
    uint32_t K;
    uint32_t NPQ;
    uint32_t CRS;
};

مضمن vk_conv_بلاک_ناپ vk_conv_بلاک_sizes[CONV_SHAPE_گنتی] = {
    // K   NPQ  CRS
    { 128, 128, 16 }, // CONV_SHAPE_128x128
    {  64,  32, 32 }, // CONV_SHAPE_64x32
    {  32, 256, 16 }, // CONV_SHAPE_32x256
    {  64, 128, 16 }, // CONV_SHAPE_64x128
};

تعداد dmmv_wg_sizes {
    DMMV_WG_ناپ_SUBGROUP,
    DMMV_WG_ناپ_LARGE,
    DMMV_WG_ناپ_گنتی,
};

تعداد Fa_کوڈ_رستہ {
    FA_SCALAR,
    FA_COOPMAT1,
    FA_COOPMAT2,
};

بنية vk_fa_pipeline_state {
    uint32_t HSK, HSV;
    uint32_t Br, Bc;
    uint32_t D_ونڈو, صف_ونڈو;
    bool shmem_staging;
    Fa_کوڈ_رستہ رستہ;
    uint32_t workgroup_ناپ, subgroup_ناپ;
    bool aligned;
    bool f32acc;
    uint32_t flags;
    uint32_t حد_بندی_occupancy_shmem;
    ggml_type k_type;
    ggml_type v_type;

    bool عامل<(ثابت vk_fa_pipeline_state &b) ثابت {
        أرجع std::tie(HSK, HSV, Br, Bc, D_ونڈو, صف_ونڈو, shmem_staging, رستہ, workgroup_ناپ, subgroup_ناپ, aligned, f32acc, flags, حد_بندی_occupancy_shmem, k_type, v_type) <
               std::tie(b.HSK, b.HSV, b.Br, b.Bc, b.D_ونڈو, b.صف_ونڈو, b.shmem_staging, b.رستہ, b.workgroup_ناپ, b.subgroup_ناپ, b.aligned, b.f32acc, b.flags, b.حد_بندی_occupancy_shmem, b.k_type, b.v_type);
    }
};

بنية vk_conv2d_pipeline_state {
    vk_conv2d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t p0, uint32_t p1, uint32_t d0, uint32_t d1, uint32_t KW, uint32_t KH, uint32_t aligned)
        : s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), KW(KW), KH(KH), aligned(aligned) {}

    uint32_t s0, s1, p0, p1, d0, d1, KW, KH;
    // when set, shader can skip K/CRS/NPQ bounds checks and address clamps
    uint32_t aligned;

    bool عامل<(ثابت vk_conv2d_pipeline_state &b) ثابت {
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

    bool عامل<(ثابت vk_conv3d_pipeline_state &b) ثابت {
        أرجع std::tie(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned) <
               std::tie(b.s0, b.s1, b.s2, b.p0, b.p1, b.p2, b.d0, b.d1, b.d2, b.KW, b.KH, b.KD, b.aligned);
    }
};

بنية vk_solve_tri_pipeline_state {
    vk_solve_tri_pipeline_state(uint32_t N, uint32_t K)
        : N(N), K(K) {}

    uint32_t N, K;

    bool عامل<(ثابت vk_solve_tri_pipeline_state &b) ثابت {
        أرجع std::tie(N, K) <
               std::tie(b.N, b.K);
    }
};

تعداد shader_reduction_منوال {
    SHADER_REDUCTION_منوال_SHMEM,
    SHADER_REDUCTION_منوال_HYBRID,
    SHADER_REDUCTION_منوال_SUBGROUP,
    SHADER_REDUCTION_منوال_گنتی,
};

ساكن constexpr uint32_t عدد_argsort_pipelines = 11;

ساكن constexpr uint32_t عدد_topk_moe_pipelines = 10;

ساكن constexpr uint32_t عدد_topk_pipelines = 11;

ساكن constexpr std::initializer_list<ggml_op> topk_moe_early_softmax_norm{ GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_صفاں, GGML_OP_RESHAPE,
                                                                             GGML_OP_جوڑ_صفاں, GGML_OP_CLAMP,    GGML_OP_DIV,
                                                                             GGML_OP_RESHAPE };

ساكن constexpr std::initializer_list<ggml_op> topk_moe_sigmoid_norm_bias{ GGML_OP_UNARY,    GGML_OP_RESHAPE,  GGML_OP_جوڑو,
                                                                            GGML_OP_ARGSORT,  GGML_OP_VIEW,     GGML_OP_GET_صفاں,
                                                                            GGML_OP_RESHAPE,  GGML_OP_جوڑ_صفاں, GGML_OP_CLAMP,
                                                                            GGML_OP_DIV,      GGML_OP_RESHAPE };

ساكن constexpr std::initializer_list<ggml_op> topk_moe_sqrt_softplus_norm_bias{ GGML_OP_UNARY,    GGML_OP_SQRT,
                                                                                  GGML_OP_RESHAPE,  GGML_OP_جوڑو,
                                                                                  GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                                  GGML_OP_GET_صفاں, GGML_OP_RESHAPE,
                                                                                  GGML_OP_جوڑ_صفاں, GGML_OP_CLAMP,
                                                                                  GGML_OP_DIV,      GGML_OP_RESHAPE };

ساكن constexpr std::initializer_list<ggml_op> topk_moe_early_softmax     { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_صفاں };

ساكن constexpr std::initializer_list<ggml_op> topk_moe_late_softmax      { GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                             GGML_OP_GET_صفاں, GGML_OP_RESHAPE,
                                                                             GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

ساكن constexpr std::initializer_list<ggml_op> snake_نمونہ              { GGML_OP_MUL,      GGML_OP_SIN,
                                                                             GGML_OP_SQR,      GGML_OP_MUL,
                                                                             GGML_OP_جوڑو };

ساكن constexpr std::initializer_list<ggml_op> topk_qsa_نمونہ { GGML_OP_GET_صفاں, GGML_OP_ترتیب_بدلو,
                                                                   GGML_OP_CONT,     GGML_OP_CPY,
                                                                   GGML_OP_RESHAPE,  GGML_OP_جوڑو,
                                                                   GGML_OP_چوٹی_K };

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_qsa_کنارے {
    { 1, 0, 0 }, // permute->src[0] == get_rows
    { 2, 0, 1 }, // cont->src[0]    == permute
    { 4, 0, 3 }, // reshape->src[0] == cpy (mask cast)
    { 5, 0, 2 }, // add->src[0]     == cont
    { 5, 1, 4 }, // add->src[1]     == reshape
    { 6, 0, 5 }, // top_k->src[0]   == add
};

ساكن constexpr std::initializer_list<ggml_op> rms_norm_mul_جوڑو_mul_نمونہ { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_جوڑو, GGML_OP_MUL };

ساكن constexpr std::initializer_list<ggml_op> rms_norm_mul_جوڑو_نمونہ     { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_جوڑو };

ساكن constexpr std::initializer_list<ggml_op> rms_norm_mul_rope_view_سیٹ_صفاں_نمونہ { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_سیٹ_صفاں };

ساكن constexpr std::initializer_list<ggml_op> rms_norm_view_سیٹ_صفاں_نمونہ { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_سیٹ_صفاں };

ساكن constexpr std::initializer_list<ggml_op> rope_view_سیٹ_صفاں_نمونہ { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_سیٹ_صفاں };

// scale_out*sigmoid(scale_in*x) as the hc_post weights (qwen4exp hc_combine)
ساكن constexpr std::initializer_list<ggml_op> hc_post_gate_نمونہ { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> hc_post_gate_کنارے {
    { 1, 0, 0 }, // sigmoid->src[0] == scale
    { 2, 0, 1 }, // scale->src[0]   == sigmoid
    { 3, 2, 2 }, // hc_post->src[2] == scale (post)
};

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_moe_early_softmax_norm_کنارے {
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

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_moe_sigmoid_norm_bias_کنارے {
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

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_moe_sqrt_softplus_norm_bias_کنارے {
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

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_moe_early_softmax_کنارے {
    { 1, 0, 0 }, // reshape->src[0]  == softmax
    { 2, 0, 0 }, // argsort->src[0]  == softmax
    { 3, 0, 2 }, // view->src[0]     == argsort
    { 4, 0, 1 }, // get_rows->src[0] == reshape
    { 4, 1, 3 }, // get_rows->src[1] == view
};

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> topk_moe_late_softmax_کنارے {
    { 1, 0, 0 }, // view->src[0]     == argsort
    { 2, 1, 1 }, // get_rows->src[1] == view
    { 3, 0, 2 }, // reshape->src[0]  == get_rows
    { 4, 0, 3 }, // soft_max->src[0] == reshape
    { 5, 0, 4 }, // reshape->src[0]  == soft_max
};

تعداد topk_moe_منوال {
    TOPK_MOE_EARLY_SOFTMAX,
    TOPK_MOE_EARLY_SOFTMAX_NORM,
    TOPK_MOE_LATE_SOFTMAX,
    TOPK_MOE_SIGMOID_NORM_BIAS,
    TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS,
    TOPK_MOE_گنتی,
};

تعداد rms_norm_منوال {
    RMS_NORM_MUL,
    RMS_NORM_MUL_جوڑو,
    RMS_NORM_MUL_جوڑو_MUL,
    RMS_NORM_MUL_ROPE,
    RMS_NORM_MUL_ROPE_VIEW_سیٹ_صفاں,
    RMS_NORM_VIEW_سیٹ_صفاں,
    RMS_NORM_گنتی,
};

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> rope_view_سیٹ_صفاں_کنارے {
    { 1, 0, 0 }, // view->src[0]     == rope
    { 2, 0, 1 }, // set_rows->src[0] == view
};

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> rms_norm_mul_rope_view_سیٹ_صفاں_کنارے {
    { 1, 0, 0 }, // mul->src[0]      == rms
    { 2, 0, 1 }, // rope->src[0]     == mul
    { 3, 0, 2 }, // view->src[0]     == rope
    { 4, 0, 3 }, // set_rows->src[0] == view
};

ساكن constexpr std::initializer_list<std::array<صحيح, 3>> rms_norm_view_سیٹ_صفاں_کنارے {
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

class vk_حافظہ_logger;

بنية vk_device_struct {
    std::recursive_mutex mutex;
    std::mutex قطار_submit_mutex;

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
    uint64_t max_حافظہ_ونڈ_ناپ;
    uint64_t max_بفر_ناپ;
    uint64_t suballocation_بلاک_ناپ;
    uint64_t min_imported_host_اشارہ_گر_alignment;
    bool external_حافظہ_host {};
    bool fp16;
    bool bf16;
    bool pipeline_robustness;
    bool حافظہ_ترجیح;
    vk::Device device;
    uint32_t vendor_id;
    vk::DriverId driver_id;
    vk_device_architecture architecture;
    std::unique_ptr<vk_قطار> compute_قطار;
    std::unique_ptr<vk_قطار> transfer_قطار;
    bool single_قطار;
    bool support_async;
    bool async_use_transfer_قطار;
    bool has_internally_synchronized_queues = false;
    uint32_t subgroup_ناپ;
    uint32_t subgroup_ناپ_log2;
    uint32_t shader_core_گنتی;
    bool uma;
    bool prefer_host_حافظہ;
    bool float_controls_rte_fp16;
    bool float_controls_denorm_preserve_fp16;
    bool subgroup_basic;
    bool subgroup_arithmetic;
    bool subgroup_shuffle;
    bool subgroup_ballot;
    bool subgroup_clustered;
    bool subgroup_vote;
    bool multi_جوڑو;
    bool shader_int64;
    bool بفر_device_پتہ;
    bool vulkan_حافظہ_model;

    bool جوڑو_rms_fusion;
    uint32_t partials_binding_alignment;
    uint32_t max_nodes_فی_submit;

    bool shader_64b_indexing;

    bool integer_dot_ضرب_حاصل;
    // 0: default, 1: force mmvq, -1: disable mmvq
    int32_t mmvq_منوال;

    bool subgroup_ناپ_control;
    uint32_t subgroup_min_ناپ;
    uint32_t subgroup_max_ناپ;
    bool subgroup_require_بھریا_support;

    // floor(log2(maxComputeWorkGroupInvocations))
    uint32_t max_workgroup_ناپ_log2 {};

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
    bool coopmat2_decode_سمتیہ;

    bool dot2_f16 {};
    bool ocp_fp4 {};

    bool pipeline_executable_properties_support {};

    bool device_fault {};
    PFN_vkGetDeviceFaultInfoEXT pfn_vkGetDeviceFaultInfoEXT {};

    bool serialize_submissions {};

    ثابت ggml_cgraph * diag_cgraph {};
    صحيح diag_prev_شروع = -1;
    صحيح diag_prev_سرا = -1;

    size_t idx;

    bool mul_mat_l[GGML_TYPE_گنتی];
    bool mul_mat_m[GGML_TYPE_گنتی];
    bool mul_mat_s[GGML_TYPE_گنتی];
    bool mul_mat_id_l[GGML_TYPE_گنتی];
    bool mul_mat_id_m[GGML_TYPE_گنتی];
    bool mul_mat_id_s[GGML_TYPE_گنتی];

    // Separate flags for the q8_1 (integer dot) mmq path, whose shader uses
    // a different shared-memory layout than the float matmul shaders.
    bool mul_mat_l_int[GGML_TYPE_گنتی];
    bool mul_mat_m_int[GGML_TYPE_گنتی];
    bool mul_mat_s_int[GGML_TYPE_گنتی];
    bool mul_mat_id_l_int[GGML_TYPE_گنتی];
    bool mul_mat_id_m_int[GGML_TYPE_گنتی];
    bool mul_mat_id_s_int[GGML_TYPE_گنتی];

    vk::Descriptor_سیٹ_Layout dsl;

    std::map<vk_matmul_pipeline_کلید, std::سمتیہ<vk_matmul_pipeline_pair>> pipeline_matmul;
    matmul_tile_selector_t matmul_tile_selector;
    matmul_tile_selector_t matmul_id_tile_selector;

    vk_pipeline pipeline_matmul_ونڈو_k_reduce;
    vk_pipeline pipeline_quantize_q8_1_x4;

    vk_pipeline pipeline_dequant[GGML_TYPE_گنتی];
    vk_pipeline pipeline_dequant_transpose[GGML_TYPE_گنتی]; // fused dequant+transpose for FA quant-KV
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_ناپ_گنتی][GGML_TYPE_گنتی][mul_mat_vec_max_خانے_گنتی];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[DMMV_WG_ناپ_گنتی][GGML_TYPE_گنتی][mul_mat_vec_max_خانے_گنتی];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[DMMV_WG_ناپ_گنتی][GGML_TYPE_گنتی];

    vk_pipeline pipeline_dequant_mul_mat_vec_q8_1_f32[DMMV_WG_ناپ_گنتی][GGML_TYPE_گنتی][mul_mat_vec_max_خانے_گنتی];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_q8_1_f32[DMMV_WG_ناپ_گنتی][GGML_TYPE_گنتی];

    vk_pipeline pipeline_mul_mat_vec_p021_f16_f32[p021_max_gqa_تناسب];
    vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
    vk_pipeline pipeline_get_صفاں[GGML_TYPE_گنتی];
    vk_pipeline pipeline_get_صفاں_f32[GGML_TYPE_گنتی];
    vk_pipeline pipeline_get_صفاں_پٹھ_f32;
    vk_pipeline pipeline_acc_f32;
    vk_pipeline pipeline_سیٹ_f32;

    // [src0 0=fp32,1=fp16][src1 0=fp32,1=fp16][dst 0=fp32,1=fp16]
    vk_pipeline pipeline_جوڑو[2][2][2];
    vk_pipeline pipeline_جوڑو_norepeat[2][2][2];
    vk_pipeline pipeline_sub[2][2][2];
    vk_pipeline pipeline_sub_norepeat[2][2][2];
    vk_pipeline pipeline_mul[2][2][2];
    vk_pipeline pipeline_mul_norepeat[2][2][2];
    vk_pipeline pipeline_div[2][2][2];
    vk_pipeline pipeline_div_norepeat[2][2][2];
    vk_pipeline pipeline_جوڑو_rms[2][2][2];
    vk_pipeline pipeline_جوڑو_rms_norepeat[2][2][2];

    // indexed by num_additional_fused_ops == num_adds - 1
    vk_pipeline pipeline_multi_جوڑو[MAX_FUSED_ADDS];
    vk_pipeline pipeline_multi_جوڑو_rms[MAX_FUSED_ADDS];

    vk_pipeline pipeline_جوڑو_id_f32;

    vk_pipeline pipeline_concat_i8, pipeline_concat_i16, pipeline_concat_i32, pipeline_concat_i64;
    vk_pipeline pipeline_upscale_سبھ_توں_نیڑے_f32, pipeline_upscale_bilinear_f32, pipeline_upscale_bicubic_f32, pipeline_upscale_bilinear_antialias_f32;
    vk_pipeline pipeline_scale_f32;
    vk_pipeline pipeline_log[2];
    vk_pipeline pipeline_tri[2];
    vk_pipeline pipeline_diag[2];
    vk_pipeline pipeline_clamp[2];
    vk_pipeline pipeline_pad_f32;
    vk_pipeline pipeline_pad_reflect_1d_f32;
    vk_pipeline pipeline_roll_f32;
    vk_pipeline pipeline_repeat_i32, pipeline_repeat_پٹھ_f32;
    vk_pipeline pipeline_repeat_i16;
    vk_pipeline pipeline_cpy_32, pipeline_cpy_f32_f16, pipeline_cpy_16, pipeline_cpy_f16_f32, pipeline_cpy_f32_bf16, pipeline_cpy_bf16_f32, pipeline_cpy_bf16_f16, pipeline_cpy_f32_i32, pipeline_cpy_i32_f32;
    vk_pipeline pipeline_contig_cpy_32, pipeline_contig_cpy_f32_f16, pipeline_contig_cpy_16, pipeline_contig_cpy_f16_f32, pipeline_contig_cpy_f32_bf16, pipeline_contig_cpy_bf16_f32, pipeline_contig_cpy_bf16_f16, pipeline_contig_cpy_f32_i32, pipeline_contig_cpy_i32_f32;
    vk_pipeline pipeline_cpy_f32_quant[GGML_TYPE_گنتی];
    vk_pipeline pipeline_cpy_quant_f32[GGML_TYPE_گنتی];
    vk_pipeline pipeline_cpy_transpose_16, pipeline_cpy_transpose_32;
    vk_pipeline pipeline_cpy_transpose_02_16, pipeline_cpy_transpose_02_32;
    // [src0 0=fp32,1=fp16][dst]
    vk_pipeline pipeline_سیٹ_صفاں_i32[2][GGML_TYPE_گنتی];
    vk_pipeline pipeline_سیٹ_صفاں_i64[2][GGML_TYPE_گنتی];
    vk_pipeline pipeline_norm_f32;
    vk_pipeline pipeline_گروہ_norm_f32;
    vk_pipeline pipeline_rms_norm_f32;
    vk_pipeline pipeline_rms_norm_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_جوڑو_f32;
    vk_pipeline pipeline_rms_norm_mul_جوڑو_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_جوڑو_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_جوڑو_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_سیٹ_صفاں_f32_f32;
    vk_pipeline pipeline_rms_norm_سیٹ_صفاں_f32_f16;
    vk_pipeline pipeline_rms_norm_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f16;
    vk_pipeline pipeline_rms_norm_پٹھ_f32;
    vk_pipeline pipeline_l2_norm_f32;

    // [src/dst 0=fp32,1=fp16]
    vk_pipeline pipeline_exp[2];
    vk_pipeline pipeline_expm1[2];
    vk_pipeline pipeline_elu[2];
    vk_pipeline pipeline_gelu[2];
    vk_pipeline pipeline_gelu_erf[2];
    vk_pipeline pipeline_gelu_تیز[2];
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
    vk_pipeline pipeline_قدم[2];
    vk_pipeline pipeline_گیڑا[2];
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
    vk_pipeline pipeline_geglu_تیز[2];

    vk_pipeline pipeline_leaky_relu[2];
    vk_pipeline pipeline_silu_پٹھ_f32;
    vk_pipeline pipeline_diag_نقاب_inf_f32;
    vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
    vk_pipeline pipeline_soft_max_f32_wg512, pipeline_soft_max_f32_f16_wg512;
    vk_pipeline pipeline_soft_max_پٹھ_f32;

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
    vk_pipeline pipeline_جوڑ_صفاں_f32;
    vk_pipeline pipeline_cross_entropy_loss_f32, pipeline_cross_entropy_loss_f32_wg512;
    vk_pipeline pipeline_cross_entropy_loss_پٹھ_f32, pipeline_cross_entropy_loss_پٹھ_f32_wg512;
    vk_pipeline pipeline_fwht_f32[4];
    vk_pipeline pipeline_cumsum_f32;
    vk_pipeline pipeline_cumsum_small_f32;
    vk_pipeline pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline_cumsum_multipass2_f32;
    vk_pipeline pipeline_argmax_f32;
    vk_pipeline pipeline_گنتی_equal_i32;
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
    vk_pipeline pipeline_باہرلا_prod_f32;
    vk_pipeline pipeline_snake_f32;
    vk_pipeline pipeline_snake_f16;
    vk_pipeline pipeline_snake_bf16;
    vk_pipeline pipeline_pool1d_f32;
    vk_pipeline pipeline_pool2d_f32;
    vk_pipeline pipeline_pool2d_subgroup_f32;
    vk_pipeline pipeline_rwkv_wkv6_f32;
    vk_pipeline pipeline_rwkv_wkv7_f32;
    vk_pipeline pipeline_gated_خطی_attn_f32;
    vk_pipeline pipeline_lightning_indexer_f32[GGML_TYPE_گنتی];
    // [size_idx][kda] where size_idx: 0=d16, 1=d32, 2=d64, 3=d128
    vk_pipeline pipeline_gated_delta_net[4][2];
    vk_pipeline pipeline_ssm_جانچ_f32_d128;
    vk_pipeline pipeline_ssm_جانچ_f32_d256;
    vk_pipeline pipeline_ssm_conv_f32;
    vk_pipeline pipeline_ssm_conv_silu_f32;
    vk_pipeline pipeline_ssm_conv_bias_silu_f32;
    vk_pipeline pipeline_opt_قدم_adamw_f32;
    vk_pipeline pipeline_opt_قدم_sgd_f32;
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f32[CONV_SHAPE_گنتی];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f16_f32[CONV_SHAPE_گنتی];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f32[CONV_SHAPE_گنتی];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f16_f32[CONV_SHAPE_گنتی];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f32[CONV_SHAPE_گنتی];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f16_f32[CONV_SHAPE_گنتی];
    vk_pipeline pipeline_conv2d_dw_whcn_f32, pipeline_conv2d_dw_whcn_f16_f32;
    vk_pipeline pipeline_conv2d_dw_cwhn_f32, pipeline_conv2d_dw_cwhn_f16_f32;

    std::map<vk_fa_pipeline_state, vk_pipeline> pipeline_flash_attn_f32_f16;

    std::map<std::pair<uint32_t, uint32_t>, vk_pipeline> pipeline_fa_نقاب_opt;

    vk_pipeline pipeline_fa_sparse_compact;
    vk_pipeline pipeline_fa_sparse_compact_subgroup;
    bool fa_sparse_compact_use_subgroups;

    vk_pipeline pipeline_flash_attn_ونڈو_k_reduce;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, std::pair<vk_pipeline, vk_pipeline>> pipeline_xe_fa_decode_dual_phases;
    vk_pipeline pipeline_گنتی_experts;

    // [2] is for whether to take n_experts from spec constant (0) or push constant (1)
    vk_pipeline pipeline_topk_moe[عدد_topk_moe_pipelines][2];

    std::سمتیہ<vk_pipeline_ref> all_pipelines;

    vk::Fence fence;
    vk_بفر sync_staging;

    ggml_backend_بفر_type بفر_type;

    bool disable_fusion;
    bool disable_descriptor_reuse;
    std::atomic<uint64_t> بفر_destroy_گنتی {};
    bool disable_host_visible_vidmem;
    bool allow_sysmem_fallback;
    bool disable_گراف_optimize;

    std::unique_ptr<vk_حافظہ_logger> حافظہ_logger;

    ~vk_device_struct();

};

مضمن فراغ vk_command_pool::init(vk_device& device, vk_قطار *q_) {
    cmd_buffers.clear();
    q = q_;

    vk::CommandPoolCreateInfo command_pool_create_info(
        vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_ثنائی_ہندسہ | VK_COMMAND_POOL_CREATE_RESET_COMMAND_بفر_ثنائی_ہندسہ),
        q->قطار_family_اشاریہ);
    pool = device->device.createCommandPool(command_pool_create_info);
}

مضمن فراغ vk_command_pool::destroy(vk::Device& device) {
    device.destroyCommandPool(pool);
    pool = nullptr;
    cmd_buffers.clear();
}

بنية vk_بفر_struct {
    vk::بفر_2 بفر = VK_NULL_HANDLE;
    vk::Device_حافظہ device_حافظہ = VK_NULL_HANDLE;
    vk::حافظہ_Property_Flags حافظہ_property_flags;
    فراغ * ptr;
    size_t ناپ = 0;
    vk::Device_پتہ bda_addr {};

    vk_device device;

    ~vk_بفر_struct() {
        إذا (ناپ == 0) {
            أرجع;
        }
        VK_LOG_DEBUG("~vk_buffer_struct(" << بفر << ", " << ناپ << ")");

        // bump before destroying, so a thread that sees the buffer gone also sees the new count
        device->بفر_destroy_گنتی.fetch_جوڑو(1, std::memory_order_release);
        device->device.free_حافظہ_2(device_حافظہ);
        device->device.destroy_بفر(بفر);
    }
};

بنية vk_subbuffer {
    vk_بفر بفر;
    uint64_t وتھ;
    uint64_t ناپ;

    عامل vk::Descriptor_بفر_Info() ثابت {
        أرجع { بفر->بفر, وتھ, ناپ };
    }
};

بنية vk_semaphore {
    vk::Semaphore s;
    uint64_t قدر;
};

بنية vk_event {
    std::سمتیہ<vk::Event> events_free; // Events available for reuse
    std::سمتیہ<vk::Event> events_submitted; // Events that are fully submitted and can be reused on next synchronize
    vk::Event event;
    bool has_event;

    vk_semaphore tl_semaphore;
    vk_command_بفر* cmd_بفر = nullptr;
    uint64_t cmd_بفر_use_counter = 0;
};

بنية vk_submission {
    vk_command_بفر* بفر = nullptr;
    std::سمتیہ<vk_semaphore> wait_semaphores;
    std::سمتیہ<vk_semaphore> signal_semaphores;
};

تعريفنوع std::سمتیہ<vk_submission> vk_sequence;

#تعریف_کرو MAT_VEC_FUSION_FLAGS_BIAS0 0x1

#تعریف_کرو MAT_VEC_FUSION_FLAGS_BIAS1 0x2

#تعریف_کرو MAT_VEC_FUSION_FLAGS_SCALE0 0x4

#تعریف_کرو MAT_VEC_FUSION_FLAGS_SCALE1 0x8

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
    std::سمتیہ<vk_sequence> seqs;

    صحيح exit_tensor_idx;

    std::سمتیہ<vk_staging_memcpy> وچ_memcpys;
    std::سمتیہ<vk_staging_memcpy> باہرلا_memcpys;
    std::سمتیہ<vk_staging_memset> memsets;

    std::سمتیہ<std::string> debug_labels;

    vk_command_pool * p {};
};

تعريفنوع std::shared_ptr<vk_context_struct> vk_context;

تعريفنوع std::weak_ptr<vk_context_struct> vk_context_ref;

بنية ggml_vk_garbage_collector {
    std::سمتیہ<vk_semaphore> tl_semaphores;
    std::سمتیہ<vk_semaphore> semaphores;
    std::سمتیہ<vk::Event> events;
    std::سمتیہ<vk_context> contexts;
};

#تعریف_کرو VK_LOG_حافظہ(پیغام) إذا (vk_حافظہ_logger_enabled) { std::cerr << "ggml_vulkan memory: " << پیغام << std::endl; }

ساكن std::string format_ناپ(size_t ناپ) {
    ثابت size_t kib = 1024;
    ثابت size_t mib = kib * 1024;
    ثابت size_t gib = mib * 1024;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    إذا (ناپ >= gib) {
        oss << static_cast<مزدوج>(ناپ) / gib << " GiB";
    } وإلا إذا (ناپ >= mib) {
        oss << static_cast<مزدوج>(ناپ) / mib << " MiB";
    } وإلا إذا (ناپ >= kib) {
        oss << static_cast<مزدوج>(ناپ) / kib << " KiB";
    } وإلا {
        oss << ناپ << " B";
    }

    أرجع oss.str();
}

class vk_حافظہ_logger {
عام:
    vk_حافظہ_logger(): کل_device(0), کل_host(0) {}
    فراغ log_ونڈ(vk_بفر_ref buf_ref, size_t ناپ);
    فراغ log_deallocation(vk_بفر_ref buf_ref);

خاص:
    std::map<vk::بفر_2, size_t> allocations; // Track allocations
    size_t کل_device;
    size_t کل_host;
    ساكن std::mutex log_mutex;
};

مضمن std::mutex vk_حافظہ_logger::log_mutex;

class vk_perf_logger {
  عام:
    فراغ print_timings(bool force = false);


    std::string get_گرہ_fusion_name(ثابت ggml_tensor * گرہ, ثابت حرف *fusion_name, uint64_t *n_flops);


    فراغ log_timing(ثابت ggml_tensor * گرہ, ثابت حرف *fusion_name, uint64_t ویلا) {
        uint64_t n_flops;
        std::string name = get_گرہ_fusion_name(گرہ, fusion_name, &n_flops);
        إذا (n_flops) {
            flops[name].ڈالو_پٹھ(n_flops);
        }
        timings[name].ڈالو_پٹھ(ویلا);
    }

    فراغ log_timing(ثابت std::سمتیہ<ggml_tensor *> &nodes, ثابت std::سمتیہ<ثابت حرف *> &ناں, uint64_t ویلا) {
        uint64_t کل_flops = 0;
        std::string name;
        لكل (size_t n = 0; n < nodes.ناپ(); ++n) {
            uint64_t n_flops = 0;
            name += get_گرہ_fusion_name(nodes[n], ناں[n], &n_flops);
            کل_flops += n_flops;

            إذا (n != nodes.ناپ() - 1) {
                name += ", ";
            }
        }
        إذا (کل_flops) {
            flops[name].ڈالو_پٹھ(کل_flops);
        }
        timings[name].ڈالو_پٹھ(ویلا);
    }

  خاص:
    std::map<std::string, std::سمتیہ<uint64_t>> timings;
    std::map<std::string, std::سمتیہ<uint64_t>> flops;
    uint32_t print_گنتی {};
};

بنية ggml_backend_vk_context {
    std::string name;

    vk_device device;

    size_t semaphore_idx, event_idx;
    ggml_vk_garbage_collector gc;
    size_t prealloc_ناپ_x, prealloc_ناپ_y, prealloc_ناپ_ونڈو_k, prealloc_ناپ_جوڑو_rms_partials, prealloc_ناپ_جوڑو_rms_partials_وتھ;
    vk_بفر prealloc_x, prealloc_y, prealloc_ونڈو_k, prealloc_جوڑو_rms_partials, sync_staging;
    vk::Fence fence, almost_ready_fence;
    bool submit_pending {};
    bool almost_ready_fence_pending {};
    // Set before op_add and unset after op_rms_norm to indicate that the add should
    // write partial sums to accumulate the square of the vector components
    bool do_جوڑو_rms_partials_وتھ_calculation;
    bool do_جوڑو_rms_partials;

    uint64_t آخری_کل_flops {UINT64_MAX};

    // Cache most recent tensor that was converted into prealloc_y, and what pipeline it used to convert.
    vk_pipeline_struct * prealloc_y_آخری_pipeline_ورتیا {};
    ثابت ggml_tensor * prealloc_y_آخری_tensor_ورتیا {};
    // True when the K dimension in prealloc_y is padded.
    bool prealloc_y_آخری_k_padded {};

    // Track which nodes have been used since the last sync, and whether they were written to
    std::سمتیہ<ثابت ggml_tensor *> unsynced_nodes_written;
    std::سمتیہ<ثابت ggml_tensor *> unsynced_nodes_read;
    // Track which prealloc buffers have pending reads that need to be synchronized.
    // These are checked before writing to the buffer (and call ggml_vk_sync_buffers if set),
    // and set to true after the buffer contents are consumed.
    bool prealloc_x_need_sync, prealloc_y_need_sync, prealloc_ونڈو_k_need_sync;

    vk_context_ref compute_ctx;

    vk_context_ref transfer_ctx;
    vk_semaphore transfer_semaphore;
    uint64_t transfer_semaphore_آخری_submitted {};

    std::سمتیہ<vk_context_ref> tensor_ctxs;

    std::سمتیہ<vk::DescriptorPool> descriptor_pools;
    std::سمتیہ<vk::Descriptor_سیٹ> descriptor_sets;
    // last bindings written to each set; descriptor_sets is append-only so an index always names the same set
    std::سمتیہ<std::سمتیہ<vk::Descriptor_بفر_Info>> descriptor_سیٹ_bindings;
    uint64_t descriptor_سیٹ_bindings_destroy_گنتی {};
    uint32_t descriptor_سیٹ_idx {};
    uint32_t pipeline_descriptor_سیٹ_requirements {};

    vk_command_pool compute_cmd_pool;
    vk_command_pool transfer_cmd_pool;

    // number of additional consecutive nodes that are being fused with the
    // node currently being processed
    صحيح عدد_additional_fused_ops {};
    // Bitmask of which fused ops need to write an intermediate value to memory.
    // Bit 'i' means nodes[start_of_fusion + i] writes to memory.
    // If there's no fusion, bit 0 is still set.
    صحيح fused_ops_write_نقاب {};
    topk_moe_منوال fused_topk_moe_منوال {};
    bool fused_topk_moe_scale {};
    // QSA indexer gather+add+top_k fused into one radix-select
    bool fused_topk_qsa {};
    bool fused_hc_post_gate {};
    rms_norm_منوال fused_rms_norm_منوال {RMS_NORM_گنتی};

    // for GGML_VK_PERF_LOGGER
    std::unique_ptr<vk_perf_logger> perf_logger;
    vk::QueryPool query_pool;
    std::سمتیہ<ثابت حرف *> query_fusion_ناں;
    std::سمتیہ<صحيح> query_fusion_گرہ_گنتی;
    std::سمتیہ<ggml_tensor *> query_nodes;
    std::سمتیہ<صحيح> query_گرہ_idx;
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

بنية vk_host_ونڈ {
    فراغ *    ادھار = nullptr;
    size_t    ناپ = 0;
    vk_بفر views[GGML_VK_MAX_DEVICES];  // indexed by device->idx; null = not (yet) mapped here
};

بنية vk_instance_t {
    vk::Instance instance;

    bool debug_utils_support = false;  // VK_EXT_debug_utils enabled
    PFN_vk_سیٹ_Debug_Utils_Object_Name_EXT pfn_vk_سیٹ_Debug_Utils_Object_Name_EXT = {};
    PFN_vk_قطار_Begin_Debug_Utils_Label_EXT pfn_vk_قطار_Begin_Debug_Utils_Label_EXT = {};
    PFN_vk_قطار_سرا_Debug_Utils_Label_EXT   pfn_vk_قطار_سرا_Debug_Utils_Label_EXT   = {};
    PFN_vkCmdBeginDebugUtilsLabelEXT   pfn_vkCmdBeginDebugUtilsLabelEXT   = {};
    PFN_vk_Cmd_سرا_Debug_Utils_Label_EXT pfn_vk_Cmd_سرا_Debug_Utils_Label_EXT = {};
    PFN_vk_Cmd_داخل_کرو_Debug_Utils_Label_EXT  pfn_vk_Cmd_داخل_کرو_Debug_Utils_Label_EXT  = {};

    std::سمتیہ<size_t> device_indices;
    std::سمتیہ<bool>   device_supports_membudget;
    std::سمتیہ<bool>   device_host_بفر_import;
    vk_device devices[GGML_VK_MAX_DEVICES];

    std::سمتیہ<vk_host_ونڈ> host_allocations;
    std::shared_mutex               host_allocations_mutex;

    size_t host_import_alignment = 0;
    uint32_t host_import_failed = 0;  // bitmask by device->idx, guarded by host_allocations_mutex

    bool multi_device = false;
};

تعريفنوع فراغ (*ggml_vk_func_t)(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);

ساكن constexpr uint32_t k_Spv_Op_Cooperative_میٹرکس_لدو_Tensor_NV = 5367;

ساكن constexpr uint32_t k_Spv_Capability_Cooperative_میٹرکس_Decode_سمتیہ_NV = 5447;

ساكن constexpr uint32_t k_Spv_Tensor_Addressing_Decode_سمتیہ_Func_ثنائی_ہندسہ = 0x4;

بنية vk_fa_tuning_params {
    Fa_کوڈ_رستہ رستہ;
    uint32_t workgroup_ناپ;
    uint32_t subgroup_ناپ;
    uint32_t بلاک_صفاں;
    uint32_t بلاک_خانے_گنتی;
    uint32_t d_ونڈو;
    uint32_t صف_ونڈو;
    bool shmem_staging;
    bool disable_subgroups;
    uint32_t حد_بندی_occupancy_shmem;

    فراغ print() ثابت {
        std::cerr << "path=" << رستہ << " workgroup_size=" << workgroup_ناپ << " subgroup_size=" << subgroup_ناپ <<
                     " block_rows=" << بلاک_صفاں << " block_cols=" << بلاک_خانے_گنتی << " d_split=" << d_ونڈو <<
                     " row_split=" << صف_ونڈو << " shmem_staging=" << shmem_staging << " disable_subgroups=" << disable_subgroups <<
                     " limit_occupancy_shmem=" << حد_بندی_occupancy_shmem << std::endl;
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
    uint32_t default_subgroup_ناپ = 0;
};

ساكن constexpr uint32_t RDNA_DEFAULT_SUBGROUP_ناپ = 32;

بنية CompileTask {
    vk_pipeline pipeline;
    size_t spv_ناپ;
    ثابت فراغ * spv_انکڑے;
    std::string entrypoint;
    uint32_t parameter_گنتی;
    std::array<uint32_t, 3> wg_denoms;
    std::سمتیہ<uint32_t> specialization_constants;
    bool disable_robustness;
    bool require_بھریا_subgroups;
    uint32_t required_subgroup_ناپ;
};

بنية ggml_vk_debug_label {
    // at most one of these is set, depending on the scope the label was opened in
    vk_context_struct * subctx {};
    vk_قطار_handle *   qhandle {};

    // one region per dispatch, e.g. "matmul_q4_k_f32_f16acc_aligned_m (192,8,1)".
    // RGP cannot recover the pipeline name on its own, it only has the hash
    ggml_vk_debug_label(vk_context & ctx, ثابت std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2);


    // one region per graph node
    // fused nodes are joined with '+', e.g. "RMS_NORM+MUL+ROPE Qcur-19"
    ggml_vk_debug_label(vk_context & ctx, ثابت ggml_cgraph * cgraph, صحيح گرہ_idx, صحيح n_fused);


    // one region per graph evaluation, opened on the queue instead of a command buffer
    // so it spans every submit the evaluation makes
    ggml_vk_debug_label(vk_قطار_handle * handle, ثابت حرف * name);


    // call before the command buffer can end, the destructor covers the rest
    فراغ close();


    ~ggml_vk_debug_label() {
        close();
    }

    ggml_vk_debug_label(ثابت ggml_vk_debug_label &) = delete;
    ggml_vk_debug_label & عامل=(ثابت ggml_vk_debug_label &) = delete;

خاص:
    // the constructors check this too, so the name is not built when markers are off
    فراغ begin(vk_context & ctx, ثابت std::string & name);

};

#تعریف_کرو بن_ورتیا GGML_بن_ورتیا

بنية ggml_backend_vk_device_context {
    size_t device;
    std::string name;
    std::string description;
    bool is_integrated_gpu;
    std::string pci_bus_id;
    صحيح op_offload_min_batch_ناپ;
};
