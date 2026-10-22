#اگر_تعریف_نشده GGML_WEBGPU_SHADER_LIB_HPP
#تعریف_کن GGML_WEBGPU_SHADER_LIB_HPP

#ضمن "ggml-impl.h"
#ضمن "ggml-wgsl-shaders.hpp"
#ضمن "ggml.h"
#ضمن "pre_wgsl.hpp"

#ضمن <webgpu/webgpu_cpp.h>

#ضمن <algorithm>
#ضمن <memory>
#ضمن <string>
#ضمن <unordered_map>
#ضمن <vector>

#تعریف_کن GGML_WEBGPU_F16_اندازه_بایتها                   2
#تعریف_کن GGML_WEBGPU_F32_اندازه_بایتها                   4
#تعریف_کن GGML_WEBGPU_I32_اندازه_بایتها                   4
#تعریف_کن GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#تعریف_کن GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#تعریف_کن GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#تعریف_کن GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#تعریف_کن GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_اندازه     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#تعریف_کن GGML_WEBGPU_KV_SEQ_PAD                       256u

#تعریف_کن GGML_WEBGPU_ARGSORT_ادغام_MAX_WG_اندازه 512u

// Matrix multiplication parameters

// Register tiling parameters
#تعریف_کن WEBGPU_MUL_MAT_TILE_M           4
#تعریف_کن WEBGPU_MUL_MAT_TILE_N           4
#تعریف_کن WEBGPU_MUL_MAT_WG_اندازه_M        8
#تعریف_کن WEBGPU_MUL_MAT_WG_اندازه_N        8
#تعریف_کن WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#تعریف_کن WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#تعریف_کن WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#تعریف_کن WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#تعریف_کن WEBGPU_MUL_MAT_SUBGROUP_ماتریس_M     4
#تعریف_کن WEBGPU_MUL_MAT_SUBGROUP_ماتریس_N     2
#تعریف_کن WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#تعریف_کن WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#تعریف_کن WEBGPU_MUL_MAT_VEC_WG_اندازه 256

#تعریف_کن WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_در_هر_WG    4
#تعریف_کن WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_در_هر_WG 4
#تعریف_کن WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_در_هر_WG      4

// default size for reg-tile matrix multiplication
#تعریف_کن WEBGPU_MUL_MAT_WG_اندازه 256

// Same hash combine function as in boost
قالب <نام_نوع T> مضمن فراغ ggml_webgpu_hash_combine(size_t & دانه, ثابت T & value) {
    دانه ^= std::hash<T>{}(value) + 0x9e3779b9 + (دانه << 6) + (دانه >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
مضمن uintptr_t ggml_webgpu_tensor_addr(ثابت ggml_tensor * tensor) {
    ثابت ggml_tensor * پایه_tensor = tensor->view_src ? tensor->view_src : tensor;
    أرجع (uintptr_t) پایه_tensor->داده + tensor->view_offs;
}

مضمن منطقي ggml_webgpu_tensor_equal(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    أرجع a->میانگیر == b->میانگیر && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

بنية ggml_webgpu_shader_lib_context {
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * src2;
    ggml_tensor * src3;
    ggml_tensor * src4;
    ggml_tensor * src5;
    ggml_tensor * dst;
    ggml_tensor * dst_fuse;

    uint32_t    max_wg_اندازه;
    size_t      wg_mem_حد_بایتها       = 0;
    منطقي        supports_subgroups       = خطأ;
    منطقي        supports_subgroup_ماتریس = خطأ;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_اندازه        = 0;
    uint32_t    max_subgroup_اندازه        = 0;
    منطقي        supports_dot_حاصلضرب     = خطأ;
    std::string vendor;
};

بنية webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<فراغ> context = اشاره_تهی;
};

بنية ggml_webgpu_generic_shader_decisions {
    uint32_t wg_اندازه = 0;
    منطقي     inplace = خطأ;
};

بنية ggml_webgpu_get_ردیفها_shader_decisions {
    uint32_t wg_اندازه    = 0;
    منطقي     vectorized = خطأ;
};

بنية ggml_webgpu_دودویی_shader_decisions {
    uint32_t wg_اندازه     = 0;
    منطقي     inplace     = خطأ;
    منطقي     overlap     = خطأ;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_glu_shader_decisions {
    uint32_t wg_اندازه     = 0;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<فراغ> decisions;
};

بنية ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t بلوک_اندازه_3;
    uint32_t نشانهها_در_هر_wg;
};

بنية ggml_webgpu_ssm_پیمایش_pipeline_کلید {
    صحيح     type;
    صحيح     d_state;
    uint8_t xbc_overlap;
    منطقي    a_overlap;
    منطقي    ids_overlap;

    منطقي عملگر==(ثابت ggml_webgpu_ssm_پیمایش_pipeline_کلید & other) ثابت {
        أرجع type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

بنية ggml_webgpu_ssm_پیمایش_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_ssm_پیمایش_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.type);
        ggml_webgpu_hash_combine(دانه, کلید.d_state);
        ggml_webgpu_hash_combine(دانه, کلید.xbc_overlap);
        ggml_webgpu_hash_combine(دانه, کلید.a_overlap);
        ggml_webgpu_hash_combine(دانه, کلید.ids_overlap);
        أرجع دانه;
    }
};

بنية ggml_webgpu_ssm_پیمایش_shader_decisions {
    uint32_t wg_اندازه;
    uint32_t نشانهها_در_هر_tile;
    uint8_t  xbc_overlap = 0;
    منطقي     a_overlap   = خطأ;
    منطقي     ids_overlap = خطأ;
};

/** Argsort **/

بنية ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_اندازه;
    size_t   wg_mem_حد_بایتها;
    int32_t  ترتیب;
};

/** Set Rows **/

بنية ggml_webgpu_مجموعه_ردیفها_pipeline_کلید {
    صحيح src0_type;
    صحيح dst_type;
    صحيح vec4;
    صحيح i64_idx;
    صحيح pair_بلوکها;

    منطقي عملگر==(ثابت ggml_webgpu_مجموعه_ردیفها_pipeline_کلید & other) ثابت {
        أرجع src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_بلوکها == other.pair_بلوکها;
    }
};

بنية ggml_webgpu_مجموعه_ردیفها_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_مجموعه_ردیفها_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.src0_type);
        ggml_webgpu_hash_combine(دانه, کلید.dst_type);
        ggml_webgpu_hash_combine(دانه, کلید.vec4);
        ggml_webgpu_hash_combine(دانه, کلید.i64_idx);
        ggml_webgpu_hash_combine(دانه, کلید.pair_بلوکها);
        أرجع دانه;
    }
};

بنية ggml_webgpu_مجموعه_ردیفها_shader_decisions {
    منطقي     vec4;
    منطقي     i64_idx;
    منطقي     pair_بلوکها;
    uint32_t wg_اندازه;
};

/** Set **/

بنية ggml_webgpu_مجموعه_pipeline_کلید {
    ggml_type type;
    منطقي      inplace;

    منطقي عملگر==(ثابت ggml_webgpu_مجموعه_pipeline_کلید & other) ثابت {
        أرجع type == other.type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_مجموعه_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_مجموعه_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.type);
        ggml_webgpu_hash_combine(دانه, کلید.inplace);
        أرجع دانه;
    }
};

/** Get Rows **/

بنية ggml_webgpu_get_ردیفها_pipeline_کلید {
    ggml_type src_type;
    صحيح       vectorized;

    منطقي عملگر==(ثابت ggml_webgpu_get_ردیفها_pipeline_کلید & other) ثابت {
        أرجع src_type == other.src_type && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_get_ردیفها_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_get_ردیفها_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.src_type);
        ggml_webgpu_hash_combine(دانه, کلید.vectorized);
        أرجع دانه;
    }
};

/** Row Norm **/

بنية ggml_webgpu_ردیف_norm_pipeline_کلید {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    منطقي      inplace;

    منطقي عملگر==(ثابت ggml_webgpu_ردیف_norm_pipeline_کلید & other) ثابت {
        أرجع op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_ردیف_norm_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_ردیف_norm_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.op);
        ggml_webgpu_hash_combine(دانه, کلید.src_type);
        ggml_webgpu_hash_combine(دانه, کلید.dst_type);
        ggml_webgpu_hash_combine(دانه, کلید.inplace);
        أرجع دانه;
    }
};

/** RMS_NORM + MUL **/

بنية ggml_webgpu_rms_norm_mul_pipeline_کلید {
    منطقي inplace;      // rn_src == dst
    منطقي overlap;      // mul_src == dst
    منطقي src_overlap;  // rn_src binding overlaps mul_src binding

    منطقي عملگر==(ثابت ggml_webgpu_rms_norm_mul_pipeline_کلید & other) ثابت {
        أرجع inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_rms_norm_mul_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_rms_norm_mul_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.inplace);
        ggml_webgpu_hash_combine(دانه, کلید.overlap);
        ggml_webgpu_hash_combine(دانه, کلید.src_overlap);
        أرجع دانه;
    }
};

بنية ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_اندازه     = 0;
    منطقي     inplace     = خطأ;
    منطقي     overlap     = خطأ;
    منطقي     src_overlap = خطأ;
};

/** Pad **/
بنية ggml_webgpu_pad_pipeline_کلید {
    منطقي circular;

    منطقي عملگر==(ثابت ggml_webgpu_pad_pipeline_کلید & other) ثابت { أرجع circular == other.circular; }
};

بنية ggml_webgpu_pad_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_pad_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.circular);
        أرجع دانه;
    }
};

/** Solve Tri **/
بنية ggml_webgpu_solve_tri_pipeline_کلید {
    صحيح type;
    صحيح n;
    صحيح k;

    منطقي عملگر==(ثابت ggml_webgpu_solve_tri_pipeline_کلید & other) ثابت {
        أرجع type == other.type && n == other.n && k == other.k;
    }
};

بنية ggml_webgpu_solve_tri_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_solve_tri_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.type);
        ggml_webgpu_hash_combine(دانه, کلید.n);
        ggml_webgpu_hash_combine(دانه, کلید.k);
        أرجع دانه;
    }
};

/** SSM Conv **/
بنية ggml_webgpu_ssm_conv_pipeline_کلید {
    صحيح type;
    صحيح vectorized;

    منطقي عملگر==(ثابت ggml_webgpu_ssm_conv_pipeline_کلید & other) ثابت {
        أرجع type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
بنية ggml_webgpu_conv2d_pipeline_کلید {
    ggml_type وزن_type;
    ggml_type درون_داد_type;
    ggml_type output_type;

    منطقي عملگر==(ثابت ggml_webgpu_conv2d_pipeline_کلید & other) ثابت {
        أرجع وزن_type == other.وزن_type && درون_داد_type == other.درون_داد_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_conv2d_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_conv2d_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.وزن_type);
        ggml_webgpu_hash_combine(دانه, کلید.درون_داد_type);
        ggml_webgpu_hash_combine(دانه, کلید.output_type);
        أرجع دانه;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
بنية ggml_webgpu_conv2d_dw_pipeline_کلید {
    ggml_type وزن_type;
    ggml_type درون_داد_type;
    ggml_type output_type;
    منطقي      whcn;

    منطقي عملگر==(ثابت ggml_webgpu_conv2d_dw_pipeline_کلید & other) ثابت {
        أرجع وزن_type == other.وزن_type && درون_داد_type == other.درون_داد_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

بنية ggml_webgpu_conv2d_dw_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_conv2d_dw_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.وزن_type);
        ggml_webgpu_hash_combine(دانه, کلید.درون_داد_type);
        ggml_webgpu_hash_combine(دانه, کلید.output_type);
        ggml_webgpu_hash_combine(دانه, کلید.whcn);
        أرجع دانه;
    }
};

/** Im2Col **/
بنية ggml_webgpu_im2col_pipeline_کلید {
    ggml_type درون_داد_type;
    ggml_type output_type;

    منطقي عملگر==(ثابت ggml_webgpu_im2col_pipeline_کلید & other) ثابت {
        أرجع درون_داد_type == other.درون_داد_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_im2col_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_im2col_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.درون_داد_type);
        ggml_webgpu_hash_combine(دانه, کلید.output_type);
        أرجع دانه;
    }
};

/** Gated Delta Net **/
بنية ggml_webgpu_gated_delta_net_pipeline_کلید {
    صحيح  type;
    صحيح  s_v;
    صحيح  kda;
    منطقي fused_cache;

    منطقي عملگر==(ثابت ggml_webgpu_gated_delta_net_pipeline_کلید & other) ثابت {
        أرجع type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

بنية ggml_webgpu_gated_delta_net_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_gated_delta_net_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.type);
        ggml_webgpu_hash_combine(دانه, کلید.s_v);
        ggml_webgpu_hash_combine(دانه, کلید.kda);
        أرجع دانه;
    }
};

بنية ggml_webgpu_ssm_conv_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_ssm_conv_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.type);
        ggml_webgpu_hash_combine(دانه, کلید.vectorized);
        أرجع دانه;
    }
};

/** Scale **/

بنية ggml_webgpu_scale_pipeline_کلید {
    صحيح inplace;

    منطقي عملگر==(ثابت ggml_webgpu_scale_pipeline_کلید & other) ثابت { أرجع inplace == other.inplace; }
};

بنية ggml_webgpu_scale_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_scale_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.inplace);
        أرجع دانه;
    }
};

/** Upscale **/

بنية ggml_webgpu_upscale_pipeline_کلید {
    ggml_type درون_داد_type;
    ggml_type output_type;
    uint32_t  پایه_نما;
    منطقي      antialias;

    منطقي عملگر==(ثابت ggml_webgpu_upscale_pipeline_کلید & other) ثابت {
        أرجع درون_داد_type == other.درون_داد_type && output_type == other.output_type && پایه_نما == other.پایه_نما &&
               antialias == other.antialias;
    }
};

بنية ggml_webgpu_upscale_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_upscale_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.درون_داد_type);
        ggml_webgpu_hash_combine(دانه, کلید.output_type);
        ggml_webgpu_hash_combine(دانه, کلید.پایه_نما);
        ggml_webgpu_hash_combine(دانه, کلید.antialias);
        أرجع دانه;
    }
};

/** Concat **/

بنية ggml_webgpu_concat_pipeline_کلید {
    صحيح  type;
    منطقي src_overlap;

    منطقي عملگر==(ثابت ggml_webgpu_concat_pipeline_کلید & other) ثابت {
        أرجع type == other.type && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_concat_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_concat_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.type);
        ggml_webgpu_hash_combine(دانه, کلید.src_overlap);
        أرجع دانه;
    }
};

/** Repeat **/

بنية ggml_webgpu_repeat_pipeline_کلید {
    صحيح type;

    منطقي عملگر==(ثابت ggml_webgpu_repeat_pipeline_کلید & other) ثابت { أرجع type == other.type; }
};

بنية ggml_webgpu_repeat_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_repeat_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.type);
        أرجع دانه;
    }
};

/** Binary **/

بنية ggml_webgpu_دودویی_pipeline_کلید {
    صحيح  type;
    صحيح  op;
    منطقي inplace;
    منطقي overlap;
    منطقي src_overlap;

    منطقي عملگر==(ثابت ggml_webgpu_دودویی_pipeline_کلید & other) ثابت {
        أرجع type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_دودویی_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_دودویی_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.type);
        ggml_webgpu_hash_combine(دانه, کلید.op);
        ggml_webgpu_hash_combine(دانه, کلید.inplace);
        ggml_webgpu_hash_combine(دانه, کلید.overlap);
        ggml_webgpu_hash_combine(دانه, کلید.src_overlap);
        أرجع دانه;
    }
};

/* Add_Id */

بنية ggml_webgpu_جمع_کن_id_pipeline_کلید {
    منطقي inplace;

    منطقي عملگر==(ثابت ggml_webgpu_جمع_کن_id_pipeline_کلید & other) ثابت { أرجع inplace == other.inplace; }
};

بنية ggml_webgpu_جمع_کن_id_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_جمع_کن_id_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.inplace);
        أرجع دانه;
    }
};

/** Unary **/

بنية ggml_webgpu_unary_pipeline_کلید {
    صحيح           type;
    صحيح           op;
    منطقي          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    منطقي          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    منطقي عملگر==(ثابت ggml_webgpu_unary_pipeline_کلید & other) ثابت {
        أرجع type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

بنية ggml_webgpu_unary_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_unary_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.type);
        ggml_webgpu_hash_combine(دانه, کلید.op);
        ggml_webgpu_hash_combine(دانه, کلید.is_unary);
        ggml_webgpu_hash_combine(دانه, کلید.inplace);
        ggml_webgpu_hash_combine(دانه, کلید.ttype);
        أرجع دانه;
    }
};

/** FlashAttention */

بنية ggml_webgpu_flash_attn_common_pipeline_کلید {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  سر_dim_qk;
    uint32_t  سر_dim_v;
    منطقي      k_مستقیم;
    منطقي      v_مستقیم;
    منطقي      kv_overlap;
    منطقي      has_ماسک;
    منطقي      has_sinks;
    منطقي      استفاده_میکند_logit_softcap;

    منطقي عملگر==(ثابت ggml_webgpu_flash_attn_common_pipeline_کلید & other) ثابت {
        أرجع q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && سر_dim_qk == other.سر_dim_qk && سر_dim_v == other.سر_dim_v &&
               k_مستقیم == other.k_مستقیم && v_مستقیم == other.v_مستقیم && kv_overlap == other.kv_overlap &&
               has_ماسک == other.has_ماسک && has_sinks == other.has_sinks &&
               استفاده_میکند_logit_softcap == other.استفاده_میکند_logit_softcap;
    }
};

مضمن فراغ ggml_webgpu_flash_attn_hash_common_pipeline_کلید(size_t &                                           دانه,
                                                            ثابت ggml_webgpu_flash_attn_common_pipeline_کلید & کلید) {
    ggml_webgpu_hash_combine(دانه, کلید.q_type);
    ggml_webgpu_hash_combine(دانه, کلید.k_type);
    ggml_webgpu_hash_combine(دانه, کلید.v_type);
    ggml_webgpu_hash_combine(دانه, کلید.dst_type);
    ggml_webgpu_hash_combine(دانه, کلید.سر_dim_qk);
    ggml_webgpu_hash_combine(دانه, کلید.سر_dim_v);
    ggml_webgpu_hash_combine(دانه, کلید.k_مستقیم);
    ggml_webgpu_hash_combine(دانه, کلید.v_مستقیم);
    ggml_webgpu_hash_combine(دانه, کلید.kv_overlap);
    ggml_webgpu_hash_combine(دانه, کلید.has_ماسک);
    ggml_webgpu_hash_combine(دانه, کلید.has_sinks);
    ggml_webgpu_hash_combine(دانه, کلید.استفاده_میکند_logit_softcap);
}

بنية ggml_webgpu_flash_attn_vec_pipeline_کلید {
    ggml_webgpu_flash_attn_common_pipeline_کلید common;

    منطقي عملگر==(ثابت ggml_webgpu_flash_attn_vec_pipeline_کلید & other) ثابت { أرجع common == other.common; }
};

بنية ggml_webgpu_flash_attn_vec_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_flash_attn_vec_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_کلید(دانه, کلید.common);
        أرجع دانه;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_کلید {
    ggml_webgpu_flash_attn_common_pipeline_کلید common;
    منطقي                                       use_sg_ماتریس;

    منطقي عملگر==(ثابت ggml_webgpu_flash_attn_pipeline_کلید & other) ثابت {
        أرجع common == other.common && use_sg_ماتریس == other.use_sg_ماتریس;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_flash_attn_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_کلید(دانه, کلید.common);
        ggml_webgpu_hash_combine(دانه, کلید.use_sg_ماتریس);
        أرجع دانه;
    }
};

بنية ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_اندازه = 0;
};

بنية ggml_webgpu_flash_attn_decisions {
    منطقي     use_sg_ماتریس = خطأ;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_اندازه       = 0;
};

مضمن constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_عرض = 4u;
مضمن constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

مضمن size_t ggml_webgpu_flash_attn_tensor_جابه_جایی(ثابت ggml_tensor * tensor) {
    constexpr uintptr_t ptr_پایه_addr = 0x1000u;
    ثابت ggml_tensor * پایه          = tensor->view_src != اشاره_تهی ? tensor->view_src : tensor;
    أرجع reinterpret_cast<uintptr_t>(پایه->داده) - ptr_پایه_addr + tensor->view_offs;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K, size_t storage_جابه_جایی_alignment) {
    ثابت uint32_t جابه_جایی_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_جابه_جایی(K) & (storage_جابه_جایی_alignment - 1)) /
                    ggml_type_اندازه(K->type));
    أرجع جابه_جایی_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_عرض == 0u;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K,
                                                      ثابت ggml_tensor * V,
                                                      size_t              storage_جابه_جایی_alignment) {
    أرجع ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_جابه_جایی_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_جابه_جایی_alignment);
}

مضمن منطقي ggml_webgpu_flash_attn_k_مستقیم(ثابت ggml_tensor * Q, ثابت ggml_tensor * K, uint32_t kv_مستقیم_align) {
    أرجع (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_مستقیم_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

مضمن منطقي ggml_webgpu_flash_attn_v_مستقیم(ثابت ggml_tensor * Q, ثابت ggml_tensor * V, uint32_t kv_مستقیم_align) {
    أرجع ggml_webgpu_flash_attn_k_مستقیم(Q, V, kv_مستقیم_align);
}

مضمن ggml_webgpu_flash_attn_common_pipeline_کلید ggml_webgpu_flash_attn_make_common_pipeline_کلید(
    ثابت ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_مستقیم_align,
    منطقي                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_کلید کلید = {};
    کلید.q_type                                     = context.src0->type;
    کلید.k_type                                     = context.src1->type;
    کلید.v_type                                     = context.src2->type;
    کلید.dst_type                                   = context.dst->type;
    کلید.سر_dim_qk                                = (uint32_t) context.src0->ne[0];
    کلید.سر_dim_v                                 = (uint32_t) context.src2->ne[0];
    کلید.k_مستقیم           = ggml_webgpu_flash_attn_k_مستقیم(context.src0, context.src1, kv_مستقیم_align);
    کلید.v_مستقیم           = ggml_webgpu_flash_attn_v_مستقیم(context.src0, context.src2, kv_مستقیم_align);
    کلید.kv_overlap         = kv_overlap;
    کلید.has_ماسک           = context.src3 != اشاره_تهی;
    کلید.has_sinks          = context.src4 != اشاره_تهی;
    کلید.استفاده_میکند_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    أرجع کلید;
}

مضمن std::vector<std::string> ggml_webgpu_flash_attn_common_تعریف_میکند(
    ثابت ggml_webgpu_flash_attn_common_pipeline_کلید & کلید,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_اندازه) {
    std::vector<std::string> تعریف_میکند;

    بدل (کلید.k_type) {
        حالة GGML_TYPE_F32:
            تعریف_میکند.push_back("K_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            تعریف_میکند.push_back("K_F16");
            اكسر;
        حالة GGML_TYPE_Q4_0:
            تعریف_میکند.push_back("K_Q4_0");
            اكسر;
        حالة GGML_TYPE_Q8_0:
            تعریف_میکند.push_back("K_Q8_0");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(کلید.k_type);

    بدل (کلید.v_type) {
        حالة GGML_TYPE_F32:
            تعریف_میکند.push_back("V_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            تعریف_میکند.push_back("V_F16");
            اكسر;
        حالة GGML_TYPE_Q4_0:
            تعریف_میکند.push_back("V_Q4_0");
            اكسر;
        حالة GGML_TYPE_Q8_0:
            تعریف_میکند.push_back("V_Q8_0");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(کلید.v_type);

    بدل (کلید.q_type) {
        حالة GGML_TYPE_F32:
            تعریف_میکند.push_back("Q_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            تعریف_میکند.push_back("Q_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(کلید.q_type);

    بدل (کلید.dst_type) {
        حالة GGML_TYPE_F32:
            تعریف_میکند.push_back("DST_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            تعریف_میکند.push_back("DST_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(کلید.dst_type);

    إذا (کلید.has_ماسک) {
        تعریف_میکند.push_back("MASK");
        variant += "_mask";
    }
    إذا (کلید.has_sinks) {
        تعریف_میکند.push_back("SINKS");
        variant += "_sinks";
    }
    إذا (کلید.استفاده_میکند_logit_softcap) {
        تعریف_میکند.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    إذا (کلید.k_مستقیم) {
        تعریف_میکند.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    إذا (کلید.v_مستقیم) {
        تعریف_میکند.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    إذا (کلید.kv_overlap) {
        تعریف_میکند.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    تعریف_میکند.push_back(std::string("HEAD_DIM_QK=") + std::to_string(کلید.سر_dim_qk));
    variant += std::string("_hsqk") + std::to_string(کلید.سر_dim_qk);

    تعریف_میکند.push_back(std::string("HEAD_DIM_V=") + std::to_string(کلید.سر_dim_v));
    variant += std::string("_hsv") + std::to_string(کلید.سر_dim_v);

    تعریف_میکند.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    تعریف_میکند.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));

    إذا (ggml_is_quantized(کلید.k_type) || ggml_is_quantized(کلید.v_type)) {
        تعریف_میکند.push_back("U32_DEQUANT_HELPERS");
        إذا (ggml_is_quantized(کلید.k_type)) {
            تعریف_میکند.push_back("LOADERS_QUANTIZED_K");
        }
        إذا (ggml_is_quantized(کلید.v_type)) {
            تعریف_میکند.push_back("LOADERS_QUANTIZED_V");
        }
    }

    أرجع تعریف_میکند;
}

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید {
    uint32_t  سر_dim_v;
    uint32_t  wg_اندازه;
    ggml_type dst_type;
};

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.سر_dim_v);
        ggml_webgpu_hash_combine(دانه, کلید.wg_اندازه);
        ggml_webgpu_hash_combine(دانه, کلید.dst_type);
        أرجع دانه;
    }
};

مضمن منطقي عملگر==(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید & lhs,
                       ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید & rhs) {
    أرجع lhs.سر_dim_v == rhs.سر_dim_v && lhs.wg_اندازه == rhs.wg_اندازه && lhs.dst_type == rhs.dst_type;
}

بنية ggml_webgpu_flash_attn_blk_pipeline_کلید {
    uint32_t kv_tile;

    منطقي عملگر==(ثابت ggml_webgpu_flash_attn_blk_pipeline_کلید & other) ثابت { أرجع kv_tile == other.kv_tile; }
};

بنية ggml_webgpu_flash_attn_blk_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_flash_attn_blk_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.kv_tile);
        أرجع دانه;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
مضمن size_t ggml_webgpu_flash_attn_wg_mem_بایتها(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t سر_dim_qk,
                                                  uint32_t سر_dim_v,
                                                  منطقي     has_ماسک,
                                                  منطقي     kv_مستقیم) {
    ثابت uint32_t max_سر_dim = std::max(سر_dim_qk, سر_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * سر_dim_qk;        // q_shmem
    إذا (!kv_مستقیم) {
        f32_elems += kv_tile * max_سر_dim;  // kv_shmem
    }
    f32_elems += q_tile * سر_dim_v;         // o_shmem
    إذا (has_ماسک) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    أرجع f16_elems * GGML_WEBGPU_F16_اندازه_بایتها + f32_elems * GGML_WEBGPU_F32_اندازه_بایتها;
}

مضمن uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   حد_بایتها,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t سر_dim_qk,
                                                   uint32_t سر_dim_v,
                                                   منطقي     has_ماسک,
                                                   منطقي     kv_مستقیم) {
    ثابت size_t پایه_q_بایتها =
        ggml_webgpu_flash_attn_wg_mem_بایتها(q_tile, 0, سر_dim_qk, سر_dim_v, has_ماسک, kv_مستقیم);
    إذا (حد_بایتها <= پایه_q_بایتها) {
        أرجع 0;
    }
    ثابت size_t one_kv_بایتها =
        ggml_webgpu_flash_attn_wg_mem_بایتها(q_tile, 1, سر_dim_qk, سر_dim_v, has_ماسک, kv_مستقیم);
    ثابت size_t بایتها_در_هر_kv = one_kv_بایتها - پایه_q_بایتها;
    إذا (بایتها_در_هر_kv == 0) {
        أرجع 0;
    }
    ثابت size_t max_kv_tile = (حد_بایتها - پایه_q_بایتها) / بایتها_در_هر_kv;
    أرجع (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

مضمن uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_حد_بایتها,
                                                       uint32_t سر_dim_qk,
                                                       uint32_t سر_dim_v,
                                                       منطقي     has_ماسک,
                                                       منطقي     kv_مستقیم) {
    ثابت uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_حد_بایتها, 1u, 1u, سر_dim_qk, سر_dim_v, has_ماسک, kv_مستقیم);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    إذا (kv_مستقیم) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        طالما (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    أرجع kv_tile;
}

مضمن منطقي ggml_webgpu_flash_attn_can_use_subgroup_ماتریس_مسیر(منطقي                supports_subgroup_ماتریس,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ثابت ggml_tensor * Q,
                                                                ثابت ggml_tensor * V) {
    أرجع supports_subgroup_ماتریس && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

بنية ggml_webgpu_mul_mat_vec_pipeline_کلید {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    uint32_t  رقم_تعداد_ستون;
    منطقي      use_mmvq;
    منطقي      src_overlap;

    منطقي عملگر==(ثابت ggml_webgpu_mul_mat_vec_pipeline_کلید & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               رقم_تعداد_ستون == other.رقم_تعداد_ستون && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_vec_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_mul_mat_vec_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.src0_type);
        ggml_webgpu_hash_combine(دانه, کلید.src1_type);
        ggml_webgpu_hash_combine(دانه, کلید.vectorized);
        ggml_webgpu_hash_combine(دانه, کلید.رقم_تعداد_ستون);
        ggml_webgpu_hash_combine(دانه, کلید.use_mmvq);
        ggml_webgpu_hash_combine(دانه, کلید.src_overlap);
        أرجع دانه;
    }
};

بنية ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_اندازه;
    uint32_t outputs_در_هر_wg;
    uint32_t vec_اندازه;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_quantize_q8_pipeline_کلید {
    ggml_type src0_type;

    منطقي عملگر==(ثابت ggml_webgpu_quantize_q8_pipeline_کلید & other) ثابت { أرجع src0_type == other.src0_type; }
};

بنية ggml_webgpu_quantize_q8_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_quantize_q8_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.src0_type);
        أرجع دانه;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_کلید {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    صحيح       use_subgroup_ماتریس;
    منطقي      src_overlap;

    منطقي عملگر==(ثابت ggml_webgpu_mul_mat_pipeline_کلید & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_ماتریس == other.use_subgroup_ماتریس && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_mul_mat_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.src0_type);
        ggml_webgpu_hash_combine(دانه, کلید.src1_type);
        ggml_webgpu_hash_combine(دانه, کلید.vectorized);
        ggml_webgpu_hash_combine(دانه, کلید.use_subgroup_ماتریس);
        ggml_webgpu_hash_combine(دانه, کلید.src_overlap);
        أرجع دانه;
    }
};

بنية ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_اندازه_m;
    uint32_t wg_اندازه_n;
    uint32_t wg_اندازه;
    uint32_t outputs_در_هر_wg;
    صحيح      use_subgroup_ماتریس;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_ماتریس_m;
    uint32_t subgroup_ماتریس_n;

    uint32_t mul_mat_wg_اندازه;
    منطقي     src_overlap = خطأ;
};

/** MUL_MAT_ID **/

بنية ggml_webgpu_mul_mat_id_gather_pipeline_کلید {
    uint32_t n_experts;

    منطقي عملگر==(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_کلید & other) ثابت {
        أرجع n_experts == other.n_experts;
    }
};

بنية ggml_webgpu_mul_mat_id_gather_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.n_experts);
        أرجع دانه;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_کلید {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  رقم_تعداد_ستون;
    صحيح       vectorized;

    منطقي عملگر==(ثابت ggml_webgpu_mul_mat_id_pipeline_کلید & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               رقم_تعداد_ستون == other.رقم_تعداد_ستون && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_mul_mat_id_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.src0_type);
        ggml_webgpu_hash_combine(دانه, کلید.src1_type);
        ggml_webgpu_hash_combine(دانه, کلید.n_experts);
        ggml_webgpu_hash_combine(دانه, کلید.رقم_تعداد_ستون);
        ggml_webgpu_hash_combine(دانه, کلید.vectorized);
        أرجع دانه;
    }
};

/** Cpy **/

بنية ggml_webgpu_cpy_pipeline_کلید {
    ggml_type src_type;
    ggml_type dst_type;

    منطقي عملگر==(ثابت ggml_webgpu_cpy_pipeline_کلید & other) ثابت {
        أرجع src_type == other.src_type && dst_type == other.dst_type;
    }
};

بنية ggml_webgpu_cpy_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_cpy_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.src_type);
        ggml_webgpu_hash_combine(دانه, کلید.dst_type);
        أرجع دانه;
    }
};

/** Glu **/

بنية ggml_webgpu_glu_pipeline_کلید {
    ggml_glu_op glu_op;
    ggml_type   type;
    منطقي        تقسیم;
    منطقي        src_overlap;

    منطقي عملگر==(ثابت ggml_webgpu_glu_pipeline_کلید & other) ثابت {
        أرجع glu_op == other.glu_op && type == other.type && تقسیم == other.تقسیم && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_glu_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_glu_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.glu_op);
        ggml_webgpu_hash_combine(دانه, کلید.type);
        ggml_webgpu_hash_combine(دانه, کلید.تقسیم);
        ggml_webgpu_hash_combine(دانه, کلید.src_overlap);
        أرجع دانه;
    }
};

/** Rope **/

بنية ggml_webgpu_rope_pipeline_کلید {
    ggml_type type;
    منطقي      inplace;
    منطقي      has_ff;

    منطقي عملگر==(ثابت ggml_webgpu_rope_pipeline_کلید & other) ثابت {
        أرجع type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

بنية ggml_webgpu_rope_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_rope_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.type);
        ggml_webgpu_hash_combine(دانه, کلید.inplace);
        ggml_webgpu_hash_combine(دانه, کلید.has_ff);
        أرجع دانه;
    }
};

/** SoftMax **/

بنية ggml_webgpu_soft_max_pipeline_کلید {
    ggml_type ماسک_type;
    منطقي      has_ماسک;
    منطقي      has_sink;
    منطقي      inplace;

    منطقي عملگر==(ثابت ggml_webgpu_soft_max_pipeline_کلید & other) ثابت {
        أرجع ماسک_type == other.ماسک_type && has_ماسک == other.has_ماسک && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

بنية ggml_webgpu_soft_max_pipeline_کلید_hash {
    size_t عملگر()(ثابت ggml_webgpu_soft_max_pipeline_کلید & کلید) ثابت {
        size_t دانه = 0;
        ggml_webgpu_hash_combine(دانه, کلید.ماسک_type);
        ggml_webgpu_hash_combine(دانه, کلید.has_ماسک);
        ggml_webgpu_hash_combine(دانه, کلید.has_sink);
        ggml_webgpu_hash_combine(دانه, کلید.inplace);
        أرجع دانه;
    }
};

/** MMVQ **/

مضمن منطقي ggml_webgpu_can_use_mmvq(ثابت ggml_tensor * src0,
                                     ثابت ggml_tensor * src1,
                                     منطقي                supports_dot_حاصلضرب,
                                     ثابت std::string & vendor) {
    إذا (src1->ne[1] <= 4) {
        منطقي supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        إذا (supports_dp4a && supports_dot_حاصلضرب) {
            بدل (src1->type) {
                حالة GGML_TYPE_F32:
                    بدل (src0->type) {
                        حالة GGML_TYPE_Q1_0:
                        حالة GGML_TYPE_Q4_0:
                        حالة GGML_TYPE_Q4_1:
                        حالة GGML_TYPE_Q5_0:
                        حالة GGML_TYPE_Q5_1:
                        حالة GGML_TYPE_Q8_0:
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_Q2_K:
                        حالة GGML_TYPE_Q3_K:
                        حالة GGML_TYPE_Q4_K:
                        حالة GGML_TYPE_Q5_K:
                        حالة GGML_TYPE_Q6_K:
                            أرجع src0->ne[0] % 4 == 0;
                        افتراضي:
                            اكسر;
                    }
                    اكسر;
                افتراضي:
                    اكسر;
            }
        }
    }
    أرجع خطأ;
}

صنف ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<صحيح, webgpu_pipeline> جمع_ردیفها_pipelines;       // key is fixed, no variants yet
    std::unordered_map<صحيح, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<صحيح, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<صحيح, webgpu_pipeline> argsort_ادغام_pipelines;  // key is order
    std::unordered_map<صحيح, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_ردیف_norm_pipeline_کلید, webgpu_pipeline, ggml_webgpu_ردیف_norm_pipeline_کلید_hash>
        ردیف_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_ردیفها_pipeline_کلید, webgpu_pipeline, ggml_webgpu_get_ردیفها_pipeline_کلید_hash>
        get_ردیفها_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_کلید, webgpu_pipeline, ggml_webgpu_unary_pipeline_کلید_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_کلید, webgpu_pipeline, ggml_webgpu_scale_pipeline_کلید_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_کلید, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_کلید_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_کلید, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_کلید_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_پیمایش_pipeline_کلید, webgpu_pipeline, ggml_webgpu_ssm_پیمایش_pipeline_کلید_hash>
        ssm_پیمایش_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_کلید,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_کلید_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_کلید, webgpu_pipeline, ggml_webgpu_pad_pipeline_کلید_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_دودویی_pipeline_کلید, webgpu_pipeline, ggml_webgpu_دودویی_pipeline_کلید_hash>
        دودویی_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_جمع_کن_id_pipeline_کلید, webgpu_pipeline, ggml_webgpu_جمع_کن_id_pipeline_کلید_hash>
        جمع_کن_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_کلید, webgpu_pipeline, ggml_webgpu_concat_pipeline_کلید_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_کلید, webgpu_pipeline, ggml_webgpu_repeat_pipeline_کلید_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_کلید,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_کلید_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_کلید, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_کلید_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_کلید,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_کلید_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_کلید, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_کلید_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_کلید, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_کلید_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_کلید, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_کلید_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_کلید,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_کلید_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_کلید, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_کلید_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_کلید, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_کلید_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_مجموعه_ردیفها_pipeline_کلید, webgpu_pipeline, ggml_webgpu_مجموعه_ردیفها_pipeline_کلید_hash>
        مجموعه_ردیفها_pipelines;
    std::unordered_map<ggml_webgpu_مجموعه_pipeline_کلید, webgpu_pipeline, ggml_webgpu_مجموعه_pipeline_کلید_hash> مجموعه_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_کلید, webgpu_pipeline, ggml_webgpu_cpy_pipeline_کلید_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_کلید, webgpu_pipeline, ggml_webgpu_glu_pipeline_کلید_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_کلید, webgpu_pipeline, ggml_webgpu_rope_pipeline_کلید_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_کلید, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_کلید_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_کلید, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_کلید_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_کلید, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_کلید_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_کلید, webgpu_pipeline, ggml_webgpu_im2col_pipeline_کلید_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_کلید,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_کلید_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_کلید, webgpu_pipeline, ggml_webgpu_upscale_pipeline_کلید_hash>
        upscale_pipelines;

  عام:
    ggml_webgpu_shader_lib(wgpu::Device device) { هذا->device = device; }

    webgpu_pipeline get_جمع_ردیفها_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        خودکار it = جمع_ردیفها_pipelines.find(1);
        إذا (it != جمع_ردیفها_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> تعریف_میکند;
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed        = preprocessor.preprocess(wgsl_جمع_ردیفها, تعریف_میکند);
        جمع_ردیفها_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        أرجع جمع_ردیفها_pipelines[1];
    }

    webgpu_pipeline get_ردیف_norm_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ردیف_norm_pipeline_کلید کلید = {};
        کلید.op                                = context.dst->op;
        کلید.src_type                          = context.src0->type;
        کلید.dst_type                          = context.dst->type;
        کلید.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = ردیف_norm_pipelines.find(کلید);
        إذا (it != ردیف_norm_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> تعریف_میکند;
        std::string              variant;

        بدل (کلید.op) {
            حالة GGML_OP_RMS_NORM:
                تعریف_میکند.push_back("RMS_NORM");
                variant = "rms_norm";
                اكسر;
            حالة GGML_OP_NORM:
                تعریف_میکند.push_back("NORM");
                variant = "norm";
                اكسر;
            حالة GGML_OP_L2_NORM:
                تعریف_میکند.push_back("L2_NORM");
                variant = "l2_norm";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        إذا (کلید.inplace) {
            تعریف_میکند.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (کلید.src_type == GGML_TYPE_F32) {
            تعریف_میکند.push_back("SRC_F32");
            variant += "_src_f32";
        } وإلا إذا (کلید.src_type == GGML_TYPE_F16) {
            تعریف_میکند.push_back("SRC_F16");
            variant += "_src_f16";
        }

        إذا (کلید.dst_type == GGML_TYPE_F32) {
            تعریف_میکند.push_back("DST_F32");
            variant += "_dst_f32";
        } وإلا إذا (کلید.dst_type == GGML_TYPE_F16) {
            تعریف_میکند.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ثابت uint32_t ردیف_norm_wg_اندازه = 128u;
        uint32_t       wg_اندازه          = std::min(context.max_wg_اندازه, ردیف_norm_wg_اندازه);
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        خودکار processed                  = preprocessor.preprocess(wgsl_ردیف_norm, تعریف_میکند);
        خودکار decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه              = wg_اندازه;
        decisions->inplace              = کلید.inplace;
        ردیف_norm_pipelines[کلید]         = ggml_webgpu_create_pipeline(device, processed, variant);
        ردیف_norm_pipelines[کلید].context = decisions;
        أرجع ردیف_norm_pipelines[کلید];
    }

    webgpu_pipeline get_argmax_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي vec4 = context.src0->ne[0] % 4 == 0;

        خودکار it = argmax_pipelines.find(vec4);
        إذا (it != argmax_pipelines.end()) {
            أرجع it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> تعریف_میکند;
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));
        إذا (vec4) {
            تعریف_میکند.push_back("VEC4");
            variant += "_vec4";
        }

        خودکار processed         = preprocessor.preprocess(wgsl_argmax, تعریف_میکند);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        أرجع argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_مجموعه_ردیفها_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_مجموعه_ردیفها_pipeline_کلید کلید       = {};
        کلید.src0_type                               = context.src0->type;
        کلید.dst_type                                = context.dst->type;
        کلید.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        کلید.i64_idx     = context.src1->type == GGML_TYPE_I64;
        کلید.pair_بلوکها = quantized && ((context.src0->ne[0] / ggml_blck_اندازه(context.dst->type)) % 2 == 0);

        خودکار it = مجموعه_ردیفها_pipelines.find(کلید);
        إذا (it != مجموعه_ردیفها_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "set_rows";

        بدل (context.dst->type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("DST_F32");
                variant += "_dstf32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("DST_F16");
                variant += "_dstf16";
                اكسر;
            حالة GGML_TYPE_Q8_0:
                تعریف_میکند.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                اكسر;
            حالة GGML_TYPE_Q4_0:
                تعریف_میکند.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        إذا (context.src0->type == GGML_TYPE_F16) {
            تعریف_میکند.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        إذا (کلید.vec4) {
            تعریف_میکند.push_back("VEC4");
            variant += "_vec4";
        }
        إذا (کلید.i64_idx) {
            تعریف_میکند.push_back("I64_IDX");
            variant += "_i64idx";
        }
        إذا (کلید.pair_بلوکها) {
            تعریف_میکند.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        ثابت خودکار & shader_منبع      = quantized ? wgsl_مجموعه_ردیفها_quant : wgsl_مجموعه_ردیفها;
        خودکار         processed          = preprocessor.preprocess(shader_منبع, تعریف_میکند);
        خودکار         decisions          = std::make_shared<ggml_webgpu_مجموعه_ردیفها_shader_decisions>();
        decisions->vec4                 = کلید.vec4;
        decisions->i64_idx              = کلید.i64_idx;
        decisions->pair_بلوکها          = کلید.pair_بلوکها;
        decisions->wg_اندازه              = context.max_wg_اندازه;
        مجموعه_ردیفها_pipelines[کلید]         = ggml_webgpu_create_pipeline(device, processed, variant);
        مجموعه_ردیفها_pipelines[کلید].context = decisions;
        أرجع مجموعه_ردیفها_pipelines[کلید];
    }

    webgpu_pipeline get_مجموعه_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_مجموعه_pipeline_کلید کلید = {};
        کلید.type                         = context.dst->type;
        کلید.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = مجموعه_pipelines.find(کلید);
        إذا (it != مجموعه_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "set";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریف_میکند.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for set shader");
        }

        إذا (کلید.inplace) {
            تعریف_میکند.push_back("INPLACE");
            variant += "_inplace";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_مجموعه, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        مجموعه_pipelines[کلید]       = pipeline;
        أرجع مجموعه_pipelines[کلید];
    }

    webgpu_pipeline get_cumsum_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        خودکار it = cumsum_pipelines.find(1);
        إذا (it != cumsum_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed      = preprocessor.preprocess(wgsl_cumsum, تعریف_میکند);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        أرجع cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_بالاترین_k = context.dst->op == GGML_OP_بالاترین_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t ترتیب =
            is_بالاترین_k ? (int32_t) GGML_مرتب_سازی_ترتیب_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        خودکار it = argsort_pipelines.find(ترتیب);
        إذا (it != argsort_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "argsort";
        تعریف_میکند.push_back(std::string("ORDER=") + std::to_string(ترتیب));
        variant += std::string("_order") + std::to_string(ترتیب);
        uint32_t wg_اندازه = 1;
        طالما (wg_اندازه * 2 <= context.max_wg_اندازه &&
               wg_اندازه * GGML_WEBGPU_I32_اندازه_بایتها <= context.wg_mem_حد_بایتها / 2) {
            wg_اندازه *= 2;
        }
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        خودکار processed                   = preprocessor.preprocess(wgsl_argsort, تعریف_میکند);
        خودکار decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه               = wg_اندازه;
        argsort_pipelines[ترتیب]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[ترتیب].context = decisions;
        أرجع argsort_pipelines[ترتیب];
    }

    webgpu_pipeline get_argsort_ادغام_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_بالاترین_k = context.dst->op == GGML_OP_بالاترین_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t ترتیب =
            is_بالاترین_k ? (int32_t) GGML_مرتب_سازی_ترتیب_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        خودکار it = argsort_ادغام_pipelines.find(ترتیب);
        إذا (it != argsort_ادغام_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "argsort_merge";
        تعریف_میکند.push_back(std::string("ORDER=") + std::to_string(ترتیب));
        variant += std::string("_order") + std::to_string(ترتیب);
        uint32_t wg_اندازه = std::min(GGML_WEBGPU_ARGSORT_ادغام_MAX_WG_اندازه, context.max_wg_اندازه);
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));

        خودکار processed                 = preprocessor.preprocess(wgsl_argsort_ادغام, تعریف_میکند);
        argsort_ادغام_pipelines[ترتیب] = ggml_webgpu_create_pipeline(device, processed, variant);
        أرجع argsort_ادغام_pipelines[ترتیب];
    }

    webgpu_pipeline get_get_ردیفها_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي vec4_aligned) {
        ثابت منطقي vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_ردیفها_pipeline_کلید کلید = {};
        کلید.src_type                          = context.src0->type;
        کلید.vectorized                        = (صحيح) vectorized;

        خودکار it = get_ردیفها_pipelines.find(کلید);
        إذا (it != get_ردیفها_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "get_rows";

        ثابت بنية ggml_type_traits * type_traits = ggml_get_type_traits(کلید.src_type);
        ثابت حرف *                    type_str    = type_traits->type_name;

        بدل (کلید.src_type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("FLOAT_PARALLEL");
                إذا (کلید.vectorized) {
                    تعریف_میکند.push_back("F32_VEC");
                    تعریف_میکند.push_back("SRC_TYPE=vec4<f32>");
                    تعریف_میکند.push_back("DST_TYPE=vec4<f32>");
                    تعریف_میکند.push_back("BLOCK_SIZE=4u");
                } وإلا {
                    تعریف_میکند.push_back("F32");
                    تعریف_میکند.push_back("SRC_TYPE=f32");
                    تعریف_میکند.push_back("DST_TYPE=f32");
                    تعریف_میکند.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("FLOAT_PARALLEL");
                تعریف_میکند.push_back("F16");
                تعریف_میکند.push_back("SRC_TYPE=f16");
                تعریف_میکند.push_back("DST_TYPE=f32");
                تعریف_میکند.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریف_میکند.push_back("FLOAT_PARALLEL");
                تعریف_میکند.push_back("I32");
                تعریف_میکند.push_back("SRC_TYPE=i32");
                تعریف_میکند.push_back("DST_TYPE=i32");
                تعریف_میکند.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                اكسر;
            حالة GGML_TYPE_BF16:
                تعریف_میکند.push_back("BF16");
                تعریف_میکند.push_back("SRC_TYPE=u32");
                تعریف_میکند.push_back("DST_TYPE=f32");
                تعریف_میکند.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_فوقانی = type_str;
                    std::transform(type_فوقانی.begin(), type_فوقانی.end(), type_فوقانی.begin(), ::به_بزرگ);

                    بدل (کلید.src_type) {
                        حالة GGML_TYPE_Q1_0:
                        حالة GGML_TYPE_Q2_0:
                        حالة GGML_TYPE_Q4_0:
                        حالة GGML_TYPE_Q5_0:
                        حالة GGML_TYPE_Q8_0:
                        حالة GGML_TYPE_Q3_K:
                        حالة GGML_TYPE_Q6_K:
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                تعریف_میکند.push_back("SRC_TYPE=u32");
                                تعریف_میکند.push_back("U32_DEQUANT_HELPERS");
                                اكسر;
                            }
                        افتراضي:
                            {
                                تعریف_میکند.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    تعریف_میکند.push_back("BYTE_HELPERS");
                    تعریف_میکند.push_back(type_فوقانی + "_T");
                    تعریف_میکند.push_back(type_فوقانی);
                    تعریف_میکند.push_back(type_فوقانی + "_SCALE_MIN");
                    تعریف_میکند.push_back(type_فوقانی + "_TABLES");
                    تعریف_میکند.push_back(type_فوقانی + "_GRID");
                    تعریف_میکند.push_back(type_فوقانی + "_LUT");

                    variant += "_";
                    variant += type_str;

                    تعریف_میکند.push_back("DST_TYPE=f32");

                    إذا (کلید.src_type == GGML_TYPE_Q1_0) {
                        تعریف_میکند.push_back("BLOCK_SIZE=128u");
                    } وإلا إذا ((کلید.src_type >= GGML_TYPE_Q4_0 && کلید.src_type <= GGML_TYPE_Q8_1) ||
                               کلید.src_type == GGML_TYPE_IQ4_NL || کلید.src_type == GGML_TYPE_MXFP4) {
                        تعریف_میکند.push_back("BLOCK_SIZE=32u");
                    } وإلا إذا (کلید.src_type == GGML_TYPE_Q2_0 || کلید.src_type == GGML_TYPE_NVFP4) {
                        تعریف_میکند.push_back("BLOCK_SIZE=64u");
                    } وإلا إذا (کلید.src_type >= GGML_TYPE_Q2_K) {
                        تعریف_میکند.push_back("BLOCK_SIZE=256u");
                    } وإلا {
                        تعریف_میکند.push_back("BLOCK_SIZE=1u");
                    }
                    اكسر;
                }
        }

        إذا (کلید.vectorized) {
            variant += "_vec";
        }

        تعریف_میکند.push_back("WG_SIZE=" + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_get_ردیفها, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_get_ردیفها_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_ردیفها_pipelines[کلید]  = pipeline;
        أرجع get_ردیفها_pipelines[کلید];
    }

    webgpu_pipeline get_scale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_کلید کلید = {};
        کلید.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = scale_pipelines.find(کلید);
        إذا (it != scale_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "scale";

        إذا (کلید.inplace) {
            تعریف_میکند.push_back("INPLACE");
            variant += "_inplace";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_scale, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[کلید]     = pipeline;
        أرجع scale_pipelines[کلید];
    }

    webgpu_pipeline get_solve_tri_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_کلید کلید = {};
        کلید.type                               = context.dst->type;
        کلید.n                                  = (صحيح) context.src0->ne[0];
        کلید.k                                  = (صحيح) context.src1->ne[0];

        خودکار it = solve_tri_pipelines.find(کلید);
        إذا (it != solve_tri_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "solve_tri";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ثابت uint32_t wg_اندازه       = std::min((uint32_t) کلید.n, context.max_wg_اندازه);
        ثابت uint32_t k_tile        = wg_اندازه;
        ثابت uint32_t بایتها_در_هر_ردیف = ((uint32_t) کلید.n + wg_اندازه) * GGML_WEBGPU_F32_اندازه_بایتها;
        ثابت uint32_t batch_n       = (uint32_t) (context.wg_mem_حد_بایتها / بایتها_در_هر_ردیف);

        تعریف_میکند.push_back(std::string("N=") + std::to_string(کلید.n));
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        تعریف_میکند.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        تعریف_میکند.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        خودکار processed           = preprocessor.preprocess(wgsl_solve_tri, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[کلید] = pipeline;
        أرجع solve_tri_pipelines[کلید];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_کلید کلید = {};
        کلید.type                              = context.dst->type;
        کلید.vectorized                        = context.src1->ne[0] == 4;

        خودکار it = ssm_conv_pipelines.find(کلید);
        إذا (it != ssm_conv_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "ssm_conv";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        إذا (کلید.vectorized) {
            تعریف_میکند.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t بلوک_اندازه_3    = 32u;
        constexpr uint32_t نشانهها_در_هر_wg = 8u;

        تعریف_میکند.push_back("BLOCK_SIZE=" + std::to_string(بلوک_اندازه_3) + "u");
        تعریف_میکند.push_back("TOKENS_PER_WG=" + std::to_string(نشانهها_در_هر_wg) + "u");

        خودکار processed           = preprocessor.preprocess(wgsl_ssm_conv, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->بلوک_اندازه_3    = بلوک_اندازه_3;
        decisions->نشانهها_در_هر_wg = نشانهها_در_هر_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[کلید]  = pipeline;
        أرجع ssm_conv_pipelines[کلید];
    }

    webgpu_pipeline get_ssm_پیمایش_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_پیمایش_pipeline_کلید کلید = {};
        کلید.type                              = context.dst->type;
        کلید.d_state                           = (صحيح) context.src0->ne[0];
        کلید.xbc_overlap                       = xbc_overlap;

        خودکار it = ssm_پیمایش_pipelines.find(کلید);
        إذا (it != ssm_پیمایش_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "ssm_scan";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ثابت uint32_t wg_اندازه = (uint32_t) کلید.d_state;

        constexpr uint32_t نشانهها_در_هر_tile = 4u;

        تعریف_میکند.push_back("WG_SIZE=" + std::to_string(wg_اندازه) + "u");
        تعریف_میکند.push_back("TOKENS_PER_TILE=" + std::to_string(نشانهها_در_هر_tile) + "u");

        إذا (context.supports_subgroups) {
            تعریف_میکند.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } وإلا {
            variant += "_wg_reduce";
        }

        إذا (کلید.xbc_overlap == 0b110) {  // x/B
            تعریف_میکند.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } وإلا إذا (کلید.xbc_overlap == 0b011) {  // B/C
            تعریف_میکند.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } وإلا إذا (کلید.xbc_overlap == 0b111) {  // x/B/C
            تعریف_میکند.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(کلید.d_state);

        خودکار processed             = preprocessor.preprocess(wgsl_ssm_پیمایش, تعریف_میکند);
        خودکار decisions             = std::make_shared<ggml_webgpu_ssm_پیمایش_shader_decisions>();
        decisions->wg_اندازه         = wg_اندازه;
        decisions->نشانهها_در_هر_tile = نشانهها_در_هر_tile;
        decisions->xbc_overlap     = کلید.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_پیمایش_pipelines[کلید]    = pipeline;
        أرجع ssm_پیمایش_pipelines[کلید];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_کلید کلید = {};
        کلید.type                                     = context.dst->type;
        کلید.s_v                                      = (صحيح) context.src2->ne[0];
        کلید.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        کلید.fused_cache                              = context.dst_fuse != اشاره_تهی;

        خودکار it = gated_delta_net_pipelines.find(کلید);
        إذا (it != gated_delta_net_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "gated_delta_net";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        إذا (کلید.kda) {
            تعریف_میکند.push_back("KDA");
            variant += "_kda";
        }

        إذا (کلید.fused_cache) {
            تعریف_میکند.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        تعریف_میکند.push_back("S_V=" + std::to_string(کلید.s_v) + "u");
        تعریف_میکند.push_back("WG_SIZE=" + std::to_string(کلید.s_v) + "u");

        خودکار            processed      = preprocessor.preprocess(wgsl_gated_delta_net, تعریف_میکند);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[کلید] = pipeline;
        أرجع gated_delta_net_pipelines[کلید];
    }

    webgpu_pipeline get_pad_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_کلید کلید = {};
        کلید.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        خودکار it = pad_pipelines.find(کلید);
        إذا (it != pad_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "pad";

        إذا (کلید.circular) {
            تعریف_میکند.push_back("CIRCULAR");
            variant += "_circular";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_pad, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[کلید]       = pipeline;
        أرجع pad_pipelines[کلید];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_کلید کلید = {};
        کلید.src0_type                            = context.src0->type;

        خودکار it = quantize_q8_pipelines.find(کلید);
        إذا (it != quantize_q8_pipelines.end()) {
            أرجع it->second;
        }
        ثابت حرف *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> تعریف_میکند;
        std::string              variant = "quantize_q8";

        uint32_t wg_اندازه = WEBGPU_MUL_MAT_VEC_WG_اندازه;

        تعریف_میکند.push_back("SRC1_INNER_TYPE=f32");
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));

        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_فوقانی  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_فوقانی.begin(), type_فوقانی.end(), type_فوقانی.begin(), ::به_بزرگ);

        تعریف_میکند.push_back("MUL_ACC_" + type_فوقانی);
        تعریف_میکند.push_back("Q8_1_T");

        تعریف_میکند.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        خودکار processed             = preprocessor.preprocess(shader_src, تعریف_میکند);
        خودکار decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه         = wg_اندازه;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[کلید] = pipeline;
        أرجع quantize_q8_pipelines[کلید];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_کلید کلید = {};
        کلید.src0_type                            = context.src0->type;
        کلید.src1_type                            = context.src1->type;
        کلید.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        کلید.رقم_تعداد_ستون   = context.dst->ne[1];
        کلید.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_حاصلضرب, context.vendor);
        کلید.src_overlap = src_overlap;

        خودکار it = mul_mat_vec_pipelines.find(کلید);
        إذا (it != mul_mat_vec_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant    = "mul_mat_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=f32");
                تعریف_میکند.push_back("MUL_ACC_FLOAT");
                تعریف_میکند.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=f16");
                تعریف_میکند.push_back("MUL_ACC_FLOAT");
                تعریف_میکند.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=u32");
                تعریف_میکند.push_back("MUL_ACC_FLOAT");
                تعریف_میکند.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_فوقانی  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_فوقانی.begin(), type_فوقانی.end(), type_فوقانی.begin(), ::به_بزرگ);

                    تعریف_میکند.push_back("BYTE_HELPERS");
                    تعریف_میکند.push_back("MUL_ACC_" + type_فوقانی);
                    تعریف_میکند.push_back("U32_DEQUANT_HELPERS");
                    تعریف_میکند.push_back("SRC0_INNER_TYPE=u32");
                    بدل (context.src0->type) {
                        حالة GGML_TYPE_Q4_0:
                        حالة GGML_TYPE_Q4_1:
                        حالة GGML_TYPE_Q5_0:
                        حالة GGML_TYPE_Q5_1:
                        حالة GGML_TYPE_Q8_0:
                            إذا (کلید.use_mmvq) {
                                تعریف_میکند.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_Q1_0:
                        حالة GGML_TYPE_Q2_K:
                        حالة GGML_TYPE_Q3_K:
                        حالة GGML_TYPE_Q4_K:
                        حالة GGML_TYPE_Q5_K:
                        حالة GGML_TYPE_Q6_K:
                            إذا (کلید.use_mmvq) {
                                تعریف_میکند.push_back("K_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            تعریف_میکند.push_back(type_فوقانی + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ3_XXS:
                            تعریف_میکند.push_back(type_فوقانی + "_GRID");
                            تعریف_میکند.push_back(type_فوقانی + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                            تعریف_میکند.push_back(type_فوقانی + "_LUT");
                            إذا (کلید.use_mmvq) {
                                تعریف_میکند.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_NVFP4:
                            تعریف_میکند.push_back(type_فوقانی + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }
                    اكسر;
                }
        }

        // src1 type (vector)
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        تعریف_میکند.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_اندازه        = WEBGPU_MUL_MAT_VEC_WG_اندازه;
        uint32_t outputs_در_هر_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_در_هر_WG;

        إذا (کلید.src0_type == GGML_TYPE_Q1_0) {
            outputs_در_هر_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_در_هر_WG;
        } وإلا إذا (کلید.src0_type >= GGML_TYPE_Q2_K) {
            outputs_در_هر_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_در_هر_WG;
        } وإلا إذا (کلید.src0_type >= GGML_TYPE_Q4_0) {
            outputs_در_هر_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_در_هر_WG;
        }

        إذا (کلید.use_mmvq) {
            تعریف_میکند.push_back("MMVQ");
            تعریف_میکند.push_back("Q8_1_T");
        }

        إذا (کلید.src_overlap) {
            تعریف_میکند.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        تعریف_میکند.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_در_هر_wg));
        تعریف_میکند.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (کلید.vectorized) {
            variant += "_vectorized";
        }
        تعریف_میکند.push_back(std::string("NUM_COLS=") + std::to_string(کلید.رقم_تعداد_ستون));

        خودکار processed            = preprocessor.preprocess(shader_src, تعریف_میکند);
        خودکار decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_اندازه        = wg_اندازه;
        decisions->outputs_در_هر_wg = outputs_در_هر_wg;
        decisions->vec_اندازه       = کلید.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[کلید] = pipeline;
        أرجع mul_mat_vec_pipelines[کلید];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_mul_mat_pipeline_کلید کلید = {};
        کلید.src0_type                        = context.src0->type;
        کلید.src1_type                        = context.src1->type;
        کلید.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        کلید.use_subgroup_ماتریس = context.supports_subgroup_ماتریس;
        کلید.src_overlap         = src_overlap;

        خودکار it = mul_mat_fast_pipelines.find(کلید);
        إذا (it != mul_mat_fast_pipelines.end()) {
            أرجع it->second;
        }

        ثابت حرف * shader_src = کلید.use_subgroup_ماتریس ? wgsl_mul_mat_subgroup_ماتریس : wgsl_mul_mat_reg_tile;
        std::vector<std::string> تعریف_میکند;
        std::string              variant = کلید.use_subgroup_ماتریس ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=f32");
                تعریف_میکند.push_back("TYPE_F32");
                تعریف_میکند.push_back("MUL_ACC_FLOAT");
                تعریف_میکند.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریف_میکند.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=f16");
                تعریف_میکند.push_back("TYPE_F16");
                تعریف_میکند.push_back("MUL_ACC_FLOAT");
                تعریف_میکند.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریف_میکند.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=u32");
                تعریف_میکند.push_back("TYPE_BF16");
                تعریف_میکند.push_back("MUL_ACC_FLOAT");
                تعریف_میکند.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریف_میکند.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_فوقانی = src0_name;
                    std::transform(type_فوقانی.begin(), type_فوقانی.end(), type_فوقانی.begin(), ::به_بزرگ);

                    تعریف_میکند.push_back("BYTE_HELPERS");
                    تعریف_میکند.push_back("MUL_ACC_" + type_فوقانی);
                    تعریف_میکند.push_back("INIT_SRC0_SHMEM_" + type_فوقانی);
                    تعریف_میکند.push_back("INIT_SRC1_SHMEM_FLOAT");
                    تعریف_میکند.push_back("U32_DEQUANT_HELPERS");
                    تعریف_میکند.push_back("SRC0_INNER_TYPE=u32");

                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            تعریف_میکند.push_back(type_فوقانی + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                            تعریف_میکند.push_back(type_فوقانی + "_GRID");
                            تعریف_میکند.push_back(type_فوقانی + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            تعریف_میکند.push_back(type_فوقانی + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }

                    variant += std::string("_") + src0_name;
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        تعریف_میکند.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        ثابت منطقي is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        إذا (کلید.use_subgroup_ماتریس) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } وإلا {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        تعریف_میکند.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        تعریف_میکند.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        إذا (کلید.use_subgroup_ماتریس) {
            تعریف_میکند.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            تعریف_میکند.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_اندازه) + "u");
            تعریف_میکند.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            تعریف_میکند.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            تعریف_میکند.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_ماتریس_M) + "u");
            تعریف_میکند.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_ماتریس_N) + "u");
            تعریف_میکند.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            تعریف_میکند.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            تعریف_میکند.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        إذا (کلید.vectorized) {
            variant += "_vectorized";
        }

        إذا (کلید.src_overlap) {
            تعریف_میکند.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        إذا (!کلید.use_subgroup_ماتریس) {
            تعریف_میکند.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_اندازه_M) + "u");
            تعریف_میکند.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_اندازه_N) + "u");
            تعریف_میکند.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        خودکار processed = preprocessor.preprocess(shader_src, تعریف_میکند);

        خودکار decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_ماتریس = کلید.use_subgroup_ماتریس;
        إذا (کلید.use_subgroup_ماتریس) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_ماتریس_m = WEBGPU_MUL_MAT_SUBGROUP_ماتریس_M;
            decisions->subgroup_ماتریس_n = WEBGPU_MUL_MAT_SUBGROUP_ماتریس_N;
            decisions->wg_اندازه           = context.max_subgroup_اندازه;
        } وإلا {
            decisions->wg_اندازه_m       = WEBGPU_MUL_MAT_WG_اندازه_M;
            decisions->wg_اندازه_n       = WEBGPU_MUL_MAT_WG_اندازه_N;
            decisions->wg_اندازه         = WEBGPU_MUL_MAT_WG_اندازه_M * WEBGPU_MUL_MAT_WG_اندازه_N;
            decisions->mul_mat_wg_اندازه = WEBGPU_MUL_MAT_WG_اندازه;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[کلید] = pipeline;
        أرجع mul_mat_fast_pipelines[کلید];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_کلید کلید = {};
        کلید.n_experts                                  = (uint32_t) context.src0->ne[2];

        خودکار it = mul_mat_id_gather_pipelines.find(کلید);
        إذا (it != mul_mat_id_gather_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> تعریف_میکند;
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه) + "u");
        تعریف_میکند.push_back(std::string("N_EXPERT=") + std::to_string(کلید.n_experts) + "u");
        تعریف_میکند.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_اندازه_N) + "u");

        خودکار processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, تعریف_میکند);
        خودکار decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه = context.max_wg_اندازه;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[کلید] = pipeline;
        أرجع pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_کلید کلید = {};
        کلید.src0_type                           = context.src0->type;
        کلید.src1_type                           = context.src1->type;
        کلید.n_experts                           = context.src0->ne[2];
        کلید.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        خودکار it = mul_mat_id_pipelines.find(کلید);
        إذا (it != mul_mat_id_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "mul_mat_id";
        تعریف_میکند.push_back("MUL_MAT_ID");

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=f32");
                تعریف_میکند.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریف_میکند.push_back("INIT_SRC1_SHMEM_FLOAT");
                تعریف_میکند.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=f16");
                تعریف_میکند.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریف_میکند.push_back("INIT_SRC1_SHMEM_FLOAT");
                تعریف_میکند.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=u32");
                تعریف_میکند.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریف_میکند.push_back("INIT_SRC1_SHMEM_FLOAT");
                تعریف_میکند.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_فوقانی = src0_name;
                    std::transform(type_فوقانی.begin(), type_فوقانی.end(), type_فوقانی.begin(), ::به_بزرگ);

                    تعریف_میکند.push_back("BYTE_HELPERS");
                    تعریف_میکند.push_back("INIT_SRC0_SHMEM_" + type_فوقانی);
                    تعریف_میکند.push_back("INIT_SRC1_SHMEM_FLOAT");
                    تعریف_میکند.push_back("U32_DEQUANT_HELPERS");
                    تعریف_میکند.push_back("SRC0_INNER_TYPE=u32");

                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            تعریف_میکند.push_back(type_فوقانی + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                            تعریف_میکند.push_back(type_فوقانی + "_GRID");
                            تعریف_میکند.push_back(type_فوقانی + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            تعریف_میکند.push_back(type_فوقانی + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }

                    variant += std::string("_") + src0_name;
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        تعریف_میکند.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ثابت uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        تعریف_میکند.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        تعریف_میکند.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        تعریف_میکند.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        تعریف_میکند.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_اندازه_M) + "u");
        تعریف_میکند.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_اندازه_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        إذا (کلید.vectorized) {
            variant += "_vectorized";
        }

        خودکار processed = preprocessor.preprocess(wgsl_mul_mat_id, تعریف_میکند);

        خودکار decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_اندازه_m = WEBGPU_MUL_MAT_WG_اندازه_M;
        decisions->wg_اندازه_n = WEBGPU_MUL_MAT_WG_اندازه_N;
        decisions->wg_اندازه   = WEBGPU_MUL_MAT_WG_اندازه_M * WEBGPU_MUL_MAT_WG_اندازه_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[کلید] = pipeline;
        أرجع mul_mat_id_pipelines[کلید];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_کلید کلید = {};
        کلید.src0_type                           = context.src0->type;
        کلید.src1_type                           = context.src1->type;
        کلید.n_experts                           = context.src0->ne[2];
        کلید.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        خودکار it = mul_mat_id_vec_pipelines.find(کلید);
        إذا (it != mul_mat_id_vec_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant    = "mul_mat_id_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=f32");
                تعریف_میکند.push_back("MUL_ACC_FLOAT");
                تعریف_میکند.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=f16");
                تعریف_میکند.push_back("MUL_ACC_FLOAT");
                تعریف_میکند.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                تعریف_میکند.push_back("SRC0_INNER_TYPE=u32");
                تعریف_میکند.push_back("MUL_ACC_FLOAT");
                تعریف_میکند.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_فوقانی  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_فوقانی.begin(), type_فوقانی.end(), type_فوقانی.begin(), ::به_بزرگ);

                    تعریف_میکند.push_back("BYTE_HELPERS");
                    تعریف_میکند.push_back("MUL_ACC_" + type_فوقانی);
                    تعریف_میکند.push_back("U32_DEQUANT_HELPERS");
                    تعریف_میکند.push_back("SRC0_INNER_TYPE=u32");
                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            تعریف_میکند.push_back(type_فوقانی + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ3_XXS:
                            تعریف_میکند.push_back(type_فوقانی + "_GRID");
                            تعریف_میکند.push_back(type_فوقانی + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            تعریف_میکند.push_back(type_فوقانی + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        تعریف_میکند.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_اندازه        = WEBGPU_MUL_MAT_VEC_WG_اندازه;
        uint32_t outputs_در_هر_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_در_هر_WG;

        إذا (کلید.src0_type == GGML_TYPE_Q1_0) {
            outputs_در_هر_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_در_هر_WG;
        } وإلا إذا (کلید.src0_type >= GGML_TYPE_Q2_K) {
            outputs_در_هر_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_در_هر_WG;
        } وإلا إذا (کلید.src0_type >= GGML_TYPE_Q4_0) {
            outputs_در_هر_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_در_هر_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        تعریف_میکند.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_در_هر_wg));
        تعریف_میکند.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (کلید.vectorized) {
            variant += "_vectorized";
        }
        تعریف_میکند.push_back(std::string("NUM_COLS=1"));

        تعریف_میکند.push_back(std::string("N_EXPERTS=") + std::to_string(کلید.n_experts));

        خودکار processed = preprocessor.preprocess(shader_src, تعریف_میکند);

        خودکار decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_اندازه        = wg_اندازه;
        decisions->outputs_در_هر_wg = outputs_در_هر_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[کلید] = pipeline;
        أرجع mul_mat_id_vec_pipelines[کلید];
    }

    webgpu_pipeline get_unary_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                     is_unary = context.dst->op == GGML_OP_UNARY;
        ثابت صحيح                      op       = is_unary ? (صحيح) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_کلید کلید      = {};
        کلید.type                                = context.dst->type;
        کلید.op                                  = op;
        کلید.is_unary                            = is_unary;
        کلید.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        کلید.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        خودکار it = unary_pipelines.find(کلید);
        إذا (it != unary_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant =
            کلید.is_unary ? ggml_unary_op_name((ggml_unary_op) کلید.op) : ggml_op_name((ggml_op) کلید.op);
        تعریف_میکند.push_back(variant);

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for unary shader");
        }

        إذا (کلید.inplace) {
            تعریف_میکند.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (op == GGML_OP_TRI) {
            بدل (کلید.ttype) {
                حالة GGML_TRI_TYPE_پایینی:
                    تعریف_میکند.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    اكسر;
                حالة GGML_TRI_TYPE_پایینی_DIAG:
                    تعریف_میکند.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    اكسر;
                حالة GGML_TRI_TYPE_فوقانی:
                    تعریف_میکند.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    اكسر;
                حالة GGML_TRI_TYPE_فوقانی_DIAG:
                    تعریف_میکند.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    اكسر;
                افتراضي:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_unary, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[کلید]     = pipeline;
        أرجع unary_pipelines[کلید];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_کلید کلید = {};
        کلید.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        کلید.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        کلید.src_overlap                           = src_overlap;

        خودکار it = rms_norm_mul_pipelines.find(کلید);
        إذا (it != rms_norm_mul_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        إذا (کلید.inplace) {
            تعریف_میکند.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (کلید.overlap) {
            تعریف_میکند.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (کلید.src_overlap) {
            تعریف_میکند.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, تعریف_میکند);
        خودکار pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_اندازه     = context.max_wg_اندازه;
        pipeline_decisions->inplace     = کلید.inplace;
        pipeline_decisions->overlap     = کلید.overlap;
        pipeline_decisions->src_overlap = کلید.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[کلید]     = pipeline;
        أرجع rms_norm_mul_pipelines[کلید];
    }

    webgpu_pipeline get_دودویی_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_دودویی_pipeline_کلید کلید = {};
        کلید.type                            = context.dst->type;
        کلید.op                              = context.dst->op;
        کلید.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        کلید.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        کلید.src_overlap                     = src_overlap;

        خودکار it = دودویی_pipelines.find(کلید);
        إذا (it != دودویی_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              op_name = ggml_op_name((ggml_op) کلید.op);
        std::string              variant = op_name;

        تعریف_میکند.push_back(std::string("OP_") + op_name);

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for binary shader");
        }

        إذا (کلید.inplace) {
            تعریف_میکند.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (کلید.overlap) {
            تعریف_میکند.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (کلید.src_overlap) {
            تعریف_میکند.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed                  = preprocessor.preprocess(wgsl_دودویی, تعریف_میکند);
        خودکار pipeline_decisions         = std::make_shared<ggml_webgpu_دودویی_shader_decisions>();
        pipeline_decisions->wg_اندازه     = context.max_wg_اندازه;
        pipeline_decisions->inplace     = کلید.inplace;
        pipeline_decisions->overlap     = کلید.overlap;
        pipeline_decisions->src_overlap = کلید.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        دودویی_pipelines[کلید]    = pipeline;
        أرجع دودویی_pipelines[کلید];
    }

    webgpu_pipeline get_جمع_کن_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_جمع_کن_id_pipeline_کلید کلید = {};
        کلید.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = جمع_کن_id_pipelines.find(کلید);
        إذا (it != جمع_کن_id_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant    = "add_id";
        ثابت حرف *             shader_src = wgsl_جمع_کن_id;

        إذا (کلید.inplace) {
            تعریف_میکند.push_back("INPLACE");
            variant += "_inplace";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed              = preprocessor.preprocess(shader_src, تعریف_میکند);
        خودکار pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_اندازه = context.max_wg_اندازه;
        pipeline_decisions->inplace = کلید.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        جمع_کن_id_pipelines[کلید]    = pipeline;
        أرجع pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_concat_pipeline_کلید کلید = {};
        کلید.type                            = context.dst->type;
        کلید.src_overlap                     = src_overlap;

        خودکار it = concat_pipelines.find(کلید);
        إذا (it != concat_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "concat";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریف_میکند.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for concat shader");
        }

        إذا (کلید.src_overlap) {
            تعریف_میکند.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_concat, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_دودویی_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->src_overlap   = کلید.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[کلید]    = pipeline;
        أرجع concat_pipelines[کلید];
    }

    webgpu_pipeline get_repeat_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_کلید کلید = {};
        کلید.type                            = context.dst->type;

        خودکار it = repeat_pipelines.find(کلید);
        إذا (it != repeat_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "repeat";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریف_میکند.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            حالة GGML_TYPE_I16:
                تعریف_میکند.push_back("TYPE_I16");
                variant += "_i16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_repeat, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[کلید]    = pipeline;
        أرجع repeat_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ثابت منطقي can_use_subgroup_ماتریس = ggml_webgpu_flash_attn_can_use_subgroup_ماتریس_مسیر(
            context.supports_subgroup_ماتریس, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_ماتریس                    = can_use_subgroup_ماتریس;
        decisions.q_tile = decisions.use_sg_ماتریس ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_کلید کلید = {};
        کلید.common                              = ggml_webgpu_flash_attn_make_common_pipeline_کلید(
            context, decisions.use_sg_ماتریس ? context.sg_mat_k : 1u, kv_overlap);
        کلید.common.k_مستقیم &= decisions.use_sg_ماتریس && کلید.common.k_type == GGML_TYPE_F16;
        کلید.common.v_مستقیم &= decisions.use_sg_ماتریس && کلید.common.v_type == GGML_TYPE_F16;
        کلید.use_sg_ماتریس = decisions.use_sg_ماتریس;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_حد_بایتها, decisions.q_tile, decisions.use_sg_ماتریس ? context.sg_mat_n : 1u,
            کلید.common.سر_dim_qk, کلید.common.سر_dim_v, کلید.common.has_ماسک,
            کلید.common.k_مستقیم || کلید.common.v_مستقیم);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        إذا (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_ماتریس);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_ماتریس = خطأ;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            کلید.common.k_مستقیم     = خطأ;
            کلید.common.v_مستقیم     = خطأ;
            کلید.use_sg_ماتریس       = خطأ;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_حد_بایتها, decisions.q_tile, 1u, کلید.common.سر_dim_qk, کلید.common.سر_dim_v,
                کلید.common.has_ماسک, کلید.common.k_مستقیم || کلید.common.v_مستقیم);
            GGML_ASSERT(max_kv_tile > 0);
        }

        decisions.kv_tile = decisions.use_sg_ماتریس ?
                                std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES) :
                                std::min(GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE, max_kv_tile);
        decisions.wg_اندازه =
            decisions.use_sg_ماتریس ?
                std::max(context.max_subgroup_اندازه, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_اندازه) :
                std::min(context.max_wg_اندازه, std::max(GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_اندازه,
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * context.max_subgroup_اندازه));

        إذا (کلید.common.k_مستقیم || کلید.common.v_مستقیم) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            طالما (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_ماتریس ? context.sg_mat_n : context.min_subgroup_اندازه;
            }
        }

        خودکار it = flash_attn_pipelines.find(کلید);
        إذا (it != flash_attn_pipelines.end()) {
            أرجع it->second;
        }

        std::string              variant = decisions.use_sg_ماتریس ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> تعریف_میکند = ggml_webgpu_flash_attn_common_تعریف_میکند(کلید.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_اندازه);
        ثابت حرف *             shader_src = اشاره_تهی;
        إذا (!کلید.use_sg_ماتریس) {
            shader_src = wgsl_flash_attn_tile;
            تعریف_میکند.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_اندازه) + "u");
            تعریف_میکند.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_اندازه) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_اندازه) + "_" +
                       std::to_string(context.max_subgroup_اندازه);
        } وإلا {
            shader_src = wgsl_flash_attn;
            تعریف_میکند.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            تعریف_میکند.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            تعریف_میکند.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        خودکار            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, تعریف_میکند), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[کلید] = pipeline;
        أرجع flash_attn_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_کلید کلید = {};
        کلید.common = ggml_webgpu_flash_attn_make_common_pipeline_کلید(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_عرض,
                                                                     kv_overlap);

        خودکار it = flash_attn_vec_pipelines.find(کلید);
        إذا (it != flash_attn_vec_pipelines.end()) {
            أرجع it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_حد_بایتها, کلید.common.سر_dim_qk,
                                                                   کلید.common.سر_dim_v, کلید.common.has_ماسک,
                                                                   کلید.common.k_مستقیم || کلید.common.v_مستقیم);
        decisions.wg_اندازه = context.max_subgroup_اندازه;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> تعریف_میکند =
            ggml_webgpu_flash_attn_common_تعریف_میکند(کلید.common, variant, 1u, decisions.kv_tile, decisions.wg_اندازه);
        إذا (کلید.common.has_ماسک) {
            تعریف_میکند.push_back("BLK");
            variant.resize(variant.size() - (حجم("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_تقسیم = context.min_subgroup_اندازه;
        ثابت uint32_t D       = کلید.common.سر_dim_qk | کلید.common.سر_dim_v;
        ثابت uint32_t D_lsb   = D & (~(D - 1u));
        d_تقسیم                = std::min(std::min(context.min_subgroup_اندازه, 4u), std::max(D_lsb / 4u, 1u));

        تعریف_میکند.push_back(std::string("D_SPLIT=") + std::to_string(d_تقسیم));
        variant += "_dsplit" + std::to_string(d_تقسیم);

        خودکار            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_تقسیم, تعریف_میکند), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[کلید] = pipeline;
        أرجع flash_attn_vec_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_کلید کلید = {};
        کلید.kv_tile                                 = kv_tile;
        خودکار it                                     = flash_attn_blk_pipelines.find(کلید);
        إذا (it != flash_attn_blk_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "flash_attn_vec_blk";

        تعریف_میکند.push_back(std::string("KV_TILE=") + std::to_string(کلید.kv_tile));
        variant += std::string("_kvt") + std::to_string(کلید.kv_tile);

        uint32_t wg_اندازه = 1;
        طالما ((wg_اندازه << 1) <= context.max_wg_اندازه) {
            wg_اندازه <<= 1;
        }
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        variant += std::string("_wg") + std::to_string(wg_اندازه);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, تعریف_میکند), variant);
        flash_attn_blk_pipelines[کلید] = pipeline;
        أرجع flash_attn_blk_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید کلید = {};
        کلید.سر_dim_v                                     = (uint32_t) context.src2->ne[0];
        کلید.dst_type                                       = context.dst->type;
        کلید.wg_اندازه                                        = context.max_wg_اندازه;
        خودکار it                                            = flash_attn_vec_reduce_pipelines.find(کلید);
        إذا (it != flash_attn_vec_reduce_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "flash_attn_vec_reduce";

        بدل (کلید.dst_type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("DST_F32");
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("DST_F16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(کلید.dst_type);

        تعریف_میکند.push_back(std::string("HEAD_DIM_V=") + std::to_string(کلید.سر_dim_v));
        variant += std::string("_hsv") + std::to_string(کلید.سر_dim_v);

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));
        variant += std::string("_wg") + std::to_string(context.max_wg_اندازه);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, تعریف_میکند), variant);
        flash_attn_vec_reduce_pipelines[کلید] = pipeline;
        أرجع flash_attn_vec_reduce_pipelines[کلید];
    }

    webgpu_pipeline get_cpy_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_کلید کلید = {};
        کلید.src_type                     = context.src0->type;
        کلید.dst_type                     = context.dst->type;

        خودکار it = cpy_pipelines.find(کلید);
        إذا (it != cpy_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "cpy";

        بدل (کلید.src_type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("SRC_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("SRC_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریف_میکند.push_back("SRC_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        بدل (کلید.dst_type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("DST_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("DST_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریف_میکند.push_back("DST_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_cpy, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[کلید]       = pipeline;
        أرجع cpy_pipelines[کلید];
    }

    webgpu_pipeline get_glu_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_glu_pipeline_کلید کلید = {};
        کلید.glu_op                       = ggml_get_glu_op(context.dst);
        کلید.type                         = context.dst->type;
        کلید.تقسیم                        = (context.src1 != اشاره_تهی);
        کلید.src_overlap                  = src_overlap;

        خودکار it = glu_pipelines.find(کلید);
        إذا (it != glu_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "glu";

        بدل (کلید.glu_op) {
            حالة GGML_GLU_OP_REGLU:
                تعریف_میکند.push_back("OP_REGLU");
                variant += "_reglu";
                اكسر;
            حالة GGML_GLU_OP_GEGLU:
                تعریف_میکند.push_back("OP_GEGLU");
                variant += "_geglu";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU:
                تعریف_میکند.push_back("OP_SWIGLU");
                variant += "_swiglu";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU_OAI:
                تعریف_میکند.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                اكسر;
            حالة GGML_GLU_OP_GEGLU_ERF:
                تعریف_میکند.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                اكسر;
            حالة GGML_GLU_OP_GEGLU_سریع:
                تعریف_میکند.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU_CLAMP:
                تعریف_میکند.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported GLU op");
        }
        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        إذا (کلید.src_overlap) {
            تعریف_میکند.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } وإلا إذا (کلید.تقسیم) {
            variant += "_split";
        } وإلا {
            تعریف_میکند.push_back("NO_SPLIT");
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_glu, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->src_overlap   = کلید.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[کلید]       = pipeline;
        أرجع glu_pipelines[کلید];
    }

    webgpu_pipeline get_rope_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_کلید کلید = {};
        کلید.type                          = context.dst->type;
        کلید.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        کلید.has_ff                        = (context.src2 != اشاره_تهی);

        خودکار it = rope_pipelines.find(کلید);
        إذا (it != rope_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "rope";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                تعریف_میکند.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریف_میکند.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        إذا (کلید.inplace) {
            تعریف_میکند.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (کلید.has_ff) {
            تعریف_میکند.push_back("FF_FUNC");
            variant += "_ff";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_rope, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[کلید]      = pipeline;
        أرجع rope_pipelines[کلید];
    }

    webgpu_pipeline get_soft_max_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_کلید کلید = {};
        کلید.ماسک_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        کلید.has_ماسک                          = (context.src1 != اشاره_تهی);
        کلید.has_sink                          = (context.src2 != اشاره_تهی);
        کلید.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = soft_max_pipelines.find(کلید);
        إذا (it != soft_max_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "soft_max";

        إذا (کلید.has_ماسک) {
            تعریف_میکند.push_back("HAS_MASK");
            بدل (کلید.ماسک_type) {
                حالة GGML_TYPE_F32:
                    تعریف_میکند.push_back("MASK_F32");
                    variant += "_mask_f32";
                    اكسر;
                حالة GGML_TYPE_F16:
                    تعریف_میکند.push_back("MASK_F16");
                    variant += "_mask_f16";
                    اكسر;
                افتراضي:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        إذا (کلید.has_sink) {
            تعریف_میکند.push_back("HAS_SINK");
            variant += "_sink";
        }

        إذا (کلید.inplace) {
            تعریف_میکند.push_back("INPLACE");
            variant += "_inplace";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_soft_max, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[کلید]  = pipeline;
        أرجع soft_max_pipelines[کلید];
    }

    webgpu_pipeline get_conv2d_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_کلید کلید = {};
        کلید.وزن_type                     = context.src0->type;
        کلید.درون_داد_type                      = context.src1->type;
        کلید.output_type                     = context.dst->type;

        خودکار it = conv2d_pipelines.find(کلید);
        إذا (it != conv2d_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "conv_2d";

        خودکار افزودن_type_تعریف_میکند = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_پیشوند = prefix;
            إذا (type == GGML_TYPE_F32) {
                تعریف_میکند.push_back(s_پیشوند + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                تعریف_میکند.push_back(s_پیشوند + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        افزودن_type_تعریف_میکند("WEIGHT_TYPE", کلید.وزن_type);
        افزودن_type_تعریف_میکند("INPUT_TYPE", کلید.درون_داد_type);
        افزودن_type_تعریف_میکند("OUTPUT_TYPE", کلید.output_type);

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_conv2d, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[کلید]    = pipeline;
        أرجع conv2d_pipelines[کلید];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي whcn) {
        ggml_webgpu_conv2d_dw_pipeline_کلید کلید = {};
        کلید.وزن_type                        = context.src0->type;
        کلید.درون_داد_type                         = context.src1->type;
        کلید.output_type                        = context.dst->type;
        کلید.whcn                               = whcn;

        خودکار it = conv2d_dw_pipelines.find(کلید);
        إذا (it != conv2d_dw_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        خودکار افزودن_type_تعریف_میکند = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_پیشوند = prefix;
            إذا (type == GGML_TYPE_F32) {
                تعریف_میکند.push_back(s_پیشوند + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                تعریف_میکند.push_back(s_پیشوند + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        افزودن_type_تعریف_میکند("WEIGHT_TYPE", کلید.وزن_type);
        افزودن_type_تعریف_میکند("INPUT_TYPE", کلید.درون_داد_type);
        افزودن_type_تعریف_میکند("OUTPUT_TYPE", کلید.output_type);

        إذا (whcn) {
            تعریف_میکند.push_back("WHCN");
        }
        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_conv2d_dw, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[کلید] = pipeline;
        أرجع conv2d_dw_pipelines[کلید];
    }

    webgpu_pipeline get_im2col_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_کلید کلید = {};
        کلید.درون_داد_type                      = context.src1->type;
        کلید.output_type                     = context.dst->type;

        خودکار it = im2col_pipelines.find(کلید);
        إذا (it != im2col_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "im2col";

        خودکار افزودن_type_تعریف_میکند = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_پیشوند = prefix;
            إذا (type == GGML_TYPE_F32) {
                تعریف_میکند.push_back(s_پیشوند + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                تعریف_میکند.push_back(s_پیشوند + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        افزودن_type_تعریف_میکند("INPUT_TYPE", کلید.درون_داد_type);
        افزودن_type_تعریف_میکند("OUTPUT_TYPE", کلید.output_type);

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_im2col, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[کلید]    = pipeline;
        أرجع im2col_pipelines[کلید];
    }

    webgpu_pipeline get_upscale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت uint32_t نما_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ثابت uint32_t پایه_نما  = نما_flags & 0xFFu;
        ثابت منطقي     antialias  = (نما_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_کلید کلید = {};
        کلید.درون_داد_type                       = context.src0->type;
        کلید.output_type                      = context.dst->type;
        کلید.پایه_نما                        = پایه_نما;
        کلید.antialias                        = antialias;

        خودکار it = upscale_pipelines.find(کلید);
        إذا (it != upscale_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریف_میکند;
        std::string              variant = "upscale";

        إذا (کلید.درون_داد_type == GGML_TYPE_F16) {
            تعریف_میکند.push_back("SRC_F16");
            variant += "_src_f16";
        } وإلا {
            variant += "_src_f32";
        }

        إذا (کلید.output_type == GGML_TYPE_F16) {
            تعریف_میکند.push_back("DST_F16");
            variant += "_dst_f16";
        } وإلا {
            variant += "_dst_f32";
        }

        بدل (پایه_نما) {
            حالة GGML_SCALE_نما_نزدیکترین:
                تعریف_میکند.push_back("NEAREST");
                variant += "_nearest";
                اكسر;
            حالة GGML_SCALE_نما_BILINEAR:
                تعریف_میکند.push_back("BILINEAR");
                variant += "_bilinear";
                اكسر;
            حالة GGML_SCALE_نما_BICUBIC:
                تعریف_میکند.push_back("BICUBIC");
                variant += "_bicubic";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported upscale mode");
        }

        إذا (antialias) {
            تعریف_میکند.push_back("ANTIALIAS");
            variant += "_aa";
        }

        تعریف_میکند.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خودکار processed           = preprocessor.preprocess(wgsl_upscale, تعریف_میکند);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[کلید]   = pipeline;
        أرجع upscale_pipelines[کلید];
    }

  خاص:
    ساكن webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_کد,
                                                       std::string    label) {
        wgpu::Shader_منبع_WGSL shader_منبع;
        shader_منبع.code = shader_کد.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.بعدی_درون_Chain = &shader_منبع;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.مدخل_نقطه_2 = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = اشاره_تهی;  // nullptr means auto layout
        أرجع { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#پایان_اگر  // GGML_WEBGPU_SHADER_LIB_HPP
