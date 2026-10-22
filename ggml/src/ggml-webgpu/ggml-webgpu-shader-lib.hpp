#جيڪڏهن_وصف_نه_ٿيل GGML_WEBGPU_SHADER_LIB_HPP
#وصف_ڪريو GGML_WEBGPU_SHADER_LIB_HPP

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

#وصف_ڪريو GGML_WEBGPU_F16_سائيز_بائيٽون                   2
#وصف_ڪريو GGML_WEBGPU_F32_سائيز_بائيٽون                   4
#وصف_ڪريو GGML_WEBGPU_I32_سائيز_بائيٽون                   4
#وصف_ڪريو GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#وصف_ڪريو GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#وصف_ڪريو GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#وصف_ڪريو GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#وصف_ڪريو GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_سائيز     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#وصف_ڪريو GGML_WEBGPU_KV_SEQ_PAD                       256u

#وصف_ڪريو GGML_WEBGPU_ARGSORT_ملايو_MAX_WG_سائيز 512u

// Matrix multiplication parameters

// Register tiling parameters
#وصف_ڪريو WEBGPU_MUL_MAT_TILE_M           4
#وصف_ڪريو WEBGPU_MUL_MAT_TILE_N           4
#وصف_ڪريو WEBGPU_MUL_MAT_WG_سائيز_M        8
#وصف_ڪريو WEBGPU_MUL_MAT_WG_سائيز_N        8
#وصف_ڪريو WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#وصف_ڪريو WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#وصف_ڪريو WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#وصف_ڪريو WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#وصف_ڪريو WEBGPU_MUL_MAT_SUBGROUP_ميٽرڪس_M     4
#وصف_ڪريو WEBGPU_MUL_MAT_SUBGROUP_ميٽرڪس_N     2
#وصف_ڪريو WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#وصف_ڪريو WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#وصف_ڪريو WEBGPU_MUL_MAT_VEC_WG_سائيز 256

#وصف_ڪريو WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_في_WG    4
#وصف_ڪريو WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_في_WG 4
#وصف_ڪريو WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_في_WG      4

// default size for reg-tile matrix multiplication
#وصف_ڪريو WEBGPU_MUL_MAT_WG_سائيز 256

// Same hash combine function as in boost
سانچو <قسم_نالو T> مضمن فراغ ggml_webgpu_hash_combine(size_t & ٻج, ثابت T & value) {
    ٻج ^= std::hash<T>{}(value) + 0x9e3779b9 + (ٻج << 6) + (ٻج >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
مضمن uintptr_t ggml_webgpu_tensor_addr(ثابت ggml_tensor * tensor) {
    ثابت ggml_tensor * بنياد_tensor = tensor->view_src ? tensor->view_src : tensor;
    أرجع (uintptr_t) بنياد_tensor->معلومات + tensor->view_offs;
}

مضمن منطقي ggml_webgpu_tensor_equal(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    أرجع a->بفر == b->بفر && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
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

    uint32_t    max_wg_سائيز;
    size_t      wg_mem_حد_بندي_بائيٽون       = 0;
    منطقي        supports_subgroups       = خطأ;
    منطقي        supports_subgroup_ميٽرڪس = خطأ;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_سائيز        = 0;
    uint32_t    max_subgroup_سائيز        = 0;
    منطقي        supports_dot_حاصل_ضرب     = خطأ;
    std::string vendor;
};

بنية webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<فراغ> context = خالي_اشارو;
};

بنية ggml_webgpu_generic_shader_decisions {
    uint32_t wg_سائيز = 0;
    منطقي     inplace = خطأ;
};

بنية ggml_webgpu_get_صفون_shader_decisions {
    uint32_t wg_سائيز    = 0;
    منطقي     vectorized = خطأ;
};

بنية ggml_webgpu_ٻٽو_shader_decisions {
    uint32_t wg_سائيز     = 0;
    منطقي     inplace     = خطأ;
    منطقي     overlap     = خطأ;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_glu_shader_decisions {
    uint32_t wg_سائيز     = 0;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<فراغ> decisions;
};

بنية ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t بلاڪ_سائيز_3;
    uint32_t ٽوڪن_في_wg;
};

بنية ggml_webgpu_ssm_جاچ_pipeline_ڪنجي {
    صحيح     type;
    صحيح     d_state;
    uint8_t xbc_overlap;
    منطقي    a_overlap;
    منطقي    ids_overlap;

    منطقي عامل==(ثابت ggml_webgpu_ssm_جاچ_pipeline_ڪنجي & other) ثابت {
        أرجع type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

بنية ggml_webgpu_ssm_جاچ_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_ssm_جاچ_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.d_state);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.xbc_overlap);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.a_overlap);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.ids_overlap);
        أرجع ٻج;
    }
};

بنية ggml_webgpu_ssm_جاچ_shader_decisions {
    uint32_t wg_سائيز;
    uint32_t ٽوڪن_في_tile;
    uint8_t  xbc_overlap = 0;
    منطقي     a_overlap   = خطأ;
    منطقي     ids_overlap = خطأ;
};

/** Argsort **/

بنية ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_سائيز;
    size_t   wg_mem_حد_بندي_بائيٽون;
    int32_t  ترتيب;
};

/** Set Rows **/

بنية ggml_webgpu_سيٽ_صفون_pipeline_ڪنجي {
    صحيح src0_type;
    صحيح dst_type;
    صحيح vec4;
    صحيح i64_idx;
    صحيح pair_بلاڪون;

    منطقي عامل==(ثابت ggml_webgpu_سيٽ_صفون_pipeline_ڪنجي & other) ثابت {
        أرجع src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_بلاڪون == other.pair_بلاڪون;
    }
};

بنية ggml_webgpu_سيٽ_صفون_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_سيٽ_صفون_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src0_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.dst_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.vec4);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.i64_idx);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.pair_بلاڪون);
        أرجع ٻج;
    }
};

بنية ggml_webgpu_سيٽ_صفون_shader_decisions {
    منطقي     vec4;
    منطقي     i64_idx;
    منطقي     pair_بلاڪون;
    uint32_t wg_سائيز;
};

/** Set **/

بنية ggml_webgpu_سيٽ_pipeline_ڪنجي {
    ggml_type type;
    منطقي      inplace;

    منطقي عامل==(ثابت ggml_webgpu_سيٽ_pipeline_ڪنجي & other) ثابت {
        أرجع type == other.type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_سيٽ_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_سيٽ_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.inplace);
        أرجع ٻج;
    }
};

/** Get Rows **/

بنية ggml_webgpu_get_صفون_pipeline_ڪنجي {
    ggml_type src_type;
    صحيح       vectorized;

    منطقي عامل==(ثابت ggml_webgpu_get_صفون_pipeline_ڪنجي & other) ثابت {
        أرجع src_type == other.src_type && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_get_صفون_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_get_صفون_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.vectorized);
        أرجع ٻج;
    }
};

/** Row Norm **/

بنية ggml_webgpu_صف_norm_pipeline_ڪنجي {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    منطقي      inplace;

    منطقي عامل==(ثابت ggml_webgpu_صف_norm_pipeline_ڪنجي & other) ثابت {
        أرجع op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_صف_norm_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_صف_norm_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.op);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.dst_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.inplace);
        أرجع ٻج;
    }
};

/** RMS_NORM + MUL **/

بنية ggml_webgpu_rms_norm_mul_pipeline_ڪنجي {
    منطقي inplace;      // rn_src == dst
    منطقي overlap;      // mul_src == dst
    منطقي src_overlap;  // rn_src binding overlaps mul_src binding

    منطقي عامل==(ثابت ggml_webgpu_rms_norm_mul_pipeline_ڪنجي & other) ثابت {
        أرجع inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_rms_norm_mul_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_rms_norm_mul_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.inplace);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.overlap);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src_overlap);
        أرجع ٻج;
    }
};

بنية ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_سائيز     = 0;
    منطقي     inplace     = خطأ;
    منطقي     overlap     = خطأ;
    منطقي     src_overlap = خطأ;
};

/** Pad **/
بنية ggml_webgpu_pad_pipeline_ڪنجي {
    منطقي circular;

    منطقي عامل==(ثابت ggml_webgpu_pad_pipeline_ڪنجي & other) ثابت { أرجع circular == other.circular; }
};

بنية ggml_webgpu_pad_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_pad_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.circular);
        أرجع ٻج;
    }
};

/** Solve Tri **/
بنية ggml_webgpu_solve_tri_pipeline_ڪنجي {
    صحيح type;
    صحيح n;
    صحيح k;

    منطقي عامل==(ثابت ggml_webgpu_solve_tri_pipeline_ڪنجي & other) ثابت {
        أرجع type == other.type && n == other.n && k == other.k;
    }
};

بنية ggml_webgpu_solve_tri_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_solve_tri_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.n);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.k);
        أرجع ٻج;
    }
};

/** SSM Conv **/
بنية ggml_webgpu_ssm_conv_pipeline_ڪنجي {
    صحيح type;
    صحيح vectorized;

    منطقي عامل==(ثابت ggml_webgpu_ssm_conv_pipeline_ڪنجي & other) ثابت {
        أرجع type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
بنية ggml_webgpu_conv2d_pipeline_ڪنجي {
    ggml_type وزن_type;
    ggml_type داخلا_type;
    ggml_type output_type;

    منطقي عامل==(ثابت ggml_webgpu_conv2d_pipeline_ڪنجي & other) ثابت {
        أرجع وزن_type == other.وزن_type && داخلا_type == other.داخلا_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_conv2d_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_conv2d_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.وزن_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.داخلا_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.output_type);
        أرجع ٻج;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
بنية ggml_webgpu_conv2d_dw_pipeline_ڪنجي {
    ggml_type وزن_type;
    ggml_type داخلا_type;
    ggml_type output_type;
    منطقي      whcn;

    منطقي عامل==(ثابت ggml_webgpu_conv2d_dw_pipeline_ڪنجي & other) ثابت {
        أرجع وزن_type == other.وزن_type && داخلا_type == other.داخلا_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

بنية ggml_webgpu_conv2d_dw_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_conv2d_dw_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.وزن_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.داخلا_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.output_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.whcn);
        أرجع ٻج;
    }
};

/** Im2Col **/
بنية ggml_webgpu_im2col_pipeline_ڪنجي {
    ggml_type داخلا_type;
    ggml_type output_type;

    منطقي عامل==(ثابت ggml_webgpu_im2col_pipeline_ڪنجي & other) ثابت {
        أرجع داخلا_type == other.داخلا_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_im2col_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_im2col_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.داخلا_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.output_type);
        أرجع ٻج;
    }
};

/** Gated Delta Net **/
بنية ggml_webgpu_gated_delta_net_pipeline_ڪنجي {
    صحيح  type;
    صحيح  s_v;
    صحيح  kda;
    منطقي fused_cache;

    منطقي عامل==(ثابت ggml_webgpu_gated_delta_net_pipeline_ڪنجي & other) ثابت {
        أرجع type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

بنية ggml_webgpu_gated_delta_net_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_gated_delta_net_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.s_v);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.kda);
        أرجع ٻج;
    }
};

بنية ggml_webgpu_ssm_conv_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_ssm_conv_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.vectorized);
        أرجع ٻج;
    }
};

/** Scale **/

بنية ggml_webgpu_scale_pipeline_ڪنجي {
    صحيح inplace;

    منطقي عامل==(ثابت ggml_webgpu_scale_pipeline_ڪنجي & other) ثابت { أرجع inplace == other.inplace; }
};

بنية ggml_webgpu_scale_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_scale_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.inplace);
        أرجع ٻج;
    }
};

/** Upscale **/

بنية ggml_webgpu_upscale_pipeline_ڪنجي {
    ggml_type داخلا_type;
    ggml_type output_type;
    uint32_t  بنياد_گهڻائي;
    منطقي      antialias;

    منطقي عامل==(ثابت ggml_webgpu_upscale_pipeline_ڪنجي & other) ثابت {
        أرجع داخلا_type == other.داخلا_type && output_type == other.output_type && بنياد_گهڻائي == other.بنياد_گهڻائي &&
               antialias == other.antialias;
    }
};

بنية ggml_webgpu_upscale_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_upscale_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.داخلا_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.output_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.بنياد_گهڻائي);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.antialias);
        أرجع ٻج;
    }
};

/** Concat **/

بنية ggml_webgpu_concat_pipeline_ڪنجي {
    صحيح  type;
    منطقي src_overlap;

    منطقي عامل==(ثابت ggml_webgpu_concat_pipeline_ڪنجي & other) ثابت {
        أرجع type == other.type && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_concat_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_concat_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src_overlap);
        أرجع ٻج;
    }
};

/** Repeat **/

بنية ggml_webgpu_repeat_pipeline_ڪنجي {
    صحيح type;

    منطقي عامل==(ثابت ggml_webgpu_repeat_pipeline_ڪنجي & other) ثابت { أرجع type == other.type; }
};

بنية ggml_webgpu_repeat_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_repeat_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        أرجع ٻج;
    }
};

/** Binary **/

بنية ggml_webgpu_ٻٽو_pipeline_ڪنجي {
    صحيح  type;
    صحيح  op;
    منطقي inplace;
    منطقي overlap;
    منطقي src_overlap;

    منطقي عامل==(ثابت ggml_webgpu_ٻٽو_pipeline_ڪنجي & other) ثابت {
        أرجع type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_ٻٽو_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_ٻٽو_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.op);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.inplace);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.overlap);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src_overlap);
        أرجع ٻج;
    }
};

/* Add_Id */

بنية ggml_webgpu_جوڙيو_id_pipeline_ڪنجي {
    منطقي inplace;

    منطقي عامل==(ثابت ggml_webgpu_جوڙيو_id_pipeline_ڪنجي & other) ثابت { أرجع inplace == other.inplace; }
};

بنية ggml_webgpu_جوڙيو_id_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_جوڙيو_id_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.inplace);
        أرجع ٻج;
    }
};

/** Unary **/

بنية ggml_webgpu_unary_pipeline_ڪنجي {
    صحيح           type;
    صحيح           op;
    منطقي          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    منطقي          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    منطقي عامل==(ثابت ggml_webgpu_unary_pipeline_ڪنجي & other) ثابت {
        أرجع type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

بنية ggml_webgpu_unary_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_unary_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.op);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.is_unary);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.inplace);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.ttype);
        أرجع ٻج;
    }
};

/** FlashAttention */

بنية ggml_webgpu_flash_attn_common_pipeline_ڪنجي {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  سر_dim_qk;
    uint32_t  سر_dim_v;
    منطقي      k_سڌو;
    منطقي      v_سڌو;
    منطقي      kv_overlap;
    منطقي      has_نقاب;
    منطقي      has_sinks;
    منطقي      استعمال_ڪري_logit_softcap;

    منطقي عامل==(ثابت ggml_webgpu_flash_attn_common_pipeline_ڪنجي & other) ثابت {
        أرجع q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && سر_dim_qk == other.سر_dim_qk && سر_dim_v == other.سر_dim_v &&
               k_سڌو == other.k_سڌو && v_سڌو == other.v_سڌو && kv_overlap == other.kv_overlap &&
               has_نقاب == other.has_نقاب && has_sinks == other.has_sinks &&
               استعمال_ڪري_logit_softcap == other.استعمال_ڪري_logit_softcap;
    }
};

مضمن فراغ ggml_webgpu_flash_attn_hash_common_pipeline_ڪنجي(size_t &                                           ٻج,
                                                            ثابت ggml_webgpu_flash_attn_common_pipeline_ڪنجي & ڪنجي) {
    ggml_webgpu_hash_combine(ٻج, ڪنجي.q_type);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.k_type);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.v_type);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.dst_type);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.سر_dim_qk);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.سر_dim_v);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.k_سڌو);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.v_سڌو);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.kv_overlap);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.has_نقاب);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.has_sinks);
    ggml_webgpu_hash_combine(ٻج, ڪنجي.استعمال_ڪري_logit_softcap);
}

بنية ggml_webgpu_flash_attn_vec_pipeline_ڪنجي {
    ggml_webgpu_flash_attn_common_pipeline_ڪنجي common;

    منطقي عامل==(ثابت ggml_webgpu_flash_attn_vec_pipeline_ڪنجي & other) ثابت { أرجع common == other.common; }
};

بنية ggml_webgpu_flash_attn_vec_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_vec_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ڪنجي(ٻج, ڪنجي.common);
        أرجع ٻج;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_ڪنجي {
    ggml_webgpu_flash_attn_common_pipeline_ڪنجي common;
    منطقي                                       use_sg_ميٽرڪس;

    منطقي عامل==(ثابت ggml_webgpu_flash_attn_pipeline_ڪنجي & other) ثابت {
        أرجع common == other.common && use_sg_ميٽرڪس == other.use_sg_ميٽرڪس;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ڪنجي(ٻج, ڪنجي.common);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.use_sg_ميٽرڪس);
        أرجع ٻج;
    }
};

بنية ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_سائيز = 0;
};

بنية ggml_webgpu_flash_attn_decisions {
    منطقي     use_sg_ميٽرڪس = خطأ;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_سائيز       = 0;
};

مضمن constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ويڪر = 4u;
مضمن constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

مضمن size_t ggml_webgpu_flash_attn_tensor_وٿي(ثابت ggml_tensor * tensor) {
    constexpr uintptr_t ptr_بنياد_addr = 0x1000u;
    ثابت ggml_tensor * بنياد          = tensor->view_src != خالي_اشارو ? tensor->view_src : tensor;
    أرجع reinterpret_cast<uintptr_t>(بنياد->معلومات) - ptr_بنياد_addr + tensor->view_offs;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K, size_t storage_وٿي_alignment) {
    ثابت uint32_t وٿي_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_وٿي(K) & (storage_وٿي_alignment - 1)) /
                    ggml_type_سائيز(K->type));
    أرجع وٿي_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ويڪر == 0u;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K,
                                                      ثابت ggml_tensor * V,
                                                      size_t              storage_وٿي_alignment) {
    أرجع ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_وٿي_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_وٿي_alignment);
}

مضمن منطقي ggml_webgpu_flash_attn_k_سڌو(ثابت ggml_tensor * Q, ثابت ggml_tensor * K, uint32_t kv_سڌو_align) {
    أرجع (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_سڌو_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

مضمن منطقي ggml_webgpu_flash_attn_v_سڌو(ثابت ggml_tensor * Q, ثابت ggml_tensor * V, uint32_t kv_سڌو_align) {
    أرجع ggml_webgpu_flash_attn_k_سڌو(Q, V, kv_سڌو_align);
}

مضمن ggml_webgpu_flash_attn_common_pipeline_ڪنجي ggml_webgpu_flash_attn_make_common_pipeline_ڪنجي(
    ثابت ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_سڌو_align,
    منطقي                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_ڪنجي ڪنجي = {};
    ڪنجي.q_type                                     = context.src0->type;
    ڪنجي.k_type                                     = context.src1->type;
    ڪنجي.v_type                                     = context.src2->type;
    ڪنجي.dst_type                                   = context.dst->type;
    ڪنجي.سر_dim_qk                                = (uint32_t) context.src0->ne[0];
    ڪنجي.سر_dim_v                                 = (uint32_t) context.src2->ne[0];
    ڪنجي.k_سڌو           = ggml_webgpu_flash_attn_k_سڌو(context.src0, context.src1, kv_سڌو_align);
    ڪنجي.v_سڌو           = ggml_webgpu_flash_attn_v_سڌو(context.src0, context.src2, kv_سڌو_align);
    ڪنجي.kv_overlap         = kv_overlap;
    ڪنجي.has_نقاب           = context.src3 != خالي_اشارو;
    ڪنجي.has_sinks          = context.src4 != خالي_اشارو;
    ڪنجي.استعمال_ڪري_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    أرجع ڪنجي;
}

مضمن std::vector<std::string> ggml_webgpu_flash_attn_common_مقرر_ڪري(
    ثابت ggml_webgpu_flash_attn_common_pipeline_ڪنجي & ڪنجي,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_سائيز) {
    std::vector<std::string> مقرر_ڪري;

    بدل (ڪنجي.k_type) {
        حالة GGML_TYPE_F32:
            مقرر_ڪري.push_back("K_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            مقرر_ڪري.push_back("K_F16");
            اكسر;
        حالة GGML_TYPE_Q4_0:
            مقرر_ڪري.push_back("K_Q4_0");
            اكسر;
        حالة GGML_TYPE_Q8_0:
            مقرر_ڪري.push_back("K_Q8_0");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(ڪنجي.k_type);

    بدل (ڪنجي.v_type) {
        حالة GGML_TYPE_F32:
            مقرر_ڪري.push_back("V_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            مقرر_ڪري.push_back("V_F16");
            اكسر;
        حالة GGML_TYPE_Q4_0:
            مقرر_ڪري.push_back("V_Q4_0");
            اكسر;
        حالة GGML_TYPE_Q8_0:
            مقرر_ڪري.push_back("V_Q8_0");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(ڪنجي.v_type);

    بدل (ڪنجي.q_type) {
        حالة GGML_TYPE_F32:
            مقرر_ڪري.push_back("Q_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            مقرر_ڪري.push_back("Q_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(ڪنجي.q_type);

    بدل (ڪنجي.dst_type) {
        حالة GGML_TYPE_F32:
            مقرر_ڪري.push_back("DST_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            مقرر_ڪري.push_back("DST_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(ڪنجي.dst_type);

    إذا (ڪنجي.has_نقاب) {
        مقرر_ڪري.push_back("MASK");
        variant += "_mask";
    }
    إذا (ڪنجي.has_sinks) {
        مقرر_ڪري.push_back("SINKS");
        variant += "_sinks";
    }
    إذا (ڪنجي.استعمال_ڪري_logit_softcap) {
        مقرر_ڪري.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    إذا (ڪنجي.k_سڌو) {
        مقرر_ڪري.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    إذا (ڪنجي.v_سڌو) {
        مقرر_ڪري.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    إذا (ڪنجي.kv_overlap) {
        مقرر_ڪري.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    مقرر_ڪري.push_back(std::string("HEAD_DIM_QK=") + std::to_string(ڪنجي.سر_dim_qk));
    variant += std::string("_hsqk") + std::to_string(ڪنجي.سر_dim_qk);

    مقرر_ڪري.push_back(std::string("HEAD_DIM_V=") + std::to_string(ڪنجي.سر_dim_v));
    variant += std::string("_hsv") + std::to_string(ڪنجي.سر_dim_v);

    مقرر_ڪري.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    مقرر_ڪري.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائيز));

    إذا (ggml_is_quantized(ڪنجي.k_type) || ggml_is_quantized(ڪنجي.v_type)) {
        مقرر_ڪري.push_back("U32_DEQUANT_HELPERS");
        إذا (ggml_is_quantized(ڪنجي.k_type)) {
            مقرر_ڪري.push_back("LOADERS_QUANTIZED_K");
        }
        إذا (ggml_is_quantized(ڪنجي.v_type)) {
            مقرر_ڪري.push_back("LOADERS_QUANTIZED_V");
        }
    }

    أرجع مقرر_ڪري;
}

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_ڪنجي {
    uint32_t  سر_dim_v;
    uint32_t  wg_سائيز;
    ggml_type dst_type;
};

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.سر_dim_v);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.wg_سائيز);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.dst_type);
        أرجع ٻج;
    }
};

مضمن منطقي عامل==(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_ڪنجي & lhs,
                       ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_ڪنجي & rhs) {
    أرجع lhs.سر_dim_v == rhs.سر_dim_v && lhs.wg_سائيز == rhs.wg_سائيز && lhs.dst_type == rhs.dst_type;
}

بنية ggml_webgpu_flash_attn_blk_pipeline_ڪنجي {
    uint32_t kv_tile;

    منطقي عامل==(ثابت ggml_webgpu_flash_attn_blk_pipeline_ڪنجي & other) ثابت { أرجع kv_tile == other.kv_tile; }
};

بنية ggml_webgpu_flash_attn_blk_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_blk_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.kv_tile);
        أرجع ٻج;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
مضمن size_t ggml_webgpu_flash_attn_wg_mem_بائيٽون(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t سر_dim_qk,
                                                  uint32_t سر_dim_v,
                                                  منطقي     has_نقاب,
                                                  منطقي     kv_سڌو) {
    ثابت uint32_t max_سر_dim = std::max(سر_dim_qk, سر_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * سر_dim_qk;        // q_shmem
    إذا (!kv_سڌو) {
        f32_elems += kv_tile * max_سر_dim;  // kv_shmem
    }
    f32_elems += q_tile * سر_dim_v;         // o_shmem
    إذا (has_نقاب) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    أرجع f16_elems * GGML_WEBGPU_F16_سائيز_بائيٽون + f32_elems * GGML_WEBGPU_F32_سائيز_بائيٽون;
}

مضمن uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   حد_بندي_بائيٽون,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t سر_dim_qk,
                                                   uint32_t سر_dim_v,
                                                   منطقي     has_نقاب,
                                                   منطقي     kv_سڌو) {
    ثابت size_t بنياد_q_بائيٽون =
        ggml_webgpu_flash_attn_wg_mem_بائيٽون(q_tile, 0, سر_dim_qk, سر_dim_v, has_نقاب, kv_سڌو);
    إذا (حد_بندي_بائيٽون <= بنياد_q_بائيٽون) {
        أرجع 0;
    }
    ثابت size_t one_kv_بائيٽون =
        ggml_webgpu_flash_attn_wg_mem_بائيٽون(q_tile, 1, سر_dim_qk, سر_dim_v, has_نقاب, kv_سڌو);
    ثابت size_t بائيٽون_في_kv = one_kv_بائيٽون - بنياد_q_بائيٽون;
    إذا (بائيٽون_في_kv == 0) {
        أرجع 0;
    }
    ثابت size_t max_kv_tile = (حد_بندي_بائيٽون - بنياد_q_بائيٽون) / بائيٽون_في_kv;
    أرجع (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

مضمن uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_حد_بندي_بائيٽون,
                                                       uint32_t سر_dim_qk,
                                                       uint32_t سر_dim_v,
                                                       منطقي     has_نقاب,
                                                       منطقي     kv_سڌو) {
    ثابت uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_حد_بندي_بائيٽون, 1u, 1u, سر_dim_qk, سر_dim_v, has_نقاب, kv_سڌو);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    إذا (kv_سڌو) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        طالما (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    أرجع kv_tile;
}

مضمن منطقي ggml_webgpu_flash_attn_can_use_subgroup_ميٽرڪس_رستو(منطقي                supports_subgroup_ميٽرڪس,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ثابت ggml_tensor * Q,
                                                                ثابت ggml_tensor * V) {
    أرجع supports_subgroup_ميٽرڪس && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

بنية ggml_webgpu_mul_mat_vec_pipeline_ڪنجي {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    uint32_t  انگ_کانن_ڳڻپ;
    منطقي      use_mmvq;
    منطقي      src_overlap;

    منطقي عامل==(ثابت ggml_webgpu_mul_mat_vec_pipeline_ڪنجي & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               انگ_کانن_ڳڻپ == other.انگ_کانن_ڳڻپ && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_vec_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_vec_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src0_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src1_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.vectorized);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.انگ_کانن_ڳڻپ);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.use_mmvq);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src_overlap);
        أرجع ٻج;
    }
};

بنية ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_سائيز;
    uint32_t outputs_في_wg;
    uint32_t vec_سائيز;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_quantize_q8_pipeline_ڪنجي {
    ggml_type src0_type;

    منطقي عامل==(ثابت ggml_webgpu_quantize_q8_pipeline_ڪنجي & other) ثابت { أرجع src0_type == other.src0_type; }
};

بنية ggml_webgpu_quantize_q8_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_quantize_q8_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src0_type);
        أرجع ٻج;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_ڪنجي {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    صحيح       use_subgroup_ميٽرڪس;
    منطقي      src_overlap;

    منطقي عامل==(ثابت ggml_webgpu_mul_mat_pipeline_ڪنجي & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_ميٽرڪس == other.use_subgroup_ميٽرڪس && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src0_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src1_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.vectorized);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.use_subgroup_ميٽرڪس);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src_overlap);
        أرجع ٻج;
    }
};

بنية ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_سائيز_m;
    uint32_t wg_سائيز_n;
    uint32_t wg_سائيز;
    uint32_t outputs_في_wg;
    صحيح      use_subgroup_ميٽرڪس;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_ميٽرڪس_m;
    uint32_t subgroup_ميٽرڪس_n;

    uint32_t mul_mat_wg_سائيز;
    منطقي     src_overlap = خطأ;
};

/** MUL_MAT_ID **/

بنية ggml_webgpu_mul_mat_id_gather_pipeline_ڪنجي {
    uint32_t n_experts;

    منطقي عامل==(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_ڪنجي & other) ثابت {
        أرجع n_experts == other.n_experts;
    }
};

بنية ggml_webgpu_mul_mat_id_gather_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.n_experts);
        أرجع ٻج;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_ڪنجي {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  انگ_کانن_ڳڻپ;
    صحيح       vectorized;

    منطقي عامل==(ثابت ggml_webgpu_mul_mat_id_pipeline_ڪنجي & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               انگ_کانن_ڳڻپ == other.انگ_کانن_ڳڻپ && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_id_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src0_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src1_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.n_experts);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.انگ_کانن_ڳڻپ);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.vectorized);
        أرجع ٻج;
    }
};

/** Cpy **/

بنية ggml_webgpu_cpy_pipeline_ڪنجي {
    ggml_type src_type;
    ggml_type dst_type;

    منطقي عامل==(ثابت ggml_webgpu_cpy_pipeline_ڪنجي & other) ثابت {
        أرجع src_type == other.src_type && dst_type == other.dst_type;
    }
};

بنية ggml_webgpu_cpy_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_cpy_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.dst_type);
        أرجع ٻج;
    }
};

/** Glu **/

بنية ggml_webgpu_glu_pipeline_ڪنجي {
    ggml_glu_op glu_op;
    ggml_type   type;
    منطقي        ورهايو;
    منطقي        src_overlap;

    منطقي عامل==(ثابت ggml_webgpu_glu_pipeline_ڪنجي & other) ثابت {
        أرجع glu_op == other.glu_op && type == other.type && ورهايو == other.ورهايو && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_glu_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_glu_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.glu_op);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.ورهايو);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.src_overlap);
        أرجع ٻج;
    }
};

/** Rope **/

بنية ggml_webgpu_rope_pipeline_ڪنجي {
    ggml_type type;
    منطقي      inplace;
    منطقي      has_ff;

    منطقي عامل==(ثابت ggml_webgpu_rope_pipeline_ڪنجي & other) ثابت {
        أرجع type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

بنية ggml_webgpu_rope_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_rope_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.inplace);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.has_ff);
        أرجع ٻج;
    }
};

/** SoftMax **/

بنية ggml_webgpu_soft_max_pipeline_ڪنجي {
    ggml_type نقاب_type;
    منطقي      has_نقاب;
    منطقي      has_sink;
    منطقي      inplace;

    منطقي عامل==(ثابت ggml_webgpu_soft_max_pipeline_ڪنجي & other) ثابت {
        أرجع نقاب_type == other.نقاب_type && has_نقاب == other.has_نقاب && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

بنية ggml_webgpu_soft_max_pipeline_ڪنجي_hash {
    size_t عامل()(ثابت ggml_webgpu_soft_max_pipeline_ڪنجي & ڪنجي) ثابت {
        size_t ٻج = 0;
        ggml_webgpu_hash_combine(ٻج, ڪنجي.نقاب_type);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.has_نقاب);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.has_sink);
        ggml_webgpu_hash_combine(ٻج, ڪنجي.inplace);
        أرجع ٻج;
    }
};

/** MMVQ **/

مضمن منطقي ggml_webgpu_can_use_mmvq(ثابت ggml_tensor * src0,
                                     ثابت ggml_tensor * src1,
                                     منطقي                supports_dot_حاصل_ضرب,
                                     ثابت std::string & vendor) {
    إذا (src1->ne[1] <= 4) {
        منطقي supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        إذا (supports_dp4a && supports_dot_حاصل_ضرب) {
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

    std::unordered_map<صحيح, webgpu_pipeline> جوڙ_صفون_pipelines;       // key is fixed, no variants yet
    std::unordered_map<صحيح, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<صحيح, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<صحيح, webgpu_pipeline> argsort_ملايو_pipelines;  // key is order
    std::unordered_map<صحيح, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_صف_norm_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_صف_norm_pipeline_ڪنجي_hash>
        صف_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_صفون_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_get_صفون_pipeline_ڪنجي_hash>
        get_صفون_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_unary_pipeline_ڪنجي_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_scale_pipeline_ڪنجي_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_ڪنجي_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_ڪنجي_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_جاچ_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_ssm_جاچ_pipeline_ڪنجي_hash>
        ssm_جاچ_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_ڪنجي,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_ڪنجي_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_pad_pipeline_ڪنجي_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_ٻٽو_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_ٻٽو_pipeline_ڪنجي_hash>
        ٻٽو_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_جوڙيو_id_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_جوڙيو_id_pipeline_ڪنجي_hash>
        جوڙيو_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_concat_pipeline_ڪنجي_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_repeat_pipeline_ڪنجي_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_ڪنجي,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_ڪنجي_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_ڪنجي_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_ڪنجي,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_ڪنجي_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_ڪنجي,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_ڪنجي_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_ڪنجي_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_ڪنجي_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_ڪنجي_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_ڪنجي,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_ڪنجي_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_ڪنجي_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_ڪنجي_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_سيٽ_صفون_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_سيٽ_صفون_pipeline_ڪنجي_hash>
        سيٽ_صفون_pipelines;
    std::unordered_map<ggml_webgpu_سيٽ_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_سيٽ_pipeline_ڪنجي_hash> سيٽ_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_cpy_pipeline_ڪنجي_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_glu_pipeline_ڪنجي_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_rope_pipeline_ڪنجي_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_ڪنجي_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_ڪنجي_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_ڪنجي_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_im2col_pipeline_ڪنجي_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_ڪنجي,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_ڪنجي_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_ڪنجي, webgpu_pipeline, ggml_webgpu_upscale_pipeline_ڪنجي_hash>
        upscale_pipelines;

  عام:
    ggml_webgpu_shader_lib(wgpu::Device device) { هذا->device = device; }

    webgpu_pipeline get_جوڙ_صفون_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        خودڪار it = جوڙ_صفون_pipelines.find(1);
        إذا (it != جوڙ_صفون_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> مقرر_ڪري;
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed        = preprocessor.preprocess(wgsl_جوڙ_صفون, مقرر_ڪري);
        جوڙ_صفون_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        أرجع جوڙ_صفون_pipelines[1];
    }

    webgpu_pipeline get_صف_norm_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_صف_norm_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.op                                = context.dst->op;
        ڪنجي.src_type                          = context.src0->type;
        ڪنجي.dst_type                          = context.dst->type;
        ڪنجي.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودڪار it = صف_norm_pipelines.find(ڪنجي);
        إذا (it != صف_norm_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> مقرر_ڪري;
        std::string              variant;

        بدل (ڪنجي.op) {
            حالة GGML_OP_RMS_NORM:
                مقرر_ڪري.push_back("RMS_NORM");
                variant = "rms_norm";
                اكسر;
            حالة GGML_OP_NORM:
                مقرر_ڪري.push_back("NORM");
                variant = "norm";
                اكسر;
            حالة GGML_OP_L2_NORM:
                مقرر_ڪري.push_back("L2_NORM");
                variant = "l2_norm";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        إذا (ڪنجي.inplace) {
            مقرر_ڪري.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (ڪنجي.src_type == GGML_TYPE_F32) {
            مقرر_ڪري.push_back("SRC_F32");
            variant += "_src_f32";
        } وإلا إذا (ڪنجي.src_type == GGML_TYPE_F16) {
            مقرر_ڪري.push_back("SRC_F16");
            variant += "_src_f16";
        }

        إذا (ڪنجي.dst_type == GGML_TYPE_F32) {
            مقرر_ڪري.push_back("DST_F32");
            variant += "_dst_f32";
        } وإلا إذا (ڪنجي.dst_type == GGML_TYPE_F16) {
            مقرر_ڪري.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ثابت uint32_t صف_norm_wg_سائيز = 128u;
        uint32_t       wg_سائيز          = std::min(context.max_wg_سائيز, صف_norm_wg_سائيز);
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائيز));
        خودڪار processed                  = preprocessor.preprocess(wgsl_صف_norm, مقرر_ڪري);
        خودڪار decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز              = wg_سائيز;
        decisions->inplace              = ڪنجي.inplace;
        صف_norm_pipelines[ڪنجي]         = ggml_webgpu_create_pipeline(device, processed, variant);
        صف_norm_pipelines[ڪنجي].context = decisions;
        أرجع صف_norm_pipelines[ڪنجي];
    }

    webgpu_pipeline get_argmax_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي vec4 = context.src0->ne[0] % 4 == 0;

        خودڪار it = argmax_pipelines.find(vec4);
        إذا (it != argmax_pipelines.end()) {
            أرجع it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> مقرر_ڪري;
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));
        إذا (vec4) {
            مقرر_ڪري.push_back("VEC4");
            variant += "_vec4";
        }

        خودڪار processed         = preprocessor.preprocess(wgsl_argmax, مقرر_ڪري);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        أرجع argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_سيٽ_صفون_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_سيٽ_صفون_pipeline_ڪنجي ڪنجي       = {};
        ڪنجي.src0_type                               = context.src0->type;
        ڪنجي.dst_type                                = context.dst->type;
        ڪنجي.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        ڪنجي.i64_idx     = context.src1->type == GGML_TYPE_I64;
        ڪنجي.pair_بلاڪون = quantized && ((context.src0->ne[0] / ggml_blck_سائيز(context.dst->type)) % 2 == 0);

        خودڪار it = سيٽ_صفون_pipelines.find(ڪنجي);
        إذا (it != سيٽ_صفون_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "set_rows";

        بدل (context.dst->type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("DST_F32");
                variant += "_dstf32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("DST_F16");
                variant += "_dstf16";
                اكسر;
            حالة GGML_TYPE_Q8_0:
                مقرر_ڪري.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                اكسر;
            حالة GGML_TYPE_Q4_0:
                مقرر_ڪري.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        إذا (context.src0->type == GGML_TYPE_F16) {
            مقرر_ڪري.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        إذا (ڪنجي.vec4) {
            مقرر_ڪري.push_back("VEC4");
            variant += "_vec4";
        }
        إذا (ڪنجي.i64_idx) {
            مقرر_ڪري.push_back("I64_IDX");
            variant += "_i64idx";
        }
        إذا (ڪنجي.pair_بلاڪون) {
            مقرر_ڪري.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        ثابت خودڪار & shader_ذريعو      = quantized ? wgsl_سيٽ_صفون_quant : wgsl_سيٽ_صفون;
        خودڪار         processed          = preprocessor.preprocess(shader_ذريعو, مقرر_ڪري);
        خودڪار         decisions          = std::make_shared<ggml_webgpu_سيٽ_صفون_shader_decisions>();
        decisions->vec4                 = ڪنجي.vec4;
        decisions->i64_idx              = ڪنجي.i64_idx;
        decisions->pair_بلاڪون          = ڪنجي.pair_بلاڪون;
        decisions->wg_سائيز              = context.max_wg_سائيز;
        سيٽ_صفون_pipelines[ڪنجي]         = ggml_webgpu_create_pipeline(device, processed, variant);
        سيٽ_صفون_pipelines[ڪنجي].context = decisions;
        أرجع سيٽ_صفون_pipelines[ڪنجي];
    }

    webgpu_pipeline get_سيٽ_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_سيٽ_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.type                         = context.dst->type;
        ڪنجي.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودڪار it = سيٽ_pipelines.find(ڪنجي);
        إذا (it != سيٽ_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "set";

        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_ڪري.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for set shader");
        }

        إذا (ڪنجي.inplace) {
            مقرر_ڪري.push_back("INPLACE");
            variant += "_inplace";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_سيٽ, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        decisions->inplace       = ڪنجي.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        سيٽ_pipelines[ڪنجي]       = pipeline;
        أرجع سيٽ_pipelines[ڪنجي];
    }

    webgpu_pipeline get_cumsum_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        خودڪار it = cumsum_pipelines.find(1);
        إذا (it != cumsum_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed      = preprocessor.preprocess(wgsl_cumsum, مقرر_ڪري);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        أرجع cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_چوٽي_k = context.dst->op == GGML_OP_چوٽي_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t ترتيب =
            is_چوٽي_k ? (int32_t) GGML_ترتيب_ڏيو_ترتيب_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        خودڪار it = argsort_pipelines.find(ترتيب);
        إذا (it != argsort_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "argsort";
        مقرر_ڪري.push_back(std::string("ORDER=") + std::to_string(ترتيب));
        variant += std::string("_order") + std::to_string(ترتيب);
        uint32_t wg_سائيز = 1;
        طالما (wg_سائيز * 2 <= context.max_wg_سائيز &&
               wg_سائيز * GGML_WEBGPU_I32_سائيز_بائيٽون <= context.wg_mem_حد_بندي_بائيٽون / 2) {
            wg_سائيز *= 2;
        }
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائيز));
        خودڪار processed                   = preprocessor.preprocess(wgsl_argsort, مقرر_ڪري);
        خودڪار decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز               = wg_سائيز;
        argsort_pipelines[ترتيب]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[ترتيب].context = decisions;
        أرجع argsort_pipelines[ترتيب];
    }

    webgpu_pipeline get_argsort_ملايو_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_چوٽي_k = context.dst->op == GGML_OP_چوٽي_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t ترتيب =
            is_چوٽي_k ? (int32_t) GGML_ترتيب_ڏيو_ترتيب_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        خودڪار it = argsort_ملايو_pipelines.find(ترتيب);
        إذا (it != argsort_ملايو_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "argsort_merge";
        مقرر_ڪري.push_back(std::string("ORDER=") + std::to_string(ترتيب));
        variant += std::string("_order") + std::to_string(ترتيب);
        uint32_t wg_سائيز = std::min(GGML_WEBGPU_ARGSORT_ملايو_MAX_WG_سائيز, context.max_wg_سائيز);
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائيز));

        خودڪار processed                 = preprocessor.preprocess(wgsl_argsort_ملايو, مقرر_ڪري);
        argsort_ملايو_pipelines[ترتيب] = ggml_webgpu_create_pipeline(device, processed, variant);
        أرجع argsort_ملايو_pipelines[ترتيب];
    }

    webgpu_pipeline get_get_صفون_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي vec4_aligned) {
        ثابت منطقي vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_صفون_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.src_type                          = context.src0->type;
        ڪنجي.vectorized                        = (صحيح) vectorized;

        خودڪار it = get_صفون_pipelines.find(ڪنجي);
        إذا (it != get_صفون_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "get_rows";

        ثابت بنية ggml_type_traits * type_traits = ggml_get_type_traits(ڪنجي.src_type);
        ثابت حرف *                    type_str    = type_traits->type_name;

        بدل (ڪنجي.src_type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("FLOAT_PARALLEL");
                إذا (ڪنجي.vectorized) {
                    مقرر_ڪري.push_back("F32_VEC");
                    مقرر_ڪري.push_back("SRC_TYPE=vec4<f32>");
                    مقرر_ڪري.push_back("DST_TYPE=vec4<f32>");
                    مقرر_ڪري.push_back("BLOCK_SIZE=4u");
                } وإلا {
                    مقرر_ڪري.push_back("F32");
                    مقرر_ڪري.push_back("SRC_TYPE=f32");
                    مقرر_ڪري.push_back("DST_TYPE=f32");
                    مقرر_ڪري.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("FLOAT_PARALLEL");
                مقرر_ڪري.push_back("F16");
                مقرر_ڪري.push_back("SRC_TYPE=f16");
                مقرر_ڪري.push_back("DST_TYPE=f32");
                مقرر_ڪري.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_ڪري.push_back("FLOAT_PARALLEL");
                مقرر_ڪري.push_back("I32");
                مقرر_ڪري.push_back("SRC_TYPE=i32");
                مقرر_ڪري.push_back("DST_TYPE=i32");
                مقرر_ڪري.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                اكسر;
            حالة GGML_TYPE_BF16:
                مقرر_ڪري.push_back("BF16");
                مقرر_ڪري.push_back("SRC_TYPE=u32");
                مقرر_ڪري.push_back("DST_TYPE=f32");
                مقرر_ڪري.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_مٿانهون = type_str;
                    std::transform(type_مٿانهون.begin(), type_مٿانهون.end(), type_مٿانهون.begin(), ::وڏو_اکر);

                    بدل (ڪنجي.src_type) {
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
                                مقرر_ڪري.push_back("SRC_TYPE=u32");
                                مقرر_ڪري.push_back("U32_DEQUANT_HELPERS");
                                اكسر;
                            }
                        افتراضي:
                            {
                                مقرر_ڪري.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    مقرر_ڪري.push_back("BYTE_HELPERS");
                    مقرر_ڪري.push_back(type_مٿانهون + "_T");
                    مقرر_ڪري.push_back(type_مٿانهون);
                    مقرر_ڪري.push_back(type_مٿانهون + "_SCALE_MIN");
                    مقرر_ڪري.push_back(type_مٿانهون + "_TABLES");
                    مقرر_ڪري.push_back(type_مٿانهون + "_GRID");
                    مقرر_ڪري.push_back(type_مٿانهون + "_LUT");

                    variant += "_";
                    variant += type_str;

                    مقرر_ڪري.push_back("DST_TYPE=f32");

                    إذا (ڪنجي.src_type == GGML_TYPE_Q1_0) {
                        مقرر_ڪري.push_back("BLOCK_SIZE=128u");
                    } وإلا إذا ((ڪنجي.src_type >= GGML_TYPE_Q4_0 && ڪنجي.src_type <= GGML_TYPE_Q8_1) ||
                               ڪنجي.src_type == GGML_TYPE_IQ4_NL || ڪنجي.src_type == GGML_TYPE_MXFP4) {
                        مقرر_ڪري.push_back("BLOCK_SIZE=32u");
                    } وإلا إذا (ڪنجي.src_type == GGML_TYPE_Q2_0 || ڪنجي.src_type == GGML_TYPE_NVFP4) {
                        مقرر_ڪري.push_back("BLOCK_SIZE=64u");
                    } وإلا إذا (ڪنجي.src_type >= GGML_TYPE_Q2_K) {
                        مقرر_ڪري.push_back("BLOCK_SIZE=256u");
                    } وإلا {
                        مقرر_ڪري.push_back("BLOCK_SIZE=1u");
                    }
                    اكسر;
                }
        }

        إذا (ڪنجي.vectorized) {
            variant += "_vec";
        }

        مقرر_ڪري.push_back("WG_SIZE=" + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_get_صفون, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_get_صفون_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_صفون_pipelines[ڪنجي]  = pipeline;
        أرجع get_صفون_pipelines[ڪنجي];
    }

    webgpu_pipeline get_scale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودڪار it = scale_pipelines.find(ڪنجي);
        إذا (it != scale_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "scale";

        إذا (ڪنجي.inplace) {
            مقرر_ڪري.push_back("INPLACE");
            variant += "_inplace";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_scale, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        decisions->inplace       = ڪنجي.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[ڪنجي]     = pipeline;
        أرجع scale_pipelines[ڪنجي];
    }

    webgpu_pipeline get_solve_tri_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.type                               = context.dst->type;
        ڪنجي.n                                  = (صحيح) context.src0->ne[0];
        ڪنجي.k                                  = (صحيح) context.src1->ne[0];

        خودڪار it = solve_tri_pipelines.find(ڪنجي);
        إذا (it != solve_tri_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "solve_tri";

        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ثابت uint32_t wg_سائيز       = std::min((uint32_t) ڪنجي.n, context.max_wg_سائيز);
        ثابت uint32_t k_tile        = wg_سائيز;
        ثابت uint32_t بائيٽون_في_صف = ((uint32_t) ڪنجي.n + wg_سائيز) * GGML_WEBGPU_F32_سائيز_بائيٽون;
        ثابت uint32_t batch_n       = (uint32_t) (context.wg_mem_حد_بندي_بائيٽون / بائيٽون_في_صف);

        مقرر_ڪري.push_back(std::string("N=") + std::to_string(ڪنجي.n));
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائيز));
        مقرر_ڪري.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        مقرر_ڪري.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        خودڪار processed           = preprocessor.preprocess(wgsl_solve_tri, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = wg_سائيز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[ڪنجي] = pipeline;
        أرجع solve_tri_pipelines[ڪنجي];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.type                              = context.dst->type;
        ڪنجي.vectorized                        = context.src1->ne[0] == 4;

        خودڪار it = ssm_conv_pipelines.find(ڪنجي);
        إذا (it != ssm_conv_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "ssm_conv";

        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        إذا (ڪنجي.vectorized) {
            مقرر_ڪري.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t بلاڪ_سائيز_3    = 32u;
        constexpr uint32_t ٽوڪن_في_wg = 8u;

        مقرر_ڪري.push_back("BLOCK_SIZE=" + std::to_string(بلاڪ_سائيز_3) + "u");
        مقرر_ڪري.push_back("TOKENS_PER_WG=" + std::to_string(ٽوڪن_في_wg) + "u");

        خودڪار processed           = preprocessor.preprocess(wgsl_ssm_conv, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->بلاڪ_سائيز_3    = بلاڪ_سائيز_3;
        decisions->ٽوڪن_في_wg = ٽوڪن_في_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[ڪنجي]  = pipeline;
        أرجع ssm_conv_pipelines[ڪنجي];
    }

    webgpu_pipeline get_ssm_جاچ_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_جاچ_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.type                              = context.dst->type;
        ڪنجي.d_state                           = (صحيح) context.src0->ne[0];
        ڪنجي.xbc_overlap                       = xbc_overlap;

        خودڪار it = ssm_جاچ_pipelines.find(ڪنجي);
        إذا (it != ssm_جاچ_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "ssm_scan";

        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ثابت uint32_t wg_سائيز = (uint32_t) ڪنجي.d_state;

        constexpr uint32_t ٽوڪن_في_tile = 4u;

        مقرر_ڪري.push_back("WG_SIZE=" + std::to_string(wg_سائيز) + "u");
        مقرر_ڪري.push_back("TOKENS_PER_TILE=" + std::to_string(ٽوڪن_في_tile) + "u");

        إذا (context.supports_subgroups) {
            مقرر_ڪري.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } وإلا {
            variant += "_wg_reduce";
        }

        إذا (ڪنجي.xbc_overlap == 0b110) {  // x/B
            مقرر_ڪري.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } وإلا إذا (ڪنجي.xbc_overlap == 0b011) {  // B/C
            مقرر_ڪري.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } وإلا إذا (ڪنجي.xbc_overlap == 0b111) {  // x/B/C
            مقرر_ڪري.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(ڪنجي.d_state);

        خودڪار processed             = preprocessor.preprocess(wgsl_ssm_جاچ, مقرر_ڪري);
        خودڪار decisions             = std::make_shared<ggml_webgpu_ssm_جاچ_shader_decisions>();
        decisions->wg_سائيز         = wg_سائيز;
        decisions->ٽوڪن_في_tile = ٽوڪن_في_tile;
        decisions->xbc_overlap     = ڪنجي.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_جاچ_pipelines[ڪنجي]    = pipeline;
        أرجع ssm_جاچ_pipelines[ڪنجي];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.type                                     = context.dst->type;
        ڪنجي.s_v                                      = (صحيح) context.src2->ne[0];
        ڪنجي.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        ڪنجي.fused_cache                              = context.dst_fuse != خالي_اشارو;

        خودڪار it = gated_delta_net_pipelines.find(ڪنجي);
        إذا (it != gated_delta_net_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "gated_delta_net";

        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        إذا (ڪنجي.kda) {
            مقرر_ڪري.push_back("KDA");
            variant += "_kda";
        }

        إذا (ڪنجي.fused_cache) {
            مقرر_ڪري.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        مقرر_ڪري.push_back("S_V=" + std::to_string(ڪنجي.s_v) + "u");
        مقرر_ڪري.push_back("WG_SIZE=" + std::to_string(ڪنجي.s_v) + "u");

        خودڪار            processed      = preprocessor.preprocess(wgsl_gated_delta_net, مقرر_ڪري);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[ڪنجي] = pipeline;
        أرجع gated_delta_net_pipelines[ڪنجي];
    }

    webgpu_pipeline get_pad_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        خودڪار it = pad_pipelines.find(ڪنجي);
        إذا (it != pad_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "pad";

        إذا (ڪنجي.circular) {
            مقرر_ڪري.push_back("CIRCULAR");
            variant += "_circular";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_pad, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[ڪنجي]       = pipeline;
        أرجع pad_pipelines[ڪنجي];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.src0_type                            = context.src0->type;

        خودڪار it = quantize_q8_pipelines.find(ڪنجي);
        إذا (it != quantize_q8_pipelines.end()) {
            أرجع it->second;
        }
        ثابت حرف *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "quantize_q8";

        uint32_t wg_سائيز = WEBGPU_MUL_MAT_VEC_WG_سائيز;

        مقرر_ڪري.push_back("SRC1_INNER_TYPE=f32");
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائيز));

        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_مٿانهون  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_مٿانهون.begin(), type_مٿانهون.end(), type_مٿانهون.begin(), ::وڏو_اکر);

        مقرر_ڪري.push_back("MUL_ACC_" + type_مٿانهون);
        مقرر_ڪري.push_back("Q8_1_T");

        مقرر_ڪري.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        خودڪار processed             = preprocessor.preprocess(shader_src, مقرر_ڪري);
        خودڪار decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز         = wg_سائيز;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[ڪنجي] = pipeline;
        أرجع quantize_q8_pipelines[ڪنجي];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.src0_type                            = context.src0->type;
        ڪنجي.src1_type                            = context.src1->type;
        ڪنجي.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        ڪنجي.انگ_کانن_ڳڻپ   = context.dst->ne[1];
        ڪنجي.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_حاصل_ضرب, context.vendor);
        ڪنجي.src_overlap = src_overlap;

        خودڪار it = mul_mat_vec_pipelines.find(ڪنجي);
        إذا (it != mul_mat_vec_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant    = "mul_mat_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=f32");
                مقرر_ڪري.push_back("MUL_ACC_FLOAT");
                مقرر_ڪري.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=f16");
                مقرر_ڪري.push_back("MUL_ACC_FLOAT");
                مقرر_ڪري.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=u32");
                مقرر_ڪري.push_back("MUL_ACC_FLOAT");
                مقرر_ڪري.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_مٿانهون  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_مٿانهون.begin(), type_مٿانهون.end(), type_مٿانهون.begin(), ::وڏو_اکر);

                    مقرر_ڪري.push_back("BYTE_HELPERS");
                    مقرر_ڪري.push_back("MUL_ACC_" + type_مٿانهون);
                    مقرر_ڪري.push_back("U32_DEQUANT_HELPERS");
                    مقرر_ڪري.push_back("SRC0_INNER_TYPE=u32");
                    بدل (context.src0->type) {
                        حالة GGML_TYPE_Q4_0:
                        حالة GGML_TYPE_Q4_1:
                        حالة GGML_TYPE_Q5_0:
                        حالة GGML_TYPE_Q5_1:
                        حالة GGML_TYPE_Q8_0:
                            إذا (ڪنجي.use_mmvq) {
                                مقرر_ڪري.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_Q1_0:
                        حالة GGML_TYPE_Q2_K:
                        حالة GGML_TYPE_Q3_K:
                        حالة GGML_TYPE_Q4_K:
                        حالة GGML_TYPE_Q5_K:
                        حالة GGML_TYPE_Q6_K:
                            إذا (ڪنجي.use_mmvq) {
                                مقرر_ڪري.push_back("K_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            مقرر_ڪري.push_back(type_مٿانهون + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ3_XXS:
                            مقرر_ڪري.push_back(type_مٿانهون + "_GRID");
                            مقرر_ڪري.push_back(type_مٿانهون + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                            مقرر_ڪري.push_back(type_مٿانهون + "_LUT");
                            إذا (ڪنجي.use_mmvq) {
                                مقرر_ڪري.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_NVFP4:
                            مقرر_ڪري.push_back(type_مٿانهون + "_LUT");
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
                مقرر_ڪري.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        مقرر_ڪري.push_back(ڪنجي.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_سائيز        = WEBGPU_MUL_MAT_VEC_WG_سائيز;
        uint32_t outputs_في_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_في_WG;

        إذا (ڪنجي.src0_type == GGML_TYPE_Q1_0) {
            outputs_في_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_في_WG;
        } وإلا إذا (ڪنجي.src0_type >= GGML_TYPE_Q2_K) {
            outputs_في_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_في_WG;
        } وإلا إذا (ڪنجي.src0_type >= GGML_TYPE_Q4_0) {
            outputs_في_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_في_WG;
        }

        إذا (ڪنجي.use_mmvq) {
            مقرر_ڪري.push_back("MMVQ");
            مقرر_ڪري.push_back("Q8_1_T");
        }

        إذا (ڪنجي.src_overlap) {
            مقرر_ڪري.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائيز));
        مقرر_ڪري.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_في_wg));
        مقرر_ڪري.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (ڪنجي.vectorized) {
            variant += "_vectorized";
        }
        مقرر_ڪري.push_back(std::string("NUM_COLS=") + std::to_string(ڪنجي.انگ_کانن_ڳڻپ));

        خودڪار processed            = preprocessor.preprocess(shader_src, مقرر_ڪري);
        خودڪار decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_سائيز        = wg_سائيز;
        decisions->outputs_في_wg = outputs_في_wg;
        decisions->vec_سائيز       = ڪنجي.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[ڪنجي] = pipeline;
        أرجع mul_mat_vec_pipelines[ڪنجي];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_mul_mat_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.src0_type                        = context.src0->type;
        ڪنجي.src1_type                        = context.src1->type;
        ڪنجي.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        ڪنجي.use_subgroup_ميٽرڪس = context.supports_subgroup_ميٽرڪس;
        ڪنجي.src_overlap         = src_overlap;

        خودڪار it = mul_mat_fast_pipelines.find(ڪنجي);
        إذا (it != mul_mat_fast_pipelines.end()) {
            أرجع it->second;
        }

        ثابت حرف * shader_src = ڪنجي.use_subgroup_ميٽرڪس ? wgsl_mul_mat_subgroup_ميٽرڪس : wgsl_mul_mat_reg_tile;
        std::vector<std::string> مقرر_ڪري;
        std::string              variant = ڪنجي.use_subgroup_ميٽرڪس ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=f32");
                مقرر_ڪري.push_back("TYPE_F32");
                مقرر_ڪري.push_back("MUL_ACC_FLOAT");
                مقرر_ڪري.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_ڪري.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=f16");
                مقرر_ڪري.push_back("TYPE_F16");
                مقرر_ڪري.push_back("MUL_ACC_FLOAT");
                مقرر_ڪري.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_ڪري.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=u32");
                مقرر_ڪري.push_back("TYPE_BF16");
                مقرر_ڪري.push_back("MUL_ACC_FLOAT");
                مقرر_ڪري.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_ڪري.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_مٿانهون = src0_name;
                    std::transform(type_مٿانهون.begin(), type_مٿانهون.end(), type_مٿانهون.begin(), ::وڏو_اکر);

                    مقرر_ڪري.push_back("BYTE_HELPERS");
                    مقرر_ڪري.push_back("MUL_ACC_" + type_مٿانهون);
                    مقرر_ڪري.push_back("INIT_SRC0_SHMEM_" + type_مٿانهون);
                    مقرر_ڪري.push_back("INIT_SRC1_SHMEM_FLOAT");
                    مقرر_ڪري.push_back("U32_DEQUANT_HELPERS");
                    مقرر_ڪري.push_back("SRC0_INNER_TYPE=u32");

                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            مقرر_ڪري.push_back(type_مٿانهون + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                            مقرر_ڪري.push_back(type_مٿانهون + "_GRID");
                            مقرر_ڪري.push_back(type_مٿانهون + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            مقرر_ڪري.push_back(type_مٿانهون + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }

                    variant += std::string("_") + src0_name;
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        مقرر_ڪري.push_back(ڪنجي.vectorized ? "VEC" : "SCALAR");

        ثابت منطقي is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        إذا (ڪنجي.use_subgroup_ميٽرڪس) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } وإلا {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        مقرر_ڪري.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        مقرر_ڪري.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        إذا (ڪنجي.use_subgroup_ميٽرڪس) {
            مقرر_ڪري.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            مقرر_ڪري.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_سائيز) + "u");
            مقرر_ڪري.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            مقرر_ڪري.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            مقرر_ڪري.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_ميٽرڪس_M) + "u");
            مقرر_ڪري.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_ميٽرڪس_N) + "u");
            مقرر_ڪري.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            مقرر_ڪري.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            مقرر_ڪري.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        إذا (ڪنجي.vectorized) {
            variant += "_vectorized";
        }

        إذا (ڪنجي.src_overlap) {
            مقرر_ڪري.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        إذا (!ڪنجي.use_subgroup_ميٽرڪس) {
            مقرر_ڪري.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_سائيز_M) + "u");
            مقرر_ڪري.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_سائيز_N) + "u");
            مقرر_ڪري.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        خودڪار processed = preprocessor.preprocess(shader_src, مقرر_ڪري);

        خودڪار decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_ميٽرڪس = ڪنجي.use_subgroup_ميٽرڪس;
        إذا (ڪنجي.use_subgroup_ميٽرڪس) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_ميٽرڪس_m = WEBGPU_MUL_MAT_SUBGROUP_ميٽرڪس_M;
            decisions->subgroup_ميٽرڪس_n = WEBGPU_MUL_MAT_SUBGROUP_ميٽرڪس_N;
            decisions->wg_سائيز           = context.max_subgroup_سائيز;
        } وإلا {
            decisions->wg_سائيز_m       = WEBGPU_MUL_MAT_WG_سائيز_M;
            decisions->wg_سائيز_n       = WEBGPU_MUL_MAT_WG_سائيز_N;
            decisions->wg_سائيز         = WEBGPU_MUL_MAT_WG_سائيز_M * WEBGPU_MUL_MAT_WG_سائيز_N;
            decisions->mul_mat_wg_سائيز = WEBGPU_MUL_MAT_WG_سائيز;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[ڪنجي] = pipeline;
        أرجع mul_mat_fast_pipelines[ڪنجي];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.n_experts                                  = (uint32_t) context.src0->ne[2];

        خودڪار it = mul_mat_id_gather_pipelines.find(ڪنجي);
        إذا (it != mul_mat_id_gather_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> مقرر_ڪري;
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز) + "u");
        مقرر_ڪري.push_back(std::string("N_EXPERT=") + std::to_string(ڪنجي.n_experts) + "u");
        مقرر_ڪري.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_سائيز_N) + "u");

        خودڪار processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, مقرر_ڪري);
        خودڪار decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز = context.max_wg_سائيز;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[ڪنجي] = pipeline;
        أرجع pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.src0_type                           = context.src0->type;
        ڪنجي.src1_type                           = context.src1->type;
        ڪنجي.n_experts                           = context.src0->ne[2];
        ڪنجي.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        خودڪار it = mul_mat_id_pipelines.find(ڪنجي);
        إذا (it != mul_mat_id_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "mul_mat_id";
        مقرر_ڪري.push_back("MUL_MAT_ID");

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=f32");
                مقرر_ڪري.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_ڪري.push_back("INIT_SRC1_SHMEM_FLOAT");
                مقرر_ڪري.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=f16");
                مقرر_ڪري.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_ڪري.push_back("INIT_SRC1_SHMEM_FLOAT");
                مقرر_ڪري.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=u32");
                مقرر_ڪري.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_ڪري.push_back("INIT_SRC1_SHMEM_FLOAT");
                مقرر_ڪري.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_مٿانهون = src0_name;
                    std::transform(type_مٿانهون.begin(), type_مٿانهون.end(), type_مٿانهون.begin(), ::وڏو_اکر);

                    مقرر_ڪري.push_back("BYTE_HELPERS");
                    مقرر_ڪري.push_back("INIT_SRC0_SHMEM_" + type_مٿانهون);
                    مقرر_ڪري.push_back("INIT_SRC1_SHMEM_FLOAT");
                    مقرر_ڪري.push_back("U32_DEQUANT_HELPERS");
                    مقرر_ڪري.push_back("SRC0_INNER_TYPE=u32");

                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            مقرر_ڪري.push_back(type_مٿانهون + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                            مقرر_ڪري.push_back(type_مٿانهون + "_GRID");
                            مقرر_ڪري.push_back(type_مٿانهون + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            مقرر_ڪري.push_back(type_مٿانهون + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }

                    variant += std::string("_") + src0_name;
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        مقرر_ڪري.push_back(ڪنجي.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ثابت uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        مقرر_ڪري.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        مقرر_ڪري.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        مقرر_ڪري.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        مقرر_ڪري.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_سائيز_M) + "u");
        مقرر_ڪري.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_سائيز_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        إذا (ڪنجي.vectorized) {
            variant += "_vectorized";
        }

        خودڪار processed = preprocessor.preprocess(wgsl_mul_mat_id, مقرر_ڪري);

        خودڪار decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_سائيز_m = WEBGPU_MUL_MAT_WG_سائيز_M;
        decisions->wg_سائيز_n = WEBGPU_MUL_MAT_WG_سائيز_N;
        decisions->wg_سائيز   = WEBGPU_MUL_MAT_WG_سائيز_M * WEBGPU_MUL_MAT_WG_سائيز_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[ڪنجي] = pipeline;
        أرجع mul_mat_id_pipelines[ڪنجي];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.src0_type                           = context.src0->type;
        ڪنجي.src1_type                           = context.src1->type;
        ڪنجي.n_experts                           = context.src0->ne[2];
        ڪنجي.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        خودڪار it = mul_mat_id_vec_pipelines.find(ڪنجي);
        إذا (it != mul_mat_id_vec_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant    = "mul_mat_id_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=f32");
                مقرر_ڪري.push_back("MUL_ACC_FLOAT");
                مقرر_ڪري.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=f16");
                مقرر_ڪري.push_back("MUL_ACC_FLOAT");
                مقرر_ڪري.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                مقرر_ڪري.push_back("SRC0_INNER_TYPE=u32");
                مقرر_ڪري.push_back("MUL_ACC_FLOAT");
                مقرر_ڪري.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_مٿانهون  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_مٿانهون.begin(), type_مٿانهون.end(), type_مٿانهون.begin(), ::وڏو_اکر);

                    مقرر_ڪري.push_back("BYTE_HELPERS");
                    مقرر_ڪري.push_back("MUL_ACC_" + type_مٿانهون);
                    مقرر_ڪري.push_back("U32_DEQUANT_HELPERS");
                    مقرر_ڪري.push_back("SRC0_INNER_TYPE=u32");
                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            مقرر_ڪري.push_back(type_مٿانهون + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ3_XXS:
                            مقرر_ڪري.push_back(type_مٿانهون + "_GRID");
                            مقرر_ڪري.push_back(type_مٿانهون + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            مقرر_ڪري.push_back(type_مٿانهون + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        مقرر_ڪري.push_back(ڪنجي.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_سائيز        = WEBGPU_MUL_MAT_VEC_WG_سائيز;
        uint32_t outputs_في_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_في_WG;

        إذا (ڪنجي.src0_type == GGML_TYPE_Q1_0) {
            outputs_في_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_في_WG;
        } وإلا إذا (ڪنجي.src0_type >= GGML_TYPE_Q2_K) {
            outputs_في_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_في_WG;
        } وإلا إذا (ڪنجي.src0_type >= GGML_TYPE_Q4_0) {
            outputs_في_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_في_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائيز));
        مقرر_ڪري.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_في_wg));
        مقرر_ڪري.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (ڪنجي.vectorized) {
            variant += "_vectorized";
        }
        مقرر_ڪري.push_back(std::string("NUM_COLS=1"));

        مقرر_ڪري.push_back(std::string("N_EXPERTS=") + std::to_string(ڪنجي.n_experts));

        خودڪار processed = preprocessor.preprocess(shader_src, مقرر_ڪري);

        خودڪار decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_سائيز        = wg_سائيز;
        decisions->outputs_في_wg = outputs_في_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[ڪنجي] = pipeline;
        أرجع mul_mat_id_vec_pipelines[ڪنجي];
    }

    webgpu_pipeline get_unary_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                     is_unary = context.dst->op == GGML_OP_UNARY;
        ثابت صحيح                      op       = is_unary ? (صحيح) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_ڪنجي ڪنجي      = {};
        ڪنجي.type                                = context.dst->type;
        ڪنجي.op                                  = op;
        ڪنجي.is_unary                            = is_unary;
        ڪنجي.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        ڪنجي.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        خودڪار it = unary_pipelines.find(ڪنجي);
        إذا (it != unary_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant =
            ڪنجي.is_unary ? ggml_unary_op_name((ggml_unary_op) ڪنجي.op) : ggml_op_name((ggml_op) ڪنجي.op);
        مقرر_ڪري.push_back(variant);

        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for unary shader");
        }

        إذا (ڪنجي.inplace) {
            مقرر_ڪري.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (op == GGML_OP_TRI) {
            بدل (ڪنجي.ttype) {
                حالة GGML_TRI_TYPE_هيٺاهون:
                    مقرر_ڪري.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    اكسر;
                حالة GGML_TRI_TYPE_هيٺاهون_DIAG:
                    مقرر_ڪري.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    اكسر;
                حالة GGML_TRI_TYPE_مٿانهون:
                    مقرر_ڪري.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    اكسر;
                حالة GGML_TRI_TYPE_مٿانهون_DIAG:
                    مقرر_ڪري.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    اكسر;
                افتراضي:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_unary, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        decisions->inplace       = ڪنجي.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[ڪنجي]     = pipeline;
        أرجع unary_pipelines[ڪنجي];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ڪنجي.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ڪنجي.src_overlap                           = src_overlap;

        خودڪار it = rms_norm_mul_pipelines.find(ڪنجي);
        إذا (it != rms_norm_mul_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        إذا (ڪنجي.inplace) {
            مقرر_ڪري.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (ڪنجي.overlap) {
            مقرر_ڪري.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (ڪنجي.src_overlap) {
            مقرر_ڪري.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, مقرر_ڪري);
        خودڪار pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_سائيز     = context.max_wg_سائيز;
        pipeline_decisions->inplace     = ڪنجي.inplace;
        pipeline_decisions->overlap     = ڪنجي.overlap;
        pipeline_decisions->src_overlap = ڪنجي.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[ڪنجي]     = pipeline;
        أرجع rms_norm_mul_pipelines[ڪنجي];
    }

    webgpu_pipeline get_ٻٽو_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_ٻٽو_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.type                            = context.dst->type;
        ڪنجي.op                              = context.dst->op;
        ڪنجي.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ڪنجي.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ڪنجي.src_overlap                     = src_overlap;

        خودڪار it = ٻٽو_pipelines.find(ڪنجي);
        إذا (it != ٻٽو_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              op_name = ggml_op_name((ggml_op) ڪنجي.op);
        std::string              variant = op_name;

        مقرر_ڪري.push_back(std::string("OP_") + op_name);

        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for binary shader");
        }

        إذا (ڪنجي.inplace) {
            مقرر_ڪري.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (ڪنجي.overlap) {
            مقرر_ڪري.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (ڪنجي.src_overlap) {
            مقرر_ڪري.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed                  = preprocessor.preprocess(wgsl_ٻٽو, مقرر_ڪري);
        خودڪار pipeline_decisions         = std::make_shared<ggml_webgpu_ٻٽو_shader_decisions>();
        pipeline_decisions->wg_سائيز     = context.max_wg_سائيز;
        pipeline_decisions->inplace     = ڪنجي.inplace;
        pipeline_decisions->overlap     = ڪنجي.overlap;
        pipeline_decisions->src_overlap = ڪنجي.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ٻٽو_pipelines[ڪنجي]    = pipeline;
        أرجع ٻٽو_pipelines[ڪنجي];
    }

    webgpu_pipeline get_جوڙيو_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_جوڙيو_id_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودڪار it = جوڙيو_id_pipelines.find(ڪنجي);
        إذا (it != جوڙيو_id_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant    = "add_id";
        ثابت حرف *             shader_src = wgsl_جوڙيو_id;

        إذا (ڪنجي.inplace) {
            مقرر_ڪري.push_back("INPLACE");
            variant += "_inplace";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed              = preprocessor.preprocess(shader_src, مقرر_ڪري);
        خودڪار pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_سائيز = context.max_wg_سائيز;
        pipeline_decisions->inplace = ڪنجي.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        جوڙيو_id_pipelines[ڪنجي]    = pipeline;
        أرجع pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_concat_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.type                            = context.dst->type;
        ڪنجي.src_overlap                     = src_overlap;

        خودڪار it = concat_pipelines.find(ڪنجي);
        إذا (it != concat_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "concat";

        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_ڪري.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for concat shader");
        }

        إذا (ڪنجي.src_overlap) {
            مقرر_ڪري.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_concat, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_ٻٽو_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        decisions->src_overlap   = ڪنجي.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[ڪنجي]    = pipeline;
        أرجع concat_pipelines[ڪنجي];
    }

    webgpu_pipeline get_repeat_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.type                            = context.dst->type;

        خودڪار it = repeat_pipelines.find(ڪنجي);
        إذا (it != repeat_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "repeat";

        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_ڪري.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            حالة GGML_TYPE_I16:
                مقرر_ڪري.push_back("TYPE_I16");
                variant += "_i16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_repeat, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[ڪنجي]    = pipeline;
        أرجع repeat_pipelines[ڪنجي];
    }

    webgpu_pipeline get_flash_attn_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ثابت منطقي can_use_subgroup_ميٽرڪس = ggml_webgpu_flash_attn_can_use_subgroup_ميٽرڪس_رستو(
            context.supports_subgroup_ميٽرڪس, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_ميٽرڪس                    = can_use_subgroup_ميٽرڪس;
        decisions.q_tile = decisions.use_sg_ميٽرڪس ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.common                              = ggml_webgpu_flash_attn_make_common_pipeline_ڪنجي(
            context, decisions.use_sg_ميٽرڪس ? context.sg_mat_k : 1u, kv_overlap);
        ڪنجي.common.k_سڌو &= decisions.use_sg_ميٽرڪس && ڪنجي.common.k_type == GGML_TYPE_F16;
        ڪنجي.common.v_سڌو &= decisions.use_sg_ميٽرڪس && ڪنجي.common.v_type == GGML_TYPE_F16;
        ڪنجي.use_sg_ميٽرڪس = decisions.use_sg_ميٽرڪس;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_حد_بندي_بائيٽون, decisions.q_tile, decisions.use_sg_ميٽرڪس ? context.sg_mat_n : 1u,
            ڪنجي.common.سر_dim_qk, ڪنجي.common.سر_dim_v, ڪنجي.common.has_نقاب,
            ڪنجي.common.k_سڌو || ڪنجي.common.v_سڌو);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        إذا (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_ميٽرڪس);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_ميٽرڪس = خطأ;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            ڪنجي.common.k_سڌو     = خطأ;
            ڪنجي.common.v_سڌو     = خطأ;
            ڪنجي.use_sg_ميٽرڪس       = خطأ;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_حد_بندي_بائيٽون, decisions.q_tile, 1u, ڪنجي.common.سر_dim_qk, ڪنجي.common.سر_dim_v,
                ڪنجي.common.has_نقاب, ڪنجي.common.k_سڌو || ڪنجي.common.v_سڌو);
            GGML_ASSERT(max_kv_tile > 0);
        }

        decisions.kv_tile = decisions.use_sg_ميٽرڪس ?
                                std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES) :
                                std::min(GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE, max_kv_tile);
        decisions.wg_سائيز =
            decisions.use_sg_ميٽرڪس ?
                std::max(context.max_subgroup_سائيز, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_سائيز) :
                std::min(context.max_wg_سائيز, std::max(GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_سائيز,
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * context.max_subgroup_سائيز));

        إذا (ڪنجي.common.k_سڌو || ڪنجي.common.v_سڌو) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            طالما (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_ميٽرڪس ? context.sg_mat_n : context.min_subgroup_سائيز;
            }
        }

        خودڪار it = flash_attn_pipelines.find(ڪنجي);
        إذا (it != flash_attn_pipelines.end()) {
            أرجع it->second;
        }

        std::string              variant = decisions.use_sg_ميٽرڪس ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> مقرر_ڪري = ggml_webgpu_flash_attn_common_مقرر_ڪري(ڪنجي.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_سائيز);
        ثابت حرف *             shader_src = خالي_اشارو;
        إذا (!ڪنجي.use_sg_ميٽرڪس) {
            shader_src = wgsl_flash_attn_tile;
            مقرر_ڪري.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_سائيز) + "u");
            مقرر_ڪري.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_سائيز) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_سائيز) + "_" +
                       std::to_string(context.max_subgroup_سائيز);
        } وإلا {
            shader_src = wgsl_flash_attn;
            مقرر_ڪري.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            مقرر_ڪري.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            مقرر_ڪري.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        خودڪار            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, مقرر_ڪري), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[ڪنجي] = pipeline;
        أرجع flash_attn_pipelines[ڪنجي];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.common = ggml_webgpu_flash_attn_make_common_pipeline_ڪنجي(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ويڪر,
                                                                     kv_overlap);

        خودڪار it = flash_attn_vec_pipelines.find(ڪنجي);
        إذا (it != flash_attn_vec_pipelines.end()) {
            أرجع it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_حد_بندي_بائيٽون, ڪنجي.common.سر_dim_qk,
                                                                   ڪنجي.common.سر_dim_v, ڪنجي.common.has_نقاب,
                                                                   ڪنجي.common.k_سڌو || ڪنجي.common.v_سڌو);
        decisions.wg_سائيز = context.max_subgroup_سائيز;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> مقرر_ڪري =
            ggml_webgpu_flash_attn_common_مقرر_ڪري(ڪنجي.common, variant, 1u, decisions.kv_tile, decisions.wg_سائيز);
        إذا (ڪنجي.common.has_نقاب) {
            مقرر_ڪري.push_back("BLK");
            variant.resize(variant.size() - (حجم("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_ورهايو = context.min_subgroup_سائيز;
        ثابت uint32_t D       = ڪنجي.common.سر_dim_qk | ڪنجي.common.سر_dim_v;
        ثابت uint32_t D_lsb   = D & (~(D - 1u));
        d_ورهايو                = std::min(std::min(context.min_subgroup_سائيز, 4u), std::max(D_lsb / 4u, 1u));

        مقرر_ڪري.push_back(std::string("D_SPLIT=") + std::to_string(d_ورهايو));
        variant += "_dsplit" + std::to_string(d_ورهايو);

        خودڪار            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_ورهايو, مقرر_ڪري), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[ڪنجي] = pipeline;
        أرجع flash_attn_vec_pipelines[ڪنجي];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.kv_tile                                 = kv_tile;
        خودڪار it                                     = flash_attn_blk_pipelines.find(ڪنجي);
        إذا (it != flash_attn_blk_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "flash_attn_vec_blk";

        مقرر_ڪري.push_back(std::string("KV_TILE=") + std::to_string(ڪنجي.kv_tile));
        variant += std::string("_kvt") + std::to_string(ڪنجي.kv_tile);

        uint32_t wg_سائيز = 1;
        طالما ((wg_سائيز << 1) <= context.max_wg_سائيز) {
            wg_سائيز <<= 1;
        }
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائيز));
        variant += std::string("_wg") + std::to_string(wg_سائيز);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, مقرر_ڪري), variant);
        flash_attn_blk_pipelines[ڪنجي] = pipeline;
        أرجع flash_attn_blk_pipelines[ڪنجي];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.سر_dim_v                                     = (uint32_t) context.src2->ne[0];
        ڪنجي.dst_type                                       = context.dst->type;
        ڪنجي.wg_سائيز                                        = context.max_wg_سائيز;
        خودڪار it                                            = flash_attn_vec_reduce_pipelines.find(ڪنجي);
        إذا (it != flash_attn_vec_reduce_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "flash_attn_vec_reduce";

        بدل (ڪنجي.dst_type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("DST_F32");
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("DST_F16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(ڪنجي.dst_type);

        مقرر_ڪري.push_back(std::string("HEAD_DIM_V=") + std::to_string(ڪنجي.سر_dim_v));
        variant += std::string("_hsv") + std::to_string(ڪنجي.سر_dim_v);

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));
        variant += std::string("_wg") + std::to_string(context.max_wg_سائيز);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, مقرر_ڪري), variant);
        flash_attn_vec_reduce_pipelines[ڪنجي] = pipeline;
        أرجع flash_attn_vec_reduce_pipelines[ڪنجي];
    }

    webgpu_pipeline get_cpy_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.src_type                     = context.src0->type;
        ڪنجي.dst_type                     = context.dst->type;

        خودڪار it = cpy_pipelines.find(ڪنجي);
        إذا (it != cpy_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "cpy";

        بدل (ڪنجي.src_type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("SRC_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("SRC_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_ڪري.push_back("SRC_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        بدل (ڪنجي.dst_type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("DST_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("DST_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_ڪري.push_back("DST_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_cpy, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[ڪنجي]       = pipeline;
        أرجع cpy_pipelines[ڪنجي];
    }

    webgpu_pipeline get_glu_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_glu_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.glu_op                       = ggml_get_glu_op(context.dst);
        ڪنجي.type                         = context.dst->type;
        ڪنجي.ورهايو                        = (context.src1 != خالي_اشارو);
        ڪنجي.src_overlap                  = src_overlap;

        خودڪار it = glu_pipelines.find(ڪنجي);
        إذا (it != glu_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "glu";

        بدل (ڪنجي.glu_op) {
            حالة GGML_GLU_OP_REGLU:
                مقرر_ڪري.push_back("OP_REGLU");
                variant += "_reglu";
                اكسر;
            حالة GGML_GLU_OP_GEGLU:
                مقرر_ڪري.push_back("OP_GEGLU");
                variant += "_geglu";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU:
                مقرر_ڪري.push_back("OP_SWIGLU");
                variant += "_swiglu";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU_OAI:
                مقرر_ڪري.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                اكسر;
            حالة GGML_GLU_OP_GEGLU_ERF:
                مقرر_ڪري.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                اكسر;
            حالة GGML_GLU_OP_GEGLU_تيز:
                مقرر_ڪري.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU_CLAMP:
                مقرر_ڪري.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported GLU op");
        }
        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        إذا (ڪنجي.src_overlap) {
            مقرر_ڪري.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } وإلا إذا (ڪنجي.ورهايو) {
            variant += "_split";
        } وإلا {
            مقرر_ڪري.push_back("NO_SPLIT");
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_glu, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        decisions->src_overlap   = ڪنجي.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[ڪنجي]       = pipeline;
        أرجع glu_pipelines[ڪنجي];
    }

    webgpu_pipeline get_rope_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.type                          = context.dst->type;
        ڪنجي.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ڪنجي.has_ff                        = (context.src2 != خالي_اشارو);

        خودڪار it = rope_pipelines.find(ڪنجي);
        إذا (it != rope_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "rope";

        بدل (ڪنجي.type) {
            حالة GGML_TYPE_F32:
                مقرر_ڪري.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_ڪري.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        إذا (ڪنجي.inplace) {
            مقرر_ڪري.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (ڪنجي.has_ff) {
            مقرر_ڪري.push_back("FF_FUNC");
            variant += "_ff";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_rope, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        decisions->inplace       = ڪنجي.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[ڪنجي]      = pipeline;
        أرجع rope_pipelines[ڪنجي];
    }

    webgpu_pipeline get_soft_max_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.نقاب_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        ڪنجي.has_نقاب                          = (context.src1 != خالي_اشارو);
        ڪنجي.has_sink                          = (context.src2 != خالي_اشارو);
        ڪنجي.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودڪار it = soft_max_pipelines.find(ڪنجي);
        إذا (it != soft_max_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "soft_max";

        إذا (ڪنجي.has_نقاب) {
            مقرر_ڪري.push_back("HAS_MASK");
            بدل (ڪنجي.نقاب_type) {
                حالة GGML_TYPE_F32:
                    مقرر_ڪري.push_back("MASK_F32");
                    variant += "_mask_f32";
                    اكسر;
                حالة GGML_TYPE_F16:
                    مقرر_ڪري.push_back("MASK_F16");
                    variant += "_mask_f16";
                    اكسر;
                افتراضي:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        إذا (ڪنجي.has_sink) {
            مقرر_ڪري.push_back("HAS_SINK");
            variant += "_sink";
        }

        إذا (ڪنجي.inplace) {
            مقرر_ڪري.push_back("INPLACE");
            variant += "_inplace";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_soft_max, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        decisions->inplace       = ڪنجي.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[ڪنجي]  = pipeline;
        أرجع soft_max_pipelines[ڪنجي];
    }

    webgpu_pipeline get_conv2d_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.وزن_type                     = context.src0->type;
        ڪنجي.داخلا_type                      = context.src1->type;
        ڪنجي.output_type                     = context.dst->type;

        خودڪار it = conv2d_pipelines.find(ڪنجي);
        إذا (it != conv2d_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "conv_2d";

        خودڪار ڌڪو_type_مقرر_ڪري = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_اڳياڙي = prefix;
            إذا (type == GGML_TYPE_F32) {
                مقرر_ڪري.push_back(s_اڳياڙي + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                مقرر_ڪري.push_back(s_اڳياڙي + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ڌڪو_type_مقرر_ڪري("WEIGHT_TYPE", ڪنجي.وزن_type);
        ڌڪو_type_مقرر_ڪري("INPUT_TYPE", ڪنجي.داخلا_type);
        ڌڪو_type_مقرر_ڪري("OUTPUT_TYPE", ڪنجي.output_type);

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_conv2d, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[ڪنجي]    = pipeline;
        أرجع conv2d_pipelines[ڪنجي];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي whcn) {
        ggml_webgpu_conv2d_dw_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.وزن_type                        = context.src0->type;
        ڪنجي.داخلا_type                         = context.src1->type;
        ڪنجي.output_type                        = context.dst->type;
        ڪنجي.whcn                               = whcn;

        خودڪار it = conv2d_dw_pipelines.find(ڪنجي);
        إذا (it != conv2d_dw_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        خودڪار ڌڪو_type_مقرر_ڪري = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_اڳياڙي = prefix;
            إذا (type == GGML_TYPE_F32) {
                مقرر_ڪري.push_back(s_اڳياڙي + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                مقرر_ڪري.push_back(s_اڳياڙي + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ڌڪو_type_مقرر_ڪري("WEIGHT_TYPE", ڪنجي.وزن_type);
        ڌڪو_type_مقرر_ڪري("INPUT_TYPE", ڪنجي.داخلا_type);
        ڌڪو_type_مقرر_ڪري("OUTPUT_TYPE", ڪنجي.output_type);

        إذا (whcn) {
            مقرر_ڪري.push_back("WHCN");
        }
        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_conv2d_dw, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[ڪنجي] = pipeline;
        أرجع conv2d_dw_pipelines[ڪنجي];
    }

    webgpu_pipeline get_im2col_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.داخلا_type                      = context.src1->type;
        ڪنجي.output_type                     = context.dst->type;

        خودڪار it = im2col_pipelines.find(ڪنجي);
        إذا (it != im2col_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "im2col";

        خودڪار ڌڪو_type_مقرر_ڪري = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_اڳياڙي = prefix;
            إذا (type == GGML_TYPE_F32) {
                مقرر_ڪري.push_back(s_اڳياڙي + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                مقرر_ڪري.push_back(s_اڳياڙي + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ڌڪو_type_مقرر_ڪري("INPUT_TYPE", ڪنجي.داخلا_type);
        ڌڪو_type_مقرر_ڪري("OUTPUT_TYPE", ڪنجي.output_type);

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_im2col, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[ڪنجي]    = pipeline;
        أرجع im2col_pipelines[ڪنجي];
    }

    webgpu_pipeline get_upscale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت uint32_t گهڻائي_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ثابت uint32_t بنياد_گهڻائي  = گهڻائي_flags & 0xFFu;
        ثابت منطقي     antialias  = (گهڻائي_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_ڪنجي ڪنجي = {};
        ڪنجي.داخلا_type                       = context.src0->type;
        ڪنجي.output_type                      = context.dst->type;
        ڪنجي.بنياد_گهڻائي                        = بنياد_گهڻائي;
        ڪنجي.antialias                        = antialias;

        خودڪار it = upscale_pipelines.find(ڪنجي);
        إذا (it != upscale_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_ڪري;
        std::string              variant = "upscale";

        إذا (ڪنجي.داخلا_type == GGML_TYPE_F16) {
            مقرر_ڪري.push_back("SRC_F16");
            variant += "_src_f16";
        } وإلا {
            variant += "_src_f32";
        }

        إذا (ڪنجي.output_type == GGML_TYPE_F16) {
            مقرر_ڪري.push_back("DST_F16");
            variant += "_dst_f16";
        } وإلا {
            variant += "_dst_f32";
        }

        بدل (بنياد_گهڻائي) {
            حالة GGML_SCALE_گهڻائي_سڀ_کان_ويجهو:
                مقرر_ڪري.push_back("NEAREST");
                variant += "_nearest";
                اكسر;
            حالة GGML_SCALE_گهڻائي_BILINEAR:
                مقرر_ڪري.push_back("BILINEAR");
                variant += "_bilinear";
                اكسر;
            حالة GGML_SCALE_گهڻائي_BICUBIC:
                مقرر_ڪري.push_back("BICUBIC");
                variant += "_bicubic";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported upscale mode");
        }

        إذا (antialias) {
            مقرر_ڪري.push_back("ANTIALIAS");
            variant += "_aa";
        }

        مقرر_ڪري.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائيز));

        خودڪار processed           = preprocessor.preprocess(wgsl_upscale, مقرر_ڪري);
        خودڪار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائيز       = context.max_wg_سائيز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[ڪنجي]   = pipeline;
        أرجع upscale_pipelines[ڪنجي];
    }

  خاص:
    ساكن webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_ڪوڊ,
                                                       std::string    label) {
        wgpu::Shader_ذريعو_WGSL shader_ذريعو;
        shader_ذريعو.code = shader_ڪوڊ.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.ايندڙ_اندريون_Chain = &shader_ذريعو;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.داخلا_نقطو_2 = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = خالي_اشارو;  // nullptr means auto layout
        أرجع { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#جيڪڏهن_پورو  // GGML_WEBGPU_SHADER_LIB_HPP
