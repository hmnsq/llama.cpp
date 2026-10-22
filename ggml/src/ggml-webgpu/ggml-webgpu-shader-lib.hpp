#که_نه_تعریف_شوی GGML_WEBGPU_SHADER_LIB_HPP
#تعریف_کړه GGML_WEBGPU_SHADER_LIB_HPP

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

#تعریف_کړه GGML_WEBGPU_F16_اندازه_بایټونه                   2
#تعریف_کړه GGML_WEBGPU_F32_اندازه_بایټونه                   4
#تعریف_کړه GGML_WEBGPU_I32_اندازه_بایټونه                   4
#تعریف_کړه GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#تعریف_کړه GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#تعریف_کړه GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#تعریف_کړه GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#تعریف_کړه GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_اندازه     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#تعریف_کړه GGML_WEBGPU_KV_SEQ_PAD                       256u

#تعریف_کړه GGML_WEBGPU_ARGSORT_یوځای_MAX_WG_اندازه 512u

// Matrix multiplication parameters

// Register tiling parameters
#تعریف_کړه WEBGPU_MUL_MAT_TILE_M           4
#تعریف_کړه WEBGPU_MUL_MAT_TILE_N           4
#تعریف_کړه WEBGPU_MUL_MAT_WG_اندازه_M        8
#تعریف_کړه WEBGPU_MUL_MAT_WG_اندازه_N        8
#تعریف_کړه WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#تعریف_کړه WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#تعریف_کړه WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#تعریف_کړه WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#تعریف_کړه WEBGPU_MUL_MAT_SUBGROUP_ماتریکس_M     4
#تعریف_کړه WEBGPU_MUL_MAT_SUBGROUP_ماتریکس_N     2
#تعریف_کړه WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#تعریف_کړه WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#تعریف_کړه WEBGPU_MUL_MAT_VEC_WG_اندازه 256

#تعریف_کړه WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_فی_WG    4
#تعریف_کړه WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG 4
#تعریف_کړه WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_فی_WG      4

// default size for reg-tile matrix multiplication
#تعریف_کړه WEBGPU_MUL_MAT_WG_اندازه 256

// Same hash combine function as in boost
کينډۍ <ډول_نوم T> مضمن فراغ ggml_webgpu_hash_combine(size_t & تخم, ثابت T & value) {
    تخم ^= std::hash<T>{}(value) + 0x9e3779b9 + (تخم << 6) + (تخم >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
مضمن uintptr_t ggml_webgpu_tensor_addr(ثابت ggml_tensor * tensor) {
    ثابت ggml_tensor * بنسټ_tensor = tensor->view_src ? tensor->view_src : tensor;
    أرجع (uintptr_t) بنسټ_tensor->معلومات + tensor->view_offs;
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

    uint32_t    max_wg_اندازه;
    size_t      wg_mem_حد_بایټونه       = 0;
    منطقي        supports_subgroups       = خطأ;
    منطقي        supports_subgroup_ماتریکس = خطأ;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_اندازه        = 0;
    uint32_t    max_subgroup_اندازه        = 0;
    منطقي        supports_dot_حاصل_ضرب     = خطأ;
    std::string vendor;
};

بنية webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<فراغ> context = تش_اشاره;
};

بنية ggml_webgpu_generic_shader_decisions {
    uint32_t wg_اندازه = 0;
    منطقي     inplace = خطأ;
};

بنية ggml_webgpu_get_لیکې_shader_decisions {
    uint32_t wg_اندازه    = 0;
    منطقي     vectorized = خطأ;
};

بنية ggml_webgpu_دوګونی_shader_decisions {
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
    uint32_t بلاک_اندازه_3;
    uint32_t نښې_فی_wg;
};

بنية ggml_webgpu_ssm_لټون_pipeline_کلی {
    صحيح     type;
    صحيح     d_state;
    uint8_t xbc_overlap;
    منطقي    a_overlap;
    منطقي    ids_overlap;

    منطقي چلوونکی==(ثابت ggml_webgpu_ssm_لټون_pipeline_کلی & other) ثابت {
        أرجع type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

بنية ggml_webgpu_ssm_لټون_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_ssm_لټون_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.type);
        ggml_webgpu_hash_combine(تخم, کلی.d_state);
        ggml_webgpu_hash_combine(تخم, کلی.xbc_overlap);
        ggml_webgpu_hash_combine(تخم, کلی.a_overlap);
        ggml_webgpu_hash_combine(تخم, کلی.ids_overlap);
        أرجع تخم;
    }
};

بنية ggml_webgpu_ssm_لټون_shader_decisions {
    uint32_t wg_اندازه;
    uint32_t نښې_فی_tile;
    uint8_t  xbc_overlap = 0;
    منطقي     a_overlap   = خطأ;
    منطقي     ids_overlap = خطأ;
};

/** Argsort **/

بنية ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_اندازه;
    size_t   wg_mem_حد_بایټونه;
    int32_t  ترتیب;
};

/** Set Rows **/

بنية ggml_webgpu_سېټ_لیکې_pipeline_کلی {
    صحيح src0_type;
    صحيح dst_type;
    صحيح vec4;
    صحيح i64_idx;
    صحيح pair_بلاکونه;

    منطقي چلوونکی==(ثابت ggml_webgpu_سېټ_لیکې_pipeline_کلی & other) ثابت {
        أرجع src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_بلاکونه == other.pair_بلاکونه;
    }
};

بنية ggml_webgpu_سېټ_لیکې_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_سېټ_لیکې_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.src0_type);
        ggml_webgpu_hash_combine(تخم, کلی.dst_type);
        ggml_webgpu_hash_combine(تخم, کلی.vec4);
        ggml_webgpu_hash_combine(تخم, کلی.i64_idx);
        ggml_webgpu_hash_combine(تخم, کلی.pair_بلاکونه);
        أرجع تخم;
    }
};

بنية ggml_webgpu_سېټ_لیکې_shader_decisions {
    منطقي     vec4;
    منطقي     i64_idx;
    منطقي     pair_بلاکونه;
    uint32_t wg_اندازه;
};

/** Set **/

بنية ggml_webgpu_سېټ_pipeline_کلی {
    ggml_type type;
    منطقي      inplace;

    منطقي چلوونکی==(ثابت ggml_webgpu_سېټ_pipeline_کلی & other) ثابت {
        أرجع type == other.type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_سېټ_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_سېټ_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.type);
        ggml_webgpu_hash_combine(تخم, کلی.inplace);
        أرجع تخم;
    }
};

/** Get Rows **/

بنية ggml_webgpu_get_لیکې_pipeline_کلی {
    ggml_type src_type;
    صحيح       vectorized;

    منطقي چلوونکی==(ثابت ggml_webgpu_get_لیکې_pipeline_کلی & other) ثابت {
        أرجع src_type == other.src_type && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_get_لیکې_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_get_لیکې_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.src_type);
        ggml_webgpu_hash_combine(تخم, کلی.vectorized);
        أرجع تخم;
    }
};

/** Row Norm **/

بنية ggml_webgpu_لیکه_norm_pipeline_کلی {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    منطقي      inplace;

    منطقي چلوونکی==(ثابت ggml_webgpu_لیکه_norm_pipeline_کلی & other) ثابت {
        أرجع op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_لیکه_norm_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_لیکه_norm_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.op);
        ggml_webgpu_hash_combine(تخم, کلی.src_type);
        ggml_webgpu_hash_combine(تخم, کلی.dst_type);
        ggml_webgpu_hash_combine(تخم, کلی.inplace);
        أرجع تخم;
    }
};

/** RMS_NORM + MUL **/

بنية ggml_webgpu_rms_norm_mul_pipeline_کلی {
    منطقي inplace;      // rn_src == dst
    منطقي overlap;      // mul_src == dst
    منطقي src_overlap;  // rn_src binding overlaps mul_src binding

    منطقي چلوونکی==(ثابت ggml_webgpu_rms_norm_mul_pipeline_کلی & other) ثابت {
        أرجع inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_rms_norm_mul_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_rms_norm_mul_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.inplace);
        ggml_webgpu_hash_combine(تخم, کلی.overlap);
        ggml_webgpu_hash_combine(تخم, کلی.src_overlap);
        أرجع تخم;
    }
};

بنية ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_اندازه     = 0;
    منطقي     inplace     = خطأ;
    منطقي     overlap     = خطأ;
    منطقي     src_overlap = خطأ;
};

/** Pad **/
بنية ggml_webgpu_pad_pipeline_کلی {
    منطقي circular;

    منطقي چلوونکی==(ثابت ggml_webgpu_pad_pipeline_کلی & other) ثابت { أرجع circular == other.circular; }
};

بنية ggml_webgpu_pad_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_pad_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.circular);
        أرجع تخم;
    }
};

/** Solve Tri **/
بنية ggml_webgpu_solve_tri_pipeline_کلی {
    صحيح type;
    صحيح n;
    صحيح k;

    منطقي چلوونکی==(ثابت ggml_webgpu_solve_tri_pipeline_کلی & other) ثابت {
        أرجع type == other.type && n == other.n && k == other.k;
    }
};

بنية ggml_webgpu_solve_tri_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_solve_tri_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.type);
        ggml_webgpu_hash_combine(تخم, کلی.n);
        ggml_webgpu_hash_combine(تخم, کلی.k);
        أرجع تخم;
    }
};

/** SSM Conv **/
بنية ggml_webgpu_ssm_conv_pipeline_کلی {
    صحيح type;
    صحيح vectorized;

    منطقي چلوونکی==(ثابت ggml_webgpu_ssm_conv_pipeline_کلی & other) ثابت {
        أرجع type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
بنية ggml_webgpu_conv2d_pipeline_کلی {
    ggml_type وزن_type;
    ggml_type ننوت_type;
    ggml_type output_type;

    منطقي چلوونکی==(ثابت ggml_webgpu_conv2d_pipeline_کلی & other) ثابت {
        أرجع وزن_type == other.وزن_type && ننوت_type == other.ننوت_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_conv2d_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_conv2d_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.وزن_type);
        ggml_webgpu_hash_combine(تخم, کلی.ننوت_type);
        ggml_webgpu_hash_combine(تخم, کلی.output_type);
        أرجع تخم;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
بنية ggml_webgpu_conv2d_dw_pipeline_کلی {
    ggml_type وزن_type;
    ggml_type ننوت_type;
    ggml_type output_type;
    منطقي      whcn;

    منطقي چلوونکی==(ثابت ggml_webgpu_conv2d_dw_pipeline_کلی & other) ثابت {
        أرجع وزن_type == other.وزن_type && ننوت_type == other.ننوت_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

بنية ggml_webgpu_conv2d_dw_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_conv2d_dw_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.وزن_type);
        ggml_webgpu_hash_combine(تخم, کلی.ننوت_type);
        ggml_webgpu_hash_combine(تخم, کلی.output_type);
        ggml_webgpu_hash_combine(تخم, کلی.whcn);
        أرجع تخم;
    }
};

/** Im2Col **/
بنية ggml_webgpu_im2col_pipeline_کلی {
    ggml_type ننوت_type;
    ggml_type output_type;

    منطقي چلوونکی==(ثابت ggml_webgpu_im2col_pipeline_کلی & other) ثابت {
        أرجع ننوت_type == other.ننوت_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_im2col_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_im2col_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.ننوت_type);
        ggml_webgpu_hash_combine(تخم, کلی.output_type);
        أرجع تخم;
    }
};

/** Gated Delta Net **/
بنية ggml_webgpu_gated_delta_net_pipeline_کلی {
    صحيح  type;
    صحيح  s_v;
    صحيح  kda;
    منطقي fused_cache;

    منطقي چلوونکی==(ثابت ggml_webgpu_gated_delta_net_pipeline_کلی & other) ثابت {
        أرجع type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

بنية ggml_webgpu_gated_delta_net_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_gated_delta_net_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.type);
        ggml_webgpu_hash_combine(تخم, کلی.s_v);
        ggml_webgpu_hash_combine(تخم, کلی.kda);
        أرجع تخم;
    }
};

بنية ggml_webgpu_ssm_conv_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_ssm_conv_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.type);
        ggml_webgpu_hash_combine(تخم, کلی.vectorized);
        أرجع تخم;
    }
};

/** Scale **/

بنية ggml_webgpu_scale_pipeline_کلی {
    صحيح inplace;

    منطقي چلوونکی==(ثابت ggml_webgpu_scale_pipeline_کلی & other) ثابت { أرجع inplace == other.inplace; }
};

بنية ggml_webgpu_scale_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_scale_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.inplace);
        أرجع تخم;
    }
};

/** Upscale **/

بنية ggml_webgpu_upscale_pipeline_کلی {
    ggml_type ننوت_type;
    ggml_type output_type;
    uint32_t  بنسټ_ډېرپېښ;
    منطقي      antialias;

    منطقي چلوونکی==(ثابت ggml_webgpu_upscale_pipeline_کلی & other) ثابت {
        أرجع ننوت_type == other.ننوت_type && output_type == other.output_type && بنسټ_ډېرپېښ == other.بنسټ_ډېرپېښ &&
               antialias == other.antialias;
    }
};

بنية ggml_webgpu_upscale_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_upscale_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.ننوت_type);
        ggml_webgpu_hash_combine(تخم, کلی.output_type);
        ggml_webgpu_hash_combine(تخم, کلی.بنسټ_ډېرپېښ);
        ggml_webgpu_hash_combine(تخم, کلی.antialias);
        أرجع تخم;
    }
};

/** Concat **/

بنية ggml_webgpu_concat_pipeline_کلی {
    صحيح  type;
    منطقي src_overlap;

    منطقي چلوونکی==(ثابت ggml_webgpu_concat_pipeline_کلی & other) ثابت {
        أرجع type == other.type && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_concat_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_concat_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.type);
        ggml_webgpu_hash_combine(تخم, کلی.src_overlap);
        أرجع تخم;
    }
};

/** Repeat **/

بنية ggml_webgpu_repeat_pipeline_کلی {
    صحيح type;

    منطقي چلوونکی==(ثابت ggml_webgpu_repeat_pipeline_کلی & other) ثابت { أرجع type == other.type; }
};

بنية ggml_webgpu_repeat_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_repeat_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.type);
        أرجع تخم;
    }
};

/** Binary **/

بنية ggml_webgpu_دوګونی_pipeline_کلی {
    صحيح  type;
    صحيح  op;
    منطقي inplace;
    منطقي overlap;
    منطقي src_overlap;

    منطقي چلوونکی==(ثابت ggml_webgpu_دوګونی_pipeline_کلی & other) ثابت {
        أرجع type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_دوګونی_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_دوګونی_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.type);
        ggml_webgpu_hash_combine(تخم, کلی.op);
        ggml_webgpu_hash_combine(تخم, کلی.inplace);
        ggml_webgpu_hash_combine(تخم, کلی.overlap);
        ggml_webgpu_hash_combine(تخم, کلی.src_overlap);
        أرجع تخم;
    }
};

/* Add_Id */

بنية ggml_webgpu_جمع_کړه_id_pipeline_کلی {
    منطقي inplace;

    منطقي چلوونکی==(ثابت ggml_webgpu_جمع_کړه_id_pipeline_کلی & other) ثابت { أرجع inplace == other.inplace; }
};

بنية ggml_webgpu_جمع_کړه_id_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_جمع_کړه_id_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.inplace);
        أرجع تخم;
    }
};

/** Unary **/

بنية ggml_webgpu_unary_pipeline_کلی {
    صحيح           type;
    صحيح           op;
    منطقي          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    منطقي          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    منطقي چلوونکی==(ثابت ggml_webgpu_unary_pipeline_کلی & other) ثابت {
        أرجع type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

بنية ggml_webgpu_unary_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_unary_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.type);
        ggml_webgpu_hash_combine(تخم, کلی.op);
        ggml_webgpu_hash_combine(تخم, کلی.is_unary);
        ggml_webgpu_hash_combine(تخم, کلی.inplace);
        ggml_webgpu_hash_combine(تخم, کلی.ttype);
        أرجع تخم;
    }
};

/** FlashAttention */

بنية ggml_webgpu_flash_attn_common_pipeline_کلی {
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
    منطقي      کاروي_logit_softcap;

    منطقي چلوونکی==(ثابت ggml_webgpu_flash_attn_common_pipeline_کلی & other) ثابت {
        أرجع q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && سر_dim_qk == other.سر_dim_qk && سر_dim_v == other.سر_dim_v &&
               k_مستقیم == other.k_مستقیم && v_مستقیم == other.v_مستقیم && kv_overlap == other.kv_overlap &&
               has_ماسک == other.has_ماسک && has_sinks == other.has_sinks &&
               کاروي_logit_softcap == other.کاروي_logit_softcap;
    }
};

مضمن فراغ ggml_webgpu_flash_attn_hash_common_pipeline_کلی(size_t &                                           تخم,
                                                            ثابت ggml_webgpu_flash_attn_common_pipeline_کلی & کلی) {
    ggml_webgpu_hash_combine(تخم, کلی.q_type);
    ggml_webgpu_hash_combine(تخم, کلی.k_type);
    ggml_webgpu_hash_combine(تخم, کلی.v_type);
    ggml_webgpu_hash_combine(تخم, کلی.dst_type);
    ggml_webgpu_hash_combine(تخم, کلی.سر_dim_qk);
    ggml_webgpu_hash_combine(تخم, کلی.سر_dim_v);
    ggml_webgpu_hash_combine(تخم, کلی.k_مستقیم);
    ggml_webgpu_hash_combine(تخم, کلی.v_مستقیم);
    ggml_webgpu_hash_combine(تخم, کلی.kv_overlap);
    ggml_webgpu_hash_combine(تخم, کلی.has_ماسک);
    ggml_webgpu_hash_combine(تخم, کلی.has_sinks);
    ggml_webgpu_hash_combine(تخم, کلی.کاروي_logit_softcap);
}

بنية ggml_webgpu_flash_attn_vec_pipeline_کلی {
    ggml_webgpu_flash_attn_common_pipeline_کلی common;

    منطقي چلوونکی==(ثابت ggml_webgpu_flash_attn_vec_pipeline_کلی & other) ثابت { أرجع common == other.common; }
};

بنية ggml_webgpu_flash_attn_vec_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_flash_attn_vec_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_کلی(تخم, کلی.common);
        أرجع تخم;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_کلی {
    ggml_webgpu_flash_attn_common_pipeline_کلی common;
    منطقي                                       use_sg_ماتریکس;

    منطقي چلوونکی==(ثابت ggml_webgpu_flash_attn_pipeline_کلی & other) ثابت {
        أرجع common == other.common && use_sg_ماتریکس == other.use_sg_ماتریکس;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_flash_attn_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_کلی(تخم, کلی.common);
        ggml_webgpu_hash_combine(تخم, کلی.use_sg_ماتریکس);
        أرجع تخم;
    }
};

بنية ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_اندازه = 0;
};

بنية ggml_webgpu_flash_attn_decisions {
    منطقي     use_sg_ماتریکس = خطأ;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_اندازه       = 0;
};

مضمن constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_پلنوالی = 4u;
مضمن constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

مضمن size_t ggml_webgpu_flash_attn_tensor_دننه_واټن(ثابت ggml_tensor * tensor) {
    constexpr uintptr_t ptr_بنسټ_addr = 0x1000u;
    ثابت ggml_tensor * بنسټ          = tensor->view_src != تش_اشاره ? tensor->view_src : tensor;
    أرجع reinterpret_cast<uintptr_t>(بنسټ->معلومات) - ptr_بنسټ_addr + tensor->view_offs;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K, size_t storage_دننه_واټن_alignment) {
    ثابت uint32_t دننه_واټن_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_دننه_واټن(K) & (storage_دننه_واټن_alignment - 1)) /
                    ggml_type_اندازه(K->type));
    أرجع دننه_واټن_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_پلنوالی == 0u;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K,
                                                      ثابت ggml_tensor * V,
                                                      size_t              storage_دننه_واټن_alignment) {
    أرجع ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_دننه_واټن_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_دننه_واټن_alignment);
}

مضمن منطقي ggml_webgpu_flash_attn_k_مستقیم(ثابت ggml_tensor * Q, ثابت ggml_tensor * K, uint32_t kv_مستقیم_align) {
    أرجع (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_مستقیم_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

مضمن منطقي ggml_webgpu_flash_attn_v_مستقیم(ثابت ggml_tensor * Q, ثابت ggml_tensor * V, uint32_t kv_مستقیم_align) {
    أرجع ggml_webgpu_flash_attn_k_مستقیم(Q, V, kv_مستقیم_align);
}

مضمن ggml_webgpu_flash_attn_common_pipeline_کلی ggml_webgpu_flash_attn_make_common_pipeline_کلی(
    ثابت ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_مستقیم_align,
    منطقي                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_کلی کلی = {};
    کلی.q_type                                     = context.src0->type;
    کلی.k_type                                     = context.src1->type;
    کلی.v_type                                     = context.src2->type;
    کلی.dst_type                                   = context.dst->type;
    کلی.سر_dim_qk                                = (uint32_t) context.src0->ne[0];
    کلی.سر_dim_v                                 = (uint32_t) context.src2->ne[0];
    کلی.k_مستقیم           = ggml_webgpu_flash_attn_k_مستقیم(context.src0, context.src1, kv_مستقیم_align);
    کلی.v_مستقیم           = ggml_webgpu_flash_attn_v_مستقیم(context.src0, context.src2, kv_مستقیم_align);
    کلی.kv_overlap         = kv_overlap;
    کلی.has_ماسک           = context.src3 != تش_اشاره;
    کلی.has_sinks          = context.src4 != تش_اشاره;
    کلی.کاروي_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    أرجع کلی;
}

مضمن std::vector<std::string> ggml_webgpu_flash_attn_common_تعریفوي(
    ثابت ggml_webgpu_flash_attn_common_pipeline_کلی & کلی,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_اندازه) {
    std::vector<std::string> تعریفوي;

    بدل (کلی.k_type) {
        حالة GGML_TYPE_F32:
            تعریفوي.push_back("K_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            تعریفوي.push_back("K_F16");
            اكسر;
        حالة GGML_TYPE_Q4_0:
            تعریفوي.push_back("K_Q4_0");
            اكسر;
        حالة GGML_TYPE_Q8_0:
            تعریفوي.push_back("K_Q8_0");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(کلی.k_type);

    بدل (کلی.v_type) {
        حالة GGML_TYPE_F32:
            تعریفوي.push_back("V_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            تعریفوي.push_back("V_F16");
            اكسر;
        حالة GGML_TYPE_Q4_0:
            تعریفوي.push_back("V_Q4_0");
            اكسر;
        حالة GGML_TYPE_Q8_0:
            تعریفوي.push_back("V_Q8_0");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(کلی.v_type);

    بدل (کلی.q_type) {
        حالة GGML_TYPE_F32:
            تعریفوي.push_back("Q_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            تعریفوي.push_back("Q_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(کلی.q_type);

    بدل (کلی.dst_type) {
        حالة GGML_TYPE_F32:
            تعریفوي.push_back("DST_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            تعریفوي.push_back("DST_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(کلی.dst_type);

    إذا (کلی.has_ماسک) {
        تعریفوي.push_back("MASK");
        variant += "_mask";
    }
    إذا (کلی.has_sinks) {
        تعریفوي.push_back("SINKS");
        variant += "_sinks";
    }
    إذا (کلی.کاروي_logit_softcap) {
        تعریفوي.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    إذا (کلی.k_مستقیم) {
        تعریفوي.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    إذا (کلی.v_مستقیم) {
        تعریفوي.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    إذا (کلی.kv_overlap) {
        تعریفوي.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    تعریفوي.push_back(std::string("HEAD_DIM_QK=") + std::to_string(کلی.سر_dim_qk));
    variant += std::string("_hsqk") + std::to_string(کلی.سر_dim_qk);

    تعریفوي.push_back(std::string("HEAD_DIM_V=") + std::to_string(کلی.سر_dim_v));
    variant += std::string("_hsv") + std::to_string(کلی.سر_dim_v);

    تعریفوي.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    تعریفوي.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));

    إذا (ggml_is_quantized(کلی.k_type) || ggml_is_quantized(کلی.v_type)) {
        تعریفوي.push_back("U32_DEQUANT_HELPERS");
        إذا (ggml_is_quantized(کلی.k_type)) {
            تعریفوي.push_back("LOADERS_QUANTIZED_K");
        }
        إذا (ggml_is_quantized(کلی.v_type)) {
            تعریفوي.push_back("LOADERS_QUANTIZED_V");
        }
    }

    أرجع تعریفوي;
}

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_کلی {
    uint32_t  سر_dim_v;
    uint32_t  wg_اندازه;
    ggml_type dst_type;
};

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.سر_dim_v);
        ggml_webgpu_hash_combine(تخم, کلی.wg_اندازه);
        ggml_webgpu_hash_combine(تخم, کلی.dst_type);
        أرجع تخم;
    }
};

مضمن منطقي چلوونکی==(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلی & lhs,
                       ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلی & rhs) {
    أرجع lhs.سر_dim_v == rhs.سر_dim_v && lhs.wg_اندازه == rhs.wg_اندازه && lhs.dst_type == rhs.dst_type;
}

بنية ggml_webgpu_flash_attn_blk_pipeline_کلی {
    uint32_t kv_tile;

    منطقي چلوونکی==(ثابت ggml_webgpu_flash_attn_blk_pipeline_کلی & other) ثابت { أرجع kv_tile == other.kv_tile; }
};

بنية ggml_webgpu_flash_attn_blk_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_flash_attn_blk_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.kv_tile);
        أرجع تخم;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
مضمن size_t ggml_webgpu_flash_attn_wg_mem_بایټونه(uint32_t q_tile,
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
    أرجع f16_elems * GGML_WEBGPU_F16_اندازه_بایټونه + f32_elems * GGML_WEBGPU_F32_اندازه_بایټونه;
}

مضمن uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   حد_بایټونه,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t سر_dim_qk,
                                                   uint32_t سر_dim_v,
                                                   منطقي     has_ماسک,
                                                   منطقي     kv_مستقیم) {
    ثابت size_t بنسټ_q_بایټونه =
        ggml_webgpu_flash_attn_wg_mem_بایټونه(q_tile, 0, سر_dim_qk, سر_dim_v, has_ماسک, kv_مستقیم);
    إذا (حد_بایټونه <= بنسټ_q_بایټونه) {
        أرجع 0;
    }
    ثابت size_t one_kv_بایټونه =
        ggml_webgpu_flash_attn_wg_mem_بایټونه(q_tile, 1, سر_dim_qk, سر_dim_v, has_ماسک, kv_مستقیم);
    ثابت size_t بایټونه_فی_kv = one_kv_بایټونه - بنسټ_q_بایټونه;
    إذا (بایټونه_فی_kv == 0) {
        أرجع 0;
    }
    ثابت size_t max_kv_tile = (حد_بایټونه - بنسټ_q_بایټونه) / بایټونه_فی_kv;
    أرجع (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

مضمن uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_حد_بایټونه,
                                                       uint32_t سر_dim_qk,
                                                       uint32_t سر_dim_v,
                                                       منطقي     has_ماسک,
                                                       منطقي     kv_مستقیم) {
    ثابت uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_حد_بایټونه, 1u, 1u, سر_dim_qk, سر_dim_v, has_ماسک, kv_مستقیم);
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

مضمن منطقي ggml_webgpu_flash_attn_can_use_subgroup_ماتریکس_لار(منطقي                supports_subgroup_ماتریکس,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ثابت ggml_tensor * Q,
                                                                ثابت ggml_tensor * V) {
    أرجع supports_subgroup_ماتریکس && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

بنية ggml_webgpu_mul_mat_vec_pipeline_کلی {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    uint32_t  عدد_ستون_شمېر;
    منطقي      use_mmvq;
    منطقي      src_overlap;

    منطقي چلوونکی==(ثابت ggml_webgpu_mul_mat_vec_pipeline_کلی & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               عدد_ستون_شمېر == other.عدد_ستون_شمېر && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_vec_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_mul_mat_vec_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.src0_type);
        ggml_webgpu_hash_combine(تخم, کلی.src1_type);
        ggml_webgpu_hash_combine(تخم, کلی.vectorized);
        ggml_webgpu_hash_combine(تخم, کلی.عدد_ستون_شمېر);
        ggml_webgpu_hash_combine(تخم, کلی.use_mmvq);
        ggml_webgpu_hash_combine(تخم, کلی.src_overlap);
        أرجع تخم;
    }
};

بنية ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_اندازه;
    uint32_t outputs_فی_wg;
    uint32_t vec_اندازه;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_quantize_q8_pipeline_کلی {
    ggml_type src0_type;

    منطقي چلوونکی==(ثابت ggml_webgpu_quantize_q8_pipeline_کلی & other) ثابت { أرجع src0_type == other.src0_type; }
};

بنية ggml_webgpu_quantize_q8_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_quantize_q8_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.src0_type);
        أرجع تخم;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_کلی {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    صحيح       use_subgroup_ماتریکس;
    منطقي      src_overlap;

    منطقي چلوونکی==(ثابت ggml_webgpu_mul_mat_pipeline_کلی & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_ماتریکس == other.use_subgroup_ماتریکس && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_mul_mat_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.src0_type);
        ggml_webgpu_hash_combine(تخم, کلی.src1_type);
        ggml_webgpu_hash_combine(تخم, کلی.vectorized);
        ggml_webgpu_hash_combine(تخم, کلی.use_subgroup_ماتریکس);
        ggml_webgpu_hash_combine(تخم, کلی.src_overlap);
        أرجع تخم;
    }
};

بنية ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_اندازه_m;
    uint32_t wg_اندازه_n;
    uint32_t wg_اندازه;
    uint32_t outputs_فی_wg;
    صحيح      use_subgroup_ماتریکس;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_ماتریکس_m;
    uint32_t subgroup_ماتریکس_n;

    uint32_t mul_mat_wg_اندازه;
    منطقي     src_overlap = خطأ;
};

/** MUL_MAT_ID **/

بنية ggml_webgpu_mul_mat_id_gather_pipeline_کلی {
    uint32_t n_experts;

    منطقي چلوونکی==(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_کلی & other) ثابت {
        أرجع n_experts == other.n_experts;
    }
};

بنية ggml_webgpu_mul_mat_id_gather_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.n_experts);
        أرجع تخم;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_کلی {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  عدد_ستون_شمېر;
    صحيح       vectorized;

    منطقي چلوونکی==(ثابت ggml_webgpu_mul_mat_id_pipeline_کلی & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               عدد_ستون_شمېر == other.عدد_ستون_شمېر && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_mul_mat_id_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.src0_type);
        ggml_webgpu_hash_combine(تخم, کلی.src1_type);
        ggml_webgpu_hash_combine(تخم, کلی.n_experts);
        ggml_webgpu_hash_combine(تخم, کلی.عدد_ستون_شمېر);
        ggml_webgpu_hash_combine(تخم, کلی.vectorized);
        أرجع تخم;
    }
};

/** Cpy **/

بنية ggml_webgpu_cpy_pipeline_کلی {
    ggml_type src_type;
    ggml_type dst_type;

    منطقي چلوونکی==(ثابت ggml_webgpu_cpy_pipeline_کلی & other) ثابت {
        أرجع src_type == other.src_type && dst_type == other.dst_type;
    }
};

بنية ggml_webgpu_cpy_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_cpy_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.src_type);
        ggml_webgpu_hash_combine(تخم, کلی.dst_type);
        أرجع تخم;
    }
};

/** Glu **/

بنية ggml_webgpu_glu_pipeline_کلی {
    ggml_glu_op glu_op;
    ggml_type   type;
    منطقي        وېشل;
    منطقي        src_overlap;

    منطقي چلوونکی==(ثابت ggml_webgpu_glu_pipeline_کلی & other) ثابت {
        أرجع glu_op == other.glu_op && type == other.type && وېشل == other.وېشل && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_glu_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_glu_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.glu_op);
        ggml_webgpu_hash_combine(تخم, کلی.type);
        ggml_webgpu_hash_combine(تخم, کلی.وېشل);
        ggml_webgpu_hash_combine(تخم, کلی.src_overlap);
        أرجع تخم;
    }
};

/** Rope **/

بنية ggml_webgpu_rope_pipeline_کلی {
    ggml_type type;
    منطقي      inplace;
    منطقي      has_ff;

    منطقي چلوونکی==(ثابت ggml_webgpu_rope_pipeline_کلی & other) ثابت {
        أرجع type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

بنية ggml_webgpu_rope_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_rope_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.type);
        ggml_webgpu_hash_combine(تخم, کلی.inplace);
        ggml_webgpu_hash_combine(تخم, کلی.has_ff);
        أرجع تخم;
    }
};

/** SoftMax **/

بنية ggml_webgpu_soft_max_pipeline_کلی {
    ggml_type ماسک_type;
    منطقي      has_ماسک;
    منطقي      has_sink;
    منطقي      inplace;

    منطقي چلوونکی==(ثابت ggml_webgpu_soft_max_pipeline_کلی & other) ثابت {
        أرجع ماسک_type == other.ماسک_type && has_ماسک == other.has_ماسک && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

بنية ggml_webgpu_soft_max_pipeline_کلی_hash {
    size_t چلوونکی()(ثابت ggml_webgpu_soft_max_pipeline_کلی & کلی) ثابت {
        size_t تخم = 0;
        ggml_webgpu_hash_combine(تخم, کلی.ماسک_type);
        ggml_webgpu_hash_combine(تخم, کلی.has_ماسک);
        ggml_webgpu_hash_combine(تخم, کلی.has_sink);
        ggml_webgpu_hash_combine(تخم, کلی.inplace);
        أرجع تخم;
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

    std::unordered_map<صحيح, webgpu_pipeline> مجموعه_لیکې_pipelines;       // key is fixed, no variants yet
    std::unordered_map<صحيح, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<صحيح, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<صحيح, webgpu_pipeline> argsort_یوځای_pipelines;  // key is order
    std::unordered_map<صحيح, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_لیکه_norm_pipeline_کلی, webgpu_pipeline, ggml_webgpu_لیکه_norm_pipeline_کلی_hash>
        لیکه_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_لیکې_pipeline_کلی, webgpu_pipeline, ggml_webgpu_get_لیکې_pipeline_کلی_hash>
        get_لیکې_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_کلی, webgpu_pipeline, ggml_webgpu_unary_pipeline_کلی_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_کلی, webgpu_pipeline, ggml_webgpu_scale_pipeline_کلی_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_کلی, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_کلی_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_کلی, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_کلی_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_لټون_pipeline_کلی, webgpu_pipeline, ggml_webgpu_ssm_لټون_pipeline_کلی_hash>
        ssm_لټون_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_کلی,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_کلی_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_کلی, webgpu_pipeline, ggml_webgpu_pad_pipeline_کلی_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_دوګونی_pipeline_کلی, webgpu_pipeline, ggml_webgpu_دوګونی_pipeline_کلی_hash>
        دوګونی_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_جمع_کړه_id_pipeline_کلی, webgpu_pipeline, ggml_webgpu_جمع_کړه_id_pipeline_کلی_hash>
        جمع_کړه_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_کلی, webgpu_pipeline, ggml_webgpu_concat_pipeline_کلی_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_کلی, webgpu_pipeline, ggml_webgpu_repeat_pipeline_کلی_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_کلی,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_کلی_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_کلی, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_کلی_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_کلی,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_کلی_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_کلی,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_کلی_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_کلی, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_کلی_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_کلی, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_کلی_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_کلی, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_کلی_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_کلی,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_کلی_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_کلی, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_کلی_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_کلی, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_کلی_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_سېټ_لیکې_pipeline_کلی, webgpu_pipeline, ggml_webgpu_سېټ_لیکې_pipeline_کلی_hash>
        سېټ_لیکې_pipelines;
    std::unordered_map<ggml_webgpu_سېټ_pipeline_کلی, webgpu_pipeline, ggml_webgpu_سېټ_pipeline_کلی_hash> سېټ_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_کلی, webgpu_pipeline, ggml_webgpu_cpy_pipeline_کلی_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_کلی, webgpu_pipeline, ggml_webgpu_glu_pipeline_کلی_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_کلی, webgpu_pipeline, ggml_webgpu_rope_pipeline_کلی_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_کلی, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_کلی_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_کلی, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_کلی_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_کلی, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_کلی_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_کلی, webgpu_pipeline, ggml_webgpu_im2col_pipeline_کلی_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_کلی,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_کلی_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_کلی, webgpu_pipeline, ggml_webgpu_upscale_pipeline_کلی_hash>
        upscale_pipelines;

  عام:
    ggml_webgpu_shader_lib(wgpu::Device device) { هذا->device = device; }

    webgpu_pipeline get_مجموعه_لیکې_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        خپلکاری it = مجموعه_لیکې_pipelines.find(1);
        إذا (it != مجموعه_لیکې_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> تعریفوي;
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed        = preprocessor.preprocess(wgsl_مجموعه_لیکې, تعریفوي);
        مجموعه_لیکې_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        أرجع مجموعه_لیکې_pipelines[1];
    }

    webgpu_pipeline get_لیکه_norm_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_لیکه_norm_pipeline_کلی کلی = {};
        کلی.op                                = context.dst->op;
        کلی.src_type                          = context.src0->type;
        کلی.dst_type                          = context.dst->type;
        کلی.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خپلکاری it = لیکه_norm_pipelines.find(کلی);
        إذا (it != لیکه_norm_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> تعریفوي;
        std::string              variant;

        بدل (کلی.op) {
            حالة GGML_OP_RMS_NORM:
                تعریفوي.push_back("RMS_NORM");
                variant = "rms_norm";
                اكسر;
            حالة GGML_OP_NORM:
                تعریفوي.push_back("NORM");
                variant = "norm";
                اكسر;
            حالة GGML_OP_L2_NORM:
                تعریفوي.push_back("L2_NORM");
                variant = "l2_norm";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        إذا (کلی.inplace) {
            تعریفوي.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (کلی.src_type == GGML_TYPE_F32) {
            تعریفوي.push_back("SRC_F32");
            variant += "_src_f32";
        } وإلا إذا (کلی.src_type == GGML_TYPE_F16) {
            تعریفوي.push_back("SRC_F16");
            variant += "_src_f16";
        }

        إذا (کلی.dst_type == GGML_TYPE_F32) {
            تعریفوي.push_back("DST_F32");
            variant += "_dst_f32";
        } وإلا إذا (کلی.dst_type == GGML_TYPE_F16) {
            تعریفوي.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ثابت uint32_t لیکه_norm_wg_اندازه = 128u;
        uint32_t       wg_اندازه          = std::min(context.max_wg_اندازه, لیکه_norm_wg_اندازه);
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        خپلکاری processed                  = preprocessor.preprocess(wgsl_لیکه_norm, تعریفوي);
        خپلکاری decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه              = wg_اندازه;
        decisions->inplace              = کلی.inplace;
        لیکه_norm_pipelines[کلی]         = ggml_webgpu_create_pipeline(device, processed, variant);
        لیکه_norm_pipelines[کلی].context = decisions;
        أرجع لیکه_norm_pipelines[کلی];
    }

    webgpu_pipeline get_argmax_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي vec4 = context.src0->ne[0] % 4 == 0;

        خپلکاری it = argmax_pipelines.find(vec4);
        إذا (it != argmax_pipelines.end()) {
            أرجع it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> تعریفوي;
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));
        إذا (vec4) {
            تعریفوي.push_back("VEC4");
            variant += "_vec4";
        }

        خپلکاری processed         = preprocessor.preprocess(wgsl_argmax, تعریفوي);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        أرجع argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_سېټ_لیکې_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_سېټ_لیکې_pipeline_کلی کلی       = {};
        کلی.src0_type                               = context.src0->type;
        کلی.dst_type                                = context.dst->type;
        کلی.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        کلی.i64_idx     = context.src1->type == GGML_TYPE_I64;
        کلی.pair_بلاکونه = quantized && ((context.src0->ne[0] / ggml_blck_اندازه(context.dst->type)) % 2 == 0);

        خپلکاری it = سېټ_لیکې_pipelines.find(کلی);
        إذا (it != سېټ_لیکې_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "set_rows";

        بدل (context.dst->type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("DST_F32");
                variant += "_dstf32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("DST_F16");
                variant += "_dstf16";
                اكسر;
            حالة GGML_TYPE_Q8_0:
                تعریفوي.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                اكسر;
            حالة GGML_TYPE_Q4_0:
                تعریفوي.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        إذا (context.src0->type == GGML_TYPE_F16) {
            تعریفوي.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        إذا (کلی.vec4) {
            تعریفوي.push_back("VEC4");
            variant += "_vec4";
        }
        إذا (کلی.i64_idx) {
            تعریفوي.push_back("I64_IDX");
            variant += "_i64idx";
        }
        إذا (کلی.pair_بلاکونه) {
            تعریفوي.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        ثابت خپلکاری & shader_سرچینه      = quantized ? wgsl_سېټ_لیکې_quant : wgsl_سېټ_لیکې;
        خپلکاری         processed          = preprocessor.preprocess(shader_سرچینه, تعریفوي);
        خپلکاری         decisions          = std::make_shared<ggml_webgpu_سېټ_لیکې_shader_decisions>();
        decisions->vec4                 = کلی.vec4;
        decisions->i64_idx              = کلی.i64_idx;
        decisions->pair_بلاکونه          = کلی.pair_بلاکونه;
        decisions->wg_اندازه              = context.max_wg_اندازه;
        سېټ_لیکې_pipelines[کلی]         = ggml_webgpu_create_pipeline(device, processed, variant);
        سېټ_لیکې_pipelines[کلی].context = decisions;
        أرجع سېټ_لیکې_pipelines[کلی];
    }

    webgpu_pipeline get_سېټ_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_سېټ_pipeline_کلی کلی = {};
        کلی.type                         = context.dst->type;
        کلی.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خپلکاری it = سېټ_pipelines.find(کلی);
        إذا (it != سېټ_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "set";

        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریفوي.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for set shader");
        }

        إذا (کلی.inplace) {
            تعریفوي.push_back("INPLACE");
            variant += "_inplace";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_سېټ, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->inplace       = کلی.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        سېټ_pipelines[کلی]       = pipeline;
        أرجع سېټ_pipelines[کلی];
    }

    webgpu_pipeline get_cumsum_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        خپلکاری it = cumsum_pipelines.find(1);
        إذا (it != cumsum_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed      = preprocessor.preprocess(wgsl_cumsum, تعریفوي);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        أرجع cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_سر_k = context.dst->op == GGML_OP_سر_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t ترتیب =
            is_سر_k ? (int32_t) GGML_ترتیبول_ترتیب_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        خپلکاری it = argsort_pipelines.find(ترتیب);
        إذا (it != argsort_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "argsort";
        تعریفوي.push_back(std::string("ORDER=") + std::to_string(ترتیب));
        variant += std::string("_order") + std::to_string(ترتیب);
        uint32_t wg_اندازه = 1;
        طالما (wg_اندازه * 2 <= context.max_wg_اندازه &&
               wg_اندازه * GGML_WEBGPU_I32_اندازه_بایټونه <= context.wg_mem_حد_بایټونه / 2) {
            wg_اندازه *= 2;
        }
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        خپلکاری processed                   = preprocessor.preprocess(wgsl_argsort, تعریفوي);
        خپلکاری decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه               = wg_اندازه;
        argsort_pipelines[ترتیب]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[ترتیب].context = decisions;
        أرجع argsort_pipelines[ترتیب];
    }

    webgpu_pipeline get_argsort_یوځای_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_سر_k = context.dst->op == GGML_OP_سر_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t ترتیب =
            is_سر_k ? (int32_t) GGML_ترتیبول_ترتیب_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        خپلکاری it = argsort_یوځای_pipelines.find(ترتیب);
        إذا (it != argsort_یوځای_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "argsort_merge";
        تعریفوي.push_back(std::string("ORDER=") + std::to_string(ترتیب));
        variant += std::string("_order") + std::to_string(ترتیب);
        uint32_t wg_اندازه = std::min(GGML_WEBGPU_ARGSORT_یوځای_MAX_WG_اندازه, context.max_wg_اندازه);
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));

        خپلکاری processed                 = preprocessor.preprocess(wgsl_argsort_یوځای, تعریفوي);
        argsort_یوځای_pipelines[ترتیب] = ggml_webgpu_create_pipeline(device, processed, variant);
        أرجع argsort_یوځای_pipelines[ترتیب];
    }

    webgpu_pipeline get_get_لیکې_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي vec4_aligned) {
        ثابت منطقي vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_لیکې_pipeline_کلی کلی = {};
        کلی.src_type                          = context.src0->type;
        کلی.vectorized                        = (صحيح) vectorized;

        خپلکاری it = get_لیکې_pipelines.find(کلی);
        إذا (it != get_لیکې_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "get_rows";

        ثابت بنية ggml_type_traits * type_traits = ggml_get_type_traits(کلی.src_type);
        ثابت حرف *                    type_str    = type_traits->type_name;

        بدل (کلی.src_type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("FLOAT_PARALLEL");
                إذا (کلی.vectorized) {
                    تعریفوي.push_back("F32_VEC");
                    تعریفوي.push_back("SRC_TYPE=vec4<f32>");
                    تعریفوي.push_back("DST_TYPE=vec4<f32>");
                    تعریفوي.push_back("BLOCK_SIZE=4u");
                } وإلا {
                    تعریفوي.push_back("F32");
                    تعریفوي.push_back("SRC_TYPE=f32");
                    تعریفوي.push_back("DST_TYPE=f32");
                    تعریفوي.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("FLOAT_PARALLEL");
                تعریفوي.push_back("F16");
                تعریفوي.push_back("SRC_TYPE=f16");
                تعریفوي.push_back("DST_TYPE=f32");
                تعریفوي.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریفوي.push_back("FLOAT_PARALLEL");
                تعریفوي.push_back("I32");
                تعریفوي.push_back("SRC_TYPE=i32");
                تعریفوي.push_back("DST_TYPE=i32");
                تعریفوي.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                اكسر;
            حالة GGML_TYPE_BF16:
                تعریفوي.push_back("BF16");
                تعریفوي.push_back("SRC_TYPE=u32");
                تعریفوي.push_back("DST_TYPE=f32");
                تعریفوي.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_پاسنی = type_str;
                    std::transform(type_پاسنی.begin(), type_پاسنی.end(), type_پاسنی.begin(), ::لوی_توری);

                    بدل (کلی.src_type) {
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
                                تعریفوي.push_back("SRC_TYPE=u32");
                                تعریفوي.push_back("U32_DEQUANT_HELPERS");
                                اكسر;
                            }
                        افتراضي:
                            {
                                تعریفوي.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    تعریفوي.push_back("BYTE_HELPERS");
                    تعریفوي.push_back(type_پاسنی + "_T");
                    تعریفوي.push_back(type_پاسنی);
                    تعریفوي.push_back(type_پاسنی + "_SCALE_MIN");
                    تعریفوي.push_back(type_پاسنی + "_TABLES");
                    تعریفوي.push_back(type_پاسنی + "_GRID");
                    تعریفوي.push_back(type_پاسنی + "_LUT");

                    variant += "_";
                    variant += type_str;

                    تعریفوي.push_back("DST_TYPE=f32");

                    إذا (کلی.src_type == GGML_TYPE_Q1_0) {
                        تعریفوي.push_back("BLOCK_SIZE=128u");
                    } وإلا إذا ((کلی.src_type >= GGML_TYPE_Q4_0 && کلی.src_type <= GGML_TYPE_Q8_1) ||
                               کلی.src_type == GGML_TYPE_IQ4_NL || کلی.src_type == GGML_TYPE_MXFP4) {
                        تعریفوي.push_back("BLOCK_SIZE=32u");
                    } وإلا إذا (کلی.src_type == GGML_TYPE_Q2_0 || کلی.src_type == GGML_TYPE_NVFP4) {
                        تعریفوي.push_back("BLOCK_SIZE=64u");
                    } وإلا إذا (کلی.src_type >= GGML_TYPE_Q2_K) {
                        تعریفوي.push_back("BLOCK_SIZE=256u");
                    } وإلا {
                        تعریفوي.push_back("BLOCK_SIZE=1u");
                    }
                    اكسر;
                }
        }

        إذا (کلی.vectorized) {
            variant += "_vec";
        }

        تعریفوي.push_back("WG_SIZE=" + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_get_لیکې, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_get_لیکې_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_لیکې_pipelines[کلی]  = pipeline;
        أرجع get_لیکې_pipelines[کلی];
    }

    webgpu_pipeline get_scale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_کلی کلی = {};
        کلی.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خپلکاری it = scale_pipelines.find(کلی);
        إذا (it != scale_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "scale";

        إذا (کلی.inplace) {
            تعریفوي.push_back("INPLACE");
            variant += "_inplace";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_scale, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->inplace       = کلی.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[کلی]     = pipeline;
        أرجع scale_pipelines[کلی];
    }

    webgpu_pipeline get_solve_tri_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_کلی کلی = {};
        کلی.type                               = context.dst->type;
        کلی.n                                  = (صحيح) context.src0->ne[0];
        کلی.k                                  = (صحيح) context.src1->ne[0];

        خپلکاری it = solve_tri_pipelines.find(کلی);
        إذا (it != solve_tri_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "solve_tri";

        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ثابت uint32_t wg_اندازه       = std::min((uint32_t) کلی.n, context.max_wg_اندازه);
        ثابت uint32_t k_tile        = wg_اندازه;
        ثابت uint32_t بایټونه_فی_لیکه = ((uint32_t) کلی.n + wg_اندازه) * GGML_WEBGPU_F32_اندازه_بایټونه;
        ثابت uint32_t batch_n       = (uint32_t) (context.wg_mem_حد_بایټونه / بایټونه_فی_لیکه);

        تعریفوي.push_back(std::string("N=") + std::to_string(کلی.n));
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        تعریفوي.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        تعریفوي.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        خپلکاری processed           = preprocessor.preprocess(wgsl_solve_tri, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[کلی] = pipeline;
        أرجع solve_tri_pipelines[کلی];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_کلی کلی = {};
        کلی.type                              = context.dst->type;
        کلی.vectorized                        = context.src1->ne[0] == 4;

        خپلکاری it = ssm_conv_pipelines.find(کلی);
        إذا (it != ssm_conv_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "ssm_conv";

        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        إذا (کلی.vectorized) {
            تعریفوي.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t بلاک_اندازه_3    = 32u;
        constexpr uint32_t نښې_فی_wg = 8u;

        تعریفوي.push_back("BLOCK_SIZE=" + std::to_string(بلاک_اندازه_3) + "u");
        تعریفوي.push_back("TOKENS_PER_WG=" + std::to_string(نښې_فی_wg) + "u");

        خپلکاری processed           = preprocessor.preprocess(wgsl_ssm_conv, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->بلاک_اندازه_3    = بلاک_اندازه_3;
        decisions->نښې_فی_wg = نښې_فی_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[کلی]  = pipeline;
        أرجع ssm_conv_pipelines[کلی];
    }

    webgpu_pipeline get_ssm_لټون_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_لټون_pipeline_کلی کلی = {};
        کلی.type                              = context.dst->type;
        کلی.d_state                           = (صحيح) context.src0->ne[0];
        کلی.xbc_overlap                       = xbc_overlap;

        خپلکاری it = ssm_لټون_pipelines.find(کلی);
        إذا (it != ssm_لټون_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "ssm_scan";

        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ثابت uint32_t wg_اندازه = (uint32_t) کلی.d_state;

        constexpr uint32_t نښې_فی_tile = 4u;

        تعریفوي.push_back("WG_SIZE=" + std::to_string(wg_اندازه) + "u");
        تعریفوي.push_back("TOKENS_PER_TILE=" + std::to_string(نښې_فی_tile) + "u");

        إذا (context.supports_subgroups) {
            تعریفوي.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } وإلا {
            variant += "_wg_reduce";
        }

        إذا (کلی.xbc_overlap == 0b110) {  // x/B
            تعریفوي.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } وإلا إذا (کلی.xbc_overlap == 0b011) {  // B/C
            تعریفوي.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } وإلا إذا (کلی.xbc_overlap == 0b111) {  // x/B/C
            تعریفوي.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(کلی.d_state);

        خپلکاری processed             = preprocessor.preprocess(wgsl_ssm_لټون, تعریفوي);
        خپلکاری decisions             = std::make_shared<ggml_webgpu_ssm_لټون_shader_decisions>();
        decisions->wg_اندازه         = wg_اندازه;
        decisions->نښې_فی_tile = نښې_فی_tile;
        decisions->xbc_overlap     = کلی.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_لټون_pipelines[کلی]    = pipeline;
        أرجع ssm_لټون_pipelines[کلی];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_کلی کلی = {};
        کلی.type                                     = context.dst->type;
        کلی.s_v                                      = (صحيح) context.src2->ne[0];
        کلی.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        کلی.fused_cache                              = context.dst_fuse != تش_اشاره;

        خپلکاری it = gated_delta_net_pipelines.find(کلی);
        إذا (it != gated_delta_net_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "gated_delta_net";

        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        إذا (کلی.kda) {
            تعریفوي.push_back("KDA");
            variant += "_kda";
        }

        إذا (کلی.fused_cache) {
            تعریفوي.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        تعریفوي.push_back("S_V=" + std::to_string(کلی.s_v) + "u");
        تعریفوي.push_back("WG_SIZE=" + std::to_string(کلی.s_v) + "u");

        خپلکاری            processed      = preprocessor.preprocess(wgsl_gated_delta_net, تعریفوي);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[کلی] = pipeline;
        أرجع gated_delta_net_pipelines[کلی];
    }

    webgpu_pipeline get_pad_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_کلی کلی = {};
        کلی.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        خپلکاری it = pad_pipelines.find(کلی);
        إذا (it != pad_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "pad";

        إذا (کلی.circular) {
            تعریفوي.push_back("CIRCULAR");
            variant += "_circular";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_pad, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[کلی]       = pipeline;
        أرجع pad_pipelines[کلی];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_کلی کلی = {};
        کلی.src0_type                            = context.src0->type;

        خپلکاری it = quantize_q8_pipelines.find(کلی);
        إذا (it != quantize_q8_pipelines.end()) {
            أرجع it->second;
        }
        ثابت حرف *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> تعریفوي;
        std::string              variant = "quantize_q8";

        uint32_t wg_اندازه = WEBGPU_MUL_MAT_VEC_WG_اندازه;

        تعریفوي.push_back("SRC1_INNER_TYPE=f32");
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));

        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_پاسنی  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_پاسنی.begin(), type_پاسنی.end(), type_پاسنی.begin(), ::لوی_توری);

        تعریفوي.push_back("MUL_ACC_" + type_پاسنی);
        تعریفوي.push_back("Q8_1_T");

        تعریفوي.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        خپلکاری processed             = preprocessor.preprocess(shader_src, تعریفوي);
        خپلکاری decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه         = wg_اندازه;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[کلی] = pipeline;
        أرجع quantize_q8_pipelines[کلی];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_کلی کلی = {};
        کلی.src0_type                            = context.src0->type;
        کلی.src1_type                            = context.src1->type;
        کلی.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        کلی.عدد_ستون_شمېر   = context.dst->ne[1];
        کلی.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_حاصل_ضرب, context.vendor);
        کلی.src_overlap = src_overlap;

        خپلکاری it = mul_mat_vec_pipelines.find(کلی);
        إذا (it != mul_mat_vec_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant    = "mul_mat_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("SRC0_INNER_TYPE=f32");
                تعریفوي.push_back("MUL_ACC_FLOAT");
                تعریفوي.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("SRC0_INNER_TYPE=f16");
                تعریفوي.push_back("MUL_ACC_FLOAT");
                تعریفوي.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                تعریفوي.push_back("SRC0_INNER_TYPE=u32");
                تعریفوي.push_back("MUL_ACC_FLOAT");
                تعریفوي.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_پاسنی  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_پاسنی.begin(), type_پاسنی.end(), type_پاسنی.begin(), ::لوی_توری);

                    تعریفوي.push_back("BYTE_HELPERS");
                    تعریفوي.push_back("MUL_ACC_" + type_پاسنی);
                    تعریفوي.push_back("U32_DEQUANT_HELPERS");
                    تعریفوي.push_back("SRC0_INNER_TYPE=u32");
                    بدل (context.src0->type) {
                        حالة GGML_TYPE_Q4_0:
                        حالة GGML_TYPE_Q4_1:
                        حالة GGML_TYPE_Q5_0:
                        حالة GGML_TYPE_Q5_1:
                        حالة GGML_TYPE_Q8_0:
                            إذا (کلی.use_mmvq) {
                                تعریفوي.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_Q1_0:
                        حالة GGML_TYPE_Q2_K:
                        حالة GGML_TYPE_Q3_K:
                        حالة GGML_TYPE_Q4_K:
                        حالة GGML_TYPE_Q5_K:
                        حالة GGML_TYPE_Q6_K:
                            إذا (کلی.use_mmvq) {
                                تعریفوي.push_back("K_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            تعریفوي.push_back(type_پاسنی + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ3_XXS:
                            تعریفوي.push_back(type_پاسنی + "_GRID");
                            تعریفوي.push_back(type_پاسنی + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                            تعریفوي.push_back(type_پاسنی + "_LUT");
                            إذا (کلی.use_mmvq) {
                                تعریفوي.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_NVFP4:
                            تعریفوي.push_back(type_پاسنی + "_LUT");
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
                تعریفوي.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        تعریفوي.push_back(کلی.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_اندازه        = WEBGPU_MUL_MAT_VEC_WG_اندازه;
        uint32_t outputs_فی_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_فی_WG;

        إذا (کلی.src0_type == GGML_TYPE_Q1_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        } وإلا إذا (کلی.src0_type >= GGML_TYPE_Q2_K) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_فی_WG;
        } وإلا إذا (کلی.src0_type >= GGML_TYPE_Q4_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        }

        إذا (کلی.use_mmvq) {
            تعریفوي.push_back("MMVQ");
            تعریفوي.push_back("Q8_1_T");
        }

        إذا (کلی.src_overlap) {
            تعریفوي.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        تعریفوي.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_فی_wg));
        تعریفوي.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (کلی.vectorized) {
            variant += "_vectorized";
        }
        تعریفوي.push_back(std::string("NUM_COLS=") + std::to_string(کلی.عدد_ستون_شمېر));

        خپلکاری processed            = preprocessor.preprocess(shader_src, تعریفوي);
        خپلکاری decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_اندازه        = wg_اندازه;
        decisions->outputs_فی_wg = outputs_فی_wg;
        decisions->vec_اندازه       = کلی.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[کلی] = pipeline;
        أرجع mul_mat_vec_pipelines[کلی];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_mul_mat_pipeline_کلی کلی = {};
        کلی.src0_type                        = context.src0->type;
        کلی.src1_type                        = context.src1->type;
        کلی.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        کلی.use_subgroup_ماتریکس = context.supports_subgroup_ماتریکس;
        کلی.src_overlap         = src_overlap;

        خپلکاری it = mul_mat_fast_pipelines.find(کلی);
        إذا (it != mul_mat_fast_pipelines.end()) {
            أرجع it->second;
        }

        ثابت حرف * shader_src = کلی.use_subgroup_ماتریکس ? wgsl_mul_mat_subgroup_ماتریکس : wgsl_mul_mat_reg_tile;
        std::vector<std::string> تعریفوي;
        std::string              variant = کلی.use_subgroup_ماتریکس ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("SRC0_INNER_TYPE=f32");
                تعریفوي.push_back("TYPE_F32");
                تعریفوي.push_back("MUL_ACC_FLOAT");
                تعریفوي.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریفوي.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("SRC0_INNER_TYPE=f16");
                تعریفوي.push_back("TYPE_F16");
                تعریفوي.push_back("MUL_ACC_FLOAT");
                تعریفوي.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریفوي.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                تعریفوي.push_back("SRC0_INNER_TYPE=u32");
                تعریفوي.push_back("TYPE_BF16");
                تعریفوي.push_back("MUL_ACC_FLOAT");
                تعریفوي.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریفوي.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_پاسنی = src0_name;
                    std::transform(type_پاسنی.begin(), type_پاسنی.end(), type_پاسنی.begin(), ::لوی_توری);

                    تعریفوي.push_back("BYTE_HELPERS");
                    تعریفوي.push_back("MUL_ACC_" + type_پاسنی);
                    تعریفوي.push_back("INIT_SRC0_SHMEM_" + type_پاسنی);
                    تعریفوي.push_back("INIT_SRC1_SHMEM_FLOAT");
                    تعریفوي.push_back("U32_DEQUANT_HELPERS");
                    تعریفوي.push_back("SRC0_INNER_TYPE=u32");

                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            تعریفوي.push_back(type_پاسنی + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                            تعریفوي.push_back(type_پاسنی + "_GRID");
                            تعریفوي.push_back(type_پاسنی + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            تعریفوي.push_back(type_پاسنی + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }

                    variant += std::string("_") + src0_name;
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        تعریفوي.push_back(کلی.vectorized ? "VEC" : "SCALAR");

        ثابت منطقي is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        إذا (کلی.use_subgroup_ماتریکس) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } وإلا {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        تعریفوي.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        تعریفوي.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        إذا (کلی.use_subgroup_ماتریکس) {
            تعریفوي.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            تعریفوي.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_اندازه) + "u");
            تعریفوي.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            تعریفوي.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            تعریفوي.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_ماتریکس_M) + "u");
            تعریفوي.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_ماتریکس_N) + "u");
            تعریفوي.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            تعریفوي.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            تعریفوي.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        إذا (کلی.vectorized) {
            variant += "_vectorized";
        }

        إذا (کلی.src_overlap) {
            تعریفوي.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        إذا (!کلی.use_subgroup_ماتریکس) {
            تعریفوي.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_اندازه_M) + "u");
            تعریفوي.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_اندازه_N) + "u");
            تعریفوي.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        خپلکاری processed = preprocessor.preprocess(shader_src, تعریفوي);

        خپلکاری decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_ماتریکس = کلی.use_subgroup_ماتریکس;
        إذا (کلی.use_subgroup_ماتریکس) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_ماتریکس_m = WEBGPU_MUL_MAT_SUBGROUP_ماتریکس_M;
            decisions->subgroup_ماتریکس_n = WEBGPU_MUL_MAT_SUBGROUP_ماتریکس_N;
            decisions->wg_اندازه           = context.max_subgroup_اندازه;
        } وإلا {
            decisions->wg_اندازه_m       = WEBGPU_MUL_MAT_WG_اندازه_M;
            decisions->wg_اندازه_n       = WEBGPU_MUL_MAT_WG_اندازه_N;
            decisions->wg_اندازه         = WEBGPU_MUL_MAT_WG_اندازه_M * WEBGPU_MUL_MAT_WG_اندازه_N;
            decisions->mul_mat_wg_اندازه = WEBGPU_MUL_MAT_WG_اندازه;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[کلی] = pipeline;
        أرجع mul_mat_fast_pipelines[کلی];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_کلی کلی = {};
        کلی.n_experts                                  = (uint32_t) context.src0->ne[2];

        خپلکاری it = mul_mat_id_gather_pipelines.find(کلی);
        إذا (it != mul_mat_id_gather_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> تعریفوي;
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه) + "u");
        تعریفوي.push_back(std::string("N_EXPERT=") + std::to_string(کلی.n_experts) + "u");
        تعریفوي.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_اندازه_N) + "u");

        خپلکاری processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, تعریفوي);
        خپلکاری decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه = context.max_wg_اندازه;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[کلی] = pipeline;
        أرجع pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_کلی کلی = {};
        کلی.src0_type                           = context.src0->type;
        کلی.src1_type                           = context.src1->type;
        کلی.n_experts                           = context.src0->ne[2];
        کلی.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        خپلکاری it = mul_mat_id_pipelines.find(کلی);
        إذا (it != mul_mat_id_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "mul_mat_id";
        تعریفوي.push_back("MUL_MAT_ID");

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("SRC0_INNER_TYPE=f32");
                تعریفوي.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریفوي.push_back("INIT_SRC1_SHMEM_FLOAT");
                تعریفوي.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("SRC0_INNER_TYPE=f16");
                تعریفوي.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریفوي.push_back("INIT_SRC1_SHMEM_FLOAT");
                تعریفوي.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                تعریفوي.push_back("SRC0_INNER_TYPE=u32");
                تعریفوي.push_back("INIT_SRC0_SHMEM_FLOAT");
                تعریفوي.push_back("INIT_SRC1_SHMEM_FLOAT");
                تعریفوي.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_پاسنی = src0_name;
                    std::transform(type_پاسنی.begin(), type_پاسنی.end(), type_پاسنی.begin(), ::لوی_توری);

                    تعریفوي.push_back("BYTE_HELPERS");
                    تعریفوي.push_back("INIT_SRC0_SHMEM_" + type_پاسنی);
                    تعریفوي.push_back("INIT_SRC1_SHMEM_FLOAT");
                    تعریفوي.push_back("U32_DEQUANT_HELPERS");
                    تعریفوي.push_back("SRC0_INNER_TYPE=u32");

                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            تعریفوي.push_back(type_پاسنی + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                            تعریفوي.push_back(type_پاسنی + "_GRID");
                            تعریفوي.push_back(type_پاسنی + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            تعریفوي.push_back(type_پاسنی + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }

                    variant += std::string("_") + src0_name;
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        تعریفوي.push_back(کلی.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ثابت uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        تعریفوي.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        تعریفوي.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        تعریفوي.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        تعریفوي.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_اندازه_M) + "u");
        تعریفوي.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_اندازه_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        إذا (کلی.vectorized) {
            variant += "_vectorized";
        }

        خپلکاری processed = preprocessor.preprocess(wgsl_mul_mat_id, تعریفوي);

        خپلکاری decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_اندازه_m = WEBGPU_MUL_MAT_WG_اندازه_M;
        decisions->wg_اندازه_n = WEBGPU_MUL_MAT_WG_اندازه_N;
        decisions->wg_اندازه   = WEBGPU_MUL_MAT_WG_اندازه_M * WEBGPU_MUL_MAT_WG_اندازه_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[کلی] = pipeline;
        أرجع mul_mat_id_pipelines[کلی];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_کلی کلی = {};
        کلی.src0_type                           = context.src0->type;
        کلی.src1_type                           = context.src1->type;
        کلی.n_experts                           = context.src0->ne[2];
        کلی.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        خپلکاری it = mul_mat_id_vec_pipelines.find(کلی);
        إذا (it != mul_mat_id_vec_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant    = "mul_mat_id_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("SRC0_INNER_TYPE=f32");
                تعریفوي.push_back("MUL_ACC_FLOAT");
                تعریفوي.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("SRC0_INNER_TYPE=f16");
                تعریفوي.push_back("MUL_ACC_FLOAT");
                تعریفوي.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                تعریفوي.push_back("SRC0_INNER_TYPE=u32");
                تعریفوي.push_back("MUL_ACC_FLOAT");
                تعریفوي.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_پاسنی  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_پاسنی.begin(), type_پاسنی.end(), type_پاسنی.begin(), ::لوی_توری);

                    تعریفوي.push_back("BYTE_HELPERS");
                    تعریفوي.push_back("MUL_ACC_" + type_پاسنی);
                    تعریفوي.push_back("U32_DEQUANT_HELPERS");
                    تعریفوي.push_back("SRC0_INNER_TYPE=u32");
                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            تعریفوي.push_back(type_پاسنی + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ3_XXS:
                            تعریفوي.push_back(type_پاسنی + "_GRID");
                            تعریفوي.push_back(type_پاسنی + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            تعریفوي.push_back(type_پاسنی + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        تعریفوي.push_back(کلی.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_اندازه        = WEBGPU_MUL_MAT_VEC_WG_اندازه;
        uint32_t outputs_فی_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_فی_WG;

        إذا (کلی.src0_type == GGML_TYPE_Q1_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        } وإلا إذا (کلی.src0_type >= GGML_TYPE_Q2_K) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_فی_WG;
        } وإلا إذا (کلی.src0_type >= GGML_TYPE_Q4_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        تعریفوي.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_فی_wg));
        تعریفوي.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (کلی.vectorized) {
            variant += "_vectorized";
        }
        تعریفوي.push_back(std::string("NUM_COLS=1"));

        تعریفوي.push_back(std::string("N_EXPERTS=") + std::to_string(کلی.n_experts));

        خپلکاری processed = preprocessor.preprocess(shader_src, تعریفوي);

        خپلکاری decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_اندازه        = wg_اندازه;
        decisions->outputs_فی_wg = outputs_فی_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[کلی] = pipeline;
        أرجع mul_mat_id_vec_pipelines[کلی];
    }

    webgpu_pipeline get_unary_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                     is_unary = context.dst->op == GGML_OP_UNARY;
        ثابت صحيح                      op       = is_unary ? (صحيح) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_کلی کلی      = {};
        کلی.type                                = context.dst->type;
        کلی.op                                  = op;
        کلی.is_unary                            = is_unary;
        کلی.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        کلی.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        خپلکاری it = unary_pipelines.find(کلی);
        إذا (it != unary_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant =
            کلی.is_unary ? ggml_unary_op_name((ggml_unary_op) کلی.op) : ggml_op_name((ggml_op) کلی.op);
        تعریفوي.push_back(variant);

        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for unary shader");
        }

        إذا (کلی.inplace) {
            تعریفوي.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (op == GGML_OP_TRI) {
            بدل (کلی.ttype) {
                حالة GGML_TRI_TYPE_ښکتنی:
                    تعریفوي.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    اكسر;
                حالة GGML_TRI_TYPE_ښکتنی_DIAG:
                    تعریفوي.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    اكسر;
                حالة GGML_TRI_TYPE_پاسنی:
                    تعریفوي.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    اكسر;
                حالة GGML_TRI_TYPE_پاسنی_DIAG:
                    تعریفوي.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    اكسر;
                افتراضي:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_unary, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->inplace       = کلی.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[کلی]     = pipeline;
        أرجع unary_pipelines[کلی];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_کلی کلی = {};
        کلی.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        کلی.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        کلی.src_overlap                           = src_overlap;

        خپلکاری it = rms_norm_mul_pipelines.find(کلی);
        إذا (it != rms_norm_mul_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        إذا (کلی.inplace) {
            تعریفوي.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (کلی.overlap) {
            تعریفوي.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (کلی.src_overlap) {
            تعریفوي.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, تعریفوي);
        خپلکاری pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_اندازه     = context.max_wg_اندازه;
        pipeline_decisions->inplace     = کلی.inplace;
        pipeline_decisions->overlap     = کلی.overlap;
        pipeline_decisions->src_overlap = کلی.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[کلی]     = pipeline;
        أرجع rms_norm_mul_pipelines[کلی];
    }

    webgpu_pipeline get_دوګونی_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_دوګونی_pipeline_کلی کلی = {};
        کلی.type                            = context.dst->type;
        کلی.op                              = context.dst->op;
        کلی.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        کلی.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        کلی.src_overlap                     = src_overlap;

        خپلکاری it = دوګونی_pipelines.find(کلی);
        إذا (it != دوګونی_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              op_name = ggml_op_name((ggml_op) کلی.op);
        std::string              variant = op_name;

        تعریفوي.push_back(std::string("OP_") + op_name);

        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for binary shader");
        }

        إذا (کلی.inplace) {
            تعریفوي.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (کلی.overlap) {
            تعریفوي.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (کلی.src_overlap) {
            تعریفوي.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed                  = preprocessor.preprocess(wgsl_دوګونی, تعریفوي);
        خپلکاری pipeline_decisions         = std::make_shared<ggml_webgpu_دوګونی_shader_decisions>();
        pipeline_decisions->wg_اندازه     = context.max_wg_اندازه;
        pipeline_decisions->inplace     = کلی.inplace;
        pipeline_decisions->overlap     = کلی.overlap;
        pipeline_decisions->src_overlap = کلی.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        دوګونی_pipelines[کلی]    = pipeline;
        أرجع دوګونی_pipelines[کلی];
    }

    webgpu_pipeline get_جمع_کړه_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_جمع_کړه_id_pipeline_کلی کلی = {};
        کلی.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خپلکاری it = جمع_کړه_id_pipelines.find(کلی);
        إذا (it != جمع_کړه_id_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant    = "add_id";
        ثابت حرف *             shader_src = wgsl_جمع_کړه_id;

        إذا (کلی.inplace) {
            تعریفوي.push_back("INPLACE");
            variant += "_inplace";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed              = preprocessor.preprocess(shader_src, تعریفوي);
        خپلکاری pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_اندازه = context.max_wg_اندازه;
        pipeline_decisions->inplace = کلی.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        جمع_کړه_id_pipelines[کلی]    = pipeline;
        أرجع pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_concat_pipeline_کلی کلی = {};
        کلی.type                            = context.dst->type;
        کلی.src_overlap                     = src_overlap;

        خپلکاری it = concat_pipelines.find(کلی);
        إذا (it != concat_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "concat";

        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریفوي.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for concat shader");
        }

        إذا (کلی.src_overlap) {
            تعریفوي.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_concat, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_دوګونی_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->src_overlap   = کلی.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[کلی]    = pipeline;
        أرجع concat_pipelines[کلی];
    }

    webgpu_pipeline get_repeat_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_کلی کلی = {};
        کلی.type                            = context.dst->type;

        خپلکاری it = repeat_pipelines.find(کلی);
        إذا (it != repeat_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "repeat";

        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریفوي.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            حالة GGML_TYPE_I16:
                تعریفوي.push_back("TYPE_I16");
                variant += "_i16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_repeat, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[کلی]    = pipeline;
        أرجع repeat_pipelines[کلی];
    }

    webgpu_pipeline get_flash_attn_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ثابت منطقي can_use_subgroup_ماتریکس = ggml_webgpu_flash_attn_can_use_subgroup_ماتریکس_لار(
            context.supports_subgroup_ماتریکس, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_ماتریکس                    = can_use_subgroup_ماتریکس;
        decisions.q_tile = decisions.use_sg_ماتریکس ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_کلی کلی = {};
        کلی.common                              = ggml_webgpu_flash_attn_make_common_pipeline_کلی(
            context, decisions.use_sg_ماتریکس ? context.sg_mat_k : 1u, kv_overlap);
        کلی.common.k_مستقیم &= decisions.use_sg_ماتریکس && کلی.common.k_type == GGML_TYPE_F16;
        کلی.common.v_مستقیم &= decisions.use_sg_ماتریکس && کلی.common.v_type == GGML_TYPE_F16;
        کلی.use_sg_ماتریکس = decisions.use_sg_ماتریکس;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_حد_بایټونه, decisions.q_tile, decisions.use_sg_ماتریکس ? context.sg_mat_n : 1u,
            کلی.common.سر_dim_qk, کلی.common.سر_dim_v, کلی.common.has_ماسک,
            کلی.common.k_مستقیم || کلی.common.v_مستقیم);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        إذا (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_ماتریکس);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_ماتریکس = خطأ;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            کلی.common.k_مستقیم     = خطأ;
            کلی.common.v_مستقیم     = خطأ;
            کلی.use_sg_ماتریکس       = خطأ;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_حد_بایټونه, decisions.q_tile, 1u, کلی.common.سر_dim_qk, کلی.common.سر_dim_v,
                کلی.common.has_ماسک, کلی.common.k_مستقیم || کلی.common.v_مستقیم);
            GGML_ASSERT(max_kv_tile > 0);
        }

        decisions.kv_tile = decisions.use_sg_ماتریکس ?
                                std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES) :
                                std::min(GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE, max_kv_tile);
        decisions.wg_اندازه =
            decisions.use_sg_ماتریکس ?
                std::max(context.max_subgroup_اندازه, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_اندازه) :
                std::min(context.max_wg_اندازه, std::max(GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_اندازه,
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * context.max_subgroup_اندازه));

        إذا (کلی.common.k_مستقیم || کلی.common.v_مستقیم) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            طالما (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_ماتریکس ? context.sg_mat_n : context.min_subgroup_اندازه;
            }
        }

        خپلکاری it = flash_attn_pipelines.find(کلی);
        إذا (it != flash_attn_pipelines.end()) {
            أرجع it->second;
        }

        std::string              variant = decisions.use_sg_ماتریکس ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> تعریفوي = ggml_webgpu_flash_attn_common_تعریفوي(کلی.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_اندازه);
        ثابت حرف *             shader_src = تش_اشاره;
        إذا (!کلی.use_sg_ماتریکس) {
            shader_src = wgsl_flash_attn_tile;
            تعریفوي.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_اندازه) + "u");
            تعریفوي.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_اندازه) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_اندازه) + "_" +
                       std::to_string(context.max_subgroup_اندازه);
        } وإلا {
            shader_src = wgsl_flash_attn;
            تعریفوي.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            تعریفوي.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            تعریفوي.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        خپلکاری            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, تعریفوي), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[کلی] = pipeline;
        أرجع flash_attn_pipelines[کلی];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_کلی کلی = {};
        کلی.common = ggml_webgpu_flash_attn_make_common_pipeline_کلی(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_پلنوالی,
                                                                     kv_overlap);

        خپلکاری it = flash_attn_vec_pipelines.find(کلی);
        إذا (it != flash_attn_vec_pipelines.end()) {
            أرجع it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_حد_بایټونه, کلی.common.سر_dim_qk,
                                                                   کلی.common.سر_dim_v, کلی.common.has_ماسک,
                                                                   کلی.common.k_مستقیم || کلی.common.v_مستقیم);
        decisions.wg_اندازه = context.max_subgroup_اندازه;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> تعریفوي =
            ggml_webgpu_flash_attn_common_تعریفوي(کلی.common, variant, 1u, decisions.kv_tile, decisions.wg_اندازه);
        إذا (کلی.common.has_ماسک) {
            تعریفوي.push_back("BLK");
            variant.resize(variant.size() - (حجم("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_وېشل = context.min_subgroup_اندازه;
        ثابت uint32_t D       = کلی.common.سر_dim_qk | کلی.common.سر_dim_v;
        ثابت uint32_t D_lsb   = D & (~(D - 1u));
        d_وېشل                = std::min(std::min(context.min_subgroup_اندازه, 4u), std::max(D_lsb / 4u, 1u));

        تعریفوي.push_back(std::string("D_SPLIT=") + std::to_string(d_وېشل));
        variant += "_dsplit" + std::to_string(d_وېشل);

        خپلکاری            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_وېشل, تعریفوي), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[کلی] = pipeline;
        أرجع flash_attn_vec_pipelines[کلی];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_کلی کلی = {};
        کلی.kv_tile                                 = kv_tile;
        خپلکاری it                                     = flash_attn_blk_pipelines.find(کلی);
        إذا (it != flash_attn_blk_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "flash_attn_vec_blk";

        تعریفوي.push_back(std::string("KV_TILE=") + std::to_string(کلی.kv_tile));
        variant += std::string("_kvt") + std::to_string(کلی.kv_tile);

        uint32_t wg_اندازه = 1;
        طالما ((wg_اندازه << 1) <= context.max_wg_اندازه) {
            wg_اندازه <<= 1;
        }
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(wg_اندازه));
        variant += std::string("_wg") + std::to_string(wg_اندازه);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, تعریفوي), variant);
        flash_attn_blk_pipelines[کلی] = pipeline;
        أرجع flash_attn_blk_pipelines[کلی];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_کلی کلی = {};
        کلی.سر_dim_v                                     = (uint32_t) context.src2->ne[0];
        کلی.dst_type                                       = context.dst->type;
        کلی.wg_اندازه                                        = context.max_wg_اندازه;
        خپلکاری it                                            = flash_attn_vec_reduce_pipelines.find(کلی);
        إذا (it != flash_attn_vec_reduce_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "flash_attn_vec_reduce";

        بدل (کلی.dst_type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("DST_F32");
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("DST_F16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(کلی.dst_type);

        تعریفوي.push_back(std::string("HEAD_DIM_V=") + std::to_string(کلی.سر_dim_v));
        variant += std::string("_hsv") + std::to_string(کلی.سر_dim_v);

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));
        variant += std::string("_wg") + std::to_string(context.max_wg_اندازه);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, تعریفوي), variant);
        flash_attn_vec_reduce_pipelines[کلی] = pipeline;
        أرجع flash_attn_vec_reduce_pipelines[کلی];
    }

    webgpu_pipeline get_cpy_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_کلی کلی = {};
        کلی.src_type                     = context.src0->type;
        کلی.dst_type                     = context.dst->type;

        خپلکاری it = cpy_pipelines.find(کلی);
        إذا (it != cpy_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "cpy";

        بدل (کلی.src_type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("SRC_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("SRC_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریفوي.push_back("SRC_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        بدل (کلی.dst_type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("DST_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("DST_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                تعریفوي.push_back("DST_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_cpy, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[کلی]       = pipeline;
        أرجع cpy_pipelines[کلی];
    }

    webgpu_pipeline get_glu_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_glu_pipeline_کلی کلی = {};
        کلی.glu_op                       = ggml_get_glu_op(context.dst);
        کلی.type                         = context.dst->type;
        کلی.وېشل                        = (context.src1 != تش_اشاره);
        کلی.src_overlap                  = src_overlap;

        خپلکاری it = glu_pipelines.find(کلی);
        إذا (it != glu_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "glu";

        بدل (کلی.glu_op) {
            حالة GGML_GLU_OP_REGLU:
                تعریفوي.push_back("OP_REGLU");
                variant += "_reglu";
                اكسر;
            حالة GGML_GLU_OP_GEGLU:
                تعریفوي.push_back("OP_GEGLU");
                variant += "_geglu";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU:
                تعریفوي.push_back("OP_SWIGLU");
                variant += "_swiglu";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU_OAI:
                تعریفوي.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                اكسر;
            حالة GGML_GLU_OP_GEGLU_ERF:
                تعریفوي.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                اكسر;
            حالة GGML_GLU_OP_GEGLU_چټک:
                تعریفوي.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU_CLAMP:
                تعریفوي.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported GLU op");
        }
        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        إذا (کلی.src_overlap) {
            تعریفوي.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } وإلا إذا (کلی.وېشل) {
            variant += "_split";
        } وإلا {
            تعریفوي.push_back("NO_SPLIT");
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_glu, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->src_overlap   = کلی.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[کلی]       = pipeline;
        أرجع glu_pipelines[کلی];
    }

    webgpu_pipeline get_rope_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_کلی کلی = {};
        کلی.type                          = context.dst->type;
        کلی.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        کلی.has_ff                        = (context.src2 != تش_اشاره);

        خپلکاری it = rope_pipelines.find(کلی);
        إذا (it != rope_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "rope";

        بدل (کلی.type) {
            حالة GGML_TYPE_F32:
                تعریفوي.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                تعریفوي.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        إذا (کلی.inplace) {
            تعریفوي.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (کلی.has_ff) {
            تعریفوي.push_back("FF_FUNC");
            variant += "_ff";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_rope, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->inplace       = کلی.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[کلی]      = pipeline;
        أرجع rope_pipelines[کلی];
    }

    webgpu_pipeline get_soft_max_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_کلی کلی = {};
        کلی.ماسک_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        کلی.has_ماسک                          = (context.src1 != تش_اشاره);
        کلی.has_sink                          = (context.src2 != تش_اشاره);
        کلی.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خپلکاری it = soft_max_pipelines.find(کلی);
        إذا (it != soft_max_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "soft_max";

        إذا (کلی.has_ماسک) {
            تعریفوي.push_back("HAS_MASK");
            بدل (کلی.ماسک_type) {
                حالة GGML_TYPE_F32:
                    تعریفوي.push_back("MASK_F32");
                    variant += "_mask_f32";
                    اكسر;
                حالة GGML_TYPE_F16:
                    تعریفوي.push_back("MASK_F16");
                    variant += "_mask_f16";
                    اكسر;
                افتراضي:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        إذا (کلی.has_sink) {
            تعریفوي.push_back("HAS_SINK");
            variant += "_sink";
        }

        إذا (کلی.inplace) {
            تعریفوي.push_back("INPLACE");
            variant += "_inplace";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_soft_max, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        decisions->inplace       = کلی.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[کلی]  = pipeline;
        أرجع soft_max_pipelines[کلی];
    }

    webgpu_pipeline get_conv2d_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_کلی کلی = {};
        کلی.وزن_type                     = context.src0->type;
        کلی.ننوت_type                      = context.src1->type;
        کلی.output_type                     = context.dst->type;

        خپلکاری it = conv2d_pipelines.find(کلی);
        إذا (it != conv2d_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "conv_2d";

        خپلکاری ټېل_type_تعریفوي = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_مختاړی = prefix;
            إذا (type == GGML_TYPE_F32) {
                تعریفوي.push_back(s_مختاړی + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                تعریفوي.push_back(s_مختاړی + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ټېل_type_تعریفوي("WEIGHT_TYPE", کلی.وزن_type);
        ټېل_type_تعریفوي("INPUT_TYPE", کلی.ننوت_type);
        ټېل_type_تعریفوي("OUTPUT_TYPE", کلی.output_type);

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_conv2d, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[کلی]    = pipeline;
        أرجع conv2d_pipelines[کلی];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي whcn) {
        ggml_webgpu_conv2d_dw_pipeline_کلی کلی = {};
        کلی.وزن_type                        = context.src0->type;
        کلی.ننوت_type                         = context.src1->type;
        کلی.output_type                        = context.dst->type;
        کلی.whcn                               = whcn;

        خپلکاری it = conv2d_dw_pipelines.find(کلی);
        إذا (it != conv2d_dw_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        خپلکاری ټېل_type_تعریفوي = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_مختاړی = prefix;
            إذا (type == GGML_TYPE_F32) {
                تعریفوي.push_back(s_مختاړی + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                تعریفوي.push_back(s_مختاړی + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ټېل_type_تعریفوي("WEIGHT_TYPE", کلی.وزن_type);
        ټېل_type_تعریفوي("INPUT_TYPE", کلی.ننوت_type);
        ټېل_type_تعریفوي("OUTPUT_TYPE", کلی.output_type);

        إذا (whcn) {
            تعریفوي.push_back("WHCN");
        }
        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_conv2d_dw, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[کلی] = pipeline;
        أرجع conv2d_dw_pipelines[کلی];
    }

    webgpu_pipeline get_im2col_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_کلی کلی = {};
        کلی.ننوت_type                      = context.src1->type;
        کلی.output_type                     = context.dst->type;

        خپلکاری it = im2col_pipelines.find(کلی);
        إذا (it != im2col_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "im2col";

        خپلکاری ټېل_type_تعریفوي = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_مختاړی = prefix;
            إذا (type == GGML_TYPE_F32) {
                تعریفوي.push_back(s_مختاړی + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                تعریفوي.push_back(s_مختاړی + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ټېل_type_تعریفوي("INPUT_TYPE", کلی.ننوت_type);
        ټېل_type_تعریفوي("OUTPUT_TYPE", کلی.output_type);

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_im2col, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[کلی]    = pipeline;
        أرجع im2col_pipelines[کلی];
    }

    webgpu_pipeline get_upscale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت uint32_t ډېرپېښ_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ثابت uint32_t بنسټ_ډېرپېښ  = ډېرپېښ_flags & 0xFFu;
        ثابت منطقي     antialias  = (ډېرپېښ_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_کلی کلی = {};
        کلی.ننوت_type                       = context.src0->type;
        کلی.output_type                      = context.dst->type;
        کلی.بنسټ_ډېرپېښ                        = بنسټ_ډېرپېښ;
        کلی.antialias                        = antialias;

        خپلکاری it = upscale_pipelines.find(کلی);
        إذا (it != upscale_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> تعریفوي;
        std::string              variant = "upscale";

        إذا (کلی.ننوت_type == GGML_TYPE_F16) {
            تعریفوي.push_back("SRC_F16");
            variant += "_src_f16";
        } وإلا {
            variant += "_src_f32";
        }

        إذا (کلی.output_type == GGML_TYPE_F16) {
            تعریفوي.push_back("DST_F16");
            variant += "_dst_f16";
        } وإلا {
            variant += "_dst_f32";
        }

        بدل (بنسټ_ډېرپېښ) {
            حالة GGML_SCALE_ډېرپېښ_تر_ټولو_نږدې:
                تعریفوي.push_back("NEAREST");
                variant += "_nearest";
                اكسر;
            حالة GGML_SCALE_ډېرپېښ_BILINEAR:
                تعریفوي.push_back("BILINEAR");
                variant += "_bilinear";
                اكسر;
            حالة GGML_SCALE_ډېرپېښ_BICUBIC:
                تعریفوي.push_back("BICUBIC");
                variant += "_bicubic";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported upscale mode");
        }

        إذا (antialias) {
            تعریفوي.push_back("ANTIALIAS");
            variant += "_aa";
        }

        تعریفوي.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_اندازه));

        خپلکاری processed           = preprocessor.preprocess(wgsl_upscale, تعریفوي);
        خپلکاری decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_اندازه       = context.max_wg_اندازه;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[کلی]   = pipeline;
        أرجع upscale_pipelines[کلی];
    }

  خاص:
    ساكن webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_کوډ,
                                                       std::string    label) {
        wgpu::Shader_سرچینه_WGSL shader_سرچینه;
        shader_سرچینه.code = shader_کوډ.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.بل_دننه_Chain = &shader_سرچینه;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.ثبت_ټکی_2 = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = تش_اشاره;  // nullptr means auto layout
        أرجع { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#د_که_پای  // GGML_WEBGPU_SHADER_LIB_HPP
