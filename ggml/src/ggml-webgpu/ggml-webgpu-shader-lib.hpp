#اگر_غیر_تعریف_شدہ GGML_WEBGPU_SHADER_LIB_HPP
#تعریف_کرو GGML_WEBGPU_SHADER_LIB_HPP

#شامل "ggml-impl.h"
#شامل "ggml-wgsl-shaders.hpp"
#شامل "ggml.h"
#شامل "pre_wgsl.hpp"

#شامل <webgpu/webgpu_cpp.h>

#شامل <algorithm>
#شامل <memory>
#شامل <string>
#شامل <unordered_map>
#شامل <vector>

#تعریف_کرو GGML_WEBGPU_F16_سائز_بائٹیں                   2
#تعریف_کرو GGML_WEBGPU_F32_سائز_بائٹیں                   4
#تعریف_کرو GGML_WEBGPU_I32_سائز_بائٹیں                   4
#تعریف_کرو GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#تعریف_کرو GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#تعریف_کرو GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#تعریف_کرو GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#تعریف_کرو GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_سائز     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#تعریف_کرو GGML_WEBGPU_KV_SEQ_PAD                       256u

#تعریف_کرو GGML_WEBGPU_ARGSORT_ملاؤ_MAX_WG_سائز 512u

// Matrix multiplication parameters

// Register tiling parameters
#تعریف_کرو WEBGPU_MUL_MAT_TILE_M           4
#تعریف_کرو WEBGPU_MUL_MAT_TILE_N           4
#تعریف_کرو WEBGPU_MUL_MAT_WG_سائز_M        8
#تعریف_کرو WEBGPU_MUL_MAT_WG_سائز_N        8
#تعریف_کرو WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#تعریف_کرو WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#تعریف_کرو WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#تعریف_کرو WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#تعریف_کرو WEBGPU_MUL_MAT_SUBGROUP_میٹرکس_M     4
#تعریف_کرو WEBGPU_MUL_MAT_SUBGROUP_میٹرکس_N     2
#تعریف_کرو WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#تعریف_کرو WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#تعریف_کرو WEBGPU_MUL_MAT_VEC_WG_سائز 256

#تعریف_کرو WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_فی_WG    4
#تعریف_کرو WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG 4
#تعریف_کرو WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_فی_WG      4

// default size for reg-tile matrix multiplication
#تعریف_کرو WEBGPU_MUL_MAT_WG_سائز 256

// Same hash combine function as in boost
سانچہ <قسم_نام T> اندرونی خالی ggml_webgpu_hash_combine(size_t & بیج, ثابت T & value) {
    بیج ^= std::hash<T>{}(value) + 0x9e3779b9 + (بیج << 6) + (بیج >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
اندرونی uintptr_t ggml_webgpu_tensor_addr(ثابت ggml_tensor * tensor) {
    ثابت ggml_tensor * بنیاد_tensor = tensor->view_src ? tensor->view_src : tensor;
    واپس (uintptr_t) بنیاد_tensor->ڈیٹا + tensor->view_offs;
}

اندرونی منطقی ggml_webgpu_tensor_equal(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    واپس a->بفر == b->بفر && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

ساخت ggml_webgpu_shader_lib_context {
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * src2;
    ggml_tensor * src3;
    ggml_tensor * src4;
    ggml_tensor * src5;
    ggml_tensor * dst;
    ggml_tensor * dst_fuse;

    uint32_t    max_wg_سائز;
    size_t      wg_mem_حد_بندی_بائٹیں       = 0;
    منطقی        supports_subgroups       = جھوٹ;
    منطقی        supports_subgroup_میٹرکس = جھوٹ;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_سائز        = 0;
    uint32_t    max_subgroup_سائز        = 0;
    منطقی        supports_dot_حاصل_ضرب     = جھوٹ;
    std::string vendor;
};

ساخت webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<خالی> context = خالی_اشاریہ;
};

ساخت ggml_webgpu_generic_shader_decisions {
    uint32_t wg_سائز = 0;
    منطقی     inplace = جھوٹ;
};

ساخت ggml_webgpu_get_صفیں_shader_decisions {
    uint32_t wg_سائز    = 0;
    منطقی     vectorized = جھوٹ;
};

ساخت ggml_webgpu_ثنائی_shader_decisions {
    uint32_t wg_سائز     = 0;
    منطقی     inplace     = جھوٹ;
    منطقی     overlap     = جھوٹ;
    منطقی     src_overlap = جھوٹ;
};

ساخت ggml_webgpu_glu_shader_decisions {
    uint32_t wg_سائز     = 0;
    منطقی     src_overlap = جھوٹ;
};

ساخت ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<خالی> decisions;
};

ساخت ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t بلاک_سائز_3;
    uint32_t علامات_فی_wg;
};

ساخت ggml_webgpu_ssm_جائزہ_pipeline_کلید {
    صحیح     type;
    صحیح     d_state;
    uint8_t xbc_overlap;
    منطقی    a_overlap;
    منطقی    ids_overlap;

    منطقی عامل==(ثابت ggml_webgpu_ssm_جائزہ_pipeline_کلید & other) ثابت {
        واپس type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

ساخت ggml_webgpu_ssm_جائزہ_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_ssm_جائزہ_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.d_state);
        ggml_webgpu_hash_combine(بیج, کلید.xbc_overlap);
        ggml_webgpu_hash_combine(بیج, کلید.a_overlap);
        ggml_webgpu_hash_combine(بیج, کلید.ids_overlap);
        واپس بیج;
    }
};

ساخت ggml_webgpu_ssm_جائزہ_shader_decisions {
    uint32_t wg_سائز;
    uint32_t علامات_فی_tile;
    uint8_t  xbc_overlap = 0;
    منطقی     a_overlap   = جھوٹ;
    منطقی     ids_overlap = جھوٹ;
};

/** Argsort **/

ساخت ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_سائز;
    size_t   wg_mem_حد_بندی_بائٹیں;
    int32_t  ترتیب;
};

/** Set Rows **/

ساخت ggml_webgpu_سیٹ_صفیں_pipeline_کلید {
    صحیح src0_type;
    صحیح dst_type;
    صحیح vec4;
    صحیح i64_idx;
    صحیح pair_بلاکیں;

    منطقی عامل==(ثابت ggml_webgpu_سیٹ_صفیں_pipeline_کلید & other) ثابت {
        واپس src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_بلاکیں == other.pair_بلاکیں;
    }
};

ساخت ggml_webgpu_سیٹ_صفیں_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_سیٹ_صفیں_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src0_type);
        ggml_webgpu_hash_combine(بیج, کلید.dst_type);
        ggml_webgpu_hash_combine(بیج, کلید.vec4);
        ggml_webgpu_hash_combine(بیج, کلید.i64_idx);
        ggml_webgpu_hash_combine(بیج, کلید.pair_بلاکیں);
        واپس بیج;
    }
};

ساخت ggml_webgpu_سیٹ_صفیں_shader_decisions {
    منطقی     vec4;
    منطقی     i64_idx;
    منطقی     pair_بلاکیں;
    uint32_t wg_سائز;
};

/** Set **/

ساخت ggml_webgpu_سیٹ_pipeline_کلید {
    ggml_type type;
    منطقی      inplace;

    منطقی عامل==(ثابت ggml_webgpu_سیٹ_pipeline_کلید & other) ثابت {
        واپس type == other.type && inplace == other.inplace;
    }
};

ساخت ggml_webgpu_سیٹ_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_سیٹ_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        واپس بیج;
    }
};

/** Get Rows **/

ساخت ggml_webgpu_get_صفیں_pipeline_کلید {
    ggml_type src_type;
    صحیح       vectorized;

    منطقی عامل==(ثابت ggml_webgpu_get_صفیں_pipeline_کلید & other) ثابت {
        واپس src_type == other.src_type && vectorized == other.vectorized;
    }
};

ساخت ggml_webgpu_get_صفیں_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_get_صفیں_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src_type);
        ggml_webgpu_hash_combine(بیج, کلید.vectorized);
        واپس بیج;
    }
};

/** Row Norm **/

ساخت ggml_webgpu_صف_norm_pipeline_کلید {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    منطقی      inplace;

    منطقی عامل==(ثابت ggml_webgpu_صف_norm_pipeline_کلید & other) ثابت {
        واپس op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

ساخت ggml_webgpu_صف_norm_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_صف_norm_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.op);
        ggml_webgpu_hash_combine(بیج, کلید.src_type);
        ggml_webgpu_hash_combine(بیج, کلید.dst_type);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        واپس بیج;
    }
};

/** RMS_NORM + MUL **/

ساخت ggml_webgpu_rms_norm_mul_pipeline_کلید {
    منطقی inplace;      // rn_src == dst
    منطقی overlap;      // mul_src == dst
    منطقی src_overlap;  // rn_src binding overlaps mul_src binding

    منطقی عامل==(ثابت ggml_webgpu_rms_norm_mul_pipeline_کلید & other) ثابت {
        واپس inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

ساخت ggml_webgpu_rms_norm_mul_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_rms_norm_mul_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        ggml_webgpu_hash_combine(بیج, کلید.overlap);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        واپس بیج;
    }
};

ساخت ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_سائز     = 0;
    منطقی     inplace     = جھوٹ;
    منطقی     overlap     = جھوٹ;
    منطقی     src_overlap = جھوٹ;
};

/** Pad **/
ساخت ggml_webgpu_pad_pipeline_کلید {
    منطقی circular;

    منطقی عامل==(ثابت ggml_webgpu_pad_pipeline_کلید & other) ثابت { واپس circular == other.circular; }
};

ساخت ggml_webgpu_pad_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_pad_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.circular);
        واپس بیج;
    }
};

/** Solve Tri **/
ساخت ggml_webgpu_solve_tri_pipeline_کلید {
    صحیح type;
    صحیح n;
    صحیح k;

    منطقی عامل==(ثابت ggml_webgpu_solve_tri_pipeline_کلید & other) ثابت {
        واپس type == other.type && n == other.n && k == other.k;
    }
};

ساخت ggml_webgpu_solve_tri_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_solve_tri_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.n);
        ggml_webgpu_hash_combine(بیج, کلید.k);
        واپس بیج;
    }
};

/** SSM Conv **/
ساخت ggml_webgpu_ssm_conv_pipeline_کلید {
    صحیح type;
    صحیح vectorized;

    منطقی عامل==(ثابت ggml_webgpu_ssm_conv_pipeline_کلید & other) ثابت {
        واپس type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
ساخت ggml_webgpu_conv2d_pipeline_کلید {
    ggml_type وزن_type;
    ggml_type داخلہ_type;
    ggml_type output_type;

    منطقی عامل==(ثابت ggml_webgpu_conv2d_pipeline_کلید & other) ثابت {
        واپس وزن_type == other.وزن_type && داخلہ_type == other.داخلہ_type && output_type == other.output_type;
    }
};

ساخت ggml_webgpu_conv2d_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_conv2d_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.وزن_type);
        ggml_webgpu_hash_combine(بیج, کلید.داخلہ_type);
        ggml_webgpu_hash_combine(بیج, کلید.output_type);
        واپس بیج;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
ساخت ggml_webgpu_conv2d_dw_pipeline_کلید {
    ggml_type وزن_type;
    ggml_type داخلہ_type;
    ggml_type output_type;
    منطقی      whcn;

    منطقی عامل==(ثابت ggml_webgpu_conv2d_dw_pipeline_کلید & other) ثابت {
        واپس وزن_type == other.وزن_type && داخلہ_type == other.داخلہ_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

ساخت ggml_webgpu_conv2d_dw_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_conv2d_dw_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.وزن_type);
        ggml_webgpu_hash_combine(بیج, کلید.داخلہ_type);
        ggml_webgpu_hash_combine(بیج, کلید.output_type);
        ggml_webgpu_hash_combine(بیج, کلید.whcn);
        واپس بیج;
    }
};

/** Im2Col **/
ساخت ggml_webgpu_im2col_pipeline_کلید {
    ggml_type داخلہ_type;
    ggml_type output_type;

    منطقی عامل==(ثابت ggml_webgpu_im2col_pipeline_کلید & other) ثابت {
        واپس داخلہ_type == other.داخلہ_type && output_type == other.output_type;
    }
};

ساخت ggml_webgpu_im2col_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_im2col_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.داخلہ_type);
        ggml_webgpu_hash_combine(بیج, کلید.output_type);
        واپس بیج;
    }
};

/** Gated Delta Net **/
ساخت ggml_webgpu_gated_delta_net_pipeline_کلید {
    صحیح  type;
    صحیح  s_v;
    صحیح  kda;
    منطقی fused_cache;

    منطقی عامل==(ثابت ggml_webgpu_gated_delta_net_pipeline_کلید & other) ثابت {
        واپس type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

ساخت ggml_webgpu_gated_delta_net_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_gated_delta_net_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.s_v);
        ggml_webgpu_hash_combine(بیج, کلید.kda);
        واپس بیج;
    }
};

ساخت ggml_webgpu_ssm_conv_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_ssm_conv_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.vectorized);
        واپس بیج;
    }
};

/** Scale **/

ساخت ggml_webgpu_scale_pipeline_کلید {
    صحیح inplace;

    منطقی عامل==(ثابت ggml_webgpu_scale_pipeline_کلید & other) ثابت { واپس inplace == other.inplace; }
};

ساخت ggml_webgpu_scale_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_scale_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        واپس بیج;
    }
};

/** Upscale **/

ساخت ggml_webgpu_upscale_pipeline_کلید {
    ggml_type داخلہ_type;
    ggml_type output_type;
    uint32_t  بنیاد_منوال;
    منطقی      antialias;

    منطقی عامل==(ثابت ggml_webgpu_upscale_pipeline_کلید & other) ثابت {
        واپس داخلہ_type == other.داخلہ_type && output_type == other.output_type && بنیاد_منوال == other.بنیاد_منوال &&
               antialias == other.antialias;
    }
};

ساخت ggml_webgpu_upscale_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_upscale_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.داخلہ_type);
        ggml_webgpu_hash_combine(بیج, کلید.output_type);
        ggml_webgpu_hash_combine(بیج, کلید.بنیاد_منوال);
        ggml_webgpu_hash_combine(بیج, کلید.antialias);
        واپس بیج;
    }
};

/** Concat **/

ساخت ggml_webgpu_concat_pipeline_کلید {
    صحیح  type;
    منطقی src_overlap;

    منطقی عامل==(ثابت ggml_webgpu_concat_pipeline_کلید & other) ثابت {
        واپس type == other.type && src_overlap == other.src_overlap;
    }
};

ساخت ggml_webgpu_concat_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_concat_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        واپس بیج;
    }
};

/** Repeat **/

ساخت ggml_webgpu_repeat_pipeline_کلید {
    صحیح type;

    منطقی عامل==(ثابت ggml_webgpu_repeat_pipeline_کلید & other) ثابت { واپس type == other.type; }
};

ساخت ggml_webgpu_repeat_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_repeat_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        واپس بیج;
    }
};

/** Binary **/

ساخت ggml_webgpu_ثنائی_pipeline_کلید {
    صحیح  type;
    صحیح  op;
    منطقی inplace;
    منطقی overlap;
    منطقی src_overlap;

    منطقی عامل==(ثابت ggml_webgpu_ثنائی_pipeline_کلید & other) ثابت {
        واپس type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

ساخت ggml_webgpu_ثنائی_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_ثنائی_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.op);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        ggml_webgpu_hash_combine(بیج, کلید.overlap);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        واپس بیج;
    }
};

/* Add_Id */

ساخت ggml_webgpu_جوڑو_id_pipeline_کلید {
    منطقی inplace;

    منطقی عامل==(ثابت ggml_webgpu_جوڑو_id_pipeline_کلید & other) ثابت { واپس inplace == other.inplace; }
};

ساخت ggml_webgpu_جوڑو_id_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_جوڑو_id_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        واپس بیج;
    }
};

/** Unary **/

ساخت ggml_webgpu_unary_pipeline_کلید {
    صحیح           type;
    صحیح           op;
    منطقی          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    منطقی          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    منطقی عامل==(ثابت ggml_webgpu_unary_pipeline_کلید & other) ثابت {
        واپس type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

ساخت ggml_webgpu_unary_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_unary_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.op);
        ggml_webgpu_hash_combine(بیج, کلید.is_unary);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        ggml_webgpu_hash_combine(بیج, کلید.ttype);
        واپس بیج;
    }
};

/** FlashAttention */

ساخت ggml_webgpu_flash_attn_common_pipeline_کلید {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  سر_dim_qk;
    uint32_t  سر_dim_v;
    منطقی      k_براہ_راست;
    منطقی      v_براہ_راست;
    منطقی      kv_overlap;
    منطقی      has_نقاب;
    منطقی      has_sinks;
    منطقی      استعمال_کرتا_logit_softcap;

    منطقی عامل==(ثابت ggml_webgpu_flash_attn_common_pipeline_کلید & other) ثابت {
        واپس q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && سر_dim_qk == other.سر_dim_qk && سر_dim_v == other.سر_dim_v &&
               k_براہ_راست == other.k_براہ_راست && v_براہ_راست == other.v_براہ_راست && kv_overlap == other.kv_overlap &&
               has_نقاب == other.has_نقاب && has_sinks == other.has_sinks &&
               استعمال_کرتا_logit_softcap == other.استعمال_کرتا_logit_softcap;
    }
};

اندرونی خالی ggml_webgpu_flash_attn_hash_common_pipeline_کلید(size_t &                                           بیج,
                                                            ثابت ggml_webgpu_flash_attn_common_pipeline_کلید & کلید) {
    ggml_webgpu_hash_combine(بیج, کلید.q_type);
    ggml_webgpu_hash_combine(بیج, کلید.k_type);
    ggml_webgpu_hash_combine(بیج, کلید.v_type);
    ggml_webgpu_hash_combine(بیج, کلید.dst_type);
    ggml_webgpu_hash_combine(بیج, کلید.سر_dim_qk);
    ggml_webgpu_hash_combine(بیج, کلید.سر_dim_v);
    ggml_webgpu_hash_combine(بیج, کلید.k_براہ_راست);
    ggml_webgpu_hash_combine(بیج, کلید.v_براہ_راست);
    ggml_webgpu_hash_combine(بیج, کلید.kv_overlap);
    ggml_webgpu_hash_combine(بیج, کلید.has_نقاب);
    ggml_webgpu_hash_combine(بیج, کلید.has_sinks);
    ggml_webgpu_hash_combine(بیج, کلید.استعمال_کرتا_logit_softcap);
}

ساخت ggml_webgpu_flash_attn_vec_pipeline_کلید {
    ggml_webgpu_flash_attn_common_pipeline_کلید common;

    منطقی عامل==(ثابت ggml_webgpu_flash_attn_vec_pipeline_کلید & other) ثابت { واپس common == other.common; }
};

ساخت ggml_webgpu_flash_attn_vec_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_vec_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_کلید(بیج, کلید.common);
        واپس بیج;
    }
};

ساخت ggml_webgpu_flash_attn_pipeline_کلید {
    ggml_webgpu_flash_attn_common_pipeline_کلید common;
    منطقی                                       use_sg_میٹرکس;

    منطقی عامل==(ثابت ggml_webgpu_flash_attn_pipeline_کلید & other) ثابت {
        واپس common == other.common && use_sg_میٹرکس == other.use_sg_میٹرکس;
    }
};

ساخت ggml_webgpu_flash_attn_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_کلید(بیج, کلید.common);
        ggml_webgpu_hash_combine(بیج, کلید.use_sg_میٹرکس);
        واپس بیج;
    }
};

ساخت ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_سائز = 0;
};

ساخت ggml_webgpu_flash_attn_decisions {
    منطقی     use_sg_میٹرکس = جھوٹ;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_سائز       = 0;
};

اندرونی constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی = 4u;
اندرونی constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

اندرونی size_t ggml_webgpu_flash_attn_tensor_انحراف(ثابت ggml_tensor * tensor) {
    constexpr uintptr_t ptr_بنیاد_addr = 0x1000u;
    ثابت ggml_tensor * بنیاد          = tensor->view_src != خالی_اشاریہ ? tensor->view_src : tensor;
    واپس reinterpret_cast<uintptr_t>(بنیاد->ڈیٹا) - ptr_بنیاد_addr + tensor->view_offs;
}

اندرونی منطقی ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K, size_t storage_انحراف_alignment) {
    ثابت uint32_t انحراف_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_انحراف(K) & (storage_انحراف_alignment - 1)) /
                    ggml_type_سائز(K->type));
    واپس انحراف_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی == 0u;
}

اندرونی منطقی ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K,
                                                      ثابت ggml_tensor * V,
                                                      size_t              storage_انحراف_alignment) {
    واپس ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_انحراف_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_انحراف_alignment);
}

اندرونی منطقی ggml_webgpu_flash_attn_k_براہ_راست(ثابت ggml_tensor * Q, ثابت ggml_tensor * K, uint32_t kv_براہ_راست_align) {
    واپس (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_براہ_راست_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

اندرونی منطقی ggml_webgpu_flash_attn_v_براہ_راست(ثابت ggml_tensor * Q, ثابت ggml_tensor * V, uint32_t kv_براہ_راست_align) {
    واپس ggml_webgpu_flash_attn_k_براہ_راست(Q, V, kv_براہ_راست_align);
}

اندرونی ggml_webgpu_flash_attn_common_pipeline_کلید ggml_webgpu_flash_attn_make_common_pipeline_کلید(
    ثابت ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_براہ_راست_align,
    منطقی                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_کلید کلید = {};
    کلید.q_type                                     = context.src0->type;
    کلید.k_type                                     = context.src1->type;
    کلید.v_type                                     = context.src2->type;
    کلید.dst_type                                   = context.dst->type;
    کلید.سر_dim_qk                                = (uint32_t) context.src0->ne[0];
    کلید.سر_dim_v                                 = (uint32_t) context.src2->ne[0];
    کلید.k_براہ_راست           = ggml_webgpu_flash_attn_k_براہ_راست(context.src0, context.src1, kv_براہ_راست_align);
    کلید.v_براہ_راست           = ggml_webgpu_flash_attn_v_براہ_راست(context.src0, context.src2, kv_براہ_راست_align);
    کلید.kv_overlap         = kv_overlap;
    کلید.has_نقاب           = context.src3 != خالی_اشاریہ;
    کلید.has_sinks          = context.src4 != خالی_اشاریہ;
    کلید.استعمال_کرتا_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    واپس کلید;
}

اندرونی std::vector<std::string> ggml_webgpu_flash_attn_common_متعین_کرتا(
    ثابت ggml_webgpu_flash_attn_common_pipeline_کلید & کلید,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_سائز) {
    std::vector<std::string> متعین_کرتا;

    انتخاب (کلید.k_type) {
        حالت GGML_TYPE_F32:
            متعین_کرتا.push_back("K_F32");
            توڑو;
        حالت GGML_TYPE_F16:
            متعین_کرتا.push_back("K_F16");
            توڑو;
        حالت GGML_TYPE_Q4_0:
            متعین_کرتا.push_back("K_Q4_0");
            توڑو;
        حالت GGML_TYPE_Q8_0:
            متعین_کرتا.push_back("K_Q8_0");
            توڑو;
        طےشدہ:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(کلید.k_type);

    انتخاب (کلید.v_type) {
        حالت GGML_TYPE_F32:
            متعین_کرتا.push_back("V_F32");
            توڑو;
        حالت GGML_TYPE_F16:
            متعین_کرتا.push_back("V_F16");
            توڑو;
        حالت GGML_TYPE_Q4_0:
            متعین_کرتا.push_back("V_Q4_0");
            توڑو;
        حالت GGML_TYPE_Q8_0:
            متعین_کرتا.push_back("V_Q8_0");
            توڑو;
        طےشدہ:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(کلید.v_type);

    انتخاب (کلید.q_type) {
        حالت GGML_TYPE_F32:
            متعین_کرتا.push_back("Q_F32");
            توڑو;
        حالت GGML_TYPE_F16:
            متعین_کرتا.push_back("Q_F16");
            توڑو;
        طےشدہ:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(کلید.q_type);

    انتخاب (کلید.dst_type) {
        حالت GGML_TYPE_F32:
            متعین_کرتا.push_back("DST_F32");
            توڑو;
        حالت GGML_TYPE_F16:
            متعین_کرتا.push_back("DST_F16");
            توڑو;
        طےشدہ:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(کلید.dst_type);

    اگر (کلید.has_نقاب) {
        متعین_کرتا.push_back("MASK");
        variant += "_mask";
    }
    اگر (کلید.has_sinks) {
        متعین_کرتا.push_back("SINKS");
        variant += "_sinks";
    }
    اگر (کلید.استعمال_کرتا_logit_softcap) {
        متعین_کرتا.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    اگر (کلید.k_براہ_راست) {
        متعین_کرتا.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    اگر (کلید.v_براہ_راست) {
        متعین_کرتا.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    اگر (کلید.kv_overlap) {
        متعین_کرتا.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    متعین_کرتا.push_back(std::string("HEAD_DIM_QK=") + std::to_string(کلید.سر_dim_qk));
    variant += std::string("_hsqk") + std::to_string(کلید.سر_dim_qk);

    متعین_کرتا.push_back(std::string("HEAD_DIM_V=") + std::to_string(کلید.سر_dim_v));
    variant += std::string("_hsv") + std::to_string(کلید.سر_dim_v);

    متعین_کرتا.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    متعین_کرتا.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائز));

    اگر (ggml_is_quantized(کلید.k_type) || ggml_is_quantized(کلید.v_type)) {
        متعین_کرتا.push_back("U32_DEQUANT_HELPERS");
        اگر (ggml_is_quantized(کلید.k_type)) {
            متعین_کرتا.push_back("LOADERS_QUANTIZED_K");
        }
        اگر (ggml_is_quantized(کلید.v_type)) {
            متعین_کرتا.push_back("LOADERS_QUANTIZED_V");
        }
    }

    واپس متعین_کرتا;
}

ساخت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید {
    uint32_t  سر_dim_v;
    uint32_t  wg_سائز;
    ggml_type dst_type;
};

ساخت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.سر_dim_v);
        ggml_webgpu_hash_combine(بیج, کلید.wg_سائز);
        ggml_webgpu_hash_combine(بیج, کلید.dst_type);
        واپس بیج;
    }
};

اندرونی منطقی عامل==(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید & lhs,
                       ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید & rhs) {
    واپس lhs.سر_dim_v == rhs.سر_dim_v && lhs.wg_سائز == rhs.wg_سائز && lhs.dst_type == rhs.dst_type;
}

ساخت ggml_webgpu_flash_attn_blk_pipeline_کلید {
    uint32_t kv_tile;

    منطقی عامل==(ثابت ggml_webgpu_flash_attn_blk_pipeline_کلید & other) ثابت { واپس kv_tile == other.kv_tile; }
};

ساخت ggml_webgpu_flash_attn_blk_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_blk_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.kv_tile);
        واپس بیج;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
اندرونی size_t ggml_webgpu_flash_attn_wg_mem_بائٹیں(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t سر_dim_qk,
                                                  uint32_t سر_dim_v,
                                                  منطقی     has_نقاب,
                                                  منطقی     kv_براہ_راست) {
    ثابت uint32_t max_سر_dim = std::max(سر_dim_qk, سر_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * سر_dim_qk;        // q_shmem
    اگر (!kv_براہ_راست) {
        f32_elems += kv_tile * max_سر_dim;  // kv_shmem
    }
    f32_elems += q_tile * سر_dim_v;         // o_shmem
    اگر (has_نقاب) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    واپس f16_elems * GGML_WEBGPU_F16_سائز_بائٹیں + f32_elems * GGML_WEBGPU_F32_سائز_بائٹیں;
}

اندرونی uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   حد_بندی_بائٹیں,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t سر_dim_qk,
                                                   uint32_t سر_dim_v,
                                                   منطقی     has_نقاب,
                                                   منطقی     kv_براہ_راست) {
    ثابت size_t بنیاد_q_بائٹیں =
        ggml_webgpu_flash_attn_wg_mem_بائٹیں(q_tile, 0, سر_dim_qk, سر_dim_v, has_نقاب, kv_براہ_راست);
    اگر (حد_بندی_بائٹیں <= بنیاد_q_بائٹیں) {
        واپس 0;
    }
    ثابت size_t one_kv_بائٹیں =
        ggml_webgpu_flash_attn_wg_mem_بائٹیں(q_tile, 1, سر_dim_qk, سر_dim_v, has_نقاب, kv_براہ_راست);
    ثابت size_t بائٹیں_فی_kv = one_kv_بائٹیں - بنیاد_q_بائٹیں;
    اگر (بائٹیں_فی_kv == 0) {
        واپس 0;
    }
    ثابت size_t max_kv_tile = (حد_بندی_بائٹیں - بنیاد_q_بائٹیں) / بائٹیں_فی_kv;
    واپس (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

اندرونی uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_حد_بندی_بائٹیں,
                                                       uint32_t سر_dim_qk,
                                                       uint32_t سر_dim_v,
                                                       منطقی     has_نقاب,
                                                       منطقی     kv_براہ_راست) {
    ثابت uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_حد_بندی_بائٹیں, 1u, 1u, سر_dim_qk, سر_dim_v, has_نقاب, kv_براہ_راست);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    اگر (kv_براہ_راست) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        جب (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    واپس kv_tile;
}

اندرونی منطقی ggml_webgpu_flash_attn_can_use_subgroup_میٹرکس_راستہ(منطقی                supports_subgroup_میٹرکس,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ثابت ggml_tensor * Q,
                                                                ثابت ggml_tensor * V) {
    واپس supports_subgroup_میٹرکس && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

ساخت ggml_webgpu_mul_mat_vec_pipeline_کلید {
    ggml_type src0_type;
    ggml_type src1_type;
    صحیح       vectorized;
    uint32_t  عدد_ستون_تعداد;
    منطقی      use_mmvq;
    منطقی      src_overlap;

    منطقی عامل==(ثابت ggml_webgpu_mul_mat_vec_pipeline_کلید & other) ثابت {
        واپس src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               عدد_ستون_تعداد == other.عدد_ستون_تعداد && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

ساخت ggml_webgpu_mul_mat_vec_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_vec_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src0_type);
        ggml_webgpu_hash_combine(بیج, کلید.src1_type);
        ggml_webgpu_hash_combine(بیج, کلید.vectorized);
        ggml_webgpu_hash_combine(بیج, کلید.عدد_ستون_تعداد);
        ggml_webgpu_hash_combine(بیج, کلید.use_mmvq);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        واپس بیج;
    }
};

ساخت ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_سائز;
    uint32_t outputs_فی_wg;
    uint32_t vec_سائز;
    منطقی     src_overlap = جھوٹ;
};

ساخت ggml_webgpu_quantize_q8_pipeline_کلید {
    ggml_type src0_type;

    منطقی عامل==(ثابت ggml_webgpu_quantize_q8_pipeline_کلید & other) ثابت { واپس src0_type == other.src0_type; }
};

ساخت ggml_webgpu_quantize_q8_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_quantize_q8_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src0_type);
        واپس بیج;
    }
};

ساخت ggml_webgpu_mul_mat_pipeline_کلید {
    ggml_type src0_type;
    ggml_type src1_type;
    صحیح       vectorized;
    صحیح       use_subgroup_میٹرکس;
    منطقی      src_overlap;

    منطقی عامل==(ثابت ggml_webgpu_mul_mat_pipeline_کلید & other) ثابت {
        واپس src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_میٹرکس == other.use_subgroup_میٹرکس && src_overlap == other.src_overlap;
    }
};

ساخت ggml_webgpu_mul_mat_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src0_type);
        ggml_webgpu_hash_combine(بیج, کلید.src1_type);
        ggml_webgpu_hash_combine(بیج, کلید.vectorized);
        ggml_webgpu_hash_combine(بیج, کلید.use_subgroup_میٹرکس);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        واپس بیج;
    }
};

ساخت ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_سائز_m;
    uint32_t wg_سائز_n;
    uint32_t wg_سائز;
    uint32_t outputs_فی_wg;
    صحیح      use_subgroup_میٹرکس;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_میٹرکس_m;
    uint32_t subgroup_میٹرکس_n;

    uint32_t mul_mat_wg_سائز;
    منطقی     src_overlap = جھوٹ;
};

/** MUL_MAT_ID **/

ساخت ggml_webgpu_mul_mat_id_gather_pipeline_کلید {
    uint32_t n_experts;

    منطقی عامل==(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_کلید & other) ثابت {
        واپس n_experts == other.n_experts;
    }
};

ساخت ggml_webgpu_mul_mat_id_gather_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.n_experts);
        واپس بیج;
    }
};

ساخت ggml_webgpu_mul_mat_id_pipeline_کلید {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  عدد_ستون_تعداد;
    صحیح       vectorized;

    منطقی عامل==(ثابت ggml_webgpu_mul_mat_id_pipeline_کلید & other) ثابت {
        واپس src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               عدد_ستون_تعداد == other.عدد_ستون_تعداد && vectorized == other.vectorized;
    }
};

ساخت ggml_webgpu_mul_mat_id_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_id_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src0_type);
        ggml_webgpu_hash_combine(بیج, کلید.src1_type);
        ggml_webgpu_hash_combine(بیج, کلید.n_experts);
        ggml_webgpu_hash_combine(بیج, کلید.عدد_ستون_تعداد);
        ggml_webgpu_hash_combine(بیج, کلید.vectorized);
        واپس بیج;
    }
};

/** Cpy **/

ساخت ggml_webgpu_cpy_pipeline_کلید {
    ggml_type src_type;
    ggml_type dst_type;

    منطقی عامل==(ثابت ggml_webgpu_cpy_pipeline_کلید & other) ثابت {
        واپس src_type == other.src_type && dst_type == other.dst_type;
    }
};

ساخت ggml_webgpu_cpy_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_cpy_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src_type);
        ggml_webgpu_hash_combine(بیج, کلید.dst_type);
        واپس بیج;
    }
};

/** Glu **/

ساخت ggml_webgpu_glu_pipeline_کلید {
    ggml_glu_op glu_op;
    ggml_type   type;
    منطقی        بانٹو;
    منطقی        src_overlap;

    منطقی عامل==(ثابت ggml_webgpu_glu_pipeline_کلید & other) ثابت {
        واپس glu_op == other.glu_op && type == other.type && بانٹو == other.بانٹو && src_overlap == other.src_overlap;
    }
};

ساخت ggml_webgpu_glu_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_glu_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.glu_op);
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.بانٹو);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        واپس بیج;
    }
};

/** Rope **/

ساخت ggml_webgpu_rope_pipeline_کلید {
    ggml_type type;
    منطقی      inplace;
    منطقی      has_ff;

    منطقی عامل==(ثابت ggml_webgpu_rope_pipeline_کلید & other) ثابت {
        واپس type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

ساخت ggml_webgpu_rope_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_rope_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        ggml_webgpu_hash_combine(بیج, کلید.has_ff);
        واپس بیج;
    }
};

/** SoftMax **/

ساخت ggml_webgpu_soft_max_pipeline_کلید {
    ggml_type نقاب_type;
    منطقی      has_نقاب;
    منطقی      has_sink;
    منطقی      inplace;

    منطقی عامل==(ثابت ggml_webgpu_soft_max_pipeline_کلید & other) ثابت {
        واپس نقاب_type == other.نقاب_type && has_نقاب == other.has_نقاب && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

ساخت ggml_webgpu_soft_max_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_soft_max_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.نقاب_type);
        ggml_webgpu_hash_combine(بیج, کلید.has_نقاب);
        ggml_webgpu_hash_combine(بیج, کلید.has_sink);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        واپس بیج;
    }
};

/** MMVQ **/

اندرونی منطقی ggml_webgpu_can_use_mmvq(ثابت ggml_tensor * src0,
                                     ثابت ggml_tensor * src1,
                                     منطقی                supports_dot_حاصل_ضرب,
                                     ثابت std::string & vendor) {
    اگر (src1->ne[1] <= 4) {
        منطقی supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        اگر (supports_dp4a && supports_dot_حاصل_ضرب) {
            انتخاب (src1->type) {
                حالت GGML_TYPE_F32:
                    انتخاب (src0->type) {
                        حالت GGML_TYPE_Q1_0:
                        حالت GGML_TYPE_Q4_0:
                        حالت GGML_TYPE_Q4_1:
                        حالت GGML_TYPE_Q5_0:
                        حالت GGML_TYPE_Q5_1:
                        حالت GGML_TYPE_Q8_0:
                        حالت GGML_TYPE_MXFP4:
                        حالت GGML_TYPE_Q2_K:
                        حالت GGML_TYPE_Q3_K:
                        حالت GGML_TYPE_Q4_K:
                        حالت GGML_TYPE_Q5_K:
                        حالت GGML_TYPE_Q6_K:
                            واپس src0->ne[0] % 4 == 0;
                        طےشدہ:
                            توڑو;
                    }
                    توڑو;
                طےشدہ:
                    توڑو;
            }
        }
    }
    واپس جھوٹ;
}

جماعت ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<صحیح, webgpu_pipeline> مجموع_صفیں_pipelines;       // key is fixed, no variants yet
    std::unordered_map<صحیح, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<صحیح, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<صحیح, webgpu_pipeline> argsort_ملاؤ_pipelines;  // key is order
    std::unordered_map<صحیح, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_صف_norm_pipeline_کلید, webgpu_pipeline, ggml_webgpu_صف_norm_pipeline_کلید_hash>
        صف_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_صفیں_pipeline_کلید, webgpu_pipeline, ggml_webgpu_get_صفیں_pipeline_کلید_hash>
        get_صفیں_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_کلید, webgpu_pipeline, ggml_webgpu_unary_pipeline_کلید_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_کلید, webgpu_pipeline, ggml_webgpu_scale_pipeline_کلید_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_کلید, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_کلید_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_کلید, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_کلید_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_جائزہ_pipeline_کلید, webgpu_pipeline, ggml_webgpu_ssm_جائزہ_pipeline_کلید_hash>
        ssm_جائزہ_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_کلید,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_کلید_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_کلید, webgpu_pipeline, ggml_webgpu_pad_pipeline_کلید_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_ثنائی_pipeline_کلید, webgpu_pipeline, ggml_webgpu_ثنائی_pipeline_کلید_hash>
        ثنائی_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_جوڑو_id_pipeline_کلید, webgpu_pipeline, ggml_webgpu_جوڑو_id_pipeline_کلید_hash>
        جوڑو_id_pipelines;           // inplace
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

    std::unordered_map<ggml_webgpu_سیٹ_صفیں_pipeline_کلید, webgpu_pipeline, ggml_webgpu_سیٹ_صفیں_pipeline_کلید_hash>
        سیٹ_صفیں_pipelines;
    std::unordered_map<ggml_webgpu_سیٹ_pipeline_کلید, webgpu_pipeline, ggml_webgpu_سیٹ_pipeline_کلید_hash> سیٹ_pipelines;
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

  عوامی:
    ggml_webgpu_shader_lib(wgpu::Device device) { یہ->device = device; }

    webgpu_pipeline get_مجموع_صفیں_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        خودکار it = مجموع_صفیں_pipelines.find(1);
        اگر (it != مجموع_صفیں_pipelines.end()) {
            واپس it->second;
        }
        std::vector<std::string> متعین_کرتا;
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed        = preprocessor.preprocess(wgsl_مجموع_صفیں, متعین_کرتا);
        مجموع_صفیں_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        واپس مجموع_صفیں_pipelines[1];
    }

    webgpu_pipeline get_صف_norm_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_صف_norm_pipeline_کلید کلید = {};
        کلید.op                                = context.dst->op;
        کلید.src_type                          = context.src0->type;
        کلید.dst_type                          = context.dst->type;
        کلید.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = صف_norm_pipelines.find(کلید);
        اگر (it != صف_norm_pipelines.end()) {
            واپس it->second;
        }
        std::vector<std::string> متعین_کرتا;
        std::string              variant;

        انتخاب (کلید.op) {
            حالت GGML_OP_RMS_NORM:
                متعین_کرتا.push_back("RMS_NORM");
                variant = "rms_norm";
                توڑو;
            حالت GGML_OP_NORM:
                متعین_کرتا.push_back("NORM");
                variant = "norm";
                توڑو;
            حالت GGML_OP_L2_NORM:
                متعین_کرتا.push_back("L2_NORM");
                variant = "l2_norm";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        اگر (کلید.inplace) {
            متعین_کرتا.push_back("INPLACE");
            variant += "_inplace";
        }

        اگر (کلید.src_type == GGML_TYPE_F32) {
            متعین_کرتا.push_back("SRC_F32");
            variant += "_src_f32";
        } ورنہ اگر (کلید.src_type == GGML_TYPE_F16) {
            متعین_کرتا.push_back("SRC_F16");
            variant += "_src_f16";
        }

        اگر (کلید.dst_type == GGML_TYPE_F32) {
            متعین_کرتا.push_back("DST_F32");
            variant += "_dst_f32";
        } ورنہ اگر (کلید.dst_type == GGML_TYPE_F16) {
            متعین_کرتا.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ثابت uint32_t صف_norm_wg_سائز = 128u;
        uint32_t       wg_سائز          = std::min(context.max_wg_سائز, صف_norm_wg_سائز);
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائز));
        خودکار processed                  = preprocessor.preprocess(wgsl_صف_norm, متعین_کرتا);
        خودکار decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز              = wg_سائز;
        decisions->inplace              = کلید.inplace;
        صف_norm_pipelines[کلید]         = ggml_webgpu_create_pipeline(device, processed, variant);
        صف_norm_pipelines[کلید].context = decisions;
        واپس صف_norm_pipelines[کلید];
    }

    webgpu_pipeline get_argmax_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقی vec4 = context.src0->ne[0] % 4 == 0;

        خودکار it = argmax_pipelines.find(vec4);
        اگر (it != argmax_pipelines.end()) {
            واپس it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> متعین_کرتا;
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));
        اگر (vec4) {
            متعین_کرتا.push_back("VEC4");
            variant += "_vec4";
        }

        خودکار processed         = preprocessor.preprocess(wgsl_argmax, متعین_کرتا);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        واپس argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_سیٹ_صفیں_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقی                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_سیٹ_صفیں_pipeline_کلید کلید       = {};
        کلید.src0_type                               = context.src0->type;
        کلید.dst_type                                = context.dst->type;
        کلید.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        کلید.i64_idx     = context.src1->type == GGML_TYPE_I64;
        کلید.pair_بلاکیں = quantized && ((context.src0->ne[0] / ggml_blck_سائز(context.dst->type)) % 2 == 0);

        خودکار it = سیٹ_صفیں_pipelines.find(کلید);
        اگر (it != سیٹ_صفیں_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "set_rows";

        انتخاب (context.dst->type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("DST_F32");
                variant += "_dstf32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("DST_F16");
                variant += "_dstf16";
                توڑو;
            حالت GGML_TYPE_Q8_0:
                متعین_کرتا.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                توڑو;
            حالت GGML_TYPE_Q4_0:
                متعین_کرتا.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        اگر (context.src0->type == GGML_TYPE_F16) {
            متعین_کرتا.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        اگر (کلید.vec4) {
            متعین_کرتا.push_back("VEC4");
            variant += "_vec4";
        }
        اگر (کلید.i64_idx) {
            متعین_کرتا.push_back("I64_IDX");
            variant += "_i64idx";
        }
        اگر (کلید.pair_بلاکیں) {
            متعین_کرتا.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        ثابت خودکار & shader_ماخذ      = quantized ? wgsl_سیٹ_صفیں_quant : wgsl_سیٹ_صفیں;
        خودکار         processed          = preprocessor.preprocess(shader_ماخذ, متعین_کرتا);
        خودکار         decisions          = std::make_shared<ggml_webgpu_سیٹ_صفیں_shader_decisions>();
        decisions->vec4                 = کلید.vec4;
        decisions->i64_idx              = کلید.i64_idx;
        decisions->pair_بلاکیں          = کلید.pair_بلاکیں;
        decisions->wg_سائز              = context.max_wg_سائز;
        سیٹ_صفیں_pipelines[کلید]         = ggml_webgpu_create_pipeline(device, processed, variant);
        سیٹ_صفیں_pipelines[کلید].context = decisions;
        واپس سیٹ_صفیں_pipelines[کلید];
    }

    webgpu_pipeline get_سیٹ_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_سیٹ_pipeline_کلید کلید = {};
        کلید.type                         = context.dst->type;
        کلید.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = سیٹ_pipelines.find(کلید);
        اگر (it != سیٹ_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "set";

        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("TYPE_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_I32:
                متعین_کرتا.push_back("TYPE_I32");
                variant += "_i32";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for set shader");
        }

        اگر (کلید.inplace) {
            متعین_کرتا.push_back("INPLACE");
            variant += "_inplace";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_سیٹ, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        سیٹ_pipelines[کلید]       = pipeline;
        واپس سیٹ_pipelines[کلید];
    }

    webgpu_pipeline get_cumsum_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        خودکار it = cumsum_pipelines.find(1);
        اگر (it != cumsum_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed      = preprocessor.preprocess(wgsl_cumsum, متعین_کرتا);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        واپس cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقی          is_چوٹی_k = context.dst->op == GGML_OP_چوٹی_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t ترتیب =
            is_چوٹی_k ? (int32_t) GGML_ترتیب_دو_ترتیب_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        خودکار it = argsort_pipelines.find(ترتیب);
        اگر (it != argsort_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "argsort";
        متعین_کرتا.push_back(std::string("ORDER=") + std::to_string(ترتیب));
        variant += std::string("_order") + std::to_string(ترتیب);
        uint32_t wg_سائز = 1;
        جب (wg_سائز * 2 <= context.max_wg_سائز &&
               wg_سائز * GGML_WEBGPU_I32_سائز_بائٹیں <= context.wg_mem_حد_بندی_بائٹیں / 2) {
            wg_سائز *= 2;
        }
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائز));
        خودکار processed                   = preprocessor.preprocess(wgsl_argsort, متعین_کرتا);
        خودکار decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز               = wg_سائز;
        argsort_pipelines[ترتیب]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[ترتیب].context = decisions;
        واپس argsort_pipelines[ترتیب];
    }

    webgpu_pipeline get_argsort_ملاؤ_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقی          is_چوٹی_k = context.dst->op == GGML_OP_چوٹی_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t ترتیب =
            is_چوٹی_k ? (int32_t) GGML_ترتیب_دو_ترتیب_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        خودکار it = argsort_ملاؤ_pipelines.find(ترتیب);
        اگر (it != argsort_ملاؤ_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "argsort_merge";
        متعین_کرتا.push_back(std::string("ORDER=") + std::to_string(ترتیب));
        variant += std::string("_order") + std::to_string(ترتیب);
        uint32_t wg_سائز = std::min(GGML_WEBGPU_ARGSORT_ملاؤ_MAX_WG_سائز, context.max_wg_سائز);
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائز));

        خودکار processed                 = preprocessor.preprocess(wgsl_argsort_ملاؤ, متعین_کرتا);
        argsort_ملاؤ_pipelines[ترتیب] = ggml_webgpu_create_pipeline(device, processed, variant);
        واپس argsort_ملاؤ_pipelines[ترتیب];
    }

    webgpu_pipeline get_get_صفیں_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقی vec4_aligned) {
        ثابت منطقی vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_صفیں_pipeline_کلید کلید = {};
        کلید.src_type                          = context.src0->type;
        کلید.vectorized                        = (صحیح) vectorized;

        خودکار it = get_صفیں_pipelines.find(کلید);
        اگر (it != get_صفیں_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "get_rows";

        ثابت ساخت ggml_type_traits * type_traits = ggml_get_type_traits(کلید.src_type);
        ثابت حرف *                    type_str    = type_traits->type_name;

        انتخاب (کلید.src_type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("FLOAT_PARALLEL");
                اگر (کلید.vectorized) {
                    متعین_کرتا.push_back("F32_VEC");
                    متعین_کرتا.push_back("SRC_TYPE=vec4<f32>");
                    متعین_کرتا.push_back("DST_TYPE=vec4<f32>");
                    متعین_کرتا.push_back("BLOCK_SIZE=4u");
                } ورنہ {
                    متعین_کرتا.push_back("F32");
                    متعین_کرتا.push_back("SRC_TYPE=f32");
                    متعین_کرتا.push_back("DST_TYPE=f32");
                    متعین_کرتا.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("FLOAT_PARALLEL");
                متعین_کرتا.push_back("F16");
                متعین_کرتا.push_back("SRC_TYPE=f16");
                متعین_کرتا.push_back("DST_TYPE=f32");
                متعین_کرتا.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                توڑو;
            حالت GGML_TYPE_I32:
                متعین_کرتا.push_back("FLOAT_PARALLEL");
                متعین_کرتا.push_back("I32");
                متعین_کرتا.push_back("SRC_TYPE=i32");
                متعین_کرتا.push_back("DST_TYPE=i32");
                متعین_کرتا.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                توڑو;
            حالت GGML_TYPE_BF16:
                متعین_کرتا.push_back("BF16");
                متعین_کرتا.push_back("SRC_TYPE=u32");
                متعین_کرتا.push_back("DST_TYPE=f32");
                متعین_کرتا.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                توڑو;
            طےشدہ:
                {
                    std::string type_بالائی = type_str;
                    std::transform(type_بالائی.begin(), type_بالائی.end(), type_بالائی.begin(), ::بڑا_حرف);

                    انتخاب (کلید.src_type) {
                        حالت GGML_TYPE_Q1_0:
                        حالت GGML_TYPE_Q2_0:
                        حالت GGML_TYPE_Q4_0:
                        حالت GGML_TYPE_Q5_0:
                        حالت GGML_TYPE_Q8_0:
                        حالت GGML_TYPE_Q3_K:
                        حالت GGML_TYPE_Q6_K:
                        حالت GGML_TYPE_IQ2_XXS:
                        حالت GGML_TYPE_IQ2_XS:
                        حالت GGML_TYPE_IQ2_S:
                        حالت GGML_TYPE_IQ3_XXS:
                        حالت GGML_TYPE_IQ3_S:
                        حالت GGML_TYPE_IQ1_S:
                        حالت GGML_TYPE_IQ4_NL:
                        حالت GGML_TYPE_MXFP4:
                        حالت GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                متعین_کرتا.push_back("SRC_TYPE=u32");
                                متعین_کرتا.push_back("U32_DEQUANT_HELPERS");
                                توڑو;
                            }
                        طےشدہ:
                            {
                                متعین_کرتا.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    متعین_کرتا.push_back("BYTE_HELPERS");
                    متعین_کرتا.push_back(type_بالائی + "_T");
                    متعین_کرتا.push_back(type_بالائی);
                    متعین_کرتا.push_back(type_بالائی + "_SCALE_MIN");
                    متعین_کرتا.push_back(type_بالائی + "_TABLES");
                    متعین_کرتا.push_back(type_بالائی + "_GRID");
                    متعین_کرتا.push_back(type_بالائی + "_LUT");

                    variant += "_";
                    variant += type_str;

                    متعین_کرتا.push_back("DST_TYPE=f32");

                    اگر (کلید.src_type == GGML_TYPE_Q1_0) {
                        متعین_کرتا.push_back("BLOCK_SIZE=128u");
                    } ورنہ اگر ((کلید.src_type >= GGML_TYPE_Q4_0 && کلید.src_type <= GGML_TYPE_Q8_1) ||
                               کلید.src_type == GGML_TYPE_IQ4_NL || کلید.src_type == GGML_TYPE_MXFP4) {
                        متعین_کرتا.push_back("BLOCK_SIZE=32u");
                    } ورنہ اگر (کلید.src_type == GGML_TYPE_Q2_0 || کلید.src_type == GGML_TYPE_NVFP4) {
                        متعین_کرتا.push_back("BLOCK_SIZE=64u");
                    } ورنہ اگر (کلید.src_type >= GGML_TYPE_Q2_K) {
                        متعین_کرتا.push_back("BLOCK_SIZE=256u");
                    } ورنہ {
                        متعین_کرتا.push_back("BLOCK_SIZE=1u");
                    }
                    توڑو;
                }
        }

        اگر (کلید.vectorized) {
            variant += "_vec";
        }

        متعین_کرتا.push_back("WG_SIZE=" + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_get_صفیں, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_get_صفیں_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_صفیں_pipelines[کلید]  = pipeline;
        واپس get_صفیں_pipelines[کلید];
    }

    webgpu_pipeline get_scale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_کلید کلید = {};
        کلید.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = scale_pipelines.find(کلید);
        اگر (it != scale_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "scale";

        اگر (کلید.inplace) {
            متعین_کرتا.push_back("INPLACE");
            variant += "_inplace";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_scale, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[کلید]     = pipeline;
        واپس scale_pipelines[کلید];
    }

    webgpu_pipeline get_solve_tri_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_کلید کلید = {};
        کلید.type                               = context.dst->type;
        کلید.n                                  = (صحیح) context.src0->ne[0];
        کلید.k                                  = (صحیح) context.src1->ne[0];

        خودکار it = solve_tri_pipelines.find(کلید);
        اگر (it != solve_tri_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "solve_tri";

        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                variant += "_f32";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ثابت uint32_t wg_سائز       = std::min((uint32_t) کلید.n, context.max_wg_سائز);
        ثابت uint32_t k_tile        = wg_سائز;
        ثابت uint32_t بائٹیں_فی_صف = ((uint32_t) کلید.n + wg_سائز) * GGML_WEBGPU_F32_سائز_بائٹیں;
        ثابت uint32_t batch_n       = (uint32_t) (context.wg_mem_حد_بندی_بائٹیں / بائٹیں_فی_صف);

        متعین_کرتا.push_back(std::string("N=") + std::to_string(کلید.n));
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائز));
        متعین_کرتا.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        متعین_کرتا.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        خودکار processed           = preprocessor.preprocess(wgsl_solve_tri, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = wg_سائز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[کلید] = pipeline;
        واپس solve_tri_pipelines[کلید];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_کلید کلید = {};
        کلید.type                              = context.dst->type;
        کلید.vectorized                        = context.src1->ne[0] == 4;

        خودکار it = ssm_conv_pipelines.find(کلید);
        اگر (it != ssm_conv_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "ssm_conv";

        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                variant += "_f32";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        اگر (کلید.vectorized) {
            متعین_کرتا.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t بلاک_سائز_3    = 32u;
        constexpr uint32_t علامات_فی_wg = 8u;

        متعین_کرتا.push_back("BLOCK_SIZE=" + std::to_string(بلاک_سائز_3) + "u");
        متعین_کرتا.push_back("TOKENS_PER_WG=" + std::to_string(علامات_فی_wg) + "u");

        خودکار processed           = preprocessor.preprocess(wgsl_ssm_conv, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->بلاک_سائز_3    = بلاک_سائز_3;
        decisions->علامات_فی_wg = علامات_فی_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[کلید]  = pipeline;
        واپس ssm_conv_pipelines[کلید];
    }

    webgpu_pipeline get_ssm_جائزہ_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_جائزہ_pipeline_کلید کلید = {};
        کلید.type                              = context.dst->type;
        کلید.d_state                           = (صحیح) context.src0->ne[0];
        کلید.xbc_overlap                       = xbc_overlap;

        خودکار it = ssm_جائزہ_pipelines.find(کلید);
        اگر (it != ssm_جائزہ_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "ssm_scan";

        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                variant += "_f32";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ثابت uint32_t wg_سائز = (uint32_t) کلید.d_state;

        constexpr uint32_t علامات_فی_tile = 4u;

        متعین_کرتا.push_back("WG_SIZE=" + std::to_string(wg_سائز) + "u");
        متعین_کرتا.push_back("TOKENS_PER_TILE=" + std::to_string(علامات_فی_tile) + "u");

        اگر (context.supports_subgroups) {
            متعین_کرتا.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } ورنہ {
            variant += "_wg_reduce";
        }

        اگر (کلید.xbc_overlap == 0b110) {  // x/B
            متعین_کرتا.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } ورنہ اگر (کلید.xbc_overlap == 0b011) {  // B/C
            متعین_کرتا.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } ورنہ اگر (کلید.xbc_overlap == 0b111) {  // x/B/C
            متعین_کرتا.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(کلید.d_state);

        خودکار processed             = preprocessor.preprocess(wgsl_ssm_جائزہ, متعین_کرتا);
        خودکار decisions             = std::make_shared<ggml_webgpu_ssm_جائزہ_shader_decisions>();
        decisions->wg_سائز         = wg_سائز;
        decisions->علامات_فی_tile = علامات_فی_tile;
        decisions->xbc_overlap     = کلید.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_جائزہ_pipelines[کلید]    = pipeline;
        واپس ssm_جائزہ_pipelines[کلید];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_کلید کلید = {};
        کلید.type                                     = context.dst->type;
        کلید.s_v                                      = (صحیح) context.src2->ne[0];
        کلید.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        کلید.fused_cache                              = context.dst_fuse != خالی_اشاریہ;

        خودکار it = gated_delta_net_pipelines.find(کلید);
        اگر (it != gated_delta_net_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "gated_delta_net";

        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                variant += "_f32";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        اگر (کلید.kda) {
            متعین_کرتا.push_back("KDA");
            variant += "_kda";
        }

        اگر (کلید.fused_cache) {
            متعین_کرتا.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        متعین_کرتا.push_back("S_V=" + std::to_string(کلید.s_v) + "u");
        متعین_کرتا.push_back("WG_SIZE=" + std::to_string(کلید.s_v) + "u");

        خودکار            processed      = preprocessor.preprocess(wgsl_gated_delta_net, متعین_کرتا);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[کلید] = pipeline;
        واپس gated_delta_net_pipelines[کلید];
    }

    webgpu_pipeline get_pad_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_کلید کلید = {};
        کلید.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        خودکار it = pad_pipelines.find(کلید);
        اگر (it != pad_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "pad";

        اگر (کلید.circular) {
            متعین_کرتا.push_back("CIRCULAR");
            variant += "_circular";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_pad, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[کلید]       = pipeline;
        واپس pad_pipelines[کلید];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_کلید کلید = {};
        کلید.src0_type                            = context.src0->type;

        خودکار it = quantize_q8_pipelines.find(کلید);
        اگر (it != quantize_q8_pipelines.end()) {
            واپس it->second;
        }
        ثابت حرف *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> متعین_کرتا;
        std::string              variant = "quantize_q8";

        uint32_t wg_سائز = WEBGPU_MUL_MAT_VEC_WG_سائز;

        متعین_کرتا.push_back("SRC1_INNER_TYPE=f32");
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائز));

        ثابت ساخت ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_بالائی  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_بالائی.begin(), type_بالائی.end(), type_بالائی.begin(), ::بڑا_حرف);

        متعین_کرتا.push_back("MUL_ACC_" + type_بالائی);
        متعین_کرتا.push_back("Q8_1_T");

        متعین_کرتا.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        خودکار processed             = preprocessor.preprocess(shader_src, متعین_کرتا);
        خودکار decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز         = wg_سائز;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[کلید] = pipeline;
        واپس quantize_q8_pipelines[کلید];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقی src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_کلید کلید = {};
        کلید.src0_type                            = context.src0->type;
        کلید.src1_type                            = context.src1->type;
        کلید.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        کلید.عدد_ستون_تعداد   = context.dst->ne[1];
        کلید.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_حاصل_ضرب, context.vendor);
        کلید.src_overlap = src_overlap;

        خودکار it = mul_mat_vec_pipelines.find(کلید);
        اگر (it != mul_mat_vec_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant    = "mul_mat_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        انتخاب (context.src0->type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=f32");
                متعین_کرتا.push_back("MUL_ACC_FLOAT");
                متعین_کرتا.push_back("TYPE_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=f16");
                متعین_کرتا.push_back("MUL_ACC_FLOAT");
                متعین_کرتا.push_back("TYPE_F16");
                variant += "_f16";
                توڑو;
            حالت GGML_TYPE_BF16:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=u32");
                متعین_کرتا.push_back("MUL_ACC_FLOAT");
                متعین_کرتا.push_back("TYPE_BF16");
                variant += "_bf16";
                توڑو;
            طےشدہ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت ساخت ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_بالائی  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_بالائی.begin(), type_بالائی.end(), type_بالائی.begin(), ::بڑا_حرف);

                    متعین_کرتا.push_back("BYTE_HELPERS");
                    متعین_کرتا.push_back("MUL_ACC_" + type_بالائی);
                    متعین_کرتا.push_back("U32_DEQUANT_HELPERS");
                    متعین_کرتا.push_back("SRC0_INNER_TYPE=u32");
                    انتخاب (context.src0->type) {
                        حالت GGML_TYPE_Q4_0:
                        حالت GGML_TYPE_Q4_1:
                        حالت GGML_TYPE_Q5_0:
                        حالت GGML_TYPE_Q5_1:
                        حالت GGML_TYPE_Q8_0:
                            اگر (کلید.use_mmvq) {
                                متعین_کرتا.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            توڑو;
                        حالت GGML_TYPE_Q1_0:
                        حالت GGML_TYPE_Q2_K:
                        حالت GGML_TYPE_Q3_K:
                        حالت GGML_TYPE_Q4_K:
                        حالت GGML_TYPE_Q5_K:
                        حالت GGML_TYPE_Q6_K:
                            اگر (کلید.use_mmvq) {
                                متعین_کرتا.push_back("K_QUANTS_HANDLING");
                            }
                            توڑو;
                        حالت GGML_TYPE_IQ1_S:
                        حالت GGML_TYPE_IQ1_M:
                        حالت GGML_TYPE_IQ2_S:
                        حالت GGML_TYPE_IQ3_S:
                        حالت GGML_TYPE_IQ4_NL:
                        حالت GGML_TYPE_IQ4_XS:
                            متعین_کرتا.push_back(type_بالائی + "_GRID");
                            توڑو;
                        حالت GGML_TYPE_IQ2_XXS:
                        حالت GGML_TYPE_IQ2_XS:
                        حالت GGML_TYPE_IQ3_XXS:
                            متعین_کرتا.push_back(type_بالائی + "_GRID");
                            متعین_کرتا.push_back(type_بالائی + "_TABLES");
                            توڑو;
                        حالت GGML_TYPE_MXFP4:
                            متعین_کرتا.push_back(type_بالائی + "_LUT");
                            اگر (کلید.use_mmvq) {
                                متعین_کرتا.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            توڑو;
                        حالت GGML_TYPE_NVFP4:
                            متعین_کرتا.push_back(type_بالائی + "_LUT");
                            توڑو;
                        طےشدہ:
                            توڑو;
                    }
                    توڑو;
                }
        }

        // src1 type (vector)
        انتخاب (context.src1->type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        متعین_کرتا.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_سائز        = WEBGPU_MUL_MAT_VEC_WG_سائز;
        uint32_t outputs_فی_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_فی_WG;

        اگر (کلید.src0_type == GGML_TYPE_Q1_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        } ورنہ اگر (کلید.src0_type >= GGML_TYPE_Q2_K) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_فی_WG;
        } ورنہ اگر (کلید.src0_type >= GGML_TYPE_Q4_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        }

        اگر (کلید.use_mmvq) {
            متعین_کرتا.push_back("MMVQ");
            متعین_کرتا.push_back("Q8_1_T");
        }

        اگر (کلید.src_overlap) {
            متعین_کرتا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائز));
        متعین_کرتا.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_فی_wg));
        متعین_کرتا.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        اگر (کلید.vectorized) {
            variant += "_vectorized";
        }
        متعین_کرتا.push_back(std::string("NUM_COLS=") + std::to_string(کلید.عدد_ستون_تعداد));

        خودکار processed            = preprocessor.preprocess(shader_src, متعین_کرتا);
        خودکار decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_سائز        = wg_سائز;
        decisions->outputs_فی_wg = outputs_فی_wg;
        decisions->vec_سائز       = کلید.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[کلید] = pipeline;
        واپس mul_mat_vec_pipelines[کلید];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقی src_overlap) {
        ggml_webgpu_mul_mat_pipeline_کلید کلید = {};
        کلید.src0_type                        = context.src0->type;
        کلید.src1_type                        = context.src1->type;
        کلید.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        کلید.use_subgroup_میٹرکس = context.supports_subgroup_میٹرکس;
        کلید.src_overlap         = src_overlap;

        خودکار it = mul_mat_fast_pipelines.find(کلید);
        اگر (it != mul_mat_fast_pipelines.end()) {
            واپس it->second;
        }

        ثابت حرف * shader_src = کلید.use_subgroup_میٹرکس ? wgsl_mul_mat_subgroup_میٹرکس : wgsl_mul_mat_reg_tile;
        std::vector<std::string> متعین_کرتا;
        std::string              variant = کلید.use_subgroup_میٹرکس ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        انتخاب (context.src1->type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("SRC1_INNER_TYPE=f32");
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("SRC1_INNER_TYPE=f16");
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت ساخت ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        انتخاب (context.src0->type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=f32");
                متعین_کرتا.push_back("TYPE_F32");
                متعین_کرتا.push_back("MUL_ACC_FLOAT");
                متعین_کرتا.push_back("INIT_SRC0_SHMEM_FLOAT");
                متعین_کرتا.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=f16");
                متعین_کرتا.push_back("TYPE_F16");
                متعین_کرتا.push_back("MUL_ACC_FLOAT");
                متعین_کرتا.push_back("INIT_SRC0_SHMEM_FLOAT");
                متعین_کرتا.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                توڑو;
            حالت GGML_TYPE_BF16:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=u32");
                متعین_کرتا.push_back("TYPE_BF16");
                متعین_کرتا.push_back("MUL_ACC_FLOAT");
                متعین_کرتا.push_back("INIT_SRC0_SHMEM_FLOAT");
                متعین_کرتا.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                توڑو;
            طےشدہ:
                {
                    std::string type_بالائی = src0_name;
                    std::transform(type_بالائی.begin(), type_بالائی.end(), type_بالائی.begin(), ::بڑا_حرف);

                    متعین_کرتا.push_back("BYTE_HELPERS");
                    متعین_کرتا.push_back("MUL_ACC_" + type_بالائی);
                    متعین_کرتا.push_back("INIT_SRC0_SHMEM_" + type_بالائی);
                    متعین_کرتا.push_back("INIT_SRC1_SHMEM_FLOAT");
                    متعین_کرتا.push_back("U32_DEQUANT_HELPERS");
                    متعین_کرتا.push_back("SRC0_INNER_TYPE=u32");

                    انتخاب (context.src0->type) {
                        حالت GGML_TYPE_IQ1_S:
                        حالت GGML_TYPE_IQ1_M:
                        حالت GGML_TYPE_IQ4_NL:
                        حالت GGML_TYPE_IQ4_XS:
                            متعین_کرتا.push_back(type_بالائی + "_GRID");
                            توڑو;
                        حالت GGML_TYPE_IQ2_XXS:
                        حالت GGML_TYPE_IQ2_XS:
                        حالت GGML_TYPE_IQ2_S:
                        حالت GGML_TYPE_IQ3_XXS:
                        حالت GGML_TYPE_IQ3_S:
                            متعین_کرتا.push_back(type_بالائی + "_GRID");
                            متعین_کرتا.push_back(type_بالائی + "_TABLES");
                            توڑو;
                        حالت GGML_TYPE_MXFP4:
                        حالت GGML_TYPE_NVFP4:
                            متعین_کرتا.push_back(type_بالائی + "_LUT");
                            توڑو;
                        طےشدہ:
                            توڑو;
                    }

                    variant += std::string("_") + src0_name;
                    توڑو;
                }
        }

        // VEC/SCALAR controls
        متعین_کرتا.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        ثابت منطقی is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        اگر (کلید.use_subgroup_میٹرکس) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } ورنہ {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        متعین_کرتا.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        متعین_کرتا.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        اگر (کلید.use_subgroup_میٹرکس) {
            متعین_کرتا.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            متعین_کرتا.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_سائز) + "u");
            متعین_کرتا.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            متعین_کرتا.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            متعین_کرتا.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_میٹرکس_M) + "u");
            متعین_کرتا.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_میٹرکس_N) + "u");
            متعین_کرتا.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            متعین_کرتا.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            متعین_کرتا.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        اگر (کلید.vectorized) {
            variant += "_vectorized";
        }

        اگر (کلید.src_overlap) {
            متعین_کرتا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        اگر (!کلید.use_subgroup_میٹرکس) {
            متعین_کرتا.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_سائز_M) + "u");
            متعین_کرتا.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_سائز_N) + "u");
            متعین_کرتا.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        خودکار processed = preprocessor.preprocess(shader_src, متعین_کرتا);

        خودکار decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_میٹرکس = کلید.use_subgroup_میٹرکس;
        اگر (کلید.use_subgroup_میٹرکس) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_میٹرکس_m = WEBGPU_MUL_MAT_SUBGROUP_میٹرکس_M;
            decisions->subgroup_میٹرکس_n = WEBGPU_MUL_MAT_SUBGROUP_میٹرکس_N;
            decisions->wg_سائز           = context.max_subgroup_سائز;
        } ورنہ {
            decisions->wg_سائز_m       = WEBGPU_MUL_MAT_WG_سائز_M;
            decisions->wg_سائز_n       = WEBGPU_MUL_MAT_WG_سائز_N;
            decisions->wg_سائز         = WEBGPU_MUL_MAT_WG_سائز_M * WEBGPU_MUL_MAT_WG_سائز_N;
            decisions->mul_mat_wg_سائز = WEBGPU_MUL_MAT_WG_سائز;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[کلید] = pipeline;
        واپس mul_mat_fast_pipelines[کلید];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_کلید کلید = {};
        کلید.n_experts                                  = (uint32_t) context.src0->ne[2];

        خودکار it = mul_mat_id_gather_pipelines.find(کلید);
        اگر (it != mul_mat_id_gather_pipelines.end()) {
            واپس it->second;
        }
        std::vector<std::string> متعین_کرتا;
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز) + "u");
        متعین_کرتا.push_back(std::string("N_EXPERT=") + std::to_string(کلید.n_experts) + "u");
        متعین_کرتا.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_سائز_N) + "u");

        خودکار processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, متعین_کرتا);
        خودکار decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز = context.max_wg_سائز;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[کلید] = pipeline;
        واپس pipeline;
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
        اگر (it != mul_mat_id_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "mul_mat_id";
        متعین_کرتا.push_back("MUL_MAT_ID");

        // src1 type
        انتخاب (context.src1->type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("SRC1_INNER_TYPE=f32");
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("SRC1_INNER_TYPE=f16");
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت ساخت ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        انتخاب (context.src0->type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=f32");
                متعین_کرتا.push_back("INIT_SRC0_SHMEM_FLOAT");
                متعین_کرتا.push_back("INIT_SRC1_SHMEM_FLOAT");
                متعین_کرتا.push_back("TYPE_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=f16");
                متعین_کرتا.push_back("INIT_SRC0_SHMEM_FLOAT");
                متعین_کرتا.push_back("INIT_SRC1_SHMEM_FLOAT");
                متعین_کرتا.push_back("TYPE_F16");
                variant += "_f16";
                توڑو;
            حالت GGML_TYPE_BF16:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=u32");
                متعین_کرتا.push_back("INIT_SRC0_SHMEM_FLOAT");
                متعین_کرتا.push_back("INIT_SRC1_SHMEM_FLOAT");
                متعین_کرتا.push_back("TYPE_BF16");
                variant += "_bf16";
                توڑو;
            طےشدہ:
                {
                    std::string type_بالائی = src0_name;
                    std::transform(type_بالائی.begin(), type_بالائی.end(), type_بالائی.begin(), ::بڑا_حرف);

                    متعین_کرتا.push_back("BYTE_HELPERS");
                    متعین_کرتا.push_back("INIT_SRC0_SHMEM_" + type_بالائی);
                    متعین_کرتا.push_back("INIT_SRC1_SHMEM_FLOAT");
                    متعین_کرتا.push_back("U32_DEQUANT_HELPERS");
                    متعین_کرتا.push_back("SRC0_INNER_TYPE=u32");

                    انتخاب (context.src0->type) {
                        حالت GGML_TYPE_IQ1_S:
                        حالت GGML_TYPE_IQ1_M:
                        حالت GGML_TYPE_IQ4_NL:
                        حالت GGML_TYPE_IQ4_XS:
                            متعین_کرتا.push_back(type_بالائی + "_GRID");
                            توڑو;
                        حالت GGML_TYPE_IQ2_XXS:
                        حالت GGML_TYPE_IQ2_XS:
                        حالت GGML_TYPE_IQ2_S:
                        حالت GGML_TYPE_IQ3_XXS:
                        حالت GGML_TYPE_IQ3_S:
                            متعین_کرتا.push_back(type_بالائی + "_GRID");
                            متعین_کرتا.push_back(type_بالائی + "_TABLES");
                            توڑو;
                        حالت GGML_TYPE_MXFP4:
                        حالت GGML_TYPE_NVFP4:
                            متعین_کرتا.push_back(type_بالائی + "_LUT");
                            توڑو;
                        طےشدہ:
                            توڑو;
                    }

                    variant += std::string("_") + src0_name;
                    توڑو;
                }
        }

        // VEC/SCALAR controls
        متعین_کرتا.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ثابت uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        متعین_کرتا.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        متعین_کرتا.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        متعین_کرتا.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        متعین_کرتا.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_سائز_M) + "u");
        متعین_کرتا.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_سائز_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        اگر (کلید.vectorized) {
            variant += "_vectorized";
        }

        خودکار processed = preprocessor.preprocess(wgsl_mul_mat_id, متعین_کرتا);

        خودکار decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_سائز_m = WEBGPU_MUL_MAT_WG_سائز_M;
        decisions->wg_سائز_n = WEBGPU_MUL_MAT_WG_سائز_N;
        decisions->wg_سائز   = WEBGPU_MUL_MAT_WG_سائز_M * WEBGPU_MUL_MAT_WG_سائز_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[کلید] = pipeline;
        واپس mul_mat_id_pipelines[کلید];
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
        اگر (it != mul_mat_id_vec_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant    = "mul_mat_id_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        انتخاب (context.src1->type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("SRC1_INNER_TYPE=f32");
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("SRC1_INNER_TYPE=f16");
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        انتخاب (context.src0->type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=f32");
                متعین_کرتا.push_back("MUL_ACC_FLOAT");
                متعین_کرتا.push_back("TYPE_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=f16");
                متعین_کرتا.push_back("MUL_ACC_FLOAT");
                متعین_کرتا.push_back("TYPE_F16");
                variant += "_f16";
                توڑو;
            حالت GGML_TYPE_BF16:
                متعین_کرتا.push_back("SRC0_INNER_TYPE=u32");
                متعین_کرتا.push_back("MUL_ACC_FLOAT");
                متعین_کرتا.push_back("TYPE_BF16");
                variant += "_bf16";
                توڑو;
            طےشدہ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت ساخت ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_بالائی  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_بالائی.begin(), type_بالائی.end(), type_بالائی.begin(), ::بڑا_حرف);

                    متعین_کرتا.push_back("BYTE_HELPERS");
                    متعین_کرتا.push_back("MUL_ACC_" + type_بالائی);
                    متعین_کرتا.push_back("U32_DEQUANT_HELPERS");
                    متعین_کرتا.push_back("SRC0_INNER_TYPE=u32");
                    انتخاب (context.src0->type) {
                        حالت GGML_TYPE_IQ1_S:
                        حالت GGML_TYPE_IQ1_M:
                        حالت GGML_TYPE_IQ2_S:
                        حالت GGML_TYPE_IQ3_S:
                        حالت GGML_TYPE_IQ4_NL:
                        حالت GGML_TYPE_IQ4_XS:
                            متعین_کرتا.push_back(type_بالائی + "_GRID");
                            توڑو;
                        حالت GGML_TYPE_IQ2_XXS:
                        حالت GGML_TYPE_IQ2_XS:
                        حالت GGML_TYPE_IQ3_XXS:
                            متعین_کرتا.push_back(type_بالائی + "_GRID");
                            متعین_کرتا.push_back(type_بالائی + "_TABLES");
                            توڑو;
                        حالت GGML_TYPE_MXFP4:
                        حالت GGML_TYPE_NVFP4:
                            متعین_کرتا.push_back(type_بالائی + "_LUT");
                            توڑو;
                        طےشدہ:
                            توڑو;
                    }
                    توڑو;
                }
        }

        // VEC/SCALAR controls
        متعین_کرتا.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_سائز        = WEBGPU_MUL_MAT_VEC_WG_سائز;
        uint32_t outputs_فی_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_فی_WG;

        اگر (کلید.src0_type == GGML_TYPE_Q1_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        } ورنہ اگر (کلید.src0_type >= GGML_TYPE_Q2_K) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_فی_WG;
        } ورنہ اگر (کلید.src0_type >= GGML_TYPE_Q4_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائز));
        متعین_کرتا.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_فی_wg));
        متعین_کرتا.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        اگر (کلید.vectorized) {
            variant += "_vectorized";
        }
        متعین_کرتا.push_back(std::string("NUM_COLS=1"));

        متعین_کرتا.push_back(std::string("N_EXPERTS=") + std::to_string(کلید.n_experts));

        خودکار processed = preprocessor.preprocess(shader_src, متعین_کرتا);

        خودکار decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_سائز        = wg_سائز;
        decisions->outputs_فی_wg = outputs_فی_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[کلید] = pipeline;
        واپس mul_mat_id_vec_pipelines[کلید];
    }

    webgpu_pipeline get_unary_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقی                     is_unary = context.dst->op == GGML_OP_UNARY;
        ثابت صحیح                      op       = is_unary ? (صحیح) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_کلید کلید      = {};
        کلید.type                                = context.dst->type;
        کلید.op                                  = op;
        کلید.is_unary                            = is_unary;
        کلید.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        کلید.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        خودکار it = unary_pipelines.find(کلید);
        اگر (it != unary_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant =
            کلید.is_unary ? ggml_unary_op_name((ggml_unary_op) کلید.op) : ggml_op_name((ggml_op) کلید.op);
        متعین_کرتا.push_back(variant);

        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("TYPE_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("TYPE_F16");
                variant += "_f16";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for unary shader");
        }

        اگر (کلید.inplace) {
            متعین_کرتا.push_back("INPLACE");
            variant += "_inplace";
        }

        اگر (op == GGML_OP_TRI) {
            انتخاب (کلید.ttype) {
                حالت GGML_TRI_TYPE_زیریں:
                    متعین_کرتا.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    توڑو;
                حالت GGML_TRI_TYPE_زیریں_DIAG:
                    متعین_کرتا.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    توڑو;
                حالت GGML_TRI_TYPE_بالائی:
                    متعین_کرتا.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    توڑو;
                حالت GGML_TRI_TYPE_بالائی_DIAG:
                    متعین_کرتا.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    توڑو;
                طےشدہ:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_unary, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[کلید]     = pipeline;
        واپس unary_pipelines[کلید];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقی src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_کلید کلید = {};
        کلید.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        کلید.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        کلید.src_overlap                           = src_overlap;

        خودکار it = rms_norm_mul_pipelines.find(کلید);
        اگر (it != rms_norm_mul_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        اگر (کلید.inplace) {
            متعین_کرتا.push_back("INPLACE");
            variant += "_inplace";
        } ورنہ اگر (کلید.overlap) {
            متعین_کرتا.push_back("OVERLAP");
            variant += "_overlap";
        } ورنہ اگر (کلید.src_overlap) {
            متعین_کرتا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, متعین_کرتا);
        خودکار pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_سائز     = context.max_wg_سائز;
        pipeline_decisions->inplace     = کلید.inplace;
        pipeline_decisions->overlap     = کلید.overlap;
        pipeline_decisions->src_overlap = کلید.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[کلید]     = pipeline;
        واپس rms_norm_mul_pipelines[کلید];
    }

    webgpu_pipeline get_ثنائی_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقی src_overlap) {
        ggml_webgpu_ثنائی_pipeline_کلید کلید = {};
        کلید.type                            = context.dst->type;
        کلید.op                              = context.dst->op;
        کلید.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        کلید.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        کلید.src_overlap                     = src_overlap;

        خودکار it = ثنائی_pipelines.find(کلید);
        اگر (it != ثنائی_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              op_name = ggml_op_name((ggml_op) کلید.op);
        std::string              variant = op_name;

        متعین_کرتا.push_back(std::string("OP_") + op_name);

        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("TYPE_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("TYPE_F16");
                variant += "_f16";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for binary shader");
        }

        اگر (کلید.inplace) {
            متعین_کرتا.push_back("INPLACE");
            variant += "_inplace";
        } ورنہ اگر (کلید.overlap) {
            متعین_کرتا.push_back("OVERLAP");
            variant += "_overlap";
        } ورنہ اگر (کلید.src_overlap) {
            متعین_کرتا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed                  = preprocessor.preprocess(wgsl_ثنائی, متعین_کرتا);
        خودکار pipeline_decisions         = std::make_shared<ggml_webgpu_ثنائی_shader_decisions>();
        pipeline_decisions->wg_سائز     = context.max_wg_سائز;
        pipeline_decisions->inplace     = کلید.inplace;
        pipeline_decisions->overlap     = کلید.overlap;
        pipeline_decisions->src_overlap = کلید.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ثنائی_pipelines[کلید]    = pipeline;
        واپس ثنائی_pipelines[کلید];
    }

    webgpu_pipeline get_جوڑو_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_جوڑو_id_pipeline_کلید کلید = {};
        کلید.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = جوڑو_id_pipelines.find(کلید);
        اگر (it != جوڑو_id_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant    = "add_id";
        ثابت حرف *             shader_src = wgsl_جوڑو_id;

        اگر (کلید.inplace) {
            متعین_کرتا.push_back("INPLACE");
            variant += "_inplace";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed              = preprocessor.preprocess(shader_src, متعین_کرتا);
        خودکار pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_سائز = context.max_wg_سائز;
        pipeline_decisions->inplace = کلید.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        جوڑو_id_pipelines[کلید]    = pipeline;
        واپس pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقی src_overlap) {
        ggml_webgpu_concat_pipeline_کلید کلید = {};
        کلید.type                            = context.dst->type;
        کلید.src_overlap                     = src_overlap;

        خودکار it = concat_pipelines.find(کلید);
        اگر (it != concat_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "concat";

        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("TYPE_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_I32:
                متعین_کرتا.push_back("TYPE_I32");
                variant += "_i32";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for concat shader");
        }

        اگر (کلید.src_overlap) {
            متعین_کرتا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_concat, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_ثنائی_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        decisions->src_overlap   = کلید.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[کلید]    = pipeline;
        واپس concat_pipelines[کلید];
    }

    webgpu_pipeline get_repeat_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_کلید کلید = {};
        کلید.type                            = context.dst->type;

        خودکار it = repeat_pipelines.find(کلید);
        اگر (it != repeat_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "repeat";

        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("TYPE_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("TYPE_F16");
                variant += "_f16";
                توڑو;
            حالت GGML_TYPE_I32:
                متعین_کرتا.push_back("TYPE_I32");
                variant += "_i32";
                توڑو;
            حالت GGML_TYPE_I16:
                متعین_کرتا.push_back("TYPE_I16");
                variant += "_i16";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_repeat, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[کلید]    = pipeline;
        واپس repeat_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقی kv_overlap) {
        ثابت منطقی can_use_subgroup_میٹرکس = ggml_webgpu_flash_attn_can_use_subgroup_میٹرکس_راستہ(
            context.supports_subgroup_میٹرکس, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_میٹرکس                    = can_use_subgroup_میٹرکس;
        decisions.q_tile = decisions.use_sg_میٹرکس ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_کلید کلید = {};
        کلید.common                              = ggml_webgpu_flash_attn_make_common_pipeline_کلید(
            context, decisions.use_sg_میٹرکس ? context.sg_mat_k : 1u, kv_overlap);
        کلید.common.k_براہ_راست &= decisions.use_sg_میٹرکس && کلید.common.k_type == GGML_TYPE_F16;
        کلید.common.v_براہ_راست &= decisions.use_sg_میٹرکس && کلید.common.v_type == GGML_TYPE_F16;
        کلید.use_sg_میٹرکس = decisions.use_sg_میٹرکس;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_حد_بندی_بائٹیں, decisions.q_tile, decisions.use_sg_میٹرکس ? context.sg_mat_n : 1u,
            کلید.common.سر_dim_qk, کلید.common.سر_dim_v, کلید.common.has_نقاب,
            کلید.common.k_براہ_راست || کلید.common.v_براہ_راست);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        اگر (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_میٹرکس);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_میٹرکس = جھوٹ;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            کلید.common.k_براہ_راست     = جھوٹ;
            کلید.common.v_براہ_راست     = جھوٹ;
            کلید.use_sg_میٹرکس       = جھوٹ;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_حد_بندی_بائٹیں, decisions.q_tile, 1u, کلید.common.سر_dim_qk, کلید.common.سر_dim_v,
                کلید.common.has_نقاب, کلید.common.k_براہ_راست || کلید.common.v_براہ_راست);
            GGML_ASSERT(max_kv_tile > 0);
        }

        decisions.kv_tile = decisions.use_sg_میٹرکس ?
                                std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES) :
                                std::min(GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE, max_kv_tile);
        decisions.wg_سائز =
            decisions.use_sg_میٹرکس ?
                std::max(context.max_subgroup_سائز, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_سائز) :
                std::min(context.max_wg_سائز, std::max(GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_سائز,
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * context.max_subgroup_سائز));

        اگر (کلید.common.k_براہ_راست || کلید.common.v_براہ_راست) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            جب (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_میٹرکس ? context.sg_mat_n : context.min_subgroup_سائز;
            }
        }

        خودکار it = flash_attn_pipelines.find(کلید);
        اگر (it != flash_attn_pipelines.end()) {
            واپس it->second;
        }

        std::string              variant = decisions.use_sg_میٹرکس ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> متعین_کرتا = ggml_webgpu_flash_attn_common_متعین_کرتا(کلید.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_سائز);
        ثابت حرف *             shader_src = خالی_اشاریہ;
        اگر (!کلید.use_sg_میٹرکس) {
            shader_src = wgsl_flash_attn_tile;
            متعین_کرتا.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_سائز) + "u");
            متعین_کرتا.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_سائز) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_سائز) + "_" +
                       std::to_string(context.max_subgroup_سائز);
        } ورنہ {
            shader_src = wgsl_flash_attn;
            متعین_کرتا.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            متعین_کرتا.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            متعین_کرتا.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        خودکار            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, متعین_کرتا), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[کلید] = pipeline;
        واپس flash_attn_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقی kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_کلید کلید = {};
        کلید.common = ggml_webgpu_flash_attn_make_common_pipeline_کلید(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی,
                                                                     kv_overlap);

        خودکار it = flash_attn_vec_pipelines.find(کلید);
        اگر (it != flash_attn_vec_pipelines.end()) {
            واپس it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_حد_بندی_بائٹیں, کلید.common.سر_dim_qk,
                                                                   کلید.common.سر_dim_v, کلید.common.has_نقاب,
                                                                   کلید.common.k_براہ_راست || کلید.common.v_براہ_راست);
        decisions.wg_سائز = context.max_subgroup_سائز;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> متعین_کرتا =
            ggml_webgpu_flash_attn_common_متعین_کرتا(کلید.common, variant, 1u, decisions.kv_tile, decisions.wg_سائز);
        اگر (کلید.common.has_نقاب) {
            متعین_کرتا.push_back("BLK");
            variant.resize(variant.size() - (حجم("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_بانٹو = context.min_subgroup_سائز;
        ثابت uint32_t D       = کلید.common.سر_dim_qk | کلید.common.سر_dim_v;
        ثابت uint32_t D_lsb   = D & (~(D - 1u));
        d_بانٹو                = std::min(std::min(context.min_subgroup_سائز, 4u), std::max(D_lsb / 4u, 1u));

        متعین_کرتا.push_back(std::string("D_SPLIT=") + std::to_string(d_بانٹو));
        variant += "_dsplit" + std::to_string(d_بانٹو);

        خودکار            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_بانٹو, متعین_کرتا), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[کلید] = pipeline;
        واپس flash_attn_vec_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_کلید کلید = {};
        کلید.kv_tile                                 = kv_tile;
        خودکار it                                     = flash_attn_blk_pipelines.find(کلید);
        اگر (it != flash_attn_blk_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "flash_attn_vec_blk";

        متعین_کرتا.push_back(std::string("KV_TILE=") + std::to_string(کلید.kv_tile));
        variant += std::string("_kvt") + std::to_string(کلید.kv_tile);

        uint32_t wg_سائز = 1;
        جب ((wg_سائز << 1) <= context.max_wg_سائز) {
            wg_سائز <<= 1;
        }
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(wg_سائز));
        variant += std::string("_wg") + std::to_string(wg_سائز);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, متعین_کرتا), variant);
        flash_attn_blk_pipelines[کلید] = pipeline;
        واپس flash_attn_blk_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید کلید = {};
        کلید.سر_dim_v                                     = (uint32_t) context.src2->ne[0];
        کلید.dst_type                                       = context.dst->type;
        کلید.wg_سائز                                        = context.max_wg_سائز;
        خودکار it                                            = flash_attn_vec_reduce_pipelines.find(کلید);
        اگر (it != flash_attn_vec_reduce_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "flash_attn_vec_reduce";

        انتخاب (کلید.dst_type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("DST_F32");
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("DST_F16");
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(کلید.dst_type);

        متعین_کرتا.push_back(std::string("HEAD_DIM_V=") + std::to_string(کلید.سر_dim_v));
        variant += std::string("_hsv") + std::to_string(کلید.سر_dim_v);

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));
        variant += std::string("_wg") + std::to_string(context.max_wg_سائز);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, متعین_کرتا), variant);
        flash_attn_vec_reduce_pipelines[کلید] = pipeline;
        واپس flash_attn_vec_reduce_pipelines[کلید];
    }

    webgpu_pipeline get_cpy_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_کلید کلید = {};
        کلید.src_type                     = context.src0->type;
        کلید.dst_type                     = context.dst->type;

        خودکار it = cpy_pipelines.find(کلید);
        اگر (it != cpy_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "cpy";

        انتخاب (کلید.src_type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("SRC_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("SRC_F16");
                variant += "_f16";
                توڑو;
            حالت GGML_TYPE_I32:
                متعین_کرتا.push_back("SRC_I32");
                variant += "_i32";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        انتخاب (کلید.dst_type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("DST_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("DST_F16");
                variant += "_f16";
                توڑو;
            حالت GGML_TYPE_I32:
                متعین_کرتا.push_back("DST_I32");
                variant += "_i32";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_cpy, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[کلید]       = pipeline;
        واپس cpy_pipelines[کلید];
    }

    webgpu_pipeline get_glu_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقی src_overlap) {
        ggml_webgpu_glu_pipeline_کلید کلید = {};
        کلید.glu_op                       = ggml_get_glu_op(context.dst);
        کلید.type                         = context.dst->type;
        کلید.بانٹو                        = (context.src1 != خالی_اشاریہ);
        کلید.src_overlap                  = src_overlap;

        خودکار it = glu_pipelines.find(کلید);
        اگر (it != glu_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "glu";

        انتخاب (کلید.glu_op) {
            حالت GGML_GLU_OP_REGLU:
                متعین_کرتا.push_back("OP_REGLU");
                variant += "_reglu";
                توڑو;
            حالت GGML_GLU_OP_GEGLU:
                متعین_کرتا.push_back("OP_GEGLU");
                variant += "_geglu";
                توڑو;
            حالت GGML_GLU_OP_SWIGLU:
                متعین_کرتا.push_back("OP_SWIGLU");
                variant += "_swiglu";
                توڑو;
            حالت GGML_GLU_OP_SWIGLU_OAI:
                متعین_کرتا.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                توڑو;
            حالت GGML_GLU_OP_GEGLU_ERF:
                متعین_کرتا.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                توڑو;
            حالت GGML_GLU_OP_GEGLU_تیز:
                متعین_کرتا.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                توڑو;
            حالت GGML_GLU_OP_SWIGLU_CLAMP:
                متعین_کرتا.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported GLU op");
        }
        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("TYPE_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("TYPE_F16");
                variant += "_f16";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        اگر (کلید.src_overlap) {
            متعین_کرتا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } ورنہ اگر (کلید.بانٹو) {
            variant += "_split";
        } ورنہ {
            متعین_کرتا.push_back("NO_SPLIT");
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_glu, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        decisions->src_overlap   = کلید.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[کلید]       = pipeline;
        واپس glu_pipelines[کلید];
    }

    webgpu_pipeline get_rope_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_کلید کلید = {};
        کلید.type                          = context.dst->type;
        کلید.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        کلید.has_ff                        = (context.src2 != خالی_اشاریہ);

        خودکار it = rope_pipelines.find(کلید);
        اگر (it != rope_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "rope";

        انتخاب (کلید.type) {
            حالت GGML_TYPE_F32:
                متعین_کرتا.push_back("TYPE_F32");
                variant += "_f32";
                توڑو;
            حالت GGML_TYPE_F16:
                متعین_کرتا.push_back("TYPE_F16");
                variant += "_f16";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        اگر (کلید.inplace) {
            متعین_کرتا.push_back("INPLACE");
            variant += "_inplace";
        }

        اگر (کلید.has_ff) {
            متعین_کرتا.push_back("FF_FUNC");
            variant += "_ff";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_rope, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[کلید]      = pipeline;
        واپس rope_pipelines[کلید];
    }

    webgpu_pipeline get_soft_max_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_کلید کلید = {};
        کلید.نقاب_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        کلید.has_نقاب                          = (context.src1 != خالی_اشاریہ);
        کلید.has_sink                          = (context.src2 != خالی_اشاریہ);
        کلید.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = soft_max_pipelines.find(کلید);
        اگر (it != soft_max_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "soft_max";

        اگر (کلید.has_نقاب) {
            متعین_کرتا.push_back("HAS_MASK");
            انتخاب (کلید.نقاب_type) {
                حالت GGML_TYPE_F32:
                    متعین_کرتا.push_back("MASK_F32");
                    variant += "_mask_f32";
                    توڑو;
                حالت GGML_TYPE_F16:
                    متعین_کرتا.push_back("MASK_F16");
                    variant += "_mask_f16";
                    توڑو;
                طےشدہ:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        اگر (کلید.has_sink) {
            متعین_کرتا.push_back("HAS_SINK");
            variant += "_sink";
        }

        اگر (کلید.inplace) {
            متعین_کرتا.push_back("INPLACE");
            variant += "_inplace";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_soft_max, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[کلید]  = pipeline;
        واپس soft_max_pipelines[کلید];
    }

    webgpu_pipeline get_conv2d_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_کلید کلید = {};
        کلید.وزن_type                     = context.src0->type;
        کلید.داخلہ_type                      = context.src1->type;
        کلید.output_type                     = context.dst->type;

        خودکار it = conv2d_pipelines.find(کلید);
        اگر (it != conv2d_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "conv_2d";

        خودکار ڈالو_type_متعین_کرتا = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_سابقہ = prefix;
            اگر (type == GGML_TYPE_F32) {
                متعین_کرتا.push_back(s_سابقہ + "=f32");
            } ورنہ اگر (type == GGML_TYPE_F16) {
                متعین_کرتا.push_back(s_سابقہ + "=f16");
            } ورنہ {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ڈالو_type_متعین_کرتا("WEIGHT_TYPE", کلید.وزن_type);
        ڈالو_type_متعین_کرتا("INPUT_TYPE", کلید.داخلہ_type);
        ڈالو_type_متعین_کرتا("OUTPUT_TYPE", کلید.output_type);

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_conv2d, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[کلید]    = pipeline;
        واپس conv2d_pipelines[کلید];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقی whcn) {
        ggml_webgpu_conv2d_dw_pipeline_کلید کلید = {};
        کلید.وزن_type                        = context.src0->type;
        کلید.داخلہ_type                         = context.src1->type;
        کلید.output_type                        = context.dst->type;
        کلید.whcn                               = whcn;

        خودکار it = conv2d_dw_pipelines.find(کلید);
        اگر (it != conv2d_dw_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        خودکار ڈالو_type_متعین_کرتا = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_سابقہ = prefix;
            اگر (type == GGML_TYPE_F32) {
                متعین_کرتا.push_back(s_سابقہ + "=f32");
            } ورنہ اگر (type == GGML_TYPE_F16) {
                متعین_کرتا.push_back(s_سابقہ + "=f16");
            } ورنہ {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ڈالو_type_متعین_کرتا("WEIGHT_TYPE", کلید.وزن_type);
        ڈالو_type_متعین_کرتا("INPUT_TYPE", کلید.داخلہ_type);
        ڈالو_type_متعین_کرتا("OUTPUT_TYPE", کلید.output_type);

        اگر (whcn) {
            متعین_کرتا.push_back("WHCN");
        }
        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_conv2d_dw, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[کلید] = pipeline;
        واپس conv2d_dw_pipelines[کلید];
    }

    webgpu_pipeline get_im2col_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_کلید کلید = {};
        کلید.داخلہ_type                      = context.src1->type;
        کلید.output_type                     = context.dst->type;

        خودکار it = im2col_pipelines.find(کلید);
        اگر (it != im2col_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "im2col";

        خودکار ڈالو_type_متعین_کرتا = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_سابقہ = prefix;
            اگر (type == GGML_TYPE_F32) {
                متعین_کرتا.push_back(s_سابقہ + "=f32");
            } ورنہ اگر (type == GGML_TYPE_F16) {
                متعین_کرتا.push_back(s_سابقہ + "=f16");
            } ورنہ {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ڈالو_type_متعین_کرتا("INPUT_TYPE", کلید.داخلہ_type);
        ڈالو_type_متعین_کرتا("OUTPUT_TYPE", کلید.output_type);

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_im2col, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[کلید]    = pipeline;
        واپس im2col_pipelines[کلید];
    }

    webgpu_pipeline get_upscale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت uint32_t منوال_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ثابت uint32_t بنیاد_منوال  = منوال_flags & 0xFFu;
        ثابت منطقی     antialias  = (منوال_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_کلید کلید = {};
        کلید.داخلہ_type                       = context.src0->type;
        کلید.output_type                      = context.dst->type;
        کلید.بنیاد_منوال                        = بنیاد_منوال;
        کلید.antialias                        = antialias;

        خودکار it = upscale_pipelines.find(کلید);
        اگر (it != upscale_pipelines.end()) {
            واپس it->second;
        }

        std::vector<std::string> متعین_کرتا;
        std::string              variant = "upscale";

        اگر (کلید.داخلہ_type == GGML_TYPE_F16) {
            متعین_کرتا.push_back("SRC_F16");
            variant += "_src_f16";
        } ورنہ {
            variant += "_src_f32";
        }

        اگر (کلید.output_type == GGML_TYPE_F16) {
            متعین_کرتا.push_back("DST_F16");
            variant += "_dst_f16";
        } ورنہ {
            variant += "_dst_f32";
        }

        انتخاب (بنیاد_منوال) {
            حالت GGML_SCALE_منوال_قریب_ترین:
                متعین_کرتا.push_back("NEAREST");
                variant += "_nearest";
                توڑو;
            حالت GGML_SCALE_منوال_BILINEAR:
                متعین_کرتا.push_back("BILINEAR");
                variant += "_bilinear";
                توڑو;
            حالت GGML_SCALE_منوال_BICUBIC:
                متعین_کرتا.push_back("BICUBIC");
                variant += "_bicubic";
                توڑو;
            طےشدہ:
                GGML_ABORT("Unsupported upscale mode");
        }

        اگر (antialias) {
            متعین_کرتا.push_back("ANTIALIAS");
            variant += "_aa";
        }

        متعین_کرتا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_سائز));

        خودکار processed           = preprocessor.preprocess(wgsl_upscale, متعین_کرتا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_سائز       = context.max_wg_سائز;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[کلید]   = pipeline;
        واپس upscale_pipelines[کلید];
    }

  نجی:
    ساکن webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_رمز,
                                                       std::string    label) {
        wgpu::Shader_ماخذ_WGSL shader_ماخذ;
        shader_ماخذ.code = shader_رمز.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.اگلا_اندر_Chain = &shader_ماخذ;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.اندراج_نقطہ_2 = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = خالی_اشاریہ;  // nullptr means auto layout
        واپس { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#اگر_ختم  // GGML_WEBGPU_SHADER_LIB_HPP
