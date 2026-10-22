#جے_بنا_تعریف GGML_WEBGPU_SHADER_LIB_HPP
#تعریف_کرو GGML_WEBGPU_SHADER_LIB_HPP

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

#تعریف_کرو GGML_WEBGPU_F16_ناپ_بائٹاں                   2
#تعریف_کرو GGML_WEBGPU_F32_ناپ_بائٹاں                   4
#تعریف_کرو GGML_WEBGPU_I32_ناپ_بائٹاں                   4
#تعریف_کرو GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#تعریف_کرو GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#تعریف_کرو GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#تعریف_کرو GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#تعریف_کرو GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_ناپ     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#تعریف_کرو GGML_WEBGPU_KV_SEQ_PAD                       256u

#تعریف_کرو GGML_WEBGPU_ARGSORT_ملاؤ_MAX_WG_ناپ 512u

// Matrix multiplication parameters

// Register tiling parameters
#تعریف_کرو WEBGPU_MUL_MAT_TILE_M           4
#تعریف_کرو WEBGPU_MUL_MAT_TILE_N           4
#تعریف_کرو WEBGPU_MUL_MAT_WG_ناپ_M        8
#تعریف_کرو WEBGPU_MUL_MAT_WG_ناپ_N        8
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
#تعریف_کرو WEBGPU_MUL_MAT_VEC_WG_ناپ 256

#تعریف_کرو WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_فی_WG    4
#تعریف_کرو WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG 4
#تعریف_کرو WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_فی_WG      4

// default size for reg-tile matrix multiplication
#تعریف_کرو WEBGPU_MUL_MAT_WG_ناپ 256

// Same hash combine function as in boost
سانچا <قسم_ناں T> مضمن فراغ ggml_webgpu_hash_combine(size_t & بیج, ثابت T & value) {
    بیج ^= std::hash<T>{}(value) + 0x9e3779b9 + (بیج << 6) + (بیج >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
مضمن uintptr_t ggml_webgpu_tensor_addr(ثابت ggml_tensor * tensor) {
    ثابت ggml_tensor * ادھار_tensor = tensor->view_src ? tensor->view_src : tensor;
    أرجع (uintptr_t) ادھار_tensor->انکڑے + tensor->view_offs;
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

    uint32_t    max_wg_ناپ;
    size_t      wg_mem_حد_بندی_بائٹاں       = 0;
    منطقي        supports_subgroups       = خطأ;
    منطقي        supports_subgroup_میٹرکس = خطأ;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_ناپ        = 0;
    uint32_t    max_subgroup_ناپ        = 0;
    منطقي        supports_dot_ضرب_حاصل     = خطأ;
    std::string vendor;
};

بنية webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<فراغ> context = خالی_سوچک;
};

بنية ggml_webgpu_generic_shader_decisions {
    uint32_t wg_ناپ = 0;
    منطقي     inplace = خطأ;
};

بنية ggml_webgpu_get_صفاں_shader_decisions {
    uint32_t wg_ناپ    = 0;
    منطقي     vectorized = خطأ;
};

بنية ggml_webgpu_ثنائی_shader_decisions {
    uint32_t wg_ناپ     = 0;
    منطقي     inplace     = خطأ;
    منطقي     overlap     = خطأ;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_glu_shader_decisions {
    uint32_t wg_ناپ     = 0;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<فراغ> decisions;
};

بنية ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t بلاک_ناپ_3;
    uint32_t ٹوکن_فی_wg;
};

بنية ggml_webgpu_ssm_جانچ_pipeline_کلید {
    صحيح     type;
    صحيح     d_state;
    uint8_t xbc_overlap;
    منطقي    a_overlap;
    منطقي    ids_overlap;

    منطقي عامل==(ثابت ggml_webgpu_ssm_جانچ_pipeline_کلید & other) ثابت {
        أرجع type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

بنية ggml_webgpu_ssm_جانچ_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_ssm_جانچ_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.d_state);
        ggml_webgpu_hash_combine(بیج, کلید.xbc_overlap);
        ggml_webgpu_hash_combine(بیج, کلید.a_overlap);
        ggml_webgpu_hash_combine(بیج, کلید.ids_overlap);
        أرجع بیج;
    }
};

بنية ggml_webgpu_ssm_جانچ_shader_decisions {
    uint32_t wg_ناپ;
    uint32_t ٹوکن_فی_tile;
    uint8_t  xbc_overlap = 0;
    منطقي     a_overlap   = خطأ;
    منطقي     ids_overlap = خطأ;
};

/** Argsort **/

بنية ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_ناپ;
    size_t   wg_mem_حد_بندی_بائٹاں;
    int32_t  ترتیب;
};

/** Set Rows **/

بنية ggml_webgpu_سیٹ_صفاں_pipeline_کلید {
    صحيح src0_type;
    صحيح dst_type;
    صحيح vec4;
    صحيح i64_idx;
    صحيح pair_بلاکاں;

    منطقي عامل==(ثابت ggml_webgpu_سیٹ_صفاں_pipeline_کلید & other) ثابت {
        أرجع src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_بلاکاں == other.pair_بلاکاں;
    }
};

بنية ggml_webgpu_سیٹ_صفاں_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_سیٹ_صفاں_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src0_type);
        ggml_webgpu_hash_combine(بیج, کلید.dst_type);
        ggml_webgpu_hash_combine(بیج, کلید.vec4);
        ggml_webgpu_hash_combine(بیج, کلید.i64_idx);
        ggml_webgpu_hash_combine(بیج, کلید.pair_بلاکاں);
        أرجع بیج;
    }
};

بنية ggml_webgpu_سیٹ_صفاں_shader_decisions {
    منطقي     vec4;
    منطقي     i64_idx;
    منطقي     pair_بلاکاں;
    uint32_t wg_ناپ;
};

/** Set **/

بنية ggml_webgpu_سیٹ_pipeline_کلید {
    ggml_type type;
    منطقي      inplace;

    منطقي عامل==(ثابت ggml_webgpu_سیٹ_pipeline_کلید & other) ثابت {
        أرجع type == other.type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_سیٹ_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_سیٹ_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        أرجع بیج;
    }
};

/** Get Rows **/

بنية ggml_webgpu_get_صفاں_pipeline_کلید {
    ggml_type src_type;
    صحيح       vectorized;

    منطقي عامل==(ثابت ggml_webgpu_get_صفاں_pipeline_کلید & other) ثابت {
        أرجع src_type == other.src_type && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_get_صفاں_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_get_صفاں_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src_type);
        ggml_webgpu_hash_combine(بیج, کلید.vectorized);
        أرجع بیج;
    }
};

/** Row Norm **/

بنية ggml_webgpu_صف_norm_pipeline_کلید {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    منطقي      inplace;

    منطقي عامل==(ثابت ggml_webgpu_صف_norm_pipeline_کلید & other) ثابت {
        أرجع op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_صف_norm_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_صف_norm_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.op);
        ggml_webgpu_hash_combine(بیج, کلید.src_type);
        ggml_webgpu_hash_combine(بیج, کلید.dst_type);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        أرجع بیج;
    }
};

/** RMS_NORM + MUL **/

بنية ggml_webgpu_rms_norm_mul_pipeline_کلید {
    منطقي inplace;      // rn_src == dst
    منطقي overlap;      // mul_src == dst
    منطقي src_overlap;  // rn_src binding overlaps mul_src binding

    منطقي عامل==(ثابت ggml_webgpu_rms_norm_mul_pipeline_کلید & other) ثابت {
        أرجع inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_rms_norm_mul_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_rms_norm_mul_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        ggml_webgpu_hash_combine(بیج, کلید.overlap);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        أرجع بیج;
    }
};

بنية ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_ناپ     = 0;
    منطقي     inplace     = خطأ;
    منطقي     overlap     = خطأ;
    منطقي     src_overlap = خطأ;
};

/** Pad **/
بنية ggml_webgpu_pad_pipeline_کلید {
    منطقي circular;

    منطقي عامل==(ثابت ggml_webgpu_pad_pipeline_کلید & other) ثابت { أرجع circular == other.circular; }
};

بنية ggml_webgpu_pad_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_pad_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.circular);
        أرجع بیج;
    }
};

/** Solve Tri **/
بنية ggml_webgpu_solve_tri_pipeline_کلید {
    صحيح type;
    صحيح n;
    صحيح k;

    منطقي عامل==(ثابت ggml_webgpu_solve_tri_pipeline_کلید & other) ثابت {
        أرجع type == other.type && n == other.n && k == other.k;
    }
};

بنية ggml_webgpu_solve_tri_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_solve_tri_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.n);
        ggml_webgpu_hash_combine(بیج, کلید.k);
        أرجع بیج;
    }
};

/** SSM Conv **/
بنية ggml_webgpu_ssm_conv_pipeline_کلید {
    صحيح type;
    صحيح vectorized;

    منطقي عامل==(ثابت ggml_webgpu_ssm_conv_pipeline_کلید & other) ثابت {
        أرجع type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
بنية ggml_webgpu_conv2d_pipeline_کلید {
    ggml_type بھار_type;
    ggml_type داخلہ_type;
    ggml_type output_type;

    منطقي عامل==(ثابت ggml_webgpu_conv2d_pipeline_کلید & other) ثابت {
        أرجع بھار_type == other.بھار_type && داخلہ_type == other.داخلہ_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_conv2d_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_conv2d_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.بھار_type);
        ggml_webgpu_hash_combine(بیج, کلید.داخلہ_type);
        ggml_webgpu_hash_combine(بیج, کلید.output_type);
        أرجع بیج;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
بنية ggml_webgpu_conv2d_dw_pipeline_کلید {
    ggml_type بھار_type;
    ggml_type داخلہ_type;
    ggml_type output_type;
    منطقي      whcn;

    منطقي عامل==(ثابت ggml_webgpu_conv2d_dw_pipeline_کلید & other) ثابت {
        أرجع بھار_type == other.بھار_type && داخلہ_type == other.داخلہ_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

بنية ggml_webgpu_conv2d_dw_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_conv2d_dw_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.بھار_type);
        ggml_webgpu_hash_combine(بیج, کلید.داخلہ_type);
        ggml_webgpu_hash_combine(بیج, کلید.output_type);
        ggml_webgpu_hash_combine(بیج, کلید.whcn);
        أرجع بیج;
    }
};

/** Im2Col **/
بنية ggml_webgpu_im2col_pipeline_کلید {
    ggml_type داخلہ_type;
    ggml_type output_type;

    منطقي عامل==(ثابت ggml_webgpu_im2col_pipeline_کلید & other) ثابت {
        أرجع داخلہ_type == other.داخلہ_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_im2col_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_im2col_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.داخلہ_type);
        ggml_webgpu_hash_combine(بیج, کلید.output_type);
        أرجع بیج;
    }
};

/** Gated Delta Net **/
بنية ggml_webgpu_gated_delta_net_pipeline_کلید {
    صحيح  type;
    صحيح  s_v;
    صحيح  kda;
    منطقي fused_cache;

    منطقي عامل==(ثابت ggml_webgpu_gated_delta_net_pipeline_کلید & other) ثابت {
        أرجع type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

بنية ggml_webgpu_gated_delta_net_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_gated_delta_net_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.s_v);
        ggml_webgpu_hash_combine(بیج, کلید.kda);
        أرجع بیج;
    }
};

بنية ggml_webgpu_ssm_conv_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_ssm_conv_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.vectorized);
        أرجع بیج;
    }
};

/** Scale **/

بنية ggml_webgpu_scale_pipeline_کلید {
    صحيح inplace;

    منطقي عامل==(ثابت ggml_webgpu_scale_pipeline_کلید & other) ثابت { أرجع inplace == other.inplace; }
};

بنية ggml_webgpu_scale_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_scale_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        أرجع بیج;
    }
};

/** Upscale **/

بنية ggml_webgpu_upscale_pipeline_کلید {
    ggml_type داخلہ_type;
    ggml_type output_type;
    uint32_t  ادھار_منوال;
    منطقي      antialias;

    منطقي عامل==(ثابت ggml_webgpu_upscale_pipeline_کلید & other) ثابت {
        أرجع داخلہ_type == other.داخلہ_type && output_type == other.output_type && ادھار_منوال == other.ادھار_منوال &&
               antialias == other.antialias;
    }
};

بنية ggml_webgpu_upscale_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_upscale_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.داخلہ_type);
        ggml_webgpu_hash_combine(بیج, کلید.output_type);
        ggml_webgpu_hash_combine(بیج, کلید.ادھار_منوال);
        ggml_webgpu_hash_combine(بیج, کلید.antialias);
        أرجع بیج;
    }
};

/** Concat **/

بنية ggml_webgpu_concat_pipeline_کلید {
    صحيح  type;
    منطقي src_overlap;

    منطقي عامل==(ثابت ggml_webgpu_concat_pipeline_کلید & other) ثابت {
        أرجع type == other.type && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_concat_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_concat_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        أرجع بیج;
    }
};

/** Repeat **/

بنية ggml_webgpu_repeat_pipeline_کلید {
    صحيح type;

    منطقي عامل==(ثابت ggml_webgpu_repeat_pipeline_کلید & other) ثابت { أرجع type == other.type; }
};

بنية ggml_webgpu_repeat_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_repeat_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        أرجع بیج;
    }
};

/** Binary **/

بنية ggml_webgpu_ثنائی_pipeline_کلید {
    صحيح  type;
    صحيح  op;
    منطقي inplace;
    منطقي overlap;
    منطقي src_overlap;

    منطقي عامل==(ثابت ggml_webgpu_ثنائی_pipeline_کلید & other) ثابت {
        أرجع type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_ثنائی_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_ثنائی_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.op);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        ggml_webgpu_hash_combine(بیج, کلید.overlap);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        أرجع بیج;
    }
};

/* Add_Id */

بنية ggml_webgpu_جوڑو_id_pipeline_کلید {
    منطقي inplace;

    منطقي عامل==(ثابت ggml_webgpu_جوڑو_id_pipeline_کلید & other) ثابت { أرجع inplace == other.inplace; }
};

بنية ggml_webgpu_جوڑو_id_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_جوڑو_id_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        أرجع بیج;
    }
};

/** Unary **/

بنية ggml_webgpu_unary_pipeline_کلید {
    صحيح           type;
    صحيح           op;
    منطقي          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    منطقي          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    منطقي عامل==(ثابت ggml_webgpu_unary_pipeline_کلید & other) ثابت {
        أرجع type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

بنية ggml_webgpu_unary_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_unary_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.op);
        ggml_webgpu_hash_combine(بیج, کلید.is_unary);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        ggml_webgpu_hash_combine(بیج, کلید.ttype);
        أرجع بیج;
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
    منطقي      k_سدھا;
    منطقي      v_سدھا;
    منطقي      kv_overlap;
    منطقي      has_نقاب;
    منطقي      has_sinks;
    منطقي      ورتدا_logit_softcap;

    منطقي عامل==(ثابت ggml_webgpu_flash_attn_common_pipeline_کلید & other) ثابت {
        أرجع q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && سر_dim_qk == other.سر_dim_qk && سر_dim_v == other.سر_dim_v &&
               k_سدھا == other.k_سدھا && v_سدھا == other.v_سدھا && kv_overlap == other.kv_overlap &&
               has_نقاب == other.has_نقاب && has_sinks == other.has_sinks &&
               ورتدا_logit_softcap == other.ورتدا_logit_softcap;
    }
};

مضمن فراغ ggml_webgpu_flash_attn_hash_common_pipeline_کلید(size_t &                                           بیج,
                                                            ثابت ggml_webgpu_flash_attn_common_pipeline_کلید & کلید) {
    ggml_webgpu_hash_combine(بیج, کلید.q_type);
    ggml_webgpu_hash_combine(بیج, کلید.k_type);
    ggml_webgpu_hash_combine(بیج, کلید.v_type);
    ggml_webgpu_hash_combine(بیج, کلید.dst_type);
    ggml_webgpu_hash_combine(بیج, کلید.سر_dim_qk);
    ggml_webgpu_hash_combine(بیج, کلید.سر_dim_v);
    ggml_webgpu_hash_combine(بیج, کلید.k_سدھا);
    ggml_webgpu_hash_combine(بیج, کلید.v_سدھا);
    ggml_webgpu_hash_combine(بیج, کلید.kv_overlap);
    ggml_webgpu_hash_combine(بیج, کلید.has_نقاب);
    ggml_webgpu_hash_combine(بیج, کلید.has_sinks);
    ggml_webgpu_hash_combine(بیج, کلید.ورتدا_logit_softcap);
}

بنية ggml_webgpu_flash_attn_vec_pipeline_کلید {
    ggml_webgpu_flash_attn_common_pipeline_کلید common;

    منطقي عامل==(ثابت ggml_webgpu_flash_attn_vec_pipeline_کلید & other) ثابت { أرجع common == other.common; }
};

بنية ggml_webgpu_flash_attn_vec_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_vec_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_کلید(بیج, کلید.common);
        أرجع بیج;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_کلید {
    ggml_webgpu_flash_attn_common_pipeline_کلید common;
    منطقي                                       use_sg_میٹرکس;

    منطقي عامل==(ثابت ggml_webgpu_flash_attn_pipeline_کلید & other) ثابت {
        أرجع common == other.common && use_sg_میٹرکس == other.use_sg_میٹرکس;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_کلید(بیج, کلید.common);
        ggml_webgpu_hash_combine(بیج, کلید.use_sg_میٹرکس);
        أرجع بیج;
    }
};

بنية ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_ناپ = 0;
};

بنية ggml_webgpu_flash_attn_decisions {
    منطقي     use_sg_میٹرکس = خطأ;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_ناپ       = 0;
};

مضمن constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی = 4u;
مضمن constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

مضمن size_t ggml_webgpu_flash_attn_tensor_وتھ(ثابت ggml_tensor * tensor) {
    constexpr uintptr_t ptr_ادھار_addr = 0x1000u;
    ثابت ggml_tensor * ادھار          = tensor->view_src != خالی_سوچک ? tensor->view_src : tensor;
    أرجع reinterpret_cast<uintptr_t>(ادھار->انکڑے) - ptr_ادھار_addr + tensor->view_offs;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K, size_t storage_وتھ_alignment) {
    ثابت uint32_t وتھ_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_وتھ(K) & (storage_وتھ_alignment - 1)) /
                    ggml_type_ناپ(K->type));
    أرجع وتھ_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی == 0u;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K,
                                                      ثابت ggml_tensor * V,
                                                      size_t              storage_وتھ_alignment) {
    أرجع ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_وتھ_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_وتھ_alignment);
}

مضمن منطقي ggml_webgpu_flash_attn_k_سدھا(ثابت ggml_tensor * Q, ثابت ggml_tensor * K, uint32_t kv_سدھا_align) {
    أرجع (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_سدھا_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

مضمن منطقي ggml_webgpu_flash_attn_v_سدھا(ثابت ggml_tensor * Q, ثابت ggml_tensor * V, uint32_t kv_سدھا_align) {
    أرجع ggml_webgpu_flash_attn_k_سدھا(Q, V, kv_سدھا_align);
}

مضمن ggml_webgpu_flash_attn_common_pipeline_کلید ggml_webgpu_flash_attn_make_common_pipeline_کلید(
    ثابت ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_سدھا_align,
    منطقي                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_کلید کلید = {};
    کلید.q_type                                     = context.src0->type;
    کلید.k_type                                     = context.src1->type;
    کلید.v_type                                     = context.src2->type;
    کلید.dst_type                                   = context.dst->type;
    کلید.سر_dim_qk                                = (uint32_t) context.src0->ne[0];
    کلید.سر_dim_v                                 = (uint32_t) context.src2->ne[0];
    کلید.k_سدھا           = ggml_webgpu_flash_attn_k_سدھا(context.src0, context.src1, kv_سدھا_align);
    کلید.v_سدھا           = ggml_webgpu_flash_attn_v_سدھا(context.src0, context.src2, kv_سدھا_align);
    کلید.kv_overlap         = kv_overlap;
    کلید.has_نقاب           = context.src3 != خالی_سوچک;
    کلید.has_sinks          = context.src4 != خالی_سوچک;
    کلید.ورتدا_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    أرجع کلید;
}

مضمن std::vector<std::string> ggml_webgpu_flash_attn_common_مقرر_کردا(
    ثابت ggml_webgpu_flash_attn_common_pipeline_کلید & کلید,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_ناپ) {
    std::vector<std::string> مقرر_کردا;

    بدل (کلید.k_type) {
        حالة GGML_TYPE_F32:
            مقرر_کردا.push_back("K_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            مقرر_کردا.push_back("K_F16");
            اكسر;
        حالة GGML_TYPE_Q4_0:
            مقرر_کردا.push_back("K_Q4_0");
            اكسر;
        حالة GGML_TYPE_Q8_0:
            مقرر_کردا.push_back("K_Q8_0");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(کلید.k_type);

    بدل (کلید.v_type) {
        حالة GGML_TYPE_F32:
            مقرر_کردا.push_back("V_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            مقرر_کردا.push_back("V_F16");
            اكسر;
        حالة GGML_TYPE_Q4_0:
            مقرر_کردا.push_back("V_Q4_0");
            اكسر;
        حالة GGML_TYPE_Q8_0:
            مقرر_کردا.push_back("V_Q8_0");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(کلید.v_type);

    بدل (کلید.q_type) {
        حالة GGML_TYPE_F32:
            مقرر_کردا.push_back("Q_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            مقرر_کردا.push_back("Q_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(کلید.q_type);

    بدل (کلید.dst_type) {
        حالة GGML_TYPE_F32:
            مقرر_کردا.push_back("DST_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            مقرر_کردا.push_back("DST_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(کلید.dst_type);

    إذا (کلید.has_نقاب) {
        مقرر_کردا.push_back("MASK");
        variant += "_mask";
    }
    إذا (کلید.has_sinks) {
        مقرر_کردا.push_back("SINKS");
        variant += "_sinks";
    }
    إذا (کلید.ورتدا_logit_softcap) {
        مقرر_کردا.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    إذا (کلید.k_سدھا) {
        مقرر_کردا.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    إذا (کلید.v_سدھا) {
        مقرر_کردا.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    إذا (کلید.kv_overlap) {
        مقرر_کردا.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    مقرر_کردا.push_back(std::string("HEAD_DIM_QK=") + std::to_string(کلید.سر_dim_qk));
    variant += std::string("_hsqk") + std::to_string(کلید.سر_dim_qk);

    مقرر_کردا.push_back(std::string("HEAD_DIM_V=") + std::to_string(کلید.سر_dim_v));
    variant += std::string("_hsv") + std::to_string(کلید.سر_dim_v);

    مقرر_کردا.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    مقرر_کردا.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(wg_ناپ));

    إذا (ggml_is_quantized(کلید.k_type) || ggml_is_quantized(کلید.v_type)) {
        مقرر_کردا.push_back("U32_DEQUANT_HELPERS");
        إذا (ggml_is_quantized(کلید.k_type)) {
            مقرر_کردا.push_back("LOADERS_QUANTIZED_K");
        }
        إذا (ggml_is_quantized(کلید.v_type)) {
            مقرر_کردا.push_back("LOADERS_QUANTIZED_V");
        }
    }

    أرجع مقرر_کردا;
}

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید {
    uint32_t  سر_dim_v;
    uint32_t  wg_ناپ;
    ggml_type dst_type;
};

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.سر_dim_v);
        ggml_webgpu_hash_combine(بیج, کلید.wg_ناپ);
        ggml_webgpu_hash_combine(بیج, کلید.dst_type);
        أرجع بیج;
    }
};

مضمن منطقي عامل==(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید & lhs,
                       ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید & rhs) {
    أرجع lhs.سر_dim_v == rhs.سر_dim_v && lhs.wg_ناپ == rhs.wg_ناپ && lhs.dst_type == rhs.dst_type;
}

بنية ggml_webgpu_flash_attn_blk_pipeline_کلید {
    uint32_t kv_tile;

    منطقي عامل==(ثابت ggml_webgpu_flash_attn_blk_pipeline_کلید & other) ثابت { أرجع kv_tile == other.kv_tile; }
};

بنية ggml_webgpu_flash_attn_blk_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_flash_attn_blk_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.kv_tile);
        أرجع بیج;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
مضمن size_t ggml_webgpu_flash_attn_wg_mem_بائٹاں(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t سر_dim_qk,
                                                  uint32_t سر_dim_v,
                                                  منطقي     has_نقاب,
                                                  منطقي     kv_سدھا) {
    ثابت uint32_t max_سر_dim = std::max(سر_dim_qk, سر_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * سر_dim_qk;        // q_shmem
    إذا (!kv_سدھا) {
        f32_elems += kv_tile * max_سر_dim;  // kv_shmem
    }
    f32_elems += q_tile * سر_dim_v;         // o_shmem
    إذا (has_نقاب) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    أرجع f16_elems * GGML_WEBGPU_F16_ناپ_بائٹاں + f32_elems * GGML_WEBGPU_F32_ناپ_بائٹاں;
}

مضمن uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   حد_بندی_بائٹاں,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t سر_dim_qk,
                                                   uint32_t سر_dim_v,
                                                   منطقي     has_نقاب,
                                                   منطقي     kv_سدھا) {
    ثابت size_t ادھار_q_بائٹاں =
        ggml_webgpu_flash_attn_wg_mem_بائٹاں(q_tile, 0, سر_dim_qk, سر_dim_v, has_نقاب, kv_سدھا);
    إذا (حد_بندی_بائٹاں <= ادھار_q_بائٹاں) {
        أرجع 0;
    }
    ثابت size_t one_kv_بائٹاں =
        ggml_webgpu_flash_attn_wg_mem_بائٹاں(q_tile, 1, سر_dim_qk, سر_dim_v, has_نقاب, kv_سدھا);
    ثابت size_t بائٹاں_فی_kv = one_kv_بائٹاں - ادھار_q_بائٹاں;
    إذا (بائٹاں_فی_kv == 0) {
        أرجع 0;
    }
    ثابت size_t max_kv_tile = (حد_بندی_بائٹاں - ادھار_q_بائٹاں) / بائٹاں_فی_kv;
    أرجع (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

مضمن uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_حد_بندی_بائٹاں,
                                                       uint32_t سر_dim_qk,
                                                       uint32_t سر_dim_v,
                                                       منطقي     has_نقاب,
                                                       منطقي     kv_سدھا) {
    ثابت uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_حد_بندی_بائٹاں, 1u, 1u, سر_dim_qk, سر_dim_v, has_نقاب, kv_سدھا);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    إذا (kv_سدھا) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        طالما (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    أرجع kv_tile;
}

مضمن منطقي ggml_webgpu_flash_attn_can_use_subgroup_میٹرکس_رستہ(منطقي                supports_subgroup_میٹرکس,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ثابت ggml_tensor * Q,
                                                                ثابت ggml_tensor * V) {
    أرجع supports_subgroup_میٹرکس && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

بنية ggml_webgpu_mul_mat_vec_pipeline_کلید {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    uint32_t  عدد_خانے_گنتی;
    منطقي      use_mmvq;
    منطقي      src_overlap;

    منطقي عامل==(ثابت ggml_webgpu_mul_mat_vec_pipeline_کلید & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               عدد_خانے_گنتی == other.عدد_خانے_گنتی && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_vec_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_vec_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src0_type);
        ggml_webgpu_hash_combine(بیج, کلید.src1_type);
        ggml_webgpu_hash_combine(بیج, کلید.vectorized);
        ggml_webgpu_hash_combine(بیج, کلید.عدد_خانے_گنتی);
        ggml_webgpu_hash_combine(بیج, کلید.use_mmvq);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        أرجع بیج;
    }
};

بنية ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_ناپ;
    uint32_t outputs_فی_wg;
    uint32_t vec_ناپ;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_quantize_q8_pipeline_کلید {
    ggml_type src0_type;

    منطقي عامل==(ثابت ggml_webgpu_quantize_q8_pipeline_کلید & other) ثابت { أرجع src0_type == other.src0_type; }
};

بنية ggml_webgpu_quantize_q8_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_quantize_q8_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src0_type);
        أرجع بیج;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_کلید {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    صحيح       use_subgroup_میٹرکس;
    منطقي      src_overlap;

    منطقي عامل==(ثابت ggml_webgpu_mul_mat_pipeline_کلید & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_میٹرکس == other.use_subgroup_میٹرکس && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src0_type);
        ggml_webgpu_hash_combine(بیج, کلید.src1_type);
        ggml_webgpu_hash_combine(بیج, کلید.vectorized);
        ggml_webgpu_hash_combine(بیج, کلید.use_subgroup_میٹرکس);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        أرجع بیج;
    }
};

بنية ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_ناپ_m;
    uint32_t wg_ناپ_n;
    uint32_t wg_ناپ;
    uint32_t outputs_فی_wg;
    صحيح      use_subgroup_میٹرکس;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_میٹرکس_m;
    uint32_t subgroup_میٹرکس_n;

    uint32_t mul_mat_wg_ناپ;
    منطقي     src_overlap = خطأ;
};

/** MUL_MAT_ID **/

بنية ggml_webgpu_mul_mat_id_gather_pipeline_کلید {
    uint32_t n_experts;

    منطقي عامل==(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_کلید & other) ثابت {
        أرجع n_experts == other.n_experts;
    }
};

بنية ggml_webgpu_mul_mat_id_gather_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.n_experts);
        أرجع بیج;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_کلید {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  عدد_خانے_گنتی;
    صحيح       vectorized;

    منطقي عامل==(ثابت ggml_webgpu_mul_mat_id_pipeline_کلید & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               عدد_خانے_گنتی == other.عدد_خانے_گنتی && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_mul_mat_id_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src0_type);
        ggml_webgpu_hash_combine(بیج, کلید.src1_type);
        ggml_webgpu_hash_combine(بیج, کلید.n_experts);
        ggml_webgpu_hash_combine(بیج, کلید.عدد_خانے_گنتی);
        ggml_webgpu_hash_combine(بیج, کلید.vectorized);
        أرجع بیج;
    }
};

/** Cpy **/

بنية ggml_webgpu_cpy_pipeline_کلید {
    ggml_type src_type;
    ggml_type dst_type;

    منطقي عامل==(ثابت ggml_webgpu_cpy_pipeline_کلید & other) ثابت {
        أرجع src_type == other.src_type && dst_type == other.dst_type;
    }
};

بنية ggml_webgpu_cpy_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_cpy_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.src_type);
        ggml_webgpu_hash_combine(بیج, کلید.dst_type);
        أرجع بیج;
    }
};

/** Glu **/

بنية ggml_webgpu_glu_pipeline_کلید {
    ggml_glu_op glu_op;
    ggml_type   type;
    منطقي        ونڈو;
    منطقي        src_overlap;

    منطقي عامل==(ثابت ggml_webgpu_glu_pipeline_کلید & other) ثابت {
        أرجع glu_op == other.glu_op && type == other.type && ونڈو == other.ونڈو && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_glu_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_glu_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.glu_op);
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.ونڈو);
        ggml_webgpu_hash_combine(بیج, کلید.src_overlap);
        أرجع بیج;
    }
};

/** Rope **/

بنية ggml_webgpu_rope_pipeline_کلید {
    ggml_type type;
    منطقي      inplace;
    منطقي      has_ff;

    منطقي عامل==(ثابت ggml_webgpu_rope_pipeline_کلید & other) ثابت {
        أرجع type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

بنية ggml_webgpu_rope_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_rope_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.type);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        ggml_webgpu_hash_combine(بیج, کلید.has_ff);
        أرجع بیج;
    }
};

/** SoftMax **/

بنية ggml_webgpu_soft_max_pipeline_کلید {
    ggml_type نقاب_type;
    منطقي      has_نقاب;
    منطقي      has_sink;
    منطقي      inplace;

    منطقي عامل==(ثابت ggml_webgpu_soft_max_pipeline_کلید & other) ثابت {
        أرجع نقاب_type == other.نقاب_type && has_نقاب == other.has_نقاب && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

بنية ggml_webgpu_soft_max_pipeline_کلید_hash {
    size_t عامل()(ثابت ggml_webgpu_soft_max_pipeline_کلید & کلید) ثابت {
        size_t بیج = 0;
        ggml_webgpu_hash_combine(بیج, کلید.نقاب_type);
        ggml_webgpu_hash_combine(بیج, کلید.has_نقاب);
        ggml_webgpu_hash_combine(بیج, کلید.has_sink);
        ggml_webgpu_hash_combine(بیج, کلید.inplace);
        أرجع بیج;
    }
};

/** MMVQ **/

مضمن منطقي ggml_webgpu_can_use_mmvq(ثابت ggml_tensor * src0,
                                     ثابت ggml_tensor * src1,
                                     منطقي                supports_dot_ضرب_حاصل,
                                     ثابت std::string & vendor) {
    إذا (src1->ne[1] <= 4) {
        منطقي supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        إذا (supports_dp4a && supports_dot_ضرب_حاصل) {
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

    std::unordered_map<صحيح, webgpu_pipeline> جوڑ_صفاں_pipelines;       // key is fixed, no variants yet
    std::unordered_map<صحيح, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<صحيح, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<صحيح, webgpu_pipeline> argsort_ملاؤ_pipelines;  // key is order
    std::unordered_map<صحيح, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_صف_norm_pipeline_کلید, webgpu_pipeline, ggml_webgpu_صف_norm_pipeline_کلید_hash>
        صف_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_صفاں_pipeline_کلید, webgpu_pipeline, ggml_webgpu_get_صفاں_pipeline_کلید_hash>
        get_صفاں_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_کلید, webgpu_pipeline, ggml_webgpu_unary_pipeline_کلید_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_کلید, webgpu_pipeline, ggml_webgpu_scale_pipeline_کلید_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_کلید, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_کلید_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_کلید, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_کلید_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_جانچ_pipeline_کلید, webgpu_pipeline, ggml_webgpu_ssm_جانچ_pipeline_کلید_hash>
        ssm_جانچ_pipelines;   // type/d_state
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

    std::unordered_map<ggml_webgpu_سیٹ_صفاں_pipeline_کلید, webgpu_pipeline, ggml_webgpu_سیٹ_صفاں_pipeline_کلید_hash>
        سیٹ_صفاں_pipelines;
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

  عام:
    ggml_webgpu_shader_lib(wgpu::Device device) { هذا->device = device; }

    webgpu_pipeline get_جوڑ_صفاں_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        خودکار it = جوڑ_صفاں_pipelines.find(1);
        إذا (it != جوڑ_صفاں_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> مقرر_کردا;
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed        = preprocessor.preprocess(wgsl_جوڑ_صفاں, مقرر_کردا);
        جوڑ_صفاں_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        أرجع جوڑ_صفاں_pipelines[1];
    }

    webgpu_pipeline get_صف_norm_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_صف_norm_pipeline_کلید کلید = {};
        کلید.op                                = context.dst->op;
        کلید.src_type                          = context.src0->type;
        کلید.dst_type                          = context.dst->type;
        کلید.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = صف_norm_pipelines.find(کلید);
        إذا (it != صف_norm_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> مقرر_کردا;
        std::string              variant;

        بدل (کلید.op) {
            حالة GGML_OP_RMS_NORM:
                مقرر_کردا.push_back("RMS_NORM");
                variant = "rms_norm";
                اكسر;
            حالة GGML_OP_NORM:
                مقرر_کردا.push_back("NORM");
                variant = "norm";
                اكسر;
            حالة GGML_OP_L2_NORM:
                مقرر_کردا.push_back("L2_NORM");
                variant = "l2_norm";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        إذا (کلید.inplace) {
            مقرر_کردا.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (کلید.src_type == GGML_TYPE_F32) {
            مقرر_کردا.push_back("SRC_F32");
            variant += "_src_f32";
        } وإلا إذا (کلید.src_type == GGML_TYPE_F16) {
            مقرر_کردا.push_back("SRC_F16");
            variant += "_src_f16";
        }

        إذا (کلید.dst_type == GGML_TYPE_F32) {
            مقرر_کردا.push_back("DST_F32");
            variant += "_dst_f32";
        } وإلا إذا (کلید.dst_type == GGML_TYPE_F16) {
            مقرر_کردا.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ثابت uint32_t صف_norm_wg_ناپ = 128u;
        uint32_t       wg_ناپ          = std::min(context.max_wg_ناپ, صف_norm_wg_ناپ);
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(wg_ناپ));
        خودکار processed                  = preprocessor.preprocess(wgsl_صف_norm, مقرر_کردا);
        خودکار decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ              = wg_ناپ;
        decisions->inplace              = کلید.inplace;
        صف_norm_pipelines[کلید]         = ggml_webgpu_create_pipeline(device, processed, variant);
        صف_norm_pipelines[کلید].context = decisions;
        أرجع صف_norm_pipelines[کلید];
    }

    webgpu_pipeline get_argmax_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي vec4 = context.src0->ne[0] % 4 == 0;

        خودکار it = argmax_pipelines.find(vec4);
        إذا (it != argmax_pipelines.end()) {
            أرجع it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> مقرر_کردا;
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));
        إذا (vec4) {
            مقرر_کردا.push_back("VEC4");
            variant += "_vec4";
        }

        خودکار processed         = preprocessor.preprocess(wgsl_argmax, مقرر_کردا);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        أرجع argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_سیٹ_صفاں_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_سیٹ_صفاں_pipeline_کلید کلید       = {};
        کلید.src0_type                               = context.src0->type;
        کلید.dst_type                                = context.dst->type;
        کلید.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        کلید.i64_idx     = context.src1->type == GGML_TYPE_I64;
        کلید.pair_بلاکاں = quantized && ((context.src0->ne[0] / ggml_blck_ناپ(context.dst->type)) % 2 == 0);

        خودکار it = سیٹ_صفاں_pipelines.find(کلید);
        إذا (it != سیٹ_صفاں_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "set_rows";

        بدل (context.dst->type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("DST_F32");
                variant += "_dstf32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("DST_F16");
                variant += "_dstf16";
                اكسر;
            حالة GGML_TYPE_Q8_0:
                مقرر_کردا.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                اكسر;
            حالة GGML_TYPE_Q4_0:
                مقرر_کردا.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        إذا (context.src0->type == GGML_TYPE_F16) {
            مقرر_کردا.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        إذا (کلید.vec4) {
            مقرر_کردا.push_back("VEC4");
            variant += "_vec4";
        }
        إذا (کلید.i64_idx) {
            مقرر_کردا.push_back("I64_IDX");
            variant += "_i64idx";
        }
        إذا (کلید.pair_بلاکاں) {
            مقرر_کردا.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        ثابت خودکار & shader_سروت      = quantized ? wgsl_سیٹ_صفاں_quant : wgsl_سیٹ_صفاں;
        خودکار         processed          = preprocessor.preprocess(shader_سروت, مقرر_کردا);
        خودکار         decisions          = std::make_shared<ggml_webgpu_سیٹ_صفاں_shader_decisions>();
        decisions->vec4                 = کلید.vec4;
        decisions->i64_idx              = کلید.i64_idx;
        decisions->pair_بلاکاں          = کلید.pair_بلاکاں;
        decisions->wg_ناپ              = context.max_wg_ناپ;
        سیٹ_صفاں_pipelines[کلید]         = ggml_webgpu_create_pipeline(device, processed, variant);
        سیٹ_صفاں_pipelines[کلید].context = decisions;
        أرجع سیٹ_صفاں_pipelines[کلید];
    }

    webgpu_pipeline get_سیٹ_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_سیٹ_pipeline_کلید کلید = {};
        کلید.type                         = context.dst->type;
        کلید.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = سیٹ_pipelines.find(کلید);
        إذا (it != سیٹ_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "set";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_کردا.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for set shader");
        }

        إذا (کلید.inplace) {
            مقرر_کردا.push_back("INPLACE");
            variant += "_inplace";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_سیٹ, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        سیٹ_pipelines[کلید]       = pipeline;
        أرجع سیٹ_pipelines[کلید];
    }

    webgpu_pipeline get_cumsum_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        خودکار it = cumsum_pipelines.find(1);
        إذا (it != cumsum_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed      = preprocessor.preprocess(wgsl_cumsum, مقرر_کردا);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        أرجع cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_چوٹی_k = context.dst->op == GGML_OP_چوٹی_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t ترتیب =
            is_چوٹی_k ? (int32_t) GGML_ترتیب_دیو_ترتیب_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        خودکار it = argsort_pipelines.find(ترتیب);
        إذا (it != argsort_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "argsort";
        مقرر_کردا.push_back(std::string("ORDER=") + std::to_string(ترتیب));
        variant += std::string("_order") + std::to_string(ترتیب);
        uint32_t wg_ناپ = 1;
        طالما (wg_ناپ * 2 <= context.max_wg_ناپ &&
               wg_ناپ * GGML_WEBGPU_I32_ناپ_بائٹاں <= context.wg_mem_حد_بندی_بائٹاں / 2) {
            wg_ناپ *= 2;
        }
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(wg_ناپ));
        خودکار processed                   = preprocessor.preprocess(wgsl_argsort, مقرر_کردا);
        خودکار decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ               = wg_ناپ;
        argsort_pipelines[ترتیب]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[ترتیب].context = decisions;
        أرجع argsort_pipelines[ترتیب];
    }

    webgpu_pipeline get_argsort_ملاؤ_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_چوٹی_k = context.dst->op == GGML_OP_چوٹی_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t ترتیب =
            is_چوٹی_k ? (int32_t) GGML_ترتیب_دیو_ترتیب_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        خودکار it = argsort_ملاؤ_pipelines.find(ترتیب);
        إذا (it != argsort_ملاؤ_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "argsort_merge";
        مقرر_کردا.push_back(std::string("ORDER=") + std::to_string(ترتیب));
        variant += std::string("_order") + std::to_string(ترتیب);
        uint32_t wg_ناپ = std::min(GGML_WEBGPU_ARGSORT_ملاؤ_MAX_WG_ناپ, context.max_wg_ناپ);
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(wg_ناپ));

        خودکار processed                 = preprocessor.preprocess(wgsl_argsort_ملاؤ, مقرر_کردا);
        argsort_ملاؤ_pipelines[ترتیب] = ggml_webgpu_create_pipeline(device, processed, variant);
        أرجع argsort_ملاؤ_pipelines[ترتیب];
    }

    webgpu_pipeline get_get_صفاں_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي vec4_aligned) {
        ثابت منطقي vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_صفاں_pipeline_کلید کلید = {};
        کلید.src_type                          = context.src0->type;
        کلید.vectorized                        = (صحيح) vectorized;

        خودکار it = get_صفاں_pipelines.find(کلید);
        إذا (it != get_صفاں_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "get_rows";

        ثابت بنية ggml_type_traits * type_traits = ggml_get_type_traits(کلید.src_type);
        ثابت حرف *                    type_str    = type_traits->type_name;

        بدل (کلید.src_type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("FLOAT_PARALLEL");
                إذا (کلید.vectorized) {
                    مقرر_کردا.push_back("F32_VEC");
                    مقرر_کردا.push_back("SRC_TYPE=vec4<f32>");
                    مقرر_کردا.push_back("DST_TYPE=vec4<f32>");
                    مقرر_کردا.push_back("BLOCK_SIZE=4u");
                } وإلا {
                    مقرر_کردا.push_back("F32");
                    مقرر_کردا.push_back("SRC_TYPE=f32");
                    مقرر_کردا.push_back("DST_TYPE=f32");
                    مقرر_کردا.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("FLOAT_PARALLEL");
                مقرر_کردا.push_back("F16");
                مقرر_کردا.push_back("SRC_TYPE=f16");
                مقرر_کردا.push_back("DST_TYPE=f32");
                مقرر_کردا.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_کردا.push_back("FLOAT_PARALLEL");
                مقرر_کردا.push_back("I32");
                مقرر_کردا.push_back("SRC_TYPE=i32");
                مقرر_کردا.push_back("DST_TYPE=i32");
                مقرر_کردا.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                اكسر;
            حالة GGML_TYPE_BF16:
                مقرر_کردا.push_back("BF16");
                مقرر_کردا.push_back("SRC_TYPE=u32");
                مقرر_کردا.push_back("DST_TYPE=f32");
                مقرر_کردا.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_اتلا = type_str;
                    std::transform(type_اتلا.begin(), type_اتلا.end(), type_اتلا.begin(), ::وڈا_حرف);

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
                                مقرر_کردا.push_back("SRC_TYPE=u32");
                                مقرر_کردا.push_back("U32_DEQUANT_HELPERS");
                                اكسر;
                            }
                        افتراضي:
                            {
                                مقرر_کردا.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    مقرر_کردا.push_back("BYTE_HELPERS");
                    مقرر_کردا.push_back(type_اتلا + "_T");
                    مقرر_کردا.push_back(type_اتلا);
                    مقرر_کردا.push_back(type_اتلا + "_SCALE_MIN");
                    مقرر_کردا.push_back(type_اتلا + "_TABLES");
                    مقرر_کردا.push_back(type_اتلا + "_GRID");
                    مقرر_کردا.push_back(type_اتلا + "_LUT");

                    variant += "_";
                    variant += type_str;

                    مقرر_کردا.push_back("DST_TYPE=f32");

                    إذا (کلید.src_type == GGML_TYPE_Q1_0) {
                        مقرر_کردا.push_back("BLOCK_SIZE=128u");
                    } وإلا إذا ((کلید.src_type >= GGML_TYPE_Q4_0 && کلید.src_type <= GGML_TYPE_Q8_1) ||
                               کلید.src_type == GGML_TYPE_IQ4_NL || کلید.src_type == GGML_TYPE_MXFP4) {
                        مقرر_کردا.push_back("BLOCK_SIZE=32u");
                    } وإلا إذا (کلید.src_type == GGML_TYPE_Q2_0 || کلید.src_type == GGML_TYPE_NVFP4) {
                        مقرر_کردا.push_back("BLOCK_SIZE=64u");
                    } وإلا إذا (کلید.src_type >= GGML_TYPE_Q2_K) {
                        مقرر_کردا.push_back("BLOCK_SIZE=256u");
                    } وإلا {
                        مقرر_کردا.push_back("BLOCK_SIZE=1u");
                    }
                    اكسر;
                }
        }

        إذا (کلید.vectorized) {
            variant += "_vec";
        }

        مقرر_کردا.push_back("WG_SIZE=" + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_get_صفاں, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_get_صفاں_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_صفاں_pipelines[کلید]  = pipeline;
        أرجع get_صفاں_pipelines[کلید];
    }

    webgpu_pipeline get_scale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_کلید کلید = {};
        کلید.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = scale_pipelines.find(کلید);
        إذا (it != scale_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "scale";

        إذا (کلید.inplace) {
            مقرر_کردا.push_back("INPLACE");
            variant += "_inplace";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_scale, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
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

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "solve_tri";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ثابت uint32_t wg_ناپ       = std::min((uint32_t) کلید.n, context.max_wg_ناپ);
        ثابت uint32_t k_tile        = wg_ناپ;
        ثابت uint32_t بائٹاں_فی_صف = ((uint32_t) کلید.n + wg_ناپ) * GGML_WEBGPU_F32_ناپ_بائٹاں;
        ثابت uint32_t batch_n       = (uint32_t) (context.wg_mem_حد_بندی_بائٹاں / بائٹاں_فی_صف);

        مقرر_کردا.push_back(std::string("N=") + std::to_string(کلید.n));
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(wg_ناپ));
        مقرر_کردا.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        مقرر_کردا.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        خودکار processed           = preprocessor.preprocess(wgsl_solve_tri, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = wg_ناپ;
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

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "ssm_conv";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        إذا (کلید.vectorized) {
            مقرر_کردا.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t بلاک_ناپ_3    = 32u;
        constexpr uint32_t ٹوکن_فی_wg = 8u;

        مقرر_کردا.push_back("BLOCK_SIZE=" + std::to_string(بلاک_ناپ_3) + "u");
        مقرر_کردا.push_back("TOKENS_PER_WG=" + std::to_string(ٹوکن_فی_wg) + "u");

        خودکار processed           = preprocessor.preprocess(wgsl_ssm_conv, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->بلاک_ناپ_3    = بلاک_ناپ_3;
        decisions->ٹوکن_فی_wg = ٹوکن_فی_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[کلید]  = pipeline;
        أرجع ssm_conv_pipelines[کلید];
    }

    webgpu_pipeline get_ssm_جانچ_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_جانچ_pipeline_کلید کلید = {};
        کلید.type                              = context.dst->type;
        کلید.d_state                           = (صحيح) context.src0->ne[0];
        کلید.xbc_overlap                       = xbc_overlap;

        خودکار it = ssm_جانچ_pipelines.find(کلید);
        إذا (it != ssm_جانچ_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "ssm_scan";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ثابت uint32_t wg_ناپ = (uint32_t) کلید.d_state;

        constexpr uint32_t ٹوکن_فی_tile = 4u;

        مقرر_کردا.push_back("WG_SIZE=" + std::to_string(wg_ناپ) + "u");
        مقرر_کردا.push_back("TOKENS_PER_TILE=" + std::to_string(ٹوکن_فی_tile) + "u");

        إذا (context.supports_subgroups) {
            مقرر_کردا.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } وإلا {
            variant += "_wg_reduce";
        }

        إذا (کلید.xbc_overlap == 0b110) {  // x/B
            مقرر_کردا.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } وإلا إذا (کلید.xbc_overlap == 0b011) {  // B/C
            مقرر_کردا.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } وإلا إذا (کلید.xbc_overlap == 0b111) {  // x/B/C
            مقرر_کردا.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(کلید.d_state);

        خودکار processed             = preprocessor.preprocess(wgsl_ssm_جانچ, مقرر_کردا);
        خودکار decisions             = std::make_shared<ggml_webgpu_ssm_جانچ_shader_decisions>();
        decisions->wg_ناپ         = wg_ناپ;
        decisions->ٹوکن_فی_tile = ٹوکن_فی_tile;
        decisions->xbc_overlap     = کلید.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_جانچ_pipelines[کلید]    = pipeline;
        أرجع ssm_جانچ_pipelines[کلید];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_کلید کلید = {};
        کلید.type                                     = context.dst->type;
        کلید.s_v                                      = (صحيح) context.src2->ne[0];
        کلید.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        کلید.fused_cache                              = context.dst_fuse != خالی_سوچک;

        خودکار it = gated_delta_net_pipelines.find(کلید);
        إذا (it != gated_delta_net_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "gated_delta_net";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        إذا (کلید.kda) {
            مقرر_کردا.push_back("KDA");
            variant += "_kda";
        }

        إذا (کلید.fused_cache) {
            مقرر_کردا.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        مقرر_کردا.push_back("S_V=" + std::to_string(کلید.s_v) + "u");
        مقرر_کردا.push_back("WG_SIZE=" + std::to_string(کلید.s_v) + "u");

        خودکار            processed      = preprocessor.preprocess(wgsl_gated_delta_net, مقرر_کردا);
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

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "pad";

        إذا (کلید.circular) {
            مقرر_کردا.push_back("CIRCULAR");
            variant += "_circular";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_pad, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
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
        std::vector<std::string> مقرر_کردا;
        std::string              variant = "quantize_q8";

        uint32_t wg_ناپ = WEBGPU_MUL_MAT_VEC_WG_ناپ;

        مقرر_کردا.push_back("SRC1_INNER_TYPE=f32");
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(wg_ناپ));

        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_اتلا  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_اتلا.begin(), type_اتلا.end(), type_اتلا.begin(), ::وڈا_حرف);

        مقرر_کردا.push_back("MUL_ACC_" + type_اتلا);
        مقرر_کردا.push_back("Q8_1_T");

        مقرر_کردا.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        خودکار processed             = preprocessor.preprocess(shader_src, مقرر_کردا);
        خودکار decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ         = wg_ناپ;
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
        کلید.عدد_خانے_گنتی   = context.dst->ne[1];
        کلید.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_ضرب_حاصل, context.vendor);
        کلید.src_overlap = src_overlap;

        خودکار it = mul_mat_vec_pipelines.find(کلید);
        إذا (it != mul_mat_vec_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant    = "mul_mat_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=f32");
                مقرر_کردا.push_back("MUL_ACC_FLOAT");
                مقرر_کردا.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=f16");
                مقرر_کردا.push_back("MUL_ACC_FLOAT");
                مقرر_کردا.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=u32");
                مقرر_کردا.push_back("MUL_ACC_FLOAT");
                مقرر_کردا.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_اتلا  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_اتلا.begin(), type_اتلا.end(), type_اتلا.begin(), ::وڈا_حرف);

                    مقرر_کردا.push_back("BYTE_HELPERS");
                    مقرر_کردا.push_back("MUL_ACC_" + type_اتلا);
                    مقرر_کردا.push_back("U32_DEQUANT_HELPERS");
                    مقرر_کردا.push_back("SRC0_INNER_TYPE=u32");
                    بدل (context.src0->type) {
                        حالة GGML_TYPE_Q4_0:
                        حالة GGML_TYPE_Q4_1:
                        حالة GGML_TYPE_Q5_0:
                        حالة GGML_TYPE_Q5_1:
                        حالة GGML_TYPE_Q8_0:
                            إذا (کلید.use_mmvq) {
                                مقرر_کردا.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_Q1_0:
                        حالة GGML_TYPE_Q2_K:
                        حالة GGML_TYPE_Q3_K:
                        حالة GGML_TYPE_Q4_K:
                        حالة GGML_TYPE_Q5_K:
                        حالة GGML_TYPE_Q6_K:
                            إذا (کلید.use_mmvq) {
                                مقرر_کردا.push_back("K_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            مقرر_کردا.push_back(type_اتلا + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ3_XXS:
                            مقرر_کردا.push_back(type_اتلا + "_GRID");
                            مقرر_کردا.push_back(type_اتلا + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                            مقرر_کردا.push_back(type_اتلا + "_LUT");
                            إذا (کلید.use_mmvq) {
                                مقرر_کردا.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_NVFP4:
                            مقرر_کردا.push_back(type_اتلا + "_LUT");
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
                مقرر_کردا.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        مقرر_کردا.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_ناپ        = WEBGPU_MUL_MAT_VEC_WG_ناپ;
        uint32_t outputs_فی_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_فی_WG;

        إذا (کلید.src0_type == GGML_TYPE_Q1_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        } وإلا إذا (کلید.src0_type >= GGML_TYPE_Q2_K) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_فی_WG;
        } وإلا إذا (کلید.src0_type >= GGML_TYPE_Q4_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        }

        إذا (کلید.use_mmvq) {
            مقرر_کردا.push_back("MMVQ");
            مقرر_کردا.push_back("Q8_1_T");
        }

        إذا (کلید.src_overlap) {
            مقرر_کردا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(wg_ناپ));
        مقرر_کردا.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_فی_wg));
        مقرر_کردا.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (کلید.vectorized) {
            variant += "_vectorized";
        }
        مقرر_کردا.push_back(std::string("NUM_COLS=") + std::to_string(کلید.عدد_خانے_گنتی));

        خودکار processed            = preprocessor.preprocess(shader_src, مقرر_کردا);
        خودکار decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_ناپ        = wg_ناپ;
        decisions->outputs_فی_wg = outputs_فی_wg;
        decisions->vec_ناپ       = کلید.vectorized ? 4 : 1;

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
        کلید.use_subgroup_میٹرکس = context.supports_subgroup_میٹرکس;
        کلید.src_overlap         = src_overlap;

        خودکار it = mul_mat_fast_pipelines.find(کلید);
        إذا (it != mul_mat_fast_pipelines.end()) {
            أرجع it->second;
        }

        ثابت حرف * shader_src = کلید.use_subgroup_میٹرکس ? wgsl_mul_mat_subgroup_میٹرکس : wgsl_mul_mat_reg_tile;
        std::vector<std::string> مقرر_کردا;
        std::string              variant = کلید.use_subgroup_میٹرکس ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=f32");
                مقرر_کردا.push_back("TYPE_F32");
                مقرر_کردا.push_back("MUL_ACC_FLOAT");
                مقرر_کردا.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_کردا.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=f16");
                مقرر_کردا.push_back("TYPE_F16");
                مقرر_کردا.push_back("MUL_ACC_FLOAT");
                مقرر_کردا.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_کردا.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=u32");
                مقرر_کردا.push_back("TYPE_BF16");
                مقرر_کردا.push_back("MUL_ACC_FLOAT");
                مقرر_کردا.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_کردا.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_اتلا = src0_name;
                    std::transform(type_اتلا.begin(), type_اتلا.end(), type_اتلا.begin(), ::وڈا_حرف);

                    مقرر_کردا.push_back("BYTE_HELPERS");
                    مقرر_کردا.push_back("MUL_ACC_" + type_اتلا);
                    مقرر_کردا.push_back("INIT_SRC0_SHMEM_" + type_اتلا);
                    مقرر_کردا.push_back("INIT_SRC1_SHMEM_FLOAT");
                    مقرر_کردا.push_back("U32_DEQUANT_HELPERS");
                    مقرر_کردا.push_back("SRC0_INNER_TYPE=u32");

                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            مقرر_کردا.push_back(type_اتلا + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                            مقرر_کردا.push_back(type_اتلا + "_GRID");
                            مقرر_کردا.push_back(type_اتلا + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            مقرر_کردا.push_back(type_اتلا + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }

                    variant += std::string("_") + src0_name;
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        مقرر_کردا.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        ثابت منطقي is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        إذا (کلید.use_subgroup_میٹرکس) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } وإلا {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        مقرر_کردا.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        مقرر_کردا.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        إذا (کلید.use_subgroup_میٹرکس) {
            مقرر_کردا.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            مقرر_کردا.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_ناپ) + "u");
            مقرر_کردا.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            مقرر_کردا.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            مقرر_کردا.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_میٹرکس_M) + "u");
            مقرر_کردا.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_میٹرکس_N) + "u");
            مقرر_کردا.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            مقرر_کردا.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            مقرر_کردا.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        إذا (کلید.vectorized) {
            variant += "_vectorized";
        }

        إذا (کلید.src_overlap) {
            مقرر_کردا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        إذا (!کلید.use_subgroup_میٹرکس) {
            مقرر_کردا.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_ناپ_M) + "u");
            مقرر_کردا.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_ناپ_N) + "u");
            مقرر_کردا.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        خودکار processed = preprocessor.preprocess(shader_src, مقرر_کردا);

        خودکار decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_میٹرکس = کلید.use_subgroup_میٹرکس;
        إذا (کلید.use_subgroup_میٹرکس) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_میٹرکس_m = WEBGPU_MUL_MAT_SUBGROUP_میٹرکس_M;
            decisions->subgroup_میٹرکس_n = WEBGPU_MUL_MAT_SUBGROUP_میٹرکس_N;
            decisions->wg_ناپ           = context.max_subgroup_ناپ;
        } وإلا {
            decisions->wg_ناپ_m       = WEBGPU_MUL_MAT_WG_ناپ_M;
            decisions->wg_ناپ_n       = WEBGPU_MUL_MAT_WG_ناپ_N;
            decisions->wg_ناپ         = WEBGPU_MUL_MAT_WG_ناپ_M * WEBGPU_MUL_MAT_WG_ناپ_N;
            decisions->mul_mat_wg_ناپ = WEBGPU_MUL_MAT_WG_ناپ;
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
        std::vector<std::string> مقرر_کردا;
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ) + "u");
        مقرر_کردا.push_back(std::string("N_EXPERT=") + std::to_string(کلید.n_experts) + "u");
        مقرر_کردا.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_ناپ_N) + "u");

        خودکار processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, مقرر_کردا);
        خودکار decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ = context.max_wg_ناپ;

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

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "mul_mat_id";
        مقرر_کردا.push_back("MUL_MAT_ID");

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=f32");
                مقرر_کردا.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_کردا.push_back("INIT_SRC1_SHMEM_FLOAT");
                مقرر_کردا.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=f16");
                مقرر_کردا.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_کردا.push_back("INIT_SRC1_SHMEM_FLOAT");
                مقرر_کردا.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=u32");
                مقرر_کردا.push_back("INIT_SRC0_SHMEM_FLOAT");
                مقرر_کردا.push_back("INIT_SRC1_SHMEM_FLOAT");
                مقرر_کردا.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_اتلا = src0_name;
                    std::transform(type_اتلا.begin(), type_اتلا.end(), type_اتلا.begin(), ::وڈا_حرف);

                    مقرر_کردا.push_back("BYTE_HELPERS");
                    مقرر_کردا.push_back("INIT_SRC0_SHMEM_" + type_اتلا);
                    مقرر_کردا.push_back("INIT_SRC1_SHMEM_FLOAT");
                    مقرر_کردا.push_back("U32_DEQUANT_HELPERS");
                    مقرر_کردا.push_back("SRC0_INNER_TYPE=u32");

                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            مقرر_کردا.push_back(type_اتلا + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                            مقرر_کردا.push_back(type_اتلا + "_GRID");
                            مقرر_کردا.push_back(type_اتلا + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            مقرر_کردا.push_back(type_اتلا + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }

                    variant += std::string("_") + src0_name;
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        مقرر_کردا.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ثابت uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        مقرر_کردا.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        مقرر_کردا.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        مقرر_کردا.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        مقرر_کردا.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_ناپ_M) + "u");
        مقرر_کردا.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_ناپ_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        إذا (کلید.vectorized) {
            variant += "_vectorized";
        }

        خودکار processed = preprocessor.preprocess(wgsl_mul_mat_id, مقرر_کردا);

        خودکار decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_ناپ_m = WEBGPU_MUL_MAT_WG_ناپ_M;
        decisions->wg_ناپ_n = WEBGPU_MUL_MAT_WG_ناپ_N;
        decisions->wg_ناپ   = WEBGPU_MUL_MAT_WG_ناپ_M * WEBGPU_MUL_MAT_WG_ناپ_N;

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

        std::vector<std::string> مقرر_کردا;
        std::string              variant    = "mul_mat_id_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=f32");
                مقرر_کردا.push_back("MUL_ACC_FLOAT");
                مقرر_کردا.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=f16");
                مقرر_کردا.push_back("MUL_ACC_FLOAT");
                مقرر_کردا.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                مقرر_کردا.push_back("SRC0_INNER_TYPE=u32");
                مقرر_کردا.push_back("MUL_ACC_FLOAT");
                مقرر_کردا.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_اتلا  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_اتلا.begin(), type_اتلا.end(), type_اتلا.begin(), ::وڈا_حرف);

                    مقرر_کردا.push_back("BYTE_HELPERS");
                    مقرر_کردا.push_back("MUL_ACC_" + type_اتلا);
                    مقرر_کردا.push_back("U32_DEQUANT_HELPERS");
                    مقرر_کردا.push_back("SRC0_INNER_TYPE=u32");
                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            مقرر_کردا.push_back(type_اتلا + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ3_XXS:
                            مقرر_کردا.push_back(type_اتلا + "_GRID");
                            مقرر_کردا.push_back(type_اتلا + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            مقرر_کردا.push_back(type_اتلا + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        مقرر_کردا.push_back(کلید.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_ناپ        = WEBGPU_MUL_MAT_VEC_WG_ناپ;
        uint32_t outputs_فی_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_فی_WG;

        إذا (کلید.src0_type == GGML_TYPE_Q1_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        } وإلا إذا (کلید.src0_type >= GGML_TYPE_Q2_K) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_فی_WG;
        } وإلا إذا (کلید.src0_type >= GGML_TYPE_Q4_0) {
            outputs_فی_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_فی_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(wg_ناپ));
        مقرر_کردا.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_فی_wg));
        مقرر_کردا.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (کلید.vectorized) {
            variant += "_vectorized";
        }
        مقرر_کردا.push_back(std::string("NUM_COLS=1"));

        مقرر_کردا.push_back(std::string("N_EXPERTS=") + std::to_string(کلید.n_experts));

        خودکار processed = preprocessor.preprocess(shader_src, مقرر_کردا);

        خودکار decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_ناپ        = wg_ناپ;
        decisions->outputs_فی_wg = outputs_فی_wg;

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

        std::vector<std::string> مقرر_کردا;
        std::string              variant =
            کلید.is_unary ? ggml_unary_op_name((ggml_unary_op) کلید.op) : ggml_op_name((ggml_op) کلید.op);
        مقرر_کردا.push_back(variant);

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for unary shader");
        }

        إذا (کلید.inplace) {
            مقرر_کردا.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (op == GGML_OP_TRI) {
            بدل (کلید.ttype) {
                حالة GGML_TRI_TYPE_ہیٹھلا:
                    مقرر_کردا.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    اكسر;
                حالة GGML_TRI_TYPE_ہیٹھلا_DIAG:
                    مقرر_کردا.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    اكسر;
                حالة GGML_TRI_TYPE_اتلا:
                    مقرر_کردا.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    اكسر;
                حالة GGML_TRI_TYPE_اتلا_DIAG:
                    مقرر_کردا.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    اكسر;
                افتراضي:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_unary, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
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

        std::vector<std::string> مقرر_کردا;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        إذا (کلید.inplace) {
            مقرر_کردا.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (کلید.overlap) {
            مقرر_کردا.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (کلید.src_overlap) {
            مقرر_کردا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, مقرر_کردا);
        خودکار pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_ناپ     = context.max_wg_ناپ;
        pipeline_decisions->inplace     = کلید.inplace;
        pipeline_decisions->overlap     = کلید.overlap;
        pipeline_decisions->src_overlap = کلید.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[کلید]     = pipeline;
        أرجع rms_norm_mul_pipelines[کلید];
    }

    webgpu_pipeline get_ثنائی_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_ثنائی_pipeline_کلید کلید = {};
        کلید.type                            = context.dst->type;
        کلید.op                              = context.dst->op;
        کلید.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        کلید.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        کلید.src_overlap                     = src_overlap;

        خودکار it = ثنائی_pipelines.find(کلید);
        إذا (it != ثنائی_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              op_name = ggml_op_name((ggml_op) کلید.op);
        std::string              variant = op_name;

        مقرر_کردا.push_back(std::string("OP_") + op_name);

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for binary shader");
        }

        إذا (کلید.inplace) {
            مقرر_کردا.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (کلید.overlap) {
            مقرر_کردا.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (کلید.src_overlap) {
            مقرر_کردا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed                  = preprocessor.preprocess(wgsl_ثنائی, مقرر_کردا);
        خودکار pipeline_decisions         = std::make_shared<ggml_webgpu_ثنائی_shader_decisions>();
        pipeline_decisions->wg_ناپ     = context.max_wg_ناپ;
        pipeline_decisions->inplace     = کلید.inplace;
        pipeline_decisions->overlap     = کلید.overlap;
        pipeline_decisions->src_overlap = کلید.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ثنائی_pipelines[کلید]    = pipeline;
        أرجع ثنائی_pipelines[کلید];
    }

    webgpu_pipeline get_جوڑو_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_جوڑو_id_pipeline_کلید کلید = {};
        کلید.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = جوڑو_id_pipelines.find(کلید);
        إذا (it != جوڑو_id_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant    = "add_id";
        ثابت حرف *             shader_src = wgsl_جوڑو_id;

        إذا (کلید.inplace) {
            مقرر_کردا.push_back("INPLACE");
            variant += "_inplace";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed              = preprocessor.preprocess(shader_src, مقرر_کردا);
        خودکار pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_ناپ = context.max_wg_ناپ;
        pipeline_decisions->inplace = کلید.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        جوڑو_id_pipelines[کلید]    = pipeline;
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

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "concat";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_کردا.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for concat shader");
        }

        إذا (کلید.src_overlap) {
            مقرر_کردا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_concat, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_ثنائی_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
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

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "repeat";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_کردا.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            حالة GGML_TYPE_I16:
                مقرر_کردا.push_back("TYPE_I16");
                variant += "_i16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_repeat, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[کلید]    = pipeline;
        أرجع repeat_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ثابت منطقي can_use_subgroup_میٹرکس = ggml_webgpu_flash_attn_can_use_subgroup_میٹرکس_رستہ(
            context.supports_subgroup_میٹرکس, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_میٹرکس                    = can_use_subgroup_میٹرکس;
        decisions.q_tile = decisions.use_sg_میٹرکس ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_کلید کلید = {};
        کلید.common                              = ggml_webgpu_flash_attn_make_common_pipeline_کلید(
            context, decisions.use_sg_میٹرکس ? context.sg_mat_k : 1u, kv_overlap);
        کلید.common.k_سدھا &= decisions.use_sg_میٹرکس && کلید.common.k_type == GGML_TYPE_F16;
        کلید.common.v_سدھا &= decisions.use_sg_میٹرکس && کلید.common.v_type == GGML_TYPE_F16;
        کلید.use_sg_میٹرکس = decisions.use_sg_میٹرکس;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_حد_بندی_بائٹاں, decisions.q_tile, decisions.use_sg_میٹرکس ? context.sg_mat_n : 1u,
            کلید.common.سر_dim_qk, کلید.common.سر_dim_v, کلید.common.has_نقاب,
            کلید.common.k_سدھا || کلید.common.v_سدھا);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        إذا (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_میٹرکس);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_میٹرکس = خطأ;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            کلید.common.k_سدھا     = خطأ;
            کلید.common.v_سدھا     = خطأ;
            کلید.use_sg_میٹرکس       = خطأ;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_حد_بندی_بائٹاں, decisions.q_tile, 1u, کلید.common.سر_dim_qk, کلید.common.سر_dim_v,
                کلید.common.has_نقاب, کلید.common.k_سدھا || کلید.common.v_سدھا);
            GGML_ASSERT(max_kv_tile > 0);
        }

        decisions.kv_tile = decisions.use_sg_میٹرکس ?
                                std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES) :
                                std::min(GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE, max_kv_tile);
        decisions.wg_ناپ =
            decisions.use_sg_میٹرکس ?
                std::max(context.max_subgroup_ناپ, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_ناپ) :
                std::min(context.max_wg_ناپ, std::max(GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_ناپ,
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * context.max_subgroup_ناپ));

        إذا (کلید.common.k_سدھا || کلید.common.v_سدھا) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            طالما (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_میٹرکس ? context.sg_mat_n : context.min_subgroup_ناپ;
            }
        }

        خودکار it = flash_attn_pipelines.find(کلید);
        إذا (it != flash_attn_pipelines.end()) {
            أرجع it->second;
        }

        std::string              variant = decisions.use_sg_میٹرکس ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> مقرر_کردا = ggml_webgpu_flash_attn_common_مقرر_کردا(کلید.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_ناپ);
        ثابت حرف *             shader_src = خالی_سوچک;
        إذا (!کلید.use_sg_میٹرکس) {
            shader_src = wgsl_flash_attn_tile;
            مقرر_کردا.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_ناپ) + "u");
            مقرر_کردا.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_ناپ) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_ناپ) + "_" +
                       std::to_string(context.max_subgroup_ناپ);
        } وإلا {
            shader_src = wgsl_flash_attn;
            مقرر_کردا.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            مقرر_کردا.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            مقرر_کردا.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        خودکار            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, مقرر_کردا), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[کلید] = pipeline;
        أرجع flash_attn_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_کلید کلید = {};
        کلید.common = ggml_webgpu_flash_attn_make_common_pipeline_کلید(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی,
                                                                     kv_overlap);

        خودکار it = flash_attn_vec_pipelines.find(کلید);
        إذا (it != flash_attn_vec_pipelines.end()) {
            أرجع it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_حد_بندی_بائٹاں, کلید.common.سر_dim_qk,
                                                                   کلید.common.سر_dim_v, کلید.common.has_نقاب,
                                                                   کلید.common.k_سدھا || کلید.common.v_سدھا);
        decisions.wg_ناپ = context.max_subgroup_ناپ;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> مقرر_کردا =
            ggml_webgpu_flash_attn_common_مقرر_کردا(کلید.common, variant, 1u, decisions.kv_tile, decisions.wg_ناپ);
        إذا (کلید.common.has_نقاب) {
            مقرر_کردا.push_back("BLK");
            variant.resize(variant.size() - (حجم("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_ونڈو = context.min_subgroup_ناپ;
        ثابت uint32_t D       = کلید.common.سر_dim_qk | کلید.common.سر_dim_v;
        ثابت uint32_t D_lsb   = D & (~(D - 1u));
        d_ونڈو                = std::min(std::min(context.min_subgroup_ناپ, 4u), std::max(D_lsb / 4u, 1u));

        مقرر_کردا.push_back(std::string("D_SPLIT=") + std::to_string(d_ونڈو));
        variant += "_dsplit" + std::to_string(d_ونڈو);

        خودکار            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_ونڈو, مقرر_کردا), variant);
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

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "flash_attn_vec_blk";

        مقرر_کردا.push_back(std::string("KV_TILE=") + std::to_string(کلید.kv_tile));
        variant += std::string("_kvt") + std::to_string(کلید.kv_tile);

        uint32_t wg_ناپ = 1;
        طالما ((wg_ناپ << 1) <= context.max_wg_ناپ) {
            wg_ناپ <<= 1;
        }
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(wg_ناپ));
        variant += std::string("_wg") + std::to_string(wg_ناپ);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, مقرر_کردا), variant);
        flash_attn_blk_pipelines[کلید] = pipeline;
        أرجع flash_attn_blk_pipelines[کلید];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_کلید کلید = {};
        کلید.سر_dim_v                                     = (uint32_t) context.src2->ne[0];
        کلید.dst_type                                       = context.dst->type;
        کلید.wg_ناپ                                        = context.max_wg_ناپ;
        خودکار it                                            = flash_attn_vec_reduce_pipelines.find(کلید);
        إذا (it != flash_attn_vec_reduce_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "flash_attn_vec_reduce";

        بدل (کلید.dst_type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("DST_F32");
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("DST_F16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(کلید.dst_type);

        مقرر_کردا.push_back(std::string("HEAD_DIM_V=") + std::to_string(کلید.سر_dim_v));
        variant += std::string("_hsv") + std::to_string(کلید.سر_dim_v);

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));
        variant += std::string("_wg") + std::to_string(context.max_wg_ناپ);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, مقرر_کردا), variant);
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

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "cpy";

        بدل (کلید.src_type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("SRC_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("SRC_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_کردا.push_back("SRC_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        بدل (کلید.dst_type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("DST_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("DST_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                مقرر_کردا.push_back("DST_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_cpy, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[کلید]       = pipeline;
        أرجع cpy_pipelines[کلید];
    }

    webgpu_pipeline get_glu_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_glu_pipeline_کلید کلید = {};
        کلید.glu_op                       = ggml_get_glu_op(context.dst);
        کلید.type                         = context.dst->type;
        کلید.ونڈو                        = (context.src1 != خالی_سوچک);
        کلید.src_overlap                  = src_overlap;

        خودکار it = glu_pipelines.find(کلید);
        إذا (it != glu_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "glu";

        بدل (کلید.glu_op) {
            حالة GGML_GLU_OP_REGLU:
                مقرر_کردا.push_back("OP_REGLU");
                variant += "_reglu";
                اكسر;
            حالة GGML_GLU_OP_GEGLU:
                مقرر_کردا.push_back("OP_GEGLU");
                variant += "_geglu";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU:
                مقرر_کردا.push_back("OP_SWIGLU");
                variant += "_swiglu";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU_OAI:
                مقرر_کردا.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                اكسر;
            حالة GGML_GLU_OP_GEGLU_ERF:
                مقرر_کردا.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                اكسر;
            حالة GGML_GLU_OP_GEGLU_تیز:
                مقرر_کردا.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU_CLAMP:
                مقرر_کردا.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported GLU op");
        }
        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        إذا (کلید.src_overlap) {
            مقرر_کردا.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } وإلا إذا (کلید.ونڈو) {
            variant += "_split";
        } وإلا {
            مقرر_کردا.push_back("NO_SPLIT");
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_glu, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
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
        کلید.has_ff                        = (context.src2 != خالی_سوچک);

        خودکار it = rope_pipelines.find(کلید);
        إذا (it != rope_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "rope";

        بدل (کلید.type) {
            حالة GGML_TYPE_F32:
                مقرر_کردا.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                مقرر_کردا.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        إذا (کلید.inplace) {
            مقرر_کردا.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (کلید.has_ff) {
            مقرر_کردا.push_back("FF_FUNC");
            variant += "_ff";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_rope, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[کلید]      = pipeline;
        أرجع rope_pipelines[کلید];
    }

    webgpu_pipeline get_soft_max_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_کلید کلید = {};
        کلید.نقاب_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        کلید.has_نقاب                          = (context.src1 != خالی_سوچک);
        کلید.has_sink                          = (context.src2 != خالی_سوچک);
        کلید.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        خودکار it = soft_max_pipelines.find(کلید);
        إذا (it != soft_max_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "soft_max";

        إذا (کلید.has_نقاب) {
            مقرر_کردا.push_back("HAS_MASK");
            بدل (کلید.نقاب_type) {
                حالة GGML_TYPE_F32:
                    مقرر_کردا.push_back("MASK_F32");
                    variant += "_mask_f32";
                    اكسر;
                حالة GGML_TYPE_F16:
                    مقرر_کردا.push_back("MASK_F16");
                    variant += "_mask_f16";
                    اكسر;
                افتراضي:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        إذا (کلید.has_sink) {
            مقرر_کردا.push_back("HAS_SINK");
            variant += "_sink";
        }

        إذا (کلید.inplace) {
            مقرر_کردا.push_back("INPLACE");
            variant += "_inplace";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_soft_max, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
        decisions->inplace       = کلید.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[کلید]  = pipeline;
        أرجع soft_max_pipelines[کلید];
    }

    webgpu_pipeline get_conv2d_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_کلید کلید = {};
        کلید.بھار_type                     = context.src0->type;
        کلید.داخلہ_type                      = context.src1->type;
        کلید.output_type                     = context.dst->type;

        خودکار it = conv2d_pipelines.find(کلید);
        إذا (it != conv2d_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "conv_2d";

        خودکار ڈالو_type_مقرر_کردا = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_اگیتر = prefix;
            إذا (type == GGML_TYPE_F32) {
                مقرر_کردا.push_back(s_اگیتر + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                مقرر_کردا.push_back(s_اگیتر + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ڈالو_type_مقرر_کردا("WEIGHT_TYPE", کلید.بھار_type);
        ڈالو_type_مقرر_کردا("INPUT_TYPE", کلید.داخلہ_type);
        ڈالو_type_مقرر_کردا("OUTPUT_TYPE", کلید.output_type);

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_conv2d, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[کلید]    = pipeline;
        أرجع conv2d_pipelines[کلید];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي whcn) {
        ggml_webgpu_conv2d_dw_pipeline_کلید کلید = {};
        کلید.بھار_type                        = context.src0->type;
        کلید.داخلہ_type                         = context.src1->type;
        کلید.output_type                        = context.dst->type;
        کلید.whcn                               = whcn;

        خودکار it = conv2d_dw_pipelines.find(کلید);
        إذا (it != conv2d_dw_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        خودکار ڈالو_type_مقرر_کردا = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_اگیتر = prefix;
            إذا (type == GGML_TYPE_F32) {
                مقرر_کردا.push_back(s_اگیتر + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                مقرر_کردا.push_back(s_اگیتر + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ڈالو_type_مقرر_کردا("WEIGHT_TYPE", کلید.بھار_type);
        ڈالو_type_مقرر_کردا("INPUT_TYPE", کلید.داخلہ_type);
        ڈالو_type_مقرر_کردا("OUTPUT_TYPE", کلید.output_type);

        إذا (whcn) {
            مقرر_کردا.push_back("WHCN");
        }
        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_conv2d_dw, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[کلید] = pipeline;
        أرجع conv2d_dw_pipelines[کلید];
    }

    webgpu_pipeline get_im2col_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_کلید کلید = {};
        کلید.داخلہ_type                      = context.src1->type;
        کلید.output_type                     = context.dst->type;

        خودکار it = im2col_pipelines.find(کلید);
        إذا (it != im2col_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "im2col";

        خودکار ڈالو_type_مقرر_کردا = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_اگیتر = prefix;
            إذا (type == GGML_TYPE_F32) {
                مقرر_کردا.push_back(s_اگیتر + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                مقرر_کردا.push_back(s_اگیتر + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ڈالو_type_مقرر_کردا("INPUT_TYPE", کلید.داخلہ_type);
        ڈالو_type_مقرر_کردا("OUTPUT_TYPE", کلید.output_type);

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_im2col, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[کلید]    = pipeline;
        أرجع im2col_pipelines[کلید];
    }

    webgpu_pipeline get_upscale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت uint32_t منوال_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ثابت uint32_t ادھار_منوال  = منوال_flags & 0xFFu;
        ثابت منطقي     antialias  = (منوال_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_کلید کلید = {};
        کلید.داخلہ_type                       = context.src0->type;
        کلید.output_type                      = context.dst->type;
        کلید.ادھار_منوال                        = ادھار_منوال;
        کلید.antialias                        = antialias;

        خودکار it = upscale_pipelines.find(کلید);
        إذا (it != upscale_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> مقرر_کردا;
        std::string              variant = "upscale";

        إذا (کلید.داخلہ_type == GGML_TYPE_F16) {
            مقرر_کردا.push_back("SRC_F16");
            variant += "_src_f16";
        } وإلا {
            variant += "_src_f32";
        }

        إذا (کلید.output_type == GGML_TYPE_F16) {
            مقرر_کردا.push_back("DST_F16");
            variant += "_dst_f16";
        } وإلا {
            variant += "_dst_f32";
        }

        بدل (ادھار_منوال) {
            حالة GGML_SCALE_منوال_سبھ_توں_نیڑے:
                مقرر_کردا.push_back("NEAREST");
                variant += "_nearest";
                اكسر;
            حالة GGML_SCALE_منوال_BILINEAR:
                مقرر_کردا.push_back("BILINEAR");
                variant += "_bilinear";
                اكسر;
            حالة GGML_SCALE_منوال_BICUBIC:
                مقرر_کردا.push_back("BICUBIC");
                variant += "_bicubic";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported upscale mode");
        }

        إذا (antialias) {
            مقرر_کردا.push_back("ANTIALIAS");
            variant += "_aa";
        }

        مقرر_کردا.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ناپ));

        خودکار processed           = preprocessor.preprocess(wgsl_upscale, مقرر_کردا);
        خودکار decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ناپ       = context.max_wg_ناپ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[کلید]   = pipeline;
        أرجع upscale_pipelines[کلید];
    }

  خاص:
    ساكن webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_کوڈ,
                                                       std::string    label) {
        wgpu::Shader_سروت_WGSL shader_سروت;
        shader_سروت.code = shader_کوڈ.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.اگلا_وچ_Chain = &shader_سروت;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.اندراج_نقطہ_2 = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = خالی_سوچک;  // nullptr means auto layout
        أرجع { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#جے_انت  // GGML_WEBGPU_SHADER_LIB_HPP
