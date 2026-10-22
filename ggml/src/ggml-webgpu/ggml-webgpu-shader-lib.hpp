#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

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

#define GGML_WEBGPU_F16_SIZE_BYTES                   2
#define GGML_WEBGPU_F32_SIZE_BYTES                   4
#define GGML_WEBGPU_I32_SIZE_BYTES                   4
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#define GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#define GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#define GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#define GGML_WEBGPU_KV_SEQ_PAD                       256u

#define GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE 512u

// Matrix multiplication parameters

// Register tiling parameters
#define WEBGPU_MUL_MAT_TILE_M           4
#define WEBGPU_MUL_MAT_TILE_N           4
#define WEBGPU_MUL_MAT_WG_SIZE_M        8
#define WEBGPU_MUL_MAT_WG_SIZE_N        8
#define WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#define WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#define WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#define WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M     4
#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N     2
#define WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#define WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#define WEBGPU_MUL_MAT_VEC_WG_SIZE 256

#define WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG    4
#define WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG 4
#define WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG      4

// default size for reg-tile matrix multiplication
#define WEBGPU_MUL_MAT_WG_SIZE 256

// Same hash combine function as in boost
template <typename T> مضمن فراغ ggml_webgpu_hash_combine(size_t & seed, ثابت T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
مضمن uintptr_t ggml_webgpu_tensor_addr(ثابت ggml_tensor * tensor) {
    ثابت ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    إرجاع (uintptr_t) base_tensor->data + tensor->view_offs;
}

مضمن منطقي ggml_webgpu_tensor_equal(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    إرجاع a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
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

    uint32_t    max_wg_size;
    size_t      wg_mem_limit_bytes       = 0;
    منطقي        supports_subgroups       = خطأ;
    منطقي        supports_subgroup_matrix = خطأ;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_size        = 0;
    uint32_t    max_subgroup_size        = 0;
    منطقي        supports_dot_product     = خطأ;
    std::string vendor;
};

بنية webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<فراغ> context = nullptr;
};

بنية ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 0;
    منطقي     inplace = خطأ;
};

بنية ggml_webgpu_get_rows_shader_decisions {
    uint32_t wg_size    = 0;
    منطقي     vectorized = خطأ;
};

بنية ggml_webgpu_ثنائي_shader_decisions {
    uint32_t wg_size     = 0;
    منطقي     inplace     = خطأ;
    منطقي     overlap     = خطأ;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_glu_shader_decisions {
    uint32_t wg_size     = 0;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<فراغ> decisions;
};

بنية ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_size;
    uint32_t tokens_per_wg;
};

بنية ggml_webgpu_ssm_scan_pipeline_مفتاح {
    صحيح     type;
    صحيح     d_state;
    uint8_t xbc_overlap;
    منطقي    a_overlap;
    منطقي    ids_overlap;

    منطقي operator==(ثابت ggml_webgpu_ssm_scan_pipeline_مفتاح & other) ثابت {
        إرجاع type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

بنية ggml_webgpu_ssm_scan_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_ssm_scan_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        ggml_webgpu_hash_combine(seed, مفتاح.d_state);
        ggml_webgpu_hash_combine(seed, مفتاح.xbc_overlap);
        ggml_webgpu_hash_combine(seed, مفتاح.a_overlap);
        ggml_webgpu_hash_combine(seed, مفتاح.ids_overlap);
        إرجاع seed;
    }
};

بنية ggml_webgpu_ssm_scan_shader_decisions {
    uint32_t wg_size;
    uint32_t tokens_per_tile;
    uint8_t  xbc_overlap = 0;
    منطقي     a_overlap   = خطأ;
    منطقي     ids_overlap = خطأ;
};

/** Argsort **/

بنية ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes;
    int32_t  order;
};

/** Set Rows **/

بنية ggml_webgpu_set_rows_pipeline_مفتاح {
    صحيح src0_type;
    صحيح dst_type;
    صحيح vec4;
    صحيح i64_idx;
    صحيح pair_blocks;

    منطقي operator==(ثابت ggml_webgpu_set_rows_pipeline_مفتاح & other) ثابت {
        إرجاع src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

بنية ggml_webgpu_set_rows_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_set_rows_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.src0_type);
        ggml_webgpu_hash_combine(seed, مفتاح.dst_type);
        ggml_webgpu_hash_combine(seed, مفتاح.vec4);
        ggml_webgpu_hash_combine(seed, مفتاح.i64_idx);
        ggml_webgpu_hash_combine(seed, مفتاح.pair_blocks);
        إرجاع seed;
    }
};

بنية ggml_webgpu_set_rows_shader_decisions {
    منطقي     vec4;
    منطقي     i64_idx;
    منطقي     pair_blocks;
    uint32_t wg_size;
};

/** Set **/

بنية ggml_webgpu_set_pipeline_مفتاح {
    ggml_type type;
    منطقي      inplace;

    منطقي operator==(ثابت ggml_webgpu_set_pipeline_مفتاح & other) ثابت {
        إرجاع type == other.type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_set_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_set_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        ggml_webgpu_hash_combine(seed, مفتاح.inplace);
        إرجاع seed;
    }
};

/** Get Rows **/

بنية ggml_webgpu_get_rows_pipeline_مفتاح {
    ggml_type src_type;
    صحيح       vectorized;

    منطقي operator==(ثابت ggml_webgpu_get_rows_pipeline_مفتاح & other) ثابت {
        إرجاع src_type == other.src_type && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_get_rows_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_get_rows_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.src_type);
        ggml_webgpu_hash_combine(seed, مفتاح.vectorized);
        إرجاع seed;
    }
};

/** Row Norm **/

بنية ggml_webgpu_row_norm_pipeline_مفتاح {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    منطقي      inplace;

    منطقي operator==(ثابت ggml_webgpu_row_norm_pipeline_مفتاح & other) ثابت {
        إرجاع op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_row_norm_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_row_norm_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.op);
        ggml_webgpu_hash_combine(seed, مفتاح.src_type);
        ggml_webgpu_hash_combine(seed, مفتاح.dst_type);
        ggml_webgpu_hash_combine(seed, مفتاح.inplace);
        إرجاع seed;
    }
};

/** RMS_NORM + MUL **/

بنية ggml_webgpu_rms_norm_mul_pipeline_مفتاح {
    منطقي inplace;      // rn_src == dst
    منطقي overlap;      // mul_src == dst
    منطقي src_overlap;  // rn_src binding overlaps mul_src binding

    منطقي operator==(ثابت ggml_webgpu_rms_norm_mul_pipeline_مفتاح & other) ثابت {
        إرجاع inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_rms_norm_mul_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_rms_norm_mul_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.inplace);
        ggml_webgpu_hash_combine(seed, مفتاح.overlap);
        ggml_webgpu_hash_combine(seed, مفتاح.src_overlap);
        إرجاع seed;
    }
};

بنية ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    منطقي     inplace     = خطأ;
    منطقي     overlap     = خطأ;
    منطقي     src_overlap = خطأ;
};

/** Pad **/
بنية ggml_webgpu_pad_pipeline_مفتاح {
    منطقي circular;

    منطقي operator==(ثابت ggml_webgpu_pad_pipeline_مفتاح & other) ثابت { إرجاع circular == other.circular; }
};

بنية ggml_webgpu_pad_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_pad_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.circular);
        إرجاع seed;
    }
};

/** Solve Tri **/
بنية ggml_webgpu_solve_tri_pipeline_مفتاح {
    صحيح type;
    صحيح n;
    صحيح k;

    منطقي operator==(ثابت ggml_webgpu_solve_tri_pipeline_مفتاح & other) ثابت {
        إرجاع type == other.type && n == other.n && k == other.k;
    }
};

بنية ggml_webgpu_solve_tri_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_solve_tri_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        ggml_webgpu_hash_combine(seed, مفتاح.n);
        ggml_webgpu_hash_combine(seed, مفتاح.k);
        إرجاع seed;
    }
};

/** SSM Conv **/
بنية ggml_webgpu_ssm_conv_pipeline_مفتاح {
    صحيح type;
    صحيح vectorized;

    منطقي operator==(ثابت ggml_webgpu_ssm_conv_pipeline_مفتاح & other) ثابت {
        إرجاع type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
بنية ggml_webgpu_conv2d_pipeline_مفتاح {
    ggml_type weight_type;
    ggml_type إدخال_type;
    ggml_type output_type;

    منطقي operator==(ثابت ggml_webgpu_conv2d_pipeline_مفتاح & other) ثابت {
        إرجاع weight_type == other.weight_type && إدخال_type == other.إدخال_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_conv2d_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_conv2d_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.weight_type);
        ggml_webgpu_hash_combine(seed, مفتاح.إدخال_type);
        ggml_webgpu_hash_combine(seed, مفتاح.output_type);
        إرجاع seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
بنية ggml_webgpu_conv2d_dw_pipeline_مفتاح {
    ggml_type weight_type;
    ggml_type إدخال_type;
    ggml_type output_type;
    منطقي      whcn;

    منطقي operator==(ثابت ggml_webgpu_conv2d_dw_pipeline_مفتاح & other) ثابت {
        إرجاع weight_type == other.weight_type && إدخال_type == other.إدخال_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

بنية ggml_webgpu_conv2d_dw_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_conv2d_dw_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.weight_type);
        ggml_webgpu_hash_combine(seed, مفتاح.إدخال_type);
        ggml_webgpu_hash_combine(seed, مفتاح.output_type);
        ggml_webgpu_hash_combine(seed, مفتاح.whcn);
        إرجاع seed;
    }
};

/** Im2Col **/
بنية ggml_webgpu_im2col_pipeline_مفتاح {
    ggml_type إدخال_type;
    ggml_type output_type;

    منطقي operator==(ثابت ggml_webgpu_im2col_pipeline_مفتاح & other) ثابت {
        إرجاع إدخال_type == other.إدخال_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_im2col_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_im2col_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.إدخال_type);
        ggml_webgpu_hash_combine(seed, مفتاح.output_type);
        إرجاع seed;
    }
};

/** Gated Delta Net **/
بنية ggml_webgpu_gated_delta_net_pipeline_مفتاح {
    صحيح  type;
    صحيح  s_v;
    صحيح  kda;
    منطقي fused_cache;

    منطقي operator==(ثابت ggml_webgpu_gated_delta_net_pipeline_مفتاح & other) ثابت {
        إرجاع type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

بنية ggml_webgpu_gated_delta_net_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_gated_delta_net_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        ggml_webgpu_hash_combine(seed, مفتاح.s_v);
        ggml_webgpu_hash_combine(seed, مفتاح.kda);
        إرجاع seed;
    }
};

بنية ggml_webgpu_ssm_conv_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_ssm_conv_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        ggml_webgpu_hash_combine(seed, مفتاح.vectorized);
        إرجاع seed;
    }
};

/** Scale **/

بنية ggml_webgpu_scale_pipeline_مفتاح {
    صحيح inplace;

    منطقي operator==(ثابت ggml_webgpu_scale_pipeline_مفتاح & other) ثابت { إرجاع inplace == other.inplace; }
};

بنية ggml_webgpu_scale_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_scale_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.inplace);
        إرجاع seed;
    }
};

/** Upscale **/

بنية ggml_webgpu_upscale_pipeline_مفتاح {
    ggml_type إدخال_type;
    ggml_type output_type;
    uint32_t  base_mode;
    منطقي      antialias;

    منطقي operator==(ثابت ggml_webgpu_upscale_pipeline_مفتاح & other) ثابت {
        إرجاع إدخال_type == other.إدخال_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

بنية ggml_webgpu_upscale_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_upscale_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.إدخال_type);
        ggml_webgpu_hash_combine(seed, مفتاح.output_type);
        ggml_webgpu_hash_combine(seed, مفتاح.base_mode);
        ggml_webgpu_hash_combine(seed, مفتاح.antialias);
        إرجاع seed;
    }
};

/** Concat **/

بنية ggml_webgpu_concat_pipeline_مفتاح {
    صحيح  type;
    منطقي src_overlap;

    منطقي operator==(ثابت ggml_webgpu_concat_pipeline_مفتاح & other) ثابت {
        إرجاع type == other.type && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_concat_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_concat_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        ggml_webgpu_hash_combine(seed, مفتاح.src_overlap);
        إرجاع seed;
    }
};

/** Repeat **/

بنية ggml_webgpu_repeat_pipeline_مفتاح {
    صحيح type;

    منطقي operator==(ثابت ggml_webgpu_repeat_pipeline_مفتاح & other) ثابت { إرجاع type == other.type; }
};

بنية ggml_webgpu_repeat_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_repeat_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        إرجاع seed;
    }
};

/** Binary **/

بنية ggml_webgpu_ثنائي_pipeline_مفتاح {
    صحيح  type;
    صحيح  op;
    منطقي inplace;
    منطقي overlap;
    منطقي src_overlap;

    منطقي operator==(ثابت ggml_webgpu_ثنائي_pipeline_مفتاح & other) ثابت {
        إرجاع type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_ثنائي_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_ثنائي_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        ggml_webgpu_hash_combine(seed, مفتاح.op);
        ggml_webgpu_hash_combine(seed, مفتاح.inplace);
        ggml_webgpu_hash_combine(seed, مفتاح.overlap);
        ggml_webgpu_hash_combine(seed, مفتاح.src_overlap);
        إرجاع seed;
    }
};

/* Add_Id */

بنية ggml_webgpu_add_id_pipeline_مفتاح {
    منطقي inplace;

    منطقي operator==(ثابت ggml_webgpu_add_id_pipeline_مفتاح & other) ثابت { إرجاع inplace == other.inplace; }
};

بنية ggml_webgpu_add_id_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_add_id_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.inplace);
        إرجاع seed;
    }
};

/** Unary **/

بنية ggml_webgpu_unary_pipeline_مفتاح {
    صحيح           type;
    صحيح           op;
    منطقي          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    منطقي          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    منطقي operator==(ثابت ggml_webgpu_unary_pipeline_مفتاح & other) ثابت {
        إرجاع type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

بنية ggml_webgpu_unary_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_unary_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        ggml_webgpu_hash_combine(seed, مفتاح.op);
        ggml_webgpu_hash_combine(seed, مفتاح.is_unary);
        ggml_webgpu_hash_combine(seed, مفتاح.inplace);
        ggml_webgpu_hash_combine(seed, مفتاح.ttype);
        إرجاع seed;
    }
};

/** FlashAttention */

بنية ggml_webgpu_flash_attn_common_pipeline_مفتاح {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    منطقي      k_direct;
    منطقي      v_direct;
    منطقي      kv_overlap;
    منطقي      has_mask;
    منطقي      has_sinks;
    منطقي      uses_logit_softcap;

    منطقي operator==(ثابت ggml_webgpu_flash_attn_common_pipeline_مفتاح & other) ثابت {
        إرجاع q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

مضمن فراغ ggml_webgpu_flash_attn_hash_common_pipeline_مفتاح(size_t &                                           seed,
                                                            ثابت ggml_webgpu_flash_attn_common_pipeline_مفتاح & مفتاح) {
    ggml_webgpu_hash_combine(seed, مفتاح.q_type);
    ggml_webgpu_hash_combine(seed, مفتاح.k_type);
    ggml_webgpu_hash_combine(seed, مفتاح.v_type);
    ggml_webgpu_hash_combine(seed, مفتاح.dst_type);
    ggml_webgpu_hash_combine(seed, مفتاح.head_dim_qk);
    ggml_webgpu_hash_combine(seed, مفتاح.head_dim_v);
    ggml_webgpu_hash_combine(seed, مفتاح.k_direct);
    ggml_webgpu_hash_combine(seed, مفتاح.v_direct);
    ggml_webgpu_hash_combine(seed, مفتاح.kv_overlap);
    ggml_webgpu_hash_combine(seed, مفتاح.has_mask);
    ggml_webgpu_hash_combine(seed, مفتاح.has_sinks);
    ggml_webgpu_hash_combine(seed, مفتاح.uses_logit_softcap);
}

بنية ggml_webgpu_flash_attn_vec_pipeline_مفتاح {
    ggml_webgpu_flash_attn_common_pipeline_مفتاح common;

    منطقي operator==(ثابت ggml_webgpu_flash_attn_vec_pipeline_مفتاح & other) ثابت { إرجاع common == other.common; }
};

بنية ggml_webgpu_flash_attn_vec_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_flash_attn_vec_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_مفتاح(seed, مفتاح.common);
        إرجاع seed;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_مفتاح {
    ggml_webgpu_flash_attn_common_pipeline_مفتاح common;
    منطقي                                       use_sg_matrix;

    منطقي operator==(ثابت ggml_webgpu_flash_attn_pipeline_مفتاح & other) ثابت {
        إرجاع common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_flash_attn_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_مفتاح(seed, مفتاح.common);
        ggml_webgpu_hash_combine(seed, مفتاح.use_sg_matrix);
        إرجاع seed;
    }
};

بنية ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

بنية ggml_webgpu_flash_attn_decisions {
    منطقي     use_sg_matrix = خطأ;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_size       = 0;
};

مضمن constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
مضمن constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

مضمن size_t ggml_webgpu_flash_attn_tensor_offset(ثابت ggml_tensor * tensor) {
    constexpr uintptr_t ptr_base_addr = 0x1000u;
    ثابت ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
    إرجاع reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K, size_t storage_offset_alignment) {
    ثابت uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_size(K->type));
    إرجاع offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K,
                                                      ثابت ggml_tensor * V,
                                                      size_t              storage_offset_alignment) {
    إرجاع ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

مضمن منطقي ggml_webgpu_flash_attn_k_direct(ثابت ggml_tensor * Q, ثابت ggml_tensor * K, uint32_t kv_direct_align) {
    إرجاع (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

مضمن منطقي ggml_webgpu_flash_attn_v_direct(ثابت ggml_tensor * Q, ثابت ggml_tensor * V, uint32_t kv_direct_align) {
    إرجاع ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

مضمن ggml_webgpu_flash_attn_common_pipeline_مفتاح ggml_webgpu_flash_attn_make_common_pipeline_مفتاح(
    ثابت ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    منطقي                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_مفتاح مفتاح = {};
    مفتاح.q_type                                     = context.src0->type;
    مفتاح.k_type                                     = context.src1->type;
    مفتاح.v_type                                     = context.src2->type;
    مفتاح.dst_type                                   = context.dst->type;
    مفتاح.head_dim_qk                                = (uint32_t) context.src0->ne[0];
    مفتاح.head_dim_v                                 = (uint32_t) context.src2->ne[0];
    مفتاح.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    مفتاح.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    مفتاح.kv_overlap         = kv_overlap;
    مفتاح.has_mask           = context.src3 != nullptr;
    مفتاح.has_sinks          = context.src4 != nullptr;
    مفتاح.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    إرجاع مفتاح;
}

مضمن std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    ثابت ggml_webgpu_flash_attn_common_pipeline_مفتاح & مفتاح,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_size) {
    std::vector<std::string> defines;

    بدل (مفتاح.k_type) {
        حالة GGML_TYPE_F32:
            defines.push_back("K_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            defines.push_back("K_F16");
            اكسر;
        حالة GGML_TYPE_Q4_0:
            defines.push_back("K_Q4_0");
            اكسر;
        حالة GGML_TYPE_Q8_0:
            defines.push_back("K_Q8_0");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(مفتاح.k_type);

    بدل (مفتاح.v_type) {
        حالة GGML_TYPE_F32:
            defines.push_back("V_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            defines.push_back("V_F16");
            اكسر;
        حالة GGML_TYPE_Q4_0:
            defines.push_back("V_Q4_0");
            اكسر;
        حالة GGML_TYPE_Q8_0:
            defines.push_back("V_Q8_0");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(مفتاح.v_type);

    بدل (مفتاح.q_type) {
        حالة GGML_TYPE_F32:
            defines.push_back("Q_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            defines.push_back("Q_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(مفتاح.q_type);

    بدل (مفتاح.dst_type) {
        حالة GGML_TYPE_F32:
            defines.push_back("DST_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            defines.push_back("DST_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(مفتاح.dst_type);

    إذا (مفتاح.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    إذا (مفتاح.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    إذا (مفتاح.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    إذا (مفتاح.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    إذا (مفتاح.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    إذا (مفتاح.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(مفتاح.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(مفتاح.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(مفتاح.head_dim_v));
    variant += std::string("_hsv") + std::to_string(مفتاح.head_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    إذا (ggml_is_quantized(مفتاح.k_type) || ggml_is_quantized(مفتاح.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        إذا (ggml_is_quantized(مفتاح.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        إذا (ggml_is_quantized(مفتاح.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    إرجاع defines;
}

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_مفتاح {
    uint32_t  head_dim_v;
    uint32_t  wg_size;
    ggml_type dst_type;
};

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.head_dim_v);
        ggml_webgpu_hash_combine(seed, مفتاح.wg_size);
        ggml_webgpu_hash_combine(seed, مفتاح.dst_type);
        إرجاع seed;
    }
};

مضمن منطقي operator==(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_مفتاح & lhs,
                       ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_مفتاح & rhs) {
    إرجاع lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size && lhs.dst_type == rhs.dst_type;
}

بنية ggml_webgpu_flash_attn_blk_pipeline_مفتاح {
    uint32_t kv_tile;

    منطقي operator==(ثابت ggml_webgpu_flash_attn_blk_pipeline_مفتاح & other) ثابت { إرجاع kv_tile == other.kv_tile; }
};

بنية ggml_webgpu_flash_attn_blk_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_flash_attn_blk_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.kv_tile);
        إرجاع seed;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
مضمن size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  منطقي     has_mask,
                                                  منطقي     kv_direct) {
    ثابت uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * head_dim_qk;        // q_shmem
    إذا (!kv_direct) {
        f32_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f32_elems += q_tile * head_dim_v;         // o_shmem
    إذا (has_mask) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    إرجاع f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

مضمن uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   limit_bytes,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t head_dim_qk,
                                                   uint32_t head_dim_v,
                                                   منطقي     has_mask,
                                                   منطقي     kv_direct) {
    ثابت size_t base_q_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 0, head_dim_qk, head_dim_v, has_mask, kv_direct);
    إذا (limit_bytes <= base_q_bytes) {
        إرجاع 0;
    }
    ثابت size_t one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, head_dim_qk, head_dim_v, has_mask, kv_direct);
    ثابت size_t bytes_per_kv = one_kv_bytes - base_q_bytes;
    إذا (bytes_per_kv == 0) {
        إرجاع 0;
    }
    ثابت size_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    إرجاع (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

مضمن uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_limit_bytes,
                                                       uint32_t head_dim_qk,
                                                       uint32_t head_dim_v,
                                                       منطقي     has_mask,
                                                       منطقي     kv_direct) {
    ثابت uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_limit_bytes, 1u, 1u, head_dim_qk, head_dim_v, has_mask, kv_direct);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    إذا (kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        بينما (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    إرجاع kv_tile;
}

مضمن منطقي ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(منطقي                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ثابت ggml_tensor * Q,
                                                                ثابت ggml_tensor * V) {
    إرجاع supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

بنية ggml_webgpu_mul_mat_vec_pipeline_مفتاح {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    uint32_t  رقم_cols;
    منطقي      use_mmvq;
    منطقي      src_overlap;

    منطقي operator==(ثابت ggml_webgpu_mul_mat_vec_pipeline_مفتاح & other) ثابت {
        إرجاع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               رقم_cols == other.رقم_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_vec_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_mul_mat_vec_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.src0_type);
        ggml_webgpu_hash_combine(seed, مفتاح.src1_type);
        ggml_webgpu_hash_combine(seed, مفتاح.vectorized);
        ggml_webgpu_hash_combine(seed, مفتاح.رقم_cols);
        ggml_webgpu_hash_combine(seed, مفتاح.use_mmvq);
        ggml_webgpu_hash_combine(seed, مفتاح.src_overlap);
        إرجاع seed;
    }
};

بنية ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_quantize_q8_pipeline_مفتاح {
    ggml_type src0_type;

    منطقي operator==(ثابت ggml_webgpu_quantize_q8_pipeline_مفتاح & other) ثابت { إرجاع src0_type == other.src0_type; }
};

بنية ggml_webgpu_quantize_q8_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_quantize_q8_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.src0_type);
        إرجاع seed;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_مفتاح {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    صحيح       use_subgroup_matrix;
    منطقي      src_overlap;

    منطقي operator==(ثابت ggml_webgpu_mul_mat_pipeline_مفتاح & other) ثابت {
        إرجاع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_mul_mat_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.src0_type);
        ggml_webgpu_hash_combine(seed, مفتاح.src1_type);
        ggml_webgpu_hash_combine(seed, مفتاح.vectorized);
        ggml_webgpu_hash_combine(seed, مفتاح.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, مفتاح.src_overlap);
        إرجاع seed;
    }
};

بنية ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    صحيح      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_size;
    منطقي     src_overlap = خطأ;
};

/** MUL_MAT_ID **/

بنية ggml_webgpu_mul_mat_id_gather_pipeline_مفتاح {
    uint32_t n_experts;

    منطقي operator==(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_مفتاح & other) ثابت {
        إرجاع n_experts == other.n_experts;
    }
};

بنية ggml_webgpu_mul_mat_id_gather_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.n_experts);
        إرجاع seed;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_مفتاح {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  رقم_cols;
    صحيح       vectorized;

    منطقي operator==(ثابت ggml_webgpu_mul_mat_id_pipeline_مفتاح & other) ثابت {
        إرجاع src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               رقم_cols == other.رقم_cols && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_mul_mat_id_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.src0_type);
        ggml_webgpu_hash_combine(seed, مفتاح.src1_type);
        ggml_webgpu_hash_combine(seed, مفتاح.n_experts);
        ggml_webgpu_hash_combine(seed, مفتاح.رقم_cols);
        ggml_webgpu_hash_combine(seed, مفتاح.vectorized);
        إرجاع seed;
    }
};

/** Cpy **/

بنية ggml_webgpu_cpy_pipeline_مفتاح {
    ggml_type src_type;
    ggml_type dst_type;

    منطقي operator==(ثابت ggml_webgpu_cpy_pipeline_مفتاح & other) ثابت {
        إرجاع src_type == other.src_type && dst_type == other.dst_type;
    }
};

بنية ggml_webgpu_cpy_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_cpy_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.src_type);
        ggml_webgpu_hash_combine(seed, مفتاح.dst_type);
        إرجاع seed;
    }
};

/** Glu **/

بنية ggml_webgpu_glu_pipeline_مفتاح {
    ggml_glu_op glu_op;
    ggml_type   type;
    منطقي        split;
    منطقي        src_overlap;

    منطقي operator==(ثابت ggml_webgpu_glu_pipeline_مفتاح & other) ثابت {
        إرجاع glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_glu_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_glu_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.glu_op);
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        ggml_webgpu_hash_combine(seed, مفتاح.split);
        ggml_webgpu_hash_combine(seed, مفتاح.src_overlap);
        إرجاع seed;
    }
};

/** Rope **/

بنية ggml_webgpu_rope_pipeline_مفتاح {
    ggml_type type;
    منطقي      inplace;
    منطقي      has_ff;

    منطقي operator==(ثابت ggml_webgpu_rope_pipeline_مفتاح & other) ثابت {
        إرجاع type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

بنية ggml_webgpu_rope_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_rope_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.type);
        ggml_webgpu_hash_combine(seed, مفتاح.inplace);
        ggml_webgpu_hash_combine(seed, مفتاح.has_ff);
        إرجاع seed;
    }
};

/** SoftMax **/

بنية ggml_webgpu_soft_max_pipeline_مفتاح {
    ggml_type mask_type;
    منطقي      has_mask;
    منطقي      has_sink;
    منطقي      inplace;

    منطقي operator==(ثابت ggml_webgpu_soft_max_pipeline_مفتاح & other) ثابت {
        إرجاع mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

بنية ggml_webgpu_soft_max_pipeline_مفتاح_hash {
    size_t operator()(ثابت ggml_webgpu_soft_max_pipeline_مفتاح & مفتاح) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مفتاح.mask_type);
        ggml_webgpu_hash_combine(seed, مفتاح.has_mask);
        ggml_webgpu_hash_combine(seed, مفتاح.has_sink);
        ggml_webgpu_hash_combine(seed, مفتاح.inplace);
        إرجاع seed;
    }
};

/** MMVQ **/

مضمن منطقي ggml_webgpu_can_use_mmvq(ثابت ggml_tensor * src0,
                                     ثابت ggml_tensor * src1,
                                     منطقي                supports_dot_product,
                                     ثابت std::string & vendor) {
    إذا (src1->ne[1] <= 4) {
        منطقي supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        إذا (supports_dp4a && supports_dot_product) {
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
                            إرجاع src0->ne[0] % 4 == 0;
                        افتراضي:
                            اكسر;
                    }
                    اكسر;
                افتراضي:
                    اكسر;
            }
        }
    }
    إرجاع خطأ;
}

صنف ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<صحيح, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<صحيح, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<صحيح, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<صحيح, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<صحيح, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_مفتاح_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_مفتاح_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_unary_pipeline_مفتاح_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_scale_pipeline_مفتاح_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_مفتاح_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_مفتاح_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_مفتاح_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_مفتاح,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_مفتاح_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_pad_pipeline_مفتاح_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_ثنائي_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_ثنائي_pipeline_مفتاح_hash>
        ثنائي_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_add_id_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_add_id_pipeline_مفتاح_hash>
        add_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_concat_pipeline_مفتاح_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_repeat_pipeline_مفتاح_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_مفتاح,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_مفتاح_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_مفتاح_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_مفتاح,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_مفتاح_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_مفتاح,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_مفتاح_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_مفتاح_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_مفتاح_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_مفتاح_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_مفتاح,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_مفتاح_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_مفتاح_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_مفتاح_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_مفتاح_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_set_pipeline_مفتاح_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_cpy_pipeline_مفتاح_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_glu_pipeline_مفتاح_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_rope_pipeline_مفتاح_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_مفتاح_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_مفتاح_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_مفتاح_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_im2col_pipeline_مفتاح_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_مفتاح,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_مفتاح_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_مفتاح, webgpu_pipeline, ggml_webgpu_upscale_pipeline_مفتاح_hash>
        upscale_pipelines;

  عام:
    ggml_webgpu_shader_lib(wgpu::Device device) { هذا->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        auto it = sum_rows_pipelines.find(1);
        إذا (it != sum_rows_pipelines.end()) {
            إرجاع it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        إرجاع sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_مفتاح مفتاح = {};
        مفتاح.op                                = context.dst->op;
        مفتاح.src_type                          = context.src0->type;
        مفتاح.dst_type                          = context.dst->type;
        مفتاح.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = row_norm_pipelines.find(مفتاح);
        إذا (it != row_norm_pipelines.end()) {
            إرجاع it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        بدل (مفتاح.op) {
            حالة GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                اكسر;
            حالة GGML_OP_NORM:
                defines.push_back("NORM");
                variant = "norm";
                اكسر;
            حالة GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        إذا (مفتاح.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (مفتاح.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } وإلا إذا (مفتاح.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        إذا (مفتاح.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } وإلا إذا (مفتاح.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ثابت uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = مفتاح.inplace;
        row_norm_pipelines[مفتاح]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[مفتاح].context = decisions;
        إرجاع row_norm_pipelines[مفتاح];
    }

    webgpu_pipeline get_argmax_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي vec4 = context.src0->ne[0] % 4 == 0;

        auto it = argmax_pipelines.find(vec4);
        إذا (it != argmax_pipelines.end()) {
            إرجاع it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        إذا (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        auto processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        إرجاع argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_مفتاح مفتاح       = {};
        مفتاح.src0_type                               = context.src0->type;
        مفتاح.dst_type                                = context.dst->type;
        مفتاح.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        مفتاح.i64_idx     = context.src1->type == GGML_TYPE_I64;
        مفتاح.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_size(context.dst->type)) % 2 == 0);

        auto it = set_rows_pipelines.find(مفتاح);
        إذا (it != set_rows_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        بدل (context.dst->type) {
            حالة GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                اكسر;
            حالة GGML_TYPE_Q8_0:
                defines.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                اكسر;
            حالة GGML_TYPE_Q4_0:
                defines.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        إذا (context.src0->type == GGML_TYPE_F16) {
            defines.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        إذا (مفتاح.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        إذا (مفتاح.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        إذا (مفتاح.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        ثابت auto & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        auto         processed          = preprocessor.preprocess(shader_source, defines);
        auto         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = مفتاح.vec4;
        decisions->i64_idx              = مفتاح.i64_idx;
        decisions->pair_blocks          = مفتاح.pair_blocks;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[مفتاح]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[مفتاح].context = decisions;
        إرجاع set_rows_pipelines[مفتاح];
    }

    webgpu_pipeline get_set_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_مفتاح مفتاح = {};
        مفتاح.type                         = context.dst->type;
        مفتاح.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = set_pipelines.find(مفتاح);
        إذا (it != set_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for set shader");
        }

        إذا (مفتاح.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_set, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = مفتاح.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[مفتاح]       = pipeline;
        إرجاع set_pipelines[مفتاح];
    }

    webgpu_pipeline get_cumsum_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        auto it = cumsum_pipelines.find(1);
        إذا (it != cumsum_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        إرجاع cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_قمة_k = context.dst->op == GGML_OP_قمة_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t order =
            is_قمة_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_pipelines.find(order);
        إذا (it != argsort_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        بينما (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        auto decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        إرجاع argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_قمة_k = context.dst->op == GGML_OP_قمة_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t order =
            is_قمة_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_merge_pipelines.find(order);
        إذا (it != argsort_merge_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        إرجاع argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي vec4_aligned) {
        ثابت منطقي vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_مفتاح مفتاح = {};
        مفتاح.src_type                          = context.src0->type;
        مفتاح.vectorized                        = (صحيح) vectorized;

        auto it = get_rows_pipelines.find(مفتاح);
        إذا (it != get_rows_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        ثابت بنية ggml_type_traits * type_traits = ggml_get_type_traits(مفتاح.src_type);
        ثابت حرف *                    type_str    = type_traits->type_name;

        بدل (مفتاح.src_type) {
            حالة GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                إذا (مفتاح.vectorized) {
                    defines.push_back("F32_VEC");
                    defines.push_back("SRC_TYPE=vec4<f32>");
                    defines.push_back("DST_TYPE=vec4<f32>");
                    defines.push_back("BLOCK_SIZE=4u");
                } وإلا {
                    defines.push_back("F32");
                    defines.push_back("SRC_TYPE=f32");
                    defines.push_back("DST_TYPE=f32");
                    defines.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                اكسر;
            حالة GGML_TYPE_BF16:
                defines.push_back("BF16");
                defines.push_back("SRC_TYPE=u32");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    بدل (مفتاح.src_type) {
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
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                اكسر;
                            }
                        افتراضي:
                            {
                                defines.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back(type_upper + "_T");
                    defines.push_back(type_upper);
                    defines.push_back(type_upper + "_SCALE_MIN");
                    defines.push_back(type_upper + "_TABLES");
                    defines.push_back(type_upper + "_GRID");
                    defines.push_back(type_upper + "_LUT");

                    variant += "_";
                    variant += type_str;

                    defines.push_back("DST_TYPE=f32");

                    إذا (مفتاح.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } وإلا إذا ((مفتاح.src_type >= GGML_TYPE_Q4_0 && مفتاح.src_type <= GGML_TYPE_Q8_1) ||
                               مفتاح.src_type == GGML_TYPE_IQ4_NL || مفتاح.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } وإلا إذا (مفتاح.src_type == GGML_TYPE_Q2_0 || مفتاح.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } وإلا إذا (مفتاح.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } وإلا {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    اكسر;
                }
        }

        إذا (مفتاح.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        auto decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[مفتاح]  = pipeline;
        إرجاع get_rows_pipelines[مفتاح];
    }

    webgpu_pipeline get_scale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_مفتاح مفتاح = {};
        مفتاح.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = scale_pipelines.find(مفتاح);
        إذا (it != scale_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        إذا (مفتاح.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_scale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = مفتاح.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[مفتاح]     = pipeline;
        إرجاع scale_pipelines[مفتاح];
    }

    webgpu_pipeline get_solve_tri_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_مفتاح مفتاح = {};
        مفتاح.type                               = context.dst->type;
        مفتاح.n                                  = (صحيح) context.src0->ne[0];
        مفتاح.k                                  = (صحيح) context.src1->ne[0];

        auto it = solve_tri_pipelines.find(مفتاح);
        إذا (it != solve_tri_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ثابت uint32_t wg_size       = std::min((uint32_t) مفتاح.n, context.max_wg_size);
        ثابت uint32_t k_tile        = wg_size;
        ثابت uint32_t bytes_per_row = ((uint32_t) مفتاح.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        ثابت uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(مفتاح.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        auto processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[مفتاح] = pipeline;
        إرجاع solve_tri_pipelines[مفتاح];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_مفتاح مفتاح = {};
        مفتاح.type                              = context.dst->type;
        مفتاح.vectorized                        = context.src1->ne[0] == 4;

        auto it = ssm_conv_pipelines.find(مفتاح);
        إذا (it != ssm_conv_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        إذا (مفتاح.vectorized) {
            defines.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t block_size    = 32u;
        constexpr uint32_t tokens_per_wg = 8u;

        defines.push_back("BLOCK_SIZE=" + std::to_string(block_size) + "u");
        defines.push_back("TOKENS_PER_WG=" + std::to_string(tokens_per_wg) + "u");

        auto processed           = preprocessor.preprocess(wgsl_ssm_conv, defines);
        auto decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->block_size    = block_size;
        decisions->tokens_per_wg = tokens_per_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[مفتاح]  = pipeline;
        إرجاع ssm_conv_pipelines[مفتاح];
    }

    webgpu_pipeline get_ssm_scan_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_مفتاح مفتاح = {};
        مفتاح.type                              = context.dst->type;
        مفتاح.d_state                           = (صحيح) context.src0->ne[0];
        مفتاح.xbc_overlap                       = xbc_overlap;

        auto it = ssm_scan_pipelines.find(مفتاح);
        إذا (it != ssm_scan_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ثابت uint32_t wg_size = (uint32_t) مفتاح.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        إذا (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } وإلا {
            variant += "_wg_reduce";
        }

        إذا (مفتاح.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } وإلا إذا (مفتاح.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } وإلا إذا (مفتاح.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(مفتاح.d_state);

        auto processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        auto decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = مفتاح.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[مفتاح]    = pipeline;
        إرجاع ssm_scan_pipelines[مفتاح];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_مفتاح مفتاح = {};
        مفتاح.type                                     = context.dst->type;
        مفتاح.s_v                                      = (صحيح) context.src2->ne[0];
        مفتاح.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        مفتاح.fused_cache                              = context.dst_fuse != nullptr;

        auto it = gated_delta_net_pipelines.find(مفتاح);
        إذا (it != gated_delta_net_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        إذا (مفتاح.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        إذا (مفتاح.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(مفتاح.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(مفتاح.s_v) + "u");

        auto            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[مفتاح] = pipeline;
        إرجاع gated_delta_net_pipelines[مفتاح];
    }

    webgpu_pipeline get_pad_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_مفتاح مفتاح = {};
        مفتاح.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        auto it = pad_pipelines.find(مفتاح);
        إذا (it != pad_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        إذا (مفتاح.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_pad, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[مفتاح]       = pipeline;
        إرجاع pad_pipelines[مفتاح];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_مفتاح مفتاح = {};
        مفتاح.src0_type                            = context.src0->type;

        auto it = quantize_q8_pipelines.find(مفتاح);
        إذا (it != quantize_q8_pipelines.end()) {
            إرجاع it->second;
        }
        ثابت حرف *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> defines;
        std::string              variant = "quantize_q8";

        uint32_t wg_size = WEBGPU_MUL_MAT_VEC_WG_SIZE;

        defines.push_back("SRC1_INNER_TYPE=f32");
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_upper  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

        defines.push_back("MUL_ACC_" + type_upper);
        defines.push_back("Q8_1_T");

        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        auto processed             = preprocessor.preprocess(shader_src, defines);
        auto decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size         = wg_size;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[مفتاح] = pipeline;
        إرجاع quantize_q8_pipelines[مفتاح];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_مفتاح مفتاح = {};
        مفتاح.src0_type                            = context.src0->type;
        مفتاح.src1_type                            = context.src1->type;
        مفتاح.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        مفتاح.رقم_cols   = context.dst->ne[1];
        مفتاح.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
        مفتاح.src_overlap = src_overlap;

        auto it = mul_mat_vec_pipelines.find(مفتاح);
        إذا (it != mul_mat_vec_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    بدل (context.src0->type) {
                        حالة GGML_TYPE_Q4_0:
                        حالة GGML_TYPE_Q4_1:
                        حالة GGML_TYPE_Q5_0:
                        حالة GGML_TYPE_Q5_1:
                        حالة GGML_TYPE_Q8_0:
                            إذا (مفتاح.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_Q1_0:
                        حالة GGML_TYPE_Q2_K:
                        حالة GGML_TYPE_Q3_K:
                        حالة GGML_TYPE_Q4_K:
                        حالة GGML_TYPE_Q5_K:
                        حالة GGML_TYPE_Q6_K:
                            إذا (مفتاح.use_mmvq) {
                                defines.push_back("K_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                            defines.push_back(type_upper + "_LUT");
                            إذا (مفتاح.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
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
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(مفتاح.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        إذا (مفتاح.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } وإلا إذا (مفتاح.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } وإلا إذا (مفتاح.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        إذا (مفتاح.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        إذا (مفتاح.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (مفتاح.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(مفتاح.رقم_cols));

        auto processed            = preprocessor.preprocess(shader_src, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = مفتاح.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[مفتاح] = pipeline;
        إرجاع mul_mat_vec_pipelines[مفتاح];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_mul_mat_pipeline_مفتاح مفتاح = {};
        مفتاح.src0_type                        = context.src0->type;
        مفتاح.src1_type                        = context.src1->type;
        مفتاح.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        مفتاح.use_subgroup_matrix = context.supports_subgroup_matrix;
        مفتاح.src_overlap         = src_overlap;

        auto it = mul_mat_fast_pipelines.find(مفتاح);
        إذا (it != mul_mat_fast_pipelines.end()) {
            إرجاع it->second;
        }

        ثابت حرف * shader_src = مفتاح.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = مفتاح.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("TYPE_F32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("TYPE_F16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("TYPE_BF16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }

                    variant += std::string("_") + src0_name;
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(مفتاح.vectorized ? "VEC" : "SCALAR");

        ثابت منطقي is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        إذا (مفتاح.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } وإلا {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        إذا (مفتاح.use_subgroup_matrix) {
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            defines.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            defines.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            defines.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M) + "u");
            defines.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N) + "u");
            defines.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            defines.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            defines.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        إذا (مفتاح.vectorized) {
            variant += "_vectorized";
        }

        إذا (مفتاح.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        إذا (!مفتاح.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = مفتاح.use_subgroup_matrix;
        إذا (مفتاح.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_size           = context.max_subgroup_size;
        } وإلا {
            decisions->wg_size_m       = WEBGPU_MUL_MAT_WG_SIZE_M;
            decisions->wg_size_n       = WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->wg_size         = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->mul_mat_wg_size = WEBGPU_MUL_MAT_WG_SIZE;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[مفتاح] = pipeline;
        إرجاع mul_mat_fast_pipelines[مفتاح];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_مفتاح مفتاح = {};
        مفتاح.n_experts                                  = (uint32_t) context.src0->ne[2];

        auto it = mul_mat_id_gather_pipelines.find(مفتاح);
        إذا (it != mul_mat_id_gather_pipelines.end()) {
            إرجاع it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(مفتاح.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        auto processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[مفتاح] = pipeline;
        إرجاع pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_مفتاح مفتاح = {};
        مفتاح.src0_type                           = context.src0->type;
        مفتاح.src1_type                           = context.src1->type;
        مفتاح.n_experts                           = context.src0->ne[2];
        مفتاح.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_pipelines.find(مفتاح);
        إذا (it != mul_mat_id_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ثابت حرف *                    src0_name   = src0_traits->type_name;

        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }

                    variant += std::string("_") + src0_name;
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(مفتاح.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ثابت uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        إذا (مفتاح.vectorized) {
            variant += "_vectorized";
        }

        auto processed = preprocessor.preprocess(wgsl_mul_mat_id, defines);

        auto decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_size_m = WEBGPU_MUL_MAT_WG_SIZE_M;
        decisions->wg_size_n = WEBGPU_MUL_MAT_WG_SIZE_N;
        decisions->wg_size   = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[مفتاح] = pipeline;
        إرجاع mul_mat_id_pipelines[مفتاح];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_مفتاح مفتاح = {};
        مفتاح.src0_type                           = context.src0->type;
        مفتاح.src1_type                           = context.src1->type;
        مفتاح.n_experts                           = context.src0->ne[2];
        مفتاح.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_vec_pipelines.find(مفتاح);
        إذا (it != mul_mat_id_vec_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        ثابت حرف *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        بدل (context.src1->type) {
            حالة GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        بدل (context.src0->type) {
            حالة GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                اكسر;
            افتراضي:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ثابت بنية ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    بدل (context.src0->type) {
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            اكسر;
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            اكسر;
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            اكسر;
                        افتراضي:
                            اكسر;
                    }
                    اكسر;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(مفتاح.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        إذا (مفتاح.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } وإلا إذا (مفتاح.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } وإلا إذا (مفتاح.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (مفتاح.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(مفتاح.n_experts));

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[مفتاح] = pipeline;
        إرجاع mul_mat_id_vec_pipelines[مفتاح];
    }

    webgpu_pipeline get_unary_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                     is_unary = context.dst->op == GGML_OP_UNARY;
        ثابت صحيح                      op       = is_unary ? (صحيح) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_مفتاح مفتاح      = {};
        مفتاح.type                                = context.dst->type;
        مفتاح.op                                  = op;
        مفتاح.is_unary                            = is_unary;
        مفتاح.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        مفتاح.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        auto it = unary_pipelines.find(مفتاح);
        إذا (it != unary_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            مفتاح.is_unary ? ggml_unary_op_name((ggml_unary_op) مفتاح.op) : ggml_op_name((ggml_op) مفتاح.op);
        defines.push_back(variant);

        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for unary shader");
        }

        إذا (مفتاح.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (op == GGML_OP_TRI) {
            بدل (مفتاح.ttype) {
                حالة GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    اكسر;
                حالة GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    اكسر;
                حالة GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    اكسر;
                حالة GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    اكسر;
                افتراضي:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_unary, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = مفتاح.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[مفتاح]     = pipeline;
        إرجاع unary_pipelines[مفتاح];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_مفتاح مفتاح = {};
        مفتاح.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        مفتاح.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        مفتاح.src_overlap                           = src_overlap;

        auto it = rms_norm_mul_pipelines.find(مفتاح);
        إذا (it != rms_norm_mul_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        إذا (مفتاح.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (مفتاح.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (مفتاح.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = مفتاح.inplace;
        pipeline_decisions->overlap     = مفتاح.overlap;
        pipeline_decisions->src_overlap = مفتاح.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[مفتاح]     = pipeline;
        إرجاع rms_norm_mul_pipelines[مفتاح];
    }

    webgpu_pipeline get_ثنائي_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_ثنائي_pipeline_مفتاح مفتاح = {};
        مفتاح.type                            = context.dst->type;
        مفتاح.op                              = context.dst->op;
        مفتاح.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        مفتاح.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        مفتاح.src_overlap                     = src_overlap;

        auto it = ثنائي_pipelines.find(مفتاح);
        إذا (it != ثنائي_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) مفتاح.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for binary shader");
        }

        إذا (مفتاح.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (مفتاح.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (مفتاح.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_ثنائي, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_ثنائي_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = مفتاح.inplace;
        pipeline_decisions->overlap     = مفتاح.overlap;
        pipeline_decisions->src_overlap = مفتاح.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ثنائي_pipelines[مفتاح]    = pipeline;
        إرجاع ثنائي_pipelines[مفتاح];
    }

    webgpu_pipeline get_add_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_مفتاح مفتاح = {};
        مفتاح.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = add_id_pipelines.find(مفتاح);
        إذا (it != add_id_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        ثابت حرف *             shader_src = wgsl_add_id;

        إذا (مفتاح.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed              = preprocessor.preprocess(shader_src, defines);
        auto pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_size = context.max_wg_size;
        pipeline_decisions->inplace = مفتاح.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[مفتاح]    = pipeline;
        إرجاع pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_concat_pipeline_مفتاح مفتاح = {};
        مفتاح.type                            = context.dst->type;
        مفتاح.src_overlap                     = src_overlap;

        auto it = concat_pipelines.find(مفتاح);
        إذا (it != concat_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for concat shader");
        }

        إذا (مفتاح.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_concat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_ثنائي_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = مفتاح.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[مفتاح]    = pipeline;
        إرجاع concat_pipelines[مفتاح];
    }

    webgpu_pipeline get_repeat_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_مفتاح مفتاح = {};
        مفتاح.type                            = context.dst->type;

        auto it = repeat_pipelines.find(مفتاح);
        إذا (it != repeat_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                اكسر;
            حالة GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_repeat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[مفتاح]    = pipeline;
        إرجاع repeat_pipelines[مفتاح];
    }

    webgpu_pipeline get_flash_attn_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ثابت منطقي can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_مفتاح مفتاح = {};
        مفتاح.common                              = ggml_webgpu_flash_attn_make_common_pipeline_مفتاح(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        مفتاح.common.k_direct &= decisions.use_sg_matrix && مفتاح.common.k_type == GGML_TYPE_F16;
        مفتاح.common.v_direct &= decisions.use_sg_matrix && مفتاح.common.v_type == GGML_TYPE_F16;
        مفتاح.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_limit_bytes, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            مفتاح.common.head_dim_qk, مفتاح.common.head_dim_v, مفتاح.common.has_mask,
            مفتاح.common.k_direct || مفتاح.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        إذا (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = خطأ;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            مفتاح.common.k_direct     = خطأ;
            مفتاح.common.v_direct     = خطأ;
            مفتاح.use_sg_matrix       = خطأ;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, مفتاح.common.head_dim_qk, مفتاح.common.head_dim_v,
                مفتاح.common.has_mask, مفتاح.common.k_direct || مفتاح.common.v_direct);
            GGML_ASSERT(max_kv_tile > 0);
        }

        decisions.kv_tile = decisions.use_sg_matrix ?
                                std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES) :
                                std::min(GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE, max_kv_tile);
        decisions.wg_size =
            decisions.use_sg_matrix ?
                std::max(context.max_subgroup_size, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE) :
                std::min(context.max_wg_size, std::max(GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE,
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * context.max_subgroup_size));

        إذا (مفتاح.common.k_direct || مفتاح.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            بينما (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_size;
            }
        }

        auto it = flash_attn_pipelines.find(مفتاح);
        إذا (it != flash_attn_pipelines.end()) {
            إرجاع it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(مفتاح.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_size);
        ثابت حرف *             shader_src = nullptr;
        إذا (!مفتاح.use_sg_matrix) {
            shader_src = wgsl_flash_attn_tile;
            defines.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_size) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_size) + "_" +
                       std::to_string(context.max_subgroup_size);
        } وإلا {
            shader_src = wgsl_flash_attn;
            defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[مفتاح] = pipeline;
        إرجاع flash_attn_pipelines[مفتاح];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_مفتاح مفتاح = {};
        مفتاح.common = ggml_webgpu_flash_attn_make_common_pipeline_مفتاح(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        auto it = flash_attn_vec_pipelines.find(مفتاح);
        إذا (it != flash_attn_vec_pipelines.end()) {
            إرجاع it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, مفتاح.common.head_dim_qk,
                                                                   مفتاح.common.head_dim_v, مفتاح.common.has_mask,
                                                                   مفتاح.common.k_direct || مفتاح.common.v_direct);
        decisions.wg_size = context.max_subgroup_size;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(مفتاح.common, variant, 1u, decisions.kv_tile, decisions.wg_size);
        إذا (مفتاح.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (حجم("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_size;
        ثابت uint32_t D       = مفتاح.common.head_dim_qk | مفتاح.common.head_dim_v;
        ثابت uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_size, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[مفتاح] = pipeline;
        إرجاع flash_attn_vec_pipelines[مفتاح];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_مفتاح مفتاح = {};
        مفتاح.kv_tile                                 = kv_tile;
        auto it                                     = flash_attn_blk_pipelines.find(مفتاح);
        إذا (it != flash_attn_blk_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(مفتاح.kv_tile));
        variant += std::string("_kvt") + std::to_string(مفتاح.kv_tile);

        uint32_t wg_size = 1;
        بينما ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[مفتاح] = pipeline;
        إرجاع flash_attn_blk_pipelines[مفتاح];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_مفتاح مفتاح = {};
        مفتاح.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        مفتاح.dst_type                                       = context.dst->type;
        مفتاح.wg_size                                        = context.max_wg_size;
        auto it                                            = flash_attn_vec_reduce_pipelines.find(مفتاح);
        إذا (it != flash_attn_vec_reduce_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        بدل (مفتاح.dst_type) {
            حالة GGML_TYPE_F32:
                defines.push_back("DST_F32");
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("DST_F16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(مفتاح.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(مفتاح.head_dim_v));
        variant += std::string("_hsv") + std::to_string(مفتاح.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[مفتاح] = pipeline;
        إرجاع flash_attn_vec_reduce_pipelines[مفتاح];
    }

    webgpu_pipeline get_cpy_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_مفتاح مفتاح = {};
        مفتاح.src_type                     = context.src0->type;
        مفتاح.dst_type                     = context.dst->type;

        auto it = cpy_pipelines.find(مفتاح);
        إذا (it != cpy_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        بدل (مفتاح.src_type) {
            حالة GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                defines.push_back("SRC_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        بدل (مفتاح.dst_type) {
            حالة GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                اكسر;
            حالة GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_cpy, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[مفتاح]       = pipeline;
        إرجاع cpy_pipelines[مفتاح];
    }

    webgpu_pipeline get_glu_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_glu_pipeline_مفتاح مفتاح = {};
        مفتاح.glu_op                       = ggml_get_glu_op(context.dst);
        مفتاح.type                         = context.dst->type;
        مفتاح.split                        = (context.src1 != nullptr);
        مفتاح.src_overlap                  = src_overlap;

        auto it = glu_pipelines.find(مفتاح);
        إذا (it != glu_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        بدل (مفتاح.glu_op) {
            حالة GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                اكسر;
            حالة GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                اكسر;
            حالة GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                اكسر;
            حالة GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                اكسر;
            حالة GGML_GLU_OP_SWIGLU_CLAMP:
                defines.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported GLU op");
        }
        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        إذا (مفتاح.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } وإلا إذا (مفتاح.split) {
            variant += "_split";
        } وإلا {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_glu, defines);
        auto decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = مفتاح.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[مفتاح]       = pipeline;
        إرجاع glu_pipelines[مفتاح];
    }

    webgpu_pipeline get_rope_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_مفتاح مفتاح = {};
        مفتاح.type                          = context.dst->type;
        مفتاح.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        مفتاح.has_ff                        = (context.src2 != nullptr);

        auto it = rope_pipelines.find(مفتاح);
        إذا (it != rope_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        بدل (مفتاح.type) {
            حالة GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        إذا (مفتاح.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (مفتاح.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_rope, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = مفتاح.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[مفتاح]      = pipeline;
        إرجاع rope_pipelines[مفتاح];
    }

    webgpu_pipeline get_soft_max_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_مفتاح مفتاح = {};
        مفتاح.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        مفتاح.has_mask                          = (context.src1 != nullptr);
        مفتاح.has_sink                          = (context.src2 != nullptr);
        مفتاح.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = soft_max_pipelines.find(مفتاح);
        إذا (it != soft_max_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        إذا (مفتاح.has_mask) {
            defines.push_back("HAS_MASK");
            بدل (مفتاح.mask_type) {
                حالة GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    اكسر;
                حالة GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    اكسر;
                افتراضي:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        إذا (مفتاح.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        إذا (مفتاح.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = مفتاح.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[مفتاح]  = pipeline;
        إرجاع soft_max_pipelines[مفتاح];
    }

    webgpu_pipeline get_conv2d_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_مفتاح مفتاح = {};
        مفتاح.weight_type                     = context.src0->type;
        مفتاح.إدخال_type                      = context.src1->type;
        مفتاح.output_type                     = context.dst->type;

        auto it = conv2d_pipelines.find(مفتاح);
        إذا (it != conv2d_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        auto ادفع_type_defines = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            إذا (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ادفع_type_defines("WEIGHT_TYPE", مفتاح.weight_type);
        ادفع_type_defines("INPUT_TYPE", مفتاح.إدخال_type);
        ادفع_type_defines("OUTPUT_TYPE", مفتاح.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[مفتاح]    = pipeline;
        إرجاع conv2d_pipelines[مفتاح];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي whcn) {
        ggml_webgpu_conv2d_dw_pipeline_مفتاح مفتاح = {};
        مفتاح.weight_type                        = context.src0->type;
        مفتاح.إدخال_type                         = context.src1->type;
        مفتاح.output_type                        = context.dst->type;
        مفتاح.whcn                               = whcn;

        auto it = conv2d_dw_pipelines.find(مفتاح);
        إذا (it != conv2d_dw_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        auto ادفع_type_defines = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            إذا (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ادفع_type_defines("WEIGHT_TYPE", مفتاح.weight_type);
        ادفع_type_defines("INPUT_TYPE", مفتاح.إدخال_type);
        ادفع_type_defines("OUTPUT_TYPE", مفتاح.output_type);

        إذا (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[مفتاح] = pipeline;
        إرجاع conv2d_dw_pipelines[مفتاح];
    }

    webgpu_pipeline get_im2col_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_مفتاح مفتاح = {};
        مفتاح.إدخال_type                      = context.src1->type;
        مفتاح.output_type                     = context.dst->type;

        auto it = im2col_pipelines.find(مفتاح);
        إذا (it != im2col_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        auto ادفع_type_defines = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            إذا (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ادفع_type_defines("INPUT_TYPE", مفتاح.إدخال_type);
        ادفع_type_defines("OUTPUT_TYPE", مفتاح.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[مفتاح]    = pipeline;
        إرجاع im2col_pipelines[مفتاح];
    }

    webgpu_pipeline get_upscale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ثابت uint32_t base_mode  = mode_flags & 0xFFu;
        ثابت منطقي     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_مفتاح مفتاح = {};
        مفتاح.إدخال_type                       = context.src0->type;
        مفتاح.output_type                      = context.dst->type;
        مفتاح.base_mode                        = base_mode;
        مفتاح.antialias                        = antialias;

        auto it = upscale_pipelines.find(مفتاح);
        إذا (it != upscale_pipelines.end()) {
            إرجاع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        إذا (مفتاح.إدخال_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } وإلا {
            variant += "_src_f32";
        }

        إذا (مفتاح.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } وإلا {
            variant += "_dst_f32";
        }

        بدل (base_mode) {
            حالة GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                اكسر;
            حالة GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                اكسر;
            حالة GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported upscale mode");
        }

        إذا (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_upscale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[مفتاح]   = pipeline;
        إرجاع upscale_pipelines[مفتاح];
    }

  خاص:
    ساكن webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_code,
                                                       std::string    label) {
        wgpu::ShaderSourceWGSL shader_source;
        shader_source.code = shader_code.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.nextInChain = &shader_source;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = nullptr;  // nullptr means auto layout
        إرجاع { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
