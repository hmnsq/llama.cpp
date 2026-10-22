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
    أرجع (uintptr_t) base_tensor->data + tensor->view_offs;
}

مضمن منطقي ggml_webgpu_tensor_equal(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    أرجع a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
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

بنية ggml_webgpu_نبيو_shader_decisions {
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

بنية ggml_webgpu_ssm_scan_pipeline_مبودي {
    صحيح     type;
    صحيح     d_state;
    uint8_t xbc_overlap;
    منطقي    a_overlap;
    منطقي    ids_overlap;

    منطقي operator==(ثابت ggml_webgpu_ssm_scan_pipeline_مبودي & other) ثابت {
        أرجع type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

بنية ggml_webgpu_ssm_scan_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_ssm_scan_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.type);
        ggml_webgpu_hash_combine(seed, مبودي.d_state);
        ggml_webgpu_hash_combine(seed, مبودي.xbc_overlap);
        ggml_webgpu_hash_combine(seed, مبودي.a_overlap);
        ggml_webgpu_hash_combine(seed, مبودي.ids_overlap);
        أرجع seed;
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

بنية ggml_webgpu_set_rows_pipeline_مبودي {
    صحيح src0_type;
    صحيح dst_type;
    صحيح vec4;
    صحيح i64_idx;
    صحيح pair_blocks;

    منطقي operator==(ثابت ggml_webgpu_set_rows_pipeline_مبودي & other) ثابت {
        أرجع src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

بنية ggml_webgpu_set_rows_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_set_rows_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.src0_type);
        ggml_webgpu_hash_combine(seed, مبودي.dst_type);
        ggml_webgpu_hash_combine(seed, مبودي.vec4);
        ggml_webgpu_hash_combine(seed, مبودي.i64_idx);
        ggml_webgpu_hash_combine(seed, مبودي.pair_blocks);
        أرجع seed;
    }
};

بنية ggml_webgpu_set_rows_shader_decisions {
    منطقي     vec4;
    منطقي     i64_idx;
    منطقي     pair_blocks;
    uint32_t wg_size;
};

/** Set **/

بنية ggml_webgpu_set_pipeline_مبودي {
    ggml_type type;
    منطقي      inplace;

    منطقي operator==(ثابت ggml_webgpu_set_pipeline_مبودي & other) ثابت {
        أرجع type == other.type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_set_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_set_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.type);
        ggml_webgpu_hash_combine(seed, مبودي.inplace);
        أرجع seed;
    }
};

/** Get Rows **/

بنية ggml_webgpu_get_rows_pipeline_مبودي {
    ggml_type src_type;
    صحيح       vectorized;

    منطقي operator==(ثابت ggml_webgpu_get_rows_pipeline_مبودي & other) ثابت {
        أرجع src_type == other.src_type && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_get_rows_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_get_rows_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.src_type);
        ggml_webgpu_hash_combine(seed, مبودي.vectorized);
        أرجع seed;
    }
};

/** Row Norm **/

بنية ggml_webgpu_row_norm_pipeline_مبودي {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    منطقي      inplace;

    منطقي operator==(ثابت ggml_webgpu_row_norm_pipeline_مبودي & other) ثابت {
        أرجع op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

بنية ggml_webgpu_row_norm_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_row_norm_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.op);
        ggml_webgpu_hash_combine(seed, مبودي.src_type);
        ggml_webgpu_hash_combine(seed, مبودي.dst_type);
        ggml_webgpu_hash_combine(seed, مبودي.inplace);
        أرجع seed;
    }
};

/** RMS_NORM + MUL **/

بنية ggml_webgpu_rms_norm_mul_pipeline_مبودي {
    منطقي inplace;      // rn_src == dst
    منطقي overlap;      // mul_src == dst
    منطقي src_overlap;  // rn_src binding overlaps mul_src binding

    منطقي operator==(ثابت ggml_webgpu_rms_norm_mul_pipeline_مبودي & other) ثابت {
        أرجع inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_rms_norm_mul_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_rms_norm_mul_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.inplace);
        ggml_webgpu_hash_combine(seed, مبودي.overlap);
        ggml_webgpu_hash_combine(seed, مبودي.src_overlap);
        أرجع seed;
    }
};

بنية ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    منطقي     inplace     = خطأ;
    منطقي     overlap     = خطأ;
    منطقي     src_overlap = خطأ;
};

/** Pad **/
بنية ggml_webgpu_pad_pipeline_مبودي {
    منطقي circular;

    منطقي operator==(ثابت ggml_webgpu_pad_pipeline_مبودي & other) ثابت { أرجع circular == other.circular; }
};

بنية ggml_webgpu_pad_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_pad_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.circular);
        أرجع seed;
    }
};

/** Solve Tri **/
بنية ggml_webgpu_solve_tri_pipeline_مبودي {
    صحيح type;
    صحيح n;
    صحيح k;

    منطقي operator==(ثابت ggml_webgpu_solve_tri_pipeline_مبودي & other) ثابت {
        أرجع type == other.type && n == other.n && k == other.k;
    }
};

بنية ggml_webgpu_solve_tri_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_solve_tri_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.type);
        ggml_webgpu_hash_combine(seed, مبودي.n);
        ggml_webgpu_hash_combine(seed, مبودي.k);
        أرجع seed;
    }
};

/** SSM Conv **/
بنية ggml_webgpu_ssm_conv_pipeline_مبودي {
    صحيح type;
    صحيح vectorized;

    منطقي operator==(ثابت ggml_webgpu_ssm_conv_pipeline_مبودي & other) ثابت {
        أرجع type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
بنية ggml_webgpu_conv2d_pipeline_مبودي {
    ggml_type weight_type;
    ggml_type شغرو_type;
    ggml_type output_type;

    منطقي operator==(ثابت ggml_webgpu_conv2d_pipeline_مبودي & other) ثابت {
        أرجع weight_type == other.weight_type && شغرو_type == other.شغرو_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_conv2d_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_conv2d_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.weight_type);
        ggml_webgpu_hash_combine(seed, مبودي.شغرو_type);
        ggml_webgpu_hash_combine(seed, مبودي.output_type);
        أرجع seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
بنية ggml_webgpu_conv2d_dw_pipeline_مبودي {
    ggml_type weight_type;
    ggml_type شغرو_type;
    ggml_type output_type;
    منطقي      whcn;

    منطقي operator==(ثابت ggml_webgpu_conv2d_dw_pipeline_مبودي & other) ثابت {
        أرجع weight_type == other.weight_type && شغرو_type == other.شغرو_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

بنية ggml_webgpu_conv2d_dw_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_conv2d_dw_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.weight_type);
        ggml_webgpu_hash_combine(seed, مبودي.شغرو_type);
        ggml_webgpu_hash_combine(seed, مبودي.output_type);
        ggml_webgpu_hash_combine(seed, مبودي.whcn);
        أرجع seed;
    }
};

/** Im2Col **/
بنية ggml_webgpu_im2col_pipeline_مبودي {
    ggml_type شغرو_type;
    ggml_type output_type;

    منطقي operator==(ثابت ggml_webgpu_im2col_pipeline_مبودي & other) ثابت {
        أرجع شغرو_type == other.شغرو_type && output_type == other.output_type;
    }
};

بنية ggml_webgpu_im2col_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_im2col_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.شغرو_type);
        ggml_webgpu_hash_combine(seed, مبودي.output_type);
        أرجع seed;
    }
};

/** Gated Delta Net **/
بنية ggml_webgpu_gated_delta_net_pipeline_مبودي {
    صحيح  type;
    صحيح  s_v;
    صحيح  kda;
    منطقي fused_cache;

    منطقي operator==(ثابت ggml_webgpu_gated_delta_net_pipeline_مبودي & other) ثابت {
        أرجع type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

بنية ggml_webgpu_gated_delta_net_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_gated_delta_net_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.type);
        ggml_webgpu_hash_combine(seed, مبودي.s_v);
        ggml_webgpu_hash_combine(seed, مبودي.kda);
        أرجع seed;
    }
};

بنية ggml_webgpu_ssm_conv_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_ssm_conv_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.type);
        ggml_webgpu_hash_combine(seed, مبودي.vectorized);
        أرجع seed;
    }
};

/** Scale **/

بنية ggml_webgpu_scale_pipeline_مبودي {
    صحيح inplace;

    منطقي operator==(ثابت ggml_webgpu_scale_pipeline_مبودي & other) ثابت { أرجع inplace == other.inplace; }
};

بنية ggml_webgpu_scale_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_scale_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.inplace);
        أرجع seed;
    }
};

/** Upscale **/

بنية ggml_webgpu_upscale_pipeline_مبودي {
    ggml_type شغرو_type;
    ggml_type output_type;
    uint32_t  base_mode;
    منطقي      antialias;

    منطقي operator==(ثابت ggml_webgpu_upscale_pipeline_مبودي & other) ثابت {
        أرجع شغرو_type == other.شغرو_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

بنية ggml_webgpu_upscale_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_upscale_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.شغرو_type);
        ggml_webgpu_hash_combine(seed, مبودي.output_type);
        ggml_webgpu_hash_combine(seed, مبودي.base_mode);
        ggml_webgpu_hash_combine(seed, مبودي.antialias);
        أرجع seed;
    }
};

/** Concat **/

بنية ggml_webgpu_concat_pipeline_مبودي {
    صحيح  type;
    منطقي src_overlap;

    منطقي operator==(ثابت ggml_webgpu_concat_pipeline_مبودي & other) ثابت {
        أرجع type == other.type && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_concat_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_concat_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.type);
        ggml_webgpu_hash_combine(seed, مبودي.src_overlap);
        أرجع seed;
    }
};

/** Repeat **/

بنية ggml_webgpu_repeat_pipeline_مبودي {
    صحيح type;

    منطقي operator==(ثابت ggml_webgpu_repeat_pipeline_مبودي & other) ثابت { أرجع type == other.type; }
};

بنية ggml_webgpu_repeat_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_repeat_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.type);
        أرجع seed;
    }
};

/** Binary **/

بنية ggml_webgpu_نبيو_pipeline_مبودي {
    صحيح  type;
    صحيح  op;
    منطقي inplace;
    منطقي overlap;
    منطقي src_overlap;

    منطقي operator==(ثابت ggml_webgpu_نبيو_pipeline_مبودي & other) ثابت {
        أرجع type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_نبيو_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_نبيو_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.type);
        ggml_webgpu_hash_combine(seed, مبودي.op);
        ggml_webgpu_hash_combine(seed, مبودي.inplace);
        ggml_webgpu_hash_combine(seed, مبودي.overlap);
        ggml_webgpu_hash_combine(seed, مبودي.src_overlap);
        أرجع seed;
    }
};

/* Add_Id */

بنية ggml_webgpu_add_id_pipeline_مبودي {
    منطقي inplace;

    منطقي operator==(ثابت ggml_webgpu_add_id_pipeline_مبودي & other) ثابت { أرجع inplace == other.inplace; }
};

بنية ggml_webgpu_add_id_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_add_id_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.inplace);
        أرجع seed;
    }
};

/** Unary **/

بنية ggml_webgpu_unary_pipeline_مبودي {
    صحيح           type;
    صحيح           op;
    منطقي          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    منطقي          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    منطقي operator==(ثابت ggml_webgpu_unary_pipeline_مبودي & other) ثابت {
        أرجع type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

بنية ggml_webgpu_unary_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_unary_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.type);
        ggml_webgpu_hash_combine(seed, مبودي.op);
        ggml_webgpu_hash_combine(seed, مبودي.is_unary);
        ggml_webgpu_hash_combine(seed, مبودي.inplace);
        ggml_webgpu_hash_combine(seed, مبودي.ttype);
        أرجع seed;
    }
};

/** FlashAttention */

بنية ggml_webgpu_flash_attn_common_pipeline_مبودي {
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

    منطقي operator==(ثابت ggml_webgpu_flash_attn_common_pipeline_مبودي & other) ثابت {
        أرجع q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

مضمن فراغ ggml_webgpu_flash_attn_hash_common_pipeline_مبودي(size_t &                                           seed,
                                                            ثابت ggml_webgpu_flash_attn_common_pipeline_مبودي & مبودي) {
    ggml_webgpu_hash_combine(seed, مبودي.q_type);
    ggml_webgpu_hash_combine(seed, مبودي.k_type);
    ggml_webgpu_hash_combine(seed, مبودي.v_type);
    ggml_webgpu_hash_combine(seed, مبودي.dst_type);
    ggml_webgpu_hash_combine(seed, مبودي.head_dim_qk);
    ggml_webgpu_hash_combine(seed, مبودي.head_dim_v);
    ggml_webgpu_hash_combine(seed, مبودي.k_direct);
    ggml_webgpu_hash_combine(seed, مبودي.v_direct);
    ggml_webgpu_hash_combine(seed, مبودي.kv_overlap);
    ggml_webgpu_hash_combine(seed, مبودي.has_mask);
    ggml_webgpu_hash_combine(seed, مبودي.has_sinks);
    ggml_webgpu_hash_combine(seed, مبودي.uses_logit_softcap);
}

بنية ggml_webgpu_flash_attn_vec_pipeline_مبودي {
    ggml_webgpu_flash_attn_common_pipeline_مبودي common;

    منطقي operator==(ثابت ggml_webgpu_flash_attn_vec_pipeline_مبودي & other) ثابت { أرجع common == other.common; }
};

بنية ggml_webgpu_flash_attn_vec_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_flash_attn_vec_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_مبودي(seed, مبودي.common);
        أرجع seed;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_مبودي {
    ggml_webgpu_flash_attn_common_pipeline_مبودي common;
    منطقي                                       use_sg_matrix;

    منطقي operator==(ثابت ggml_webgpu_flash_attn_pipeline_مبودي & other) ثابت {
        أرجع common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

بنية ggml_webgpu_flash_attn_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_flash_attn_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_مبودي(seed, مبودي.common);
        ggml_webgpu_hash_combine(seed, مبودي.use_sg_matrix);
        أرجع seed;
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
    أرجع reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K, size_t storage_offset_alignment) {
    ثابت uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_size(K->type));
    أرجع offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

مضمن منطقي ggml_webgpu_flash_attn_float_vec4_aligned(ثابت ggml_tensor * K,
                                                      ثابت ggml_tensor * V,
                                                      size_t              storage_offset_alignment) {
    أرجع ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

مضمن منطقي ggml_webgpu_flash_attn_k_direct(ثابت ggml_tensor * Q, ثابت ggml_tensor * K, uint32_t kv_direct_align) {
    أرجع (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

مضمن منطقي ggml_webgpu_flash_attn_v_direct(ثابت ggml_tensor * Q, ثابت ggml_tensor * V, uint32_t kv_direct_align) {
    أرجع ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

مضمن ggml_webgpu_flash_attn_common_pipeline_مبودي ggml_webgpu_flash_attn_make_common_pipeline_مبودي(
    ثابت ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    منطقي                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_مبودي مبودي = {};
    مبودي.q_type                                     = context.src0->type;
    مبودي.k_type                                     = context.src1->type;
    مبودي.v_type                                     = context.src2->type;
    مبودي.dst_type                                   = context.dst->type;
    مبودي.head_dim_qk                                = (uint32_t) context.src0->ne[0];
    مبودي.head_dim_v                                 = (uint32_t) context.src2->ne[0];
    مبودي.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    مبودي.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    مبودي.kv_overlap         = kv_overlap;
    مبودي.has_mask           = context.src3 != nullptr;
    مبودي.has_sinks          = context.src4 != nullptr;
    مبودي.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    أرجع مبودي;
}

مضمن std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    ثابت ggml_webgpu_flash_attn_common_pipeline_مبودي & مبودي,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_size) {
    std::vector<std::string> defines;

    بدل (مبودي.k_type) {
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
    variant += std::string("_k") + ggml_type_name(مبودي.k_type);

    بدل (مبودي.v_type) {
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
    variant += std::string("_v") + ggml_type_name(مبودي.v_type);

    بدل (مبودي.q_type) {
        حالة GGML_TYPE_F32:
            defines.push_back("Q_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            defines.push_back("Q_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(مبودي.q_type);

    بدل (مبودي.dst_type) {
        حالة GGML_TYPE_F32:
            defines.push_back("DST_F32");
            اكسر;
        حالة GGML_TYPE_F16:
            defines.push_back("DST_F16");
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(مبودي.dst_type);

    إذا (مبودي.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    إذا (مبودي.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    إذا (مبودي.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    إذا (مبودي.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    إذا (مبودي.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    إذا (مبودي.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(مبودي.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(مبودي.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(مبودي.head_dim_v));
    variant += std::string("_hsv") + std::to_string(مبودي.head_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    إذا (ggml_is_quantized(مبودي.k_type) || ggml_is_quantized(مبودي.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        إذا (ggml_is_quantized(مبودي.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        إذا (ggml_is_quantized(مبودي.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    أرجع defines;
}

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_مبودي {
    uint32_t  head_dim_v;
    uint32_t  wg_size;
    ggml_type dst_type;
};

بنية ggml_webgpu_flash_attn_vec_reduce_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.head_dim_v);
        ggml_webgpu_hash_combine(seed, مبودي.wg_size);
        ggml_webgpu_hash_combine(seed, مبودي.dst_type);
        أرجع seed;
    }
};

مضمن منطقي operator==(ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_مبودي & lhs,
                       ثابت ggml_webgpu_flash_attn_vec_reduce_pipeline_مبودي & rhs) {
    أرجع lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size && lhs.dst_type == rhs.dst_type;
}

بنية ggml_webgpu_flash_attn_blk_pipeline_مبودي {
    uint32_t kv_tile;

    منطقي operator==(ثابت ggml_webgpu_flash_attn_blk_pipeline_مبودي & other) ثابت { أرجع kv_tile == other.kv_tile; }
};

بنية ggml_webgpu_flash_attn_blk_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_flash_attn_blk_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.kv_tile);
        أرجع seed;
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
    أرجع f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
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
        أرجع 0;
    }
    ثابت size_t one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, head_dim_qk, head_dim_v, has_mask, kv_direct);
    ثابت size_t bytes_per_kv = one_kv_bytes - base_q_bytes;
    إذا (bytes_per_kv == 0) {
        أرجع 0;
    }
    ثابت size_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    أرجع (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
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
        طالما (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    أرجع kv_tile;
}

مضمن منطقي ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(منطقي                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ثابت ggml_tensor * Q,
                                                                ثابت ggml_tensor * V) {
    أرجع supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

بنية ggml_webgpu_mul_mat_vec_pipeline_مبودي {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    uint32_t  لمب_cols;
    منطقي      use_mmvq;
    منطقي      src_overlap;

    منطقي operator==(ثابت ggml_webgpu_mul_mat_vec_pipeline_مبودي & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               لمب_cols == other.لمب_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_vec_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_mul_mat_vec_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.src0_type);
        ggml_webgpu_hash_combine(seed, مبودي.src1_type);
        ggml_webgpu_hash_combine(seed, مبودي.vectorized);
        ggml_webgpu_hash_combine(seed, مبودي.لمب_cols);
        ggml_webgpu_hash_combine(seed, مبودي.use_mmvq);
        ggml_webgpu_hash_combine(seed, مبودي.src_overlap);
        أرجع seed;
    }
};

بنية ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
    منطقي     src_overlap = خطأ;
};

بنية ggml_webgpu_quantize_q8_pipeline_مبودي {
    ggml_type src0_type;

    منطقي operator==(ثابت ggml_webgpu_quantize_q8_pipeline_مبودي & other) ثابت { أرجع src0_type == other.src0_type; }
};

بنية ggml_webgpu_quantize_q8_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_quantize_q8_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.src0_type);
        أرجع seed;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_مبودي {
    ggml_type src0_type;
    ggml_type src1_type;
    صحيح       vectorized;
    صحيح       use_subgroup_matrix;
    منطقي      src_overlap;

    منطقي operator==(ثابت ggml_webgpu_mul_mat_pipeline_مبودي & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_mul_mat_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_mul_mat_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.src0_type);
        ggml_webgpu_hash_combine(seed, مبودي.src1_type);
        ggml_webgpu_hash_combine(seed, مبودي.vectorized);
        ggml_webgpu_hash_combine(seed, مبودي.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, مبودي.src_overlap);
        أرجع seed;
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

بنية ggml_webgpu_mul_mat_id_gather_pipeline_مبودي {
    uint32_t n_experts;

    منطقي operator==(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_مبودي & other) ثابت {
        أرجع n_experts == other.n_experts;
    }
};

بنية ggml_webgpu_mul_mat_id_gather_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_mul_mat_id_gather_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.n_experts);
        أرجع seed;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_مبودي {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  لمب_cols;
    صحيح       vectorized;

    منطقي operator==(ثابت ggml_webgpu_mul_mat_id_pipeline_مبودي & other) ثابت {
        أرجع src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               لمب_cols == other.لمب_cols && vectorized == other.vectorized;
    }
};

بنية ggml_webgpu_mul_mat_id_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_mul_mat_id_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.src0_type);
        ggml_webgpu_hash_combine(seed, مبودي.src1_type);
        ggml_webgpu_hash_combine(seed, مبودي.n_experts);
        ggml_webgpu_hash_combine(seed, مبودي.لمب_cols);
        ggml_webgpu_hash_combine(seed, مبودي.vectorized);
        أرجع seed;
    }
};

/** Cpy **/

بنية ggml_webgpu_cpy_pipeline_مبودي {
    ggml_type src_type;
    ggml_type dst_type;

    منطقي operator==(ثابت ggml_webgpu_cpy_pipeline_مبودي & other) ثابت {
        أرجع src_type == other.src_type && dst_type == other.dst_type;
    }
};

بنية ggml_webgpu_cpy_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_cpy_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.src_type);
        ggml_webgpu_hash_combine(seed, مبودي.dst_type);
        أرجع seed;
    }
};

/** Glu **/

بنية ggml_webgpu_glu_pipeline_مبودي {
    ggml_glu_op glu_op;
    ggml_type   type;
    منطقي        split;
    منطقي        src_overlap;

    منطقي operator==(ثابت ggml_webgpu_glu_pipeline_مبودي & other) ثابت {
        أرجع glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

بنية ggml_webgpu_glu_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_glu_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.glu_op);
        ggml_webgpu_hash_combine(seed, مبودي.type);
        ggml_webgpu_hash_combine(seed, مبودي.split);
        ggml_webgpu_hash_combine(seed, مبودي.src_overlap);
        أرجع seed;
    }
};

/** Rope **/

بنية ggml_webgpu_rope_pipeline_مبودي {
    ggml_type type;
    منطقي      inplace;
    منطقي      has_ff;

    منطقي operator==(ثابت ggml_webgpu_rope_pipeline_مبودي & other) ثابت {
        أرجع type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

بنية ggml_webgpu_rope_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_rope_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.type);
        ggml_webgpu_hash_combine(seed, مبودي.inplace);
        ggml_webgpu_hash_combine(seed, مبودي.has_ff);
        أرجع seed;
    }
};

/** SoftMax **/

بنية ggml_webgpu_soft_max_pipeline_مبودي {
    ggml_type mask_type;
    منطقي      has_mask;
    منطقي      has_sink;
    منطقي      inplace;

    منطقي operator==(ثابت ggml_webgpu_soft_max_pipeline_مبودي & other) ثابت {
        أرجع mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

بنية ggml_webgpu_soft_max_pipeline_مبودي_hash {
    size_t operator()(ثابت ggml_webgpu_soft_max_pipeline_مبودي & مبودي) ثابت {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, مبودي.mask_type);
        ggml_webgpu_hash_combine(seed, مبودي.has_mask);
        ggml_webgpu_hash_combine(seed, مبودي.has_sink);
        ggml_webgpu_hash_combine(seed, مبودي.inplace);
        أرجع seed;
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

    std::unordered_map<صحيح, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<صحيح, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<صحيح, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<صحيح, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<صحيح, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_مبودي_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_مبودي_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_unary_pipeline_مبودي_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_scale_pipeline_مبودي_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_مبودي_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_مبودي_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_مبودي_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_مبودي,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_مبودي_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_pad_pipeline_مبودي_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_نبيو_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_نبيو_pipeline_مبودي_hash>
        نبيو_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_add_id_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_add_id_pipeline_مبودي_hash>
        add_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_concat_pipeline_مبودي_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_repeat_pipeline_مبودي_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_مبودي,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_مبودي_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_مبودي_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_مبودي,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_مبودي_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_مبودي,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_مبودي_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_مبودي_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_مبودي_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_مبودي_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_مبودي,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_مبودي_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_مبودي_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_مبودي_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_مبودي_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_set_pipeline_مبودي_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_cpy_pipeline_مبودي_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_glu_pipeline_مبودي_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_rope_pipeline_مبودي_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_مبودي_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_مبودي_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_مبودي_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_im2col_pipeline_مبودي_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_مبودي,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_مبودي_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_مبودي, webgpu_pipeline, ggml_webgpu_upscale_pipeline_مبودي_hash>
        upscale_pipelines;

  عام:
    ggml_webgpu_shader_lib(wgpu::Device device) { هذا->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        auto it = sum_rows_pipelines.find(1);
        إذا (it != sum_rows_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        أرجع sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_مبودي مبودي = {};
        مبودي.op                                = context.dst->op;
        مبودي.src_type                          = context.src0->type;
        مبودي.dst_type                          = context.dst->type;
        مبودي.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = row_norm_pipelines.find(مبودي);
        إذا (it != row_norm_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        بدل (مبودي.op) {
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

        إذا (مبودي.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (مبودي.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } وإلا إذا (مبودي.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        إذا (مبودي.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } وإلا إذا (مبودي.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ثابت uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = مبودي.inplace;
        row_norm_pipelines[مبودي]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[مبودي].context = decisions;
        أرجع row_norm_pipelines[مبودي];
    }

    webgpu_pipeline get_argmax_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي vec4 = context.src0->ne[0] % 4 == 0;

        auto it = argmax_pipelines.find(vec4);
        إذا (it != argmax_pipelines.end()) {
            أرجع it->second;
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
        أرجع argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_مبودي مبودي       = {};
        مبودي.src0_type                               = context.src0->type;
        مبودي.dst_type                                = context.dst->type;
        مبودي.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        مبودي.i64_idx     = context.src1->type == GGML_TYPE_I64;
        مبودي.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_size(context.dst->type)) % 2 == 0);

        auto it = set_rows_pipelines.find(مبودي);
        إذا (it != set_rows_pipelines.end()) {
            أرجع it->second;
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

        إذا (مبودي.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        إذا (مبودي.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        إذا (مبودي.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        ثابت auto & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        auto         processed          = preprocessor.preprocess(shader_source, defines);
        auto         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = مبودي.vec4;
        decisions->i64_idx              = مبودي.i64_idx;
        decisions->pair_blocks          = مبودي.pair_blocks;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[مبودي]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[مبودي].context = decisions;
        أرجع set_rows_pipelines[مبودي];
    }

    webgpu_pipeline get_set_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_مبودي مبودي = {};
        مبودي.type                         = context.dst->type;
        مبودي.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = set_pipelines.find(مبودي);
        إذا (it != set_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        بدل (مبودي.type) {
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

        إذا (مبودي.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_set, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = مبودي.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[مبودي]       = pipeline;
        أرجع set_pipelines[مبودي];
    }

    webgpu_pipeline get_cumsum_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        auto it = cumsum_pipelines.find(1);
        إذا (it != cumsum_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        أرجع cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_كولي_k = context.dst->op == GGML_OP_كولي_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t order =
            is_كولي_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_pipelines.find(order);
        إذا (it != argsort_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        طالما (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        auto decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        أرجع argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        منطقي          is_كولي_k = context.dst->op == GGML_OP_كولي_K;
        // ascending order is 0, descending order is 1
        ثابت int32_t order =
            is_كولي_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_merge_pipelines.find(order);
        إذا (it != argsort_merge_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        أرجع argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي vec4_aligned) {
        ثابت منطقي vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_مبودي مبودي = {};
        مبودي.src_type                          = context.src0->type;
        مبودي.vectorized                        = (صحيح) vectorized;

        auto it = get_rows_pipelines.find(مبودي);
        إذا (it != get_rows_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        ثابت بنية ggml_type_traits * type_traits = ggml_get_type_traits(مبودي.src_type);
        ثابت حرف *                    type_str    = type_traits->type_name;

        بدل (مبودي.src_type) {
            حالة GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                إذا (مبودي.vectorized) {
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

                    بدل (مبودي.src_type) {
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

                    إذا (مبودي.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } وإلا إذا ((مبودي.src_type >= GGML_TYPE_Q4_0 && مبودي.src_type <= GGML_TYPE_Q8_1) ||
                               مبودي.src_type == GGML_TYPE_IQ4_NL || مبودي.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } وإلا إذا (مبودي.src_type == GGML_TYPE_Q2_0 || مبودي.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } وإلا إذا (مبودي.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } وإلا {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    اكسر;
                }
        }

        إذا (مبودي.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        auto decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[مبودي]  = pipeline;
        أرجع get_rows_pipelines[مبودي];
    }

    webgpu_pipeline get_scale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_مبودي مبودي = {};
        مبودي.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = scale_pipelines.find(مبودي);
        إذا (it != scale_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        إذا (مبودي.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_scale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = مبودي.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[مبودي]     = pipeline;
        أرجع scale_pipelines[مبودي];
    }

    webgpu_pipeline get_solve_tri_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_مبودي مبودي = {};
        مبودي.type                               = context.dst->type;
        مبودي.n                                  = (صحيح) context.src0->ne[0];
        مبودي.k                                  = (صحيح) context.src1->ne[0];

        auto it = solve_tri_pipelines.find(مبودي);
        إذا (it != solve_tri_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        بدل (مبودي.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ثابت uint32_t wg_size       = std::min((uint32_t) مبودي.n, context.max_wg_size);
        ثابت uint32_t k_tile        = wg_size;
        ثابت uint32_t bytes_per_row = ((uint32_t) مبودي.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        ثابت uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(مبودي.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        auto processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[مبودي] = pipeline;
        أرجع solve_tri_pipelines[مبودي];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_مبودي مبودي = {};
        مبودي.type                              = context.dst->type;
        مبودي.vectorized                        = context.src1->ne[0] == 4;

        auto it = ssm_conv_pipelines.find(مبودي);
        إذا (it != ssm_conv_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        بدل (مبودي.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        إذا (مبودي.vectorized) {
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
        ssm_conv_pipelines[مبودي]  = pipeline;
        أرجع ssm_conv_pipelines[مبودي];
    }

    webgpu_pipeline get_ssm_scan_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_مبودي مبودي = {};
        مبودي.type                              = context.dst->type;
        مبودي.d_state                           = (صحيح) context.src0->ne[0];
        مبودي.xbc_overlap                       = xbc_overlap;

        auto it = ssm_scan_pipelines.find(مبودي);
        إذا (it != ssm_scan_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        بدل (مبودي.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ثابت uint32_t wg_size = (uint32_t) مبودي.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        إذا (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } وإلا {
            variant += "_wg_reduce";
        }

        إذا (مبودي.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } وإلا إذا (مبودي.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } وإلا إذا (مبودي.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(مبودي.d_state);

        auto processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        auto decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = مبودي.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[مبودي]    = pipeline;
        أرجع ssm_scan_pipelines[مبودي];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_مبودي مبودي = {};
        مبودي.type                                     = context.dst->type;
        مبودي.s_v                                      = (صحيح) context.src2->ne[0];
        مبودي.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        مبودي.fused_cache                              = context.dst_fuse != nullptr;

        auto it = gated_delta_net_pipelines.find(مبودي);
        إذا (it != gated_delta_net_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        بدل (مبودي.type) {
            حالة GGML_TYPE_F32:
                variant += "_f32";
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        إذا (مبودي.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        إذا (مبودي.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(مبودي.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(مبودي.s_v) + "u");

        auto            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[مبودي] = pipeline;
        أرجع gated_delta_net_pipelines[مبودي];
    }

    webgpu_pipeline get_pad_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_مبودي مبودي = {};
        مبودي.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        auto it = pad_pipelines.find(مبودي);
        إذا (it != pad_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        إذا (مبودي.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_pad, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[مبودي]       = pipeline;
        أرجع pad_pipelines[مبودي];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_مبودي مبودي = {};
        مبودي.src0_type                            = context.src0->type;

        auto it = quantize_q8_pipelines.find(مبودي);
        إذا (it != quantize_q8_pipelines.end()) {
            أرجع it->second;
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
        quantize_q8_pipelines[مبودي] = pipeline;
        أرجع quantize_q8_pipelines[مبودي];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_مبودي مبودي = {};
        مبودي.src0_type                            = context.src0->type;
        مبودي.src1_type                            = context.src1->type;
        مبودي.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        مبودي.لمب_cols   = context.dst->ne[1];
        مبودي.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
        مبودي.src_overlap = src_overlap;

        auto it = mul_mat_vec_pipelines.find(مبودي);
        إذا (it != mul_mat_vec_pipelines.end()) {
            أرجع it->second;
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
                            إذا (مبودي.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            اكسر;
                        حالة GGML_TYPE_Q1_0:
                        حالة GGML_TYPE_Q2_K:
                        حالة GGML_TYPE_Q3_K:
                        حالة GGML_TYPE_Q4_K:
                        حالة GGML_TYPE_Q5_K:
                        حالة GGML_TYPE_Q6_K:
                            إذا (مبودي.use_mmvq) {
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
                            إذا (مبودي.use_mmvq) {
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
        defines.push_back(مبودي.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        إذا (مبودي.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } وإلا إذا (مبودي.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } وإلا إذا (مبودي.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        إذا (مبودي.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        إذا (مبودي.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (مبودي.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(مبودي.لمب_cols));

        auto processed            = preprocessor.preprocess(shader_src, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = مبودي.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[مبودي] = pipeline;
        أرجع mul_mat_vec_pipelines[مبودي];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_mul_mat_pipeline_مبودي مبودي = {};
        مبودي.src0_type                        = context.src0->type;
        مبودي.src1_type                        = context.src1->type;
        مبودي.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        مبودي.use_subgroup_matrix = context.supports_subgroup_matrix;
        مبودي.src_overlap         = src_overlap;

        auto it = mul_mat_fast_pipelines.find(مبودي);
        إذا (it != mul_mat_fast_pipelines.end()) {
            أرجع it->second;
        }

        ثابت حرف * shader_src = مبودي.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = مبودي.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

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
        defines.push_back(مبودي.vectorized ? "VEC" : "SCALAR");

        ثابت منطقي is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        إذا (مبودي.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } وإلا {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        إذا (مبودي.use_subgroup_matrix) {
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
        إذا (مبودي.vectorized) {
            variant += "_vectorized";
        }

        إذا (مبودي.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        إذا (!مبودي.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = مبودي.use_subgroup_matrix;
        إذا (مبودي.use_subgroup_matrix) {
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
        mul_mat_fast_pipelines[مبودي] = pipeline;
        أرجع mul_mat_fast_pipelines[مبودي];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_مبودي مبودي = {};
        مبودي.n_experts                                  = (uint32_t) context.src0->ne[2];

        auto it = mul_mat_id_gather_pipelines.find(مبودي);
        إذا (it != mul_mat_id_gather_pipelines.end()) {
            أرجع it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(مبودي.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        auto processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[مبودي] = pipeline;
        أرجع pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_مبودي مبودي = {};
        مبودي.src0_type                           = context.src0->type;
        مبودي.src1_type                           = context.src1->type;
        مبودي.n_experts                           = context.src0->ne[2];
        مبودي.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_pipelines.find(مبودي);
        إذا (it != mul_mat_id_pipelines.end()) {
            أرجع it->second;
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
        defines.push_back(مبودي.vectorized ? "VEC" : "SCALAR");

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
        إذا (مبودي.vectorized) {
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
        mul_mat_id_pipelines[مبودي] = pipeline;
        أرجع mul_mat_id_pipelines[مبودي];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_مبودي مبودي = {};
        مبودي.src0_type                           = context.src0->type;
        مبودي.src1_type                           = context.src1->type;
        مبودي.n_experts                           = context.src0->ne[2];
        مبودي.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_vec_pipelines.find(مبودي);
        إذا (it != mul_mat_id_vec_pipelines.end()) {
            أرجع it->second;
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
        defines.push_back(مبودي.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        إذا (مبودي.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } وإلا إذا (مبودي.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } وإلا إذا (مبودي.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        إذا (مبودي.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(مبودي.n_experts));

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[مبودي] = pipeline;
        أرجع mul_mat_id_vec_pipelines[مبودي];
    }

    webgpu_pipeline get_unary_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت منطقي                     is_unary = context.dst->op == GGML_OP_UNARY;
        ثابت صحيح                      op       = is_unary ? (صحيح) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_مبودي مبودي      = {};
        مبودي.type                                = context.dst->type;
        مبودي.op                                  = op;
        مبودي.is_unary                            = is_unary;
        مبودي.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        مبودي.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        auto it = unary_pipelines.find(مبودي);
        إذا (it != unary_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            مبودي.is_unary ? ggml_unary_op_name((ggml_unary_op) مبودي.op) : ggml_op_name((ggml_op) مبودي.op);
        defines.push_back(variant);

        بدل (مبودي.type) {
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

        إذا (مبودي.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (op == GGML_OP_TRI) {
            بدل (مبودي.ttype) {
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
        decisions->inplace       = مبودي.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[مبودي]     = pipeline;
        أرجع unary_pipelines[مبودي];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_مبودي مبودي = {};
        مبودي.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        مبودي.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        مبودي.src_overlap                           = src_overlap;

        auto it = rms_norm_mul_pipelines.find(مبودي);
        إذا (it != rms_norm_mul_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        إذا (مبودي.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (مبودي.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (مبودي.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = مبودي.inplace;
        pipeline_decisions->overlap     = مبودي.overlap;
        pipeline_decisions->src_overlap = مبودي.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[مبودي]     = pipeline;
        أرجع rms_norm_mul_pipelines[مبودي];
    }

    webgpu_pipeline get_نبيو_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_نبيو_pipeline_مبودي مبودي = {};
        مبودي.type                            = context.dst->type;
        مبودي.op                              = context.dst->op;
        مبودي.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        مبودي.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        مبودي.src_overlap                     = src_overlap;

        auto it = نبيو_pipelines.find(مبودي);
        إذا (it != نبيو_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) مبودي.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        بدل (مبودي.type) {
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

        إذا (مبودي.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } وإلا إذا (مبودي.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } وإلا إذا (مبودي.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_نبيو, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_نبيو_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = مبودي.inplace;
        pipeline_decisions->overlap     = مبودي.overlap;
        pipeline_decisions->src_overlap = مبودي.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        نبيو_pipelines[مبودي]    = pipeline;
        أرجع نبيو_pipelines[مبودي];
    }

    webgpu_pipeline get_add_id_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_مبودي مبودي = {};
        مبودي.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = add_id_pipelines.find(مبودي);
        إذا (it != add_id_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        ثابت حرف *             shader_src = wgsl_add_id;

        إذا (مبودي.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed              = preprocessor.preprocess(shader_src, defines);
        auto pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_size = context.max_wg_size;
        pipeline_decisions->inplace = مبودي.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[مبودي]    = pipeline;
        أرجع pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_concat_pipeline_مبودي مبودي = {};
        مبودي.type                            = context.dst->type;
        مبودي.src_overlap                     = src_overlap;

        auto it = concat_pipelines.find(مبودي);
        إذا (it != concat_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        بدل (مبودي.type) {
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

        إذا (مبودي.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_concat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_نبيو_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = مبودي.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[مبودي]    = pipeline;
        أرجع concat_pipelines[مبودي];
    }

    webgpu_pipeline get_repeat_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_مبودي مبودي = {};
        مبودي.type                            = context.dst->type;

        auto it = repeat_pipelines.find(مبودي);
        إذا (it != repeat_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        بدل (مبودي.type) {
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
        repeat_pipelines[مبودي]    = pipeline;
        أرجع repeat_pipelines[مبودي];
    }

    webgpu_pipeline get_flash_attn_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ثابت منطقي can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_مبودي مبودي = {};
        مبودي.common                              = ggml_webgpu_flash_attn_make_common_pipeline_مبودي(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        مبودي.common.k_direct &= decisions.use_sg_matrix && مبودي.common.k_type == GGML_TYPE_F16;
        مبودي.common.v_direct &= decisions.use_sg_matrix && مبودي.common.v_type == GGML_TYPE_F16;
        مبودي.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_limit_bytes, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            مبودي.common.head_dim_qk, مبودي.common.head_dim_v, مبودي.common.has_mask,
            مبودي.common.k_direct || مبودي.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        إذا (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = خطأ;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            مبودي.common.k_direct     = خطأ;
            مبودي.common.v_direct     = خطأ;
            مبودي.use_sg_matrix       = خطأ;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, مبودي.common.head_dim_qk, مبودي.common.head_dim_v,
                مبودي.common.has_mask, مبودي.common.k_direct || مبودي.common.v_direct);
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

        إذا (مبودي.common.k_direct || مبودي.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            طالما (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_size;
            }
        }

        auto it = flash_attn_pipelines.find(مبودي);
        إذا (it != flash_attn_pipelines.end()) {
            أرجع it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(مبودي.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_size);
        ثابت حرف *             shader_src = nullptr;
        إذا (!مبودي.use_sg_matrix) {
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
        flash_attn_pipelines[مبودي] = pipeline;
        أرجع flash_attn_pipelines[مبودي];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_مبودي مبودي = {};
        مبودي.common = ggml_webgpu_flash_attn_make_common_pipeline_مبودي(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        auto it = flash_attn_vec_pipelines.find(مبودي);
        إذا (it != flash_attn_vec_pipelines.end()) {
            أرجع it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, مبودي.common.head_dim_qk,
                                                                   مبودي.common.head_dim_v, مبودي.common.has_mask,
                                                                   مبودي.common.k_direct || مبودي.common.v_direct);
        decisions.wg_size = context.max_subgroup_size;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(مبودي.common, variant, 1u, decisions.kv_tile, decisions.wg_size);
        إذا (مبودي.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (حجم("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_size;
        ثابت uint32_t D       = مبودي.common.head_dim_qk | مبودي.common.head_dim_v;
        ثابت uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_size, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[مبودي] = pipeline;
        أرجع flash_attn_vec_pipelines[مبودي];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ثابت ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_مبودي مبودي = {};
        مبودي.kv_tile                                 = kv_tile;
        auto it                                     = flash_attn_blk_pipelines.find(مبودي);
        إذا (it != flash_attn_blk_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(مبودي.kv_tile));
        variant += std::string("_kvt") + std::to_string(مبودي.kv_tile);

        uint32_t wg_size = 1;
        طالما ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[مبودي] = pipeline;
        أرجع flash_attn_blk_pipelines[مبودي];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_مبودي مبودي = {};
        مبودي.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        مبودي.dst_type                                       = context.dst->type;
        مبودي.wg_size                                        = context.max_wg_size;
        auto it                                            = flash_attn_vec_reduce_pipelines.find(مبودي);
        إذا (it != flash_attn_vec_reduce_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        بدل (مبودي.dst_type) {
            حالة GGML_TYPE_F32:
                defines.push_back("DST_F32");
                اكسر;
            حالة GGML_TYPE_F16:
                defines.push_back("DST_F16");
                اكسر;
            افتراضي:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(مبودي.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(مبودي.head_dim_v));
        variant += std::string("_hsv") + std::to_string(مبودي.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[مبودي] = pipeline;
        أرجع flash_attn_vec_reduce_pipelines[مبودي];
    }

    webgpu_pipeline get_cpy_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_مبودي مبودي = {};
        مبودي.src_type                     = context.src0->type;
        مبودي.dst_type                     = context.dst->type;

        auto it = cpy_pipelines.find(مبودي);
        إذا (it != cpy_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        بدل (مبودي.src_type) {
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

        بدل (مبودي.dst_type) {
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
        cpy_pipelines[مبودي]       = pipeline;
        أرجع cpy_pipelines[مبودي];
    }

    webgpu_pipeline get_glu_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي src_overlap) {
        ggml_webgpu_glu_pipeline_مبودي مبودي = {};
        مبودي.glu_op                       = ggml_get_glu_op(context.dst);
        مبودي.type                         = context.dst->type;
        مبودي.split                        = (context.src1 != nullptr);
        مبودي.src_overlap                  = src_overlap;

        auto it = glu_pipelines.find(مبودي);
        إذا (it != glu_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        بدل (مبودي.glu_op) {
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
        بدل (مبودي.type) {
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

        إذا (مبودي.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } وإلا إذا (مبودي.split) {
            variant += "_split";
        } وإلا {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_glu, defines);
        auto decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = مبودي.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[مبودي]       = pipeline;
        أرجع glu_pipelines[مبودي];
    }

    webgpu_pipeline get_rope_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_مبودي مبودي = {};
        مبودي.type                          = context.dst->type;
        مبودي.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        مبودي.has_ff                        = (context.src2 != nullptr);

        auto it = rope_pipelines.find(مبودي);
        إذا (it != rope_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        بدل (مبودي.type) {
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

        إذا (مبودي.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        إذا (مبودي.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_rope, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = مبودي.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[مبودي]      = pipeline;
        أرجع rope_pipelines[مبودي];
    }

    webgpu_pipeline get_soft_max_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_مبودي مبودي = {};
        مبودي.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        مبودي.has_mask                          = (context.src1 != nullptr);
        مبودي.has_sink                          = (context.src2 != nullptr);
        مبودي.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = soft_max_pipelines.find(مبودي);
        إذا (it != soft_max_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        إذا (مبودي.has_mask) {
            defines.push_back("HAS_MASK");
            بدل (مبودي.mask_type) {
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

        إذا (مبودي.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        إذا (مبودي.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = مبودي.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[مبودي]  = pipeline;
        أرجع soft_max_pipelines[مبودي];
    }

    webgpu_pipeline get_conv2d_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_مبودي مبودي = {};
        مبودي.weight_type                     = context.src0->type;
        مبودي.شغرو_type                      = context.src1->type;
        مبودي.output_type                     = context.dst->type;

        auto it = conv2d_pipelines.find(مبودي);
        إذا (it != conv2d_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        auto تور_type_defines = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            إذا (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        تور_type_defines("WEIGHT_TYPE", مبودي.weight_type);
        تور_type_defines("INPUT_TYPE", مبودي.شغرو_type);
        تور_type_defines("OUTPUT_TYPE", مبودي.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[مبودي]    = pipeline;
        أرجع conv2d_pipelines[مبودي];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ثابت ggml_webgpu_shader_lib_context & context, منطقي whcn) {
        ggml_webgpu_conv2d_dw_pipeline_مبودي مبودي = {};
        مبودي.weight_type                        = context.src0->type;
        مبودي.شغرو_type                         = context.src1->type;
        مبودي.output_type                        = context.dst->type;
        مبودي.whcn                               = whcn;

        auto it = conv2d_dw_pipelines.find(مبودي);
        إذا (it != conv2d_dw_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        auto تور_type_defines = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            إذا (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        تور_type_defines("WEIGHT_TYPE", مبودي.weight_type);
        تور_type_defines("INPUT_TYPE", مبودي.شغرو_type);
        تور_type_defines("OUTPUT_TYPE", مبودي.output_type);

        إذا (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[مبودي] = pipeline;
        أرجع conv2d_dw_pipelines[مبودي];
    }

    webgpu_pipeline get_im2col_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_مبودي مبودي = {};
        مبودي.شغرو_type                      = context.src1->type;
        مبودي.output_type                     = context.dst->type;

        auto it = im2col_pipelines.find(مبودي);
        إذا (it != im2col_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        auto تور_type_defines = [&](ثابت حرف * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            إذا (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } وإلا إذا (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } وإلا {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        تور_type_defines("INPUT_TYPE", مبودي.شغرو_type);
        تور_type_defines("OUTPUT_TYPE", مبودي.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[مبودي]    = pipeline;
        أرجع im2col_pipelines[مبودي];
    }

    webgpu_pipeline get_upscale_pipeline(ثابت ggml_webgpu_shader_lib_context & context) {
        ثابت uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ثابت uint32_t base_mode  = mode_flags & 0xFFu;
        ثابت منطقي     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_مبودي مبودي = {};
        مبودي.شغرو_type                       = context.src0->type;
        مبودي.output_type                      = context.dst->type;
        مبودي.base_mode                        = base_mode;
        مبودي.antialias                        = antialias;

        auto it = upscale_pipelines.find(مبودي);
        إذا (it != upscale_pipelines.end()) {
            أرجع it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        إذا (مبودي.شغرو_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } وإلا {
            variant += "_src_f32";
        }

        إذا (مبودي.output_type == GGML_TYPE_F16) {
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
        upscale_pipelines[مبودي]   = pipeline;
        أرجع upscale_pipelines[مبودي];
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
        أرجع { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
