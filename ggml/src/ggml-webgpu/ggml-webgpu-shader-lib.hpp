#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-wgsl-shaders.hpp"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "pre_wgsl.hpp"

#அந்தர்பாவ <webgpu/webgpu_cpp.h>

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <string>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <vector>

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
template <typename T> உள்வரி வெறுமை ggml_webgpu_hash_combine(size_t & seed, மாறிலி T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
உள்வரி uintptr_t ggml_webgpu_tensor_addr(மாறிலி ggml_tensor * tensor) {
    மாறிலி ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    திரும்பு (uintptr_t) base_tensor->data + tensor->view_offs;
}

உள்வரி தருக்க ggml_webgpu_tensor_equal(மாறிலி ggml_tensor * a, மாறிலி ggml_tensor * b) {
    திரும்பு a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

கட்டமைப்பு ggml_webgpu_shader_lib_context {
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
    தருக்க        supports_subgroups       = பொய்;
    தருக்க        supports_subgroup_matrix = பொய்;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_size        = 0;
    uint32_t    max_subgroup_size        = 0;
    தருக்க        supports_dot_product     = பொய்;
    std::string vendor;
};

கட்டமைப்பு webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<வெறுமை> context = nullptr;
};

கட்டமைப்பு ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 0;
    தருக்க     inplace = பொய்;
};

கட்டமைப்பு ggml_webgpu_get_rows_shader_decisions {
    uint32_t wg_size    = 0;
    தருக்க     vectorized = பொய்;
};

கட்டமைப்பு ggml_webgpu_இருமம்_shader_decisions {
    uint32_t wg_size     = 0;
    தருக்க     inplace     = பொய்;
    தருக்க     overlap     = பொய்;
    தருக்க     src_overlap = பொய்;
};

கட்டமைப்பு ggml_webgpu_glu_shader_decisions {
    uint32_t wg_size     = 0;
    தருக்க     src_overlap = பொய்;
};

கட்டமைப்பு ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<வெறுமை> decisions;
};

கட்டமைப்பு ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_size;
    uint32_t tokens_per_wg;
};

கட்டமைப்பு ggml_webgpu_ssm_scan_pipeline_சாவி {
    முழு     type;
    முழு     d_state;
    uint8_t xbc_overlap;
    தருக்க    a_overlap;
    தருக்க    ids_overlap;

    தருக்க operator==(மாறிலி ggml_webgpu_ssm_scan_pipeline_சாவி & other) மாறிலி {
        திரும்பு type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

கட்டமைப்பு ggml_webgpu_ssm_scan_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_ssm_scan_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.type);
        ggml_webgpu_hash_combine(seed, சாவி.d_state);
        ggml_webgpu_hash_combine(seed, சாவி.xbc_overlap);
        ggml_webgpu_hash_combine(seed, சாவி.a_overlap);
        ggml_webgpu_hash_combine(seed, சாவி.ids_overlap);
        திரும்பு seed;
    }
};

கட்டமைப்பு ggml_webgpu_ssm_scan_shader_decisions {
    uint32_t wg_size;
    uint32_t tokens_per_tile;
    uint8_t  xbc_overlap = 0;
    தருக்க     a_overlap   = பொய்;
    தருக்க     ids_overlap = பொய்;
};

/** Argsort **/

கட்டமைப்பு ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes;
    int32_t  order;
};

/** Set Rows **/

கட்டமைப்பு ggml_webgpu_set_rows_pipeline_சாவி {
    முழு src0_type;
    முழு dst_type;
    முழு vec4;
    முழு i64_idx;
    முழு pair_blocks;

    தருக்க operator==(மாறிலி ggml_webgpu_set_rows_pipeline_சாவி & other) மாறிலி {
        திரும்பு src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

கட்டமைப்பு ggml_webgpu_set_rows_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_set_rows_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.src0_type);
        ggml_webgpu_hash_combine(seed, சாவி.dst_type);
        ggml_webgpu_hash_combine(seed, சாவி.vec4);
        ggml_webgpu_hash_combine(seed, சாவி.i64_idx);
        ggml_webgpu_hash_combine(seed, சாவி.pair_blocks);
        திரும்பு seed;
    }
};

கட்டமைப்பு ggml_webgpu_set_rows_shader_decisions {
    தருக்க     vec4;
    தருக்க     i64_idx;
    தருக்க     pair_blocks;
    uint32_t wg_size;
};

/** Set **/

கட்டமைப்பு ggml_webgpu_set_pipeline_சாவி {
    ggml_type type;
    தருக்க      inplace;

    தருக்க operator==(மாறிலி ggml_webgpu_set_pipeline_சாவி & other) மாறிலி {
        திரும்பு type == other.type && inplace == other.inplace;
    }
};

கட்டமைப்பு ggml_webgpu_set_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_set_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.type);
        ggml_webgpu_hash_combine(seed, சாவி.inplace);
        திரும்பு seed;
    }
};

/** Get Rows **/

கட்டமைப்பு ggml_webgpu_get_rows_pipeline_சாவி {
    ggml_type src_type;
    முழு       vectorized;

    தருக்க operator==(மாறிலி ggml_webgpu_get_rows_pipeline_சாவி & other) மாறிலி {
        திரும்பு src_type == other.src_type && vectorized == other.vectorized;
    }
};

கட்டமைப்பு ggml_webgpu_get_rows_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_get_rows_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.src_type);
        ggml_webgpu_hash_combine(seed, சாவி.vectorized);
        திரும்பு seed;
    }
};

/** Row Norm **/

கட்டமைப்பு ggml_webgpu_row_norm_pipeline_சாவி {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    தருக்க      inplace;

    தருக்க operator==(மாறிலி ggml_webgpu_row_norm_pipeline_சாவி & other) மாறிலி {
        திரும்பு op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

கட்டமைப்பு ggml_webgpu_row_norm_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_row_norm_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.op);
        ggml_webgpu_hash_combine(seed, சாவி.src_type);
        ggml_webgpu_hash_combine(seed, சாவி.dst_type);
        ggml_webgpu_hash_combine(seed, சாவி.inplace);
        திரும்பு seed;
    }
};

/** RMS_NORM + MUL **/

கட்டமைப்பு ggml_webgpu_rms_norm_mul_pipeline_சாவி {
    தருக்க inplace;      // rn_src == dst
    தருக்க overlap;      // mul_src == dst
    தருக்க src_overlap;  // rn_src binding overlaps mul_src binding

    தருக்க operator==(மாறிலி ggml_webgpu_rms_norm_mul_pipeline_சாவி & other) மாறிலி {
        திரும்பு inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

கட்டமைப்பு ggml_webgpu_rms_norm_mul_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_rms_norm_mul_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.inplace);
        ggml_webgpu_hash_combine(seed, சாவி.overlap);
        ggml_webgpu_hash_combine(seed, சாவி.src_overlap);
        திரும்பு seed;
    }
};

கட்டமைப்பு ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    தருக்க     inplace     = பொய்;
    தருக்க     overlap     = பொய்;
    தருக்க     src_overlap = பொய்;
};

/** Pad **/
கட்டமைப்பு ggml_webgpu_pad_pipeline_சாவி {
    தருக்க circular;

    தருக்க operator==(மாறிலி ggml_webgpu_pad_pipeline_சாவி & other) மாறிலி { திரும்பு circular == other.circular; }
};

கட்டமைப்பு ggml_webgpu_pad_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_pad_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.circular);
        திரும்பு seed;
    }
};

/** Solve Tri **/
கட்டமைப்பு ggml_webgpu_solve_tri_pipeline_சாவி {
    முழு type;
    முழு n;
    முழு k;

    தருக்க operator==(மாறிலி ggml_webgpu_solve_tri_pipeline_சாவி & other) மாறிலி {
        திரும்பு type == other.type && n == other.n && k == other.k;
    }
};

கட்டமைப்பு ggml_webgpu_solve_tri_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_solve_tri_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.type);
        ggml_webgpu_hash_combine(seed, சாவி.n);
        ggml_webgpu_hash_combine(seed, சாவி.k);
        திரும்பு seed;
    }
};

/** SSM Conv **/
கட்டமைப்பு ggml_webgpu_ssm_conv_pipeline_சாவி {
    முழு type;
    முழு vectorized;

    தருக்க operator==(மாறிலி ggml_webgpu_ssm_conv_pipeline_சாவி & other) மாறிலி {
        திரும்பு type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
கட்டமைப்பு ggml_webgpu_conv2d_pipeline_சாவி {
    ggml_type weight_type;
    ggml_type உள்ளீடு_type;
    ggml_type output_type;

    தருக்க operator==(மாறிலி ggml_webgpu_conv2d_pipeline_சாவி & other) மாறிலி {
        திரும்பு weight_type == other.weight_type && உள்ளீடு_type == other.உள்ளீடு_type && output_type == other.output_type;
    }
};

கட்டமைப்பு ggml_webgpu_conv2d_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_conv2d_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.weight_type);
        ggml_webgpu_hash_combine(seed, சாவி.உள்ளீடு_type);
        ggml_webgpu_hash_combine(seed, சாவி.output_type);
        திரும்பு seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
கட்டமைப்பு ggml_webgpu_conv2d_dw_pipeline_சாவி {
    ggml_type weight_type;
    ggml_type உள்ளீடு_type;
    ggml_type output_type;
    தருக்க      whcn;

    தருக்க operator==(மாறிலி ggml_webgpu_conv2d_dw_pipeline_சாவி & other) மாறிலி {
        திரும்பு weight_type == other.weight_type && உள்ளீடு_type == other.உள்ளீடு_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

கட்டமைப்பு ggml_webgpu_conv2d_dw_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_conv2d_dw_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.weight_type);
        ggml_webgpu_hash_combine(seed, சாவி.உள்ளீடு_type);
        ggml_webgpu_hash_combine(seed, சாவி.output_type);
        ggml_webgpu_hash_combine(seed, சாவி.whcn);
        திரும்பு seed;
    }
};

/** Im2Col **/
கட்டமைப்பு ggml_webgpu_im2col_pipeline_சாவி {
    ggml_type உள்ளீடு_type;
    ggml_type output_type;

    தருக்க operator==(மாறிலி ggml_webgpu_im2col_pipeline_சாவி & other) மாறிலி {
        திரும்பு உள்ளீடு_type == other.உள்ளீடு_type && output_type == other.output_type;
    }
};

கட்டமைப்பு ggml_webgpu_im2col_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_im2col_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.உள்ளீடு_type);
        ggml_webgpu_hash_combine(seed, சாவி.output_type);
        திரும்பு seed;
    }
};

/** Gated Delta Net **/
கட்டமைப்பு ggml_webgpu_gated_delta_net_pipeline_சாவி {
    முழு  type;
    முழு  s_v;
    முழு  kda;
    தருக்க fused_cache;

    தருக்க operator==(மாறிலி ggml_webgpu_gated_delta_net_pipeline_சாவி & other) மாறிலி {
        திரும்பு type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

கட்டமைப்பு ggml_webgpu_gated_delta_net_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_gated_delta_net_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.type);
        ggml_webgpu_hash_combine(seed, சாவி.s_v);
        ggml_webgpu_hash_combine(seed, சாவி.kda);
        திரும்பு seed;
    }
};

கட்டமைப்பு ggml_webgpu_ssm_conv_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_ssm_conv_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.type);
        ggml_webgpu_hash_combine(seed, சாவி.vectorized);
        திரும்பு seed;
    }
};

/** Scale **/

கட்டமைப்பு ggml_webgpu_scale_pipeline_சாவி {
    முழு inplace;

    தருக்க operator==(மாறிலி ggml_webgpu_scale_pipeline_சாவி & other) மாறிலி { திரும்பு inplace == other.inplace; }
};

கட்டமைப்பு ggml_webgpu_scale_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_scale_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.inplace);
        திரும்பு seed;
    }
};

/** Upscale **/

கட்டமைப்பு ggml_webgpu_upscale_pipeline_சாவி {
    ggml_type உள்ளீடு_type;
    ggml_type output_type;
    uint32_t  base_mode;
    தருக்க      antialias;

    தருக்க operator==(மாறிலி ggml_webgpu_upscale_pipeline_சாவி & other) மாறிலி {
        திரும்பு உள்ளீடு_type == other.உள்ளீடு_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

கட்டமைப்பு ggml_webgpu_upscale_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_upscale_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.உள்ளீடு_type);
        ggml_webgpu_hash_combine(seed, சாவி.output_type);
        ggml_webgpu_hash_combine(seed, சாவி.base_mode);
        ggml_webgpu_hash_combine(seed, சாவி.antialias);
        திரும்பு seed;
    }
};

/** Concat **/

கட்டமைப்பு ggml_webgpu_concat_pipeline_சாவி {
    முழு  type;
    தருக்க src_overlap;

    தருக்க operator==(மாறிலி ggml_webgpu_concat_pipeline_சாவி & other) மாறிலி {
        திரும்பு type == other.type && src_overlap == other.src_overlap;
    }
};

கட்டமைப்பு ggml_webgpu_concat_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_concat_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.type);
        ggml_webgpu_hash_combine(seed, சாவி.src_overlap);
        திரும்பு seed;
    }
};

/** Repeat **/

கட்டமைப்பு ggml_webgpu_repeat_pipeline_சாவி {
    முழு type;

    தருக்க operator==(மாறிலி ggml_webgpu_repeat_pipeline_சாவி & other) மாறிலி { திரும்பு type == other.type; }
};

கட்டமைப்பு ggml_webgpu_repeat_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_repeat_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.type);
        திரும்பு seed;
    }
};

/** Binary **/

கட்டமைப்பு ggml_webgpu_இருமம்_pipeline_சாவி {
    முழு  type;
    முழு  op;
    தருக்க inplace;
    தருக்க overlap;
    தருக்க src_overlap;

    தருக்க operator==(மாறிலி ggml_webgpu_இருமம்_pipeline_சாவி & other) மாறிலி {
        திரும்பு type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

கட்டமைப்பு ggml_webgpu_இருமம்_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_இருமம்_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.type);
        ggml_webgpu_hash_combine(seed, சாவி.op);
        ggml_webgpu_hash_combine(seed, சாவி.inplace);
        ggml_webgpu_hash_combine(seed, சாவி.overlap);
        ggml_webgpu_hash_combine(seed, சாவி.src_overlap);
        திரும்பு seed;
    }
};

/* Add_Id */

கட்டமைப்பு ggml_webgpu_add_id_pipeline_சாவி {
    தருக்க inplace;

    தருக்க operator==(மாறிலி ggml_webgpu_add_id_pipeline_சாவி & other) மாறிலி { திரும்பு inplace == other.inplace; }
};

கட்டமைப்பு ggml_webgpu_add_id_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_add_id_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.inplace);
        திரும்பு seed;
    }
};

/** Unary **/

கட்டமைப்பு ggml_webgpu_unary_pipeline_சாவி {
    முழு           type;
    முழு           op;
    தருக்க          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    தருக்க          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    தருக்க operator==(மாறிலி ggml_webgpu_unary_pipeline_சாவி & other) மாறிலி {
        திரும்பு type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

கட்டமைப்பு ggml_webgpu_unary_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_unary_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.type);
        ggml_webgpu_hash_combine(seed, சாவி.op);
        ggml_webgpu_hash_combine(seed, சாவி.is_unary);
        ggml_webgpu_hash_combine(seed, சாவி.inplace);
        ggml_webgpu_hash_combine(seed, சாவி.ttype);
        திரும்பு seed;
    }
};

/** FlashAttention */

கட்டமைப்பு ggml_webgpu_flash_attn_common_pipeline_சாவி {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    தருக்க      k_direct;
    தருக்க      v_direct;
    தருக்க      kv_overlap;
    தருக்க      has_mask;
    தருக்க      has_sinks;
    தருக்க      uses_logit_softcap;

    தருக்க operator==(மாறிலி ggml_webgpu_flash_attn_common_pipeline_சாவி & other) மாறிலி {
        திரும்பு q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

உள்வரி வெறுமை ggml_webgpu_flash_attn_hash_common_pipeline_சாவி(size_t &                                           seed,
                                                            மாறிலி ggml_webgpu_flash_attn_common_pipeline_சாவி & சாவி) {
    ggml_webgpu_hash_combine(seed, சாவி.q_type);
    ggml_webgpu_hash_combine(seed, சாவி.k_type);
    ggml_webgpu_hash_combine(seed, சாவி.v_type);
    ggml_webgpu_hash_combine(seed, சாவி.dst_type);
    ggml_webgpu_hash_combine(seed, சாவி.head_dim_qk);
    ggml_webgpu_hash_combine(seed, சாவி.head_dim_v);
    ggml_webgpu_hash_combine(seed, சாவி.k_direct);
    ggml_webgpu_hash_combine(seed, சாவி.v_direct);
    ggml_webgpu_hash_combine(seed, சாவி.kv_overlap);
    ggml_webgpu_hash_combine(seed, சாவி.has_mask);
    ggml_webgpu_hash_combine(seed, சாவி.has_sinks);
    ggml_webgpu_hash_combine(seed, சாவி.uses_logit_softcap);
}

கட்டமைப்பு ggml_webgpu_flash_attn_vec_pipeline_சாவி {
    ggml_webgpu_flash_attn_common_pipeline_சாவி common;

    தருக்க operator==(மாறிலி ggml_webgpu_flash_attn_vec_pipeline_சாவி & other) மாறிலி { திரும்பு common == other.common; }
};

கட்டமைப்பு ggml_webgpu_flash_attn_vec_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_flash_attn_vec_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_சாவி(seed, சாவி.common);
        திரும்பு seed;
    }
};

கட்டமைப்பு ggml_webgpu_flash_attn_pipeline_சாவி {
    ggml_webgpu_flash_attn_common_pipeline_சாவி common;
    தருக்க                                       use_sg_matrix;

    தருக்க operator==(மாறிலி ggml_webgpu_flash_attn_pipeline_சாவி & other) மாறிலி {
        திரும்பு common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

கட்டமைப்பு ggml_webgpu_flash_attn_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_flash_attn_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_சாவி(seed, சாவி.common);
        ggml_webgpu_hash_combine(seed, சாவி.use_sg_matrix);
        திரும்பு seed;
    }
};

கட்டமைப்பு ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

கட்டமைப்பு ggml_webgpu_flash_attn_decisions {
    தருக்க     use_sg_matrix = பொய்;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_size       = 0;
};

உள்வரி constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
உள்வரி constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

உள்வரி size_t ggml_webgpu_flash_attn_tensor_offset(மாறிலி ggml_tensor * tensor) {
    constexpr uintptr_t ptr_base_addr = 0x1000u;
    மாறிலி ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
    திரும்பு reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
}

உள்வரி தருக்க ggml_webgpu_flash_attn_float_vec4_aligned(மாறிலி ggml_tensor * K, size_t storage_offset_alignment) {
    மாறிலி uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_size(K->type));
    திரும்பு offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

உள்வரி தருக்க ggml_webgpu_flash_attn_float_vec4_aligned(மாறிலி ggml_tensor * K,
                                                      மாறிலி ggml_tensor * V,
                                                      size_t              storage_offset_alignment) {
    திரும்பு ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

உள்வரி தருக்க ggml_webgpu_flash_attn_k_direct(மாறிலி ggml_tensor * Q, மாறிலி ggml_tensor * K, uint32_t kv_direct_align) {
    திரும்பு (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

உள்வரி தருக்க ggml_webgpu_flash_attn_v_direct(மாறிலி ggml_tensor * Q, மாறிலி ggml_tensor * V, uint32_t kv_direct_align) {
    திரும்பு ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

உள்வரி ggml_webgpu_flash_attn_common_pipeline_சாவி ggml_webgpu_flash_attn_make_common_pipeline_சாவி(
    மாறிலி ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    தருக்க                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_சாவி சாவி = {};
    சாவி.q_type                                     = context.src0->type;
    சாவி.k_type                                     = context.src1->type;
    சாவி.v_type                                     = context.src2->type;
    சாவி.dst_type                                   = context.dst->type;
    சாவி.head_dim_qk                                = (uint32_t) context.src0->ne[0];
    சாவி.head_dim_v                                 = (uint32_t) context.src2->ne[0];
    சாவி.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    சாவி.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    சாவி.kv_overlap         = kv_overlap;
    சாவி.has_mask           = context.src3 != nullptr;
    சாவி.has_sinks          = context.src4 != nullptr;
    சாவி.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    திரும்பு சாவி;
}

உள்வரி std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    மாறிலி ggml_webgpu_flash_attn_common_pipeline_சாவி & சாவி,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_size) {
    std::vector<std::string> defines;

    தேர்வு (சாவி.k_type) {
        நிலை GGML_TYPE_F32:
            defines.push_back("K_F32");
            நிறுத்து;
        நிலை GGML_TYPE_F16:
            defines.push_back("K_F16");
            நிறுத்து;
        நிலை GGML_TYPE_Q4_0:
            defines.push_back("K_Q4_0");
            நிறுத்து;
        நிலை GGML_TYPE_Q8_0:
            defines.push_back("K_Q8_0");
            நிறுத்து;
        இயல்பு:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(சாவி.k_type);

    தேர்வு (சாவி.v_type) {
        நிலை GGML_TYPE_F32:
            defines.push_back("V_F32");
            நிறுத்து;
        நிலை GGML_TYPE_F16:
            defines.push_back("V_F16");
            நிறுத்து;
        நிலை GGML_TYPE_Q4_0:
            defines.push_back("V_Q4_0");
            நிறுத்து;
        நிலை GGML_TYPE_Q8_0:
            defines.push_back("V_Q8_0");
            நிறுத்து;
        இயல்பு:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(சாவி.v_type);

    தேர்வு (சாவி.q_type) {
        நிலை GGML_TYPE_F32:
            defines.push_back("Q_F32");
            நிறுத்து;
        நிலை GGML_TYPE_F16:
            defines.push_back("Q_F16");
            நிறுத்து;
        இயல்பு:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(சாவி.q_type);

    தேர்வு (சாவி.dst_type) {
        நிலை GGML_TYPE_F32:
            defines.push_back("DST_F32");
            நிறுத்து;
        நிலை GGML_TYPE_F16:
            defines.push_back("DST_F16");
            நிறுத்து;
        இயல்பு:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(சாவி.dst_type);

    எனில் (சாவி.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    எனில் (சாவி.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    எனில் (சாவி.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    எனில் (சாவி.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    எனில் (சாவி.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    எனில் (சாவி.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(சாவி.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(சாவி.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(சாவி.head_dim_v));
    variant += std::string("_hsv") + std::to_string(சாவி.head_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    எனில் (ggml_is_quantized(சாவி.k_type) || ggml_is_quantized(சாவி.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        எனில் (ggml_is_quantized(சாவி.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        எனில் (ggml_is_quantized(சாவி.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    திரும்பு defines;
}

கட்டமைப்பு ggml_webgpu_flash_attn_vec_reduce_pipeline_சாவி {
    uint32_t  head_dim_v;
    uint32_t  wg_size;
    ggml_type dst_type;
};

கட்டமைப்பு ggml_webgpu_flash_attn_vec_reduce_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_flash_attn_vec_reduce_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.head_dim_v);
        ggml_webgpu_hash_combine(seed, சாவி.wg_size);
        ggml_webgpu_hash_combine(seed, சாவி.dst_type);
        திரும்பு seed;
    }
};

உள்வரி தருக்க operator==(மாறிலி ggml_webgpu_flash_attn_vec_reduce_pipeline_சாவி & lhs,
                       மாறிலி ggml_webgpu_flash_attn_vec_reduce_pipeline_சாவி & rhs) {
    திரும்பு lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size && lhs.dst_type == rhs.dst_type;
}

கட்டமைப்பு ggml_webgpu_flash_attn_blk_pipeline_சாவி {
    uint32_t kv_tile;

    தருக்க operator==(மாறிலி ggml_webgpu_flash_attn_blk_pipeline_சாவி & other) மாறிலி { திரும்பு kv_tile == other.kv_tile; }
};

கட்டமைப்பு ggml_webgpu_flash_attn_blk_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_flash_attn_blk_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.kv_tile);
        திரும்பு seed;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
உள்வரி size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  தருக்க     has_mask,
                                                  தருக்க     kv_direct) {
    மாறிலி uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * head_dim_qk;        // q_shmem
    எனில் (!kv_direct) {
        f32_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f32_elems += q_tile * head_dim_v;         // o_shmem
    எனில் (has_mask) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    திரும்பு f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

உள்வரி uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   limit_bytes,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t head_dim_qk,
                                                   uint32_t head_dim_v,
                                                   தருக்க     has_mask,
                                                   தருக்க     kv_direct) {
    மாறிலி size_t base_q_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 0, head_dim_qk, head_dim_v, has_mask, kv_direct);
    எனில் (limit_bytes <= base_q_bytes) {
        திரும்பு 0;
    }
    மாறிலி size_t one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, head_dim_qk, head_dim_v, has_mask, kv_direct);
    மாறிலி size_t bytes_per_kv = one_kv_bytes - base_q_bytes;
    எனில் (bytes_per_kv == 0) {
        திரும்பு 0;
    }
    மாறிலி size_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    திரும்பு (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

உள்வரி uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_limit_bytes,
                                                       uint32_t head_dim_qk,
                                                       uint32_t head_dim_v,
                                                       தருக்க     has_mask,
                                                       தருக்க     kv_direct) {
    மாறிலி uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_limit_bytes, 1u, 1u, head_dim_qk, head_dim_v, has_mask, kv_direct);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    எனில் (kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        போது (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    திரும்பு kv_tile;
}

உள்வரி தருக்க ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(தருக்க                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                மாறிலி ggml_tensor * Q,
                                                                மாறிலி ggml_tensor * V) {
    திரும்பு supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

கட்டமைப்பு ggml_webgpu_mul_mat_vec_pipeline_சாவி {
    ggml_type src0_type;
    ggml_type src1_type;
    முழு       vectorized;
    uint32_t  இலக்கம்_cols;
    தருக்க      use_mmvq;
    தருக்க      src_overlap;

    தருக்க operator==(மாறிலி ggml_webgpu_mul_mat_vec_pipeline_சாவி & other) மாறிலி {
        திரும்பு src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               இலக்கம்_cols == other.இலக்கம்_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

கட்டமைப்பு ggml_webgpu_mul_mat_vec_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_mul_mat_vec_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.src0_type);
        ggml_webgpu_hash_combine(seed, சாவி.src1_type);
        ggml_webgpu_hash_combine(seed, சாவி.vectorized);
        ggml_webgpu_hash_combine(seed, சாவி.இலக்கம்_cols);
        ggml_webgpu_hash_combine(seed, சாவி.use_mmvq);
        ggml_webgpu_hash_combine(seed, சாவி.src_overlap);
        திரும்பு seed;
    }
};

கட்டமைப்பு ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
    தருக்க     src_overlap = பொய்;
};

கட்டமைப்பு ggml_webgpu_quantize_q8_pipeline_சாவி {
    ggml_type src0_type;

    தருக்க operator==(மாறிலி ggml_webgpu_quantize_q8_pipeline_சாவி & other) மாறிலி { திரும்பு src0_type == other.src0_type; }
};

கட்டமைப்பு ggml_webgpu_quantize_q8_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_quantize_q8_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.src0_type);
        திரும்பு seed;
    }
};

கட்டமைப்பு ggml_webgpu_mul_mat_pipeline_சாவி {
    ggml_type src0_type;
    ggml_type src1_type;
    முழு       vectorized;
    முழு       use_subgroup_matrix;
    தருக்க      src_overlap;

    தருக்க operator==(மாறிலி ggml_webgpu_mul_mat_pipeline_சாவி & other) மாறிலி {
        திரும்பு src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

கட்டமைப்பு ggml_webgpu_mul_mat_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_mul_mat_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.src0_type);
        ggml_webgpu_hash_combine(seed, சாவி.src1_type);
        ggml_webgpu_hash_combine(seed, சாவி.vectorized);
        ggml_webgpu_hash_combine(seed, சாவி.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, சாவி.src_overlap);
        திரும்பு seed;
    }
};

கட்டமைப்பு ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    முழு      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_size;
    தருக்க     src_overlap = பொய்;
};

/** MUL_MAT_ID **/

கட்டமைப்பு ggml_webgpu_mul_mat_id_gather_pipeline_சாவி {
    uint32_t n_experts;

    தருக்க operator==(மாறிலி ggml_webgpu_mul_mat_id_gather_pipeline_சாவி & other) மாறிலி {
        திரும்பு n_experts == other.n_experts;
    }
};

கட்டமைப்பு ggml_webgpu_mul_mat_id_gather_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_mul_mat_id_gather_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.n_experts);
        திரும்பு seed;
    }
};

கட்டமைப்பு ggml_webgpu_mul_mat_id_pipeline_சாவி {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  இலக்கம்_cols;
    முழு       vectorized;

    தருக்க operator==(மாறிலி ggml_webgpu_mul_mat_id_pipeline_சாவி & other) மாறிலி {
        திரும்பு src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               இலக்கம்_cols == other.இலக்கம்_cols && vectorized == other.vectorized;
    }
};

கட்டமைப்பு ggml_webgpu_mul_mat_id_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_mul_mat_id_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.src0_type);
        ggml_webgpu_hash_combine(seed, சாவி.src1_type);
        ggml_webgpu_hash_combine(seed, சாவி.n_experts);
        ggml_webgpu_hash_combine(seed, சாவி.இலக்கம்_cols);
        ggml_webgpu_hash_combine(seed, சாவி.vectorized);
        திரும்பு seed;
    }
};

/** Cpy **/

கட்டமைப்பு ggml_webgpu_cpy_pipeline_சாவி {
    ggml_type src_type;
    ggml_type dst_type;

    தருக்க operator==(மாறிலி ggml_webgpu_cpy_pipeline_சாவி & other) மாறிலி {
        திரும்பு src_type == other.src_type && dst_type == other.dst_type;
    }
};

கட்டமைப்பு ggml_webgpu_cpy_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_cpy_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.src_type);
        ggml_webgpu_hash_combine(seed, சாவி.dst_type);
        திரும்பு seed;
    }
};

/** Glu **/

கட்டமைப்பு ggml_webgpu_glu_pipeline_சாவி {
    ggml_glu_op glu_op;
    ggml_type   type;
    தருக்க        split;
    தருக்க        src_overlap;

    தருக்க operator==(மாறிலி ggml_webgpu_glu_pipeline_சாவி & other) மாறிலி {
        திரும்பு glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

கட்டமைப்பு ggml_webgpu_glu_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_glu_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.glu_op);
        ggml_webgpu_hash_combine(seed, சாவி.type);
        ggml_webgpu_hash_combine(seed, சாவி.split);
        ggml_webgpu_hash_combine(seed, சாவி.src_overlap);
        திரும்பு seed;
    }
};

/** Rope **/

கட்டமைப்பு ggml_webgpu_rope_pipeline_சாவி {
    ggml_type type;
    தருக்க      inplace;
    தருக்க      has_ff;

    தருக்க operator==(மாறிலி ggml_webgpu_rope_pipeline_சாவி & other) மாறிலி {
        திரும்பு type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

கட்டமைப்பு ggml_webgpu_rope_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_rope_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.type);
        ggml_webgpu_hash_combine(seed, சாவி.inplace);
        ggml_webgpu_hash_combine(seed, சாவி.has_ff);
        திரும்பு seed;
    }
};

/** SoftMax **/

கட்டமைப்பு ggml_webgpu_soft_max_pipeline_சாவி {
    ggml_type mask_type;
    தருக்க      has_mask;
    தருக்க      has_sink;
    தருக்க      inplace;

    தருக்க operator==(மாறிலி ggml_webgpu_soft_max_pipeline_சாவி & other) மாறிலி {
        திரும்பு mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

கட்டமைப்பு ggml_webgpu_soft_max_pipeline_சாவி_hash {
    size_t operator()(மாறிலி ggml_webgpu_soft_max_pipeline_சாவி & சாவி) மாறிலி {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, சாவி.mask_type);
        ggml_webgpu_hash_combine(seed, சாவி.has_mask);
        ggml_webgpu_hash_combine(seed, சாவி.has_sink);
        ggml_webgpu_hash_combine(seed, சாவி.inplace);
        திரும்பு seed;
    }
};

/** MMVQ **/

உள்வரி தருக்க ggml_webgpu_can_use_mmvq(மாறிலி ggml_tensor * src0,
                                     மாறிலி ggml_tensor * src1,
                                     தருக்க                supports_dot_product,
                                     மாறிலி std::string & vendor) {
    எனில் (src1->ne[1] <= 4) {
        தருக்க supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        எனில் (supports_dp4a && supports_dot_product) {
            தேர்வு (src1->type) {
                நிலை GGML_TYPE_F32:
                    தேர்வு (src0->type) {
                        நிலை GGML_TYPE_Q1_0:
                        நிலை GGML_TYPE_Q4_0:
                        நிலை GGML_TYPE_Q4_1:
                        நிலை GGML_TYPE_Q5_0:
                        நிலை GGML_TYPE_Q5_1:
                        நிலை GGML_TYPE_Q8_0:
                        நிலை GGML_TYPE_MXFP4:
                        நிலை GGML_TYPE_Q2_K:
                        நிலை GGML_TYPE_Q3_K:
                        நிலை GGML_TYPE_Q4_K:
                        நிலை GGML_TYPE_Q5_K:
                        நிலை GGML_TYPE_Q6_K:
                            திரும்பு src0->ne[0] % 4 == 0;
                        இயல்பு:
                            நிறுத்து;
                    }
                    நிறுத்து;
                இயல்பு:
                    நிறுத்து;
            }
        }
    }
    திரும்பு பொய்;
}

வகுப்பு ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<முழு, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<முழு, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<முழு, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<முழு, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<முழு, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_சாவி_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_சாவி_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_unary_pipeline_சாவி_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_scale_pipeline_சாவி_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_சாவி_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_சாவி_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_சாவி_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_சாவி,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_சாவி_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_pad_pipeline_சாவி_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_இருமம்_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_இருமம்_pipeline_சாவி_hash>
        இருமம்_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_add_id_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_add_id_pipeline_சாவி_hash>
        add_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_concat_pipeline_சாவி_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_repeat_pipeline_சாவி_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_சாவி,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_சாவி_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_சாவி_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_சாவி,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_சாவி_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_சாவி,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_சாவி_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_சாவி_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_சாவி_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_சாவி_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_சாவி,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_சாவி_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_சாவி_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_சாவி_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_சாவி_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_set_pipeline_சாவி_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_cpy_pipeline_சாவி_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_glu_pipeline_சாவி_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_rope_pipeline_சாவி_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_சாவி_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_சாவி_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_சாவி_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_im2col_pipeline_சாவி_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_சாவி,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_சாவி_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_சாவி, webgpu_pipeline, ggml_webgpu_upscale_pipeline_சாவி_hash>
        upscale_pipelines;

  பொது:
    ggml_webgpu_shader_lib(wgpu::Device device) { இது->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        auto it = sum_rows_pipelines.find(1);
        எனில் (it != sum_rows_pipelines.end()) {
            திரும்பு it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        திரும்பு sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_சாவி சாவி = {};
        சாவி.op                                = context.dst->op;
        சாவி.src_type                          = context.src0->type;
        சாவி.dst_type                          = context.dst->type;
        சாவி.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = row_norm_pipelines.find(சாவி);
        எனில் (it != row_norm_pipelines.end()) {
            திரும்பு it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        தேர்வு (சாவி.op) {
            நிலை GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                நிறுத்து;
            நிலை GGML_OP_NORM:
                defines.push_back("NORM");
                variant = "norm";
                நிறுத்து;
            நிலை GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        எனில் (சாவி.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        எனில் (சாவி.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } இல்லையெனில் எனில் (சாவி.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        எனில் (சாவி.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } இல்லையெனில் எனில் (சாவி.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        மாறிலி uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = சாவி.inplace;
        row_norm_pipelines[சாவி]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[சாவி].context = decisions;
        திரும்பு row_norm_pipelines[சாவி];
    }

    webgpu_pipeline get_argmax_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        தருக்க vec4 = context.src0->ne[0] % 4 == 0;

        auto it = argmax_pipelines.find(vec4);
        எனில் (it != argmax_pipelines.end()) {
            திரும்பு it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        எனில் (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        auto processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        திரும்பு argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        மாறிலி தருக்க                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_சாவி சாவி       = {};
        சாவி.src0_type                               = context.src0->type;
        சாவி.dst_type                                = context.dst->type;
        சாவி.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        சாவி.i64_idx     = context.src1->type == GGML_TYPE_I64;
        சாவி.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_size(context.dst->type)) % 2 == 0);

        auto it = set_rows_pipelines.find(சாவி);
        எனில் (it != set_rows_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        தேர்வு (context.dst->type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                நிறுத்து;
            நிலை GGML_TYPE_Q8_0:
                defines.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                நிறுத்து;
            நிலை GGML_TYPE_Q4_0:
                defines.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        எனில் (context.src0->type == GGML_TYPE_F16) {
            defines.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        எனில் (சாவி.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        எனில் (சாவி.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        எனில் (சாவி.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        மாறிலி auto & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        auto         processed          = preprocessor.preprocess(shader_source, defines);
        auto         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = சாவி.vec4;
        decisions->i64_idx              = சாவி.i64_idx;
        decisions->pair_blocks          = சாவி.pair_blocks;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[சாவி]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[சாவி].context = decisions;
        திரும்பு set_rows_pipelines[சாவி];
    }

    webgpu_pipeline get_set_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_சாவி சாவி = {};
        சாவி.type                         = context.dst->type;
        சாவி.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = set_pipelines.find(சாவி);
        எனில் (it != set_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for set shader");
        }

        எனில் (சாவி.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_set, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = சாவி.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[சாவி]       = pipeline;
        திரும்பு set_pipelines[சாவி];
    }

    webgpu_pipeline get_cumsum_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        auto it = cumsum_pipelines.find(1);
        எனில் (it != cumsum_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        திரும்பு cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        தருக்க          is_உச்சி_k = context.dst->op == GGML_OP_உச்சி_K;
        // ascending order is 0, descending order is 1
        மாறிலி int32_t order =
            is_உச்சி_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_pipelines.find(order);
        எனில் (it != argsort_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        போது (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        auto decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        திரும்பு argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        தருக்க          is_உச்சி_k = context.dst->op == GGML_OP_உச்சி_K;
        // ascending order is 0, descending order is 1
        மாறிலி int32_t order =
            is_உச்சி_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_merge_pipelines.find(order);
        எனில் (it != argsort_merge_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        திரும்பு argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, தருக்க vec4_aligned) {
        மாறிலி தருக்க vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_சாவி சாவி = {};
        சாவி.src_type                          = context.src0->type;
        சாவி.vectorized                        = (முழு) vectorized;

        auto it = get_rows_pipelines.find(சாவி);
        எனில் (it != get_rows_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        மாறிலி கட்டமைப்பு ggml_type_traits * type_traits = ggml_get_type_traits(சாவி.src_type);
        மாறிலி எழுத்து *                    type_str    = type_traits->type_name;

        தேர்வு (சாவி.src_type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                எனில் (சாவி.vectorized) {
                    defines.push_back("F32_VEC");
                    defines.push_back("SRC_TYPE=vec4<f32>");
                    defines.push_back("DST_TYPE=vec4<f32>");
                    defines.push_back("BLOCK_SIZE=4u");
                } இல்லையெனில் {
                    defines.push_back("F32");
                    defines.push_back("SRC_TYPE=f32");
                    defines.push_back("DST_TYPE=f32");
                    defines.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                நிறுத்து;
            நிலை GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                நிறுத்து;
            நிலை GGML_TYPE_BF16:
                defines.push_back("BF16");
                defines.push_back("SRC_TYPE=u32");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                நிறுத்து;
            இயல்பு:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    தேர்வு (சாவி.src_type) {
                        நிலை GGML_TYPE_Q1_0:
                        நிலை GGML_TYPE_Q2_0:
                        நிலை GGML_TYPE_Q4_0:
                        நிலை GGML_TYPE_Q5_0:
                        நிலை GGML_TYPE_Q8_0:
                        நிலை GGML_TYPE_Q3_K:
                        நிலை GGML_TYPE_Q6_K:
                        நிலை GGML_TYPE_IQ2_XXS:
                        நிலை GGML_TYPE_IQ2_XS:
                        நிலை GGML_TYPE_IQ2_S:
                        நிலை GGML_TYPE_IQ3_XXS:
                        நிலை GGML_TYPE_IQ3_S:
                        நிலை GGML_TYPE_IQ1_S:
                        நிலை GGML_TYPE_IQ4_NL:
                        நிலை GGML_TYPE_MXFP4:
                        நிலை GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                நிறுத்து;
                            }
                        இயல்பு:
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

                    எனில் (சாவி.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } இல்லையெனில் எனில் ((சாவி.src_type >= GGML_TYPE_Q4_0 && சாவி.src_type <= GGML_TYPE_Q8_1) ||
                               சாவி.src_type == GGML_TYPE_IQ4_NL || சாவி.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } இல்லையெனில் எனில் (சாவி.src_type == GGML_TYPE_Q2_0 || சாவி.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } இல்லையெனில் எனில் (சாவி.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } இல்லையெனில் {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    நிறுத்து;
                }
        }

        எனில் (சாவி.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        auto decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[சாவி]  = pipeline;
        திரும்பு get_rows_pipelines[சாவி];
    }

    webgpu_pipeline get_scale_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_சாவி சாவி = {};
        சாவி.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = scale_pipelines.find(சாவி);
        எனில் (it != scale_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        எனில் (சாவி.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_scale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = சாவி.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[சாவி]     = pipeline;
        திரும்பு scale_pipelines[சாவி];
    }

    webgpu_pipeline get_solve_tri_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_சாவி சாவி = {};
        சாவி.type                               = context.dst->type;
        சாவி.n                                  = (முழு) context.src0->ne[0];
        சாவி.k                                  = (முழு) context.src1->ne[0];

        auto it = solve_tri_pipelines.find(சாவி);
        எனில் (it != solve_tri_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                variant += "_f32";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        மாறிலி uint32_t wg_size       = std::min((uint32_t) சாவி.n, context.max_wg_size);
        மாறிலி uint32_t k_tile        = wg_size;
        மாறிலி uint32_t bytes_per_row = ((uint32_t) சாவி.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        மாறிலி uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(சாவி.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        auto processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[சாவி] = pipeline;
        திரும்பு solve_tri_pipelines[சாவி];
    }

    webgpu_pipeline get_ssm_conv_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_சாவி சாவி = {};
        சாவி.type                              = context.dst->type;
        சாவி.vectorized                        = context.src1->ne[0] == 4;

        auto it = ssm_conv_pipelines.find(சாவி);
        எனில் (it != ssm_conv_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                variant += "_f32";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        எனில் (சாவி.vectorized) {
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
        ssm_conv_pipelines[சாவி]  = pipeline;
        திரும்பு ssm_conv_pipelines[சாவி];
    }

    webgpu_pipeline get_ssm_scan_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_சாவி சாவி = {};
        சாவி.type                              = context.dst->type;
        சாவி.d_state                           = (முழு) context.src0->ne[0];
        சாவி.xbc_overlap                       = xbc_overlap;

        auto it = ssm_scan_pipelines.find(சாவி);
        எனில் (it != ssm_scan_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                variant += "_f32";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        மாறிலி uint32_t wg_size = (uint32_t) சாவி.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        எனில் (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } இல்லையெனில் {
            variant += "_wg_reduce";
        }

        எனில் (சாவி.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } இல்லையெனில் எனில் (சாவி.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } இல்லையெனில் எனில் (சாவி.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(சாவி.d_state);

        auto processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        auto decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = சாவி.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[சாவி]    = pipeline;
        திரும்பு ssm_scan_pipelines[சாவி];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_சாவி சாவி = {};
        சாவி.type                                     = context.dst->type;
        சாவி.s_v                                      = (முழு) context.src2->ne[0];
        சாவி.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        சாவி.fused_cache                              = context.dst_fuse != nullptr;

        auto it = gated_delta_net_pipelines.find(சாவி);
        எனில் (it != gated_delta_net_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                variant += "_f32";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        எனில் (சாவி.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        எனில் (சாவி.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(சாவி.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(சாவி.s_v) + "u");

        auto            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[சாவி] = pipeline;
        திரும்பு gated_delta_net_pipelines[சாவி];
    }

    webgpu_pipeline get_pad_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_சாவி சாவி = {};
        சாவி.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        auto it = pad_pipelines.find(சாவி);
        எனில் (it != pad_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        எனில் (சாவி.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_pad, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[சாவி]       = pipeline;
        திரும்பு pad_pipelines[சாவி];
    }

    webgpu_pipeline get_quantize_q8_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_சாவி சாவி = {};
        சாவி.src0_type                            = context.src0->type;

        auto it = quantize_q8_pipelines.find(சாவி);
        எனில் (it != quantize_q8_pipelines.end()) {
            திரும்பு it->second;
        }
        மாறிலி எழுத்து *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> defines;
        std::string              variant = "quantize_q8";

        uint32_t wg_size = WEBGPU_MUL_MAT_VEC_WG_SIZE;

        defines.push_back("SRC1_INNER_TYPE=f32");
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        மாறிலி கட்டமைப்பு ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
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
        quantize_q8_pipelines[சாவி] = pipeline;
        திரும்பு quantize_q8_pipelines[சாவி];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, தருக்க src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_சாவி சாவி = {};
        சாவி.src0_type                            = context.src0->type;
        சாவி.src1_type                            = context.src1->type;
        சாவி.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        சாவி.இலக்கம்_cols   = context.dst->ne[1];
        சாவி.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
        சாவி.src_overlap = src_overlap;

        auto it = mul_mat_vec_pipelines.find(சாவி);
        எனில் (it != mul_mat_vec_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        மாறிலி எழுத்து *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        தேர்வு (context.src0->type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                நிறுத்து;
            நிலை GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                நிறுத்து;
            இயல்பு:
                {
                    // Quantized types: use helpers but accumulate in f16
                    மாறிலி கட்டமைப்பு ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    தேர்வு (context.src0->type) {
                        நிலை GGML_TYPE_Q4_0:
                        நிலை GGML_TYPE_Q4_1:
                        நிலை GGML_TYPE_Q5_0:
                        நிலை GGML_TYPE_Q5_1:
                        நிலை GGML_TYPE_Q8_0:
                            எனில் (சாவி.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            நிறுத்து;
                        நிலை GGML_TYPE_Q1_0:
                        நிலை GGML_TYPE_Q2_K:
                        நிலை GGML_TYPE_Q3_K:
                        நிலை GGML_TYPE_Q4_K:
                        நிலை GGML_TYPE_Q5_K:
                        நிலை GGML_TYPE_Q6_K:
                            எனில் (சாவி.use_mmvq) {
                                defines.push_back("K_QUANTS_HANDLING");
                            }
                            நிறுத்து;
                        நிலை GGML_TYPE_IQ1_S:
                        நிலை GGML_TYPE_IQ1_M:
                        நிலை GGML_TYPE_IQ2_S:
                        நிலை GGML_TYPE_IQ3_S:
                        நிலை GGML_TYPE_IQ4_NL:
                        நிலை GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            நிறுத்து;
                        நிலை GGML_TYPE_IQ2_XXS:
                        நிலை GGML_TYPE_IQ2_XS:
                        நிலை GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            நிறுத்து;
                        நிலை GGML_TYPE_MXFP4:
                            defines.push_back(type_upper + "_LUT");
                            எனில் (சாவி.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            நிறுத்து;
                        நிலை GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            நிறுத்து;
                        இயல்பு:
                            நிறுத்து;
                    }
                    நிறுத்து;
                }
        }

        // src1 type (vector)
        தேர்வு (context.src1->type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(சாவி.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        எனில் (சாவி.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } இல்லையெனில் எனில் (சாவி.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } இல்லையெனில் எனில் (சாவி.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        எனில் (சாவி.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        எனில் (சாவி.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        எனில் (சாவி.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(சாவி.இலக்கம்_cols));

        auto processed            = preprocessor.preprocess(shader_src, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = சாவி.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[சாவி] = pipeline;
        திரும்பு mul_mat_vec_pipelines[சாவி];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, தருக்க src_overlap) {
        ggml_webgpu_mul_mat_pipeline_சாவி சாவி = {};
        சாவி.src0_type                        = context.src0->type;
        சாவி.src1_type                        = context.src1->type;
        சாவி.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        சாவி.use_subgroup_matrix = context.supports_subgroup_matrix;
        சாவி.src_overlap         = src_overlap;

        auto it = mul_mat_fast_pipelines.find(சாவி);
        எனில் (it != mul_mat_fast_pipelines.end()) {
            திரும்பு it->second;
        }

        மாறிலி எழுத்து * shader_src = சாவி.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = சாவி.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        தேர்வு (context.src1->type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        மாறிலி கட்டமைப்பு ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        மாறிலி எழுத்து *                    src0_name   = src0_traits->type_name;

        தேர்வு (context.src0->type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("TYPE_F32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("TYPE_F16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                நிறுத்து;
            நிலை GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("TYPE_BF16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                நிறுத்து;
            இயல்பு:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    தேர்வு (context.src0->type) {
                        நிலை GGML_TYPE_IQ1_S:
                        நிலை GGML_TYPE_IQ1_M:
                        நிலை GGML_TYPE_IQ4_NL:
                        நிலை GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            நிறுத்து;
                        நிலை GGML_TYPE_IQ2_XXS:
                        நிலை GGML_TYPE_IQ2_XS:
                        நிலை GGML_TYPE_IQ2_S:
                        நிலை GGML_TYPE_IQ3_XXS:
                        நிலை GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            நிறுத்து;
                        நிலை GGML_TYPE_MXFP4:
                        நிலை GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            நிறுத்து;
                        இயல்பு:
                            நிறுத்து;
                    }

                    variant += std::string("_") + src0_name;
                    நிறுத்து;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(சாவி.vectorized ? "VEC" : "SCALAR");

        மாறிலி தருக்க is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        எனில் (சாவி.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } இல்லையெனில் {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        எனில் (சாவி.use_subgroup_matrix) {
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
        எனில் (சாவி.vectorized) {
            variant += "_vectorized";
        }

        எனில் (சாவி.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        எனில் (!சாவி.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = சாவி.use_subgroup_matrix;
        எனில் (சாவி.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_size           = context.max_subgroup_size;
        } இல்லையெனில் {
            decisions->wg_size_m       = WEBGPU_MUL_MAT_WG_SIZE_M;
            decisions->wg_size_n       = WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->wg_size         = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->mul_mat_wg_size = WEBGPU_MUL_MAT_WG_SIZE;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[சாவி] = pipeline;
        திரும்பு mul_mat_fast_pipelines[சாவி];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_சாவி சாவி = {};
        சாவி.n_experts                                  = (uint32_t) context.src0->ne[2];

        auto it = mul_mat_id_gather_pipelines.find(சாவி);
        எனில் (it != mul_mat_id_gather_pipelines.end()) {
            திரும்பு it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(சாவி.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        auto processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[சாவி] = pipeline;
        திரும்பு pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_சாவி சாவி = {};
        சாவி.src0_type                           = context.src0->type;
        சாவி.src1_type                           = context.src1->type;
        சாவி.n_experts                           = context.src0->ne[2];
        சாவி.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_pipelines.find(சாவி);
        எனில் (it != mul_mat_id_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        தேர்வு (context.src1->type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        மாறிலி கட்டமைப்பு ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        மாறிலி எழுத்து *                    src0_name   = src0_traits->type_name;

        தேர்வு (context.src0->type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                நிறுத்து;
            நிலை GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                நிறுத்து;
            இயல்பு:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    தேர்வு (context.src0->type) {
                        நிலை GGML_TYPE_IQ1_S:
                        நிலை GGML_TYPE_IQ1_M:
                        நிலை GGML_TYPE_IQ4_NL:
                        நிலை GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            நிறுத்து;
                        நிலை GGML_TYPE_IQ2_XXS:
                        நிலை GGML_TYPE_IQ2_XS:
                        நிலை GGML_TYPE_IQ2_S:
                        நிலை GGML_TYPE_IQ3_XXS:
                        நிலை GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            நிறுத்து;
                        நிலை GGML_TYPE_MXFP4:
                        நிலை GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            நிறுத்து;
                        இயல்பு:
                            நிறுத்து;
                    }

                    variant += std::string("_") + src0_name;
                    நிறுத்து;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(சாவி.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        மாறிலி uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        எனில் (சாவி.vectorized) {
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
        mul_mat_id_pipelines[சாவி] = pipeline;
        திரும்பு mul_mat_id_pipelines[சாவி];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_சாவி சாவி = {};
        சாவி.src0_type                           = context.src0->type;
        சாவி.src1_type                           = context.src1->type;
        சாவி.n_experts                           = context.src0->ne[2];
        சாவி.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_vec_pipelines.find(சாவி);
        எனில் (it != mul_mat_id_vec_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        மாறிலி எழுத்து *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        தேர்வு (context.src1->type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        தேர்வு (context.src0->type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                நிறுத்து;
            நிலை GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                நிறுத்து;
            இயல்பு:
                {
                    // Quantized types: use helpers but accumulate in f16
                    மாறிலி கட்டமைப்பு ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    தேர்வு (context.src0->type) {
                        நிலை GGML_TYPE_IQ1_S:
                        நிலை GGML_TYPE_IQ1_M:
                        நிலை GGML_TYPE_IQ2_S:
                        நிலை GGML_TYPE_IQ3_S:
                        நிலை GGML_TYPE_IQ4_NL:
                        நிலை GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            நிறுத்து;
                        நிலை GGML_TYPE_IQ2_XXS:
                        நிலை GGML_TYPE_IQ2_XS:
                        நிலை GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            நிறுத்து;
                        நிலை GGML_TYPE_MXFP4:
                        நிலை GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            நிறுத்து;
                        இயல்பு:
                            நிறுத்து;
                    }
                    நிறுத்து;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(சாவி.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        எனில் (சாவி.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } இல்லையெனில் எனில் (சாவி.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } இல்லையெனில் எனில் (சாவி.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        எனில் (சாவி.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(சாவி.n_experts));

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[சாவி] = pipeline;
        திரும்பு mul_mat_id_vec_pipelines[சாவி];
    }

    webgpu_pipeline get_unary_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        மாறிலி தருக்க                     is_unary = context.dst->op == GGML_OP_UNARY;
        மாறிலி முழு                      op       = is_unary ? (முழு) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_சாவி சாவி      = {};
        சாவி.type                                = context.dst->type;
        சாவி.op                                  = op;
        சாவி.is_unary                            = is_unary;
        சாவி.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        சாவி.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        auto it = unary_pipelines.find(சாவி);
        எனில் (it != unary_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            சாவி.is_unary ? ggml_unary_op_name((ggml_unary_op) சாவி.op) : ggml_op_name((ggml_op) சாவி.op);
        defines.push_back(variant);

        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for unary shader");
        }

        எனில் (சாவி.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        எனில் (op == GGML_OP_TRI) {
            தேர்வு (சாவி.ttype) {
                நிலை GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    நிறுத்து;
                நிலை GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    நிறுத்து;
                நிலை GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    நிறுத்து;
                நிலை GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    நிறுத்து;
                இயல்பு:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_unary, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = சாவி.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[சாவி]     = pipeline;
        திரும்பு unary_pipelines[சாவி];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, தருக்க src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_சாவி சாவி = {};
        சாவி.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        சாவி.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        சாவி.src_overlap                           = src_overlap;

        auto it = rms_norm_mul_pipelines.find(சாவி);
        எனில் (it != rms_norm_mul_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        எனில் (சாவி.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } இல்லையெனில் எனில் (சாவி.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } இல்லையெனில் எனில் (சாவி.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = சாவி.inplace;
        pipeline_decisions->overlap     = சாவி.overlap;
        pipeline_decisions->src_overlap = சாவி.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[சாவி]     = pipeline;
        திரும்பு rms_norm_mul_pipelines[சாவி];
    }

    webgpu_pipeline get_இருமம்_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, தருக்க src_overlap) {
        ggml_webgpu_இருமம்_pipeline_சாவி சாவி = {};
        சாவி.type                            = context.dst->type;
        சாவி.op                              = context.dst->op;
        சாவி.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        சாவி.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        சாவி.src_overlap                     = src_overlap;

        auto it = இருமம்_pipelines.find(சாவி);
        எனில் (it != இருமம்_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) சாவி.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for binary shader");
        }

        எனில் (சாவி.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } இல்லையெனில் எனில் (சாவி.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } இல்லையெனில் எனில் (சாவி.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_இருமம், defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_இருமம்_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = சாவி.inplace;
        pipeline_decisions->overlap     = சாவி.overlap;
        pipeline_decisions->src_overlap = சாவி.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        இருமம்_pipelines[சாவி]    = pipeline;
        திரும்பு இருமம்_pipelines[சாவி];
    }

    webgpu_pipeline get_add_id_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_சாவி சாவி = {};
        சாவி.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = add_id_pipelines.find(சாவி);
        எனில் (it != add_id_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        மாறிலி எழுத்து *             shader_src = wgsl_add_id;

        எனில் (சாவி.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed              = preprocessor.preprocess(shader_src, defines);
        auto pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_size = context.max_wg_size;
        pipeline_decisions->inplace = சாவி.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[சாவி]    = pipeline;
        திரும்பு pipeline;
    }

    webgpu_pipeline get_concat_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, தருக்க src_overlap) {
        ggml_webgpu_concat_pipeline_சாவி சாவி = {};
        சாவி.type                            = context.dst->type;
        சாவி.src_overlap                     = src_overlap;

        auto it = concat_pipelines.find(சாவி);
        எனில் (it != concat_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for concat shader");
        }

        எனில் (சாவி.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_concat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_இருமம்_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = சாவி.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[சாவி]    = pipeline;
        திரும்பு concat_pipelines[சாவி];
    }

    webgpu_pipeline get_repeat_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_சாவி சாவி = {};
        சாவி.type                            = context.dst->type;

        auto it = repeat_pipelines.find(சாவி);
        எனில் (it != repeat_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                நிறுத்து;
            நிலை GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                நிறுத்து;
            நிலை GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_repeat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[சாவி]    = pipeline;
        திரும்பு repeat_pipelines[சாவி];
    }

    webgpu_pipeline get_flash_attn_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, தருக்க kv_overlap) {
        மாறிலி தருக்க can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_சாவி சாவி = {};
        சாவி.common                              = ggml_webgpu_flash_attn_make_common_pipeline_சாவி(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        சாவி.common.k_direct &= decisions.use_sg_matrix && சாவி.common.k_type == GGML_TYPE_F16;
        சாவி.common.v_direct &= decisions.use_sg_matrix && சாவி.common.v_type == GGML_TYPE_F16;
        சாவி.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_limit_bytes, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            சாவி.common.head_dim_qk, சாவி.common.head_dim_v, சாவி.common.has_mask,
            சாவி.common.k_direct || சாவி.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        எனில் (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = பொய்;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            சாவி.common.k_direct     = பொய்;
            சாவி.common.v_direct     = பொய்;
            சாவி.use_sg_matrix       = பொய்;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, சாவி.common.head_dim_qk, சாவி.common.head_dim_v,
                சாவி.common.has_mask, சாவி.common.k_direct || சாவி.common.v_direct);
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

        எனில் (சாவி.common.k_direct || சாவி.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            போது (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_size;
            }
        }

        auto it = flash_attn_pipelines.find(சாவி);
        எனில் (it != flash_attn_pipelines.end()) {
            திரும்பு it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(சாவி.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_size);
        மாறிலி எழுத்து *             shader_src = nullptr;
        எனில் (!சாவி.use_sg_matrix) {
            shader_src = wgsl_flash_attn_tile;
            defines.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_size) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_size) + "_" +
                       std::to_string(context.max_subgroup_size);
        } இல்லையெனில் {
            shader_src = wgsl_flash_attn;
            defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[சாவி] = pipeline;
        திரும்பு flash_attn_pipelines[சாவி];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, தருக்க kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_சாவி சாவி = {};
        சாவி.common = ggml_webgpu_flash_attn_make_common_pipeline_சாவி(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        auto it = flash_attn_vec_pipelines.find(சாவி);
        எனில் (it != flash_attn_vec_pipelines.end()) {
            திரும்பு it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, சாவி.common.head_dim_qk,
                                                                   சாவி.common.head_dim_v, சாவி.common.has_mask,
                                                                   சாவி.common.k_direct || சாவி.common.v_direct);
        decisions.wg_size = context.max_subgroup_size;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(சாவி.common, variant, 1u, decisions.kv_tile, decisions.wg_size);
        எனில் (சாவி.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (அளவு("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_size;
        மாறிலி uint32_t D       = சாவி.common.head_dim_qk | சாவி.common.head_dim_v;
        மாறிலி uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_size, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[சாவி] = pipeline;
        திரும்பு flash_attn_vec_pipelines[சாவி];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_சாவி சாவி = {};
        சாவி.kv_tile                                 = kv_tile;
        auto it                                     = flash_attn_blk_pipelines.find(சாவி);
        எனில் (it != flash_attn_blk_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(சாவி.kv_tile));
        variant += std::string("_kvt") + std::to_string(சாவி.kv_tile);

        uint32_t wg_size = 1;
        போது ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[சாவி] = pipeline;
        திரும்பு flash_attn_blk_pipelines[சாவி];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_சாவி சாவி = {};
        சாவி.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        சாவி.dst_type                                       = context.dst->type;
        சாவி.wg_size                                        = context.max_wg_size;
        auto it                                            = flash_attn_vec_reduce_pipelines.find(சாவி);
        எனில் (it != flash_attn_vec_reduce_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        தேர்வு (சாவி.dst_type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("DST_F32");
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("DST_F16");
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(சாவி.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(சாவி.head_dim_v));
        variant += std::string("_hsv") + std::to_string(சாவி.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[சாவி] = pipeline;
        திரும்பு flash_attn_vec_reduce_pipelines[சாவி];
    }

    webgpu_pipeline get_cpy_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_சாவி சாவி = {};
        சாவி.src_type                     = context.src0->type;
        சாவி.dst_type                     = context.dst->type;

        auto it = cpy_pipelines.find(சாவி);
        எனில் (it != cpy_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        தேர்வு (சாவி.src_type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                நிறுத்து;
            நிலை GGML_TYPE_I32:
                defines.push_back("SRC_I32");
                variant += "_i32";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        தேர்வு (சாவி.dst_type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                நிறுத்து;
            நிலை GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_cpy, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[சாவி]       = pipeline;
        திரும்பு cpy_pipelines[சாவி];
    }

    webgpu_pipeline get_glu_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, தருக்க src_overlap) {
        ggml_webgpu_glu_pipeline_சாவி சாவி = {};
        சாவி.glu_op                       = ggml_get_glu_op(context.dst);
        சாவி.type                         = context.dst->type;
        சாவி.split                        = (context.src1 != nullptr);
        சாவி.src_overlap                  = src_overlap;

        auto it = glu_pipelines.find(சாவி);
        எனில் (it != glu_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        தேர்வு (சாவி.glu_op) {
            நிலை GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                நிறுத்து;
            நிலை GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                நிறுத்து;
            நிலை GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                நிறுத்து;
            நிலை GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                நிறுத்து;
            நிலை GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                நிறுத்து;
            நிலை GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                நிறுத்து;
            நிலை GGML_GLU_OP_SWIGLU_CLAMP:
                defines.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported GLU op");
        }
        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        எனில் (சாவி.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } இல்லையெனில் எனில் (சாவி.split) {
            variant += "_split";
        } இல்லையெனில் {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_glu, defines);
        auto decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = சாவி.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[சாவி]       = pipeline;
        திரும்பு glu_pipelines[சாவி];
    }

    webgpu_pipeline get_rope_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_சாவி சாவி = {};
        சாவி.type                          = context.dst->type;
        சாவி.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        சாவி.has_ff                        = (context.src2 != nullptr);

        auto it = rope_pipelines.find(சாவி);
        எனில் (it != rope_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        தேர்வு (சாவி.type) {
            நிலை GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        எனில் (சாவி.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        எனில் (சாவி.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_rope, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = சாவி.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[சாவி]      = pipeline;
        திரும்பு rope_pipelines[சாவி];
    }

    webgpu_pipeline get_soft_max_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_சாவி சாவி = {};
        சாவி.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        சாவி.has_mask                          = (context.src1 != nullptr);
        சாவி.has_sink                          = (context.src2 != nullptr);
        சாவி.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = soft_max_pipelines.find(சாவி);
        எனில் (it != soft_max_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        எனில் (சாவி.has_mask) {
            defines.push_back("HAS_MASK");
            தேர்வு (சாவி.mask_type) {
                நிலை GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    நிறுத்து;
                நிலை GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    நிறுத்து;
                இயல்பு:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        எனில் (சாவி.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        எனில் (சாவி.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = சாவி.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[சாவி]  = pipeline;
        திரும்பு soft_max_pipelines[சாவி];
    }

    webgpu_pipeline get_conv2d_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_சாவி சாவி = {};
        சாவி.weight_type                     = context.src0->type;
        சாவி.உள்ளீடு_type                      = context.src1->type;
        சாவி.output_type                     = context.dst->type;

        auto it = conv2d_pipelines.find(சாவி);
        எனில் (it != conv2d_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        auto தள்ளு_type_defines = [&](மாறிலி எழுத்து * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            எனில் (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } இல்லையெனில் எனில் (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } இல்லையெனில் {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        தள்ளு_type_defines("WEIGHT_TYPE", சாவி.weight_type);
        தள்ளு_type_defines("INPUT_TYPE", சாவி.உள்ளீடு_type);
        தள்ளு_type_defines("OUTPUT_TYPE", சாவி.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[சாவி]    = pipeline;
        திரும்பு conv2d_pipelines[சாவி];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context, தருக்க whcn) {
        ggml_webgpu_conv2d_dw_pipeline_சாவி சாவி = {};
        சாவி.weight_type                        = context.src0->type;
        சாவி.உள்ளீடு_type                         = context.src1->type;
        சாவி.output_type                        = context.dst->type;
        சாவி.whcn                               = whcn;

        auto it = conv2d_dw_pipelines.find(சாவி);
        எனில் (it != conv2d_dw_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        auto தள்ளு_type_defines = [&](மாறிலி எழுத்து * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            எனில் (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } இல்லையெனில் எனில் (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } இல்லையெனில் {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        தள்ளு_type_defines("WEIGHT_TYPE", சாவி.weight_type);
        தள்ளு_type_defines("INPUT_TYPE", சாவி.உள்ளீடு_type);
        தள்ளு_type_defines("OUTPUT_TYPE", சாவி.output_type);

        எனில் (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[சாவி] = pipeline;
        திரும்பு conv2d_dw_pipelines[சாவி];
    }

    webgpu_pipeline get_im2col_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_சாவி சாவி = {};
        சாவி.உள்ளீடு_type                      = context.src1->type;
        சாவி.output_type                     = context.dst->type;

        auto it = im2col_pipelines.find(சாவி);
        எனில் (it != im2col_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        auto தள்ளு_type_defines = [&](மாறிலி எழுத்து * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            எனில் (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } இல்லையெனில் எனில் (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } இல்லையெனில் {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        தள்ளு_type_defines("INPUT_TYPE", சாவி.உள்ளீடு_type);
        தள்ளு_type_defines("OUTPUT_TYPE", சாவி.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[சாவி]    = pipeline;
        திரும்பு im2col_pipelines[சாவி];
    }

    webgpu_pipeline get_upscale_pipeline(மாறிலி ggml_webgpu_shader_lib_context & context) {
        மாறிலி uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        மாறிலி uint32_t base_mode  = mode_flags & 0xFFu;
        மாறிலி தருக்க     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_சாவி சாவி = {};
        சாவி.உள்ளீடு_type                       = context.src0->type;
        சாவி.output_type                      = context.dst->type;
        சாவி.base_mode                        = base_mode;
        சாவி.antialias                        = antialias;

        auto it = upscale_pipelines.find(சாவி);
        எனில் (it != upscale_pipelines.end()) {
            திரும்பு it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        எனில் (சாவி.உள்ளீடு_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } இல்லையெனில் {
            variant += "_src_f32";
        }

        எனில் (சாவி.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } இல்லையெனில் {
            variant += "_dst_f32";
        }

        தேர்வு (base_mode) {
            நிலை GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                நிறுத்து;
            நிலை GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                நிறுத்து;
            நிலை GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("Unsupported upscale mode");
        }

        எனில் (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_upscale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[சாவி]   = pipeline;
        திரும்பு upscale_pipelines[சாவி];
    }

  தனிப்பட்ட:
    நிலையான webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
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
        திரும்பு { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
