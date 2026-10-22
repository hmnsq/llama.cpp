#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml-wgsl-shaders.hpp"
#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "pre_wgsl.hpp"

#অন্তর্ভাৱ <webgpu/webgpu_cpp.h>

#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <string>
#অন্তর্ভাৱ <unordered_map>
#অন্তর্ভাৱ <vector>

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
template <typename T> অন্তঃস্থ শূন্য ggml_webgpu_hash_combine(size_t & seed, ধ্রুৱ T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
অন্তঃস্থ uintptr_t ggml_webgpu_tensor_addr(ধ্রুৱ ggml_tensor * tensor) {
    ধ্রুৱ ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    ঘূৰাই (uintptr_t) base_tensor->data + tensor->view_offs;
}

অন্তঃস্থ তার্কিক ggml_webgpu_tensor_equal(ধ্রুৱ ggml_tensor * a, ধ্রুৱ ggml_tensor * b) {
    ঘূৰাই a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

সংরচনা ggml_webgpu_shader_lib_context {
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
    তার্কিক        supports_subgroups       = মিছা;
    তার্কিক        supports_subgroup_matrix = মিছা;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_size        = 0;
    uint32_t    max_subgroup_size        = 0;
    তার্কিক        supports_dot_product     = মিছা;
    std::string vendor;
};

সংরচনা webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<শূন্য> context = nullptr;
};

সংরচনা ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 0;
    তার্কিক     inplace = মিছা;
};

সংরচনা ggml_webgpu_get_rows_shader_decisions {
    uint32_t wg_size    = 0;
    তার্কিক     vectorized = মিছা;
};

সংরচনা ggml_webgpu_দ্বিমিক_shader_decisions {
    uint32_t wg_size     = 0;
    তার্কিক     inplace     = মিছা;
    তার্কিক     overlap     = মিছা;
    তার্কিক     src_overlap = মিছা;
};

সংরচনা ggml_webgpu_glu_shader_decisions {
    uint32_t wg_size     = 0;
    তার্কিক     src_overlap = মিছা;
};

সংরচনা ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<শূন্য> decisions;
};

সংরচনা ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_size;
    uint32_t tokens_per_wg;
};

সংরচনা ggml_webgpu_ssm_scan_pipeline_চাবি {
    পূর্ণাঙ্ক     type;
    পূর্ণাঙ্ক     d_state;
    uint8_t xbc_overlap;
    তার্কিক    a_overlap;
    তার্কিক    ids_overlap;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_ssm_scan_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

সংরচনা ggml_webgpu_ssm_scan_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_ssm_scan_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ggml_webgpu_hash_combine(seed, চাবি.d_state);
        ggml_webgpu_hash_combine(seed, চাবি.xbc_overlap);
        ggml_webgpu_hash_combine(seed, চাবি.a_overlap);
        ggml_webgpu_hash_combine(seed, চাবি.ids_overlap);
        ঘূৰাই seed;
    }
};

সংরচনা ggml_webgpu_ssm_scan_shader_decisions {
    uint32_t wg_size;
    uint32_t tokens_per_tile;
    uint8_t  xbc_overlap = 0;
    তার্কিক     a_overlap   = মিছা;
    তার্কিক     ids_overlap = মিছা;
};

/** Argsort **/

সংরচনা ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes;
    int32_t  order;
};

/** Set Rows **/

সংরচনা ggml_webgpu_set_rows_pipeline_চাবি {
    পূর্ণাঙ্ক src0_type;
    পূর্ণাঙ্ক dst_type;
    পূর্ণাঙ্ক vec4;
    পূর্ণাঙ্ক i64_idx;
    পূর্ণাঙ্ক pair_blocks;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_set_rows_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

সংরচনা ggml_webgpu_set_rows_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_set_rows_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.src0_type);
        ggml_webgpu_hash_combine(seed, চাবি.dst_type);
        ggml_webgpu_hash_combine(seed, চাবি.vec4);
        ggml_webgpu_hash_combine(seed, চাবি.i64_idx);
        ggml_webgpu_hash_combine(seed, চাবি.pair_blocks);
        ঘূৰাই seed;
    }
};

সংরচনা ggml_webgpu_set_rows_shader_decisions {
    তার্কিক     vec4;
    তার্কিক     i64_idx;
    তার্কিক     pair_blocks;
    uint32_t wg_size;
};

/** Set **/

সংরচনা ggml_webgpu_set_pipeline_চাবি {
    ggml_type type;
    তার্কিক      inplace;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_set_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই type == other.type && inplace == other.inplace;
    }
};

সংরচনা ggml_webgpu_set_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_set_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ggml_webgpu_hash_combine(seed, চাবি.inplace);
        ঘূৰাই seed;
    }
};

/** Get Rows **/

সংরচনা ggml_webgpu_get_rows_pipeline_চাবি {
    ggml_type src_type;
    পূর্ণাঙ্ক       vectorized;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_get_rows_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই src_type == other.src_type && vectorized == other.vectorized;
    }
};

সংরচনা ggml_webgpu_get_rows_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_get_rows_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.src_type);
        ggml_webgpu_hash_combine(seed, চাবি.vectorized);
        ঘূৰাই seed;
    }
};

/** Row Norm **/

সংরচনা ggml_webgpu_row_norm_pipeline_চাবি {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    তার্কিক      inplace;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_row_norm_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

সংরচনা ggml_webgpu_row_norm_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_row_norm_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.op);
        ggml_webgpu_hash_combine(seed, চাবি.src_type);
        ggml_webgpu_hash_combine(seed, চাবি.dst_type);
        ggml_webgpu_hash_combine(seed, চাবি.inplace);
        ঘূৰাই seed;
    }
};

/** RMS_NORM + MUL **/

সংরচনা ggml_webgpu_rms_norm_mul_pipeline_চাবি {
    তার্কিক inplace;      // rn_src == dst
    তার্কিক overlap;      // mul_src == dst
    তার্কিক src_overlap;  // rn_src binding overlaps mul_src binding

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_rms_norm_mul_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

সংরচনা ggml_webgpu_rms_norm_mul_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_rms_norm_mul_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.inplace);
        ggml_webgpu_hash_combine(seed, চাবি.overlap);
        ggml_webgpu_hash_combine(seed, চাবি.src_overlap);
        ঘূৰাই seed;
    }
};

সংরচনা ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    তার্কিক     inplace     = মিছা;
    তার্কিক     overlap     = মিছা;
    তার্কিক     src_overlap = মিছা;
};

/** Pad **/
সংরচনা ggml_webgpu_pad_pipeline_চাবি {
    তার্কিক circular;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_pad_pipeline_চাবি & other) ধ্রুৱ { ঘূৰাই circular == other.circular; }
};

সংরচনা ggml_webgpu_pad_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_pad_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.circular);
        ঘূৰাই seed;
    }
};

/** Solve Tri **/
সংরচনা ggml_webgpu_solve_tri_pipeline_চাবি {
    পূর্ণাঙ্ক type;
    পূর্ণাঙ্ক n;
    পূর্ণাঙ্ক k;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_solve_tri_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই type == other.type && n == other.n && k == other.k;
    }
};

সংরচনা ggml_webgpu_solve_tri_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_solve_tri_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ggml_webgpu_hash_combine(seed, চাবি.n);
        ggml_webgpu_hash_combine(seed, চাবি.k);
        ঘূৰাই seed;
    }
};

/** SSM Conv **/
সংরচনা ggml_webgpu_ssm_conv_pipeline_চাবি {
    পূর্ণাঙ্ক type;
    পূর্ণাঙ্ক vectorized;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_ssm_conv_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
সংরচনা ggml_webgpu_conv2d_pipeline_চাবি {
    ggml_type weight_type;
    ggml_type প্ৰবেশ_type;
    ggml_type output_type;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_conv2d_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই weight_type == other.weight_type && প্ৰবেশ_type == other.প্ৰবেশ_type && output_type == other.output_type;
    }
};

সংরচনা ggml_webgpu_conv2d_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_conv2d_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.weight_type);
        ggml_webgpu_hash_combine(seed, চাবি.প্ৰবেশ_type);
        ggml_webgpu_hash_combine(seed, চাবি.output_type);
        ঘূৰাই seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
সংরচনা ggml_webgpu_conv2d_dw_pipeline_চাবি {
    ggml_type weight_type;
    ggml_type প্ৰবেশ_type;
    ggml_type output_type;
    তার্কিক      whcn;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_conv2d_dw_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই weight_type == other.weight_type && প্ৰবেশ_type == other.প্ৰবেশ_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

সংরচনা ggml_webgpu_conv2d_dw_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_conv2d_dw_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.weight_type);
        ggml_webgpu_hash_combine(seed, চাবি.প্ৰবেশ_type);
        ggml_webgpu_hash_combine(seed, চাবি.output_type);
        ggml_webgpu_hash_combine(seed, চাবি.whcn);
        ঘূৰাই seed;
    }
};

/** Im2Col **/
সংরচনা ggml_webgpu_im2col_pipeline_চাবি {
    ggml_type প্ৰবেশ_type;
    ggml_type output_type;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_im2col_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই প্ৰবেশ_type == other.প্ৰবেশ_type && output_type == other.output_type;
    }
};

সংরচনা ggml_webgpu_im2col_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_im2col_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.প্ৰবেশ_type);
        ggml_webgpu_hash_combine(seed, চাবি.output_type);
        ঘূৰাই seed;
    }
};

/** Gated Delta Net **/
সংরচনা ggml_webgpu_gated_delta_net_pipeline_চাবি {
    পূর্ণাঙ্ক  type;
    পূর্ণাঙ্ক  s_v;
    পূর্ণাঙ্ক  kda;
    তার্কিক fused_cache;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_gated_delta_net_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

সংরচনা ggml_webgpu_gated_delta_net_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_gated_delta_net_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ggml_webgpu_hash_combine(seed, চাবি.s_v);
        ggml_webgpu_hash_combine(seed, চাবি.kda);
        ঘূৰাই seed;
    }
};

সংরচনা ggml_webgpu_ssm_conv_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_ssm_conv_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ggml_webgpu_hash_combine(seed, চাবি.vectorized);
        ঘূৰাই seed;
    }
};

/** Scale **/

সংরচনা ggml_webgpu_scale_pipeline_চাবি {
    পূর্ণাঙ্ক inplace;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_scale_pipeline_চাবি & other) ধ্রুৱ { ঘূৰাই inplace == other.inplace; }
};

সংরচনা ggml_webgpu_scale_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_scale_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.inplace);
        ঘূৰাই seed;
    }
};

/** Upscale **/

সংরচনা ggml_webgpu_upscale_pipeline_চাবি {
    ggml_type প্ৰবেশ_type;
    ggml_type output_type;
    uint32_t  base_mode;
    তার্কিক      antialias;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_upscale_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই প্ৰবেশ_type == other.প্ৰবেশ_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

সংরচনা ggml_webgpu_upscale_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_upscale_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.প্ৰবেশ_type);
        ggml_webgpu_hash_combine(seed, চাবি.output_type);
        ggml_webgpu_hash_combine(seed, চাবি.base_mode);
        ggml_webgpu_hash_combine(seed, চাবি.antialias);
        ঘূৰাই seed;
    }
};

/** Concat **/

সংরচনা ggml_webgpu_concat_pipeline_চাবি {
    পূর্ণাঙ্ক  type;
    তার্কিক src_overlap;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_concat_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই type == other.type && src_overlap == other.src_overlap;
    }
};

সংরচনা ggml_webgpu_concat_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_concat_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ggml_webgpu_hash_combine(seed, চাবি.src_overlap);
        ঘূৰাই seed;
    }
};

/** Repeat **/

সংরচনা ggml_webgpu_repeat_pipeline_চাবি {
    পূর্ণাঙ্ক type;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_repeat_pipeline_চাবি & other) ধ্রুৱ { ঘূৰাই type == other.type; }
};

সংরচনা ggml_webgpu_repeat_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_repeat_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ঘূৰাই seed;
    }
};

/** Binary **/

সংরচনা ggml_webgpu_দ্বিমিক_pipeline_চাবি {
    পূর্ণাঙ্ক  type;
    পূর্ণাঙ্ক  op;
    তার্কিক inplace;
    তার্কিক overlap;
    তার্কিক src_overlap;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_দ্বিমিক_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

সংরচনা ggml_webgpu_দ্বিমিক_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_দ্বিমিক_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ggml_webgpu_hash_combine(seed, চাবি.op);
        ggml_webgpu_hash_combine(seed, চাবি.inplace);
        ggml_webgpu_hash_combine(seed, চাবি.overlap);
        ggml_webgpu_hash_combine(seed, চাবি.src_overlap);
        ঘূৰাই seed;
    }
};

/* Add_Id */

সংরচনা ggml_webgpu_add_id_pipeline_চাবি {
    তার্কিক inplace;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_add_id_pipeline_চাবি & other) ধ্রুৱ { ঘূৰাই inplace == other.inplace; }
};

সংরচনা ggml_webgpu_add_id_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_add_id_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.inplace);
        ঘূৰাই seed;
    }
};

/** Unary **/

সংরচনা ggml_webgpu_unary_pipeline_চাবি {
    পূর্ণাঙ্ক           type;
    পূর্ণাঙ্ক           op;
    তার্কিক          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    তার্কিক          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_unary_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

সংরচনা ggml_webgpu_unary_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_unary_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ggml_webgpu_hash_combine(seed, চাবি.op);
        ggml_webgpu_hash_combine(seed, চাবি.is_unary);
        ggml_webgpu_hash_combine(seed, চাবি.inplace);
        ggml_webgpu_hash_combine(seed, চাবি.ttype);
        ঘূৰাই seed;
    }
};

/** FlashAttention */

সংরচনা ggml_webgpu_flash_attn_common_pipeline_চাবি {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    তার্কিক      k_direct;
    তার্কিক      v_direct;
    তার্কিক      kv_overlap;
    তার্কিক      has_mask;
    তার্কিক      has_sinks;
    তার্কিক      uses_logit_softcap;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_flash_attn_common_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

অন্তঃস্থ শূন্য ggml_webgpu_flash_attn_hash_common_pipeline_চাবি(size_t &                                           seed,
                                                            ধ্রুৱ ggml_webgpu_flash_attn_common_pipeline_চাবি & চাবি) {
    ggml_webgpu_hash_combine(seed, চাবি.q_type);
    ggml_webgpu_hash_combine(seed, চাবি.k_type);
    ggml_webgpu_hash_combine(seed, চাবি.v_type);
    ggml_webgpu_hash_combine(seed, চাবি.dst_type);
    ggml_webgpu_hash_combine(seed, চাবি.head_dim_qk);
    ggml_webgpu_hash_combine(seed, চাবি.head_dim_v);
    ggml_webgpu_hash_combine(seed, চাবি.k_direct);
    ggml_webgpu_hash_combine(seed, চাবি.v_direct);
    ggml_webgpu_hash_combine(seed, চাবি.kv_overlap);
    ggml_webgpu_hash_combine(seed, চাবি.has_mask);
    ggml_webgpu_hash_combine(seed, চাবি.has_sinks);
    ggml_webgpu_hash_combine(seed, চাবি.uses_logit_softcap);
}

সংরচনা ggml_webgpu_flash_attn_vec_pipeline_চাবি {
    ggml_webgpu_flash_attn_common_pipeline_চাবি common;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_flash_attn_vec_pipeline_চাবি & other) ধ্রুৱ { ঘূৰাই common == other.common; }
};

সংরচনা ggml_webgpu_flash_attn_vec_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_flash_attn_vec_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_চাবি(seed, চাবি.common);
        ঘূৰাই seed;
    }
};

সংরচনা ggml_webgpu_flash_attn_pipeline_চাবি {
    ggml_webgpu_flash_attn_common_pipeline_চাবি common;
    তার্কিক                                       use_sg_matrix;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_flash_attn_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

সংরচনা ggml_webgpu_flash_attn_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_flash_attn_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_চাবি(seed, চাবি.common);
        ggml_webgpu_hash_combine(seed, চাবি.use_sg_matrix);
        ঘূৰাই seed;
    }
};

সংরচনা ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

সংরচনা ggml_webgpu_flash_attn_decisions {
    তার্কিক     use_sg_matrix = মিছা;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_size       = 0;
};

অন্তঃস্থ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
অন্তঃস্থ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

অন্তঃস্থ size_t ggml_webgpu_flash_attn_tensor_offset(ধ্রুৱ ggml_tensor * tensor) {
    constexpr uintptr_t ptr_base_addr = 0x1000u;
    ধ্রুৱ ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
    ঘূৰাই reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
}

অন্তঃস্থ তার্কিক ggml_webgpu_flash_attn_float_vec4_aligned(ধ্রুৱ ggml_tensor * K, size_t storage_offset_alignment) {
    ধ্রুৱ uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_size(K->type));
    ঘূৰাই offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

অন্তঃস্থ তার্কিক ggml_webgpu_flash_attn_float_vec4_aligned(ধ্রুৱ ggml_tensor * K,
                                                      ধ্রুৱ ggml_tensor * V,
                                                      size_t              storage_offset_alignment) {
    ঘূৰাই ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

অন্তঃস্থ তার্কিক ggml_webgpu_flash_attn_k_direct(ধ্রুৱ ggml_tensor * Q, ধ্রুৱ ggml_tensor * K, uint32_t kv_direct_align) {
    ঘূৰাই (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

অন্তঃস্থ তার্কিক ggml_webgpu_flash_attn_v_direct(ধ্রুৱ ggml_tensor * Q, ধ্রুৱ ggml_tensor * V, uint32_t kv_direct_align) {
    ঘূৰাই ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

অন্তঃস্থ ggml_webgpu_flash_attn_common_pipeline_চাবি ggml_webgpu_flash_attn_make_common_pipeline_চাবি(
    ধ্রুৱ ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    তার্কিক                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_চাবি চাবি = {};
    চাবি.q_type                                     = context.src0->type;
    চাবি.k_type                                     = context.src1->type;
    চাবি.v_type                                     = context.src2->type;
    চাবি.dst_type                                   = context.dst->type;
    চাবি.head_dim_qk                                = (uint32_t) context.src0->ne[0];
    চাবি.head_dim_v                                 = (uint32_t) context.src2->ne[0];
    চাবি.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    চাবি.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    চাবি.kv_overlap         = kv_overlap;
    চাবি.has_mask           = context.src3 != nullptr;
    চাবি.has_sinks          = context.src4 != nullptr;
    চাবি.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    ঘূৰাই চাবি;
}

অন্তঃস্থ std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    ধ্রুৱ ggml_webgpu_flash_attn_common_pipeline_চাবি & চাবি,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_size) {
    std::vector<std::string> defines;

    ৱিকল্প (চাবি.k_type) {
        পক্ষ GGML_TYPE_F32:
            defines.push_back("K_F32");
            ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            defines.push_back("K_F16");
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_0:
            defines.push_back("K_Q4_0");
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q8_0:
            defines.push_back("K_Q8_0");
            ভঙ্গ;
        স্ৱভাৱ:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(চাবি.k_type);

    ৱিকল্প (চাবি.v_type) {
        পক্ষ GGML_TYPE_F32:
            defines.push_back("V_F32");
            ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            defines.push_back("V_F16");
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_0:
            defines.push_back("V_Q4_0");
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q8_0:
            defines.push_back("V_Q8_0");
            ভঙ্গ;
        স্ৱভাৱ:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(চাবি.v_type);

    ৱিকল্প (চাবি.q_type) {
        পক্ষ GGML_TYPE_F32:
            defines.push_back("Q_F32");
            ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            defines.push_back("Q_F16");
            ভঙ্গ;
        স্ৱভাৱ:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(চাবি.q_type);

    ৱিকল্প (চাবি.dst_type) {
        পক্ষ GGML_TYPE_F32:
            defines.push_back("DST_F32");
            ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            defines.push_back("DST_F16");
            ভঙ্গ;
        স্ৱভাৱ:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(চাবি.dst_type);

    যদি (চাবি.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    যদি (চাবি.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    যদি (চাবি.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    যদি (চাবি.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    যদি (চাবি.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    যদি (চাবি.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(চাবি.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(চাবি.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(চাবি.head_dim_v));
    variant += std::string("_hsv") + std::to_string(চাবি.head_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    যদি (ggml_is_quantized(চাবি.k_type) || ggml_is_quantized(চাবি.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        যদি (ggml_is_quantized(চাবি.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        যদি (ggml_is_quantized(চাবি.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    ঘূৰাই defines;
}

সংরচনা ggml_webgpu_flash_attn_vec_reduce_pipeline_চাবি {
    uint32_t  head_dim_v;
    uint32_t  wg_size;
    ggml_type dst_type;
};

সংরচনা ggml_webgpu_flash_attn_vec_reduce_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_flash_attn_vec_reduce_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.head_dim_v);
        ggml_webgpu_hash_combine(seed, চাবি.wg_size);
        ggml_webgpu_hash_combine(seed, চাবি.dst_type);
        ঘূৰাই seed;
    }
};

অন্তঃস্থ তার্কিক operator==(ধ্রুৱ ggml_webgpu_flash_attn_vec_reduce_pipeline_চাবি & lhs,
                       ধ্রুৱ ggml_webgpu_flash_attn_vec_reduce_pipeline_চাবি & rhs) {
    ঘূৰাই lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size && lhs.dst_type == rhs.dst_type;
}

সংরচনা ggml_webgpu_flash_attn_blk_pipeline_চাবি {
    uint32_t kv_tile;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_flash_attn_blk_pipeline_চাবি & other) ধ্রুৱ { ঘূৰাই kv_tile == other.kv_tile; }
};

সংরচনা ggml_webgpu_flash_attn_blk_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_flash_attn_blk_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.kv_tile);
        ঘূৰাই seed;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
অন্তঃস্থ size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  তার্কিক     has_mask,
                                                  তার্কিক     kv_direct) {
    ধ্রুৱ uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * head_dim_qk;        // q_shmem
    যদি (!kv_direct) {
        f32_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f32_elems += q_tile * head_dim_v;         // o_shmem
    যদি (has_mask) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    ঘূৰাই f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

অন্তঃস্থ uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   limit_bytes,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t head_dim_qk,
                                                   uint32_t head_dim_v,
                                                   তার্কিক     has_mask,
                                                   তার্কিক     kv_direct) {
    ধ্রুৱ size_t base_q_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 0, head_dim_qk, head_dim_v, has_mask, kv_direct);
    যদি (limit_bytes <= base_q_bytes) {
        ঘূৰাই 0;
    }
    ধ্রুৱ size_t one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, head_dim_qk, head_dim_v, has_mask, kv_direct);
    ধ্রুৱ size_t bytes_per_kv = one_kv_bytes - base_q_bytes;
    যদি (bytes_per_kv == 0) {
        ঘূৰাই 0;
    }
    ধ্রুৱ size_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    ঘূৰাই (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

অন্তঃস্থ uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_limit_bytes,
                                                       uint32_t head_dim_qk,
                                                       uint32_t head_dim_v,
                                                       তার্কিক     has_mask,
                                                       তার্কিক     kv_direct) {
    ধ্রুৱ uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_limit_bytes, 1u, 1u, head_dim_qk, head_dim_v, has_mask, kv_direct);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    যদি (kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        যেতিয়ালৈ (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    ঘূৰাই kv_tile;
}

অন্তঃস্থ তার্কিক ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(তার্কিক                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ধ্রুৱ ggml_tensor * Q,
                                                                ধ্রুৱ ggml_tensor * V) {
    ঘূৰাই supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

সংরচনা ggml_webgpu_mul_mat_vec_pipeline_চাবি {
    ggml_type src0_type;
    ggml_type src1_type;
    পূর্ণাঙ্ক       vectorized;
    uint32_t  অঙ্ক_cols;
    তার্কিক      use_mmvq;
    তার্কিক      src_overlap;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_mul_mat_vec_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               অঙ্ক_cols == other.অঙ্ক_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

সংরচনা ggml_webgpu_mul_mat_vec_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_mul_mat_vec_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.src0_type);
        ggml_webgpu_hash_combine(seed, চাবি.src1_type);
        ggml_webgpu_hash_combine(seed, চাবি.vectorized);
        ggml_webgpu_hash_combine(seed, চাবি.অঙ্ক_cols);
        ggml_webgpu_hash_combine(seed, চাবি.use_mmvq);
        ggml_webgpu_hash_combine(seed, চাবি.src_overlap);
        ঘূৰাই seed;
    }
};

সংরচনা ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
    তার্কিক     src_overlap = মিছা;
};

সংরচনা ggml_webgpu_quantize_q8_pipeline_চাবি {
    ggml_type src0_type;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_quantize_q8_pipeline_চাবি & other) ধ্রুৱ { ঘূৰাই src0_type == other.src0_type; }
};

সংরচনা ggml_webgpu_quantize_q8_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_quantize_q8_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.src0_type);
        ঘূৰাই seed;
    }
};

সংরচনা ggml_webgpu_mul_mat_pipeline_চাবি {
    ggml_type src0_type;
    ggml_type src1_type;
    পূর্ণাঙ্ক       vectorized;
    পূর্ণাঙ্ক       use_subgroup_matrix;
    তার্কিক      src_overlap;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_mul_mat_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

সংরচনা ggml_webgpu_mul_mat_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_mul_mat_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.src0_type);
        ggml_webgpu_hash_combine(seed, চাবি.src1_type);
        ggml_webgpu_hash_combine(seed, চাবি.vectorized);
        ggml_webgpu_hash_combine(seed, চাবি.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, চাবি.src_overlap);
        ঘূৰাই seed;
    }
};

সংরচনা ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    পূর্ণাঙ্ক      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_size;
    তার্কিক     src_overlap = মিছা;
};

/** MUL_MAT_ID **/

সংরচনা ggml_webgpu_mul_mat_id_gather_pipeline_চাবি {
    uint32_t n_experts;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_mul_mat_id_gather_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই n_experts == other.n_experts;
    }
};

সংরচনা ggml_webgpu_mul_mat_id_gather_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_mul_mat_id_gather_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.n_experts);
        ঘূৰাই seed;
    }
};

সংরচনা ggml_webgpu_mul_mat_id_pipeline_চাবি {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  অঙ্ক_cols;
    পূর্ণাঙ্ক       vectorized;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_mul_mat_id_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               অঙ্ক_cols == other.অঙ্ক_cols && vectorized == other.vectorized;
    }
};

সংরচনা ggml_webgpu_mul_mat_id_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_mul_mat_id_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.src0_type);
        ggml_webgpu_hash_combine(seed, চাবি.src1_type);
        ggml_webgpu_hash_combine(seed, চাবি.n_experts);
        ggml_webgpu_hash_combine(seed, চাবি.অঙ্ক_cols);
        ggml_webgpu_hash_combine(seed, চাবি.vectorized);
        ঘূৰাই seed;
    }
};

/** Cpy **/

সংরচনা ggml_webgpu_cpy_pipeline_চাবি {
    ggml_type src_type;
    ggml_type dst_type;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_cpy_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই src_type == other.src_type && dst_type == other.dst_type;
    }
};

সংরচনা ggml_webgpu_cpy_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_cpy_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.src_type);
        ggml_webgpu_hash_combine(seed, চাবি.dst_type);
        ঘূৰাই seed;
    }
};

/** Glu **/

সংরচনা ggml_webgpu_glu_pipeline_চাবি {
    ggml_glu_op glu_op;
    ggml_type   type;
    তার্কিক        split;
    তার্কিক        src_overlap;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_glu_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

সংরচনা ggml_webgpu_glu_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_glu_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.glu_op);
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ggml_webgpu_hash_combine(seed, চাবি.split);
        ggml_webgpu_hash_combine(seed, চাবি.src_overlap);
        ঘূৰাই seed;
    }
};

/** Rope **/

সংরচনা ggml_webgpu_rope_pipeline_চাবি {
    ggml_type type;
    তার্কিক      inplace;
    তার্কিক      has_ff;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_rope_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

সংরচনা ggml_webgpu_rope_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_rope_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.type);
        ggml_webgpu_hash_combine(seed, চাবি.inplace);
        ggml_webgpu_hash_combine(seed, চাবি.has_ff);
        ঘূৰাই seed;
    }
};

/** SoftMax **/

সংরচনা ggml_webgpu_soft_max_pipeline_চাবি {
    ggml_type mask_type;
    তার্কিক      has_mask;
    তার্কিক      has_sink;
    তার্কিক      inplace;

    তার্কিক operator==(ধ্রুৱ ggml_webgpu_soft_max_pipeline_চাবি & other) ধ্রুৱ {
        ঘূৰাই mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

সংরচনা ggml_webgpu_soft_max_pipeline_চাবি_hash {
    size_t operator()(ধ্রুৱ ggml_webgpu_soft_max_pipeline_চাবি & চাবি) ধ্রুৱ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, চাবি.mask_type);
        ggml_webgpu_hash_combine(seed, চাবি.has_mask);
        ggml_webgpu_hash_combine(seed, চাবি.has_sink);
        ggml_webgpu_hash_combine(seed, চাবি.inplace);
        ঘূৰাই seed;
    }
};

/** MMVQ **/

অন্তঃস্থ তার্কিক ggml_webgpu_can_use_mmvq(ধ্রুৱ ggml_tensor * src0,
                                     ধ্রুৱ ggml_tensor * src1,
                                     তার্কিক                supports_dot_product,
                                     ধ্রুৱ std::string & vendor) {
    যদি (src1->ne[1] <= 4) {
        তার্কিক supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        যদি (supports_dp4a && supports_dot_product) {
            ৱিকল্প (src1->type) {
                পক্ষ GGML_TYPE_F32:
                    ৱিকল্প (src0->type) {
                        পক্ষ GGML_TYPE_Q1_0:
                        পক্ষ GGML_TYPE_Q4_0:
                        পক্ষ GGML_TYPE_Q4_1:
                        পক্ষ GGML_TYPE_Q5_0:
                        পক্ষ GGML_TYPE_Q5_1:
                        পক্ষ GGML_TYPE_Q8_0:
                        পক্ষ GGML_TYPE_MXFP4:
                        পক্ষ GGML_TYPE_Q2_K:
                        পক্ষ GGML_TYPE_Q3_K:
                        পক্ষ GGML_TYPE_Q4_K:
                        পক্ষ GGML_TYPE_Q5_K:
                        পক্ষ GGML_TYPE_Q6_K:
                            ঘূৰাই src0->ne[0] % 4 == 0;
                        স্ৱভাৱ:
                            ভঙ্গ;
                    }
                    ভঙ্গ;
                স্ৱভাৱ:
                    ভঙ্গ;
            }
        }
    }
    ঘূৰাই মিছা;
}

শ্ৰেণী ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<পূর্ণাঙ্ক, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<পূর্ণাঙ্ক, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<পূর্ণাঙ্ক, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<পূর্ণাঙ্ক, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<পূর্ণাঙ্ক, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_চাবি_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_চাবি_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_unary_pipeline_চাবি_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_scale_pipeline_চাবি_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_চাবি_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_চাবি_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_চাবি_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_চাবি,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_চাবি_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_pad_pipeline_চাবি_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_দ্বিমিক_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_দ্বিমিক_pipeline_চাবি_hash>
        দ্বিমিক_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_add_id_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_add_id_pipeline_চাবি_hash>
        add_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_concat_pipeline_চাবি_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_repeat_pipeline_চাবি_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_চাবি,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_চাবি_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_চাবি_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_চাবি,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_চাবি_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_চাবি,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_চাবি_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_চাবি_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_চাবি_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_চাবি_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_চাবি,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_চাবি_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_চাবি_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_চাবি_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_চাবি_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_set_pipeline_চাবি_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_cpy_pipeline_চাবি_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_glu_pipeline_চাবি_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_rope_pipeline_চাবি_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_চাবি_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_চাবি_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_চাবি_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_im2col_pipeline_চাবি_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_চাবি,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_চাবি_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_চাবি, webgpu_pipeline, ggml_webgpu_upscale_pipeline_চাবি_hash>
        upscale_pipelines;

  সার্ৱজনিক:
    ggml_webgpu_shader_lib(wgpu::Device device) { এতত্->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        auto it = sum_rows_pipelines.find(1);
        যদি (it != sum_rows_pipelines.end()) {
            ঘূৰাই it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        ঘূৰাই sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_চাবি চাবি = {};
        চাবি.op                                = context.dst->op;
        চাবি.src_type                          = context.src0->type;
        চাবি.dst_type                          = context.dst->type;
        চাবি.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = row_norm_pipelines.find(চাবি);
        যদি (it != row_norm_pipelines.end()) {
            ঘূৰাই it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        ৱিকল্প (চাবি.op) {
            পক্ষ GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                ভঙ্গ;
            পক্ষ GGML_OP_NORM:
                defines.push_back("NORM");
                variant = "norm";
                ভঙ্গ;
            পক্ষ GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        যদি (চাবি.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        যদি (চাবি.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } অন্যথা যদি (চাবি.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        যদি (চাবি.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } অন্যথা যদি (চাবি.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ধ্রুৱ uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = চাবি.inplace;
        row_norm_pipelines[চাবি]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[চাবি].context = decisions;
        ঘূৰাই row_norm_pipelines[চাবি];
    }

    webgpu_pipeline get_argmax_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        তার্কিক vec4 = context.src0->ne[0] % 4 == 0;

        auto it = argmax_pipelines.find(vec4);
        যদি (it != argmax_pipelines.end()) {
            ঘূৰাই it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        যদি (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        auto processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        ঘূৰাই argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ধ্রুৱ তার্কিক                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_চাবি চাবি       = {};
        চাবি.src0_type                               = context.src0->type;
        চাবি.dst_type                                = context.dst->type;
        চাবি.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        চাবি.i64_idx     = context.src1->type == GGML_TYPE_I64;
        চাবি.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_size(context.dst->type)) % 2 == 0);

        auto it = set_rows_pipelines.find(চাবি);
        যদি (it != set_rows_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        ৱিকল্প (context.dst->type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                ভঙ্গ;
            পক্ষ GGML_TYPE_Q8_0:
                defines.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                ভঙ্গ;
            পক্ষ GGML_TYPE_Q4_0:
                defines.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        যদি (context.src0->type == GGML_TYPE_F16) {
            defines.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        যদি (চাবি.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        যদি (চাবি.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        যদি (চাবি.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        ধ্রুৱ auto & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        auto         processed          = preprocessor.preprocess(shader_source, defines);
        auto         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = চাবি.vec4;
        decisions->i64_idx              = চাবি.i64_idx;
        decisions->pair_blocks          = চাবি.pair_blocks;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[চাবি]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[চাবি].context = decisions;
        ঘূৰাই set_rows_pipelines[চাবি];
    }

    webgpu_pipeline get_set_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_চাবি চাবি = {};
        চাবি.type                         = context.dst->type;
        চাবি.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = set_pipelines.find(চাবি);
        যদি (it != set_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for set shader");
        }

        যদি (চাবি.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_set, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = চাবি.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[চাবি]       = pipeline;
        ঘূৰাই set_pipelines[চাবি];
    }

    webgpu_pipeline get_cumsum_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        auto it = cumsum_pipelines.find(1);
        যদি (it != cumsum_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        ঘূৰাই cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        তার্কিক          is_শীৰ্ষ_k = context.dst->op == GGML_OP_শীৰ্ষ_K;
        // ascending order is 0, descending order is 1
        ধ্রুৱ int32_t order =
            is_শীৰ্ষ_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_pipelines.find(order);
        যদি (it != argsort_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        যেতিয়ালৈ (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        auto decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        ঘূৰাই argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        তার্কিক          is_শীৰ্ষ_k = context.dst->op == GGML_OP_শীৰ্ষ_K;
        // ascending order is 0, descending order is 1
        ধ্রুৱ int32_t order =
            is_শীৰ্ষ_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_merge_pipelines.find(order);
        যদি (it != argsort_merge_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        ঘূৰাই argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, তার্কিক vec4_aligned) {
        ধ্রুৱ তার্কিক vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_চাবি চাবি = {};
        চাবি.src_type                          = context.src0->type;
        চাবি.vectorized                        = (পূর্ণাঙ্ক) vectorized;

        auto it = get_rows_pipelines.find(চাবি);
        যদি (it != get_rows_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        ধ্রুৱ সংরচনা ggml_type_traits * type_traits = ggml_get_type_traits(চাবি.src_type);
        ধ্রুৱ ৱর্ণ *                    type_str    = type_traits->type_name;

        ৱিকল্প (চাবি.src_type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                যদি (চাবি.vectorized) {
                    defines.push_back("F32_VEC");
                    defines.push_back("SRC_TYPE=vec4<f32>");
                    defines.push_back("DST_TYPE=vec4<f32>");
                    defines.push_back("BLOCK_SIZE=4u");
                } অন্যথা {
                    defines.push_back("F32");
                    defines.push_back("SRC_TYPE=f32");
                    defines.push_back("DST_TYPE=f32");
                    defines.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                ভঙ্গ;
            পক্ষ GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_BF16:
                defines.push_back("BF16");
                defines.push_back("SRC_TYPE=u32");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                ভঙ্গ;
            স্ৱভাৱ:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    ৱিকল্প (চাবি.src_type) {
                        পক্ষ GGML_TYPE_Q1_0:
                        পক্ষ GGML_TYPE_Q2_0:
                        পক্ষ GGML_TYPE_Q4_0:
                        পক্ষ GGML_TYPE_Q5_0:
                        পক্ষ GGML_TYPE_Q8_0:
                        পক্ষ GGML_TYPE_Q3_K:
                        পক্ষ GGML_TYPE_Q6_K:
                        পক্ষ GGML_TYPE_IQ2_XXS:
                        পক্ষ GGML_TYPE_IQ2_XS:
                        পক্ষ GGML_TYPE_IQ2_S:
                        পক্ষ GGML_TYPE_IQ3_XXS:
                        পক্ষ GGML_TYPE_IQ3_S:
                        পক্ষ GGML_TYPE_IQ1_S:
                        পক্ষ GGML_TYPE_IQ4_NL:
                        পক্ষ GGML_TYPE_MXFP4:
                        পক্ষ GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                ভঙ্গ;
                            }
                        স্ৱভাৱ:
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

                    যদি (চাবি.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } অন্যথা যদি ((চাবি.src_type >= GGML_TYPE_Q4_0 && চাবি.src_type <= GGML_TYPE_Q8_1) ||
                               চাবি.src_type == GGML_TYPE_IQ4_NL || চাবি.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } অন্যথা যদি (চাবি.src_type == GGML_TYPE_Q2_0 || চাবি.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } অন্যথা যদি (চাবি.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } অন্যথা {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    ভঙ্গ;
                }
        }

        যদি (চাবি.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        auto decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[চাবি]  = pipeline;
        ঘূৰাই get_rows_pipelines[চাবি];
    }

    webgpu_pipeline get_scale_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_চাবি চাবি = {};
        চাবি.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = scale_pipelines.find(চাবি);
        যদি (it != scale_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        যদি (চাবি.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_scale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = চাবি.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[চাবি]     = pipeline;
        ঘূৰাই scale_pipelines[চাবি];
    }

    webgpu_pipeline get_solve_tri_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_চাবি চাবি = {};
        চাবি.type                               = context.dst->type;
        চাবি.n                                  = (পূর্ণাঙ্ক) context.src0->ne[0];
        চাবি.k                                  = (পূর্ণাঙ্ক) context.src1->ne[0];

        auto it = solve_tri_pipelines.find(চাবি);
        যদি (it != solve_tri_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                variant += "_f32";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ধ্রুৱ uint32_t wg_size       = std::min((uint32_t) চাবি.n, context.max_wg_size);
        ধ্রুৱ uint32_t k_tile        = wg_size;
        ধ্রুৱ uint32_t bytes_per_row = ((uint32_t) চাবি.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        ধ্রুৱ uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(চাবি.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        auto processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[চাবি] = pipeline;
        ঘূৰাই solve_tri_pipelines[চাবি];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_চাবি চাবি = {};
        চাবি.type                              = context.dst->type;
        চাবি.vectorized                        = context.src1->ne[0] == 4;

        auto it = ssm_conv_pipelines.find(চাবি);
        যদি (it != ssm_conv_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                variant += "_f32";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        যদি (চাবি.vectorized) {
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
        ssm_conv_pipelines[চাবি]  = pipeline;
        ঘূৰাই ssm_conv_pipelines[চাবি];
    }

    webgpu_pipeline get_ssm_scan_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_চাবি চাবি = {};
        চাবি.type                              = context.dst->type;
        চাবি.d_state                           = (পূর্ণাঙ্ক) context.src0->ne[0];
        চাবি.xbc_overlap                       = xbc_overlap;

        auto it = ssm_scan_pipelines.find(চাবি);
        যদি (it != ssm_scan_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                variant += "_f32";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ধ্রুৱ uint32_t wg_size = (uint32_t) চাবি.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        যদি (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } অন্যথা {
            variant += "_wg_reduce";
        }

        যদি (চাবি.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } অন্যথা যদি (চাবি.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } অন্যথা যদি (চাবি.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(চাবি.d_state);

        auto processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        auto decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = চাবি.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[চাবি]    = pipeline;
        ঘূৰাই ssm_scan_pipelines[চাবি];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_চাবি চাবি = {};
        চাবি.type                                     = context.dst->type;
        চাবি.s_v                                      = (পূর্ণাঙ্ক) context.src2->ne[0];
        চাবি.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        চাবি.fused_cache                              = context.dst_fuse != nullptr;

        auto it = gated_delta_net_pipelines.find(চাবি);
        যদি (it != gated_delta_net_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                variant += "_f32";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        যদি (চাবি.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        যদি (চাবি.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(চাবি.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(চাবি.s_v) + "u");

        auto            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[চাবি] = pipeline;
        ঘূৰাই gated_delta_net_pipelines[চাবি];
    }

    webgpu_pipeline get_pad_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_চাবি চাবি = {};
        চাবি.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        auto it = pad_pipelines.find(চাবি);
        যদি (it != pad_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        যদি (চাবি.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_pad, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[চাবি]       = pipeline;
        ঘূৰাই pad_pipelines[চাবি];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_চাবি চাবি = {};
        চাবি.src0_type                            = context.src0->type;

        auto it = quantize_q8_pipelines.find(চাবি);
        যদি (it != quantize_q8_pipelines.end()) {
            ঘূৰাই it->second;
        }
        ধ্রুৱ ৱর্ণ *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> defines;
        std::string              variant = "quantize_q8";

        uint32_t wg_size = WEBGPU_MUL_MAT_VEC_WG_SIZE;

        defines.push_back("SRC1_INNER_TYPE=f32");
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        ধ্রুৱ সংরচনা ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
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
        quantize_q8_pipelines[চাবি] = pipeline;
        ঘূৰাই quantize_q8_pipelines[চাবি];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, তার্কিক src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_চাবি চাবি = {};
        চাবি.src0_type                            = context.src0->type;
        চাবি.src1_type                            = context.src1->type;
        চাবি.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        চাবি.অঙ্ক_cols   = context.dst->ne[1];
        চাবি.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
        চাবি.src_overlap = src_overlap;

        auto it = mul_mat_vec_pipelines.find(চাবি);
        যদি (it != mul_mat_vec_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        ধ্রুৱ ৱর্ণ *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        ৱিকল্প (context.src0->type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ভঙ্গ;
            পক্ষ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                ভঙ্গ;
            স্ৱভাৱ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ধ্রুৱ সংরচনা ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    ৱিকল্প (context.src0->type) {
                        পক্ষ GGML_TYPE_Q4_0:
                        পক্ষ GGML_TYPE_Q4_1:
                        পক্ষ GGML_TYPE_Q5_0:
                        পক্ষ GGML_TYPE_Q5_1:
                        পক্ষ GGML_TYPE_Q8_0:
                            যদি (চাবি.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_Q1_0:
                        পক্ষ GGML_TYPE_Q2_K:
                        পক্ষ GGML_TYPE_Q3_K:
                        পক্ষ GGML_TYPE_Q4_K:
                        পক্ষ GGML_TYPE_Q5_K:
                        পক্ষ GGML_TYPE_Q6_K:
                            যদি (চাবি.use_mmvq) {
                                defines.push_back("K_QUANTS_HANDLING");
                            }
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_IQ1_S:
                        পক্ষ GGML_TYPE_IQ1_M:
                        পক্ষ GGML_TYPE_IQ2_S:
                        পক্ষ GGML_TYPE_IQ3_S:
                        পক্ষ GGML_TYPE_IQ4_NL:
                        পক্ষ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_IQ2_XXS:
                        পক্ষ GGML_TYPE_IQ2_XS:
                        পক্ষ GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_MXFP4:
                            defines.push_back(type_upper + "_LUT");
                            যদি (চাবি.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ভঙ্গ;
                        স্ৱভাৱ:
                            ভঙ্গ;
                    }
                    ভঙ্গ;
                }
        }

        // src1 type (vector)
        ৱিকল্প (context.src1->type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(চাবি.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        যদি (চাবি.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } অন্যথা যদি (চাবি.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } অন্যথা যদি (চাবি.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        যদি (চাবি.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        যদি (চাবি.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        যদি (চাবি.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(চাবি.অঙ্ক_cols));

        auto processed            = preprocessor.preprocess(shader_src, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = চাবি.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[চাবি] = pipeline;
        ঘূৰাই mul_mat_vec_pipelines[চাবি];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, তার্কিক src_overlap) {
        ggml_webgpu_mul_mat_pipeline_চাবি চাবি = {};
        চাবি.src0_type                        = context.src0->type;
        চাবি.src1_type                        = context.src1->type;
        চাবি.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        চাবি.use_subgroup_matrix = context.supports_subgroup_matrix;
        চাবি.src_overlap         = src_overlap;

        auto it = mul_mat_fast_pipelines.find(চাবি);
        যদি (it != mul_mat_fast_pipelines.end()) {
            ঘূৰাই it->second;
        }

        ধ্রুৱ ৱর্ণ * shader_src = চাবি.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = চাবি.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        ৱিকল্প (context.src1->type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ধ্রুৱ সংরচনা ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ধ্রুৱ ৱর্ণ *                    src0_name   = src0_traits->type_name;

        ৱিকল্প (context.src0->type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("TYPE_F32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("TYPE_F16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                ভঙ্গ;
            পক্ষ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("TYPE_BF16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                ভঙ্গ;
            স্ৱভাৱ:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    ৱিকল্প (context.src0->type) {
                        পক্ষ GGML_TYPE_IQ1_S:
                        পক্ষ GGML_TYPE_IQ1_M:
                        পক্ষ GGML_TYPE_IQ4_NL:
                        পক্ষ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_IQ2_XXS:
                        পক্ষ GGML_TYPE_IQ2_XS:
                        পক্ষ GGML_TYPE_IQ2_S:
                        পক্ষ GGML_TYPE_IQ3_XXS:
                        পক্ষ GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_MXFP4:
                        পক্ষ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ভঙ্গ;
                        স্ৱভাৱ:
                            ভঙ্গ;
                    }

                    variant += std::string("_") + src0_name;
                    ভঙ্গ;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(চাবি.vectorized ? "VEC" : "SCALAR");

        ধ্রুৱ তার্কিক is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        যদি (চাবি.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } অন্যথা {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        যদি (চাবি.use_subgroup_matrix) {
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
        যদি (চাবি.vectorized) {
            variant += "_vectorized";
        }

        যদি (চাবি.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        যদি (!চাবি.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = চাবি.use_subgroup_matrix;
        যদি (চাবি.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_size           = context.max_subgroup_size;
        } অন্যথা {
            decisions->wg_size_m       = WEBGPU_MUL_MAT_WG_SIZE_M;
            decisions->wg_size_n       = WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->wg_size         = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->mul_mat_wg_size = WEBGPU_MUL_MAT_WG_SIZE;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[চাবি] = pipeline;
        ঘূৰাই mul_mat_fast_pipelines[চাবি];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_চাবি চাবি = {};
        চাবি.n_experts                                  = (uint32_t) context.src0->ne[2];

        auto it = mul_mat_id_gather_pipelines.find(চাবি);
        যদি (it != mul_mat_id_gather_pipelines.end()) {
            ঘূৰাই it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(চাবি.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        auto processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[চাবি] = pipeline;
        ঘূৰাই pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_চাবি চাবি = {};
        চাবি.src0_type                           = context.src0->type;
        চাবি.src1_type                           = context.src1->type;
        চাবি.n_experts                           = context.src0->ne[2];
        চাবি.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_pipelines.find(চাবি);
        যদি (it != mul_mat_id_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        ৱিকল্প (context.src1->type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ধ্রুৱ সংরচনা ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ধ্রুৱ ৱর্ণ *                    src0_name   = src0_traits->type_name;

        ৱিকল্প (context.src0->type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ভঙ্গ;
            পক্ষ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                ভঙ্গ;
            স্ৱভাৱ:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    ৱিকল্প (context.src0->type) {
                        পক্ষ GGML_TYPE_IQ1_S:
                        পক্ষ GGML_TYPE_IQ1_M:
                        পক্ষ GGML_TYPE_IQ4_NL:
                        পক্ষ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_IQ2_XXS:
                        পক্ষ GGML_TYPE_IQ2_XS:
                        পক্ষ GGML_TYPE_IQ2_S:
                        পক্ষ GGML_TYPE_IQ3_XXS:
                        পক্ষ GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_MXFP4:
                        পক্ষ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ভঙ্গ;
                        স্ৱভাৱ:
                            ভঙ্গ;
                    }

                    variant += std::string("_") + src0_name;
                    ভঙ্গ;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(চাবি.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ধ্রুৱ uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        যদি (চাবি.vectorized) {
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
        mul_mat_id_pipelines[চাবি] = pipeline;
        ঘূৰাই mul_mat_id_pipelines[চাবি];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_চাবি চাবি = {};
        চাবি.src0_type                           = context.src0->type;
        চাবি.src1_type                           = context.src1->type;
        চাবি.n_experts                           = context.src0->ne[2];
        চাবি.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_vec_pipelines.find(চাবি);
        যদি (it != mul_mat_id_vec_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        ধ্রুৱ ৱর্ণ *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        ৱিকল্প (context.src1->type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ৱিকল্প (context.src0->type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ভঙ্গ;
            পক্ষ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                ভঙ্গ;
            স্ৱভাৱ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ধ্রুৱ সংরচনা ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    ৱিকল্প (context.src0->type) {
                        পক্ষ GGML_TYPE_IQ1_S:
                        পক্ষ GGML_TYPE_IQ1_M:
                        পক্ষ GGML_TYPE_IQ2_S:
                        পক্ষ GGML_TYPE_IQ3_S:
                        পক্ষ GGML_TYPE_IQ4_NL:
                        পক্ষ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_IQ2_XXS:
                        পক্ষ GGML_TYPE_IQ2_XS:
                        পক্ষ GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ভঙ্গ;
                        পক্ষ GGML_TYPE_MXFP4:
                        পক্ষ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ভঙ্গ;
                        স্ৱভাৱ:
                            ভঙ্গ;
                    }
                    ভঙ্গ;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(চাবি.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        যদি (চাবি.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } অন্যথা যদি (চাবি.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } অন্যথা যদি (চাবি.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        যদি (চাবি.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(চাবি.n_experts));

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[চাবি] = pipeline;
        ঘূৰাই mul_mat_id_vec_pipelines[চাবি];
    }

    webgpu_pipeline get_unary_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ধ্রুৱ তার্কিক                     is_unary = context.dst->op == GGML_OP_UNARY;
        ধ্রুৱ পূর্ণাঙ্ক                      op       = is_unary ? (পূর্ণাঙ্ক) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_চাবি চাবি      = {};
        চাবি.type                                = context.dst->type;
        চাবি.op                                  = op;
        চাবি.is_unary                            = is_unary;
        চাবি.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        চাবি.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        auto it = unary_pipelines.find(চাবি);
        যদি (it != unary_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            চাবি.is_unary ? ggml_unary_op_name((ggml_unary_op) চাবি.op) : ggml_op_name((ggml_op) চাবি.op);
        defines.push_back(variant);

        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for unary shader");
        }

        যদি (চাবি.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        যদি (op == GGML_OP_TRI) {
            ৱিকল্প (চাবি.ttype) {
                পক্ষ GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    ভঙ্গ;
                পক্ষ GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    ভঙ্গ;
                পক্ষ GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    ভঙ্গ;
                পক্ষ GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    ভঙ্গ;
                স্ৱভাৱ:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_unary, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = চাবি.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[চাবি]     = pipeline;
        ঘূৰাই unary_pipelines[চাবি];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, তার্কিক src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_চাবি চাবি = {};
        চাবি.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        চাবি.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        চাবি.src_overlap                           = src_overlap;

        auto it = rms_norm_mul_pipelines.find(চাবি);
        যদি (it != rms_norm_mul_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        যদি (চাবি.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } অন্যথা যদি (চাবি.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } অন্যথা যদি (চাবি.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = চাবি.inplace;
        pipeline_decisions->overlap     = চাবি.overlap;
        pipeline_decisions->src_overlap = চাবি.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[চাবি]     = pipeline;
        ঘূৰাই rms_norm_mul_pipelines[চাবি];
    }

    webgpu_pipeline get_দ্বিমিক_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, তার্কিক src_overlap) {
        ggml_webgpu_দ্বিমিক_pipeline_চাবি চাবি = {};
        চাবি.type                            = context.dst->type;
        চাবি.op                              = context.dst->op;
        চাবি.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        চাবি.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        চাবি.src_overlap                     = src_overlap;

        auto it = দ্বিমিক_pipelines.find(চাবি);
        যদি (it != দ্বিমিক_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) চাবি.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for binary shader");
        }

        যদি (চাবি.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } অন্যথা যদি (চাবি.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } অন্যথা যদি (চাবি.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_দ্বিমিক, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_দ্বিমিক_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = চাবি.inplace;
        pipeline_decisions->overlap     = চাবি.overlap;
        pipeline_decisions->src_overlap = চাবি.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        দ্বিমিক_pipelines[চাবি]    = pipeline;
        ঘূৰাই দ্বিমিক_pipelines[চাবি];
    }

    webgpu_pipeline get_add_id_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_চাবি চাবি = {};
        চাবি.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = add_id_pipelines.find(চাবি);
        যদি (it != add_id_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        ধ্রুৱ ৱর্ণ *             shader_src = wgsl_add_id;

        যদি (চাবি.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed              = preprocessor.preprocess(shader_src, defines);
        auto pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_size = context.max_wg_size;
        pipeline_decisions->inplace = চাবি.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[চাবি]    = pipeline;
        ঘূৰাই pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, তার্কিক src_overlap) {
        ggml_webgpu_concat_pipeline_চাবি চাবি = {};
        চাবি.type                            = context.dst->type;
        চাবি.src_overlap                     = src_overlap;

        auto it = concat_pipelines.find(চাবি);
        যদি (it != concat_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for concat shader");
        }

        যদি (চাবি.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_concat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_দ্বিমিক_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = চাবি.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[চাবি]    = pipeline;
        ঘূৰাই concat_pipelines[চাবি];
    }

    webgpu_pipeline get_repeat_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_চাবি চাবি = {};
        চাবি.type                            = context.dst->type;

        auto it = repeat_pipelines.find(চাবি);
        যদি (it != repeat_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ভঙ্গ;
            পক্ষ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_repeat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[চাবি]    = pipeline;
        ঘূৰাই repeat_pipelines[চাবি];
    }

    webgpu_pipeline get_flash_attn_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, তার্কিক kv_overlap) {
        ধ্রুৱ তার্কিক can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_চাবি চাবি = {};
        চাবি.common                              = ggml_webgpu_flash_attn_make_common_pipeline_চাবি(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        চাবি.common.k_direct &= decisions.use_sg_matrix && চাবি.common.k_type == GGML_TYPE_F16;
        চাবি.common.v_direct &= decisions.use_sg_matrix && চাবি.common.v_type == GGML_TYPE_F16;
        চাবি.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_limit_bytes, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            চাবি.common.head_dim_qk, চাবি.common.head_dim_v, চাবি.common.has_mask,
            চাবি.common.k_direct || চাবি.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        যদি (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = মিছা;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            চাবি.common.k_direct     = মিছা;
            চাবি.common.v_direct     = মিছা;
            চাবি.use_sg_matrix       = মিছা;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, চাবি.common.head_dim_qk, চাবি.common.head_dim_v,
                চাবি.common.has_mask, চাবি.common.k_direct || চাবি.common.v_direct);
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

        যদি (চাবি.common.k_direct || চাবি.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            যেতিয়ালৈ (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_size;
            }
        }

        auto it = flash_attn_pipelines.find(চাবি);
        যদি (it != flash_attn_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(চাবি.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_size);
        ধ্রুৱ ৱর্ণ *             shader_src = nullptr;
        যদি (!চাবি.use_sg_matrix) {
            shader_src = wgsl_flash_attn_tile;
            defines.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_size) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_size) + "_" +
                       std::to_string(context.max_subgroup_size);
        } অন্যথা {
            shader_src = wgsl_flash_attn;
            defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[চাবি] = pipeline;
        ঘূৰাই flash_attn_pipelines[চাবি];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, তার্কিক kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_চাবি চাবি = {};
        চাবি.common = ggml_webgpu_flash_attn_make_common_pipeline_চাবি(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        auto it = flash_attn_vec_pipelines.find(চাবি);
        যদি (it != flash_attn_vec_pipelines.end()) {
            ঘূৰাই it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, চাবি.common.head_dim_qk,
                                                                   চাবি.common.head_dim_v, চাবি.common.has_mask,
                                                                   চাবি.common.k_direct || চাবি.common.v_direct);
        decisions.wg_size = context.max_subgroup_size;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(চাবি.common, variant, 1u, decisions.kv_tile, decisions.wg_size);
        যদি (চাবি.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (পরিমাণ("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_size;
        ধ্রুৱ uint32_t D       = চাবি.common.head_dim_qk | চাবি.common.head_dim_v;
        ধ্রুৱ uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_size, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[চাবি] = pipeline;
        ঘূৰাই flash_attn_vec_pipelines[চাবি];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_চাবি চাবি = {};
        চাবি.kv_tile                                 = kv_tile;
        auto it                                     = flash_attn_blk_pipelines.find(চাবি);
        যদি (it != flash_attn_blk_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(চাবি.kv_tile));
        variant += std::string("_kvt") + std::to_string(চাবি.kv_tile);

        uint32_t wg_size = 1;
        যেতিয়ালৈ ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[চাবি] = pipeline;
        ঘূৰাই flash_attn_blk_pipelines[চাবি];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_চাবি চাবি = {};
        চাবি.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        চাবি.dst_type                                       = context.dst->type;
        চাবি.wg_size                                        = context.max_wg_size;
        auto it                                            = flash_attn_vec_reduce_pipelines.find(চাবি);
        যদি (it != flash_attn_vec_reduce_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        ৱিকল্প (চাবি.dst_type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(চাবি.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(চাবি.head_dim_v));
        variant += std::string("_hsv") + std::to_string(চাবি.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[চাবি] = pipeline;
        ঘূৰাই flash_attn_vec_reduce_pipelines[চাবি];
    }

    webgpu_pipeline get_cpy_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_চাবি চাবি = {};
        চাবি.src_type                     = context.src0->type;
        চাবি.dst_type                     = context.dst->type;

        auto it = cpy_pipelines.find(চাবি);
        যদি (it != cpy_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        ৱিকল্প (চাবি.src_type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                ভঙ্গ;
            পক্ষ GGML_TYPE_I32:
                defines.push_back("SRC_I32");
                variant += "_i32";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        ৱিকল্প (চাবি.dst_type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                ভঙ্গ;
            পক্ষ GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_cpy, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[চাবি]       = pipeline;
        ঘূৰাই cpy_pipelines[চাবি];
    }

    webgpu_pipeline get_glu_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, তার্কিক src_overlap) {
        ggml_webgpu_glu_pipeline_চাবি চাবি = {};
        চাবি.glu_op                       = ggml_get_glu_op(context.dst);
        চাবি.type                         = context.dst->type;
        চাবি.split                        = (context.src1 != nullptr);
        চাবি.src_overlap                  = src_overlap;

        auto it = glu_pipelines.find(চাবি);
        যদি (it != glu_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        ৱিকল্প (চাবি.glu_op) {
            পক্ষ GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                ভঙ্গ;
            পক্ষ GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                ভঙ্গ;
            পক্ষ GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                ভঙ্গ;
            পক্ষ GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                ভঙ্গ;
            পক্ষ GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                ভঙ্গ;
            পক্ষ GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                ভঙ্গ;
            পক্ষ GGML_GLU_OP_SWIGLU_CLAMP:
                defines.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported GLU op");
        }
        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        যদি (চাবি.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } অন্যথা যদি (চাবি.split) {
            variant += "_split";
        } অন্যথা {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_glu, defines);
        auto decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = চাবি.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[চাবি]       = pipeline;
        ঘূৰাই glu_pipelines[চাবি];
    }

    webgpu_pipeline get_rope_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_চাবি চাবি = {};
        চাবি.type                          = context.dst->type;
        চাবি.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        চাবি.has_ff                        = (context.src2 != nullptr);

        auto it = rope_pipelines.find(চাবি);
        যদি (it != rope_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        ৱিকল্প (চাবি.type) {
            পক্ষ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        যদি (চাবি.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        যদি (চাবি.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_rope, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = চাবি.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[চাবি]      = pipeline;
        ঘূৰাই rope_pipelines[চাবি];
    }

    webgpu_pipeline get_soft_max_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_চাবি চাবি = {};
        চাবি.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        চাবি.has_mask                          = (context.src1 != nullptr);
        চাবি.has_sink                          = (context.src2 != nullptr);
        চাবি.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = soft_max_pipelines.find(চাবি);
        যদি (it != soft_max_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        যদি (চাবি.has_mask) {
            defines.push_back("HAS_MASK");
            ৱিকল্প (চাবি.mask_type) {
                পক্ষ GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    ভঙ্গ;
                পক্ষ GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    ভঙ্গ;
                স্ৱভাৱ:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        যদি (চাবি.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        যদি (চাবি.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = চাবি.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[চাবি]  = pipeline;
        ঘূৰাই soft_max_pipelines[চাবি];
    }

    webgpu_pipeline get_conv2d_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_চাবি চাবি = {};
        চাবি.weight_type                     = context.src0->type;
        চাবি.প্ৰবেশ_type                      = context.src1->type;
        চাবি.output_type                     = context.dst->type;

        auto it = conv2d_pipelines.find(চাবি);
        যদি (it != conv2d_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        auto ঠেলো_type_defines = [&](ধ্রুৱ ৱর্ণ * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            যদি (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } অন্যথা যদি (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } অন্যথা {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ঠেলো_type_defines("WEIGHT_TYPE", চাবি.weight_type);
        ঠেলো_type_defines("INPUT_TYPE", চাবি.প্ৰবেশ_type);
        ঠেলো_type_defines("OUTPUT_TYPE", চাবি.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[চাবি]    = pipeline;
        ঘূৰাই conv2d_pipelines[চাবি];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context, তার্কিক whcn) {
        ggml_webgpu_conv2d_dw_pipeline_চাবি চাবি = {};
        চাবি.weight_type                        = context.src0->type;
        চাবি.প্ৰবেশ_type                         = context.src1->type;
        চাবি.output_type                        = context.dst->type;
        চাবি.whcn                               = whcn;

        auto it = conv2d_dw_pipelines.find(চাবি);
        যদি (it != conv2d_dw_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        auto ঠেলো_type_defines = [&](ধ্রুৱ ৱর্ণ * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            যদি (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } অন্যথা যদি (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } অন্যথা {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ঠেলো_type_defines("WEIGHT_TYPE", চাবি.weight_type);
        ঠেলো_type_defines("INPUT_TYPE", চাবি.প্ৰবেশ_type);
        ঠেলো_type_defines("OUTPUT_TYPE", চাবি.output_type);

        যদি (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[চাবি] = pipeline;
        ঘূৰাই conv2d_dw_pipelines[চাবি];
    }

    webgpu_pipeline get_im2col_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_চাবি চাবি = {};
        চাবি.প্ৰবেশ_type                      = context.src1->type;
        চাবি.output_type                     = context.dst->type;

        auto it = im2col_pipelines.find(চাবি);
        যদি (it != im2col_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        auto ঠেলো_type_defines = [&](ধ্রুৱ ৱর্ণ * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            যদি (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } অন্যথা যদি (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } অন্যথা {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ঠেলো_type_defines("INPUT_TYPE", চাবি.প্ৰবেশ_type);
        ঠেলো_type_defines("OUTPUT_TYPE", চাবি.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[চাবি]    = pipeline;
        ঘূৰাই im2col_pipelines[চাবি];
    }

    webgpu_pipeline get_upscale_pipeline(ধ্রুৱ ggml_webgpu_shader_lib_context & context) {
        ধ্রুৱ uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ধ্রুৱ uint32_t base_mode  = mode_flags & 0xFFu;
        ধ্রুৱ তার্কিক     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_চাবি চাবি = {};
        চাবি.প্ৰবেশ_type                       = context.src0->type;
        চাবি.output_type                      = context.dst->type;
        চাবি.base_mode                        = base_mode;
        চাবি.antialias                        = antialias;

        auto it = upscale_pipelines.find(চাবি);
        যদি (it != upscale_pipelines.end()) {
            ঘূৰাই it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        যদি (চাবি.প্ৰবেশ_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } অন্যথা {
            variant += "_src_f32";
        }

        যদি (চাবি.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } অন্যথা {
            variant += "_dst_f32";
        }

        ৱিকল্প (base_mode) {
            পক্ষ GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                ভঙ্গ;
            পক্ষ GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                ভঙ্গ;
            পক্ষ GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                ভঙ্গ;
            স্ৱভাৱ:
                GGML_ABORT("Unsupported upscale mode");
        }

        যদি (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_upscale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[চাবি]   = pipeline;
        ঘূৰাই upscale_pipelines[চাবি];
    }

  গুপ্ত:
    স্থির webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
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
        ঘূৰাই { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
