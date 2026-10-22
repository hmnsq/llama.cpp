#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-wgsl-shaders.hpp"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "pre_wgsl.hpp"

#അന്തര്ഭാവ <webgpu/webgpu_cpp.h>

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <vector>

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
template <typename T> വരിയിൽ ശൂന്യം ggml_webgpu_hash_combine(size_t & seed, സ്ഥിരം T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
വരിയിൽ uintptr_t ggml_webgpu_tensor_addr(സ്ഥിരം ggml_tensor * tensor) {
    സ്ഥിരം ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    തിരികെ (uintptr_t) base_tensor->data + tensor->view_offs;
}

വരിയിൽ തർക്കം ggml_webgpu_tensor_equal(സ്ഥിരം ggml_tensor * a, സ്ഥിരം ggml_tensor * b) {
    തിരികെ a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

ഘടന ggml_webgpu_shader_lib_context {
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
    തർക്കം        supports_subgroups       = അസത്യം;
    തർക്കം        supports_subgroup_matrix = അസത്യം;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_size        = 0;
    uint32_t    max_subgroup_size        = 0;
    തർക്കം        supports_dot_product     = അസത്യം;
    std::string vendor;
};

ഘടന webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<ശൂന്യം> context = nullptr;
};

ഘടന ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 0;
    തർക്കം     inplace = അസത്യം;
};

ഘടന ggml_webgpu_get_rows_shader_decisions {
    uint32_t wg_size    = 0;
    തർക്കം     vectorized = അസത്യം;
};

ഘടന ggml_webgpu_ദ്വിമാന_shader_decisions {
    uint32_t wg_size     = 0;
    തർക്കം     inplace     = അസത്യം;
    തർക്കം     overlap     = അസത്യം;
    തർക്കം     src_overlap = അസത്യം;
};

ഘടന ggml_webgpu_glu_shader_decisions {
    uint32_t wg_size     = 0;
    തർക്കം     src_overlap = അസത്യം;
};

ഘടന ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<ശൂന്യം> decisions;
};

ഘടന ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_size;
    uint32_t tokens_per_wg;
};

ഘടന ggml_webgpu_ssm_scan_pipeline_താക്കോൽ {
    പൂർണ്ണസംഖ്യ     type;
    പൂർണ്ണസംഖ്യ     d_state;
    uint8_t xbc_overlap;
    തർക്കം    a_overlap;
    തർക്കം    ids_overlap;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_ssm_scan_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

ഘടന ggml_webgpu_ssm_scan_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_ssm_scan_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.d_state);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.xbc_overlap);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.a_overlap);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.ids_overlap);
        തിരികെ seed;
    }
};

ഘടന ggml_webgpu_ssm_scan_shader_decisions {
    uint32_t wg_size;
    uint32_t tokens_per_tile;
    uint8_t  xbc_overlap = 0;
    തർക്കം     a_overlap   = അസത്യം;
    തർക്കം     ids_overlap = അസത്യം;
};

/** Argsort **/

ഘടന ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes;
    int32_t  order;
};

/** Set Rows **/

ഘടന ggml_webgpu_set_rows_pipeline_താക്കോൽ {
    പൂർണ്ണസംഖ്യ src0_type;
    പൂർണ്ണസംഖ്യ dst_type;
    പൂർണ്ണസംഖ്യ vec4;
    പൂർണ്ണസംഖ്യ i64_idx;
    പൂർണ്ണസംഖ്യ pair_blocks;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_set_rows_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

ഘടന ggml_webgpu_set_rows_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_set_rows_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src0_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.dst_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.vec4);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.i64_idx);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.pair_blocks);
        തിരികെ seed;
    }
};

ഘടന ggml_webgpu_set_rows_shader_decisions {
    തർക്കം     vec4;
    തർക്കം     i64_idx;
    തർക്കം     pair_blocks;
    uint32_t wg_size;
};

/** Set **/

ഘടന ggml_webgpu_set_pipeline_താക്കോൽ {
    ggml_type type;
    തർക്കം      inplace;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_set_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ type == other.type && inplace == other.inplace;
    }
};

ഘടന ggml_webgpu_set_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_set_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.inplace);
        തിരികെ seed;
    }
};

/** Get Rows **/

ഘടന ggml_webgpu_get_rows_pipeline_താക്കോൽ {
    ggml_type src_type;
    പൂർണ്ണസംഖ്യ       vectorized;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_get_rows_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ src_type == other.src_type && vectorized == other.vectorized;
    }
};

ഘടന ggml_webgpu_get_rows_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_get_rows_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.vectorized);
        തിരികെ seed;
    }
};

/** Row Norm **/

ഘടന ggml_webgpu_row_norm_pipeline_താക്കോൽ {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    തർക്കം      inplace;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_row_norm_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

ഘടന ggml_webgpu_row_norm_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_row_norm_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.op);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.dst_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.inplace);
        തിരികെ seed;
    }
};

/** RMS_NORM + MUL **/

ഘടന ggml_webgpu_rms_norm_mul_pipeline_താക്കോൽ {
    തർക്കം inplace;      // rn_src == dst
    തർക്കം overlap;      // mul_src == dst
    തർക്കം src_overlap;  // rn_src binding overlaps mul_src binding

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_rms_norm_mul_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

ഘടന ggml_webgpu_rms_norm_mul_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_rms_norm_mul_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.inplace);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.overlap);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src_overlap);
        തിരികെ seed;
    }
};

ഘടന ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    തർക്കം     inplace     = അസത്യം;
    തർക്കം     overlap     = അസത്യം;
    തർക്കം     src_overlap = അസത്യം;
};

/** Pad **/
ഘടന ggml_webgpu_pad_pipeline_താക്കോൽ {
    തർക്കം circular;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_pad_pipeline_താക്കോൽ & other) സ്ഥിരം { തിരികെ circular == other.circular; }
};

ഘടന ggml_webgpu_pad_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_pad_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.circular);
        തിരികെ seed;
    }
};

/** Solve Tri **/
ഘടന ggml_webgpu_solve_tri_pipeline_താക്കോൽ {
    പൂർണ്ണസംഖ്യ type;
    പൂർണ്ണസംഖ്യ n;
    പൂർണ്ണസംഖ്യ k;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_solve_tri_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ type == other.type && n == other.n && k == other.k;
    }
};

ഘടന ggml_webgpu_solve_tri_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_solve_tri_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.n);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.k);
        തിരികെ seed;
    }
};

/** SSM Conv **/
ഘടന ggml_webgpu_ssm_conv_pipeline_താക്കോൽ {
    പൂർണ്ണസംഖ്യ type;
    പൂർണ്ണസംഖ്യ vectorized;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_ssm_conv_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
ഘടന ggml_webgpu_conv2d_pipeline_താക്കോൽ {
    ggml_type weight_type;
    ggml_type നിവേശം_type;
    ggml_type output_type;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_conv2d_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ weight_type == other.weight_type && നിവേശം_type == other.നിവേശം_type && output_type == other.output_type;
    }
};

ഘടന ggml_webgpu_conv2d_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_conv2d_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.weight_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.നിവേശം_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.output_type);
        തിരികെ seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
ഘടന ggml_webgpu_conv2d_dw_pipeline_താക്കോൽ {
    ggml_type weight_type;
    ggml_type നിവേശം_type;
    ggml_type output_type;
    തർക്കം      whcn;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_conv2d_dw_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ weight_type == other.weight_type && നിവേശം_type == other.നിവേശം_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

ഘടന ggml_webgpu_conv2d_dw_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_conv2d_dw_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.weight_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.നിവേശം_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.output_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.whcn);
        തിരികെ seed;
    }
};

/** Im2Col **/
ഘടന ggml_webgpu_im2col_pipeline_താക്കോൽ {
    ggml_type നിവേശം_type;
    ggml_type output_type;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_im2col_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ നിവേശം_type == other.നിവേശം_type && output_type == other.output_type;
    }
};

ഘടന ggml_webgpu_im2col_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_im2col_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.നിവേശം_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.output_type);
        തിരികെ seed;
    }
};

/** Gated Delta Net **/
ഘടന ggml_webgpu_gated_delta_net_pipeline_താക്കോൽ {
    പൂർണ്ണസംഖ്യ  type;
    പൂർണ്ണസംഖ്യ  s_v;
    പൂർണ്ണസംഖ്യ  kda;
    തർക്കം fused_cache;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_gated_delta_net_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

ഘടന ggml_webgpu_gated_delta_net_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_gated_delta_net_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.s_v);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.kda);
        തിരികെ seed;
    }
};

ഘടന ggml_webgpu_ssm_conv_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_ssm_conv_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.vectorized);
        തിരികെ seed;
    }
};

/** Scale **/

ഘടന ggml_webgpu_scale_pipeline_താക്കോൽ {
    പൂർണ്ണസംഖ്യ inplace;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_scale_pipeline_താക്കോൽ & other) സ്ഥിരം { തിരികെ inplace == other.inplace; }
};

ഘടന ggml_webgpu_scale_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_scale_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.inplace);
        തിരികെ seed;
    }
};

/** Upscale **/

ഘടന ggml_webgpu_upscale_pipeline_താക്കോൽ {
    ggml_type നിവേശം_type;
    ggml_type output_type;
    uint32_t  base_mode;
    തർക്കം      antialias;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_upscale_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ നിവേശം_type == other.നിവേശം_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

ഘടന ggml_webgpu_upscale_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_upscale_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.നിവേശം_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.output_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.base_mode);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.antialias);
        തിരികെ seed;
    }
};

/** Concat **/

ഘടന ggml_webgpu_concat_pipeline_താക്കോൽ {
    പൂർണ്ണസംഖ്യ  type;
    തർക്കം src_overlap;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_concat_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ type == other.type && src_overlap == other.src_overlap;
    }
};

ഘടന ggml_webgpu_concat_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_concat_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src_overlap);
        തിരികെ seed;
    }
};

/** Repeat **/

ഘടന ggml_webgpu_repeat_pipeline_താക്കോൽ {
    പൂർണ്ണസംഖ്യ type;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_repeat_pipeline_താക്കോൽ & other) സ്ഥിരം { തിരികെ type == other.type; }
};

ഘടന ggml_webgpu_repeat_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_repeat_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        തിരികെ seed;
    }
};

/** Binary **/

ഘടന ggml_webgpu_ദ്വിമാന_pipeline_താക്കോൽ {
    പൂർണ്ണസംഖ്യ  type;
    പൂർണ്ണസംഖ്യ  op;
    തർക്കം inplace;
    തർക്കം overlap;
    തർക്കം src_overlap;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_ദ്വിമാന_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

ഘടന ggml_webgpu_ദ്വിമാന_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_ദ്വിമാന_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.op);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.inplace);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.overlap);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src_overlap);
        തിരികെ seed;
    }
};

/* Add_Id */

ഘടന ggml_webgpu_add_id_pipeline_താക്കോൽ {
    തർക്കം inplace;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_add_id_pipeline_താക്കോൽ & other) സ്ഥിരം { തിരികെ inplace == other.inplace; }
};

ഘടന ggml_webgpu_add_id_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_add_id_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.inplace);
        തിരികെ seed;
    }
};

/** Unary **/

ഘടന ggml_webgpu_unary_pipeline_താക്കോൽ {
    പൂർണ്ണസംഖ്യ           type;
    പൂർണ്ണസംഖ്യ           op;
    തർക്കം          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    തർക്കം          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_unary_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

ഘടന ggml_webgpu_unary_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_unary_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.op);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.is_unary);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.inplace);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.ttype);
        തിരികെ seed;
    }
};

/** FlashAttention */

ഘടന ggml_webgpu_flash_attn_common_pipeline_താക്കോൽ {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    തർക്കം      k_direct;
    തർക്കം      v_direct;
    തർക്കം      kv_overlap;
    തർക്കം      has_mask;
    തർക്കം      has_sinks;
    തർക്കം      uses_logit_softcap;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_flash_attn_common_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

വരിയിൽ ശൂന്യം ggml_webgpu_flash_attn_hash_common_pipeline_താക്കോൽ(size_t &                                           seed,
                                                            സ്ഥിരം ggml_webgpu_flash_attn_common_pipeline_താക്കോൽ & താക്കോൽ) {
    ggml_webgpu_hash_combine(seed, താക്കോൽ.q_type);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.k_type);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.v_type);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.dst_type);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.head_dim_qk);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.head_dim_v);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.k_direct);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.v_direct);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.kv_overlap);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.has_mask);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.has_sinks);
    ggml_webgpu_hash_combine(seed, താക്കോൽ.uses_logit_softcap);
}

ഘടന ggml_webgpu_flash_attn_vec_pipeline_താക്കോൽ {
    ggml_webgpu_flash_attn_common_pipeline_താക്കോൽ common;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_flash_attn_vec_pipeline_താക്കോൽ & other) സ്ഥിരം { തിരികെ common == other.common; }
};

ഘടന ggml_webgpu_flash_attn_vec_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_flash_attn_vec_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_താക്കോൽ(seed, താക്കോൽ.common);
        തിരികെ seed;
    }
};

ഘടന ggml_webgpu_flash_attn_pipeline_താക്കോൽ {
    ggml_webgpu_flash_attn_common_pipeline_താക്കോൽ common;
    തർക്കം                                       use_sg_matrix;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_flash_attn_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

ഘടന ggml_webgpu_flash_attn_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_flash_attn_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_താക്കോൽ(seed, താക്കോൽ.common);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.use_sg_matrix);
        തിരികെ seed;
    }
};

ഘടന ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

ഘടന ggml_webgpu_flash_attn_decisions {
    തർക്കം     use_sg_matrix = അസത്യം;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_size       = 0;
};

വരിയിൽ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
വരിയിൽ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

വരിയിൽ size_t ggml_webgpu_flash_attn_tensor_offset(സ്ഥിരം ggml_tensor * tensor) {
    constexpr uintptr_t ptr_base_addr = 0x1000u;
    സ്ഥിരം ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
    തിരികെ reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
}

വരിയിൽ തർക്കം ggml_webgpu_flash_attn_float_vec4_aligned(സ്ഥിരം ggml_tensor * K, size_t storage_offset_alignment) {
    സ്ഥിരം uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_size(K->type));
    തിരികെ offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

വരിയിൽ തർക്കം ggml_webgpu_flash_attn_float_vec4_aligned(സ്ഥിരം ggml_tensor * K,
                                                      സ്ഥിരം ggml_tensor * V,
                                                      size_t              storage_offset_alignment) {
    തിരികെ ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

വരിയിൽ തർക്കം ggml_webgpu_flash_attn_k_direct(സ്ഥിരം ggml_tensor * Q, സ്ഥിരം ggml_tensor * K, uint32_t kv_direct_align) {
    തിരികെ (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

വരിയിൽ തർക്കം ggml_webgpu_flash_attn_v_direct(സ്ഥിരം ggml_tensor * Q, സ്ഥിരം ggml_tensor * V, uint32_t kv_direct_align) {
    തിരികെ ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

വരിയിൽ ggml_webgpu_flash_attn_common_pipeline_താക്കോൽ ggml_webgpu_flash_attn_make_common_pipeline_താക്കോൽ(
    സ്ഥിരം ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    തർക്കം                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_താക്കോൽ താക്കോൽ = {};
    താക്കോൽ.q_type                                     = context.src0->type;
    താക്കോൽ.k_type                                     = context.src1->type;
    താക്കോൽ.v_type                                     = context.src2->type;
    താക്കോൽ.dst_type                                   = context.dst->type;
    താക്കോൽ.head_dim_qk                                = (uint32_t) context.src0->ne[0];
    താക്കോൽ.head_dim_v                                 = (uint32_t) context.src2->ne[0];
    താക്കോൽ.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    താക്കോൽ.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    താക്കോൽ.kv_overlap         = kv_overlap;
    താക്കോൽ.has_mask           = context.src3 != nullptr;
    താക്കോൽ.has_sinks          = context.src4 != nullptr;
    താക്കോൽ.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    തിരികെ താക്കോൽ;
}

വരിയിൽ std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    സ്ഥിരം ggml_webgpu_flash_attn_common_pipeline_താക്കോൽ & താക്കോൽ,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_size) {
    std::vector<std::string> defines;

    തിരഞ്ഞെടുപ്പ് (താക്കോൽ.k_type) {
        അവസ്ഥ GGML_TYPE_F32:
            defines.push_back("K_F32");
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            defines.push_back("K_F16");
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_0:
            defines.push_back("K_Q4_0");
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q8_0:
            defines.push_back("K_Q8_0");
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(താക്കോൽ.k_type);

    തിരഞ്ഞെടുപ്പ് (താക്കോൽ.v_type) {
        അവസ്ഥ GGML_TYPE_F32:
            defines.push_back("V_F32");
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            defines.push_back("V_F16");
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_0:
            defines.push_back("V_Q4_0");
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q8_0:
            defines.push_back("V_Q8_0");
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(താക്കോൽ.v_type);

    തിരഞ്ഞെടുപ്പ് (താക്കോൽ.q_type) {
        അവസ്ഥ GGML_TYPE_F32:
            defines.push_back("Q_F32");
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            defines.push_back("Q_F16");
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(താക്കോൽ.q_type);

    തിരഞ്ഞെടുപ്പ് (താക്കോൽ.dst_type) {
        അവസ്ഥ GGML_TYPE_F32:
            defines.push_back("DST_F32");
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            defines.push_back("DST_F16");
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(താക്കോൽ.dst_type);

    എങ്കിൽ (താക്കോൽ.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    എങ്കിൽ (താക്കോൽ.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    എങ്കിൽ (താക്കോൽ.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    എങ്കിൽ (താക്കോൽ.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    എങ്കിൽ (താക്കോൽ.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    എങ്കിൽ (താക്കോൽ.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(താക്കോൽ.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(താക്കോൽ.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(താക്കോൽ.head_dim_v));
    variant += std::string("_hsv") + std::to_string(താക്കോൽ.head_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    എങ്കിൽ (ggml_is_quantized(താക്കോൽ.k_type) || ggml_is_quantized(താക്കോൽ.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        എങ്കിൽ (ggml_is_quantized(താക്കോൽ.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        എങ്കിൽ (ggml_is_quantized(താക്കോൽ.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    തിരികെ defines;
}

ഘടന ggml_webgpu_flash_attn_vec_reduce_pipeline_താക്കോൽ {
    uint32_t  head_dim_v;
    uint32_t  wg_size;
    ggml_type dst_type;
};

ഘടന ggml_webgpu_flash_attn_vec_reduce_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_flash_attn_vec_reduce_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.head_dim_v);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.wg_size);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.dst_type);
        തിരികെ seed;
    }
};

വരിയിൽ തർക്കം operator==(സ്ഥിരം ggml_webgpu_flash_attn_vec_reduce_pipeline_താക്കോൽ & lhs,
                       സ്ഥിരം ggml_webgpu_flash_attn_vec_reduce_pipeline_താക്കോൽ & rhs) {
    തിരികെ lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size && lhs.dst_type == rhs.dst_type;
}

ഘടന ggml_webgpu_flash_attn_blk_pipeline_താക്കോൽ {
    uint32_t kv_tile;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_flash_attn_blk_pipeline_താക്കോൽ & other) സ്ഥിരം { തിരികെ kv_tile == other.kv_tile; }
};

ഘടന ggml_webgpu_flash_attn_blk_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_flash_attn_blk_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.kv_tile);
        തിരികെ seed;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
വരിയിൽ size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  തർക്കം     has_mask,
                                                  തർക്കം     kv_direct) {
    സ്ഥിരം uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * head_dim_qk;        // q_shmem
    എങ്കിൽ (!kv_direct) {
        f32_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f32_elems += q_tile * head_dim_v;         // o_shmem
    എങ്കിൽ (has_mask) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    തിരികെ f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

വരിയിൽ uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   limit_bytes,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t head_dim_qk,
                                                   uint32_t head_dim_v,
                                                   തർക്കം     has_mask,
                                                   തർക്കം     kv_direct) {
    സ്ഥിരം size_t base_q_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 0, head_dim_qk, head_dim_v, has_mask, kv_direct);
    എങ്കിൽ (limit_bytes <= base_q_bytes) {
        തിരികെ 0;
    }
    സ്ഥിരം size_t one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, head_dim_qk, head_dim_v, has_mask, kv_direct);
    സ്ഥിരം size_t bytes_per_kv = one_kv_bytes - base_q_bytes;
    എങ്കിൽ (bytes_per_kv == 0) {
        തിരികെ 0;
    }
    സ്ഥിരം size_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    തിരികെ (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

വരിയിൽ uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_limit_bytes,
                                                       uint32_t head_dim_qk,
                                                       uint32_t head_dim_v,
                                                       തർക്കം     has_mask,
                                                       തർക്കം     kv_direct) {
    സ്ഥിരം uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_limit_bytes, 1u, 1u, head_dim_qk, head_dim_v, has_mask, kv_direct);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    എങ്കിൽ (kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        ആകുമ്പോൾ (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    തിരികെ kv_tile;
}

വരിയിൽ തർക്കം ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(തർക്കം                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                സ്ഥിരം ggml_tensor * Q,
                                                                സ്ഥിരം ggml_tensor * V) {
    തിരികെ supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

ഘടന ggml_webgpu_mul_mat_vec_pipeline_താക്കോൽ {
    ggml_type src0_type;
    ggml_type src1_type;
    പൂർണ്ണസംഖ്യ       vectorized;
    uint32_t  അക്കം_cols;
    തർക്കം      use_mmvq;
    തർക്കം      src_overlap;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_mul_mat_vec_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               അക്കം_cols == other.അക്കം_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

ഘടന ggml_webgpu_mul_mat_vec_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_mul_mat_vec_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src0_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src1_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.vectorized);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.അക്കം_cols);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.use_mmvq);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src_overlap);
        തിരികെ seed;
    }
};

ഘടന ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
    തർക്കം     src_overlap = അസത്യം;
};

ഘടന ggml_webgpu_quantize_q8_pipeline_താക്കോൽ {
    ggml_type src0_type;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_quantize_q8_pipeline_താക്കോൽ & other) സ്ഥിരം { തിരികെ src0_type == other.src0_type; }
};

ഘടന ggml_webgpu_quantize_q8_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_quantize_q8_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src0_type);
        തിരികെ seed;
    }
};

ഘടന ggml_webgpu_mul_mat_pipeline_താക്കോൽ {
    ggml_type src0_type;
    ggml_type src1_type;
    പൂർണ്ണസംഖ്യ       vectorized;
    പൂർണ്ണസംഖ്യ       use_subgroup_matrix;
    തർക്കം      src_overlap;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_mul_mat_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

ഘടന ggml_webgpu_mul_mat_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_mul_mat_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src0_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src1_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.vectorized);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src_overlap);
        തിരികെ seed;
    }
};

ഘടന ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    പൂർണ്ണസംഖ്യ      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_size;
    തർക്കം     src_overlap = അസത്യം;
};

/** MUL_MAT_ID **/

ഘടന ggml_webgpu_mul_mat_id_gather_pipeline_താക്കോൽ {
    uint32_t n_experts;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_mul_mat_id_gather_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ n_experts == other.n_experts;
    }
};

ഘടന ggml_webgpu_mul_mat_id_gather_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_mul_mat_id_gather_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.n_experts);
        തിരികെ seed;
    }
};

ഘടന ggml_webgpu_mul_mat_id_pipeline_താക്കോൽ {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  അക്കം_cols;
    പൂർണ്ണസംഖ്യ       vectorized;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_mul_mat_id_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               അക്കം_cols == other.അക്കം_cols && vectorized == other.vectorized;
    }
};

ഘടന ggml_webgpu_mul_mat_id_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_mul_mat_id_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src0_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src1_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.n_experts);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.അക്കം_cols);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.vectorized);
        തിരികെ seed;
    }
};

/** Cpy **/

ഘടന ggml_webgpu_cpy_pipeline_താക്കോൽ {
    ggml_type src_type;
    ggml_type dst_type;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_cpy_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ src_type == other.src_type && dst_type == other.dst_type;
    }
};

ഘടന ggml_webgpu_cpy_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_cpy_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.dst_type);
        തിരികെ seed;
    }
};

/** Glu **/

ഘടന ggml_webgpu_glu_pipeline_താക്കോൽ {
    ggml_glu_op glu_op;
    ggml_type   type;
    തർക്കം        split;
    തർക്കം        src_overlap;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_glu_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

ഘടന ggml_webgpu_glu_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_glu_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.glu_op);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.split);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.src_overlap);
        തിരികെ seed;
    }
};

/** Rope **/

ഘടന ggml_webgpu_rope_pipeline_താക്കോൽ {
    ggml_type type;
    തർക്കം      inplace;
    തർക്കം      has_ff;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_rope_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

ഘടന ggml_webgpu_rope_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_rope_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.inplace);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.has_ff);
        തിരികെ seed;
    }
};

/** SoftMax **/

ഘടന ggml_webgpu_soft_max_pipeline_താക്കോൽ {
    ggml_type mask_type;
    തർക്കം      has_mask;
    തർക്കം      has_sink;
    തർക്കം      inplace;

    തർക്കം operator==(സ്ഥിരം ggml_webgpu_soft_max_pipeline_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

ഘടന ggml_webgpu_soft_max_pipeline_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം ggml_webgpu_soft_max_pipeline_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, താക്കോൽ.mask_type);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.has_mask);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.has_sink);
        ggml_webgpu_hash_combine(seed, താക്കോൽ.inplace);
        തിരികെ seed;
    }
};

/** MMVQ **/

വരിയിൽ തർക്കം ggml_webgpu_can_use_mmvq(സ്ഥിരം ggml_tensor * src0,
                                     സ്ഥിരം ggml_tensor * src1,
                                     തർക്കം                supports_dot_product,
                                     സ്ഥിരം std::string & vendor) {
    എങ്കിൽ (src1->ne[1] <= 4) {
        തർക്കം supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        എങ്കിൽ (supports_dp4a && supports_dot_product) {
            തിരഞ്ഞെടുപ്പ് (src1->type) {
                അവസ്ഥ GGML_TYPE_F32:
                    തിരഞ്ഞെടുപ്പ് (src0->type) {
                        അവസ്ഥ GGML_TYPE_Q1_0:
                        അവസ്ഥ GGML_TYPE_Q4_0:
                        അവസ്ഥ GGML_TYPE_Q4_1:
                        അവസ്ഥ GGML_TYPE_Q5_0:
                        അവസ്ഥ GGML_TYPE_Q5_1:
                        അവസ്ഥ GGML_TYPE_Q8_0:
                        അവസ്ഥ GGML_TYPE_MXFP4:
                        അവസ്ഥ GGML_TYPE_Q2_K:
                        അവസ്ഥ GGML_TYPE_Q3_K:
                        അവസ്ഥ GGML_TYPE_Q4_K:
                        അവസ്ഥ GGML_TYPE_Q5_K:
                        അവസ്ഥ GGML_TYPE_Q6_K:
                            തിരികെ src0->ne[0] % 4 == 0;
                        സ്ഥിരസ്ഥിതി:
                            നിർത്തുക;
                    }
                    നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    നിർത്തുക;
            }
        }
    }
    തിരികെ അസത്യം;
}

വർഗ്ഗം ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<പൂർണ്ണസംഖ്യ, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<പൂർണ്ണസംഖ്യ, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<പൂർണ്ണസംഖ്യ, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<പൂർണ്ണസംഖ്യ, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<പൂർണ്ണസംഖ്യ, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_താക്കോൽ_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_താക്കോൽ_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_unary_pipeline_താക്കോൽ_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_scale_pipeline_താക്കോൽ_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_താക്കോൽ_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_താക്കോൽ_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_താക്കോൽ_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_താക്കോൽ,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_താക്കോൽ_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_pad_pipeline_താക്കോൽ_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_ദ്വിമാന_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_ദ്വിമാന_pipeline_താക്കോൽ_hash>
        ദ്വിമാന_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_add_id_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_add_id_pipeline_താക്കോൽ_hash>
        add_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_concat_pipeline_താക്കോൽ_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_repeat_pipeline_താക്കോൽ_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_താക്കോൽ,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_താക്കോൽ_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_താക്കോൽ_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_താക്കോൽ,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_താക്കോൽ_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_താക്കോൽ,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_താക്കോൽ_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_താക്കോൽ_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_താക്കോൽ_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_താക്കോൽ_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_താക്കോൽ,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_താക്കോൽ_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_താക്കോൽ_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_താക്കോൽ_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_താക്കോൽ_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_set_pipeline_താക്കോൽ_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_cpy_pipeline_താക്കോൽ_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_glu_pipeline_താക്കോൽ_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_rope_pipeline_താക്കോൽ_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_താക്കോൽ_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_താക്കോൽ_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_താക്കോൽ_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_im2col_pipeline_താക്കോൽ_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_താക്കോൽ,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_താക്കോൽ_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_താക്കോൽ, webgpu_pipeline, ggml_webgpu_upscale_pipeline_താക്കോൽ_hash>
        upscale_pipelines;

  പൊതു:
    ggml_webgpu_shader_lib(wgpu::Device device) { ഇത്->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        auto it = sum_rows_pipelines.find(1);
        എങ്കിൽ (it != sum_rows_pipelines.end()) {
            തിരികെ it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        തിരികെ sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.op                                = context.dst->op;
        താക്കോൽ.src_type                          = context.src0->type;
        താക്കോൽ.dst_type                          = context.dst->type;
        താക്കോൽ.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = row_norm_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != row_norm_pipelines.end()) {
            തിരികെ it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.op) {
            അവസ്ഥ GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                നിർത്തുക;
            അവസ്ഥ GGML_OP_NORM:
                defines.push_back("NORM");
                variant = "norm";
                നിർത്തുക;
            അവസ്ഥ GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        എങ്കിൽ (താക്കോൽ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        എങ്കിൽ (താക്കോൽ.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        എങ്കിൽ (താക്കോൽ.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        സ്ഥിരം uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = താക്കോൽ.inplace;
        row_norm_pipelines[താക്കോൽ]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[താക്കോൽ].context = decisions;
        തിരികെ row_norm_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_argmax_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        തർക്കം vec4 = context.src0->ne[0] % 4 == 0;

        auto it = argmax_pipelines.find(vec4);
        എങ്കിൽ (it != argmax_pipelines.end()) {
            തിരികെ it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        എങ്കിൽ (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        auto processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        തിരികെ argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        സ്ഥിരം തർക്കം                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_താക്കോൽ താക്കോൽ       = {};
        താക്കോൽ.src0_type                               = context.src0->type;
        താക്കോൽ.dst_type                                = context.dst->type;
        താക്കോൽ.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        താക്കോൽ.i64_idx     = context.src1->type == GGML_TYPE_I64;
        താക്കോൽ.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_size(context.dst->type)) % 2 == 0);

        auto it = set_rows_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != set_rows_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        തിരഞ്ഞെടുപ്പ് (context.dst->type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_Q8_0:
                defines.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_Q4_0:
                defines.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        എങ്കിൽ (context.src0->type == GGML_TYPE_F16) {
            defines.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        എങ്കിൽ (താക്കോൽ.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        എങ്കിൽ (താക്കോൽ.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        എങ്കിൽ (താക്കോൽ.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        സ്ഥിരം auto & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        auto         processed          = preprocessor.preprocess(shader_source, defines);
        auto         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = താക്കോൽ.vec4;
        decisions->i64_idx              = താക്കോൽ.i64_idx;
        decisions->pair_blocks          = താക്കോൽ.pair_blocks;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[താക്കോൽ]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[താക്കോൽ].context = decisions;
        തിരികെ set_rows_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_set_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.type                         = context.dst->type;
        താക്കോൽ.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = set_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != set_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for set shader");
        }

        എങ്കിൽ (താക്കോൽ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_set, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = താക്കോൽ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[താക്കോൽ]       = pipeline;
        തിരികെ set_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_cumsum_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        auto it = cumsum_pipelines.find(1);
        എങ്കിൽ (it != cumsum_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        തിരികെ cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        തർക്കം          is_മുകൾ_k = context.dst->op == GGML_OP_മുകൾ_K;
        // ascending order is 0, descending order is 1
        സ്ഥിരം int32_t order =
            is_മുകൾ_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_pipelines.find(order);
        എങ്കിൽ (it != argsort_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        ആകുമ്പോൾ (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        auto decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        തിരികെ argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        തർക്കം          is_മുകൾ_k = context.dst->op == GGML_OP_മുകൾ_K;
        // ascending order is 0, descending order is 1
        സ്ഥിരം int32_t order =
            is_മുകൾ_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_merge_pipelines.find(order);
        എങ്കിൽ (it != argsort_merge_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        തിരികെ argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, തർക്കം vec4_aligned) {
        സ്ഥിരം തർക്കം vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.src_type                          = context.src0->type;
        താക്കോൽ.vectorized                        = (പൂർണ്ണസംഖ്യ) vectorized;

        auto it = get_rows_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != get_rows_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        സ്ഥിരം ഘടന ggml_type_traits * type_traits = ggml_get_type_traits(താക്കോൽ.src_type);
        സ്ഥിരം അക്ഷരം *                    type_str    = type_traits->type_name;

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.src_type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                എങ്കിൽ (താക്കോൽ.vectorized) {
                    defines.push_back("F32_VEC");
                    defines.push_back("SRC_TYPE=vec4<f32>");
                    defines.push_back("DST_TYPE=vec4<f32>");
                    defines.push_back("BLOCK_SIZE=4u");
                } അല്ലെങ്കിൽ {
                    defines.push_back("F32");
                    defines.push_back("SRC_TYPE=f32");
                    defines.push_back("DST_TYPE=f32");
                    defines.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_BF16:
                defines.push_back("BF16");
                defines.push_back("SRC_TYPE=u32");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    തിരഞ്ഞെടുപ്പ് (താക്കോൽ.src_type) {
                        അവസ്ഥ GGML_TYPE_Q1_0:
                        അവസ്ഥ GGML_TYPE_Q2_0:
                        അവസ്ഥ GGML_TYPE_Q4_0:
                        അവസ്ഥ GGML_TYPE_Q5_0:
                        അവസ്ഥ GGML_TYPE_Q8_0:
                        അവസ്ഥ GGML_TYPE_Q3_K:
                        അവസ്ഥ GGML_TYPE_Q6_K:
                        അവസ്ഥ GGML_TYPE_IQ2_XXS:
                        അവസ്ഥ GGML_TYPE_IQ2_XS:
                        അവസ്ഥ GGML_TYPE_IQ2_S:
                        അവസ്ഥ GGML_TYPE_IQ3_XXS:
                        അവസ്ഥ GGML_TYPE_IQ3_S:
                        അവസ്ഥ GGML_TYPE_IQ1_S:
                        അവസ്ഥ GGML_TYPE_IQ4_NL:
                        അവസ്ഥ GGML_TYPE_MXFP4:
                        അവസ്ഥ GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                നിർത്തുക;
                            }
                        സ്ഥിരസ്ഥിതി:
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

                    എങ്കിൽ (താക്കോൽ.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } അല്ലെങ്കിൽ എങ്കിൽ ((താക്കോൽ.src_type >= GGML_TYPE_Q4_0 && താക്കോൽ.src_type <= GGML_TYPE_Q8_1) ||
                               താക്കോൽ.src_type == GGML_TYPE_IQ4_NL || താക്കോൽ.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.src_type == GGML_TYPE_Q2_0 || താക്കോൽ.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } അല്ലെങ്കിൽ {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    നിർത്തുക;
                }
        }

        എങ്കിൽ (താക്കോൽ.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        auto decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[താക്കോൽ]  = pipeline;
        തിരികെ get_rows_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_scale_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = scale_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != scale_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        എങ്കിൽ (താക്കോൽ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_scale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = താക്കോൽ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[താക്കോൽ]     = pipeline;
        തിരികെ scale_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_solve_tri_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.type                               = context.dst->type;
        താക്കോൽ.n                                  = (പൂർണ്ണസംഖ്യ) context.src0->ne[0];
        താക്കോൽ.k                                  = (പൂർണ്ണസംഖ്യ) context.src1->ne[0];

        auto it = solve_tri_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != solve_tri_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                variant += "_f32";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        സ്ഥിരം uint32_t wg_size       = std::min((uint32_t) താക്കോൽ.n, context.max_wg_size);
        സ്ഥിരം uint32_t k_tile        = wg_size;
        സ്ഥിരം uint32_t bytes_per_row = ((uint32_t) താക്കോൽ.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        സ്ഥിരം uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(താക്കോൽ.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        auto processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[താക്കോൽ] = pipeline;
        തിരികെ solve_tri_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_ssm_conv_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.type                              = context.dst->type;
        താക്കോൽ.vectorized                        = context.src1->ne[0] == 4;

        auto it = ssm_conv_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != ssm_conv_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                variant += "_f32";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        എങ്കിൽ (താക്കോൽ.vectorized) {
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
        ssm_conv_pipelines[താക്കോൽ]  = pipeline;
        തിരികെ ssm_conv_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_ssm_scan_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.type                              = context.dst->type;
        താക്കോൽ.d_state                           = (പൂർണ്ണസംഖ്യ) context.src0->ne[0];
        താക്കോൽ.xbc_overlap                       = xbc_overlap;

        auto it = ssm_scan_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != ssm_scan_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                variant += "_f32";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        സ്ഥിരം uint32_t wg_size = (uint32_t) താക്കോൽ.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        എങ്കിൽ (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } അല്ലെങ്കിൽ {
            variant += "_wg_reduce";
        }

        എങ്കിൽ (താക്കോൽ.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(താക്കോൽ.d_state);

        auto processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        auto decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = താക്കോൽ.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[താക്കോൽ]    = pipeline;
        തിരികെ ssm_scan_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.type                                     = context.dst->type;
        താക്കോൽ.s_v                                      = (പൂർണ്ണസംഖ്യ) context.src2->ne[0];
        താക്കോൽ.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        താക്കോൽ.fused_cache                              = context.dst_fuse != nullptr;

        auto it = gated_delta_net_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != gated_delta_net_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                variant += "_f32";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        എങ്കിൽ (താക്കോൽ.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        എങ്കിൽ (താക്കോൽ.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(താക്കോൽ.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(താക്കോൽ.s_v) + "u");

        auto            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[താക്കോൽ] = pipeline;
        തിരികെ gated_delta_net_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_pad_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        auto it = pad_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != pad_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        എങ്കിൽ (താക്കോൽ.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_pad, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[താക്കോൽ]       = pipeline;
        തിരികെ pad_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_quantize_q8_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.src0_type                            = context.src0->type;

        auto it = quantize_q8_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != quantize_q8_pipelines.end()) {
            തിരികെ it->second;
        }
        സ്ഥിരം അക്ഷരം *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> defines;
        std::string              variant = "quantize_q8";

        uint32_t wg_size = WEBGPU_MUL_MAT_VEC_WG_SIZE;

        defines.push_back("SRC1_INNER_TYPE=f32");
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        സ്ഥിരം ഘടന ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
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
        quantize_q8_pipelines[താക്കോൽ] = pipeline;
        തിരികെ quantize_q8_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, തർക്കം src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.src0_type                            = context.src0->type;
        താക്കോൽ.src1_type                            = context.src1->type;
        താക്കോൽ.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        താക്കോൽ.അക്കം_cols   = context.dst->ne[1];
        താക്കോൽ.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
        താക്കോൽ.src_overlap = src_overlap;

        auto it = mul_mat_vec_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != mul_mat_vec_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        സ്ഥിരം അക്ഷരം *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        തിരഞ്ഞെടുപ്പ് (context.src0->type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                {
                    // Quantized types: use helpers but accumulate in f16
                    സ്ഥിരം ഘടന ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    തിരഞ്ഞെടുപ്പ് (context.src0->type) {
                        അവസ്ഥ GGML_TYPE_Q4_0:
                        അവസ്ഥ GGML_TYPE_Q4_1:
                        അവസ്ഥ GGML_TYPE_Q5_0:
                        അവസ്ഥ GGML_TYPE_Q5_1:
                        അവസ്ഥ GGML_TYPE_Q8_0:
                            എങ്കിൽ (താക്കോൽ.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_Q1_0:
                        അവസ്ഥ GGML_TYPE_Q2_K:
                        അവസ്ഥ GGML_TYPE_Q3_K:
                        അവസ്ഥ GGML_TYPE_Q4_K:
                        അവസ്ഥ GGML_TYPE_Q5_K:
                        അവസ്ഥ GGML_TYPE_Q6_K:
                            എങ്കിൽ (താക്കോൽ.use_mmvq) {
                                defines.push_back("K_QUANTS_HANDLING");
                            }
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_IQ1_S:
                        അവസ്ഥ GGML_TYPE_IQ1_M:
                        അവസ്ഥ GGML_TYPE_IQ2_S:
                        അവസ്ഥ GGML_TYPE_IQ3_S:
                        അവസ്ഥ GGML_TYPE_IQ4_NL:
                        അവസ്ഥ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_IQ2_XXS:
                        അവസ്ഥ GGML_TYPE_IQ2_XS:
                        അവസ്ഥ GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_MXFP4:
                            defines.push_back(type_upper + "_LUT");
                            എങ്കിൽ (താക്കോൽ.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            നിർത്തുക;
                        സ്ഥിരസ്ഥിതി:
                            നിർത്തുക;
                    }
                    നിർത്തുക;
                }
        }

        // src1 type (vector)
        തിരഞ്ഞെടുപ്പ് (context.src1->type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(താക്കോൽ.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        എങ്കിൽ (താക്കോൽ.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        എങ്കിൽ (താക്കോൽ.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        എങ്കിൽ (താക്കോൽ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        എങ്കിൽ (താക്കോൽ.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(താക്കോൽ.അക്കം_cols));

        auto processed            = preprocessor.preprocess(shader_src, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = താക്കോൽ.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[താക്കോൽ] = pipeline;
        തിരികെ mul_mat_vec_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, തർക്കം src_overlap) {
        ggml_webgpu_mul_mat_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.src0_type                        = context.src0->type;
        താക്കോൽ.src1_type                        = context.src1->type;
        താക്കോൽ.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        താക്കോൽ.use_subgroup_matrix = context.supports_subgroup_matrix;
        താക്കോൽ.src_overlap         = src_overlap;

        auto it = mul_mat_fast_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != mul_mat_fast_pipelines.end()) {
            തിരികെ it->second;
        }

        സ്ഥിരം അക്ഷരം * shader_src = താക്കോൽ.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = താക്കോൽ.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        തിരഞ്ഞെടുപ്പ് (context.src1->type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        സ്ഥിരം ഘടന ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        സ്ഥിരം അക്ഷരം *                    src0_name   = src0_traits->type_name;

        തിരഞ്ഞെടുപ്പ് (context.src0->type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("TYPE_F32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("TYPE_F16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("TYPE_BF16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    തിരഞ്ഞെടുപ്പ് (context.src0->type) {
                        അവസ്ഥ GGML_TYPE_IQ1_S:
                        അവസ്ഥ GGML_TYPE_IQ1_M:
                        അവസ്ഥ GGML_TYPE_IQ4_NL:
                        അവസ്ഥ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_IQ2_XXS:
                        അവസ്ഥ GGML_TYPE_IQ2_XS:
                        അവസ്ഥ GGML_TYPE_IQ2_S:
                        അവസ്ഥ GGML_TYPE_IQ3_XXS:
                        അവസ്ഥ GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_MXFP4:
                        അവസ്ഥ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            നിർത്തുക;
                        സ്ഥിരസ്ഥിതി:
                            നിർത്തുക;
                    }

                    variant += std::string("_") + src0_name;
                    നിർത്തുക;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(താക്കോൽ.vectorized ? "VEC" : "SCALAR");

        സ്ഥിരം തർക്കം is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        എങ്കിൽ (താക്കോൽ.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } അല്ലെങ്കിൽ {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        എങ്കിൽ (താക്കോൽ.use_subgroup_matrix) {
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
        എങ്കിൽ (താക്കോൽ.vectorized) {
            variant += "_vectorized";
        }

        എങ്കിൽ (താക്കോൽ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        എങ്കിൽ (!താക്കോൽ.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = താക്കോൽ.use_subgroup_matrix;
        എങ്കിൽ (താക്കോൽ.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_size           = context.max_subgroup_size;
        } അല്ലെങ്കിൽ {
            decisions->wg_size_m       = WEBGPU_MUL_MAT_WG_SIZE_M;
            decisions->wg_size_n       = WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->wg_size         = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->mul_mat_wg_size = WEBGPU_MUL_MAT_WG_SIZE;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[താക്കോൽ] = pipeline;
        തിരികെ mul_mat_fast_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.n_experts                                  = (uint32_t) context.src0->ne[2];

        auto it = mul_mat_id_gather_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != mul_mat_id_gather_pipelines.end()) {
            തിരികെ it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(താക്കോൽ.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        auto processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[താക്കോൽ] = pipeline;
        തിരികെ pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.src0_type                           = context.src0->type;
        താക്കോൽ.src1_type                           = context.src1->type;
        താക്കോൽ.n_experts                           = context.src0->ne[2];
        താക്കോൽ.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != mul_mat_id_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        തിരഞ്ഞെടുപ്പ് (context.src1->type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        സ്ഥിരം ഘടന ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        സ്ഥിരം അക്ഷരം *                    src0_name   = src0_traits->type_name;

        തിരഞ്ഞെടുപ്പ് (context.src0->type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    തിരഞ്ഞെടുപ്പ് (context.src0->type) {
                        അവസ്ഥ GGML_TYPE_IQ1_S:
                        അവസ്ഥ GGML_TYPE_IQ1_M:
                        അവസ്ഥ GGML_TYPE_IQ4_NL:
                        അവസ്ഥ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_IQ2_XXS:
                        അവസ്ഥ GGML_TYPE_IQ2_XS:
                        അവസ്ഥ GGML_TYPE_IQ2_S:
                        അവസ്ഥ GGML_TYPE_IQ3_XXS:
                        അവസ്ഥ GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_MXFP4:
                        അവസ്ഥ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            നിർത്തുക;
                        സ്ഥിരസ്ഥിതി:
                            നിർത്തുക;
                    }

                    variant += std::string("_") + src0_name;
                    നിർത്തുക;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(താക്കോൽ.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        സ്ഥിരം uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        എങ്കിൽ (താക്കോൽ.vectorized) {
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
        mul_mat_id_pipelines[താക്കോൽ] = pipeline;
        തിരികെ mul_mat_id_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.src0_type                           = context.src0->type;
        താക്കോൽ.src1_type                           = context.src1->type;
        താക്കോൽ.n_experts                           = context.src0->ne[2];
        താക്കോൽ.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_vec_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != mul_mat_id_vec_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        സ്ഥിരം അക്ഷരം *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        തിരഞ്ഞെടുപ്പ് (context.src1->type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        തിരഞ്ഞെടുപ്പ് (context.src0->type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                {
                    // Quantized types: use helpers but accumulate in f16
                    സ്ഥിരം ഘടന ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    തിരഞ്ഞെടുപ്പ് (context.src0->type) {
                        അവസ്ഥ GGML_TYPE_IQ1_S:
                        അവസ്ഥ GGML_TYPE_IQ1_M:
                        അവസ്ഥ GGML_TYPE_IQ2_S:
                        അവസ്ഥ GGML_TYPE_IQ3_S:
                        അവസ്ഥ GGML_TYPE_IQ4_NL:
                        അവസ്ഥ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_IQ2_XXS:
                        അവസ്ഥ GGML_TYPE_IQ2_XS:
                        അവസ്ഥ GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            നിർത്തുക;
                        അവസ്ഥ GGML_TYPE_MXFP4:
                        അവസ്ഥ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            നിർത്തുക;
                        സ്ഥിരസ്ഥിതി:
                            നിർത്തുക;
                    }
                    നിർത്തുക;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(താക്കോൽ.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        എങ്കിൽ (താക്കോൽ.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        എങ്കിൽ (താക്കോൽ.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(താക്കോൽ.n_experts));

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[താക്കോൽ] = pipeline;
        തിരികെ mul_mat_id_vec_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_unary_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        സ്ഥിരം തർക്കം                     is_unary = context.dst->op == GGML_OP_UNARY;
        സ്ഥിരം പൂർണ്ണസംഖ്യ                      op       = is_unary ? (പൂർണ്ണസംഖ്യ) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_താക്കോൽ താക്കോൽ      = {};
        താക്കോൽ.type                                = context.dst->type;
        താക്കോൽ.op                                  = op;
        താക്കോൽ.is_unary                            = is_unary;
        താക്കോൽ.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        താക്കോൽ.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        auto it = unary_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != unary_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            താക്കോൽ.is_unary ? ggml_unary_op_name((ggml_unary_op) താക്കോൽ.op) : ggml_op_name((ggml_op) താക്കോൽ.op);
        defines.push_back(variant);

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for unary shader");
        }

        എങ്കിൽ (താക്കോൽ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        എങ്കിൽ (op == GGML_OP_TRI) {
            തിരഞ്ഞെടുപ്പ് (താക്കോൽ.ttype) {
                അവസ്ഥ GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    നിർത്തുക;
                അവസ്ഥ GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    നിർത്തുക;
                അവസ്ഥ GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    നിർത്തുക;
                അവസ്ഥ GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_unary, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = താക്കോൽ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[താക്കോൽ]     = pipeline;
        തിരികെ unary_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, തർക്കം src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        താക്കോൽ.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        താക്കോൽ.src_overlap                           = src_overlap;

        auto it = rms_norm_mul_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != rms_norm_mul_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        എങ്കിൽ (താക്കോൽ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = താക്കോൽ.inplace;
        pipeline_decisions->overlap     = താക്കോൽ.overlap;
        pipeline_decisions->src_overlap = താക്കോൽ.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[താക്കോൽ]     = pipeline;
        തിരികെ rms_norm_mul_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_ദ്വിമാന_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, തർക്കം src_overlap) {
        ggml_webgpu_ദ്വിമാന_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.type                            = context.dst->type;
        താക്കോൽ.op                              = context.dst->op;
        താക്കോൽ.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        താക്കോൽ.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        താക്കോൽ.src_overlap                     = src_overlap;

        auto it = ദ്വിമാന_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != ദ്വിമാന_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) താക്കോൽ.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for binary shader");
        }

        എങ്കിൽ (താക്കോൽ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_ദ്വിമാന, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_ദ്വിമാന_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = താക്കോൽ.inplace;
        pipeline_decisions->overlap     = താക്കോൽ.overlap;
        pipeline_decisions->src_overlap = താക്കോൽ.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ദ്വിമാന_pipelines[താക്കോൽ]    = pipeline;
        തിരികെ ദ്വിമാന_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_add_id_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = add_id_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != add_id_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        സ്ഥിരം അക്ഷരം *             shader_src = wgsl_add_id;

        എങ്കിൽ (താക്കോൽ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed              = preprocessor.preprocess(shader_src, defines);
        auto pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_size = context.max_wg_size;
        pipeline_decisions->inplace = താക്കോൽ.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[താക്കോൽ]    = pipeline;
        തിരികെ pipeline;
    }

    webgpu_pipeline get_concat_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, തർക്കം src_overlap) {
        ggml_webgpu_concat_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.type                            = context.dst->type;
        താക്കോൽ.src_overlap                     = src_overlap;

        auto it = concat_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != concat_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for concat shader");
        }

        എങ്കിൽ (താക്കോൽ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_concat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_ദ്വിമാന_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = താക്കോൽ.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[താക്കോൽ]    = pipeline;
        തിരികെ concat_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_repeat_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.type                            = context.dst->type;

        auto it = repeat_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != repeat_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_repeat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[താക്കോൽ]    = pipeline;
        തിരികെ repeat_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_flash_attn_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, തർക്കം kv_overlap) {
        സ്ഥിരം തർക്കം can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.common                              = ggml_webgpu_flash_attn_make_common_pipeline_താക്കോൽ(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        താക്കോൽ.common.k_direct &= decisions.use_sg_matrix && താക്കോൽ.common.k_type == GGML_TYPE_F16;
        താക്കോൽ.common.v_direct &= decisions.use_sg_matrix && താക്കോൽ.common.v_type == GGML_TYPE_F16;
        താക്കോൽ.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_limit_bytes, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            താക്കോൽ.common.head_dim_qk, താക്കോൽ.common.head_dim_v, താക്കോൽ.common.has_mask,
            താക്കോൽ.common.k_direct || താക്കോൽ.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        എങ്കിൽ (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = അസത്യം;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            താക്കോൽ.common.k_direct     = അസത്യം;
            താക്കോൽ.common.v_direct     = അസത്യം;
            താക്കോൽ.use_sg_matrix       = അസത്യം;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, താക്കോൽ.common.head_dim_qk, താക്കോൽ.common.head_dim_v,
                താക്കോൽ.common.has_mask, താക്കോൽ.common.k_direct || താക്കോൽ.common.v_direct);
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

        എങ്കിൽ (താക്കോൽ.common.k_direct || താക്കോൽ.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            ആകുമ്പോൾ (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_size;
            }
        }

        auto it = flash_attn_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != flash_attn_pipelines.end()) {
            തിരികെ it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(താക്കോൽ.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_size);
        സ്ഥിരം അക്ഷരം *             shader_src = nullptr;
        എങ്കിൽ (!താക്കോൽ.use_sg_matrix) {
            shader_src = wgsl_flash_attn_tile;
            defines.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_size) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_size) + "_" +
                       std::to_string(context.max_subgroup_size);
        } അല്ലെങ്കിൽ {
            shader_src = wgsl_flash_attn;
            defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[താക്കോൽ] = pipeline;
        തിരികെ flash_attn_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, തർക്കം kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.common = ggml_webgpu_flash_attn_make_common_pipeline_താക്കോൽ(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        auto it = flash_attn_vec_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != flash_attn_vec_pipelines.end()) {
            തിരികെ it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, താക്കോൽ.common.head_dim_qk,
                                                                   താക്കോൽ.common.head_dim_v, താക്കോൽ.common.has_mask,
                                                                   താക്കോൽ.common.k_direct || താക്കോൽ.common.v_direct);
        decisions.wg_size = context.max_subgroup_size;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(താക്കോൽ.common, variant, 1u, decisions.kv_tile, decisions.wg_size);
        എങ്കിൽ (താക്കോൽ.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (വലുപ്പം("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_size;
        സ്ഥിരം uint32_t D       = താക്കോൽ.common.head_dim_qk | താക്കോൽ.common.head_dim_v;
        സ്ഥിരം uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_size, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[താക്കോൽ] = pipeline;
        തിരികെ flash_attn_vec_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.kv_tile                                 = kv_tile;
        auto it                                     = flash_attn_blk_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != flash_attn_blk_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(താക്കോൽ.kv_tile));
        variant += std::string("_kvt") + std::to_string(താക്കോൽ.kv_tile);

        uint32_t wg_size = 1;
        ആകുമ്പോൾ ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[താക്കോൽ] = pipeline;
        തിരികെ flash_attn_blk_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        താക്കോൽ.dst_type                                       = context.dst->type;
        താക്കോൽ.wg_size                                        = context.max_wg_size;
        auto it                                            = flash_attn_vec_reduce_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != flash_attn_vec_reduce_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.dst_type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(താക്കോൽ.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(താക്കോൽ.head_dim_v));
        variant += std::string("_hsv") + std::to_string(താക്കോൽ.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[താക്കോൽ] = pipeline;
        തിരികെ flash_attn_vec_reduce_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_cpy_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.src_type                     = context.src0->type;
        താക്കോൽ.dst_type                     = context.dst->type;

        auto it = cpy_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != cpy_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.src_type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_I32:
                defines.push_back("SRC_I32");
                variant += "_i32";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.dst_type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_cpy, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[താക്കോൽ]       = pipeline;
        തിരികെ cpy_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_glu_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, തർക്കം src_overlap) {
        ggml_webgpu_glu_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.glu_op                       = ggml_get_glu_op(context.dst);
        താക്കോൽ.type                         = context.dst->type;
        താക്കോൽ.split                        = (context.src1 != nullptr);
        താക്കോൽ.src_overlap                  = src_overlap;

        auto it = glu_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != glu_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.glu_op) {
            അവസ്ഥ GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                നിർത്തുക;
            അവസ്ഥ GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                നിർത്തുക;
            അവസ്ഥ GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                നിർത്തുക;
            അവസ്ഥ GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                നിർത്തുക;
            അവസ്ഥ GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                നിർത്തുക;
            അവസ്ഥ GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                നിർത്തുക;
            അവസ്ഥ GGML_GLU_OP_SWIGLU_CLAMP:
                defines.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported GLU op");
        }
        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        എങ്കിൽ (താക്കോൽ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } അല്ലെങ്കിൽ എങ്കിൽ (താക്കോൽ.split) {
            variant += "_split";
        } അല്ലെങ്കിൽ {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_glu, defines);
        auto decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = താക്കോൽ.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[താക്കോൽ]       = pipeline;
        തിരികെ glu_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_rope_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.type                          = context.dst->type;
        താക്കോൽ.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        താക്കോൽ.has_ff                        = (context.src2 != nullptr);

        auto it = rope_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != rope_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        തിരഞ്ഞെടുപ്പ് (താക്കോൽ.type) {
            അവസ്ഥ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        എങ്കിൽ (താക്കോൽ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        എങ്കിൽ (താക്കോൽ.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_rope, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = താക്കോൽ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[താക്കോൽ]      = pipeline;
        തിരികെ rope_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_soft_max_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        താക്കോൽ.has_mask                          = (context.src1 != nullptr);
        താക്കോൽ.has_sink                          = (context.src2 != nullptr);
        താക്കോൽ.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = soft_max_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != soft_max_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        എങ്കിൽ (താക്കോൽ.has_mask) {
            defines.push_back("HAS_MASK");
            തിരഞ്ഞെടുപ്പ് (താക്കോൽ.mask_type) {
                അവസ്ഥ GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    നിർത്തുക;
                അവസ്ഥ GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        എങ്കിൽ (താക്കോൽ.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        എങ്കിൽ (താക്കോൽ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = താക്കോൽ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[താക്കോൽ]  = pipeline;
        തിരികെ soft_max_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_conv2d_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.weight_type                     = context.src0->type;
        താക്കോൽ.നിവേശം_type                      = context.src1->type;
        താക്കോൽ.output_type                     = context.dst->type;

        auto it = conv2d_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != conv2d_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        auto തള്ളുക_type_defines = [&](സ്ഥിരം അക്ഷരം * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            എങ്കിൽ (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } അല്ലെങ്കിൽ എങ്കിൽ (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } അല്ലെങ്കിൽ {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        തള്ളുക_type_defines("WEIGHT_TYPE", താക്കോൽ.weight_type);
        തള്ളുക_type_defines("INPUT_TYPE", താക്കോൽ.നിവേശം_type);
        തള്ളുക_type_defines("OUTPUT_TYPE", താക്കോൽ.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[താക്കോൽ]    = pipeline;
        തിരികെ conv2d_pipelines[താക്കോൽ];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context, തർക്കം whcn) {
        ggml_webgpu_conv2d_dw_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.weight_type                        = context.src0->type;
        താക്കോൽ.നിവേശം_type                         = context.src1->type;
        താക്കോൽ.output_type                        = context.dst->type;
        താക്കോൽ.whcn                               = whcn;

        auto it = conv2d_dw_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != conv2d_dw_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        auto തള്ളുക_type_defines = [&](സ്ഥിരം അക്ഷരം * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            എങ്കിൽ (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } അല്ലെങ്കിൽ എങ്കിൽ (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } അല്ലെങ്കിൽ {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        തള്ളുക_type_defines("WEIGHT_TYPE", താക്കോൽ.weight_type);
        തള്ളുക_type_defines("INPUT_TYPE", താക്കോൽ.നിവേശം_type);
        തള്ളുക_type_defines("OUTPUT_TYPE", താക്കോൽ.output_type);

        എങ്കിൽ (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[താക്കോൽ] = pipeline;
        തിരികെ conv2d_dw_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_im2col_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.നിവേശം_type                      = context.src1->type;
        താക്കോൽ.output_type                     = context.dst->type;

        auto it = im2col_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != im2col_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        auto തള്ളുക_type_defines = [&](സ്ഥിരം അക്ഷരം * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            എങ്കിൽ (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } അല്ലെങ്കിൽ എങ്കിൽ (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } അല്ലെങ്കിൽ {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        തള്ളുക_type_defines("INPUT_TYPE", താക്കോൽ.നിവേശം_type);
        തള്ളുക_type_defines("OUTPUT_TYPE", താക്കോൽ.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[താക്കോൽ]    = pipeline;
        തിരികെ im2col_pipelines[താക്കോൽ];
    }

    webgpu_pipeline get_upscale_pipeline(സ്ഥിരം ggml_webgpu_shader_lib_context & context) {
        സ്ഥിരം uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        സ്ഥിരം uint32_t base_mode  = mode_flags & 0xFFu;
        സ്ഥിരം തർക്കം     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.നിവേശം_type                       = context.src0->type;
        താക്കോൽ.output_type                      = context.dst->type;
        താക്കോൽ.base_mode                        = base_mode;
        താക്കോൽ.antialias                        = antialias;

        auto it = upscale_pipelines.find(താക്കോൽ);
        എങ്കിൽ (it != upscale_pipelines.end()) {
            തിരികെ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        എങ്കിൽ (താക്കോൽ.നിവേശം_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } അല്ലെങ്കിൽ {
            variant += "_src_f32";
        }

        എങ്കിൽ (താക്കോൽ.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } അല്ലെങ്കിൽ {
            variant += "_dst_f32";
        }

        തിരഞ്ഞെടുപ്പ് (base_mode) {
            അവസ്ഥ GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                നിർത്തുക;
            അവസ്ഥ GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                നിർത്തുക;
            അവസ്ഥ GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("Unsupported upscale mode");
        }

        എങ്കിൽ (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_upscale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[താക്കോൽ]   = pipeline;
        തിരികെ upscale_pipelines[താക്കോൽ];
    }

  സ്വകാര്യ:
    സ്ഥിതം webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
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
        തിരികെ { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
