#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-wgsl-shaders.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "pre_wgsl.hpp"

#ಅನ್ತರ್ಭಾವ <webgpu/webgpu_cpp.h>

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <vector>

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
template <typename T> ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_webgpu_hash_combine(size_t & seed, ಧ್ರುವ T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
ಅನ್ತಃಸ್ಥ uintptr_t ggml_webgpu_tensor_addr(ಧ್ರುವ ggml_tensor * tensor) {
    ಧ್ರುವ ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    ಪ್ರತ್ಯಾಗಮ (uintptr_t) base_tensor->data + tensor->view_offs;
}

ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_webgpu_tensor_equal(ಧ್ರುವ ggml_tensor * a, ಧ್ರುವ ggml_tensor * b) {
    ಪ್ರತ್ಯಾಗಮ a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

ಸಂರಚನಾ ggml_webgpu_shader_lib_context {
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
    ತಾರ್ಕಿಕ        supports_subgroups       = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ        supports_subgroup_matrix = ಅಸತ್ಯ;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_size        = 0;
    uint32_t    max_subgroup_size        = 0;
    ತಾರ್ಕಿಕ        supports_dot_product     = ಅಸತ್ಯ;
    std::string vendor;
};

ಸಂರಚನಾ webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<ಶೂನ್ಯ> context = nullptr;
};

ಸಂರಚನಾ ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 0;
    ತಾರ್ಕಿಕ     inplace = ಅಸತ್ಯ;
};

ಸಂರಚನಾ ggml_webgpu_get_rows_shader_decisions {
    uint32_t wg_size    = 0;
    ತಾರ್ಕಿಕ     vectorized = ಅಸತ್ಯ;
};

ಸಂರಚನಾ ggml_webgpu_ದ್ವಿಮಾನ_shader_decisions {
    uint32_t wg_size     = 0;
    ತಾರ್ಕಿಕ     inplace     = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ     overlap     = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ     src_overlap = ಅಸತ್ಯ;
};

ಸಂರಚನಾ ggml_webgpu_glu_shader_decisions {
    uint32_t wg_size     = 0;
    ತಾರ್ಕಿಕ     src_overlap = ಅಸತ್ಯ;
};

ಸಂರಚನಾ ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<ಶೂನ್ಯ> decisions;
};

ಸಂರಚನಾ ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_size;
    uint32_t tokens_per_wg;
};

ಸಂರಚನಾ ggml_webgpu_ssm_scan_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಙ್ಕ     type;
    ಪೂರ್ಣಾಙ್ಕ     d_state;
    uint8_t xbc_overlap;
    ತಾರ್ಕಿಕ    a_overlap;
    ತಾರ್ಕಿಕ    ids_overlap;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_ssm_scan_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

ಸಂರಚನಾ ggml_webgpu_ssm_scan_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_ssm_scan_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.d_state);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.xbc_overlap);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.a_overlap);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.ids_overlap);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಸಂರಚನಾ ggml_webgpu_ssm_scan_shader_decisions {
    uint32_t wg_size;
    uint32_t tokens_per_tile;
    uint8_t  xbc_overlap = 0;
    ತಾರ್ಕಿಕ     a_overlap   = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ     ids_overlap = ಅಸತ್ಯ;
};

/** Argsort **/

ಸಂರಚನಾ ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes;
    int32_t  order;
};

/** Set Rows **/

ಸಂರಚನಾ ggml_webgpu_set_rows_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಙ್ಕ src0_type;
    ಪೂರ್ಣಾಙ್ಕ dst_type;
    ಪೂರ್ಣಾಙ್ಕ vec4;
    ಪೂರ್ಣಾಙ್ಕ i64_idx;
    ಪೂರ್ಣಾಙ್ಕ pair_blocks;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_set_rows_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

ಸಂರಚನಾ ggml_webgpu_set_rows_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_set_rows_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src0_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.dst_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.vec4);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.i64_idx);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.pair_blocks);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಸಂರಚನಾ ggml_webgpu_set_rows_shader_decisions {
    ತಾರ್ಕಿಕ     vec4;
    ತಾರ್ಕಿಕ     i64_idx;
    ತಾರ್ಕಿಕ     pair_blocks;
    uint32_t wg_size;
};

/** Set **/

ಸಂರಚನಾ ggml_webgpu_set_pipeline_ಕೀಲಿ {
    ggml_type type;
    ತಾರ್ಕಿಕ      inplace;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_set_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ type == other.type && inplace == other.inplace;
    }
};

ಸಂರಚನಾ ggml_webgpu_set_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_set_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.inplace);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Get Rows **/

ಸಂರಚನಾ ggml_webgpu_get_rows_pipeline_ಕೀಲಿ {
    ggml_type src_type;
    ಪೂರ್ಣಾಙ್ಕ       vectorized;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_get_rows_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ src_type == other.src_type && vectorized == other.vectorized;
    }
};

ಸಂರಚನಾ ggml_webgpu_get_rows_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_get_rows_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.vectorized);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Row Norm **/

ಸಂರಚನಾ ggml_webgpu_row_norm_pipeline_ಕೀಲಿ {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    ತಾರ್ಕಿಕ      inplace;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_row_norm_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

ಸಂರಚನಾ ggml_webgpu_row_norm_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_row_norm_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.op);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.dst_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.inplace);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** RMS_NORM + MUL **/

ಸಂರಚನಾ ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ {
    ತಾರ್ಕಿಕ inplace;      // rn_src == dst
    ತಾರ್ಕಿಕ overlap;      // mul_src == dst
    ತಾರ್ಕಿಕ src_overlap;  // rn_src binding overlaps mul_src binding

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

ಸಂರಚನಾ ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.inplace);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.overlap);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src_overlap);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಸಂರಚನಾ ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    ತಾರ್ಕಿಕ     inplace     = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ     overlap     = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ     src_overlap = ಅಸತ್ಯ;
};

/** Pad **/
ಸಂರಚನಾ ggml_webgpu_pad_pipeline_ಕೀಲಿ {
    ತಾರ್ಕಿಕ circular;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_pad_pipeline_ಕೀಲಿ & other) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ circular == other.circular; }
};

ಸಂರಚನಾ ggml_webgpu_pad_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_pad_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.circular);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Solve Tri **/
ಸಂರಚನಾ ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಙ್ಕ type;
    ಪೂರ್ಣಾಙ್ಕ n;
    ಪೂರ್ಣಾಙ್ಕ k;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ type == other.type && n == other.n && k == other.k;
    }
};

ಸಂರಚನಾ ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.n);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.k);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** SSM Conv **/
ಸಂರಚನಾ ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಙ್ಕ type;
    ಪೂರ್ಣಾಙ್ಕ vectorized;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
ಸಂರಚನಾ ggml_webgpu_conv2d_pipeline_ಕೀಲಿ {
    ggml_type weight_type;
    ggml_type ಒಳಹರಿವು_type;
    ggml_type output_type;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_conv2d_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ weight_type == other.weight_type && ಒಳಹರಿವು_type == other.ಒಳಹರಿವು_type && output_type == other.output_type;
    }
};

ಸಂರಚನಾ ggml_webgpu_conv2d_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_conv2d_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.weight_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.ಒಳಹರಿವು_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.output_type);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
ಸಂರಚನಾ ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ {
    ggml_type weight_type;
    ggml_type ಒಳಹರಿವು_type;
    ggml_type output_type;
    ತಾರ್ಕಿಕ      whcn;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ weight_type == other.weight_type && ಒಳಹರಿವು_type == other.ಒಳಹರಿವು_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

ಸಂರಚನಾ ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.weight_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.ಒಳಹರಿವು_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.output_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.whcn);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Im2Col **/
ಸಂರಚನಾ ggml_webgpu_im2col_pipeline_ಕೀಲಿ {
    ggml_type ಒಳಹರಿವು_type;
    ggml_type output_type;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_im2col_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ ಒಳಹರಿವು_type == other.ಒಳಹರಿವು_type && output_type == other.output_type;
    }
};

ಸಂರಚನಾ ggml_webgpu_im2col_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_im2col_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.ಒಳಹರಿವು_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.output_type);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Gated Delta Net **/
ಸಂರಚನಾ ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಙ್ಕ  type;
    ಪೂರ್ಣಾಙ್ಕ  s_v;
    ಪೂರ್ಣಾಙ್ಕ  kda;
    ತಾರ್ಕಿಕ fused_cache;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

ಸಂರಚನಾ ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.s_v);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.kda);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಸಂರಚನಾ ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.vectorized);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Scale **/

ಸಂರಚನಾ ggml_webgpu_scale_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಙ್ಕ inplace;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_scale_pipeline_ಕೀಲಿ & other) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ inplace == other.inplace; }
};

ಸಂರಚನಾ ggml_webgpu_scale_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_scale_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.inplace);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Upscale **/

ಸಂರಚನಾ ggml_webgpu_upscale_pipeline_ಕೀಲಿ {
    ggml_type ಒಳಹರಿವು_type;
    ggml_type output_type;
    uint32_t  base_mode;
    ತಾರ್ಕಿಕ      antialias;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_upscale_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ ಒಳಹರಿವು_type == other.ಒಳಹರಿವು_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

ಸಂರಚನಾ ggml_webgpu_upscale_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_upscale_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.ಒಳಹರಿವು_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.output_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.base_mode);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.antialias);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Concat **/

ಸಂರಚನಾ ggml_webgpu_concat_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಙ್ಕ  type;
    ತಾರ್ಕಿಕ src_overlap;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_concat_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ type == other.type && src_overlap == other.src_overlap;
    }
};

ಸಂರಚನಾ ggml_webgpu_concat_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_concat_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src_overlap);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Repeat **/

ಸಂರಚನಾ ggml_webgpu_repeat_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಙ್ಕ type;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_repeat_pipeline_ಕೀಲಿ & other) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ type == other.type; }
};

ಸಂರಚನಾ ggml_webgpu_repeat_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_repeat_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Binary **/

ಸಂರಚನಾ ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಙ್ಕ  type;
    ಪೂರ್ಣಾಙ್ಕ  op;
    ತಾರ್ಕಿಕ inplace;
    ತಾರ್ಕಿಕ overlap;
    ತಾರ್ಕಿಕ src_overlap;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

ಸಂರಚನಾ ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.op);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.inplace);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.overlap);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src_overlap);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/* Add_Id */

ಸಂರಚನಾ ggml_webgpu_add_id_pipeline_ಕೀಲಿ {
    ತಾರ್ಕಿಕ inplace;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_add_id_pipeline_ಕೀಲಿ & other) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ inplace == other.inplace; }
};

ಸಂರಚನಾ ggml_webgpu_add_id_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_add_id_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.inplace);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Unary **/

ಸಂರಚನಾ ggml_webgpu_unary_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಙ್ಕ           type;
    ಪೂರ್ಣಾಙ್ಕ           op;
    ತಾರ್ಕಿಕ          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    ತಾರ್ಕಿಕ          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_unary_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

ಸಂರಚನಾ ggml_webgpu_unary_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_unary_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.op);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.is_unary);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.inplace);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.ttype);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** FlashAttention */

ಸಂರಚನಾ ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    ತಾರ್ಕಿಕ      k_direct;
    ತಾರ್ಕಿಕ      v_direct;
    ತಾರ್ಕಿಕ      kv_overlap;
    ತಾರ್ಕಿಕ      has_mask;
    ತಾರ್ಕಿಕ      has_sinks;
    ತಾರ್ಕಿಕ      uses_logit_softcap;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_webgpu_flash_attn_hash_common_pipeline_ಕೀಲಿ(size_t &                                           seed,
                                                            ಧ್ರುವ ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ & ಕೀಲಿ) {
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.q_type);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.k_type);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.v_type);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.dst_type);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.head_dim_qk);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.head_dim_v);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.k_direct);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.v_direct);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.kv_overlap);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.has_mask);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.has_sinks);
    ggml_webgpu_hash_combine(seed, ಕೀಲಿ.uses_logit_softcap);
}

ಸಂರಚನಾ ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ {
    ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ common;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ & other) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ common == other.common; }
};

ಸಂರಚನಾ ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ಕೀಲಿ(seed, ಕೀಲಿ.common);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಸಂರಚನಾ ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ {
    ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ common;
    ತಾರ್ಕಿಕ                                       use_sg_matrix;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

ಸಂರಚನಾ ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ಕೀಲಿ(seed, ಕೀಲಿ.common);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.use_sg_matrix);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಸಂರಚನಾ ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

ಸಂರಚನಾ ggml_webgpu_flash_attn_decisions {
    ತಾರ್ಕಿಕ     use_sg_matrix = ಅಸತ್ಯ;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_size       = 0;
};

ಅನ್ತಃಸ್ಥ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
ಅನ್ತಃಸ್ಥ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

ಅನ್ತಃಸ್ಥ size_t ggml_webgpu_flash_attn_tensor_offset(ಧ್ರುವ ggml_tensor * tensor) {
    constexpr uintptr_t ptr_base_addr = 0x1000u;
    ಧ್ರುವ ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
    ಪ್ರತ್ಯಾಗಮ reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
}

ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_float_vec4_aligned(ಧ್ರುವ ggml_tensor * K, size_t storage_offset_alignment) {
    ಧ್ರುವ uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_size(K->type));
    ಪ್ರತ್ಯಾಗಮ offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_float_vec4_aligned(ಧ್ರುವ ggml_tensor * K,
                                                      ಧ್ರುವ ggml_tensor * V,
                                                      size_t              storage_offset_alignment) {
    ಪ್ರತ್ಯಾಗಮ ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_k_direct(ಧ್ರುವ ggml_tensor * Q, ಧ್ರುವ ggml_tensor * K, uint32_t kv_direct_align) {
    ಪ್ರತ್ಯಾಗಮ (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_v_direct(ಧ್ರುವ ggml_tensor * Q, ಧ್ರುವ ggml_tensor * V, uint32_t kv_direct_align) {
    ಪ್ರತ್ಯಾಗಮ ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

ಅನ್ತಃಸ್ಥ ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ ggml_webgpu_flash_attn_make_common_pipeline_ಕೀಲಿ(
    ಧ್ರುವ ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    ತಾರ್ಕಿಕ                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
    ಕೀಲಿ.q_type                                     = context.src0->type;
    ಕೀಲಿ.k_type                                     = context.src1->type;
    ಕೀಲಿ.v_type                                     = context.src2->type;
    ಕೀಲಿ.dst_type                                   = context.dst->type;
    ಕೀಲಿ.head_dim_qk                                = (uint32_t) context.src0->ne[0];
    ಕೀಲಿ.head_dim_v                                 = (uint32_t) context.src2->ne[0];
    ಕೀಲಿ.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    ಕೀಲಿ.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    ಕೀಲಿ.kv_overlap         = kv_overlap;
    ಕೀಲಿ.has_mask           = context.src3 != nullptr;
    ಕೀಲಿ.has_sinks          = context.src4 != nullptr;
    ಕೀಲಿ.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    ಪ್ರತ್ಯಾಗಮ ಕೀಲಿ;
}

ಅನ್ತಃಸ್ಥ std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    ಧ್ರುವ ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ & ಕೀಲಿ,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_size) {
    std::vector<std::string> defines;

    ವಿಕಲ್ಪ (ಕೀಲಿ.k_type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            defines.push_back("K_F32");
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_F16:
            defines.push_back("K_F16");
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_0:
            defines.push_back("K_Q4_0");
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            defines.push_back("K_Q8_0");
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(ಕೀಲಿ.k_type);

    ವಿಕಲ್ಪ (ಕೀಲಿ.v_type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            defines.push_back("V_F32");
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_F16:
            defines.push_back("V_F16");
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_0:
            defines.push_back("V_Q4_0");
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            defines.push_back("V_Q8_0");
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(ಕೀಲಿ.v_type);

    ವಿಕಲ್ಪ (ಕೀಲಿ.q_type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            defines.push_back("Q_F32");
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_F16:
            defines.push_back("Q_F16");
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(ಕೀಲಿ.q_type);

    ವಿಕಲ್ಪ (ಕೀಲಿ.dst_type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            defines.push_back("DST_F32");
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_F16:
            defines.push_back("DST_F16");
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(ಕೀಲಿ.dst_type);

    ಯದಿ (ಕೀಲಿ.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    ಯದಿ (ಕೀಲಿ.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    ಯದಿ (ಕೀಲಿ.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    ಯದಿ (ಕೀಲಿ.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    ಯದಿ (ಕೀಲಿ.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    ಯದಿ (ಕೀಲಿ.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(ಕೀಲಿ.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(ಕೀಲಿ.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(ಕೀಲಿ.head_dim_v));
    variant += std::string("_hsv") + std::to_string(ಕೀಲಿ.head_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    ಯದಿ (ggml_is_quantized(ಕೀಲಿ.k_type) || ggml_is_quantized(ಕೀಲಿ.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        ಯದಿ (ggml_is_quantized(ಕೀಲಿ.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        ಯದಿ (ggml_is_quantized(ಕೀಲಿ.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    ಪ್ರತ್ಯಾಗಮ defines;
}

ಸಂರಚನಾ ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ {
    uint32_t  head_dim_v;
    uint32_t  wg_size;
    ggml_type dst_type;
};

ಸಂರಚನಾ ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.head_dim_v);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.wg_size);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.dst_type);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ & lhs,
                       ಧ್ರುವ ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ & rhs) {
    ಪ್ರತ್ಯಾಗಮ lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size && lhs.dst_type == rhs.dst_type;
}

ಸಂರಚನಾ ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ {
    uint32_t kv_tile;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ & other) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ kv_tile == other.kv_tile; }
};

ಸಂರಚನಾ ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.kv_tile);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
ಅನ್ತಃಸ್ಥ size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  ತಾರ್ಕಿಕ     has_mask,
                                                  ತಾರ್ಕಿಕ     kv_direct) {
    ಧ್ರುವ uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * head_dim_qk;        // q_shmem
    ಯದಿ (!kv_direct) {
        f32_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f32_elems += q_tile * head_dim_v;         // o_shmem
    ಯದಿ (has_mask) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    ಪ್ರತ್ಯಾಗಮ f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

ಅನ್ತಃಸ್ಥ uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   limit_bytes,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t head_dim_qk,
                                                   uint32_t head_dim_v,
                                                   ತಾರ್ಕಿಕ     has_mask,
                                                   ತಾರ್ಕಿಕ     kv_direct) {
    ಧ್ರುವ size_t base_q_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 0, head_dim_qk, head_dim_v, has_mask, kv_direct);
    ಯದಿ (limit_bytes <= base_q_bytes) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ಧ್ರುವ size_t one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, head_dim_qk, head_dim_v, has_mask, kv_direct);
    ಧ್ರುವ size_t bytes_per_kv = one_kv_bytes - base_q_bytes;
    ಯದಿ (bytes_per_kv == 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ಧ್ರುವ size_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    ಪ್ರತ್ಯಾಗಮ (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

ಅನ್ತಃಸ್ಥ uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_limit_bytes,
                                                       uint32_t head_dim_qk,
                                                       uint32_t head_dim_v,
                                                       ತಾರ್ಕಿಕ     has_mask,
                                                       ತಾರ್ಕಿಕ     kv_direct) {
    ಧ್ರುವ uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_limit_bytes, 1u, 1u, head_dim_qk, head_dim_v, has_mask, kv_direct);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    ಯದಿ (kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        ಯಾವತ್ (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    ಪ್ರತ್ಯಾಗಮ kv_tile;
}

ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(ತಾರ್ಕಿಕ                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ಧ್ರುವ ggml_tensor * Q,
                                                                ಧ್ರುವ ggml_tensor * V) {
    ಪ್ರತ್ಯಾಗಮ supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

ಸಂರಚನಾ ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ {
    ggml_type src0_type;
    ggml_type src1_type;
    ಪೂರ್ಣಾಙ್ಕ       vectorized;
    uint32_t  ಅಂಕೆ_cols;
    ತಾರ್ಕಿಕ      use_mmvq;
    ತಾರ್ಕಿಕ      src_overlap;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               ಅಂಕೆ_cols == other.ಅಂಕೆ_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

ಸಂರಚನಾ ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src0_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src1_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.vectorized);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.ಅಂಕೆ_cols);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.use_mmvq);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src_overlap);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಸಂರಚನಾ ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
    ತಾರ್ಕಿಕ     src_overlap = ಅಸತ್ಯ;
};

ಸಂರಚನಾ ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ {
    ggml_type src0_type;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ & other) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ src0_type == other.src0_type; }
};

ಸಂರಚನಾ ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src0_type);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಸಂರಚನಾ ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ {
    ggml_type src0_type;
    ggml_type src1_type;
    ಪೂರ್ಣಾಙ್ಕ       vectorized;
    ಪೂರ್ಣಾಙ್ಕ       use_subgroup_matrix;
    ತಾರ್ಕಿಕ      src_overlap;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

ಸಂರಚನಾ ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src0_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src1_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.vectorized);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src_overlap);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಸಂರಚನಾ ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    ಪೂರ್ಣಾಙ್ಕ      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_size;
    ತಾರ್ಕಿಕ     src_overlap = ಅಸತ್ಯ;
};

/** MUL_MAT_ID **/

ಸಂರಚನಾ ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ {
    uint32_t n_experts;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ n_experts == other.n_experts;
    }
};

ಸಂರಚನಾ ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.n_experts);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

ಸಂರಚನಾ ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  ಅಂಕೆ_cols;
    ಪೂರ್ಣಾಙ್ಕ       vectorized;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               ಅಂಕೆ_cols == other.ಅಂಕೆ_cols && vectorized == other.vectorized;
    }
};

ಸಂರಚನಾ ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src0_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src1_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.n_experts);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.ಅಂಕೆ_cols);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.vectorized);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Cpy **/

ಸಂರಚನಾ ggml_webgpu_cpy_pipeline_ಕೀಲಿ {
    ggml_type src_type;
    ggml_type dst_type;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_cpy_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ src_type == other.src_type && dst_type == other.dst_type;
    }
};

ಸಂರಚನಾ ggml_webgpu_cpy_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_cpy_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.dst_type);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Glu **/

ಸಂರಚನಾ ggml_webgpu_glu_pipeline_ಕೀಲಿ {
    ggml_glu_op glu_op;
    ggml_type   type;
    ತಾರ್ಕಿಕ        split;
    ತಾರ್ಕಿಕ        src_overlap;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_glu_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

ಸಂರಚನಾ ggml_webgpu_glu_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_glu_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.glu_op);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.split);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.src_overlap);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** Rope **/

ಸಂರಚನಾ ggml_webgpu_rope_pipeline_ಕೀಲಿ {
    ggml_type type;
    ತಾರ್ಕಿಕ      inplace;
    ತಾರ್ಕಿಕ      has_ff;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_rope_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

ಸಂರಚನಾ ggml_webgpu_rope_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_rope_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.inplace);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.has_ff);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** SoftMax **/

ಸಂರಚನಾ ggml_webgpu_soft_max_pipeline_ಕೀಲಿ {
    ggml_type mask_type;
    ತಾರ್ಕಿಕ      has_mask;
    ತಾರ್ಕಿಕ      has_sink;
    ತಾರ್ಕಿಕ      inplace;

    ತಾರ್ಕಿಕ operator==(ಧ್ರುವ ggml_webgpu_soft_max_pipeline_ಕೀಲಿ & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

ಸಂರಚನಾ ggml_webgpu_soft_max_pipeline_ಕೀಲಿ_hash {
    size_t operator()(ಧ್ರುವ ggml_webgpu_soft_max_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಧ್ರುವ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.mask_type);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.has_mask);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.has_sink);
        ggml_webgpu_hash_combine(seed, ಕೀಲಿ.inplace);
        ಪ್ರತ್ಯಾಗಮ seed;
    }
};

/** MMVQ **/

ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_webgpu_can_use_mmvq(ಧ್ರುವ ggml_tensor * src0,
                                     ಧ್ರುವ ggml_tensor * src1,
                                     ತಾರ್ಕಿಕ                supports_dot_product,
                                     ಧ್ರುವ std::string & vendor) {
    ಯದಿ (src1->ne[1] <= 4) {
        ತಾರ್ಕಿಕ supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        ಯದಿ (supports_dp4a && supports_dot_product) {
            ವಿಕಲ್ಪ (src1->type) {
                ಪಕ್ಷ GGML_TYPE_F32:
                    ವಿಕಲ್ಪ (src0->type) {
                        ಪಕ್ಷ GGML_TYPE_Q1_0:
                        ಪಕ್ಷ GGML_TYPE_Q4_0:
                        ಪಕ್ಷ GGML_TYPE_Q4_1:
                        ಪಕ್ಷ GGML_TYPE_Q5_0:
                        ಪಕ್ಷ GGML_TYPE_Q5_1:
                        ಪಕ್ಷ GGML_TYPE_Q8_0:
                        ಪಕ್ಷ GGML_TYPE_MXFP4:
                        ಪಕ್ಷ GGML_TYPE_Q2_K:
                        ಪಕ್ಷ GGML_TYPE_Q3_K:
                        ಪಕ್ಷ GGML_TYPE_Q4_K:
                        ಪಕ್ಷ GGML_TYPE_Q5_K:
                        ಪಕ್ಷ GGML_TYPE_Q6_K:
                            ಪ್ರತ್ಯಾಗಮ src0->ne[0] % 4 == 0;
                        ಸ್ವಭಾವ:
                            ಭಙ್ಗ;
                    }
                    ಭಙ್ಗ;
                ಸ್ವಭಾವ:
                    ಭಙ್ಗ;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ವರ್ಗ ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<ಪೂರ್ಣಾಙ್ಕ, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<ಪೂರ್ಣಾಙ್ಕ, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<ಪೂರ್ಣಾಙ್ಕ, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<ಪೂರ್ಣಾಙ್ಕ, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<ಪೂರ್ಣಾಙ್ಕ, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_ಕೀಲಿ_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_ಕೀಲಿ_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_unary_pipeline_ಕೀಲಿ_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_scale_pipeline_ಕೀಲಿ_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_ಕೀಲಿ_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_pad_pipeline_ಕೀಲಿ_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ_hash>
        ದ್ವಿಮಾನ_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_add_id_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_add_id_pipeline_ಕೀಲಿ_hash>
        add_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_concat_pipeline_ಕೀಲಿ_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_repeat_pipeline_ಕೀಲಿ_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_ಕೀಲಿ_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_set_pipeline_ಕೀಲಿ_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_cpy_pipeline_ಕೀಲಿ_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_glu_pipeline_ಕೀಲಿ_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_rope_pipeline_ಕೀಲಿ_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_ಕೀಲಿ_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_ಕೀಲಿ_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_im2col_pipeline_ಕೀಲಿ_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_upscale_pipeline_ಕೀಲಿ_hash>
        upscale_pipelines;

  ಸಾರ್ವಜನಿಕ:
    ggml_webgpu_shader_lib(wgpu::Device device) { ಏತತ್->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        auto it = sum_rows_pipelines.find(1);
        ಯದಿ (it != sum_rows_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        ಪ್ರತ್ಯಾಗಮ sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.op                                = context.dst->op;
        ಕೀಲಿ.src_type                          = context.src0->type;
        ಕೀಲಿ.dst_type                          = context.dst->type;
        ಕೀಲಿ.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = row_norm_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != row_norm_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        ವಿಕಲ್ಪ (ಕೀಲಿ.op) {
            ಪಕ್ಷ GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_NORM:
                defines.push_back("NORM");
                variant = "norm";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        ಯದಿ (ಕೀಲಿ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        ಯದಿ (ಕೀಲಿ.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        ಯದಿ (ಕೀಲಿ.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ಧ್ರುವ uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = ಕೀಲಿ.inplace;
        row_norm_pipelines[ಕೀಲಿ]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[ಕೀಲಿ].context = decisions;
        ಪ್ರತ್ಯಾಗಮ row_norm_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_argmax_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ತಾರ್ಕಿಕ vec4 = context.src0->ne[0] % 4 == 0;

        auto it = argmax_pipelines.find(vec4);
        ಯದಿ (it != argmax_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        ಯದಿ (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        auto processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        ಪ್ರತ್ಯಾಗಮ argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ಧ್ರುವ ತಾರ್ಕಿಕ                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_ಕೀಲಿ ಕೀಲಿ       = {};
        ಕೀಲಿ.src0_type                               = context.src0->type;
        ಕೀಲಿ.dst_type                                = context.dst->type;
        ಕೀಲಿ.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        ಕೀಲಿ.i64_idx     = context.src1->type == GGML_TYPE_I64;
        ಕೀಲಿ.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_size(context.dst->type)) % 2 == 0);

        auto it = set_rows_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != set_rows_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        ವಿಕಲ್ಪ (context.dst->type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_Q8_0:
                defines.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_Q4_0:
                defines.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        ಯದಿ (context.src0->type == GGML_TYPE_F16) {
            defines.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        ಯದಿ (ಕೀಲಿ.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        ಯದಿ (ಕೀಲಿ.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        ಯದಿ (ಕೀಲಿ.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        ಧ್ರುವ auto & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        auto         processed          = preprocessor.preprocess(shader_source, defines);
        auto         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = ಕೀಲಿ.vec4;
        decisions->i64_idx              = ಕೀಲಿ.i64_idx;
        decisions->pair_blocks          = ಕೀಲಿ.pair_blocks;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[ಕೀಲಿ]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[ಕೀಲಿ].context = decisions;
        ಪ್ರತ್ಯಾಗಮ set_rows_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_set_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                         = context.dst->type;
        ಕೀಲಿ.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = set_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != set_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for set shader");
        }

        ಯದಿ (ಕೀಲಿ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_set, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ಕೀಲಿ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[ಕೀಲಿ]       = pipeline;
        ಪ್ರತ್ಯಾಗಮ set_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_cumsum_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        auto it = cumsum_pipelines.find(1);
        ಯದಿ (it != cumsum_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        ಪ್ರತ್ಯಾಗಮ cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ತಾರ್ಕಿಕ          is_ತುದಿ_k = context.dst->op == GGML_OP_ತುದಿ_K;
        // ascending order is 0, descending order is 1
        ಧ್ರುವ int32_t order =
            is_ತುದಿ_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_pipelines.find(order);
        ಯದಿ (it != argsort_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        ಯಾವತ್ (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        auto decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        ಪ್ರತ್ಯಾಗಮ argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ತಾರ್ಕಿಕ          is_ತುದಿ_k = context.dst->op == GGML_OP_ತುದಿ_K;
        // ascending order is 0, descending order is 1
        ಧ್ರುವ int32_t order =
            is_ತುದಿ_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_merge_pipelines.find(order);
        ಯದಿ (it != argsort_merge_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        ಪ್ರತ್ಯಾಗಮ argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ vec4_aligned) {
        ಧ್ರುವ ತಾರ್ಕಿಕ vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src_type                          = context.src0->type;
        ಕೀಲಿ.vectorized                        = (ಪೂರ್ಣಾಙ್ಕ) vectorized;

        auto it = get_rows_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != get_rows_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        ಧ್ರುವ ಸಂರಚನಾ ggml_type_traits * type_traits = ggml_get_type_traits(ಕೀಲಿ.src_type);
        ಧ್ರುವ ವರ್ಣ *                    type_str    = type_traits->type_name;

        ವಿಕಲ್ಪ (ಕೀಲಿ.src_type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                ಯದಿ (ಕೀಲಿ.vectorized) {
                    defines.push_back("F32_VEC");
                    defines.push_back("SRC_TYPE=vec4<f32>");
                    defines.push_back("DST_TYPE=vec4<f32>");
                    defines.push_back("BLOCK_SIZE=4u");
                } ಅನ್ಯಥಾ {
                    defines.push_back("F32");
                    defines.push_back("SRC_TYPE=f32");
                    defines.push_back("DST_TYPE=f32");
                    defines.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_BF16:
                defines.push_back("BF16");
                defines.push_back("SRC_TYPE=u32");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    ವಿಕಲ್ಪ (ಕೀಲಿ.src_type) {
                        ಪಕ್ಷ GGML_TYPE_Q1_0:
                        ಪಕ್ಷ GGML_TYPE_Q2_0:
                        ಪಕ್ಷ GGML_TYPE_Q4_0:
                        ಪಕ್ಷ GGML_TYPE_Q5_0:
                        ಪಕ್ಷ GGML_TYPE_Q8_0:
                        ಪಕ್ಷ GGML_TYPE_Q3_K:
                        ಪಕ್ಷ GGML_TYPE_Q6_K:
                        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
                        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
                        ಪಕ್ಷ GGML_TYPE_IQ2_S:
                        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
                        ಪಕ್ಷ GGML_TYPE_IQ3_S:
                        ಪಕ್ಷ GGML_TYPE_IQ1_S:
                        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                        ಪಕ್ಷ GGML_TYPE_MXFP4:
                        ಪಕ್ಷ GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                ಭಙ್ಗ;
                            }
                        ಸ್ವಭಾವ:
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

                    ಯದಿ (ಕೀಲಿ.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } ಅನ್ಯಥಾ ಯದಿ ((ಕೀಲಿ.src_type >= GGML_TYPE_Q4_0 && ಕೀಲಿ.src_type <= GGML_TYPE_Q8_1) ||
                               ಕೀಲಿ.src_type == GGML_TYPE_IQ4_NL || ಕೀಲಿ.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.src_type == GGML_TYPE_Q2_0 || ಕೀಲಿ.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } ಅನ್ಯಥಾ {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    ಭಙ್ಗ;
                }
        }

        ಯದಿ (ಕೀಲಿ.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        auto decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[ಕೀಲಿ]  = pipeline;
        ಪ್ರತ್ಯಾಗಮ get_rows_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_scale_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = scale_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != scale_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        ಯದಿ (ಕೀಲಿ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_scale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ಕೀಲಿ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[ಕೀಲಿ]     = pipeline;
        ಪ್ರತ್ಯಾಗಮ scale_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_solve_tri_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                               = context.dst->type;
        ಕೀಲಿ.n                                  = (ಪೂರ್ಣಾಙ್ಕ) context.src0->ne[0];
        ಕೀಲಿ.k                                  = (ಪೂರ್ಣಾಙ್ಕ) context.src1->ne[0];

        auto it = solve_tri_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != solve_tri_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                variant += "_f32";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ಧ್ರುವ uint32_t wg_size       = std::min((uint32_t) ಕೀಲಿ.n, context.max_wg_size);
        ಧ್ರುವ uint32_t k_tile        = wg_size;
        ಧ್ರುವ uint32_t bytes_per_row = ((uint32_t) ಕೀಲಿ.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        ಧ್ರುವ uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(ಕೀಲಿ.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        auto processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ solve_tri_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                              = context.dst->type;
        ಕೀಲಿ.vectorized                        = context.src1->ne[0] == 4;

        auto it = ssm_conv_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != ssm_conv_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                variant += "_f32";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        ಯದಿ (ಕೀಲಿ.vectorized) {
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
        ssm_conv_pipelines[ಕೀಲಿ]  = pipeline;
        ಪ್ರತ್ಯಾಗಮ ssm_conv_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_ssm_scan_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                              = context.dst->type;
        ಕೀಲಿ.d_state                           = (ಪೂರ್ಣಾಙ್ಕ) context.src0->ne[0];
        ಕೀಲಿ.xbc_overlap                       = xbc_overlap;

        auto it = ssm_scan_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != ssm_scan_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                variant += "_f32";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ಧ್ರುವ uint32_t wg_size = (uint32_t) ಕೀಲಿ.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        ಯದಿ (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } ಅನ್ಯಥಾ {
            variant += "_wg_reduce";
        }

        ಯದಿ (ಕೀಲಿ.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(ಕೀಲಿ.d_state);

        auto processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        auto decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = ಕೀಲಿ.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[ಕೀಲಿ]    = pipeline;
        ಪ್ರತ್ಯಾಗಮ ssm_scan_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                                     = context.dst->type;
        ಕೀಲಿ.s_v                                      = (ಪೂರ್ಣಾಙ್ಕ) context.src2->ne[0];
        ಕೀಲಿ.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        ಕೀಲಿ.fused_cache                              = context.dst_fuse != nullptr;

        auto it = gated_delta_net_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != gated_delta_net_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                variant += "_f32";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        ಯದಿ (ಕೀಲಿ.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        ಯದಿ (ಕೀಲಿ.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(ಕೀಲಿ.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(ಕೀಲಿ.s_v) + "u");

        auto            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ gated_delta_net_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_pad_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        auto it = pad_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != pad_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        ಯದಿ (ಕೀಲಿ.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_pad, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[ಕೀಲಿ]       = pipeline;
        ಪ್ರತ್ಯಾಗಮ pad_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src0_type                            = context.src0->type;

        auto it = quantize_q8_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != quantize_q8_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }
        ಧ್ರುವ ವರ್ಣ *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> defines;
        std::string              variant = "quantize_q8";

        uint32_t wg_size = WEBGPU_MUL_MAT_VEC_WG_SIZE;

        defines.push_back("SRC1_INNER_TYPE=f32");
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        ಧ್ರುವ ಸಂರಚನಾ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
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
        quantize_q8_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ quantize_q8_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src0_type                            = context.src0->type;
        ಕೀಲಿ.src1_type                            = context.src1->type;
        ಕೀಲಿ.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        ಕೀಲಿ.ಅಂಕೆ_cols   = context.dst->ne[1];
        ಕೀಲಿ.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
        ಕೀಲಿ.src_overlap = src_overlap;

        auto it = mul_mat_vec_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != mul_mat_vec_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        ಧ್ರುವ ವರ್ಣ *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        ವಿಕಲ್ಪ (context.src0->type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ಧ್ರುವ ಸಂರಚನಾ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    ವಿಕಲ್ಪ (context.src0->type) {
                        ಪಕ್ಷ GGML_TYPE_Q4_0:
                        ಪಕ್ಷ GGML_TYPE_Q4_1:
                        ಪಕ್ಷ GGML_TYPE_Q5_0:
                        ಪಕ್ಷ GGML_TYPE_Q5_1:
                        ಪಕ್ಷ GGML_TYPE_Q8_0:
                            ಯದಿ (ಕೀಲಿ.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_Q1_0:
                        ಪಕ್ಷ GGML_TYPE_Q2_K:
                        ಪಕ್ಷ GGML_TYPE_Q3_K:
                        ಪಕ್ಷ GGML_TYPE_Q4_K:
                        ಪಕ್ಷ GGML_TYPE_Q5_K:
                        ಪಕ್ಷ GGML_TYPE_Q6_K:
                            ಯದಿ (ಕೀಲಿ.use_mmvq) {
                                defines.push_back("K_QUANTS_HANDLING");
                            }
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_IQ1_S:
                        ಪಕ್ಷ GGML_TYPE_IQ1_M:
                        ಪಕ್ಷ GGML_TYPE_IQ2_S:
                        ಪಕ್ಷ GGML_TYPE_IQ3_S:
                        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
                        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
                        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_MXFP4:
                            defines.push_back(type_upper + "_LUT");
                            ಯದಿ (ಕೀಲಿ.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ಭಙ್ಗ;
                        ಸ್ವಭಾವ:
                            ಭಙ್ಗ;
                    }
                    ಭಙ್ಗ;
                }
        }

        // src1 type (vector)
        ವಿಕಲ್ಪ (context.src1->type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(ಕೀಲಿ.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        ಯದಿ (ಕೀಲಿ.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        ಯದಿ (ಕೀಲಿ.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        ಯದಿ (ಕೀಲಿ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        ಯದಿ (ಕೀಲಿ.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(ಕೀಲಿ.ಅಂಕೆ_cols));

        auto processed            = preprocessor.preprocess(shader_src, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = ಕೀಲಿ.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ mul_mat_vec_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src0_type                        = context.src0->type;
        ಕೀಲಿ.src1_type                        = context.src1->type;
        ಕೀಲಿ.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        ಕೀಲಿ.use_subgroup_matrix = context.supports_subgroup_matrix;
        ಕೀಲಿ.src_overlap         = src_overlap;

        auto it = mul_mat_fast_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != mul_mat_fast_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        ಧ್ರುವ ವರ್ಣ * shader_src = ಕೀಲಿ.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = ಕೀಲಿ.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        ವಿಕಲ್ಪ (context.src1->type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ಧ್ರುವ ಸಂರಚನಾ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ಧ್ರುವ ವರ್ಣ *                    src0_name   = src0_traits->type_name;

        ವಿಕಲ್ಪ (context.src0->type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("TYPE_F32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("TYPE_F16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("TYPE_BF16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    ವಿಕಲ್ಪ (context.src0->type) {
                        ಪಕ್ಷ GGML_TYPE_IQ1_S:
                        ಪಕ್ಷ GGML_TYPE_IQ1_M:
                        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
                        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
                        ಪಕ್ಷ GGML_TYPE_IQ2_S:
                        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
                        ಪಕ್ಷ GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_MXFP4:
                        ಪಕ್ಷ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ಭಙ್ಗ;
                        ಸ್ವಭಾವ:
                            ಭಙ್ಗ;
                    }

                    variant += std::string("_") + src0_name;
                    ಭಙ್ಗ;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(ಕೀಲಿ.vectorized ? "VEC" : "SCALAR");

        ಧ್ರುವ ತಾರ್ಕಿಕ is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        ಯದಿ (ಕೀಲಿ.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } ಅನ್ಯಥಾ {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        ಯದಿ (ಕೀಲಿ.use_subgroup_matrix) {
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
        ಯದಿ (ಕೀಲಿ.vectorized) {
            variant += "_vectorized";
        }

        ಯದಿ (ಕೀಲಿ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ಯದಿ (!ಕೀಲಿ.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = ಕೀಲಿ.use_subgroup_matrix;
        ಯದಿ (ಕೀಲಿ.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_size           = context.max_subgroup_size;
        } ಅನ್ಯಥಾ {
            decisions->wg_size_m       = WEBGPU_MUL_MAT_WG_SIZE_M;
            decisions->wg_size_n       = WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->wg_size         = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->mul_mat_wg_size = WEBGPU_MUL_MAT_WG_SIZE;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ mul_mat_fast_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.n_experts                                  = (uint32_t) context.src0->ne[2];

        auto it = mul_mat_id_gather_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != mul_mat_id_gather_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(ಕೀಲಿ.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        auto processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src0_type                           = context.src0->type;
        ಕೀಲಿ.src1_type                           = context.src1->type;
        ಕೀಲಿ.n_experts                           = context.src0->ne[2];
        ಕೀಲಿ.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != mul_mat_id_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        ವಿಕಲ್ಪ (context.src1->type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ಧ್ರುವ ಸಂರಚನಾ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ಧ್ರುವ ವರ್ಣ *                    src0_name   = src0_traits->type_name;

        ವಿಕಲ್ಪ (context.src0->type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    ವಿಕಲ್ಪ (context.src0->type) {
                        ಪಕ್ಷ GGML_TYPE_IQ1_S:
                        ಪಕ್ಷ GGML_TYPE_IQ1_M:
                        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
                        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
                        ಪಕ್ಷ GGML_TYPE_IQ2_S:
                        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
                        ಪಕ್ಷ GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_MXFP4:
                        ಪಕ್ಷ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ಭಙ್ಗ;
                        ಸ್ವಭಾವ:
                            ಭಙ್ಗ;
                    }

                    variant += std::string("_") + src0_name;
                    ಭಙ್ಗ;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(ಕೀಲಿ.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ಧ್ರುವ uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        ಯದಿ (ಕೀಲಿ.vectorized) {
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
        mul_mat_id_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ mul_mat_id_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src0_type                           = context.src0->type;
        ಕೀಲಿ.src1_type                           = context.src1->type;
        ಕೀಲಿ.n_experts                           = context.src0->ne[2];
        ಕೀಲಿ.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_vec_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != mul_mat_id_vec_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        ಧ್ರುವ ವರ್ಣ *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        ವಿಕಲ್ಪ (context.src1->type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ವಿಕಲ್ಪ (context.src0->type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ಧ್ರುವ ಸಂರಚನಾ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    ವಿಕಲ್ಪ (context.src0->type) {
                        ಪಕ್ಷ GGML_TYPE_IQ1_S:
                        ಪಕ್ಷ GGML_TYPE_IQ1_M:
                        ಪಕ್ಷ GGML_TYPE_IQ2_S:
                        ಪಕ್ಷ GGML_TYPE_IQ3_S:
                        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
                        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
                        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ಭಙ್ಗ;
                        ಪಕ್ಷ GGML_TYPE_MXFP4:
                        ಪಕ್ಷ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ಭಙ್ಗ;
                        ಸ್ವಭಾವ:
                            ಭಙ್ಗ;
                    }
                    ಭಙ್ಗ;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(ಕೀಲಿ.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        ಯದಿ (ಕೀಲಿ.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        ಯದಿ (ಕೀಲಿ.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(ಕೀಲಿ.n_experts));

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ mul_mat_id_vec_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_unary_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ಧ್ರುವ ತಾರ್ಕಿಕ                     is_unary = context.dst->op == GGML_OP_UNARY;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                      op       = is_unary ? (ಪೂರ್ಣಾಙ್ಕ) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_ಕೀಲಿ ಕೀಲಿ      = {};
        ಕೀಲಿ.type                                = context.dst->type;
        ಕೀಲಿ.op                                  = op;
        ಕೀಲಿ.is_unary                            = is_unary;
        ಕೀಲಿ.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        ಕೀಲಿ.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        auto it = unary_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != unary_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            ಕೀಲಿ.is_unary ? ggml_unary_op_name((ggml_unary_op) ಕೀಲಿ.op) : ggml_op_name((ggml_op) ಕೀಲಿ.op);
        defines.push_back(variant);

        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for unary shader");
        }

        ಯದಿ (ಕೀಲಿ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        ಯದಿ (op == GGML_OP_TRI) {
            ವಿಕಲ್ಪ (ಕೀಲಿ.ttype) {
                ಪಕ್ಷ GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    ಭಙ್ಗ;
                ಪಕ್ಷ GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    ಭಙ್ಗ;
                ಪಕ್ಷ GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    ಭಙ್ಗ;
                ಪಕ್ಷ GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    ಭಙ್ಗ;
                ಸ್ವಭಾವ:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_unary, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ಕೀಲಿ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[ಕೀಲಿ]     = pipeline;
        ಪ್ರತ್ಯಾಗಮ unary_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ಕೀಲಿ.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ಕೀಲಿ.src_overlap                           = src_overlap;

        auto it = rms_norm_mul_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != rms_norm_mul_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        ಯದಿ (ಕೀಲಿ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = ಕೀಲಿ.inplace;
        pipeline_decisions->overlap     = ಕೀಲಿ.overlap;
        pipeline_decisions->src_overlap = ಕೀಲಿ.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[ಕೀಲಿ]     = pipeline;
        ಪ್ರತ್ಯಾಗಮ rms_norm_mul_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_ದ್ವಿಮಾನ_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                            = context.dst->type;
        ಕೀಲಿ.op                              = context.dst->op;
        ಕೀಲಿ.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ಕೀಲಿ.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ಕೀಲಿ.src_overlap                     = src_overlap;

        auto it = ದ್ವಿಮಾನ_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != ದ್ವಿಮಾನ_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) ಕೀಲಿ.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for binary shader");
        }

        ಯದಿ (ಕೀಲಿ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_ದ್ವಿಮಾನ, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_ದ್ವಿಮಾನ_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = ಕೀಲಿ.inplace;
        pipeline_decisions->overlap     = ಕೀಲಿ.overlap;
        pipeline_decisions->src_overlap = ಕೀಲಿ.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ದ್ವಿಮಾನ_pipelines[ಕೀಲಿ]    = pipeline;
        ಪ್ರತ್ಯಾಗಮ ದ್ವಿಮಾನ_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_add_id_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = add_id_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != add_id_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        ಧ್ರುವ ವರ್ಣ *             shader_src = wgsl_add_id;

        ಯದಿ (ಕೀಲಿ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed              = preprocessor.preprocess(shader_src, defines);
        auto pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_size = context.max_wg_size;
        pipeline_decisions->inplace = ಕೀಲಿ.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[ಕೀಲಿ]    = pipeline;
        ಪ್ರತ್ಯಾಗಮ pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_concat_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                            = context.dst->type;
        ಕೀಲಿ.src_overlap                     = src_overlap;

        auto it = concat_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != concat_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for concat shader");
        }

        ಯದಿ (ಕೀಲಿ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_concat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_ದ್ವಿಮಾನ_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = ಕೀಲಿ.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[ಕೀಲಿ]    = pipeline;
        ಪ್ರತ್ಯಾಗಮ concat_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_repeat_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                            = context.dst->type;

        auto it = repeat_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != repeat_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_repeat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[ಕೀಲಿ]    = pipeline;
        ಪ್ರತ್ಯಾಗಮ repeat_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_flash_attn_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ kv_overlap) {
        ಧ್ರುವ ತಾರ್ಕಿಕ can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.common                              = ggml_webgpu_flash_attn_make_common_pipeline_ಕೀಲಿ(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        ಕೀಲಿ.common.k_direct &= decisions.use_sg_matrix && ಕೀಲಿ.common.k_type == GGML_TYPE_F16;
        ಕೀಲಿ.common.v_direct &= decisions.use_sg_matrix && ಕೀಲಿ.common.v_type == GGML_TYPE_F16;
        ಕೀಲಿ.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_limit_bytes, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            ಕೀಲಿ.common.head_dim_qk, ಕೀಲಿ.common.head_dim_v, ಕೀಲಿ.common.has_mask,
            ಕೀಲಿ.common.k_direct || ಕೀಲಿ.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        ಯದಿ (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = ಅಸತ್ಯ;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            ಕೀಲಿ.common.k_direct     = ಅಸತ್ಯ;
            ಕೀಲಿ.common.v_direct     = ಅಸತ್ಯ;
            ಕೀಲಿ.use_sg_matrix       = ಅಸತ್ಯ;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, ಕೀಲಿ.common.head_dim_qk, ಕೀಲಿ.common.head_dim_v,
                ಕೀಲಿ.common.has_mask, ಕೀಲಿ.common.k_direct || ಕೀಲಿ.common.v_direct);
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

        ಯದಿ (ಕೀಲಿ.common.k_direct || ಕೀಲಿ.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            ಯಾವತ್ (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_size;
            }
        }

        auto it = flash_attn_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != flash_attn_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(ಕೀಲಿ.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_size);
        ಧ್ರುವ ವರ್ಣ *             shader_src = nullptr;
        ಯದಿ (!ಕೀಲಿ.use_sg_matrix) {
            shader_src = wgsl_flash_attn_tile;
            defines.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_size) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_size) + "_" +
                       std::to_string(context.max_subgroup_size);
        } ಅನ್ಯಥಾ {
            shader_src = wgsl_flash_attn;
            defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ flash_attn_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.common = ggml_webgpu_flash_attn_make_common_pipeline_ಕೀಲಿ(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        auto it = flash_attn_vec_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != flash_attn_vec_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, ಕೀಲಿ.common.head_dim_qk,
                                                                   ಕೀಲಿ.common.head_dim_v, ಕೀಲಿ.common.has_mask,
                                                                   ಕೀಲಿ.common.k_direct || ಕೀಲಿ.common.v_direct);
        decisions.wg_size = context.max_subgroup_size;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(ಕೀಲಿ.common, variant, 1u, decisions.kv_tile, decisions.wg_size);
        ಯದಿ (ಕೀಲಿ.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (ಪರಿಮಾಣ("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_size;
        ಧ್ರುವ uint32_t D       = ಕೀಲಿ.common.head_dim_qk | ಕೀಲಿ.common.head_dim_v;
        ಧ್ರುವ uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_size, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ flash_attn_vec_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.kv_tile                                 = kv_tile;
        auto it                                     = flash_attn_blk_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != flash_attn_blk_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(ಕೀಲಿ.kv_tile));
        variant += std::string("_kvt") + std::to_string(ಕೀಲಿ.kv_tile);

        uint32_t wg_size = 1;
        ಯಾವತ್ ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ flash_attn_blk_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        ಕೀಲಿ.dst_type                                       = context.dst->type;
        ಕೀಲಿ.wg_size                                        = context.max_wg_size;
        auto it                                            = flash_attn_vec_reduce_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != flash_attn_vec_reduce_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        ವಿಕಲ್ಪ (ಕೀಲಿ.dst_type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(ಕೀಲಿ.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(ಕೀಲಿ.head_dim_v));
        variant += std::string("_hsv") + std::to_string(ಕೀಲಿ.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ flash_attn_vec_reduce_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_cpy_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src_type                     = context.src0->type;
        ಕೀಲಿ.dst_type                     = context.dst->type;

        auto it = cpy_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != cpy_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        ವಿಕಲ್ಪ (ಕೀಲಿ.src_type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_I32:
                defines.push_back("SRC_I32");
                variant += "_i32";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        ವಿಕಲ್ಪ (ಕೀಲಿ.dst_type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_cpy, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[ಕೀಲಿ]       = pipeline;
        ಪ್ರತ್ಯಾಗಮ cpy_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_glu_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_glu_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.glu_op                       = ggml_get_glu_op(context.dst);
        ಕೀಲಿ.type                         = context.dst->type;
        ಕೀಲಿ.split                        = (context.src1 != nullptr);
        ಕೀಲಿ.src_overlap                  = src_overlap;

        auto it = glu_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != glu_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        ವಿಕಲ್ಪ (ಕೀಲಿ.glu_op) {
            ಪಕ್ಷ GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_GLU_OP_SWIGLU_CLAMP:
                defines.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported GLU op");
        }
        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        ಯದಿ (ಕೀಲಿ.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } ಅನ್ಯಥಾ ಯದಿ (ಕೀಲಿ.split) {
            variant += "_split";
        } ಅನ್ಯಥಾ {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_glu, defines);
        auto decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = ಕೀಲಿ.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[ಕೀಲಿ]       = pipeline;
        ಪ್ರತ್ಯಾಗಮ glu_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_rope_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                          = context.dst->type;
        ಕೀಲಿ.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ಕೀಲಿ.has_ff                        = (context.src2 != nullptr);

        auto it = rope_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != rope_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        ವಿಕಲ್ಪ (ಕೀಲಿ.type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        ಯದಿ (ಕೀಲಿ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        ಯದಿ (ಕೀಲಿ.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_rope, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ಕೀಲಿ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[ಕೀಲಿ]      = pipeline;
        ಪ್ರತ್ಯಾಗಮ rope_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_soft_max_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        ಕೀಲಿ.has_mask                          = (context.src1 != nullptr);
        ಕೀಲಿ.has_sink                          = (context.src2 != nullptr);
        ಕೀಲಿ.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = soft_max_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != soft_max_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        ಯದಿ (ಕೀಲಿ.has_mask) {
            defines.push_back("HAS_MASK");
            ವಿಕಲ್ಪ (ಕೀಲಿ.mask_type) {
                ಪಕ್ಷ GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    ಭಙ್ಗ;
                ಪಕ್ಷ GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    ಭಙ್ಗ;
                ಸ್ವಭಾವ:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        ಯದಿ (ಕೀಲಿ.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        ಯದಿ (ಕೀಲಿ.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ಕೀಲಿ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[ಕೀಲಿ]  = pipeline;
        ಪ್ರತ್ಯಾಗಮ soft_max_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_conv2d_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.weight_type                     = context.src0->type;
        ಕೀಲಿ.ಒಳಹರಿವು_type                      = context.src1->type;
        ಕೀಲಿ.output_type                     = context.dst->type;

        auto it = conv2d_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != conv2d_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        auto ತಳ್ಳು_type_defines = [&](ಧ್ರುವ ವರ್ಣ * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            ಯದಿ (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } ಅನ್ಯಥಾ ಯದಿ (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } ಅನ್ಯಥಾ {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ತಳ್ಳು_type_defines("WEIGHT_TYPE", ಕೀಲಿ.weight_type);
        ತಳ್ಳು_type_defines("INPUT_TYPE", ಕೀಲಿ.ಒಳಹರಿವು_type);
        ತಳ್ಳು_type_defines("OUTPUT_TYPE", ಕೀಲಿ.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[ಕೀಲಿ]    = pipeline;
        ಪ್ರತ್ಯಾಗಮ conv2d_pipelines[ಕೀಲಿ];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ whcn) {
        ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.weight_type                        = context.src0->type;
        ಕೀಲಿ.ಒಳಹರಿವು_type                         = context.src1->type;
        ಕೀಲಿ.output_type                        = context.dst->type;
        ಕೀಲಿ.whcn                               = whcn;

        auto it = conv2d_dw_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != conv2d_dw_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        auto ತಳ್ಳು_type_defines = [&](ಧ್ರುವ ವರ್ಣ * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            ಯದಿ (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } ಅನ್ಯಥಾ ಯದಿ (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } ಅನ್ಯಥಾ {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ತಳ್ಳು_type_defines("WEIGHT_TYPE", ಕೀಲಿ.weight_type);
        ತಳ್ಳು_type_defines("INPUT_TYPE", ಕೀಲಿ.ಒಳಹರಿವು_type);
        ತಳ್ಳು_type_defines("OUTPUT_TYPE", ಕೀಲಿ.output_type);

        ಯದಿ (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[ಕೀಲಿ] = pipeline;
        ಪ್ರತ್ಯಾಗಮ conv2d_dw_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_im2col_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.ಒಳಹರಿವು_type                      = context.src1->type;
        ಕೀಲಿ.output_type                     = context.dst->type;

        auto it = im2col_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != im2col_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        auto ತಳ್ಳು_type_defines = [&](ಧ್ರುವ ವರ್ಣ * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            ಯದಿ (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } ಅನ್ಯಥಾ ಯದಿ (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } ಅನ್ಯಥಾ {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ತಳ್ಳು_type_defines("INPUT_TYPE", ಕೀಲಿ.ಒಳಹರಿವು_type);
        ತಳ್ಳು_type_defines("OUTPUT_TYPE", ಕೀಲಿ.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[ಕೀಲಿ]    = pipeline;
        ಪ್ರತ್ಯಾಗಮ im2col_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_upscale_pipeline(ಧ್ರುವ ggml_webgpu_shader_lib_context & context) {
        ಧ್ರುವ uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ಧ್ರುವ uint32_t base_mode  = mode_flags & 0xFFu;
        ಧ್ರುವ ತಾರ್ಕಿಕ     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.ಒಳಹರಿವು_type                       = context.src0->type;
        ಕೀಲಿ.output_type                      = context.dst->type;
        ಕೀಲಿ.base_mode                        = base_mode;
        ಕೀಲಿ.antialias                        = antialias;

        auto it = upscale_pipelines.find(ಕೀಲಿ);
        ಯದಿ (it != upscale_pipelines.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        ಯದಿ (ಕೀಲಿ.ಒಳಹರಿವು_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } ಅನ್ಯಥಾ {
            variant += "_src_f32";
        }

        ಯದಿ (ಕೀಲಿ.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } ಅನ್ಯಥಾ {
            variant += "_dst_f32";
        }

        ವಿಕಲ್ಪ (base_mode) {
            ಪಕ್ಷ GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("Unsupported upscale mode");
        }

        ಯದಿ (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_upscale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[ಕೀಲಿ]   = pipeline;
        ಪ್ರತ್ಯಾಗಮ upscale_pipelines[ಕೀಲಿ];
    }

  ಗುಪ್ತ:
    ಸ್ಥಿರ webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
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
        ಪ್ರತ್ಯಾಗಮ { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
