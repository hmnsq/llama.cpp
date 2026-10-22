#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-wgsl-shaders.hpp"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "pre_wgsl.hpp"

#अन्तर्भाव <webgpu/webgpu_cpp.h>

#अन्तर्भाव <algorithm>
#अन्तर्भाव <memory>
#अन्तर्भाव <string>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <vector>

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
ढाँचा <typename T> अन्तःस्थ शून्य ggml_webgpu_hash_combine(माप_प्रकार & seed, ध्रुव T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
अन्तःस्थ uintptr_t ggml_webgpu_tensor_addr(ध्रुव ggml_tensor * tensor) {
    ध्रुव ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    प्रत्यागम (uintptr_t) base_tensor->data + tensor->view_offs;
}

अन्तःस्थ तार्किक ggml_webgpu_tensor_equal(ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
    प्रत्यागम a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

संरचना ggml_webgpu_shader_lib_context {
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * src2;
    ggml_tensor * src3;
    ggml_tensor * src4;
    ggml_tensor * src5;
    ggml_tensor * dst;
    ggml_tensor * dst_fuse;

    uint32_t    max_wg_size;
    माप_प्रकार      wg_mem_limit_bytes       = 0;
    तार्किक        supports_subgroups       = असत्य;
    तार्किक        supports_subgroup_matrix = असत्य;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_size        = 0;
    uint32_t    max_subgroup_size        = 0;
    तार्किक        supports_dot_product     = असत्य;
    std::string vendor;
};

संरचना webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<शून्य> context = nullptr;
};

संरचना ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 0;
    तार्किक     inplace = असत्य;
};

संरचना ggml_webgpu_get_rows_shader_decisions {
    uint32_t wg_size    = 0;
    तार्किक     vectorized = असत्य;
};

संरचना ggml_webgpu_द्विआधारी_shader_decisions {
    uint32_t wg_size     = 0;
    तार्किक     inplace     = असत्य;
    तार्किक     overlap     = असत्य;
    तार्किक     src_overlap = असत्य;
};

संरचना ggml_webgpu_glu_shader_decisions {
    uint32_t wg_size     = 0;
    तार्किक     src_overlap = असत्य;
};

संरचना ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<शून्य> decisions;
};

संरचना ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_size;
    uint32_t tokens_per_wg;
};

संरचना ggml_webgpu_ssm_scan_pipeline_कुंजी {
    पूर्णाङ्क     type;
    पूर्णाङ्क     d_state;
    uint8_t xbc_overlap;
    तार्किक    a_overlap;
    तार्किक    ids_overlap;

    तार्किक चालक==(ध्रुव ggml_webgpu_ssm_scan_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

संरचना ggml_webgpu_ssm_scan_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_ssm_scan_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.d_state);
        ggml_webgpu_hash_combine(seed, कुंजी.xbc_overlap);
        ggml_webgpu_hash_combine(seed, कुंजी.a_overlap);
        ggml_webgpu_hash_combine(seed, कुंजी.ids_overlap);
        प्रत्यागम seed;
    }
};

संरचना ggml_webgpu_ssm_scan_shader_decisions {
    uint32_t wg_size;
    uint32_t tokens_per_tile;
    uint8_t  xbc_overlap = 0;
    तार्किक     a_overlap   = असत्य;
    तार्किक     ids_overlap = असत्य;
};

/** Argsort **/

संरचना ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    माप_प्रकार   wg_mem_limit_bytes;
    int32_t  order;
};

/** Set Rows **/

संरचना ggml_webgpu_set_rows_pipeline_कुंजी {
    पूर्णाङ्क src0_type;
    पूर्णाङ्क dst_type;
    पूर्णाङ्क vec4;
    पूर्णाङ्क i64_idx;
    पूर्णाङ्क pair_blocks;

    तार्किक चालक==(ध्रुव ggml_webgpu_set_rows_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

संरचना ggml_webgpu_set_rows_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_set_rows_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src0_type);
        ggml_webgpu_hash_combine(seed, कुंजी.dst_type);
        ggml_webgpu_hash_combine(seed, कुंजी.vec4);
        ggml_webgpu_hash_combine(seed, कुंजी.i64_idx);
        ggml_webgpu_hash_combine(seed, कुंजी.pair_blocks);
        प्रत्यागम seed;
    }
};

संरचना ggml_webgpu_set_rows_shader_decisions {
    तार्किक     vec4;
    तार्किक     i64_idx;
    तार्किक     pair_blocks;
    uint32_t wg_size;
};

/** Set **/

संरचना ggml_webgpu_set_pipeline_कुंजी {
    ggml_type type;
    तार्किक      inplace;

    तार्किक चालक==(ध्रुव ggml_webgpu_set_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम type == other.type && inplace == other.inplace;
    }
};

संरचना ggml_webgpu_set_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_set_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        प्रत्यागम seed;
    }
};

/** Get Rows **/

संरचना ggml_webgpu_get_rows_pipeline_कुंजी {
    ggml_type src_type;
    पूर्णाङ्क       vectorized;

    तार्किक चालक==(ध्रुव ggml_webgpu_get_rows_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम src_type == other.src_type && vectorized == other.vectorized;
    }
};

संरचना ggml_webgpu_get_rows_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_get_rows_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src_type);
        ggml_webgpu_hash_combine(seed, कुंजी.vectorized);
        प्रत्यागम seed;
    }
};

/** Row Norm **/

संरचना ggml_webgpu_row_norm_pipeline_कुंजी {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    तार्किक      inplace;

    तार्किक चालक==(ध्रुव ggml_webgpu_row_norm_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

संरचना ggml_webgpu_row_norm_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_row_norm_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.op);
        ggml_webgpu_hash_combine(seed, कुंजी.src_type);
        ggml_webgpu_hash_combine(seed, कुंजी.dst_type);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        प्रत्यागम seed;
    }
};

/** RMS_NORM + MUL **/

संरचना ggml_webgpu_rms_norm_mul_pipeline_कुंजी {
    तार्किक inplace;      // rn_src == dst
    तार्किक overlap;      // mul_src == dst
    तार्किक src_overlap;  // rn_src binding overlaps mul_src binding

    तार्किक चालक==(ध्रुव ggml_webgpu_rms_norm_mul_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_rms_norm_mul_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_rms_norm_mul_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        ggml_webgpu_hash_combine(seed, कुंजी.overlap);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        प्रत्यागम seed;
    }
};

संरचना ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    तार्किक     inplace     = असत्य;
    तार्किक     overlap     = असत्य;
    तार्किक     src_overlap = असत्य;
};

/** Pad **/
संरचना ggml_webgpu_pad_pipeline_कुंजी {
    तार्किक circular;

    तार्किक चालक==(ध्रुव ggml_webgpu_pad_pipeline_कुंजी & other) ध्रुव { प्रत्यागम circular == other.circular; }
};

संरचना ggml_webgpu_pad_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_pad_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.circular);
        प्रत्यागम seed;
    }
};

/** Solve Tri **/
संरचना ggml_webgpu_solve_tri_pipeline_कुंजी {
    पूर्णाङ्क type;
    पूर्णाङ्क n;
    पूर्णाङ्क k;

    तार्किक चालक==(ध्रुव ggml_webgpu_solve_tri_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम type == other.type && n == other.n && k == other.k;
    }
};

संरचना ggml_webgpu_solve_tri_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_solve_tri_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.n);
        ggml_webgpu_hash_combine(seed, कुंजी.k);
        प्रत्यागम seed;
    }
};

/** SSM Conv **/
संरचना ggml_webgpu_ssm_conv_pipeline_कुंजी {
    पूर्णाङ्क type;
    पूर्णाङ्क vectorized;

    तार्किक चालक==(ध्रुव ggml_webgpu_ssm_conv_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
संरचना ggml_webgpu_conv2d_pipeline_कुंजी {
    ggml_type weight_type;
    ggml_type निवेश_type;
    ggml_type output_type;

    तार्किक चालक==(ध्रुव ggml_webgpu_conv2d_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम weight_type == other.weight_type && निवेश_type == other.निवेश_type && output_type == other.output_type;
    }
};

संरचना ggml_webgpu_conv2d_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_conv2d_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.weight_type);
        ggml_webgpu_hash_combine(seed, कुंजी.निवेश_type);
        ggml_webgpu_hash_combine(seed, कुंजी.output_type);
        प्रत्यागम seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
संरचना ggml_webgpu_conv2d_dw_pipeline_कुंजी {
    ggml_type weight_type;
    ggml_type निवेश_type;
    ggml_type output_type;
    तार्किक      whcn;

    तार्किक चालक==(ध्रुव ggml_webgpu_conv2d_dw_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम weight_type == other.weight_type && निवेश_type == other.निवेश_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

संरचना ggml_webgpu_conv2d_dw_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_conv2d_dw_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.weight_type);
        ggml_webgpu_hash_combine(seed, कुंजी.निवेश_type);
        ggml_webgpu_hash_combine(seed, कुंजी.output_type);
        ggml_webgpu_hash_combine(seed, कुंजी.whcn);
        प्रत्यागम seed;
    }
};

/** Im2Col **/
संरचना ggml_webgpu_im2col_pipeline_कुंजी {
    ggml_type निवेश_type;
    ggml_type output_type;

    तार्किक चालक==(ध्रुव ggml_webgpu_im2col_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम निवेश_type == other.निवेश_type && output_type == other.output_type;
    }
};

संरचना ggml_webgpu_im2col_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_im2col_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.निवेश_type);
        ggml_webgpu_hash_combine(seed, कुंजी.output_type);
        प्रत्यागम seed;
    }
};

/** Gated Delta Net **/
संरचना ggml_webgpu_gated_delta_net_pipeline_कुंजी {
    पूर्णाङ्क  type;
    पूर्णाङ्क  s_v;
    पूर्णाङ्क  kda;
    तार्किक fused_cache;

    तार्किक चालक==(ध्रुव ggml_webgpu_gated_delta_net_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

संरचना ggml_webgpu_gated_delta_net_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_gated_delta_net_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.s_v);
        ggml_webgpu_hash_combine(seed, कुंजी.kda);
        प्रत्यागम seed;
    }
};

संरचना ggml_webgpu_ssm_conv_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_ssm_conv_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.vectorized);
        प्रत्यागम seed;
    }
};

/** Scale **/

संरचना ggml_webgpu_scale_pipeline_कुंजी {
    पूर्णाङ्क inplace;

    तार्किक चालक==(ध्रुव ggml_webgpu_scale_pipeline_कुंजी & other) ध्रुव { प्रत्यागम inplace == other.inplace; }
};

संरचना ggml_webgpu_scale_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_scale_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        प्रत्यागम seed;
    }
};

/** Upscale **/

संरचना ggml_webgpu_upscale_pipeline_कुंजी {
    ggml_type निवेश_type;
    ggml_type output_type;
    uint32_t  base_mode;
    तार्किक      antialias;

    तार्किक चालक==(ध्रुव ggml_webgpu_upscale_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम निवेश_type == other.निवेश_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

संरचना ggml_webgpu_upscale_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_upscale_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.निवेश_type);
        ggml_webgpu_hash_combine(seed, कुंजी.output_type);
        ggml_webgpu_hash_combine(seed, कुंजी.base_mode);
        ggml_webgpu_hash_combine(seed, कुंजी.antialias);
        प्रत्यागम seed;
    }
};

/** Concat **/

संरचना ggml_webgpu_concat_pipeline_कुंजी {
    पूर्णाङ्क  type;
    तार्किक src_overlap;

    तार्किक चालक==(ध्रुव ggml_webgpu_concat_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम type == other.type && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_concat_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_concat_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        प्रत्यागम seed;
    }
};

/** Repeat **/

संरचना ggml_webgpu_repeat_pipeline_कुंजी {
    पूर्णाङ्क type;

    तार्किक चालक==(ध्रुव ggml_webgpu_repeat_pipeline_कुंजी & other) ध्रुव { प्रत्यागम type == other.type; }
};

संरचना ggml_webgpu_repeat_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_repeat_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        प्रत्यागम seed;
    }
};

/** Binary **/

संरचना ggml_webgpu_द्विआधारी_pipeline_कुंजी {
    पूर्णाङ्क  type;
    पूर्णाङ्क  op;
    तार्किक inplace;
    तार्किक overlap;
    तार्किक src_overlap;

    तार्किक चालक==(ध्रुव ggml_webgpu_द्विआधारी_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_द्विआधारी_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_द्विआधारी_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.op);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        ggml_webgpu_hash_combine(seed, कुंजी.overlap);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        प्रत्यागम seed;
    }
};

/* Add_Id */

संरचना ggml_webgpu_add_id_pipeline_कुंजी {
    तार्किक inplace;

    तार्किक चालक==(ध्रुव ggml_webgpu_add_id_pipeline_कुंजी & other) ध्रुव { प्रत्यागम inplace == other.inplace; }
};

संरचना ggml_webgpu_add_id_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_add_id_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        प्रत्यागम seed;
    }
};

/** Unary **/

संरचना ggml_webgpu_unary_pipeline_कुंजी {
    पूर्णाङ्क           type;
    पूर्णाङ्क           op;
    तार्किक          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    तार्किक          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    तार्किक चालक==(ध्रुव ggml_webgpu_unary_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

संरचना ggml_webgpu_unary_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_unary_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.op);
        ggml_webgpu_hash_combine(seed, कुंजी.is_unary);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        ggml_webgpu_hash_combine(seed, कुंजी.ttype);
        प्रत्यागम seed;
    }
};

/** FlashAttention */

संरचना ggml_webgpu_flash_attn_common_pipeline_कुंजी {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    तार्किक      k_direct;
    तार्किक      v_direct;
    तार्किक      kv_overlap;
    तार्किक      has_mask;
    तार्किक      has_sinks;
    तार्किक      uses_logit_softcap;

    तार्किक चालक==(ध्रुव ggml_webgpu_flash_attn_common_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

अन्तःस्थ शून्य ggml_webgpu_flash_attn_hash_common_pipeline_कुंजी(माप_प्रकार &                                           seed,
                                                            ध्रुव ggml_webgpu_flash_attn_common_pipeline_कुंजी & कुंजी) {
    ggml_webgpu_hash_combine(seed, कुंजी.q_type);
    ggml_webgpu_hash_combine(seed, कुंजी.k_type);
    ggml_webgpu_hash_combine(seed, कुंजी.v_type);
    ggml_webgpu_hash_combine(seed, कुंजी.dst_type);
    ggml_webgpu_hash_combine(seed, कुंजी.head_dim_qk);
    ggml_webgpu_hash_combine(seed, कुंजी.head_dim_v);
    ggml_webgpu_hash_combine(seed, कुंजी.k_direct);
    ggml_webgpu_hash_combine(seed, कुंजी.v_direct);
    ggml_webgpu_hash_combine(seed, कुंजी.kv_overlap);
    ggml_webgpu_hash_combine(seed, कुंजी.has_mask);
    ggml_webgpu_hash_combine(seed, कुंजी.has_sinks);
    ggml_webgpu_hash_combine(seed, कुंजी.uses_logit_softcap);
}

संरचना ggml_webgpu_flash_attn_vec_pipeline_कुंजी {
    ggml_webgpu_flash_attn_common_pipeline_कुंजी common;

    तार्किक चालक==(ध्रुव ggml_webgpu_flash_attn_vec_pipeline_कुंजी & other) ध्रुव { प्रत्यागम common == other.common; }
};

संरचना ggml_webgpu_flash_attn_vec_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_flash_attn_vec_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_कुंजी(seed, कुंजी.common);
        प्रत्यागम seed;
    }
};

संरचना ggml_webgpu_flash_attn_pipeline_कुंजी {
    ggml_webgpu_flash_attn_common_pipeline_कुंजी common;
    तार्किक                                       use_sg_matrix;

    तार्किक चालक==(ध्रुव ggml_webgpu_flash_attn_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

संरचना ggml_webgpu_flash_attn_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_flash_attn_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_कुंजी(seed, कुंजी.common);
        ggml_webgpu_hash_combine(seed, कुंजी.use_sg_matrix);
        प्रत्यागम seed;
    }
};

संरचना ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

संरचना ggml_webgpu_flash_attn_decisions {
    तार्किक     use_sg_matrix = असत्य;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_size       = 0;
};

अन्तःस्थ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
अन्तःस्थ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

अन्तःस्थ माप_प्रकार ggml_webgpu_flash_attn_tensor_offset(ध्रुव ggml_tensor * tensor) {
    constexpr uintptr_t ptr_base_addr = 0x1000u;
    ध्रुव ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
    प्रत्यागम reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
}

अन्तःस्थ तार्किक ggml_webgpu_flash_attn_float_vec4_aligned(ध्रुव ggml_tensor * K, माप_प्रकार storage_offset_alignment) {
    ध्रुव uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_size(K->type));
    प्रत्यागम offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

अन्तःस्थ तार्किक ggml_webgpu_flash_attn_float_vec4_aligned(ध्रुव ggml_tensor * K,
                                                      ध्रुव ggml_tensor * V,
                                                      माप_प्रकार              storage_offset_alignment) {
    प्रत्यागम ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

अन्तःस्थ तार्किक ggml_webgpu_flash_attn_k_direct(ध्रुव ggml_tensor * Q, ध्रुव ggml_tensor * K, uint32_t kv_direct_align) {
    प्रत्यागम (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

अन्तःस्थ तार्किक ggml_webgpu_flash_attn_v_direct(ध्रुव ggml_tensor * Q, ध्रुव ggml_tensor * V, uint32_t kv_direct_align) {
    प्रत्यागम ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

अन्तःस्थ ggml_webgpu_flash_attn_common_pipeline_कुंजी ggml_webgpu_flash_attn_make_common_pipeline_कुंजी(
    ध्रुव ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    तार्किक                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_कुंजी कुंजी = {};
    कुंजी.q_type                                     = context.src0->type;
    कुंजी.k_type                                     = context.src1->type;
    कुंजी.v_type                                     = context.src2->type;
    कुंजी.dst_type                                   = context.dst->type;
    कुंजी.head_dim_qk                                = (uint32_t) context.src0->ne[0];
    कुंजी.head_dim_v                                 = (uint32_t) context.src2->ne[0];
    कुंजी.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    कुंजी.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    कुंजी.kv_overlap         = kv_overlap;
    कुंजी.has_mask           = context.src3 != nullptr;
    कुंजी.has_sinks          = context.src4 != nullptr;
    कुंजी.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    प्रत्यागम कुंजी;
}

अन्तःस्थ std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    ध्रुव ggml_webgpu_flash_attn_common_pipeline_कुंजी & कुंजी,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_size) {
    std::vector<std::string> defines;

    विकल्प (कुंजी.k_type) {
        पक्ष GGML_TYPE_F32:
            defines.push_back("K_F32");
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            defines.push_back("K_F16");
            भङ्ग;
        पक्ष GGML_TYPE_Q4_0:
            defines.push_back("K_Q4_0");
            भङ्ग;
        पक्ष GGML_TYPE_Q8_0:
            defines.push_back("K_Q8_0");
            भङ्ग;
        स्वभाव:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(कुंजी.k_type);

    विकल्प (कुंजी.v_type) {
        पक्ष GGML_TYPE_F32:
            defines.push_back("V_F32");
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            defines.push_back("V_F16");
            भङ्ग;
        पक्ष GGML_TYPE_Q4_0:
            defines.push_back("V_Q4_0");
            भङ्ग;
        पक्ष GGML_TYPE_Q8_0:
            defines.push_back("V_Q8_0");
            भङ्ग;
        स्वभाव:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(कुंजी.v_type);

    विकल्प (कुंजी.q_type) {
        पक्ष GGML_TYPE_F32:
            defines.push_back("Q_F32");
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            defines.push_back("Q_F16");
            भङ्ग;
        स्वभाव:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(कुंजी.q_type);

    विकल्प (कुंजी.dst_type) {
        पक्ष GGML_TYPE_F32:
            defines.push_back("DST_F32");
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            defines.push_back("DST_F16");
            भङ्ग;
        स्वभाव:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(कुंजी.dst_type);

    यदि (कुंजी.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    यदि (कुंजी.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    यदि (कुंजी.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    यदि (कुंजी.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    यदि (कुंजी.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    यदि (कुंजी.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(कुंजी.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(कुंजी.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(कुंजी.head_dim_v));
    variant += std::string("_hsv") + std::to_string(कुंजी.head_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    यदि (ggml_is_quantized(कुंजी.k_type) || ggml_is_quantized(कुंजी.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        यदि (ggml_is_quantized(कुंजी.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        यदि (ggml_is_quantized(कुंजी.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    प्रत्यागम defines;
}

संरचना ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी {
    uint32_t  head_dim_v;
    uint32_t  wg_size;
    ggml_type dst_type;
};

संरचना ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.head_dim_v);
        ggml_webgpu_hash_combine(seed, कुंजी.wg_size);
        ggml_webgpu_hash_combine(seed, कुंजी.dst_type);
        प्रत्यागम seed;
    }
};

अन्तःस्थ तार्किक चालक==(ध्रुव ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी & lhs,
                       ध्रुव ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी & rhs) {
    प्रत्यागम lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size && lhs.dst_type == rhs.dst_type;
}

संरचना ggml_webgpu_flash_attn_blk_pipeline_कुंजी {
    uint32_t kv_tile;

    तार्किक चालक==(ध्रुव ggml_webgpu_flash_attn_blk_pipeline_कुंजी & other) ध्रुव { प्रत्यागम kv_tile == other.kv_tile; }
};

संरचना ggml_webgpu_flash_attn_blk_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_flash_attn_blk_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.kv_tile);
        प्रत्यागम seed;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
अन्तःस्थ माप_प्रकार ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  तार्किक     has_mask,
                                                  तार्किक     kv_direct) {
    ध्रुव uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    माप_प्रकार         f16_elems    = 0;
    माप_प्रकार         f32_elems    = 0;

    f32_elems += q_tile * head_dim_qk;        // q_shmem
    यदि (!kv_direct) {
        f32_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f32_elems += q_tile * head_dim_v;         // o_shmem
    यदि (has_mask) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    प्रत्यागम f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

अन्तःस्थ uint32_t ggml_webgpu_flash_attn_max_kv_tile(माप_प्रकार   limit_bytes,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t head_dim_qk,
                                                   uint32_t head_dim_v,
                                                   तार्किक     has_mask,
                                                   तार्किक     kv_direct) {
    ध्रुव माप_प्रकार base_q_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 0, head_dim_qk, head_dim_v, has_mask, kv_direct);
    यदि (limit_bytes <= base_q_bytes) {
        प्रत्यागम 0;
    }
    ध्रुव माप_प्रकार one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, head_dim_qk, head_dim_v, has_mask, kv_direct);
    ध्रुव माप_प्रकार bytes_per_kv = one_kv_bytes - base_q_bytes;
    यदि (bytes_per_kv == 0) {
        प्रत्यागम 0;
    }
    ध्रुव माप_प्रकार max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    प्रत्यागम (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

अन्तःस्थ uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(माप_प्रकार   wg_mem_limit_bytes,
                                                       uint32_t head_dim_qk,
                                                       uint32_t head_dim_v,
                                                       तार्किक     has_mask,
                                                       तार्किक     kv_direct) {
    ध्रुव uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_limit_bytes, 1u, 1u, head_dim_qk, head_dim_v, has_mask, kv_direct);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    यदि (kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        यावत् (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    प्रत्यागम kv_tile;
}

अन्तःस्थ तार्किक ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(तार्किक                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ध्रुव ggml_tensor * Q,
                                                                ध्रुव ggml_tensor * V) {
    प्रत्यागम supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

संरचना ggml_webgpu_mul_mat_vec_pipeline_कुंजी {
    ggml_type src0_type;
    ggml_type src1_type;
    पूर्णाङ्क       vectorized;
    uint32_t  अंक_cols;
    तार्किक      use_mmvq;
    तार्किक      src_overlap;

    तार्किक चालक==(ध्रुव ggml_webgpu_mul_mat_vec_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               अंक_cols == other.अंक_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_mul_mat_vec_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_mul_mat_vec_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src0_type);
        ggml_webgpu_hash_combine(seed, कुंजी.src1_type);
        ggml_webgpu_hash_combine(seed, कुंजी.vectorized);
        ggml_webgpu_hash_combine(seed, कुंजी.अंक_cols);
        ggml_webgpu_hash_combine(seed, कुंजी.use_mmvq);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        प्रत्यागम seed;
    }
};

संरचना ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
    तार्किक     src_overlap = असत्य;
};

संरचना ggml_webgpu_quantize_q8_pipeline_कुंजी {
    ggml_type src0_type;

    तार्किक चालक==(ध्रुव ggml_webgpu_quantize_q8_pipeline_कुंजी & other) ध्रुव { प्रत्यागम src0_type == other.src0_type; }
};

संरचना ggml_webgpu_quantize_q8_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_quantize_q8_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src0_type);
        प्रत्यागम seed;
    }
};

संरचना ggml_webgpu_mul_mat_pipeline_कुंजी {
    ggml_type src0_type;
    ggml_type src1_type;
    पूर्णाङ्क       vectorized;
    पूर्णाङ्क       use_subgroup_matrix;
    तार्किक      src_overlap;

    तार्किक चालक==(ध्रुव ggml_webgpu_mul_mat_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_mul_mat_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_mul_mat_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src0_type);
        ggml_webgpu_hash_combine(seed, कुंजी.src1_type);
        ggml_webgpu_hash_combine(seed, कुंजी.vectorized);
        ggml_webgpu_hash_combine(seed, कुंजी.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        प्रत्यागम seed;
    }
};

संरचना ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    पूर्णाङ्क      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_size;
    तार्किक     src_overlap = असत्य;
};

/** MUL_MAT_ID **/

संरचना ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी {
    uint32_t n_experts;

    तार्किक चालक==(ध्रुव ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम n_experts == other.n_experts;
    }
};

संरचना ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.n_experts);
        प्रत्यागम seed;
    }
};

संरचना ggml_webgpu_mul_mat_id_pipeline_कुंजी {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  अंक_cols;
    पूर्णाङ्क       vectorized;

    तार्किक चालक==(ध्रुव ggml_webgpu_mul_mat_id_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               अंक_cols == other.अंक_cols && vectorized == other.vectorized;
    }
};

संरचना ggml_webgpu_mul_mat_id_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_mul_mat_id_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src0_type);
        ggml_webgpu_hash_combine(seed, कुंजी.src1_type);
        ggml_webgpu_hash_combine(seed, कुंजी.n_experts);
        ggml_webgpu_hash_combine(seed, कुंजी.अंक_cols);
        ggml_webgpu_hash_combine(seed, कुंजी.vectorized);
        प्रत्यागम seed;
    }
};

/** Cpy **/

संरचना ggml_webgpu_cpy_pipeline_कुंजी {
    ggml_type src_type;
    ggml_type dst_type;

    तार्किक चालक==(ध्रुव ggml_webgpu_cpy_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम src_type == other.src_type && dst_type == other.dst_type;
    }
};

संरचना ggml_webgpu_cpy_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_cpy_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src_type);
        ggml_webgpu_hash_combine(seed, कुंजी.dst_type);
        प्रत्यागम seed;
    }
};

/** Glu **/

संरचना ggml_webgpu_glu_pipeline_कुंजी {
    ggml_glu_op glu_op;
    ggml_type   type;
    तार्किक        split;
    तार्किक        src_overlap;

    तार्किक चालक==(ध्रुव ggml_webgpu_glu_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_glu_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_glu_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.glu_op);
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.split);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        प्रत्यागम seed;
    }
};

/** Rope **/

संरचना ggml_webgpu_rope_pipeline_कुंजी {
    ggml_type type;
    तार्किक      inplace;
    तार्किक      has_ff;

    तार्किक चालक==(ध्रुव ggml_webgpu_rope_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

संरचना ggml_webgpu_rope_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_rope_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        ggml_webgpu_hash_combine(seed, कुंजी.has_ff);
        प्रत्यागम seed;
    }
};

/** SoftMax **/

संरचना ggml_webgpu_soft_max_pipeline_कुंजी {
    ggml_type mask_type;
    तार्किक      has_mask;
    तार्किक      has_sink;
    तार्किक      inplace;

    तार्किक चालक==(ध्रुव ggml_webgpu_soft_max_pipeline_कुंजी & other) ध्रुव {
        प्रत्यागम mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

संरचना ggml_webgpu_soft_max_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव ggml_webgpu_soft_max_pipeline_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.mask_type);
        ggml_webgpu_hash_combine(seed, कुंजी.has_mask);
        ggml_webgpu_hash_combine(seed, कुंजी.has_sink);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        प्रत्यागम seed;
    }
};

/** MMVQ **/

अन्तःस्थ तार्किक ggml_webgpu_can_use_mmvq(ध्रुव ggml_tensor * src0,
                                     ध्रुव ggml_tensor * src1,
                                     तार्किक                supports_dot_product,
                                     ध्रुव std::string & vendor) {
    यदि (src1->ne[1] <= 4) {
        तार्किक supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        यदि (supports_dp4a && supports_dot_product) {
            विकल्प (src1->type) {
                पक्ष GGML_TYPE_F32:
                    विकल्प (src0->type) {
                        पक्ष GGML_TYPE_Q1_0:
                        पक्ष GGML_TYPE_Q4_0:
                        पक्ष GGML_TYPE_Q4_1:
                        पक्ष GGML_TYPE_Q5_0:
                        पक्ष GGML_TYPE_Q5_1:
                        पक्ष GGML_TYPE_Q8_0:
                        पक्ष GGML_TYPE_MXFP4:
                        पक्ष GGML_TYPE_Q2_K:
                        पक्ष GGML_TYPE_Q3_K:
                        पक्ष GGML_TYPE_Q4_K:
                        पक्ष GGML_TYPE_Q5_K:
                        पक्ष GGML_TYPE_Q6_K:
                            प्रत्यागम src0->ne[0] % 4 == 0;
                        स्वभाव:
                            भङ्ग;
                    }
                    भङ्ग;
                स्वभाव:
                    भङ्ग;
            }
        }
    }
    प्रत्यागम असत्य;
}

वर्ग ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<पूर्णाङ्क, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<पूर्णाङ्क, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<पूर्णाङ्क, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<पूर्णाङ्क, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<पूर्णाङ्क, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_कुंजी_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_कुंजी_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_unary_pipeline_कुंजी_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_scale_pipeline_कुंजी_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_कुंजी_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_कुंजी_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_कुंजी_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_कुंजी,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_कुंजी_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_pad_pipeline_कुंजी_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_द्विआधारी_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_द्विआधारी_pipeline_कुंजी_hash>
        द्विआधारी_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_add_id_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_add_id_pipeline_कुंजी_hash>
        add_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_concat_pipeline_कुंजी_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_repeat_pipeline_कुंजी_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_कुंजी,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_कुंजी_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_कुंजी_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_कुंजी,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_कुंजी_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_कुंजी_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_कुंजी_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_कुंजी_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_कुंजी_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_कुंजी_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_कुंजी_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_set_pipeline_कुंजी_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_cpy_pipeline_कुंजी_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_glu_pipeline_कुंजी_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_rope_pipeline_कुंजी_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_कुंजी_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_कुंजी_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_कुंजी_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_im2col_pipeline_कुंजी_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_कुंजी,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_कुंजी_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_कुंजी, webgpu_pipeline, ggml_webgpu_upscale_pipeline_कुंजी_hash>
        upscale_pipelines;

  सार्वजनिक:
    ggml_webgpu_shader_lib(wgpu::Device device) { एतत्->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        स्वतः it = sum_rows_pipelines.find(1);
        यदि (it != sum_rows_pipelines.end()) {
            प्रत्यागम it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        प्रत्यागम sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_कुंजी कुंजी = {};
        कुंजी.op                                = context.dst->op;
        कुंजी.src_type                          = context.src0->type;
        कुंजी.dst_type                          = context.dst->type;
        कुंजी.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = row_norm_pipelines.find(कुंजी);
        यदि (it != row_norm_pipelines.end()) {
            प्रत्यागम it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        विकल्प (कुंजी.op) {
            पक्ष GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                भङ्ग;
            पक्ष GGML_OP_NORM:
                defines.push_back("NORM");
                variant = "norm";
                भङ्ग;
            पक्ष GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        यदि (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        यदि (कुंजी.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } अन्यथा यदि (कुंजी.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        यदि (कुंजी.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } अन्यथा यदि (कुंजी.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ध्रुव uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        स्वतः processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        स्वतः decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = कुंजी.inplace;
        row_norm_pipelines[कुंजी]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[कुंजी].context = decisions;
        प्रत्यागम row_norm_pipelines[कुंजी];
    }

    webgpu_pipeline get_argmax_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        तार्किक vec4 = context.src0->ne[0] % 4 == 0;

        स्वतः it = argmax_pipelines.find(vec4);
        यदि (it != argmax_pipelines.end()) {
            प्रत्यागम it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        यदि (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        स्वतः processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        प्रत्यागम argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ध्रुव तार्किक                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_कुंजी कुंजी       = {};
        कुंजी.src0_type                               = context.src0->type;
        कुंजी.dst_type                                = context.dst->type;
        कुंजी.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        कुंजी.i64_idx     = context.src1->type == GGML_TYPE_I64;
        कुंजी.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_size(context.dst->type)) % 2 == 0);

        स्वतः it = set_rows_pipelines.find(कुंजी);
        यदि (it != set_rows_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        विकल्प (context.dst->type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                भङ्ग;
            पक्ष GGML_TYPE_Q8_0:
                defines.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                भङ्ग;
            पक्ष GGML_TYPE_Q4_0:
                defines.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        यदि (context.src0->type == GGML_TYPE_F16) {
            defines.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        यदि (कुंजी.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        यदि (कुंजी.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        यदि (कुंजी.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        ध्रुव स्वतः & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        स्वतः         processed          = preprocessor.preprocess(shader_source, defines);
        स्वतः         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = कुंजी.vec4;
        decisions->i64_idx              = कुंजी.i64_idx;
        decisions->pair_blocks          = कुंजी.pair_blocks;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[कुंजी]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[कुंजी].context = decisions;
        प्रत्यागम set_rows_pipelines[कुंजी];
    }

    webgpu_pipeline get_set_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_कुंजी कुंजी = {};
        कुंजी.type                         = context.dst->type;
        कुंजी.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = set_pipelines.find(कुंजी);
        यदि (it != set_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for set shader");
        }

        यदि (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_set, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = कुंजी.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[कुंजी]       = pipeline;
        प्रत्यागम set_pipelines[कुंजी];
    }

    webgpu_pipeline get_cumsum_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        स्वतः it = cumsum_pipelines.find(1);
        यदि (it != cumsum_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        प्रत्यागम cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        तार्किक          is_शीर्ष_k = context.dst->op == GGML_OP_शीर्ष_K;
        // ascending order is 0, descending order is 1
        ध्रुव int32_t order =
            is_शीर्ष_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        स्वतः it = argsort_pipelines.find(order);
        यदि (it != argsort_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        यावत् (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        स्वतः processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        स्वतः decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        प्रत्यागम argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        तार्किक          is_शीर्ष_k = context.dst->op == GGML_OP_शीर्ष_K;
        // ascending order is 0, descending order is 1
        ध्रुव int32_t order =
            is_शीर्ष_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        स्वतः it = argsort_merge_pipelines.find(order);
        यदि (it != argsort_merge_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        स्वतः processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        प्रत्यागम argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक vec4_aligned) {
        ध्रुव तार्किक vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_कुंजी कुंजी = {};
        कुंजी.src_type                          = context.src0->type;
        कुंजी.vectorized                        = (पूर्णाङ्क) vectorized;

        स्वतः it = get_rows_pipelines.find(कुंजी);
        यदि (it != get_rows_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        ध्रुव संरचना ggml_type_traits * type_traits = ggml_get_type_traits(कुंजी.src_type);
        ध्रुव वर्ण *                    type_str    = type_traits->type_name;

        विकल्प (कुंजी.src_type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                यदि (कुंजी.vectorized) {
                    defines.push_back("F32_VEC");
                    defines.push_back("SRC_TYPE=vec4<f32>");
                    defines.push_back("DST_TYPE=vec4<f32>");
                    defines.push_back("BLOCK_SIZE=4u");
                } अन्यथा {
                    defines.push_back("F32");
                    defines.push_back("SRC_TYPE=f32");
                    defines.push_back("DST_TYPE=f32");
                    defines.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                भङ्ग;
            पक्ष GGML_TYPE_BF16:
                defines.push_back("BF16");
                defines.push_back("SRC_TYPE=u32");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                भङ्ग;
            स्वभाव:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    विकल्प (कुंजी.src_type) {
                        पक्ष GGML_TYPE_Q1_0:
                        पक्ष GGML_TYPE_Q2_0:
                        पक्ष GGML_TYPE_Q4_0:
                        पक्ष GGML_TYPE_Q5_0:
                        पक्ष GGML_TYPE_Q8_0:
                        पक्ष GGML_TYPE_Q3_K:
                        पक्ष GGML_TYPE_Q6_K:
                        पक्ष GGML_TYPE_IQ2_XXS:
                        पक्ष GGML_TYPE_IQ2_XS:
                        पक्ष GGML_TYPE_IQ2_S:
                        पक्ष GGML_TYPE_IQ3_XXS:
                        पक्ष GGML_TYPE_IQ3_S:
                        पक्ष GGML_TYPE_IQ1_S:
                        पक्ष GGML_TYPE_IQ4_NL:
                        पक्ष GGML_TYPE_MXFP4:
                        पक्ष GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                भङ्ग;
                            }
                        स्वभाव:
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

                    यदि (कुंजी.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } अन्यथा यदि ((कुंजी.src_type >= GGML_TYPE_Q4_0 && कुंजी.src_type <= GGML_TYPE_Q8_1) ||
                               कुंजी.src_type == GGML_TYPE_IQ4_NL || कुंजी.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } अन्यथा यदि (कुंजी.src_type == GGML_TYPE_Q2_0 || कुंजी.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } अन्यथा यदि (कुंजी.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } अन्यथा {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    भङ्ग;
                }
        }

        यदि (कुंजी.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[कुंजी]  = pipeline;
        प्रत्यागम get_rows_pipelines[कुंजी];
    }

    webgpu_pipeline get_scale_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_कुंजी कुंजी = {};
        कुंजी.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = scale_pipelines.find(कुंजी);
        यदि (it != scale_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        यदि (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_scale, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = कुंजी.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[कुंजी]     = pipeline;
        प्रत्यागम scale_pipelines[कुंजी];
    }

    webgpu_pipeline get_solve_tri_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_कुंजी कुंजी = {};
        कुंजी.type                               = context.dst->type;
        कुंजी.n                                  = (पूर्णाङ्क) context.src0->ne[0];
        कुंजी.k                                  = (पूर्णाङ्क) context.src1->ne[0];

        स्वतः it = solve_tri_pipelines.find(कुंजी);
        यदि (it != solve_tri_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                variant += "_f32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ध्रुव uint32_t wg_size       = std::min((uint32_t) कुंजी.n, context.max_wg_size);
        ध्रुव uint32_t k_tile        = wg_size;
        ध्रुव uint32_t bytes_per_row = ((uint32_t) कुंजी.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        ध्रुव uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(कुंजी.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        स्वतः processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[कुंजी] = pipeline;
        प्रत्यागम solve_tri_pipelines[कुंजी];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_कुंजी कुंजी = {};
        कुंजी.type                              = context.dst->type;
        कुंजी.vectorized                        = context.src1->ne[0] == 4;

        स्वतः it = ssm_conv_pipelines.find(कुंजी);
        यदि (it != ssm_conv_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                variant += "_f32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        यदि (कुंजी.vectorized) {
            defines.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t block_size    = 32u;
        constexpr uint32_t tokens_per_wg = 8u;

        defines.push_back("BLOCK_SIZE=" + std::to_string(block_size) + "u");
        defines.push_back("TOKENS_PER_WG=" + std::to_string(tokens_per_wg) + "u");

        स्वतः processed           = preprocessor.preprocess(wgsl_ssm_conv, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->block_size    = block_size;
        decisions->tokens_per_wg = tokens_per_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[कुंजी]  = pipeline;
        प्रत्यागम ssm_conv_pipelines[कुंजी];
    }

    webgpu_pipeline get_ssm_scan_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_कुंजी कुंजी = {};
        कुंजी.type                              = context.dst->type;
        कुंजी.d_state                           = (पूर्णाङ्क) context.src0->ne[0];
        कुंजी.xbc_overlap                       = xbc_overlap;

        स्वतः it = ssm_scan_pipelines.find(कुंजी);
        यदि (it != ssm_scan_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                variant += "_f32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ध्रुव uint32_t wg_size = (uint32_t) कुंजी.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        यदि (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } अन्यथा {
            variant += "_wg_reduce";
        }

        यदि (कुंजी.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } अन्यथा यदि (कुंजी.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } अन्यथा यदि (कुंजी.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(कुंजी.d_state);

        स्वतः processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        स्वतः decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = कुंजी.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[कुंजी]    = pipeline;
        प्रत्यागम ssm_scan_pipelines[कुंजी];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_कुंजी कुंजी = {};
        कुंजी.type                                     = context.dst->type;
        कुंजी.s_v                                      = (पूर्णाङ्क) context.src2->ne[0];
        कुंजी.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        कुंजी.fused_cache                              = context.dst_fuse != nullptr;

        स्वतः it = gated_delta_net_pipelines.find(कुंजी);
        यदि (it != gated_delta_net_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                variant += "_f32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        यदि (कुंजी.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        यदि (कुंजी.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(कुंजी.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(कुंजी.s_v) + "u");

        स्वतः            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[कुंजी] = pipeline;
        प्रत्यागम gated_delta_net_pipelines[कुंजी];
    }

    webgpu_pipeline get_pad_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_कुंजी कुंजी = {};
        कुंजी.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        स्वतः it = pad_pipelines.find(कुंजी);
        यदि (it != pad_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        यदि (कुंजी.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_pad, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[कुंजी]       = pipeline;
        प्रत्यागम pad_pipelines[कुंजी];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_कुंजी कुंजी = {};
        कुंजी.src0_type                            = context.src0->type;

        स्वतः it = quantize_q8_pipelines.find(कुंजी);
        यदि (it != quantize_q8_pipelines.end()) {
            प्रत्यागम it->second;
        }
        ध्रुव वर्ण *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> defines;
        std::string              variant = "quantize_q8";

        uint32_t wg_size = WEBGPU_MUL_MAT_VEC_WG_SIZE;

        defines.push_back("SRC1_INNER_TYPE=f32");
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        ध्रुव संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_upper  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

        defines.push_back("MUL_ACC_" + type_upper);
        defines.push_back("Q8_1_T");

        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        स्वतः processed             = preprocessor.preprocess(shader_src, defines);
        स्वतः decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size         = wg_size;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[कुंजी] = pipeline;
        प्रत्यागम quantize_q8_pipelines[कुंजी];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_कुंजी कुंजी = {};
        कुंजी.src0_type                            = context.src0->type;
        कुंजी.src1_type                            = context.src1->type;
        कुंजी.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        कुंजी.अंक_cols   = context.dst->ne[1];
        कुंजी.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
        कुंजी.src_overlap = src_overlap;

        स्वतः it = mul_mat_vec_pipelines.find(कुंजी);
        यदि (it != mul_mat_vec_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        ध्रुव वर्ण *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        विकल्प (context.src0->type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                भङ्ग;
            स्वभाव:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ध्रुव संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    विकल्प (context.src0->type) {
                        पक्ष GGML_TYPE_Q4_0:
                        पक्ष GGML_TYPE_Q4_1:
                        पक्ष GGML_TYPE_Q5_0:
                        पक्ष GGML_TYPE_Q5_1:
                        पक्ष GGML_TYPE_Q8_0:
                            यदि (कुंजी.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            भङ्ग;
                        पक्ष GGML_TYPE_Q1_0:
                        पक्ष GGML_TYPE_Q2_K:
                        पक्ष GGML_TYPE_Q3_K:
                        पक्ष GGML_TYPE_Q4_K:
                        पक्ष GGML_TYPE_Q5_K:
                        पक्ष GGML_TYPE_Q6_K:
                            यदि (कुंजी.use_mmvq) {
                                defines.push_back("K_QUANTS_HANDLING");
                            }
                            भङ्ग;
                        पक्ष GGML_TYPE_IQ1_S:
                        पक्ष GGML_TYPE_IQ1_M:
                        पक्ष GGML_TYPE_IQ2_S:
                        पक्ष GGML_TYPE_IQ3_S:
                        पक्ष GGML_TYPE_IQ4_NL:
                        पक्ष GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            भङ्ग;
                        पक्ष GGML_TYPE_IQ2_XXS:
                        पक्ष GGML_TYPE_IQ2_XS:
                        पक्ष GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            भङ्ग;
                        पक्ष GGML_TYPE_MXFP4:
                            defines.push_back(type_upper + "_LUT");
                            यदि (कुंजी.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            भङ्ग;
                        पक्ष GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            भङ्ग;
                        स्वभाव:
                            भङ्ग;
                    }
                    भङ्ग;
                }
        }

        // src1 type (vector)
        विकल्प (context.src1->type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(कुंजी.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        यदि (कुंजी.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } अन्यथा यदि (कुंजी.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } अन्यथा यदि (कुंजी.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        यदि (कुंजी.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        यदि (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        यदि (कुंजी.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(कुंजी.अंक_cols));

        स्वतः processed            = preprocessor.preprocess(shader_src, defines);
        स्वतः decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = कुंजी.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[कुंजी] = pipeline;
        प्रत्यागम mul_mat_vec_pipelines[कुंजी];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_mul_mat_pipeline_कुंजी कुंजी = {};
        कुंजी.src0_type                        = context.src0->type;
        कुंजी.src1_type                        = context.src1->type;
        कुंजी.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        कुंजी.use_subgroup_matrix = context.supports_subgroup_matrix;
        कुंजी.src_overlap         = src_overlap;

        स्वतः it = mul_mat_fast_pipelines.find(कुंजी);
        यदि (it != mul_mat_fast_pipelines.end()) {
            प्रत्यागम it->second;
        }

        ध्रुव वर्ण * shader_src = कुंजी.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = कुंजी.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        विकल्प (context.src1->type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ध्रुव संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ध्रुव वर्ण *                    src0_name   = src0_traits->type_name;

        विकल्प (context.src0->type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("TYPE_F32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("TYPE_F16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("TYPE_BF16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                भङ्ग;
            स्वभाव:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    विकल्प (context.src0->type) {
                        पक्ष GGML_TYPE_IQ1_S:
                        पक्ष GGML_TYPE_IQ1_M:
                        पक्ष GGML_TYPE_IQ4_NL:
                        पक्ष GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            भङ्ग;
                        पक्ष GGML_TYPE_IQ2_XXS:
                        पक्ष GGML_TYPE_IQ2_XS:
                        पक्ष GGML_TYPE_IQ2_S:
                        पक्ष GGML_TYPE_IQ3_XXS:
                        पक्ष GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            भङ्ग;
                        पक्ष GGML_TYPE_MXFP4:
                        पक्ष GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            भङ्ग;
                        स्वभाव:
                            भङ्ग;
                    }

                    variant += std::string("_") + src0_name;
                    भङ्ग;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(कुंजी.vectorized ? "VEC" : "SCALAR");

        ध्रुव तार्किक is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        यदि (कुंजी.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } अन्यथा {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        यदि (कुंजी.use_subgroup_matrix) {
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
        यदि (कुंजी.vectorized) {
            variant += "_vectorized";
        }

        यदि (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        यदि (!कुंजी.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        स्वतः processed = preprocessor.preprocess(shader_src, defines);

        स्वतः decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = कुंजी.use_subgroup_matrix;
        यदि (कुंजी.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_size           = context.max_subgroup_size;
        } अन्यथा {
            decisions->wg_size_m       = WEBGPU_MUL_MAT_WG_SIZE_M;
            decisions->wg_size_n       = WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->wg_size         = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->mul_mat_wg_size = WEBGPU_MUL_MAT_WG_SIZE;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[कुंजी] = pipeline;
        प्रत्यागम mul_mat_fast_pipelines[कुंजी];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी कुंजी = {};
        कुंजी.n_experts                                  = (uint32_t) context.src0->ne[2];

        स्वतः it = mul_mat_id_gather_pipelines.find(कुंजी);
        यदि (it != mul_mat_id_gather_pipelines.end()) {
            प्रत्यागम it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(कुंजी.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        स्वतः processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        स्वतः decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[कुंजी] = pipeline;
        प्रत्यागम pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_कुंजी कुंजी = {};
        कुंजी.src0_type                           = context.src0->type;
        कुंजी.src1_type                           = context.src1->type;
        कुंजी.n_experts                           = context.src0->ne[2];
        कुंजी.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        स्वतः it = mul_mat_id_pipelines.find(कुंजी);
        यदि (it != mul_mat_id_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        विकल्प (context.src1->type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ध्रुव संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ध्रुव वर्ण *                    src0_name   = src0_traits->type_name;

        विकल्प (context.src0->type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                भङ्ग;
            स्वभाव:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    विकल्प (context.src0->type) {
                        पक्ष GGML_TYPE_IQ1_S:
                        पक्ष GGML_TYPE_IQ1_M:
                        पक्ष GGML_TYPE_IQ4_NL:
                        पक्ष GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            भङ्ग;
                        पक्ष GGML_TYPE_IQ2_XXS:
                        पक्ष GGML_TYPE_IQ2_XS:
                        पक्ष GGML_TYPE_IQ2_S:
                        पक्ष GGML_TYPE_IQ3_XXS:
                        पक्ष GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            भङ्ग;
                        पक्ष GGML_TYPE_MXFP4:
                        पक्ष GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            भङ्ग;
                        स्वभाव:
                            भङ्ग;
                    }

                    variant += std::string("_") + src0_name;
                    भङ्ग;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(कुंजी.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ध्रुव uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        यदि (कुंजी.vectorized) {
            variant += "_vectorized";
        }

        स्वतः processed = preprocessor.preprocess(wgsl_mul_mat_id, defines);

        स्वतः decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_size_m = WEBGPU_MUL_MAT_WG_SIZE_M;
        decisions->wg_size_n = WEBGPU_MUL_MAT_WG_SIZE_N;
        decisions->wg_size   = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[कुंजी] = pipeline;
        प्रत्यागम mul_mat_id_pipelines[कुंजी];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_कुंजी कुंजी = {};
        कुंजी.src0_type                           = context.src0->type;
        कुंजी.src1_type                           = context.src1->type;
        कुंजी.n_experts                           = context.src0->ne[2];
        कुंजी.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        स्वतः it = mul_mat_id_vec_pipelines.find(कुंजी);
        यदि (it != mul_mat_id_vec_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        ध्रुव वर्ण *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        विकल्प (context.src1->type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        विकल्प (context.src0->type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                भङ्ग;
            स्वभाव:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ध्रुव संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    विकल्प (context.src0->type) {
                        पक्ष GGML_TYPE_IQ1_S:
                        पक्ष GGML_TYPE_IQ1_M:
                        पक्ष GGML_TYPE_IQ2_S:
                        पक्ष GGML_TYPE_IQ3_S:
                        पक्ष GGML_TYPE_IQ4_NL:
                        पक्ष GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            भङ्ग;
                        पक्ष GGML_TYPE_IQ2_XXS:
                        पक्ष GGML_TYPE_IQ2_XS:
                        पक्ष GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            भङ्ग;
                        पक्ष GGML_TYPE_MXFP4:
                        पक्ष GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            भङ्ग;
                        स्वभाव:
                            भङ्ग;
                    }
                    भङ्ग;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(कुंजी.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        यदि (कुंजी.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } अन्यथा यदि (कुंजी.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } अन्यथा यदि (कुंजी.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        यदि (कुंजी.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(कुंजी.n_experts));

        स्वतः processed = preprocessor.preprocess(shader_src, defines);

        स्वतः decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[कुंजी] = pipeline;
        प्रत्यागम mul_mat_id_vec_pipelines[कुंजी];
    }

    webgpu_pipeline get_unary_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ध्रुव तार्किक                     is_unary = context.dst->op == GGML_OP_UNARY;
        ध्रुव पूर्णाङ्क                      op       = is_unary ? (पूर्णाङ्क) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_कुंजी कुंजी      = {};
        कुंजी.type                                = context.dst->type;
        कुंजी.op                                  = op;
        कुंजी.is_unary                            = is_unary;
        कुंजी.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        कुंजी.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        स्वतः it = unary_pipelines.find(कुंजी);
        यदि (it != unary_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            कुंजी.is_unary ? ggml_unary_op_name((ggml_unary_op) कुंजी.op) : ggml_op_name((ggml_op) कुंजी.op);
        defines.push_back(variant);

        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for unary shader");
        }

        यदि (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        यदि (op == GGML_OP_TRI) {
            विकल्प (कुंजी.ttype) {
                पक्ष GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    भङ्ग;
                पक्ष GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    भङ्ग;
                पक्ष GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    भङ्ग;
                पक्ष GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    भङ्ग;
                स्वभाव:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_unary, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = कुंजी.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[कुंजी]     = pipeline;
        प्रत्यागम unary_pipelines[कुंजी];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_कुंजी कुंजी = {};
        कुंजी.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        कुंजी.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        कुंजी.src_overlap                           = src_overlap;

        स्वतः it = rms_norm_mul_pipelines.find(कुंजी);
        यदि (it != rms_norm_mul_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        यदि (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } अन्यथा यदि (कुंजी.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } अन्यथा यदि (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        स्वतः pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = कुंजी.inplace;
        pipeline_decisions->overlap     = कुंजी.overlap;
        pipeline_decisions->src_overlap = कुंजी.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[कुंजी]     = pipeline;
        प्रत्यागम rms_norm_mul_pipelines[कुंजी];
    }

    webgpu_pipeline get_द्विआधारी_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_द्विआधारी_pipeline_कुंजी कुंजी = {};
        कुंजी.type                            = context.dst->type;
        कुंजी.op                              = context.dst->op;
        कुंजी.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        कुंजी.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        कुंजी.src_overlap                     = src_overlap;

        स्वतः it = द्विआधारी_pipelines.find(कुंजी);
        यदि (it != द्विआधारी_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) कुंजी.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for binary shader");
        }

        यदि (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } अन्यथा यदि (कुंजी.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } अन्यथा यदि (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed                  = preprocessor.preprocess(wgsl_द्विआधारी, defines);
        स्वतः pipeline_decisions         = std::make_shared<ggml_webgpu_द्विआधारी_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = कुंजी.inplace;
        pipeline_decisions->overlap     = कुंजी.overlap;
        pipeline_decisions->src_overlap = कुंजी.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        द्विआधारी_pipelines[कुंजी]    = pipeline;
        प्रत्यागम द्विआधारी_pipelines[कुंजी];
    }

    webgpu_pipeline get_add_id_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_कुंजी कुंजी = {};
        कुंजी.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = add_id_pipelines.find(कुंजी);
        यदि (it != add_id_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        ध्रुव वर्ण *             shader_src = wgsl_add_id;

        यदि (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed              = preprocessor.preprocess(shader_src, defines);
        स्वतः pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_size = context.max_wg_size;
        pipeline_decisions->inplace = कुंजी.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[कुंजी]    = pipeline;
        प्रत्यागम pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_concat_pipeline_कुंजी कुंजी = {};
        कुंजी.type                            = context.dst->type;
        कुंजी.src_overlap                     = src_overlap;

        स्वतः it = concat_pipelines.find(कुंजी);
        यदि (it != concat_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for concat shader");
        }

        यदि (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_concat, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_द्विआधारी_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = कुंजी.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[कुंजी]    = pipeline;
        प्रत्यागम concat_pipelines[कुंजी];
    }

    webgpu_pipeline get_repeat_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_कुंजी कुंजी = {};
        कुंजी.type                            = context.dst->type;

        स्वतः it = repeat_pipelines.find(कुंजी);
        यदि (it != repeat_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                भङ्ग;
            पक्ष GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_repeat, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[कुंजी]    = pipeline;
        प्रत्यागम repeat_pipelines[कुंजी];
    }

    webgpu_pipeline get_flash_attn_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक kv_overlap) {
        ध्रुव तार्किक can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_कुंजी कुंजी = {};
        कुंजी.common                              = ggml_webgpu_flash_attn_make_common_pipeline_कुंजी(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        कुंजी.common.k_direct &= decisions.use_sg_matrix && कुंजी.common.k_type == GGML_TYPE_F16;
        कुंजी.common.v_direct &= decisions.use_sg_matrix && कुंजी.common.v_type == GGML_TYPE_F16;
        कुंजी.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_limit_bytes, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            कुंजी.common.head_dim_qk, कुंजी.common.head_dim_v, कुंजी.common.has_mask,
            कुंजी.common.k_direct || कुंजी.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        यदि (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = असत्य;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            कुंजी.common.k_direct     = असत्य;
            कुंजी.common.v_direct     = असत्य;
            कुंजी.use_sg_matrix       = असत्य;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, कुंजी.common.head_dim_qk, कुंजी.common.head_dim_v,
                कुंजी.common.has_mask, कुंजी.common.k_direct || कुंजी.common.v_direct);
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

        यदि (कुंजी.common.k_direct || कुंजी.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            यावत् (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_size;
            }
        }

        स्वतः it = flash_attn_pipelines.find(कुंजी);
        यदि (it != flash_attn_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(कुंजी.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_size);
        ध्रुव वर्ण *             shader_src = nullptr;
        यदि (!कुंजी.use_sg_matrix) {
            shader_src = wgsl_flash_attn_tile;
            defines.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_size) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_size) + "_" +
                       std::to_string(context.max_subgroup_size);
        } अन्यथा {
            shader_src = wgsl_flash_attn;
            defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        स्वतः            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[कुंजी] = pipeline;
        प्रत्यागम flash_attn_pipelines[कुंजी];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_कुंजी कुंजी = {};
        कुंजी.common = ggml_webgpu_flash_attn_make_common_pipeline_कुंजी(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        स्वतः it = flash_attn_vec_pipelines.find(कुंजी);
        यदि (it != flash_attn_vec_pipelines.end()) {
            प्रत्यागम it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, कुंजी.common.head_dim_qk,
                                                                   कुंजी.common.head_dim_v, कुंजी.common.has_mask,
                                                                   कुंजी.common.k_direct || कुंजी.common.v_direct);
        decisions.wg_size = context.max_subgroup_size;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(कुंजी.common, variant, 1u, decisions.kv_tile, decisions.wg_size);
        यदि (कुंजी.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (परिमाण("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_size;
        ध्रुव uint32_t D       = कुंजी.common.head_dim_qk | कुंजी.common.head_dim_v;
        ध्रुव uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_size, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        स्वतः            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[कुंजी] = pipeline;
        प्रत्यागम flash_attn_vec_pipelines[कुंजी];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_कुंजी कुंजी = {};
        कुंजी.kv_tile                                 = kv_tile;
        स्वतः it                                     = flash_attn_blk_pipelines.find(कुंजी);
        यदि (it != flash_attn_blk_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(कुंजी.kv_tile));
        variant += std::string("_kvt") + std::to_string(कुंजी.kv_tile);

        uint32_t wg_size = 1;
        यावत् ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[कुंजी] = pipeline;
        प्रत्यागम flash_attn_blk_pipelines[कुंजी];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी कुंजी = {};
        कुंजी.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        कुंजी.dst_type                                       = context.dst->type;
        कुंजी.wg_size                                        = context.max_wg_size;
        स्वतः it                                            = flash_attn_vec_reduce_pipelines.find(कुंजी);
        यदि (it != flash_attn_vec_reduce_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        विकल्प (कुंजी.dst_type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("DST_F32");
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("DST_F16");
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(कुंजी.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(कुंजी.head_dim_v));
        variant += std::string("_hsv") + std::to_string(कुंजी.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[कुंजी] = pipeline;
        प्रत्यागम flash_attn_vec_reduce_pipelines[कुंजी];
    }

    webgpu_pipeline get_cpy_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_कुंजी कुंजी = {};
        कुंजी.src_type                     = context.src0->type;
        कुंजी.dst_type                     = context.dst->type;

        स्वतः it = cpy_pipelines.find(कुंजी);
        यदि (it != cpy_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        विकल्प (कुंजी.src_type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                defines.push_back("SRC_I32");
                variant += "_i32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        विकल्प (कुंजी.dst_type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_cpy, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[कुंजी]       = pipeline;
        प्रत्यागम cpy_pipelines[कुंजी];
    }

    webgpu_pipeline get_glu_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_glu_pipeline_कुंजी कुंजी = {};
        कुंजी.glu_op                       = ggml_get_glu_op(context.dst);
        कुंजी.type                         = context.dst->type;
        कुंजी.split                        = (context.src1 != nullptr);
        कुंजी.src_overlap                  = src_overlap;

        स्वतः it = glu_pipelines.find(कुंजी);
        यदि (it != glu_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        विकल्प (कुंजी.glu_op) {
            पक्ष GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                भङ्ग;
            पक्ष GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                भङ्ग;
            पक्ष GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                भङ्ग;
            पक्ष GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                भङ्ग;
            पक्ष GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                भङ्ग;
            पक्ष GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                भङ्ग;
            पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
                defines.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported GLU op");
        }
        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        यदि (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } अन्यथा यदि (कुंजी.split) {
            variant += "_split";
        } अन्यथा {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_glu, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = कुंजी.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[कुंजी]       = pipeline;
        प्रत्यागम glu_pipelines[कुंजी];
    }

    webgpu_pipeline get_rope_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_कुंजी कुंजी = {};
        कुंजी.type                          = context.dst->type;
        कुंजी.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        कुंजी.has_ff                        = (context.src2 != nullptr);

        स्वतः it = rope_pipelines.find(कुंजी);
        यदि (it != rope_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        विकल्प (कुंजी.type) {
            पक्ष GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        यदि (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        यदि (कुंजी.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_rope, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = कुंजी.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[कुंजी]      = pipeline;
        प्रत्यागम rope_pipelines[कुंजी];
    }

    webgpu_pipeline get_soft_max_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_कुंजी कुंजी = {};
        कुंजी.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        कुंजी.has_mask                          = (context.src1 != nullptr);
        कुंजी.has_sink                          = (context.src2 != nullptr);
        कुंजी.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = soft_max_pipelines.find(कुंजी);
        यदि (it != soft_max_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        यदि (कुंजी.has_mask) {
            defines.push_back("HAS_MASK");
            विकल्प (कुंजी.mask_type) {
                पक्ष GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    भङ्ग;
                पक्ष GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    भङ्ग;
                स्वभाव:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        यदि (कुंजी.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        यदि (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = कुंजी.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[कुंजी]  = pipeline;
        प्रत्यागम soft_max_pipelines[कुंजी];
    }

    webgpu_pipeline get_conv2d_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_कुंजी कुंजी = {};
        कुंजी.weight_type                     = context.src0->type;
        कुंजी.निवेश_type                      = context.src1->type;
        कुंजी.output_type                     = context.dst->type;

        स्वतः it = conv2d_pipelines.find(कुंजी);
        यदि (it != conv2d_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        स्वतः धकेलो_type_defines = [&](ध्रुव वर्ण * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            यदि (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } अन्यथा यदि (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        धकेलो_type_defines("WEIGHT_TYPE", कुंजी.weight_type);
        धकेलो_type_defines("INPUT_TYPE", कुंजी.निवेश_type);
        धकेलो_type_defines("OUTPUT_TYPE", कुंजी.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[कुंजी]    = pipeline;
        प्रत्यागम conv2d_pipelines[कुंजी];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक whcn) {
        ggml_webgpu_conv2d_dw_pipeline_कुंजी कुंजी = {};
        कुंजी.weight_type                        = context.src0->type;
        कुंजी.निवेश_type                         = context.src1->type;
        कुंजी.output_type                        = context.dst->type;
        कुंजी.whcn                               = whcn;

        स्वतः it = conv2d_dw_pipelines.find(कुंजी);
        यदि (it != conv2d_dw_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        स्वतः धकेलो_type_defines = [&](ध्रुव वर्ण * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            यदि (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } अन्यथा यदि (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        धकेलो_type_defines("WEIGHT_TYPE", कुंजी.weight_type);
        धकेलो_type_defines("INPUT_TYPE", कुंजी.निवेश_type);
        धकेलो_type_defines("OUTPUT_TYPE", कुंजी.output_type);

        यदि (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[कुंजी] = pipeline;
        प्रत्यागम conv2d_dw_pipelines[कुंजी];
    }

    webgpu_pipeline get_im2col_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_कुंजी कुंजी = {};
        कुंजी.निवेश_type                      = context.src1->type;
        कुंजी.output_type                     = context.dst->type;

        स्वतः it = im2col_pipelines.find(कुंजी);
        यदि (it != im2col_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        स्वतः धकेलो_type_defines = [&](ध्रुव वर्ण * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            यदि (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } अन्यथा यदि (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        धकेलो_type_defines("INPUT_TYPE", कुंजी.निवेश_type);
        धकेलो_type_defines("OUTPUT_TYPE", कुंजी.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_im2col, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[कुंजी]    = pipeline;
        प्रत्यागम im2col_pipelines[कुंजी];
    }

    webgpu_pipeline get_upscale_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ध्रुव uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ध्रुव uint32_t base_mode  = mode_flags & 0xFFu;
        ध्रुव तार्किक     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_कुंजी कुंजी = {};
        कुंजी.निवेश_type                       = context.src0->type;
        कुंजी.output_type                      = context.dst->type;
        कुंजी.base_mode                        = base_mode;
        कुंजी.antialias                        = antialias;

        स्वतः it = upscale_pipelines.find(कुंजी);
        यदि (it != upscale_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        यदि (कुंजी.निवेश_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } अन्यथा {
            variant += "_src_f32";
        }

        यदि (कुंजी.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } अन्यथा {
            variant += "_dst_f32";
        }

        विकल्प (base_mode) {
            पक्ष GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                भङ्ग;
            पक्ष GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                भङ्ग;
            पक्ष GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported upscale mode");
        }

        यदि (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_upscale, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[कुंजी]   = pipeline;
        प्रत्यागम upscale_pipelines[कुंजी];
    }

  गुप्त:
    स्थिर webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
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
        प्रत्यागम { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
