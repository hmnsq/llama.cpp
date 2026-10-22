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
ढाँचा <typename T> ओळीत शून्य ggml_webgpu_hash_combine(माप_प्रकार & seed, स्थिरांक T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
ओळीत uintptr_t ggml_webgpu_tensor_addr(स्थिरांक ggml_tensor * tensor) {
    स्थिरांक ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    परत (uintptr_t) base_tensor->data + tensor->view_offs;
}

ओळीत तार्किक ggml_webgpu_tensor_equal(स्थिरांक ggml_tensor * a, स्थिरांक ggml_tensor * b) {
    परत a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
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

संरचना ggml_webgpu_द्विमान_shader_decisions {
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

संरचना ggml_webgpu_ssm_scan_pipeline_किल्ली {
    पूर्णांक     type;
    पूर्णांक     d_state;
    uint8_t xbc_overlap;
    तार्किक    a_overlap;
    तार्किक    ids_overlap;

    तार्किक चालक==(स्थिरांक ggml_webgpu_ssm_scan_pipeline_किल्ली & other) स्थिरांक {
        परत type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

संरचना ggml_webgpu_ssm_scan_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_ssm_scan_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        ggml_webgpu_hash_combine(seed, किल्ली.d_state);
        ggml_webgpu_hash_combine(seed, किल्ली.xbc_overlap);
        ggml_webgpu_hash_combine(seed, किल्ली.a_overlap);
        ggml_webgpu_hash_combine(seed, किल्ली.ids_overlap);
        परत seed;
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

संरचना ggml_webgpu_set_rows_pipeline_किल्ली {
    पूर्णांक src0_type;
    पूर्णांक dst_type;
    पूर्णांक vec4;
    पूर्णांक i64_idx;
    पूर्णांक pair_blocks;

    तार्किक चालक==(स्थिरांक ggml_webgpu_set_rows_pipeline_किल्ली & other) स्थिरांक {
        परत src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

संरचना ggml_webgpu_set_rows_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_set_rows_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.src0_type);
        ggml_webgpu_hash_combine(seed, किल्ली.dst_type);
        ggml_webgpu_hash_combine(seed, किल्ली.vec4);
        ggml_webgpu_hash_combine(seed, किल्ली.i64_idx);
        ggml_webgpu_hash_combine(seed, किल्ली.pair_blocks);
        परत seed;
    }
};

संरचना ggml_webgpu_set_rows_shader_decisions {
    तार्किक     vec4;
    तार्किक     i64_idx;
    तार्किक     pair_blocks;
    uint32_t wg_size;
};

/** Set **/

संरचना ggml_webgpu_set_pipeline_किल्ली {
    ggml_type type;
    तार्किक      inplace;

    तार्किक चालक==(स्थिरांक ggml_webgpu_set_pipeline_किल्ली & other) स्थिरांक {
        परत type == other.type && inplace == other.inplace;
    }
};

संरचना ggml_webgpu_set_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_set_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        ggml_webgpu_hash_combine(seed, किल्ली.inplace);
        परत seed;
    }
};

/** Get Rows **/

संरचना ggml_webgpu_get_rows_pipeline_किल्ली {
    ggml_type src_type;
    पूर्णांक       vectorized;

    तार्किक चालक==(स्थिरांक ggml_webgpu_get_rows_pipeline_किल्ली & other) स्थिरांक {
        परत src_type == other.src_type && vectorized == other.vectorized;
    }
};

संरचना ggml_webgpu_get_rows_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_get_rows_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.src_type);
        ggml_webgpu_hash_combine(seed, किल्ली.vectorized);
        परत seed;
    }
};

/** Row Norm **/

संरचना ggml_webgpu_row_norm_pipeline_किल्ली {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    तार्किक      inplace;

    तार्किक चालक==(स्थिरांक ggml_webgpu_row_norm_pipeline_किल्ली & other) स्थिरांक {
        परत op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

संरचना ggml_webgpu_row_norm_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_row_norm_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.op);
        ggml_webgpu_hash_combine(seed, किल्ली.src_type);
        ggml_webgpu_hash_combine(seed, किल्ली.dst_type);
        ggml_webgpu_hash_combine(seed, किल्ली.inplace);
        परत seed;
    }
};

/** RMS_NORM + MUL **/

संरचना ggml_webgpu_rms_norm_mul_pipeline_किल्ली {
    तार्किक inplace;      // rn_src == dst
    तार्किक overlap;      // mul_src == dst
    तार्किक src_overlap;  // rn_src binding overlaps mul_src binding

    तार्किक चालक==(स्थिरांक ggml_webgpu_rms_norm_mul_pipeline_किल्ली & other) स्थिरांक {
        परत inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_rms_norm_mul_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_rms_norm_mul_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.inplace);
        ggml_webgpu_hash_combine(seed, किल्ली.overlap);
        ggml_webgpu_hash_combine(seed, किल्ली.src_overlap);
        परत seed;
    }
};

संरचना ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    तार्किक     inplace     = असत्य;
    तार्किक     overlap     = असत्य;
    तार्किक     src_overlap = असत्य;
};

/** Pad **/
संरचना ggml_webgpu_pad_pipeline_किल्ली {
    तार्किक circular;

    तार्किक चालक==(स्थिरांक ggml_webgpu_pad_pipeline_किल्ली & other) स्थिरांक { परत circular == other.circular; }
};

संरचना ggml_webgpu_pad_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_pad_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.circular);
        परत seed;
    }
};

/** Solve Tri **/
संरचना ggml_webgpu_solve_tri_pipeline_किल्ली {
    पूर्णांक type;
    पूर्णांक n;
    पूर्णांक k;

    तार्किक चालक==(स्थिरांक ggml_webgpu_solve_tri_pipeline_किल्ली & other) स्थिरांक {
        परत type == other.type && n == other.n && k == other.k;
    }
};

संरचना ggml_webgpu_solve_tri_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_solve_tri_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        ggml_webgpu_hash_combine(seed, किल्ली.n);
        ggml_webgpu_hash_combine(seed, किल्ली.k);
        परत seed;
    }
};

/** SSM Conv **/
संरचना ggml_webgpu_ssm_conv_pipeline_किल्ली {
    पूर्णांक type;
    पूर्णांक vectorized;

    तार्किक चालक==(स्थिरांक ggml_webgpu_ssm_conv_pipeline_किल्ली & other) स्थिरांक {
        परत type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
संरचना ggml_webgpu_conv2d_pipeline_किल्ली {
    ggml_type weight_type;
    ggml_type आदान_type;
    ggml_type output_type;

    तार्किक चालक==(स्थिरांक ggml_webgpu_conv2d_pipeline_किल्ली & other) स्थिरांक {
        परत weight_type == other.weight_type && आदान_type == other.आदान_type && output_type == other.output_type;
    }
};

संरचना ggml_webgpu_conv2d_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_conv2d_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.weight_type);
        ggml_webgpu_hash_combine(seed, किल्ली.आदान_type);
        ggml_webgpu_hash_combine(seed, किल्ली.output_type);
        परत seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
संरचना ggml_webgpu_conv2d_dw_pipeline_किल्ली {
    ggml_type weight_type;
    ggml_type आदान_type;
    ggml_type output_type;
    तार्किक      whcn;

    तार्किक चालक==(स्थिरांक ggml_webgpu_conv2d_dw_pipeline_किल्ली & other) स्थिरांक {
        परत weight_type == other.weight_type && आदान_type == other.आदान_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

संरचना ggml_webgpu_conv2d_dw_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_conv2d_dw_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.weight_type);
        ggml_webgpu_hash_combine(seed, किल्ली.आदान_type);
        ggml_webgpu_hash_combine(seed, किल्ली.output_type);
        ggml_webgpu_hash_combine(seed, किल्ली.whcn);
        परत seed;
    }
};

/** Im2Col **/
संरचना ggml_webgpu_im2col_pipeline_किल्ली {
    ggml_type आदान_type;
    ggml_type output_type;

    तार्किक चालक==(स्थिरांक ggml_webgpu_im2col_pipeline_किल्ली & other) स्थिरांक {
        परत आदान_type == other.आदान_type && output_type == other.output_type;
    }
};

संरचना ggml_webgpu_im2col_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_im2col_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.आदान_type);
        ggml_webgpu_hash_combine(seed, किल्ली.output_type);
        परत seed;
    }
};

/** Gated Delta Net **/
संरचना ggml_webgpu_gated_delta_net_pipeline_किल्ली {
    पूर्णांक  type;
    पूर्णांक  s_v;
    पूर्णांक  kda;
    तार्किक fused_cache;

    तार्किक चालक==(स्थिरांक ggml_webgpu_gated_delta_net_pipeline_किल्ली & other) स्थिरांक {
        परत type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

संरचना ggml_webgpu_gated_delta_net_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_gated_delta_net_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        ggml_webgpu_hash_combine(seed, किल्ली.s_v);
        ggml_webgpu_hash_combine(seed, किल्ली.kda);
        परत seed;
    }
};

संरचना ggml_webgpu_ssm_conv_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_ssm_conv_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        ggml_webgpu_hash_combine(seed, किल्ली.vectorized);
        परत seed;
    }
};

/** Scale **/

संरचना ggml_webgpu_scale_pipeline_किल्ली {
    पूर्णांक inplace;

    तार्किक चालक==(स्थिरांक ggml_webgpu_scale_pipeline_किल्ली & other) स्थिरांक { परत inplace == other.inplace; }
};

संरचना ggml_webgpu_scale_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_scale_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.inplace);
        परत seed;
    }
};

/** Upscale **/

संरचना ggml_webgpu_upscale_pipeline_किल्ली {
    ggml_type आदान_type;
    ggml_type output_type;
    uint32_t  base_mode;
    तार्किक      antialias;

    तार्किक चालक==(स्थिरांक ggml_webgpu_upscale_pipeline_किल्ली & other) स्थिरांक {
        परत आदान_type == other.आदान_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

संरचना ggml_webgpu_upscale_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_upscale_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.आदान_type);
        ggml_webgpu_hash_combine(seed, किल्ली.output_type);
        ggml_webgpu_hash_combine(seed, किल्ली.base_mode);
        ggml_webgpu_hash_combine(seed, किल्ली.antialias);
        परत seed;
    }
};

/** Concat **/

संरचना ggml_webgpu_concat_pipeline_किल्ली {
    पूर्णांक  type;
    तार्किक src_overlap;

    तार्किक चालक==(स्थिरांक ggml_webgpu_concat_pipeline_किल्ली & other) स्थिरांक {
        परत type == other.type && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_concat_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_concat_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        ggml_webgpu_hash_combine(seed, किल्ली.src_overlap);
        परत seed;
    }
};

/** Repeat **/

संरचना ggml_webgpu_repeat_pipeline_किल्ली {
    पूर्णांक type;

    तार्किक चालक==(स्थिरांक ggml_webgpu_repeat_pipeline_किल्ली & other) स्थिरांक { परत type == other.type; }
};

संरचना ggml_webgpu_repeat_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_repeat_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        परत seed;
    }
};

/** Binary **/

संरचना ggml_webgpu_द्विमान_pipeline_किल्ली {
    पूर्णांक  type;
    पूर्णांक  op;
    तार्किक inplace;
    तार्किक overlap;
    तार्किक src_overlap;

    तार्किक चालक==(स्थिरांक ggml_webgpu_द्विमान_pipeline_किल्ली & other) स्थिरांक {
        परत type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_द्विमान_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_द्विमान_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        ggml_webgpu_hash_combine(seed, किल्ली.op);
        ggml_webgpu_hash_combine(seed, किल्ली.inplace);
        ggml_webgpu_hash_combine(seed, किल्ली.overlap);
        ggml_webgpu_hash_combine(seed, किल्ली.src_overlap);
        परत seed;
    }
};

/* Add_Id */

संरचना ggml_webgpu_add_id_pipeline_किल्ली {
    तार्किक inplace;

    तार्किक चालक==(स्थिरांक ggml_webgpu_add_id_pipeline_किल्ली & other) स्थिरांक { परत inplace == other.inplace; }
};

संरचना ggml_webgpu_add_id_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_add_id_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.inplace);
        परत seed;
    }
};

/** Unary **/

संरचना ggml_webgpu_unary_pipeline_किल्ली {
    पूर्णांक           type;
    पूर्णांक           op;
    तार्किक          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    तार्किक          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    तार्किक चालक==(स्थिरांक ggml_webgpu_unary_pipeline_किल्ली & other) स्थिरांक {
        परत type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

संरचना ggml_webgpu_unary_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_unary_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        ggml_webgpu_hash_combine(seed, किल्ली.op);
        ggml_webgpu_hash_combine(seed, किल्ली.is_unary);
        ggml_webgpu_hash_combine(seed, किल्ली.inplace);
        ggml_webgpu_hash_combine(seed, किल्ली.ttype);
        परत seed;
    }
};

/** FlashAttention */

संरचना ggml_webgpu_flash_attn_common_pipeline_किल्ली {
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

    तार्किक चालक==(स्थिरांक ggml_webgpu_flash_attn_common_pipeline_किल्ली & other) स्थिरांक {
        परत q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

ओळीत शून्य ggml_webgpu_flash_attn_hash_common_pipeline_किल्ली(माप_प्रकार &                                           seed,
                                                            स्थिरांक ggml_webgpu_flash_attn_common_pipeline_किल्ली & किल्ली) {
    ggml_webgpu_hash_combine(seed, किल्ली.q_type);
    ggml_webgpu_hash_combine(seed, किल्ली.k_type);
    ggml_webgpu_hash_combine(seed, किल्ली.v_type);
    ggml_webgpu_hash_combine(seed, किल्ली.dst_type);
    ggml_webgpu_hash_combine(seed, किल्ली.head_dim_qk);
    ggml_webgpu_hash_combine(seed, किल्ली.head_dim_v);
    ggml_webgpu_hash_combine(seed, किल्ली.k_direct);
    ggml_webgpu_hash_combine(seed, किल्ली.v_direct);
    ggml_webgpu_hash_combine(seed, किल्ली.kv_overlap);
    ggml_webgpu_hash_combine(seed, किल्ली.has_mask);
    ggml_webgpu_hash_combine(seed, किल्ली.has_sinks);
    ggml_webgpu_hash_combine(seed, किल्ली.uses_logit_softcap);
}

संरचना ggml_webgpu_flash_attn_vec_pipeline_किल्ली {
    ggml_webgpu_flash_attn_common_pipeline_किल्ली common;

    तार्किक चालक==(स्थिरांक ggml_webgpu_flash_attn_vec_pipeline_किल्ली & other) स्थिरांक { परत common == other.common; }
};

संरचना ggml_webgpu_flash_attn_vec_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_flash_attn_vec_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_किल्ली(seed, किल्ली.common);
        परत seed;
    }
};

संरचना ggml_webgpu_flash_attn_pipeline_किल्ली {
    ggml_webgpu_flash_attn_common_pipeline_किल्ली common;
    तार्किक                                       use_sg_matrix;

    तार्किक चालक==(स्थिरांक ggml_webgpu_flash_attn_pipeline_किल्ली & other) स्थिरांक {
        परत common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

संरचना ggml_webgpu_flash_attn_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_flash_attn_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_किल्ली(seed, किल्ली.common);
        ggml_webgpu_hash_combine(seed, किल्ली.use_sg_matrix);
        परत seed;
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

ओळीत constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
ओळीत constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

ओळीत माप_प्रकार ggml_webgpu_flash_attn_tensor_offset(स्थिरांक ggml_tensor * tensor) {
    constexpr uintptr_t ptr_base_addr = 0x1000u;
    स्थिरांक ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
    परत reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
}

ओळीत तार्किक ggml_webgpu_flash_attn_float_vec4_aligned(स्थिरांक ggml_tensor * K, माप_प्रकार storage_offset_alignment) {
    स्थिरांक uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_size(K->type));
    परत offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

ओळीत तार्किक ggml_webgpu_flash_attn_float_vec4_aligned(स्थिरांक ggml_tensor * K,
                                                      स्थिरांक ggml_tensor * V,
                                                      माप_प्रकार              storage_offset_alignment) {
    परत ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

ओळीत तार्किक ggml_webgpu_flash_attn_k_direct(स्थिरांक ggml_tensor * Q, स्थिरांक ggml_tensor * K, uint32_t kv_direct_align) {
    परत (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

ओळीत तार्किक ggml_webgpu_flash_attn_v_direct(स्थिरांक ggml_tensor * Q, स्थिरांक ggml_tensor * V, uint32_t kv_direct_align) {
    परत ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

ओळीत ggml_webgpu_flash_attn_common_pipeline_किल्ली ggml_webgpu_flash_attn_make_common_pipeline_किल्ली(
    स्थिरांक ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    तार्किक                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_किल्ली किल्ली = {};
    किल्ली.q_type                                     = context.src0->type;
    किल्ली.k_type                                     = context.src1->type;
    किल्ली.v_type                                     = context.src2->type;
    किल्ली.dst_type                                   = context.dst->type;
    किल्ली.head_dim_qk                                = (uint32_t) context.src0->ne[0];
    किल्ली.head_dim_v                                 = (uint32_t) context.src2->ne[0];
    किल्ली.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    किल्ली.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    किल्ली.kv_overlap         = kv_overlap;
    किल्ली.has_mask           = context.src3 != nullptr;
    किल्ली.has_sinks          = context.src4 != nullptr;
    किल्ली.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    परत किल्ली;
}

ओळीत std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    स्थिरांक ggml_webgpu_flash_attn_common_pipeline_किल्ली & किल्ली,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_size) {
    std::vector<std::string> defines;

    निवड (किल्ली.k_type) {
        स्थिती GGML_TYPE_F32:
            defines.push_back("K_F32");
            थांबा;
        स्थिती GGML_TYPE_F16:
            defines.push_back("K_F16");
            थांबा;
        स्थिती GGML_TYPE_Q4_0:
            defines.push_back("K_Q4_0");
            थांबा;
        स्थिती GGML_TYPE_Q8_0:
            defines.push_back("K_Q8_0");
            थांबा;
        पूर्वनिर्धारित:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(किल्ली.k_type);

    निवड (किल्ली.v_type) {
        स्थिती GGML_TYPE_F32:
            defines.push_back("V_F32");
            थांबा;
        स्थिती GGML_TYPE_F16:
            defines.push_back("V_F16");
            थांबा;
        स्थिती GGML_TYPE_Q4_0:
            defines.push_back("V_Q4_0");
            थांबा;
        स्थिती GGML_TYPE_Q8_0:
            defines.push_back("V_Q8_0");
            थांबा;
        पूर्वनिर्धारित:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(किल्ली.v_type);

    निवड (किल्ली.q_type) {
        स्थिती GGML_TYPE_F32:
            defines.push_back("Q_F32");
            थांबा;
        स्थिती GGML_TYPE_F16:
            defines.push_back("Q_F16");
            थांबा;
        पूर्वनिर्धारित:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(किल्ली.q_type);

    निवड (किल्ली.dst_type) {
        स्थिती GGML_TYPE_F32:
            defines.push_back("DST_F32");
            थांबा;
        स्थिती GGML_TYPE_F16:
            defines.push_back("DST_F16");
            थांबा;
        पूर्वनिर्धारित:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(किल्ली.dst_type);

    जर (किल्ली.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    जर (किल्ली.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    जर (किल्ली.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    जर (किल्ली.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    जर (किल्ली.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    जर (किल्ली.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(किल्ली.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(किल्ली.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(किल्ली.head_dim_v));
    variant += std::string("_hsv") + std::to_string(किल्ली.head_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    जर (ggml_is_quantized(किल्ली.k_type) || ggml_is_quantized(किल्ली.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        जर (ggml_is_quantized(किल्ली.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        जर (ggml_is_quantized(किल्ली.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    परत defines;
}

संरचना ggml_webgpu_flash_attn_vec_reduce_pipeline_किल्ली {
    uint32_t  head_dim_v;
    uint32_t  wg_size;
    ggml_type dst_type;
};

संरचना ggml_webgpu_flash_attn_vec_reduce_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_flash_attn_vec_reduce_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.head_dim_v);
        ggml_webgpu_hash_combine(seed, किल्ली.wg_size);
        ggml_webgpu_hash_combine(seed, किल्ली.dst_type);
        परत seed;
    }
};

ओळीत तार्किक चालक==(स्थिरांक ggml_webgpu_flash_attn_vec_reduce_pipeline_किल्ली & lhs,
                       स्थिरांक ggml_webgpu_flash_attn_vec_reduce_pipeline_किल्ली & rhs) {
    परत lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size && lhs.dst_type == rhs.dst_type;
}

संरचना ggml_webgpu_flash_attn_blk_pipeline_किल्ली {
    uint32_t kv_tile;

    तार्किक चालक==(स्थिरांक ggml_webgpu_flash_attn_blk_pipeline_किल्ली & other) स्थिरांक { परत kv_tile == other.kv_tile; }
};

संरचना ggml_webgpu_flash_attn_blk_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_flash_attn_blk_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.kv_tile);
        परत seed;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
ओळीत माप_प्रकार ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  तार्किक     has_mask,
                                                  तार्किक     kv_direct) {
    स्थिरांक uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    माप_प्रकार         f16_elems    = 0;
    माप_प्रकार         f32_elems    = 0;

    f32_elems += q_tile * head_dim_qk;        // q_shmem
    जर (!kv_direct) {
        f32_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f32_elems += q_tile * head_dim_v;         // o_shmem
    जर (has_mask) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    परत f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

ओळीत uint32_t ggml_webgpu_flash_attn_max_kv_tile(माप_प्रकार   limit_bytes,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t head_dim_qk,
                                                   uint32_t head_dim_v,
                                                   तार्किक     has_mask,
                                                   तार्किक     kv_direct) {
    स्थिरांक माप_प्रकार base_q_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 0, head_dim_qk, head_dim_v, has_mask, kv_direct);
    जर (limit_bytes <= base_q_bytes) {
        परत 0;
    }
    स्थिरांक माप_प्रकार one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, head_dim_qk, head_dim_v, has_mask, kv_direct);
    स्थिरांक माप_प्रकार bytes_per_kv = one_kv_bytes - base_q_bytes;
    जर (bytes_per_kv == 0) {
        परत 0;
    }
    स्थिरांक माप_प्रकार max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    परत (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

ओळीत uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(माप_प्रकार   wg_mem_limit_bytes,
                                                       uint32_t head_dim_qk,
                                                       uint32_t head_dim_v,
                                                       तार्किक     has_mask,
                                                       तार्किक     kv_direct) {
    स्थिरांक uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_limit_bytes, 1u, 1u, head_dim_qk, head_dim_v, has_mask, kv_direct);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    जर (kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        पर्यंत (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    परत kv_tile;
}

ओळीत तार्किक ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(तार्किक                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                स्थिरांक ggml_tensor * Q,
                                                                स्थिरांक ggml_tensor * V) {
    परत supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

संरचना ggml_webgpu_mul_mat_vec_pipeline_किल्ली {
    ggml_type src0_type;
    ggml_type src1_type;
    पूर्णांक       vectorized;
    uint32_t  अंक_cols;
    तार्किक      use_mmvq;
    तार्किक      src_overlap;

    तार्किक चालक==(स्थिरांक ggml_webgpu_mul_mat_vec_pipeline_किल्ली & other) स्थिरांक {
        परत src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               अंक_cols == other.अंक_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_mul_mat_vec_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_mul_mat_vec_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.src0_type);
        ggml_webgpu_hash_combine(seed, किल्ली.src1_type);
        ggml_webgpu_hash_combine(seed, किल्ली.vectorized);
        ggml_webgpu_hash_combine(seed, किल्ली.अंक_cols);
        ggml_webgpu_hash_combine(seed, किल्ली.use_mmvq);
        ggml_webgpu_hash_combine(seed, किल्ली.src_overlap);
        परत seed;
    }
};

संरचना ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
    तार्किक     src_overlap = असत्य;
};

संरचना ggml_webgpu_quantize_q8_pipeline_किल्ली {
    ggml_type src0_type;

    तार्किक चालक==(स्थिरांक ggml_webgpu_quantize_q8_pipeline_किल्ली & other) स्थिरांक { परत src0_type == other.src0_type; }
};

संरचना ggml_webgpu_quantize_q8_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_quantize_q8_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.src0_type);
        परत seed;
    }
};

संरचना ggml_webgpu_mul_mat_pipeline_किल्ली {
    ggml_type src0_type;
    ggml_type src1_type;
    पूर्णांक       vectorized;
    पूर्णांक       use_subgroup_matrix;
    तार्किक      src_overlap;

    तार्किक चालक==(स्थिरांक ggml_webgpu_mul_mat_pipeline_किल्ली & other) स्थिरांक {
        परत src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_mul_mat_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_mul_mat_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.src0_type);
        ggml_webgpu_hash_combine(seed, किल्ली.src1_type);
        ggml_webgpu_hash_combine(seed, किल्ली.vectorized);
        ggml_webgpu_hash_combine(seed, किल्ली.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, किल्ली.src_overlap);
        परत seed;
    }
};

संरचना ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    पूर्णांक      use_subgroup_matrix;

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

संरचना ggml_webgpu_mul_mat_id_gather_pipeline_किल्ली {
    uint32_t n_experts;

    तार्किक चालक==(स्थिरांक ggml_webgpu_mul_mat_id_gather_pipeline_किल्ली & other) स्थिरांक {
        परत n_experts == other.n_experts;
    }
};

संरचना ggml_webgpu_mul_mat_id_gather_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_mul_mat_id_gather_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.n_experts);
        परत seed;
    }
};

संरचना ggml_webgpu_mul_mat_id_pipeline_किल्ली {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  अंक_cols;
    पूर्णांक       vectorized;

    तार्किक चालक==(स्थिरांक ggml_webgpu_mul_mat_id_pipeline_किल्ली & other) स्थिरांक {
        परत src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               अंक_cols == other.अंक_cols && vectorized == other.vectorized;
    }
};

संरचना ggml_webgpu_mul_mat_id_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_mul_mat_id_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.src0_type);
        ggml_webgpu_hash_combine(seed, किल्ली.src1_type);
        ggml_webgpu_hash_combine(seed, किल्ली.n_experts);
        ggml_webgpu_hash_combine(seed, किल्ली.अंक_cols);
        ggml_webgpu_hash_combine(seed, किल्ली.vectorized);
        परत seed;
    }
};

/** Cpy **/

संरचना ggml_webgpu_cpy_pipeline_किल्ली {
    ggml_type src_type;
    ggml_type dst_type;

    तार्किक चालक==(स्थिरांक ggml_webgpu_cpy_pipeline_किल्ली & other) स्थिरांक {
        परत src_type == other.src_type && dst_type == other.dst_type;
    }
};

संरचना ggml_webgpu_cpy_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_cpy_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.src_type);
        ggml_webgpu_hash_combine(seed, किल्ली.dst_type);
        परत seed;
    }
};

/** Glu **/

संरचना ggml_webgpu_glu_pipeline_किल्ली {
    ggml_glu_op glu_op;
    ggml_type   type;
    तार्किक        split;
    तार्किक        src_overlap;

    तार्किक चालक==(स्थिरांक ggml_webgpu_glu_pipeline_किल्ली & other) स्थिरांक {
        परत glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_glu_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_glu_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.glu_op);
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        ggml_webgpu_hash_combine(seed, किल्ली.split);
        ggml_webgpu_hash_combine(seed, किल्ली.src_overlap);
        परत seed;
    }
};

/** Rope **/

संरचना ggml_webgpu_rope_pipeline_किल्ली {
    ggml_type type;
    तार्किक      inplace;
    तार्किक      has_ff;

    तार्किक चालक==(स्थिरांक ggml_webgpu_rope_pipeline_किल्ली & other) स्थिरांक {
        परत type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

संरचना ggml_webgpu_rope_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_rope_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.type);
        ggml_webgpu_hash_combine(seed, किल्ली.inplace);
        ggml_webgpu_hash_combine(seed, किल्ली.has_ff);
        परत seed;
    }
};

/** SoftMax **/

संरचना ggml_webgpu_soft_max_pipeline_किल्ली {
    ggml_type mask_type;
    तार्किक      has_mask;
    तार्किक      has_sink;
    तार्किक      inplace;

    तार्किक चालक==(स्थिरांक ggml_webgpu_soft_max_pipeline_किल्ली & other) स्थिरांक {
        परत mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

संरचना ggml_webgpu_soft_max_pipeline_किल्ली_hash {
    माप_प्रकार चालक()(स्थिरांक ggml_webgpu_soft_max_pipeline_किल्ली & किल्ली) स्थिरांक {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, किल्ली.mask_type);
        ggml_webgpu_hash_combine(seed, किल्ली.has_mask);
        ggml_webgpu_hash_combine(seed, किल्ली.has_sink);
        ggml_webgpu_hash_combine(seed, किल्ली.inplace);
        परत seed;
    }
};

/** MMVQ **/

ओळीत तार्किक ggml_webgpu_can_use_mmvq(स्थिरांक ggml_tensor * src0,
                                     स्थिरांक ggml_tensor * src1,
                                     तार्किक                supports_dot_product,
                                     स्थिरांक std::string & vendor) {
    जर (src1->ne[1] <= 4) {
        तार्किक supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        जर (supports_dp4a && supports_dot_product) {
            निवड (src1->type) {
                स्थिती GGML_TYPE_F32:
                    निवड (src0->type) {
                        स्थिती GGML_TYPE_Q1_0:
                        स्थिती GGML_TYPE_Q4_0:
                        स्थिती GGML_TYPE_Q4_1:
                        स्थिती GGML_TYPE_Q5_0:
                        स्थिती GGML_TYPE_Q5_1:
                        स्थिती GGML_TYPE_Q8_0:
                        स्थिती GGML_TYPE_MXFP4:
                        स्थिती GGML_TYPE_Q2_K:
                        स्थिती GGML_TYPE_Q3_K:
                        स्थिती GGML_TYPE_Q4_K:
                        स्थिती GGML_TYPE_Q5_K:
                        स्थिती GGML_TYPE_Q6_K:
                            परत src0->ne[0] % 4 == 0;
                        पूर्वनिर्धारित:
                            थांबा;
                    }
                    थांबा;
                पूर्वनिर्धारित:
                    थांबा;
            }
        }
    }
    परत असत्य;
}

वर्ग ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<पूर्णांक, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<पूर्णांक, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<पूर्णांक, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<पूर्णांक, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<पूर्णांक, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_किल्ली_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_किल्ली_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_unary_pipeline_किल्ली_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_scale_pipeline_किल्ली_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_किल्ली_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_किल्ली_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_किल्ली_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_किल्ली,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_किल्ली_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_pad_pipeline_किल्ली_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_द्विमान_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_द्विमान_pipeline_किल्ली_hash>
        द्विमान_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_add_id_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_add_id_pipeline_किल्ली_hash>
        add_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_concat_pipeline_किल्ली_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_repeat_pipeline_किल्ली_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_किल्ली,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_किल्ली_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_किल्ली_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_किल्ली,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_किल्ली_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_किल्ली,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_किल्ली_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_किल्ली_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_किल्ली_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_किल्ली_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_किल्ली,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_किल्ली_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_किल्ली_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_किल्ली_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_किल्ली_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_set_pipeline_किल्ली_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_cpy_pipeline_किल्ली_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_glu_pipeline_किल्ली_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_rope_pipeline_किल्ली_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_किल्ली_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_किल्ली_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_किल्ली_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_im2col_pipeline_किल्ली_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_किल्ली,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_किल्ली_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_किल्ली, webgpu_pipeline, ggml_webgpu_upscale_pipeline_किल्ली_hash>
        upscale_pipelines;

  सार्वजनिक:
    ggml_webgpu_shader_lib(wgpu::Device device) { हे->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        स्वतः it = sum_rows_pipelines.find(1);
        जर (it != sum_rows_pipelines.end()) {
            परत it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        परत sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_किल्ली किल्ली = {};
        किल्ली.op                                = context.dst->op;
        किल्ली.src_type                          = context.src0->type;
        किल्ली.dst_type                          = context.dst->type;
        किल्ली.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = row_norm_pipelines.find(किल्ली);
        जर (it != row_norm_pipelines.end()) {
            परत it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        निवड (किल्ली.op) {
            स्थिती GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                थांबा;
            स्थिती GGML_OP_NORM:
                defines.push_back("NORM");
                variant = "norm";
                थांबा;
            स्थिती GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        जर (किल्ली.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        जर (किल्ली.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } अन्यथा जर (किल्ली.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        जर (किल्ली.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } अन्यथा जर (किल्ली.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        स्थिरांक uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        स्वतः processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        स्वतः decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = किल्ली.inplace;
        row_norm_pipelines[किल्ली]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[किल्ली].context = decisions;
        परत row_norm_pipelines[किल्ली];
    }

    webgpu_pipeline get_argmax_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        तार्किक vec4 = context.src0->ne[0] % 4 == 0;

        स्वतः it = argmax_pipelines.find(vec4);
        जर (it != argmax_pipelines.end()) {
            परत it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        जर (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        स्वतः processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        परत argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        स्थिरांक तार्किक                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_किल्ली किल्ली       = {};
        किल्ली.src0_type                               = context.src0->type;
        किल्ली.dst_type                                = context.dst->type;
        किल्ली.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        किल्ली.i64_idx     = context.src1->type == GGML_TYPE_I64;
        किल्ली.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_size(context.dst->type)) % 2 == 0);

        स्वतः it = set_rows_pipelines.find(किल्ली);
        जर (it != set_rows_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        निवड (context.dst->type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                थांबा;
            स्थिती GGML_TYPE_Q8_0:
                defines.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                थांबा;
            स्थिती GGML_TYPE_Q4_0:
                defines.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        जर (context.src0->type == GGML_TYPE_F16) {
            defines.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        जर (किल्ली.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        जर (किल्ली.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        जर (किल्ली.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्थिरांक स्वतः & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        स्वतः         processed          = preprocessor.preprocess(shader_source, defines);
        स्वतः         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = किल्ली.vec4;
        decisions->i64_idx              = किल्ली.i64_idx;
        decisions->pair_blocks          = किल्ली.pair_blocks;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[किल्ली]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[किल्ली].context = decisions;
        परत set_rows_pipelines[किल्ली];
    }

    webgpu_pipeline get_set_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_किल्ली किल्ली = {};
        किल्ली.type                         = context.dst->type;
        किल्ली.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = set_pipelines.find(किल्ली);
        जर (it != set_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for set shader");
        }

        जर (किल्ली.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_set, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = किल्ली.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[किल्ली]       = pipeline;
        परत set_pipelines[किल्ली];
    }

    webgpu_pipeline get_cumsum_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        स्वतः it = cumsum_pipelines.find(1);
        जर (it != cumsum_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        परत cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        तार्किक          is_शीर्ष_k = context.dst->op == GGML_OP_शीर्ष_K;
        // ascending order is 0, descending order is 1
        स्थिरांक int32_t order =
            is_शीर्ष_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        स्वतः it = argsort_pipelines.find(order);
        जर (it != argsort_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        पर्यंत (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        स्वतः processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        स्वतः decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        परत argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        तार्किक          is_शीर्ष_k = context.dst->op == GGML_OP_शीर्ष_K;
        // ascending order is 0, descending order is 1
        स्थिरांक int32_t order =
            is_शीर्ष_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        स्वतः it = argsort_merge_pipelines.find(order);
        जर (it != argsort_merge_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        स्वतः processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        परत argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, तार्किक vec4_aligned) {
        स्थिरांक तार्किक vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_किल्ली किल्ली = {};
        किल्ली.src_type                          = context.src0->type;
        किल्ली.vectorized                        = (पूर्णांक) vectorized;

        स्वतः it = get_rows_pipelines.find(किल्ली);
        जर (it != get_rows_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        स्थिरांक संरचना ggml_type_traits * type_traits = ggml_get_type_traits(किल्ली.src_type);
        स्थिरांक अक्षर *                    type_str    = type_traits->type_name;

        निवड (किल्ली.src_type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                जर (किल्ली.vectorized) {
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
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                थांबा;
            स्थिती GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                थांबा;
            स्थिती GGML_TYPE_BF16:
                defines.push_back("BF16");
                defines.push_back("SRC_TYPE=u32");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                थांबा;
            पूर्वनिर्धारित:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    निवड (किल्ली.src_type) {
                        स्थिती GGML_TYPE_Q1_0:
                        स्थिती GGML_TYPE_Q2_0:
                        स्थिती GGML_TYPE_Q4_0:
                        स्थिती GGML_TYPE_Q5_0:
                        स्थिती GGML_TYPE_Q8_0:
                        स्थिती GGML_TYPE_Q3_K:
                        स्थिती GGML_TYPE_Q6_K:
                        स्थिती GGML_TYPE_IQ2_XXS:
                        स्थिती GGML_TYPE_IQ2_XS:
                        स्थिती GGML_TYPE_IQ2_S:
                        स्थिती GGML_TYPE_IQ3_XXS:
                        स्थिती GGML_TYPE_IQ3_S:
                        स्थिती GGML_TYPE_IQ1_S:
                        स्थिती GGML_TYPE_IQ4_NL:
                        स्थिती GGML_TYPE_MXFP4:
                        स्थिती GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                थांबा;
                            }
                        पूर्वनिर्धारित:
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

                    जर (किल्ली.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } अन्यथा जर ((किल्ली.src_type >= GGML_TYPE_Q4_0 && किल्ली.src_type <= GGML_TYPE_Q8_1) ||
                               किल्ली.src_type == GGML_TYPE_IQ4_NL || किल्ली.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } अन्यथा जर (किल्ली.src_type == GGML_TYPE_Q2_0 || किल्ली.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } अन्यथा जर (किल्ली.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } अन्यथा {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    थांबा;
                }
        }

        जर (किल्ली.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[किल्ली]  = pipeline;
        परत get_rows_pipelines[किल्ली];
    }

    webgpu_pipeline get_scale_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_किल्ली किल्ली = {};
        किल्ली.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = scale_pipelines.find(किल्ली);
        जर (it != scale_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        जर (किल्ली.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_scale, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = किल्ली.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[किल्ली]     = pipeline;
        परत scale_pipelines[किल्ली];
    }

    webgpu_pipeline get_solve_tri_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_किल्ली किल्ली = {};
        किल्ली.type                               = context.dst->type;
        किल्ली.n                                  = (पूर्णांक) context.src0->ne[0];
        किल्ली.k                                  = (पूर्णांक) context.src1->ne[0];

        स्वतः it = solve_tri_pipelines.find(किल्ली);
        जर (it != solve_tri_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                variant += "_f32";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        स्थिरांक uint32_t wg_size       = std::min((uint32_t) किल्ली.n, context.max_wg_size);
        स्थिरांक uint32_t k_tile        = wg_size;
        स्थिरांक uint32_t bytes_per_row = ((uint32_t) किल्ली.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        स्थिरांक uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(किल्ली.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        स्वतः processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[किल्ली] = pipeline;
        परत solve_tri_pipelines[किल्ली];
    }

    webgpu_pipeline get_ssm_conv_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_किल्ली किल्ली = {};
        किल्ली.type                              = context.dst->type;
        किल्ली.vectorized                        = context.src1->ne[0] == 4;

        स्वतः it = ssm_conv_pipelines.find(किल्ली);
        जर (it != ssm_conv_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                variant += "_f32";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        जर (किल्ली.vectorized) {
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
        ssm_conv_pipelines[किल्ली]  = pipeline;
        परत ssm_conv_pipelines[किल्ली];
    }

    webgpu_pipeline get_ssm_scan_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_किल्ली किल्ली = {};
        किल्ली.type                              = context.dst->type;
        किल्ली.d_state                           = (पूर्णांक) context.src0->ne[0];
        किल्ली.xbc_overlap                       = xbc_overlap;

        स्वतः it = ssm_scan_pipelines.find(किल्ली);
        जर (it != ssm_scan_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                variant += "_f32";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        स्थिरांक uint32_t wg_size = (uint32_t) किल्ली.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        जर (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } अन्यथा {
            variant += "_wg_reduce";
        }

        जर (किल्ली.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } अन्यथा जर (किल्ली.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } अन्यथा जर (किल्ली.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(किल्ली.d_state);

        स्वतः processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        स्वतः decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = किल्ली.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[किल्ली]    = pipeline;
        परत ssm_scan_pipelines[किल्ली];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_किल्ली किल्ली = {};
        किल्ली.type                                     = context.dst->type;
        किल्ली.s_v                                      = (पूर्णांक) context.src2->ne[0];
        किल्ली.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        किल्ली.fused_cache                              = context.dst_fuse != nullptr;

        स्वतः it = gated_delta_net_pipelines.find(किल्ली);
        जर (it != gated_delta_net_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                variant += "_f32";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        जर (किल्ली.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        जर (किल्ली.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(किल्ली.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(किल्ली.s_v) + "u");

        स्वतः            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[किल्ली] = pipeline;
        परत gated_delta_net_pipelines[किल्ली];
    }

    webgpu_pipeline get_pad_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_किल्ली किल्ली = {};
        किल्ली.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        स्वतः it = pad_pipelines.find(किल्ली);
        जर (it != pad_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        जर (किल्ली.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_pad, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[किल्ली]       = pipeline;
        परत pad_pipelines[किल्ली];
    }

    webgpu_pipeline get_quantize_q8_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_किल्ली किल्ली = {};
        किल्ली.src0_type                            = context.src0->type;

        स्वतः it = quantize_q8_pipelines.find(किल्ली);
        जर (it != quantize_q8_pipelines.end()) {
            परत it->second;
        }
        स्थिरांक अक्षर *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> defines;
        std::string              variant = "quantize_q8";

        uint32_t wg_size = WEBGPU_MUL_MAT_VEC_WG_SIZE;

        defines.push_back("SRC1_INNER_TYPE=f32");
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        स्थिरांक संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
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
        quantize_q8_pipelines[किल्ली] = pipeline;
        परत quantize_q8_pipelines[किल्ली];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_किल्ली किल्ली = {};
        किल्ली.src0_type                            = context.src0->type;
        किल्ली.src1_type                            = context.src1->type;
        किल्ली.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        किल्ली.अंक_cols   = context.dst->ne[1];
        किल्ली.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
        किल्ली.src_overlap = src_overlap;

        स्वतः it = mul_mat_vec_pipelines.find(किल्ली);
        जर (it != mul_mat_vec_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        स्थिरांक अक्षर *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        निवड (context.src0->type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                थांबा;
            स्थिती GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                थांबा;
            पूर्वनिर्धारित:
                {
                    // Quantized types: use helpers but accumulate in f16
                    स्थिरांक संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    निवड (context.src0->type) {
                        स्थिती GGML_TYPE_Q4_0:
                        स्थिती GGML_TYPE_Q4_1:
                        स्थिती GGML_TYPE_Q5_0:
                        स्थिती GGML_TYPE_Q5_1:
                        स्थिती GGML_TYPE_Q8_0:
                            जर (किल्ली.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            थांबा;
                        स्थिती GGML_TYPE_Q1_0:
                        स्थिती GGML_TYPE_Q2_K:
                        स्थिती GGML_TYPE_Q3_K:
                        स्थिती GGML_TYPE_Q4_K:
                        स्थिती GGML_TYPE_Q5_K:
                        स्थिती GGML_TYPE_Q6_K:
                            जर (किल्ली.use_mmvq) {
                                defines.push_back("K_QUANTS_HANDLING");
                            }
                            थांबा;
                        स्थिती GGML_TYPE_IQ1_S:
                        स्थिती GGML_TYPE_IQ1_M:
                        स्थिती GGML_TYPE_IQ2_S:
                        स्थिती GGML_TYPE_IQ3_S:
                        स्थिती GGML_TYPE_IQ4_NL:
                        स्थिती GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            थांबा;
                        स्थिती GGML_TYPE_IQ2_XXS:
                        स्थिती GGML_TYPE_IQ2_XS:
                        स्थिती GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            थांबा;
                        स्थिती GGML_TYPE_MXFP4:
                            defines.push_back(type_upper + "_LUT");
                            जर (किल्ली.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            थांबा;
                        स्थिती GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            थांबा;
                        पूर्वनिर्धारित:
                            थांबा;
                    }
                    थांबा;
                }
        }

        // src1 type (vector)
        निवड (context.src1->type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(किल्ली.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        जर (किल्ली.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } अन्यथा जर (किल्ली.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } अन्यथा जर (किल्ली.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        जर (किल्ली.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        जर (किल्ली.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        जर (किल्ली.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(किल्ली.अंक_cols));

        स्वतः processed            = preprocessor.preprocess(shader_src, defines);
        स्वतः decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = किल्ली.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[किल्ली] = pipeline;
        परत mul_mat_vec_pipelines[किल्ली];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_mul_mat_pipeline_किल्ली किल्ली = {};
        किल्ली.src0_type                        = context.src0->type;
        किल्ली.src1_type                        = context.src1->type;
        किल्ली.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        किल्ली.use_subgroup_matrix = context.supports_subgroup_matrix;
        किल्ली.src_overlap         = src_overlap;

        स्वतः it = mul_mat_fast_pipelines.find(किल्ली);
        जर (it != mul_mat_fast_pipelines.end()) {
            परत it->second;
        }

        स्थिरांक अक्षर * shader_src = किल्ली.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = किल्ली.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        निवड (context.src1->type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        स्थिरांक संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        स्थिरांक अक्षर *                    src0_name   = src0_traits->type_name;

        निवड (context.src0->type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("TYPE_F32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("TYPE_F16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                थांबा;
            स्थिती GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("TYPE_BF16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                थांबा;
            पूर्वनिर्धारित:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    निवड (context.src0->type) {
                        स्थिती GGML_TYPE_IQ1_S:
                        स्थिती GGML_TYPE_IQ1_M:
                        स्थिती GGML_TYPE_IQ4_NL:
                        स्थिती GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            थांबा;
                        स्थिती GGML_TYPE_IQ2_XXS:
                        स्थिती GGML_TYPE_IQ2_XS:
                        स्थिती GGML_TYPE_IQ2_S:
                        स्थिती GGML_TYPE_IQ3_XXS:
                        स्थिती GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            थांबा;
                        स्थिती GGML_TYPE_MXFP4:
                        स्थिती GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            थांबा;
                        पूर्वनिर्धारित:
                            थांबा;
                    }

                    variant += std::string("_") + src0_name;
                    थांबा;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(किल्ली.vectorized ? "VEC" : "SCALAR");

        स्थिरांक तार्किक is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        जर (किल्ली.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } अन्यथा {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        जर (किल्ली.use_subgroup_matrix) {
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
        जर (किल्ली.vectorized) {
            variant += "_vectorized";
        }

        जर (किल्ली.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        जर (!किल्ली.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        स्वतः processed = preprocessor.preprocess(shader_src, defines);

        स्वतः decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = किल्ली.use_subgroup_matrix;
        जर (किल्ली.use_subgroup_matrix) {
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
        mul_mat_fast_pipelines[किल्ली] = pipeline;
        परत mul_mat_fast_pipelines[किल्ली];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_किल्ली किल्ली = {};
        किल्ली.n_experts                                  = (uint32_t) context.src0->ne[2];

        स्वतः it = mul_mat_id_gather_pipelines.find(किल्ली);
        जर (it != mul_mat_id_gather_pipelines.end()) {
            परत it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(किल्ली.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        स्वतः processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        स्वतः decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[किल्ली] = pipeline;
        परत pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_किल्ली किल्ली = {};
        किल्ली.src0_type                           = context.src0->type;
        किल्ली.src1_type                           = context.src1->type;
        किल्ली.n_experts                           = context.src0->ne[2];
        किल्ली.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        स्वतः it = mul_mat_id_pipelines.find(किल्ली);
        जर (it != mul_mat_id_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        निवड (context.src1->type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        स्थिरांक संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        स्थिरांक अक्षर *                    src0_name   = src0_traits->type_name;

        निवड (context.src0->type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                थांबा;
            स्थिती GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                थांबा;
            पूर्वनिर्धारित:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    निवड (context.src0->type) {
                        स्थिती GGML_TYPE_IQ1_S:
                        स्थिती GGML_TYPE_IQ1_M:
                        स्थिती GGML_TYPE_IQ4_NL:
                        स्थिती GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            थांबा;
                        स्थिती GGML_TYPE_IQ2_XXS:
                        स्थिती GGML_TYPE_IQ2_XS:
                        स्थिती GGML_TYPE_IQ2_S:
                        स्थिती GGML_TYPE_IQ3_XXS:
                        स्थिती GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            थांबा;
                        स्थिती GGML_TYPE_MXFP4:
                        स्थिती GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            थांबा;
                        पूर्वनिर्धारित:
                            थांबा;
                    }

                    variant += std::string("_") + src0_name;
                    थांबा;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(किल्ली.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        स्थिरांक uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        जर (किल्ली.vectorized) {
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
        mul_mat_id_pipelines[किल्ली] = pipeline;
        परत mul_mat_id_pipelines[किल्ली];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_किल्ली किल्ली = {};
        किल्ली.src0_type                           = context.src0->type;
        किल्ली.src1_type                           = context.src1->type;
        किल्ली.n_experts                           = context.src0->ne[2];
        किल्ली.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        स्वतः it = mul_mat_id_vec_pipelines.find(किल्ली);
        जर (it != mul_mat_id_vec_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        स्थिरांक अक्षर *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        निवड (context.src1->type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        निवड (context.src0->type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                थांबा;
            स्थिती GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                थांबा;
            पूर्वनिर्धारित:
                {
                    // Quantized types: use helpers but accumulate in f16
                    स्थिरांक संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    निवड (context.src0->type) {
                        स्थिती GGML_TYPE_IQ1_S:
                        स्थिती GGML_TYPE_IQ1_M:
                        स्थिती GGML_TYPE_IQ2_S:
                        स्थिती GGML_TYPE_IQ3_S:
                        स्थिती GGML_TYPE_IQ4_NL:
                        स्थिती GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            थांबा;
                        स्थिती GGML_TYPE_IQ2_XXS:
                        स्थिती GGML_TYPE_IQ2_XS:
                        स्थिती GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            थांबा;
                        स्थिती GGML_TYPE_MXFP4:
                        स्थिती GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            थांबा;
                        पूर्वनिर्धारित:
                            थांबा;
                    }
                    थांबा;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(किल्ली.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        जर (किल्ली.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } अन्यथा जर (किल्ली.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } अन्यथा जर (किल्ली.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        जर (किल्ली.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(किल्ली.n_experts));

        स्वतः processed = preprocessor.preprocess(shader_src, defines);

        स्वतः decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[किल्ली] = pipeline;
        परत mul_mat_id_vec_pipelines[किल्ली];
    }

    webgpu_pipeline get_unary_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        स्थिरांक तार्किक                     is_unary = context.dst->op == GGML_OP_UNARY;
        स्थिरांक पूर्णांक                      op       = is_unary ? (पूर्णांक) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_किल्ली किल्ली      = {};
        किल्ली.type                                = context.dst->type;
        किल्ली.op                                  = op;
        किल्ली.is_unary                            = is_unary;
        किल्ली.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        किल्ली.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        स्वतः it = unary_pipelines.find(किल्ली);
        जर (it != unary_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            किल्ली.is_unary ? ggml_unary_op_name((ggml_unary_op) किल्ली.op) : ggml_op_name((ggml_op) किल्ली.op);
        defines.push_back(variant);

        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for unary shader");
        }

        जर (किल्ली.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        जर (op == GGML_OP_TRI) {
            निवड (किल्ली.ttype) {
                स्थिती GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    थांबा;
                स्थिती GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    थांबा;
                स्थिती GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    थांबा;
                स्थिती GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    थांबा;
                पूर्वनिर्धारित:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_unary, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = किल्ली.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[किल्ली]     = pipeline;
        परत unary_pipelines[किल्ली];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_किल्ली किल्ली = {};
        किल्ली.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        किल्ली.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        किल्ली.src_overlap                           = src_overlap;

        स्वतः it = rms_norm_mul_pipelines.find(किल्ली);
        जर (it != rms_norm_mul_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        जर (किल्ली.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } अन्यथा जर (किल्ली.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } अन्यथा जर (किल्ली.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        स्वतः pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = किल्ली.inplace;
        pipeline_decisions->overlap     = किल्ली.overlap;
        pipeline_decisions->src_overlap = किल्ली.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[किल्ली]     = pipeline;
        परत rms_norm_mul_pipelines[किल्ली];
    }

    webgpu_pipeline get_द्विमान_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_द्विमान_pipeline_किल्ली किल्ली = {};
        किल्ली.type                            = context.dst->type;
        किल्ली.op                              = context.dst->op;
        किल्ली.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        किल्ली.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        किल्ली.src_overlap                     = src_overlap;

        स्वतः it = द्विमान_pipelines.find(किल्ली);
        जर (it != द्विमान_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) किल्ली.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for binary shader");
        }

        जर (किल्ली.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } अन्यथा जर (किल्ली.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } अन्यथा जर (किल्ली.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed                  = preprocessor.preprocess(wgsl_द्विमान, defines);
        स्वतः pipeline_decisions         = std::make_shared<ggml_webgpu_द्विमान_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = किल्ली.inplace;
        pipeline_decisions->overlap     = किल्ली.overlap;
        pipeline_decisions->src_overlap = किल्ली.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        द्विमान_pipelines[किल्ली]    = pipeline;
        परत द्विमान_pipelines[किल्ली];
    }

    webgpu_pipeline get_add_id_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_किल्ली किल्ली = {};
        किल्ली.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = add_id_pipelines.find(किल्ली);
        जर (it != add_id_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        स्थिरांक अक्षर *             shader_src = wgsl_add_id;

        जर (किल्ली.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed              = preprocessor.preprocess(shader_src, defines);
        स्वतः pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_size = context.max_wg_size;
        pipeline_decisions->inplace = किल्ली.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[किल्ली]    = pipeline;
        परत pipeline;
    }

    webgpu_pipeline get_concat_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_concat_pipeline_किल्ली किल्ली = {};
        किल्ली.type                            = context.dst->type;
        किल्ली.src_overlap                     = src_overlap;

        स्वतः it = concat_pipelines.find(किल्ली);
        जर (it != concat_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for concat shader");
        }

        जर (किल्ली.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_concat, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_द्विमान_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = किल्ली.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[किल्ली]    = pipeline;
        परत concat_pipelines[किल्ली];
    }

    webgpu_pipeline get_repeat_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_किल्ली किल्ली = {};
        किल्ली.type                            = context.dst->type;

        स्वतः it = repeat_pipelines.find(किल्ली);
        जर (it != repeat_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                थांबा;
            स्थिती GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                थांबा;
            स्थिती GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_repeat, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[किल्ली]    = pipeline;
        परत repeat_pipelines[किल्ली];
    }

    webgpu_pipeline get_flash_attn_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, तार्किक kv_overlap) {
        स्थिरांक तार्किक can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_किल्ली किल्ली = {};
        किल्ली.common                              = ggml_webgpu_flash_attn_make_common_pipeline_किल्ली(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        किल्ली.common.k_direct &= decisions.use_sg_matrix && किल्ली.common.k_type == GGML_TYPE_F16;
        किल्ली.common.v_direct &= decisions.use_sg_matrix && किल्ली.common.v_type == GGML_TYPE_F16;
        किल्ली.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_limit_bytes, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            किल्ली.common.head_dim_qk, किल्ली.common.head_dim_v, किल्ली.common.has_mask,
            किल्ली.common.k_direct || किल्ली.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        जर (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = असत्य;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            किल्ली.common.k_direct     = असत्य;
            किल्ली.common.v_direct     = असत्य;
            किल्ली.use_sg_matrix       = असत्य;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, किल्ली.common.head_dim_qk, किल्ली.common.head_dim_v,
                किल्ली.common.has_mask, किल्ली.common.k_direct || किल्ली.common.v_direct);
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

        जर (किल्ली.common.k_direct || किल्ली.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            पर्यंत (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_size;
            }
        }

        स्वतः it = flash_attn_pipelines.find(किल्ली);
        जर (it != flash_attn_pipelines.end()) {
            परत it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(किल्ली.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_size);
        स्थिरांक अक्षर *             shader_src = nullptr;
        जर (!किल्ली.use_sg_matrix) {
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
        flash_attn_pipelines[किल्ली] = pipeline;
        परत flash_attn_pipelines[किल्ली];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, तार्किक kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_किल्ली किल्ली = {};
        किल्ली.common = ggml_webgpu_flash_attn_make_common_pipeline_किल्ली(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        स्वतः it = flash_attn_vec_pipelines.find(किल्ली);
        जर (it != flash_attn_vec_pipelines.end()) {
            परत it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, किल्ली.common.head_dim_qk,
                                                                   किल्ली.common.head_dim_v, किल्ली.common.has_mask,
                                                                   किल्ली.common.k_direct || किल्ली.common.v_direct);
        decisions.wg_size = context.max_subgroup_size;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(किल्ली.common, variant, 1u, decisions.kv_tile, decisions.wg_size);
        जर (किल्ली.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (आकार("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_size;
        स्थिरांक uint32_t D       = किल्ली.common.head_dim_qk | किल्ली.common.head_dim_v;
        स्थिरांक uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_size, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        स्वतः            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[किल्ली] = pipeline;
        परत flash_attn_vec_pipelines[किल्ली];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_किल्ली किल्ली = {};
        किल्ली.kv_tile                                 = kv_tile;
        स्वतः it                                     = flash_attn_blk_pipelines.find(किल्ली);
        जर (it != flash_attn_blk_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(किल्ली.kv_tile));
        variant += std::string("_kvt") + std::to_string(किल्ली.kv_tile);

        uint32_t wg_size = 1;
        पर्यंत ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[किल्ली] = pipeline;
        परत flash_attn_blk_pipelines[किल्ली];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_किल्ली किल्ली = {};
        किल्ली.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        किल्ली.dst_type                                       = context.dst->type;
        किल्ली.wg_size                                        = context.max_wg_size;
        स्वतः it                                            = flash_attn_vec_reduce_pipelines.find(किल्ली);
        जर (it != flash_attn_vec_reduce_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        निवड (किल्ली.dst_type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("DST_F32");
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("DST_F16");
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(किल्ली.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(किल्ली.head_dim_v));
        variant += std::string("_hsv") + std::to_string(किल्ली.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[किल्ली] = pipeline;
        परत flash_attn_vec_reduce_pipelines[किल्ली];
    }

    webgpu_pipeline get_cpy_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_किल्ली किल्ली = {};
        किल्ली.src_type                     = context.src0->type;
        किल्ली.dst_type                     = context.dst->type;

        स्वतः it = cpy_pipelines.find(किल्ली);
        जर (it != cpy_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        निवड (किल्ली.src_type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                थांबा;
            स्थिती GGML_TYPE_I32:
                defines.push_back("SRC_I32");
                variant += "_i32";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        निवड (किल्ली.dst_type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                थांबा;
            स्थिती GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_cpy, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[किल्ली]       = pipeline;
        परत cpy_pipelines[किल्ली];
    }

    webgpu_pipeline get_glu_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_glu_pipeline_किल्ली किल्ली = {};
        किल्ली.glu_op                       = ggml_get_glu_op(context.dst);
        किल्ली.type                         = context.dst->type;
        किल्ली.split                        = (context.src1 != nullptr);
        किल्ली.src_overlap                  = src_overlap;

        स्वतः it = glu_pipelines.find(किल्ली);
        जर (it != glu_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        निवड (किल्ली.glu_op) {
            स्थिती GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                थांबा;
            स्थिती GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                थांबा;
            स्थिती GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                थांबा;
            स्थिती GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                थांबा;
            स्थिती GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                थांबा;
            स्थिती GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                थांबा;
            स्थिती GGML_GLU_OP_SWIGLU_CLAMP:
                defines.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported GLU op");
        }
        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        जर (किल्ली.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } अन्यथा जर (किल्ली.split) {
            variant += "_split";
        } अन्यथा {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_glu, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = किल्ली.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[किल्ली]       = pipeline;
        परत glu_pipelines[किल्ली];
    }

    webgpu_pipeline get_rope_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_किल्ली किल्ली = {};
        किल्ली.type                          = context.dst->type;
        किल्ली.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        किल्ली.has_ff                        = (context.src2 != nullptr);

        स्वतः it = rope_pipelines.find(किल्ली);
        जर (it != rope_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        निवड (किल्ली.type) {
            स्थिती GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                थांबा;
            स्थिती GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        जर (किल्ली.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        जर (किल्ली.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_rope, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = किल्ली.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[किल्ली]      = pipeline;
        परत rope_pipelines[किल्ली];
    }

    webgpu_pipeline get_soft_max_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_किल्ली किल्ली = {};
        किल्ली.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        किल्ली.has_mask                          = (context.src1 != nullptr);
        किल्ली.has_sink                          = (context.src2 != nullptr);
        किल्ली.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = soft_max_pipelines.find(किल्ली);
        जर (it != soft_max_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        जर (किल्ली.has_mask) {
            defines.push_back("HAS_MASK");
            निवड (किल्ली.mask_type) {
                स्थिती GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    थांबा;
                स्थिती GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    थांबा;
                पूर्वनिर्धारित:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        जर (किल्ली.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        जर (किल्ली.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = किल्ली.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[किल्ली]  = pipeline;
        परत soft_max_pipelines[किल्ली];
    }

    webgpu_pipeline get_conv2d_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_किल्ली किल्ली = {};
        किल्ली.weight_type                     = context.src0->type;
        किल्ली.आदान_type                      = context.src1->type;
        किल्ली.output_type                     = context.dst->type;

        स्वतः it = conv2d_pipelines.find(किल्ली);
        जर (it != conv2d_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        स्वतः ढकला_type_defines = [&](स्थिरांक अक्षर * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            जर (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } अन्यथा जर (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ढकला_type_defines("WEIGHT_TYPE", किल्ली.weight_type);
        ढकला_type_defines("INPUT_TYPE", किल्ली.आदान_type);
        ढकला_type_defines("OUTPUT_TYPE", किल्ली.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[किल्ली]    = pipeline;
        परत conv2d_pipelines[किल्ली];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context, तार्किक whcn) {
        ggml_webgpu_conv2d_dw_pipeline_किल्ली किल्ली = {};
        किल्ली.weight_type                        = context.src0->type;
        किल्ली.आदान_type                         = context.src1->type;
        किल्ली.output_type                        = context.dst->type;
        किल्ली.whcn                               = whcn;

        स्वतः it = conv2d_dw_pipelines.find(किल्ली);
        जर (it != conv2d_dw_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        स्वतः ढकला_type_defines = [&](स्थिरांक अक्षर * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            जर (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } अन्यथा जर (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ढकला_type_defines("WEIGHT_TYPE", किल्ली.weight_type);
        ढकला_type_defines("INPUT_TYPE", किल्ली.आदान_type);
        ढकला_type_defines("OUTPUT_TYPE", किल्ली.output_type);

        जर (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[किल्ली] = pipeline;
        परत conv2d_dw_pipelines[किल्ली];
    }

    webgpu_pipeline get_im2col_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_किल्ली किल्ली = {};
        किल्ली.आदान_type                      = context.src1->type;
        किल्ली.output_type                     = context.dst->type;

        स्वतः it = im2col_pipelines.find(किल्ली);
        जर (it != im2col_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        स्वतः ढकला_type_defines = [&](स्थिरांक अक्षर * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            जर (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } अन्यथा जर (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ढकला_type_defines("INPUT_TYPE", किल्ली.आदान_type);
        ढकला_type_defines("OUTPUT_TYPE", किल्ली.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_im2col, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[किल्ली]    = pipeline;
        परत im2col_pipelines[किल्ली];
    }

    webgpu_pipeline get_upscale_pipeline(स्थिरांक ggml_webgpu_shader_lib_context & context) {
        स्थिरांक uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        स्थिरांक uint32_t base_mode  = mode_flags & 0xFFu;
        स्थिरांक तार्किक     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_किल्ली किल्ली = {};
        किल्ली.आदान_type                       = context.src0->type;
        किल्ली.output_type                      = context.dst->type;
        किल्ली.base_mode                        = base_mode;
        किल्ली.antialias                        = antialias;

        स्वतः it = upscale_pipelines.find(किल्ली);
        जर (it != upscale_pipelines.end()) {
            परत it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        जर (किल्ली.आदान_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } अन्यथा {
            variant += "_src_f32";
        }

        जर (किल्ली.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } अन्यथा {
            variant += "_dst_f32";
        }

        निवड (base_mode) {
            स्थिती GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                थांबा;
            स्थिती GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                थांबा;
            स्थिती GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                थांबा;
            पूर्वनिर्धारित:
                GGML_ABORT("Unsupported upscale mode");
        }

        जर (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        स्वतः processed           = preprocessor.preprocess(wgsl_upscale, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[किल्ली]   = pipeline;
        परत upscale_pipelines[किल्ली];
    }

  खाजगी:
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
        परत { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
