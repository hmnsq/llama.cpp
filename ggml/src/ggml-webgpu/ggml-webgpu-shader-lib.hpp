#अगर_अघोषित GGML_WEBGPU_SHADER_LIB_HPP
#घोषणा GGML_WEBGPU_SHADER_LIB_HPP

#समावेश "ggml-impl.h"
#समावेश "ggml-wgsl-shaders.hpp"
#समावेश "ggml.h"
#समावेश "pre_wgsl.hpp"

#समावेश <webgpu/webgpu_cpp.h>

#समावेश <algorithm>
#समावेश <memory>
#समावेश <string>
#समावेश <unordered_map>
#समावेश <vector>

#घोषणा GGML_WEBGPU_F16_आकार_BYTES                   2
#घोषणा GGML_WEBGPU_F32_आकार_BYTES                   4
#घोषणा GGML_WEBGPU_I32_आकार_BYTES                   4
#घोषणा GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#घोषणा GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#घोषणा GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#घोषणा GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#घोषणा GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_आकार     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#घोषणा GGML_WEBGPU_KV_SEQ_PAD                       256u

#घोषणा GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_आकार 512u

// Matrix multiplication parameters

// Register tiling parameters
#घोषणा WEBGPU_MUL_MAT_TILE_M           4
#घोषणा WEBGPU_MUL_MAT_TILE_N           4
#घोषणा WEBGPU_MUL_MAT_WG_आकार_M        8
#घोषणा WEBGPU_MUL_MAT_WG_आकार_N        8
#घोषणा WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#घोषणा WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#घोषणा WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#घोषणा WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#घोषणा WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M     4
#घोषणा WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N     2
#घोषणा WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#घोषणा WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#घोषणा WEBGPU_MUL_MAT_VEC_WG_आकार 256

#घोषणा WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG    4
#घोषणा WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG 4
#घोषणा WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG      4

// default size for reg-tile matrix multiplication
#घोषणा WEBGPU_MUL_MAT_WG_आकार 256

// Same hash combine function as in boost
ढाँचा <typename T> अंतरभूत व्योम ggml_webgpu_hash_combine(माप_प्रकार & seed, स्थिर T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
अंतरभूत uintptr_t ggml_webgpu_tensor_addr(स्थिर ggml_tensor * tensor) {
    स्थिर ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    वापस (uintptr_t) base_tensor->तथ्य + tensor->view_offs;
}

अंतरभूत bool ggml_webgpu_tensor_equal(स्थिर ggml_tensor * a, स्थिर ggml_tensor * b) {
    वापस a->बफर == b->बफर && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

काष्ठा ggml_webgpu_shader_lib_context {
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * src2;
    ggml_tensor * src3;
    ggml_tensor * src4;
    ggml_tensor * src5;
    ggml_tensor * dst;
    ggml_tensor * dst_fuse;

    uint32_t    max_wg_आकार;
    माप_प्रकार      wg_mem_limit_bytes       = 0;
    bool        supports_subgroups       = false;
    bool        supports_subgroup_matrix = false;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_आकार        = 0;
    uint32_t    max_subgroup_आकार        = 0;
    bool        supports_dot_product     = false;
    std::string vendor;
};

काष्ठा webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<व्योम> context = nullptr;
};

काष्ठा ggml_webgpu_generic_shader_decisions {
    uint32_t wg_आकार = 0;
    bool     inplace = false;
};

काष्ठा ggml_webgpu_get_rows_shader_decisions {
    uint32_t wg_आकार    = 0;
    bool     vectorized = false;
};

काष्ठा ggml_webgpu_द्विआधारी_shader_decisions {
    uint32_t wg_आकार     = 0;
    bool     inplace     = false;
    bool     overlap     = false;
    bool     src_overlap = false;
};

काष्ठा ggml_webgpu_glu_shader_decisions {
    uint32_t wg_आकार     = 0;
    bool     src_overlap = false;
};

काष्ठा ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<व्योम> decisions;
};

काष्ठा ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_आकार_2;
    uint32_t tokens_per_wg;
};

काष्ठा ggml_webgpu_ssm_scan_pipeline_कुंजी {
    पूर्णांक     type;
    पूर्णांक     d_state;
    uint8_t xbc_overlap;
    bool    a_overlap;
    bool    ids_overlap;

    bool चालक==(स्थिर ggml_webgpu_ssm_scan_pipeline_कुंजी & other) स्थिर {
        वापस type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

काष्ठा ggml_webgpu_ssm_scan_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_ssm_scan_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.d_state);
        ggml_webgpu_hash_combine(seed, कुंजी.xbc_overlap);
        ggml_webgpu_hash_combine(seed, कुंजी.a_overlap);
        ggml_webgpu_hash_combine(seed, कुंजी.ids_overlap);
        वापस seed;
    }
};

काष्ठा ggml_webgpu_ssm_scan_shader_decisions {
    uint32_t wg_आकार;
    uint32_t tokens_per_tile;
    uint8_t  xbc_overlap = 0;
    bool     a_overlap   = false;
    bool     ids_overlap = false;
};

/** Argsort **/

काष्ठा ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_आकार;
    माप_प्रकार   wg_mem_limit_bytes;
    int32_t  order;
};

/** Set Rows **/

काष्ठा ggml_webgpu_set_rows_pipeline_कुंजी {
    पूर्णांक src0_type;
    पूर्णांक dst_type;
    पूर्णांक vec4;
    पूर्णांक i64_idx;
    पूर्णांक pair_blocks;

    bool चालक==(स्थिर ggml_webgpu_set_rows_pipeline_कुंजी & other) स्थिर {
        वापस src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

काष्ठा ggml_webgpu_set_rows_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_set_rows_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src0_type);
        ggml_webgpu_hash_combine(seed, कुंजी.dst_type);
        ggml_webgpu_hash_combine(seed, कुंजी.vec4);
        ggml_webgpu_hash_combine(seed, कुंजी.i64_idx);
        ggml_webgpu_hash_combine(seed, कुंजी.pair_blocks);
        वापस seed;
    }
};

काष्ठा ggml_webgpu_set_rows_shader_decisions {
    bool     vec4;
    bool     i64_idx;
    bool     pair_blocks;
    uint32_t wg_आकार;
};

/** Set **/

काष्ठा ggml_webgpu_set_pipeline_कुंजी {
    ggml_type type;
    bool      inplace;

    bool चालक==(स्थिर ggml_webgpu_set_pipeline_कुंजी & other) स्थिर {
        वापस type == other.type && inplace == other.inplace;
    }
};

काष्ठा ggml_webgpu_set_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_set_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        वापस seed;
    }
};

/** Get Rows **/

काष्ठा ggml_webgpu_get_rows_pipeline_कुंजी {
    ggml_type src_type;
    पूर्णांक       vectorized;

    bool चालक==(स्थिर ggml_webgpu_get_rows_pipeline_कुंजी & other) स्थिर {
        वापस src_type == other.src_type && vectorized == other.vectorized;
    }
};

काष्ठा ggml_webgpu_get_rows_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_get_rows_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src_type);
        ggml_webgpu_hash_combine(seed, कुंजी.vectorized);
        वापस seed;
    }
};

/** Row Norm **/

काष्ठा ggml_webgpu_row_norm_pipeline_कुंजी {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    bool      inplace;

    bool चालक==(स्थिर ggml_webgpu_row_norm_pipeline_कुंजी & other) स्थिर {
        वापस op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

काष्ठा ggml_webgpu_row_norm_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_row_norm_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.op);
        ggml_webgpu_hash_combine(seed, कुंजी.src_type);
        ggml_webgpu_hash_combine(seed, कुंजी.dst_type);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        वापस seed;
    }
};

/** RMS_NORM + MUL **/

काष्ठा ggml_webgpu_rms_norm_mul_pipeline_कुंजी {
    bool inplace;      // rn_src == dst
    bool overlap;      // mul_src == dst
    bool src_overlap;  // rn_src binding overlaps mul_src binding

    bool चालक==(स्थिर ggml_webgpu_rms_norm_mul_pipeline_कुंजी & other) स्थिर {
        वापस inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

काष्ठा ggml_webgpu_rms_norm_mul_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_rms_norm_mul_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        ggml_webgpu_hash_combine(seed, कुंजी.overlap);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        वापस seed;
    }
};

काष्ठा ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_आकार     = 0;
    bool     inplace     = false;
    bool     overlap     = false;
    bool     src_overlap = false;
};

/** Pad **/
काष्ठा ggml_webgpu_pad_pipeline_कुंजी {
    bool circular;

    bool चालक==(स्थिर ggml_webgpu_pad_pipeline_कुंजी & other) स्थिर { वापस circular == other.circular; }
};

काष्ठा ggml_webgpu_pad_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_pad_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.circular);
        वापस seed;
    }
};

/** Solve Tri **/
काष्ठा ggml_webgpu_solve_tri_pipeline_कुंजी {
    पूर्णांक type;
    पूर्णांक n;
    पूर्णांक k;

    bool चालक==(स्थिर ggml_webgpu_solve_tri_pipeline_कुंजी & other) स्थिर {
        वापस type == other.type && n == other.n && k == other.k;
    }
};

काष्ठा ggml_webgpu_solve_tri_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_solve_tri_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.n);
        ggml_webgpu_hash_combine(seed, कुंजी.k);
        वापस seed;
    }
};

/** SSM Conv **/
काष्ठा ggml_webgpu_ssm_conv_pipeline_कुंजी {
    पूर्णांक type;
    पूर्णांक vectorized;

    bool चालक==(स्थिर ggml_webgpu_ssm_conv_pipeline_कुंजी & other) स्थिर {
        वापस type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
काष्ठा ggml_webgpu_conv2d_pipeline_कुंजी {
    ggml_type weight_type;
    ggml_type निवेश_type;
    ggml_type output_type;

    bool चालक==(स्थिर ggml_webgpu_conv2d_pipeline_कुंजी & other) स्थिर {
        वापस weight_type == other.weight_type && निवेश_type == other.निवेश_type && output_type == other.output_type;
    }
};

काष्ठा ggml_webgpu_conv2d_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_conv2d_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.weight_type);
        ggml_webgpu_hash_combine(seed, कुंजी.निवेश_type);
        ggml_webgpu_hash_combine(seed, कुंजी.output_type);
        वापस seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
काष्ठा ggml_webgpu_conv2d_dw_pipeline_कुंजी {
    ggml_type weight_type;
    ggml_type निवेश_type;
    ggml_type output_type;
    bool      whcn;

    bool चालक==(स्थिर ggml_webgpu_conv2d_dw_pipeline_कुंजी & other) स्थिर {
        वापस weight_type == other.weight_type && निवेश_type == other.निवेश_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

काष्ठा ggml_webgpu_conv2d_dw_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_conv2d_dw_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.weight_type);
        ggml_webgpu_hash_combine(seed, कुंजी.निवेश_type);
        ggml_webgpu_hash_combine(seed, कुंजी.output_type);
        ggml_webgpu_hash_combine(seed, कुंजी.whcn);
        वापस seed;
    }
};

/** Im2Col **/
काष्ठा ggml_webgpu_im2col_pipeline_कुंजी {
    ggml_type निवेश_type;
    ggml_type output_type;

    bool चालक==(स्थिर ggml_webgpu_im2col_pipeline_कुंजी & other) स्थिर {
        वापस निवेश_type == other.निवेश_type && output_type == other.output_type;
    }
};

काष्ठा ggml_webgpu_im2col_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_im2col_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.निवेश_type);
        ggml_webgpu_hash_combine(seed, कुंजी.output_type);
        वापस seed;
    }
};

/** Gated Delta Net **/
काष्ठा ggml_webgpu_gated_delta_net_pipeline_कुंजी {
    पूर्णांक  type;
    पूर्णांक  s_v;
    पूर्णांक  kda;
    bool fused_cache;

    bool चालक==(स्थिर ggml_webgpu_gated_delta_net_pipeline_कुंजी & other) स्थिर {
        वापस type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

काष्ठा ggml_webgpu_gated_delta_net_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_gated_delta_net_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.s_v);
        ggml_webgpu_hash_combine(seed, कुंजी.kda);
        वापस seed;
    }
};

काष्ठा ggml_webgpu_ssm_conv_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_ssm_conv_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.vectorized);
        वापस seed;
    }
};

/** Scale **/

काष्ठा ggml_webgpu_scale_pipeline_कुंजी {
    पूर्णांक inplace;

    bool चालक==(स्थिर ggml_webgpu_scale_pipeline_कुंजी & other) स्थिर { वापस inplace == other.inplace; }
};

काष्ठा ggml_webgpu_scale_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_scale_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        वापस seed;
    }
};

/** Upscale **/

काष्ठा ggml_webgpu_upscale_pipeline_कुंजी {
    ggml_type निवेश_type;
    ggml_type output_type;
    uint32_t  base_mode;
    bool      antialias;

    bool चालक==(स्थिर ggml_webgpu_upscale_pipeline_कुंजी & other) स्थिर {
        वापस निवेश_type == other.निवेश_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

काष्ठा ggml_webgpu_upscale_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_upscale_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.निवेश_type);
        ggml_webgpu_hash_combine(seed, कुंजी.output_type);
        ggml_webgpu_hash_combine(seed, कुंजी.base_mode);
        ggml_webgpu_hash_combine(seed, कुंजी.antialias);
        वापस seed;
    }
};

/** Concat **/

काष्ठा ggml_webgpu_concat_pipeline_कुंजी {
    पूर्णांक  type;
    bool src_overlap;

    bool चालक==(स्थिर ggml_webgpu_concat_pipeline_कुंजी & other) स्थिर {
        वापस type == other.type && src_overlap == other.src_overlap;
    }
};

काष्ठा ggml_webgpu_concat_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_concat_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        वापस seed;
    }
};

/** Repeat **/

काष्ठा ggml_webgpu_repeat_pipeline_कुंजी {
    पूर्णांक type;

    bool चालक==(स्थिर ggml_webgpu_repeat_pipeline_कुंजी & other) स्थिर { वापस type == other.type; }
};

काष्ठा ggml_webgpu_repeat_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_repeat_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        वापस seed;
    }
};

/** Binary **/

काष्ठा ggml_webgpu_द्विआधारी_pipeline_कुंजी {
    पूर्णांक  type;
    पूर्णांक  op;
    bool inplace;
    bool overlap;
    bool src_overlap;

    bool चालक==(स्थिर ggml_webgpu_द्विआधारी_pipeline_कुंजी & other) स्थिर {
        वापस type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

काष्ठा ggml_webgpu_द्विआधारी_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_द्विआधारी_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.op);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        ggml_webgpu_hash_combine(seed, कुंजी.overlap);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        वापस seed;
    }
};

/* Add_Id */

काष्ठा ggml_webgpu_add_id_pipeline_कुंजी {
    bool inplace;

    bool चालक==(स्थिर ggml_webgpu_add_id_pipeline_कुंजी & other) स्थिर { वापस inplace == other.inplace; }
};

काष्ठा ggml_webgpu_add_id_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_add_id_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        वापस seed;
    }
};

/** Unary **/

काष्ठा ggml_webgpu_unary_pipeline_कुंजी {
    पूर्णांक           type;
    पूर्णांक           op;
    bool          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    bool          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    bool चालक==(स्थिर ggml_webgpu_unary_pipeline_कुंजी & other) स्थिर {
        वापस type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

काष्ठा ggml_webgpu_unary_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_unary_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.op);
        ggml_webgpu_hash_combine(seed, कुंजी.is_unary);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        ggml_webgpu_hash_combine(seed, कुंजी.ttype);
        वापस seed;
    }
};

/** FlashAttention */

काष्ठा ggml_webgpu_flash_attn_common_pipeline_कुंजी {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  शीर्ष_dim_qk;
    uint32_t  शीर्ष_dim_v;
    bool      k_direct;
    bool      v_direct;
    bool      kv_overlap;
    bool      has_mask;
    bool      has_sinks;
    bool      uses_logit_softcap;

    bool चालक==(स्थिर ggml_webgpu_flash_attn_common_pipeline_कुंजी & other) स्थिर {
        वापस q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && शीर्ष_dim_qk == other.शीर्ष_dim_qk && शीर्ष_dim_v == other.शीर्ष_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

अंतरभूत व्योम ggml_webgpu_flash_attn_hash_common_pipeline_कुंजी(माप_प्रकार &                                           seed,
                                                            स्थिर ggml_webgpu_flash_attn_common_pipeline_कुंजी & कुंजी) {
    ggml_webgpu_hash_combine(seed, कुंजी.q_type);
    ggml_webgpu_hash_combine(seed, कुंजी.k_type);
    ggml_webgpu_hash_combine(seed, कुंजी.v_type);
    ggml_webgpu_hash_combine(seed, कुंजी.dst_type);
    ggml_webgpu_hash_combine(seed, कुंजी.शीर्ष_dim_qk);
    ggml_webgpu_hash_combine(seed, कुंजी.शीर्ष_dim_v);
    ggml_webgpu_hash_combine(seed, कुंजी.k_direct);
    ggml_webgpu_hash_combine(seed, कुंजी.v_direct);
    ggml_webgpu_hash_combine(seed, कुंजी.kv_overlap);
    ggml_webgpu_hash_combine(seed, कुंजी.has_mask);
    ggml_webgpu_hash_combine(seed, कुंजी.has_sinks);
    ggml_webgpu_hash_combine(seed, कुंजी.uses_logit_softcap);
}

काष्ठा ggml_webgpu_flash_attn_vec_pipeline_कुंजी {
    ggml_webgpu_flash_attn_common_pipeline_कुंजी common;

    bool चालक==(स्थिर ggml_webgpu_flash_attn_vec_pipeline_कुंजी & other) स्थिर { वापस common == other.common; }
};

काष्ठा ggml_webgpu_flash_attn_vec_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_flash_attn_vec_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_कुंजी(seed, कुंजी.common);
        वापस seed;
    }
};

काष्ठा ggml_webgpu_flash_attn_pipeline_कुंजी {
    ggml_webgpu_flash_attn_common_pipeline_कुंजी common;
    bool                                       use_sg_matrix;

    bool चालक==(स्थिर ggml_webgpu_flash_attn_pipeline_कुंजी & other) स्थिर {
        वापस common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

काष्ठा ggml_webgpu_flash_attn_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_flash_attn_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_कुंजी(seed, कुंजी.common);
        ggml_webgpu_hash_combine(seed, कुंजी.use_sg_matrix);
        वापस seed;
    }
};

काष्ठा ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_आकार = 0;
};

काष्ठा ggml_webgpu_flash_attn_decisions {
    bool     use_sg_matrix = false;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_आकार       = 0;
};

अंतरभूत constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
अंतरभूत constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

अंतरभूत माप_प्रकार ggml_webgpu_flash_attn_tensor_offset(स्थिर ggml_tensor * tensor) {
    constexpr uintptr_t ptr_base_addr = 0x1000u;
    स्थिर ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
    वापस reinterpret_cast<uintptr_t>(base->तथ्य) - ptr_base_addr + tensor->view_offs;
}

अंतरभूत bool ggml_webgpu_flash_attn_float_vec4_aligned(स्थिर ggml_tensor * K, माप_प्रकार storage_offset_alignment) {
    स्थिर uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_आकार(K->type));
    वापस offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

अंतरभूत bool ggml_webgpu_flash_attn_float_vec4_aligned(स्थिर ggml_tensor * K,
                                                      स्थिर ggml_tensor * V,
                                                      माप_प्रकार              storage_offset_alignment) {
    वापस ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

अंतरभूत bool ggml_webgpu_flash_attn_k_direct(स्थिर ggml_tensor * Q, स्थिर ggml_tensor * K, uint32_t kv_direct_align) {
    वापस (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

अंतरभूत bool ggml_webgpu_flash_attn_v_direct(स्थिर ggml_tensor * Q, स्थिर ggml_tensor * V, uint32_t kv_direct_align) {
    वापस ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

अंतरभूत ggml_webgpu_flash_attn_common_pipeline_कुंजी ggml_webgpu_flash_attn_make_common_pipeline_कुंजी(
    स्थिर ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    bool                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_कुंजी कुंजी = {};
    कुंजी.q_type                                     = context.src0->type;
    कुंजी.k_type                                     = context.src1->type;
    कुंजी.v_type                                     = context.src2->type;
    कुंजी.dst_type                                   = context.dst->type;
    कुंजी.शीर्ष_dim_qk                                = (uint32_t) context.src0->ne[0];
    कुंजी.शीर्ष_dim_v                                 = (uint32_t) context.src2->ne[0];
    कुंजी.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    कुंजी.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    कुंजी.kv_overlap         = kv_overlap;
    कुंजी.has_mask           = context.src3 != nullptr;
    कुंजी.has_sinks          = context.src4 != nullptr;
    कुंजी.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    वापस कुंजी;
}

अंतरभूत std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    स्थिर ggml_webgpu_flash_attn_common_pipeline_कुंजी & कुंजी,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_आकार) {
    std::vector<std::string> defines;

    चयन (कुंजी.k_type) {
        हाल GGML_TYPE_F32:
            defines.push_back("K_F32");
            अवरोध;
        हाल GGML_TYPE_F16:
            defines.push_back("K_F16");
            अवरोध;
        हाल GGML_TYPE_Q4_0:
            defines.push_back("K_Q4_0");
            अवरोध;
        हाल GGML_TYPE_Q8_0:
            defines.push_back("K_Q8_0");
            अवरोध;
        शेष:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(कुंजी.k_type);

    चयन (कुंजी.v_type) {
        हाल GGML_TYPE_F32:
            defines.push_back("V_F32");
            अवरोध;
        हाल GGML_TYPE_F16:
            defines.push_back("V_F16");
            अवरोध;
        हाल GGML_TYPE_Q4_0:
            defines.push_back("V_Q4_0");
            अवरोध;
        हाल GGML_TYPE_Q8_0:
            defines.push_back("V_Q8_0");
            अवरोध;
        शेष:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(कुंजी.v_type);

    चयन (कुंजी.q_type) {
        हाल GGML_TYPE_F32:
            defines.push_back("Q_F32");
            अवरोध;
        हाल GGML_TYPE_F16:
            defines.push_back("Q_F16");
            अवरोध;
        शेष:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(कुंजी.q_type);

    चयन (कुंजी.dst_type) {
        हाल GGML_TYPE_F32:
            defines.push_back("DST_F32");
            अवरोध;
        हाल GGML_TYPE_F16:
            defines.push_back("DST_F16");
            अवरोध;
        शेष:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(कुंजी.dst_type);

    अगर (कुंजी.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    अगर (कुंजी.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    अगर (कुंजी.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    अगर (कुंजी.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    अगर (कुंजी.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    अगर (कुंजी.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(कुंजी.शीर्ष_dim_qk));
    variant += std::string("_hsqk") + std::to_string(कुंजी.शीर्ष_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(कुंजी.शीर्ष_dim_v));
    variant += std::string("_hsv") + std::to_string(कुंजी.शीर्ष_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));

    अगर (ggml_is_quantized(कुंजी.k_type) || ggml_is_quantized(कुंजी.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        अगर (ggml_is_quantized(कुंजी.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        अगर (ggml_is_quantized(कुंजी.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    वापस defines;
}

काष्ठा ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी {
    uint32_t  शीर्ष_dim_v;
    uint32_t  wg_आकार;
    ggml_type dst_type;
};

काष्ठा ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.शीर्ष_dim_v);
        ggml_webgpu_hash_combine(seed, कुंजी.wg_आकार);
        ggml_webgpu_hash_combine(seed, कुंजी.dst_type);
        वापस seed;
    }
};

अंतरभूत bool चालक==(स्थिर ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी & lhs,
                       स्थिर ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी & rhs) {
    वापस lhs.शीर्ष_dim_v == rhs.शीर्ष_dim_v && lhs.wg_आकार == rhs.wg_आकार && lhs.dst_type == rhs.dst_type;
}

काष्ठा ggml_webgpu_flash_attn_blk_pipeline_कुंजी {
    uint32_t kv_tile;

    bool चालक==(स्थिर ggml_webgpu_flash_attn_blk_pipeline_कुंजी & other) स्थिर { वापस kv_tile == other.kv_tile; }
};

काष्ठा ggml_webgpu_flash_attn_blk_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_flash_attn_blk_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.kv_tile);
        वापस seed;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
अंतरभूत माप_प्रकार ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t शीर्ष_dim_qk,
                                                  uint32_t शीर्ष_dim_v,
                                                  bool     has_mask,
                                                  bool     kv_direct) {
    स्थिर uint32_t max_शीर्ष_dim = std::max(शीर्ष_dim_qk, शीर्ष_dim_v);
    माप_प्रकार         f16_elems    = 0;
    माप_प्रकार         f32_elems    = 0;

    f32_elems += q_tile * शीर्ष_dim_qk;        // q_shmem
    अगर (!kv_direct) {
        f32_elems += kv_tile * max_शीर्ष_dim;  // kv_shmem
    }
    f32_elems += q_tile * शीर्ष_dim_v;         // o_shmem
    अगर (has_mask) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    वापस f16_elems * GGML_WEBGPU_F16_आकार_BYTES + f32_elems * GGML_WEBGPU_F32_आकार_BYTES;
}

अंतरभूत uint32_t ggml_webgpu_flash_attn_max_kv_tile(माप_प्रकार   limit_bytes,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t शीर्ष_dim_qk,
                                                   uint32_t शीर्ष_dim_v,
                                                   bool     has_mask,
                                                   bool     kv_direct) {
    स्थिर माप_प्रकार base_q_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 0, शीर्ष_dim_qk, शीर्ष_dim_v, has_mask, kv_direct);
    अगर (limit_bytes <= base_q_bytes) {
        वापस 0;
    }
    स्थिर माप_प्रकार one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, शीर्ष_dim_qk, शीर्ष_dim_v, has_mask, kv_direct);
    स्थिर माप_प्रकार bytes_per_kv = one_kv_bytes - base_q_bytes;
    अगर (bytes_per_kv == 0) {
        वापस 0;
    }
    स्थिर माप_प्रकार max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    वापस (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

अंतरभूत uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(माप_प्रकार   wg_mem_limit_bytes,
                                                       uint32_t शीर्ष_dim_qk,
                                                       uint32_t शीर्ष_dim_v,
                                                       bool     has_mask,
                                                       bool     kv_direct) {
    स्थिर uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_limit_bytes, 1u, 1u, शीर्ष_dim_qk, शीर्ष_dim_v, has_mask, kv_direct);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    अगर (kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        जबतक (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    वापस kv_tile;
}

अंतरभूत bool ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(bool                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                स्थिर ggml_tensor * Q,
                                                                स्थिर ggml_tensor * V) {
    वापस supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

काष्ठा ggml_webgpu_mul_mat_vec_pipeline_कुंजी {
    ggml_type src0_type;
    ggml_type src1_type;
    पूर्णांक       vectorized;
    uint32_t  अंक_cols;
    bool      use_mmvq;
    bool      src_overlap;

    bool चालक==(स्थिर ggml_webgpu_mul_mat_vec_pipeline_कुंजी & other) स्थिर {
        वापस src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               अंक_cols == other.अंक_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

काष्ठा ggml_webgpu_mul_mat_vec_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_mul_mat_vec_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src0_type);
        ggml_webgpu_hash_combine(seed, कुंजी.src1_type);
        ggml_webgpu_hash_combine(seed, कुंजी.vectorized);
        ggml_webgpu_hash_combine(seed, कुंजी.अंक_cols);
        ggml_webgpu_hash_combine(seed, कुंजी.use_mmvq);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        वापस seed;
    }
};

काष्ठा ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_आकार;
    uint32_t outputs_per_wg;
    uint32_t vec_आकार;
    bool     src_overlap = false;
};

काष्ठा ggml_webgpu_quantize_q8_pipeline_कुंजी {
    ggml_type src0_type;

    bool चालक==(स्थिर ggml_webgpu_quantize_q8_pipeline_कुंजी & other) स्थिर { वापस src0_type == other.src0_type; }
};

काष्ठा ggml_webgpu_quantize_q8_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_quantize_q8_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src0_type);
        वापस seed;
    }
};

काष्ठा ggml_webgpu_mul_mat_pipeline_कुंजी {
    ggml_type src0_type;
    ggml_type src1_type;
    पूर्णांक       vectorized;
    पूर्णांक       use_subgroup_matrix;
    bool      src_overlap;

    bool चालक==(स्थिर ggml_webgpu_mul_mat_pipeline_कुंजी & other) स्थिर {
        वापस src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

काष्ठा ggml_webgpu_mul_mat_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_mul_mat_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src0_type);
        ggml_webgpu_hash_combine(seed, कुंजी.src1_type);
        ggml_webgpu_hash_combine(seed, कुंजी.vectorized);
        ggml_webgpu_hash_combine(seed, कुंजी.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        वापस seed;
    }
};

काष्ठा ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_आकार_m;
    uint32_t wg_आकार_n;
    uint32_t wg_आकार;
    uint32_t outputs_per_wg;
    पूर्णांक      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_आकार;
    bool     src_overlap = false;
};

/** MUL_MAT_ID **/

काष्ठा ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी {
    uint32_t n_experts;

    bool चालक==(स्थिर ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी & other) स्थिर {
        वापस n_experts == other.n_experts;
    }
};

काष्ठा ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.n_experts);
        वापस seed;
    }
};

काष्ठा ggml_webgpu_mul_mat_id_pipeline_कुंजी {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  अंक_cols;
    पूर्णांक       vectorized;

    bool चालक==(स्थिर ggml_webgpu_mul_mat_id_pipeline_कुंजी & other) स्थिर {
        वापस src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               अंक_cols == other.अंक_cols && vectorized == other.vectorized;
    }
};

काष्ठा ggml_webgpu_mul_mat_id_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_mul_mat_id_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src0_type);
        ggml_webgpu_hash_combine(seed, कुंजी.src1_type);
        ggml_webgpu_hash_combine(seed, कुंजी.n_experts);
        ggml_webgpu_hash_combine(seed, कुंजी.अंक_cols);
        ggml_webgpu_hash_combine(seed, कुंजी.vectorized);
        वापस seed;
    }
};

/** Cpy **/

काष्ठा ggml_webgpu_cpy_pipeline_कुंजी {
    ggml_type src_type;
    ggml_type dst_type;

    bool चालक==(स्थिर ggml_webgpu_cpy_pipeline_कुंजी & other) स्थिर {
        वापस src_type == other.src_type && dst_type == other.dst_type;
    }
};

काष्ठा ggml_webgpu_cpy_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_cpy_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.src_type);
        ggml_webgpu_hash_combine(seed, कुंजी.dst_type);
        वापस seed;
    }
};

/** Glu **/

काष्ठा ggml_webgpu_glu_pipeline_कुंजी {
    ggml_glu_op glu_op;
    ggml_type   type;
    bool        split;
    bool        src_overlap;

    bool चालक==(स्थिर ggml_webgpu_glu_pipeline_कुंजी & other) स्थिर {
        वापस glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

काष्ठा ggml_webgpu_glu_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_glu_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.glu_op);
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.split);
        ggml_webgpu_hash_combine(seed, कुंजी.src_overlap);
        वापस seed;
    }
};

/** Rope **/

काष्ठा ggml_webgpu_rope_pipeline_कुंजी {
    ggml_type type;
    bool      inplace;
    bool      has_ff;

    bool चालक==(स्थिर ggml_webgpu_rope_pipeline_कुंजी & other) स्थिर {
        वापस type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

काष्ठा ggml_webgpu_rope_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_rope_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.type);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        ggml_webgpu_hash_combine(seed, कुंजी.has_ff);
        वापस seed;
    }
};

/** SoftMax **/

काष्ठा ggml_webgpu_soft_max_pipeline_कुंजी {
    ggml_type mask_type;
    bool      has_mask;
    bool      has_sink;
    bool      inplace;

    bool चालक==(स्थिर ggml_webgpu_soft_max_pipeline_कुंजी & other) स्थिर {
        वापस mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

काष्ठा ggml_webgpu_soft_max_pipeline_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर ggml_webgpu_soft_max_pipeline_कुंजी & कुंजी) स्थिर {
        माप_प्रकार seed = 0;
        ggml_webgpu_hash_combine(seed, कुंजी.mask_type);
        ggml_webgpu_hash_combine(seed, कुंजी.has_mask);
        ggml_webgpu_hash_combine(seed, कुंजी.has_sink);
        ggml_webgpu_hash_combine(seed, कुंजी.inplace);
        वापस seed;
    }
};

/** MMVQ **/

अंतरभूत bool ggml_webgpu_can_use_mmvq(स्थिर ggml_tensor * src0,
                                     स्थिर ggml_tensor * src1,
                                     bool                supports_dot_product,
                                     स्थिर std::string & vendor) {
    अगर (src1->ne[1] <= 4) {
        bool supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        अगर (supports_dp4a && supports_dot_product) {
            चयन (src1->type) {
                हाल GGML_TYPE_F32:
                    चयन (src0->type) {
                        हाल GGML_TYPE_Q1_0:
                        हाल GGML_TYPE_Q4_0:
                        हाल GGML_TYPE_Q4_1:
                        हाल GGML_TYPE_Q5_0:
                        हाल GGML_TYPE_Q5_1:
                        हाल GGML_TYPE_Q8_0:
                        हाल GGML_TYPE_MXFP4:
                        हाल GGML_TYPE_Q2_K:
                        हाल GGML_TYPE_Q3_K:
                        हाल GGML_TYPE_Q4_K:
                        हाल GGML_TYPE_Q5_K:
                        हाल GGML_TYPE_Q6_K:
                            वापस src0->ne[0] % 4 == 0;
                        शेष:
                            अवरोध;
                    }
                    अवरोध;
                शेष:
                    अवरोध;
            }
        }
    }
    वापस false;
}

श्रेणी ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<पूर्णांक, webgpu_pipeline> योग_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<पूर्णांक, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<पूर्णांक, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<पूर्णांक, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<पूर्णांक, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
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

  खुला:
    ggml_webgpu_shader_lib(wgpu::Device device) { यह->device = device; }

    webgpu_pipeline get_योग_rows_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        स्वतः it = योग_rows_pipelines.find(1);
        अगर (it != योग_rows_pipelines.end()) {
            वापस it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed        = preprocessor.preprocess(wgsl_योग_rows, defines);
        योग_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        वापस योग_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_कुंजी कुंजी = {};
        कुंजी.op                                = context.dst->op;
        कुंजी.src_type                          = context.src0->type;
        कुंजी.dst_type                          = context.dst->type;
        कुंजी.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = row_norm_pipelines.find(कुंजी);
        अगर (it != row_norm_pipelines.end()) {
            वापस it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        चयन (कुंजी.op) {
            हाल GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                अवरोध;
            हाल GGML_OP_NORM:
                defines.push_back("NORM");
                variant = "norm";
                अवरोध;
            हाल GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        अगर (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        अगर (कुंजी.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } अन्यथा अगर (कुंजी.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        अगर (कुंजी.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } अन्यथा अगर (कुंजी.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        स्थिर uint32_t row_norm_wg_आकार = 128u;
        uint32_t       wg_आकार          = std::min(context.max_wg_आकार, row_norm_wg_आकार);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        स्वतः processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        स्वतः decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार              = wg_आकार;
        decisions->inplace              = कुंजी.inplace;
        row_norm_pipelines[कुंजी]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[कुंजी].context = decisions;
        वापस row_norm_pipelines[कुंजी];
    }

    webgpu_pipeline get_argmax_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        bool vec4 = context.src0->ne[0] % 4 == 0;

        स्वतः it = argmax_pipelines.find(vec4);
        अगर (it != argmax_pipelines.end()) {
            वापस it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));
        अगर (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        स्वतः processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        वापस argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        स्थिर bool                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_कुंजी कुंजी       = {};
        कुंजी.src0_type                               = context.src0->type;
        कुंजी.dst_type                                = context.dst->type;
        कुंजी.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        कुंजी.i64_idx     = context.src1->type == GGML_TYPE_I64;
        कुंजी.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_आकार(context.dst->type)) % 2 == 0);

        स्वतः it = set_rows_pipelines.find(कुंजी);
        अगर (it != set_rows_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        चयन (context.dst->type) {
            हाल GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                अवरोध;
            हाल GGML_TYPE_Q8_0:
                defines.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                अवरोध;
            हाल GGML_TYPE_Q4_0:
                defines.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        अगर (context.src0->type == GGML_TYPE_F16) {
            defines.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        अगर (कुंजी.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        अगर (कुंजी.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        अगर (कुंजी.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्थिर स्वतः & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        स्वतः         processed          = preprocessor.preprocess(shader_source, defines);
        स्वतः         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = कुंजी.vec4;
        decisions->i64_idx              = कुंजी.i64_idx;
        decisions->pair_blocks          = कुंजी.pair_blocks;
        decisions->wg_आकार              = context.max_wg_आकार;
        set_rows_pipelines[कुंजी]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[कुंजी].context = decisions;
        वापस set_rows_pipelines[कुंजी];
    }

    webgpu_pipeline get_set_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_कुंजी कुंजी = {};
        कुंजी.type                         = context.dst->type;
        कुंजी.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = set_pipelines.find(कुंजी);
        अगर (it != set_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for set shader");
        }

        अगर (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_set, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->inplace       = कुंजी.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[कुंजी]       = pipeline;
        वापस set_pipelines[कुंजी];
    }

    webgpu_pipeline get_cumsum_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        स्वतः it = cumsum_pipelines.find(1);
        अगर (it != cumsum_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        वापस cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        bool          is_शीर्ष_k = context.dst->op == GGML_OP_शीर्ष_K;
        // ascending order is 0, descending order is 1
        स्थिर int32_t order =
            is_शीर्ष_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        स्वतः it = argsort_pipelines.find(order);
        अगर (it != argsort_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_आकार = 1;
        जबतक (wg_आकार * 2 <= context.max_wg_आकार &&
               wg_आकार * GGML_WEBGPU_I32_आकार_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_आकार *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        स्वतः processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        स्वतः decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार               = wg_आकार;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        वापस argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        bool          is_शीर्ष_k = context.dst->op == GGML_OP_शीर्ष_K;
        // ascending order is 0, descending order is 1
        स्थिर int32_t order =
            is_शीर्ष_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        स्वतः it = argsort_merge_pipelines.find(order);
        अगर (it != argsort_merge_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_आकार = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_आकार, context.max_wg_आकार);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));

        स्वतः processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        वापस argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, bool vec4_aligned) {
        स्थिर bool vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_कुंजी कुंजी = {};
        कुंजी.src_type                          = context.src0->type;
        कुंजी.vectorized                        = (पूर्णांक) vectorized;

        स्वतः it = get_rows_pipelines.find(कुंजी);
        अगर (it != get_rows_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        स्थिर काष्ठा ggml_type_traits * type_traits = ggml_get_type_traits(कुंजी.src_type);
        स्थिर अक्षर *                    type_str    = type_traits->type_name;

        चयन (कुंजी.src_type) {
            हाल GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                अगर (कुंजी.vectorized) {
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
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                अवरोध;
            हाल GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                अवरोध;
            हाल GGML_TYPE_BF16:
                defines.push_back("BF16");
                defines.push_back("SRC_TYPE=u32");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                अवरोध;
            शेष:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    चयन (कुंजी.src_type) {
                        हाल GGML_TYPE_Q1_0:
                        हाल GGML_TYPE_Q2_0:
                        हाल GGML_TYPE_Q4_0:
                        हाल GGML_TYPE_Q5_0:
                        हाल GGML_TYPE_Q8_0:
                        हाल GGML_TYPE_Q3_K:
                        हाल GGML_TYPE_Q6_K:
                        हाल GGML_TYPE_IQ2_XXS:
                        हाल GGML_TYPE_IQ2_XS:
                        हाल GGML_TYPE_IQ2_S:
                        हाल GGML_TYPE_IQ3_XXS:
                        हाल GGML_TYPE_IQ3_S:
                        हाल GGML_TYPE_IQ1_S:
                        हाल GGML_TYPE_IQ4_NL:
                        हाल GGML_TYPE_MXFP4:
                        हाल GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                अवरोध;
                            }
                        शेष:
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

                    अगर (कुंजी.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } अन्यथा अगर ((कुंजी.src_type >= GGML_TYPE_Q4_0 && कुंजी.src_type <= GGML_TYPE_Q8_1) ||
                               कुंजी.src_type == GGML_TYPE_IQ4_NL || कुंजी.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } अन्यथा अगर (कुंजी.src_type == GGML_TYPE_Q2_0 || कुंजी.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } अन्यथा अगर (कुंजी.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } अन्यथा {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    अवरोध;
                }
        }

        अगर (कुंजी.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[कुंजी]  = pipeline;
        वापस get_rows_pipelines[कुंजी];
    }

    webgpu_pipeline get_scale_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_कुंजी कुंजी = {};
        कुंजी.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = scale_pipelines.find(कुंजी);
        अगर (it != scale_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        अगर (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_scale, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->inplace       = कुंजी.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[कुंजी]     = pipeline;
        वापस scale_pipelines[कुंजी];
    }

    webgpu_pipeline get_solve_tri_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_कुंजी कुंजी = {};
        कुंजी.type                               = context.dst->type;
        कुंजी.n                                  = (पूर्णांक) context.src0->ne[0];
        कुंजी.k                                  = (पूर्णांक) context.src1->ne[0];

        स्वतः it = solve_tri_pipelines.find(कुंजी);
        अगर (it != solve_tri_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                variant += "_f32";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        स्थिर uint32_t wg_आकार       = std::min((uint32_t) कुंजी.n, context.max_wg_आकार);
        स्थिर uint32_t k_tile        = wg_आकार;
        स्थिर uint32_t bytes_per_row = ((uint32_t) कुंजी.n + wg_आकार) * GGML_WEBGPU_F32_आकार_BYTES;
        स्थिर uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(कुंजी.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        स्वतः processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[कुंजी] = pipeline;
        वापस solve_tri_pipelines[कुंजी];
    }

    webgpu_pipeline get_ssm_conv_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_कुंजी कुंजी = {};
        कुंजी.type                              = context.dst->type;
        कुंजी.vectorized                        = context.src1->ne[0] == 4;

        स्वतः it = ssm_conv_pipelines.find(कुंजी);
        अगर (it != ssm_conv_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                variant += "_f32";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        अगर (कुंजी.vectorized) {
            defines.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t block_आकार_2    = 32u;
        constexpr uint32_t tokens_per_wg = 8u;

        defines.push_back("BLOCK_SIZE=" + std::to_string(block_आकार_2) + "u");
        defines.push_back("TOKENS_PER_WG=" + std::to_string(tokens_per_wg) + "u");

        स्वतः processed           = preprocessor.preprocess(wgsl_ssm_conv, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->block_आकार_2    = block_आकार_2;
        decisions->tokens_per_wg = tokens_per_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[कुंजी]  = pipeline;
        वापस ssm_conv_pipelines[कुंजी];
    }

    webgpu_pipeline get_ssm_scan_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_कुंजी कुंजी = {};
        कुंजी.type                              = context.dst->type;
        कुंजी.d_state                           = (पूर्णांक) context.src0->ne[0];
        कुंजी.xbc_overlap                       = xbc_overlap;

        स्वतः it = ssm_scan_pipelines.find(कुंजी);
        अगर (it != ssm_scan_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                variant += "_f32";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        स्थिर uint32_t wg_आकार = (uint32_t) कुंजी.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_आकार) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        अगर (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } अन्यथा {
            variant += "_wg_reduce";
        }

        अगर (कुंजी.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } अन्यथा अगर (कुंजी.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } अन्यथा अगर (कुंजी.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(कुंजी.d_state);

        स्वतः processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        स्वतः decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_आकार         = wg_आकार;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = कुंजी.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[कुंजी]    = pipeline;
        वापस ssm_scan_pipelines[कुंजी];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_कुंजी कुंजी = {};
        कुंजी.type                                     = context.dst->type;
        कुंजी.s_v                                      = (पूर्णांक) context.src2->ne[0];
        कुंजी.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        कुंजी.fused_cache                              = context.dst_fuse != nullptr;

        स्वतः it = gated_delta_net_pipelines.find(कुंजी);
        अगर (it != gated_delta_net_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                variant += "_f32";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        अगर (कुंजी.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        अगर (कुंजी.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(कुंजी.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(कुंजी.s_v) + "u");

        स्वतः            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[कुंजी] = pipeline;
        वापस gated_delta_net_pipelines[कुंजी];
    }

    webgpu_pipeline get_pad_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_कुंजी कुंजी = {};
        कुंजी.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        स्वतः it = pad_pipelines.find(कुंजी);
        अगर (it != pad_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        अगर (कुंजी.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_pad, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[कुंजी]       = pipeline;
        वापस pad_pipelines[कुंजी];
    }

    webgpu_pipeline get_quantize_q8_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_कुंजी कुंजी = {};
        कुंजी.src0_type                            = context.src0->type;

        स्वतः it = quantize_q8_pipelines.find(कुंजी);
        अगर (it != quantize_q8_pipelines.end()) {
            वापस it->second;
        }
        स्थिर अक्षर *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> defines;
        std::string              variant = "quantize_q8";

        uint32_t wg_आकार = WEBGPU_MUL_MAT_VEC_WG_आकार;

        defines.push_back("SRC1_INNER_TYPE=f32");
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));

        स्थिर काष्ठा ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
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
        decisions->wg_आकार         = wg_आकार;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[कुंजी] = pipeline;
        वापस quantize_q8_pipelines[कुंजी];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, bool src_overlap) {
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
        अगर (it != mul_mat_vec_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        स्थिर अक्षर *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        चयन (context.src0->type) {
            हाल GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                अवरोध;
            हाल GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                अवरोध;
            शेष:
                {
                    // Quantized types: use helpers but accumulate in f16
                    स्थिर काष्ठा ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    चयन (context.src0->type) {
                        हाल GGML_TYPE_Q4_0:
                        हाल GGML_TYPE_Q4_1:
                        हाल GGML_TYPE_Q5_0:
                        हाल GGML_TYPE_Q5_1:
                        हाल GGML_TYPE_Q8_0:
                            अगर (कुंजी.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            अवरोध;
                        हाल GGML_TYPE_Q1_0:
                        हाल GGML_TYPE_Q2_K:
                        हाल GGML_TYPE_Q3_K:
                        हाल GGML_TYPE_Q4_K:
                        हाल GGML_TYPE_Q5_K:
                        हाल GGML_TYPE_Q6_K:
                            अगर (कुंजी.use_mmvq) {
                                defines.push_back("K_QUANTS_HANDLING");
                            }
                            अवरोध;
                        हाल GGML_TYPE_IQ1_S:
                        हाल GGML_TYPE_IQ1_M:
                        हाल GGML_TYPE_IQ2_S:
                        हाल GGML_TYPE_IQ3_S:
                        हाल GGML_TYPE_IQ4_NL:
                        हाल GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            अवरोध;
                        हाल GGML_TYPE_IQ2_XXS:
                        हाल GGML_TYPE_IQ2_XS:
                        हाल GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            अवरोध;
                        हाल GGML_TYPE_MXFP4:
                            defines.push_back(type_upper + "_LUT");
                            अगर (कुंजी.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            अवरोध;
                        हाल GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            अवरोध;
                        शेष:
                            अवरोध;
                    }
                    अवरोध;
                }
        }

        // src1 type (vector)
        चयन (context.src1->type) {
            हाल GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(कुंजी.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_आकार        = WEBGPU_MUL_MAT_VEC_WG_आकार;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        अगर (कुंजी.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } अन्यथा अगर (कुंजी.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } अन्यथा अगर (कुंजी.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        अगर (कुंजी.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        अगर (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        अगर (कुंजी.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(कुंजी.अंक_cols));

        स्वतः processed            = preprocessor.preprocess(shader_src, defines);
        स्वतः decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_आकार        = wg_आकार;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_आकार       = कुंजी.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[कुंजी] = pipeline;
        वापस mul_mat_vec_pipelines[कुंजी];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, bool src_overlap) {
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
        अगर (it != mul_mat_fast_pipelines.end()) {
            वापस it->second;
        }

        स्थिर अक्षर * shader_src = कुंजी.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = कुंजी.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        चयन (context.src1->type) {
            हाल GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        स्थिर काष्ठा ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        स्थिर अक्षर *                    src0_name   = src0_traits->type_name;

        चयन (context.src0->type) {
            हाल GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("TYPE_F32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("TYPE_F16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                अवरोध;
            हाल GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("TYPE_BF16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                अवरोध;
            शेष:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    चयन (context.src0->type) {
                        हाल GGML_TYPE_IQ1_S:
                        हाल GGML_TYPE_IQ1_M:
                        हाल GGML_TYPE_IQ4_NL:
                        हाल GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            अवरोध;
                        हाल GGML_TYPE_IQ2_XXS:
                        हाल GGML_TYPE_IQ2_XS:
                        हाल GGML_TYPE_IQ2_S:
                        हाल GGML_TYPE_IQ3_XXS:
                        हाल GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            अवरोध;
                        हाल GGML_TYPE_MXFP4:
                        हाल GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            अवरोध;
                        शेष:
                            अवरोध;
                    }

                    variant += std::string("_") + src0_name;
                    अवरोध;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(कुंजी.vectorized ? "VEC" : "SCALAR");

        स्थिर bool is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        अगर (कुंजी.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } अन्यथा {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        अगर (कुंजी.use_subgroup_matrix) {
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_आकार) + "u");
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
        अगर (कुंजी.vectorized) {
            variant += "_vectorized";
        }

        अगर (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        अगर (!कुंजी.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_आकार_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_आकार_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        स्वतः processed = preprocessor.preprocess(shader_src, defines);

        स्वतः decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = कुंजी.use_subgroup_matrix;
        अगर (कुंजी.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_आकार           = context.max_subgroup_आकार;
        } अन्यथा {
            decisions->wg_आकार_m       = WEBGPU_MUL_MAT_WG_आकार_M;
            decisions->wg_आकार_n       = WEBGPU_MUL_MAT_WG_आकार_N;
            decisions->wg_आकार         = WEBGPU_MUL_MAT_WG_आकार_M * WEBGPU_MUL_MAT_WG_आकार_N;
            decisions->mul_mat_wg_आकार = WEBGPU_MUL_MAT_WG_आकार;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[कुंजी] = pipeline;
        वापस mul_mat_fast_pipelines[कुंजी];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_कुंजी कुंजी = {};
        कुंजी.n_experts                                  = (uint32_t) context.src0->ne[2];

        स्वतः it = mul_mat_id_gather_pipelines.find(कुंजी);
        अगर (it != mul_mat_id_gather_pipelines.end()) {
            वापस it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(कुंजी.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_आकार_N) + "u");

        स्वतः processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        स्वतः decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार = context.max_wg_आकार;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[कुंजी] = pipeline;
        वापस pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_कुंजी कुंजी = {};
        कुंजी.src0_type                           = context.src0->type;
        कुंजी.src1_type                           = context.src1->type;
        कुंजी.n_experts                           = context.src0->ne[2];
        कुंजी.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        स्वतः it = mul_mat_id_pipelines.find(कुंजी);
        अगर (it != mul_mat_id_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        चयन (context.src1->type) {
            हाल GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        स्थिर काष्ठा ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        स्थिर अक्षर *                    src0_name   = src0_traits->type_name;

        चयन (context.src0->type) {
            हाल GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                अवरोध;
            हाल GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                अवरोध;
            शेष:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    चयन (context.src0->type) {
                        हाल GGML_TYPE_IQ1_S:
                        हाल GGML_TYPE_IQ1_M:
                        हाल GGML_TYPE_IQ4_NL:
                        हाल GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            अवरोध;
                        हाल GGML_TYPE_IQ2_XXS:
                        हाल GGML_TYPE_IQ2_XS:
                        हाल GGML_TYPE_IQ2_S:
                        हाल GGML_TYPE_IQ3_XXS:
                        हाल GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            अवरोध;
                        हाल GGML_TYPE_MXFP4:
                        हाल GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            अवरोध;
                        शेष:
                            अवरोध;
                    }

                    variant += std::string("_") + src0_name;
                    अवरोध;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(कुंजी.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        स्थिर uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_आकार_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_आकार_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        अगर (कुंजी.vectorized) {
            variant += "_vectorized";
        }

        स्वतः processed = preprocessor.preprocess(wgsl_mul_mat_id, defines);

        स्वतः decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_आकार_m = WEBGPU_MUL_MAT_WG_आकार_M;
        decisions->wg_आकार_n = WEBGPU_MUL_MAT_WG_आकार_N;
        decisions->wg_आकार   = WEBGPU_MUL_MAT_WG_आकार_M * WEBGPU_MUL_MAT_WG_आकार_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[कुंजी] = pipeline;
        वापस mul_mat_id_pipelines[कुंजी];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_कुंजी कुंजी = {};
        कुंजी.src0_type                           = context.src0->type;
        कुंजी.src1_type                           = context.src1->type;
        कुंजी.n_experts                           = context.src0->ne[2];
        कुंजी.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        स्वतः it = mul_mat_id_vec_pipelines.find(कुंजी);
        अगर (it != mul_mat_id_vec_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        स्थिर अक्षर *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        चयन (context.src1->type) {
            हाल GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        चयन (context.src0->type) {
            हाल GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                अवरोध;
            हाल GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                अवरोध;
            शेष:
                {
                    // Quantized types: use helpers but accumulate in f16
                    स्थिर काष्ठा ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::बड़े);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    चयन (context.src0->type) {
                        हाल GGML_TYPE_IQ1_S:
                        हाल GGML_TYPE_IQ1_M:
                        हाल GGML_TYPE_IQ2_S:
                        हाल GGML_TYPE_IQ3_S:
                        हाल GGML_TYPE_IQ4_NL:
                        हाल GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            अवरोध;
                        हाल GGML_TYPE_IQ2_XXS:
                        हाल GGML_TYPE_IQ2_XS:
                        हाल GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            अवरोध;
                        हाल GGML_TYPE_MXFP4:
                        हाल GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            अवरोध;
                        शेष:
                            अवरोध;
                    }
                    अवरोध;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(कुंजी.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_आकार        = WEBGPU_MUL_MAT_VEC_WG_आकार;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        अगर (कुंजी.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } अन्यथा अगर (कुंजी.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } अन्यथा अगर (कुंजी.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        अगर (कुंजी.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(कुंजी.n_experts));

        स्वतः processed = preprocessor.preprocess(shader_src, defines);

        स्वतः decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_आकार        = wg_आकार;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[कुंजी] = pipeline;
        वापस mul_mat_id_vec_pipelines[कुंजी];
    }

    webgpu_pipeline get_unary_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        स्थिर bool                     is_unary = context.dst->op == GGML_OP_UNARY;
        स्थिर पूर्णांक                      op       = is_unary ? (पूर्णांक) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_कुंजी कुंजी      = {};
        कुंजी.type                                = context.dst->type;
        कुंजी.op                                  = op;
        कुंजी.is_unary                            = is_unary;
        कुंजी.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        कुंजी.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        स्वतः it = unary_pipelines.find(कुंजी);
        अगर (it != unary_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            कुंजी.is_unary ? ggml_unary_op_name((ggml_unary_op) कुंजी.op) : ggml_op_name((ggml_op) कुंजी.op);
        defines.push_back(variant);

        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for unary shader");
        }

        अगर (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        अगर (op == GGML_OP_TRI) {
            चयन (कुंजी.ttype) {
                हाल GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    अवरोध;
                हाल GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    अवरोध;
                हाल GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    अवरोध;
                हाल GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    अवरोध;
                शेष:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_unary, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->inplace       = कुंजी.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[कुंजी]     = pipeline;
        वापस unary_pipelines[कुंजी];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, bool src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_कुंजी कुंजी = {};
        कुंजी.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        कुंजी.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        कुंजी.src_overlap                           = src_overlap;

        स्वतः it = rms_norm_mul_pipelines.find(कुंजी);
        अगर (it != rms_norm_mul_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        अगर (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } अन्यथा अगर (कुंजी.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } अन्यथा अगर (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        स्वतः pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_आकार     = context.max_wg_आकार;
        pipeline_decisions->inplace     = कुंजी.inplace;
        pipeline_decisions->overlap     = कुंजी.overlap;
        pipeline_decisions->src_overlap = कुंजी.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[कुंजी]     = pipeline;
        वापस rms_norm_mul_pipelines[कुंजी];
    }

    webgpu_pipeline get_द्विआधारी_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, bool src_overlap) {
        ggml_webgpu_द्विआधारी_pipeline_कुंजी कुंजी = {};
        कुंजी.type                            = context.dst->type;
        कुंजी.op                              = context.dst->op;
        कुंजी.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        कुंजी.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        कुंजी.src_overlap                     = src_overlap;

        स्वतः it = द्विआधारी_pipelines.find(कुंजी);
        अगर (it != द्विआधारी_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) कुंजी.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for binary shader");
        }

        अगर (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } अन्यथा अगर (कुंजी.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } अन्यथा अगर (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed                  = preprocessor.preprocess(wgsl_द्विआधारी, defines);
        स्वतः pipeline_decisions         = std::make_shared<ggml_webgpu_द्विआधारी_shader_decisions>();
        pipeline_decisions->wg_आकार     = context.max_wg_आकार;
        pipeline_decisions->inplace     = कुंजी.inplace;
        pipeline_decisions->overlap     = कुंजी.overlap;
        pipeline_decisions->src_overlap = कुंजी.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        द्विआधारी_pipelines[कुंजी]    = pipeline;
        वापस द्विआधारी_pipelines[कुंजी];
    }

    webgpu_pipeline get_add_id_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_कुंजी कुंजी = {};
        कुंजी.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = add_id_pipelines.find(कुंजी);
        अगर (it != add_id_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        स्थिर अक्षर *             shader_src = wgsl_add_id;

        अगर (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed              = preprocessor.preprocess(shader_src, defines);
        स्वतः pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_आकार = context.max_wg_आकार;
        pipeline_decisions->inplace = कुंजी.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[कुंजी]    = pipeline;
        वापस pipeline;
    }

    webgpu_pipeline get_concat_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, bool src_overlap) {
        ggml_webgpu_concat_pipeline_कुंजी कुंजी = {};
        कुंजी.type                            = context.dst->type;
        कुंजी.src_overlap                     = src_overlap;

        स्वतः it = concat_pipelines.find(कुंजी);
        अगर (it != concat_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for concat shader");
        }

        अगर (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_concat, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_द्विआधारी_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->src_overlap   = कुंजी.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[कुंजी]    = pipeline;
        वापस concat_pipelines[कुंजी];
    }

    webgpu_pipeline get_repeat_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_कुंजी कुंजी = {};
        कुंजी.type                            = context.dst->type;

        स्वतः it = repeat_pipelines.find(कुंजी);
        अगर (it != repeat_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                अवरोध;
            हाल GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                अवरोध;
            हाल GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_repeat, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[कुंजी]    = pipeline;
        वापस repeat_pipelines[कुंजी];
    }

    webgpu_pipeline get_flash_attn_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, bool kv_overlap) {
        स्थिर bool can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
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
            कुंजी.common.शीर्ष_dim_qk, कुंजी.common.शीर्ष_dim_v, कुंजी.common.has_mask,
            कुंजी.common.k_direct || कुंजी.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        अगर (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = false;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            कुंजी.common.k_direct     = false;
            कुंजी.common.v_direct     = false;
            कुंजी.use_sg_matrix       = false;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, कुंजी.common.शीर्ष_dim_qk, कुंजी.common.शीर्ष_dim_v,
                कुंजी.common.has_mask, कुंजी.common.k_direct || कुंजी.common.v_direct);
            GGML_ASSERT(max_kv_tile > 0);
        }

        decisions.kv_tile = decisions.use_sg_matrix ?
                                std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES) :
                                std::min(GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE, max_kv_tile);
        decisions.wg_आकार =
            decisions.use_sg_matrix ?
                std::max(context.max_subgroup_आकार, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_आकार) :
                std::min(context.max_wg_आकार, std::max(GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_आकार,
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * context.max_subgroup_आकार));

        अगर (कुंजी.common.k_direct || कुंजी.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            जबतक (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_आकार;
            }
        }

        स्वतः it = flash_attn_pipelines.find(कुंजी);
        अगर (it != flash_attn_pipelines.end()) {
            वापस it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(कुंजी.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_आकार);
        स्थिर अक्षर *             shader_src = nullptr;
        अगर (!कुंजी.use_sg_matrix) {
            shader_src = wgsl_flash_attn_tile;
            defines.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_आकार) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_आकार) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_आकार) + "_" +
                       std::to_string(context.max_subgroup_आकार);
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
        वापस flash_attn_pipelines[कुंजी];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, bool kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_कुंजी कुंजी = {};
        कुंजी.common = ggml_webgpu_flash_attn_make_common_pipeline_कुंजी(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        स्वतः it = flash_attn_vec_pipelines.find(कुंजी);
        अगर (it != flash_attn_vec_pipelines.end()) {
            वापस it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, कुंजी.common.शीर्ष_dim_qk,
                                                                   कुंजी.common.शीर्ष_dim_v, कुंजी.common.has_mask,
                                                                   कुंजी.common.k_direct || कुंजी.common.v_direct);
        decisions.wg_आकार = context.max_subgroup_आकार;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(कुंजी.common, variant, 1u, decisions.kv_tile, decisions.wg_आकार);
        अगर (कुंजी.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (माप("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_आकार;
        स्थिर uint32_t D       = कुंजी.common.शीर्ष_dim_qk | कुंजी.common.शीर्ष_dim_v;
        स्थिर uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_आकार, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        स्वतः            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[कुंजी] = pipeline;
        वापस flash_attn_vec_pipelines[कुंजी];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_कुंजी कुंजी = {};
        कुंजी.kv_tile                                 = kv_tile;
        स्वतः it                                     = flash_attn_blk_pipelines.find(कुंजी);
        अगर (it != flash_attn_blk_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(कुंजी.kv_tile));
        variant += std::string("_kvt") + std::to_string(कुंजी.kv_tile);

        uint32_t wg_आकार = 1;
        जबतक ((wg_आकार << 1) <= context.max_wg_आकार) {
            wg_आकार <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        variant += std::string("_wg") + std::to_string(wg_आकार);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[कुंजी] = pipeline;
        वापस flash_attn_blk_pipelines[कुंजी];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_कुंजी कुंजी = {};
        कुंजी.शीर्ष_dim_v                                     = (uint32_t) context.src2->ne[0];
        कुंजी.dst_type                                       = context.dst->type;
        कुंजी.wg_आकार                                        = context.max_wg_आकार;
        स्वतः it                                            = flash_attn_vec_reduce_pipelines.find(कुंजी);
        अगर (it != flash_attn_vec_reduce_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        चयन (कुंजी.dst_type) {
            हाल GGML_TYPE_F32:
                defines.push_back("DST_F32");
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("DST_F16");
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(कुंजी.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(कुंजी.शीर्ष_dim_v));
        variant += std::string("_hsv") + std::to_string(कुंजी.शीर्ष_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));
        variant += std::string("_wg") + std::to_string(context.max_wg_आकार);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[कुंजी] = pipeline;
        वापस flash_attn_vec_reduce_pipelines[कुंजी];
    }

    webgpu_pipeline get_cpy_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_कुंजी कुंजी = {};
        कुंजी.src_type                     = context.src0->type;
        कुंजी.dst_type                     = context.dst->type;

        स्वतः it = cpy_pipelines.find(कुंजी);
        अगर (it != cpy_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        चयन (कुंजी.src_type) {
            हाल GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                अवरोध;
            हाल GGML_TYPE_I32:
                defines.push_back("SRC_I32");
                variant += "_i32";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        चयन (कुंजी.dst_type) {
            हाल GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                अवरोध;
            हाल GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_cpy, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[कुंजी]       = pipeline;
        वापस cpy_pipelines[कुंजी];
    }

    webgpu_pipeline get_glu_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, bool src_overlap) {
        ggml_webgpu_glu_pipeline_कुंजी कुंजी = {};
        कुंजी.glu_op                       = ggml_get_glu_op(context.dst);
        कुंजी.type                         = context.dst->type;
        कुंजी.split                        = (context.src1 != nullptr);
        कुंजी.src_overlap                  = src_overlap;

        स्वतः it = glu_pipelines.find(कुंजी);
        अगर (it != glu_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        चयन (कुंजी.glu_op) {
            हाल GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                अवरोध;
            हाल GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                अवरोध;
            हाल GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                अवरोध;
            हाल GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                अवरोध;
            हाल GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                अवरोध;
            हाल GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                अवरोध;
            हाल GGML_GLU_OP_SWIGLU_CLAMP:
                defines.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported GLU op");
        }
        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        अगर (कुंजी.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } अन्यथा अगर (कुंजी.split) {
            variant += "_split";
        } अन्यथा {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_glu, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->src_overlap   = कुंजी.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[कुंजी]       = pipeline;
        वापस glu_pipelines[कुंजी];
    }

    webgpu_pipeline get_rope_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_कुंजी कुंजी = {};
        कुंजी.type                          = context.dst->type;
        कुंजी.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        कुंजी.has_ff                        = (context.src2 != nullptr);

        स्वतः it = rope_pipelines.find(कुंजी);
        अगर (it != rope_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        चयन (कुंजी.type) {
            हाल GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                अवरोध;
            हाल GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        अगर (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        अगर (कुंजी.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_rope, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->inplace       = कुंजी.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[कुंजी]      = pipeline;
        वापस rope_pipelines[कुंजी];
    }

    webgpu_pipeline get_soft_max_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_कुंजी कुंजी = {};
        कुंजी.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        कुंजी.has_mask                          = (context.src1 != nullptr);
        कुंजी.has_sink                          = (context.src2 != nullptr);
        कुंजी.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वतः it = soft_max_pipelines.find(कुंजी);
        अगर (it != soft_max_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        अगर (कुंजी.has_mask) {
            defines.push_back("HAS_MASK");
            चयन (कुंजी.mask_type) {
                हाल GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    अवरोध;
                हाल GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    अवरोध;
                शेष:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        अगर (कुंजी.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        अगर (कुंजी.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->inplace       = कुंजी.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[कुंजी]  = pipeline;
        वापस soft_max_pipelines[कुंजी];
    }

    webgpu_pipeline get_conv2d_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_कुंजी कुंजी = {};
        कुंजी.weight_type                     = context.src0->type;
        कुंजी.निवेश_type                      = context.src1->type;
        कुंजी.output_type                     = context.dst->type;

        स्वतः it = conv2d_pipelines.find(कुंजी);
        अगर (it != conv2d_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        स्वतः धकेलो_type_defines = [&](स्थिर अक्षर * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            अगर (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } अन्यथा अगर (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        धकेलो_type_defines("WEIGHT_TYPE", कुंजी.weight_type);
        धकेलो_type_defines("INPUT_TYPE", कुंजी.निवेश_type);
        धकेलो_type_defines("OUTPUT_TYPE", कुंजी.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[कुंजी]    = pipeline;
        वापस conv2d_pipelines[कुंजी];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(स्थिर ggml_webgpu_shader_lib_context & context, bool whcn) {
        ggml_webgpu_conv2d_dw_pipeline_कुंजी कुंजी = {};
        कुंजी.weight_type                        = context.src0->type;
        कुंजी.निवेश_type                         = context.src1->type;
        कुंजी.output_type                        = context.dst->type;
        कुंजी.whcn                               = whcn;

        स्वतः it = conv2d_dw_pipelines.find(कुंजी);
        अगर (it != conv2d_dw_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        स्वतः धकेलो_type_defines = [&](स्थिर अक्षर * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            अगर (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } अन्यथा अगर (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        धकेलो_type_defines("WEIGHT_TYPE", कुंजी.weight_type);
        धकेलो_type_defines("INPUT_TYPE", कुंजी.निवेश_type);
        धकेलो_type_defines("OUTPUT_TYPE", कुंजी.output_type);

        अगर (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[कुंजी] = pipeline;
        वापस conv2d_dw_pipelines[कुंजी];
    }

    webgpu_pipeline get_im2col_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_कुंजी कुंजी = {};
        कुंजी.निवेश_type                      = context.src1->type;
        कुंजी.output_type                     = context.dst->type;

        स्वतः it = im2col_pipelines.find(कुंजी);
        अगर (it != im2col_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        स्वतः धकेलो_type_defines = [&](स्थिर अक्षर * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            अगर (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } अन्यथा अगर (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        धकेलो_type_defines("INPUT_TYPE", कुंजी.निवेश_type);
        धकेलो_type_defines("OUTPUT_TYPE", कुंजी.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_im2col, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[कुंजी]    = pipeline;
        वापस im2col_pipelines[कुंजी];
    }

    webgpu_pipeline get_upscale_pipeline(स्थिर ggml_webgpu_shader_lib_context & context) {
        स्थिर uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        स्थिर uint32_t base_mode  = mode_flags & 0xFFu;
        स्थिर bool     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_कुंजी कुंजी = {};
        कुंजी.निवेश_type                       = context.src0->type;
        कुंजी.output_type                      = context.dst->type;
        कुंजी.base_mode                        = base_mode;
        कुंजी.antialias                        = antialias;

        स्वतः it = upscale_pipelines.find(कुंजी);
        अगर (it != upscale_pipelines.end()) {
            वापस it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        अगर (कुंजी.निवेश_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } अन्यथा {
            variant += "_src_f32";
        }

        अगर (कुंजी.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } अन्यथा {
            variant += "_dst_f32";
        }

        चयन (base_mode) {
            हाल GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                अवरोध;
            हाल GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                अवरोध;
            हाल GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                अवरोध;
            शेष:
                GGML_ABORT("Unsupported upscale mode");
        }

        अगर (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वतः processed           = preprocessor.preprocess(wgsl_upscale, defines);
        स्वतः decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[कुंजी]   = pipeline;
        वापस upscale_pipelines[कुंजी];
    }

  निजी:
    अटल webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_code,
                                                       std::string    label) {
        wgpu::ShaderSourceWGSL shader_source;
        shader_source.code = shader_code.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.अग्र_In_Chain = &shader_source;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = nullptr;  // nullptr means auto layout
        वापस { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#पूर्ण_अगर  // GGML_WEBGPU_SHADER_LIB_HPP
