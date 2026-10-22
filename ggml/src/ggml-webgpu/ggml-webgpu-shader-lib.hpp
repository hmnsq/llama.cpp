#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-wgsl-shaders.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "pre_wgsl.hpp"

#ଅନ୍ତର୍ଭାଵ <webgpu/webgpu_cpp.h>

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <vector>

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
template <typename T> ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_webgpu_hash_combine(size_t & seed, ଅଚଳ T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
ଧାଡ଼ିଭିତରେ uintptr_t ggml_webgpu_tensor_addr(ଅଚଳ ggml_tensor * tensor) {
    ଅଚଳ ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    ଫେରନ୍ତୁ (uintptr_t) base_tensor->data + tensor->view_offs;
}

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_webgpu_tensor_equal(ଅଚଳ ggml_tensor * a, ଅଚଳ ggml_tensor * b) {
    ଫେରନ୍ତୁ a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

ସଂରଚନା ggml_webgpu_shader_lib_context {
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
    ତାର୍କିକ        supports_subgroups       = ମିଥ୍ୟା;
    ତାର୍କିକ        supports_subgroup_matrix = ମିଥ୍ୟା;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_size        = 0;
    uint32_t    max_subgroup_size        = 0;
    ତାର୍କିକ        supports_dot_product     = ମିଥ୍ୟା;
    std::string vendor;
};

ସଂରଚନା webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<ଶୂନ୍ୟ> context = nullptr;
};

ସଂରଚନା ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 0;
    ତାର୍କିକ     inplace = ମିଥ୍ୟା;
};

ସଂରଚନା ggml_webgpu_get_rows_shader_decisions {
    uint32_t wg_size    = 0;
    ତାର୍କିକ     vectorized = ମିଥ୍ୟା;
};

ସଂରଚନା ggml_webgpu_ଦ୍ୱିମିକ_shader_decisions {
    uint32_t wg_size     = 0;
    ତାର୍କିକ     inplace     = ମିଥ୍ୟା;
    ତାର୍କିକ     overlap     = ମିଥ୍ୟା;
    ତାର୍କିକ     src_overlap = ମିଥ୍ୟା;
};

ସଂରଚନା ggml_webgpu_glu_shader_decisions {
    uint32_t wg_size     = 0;
    ତାର୍କିକ     src_overlap = ମିଥ୍ୟା;
};

ସଂରଚନା ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<ଶୂନ୍ୟ> decisions;
};

ସଂରଚନା ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_size;
    uint32_t tokens_per_wg;
};

ସଂରଚନା ggml_webgpu_ssm_scan_pipeline_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା     type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା     d_state;
    uint8_t xbc_overlap;
    ତାର୍କିକ    a_overlap;
    ତାର୍କିକ    ids_overlap;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_ssm_scan_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

ସଂରଚନା ggml_webgpu_ssm_scan_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_ssm_scan_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ggml_webgpu_hash_combine(seed, ଚାବି.d_state);
        ggml_webgpu_hash_combine(seed, ଚାବି.xbc_overlap);
        ggml_webgpu_hash_combine(seed, ଚାବି.a_overlap);
        ggml_webgpu_hash_combine(seed, ଚାବି.ids_overlap);
        ଫେରନ୍ତୁ seed;
    }
};

ସଂରଚନା ggml_webgpu_ssm_scan_shader_decisions {
    uint32_t wg_size;
    uint32_t tokens_per_tile;
    uint8_t  xbc_overlap = 0;
    ତାର୍କିକ     a_overlap   = ମିଥ୍ୟା;
    ତାର୍କିକ     ids_overlap = ମିଥ୍ୟା;
};

/** Argsort **/

ସଂରଚନା ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes;
    int32_t  order;
};

/** Set Rows **/

ସଂରଚନା ggml_webgpu_set_rows_pipeline_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା vec4;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i64_idx;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା pair_blocks;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_set_rows_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

ସଂରଚନା ggml_webgpu_set_rows_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_set_rows_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.src0_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.dst_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.vec4);
        ggml_webgpu_hash_combine(seed, ଚାବି.i64_idx);
        ggml_webgpu_hash_combine(seed, ଚାବି.pair_blocks);
        ଫେରନ୍ତୁ seed;
    }
};

ସଂରଚନା ggml_webgpu_set_rows_shader_decisions {
    ତାର୍କିକ     vec4;
    ତାର୍କିକ     i64_idx;
    ତାର୍କିକ     pair_blocks;
    uint32_t wg_size;
};

/** Set **/

ସଂରଚନା ggml_webgpu_set_pipeline_ଚାବି {
    ggml_type type;
    ତାର୍କିକ      inplace;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_set_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ type == other.type && inplace == other.inplace;
    }
};

ସଂରଚନା ggml_webgpu_set_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_set_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ggml_webgpu_hash_combine(seed, ଚାବି.inplace);
        ଫେରନ୍ତୁ seed;
    }
};

/** Get Rows **/

ସଂରଚନା ggml_webgpu_get_rows_pipeline_ଚାବି {
    ggml_type src_type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା       vectorized;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_get_rows_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ src_type == other.src_type && vectorized == other.vectorized;
    }
};

ସଂରଚନା ggml_webgpu_get_rows_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_get_rows_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.src_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.vectorized);
        ଫେରନ୍ତୁ seed;
    }
};

/** Row Norm **/

ସଂରଚନା ggml_webgpu_row_norm_pipeline_ଚାବି {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    ତାର୍କିକ      inplace;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_row_norm_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

ସଂରଚନା ggml_webgpu_row_norm_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_row_norm_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.op);
        ggml_webgpu_hash_combine(seed, ଚାବି.src_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.dst_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.inplace);
        ଫେରନ୍ତୁ seed;
    }
};

/** RMS_NORM + MUL **/

ସଂରଚନା ggml_webgpu_rms_norm_mul_pipeline_ଚାବି {
    ତାର୍କିକ inplace;      // rn_src == dst
    ତାର୍କିକ overlap;      // mul_src == dst
    ତାର୍କିକ src_overlap;  // rn_src binding overlaps mul_src binding

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_rms_norm_mul_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

ସଂରଚନା ggml_webgpu_rms_norm_mul_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_rms_norm_mul_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.inplace);
        ggml_webgpu_hash_combine(seed, ଚାବି.overlap);
        ggml_webgpu_hash_combine(seed, ଚାବି.src_overlap);
        ଫେରନ୍ତୁ seed;
    }
};

ସଂରଚନା ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    ତାର୍କିକ     inplace     = ମିଥ୍ୟା;
    ତାର୍କିକ     overlap     = ମିଥ୍ୟା;
    ତାର୍କିକ     src_overlap = ମିଥ୍ୟା;
};

/** Pad **/
ସଂରଚନା ggml_webgpu_pad_pipeline_ଚାବି {
    ତାର୍କିକ circular;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_pad_pipeline_ଚାବି & other) ଅଚଳ { ଫେରନ୍ତୁ circular == other.circular; }
};

ସଂରଚନା ggml_webgpu_pad_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_pad_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.circular);
        ଫେରନ୍ତୁ seed;
    }
};

/** Solve Tri **/
ସଂରଚନା ggml_webgpu_solve_tri_pipeline_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା k;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_solve_tri_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ type == other.type && n == other.n && k == other.k;
    }
};

ସଂରଚନା ggml_webgpu_solve_tri_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_solve_tri_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ggml_webgpu_hash_combine(seed, ଚାବି.n);
        ggml_webgpu_hash_combine(seed, ଚାବି.k);
        ଫେରନ୍ତୁ seed;
    }
};

/** SSM Conv **/
ସଂରଚନା ggml_webgpu_ssm_conv_pipeline_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା vectorized;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_ssm_conv_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
ସଂରଚନା ggml_webgpu_conv2d_pipeline_ଚାବି {
    ggml_type weight_type;
    ggml_type ନିବେଶ_type;
    ggml_type output_type;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_conv2d_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ weight_type == other.weight_type && ନିବେଶ_type == other.ନିବେଶ_type && output_type == other.output_type;
    }
};

ସଂରଚନା ggml_webgpu_conv2d_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_conv2d_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.weight_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.ନିବେଶ_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.output_type);
        ଫେରନ୍ତୁ seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
ସଂରଚନା ggml_webgpu_conv2d_dw_pipeline_ଚାବି {
    ggml_type weight_type;
    ggml_type ନିବେଶ_type;
    ggml_type output_type;
    ତାର୍କିକ      whcn;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_conv2d_dw_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ weight_type == other.weight_type && ନିବେଶ_type == other.ନିବେଶ_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

ସଂରଚନା ggml_webgpu_conv2d_dw_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_conv2d_dw_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.weight_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.ନିବେଶ_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.output_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.whcn);
        ଫେରନ୍ତୁ seed;
    }
};

/** Im2Col **/
ସଂରଚନା ggml_webgpu_im2col_pipeline_ଚାବି {
    ggml_type ନିବେଶ_type;
    ggml_type output_type;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_im2col_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ ନିବେଶ_type == other.ନିବେଶ_type && output_type == other.output_type;
    }
};

ସଂରଚନା ggml_webgpu_im2col_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_im2col_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.ନିବେଶ_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.output_type);
        ଫେରନ୍ତୁ seed;
    }
};

/** Gated Delta Net **/
ସଂରଚନା ggml_webgpu_gated_delta_net_pipeline_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  s_v;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  kda;
    ତାର୍କିକ fused_cache;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_gated_delta_net_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

ସଂରଚନା ggml_webgpu_gated_delta_net_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_gated_delta_net_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ggml_webgpu_hash_combine(seed, ଚାବି.s_v);
        ggml_webgpu_hash_combine(seed, ଚାବି.kda);
        ଫେରନ୍ତୁ seed;
    }
};

ସଂରଚନା ggml_webgpu_ssm_conv_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_ssm_conv_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ggml_webgpu_hash_combine(seed, ଚାବି.vectorized);
        ଫେରନ୍ତୁ seed;
    }
};

/** Scale **/

ସଂରଚନା ggml_webgpu_scale_pipeline_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା inplace;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_scale_pipeline_ଚାବି & other) ଅଚଳ { ଫେରନ୍ତୁ inplace == other.inplace; }
};

ସଂରଚନା ggml_webgpu_scale_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_scale_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.inplace);
        ଫେରନ୍ତୁ seed;
    }
};

/** Upscale **/

ସଂରଚନା ggml_webgpu_upscale_pipeline_ଚାବି {
    ggml_type ନିବେଶ_type;
    ggml_type output_type;
    uint32_t  base_mode;
    ତାର୍କିକ      antialias;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_upscale_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ ନିବେଶ_type == other.ନିବେଶ_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

ସଂରଚନା ggml_webgpu_upscale_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_upscale_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.ନିବେଶ_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.output_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.base_mode);
        ggml_webgpu_hash_combine(seed, ଚାବି.antialias);
        ଫେରନ୍ତୁ seed;
    }
};

/** Concat **/

ସଂରଚନା ggml_webgpu_concat_pipeline_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  type;
    ତାର୍କିକ src_overlap;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_concat_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ type == other.type && src_overlap == other.src_overlap;
    }
};

ସଂରଚନା ggml_webgpu_concat_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_concat_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ggml_webgpu_hash_combine(seed, ଚାବି.src_overlap);
        ଫେରନ୍ତୁ seed;
    }
};

/** Repeat **/

ସଂରଚନା ggml_webgpu_repeat_pipeline_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା type;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_repeat_pipeline_ଚାବି & other) ଅଚଳ { ଫେରନ୍ତୁ type == other.type; }
};

ସଂରଚନା ggml_webgpu_repeat_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_repeat_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ଫେରନ୍ତୁ seed;
    }
};

/** Binary **/

ସଂରଚନା ggml_webgpu_ଦ୍ୱିମିକ_pipeline_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  op;
    ତାର୍କିକ inplace;
    ତାର୍କିକ overlap;
    ତାର୍କିକ src_overlap;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_ଦ୍ୱିମିକ_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

ସଂରଚନା ggml_webgpu_ଦ୍ୱିମିକ_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_ଦ୍ୱିମିକ_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ggml_webgpu_hash_combine(seed, ଚାବି.op);
        ggml_webgpu_hash_combine(seed, ଚାବି.inplace);
        ggml_webgpu_hash_combine(seed, ଚାବି.overlap);
        ggml_webgpu_hash_combine(seed, ଚାବି.src_overlap);
        ଫେରନ୍ତୁ seed;
    }
};

/* Add_Id */

ସଂରଚନା ggml_webgpu_add_id_pipeline_ଚାବି {
    ତାର୍କିକ inplace;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_add_id_pipeline_ଚାବି & other) ଅଚଳ { ଫେରନ୍ତୁ inplace == other.inplace; }
};

ସଂରଚନା ggml_webgpu_add_id_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_add_id_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.inplace);
        ଫେରନ୍ତୁ seed;
    }
};

/** Unary **/

ସଂରଚନା ggml_webgpu_unary_pipeline_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           op;
    ତାର୍କିକ          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    ତାର୍କିକ          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_unary_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

ସଂରଚନା ggml_webgpu_unary_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_unary_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ggml_webgpu_hash_combine(seed, ଚାବି.op);
        ggml_webgpu_hash_combine(seed, ଚାବି.is_unary);
        ggml_webgpu_hash_combine(seed, ଚାବି.inplace);
        ggml_webgpu_hash_combine(seed, ଚାବି.ttype);
        ଫେରନ୍ତୁ seed;
    }
};

/** FlashAttention */

ସଂରଚନା ggml_webgpu_flash_attn_common_pipeline_ଚାବି {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    ତାର୍କିକ      k_direct;
    ତାର୍କିକ      v_direct;
    ତାର୍କିକ      kv_overlap;
    ତାର୍କିକ      has_mask;
    ତାର୍କିକ      has_sinks;
    ତାର୍କିକ      uses_logit_softcap;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_flash_attn_common_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_webgpu_flash_attn_hash_common_pipeline_ଚାବି(size_t &                                           seed,
                                                            ଅଚଳ ggml_webgpu_flash_attn_common_pipeline_ଚାବି & ଚାବି) {
    ggml_webgpu_hash_combine(seed, ଚାବି.q_type);
    ggml_webgpu_hash_combine(seed, ଚାବି.k_type);
    ggml_webgpu_hash_combine(seed, ଚାବି.v_type);
    ggml_webgpu_hash_combine(seed, ଚାବି.dst_type);
    ggml_webgpu_hash_combine(seed, ଚାବି.head_dim_qk);
    ggml_webgpu_hash_combine(seed, ଚାବି.head_dim_v);
    ggml_webgpu_hash_combine(seed, ଚାବି.k_direct);
    ggml_webgpu_hash_combine(seed, ଚାବି.v_direct);
    ggml_webgpu_hash_combine(seed, ଚାବି.kv_overlap);
    ggml_webgpu_hash_combine(seed, ଚାବି.has_mask);
    ggml_webgpu_hash_combine(seed, ଚାବି.has_sinks);
    ggml_webgpu_hash_combine(seed, ଚାବି.uses_logit_softcap);
}

ସଂରଚନା ggml_webgpu_flash_attn_vec_pipeline_ଚାବି {
    ggml_webgpu_flash_attn_common_pipeline_ଚାବି common;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_flash_attn_vec_pipeline_ଚାବି & other) ଅଚଳ { ଫେରନ୍ତୁ common == other.common; }
};

ସଂରଚନା ggml_webgpu_flash_attn_vec_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_flash_attn_vec_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ଚାବି(seed, ଚାବି.common);
        ଫେରନ୍ତୁ seed;
    }
};

ସଂରଚନା ggml_webgpu_flash_attn_pipeline_ଚାବି {
    ggml_webgpu_flash_attn_common_pipeline_ଚାବି common;
    ତାର୍କିକ                                       use_sg_matrix;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_flash_attn_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

ସଂରଚନା ggml_webgpu_flash_attn_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_flash_attn_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ଚାବି(seed, ଚାବି.common);
        ggml_webgpu_hash_combine(seed, ଚାବି.use_sg_matrix);
        ଫେରନ୍ତୁ seed;
    }
};

ସଂରଚନା ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

ସଂରଚନା ggml_webgpu_flash_attn_decisions {
    ତାର୍କିକ     use_sg_matrix = ମିଥ୍ୟା;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_size       = 0;
};

ଧାଡ଼ିଭିତରେ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
ଧାଡ଼ିଭିତରେ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

ଧାଡ଼ିଭିତରେ size_t ggml_webgpu_flash_attn_tensor_offset(ଅଚଳ ggml_tensor * tensor) {
    constexpr uintptr_t ptr_base_addr = 0x1000u;
    ଅଚଳ ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
    ଫେରନ୍ତୁ reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
}

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_webgpu_flash_attn_float_vec4_aligned(ଅଚଳ ggml_tensor * K, size_t storage_offset_alignment) {
    ଅଚଳ uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_size(K->type));
    ଫେରନ୍ତୁ offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_webgpu_flash_attn_float_vec4_aligned(ଅଚଳ ggml_tensor * K,
                                                      ଅଚଳ ggml_tensor * V,
                                                      size_t              storage_offset_alignment) {
    ଫେରନ୍ତୁ ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_webgpu_flash_attn_k_direct(ଅଚଳ ggml_tensor * Q, ଅଚଳ ggml_tensor * K, uint32_t kv_direct_align) {
    ଫେରନ୍ତୁ (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_webgpu_flash_attn_v_direct(ଅଚଳ ggml_tensor * Q, ଅଚଳ ggml_tensor * V, uint32_t kv_direct_align) {
    ଫେରନ୍ତୁ ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

ଧାଡ଼ିଭିତରେ ggml_webgpu_flash_attn_common_pipeline_ଚାବି ggml_webgpu_flash_attn_make_common_pipeline_ଚାବି(
    ଅଚଳ ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    ତାର୍କିକ                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_ଚାବି ଚାବି = {};
    ଚାବି.q_type                                     = context.src0->type;
    ଚାବି.k_type                                     = context.src1->type;
    ଚାବି.v_type                                     = context.src2->type;
    ଚାବି.dst_type                                   = context.dst->type;
    ଚାବି.head_dim_qk                                = (uint32_t) context.src0->ne[0];
    ଚାବି.head_dim_v                                 = (uint32_t) context.src2->ne[0];
    ଚାବି.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    ଚାବି.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    ଚାବି.kv_overlap         = kv_overlap;
    ଚାବି.has_mask           = context.src3 != nullptr;
    ଚାବି.has_sinks          = context.src4 != nullptr;
    ଚାବି.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    ଫେରନ୍ତୁ ଚାବି;
}

ଧାଡ଼ିଭିତରେ std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    ଅଚଳ ggml_webgpu_flash_attn_common_pipeline_ଚାବି & ଚାବି,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_size) {
    std::vector<std::string> defines;

    ବାଛ (ଚାବି.k_type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            defines.push_back("K_F32");
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            defines.push_back("K_F16");
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            defines.push_back("K_Q4_0");
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            defines.push_back("K_Q8_0");
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(ଚାବି.k_type);

    ବାଛ (ଚାବି.v_type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            defines.push_back("V_F32");
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            defines.push_back("V_F16");
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            defines.push_back("V_Q4_0");
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            defines.push_back("V_Q8_0");
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(ଚାବି.v_type);

    ବାଛ (ଚାବି.q_type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            defines.push_back("Q_F32");
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            defines.push_back("Q_F16");
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(ଚାବି.q_type);

    ବାଛ (ଚାବି.dst_type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            defines.push_back("DST_F32");
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            defines.push_back("DST_F16");
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(ଚାବି.dst_type);

    ଯଦି (ଚାବି.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    ଯଦି (ଚାବି.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    ଯଦି (ଚାବି.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    ଯଦି (ଚାବି.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    ଯଦି (ଚାବି.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    ଯଦି (ଚାବି.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(ଚାବି.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(ଚାବି.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(ଚାବି.head_dim_v));
    variant += std::string("_hsv") + std::to_string(ଚାବି.head_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    ଯଦି (ggml_is_quantized(ଚାବି.k_type) || ggml_is_quantized(ଚାବି.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        ଯଦି (ggml_is_quantized(ଚାବି.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        ଯଦି (ggml_is_quantized(ଚାବି.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    ଫେରନ୍ତୁ defines;
}

ସଂରଚନା ggml_webgpu_flash_attn_vec_reduce_pipeline_ଚାବି {
    uint32_t  head_dim_v;
    uint32_t  wg_size;
    ggml_type dst_type;
};

ସଂରଚନା ggml_webgpu_flash_attn_vec_reduce_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_flash_attn_vec_reduce_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.head_dim_v);
        ggml_webgpu_hash_combine(seed, ଚାବି.wg_size);
        ggml_webgpu_hash_combine(seed, ଚାବି.dst_type);
        ଫେରନ୍ତୁ seed;
    }
};

ଧାଡ଼ିଭିତରେ ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_flash_attn_vec_reduce_pipeline_ଚାବି & lhs,
                       ଅଚଳ ggml_webgpu_flash_attn_vec_reduce_pipeline_ଚାବି & rhs) {
    ଫେରନ୍ତୁ lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size && lhs.dst_type == rhs.dst_type;
}

ସଂରଚନା ggml_webgpu_flash_attn_blk_pipeline_ଚାବି {
    uint32_t kv_tile;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_flash_attn_blk_pipeline_ଚାବି & other) ଅଚଳ { ଫେରନ୍ତୁ kv_tile == other.kv_tile; }
};

ସଂରଚନା ggml_webgpu_flash_attn_blk_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_flash_attn_blk_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.kv_tile);
        ଫେରନ୍ତୁ seed;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
ଧାଡ଼ିଭିତରେ size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  ତାର୍କିକ     has_mask,
                                                  ତାର୍କିକ     kv_direct) {
    ଅଚଳ uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * head_dim_qk;        // q_shmem
    ଯଦି (!kv_direct) {
        f32_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f32_elems += q_tile * head_dim_v;         // o_shmem
    ଯଦି (has_mask) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    ଫେରନ୍ତୁ f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

ଧାଡ଼ିଭିତରେ uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   limit_bytes,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t head_dim_qk,
                                                   uint32_t head_dim_v,
                                                   ତାର୍କିକ     has_mask,
                                                   ତାର୍କିକ     kv_direct) {
    ଅଚଳ size_t base_q_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 0, head_dim_qk, head_dim_v, has_mask, kv_direct);
    ଯଦି (limit_bytes <= base_q_bytes) {
        ଫେରନ୍ତୁ 0;
    }
    ଅଚଳ size_t one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, head_dim_qk, head_dim_v, has_mask, kv_direct);
    ଅଚଳ size_t bytes_per_kv = one_kv_bytes - base_q_bytes;
    ଯଦି (bytes_per_kv == 0) {
        ଫେରନ୍ତୁ 0;
    }
    ଅଚଳ size_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    ଫେରନ୍ତୁ (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

ଧାଡ଼ିଭିତରେ uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_limit_bytes,
                                                       uint32_t head_dim_qk,
                                                       uint32_t head_dim_v,
                                                       ତାର୍କିକ     has_mask,
                                                       ତାର୍କିକ     kv_direct) {
    ଅଚଳ uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_limit_bytes, 1u, 1u, head_dim_qk, head_dim_v, has_mask, kv_direct);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    ଯଦି (kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        ଯେତେବେଳେ (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    ଫେରନ୍ତୁ kv_tile;
}

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(ତାର୍କିକ                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ଅଚଳ ggml_tensor * Q,
                                                                ଅଚଳ ggml_tensor * V) {
    ଫେରନ୍ତୁ supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

ସଂରଚନା ggml_webgpu_mul_mat_vec_pipeline_ଚାବି {
    ggml_type src0_type;
    ggml_type src1_type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା       vectorized;
    uint32_t  ଅଙ୍କ_cols;
    ତାର୍କିକ      use_mmvq;
    ତାର୍କିକ      src_overlap;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_mul_mat_vec_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               ଅଙ୍କ_cols == other.ଅଙ୍କ_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

ସଂରଚନା ggml_webgpu_mul_mat_vec_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_mul_mat_vec_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.src0_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.src1_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.vectorized);
        ggml_webgpu_hash_combine(seed, ଚାବି.ଅଙ୍କ_cols);
        ggml_webgpu_hash_combine(seed, ଚାବି.use_mmvq);
        ggml_webgpu_hash_combine(seed, ଚାବି.src_overlap);
        ଫେରନ୍ତୁ seed;
    }
};

ସଂରଚନା ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
    ତାର୍କିକ     src_overlap = ମିଥ୍ୟା;
};

ସଂରଚନା ggml_webgpu_quantize_q8_pipeline_ଚାବି {
    ggml_type src0_type;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_quantize_q8_pipeline_ଚାବି & other) ଅଚଳ { ଫେରନ୍ତୁ src0_type == other.src0_type; }
};

ସଂରଚନା ggml_webgpu_quantize_q8_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_quantize_q8_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.src0_type);
        ଫେରନ୍ତୁ seed;
    }
};

ସଂରଚନା ggml_webgpu_mul_mat_pipeline_ଚାବି {
    ggml_type src0_type;
    ggml_type src1_type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା       vectorized;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା       use_subgroup_matrix;
    ତାର୍କିକ      src_overlap;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_mul_mat_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

ସଂରଚନା ggml_webgpu_mul_mat_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_mul_mat_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.src0_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.src1_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.vectorized);
        ggml_webgpu_hash_combine(seed, ଚାବି.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, ଚାବି.src_overlap);
        ଫେରନ୍ତୁ seed;
    }
};

ସଂରଚନା ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_size;
    ତାର୍କିକ     src_overlap = ମିଥ୍ୟା;
};

/** MUL_MAT_ID **/

ସଂରଚନା ggml_webgpu_mul_mat_id_gather_pipeline_ଚାବି {
    uint32_t n_experts;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_mul_mat_id_gather_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ n_experts == other.n_experts;
    }
};

ସଂରଚନା ggml_webgpu_mul_mat_id_gather_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_mul_mat_id_gather_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.n_experts);
        ଫେରନ୍ତୁ seed;
    }
};

ସଂରଚନା ggml_webgpu_mul_mat_id_pipeline_ଚାବି {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  ଅଙ୍କ_cols;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା       vectorized;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_mul_mat_id_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               ଅଙ୍କ_cols == other.ଅଙ୍କ_cols && vectorized == other.vectorized;
    }
};

ସଂରଚନା ggml_webgpu_mul_mat_id_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_mul_mat_id_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.src0_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.src1_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.n_experts);
        ggml_webgpu_hash_combine(seed, ଚାବି.ଅଙ୍କ_cols);
        ggml_webgpu_hash_combine(seed, ଚାବି.vectorized);
        ଫେରନ୍ତୁ seed;
    }
};

/** Cpy **/

ସଂରଚନା ggml_webgpu_cpy_pipeline_ଚାବି {
    ggml_type src_type;
    ggml_type dst_type;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_cpy_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ src_type == other.src_type && dst_type == other.dst_type;
    }
};

ସଂରଚନା ggml_webgpu_cpy_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_cpy_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.src_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.dst_type);
        ଫେରନ୍ତୁ seed;
    }
};

/** Glu **/

ସଂରଚନା ggml_webgpu_glu_pipeline_ଚାବି {
    ggml_glu_op glu_op;
    ggml_type   type;
    ତାର୍କିକ        split;
    ତାର୍କିକ        src_overlap;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_glu_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

ସଂରଚନା ggml_webgpu_glu_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_glu_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.glu_op);
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ggml_webgpu_hash_combine(seed, ଚାବି.split);
        ggml_webgpu_hash_combine(seed, ଚାବି.src_overlap);
        ଫେରନ୍ତୁ seed;
    }
};

/** Rope **/

ସଂରଚନା ggml_webgpu_rope_pipeline_ଚାବି {
    ggml_type type;
    ତାର୍କିକ      inplace;
    ତାର୍କିକ      has_ff;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_rope_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

ସଂରଚନା ggml_webgpu_rope_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_rope_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.type);
        ggml_webgpu_hash_combine(seed, ଚାବି.inplace);
        ggml_webgpu_hash_combine(seed, ଚାବି.has_ff);
        ଫେରନ୍ତୁ seed;
    }
};

/** SoftMax **/

ସଂରଚନା ggml_webgpu_soft_max_pipeline_ଚାବି {
    ggml_type mask_type;
    ତାର୍କିକ      has_mask;
    ତାର୍କିକ      has_sink;
    ତାର୍କିକ      inplace;

    ତାର୍କିକ operator==(ଅଚଳ ggml_webgpu_soft_max_pipeline_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

ସଂରଚନା ggml_webgpu_soft_max_pipeline_ଚାବି_hash {
    size_t operator()(ଅଚଳ ggml_webgpu_soft_max_pipeline_ଚାବି & ଚାବି) ଅଚଳ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ଚାବି.mask_type);
        ggml_webgpu_hash_combine(seed, ଚାବି.has_mask);
        ggml_webgpu_hash_combine(seed, ଚାବି.has_sink);
        ggml_webgpu_hash_combine(seed, ଚାବି.inplace);
        ଫେରନ୍ତୁ seed;
    }
};

/** MMVQ **/

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_webgpu_can_use_mmvq(ଅଚଳ ggml_tensor * src0,
                                     ଅଚଳ ggml_tensor * src1,
                                     ତାର୍କିକ                supports_dot_product,
                                     ଅଚଳ std::string & vendor) {
    ଯଦି (src1->ne[1] <= 4) {
        ତାର୍କିକ supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        ଯଦି (supports_dp4a && supports_dot_product) {
            ବାଛ (src1->type) {
                ସ୍ଥିତି GGML_TYPE_F32:
                    ବାଛ (src0->type) {
                        ସ୍ଥିତି GGML_TYPE_Q1_0:
                        ସ୍ଥିତି GGML_TYPE_Q4_0:
                        ସ୍ଥିତି GGML_TYPE_Q4_1:
                        ସ୍ଥିତି GGML_TYPE_Q5_0:
                        ସ୍ଥିତି GGML_TYPE_Q5_1:
                        ସ୍ଥିତି GGML_TYPE_Q8_0:
                        ସ୍ଥିତି GGML_TYPE_MXFP4:
                        ସ୍ଥିତି GGML_TYPE_Q2_K:
                        ସ୍ଥିତି GGML_TYPE_Q3_K:
                        ସ୍ଥିତି GGML_TYPE_Q4_K:
                        ସ୍ଥିତି GGML_TYPE_Q5_K:
                        ସ୍ଥିତି GGML_TYPE_Q6_K:
                            ଫେରନ୍ତୁ src0->ne[0] % 4 == 0;
                        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                            ଭାଙ୍ଗ;
                    }
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଭାଙ୍ଗ;
            }
        }
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ଶ୍ରେଣୀ ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_ଚାବି_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_ଚାବି_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_unary_pipeline_ଚାବି_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_scale_pipeline_ଚାବି_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_ଚାବି_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_ଚାବି_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_ଚାବି_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_ଚାବି,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_ଚାବି_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_pad_pipeline_ଚାବି_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_ଦ୍ୱିମିକ_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_ଦ୍ୱିମିକ_pipeline_ଚାବି_hash>
        ଦ୍ୱିମିକ_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_add_id_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_add_id_pipeline_ଚାବି_hash>
        add_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_concat_pipeline_ଚାବି_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_repeat_pipeline_ଚାବି_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_ଚାବି,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_ଚାବି_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_ଚାବି_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_ଚାବି,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_ଚାବି_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_ଚାବି,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_ଚାବି_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_ଚାବି_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_ଚାବି_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_ଚାବି_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_ଚାବି,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_ଚାବି_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_ଚାବି_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_ଚାବି_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_ଚାବି_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_set_pipeline_ଚାବି_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_cpy_pipeline_ଚାବି_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_glu_pipeline_ଚାବି_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_rope_pipeline_ଚାବି_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_ଚାବି_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_ଚାବି_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_ଚାବି_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_im2col_pipeline_ଚାବି_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_ଚାବି,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_ଚାବି_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_ଚାବି, webgpu_pipeline, ggml_webgpu_upscale_pipeline_ଚାବି_hash>
        upscale_pipelines;

  ସର୍ବସାଧାରଣ:
    ggml_webgpu_shader_lib(wgpu::Device device) { ଏହି->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        auto it = sum_rows_pipelines.find(1);
        ଯଦି (it != sum_rows_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        ଫେରନ୍ତୁ sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.op                                = context.dst->op;
        ଚାବି.src_type                          = context.src0->type;
        ଚାବି.dst_type                          = context.dst->type;
        ଚାବି.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = row_norm_pipelines.find(ଚାବି);
        ଯଦି (it != row_norm_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        ବାଛ (ଚାବି.op) {
            ସ୍ଥିତି GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_NORM:
                defines.push_back("NORM");
                variant = "norm";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        ଯଦି (ଚାବି.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        ଯଦି (ଚାବି.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        ଯଦି (ଚାବି.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ଅଚଳ uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = ଚାବି.inplace;
        row_norm_pipelines[ଚାବି]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[ଚାବି].context = decisions;
        ଫେରନ୍ତୁ row_norm_pipelines[ଚାବି];
    }

    webgpu_pipeline get_argmax_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ତାର୍କିକ vec4 = context.src0->ne[0] % 4 == 0;

        auto it = argmax_pipelines.find(vec4);
        ଯଦି (it != argmax_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        ଯଦି (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        auto processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        ଫେରନ୍ତୁ argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ଅଚଳ ତାର୍କିକ                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_ଚାବି ଚାବି       = {};
        ଚାବି.src0_type                               = context.src0->type;
        ଚାବି.dst_type                                = context.dst->type;
        ଚାବି.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        ଚାବି.i64_idx     = context.src1->type == GGML_TYPE_I64;
        ଚାବି.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_size(context.dst->type)) % 2 == 0);

        auto it = set_rows_pipelines.find(ଚାବି);
        ଯଦି (it != set_rows_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        ବାଛ (context.dst->type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q8_0:
                defines.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q4_0:
                defines.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        ଯଦି (context.src0->type == GGML_TYPE_F16) {
            defines.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        ଯଦି (ଚାବି.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        ଯଦି (ଚାବି.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        ଯଦି (ଚାବି.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        ଅଚଳ auto & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        auto         processed          = preprocessor.preprocess(shader_source, defines);
        auto         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = ଚାବି.vec4;
        decisions->i64_idx              = ଚାବି.i64_idx;
        decisions->pair_blocks          = ଚାବି.pair_blocks;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[ଚାବି]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[ଚାବି].context = decisions;
        ଫେରନ୍ତୁ set_rows_pipelines[ଚାବି];
    }

    webgpu_pipeline get_set_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.type                         = context.dst->type;
        ଚାବି.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = set_pipelines.find(ଚାବି);
        ଯଦି (it != set_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for set shader");
        }

        ଯଦି (ଚାବି.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_set, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ଚାବି.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[ଚାବି]       = pipeline;
        ଫେରନ୍ତୁ set_pipelines[ଚାବି];
    }

    webgpu_pipeline get_cumsum_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        auto it = cumsum_pipelines.find(1);
        ଯଦି (it != cumsum_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        ଫେରନ୍ତୁ cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ତାର୍କିକ          is_ଶୀର୍ଷ_k = context.dst->op == GGML_OP_ଶୀର୍ଷ_K;
        // ascending order is 0, descending order is 1
        ଅଚଳ int32_t order =
            is_ଶୀର୍ଷ_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_pipelines.find(order);
        ଯଦି (it != argsort_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        ଯେତେବେଳେ (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        auto decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        ଫେରନ୍ତୁ argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ତାର୍କିକ          is_ଶୀର୍ଷ_k = context.dst->op == GGML_OP_ଶୀର୍ଷ_K;
        // ascending order is 0, descending order is 1
        ଅଚଳ int32_t order =
            is_ଶୀର୍ଷ_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_merge_pipelines.find(order);
        ଯଦି (it != argsort_merge_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        ଫେରନ୍ତୁ argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, ତାର୍କିକ vec4_aligned) {
        ଅଚଳ ତାର୍କିକ vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.src_type                          = context.src0->type;
        ଚାବି.vectorized                        = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) vectorized;

        auto it = get_rows_pipelines.find(ଚାବି);
        ଯଦି (it != get_rows_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        ଅଚଳ ସଂରଚନା ggml_type_traits * type_traits = ggml_get_type_traits(ଚାବି.src_type);
        ଅଚଳ ଅକ୍ଷର *                    type_str    = type_traits->type_name;

        ବାଛ (ଚାବି.src_type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                ଯଦି (ଚାବି.vectorized) {
                    defines.push_back("F32_VEC");
                    defines.push_back("SRC_TYPE=vec4<f32>");
                    defines.push_back("DST_TYPE=vec4<f32>");
                    defines.push_back("BLOCK_SIZE=4u");
                } ଅନ୍ୟଥା {
                    defines.push_back("F32");
                    defines.push_back("SRC_TYPE=f32");
                    defines.push_back("DST_TYPE=f32");
                    defines.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_BF16:
                defines.push_back("BF16");
                defines.push_back("SRC_TYPE=u32");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    ବାଛ (ଚାବି.src_type) {
                        ସ୍ଥିତି GGML_TYPE_Q1_0:
                        ସ୍ଥିତି GGML_TYPE_Q2_0:
                        ସ୍ଥିତି GGML_TYPE_Q4_0:
                        ସ୍ଥିତି GGML_TYPE_Q5_0:
                        ସ୍ଥିତି GGML_TYPE_Q8_0:
                        ସ୍ଥିତି GGML_TYPE_Q3_K:
                        ସ୍ଥିତି GGML_TYPE_Q6_K:
                        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
                        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
                        ସ୍ଥିତି GGML_TYPE_IQ2_S:
                        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
                        ସ୍ଥିତି GGML_TYPE_IQ3_S:
                        ସ୍ଥିତି GGML_TYPE_IQ1_S:
                        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
                        ସ୍ଥିତି GGML_TYPE_MXFP4:
                        ସ୍ଥିତି GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                ଭାଙ୍ଗ;
                            }
                        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
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

                    ଯଦି (ଚାବି.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } ଅନ୍ୟଥା ଯଦି ((ଚାବି.src_type >= GGML_TYPE_Q4_0 && ଚାବି.src_type <= GGML_TYPE_Q8_1) ||
                               ଚାବି.src_type == GGML_TYPE_IQ4_NL || ଚାବି.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } ଅନ୍ୟଥା ଯଦି (ଚାବି.src_type == GGML_TYPE_Q2_0 || ଚାବି.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } ଅନ୍ୟଥା ଯଦି (ଚାବି.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } ଅନ୍ୟଥା {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    ଭାଙ୍ଗ;
                }
        }

        ଯଦି (ଚାବି.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        auto decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[ଚାବି]  = pipeline;
        ଫେରନ୍ତୁ get_rows_pipelines[ଚାବି];
    }

    webgpu_pipeline get_scale_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = scale_pipelines.find(ଚାବି);
        ଯଦି (it != scale_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        ଯଦି (ଚାବି.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_scale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ଚାବି.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[ଚାବି]     = pipeline;
        ଫେରନ୍ତୁ scale_pipelines[ଚାବି];
    }

    webgpu_pipeline get_solve_tri_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.type                               = context.dst->type;
        ଚାବି.n                                  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) context.src0->ne[0];
        ଚାବି.k                                  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) context.src1->ne[0];

        auto it = solve_tri_pipelines.find(ଚାବି);
        ଯଦି (it != solve_tri_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                variant += "_f32";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ଅଚଳ uint32_t wg_size       = std::min((uint32_t) ଚାବି.n, context.max_wg_size);
        ଅଚଳ uint32_t k_tile        = wg_size;
        ଅଚଳ uint32_t bytes_per_row = ((uint32_t) ଚାବି.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        ଅଚଳ uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(ଚାବି.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        auto processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ solve_tri_pipelines[ଚାବି];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.type                              = context.dst->type;
        ଚାବି.vectorized                        = context.src1->ne[0] == 4;

        auto it = ssm_conv_pipelines.find(ଚାବି);
        ଯଦି (it != ssm_conv_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                variant += "_f32";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        ଯଦି (ଚାବି.vectorized) {
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
        ssm_conv_pipelines[ଚାବି]  = pipeline;
        ଫେରନ୍ତୁ ssm_conv_pipelines[ଚାବି];
    }

    webgpu_pipeline get_ssm_scan_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.type                              = context.dst->type;
        ଚାବି.d_state                           = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) context.src0->ne[0];
        ଚାବି.xbc_overlap                       = xbc_overlap;

        auto it = ssm_scan_pipelines.find(ଚାବି);
        ଯଦି (it != ssm_scan_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                variant += "_f32";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ଅଚଳ uint32_t wg_size = (uint32_t) ଚାବି.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        ଯଦି (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } ଅନ୍ୟଥା {
            variant += "_wg_reduce";
        }

        ଯଦି (ଚାବି.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(ଚାବି.d_state);

        auto processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        auto decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = ଚାବି.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[ଚାବି]    = pipeline;
        ଫେରନ୍ତୁ ssm_scan_pipelines[ଚାବି];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.type                                     = context.dst->type;
        ଚାବି.s_v                                      = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) context.src2->ne[0];
        ଚାବି.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        ଚାବି.fused_cache                              = context.dst_fuse != nullptr;

        auto it = gated_delta_net_pipelines.find(ଚାବି);
        ଯଦି (it != gated_delta_net_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                variant += "_f32";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        ଯଦି (ଚାବି.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        ଯଦି (ଚାବି.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(ଚାବି.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(ଚାବି.s_v) + "u");

        auto            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ gated_delta_net_pipelines[ଚାବି];
    }

    webgpu_pipeline get_pad_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        auto it = pad_pipelines.find(ଚାବି);
        ଯଦି (it != pad_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        ଯଦି (ଚାବି.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_pad, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[ଚାବି]       = pipeline;
        ଫେରନ୍ତୁ pad_pipelines[ଚାବି];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.src0_type                            = context.src0->type;

        auto it = quantize_q8_pipelines.find(ଚାବି);
        ଯଦି (it != quantize_q8_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }
        ଅଚଳ ଅକ୍ଷର *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> defines;
        std::string              variant = "quantize_q8";

        uint32_t wg_size = WEBGPU_MUL_MAT_VEC_WG_SIZE;

        defines.push_back("SRC1_INNER_TYPE=f32");
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        ଅଚଳ ସଂରଚନା ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
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
        quantize_q8_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ quantize_q8_pipelines[ଚାବି];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, ତାର୍କିକ src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.src0_type                            = context.src0->type;
        ଚାବି.src1_type                            = context.src1->type;
        ଚାବି.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        ଚାବି.ଅଙ୍କ_cols   = context.dst->ne[1];
        ଚାବି.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
        ଚାବି.src_overlap = src_overlap;

        auto it = mul_mat_vec_pipelines.find(ଚାବି);
        ଯଦି (it != mul_mat_vec_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        ଅଚଳ ଅକ୍ଷର *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        ବାଛ (context.src0->type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ଅଚଳ ସଂରଚନା ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    ବାଛ (context.src0->type) {
                        ସ୍ଥିତି GGML_TYPE_Q4_0:
                        ସ୍ଥିତି GGML_TYPE_Q4_1:
                        ସ୍ଥିତି GGML_TYPE_Q5_0:
                        ସ୍ଥିତି GGML_TYPE_Q5_1:
                        ସ୍ଥିତି GGML_TYPE_Q8_0:
                            ଯଦି (ଚାବି.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_Q1_0:
                        ସ୍ଥିତି GGML_TYPE_Q2_K:
                        ସ୍ଥିତି GGML_TYPE_Q3_K:
                        ସ୍ଥିତି GGML_TYPE_Q4_K:
                        ସ୍ଥିତି GGML_TYPE_Q5_K:
                        ସ୍ଥିତି GGML_TYPE_Q6_K:
                            ଯଦି (ଚାବି.use_mmvq) {
                                defines.push_back("K_QUANTS_HANDLING");
                            }
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_IQ1_S:
                        ସ୍ଥିତି GGML_TYPE_IQ1_M:
                        ସ୍ଥିତି GGML_TYPE_IQ2_S:
                        ସ୍ଥିତି GGML_TYPE_IQ3_S:
                        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
                        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
                        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
                        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_MXFP4:
                            defines.push_back(type_upper + "_LUT");
                            ଯଦି (ଚାବି.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ଭାଙ୍ଗ;
                        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                            ଭାଙ୍ଗ;
                    }
                    ଭାଙ୍ଗ;
                }
        }

        // src1 type (vector)
        ବାଛ (context.src1->type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(ଚାବି.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        ଯଦି (ଚାବି.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        ଯଦି (ଚାବି.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        ଯଦି (ଚାବି.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        ଯଦି (ଚାବି.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(ଚାବି.ଅଙ୍କ_cols));

        auto processed            = preprocessor.preprocess(shader_src, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = ଚାବି.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ mul_mat_vec_pipelines[ଚାବି];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, ତାର୍କିକ src_overlap) {
        ggml_webgpu_mul_mat_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.src0_type                        = context.src0->type;
        ଚାବି.src1_type                        = context.src1->type;
        ଚାବି.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        ଚାବି.use_subgroup_matrix = context.supports_subgroup_matrix;
        ଚାବି.src_overlap         = src_overlap;

        auto it = mul_mat_fast_pipelines.find(ଚାବି);
        ଯଦି (it != mul_mat_fast_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        ଅଚଳ ଅକ୍ଷର * shader_src = ଚାବି.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = ଚାବି.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        ବାଛ (context.src1->type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ଅଚଳ ସଂରଚନା ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ଅଚଳ ଅକ୍ଷର *                    src0_name   = src0_traits->type_name;

        ବାଛ (context.src0->type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("TYPE_F32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("TYPE_F16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("TYPE_BF16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    ବାଛ (context.src0->type) {
                        ସ୍ଥିତି GGML_TYPE_IQ1_S:
                        ସ୍ଥିତି GGML_TYPE_IQ1_M:
                        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
                        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
                        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
                        ସ୍ଥିତି GGML_TYPE_IQ2_S:
                        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
                        ସ୍ଥିତି GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_MXFP4:
                        ସ୍ଥିତି GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ଭାଙ୍ଗ;
                        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                            ଭାଙ୍ଗ;
                    }

                    variant += std::string("_") + src0_name;
                    ଭାଙ୍ଗ;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(ଚାବି.vectorized ? "VEC" : "SCALAR");

        ଅଚଳ ତାର୍କିକ is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        ଯଦି (ଚାବି.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } ଅନ୍ୟଥା {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        ଯଦି (ଚାବି.use_subgroup_matrix) {
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
        ଯଦି (ଚାବି.vectorized) {
            variant += "_vectorized";
        }

        ଯଦି (ଚାବି.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ଯଦି (!ଚାବି.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = ଚାବି.use_subgroup_matrix;
        ଯଦି (ଚାବି.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_size           = context.max_subgroup_size;
        } ଅନ୍ୟଥା {
            decisions->wg_size_m       = WEBGPU_MUL_MAT_WG_SIZE_M;
            decisions->wg_size_n       = WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->wg_size         = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->mul_mat_wg_size = WEBGPU_MUL_MAT_WG_SIZE;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ mul_mat_fast_pipelines[ଚାବି];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.n_experts                                  = (uint32_t) context.src0->ne[2];

        auto it = mul_mat_id_gather_pipelines.find(ଚାବି);
        ଯଦି (it != mul_mat_id_gather_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(ଚାବି.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        auto processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.src0_type                           = context.src0->type;
        ଚାବି.src1_type                           = context.src1->type;
        ଚାବି.n_experts                           = context.src0->ne[2];
        ଚାବି.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_pipelines.find(ଚାବି);
        ଯଦି (it != mul_mat_id_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        ବାଛ (context.src1->type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ଅଚଳ ସଂରଚନା ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ଅଚଳ ଅକ୍ଷର *                    src0_name   = src0_traits->type_name;

        ବାଛ (context.src0->type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    ବାଛ (context.src0->type) {
                        ସ୍ଥିତି GGML_TYPE_IQ1_S:
                        ସ୍ଥିତି GGML_TYPE_IQ1_M:
                        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
                        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
                        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
                        ସ୍ଥିତି GGML_TYPE_IQ2_S:
                        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
                        ସ୍ଥିତି GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_MXFP4:
                        ସ୍ଥିତି GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ଭାଙ୍ଗ;
                        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                            ଭାଙ୍ଗ;
                    }

                    variant += std::string("_") + src0_name;
                    ଭାଙ୍ଗ;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(ଚାବି.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ଅଚଳ uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        ଯଦି (ଚାବି.vectorized) {
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
        mul_mat_id_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ mul_mat_id_pipelines[ଚାବି];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.src0_type                           = context.src0->type;
        ଚାବି.src1_type                           = context.src1->type;
        ଚାବି.n_experts                           = context.src0->ne[2];
        ଚାବି.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_vec_pipelines.find(ଚାବି);
        ଯଦି (it != mul_mat_id_vec_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        ଅଚଳ ଅକ୍ଷର *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        ବାଛ (context.src1->type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ବାଛ (context.src0->type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ଅଚଳ ସଂରଚନା ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    ବାଛ (context.src0->type) {
                        ସ୍ଥିତି GGML_TYPE_IQ1_S:
                        ସ୍ଥିତି GGML_TYPE_IQ1_M:
                        ସ୍ଥିତି GGML_TYPE_IQ2_S:
                        ସ୍ଥିତି GGML_TYPE_IQ3_S:
                        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
                        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
                        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
                        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            ଭାଙ୍ଗ;
                        ସ୍ଥିତି GGML_TYPE_MXFP4:
                        ସ୍ଥିତି GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            ଭାଙ୍ଗ;
                        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                            ଭାଙ୍ଗ;
                    }
                    ଭାଙ୍ଗ;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(ଚାବି.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        ଯଦି (ଚାବି.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        ଯଦି (ଚାବି.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(ଚାବି.n_experts));

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ mul_mat_id_vec_pipelines[ଚାବି];
    }

    webgpu_pipeline get_unary_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ଅଚଳ ତାର୍କିକ                     is_unary = context.dst->op == GGML_OP_UNARY;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                      op       = is_unary ? (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_ଚାବି ଚାବି      = {};
        ଚାବି.type                                = context.dst->type;
        ଚାବି.op                                  = op;
        ଚାବି.is_unary                            = is_unary;
        ଚାବି.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        ଚାବି.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        auto it = unary_pipelines.find(ଚାବି);
        ଯଦି (it != unary_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            ଚାବି.is_unary ? ggml_unary_op_name((ggml_unary_op) ଚାବି.op) : ggml_op_name((ggml_op) ଚାବି.op);
        defines.push_back(variant);

        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for unary shader");
        }

        ଯଦି (ଚାବି.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        ଯଦି (op == GGML_OP_TRI) {
            ବାଛ (ଚାବି.ttype) {
                ସ୍ଥିତି GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_unary, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ଚାବି.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[ଚାବି]     = pipeline;
        ଫେରନ୍ତୁ unary_pipelines[ଚାବି];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, ତାର୍କିକ src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ଚାବି.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ଚାବି.src_overlap                           = src_overlap;

        auto it = rms_norm_mul_pipelines.find(ଚାବି);
        ଯଦି (it != rms_norm_mul_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        ଯଦି (ଚାବି.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = ଚାବି.inplace;
        pipeline_decisions->overlap     = ଚାବି.overlap;
        pipeline_decisions->src_overlap = ଚାବି.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[ଚାବି]     = pipeline;
        ଫେରନ୍ତୁ rms_norm_mul_pipelines[ଚାବି];
    }

    webgpu_pipeline get_ଦ୍ୱିମିକ_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, ତାର୍କିକ src_overlap) {
        ggml_webgpu_ଦ୍ୱିମିକ_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.type                            = context.dst->type;
        ଚାବି.op                              = context.dst->op;
        ଚାବି.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ଚାବି.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ଚାବି.src_overlap                     = src_overlap;

        auto it = ଦ୍ୱିମିକ_pipelines.find(ଚାବି);
        ଯଦି (it != ଦ୍ୱିମିକ_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) ଚାବି.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for binary shader");
        }

        ଯଦି (ଚାବି.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_ଦ୍ୱିମିକ, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_ଦ୍ୱିମିକ_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = ଚାବି.inplace;
        pipeline_decisions->overlap     = ଚାବି.overlap;
        pipeline_decisions->src_overlap = ଚାବି.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ଦ୍ୱିମିକ_pipelines[ଚାବି]    = pipeline;
        ଫେରନ୍ତୁ ଦ୍ୱିମିକ_pipelines[ଚାବି];
    }

    webgpu_pipeline get_add_id_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = add_id_pipelines.find(ଚାବି);
        ଯଦି (it != add_id_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        ଅଚଳ ଅକ୍ଷର *             shader_src = wgsl_add_id;

        ଯଦି (ଚାବି.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed              = preprocessor.preprocess(shader_src, defines);
        auto pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_size = context.max_wg_size;
        pipeline_decisions->inplace = ଚାବି.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[ଚାବି]    = pipeline;
        ଫେରନ୍ତୁ pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, ତାର୍କିକ src_overlap) {
        ggml_webgpu_concat_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.type                            = context.dst->type;
        ଚାବି.src_overlap                     = src_overlap;

        auto it = concat_pipelines.find(ଚାବି);
        ଯଦି (it != concat_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for concat shader");
        }

        ଯଦି (ଚାବି.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_concat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_ଦ୍ୱିମିକ_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = ଚାବି.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[ଚାବି]    = pipeline;
        ଫେରନ୍ତୁ concat_pipelines[ଚାବି];
    }

    webgpu_pipeline get_repeat_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.type                            = context.dst->type;

        auto it = repeat_pipelines.find(ଚାବି);
        ଯଦି (it != repeat_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_repeat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[ଚାବି]    = pipeline;
        ଫେରନ୍ତୁ repeat_pipelines[ଚାବି];
    }

    webgpu_pipeline get_flash_attn_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, ତାର୍କିକ kv_overlap) {
        ଅଚଳ ତାର୍କିକ can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.common                              = ggml_webgpu_flash_attn_make_common_pipeline_ଚାବି(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        ଚାବି.common.k_direct &= decisions.use_sg_matrix && ଚାବି.common.k_type == GGML_TYPE_F16;
        ଚାବି.common.v_direct &= decisions.use_sg_matrix && ଚାବି.common.v_type == GGML_TYPE_F16;
        ଚାବି.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_limit_bytes, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            ଚାବି.common.head_dim_qk, ଚାବି.common.head_dim_v, ଚାବି.common.has_mask,
            ଚାବି.common.k_direct || ଚାବି.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        ଯଦି (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = ମିଥ୍ୟା;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            ଚାବି.common.k_direct     = ମିଥ୍ୟା;
            ଚାବି.common.v_direct     = ମିଥ୍ୟା;
            ଚାବି.use_sg_matrix       = ମିଥ୍ୟା;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, ଚାବି.common.head_dim_qk, ଚାବି.common.head_dim_v,
                ଚାବି.common.has_mask, ଚାବି.common.k_direct || ଚାବି.common.v_direct);
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

        ଯଦି (ଚାବି.common.k_direct || ଚାବି.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            ଯେତେବେଳେ (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_size;
            }
        }

        auto it = flash_attn_pipelines.find(ଚାବି);
        ଯଦି (it != flash_attn_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(ଚାବି.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_size);
        ଅଚଳ ଅକ୍ଷର *             shader_src = nullptr;
        ଯଦି (!ଚାବି.use_sg_matrix) {
            shader_src = wgsl_flash_attn_tile;
            defines.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_size) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_size) + "_" +
                       std::to_string(context.max_subgroup_size);
        } ଅନ୍ୟଥା {
            shader_src = wgsl_flash_attn;
            defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ flash_attn_pipelines[ଚାବି];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, ତାର୍କିକ kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.common = ggml_webgpu_flash_attn_make_common_pipeline_ଚାବି(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        auto it = flash_attn_vec_pipelines.find(ଚାବି);
        ଯଦି (it != flash_attn_vec_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, ଚାବି.common.head_dim_qk,
                                                                   ଚାବି.common.head_dim_v, ଚାବି.common.has_mask,
                                                                   ଚାବି.common.k_direct || ଚାବି.common.v_direct);
        decisions.wg_size = context.max_subgroup_size;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(ଚାବି.common, variant, 1u, decisions.kv_tile, decisions.wg_size);
        ଯଦି (ଚାବି.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (ଆକାର("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_size;
        ଅଚଳ uint32_t D       = ଚାବି.common.head_dim_qk | ଚାବି.common.head_dim_v;
        ଅଚଳ uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_size, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ flash_attn_vec_pipelines[ଚାବି];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.kv_tile                                 = kv_tile;
        auto it                                     = flash_attn_blk_pipelines.find(ଚାବି);
        ଯଦି (it != flash_attn_blk_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(ଚାବି.kv_tile));
        variant += std::string("_kvt") + std::to_string(ଚାବି.kv_tile);

        uint32_t wg_size = 1;
        ଯେତେବେଳେ ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ flash_attn_blk_pipelines[ଚାବି];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        ଚାବି.dst_type                                       = context.dst->type;
        ଚାବି.wg_size                                        = context.max_wg_size;
        auto it                                            = flash_attn_vec_reduce_pipelines.find(ଚାବି);
        ଯଦି (it != flash_attn_vec_reduce_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        ବାଛ (ଚାବି.dst_type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("DST_F32");
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("DST_F16");
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(ଚାବି.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(ଚାବି.head_dim_v));
        variant += std::string("_hsv") + std::to_string(ଚାବି.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ flash_attn_vec_reduce_pipelines[ଚାବି];
    }

    webgpu_pipeline get_cpy_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.src_type                     = context.src0->type;
        ଚାବି.dst_type                     = context.dst->type;

        auto it = cpy_pipelines.find(ଚାବି);
        ଯଦି (it != cpy_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        ବାଛ (ଚାବି.src_type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_I32:
                defines.push_back("SRC_I32");
                variant += "_i32";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        ବାଛ (ଚାବି.dst_type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_cpy, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[ଚାବି]       = pipeline;
        ଫେରନ୍ତୁ cpy_pipelines[ଚାବି];
    }

    webgpu_pipeline get_glu_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, ତାର୍କିକ src_overlap) {
        ggml_webgpu_glu_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.glu_op                       = ggml_get_glu_op(context.dst);
        ଚାବି.type                         = context.dst->type;
        ଚାବି.split                        = (context.src1 != nullptr);
        ଚାବି.src_overlap                  = src_overlap;

        auto it = glu_pipelines.find(ଚାବି);
        ଯଦି (it != glu_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        ବାଛ (ଚାବି.glu_op) {
            ସ୍ଥିତି GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_GLU_OP_SWIGLU_CLAMP:
                defines.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported GLU op");
        }
        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        ଯଦି (ଚାବି.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } ଅନ୍ୟଥା ଯଦି (ଚାବି.split) {
            variant += "_split";
        } ଅନ୍ୟଥା {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_glu, defines);
        auto decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = ଚାବି.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[ଚାବି]       = pipeline;
        ଫେରନ୍ତୁ glu_pipelines[ଚାବି];
    }

    webgpu_pipeline get_rope_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.type                          = context.dst->type;
        ଚାବି.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ଚାବି.has_ff                        = (context.src2 != nullptr);

        auto it = rope_pipelines.find(ଚାବି);
        ଯଦି (it != rope_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        ବାଛ (ଚାବି.type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        ଯଦି (ଚାବି.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        ଯଦି (ଚାବି.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_rope, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ଚାବି.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[ଚାବି]      = pipeline;
        ଫେରନ୍ତୁ rope_pipelines[ଚାବି];
    }

    webgpu_pipeline get_soft_max_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        ଚାବି.has_mask                          = (context.src1 != nullptr);
        ଚାବି.has_sink                          = (context.src2 != nullptr);
        ଚାବି.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = soft_max_pipelines.find(ଚାବି);
        ଯଦି (it != soft_max_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        ଯଦି (ଚାବି.has_mask) {
            defines.push_back("HAS_MASK");
            ବାଛ (ଚାବି.mask_type) {
                ସ୍ଥିତି GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        ଯଦି (ଚାବି.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        ଯଦି (ଚାବି.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ଚାବି.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[ଚାବି]  = pipeline;
        ଫେରନ୍ତୁ soft_max_pipelines[ଚାବି];
    }

    webgpu_pipeline get_conv2d_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.weight_type                     = context.src0->type;
        ଚାବି.ନିବେଶ_type                      = context.src1->type;
        ଚାବି.output_type                     = context.dst->type;

        auto it = conv2d_pipelines.find(ଚାବି);
        ଯଦି (it != conv2d_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        auto ଠେଲ_type_defines = [&](ଅଚଳ ଅକ୍ଷର * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            ଯଦି (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } ଅନ୍ୟଥା ଯଦି (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } ଅନ୍ୟଥା {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ଠେଲ_type_defines("WEIGHT_TYPE", ଚାବି.weight_type);
        ଠେଲ_type_defines("INPUT_TYPE", ଚାବି.ନିବେଶ_type);
        ଠେଲ_type_defines("OUTPUT_TYPE", ଚାବି.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[ଚାବି]    = pipeline;
        ଫେରନ୍ତୁ conv2d_pipelines[ଚାବି];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context, ତାର୍କିକ whcn) {
        ggml_webgpu_conv2d_dw_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.weight_type                        = context.src0->type;
        ଚାବି.ନିବେଶ_type                         = context.src1->type;
        ଚାବି.output_type                        = context.dst->type;
        ଚାବି.whcn                               = whcn;

        auto it = conv2d_dw_pipelines.find(ଚାବି);
        ଯଦି (it != conv2d_dw_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        auto ଠେଲ_type_defines = [&](ଅଚଳ ଅକ୍ଷର * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            ଯଦି (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } ଅନ୍ୟଥା ଯଦି (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } ଅନ୍ୟଥା {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ଠେଲ_type_defines("WEIGHT_TYPE", ଚାବି.weight_type);
        ଠେଲ_type_defines("INPUT_TYPE", ଚାବି.ନିବେଶ_type);
        ଠେଲ_type_defines("OUTPUT_TYPE", ଚାବି.output_type);

        ଯଦି (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[ଚାବି] = pipeline;
        ଫେରନ୍ତୁ conv2d_dw_pipelines[ଚାବି];
    }

    webgpu_pipeline get_im2col_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.ନିବେଶ_type                      = context.src1->type;
        ଚାବି.output_type                     = context.dst->type;

        auto it = im2col_pipelines.find(ଚାବି);
        ଯଦି (it != im2col_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        auto ଠେଲ_type_defines = [&](ଅଚଳ ଅକ୍ଷର * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            ଯଦି (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } ଅନ୍ୟଥା ଯଦି (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } ଅନ୍ୟଥା {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ଠେଲ_type_defines("INPUT_TYPE", ଚାବି.ନିବେଶ_type);
        ଠେଲ_type_defines("OUTPUT_TYPE", ଚାବି.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[ଚାବି]    = pipeline;
        ଫେରନ୍ତୁ im2col_pipelines[ଚାବି];
    }

    webgpu_pipeline get_upscale_pipeline(ଅଚଳ ggml_webgpu_shader_lib_context & context) {
        ଅଚଳ uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ଅଚଳ uint32_t base_mode  = mode_flags & 0xFFu;
        ଅଚଳ ତାର୍କିକ     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_ଚାବି ଚାବି = {};
        ଚାବି.ନିବେଶ_type                       = context.src0->type;
        ଚାବି.output_type                      = context.dst->type;
        ଚାବି.base_mode                        = base_mode;
        ଚାବି.antialias                        = antialias;

        auto it = upscale_pipelines.find(ଚାବି);
        ଯଦି (it != upscale_pipelines.end()) {
            ଫେରନ୍ତୁ it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        ଯଦି (ଚାବି.ନିବେଶ_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } ଅନ୍ୟଥା {
            variant += "_src_f32";
        }

        ଯଦି (ଚାବି.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } ଅନ୍ୟଥା {
            variant += "_dst_f32";
        }

        ବାଛ (base_mode) {
            ସ୍ଥିତି GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("Unsupported upscale mode");
        }

        ଯଦି (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_upscale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[ଚାବି]   = pipeline;
        ଫେରନ୍ତୁ upscale_pipelines[ଚାବି];
    }

  ବ୍ୟକ୍ତିଗତ:
    ସ୍ଥିର webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
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
        ଫେରନ୍ତୁ { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
