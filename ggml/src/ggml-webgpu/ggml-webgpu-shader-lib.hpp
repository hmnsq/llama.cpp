#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-wgsl-shaders.hpp"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "pre_wgsl.hpp"

#અન્તર્ભાવ <webgpu/webgpu_cpp.h>

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <unordered_map>
#અન્તર્ભાવ <vector>

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
template <typename T> અંતઃરેખ શૂન્ય ggml_webgpu_hash_combine(size_t & seed, અચલ T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
અંતઃરેખ uintptr_t ggml_webgpu_tensor_addr(અચલ ggml_tensor * tensor) {
    અચલ ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    પાછા (uintptr_t) base_tensor->data + tensor->view_offs;
}

અંતઃરેખ તાર્કિક ggml_webgpu_tensor_equal(અચલ ggml_tensor * a, અચલ ggml_tensor * b) {
    પાછા a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

સંરચના ggml_webgpu_shader_lib_context {
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
    તાર્કિક        supports_subgroups       = ખોટું;
    તાર્કિક        supports_subgroup_matrix = ખોટું;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_size        = 0;
    uint32_t    max_subgroup_size        = 0;
    તાર્કિક        supports_dot_product     = ખોટું;
    std::string vendor;
};

સંરચના webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<શૂન્ય> context = nullptr;
};

સંરચના ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 0;
    તાર્કિક     inplace = ખોટું;
};

સંરચના ggml_webgpu_get_rows_shader_decisions {
    uint32_t wg_size    = 0;
    તાર્કિક     vectorized = ખોટું;
};

સંરચના ggml_webgpu_દ્વિઅંકી_shader_decisions {
    uint32_t wg_size     = 0;
    તાર્કિક     inplace     = ખોટું;
    તાર્કિક     overlap     = ખોટું;
    તાર્કિક     src_overlap = ખોટું;
};

સંરચના ggml_webgpu_glu_shader_decisions {
    uint32_t wg_size     = 0;
    તાર્કિક     src_overlap = ખોટું;
};

સંરચના ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<શૂન્ય> decisions;
};

સંરચના ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_size;
    uint32_t tokens_per_wg;
};

સંરચના ggml_webgpu_ssm_scan_pipeline_ચાવી {
    પૂર્ણાંક     type;
    પૂર્ણાંક     d_state;
    uint8_t xbc_overlap;
    તાર્કિક    a_overlap;
    તાર્કિક    ids_overlap;

    તાર્કિક operator==(અચલ ggml_webgpu_ssm_scan_pipeline_ચાવી & other) અચલ {
        પાછા type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

સંરચના ggml_webgpu_ssm_scan_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_ssm_scan_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        ggml_webgpu_hash_combine(seed, ચાવી.d_state);
        ggml_webgpu_hash_combine(seed, ચાવી.xbc_overlap);
        ggml_webgpu_hash_combine(seed, ચાવી.a_overlap);
        ggml_webgpu_hash_combine(seed, ચાવી.ids_overlap);
        પાછા seed;
    }
};

સંરચના ggml_webgpu_ssm_scan_shader_decisions {
    uint32_t wg_size;
    uint32_t tokens_per_tile;
    uint8_t  xbc_overlap = 0;
    તાર્કિક     a_overlap   = ખોટું;
    તાર્કિક     ids_overlap = ખોટું;
};

/** Argsort **/

સંરચના ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes;
    int32_t  order;
};

/** Set Rows **/

સંરચના ggml_webgpu_set_rows_pipeline_ચાવી {
    પૂર્ણાંક src0_type;
    પૂર્ણાંક dst_type;
    પૂર્ણાંક vec4;
    પૂર્ણાંક i64_idx;
    પૂર્ણાંક pair_blocks;

    તાર્કિક operator==(અચલ ggml_webgpu_set_rows_pipeline_ચાવી & other) અચલ {
        પાછા src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_blocks == other.pair_blocks;
    }
};

સંરચના ggml_webgpu_set_rows_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_set_rows_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.src0_type);
        ggml_webgpu_hash_combine(seed, ચાવી.dst_type);
        ggml_webgpu_hash_combine(seed, ચાવી.vec4);
        ggml_webgpu_hash_combine(seed, ચાવી.i64_idx);
        ggml_webgpu_hash_combine(seed, ચાવી.pair_blocks);
        પાછા seed;
    }
};

સંરચના ggml_webgpu_set_rows_shader_decisions {
    તાર્કિક     vec4;
    તાર્કિક     i64_idx;
    તાર્કિક     pair_blocks;
    uint32_t wg_size;
};

/** Set **/

સંરચના ggml_webgpu_set_pipeline_ચાવી {
    ggml_type type;
    તાર્કિક      inplace;

    તાર્કિક operator==(અચલ ggml_webgpu_set_pipeline_ચાવી & other) અચલ {
        પાછા type == other.type && inplace == other.inplace;
    }
};

સંરચના ggml_webgpu_set_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_set_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        ggml_webgpu_hash_combine(seed, ચાવી.inplace);
        પાછા seed;
    }
};

/** Get Rows **/

સંરચના ggml_webgpu_get_rows_pipeline_ચાવી {
    ggml_type src_type;
    પૂર્ણાંક       vectorized;

    તાર્કિક operator==(અચલ ggml_webgpu_get_rows_pipeline_ચાવી & other) અચલ {
        પાછા src_type == other.src_type && vectorized == other.vectorized;
    }
};

સંરચના ggml_webgpu_get_rows_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_get_rows_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.src_type);
        ggml_webgpu_hash_combine(seed, ચાવી.vectorized);
        પાછા seed;
    }
};

/** Row Norm **/

સંરચના ggml_webgpu_row_norm_pipeline_ચાવી {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    તાર્કિક      inplace;

    તાર્કિક operator==(અચલ ggml_webgpu_row_norm_pipeline_ચાવી & other) અચલ {
        પાછા op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

સંરચના ggml_webgpu_row_norm_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_row_norm_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.op);
        ggml_webgpu_hash_combine(seed, ચાવી.src_type);
        ggml_webgpu_hash_combine(seed, ચાવી.dst_type);
        ggml_webgpu_hash_combine(seed, ચાવી.inplace);
        પાછા seed;
    }
};

/** RMS_NORM + MUL **/

સંરચના ggml_webgpu_rms_norm_mul_pipeline_ચાવી {
    તાર્કિક inplace;      // rn_src == dst
    તાર્કિક overlap;      // mul_src == dst
    તાર્કિક src_overlap;  // rn_src binding overlaps mul_src binding

    તાર્કિક operator==(અચલ ggml_webgpu_rms_norm_mul_pipeline_ચાવી & other) અચલ {
        પાછા inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

સંરચના ggml_webgpu_rms_norm_mul_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_rms_norm_mul_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.inplace);
        ggml_webgpu_hash_combine(seed, ચાવી.overlap);
        ggml_webgpu_hash_combine(seed, ચાવી.src_overlap);
        પાછા seed;
    }
};

સંરચના ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    તાર્કિક     inplace     = ખોટું;
    તાર્કિક     overlap     = ખોટું;
    તાર્કિક     src_overlap = ખોટું;
};

/** Pad **/
સંરચના ggml_webgpu_pad_pipeline_ચાવી {
    તાર્કિક circular;

    તાર્કિક operator==(અચલ ggml_webgpu_pad_pipeline_ચાવી & other) અચલ { પાછા circular == other.circular; }
};

સંરચના ggml_webgpu_pad_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_pad_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.circular);
        પાછા seed;
    }
};

/** Solve Tri **/
સંરચના ggml_webgpu_solve_tri_pipeline_ચાવી {
    પૂર્ણાંક type;
    પૂર્ણાંક n;
    પૂર્ણાંક k;

    તાર્કિક operator==(અચલ ggml_webgpu_solve_tri_pipeline_ચાવી & other) અચલ {
        પાછા type == other.type && n == other.n && k == other.k;
    }
};

સંરચના ggml_webgpu_solve_tri_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_solve_tri_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        ggml_webgpu_hash_combine(seed, ચાવી.n);
        ggml_webgpu_hash_combine(seed, ચાવી.k);
        પાછા seed;
    }
};

/** SSM Conv **/
સંરચના ggml_webgpu_ssm_conv_pipeline_ચાવી {
    પૂર્ણાંક type;
    પૂર્ણાંક vectorized;

    તાર્કિક operator==(અચલ ggml_webgpu_ssm_conv_pipeline_ચાવી & other) અચલ {
        પાછા type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
સંરચના ggml_webgpu_conv2d_pipeline_ચાવી {
    ggml_type weight_type;
    ggml_type આગત_type;
    ggml_type output_type;

    તાર્કિક operator==(અચલ ggml_webgpu_conv2d_pipeline_ચાવી & other) અચલ {
        પાછા weight_type == other.weight_type && આગત_type == other.આગત_type && output_type == other.output_type;
    }
};

સંરચના ggml_webgpu_conv2d_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_conv2d_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.weight_type);
        ggml_webgpu_hash_combine(seed, ચાવી.આગત_type);
        ggml_webgpu_hash_combine(seed, ચાવી.output_type);
        પાછા seed;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
સંરચના ggml_webgpu_conv2d_dw_pipeline_ચાવી {
    ggml_type weight_type;
    ggml_type આગત_type;
    ggml_type output_type;
    તાર્કિક      whcn;

    તાર્કિક operator==(અચલ ggml_webgpu_conv2d_dw_pipeline_ચાવી & other) અચલ {
        પાછા weight_type == other.weight_type && આગત_type == other.આગત_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

સંરચના ggml_webgpu_conv2d_dw_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_conv2d_dw_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.weight_type);
        ggml_webgpu_hash_combine(seed, ચાવી.આગત_type);
        ggml_webgpu_hash_combine(seed, ચાવી.output_type);
        ggml_webgpu_hash_combine(seed, ચાવી.whcn);
        પાછા seed;
    }
};

/** Im2Col **/
સંરચના ggml_webgpu_im2col_pipeline_ચાવી {
    ggml_type આગત_type;
    ggml_type output_type;

    તાર્કિક operator==(અચલ ggml_webgpu_im2col_pipeline_ચાવી & other) અચલ {
        પાછા આગત_type == other.આગત_type && output_type == other.output_type;
    }
};

સંરચના ggml_webgpu_im2col_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_im2col_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.આગત_type);
        ggml_webgpu_hash_combine(seed, ચાવી.output_type);
        પાછા seed;
    }
};

/** Gated Delta Net **/
સંરચના ggml_webgpu_gated_delta_net_pipeline_ચાવી {
    પૂર્ણાંક  type;
    પૂર્ણાંક  s_v;
    પૂર્ણાંક  kda;
    તાર્કિક fused_cache;

    તાર્કિક operator==(અચલ ggml_webgpu_gated_delta_net_pipeline_ચાવી & other) અચલ {
        પાછા type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

સંરચના ggml_webgpu_gated_delta_net_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_gated_delta_net_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        ggml_webgpu_hash_combine(seed, ચાવી.s_v);
        ggml_webgpu_hash_combine(seed, ચાવી.kda);
        પાછા seed;
    }
};

સંરચના ggml_webgpu_ssm_conv_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_ssm_conv_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        ggml_webgpu_hash_combine(seed, ચાવી.vectorized);
        પાછા seed;
    }
};

/** Scale **/

સંરચના ggml_webgpu_scale_pipeline_ચાવી {
    પૂર્ણાંક inplace;

    તાર્કિક operator==(અચલ ggml_webgpu_scale_pipeline_ચાવી & other) અચલ { પાછા inplace == other.inplace; }
};

સંરચના ggml_webgpu_scale_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_scale_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.inplace);
        પાછા seed;
    }
};

/** Upscale **/

સંરચના ggml_webgpu_upscale_pipeline_ચાવી {
    ggml_type આગત_type;
    ggml_type output_type;
    uint32_t  base_mode;
    તાર્કિક      antialias;

    તાર્કિક operator==(અચલ ggml_webgpu_upscale_pipeline_ચાવી & other) અચલ {
        પાછા આગત_type == other.આગત_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

સંરચના ggml_webgpu_upscale_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_upscale_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.આગત_type);
        ggml_webgpu_hash_combine(seed, ચાવી.output_type);
        ggml_webgpu_hash_combine(seed, ચાવી.base_mode);
        ggml_webgpu_hash_combine(seed, ચાવી.antialias);
        પાછા seed;
    }
};

/** Concat **/

સંરચના ggml_webgpu_concat_pipeline_ચાવી {
    પૂર્ણાંક  type;
    તાર્કિક src_overlap;

    તાર્કિક operator==(અચલ ggml_webgpu_concat_pipeline_ચાવી & other) અચલ {
        પાછા type == other.type && src_overlap == other.src_overlap;
    }
};

સંરચના ggml_webgpu_concat_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_concat_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        ggml_webgpu_hash_combine(seed, ચાવી.src_overlap);
        પાછા seed;
    }
};

/** Repeat **/

સંરચના ggml_webgpu_repeat_pipeline_ચાવી {
    પૂર્ણાંક type;

    તાર્કિક operator==(અચલ ggml_webgpu_repeat_pipeline_ચાવી & other) અચલ { પાછા type == other.type; }
};

સંરચના ggml_webgpu_repeat_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_repeat_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        પાછા seed;
    }
};

/** Binary **/

સંરચના ggml_webgpu_દ્વિઅંકી_pipeline_ચાવી {
    પૂર્ણાંક  type;
    પૂર્ણાંક  op;
    તાર્કિક inplace;
    તાર્કિક overlap;
    તાર્કિક src_overlap;

    તાર્કિક operator==(અચલ ggml_webgpu_દ્વિઅંકી_pipeline_ચાવી & other) અચલ {
        પાછા type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

સંરચના ggml_webgpu_દ્વિઅંકી_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_દ્વિઅંકી_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        ggml_webgpu_hash_combine(seed, ચાવી.op);
        ggml_webgpu_hash_combine(seed, ચાવી.inplace);
        ggml_webgpu_hash_combine(seed, ચાવી.overlap);
        ggml_webgpu_hash_combine(seed, ચાવી.src_overlap);
        પાછા seed;
    }
};

/* Add_Id */

સંરચના ggml_webgpu_add_id_pipeline_ચાવી {
    તાર્કિક inplace;

    તાર્કિક operator==(અચલ ggml_webgpu_add_id_pipeline_ચાવી & other) અચલ { પાછા inplace == other.inplace; }
};

સંરચના ggml_webgpu_add_id_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_add_id_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.inplace);
        પાછા seed;
    }
};

/** Unary **/

સંરચના ggml_webgpu_unary_pipeline_ચાવી {
    પૂર્ણાંક           type;
    પૂર્ણાંક           op;
    તાર્કિક          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    તાર્કિક          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    તાર્કિક operator==(અચલ ggml_webgpu_unary_pipeline_ચાવી & other) અચલ {
        પાછા type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

સંરચના ggml_webgpu_unary_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_unary_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        ggml_webgpu_hash_combine(seed, ચાવી.op);
        ggml_webgpu_hash_combine(seed, ચાવી.is_unary);
        ggml_webgpu_hash_combine(seed, ચાવી.inplace);
        ggml_webgpu_hash_combine(seed, ચાવી.ttype);
        પાછા seed;
    }
};

/** FlashAttention */

સંરચના ggml_webgpu_flash_attn_common_pipeline_ચાવી {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    તાર્કિક      k_direct;
    તાર્કિક      v_direct;
    તાર્કિક      kv_overlap;
    તાર્કિક      has_mask;
    તાર્કિક      has_sinks;
    તાર્કિક      uses_logit_softcap;

    તાર્કિક operator==(અચલ ggml_webgpu_flash_attn_common_pipeline_ચાવી & other) અચલ {
        પાછા q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               k_direct == other.k_direct && v_direct == other.v_direct && kv_overlap == other.kv_overlap &&
               has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

અંતઃરેખ શૂન્ય ggml_webgpu_flash_attn_hash_common_pipeline_ચાવી(size_t &                                           seed,
                                                            અચલ ggml_webgpu_flash_attn_common_pipeline_ચાવી & ચાવી) {
    ggml_webgpu_hash_combine(seed, ચાવી.q_type);
    ggml_webgpu_hash_combine(seed, ચાવી.k_type);
    ggml_webgpu_hash_combine(seed, ચાવી.v_type);
    ggml_webgpu_hash_combine(seed, ચાવી.dst_type);
    ggml_webgpu_hash_combine(seed, ચાવી.head_dim_qk);
    ggml_webgpu_hash_combine(seed, ચાવી.head_dim_v);
    ggml_webgpu_hash_combine(seed, ચાવી.k_direct);
    ggml_webgpu_hash_combine(seed, ચાવી.v_direct);
    ggml_webgpu_hash_combine(seed, ચાવી.kv_overlap);
    ggml_webgpu_hash_combine(seed, ચાવી.has_mask);
    ggml_webgpu_hash_combine(seed, ચાવી.has_sinks);
    ggml_webgpu_hash_combine(seed, ચાવી.uses_logit_softcap);
}

સંરચના ggml_webgpu_flash_attn_vec_pipeline_ચાવી {
    ggml_webgpu_flash_attn_common_pipeline_ચાવી common;

    તાર્કિક operator==(અચલ ggml_webgpu_flash_attn_vec_pipeline_ચાવી & other) અચલ { પાછા common == other.common; }
};

સંરચના ggml_webgpu_flash_attn_vec_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_flash_attn_vec_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ચાવી(seed, ચાવી.common);
        પાછા seed;
    }
};

સંરચના ggml_webgpu_flash_attn_pipeline_ચાવી {
    ggml_webgpu_flash_attn_common_pipeline_ચાવી common;
    તાર્કિક                                       use_sg_matrix;

    તાર્કિક operator==(અચલ ggml_webgpu_flash_attn_pipeline_ચાવી & other) અચલ {
        પાછા common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

સંરચના ggml_webgpu_flash_attn_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_flash_attn_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ચાવી(seed, ચાવી.common);
        ggml_webgpu_hash_combine(seed, ચાવી.use_sg_matrix);
        પાછા seed;
    }
};

સંરચના ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

સંરચના ggml_webgpu_flash_attn_decisions {
    તાર્કિક     use_sg_matrix = ખોટું;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_size       = 0;
};

અંતઃરેખ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
અંતઃરેખ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

અંતઃરેખ size_t ggml_webgpu_flash_attn_tensor_offset(અચલ ggml_tensor * tensor) {
    constexpr uintptr_t ptr_base_addr = 0x1000u;
    અચલ ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
    પાછા reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
}

અંતઃરેખ તાર્કિક ggml_webgpu_flash_attn_float_vec4_aligned(અચલ ggml_tensor * K, size_t storage_offset_alignment) {
    અચલ uint32_t offset_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_offset(K) & (storage_offset_alignment - 1)) /
                    ggml_type_size(K->type));
    પાછા offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u;
}

અંતઃરેખ તાર્કિક ggml_webgpu_flash_attn_float_vec4_aligned(અચલ ggml_tensor * K,
                                                      અચલ ggml_tensor * V,
                                                      size_t              storage_offset_alignment) {
    પાછા ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);
}

અંતઃરેખ તાર્કિક ggml_webgpu_flash_attn_k_direct(અચલ ggml_tensor * Q, અચલ ggml_tensor * K, uint32_t kv_direct_align) {
    પાછા (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_direct_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

અંતઃરેખ તાર્કિક ggml_webgpu_flash_attn_v_direct(અચલ ggml_tensor * Q, અચલ ggml_tensor * V, uint32_t kv_direct_align) {
    પાછા ggml_webgpu_flash_attn_k_direct(Q, V, kv_direct_align);
}

અંતઃરેખ ggml_webgpu_flash_attn_common_pipeline_ચાવી ggml_webgpu_flash_attn_make_common_pipeline_ચાવી(
    અચલ ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_direct_align,
    તાર્કિક                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_ચાવી ચાવી = {};
    ચાવી.q_type                                     = context.src0->type;
    ચાવી.k_type                                     = context.src1->type;
    ચાવી.v_type                                     = context.src2->type;
    ચાવી.dst_type                                   = context.dst->type;
    ચાવી.head_dim_qk                                = (uint32_t) context.src0->ne[0];
    ચાવી.head_dim_v                                 = (uint32_t) context.src2->ne[0];
    ચાવી.k_direct           = ggml_webgpu_flash_attn_k_direct(context.src0, context.src1, kv_direct_align);
    ચાવી.v_direct           = ggml_webgpu_flash_attn_v_direct(context.src0, context.src2, kv_direct_align);
    ચાવી.kv_overlap         = kv_overlap;
    ચાવી.has_mask           = context.src3 != nullptr;
    ચાવી.has_sinks          = context.src4 != nullptr;
    ચાવી.uses_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    પાછા ચાવી;
}

અંતઃરેખ std::vector<std::string> ggml_webgpu_flash_attn_common_defines(
    અચલ ggml_webgpu_flash_attn_common_pipeline_ચાવી & ચાવી,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_size) {
    std::vector<std::string> defines;

    પસંદગી (ચાવી.k_type) {
        સ્થિતિ GGML_TYPE_F32:
            defines.push_back("K_F32");
            તોડો;
        સ્થિતિ GGML_TYPE_F16:
            defines.push_back("K_F16");
            તોડો;
        સ્થિતિ GGML_TYPE_Q4_0:
            defines.push_back("K_Q4_0");
            તોડો;
        સ્થિતિ GGML_TYPE_Q8_0:
            defines.push_back("K_Q8_0");
            તોડો;
        મૂળભૂત:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(ચાવી.k_type);

    પસંદગી (ચાવી.v_type) {
        સ્થિતિ GGML_TYPE_F32:
            defines.push_back("V_F32");
            તોડો;
        સ્થિતિ GGML_TYPE_F16:
            defines.push_back("V_F16");
            તોડો;
        સ્થિતિ GGML_TYPE_Q4_0:
            defines.push_back("V_Q4_0");
            તોડો;
        સ્થિતિ GGML_TYPE_Q8_0:
            defines.push_back("V_Q8_0");
            તોડો;
        મૂળભૂત:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(ચાવી.v_type);

    પસંદગી (ચાવી.q_type) {
        સ્થિતિ GGML_TYPE_F32:
            defines.push_back("Q_F32");
            તોડો;
        સ્થિતિ GGML_TYPE_F16:
            defines.push_back("Q_F16");
            તોડો;
        મૂળભૂત:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(ચાવી.q_type);

    પસંદગી (ચાવી.dst_type) {
        સ્થિતિ GGML_TYPE_F32:
            defines.push_back("DST_F32");
            તોડો;
        સ્થિતિ GGML_TYPE_F16:
            defines.push_back("DST_F16");
            તોડો;
        મૂળભૂત:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(ચાવી.dst_type);

    જો (ચાવી.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    જો (ચાવી.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    જો (ચાવી.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    જો (ચાવી.k_direct) {
        defines.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    જો (ચાવી.v_direct) {
        defines.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    જો (ચાવી.kv_overlap) {
        defines.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(ચાવી.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(ચાવી.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(ચાવી.head_dim_v));
    variant += std::string("_hsv") + std::to_string(ચાવી.head_dim_v);

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    જો (ggml_is_quantized(ચાવી.k_type) || ggml_is_quantized(ચાવી.v_type)) {
        defines.push_back("U32_DEQUANT_HELPERS");
        જો (ggml_is_quantized(ચાવી.k_type)) {
            defines.push_back("LOADERS_QUANTIZED_K");
        }
        જો (ggml_is_quantized(ચાવી.v_type)) {
            defines.push_back("LOADERS_QUANTIZED_V");
        }
    }

    પાછા defines;
}

સંરચના ggml_webgpu_flash_attn_vec_reduce_pipeline_ચાવી {
    uint32_t  head_dim_v;
    uint32_t  wg_size;
    ggml_type dst_type;
};

સંરચના ggml_webgpu_flash_attn_vec_reduce_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_flash_attn_vec_reduce_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.head_dim_v);
        ggml_webgpu_hash_combine(seed, ચાવી.wg_size);
        ggml_webgpu_hash_combine(seed, ચાવી.dst_type);
        પાછા seed;
    }
};

અંતઃરેખ તાર્કિક operator==(અચલ ggml_webgpu_flash_attn_vec_reduce_pipeline_ચાવી & lhs,
                       અચલ ggml_webgpu_flash_attn_vec_reduce_pipeline_ચાવી & rhs) {
    પાછા lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size && lhs.dst_type == rhs.dst_type;
}

સંરચના ggml_webgpu_flash_attn_blk_pipeline_ચાવી {
    uint32_t kv_tile;

    તાર્કિક operator==(અચલ ggml_webgpu_flash_attn_blk_pipeline_ચાવી & other) અચલ { પાછા kv_tile == other.kv_tile; }
};

સંરચના ggml_webgpu_flash_attn_blk_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_flash_attn_blk_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.kv_tile);
        પાછા seed;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
અંતઃરેખ size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  તાર્કિક     has_mask,
                                                  તાર્કિક     kv_direct) {
    અચલ uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * head_dim_qk;        // q_shmem
    જો (!kv_direct) {
        f32_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f32_elems += q_tile * head_dim_v;         // o_shmem
    જો (has_mask) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    પાછા f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

અંતઃરેખ uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   limit_bytes,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t head_dim_qk,
                                                   uint32_t head_dim_v,
                                                   તાર્કિક     has_mask,
                                                   તાર્કિક     kv_direct) {
    અચલ size_t base_q_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 0, head_dim_qk, head_dim_v, has_mask, kv_direct);
    જો (limit_bytes <= base_q_bytes) {
        પાછા 0;
    }
    અચલ size_t one_kv_bytes =
        ggml_webgpu_flash_attn_wg_mem_bytes(q_tile, 1, head_dim_qk, head_dim_v, has_mask, kv_direct);
    અચલ size_t bytes_per_kv = one_kv_bytes - base_q_bytes;
    જો (bytes_per_kv == 0) {
        પાછા 0;
    }
    અચલ size_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    પાછા (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

અંતઃરેખ uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_limit_bytes,
                                                       uint32_t head_dim_qk,
                                                       uint32_t head_dim_v,
                                                       તાર્કિક     has_mask,
                                                       તાર્કિક     kv_direct) {
    અચલ uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_limit_bytes, 1u, 1u, head_dim_qk, head_dim_v, has_mask, kv_direct);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    જો (kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        જ્યારે (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    પાછા kv_tile;
}

અંતઃરેખ તાર્કિક ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(તાર્કિક                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                અચલ ggml_tensor * Q,
                                                                અચલ ggml_tensor * V) {
    પાછા supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

સંરચના ggml_webgpu_mul_mat_vec_pipeline_ચાવી {
    ggml_type src0_type;
    ggml_type src1_type;
    પૂર્ણાંક       vectorized;
    uint32_t  અંક_cols;
    તાર્કિક      use_mmvq;
    તાર્કિક      src_overlap;

    તાર્કિક operator==(અચલ ggml_webgpu_mul_mat_vec_pipeline_ચાવી & other) અચલ {
        પાછા src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               અંક_cols == other.અંક_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

સંરચના ggml_webgpu_mul_mat_vec_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_mul_mat_vec_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.src0_type);
        ggml_webgpu_hash_combine(seed, ચાવી.src1_type);
        ggml_webgpu_hash_combine(seed, ચાવી.vectorized);
        ggml_webgpu_hash_combine(seed, ચાવી.અંક_cols);
        ggml_webgpu_hash_combine(seed, ચાવી.use_mmvq);
        ggml_webgpu_hash_combine(seed, ચાવી.src_overlap);
        પાછા seed;
    }
};

સંરચના ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
    તાર્કિક     src_overlap = ખોટું;
};

સંરચના ggml_webgpu_quantize_q8_pipeline_ચાવી {
    ggml_type src0_type;

    તાર્કિક operator==(અચલ ggml_webgpu_quantize_q8_pipeline_ચાવી & other) અચલ { પાછા src0_type == other.src0_type; }
};

સંરચના ggml_webgpu_quantize_q8_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_quantize_q8_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.src0_type);
        પાછા seed;
    }
};

સંરચના ggml_webgpu_mul_mat_pipeline_ચાવી {
    ggml_type src0_type;
    ggml_type src1_type;
    પૂર્ણાંક       vectorized;
    પૂર્ણાંક       use_subgroup_matrix;
    તાર્કિક      src_overlap;

    તાર્કિક operator==(અચલ ggml_webgpu_mul_mat_pipeline_ચાવી & other) અચલ {
        પાછા src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

સંરચના ggml_webgpu_mul_mat_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_mul_mat_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.src0_type);
        ggml_webgpu_hash_combine(seed, ચાવી.src1_type);
        ggml_webgpu_hash_combine(seed, ચાવી.vectorized);
        ggml_webgpu_hash_combine(seed, ચાવી.use_subgroup_matrix);
        ggml_webgpu_hash_combine(seed, ચાવી.src_overlap);
        પાછા seed;
    }
};

સંરચના ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    પૂર્ણાંક      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_size;
    તાર્કિક     src_overlap = ખોટું;
};

/** MUL_MAT_ID **/

સંરચના ggml_webgpu_mul_mat_id_gather_pipeline_ચાવી {
    uint32_t n_experts;

    તાર્કિક operator==(અચલ ggml_webgpu_mul_mat_id_gather_pipeline_ચાવી & other) અચલ {
        પાછા n_experts == other.n_experts;
    }
};

સંરચના ggml_webgpu_mul_mat_id_gather_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_mul_mat_id_gather_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.n_experts);
        પાછા seed;
    }
};

સંરચના ggml_webgpu_mul_mat_id_pipeline_ચાવી {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  અંક_cols;
    પૂર્ણાંક       vectorized;

    તાર્કિક operator==(અચલ ggml_webgpu_mul_mat_id_pipeline_ચાવી & other) અચલ {
        પાછા src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               અંક_cols == other.અંક_cols && vectorized == other.vectorized;
    }
};

સંરચના ggml_webgpu_mul_mat_id_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_mul_mat_id_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.src0_type);
        ggml_webgpu_hash_combine(seed, ચાવી.src1_type);
        ggml_webgpu_hash_combine(seed, ચાવી.n_experts);
        ggml_webgpu_hash_combine(seed, ચાવી.અંક_cols);
        ggml_webgpu_hash_combine(seed, ચાવી.vectorized);
        પાછા seed;
    }
};

/** Cpy **/

સંરચના ggml_webgpu_cpy_pipeline_ચાવી {
    ggml_type src_type;
    ggml_type dst_type;

    તાર્કિક operator==(અચલ ggml_webgpu_cpy_pipeline_ચાવી & other) અચલ {
        પાછા src_type == other.src_type && dst_type == other.dst_type;
    }
};

સંરચના ggml_webgpu_cpy_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_cpy_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.src_type);
        ggml_webgpu_hash_combine(seed, ચાવી.dst_type);
        પાછા seed;
    }
};

/** Glu **/

સંરચના ggml_webgpu_glu_pipeline_ચાવી {
    ggml_glu_op glu_op;
    ggml_type   type;
    તાર્કિક        split;
    તાર્કિક        src_overlap;

    તાર્કિક operator==(અચલ ggml_webgpu_glu_pipeline_ચાવી & other) અચલ {
        પાછા glu_op == other.glu_op && type == other.type && split == other.split && src_overlap == other.src_overlap;
    }
};

સંરચના ggml_webgpu_glu_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_glu_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.glu_op);
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        ggml_webgpu_hash_combine(seed, ચાવી.split);
        ggml_webgpu_hash_combine(seed, ચાવી.src_overlap);
        પાછા seed;
    }
};

/** Rope **/

સંરચના ggml_webgpu_rope_pipeline_ચાવી {
    ggml_type type;
    તાર્કિક      inplace;
    તાર્કિક      has_ff;

    તાર્કિક operator==(અચલ ggml_webgpu_rope_pipeline_ચાવી & other) અચલ {
        પાછા type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

સંરચના ggml_webgpu_rope_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_rope_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.type);
        ggml_webgpu_hash_combine(seed, ચાવી.inplace);
        ggml_webgpu_hash_combine(seed, ચાવી.has_ff);
        પાછા seed;
    }
};

/** SoftMax **/

સંરચના ggml_webgpu_soft_max_pipeline_ચાવી {
    ggml_type mask_type;
    તાર્કિક      has_mask;
    તાર્કિક      has_sink;
    તાર્કિક      inplace;

    તાર્કિક operator==(અચલ ggml_webgpu_soft_max_pipeline_ચાવી & other) અચલ {
        પાછા mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

સંરચના ggml_webgpu_soft_max_pipeline_ચાવી_hash {
    size_t operator()(અચલ ggml_webgpu_soft_max_pipeline_ચાવી & ચાવી) અચલ {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, ચાવી.mask_type);
        ggml_webgpu_hash_combine(seed, ચાવી.has_mask);
        ggml_webgpu_hash_combine(seed, ચાવી.has_sink);
        ggml_webgpu_hash_combine(seed, ચાવી.inplace);
        પાછા seed;
    }
};

/** MMVQ **/

અંતઃરેખ તાર્કિક ggml_webgpu_can_use_mmvq(અચલ ggml_tensor * src0,
                                     અચલ ggml_tensor * src1,
                                     તાર્કિક                supports_dot_product,
                                     અચલ std::string & vendor) {
    જો (src1->ne[1] <= 4) {
        તાર્કિક supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        જો (supports_dp4a && supports_dot_product) {
            પસંદગી (src1->type) {
                સ્થિતિ GGML_TYPE_F32:
                    પસંદગી (src0->type) {
                        સ્થિતિ GGML_TYPE_Q1_0:
                        સ્થિતિ GGML_TYPE_Q4_0:
                        સ્થિતિ GGML_TYPE_Q4_1:
                        સ્થિતિ GGML_TYPE_Q5_0:
                        સ્થિતિ GGML_TYPE_Q5_1:
                        સ્થિતિ GGML_TYPE_Q8_0:
                        સ્થિતિ GGML_TYPE_MXFP4:
                        સ્થિતિ GGML_TYPE_Q2_K:
                        સ્થિતિ GGML_TYPE_Q3_K:
                        સ્થિતિ GGML_TYPE_Q4_K:
                        સ્થિતિ GGML_TYPE_Q5_K:
                        સ્થિતિ GGML_TYPE_Q6_K:
                            પાછા src0->ne[0] % 4 == 0;
                        મૂળભૂત:
                            તોડો;
                    }
                    તોડો;
                મૂળભૂત:
                    તોડો;
            }
        }
    }
    પાછા ખોટું;
}

વર્ગ ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<પૂર્ણાંક, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<પૂર્ણાંક, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<પૂર્ણાંક, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<પૂર્ણાંક, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<પૂર્ણાંક, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_ચાવી_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_ચાવી_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_unary_pipeline_ચાવી_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_scale_pipeline_ચાવી_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_ચાવી_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_ચાવી_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_ચાવી_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_ચાવી,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_ચાવી_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_pad_pipeline_ચાવી_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_દ્વિઅંકી_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_દ્વિઅંકી_pipeline_ચાવી_hash>
        દ્વિઅંકી_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_add_id_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_add_id_pipeline_ચાવી_hash>
        add_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_concat_pipeline_ચાવી_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_repeat_pipeline_ચાવી_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_ચાવી,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_ચાવી_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_ચાવી_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_ચાવી,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_ચાવી_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_ચાવી,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_ચાવી_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_ચાવી_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_ચાવી_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_ચાવી_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_ચાવી,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_ચાવી_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_ચાવી_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_ચાવી_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_ચાવી_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_set_pipeline_ચાવી_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_cpy_pipeline_ચાવી_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_glu_pipeline_ચાવી_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_rope_pipeline_ચાવી_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_ચાવી_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_ચાવી_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_ચાવી_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_im2col_pipeline_ચાવી_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_ચાવી,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_ચાવી_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_ચાવી, webgpu_pipeline, ggml_webgpu_upscale_pipeline_ચાવી_hash>
        upscale_pipelines;

  જાહેર:
    ggml_webgpu_shader_lib(wgpu::Device device) { આ->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        auto it = sum_rows_pipelines.find(1);
        જો (it != sum_rows_pipelines.end()) {
            પાછા it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        પાછા sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_ચાવી ચાવી = {};
        ચાવી.op                                = context.dst->op;
        ચાવી.src_type                          = context.src0->type;
        ચાવી.dst_type                          = context.dst->type;
        ચાવી.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = row_norm_pipelines.find(ચાવી);
        જો (it != row_norm_pipelines.end()) {
            પાછા it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        પસંદગી (ચાવી.op) {
            સ્થિતિ GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                તોડો;
            સ્થિતિ GGML_OP_NORM:
                defines.push_back("NORM");
                variant = "norm";
                તોડો;
            સ્થિતિ GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        જો (ચાવી.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        જો (ચાવી.src_type == GGML_TYPE_F32) {
            defines.push_back("SRC_F32");
            variant += "_src_f32";
        } નહીંતર જો (ચાવી.src_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        }

        જો (ચાવી.dst_type == GGML_TYPE_F32) {
            defines.push_back("DST_F32");
            variant += "_dst_f32";
        } નહીંતર જો (ચાવી.dst_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        }

        અચલ uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = ચાવી.inplace;
        row_norm_pipelines[ચાવી]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[ચાવી].context = decisions;
        પાછા row_norm_pipelines[ચાવી];
    }

    webgpu_pipeline get_argmax_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        તાર્કિક vec4 = context.src0->ne[0] % 4 == 0;

        auto it = argmax_pipelines.find(vec4);
        જો (it != argmax_pipelines.end()) {
            પાછા it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        જો (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        auto processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        પાછા argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        અચલ તાર્કિક                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_set_rows_pipeline_ચાવી ચાવી       = {};
        ચાવી.src0_type                               = context.src0->type;
        ચાવી.dst_type                                = context.dst->type;
        ચાવી.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        ચાવી.i64_idx     = context.src1->type == GGML_TYPE_I64;
        ચાવી.pair_blocks = quantized && ((context.src0->ne[0] / ggml_blck_size(context.dst->type)) % 2 == 0);

        auto it = set_rows_pipelines.find(ચાવી);
        જો (it != set_rows_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        પસંદગી (context.dst->type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                તોડો;
            સ્થિતિ GGML_TYPE_Q8_0:
                defines.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                તોડો;
            સ્થિતિ GGML_TYPE_Q4_0:
                defines.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        જો (context.src0->type == GGML_TYPE_F16) {
            defines.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        જો (ચાવી.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        જો (ચાવી.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }
        જો (ચાવી.pair_blocks) {
            defines.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        અચલ auto & shader_source      = quantized ? wgsl_set_rows_quant : wgsl_set_rows;
        auto         processed          = preprocessor.preprocess(shader_source, defines);
        auto         decisions          = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = ચાવી.vec4;
        decisions->i64_idx              = ચાવી.i64_idx;
        decisions->pair_blocks          = ચાવી.pair_blocks;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[ચાવી]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[ચાવી].context = decisions;
        પાછા set_rows_pipelines[ચાવી];
    }

    webgpu_pipeline get_set_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_ચાવી ચાવી = {};
        ચાવી.type                         = context.dst->type;
        ચાવી.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = set_pipelines.find(ચાવી);
        જો (it != set_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for set shader");
        }

        જો (ચાવી.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_set, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ચાવી.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[ચાવી]       = pipeline;
        પાછા set_pipelines[ચાવી];
    }

    webgpu_pipeline get_cumsum_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        auto it = cumsum_pipelines.find(1);
        જો (it != cumsum_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        પાછા cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        તાર્કિક          is_ટોચ_k = context.dst->op == GGML_OP_ટોચ_K;
        // ascending order is 0, descending order is 1
        અચલ int32_t order =
            is_ટોચ_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_pipelines.find(order);
        જો (it != argsort_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        જ્યારે (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        auto decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        પાછા argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        તાર્કિક          is_ટોચ_k = context.dst->op == GGML_OP_ટોચ_K;
        // ascending order is 0, descending order is 1
        અચલ int32_t order =
            is_ટોચ_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_merge_pipelines.find(order);
        જો (it != argsort_merge_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        પાછા argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(અચલ ggml_webgpu_shader_lib_context & context, તાર્કિક vec4_aligned) {
        અચલ તાર્કિક vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_rows_pipeline_ચાવી ચાવી = {};
        ચાવી.src_type                          = context.src0->type;
        ચાવી.vectorized                        = (પૂર્ણાંક) vectorized;

        auto it = get_rows_pipelines.find(ચાવી);
        જો (it != get_rows_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        અચલ સંરચના ggml_type_traits * type_traits = ggml_get_type_traits(ચાવી.src_type);
        અચલ અક્ષર *                    type_str    = type_traits->type_name;

        પસંદગી (ચાવી.src_type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                જો (ચાવી.vectorized) {
                    defines.push_back("F32_VEC");
                    defines.push_back("SRC_TYPE=vec4<f32>");
                    defines.push_back("DST_TYPE=vec4<f32>");
                    defines.push_back("BLOCK_SIZE=4u");
                } નહીંતર {
                    defines.push_back("F32");
                    defines.push_back("SRC_TYPE=f32");
                    defines.push_back("DST_TYPE=f32");
                    defines.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                તોડો;
            સ્થિતિ GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                તોડો;
            સ્થિતિ GGML_TYPE_BF16:
                defines.push_back("BF16");
                defines.push_back("SRC_TYPE=u32");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                તોડો;
            મૂળભૂત:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    પસંદગી (ચાવી.src_type) {
                        સ્થિતિ GGML_TYPE_Q1_0:
                        સ્થિતિ GGML_TYPE_Q2_0:
                        સ્થિતિ GGML_TYPE_Q4_0:
                        સ્થિતિ GGML_TYPE_Q5_0:
                        સ્થિતિ GGML_TYPE_Q8_0:
                        સ્થિતિ GGML_TYPE_Q3_K:
                        સ્થિતિ GGML_TYPE_Q6_K:
                        સ્થિતિ GGML_TYPE_IQ2_XXS:
                        સ્થિતિ GGML_TYPE_IQ2_XS:
                        સ્થિતિ GGML_TYPE_IQ2_S:
                        સ્થિતિ GGML_TYPE_IQ3_XXS:
                        સ્થિતિ GGML_TYPE_IQ3_S:
                        સ્થિતિ GGML_TYPE_IQ1_S:
                        સ્થિતિ GGML_TYPE_IQ4_NL:
                        સ્થિતિ GGML_TYPE_MXFP4:
                        સ્થિતિ GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                તોડો;
                            }
                        મૂળભૂત:
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

                    જો (ચાવી.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } નહીંતર જો ((ચાવી.src_type >= GGML_TYPE_Q4_0 && ચાવી.src_type <= GGML_TYPE_Q8_1) ||
                               ચાવી.src_type == GGML_TYPE_IQ4_NL || ચાવી.src_type == GGML_TYPE_MXFP4) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } નહીંતર જો (ચાવી.src_type == GGML_TYPE_Q2_0 || ચાવી.src_type == GGML_TYPE_NVFP4) {
                        defines.push_back("BLOCK_SIZE=64u");
                    } નહીંતર જો (ચાવી.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } નહીંતર {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    તોડો;
                }
        }

        જો (ચાવી.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        auto decisions           = std::make_shared<ggml_webgpu_get_rows_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[ચાવી]  = pipeline;
        પાછા get_rows_pipelines[ચાવી];
    }

    webgpu_pipeline get_scale_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_ચાવી ચાવી = {};
        ચાવી.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = scale_pipelines.find(ચાવી);
        જો (it != scale_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        જો (ચાવી.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_scale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ચાવી.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[ચાવી]     = pipeline;
        પાછા scale_pipelines[ચાવી];
    }

    webgpu_pipeline get_solve_tri_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_ચાવી ચાવી = {};
        ચાવી.type                               = context.dst->type;
        ચાવી.n                                  = (પૂર્ણાંક) context.src0->ne[0];
        ચાવી.k                                  = (પૂર્ણાંક) context.src1->ne[0];

        auto it = solve_tri_pipelines.find(ચાવી);
        જો (it != solve_tri_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                variant += "_f32";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        અચલ uint32_t wg_size       = std::min((uint32_t) ચાવી.n, context.max_wg_size);
        અચલ uint32_t k_tile        = wg_size;
        અચલ uint32_t bytes_per_row = ((uint32_t) ચાવી.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        અચલ uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(ચાવી.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        auto processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[ચાવી] = pipeline;
        પાછા solve_tri_pipelines[ચાવી];
    }

    webgpu_pipeline get_ssm_conv_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_ચાવી ચાવી = {};
        ચાવી.type                              = context.dst->type;
        ચાવી.vectorized                        = context.src1->ne[0] == 4;

        auto it = ssm_conv_pipelines.find(ચાવી);
        જો (it != ssm_conv_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                variant += "_f32";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        જો (ચાવી.vectorized) {
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
        ssm_conv_pipelines[ચાવી]  = pipeline;
        પાછા ssm_conv_pipelines[ચાવી];
    }

    webgpu_pipeline get_ssm_scan_pipeline(અચલ ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_scan_pipeline_ચાવી ચાવી = {};
        ચાવી.type                              = context.dst->type;
        ચાવી.d_state                           = (પૂર્ણાંક) context.src0->ne[0];
        ચાવી.xbc_overlap                       = xbc_overlap;

        auto it = ssm_scan_pipelines.find(ચાવી);
        જો (it != ssm_scan_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                variant += "_f32";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        અચલ uint32_t wg_size = (uint32_t) ચાવી.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        જો (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } નહીંતર {
            variant += "_wg_reduce";
        }

        જો (ચાવી.xbc_overlap == 0b110) {  // x/B
            defines.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } નહીંતર જો (ચાવી.xbc_overlap == 0b011) {  // B/C
            defines.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } નહીંતર જો (ચાવી.xbc_overlap == 0b111) {  // x/B/C
            defines.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(ચાવી.d_state);

        auto processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        auto decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = ચાવી.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[ચાવી]    = pipeline;
        પાછા ssm_scan_pipelines[ચાવી];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_ચાવી ચાવી = {};
        ચાવી.type                                     = context.dst->type;
        ચાવી.s_v                                      = (પૂર્ણાંક) context.src2->ne[0];
        ચાવી.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        ચાવી.fused_cache                              = context.dst_fuse != nullptr;

        auto it = gated_delta_net_pipelines.find(ચાવી);
        જો (it != gated_delta_net_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                variant += "_f32";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        જો (ચાવી.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        જો (ચાવી.fused_cache) {
            defines.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        defines.push_back("S_V=" + std::to_string(ચાવી.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(ચાવી.s_v) + "u");

        auto            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[ચાવી] = pipeline;
        પાછા gated_delta_net_pipelines[ચાવી];
    }

    webgpu_pipeline get_pad_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_ચાવી ચાવી = {};
        ચાવી.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        auto it = pad_pipelines.find(ચાવી);
        જો (it != pad_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        જો (ચાવી.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_pad, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[ચાવી]       = pipeline;
        પાછા pad_pipelines[ચાવી];
    }

    webgpu_pipeline get_quantize_q8_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_ચાવી ચાવી = {};
        ચાવી.src0_type                            = context.src0->type;

        auto it = quantize_q8_pipelines.find(ચાવી);
        જો (it != quantize_q8_pipelines.end()) {
            પાછા it->second;
        }
        અચલ અક્ષર *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> defines;
        std::string              variant = "quantize_q8";

        uint32_t wg_size = WEBGPU_MUL_MAT_VEC_WG_SIZE;

        defines.push_back("SRC1_INNER_TYPE=f32");
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        અચલ સંરચના ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
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
        quantize_q8_pipelines[ચાવી] = pipeline;
        પાછા quantize_q8_pipelines[ચાવી];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(અચલ ggml_webgpu_shader_lib_context & context, તાર્કિક src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_ચાવી ચાવી = {};
        ચાવી.src0_type                            = context.src0->type;
        ચાવી.src1_type                            = context.src1->type;
        ચાવી.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        ચાવી.અંક_cols   = context.dst->ne[1];
        ચાવી.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
        ચાવી.src_overlap = src_overlap;

        auto it = mul_mat_vec_pipelines.find(ચાવી);
        જો (it != mul_mat_vec_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        અચલ અક્ષર *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        પસંદગી (context.src0->type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                તોડો;
            સ્થિતિ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                તોડો;
            મૂળભૂત:
                {
                    // Quantized types: use helpers but accumulate in f16
                    અચલ સંરચના ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    પસંદગી (context.src0->type) {
                        સ્થિતિ GGML_TYPE_Q4_0:
                        સ્થિતિ GGML_TYPE_Q4_1:
                        સ્થિતિ GGML_TYPE_Q5_0:
                        સ્થિતિ GGML_TYPE_Q5_1:
                        સ્થિતિ GGML_TYPE_Q8_0:
                            જો (ચાવી.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            તોડો;
                        સ્થિતિ GGML_TYPE_Q1_0:
                        સ્થિતિ GGML_TYPE_Q2_K:
                        સ્થિતિ GGML_TYPE_Q3_K:
                        સ્થિતિ GGML_TYPE_Q4_K:
                        સ્થિતિ GGML_TYPE_Q5_K:
                        સ્થિતિ GGML_TYPE_Q6_K:
                            જો (ચાવી.use_mmvq) {
                                defines.push_back("K_QUANTS_HANDLING");
                            }
                            તોડો;
                        સ્થિતિ GGML_TYPE_IQ1_S:
                        સ્થિતિ GGML_TYPE_IQ1_M:
                        સ્થિતિ GGML_TYPE_IQ2_S:
                        સ્થિતિ GGML_TYPE_IQ3_S:
                        સ્થિતિ GGML_TYPE_IQ4_NL:
                        સ્થિતિ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            તોડો;
                        સ્થિતિ GGML_TYPE_IQ2_XXS:
                        સ્થિતિ GGML_TYPE_IQ2_XS:
                        સ્થિતિ GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            તોડો;
                        સ્થિતિ GGML_TYPE_MXFP4:
                            defines.push_back(type_upper + "_LUT");
                            જો (ચાવી.use_mmvq) {
                                defines.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            તોડો;
                        સ્થિતિ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            તોડો;
                        મૂળભૂત:
                            તોડો;
                    }
                    તોડો;
                }
        }

        // src1 type (vector)
        પસંદગી (context.src1->type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(ચાવી.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        જો (ચાવી.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } નહીંતર જો (ચાવી.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } નહીંતર જો (ચાવી.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        જો (ચાવી.use_mmvq) {
            defines.push_back("MMVQ");
            defines.push_back("Q8_1_T");
        }

        જો (ચાવી.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        જો (ચાવી.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=") + std::to_string(ચાવી.અંક_cols));

        auto processed            = preprocessor.preprocess(shader_src, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = ચાવી.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[ચાવી] = pipeline;
        પાછા mul_mat_vec_pipelines[ચાવી];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(અચલ ggml_webgpu_shader_lib_context & context, તાર્કિક src_overlap) {
        ggml_webgpu_mul_mat_pipeline_ચાવી ચાવી = {};
        ચાવી.src0_type                        = context.src0->type;
        ચાવી.src1_type                        = context.src1->type;
        ચાવી.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        ચાવી.use_subgroup_matrix = context.supports_subgroup_matrix;
        ચાવી.src_overlap         = src_overlap;

        auto it = mul_mat_fast_pipelines.find(ચાવી);
        જો (it != mul_mat_fast_pipelines.end()) {
            પાછા it->second;
        }

        અચલ અક્ષર * shader_src = ચાવી.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = ચાવી.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        પસંદગી (context.src1->type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        અચલ સંરચના ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        અચલ અક્ષર *                    src0_name   = src0_traits->type_name;

        પસંદગી (context.src0->type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("TYPE_F32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("TYPE_F16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                તોડો;
            સ્થિતિ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("TYPE_BF16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                તોડો;
            મૂળભૂત:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    પસંદગી (context.src0->type) {
                        સ્થિતિ GGML_TYPE_IQ1_S:
                        સ્થિતિ GGML_TYPE_IQ1_M:
                        સ્થિતિ GGML_TYPE_IQ4_NL:
                        સ્થિતિ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            તોડો;
                        સ્થિતિ GGML_TYPE_IQ2_XXS:
                        સ્થિતિ GGML_TYPE_IQ2_XS:
                        સ્થિતિ GGML_TYPE_IQ2_S:
                        સ્થિતિ GGML_TYPE_IQ3_XXS:
                        સ્થિતિ GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            તોડો;
                        સ્થિતિ GGML_TYPE_MXFP4:
                        સ્થિતિ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            તોડો;
                        મૂળભૂત:
                            તોડો;
                    }

                    variant += std::string("_") + src0_name;
                    તોડો;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(ચાવી.vectorized ? "VEC" : "SCALAR");

        અચલ તાર્કિક is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        જો (ચાવી.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } નહીંતર {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        જો (ચાવી.use_subgroup_matrix) {
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
        જો (ચાવી.vectorized) {
            variant += "_vectorized";
        }

        જો (ચાવી.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        જો (!ચાવી.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = ચાવી.use_subgroup_matrix;
        જો (ચાવી.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_size           = context.max_subgroup_size;
        } નહીંતર {
            decisions->wg_size_m       = WEBGPU_MUL_MAT_WG_SIZE_M;
            decisions->wg_size_n       = WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->wg_size         = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->mul_mat_wg_size = WEBGPU_MUL_MAT_WG_SIZE;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[ચાવી] = pipeline;
        પાછા mul_mat_fast_pipelines[ચાવી];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_ચાવી ચાવી = {};
        ચાવી.n_experts                                  = (uint32_t) context.src0->ne[2];

        auto it = mul_mat_id_gather_pipelines.find(ચાવી);
        જો (it != mul_mat_id_gather_pipelines.end()) {
            પાછા it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size) + "u");
        defines.push_back(std::string("N_EXPERT=") + std::to_string(ચાવી.n_experts) + "u");
        defines.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        auto processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[ચાવી] = pipeline;
        પાછા pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ચાવી ચાવી = {};
        ચાવી.src0_type                           = context.src0->type;
        ચાવી.src1_type                           = context.src1->type;
        ચાવી.n_experts                           = context.src0->ne[2];
        ચાવી.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_pipelines.find(ચાવી);
        જો (it != mul_mat_id_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        પસંદગી (context.src1->type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        અચલ સંરચના ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        અચલ અક્ષર *                    src0_name   = src0_traits->type_name;

        પસંદગી (context.src0->type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                તોડો;
            સ્થિતિ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                તોડો;
            મૂળભૂત:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    પસંદગી (context.src0->type) {
                        સ્થિતિ GGML_TYPE_IQ1_S:
                        સ્થિતિ GGML_TYPE_IQ1_M:
                        સ્થિતિ GGML_TYPE_IQ4_NL:
                        સ્થિતિ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            તોડો;
                        સ્થિતિ GGML_TYPE_IQ2_XXS:
                        સ્થિતિ GGML_TYPE_IQ2_XS:
                        સ્થિતિ GGML_TYPE_IQ2_S:
                        સ્થિતિ GGML_TYPE_IQ3_XXS:
                        સ્થિતિ GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            તોડો;
                        સ્થિતિ GGML_TYPE_MXFP4:
                        સ્થિતિ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            તોડો;
                        મૂળભૂત:
                            તોડો;
                    }

                    variant += std::string("_") + src0_name;
                    તોડો;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(ચાવી.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        અચલ uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        જો (ચાવી.vectorized) {
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
        mul_mat_id_pipelines[ચાવી] = pipeline;
        પાછા mul_mat_id_pipelines[ચાવી];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ચાવી ચાવી = {};
        ચાવી.src0_type                           = context.src0->type;
        ચાવી.src1_type                           = context.src1->type;
        ચાવી.n_experts                           = context.src0->ne[2];
        ચાવી.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_vec_pipelines.find(ચાવી);
        જો (it != mul_mat_id_vec_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        અચલ અક્ષર *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        પસંદગી (context.src1->type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        પસંદગી (context.src0->type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_F16");
                variant += "_f16";
                તોડો;
            સ્થિતિ GGML_TYPE_BF16:
                defines.push_back("SRC0_INNER_TYPE=u32");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("TYPE_BF16");
                variant += "_bf16";
                તોડો;
            મૂળભૂત:
                {
                    // Quantized types: use helpers but accumulate in f16
                    અચલ સંરચના ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    પસંદગી (context.src0->type) {
                        સ્થિતિ GGML_TYPE_IQ1_S:
                        સ્થિતિ GGML_TYPE_IQ1_M:
                        સ્થિતિ GGML_TYPE_IQ2_S:
                        સ્થિતિ GGML_TYPE_IQ3_S:
                        સ્થિતિ GGML_TYPE_IQ4_NL:
                        સ્થિતિ GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            તોડો;
                        સ્થિતિ GGML_TYPE_IQ2_XXS:
                        સ્થિતિ GGML_TYPE_IQ2_XS:
                        સ્થિતિ GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            તોડો;
                        સ્થિતિ GGML_TYPE_MXFP4:
                        સ્થિતિ GGML_TYPE_NVFP4:
                            defines.push_back(type_upper + "_LUT");
                            તોડો;
                        મૂળભૂત:
                            તોડો;
                    }
                    તોડો;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(ચાવી.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        જો (ચાવી.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } નહીંતર જો (ચાવી.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } નહીંતર જો (ચાવી.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        જો (ચાવી.vectorized) {
            variant += "_vectorized";
        }
        defines.push_back(std::string("NUM_COLS=1"));

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(ચાવી.n_experts));

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[ચાવી] = pipeline;
        પાછા mul_mat_id_vec_pipelines[ચાવી];
    }

    webgpu_pipeline get_unary_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        અચલ તાર્કિક                     is_unary = context.dst->op == GGML_OP_UNARY;
        અચલ પૂર્ણાંક                      op       = is_unary ? (પૂર્ણાંક) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_ચાવી ચાવી      = {};
        ચાવી.type                                = context.dst->type;
        ચાવી.op                                  = op;
        ચાવી.is_unary                            = is_unary;
        ચાવી.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        ચાવી.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        auto it = unary_pipelines.find(ચાવી);
        જો (it != unary_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            ચાવી.is_unary ? ggml_unary_op_name((ggml_unary_op) ચાવી.op) : ggml_op_name((ggml_op) ચાવી.op);
        defines.push_back(variant);

        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for unary shader");
        }

        જો (ચાવી.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        જો (op == GGML_OP_TRI) {
            પસંદગી (ચાવી.ttype) {
                સ્થિતિ GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    તોડો;
                સ્થિતિ GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    તોડો;
                સ્થિતિ GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    તોડો;
                સ્થિતિ GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    તોડો;
                મૂળભૂત:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_unary, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ચાવી.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[ચાવી]     = pipeline;
        પાછા unary_pipelines[ચાવી];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(અચલ ggml_webgpu_shader_lib_context & context, તાર્કિક src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_ચાવી ચાવી = {};
        ચાવી.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ચાવી.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ચાવી.src_overlap                           = src_overlap;

        auto it = rms_norm_mul_pipelines.find(ચાવી);
        જો (it != rms_norm_mul_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        જો (ચાવી.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } નહીંતર જો (ચાવી.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } નહીંતર જો (ચાવી.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = ચાવી.inplace;
        pipeline_decisions->overlap     = ચાવી.overlap;
        pipeline_decisions->src_overlap = ચાવી.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[ચાવી]     = pipeline;
        પાછા rms_norm_mul_pipelines[ચાવી];
    }

    webgpu_pipeline get_દ્વિઅંકી_pipeline(અચલ ggml_webgpu_shader_lib_context & context, તાર્કિક src_overlap) {
        ggml_webgpu_દ્વિઅંકી_pipeline_ચાવી ચાવી = {};
        ચાવી.type                            = context.dst->type;
        ચાવી.op                              = context.dst->op;
        ચાવી.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ચાવી.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ચાવી.src_overlap                     = src_overlap;

        auto it = દ્વિઅંકી_pipelines.find(ચાવી);
        જો (it != દ્વિઅંકી_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) ચાવી.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for binary shader");
        }

        જો (ચાવી.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } નહીંતર જો (ચાવી.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } નહીંતર જો (ચાવી.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_દ્વિઅંકી, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_દ્વિઅંકી_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = ચાવી.inplace;
        pipeline_decisions->overlap     = ચાવી.overlap;
        pipeline_decisions->src_overlap = ચાવી.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        દ્વિઅંકી_pipelines[ચાવી]    = pipeline;
        પાછા દ્વિઅંકી_pipelines[ચાવી];
    }

    webgpu_pipeline get_add_id_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_add_id_pipeline_ચાવી ચાવી = {};
        ચાવી.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = add_id_pipelines.find(ચાવી);
        જો (it != add_id_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "add_id";
        અચલ અક્ષર *             shader_src = wgsl_add_id;

        જો (ચાવી.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed              = preprocessor.preprocess(shader_src, defines);
        auto pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_size = context.max_wg_size;
        pipeline_decisions->inplace = ચાવી.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        add_id_pipelines[ચાવી]    = pipeline;
        પાછા pipeline;
    }

    webgpu_pipeline get_concat_pipeline(અચલ ggml_webgpu_shader_lib_context & context, તાર્કિક src_overlap) {
        ggml_webgpu_concat_pipeline_ચાવી ચાવી = {};
        ચાવી.type                            = context.dst->type;
        ચાવી.src_overlap                     = src_overlap;

        auto it = concat_pipelines.find(ચાવી);
        જો (it != concat_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for concat shader");
        }

        જો (ચાવી.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_concat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_દ્વિઅંકી_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = ચાવી.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[ચાવી]    = pipeline;
        પાછા concat_pipelines[ચાવી];
    }

    webgpu_pipeline get_repeat_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_ચાવી ચાવી = {};
        ચાવી.type                            = context.dst->type;

        auto it = repeat_pipelines.find(ચાવી);
        જો (it != repeat_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                તોડો;
            સ્થિતિ GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                તોડો;
            સ્થિતિ GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_repeat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[ચાવી]    = pipeline;
        પાછા repeat_pipelines[ચાવી];
    }

    webgpu_pipeline get_flash_attn_pipeline(અચલ ggml_webgpu_shader_lib_context & context, તાર્કિક kv_overlap) {
        અચલ તાર્કિક can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_ચાવી ચાવી = {};
        ચાવી.common                              = ggml_webgpu_flash_attn_make_common_pipeline_ચાવી(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        ચાવી.common.k_direct &= decisions.use_sg_matrix && ચાવી.common.k_type == GGML_TYPE_F16;
        ચાવી.common.v_direct &= decisions.use_sg_matrix && ચાવી.common.v_type == GGML_TYPE_F16;
        ચાવી.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_limit_bytes, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            ચાવી.common.head_dim_qk, ચાવી.common.head_dim_v, ચાવી.common.has_mask,
            ચાવી.common.k_direct || ચાવી.common.v_direct);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        જો (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = ખોટું;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            ચાવી.common.k_direct     = ખોટું;
            ચાવી.common.v_direct     = ખોટું;
            ચાવી.use_sg_matrix       = ખોટું;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_limit_bytes, decisions.q_tile, 1u, ચાવી.common.head_dim_qk, ચાવી.common.head_dim_v,
                ચાવી.common.has_mask, ચાવી.common.k_direct || ચાવી.common.v_direct);
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

        જો (ચાવી.common.k_direct || ચાવી.common.v_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            જ્યારે (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_size;
            }
        }

        auto it = flash_attn_pipelines.find(ચાવી);
        જો (it != flash_attn_pipelines.end()) {
            પાછા it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> defines = ggml_webgpu_flash_attn_common_defines(ચાવી.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_size);
        અચલ અક્ષર *             shader_src = nullptr;
        જો (!ચાવી.use_sg_matrix) {
            shader_src = wgsl_flash_attn_tile;
            defines.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_size) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_size) + "_" +
                       std::to_string(context.max_subgroup_size);
        } નહીંતર {
            shader_src = wgsl_flash_attn;
            defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[ચાવી] = pipeline;
        પાછા flash_attn_pipelines[ચાવી];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(અચલ ggml_webgpu_shader_lib_context & context, તાર્કિક kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_ચાવી ચાવી = {};
        ચાવી.common = ggml_webgpu_flash_attn_make_common_pipeline_ચાવી(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH,
                                                                     kv_overlap);

        auto it = flash_attn_vec_pipelines.find(ચાવી);
        જો (it != flash_attn_vec_pipelines.end()) {
            પાછા it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_limit_bytes, ચાવી.common.head_dim_qk,
                                                                   ચાવી.common.head_dim_v, ચાવી.common.has_mask,
                                                                   ચાવી.common.k_direct || ચાવી.common.v_direct);
        decisions.wg_size = context.max_subgroup_size;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> defines =
            ggml_webgpu_flash_attn_common_defines(ચાવી.common, variant, 1u, decisions.kv_tile, decisions.wg_size);
        જો (ચાવી.common.has_mask) {
            defines.push_back("BLK");
            variant.resize(variant.size() - (કદ("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_split = context.min_subgroup_size;
        અચલ uint32_t D       = ચાવી.common.head_dim_qk | ચાવી.common.head_dim_v;
        અચલ uint32_t D_lsb   = D & (~(D - 1u));
        d_split                = std::min(std::min(context.min_subgroup_size, 4u), std::max(D_lsb / 4u, 1u));

        defines.push_back(std::string("D_SPLIT=") + std::to_string(d_split));
        variant += "_dsplit" + std::to_string(d_split);

        auto            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[ચાવી] = pipeline;
        પાછા flash_attn_vec_pipelines[ચાવી];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(અચલ ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_ચાવી ચાવી = {};
        ચાવી.kv_tile                                 = kv_tile;
        auto it                                     = flash_attn_blk_pipelines.find(ચાવી);
        જો (it != flash_attn_blk_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(ચાવી.kv_tile));
        variant += std::string("_kvt") + std::to_string(ચાવી.kv_tile);

        uint32_t wg_size = 1;
        જ્યારે ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[ચાવી] = pipeline;
        પાછા flash_attn_blk_pipelines[ચાવી];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_ચાવી ચાવી = {};
        ચાવી.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        ચાવી.dst_type                                       = context.dst->type;
        ચાવી.wg_size                                        = context.max_wg_size;
        auto it                                            = flash_attn_vec_reduce_pipelines.find(ચાવી);
        જો (it != flash_attn_vec_reduce_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        પસંદગી (ચાવી.dst_type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(ચાવી.dst_type);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(ચાવી.head_dim_v));
        variant += std::string("_hsv") + std::to_string(ચાવી.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[ચાવી] = pipeline;
        પાછા flash_attn_vec_reduce_pipelines[ચાવી];
    }

    webgpu_pipeline get_cpy_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_ચાવી ચાવી = {};
        ચાવી.src_type                     = context.src0->type;
        ચાવી.dst_type                     = context.dst->type;

        auto it = cpy_pipelines.find(ચાવી);
        જો (it != cpy_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        પસંદગી (ચાવી.src_type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                તોડો;
            સ્થિતિ GGML_TYPE_I32:
                defines.push_back("SRC_I32");
                variant += "_i32";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        પસંદગી (ચાવી.dst_type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                તોડો;
            સ્થિતિ GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_cpy, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[ચાવી]       = pipeline;
        પાછા cpy_pipelines[ચાવી];
    }

    webgpu_pipeline get_glu_pipeline(અચલ ggml_webgpu_shader_lib_context & context, તાર્કિક src_overlap) {
        ggml_webgpu_glu_pipeline_ચાવી ચાવી = {};
        ચાવી.glu_op                       = ggml_get_glu_op(context.dst);
        ચાવી.type                         = context.dst->type;
        ચાવી.split                        = (context.src1 != nullptr);
        ચાવી.src_overlap                  = src_overlap;

        auto it = glu_pipelines.find(ચાવી);
        જો (it != glu_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        પસંદગી (ચાવી.glu_op) {
            સ્થિતિ GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                તોડો;
            સ્થિતિ GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                તોડો;
            સ્થિતિ GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                તોડો;
            સ્થિતિ GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                તોડો;
            સ્થિતિ GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                તોડો;
            સ્થિતિ GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                તોડો;
            સ્થિતિ GGML_GLU_OP_SWIGLU_CLAMP:
                defines.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported GLU op");
        }
        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        જો (ચાવી.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } નહીંતર જો (ચાવી.split) {
            variant += "_split";
        } નહીંતર {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_glu, defines);
        auto decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->src_overlap   = ચાવી.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[ચાવી]       = pipeline;
        પાછા glu_pipelines[ચાવી];
    }

    webgpu_pipeline get_rope_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_ચાવી ચાવી = {};
        ચાવી.type                          = context.dst->type;
        ચાવી.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ચાવી.has_ff                        = (context.src2 != nullptr);

        auto it = rope_pipelines.find(ચાવી);
        જો (it != rope_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        પસંદગી (ચાવી.type) {
            સ્થિતિ GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        જો (ચાવી.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        જો (ચાવી.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_rope, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ચાવી.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[ચાવી]      = pipeline;
        પાછા rope_pipelines[ચાવી];
    }

    webgpu_pipeline get_soft_max_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_ચાવી ચાવી = {};
        ચાવી.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        ચાવી.has_mask                          = (context.src1 != nullptr);
        ચાવી.has_sink                          = (context.src2 != nullptr);
        ચાવી.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = soft_max_pipelines.find(ચાવી);
        જો (it != soft_max_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        જો (ચાવી.has_mask) {
            defines.push_back("HAS_MASK");
            પસંદગી (ચાવી.mask_type) {
                સ્થિતિ GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    તોડો;
                સ્થિતિ GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    તોડો;
                મૂળભૂત:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        જો (ચાવી.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        જો (ચાવી.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = ચાવી.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[ચાવી]  = pipeline;
        પાછા soft_max_pipelines[ચાવી];
    }

    webgpu_pipeline get_conv2d_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_ચાવી ચાવી = {};
        ચાવી.weight_type                     = context.src0->type;
        ચાવી.આગત_type                      = context.src1->type;
        ચાવી.output_type                     = context.dst->type;

        auto it = conv2d_pipelines.find(ચાવી);
        જો (it != conv2d_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        auto ધકેલો_type_defines = [&](અચલ અક્ષર * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            જો (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } નહીંતર જો (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } નહીંતર {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ધકેલો_type_defines("WEIGHT_TYPE", ચાવી.weight_type);
        ધકેલો_type_defines("INPUT_TYPE", ચાવી.આગત_type);
        ધકેલો_type_defines("OUTPUT_TYPE", ચાવી.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[ચાવી]    = pipeline;
        પાછા conv2d_pipelines[ચાવી];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(અચલ ggml_webgpu_shader_lib_context & context, તાર્કિક whcn) {
        ggml_webgpu_conv2d_dw_pipeline_ચાવી ચાવી = {};
        ચાવી.weight_type                        = context.src0->type;
        ચાવી.આગત_type                         = context.src1->type;
        ચાવી.output_type                        = context.dst->type;
        ચાવી.whcn                               = whcn;

        auto it = conv2d_dw_pipelines.find(ચાવી);
        જો (it != conv2d_dw_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        auto ધકેલો_type_defines = [&](અચલ અક્ષર * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            જો (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } નહીંતર જો (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } નહીંતર {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ધકેલો_type_defines("WEIGHT_TYPE", ચાવી.weight_type);
        ધકેલો_type_defines("INPUT_TYPE", ચાવી.આગત_type);
        ધકેલો_type_defines("OUTPUT_TYPE", ચાવી.output_type);

        જો (whcn) {
            defines.push_back("WHCN");
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d_dw, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[ચાવી] = pipeline;
        પાછા conv2d_dw_pipelines[ચાવી];
    }

    webgpu_pipeline get_im2col_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_ચાવી ચાવી = {};
        ચાવી.આગત_type                      = context.src1->type;
        ચાવી.output_type                     = context.dst->type;

        auto it = im2col_pipelines.find(ચાવી);
        જો (it != im2col_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        auto ધકેલો_type_defines = [&](અચલ અક્ષર * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            જો (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "=f32");
            } નહીંતર જો (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "=f16");
            } નહીંતર {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ધકેલો_type_defines("INPUT_TYPE", ચાવી.આગત_type);
        ધકેલો_type_defines("OUTPUT_TYPE", ચાવી.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[ચાવી]    = pipeline;
        પાછા im2col_pipelines[ચાવી];
    }

    webgpu_pipeline get_upscale_pipeline(અચલ ggml_webgpu_shader_lib_context & context) {
        અચલ uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        અચલ uint32_t base_mode  = mode_flags & 0xFFu;
        અચલ તાર્કિક     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_ચાવી ચાવી = {};
        ચાવી.આગત_type                       = context.src0->type;
        ચાવી.output_type                      = context.dst->type;
        ચાવી.base_mode                        = base_mode;
        ચાવી.antialias                        = antialias;

        auto it = upscale_pipelines.find(ચાવી);
        જો (it != upscale_pipelines.end()) {
            પાછા it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        જો (ચાવી.આગત_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } નહીંતર {
            variant += "_src_f32";
        }

        જો (ચાવી.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } નહીંતર {
            variant += "_dst_f32";
        }

        પસંદગી (base_mode) {
            સ્થિતિ GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                તોડો;
            સ્થિતિ GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                તોડો;
            સ્થિતિ GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                તોડો;
            મૂળભૂત:
                GGML_ABORT("Unsupported upscale mode");
        }

        જો (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_upscale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[ચાવી]   = pipeline;
        પાછા upscale_pipelines[ચાવી];
    }

  ખાનગી:
    સ્થિર webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
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
        પાછા { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
