#यदि_अपरिभाषितम् GGML_WEBGPU_SHADER_LIB_HPP
#परिभाषय GGML_WEBGPU_SHADER_LIB_HPP

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

#परिभाषय GGML_WEBGPU_F16_आकार_अष्टकानि                   2
#परिभाषय GGML_WEBGPU_F32_आकार_अष्टकानि                   4
#परिभाषय GGML_WEBGPU_I32_आकार_अष्टकानि                   4
#परिभाषय GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#परिभाषय GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#परिभाषय GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#परिभाषय GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#परिभाषय GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_आकार     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#परिभाषय GGML_WEBGPU_KV_SEQ_PAD                       256u

#परिभाषय GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_आकार 512u

// Matrix multiplication parameters

// Register tiling parameters
#परिभाषय WEBGPU_MUL_MAT_TILE_M           4
#परिभाषय WEBGPU_MUL_MAT_TILE_N           4
#परिभाषय WEBGPU_MUL_MAT_WG_आकार_M        8
#परिभाषय WEBGPU_MUL_MAT_WG_आकार_N        8
#परिभाषय WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#परिभाषय WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#परिभाषय WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#परिभाषय WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#परिभाषय WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M     4
#परिभाषय WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N     2
#परिभाषय WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#परिभाषय WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#परिभाषय WEBGPU_MUL_MAT_VEC_WG_आकार 256

#परिभाषय WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_प्रत्येकम्_WG    4
#परिभाषय WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_प्रत्येकम्_WG 4
#परिभाषय WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_प्रत्येकम्_WG      4

// default size for reg-tile matrix multiplication
#परिभाषय WEBGPU_MUL_MAT_WG_आकार 256

// Same hash combine function as in boost
प्रतिरूपम् <प्रकारनाम T> अन्तःस्थ शून्य ggml_webgpu_hash_combine(size_t & बीजम्, ध्रुव T & value) {
    बीजम् ^= std::hash<T>{}(value) + 0x9e3779b9 + (बीजम् << 6) + (बीजम् >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
अन्तःस्थ uintptr_t ggml_webgpu_tensor_addr(ध्रुव ggml_tensor * tensor) {
    ध्रुव ggml_tensor * आधारः_tensor = tensor->view_src ? tensor->view_src : tensor;
    प्रत्यागम (uintptr_t) आधारः_tensor->दत्त + tensor->view_offs;
}

अन्तःस्थ तार्किक ggml_webgpu_tensor_equal(ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
    प्रत्यागम a->बफर == b->बफर && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
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

    uint32_t    max_wg_आकार;
    size_t      wg_mem_सीमा_अष्टकानि       = 0;
    तार्किक        supports_subgroups       = असत्य;
    तार्किक        supports_subgroup_matrix = असत्य;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_आकार        = 0;
    uint32_t    max_subgroup_आकार        = 0;
    तार्किक        supports_dot_गुणनफलम्     = असत्य;
    std::string vendor;
};

संरचना webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<शून्य> context = शून्यसूचकः;
};

संरचना ggml_webgpu_generic_shader_decisions {
    uint32_t wg_आकार = 0;
    तार्किक     inplace = असत्य;
};

संरचना ggml_webgpu_get_पङ्क्तिकाः_shader_decisions {
    uint32_t wg_आकार    = 0;
    तार्किक     vectorized = असत्य;
};

संरचना ggml_webgpu_द्विमान_shader_decisions {
    uint32_t wg_आकार     = 0;
    तार्किक     inplace     = असत्य;
    तार्किक     overlap     = असत्य;
    तार्किक     src_overlap = असत्य;
};

संरचना ggml_webgpu_glu_shader_decisions {
    uint32_t wg_आकार     = 0;
    तार्किक     src_overlap = असत्य;
};

संरचना ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<शून्य> decisions;
};

संरचना ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t खण्डम्_आकार_3;
    uint32_t tokens_प्रत्येकम्_wg;
};

संरचना ggml_webgpu_ssm_अवलोकनम्_pipeline_कुञ्चिका {
    पूर्णाङ्क     type;
    पूर्णाङ्क     d_state;
    uint8_t xbc_overlap;
    तार्किक    a_overlap;
    तार्किक    ids_overlap;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_ssm_अवलोकनम्_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

संरचना ggml_webgpu_ssm_अवलोकनम्_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_ssm_अवलोकनम्_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.d_state);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.xbc_overlap);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.a_overlap);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.ids_overlap);
        प्रत्यागम बीजम्;
    }
};

संरचना ggml_webgpu_ssm_अवलोकनम्_shader_decisions {
    uint32_t wg_आकार;
    uint32_t tokens_प्रत्येकम्_tile;
    uint8_t  xbc_overlap = 0;
    तार्किक     a_overlap   = असत्य;
    तार्किक     ids_overlap = असत्य;
};

/** Argsort **/

संरचना ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_आकार;
    size_t   wg_mem_सीमा_अष्टकानि;
    int32_t  क्रमः;
};

/** Set Rows **/

संरचना ggml_webgpu_समुच्चयः_पङ्क्तिकाः_pipeline_कुञ्चिका {
    पूर्णाङ्क src0_type;
    पूर्णाङ्क dst_type;
    पूर्णाङ्क vec4;
    पूर्णाङ्क i64_idx;
    पूर्णाङ्क pair_खण्डानि;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_समुच्चयः_पङ्क्तिकाः_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_खण्डानि == other.pair_खण्डानि;
    }
};

संरचना ggml_webgpu_समुच्चयः_पङ्क्तिकाः_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_समुच्चयः_पङ्क्तिकाः_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src0_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.dst_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.vec4);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.i64_idx);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.pair_खण्डानि);
        प्रत्यागम बीजम्;
    }
};

संरचना ggml_webgpu_समुच्चयः_पङ्क्तिकाः_shader_decisions {
    तार्किक     vec4;
    तार्किक     i64_idx;
    तार्किक     pair_खण्डानि;
    uint32_t wg_आकार;
};

/** Set **/

संरचना ggml_webgpu_समुच्चयः_pipeline_कुञ्चिका {
    ggml_type type;
    तार्किक      inplace;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_समुच्चयः_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम type == other.type && inplace == other.inplace;
    }
};

संरचना ggml_webgpu_समुच्चयः_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_समुच्चयः_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.inplace);
        प्रत्यागम बीजम्;
    }
};

/** Get Rows **/

संरचना ggml_webgpu_get_पङ्क्तिकाः_pipeline_कुञ्चिका {
    ggml_type src_type;
    पूर्णाङ्क       vectorized;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_get_पङ्क्तिकाः_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम src_type == other.src_type && vectorized == other.vectorized;
    }
};

संरचना ggml_webgpu_get_पङ्क्तिकाः_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_get_पङ्क्तिकाः_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.vectorized);
        प्रत्यागम बीजम्;
    }
};

/** Row Norm **/

संरचना ggml_webgpu_row_norm_pipeline_कुञ्चिका {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    तार्किक      inplace;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_row_norm_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

संरचना ggml_webgpu_row_norm_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_row_norm_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.op);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.dst_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.inplace);
        प्रत्यागम बीजम्;
    }
};

/** RMS_NORM + MUL **/

संरचना ggml_webgpu_rms_norm_mul_pipeline_कुञ्चिका {
    तार्किक inplace;      // rn_src == dst
    तार्किक overlap;      // mul_src == dst
    तार्किक src_overlap;  // rn_src binding overlaps mul_src binding

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_rms_norm_mul_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_rms_norm_mul_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_rms_norm_mul_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.inplace);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.overlap);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src_overlap);
        प्रत्यागम बीजम्;
    }
};

संरचना ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_आकार     = 0;
    तार्किक     inplace     = असत्य;
    तार्किक     overlap     = असत्य;
    तार्किक     src_overlap = असत्य;
};

/** Pad **/
संरचना ggml_webgpu_pad_pipeline_कुञ्चिका {
    तार्किक circular;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_pad_pipeline_कुञ्चिका & other) ध्रुव { प्रत्यागम circular == other.circular; }
};

संरचना ggml_webgpu_pad_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_pad_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.circular);
        प्रत्यागम बीजम्;
    }
};

/** Solve Tri **/
संरचना ggml_webgpu_solve_tri_pipeline_कुञ्चिका {
    पूर्णाङ्क type;
    पूर्णाङ्क n;
    पूर्णाङ्क k;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_solve_tri_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम type == other.type && n == other.n && k == other.k;
    }
};

संरचना ggml_webgpu_solve_tri_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_solve_tri_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.n);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.k);
        प्रत्यागम बीजम्;
    }
};

/** SSM Conv **/
संरचना ggml_webgpu_ssm_conv_pipeline_कुञ्चिका {
    पूर्णाङ्क type;
    पूर्णाङ्क vectorized;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_ssm_conv_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
संरचना ggml_webgpu_conv2d_pipeline_कुञ्चिका {
    ggml_type weight_type;
    ggml_type निवेशः_type;
    ggml_type output_type;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_conv2d_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम weight_type == other.weight_type && निवेशः_type == other.निवेशः_type && output_type == other.output_type;
    }
};

संरचना ggml_webgpu_conv2d_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_conv2d_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.weight_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.निवेशः_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.output_type);
        प्रत्यागम बीजम्;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
संरचना ggml_webgpu_conv2d_dw_pipeline_कुञ्चिका {
    ggml_type weight_type;
    ggml_type निवेशः_type;
    ggml_type output_type;
    तार्किक      whcn;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_conv2d_dw_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम weight_type == other.weight_type && निवेशः_type == other.निवेशः_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

संरचना ggml_webgpu_conv2d_dw_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_conv2d_dw_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.weight_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.निवेशः_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.output_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.whcn);
        प्रत्यागम बीजम्;
    }
};

/** Im2Col **/
संरचना ggml_webgpu_im2col_pipeline_कुञ्चिका {
    ggml_type निवेशः_type;
    ggml_type output_type;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_im2col_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम निवेशः_type == other.निवेशः_type && output_type == other.output_type;
    }
};

संरचना ggml_webgpu_im2col_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_im2col_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.निवेशः_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.output_type);
        प्रत्यागम बीजम्;
    }
};

/** Gated Delta Net **/
संरचना ggml_webgpu_gated_delta_net_pipeline_कुञ्चिका {
    पूर्णाङ्क  type;
    पूर्णाङ्क  s_v;
    पूर्णाङ्क  kda;
    तार्किक fused_cache;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_gated_delta_net_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

संरचना ggml_webgpu_gated_delta_net_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_gated_delta_net_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.s_v);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.kda);
        प्रत्यागम बीजम्;
    }
};

संरचना ggml_webgpu_ssm_conv_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_ssm_conv_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.vectorized);
        प्रत्यागम बीजम्;
    }
};

/** Scale **/

संरचना ggml_webgpu_scale_pipeline_कुञ्चिका {
    पूर्णाङ्क inplace;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_scale_pipeline_कुञ्चिका & other) ध्रुव { प्रत्यागम inplace == other.inplace; }
};

संरचना ggml_webgpu_scale_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_scale_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.inplace);
        प्रत्यागम बीजम्;
    }
};

/** Upscale **/

संरचना ggml_webgpu_upscale_pipeline_कुञ्चिका {
    ggml_type निवेशः_type;
    ggml_type output_type;
    uint32_t  आधारः_बहुलकम्;
    तार्किक      antialias;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_upscale_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम निवेशः_type == other.निवेशः_type && output_type == other.output_type && आधारः_बहुलकम् == other.आधारः_बहुलकम् &&
               antialias == other.antialias;
    }
};

संरचना ggml_webgpu_upscale_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_upscale_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.निवेशः_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.output_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.आधारः_बहुलकम्);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.antialias);
        प्रत्यागम बीजम्;
    }
};

/** Concat **/

संरचना ggml_webgpu_concat_pipeline_कुञ्चिका {
    पूर्णाङ्क  type;
    तार्किक src_overlap;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_concat_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम type == other.type && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_concat_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_concat_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src_overlap);
        प्रत्यागम बीजम्;
    }
};

/** Repeat **/

संरचना ggml_webgpu_repeat_pipeline_कुञ्चिका {
    पूर्णाङ्क type;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_repeat_pipeline_कुञ्चिका & other) ध्रुव { प्रत्यागम type == other.type; }
};

संरचना ggml_webgpu_repeat_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_repeat_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        प्रत्यागम बीजम्;
    }
};

/** Binary **/

संरचना ggml_webgpu_द्विमान_pipeline_कुञ्चिका {
    पूर्णाङ्क  type;
    पूर्णाङ्क  op;
    तार्किक inplace;
    तार्किक overlap;
    तार्किक src_overlap;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_द्विमान_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_द्विमान_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_द्विमान_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.op);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.inplace);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.overlap);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src_overlap);
        प्रत्यागम बीजम्;
    }
};

/* Add_Id */

संरचना ggml_webgpu_सङ्कलय_id_pipeline_कुञ्चिका {
    तार्किक inplace;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_सङ्कलय_id_pipeline_कुञ्चिका & other) ध्रुव { प्रत्यागम inplace == other.inplace; }
};

संरचना ggml_webgpu_सङ्कलय_id_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_सङ्कलय_id_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.inplace);
        प्रत्यागम बीजम्;
    }
};

/** Unary **/

संरचना ggml_webgpu_unary_pipeline_कुञ्चिका {
    पूर्णाङ्क           type;
    पूर्णाङ्क           op;
    तार्किक          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    तार्किक          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_unary_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

संरचना ggml_webgpu_unary_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_unary_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.op);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.is_unary);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.inplace);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.ttype);
        प्रत्यागम बीजम्;
    }
};

/** FlashAttention */

संरचना ggml_webgpu_flash_attn_common_pipeline_कुञ्चिका {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  शीर्ष_dim_qk;
    uint32_t  शीर्ष_dim_v;
    तार्किक      k_प्रत्यक्षम्;
    तार्किक      v_प्रत्यक्षम्;
    तार्किक      kv_overlap;
    तार्किक      has_आवरणम्;
    तार्किक      has_sinks;
    तार्किक      उपयुङ्क्ते_logit_softcap;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_flash_attn_common_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && शीर्ष_dim_qk == other.शीर्ष_dim_qk && शीर्ष_dim_v == other.शीर्ष_dim_v &&
               k_प्रत्यक्षम् == other.k_प्रत्यक्षम् && v_प्रत्यक्षम् == other.v_प्रत्यक्षम् && kv_overlap == other.kv_overlap &&
               has_आवरणम् == other.has_आवरणम् && has_sinks == other.has_sinks &&
               उपयुङ्क्ते_logit_softcap == other.उपयुङ्क्ते_logit_softcap;
    }
};

अन्तःस्थ शून्य ggml_webgpu_flash_attn_hash_common_pipeline_कुञ्चिका(size_t &                                           बीजम्,
                                                            ध्रुव ggml_webgpu_flash_attn_common_pipeline_कुञ्चिका & कुञ्चिका) {
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.q_type);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.k_type);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.v_type);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.dst_type);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.शीर्ष_dim_qk);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.शीर्ष_dim_v);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.k_प्रत्यक्षम्);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.v_प्रत्यक्षम्);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.kv_overlap);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.has_आवरणम्);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.has_sinks);
    ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.उपयुङ्क्ते_logit_softcap);
}

संरचना ggml_webgpu_flash_attn_vec_pipeline_कुञ्चिका {
    ggml_webgpu_flash_attn_common_pipeline_कुञ्चिका common;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_flash_attn_vec_pipeline_कुञ्चिका & other) ध्रुव { प्रत्यागम common == other.common; }
};

संरचना ggml_webgpu_flash_attn_vec_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_flash_attn_vec_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_कुञ्चिका(बीजम्, कुञ्चिका.common);
        प्रत्यागम बीजम्;
    }
};

संरचना ggml_webgpu_flash_attn_pipeline_कुञ्चिका {
    ggml_webgpu_flash_attn_common_pipeline_कुञ्चिका common;
    तार्किक                                       use_sg_matrix;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_flash_attn_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम common == other.common && use_sg_matrix == other.use_sg_matrix;
    }
};

संरचना ggml_webgpu_flash_attn_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_flash_attn_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_कुञ्चिका(बीजम्, कुञ्चिका.common);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.use_sg_matrix);
        प्रत्यागम बीजम्;
    }
};

संरचना ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_आकार = 0;
};

संरचना ggml_webgpu_flash_attn_decisions {
    तार्किक     use_sg_matrix = असत्य;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_आकार       = 0;
};

अन्तःस्थ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_विस्तारः = 4u;
अन्तःस्थ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

अन्तःस्थ size_t ggml_webgpu_flash_attn_tensor_विस्थापनम्(ध्रुव ggml_tensor * tensor) {
    constexpr uintptr_t ptr_आधारः_addr = 0x1000u;
    ध्रुव ggml_tensor * आधारः          = tensor->view_src != शून्यसूचकः ? tensor->view_src : tensor;
    प्रत्यागम reinterpret_cast<uintptr_t>(आधारः->दत्त) - ptr_आधारः_addr + tensor->view_offs;
}

अन्तःस्थ तार्किक ggml_webgpu_flash_attn_float_vec4_aligned(ध्रुव ggml_tensor * K, size_t storage_विस्थापनम्_alignment) {
    ध्रुव uint32_t विस्थापनम्_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_विस्थापनम्(K) & (storage_विस्थापनम्_alignment - 1)) /
                    ggml_type_आकार(K->type));
    प्रत्यागम विस्थापनम्_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_विस्तारः == 0u;
}

अन्तःस्थ तार्किक ggml_webgpu_flash_attn_float_vec4_aligned(ध्रुव ggml_tensor * K,
                                                      ध्रुव ggml_tensor * V,
                                                      size_t              storage_विस्थापनम्_alignment) {
    प्रत्यागम ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_विस्थापनम्_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_विस्थापनम्_alignment);
}

अन्तःस्थ तार्किक ggml_webgpu_flash_attn_k_प्रत्यक्षम्(ध्रुव ggml_tensor * Q, ध्रुव ggml_tensor * K, uint32_t kv_प्रत्यक्षम्_align) {
    प्रत्यागम (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_प्रत्यक्षम्_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

अन्तःस्थ तार्किक ggml_webgpu_flash_attn_v_प्रत्यक्षम्(ध्रुव ggml_tensor * Q, ध्रुव ggml_tensor * V, uint32_t kv_प्रत्यक्षम्_align) {
    प्रत्यागम ggml_webgpu_flash_attn_k_प्रत्यक्षम्(Q, V, kv_प्रत्यक्षम्_align);
}

अन्तःस्थ ggml_webgpu_flash_attn_common_pipeline_कुञ्चिका ggml_webgpu_flash_attn_make_common_pipeline_कुञ्चिका(
    ध्रुव ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_प्रत्यक्षम्_align,
    तार्किक                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_कुञ्चिका कुञ्चिका = {};
    कुञ्चिका.q_type                                     = context.src0->type;
    कुञ्चिका.k_type                                     = context.src1->type;
    कुञ्चिका.v_type                                     = context.src2->type;
    कुञ्चिका.dst_type                                   = context.dst->type;
    कुञ्चिका.शीर्ष_dim_qk                                = (uint32_t) context.src0->ne[0];
    कुञ्चिका.शीर्ष_dim_v                                 = (uint32_t) context.src2->ne[0];
    कुञ्चिका.k_प्रत्यक्षम्           = ggml_webgpu_flash_attn_k_प्रत्यक्षम्(context.src0, context.src1, kv_प्रत्यक्षम्_align);
    कुञ्चिका.v_प्रत्यक्षम्           = ggml_webgpu_flash_attn_v_प्रत्यक्षम्(context.src0, context.src2, kv_प्रत्यक्षम्_align);
    कुञ्चिका.kv_overlap         = kv_overlap;
    कुञ्चिका.has_आवरणम्           = context.src3 != शून्यसूचकः;
    कुञ्चिका.has_sinks          = context.src4 != शून्यसूचकः;
    कुञ्चिका.उपयुङ्क्ते_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    प्रत्यागम कुञ्चिका;
}

अन्तःस्थ std::vector<std::string> ggml_webgpu_flash_attn_common_परिभाषते(
    ध्रुव ggml_webgpu_flash_attn_common_pipeline_कुञ्चिका & कुञ्चिका,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_आकार) {
    std::vector<std::string> परिभाषते;

    विकल्प (कुञ्चिका.k_type) {
        पक्ष GGML_TYPE_F32:
            परिभाषते.push_back("K_F32");
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            परिभाषते.push_back("K_F16");
            भङ्ग;
        पक्ष GGML_TYPE_Q4_0:
            परिभाषते.push_back("K_Q4_0");
            भङ्ग;
        पक्ष GGML_TYPE_Q8_0:
            परिभाषते.push_back("K_Q8_0");
            भङ्ग;
        स्वभाव:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(कुञ्चिका.k_type);

    विकल्प (कुञ्चिका.v_type) {
        पक्ष GGML_TYPE_F32:
            परिभाषते.push_back("V_F32");
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            परिभाषते.push_back("V_F16");
            भङ्ग;
        पक्ष GGML_TYPE_Q4_0:
            परिभाषते.push_back("V_Q4_0");
            भङ्ग;
        पक्ष GGML_TYPE_Q8_0:
            परिभाषते.push_back("V_Q8_0");
            भङ्ग;
        स्वभाव:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(कुञ्चिका.v_type);

    विकल्प (कुञ्चिका.q_type) {
        पक्ष GGML_TYPE_F32:
            परिभाषते.push_back("Q_F32");
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            परिभाषते.push_back("Q_F16");
            भङ्ग;
        स्वभाव:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(कुञ्चिका.q_type);

    विकल्प (कुञ्चिका.dst_type) {
        पक्ष GGML_TYPE_F32:
            परिभाषते.push_back("DST_F32");
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            परिभाषते.push_back("DST_F16");
            भङ्ग;
        स्वभाव:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(कुञ्चिका.dst_type);

    यदि (कुञ्चिका.has_आवरणम्) {
        परिभाषते.push_back("MASK");
        variant += "_mask";
    }
    यदि (कुञ्चिका.has_sinks) {
        परिभाषते.push_back("SINKS");
        variant += "_sinks";
    }
    यदि (कुञ्चिका.उपयुङ्क्ते_logit_softcap) {
        परिभाषते.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    यदि (कुञ्चिका.k_प्रत्यक्षम्) {
        परिभाषते.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    यदि (कुञ्चिका.v_प्रत्यक्षम्) {
        परिभाषते.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    यदि (कुञ्चिका.kv_overlap) {
        परिभाषते.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    परिभाषते.push_back(std::string("HEAD_DIM_QK=") + std::to_string(कुञ्चिका.शीर्ष_dim_qk));
    variant += std::string("_hsqk") + std::to_string(कुञ्चिका.शीर्ष_dim_qk);

    परिभाषते.push_back(std::string("HEAD_DIM_V=") + std::to_string(कुञ्चिका.शीर्ष_dim_v));
    variant += std::string("_hsv") + std::to_string(कुञ्चिका.शीर्ष_dim_v);

    परिभाषते.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    परिभाषते.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));

    यदि (ggml_is_quantized(कुञ्चिका.k_type) || ggml_is_quantized(कुञ्चिका.v_type)) {
        परिभाषते.push_back("U32_DEQUANT_HELPERS");
        यदि (ggml_is_quantized(कुञ्चिका.k_type)) {
            परिभाषते.push_back("LOADERS_QUANTIZED_K");
        }
        यदि (ggml_is_quantized(कुञ्चिका.v_type)) {
            परिभाषते.push_back("LOADERS_QUANTIZED_V");
        }
    }

    प्रत्यागम परिभाषते;
}

संरचना ggml_webgpu_flash_attn_vec_reduce_pipeline_कुञ्चिका {
    uint32_t  शीर्ष_dim_v;
    uint32_t  wg_आकार;
    ggml_type dst_type;
};

संरचना ggml_webgpu_flash_attn_vec_reduce_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_flash_attn_vec_reduce_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.शीर्ष_dim_v);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.wg_आकार);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.dst_type);
        प्रत्यागम बीजम्;
    }
};

अन्तःस्थ तार्किक संक्रियकः==(ध्रुव ggml_webgpu_flash_attn_vec_reduce_pipeline_कुञ्चिका & lhs,
                       ध्रुव ggml_webgpu_flash_attn_vec_reduce_pipeline_कुञ्चिका & rhs) {
    प्रत्यागम lhs.शीर्ष_dim_v == rhs.शीर्ष_dim_v && lhs.wg_आकार == rhs.wg_आकार && lhs.dst_type == rhs.dst_type;
}

संरचना ggml_webgpu_flash_attn_blk_pipeline_कुञ्चिका {
    uint32_t kv_tile;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_flash_attn_blk_pipeline_कुञ्चिका & other) ध्रुव { प्रत्यागम kv_tile == other.kv_tile; }
};

संरचना ggml_webgpu_flash_attn_blk_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_flash_attn_blk_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.kv_tile);
        प्रत्यागम बीजम्;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
अन्तःस्थ size_t ggml_webgpu_flash_attn_wg_mem_अष्टकानि(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t शीर्ष_dim_qk,
                                                  uint32_t शीर्ष_dim_v,
                                                  तार्किक     has_आवरणम्,
                                                  तार्किक     kv_प्रत्यक्षम्) {
    ध्रुव uint32_t max_शीर्ष_dim = std::max(शीर्ष_dim_qk, शीर्ष_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * शीर्ष_dim_qk;        // q_shmem
    यदि (!kv_प्रत्यक्षम्) {
        f32_elems += kv_tile * max_शीर्ष_dim;  // kv_shmem
    }
    f32_elems += q_tile * शीर्ष_dim_v;         // o_shmem
    यदि (has_आवरणम्) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    प्रत्यागम f16_elems * GGML_WEBGPU_F16_आकार_अष्टकानि + f32_elems * GGML_WEBGPU_F32_आकार_अष्टकानि;
}

अन्तःस्थ uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   सीमा_अष्टकानि,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t शीर्ष_dim_qk,
                                                   uint32_t शीर्ष_dim_v,
                                                   तार्किक     has_आवरणम्,
                                                   तार्किक     kv_प्रत्यक्षम्) {
    ध्रुव size_t आधारः_q_अष्टकानि =
        ggml_webgpu_flash_attn_wg_mem_अष्टकानि(q_tile, 0, शीर्ष_dim_qk, शीर्ष_dim_v, has_आवरणम्, kv_प्रत्यक्षम्);
    यदि (सीमा_अष्टकानि <= आधारः_q_अष्टकानि) {
        प्रत्यागम 0;
    }
    ध्रुव size_t one_kv_अष्टकानि =
        ggml_webgpu_flash_attn_wg_mem_अष्टकानि(q_tile, 1, शीर्ष_dim_qk, शीर्ष_dim_v, has_आवरणम्, kv_प्रत्यक्षम्);
    ध्रुव size_t अष्टकानि_प्रत्येकम्_kv = one_kv_अष्टकानि - आधारः_q_अष्टकानि;
    यदि (अष्टकानि_प्रत्येकम्_kv == 0) {
        प्रत्यागम 0;
    }
    ध्रुव size_t max_kv_tile = (सीमा_अष्टकानि - आधारः_q_अष्टकानि) / अष्टकानि_प्रत्येकम्_kv;
    प्रत्यागम (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

अन्तःस्थ uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_सीमा_अष्टकानि,
                                                       uint32_t शीर्ष_dim_qk,
                                                       uint32_t शीर्ष_dim_v,
                                                       तार्किक     has_आवरणम्,
                                                       तार्किक     kv_प्रत्यक्षम्) {
    ध्रुव uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_सीमा_अष्टकानि, 1u, 1u, शीर्ष_dim_qk, शीर्ष_dim_v, has_आवरणम्, kv_प्रत्यक्षम्);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    यदि (kv_प्रत्यक्षम्) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        यावत् (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    प्रत्यागम kv_tile;
}

अन्तःस्थ तार्किक ggml_webgpu_flash_attn_can_use_subgroup_matrix_मार्गः(तार्किक                supports_subgroup_matrix,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ध्रुव ggml_tensor * Q,
                                                                ध्रुव ggml_tensor * V) {
    प्रत्यागम supports_subgroup_matrix && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

संरचना ggml_webgpu_mul_mat_vec_pipeline_कुञ्चिका {
    ggml_type src0_type;
    ggml_type src1_type;
    पूर्णाङ्क       vectorized;
    uint32_t  अङ्कः_स्तम्भसङ्ख्या;
    तार्किक      use_mmvq;
    तार्किक      src_overlap;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_mul_mat_vec_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               अङ्कः_स्तम्भसङ्ख्या == other.अङ्कः_स्तम्भसङ्ख्या && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_mul_mat_vec_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_mul_mat_vec_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src0_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src1_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.vectorized);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.अङ्कः_स्तम्भसङ्ख्या);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.use_mmvq);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src_overlap);
        प्रत्यागम बीजम्;
    }
};

संरचना ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_आकार;
    uint32_t outputs_प्रत्येकम्_wg;
    uint32_t vec_आकार;
    तार्किक     src_overlap = असत्य;
};

संरचना ggml_webgpu_quantize_q8_pipeline_कुञ्चिका {
    ggml_type src0_type;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_quantize_q8_pipeline_कुञ्चिका & other) ध्रुव { प्रत्यागम src0_type == other.src0_type; }
};

संरचना ggml_webgpu_quantize_q8_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_quantize_q8_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src0_type);
        प्रत्यागम बीजम्;
    }
};

संरचना ggml_webgpu_mul_mat_pipeline_कुञ्चिका {
    ggml_type src0_type;
    ggml_type src1_type;
    पूर्णाङ्क       vectorized;
    पूर्णाङ्क       use_subgroup_matrix;
    तार्किक      src_overlap;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_mul_mat_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_mul_mat_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_mul_mat_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src0_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src1_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.vectorized);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.use_subgroup_matrix);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src_overlap);
        प्रत्यागम बीजम्;
    }
};

संरचना ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_आकार_m;
    uint32_t wg_आकार_n;
    uint32_t wg_आकार;
    uint32_t outputs_प्रत्येकम्_wg;
    पूर्णाङ्क      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_आकार;
    तार्किक     src_overlap = असत्य;
};

/** MUL_MAT_ID **/

संरचना ggml_webgpu_mul_mat_id_gather_pipeline_कुञ्चिका {
    uint32_t n_experts;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_mul_mat_id_gather_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम n_experts == other.n_experts;
    }
};

संरचना ggml_webgpu_mul_mat_id_gather_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_mul_mat_id_gather_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.n_experts);
        प्रत्यागम बीजम्;
    }
};

संरचना ggml_webgpu_mul_mat_id_pipeline_कुञ्चिका {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  अङ्कः_स्तम्भसङ्ख्या;
    पूर्णाङ्क       vectorized;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_mul_mat_id_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               अङ्कः_स्तम्भसङ्ख्या == other.अङ्कः_स्तम्भसङ्ख्या && vectorized == other.vectorized;
    }
};

संरचना ggml_webgpu_mul_mat_id_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_mul_mat_id_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src0_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src1_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.n_experts);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.अङ्कः_स्तम्भसङ्ख्या);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.vectorized);
        प्रत्यागम बीजम्;
    }
};

/** Cpy **/

संरचना ggml_webgpu_cpy_pipeline_कुञ्चिका {
    ggml_type src_type;
    ggml_type dst_type;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_cpy_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम src_type == other.src_type && dst_type == other.dst_type;
    }
};

संरचना ggml_webgpu_cpy_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_cpy_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.dst_type);
        प्रत्यागम बीजम्;
    }
};

/** Glu **/

संरचना ggml_webgpu_glu_pipeline_कुञ्चिका {
    ggml_glu_op glu_op;
    ggml_type   type;
    तार्किक        विभज;
    तार्किक        src_overlap;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_glu_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम glu_op == other.glu_op && type == other.type && विभज == other.विभज && src_overlap == other.src_overlap;
    }
};

संरचना ggml_webgpu_glu_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_glu_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.glu_op);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.विभज);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.src_overlap);
        प्रत्यागम बीजम्;
    }
};

/** Rope **/

संरचना ggml_webgpu_rope_pipeline_कुञ्चिका {
    ggml_type type;
    तार्किक      inplace;
    तार्किक      has_ff;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_rope_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

संरचना ggml_webgpu_rope_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_rope_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.inplace);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.has_ff);
        प्रत्यागम बीजम्;
    }
};

/** SoftMax **/

संरचना ggml_webgpu_soft_max_pipeline_कुञ्चिका {
    ggml_type आवरणम्_type;
    तार्किक      has_आवरणम्;
    तार्किक      has_sink;
    तार्किक      inplace;

    तार्किक संक्रियकः==(ध्रुव ggml_webgpu_soft_max_pipeline_कुञ्चिका & other) ध्रुव {
        प्रत्यागम आवरणम्_type == other.आवरणम्_type && has_आवरणम् == other.has_आवरणम् && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

संरचना ggml_webgpu_soft_max_pipeline_कुञ्चिका_hash {
    size_t संक्रियकः()(ध्रुव ggml_webgpu_soft_max_pipeline_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t बीजम् = 0;
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.आवरणम्_type);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.has_आवरणम्);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.has_sink);
        ggml_webgpu_hash_combine(बीजम्, कुञ्चिका.inplace);
        प्रत्यागम बीजम्;
    }
};

/** MMVQ **/

अन्तःस्थ तार्किक ggml_webgpu_can_use_mmvq(ध्रुव ggml_tensor * src0,
                                     ध्रुव ggml_tensor * src1,
                                     तार्किक                supports_dot_गुणनफलम्,
                                     ध्रुव std::string & vendor) {
    यदि (src1->ne[1] <= 4) {
        तार्किक supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        यदि (supports_dp4a && supports_dot_गुणनफलम्) {
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

    std::unordered_map<पूर्णाङ्क, webgpu_pipeline> योग_पङ्क्तिकाः_pipelines;       // key is fixed, no variants yet
    std::unordered_map<पूर्णाङ्क, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<पूर्णाङ्क, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<पूर्णाङ्क, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<पूर्णाङ्क, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_कुञ्चिका_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_पङ्क्तिकाः_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_get_पङ्क्तिकाः_pipeline_कुञ्चिका_hash>
        get_पङ्क्तिकाः_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_unary_pipeline_कुञ्चिका_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_scale_pipeline_कुञ्चिका_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_कुञ्चिका_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_कुञ्चिका_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_अवलोकनम्_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_ssm_अवलोकनम्_pipeline_कुञ्चिका_hash>
        ssm_अवलोकनम्_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_कुञ्चिका,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_कुञ्चिका_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_pad_pipeline_कुञ्चिका_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_द्विमान_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_द्विमान_pipeline_कुञ्चिका_hash>
        द्विमान_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_सङ्कलय_id_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_सङ्कलय_id_pipeline_कुञ्चिका_hash>
        सङ्कलय_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_concat_pipeline_कुञ्चिका_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_repeat_pipeline_कुञ्चिका_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_कुञ्चिका,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_कुञ्चिका_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_कुञ्चिका_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_कुञ्चिका,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_कुञ्चिका_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_कुञ्चिका,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_कुञ्चिका_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_कुञ्चिका_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_कुञ्चिका_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_कुञ्चिका_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_कुञ्चिका,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_कुञ्चिका_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_कुञ्चिका_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_कुञ्चिका_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_समुच्चयः_पङ्क्तिकाः_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_समुच्चयः_पङ्क्तिकाः_pipeline_कुञ्चिका_hash>
        समुच्चयः_पङ्क्तिकाः_pipelines;
    std::unordered_map<ggml_webgpu_समुच्चयः_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_समुच्चयः_pipeline_कुञ्चिका_hash> समुच्चयः_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_cpy_pipeline_कुञ्चिका_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_glu_pipeline_कुञ्चिका_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_rope_pipeline_कुञ्चिका_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_कुञ्चिका_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_कुञ्चिका_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_कुञ्चिका_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_im2col_pipeline_कुञ्चिका_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_कुञ्चिका,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_कुञ्चिका_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_कुञ्चिका, webgpu_pipeline, ggml_webgpu_upscale_pipeline_कुञ्चिका_hash>
        upscale_pipelines;

  सार्वजनिक:
    ggml_webgpu_shader_lib(wgpu::Device device) { एतत्->device = device; }

    webgpu_pipeline get_योग_पङ्क्तिकाः_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        स्वचालित it = योग_पङ्क्तिकाः_pipelines.find(1);
        यदि (it != योग_पङ्क्तिकाः_pipelines.end()) {
            प्रत्यागम it->second;
        }
        std::vector<std::string> परिभाषते;
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed        = preprocessor.preprocess(wgsl_योग_पङ्क्तिकाः, परिभाषते);
        योग_पङ्क्तिकाः_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        प्रत्यागम योग_पङ्क्तिकाः_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.op                                = context.dst->op;
        कुञ्चिका.src_type                          = context.src0->type;
        कुञ्चिका.dst_type                          = context.dst->type;
        कुञ्चिका.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वचालित it = row_norm_pipelines.find(कुञ्चिका);
        यदि (it != row_norm_pipelines.end()) {
            प्रत्यागम it->second;
        }
        std::vector<std::string> परिभाषते;
        std::string              variant;

        विकल्प (कुञ्चिका.op) {
            पक्ष GGML_OP_RMS_NORM:
                परिभाषते.push_back("RMS_NORM");
                variant = "rms_norm";
                भङ्ग;
            पक्ष GGML_OP_NORM:
                परिभाषते.push_back("NORM");
                variant = "norm";
                भङ्ग;
            पक्ष GGML_OP_L2_NORM:
                परिभाषते.push_back("L2_NORM");
                variant = "l2_norm";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        यदि (कुञ्चिका.inplace) {
            परिभाषते.push_back("INPLACE");
            variant += "_inplace";
        }

        यदि (कुञ्चिका.src_type == GGML_TYPE_F32) {
            परिभाषते.push_back("SRC_F32");
            variant += "_src_f32";
        } अन्यथा यदि (कुञ्चिका.src_type == GGML_TYPE_F16) {
            परिभाषते.push_back("SRC_F16");
            variant += "_src_f16";
        }

        यदि (कुञ्चिका.dst_type == GGML_TYPE_F32) {
            परिभाषते.push_back("DST_F32");
            variant += "_dst_f32";
        } अन्यथा यदि (कुञ्चिका.dst_type == GGML_TYPE_F16) {
            परिभाषते.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ध्रुव uint32_t row_norm_wg_आकार = 128u;
        uint32_t       wg_आकार          = std::min(context.max_wg_आकार, row_norm_wg_आकार);
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        स्वचालित processed                  = preprocessor.preprocess(wgsl_row_norm, परिभाषते);
        स्वचालित decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार              = wg_आकार;
        decisions->inplace              = कुञ्चिका.inplace;
        row_norm_pipelines[कुञ्चिका]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[कुञ्चिका].context = decisions;
        प्रत्यागम row_norm_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_argmax_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        तार्किक vec4 = context.src0->ne[0] % 4 == 0;

        स्वचालित it = argmax_pipelines.find(vec4);
        यदि (it != argmax_pipelines.end()) {
            प्रत्यागम it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> परिभाषते;
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));
        यदि (vec4) {
            परिभाषते.push_back("VEC4");
            variant += "_vec4";
        }

        स्वचालित processed         = preprocessor.preprocess(wgsl_argmax, परिभाषते);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        प्रत्यागम argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_समुच्चयः_पङ्क्तिकाः_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ध्रुव तार्किक                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_समुच्चयः_पङ्क्तिकाः_pipeline_कुञ्चिका कुञ्चिका       = {};
        कुञ्चिका.src0_type                               = context.src0->type;
        कुञ्चिका.dst_type                                = context.dst->type;
        कुञ्चिका.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        कुञ्चिका.i64_idx     = context.src1->type == GGML_TYPE_I64;
        कुञ्चिका.pair_खण्डानि = quantized && ((context.src0->ne[0] / ggml_blck_आकार(context.dst->type)) % 2 == 0);

        स्वचालित it = समुच्चयः_पङ्क्तिकाः_pipelines.find(कुञ्चिका);
        यदि (it != समुच्चयः_पङ्क्तिकाः_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "set_rows";

        विकल्प (context.dst->type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("DST_F32");
                variant += "_dstf32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("DST_F16");
                variant += "_dstf16";
                भङ्ग;
            पक्ष GGML_TYPE_Q8_0:
                परिभाषते.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                भङ्ग;
            पक्ष GGML_TYPE_Q4_0:
                परिभाषते.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        यदि (context.src0->type == GGML_TYPE_F16) {
            परिभाषते.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        यदि (कुञ्चिका.vec4) {
            परिभाषते.push_back("VEC4");
            variant += "_vec4";
        }
        यदि (कुञ्चिका.i64_idx) {
            परिभाषते.push_back("I64_IDX");
            variant += "_i64idx";
        }
        यदि (कुञ्चिका.pair_खण्डानि) {
            परिभाषते.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        ध्रुव स्वचालित & shader_source      = quantized ? wgsl_समुच्चयः_पङ्क्तिकाः_quant : wgsl_समुच्चयः_पङ्क्तिकाः;
        स्वचालित         processed          = preprocessor.preprocess(shader_source, परिभाषते);
        स्वचालित         decisions          = std::make_shared<ggml_webgpu_समुच्चयः_पङ्क्तिकाः_shader_decisions>();
        decisions->vec4                 = कुञ्चिका.vec4;
        decisions->i64_idx              = कुञ्चिका.i64_idx;
        decisions->pair_खण्डानि          = कुञ्चिका.pair_खण्डानि;
        decisions->wg_आकार              = context.max_wg_आकार;
        समुच्चयः_पङ्क्तिकाः_pipelines[कुञ्चिका]         = ggml_webgpu_create_pipeline(device, processed, variant);
        समुच्चयः_पङ्क्तिकाः_pipelines[कुञ्चिका].context = decisions;
        प्रत्यागम समुच्चयः_पङ्क्तिकाः_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_समुच्चयः_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_समुच्चयः_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.type                         = context.dst->type;
        कुञ्चिका.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वचालित it = समुच्चयः_pipelines.find(कुञ्चिका);
        यदि (it != समुच्चयः_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "set";

        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                परिभाषते.push_back("TYPE_I32");
                variant += "_i32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for set shader");
        }

        यदि (कुञ्चिका.inplace) {
            परिभाषते.push_back("INPLACE");
            variant += "_inplace";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_समुच्चयः, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->inplace       = कुञ्चिका.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        समुच्चयः_pipelines[कुञ्चिका]       = pipeline;
        प्रत्यागम समुच्चयः_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_cumsum_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        स्वचालित it = cumsum_pipelines.find(1);
        यदि (it != cumsum_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed      = preprocessor.preprocess(wgsl_cumsum, परिभाषते);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        प्रत्यागम cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        तार्किक          is_शिखरम्_k = context.dst->op == GGML_OP_शिखरम्_K;
        // ascending order is 0, descending order is 1
        ध्रुव int32_t क्रमः =
            is_शिखरम्_k ? (int32_t) GGML_क्रमय_क्रमः_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        स्वचालित it = argsort_pipelines.find(क्रमः);
        यदि (it != argsort_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "argsort";
        परिभाषते.push_back(std::string("ORDER=") + std::to_string(क्रमः));
        variant += std::string("_order") + std::to_string(क्रमः);
        uint32_t wg_आकार = 1;
        यावत् (wg_आकार * 2 <= context.max_wg_आकार &&
               wg_आकार * GGML_WEBGPU_I32_आकार_अष्टकानि <= context.wg_mem_सीमा_अष्टकानि / 2) {
            wg_आकार *= 2;
        }
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        स्वचालित processed                   = preprocessor.preprocess(wgsl_argsort, परिभाषते);
        स्वचालित decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार               = wg_आकार;
        argsort_pipelines[क्रमः]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[क्रमः].context = decisions;
        प्रत्यागम argsort_pipelines[क्रमः];
    }

    webgpu_pipeline get_argsort_merge_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        तार्किक          is_शिखरम्_k = context.dst->op == GGML_OP_शिखरम्_K;
        // ascending order is 0, descending order is 1
        ध्रुव int32_t क्रमः =
            is_शिखरम्_k ? (int32_t) GGML_क्रमय_क्रमः_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        स्वचालित it = argsort_merge_pipelines.find(क्रमः);
        यदि (it != argsort_merge_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "argsort_merge";
        परिभाषते.push_back(std::string("ORDER=") + std::to_string(क्रमः));
        variant += std::string("_order") + std::to_string(क्रमः);
        uint32_t wg_आकार = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_आकार, context.max_wg_आकार);
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));

        स्वचालित processed                 = preprocessor.preprocess(wgsl_argsort_merge, परिभाषते);
        argsort_merge_pipelines[क्रमः] = ggml_webgpu_create_pipeline(device, processed, variant);
        प्रत्यागम argsort_merge_pipelines[क्रमः];
    }

    webgpu_pipeline get_get_पङ्क्तिकाः_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक vec4_aligned) {
        ध्रुव तार्किक vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_पङ्क्तिकाः_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.src_type                          = context.src0->type;
        कुञ्चिका.vectorized                        = (पूर्णाङ्क) vectorized;

        स्वचालित it = get_पङ्क्तिकाः_pipelines.find(कुञ्चिका);
        यदि (it != get_पङ्क्तिकाः_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "get_rows";

        ध्रुव संरचना ggml_type_traits * type_traits = ggml_get_type_traits(कुञ्चिका.src_type);
        ध्रुव वर्ण *                    type_str    = type_traits->type_name;

        विकल्प (कुञ्चिका.src_type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("FLOAT_PARALLEL");
                यदि (कुञ्चिका.vectorized) {
                    परिभाषते.push_back("F32_VEC");
                    परिभाषते.push_back("SRC_TYPE=vec4<f32>");
                    परिभाषते.push_back("DST_TYPE=vec4<f32>");
                    परिभाषते.push_back("BLOCK_SIZE=4u");
                } अन्यथा {
                    परिभाषते.push_back("F32");
                    परिभाषते.push_back("SRC_TYPE=f32");
                    परिभाषते.push_back("DST_TYPE=f32");
                    परिभाषते.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("FLOAT_PARALLEL");
                परिभाषते.push_back("F16");
                परिभाषते.push_back("SRC_TYPE=f16");
                परिभाषते.push_back("DST_TYPE=f32");
                परिभाषते.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                परिभाषते.push_back("FLOAT_PARALLEL");
                परिभाषते.push_back("I32");
                परिभाषते.push_back("SRC_TYPE=i32");
                परिभाषते.push_back("DST_TYPE=i32");
                परिभाषते.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                भङ्ग;
            पक्ष GGML_TYPE_BF16:
                परिभाषते.push_back("BF16");
                परिभाषते.push_back("SRC_TYPE=u32");
                परिभाषते.push_back("DST_TYPE=f32");
                परिभाषते.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                भङ्ग;
            स्वभाव:
                {
                    std::string type_ऊर्ध्वस्थम् = type_str;
                    std::transform(type_ऊर्ध्वस्थम्.begin(), type_ऊर्ध्वस्थम्.end(), type_ऊर्ध्वस्थम्.begin(), ::महाक्षरम्);

                    विकल्प (कुञ्चिका.src_type) {
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
                                परिभाषते.push_back("SRC_TYPE=u32");
                                परिभाषते.push_back("U32_DEQUANT_HELPERS");
                                भङ्ग;
                            }
                        स्वभाव:
                            {
                                परिभाषते.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    परिभाषते.push_back("BYTE_HELPERS");
                    परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_T");
                    परिभाषते.push_back(type_ऊर्ध्वस्थम्);
                    परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_SCALE_MIN");
                    परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_TABLES");
                    परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_GRID");
                    परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_LUT");

                    variant += "_";
                    variant += type_str;

                    परिभाषते.push_back("DST_TYPE=f32");

                    यदि (कुञ्चिका.src_type == GGML_TYPE_Q1_0) {
                        परिभाषते.push_back("BLOCK_SIZE=128u");
                    } अन्यथा यदि ((कुञ्चिका.src_type >= GGML_TYPE_Q4_0 && कुञ्चिका.src_type <= GGML_TYPE_Q8_1) ||
                               कुञ्चिका.src_type == GGML_TYPE_IQ4_NL || कुञ्चिका.src_type == GGML_TYPE_MXFP4) {
                        परिभाषते.push_back("BLOCK_SIZE=32u");
                    } अन्यथा यदि (कुञ्चिका.src_type == GGML_TYPE_Q2_0 || कुञ्चिका.src_type == GGML_TYPE_NVFP4) {
                        परिभाषते.push_back("BLOCK_SIZE=64u");
                    } अन्यथा यदि (कुञ्चिका.src_type >= GGML_TYPE_Q2_K) {
                        परिभाषते.push_back("BLOCK_SIZE=256u");
                    } अन्यथा {
                        परिभाषते.push_back("BLOCK_SIZE=1u");
                    }
                    भङ्ग;
                }
        }

        यदि (कुञ्चिका.vectorized) {
            variant += "_vec";
        }

        परिभाषते.push_back("WG_SIZE=" + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_get_पङ्क्तिकाः, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_get_पङ्क्तिकाः_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_पङ्क्तिकाः_pipelines[कुञ्चिका]  = pipeline;
        प्रत्यागम get_पङ्क्तिकाः_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_scale_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वचालित it = scale_pipelines.find(कुञ्चिका);
        यदि (it != scale_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "scale";

        यदि (कुञ्चिका.inplace) {
            परिभाषते.push_back("INPLACE");
            variant += "_inplace";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_scale, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->inplace       = कुञ्चिका.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[कुञ्चिका]     = pipeline;
        प्रत्यागम scale_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_solve_tri_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.type                               = context.dst->type;
        कुञ्चिका.n                                  = (पूर्णाङ्क) context.src0->ne[0];
        कुञ्चिका.k                                  = (पूर्णाङ्क) context.src1->ne[0];

        स्वचालित it = solve_tri_pipelines.find(कुञ्चिका);
        यदि (it != solve_tri_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "solve_tri";

        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                variant += "_f32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ध्रुव uint32_t wg_आकार       = std::min((uint32_t) कुञ्चिका.n, context.max_wg_आकार);
        ध्रुव uint32_t k_tile        = wg_आकार;
        ध्रुव uint32_t अष्टकानि_प्रत्येकम्_row = ((uint32_t) कुञ्चिका.n + wg_आकार) * GGML_WEBGPU_F32_आकार_अष्टकानि;
        ध्रुव uint32_t batch_n       = (uint32_t) (context.wg_mem_सीमा_अष्टकानि / अष्टकानि_प्रत्येकम्_row);

        परिभाषते.push_back(std::string("N=") + std::to_string(कुञ्चिका.n));
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        परिभाषते.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        परिभाषते.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        स्वचालित processed           = preprocessor.preprocess(wgsl_solve_tri, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम solve_tri_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.type                              = context.dst->type;
        कुञ्चिका.vectorized                        = context.src1->ne[0] == 4;

        स्वचालित it = ssm_conv_pipelines.find(कुञ्चिका);
        यदि (it != ssm_conv_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "ssm_conv";

        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                variant += "_f32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        यदि (कुञ्चिका.vectorized) {
            परिभाषते.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t खण्डम्_आकार_3    = 32u;
        constexpr uint32_t tokens_प्रत्येकम्_wg = 8u;

        परिभाषते.push_back("BLOCK_SIZE=" + std::to_string(खण्डम्_आकार_3) + "u");
        परिभाषते.push_back("TOKENS_PER_WG=" + std::to_string(tokens_प्रत्येकम्_wg) + "u");

        स्वचालित processed           = preprocessor.preprocess(wgsl_ssm_conv, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->खण्डम्_आकार_3    = खण्डम्_आकार_3;
        decisions->tokens_प्रत्येकम्_wg = tokens_प्रत्येकम्_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[कुञ्चिका]  = pipeline;
        प्रत्यागम ssm_conv_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_ssm_अवलोकनम्_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_अवलोकनम्_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.type                              = context.dst->type;
        कुञ्चिका.d_state                           = (पूर्णाङ्क) context.src0->ne[0];
        कुञ्चिका.xbc_overlap                       = xbc_overlap;

        स्वचालित it = ssm_अवलोकनम्_pipelines.find(कुञ्चिका);
        यदि (it != ssm_अवलोकनम्_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "ssm_scan";

        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                variant += "_f32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ध्रुव uint32_t wg_आकार = (uint32_t) कुञ्चिका.d_state;

        constexpr uint32_t tokens_प्रत्येकम्_tile = 4u;

        परिभाषते.push_back("WG_SIZE=" + std::to_string(wg_आकार) + "u");
        परिभाषते.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_प्रत्येकम्_tile) + "u");

        यदि (context.supports_subgroups) {
            परिभाषते.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } अन्यथा {
            variant += "_wg_reduce";
        }

        यदि (कुञ्चिका.xbc_overlap == 0b110) {  // x/B
            परिभाषते.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } अन्यथा यदि (कुञ्चिका.xbc_overlap == 0b011) {  // B/C
            परिभाषते.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } अन्यथा यदि (कुञ्चिका.xbc_overlap == 0b111) {  // x/B/C
            परिभाषते.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(कुञ्चिका.d_state);

        स्वचालित processed             = preprocessor.preprocess(wgsl_ssm_अवलोकनम्, परिभाषते);
        स्वचालित decisions             = std::make_shared<ggml_webgpu_ssm_अवलोकनम्_shader_decisions>();
        decisions->wg_आकार         = wg_आकार;
        decisions->tokens_प्रत्येकम्_tile = tokens_प्रत्येकम्_tile;
        decisions->xbc_overlap     = कुञ्चिका.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_अवलोकनम्_pipelines[कुञ्चिका]    = pipeline;
        प्रत्यागम ssm_अवलोकनम्_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.type                                     = context.dst->type;
        कुञ्चिका.s_v                                      = (पूर्णाङ्क) context.src2->ne[0];
        कुञ्चिका.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        कुञ्चिका.fused_cache                              = context.dst_fuse != शून्यसूचकः;

        स्वचालित it = gated_delta_net_pipelines.find(कुञ्चिका);
        यदि (it != gated_delta_net_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "gated_delta_net";

        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                variant += "_f32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        यदि (कुञ्चिका.kda) {
            परिभाषते.push_back("KDA");
            variant += "_kda";
        }

        यदि (कुञ्चिका.fused_cache) {
            परिभाषते.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        परिभाषते.push_back("S_V=" + std::to_string(कुञ्चिका.s_v) + "u");
        परिभाषते.push_back("WG_SIZE=" + std::to_string(कुञ्चिका.s_v) + "u");

        स्वचालित            processed      = preprocessor.preprocess(wgsl_gated_delta_net, परिभाषते);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम gated_delta_net_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_pad_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        स्वचालित it = pad_pipelines.find(कुञ्चिका);
        यदि (it != pad_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "pad";

        यदि (कुञ्चिका.circular) {
            परिभाषते.push_back("CIRCULAR");
            variant += "_circular";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_pad, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[कुञ्चिका]       = pipeline;
        प्रत्यागम pad_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.src0_type                            = context.src0->type;

        स्वचालित it = quantize_q8_pipelines.find(कुञ्चिका);
        यदि (it != quantize_q8_pipelines.end()) {
            प्रत्यागम it->second;
        }
        ध्रुव वर्ण *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> परिभाषते;
        std::string              variant = "quantize_q8";

        uint32_t wg_आकार = WEBGPU_MUL_MAT_VEC_WG_आकार;

        परिभाषते.push_back("SRC1_INNER_TYPE=f32");
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));

        ध्रुव संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_ऊर्ध्वस्थम्  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_ऊर्ध्वस्थम्.begin(), type_ऊर्ध्वस्थम्.end(), type_ऊर्ध्वस्थम्.begin(), ::महाक्षरम्);

        परिभाषते.push_back("MUL_ACC_" + type_ऊर्ध्वस्थम्);
        परिभाषते.push_back("Q8_1_T");

        परिभाषते.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        स्वचालित processed             = preprocessor.preprocess(shader_src, परिभाषते);
        स्वचालित decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार         = wg_आकार;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम quantize_q8_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.src0_type                            = context.src0->type;
        कुञ्चिका.src1_type                            = context.src1->type;
        कुञ्चिका.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        कुञ्चिका.अङ्कः_स्तम्भसङ्ख्या   = context.dst->ne[1];
        कुञ्चिका.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_गुणनफलम्, context.vendor);
        कुञ्चिका.src_overlap = src_overlap;

        स्वचालित it = mul_mat_vec_pipelines.find(कुञ्चिका);
        यदि (it != mul_mat_vec_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant    = "mul_mat_vec";
        ध्रुव वर्ण *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        विकल्प (context.src0->type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("SRC0_INNER_TYPE=f32");
                परिभाषते.push_back("MUL_ACC_FLOAT");
                परिभाषते.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("SRC0_INNER_TYPE=f16");
                परिभाषते.push_back("MUL_ACC_FLOAT");
                परिभाषते.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_BF16:
                परिभाषते.push_back("SRC0_INNER_TYPE=u32");
                परिभाषते.push_back("MUL_ACC_FLOAT");
                परिभाषते.push_back("TYPE_BF16");
                variant += "_bf16";
                भङ्ग;
            स्वभाव:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ध्रुव संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_ऊर्ध्वस्थम्  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_ऊर्ध्वस्थम्.begin(), type_ऊर्ध्वस्थम्.end(), type_ऊर्ध्वस्थम्.begin(), ::महाक्षरम्);

                    परिभाषते.push_back("BYTE_HELPERS");
                    परिभाषते.push_back("MUL_ACC_" + type_ऊर्ध्वस्थम्);
                    परिभाषते.push_back("U32_DEQUANT_HELPERS");
                    परिभाषते.push_back("SRC0_INNER_TYPE=u32");
                    विकल्प (context.src0->type) {
                        पक्ष GGML_TYPE_Q4_0:
                        पक्ष GGML_TYPE_Q4_1:
                        पक्ष GGML_TYPE_Q5_0:
                        पक्ष GGML_TYPE_Q5_1:
                        पक्ष GGML_TYPE_Q8_0:
                            यदि (कुञ्चिका.use_mmvq) {
                                परिभाषते.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            भङ्ग;
                        पक्ष GGML_TYPE_Q1_0:
                        पक्ष GGML_TYPE_Q2_K:
                        पक्ष GGML_TYPE_Q3_K:
                        पक्ष GGML_TYPE_Q4_K:
                        पक्ष GGML_TYPE_Q5_K:
                        पक्ष GGML_TYPE_Q6_K:
                            यदि (कुञ्चिका.use_mmvq) {
                                परिभाषते.push_back("K_QUANTS_HANDLING");
                            }
                            भङ्ग;
                        पक्ष GGML_TYPE_IQ1_S:
                        पक्ष GGML_TYPE_IQ1_M:
                        पक्ष GGML_TYPE_IQ2_S:
                        पक्ष GGML_TYPE_IQ3_S:
                        पक्ष GGML_TYPE_IQ4_NL:
                        पक्ष GGML_TYPE_IQ4_XS:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_GRID");
                            भङ्ग;
                        पक्ष GGML_TYPE_IQ2_XXS:
                        पक्ष GGML_TYPE_IQ2_XS:
                        पक्ष GGML_TYPE_IQ3_XXS:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_GRID");
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_TABLES");
                            भङ्ग;
                        पक्ष GGML_TYPE_MXFP4:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_LUT");
                            यदि (कुञ्चिका.use_mmvq) {
                                परिभाषते.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            भङ्ग;
                        पक्ष GGML_TYPE_NVFP4:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_LUT");
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
                परिभाषते.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        परिभाषते.push_back(कुञ्चिका.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_आकार        = WEBGPU_MUL_MAT_VEC_WG_आकार;
        uint32_t outputs_प्रत्येकम्_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_प्रत्येकम्_WG;

        यदि (कुञ्चिका.src0_type == GGML_TYPE_Q1_0) {
            outputs_प्रत्येकम्_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_प्रत्येकम्_WG;
        } अन्यथा यदि (कुञ्चिका.src0_type >= GGML_TYPE_Q2_K) {
            outputs_प्रत्येकम्_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_प्रत्येकम्_WG;
        } अन्यथा यदि (कुञ्चिका.src0_type >= GGML_TYPE_Q4_0) {
            outputs_प्रत्येकम्_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_प्रत्येकम्_WG;
        }

        यदि (कुञ्चिका.use_mmvq) {
            परिभाषते.push_back("MMVQ");
            परिभाषते.push_back("Q8_1_T");
        }

        यदि (कुञ्चिका.src_overlap) {
            परिभाषते.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        परिभाषते.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_प्रत्येकम्_wg));
        परिभाषते.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        यदि (कुञ्चिका.vectorized) {
            variant += "_vectorized";
        }
        परिभाषते.push_back(std::string("NUM_COLS=") + std::to_string(कुञ्चिका.अङ्कः_स्तम्भसङ्ख्या));

        स्वचालित processed            = preprocessor.preprocess(shader_src, परिभाषते);
        स्वचालित decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_आकार        = wg_आकार;
        decisions->outputs_प्रत्येकम्_wg = outputs_प्रत्येकम्_wg;
        decisions->vec_आकार       = कुञ्चिका.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम mul_mat_vec_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_mul_mat_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.src0_type                        = context.src0->type;
        कुञ्चिका.src1_type                        = context.src1->type;
        कुञ्चिका.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        कुञ्चिका.use_subgroup_matrix = context.supports_subgroup_matrix;
        कुञ्चिका.src_overlap         = src_overlap;

        स्वचालित it = mul_mat_fast_pipelines.find(कुञ्चिका);
        यदि (it != mul_mat_fast_pipelines.end()) {
            प्रत्यागम it->second;
        }

        ध्रुव वर्ण * shader_src = कुञ्चिका.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> परिभाषते;
        std::string              variant = कुञ्चिका.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        विकल्प (context.src1->type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("SRC1_INNER_TYPE=f32");
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("SRC1_INNER_TYPE=f16");
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ध्रुव संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ध्रुव वर्ण *                    src0_name   = src0_traits->type_name;

        विकल्प (context.src0->type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("SRC0_INNER_TYPE=f32");
                परिभाषते.push_back("TYPE_F32");
                परिभाषते.push_back("MUL_ACC_FLOAT");
                परिभाषते.push_back("INIT_SRC0_SHMEM_FLOAT");
                परिभाषते.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("SRC0_INNER_TYPE=f16");
                परिभाषते.push_back("TYPE_F16");
                परिभाषते.push_back("MUL_ACC_FLOAT");
                परिभाषते.push_back("INIT_SRC0_SHMEM_FLOAT");
                परिभाषते.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_BF16:
                परिभाषते.push_back("SRC0_INNER_TYPE=u32");
                परिभाषते.push_back("TYPE_BF16");
                परिभाषते.push_back("MUL_ACC_FLOAT");
                परिभाषते.push_back("INIT_SRC0_SHMEM_FLOAT");
                परिभाषते.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                भङ्ग;
            स्वभाव:
                {
                    std::string type_ऊर्ध्वस्थम् = src0_name;
                    std::transform(type_ऊर्ध्वस्थम्.begin(), type_ऊर्ध्वस्थम्.end(), type_ऊर्ध्वस्थम्.begin(), ::महाक्षरम्);

                    परिभाषते.push_back("BYTE_HELPERS");
                    परिभाषते.push_back("MUL_ACC_" + type_ऊर्ध्वस्थम्);
                    परिभाषते.push_back("INIT_SRC0_SHMEM_" + type_ऊर्ध्वस्थम्);
                    परिभाषते.push_back("INIT_SRC1_SHMEM_FLOAT");
                    परिभाषते.push_back("U32_DEQUANT_HELPERS");
                    परिभाषते.push_back("SRC0_INNER_TYPE=u32");

                    विकल्प (context.src0->type) {
                        पक्ष GGML_TYPE_IQ1_S:
                        पक्ष GGML_TYPE_IQ1_M:
                        पक्ष GGML_TYPE_IQ4_NL:
                        पक्ष GGML_TYPE_IQ4_XS:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_GRID");
                            भङ्ग;
                        पक्ष GGML_TYPE_IQ2_XXS:
                        पक्ष GGML_TYPE_IQ2_XS:
                        पक्ष GGML_TYPE_IQ2_S:
                        पक्ष GGML_TYPE_IQ3_XXS:
                        पक्ष GGML_TYPE_IQ3_S:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_GRID");
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_TABLES");
                            भङ्ग;
                        पक्ष GGML_TYPE_MXFP4:
                        पक्ष GGML_TYPE_NVFP4:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_LUT");
                            भङ्ग;
                        स्वभाव:
                            भङ्ग;
                    }

                    variant += std::string("_") + src0_name;
                    भङ्ग;
                }
        }

        // VEC/SCALAR controls
        परिभाषते.push_back(कुञ्चिका.vectorized ? "VEC" : "SCALAR");

        ध्रुव तार्किक is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        यदि (कुञ्चिका.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } अन्यथा {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        परिभाषते.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        परिभाषते.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        यदि (कुञ्चिका.use_subgroup_matrix) {
            परिभाषते.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            परिभाषते.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_आकार) + "u");
            परिभाषते.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            परिभाषते.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            परिभाषते.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M) + "u");
            परिभाषते.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N) + "u");
            परिभाषते.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            परिभाषते.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            परिभाषते.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        यदि (कुञ्चिका.vectorized) {
            variant += "_vectorized";
        }

        यदि (कुञ्चिका.src_overlap) {
            परिभाषते.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        यदि (!कुञ्चिका.use_subgroup_matrix) {
            परिभाषते.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_आकार_M) + "u");
            परिभाषते.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_आकार_N) + "u");
            परिभाषते.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        स्वचालित processed = preprocessor.preprocess(shader_src, परिभाषते);

        स्वचालित decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = कुञ्चिका.use_subgroup_matrix;
        यदि (कुञ्चिका.use_subgroup_matrix) {
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
        mul_mat_fast_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम mul_mat_fast_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.n_experts                                  = (uint32_t) context.src0->ne[2];

        स्वचालित it = mul_mat_id_gather_pipelines.find(कुञ्चिका);
        यदि (it != mul_mat_id_gather_pipelines.end()) {
            प्रत्यागम it->second;
        }
        std::vector<std::string> परिभाषते;
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार) + "u");
        परिभाषते.push_back(std::string("N_EXPERT=") + std::to_string(कुञ्चिका.n_experts) + "u");
        परिभाषते.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_आकार_N) + "u");

        स्वचालित processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, परिभाषते);
        स्वचालित decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार = context.max_wg_आकार;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.src0_type                           = context.src0->type;
        कुञ्चिका.src1_type                           = context.src1->type;
        कुञ्चिका.n_experts                           = context.src0->ne[2];
        कुञ्चिका.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        स्वचालित it = mul_mat_id_pipelines.find(कुञ्चिका);
        यदि (it != mul_mat_id_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "mul_mat_id";
        परिभाषते.push_back("MUL_MAT_ID");

        // src1 type
        विकल्प (context.src1->type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("SRC1_INNER_TYPE=f32");
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("SRC1_INNER_TYPE=f16");
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ध्रुव संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ध्रुव वर्ण *                    src0_name   = src0_traits->type_name;

        विकल्प (context.src0->type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("SRC0_INNER_TYPE=f32");
                परिभाषते.push_back("INIT_SRC0_SHMEM_FLOAT");
                परिभाषते.push_back("INIT_SRC1_SHMEM_FLOAT");
                परिभाषते.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("SRC0_INNER_TYPE=f16");
                परिभाषते.push_back("INIT_SRC0_SHMEM_FLOAT");
                परिभाषते.push_back("INIT_SRC1_SHMEM_FLOAT");
                परिभाषते.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_BF16:
                परिभाषते.push_back("SRC0_INNER_TYPE=u32");
                परिभाषते.push_back("INIT_SRC0_SHMEM_FLOAT");
                परिभाषते.push_back("INIT_SRC1_SHMEM_FLOAT");
                परिभाषते.push_back("TYPE_BF16");
                variant += "_bf16";
                भङ्ग;
            स्वभाव:
                {
                    std::string type_ऊर्ध्वस्थम् = src0_name;
                    std::transform(type_ऊर्ध्वस्थम्.begin(), type_ऊर्ध्वस्थम्.end(), type_ऊर्ध्वस्थम्.begin(), ::महाक्षरम्);

                    परिभाषते.push_back("BYTE_HELPERS");
                    परिभाषते.push_back("INIT_SRC0_SHMEM_" + type_ऊर्ध्वस्थम्);
                    परिभाषते.push_back("INIT_SRC1_SHMEM_FLOAT");
                    परिभाषते.push_back("U32_DEQUANT_HELPERS");
                    परिभाषते.push_back("SRC0_INNER_TYPE=u32");

                    विकल्प (context.src0->type) {
                        पक्ष GGML_TYPE_IQ1_S:
                        पक्ष GGML_TYPE_IQ1_M:
                        पक्ष GGML_TYPE_IQ4_NL:
                        पक्ष GGML_TYPE_IQ4_XS:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_GRID");
                            भङ्ग;
                        पक्ष GGML_TYPE_IQ2_XXS:
                        पक्ष GGML_TYPE_IQ2_XS:
                        पक्ष GGML_TYPE_IQ2_S:
                        पक्ष GGML_TYPE_IQ3_XXS:
                        पक्ष GGML_TYPE_IQ3_S:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_GRID");
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_TABLES");
                            भङ्ग;
                        पक्ष GGML_TYPE_MXFP4:
                        पक्ष GGML_TYPE_NVFP4:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_LUT");
                            भङ्ग;
                        स्वभाव:
                            भङ्ग;
                    }

                    variant += std::string("_") + src0_name;
                    भङ्ग;
                }
        }

        // VEC/SCALAR controls
        परिभाषते.push_back(कुञ्चिका.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ध्रुव uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        परिभाषते.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        परिभाषते.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        परिभाषते.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        परिभाषते.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_आकार_M) + "u");
        परिभाषते.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_आकार_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        यदि (कुञ्चिका.vectorized) {
            variant += "_vectorized";
        }

        स्वचालित processed = preprocessor.preprocess(wgsl_mul_mat_id, परिभाषते);

        स्वचालित decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_आकार_m = WEBGPU_MUL_MAT_WG_आकार_M;
        decisions->wg_आकार_n = WEBGPU_MUL_MAT_WG_आकार_N;
        decisions->wg_आकार   = WEBGPU_MUL_MAT_WG_आकार_M * WEBGPU_MUL_MAT_WG_आकार_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम mul_mat_id_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.src0_type                           = context.src0->type;
        कुञ्चिका.src1_type                           = context.src1->type;
        कुञ्चिका.n_experts                           = context.src0->ne[2];
        कुञ्चिका.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        स्वचालित it = mul_mat_id_vec_pipelines.find(कुञ्चिका);
        यदि (it != mul_mat_id_vec_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant    = "mul_mat_id_vec";
        ध्रुव वर्ण *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        विकल्प (context.src1->type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("SRC1_INNER_TYPE=f32");
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("SRC1_INNER_TYPE=f16");
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        विकल्प (context.src0->type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("SRC0_INNER_TYPE=f32");
                परिभाषते.push_back("MUL_ACC_FLOAT");
                परिभाषते.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("SRC0_INNER_TYPE=f16");
                परिभाषते.push_back("MUL_ACC_FLOAT");
                परिभाषते.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_BF16:
                परिभाषते.push_back("SRC0_INNER_TYPE=u32");
                परिभाषते.push_back("MUL_ACC_FLOAT");
                परिभाषते.push_back("TYPE_BF16");
                variant += "_bf16";
                भङ्ग;
            स्वभाव:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ध्रुव संरचना ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_ऊर्ध्वस्थम्  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_ऊर्ध्वस्थम्.begin(), type_ऊर्ध्वस्थम्.end(), type_ऊर्ध्वस्थम्.begin(), ::महाक्षरम्);

                    परिभाषते.push_back("BYTE_HELPERS");
                    परिभाषते.push_back("MUL_ACC_" + type_ऊर्ध्वस्थम्);
                    परिभाषते.push_back("U32_DEQUANT_HELPERS");
                    परिभाषते.push_back("SRC0_INNER_TYPE=u32");
                    विकल्प (context.src0->type) {
                        पक्ष GGML_TYPE_IQ1_S:
                        पक्ष GGML_TYPE_IQ1_M:
                        पक्ष GGML_TYPE_IQ2_S:
                        पक्ष GGML_TYPE_IQ3_S:
                        पक्ष GGML_TYPE_IQ4_NL:
                        पक्ष GGML_TYPE_IQ4_XS:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_GRID");
                            भङ्ग;
                        पक्ष GGML_TYPE_IQ2_XXS:
                        पक्ष GGML_TYPE_IQ2_XS:
                        पक्ष GGML_TYPE_IQ3_XXS:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_GRID");
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_TABLES");
                            भङ्ग;
                        पक्ष GGML_TYPE_MXFP4:
                        पक्ष GGML_TYPE_NVFP4:
                            परिभाषते.push_back(type_ऊर्ध्वस्थम् + "_LUT");
                            भङ्ग;
                        स्वभाव:
                            भङ्ग;
                    }
                    भङ्ग;
                }
        }

        // VEC/SCALAR controls
        परिभाषते.push_back(कुञ्चिका.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_आकार        = WEBGPU_MUL_MAT_VEC_WG_आकार;
        uint32_t outputs_प्रत्येकम्_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_प्रत्येकम्_WG;

        यदि (कुञ्चिका.src0_type == GGML_TYPE_Q1_0) {
            outputs_प्रत्येकम्_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_प्रत्येकम्_WG;
        } अन्यथा यदि (कुञ्चिका.src0_type >= GGML_TYPE_Q2_K) {
            outputs_प्रत्येकम्_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_प्रत्येकम्_WG;
        } अन्यथा यदि (कुञ्चिका.src0_type >= GGML_TYPE_Q4_0) {
            outputs_प्रत्येकम्_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_प्रत्येकम्_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        परिभाषते.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_प्रत्येकम्_wg));
        परिभाषते.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        यदि (कुञ्चिका.vectorized) {
            variant += "_vectorized";
        }
        परिभाषते.push_back(std::string("NUM_COLS=1"));

        परिभाषते.push_back(std::string("N_EXPERTS=") + std::to_string(कुञ्चिका.n_experts));

        स्वचालित processed = preprocessor.preprocess(shader_src, परिभाषते);

        स्वचालित decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_आकार        = wg_आकार;
        decisions->outputs_प्रत्येकम्_wg = outputs_प्रत्येकम्_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम mul_mat_id_vec_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_unary_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ध्रुव तार्किक                     is_unary = context.dst->op == GGML_OP_UNARY;
        ध्रुव पूर्णाङ्क                      op       = is_unary ? (पूर्णाङ्क) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_कुञ्चिका कुञ्चिका      = {};
        कुञ्चिका.type                                = context.dst->type;
        कुञ्चिका.op                                  = op;
        कुञ्चिका.is_unary                            = is_unary;
        कुञ्चिका.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        कुञ्चिका.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        स्वचालित it = unary_pipelines.find(कुञ्चिका);
        यदि (it != unary_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant =
            कुञ्चिका.is_unary ? ggml_unary_op_name((ggml_unary_op) कुञ्चिका.op) : ggml_op_name((ggml_op) कुञ्चिका.op);
        परिभाषते.push_back(variant);

        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for unary shader");
        }

        यदि (कुञ्चिका.inplace) {
            परिभाषते.push_back("INPLACE");
            variant += "_inplace";
        }

        यदि (op == GGML_OP_TRI) {
            विकल्प (कुञ्चिका.ttype) {
                पक्ष GGML_TRI_TYPE_अधस्तनम्:
                    परिभाषते.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    भङ्ग;
                पक्ष GGML_TRI_TYPE_अधस्तनम्_DIAG:
                    परिभाषते.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    भङ्ग;
                पक्ष GGML_TRI_TYPE_ऊर्ध्वस्थम्:
                    परिभाषते.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    भङ्ग;
                पक्ष GGML_TRI_TYPE_ऊर्ध्वस्थम्_DIAG:
                    परिभाषते.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    भङ्ग;
                स्वभाव:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_unary, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->inplace       = कुञ्चिका.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[कुञ्चिका]     = pipeline;
        प्रत्यागम unary_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        कुञ्चिका.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        कुञ्चिका.src_overlap                           = src_overlap;

        स्वचालित it = rms_norm_mul_pipelines.find(कुञ्चिका);
        यदि (it != rms_norm_mul_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        यदि (कुञ्चिका.inplace) {
            परिभाषते.push_back("INPLACE");
            variant += "_inplace";
        } अन्यथा यदि (कुञ्चिका.overlap) {
            परिभाषते.push_back("OVERLAP");
            variant += "_overlap";
        } अन्यथा यदि (कुञ्चिका.src_overlap) {
            परिभाषते.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, परिभाषते);
        स्वचालित pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_आकार     = context.max_wg_आकार;
        pipeline_decisions->inplace     = कुञ्चिका.inplace;
        pipeline_decisions->overlap     = कुञ्चिका.overlap;
        pipeline_decisions->src_overlap = कुञ्चिका.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[कुञ्चिका]     = pipeline;
        प्रत्यागम rms_norm_mul_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_द्विमान_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_द्विमान_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.type                            = context.dst->type;
        कुञ्चिका.op                              = context.dst->op;
        कुञ्चिका.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        कुञ्चिका.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        कुञ्चिका.src_overlap                     = src_overlap;

        स्वचालित it = द्विमान_pipelines.find(कुञ्चिका);
        यदि (it != द्विमान_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              op_name = ggml_op_name((ggml_op) कुञ्चिका.op);
        std::string              variant = op_name;

        परिभाषते.push_back(std::string("OP_") + op_name);

        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for binary shader");
        }

        यदि (कुञ्चिका.inplace) {
            परिभाषते.push_back("INPLACE");
            variant += "_inplace";
        } अन्यथा यदि (कुञ्चिका.overlap) {
            परिभाषते.push_back("OVERLAP");
            variant += "_overlap";
        } अन्यथा यदि (कुञ्चिका.src_overlap) {
            परिभाषते.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed                  = preprocessor.preprocess(wgsl_द्विमान, परिभाषते);
        स्वचालित pipeline_decisions         = std::make_shared<ggml_webgpu_द्विमान_shader_decisions>();
        pipeline_decisions->wg_आकार     = context.max_wg_आकार;
        pipeline_decisions->inplace     = कुञ्चिका.inplace;
        pipeline_decisions->overlap     = कुञ्चिका.overlap;
        pipeline_decisions->src_overlap = कुञ्चिका.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        द्विमान_pipelines[कुञ्चिका]    = pipeline;
        प्रत्यागम द्विमान_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_सङ्कलय_id_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_सङ्कलय_id_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वचालित it = सङ्कलय_id_pipelines.find(कुञ्चिका);
        यदि (it != सङ्कलय_id_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant    = "add_id";
        ध्रुव वर्ण *             shader_src = wgsl_सङ्कलय_id;

        यदि (कुञ्चिका.inplace) {
            परिभाषते.push_back("INPLACE");
            variant += "_inplace";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed              = preprocessor.preprocess(shader_src, परिभाषते);
        स्वचालित pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_आकार = context.max_wg_आकार;
        pipeline_decisions->inplace = कुञ्चिका.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        सङ्कलय_id_pipelines[कुञ्चिका]    = pipeline;
        प्रत्यागम pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_concat_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.type                            = context.dst->type;
        कुञ्चिका.src_overlap                     = src_overlap;

        स्वचालित it = concat_pipelines.find(कुञ्चिका);
        यदि (it != concat_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "concat";

        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                परिभाषते.push_back("TYPE_I32");
                variant += "_i32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for concat shader");
        }

        यदि (कुञ्चिका.src_overlap) {
            परिभाषते.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_concat, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_द्विमान_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->src_overlap   = कुञ्चिका.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[कुञ्चिका]    = pipeline;
        प्रत्यागम concat_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_repeat_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.type                            = context.dst->type;

        स्वचालित it = repeat_pipelines.find(कुञ्चिका);
        यदि (it != repeat_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "repeat";

        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                परिभाषते.push_back("TYPE_I32");
                variant += "_i32";
                भङ्ग;
            पक्ष GGML_TYPE_I16:
                परिभाषते.push_back("TYPE_I16");
                variant += "_i16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_repeat, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[कुञ्चिका]    = pipeline;
        प्रत्यागम repeat_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_flash_attn_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक kv_overlap) {
        ध्रुव तार्किक can_use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_मार्गः(
            context.supports_subgroup_matrix, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_matrix                    = can_use_subgroup_matrix;
        decisions.q_tile = decisions.use_sg_matrix ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.common                              = ggml_webgpu_flash_attn_make_common_pipeline_कुञ्चिका(
            context, decisions.use_sg_matrix ? context.sg_mat_k : 1u, kv_overlap);
        कुञ्चिका.common.k_प्रत्यक्षम् &= decisions.use_sg_matrix && कुञ्चिका.common.k_type == GGML_TYPE_F16;
        कुञ्चिका.common.v_प्रत्यक्षम् &= decisions.use_sg_matrix && कुञ्चिका.common.v_type == GGML_TYPE_F16;
        कुञ्चिका.use_sg_matrix = decisions.use_sg_matrix;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_सीमा_अष्टकानि, decisions.q_tile, decisions.use_sg_matrix ? context.sg_mat_n : 1u,
            कुञ्चिका.common.शीर्ष_dim_qk, कुञ्चिका.common.शीर्ष_dim_v, कुञ्चिका.common.has_आवरणम्,
            कुञ्चिका.common.k_प्रत्यक्षम् || कुञ्चिका.common.v_प्रत्यक्षम्);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        यदि (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_matrix);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_matrix = असत्य;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            कुञ्चिका.common.k_प्रत्यक्षम्     = असत्य;
            कुञ्चिका.common.v_प्रत्यक्षम्     = असत्य;
            कुञ्चिका.use_sg_matrix       = असत्य;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_सीमा_अष्टकानि, decisions.q_tile, 1u, कुञ्चिका.common.शीर्ष_dim_qk, कुञ्चिका.common.शीर्ष_dim_v,
                कुञ्चिका.common.has_आवरणम्, कुञ्चिका.common.k_प्रत्यक्षम् || कुञ्चिका.common.v_प्रत्यक्षम्);
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

        यदि (कुञ्चिका.common.k_प्रत्यक्षम् || कुञ्चिका.common.v_प्रत्यक्षम्) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            यावत् (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_matrix ? context.sg_mat_n : context.min_subgroup_आकार;
            }
        }

        स्वचालित it = flash_attn_pipelines.find(कुञ्चिका);
        यदि (it != flash_attn_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::string              variant = decisions.use_sg_matrix ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> परिभाषते = ggml_webgpu_flash_attn_common_परिभाषते(कुञ्चिका.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_आकार);
        ध्रुव वर्ण *             shader_src = शून्यसूचकः;
        यदि (!कुञ्चिका.use_sg_matrix) {
            shader_src = wgsl_flash_attn_tile;
            परिभाषते.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_आकार) + "u");
            परिभाषते.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_आकार) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_आकार) + "_" +
                       std::to_string(context.max_subgroup_आकार);
        } अन्यथा {
            shader_src = wgsl_flash_attn;
            परिभाषते.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            परिभाषते.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            परिभाषते.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        स्वचालित            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, परिभाषते), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम flash_attn_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.common = ggml_webgpu_flash_attn_make_common_pipeline_कुञ्चिका(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_विस्तारः,
                                                                     kv_overlap);

        स्वचालित it = flash_attn_vec_pipelines.find(कुञ्चिका);
        यदि (it != flash_attn_vec_pipelines.end()) {
            प्रत्यागम it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_सीमा_अष्टकानि, कुञ्चिका.common.शीर्ष_dim_qk,
                                                                   कुञ्चिका.common.शीर्ष_dim_v, कुञ्चिका.common.has_आवरणम्,
                                                                   कुञ्चिका.common.k_प्रत्यक्षम् || कुञ्चिका.common.v_प्रत्यक्षम्);
        decisions.wg_आकार = context.max_subgroup_आकार;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> परिभाषते =
            ggml_webgpu_flash_attn_common_परिभाषते(कुञ्चिका.common, variant, 1u, decisions.kv_tile, decisions.wg_आकार);
        यदि (कुञ्चिका.common.has_आवरणम्) {
            परिभाषते.push_back("BLK");
            variant.resize(variant.size() - (परिमाण("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_विभज = context.min_subgroup_आकार;
        ध्रुव uint32_t D       = कुञ्चिका.common.शीर्ष_dim_qk | कुञ्चिका.common.शीर्ष_dim_v;
        ध्रुव uint32_t D_lsb   = D & (~(D - 1u));
        d_विभज                = std::min(std::min(context.min_subgroup_आकार, 4u), std::max(D_lsb / 4u, 1u));

        परिभाषते.push_back(std::string("D_SPLIT=") + std::to_string(d_विभज));
        variant += "_dsplit" + std::to_string(d_विभज);

        स्वचालित            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_विभज, परिभाषते), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम flash_attn_vec_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.kv_tile                                 = kv_tile;
        स्वचालित it                                     = flash_attn_blk_pipelines.find(कुञ्चिका);
        यदि (it != flash_attn_blk_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "flash_attn_vec_blk";

        परिभाषते.push_back(std::string("KV_TILE=") + std::to_string(कुञ्चिका.kv_tile));
        variant += std::string("_kvt") + std::to_string(कुञ्चिका.kv_tile);

        uint32_t wg_आकार = 1;
        यावत् ((wg_आकार << 1) <= context.max_wg_आकार) {
            wg_आकार <<= 1;
        }
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(wg_आकार));
        variant += std::string("_wg") + std::to_string(wg_आकार);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, परिभाषते), variant);
        flash_attn_blk_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम flash_attn_blk_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.शीर्ष_dim_v                                     = (uint32_t) context.src2->ne[0];
        कुञ्चिका.dst_type                                       = context.dst->type;
        कुञ्चिका.wg_आकार                                        = context.max_wg_आकार;
        स्वचालित it                                            = flash_attn_vec_reduce_pipelines.find(कुञ्चिका);
        यदि (it != flash_attn_vec_reduce_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "flash_attn_vec_reduce";

        विकल्प (कुञ्चिका.dst_type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("DST_F32");
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("DST_F16");
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(कुञ्चिका.dst_type);

        परिभाषते.push_back(std::string("HEAD_DIM_V=") + std::to_string(कुञ्चिका.शीर्ष_dim_v));
        variant += std::string("_hsv") + std::to_string(कुञ्चिका.शीर्ष_dim_v);

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));
        variant += std::string("_wg") + std::to_string(context.max_wg_आकार);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, परिभाषते), variant);
        flash_attn_vec_reduce_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम flash_attn_vec_reduce_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_cpy_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.src_type                     = context.src0->type;
        कुञ्चिका.dst_type                     = context.dst->type;

        स्वचालित it = cpy_pipelines.find(कुञ्चिका);
        यदि (it != cpy_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "cpy";

        विकल्प (कुञ्चिका.src_type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("SRC_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("SRC_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                परिभाषते.push_back("SRC_I32");
                variant += "_i32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        विकल्प (कुञ्चिका.dst_type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("DST_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("DST_F16");
                variant += "_f16";
                भङ्ग;
            पक्ष GGML_TYPE_I32:
                परिभाषते.push_back("DST_I32");
                variant += "_i32";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_cpy, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[कुञ्चिका]       = pipeline;
        प्रत्यागम cpy_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_glu_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक src_overlap) {
        ggml_webgpu_glu_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.glu_op                       = ggml_get_glu_op(context.dst);
        कुञ्चिका.type                         = context.dst->type;
        कुञ्चिका.विभज                        = (context.src1 != शून्यसूचकः);
        कुञ्चिका.src_overlap                  = src_overlap;

        स्वचालित it = glu_pipelines.find(कुञ्चिका);
        यदि (it != glu_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "glu";

        विकल्प (कुञ्चिका.glu_op) {
            पक्ष GGML_GLU_OP_REGLU:
                परिभाषते.push_back("OP_REGLU");
                variant += "_reglu";
                भङ्ग;
            पक्ष GGML_GLU_OP_GEGLU:
                परिभाषते.push_back("OP_GEGLU");
                variant += "_geglu";
                भङ्ग;
            पक्ष GGML_GLU_OP_SWIGLU:
                परिभाषते.push_back("OP_SWIGLU");
                variant += "_swiglu";
                भङ्ग;
            पक्ष GGML_GLU_OP_SWIGLU_OAI:
                परिभाषते.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                भङ्ग;
            पक्ष GGML_GLU_OP_GEGLU_ERF:
                परिभाषते.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                भङ्ग;
            पक्ष GGML_GLU_OP_GEGLU_शीघ्र:
                परिभाषते.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                भङ्ग;
            पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
                परिभाषते.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported GLU op");
        }
        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        यदि (कुञ्चिका.src_overlap) {
            परिभाषते.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } अन्यथा यदि (कुञ्चिका.विभज) {
            variant += "_split";
        } अन्यथा {
            परिभाषते.push_back("NO_SPLIT");
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_glu, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->src_overlap   = कुञ्चिका.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[कुञ्चिका]       = pipeline;
        प्रत्यागम glu_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_rope_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.type                          = context.dst->type;
        कुञ्चिका.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        कुञ्चिका.has_ff                        = (context.src2 != शून्यसूचकः);

        स्वचालित it = rope_pipelines.find(कुञ्चिका);
        यदि (it != rope_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "rope";

        विकल्प (कुञ्चिका.type) {
            पक्ष GGML_TYPE_F32:
                परिभाषते.push_back("TYPE_F32");
                variant += "_f32";
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                परिभाषते.push_back("TYPE_F16");
                variant += "_f16";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        यदि (कुञ्चिका.inplace) {
            परिभाषते.push_back("INPLACE");
            variant += "_inplace";
        }

        यदि (कुञ्चिका.has_ff) {
            परिभाषते.push_back("FF_FUNC");
            variant += "_ff";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_rope, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->inplace       = कुञ्चिका.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[कुञ्चिका]      = pipeline;
        प्रत्यागम rope_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_soft_max_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.आवरणम्_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        कुञ्चिका.has_आवरणम्                          = (context.src1 != शून्यसूचकः);
        कुञ्चिका.has_sink                          = (context.src2 != शून्यसूचकः);
        कुञ्चिका.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        स्वचालित it = soft_max_pipelines.find(कुञ्चिका);
        यदि (it != soft_max_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "soft_max";

        यदि (कुञ्चिका.has_आवरणम्) {
            परिभाषते.push_back("HAS_MASK");
            विकल्प (कुञ्चिका.आवरणम्_type) {
                पक्ष GGML_TYPE_F32:
                    परिभाषते.push_back("MASK_F32");
                    variant += "_mask_f32";
                    भङ्ग;
                पक्ष GGML_TYPE_F16:
                    परिभाषते.push_back("MASK_F16");
                    variant += "_mask_f16";
                    भङ्ग;
                स्वभाव:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        यदि (कुञ्चिका.has_sink) {
            परिभाषते.push_back("HAS_SINK");
            variant += "_sink";
        }

        यदि (कुञ्चिका.inplace) {
            परिभाषते.push_back("INPLACE");
            variant += "_inplace";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_soft_max, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        decisions->inplace       = कुञ्चिका.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[कुञ्चिका]  = pipeline;
        प्रत्यागम soft_max_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_conv2d_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.weight_type                     = context.src0->type;
        कुञ्चिका.निवेशः_type                      = context.src1->type;
        कुञ्चिका.output_type                     = context.dst->type;

        स्वचालित it = conv2d_pipelines.find(कुञ्चिका);
        यदि (it != conv2d_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "conv_2d";

        स्वचालित प्रक्षिप_type_परिभाषते = [&](ध्रुव वर्ण * prefix, ggml_type type) {
            std::string s_उपसर्गः = prefix;
            यदि (type == GGML_TYPE_F32) {
                परिभाषते.push_back(s_उपसर्गः + "=f32");
            } अन्यथा यदि (type == GGML_TYPE_F16) {
                परिभाषते.push_back(s_उपसर्गः + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        प्रक्षिप_type_परिभाषते("WEIGHT_TYPE", कुञ्चिका.weight_type);
        प्रक्षिप_type_परिभाषते("INPUT_TYPE", कुञ्चिका.निवेशः_type);
        प्रक्षिप_type_परिभाषते("OUTPUT_TYPE", कुञ्चिका.output_type);

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_conv2d, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[कुञ्चिका]    = pipeline;
        प्रत्यागम conv2d_pipelines[कुञ्चिका];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context, तार्किक whcn) {
        ggml_webgpu_conv2d_dw_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.weight_type                        = context.src0->type;
        कुञ्चिका.निवेशः_type                         = context.src1->type;
        कुञ्चिका.output_type                        = context.dst->type;
        कुञ्चिका.whcn                               = whcn;

        स्वचालित it = conv2d_dw_pipelines.find(कुञ्चिका);
        यदि (it != conv2d_dw_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        स्वचालित प्रक्षिप_type_परिभाषते = [&](ध्रुव वर्ण * prefix, ggml_type type) {
            std::string s_उपसर्गः = prefix;
            यदि (type == GGML_TYPE_F32) {
                परिभाषते.push_back(s_उपसर्गः + "=f32");
            } अन्यथा यदि (type == GGML_TYPE_F16) {
                परिभाषते.push_back(s_उपसर्गः + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        प्रक्षिप_type_परिभाषते("WEIGHT_TYPE", कुञ्चिका.weight_type);
        प्रक्षिप_type_परिभाषते("INPUT_TYPE", कुञ्चिका.निवेशः_type);
        प्रक्षिप_type_परिभाषते("OUTPUT_TYPE", कुञ्चिका.output_type);

        यदि (whcn) {
            परिभाषते.push_back("WHCN");
        }
        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_conv2d_dw, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[कुञ्चिका] = pipeline;
        प्रत्यागम conv2d_dw_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_im2col_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.निवेशः_type                      = context.src1->type;
        कुञ्चिका.output_type                     = context.dst->type;

        स्वचालित it = im2col_pipelines.find(कुञ्चिका);
        यदि (it != im2col_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "im2col";

        स्वचालित प्रक्षिप_type_परिभाषते = [&](ध्रुव वर्ण * prefix, ggml_type type) {
            std::string s_उपसर्गः = prefix;
            यदि (type == GGML_TYPE_F32) {
                परिभाषते.push_back(s_उपसर्गः + "=f32");
            } अन्यथा यदि (type == GGML_TYPE_F16) {
                परिभाषते.push_back(s_उपसर्गः + "=f16");
            } अन्यथा {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        प्रक्षिप_type_परिभाषते("INPUT_TYPE", कुञ्चिका.निवेशः_type);
        प्रक्षिप_type_परिभाषते("OUTPUT_TYPE", कुञ्चिका.output_type);

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_im2col, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[कुञ्चिका]    = pipeline;
        प्रत्यागम im2col_pipelines[कुञ्चिका];
    }

    webgpu_pipeline get_upscale_pipeline(ध्रुव ggml_webgpu_shader_lib_context & context) {
        ध्रुव uint32_t बहुलकम्_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ध्रुव uint32_t आधारः_बहुलकम्  = बहुलकम्_flags & 0xFFu;
        ध्रुव तार्किक     antialias  = (बहुलकम्_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.निवेशः_type                       = context.src0->type;
        कुञ्चिका.output_type                      = context.dst->type;
        कुञ्चिका.आधारः_बहुलकम्                        = आधारः_बहुलकम्;
        कुञ्चिका.antialias                        = antialias;

        स्वचालित it = upscale_pipelines.find(कुञ्चिका);
        यदि (it != upscale_pipelines.end()) {
            प्रत्यागम it->second;
        }

        std::vector<std::string> परिभाषते;
        std::string              variant = "upscale";

        यदि (कुञ्चिका.निवेशः_type == GGML_TYPE_F16) {
            परिभाषते.push_back("SRC_F16");
            variant += "_src_f16";
        } अन्यथा {
            variant += "_src_f32";
        }

        यदि (कुञ्चिका.output_type == GGML_TYPE_F16) {
            परिभाषते.push_back("DST_F16");
            variant += "_dst_f16";
        } अन्यथा {
            variant += "_dst_f32";
        }

        विकल्प (आधारः_बहुलकम्) {
            पक्ष GGML_SCALE_बहुलकम्_निकटतमः:
                परिभाषते.push_back("NEAREST");
                variant += "_nearest";
                भङ्ग;
            पक्ष GGML_SCALE_बहुलकम्_BILINEAR:
                परिभाषते.push_back("BILINEAR");
                variant += "_bilinear";
                भङ्ग;
            पक्ष GGML_SCALE_बहुलकम्_BICUBIC:
                परिभाषते.push_back("BICUBIC");
                variant += "_bicubic";
                भङ्ग;
            स्वभाव:
                GGML_ABORT("Unsupported upscale mode");
        }

        यदि (antialias) {
            परिभाषते.push_back("ANTIALIAS");
            variant += "_aa";
        }

        परिभाषते.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_आकार));

        स्वचालित processed           = preprocessor.preprocess(wgsl_upscale, परिभाषते);
        स्वचालित decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_आकार       = context.max_wg_आकार;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[कुञ्चिका]   = pipeline;
        प्रत्यागम upscale_pipelines[कुञ्चिका];
    }

  गुप्त:
    स्थिर webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_संहिता,
                                                       std::string    label) {
        wgpu::ShaderSourceWGSL shader_source;
        shader_source.code = shader_संहिता.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.अग्र_अन्तर्_Chain = &shader_source;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.प्रविष्टिः_Point = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = शून्यसूचकः;  // nullptr means auto layout
        प्रत्यागम { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#यदि_अन्तः  // GGML_WEBGPU_SHADER_LIB_HPP
