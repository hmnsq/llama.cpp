#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_WEBGPU_SHADER_LIB_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_SHADER_LIB_HPP

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

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_F16_ಗಾತ್ರ_ಬೈಟುಗಳು                   2
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_F32_ಗಾತ್ರ_ಬೈಟುಗಳು                   4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_I32_ಗಾತ್ರ_ಬೈಟುಗಳು                   4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_ಗಾತ್ರ     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_KV_SEQ_PAD                       256u

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_WEBGPU_ARGSORT_ವಿಲೀನ_MAX_WG_ಗಾತ್ರ 512u

// Matrix multiplication parameters

// Register tiling parameters
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_TILE_M           4
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_TILE_N           4
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_M        8
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_N        8
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_SUBGROUP_ಮಾತೃಕೆ_M     4
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_SUBGROUP_ಮಾತೃಕೆ_N     2
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_VEC_WG_ಗಾತ್ರ 256

#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_ಪ್ರತಿ_WG    4
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_ಪ್ರತಿ_WG 4
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_ಪ್ರತಿ_WG      4

// default size for reg-tile matrix multiplication
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_MUL_MAT_WG_ಗಾತ್ರ 256

// Same hash combine function as in boost
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_webgpu_hash_combine(size_t & ಬೀಜ, ಸ್ಥಿರಾಂಕ T & value) {
    ಬೀಜ ^= std::hash<T>{}(value) + 0x9e3779b9 + (ಬೀಜ << 6) + (ಬೀಜ >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
ಸಾಲೊಳಗೆ uintptr_t ggml_webgpu_tensor_addr(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ_tensor = tensor->view_src ? tensor->view_src : tensor;
    ಹಿಂತಿರುಗಿ (uintptr_t) ತಳ_tensor->ದತ್ತಾಂಶ + tensor->view_offs;
}

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_webgpu_tensor_equal(ಸ್ಥಿರಾಂಕ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ggml_tensor * b) {
    ಹಿಂತಿರುಗಿ a->ಬಫರ್ == b->ಬಫರ್ && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

ರಚನೆ ggml_webgpu_shader_lib_context {
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * src2;
    ggml_tensor * src3;
    ggml_tensor * src4;
    ggml_tensor * src5;
    ggml_tensor * dst;
    ggml_tensor * dst_fuse;

    uint32_t    max_wg_ಗಾತ್ರ;
    size_t      wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು       = 0;
    ತಾರ್ಕಿಕ        supports_subgroups       = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ        supports_subgroup_ಮಾತೃಕೆ = ಸುಳ್ಳು;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_ಗಾತ್ರ        = 0;
    uint32_t    max_subgroup_ಗಾತ್ರ        = 0;
    ತಾರ್ಕಿಕ        supports_dot_ಗುಣಲಬ್ಧ     = ಸುಳ್ಳು;
    std::string vendor;
};

ರಚನೆ webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<ಶೂನ್ಯ> context = ಶೂನ್ಯಸೂಚಕ;
};

ರಚನೆ ggml_webgpu_generic_shader_decisions {
    uint32_t wg_ಗಾತ್ರ = 0;
    ತಾರ್ಕಿಕ     inplace = ಸುಳ್ಳು;
};

ರಚನೆ ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು_shader_decisions {
    uint32_t wg_ಗಾತ್ರ    = 0;
    ತಾರ್ಕಿಕ     vectorized = ಸುಳ್ಳು;
};

ರಚನೆ ggml_webgpu_ದ್ವಿಮಾನ_shader_decisions {
    uint32_t wg_ಗಾತ್ರ     = 0;
    ತಾರ್ಕಿಕ     inplace     = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ     overlap     = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ     src_overlap = ಸುಳ್ಳು;
};

ರಚನೆ ggml_webgpu_glu_shader_decisions {
    uint32_t wg_ಗಾತ್ರ     = 0;
    ತಾರ್ಕಿಕ     src_overlap = ಸುಳ್ಳು;
};

ರಚನೆ ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<ಶೂನ್ಯ> decisions;
};

ರಚನೆ ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t ಖಂಡ_ಗಾತ್ರ_3;
    uint32_t ಟೋಕನುಗಳು_ಪ್ರತಿ_wg;
};

ರಚನೆ ggml_webgpu_ssm_ಪರಿಶೀಲನೆ_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ     type;
    ಪೂರ್ಣಾಂಕ     d_state;
    uint8_t xbc_overlap;
    ತಾರ್ಕಿಕ    a_overlap;
    ತಾರ್ಕಿಕ    ids_overlap;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_ssm_ಪರಿಶೀಲನೆ_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

ರಚನೆ ggml_webgpu_ssm_ಪರಿಶೀಲನೆ_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_ssm_ಪರಿಶೀಲನೆ_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.d_state);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.xbc_overlap);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.a_overlap);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ids_overlap);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ರಚನೆ ggml_webgpu_ssm_ಪರಿಶೀಲನೆ_shader_decisions {
    uint32_t wg_ಗಾತ್ರ;
    uint32_t ಟೋಕನುಗಳು_ಪ್ರತಿ_tile;
    uint8_t  xbc_overlap = 0;
    ತಾರ್ಕಿಕ     a_overlap   = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ     ids_overlap = ಸುಳ್ಳು;
};

/** Argsort **/

ರಚನೆ ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_ಗಾತ್ರ;
    size_t   wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು;
    int32_t  ಕ್ರಮ;
};

/** Set Rows **/

ರಚನೆ ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ src0_type;
    ಪೂರ್ಣಾಂಕ dst_type;
    ಪೂರ್ಣಾಂಕ vec4;
    ಪೂರ್ಣಾಂಕ i64_idx;
    ಪೂರ್ಣಾಂಕ pair_ಖಂಡಗಳು;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_ಖಂಡಗಳು == other.pair_ಖಂಡಗಳು;
    }
};

ರಚನೆ ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src0_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.dst_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.vec4);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.i64_idx);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.pair_ಖಂಡಗಳು);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ರಚನೆ ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_shader_decisions {
    ತಾರ್ಕಿಕ     vec4;
    ತಾರ್ಕಿಕ     i64_idx;
    ತಾರ್ಕಿಕ     pair_ಖಂಡಗಳು;
    uint32_t wg_ಗಾತ್ರ;
};

/** Set **/

ರಚನೆ ggml_webgpu_ಗಣ_pipeline_ಕೀಲಿ {
    ggml_type type;
    ತಾರ್ಕಿಕ      inplace;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_ಗಣ_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ type == other.type && inplace == other.inplace;
    }
};

ರಚನೆ ggml_webgpu_ಗಣ_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_ಗಣ_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.inplace);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Get Rows **/

ರಚನೆ ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ {
    ggml_type src_type;
    ಪೂರ್ಣಾಂಕ       vectorized;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ src_type == other.src_type && vectorized == other.vectorized;
    }
};

ರಚನೆ ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.vectorized);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Row Norm **/

ರಚನೆ ggml_webgpu_ಅಡ್ಡಸಾಲು_norm_pipeline_ಕೀಲಿ {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    ತಾರ್ಕಿಕ      inplace;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_ಅಡ್ಡಸಾಲು_norm_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

ರಚನೆ ggml_webgpu_ಅಡ್ಡಸಾಲು_norm_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_ಅಡ್ಡಸಾಲು_norm_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.op);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.dst_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.inplace);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** RMS_NORM + MUL **/

ರಚನೆ ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ {
    ತಾರ್ಕಿಕ inplace;      // rn_src == dst
    ತಾರ್ಕಿಕ overlap;      // mul_src == dst
    ತಾರ್ಕಿಕ src_overlap;  // rn_src binding overlaps mul_src binding

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

ರಚನೆ ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.inplace);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.overlap);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src_overlap);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ರಚನೆ ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_ಗಾತ್ರ     = 0;
    ತಾರ್ಕಿಕ     inplace     = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ     overlap     = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ     src_overlap = ಸುಳ್ಳು;
};

/** Pad **/
ರಚನೆ ggml_webgpu_pad_pipeline_ಕೀಲಿ {
    ತಾರ್ಕಿಕ circular;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_pad_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ circular == other.circular; }
};

ರಚನೆ ggml_webgpu_pad_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_pad_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.circular);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Solve Tri **/
ರಚನೆ ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ type;
    ಪೂರ್ಣಾಂಕ n;
    ಪೂರ್ಣಾಂಕ k;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ type == other.type && n == other.n && k == other.k;
    }
};

ರಚನೆ ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.n);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.k);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** SSM Conv **/
ರಚನೆ ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ type;
    ಪೂರ್ಣಾಂಕ vectorized;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
ರಚನೆ ggml_webgpu_conv2d_pipeline_ಕೀಲಿ {
    ggml_type ತೂಕ_type;
    ggml_type ಒಳಹರಿವು_type;
    ggml_type output_type;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_conv2d_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ತೂಕ_type == other.ತೂಕ_type && ಒಳಹರಿವು_type == other.ಒಳಹರಿವು_type && output_type == other.output_type;
    }
};

ರಚನೆ ggml_webgpu_conv2d_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_conv2d_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ತೂಕ_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ಒಳಹರಿವು_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.output_type);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
ರಚನೆ ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ {
    ggml_type ತೂಕ_type;
    ggml_type ಒಳಹರಿವು_type;
    ggml_type output_type;
    ತಾರ್ಕಿಕ      whcn;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ತೂಕ_type == other.ತೂಕ_type && ಒಳಹರಿವು_type == other.ಒಳಹರಿವು_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

ರಚನೆ ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ತೂಕ_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ಒಳಹರಿವು_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.output_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.whcn);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Im2Col **/
ರಚನೆ ggml_webgpu_im2col_pipeline_ಕೀಲಿ {
    ggml_type ಒಳಹರಿವು_type;
    ggml_type output_type;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_im2col_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_type == other.ಒಳಹರಿವು_type && output_type == other.output_type;
    }
};

ರಚನೆ ggml_webgpu_im2col_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_im2col_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ಒಳಹರಿವು_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.output_type);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Gated Delta Net **/
ರಚನೆ ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ  type;
    ಪೂರ್ಣಾಂಕ  s_v;
    ಪೂರ್ಣಾಂಕ  kda;
    ತಾರ್ಕಿಕ fused_cache;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

ರಚನೆ ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.s_v);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.kda);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ರಚನೆ ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.vectorized);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Scale **/

ರಚನೆ ggml_webgpu_scale_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ inplace;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_scale_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ inplace == other.inplace; }
};

ರಚನೆ ggml_webgpu_scale_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_scale_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.inplace);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Upscale **/

ರಚನೆ ggml_webgpu_upscale_pipeline_ಕೀಲಿ {
    ggml_type ಒಳಹರಿವು_type;
    ggml_type output_type;
    uint32_t  ತಳ_ಬಹುಲಕ;
    ತಾರ್ಕಿಕ      antialias;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_upscale_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_type == other.ಒಳಹರಿವು_type && output_type == other.output_type && ತಳ_ಬಹುಲಕ == other.ತಳ_ಬಹುಲಕ &&
               antialias == other.antialias;
    }
};

ರಚನೆ ggml_webgpu_upscale_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_upscale_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ಒಳಹರಿವು_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.output_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ತಳ_ಬಹುಲಕ);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.antialias);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Concat **/

ರಚನೆ ggml_webgpu_concat_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ  type;
    ತಾರ್ಕಿಕ src_overlap;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_concat_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ type == other.type && src_overlap == other.src_overlap;
    }
};

ರಚನೆ ggml_webgpu_concat_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_concat_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src_overlap);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Repeat **/

ರಚನೆ ggml_webgpu_repeat_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ type;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_repeat_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ type == other.type; }
};

ರಚನೆ ggml_webgpu_repeat_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_repeat_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Binary **/

ರಚನೆ ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ  type;
    ಪೂರ್ಣಾಂಕ  op;
    ತಾರ್ಕಿಕ inplace;
    ತಾರ್ಕಿಕ overlap;
    ತಾರ್ಕಿಕ src_overlap;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

ರಚನೆ ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.op);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.inplace);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.overlap);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src_overlap);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/* Add_Id */

ರಚನೆ ggml_webgpu_ಕೂಡಿಸು_id_pipeline_ಕೀಲಿ {
    ತಾರ್ಕಿಕ inplace;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_ಕೂಡಿಸು_id_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ inplace == other.inplace; }
};

ರಚನೆ ggml_webgpu_ಕೂಡಿಸು_id_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_ಕೂಡಿಸು_id_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.inplace);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Unary **/

ರಚನೆ ggml_webgpu_unary_pipeline_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ           type;
    ಪೂರ್ಣಾಂಕ           op;
    ತಾರ್ಕಿಕ          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    ತಾರ್ಕಿಕ          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_unary_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

ರಚನೆ ggml_webgpu_unary_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_unary_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.op);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.is_unary);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.inplace);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ttype);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** FlashAttention */

ರಚನೆ ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  ತಲೆ_dim_qk;
    uint32_t  ತಲೆ_dim_v;
    ತಾರ್ಕಿಕ      k_ನೇರ;
    ತಾರ್ಕಿಕ      v_ನೇರ;
    ತಾರ್ಕಿಕ      kv_overlap;
    ತಾರ್ಕಿಕ      has_ಮುಖವಾಡ;
    ತಾರ್ಕಿಕ      has_sinks;
    ತಾರ್ಕಿಕ      ಬಳಸುತ್ತದೆ_logit_softcap;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && ತಲೆ_dim_qk == other.ತಲೆ_dim_qk && ತಲೆ_dim_v == other.ತಲೆ_dim_v &&
               k_ನೇರ == other.k_ನೇರ && v_ನೇರ == other.v_ನೇರ && kv_overlap == other.kv_overlap &&
               has_ಮುಖವಾಡ == other.has_ಮುಖವಾಡ && has_sinks == other.has_sinks &&
               ಬಳಸುತ್ತದೆ_logit_softcap == other.ಬಳಸುತ್ತದೆ_logit_softcap;
    }
};

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_webgpu_flash_attn_hash_common_pipeline_ಕೀಲಿ(size_t &                                           ಬೀಜ,
                                                            ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ & ಕೀಲಿ) {
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.q_type);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.k_type);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.v_type);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.dst_type);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ತಲೆ_dim_qk);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ತಲೆ_dim_v);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.k_ನೇರ);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.v_ನೇರ);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.kv_overlap);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.has_ಮುಖವಾಡ);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.has_sinks);
    ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ಬಳಸುತ್ತದೆ_logit_softcap);
}

ರಚನೆ ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ {
    ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ common;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ common == other.common; }
};

ರಚನೆ ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ಕೀಲಿ(ಬೀಜ, ಕೀಲಿ.common);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ರಚನೆ ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ {
    ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ common;
    ತಾರ್ಕಿಕ                                       use_sg_ಮಾತೃಕೆ;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ common == other.common && use_sg_ಮಾತೃಕೆ == other.use_sg_ಮಾತೃಕೆ;
    }
};

ರಚನೆ ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ಕೀಲಿ(ಬೀಜ, ಕೀಲಿ.common);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.use_sg_ಮಾತೃಕೆ);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ರಚನೆ ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_ಗಾತ್ರ = 0;
};

ರಚನೆ ggml_webgpu_flash_attn_decisions {
    ತಾರ್ಕಿಕ     use_sg_ಮಾತೃಕೆ = ಸುಳ್ಳು;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_ಗಾತ್ರ       = 0;
};

ಸಾಲೊಳಗೆ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ಅಗಲ = 4u;
ಸಾಲೊಳಗೆ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

ಸಾಲೊಳಗೆ size_t ggml_webgpu_flash_attn_tensor_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    constexpr uintptr_t ptr_ತಳ_addr = 0x1000u;
    ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ          = tensor->view_src != ಶೂನ್ಯಸೂಚಕ ? tensor->view_src : tensor;
    ಹಿಂತಿರುಗಿ reinterpret_cast<uintptr_t>(ತಳ->ದತ್ತಾಂಶ) - ptr_ತಳ_addr + tensor->view_offs;
}

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_float_vec4_aligned(ಸ್ಥಿರಾಂಕ ggml_tensor * K, size_t storage_ಸ್ಥಾನಾಂತರ_alignment) {
    ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನಾಂತರ_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_ಸ್ಥಾನಾಂತರ(K) & (storage_ಸ್ಥಾನಾಂತರ_alignment - 1)) /
                    ggml_type_ಗಾತ್ರ(K->type));
    ಹಿಂತಿರುಗಿ ಸ್ಥಾನಾಂತರ_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ಅಗಲ == 0u;
}

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_float_vec4_aligned(ಸ್ಥಿರಾಂಕ ggml_tensor * K,
                                                      ಸ್ಥಿರಾಂಕ ggml_tensor * V,
                                                      size_t              storage_ಸ್ಥಾನಾಂತರ_alignment) {
    ಹಿಂತಿರುಗಿ ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_ಸ್ಥಾನಾಂತರ_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_ಸ್ಥಾನಾಂತರ_alignment);
}

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_k_ನೇರ(ಸ್ಥಿರಾಂಕ ggml_tensor * Q, ಸ್ಥಿರಾಂಕ ggml_tensor * K, uint32_t kv_ನೇರ_align) {
    ಹಿಂತಿರುಗಿ (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_ನೇರ_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_v_ನೇರ(ಸ್ಥಿರಾಂಕ ggml_tensor * Q, ಸ್ಥಿರಾಂಕ ggml_tensor * V, uint32_t kv_ನೇರ_align) {
    ಹಿಂತಿರುಗಿ ggml_webgpu_flash_attn_k_ನೇರ(Q, V, kv_ನೇರ_align);
}

ಸಾಲೊಳಗೆ ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ ggml_webgpu_flash_attn_make_common_pipeline_ಕೀಲಿ(
    ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_ನೇರ_align,
    ತಾರ್ಕಿಕ                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
    ಕೀಲಿ.q_type                                     = context.src0->type;
    ಕೀಲಿ.k_type                                     = context.src1->type;
    ಕೀಲಿ.v_type                                     = context.src2->type;
    ಕೀಲಿ.dst_type                                   = context.dst->type;
    ಕೀಲಿ.ತಲೆ_dim_qk                                = (uint32_t) context.src0->ne[0];
    ಕೀಲಿ.ತಲೆ_dim_v                                 = (uint32_t) context.src2->ne[0];
    ಕೀಲಿ.k_ನೇರ           = ggml_webgpu_flash_attn_k_ನೇರ(context.src0, context.src1, kv_ನೇರ_align);
    ಕೀಲಿ.v_ನೇರ           = ggml_webgpu_flash_attn_v_ನೇರ(context.src0, context.src2, kv_ನೇರ_align);
    ಕೀಲಿ.kv_overlap         = kv_overlap;
    ಕೀಲಿ.has_ಮುಖವಾಡ           = context.src3 != ಶೂನ್ಯಸೂಚಕ;
    ಕೀಲಿ.has_sinks          = context.src4 != ಶೂನ್ಯಸೂಚಕ;
    ಕೀಲಿ.ಬಳಸುತ್ತದೆ_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    ಹಿಂತಿರುಗಿ ಕೀಲಿ;
}

ಸಾಲೊಳಗೆ std::vector<std::string> ggml_webgpu_flash_attn_common_ನಿರೂಪಿಸುತ್ತದೆ(
    ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_common_pipeline_ಕೀಲಿ & ಕೀಲಿ,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_ಗಾತ್ರ) {
    std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;

    ಆಯ್ಕೆ (ಕೀಲಿ.k_type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("K_F32");
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("K_F16");
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("K_Q4_0");
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("K_Q8_0");
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(ಕೀಲಿ.k_type);

    ಆಯ್ಕೆ (ಕೀಲಿ.v_type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("V_F32");
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("V_F16");
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("V_Q4_0");
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("V_Q8_0");
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(ಕೀಲಿ.v_type);

    ಆಯ್ಕೆ (ಕೀಲಿ.q_type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("Q_F32");
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("Q_F16");
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(ಕೀಲಿ.q_type);

    ಆಯ್ಕೆ (ಕೀಲಿ.dst_type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F32");
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F16");
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(ಕೀಲಿ.dst_type);

    ಆದರೆ (ಕೀಲಿ.has_ಮುಖವಾಡ) {
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("MASK");
        variant += "_mask";
    }
    ಆದರೆ (ಕೀಲಿ.has_sinks) {
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("SINKS");
        variant += "_sinks";
    }
    ಆದರೆ (ಕೀಲಿ.ಬಳಸುತ್ತದೆ_logit_softcap) {
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    ಆದರೆ (ಕೀಲಿ.k_ನೇರ) {
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    ಆದರೆ (ಕೀಲಿ.v_ನೇರ) {
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    ಆದರೆ (ಕೀಲಿ.kv_overlap) {
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("HEAD_DIM_QK=") + std::to_string(ಕೀಲಿ.ತಲೆ_dim_qk));
    variant += std::string("_hsqk") + std::to_string(ಕೀಲಿ.ತಲೆ_dim_qk);

    ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("HEAD_DIM_V=") + std::to_string(ಕೀಲಿ.ತಲೆ_dim_v));
    variant += std::string("_hsv") + std::to_string(ಕೀಲಿ.ತಲೆ_dim_v);

    ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ಗಾತ್ರ));

    ಆದರೆ (ggml_is_quantized(ಕೀಲಿ.k_type) || ggml_is_quantized(ಕೀಲಿ.v_type)) {
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("U32_DEQUANT_HELPERS");
        ಆದರೆ (ggml_is_quantized(ಕೀಲಿ.k_type)) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("LOADERS_QUANTIZED_K");
        }
        ಆದರೆ (ggml_is_quantized(ಕೀಲಿ.v_type)) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("LOADERS_QUANTIZED_V");
        }
    }

    ಹಿಂತಿರುಗಿ ನಿರೂಪಿಸುತ್ತದೆ;
}

ರಚನೆ ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ {
    uint32_t  ತಲೆ_dim_v;
    uint32_t  wg_ಗಾತ್ರ;
    ggml_type dst_type;
};

ರಚನೆ ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ತಲೆ_dim_v);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.wg_ಗಾತ್ರ);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.dst_type);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ & lhs,
                       ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ & rhs) {
    ಹಿಂತಿರುಗಿ lhs.ತಲೆ_dim_v == rhs.ತಲೆ_dim_v && lhs.wg_ಗಾತ್ರ == rhs.wg_ಗಾತ್ರ && lhs.dst_type == rhs.dst_type;
}

ರಚನೆ ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ {
    uint32_t kv_tile;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ kv_tile == other.kv_tile; }
};

ರಚನೆ ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.kv_tile);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
ಸಾಲೊಳಗೆ size_t ggml_webgpu_flash_attn_wg_mem_ಬೈಟುಗಳು(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t ತಲೆ_dim_qk,
                                                  uint32_t ತಲೆ_dim_v,
                                                  ತಾರ್ಕಿಕ     has_ಮುಖವಾಡ,
                                                  ತಾರ್ಕಿಕ     kv_ನೇರ) {
    ಸ್ಥಿರಾಂಕ uint32_t max_ತಲೆ_dim = std::max(ತಲೆ_dim_qk, ತಲೆ_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * ತಲೆ_dim_qk;        // q_shmem
    ಆದರೆ (!kv_ನೇರ) {
        f32_elems += kv_tile * max_ತಲೆ_dim;  // kv_shmem
    }
    f32_elems += q_tile * ತಲೆ_dim_v;         // o_shmem
    ಆದರೆ (has_ಮುಖವಾಡ) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    ಹಿಂತಿರುಗಿ f16_elems * GGML_WEBGPU_F16_ಗಾತ್ರ_ಬೈಟುಗಳು + f32_elems * GGML_WEBGPU_F32_ಗಾತ್ರ_ಬೈಟುಗಳು;
}

ಸಾಲೊಳಗೆ uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   ಎಲ್ಲೆ_ಬೈಟುಗಳು,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t ತಲೆ_dim_qk,
                                                   uint32_t ತಲೆ_dim_v,
                                                   ತಾರ್ಕಿಕ     has_ಮುಖವಾಡ,
                                                   ತಾರ್ಕಿಕ     kv_ನೇರ) {
    ಸ್ಥಿರಾಂಕ size_t ತಳ_q_ಬೈಟುಗಳು =
        ggml_webgpu_flash_attn_wg_mem_ಬೈಟುಗಳು(q_tile, 0, ತಲೆ_dim_qk, ತಲೆ_dim_v, has_ಮುಖವಾಡ, kv_ನೇರ);
    ಆದರೆ (ಎಲ್ಲೆ_ಬೈಟುಗಳು <= ತಳ_q_ಬೈಟುಗಳು) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಸ್ಥಿರಾಂಕ size_t one_kv_ಬೈಟುಗಳು =
        ggml_webgpu_flash_attn_wg_mem_ಬೈಟುಗಳು(q_tile, 1, ತಲೆ_dim_qk, ತಲೆ_dim_v, has_ಮುಖವಾಡ, kv_ನೇರ);
    ಸ್ಥಿರಾಂಕ size_t ಬೈಟುಗಳು_ಪ್ರತಿ_kv = one_kv_ಬೈಟುಗಳು - ತಳ_q_ಬೈಟುಗಳು;
    ಆದರೆ (ಬೈಟುಗಳು_ಪ್ರತಿ_kv == 0) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಸ್ಥಿರಾಂಕ size_t max_kv_tile = (ಎಲ್ಲೆ_ಬೈಟುಗಳು - ತಳ_q_ಬೈಟುಗಳು) / ಬೈಟುಗಳು_ಪ್ರತಿ_kv;
    ಹಿಂತಿರುಗಿ (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

ಸಾಲೊಳಗೆ uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು,
                                                       uint32_t ತಲೆ_dim_qk,
                                                       uint32_t ತಲೆ_dim_v,
                                                       ತಾರ್ಕಿಕ     has_ಮುಖವಾಡ,
                                                       ತಾರ್ಕಿಕ     kv_ನೇರ) {
    ಸ್ಥಿರಾಂಕ uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು, 1u, 1u, ತಲೆ_dim_qk, ತಲೆ_dim_v, has_ಮುಖವಾಡ, kv_ನೇರ);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    ಆದರೆ (kv_ನೇರ) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        ಆಗ (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    ಹಿಂತಿರುಗಿ kv_tile;
}

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_can_use_subgroup_ಮಾತೃಕೆ_ಮಾರ್ಗ(ತಾರ್ಕಿಕ                supports_subgroup_ಮಾತೃಕೆ,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ಸ್ಥಿರಾಂಕ ggml_tensor * Q,
                                                                ಸ್ಥಿರಾಂಕ ggml_tensor * V) {
    ಹಿಂತಿರುಗಿ supports_subgroup_ಮಾತೃಕೆ && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

ರಚನೆ ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ {
    ggml_type src0_type;
    ggml_type src1_type;
    ಪೂರ್ಣಾಂಕ       vectorized;
    uint32_t  ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    ತಾರ್ಕಿಕ      use_mmvq;
    ತಾರ್ಕಿಕ      src_overlap;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ == other.ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

ರಚನೆ ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src0_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src1_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.vectorized);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.use_mmvq);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src_overlap);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ರಚನೆ ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_ಗಾತ್ರ;
    uint32_t outputs_ಪ್ರತಿ_wg;
    uint32_t vec_ಗಾತ್ರ;
    ತಾರ್ಕಿಕ     src_overlap = ಸುಳ್ಳು;
};

ರಚನೆ ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ {
    ggml_type src0_type;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ src0_type == other.src0_type; }
};

ರಚನೆ ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src0_type);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ರಚನೆ ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ {
    ggml_type src0_type;
    ggml_type src1_type;
    ಪೂರ್ಣಾಂಕ       vectorized;
    ಪೂರ್ಣಾಂಕ       use_subgroup_ಮಾತೃಕೆ;
    ತಾರ್ಕಿಕ      src_overlap;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_ಮಾತೃಕೆ == other.use_subgroup_ಮಾತೃಕೆ && src_overlap == other.src_overlap;
    }
};

ರಚನೆ ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src0_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src1_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.vectorized);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.use_subgroup_ಮಾತೃಕೆ);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src_overlap);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ರಚನೆ ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_ಗಾತ್ರ_m;
    uint32_t wg_ಗಾತ್ರ_n;
    uint32_t wg_ಗಾತ್ರ;
    uint32_t outputs_ಪ್ರತಿ_wg;
    ಪೂರ್ಣಾಂಕ      use_subgroup_ಮಾತೃಕೆ;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_ಮಾತೃಕೆ_m;
    uint32_t subgroup_ಮಾತೃಕೆ_n;

    uint32_t mul_mat_wg_ಗಾತ್ರ;
    ತಾರ್ಕಿಕ     src_overlap = ಸುಳ್ಳು;
};

/** MUL_MAT_ID **/

ರಚನೆ ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ {
    uint32_t n_experts;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ n_experts == other.n_experts;
    }
};

ರಚನೆ ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.n_experts);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

ರಚನೆ ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    ಪೂರ್ಣಾಂಕ       vectorized;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ == other.ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ && vectorized == other.vectorized;
    }
};

ರಚನೆ ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src0_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src1_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.n_experts);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.vectorized);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Cpy **/

ರಚನೆ ggml_webgpu_cpy_pipeline_ಕೀಲಿ {
    ggml_type src_type;
    ggml_type dst_type;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_cpy_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ src_type == other.src_type && dst_type == other.dst_type;
    }
};

ರಚನೆ ggml_webgpu_cpy_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_cpy_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.dst_type);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Glu **/

ರಚನೆ ggml_webgpu_glu_pipeline_ಕೀಲಿ {
    ggml_glu_op glu_op;
    ggml_type   type;
    ತಾರ್ಕಿಕ        ವಿಭಜಿಸು;
    ತಾರ್ಕಿಕ        src_overlap;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_glu_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ glu_op == other.glu_op && type == other.type && ವಿಭಜಿಸು == other.ವಿಭಜಿಸು && src_overlap == other.src_overlap;
    }
};

ರಚನೆ ggml_webgpu_glu_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_glu_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.glu_op);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ವಿಭಜಿಸು);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.src_overlap);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** Rope **/

ರಚನೆ ggml_webgpu_rope_pipeline_ಕೀಲಿ {
    ggml_type type;
    ತಾರ್ಕಿಕ      inplace;
    ತಾರ್ಕಿಕ      has_ff;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_rope_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

ರಚನೆ ggml_webgpu_rope_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_rope_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.inplace);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.has_ff);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** SoftMax **/

ರಚನೆ ggml_webgpu_soft_max_pipeline_ಕೀಲಿ {
    ggml_type ಮುಖವಾಡ_type;
    ತಾರ್ಕಿಕ      has_ಮುಖವಾಡ;
    ತಾರ್ಕಿಕ      has_sink;
    ತಾರ್ಕಿಕ      inplace;

    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ggml_webgpu_soft_max_pipeline_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ಮುಖವಾಡ_type == other.ಮುಖವಾಡ_type && has_ಮುಖವಾಡ == other.has_ಮುಖವಾಡ && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

ರಚನೆ ggml_webgpu_soft_max_pipeline_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ggml_webgpu_soft_max_pipeline_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t ಬೀಜ = 0;
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.ಮುಖವಾಡ_type);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.has_ಮುಖವಾಡ);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.has_sink);
        ggml_webgpu_hash_combine(ಬೀಜ, ಕೀಲಿ.inplace);
        ಹಿಂತಿರುಗಿ ಬೀಜ;
    }
};

/** MMVQ **/

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_webgpu_can_use_mmvq(ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
                                     ಸ್ಥಿರಾಂಕ ggml_tensor * src1,
                                     ತಾರ್ಕಿಕ                supports_dot_ಗುಣಲಬ್ಧ,
                                     ಸ್ಥಿರಾಂಕ std::string & vendor) {
    ಆದರೆ (src1->ne[1] <= 4) {
        ತಾರ್ಕಿಕ supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        ಆದರೆ (supports_dp4a && supports_dot_ಗುಣಲಬ್ಧ) {
            ಆಯ್ಕೆ (src1->type) {
                ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಆಯ್ಕೆ (src0->type) {
                        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                            ಹಿಂತಿರುಗಿ src0->ne[0] % 4 == 0;
                        ಪೂರ್ವನಿಯೋಜಿತ:
                            ನಿಲ್ಲಿಸು;
                    }
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ನಿಲ್ಲಿಸು;
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ವರ್ಗ ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<ಪೂರ್ಣಾಂಕ, webgpu_pipeline> ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_pipelines;       // key is fixed, no variants yet
    std::unordered_map<ಪೂರ್ಣಾಂಕ, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<ಪೂರ್ಣಾಂಕ, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<ಪೂರ್ಣಾಂಕ, webgpu_pipeline> argsort_ವಿಲೀನ_pipelines;  // key is order
    std::unordered_map<ಪೂರ್ಣಾಂಕ, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_ಅಡ್ಡಸಾಲು_norm_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_ಅಡ್ಡಸಾಲು_norm_pipeline_ಕೀಲಿ_hash>
        ಅಡ್ಡಸಾಲು_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ_hash>
        get_ಅಡ್ಡಸಾಲುಗಳು_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_unary_pipeline_ಕೀಲಿ_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_scale_pipeline_ಕೀಲಿ_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_ಪರಿಶೀಲನೆ_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_ssm_ಪರಿಶೀಲನೆ_pipeline_ಕೀಲಿ_hash>
        ssm_ಪರಿಶೀಲನೆ_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_pad_pipeline_ಕೀಲಿ_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ_hash>
        ದ್ವಿಮಾನ_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_ಕೂಡಿಸು_id_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_ಕೂಡಿಸು_id_pipeline_ಕೀಲಿ_hash>
        ಕೂಡಿಸು_id_pipelines;           // inplace
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

    std::unordered_map<ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ_hash>
        ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipelines;
    std::unordered_map<ggml_webgpu_ಗಣ_pipeline_ಕೀಲಿ, webgpu_pipeline, ggml_webgpu_ಗಣ_pipeline_ಕೀಲಿ_hash> ಗಣ_pipelines;
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
    ggml_webgpu_shader_lib(wgpu::Device device) { ಇದು->device = device; }

    webgpu_pipeline get_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ಸ್ವಯಂಚಾಲಿತ it = ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_pipelines.find(1);
        ಆದರೆ (it != ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }
        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed        = preprocessor.preprocess(wgsl_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು, ನಿರೂಪಿಸುತ್ತದೆ);
        ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        ಹಿಂತಿರುಗಿ ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_pipelines[1];
    }

    webgpu_pipeline get_ಅಡ್ಡಸಾಲು_norm_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ಅಡ್ಡಸಾಲು_norm_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.op                                = context.dst->op;
        ಕೀಲಿ.src_type                          = context.src0->type;
        ಕೀಲಿ.dst_type                          = context.dst->type;
        ಕೀಲಿ.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        ಸ್ವಯಂಚಾಲಿತ it = ಅಡ್ಡಸಾಲು_norm_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != ಅಡ್ಡಸಾಲು_norm_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }
        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant;

        ಆಯ್ಕೆ (ಕೀಲಿ.op) {
            ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("RMS_NORM");
                variant = "rms_norm";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_NORM:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("NORM");
                variant = "norm";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_L2_NORM:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("L2_NORM");
                variant = "l2_norm";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        ಆದರೆ (ಕೀಲಿ.inplace) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("INPLACE");
            variant += "_inplace";
        }

        ಆದರೆ (ಕೀಲಿ.src_type == GGML_TYPE_F32) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_F32");
            variant += "_src_f32";
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.src_type == GGML_TYPE_F16) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_F16");
            variant += "_src_f16";
        }

        ಆದರೆ (ಕೀಲಿ.dst_type == GGML_TYPE_F32) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F32");
            variant += "_dst_f32";
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.dst_type == GGML_TYPE_F16) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_norm_wg_ಗಾತ್ರ = 128u;
        uint32_t       wg_ಗಾತ್ರ          = std::min(context.max_wg_ಗಾತ್ರ, ಅಡ್ಡಸಾಲು_norm_wg_ಗಾತ್ರ);
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ಗಾತ್ರ));
        ಸ್ವಯಂಚಾಲಿತ processed                  = preprocessor.preprocess(wgsl_ಅಡ್ಡಸಾಲು_norm, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ              = wg_ಗಾತ್ರ;
        decisions->inplace              = ಕೀಲಿ.inplace;
        ಅಡ್ಡಸಾಲು_norm_pipelines[ಕೀಲಿ]         = ggml_webgpu_create_pipeline(device, processed, variant);
        ಅಡ್ಡಸಾಲು_norm_pipelines[ಕೀಲಿ].context = decisions;
        ಹಿಂತಿರುಗಿ ಅಡ್ಡಸಾಲು_norm_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_argmax_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ತಾರ್ಕಿಕ vec4 = context.src0->ne[0] % 4 == 0;

        ಸ್ವಯಂಚಾಲಿತ it = argmax_pipelines.find(vec4);
        ಆದರೆ (it != argmax_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));
        ಆದರೆ (vec4) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("VEC4");
            variant += "_vec4";
        }

        ಸ್ವಯಂಚಾಲಿತ processed         = preprocessor.preprocess(wgsl_argmax, ನಿರೂಪಿಸುತ್ತದೆ);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        ಹಿಂತಿರುಗಿ argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ ಕೀಲಿ       = {};
        ಕೀಲಿ.src0_type                               = context.src0->type;
        ಕೀಲಿ.dst_type                                = context.dst->type;
        ಕೀಲಿ.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        ಕೀಲಿ.i64_idx     = context.src1->type == GGML_TYPE_I64;
        ಕೀಲಿ.pair_ಖಂಡಗಳು = quantized && ((context.src0->ne[0] / ggml_blck_ಗಾತ್ರ(context.dst->type)) % 2 == 0);

        ಸ್ವಯಂಚಾಲಿತ it = ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "set_rows";

        ಆಯ್ಕೆ (context.dst->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F32");
                variant += "_dstf32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F16");
                variant += "_dstf16";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        ಆದರೆ (context.src0->type == GGML_TYPE_F16) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        ಆದರೆ (ಕೀಲಿ.vec4) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("VEC4");
            variant += "_vec4";
        }
        ಆದರೆ (ಕೀಲಿ.i64_idx) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("I64_IDX");
            variant += "_i64idx";
        }
        ಆದರೆ (ಕೀಲಿ.pair_ಖಂಡಗಳು) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & shader_ಮೂಲ      = quantized ? wgsl_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_quant : wgsl_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು;
        ಸ್ವಯಂಚಾಲಿತ         processed          = preprocessor.preprocess(shader_ಮೂಲ, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ         decisions          = std::make_shared<ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_shader_decisions>();
        decisions->vec4                 = ಕೀಲಿ.vec4;
        decisions->i64_idx              = ಕೀಲಿ.i64_idx;
        decisions->pair_ಖಂಡಗಳು          = ಕೀಲಿ.pair_ಖಂಡಗಳು;
        decisions->wg_ಗಾತ್ರ              = context.max_wg_ಗಾತ್ರ;
        ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipelines[ಕೀಲಿ]         = ggml_webgpu_create_pipeline(device, processed, variant);
        ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipelines[ಕೀಲಿ].context = decisions;
        ಹಿಂತಿರುಗಿ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_ಗಣ_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ಗಣ_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                         = context.dst->type;
        ಕೀಲಿ.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        ಸ್ವಯಂಚಾಲಿತ it = ಗಣ_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != ಗಣ_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "set";

        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_I32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_I32");
                variant += "_i32";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for set shader");
        }

        ಆದರೆ (ಕೀಲಿ.inplace) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("INPLACE");
            variant += "_inplace";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_ಗಣ, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        decisions->inplace       = ಕೀಲಿ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ಗಣ_pipelines[ಕೀಲಿ]       = pipeline;
        ಹಿಂತಿರುಗಿ ಗಣ_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_cumsum_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ಸ್ವಯಂಚಾಲಿತ it = cumsum_pipelines.find(1);
        ಆದರೆ (it != cumsum_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed      = preprocessor.preprocess(wgsl_cumsum, ನಿರೂಪಿಸುತ್ತದೆ);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        ಹಿಂತಿರುಗಿ cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ತಾರ್ಕಿಕ          is_ತುದಿ_k = context.dst->op == GGML_OP_ತುದಿ_K;
        // ascending order is 0, descending order is 1
        ಸ್ಥಿರಾಂಕ int32_t ಕ್ರಮ =
            is_ತುದಿ_k ? (int32_t) GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        ಸ್ವಯಂಚಾಲಿತ it = argsort_pipelines.find(ಕ್ರಮ);
        ಆದರೆ (it != argsort_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "argsort";
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("ORDER=") + std::to_string(ಕ್ರಮ));
        variant += std::string("_order") + std::to_string(ಕ್ರಮ);
        uint32_t wg_ಗಾತ್ರ = 1;
        ಆಗ (wg_ಗಾತ್ರ * 2 <= context.max_wg_ಗಾತ್ರ &&
               wg_ಗಾತ್ರ * GGML_WEBGPU_I32_ಗಾತ್ರ_ಬೈಟುಗಳು <= context.wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು / 2) {
            wg_ಗಾತ್ರ *= 2;
        }
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ಗಾತ್ರ));
        ಸ್ವಯಂಚಾಲಿತ processed                   = preprocessor.preprocess(wgsl_argsort, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ               = wg_ಗಾತ್ರ;
        argsort_pipelines[ಕ್ರಮ]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[ಕ್ರಮ].context = decisions;
        ಹಿಂತಿರುಗಿ argsort_pipelines[ಕ್ರಮ];
    }

    webgpu_pipeline get_argsort_ವಿಲೀನ_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ತಾರ್ಕಿಕ          is_ತುದಿ_k = context.dst->op == GGML_OP_ತುದಿ_K;
        // ascending order is 0, descending order is 1
        ಸ್ಥಿರಾಂಕ int32_t ಕ್ರಮ =
            is_ತುದಿ_k ? (int32_t) GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        ಸ್ವಯಂಚಾಲಿತ it = argsort_ವಿಲೀನ_pipelines.find(ಕ್ರಮ);
        ಆದರೆ (it != argsort_ವಿಲೀನ_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "argsort_merge";
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("ORDER=") + std::to_string(ಕ್ರಮ));
        variant += std::string("_order") + std::to_string(ಕ್ರಮ);
        uint32_t wg_ಗಾತ್ರ = std::min(GGML_WEBGPU_ARGSORT_ವಿಲೀನ_MAX_WG_ಗಾತ್ರ, context.max_wg_ಗಾತ್ರ);
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed                 = preprocessor.preprocess(wgsl_argsort_ವಿಲೀನ, ನಿರೂಪಿಸುತ್ತದೆ);
        argsort_ವಿಲೀನ_pipelines[ಕ್ರಮ] = ggml_webgpu_create_pipeline(device, processed, variant);
        ಹಿಂತಿರುಗಿ argsort_ವಿಲೀನ_pipelines[ಕ್ರಮ];
    }

    webgpu_pipeline get_get_ಅಡ್ಡಸಾಲುಗಳು_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ vec4_aligned) {
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src_type                          = context.src0->type;
        ಕೀಲಿ.vectorized                        = (ಪೂರ್ಣಾಂಕ) vectorized;

        ಸ್ವಯಂಚಾಲಿತ it = get_ಅಡ್ಡಸಾಲುಗಳು_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != get_ಅಡ್ಡಸಾಲುಗಳು_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "get_rows";

        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_type_traits * type_traits = ggml_get_type_traits(ಕೀಲಿ.src_type);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *                    type_str    = type_traits->type_name;

        ಆಯ್ಕೆ (ಕೀಲಿ.src_type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("FLOAT_PARALLEL");
                ಆದರೆ (ಕೀಲಿ.vectorized) {
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("F32_VEC");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_TYPE=vec4<f32>");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_TYPE=vec4<f32>");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=4u");
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("F32");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_TYPE=f32");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_TYPE=f32");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("FLOAT_PARALLEL");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("F16");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_TYPE=f16");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_TYPE=f32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_I32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("FLOAT_PARALLEL");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("I32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_TYPE=i32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_TYPE=i32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_BF16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("BF16");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_TYPE=u32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_TYPE=f32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                {
                    std::string type_ಊರ್ಧ್ವ = type_str;
                    std::transform(type_ಊರ್ಧ್ವ.begin(), type_ಊರ್ಧ್ವ.end(), type_ಊರ್ಧ್ವ.begin(), ::ದೊಡ್ಡ_ಅಕ್ಷರ);

                    ಆಯ್ಕೆ (ಕೀಲಿ.src_type) {
                        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_TYPE=u32");
                                ನಿರೂಪಿಸುತ್ತದೆ.push_back("U32_DEQUANT_HELPERS");
                                ನಿಲ್ಲಿಸು;
                            }
                        ಪೂರ್ವನಿಯೋಜಿತ:
                            {
                                ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("BYTE_HELPERS");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_T");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ);
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_SCALE_MIN");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_TABLES");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_GRID");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_LUT");

                    variant += "_";
                    variant += type_str;

                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_TYPE=f32");

                    ಆದರೆ (ಕೀಲಿ.src_type == GGML_TYPE_Q1_0) {
                        ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=128u");
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((ಕೀಲಿ.src_type >= GGML_TYPE_Q4_0 && ಕೀಲಿ.src_type <= GGML_TYPE_Q8_1) ||
                               ಕೀಲಿ.src_type == GGML_TYPE_IQ4_NL || ಕೀಲಿ.src_type == GGML_TYPE_MXFP4) {
                        ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=32u");
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.src_type == GGML_TYPE_Q2_0 || ಕೀಲಿ.src_type == GGML_TYPE_NVFP4) {
                        ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=64u");
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.src_type >= GGML_TYPE_Q2_K) {
                        ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=256u");
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=1u");
                    }
                    ನಿಲ್ಲಿಸು;
                }
        }

        ಆದರೆ (ಕೀಲಿ.vectorized) {
            variant += "_vec";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back("WG_SIZE=" + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_get_ಅಡ್ಡಸಾಲುಗಳು, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_ಅಡ್ಡಸಾಲುಗಳು_pipelines[ಕೀಲಿ]  = pipeline;
        ಹಿಂತಿರುಗಿ get_ಅಡ್ಡಸಾಲುಗಳು_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_scale_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        ಸ್ವಯಂಚಾಲಿತ it = scale_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != scale_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "scale";

        ಆದರೆ (ಕೀಲಿ.inplace) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("INPLACE");
            variant += "_inplace";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_scale, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        decisions->inplace       = ಕೀಲಿ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[ಕೀಲಿ]     = pipeline;
        ಹಿಂತಿರುಗಿ scale_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_solve_tri_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                               = context.dst->type;
        ಕೀಲಿ.n                                  = (ಪೂರ್ಣಾಂಕ) context.src0->ne[0];
        ಕೀಲಿ.k                                  = (ಪೂರ್ಣಾಂಕ) context.src1->ne[0];

        ಸ್ವಯಂಚಾಲಿತ it = solve_tri_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != solve_tri_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "solve_tri";

        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ಸ್ಥಿರಾಂಕ uint32_t wg_ಗಾತ್ರ       = std::min((uint32_t) ಕೀಲಿ.n, context.max_wg_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ uint32_t k_tile        = wg_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t ಬೈಟುಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ((uint32_t) ಕೀಲಿ.n + wg_ಗಾತ್ರ) * GGML_WEBGPU_F32_ಗಾತ್ರ_ಬೈಟುಗಳು;
        ಸ್ಥಿರಾಂಕ uint32_t batch_n       = (uint32_t) (context.wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು / ಬೈಟುಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು);

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("N=") + std::to_string(ಕೀಲಿ.n));
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ಗಾತ್ರ));
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_solve_tri, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = wg_ಗಾತ್ರ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ solve_tri_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                              = context.dst->type;
        ಕೀಲಿ.vectorized                        = context.src1->ne[0] == 4;

        ಸ್ವಯಂಚಾಲಿತ it = ssm_conv_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != ssm_conv_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "ssm_conv";

        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        ಆದರೆ (ಕೀಲಿ.vectorized) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t ಖಂಡ_ಗಾತ್ರ_3    = 32u;
        constexpr uint32_t ಟೋಕನುಗಳು_ಪ್ರತಿ_wg = 8u;

        ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLOCK_SIZE=" + std::to_string(ಖಂಡ_ಗಾತ್ರ_3) + "u");
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("TOKENS_PER_WG=" + std::to_string(ಟೋಕನುಗಳು_ಪ್ರತಿ_wg) + "u");

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_ssm_conv, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->ಖಂಡ_ಗಾತ್ರ_3    = ಖಂಡ_ಗಾತ್ರ_3;
        decisions->ಟೋಕನುಗಳು_ಪ್ರತಿ_wg = ಟೋಕನುಗಳು_ಪ್ರತಿ_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[ಕೀಲಿ]  = pipeline;
        ಹಿಂತಿರುಗಿ ssm_conv_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_ssm_ಪರಿಶೀಲನೆ_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_ಪರಿಶೀಲನೆ_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                              = context.dst->type;
        ಕೀಲಿ.d_state                           = (ಪೂರ್ಣಾಂಕ) context.src0->ne[0];
        ಕೀಲಿ.xbc_overlap                       = xbc_overlap;

        ಸ್ವಯಂಚಾಲಿತ it = ssm_ಪರಿಶೀಲನೆ_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != ssm_ಪರಿಶೀಲನೆ_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "ssm_scan";

        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ಸ್ಥಿರಾಂಕ uint32_t wg_ಗಾತ್ರ = (uint32_t) ಕೀಲಿ.d_state;

        constexpr uint32_t ಟೋಕನುಗಳು_ಪ್ರತಿ_tile = 4u;

        ನಿರೂಪಿಸುತ್ತದೆ.push_back("WG_SIZE=" + std::to_string(wg_ಗಾತ್ರ) + "u");
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("TOKENS_PER_TILE=" + std::to_string(ಟೋಕನುಗಳು_ಪ್ರತಿ_tile) + "u");

        ಆದರೆ (context.supports_subgroups) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } ಇಲ್ಲದಿದ್ದರೆ {
            variant += "_wg_reduce";
        }

        ಆದರೆ (ಕೀಲಿ.xbc_overlap == 0b110) {  // x/B
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.xbc_overlap == 0b011) {  // B/C
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.xbc_overlap == 0b111) {  // x/B/C
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(ಕೀಲಿ.d_state);

        ಸ್ವಯಂಚಾಲಿತ processed             = preprocessor.preprocess(wgsl_ssm_ಪರಿಶೀಲನೆ, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions             = std::make_shared<ggml_webgpu_ssm_ಪರಿಶೀಲನೆ_shader_decisions>();
        decisions->wg_ಗಾತ್ರ         = wg_ಗಾತ್ರ;
        decisions->ಟೋಕನುಗಳು_ಪ್ರತಿ_tile = ಟೋಕನುಗಳು_ಪ್ರತಿ_tile;
        decisions->xbc_overlap     = ಕೀಲಿ.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_ಪರಿಶೀಲನೆ_pipelines[ಕೀಲಿ]    = pipeline;
        ಹಿಂತಿರುಗಿ ssm_ಪರಿಶೀಲನೆ_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                                     = context.dst->type;
        ಕೀಲಿ.s_v                                      = (ಪೂರ್ಣಾಂಕ) context.src2->ne[0];
        ಕೀಲಿ.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        ಕೀಲಿ.fused_cache                              = context.dst_fuse != ಶೂನ್ಯಸೂಚಕ;

        ಸ್ವಯಂಚಾಲಿತ it = gated_delta_net_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != gated_delta_net_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "gated_delta_net";

        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        ಆದರೆ (ಕೀಲಿ.kda) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("KDA");
            variant += "_kda";
        }

        ಆದರೆ (ಕೀಲಿ.fused_cache) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back("S_V=" + std::to_string(ಕೀಲಿ.s_v) + "u");
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("WG_SIZE=" + std::to_string(ಕೀಲಿ.s_v) + "u");

        ಸ್ವಯಂಚಾಲಿತ            processed      = preprocessor.preprocess(wgsl_gated_delta_net, ನಿರೂಪಿಸುತ್ತದೆ);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ gated_delta_net_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_pad_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        ಸ್ವಯಂಚಾಲಿತ it = pad_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != pad_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "pad";

        ಆದರೆ (ಕೀಲಿ.circular) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("CIRCULAR");
            variant += "_circular";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_pad, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[ಕೀಲಿ]       = pipeline;
        ಹಿಂತಿರುಗಿ pad_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src0_type                            = context.src0->type;

        ಸ್ವಯಂಚಾಲಿತ it = quantize_q8_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != quantize_q8_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "quantize_q8";

        uint32_t wg_ಗಾತ್ರ = WEBGPU_MUL_MAT_VEC_WG_ಗಾತ್ರ;

        ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC1_INNER_TYPE=f32");
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ಗಾತ್ರ));

        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_ಊರ್ಧ್ವ  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_ಊರ್ಧ್ವ.begin(), type_ಊರ್ಧ್ವ.end(), type_ಊರ್ಧ್ವ.begin(), ::ದೊಡ್ಡ_ಅಕ್ಷರ);

        ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_" + type_ಊರ್ಧ್ವ);
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("Q8_1_T");

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        ಸ್ವಯಂಚಾಲಿತ processed             = preprocessor.preprocess(shader_src, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ         = wg_ಗಾತ್ರ;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ quantize_q8_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src0_type                            = context.src0->type;
        ಕೀಲಿ.src1_type                            = context.src1->type;
        ಕೀಲಿ.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        ಕೀಲಿ.ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ   = context.dst->ne[1];
        ಕೀಲಿ.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_ಗುಣಲಬ್ಧ, context.vendor);
        ಕೀಲಿ.src_overlap = src_overlap;

        ಸ್ವಯಂಚಾಲಿತ it = mul_mat_vec_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != mul_mat_vec_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant    = "mul_mat_vec";
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        ಆಯ್ಕೆ (context.src0->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=f32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=f16");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_BF16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=u32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_BF16");
                variant += "_bf16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_ಊರ್ಧ್ವ  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_ಊರ್ಧ್ವ.begin(), type_ಊರ್ಧ್ವ.end(), type_ಊರ್ಧ್ವ.begin(), ::ದೊಡ್ಡ_ಅಕ್ಷರ);

                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("BYTE_HELPERS");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_" + type_ಊರ್ಧ್ವ);
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("U32_DEQUANT_HELPERS");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=u32");
                    ಆಯ್ಕೆ (context.src0->type) {
                        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                            ಆದರೆ (ಕೀಲಿ.use_mmvq) {
                                ನಿರೂಪಿಸುತ್ತದೆ.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                            ಆದರೆ (ಕೀಲಿ.use_mmvq) {
                                ನಿರೂಪಿಸುತ್ತದೆ.push_back("K_QUANTS_HANDLING");
                            }
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_GRID");
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_GRID");
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_TABLES");
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_LUT");
                            ಆದರೆ (ಕೀಲಿ.use_mmvq) {
                                ನಿರೂಪಿಸುತ್ತದೆ.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_LUT");
                            ನಿಲ್ಲಿಸು;
                        ಪೂರ್ವನಿಯೋಜಿತ:
                            ನಿಲ್ಲಿಸು;
                    }
                    ನಿಲ್ಲಿಸು;
                }
        }

        // src1 type (vector)
        ಆಯ್ಕೆ (context.src1->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(ಕೀಲಿ.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_ಗಾತ್ರ        = WEBGPU_MUL_MAT_VEC_WG_ಗಾತ್ರ;
        uint32_t outputs_ಪ್ರತಿ_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_ಪ್ರತಿ_WG;

        ಆದರೆ (ಕೀಲಿ.src0_type == GGML_TYPE_Q1_0) {
            outputs_ಪ್ರತಿ_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_ಪ್ರತಿ_WG;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.src0_type >= GGML_TYPE_Q2_K) {
            outputs_ಪ್ರತಿ_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_ಪ್ರತಿ_WG;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.src0_type >= GGML_TYPE_Q4_0) {
            outputs_ಪ್ರತಿ_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_ಪ್ರತಿ_WG;
        }

        ಆದರೆ (ಕೀಲಿ.use_mmvq) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("MMVQ");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("Q8_1_T");
        }

        ಆದರೆ (ಕೀಲಿ.src_overlap) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ಗಾತ್ರ));
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_ಪ್ರತಿ_wg));
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        ಆದರೆ (ಕೀಲಿ.vectorized) {
            variant += "_vectorized";
        }
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("NUM_COLS=") + std::to_string(ಕೀಲಿ.ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ));

        ಸ್ವಯಂಚಾಲಿತ processed            = preprocessor.preprocess(shader_src, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_ಗಾತ್ರ        = wg_ಗಾತ್ರ;
        decisions->outputs_ಪ್ರತಿ_wg = outputs_ಪ್ರತಿ_wg;
        decisions->vec_ಗಾತ್ರ       = ಕೀಲಿ.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ mul_mat_vec_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_mul_mat_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src0_type                        = context.src0->type;
        ಕೀಲಿ.src1_type                        = context.src1->type;
        ಕೀಲಿ.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        ಕೀಲಿ.use_subgroup_ಮಾತೃಕೆ = context.supports_subgroup_ಮಾತೃಕೆ;
        ಕೀಲಿ.src_overlap         = src_overlap;

        ಸ್ವಯಂಚಾಲಿತ it = mul_mat_fast_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != mul_mat_fast_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * shader_src = ಕೀಲಿ.use_subgroup_ಮಾತೃಕೆ ? wgsl_mul_mat_subgroup_ಮಾತೃಕೆ : wgsl_mul_mat_reg_tile;
        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = ಕೀಲಿ.use_subgroup_ಮಾತೃಕೆ ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        ಆಯ್ಕೆ (context.src1->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC1_INNER_TYPE=f32");
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC1_INNER_TYPE=f16");
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *                    src0_name   = src0_traits->type_name;

        ಆಯ್ಕೆ (context.src0->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=f32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=f16");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F16");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_BF16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=u32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_BF16");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                {
                    std::string type_ಊರ್ಧ್ವ = src0_name;
                    std::transform(type_ಊರ್ಧ್ವ.begin(), type_ಊರ್ಧ್ವ.end(), type_ಊರ್ಧ್ವ.begin(), ::ದೊಡ್ಡ_ಅಕ್ಷರ);

                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("BYTE_HELPERS");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_" + type_ಊರ್ಧ್ವ);
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC0_SHMEM_" + type_ಊರ್ಧ್ವ);
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC1_SHMEM_FLOAT");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("U32_DEQUANT_HELPERS");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=u32");

                    ಆಯ್ಕೆ (context.src0->type) {
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_GRID");
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_GRID");
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_TABLES");
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_LUT");
                            ನಿಲ್ಲಿಸು;
                        ಪೂರ್ವನಿಯೋಜಿತ:
                            ನಿಲ್ಲಿಸು;
                    }

                    variant += std::string("_") + src0_name;
                    ನಿಲ್ಲಿಸು;
                }
        }

        // VEC/SCALAR controls
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(ಕೀಲಿ.vectorized ? "VEC" : "SCALAR");

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        ಆದರೆ (ಕೀಲಿ.use_subgroup_ಮಾತೃಕೆ) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } ಇಲ್ಲದಿದ್ದರೆ {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        ಆದರೆ (ಕೀಲಿ.use_subgroup_ಮಾತೃಕೆ) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_ಗಾತ್ರ) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_ಮಾತೃಕೆ_M) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_ಮಾತೃಕೆ_N) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        ಆದರೆ (ಕೀಲಿ.vectorized) {
            variant += "_vectorized";
        }

        ಆದರೆ (ಕೀಲಿ.src_overlap) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ಆದರೆ (!ಕೀಲಿ.use_subgroup_ಮಾತೃಕೆ) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_M) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_N) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        ಸ್ವಯಂಚಾಲಿತ processed = preprocessor.preprocess(shader_src, ನಿರೂಪಿಸುತ್ತದೆ);

        ಸ್ವಯಂಚಾಲಿತ decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_ಮಾತೃಕೆ = ಕೀಲಿ.use_subgroup_ಮಾತೃಕೆ;
        ಆದರೆ (ಕೀಲಿ.use_subgroup_ಮಾತೃಕೆ) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_ಮಾತೃಕೆ_m = WEBGPU_MUL_MAT_SUBGROUP_ಮಾತೃಕೆ_M;
            decisions->subgroup_ಮಾತೃಕೆ_n = WEBGPU_MUL_MAT_SUBGROUP_ಮಾತೃಕೆ_N;
            decisions->wg_ಗಾತ್ರ           = context.max_subgroup_ಗಾತ್ರ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            decisions->wg_ಗಾತ್ರ_m       = WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_M;
            decisions->wg_ಗಾತ್ರ_n       = WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_N;
            decisions->wg_ಗಾತ್ರ         = WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_M * WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_N;
            decisions->mul_mat_wg_ಗಾತ್ರ = WEBGPU_MUL_MAT_WG_ಗಾತ್ರ;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ mul_mat_fast_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.n_experts                                  = (uint32_t) context.src0->ne[2];

        ಸ್ವಯಂಚಾಲಿತ it = mul_mat_id_gather_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != mul_mat_id_gather_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }
        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ) + "u");
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("N_EXPERT=") + std::to_string(ಕೀಲಿ.n_experts) + "u");
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_N) + "u");

        ಸ್ವಯಂಚಾಲಿತ processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ = context.max_wg_ಗಾತ್ರ;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src0_type                           = context.src0->type;
        ಕೀಲಿ.src1_type                           = context.src1->type;
        ಕೀಲಿ.n_experts                           = context.src0->ne[2];
        ಕೀಲಿ.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        ಸ್ವಯಂಚಾಲಿತ it = mul_mat_id_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != mul_mat_id_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "mul_mat_id";
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_MAT_ID");

        // src1 type
        ಆಯ್ಕೆ (context.src1->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC1_INNER_TYPE=f32");
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC1_INNER_TYPE=f16");
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *                    src0_name   = src0_traits->type_name;

        ಆಯ್ಕೆ (context.src0->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=f32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC1_SHMEM_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=f16");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC1_SHMEM_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_BF16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=u32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC1_SHMEM_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_BF16");
                variant += "_bf16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                {
                    std::string type_ಊರ್ಧ್ವ = src0_name;
                    std::transform(type_ಊರ್ಧ್ವ.begin(), type_ಊರ್ಧ್ವ.end(), type_ಊರ್ಧ್ವ.begin(), ::ದೊಡ್ಡ_ಅಕ್ಷರ);

                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("BYTE_HELPERS");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC0_SHMEM_" + type_ಊರ್ಧ್ವ);
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("INIT_SRC1_SHMEM_FLOAT");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("U32_DEQUANT_HELPERS");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=u32");

                    ಆಯ್ಕೆ (context.src0->type) {
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_GRID");
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_GRID");
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_TABLES");
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_LUT");
                            ನಿಲ್ಲಿಸು;
                        ಪೂರ್ವನಿಯೋಜಿತ:
                            ನಿಲ್ಲಿಸು;
                    }

                    variant += std::string("_") + src0_name;
                    ನಿಲ್ಲಿಸು;
                }
        }

        // VEC/SCALAR controls
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(ಕೀಲಿ.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ಸ್ಥಿರಾಂಕ uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        ನಿರೂಪಿಸುತ್ತದೆ.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_M) + "u");
        ನಿರೂಪಿಸುತ್ತದೆ.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        ಆದರೆ (ಕೀಲಿ.vectorized) {
            variant += "_vectorized";
        }

        ಸ್ವಯಂಚಾಲಿತ processed = preprocessor.preprocess(wgsl_mul_mat_id, ನಿರೂಪಿಸುತ್ತದೆ);

        ಸ್ವಯಂಚಾಲಿತ decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_ಗಾತ್ರ_m = WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_M;
        decisions->wg_ಗಾತ್ರ_n = WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_N;
        decisions->wg_ಗಾತ್ರ   = WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_M * WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ mul_mat_id_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src0_type                           = context.src0->type;
        ಕೀಲಿ.src1_type                           = context.src1->type;
        ಕೀಲಿ.n_experts                           = context.src0->ne[2];
        ಕೀಲಿ.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        ಸ್ವಯಂಚಾಲಿತ it = mul_mat_id_vec_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != mul_mat_id_vec_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant    = "mul_mat_id_vec";
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        ಆಯ್ಕೆ (context.src1->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC1_INNER_TYPE=f32");
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC1_INNER_TYPE=f16");
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ಆಯ್ಕೆ (context.src0->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=f32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=f16");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_BF16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=u32");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_FLOAT");
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_BF16");
                variant += "_bf16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_ಊರ್ಧ್ವ  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_ಊರ್ಧ್ವ.begin(), type_ಊರ್ಧ್ವ.end(), type_ಊರ್ಧ್ವ.begin(), ::ದೊಡ್ಡ_ಅಕ್ಷರ);

                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("BYTE_HELPERS");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("MUL_ACC_" + type_ಊರ್ಧ್ವ);
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("U32_DEQUANT_HELPERS");
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC0_INNER_TYPE=u32");
                    ಆಯ್ಕೆ (context.src0->type) {
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_GRID");
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_GRID");
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_TABLES");
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                            ನಿರೂಪಿಸುತ್ತದೆ.push_back(type_ಊರ್ಧ್ವ + "_LUT");
                            ನಿಲ್ಲಿಸು;
                        ಪೂರ್ವನಿಯೋಜಿತ:
                            ನಿಲ್ಲಿಸು;
                    }
                    ನಿಲ್ಲಿಸು;
                }
        }

        // VEC/SCALAR controls
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(ಕೀಲಿ.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_ಗಾತ್ರ        = WEBGPU_MUL_MAT_VEC_WG_ಗಾತ್ರ;
        uint32_t outputs_ಪ್ರತಿ_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_ಪ್ರತಿ_WG;

        ಆದರೆ (ಕೀಲಿ.src0_type == GGML_TYPE_Q1_0) {
            outputs_ಪ್ರತಿ_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_ಪ್ರತಿ_WG;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.src0_type >= GGML_TYPE_Q2_K) {
            outputs_ಪ್ರತಿ_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_ಪ್ರತಿ_WG;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.src0_type >= GGML_TYPE_Q4_0) {
            outputs_ಪ್ರತಿ_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_ಪ್ರತಿ_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ಗಾತ್ರ));
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_ಪ್ರತಿ_wg));
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        ಆದರೆ (ಕೀಲಿ.vectorized) {
            variant += "_vectorized";
        }
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("NUM_COLS=1"));

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("N_EXPERTS=") + std::to_string(ಕೀಲಿ.n_experts));

        ಸ್ವಯಂಚಾಲಿತ processed = preprocessor.preprocess(shader_src, ನಿರೂಪಿಸುತ್ತದೆ);

        ಸ್ವಯಂಚಾಲಿತ decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_ಗಾತ್ರ        = wg_ಗಾತ್ರ;
        decisions->outputs_ಪ್ರತಿ_wg = outputs_ಪ್ರತಿ_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ mul_mat_id_vec_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_unary_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ                     is_unary = context.dst->op == GGML_OP_UNARY;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                      op       = is_unary ? (ಪೂರ್ಣಾಂಕ) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_ಕೀಲಿ ಕೀಲಿ      = {};
        ಕೀಲಿ.type                                = context.dst->type;
        ಕೀಲಿ.op                                  = op;
        ಕೀಲಿ.is_unary                            = is_unary;
        ಕೀಲಿ.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        ಕೀಲಿ.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        ಸ್ವಯಂಚಾಲಿತ it = unary_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != unary_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant =
            ಕೀಲಿ.is_unary ? ggml_unary_op_name((ggml_unary_op) ಕೀಲಿ.op) : ggml_op_name((ggml_op) ಕೀಲಿ.op);
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(variant);

        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for unary shader");
        }

        ಆದರೆ (ಕೀಲಿ.inplace) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("INPLACE");
            variant += "_inplace";
        }

        ಆದರೆ (op == GGML_OP_TRI) {
            ಆಯ್ಕೆ (ಕೀಲಿ.ttype) {
                ಸ್ಥಿತಿ GGML_TRI_TYPE_ಅಧೋ:
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_TRI_TYPE_ಅಧೋ_DIAG:
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_TRI_TYPE_ಊರ್ಧ್ವ:
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_TRI_TYPE_ಊರ್ಧ್ವ_DIAG:
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_unary, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        decisions->inplace       = ಕೀಲಿ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[ಕೀಲಿ]     = pipeline;
        ಹಿಂತಿರುಗಿ unary_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ಕೀಲಿ.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ಕೀಲಿ.src_overlap                           = src_overlap;

        ಸ್ವಯಂಚಾಲಿತ it = rms_norm_mul_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != rms_norm_mul_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        ಆದರೆ (ಕೀಲಿ.inplace) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("INPLACE");
            variant += "_inplace";
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.overlap) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("OVERLAP");
            variant += "_overlap";
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.src_overlap) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_ಗಾತ್ರ     = context.max_wg_ಗಾತ್ರ;
        pipeline_decisions->inplace     = ಕೀಲಿ.inplace;
        pipeline_decisions->overlap     = ಕೀಲಿ.overlap;
        pipeline_decisions->src_overlap = ಕೀಲಿ.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[ಕೀಲಿ]     = pipeline;
        ಹಿಂತಿರುಗಿ rms_norm_mul_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_ದ್ವಿಮಾನ_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_ದ್ವಿಮಾನ_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                            = context.dst->type;
        ಕೀಲಿ.op                              = context.dst->op;
        ಕೀಲಿ.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ಕೀಲಿ.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ಕೀಲಿ.src_overlap                     = src_overlap;

        ಸ್ವಯಂಚಾಲಿತ it = ದ್ವಿಮಾನ_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != ದ್ವಿಮಾನ_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              op_name = ggml_op_name((ggml_op) ಕೀಲಿ.op);
        std::string              variant = op_name;

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("OP_") + op_name);

        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for binary shader");
        }

        ಆದರೆ (ಕೀಲಿ.inplace) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("INPLACE");
            variant += "_inplace";
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.overlap) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("OVERLAP");
            variant += "_overlap";
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.src_overlap) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed                  = preprocessor.preprocess(wgsl_ದ್ವಿಮಾನ, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ pipeline_decisions         = std::make_shared<ggml_webgpu_ದ್ವಿಮಾನ_shader_decisions>();
        pipeline_decisions->wg_ಗಾತ್ರ     = context.max_wg_ಗಾತ್ರ;
        pipeline_decisions->inplace     = ಕೀಲಿ.inplace;
        pipeline_decisions->overlap     = ಕೀಲಿ.overlap;
        pipeline_decisions->src_overlap = ಕೀಲಿ.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ದ್ವಿಮಾನ_pipelines[ಕೀಲಿ]    = pipeline;
        ಹಿಂತಿರುಗಿ ದ್ವಿಮಾನ_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_ಕೂಡಿಸು_id_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ಕೂಡಿಸು_id_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        ಸ್ವಯಂಚಾಲಿತ it = ಕೂಡಿಸು_id_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != ಕೂಡಿಸು_id_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant    = "add_id";
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *             shader_src = wgsl_ಕೂಡಿಸು_id;

        ಆದರೆ (ಕೀಲಿ.inplace) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("INPLACE");
            variant += "_inplace";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed              = preprocessor.preprocess(shader_src, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_ಗಾತ್ರ = context.max_wg_ಗಾತ್ರ;
        pipeline_decisions->inplace = ಕೀಲಿ.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ಕೂಡಿಸು_id_pipelines[ಕೀಲಿ]    = pipeline;
        ಹಿಂತಿರುಗಿ pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_concat_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                            = context.dst->type;
        ಕೀಲಿ.src_overlap                     = src_overlap;

        ಸ್ವಯಂಚಾಲಿತ it = concat_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != concat_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "concat";

        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_I32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_I32");
                variant += "_i32";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for concat shader");
        }

        ಆದರೆ (ಕೀಲಿ.src_overlap) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_concat, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_ದ್ವಿಮಾನ_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        decisions->src_overlap   = ಕೀಲಿ.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[ಕೀಲಿ]    = pipeline;
        ಹಿಂತಿರುಗಿ concat_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_repeat_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                            = context.dst->type;

        ಸ್ವಯಂಚಾಲಿತ it = repeat_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != repeat_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "repeat";

        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_I32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_I32");
                variant += "_i32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_I16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_I16");
                variant += "_i16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_repeat, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[ಕೀಲಿ]    = pipeline;
        ಹಿಂತಿರುಗಿ repeat_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_flash_attn_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ kv_overlap) {
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ can_use_subgroup_ಮಾತೃಕೆ = ggml_webgpu_flash_attn_can_use_subgroup_ಮಾತೃಕೆ_ಮಾರ್ಗ(
            context.supports_subgroup_ಮಾತೃಕೆ, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_ಮಾತೃಕೆ                    = can_use_subgroup_ಮಾತೃಕೆ;
        decisions.q_tile = decisions.use_sg_ಮಾತೃಕೆ ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.common                              = ggml_webgpu_flash_attn_make_common_pipeline_ಕೀಲಿ(
            context, decisions.use_sg_ಮಾತೃಕೆ ? context.sg_mat_k : 1u, kv_overlap);
        ಕೀಲಿ.common.k_ನೇರ &= decisions.use_sg_ಮಾತೃಕೆ && ಕೀಲಿ.common.k_type == GGML_TYPE_F16;
        ಕೀಲಿ.common.v_ನೇರ &= decisions.use_sg_ಮಾತೃಕೆ && ಕೀಲಿ.common.v_type == GGML_TYPE_F16;
        ಕೀಲಿ.use_sg_ಮಾತೃಕೆ = decisions.use_sg_ಮಾತೃಕೆ;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು, decisions.q_tile, decisions.use_sg_ಮಾತೃಕೆ ? context.sg_mat_n : 1u,
            ಕೀಲಿ.common.ತಲೆ_dim_qk, ಕೀಲಿ.common.ತಲೆ_dim_v, ಕೀಲಿ.common.has_ಮುಖವಾಡ,
            ಕೀಲಿ.common.k_ನೇರ || ಕೀಲಿ.common.v_ನೇರ);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        ಆದರೆ (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_ಮಾತೃಕೆ);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_ಮಾತೃಕೆ = ಸುಳ್ಳು;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            ಕೀಲಿ.common.k_ನೇರ     = ಸುಳ್ಳು;
            ಕೀಲಿ.common.v_ನೇರ     = ಸುಳ್ಳು;
            ಕೀಲಿ.use_sg_ಮಾತೃಕೆ       = ಸುಳ್ಳು;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು, decisions.q_tile, 1u, ಕೀಲಿ.common.ತಲೆ_dim_qk, ಕೀಲಿ.common.ತಲೆ_dim_v,
                ಕೀಲಿ.common.has_ಮುಖವಾಡ, ಕೀಲಿ.common.k_ನೇರ || ಕೀಲಿ.common.v_ನೇರ);
            GGML_ASSERT(max_kv_tile > 0);
        }

        decisions.kv_tile = decisions.use_sg_ಮಾತೃಕೆ ?
                                std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES) :
                                std::min(GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE, max_kv_tile);
        decisions.wg_ಗಾತ್ರ =
            decisions.use_sg_ಮಾತೃಕೆ ?
                std::max(context.max_subgroup_ಗಾತ್ರ, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_ಗಾತ್ರ) :
                std::min(context.max_wg_ಗಾತ್ರ, std::max(GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_ಗಾತ್ರ,
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * context.max_subgroup_ಗಾತ್ರ));

        ಆದರೆ (ಕೀಲಿ.common.k_ನೇರ || ಕೀಲಿ.common.v_ನೇರ) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            ಆಗ (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_ಮಾತೃಕೆ ? context.sg_mat_n : context.min_subgroup_ಗಾತ್ರ;
            }
        }

        ಸ್ವಯಂಚಾಲಿತ it = flash_attn_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != flash_attn_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::string              variant = decisions.use_sg_ಮಾತೃಕೆ ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ = ggml_webgpu_flash_attn_common_ನಿರೂಪಿಸುತ್ತದೆ(ಕೀಲಿ.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *             shader_src = ಶೂನ್ಯಸೂಚಕ;
        ಆದರೆ (!ಕೀಲಿ.use_sg_ಮಾತೃಕೆ) {
            shader_src = wgsl_flash_attn_tile;
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_ಗಾತ್ರ) + "u");
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_ಗಾತ್ರ) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_ಗಾತ್ರ) + "_" +
                       std::to_string(context.max_subgroup_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            shader_src = wgsl_flash_attn;
            ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        ಸ್ವಯಂಚಾಲಿತ            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, ನಿರೂಪಿಸುತ್ತದೆ), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ flash_attn_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.common = ggml_webgpu_flash_attn_make_common_pipeline_ಕೀಲಿ(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ಅಗಲ,
                                                                     kv_overlap);

        ಸ್ವಯಂಚಾಲಿತ it = flash_attn_vec_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != flash_attn_vec_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು, ಕೀಲಿ.common.ತಲೆ_dim_qk,
                                                                   ಕೀಲಿ.common.ತಲೆ_dim_v, ಕೀಲಿ.common.has_ಮುಖವಾಡ,
                                                                   ಕೀಲಿ.common.k_ನೇರ || ಕೀಲಿ.common.v_ನೇರ);
        decisions.wg_ಗಾತ್ರ = context.max_subgroup_ಗಾತ್ರ;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ =
            ggml_webgpu_flash_attn_common_ನಿರೂಪಿಸುತ್ತದೆ(ಕೀಲಿ.common, variant, 1u, decisions.kv_tile, decisions.wg_ಗಾತ್ರ);
        ಆದರೆ (ಕೀಲಿ.common.has_ಮುಖವಾಡ) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("BLK");
            variant.resize(variant.size() - (ಗಾತ್ರ("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_ವಿಭಜಿಸು = context.min_subgroup_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t D       = ಕೀಲಿ.common.ತಲೆ_dim_qk | ಕೀಲಿ.common.ತಲೆ_dim_v;
        ಸ್ಥಿರಾಂಕ uint32_t D_lsb   = D & (~(D - 1u));
        d_ವಿಭಜಿಸು                = std::min(std::min(context.min_subgroup_ಗಾತ್ರ, 4u), std::max(D_lsb / 4u, 1u));

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("D_SPLIT=") + std::to_string(d_ವಿಭಜಿಸು));
        variant += "_dsplit" + std::to_string(d_ವಿಭಜಿಸು);

        ಸ್ವಯಂಚಾಲಿತ            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_ವಿಭಜಿಸು, ನಿರೂಪಿಸುತ್ತದೆ), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ flash_attn_vec_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.kv_tile                                 = kv_tile;
        ಸ್ವಯಂಚಾಲಿತ it                                     = flash_attn_blk_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != flash_attn_blk_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "flash_attn_vec_blk";

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("KV_TILE=") + std::to_string(ಕೀಲಿ.kv_tile));
        variant += std::string("_kvt") + std::to_string(ಕೀಲಿ.kv_tile);

        uint32_t wg_ಗಾತ್ರ = 1;
        ಆಗ ((wg_ಗಾತ್ರ << 1) <= context.max_wg_ಗಾತ್ರ) {
            wg_ಗಾತ್ರ <<= 1;
        }
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ಗಾತ್ರ));
        variant += std::string("_wg") + std::to_string(wg_ಗಾತ್ರ);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, ನಿರೂಪಿಸುತ್ತದೆ), variant);
        flash_attn_blk_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ flash_attn_blk_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.ತಲೆ_dim_v                                     = (uint32_t) context.src2->ne[0];
        ಕೀಲಿ.dst_type                                       = context.dst->type;
        ಕೀಲಿ.wg_ಗಾತ್ರ                                        = context.max_wg_ಗಾತ್ರ;
        ಸ್ವಯಂಚಾಲಿತ it                                            = flash_attn_vec_reduce_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != flash_attn_vec_reduce_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "flash_attn_vec_reduce";

        ಆಯ್ಕೆ (ಕೀಲಿ.dst_type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F32");
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F16");
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(ಕೀಲಿ.dst_type);

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("HEAD_DIM_V=") + std::to_string(ಕೀಲಿ.ತಲೆ_dim_v));
        variant += std::string("_hsv") + std::to_string(ಕೀಲಿ.ತಲೆ_dim_v);

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));
        variant += std::string("_wg") + std::to_string(context.max_wg_ಗಾತ್ರ);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, ನಿರೂಪಿಸುತ್ತದೆ), variant);
        flash_attn_vec_reduce_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ flash_attn_vec_reduce_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_cpy_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.src_type                     = context.src0->type;
        ಕೀಲಿ.dst_type                     = context.dst->type;

        ಸ್ವಯಂಚಾಲಿತ it = cpy_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != cpy_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "cpy";

        ಆಯ್ಕೆ (ಕೀಲಿ.src_type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_F16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_I32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_I32");
                variant += "_i32";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        ಆಯ್ಕೆ (ಕೀಲಿ.dst_type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_I32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_I32");
                variant += "_i32";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_cpy, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[ಕೀಲಿ]       = pipeline;
        ಹಿಂತಿರುಗಿ cpy_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_glu_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ src_overlap) {
        ggml_webgpu_glu_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.glu_op                       = ggml_get_glu_op(context.dst);
        ಕೀಲಿ.type                         = context.dst->type;
        ಕೀಲಿ.ವಿಭಜಿಸು                        = (context.src1 != ಶೂನ್ಯಸೂಚಕ);
        ಕೀಲಿ.src_overlap                  = src_overlap;

        ಸ್ವಯಂಚಾಲಿತ it = glu_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != glu_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "glu";

        ಆಯ್ಕೆ (ಕೀಲಿ.glu_op) {
            ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("OP_REGLU");
                variant += "_reglu";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("OP_GEGLU");
                variant += "_geglu";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("OP_SWIGLU");
                variant += "_swiglu";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported GLU op");
        }
        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        ಆದರೆ (ಕೀಲಿ.src_overlap) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೀಲಿ.ವಿಭಜಿಸು) {
            variant += "_split";
        } ಇಲ್ಲದಿದ್ದರೆ {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("NO_SPLIT");
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_glu, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        decisions->src_overlap   = ಕೀಲಿ.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[ಕೀಲಿ]       = pipeline;
        ಹಿಂತಿರುಗಿ glu_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_rope_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.type                          = context.dst->type;
        ಕೀಲಿ.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ಕೀಲಿ.has_ff                        = (context.src2 != ಶೂನ್ಯಸೂಚಕ);

        ಸ್ವಯಂಚಾಲಿತ it = rope_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != rope_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "rope";

        ಆಯ್ಕೆ (ಕೀಲಿ.type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F32");
                variant += "_f32";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("TYPE_F16");
                variant += "_f16";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        ಆದರೆ (ಕೀಲಿ.inplace) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("INPLACE");
            variant += "_inplace";
        }

        ಆದರೆ (ಕೀಲಿ.has_ff) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("FF_FUNC");
            variant += "_ff";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_rope, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        decisions->inplace       = ಕೀಲಿ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[ಕೀಲಿ]      = pipeline;
        ಹಿಂತಿರುಗಿ rope_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_soft_max_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.ಮುಖವಾಡ_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        ಕೀಲಿ.has_ಮುಖವಾಡ                          = (context.src1 != ಶೂನ್ಯಸೂಚಕ);
        ಕೀಲಿ.has_sink                          = (context.src2 != ಶೂನ್ಯಸೂಚಕ);
        ಕೀಲಿ.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        ಸ್ವಯಂಚಾಲಿತ it = soft_max_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != soft_max_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "soft_max";

        ಆದರೆ (ಕೀಲಿ.has_ಮುಖವಾಡ) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("HAS_MASK");
            ಆಯ್ಕೆ (ಕೀಲಿ.ಮುಖವಾಡ_type) {
                ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("MASK_F32");
                    variant += "_mask_f32";
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ನಿರೂಪಿಸುತ್ತದೆ.push_back("MASK_F16");
                    variant += "_mask_f16";
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        ಆದರೆ (ಕೀಲಿ.has_sink) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("HAS_SINK");
            variant += "_sink";
        }

        ಆದರೆ (ಕೀಲಿ.inplace) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("INPLACE");
            variant += "_inplace";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_soft_max, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        decisions->inplace       = ಕೀಲಿ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[ಕೀಲಿ]  = pipeline;
        ಹಿಂತಿರುಗಿ soft_max_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_conv2d_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.ತೂಕ_type                     = context.src0->type;
        ಕೀಲಿ.ಒಳಹರಿವು_type                      = context.src1->type;
        ಕೀಲಿ.output_type                     = context.dst->type;

        ಸ್ವಯಂಚಾಲಿತ it = conv2d_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != conv2d_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "conv_2d";

        ಸ್ವಯಂಚಾಲಿತ ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ = [&](ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * prefix, ggml_type type) {
            std::string s_ಪೂರ್ವಪ್ರತ್ಯಯ = prefix;
            ಆದರೆ (type == GGML_TYPE_F32) {
                ನಿರೂಪಿಸುತ್ತದೆ.push_back(s_ಪೂರ್ವಪ್ರತ್ಯಯ + "=f32");
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (type == GGML_TYPE_F16) {
                ನಿರೂಪಿಸುತ್ತದೆ.push_back(s_ಪೂರ್ವಪ್ರತ್ಯಯ + "=f16");
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ("WEIGHT_TYPE", ಕೀಲಿ.ತೂಕ_type);
        ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ("INPUT_TYPE", ಕೀಲಿ.ಒಳಹರಿವು_type);
        ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ("OUTPUT_TYPE", ಕೀಲಿ.output_type);

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_conv2d, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[ಕೀಲಿ]    = pipeline;
        ಹಿಂತಿರುಗಿ conv2d_pipelines[ಕೀಲಿ];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context, ತಾರ್ಕಿಕ whcn) {
        ggml_webgpu_conv2d_dw_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.ತೂಕ_type                        = context.src0->type;
        ಕೀಲಿ.ಒಳಹರಿವು_type                         = context.src1->type;
        ಕೀಲಿ.output_type                        = context.dst->type;
        ಕೀಲಿ.whcn                               = whcn;

        ಸ್ವಯಂಚಾಲಿತ it = conv2d_dw_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != conv2d_dw_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        ಸ್ವಯಂಚಾಲಿತ ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ = [&](ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * prefix, ggml_type type) {
            std::string s_ಪೂರ್ವಪ್ರತ್ಯಯ = prefix;
            ಆದರೆ (type == GGML_TYPE_F32) {
                ನಿರೂಪಿಸುತ್ತದೆ.push_back(s_ಪೂರ್ವಪ್ರತ್ಯಯ + "=f32");
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (type == GGML_TYPE_F16) {
                ನಿರೂಪಿಸುತ್ತದೆ.push_back(s_ಪೂರ್ವಪ್ರತ್ಯಯ + "=f16");
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ("WEIGHT_TYPE", ಕೀಲಿ.ತೂಕ_type);
        ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ("INPUT_TYPE", ಕೀಲಿ.ಒಳಹರಿವು_type);
        ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ("OUTPUT_TYPE", ಕೀಲಿ.output_type);

        ಆದರೆ (whcn) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("WHCN");
        }
        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_conv2d_dw, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[ಕೀಲಿ] = pipeline;
        ಹಿಂತಿರುಗಿ conv2d_dw_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_im2col_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.ಒಳಹರಿವು_type                      = context.src1->type;
        ಕೀಲಿ.output_type                     = context.dst->type;

        ಸ್ವಯಂಚಾಲಿತ it = im2col_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != im2col_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "im2col";

        ಸ್ವಯಂಚಾಲಿತ ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ = [&](ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * prefix, ggml_type type) {
            std::string s_ಪೂರ್ವಪ್ರತ್ಯಯ = prefix;
            ಆದರೆ (type == GGML_TYPE_F32) {
                ನಿರೂಪಿಸುತ್ತದೆ.push_back(s_ಪೂರ್ವಪ್ರತ್ಯಯ + "=f32");
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (type == GGML_TYPE_F16) {
                ನಿರೂಪಿಸುತ್ತದೆ.push_back(s_ಪೂರ್ವಪ್ರತ್ಯಯ + "=f16");
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ("INPUT_TYPE", ಕೀಲಿ.ಒಳಹರಿವು_type);
        ತಳ್ಳು_type_ನಿರೂಪಿಸುತ್ತದೆ("OUTPUT_TYPE", ಕೀಲಿ.output_type);

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_im2col, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[ಕೀಲಿ]    = pipeline;
        ಹಿಂತಿರುಗಿ im2col_pipelines[ಕೀಲಿ];
    }

    webgpu_pipeline get_upscale_pipeline(ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context & context) {
        ಸ್ಥಿರಾಂಕ uint32_t ಬಹುಲಕ_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ಸ್ಥಿರಾಂಕ uint32_t ತಳ_ಬಹುಲಕ  = ಬಹುಲಕ_flags & 0xFFu;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ     antialias  = (ಬಹುಲಕ_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.ಒಳಹರಿವು_type                       = context.src0->type;
        ಕೀಲಿ.output_type                      = context.dst->type;
        ಕೀಲಿ.ತಳ_ಬಹುಲಕ                        = ತಳ_ಬಹುಲಕ;
        ಕೀಲಿ.antialias                        = antialias;

        ಸ್ವಯಂಚಾಲಿತ it = upscale_pipelines.find(ಕೀಲಿ);
        ಆದರೆ (it != upscale_pipelines.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }

        std::vector<std::string> ನಿರೂಪಿಸುತ್ತದೆ;
        std::string              variant = "upscale";

        ಆದರೆ (ಕೀಲಿ.ಒಳಹರಿವು_type == GGML_TYPE_F16) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("SRC_F16");
            variant += "_src_f16";
        } ಇಲ್ಲದಿದ್ದರೆ {
            variant += "_src_f32";
        }

        ಆದರೆ (ಕೀಲಿ.output_type == GGML_TYPE_F16) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("DST_F16");
            variant += "_dst_f16";
        } ಇಲ್ಲದಿದ್ದರೆ {
            variant += "_dst_f32";
        }

        ಆಯ್ಕೆ (ತಳ_ಬಹುಲಕ) {
            ಸ್ಥಿತಿ GGML_SCALE_ಬಹುಲಕ_ಹತ್ತಿರದ:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("NEAREST");
                variant += "_nearest";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_SCALE_ಬಹುಲಕ_BILINEAR:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("BILINEAR");
                variant += "_bilinear";
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_SCALE_ಬಹುಲಕ_BICUBIC:
                ನಿರೂಪಿಸುತ್ತದೆ.push_back("BICUBIC");
                variant += "_bicubic";
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("Unsupported upscale mode");
        }

        ಆದರೆ (antialias) {
            ನಿರೂಪಿಸುತ್ತದೆ.push_back("ANTIALIAS");
            variant += "_aa";
        }

        ನಿರೂಪಿಸುತ್ತದೆ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ಗಾತ್ರ));

        ಸ್ವಯಂಚಾಲಿತ processed           = preprocessor.preprocess(wgsl_upscale, ನಿರೂಪಿಸುತ್ತದೆ);
        ಸ್ವಯಂಚಾಲಿತ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ಗಾತ್ರ       = context.max_wg_ಗಾತ್ರ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[ಕೀಲಿ]   = pipeline;
        ಹಿಂತಿರುಗಿ upscale_pipelines[ಕೀಲಿ];
    }

  ಖಾಸಗಿ:
    ಸ್ಥಿರ webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_ಸಂಕೇತ,
                                                       std::string    label) {
        wgpu::Shader_ಮೂಲ_WGSL shader_ಮೂಲ;
        shader_ಮೂಲ.code = shader_ಸಂಕೇತ.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.ಮುಂದಿನ_ಒಳ_Chain = &shader_ಮೂಲ;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.ನಮೂದು_ಬಿಂದು_2 = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = ಶೂನ್ಯಸೂಚಕ;  // nullptr means auto layout
        ಹಿಂತಿರುಗಿ { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#ಆದರೆ_ಅಂತ್ಯ  // GGML_WEBGPU_SHADER_LIB_HPP
