#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_SHADER_LIB_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_SHADER_LIB_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-wgsl-shaders.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "pre_wgsl.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <webgpu/webgpu_cpp.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_F16_ਮਾਪ_ਬਾਈਟਾਂ                   2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_F32_ਮਾਪ_ਬਾਈਟਾਂ                   4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_I32_ਮਾਪ_ਬਾਈਟਾਂ                   4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_ਮਾਪ     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_KV_SEQ_PAD                       256u

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_WEBGPU_ARGSORT_ਮਿਲਾਓ_MAX_WG_ਮਾਪ 512u

// Matrix multiplication parameters

// Register tiling parameters
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_TILE_M           4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_TILE_N           4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_WG_ਮਾਪ_M        8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_WG_ਮਾਪ_N        8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_SUBGROUP_ਮੈਟ੍ਰਿਕਸ_M     4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_SUBGROUP_ਮੈਟ੍ਰਿਕਸ_N     2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_VEC_WG_ਮਾਪ 256

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_ਪ੍ਰਤੀ_WG    4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_ਪ੍ਰਤੀ_WG 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_ਪ੍ਰਤੀ_WG      4

// default size for reg-tile matrix multiplication
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_MUL_MAT_WG_ਮਾਪ 256

// Same hash combine function as in boost
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_webgpu_hash_combine(size_t & ਬੀਜ, ਅਚਲ T & value) {
    ਬੀਜ ^= std::hash<T>{}(value) + 0x9e3779b9 + (ਬੀਜ << 6) + (ਬੀਜ >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
ਸਤਰਅੰਦਰ uintptr_t ggml_webgpu_tensor_addr(ਅਚਲ ggml_tensor * tensor) {
    ਅਚਲ ggml_tensor * ਅਧਾਰ_tensor = tensor->view_src ? tensor->view_src : tensor;
    ਵਾਪਸ (uintptr_t) ਅਧਾਰ_tensor->ਅੰਕੜੇ + tensor->view_offs;
}

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_webgpu_tensor_equal(ਅਚਲ ggml_tensor * a, ਅਚਲ ggml_tensor * b) {
    ਵਾਪਸ a->ਬਫਰ == b->ਬਫਰ && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

ਬਣਤਰ ggml_webgpu_shader_lib_context {
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * src2;
    ggml_tensor * src3;
    ggml_tensor * src4;
    ggml_tensor * src5;
    ggml_tensor * dst;
    ggml_tensor * dst_fuse;

    uint32_t    max_wg_ਮਾਪ;
    size_t      wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ       = 0;
    ਤਾਰਕਿਕ        supports_subgroups       = ਝੂਠ;
    ਤਾਰਕਿਕ        supports_subgroup_ਮੈਟ੍ਰਿਕਸ = ਝੂਠ;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_ਮਾਪ        = 0;
    uint32_t    max_subgroup_ਮਾਪ        = 0;
    ਤਾਰਕਿਕ        supports_dot_ਗੁਣਨਫਲ     = ਝੂਠ;
    std::string vendor;
};

ਬਣਤਰ webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<ਖਾਲੀ> context = ਖਾਲੀ_ਸੂਚਕ;
};

ਬਣਤਰ ggml_webgpu_generic_shader_decisions {
    uint32_t wg_ਮਾਪ = 0;
    ਤਾਰਕਿਕ     inplace = ਝੂਠ;
};

ਬਣਤਰ ggml_webgpu_get_ਪੰਕਤੀਆਂ_shader_decisions {
    uint32_t wg_ਮਾਪ    = 0;
    ਤਾਰਕਿਕ     vectorized = ਝੂਠ;
};

ਬਣਤਰ ggml_webgpu_ਦੁਆਧਾਰੀ_shader_decisions {
    uint32_t wg_ਮਾਪ     = 0;
    ਤਾਰਕਿਕ     inplace     = ਝੂਠ;
    ਤਾਰਕਿਕ     overlap     = ਝੂਠ;
    ਤਾਰਕਿਕ     src_overlap = ਝੂਠ;
};

ਬਣਤਰ ggml_webgpu_glu_shader_decisions {
    uint32_t wg_ਮਾਪ     = 0;
    ਤਾਰਕਿਕ     src_overlap = ਝੂਠ;
};

ਬਣਤਰ ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<ਖਾਲੀ> decisions;
};

ਬਣਤਰ ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t ਖੰਡ_ਮਾਪ_3;
    uint32_t ਟੋਕਨ_ਪ੍ਰਤੀ_wg;
};

ਬਣਤਰ ggml_webgpu_ssm_ਜਾਂਚ_pipeline_ਕੁੰਜੀ {
    ਪੂਰਨ     type;
    ਪੂਰਨ     d_state;
    uint8_t xbc_overlap;
    ਤਾਰਕਿਕ    a_overlap;
    ਤਾਰਕਿਕ    ids_overlap;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_ssm_ਜਾਂਚ_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

ਬਣਤਰ ggml_webgpu_ssm_ਜਾਂਚ_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_ssm_ਜਾਂਚ_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.d_state);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.xbc_overlap);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.a_overlap);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ids_overlap);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਬਣਤਰ ggml_webgpu_ssm_ਜਾਂਚ_shader_decisions {
    uint32_t wg_ਮਾਪ;
    uint32_t ਟੋਕਨ_ਪ੍ਰਤੀ_tile;
    uint8_t  xbc_overlap = 0;
    ਤਾਰਕਿਕ     a_overlap   = ਝੂਠ;
    ਤਾਰਕਿਕ     ids_overlap = ਝੂਠ;
};

/** Argsort **/

ਬਣਤਰ ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_ਮਾਪ;
    size_t   wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ;
    int32_t  ਕ੍ਰਮ;
};

/** Set Rows **/

ਬਣਤਰ ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ {
    ਪੂਰਨ src0_type;
    ਪੂਰਨ dst_type;
    ਪੂਰਨ vec4;
    ਪੂਰਨ i64_idx;
    ਪੂਰਨ pair_ਖੰਡਾਂ;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_ਖੰਡਾਂ == other.pair_ਖੰਡਾਂ;
    }
};

ਬਣਤਰ ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src0_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.dst_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.vec4);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.i64_idx);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.pair_ਖੰਡਾਂ);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਬਣਤਰ ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ_shader_decisions {
    ਤਾਰਕਿਕ     vec4;
    ਤਾਰਕਿਕ     i64_idx;
    ਤਾਰਕਿਕ     pair_ਖੰਡਾਂ;
    uint32_t wg_ਮਾਪ;
};

/** Set **/

ਬਣਤਰ ggml_webgpu_ਸੈੱਟ_pipeline_ਕੁੰਜੀ {
    ggml_type type;
    ਤਾਰਕਿਕ      inplace;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_ਸੈੱਟ_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ type == other.type && inplace == other.inplace;
    }
};

ਬਣਤਰ ggml_webgpu_ਸੈੱਟ_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_ਸੈੱਟ_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.inplace);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Get Rows **/

ਬਣਤਰ ggml_webgpu_get_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ {
    ggml_type src_type;
    ਪੂਰਨ       vectorized;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_get_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ src_type == other.src_type && vectorized == other.vectorized;
    }
};

ਬਣਤਰ ggml_webgpu_get_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_get_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.vectorized);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Row Norm **/

ਬਣਤਰ ggml_webgpu_ਪੰਕਤੀ_norm_pipeline_ਕੁੰਜੀ {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    ਤਾਰਕਿਕ      inplace;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_ਪੰਕਤੀ_norm_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

ਬਣਤਰ ggml_webgpu_ਪੰਕਤੀ_norm_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_ਪੰਕਤੀ_norm_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.op);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.dst_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.inplace);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** RMS_NORM + MUL **/

ਬਣਤਰ ggml_webgpu_rms_norm_mul_pipeline_ਕੁੰਜੀ {
    ਤਾਰਕਿਕ inplace;      // rn_src == dst
    ਤਾਰਕਿਕ overlap;      // mul_src == dst
    ਤਾਰਕਿਕ src_overlap;  // rn_src binding overlaps mul_src binding

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_rms_norm_mul_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

ਬਣਤਰ ggml_webgpu_rms_norm_mul_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_rms_norm_mul_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.inplace);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.overlap);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src_overlap);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਬਣਤਰ ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_ਮਾਪ     = 0;
    ਤਾਰਕਿਕ     inplace     = ਝੂਠ;
    ਤਾਰਕਿਕ     overlap     = ਝੂਠ;
    ਤਾਰਕਿਕ     src_overlap = ਝੂਠ;
};

/** Pad **/
ਬਣਤਰ ggml_webgpu_pad_pipeline_ਕੁੰਜੀ {
    ਤਾਰਕਿਕ circular;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_pad_pipeline_ਕੁੰਜੀ & other) ਅਚਲ { ਵਾਪਸ circular == other.circular; }
};

ਬਣਤਰ ggml_webgpu_pad_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_pad_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.circular);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Solve Tri **/
ਬਣਤਰ ggml_webgpu_solve_tri_pipeline_ਕੁੰਜੀ {
    ਪੂਰਨ type;
    ਪੂਰਨ n;
    ਪੂਰਨ k;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_solve_tri_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ type == other.type && n == other.n && k == other.k;
    }
};

ਬਣਤਰ ggml_webgpu_solve_tri_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_solve_tri_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.n);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.k);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** SSM Conv **/
ਬਣਤਰ ggml_webgpu_ssm_conv_pipeline_ਕੁੰਜੀ {
    ਪੂਰਨ type;
    ਪੂਰਨ vectorized;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_ssm_conv_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
ਬਣਤਰ ggml_webgpu_conv2d_pipeline_ਕੁੰਜੀ {
    ggml_type ਭਾਰ_type;
    ggml_type ਨਿਵੇਸ਼_type;
    ggml_type output_type;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_conv2d_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ ਭਾਰ_type == other.ਭਾਰ_type && ਨਿਵੇਸ਼_type == other.ਨਿਵੇਸ਼_type && output_type == other.output_type;
    }
};

ਬਣਤਰ ggml_webgpu_conv2d_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_conv2d_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਭਾਰ_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਨਿਵੇਸ਼_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.output_type);
        ਵਾਪਸ ਬੀਜ;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
ਬਣਤਰ ggml_webgpu_conv2d_dw_pipeline_ਕੁੰਜੀ {
    ggml_type ਭਾਰ_type;
    ggml_type ਨਿਵੇਸ਼_type;
    ggml_type output_type;
    ਤਾਰਕਿਕ      whcn;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_conv2d_dw_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ ਭਾਰ_type == other.ਭਾਰ_type && ਨਿਵੇਸ਼_type == other.ਨਿਵੇਸ਼_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

ਬਣਤਰ ggml_webgpu_conv2d_dw_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_conv2d_dw_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਭਾਰ_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਨਿਵੇਸ਼_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.output_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.whcn);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Im2Col **/
ਬਣਤਰ ggml_webgpu_im2col_pipeline_ਕੁੰਜੀ {
    ggml_type ਨਿਵੇਸ਼_type;
    ggml_type output_type;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_im2col_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ ਨਿਵੇਸ਼_type == other.ਨਿਵੇਸ਼_type && output_type == other.output_type;
    }
};

ਬਣਤਰ ggml_webgpu_im2col_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_im2col_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਨਿਵੇਸ਼_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.output_type);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Gated Delta Net **/
ਬਣਤਰ ggml_webgpu_gated_delta_net_pipeline_ਕੁੰਜੀ {
    ਪੂਰਨ  type;
    ਪੂਰਨ  s_v;
    ਪੂਰਨ  kda;
    ਤਾਰਕਿਕ fused_cache;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_gated_delta_net_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

ਬਣਤਰ ggml_webgpu_gated_delta_net_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_gated_delta_net_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.s_v);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.kda);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਬਣਤਰ ggml_webgpu_ssm_conv_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_ssm_conv_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.vectorized);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Scale **/

ਬਣਤਰ ggml_webgpu_scale_pipeline_ਕੁੰਜੀ {
    ਪੂਰਨ inplace;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_scale_pipeline_ਕੁੰਜੀ & other) ਅਚਲ { ਵਾਪਸ inplace == other.inplace; }
};

ਬਣਤਰ ggml_webgpu_scale_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_scale_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.inplace);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Upscale **/

ਬਣਤਰ ggml_webgpu_upscale_pipeline_ਕੁੰਜੀ {
    ggml_type ਨਿਵੇਸ਼_type;
    ggml_type output_type;
    uint32_t  ਅਧਾਰ_ਬਹੁਲਕ;
    ਤਾਰਕਿਕ      antialias;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_upscale_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ ਨਿਵੇਸ਼_type == other.ਨਿਵੇਸ਼_type && output_type == other.output_type && ਅਧਾਰ_ਬਹੁਲਕ == other.ਅਧਾਰ_ਬਹੁਲਕ &&
               antialias == other.antialias;
    }
};

ਬਣਤਰ ggml_webgpu_upscale_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_upscale_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਨਿਵੇਸ਼_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.output_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਅਧਾਰ_ਬਹੁਲਕ);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.antialias);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Concat **/

ਬਣਤਰ ggml_webgpu_concat_pipeline_ਕੁੰਜੀ {
    ਪੂਰਨ  type;
    ਤਾਰਕਿਕ src_overlap;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_concat_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ type == other.type && src_overlap == other.src_overlap;
    }
};

ਬਣਤਰ ggml_webgpu_concat_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_concat_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src_overlap);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Repeat **/

ਬਣਤਰ ggml_webgpu_repeat_pipeline_ਕੁੰਜੀ {
    ਪੂਰਨ type;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_repeat_pipeline_ਕੁੰਜੀ & other) ਅਚਲ { ਵਾਪਸ type == other.type; }
};

ਬਣਤਰ ggml_webgpu_repeat_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_repeat_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Binary **/

ਬਣਤਰ ggml_webgpu_ਦੁਆਧਾਰੀ_pipeline_ਕੁੰਜੀ {
    ਪੂਰਨ  type;
    ਪੂਰਨ  op;
    ਤਾਰਕਿਕ inplace;
    ਤਾਰਕਿਕ overlap;
    ਤਾਰਕਿਕ src_overlap;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_ਦੁਆਧਾਰੀ_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

ਬਣਤਰ ggml_webgpu_ਦੁਆਧਾਰੀ_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_ਦੁਆਧਾਰੀ_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.op);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.inplace);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.overlap);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src_overlap);
        ਵਾਪਸ ਬੀਜ;
    }
};

/* Add_Id */

ਬਣਤਰ ggml_webgpu_ਜੋੜੋ_id_pipeline_ਕੁੰਜੀ {
    ਤਾਰਕਿਕ inplace;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_ਜੋੜੋ_id_pipeline_ਕੁੰਜੀ & other) ਅਚਲ { ਵਾਪਸ inplace == other.inplace; }
};

ਬਣਤਰ ggml_webgpu_ਜੋੜੋ_id_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_ਜੋੜੋ_id_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.inplace);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Unary **/

ਬਣਤਰ ggml_webgpu_unary_pipeline_ਕੁੰਜੀ {
    ਪੂਰਨ           type;
    ਪੂਰਨ           op;
    ਤਾਰਕਿਕ          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    ਤਾਰਕਿਕ          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_unary_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

ਬਣਤਰ ggml_webgpu_unary_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_unary_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.op);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.is_unary);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.inplace);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ttype);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** FlashAttention */

ਬਣਤਰ ggml_webgpu_flash_attn_common_pipeline_ਕੁੰਜੀ {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  ਸਿਰ_dim_qk;
    uint32_t  ਸਿਰ_dim_v;
    ਤਾਰਕਿਕ      k_ਸਿੱਧਾ;
    ਤਾਰਕਿਕ      v_ਸਿੱਧਾ;
    ਤਾਰਕਿਕ      kv_overlap;
    ਤਾਰਕਿਕ      has_ਨਕਾਬ;
    ਤਾਰਕਿਕ      has_sinks;
    ਤਾਰਕਿਕ      ਵਰਤਦਾ_logit_softcap;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_flash_attn_common_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && ਸਿਰ_dim_qk == other.ਸਿਰ_dim_qk && ਸਿਰ_dim_v == other.ਸਿਰ_dim_v &&
               k_ਸਿੱਧਾ == other.k_ਸਿੱਧਾ && v_ਸਿੱਧਾ == other.v_ਸਿੱਧਾ && kv_overlap == other.kv_overlap &&
               has_ਨਕਾਬ == other.has_ਨਕਾਬ && has_sinks == other.has_sinks &&
               ਵਰਤਦਾ_logit_softcap == other.ਵਰਤਦਾ_logit_softcap;
    }
};

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_webgpu_flash_attn_hash_common_pipeline_ਕੁੰਜੀ(size_t &                                           ਬੀਜ,
                                                            ਅਚਲ ggml_webgpu_flash_attn_common_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) {
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.q_type);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.k_type);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.v_type);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.dst_type);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਸਿਰ_dim_qk);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਸਿਰ_dim_v);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.k_ਸਿੱਧਾ);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.v_ਸਿੱਧਾ);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.kv_overlap);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.has_ਨਕਾਬ);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.has_sinks);
    ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਵਰਤਦਾ_logit_softcap);
}

ਬਣਤਰ ggml_webgpu_flash_attn_vec_pipeline_ਕੁੰਜੀ {
    ggml_webgpu_flash_attn_common_pipeline_ਕੁੰਜੀ common;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_flash_attn_vec_pipeline_ਕੁੰਜੀ & other) ਅਚਲ { ਵਾਪਸ common == other.common; }
};

ਬਣਤਰ ggml_webgpu_flash_attn_vec_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_flash_attn_vec_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ਕੁੰਜੀ(ਬੀਜ, ਕੁੰਜੀ.common);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਬਣਤਰ ggml_webgpu_flash_attn_pipeline_ਕੁੰਜੀ {
    ggml_webgpu_flash_attn_common_pipeline_ਕੁੰਜੀ common;
    ਤਾਰਕਿਕ                                       use_sg_ਮੈਟ੍ਰਿਕਸ;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_flash_attn_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ common == other.common && use_sg_ਮੈਟ੍ਰਿਕਸ == other.use_sg_ਮੈਟ੍ਰਿਕਸ;
    }
};

ਬਣਤਰ ggml_webgpu_flash_attn_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_flash_attn_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_ਕੁੰਜੀ(ਬੀਜ, ਕੁੰਜੀ.common);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.use_sg_ਮੈਟ੍ਰਿਕਸ);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਬਣਤਰ ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_ਮਾਪ = 0;
};

ਬਣਤਰ ggml_webgpu_flash_attn_decisions {
    ਤਾਰਕਿਕ     use_sg_ਮੈਟ੍ਰਿਕਸ = ਝੂਠ;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_ਮਾਪ       = 0;
};

ਸਤਰਅੰਦਰ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ਚੌੜਾਈ = 4u;
ਸਤਰਅੰਦਰ constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

ਸਤਰਅੰਦਰ size_t ggml_webgpu_flash_attn_tensor_ਵਿੱਥ(ਅਚਲ ggml_tensor * tensor) {
    constexpr uintptr_t ptr_ਅਧਾਰ_addr = 0x1000u;
    ਅਚਲ ggml_tensor * ਅਧਾਰ          = tensor->view_src != ਖਾਲੀ_ਸੂਚਕ ? tensor->view_src : tensor;
    ਵਾਪਸ reinterpret_cast<uintptr_t>(ਅਧਾਰ->ਅੰਕੜੇ) - ptr_ਅਧਾਰ_addr + tensor->view_offs;
}

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_webgpu_flash_attn_float_vec4_aligned(ਅਚਲ ggml_tensor * K, size_t storage_ਵਿੱਥ_alignment) {
    ਅਚਲ uint32_t ਵਿੱਥ_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_ਵਿੱਥ(K) & (storage_ਵਿੱਥ_alignment - 1)) /
                    ggml_type_ਮਾਪ(K->type));
    ਵਾਪਸ ਵਿੱਥ_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ਚੌੜਾਈ == 0u;
}

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_webgpu_flash_attn_float_vec4_aligned(ਅਚਲ ggml_tensor * K,
                                                      ਅਚਲ ggml_tensor * V,
                                                      size_t              storage_ਵਿੱਥ_alignment) {
    ਵਾਪਸ ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_ਵਿੱਥ_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_ਵਿੱਥ_alignment);
}

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_webgpu_flash_attn_k_ਸਿੱਧਾ(ਅਚਲ ggml_tensor * Q, ਅਚਲ ggml_tensor * K, uint32_t kv_ਸਿੱਧਾ_align) {
    ਵਾਪਸ (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_ਸਿੱਧਾ_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_webgpu_flash_attn_v_ਸਿੱਧਾ(ਅਚਲ ggml_tensor * Q, ਅਚਲ ggml_tensor * V, uint32_t kv_ਸਿੱਧਾ_align) {
    ਵਾਪਸ ggml_webgpu_flash_attn_k_ਸਿੱਧਾ(Q, V, kv_ਸਿੱਧਾ_align);
}

ਸਤਰਅੰਦਰ ggml_webgpu_flash_attn_common_pipeline_ਕੁੰਜੀ ggml_webgpu_flash_attn_make_common_pipeline_ਕੁੰਜੀ(
    ਅਚਲ ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_ਸਿੱਧਾ_align,
    ਤਾਰਕਿਕ                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
    ਕੁੰਜੀ.q_type                                     = context.src0->type;
    ਕੁੰਜੀ.k_type                                     = context.src1->type;
    ਕੁੰਜੀ.v_type                                     = context.src2->type;
    ਕੁੰਜੀ.dst_type                                   = context.dst->type;
    ਕੁੰਜੀ.ਸਿਰ_dim_qk                                = (uint32_t) context.src0->ne[0];
    ਕੁੰਜੀ.ਸਿਰ_dim_v                                 = (uint32_t) context.src2->ne[0];
    ਕੁੰਜੀ.k_ਸਿੱਧਾ           = ggml_webgpu_flash_attn_k_ਸਿੱਧਾ(context.src0, context.src1, kv_ਸਿੱਧਾ_align);
    ਕੁੰਜੀ.v_ਸਿੱਧਾ           = ggml_webgpu_flash_attn_v_ਸਿੱਧਾ(context.src0, context.src2, kv_ਸਿੱਧਾ_align);
    ਕੁੰਜੀ.kv_overlap         = kv_overlap;
    ਕੁੰਜੀ.has_ਨਕਾਬ           = context.src3 != ਖਾਲੀ_ਸੂਚਕ;
    ਕੁੰਜੀ.has_sinks          = context.src4 != ਖਾਲੀ_ਸੂਚਕ;
    ਕੁੰਜੀ.ਵਰਤਦਾ_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    ਵਾਪਸ ਕੁੰਜੀ;
}

ਸਤਰਅੰਦਰ std::vector<std::string> ggml_webgpu_flash_attn_common_ਨਿਰਧਾਰਤ_ਕਰਦਾ(
    ਅਚਲ ggml_webgpu_flash_attn_common_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_ਮਾਪ) {
    std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;

    ਚੋਣ (ਕੁੰਜੀ.k_type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("K_F32");
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("K_F16");
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("K_Q4_0");
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("K_Q8_0");
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(ਕੁੰਜੀ.k_type);

    ਚੋਣ (ਕੁੰਜੀ.v_type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("V_F32");
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("V_F16");
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("V_Q4_0");
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("V_Q8_0");
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(ਕੁੰਜੀ.v_type);

    ਚੋਣ (ਕੁੰਜੀ.q_type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("Q_F32");
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("Q_F16");
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(ਕੁੰਜੀ.q_type);

    ਚੋਣ (ਕੁੰਜੀ.dst_type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F32");
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F16");
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(ਕੁੰਜੀ.dst_type);

    ਜੇ (ਕੁੰਜੀ.has_ਨਕਾਬ) {
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MASK");
        variant += "_mask";
    }
    ਜੇ (ਕੁੰਜੀ.has_sinks) {
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SINKS");
        variant += "_sinks";
    }
    ਜੇ (ਕੁੰਜੀ.ਵਰਤਦਾ_logit_softcap) {
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    ਜੇ (ਕੁੰਜੀ.k_ਸਿੱਧਾ) {
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    ਜੇ (ਕੁੰਜੀ.v_ਸਿੱਧਾ) {
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    ਜੇ (ਕੁੰਜੀ.kv_overlap) {
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("HEAD_DIM_QK=") + std::to_string(ਕੁੰਜੀ.ਸਿਰ_dim_qk));
    variant += std::string("_hsqk") + std::to_string(ਕੁੰਜੀ.ਸਿਰ_dim_qk);

    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("HEAD_DIM_V=") + std::to_string(ਕੁੰਜੀ.ਸਿਰ_dim_v));
    variant += std::string("_hsv") + std::to_string(ਕੁੰਜੀ.ਸਿਰ_dim_v);

    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ਮਾਪ));

    ਜੇ (ggml_is_quantized(ਕੁੰਜੀ.k_type) || ggml_is_quantized(ਕੁੰਜੀ.v_type)) {
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("U32_DEQUANT_HELPERS");
        ਜੇ (ggml_is_quantized(ਕੁੰਜੀ.k_type)) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("LOADERS_QUANTIZED_K");
        }
        ਜੇ (ggml_is_quantized(ਕੁੰਜੀ.v_type)) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("LOADERS_QUANTIZED_V");
        }
    }

    ਵਾਪਸ ਨਿਰਧਾਰਤ_ਕਰਦਾ;
}

ਬਣਤਰ ggml_webgpu_flash_attn_vec_reduce_pipeline_ਕੁੰਜੀ {
    uint32_t  ਸਿਰ_dim_v;
    uint32_t  wg_ਮਾਪ;
    ggml_type dst_type;
};

ਬਣਤਰ ggml_webgpu_flash_attn_vec_reduce_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_flash_attn_vec_reduce_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਸਿਰ_dim_v);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.wg_ਮਾਪ);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.dst_type);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_flash_attn_vec_reduce_pipeline_ਕੁੰਜੀ & lhs,
                       ਅਚਲ ggml_webgpu_flash_attn_vec_reduce_pipeline_ਕੁੰਜੀ & rhs) {
    ਵਾਪਸ lhs.ਸਿਰ_dim_v == rhs.ਸਿਰ_dim_v && lhs.wg_ਮਾਪ == rhs.wg_ਮਾਪ && lhs.dst_type == rhs.dst_type;
}

ਬਣਤਰ ggml_webgpu_flash_attn_blk_pipeline_ਕੁੰਜੀ {
    uint32_t kv_tile;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_flash_attn_blk_pipeline_ਕੁੰਜੀ & other) ਅਚਲ { ਵਾਪਸ kv_tile == other.kv_tile; }
};

ਬਣਤਰ ggml_webgpu_flash_attn_blk_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_flash_attn_blk_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.kv_tile);
        ਵਾਪਸ ਬੀਜ;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
ਸਤਰਅੰਦਰ size_t ggml_webgpu_flash_attn_wg_mem_ਬਾਈਟਾਂ(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t ਸਿਰ_dim_qk,
                                                  uint32_t ਸਿਰ_dim_v,
                                                  ਤਾਰਕਿਕ     has_ਨਕਾਬ,
                                                  ਤਾਰਕਿਕ     kv_ਸਿੱਧਾ) {
    ਅਚਲ uint32_t max_ਸਿਰ_dim = std::max(ਸਿਰ_dim_qk, ਸਿਰ_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * ਸਿਰ_dim_qk;        // q_shmem
    ਜੇ (!kv_ਸਿੱਧਾ) {
        f32_elems += kv_tile * max_ਸਿਰ_dim;  // kv_shmem
    }
    f32_elems += q_tile * ਸਿਰ_dim_v;         // o_shmem
    ਜੇ (has_ਨਕਾਬ) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    ਵਾਪਸ f16_elems * GGML_WEBGPU_F16_ਮਾਪ_ਬਾਈਟਾਂ + f32_elems * GGML_WEBGPU_F32_ਮਾਪ_ਬਾਈਟਾਂ;
}

ਸਤਰਅੰਦਰ uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t ਸਿਰ_dim_qk,
                                                   uint32_t ਸਿਰ_dim_v,
                                                   ਤਾਰਕਿਕ     has_ਨਕਾਬ,
                                                   ਤਾਰਕਿਕ     kv_ਸਿੱਧਾ) {
    ਅਚਲ size_t ਅਧਾਰ_q_ਬਾਈਟਾਂ =
        ggml_webgpu_flash_attn_wg_mem_ਬਾਈਟਾਂ(q_tile, 0, ਸਿਰ_dim_qk, ਸਿਰ_dim_v, has_ਨਕਾਬ, kv_ਸਿੱਧਾ);
    ਜੇ (ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ <= ਅਧਾਰ_q_ਬਾਈਟਾਂ) {
        ਵਾਪਸ 0;
    }
    ਅਚਲ size_t one_kv_ਬਾਈਟਾਂ =
        ggml_webgpu_flash_attn_wg_mem_ਬਾਈਟਾਂ(q_tile, 1, ਸਿਰ_dim_qk, ਸਿਰ_dim_v, has_ਨਕਾਬ, kv_ਸਿੱਧਾ);
    ਅਚਲ size_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_kv = one_kv_ਬਾਈਟਾਂ - ਅਧਾਰ_q_ਬਾਈਟਾਂ;
    ਜੇ (ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_kv == 0) {
        ਵਾਪਸ 0;
    }
    ਅਚਲ size_t max_kv_tile = (ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ - ਅਧਾਰ_q_ਬਾਈਟਾਂ) / ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_kv;
    ਵਾਪਸ (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

ਸਤਰਅੰਦਰ uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ,
                                                       uint32_t ਸਿਰ_dim_qk,
                                                       uint32_t ਸਿਰ_dim_v,
                                                       ਤਾਰਕਿਕ     has_ਨਕਾਬ,
                                                       ਤਾਰਕਿਕ     kv_ਸਿੱਧਾ) {
    ਅਚਲ uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ, 1u, 1u, ਸਿਰ_dim_qk, ਸਿਰ_dim_v, has_ਨਕਾਬ, kv_ਸਿੱਧਾ);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    ਜੇ (kv_ਸਿੱਧਾ) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        ਜਦੋਂ (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    ਵਾਪਸ kv_tile;
}

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_webgpu_flash_attn_can_use_subgroup_ਮੈਟ੍ਰਿਕਸ_ਰਸਤਾ(ਤਾਰਕਿਕ                supports_subgroup_ਮੈਟ੍ਰਿਕਸ,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                ਅਚਲ ggml_tensor * Q,
                                                                ਅਚਲ ggml_tensor * V) {
    ਵਾਪਸ supports_subgroup_ਮੈਟ੍ਰਿਕਸ && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

ਬਣਤਰ ggml_webgpu_mul_mat_vec_pipeline_ਕੁੰਜੀ {
    ggml_type src0_type;
    ggml_type src1_type;
    ਪੂਰਨ       vectorized;
    uint32_t  ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ;
    ਤਾਰਕਿਕ      use_mmvq;
    ਤਾਰਕਿਕ      src_overlap;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_mul_mat_vec_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ == other.ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

ਬਣਤਰ ggml_webgpu_mul_mat_vec_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_mul_mat_vec_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src0_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src1_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.vectorized);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.use_mmvq);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src_overlap);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਬਣਤਰ ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_ਮਾਪ;
    uint32_t outputs_ਪ੍ਰਤੀ_wg;
    uint32_t vec_ਮਾਪ;
    ਤਾਰਕਿਕ     src_overlap = ਝੂਠ;
};

ਬਣਤਰ ggml_webgpu_quantize_q8_pipeline_ਕੁੰਜੀ {
    ggml_type src0_type;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_quantize_q8_pipeline_ਕੁੰਜੀ & other) ਅਚਲ { ਵਾਪਸ src0_type == other.src0_type; }
};

ਬਣਤਰ ggml_webgpu_quantize_q8_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_quantize_q8_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src0_type);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਬਣਤਰ ggml_webgpu_mul_mat_pipeline_ਕੁੰਜੀ {
    ggml_type src0_type;
    ggml_type src1_type;
    ਪੂਰਨ       vectorized;
    ਪੂਰਨ       use_subgroup_ਮੈਟ੍ਰਿਕਸ;
    ਤਾਰਕਿਕ      src_overlap;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_mul_mat_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_ਮੈਟ੍ਰਿਕਸ == other.use_subgroup_ਮੈਟ੍ਰਿਕਸ && src_overlap == other.src_overlap;
    }
};

ਬਣਤਰ ggml_webgpu_mul_mat_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_mul_mat_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src0_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src1_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.vectorized);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.use_subgroup_ਮੈਟ੍ਰਿਕਸ);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src_overlap);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਬਣਤਰ ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_ਮਾਪ_m;
    uint32_t wg_ਮਾਪ_n;
    uint32_t wg_ਮਾਪ;
    uint32_t outputs_ਪ੍ਰਤੀ_wg;
    ਪੂਰਨ      use_subgroup_ਮੈਟ੍ਰਿਕਸ;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_ਮੈਟ੍ਰਿਕਸ_m;
    uint32_t subgroup_ਮੈਟ੍ਰਿਕਸ_n;

    uint32_t mul_mat_wg_ਮਾਪ;
    ਤਾਰਕਿਕ     src_overlap = ਝੂਠ;
};

/** MUL_MAT_ID **/

ਬਣਤਰ ggml_webgpu_mul_mat_id_gather_pipeline_ਕੁੰਜੀ {
    uint32_t n_experts;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_mul_mat_id_gather_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ n_experts == other.n_experts;
    }
};

ਬਣਤਰ ggml_webgpu_mul_mat_id_gather_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_mul_mat_id_gather_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.n_experts);
        ਵਾਪਸ ਬੀਜ;
    }
};

ਬਣਤਰ ggml_webgpu_mul_mat_id_pipeline_ਕੁੰਜੀ {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ;
    ਪੂਰਨ       vectorized;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_mul_mat_id_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ == other.ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ && vectorized == other.vectorized;
    }
};

ਬਣਤਰ ggml_webgpu_mul_mat_id_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_mul_mat_id_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src0_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src1_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.n_experts);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.vectorized);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Cpy **/

ਬਣਤਰ ggml_webgpu_cpy_pipeline_ਕੁੰਜੀ {
    ggml_type src_type;
    ggml_type dst_type;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_cpy_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ src_type == other.src_type && dst_type == other.dst_type;
    }
};

ਬਣਤਰ ggml_webgpu_cpy_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_cpy_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.dst_type);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Glu **/

ਬਣਤਰ ggml_webgpu_glu_pipeline_ਕੁੰਜੀ {
    ggml_glu_op glu_op;
    ggml_type   type;
    ਤਾਰਕਿਕ        ਵੰਡੋ;
    ਤਾਰਕਿਕ        src_overlap;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_glu_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ glu_op == other.glu_op && type == other.type && ਵੰਡੋ == other.ਵੰਡੋ && src_overlap == other.src_overlap;
    }
};

ਬਣਤਰ ggml_webgpu_glu_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_glu_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.glu_op);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਵੰਡੋ);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.src_overlap);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** Rope **/

ਬਣਤਰ ggml_webgpu_rope_pipeline_ਕੁੰਜੀ {
    ggml_type type;
    ਤਾਰਕਿਕ      inplace;
    ਤਾਰਕਿਕ      has_ff;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_rope_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

ਬਣਤਰ ggml_webgpu_rope_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_rope_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.inplace);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.has_ff);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** SoftMax **/

ਬਣਤਰ ggml_webgpu_soft_max_pipeline_ਕੁੰਜੀ {
    ggml_type ਨਕਾਬ_type;
    ਤਾਰਕਿਕ      has_ਨਕਾਬ;
    ਤਾਰਕਿਕ      has_sink;
    ਤਾਰਕਿਕ      inplace;

    ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ ggml_webgpu_soft_max_pipeline_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ ਨਕਾਬ_type == other.ਨਕਾਬ_type && has_ਨਕਾਬ == other.has_ਨਕਾਬ && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

ਬਣਤਰ ggml_webgpu_soft_max_pipeline_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ggml_webgpu_soft_max_pipeline_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t ਬੀਜ = 0;
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.ਨਕਾਬ_type);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.has_ਨਕਾਬ);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.has_sink);
        ggml_webgpu_hash_combine(ਬੀਜ, ਕੁੰਜੀ.inplace);
        ਵਾਪਸ ਬੀਜ;
    }
};

/** MMVQ **/

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_webgpu_can_use_mmvq(ਅਚਲ ggml_tensor * src0,
                                     ਅਚਲ ggml_tensor * src1,
                                     ਤਾਰਕਿਕ                supports_dot_ਗੁਣਨਫਲ,
                                     ਅਚਲ std::string & vendor) {
    ਜੇ (src1->ne[1] <= 4) {
        ਤਾਰਕਿਕ supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        ਜੇ (supports_dp4a && supports_dot_ਗੁਣਨਫਲ) {
            ਚੋਣ (src1->type) {
                ਹਾਲਤ GGML_TYPE_F32:
                    ਚੋਣ (src0->type) {
                        ਹਾਲਤ GGML_TYPE_Q1_0:
                        ਹਾਲਤ GGML_TYPE_Q4_0:
                        ਹਾਲਤ GGML_TYPE_Q4_1:
                        ਹਾਲਤ GGML_TYPE_Q5_0:
                        ਹਾਲਤ GGML_TYPE_Q5_1:
                        ਹਾਲਤ GGML_TYPE_Q8_0:
                        ਹਾਲਤ GGML_TYPE_MXFP4:
                        ਹਾਲਤ GGML_TYPE_Q2_K:
                        ਹਾਲਤ GGML_TYPE_Q3_K:
                        ਹਾਲਤ GGML_TYPE_Q4_K:
                        ਹਾਲਤ GGML_TYPE_Q5_K:
                        ਹਾਲਤ GGML_TYPE_Q6_K:
                            ਵਾਪਸ src0->ne[0] % 4 == 0;
                        ਮੂਲ:
                            ਰੋਕੋ;
                    }
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਰੋਕੋ;
            }
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

ਵਰਗ ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<ਪੂਰਨ, webgpu_pipeline> ਜੋੜ_ਪੰਕਤੀਆਂ_pipelines;       // key is fixed, no variants yet
    std::unordered_map<ਪੂਰਨ, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<ਪੂਰਨ, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<ਪੂਰਨ, webgpu_pipeline> argsort_ਮਿਲਾਓ_pipelines;  // key is order
    std::unordered_map<ਪੂਰਨ, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_ਪੰਕਤੀ_norm_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_ਪੰਕਤੀ_norm_pipeline_ਕੁੰਜੀ_hash>
        ਪੰਕਤੀ_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_get_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ_hash>
        get_ਪੰਕਤੀਆਂ_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_unary_pipeline_ਕੁੰਜੀ_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_scale_pipeline_ਕੁੰਜੀ_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_ਕੁੰਜੀ_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_ਕੁੰਜੀ_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_ਜਾਂਚ_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_ssm_ਜਾਂਚ_pipeline_ਕੁੰਜੀ_hash>
        ssm_ਜਾਂਚ_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_ਕੁੰਜੀ,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_ਕੁੰਜੀ_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_pad_pipeline_ਕੁੰਜੀ_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_ਦੁਆਧਾਰੀ_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_ਦੁਆਧਾਰੀ_pipeline_ਕੁੰਜੀ_hash>
        ਦੁਆਧਾਰੀ_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_ਜੋੜੋ_id_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_ਜੋੜੋ_id_pipeline_ਕੁੰਜੀ_hash>
        ਜੋੜੋ_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_concat_pipeline_ਕੁੰਜੀ_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_repeat_pipeline_ਕੁੰਜੀ_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_ਕੁੰਜੀ,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_ਕੁੰਜੀ_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_ਕੁੰਜੀ_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_ਕੁੰਜੀ,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_ਕੁੰਜੀ_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_ਕੁੰਜੀ,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_ਕੁੰਜੀ_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_ਕੁੰਜੀ_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_ਕੁੰਜੀ_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_ਕੁੰਜੀ_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_ਕੁੰਜੀ,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_ਕੁੰਜੀ_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_ਕੁੰਜੀ_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_ਕੁੰਜੀ_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ_hash>
        ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipelines;
    std::unordered_map<ggml_webgpu_ਸੈੱਟ_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_ਸੈੱਟ_pipeline_ਕੁੰਜੀ_hash> ਸੈੱਟ_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_cpy_pipeline_ਕੁੰਜੀ_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_glu_pipeline_ਕੁੰਜੀ_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_rope_pipeline_ਕੁੰਜੀ_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_ਕੁੰਜੀ_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_ਕੁੰਜੀ_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_ਕੁੰਜੀ_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_im2col_pipeline_ਕੁੰਜੀ_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_ਕੁੰਜੀ,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_ਕੁੰਜੀ_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_ਕੁੰਜੀ, webgpu_pipeline, ggml_webgpu_upscale_pipeline_ਕੁੰਜੀ_hash>
        upscale_pipelines;

  ਜਨਤਕ:
    ggml_webgpu_shader_lib(wgpu::Device device) { ਇਹ->device = device; }

    webgpu_pipeline get_ਜੋੜ_ਪੰਕਤੀਆਂ_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ਸਵੈਚਾਲਿਤ it = ਜੋੜ_ਪੰਕਤੀਆਂ_pipelines.find(1);
        ਜੇ (it != ਜੋੜ_ਪੰਕਤੀਆਂ_pipelines.end()) {
            ਵਾਪਸ it->second;
        }
        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed        = preprocessor.preprocess(wgsl_ਜੋੜ_ਪੰਕਤੀਆਂ, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਜੋੜ_ਪੰਕਤੀਆਂ_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        ਵਾਪਸ ਜੋੜ_ਪੰਕਤੀਆਂ_pipelines[1];
    }

    webgpu_pipeline get_ਪੰਕਤੀ_norm_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ਪੰਕਤੀ_norm_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.op                                = context.dst->op;
        ਕੁੰਜੀ.src_type                          = context.src0->type;
        ਕੁੰਜੀ.dst_type                          = context.dst->type;
        ਕੁੰਜੀ.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        ਸਵੈਚਾਲਿਤ it = ਪੰਕਤੀ_norm_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != ਪੰਕਤੀ_norm_pipelines.end()) {
            ਵਾਪਸ it->second;
        }
        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant;

        ਚੋਣ (ਕੁੰਜੀ.op) {
            ਹਾਲਤ GGML_OP_RMS_NORM:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("RMS_NORM");
                variant = "rms_norm";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_NORM:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("NORM");
                variant = "norm";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_L2_NORM:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("L2_NORM");
                variant = "l2_norm";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        ਜੇ (ਕੁੰਜੀ.inplace) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INPLACE");
            variant += "_inplace";
        }

        ਜੇ (ਕੁੰਜੀ.src_type == GGML_TYPE_F32) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_F32");
            variant += "_src_f32";
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.src_type == GGML_TYPE_F16) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_F16");
            variant += "_src_f16";
        }

        ਜੇ (ਕੁੰਜੀ.dst_type == GGML_TYPE_F32) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F32");
            variant += "_dst_f32";
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.dst_type == GGML_TYPE_F16) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F16");
            variant += "_dst_f16";
        }

        ਅਚਲ uint32_t ਪੰਕਤੀ_norm_wg_ਮਾਪ = 128u;
        uint32_t       wg_ਮਾਪ          = std::min(context.max_wg_ਮਾਪ, ਪੰਕਤੀ_norm_wg_ਮਾਪ);
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ਮਾਪ));
        ਸਵੈਚਾਲਿਤ processed                  = preprocessor.preprocess(wgsl_ਪੰਕਤੀ_norm, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ              = wg_ਮਾਪ;
        decisions->inplace              = ਕੁੰਜੀ.inplace;
        ਪੰਕਤੀ_norm_pipelines[ਕੁੰਜੀ]         = ggml_webgpu_create_pipeline(device, processed, variant);
        ਪੰਕਤੀ_norm_pipelines[ਕੁੰਜੀ].context = decisions;
        ਵਾਪਸ ਪੰਕਤੀ_norm_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_argmax_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ਤਾਰਕਿਕ vec4 = context.src0->ne[0] % 4 == 0;

        ਸਵੈਚਾਲਿਤ it = argmax_pipelines.find(vec4);
        ਜੇ (it != argmax_pipelines.end()) {
            ਵਾਪਸ it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));
        ਜੇ (vec4) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("VEC4");
            variant += "_vec4";
        }

        ਸਵੈਚਾਲਿਤ processed         = preprocessor.preprocess(wgsl_argmax, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        ਵਾਪਸ argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ਅਚਲ ਤਾਰਕਿਕ                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ       = {};
        ਕੁੰਜੀ.src0_type                               = context.src0->type;
        ਕੁੰਜੀ.dst_type                                = context.dst->type;
        ਕੁੰਜੀ.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        ਕੁੰਜੀ.i64_idx     = context.src1->type == GGML_TYPE_I64;
        ਕੁੰਜੀ.pair_ਖੰਡਾਂ = quantized && ((context.src0->ne[0] / ggml_blck_ਮਾਪ(context.dst->type)) % 2 == 0);

        ਸਵੈਚਾਲਿਤ it = ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "set_rows";

        ਚੋਣ (context.dst->type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F32");
                variant += "_dstf32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F16");
                variant += "_dstf16";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q8_0:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q4_0:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        ਜੇ (context.src0->type == GGML_TYPE_F16) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        ਜੇ (ਕੁੰਜੀ.vec4) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("VEC4");
            variant += "_vec4";
        }
        ਜੇ (ਕੁੰਜੀ.i64_idx) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("I64_IDX");
            variant += "_i64idx";
        }
        ਜੇ (ਕੁੰਜੀ.pair_ਖੰਡਾਂ) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਅਚਲ ਸਵੈਚਾਲਿਤ & shader_ਸਰੋਤ      = quantized ? wgsl_ਸੈੱਟ_ਪੰਕਤੀਆਂ_quant : wgsl_ਸੈੱਟ_ਪੰਕਤੀਆਂ;
        ਸਵੈਚਾਲਿਤ         processed          = preprocessor.preprocess(shader_ਸਰੋਤ, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ         decisions          = std::make_shared<ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ_shader_decisions>();
        decisions->vec4                 = ਕੁੰਜੀ.vec4;
        decisions->i64_idx              = ਕੁੰਜੀ.i64_idx;
        decisions->pair_ਖੰਡਾਂ          = ਕੁੰਜੀ.pair_ਖੰਡਾਂ;
        decisions->wg_ਮਾਪ              = context.max_wg_ਮਾਪ;
        ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipelines[ਕੁੰਜੀ]         = ggml_webgpu_create_pipeline(device, processed, variant);
        ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipelines[ਕੁੰਜੀ].context = decisions;
        ਵਾਪਸ ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_ਸੈੱਟ_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ਸੈੱਟ_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.type                         = context.dst->type;
        ਕੁੰਜੀ.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        ਸਵੈਚਾਲਿਤ it = ਸੈੱਟ_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != ਸੈੱਟ_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "set";

        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_I32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_I32");
                variant += "_i32";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for set shader");
        }

        ਜੇ (ਕੁੰਜੀ.inplace) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INPLACE");
            variant += "_inplace";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_ਸੈੱਟ, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        decisions->inplace       = ਕੁੰਜੀ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ਸੈੱਟ_pipelines[ਕੁੰਜੀ]       = pipeline;
        ਵਾਪਸ ਸੈੱਟ_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_cumsum_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ਸਵੈਚਾਲਿਤ it = cumsum_pipelines.find(1);
        ਜੇ (it != cumsum_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed      = preprocessor.preprocess(wgsl_cumsum, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        ਵਾਪਸ cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ਤਾਰਕਿਕ          is_ਸਿਖਰ_k = context.dst->op == GGML_OP_ਸਿਖਰ_K;
        // ascending order is 0, descending order is 1
        ਅਚਲ int32_t ਕ੍ਰਮ =
            is_ਸਿਖਰ_k ? (int32_t) GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        ਸਵੈਚਾਲਿਤ it = argsort_pipelines.find(ਕ੍ਰਮ);
        ਜੇ (it != argsort_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "argsort";
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("ORDER=") + std::to_string(ਕ੍ਰਮ));
        variant += std::string("_order") + std::to_string(ਕ੍ਰਮ);
        uint32_t wg_ਮਾਪ = 1;
        ਜਦੋਂ (wg_ਮਾਪ * 2 <= context.max_wg_ਮਾਪ &&
               wg_ਮਾਪ * GGML_WEBGPU_I32_ਮਾਪ_ਬਾਈਟਾਂ <= context.wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ / 2) {
            wg_ਮਾਪ *= 2;
        }
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ਮਾਪ));
        ਸਵੈਚਾਲਿਤ processed                   = preprocessor.preprocess(wgsl_argsort, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ               = wg_ਮਾਪ;
        argsort_pipelines[ਕ੍ਰਮ]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[ਕ੍ਰਮ].context = decisions;
        ਵਾਪਸ argsort_pipelines[ਕ੍ਰਮ];
    }

    webgpu_pipeline get_argsort_ਮਿਲਾਓ_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ਤਾਰਕਿਕ          is_ਸਿਖਰ_k = context.dst->op == GGML_OP_ਸਿਖਰ_K;
        // ascending order is 0, descending order is 1
        ਅਚਲ int32_t ਕ੍ਰਮ =
            is_ਸਿਖਰ_k ? (int32_t) GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        ਸਵੈਚਾਲਿਤ it = argsort_ਮਿਲਾਓ_pipelines.find(ਕ੍ਰਮ);
        ਜੇ (it != argsort_ਮਿਲਾਓ_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "argsort_merge";
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("ORDER=") + std::to_string(ਕ੍ਰਮ));
        variant += std::string("_order") + std::to_string(ਕ੍ਰਮ);
        uint32_t wg_ਮਾਪ = std::min(GGML_WEBGPU_ARGSORT_ਮਿਲਾਓ_MAX_WG_ਮਾਪ, context.max_wg_ਮਾਪ);
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed                 = preprocessor.preprocess(wgsl_argsort_ਮਿਲਾਓ, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        argsort_ਮਿਲਾਓ_pipelines[ਕ੍ਰਮ] = ggml_webgpu_create_pipeline(device, processed, variant);
        ਵਾਪਸ argsort_ਮਿਲਾਓ_pipelines[ਕ੍ਰਮ];
    }

    webgpu_pipeline get_get_ਪੰਕਤੀਆਂ_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, ਤਾਰਕਿਕ vec4_aligned) {
        ਅਚਲ ਤਾਰਕਿਕ vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_ਪੰਕਤੀਆਂ_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.src_type                          = context.src0->type;
        ਕੁੰਜੀ.vectorized                        = (ਪੂਰਨ) vectorized;

        ਸਵੈਚਾਲਿਤ it = get_ਪੰਕਤੀਆਂ_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != get_ਪੰਕਤੀਆਂ_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "get_rows";

        ਅਚਲ ਬਣਤਰ ggml_type_traits * type_traits = ggml_get_type_traits(ਕੁੰਜੀ.src_type);
        ਅਚਲ ਅੱਖਰ *                    type_str    = type_traits->type_name;

        ਚੋਣ (ਕੁੰਜੀ.src_type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("FLOAT_PARALLEL");
                ਜੇ (ਕੁੰਜੀ.vectorized) {
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("F32_VEC");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_TYPE=vec4<f32>");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_TYPE=vec4<f32>");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=4u");
                } ਨਹੀਂਤਾਂ {
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("F32");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_TYPE=f32");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_TYPE=f32");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("FLOAT_PARALLEL");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("F16");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_TYPE=f16");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_TYPE=f32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_I32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("FLOAT_PARALLEL");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("I32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_TYPE=i32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_TYPE=i32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_BF16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BF16");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_TYPE=u32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_TYPE=f32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                ਰੋਕੋ;
            ਮੂਲ:
                {
                    std::string type_ਉੱਤਲਾ = type_str;
                    std::transform(type_ਉੱਤਲਾ.begin(), type_ਉੱਤਲਾ.end(), type_ਉੱਤਲਾ.begin(), ::ਵੱਡਾ_ਅੱਖਰ);

                    ਚੋਣ (ਕੁੰਜੀ.src_type) {
                        ਹਾਲਤ GGML_TYPE_Q1_0:
                        ਹਾਲਤ GGML_TYPE_Q2_0:
                        ਹਾਲਤ GGML_TYPE_Q4_0:
                        ਹਾਲਤ GGML_TYPE_Q5_0:
                        ਹਾਲਤ GGML_TYPE_Q8_0:
                        ਹਾਲਤ GGML_TYPE_Q3_K:
                        ਹਾਲਤ GGML_TYPE_Q6_K:
                        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                        ਹਾਲਤ GGML_TYPE_IQ2_XS:
                        ਹਾਲਤ GGML_TYPE_IQ2_S:
                        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                        ਹਾਲਤ GGML_TYPE_IQ3_S:
                        ਹਾਲਤ GGML_TYPE_IQ1_S:
                        ਹਾਲਤ GGML_TYPE_IQ4_NL:
                        ਹਾਲਤ GGML_TYPE_MXFP4:
                        ਹਾਲਤ GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_TYPE=u32");
                                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("U32_DEQUANT_HELPERS");
                                ਰੋਕੋ;
                            }
                        ਮੂਲ:
                            {
                                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BYTE_HELPERS");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_T");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ);
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_SCALE_MIN");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_TABLES");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_GRID");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_LUT");

                    variant += "_";
                    variant += type_str;

                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_TYPE=f32");

                    ਜੇ (ਕੁੰਜੀ.src_type == GGML_TYPE_Q1_0) {
                        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=128u");
                    } ਨਹੀਂਤਾਂ ਜੇ ((ਕੁੰਜੀ.src_type >= GGML_TYPE_Q4_0 && ਕੁੰਜੀ.src_type <= GGML_TYPE_Q8_1) ||
                               ਕੁੰਜੀ.src_type == GGML_TYPE_IQ4_NL || ਕੁੰਜੀ.src_type == GGML_TYPE_MXFP4) {
                        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=32u");
                    } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.src_type == GGML_TYPE_Q2_0 || ਕੁੰਜੀ.src_type == GGML_TYPE_NVFP4) {
                        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=64u");
                    } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.src_type >= GGML_TYPE_Q2_K) {
                        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=256u");
                    } ਨਹੀਂਤਾਂ {
                        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=1u");
                    }
                    ਰੋਕੋ;
                }
        }

        ਜੇ (ਕੁੰਜੀ.vectorized) {
            variant += "_vec";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("WG_SIZE=" + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_get_ਪੰਕਤੀਆਂ, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_get_ਪੰਕਤੀਆਂ_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_ਪੰਕਤੀਆਂ_pipelines[ਕੁੰਜੀ]  = pipeline;
        ਵਾਪਸ get_ਪੰਕਤੀਆਂ_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_scale_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        ਸਵੈਚਾਲਿਤ it = scale_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != scale_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "scale";

        ਜੇ (ਕੁੰਜੀ.inplace) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INPLACE");
            variant += "_inplace";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_scale, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        decisions->inplace       = ਕੁੰਜੀ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[ਕੁੰਜੀ]     = pipeline;
        ਵਾਪਸ scale_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_solve_tri_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.type                               = context.dst->type;
        ਕੁੰਜੀ.n                                  = (ਪੂਰਨ) context.src0->ne[0];
        ਕੁੰਜੀ.k                                  = (ਪੂਰਨ) context.src1->ne[0];

        ਸਵੈਚਾਲਿਤ it = solve_tri_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != solve_tri_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "solve_tri";

        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                variant += "_f32";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        ਅਚਲ uint32_t wg_ਮਾਪ       = std::min((uint32_t) ਕੁੰਜੀ.n, context.max_wg_ਮਾਪ);
        ਅਚਲ uint32_t k_tile        = wg_ਮਾਪ;
        ਅਚਲ uint32_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ((uint32_t) ਕੁੰਜੀ.n + wg_ਮਾਪ) * GGML_WEBGPU_F32_ਮਾਪ_ਬਾਈਟਾਂ;
        ਅਚਲ uint32_t batch_n       = (uint32_t) (context.wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ / ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ);

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("N=") + std::to_string(ਕੁੰਜੀ.n));
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ਮਾਪ));
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_solve_tri, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = wg_ਮਾਪ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ solve_tri_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_ssm_conv_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.type                              = context.dst->type;
        ਕੁੰਜੀ.vectorized                        = context.src1->ne[0] == 4;

        ਸਵੈਚਾਲਿਤ it = ssm_conv_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != ssm_conv_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "ssm_conv";

        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                variant += "_f32";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        ਜੇ (ਕੁੰਜੀ.vectorized) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t ਖੰਡ_ਮਾਪ_3    = 32u;
        constexpr uint32_t ਟੋਕਨ_ਪ੍ਰਤੀ_wg = 8u;

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLOCK_SIZE=" + std::to_string(ਖੰਡ_ਮਾਪ_3) + "u");
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TOKENS_PER_WG=" + std::to_string(ਟੋਕਨ_ਪ੍ਰਤੀ_wg) + "u");

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_ssm_conv, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->ਖੰਡ_ਮਾਪ_3    = ਖੰਡ_ਮਾਪ_3;
        decisions->ਟੋਕਨ_ਪ੍ਰਤੀ_wg = ਟੋਕਨ_ਪ੍ਰਤੀ_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[ਕੁੰਜੀ]  = pipeline;
        ਵਾਪਸ ssm_conv_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_ssm_ਜਾਂਚ_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_ਜਾਂਚ_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.type                              = context.dst->type;
        ਕੁੰਜੀ.d_state                           = (ਪੂਰਨ) context.src0->ne[0];
        ਕੁੰਜੀ.xbc_overlap                       = xbc_overlap;

        ਸਵੈਚਾਲਿਤ it = ssm_ਜਾਂਚ_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != ssm_ਜਾਂਚ_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "ssm_scan";

        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                variant += "_f32";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        ਅਚਲ uint32_t wg_ਮਾਪ = (uint32_t) ਕੁੰਜੀ.d_state;

        constexpr uint32_t ਟੋਕਨ_ਪ੍ਰਤੀ_tile = 4u;

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("WG_SIZE=" + std::to_string(wg_ਮਾਪ) + "u");
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TOKENS_PER_TILE=" + std::to_string(ਟੋਕਨ_ਪ੍ਰਤੀ_tile) + "u");

        ਜੇ (context.supports_subgroups) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } ਨਹੀਂਤਾਂ {
            variant += "_wg_reduce";
        }

        ਜੇ (ਕੁੰਜੀ.xbc_overlap == 0b110) {  // x/B
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.xbc_overlap == 0b011) {  // B/C
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.xbc_overlap == 0b111) {  // x/B/C
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(ਕੁੰਜੀ.d_state);

        ਸਵੈਚਾਲਿਤ processed             = preprocessor.preprocess(wgsl_ssm_ਜਾਂਚ, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions             = std::make_shared<ggml_webgpu_ssm_ਜਾਂਚ_shader_decisions>();
        decisions->wg_ਮਾਪ         = wg_ਮਾਪ;
        decisions->ਟੋਕਨ_ਪ੍ਰਤੀ_tile = ਟੋਕਨ_ਪ੍ਰਤੀ_tile;
        decisions->xbc_overlap     = ਕੁੰਜੀ.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_ਜਾਂਚ_pipelines[ਕੁੰਜੀ]    = pipeline;
        ਵਾਪਸ ssm_ਜਾਂਚ_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.type                                     = context.dst->type;
        ਕੁੰਜੀ.s_v                                      = (ਪੂਰਨ) context.src2->ne[0];
        ਕੁੰਜੀ.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        ਕੁੰਜੀ.fused_cache                              = context.dst_fuse != ਖਾਲੀ_ਸੂਚਕ;

        ਸਵੈਚਾਲਿਤ it = gated_delta_net_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != gated_delta_net_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "gated_delta_net";

        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                variant += "_f32";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        ਜੇ (ਕੁੰਜੀ.kda) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("KDA");
            variant += "_kda";
        }

        ਜੇ (ਕੁੰਜੀ.fused_cache) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("S_V=" + std::to_string(ਕੁੰਜੀ.s_v) + "u");
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("WG_SIZE=" + std::to_string(ਕੁੰਜੀ.s_v) + "u");

        ਸਵੈਚਾਲਿਤ            processed      = preprocessor.preprocess(wgsl_gated_delta_net, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ gated_delta_net_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_pad_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        ਸਵੈਚਾਲਿਤ it = pad_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != pad_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "pad";

        ਜੇ (ਕੁੰਜੀ.circular) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("CIRCULAR");
            variant += "_circular";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_pad, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[ਕੁੰਜੀ]       = pipeline;
        ਵਾਪਸ pad_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_quantize_q8_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.src0_type                            = context.src0->type;

        ਸਵੈਚਾਲਿਤ it = quantize_q8_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != quantize_q8_pipelines.end()) {
            ਵਾਪਸ it->second;
        }
        ਅਚਲ ਅੱਖਰ *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "quantize_q8";

        uint32_t wg_ਮਾਪ = WEBGPU_MUL_MAT_VEC_WG_ਮਾਪ;

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC1_INNER_TYPE=f32");
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ਮਾਪ));

        ਅਚਲ ਬਣਤਰ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_ਉੱਤਲਾ  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_ਉੱਤਲਾ.begin(), type_ਉੱਤਲਾ.end(), type_ਉੱਤਲਾ.begin(), ::ਵੱਡਾ_ਅੱਖਰ);

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_" + type_ਉੱਤਲਾ);
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("Q8_1_T");

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        ਸਵੈਚਾਲਿਤ processed             = preprocessor.preprocess(shader_src, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ         = wg_ਮਾਪ;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ quantize_q8_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, ਤਾਰਕਿਕ src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.src0_type                            = context.src0->type;
        ਕੁੰਜੀ.src1_type                            = context.src1->type;
        ਕੁੰਜੀ.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        ਕੁੰਜੀ.ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ   = context.dst->ne[1];
        ਕੁੰਜੀ.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_ਗੁਣਨਫਲ, context.vendor);
        ਕੁੰਜੀ.src_overlap = src_overlap;

        ਸਵੈਚਾਲਿਤ it = mul_mat_vec_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != mul_mat_vec_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant    = "mul_mat_vec";
        ਅਚਲ ਅੱਖਰ *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        ਚੋਣ (context.src0->type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=f32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=f16");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F16");
                variant += "_f16";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_BF16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=u32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_BF16");
                variant += "_bf16";
                ਰੋਕੋ;
            ਮੂਲ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ਅਚਲ ਬਣਤਰ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_ਉੱਤਲਾ  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_ਉੱਤਲਾ.begin(), type_ਉੱਤਲਾ.end(), type_ਉੱਤਲਾ.begin(), ::ਵੱਡਾ_ਅੱਖਰ);

                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BYTE_HELPERS");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_" + type_ਉੱਤਲਾ);
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("U32_DEQUANT_HELPERS");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=u32");
                    ਚੋਣ (context.src0->type) {
                        ਹਾਲਤ GGML_TYPE_Q4_0:
                        ਹਾਲਤ GGML_TYPE_Q4_1:
                        ਹਾਲਤ GGML_TYPE_Q5_0:
                        ਹਾਲਤ GGML_TYPE_Q5_1:
                        ਹਾਲਤ GGML_TYPE_Q8_0:
                            ਜੇ (ਕੁੰਜੀ.use_mmvq) {
                                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_Q1_0:
                        ਹਾਲਤ GGML_TYPE_Q2_K:
                        ਹਾਲਤ GGML_TYPE_Q3_K:
                        ਹਾਲਤ GGML_TYPE_Q4_K:
                        ਹਾਲਤ GGML_TYPE_Q5_K:
                        ਹਾਲਤ GGML_TYPE_Q6_K:
                            ਜੇ (ਕੁੰਜੀ.use_mmvq) {
                                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("K_QUANTS_HANDLING");
                            }
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_IQ1_S:
                        ਹਾਲਤ GGML_TYPE_IQ1_M:
                        ਹਾਲਤ GGML_TYPE_IQ2_S:
                        ਹਾਲਤ GGML_TYPE_IQ3_S:
                        ਹਾਲਤ GGML_TYPE_IQ4_NL:
                        ਹਾਲਤ GGML_TYPE_IQ4_XS:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_GRID");
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                        ਹਾਲਤ GGML_TYPE_IQ2_XS:
                        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_GRID");
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_TABLES");
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_MXFP4:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_LUT");
                            ਜੇ (ਕੁੰਜੀ.use_mmvq) {
                                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_NVFP4:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_LUT");
                            ਰੋਕੋ;
                        ਮੂਲ:
                            ਰੋਕੋ;
                    }
                    ਰੋਕੋ;
                }
        }

        // src1 type (vector)
        ਚੋਣ (context.src1->type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(ਕੁੰਜੀ.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_ਮਾਪ        = WEBGPU_MUL_MAT_VEC_WG_ਮਾਪ;
        uint32_t outputs_ਪ੍ਰਤੀ_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_ਪ੍ਰਤੀ_WG;

        ਜੇ (ਕੁੰਜੀ.src0_type == GGML_TYPE_Q1_0) {
            outputs_ਪ੍ਰਤੀ_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_ਪ੍ਰਤੀ_WG;
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.src0_type >= GGML_TYPE_Q2_K) {
            outputs_ਪ੍ਰਤੀ_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_ਪ੍ਰਤੀ_WG;
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.src0_type >= GGML_TYPE_Q4_0) {
            outputs_ਪ੍ਰਤੀ_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_ਪ੍ਰਤੀ_WG;
        }

        ਜੇ (ਕੁੰਜੀ.use_mmvq) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MMVQ");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("Q8_1_T");
        }

        ਜੇ (ਕੁੰਜੀ.src_overlap) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ਮਾਪ));
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_ਪ੍ਰਤੀ_wg));
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        ਜੇ (ਕੁੰਜੀ.vectorized) {
            variant += "_vectorized";
        }
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("NUM_COLS=") + std::to_string(ਕੁੰਜੀ.ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ));

        ਸਵੈਚਾਲਿਤ processed            = preprocessor.preprocess(shader_src, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_ਮਾਪ        = wg_ਮਾਪ;
        decisions->outputs_ਪ੍ਰਤੀ_wg = outputs_ਪ੍ਰਤੀ_wg;
        decisions->vec_ਮਾਪ       = ਕੁੰਜੀ.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ mul_mat_vec_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, ਤਾਰਕਿਕ src_overlap) {
        ggml_webgpu_mul_mat_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.src0_type                        = context.src0->type;
        ਕੁੰਜੀ.src1_type                        = context.src1->type;
        ਕੁੰਜੀ.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        ਕੁੰਜੀ.use_subgroup_ਮੈਟ੍ਰਿਕਸ = context.supports_subgroup_ਮੈਟ੍ਰਿਕਸ;
        ਕੁੰਜੀ.src_overlap         = src_overlap;

        ਸਵੈਚਾਲਿਤ it = mul_mat_fast_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != mul_mat_fast_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        ਅਚਲ ਅੱਖਰ * shader_src = ਕੁੰਜੀ.use_subgroup_ਮੈਟ੍ਰਿਕਸ ? wgsl_mul_mat_subgroup_ਮੈਟ੍ਰਿਕਸ : wgsl_mul_mat_reg_tile;
        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = ਕੁੰਜੀ.use_subgroup_ਮੈਟ੍ਰਿਕਸ ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        ਚੋਣ (context.src1->type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC1_INNER_TYPE=f32");
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC1_INNER_TYPE=f16");
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ਅਚਲ ਬਣਤਰ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ਅਚਲ ਅੱਖਰ *                    src0_name   = src0_traits->type_name;

        ਚੋਣ (context.src0->type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=f32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=f16");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F16");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_BF16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=u32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_BF16");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                ਰੋਕੋ;
            ਮੂਲ:
                {
                    std::string type_ਉੱਤਲਾ = src0_name;
                    std::transform(type_ਉੱਤਲਾ.begin(), type_ਉੱਤਲਾ.end(), type_ਉੱਤਲਾ.begin(), ::ਵੱਡਾ_ਅੱਖਰ);

                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BYTE_HELPERS");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_" + type_ਉੱਤਲਾ);
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC0_SHMEM_" + type_ਉੱਤਲਾ);
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC1_SHMEM_FLOAT");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("U32_DEQUANT_HELPERS");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=u32");

                    ਚੋਣ (context.src0->type) {
                        ਹਾਲਤ GGML_TYPE_IQ1_S:
                        ਹਾਲਤ GGML_TYPE_IQ1_M:
                        ਹਾਲਤ GGML_TYPE_IQ4_NL:
                        ਹਾਲਤ GGML_TYPE_IQ4_XS:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_GRID");
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                        ਹਾਲਤ GGML_TYPE_IQ2_XS:
                        ਹਾਲਤ GGML_TYPE_IQ2_S:
                        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                        ਹਾਲਤ GGML_TYPE_IQ3_S:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_GRID");
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_TABLES");
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_MXFP4:
                        ਹਾਲਤ GGML_TYPE_NVFP4:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_LUT");
                            ਰੋਕੋ;
                        ਮੂਲ:
                            ਰੋਕੋ;
                    }

                    variant += std::string("_") + src0_name;
                    ਰੋਕੋ;
                }
        }

        // VEC/SCALAR controls
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(ਕੁੰਜੀ.vectorized ? "VEC" : "SCALAR");

        ਅਚਲ ਤਾਰਕਿਕ is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        ਜੇ (ਕੁੰਜੀ.use_subgroup_ਮੈਟ੍ਰਿਕਸ) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } ਨਹੀਂਤਾਂ {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        ਜੇ (ਕੁੰਜੀ.use_subgroup_ਮੈਟ੍ਰਿਕਸ) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_ਮਾਪ) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_ਮੈਟ੍ਰਿਕਸ_M) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_ਮੈਟ੍ਰਿਕਸ_N) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        ਜੇ (ਕੁੰਜੀ.vectorized) {
            variant += "_vectorized";
        }

        ਜੇ (ਕੁੰਜੀ.src_overlap) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ਜੇ (!ਕੁੰਜੀ.use_subgroup_ਮੈਟ੍ਰਿਕਸ) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_ਮਾਪ_M) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_ਮਾਪ_N) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        ਸਵੈਚਾਲਿਤ processed = preprocessor.preprocess(shader_src, ਨਿਰਧਾਰਤ_ਕਰਦਾ);

        ਸਵੈਚਾਲਿਤ decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_ਮੈਟ੍ਰਿਕਸ = ਕੁੰਜੀ.use_subgroup_ਮੈਟ੍ਰਿਕਸ;
        ਜੇ (ਕੁੰਜੀ.use_subgroup_ਮੈਟ੍ਰਿਕਸ) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_ਮੈਟ੍ਰਿਕਸ_m = WEBGPU_MUL_MAT_SUBGROUP_ਮੈਟ੍ਰਿਕਸ_M;
            decisions->subgroup_ਮੈਟ੍ਰਿਕਸ_n = WEBGPU_MUL_MAT_SUBGROUP_ਮੈਟ੍ਰਿਕਸ_N;
            decisions->wg_ਮਾਪ           = context.max_subgroup_ਮਾਪ;
        } ਨਹੀਂਤਾਂ {
            decisions->wg_ਮਾਪ_m       = WEBGPU_MUL_MAT_WG_ਮਾਪ_M;
            decisions->wg_ਮਾਪ_n       = WEBGPU_MUL_MAT_WG_ਮਾਪ_N;
            decisions->wg_ਮਾਪ         = WEBGPU_MUL_MAT_WG_ਮਾਪ_M * WEBGPU_MUL_MAT_WG_ਮਾਪ_N;
            decisions->mul_mat_wg_ਮਾਪ = WEBGPU_MUL_MAT_WG_ਮਾਪ;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ mul_mat_fast_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.n_experts                                  = (uint32_t) context.src0->ne[2];

        ਸਵੈਚਾਲਿਤ it = mul_mat_id_gather_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != mul_mat_id_gather_pipelines.end()) {
            ਵਾਪਸ it->second;
        }
        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ) + "u");
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("N_EXPERT=") + std::to_string(ਕੁੰਜੀ.n_experts) + "u");
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_ਮਾਪ_N) + "u");

        ਸਵੈਚਾਲਿਤ processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ = context.max_wg_ਮਾਪ;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.src0_type                           = context.src0->type;
        ਕੁੰਜੀ.src1_type                           = context.src1->type;
        ਕੁੰਜੀ.n_experts                           = context.src0->ne[2];
        ਕੁੰਜੀ.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        ਸਵੈਚਾਲਿਤ it = mul_mat_id_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != mul_mat_id_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "mul_mat_id";
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_MAT_ID");

        // src1 type
        ਚੋਣ (context.src1->type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC1_INNER_TYPE=f32");
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC1_INNER_TYPE=f16");
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ਅਚਲ ਬਣਤਰ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        ਅਚਲ ਅੱਖਰ *                    src0_name   = src0_traits->type_name;

        ਚੋਣ (context.src0->type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=f32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC1_SHMEM_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=f16");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC1_SHMEM_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F16");
                variant += "_f16";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_BF16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=u32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC0_SHMEM_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC1_SHMEM_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_BF16");
                variant += "_bf16";
                ਰੋਕੋ;
            ਮੂਲ:
                {
                    std::string type_ਉੱਤਲਾ = src0_name;
                    std::transform(type_ਉੱਤਲਾ.begin(), type_ਉੱਤਲਾ.end(), type_ਉੱਤਲਾ.begin(), ::ਵੱਡਾ_ਅੱਖਰ);

                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BYTE_HELPERS");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC0_SHMEM_" + type_ਉੱਤਲਾ);
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INIT_SRC1_SHMEM_FLOAT");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("U32_DEQUANT_HELPERS");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=u32");

                    ਚੋਣ (context.src0->type) {
                        ਹਾਲਤ GGML_TYPE_IQ1_S:
                        ਹਾਲਤ GGML_TYPE_IQ1_M:
                        ਹਾਲਤ GGML_TYPE_IQ4_NL:
                        ਹਾਲਤ GGML_TYPE_IQ4_XS:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_GRID");
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                        ਹਾਲਤ GGML_TYPE_IQ2_XS:
                        ਹਾਲਤ GGML_TYPE_IQ2_S:
                        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                        ਹਾਲਤ GGML_TYPE_IQ3_S:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_GRID");
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_TABLES");
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_MXFP4:
                        ਹਾਲਤ GGML_TYPE_NVFP4:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_LUT");
                            ਰੋਕੋ;
                        ਮੂਲ:
                            ਰੋਕੋ;
                    }

                    variant += std::string("_") + src0_name;
                    ਰੋਕੋ;
                }
        }

        // VEC/SCALAR controls
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(ਕੁੰਜੀ.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        ਅਚਲ uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_ਮਾਪ_M) + "u");
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_ਮਾਪ_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        ਜੇ (ਕੁੰਜੀ.vectorized) {
            variant += "_vectorized";
        }

        ਸਵੈਚਾਲਿਤ processed = preprocessor.preprocess(wgsl_mul_mat_id, ਨਿਰਧਾਰਤ_ਕਰਦਾ);

        ਸਵੈਚਾਲਿਤ decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_ਮਾਪ_m = WEBGPU_MUL_MAT_WG_ਮਾਪ_M;
        decisions->wg_ਮਾਪ_n = WEBGPU_MUL_MAT_WG_ਮਾਪ_N;
        decisions->wg_ਮਾਪ   = WEBGPU_MUL_MAT_WG_ਮਾਪ_M * WEBGPU_MUL_MAT_WG_ਮਾਪ_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ mul_mat_id_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.src0_type                           = context.src0->type;
        ਕੁੰਜੀ.src1_type                           = context.src1->type;
        ਕੁੰਜੀ.n_experts                           = context.src0->ne[2];
        ਕੁੰਜੀ.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        ਸਵੈਚਾਲਿਤ it = mul_mat_id_vec_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != mul_mat_id_vec_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant    = "mul_mat_id_vec";
        ਅਚਲ ਅੱਖਰ *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        ਚੋਣ (context.src1->type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC1_INNER_TYPE=f32");
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC1_INNER_TYPE=f16");
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ਚੋਣ (context.src0->type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=f32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=f16");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F16");
                variant += "_f16";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_BF16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=u32");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_FLOAT");
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_BF16");
                variant += "_bf16";
                ਰੋਕੋ;
            ਮੂਲ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    ਅਚਲ ਬਣਤਰ ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_ਉੱਤਲਾ  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_ਉੱਤਲਾ.begin(), type_ਉੱਤਲਾ.end(), type_ਉੱਤਲਾ.begin(), ::ਵੱਡਾ_ਅੱਖਰ);

                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BYTE_HELPERS");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MUL_ACC_" + type_ਉੱਤਲਾ);
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("U32_DEQUANT_HELPERS");
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC0_INNER_TYPE=u32");
                    ਚੋਣ (context.src0->type) {
                        ਹਾਲਤ GGML_TYPE_IQ1_S:
                        ਹਾਲਤ GGML_TYPE_IQ1_M:
                        ਹਾਲਤ GGML_TYPE_IQ2_S:
                        ਹਾਲਤ GGML_TYPE_IQ3_S:
                        ਹਾਲਤ GGML_TYPE_IQ4_NL:
                        ਹਾਲਤ GGML_TYPE_IQ4_XS:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_GRID");
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                        ਹਾਲਤ GGML_TYPE_IQ2_XS:
                        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_GRID");
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_TABLES");
                            ਰੋਕੋ;
                        ਹਾਲਤ GGML_TYPE_MXFP4:
                        ਹਾਲਤ GGML_TYPE_NVFP4:
                            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(type_ਉੱਤਲਾ + "_LUT");
                            ਰੋਕੋ;
                        ਮੂਲ:
                            ਰੋਕੋ;
                    }
                    ਰੋਕੋ;
                }
        }

        // VEC/SCALAR controls
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(ਕੁੰਜੀ.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_ਮਾਪ        = WEBGPU_MUL_MAT_VEC_WG_ਮਾਪ;
        uint32_t outputs_ਪ੍ਰਤੀ_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_ਪ੍ਰਤੀ_WG;

        ਜੇ (ਕੁੰਜੀ.src0_type == GGML_TYPE_Q1_0) {
            outputs_ਪ੍ਰਤੀ_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_ਪ੍ਰਤੀ_WG;
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.src0_type >= GGML_TYPE_Q2_K) {
            outputs_ਪ੍ਰਤੀ_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_ਪ੍ਰਤੀ_WG;
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.src0_type >= GGML_TYPE_Q4_0) {
            outputs_ਪ੍ਰਤੀ_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_ਪ੍ਰਤੀ_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ਮਾਪ));
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_ਪ੍ਰਤੀ_wg));
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        ਜੇ (ਕੁੰਜੀ.vectorized) {
            variant += "_vectorized";
        }
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("NUM_COLS=1"));

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("N_EXPERTS=") + std::to_string(ਕੁੰਜੀ.n_experts));

        ਸਵੈਚਾਲਿਤ processed = preprocessor.preprocess(shader_src, ਨਿਰਧਾਰਤ_ਕਰਦਾ);

        ਸਵੈਚਾਲਿਤ decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_ਮਾਪ        = wg_ਮਾਪ;
        decisions->outputs_ਪ੍ਰਤੀ_wg = outputs_ਪ੍ਰਤੀ_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ mul_mat_id_vec_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_unary_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ਅਚਲ ਤਾਰਕਿਕ                     is_unary = context.dst->op == GGML_OP_UNARY;
        ਅਚਲ ਪੂਰਨ                      op       = is_unary ? (ਪੂਰਨ) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ      = {};
        ਕੁੰਜੀ.type                                = context.dst->type;
        ਕੁੰਜੀ.op                                  = op;
        ਕੁੰਜੀ.is_unary                            = is_unary;
        ਕੁੰਜੀ.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        ਕੁੰਜੀ.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        ਸਵੈਚਾਲਿਤ it = unary_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != unary_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant =
            ਕੁੰਜੀ.is_unary ? ggml_unary_op_name((ggml_unary_op) ਕੁੰਜੀ.op) : ggml_op_name((ggml_op) ਕੁੰਜੀ.op);
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(variant);

        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F16");
                variant += "_f16";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for unary shader");
        }

        ਜੇ (ਕੁੰਜੀ.inplace) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INPLACE");
            variant += "_inplace";
        }

        ਜੇ (op == GGML_OP_TRI) {
            ਚੋਣ (ਕੁੰਜੀ.ttype) {
                ਹਾਲਤ GGML_TRI_TYPE_ਹੇਠਲਾ:
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_TRI_TYPE_ਹੇਠਲਾ_DIAG:
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_TRI_TYPE_ਉੱਤਲਾ:
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_TRI_TYPE_ਉੱਤਲਾ_DIAG:
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    ਰੋਕੋ;
                ਮੂਲ:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_unary, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        decisions->inplace       = ਕੁੰਜੀ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[ਕੁੰਜੀ]     = pipeline;
        ਵਾਪਸ unary_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, ਤਾਰਕਿਕ src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ਕੁੰਜੀ.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ਕੁੰਜੀ.src_overlap                           = src_overlap;

        ਸਵੈਚਾਲਿਤ it = rms_norm_mul_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != rms_norm_mul_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        ਜੇ (ਕੁੰਜੀ.inplace) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INPLACE");
            variant += "_inplace";
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.overlap) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("OVERLAP");
            variant += "_overlap";
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.src_overlap) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_ਮਾਪ     = context.max_wg_ਮਾਪ;
        pipeline_decisions->inplace     = ਕੁੰਜੀ.inplace;
        pipeline_decisions->overlap     = ਕੁੰਜੀ.overlap;
        pipeline_decisions->src_overlap = ਕੁੰਜੀ.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[ਕੁੰਜੀ]     = pipeline;
        ਵਾਪਸ rms_norm_mul_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_ਦੁਆਧਾਰੀ_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, ਤਾਰਕਿਕ src_overlap) {
        ggml_webgpu_ਦੁਆਧਾਰੀ_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.type                            = context.dst->type;
        ਕੁੰਜੀ.op                              = context.dst->op;
        ਕੁੰਜੀ.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ਕੁੰਜੀ.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        ਕੁੰਜੀ.src_overlap                     = src_overlap;

        ਸਵੈਚਾਲਿਤ it = ਦੁਆਧਾਰੀ_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != ਦੁਆਧਾਰੀ_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              op_name = ggml_op_name((ggml_op) ਕੁੰਜੀ.op);
        std::string              variant = op_name;

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("OP_") + op_name);

        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F16");
                variant += "_f16";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for binary shader");
        }

        ਜੇ (ਕੁੰਜੀ.inplace) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INPLACE");
            variant += "_inplace";
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.overlap) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("OVERLAP");
            variant += "_overlap";
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.src_overlap) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed                  = preprocessor.preprocess(wgsl_ਦੁਆਧਾਰੀ, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ pipeline_decisions         = std::make_shared<ggml_webgpu_ਦੁਆਧਾਰੀ_shader_decisions>();
        pipeline_decisions->wg_ਮਾਪ     = context.max_wg_ਮਾਪ;
        pipeline_decisions->inplace     = ਕੁੰਜੀ.inplace;
        pipeline_decisions->overlap     = ਕੁੰਜੀ.overlap;
        pipeline_decisions->src_overlap = ਕੁੰਜੀ.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ਦੁਆਧਾਰੀ_pipelines[ਕੁੰਜੀ]    = pipeline;
        ਵਾਪਸ ਦੁਆਧਾਰੀ_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_ਜੋੜੋ_id_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ਜੋੜੋ_id_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        ਸਵੈਚਾਲਿਤ it = ਜੋੜੋ_id_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != ਜੋੜੋ_id_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant    = "add_id";
        ਅਚਲ ਅੱਖਰ *             shader_src = wgsl_ਜੋੜੋ_id;

        ਜੇ (ਕੁੰਜੀ.inplace) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INPLACE");
            variant += "_inplace";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed              = preprocessor.preprocess(shader_src, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_ਮਾਪ = context.max_wg_ਮਾਪ;
        pipeline_decisions->inplace = ਕੁੰਜੀ.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ਜੋੜੋ_id_pipelines[ਕੁੰਜੀ]    = pipeline;
        ਵਾਪਸ pipeline;
    }

    webgpu_pipeline get_concat_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, ਤਾਰਕਿਕ src_overlap) {
        ggml_webgpu_concat_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.type                            = context.dst->type;
        ਕੁੰਜੀ.src_overlap                     = src_overlap;

        ਸਵੈਚਾਲਿਤ it = concat_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != concat_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "concat";

        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_I32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_I32");
                variant += "_i32";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for concat shader");
        }

        ਜੇ (ਕੁੰਜੀ.src_overlap) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_concat, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_ਦੁਆਧਾਰੀ_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        decisions->src_overlap   = ਕੁੰਜੀ.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[ਕੁੰਜੀ]    = pipeline;
        ਵਾਪਸ concat_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_repeat_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.type                            = context.dst->type;

        ਸਵੈਚਾਲਿਤ it = repeat_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != repeat_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "repeat";

        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F16");
                variant += "_f16";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_I32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_I32");
                variant += "_i32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_I16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_I16");
                variant += "_i16";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_repeat, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[ਕੁੰਜੀ]    = pipeline;
        ਵਾਪਸ repeat_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_flash_attn_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, ਤਾਰਕਿਕ kv_overlap) {
        ਅਚਲ ਤਾਰਕਿਕ can_use_subgroup_ਮੈਟ੍ਰਿਕਸ = ggml_webgpu_flash_attn_can_use_subgroup_ਮੈਟ੍ਰਿਕਸ_ਰਸਤਾ(
            context.supports_subgroup_ਮੈਟ੍ਰਿਕਸ, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_ਮੈਟ੍ਰਿਕਸ                    = can_use_subgroup_ਮੈਟ੍ਰਿਕਸ;
        decisions.q_tile = decisions.use_sg_ਮੈਟ੍ਰਿਕਸ ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.common                              = ggml_webgpu_flash_attn_make_common_pipeline_ਕੁੰਜੀ(
            context, decisions.use_sg_ਮੈਟ੍ਰਿਕਸ ? context.sg_mat_k : 1u, kv_overlap);
        ਕੁੰਜੀ.common.k_ਸਿੱਧਾ &= decisions.use_sg_ਮੈਟ੍ਰਿਕਸ && ਕੁੰਜੀ.common.k_type == GGML_TYPE_F16;
        ਕੁੰਜੀ.common.v_ਸਿੱਧਾ &= decisions.use_sg_ਮੈਟ੍ਰਿਕਸ && ਕੁੰਜੀ.common.v_type == GGML_TYPE_F16;
        ਕੁੰਜੀ.use_sg_ਮੈਟ੍ਰਿਕਸ = decisions.use_sg_ਮੈਟ੍ਰਿਕਸ;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ, decisions.q_tile, decisions.use_sg_ਮੈਟ੍ਰਿਕਸ ? context.sg_mat_n : 1u,
            ਕੁੰਜੀ.common.ਸਿਰ_dim_qk, ਕੁੰਜੀ.common.ਸਿਰ_dim_v, ਕੁੰਜੀ.common.has_ਨਕਾਬ,
            ਕੁੰਜੀ.common.k_ਸਿੱਧਾ || ਕੁੰਜੀ.common.v_ਸਿੱਧਾ);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        ਜੇ (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_ਮੈਟ੍ਰਿਕਸ);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_ਮੈਟ੍ਰਿਕਸ = ਝੂਠ;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            ਕੁੰਜੀ.common.k_ਸਿੱਧਾ     = ਝੂਠ;
            ਕੁੰਜੀ.common.v_ਸਿੱਧਾ     = ਝੂਠ;
            ਕੁੰਜੀ.use_sg_ਮੈਟ੍ਰਿਕਸ       = ਝੂਠ;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ, decisions.q_tile, 1u, ਕੁੰਜੀ.common.ਸਿਰ_dim_qk, ਕੁੰਜੀ.common.ਸਿਰ_dim_v,
                ਕੁੰਜੀ.common.has_ਨਕਾਬ, ਕੁੰਜੀ.common.k_ਸਿੱਧਾ || ਕੁੰਜੀ.common.v_ਸਿੱਧਾ);
            GGML_ASSERT(max_kv_tile > 0);
        }

        decisions.kv_tile = decisions.use_sg_ਮੈਟ੍ਰਿਕਸ ?
                                std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES) :
                                std::min(GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE, max_kv_tile);
        decisions.wg_ਮਾਪ =
            decisions.use_sg_ਮੈਟ੍ਰਿਕਸ ?
                std::max(context.max_subgroup_ਮਾਪ, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_ਮਾਪ) :
                std::min(context.max_wg_ਮਾਪ, std::max(GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_ਮਾਪ,
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * context.max_subgroup_ਮਾਪ));

        ਜੇ (ਕੁੰਜੀ.common.k_ਸਿੱਧਾ || ਕੁੰਜੀ.common.v_ਸਿੱਧਾ) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            ਜਦੋਂ (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_ਮੈਟ੍ਰਿਕਸ ? context.sg_mat_n : context.min_subgroup_ਮਾਪ;
            }
        }

        ਸਵੈਚਾਲਿਤ it = flash_attn_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != flash_attn_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::string              variant = decisions.use_sg_ਮੈਟ੍ਰਿਕਸ ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ = ggml_webgpu_flash_attn_common_ਨਿਰਧਾਰਤ_ਕਰਦਾ(ਕੁੰਜੀ.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_ਮਾਪ);
        ਅਚਲ ਅੱਖਰ *             shader_src = ਖਾਲੀ_ਸੂਚਕ;
        ਜੇ (!ਕੁੰਜੀ.use_sg_ਮੈਟ੍ਰਿਕਸ) {
            shader_src = wgsl_flash_attn_tile;
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_ਮਾਪ) + "u");
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_ਮਾਪ) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_ਮਾਪ) + "_" +
                       std::to_string(context.max_subgroup_ਮਾਪ);
        } ਨਹੀਂਤਾਂ {
            shader_src = wgsl_flash_attn;
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        ਸਵੈਚਾਲਿਤ            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, ਨਿਰਧਾਰਤ_ਕਰਦਾ), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ flash_attn_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, ਤਾਰਕਿਕ kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.common = ggml_webgpu_flash_attn_make_common_pipeline_ਕੁੰਜੀ(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ਚੌੜਾਈ,
                                                                     kv_overlap);

        ਸਵੈਚਾਲਿਤ it = flash_attn_vec_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != flash_attn_vec_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ, ਕੁੰਜੀ.common.ਸਿਰ_dim_qk,
                                                                   ਕੁੰਜੀ.common.ਸਿਰ_dim_v, ਕੁੰਜੀ.common.has_ਨਕਾਬ,
                                                                   ਕੁੰਜੀ.common.k_ਸਿੱਧਾ || ਕੁੰਜੀ.common.v_ਸਿੱਧਾ);
        decisions.wg_ਮਾਪ = context.max_subgroup_ਮਾਪ;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ =
            ggml_webgpu_flash_attn_common_ਨਿਰਧਾਰਤ_ਕਰਦਾ(ਕੁੰਜੀ.common, variant, 1u, decisions.kv_tile, decisions.wg_ਮਾਪ);
        ਜੇ (ਕੁੰਜੀ.common.has_ਨਕਾਬ) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BLK");
            variant.resize(variant.size() - (ਆਕਾਰ("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_ਵੰਡੋ = context.min_subgroup_ਮਾਪ;
        ਅਚਲ uint32_t D       = ਕੁੰਜੀ.common.ਸਿਰ_dim_qk | ਕੁੰਜੀ.common.ਸਿਰ_dim_v;
        ਅਚਲ uint32_t D_lsb   = D & (~(D - 1u));
        d_ਵੰਡੋ                = std::min(std::min(context.min_subgroup_ਮਾਪ, 4u), std::max(D_lsb / 4u, 1u));

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("D_SPLIT=") + std::to_string(d_ਵੰਡੋ));
        variant += "_dsplit" + std::to_string(d_ਵੰਡੋ);

        ਸਵੈਚਾਲਿਤ            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_ਵੰਡੋ, ਨਿਰਧਾਰਤ_ਕਰਦਾ), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ flash_attn_vec_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.kv_tile                                 = kv_tile;
        ਸਵੈਚਾਲਿਤ it                                     = flash_attn_blk_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != flash_attn_blk_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "flash_attn_vec_blk";

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("KV_TILE=") + std::to_string(ਕੁੰਜੀ.kv_tile));
        variant += std::string("_kvt") + std::to_string(ਕੁੰਜੀ.kv_tile);

        uint32_t wg_ਮਾਪ = 1;
        ਜਦੋਂ ((wg_ਮਾਪ << 1) <= context.max_wg_ਮਾਪ) {
            wg_ਮਾਪ <<= 1;
        }
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(wg_ਮਾਪ));
        variant += std::string("_wg") + std::to_string(wg_ਮਾਪ);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, ਨਿਰਧਾਰਤ_ਕਰਦਾ), variant);
        flash_attn_blk_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ flash_attn_blk_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.ਸਿਰ_dim_v                                     = (uint32_t) context.src2->ne[0];
        ਕੁੰਜੀ.dst_type                                       = context.dst->type;
        ਕੁੰਜੀ.wg_ਮਾਪ                                        = context.max_wg_ਮਾਪ;
        ਸਵੈਚਾਲਿਤ it                                            = flash_attn_vec_reduce_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != flash_attn_vec_reduce_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "flash_attn_vec_reduce";

        ਚੋਣ (ਕੁੰਜੀ.dst_type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F32");
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F16");
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(ਕੁੰਜੀ.dst_type);

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("HEAD_DIM_V=") + std::to_string(ਕੁੰਜੀ.ਸਿਰ_dim_v));
        variant += std::string("_hsv") + std::to_string(ਕੁੰਜੀ.ਸਿਰ_dim_v);

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));
        variant += std::string("_wg") + std::to_string(context.max_wg_ਮਾਪ);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, ਨਿਰਧਾਰਤ_ਕਰਦਾ), variant);
        flash_attn_vec_reduce_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ flash_attn_vec_reduce_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_cpy_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.src_type                     = context.src0->type;
        ਕੁੰਜੀ.dst_type                     = context.dst->type;

        ਸਵੈਚਾਲਿਤ it = cpy_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != cpy_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "cpy";

        ਚੋਣ (ਕੁੰਜੀ.src_type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_F16");
                variant += "_f16";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_I32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_I32");
                variant += "_i32";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        ਚੋਣ (ਕੁੰਜੀ.dst_type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F16");
                variant += "_f16";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_I32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_I32");
                variant += "_i32";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_cpy, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[ਕੁੰਜੀ]       = pipeline;
        ਵਾਪਸ cpy_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_glu_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, ਤਾਰਕਿਕ src_overlap) {
        ggml_webgpu_glu_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.glu_op                       = ggml_get_glu_op(context.dst);
        ਕੁੰਜੀ.type                         = context.dst->type;
        ਕੁੰਜੀ.ਵੰਡੋ                        = (context.src1 != ਖਾਲੀ_ਸੂਚਕ);
        ਕੁੰਜੀ.src_overlap                  = src_overlap;

        ਸਵੈਚਾਲਿਤ it = glu_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != glu_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "glu";

        ਚੋਣ (ਕੁੰਜੀ.glu_op) {
            ਹਾਲਤ GGML_GLU_OP_REGLU:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("OP_REGLU");
                variant += "_reglu";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_GLU_OP_GEGLU:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("OP_GEGLU");
                variant += "_geglu";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("OP_SWIGLU");
                variant += "_swiglu";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported GLU op");
        }
        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F16");
                variant += "_f16";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        ਜੇ (ਕੁੰਜੀ.src_overlap) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } ਨਹੀਂਤਾਂ ਜੇ (ਕੁੰਜੀ.ਵੰਡੋ) {
            variant += "_split";
        } ਨਹੀਂਤਾਂ {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("NO_SPLIT");
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_glu, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        decisions->src_overlap   = ਕੁੰਜੀ.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[ਕੁੰਜੀ]       = pipeline;
        ਵਾਪਸ glu_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_rope_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.type                          = context.dst->type;
        ਕੁੰਜੀ.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        ਕੁੰਜੀ.has_ff                        = (context.src2 != ਖਾਲੀ_ਸੂਚਕ);

        ਸਵੈਚਾਲਿਤ it = rope_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != rope_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "rope";

        ਚੋਣ (ਕੁੰਜੀ.type) {
            ਹਾਲਤ GGML_TYPE_F32:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F32");
                variant += "_f32";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("TYPE_F16");
                variant += "_f16";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        ਜੇ (ਕੁੰਜੀ.inplace) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INPLACE");
            variant += "_inplace";
        }

        ਜੇ (ਕੁੰਜੀ.has_ff) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("FF_FUNC");
            variant += "_ff";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_rope, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        decisions->inplace       = ਕੁੰਜੀ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[ਕੁੰਜੀ]      = pipeline;
        ਵਾਪਸ rope_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_soft_max_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.ਨਕਾਬ_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        ਕੁੰਜੀ.has_ਨਕਾਬ                          = (context.src1 != ਖਾਲੀ_ਸੂਚਕ);
        ਕੁੰਜੀ.has_sink                          = (context.src2 != ਖਾਲੀ_ਸੂਚਕ);
        ਕੁੰਜੀ.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        ਸਵੈਚਾਲਿਤ it = soft_max_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != soft_max_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "soft_max";

        ਜੇ (ਕੁੰਜੀ.has_ਨਕਾਬ) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("HAS_MASK");
            ਚੋਣ (ਕੁੰਜੀ.ਨਕਾਬ_type) {
                ਹਾਲਤ GGML_TYPE_F32:
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MASK_F32");
                    variant += "_mask_f32";
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_TYPE_F16:
                    ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("MASK_F16");
                    variant += "_mask_f16";
                    ਰੋਕੋ;
                ਮੂਲ:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        ਜੇ (ਕੁੰਜੀ.has_sink) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("HAS_SINK");
            variant += "_sink";
        }

        ਜੇ (ਕੁੰਜੀ.inplace) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("INPLACE");
            variant += "_inplace";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_soft_max, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        decisions->inplace       = ਕੁੰਜੀ.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[ਕੁੰਜੀ]  = pipeline;
        ਵਾਪਸ soft_max_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_conv2d_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.ਭਾਰ_type                     = context.src0->type;
        ਕੁੰਜੀ.ਨਿਵੇਸ਼_type                      = context.src1->type;
        ਕੁੰਜੀ.output_type                     = context.dst->type;

        ਸਵੈਚਾਲਿਤ it = conv2d_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != conv2d_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "conv_2d";

        ਸਵੈਚਾਲਿਤ ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ = [&](ਅਚਲ ਅੱਖਰ * prefix, ggml_type type) {
            std::string s_ਅਗੇਤਰ = prefix;
            ਜੇ (type == GGML_TYPE_F32) {
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(s_ਅਗੇਤਰ + "=f32");
            } ਨਹੀਂਤਾਂ ਜੇ (type == GGML_TYPE_F16) {
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(s_ਅਗੇਤਰ + "=f16");
            } ਨਹੀਂਤਾਂ {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ("WEIGHT_TYPE", ਕੁੰਜੀ.ਭਾਰ_type);
        ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ("INPUT_TYPE", ਕੁੰਜੀ.ਨਿਵੇਸ਼_type);
        ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ("OUTPUT_TYPE", ਕੁੰਜੀ.output_type);

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_conv2d, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[ਕੁੰਜੀ]    = pipeline;
        ਵਾਪਸ conv2d_pipelines[ਕੁੰਜੀ];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context, ਤਾਰਕਿਕ whcn) {
        ggml_webgpu_conv2d_dw_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.ਭਾਰ_type                        = context.src0->type;
        ਕੁੰਜੀ.ਨਿਵੇਸ਼_type                         = context.src1->type;
        ਕੁੰਜੀ.output_type                        = context.dst->type;
        ਕੁੰਜੀ.whcn                               = whcn;

        ਸਵੈਚਾਲਿਤ it = conv2d_dw_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != conv2d_dw_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        ਸਵੈਚਾਲਿਤ ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ = [&](ਅਚਲ ਅੱਖਰ * prefix, ggml_type type) {
            std::string s_ਅਗੇਤਰ = prefix;
            ਜੇ (type == GGML_TYPE_F32) {
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(s_ਅਗੇਤਰ + "=f32");
            } ਨਹੀਂਤਾਂ ਜੇ (type == GGML_TYPE_F16) {
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(s_ਅਗੇਤਰ + "=f16");
            } ਨਹੀਂਤਾਂ {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ("WEIGHT_TYPE", ਕੁੰਜੀ.ਭਾਰ_type);
        ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ("INPUT_TYPE", ਕੁੰਜੀ.ਨਿਵੇਸ਼_type);
        ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ("OUTPUT_TYPE", ਕੁੰਜੀ.output_type);

        ਜੇ (whcn) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("WHCN");
        }
        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_conv2d_dw, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[ਕੁੰਜੀ] = pipeline;
        ਵਾਪਸ conv2d_dw_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_im2col_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.ਨਿਵੇਸ਼_type                      = context.src1->type;
        ਕੁੰਜੀ.output_type                     = context.dst->type;

        ਸਵੈਚਾਲਿਤ it = im2col_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != im2col_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "im2col";

        ਸਵੈਚਾਲਿਤ ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ = [&](ਅਚਲ ਅੱਖਰ * prefix, ggml_type type) {
            std::string s_ਅਗੇਤਰ = prefix;
            ਜੇ (type == GGML_TYPE_F32) {
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(s_ਅਗੇਤਰ + "=f32");
            } ਨਹੀਂਤਾਂ ਜੇ (type == GGML_TYPE_F16) {
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(s_ਅਗੇਤਰ + "=f16");
            } ਨਹੀਂਤਾਂ {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ("INPUT_TYPE", ਕੁੰਜੀ.ਨਿਵੇਸ਼_type);
        ਧੱਕੋ_type_ਨਿਰਧਾਰਤ_ਕਰਦਾ("OUTPUT_TYPE", ਕੁੰਜੀ.output_type);

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_im2col, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[ਕੁੰਜੀ]    = pipeline;
        ਵਾਪਸ im2col_pipelines[ਕੁੰਜੀ];
    }

    webgpu_pipeline get_upscale_pipeline(ਅਚਲ ggml_webgpu_shader_lib_context & context) {
        ਅਚਲ uint32_t ਬਹੁਲਕ_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        ਅਚਲ uint32_t ਅਧਾਰ_ਬਹੁਲਕ  = ਬਹੁਲਕ_flags & 0xFFu;
        ਅਚਲ ਤਾਰਕਿਕ     antialias  = (ਬਹੁਲਕ_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.ਨਿਵੇਸ਼_type                       = context.src0->type;
        ਕੁੰਜੀ.output_type                      = context.dst->type;
        ਕੁੰਜੀ.ਅਧਾਰ_ਬਹੁਲਕ                        = ਅਧਾਰ_ਬਹੁਲਕ;
        ਕੁੰਜੀ.antialias                        = antialias;

        ਸਵੈਚਾਲਿਤ it = upscale_pipelines.find(ਕੁੰਜੀ);
        ਜੇ (it != upscale_pipelines.end()) {
            ਵਾਪਸ it->second;
        }

        std::vector<std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ;
        std::string              variant = "upscale";

        ਜੇ (ਕੁੰਜੀ.ਨਿਵੇਸ਼_type == GGML_TYPE_F16) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("SRC_F16");
            variant += "_src_f16";
        } ਨਹੀਂਤਾਂ {
            variant += "_src_f32";
        }

        ਜੇ (ਕੁੰਜੀ.output_type == GGML_TYPE_F16) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("DST_F16");
            variant += "_dst_f16";
        } ਨਹੀਂਤਾਂ {
            variant += "_dst_f32";
        }

        ਚੋਣ (ਅਧਾਰ_ਬਹੁਲਕ) {
            ਹਾਲਤ GGML_SCALE_ਬਹੁਲਕ_ਸਭ_ਤੋਂ_ਨੇੜੇ:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("NEAREST");
                variant += "_nearest";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_SCALE_ਬਹੁਲਕ_BILINEAR:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BILINEAR");
                variant += "_bilinear";
                ਰੋਕੋ;
            ਹਾਲਤ GGML_SCALE_ਬਹੁਲਕ_BICUBIC:
                ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("BICUBIC");
                variant += "_bicubic";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("Unsupported upscale mode");
        }

        ਜੇ (antialias) {
            ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back("ANTIALIAS");
            variant += "_aa";
        }

        ਨਿਰਧਾਰਤ_ਕਰਦਾ.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_ਮਾਪ));

        ਸਵੈਚਾਲਿਤ processed           = preprocessor.preprocess(wgsl_upscale, ਨਿਰਧਾਰਤ_ਕਰਦਾ);
        ਸਵੈਚਾਲਿਤ decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_ਮਾਪ       = context.max_wg_ਮਾਪ;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[ਕੁੰਜੀ]   = pipeline;
        ਵਾਪਸ upscale_pipelines[ਕੁੰਜੀ];
    }

  ਨਿਜੀ:
    ਸਥਿਰ webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_ਕੋਡ,
                                                       std::string    label) {
        wgpu::Shader_ਸਰੋਤ_WGSL shader_ਸਰੋਤ;
        shader_ਸਰੋਤ.code = shader_ਕੋਡ.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.ਅਗਲਾ_ਵਿੱਚ_Chain = &shader_ਸਰੋਤ;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.ਇੰਦਰਾਜ_ਬਿੰਦੂ_2 = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = ਖਾਲੀ_ਸੂਚਕ;  // nullptr means auto layout
        ਵਾਪਸ { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#ਜੇ_ਅੰਤ  // GGML_WEBGPU_SHADER_LIB_HPP
