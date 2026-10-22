#నిర్వచితం_కాకపోతే GGML_WEBGPU_SHADER_LIB_HPP
#నిర్వచించు GGML_WEBGPU_SHADER_LIB_HPP

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-wgsl-shaders.hpp"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "pre_wgsl.hpp"

#అన్తర్భావ <webgpu/webgpu_cpp.h>

#అన్తర్భావ <algorithm>
#అన్తర్భావ <memory>
#అన్తర్భావ <string>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <vector>

#నిర్వచించు GGML_WEBGPU_F16_కొలత_బైట్లు                   2
#నిర్వచించు GGML_WEBGPU_F32_కొలత_బైట్లు                   4
#నిర్వచించు GGML_WEBGPU_I32_కొలత_బైట్లు                   4
#నిర్వచించు GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#నిర్వచించు GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN       20u
#నిర్వచించు GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE       32u
#నిర్వచించు GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE      64u
#నిర్వచించు GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_కొలత     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#నిర్వచించు GGML_WEBGPU_KV_SEQ_PAD                       256u

#నిర్వచించు GGML_WEBGPU_ARGSORT_విలీనం_MAX_WG_కొలత 512u

// Matrix multiplication parameters

// Register tiling parameters
#నిర్వచించు WEBGPU_MUL_MAT_TILE_M           4
#నిర్వచించు WEBGPU_MUL_MAT_TILE_N           4
#నిర్వచించు WEBGPU_MUL_MAT_WG_కొలత_M        8
#నిర్వచించు WEBGPU_MUL_MAT_WG_కొలత_N        8
#నిర్వచించు WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#నిర్వచించు WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#నిర్వచించు WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#నిర్వచించు WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#నిర్వచించు WEBGPU_MUL_MAT_SUBGROUP_మాతృక_M     4
#నిర్వచించు WEBGPU_MUL_MAT_SUBGROUP_మాతృక_N     2
#నిర్వచించు WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#నిర్వచించు WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#నిర్వచించు WEBGPU_MUL_MAT_VEC_WG_కొలత 256

#నిర్వచించు WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_చొప్పున_WG    4
#నిర్వచించు WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_చొప్పున_WG 4
#నిర్వచించు WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_చొప్పున_WG      4

// default size for reg-tile matrix multiplication
#నిర్వచించు WEBGPU_MUL_MAT_WG_కొలత 256

// Same hash combine function as in boost
మూస <రకం_పేరు T> అంతఃవరుస శూన్య ggml_webgpu_hash_combine(size_t & విత్తనం, స్థిరరాశి T & value) {
    విత్తనం ^= std::hash<T>{}(value) + 0x9e3779b9 + (విత్తనం << 6) + (విత్తనం >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
అంతఃవరుస uintptr_t ggml_webgpu_tensor_addr(స్థిరరాశి ggml_tensor * tensor) {
    స్థిరరాశి ggml_tensor * పునాది_tensor = tensor->view_src ? tensor->view_src : tensor;
    తిరిగి (uintptr_t) పునాది_tensor->దత్తాంశం + tensor->view_offs;
}

అంతఃవరుస తార్కిక ggml_webgpu_tensor_equal(స్థిరరాశి ggml_tensor * a, స్థిరరాశి ggml_tensor * b) {
    తిరిగి a->బఫరు == b->బఫరు && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

నిర్మాణం ggml_webgpu_shader_lib_context {
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * src2;
    ggml_tensor * src3;
    ggml_tensor * src4;
    ggml_tensor * src5;
    ggml_tensor * dst;
    ggml_tensor * dst_fuse;

    uint32_t    max_wg_కొలత;
    size_t      wg_mem_హద్దు_బైట్లు       = 0;
    తార్కిక        supports_subgroups       = అబద్ధం;
    తార్కిక        supports_subgroup_మాతృక = అబద్ధం;
    uint32_t    sg_mat_m                 = 0;
    uint32_t    sg_mat_n                 = 0;
    uint32_t    sg_mat_k                 = 0;
    uint32_t    min_subgroup_కొలత        = 0;
    uint32_t    max_subgroup_కొలత        = 0;
    తార్కిక        supports_dot_లబ్ధం     = అబద్ధం;
    std::string vendor;
};

నిర్మాణం webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<శూన్య> context = శూన్యసూచిక;
};

నిర్మాణం ggml_webgpu_generic_shader_decisions {
    uint32_t wg_కొలత = 0;
    తార్కిక     inplace = అబద్ధం;
};

నిర్మాణం ggml_webgpu_get_అడ్డువరుసలు_shader_decisions {
    uint32_t wg_కొలత    = 0;
    తార్కిక     vectorized = అబద్ధం;
};

నిర్మాణం ggml_webgpu_ద్విమాన_shader_decisions {
    uint32_t wg_కొలత     = 0;
    తార్కిక     inplace     = అబద్ధం;
    తార్కిక     overlap     = అబద్ధం;
    తార్కిక     src_overlap = అబద్ధం;
};

నిర్మాణం ggml_webgpu_glu_shader_decisions {
    uint32_t wg_కొలత     = 0;
    తార్కిక     src_overlap = అబద్ధం;
};

నిర్మాణం ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<శూన్య> decisions;
};

నిర్మాణం ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t ఖండం_కొలత_3;
    uint32_t టోకెన్లు_చొప్పున_wg;
};

నిర్మాణం ggml_webgpu_ssm_పరిశీలన_pipeline_కీలకం {
    పూర్ణాంకం     type;
    పూర్ణాంకం     d_state;
    uint8_t xbc_overlap;
    తార్కిక    a_overlap;
    తార్కిక    ids_overlap;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_ssm_పరిశీలన_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap &&
               a_overlap == other.a_overlap && ids_overlap == other.ids_overlap;
    }
};

నిర్మాణం ggml_webgpu_ssm_పరిశీలన_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_ssm_పరిశీలన_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.d_state);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.xbc_overlap);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.a_overlap);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.ids_overlap);
        తిరిగి విత్తనం;
    }
};

నిర్మాణం ggml_webgpu_ssm_పరిశీలన_shader_decisions {
    uint32_t wg_కొలత;
    uint32_t టోకెన్లు_చొప్పున_tile;
    uint8_t  xbc_overlap = 0;
    తార్కిక     a_overlap   = అబద్ధం;
    తార్కిక     ids_overlap = అబద్ధం;
};

/** Argsort **/

నిర్మాణం ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_కొలత;
    size_t   wg_mem_హద్దు_బైట్లు;
    int32_t  క్రమం;
};

/** Set Rows **/

నిర్మాణం ggml_webgpu_సమితి_అడ్డువరుసలు_pipeline_కీలకం {
    పూర్ణాంకం src0_type;
    పూర్ణాంకం dst_type;
    పూర్ణాంకం vec4;
    పూర్ణాంకం i64_idx;
    పూర్ణాంకం pair_ఖండాలు;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_సమితి_అడ్డువరుసలు_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి src0_type == other.src0_type && dst_type == other.dst_type && vec4 == other.vec4 &&
               i64_idx == other.i64_idx && pair_ఖండాలు == other.pair_ఖండాలు;
    }
};

నిర్మాణం ggml_webgpu_సమితి_అడ్డువరుసలు_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_సమితి_అడ్డువరుసలు_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src0_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.dst_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.vec4);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.i64_idx);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.pair_ఖండాలు);
        తిరిగి విత్తనం;
    }
};

నిర్మాణం ggml_webgpu_సమితి_అడ్డువరుసలు_shader_decisions {
    తార్కిక     vec4;
    తార్కిక     i64_idx;
    తార్కిక     pair_ఖండాలు;
    uint32_t wg_కొలత;
};

/** Set **/

నిర్మాణం ggml_webgpu_సమితి_pipeline_కీలకం {
    ggml_type type;
    తార్కిక      inplace;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_సమితి_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి type == other.type && inplace == other.inplace;
    }
};

నిర్మాణం ggml_webgpu_సమితి_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_సమితి_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.inplace);
        తిరిగి విత్తనం;
    }
};

/** Get Rows **/

నిర్మాణం ggml_webgpu_get_అడ్డువరుసలు_pipeline_కీలకం {
    ggml_type src_type;
    పూర్ణాంకం       vectorized;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_get_అడ్డువరుసలు_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి src_type == other.src_type && vectorized == other.vectorized;
    }
};

నిర్మాణం ggml_webgpu_get_అడ్డువరుసలు_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_get_అడ్డువరుసలు_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.vectorized);
        తిరిగి విత్తనం;
    }
};

/** Row Norm **/

నిర్మాణం ggml_webgpu_అడ్డువరుస_norm_pipeline_కీలకం {
    ggml_op   op;
    ggml_type src_type;
    ggml_type dst_type;
    తార్కిక      inplace;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_అడ్డువరుస_norm_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి op == other.op && src_type == other.src_type && dst_type == other.dst_type && inplace == other.inplace;
    }
};

నిర్మాణం ggml_webgpu_అడ్డువరుస_norm_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_అడ్డువరుస_norm_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.op);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.dst_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.inplace);
        తిరిగి విత్తనం;
    }
};

/** RMS_NORM + MUL **/

నిర్మాణం ggml_webgpu_rms_norm_mul_pipeline_కీలకం {
    తార్కిక inplace;      // rn_src == dst
    తార్కిక overlap;      // mul_src == dst
    తార్కిక src_overlap;  // rn_src binding overlaps mul_src binding

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_rms_norm_mul_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

నిర్మాణం ggml_webgpu_rms_norm_mul_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_rms_norm_mul_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.inplace);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.overlap);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src_overlap);
        తిరిగి విత్తనం;
    }
};

నిర్మాణం ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_కొలత     = 0;
    తార్కిక     inplace     = అబద్ధం;
    తార్కిక     overlap     = అబద్ధం;
    తార్కిక     src_overlap = అబద్ధం;
};

/** Pad **/
నిర్మాణం ggml_webgpu_pad_pipeline_కీలకం {
    తార్కిక circular;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_pad_pipeline_కీలకం & other) స్థిరరాశి { తిరిగి circular == other.circular; }
};

నిర్మాణం ggml_webgpu_pad_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_pad_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.circular);
        తిరిగి విత్తనం;
    }
};

/** Solve Tri **/
నిర్మాణం ggml_webgpu_solve_tri_pipeline_కీలకం {
    పూర్ణాంకం type;
    పూర్ణాంకం n;
    పూర్ణాంకం k;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_solve_tri_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి type == other.type && n == other.n && k == other.k;
    }
};

నిర్మాణం ggml_webgpu_solve_tri_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_solve_tri_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.n);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.k);
        తిరిగి విత్తనం;
    }
};

/** SSM Conv **/
నిర్మాణం ggml_webgpu_ssm_conv_pipeline_కీలకం {
    పూర్ణాంకం type;
    పూర్ణాంకం vectorized;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_ssm_conv_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
నిర్మాణం ggml_webgpu_conv2d_pipeline_కీలకం {
    ggml_type బరువు_type;
    ggml_type ఆదానం_type;
    ggml_type output_type;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_conv2d_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి బరువు_type == other.బరువు_type && ఆదానం_type == other.ఆదానం_type && output_type == other.output_type;
    }
};

నిర్మాణం ggml_webgpu_conv2d_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_conv2d_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.బరువు_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.ఆదానం_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.output_type);
        తిరిగి విత్తనం;
    }
};

// Same type fields as conv2d plus the input layout (WHCN vs CWHN).
నిర్మాణం ggml_webgpu_conv2d_dw_pipeline_కీలకం {
    ggml_type బరువు_type;
    ggml_type ఆదానం_type;
    ggml_type output_type;
    తార్కిక      whcn;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_conv2d_dw_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి బరువు_type == other.బరువు_type && ఆదానం_type == other.ఆదానం_type && output_type == other.output_type &&
               whcn == other.whcn;
    }
};

నిర్మాణం ggml_webgpu_conv2d_dw_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_conv2d_dw_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.బరువు_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.ఆదానం_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.output_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.whcn);
        తిరిగి విత్తనం;
    }
};

/** Im2Col **/
నిర్మాణం ggml_webgpu_im2col_pipeline_కీలకం {
    ggml_type ఆదానం_type;
    ggml_type output_type;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_im2col_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి ఆదానం_type == other.ఆదానం_type && output_type == other.output_type;
    }
};

నిర్మాణం ggml_webgpu_im2col_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_im2col_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.ఆదానం_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.output_type);
        తిరిగి విత్తనం;
    }
};

/** Gated Delta Net **/
నిర్మాణం ggml_webgpu_gated_delta_net_pipeline_కీలకం {
    పూర్ణాంకం  type;
    పూర్ణాంకం  s_v;
    పూర్ణాంకం  kda;
    తార్కిక fused_cache;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_gated_delta_net_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి type == other.type && s_v == other.s_v && kda == other.kda && fused_cache == other.fused_cache;
    }
};

నిర్మాణం ggml_webgpu_gated_delta_net_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_gated_delta_net_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.s_v);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.kda);
        తిరిగి విత్తనం;
    }
};

నిర్మాణం ggml_webgpu_ssm_conv_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_ssm_conv_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.vectorized);
        తిరిగి విత్తనం;
    }
};

/** Scale **/

నిర్మాణం ggml_webgpu_scale_pipeline_కీలకం {
    పూర్ణాంకం inplace;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_scale_pipeline_కీలకం & other) స్థిరరాశి { తిరిగి inplace == other.inplace; }
};

నిర్మాణం ggml_webgpu_scale_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_scale_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.inplace);
        తిరిగి విత్తనం;
    }
};

/** Upscale **/

నిర్మాణం ggml_webgpu_upscale_pipeline_కీలకం {
    ggml_type ఆదానం_type;
    ggml_type output_type;
    uint32_t  పునాది_బహుళకం;
    తార్కిక      antialias;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_upscale_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి ఆదానం_type == other.ఆదానం_type && output_type == other.output_type && పునాది_బహుళకం == other.పునాది_బహుళకం &&
               antialias == other.antialias;
    }
};

నిర్మాణం ggml_webgpu_upscale_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_upscale_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.ఆదానం_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.output_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.పునాది_బహుళకం);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.antialias);
        తిరిగి విత్తనం;
    }
};

/** Concat **/

నిర్మాణం ggml_webgpu_concat_pipeline_కీలకం {
    పూర్ణాంకం  type;
    తార్కిక src_overlap;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_concat_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి type == other.type && src_overlap == other.src_overlap;
    }
};

నిర్మాణం ggml_webgpu_concat_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_concat_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src_overlap);
        తిరిగి విత్తనం;
    }
};

/** Repeat **/

నిర్మాణం ggml_webgpu_repeat_pipeline_కీలకం {
    పూర్ణాంకం type;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_repeat_pipeline_కీలకం & other) స్థిరరాశి { తిరిగి type == other.type; }
};

నిర్మాణం ggml_webgpu_repeat_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_repeat_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        తిరిగి విత్తనం;
    }
};

/** Binary **/

నిర్మాణం ggml_webgpu_ద్విమాన_pipeline_కీలకం {
    పూర్ణాంకం  type;
    పూర్ణాంకం  op;
    తార్కిక inplace;
    తార్కిక overlap;
    తార్కిక src_overlap;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_ద్విమాన_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

నిర్మాణం ggml_webgpu_ద్విమాన_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_ద్విమాన_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.op);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.inplace);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.overlap);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src_overlap);
        తిరిగి విత్తనం;
    }
};

/* Add_Id */

నిర్మాణం ggml_webgpu_కూడు_id_pipeline_కీలకం {
    తార్కిక inplace;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_కూడు_id_pipeline_కీలకం & other) స్థిరరాశి { తిరిగి inplace == other.inplace; }
};

నిర్మాణం ggml_webgpu_కూడు_id_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_కూడు_id_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.inplace);
        తిరిగి విత్తనం;
    }
};

/** Unary **/

నిర్మాణం ggml_webgpu_unary_pipeline_కీలకం {
    పూర్ణాంకం           type;
    పూర్ణాంకం           op;
    తార్కిక          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    తార్కిక          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_unary_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

నిర్మాణం ggml_webgpu_unary_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_unary_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.op);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.is_unary);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.inplace);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.ttype);
        తిరిగి విత్తనం;
    }
};

/** FlashAttention */

నిర్మాణం ggml_webgpu_flash_attn_common_pipeline_కీలకం {
    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;
    ggml_type dst_type;
    uint32_t  తల_dim_qk;
    uint32_t  తల_dim_v;
    తార్కిక      k_ప్రత్యక్ష;
    తార్కిక      v_ప్రత్యక్ష;
    తార్కిక      kv_overlap;
    తార్కిక      has_ముసుగు;
    తార్కిక      has_sinks;
    తార్కిక      ఉపయోగిస్తుంది_logit_softcap;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_flash_attn_common_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి q_type == other.q_type && k_type == other.k_type && v_type == other.v_type &&
               dst_type == other.dst_type && తల_dim_qk == other.తల_dim_qk && తల_dim_v == other.తల_dim_v &&
               k_ప్రత్యక్ష == other.k_ప్రత్యక్ష && v_ప్రత్యక్ష == other.v_ప్రత్యక్ష && kv_overlap == other.kv_overlap &&
               has_ముసుగు == other.has_ముసుగు && has_sinks == other.has_sinks &&
               ఉపయోగిస్తుంది_logit_softcap == other.ఉపయోగిస్తుంది_logit_softcap;
    }
};

అంతఃవరుస శూన్య ggml_webgpu_flash_attn_hash_common_pipeline_కీలకం(size_t &                                           విత్తనం,
                                                            స్థిరరాశి ggml_webgpu_flash_attn_common_pipeline_కీలకం & కీలకం) {
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.q_type);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.k_type);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.v_type);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.dst_type);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.తల_dim_qk);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.తల_dim_v);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.k_ప్రత్యక్ష);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.v_ప్రత్యక్ష);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.kv_overlap);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.has_ముసుగు);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.has_sinks);
    ggml_webgpu_hash_combine(విత్తనం, కీలకం.ఉపయోగిస్తుంది_logit_softcap);
}

నిర్మాణం ggml_webgpu_flash_attn_vec_pipeline_కీలకం {
    ggml_webgpu_flash_attn_common_pipeline_కీలకం common;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_flash_attn_vec_pipeline_కీలకం & other) స్థిరరాశి { తిరిగి common == other.common; }
};

నిర్మాణం ggml_webgpu_flash_attn_vec_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_flash_attn_vec_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_కీలకం(విత్తనం, కీలకం.common);
        తిరిగి విత్తనం;
    }
};

నిర్మాణం ggml_webgpu_flash_attn_pipeline_కీలకం {
    ggml_webgpu_flash_attn_common_pipeline_కీలకం common;
    తార్కిక                                       use_sg_మాతృక;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_flash_attn_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి common == other.common && use_sg_మాతృక == other.use_sg_మాతృక;
    }
};

నిర్మాణం ggml_webgpu_flash_attn_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_flash_attn_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_flash_attn_hash_common_pipeline_కీలకం(విత్తనం, కీలకం.common);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.use_sg_మాతృక);
        తిరిగి విత్తనం;
    }
};

నిర్మాణం ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_కొలత = 0;
};

నిర్మాణం ggml_webgpu_flash_attn_decisions {
    తార్కిక     use_sg_మాతృక = అబద్ధం;
    uint32_t q_tile        = 0;
    uint32_t kv_tile       = 0;
    uint32_t wg_కొలత       = 0;
};

అంతఃవరుస constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_వెడల్పు = 4u;
అంతఃవరుస constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

అంతఃవరుస size_t ggml_webgpu_flash_attn_tensor_స్థానభ్రంశం(స్థిరరాశి ggml_tensor * tensor) {
    constexpr uintptr_t ptr_పునాది_addr = 0x1000u;
    స్థిరరాశి ggml_tensor * పునాది          = tensor->view_src != శూన్యసూచిక ? tensor->view_src : tensor;
    తిరిగి reinterpret_cast<uintptr_t>(పునాది->దత్తాంశం) - ptr_పునాది_addr + tensor->view_offs;
}

అంతఃవరుస తార్కిక ggml_webgpu_flash_attn_float_vec4_aligned(స్థిరరాశి ggml_tensor * K, size_t storage_స్థానభ్రంశం_alignment) {
    స్థిరరాశి uint32_t స్థానభ్రంశం_elems =
        (uint32_t) ((ggml_webgpu_flash_attn_tensor_స్థానభ్రంశం(K) & (storage_స్థానభ్రంశం_alignment - 1)) /
                    ggml_type_కొలత(K->type));
    తిరిగి స్థానభ్రంశం_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_వెడల్పు == 0u;
}

అంతఃవరుస తార్కిక ggml_webgpu_flash_attn_float_vec4_aligned(స్థిరరాశి ggml_tensor * K,
                                                      స్థిరరాశి ggml_tensor * V,
                                                      size_t              storage_స్థానభ్రంశం_alignment) {
    తిరిగి ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_స్థానభ్రంశం_alignment) &&
           ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_స్థానభ్రంశం_alignment);
}

అంతఃవరుస తార్కిక ggml_webgpu_flash_attn_k_ప్రత్యక్ష(స్థిరరాశి ggml_tensor * Q, స్థిరరాశి ggml_tensor * K, uint32_t kv_ప్రత్యక్ష_align) {
    తిరిగి (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
           (Q->ne[0] % kv_ప్రత్యక్ష_align == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
}

అంతఃవరుస తార్కిక ggml_webgpu_flash_attn_v_ప్రత్యక్ష(స్థిరరాశి ggml_tensor * Q, స్థిరరాశి ggml_tensor * V, uint32_t kv_ప్రత్యక్ష_align) {
    తిరిగి ggml_webgpu_flash_attn_k_ప్రత్యక్ష(Q, V, kv_ప్రత్యక్ష_align);
}

అంతఃవరుస ggml_webgpu_flash_attn_common_pipeline_కీలకం ggml_webgpu_flash_attn_make_common_pipeline_కీలకం(
    స్థిరరాశి ggml_webgpu_shader_lib_context & context,
    uint32_t                               kv_ప్రత్యక్ష_align,
    తార్కిక                                   kv_overlap) {
    ggml_webgpu_flash_attn_common_pipeline_కీలకం కీలకం = {};
    కీలకం.q_type                                     = context.src0->type;
    కీలకం.k_type                                     = context.src1->type;
    కీలకం.v_type                                     = context.src2->type;
    కీలకం.dst_type                                   = context.dst->type;
    కీలకం.తల_dim_qk                                = (uint32_t) context.src0->ne[0];
    కీలకం.తల_dim_v                                 = (uint32_t) context.src2->ne[0];
    కీలకం.k_ప్రత్యక్ష           = ggml_webgpu_flash_attn_k_ప్రత్యక్ష(context.src0, context.src1, kv_ప్రత్యక్ష_align);
    కీలకం.v_ప్రత్యక్ష           = ggml_webgpu_flash_attn_v_ప్రత్యక్ష(context.src0, context.src2, kv_ప్రత్యక్ష_align);
    కీలకం.kv_overlap         = kv_overlap;
    కీలకం.has_ముసుగు           = context.src3 != శూన్యసూచిక;
    కీలకం.has_sinks          = context.src4 != శూన్యసూచిక;
    కీలకం.ఉపయోగిస్తుంది_logit_softcap = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    తిరిగి కీలకం;
}

అంతఃవరుస std::vector<std::string> ggml_webgpu_flash_attn_common_నిర్వచిస్తుంది(
    స్థిరరాశి ggml_webgpu_flash_attn_common_pipeline_కీలకం & కీలకం,
    std::string &                                      variant,
    uint32_t                                           q_tile,
    uint32_t                                           kv_tile,
    uint32_t                                           wg_కొలత) {
    std::vector<std::string> నిర్వచిస్తుంది;

    ఎంపిక (కీలకం.k_type) {
        స్థితి GGML_TYPE_F32:
            నిర్వచిస్తుంది.push_back("K_F32");
            ఆపు;
        స్థితి GGML_TYPE_F16:
            నిర్వచిస్తుంది.push_back("K_F16");
            ఆపు;
        స్థితి GGML_TYPE_Q4_0:
            నిర్వచిస్తుంది.push_back("K_Q4_0");
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            నిర్వచిస్తుంది.push_back("K_Q8_0");
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("Unsupported K type for flash attention shader");
    }
    variant += std::string("_k") + ggml_type_name(కీలకం.k_type);

    ఎంపిక (కీలకం.v_type) {
        స్థితి GGML_TYPE_F32:
            నిర్వచిస్తుంది.push_back("V_F32");
            ఆపు;
        స్థితి GGML_TYPE_F16:
            నిర్వచిస్తుంది.push_back("V_F16");
            ఆపు;
        స్థితి GGML_TYPE_Q4_0:
            నిర్వచిస్తుంది.push_back("V_Q4_0");
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            నిర్వచిస్తుంది.push_back("V_Q8_0");
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("Unsupported V type for flash attention shader");
    }
    variant += std::string("_v") + ggml_type_name(కీలకం.v_type);

    ఎంపిక (కీలకం.q_type) {
        స్థితి GGML_TYPE_F32:
            నిర్వచిస్తుంది.push_back("Q_F32");
            ఆపు;
        స్థితి GGML_TYPE_F16:
            నిర్వచిస్తుంది.push_back("Q_F16");
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("Unsupported Q type for flash attention shader");
    }
    variant += std::string("_q") + ggml_type_name(కీలకం.q_type);

    ఎంపిక (కీలకం.dst_type) {
        స్థితి GGML_TYPE_F32:
            నిర్వచిస్తుంది.push_back("DST_F32");
            ఆపు;
        స్థితి GGML_TYPE_F16:
            నిర్వచిస్తుంది.push_back("DST_F16");
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("Unsupported dst type for flash attention shader");
    }
    variant += std::string("_dst") + ggml_type_name(కీలకం.dst_type);

    ఒకవేళ (కీలకం.has_ముసుగు) {
        నిర్వచిస్తుంది.push_back("MASK");
        variant += "_mask";
    }
    ఒకవేళ (కీలకం.has_sinks) {
        నిర్వచిస్తుంది.push_back("SINKS");
        variant += "_sinks";
    }
    ఒకవేళ (కీలకం.ఉపయోగిస్తుంది_logit_softcap) {
        నిర్వచిస్తుంది.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }
    ఒకవేళ (కీలకం.k_ప్రత్యక్ష) {
        నిర్వచిస్తుంది.push_back("K_DIRECT");
        variant += "_k_direct";
    }
    ఒకవేళ (కీలకం.v_ప్రత్యక్ష) {
        నిర్వచిస్తుంది.push_back("V_DIRECT");
        variant += "_v_direct";
    }
    ఒకవేళ (కీలకం.kv_overlap) {
        నిర్వచిస్తుంది.push_back("KV_OVERLAP");
        variant += "_kv_overlap";
    }

    నిర్వచిస్తుంది.push_back(std::string("HEAD_DIM_QK=") + std::to_string(కీలకం.తల_dim_qk));
    variant += std::string("_hsqk") + std::to_string(కీలకం.తల_dim_qk);

    నిర్వచిస్తుంది.push_back(std::string("HEAD_DIM_V=") + std::to_string(కీలకం.తల_dim_v));
    variant += std::string("_hsv") + std::to_string(కీలకం.తల_dim_v);

    నిర్వచిస్తుంది.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    నిర్వచిస్తుంది.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
    నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(wg_కొలత));

    ఒకవేళ (ggml_is_quantized(కీలకం.k_type) || ggml_is_quantized(కీలకం.v_type)) {
        నిర్వచిస్తుంది.push_back("U32_DEQUANT_HELPERS");
        ఒకవేళ (ggml_is_quantized(కీలకం.k_type)) {
            నిర్వచిస్తుంది.push_back("LOADERS_QUANTIZED_K");
        }
        ఒకవేళ (ggml_is_quantized(కీలకం.v_type)) {
            నిర్వచిస్తుంది.push_back("LOADERS_QUANTIZED_V");
        }
    }

    తిరిగి నిర్వచిస్తుంది;
}

నిర్మాణం ggml_webgpu_flash_attn_vec_reduce_pipeline_కీలకం {
    uint32_t  తల_dim_v;
    uint32_t  wg_కొలత;
    ggml_type dst_type;
};

నిర్మాణం ggml_webgpu_flash_attn_vec_reduce_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_flash_attn_vec_reduce_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.తల_dim_v);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.wg_కొలత);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.dst_type);
        తిరిగి విత్తనం;
    }
};

అంతఃవరుస తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_flash_attn_vec_reduce_pipeline_కీలకం & lhs,
                       స్థిరరాశి ggml_webgpu_flash_attn_vec_reduce_pipeline_కీలకం & rhs) {
    తిరిగి lhs.తల_dim_v == rhs.తల_dim_v && lhs.wg_కొలత == rhs.wg_కొలత && lhs.dst_type == rhs.dst_type;
}

నిర్మాణం ggml_webgpu_flash_attn_blk_pipeline_కీలకం {
    uint32_t kv_tile;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_flash_attn_blk_pipeline_కీలకం & other) స్థిరరాశి { తిరిగి kv_tile == other.kv_tile; }
};

నిర్మాణం ggml_webgpu_flash_attn_blk_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_flash_attn_blk_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.kv_tile);
        తిరిగి విత్తనం;
    }
};

// Note: this will slightly overestimate memory usage for vec path
// since row_max and exp_sum shmem are not needed.
అంతఃవరుస size_t ggml_webgpu_flash_attn_wg_mem_బైట్లు(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t తల_dim_qk,
                                                  uint32_t తల_dim_v,
                                                  తార్కిక     has_ముసుగు,
                                                  తార్కిక     kv_ప్రత్యక్ష) {
    స్థిరరాశి uint32_t max_తల_dim = std::max(తల_dim_qk, తల_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;

    f32_elems += q_tile * తల_dim_qk;        // q_shmem
    ఒకవేళ (!kv_ప్రత్యక్ష) {
        f32_elems += kv_tile * max_తల_dim;  // kv_shmem
    }
    f32_elems += q_tile * తల_dim_v;         // o_shmem
    ఒకవేళ (has_ముసుగు) {
        f32_elems += q_tile * kv_tile;        // mask_shmem
    }
    f32_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    తిరిగి f16_elems * GGML_WEBGPU_F16_కొలత_బైట్లు + f32_elems * GGML_WEBGPU_F32_కొలత_బైట్లు;
}

అంతఃవరుస uint32_t ggml_webgpu_flash_attn_max_kv_tile(size_t   హద్దు_బైట్లు,
                                                   uint32_t q_tile,
                                                   uint32_t kv_granularity,
                                                   uint32_t తల_dim_qk,
                                                   uint32_t తల_dim_v,
                                                   తార్కిక     has_ముసుగు,
                                                   తార్కిక     kv_ప్రత్యక్ష) {
    స్థిరరాశి size_t పునాది_q_బైట్లు =
        ggml_webgpu_flash_attn_wg_mem_బైట్లు(q_tile, 0, తల_dim_qk, తల_dim_v, has_ముసుగు, kv_ప్రత్యక్ష);
    ఒకవేళ (హద్దు_బైట్లు <= పునాది_q_బైట్లు) {
        తిరిగి 0;
    }
    స్థిరరాశి size_t one_kv_బైట్లు =
        ggml_webgpu_flash_attn_wg_mem_బైట్లు(q_tile, 1, తల_dim_qk, తల_dim_v, has_ముసుగు, kv_ప్రత్యక్ష);
    స్థిరరాశి size_t బైట్లు_చొప్పున_kv = one_kv_బైట్లు - పునాది_q_బైట్లు;
    ఒకవేళ (బైట్లు_చొప్పున_kv == 0) {
        తిరిగి 0;
    }
    స్థిరరాశి size_t max_kv_tile = (హద్దు_బైట్లు - పునాది_q_బైట్లు) / బైట్లు_చొప్పున_kv;
    తిరిగి (uint32_t) ((max_kv_tile / kv_granularity) * kv_granularity);
}

అంతఃవరుస uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(size_t   wg_mem_హద్దు_బైట్లు,
                                                       uint32_t తల_dim_qk,
                                                       uint32_t తల_dim_v,
                                                       తార్కిక     has_ముసుగు,
                                                       తార్కిక     kv_ప్రత్యక్ష) {
    స్థిరరాశి uint32_t max_kv_tile =
        ggml_webgpu_flash_attn_max_kv_tile(wg_mem_హద్దు_బైట్లు, 1u, 1u, తల_dim_qk, తల_dim_v, has_ముసుగు, kv_ప్రత్యక్ష);
    GGML_ASSERT(max_kv_tile > 0);

    uint32_t kv_tile = std::min(GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE, max_kv_tile);
    ఒకవేళ (kv_ప్రత్యక్ష) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        అయితే (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= 1u;
        }
    }

    తిరిగి kv_tile;
}

అంతఃవరుస తార్కిక ggml_webgpu_flash_attn_can_use_subgroup_మాతృక_మార్గం(తార్కిక                supports_subgroup_మాతృక,
                                                                uint32_t            sg_mat_k,
                                                                uint32_t            sg_mat_n,
                                                                స్థిరరాశి ggml_tensor * Q,
                                                                స్థిరరాశి ggml_tensor * V) {
    తిరిగి supports_subgroup_మాతృక && Q->ne[0] % sg_mat_k == 0 && V->ne[0] % sg_mat_n == 0;
}

/** Matrix Multiplication **/

నిర్మాణం ggml_webgpu_mul_mat_vec_pipeline_కీలకం {
    ggml_type src0_type;
    ggml_type src1_type;
    పూర్ణాంకం       vectorized;
    uint32_t  అంకె_నిలువుల_సంఖ్య;
    తార్కిక      use_mmvq;
    తార్కిక      src_overlap;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_mul_mat_vec_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               అంకె_నిలువుల_సంఖ్య == other.అంకె_నిలువుల_సంఖ్య && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
    }
};

నిర్మాణం ggml_webgpu_mul_mat_vec_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_mul_mat_vec_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src0_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src1_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.vectorized);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.అంకె_నిలువుల_సంఖ్య);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.use_mmvq);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src_overlap);
        తిరిగి విత్తనం;
    }
};

నిర్మాణం ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_కొలత;
    uint32_t outputs_చొప్పున_wg;
    uint32_t vec_కొలత;
    తార్కిక     src_overlap = అబద్ధం;
};

నిర్మాణం ggml_webgpu_quantize_q8_pipeline_కీలకం {
    ggml_type src0_type;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_quantize_q8_pipeline_కీలకం & other) స్థిరరాశి { తిరిగి src0_type == other.src0_type; }
};

నిర్మాణం ggml_webgpu_quantize_q8_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_quantize_q8_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src0_type);
        తిరిగి విత్తనం;
    }
};

నిర్మాణం ggml_webgpu_mul_mat_pipeline_కీలకం {
    ggml_type src0_type;
    ggml_type src1_type;
    పూర్ణాంకం       vectorized;
    పూర్ణాంకం       use_subgroup_మాతృక;
    తార్కిక      src_overlap;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_mul_mat_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_మాతృక == other.use_subgroup_మాతృక && src_overlap == other.src_overlap;
    }
};

నిర్మాణం ggml_webgpu_mul_mat_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_mul_mat_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src0_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src1_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.vectorized);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.use_subgroup_మాతృక);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src_overlap);
        తిరిగి విత్తనం;
    }
};

నిర్మాణం ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_కొలత_m;
    uint32_t wg_కొలత_n;
    uint32_t wg_కొలత;
    uint32_t outputs_చొప్పున_wg;
    పూర్ణాంకం      use_subgroup_మాతృక;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_మాతృక_m;
    uint32_t subgroup_మాతృక_n;

    uint32_t mul_mat_wg_కొలత;
    తార్కిక     src_overlap = అబద్ధం;
};

/** MUL_MAT_ID **/

నిర్మాణం ggml_webgpu_mul_mat_id_gather_pipeline_కీలకం {
    uint32_t n_experts;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_mul_mat_id_gather_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి n_experts == other.n_experts;
    }
};

నిర్మాణం ggml_webgpu_mul_mat_id_gather_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_mul_mat_id_gather_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.n_experts);
        తిరిగి విత్తనం;
    }
};

నిర్మాణం ggml_webgpu_mul_mat_id_pipeline_కీలకం {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    uint32_t  అంకె_నిలువుల_సంఖ్య;
    పూర్ణాంకం       vectorized;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_mul_mat_id_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               అంకె_నిలువుల_సంఖ్య == other.అంకె_నిలువుల_సంఖ్య && vectorized == other.vectorized;
    }
};

నిర్మాణం ggml_webgpu_mul_mat_id_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_mul_mat_id_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src0_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src1_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.n_experts);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.అంకె_నిలువుల_సంఖ్య);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.vectorized);
        తిరిగి విత్తనం;
    }
};

/** Cpy **/

నిర్మాణం ggml_webgpu_cpy_pipeline_కీలకం {
    ggml_type src_type;
    ggml_type dst_type;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_cpy_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి src_type == other.src_type && dst_type == other.dst_type;
    }
};

నిర్మాణం ggml_webgpu_cpy_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_cpy_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.dst_type);
        తిరిగి విత్తనం;
    }
};

/** Glu **/

నిర్మాణం ggml_webgpu_glu_pipeline_కీలకం {
    ggml_glu_op glu_op;
    ggml_type   type;
    తార్కిక        విభజించు;
    తార్కిక        src_overlap;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_glu_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి glu_op == other.glu_op && type == other.type && విభజించు == other.విభజించు && src_overlap == other.src_overlap;
    }
};

నిర్మాణం ggml_webgpu_glu_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_glu_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.glu_op);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.విభజించు);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.src_overlap);
        తిరిగి విత్తనం;
    }
};

/** Rope **/

నిర్మాణం ggml_webgpu_rope_pipeline_కీలకం {
    ggml_type type;
    తార్కిక      inplace;
    తార్కిక      has_ff;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_rope_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

నిర్మాణం ggml_webgpu_rope_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_rope_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.inplace);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.has_ff);
        తిరిగి విత్తనం;
    }
};

/** SoftMax **/

నిర్మాణం ggml_webgpu_soft_max_pipeline_కీలకం {
    ggml_type ముసుగు_type;
    తార్కిక      has_ముసుగు;
    తార్కిక      has_sink;
    తార్కిక      inplace;

    తార్కిక క్రియాకారకం==(స్థిరరాశి ggml_webgpu_soft_max_pipeline_కీలకం & other) స్థిరరాశి {
        తిరిగి ముసుగు_type == other.ముసుగు_type && has_ముసుగు == other.has_ముసుగు && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

నిర్మాణం ggml_webgpu_soft_max_pipeline_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి ggml_webgpu_soft_max_pipeline_కీలకం & కీలకం) స్థిరరాశి {
        size_t విత్తనం = 0;
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.ముసుగు_type);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.has_ముసుగు);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.has_sink);
        ggml_webgpu_hash_combine(విత్తనం, కీలకం.inplace);
        తిరిగి విత్తనం;
    }
};

/** MMVQ **/

అంతఃవరుస తార్కిక ggml_webgpu_can_use_mmvq(స్థిరరాశి ggml_tensor * src0,
                                     స్థిరరాశి ggml_tensor * src1,
                                     తార్కిక                supports_dot_లబ్ధం,
                                     స్థిరరాశి std::string & vendor) {
    ఒకవేళ (src1->ne[1] <= 4) {
        తార్కిక supports_dp4a = vendor == "amd" || vendor == "intel" || vendor == "nvidia";
        ఒకవేళ (supports_dp4a && supports_dot_లబ్ధం) {
            ఎంపిక (src1->type) {
                స్థితి GGML_TYPE_F32:
                    ఎంపిక (src0->type) {
                        స్థితి GGML_TYPE_Q1_0:
                        స్థితి GGML_TYPE_Q4_0:
                        స్థితి GGML_TYPE_Q4_1:
                        స్థితి GGML_TYPE_Q5_0:
                        స్థితి GGML_TYPE_Q5_1:
                        స్థితి GGML_TYPE_Q8_0:
                        స్థితి GGML_TYPE_MXFP4:
                        స్థితి GGML_TYPE_Q2_K:
                        స్థితి GGML_TYPE_Q3_K:
                        స్థితి GGML_TYPE_Q4_K:
                        స్థితి GGML_TYPE_Q5_K:
                        స్థితి GGML_TYPE_Q6_K:
                            తిరిగి src0->ne[0] % 4 == 0;
                        అప్రమేయ:
                            ఆపు;
                    }
                    ఆపు;
                అప్రమేయ:
                    ఆపు;
            }
        }
    }
    తిరిగి అబద్ధం;
}

తరగతి ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<పూర్ణాంకం, webgpu_pipeline> కూడిక_అడ్డువరుసలు_pipelines;       // key is fixed, no variants yet
    std::unordered_map<పూర్ణాంకం, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<పూర్ణాంకం, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<పూర్ణాంకం, webgpu_pipeline> argsort_విలీనం_pipelines;  // key is order
    std::unordered_map<పూర్ణాంకం, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_అడ్డువరుస_norm_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_అడ్డువరుస_norm_pipeline_కీలకం_hash>
        అడ్డువరుస_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_అడ్డువరుసలు_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_get_అడ్డువరుసలు_pipeline_కీలకం_hash>
        get_అడ్డువరుసలు_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_unary_pipeline_కీలకం_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_scale_pipeline_కీలకం_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_కీలకం_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_కీలకం_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_పరిశీలన_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_ssm_పరిశీలన_pipeline_కీలకం_hash>
        ssm_పరిశీలన_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_కీలకం,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_కీలకం_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_pad_pipeline_కీలకం_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_ద్విమాన_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_ద్విమాన_pipeline_కీలకం_hash>
        ద్విమాన_pipelines;           // type/op/inplace/overlap/src_overlap
    std::unordered_map<ggml_webgpu_కూడు_id_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_కూడు_id_pipeline_కీలకం_hash>
        కూడు_id_pipelines;           // inplace
    std::unordered_map<ggml_webgpu_concat_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_concat_pipeline_కీలకం_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_repeat_pipeline_కీలకం_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_vec_pipeline_కీలకం,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_pipeline_కీలకం_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_కీలకం_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_కీలకం,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_కీలకం_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_కీలకం,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_కీలకం_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_కీలకం_hash>
        mul_mat_vec_pipelines;   // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_కీలకం_hash>
        mul_mat_fast_pipelines;  // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<ggml_webgpu_quantize_q8_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_quantize_q8_pipeline_కీలకం_hash>
        quantize_q8_pipelines;
    std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_కీలకం,
                       webgpu_pipeline,
                       ggml_webgpu_mul_mat_id_gather_pipeline_కీలకం_hash>
        mul_mat_id_gather_pipelines;  // n_experts
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_కీలకం_hash>
        mul_mat_id_pipelines;         // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_కీలకం_hash>
        mul_mat_id_vec_pipelines;     // src0_type/src1_type

    std::unordered_map<ggml_webgpu_సమితి_అడ్డువరుసలు_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_సమితి_అడ్డువరుసలు_pipeline_కీలకం_hash>
        సమితి_అడ్డువరుసలు_pipelines;
    std::unordered_map<ggml_webgpu_సమితి_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_సమితి_pipeline_కీలకం_hash> సమితి_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_cpy_pipeline_కీలకం_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_glu_pipeline_కీలకం_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_rope_pipeline_కీలకం_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_కీలకం_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_కీలకం_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_dw_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_conv2d_dw_pipeline_కీలకం_hash>
        conv2d_dw_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_im2col_pipeline_కీలకం_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_కీలకం,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_కీలకం_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_కీలకం, webgpu_pipeline, ggml_webgpu_upscale_pipeline_కీలకం_hash>
        upscale_pipelines;

  బహిరంగ:
    ggml_webgpu_shader_lib(wgpu::Device device) { ఇది->device = device; }

    webgpu_pipeline get_కూడిక_అడ్డువరుసలు_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        స్వయంచాలక it = కూడిక_అడ్డువరుసలు_pipelines.find(1);
        ఒకవేళ (it != కూడిక_అడ్డువరుసలు_pipelines.end()) {
            తిరిగి it->second;
        }
        std::vector<std::string> నిర్వచిస్తుంది;
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed        = preprocessor.preprocess(wgsl_కూడిక_అడ్డువరుసలు, నిర్వచిస్తుంది);
        కూడిక_అడ్డువరుసలు_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        తిరిగి కూడిక_అడ్డువరుసలు_pipelines[1];
    }

    webgpu_pipeline get_అడ్డువరుస_norm_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_అడ్డువరుస_norm_pipeline_కీలకం కీలకం = {};
        కీలకం.op                                = context.dst->op;
        కీలకం.src_type                          = context.src0->type;
        కీలకం.dst_type                          = context.dst->type;
        కీలకం.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        స్వయంచాలక it = అడ్డువరుస_norm_pipelines.find(కీలకం);
        ఒకవేళ (it != అడ్డువరుస_norm_pipelines.end()) {
            తిరిగి it->second;
        }
        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant;

        ఎంపిక (కీలకం.op) {
            స్థితి GGML_OP_RMS_NORM:
                నిర్వచిస్తుంది.push_back("RMS_NORM");
                variant = "rms_norm";
                ఆపు;
            స్థితి GGML_OP_NORM:
                నిర్వచిస్తుంది.push_back("NORM");
                variant = "norm";
                ఆపు;
            స్థితి GGML_OP_L2_NORM:
                నిర్వచిస్తుంది.push_back("L2_NORM");
                variant = "l2_norm";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        ఒకవేళ (కీలకం.inplace) {
            నిర్వచిస్తుంది.push_back("INPLACE");
            variant += "_inplace";
        }

        ఒకవేళ (కీలకం.src_type == GGML_TYPE_F32) {
            నిర్వచిస్తుంది.push_back("SRC_F32");
            variant += "_src_f32";
        } లేకపోతే ఒకవేళ (కీలకం.src_type == GGML_TYPE_F16) {
            నిర్వచిస్తుంది.push_back("SRC_F16");
            variant += "_src_f16";
        }

        ఒకవేళ (కీలకం.dst_type == GGML_TYPE_F32) {
            నిర్వచిస్తుంది.push_back("DST_F32");
            variant += "_dst_f32";
        } లేకపోతే ఒకవేళ (కీలకం.dst_type == GGML_TYPE_F16) {
            నిర్వచిస్తుంది.push_back("DST_F16");
            variant += "_dst_f16";
        }

        స్థిరరాశి uint32_t అడ్డువరుస_norm_wg_కొలత = 128u;
        uint32_t       wg_కొలత          = std::min(context.max_wg_కొలత, అడ్డువరుస_norm_wg_కొలత);
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(wg_కొలత));
        స్వయంచాలక processed                  = preprocessor.preprocess(wgsl_అడ్డువరుస_norm, నిర్వచిస్తుంది);
        స్వయంచాలక decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత              = wg_కొలత;
        decisions->inplace              = కీలకం.inplace;
        అడ్డువరుస_norm_pipelines[కీలకం]         = ggml_webgpu_create_pipeline(device, processed, variant);
        అడ్డువరుస_norm_pipelines[కీలకం].context = decisions;
        తిరిగి అడ్డువరుస_norm_pipelines[కీలకం];
    }

    webgpu_pipeline get_argmax_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        తార్కిక vec4 = context.src0->ne[0] % 4 == 0;

        స్వయంచాలక it = argmax_pipelines.find(vec4);
        ఒకవేళ (it != argmax_pipelines.end()) {
            తిరిగి it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> నిర్వచిస్తుంది;
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));
        ఒకవేళ (vec4) {
            నిర్వచిస్తుంది.push_back("VEC4");
            variant += "_vec4";
        }

        స్వయంచాలక processed         = preprocessor.preprocess(wgsl_argmax, నిర్వచిస్తుంది);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        తిరిగి argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_సమితి_అడ్డువరుసలు_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        స్థిరరాశి తార్కిక                        quantized = ggml_is_quantized(context.dst->type);
        ggml_webgpu_సమితి_అడ్డువరుసలు_pipeline_కీలకం కీలకం       = {};
        కీలకం.src0_type                               = context.src0->type;
        కీలకం.dst_type                                = context.dst->type;
        కీలకం.vec4        = (context.dst->type == GGML_TYPE_F32 || context.dst->type == GGML_TYPE_F16) &&
                          context.src0->type == GGML_TYPE_F32 && context.src0->ne[0] % 4 == 0;
        కీలకం.i64_idx     = context.src1->type == GGML_TYPE_I64;
        కీలకం.pair_ఖండాలు = quantized && ((context.src0->ne[0] / ggml_blck_కొలత(context.dst->type)) % 2 == 0);

        స్వయంచాలక it = సమితి_అడ్డువరుసలు_pipelines.find(కీలకం);
        ఒకవేళ (it != సమితి_అడ్డువరుసలు_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "set_rows";

        ఎంపిక (context.dst->type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("DST_F32");
                variant += "_dstf32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("DST_F16");
                variant += "_dstf16";
                ఆపు;
            స్థితి GGML_TYPE_Q8_0:
                నిర్వచిస్తుంది.push_back("DST_Q8_0");
                variant += "_dstq8_0";
                ఆపు;
            స్థితి GGML_TYPE_Q4_0:
                నిర్వచిస్తుంది.push_back("DST_Q4_0");
                variant += "_dstq4_0";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        ఒకవేళ (context.src0->type == GGML_TYPE_F16) {
            నిర్వచిస్తుంది.push_back("TYPE_F16");
            variant += "_src0_f16";
        }

        ఒకవేళ (కీలకం.vec4) {
            నిర్వచిస్తుంది.push_back("VEC4");
            variant += "_vec4";
        }
        ఒకవేళ (కీలకం.i64_idx) {
            నిర్వచిస్తుంది.push_back("I64_IDX");
            variant += "_i64idx";
        }
        ఒకవేళ (కీలకం.pair_ఖండాలు) {
            నిర్వచిస్తుంది.push_back("PAIR_BLOCKS");
            variant += "_pair_blocks";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్థిరరాశి స్వయంచాలక & shader_ఆధారం      = quantized ? wgsl_సమితి_అడ్డువరుసలు_quant : wgsl_సమితి_అడ్డువరుసలు;
        స్వయంచాలక         processed          = preprocessor.preprocess(shader_ఆధారం, నిర్వచిస్తుంది);
        స్వయంచాలక         decisions          = std::make_shared<ggml_webgpu_సమితి_అడ్డువరుసలు_shader_decisions>();
        decisions->vec4                 = కీలకం.vec4;
        decisions->i64_idx              = కీలకం.i64_idx;
        decisions->pair_ఖండాలు          = కీలకం.pair_ఖండాలు;
        decisions->wg_కొలత              = context.max_wg_కొలత;
        సమితి_అడ్డువరుసలు_pipelines[కీలకం]         = ggml_webgpu_create_pipeline(device, processed, variant);
        సమితి_అడ్డువరుసలు_pipelines[కీలకం].context = decisions;
        తిరిగి సమితి_అడ్డువరుసలు_pipelines[కీలకం];
    }

    webgpu_pipeline get_సమితి_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_సమితి_pipeline_కీలకం కీలకం = {};
        కీలకం.type                         = context.dst->type;
        కీలకం.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        స్వయంచాలక it = సమితి_pipelines.find(కీలకం);
        ఒకవేళ (it != సమితి_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "set";

        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_I32:
                నిర్వచిస్తుంది.push_back("TYPE_I32");
                variant += "_i32";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for set shader");
        }

        ఒకవేళ (కీలకం.inplace) {
            నిర్వచిస్తుంది.push_back("INPLACE");
            variant += "_inplace";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_సమితి, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        decisions->inplace       = కీలకం.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        సమితి_pipelines[కీలకం]       = pipeline;
        తిరిగి సమితి_pipelines[కీలకం];
    }

    webgpu_pipeline get_cumsum_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        స్వయంచాలక it = cumsum_pipelines.find(1);
        ఒకవేళ (it != cumsum_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed      = preprocessor.preprocess(wgsl_cumsum, నిర్వచిస్తుంది);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        తిరిగి cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        తార్కిక          is_శిఖరం_k = context.dst->op == GGML_OP_శిఖరం_K;
        // ascending order is 0, descending order is 1
        స్థిరరాశి int32_t క్రమం =
            is_శిఖరం_k ? (int32_t) GGML_క్రమీకరించు_క్రమం_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        స్వయంచాలక it = argsort_pipelines.find(క్రమం);
        ఒకవేళ (it != argsort_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "argsort";
        నిర్వచిస్తుంది.push_back(std::string("ORDER=") + std::to_string(క్రమం));
        variant += std::string("_order") + std::to_string(క్రమం);
        uint32_t wg_కొలత = 1;
        అయితే (wg_కొలత * 2 <= context.max_wg_కొలత &&
               wg_కొలత * GGML_WEBGPU_I32_కొలత_బైట్లు <= context.wg_mem_హద్దు_బైట్లు / 2) {
            wg_కొలత *= 2;
        }
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(wg_కొలత));
        స్వయంచాలక processed                   = preprocessor.preprocess(wgsl_argsort, నిర్వచిస్తుంది);
        స్వయంచాలక decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత               = wg_కొలత;
        argsort_pipelines[క్రమం]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[క్రమం].context = decisions;
        తిరిగి argsort_pipelines[క్రమం];
    }

    webgpu_pipeline get_argsort_విలీనం_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        తార్కిక          is_శిఖరం_k = context.dst->op == GGML_OP_శిఖరం_K;
        // ascending order is 0, descending order is 1
        స్థిరరాశి int32_t క్రమం =
            is_శిఖరం_k ? (int32_t) GGML_క్రమీకరించు_క్రమం_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        స్వయంచాలక it = argsort_విలీనం_pipelines.find(క్రమం);
        ఒకవేళ (it != argsort_విలీనం_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "argsort_merge";
        నిర్వచిస్తుంది.push_back(std::string("ORDER=") + std::to_string(క్రమం));
        variant += std::string("_order") + std::to_string(క్రమం);
        uint32_t wg_కొలత = std::min(GGML_WEBGPU_ARGSORT_విలీనం_MAX_WG_కొలత, context.max_wg_కొలత);
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(wg_కొలత));

        స్వయంచాలక processed                 = preprocessor.preprocess(wgsl_argsort_విలీనం, నిర్వచిస్తుంది);
        argsort_విలీనం_pipelines[క్రమం] = ggml_webgpu_create_pipeline(device, processed, variant);
        తిరిగి argsort_విలీనం_pipelines[క్రమం];
    }

    webgpu_pipeline get_get_అడ్డువరుసలు_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, తార్కిక vec4_aligned) {
        స్థిరరాశి తార్కిక vectorized = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0 && vec4_aligned;
        ggml_webgpu_get_అడ్డువరుసలు_pipeline_కీలకం కీలకం = {};
        కీలకం.src_type                          = context.src0->type;
        కీలకం.vectorized                        = (పూర్ణాంకం) vectorized;

        స్వయంచాలక it = get_అడ్డువరుసలు_pipelines.find(కీలకం);
        ఒకవేళ (it != get_అడ్డువరుసలు_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "get_rows";

        స్థిరరాశి నిర్మాణం ggml_type_traits * type_traits = ggml_get_type_traits(కీలకం.src_type);
        స్థిరరాశి అక్షరం *                    type_str    = type_traits->type_name;

        ఎంపిక (కీలకం.src_type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("FLOAT_PARALLEL");
                ఒకవేళ (కీలకం.vectorized) {
                    నిర్వచిస్తుంది.push_back("F32_VEC");
                    నిర్వచిస్తుంది.push_back("SRC_TYPE=vec4<f32>");
                    నిర్వచిస్తుంది.push_back("DST_TYPE=vec4<f32>");
                    నిర్వచిస్తుంది.push_back("BLOCK_SIZE=4u");
                } లేకపోతే {
                    నిర్వచిస్తుంది.push_back("F32");
                    నిర్వచిస్తుంది.push_back("SRC_TYPE=f32");
                    నిర్వచిస్తుంది.push_back("DST_TYPE=f32");
                    నిర్వచిస్తుంది.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("FLOAT_PARALLEL");
                నిర్వచిస్తుంది.push_back("F16");
                నిర్వచిస్తుంది.push_back("SRC_TYPE=f16");
                నిర్వచిస్తుంది.push_back("DST_TYPE=f32");
                నిర్వచిస్తుంది.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                ఆపు;
            స్థితి GGML_TYPE_I32:
                నిర్వచిస్తుంది.push_back("FLOAT_PARALLEL");
                నిర్వచిస్తుంది.push_back("I32");
                నిర్వచిస్తుంది.push_back("SRC_TYPE=i32");
                నిర్వచిస్తుంది.push_back("DST_TYPE=i32");
                నిర్వచిస్తుంది.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                ఆపు;
            స్థితి GGML_TYPE_BF16:
                నిర్వచిస్తుంది.push_back("BF16");
                నిర్వచిస్తుంది.push_back("SRC_TYPE=u32");
                నిర్వచిస్తుంది.push_back("DST_TYPE=f32");
                నిర్వచిస్తుంది.push_back("BLOCK_SIZE=1u");
                variant += "_bf16";
                ఆపు;
            అప్రమేయ:
                {
                    std::string type_పై = type_str;
                    std::transform(type_పై.begin(), type_పై.end(), type_పై.begin(), ::పెద్ద_అక్షరం);

                    ఎంపిక (కీలకం.src_type) {
                        స్థితి GGML_TYPE_Q1_0:
                        స్థితి GGML_TYPE_Q2_0:
                        స్థితి GGML_TYPE_Q4_0:
                        స్థితి GGML_TYPE_Q5_0:
                        స్థితి GGML_TYPE_Q8_0:
                        స్థితి GGML_TYPE_Q3_K:
                        స్థితి GGML_TYPE_Q6_K:
                        స్థితి GGML_TYPE_IQ2_XXS:
                        స్థితి GGML_TYPE_IQ2_XS:
                        స్థితి GGML_TYPE_IQ2_S:
                        స్థితి GGML_TYPE_IQ3_XXS:
                        స్థితి GGML_TYPE_IQ3_S:
                        స్థితి GGML_TYPE_IQ1_S:
                        స్థితి GGML_TYPE_IQ4_NL:
                        స్థితి GGML_TYPE_MXFP4:
                        స్థితి GGML_TYPE_NVFP4:
                            {
                                // Quantized types using u32 buffers for portability.
                                నిర్వచిస్తుంది.push_back("SRC_TYPE=u32");
                                నిర్వచిస్తుంది.push_back("U32_DEQUANT_HELPERS");
                                ఆపు;
                            }
                        అప్రమేయ:
                            {
                                నిర్వచిస్తుంది.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    నిర్వచిస్తుంది.push_back("BYTE_HELPERS");
                    నిర్వచిస్తుంది.push_back(type_పై + "_T");
                    నిర్వచిస్తుంది.push_back(type_పై);
                    నిర్వచిస్తుంది.push_back(type_పై + "_SCALE_MIN");
                    నిర్వచిస్తుంది.push_back(type_పై + "_TABLES");
                    నిర్వచిస్తుంది.push_back(type_పై + "_GRID");
                    నిర్వచిస్తుంది.push_back(type_పై + "_LUT");

                    variant += "_";
                    variant += type_str;

                    నిర్వచిస్తుంది.push_back("DST_TYPE=f32");

                    ఒకవేళ (కీలకం.src_type == GGML_TYPE_Q1_0) {
                        నిర్వచిస్తుంది.push_back("BLOCK_SIZE=128u");
                    } లేకపోతే ఒకవేళ ((కీలకం.src_type >= GGML_TYPE_Q4_0 && కీలకం.src_type <= GGML_TYPE_Q8_1) ||
                               కీలకం.src_type == GGML_TYPE_IQ4_NL || కీలకం.src_type == GGML_TYPE_MXFP4) {
                        నిర్వచిస్తుంది.push_back("BLOCK_SIZE=32u");
                    } లేకపోతే ఒకవేళ (కీలకం.src_type == GGML_TYPE_Q2_0 || కీలకం.src_type == GGML_TYPE_NVFP4) {
                        నిర్వచిస్తుంది.push_back("BLOCK_SIZE=64u");
                    } లేకపోతే ఒకవేళ (కీలకం.src_type >= GGML_TYPE_Q2_K) {
                        నిర్వచిస్తుంది.push_back("BLOCK_SIZE=256u");
                    } లేకపోతే {
                        నిర్వచిస్తుంది.push_back("BLOCK_SIZE=1u");
                    }
                    ఆపు;
                }
        }

        ఒకవేళ (కీలకం.vectorized) {
            variant += "_vec";
        }

        నిర్వచిస్తుంది.push_back("WG_SIZE=" + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_get_అడ్డువరుసలు, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_get_అడ్డువరుసలు_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        decisions->vectorized    = vectorized;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_అడ్డువరుసలు_pipelines[కీలకం]  = pipeline;
        తిరిగి get_అడ్డువరుసలు_pipelines[కీలకం];
    }

    webgpu_pipeline get_scale_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_కీలకం కీలకం = {};
        కీలకం.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        స్వయంచాలక it = scale_pipelines.find(కీలకం);
        ఒకవేళ (it != scale_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "scale";

        ఒకవేళ (కీలకం.inplace) {
            నిర్వచిస్తుంది.push_back("INPLACE");
            variant += "_inplace";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_scale, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        decisions->inplace       = కీలకం.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[కీలకం]     = pipeline;
        తిరిగి scale_pipelines[కీలకం];
    }

    webgpu_pipeline get_solve_tri_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_కీలకం కీలకం = {};
        కీలకం.type                               = context.dst->type;
        కీలకం.n                                  = (పూర్ణాంకం) context.src0->ne[0];
        కీలకం.k                                  = (పూర్ణాంకం) context.src1->ne[0];

        స్వయంచాలక it = solve_tri_pipelines.find(కీలకం);
        ఒకవేళ (it != solve_tri_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "solve_tri";

        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                variant += "_f32";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        స్థిరరాశి uint32_t wg_కొలత       = std::min((uint32_t) కీలకం.n, context.max_wg_కొలత);
        స్థిరరాశి uint32_t k_tile        = wg_కొలత;
        స్థిరరాశి uint32_t బైట్లు_చొప్పున_అడ్డువరుస = ((uint32_t) కీలకం.n + wg_కొలత) * GGML_WEBGPU_F32_కొలత_బైట్లు;
        స్థిరరాశి uint32_t batch_n       = (uint32_t) (context.wg_mem_హద్దు_బైట్లు / బైట్లు_చొప్పున_అడ్డువరుస);

        నిర్వచిస్తుంది.push_back(std::string("N=") + std::to_string(కీలకం.n));
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(wg_కొలత));
        నిర్వచిస్తుంది.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        నిర్వచిస్తుంది.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_solve_tri, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = wg_కొలత;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[కీలకం] = pipeline;
        తిరిగి solve_tri_pipelines[కీలకం];
    }

    webgpu_pipeline get_ssm_conv_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_కీలకం కీలకం = {};
        కీలకం.type                              = context.dst->type;
        కీలకం.vectorized                        = context.src1->ne[0] == 4;

        స్వయంచాలక it = ssm_conv_pipelines.find(కీలకం);
        ఒకవేళ (it != ssm_conv_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "ssm_conv";

        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                variant += "_f32";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        ఒకవేళ (కీలకం.vectorized) {
            నిర్వచిస్తుంది.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t ఖండం_కొలత_3    = 32u;
        constexpr uint32_t టోకెన్లు_చొప్పున_wg = 8u;

        నిర్వచిస్తుంది.push_back("BLOCK_SIZE=" + std::to_string(ఖండం_కొలత_3) + "u");
        నిర్వచిస్తుంది.push_back("TOKENS_PER_WG=" + std::to_string(టోకెన్లు_చొప్పున_wg) + "u");

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_ssm_conv, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->ఖండం_కొలత_3    = ఖండం_కొలత_3;
        decisions->టోకెన్లు_చొప్పున_wg = టోకెన్లు_చొప్పున_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[కీలకం]  = pipeline;
        తిరిగి ssm_conv_pipelines[కీలకం];
    }

    webgpu_pipeline get_ssm_పరిశీలన_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, uint8_t xbc_overlap) {
        ggml_webgpu_ssm_పరిశీలన_pipeline_కీలకం కీలకం = {};
        కీలకం.type                              = context.dst->type;
        కీలకం.d_state                           = (పూర్ణాంకం) context.src0->ne[0];
        కీలకం.xbc_overlap                       = xbc_overlap;

        స్వయంచాలక it = ssm_పరిశీలన_pipelines.find(కీలకం);
        ఒకవేళ (it != ssm_పరిశీలన_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "ssm_scan";

        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                variant += "_f32";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        స్థిరరాశి uint32_t wg_కొలత = (uint32_t) కీలకం.d_state;

        constexpr uint32_t టోకెన్లు_చొప్పున_tile = 4u;

        నిర్వచిస్తుంది.push_back("WG_SIZE=" + std::to_string(wg_కొలత) + "u");
        నిర్వచిస్తుంది.push_back("TOKENS_PER_TILE=" + std::to_string(టోకెన్లు_చొప్పున_tile) + "u");

        ఒకవేళ (context.supports_subgroups) {
            నిర్వచిస్తుంది.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } లేకపోతే {
            variant += "_wg_reduce";
        }

        ఒకవేళ (కీలకం.xbc_overlap == 0b110) {  // x/B
            నిర్వచిస్తుంది.push_back("XB_OVERLAP");
            variant += "_xb_overlap";
        } లేకపోతే ఒకవేళ (కీలకం.xbc_overlap == 0b011) {  // B/C
            నిర్వచిస్తుంది.push_back("BC_OVERLAP");
            variant += "_bc_overlap";
        } లేకపోతే ఒకవేళ (కీలకం.xbc_overlap == 0b111) {  // x/B/C
            నిర్వచిస్తుంది.push_back("XBC_OVERLAP");
            variant += "_xbc_overlap";
        }

        variant += "_d" + std::to_string(కీలకం.d_state);

        స్వయంచాలక processed             = preprocessor.preprocess(wgsl_ssm_పరిశీలన, నిర్వచిస్తుంది);
        స్వయంచాలక decisions             = std::make_shared<ggml_webgpu_ssm_పరిశీలన_shader_decisions>();
        decisions->wg_కొలత         = wg_కొలత;
        decisions->టోకెన్లు_చొప్పున_tile = టోకెన్లు_చొప్పున_tile;
        decisions->xbc_overlap     = కీలకం.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_పరిశీలన_pipelines[కీలకం]    = pipeline;
        తిరిగి ssm_పరిశీలన_pipelines[కీలకం];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_కీలకం కీలకం = {};
        కీలకం.type                                     = context.dst->type;
        కీలకం.s_v                                      = (పూర్ణాంకం) context.src2->ne[0];
        కీలకం.kda                                      = context.src3->ne[0] == context.src2->ne[0];
        కీలకం.fused_cache                              = context.dst_fuse != శూన్యసూచిక;

        స్వయంచాలక it = gated_delta_net_pipelines.find(కీలకం);
        ఒకవేళ (it != gated_delta_net_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "gated_delta_net";

        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                variant += "_f32";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        ఒకవేళ (కీలకం.kda) {
            నిర్వచిస్తుంది.push_back("KDA");
            variant += "_kda";
        }

        ఒకవేళ (కీలకం.fused_cache) {
            నిర్వచిస్తుంది.push_back("FUSED_CACHE");
            variant += "_fused_cache";
        }

        నిర్వచిస్తుంది.push_back("S_V=" + std::to_string(కీలకం.s_v) + "u");
        నిర్వచిస్తుంది.push_back("WG_SIZE=" + std::to_string(కీలకం.s_v) + "u");

        స్వయంచాలక            processed      = preprocessor.preprocess(wgsl_gated_delta_net, నిర్వచిస్తుంది);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[కీలకం] = pipeline;
        తిరిగి gated_delta_net_pipelines[కీలకం];
    }

    webgpu_pipeline get_pad_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_కీలకం కీలకం = {};
        కీలకం.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        స్వయంచాలక it = pad_pipelines.find(కీలకం);
        ఒకవేళ (it != pad_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "pad";

        ఒకవేళ (కీలకం.circular) {
            నిర్వచిస్తుంది.push_back("CIRCULAR");
            variant += "_circular";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_pad, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[కీలకం]       = pipeline;
        తిరిగి pad_pipelines[కీలకం];
    }

    webgpu_pipeline get_quantize_q8_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_quantize_q8_pipeline_కీలకం కీలకం = {};
        కీలకం.src0_type                            = context.src0->type;

        స్వయంచాలక it = quantize_q8_pipelines.find(కీలకం);
        ఒకవేళ (it != quantize_q8_pipelines.end()) {
            తిరిగి it->second;
        }
        స్థిరరాశి అక్షరం *             shader_src = wgsl_quantize_q8;
        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "quantize_q8";

        uint32_t wg_కొలత = WEBGPU_MUL_MAT_VEC_WG_కొలత;

        నిర్వచిస్తుంది.push_back("SRC1_INNER_TYPE=f32");
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(wg_కొలత));

        స్థిరరాశి నిర్మాణం ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        std::string                     src0_name   = src0_traits->type_name;
        std::string                     type_పై  = src0_name;
        variant += "_" + src0_name;
        std::transform(type_పై.begin(), type_పై.end(), type_పై.begin(), ::పెద్ద_అక్షరం);

        నిర్వచిస్తుంది.push_back("MUL_ACC_" + type_పై);
        నిర్వచిస్తుంది.push_back("Q8_1_T");

        నిర్వచిస్తుంది.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";

        స్వయంచాలక processed             = preprocessor.preprocess(shader_src, నిర్వచిస్తుంది);
        స్వయంచాలక decisions             = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత         = wg_కొలత;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        quantize_q8_pipelines[కీలకం] = pipeline;
        తిరిగి quantize_q8_pipelines[కీలకం];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, తార్కిక src_overlap) {
        ggml_webgpu_mul_mat_vec_pipeline_కీలకం కీలకం = {};
        కీలకం.src0_type                            = context.src0->type;
        కీలకం.src1_type                            = context.src1->type;
        కీలకం.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        కీలకం.అంకె_నిలువుల_సంఖ్య   = context.dst->ne[1];
        కీలకం.use_mmvq =
            ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_లబ్ధం, context.vendor);
        కీలకం.src_overlap = src_overlap;

        స్వయంచాలక it = mul_mat_vec_pipelines.find(కీలకం);
        ఒకవేళ (it != mul_mat_vec_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant    = "mul_mat_vec";
        స్థిరరాశి అక్షరం *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        ఎంపిక (context.src0->type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=f32");
                నిర్వచిస్తుంది.push_back("MUL_ACC_FLOAT");
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=f16");
                నిర్వచిస్తుంది.push_back("MUL_ACC_FLOAT");
                నిర్వచిస్తుంది.push_back("TYPE_F16");
                variant += "_f16";
                ఆపు;
            స్థితి GGML_TYPE_BF16:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=u32");
                నిర్వచిస్తుంది.push_back("MUL_ACC_FLOAT");
                నిర్వచిస్తుంది.push_back("TYPE_BF16");
                variant += "_bf16";
                ఆపు;
            అప్రమేయ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    స్థిరరాశి నిర్మాణం ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_పై  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_పై.begin(), type_పై.end(), type_పై.begin(), ::పెద్ద_అక్షరం);

                    నిర్వచిస్తుంది.push_back("BYTE_HELPERS");
                    నిర్వచిస్తుంది.push_back("MUL_ACC_" + type_పై);
                    నిర్వచిస్తుంది.push_back("U32_DEQUANT_HELPERS");
                    నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=u32");
                    ఎంపిక (context.src0->type) {
                        స్థితి GGML_TYPE_Q4_0:
                        స్థితి GGML_TYPE_Q4_1:
                        స్థితి GGML_TYPE_Q5_0:
                        స్థితి GGML_TYPE_Q5_1:
                        స్థితి GGML_TYPE_Q8_0:
                            ఒకవేళ (కీలకం.use_mmvq) {
                                నిర్వచిస్తుంది.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ఆపు;
                        స్థితి GGML_TYPE_Q1_0:
                        స్థితి GGML_TYPE_Q2_K:
                        స్థితి GGML_TYPE_Q3_K:
                        స్థితి GGML_TYPE_Q4_K:
                        స్థితి GGML_TYPE_Q5_K:
                        స్థితి GGML_TYPE_Q6_K:
                            ఒకవేళ (కీలకం.use_mmvq) {
                                నిర్వచిస్తుంది.push_back("K_QUANTS_HANDLING");
                            }
                            ఆపు;
                        స్థితి GGML_TYPE_IQ1_S:
                        స్థితి GGML_TYPE_IQ1_M:
                        స్థితి GGML_TYPE_IQ2_S:
                        స్థితి GGML_TYPE_IQ3_S:
                        స్థితి GGML_TYPE_IQ4_NL:
                        స్థితి GGML_TYPE_IQ4_XS:
                            నిర్వచిస్తుంది.push_back(type_పై + "_GRID");
                            ఆపు;
                        స్థితి GGML_TYPE_IQ2_XXS:
                        స్థితి GGML_TYPE_IQ2_XS:
                        స్థితి GGML_TYPE_IQ3_XXS:
                            నిర్వచిస్తుంది.push_back(type_పై + "_GRID");
                            నిర్వచిస్తుంది.push_back(type_పై + "_TABLES");
                            ఆపు;
                        స్థితి GGML_TYPE_MXFP4:
                            నిర్వచిస్తుంది.push_back(type_పై + "_LUT");
                            ఒకవేళ (కీలకం.use_mmvq) {
                                నిర్వచిస్తుంది.push_back("LEGACY_QUANTS_HANDLING");
                            }
                            ఆపు;
                        స్థితి GGML_TYPE_NVFP4:
                            నిర్వచిస్తుంది.push_back(type_పై + "_LUT");
                            ఆపు;
                        అప్రమేయ:
                            ఆపు;
                    }
                    ఆపు;
                }
        }

        // src1 type (vector)
        ఎంపిక (context.src1->type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        నిర్వచిస్తుంది.push_back(కీలకం.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_కొలత        = WEBGPU_MUL_MAT_VEC_WG_కొలత;
        uint32_t outputs_చొప్పున_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_చొప్పున_WG;

        ఒకవేళ (కీలకం.src0_type == GGML_TYPE_Q1_0) {
            outputs_చొప్పున_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_చొప్పున_WG;
        } లేకపోతే ఒకవేళ (కీలకం.src0_type >= GGML_TYPE_Q2_K) {
            outputs_చొప్పున_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_చొప్పున_WG;
        } లేకపోతే ఒకవేళ (కీలకం.src0_type >= GGML_TYPE_Q4_0) {
            outputs_చొప్పున_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_చొప్పున_WG;
        }

        ఒకవేళ (కీలకం.use_mmvq) {
            నిర్వచిస్తుంది.push_back("MMVQ");
            నిర్వచిస్తుంది.push_back("Q8_1_T");
        }

        ఒకవేళ (కీలకం.src_overlap) {
            నిర్వచిస్తుంది.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(wg_కొలత));
        నిర్వచిస్తుంది.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_చొప్పున_wg));
        నిర్వచిస్తుంది.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        ఒకవేళ (కీలకం.vectorized) {
            variant += "_vectorized";
        }
        నిర్వచిస్తుంది.push_back(std::string("NUM_COLS=") + std::to_string(కీలకం.అంకె_నిలువుల_సంఖ్య));

        స్వయంచాలక processed            = preprocessor.preprocess(shader_src, నిర్వచిస్తుంది);
        స్వయంచాలక decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_కొలత        = wg_కొలత;
        decisions->outputs_చొప్పున_wg = outputs_చొప్పున_wg;
        decisions->vec_కొలత       = కీలకం.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[కీలకం] = pipeline;
        తిరిగి mul_mat_vec_pipelines[కీలకం];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, తార్కిక src_overlap) {
        ggml_webgpu_mul_mat_pipeline_కీలకం కీలకం = {};
        కీలకం.src0_type                        = context.src0->type;
        కీలకం.src1_type                        = context.src1->type;
        కీలకం.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        కీలకం.use_subgroup_మాతృక = context.supports_subgroup_మాతృక;
        కీలకం.src_overlap         = src_overlap;

        స్వయంచాలక it = mul_mat_fast_pipelines.find(కీలకం);
        ఒకవేళ (it != mul_mat_fast_pipelines.end()) {
            తిరిగి it->second;
        }

        స్థిరరాశి అక్షరం * shader_src = కీలకం.use_subgroup_మాతృక ? wgsl_mul_mat_subgroup_మాతృక : wgsl_mul_mat_reg_tile;
        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = కీలకం.use_subgroup_మాతృక ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        ఎంపిక (context.src1->type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("SRC1_INNER_TYPE=f32");
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("SRC1_INNER_TYPE=f16");
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        స్థిరరాశి నిర్మాణం ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        స్థిరరాశి అక్షరం *                    src0_name   = src0_traits->type_name;

        ఎంపిక (context.src0->type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=f32");
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                నిర్వచిస్తుంది.push_back("MUL_ACC_FLOAT");
                నిర్వచిస్తుంది.push_back("INIT_SRC0_SHMEM_FLOAT");
                నిర్వచిస్తుంది.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=f16");
                నిర్వచిస్తుంది.push_back("TYPE_F16");
                నిర్వచిస్తుంది.push_back("MUL_ACC_FLOAT");
                నిర్వచిస్తుంది.push_back("INIT_SRC0_SHMEM_FLOAT");
                నిర్వచిస్తుంది.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                ఆపు;
            స్థితి GGML_TYPE_BF16:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=u32");
                నిర్వచిస్తుంది.push_back("TYPE_BF16");
                నిర్వచిస్తుంది.push_back("MUL_ACC_FLOAT");
                నిర్వచిస్తుంది.push_back("INIT_SRC0_SHMEM_FLOAT");
                నిర్వచిస్తుంది.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_bf16";
                ఆపు;
            అప్రమేయ:
                {
                    std::string type_పై = src0_name;
                    std::transform(type_పై.begin(), type_పై.end(), type_పై.begin(), ::పెద్ద_అక్షరం);

                    నిర్వచిస్తుంది.push_back("BYTE_HELPERS");
                    నిర్వచిస్తుంది.push_back("MUL_ACC_" + type_పై);
                    నిర్వచిస్తుంది.push_back("INIT_SRC0_SHMEM_" + type_పై);
                    నిర్వచిస్తుంది.push_back("INIT_SRC1_SHMEM_FLOAT");
                    నిర్వచిస్తుంది.push_back("U32_DEQUANT_HELPERS");
                    నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=u32");

                    ఎంపిక (context.src0->type) {
                        స్థితి GGML_TYPE_IQ1_S:
                        స్థితి GGML_TYPE_IQ1_M:
                        స్థితి GGML_TYPE_IQ4_NL:
                        స్థితి GGML_TYPE_IQ4_XS:
                            నిర్వచిస్తుంది.push_back(type_పై + "_GRID");
                            ఆపు;
                        స్థితి GGML_TYPE_IQ2_XXS:
                        స్థితి GGML_TYPE_IQ2_XS:
                        స్థితి GGML_TYPE_IQ2_S:
                        స్థితి GGML_TYPE_IQ3_XXS:
                        స్థితి GGML_TYPE_IQ3_S:
                            నిర్వచిస్తుంది.push_back(type_పై + "_GRID");
                            నిర్వచిస్తుంది.push_back(type_పై + "_TABLES");
                            ఆపు;
                        స్థితి GGML_TYPE_MXFP4:
                        స్థితి GGML_TYPE_NVFP4:
                            నిర్వచిస్తుంది.push_back(type_పై + "_LUT");
                            ఆపు;
                        అప్రమేయ:
                            ఆపు;
                    }

                    variant += std::string("_") + src0_name;
                    ఆపు;
                }
        }

        // VEC/SCALAR controls
        నిర్వచిస్తుంది.push_back(కీలకం.vectorized ? "VEC" : "SCALAR");

        స్థిరరాశి తార్కిక is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        ఒకవేళ (కీలకం.use_subgroup_మాతృక) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } లేకపోతే {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        నిర్వచిస్తుంది.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        నిర్వచిస్తుంది.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        ఒకవేళ (కీలకం.use_subgroup_మాతృక) {
            నిర్వచిస్తుంది.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            నిర్వచిస్తుంది.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_కొలత) + "u");
            నిర్వచిస్తుంది.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            నిర్వచిస్తుంది.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            నిర్వచిస్తుంది.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_మాతృక_M) + "u");
            నిర్వచిస్తుంది.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_మాతృక_N) + "u");
            నిర్వచిస్తుంది.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            నిర్వచిస్తుంది.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            నిర్వచిస్తుంది.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        ఒకవేళ (కీలకం.vectorized) {
            variant += "_vectorized";
        }

        ఒకవేళ (కీలకం.src_overlap) {
            నిర్వచిస్తుంది.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        ఒకవేళ (!కీలకం.use_subgroup_మాతృక) {
            నిర్వచిస్తుంది.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_కొలత_M) + "u");
            నిర్వచిస్తుంది.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_కొలత_N) + "u");
            నిర్వచిస్తుంది.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        స్వయంచాలక processed = preprocessor.preprocess(shader_src, నిర్వచిస్తుంది);

        స్వయంచాలక decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_మాతృక = కీలకం.use_subgroup_మాతృక;
        ఒకవేళ (కీలకం.use_subgroup_మాతృక) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_మాతృక_m = WEBGPU_MUL_MAT_SUBGROUP_మాతృక_M;
            decisions->subgroup_మాతృక_n = WEBGPU_MUL_MAT_SUBGROUP_మాతృక_N;
            decisions->wg_కొలత           = context.max_subgroup_కొలత;
        } లేకపోతే {
            decisions->wg_కొలత_m       = WEBGPU_MUL_MAT_WG_కొలత_M;
            decisions->wg_కొలత_n       = WEBGPU_MUL_MAT_WG_కొలత_N;
            decisions->wg_కొలత         = WEBGPU_MUL_MAT_WG_కొలత_M * WEBGPU_MUL_MAT_WG_కొలత_N;
            decisions->mul_mat_wg_కొలత = WEBGPU_MUL_MAT_WG_కొలత;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[కీలకం] = pipeline;
        తిరిగి mul_mat_fast_pipelines[కీలకం];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_gather_pipeline_కీలకం కీలకం = {};
        కీలకం.n_experts                                  = (uint32_t) context.src0->ne[2];

        స్వయంచాలక it = mul_mat_id_gather_pipelines.find(కీలకం);
        ఒకవేళ (it != mul_mat_id_gather_pipelines.end()) {
            తిరిగి it->second;
        }
        std::vector<std::string> నిర్వచిస్తుంది;
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత) + "u");
        నిర్వచిస్తుంది.push_back(std::string("N_EXPERT=") + std::to_string(కీలకం.n_experts) + "u");
        నిర్వచిస్తుంది.push_back(std::string("TILE_N_SLOTS=") +
                          std::to_string(WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_కొలత_N) + "u");

        స్వయంచాలక processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, నిర్వచిస్తుంది);
        స్వయంచాలక decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత = context.max_wg_కొలత;

        webgpu_pipeline pipeline         = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context                 = decisions;
        mul_mat_id_gather_pipelines[కీలకం] = pipeline;
        తిరిగి pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_కీలకం కీలకం = {};
        కీలకం.src0_type                           = context.src0->type;
        కీలకం.src1_type                           = context.src1->type;
        కీలకం.n_experts                           = context.src0->ne[2];
        కీలకం.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        స్వయంచాలక it = mul_mat_id_pipelines.find(కీలకం);
        ఒకవేళ (it != mul_mat_id_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "mul_mat_id";
        నిర్వచిస్తుంది.push_back("MUL_MAT_ID");

        // src1 type
        ఎంపిక (context.src1->type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("SRC1_INNER_TYPE=f32");
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("SRC1_INNER_TYPE=f16");
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        స్థిరరాశి నిర్మాణం ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        స్థిరరాశి అక్షరం *                    src0_name   = src0_traits->type_name;

        ఎంపిక (context.src0->type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=f32");
                నిర్వచిస్తుంది.push_back("INIT_SRC0_SHMEM_FLOAT");
                నిర్వచిస్తుంది.push_back("INIT_SRC1_SHMEM_FLOAT");
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=f16");
                నిర్వచిస్తుంది.push_back("INIT_SRC0_SHMEM_FLOAT");
                నిర్వచిస్తుంది.push_back("INIT_SRC1_SHMEM_FLOAT");
                నిర్వచిస్తుంది.push_back("TYPE_F16");
                variant += "_f16";
                ఆపు;
            స్థితి GGML_TYPE_BF16:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=u32");
                నిర్వచిస్తుంది.push_back("INIT_SRC0_SHMEM_FLOAT");
                నిర్వచిస్తుంది.push_back("INIT_SRC1_SHMEM_FLOAT");
                నిర్వచిస్తుంది.push_back("TYPE_BF16");
                variant += "_bf16";
                ఆపు;
            అప్రమేయ:
                {
                    std::string type_పై = src0_name;
                    std::transform(type_పై.begin(), type_పై.end(), type_పై.begin(), ::పెద్ద_అక్షరం);

                    నిర్వచిస్తుంది.push_back("BYTE_HELPERS");
                    నిర్వచిస్తుంది.push_back("INIT_SRC0_SHMEM_" + type_పై);
                    నిర్వచిస్తుంది.push_back("INIT_SRC1_SHMEM_FLOAT");
                    నిర్వచిస్తుంది.push_back("U32_DEQUANT_HELPERS");
                    నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=u32");

                    ఎంపిక (context.src0->type) {
                        స్థితి GGML_TYPE_IQ1_S:
                        స్థితి GGML_TYPE_IQ1_M:
                        స్థితి GGML_TYPE_IQ4_NL:
                        స్థితి GGML_TYPE_IQ4_XS:
                            నిర్వచిస్తుంది.push_back(type_పై + "_GRID");
                            ఆపు;
                        స్థితి GGML_TYPE_IQ2_XXS:
                        స్థితి GGML_TYPE_IQ2_XS:
                        స్థితి GGML_TYPE_IQ2_S:
                        స్థితి GGML_TYPE_IQ3_XXS:
                        స్థితి GGML_TYPE_IQ3_S:
                            నిర్వచిస్తుంది.push_back(type_పై + "_GRID");
                            నిర్వచిస్తుంది.push_back(type_పై + "_TABLES");
                            ఆపు;
                        స్థితి GGML_TYPE_MXFP4:
                        స్థితి GGML_TYPE_NVFP4:
                            నిర్వచిస్తుంది.push_back(type_పై + "_LUT");
                            ఆపు;
                        అప్రమేయ:
                            ఆపు;
                    }

                    variant += std::string("_") + src0_name;
                    ఆపు;
                }
        }

        // VEC/SCALAR controls
        నిర్వచిస్తుంది.push_back(కీలకం.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        స్థిరరాశి uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        నిర్వచిస్తుంది.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        నిర్వచిస్తుంది.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        నిర్వచిస్తుంది.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        నిర్వచిస్తుంది.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_కొలత_M) + "u");
        నిర్వచిస్తుంది.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_కొలత_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        ఒకవేళ (కీలకం.vectorized) {
            variant += "_vectorized";
        }

        స్వయంచాలక processed = preprocessor.preprocess(wgsl_mul_mat_id, నిర్వచిస్తుంది);

        స్వయంచాలక decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_కొలత_m = WEBGPU_MUL_MAT_WG_కొలత_M;
        decisions->wg_కొలత_n = WEBGPU_MUL_MAT_WG_కొలత_N;
        decisions->wg_కొలత   = WEBGPU_MUL_MAT_WG_కొలత_M * WEBGPU_MUL_MAT_WG_కొలత_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[కీలకం] = pipeline;
        తిరిగి mul_mat_id_pipelines[కీలకం];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_కీలకం కీలకం = {};
        కీలకం.src0_type                           = context.src0->type;
        కీలకం.src1_type                           = context.src1->type;
        కీలకం.n_experts                           = context.src0->ne[2];
        కీలకం.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        స్వయంచాలక it = mul_mat_id_vec_pipelines.find(కీలకం);
        ఒకవేళ (it != mul_mat_id_vec_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant    = "mul_mat_id_vec";
        స్థిరరాశి అక్షరం *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        ఎంపిక (context.src1->type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("SRC1_INNER_TYPE=f32");
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("SRC1_INNER_TYPE=f16");
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        ఎంపిక (context.src0->type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=f32");
                నిర్వచిస్తుంది.push_back("MUL_ACC_FLOAT");
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=f16");
                నిర్వచిస్తుంది.push_back("MUL_ACC_FLOAT");
                నిర్వచిస్తుంది.push_back("TYPE_F16");
                variant += "_f16";
                ఆపు;
            స్థితి GGML_TYPE_BF16:
                నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=u32");
                నిర్వచిస్తుంది.push_back("MUL_ACC_FLOAT");
                నిర్వచిస్తుంది.push_back("TYPE_BF16");
                variant += "_bf16";
                ఆపు;
            అప్రమేయ:
                {
                    // Quantized types: use helpers but accumulate in f16
                    స్థిరరాశి నిర్మాణం ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_పై  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_పై.begin(), type_పై.end(), type_పై.begin(), ::పెద్ద_అక్షరం);

                    నిర్వచిస్తుంది.push_back("BYTE_HELPERS");
                    నిర్వచిస్తుంది.push_back("MUL_ACC_" + type_పై);
                    నిర్వచిస్తుంది.push_back("U32_DEQUANT_HELPERS");
                    నిర్వచిస్తుంది.push_back("SRC0_INNER_TYPE=u32");
                    ఎంపిక (context.src0->type) {
                        స్థితి GGML_TYPE_IQ1_S:
                        స్థితి GGML_TYPE_IQ1_M:
                        స్థితి GGML_TYPE_IQ2_S:
                        స్థితి GGML_TYPE_IQ3_S:
                        స్థితి GGML_TYPE_IQ4_NL:
                        స్థితి GGML_TYPE_IQ4_XS:
                            నిర్వచిస్తుంది.push_back(type_పై + "_GRID");
                            ఆపు;
                        స్థితి GGML_TYPE_IQ2_XXS:
                        స్థితి GGML_TYPE_IQ2_XS:
                        స్థితి GGML_TYPE_IQ3_XXS:
                            నిర్వచిస్తుంది.push_back(type_పై + "_GRID");
                            నిర్వచిస్తుంది.push_back(type_పై + "_TABLES");
                            ఆపు;
                        స్థితి GGML_TYPE_MXFP4:
                        స్థితి GGML_TYPE_NVFP4:
                            నిర్వచిస్తుంది.push_back(type_పై + "_LUT");
                            ఆపు;
                        అప్రమేయ:
                            ఆపు;
                    }
                    ఆపు;
                }
        }

        // VEC/SCALAR controls
        నిర్వచిస్తుంది.push_back(కీలకం.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_కొలత        = WEBGPU_MUL_MAT_VEC_WG_కొలత;
        uint32_t outputs_చొప్పున_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_చొప్పున_WG;

        ఒకవేళ (కీలకం.src0_type == GGML_TYPE_Q1_0) {
            outputs_చొప్పున_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_చొప్పున_WG;
        } లేకపోతే ఒకవేళ (కీలకం.src0_type >= GGML_TYPE_Q2_K) {
            outputs_చొప్పున_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_చొప్పున_WG;
        } లేకపోతే ఒకవేళ (కీలకం.src0_type >= GGML_TYPE_Q4_0) {
            outputs_చొప్పున_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_చొప్పున_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(wg_కొలత));
        నిర్వచిస్తుంది.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_చొప్పున_wg));
        నిర్వచిస్తుంది.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        ఒకవేళ (కీలకం.vectorized) {
            variant += "_vectorized";
        }
        నిర్వచిస్తుంది.push_back(std::string("NUM_COLS=1"));

        నిర్వచిస్తుంది.push_back(std::string("N_EXPERTS=") + std::to_string(కీలకం.n_experts));

        స్వయంచాలక processed = preprocessor.preprocess(shader_src, నిర్వచిస్తుంది);

        స్వయంచాలక decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_కొలత        = wg_కొలత;
        decisions->outputs_చొప్పున_wg = outputs_చొప్పున_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[కీలకం] = pipeline;
        తిరిగి mul_mat_id_vec_pipelines[కీలకం];
    }

    webgpu_pipeline get_unary_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        స్థిరరాశి తార్కిక                     is_unary = context.dst->op == GGML_OP_UNARY;
        స్థిరరాశి పూర్ణాంకం                      op       = is_unary ? (పూర్ణాంకం) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_కీలకం కీలకం      = {};
        కీలకం.type                                = context.dst->type;
        కీలకం.op                                  = op;
        కీలకం.is_unary                            = is_unary;
        కీలకం.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        కీలకం.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        స్వయంచాలక it = unary_pipelines.find(కీలకం);
        ఒకవేళ (it != unary_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant =
            కీలకం.is_unary ? ggml_unary_op_name((ggml_unary_op) కీలకం.op) : ggml_op_name((ggml_op) కీలకం.op);
        నిర్వచిస్తుంది.push_back(variant);

        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("TYPE_F16");
                variant += "_f16";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for unary shader");
        }

        ఒకవేళ (కీలకం.inplace) {
            నిర్వచిస్తుంది.push_back("INPLACE");
            variant += "_inplace";
        }

        ఒకవేళ (op == GGML_OP_TRI) {
            ఎంపిక (కీలకం.ttype) {
                స్థితి GGML_TRI_TYPE_క్రింది:
                    నిర్వచిస్తుంది.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    ఆపు;
                స్థితి GGML_TRI_TYPE_క్రింది_DIAG:
                    నిర్వచిస్తుంది.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    ఆపు;
                స్థితి GGML_TRI_TYPE_పై:
                    నిర్వచిస్తుంది.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    ఆపు;
                స్థితి GGML_TRI_TYPE_పై_DIAG:
                    నిర్వచిస్తుంది.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    ఆపు;
                అప్రమేయ:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_unary, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        decisions->inplace       = కీలకం.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[కీలకం]     = pipeline;
        తిరిగి unary_pipelines[కీలకం];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, తార్కిక src_overlap) {
        ggml_webgpu_rms_norm_mul_pipeline_కీలకం కీలకం = {};
        కీలకం.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        కీలకం.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        కీలకం.src_overlap                           = src_overlap;

        స్వయంచాలక it = rms_norm_mul_pipelines.find(కీలకం);
        ఒకవేళ (it != rms_norm_mul_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        ఒకవేళ (కీలకం.inplace) {
            నిర్వచిస్తుంది.push_back("INPLACE");
            variant += "_inplace";
        } లేకపోతే ఒకవేళ (కీలకం.overlap) {
            నిర్వచిస్తుంది.push_back("OVERLAP");
            variant += "_overlap";
        } లేకపోతే ఒకవేళ (కీలకం.src_overlap) {
            నిర్వచిస్తుంది.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, నిర్వచిస్తుంది);
        స్వయంచాలక pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_కొలత     = context.max_wg_కొలత;
        pipeline_decisions->inplace     = కీలకం.inplace;
        pipeline_decisions->overlap     = కీలకం.overlap;
        pipeline_decisions->src_overlap = కీలకం.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[కీలకం]     = pipeline;
        తిరిగి rms_norm_mul_pipelines[కీలకం];
    }

    webgpu_pipeline get_ద్విమాన_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, తార్కిక src_overlap) {
        ggml_webgpu_ద్విమాన_pipeline_కీలకం కీలకం = {};
        కీలకం.type                            = context.dst->type;
        కీలకం.op                              = context.dst->op;
        కీలకం.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        కీలకం.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        కీలకం.src_overlap                     = src_overlap;

        స్వయంచాలక it = ద్విమాన_pipelines.find(కీలకం);
        ఒకవేళ (it != ద్విమాన_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              op_name = ggml_op_name((ggml_op) కీలకం.op);
        std::string              variant = op_name;

        నిర్వచిస్తుంది.push_back(std::string("OP_") + op_name);

        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("TYPE_F16");
                variant += "_f16";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for binary shader");
        }

        ఒకవేళ (కీలకం.inplace) {
            నిర్వచిస్తుంది.push_back("INPLACE");
            variant += "_inplace";
        } లేకపోతే ఒకవేళ (కీలకం.overlap) {
            నిర్వచిస్తుంది.push_back("OVERLAP");
            variant += "_overlap";
        } లేకపోతే ఒకవేళ (కీలకం.src_overlap) {
            నిర్వచిస్తుంది.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed                  = preprocessor.preprocess(wgsl_ద్విమాన, నిర్వచిస్తుంది);
        స్వయంచాలక pipeline_decisions         = std::make_shared<ggml_webgpu_ద్విమాన_shader_decisions>();
        pipeline_decisions->wg_కొలత     = context.max_wg_కొలత;
        pipeline_decisions->inplace     = కీలకం.inplace;
        pipeline_decisions->overlap     = కీలకం.overlap;
        pipeline_decisions->src_overlap = కీలకం.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        ద్విమాన_pipelines[కీలకం]    = pipeline;
        తిరిగి ద్విమాన_pipelines[కీలకం];
    }

    webgpu_pipeline get_కూడు_id_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_కూడు_id_pipeline_కీలకం కీలకం = {};
        కీలకం.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);

        స్వయంచాలక it = కూడు_id_pipelines.find(కీలకం);
        ఒకవేళ (it != కూడు_id_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant    = "add_id";
        స్థిరరాశి అక్షరం *             shader_src = wgsl_కూడు_id;

        ఒకవేళ (కీలకం.inplace) {
            నిర్వచిస్తుంది.push_back("INPLACE");
            variant += "_inplace";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed              = preprocessor.preprocess(shader_src, నిర్వచిస్తుంది);
        స్వయంచాలక pipeline_decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        pipeline_decisions->wg_కొలత = context.max_wg_కొలత;
        pipeline_decisions->inplace = కీలకం.inplace;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        కూడు_id_pipelines[కీలకం]    = pipeline;
        తిరిగి pipeline;
    }

    webgpu_pipeline get_concat_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, తార్కిక src_overlap) {
        ggml_webgpu_concat_pipeline_కీలకం కీలకం = {};
        కీలకం.type                            = context.dst->type;
        కీలకం.src_overlap                     = src_overlap;

        స్వయంచాలక it = concat_pipelines.find(కీలకం);
        ఒకవేళ (it != concat_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "concat";

        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_I32:
                నిర్వచిస్తుంది.push_back("TYPE_I32");
                variant += "_i32";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for concat shader");
        }

        ఒకవేళ (కీలకం.src_overlap) {
            నిర్వచిస్తుంది.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_concat, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_ద్విమాన_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        decisions->src_overlap   = కీలకం.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[కీలకం]    = pipeline;
        తిరిగి concat_pipelines[కీలకం];
    }

    webgpu_pipeline get_repeat_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_కీలకం కీలకం = {};
        కీలకం.type                            = context.dst->type;

        స్వయంచాలక it = repeat_pipelines.find(కీలకం);
        ఒకవేళ (it != repeat_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "repeat";

        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("TYPE_F16");
                variant += "_f16";
                ఆపు;
            స్థితి GGML_TYPE_I32:
                నిర్వచిస్తుంది.push_back("TYPE_I32");
                variant += "_i32";
                ఆపు;
            స్థితి GGML_TYPE_I16:
                నిర్వచిస్తుంది.push_back("TYPE_I16");
                variant += "_i16";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_repeat, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[కీలకం]    = pipeline;
        తిరిగి repeat_pipelines[కీలకం];
    }

    webgpu_pipeline get_flash_attn_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, తార్కిక kv_overlap) {
        స్థిరరాశి తార్కిక can_use_subgroup_మాతృక = ggml_webgpu_flash_attn_can_use_subgroup_మాతృక_మార్గం(
            context.supports_subgroup_మాతృక, context.sg_mat_k, context.sg_mat_n, context.src0, context.src2);
        ggml_webgpu_flash_attn_decisions decisions = {};
        decisions.use_sg_మాతృక                    = can_use_subgroup_మాతృక;
        decisions.q_tile = decisions.use_sg_మాతృక ? context.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;

        ggml_webgpu_flash_attn_pipeline_కీలకం కీలకం = {};
        కీలకం.common                              = ggml_webgpu_flash_attn_make_common_pipeline_కీలకం(
            context, decisions.use_sg_మాతృక ? context.sg_mat_k : 1u, kv_overlap);
        కీలకం.common.k_ప్రత్యక్ష &= decisions.use_sg_మాతృక && కీలకం.common.k_type == GGML_TYPE_F16;
        కీలకం.common.v_ప్రత్యక్ష &= decisions.use_sg_మాతృక && కీలకం.common.v_type == GGML_TYPE_F16;
        కీలకం.use_sg_మాతృక = decisions.use_sg_మాతృక;

        uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
            context.wg_mem_హద్దు_బైట్లు, decisions.q_tile, decisions.use_sg_మాతృక ? context.sg_mat_n : 1u,
            కీలకం.common.తల_dim_qk, కీలకం.common.తల_dim_v, కీలకం.common.has_ముసుగు,
            కీలకం.common.k_ప్రత్యక్ష || కీలకం.common.v_ప్రత్యక్ష);

        // WorkGroup storage size isn't enough for some params with subgroup matrices path (ref. https://github.com/ggml-org/llama.cpp/pull/26566)
        ఒకవేళ (max_kv_tile == 0) {
            GGML_ASSERT(decisions.use_sg_మాతృక);
            // switch to flash_attn_reg_tile path
            decisions.use_sg_మాతృక = అబద్ధం;
            decisions.q_tile        = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
            కీలకం.common.k_ప్రత్యక్ష     = అబద్ధం;
            కీలకం.common.v_ప్రత్యక్ష     = అబద్ధం;
            కీలకం.use_sg_మాతృక       = అబద్ధం;
            max_kv_tile             = ggml_webgpu_flash_attn_max_kv_tile(
                context.wg_mem_హద్దు_బైట్లు, decisions.q_tile, 1u, కీలకం.common.తల_dim_qk, కీలకం.common.తల_dim_v,
                కీలకం.common.has_ముసుగు, కీలకం.common.k_ప్రత్యక్ష || కీలకం.common.v_ప్రత్యక్ష);
            GGML_ASSERT(max_kv_tile > 0);
        }

        decisions.kv_tile = decisions.use_sg_మాతృక ?
                                std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES) :
                                std::min(GGML_WEBGPU_FLASH_ATTN_TILE_MAX_KV_TILE, max_kv_tile);
        decisions.wg_కొలత =
            decisions.use_sg_మాతృక ?
                std::max(context.max_subgroup_కొలత, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_కొలత) :
                std::min(context.max_wg_కొలత, std::max(GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_కొలత,
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * context.max_subgroup_కొలత));

        ఒకవేళ (కీలకం.common.k_ప్రత్యక్ష || కీలకం.common.v_ప్రత్యక్ష) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            అయితే (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= decisions.use_sg_మాతృక ? context.sg_mat_n : context.min_subgroup_కొలత;
            }
        }

        స్వయంచాలక it = flash_attn_pipelines.find(కీలకం);
        ఒకవేళ (it != flash_attn_pipelines.end()) {
            తిరిగి it->second;
        }

        std::string              variant = decisions.use_sg_మాతృక ? "flash_attn" : "flash_attn_tile";
        std::vector<std::string> నిర్వచిస్తుంది = ggml_webgpu_flash_attn_common_నిర్వచిస్తుంది(కీలకం.common, variant, decisions.q_tile,
                                                                                 decisions.kv_tile, decisions.wg_కొలత);
        స్థిరరాశి అక్షరం *             shader_src = శూన్యసూచిక;
        ఒకవేళ (!కీలకం.use_sg_మాతృక) {
            shader_src = wgsl_flash_attn_tile;
            నిర్వచిస్తుంది.push_back("MIN_SUBGROUP_SIZE=" + std::to_string(context.min_subgroup_కొలత) + "u");
            నిర్వచిస్తుంది.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_కొలత) + "u");
            variant += "_tile_sg" + std::to_string(context.min_subgroup_కొలత) + "_" +
                       std::to_string(context.max_subgroup_కొలత);
        } లేకపోతే {
            shader_src = wgsl_flash_attn;
            నిర్వచిస్తుంది.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            నిర్వచిస్తుంది.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            నిర్వచిస్తుంది.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }
        స్వయంచాలక            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, నిర్వచిస్తుంది), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[కీలకం] = pipeline;
        తిరిగి flash_attn_pipelines[కీలకం];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, తార్కిక kv_overlap) {
        ggml_webgpu_flash_attn_vec_pipeline_కీలకం కీలకం = {};
        కీలకం.common = ggml_webgpu_flash_attn_make_common_pipeline_కీలకం(context, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_వెడల్పు,
                                                                     kv_overlap);

        స్వయంచాలక it = flash_attn_vec_pipelines.find(కీలకం);
        ఒకవేళ (it != flash_attn_vec_pipelines.end()) {
            తిరిగి it->second;
        }

        ggml_webgpu_flash_attn_vec_decisions decisions = {};
        decisions.kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(context.wg_mem_హద్దు_బైట్లు, కీలకం.common.తల_dim_qk,
                                                                   కీలకం.common.తల_dim_v, కీలకం.common.has_ముసుగు,
                                                                   కీలకం.common.k_ప్రత్యక్ష || కీలకం.common.v_ప్రత్యక్ష);
        decisions.wg_కొలత = context.max_subgroup_కొలత;

        std::string              variant = "flash_attn_vec";
        std::vector<std::string> నిర్వచిస్తుంది =
            ggml_webgpu_flash_attn_common_నిర్వచిస్తుంది(కీలకం.common, variant, 1u, decisions.kv_tile, decisions.wg_కొలత);
        ఒకవేళ (కీలకం.common.has_ముసుగు) {
            నిర్వచిస్తుంది.push_back("BLK");
            variant.resize(variant.size() - (పరిమాణం("_mask") - 1));
            variant += "_mask_blk";
        }

        uint32_t       d_విభజించు = context.min_subgroup_కొలత;
        స్థిరరాశి uint32_t D       = కీలకం.common.తల_dim_qk | కీలకం.common.తల_dim_v;
        స్థిరరాశి uint32_t D_lsb   = D & (~(D - 1u));
        d_విభజించు                = std::min(std::min(context.min_subgroup_కొలత, 4u), std::max(D_lsb / 4u, 1u));

        నిర్వచిస్తుంది.push_back(std::string("D_SPLIT=") + std::to_string(d_విభజించు));
        variant += "_dsplit" + std::to_string(d_విభజించు);

        స్వయంచాలక            pipeline_decisions = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>(decisions);
        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_విభజించు, నిర్వచిస్తుంది), variant);
        pipeline.context              = pipeline_decisions;
        flash_attn_vec_pipelines[కీలకం] = pipeline;
        తిరిగి flash_attn_vec_pipelines[కీలకం];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_కీలకం కీలకం = {};
        కీలకం.kv_tile                                 = kv_tile;
        స్వయంచాలక it                                     = flash_attn_blk_pipelines.find(కీలకం);
        ఒకవేళ (it != flash_attn_blk_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "flash_attn_vec_blk";

        నిర్వచిస్తుంది.push_back(std::string("KV_TILE=") + std::to_string(కీలకం.kv_tile));
        variant += std::string("_kvt") + std::to_string(కీలకం.kv_tile);

        uint32_t wg_కొలత = 1;
        అయితే ((wg_కొలత << 1) <= context.max_wg_కొలత) {
            wg_కొలత <<= 1;
        }
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(wg_కొలత));
        variant += std::string("_wg") + std::to_string(wg_కొలత);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, నిర్వచిస్తుంది), variant);
        flash_attn_blk_pipelines[కీలకం] = pipeline;
        తిరిగి flash_attn_blk_pipelines[కీలకం];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_కీలకం కీలకం = {};
        కీలకం.తల_dim_v                                     = (uint32_t) context.src2->ne[0];
        కీలకం.dst_type                                       = context.dst->type;
        కీలకం.wg_కొలత                                        = context.max_wg_కొలత;
        స్వయంచాలక it                                            = flash_attn_vec_reduce_pipelines.find(కీలకం);
        ఒకవేళ (it != flash_attn_vec_reduce_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "flash_attn_vec_reduce";

        ఎంపిక (కీలకం.dst_type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("DST_F32");
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("DST_F16");
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported dst type for flash attention vec reduce shader");
        }
        variant += std::string("_dst") + ggml_type_name(కీలకం.dst_type);

        నిర్వచిస్తుంది.push_back(std::string("HEAD_DIM_V=") + std::to_string(కీలకం.తల_dim_v));
        variant += std::string("_hsv") + std::to_string(కీలకం.తల_dim_v);

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));
        variant += std::string("_wg") + std::to_string(context.max_wg_కొలత);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, నిర్వచిస్తుంది), variant);
        flash_attn_vec_reduce_pipelines[కీలకం] = pipeline;
        తిరిగి flash_attn_vec_reduce_pipelines[కీలకం];
    }

    webgpu_pipeline get_cpy_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_కీలకం కీలకం = {};
        కీలకం.src_type                     = context.src0->type;
        కీలకం.dst_type                     = context.dst->type;

        స్వయంచాలక it = cpy_pipelines.find(కీలకం);
        ఒకవేళ (it != cpy_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "cpy";

        ఎంపిక (కీలకం.src_type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("SRC_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("SRC_F16");
                variant += "_f16";
                ఆపు;
            స్థితి GGML_TYPE_I32:
                నిర్వచిస్తుంది.push_back("SRC_I32");
                variant += "_i32";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        ఎంపిక (కీలకం.dst_type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("DST_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("DST_F16");
                variant += "_f16";
                ఆపు;
            స్థితి GGML_TYPE_I32:
                నిర్వచిస్తుంది.push_back("DST_I32");
                variant += "_i32";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_cpy, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[కీలకం]       = pipeline;
        తిరిగి cpy_pipelines[కీలకం];
    }

    webgpu_pipeline get_glu_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, తార్కిక src_overlap) {
        ggml_webgpu_glu_pipeline_కీలకం కీలకం = {};
        కీలకం.glu_op                       = ggml_get_glu_op(context.dst);
        కీలకం.type                         = context.dst->type;
        కీలకం.విభజించు                        = (context.src1 != శూన్యసూచిక);
        కీలకం.src_overlap                  = src_overlap;

        స్వయంచాలక it = glu_pipelines.find(కీలకం);
        ఒకవేళ (it != glu_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "glu";

        ఎంపిక (కీలకం.glu_op) {
            స్థితి GGML_GLU_OP_REGLU:
                నిర్వచిస్తుంది.push_back("OP_REGLU");
                variant += "_reglu";
                ఆపు;
            స్థితి GGML_GLU_OP_GEGLU:
                నిర్వచిస్తుంది.push_back("OP_GEGLU");
                variant += "_geglu";
                ఆపు;
            స్థితి GGML_GLU_OP_SWIGLU:
                నిర్వచిస్తుంది.push_back("OP_SWIGLU");
                variant += "_swiglu";
                ఆపు;
            స్థితి GGML_GLU_OP_SWIGLU_OAI:
                నిర్వచిస్తుంది.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                ఆపు;
            స్థితి GGML_GLU_OP_GEGLU_ERF:
                నిర్వచిస్తుంది.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                ఆపు;
            స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                నిర్వచిస్తుంది.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                ఆపు;
            స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                నిర్వచిస్తుంది.push_back("OP_SWIGLU_CLAMP");
                variant += "_swiglu_clamp";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported GLU op");
        }
        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("TYPE_F16");
                variant += "_f16";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        ఒకవేళ (కీలకం.src_overlap) {
            నిర్వచిస్తుంది.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        } లేకపోతే ఒకవేళ (కీలకం.విభజించు) {
            variant += "_split";
        } లేకపోతే {
            నిర్వచిస్తుంది.push_back("NO_SPLIT");
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_glu, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_glu_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        decisions->src_overlap   = కీలకం.src_overlap;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[కీలకం]       = pipeline;
        తిరిగి glu_pipelines[కీలకం];
    }

    webgpu_pipeline get_rope_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_కీలకం కీలకం = {};
        కీలకం.type                          = context.dst->type;
        కీలకం.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        కీలకం.has_ff                        = (context.src2 != శూన్యసూచిక);

        స్వయంచాలక it = rope_pipelines.find(కీలకం);
        ఒకవేళ (it != rope_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "rope";

        ఎంపిక (కీలకం.type) {
            స్థితి GGML_TYPE_F32:
                నిర్వచిస్తుంది.push_back("TYPE_F32");
                variant += "_f32";
                ఆపు;
            స్థితి GGML_TYPE_F16:
                నిర్వచిస్తుంది.push_back("TYPE_F16");
                variant += "_f16";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        ఒకవేళ (కీలకం.inplace) {
            నిర్వచిస్తుంది.push_back("INPLACE");
            variant += "_inplace";
        }

        ఒకవేళ (కీలకం.has_ff) {
            నిర్వచిస్తుంది.push_back("FF_FUNC");
            variant += "_ff";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_rope, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        decisions->inplace       = కీలకం.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[కీలకం]      = pipeline;
        తిరిగి rope_pipelines[కీలకం];
    }

    webgpu_pipeline get_soft_max_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_కీలకం కీలకం = {};
        కీలకం.ముసుగు_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        కీలకం.has_ముసుగు                          = (context.src1 != శూన్యసూచిక);
        కీలకం.has_sink                          = (context.src2 != శూన్యసూచిక);
        కీలకం.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        స్వయంచాలక it = soft_max_pipelines.find(కీలకం);
        ఒకవేళ (it != soft_max_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "soft_max";

        ఒకవేళ (కీలకం.has_ముసుగు) {
            నిర్వచిస్తుంది.push_back("HAS_MASK");
            ఎంపిక (కీలకం.ముసుగు_type) {
                స్థితి GGML_TYPE_F32:
                    నిర్వచిస్తుంది.push_back("MASK_F32");
                    variant += "_mask_f32";
                    ఆపు;
                స్థితి GGML_TYPE_F16:
                    నిర్వచిస్తుంది.push_back("MASK_F16");
                    variant += "_mask_f16";
                    ఆపు;
                అప్రమేయ:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        ఒకవేళ (కీలకం.has_sink) {
            నిర్వచిస్తుంది.push_back("HAS_SINK");
            variant += "_sink";
        }

        ఒకవేళ (కీలకం.inplace) {
            నిర్వచిస్తుంది.push_back("INPLACE");
            variant += "_inplace";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_soft_max, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        decisions->inplace       = కీలకం.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[కీలకం]  = pipeline;
        తిరిగి soft_max_pipelines[కీలకం];
    }

    webgpu_pipeline get_conv2d_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_కీలకం కీలకం = {};
        కీలకం.బరువు_type                     = context.src0->type;
        కీలకం.ఆదానం_type                      = context.src1->type;
        కీలకం.output_type                     = context.dst->type;

        స్వయంచాలక it = conv2d_pipelines.find(కీలకం);
        ఒకవేళ (it != conv2d_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "conv_2d";

        స్వయంచాలక నెట్టు_type_నిర్వచిస్తుంది = [&](స్థిరరాశి అక్షరం * prefix, ggml_type type) {
            std::string s_ఉపసర్గ = prefix;
            ఒకవేళ (type == GGML_TYPE_F32) {
                నిర్వచిస్తుంది.push_back(s_ఉపసర్గ + "=f32");
            } లేకపోతే ఒకవేళ (type == GGML_TYPE_F16) {
                నిర్వచిస్తుంది.push_back(s_ఉపసర్గ + "=f16");
            } లేకపోతే {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        నెట్టు_type_నిర్వచిస్తుంది("WEIGHT_TYPE", కీలకం.బరువు_type);
        నెట్టు_type_నిర్వచిస్తుంది("INPUT_TYPE", కీలకం.ఆదానం_type);
        నెట్టు_type_నిర్వచిస్తుంది("OUTPUT_TYPE", కీలకం.output_type);

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_conv2d, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[కీలకం]    = pipeline;
        తిరిగి conv2d_pipelines[కీలకం];
    }

    // whcn selects the input layout: contiguous WHCN vs contiguous-channels CWHN
    webgpu_pipeline get_conv2d_dw_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context, తార్కిక whcn) {
        ggml_webgpu_conv2d_dw_pipeline_కీలకం కీలకం = {};
        కీలకం.బరువు_type                        = context.src0->type;
        కీలకం.ఆదానం_type                         = context.src1->type;
        కీలకం.output_type                        = context.dst->type;
        కీలకం.whcn                               = whcn;

        స్వయంచాలక it = conv2d_dw_pipelines.find(కీలకం);
        ఒకవేళ (it != conv2d_dw_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = whcn ? "conv_2d_dw_whcn" : "conv_2d_dw_cwhn";

        స్వయంచాలక నెట్టు_type_నిర్వచిస్తుంది = [&](స్థిరరాశి అక్షరం * prefix, ggml_type type) {
            std::string s_ఉపసర్గ = prefix;
            ఒకవేళ (type == GGML_TYPE_F32) {
                నిర్వచిస్తుంది.push_back(s_ఉపసర్గ + "=f32");
            } లేకపోతే ఒకవేళ (type == GGML_TYPE_F16) {
                నిర్వచిస్తుంది.push_back(s_ఉపసర్గ + "=f16");
            } లేకపోతే {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        నెట్టు_type_నిర్వచిస్తుంది("WEIGHT_TYPE", కీలకం.బరువు_type);
        నెట్టు_type_నిర్వచిస్తుంది("INPUT_TYPE", కీలకం.ఆదానం_type);
        నెట్టు_type_నిర్వచిస్తుంది("OUTPUT_TYPE", కీలకం.output_type);

        ఒకవేళ (whcn) {
            నిర్వచిస్తుంది.push_back("WHCN");
        }
        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_conv2d_dw, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_dw_pipelines[కీలకం] = pipeline;
        తిరిగి conv2d_dw_pipelines[కీలకం];
    }

    webgpu_pipeline get_im2col_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_కీలకం కీలకం = {};
        కీలకం.ఆదానం_type                      = context.src1->type;
        కీలకం.output_type                     = context.dst->type;

        స్వయంచాలక it = im2col_pipelines.find(కీలకం);
        ఒకవేళ (it != im2col_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "im2col";

        స్వయంచాలక నెట్టు_type_నిర్వచిస్తుంది = [&](స్థిరరాశి అక్షరం * prefix, ggml_type type) {
            std::string s_ఉపసర్గ = prefix;
            ఒకవేళ (type == GGML_TYPE_F32) {
                నిర్వచిస్తుంది.push_back(s_ఉపసర్గ + "=f32");
            } లేకపోతే ఒకవేళ (type == GGML_TYPE_F16) {
                నిర్వచిస్తుంది.push_back(s_ఉపసర్గ + "=f16");
            } లేకపోతే {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        నెట్టు_type_నిర్వచిస్తుంది("INPUT_TYPE", కీలకం.ఆదానం_type);
        నెట్టు_type_నిర్వచిస్తుంది("OUTPUT_TYPE", కీలకం.output_type);

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_im2col, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[కీలకం]    = pipeline;
        తిరిగి im2col_pipelines[కీలకం];
    }

    webgpu_pipeline get_upscale_pipeline(స్థిరరాశి ggml_webgpu_shader_lib_context & context) {
        స్థిరరాశి uint32_t బహుళకం_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        స్థిరరాశి uint32_t పునాది_బహుళకం  = బహుళకం_flags & 0xFFu;
        స్థిరరాశి తార్కిక     antialias  = (బహుళకం_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_కీలకం కీలకం = {};
        కీలకం.ఆదానం_type                       = context.src0->type;
        కీలకం.output_type                      = context.dst->type;
        కీలకం.పునాది_బహుళకం                        = పునాది_బహుళకం;
        కీలకం.antialias                        = antialias;

        స్వయంచాలక it = upscale_pipelines.find(కీలకం);
        ఒకవేళ (it != upscale_pipelines.end()) {
            తిరిగి it->second;
        }

        std::vector<std::string> నిర్వచిస్తుంది;
        std::string              variant = "upscale";

        ఒకవేళ (కీలకం.ఆదానం_type == GGML_TYPE_F16) {
            నిర్వచిస్తుంది.push_back("SRC_F16");
            variant += "_src_f16";
        } లేకపోతే {
            variant += "_src_f32";
        }

        ఒకవేళ (కీలకం.output_type == GGML_TYPE_F16) {
            నిర్వచిస్తుంది.push_back("DST_F16");
            variant += "_dst_f16";
        } లేకపోతే {
            variant += "_dst_f32";
        }

        ఎంపిక (పునాది_బహుళకం) {
            స్థితి GGML_SCALE_బహుళకం_సమీప:
                నిర్వచిస్తుంది.push_back("NEAREST");
                variant += "_nearest";
                ఆపు;
            స్థితి GGML_SCALE_బహుళకం_BILINEAR:
                నిర్వచిస్తుంది.push_back("BILINEAR");
                variant += "_bilinear";
                ఆపు;
            స్థితి GGML_SCALE_బహుళకం_BICUBIC:
                నిర్వచిస్తుంది.push_back("BICUBIC");
                variant += "_bicubic";
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("Unsupported upscale mode");
        }

        ఒకవేళ (antialias) {
            నిర్వచిస్తుంది.push_back("ANTIALIAS");
            variant += "_aa";
        }

        నిర్వచిస్తుంది.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_కొలత));

        స్వయంచాలక processed           = preprocessor.preprocess(wgsl_upscale, నిర్వచిస్తుంది);
        స్వయంచాలక decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_కొలత       = context.max_wg_కొలత;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[కీలకం]   = pipeline;
        తిరిగి upscale_pipelines[కీలకం];
    }

  అంతరంగ:
    స్థిర webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_సంకేతం,
                                                       std::string    label) {
        wgpu::Shader_ఆధారం_WGSL shader_ఆధారం;
        shader_ఆధారం.code = shader_సంకేతం.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.తదుపరి_లోపల_Chain = &shader_ఆధారం;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.నమోదు_బిందువు_2 = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = శూన్యసూచిక;  // nullptr means auto layout
        తిరిగి { device.CreateComputePipeline(&pipeline_desc), label };
    }
};

#అయితే_ముగింపు  // GGML_WEBGPU_SHADER_LIB_HPP
