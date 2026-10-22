#ಸೂಚನೆ clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>
#ಅನ್ತರ್ಭಾವ <HAP_compute_res.h>

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <stdatomic.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-dump.h"
#ಅನ್ತರ್ಭಾವ "hvx-arith.h"
#ಅನ್ತರ್ಭಾವ "hvx-reduce.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "matmul-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ದಶಮಾಂಶ        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_ಬೈಟುಗಳು;
    dma_addr_t    act_dma_addr;
    dma_addr_t    ತೂಕ;
    dma_ಸರತಿ *   ತೂಕ_dma;
    ಪೂರ್ಣಾಂಕ           m;
    ಪೂರ್ಣಾಂಕ           k;
    ಪೂರ್ಣಾಂಕ           n;
    ಪೂರ್ಣಾಂಕ           act_stride;
    uint32_t      act_elem_ಗಾತ್ರ;  // 4=F32 src1, 2=F16 src1
    ಪೂರ್ಣಾಂಕ           ತೂಕ_stride;
    ಪೂರ್ಣಾಂಕ           ತೂಕ_type;
    ಪೂರ್ಣಾಂಕ           dst_stride;
    ಪೂರ್ಣಾಂಕ           dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    uint32_t      bias_stride;
    ಪೂರ್ಣಾಂಕ           tile_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ           aligned_tile_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ           ne02;
    ಪೂರ್ಣಾಂಕ           ne03;
    ಪೂರ್ಣಾಂಕ           ne12;
    ಪೂರ್ಣಾಂಕ           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    ಪೂರ್ಣಾಂಕ           r2;
    ಪೂರ್ಣಾಂಕ           r3;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_r2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_r3;
} hmx_mm_batched_params_t;

ಸ್ಥಿರ bool htp_matmul_has_extended_ತೂಕ(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx, uint32_t n_weights) {
    ಗಾಗಿ (uint32_t i = 0; i < n_weights; ++i) {
        ಆದರೆ (htp_tensor_is_extended(octx->src[i])) {
            ಹಿಂತಿರುಗಿ true;
        }
    }
    ಹಿಂತಿರುಗಿ false;
}

ರಚನೆ htp_mm_context {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * type;
    ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * act;

    ಶೂನ್ಯ (*vec_dot_1x1)(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s0,
         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx0,
         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy0);

    ಶೂನ್ಯ (*vec_dot_2x1)(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s0,
         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx1,
         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy0);

    ಶೂನ್ಯ (*vec_dot_2x2)(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s0, ದಶಮಾಂಶ * ಸೀಮಿತ s1,
         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx1,
         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy1);

    ಶೂನ್ಯ (*vec_dot_32x1)(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s,
         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx,
         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy, uint32_t valid_ಅಡ್ಡಸಾಲುಗಳು,
         ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ sz);

    ಶೂನ್ಯ (*vec_dot_32x2)(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s0, ದಶಮಾಂಶ * ಸೀಮಿತ s1,
         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx,
         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy1, uint32_t valid_ಅಡ್ಡಸಾಲುಗಳು,
         ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ sz0, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ sz1);

    // Precomputed values
    uint32_t src0_nrows_ಪ್ರತಿ_thread;
    uint32_t src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ;
    uint32_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded;
    uint32_t act_nrows;
    uint32_t cur_m_ಆರಂಭ;
    uint32_t cur_m_ಅಡ್ಡಸಾಲುಗಳು;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು mm_div_ne12_ne1;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು mm_div_ne1;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು mm_div_r2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು mm_div_r3;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_ಮೊದಲ[WORK_ಸರತಿ_MAX_N_ಎಳೆಗಳು];
    uint32_t          quant_ib_ಕೊನೆಯ[WORK_ಸರತಿ_MAX_N_ಎಳೆಗಳು];
    uint32_t          quant_r[WORK_ಸರತಿ_MAX_N_ಎಳೆಗಳು];
    uint32_t          quant_c[WORK_ಸರತಿ_MAX_N_ಎಳೆಗಳು];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ಸ್ಥಿರಾಂಕ uint32_t * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts;
    ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t mapping_stride;

    // Dynamic VTCM pointers allocated sequentially
    uint8_t * vtcm_src0;
    uint8_t * vtcm_act;
    uint8_t * vtcm_bias;
    uint8_t * vtcm_dst;
    uint8_t * vtcm_act_raw;

    // Cached strides
    uint32_t vtcm_src0_stride;
    uint32_t vtcm_act_stride;
    uint32_t vtcm_act_raw_stride;

    // Cached thread offsets/sizes
    uint32_t vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t vtcm_act_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;
};

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ htp_mm_init_context(
    ರಚನೆ htp_ops_context * octx,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams
) {
    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, (uint32_t) kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಆದರೆ (kparams->n_hmx) {
        ಆದರೆ (kparams->n_act_ಎಳೆಗಳು <= 0 || kparams->n_act_ಎಳೆಗಳು > (int32_t) octx->n_ಎಳೆಗಳು) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
        }
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
    0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00, 0x00,
    0x00, 0x11, 0x10, 0x10, 0x10, 0x02, 0x00, 0x04, 0x00, 0x01, 0x02, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04,
    0x00, 0x00, 0x22, 0x20, 0x20, 0x20, 0x21, 0x22, 0x20, 0x24, 0x04, 0x00, 0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x11, 0x12, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08,
    0x01, 0x02, 0x00, 0x04, 0x44, 0x40, 0x40, 0x40, 0x41, 0x40, 0x40, 0x40, 0x42, 0x40, 0x44, 0x40, 0x41, 0x42, 0x48,
    0x48, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x12, 0x10, 0x10, 0x10, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00,
    0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x22, 0x20, 0x24, 0x20, 0x21, 0x22, 0x20, 0x20,
};

// IQ4_NL dequantization LUT: maps 4-bit index (0-15) to int8 kvalue
// kvalues: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#ವ್ಯಾಖ್ಯಾನಿಸು htp_matmul_tensors_preamble                                 \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0];         \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src1 = octx->src[1];         \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src2 = octx->src[2];         \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ  dst = octx->dst;            \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ  act = src1;                 \
                                                                    \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];                              \
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src0->ne[1];                              \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];                              \
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    ಸ್ಥಿರಾಂಕ uint32_t ne10 = src1->ne[0];                              \
    ಸ್ಥಿರಾಂಕ uint32_t ne11 = src1->ne[1];                              \
    ಸ್ಥಿರಾಂಕ uint32_t ne12 = src1->ne[2];                              \
    ಸ್ಥಿರಾಂಕ uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    ಸ್ಥಿರಾಂಕ uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    ಸ್ಥಿರಾಂಕ uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    ಸ್ಥಿರಾಂಕ uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    ಸ್ಥಿರಾಂಕ uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];                                \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];                                \
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];                                \
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    ಸ್ಥಿರಾಂಕ uint32_t nb00 = src0->nb[0];                              \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src0->nb[1];                              \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2];                              \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    ಸ್ಥಿರಾಂಕ uint32_t nb10 = src1->nb[0];                              \
    ಸ್ಥಿರಾಂಕ uint32_t nb11 = src1->nb[1];                              \
    ಸ್ಥಿರಾಂಕ uint32_t nb12 = src1->nb[2];                              \
    ಸ್ಥಿರಾಂಕ uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    ಸ್ಥಿರಾಂಕ uint32_t nb0 = dst->nb[0];                                \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];                                \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];                                \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

#ವ್ಯಾಖ್ಯಾನಿಸು htp_matmul_preamble                                         \
    ರಚನೆ htp_mm_context * mmctx  = ದತ್ತಾಂಶ;                          \
    ರಚನೆ htp_ops_context * octx  = mmctx->octx;                   \
    dma_ಸರತಿ * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_ಪ್ರತಿ_thread = mmctx->src0_nrows_ಪ್ರತಿ_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#ಅನ್ತರ್ಭಾವ "hvx-mm-kernels-tiled.h"
#ಅನ್ತರ್ಭಾವ "hvx-mm-kernels-float.h"
#ಅನ್ತರ್ಭಾವ "hmx-mm-kernels-tiled.h"

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mm_2d_tiled(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_matmul_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ - mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ uint32_t act_nrows = mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು ? mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು : (ne11 * ne12 * ne13);
    ಸ್ಥಿರಾಂಕ uint32_t cur_m_ಆರಂಭ = mmctx->cur_m_ಆರಂಭ;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು  = mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb1;
    ಸ್ಥಿರಾಂಕ size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb11;
    ಸ್ಥಿರಾಂಕ size_t act_stride = mmctx->vtcm_act_stride;
    ಸ್ಥಿರಾಂಕ size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * ಸೀಮಿತ vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  * ith;
    uint8_t * ಸೀಮಿತ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * ಸೀಮಿತ act_ದತ್ತಾಂಶ = mmctx->vtcm_act;

    ಸ್ಥಿರಾಂಕ dma_addr_t src0_ಅಡ್ಡಸಾಲು = src0->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ uint32_t tile_ಗಾತ್ರ = htp_tensor_get_tile_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_ಅಡ್ಡಸಾಲು_stride = n_k_tiles_w * tile_ಗಾತ್ರ;
    uint32_t tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned = n_k_tiles_w * aligned_tile_ಗಾತ್ರ;

    uint32_t ct_ಆರಂಭ = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು / 32;
    uint32_t ct_ಕೊನೆ   = (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು + 31) / 32;

    uint32_t ತಳ್ಳು_ct = ct_ಆರಂಭ;
    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಗಾಗಿ (uint32_t d = 0; d < n_prefetch && ತಳ್ಳು_ct < ct_ಕೊನೆ; d++, ತಳ್ಳು_ct++) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + d * tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned,
                           src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride), aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
        }
    }

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಗಾಗಿ (uint32_t ct = ct_ಆರಂಭ; ct < ct_ಕೊನೆ; ct++) {
        ಸ್ಥಿರಾಂಕ uint8_t * w_tile = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        ಪೂರ್ಣಾಂಕ valid_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ)ne0 - (ಪೂರ್ಣಾಂಕ)(ct * 32);
        valid_ಅಡ್ಡಸಾಲುಗಳು = MIN(32, MAX(0, valid_ಅಡ್ಡಸಾಲುಗಳು));

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        ಗಾಗಿ (; ir1 + 1 < act_nrows; ir1 += 2) {
            ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_col0 = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (ir1+0) * act_stride);
            ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_col1 = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (ir1+1) * act_stride);
            ದಶಮಾಂಶ * ಸೀಮಿತ dst_row0 = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ((cur_m_ಆರಂಭ + ir1+0) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ));
            ದಶಮಾಂಶ * ಸೀಮಿತ dst_row1 = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ((cur_m_ಆರಂಭ + ir1+1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ));

            ದಶಮಾಂಶ * dst_ptr0 = &dst_row0[ct * 32];
            ದಶಮಾಂಶ * dst_ptr1 = &dst_row1[ct * 32];

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src2_ptr0 = ಖಾಲಿ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src2_ptr1 = ಖಾಲಿ;
            ಆದರೆ (src2) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src2_row0 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src2->ದತ್ತಾಂಶ + ((cur_m_ಆರಂಭ + ir1+0) * src2_stride));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src2_row1 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src2->ದತ್ತಾಂಶ + ((cur_m_ಆರಂಭ + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_ಅಡ್ಡಸಾಲುಗಳು, src2_ptr0, src2_ptr1);
        }

        ಗಾಗಿ (; ir1 < act_nrows; ++ir1) {
            ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + ir1 * act_stride);
            ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು          = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ((cur_m_ಆರಂಭ + ir1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ));
            ದಶಮಾಂಶ * dst_ptr = &dst_ಅಡ್ಡಸಾಲು[ct * 32];

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src2_ptr = ಖಾಲಿ;
            ಆದರೆ (src2) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src2_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src2->ದತ್ತಾಂಶ + ((cur_m_ಆರಂಭ + ir1) * src2_stride));
                src2_ptr = &src2_ಅಡ್ಡಸಾಲು[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ಲಂಬಸಾಲು, valid_ಅಡ್ಡಸಾಲುಗಳು, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        ಆದರೆ (ತಳ್ಳು_ct < ct_ಕೊನೆ) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(w_tile, src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                           aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
            ತಳ್ಳು_ct++;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mv_2d_tiled(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_matmul_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ - mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು  = mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb1;
    ಸ್ಥಿರಾಂಕ size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb11;
    ಸ್ಥಿರಾಂಕ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * act_ದತ್ತಾಂಶ = mmctx->vtcm_act;

    ದಶಮಾಂಶ * tmp = (ದಶಮಾಂಶ *) vtcm_dst_ptr;

    ಸ್ಥಿರಾಂಕ dma_addr_t src0_ಅಡ್ಡಸಾಲು = src0->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) act_ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಲಂಬಸಾಲು          = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ uint32_t tile_ಗಾತ್ರ = htp_tensor_get_tile_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_ಅಡ್ಡಸಾಲು_stride = n_k_tiles_w * tile_ಗಾತ್ರ;
    uint32_t tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned = n_k_tiles_w * aligned_tile_ಗಾತ್ರ;

    uint32_t ct_ಆರಂಭ = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು / 32;
    uint32_t ct_ಕೊನೆ   = (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು + 31) / 32;

    uint32_t ತಳ್ಳು_ct = ct_ಆರಂಭ;
    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಆದರೆ (src2) {
            ದಶಮಾಂಶ * vtcm_bias_ptr = (ದಶಮಾಂಶ *) mmctx->vtcm_bias + src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
            ಸ್ಥಿರಾಂಕ dma_addr_t src2_addr = src2->ದತ್ತಾಂಶ + src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಗಾತ್ರ(ದಶಮಾಂಶ);
            ಪೂರ್ಣಾಂಕ ತುಂಡು_ಗಾತ್ರ = (ಪೂರ್ಣಾಂಕ)MIN(src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne0) - (ಪೂರ್ಣಾಂಕ)src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
            ಆದರೆ (ತುಂಡು_ಗಾತ್ರ > 0) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_bias_ptr, src2_addr),
                               ತುಂಡು_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), ತುಂಡು_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), ತುಂಡು_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), 1);
                dma_ಸರತಿ_ತೆಗೆ_nowait(dma_q);
            }
        }
        ಗಾಗಿ (uint32_t d = 0; d < n_prefetch && ತಳ್ಳು_ct < ct_ಕೊನೆ; d++, ತಳ್ಳು_ct++) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + d * tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned,
                           src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride), aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
        }
    }

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಗಾಗಿ (uint32_t ct = ct_ಆರಂಭ; ct < ct_ಕೊನೆ; ct++) {
        ಸ್ಥಿರಾಂಕ uint8_t * w_tile = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        ದಶಮಾಂಶ * dst_ptr = &tmp[ct * 32 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು];
        ಪೂರ್ಣಾಂಕ valid_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ)ne0 - (ಪೂರ್ಣಾಂಕ)(ct * 32);
        valid_ಅಡ್ಡಸಾಲುಗಳು = MIN(32, MAX(0, valid_ಅಡ್ಡಸಾಲುಗಳು));

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ಲಂಬಸಾಲು, valid_ಅಡ್ಡಸಾಲುಗಳು, ಖಾಲಿ);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        ಆದರೆ (ತಳ್ಳು_ct < ct_ಕೊನೆ) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(w_tile, src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                           aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
            ತಳ್ಳು_ct++;
        }
    }

    ಪೂರ್ಣಾಂಕ copy_cnt = (ಪೂರ್ಣಾಂಕ)MIN(src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne0) - (ಪೂರ್ಣಾಂಕ)src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಆದರೆ (copy_cnt > 0) {
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ct_ಕೊನೆ);
        ಆದರೆ (src2) {
            hvx_ಕೂಡಿಸು_f32_uuu((uint8_t *) &dst_ಲಂಬಸಾಲು[src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು],
                            (ಸ್ಥಿರಾಂಕ uint8_t *) tmp,
                            (ಸ್ಥಿರಾಂಕ uint8_t *) ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) mmctx->vtcm_bias + src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು),
                            copy_cnt);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_copy_f32_ua((uint8_t *) &dst_ಲಂಬಸಾಲು[src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_ಕೊನೆ);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mm_nx_2d_tiled(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_mm_context * mmctx = ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = mmctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_weights = kparams->n_weights;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ act = octx->src[n_weights]; /* x */
    ಸ್ಥಿರಾಂಕ uint32_t ne10 = act->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ಸ್ಥಿರಾಂಕ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * ಸೀಮಿತ vtcm_ತೂಕ_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * ಸೀಮಿತ act_ದತ್ತಾಂಶ        = mmctx->vtcm_act;

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ಗಾಗಿ (uint32_t widx = 0; widx < n_weights; widx++) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src_w = octx->src[widx];
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst   = octx->dsts[widx];
        ಆದರೆ (!src_w || !dst) ಮುಂದುವರಿ;
        dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

        ಸ್ಥಿರಾಂಕ uint32_t tile_ಗಾತ್ರ = htp_tensor_get_tile_ಗಾತ್ರ(src_w->type);
        ಸ್ಥಿರಾಂಕ uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(src_w->type);
        ಸ್ಥಿರಾಂಕ uint32_t ne00 = src_w->ne[0];
        ಸ್ಥಿರಾಂಕ uint32_t ne01 = src_w->ne[1];
        ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->nb[1];
        ಸ್ಥಿರಾಂಕ dma_addr_t src_w_ಅಡ್ಡಸಾಲು = src_w->ದತ್ತಾಂಶ;

        ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_stride = n_k_tiles_w * tile_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned = n_k_tiles_w * aligned_tile_ಗಾತ್ರ;

        uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = 0;
        uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = ne01;
        ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
            ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, ಗಾತ್ರ(ದಶಮಾಂಶ));
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ne01, can_ವಿಭಜಿಸು ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = range.ಆರಂಭ;
            src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = range.ಆರಂಭ + range.ಎಣಿಕೆ_2;
        }

        ಸ್ಥಿರಾಂಕ uint32_t nrows = src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
        uint32_t src0_nrows_ಪ್ರತಿ_thread = fastdiv(nrows + nth - 1, &octx->n_ಎಳೆಗಳು_div);
        src0_nrows_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ(src0_nrows_ಪ್ರತಿ_thread, 32);

        ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread * ith;
        ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು);
        ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಕೊನೆ_ಅಡ್ಡಸಾಲು) ಮುಂದುವರಿ;

        uint32_t ct_ಆರಂಭ = ಆರಂಭ_ಅಡ್ಡಸಾಲು / 32;
        uint32_t ct_ಕೊನೆ   = (ಕೊನೆ_ಅಡ್ಡಸಾಲು + 31) / 32;

        uint32_t ತಳ್ಳು_ct = ct_ಆರಂಭ;
        ಗಾಗಿ (uint32_t d = 0; d < n_prefetch && ತಳ್ಳು_ct < ct_ಕೊನೆ; d++, ತಳ್ಳು_ct++) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_ತೂಕ_ptr + d * tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned,
                            src_w_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride), aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
        }

        ಗಾಗಿ (uint32_t ct = ct_ಆರಂಭ; ct < ct_ಕೊನೆ; ct++) {
            ಸ್ಥಿರಾಂಕ uint8_t * w_tile = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
            ಪೂರ್ಣಾಂಕ valid_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ)dst->ne[0] - (ಪೂರ್ಣಾಂಕ)(ct * 32);
            valid_ಅಡ್ಡಸಾಲುಗಳು = MIN(32, MAX(0, valid_ಅಡ್ಡಸಾಲುಗಳು));

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            ಗಾಗಿ (; ir1 + 1 < act_nrows; ir1 += 2) {
                ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_col0 = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (ir1+0) * act_stride);
                ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_col1 = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (ir1+1) * act_stride);

                ದಶಮಾಂಶ * ಸೀಮಿತ dst_row0 = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ((ir1+0) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ));
                ದಶಮಾಂಶ * ಸೀಮಿತ dst_row1 = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ((ir1+1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ));
                ದಶಮಾಂಶ * dst_ptr0 = &dst_row0[ct * 32];
                ದಶಮಾಂಶ * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_ಅಡ್ಡಸಾಲುಗಳು, ಖಾಲಿ, ಖಾಲಿ);
            }

            ಗಾಗಿ (; ir1 < act_nrows; ++ir1) {
                ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + ir1 * act_stride);
                ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + (ir1 * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ));
                ದಶಮಾಂಶ * dst_ptr = &dst_ಅಡ್ಡಸಾಲು[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ಲಂಬಸಾಲು, valid_ಅಡ್ಡಸಾಲುಗಳು, ಖಾಲಿ);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ಆದರೆ (ತಳ್ಳು_ct < ct_ಕೊನೆ) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(w_tile, src_w_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                                aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
                ತಳ್ಳು_ct++;
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mm_transfer_act_dma(
    ರಚನೆ htp_ops_context * octx,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * act,
    uint8_t * dst_ತಳ,
    size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    uint32_t m_ಆರಂಭ,
    uint32_t m_ಅಡ್ಡಸಾಲುಗಳು
) {
    ಆದರೆ (m_ಅಡ್ಡಸಾಲುಗಳು == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    dma_ಸರತಿ * dma_q = octx->ctx->dma[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = act->ne[0];
    ಸ್ಥಿರಾಂಕ size_t elem_ಗಾತ್ರ = (act->type == HTP_TYPE_F16) ? ಗಾತ್ರ(__fp16) : ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = ne0 * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t act_nb1 = act->nb[1];
    ಸ್ಥಿರಾಂಕ dma_addr_t act_ತಳ = act->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    ಆದರೆ (is_contiguous) {
        ಸ್ಥಿರಾಂಕ dma_addr_t src_addr = act_ತಳ + m_ಆರಂಭ * act_nb1;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_ತಳ, src_addr),
                       dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_nb1, ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, m_ಅಡ್ಡಸಾಲುಗಳು);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ಸ್ಥಿರಾಂಕ bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        ಗಾಗಿ (uint32_t ir = 0; ir < m_ಅಡ್ಡಸಾಲುಗಳು; ++ir) {
            ಸ್ಥಿರಾಂಕ uint32_t ir1 = m_ಆರಂಭ + ir;
            uint32_t i11, i12, i13;
            ಆದರೆ (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                ಸ್ಥಿರಾಂಕ uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } ಇಲ್ಲದಿದ್ದರೆ {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                ಸ್ಥಿರಾಂಕ uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            ಸ್ಥಿರಾಂಕ dma_addr_t ಅಡ್ಡಸಾಲು_src = act_ತಳ + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * ಅಡ್ಡಸಾಲು_dst = dst_ತಳ + ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ಅಡ್ಡಸಾಲು_dst, ಅಡ್ಡಸಾಲು_src),
                           dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_nb1, ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, 1);
            dma_ಸರತಿ_ತೆಗೆ(dma_q);
        }
    }

    ಆದರೆ (dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ > ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು) {
        ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_ಆರಂಭ);
        ಸ್ಥಿರಾಂಕ uint32_t pad_elems = (dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ - ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು) / elem_ಗಾತ್ರ;
        ಆದರೆ (elem_ಗಾತ್ರ == ಗಾತ್ರ(ದಶಮಾಂಶ)) {
            ಗಾಗಿ (uint32_t ir = 0; ir < m_ಅಡ್ಡಸಾಲುಗಳು; ++ir) {
                hvx_splat_f32_u(dst_ತಳ + ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ + ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, 0.0f, pad_elems);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (uint32_t ir = 0; ir < m_ಅಡ್ಡಸಾಲುಗಳು; ++ir) {
                hvx_splat_f16_u(dst_ತಳ + ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ + ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_ಆರಂಭ);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು QUANTIZE_IMPL(name, log_name, kernel_fn, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಅಭಿವ್ಯಕ್ತಿ)                                        \
ಸ್ಥಿರ ಶೂನ್ಯ name(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {                                        \
    (ಶೂನ್ಯ) nth;                                                                                            \
    ರಚನೆ htp_mm_context * mmctx = ದತ್ತಾಂಶ;                                                                  \
    ರಚನೆ htp_ops_context * octx = mmctx->octx;                                                           \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = mmctx->act;                                                            \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = src->ne[0];                                                                       \
    ಸ್ಥಿರಾಂಕ uint32_t nrows = mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು ? mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು : mmctx->act_nrows;                       \
    ಸ್ಥಿರಾಂಕ uint32_t nrows_ಪ್ರತಿ_thread = mmctx->n_quant_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;                                      \
                                                                                                           \
    ಸ್ಥಿರಾಂಕ uint32_t ir_ಮೊದಲ = nrows_ಪ್ರತಿ_thread * ith;                                                      \
    ಆದರೆ (ir_ಮೊದಲ >= nrows) {                                                                               \
        ಹಿಂತಿರುಗಿ;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_ಮೊದಲ);                                        \
                                                                                                           \
    uint8_t * ಸೀಮಿತ dst = mmctx->vtcm_act;                                                              \
    ಸ್ಥಿರಾಂಕ uint32_t ir_ಕೊನೆಯ = MIN(ir_ಮೊದಲ + nrows_ಪ್ರತಿ_thread, nrows);                                      \
    ಸ್ಥಿರಾಂಕ size_t raw_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = mmctx->vtcm_act_raw_stride;                                                \
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = (dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಅಭಿವ್ಯಕ್ತಿ);                                                       \
                                                                                                           \
    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ uint8_t *) mmctx->vtcm_act_raw + (raw_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ * ir_ಮೊದಲ); \
    uint8_t * ಸೀಮಿತ dst_ದತ್ತಾಂಶ = (uint8_t *) dst + (dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ * ir_ಮೊದಲ);                             \
    kernel_fn(src_ದತ್ತಾಂಶ, dst_ದತ್ತಾಂಶ, ಖಾಲಿ, ne0, ir_ಕೊನೆಯ - ir_ಮೊದಲ, raw_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_ಮೊದಲ);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled,     "quantize-f32-q8_0_tiled",     quantize_f32_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled,     "quantize-f32-q8_1_tiled",     quantize_f32_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ne0))
QUANTIZE_IMPL(quantize_f32_f16_tiled,      "quantize-f32-f16_tiled",      quantize_f32_f16_tiled_kernel,      htp_mm_f16_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ne0))
QUANTIZE_IMPL(quantize_f16_f16_tiled,      "quantize-f16-f16_tiled",      quantize_f16_f16_tiled_kernel,      htp_mm_f16_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ne0))
QUANTIZE_IMPL(quantize_f32_f32,            "quantize-f32-f32",            quantize_f32_f32_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,            "quantize-f32-f16",            quantize_f32_f16_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,            "quantize-f16-f16",            quantize_f16_f16_kernel,            mmctx->vtcm_act_stride)

ಸ್ಥಿರ ಶೂನ್ಯ quantize_f32_q8_0_tiled_ಖಂಡ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ರಚನೆ htp_mm_context * mmctx = ದತ್ತಾಂಶ;
    ಆದರೆ (mmctx->quant_ib_ಮೊದಲ[ith] >= mmctx->quant_ib_ಕೊನೆಯ[ith]) {
        ಹಿಂತಿರುಗಿ;
    }
    ರಚನೆ htp_ops_context * octx = mmctx->octx;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ಮೊದಲ[ith]);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_ಖಂಡ_kernel(
        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ಖಾಲಿ,
        src->ne[0],
        mmctx->quant_ib_ಮೊದಲ[ith],
        mmctx->quant_ib_ಕೊನೆಯ[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ಮೊದಲ[ith]);
}

ಸ್ಥಿರ ಶೂನ್ಯ quantize_f32_q8_1_tiled_ಖಂಡ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ರಚನೆ htp_mm_context * mmctx = ದತ್ತಾಂಶ;
    ಆದರೆ (mmctx->quant_ib_ಮೊದಲ[ith] >= mmctx->quant_ib_ಕೊನೆಯ[ith]) {
        ಹಿಂತಿರುಗಿ;
    }
    ರಚನೆ htp_ops_context * octx = mmctx->octx;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ಮೊದಲ[ith]);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_ಖಂಡ_kernel(
        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ಖಾಲಿ,
        src->ne[0],
        mmctx->quant_ib_ಮೊದಲ[ith],
        mmctx->quant_ib_ಕೊನೆಯ[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ಮೊದಲ[ith]);
}

ಸ್ಥಿರ ಶೂನ್ಯ quantize_f32_q8_1_s16_tiled_ಖಂಡ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ರಚನೆ htp_mm_context * mmctx = ದತ್ತಾಂಶ;
    ಆದರೆ (mmctx->quant_ib_ಮೊದಲ[ith] >= mmctx->quant_ib_ಕೊನೆಯ[ith]) {
        ಹಿಂತಿರುಗಿ;
    }
    ರಚನೆ htp_ops_context * octx = mmctx->octx;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ಮೊದಲ[ith]);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_ಖಂಡ_kernel(
        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ಖಾಲಿ,
        src->ne[0],
        mmctx->quant_ib_ಮೊದಲ[ith],
        mmctx->quant_ib_ಕೊನೆಯ[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ಮೊದಲ[ith]);
}

ಸ್ಥಿರ ಶೂನ್ಯ quantize_f32_f16_tiled_ಖಂಡ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ರಚನೆ htp_mm_context * mmctx = ದತ್ತಾಂಶ;
    ಆದರೆ (mmctx->quant_ib_ಮೊದಲ[ith] >= mmctx->quant_ib_ಕೊನೆಯ[ith]) {
        ಹಿಂತಿರುಗಿ;
    }
    ರಚನೆ htp_ops_context * octx = mmctx->octx;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ಮೊದಲ[ith]);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_ಖಂಡ_kernel(
        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ಖಾಲಿ,
        src->ne[0],
        mmctx->quant_ib_ಮೊದಲ[ith],
        mmctx->quant_ib_ಕೊನೆಯ[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ಮೊದಲ[ith]);
}

ಸ್ಥಿರ ಶೂನ್ಯ quantize_f16_f16_tiled_ಖಂಡ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ರಚನೆ htp_mm_context * mmctx = ದತ್ತಾಂಶ;
    ಆದರೆ (mmctx->quant_ib_ಮೊದಲ[ith] >= mmctx->quant_ib_ಕೊನೆಯ[ith]) {
        ಹಿಂತಿರುಗಿ;
    }
    ರಚನೆ htp_ops_context * octx = mmctx->octx;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ಮೊದಲ[ith]);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_ಖಂಡ_kernel(
        (ಸ್ಥಿರಾಂಕ __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ಖಾಲಿ,
        src->ne[0],
        mmctx->quant_ib_ಮೊದಲ[ith],
        mmctx->quant_ib_ಕೊನೆಯ[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ಮೊದಲ[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಕೆಲಸಗಾರ_callback_t htp_mm_act_quant_ಅಡ್ಡಸಾಲು_func(ಪೂರ್ಣಾಂಕ ತೂಕ_type, ಪೂರ್ಣಾಂಕ act_type) {
    ಆದರೆ (ತೂಕ_type == HTP_TYPE_F16) {
        ಹಿಂತಿರುಗಿ (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    ಆದರೆ (ತೂಕ_type == HTP_TYPE_Q2_K) {
        ಹಿಂತಿರುಗಿ quantize_f32_q8_1_s16_tiled;
    }
    ಹಿಂತಿರುಗಿ htp_mm_ತೂಕ_has_ಸ್ಥಾನಾಂತರ(ತೂಕ_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಕೆಲಸಗಾರ_callback_t htp_mm_act_quant_ಖಂಡ_func(ಪೂರ್ಣಾಂಕ ತೂಕ_type, ಪೂರ್ಣಾಂಕ act_type) {
    ಆದರೆ (ತೂಕ_type == HTP_TYPE_F16) {
        ಹಿಂತಿರುಗಿ (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_ಖಂಡ : quantize_f32_f16_tiled_ಖಂಡ;
    }
    ಆದರೆ (ತೂಕ_type == HTP_TYPE_Q2_K) {
        ಹಿಂತಿರುಗಿ quantize_f32_q8_1_s16_tiled_ಖಂಡ;
    }
    ಹಿಂತಿರುಗಿ htp_mm_ತೂಕ_has_ಸ್ಥಾನಾಂತರ(ತೂಕ_type) ? quantize_f32_q8_1_tiled_ಖಂಡ : quantize_f32_q8_0_tiled_ಖಂಡ;
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mm_4d_tiled(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_matmul_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ - mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ uint32_t cur_m_ಅಡ್ಡಸಾಲುಗಳು = mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು ? mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು : (ne11 * ne12 * ne13);
    ಸ್ಥಿರಾಂಕ uint32_t cur_m_ಆರಂಭ = mmctx->cur_m_ಆರಂಭ;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು  = mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb1;
    ಸ್ಥಿರಾಂಕ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * ಸೀಮಿತ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * ಸೀಮಿತ act_ದತ್ತಾಂಶ = mmctx->vtcm_act;

    ಸ್ಥಿರಾಂಕ uint32_t tile_ಗಾತ್ರ = htp_tensor_get_tile_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(src0->type);

    ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_stride = n_k_tiles_w * tile_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned = n_k_tiles_w * aligned_tile_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ತುಂಡು_stride = ((ne01 + 31) / 32) * tile_ಅಡ್ಡಸಾಲು_stride;

    ಸ್ಥಿರಾಂಕ uint32_t ct_ಆರಂಭ = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು / 32;
    ಸ್ಥಿರಾಂಕ uint32_t ct_ಕೊನೆ   = (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು + 31) / 32;

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು || cur_m_ಅಡ್ಡಸಾಲುಗಳು == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_batches = ne12 * ne13;
    ಸ್ಥಿರಾಂಕ uint32_t b_ಆರಂಭ = fastdiv(cur_m_ಆರಂಭ, &kparams->div_ne1);
    uint32_t b_ಕೊನೆ = fastdiv(cur_m_ಆರಂಭ + cur_m_ಅಡ್ಡಸಾಲುಗಳು + ne11 - 1, &kparams->div_ne1);
    b_ಕೊನೆ = MIN(b_ಕೊನೆ, ಒಟ್ಟು_batches);

    uint32_t b_grp_ಆರಂಭ = b_ಆರಂಭ;
    ಆಗ (b_grp_ಆರಂಭ < b_ಕೊನೆ) {
        ಸ್ಥಿರಾಂಕ uint32_t b3 = fastdiv(b_grp_ಆರಂಭ, &kparams->div_ne12);
        ಸ್ಥಿರಾಂಕ uint32_t b2 = b_grp_ಆರಂಭ - b3 * ne12;
        ಸ್ಥಿರಾಂಕ uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ಸ್ಥಿರಾಂಕ uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_ಕೊನೆ = b_grp_ಆರಂಭ + 1;
        ಆಗ (b_grp_ಕೊನೆ < b_ಕೊನೆ) {
            ಸ್ಥಿರಾಂಕ uint32_t cur_b3 = fastdiv(b_grp_ಕೊನೆ, &kparams->div_ne12);
            ಸ್ಥಿರಾಂಕ uint32_t cur_b2 = b_grp_ಕೊನೆ - cur_b3 * ne12;
            ಸ್ಥಿರಾಂಕ uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ಸ್ಥಿರಾಂಕ uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            ಆದರೆ (cur_i02 != i02 || cur_i03 != i03) {
                ನಿಲ್ಲಿಸು;
            }
            b_grp_ಕೊನೆ++;
        }

        ಸ್ಥಿರಾಂಕ uint32_t ತುಂಡು_idx = i03 * ne02 + i02;
        ಸ್ಥಿರಾಂಕ dma_addr_t src0_ತುಂಡು = src0->ದತ್ತಾಂಶ + (size_t) ತುಂಡು_idx * src0_ತುಂಡು_stride;

        uint32_t ತಳ್ಳು_ct = ct_ಆರಂಭ;
        ಗಾಗಿ (uint32_t d = 0; d < n_prefetch && ತಳ್ಳು_ct < ct_ಕೊನೆ; d++, ತಳ್ಳು_ct++) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + d * tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned,
                           src0_ತುಂಡು + (size_t) ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                           aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
        }

        ಗಾಗಿ (uint32_t ct = ct_ಆರಂಭ; ct < ct_ಕೊನೆ; ct++) {
            ಸ್ಥಿರಾಂಕ uint8_t * w_tile = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

            ಪೂರ್ಣಾಂಕ valid_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ)ne0 - (ಪೂರ್ಣಾಂಕ)(ct * 32);
            valid_ಅಡ್ಡಸಾಲುಗಳು = MIN(32, MAX(0, valid_ಅಡ್ಡಸಾಲುಗಳು));

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            ಗಾಗಿ (uint32_t b = b_grp_ಆರಂಭ; b < b_grp_ಕೊನೆ; b++) {
                ಸ್ಥಿರಾಂಕ uint32_t b_m_ಆರಂಭ = b * ne11;
                ಸ್ಥಿರಾಂಕ uint32_t m_ಮೊದಲ = MAX(cur_m_ಆರಂಭ, b_m_ಆರಂಭ);
                ಸ್ಥಿರಾಂಕ uint32_t m_ಕೊನೆಯ  = MIN(cur_m_ಆರಂಭ + cur_m_ಅಡ್ಡಸಾಲುಗಳು, b_m_ಆರಂಭ + ne11);
                ಆದರೆ (m_ಮೊದಲ >= m_ಕೊನೆಯ) ಮುಂದುವರಿ;

                ಸ್ಥಿರಾಂಕ uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ಸ್ಥಿರಾಂಕ uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_ತಳ = (uint8_t *) dst->ದತ್ತಾಂಶ + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ಸ್ಥಿರಾಂಕ uint32_t chunk_m_ಸ್ಥಾನಾಂತರ = m_ಮೊದಲ - cur_m_ಆರಂಭ;
                ಸ್ಥಿರಾಂಕ uint32_t dst_m_ಸ್ಥಾನಾಂತರ   = m_ಮೊದಲ - b_m_ಆರಂಭ;
                ಸ್ಥಿರಾಂಕ uint32_t batch_nrows    = m_ಕೊನೆಯ - m_ಮೊದಲ;

                uint32_t ir1 = 0;
                ಗಾಗಿ (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_col0 = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (chunk_m_ಸ್ಥಾನಾಂತರ + ir1 + 0) * act_stride);
                    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_col1 = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (chunk_m_ಸ್ಥಾನಾಂತರ + ir1 + 1) * act_stride);
                    ದಶಮಾಂಶ * ಸೀಮಿತ dst_row0 = (ದಶಮಾಂಶ *) (dst_batch_ತಳ + (dst_m_ಸ್ಥಾನಾಂತರ + ir1 + 0) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
                    ದಶಮಾಂಶ * ಸೀಮಿತ dst_row1 = (ದಶಮಾಂಶ *) (dst_batch_ತಳ + (dst_m_ಸ್ಥಾನಾಂತರ + ir1 + 1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
                    ದಶಮಾಂಶ * dst_ptr0 = &dst_row0[ct * 32];
                    ದಶಮಾಂಶ * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_ಅಡ್ಡಸಾಲುಗಳು, ಖಾಲಿ, ಖಾಲಿ);
                }
                ಗಾಗಿ (; ir1 < batch_nrows; ++ir1) {
                    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (chunk_m_ಸ್ಥಾನಾಂತರ + ir1) * act_stride);
                    ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) (dst_batch_ತಳ + (dst_m_ಸ್ಥಾನಾಂತರ + ir1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
                    ದಶಮಾಂಶ * dst_ptr = &dst_ಅಡ್ಡಸಾಲು[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ಲಂಬಸಾಲು, valid_ಅಡ್ಡಸಾಲುಗಳು, ಖಾಲಿ);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ಆದರೆ (ತಳ್ಳು_ct < ct_ಕೊನೆ) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(w_tile, src0_ತುಂಡು + (size_t) ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                               aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
                ತಳ್ಳು_ct++;
            }
        }
        b_grp_ಆರಂಭ = b_grp_ಕೊನೆ;
    }
    ಆದರೆ (src2) {
        hvx_tensor_ಕೂಡಿಸು_f32_grid(dst, src2, cur_m_ಆರಂಭ, cur_m_ಆರಂಭ + cur_m_ಅಡ್ಡಸಾಲುಗಳು, src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mm_2d_flat(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_matmul_preamble;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ಸ್ಥಿರಾಂಕ uint32_t prefetch_ಮುಖವಾಡ = n_prefetch - 1;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ - mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;  // src0 rows
    ಸ್ಥಿರಾಂಕ uint32_t act_nrows = mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು ? mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು : mmctx->act_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t cur_m_ಆರಂಭ = mmctx->cur_m_ಆರಂಭ;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು  = mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ);
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2 = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + ((src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) & ~1U);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb1;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb01;
    ಸ್ಥಿರಾಂಕ size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb11;

    ಸ್ಥಿರಾಂಕ size_t src0_stride = mmctx->vtcm_src0_stride;
    ಸ್ಥಿರಾಂಕ size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * ಸೀಮಿತ vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  * ith;
    uint8_t * ಸೀಮಿತ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * ಸೀಮಿತ act_ದತ್ತಾಂಶ      = mmctx->vtcm_act;

    ಸ್ಥಿರಾಂಕ dma_addr_t src0_ಅಡ್ಡಸಾಲು = src0->ದತ್ತಾಂಶ;


    // Prefill vtcm with src0 rows
    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಗಾಗಿ (uint32_t ir0 = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir0 < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2; ir0 += 2) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is0 = (ir0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು);
            ಆದರೆ (is0 >= (ಪೂರ್ಣಾಂಕ)n_prefetch) {
                ನಿಲ್ಲಿಸು;
            }
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + is0 * src0_stride, src0_ಅಡ್ಡಸಾಲು + ir0 * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ),
                           src0_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 2);
        }
    }

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    // Process src0 rows
    ಗಾಗಿ (uint32_t ir0 = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir0 < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2; ir0 += 2) {
        ಸ್ಥಿರಾಂಕ uint8_t * ss0 = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        ಗಾಗಿ (; ir1 + 1 < act_nrows; ir1 += 2) {
            ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_col0 = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (ir1+0) * act_stride);
            ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_col1 = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (ir1+1) * act_stride);
            ದಶಮಾಂಶ * ಸೀಮಿತ dst_row0 = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ((cur_m_ಆರಂಭ + ir1+0) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ));
            ದಶಮಾಂಶ * ಸೀಮಿತ dst_row1 = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ((cur_m_ಆರಂಭ + ir1+1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        ಗಾಗಿ (; ir1 < act_nrows; ++ir1) {
            ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + ir1 * act_stride);
            ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು          = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ((cur_m_ಆರಂಭ + ir1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ));
            mmctx->vec_dot_2x1(ne00, &dst_ಅಡ್ಡಸಾಲು[ir0], ss0, ss0 + src0_stride, act_ಲಂಬಸಾಲು);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pr0 = (ir0 + n_prefetch);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is0 = (pr0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) & prefetch_ಮುಖವಾಡ;
        ಆದರೆ (pr0 < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + is0 * src0_stride, src0_ಅಡ್ಡಸಾಲು + pr0 * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ),
                           src0_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 2);
        }
    }

    // Process the last row (if any)
    ಆದರೆ (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು != src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2) {
        uint32_t  ir0 = src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is0 = (ir0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) & prefetch_ಮುಖವಾಡ;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + is0 * src0_stride, src0_ಅಡ್ಡಸಾಲು + ir0 * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ),
                       src0_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
        ಸ್ಥಿರಾಂಕ uint8_t * ss0 = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #ಸೂಚನೆ unroll(2)
        ಗಾಗಿ (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + ir1 * act_stride);
            ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು          = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ((cur_m_ಆರಂಭ + ir1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ));
            mmctx->vec_dot_1x1(ne00, &dst_ಅಡ್ಡಸಾಲು[ir0], ss0, act_ಲಂಬಸಾಲು);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    ಆದರೆ (src2) {
        hvx_tensor_ಕೂಡಿಸು_f32_grid(dst, src2, cur_m_ಆರಂಭ, cur_m_ಆರಂಭ + act_nrows, src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mv_2d_flat(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_matmul_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ - mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು  = mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb1;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb01;
    ಸ್ಥಿರಾಂಕ size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb11;

    ಸ್ಥಿರಾಂಕ size_t src0_stride = mmctx->vtcm_src0_stride;
    ಸ್ಥಿರಾಂಕ size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * act_ದತ್ತಾಂಶ      = mmctx->vtcm_act;

    ದಶಮಾಂಶ * tmp = (ದಶಮಾಂಶ *) vtcm_dst_ptr;

    ಸ್ಥಿರಾಂಕ dma_addr_t src0_ಅಡ್ಡಸಾಲು = src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) act_ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಲಂಬಸಾಲು         = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2 = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + ((src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) & ~1U);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ಸ್ಥಿರಾಂಕ uint32_t prefetch_ಮುಖವಾಡ = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಆದರೆ (src2) {
            ದಶಮಾಂಶ * vtcm_bias_ptr = (ದಶಮಾಂಶ *) mmctx->vtcm_bias + src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
            ಸ್ಥಿರಾಂಕ dma_addr_t src2_addr = src2->ದತ್ತಾಂಶ + src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು * ಗಾತ್ರ(ದಶಮಾಂಶ);
            ಪೂರ್ಣಾಂಕ ತುಂಡು_ಗಾತ್ರ = (ಪೂರ್ಣಾಂಕ)src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - (ಪೂರ್ಣಾಂಕ)src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
            ಆದರೆ (ತುಂಡು_ಗಾತ್ರ > 0) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_bias_ptr, src2_addr),
                               ತುಂಡು_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), ತುಂಡು_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), ತುಂಡು_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), 1);
                dma_ಸರತಿ_ತೆಗೆ_nowait(dma_q);
            }
        }
        ಗಾಗಿ (uint32_t ir0 = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir0 < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2; ir0 += 2) {
            ಸ್ಥಿರಾಂಕ uint32_t is0 = (ir0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು);
            ಆದರೆ (is0 >= n_prefetch) {
                ನಿಲ್ಲಿಸು;
            }
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + is0 * src0_stride, src0_ಅಡ್ಡಸಾಲು + ir0 * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ),
                           src0_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 2);
        }
    }


    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    // Process src0 rows
    ಗಾಗಿ (uint32_t ir0 = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir0 < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2; ir0 += 2) {
        ಸ್ಥಿರಾಂಕ uint8_t * ss0 = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು], ss0, ss0 + src0_stride, act_ಲಂಬಸಾಲು);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ಸ್ಥಿರಾಂಕ uint32_t pr0 = (ir0 + n_prefetch);
        ಸ್ಥಿರಾಂಕ uint32_t is0 = (pr0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) & prefetch_ಮುಖವಾಡ;
        ಆದರೆ (pr0 < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + is0 * src0_stride, src0_ಅಡ್ಡಸಾಲು + pr0 * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ),
                           src0_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 2);
        }
    }

    // Process the last row (if any)
    ಆದರೆ (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು != src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2) {
        ಸ್ಥಿರಾಂಕ uint32_t ir0 = src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2;
        ಸ್ಥಿರಾಂಕ uint32_t is0 = (ir0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) & prefetch_ಮುಖವಾಡ;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + is0 * src0_stride, src0_ಅಡ್ಡಸಾಲು + ir0 * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ),
                       src0_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
        ಸ್ಥಿರಾಂಕ uint8_t * ss0 = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು], ss0, act_ಲಂಬಸಾಲು);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    ಪೂರ್ಣಾಂಕ copy_cnt = src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಆದರೆ (copy_cnt > 0) {
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು);
        ಆದರೆ (src2) {
            hvx_ಕೂಡಿಸು_f32_uuu((uint8_t *) &dst_ಲಂಬಸಾಲು[src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು],
                            (ಸ್ಥಿರಾಂಕ uint8_t *) tmp,
                            (ಸ್ಥಿರಾಂಕ uint8_t *) ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) mmctx->vtcm_bias + src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು),
                            copy_cnt);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_copy_f32_ua((uint8_t *) &dst_ಲಂಬಸಾಲು[src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mm_4d_flat(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_matmul_preamble;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ಸ್ಥಿರಾಂಕ uint32_t prefetch_ಮುಖವಾಡ = n_prefetch - 1;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ - mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ uint32_t cur_m_ಅಡ್ಡಸಾಲುಗಳು = mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು ? mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು : (ne11 * ne12 * ne13);
    ಸ್ಥಿರಾಂಕ uint32_t cur_m_ಆರಂಭ = mmctx->cur_m_ಆರಂಭ;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು  = mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ);
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2 = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + ((src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) & ~1U);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb1;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb01;
    ಸ್ಥಿರಾಂಕ size_t src0_stride = mmctx->vtcm_src0_stride;
    ಸ್ಥಿರಾಂಕ size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * ಸೀಮಿತ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * ಸೀಮಿತ act_ದತ್ತಾಂಶ      = mmctx->vtcm_act;


    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು || cur_m_ಅಡ್ಡಸಾಲುಗಳು == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_batches = ne12 * ne13;
    ಸ್ಥಿರಾಂಕ uint32_t b_ಆರಂಭ = fastdiv(cur_m_ಆರಂಭ, &kparams->div_ne1);
    uint32_t b_ಕೊನೆ = fastdiv(cur_m_ಆರಂಭ + cur_m_ಅಡ್ಡಸಾಲುಗಳು + ne11 - 1, &kparams->div_ne1);
    b_ಕೊನೆ = MIN(b_ಕೊನೆ, ಒಟ್ಟು_batches);

    uint32_t b_grp_ಆರಂಭ = b_ಆರಂಭ;
    ಆಗ (b_grp_ಆರಂಭ < b_ಕೊನೆ) {
        ಸ್ಥಿರಾಂಕ uint32_t b3 = fastdiv(b_grp_ಆರಂಭ, &kparams->div_ne12);
        ಸ್ಥಿರಾಂಕ uint32_t b2 = b_grp_ಆರಂಭ - b3 * ne12;
        ಸ್ಥಿರಾಂಕ uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ಸ್ಥಿರಾಂಕ uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_ಕೊನೆ = b_grp_ಆರಂಭ + 1;
        ಆಗ (b_grp_ಕೊನೆ < b_ಕೊನೆ) {
            ಸ್ಥಿರಾಂಕ uint32_t cur_b3 = fastdiv(b_grp_ಕೊನೆ, &kparams->div_ne12);
            ಸ್ಥಿರಾಂಕ uint32_t cur_b2 = b_grp_ಕೊನೆ - cur_b3 * ne12;
            ಸ್ಥಿರಾಂಕ uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ಸ್ಥಿರಾಂಕ uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            ಆದರೆ (cur_i02 != i02 || cur_i03 != i03) {
                ನಿಲ್ಲಿಸು;
            }
            b_grp_ಕೊನೆ++;
        }

        ಸ್ಥಿರಾಂಕ dma_addr_t src0_ಅಡ್ಡಸಾಲು = src0->ದತ್ತಾಂಶ + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        ಗಾಗಿ (uint32_t ir0 = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir0 < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2; ir0 += 2) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is0 = (ir0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು);
            ಆದರೆ (is0 >= (ಪೂರ್ಣಾಂಕ)n_prefetch) {
                ನಿಲ್ಲಿಸು;
            }
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + is0 * src0_stride, src0_ಅಡ್ಡಸಾಲು + (size_t) ir0 * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ),
                           src0_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 2);
        }

        ಗಾಗಿ (uint32_t ir0 = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir0 < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2; ir0 += 2) {
            ಸ್ಥಿರಾಂಕ uint8_t * ss0 = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            ಗಾಗಿ (uint32_t b = b_grp_ಆರಂಭ; b < b_grp_ಕೊನೆ; b++) {
                ಸ್ಥಿರಾಂಕ uint32_t b_m_ಆರಂಭ = b * ne11;
                ಸ್ಥಿರಾಂಕ uint32_t m_ಮೊದಲ = MAX(cur_m_ಆರಂಭ, b_m_ಆರಂಭ);
                ಸ್ಥಿರಾಂಕ uint32_t m_ಕೊನೆಯ  = MIN(cur_m_ಆರಂಭ + cur_m_ಅಡ್ಡಸಾಲುಗಳು, b_m_ಆರಂಭ + ne11);
                ಆದರೆ (m_ಮೊದಲ >= m_ಕೊನೆಯ) ಮುಂದುವರಿ;

                ಸ್ಥಿರಾಂಕ uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ಸ್ಥಿರಾಂಕ uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_ತಳ = (uint8_t *) dst->ದತ್ತಾಂಶ + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ಸ್ಥಿರಾಂಕ uint32_t chunk_m_ಸ್ಥಾನಾಂತರ = m_ಮೊದಲ - cur_m_ಆರಂಭ;
                ಸ್ಥಿರಾಂಕ uint32_t dst_m_ಸ್ಥಾನಾಂತರ   = m_ಮೊದಲ - b_m_ಆರಂಭ;
                ಸ್ಥಿರಾಂಕ uint32_t batch_nrows    = m_ಕೊನೆಯ - m_ಮೊದಲ;

                uint32_t ir1 = 0;
                ಗಾಗಿ (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_col0 = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (chunk_m_ಸ್ಥಾನಾಂತರ + ir1 + 0) * act_stride);
                    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_col1 = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (chunk_m_ಸ್ಥಾನಾಂತರ + ir1 + 1) * act_stride);
                    ದಶಮಾಂಶ * ಸೀಮಿತ dst_row0 = (ದಶಮಾಂಶ *) (dst_batch_ತಳ + (dst_m_ಸ್ಥಾನಾಂತರ + ir1 + 0) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
                    ದಶಮಾಂಶ * ಸೀಮಿತ dst_row1 = (ದಶಮಾಂಶ *) (dst_batch_ತಳ + (dst_m_ಸ್ಥಾನಾಂತರ + ir1 + 1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                ಗಾಗಿ (; ir1 < batch_nrows; ++ir1) {
                    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (chunk_m_ಸ್ಥಾನಾಂತರ + ir1) * act_stride);
                    ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು          = (ದಶಮಾಂಶ *) (dst_batch_ತಳ + (dst_m_ಸ್ಥಾನಾಂತರ + ir1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
                    mmctx->vec_dot_2x1(ne00, &dst_ಅಡ್ಡಸಾಲು[ir0], ss0, ss0 + src0_stride, act_ಲಂಬಸಾಲು);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pr0 = (ir0 + n_prefetch);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is0 = (pr0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) & prefetch_ಮುಖವಾಡ;
            ಆದರೆ (pr0 < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + is0 * src0_stride, src0_ಅಡ್ಡಸಾಲು + (size_t) pr0 * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ),
                               src0_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 2);
            }
        }

        ಆದರೆ (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು != src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2) {
            uint32_t ir0 = src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು_x2;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is0 = (ir0 - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) & prefetch_ಮುಖವಾಡ;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + is0 * src0_stride, src0_ಅಡ್ಡಸಾಲು + (size_t) ir0 * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ),
                           src0_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
            ಸ್ಥಿರಾಂಕ uint8_t * ss0 = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            ಗಾಗಿ (uint32_t b = b_grp_ಆರಂಭ; b < b_grp_ಕೊನೆ; b++) {
                ಸ್ಥಿರಾಂಕ uint32_t b_m_ಆರಂಭ = b * ne11;
                ಸ್ಥಿರಾಂಕ uint32_t m_ಮೊದಲ = MAX(cur_m_ಆರಂಭ, b_m_ಆರಂಭ);
                ಸ್ಥಿರಾಂಕ uint32_t m_ಕೊನೆಯ  = MIN(cur_m_ಆರಂಭ + cur_m_ಅಡ್ಡಸಾಲುಗಳು, b_m_ಆರಂಭ + ne11);
                ಆದರೆ (m_ಮೊದಲ >= m_ಕೊನೆಯ) ಮುಂದುವರಿ;

                ಸ್ಥಿರಾಂಕ uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ಸ್ಥಿರಾಂಕ uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_ತಳ = (uint8_t *) dst->ದತ್ತಾಂಶ + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ಸ್ಥಿರಾಂಕ uint32_t chunk_m_ಸ್ಥಾನಾಂತರ = m_ಮೊದಲ - cur_m_ಆರಂಭ;
                ಸ್ಥಿರಾಂಕ uint32_t dst_m_ಸ್ಥಾನಾಂತರ   = m_ಮೊದಲ - b_m_ಆರಂಭ;
                ಸ್ಥಿರಾಂಕ uint32_t batch_nrows    = m_ಕೊನೆಯ - m_ಮೊದಲ;

                ಗಾಗಿ (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (chunk_m_ಸ್ಥಾನಾಂತರ + ir1) * act_stride);
                    ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು          = (ದಶಮಾಂಶ *) (dst_batch_ತಳ + (dst_m_ಸ್ಥಾನಾಂತರ + ir1) * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
                    mmctx->vec_dot_1x1(ne00, &dst_ಅಡ್ಡಸಾಲು[ir0], ss0, act_ಲಂಬಸಾಲು);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_ಆರಂಭ = b_grp_ಕೊನೆ;
    }

    ಆದರೆ (src2) {
        hvx_tensor_ಕೂಡಿಸು_f32_grid(dst, src2, cur_m_ಆರಂಭ, cur_m_ಆರಂಭ + cur_m_ಅಡ್ಡಸಾಲುಗಳು, src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(ಅಡ್ಡಸಾಲು_id, i1) ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[(ಅಡ್ಡಸಾಲು_id) * mmctx->mapping_stride + (i1)]

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mm_id_tiled(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_matmul_preamble;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ ids = octx->src[2];

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows      = mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ - mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;  // src0 rows per expert
    ಸ್ಥಿರಾಂಕ uint32_t act_nrows       = ne11;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು  = mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ);


    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ಸ್ಥಿರಾಂಕ uint32_t n_ids = ids->ne[0];  // n_expert_used
    ಸ್ಥಿರಾಂಕ uint32_t n_as  = ne02;        // n_expert

    ಸ್ಥಿರಾಂಕ uint32_t *                ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts = mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts;
    ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು       = mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು;

    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb1;
    ಸ್ಥಿರಾಂಕ size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = htp_mm_q8_0_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ne10);

    ಸ್ಥಿರಾಂಕ size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * ಸೀಮಿತ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * ಸೀಮಿತ act_ದತ್ತಾಂಶ = mmctx->vtcm_act;

    ಗಾಗಿ (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ಸ್ಥಿರಾಂಕ int32_t cne1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a];
        ಆದರೆ (cne1 == 0) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ dma_addr_t src0_ಅಡ್ಡಸಾಲು = src0->ದತ್ತಾಂಶ + cur_a * nb02;

        ಸ್ಥಿರಾಂಕ uint32_t tile_ಗಾತ್ರ = htp_tensor_get_tile_ಗಾತ್ರ(src0->type);
        ಸ್ಥಿರಾಂಕ uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(src0->type);
        ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_stride = n_k_tiles_w * tile_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned = n_k_tiles_w * aligned_tile_ಗಾತ್ರ;

        ಸ್ಥಿರಾಂಕ uint32_t ct_ಆರಂಭ = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು / 32;
        ಸ್ಥಿರಾಂಕ uint32_t ct_ಕೊನೆ   = (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು + 31) / 32;

        uint32_t ತಳ್ಳು_ct = ct_ಆರಂಭ;
        ಗಾಗಿ (uint32_t d = 0; d < n_prefetch && ತಳ್ಳು_ct < ct_ಕೊನೆ; d++, ತಳ್ಳು_ct++) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + d * tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned, src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                           aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
        }

        ಗಾಗಿ (uint32_t ct = ct_ಆರಂಭ; ct < ct_ಕೊನೆ; ct++) {
            ಸ್ಥಿರಾಂಕ uint8_t * w_tile = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

            ಪೂರ್ಣಾಂಕ valid_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ)ne0 - (ಪೂರ್ಣಾಂಕ)(ct * 32);
            valid_ಅಡ್ಡಸಾಲುಗಳು = MIN(32, MAX(0, valid_ಅಡ್ಡಸಾಲುಗಳು));

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            ಗಾಗಿ (uint32_t cid = 0; cid < cne1; ++cid) {
                ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping ಅಡ್ಡಸಾಲು_mapping = MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, cid);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ               rm1         = ಅಡ್ಡಸಾಲು_mapping.i1;  // expert idx
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ               rm2         = ಅಡ್ಡಸಾಲು_mapping.i2;  // token idx

                ಸ್ಥಿರಾಂಕ uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (ir1 + rm2 * ne11 + 0) * act_stride);
                ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_ಅಡ್ಡಸಾಲು[ct * 32], w_tile, act_ಲಂಬಸಾಲು, valid_ಅಡ್ಡಸಾಲುಗಳು, ಖಾಲಿ);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ಆದರೆ (ತಳ್ಳು_ct < ct_ಕೊನೆ) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(w_tile, src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                               aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
                ತಳ್ಳು_ct++;
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mv_id_tiled(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_matmul_preamble;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ ids = octx->src[2];

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows      = mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ - mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;  // src0 rows per expert
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು  = mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ);


    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ಪ್ರತಿಜ್ಞಾ(ne13 % ne03 == 0);

    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb1;
    ಸ್ಥಿರಾಂಕ size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_q8_0_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ne10);

    ಸ್ಥಿರಾಂಕ uint32_t n_aids = src2->ne[0];  // num activated experts
    ಸ್ಥಿರಾಂಕ uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * ಸೀಮಿತ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * ಸೀಮಿತ act_ದತ್ತಾಂಶ = mmctx->vtcm_act;

    ಗಾಗಿ (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ಸ್ಥಿರಾಂಕ int32_t eid = *(ಸ್ಥಿರಾಂಕ int32_t *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src2->ದತ್ತಾಂಶ + ie1 * src2->nb[0]);
        ಆದರೆ (eid < 0) {
            ಮುಂದುವರಿ;
        }
        ಪ್ರತಿಜ್ಞಾ(eid < (int32_t) n_ids);

        ಸ್ಥಿರಾಂಕ dma_addr_t src0_ಅಡ್ಡಸಾಲು = src0->ದತ್ತಾಂಶ + eid * nb02;
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) act_ದತ್ತಾಂಶ;
        ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು          = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ie1 * nb1);

        ಸ್ಥಿರಾಂಕ uint32_t tile_ಗಾತ್ರ = htp_tensor_get_tile_ಗಾತ್ರ(src0->type);
        ಸ್ಥಿರಾಂಕ uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(src0->type);
        ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_stride = n_k_tiles_w * tile_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned = n_k_tiles_w * aligned_tile_ಗಾತ್ರ;

        ಸ್ಥಿರಾಂಕ uint32_t ct_ಆರಂಭ = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು / 32;
        ಸ್ಥಿರಾಂಕ uint32_t ct_ಕೊನೆ   = (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು + 31) / 32;

        uint32_t ತಳ್ಳು_ct = ct_ಆರಂಭ;
        ಗಾಗಿ (uint32_t d = 0; d < n_prefetch && ತಳ್ಳು_ct < ct_ಕೊನೆ; d++, ತಳ್ಳು_ct++) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + d * tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned, src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                           aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
        }

        ಗಾಗಿ (uint32_t ct = ct_ಆರಂಭ; ct < ct_ಕೊನೆ; ct++) {
            ಸ್ಥಿರಾಂಕ uint8_t * w_tile = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

            ಪೂರ್ಣಾಂಕ valid_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ)ne0 - (ಪೂರ್ಣಾಂಕ)(ct * 32);
            valid_ಅಡ್ಡಸಾಲುಗಳು = MIN(32, MAX(0, valid_ಅಡ್ಡಸಾಲುಗಳು));

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_ಅಡ್ಡಸಾಲು[ct * 32], w_tile, act_ಲಂಬಸಾಲು, valid_ಅಡ್ಡಸಾಲುಗಳು, ಖಾಲಿ);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ಆದರೆ (ತಳ್ಳು_ct < ct_ಕೊನೆ) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(w_tile, src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                               aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
                ತಳ್ಳು_ct++;
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mv_id_nx_tiled(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_mm_context * mmctx = (ರಚನೆ htp_mm_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = mmctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_weights      = kparams->n_weights;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ act  = octx->src[n_weights];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ ids  = octx->src[n_weights + 1];


    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ಸ್ಥಿರಾಂಕ uint32_t n_aids = ids->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t n_ids  = src0->ne[2];

    uint8_t * ಸೀಮಿತ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * ಸೀಮಿತ act_ದತ್ತಾಂಶ = mmctx->vtcm_act;

    ಗಾಗಿ (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ಸ್ಥಿರಾಂಕ int32_t eid = *(ಸ್ಥಿರಾಂಕ int32_t *) ((ಸ್ಥಿರಾಂಕ uint8_t *) ids->ದತ್ತಾಂಶ + ie1 * ids->nb[0]);
        ಆದರೆ (eid < 0) ಮುಂದುವರಿ;
        ಪ್ರತಿಜ್ಞಾ(eid < (int32_t) n_ids);

        ಗಾಗಿ (uint32_t p = 0; p < n_weights; ++p) {
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src_w = octx->src[p];
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst   = octx->dsts[p];
            ಆದರೆ (!src_w || !dst) ಮುಂದುವರಿ;
            dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

            ಸ್ಥಿರಾಂಕ uint32_t ne01 = src_w->ne[1];
            uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = 0;
            uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು   = ne01;
            ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
                ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ne01, can_ವಿಭಜಿಸು ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
                ಆರಂಭ_ಅಡ್ಡಸಾಲು = range.ಆರಂಭ;
                ಕೊನೆ_ಅಡ್ಡಸಾಲು   = range.ಆರಂಭ + range.ಎಣಿಕೆ_2;
            }

            ಸ್ಥಿರಾಂಕ uint32_t nrows = ಕೊನೆ_ಅಡ್ಡಸಾಲು - ಆರಂಭ_ಅಡ್ಡಸಾಲು;
            uint32_t src0_nrows_ಪ್ರತಿ_thread = fastdiv(nrows + nth - 1, &octx->n_ಎಳೆಗಳು_div);
            src0_nrows_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ(src0_nrows_ಪ್ರತಿ_thread, 32);

            ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread * ith;
            ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, ಕೊನೆ_ಅಡ್ಡಸಾಲು);
            ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) ಮುಂದುವರಿ;

            ಸ್ಥಿರಾಂಕ dma_addr_t src0_ಅಡ್ಡಸಾಲು = src_w->ದತ್ತಾಂಶ + eid * src_w->nb[2];
            ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) act_ದತ್ತಾಂಶ;
            ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + ie1 * dst->nb[1]);

            ಸ್ಥಿರಾಂಕ uint32_t tile_ಗಾತ್ರ = htp_tensor_get_tile_ಗಾತ್ರ(src_w->type);
            ಸ್ಥಿರಾಂಕ uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(src_w->type);
            ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_stride = n_k_tiles_w * tile_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned = n_k_tiles_w * aligned_tile_ಗಾತ್ರ;

            ಸ್ಥಿರಾಂಕ uint32_t ct_ಆರಂಭ = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು / 32;
            ಸ್ಥಿರಾಂಕ uint32_t ct_ಕೊನೆ   = (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು + 31) / 32;

            uint32_t ತಳ್ಳು_ct = ct_ಆರಂಭ;
            ಗಾಗಿ (uint32_t d = 0; d < n_prefetch && ತಳ್ಳು_ct < ct_ಕೊನೆ; d++, ತಳ್ಳು_ct++) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + d * tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned, src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                               aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
            }

            ಗಾಗಿ (uint32_t ct = ct_ಆರಂಭ; ct < ct_ಕೊನೆ; ct++) {
                ಸ್ಥಿರಾಂಕ uint8_t * w_tile = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

                ಪೂರ್ಣಾಂಕ valid_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ)dst->ne[0] - (ಪೂರ್ಣಾಂಕ)(ct * 32);
                valid_ಅಡ್ಡಸಾಲುಗಳು = MIN(32, MAX(0, valid_ಅಡ್ಡಸಾಲುಗಳು));

                htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_ಅಡ್ಡಸಾಲು[ct * 32], w_tile, act_ಲಂಬಸಾಲು, valid_ಅಡ್ಡಸಾಲುಗಳು, ಖಾಲಿ);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                ಆದರೆ (ತಳ್ಳು_ct < ct_ಕೊನೆ) {
                    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(w_tile, src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                                    aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
                    ತಳ್ಳು_ct++;
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_mm_id_nx_tiled(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_mm_context * mmctx = (ರಚನೆ htp_mm_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = mmctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_weights      = kparams->n_weights;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ act  = octx->src[n_weights];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ ids  = octx->src[n_weights + 1];


    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ uint32_t n_prefetch = kparams->n_prefetch;
    ಪ್ರತಿಜ್ಞಾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ಸ್ಥಿರಾಂಕ uint32_t n_as = src0->ne[2];

    ಸ್ಥಿರಾಂಕ uint32_t * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts = mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts;
    ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು = mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು;

    ಸ್ಥಿರಾಂಕ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * ಸೀಮಿತ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * ith;
    uint8_t * ಸೀಮಿತ act_ದತ್ತಾಂಶ = mmctx->vtcm_act;

    ಗಾಗಿ (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ಸ್ಥಿರಾಂಕ int32_t cne1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a];
        ಆದರೆ (cne1 == 0) ಮುಂದುವರಿ;

        ಗಾಗಿ (uint32_t p = 0; p < n_weights; ++p) {
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src_w = octx->src[p];
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst   = octx->dsts[p];
            ಆದರೆ (!src_w || !dst) ಮುಂದುವರಿ;
            dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

            ಸ್ಥಿರಾಂಕ uint32_t ne01 = src_w->ne[1];
            uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = 0;
            uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು   = ne01;
            ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
                ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ne01, can_ವಿಭಜಿಸು ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
                ಆರಂಭ_ಅಡ್ಡಸಾಲು = range.ಆರಂಭ;
                ಕೊನೆ_ಅಡ್ಡಸಾಲು   = range.ಆರಂಭ + range.ಎಣಿಕೆ_2;
            }

            ಸ್ಥಿರಾಂಕ uint32_t nrows = ಕೊನೆ_ಅಡ್ಡಸಾಲು - ಆರಂಭ_ಅಡ್ಡಸಾಲು;
            uint32_t src0_nrows_ಪ್ರತಿ_thread = fastdiv(nrows + nth - 1, &octx->n_ಎಳೆಗಳು_div);
            src0_nrows_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ(src0_nrows_ಪ್ರತಿ_thread, 32);

            ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread * ith;
            ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, ಕೊನೆ_ಅಡ್ಡಸಾಲು);
            ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) ಮುಂದುವರಿ;

            ಸ್ಥಿರಾಂಕ dma_addr_t src0_ಅಡ್ಡಸಾಲು = src_w->ದತ್ತಾಂಶ + cur_a * src_w->nb[2];

            ಸ್ಥಿರಾಂಕ uint32_t tile_ಗಾತ್ರ = htp_tensor_get_tile_ಗಾತ್ರ(src_w->type);
            ಸ್ಥಿರಾಂಕ uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(src_w->type);
            ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_stride = n_k_tiles_w * tile_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned = n_k_tiles_w * aligned_tile_ಗಾತ್ರ;

            ಸ್ಥಿರಾಂಕ uint32_t ct_ಆರಂಭ = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು / 32;
            ಸ್ಥಿರಾಂಕ uint32_t ct_ಕೊನೆ   = (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು + 31) / 32;

            uint32_t ತಳ್ಳು_ct = ct_ಆರಂಭ;
            ಗಾಗಿ (uint32_t d = 0; d < n_prefetch && ತಳ್ಳು_ct < ct_ಕೊನೆ; d++, ತಳ್ಳು_ct++) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src0_ptr + d * tile_ಅಡ್ಡಸಾಲು_transfer_ಗಾತ್ರ_aligned, src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                               aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
            }

            ಗಾಗಿ (uint32_t ct = ct_ಆರಂಭ; ct < ct_ಕೊನೆ; ct++) {
                ಸ್ಥಿರಾಂಕ uint8_t * w_tile = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

                ಪೂರ್ಣಾಂಕ valid_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ)dst->ne[0] - (ಪೂರ್ಣಾಂಕ)(ct * 32);
                valid_ಅಡ್ಡಸಾಲುಗಳು = MIN(32, MAX(0, valid_ಅಡ್ಡಸಾಲುಗಳು));

                htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                ಗಾಗಿ (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping ಅಡ್ಡಸಾಲು_mapping = MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, cid);
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rm1 = ಅಡ್ಡಸಾಲು_mapping.i1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rm2 = ಅಡ್ಡಸಾಲು_mapping.i2;

                    ಸ್ಥಿರಾಂಕ uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ act_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) (act_ದತ್ತಾಂಶ + (ir1 + rm2 * act->ne[1]) * act_stride);
                    ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) (dst->ದತ್ತಾಂಶ + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_ಅಡ್ಡಸಾಲು[ct * 32], w_tile, act_ಲಂಬಸಾಲು, valid_ಅಡ್ಡಸಾಲುಗಳು, ಖಾಲಿ);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                ಆದರೆ (ತಳ್ಳು_ct < ct_ಕೊನೆ) {
                    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(w_tile, src0_ಅಡ್ಡಸಾಲು + ತಳ್ಳು_ct * tile_ಅಡ್ಡಸಾಲು_stride),
                                    aligned_tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles_w);
                    ತಳ್ಳು_ct++;
                }
            }
        }
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hvx_mm_init_tiled_vec_dot(ರಚನೆ htp_mm_context * mmctx, ಎಣಿಕೆ htp_ದತ್ತಾಂಶ_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_Q4_1:
        ಸ್ಥಿತಿ HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            ಹಿಂತಿರುಗಿ 0;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ -1;
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hvx_mm_matmul(ರಚನೆ htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, 0);

    ರಚನೆ htp_mm_context mmctx_struct = {0};
    ರಚನೆ htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = ne01;
    ಸ್ಥಿರಾಂಕ uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ   = src0_nrows;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(src0_nrows, can_ವಿಭಜಿಸು ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ   = range.ಆರಂಭ + range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ >= src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t nrows = src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ - src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ = src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ   = src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ;

    ಸ್ಥಿರಾಂಕ bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_ಪ್ರತಿ_thread  = fastdiv(nrows + octx->n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);
    ಆದರೆ (is_tiled) {
        mmctx->src0_nrows_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ(mmctx->src0_nrows_ಪ್ರತಿ_thread, 32);
    } ಇಲ್ಲದಿದ್ದರೆ {
        mmctx->src0_nrows_ಪ್ರತಿ_thread += (mmctx->src0_nrows_ಪ್ರತಿ_thread & 1); // round up to even
    }

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb01;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb1;
    size_t       act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb11;

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);

    ಕೆಲಸಗಾರ_callback_t quant_task_func;
    ಕೆಲಸಗಾರ_callback_t matmul_ಕೆಲಸ_func;
    uint32_t n_quant_tasks = 1;
    ಸ್ಥಿರಾಂಕ bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    ಆದರೆ (is_batched) {
        matmul_ಕೆಲಸ_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (act_nrows > 1) {
        matmul_ಕೆಲಸ_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } ಇಲ್ಲದಿದ್ದರೆ {
        matmul_ಕೆಲಸ_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    bool need_quant = true;

    ಆಯ್ಕೆ (kparams->kernel_type) {
        ಸ್ಥಿತಿ HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ           = kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = ಖಾಲಿ;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ           = kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = ಖಾಲಿ;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ           = kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ:
        ಸ್ಥಿತಿ HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು:
        ಪೂರ್ವನಿಯೋಜಿತ: {
            ಆದರೆ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
            }

            ಸ್ಥಿರಾಂಕ uint32_t qk = QK_Q8_0_TILED;
            ಸ್ಥಿರಾಂಕ uint32_t nb = (ne10 + qk - 1) / qk;
            ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_nb = act_nrows * nb;

            ಆದರೆ (act_nrows < octx->n_ಎಳೆಗಳು && !is_batched) {
                n_quant_tasks = MIN(ಒಟ್ಟು_nb, octx->n_ಎಳೆಗಳು);
                quant_task_func = htp_mm_act_quant_ಖಂಡ_func(src0->type, act->type);
                ಗಾಗಿ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_ಮೊದಲ = (ಒಟ್ಟು_nb * ith) / n_quant_tasks;
                    uint32_t ib_ಕೊನೆಯ  = (ಒಟ್ಟು_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_ಮೊದಲ[ith] = ib_ಮೊದಲ;
                    mmctx->quant_ib_ಕೊನೆಯ[ith]  = ib_ಕೊನೆಯ;
                    mmctx->quant_r[ith]        = ib_ಮೊದಲ / nb;
                    mmctx->quant_c[ith]        = ib_ಮೊದಲ % nb;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                n_quant_tasks = MIN(act_nrows, octx->n_ಎಳೆಗಳು);
                quant_task_func = htp_mm_act_quant_ಅಡ್ಡಸಾಲು_func(src0->type, act->type);
            }
            act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = kparams->act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
            ನಿಲ್ಲಿಸು;
        }
    }

    ಸ್ಥಿರಾಂಕ uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t m_layout_ಅಡ್ಡಸಾಲುಗಳು = m_chunk;

    ರಚನೆ htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_ಅಡ್ಡಸಾಲುಗಳು, octx->n_ಎಳೆಗಳು,
                                 dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    ಆದರೆ (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ) {
        mmctx->vtcm_act_ಗಾತ್ರ_ಪ್ರತಿ_thread = L.act_ಬೈಟುಗಳು;
    } ಇಲ್ಲದಿದ್ದರೆ {
        mmctx->vtcm_act_ಗಾತ್ರ_ಪ್ರತಿ_thread = fastdiv(L.act_ಬೈಟುಗಳು, &octx->n_ಎಳೆಗಳು_div);
    }

    mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = fastdiv(L.src0_ಬೈಟುಗಳು, &octx->n_ಎಳೆಗಳು_div);
    mmctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = fastdiv(L.dst_ಬೈಟುಗಳು, &octx->n_ಎಳೆಗಳು_div);

    ಸ್ಥಿರಾಂಕ size_t vtcm_ಗಾತ್ರ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;

    FARF(ಮೇಲಿನ_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_ಬೈಟುಗಳು, L.act_ಬೈಟುಗಳು, L.dst_ಬೈಟುಗಳು, vtcm_ಗಾತ್ರ);

    FARF(ಮೇಲಿನ_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->ದತ್ತಾಂಶ, act->ದತ್ತಾಂಶ, dst->ದತ್ತಾಂಶ);

    ಆದರೆ (octx->ctx->vtcm_ಗಾತ್ರ < vtcm_ಗಾತ್ರ) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_ಗಾತ್ರ, vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded;
    mmctx->vtcm_act_stride  = act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಆದರೆ (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು) {
        mmctx->vtcm_act_raw_stride = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * ಗಾತ್ರ(ದಶಮಾಂಶ), QK_Q8_0_TILED * ಗಾತ್ರ(ದಶಮಾಂಶ));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
    } ಇಲ್ಲದಿದ್ದರೆ {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ಆದರೆ (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        ಗಾಗಿ (uint32_t m_ಆರಂಭ = 0; m_ಆರಂಭ < act_nrows; m_ಆರಂಭ += m_chunk) {
            ಸ್ಥಿರಾಂಕ uint32_t cur_m_ಅಡ್ಡಸಾಲುಗಳು = MIN(act_nrows - m_ಆರಂಭ, m_chunk);
            mmctx->cur_m_ಆರಂಭ = m_ಆರಂಭ;
            mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು  = cur_m_ಅಡ್ಡಸಾಲುಗಳು;

            ಆದರೆ (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_ಆರಂಭ, cur_m_ಅಡ್ಡಸಾಲುಗಳು);

                ಸ್ಥಿರಾಂಕ uint32_t qk = QK_Q8_0_TILED;
                ಸ್ಥಿರಾಂಕ uint32_t nb = (ne10 + qk - 1) / qk;
                ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_nb = cur_m_ಅಡ್ಡಸಾಲುಗಳು * nb;
                uint32_t quant_tasks;
                work_ಸರತಿ_func_t q_func;
                ಆದರೆ (cur_m_ಅಡ್ಡಸಾಲುಗಳು < octx->n_ಎಳೆಗಳು && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು)) {
                    quant_tasks = MIN(ಒಟ್ಟು_nb, octx->n_ಎಳೆಗಳು);
                    q_func = htp_mm_act_quant_ಖಂಡ_func(src0->type, act->type);
                    ಗಾಗಿ (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_ಮೊದಲ = (ಒಟ್ಟು_nb * ith) / quant_tasks;
                        uint32_t ib_ಕೊನೆಯ  = (ಒಟ್ಟು_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_ಮೊದಲ[ith] = ib_ಮೊದಲ;
                        mmctx->quant_ib_ಕೊನೆಯ[ith]  = ib_ಕೊನೆಯ;
                        mmctx->quant_r[ith]        = ib_ಮೊದಲ / nb;
                        mmctx->quant_c[ith]        = ib_ಮೊದಲ % nb;
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    quant_tasks = MIN(cur_m_ಅಡ್ಡಸಾಲುಗಳು, octx->n_ಎಳೆಗಳು);
                    q_func = quant_task_func;
                    mmctx->n_quant_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = (cur_m_ಅಡ್ಡಸಾಲುಗಳು + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, q_func, mmctx, quant_tasks);
            } ಇಲ್ಲದಿದ್ದರೆ {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_ಆರಂಭ, cur_m_ಅಡ್ಡಸಾಲುಗಳು);
            }

            work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, matmul_ಕೆಲಸ_func, mmctx, octx->n_ಎಳೆಗಳು);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        mmctx->cur_m_ಆರಂಭ = 0;
        mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು  = act_nrows;

        ಆದರೆ (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, quant_task_func, mmctx, n_quant_tasks);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, matmul_ಕೆಲಸ_func, mmctx, octx->n_ಎಳೆಗಳು);
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ಕೆಲಸಗಾರ_loop(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ task_id = i; task_id < (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) state->n_tasks; task_id += n) {
        ಪೂರ್ಣಾಂಕ ಆರಂಭ = task_id * state->n_tiles_ಪ್ರತಿ_task;
        ಪೂರ್ಣಾಂಕ ಕೊನೆ   = hex_smin(ಆರಂಭ + state->n_tiles_ಪ್ರತಿ_task, state->n_tot_tiles);
        state->task_fn(state, ಆರಂಭ, ಕೊನೆ);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_output_chunk_ಕೆಲಸಗಾರ_fn(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ *ದತ್ತಾಂಶ) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) ದತ್ತಾಂಶ;

    ರಚನೆ htp_thread_trace * tr = &st->traces[i];

    ಪೂರ್ಣಾಂಕ ಆರಂಭ_chunk_idx = i * st->n_chunks_ಪ್ರತಿ_task;
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_O_PROC, ಆರಂಭ_chunk_idx);

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ task_id = i; task_id < (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)st->n_tasks; task_id += n) {
        ಪೂರ್ಣಾಂಕ    chunk_idx  = task_id * st->n_chunks_ಪ್ರತಿ_task;
        size_t chunk_ಗಾತ್ರ = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_ಪ್ರತಿ_task);

        ದಶಮಾಂಶ        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : ಖಾಲಿ;
        transfer_output_chunk_fp16_ಗಮ್ಯ_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_ಗಾತ್ರ, st->n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, st->dst_stride, st->src2_stride, st->dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, ಆರಂಭ_chunk_idx);
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ರಚನೆ htp_context            * ctx;
    ರಚನೆ htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    ದಶಮಾಂಶ                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_ಪ್ರತಿ_task;
    uint32_t                        k_ಖಂಡ;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    uint32_t                        dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t                        dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು;
    uint32_t                        act_elem_ಗಾತ್ರ;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ರಚನೆ htp_context            * ctx;
    ರಚನೆ htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    ದಶಮಾಂಶ                         * vtcm_f32_act;
    uint32_t                        n_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t                        k_ಖಂಡ;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_ಲಂಬಸಾಲು_chunks;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು           n_ಎಳೆಗಳು_div;
    size_t                          vtcm_f32_act_ಬೈಟುಗಳು;
    uint32_t                        dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t                        dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು;
} activation_transfer_ಲಂಬಸಾಲು_chunk_state_t;

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_chunk_fp32_ಗಮ್ಯ_fp16_dma_pipelined_ಲಂಬಸಾಲು_chunk(
        dma_ಸರತಿ *dma_q,
        __fp16 *ಸೀಮಿತ vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,
        uint32_t k_ಖಂಡ,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_ಮೊದಲ,
        uint32_t c_len,
        ದಶಮಾಂಶ *thread_f32_act,
        ರಚನೆ htp_thread_trace *tr,
        uint32_t dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು,
        uint32_t dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು) {

    ಸ್ಥಿರಾಂಕ uint32_t R = dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t n_ಅಡ್ಡಸಾಲುಗಳು_padded = hex_align_ಮೇಲೆ(n_ಅಡ್ಡಸಾಲುಗಳು, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);

    ಸ್ಥಿರಾಂಕ uint32_t n_ಹೆಜ್ಜೆಗಳು = n_ಅಡ್ಡಸಾಲುಗಳು_padded >> dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು;

    // Push step 0
    ಆದರೆ (n_ಹೆಜ್ಜೆಗಳು > 0 && n_ಅಡ್ಡಸಾಲುಗಳು > 0) {
        uint32_t nrows_ಗಮ್ಯ_fetch = hex_smin(n_ಅಡ್ಡಸಾಲುಗಳು, R);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(thread_f32_act, act_dma_addr + (size_t) c_ಮೊದಲ * ಗಾತ್ರ(ದಶಮಾಂಶ)),
                       c_len * ಗಾತ್ರ(ದಶಮಾಂಶ), k_stride * ಗಾತ್ರ(ದಶಮಾಂಶ), k_chunk_valid * ಗಾತ್ರ(ದಶಮಾಂಶ), nrows_ಗಮ್ಯ_fetch);
    }
    // Push step 1
    ಆದರೆ (n_ಹೆಜ್ಜೆಗಳು > 1) {
        uint32_t ಮುಂದಿನ_r = R * 1;
        ಆದರೆ (ಮುಂದಿನ_r < n_ಅಡ್ಡಸಾಲುಗಳು) {
            uint32_t nrows_ಗಮ್ಯ_fetch = hex_smin(n_ಅಡ್ಡಸಾಲುಗಳು - ಮುಂದಿನ_r, R);
            ದಶಮಾಂಶ *ಮುಂದಿನ_buf = thread_f32_act + 1 * R * c_len;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ಮುಂದಿನ_buf, act_dma_addr + ((size_t) ಮುಂದಿನ_r * k_stride + c_ಮೊದಲ) * ಗಾತ್ರ(ದಶಮಾಂಶ)),
                           c_len * ಗಾತ್ರ(ದಶಮಾಂಶ), k_stride * ಗಾತ್ರ(ದಶಮಾಂಶ), k_chunk_valid * ಗಾತ್ರ(ದಶಮಾಂಶ), nrows_ಗಮ್ಯ_fetch);
        }
    }
    ಗಾಗಿ (uint32_t s = 0; s < n_ಹೆಜ್ಜೆಗಳು; ++s) {
        uint32_t r = s << dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು;
        ದಶಮಾಂಶ *curr_buf = thread_f32_act;

        ಆದರೆ (r < n_ಅಡ್ಡಸಾಲುಗಳು) {
            curr_buf = (ದಶಮಾಂಶ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
        }

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        ಗಾಗಿ (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t ಅಡ್ಡಸಾಲು_idx = r + (p << 1);
            ದಶಮಾಂಶ *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((ಅಡ್ಡಸಾಲು_idx + 0) < n_ಅಡ್ಡಸಾಲುಗಳು);
            bool r1_valid = ((ಅಡ್ಡಸಾಲು_idx + 1) < n_ಅಡ್ಡಸಾಲುಗಳು);

            transfer_activation_ಅಡ್ಡಸಾಲು_pair_fp32_ಗಮ್ಯ_fp16_ಲಂಬಸಾಲು_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, ಅಡ್ಡಸಾಲು_idx, k_ಖಂಡ, c_ಮೊದಲ, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t ಮುಂದಿನ_s = s + 2;
        uint32_t ಮುಂದಿನ_r = ಮುಂದಿನ_s << dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು;
        ಆದರೆ (ಮುಂದಿನ_r < n_ಅಡ್ಡಸಾಲುಗಳು) {
            uint32_t nrows_ಗಮ್ಯ_fetch = hex_smin(n_ಅಡ್ಡಸಾಲುಗಳು - ಮುಂದಿನ_r, R);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(curr_buf, act_dma_addr + ((size_t) ಮುಂದಿನ_r * k_stride + c_ಮೊದಲ) * ಗಾತ್ರ(ದಶಮಾಂಶ)),
                           c_len * ಗಾತ್ರ(ದಶಮಾಂಶ), k_stride * ಗಾತ್ರ(ದಶಮಾಂಶ), k_chunk_valid * ಗಾತ್ರ(ದಶಮಾಂಶ), nrows_ಗಮ್ಯ_fetch);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_chunk_ಲಂಬಸಾಲು_chunk_ಕೆಲಸಗಾರ_fn(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ *ದತ್ತಾಂಶ) {
    activation_transfer_ಲಂಬಸಾಲು_chunk_state_t *st = (activation_transfer_ಲಂಬಸಾಲು_chunk_state_t *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &st->traces[i];

    uint32_t n_ಖಂಡಗಳು = st->k_ಖಂಡ / 32;
    uint32_t b_ಮೊದಲ = fastdiv(n_ಖಂಡಗಳು * i, &st->n_ಎಳೆಗಳು_div);
    uint32_t b_ಕೊನೆಯ  = fastdiv(n_ಖಂಡಗಳು * (i + 1), &st->n_ಎಳೆಗಳು_div);
    uint32_t c_ಮೊದಲ = b_ಮೊದಲ * 32;
    uint32_t c_ಕೊನೆಯ = b_ಕೊನೆಯ * 32;
    uint32_t c_len = c_ಕೊನೆಯ - c_ಮೊದಲ;

    ಆದರೆ (c_len == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    uint32_t k_chunk_valid = 0;
    ಆದರೆ (st->k_valid > c_ಮೊದಲ) {
        k_chunk_valid = hex_smin(st->k_valid, c_ಕೊನೆಯ) - c_ಮೊದಲ;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_ಬೈಟುಗಳು = hex_align_ಕೆಳಕ್ಕೆ(fastdiv(st->vtcm_f32_act_ಬೈಟುಗಳು, &st->n_ಎಳೆಗಳು_div), 128);
    ದಶಮಾಂಶ *thread_f32_act = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)st->vtcm_f32_act + i * thread_scratch_ಬೈಟುಗಳು);

    transfer_activation_chunk_fp32_ಗಮ್ಯ_fp16_dma_pipelined_ಲಂಬಸಾಲು_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_ಅಡ್ಡಸಾಲುಗಳು, st->k_ಖಂಡ, st->k_stride, k_chunk_valid,
        c_ಮೊದಲ, c_len, thread_f32_act, tr, st->dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು, st->dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು
    );
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_chunk_ಗಮ್ಯ_fp16_dma_pipelined(
        dma_ಸರತಿ *dma_q,
        __fp16 *ಸೀಮಿತ vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,
        uint32_t k_ಖಂಡ,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_ಗಾತ್ರ,
        ರಚನೆ htp_thread_trace *tr,
        uint32_t dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು,
        uint32_t dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು) {

    ಸ್ಥಿರಾಂಕ uint32_t R = dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t n_ಅಡ್ಡಸಾಲುಗಳು_padded = hex_align_ಮೇಲೆ(n_ಅಡ್ಡಸಾಲುಗಳು, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);

    ಸ್ಥಿರಾಂಕ uint32_t n_ಹೆಜ್ಜೆಗಳು = n_ಅಡ್ಡಸಾಲುಗಳು_padded >> dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು;

    ಸ್ಥಿರಾಂಕ bool act_is_f16 = (act_elem_ಗಾತ್ರ == ಗಾತ್ರ(__fp16));
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = (size_t) k_ಖಂಡ * act_elem_ಗಾತ್ರ;  // staging row (DMA dst stride)
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = (size_t) k_stride * act_elem_ಗಾತ್ರ;  // DDR row (DMA src stride)
    ಸ್ಥಿರಾಂಕ size_t ಅಗಲ_ಬೈಟುಗಳು = (size_t) k_valid * act_elem_ಗಾತ್ರ;

    // Push step 0
    ಆದರೆ (n_ಹೆಜ್ಜೆಗಳು > 0 && n_ಅಡ್ಡಸಾಲುಗಳು > 0) {
        uint32_t nrows_ಗಮ್ಯ_fetch = hex_smin(n_ಅಡ್ಡಸಾಲುಗಳು, R);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(thread_act, act_dma_addr),
                       ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ಅಗಲ_ಬೈಟುಗಳು, nrows_ಗಮ್ಯ_fetch);
    }
    // Push step 1 (if valid)
    ಆದರೆ (n_ಹೆಜ್ಜೆಗಳು > 1) {
        uint32_t ಮುಂದಿನ_r = R * 1;
        ಆದರೆ (ಮುಂದಿನ_r < n_ಅಡ್ಡಸಾಲುಗಳು) {
            uint32_t nrows_ಗಮ್ಯ_fetch = hex_smin(n_ಅಡ್ಡಸಾಲುಗಳು - ಮುಂದಿನ_r, R);
            uint8_t *ಮುಂದಿನ_buf = thread_act + 1 * R * ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ಮುಂದಿನ_buf, act_dma_addr + (size_t) ಮುಂದಿನ_r * src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು),
                           ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ಅಗಲ_ಬೈಟುಗಳು, nrows_ಗಮ್ಯ_fetch);
        }
    }
    ಗಾಗಿ (uint32_t s = 0; s < n_ಹೆಜ್ಜೆಗಳು; ++s) {
        uint32_t r = s << dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು;
        uint8_t *curr_buf = thread_act;

        ಆದರೆ (r < n_ಅಡ್ಡಸಾಲುಗಳು) {
            curr_buf = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
        }

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        ಆದರೆ (act_is_f16) {
            ಗಾಗಿ (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t ಅಡ್ಡಸಾಲು_idx = r + (p << 1);
                ಸ್ಥಿರಾಂಕ __fp16 *pair_buf = (ಸ್ಥಿರಾಂಕ __fp16 *) (curr_buf + (p << 1) * ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು);
                bool r0_valid = ((ಅಡ್ಡಸಾಲು_idx + 0) < n_ಅಡ್ಡಸಾಲುಗಳು);
                bool r1_valid = ((ಅಡ್ಡಸಾಲು_idx + 1) < n_ಅಡ್ಡಸಾಲುಗಳು);

                transfer_activation_ಅಡ್ಡಸಾಲು_pair_f16_ಗಮ್ಯ_f16(vtcm_dst, pair_buf, pair_buf + k_ಖಂಡ, ಅಡ್ಡಸಾಲು_idx, k_ಖಂಡ, k_valid, r0_valid, r1_valid);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t ಅಡ್ಡಸಾಲು_idx = r + (p << 1);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *pair_buf = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (curr_buf + (p << 1) * ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು);
                bool r0_valid = ((ಅಡ್ಡಸಾಲು_idx + 0) < n_ಅಡ್ಡಸಾಲುಗಳು);
                bool r1_valid = ((ಅಡ್ಡಸಾಲು_idx + 1) < n_ಅಡ್ಡಸಾಲುಗಳು);

                transfer_activation_ಅಡ್ಡಸಾಲು_pair_fp32_ಗಮ್ಯ_fp16(vtcm_dst, pair_buf, pair_buf + k_ಖಂಡ, ಅಡ್ಡಸಾಲು_idx, k_ಖಂಡ, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t ಮುಂದಿನ_s = s + 2;
        uint32_t ಮುಂದಿನ_r = ಮುಂದಿನ_s << dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು;
        ಆದರೆ (ಮುಂದಿನ_r < n_ಅಡ್ಡಸಾಲುಗಳು) {
            uint32_t nrows_ಗಮ್ಯ_fetch = hex_smin(n_ಅಡ್ಡಸಾಲುಗಳು - ಮುಂದಿನ_r, R);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(curr_buf, act_dma_addr + (size_t) ಮುಂದಿನ_r * src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು),
                           ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ಅಗಲ_ಬೈಟುಗಳು, nrows_ಗಮ್ಯ_fetch);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_chunk_ಕೆಲಸಗಾರ_fn(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ *ದತ್ತಾಂಶ) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) ದತ್ತಾಂಶ;

    ರಚನೆ htp_thread_trace * tr = &st->traces[i];

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ task_id = i; task_id < (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)st->n_tasks; task_id += n) {
        ಪೂರ್ಣಾಂಕ    chunk_idx  = task_id * st->n_chunks_ಪ್ರತಿ_task;
        size_t chunk_ಗಾತ್ರ = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_ಪ್ರತಿ_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_ಖಂಡ;
        ಸ್ಥಿರಾಂಕ dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_ಗಾತ್ರ;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
        transfer_activation_chunk_ಗಮ್ಯ_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_ಗಾತ್ರ, st->k_ಖಂಡ, st->k_stride, st->k_valid, thread_act, st->act_elem_ಗಾತ್ರ, tr, st->dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು, st->dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು
        );
    }
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ರಚನೆ htp_thread_trace       * traces;
    ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು;
    __fp16                        * dst;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_ಪ್ರತಿ_task;
    uint32_t                        k_ಖಂಡ;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ರಚನೆ htp_thread_trace       * traces;
    ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ __fp16                  * vtcm_src;
    ದಶಮಾಂಶ                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_ಪ್ರತಿ_task;
    uint32_t                        n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_chunk_gathered_ಕೆಲಸಗಾರ_fn(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ *ದತ್ತಾಂಶ) {
    activation_transfer_gathered_task_state_t *st = ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &st->traces[i];
    ಪೂರ್ಣಾಂಕ chunk_idx      = i;
    ಪೂರ್ಣಾಂಕ chunk_ಗಾತ್ರ     = st->n_chunks_ಪ್ರತಿ_task;
    ಪೂರ್ಣಾಂಕ vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು = chunk_idx * chunk_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ ಆರಂಭ_ಅಡ್ಡಸಾಲು      = st->ಆರಂಭ_ಅಡ್ಡಸಾಲು + vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು         = hex_smin(st->cne1 - ಆರಂಭ_ಅಡ್ಡಸಾಲು, chunk_ಗಾತ್ರ);
    ಆದರೆ (n_ಅಡ್ಡಸಾಲುಗಳು > 0) {
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_ಗಮ್ಯ_fp16_gathered(
            st->dst, st->src, ಆರಂಭ_ಅಡ್ಡಸಾಲು, vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು, n_ಅಡ್ಡಸಾಲುಗಳು, st->k_ಖಂಡ,
            st->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_chunk_gathered_ಕೆಲಸಗಾರ_flat_fn(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ *ದತ್ತಾಂಶ) {
    activation_transfer_gathered_task_state_t *st = ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &st->traces[i];
    ಪೂರ್ಣಾಂಕ chunk_idx = i;
    ಪೂರ್ಣಾಂಕ chunk_ಗಾತ್ರ = st->n_chunks_ಪ್ರತಿ_task;
    ಪೂರ್ಣಾಂಕ vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು = chunk_idx * chunk_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ ಆರಂಭ_ಅಡ್ಡಸಾಲು = st->ಆರಂಭ_ಅಡ್ಡಸಾಲು + vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು = hex_smin(st->cne1 - ಆರಂಭ_ಅಡ್ಡಸಾಲು, chunk_ಗಾತ್ರ);
    ಆದರೆ (n_ಅಡ್ಡಸಾಲುಗಳು > 0) {
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_ಗಮ್ಯ_fp16_gathered_flat(
            st->dst, st->src, ಆರಂಭ_ಅಡ್ಡಸಾಲು, vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು, n_ಅಡ್ಡಸಾಲುಗಳು, st->k_ಖಂಡ,
            st->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_output_chunk_scattered_ಕೆಲಸಗಾರ_fn(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ *ದತ್ತಾಂಶ) {
    output_transfer_scattered_task_state_t *st = ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &st->traces[i];
    ಪೂರ್ಣಾಂಕ chunk_idx = i;
    ಪೂರ್ಣಾಂಕ chunk_ಗಾತ್ರ = st->n_chunks_ಪ್ರತಿ_task;
    ಪೂರ್ಣಾಂಕ vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು = chunk_idx * chunk_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ ಆರಂಭ_ಅಡ್ಡಸಾಲು = st->ಆರಂಭ_ಅಡ್ಡಸಾಲು + vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು = hex_smin(st->cne1 - ಆರಂಭ_ಅಡ್ಡಸಾಲು, chunk_ಗಾತ್ರ);
    ಆದರೆ (n_ಅಡ್ಡಸಾಲುಗಳು > 0) {
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_ಗಮ್ಯ_fp32_scattered(
            st->dst, st->vtcm_src, ಆರಂಭ_ಅಡ್ಡಸಾಲು, vtcm_ಆರಂಭ_ಅಡ್ಡಸಾಲು, n_ಅಡ್ಡಸಾಲುಗಳು, st->n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
            st->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

ಸ್ಥಿರ ಸಾಲೊಳಗೆ tiled_dequant_task_fn_t hmx_get_dequant_ಕೆಲಸಗಾರ_fn(ಪೂರ್ಣಾಂಕ ತೂಕ_type, bool is_tiled) {
    ಆಯ್ಕೆ (ತೂಕ_type) {
        ಸ್ಥಿತಿ HTP_TYPE_Q4_0:   ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q4_0;
        ಸ್ಥಿತಿ HTP_TYPE_IQ4_NL: ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_iq4_nl;
        ಸ್ಥಿತಿ HTP_TYPE_Q4_1:
        ಸ್ಥಿತಿ HTP_TYPE_Q4_K:   ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q4_1;
        ಸ್ಥಿತಿ HTP_TYPE_MXFP4:  ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_mxfp4;
        ಸ್ಥಿತಿ HTP_TYPE_Q8_0:   ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q8_0;
        ಸ್ಥಿತಿ HTP_TYPE_Q5_K:   ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q5_k;
        ಸ್ಥಿತಿ HTP_TYPE_Q6_K:   ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q6_k;
        ಸ್ಥಿತಿ HTP_TYPE_Q3_K:   ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q3_k;
        ಸ್ಥಿತಿ HTP_TYPE_Q2_K:   ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q2_k;
        ಸ್ಥಿತಿ HTP_TYPE_Q1_0:   ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q1_0;
        ಸ್ಥಿತಿ HTP_TYPE_Q2_0:   ಹಿಂತಿರುಗಿ dequantize_tiled_ತೂಕ_ಗಮ್ಯ_fp16_task_q2_0;
        ಸ್ಥಿತಿ HTP_TYPE_F16:    ಹಿಂತಿರುಗಿ is_tiled ? ಖಾಲಿ : convert_f16_ತೂಕ_ಗಮ್ಯ_fp16_tiles_task;
        ಸ್ಥಿತಿ HTP_TYPE_BF16:   ಹಿಂತಿರುಗಿ convert_bf16_ತೂಕ_ಗಮ್ಯ_fp16_tiles_task;
        ಸ್ಥಿತಿ HTP_TYPE_F32:    ಹಿಂತಿರುಗಿ quantize_f32_ತೂಕ_ಗಮ್ಯ_fp16_tiles_task;
        ಪೂರ್ವನಿಯೋಜಿತ:              ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_tiled_ತೂಕ_chunk_ಗಮ್ಯ_fp16_tiles(
        ರಚನೆ htp_context *ctx, __fp16 *vtcm_dst,
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ತೂಕ_src_ddr,
        ಪೂರ್ಣಾಂಕ n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, ಪೂರ್ಣಾಂಕ k_ಖಂಡ,
        size_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ತೂಕ_type,
        ಪೂರ್ಣಾಂಕ n_k_tiles, ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_ಕೆಲಸಗಾರ_fn, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {

    ಆದರೆ (!dequant_ಕೆಲಸಗಾರ_fn) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪ್ರತಿಜ್ಞಾ(k_ಖಂಡ % HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ == 0);

    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_tiles = hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
    ಸ್ಥಿರಾಂಕ uint32_t n_ತೂಕ_tiles = k_ಖಂಡ / htp_tensor_get_tile_k(ತೂಕ_type);
    ಸ್ಥಿರಾಂಕ size_t n_tot_tiles = n_ಲಂಬಸಾಲು_tiles * n_ತೂಕ_tiles;

    size_t n_tiles_ಪ್ರತಿ_task = (n_ಎಳೆಗಳು == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_ಎಳೆಗಳು);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_ಪ್ರತಿ_task - 1) / n_tiles_ಪ್ರತಿ_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_ಪ್ರತಿ_task = n_tiles_ಪ್ರತಿ_task;
    state.dst              = vtcm_dst;
    state.src              = (ಸ್ಥಿರಾಂಕ uint8_t *)ತೂಕ_src_ddr;
    state.task_fn          = dequant_ಕೆಲಸಗಾರ_fn;
    state.n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ           = n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    state.k_ಖಂಡ          = k_ಖಂಡ;
    state.ಅಡ್ಡಸಾಲು_stride       = ಅಡ್ಡಸಾಲು_stride;
    state.ತೂಕ_type      = ತೂಕ_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_ತೂಕ_tiles   = n_ತೂಕ_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_ಗಾತ್ರ = htp_tensor_get_tile_ಗಾತ್ರ(ತೂಕ_type);
    state.aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(ತೂಕ_type);

    ಆದರೆ (state.n_tasks == 1 || n_ಎಳೆಗಳು == 1) {
        dequantize_tiled_ಕೆಲಸಗಾರ_loop(1, 0, &state);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಪೂರ್ಣಾಂಕ n_tasks = hex_smin((ಪೂರ್ಣಾಂಕ) state.n_tasks, n_ಎಳೆಗಳು);
        ಕೆಲಸಗಾರ_pool_ಸತತ_func(ctx->ಕೆಲಸಗಾರ_pool, dequantize_tiled_ಕೆಲಸಗಾರ_loop, &state, n_tasks);
    }
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ರಚನೆ htp_context      * ctx;
    ರಚನೆ htp_thread_trace * traces;
    ದಶಮಾಂಶ                   * dst;
    ಸ್ಥಿರಾಂಕ __fp16            * vtcm_src;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ             * src2;
    uint32_t                  n_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t                  n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು     n_ಎಳೆಗಳು_div;
} output_transfer_ಲಂಬಸಾಲು_chunk_state_t;

ಸ್ಥಿರ ಶೂನ್ಯ transfer_output_chunk_ಲಂಬಸಾಲು_chunk_ಕೆಲಸಗಾರ_fn(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ *ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) n;
    output_transfer_ಲಂಬಸಾಲು_chunk_state_t *st = (output_transfer_ಲಂಬಸಾಲು_chunk_state_t *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_ಖಂಡಗಳು = hmx_ceil_div(st->n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, 32);
    uint32_t b_ಮೊದಲ  = fastdiv(n_ಖಂಡಗಳು * i, &st->n_ಎಳೆಗಳು_div);
    uint32_t b_ಕೊನೆಯ   = fastdiv(n_ಖಂಡಗಳು * (i + 1), &st->n_ಎಳೆಗಳು_div);
    uint32_t c_ಮೊದಲ  = b_ಮೊದಲ * 32;
    uint32_t c_ಕೊನೆಯ   = b_ಕೊನೆಯ * 32;
    uint32_t c_len    = c_ಕೊನೆಯ - c_ಮೊದಲ;

    ಆದರೆ (c_len == 0) ಹಿಂತಿರುಗಿ;

    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_O_PROC, c_ಮೊದಲ);

    ಸ್ಥಿರಾಂಕ __fp16 *vtcm_src = st->vtcm_src + b_ಮೊದಲ * HTP_MM_HMX_TILE_N_ELMS;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src2      = st->src2 ? (st->src2 + c_ಮೊದಲ) : ಖಾಲಿ;
    ದಶಮಾಂಶ *dst             = st->dst + c_ಮೊದಲ;

    ಪೂರ್ಣಾಂಕ chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = (ಪೂರ್ಣಾಂಕ)st->dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ - (ಪೂರ್ಣಾಂಕ)c_ಮೊದಲ;
    ಆದರೆ (chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ > 0) {
        transfer_output_chunk_fp16_ಗಮ್ಯ_fp32_ಲಂಬಸಾಲು_chunk(
            dst, src2, vtcm_src, 0, st->n_ಅಡ್ಡಸಾಲುಗಳು, c_len, st->n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_ಮೊದಲ);
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_output_chunk_threaded(ರಚನೆ htp_context *ctx, ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src2, ಸ್ಥಿರಾಂಕ __fp16 *vtcm_src,
                                              ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, ಪೂರ್ಣಾಂಕ dst_stride, uint32_t src2_stride, ಪೂರ್ಣಾಂಕ dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    ಆದರೆ (n_ಅಡ್ಡಸಾಲುಗಳು <= 0) ಹಿಂತಿರುಗಿ;

    uint32_t n_ಖಂಡಗಳು = hmx_ceil_div((uint32_t) n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, 32);
    ಆದರೆ (n_ಎಳೆಗಳು > 1 && n_ಖಂಡಗಳು >= (uint32_t)n_ಎಳೆಗಳು) {
        ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು n_ಎಳೆಗಳು_div = (n_ಎಳೆಗಳು == (ಪೂರ್ಣಾಂಕ)ctx->n_ಎಳೆಗಳು) ? ctx->n_ಎಳೆಗಳು_div : init_fastdiv_ಮೌಲ್ಯಗಳು(n_ಎಳೆಗಳು);
        output_transfer_ಲಂಬಸಾಲು_chunk_state_t ಲಂಬಸಾಲು_state;
        ಲಂಬಸಾಲು_state.dst = dst;
        ಲಂಬಸಾಲು_state.src2 = src2;
        ಲಂಬಸಾಲು_state.vtcm_src = vtcm_src;
        ಲಂಬಸಾಲು_state.n_ಅಡ್ಡಸಾಲುಗಳು = (uint32_t)n_ಅಡ್ಡಸಾಲುಗಳು;
        ಲಂಬಸಾಲು_state.n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = (uint32_t)n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
        ಲಂಬಸಾಲು_state.dst_stride = (uint32_t)dst_stride;
        ಲಂಬಸಾಲು_state.src2_stride = src2_stride;
        ಲಂಬಸಾಲು_state.dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = (uint32_t)dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
        ಲಂಬಸಾಲು_state.n_ಎಳೆಗಳು_div = n_ಎಳೆಗಳು_div;
        ಲಂಬಸಾಲು_state.traces = ctx->trace;
        ಲಂಬಸಾಲು_state.ctx = ctx;

        ಕೆಲಸಗಾರ_pool_ಸತತ_func(ctx->ಕೆಲಸಗಾರ_pool, transfer_output_chunk_ಲಂಬಸಾಲು_chunk_ಕೆಲಸಗಾರ_fn, &ಲಂಬಸಾಲು_state, n_ಎಳೆಗಳು);
        ಹಿಂತಿರುಗಿ;
    }

    size_t n_tot_chunks      = n_ಅಡ್ಡಸಾಲುಗಳು;
    size_t n_chunks_ಪ್ರತಿ_task = (n_ಎಳೆಗಳು == 1) ? n_tot_chunks : hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, n_ಎಳೆಗಳು);
    n_chunks_ಪ್ರತಿ_task        = hex_align_ಮೇಲೆ(n_chunks_ಪ್ರತಿ_task, 2);

    ಪೂರ್ಣಾಂಕ actual_ಎಳೆಗಳು = hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, n_chunks_ಪ್ರತಿ_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_ಎಳೆಗಳು;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_ಪ್ರತಿ_task = n_chunks_ಪ್ರತಿ_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ            = n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ          = dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    state.traces            = ctx->trace;

    ಆದರೆ (actual_ಎಳೆಗಳು <= 1) {
        transfer_output_chunk_ಕೆಲಸಗಾರ_fn(1, 0, &state);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಕೆಲಸಗಾರ_pool_ಸತತ_func(ctx->ಕೆಲಸಗಾರ_pool, transfer_output_chunk_ಕೆಲಸಗಾರ_fn, &state, actual_ಎಳೆಗಳು);
    }
}

ರಚನೆ activation_transfer_params {
    ರಚನೆ htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    ಪೂರ್ಣಾಂಕ                           n_ಅಡ್ಡಸಾಲುಗಳು;
    ಪೂರ್ಣಾಂಕ                           k_ಖಂಡ;
    ಪೂರ್ಣಾಂಕ                           k_stride;
    ಪೂರ್ಣಾಂಕ                           n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * act_ಎಳೆಗಳು_div;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * k_div;
    ಪೂರ್ಣಾಂಕ                           k_valid;
    ದಶಮಾಂಶ *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_ಬೈಟುಗಳು;
    uint32_t                      act_elem_ಗಾತ್ರ;  // 4=F32 src1, 2=F16 src1
};

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_chunk_threaded(ಸ್ಥಿರಾಂಕ ರಚನೆ activation_transfer_params * params) {
    ರಚನೆ htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ಸ್ಥಿರಾಂಕ dma_addr_t              act_dma_addr       = params->act_dma_addr;
    ಪೂರ್ಣಾಂಕ                           n_ಅಡ್ಡಸಾಲುಗಳು             = params->n_ಅಡ್ಡಸಾಲುಗಳು;
    ಪೂರ್ಣಾಂಕ                           k_ಖಂಡ            = params->k_ಖಂಡ;
    ಪೂರ್ಣಾಂಕ                           k_stride           = params->k_stride;
    ಪೂರ್ಣಾಂಕ                           n_ಎಳೆಗಳು          = params->n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * act_ಎಳೆಗಳು_div    = params->act_ಎಳೆಗಳು_div;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * k_div              = params->k_div;
    ಪೂರ್ಣಾಂಕ                           k_valid            = params->k_valid;
    ದಶಮಾಂಶ *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_ಬೈಟುಗಳು = params->vtcm_f32_act_ಬೈಟುಗಳು;
    // element size of the activation rows (4 = F32, 2 = F16).
    ಸ್ಥಿರಾಂಕ uint32_t                act_elem_ಗಾತ್ರ      = params->act_elem_ಗಾತ್ರ ? params->act_elem_ಗಾತ್ರ : (uint32_t) ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಆದರೆ (n_ಅಡ್ಡಸಾಲುಗಳು <= 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ size_t n_tasks = (n_ಅಡ್ಡಸಾಲುಗಳು + 31) >> 5;
    ಆದರೆ (act_elem_ಗಾತ್ರ == ಗಾತ್ರ(ದಶಮಾಂಶ) && n_ಎಳೆಗಳು > 1 && k_ಖಂಡ > 32 && n_tasks < (size_t) n_ಎಳೆಗಳು) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು = 2;
        uint32_t dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು = 1;
        ಆದರೆ (vtcm_f32_act_ಬೈಟುಗಳು > 0) {
            size_t thread_scratch_ಬೈಟುಗಳು = hex_align_ಕೆಳಕ್ಕೆ(fastdiv(vtcm_f32_act_ಬೈಟುಗಳು, act_ಎಳೆಗಳು_div), 128);
            size_t thread_scratch_elements = thread_scratch_ಬೈಟುಗಳು / ಗಾತ್ರ(ದಶಮಾಂಶ);
            size_t dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_max = fastdiv(thread_scratch_elements / 2, k_div);
            ಆದರೆ (dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_max >= 4) {
                dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು = 4;
                dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು = 2;
            }
        }

        activation_transfer_ಲಂಬಸಾಲು_chunk_state_t ಲಂಬಸಾಲು_state;
        ಲಂಬಸಾಲು_state.dst = dst;
        ಲಂಬಸಾಲು_state.act_dma_addr = act_dma_addr;
        ಲಂಬಸಾಲು_state.n_ಅಡ್ಡಸಾಲುಗಳು = n_ಅಡ್ಡಸಾಲುಗಳು;
        ಲಂಬಸಾಲು_state.k_ಖಂಡ = k_ಖಂಡ;
        ಲಂಬಸಾಲು_state.k_stride = k_stride;
        ಲಂಬಸಾಲು_state.k_valid = k_valid;
        ಲಂಬಸಾಲು_state.n_ಲಂಬಸಾಲು_chunks = n_ಎಳೆಗಳು;
        ಲಂಬಸಾಲು_state.n_ಎಳೆಗಳು_div = *act_ಎಳೆಗಳು_div;
        ಲಂಬಸಾಲು_state.vtcm_f32_act = vtcm_f32_act;
        ಲಂಬಸಾಲು_state.vtcm_f32_act_ಬೈಟುಗಳು = vtcm_f32_act_ಬೈಟುಗಳು;
        ಲಂಬಸಾಲು_state.traces = ctx->trace;
        ಲಂಬಸಾಲು_state.ctx = ctx;
        ಲಂಬಸಾಲು_state.dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು = dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು;
        ಲಂಬಸಾಲು_state.dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು = dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು;

        ಕೆಲಸಗಾರ_pool_ಸತತ_func(ctx->ಕೆಲಸಗಾರ_pool, transfer_activation_chunk_ಲಂಬಸಾಲು_chunk_ಕೆಲಸಗಾರ_fn, &ಲಂಬಸಾಲು_state, n_ಎಳೆಗಳು);
        ಹಿಂತಿರುಗಿ;
    }

    ಪ್ರತಿಜ್ಞಾ(k_ಖಂಡ % HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ == 0 && k_stride % HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ == 0);

    size_t n_tot_chunks      = n_ಅಡ್ಡಸಾಲುಗಳು;
    size_t n_chunks_ಪ್ರತಿ_task = (n_ಎಳೆಗಳು == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_ಎಳೆಗಳು == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
    state.n_tot_chunks       = n_tot_chunks;
    state.n_chunks_ಪ್ರತಿ_task  = n_chunks_ಪ್ರತಿ_task;
    state.dst                = dst;
    state.act_dma_addr       = act_dma_addr;
    state.k_ಖಂಡ            = k_ಖಂಡ;
    state.k_stride           = k_stride;
    state.k_valid            = k_valid;
    state.traces             = ctx->trace;
    state.ctx                = ctx;
    state.vtcm_f32_act       = vtcm_f32_act;
    state.act_elem_ಗಾತ್ರ = act_elem_ಗಾತ್ರ;

    state.vtcm_f32_act_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = hex_align_ಕೆಳಕ್ಕೆ(fastdiv(vtcm_f32_act_ಬೈಟುಗಳು, act_ಎಳೆಗಳು_div), 128);

    uint32_t dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು = 2;
    uint32_t dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು = 1;
    ಆದರೆ (state.vtcm_f32_act_ಬೈಟುಗಳು_ಪ್ರತಿ_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_ಬೈಟುಗಳು_ಪ್ರತಿ_thread / ಗಾತ್ರ(ದಶಮಾಂಶ);
        size_t dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_max = fastdiv(thread_scratch_elements / 2, k_div);
        ಆದರೆ (dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_max >= 4) {
            dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು = 4;
            dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು = 2;
        }
    }
    state.dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು      = dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು;
    state.dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು = dma_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲುಗಳು_ಸರಿಸು;

    ಪೂರ್ಣಾಂಕ ಸಕ್ರಿಯ_ಎಳೆಗಳು = hex_smin(n_ಎಳೆಗಳು, (ಪೂರ್ಣಾಂಕ)state.n_tasks);
    ಆದರೆ (state.n_tasks == 1 || n_ಎಳೆಗಳು == 1) {
        transfer_activation_chunk_ಕೆಲಸಗಾರ_fn(1, 0, &state);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಕೆಲಸಗಾರ_pool_ಸತತ_func(ctx->ಕೆಲಸಗಾರ_pool, transfer_activation_chunk_ಕೆಲಸಗಾರ_fn, &state, ಸಕ್ರಿಯ_ಎಳೆಗಳು);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    __fp16 *       output;
    ಸ್ಥಿರಾಂಕ __fp16 * activation;
    ಸ್ಥಿರಾಂಕ __fp16 * ತೂಕ;
    ಸ್ಥಿರಾಂಕ __fp16 * scales;
    uint32_t       n_ಅಡ್ಡಸಾಲು_tiles;
    uint32_t       n_ಲಂಬಸಾಲು_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_ಕೆಲಸ_t;

ಸ್ಥಿರ ಶೂನ್ಯ hmx_matmul_ಕೆಲಸಗಾರ_fn(ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    hmx_matmul_ಕೆಲಸ_t * ಕೆಲಸ = (hmx_matmul_ಕೆಲಸ_t *) ದತ್ತಾಂಶ;
    FARF(ಮೇಲಿನ_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", ಕೆಲಸ->n_ಅಡ್ಡಸಾಲು_tiles, ಕೆಲಸ->n_ಲಂಬಸಾಲು_tiles, ಕೆಲಸ->n_dot_tiles);
    core_dot_chunk_fp16(ಕೆಲಸ->output, ಕೆಲಸ->activation, ಕೆಲಸ->ತೂಕ, ಕೆಲಸ->scales, ಕೆಲಸ->n_ಅಡ್ಡಸಾಲು_tiles, ಕೆಲಸ->n_ಲಂಬಸಾಲು_tiles, ಕೆಲಸ->n_dot_tiles);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_matmul_ಕೆಲಸ_init(hmx_matmul_ಕೆಲಸ_t * ಕೆಲಸ,
                                       __fp16 *           output,
                                       ಸ್ಥಿರಾಂಕ __fp16 *     activation,
                                       ಸ್ಥಿರಾಂಕ __fp16 *     ತೂಕ,
                                       ಸ್ಥಿರಾಂಕ __fp16 *     scales,
                                       uint32_t           n_ಅಡ್ಡಸಾಲು_tiles,
                                       uint32_t           n_ಲಂಬಸಾಲು_tiles,
                                       uint32_t           n_dot_tiles) {
    ಕೆಲಸ->output      = output;
    ಕೆಲಸ->activation  = activation;
    ಕೆಲಸ->ತೂಕ      = ತೂಕ;
    ಕೆಲಸ->scales      = scales;
    ಕೆಲಸ->n_ಅಡ್ಡಸಾಲು_tiles = n_ಅಡ್ಡಸಾಲು_tiles;
    ಕೆಲಸ->n_ಲಂಬಸಾಲು_tiles = n_ಲಂಬಸಾಲು_tiles;
    ಕೆಲಸ->n_dot_tiles = n_dot_tiles;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hmx_mm_2d_f32(ರಚನೆ htp_context *ctx,
                                  dma_ಸರತಿ *ತೂಕ_dma,
                                  ದಶಮಾಂಶ *ಸೀಮಿತ dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_ಬೈಟುಗಳು,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t ತೂಕ,
                                  ಪೂರ್ಣಾಂಕ m, ಪೂರ್ಣಾಂಕ k, ಪೂರ್ಣಾಂಕ n,
                                  ಪೂರ್ಣಾಂಕ act_stride,
                                  uint32_t act_elem_ಗಾತ್ರ,  // 4=F32 src1, 2=F16 src1
                                  ಪೂರ್ಣಾಂಕ ತೂಕ_stride,
                                  ಪೂರ್ಣಾಂಕ ತೂಕ_type,
                                  ಪೂರ್ಣಾಂಕ k_valid,
                                  ಪೂರ್ಣಾಂಕ dst_stride,
                                  uint32_t src2_stride,
                                  ಪೂರ್ಣಾಂಕ dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
                                  ಪೂರ್ಣಾಂಕ m_chunk,
                                  ಪೂರ್ಣಾಂಕ n_chunk,
                                  ಪೂರ್ಣಾಂಕ pipeline,
                                  ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು,
                                  ಪೂರ್ಣಾಂಕ act_ಎಳೆಗಳು,
                                  ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * act_ಎಳೆಗಳು_div,
                                  ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * k_div,
                                  ಪೂರ್ಣಾಂಕ tile_ಗಾತ್ರ,
                                  ಪೂರ್ಣಾಂಕ aligned_tile_ಗಾತ್ರ,
                                  ಪೂರ್ಣಾಂಕ vtcm_ಗಾತ್ರ) {
    ರಚನೆ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, 0);

    ಸ್ಥಿರಾಂಕ bool is_tiled = (tile_ಗಾತ್ರ > 0);

    ಆದರೆ (k % 32 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆದರೆ (!hex_is_aligned(dst, ಗಾತ್ರ(ದಶಮಾಂಶ))) { ಹಿಂತಿರುಗಿ -1; }

    size_t ಅಡ್ಡಸಾಲು_stride = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ತೂಕ_type, k);
    ಆದರೆ (ಅಡ್ಡಸಾಲು_stride == 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    tiled_dequant_task_fn_t dequant_ಕೆಲಸಗಾರ_fn = hmx_get_dequant_ಕೆಲಸಗಾರ_fn(ತೂಕ_type, is_tiled);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_k_tiles = k / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ತೂಕ_tiles = k / htp_tensor_get_tile_k(ತೂಕ_type);
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು n_k_tiles_div = init_fastdiv_ಮೌಲ್ಯಗಳು(n_k_tiles);

    ಸ್ಥಿರಾಂಕ size_t vec_dot_ಗಾತ್ರ = k * ಗಾತ್ರ(__fp16);
    ಸ್ಥಿರಾಂಕ size_t vtcm_budget  = ctx->vtcm_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ uint32_t dma_dst_stride  = is_tiled ? aligned_tile_ಗಾತ್ರ : ಅಡ್ಡಸಾಲು_stride;
    ಸ್ಥಿರಾಂಕ uint32_t dma_src_stride  = is_tiled ? tile_ಗಾತ್ರ : ತೂಕ_stride;
    ಸ್ಥಿರಾಂಕ uint32_t dma_ಅಗಲ_ಬೈಟುಗಳು = is_tiled ? tile_ಗಾತ್ರ : ಅಡ್ಡಸಾಲು_stride;

    size_t m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು = m_chunk;
    size_t n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = n_chunk;
    size_t vtcm_ಬಳಸಿದ      = vtcm_ಗಾತ್ರ;

    ರಚನೆ htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, ತೂಕ_type, k, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, 1, pipeline, act_ಎಳೆಗಳು, aligned_tile_ಗಾತ್ರ, src2_ಬೈಟುಗಳು);

    vtcm_ಬಳಸಿದ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;
    ಆದರೆ (vtcm_ಬಳಸಿದ > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_ಬಳಸಿದ, vtcm_budget, m, k, n, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
        ಹಿಂತಿರುಗಿ -1;
    }

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = (uint8_t *) ctx->vtcm_ತಳ;
    __fp16  *vtcm_ತೂಕ_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_ತೂಕ[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ತಳ, L.off_ತೂಕ[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_act);
    ದಶಮಾಂಶ   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, ತಳ, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_dst[0]);
    ಶೂನ್ಯ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(ಶೂನ್ಯ, ತಳ, L.off_scratch[0]);
    ಶೂನ್ಯ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(ಶೂನ್ಯ, ತಳ, L.off_scratch[1], pipeline);
    ಶೂನ್ಯ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(ಶೂನ್ಯ, ತಳ, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_scales);

    hmx_init_ಸ್ತಂಭ_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ಸ್ಥಿರಾಂಕ bool has_bias      = (src2_ಬೈಟುಗಳು > 0 && src2_addr != 0);
    ದಶಮಾಂಶ   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(ದಶಮಾಂಶ, ತಳ, L.off_bias, has_bias);
    ಆದರೆ (has_bias) {
        dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_bias, src2_addr), hex_align_ಮೇಲೆ(src2_ಬೈಟುಗಳು, 128), 0, src2_ಬೈಟುಗಳು, 1);
        dma_ಸರತಿ_ತೆಗೆ(ತೂಕ_dma);
    }

    FARF(ಮೇಲಿನ_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, ತೂಕ_type, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, vtcm_ಬಳಸಿದ, vtcm_budget);

    ಪೂರ್ಣಾಂಕ n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ಆದರೆ (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_ಕೆಲಸ_t ಕೆಲಸ_slots[2];  // persistent double-buffered job descriptors
        ಶೂನ್ಯ *vtcm_raw_bufs[4] = {
            vtcm_ತೂಕ_raw[0],
            vtcm_ತೂಕ_raw[1],
            dequant_ಕೆಲಸಗಾರ_fn ? vtcm_ತೂಕ_raw[0] : vtcm_scratch0,
            dequant_ಕೆಲಸಗಾರ_fn ? vtcm_ತೂಕ_raw[1] : vtcm_scratch1,
        };
        ಶೂನ್ಯ *vtcm_ತೂಕ_bufs[4] = {
            dequant_ಕೆಲಸಗಾರ_fn ? vtcm_scratch0 : vtcm_ತೂಕ_raw[0],
            dequant_ಕೆಲಸಗಾರ_fn ? vtcm_scratch1 : vtcm_ತೂಕ_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        ಶೂನ್ಯ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        ಗಾಗಿ (size_t mr = 0; mr < m; mr += m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು) {
            ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು = hex_smin(m - mr, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು);

            ರಚನೆ activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ಗಾತ್ರ,  // byte offset (F16/F32)
                .n_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ) n_ಅಡ್ಡಸಾಲುಗಳು,
                .k_ಖಂಡ = k,
                .k_stride = act_stride,
                .n_ಎಳೆಗಳು = act_ಎಳೆಗಳು,
                .act_ಎಳೆಗಳು_div = act_ಎಳೆಗಳು_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_ಬೈಟುಗಳು = L.act_f32_ಬೈಟುಗಳು,
                .act_elem_ಗಾತ್ರ = act_elem_ಗಾತ್ರ,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            ಸ್ಥಿರಾಂಕ size_t   n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A0 = hex_smin(n - 0 * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
            ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ_A0 = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A0, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A0;
            dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[0], ತೂಕ),
                           dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ_A0);

            ಆದರೆ (1 < n_chunk_cnt) {
                ಸ್ಥಿರಾಂಕ size_t   n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A1 = hex_smin(n - 1 * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ_A1 = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A1, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A1;
                dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[1], ತೂಕ + n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ತೂಕ_stride),
                               dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_chunk_cnt; ++i) {
                ಸ್ಥಿರಾಂಕ size_t nc    = i * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                ಸ್ಥಿರಾಂಕ size_t nc_p2 = nc + 2 * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

                ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ    = hex_smin(n - nc, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_p2 = hex_smin(n - nc_p2, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                // 1. pop A_i
                ಶೂನ್ಯ * curr_raw = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(ತೂಕ_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_ತೂಕ_chunk_ಗಮ್ಯ_fp16_tiles(
                    ctx, vtcm_ತೂಕ_bufs[i % 4], curr_raw,
                    n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, k, ಅಡ್ಡಸಾಲು_stride, ತೂಕ_type,
                    n_k_tiles, n_k_tiles_div, dequant_ಕೆಲಸಗಾರ_fn, n_ಎಳೆಗಳು);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                ಆದರೆ (i + 2 < n_chunk_cnt) {
                    ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ_p2 = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_p2, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_p2;
                    dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[(i + 2) % 4], ತೂಕ + nc_p2 * ತೂಕ_stride),
                                   dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ_p2);
                }

                // 4. submit C_i
                hmx_matmul_ಕೆಲಸ_init(&ಕೆಲಸ_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_ತೂಕ_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು),
                                    hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ), k / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
                hmx_ಸರತಿ_ತಳ್ಳು(ctx->hmx_ಸರತಿ, hmx_ಸರತಿ_make_desc(hmx_matmul_ಕೆಲಸಗಾರ_fn, &ಕೆಲಸ_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                ಆದರೆ (i > 0) {
                    hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);
                    ಸ್ಥಿರಾಂಕ size_t nc_prev = (i - 1) * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_prev = hex_smin(n - nc_prev, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    ದಶಮಾಂಶ *output_chunk = dst + (mr * dst_stride + nc_prev);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : ಖಾಲಿ;
                    ಪೂರ್ಣಾಂಕ chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ - (ಪೂರ್ಣಾಂಕ)nc_prev;
                    ಆದರೆ (chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_ಅಡ್ಡಸಾಲುಗಳು, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_prev, dst_stride, src2_stride, chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_ಎಳೆಗಳು);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);
            ಸ್ಥಿರಾಂಕ size_t nc_ಕೊನೆಯ = (n_chunk_cnt - 1) * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಕೊನೆಯ = hex_smin(n - nc_ಕೊನೆಯ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
            ದಶಮಾಂಶ *output_chunk = dst + (mr * dst_stride + nc_ಕೊನೆಯ);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_ಕೊನೆಯ) : ಖಾಲಿ;
            ಪೂರ್ಣಾಂಕ chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ - (ಪೂರ್ಣಾಂಕ)nc_ಕೊನೆಯ;
            ಆದರೆ (chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_ಅಡ್ಡಸಾಲುಗಳು, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಕೊನೆಯ, dst_stride, src2_stride, chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_ಎಳೆಗಳು);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_ಕೆಲಸ_t ಕೆಲಸ;
        ಶೂನ್ಯ *vtcm_raw_bufs[2] = { vtcm_ತೂಕ_raw[0], dequant_ಕೆಲಸಗಾರ_fn ? vtcm_ತೂಕ_raw[0] : vtcm_scratch0 };
        ಶೂನ್ಯ *vtcm_ತೂಕ_bufs[2] = { dequant_ಕೆಲಸಗಾರ_fn ? vtcm_scratch0 : vtcm_ತೂಕ_raw[0], vtcm_scratch0 };

        ಗಾಗಿ (size_t mr = 0; mr < m; mr += m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು) {
            ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು = hex_smin(m - mr, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು);

            ರಚನೆ activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ಗಾತ್ರ,  // byte offset (F16/F32)
                .n_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ) n_ಅಡ್ಡಸಾಲುಗಳು,
                .k_ಖಂಡ = k,
                .k_stride = act_stride,
                .n_ಎಳೆಗಳು = act_ಎಳೆಗಳು,
                .act_ಎಳೆಗಳು_div = act_ಎಳೆಗಳು_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_ಬೈಟುಗಳು = L.act_f32_ಬೈಟುಗಳು,
                .act_elem_ಗಾತ್ರ = act_elem_ಗಾತ್ರ,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            ಆದರೆ (n > 0) {
                ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = hex_smin(n, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[0], ತೂಕ), dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ);
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_chunk_cnt; ++i) {
                ಸ್ಥಿರಾಂಕ size_t nc = (size_t) i * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = hex_smin(n - nc, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲು_tiles = hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
                ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_tiles = hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                // A: Wait for weight DMA
                ಶೂನ್ಯ * curr_raw = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(ತೂಕ_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_ತೂಕ_chunk_ಗಮ್ಯ_fp16_tiles(
                    ctx, vtcm_ತೂಕ_bufs[i % 2], curr_raw,
                    n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, k, ಅಡ್ಡಸಾಲು_stride, ತೂಕ_type,
                    n_k_tiles, n_k_tiles_div, dequant_ಕೆಲಸಗಾರ_fn, n_ಎಳೆಗಳು);

                // Start weight DMA for the next chunk early
                ಸ್ಥಿರಾಂಕ size_t nc_ಮುಂದಿನ = nc + n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                ಆದರೆ (nc_ಮುಂದಿನ < n) {
                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ = hex_smin(n - nc_ಮುಂದಿನ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ_ಮುಂದಿನ = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ;
                    dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[(i + 1) % 2], ತೂಕ + nc_ಮುಂದಿನ * ತೂಕ_stride), dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ_ಮುಂದಿನ);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_ಕೆಲಸ_init(&ಕೆಲಸ, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_ತೂಕ_bufs[i % 2], vtcm_scales, n_ಅಡ್ಡಸಾಲು_tiles, n_ಲಂಬಸಾಲು_tiles, k / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
                hmx_ಸರತಿ_ತಳ್ಳು(ctx->hmx_ಸರತಿ, hmx_ಸರತಿ_make_desc(hmx_matmul_ಕೆಲಸಗಾರ_fn, &ಕೆಲಸ));
                hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);

                // D: Output Store
                ದಶಮಾಂಶ *output_chunk = dst + (mr * dst_stride + nc);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : ಖಾಲಿ;
                ಪೂರ್ಣಾಂಕ chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ - (ಪೂರ್ಣಾಂಕ)nc;
                ಆದರೆ (chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_ಅಡ್ಡಸಾಲುಗಳು, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, dst_stride, src2_stride, chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_ಎಳೆಗಳು);
                }
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hmx_mm_nx_2d_f32(ರಚನೆ htp_ops_context * octx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams) {
    ರಚನೆ htp_context * ctx = octx->ctx;
    ರಚನೆ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, 0);

    ಸ್ಥಿರಾಂಕ uint32_t n_weights = kparams->n_weights;
    ಆದರೆ (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ act  = octx->src[n_weights];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತೂಕ_type = (ಪೂರ್ಣಾಂಕ) src0->type;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k           = (ಪೂರ್ಣಾಂಕ) act->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_valid     = (ಪೂರ್ಣಾಂಕ) act->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m           = (ಪೂರ್ಣಾಂಕ) (act->ne[1] * act->ne[2] * act->ne[3]);
    ಸ್ಥಿರಾಂಕ uint32_t act_elem_ಗಾತ್ರ = (act->type == HTP_TYPE_F16) ? ಗಾತ್ರ(__fp16) : ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ act_stride  = (ಪೂರ್ಣಾಂಕ) (act->nb[1] / act_elem_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ dma_addr_t act_dma_addr = act->ದತ್ತಾಂಶ;

    ಆದರೆ (k % 32 != 0) { ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT; }

    size_t ಅಡ್ಡಸಾಲು_stride = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ತೂಕ_type, k);
    ಆದರೆ (ಅಡ್ಡಸಾಲು_stride == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_k_tiles = k / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ತೂಕ_tiles = k / htp_tensor_get_tile_k(ತೂಕ_type);
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು n_k_tiles_div = init_fastdiv_ಮೌಲ್ಯಗಳು(n_k_tiles);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tile_ಗಾತ್ರ       = kparams->tile_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ aligned_tile_ಗಾತ್ರ = kparams->aligned_tile_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ bool is_tiled       = (tile_ಗಾತ್ರ > 0);
    tiled_dequant_task_fn_t dequant_ಕೆಲಸಗಾರ_fn = hmx_get_dequant_ಕೆಲಸಗಾರ_fn(ತೂಕ_type, is_tiled);
    ಸ್ಥಿರಾಂಕ size_t vtcm_budget  = ctx->vtcm_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು  = kparams->m_chunk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ  = kparams->n_chunk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pipeline        = kparams->pipeline;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು       = octx->n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ act_ಎಳೆಗಳು     = kparams->n_act_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * act_ಎಳೆಗಳು_div = &kparams->div_n_act_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * k_div           = &kparams->div_ne00_padded;

    ಸ್ಥಿರಾಂಕ uint32_t dma_dst_stride  = is_tiled ? aligned_tile_ಗಾತ್ರ : ಅಡ್ಡಸಾಲು_stride;
    ಸ್ಥಿರಾಂಕ uint32_t dma_ಅಗಲ_ಬೈಟುಗಳು = is_tiled ? tile_ಗಾತ್ರ : ಅಡ್ಡಸಾಲು_stride;

    ರಚನೆ htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, ತೂಕ_type, k, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, 1, pipeline, act_ಎಳೆಗಳು, aligned_tile_ಗಾತ್ರ, 0);

    ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.ಒಟ್ಟು_ಬೈಟುಗಳು, vtcm_budget, m, k, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = (uint8_t *) ctx->vtcm_ತಳ;
    __fp16  *vtcm_ತೂಕ_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_ತೂಕ[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ತಳ, L.off_ತೂಕ[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_act);
    ದಶಮಾಂಶ   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, ತಳ, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_dst[0]);
    ಶೂನ್ಯ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(ಶೂನ್ಯ, ತಳ, L.off_scratch[0]);
    ಶೂನ್ಯ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(ಶೂನ್ಯ, ತಳ, L.off_scratch[1], pipeline);
    ಶೂನ್ಯ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(ಶೂನ್ಯ, ತಳ, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_scales);

    hmx_init_ಸ್ತಂಭ_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ಪೂರ್ಣಾಂಕ m_ಆರಂಭ = 0;
    ಪೂರ್ಣಾಂಕ m_ಅಡ್ಡಸಾಲುಗಳು  = m;
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        bool can_ವಿಭಜಿಸು = htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(act, act_elem_ಗಾತ್ರ);
        ಆದರೆ (can_ವಿಭಜಿಸು) {
            ಗಾಗಿ (uint32_t p = 0; p < n_weights; ++p) {
                ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst = octx->dsts[p];
                ಆದರೆ (!htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, ಗಾತ್ರ(ದಶಮಾಂಶ))) {
                    can_ವಿಭಜಿಸು = false;
                    ನಿಲ್ಲಿಸು;
                }
            }
        }
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ((uint32_t) m, can_ವಿಭಜಿಸು ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        m_ಆರಂಭ = (ಪೂರ್ಣಾಂಕ) range.ಆರಂಭ;
        m_ಅಡ್ಡಸಾಲುಗಳು  = (ಪೂರ್ಣಾಂಕ) range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (m_ಅಡ್ಡಸಾಲುಗಳು == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    FARF(ಮೇಲಿನ_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_ಆರಂಭ, m_ಆರಂಭ + m_ಅಡ್ಡಸಾಲುಗಳು, k, ತೂಕ_type, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, L.ಒಟ್ಟು_ಬೈಟುಗಳು, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ಸ್ಥಿರಾಂಕ size_t mr_ಕೊನೆ = (size_t)(m_ಆರಂಭ + m_ಅಡ್ಡಸಾಲುಗಳು);

    ಆದರೆ (pipeline) {
        hmx_matmul_ಕೆಲಸ_t ಕೆಲಸ_slots[2];
        ಶೂನ್ಯ *vtcm_raw_bufs[4] = {
            vtcm_ತೂಕ_raw[0],
            vtcm_ತೂಕ_raw[1],
            dequant_ಕೆಲಸಗಾರ_fn ? vtcm_ತೂಕ_raw[0] : vtcm_scratch0,
            dequant_ಕೆಲಸಗಾರ_fn ? vtcm_ತೂಕ_raw[1] : vtcm_scratch1,
        };
        ಶೂನ್ಯ *vtcm_ತೂಕ_bufs[4] = {
            dequant_ಕೆಲಸಗಾರ_fn ? vtcm_scratch0 : vtcm_ತೂಕ_raw[0],
            dequant_ಕೆಲಸಗಾರ_fn ? vtcm_scratch1 : vtcm_ತೂಕ_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        ಶೂನ್ಯ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        ಗಾಗಿ (size_t mr = (size_t) m_ಆರಂಭ; mr < mr_ಕೊನೆ; mr += m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು) {
            ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು = hex_smin(mr_ಕೊನೆ - mr, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು);

            ರಚನೆ activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ಗಾತ್ರ,
                .n_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ) n_ಅಡ್ಡಸಾಲುಗಳು,
                .k_ಖಂಡ = k,
                .k_stride = act_stride,
                .n_ಎಳೆಗಳು = act_ಎಳೆಗಳು,
                .act_ಎಳೆಗಳು_div = act_ಎಳೆಗಳು_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_ಬೈಟುಗಳು = L.act_f32_ಬೈಟುಗಳು,
                .act_elem_ಗಾತ್ರ = act_elem_ಗಾತ್ರ,
            };
            transfer_activation_chunk_threaded(&act_params);

            ಗಾಗಿ (uint32_t p = 0; p < n_weights; p++) {
                ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src_w = octx->src[p];
                ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst   = octx->dsts[p];
                ಆದರೆ (!src_w || !dst) ಮುಂದುವರಿ;

                ಸ್ಥಿರಾಂಕ dma_addr_t ತೂಕ       = src_w->ದತ್ತಾಂಶ;
                dma_ಸರತಿ * ತೂಕ_dma      = octx->ctx->dma[0];
                ದಶಮಾಂಶ * dst_ptr            = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
                ಸ್ಥಿರಾಂಕ size_t n             = src_w->ne[1];
                ಆದರೆ (n == 0) ಮುಂದುವರಿ;
                ಸ್ಥಿರಾಂಕ size_t ತೂಕ_stride = src_w->nb[1];
                ಸ್ಥಿರಾಂಕ size_t dst_stride    = dst->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ         = (ಪೂರ್ಣಾಂಕ) dst->ne[0];
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                ಸ್ಥಿರಾಂಕ uint32_t dma_src_stride = is_tiled ? tile_ಗಾತ್ರ : ತೂಕ_stride;

                ಸ್ಥಿರಾಂಕ size_t   n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A0 = hex_smin(n - 0 * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ_A0 = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A0, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A0;
                dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[0], ತೂಕ),
                               dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ_A0);

                ಆದರೆ (1 < n_chunk_cnt) {
                    ಸ್ಥಿರಾಂಕ size_t   n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A1 = hex_smin(n - 1 * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ_A1 = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A1, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_A1;
                    dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[1], ತೂಕ + n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ತೂಕ_stride),
                                   dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ_A1);
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_chunk_cnt; ++i) {
                    ಸ್ಥಿರಾಂಕ size_t nc    = i * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                    ಸ್ಥಿರಾಂಕ size_t nc_p2 = nc + 2 * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ    = hex_smin(n - nc, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_p2 = hex_smin(n - nc_p2, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                    ಶೂನ್ಯ * curr_raw = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(ತೂಕ_dma).dst;

                    dequantize_tiled_ತೂಕ_chunk_ಗಮ್ಯ_fp16_tiles(
                        ctx, vtcm_ತೂಕ_bufs[i % 4], curr_raw,
                        n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, k, ಅಡ್ಡಸಾಲು_stride, ತೂಕ_type,
                        n_k_tiles, n_k_tiles_div, dequant_ಕೆಲಸಗಾರ_fn, n_ಎಳೆಗಳು);

                    ಆದರೆ (i + 2 < n_chunk_cnt) {
                        ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ_p2 = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_p2, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_p2;
                        dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[(i + 2) % 4], ತೂಕ + nc_p2 * ತೂಕ_stride),
                                       dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ_p2);
                    }

                    hmx_matmul_ಕೆಲಸ_init(&ಕೆಲಸ_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_ತೂಕ_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು),
                                        hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ), k / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
                    hmx_ಸರತಿ_ತಳ್ಳು(ctx->hmx_ಸರತಿ, hmx_ಸರತಿ_make_desc(hmx_matmul_ಕೆಲಸಗಾರ_fn, &ಕೆಲಸ_slots[i % 2]));

                    ಆದರೆ (i > 0) {
                        hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);
                        ಸ್ಥಿರಾಂಕ size_t nc_prev = (i - 1) * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                        ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_prev = hex_smin(n - nc_prev, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                        ದಶಮಾಂಶ *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        ಪೂರ್ಣಾಂಕ chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ - (ಪೂರ್ಣಾಂಕ)nc_prev;
                        ಆದರೆ (chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, ಖಾಲಿ, vtcm_output_bufs[(i - 1) % 2], n_ಅಡ್ಡಸಾಲುಗಳು, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_prev, dst_stride, 0, chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_ಎಳೆಗಳು);
                        }
                    }
                }

                hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);
                ಸ್ಥಿರಾಂಕ size_t nc_ಕೊನೆಯ = (n_chunk_cnt - 1) * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಕೊನೆಯ = hex_smin(n - nc_ಕೊನೆಯ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                ದಶಮಾಂಶ *output_chunk = dst_ptr + (mr * dst_stride + nc_ಕೊನೆಯ);
                ಪೂರ್ಣಾಂಕ chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ - (ಪೂರ್ಣಾಂಕ)nc_ಕೊನೆಯ;
                ಆದರೆ (chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, ಖಾಲಿ, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_ಅಡ್ಡಸಾಲುಗಳು, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಕೊನೆಯ, dst_stride, 0, chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_ಎಳೆಗಳು);
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        hmx_matmul_ಕೆಲಸ_t ಕೆಲಸ;
        ಶೂನ್ಯ *vtcm_raw_bufs[2] = { vtcm_ತೂಕ_raw[0], dequant_ಕೆಲಸಗಾರ_fn ? vtcm_ತೂಕ_raw[0] : vtcm_scratch0 };
        ಶೂನ್ಯ *vtcm_ತೂಕ_bufs[2] = { dequant_ಕೆಲಸಗಾರ_fn ? vtcm_scratch0 : vtcm_ತೂಕ_raw[0], vtcm_scratch0 };

        ಗಾಗಿ (size_t mr = (size_t) m_ಆರಂಭ; mr < mr_ಕೊನೆ; mr += m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು) {
            ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು = hex_smin(mr_ಕೊನೆ - mr, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು);

            ರಚನೆ activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ಗಾತ್ರ,
                .n_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ) n_ಅಡ್ಡಸಾಲುಗಳು,
                .k_ಖಂಡ = k,
                .k_stride = act_stride,
                .n_ಎಳೆಗಳು = act_ಎಳೆಗಳು,
                .act_ಎಳೆಗಳು_div = act_ಎಳೆಗಳು_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_ಬೈಟುಗಳು = L.act_f32_ಬೈಟುಗಳು,
                .act_elem_ಗಾತ್ರ = act_elem_ಗಾತ್ರ,
            };
            transfer_activation_chunk_threaded(&act_params);

            ಗಾಗಿ (uint32_t p = 0; p < n_weights; p++) {
                ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src_w = octx->src[p];
                ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst   = octx->dsts[p];
                ಆದರೆ (!src_w || !dst) ಮುಂದುವರಿ;

                ಸ್ಥಿರಾಂಕ dma_addr_t ತೂಕ       = src_w->ದತ್ತಾಂಶ;
                dma_ಸರತಿ * ತೂಕ_dma      = octx->ctx->dma[0];
                ದಶಮಾಂಶ * dst_ptr            = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
                ಸ್ಥಿರಾಂಕ size_t n             = src_w->ne[1];
                ಆದರೆ (n == 0) ಮುಂದುವರಿ;
                ಸ್ಥಿರಾಂಕ size_t ತೂಕ_stride = src_w->nb[1];
                ಸ್ಥಿರಾಂಕ size_t dst_stride    = dst->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ         = (ಪೂರ್ಣಾಂಕ) dst->ne[0];
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                ಸ್ಥಿರಾಂಕ uint32_t dma_src_stride = is_tiled ? tile_ಗಾತ್ರ : ತೂಕ_stride;

                ಆದರೆ (n > 0) {
                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = hex_smin(n, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                    dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[0], ತೂಕ), dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ);
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_chunk_cnt; ++i) {
                    ಸ್ಥಿರಾಂಕ size_t nc = (size_t) i * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = hex_smin(n - nc, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲು_tiles = hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_tiles = hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                    ಶೂನ್ಯ * curr_raw = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(ತೂಕ_dma).dst;

                    dequantize_tiled_ತೂಕ_chunk_ಗಮ್ಯ_fp16_tiles(
                        ctx, vtcm_ತೂಕ_bufs[i % 2], curr_raw,
                        n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, k, ಅಡ್ಡಸಾಲು_stride, ತೂಕ_type,
                        n_k_tiles, n_k_tiles_div, dequant_ಕೆಲಸಗಾರ_fn, n_ಎಳೆಗಳು);

                    ಸ್ಥಿರಾಂಕ size_t nc_ಮುಂದಿನ = nc + n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                    ಆದರೆ (nc_ಮುಂದಿನ < n) {
                        ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ = hex_smin(n - nc_ಮುಂದಿನ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                        ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ_ಮುಂದಿನ = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ;
                        dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[(i + 1) % 2], ತೂಕ + nc_ಮುಂದಿನ * ತೂಕ_stride), dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ_ಮುಂದಿನ);
                    }

                    hmx_matmul_ಕೆಲಸ_init(&ಕೆಲಸ, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_ತೂಕ_bufs[i % 2], vtcm_scales, n_ಅಡ್ಡಸಾಲು_tiles, n_ಲಂಬಸಾಲು_tiles, k / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
                    hmx_ಸರತಿ_ತಳ್ಳು(ctx->hmx_ಸರತಿ, hmx_ಸರತಿ_make_desc(hmx_matmul_ಕೆಲಸಗಾರ_fn, &ಕೆಲಸ));
                    hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);

                    ದಶಮಾಂಶ *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    ಪೂರ್ಣಾಂಕ chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ - (ಪೂರ್ಣಾಂಕ)nc;
                    ಆದರೆ (chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, ಖಾಲಿ, vtcm_output, n_ಅಡ್ಡಸಾಲುಗಳು, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, dst_stride, 0, chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_ಎಳೆಗಳು);
                    }
                }
            }
        }
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_addr_t hmx_mm_ತೂಕ_batch_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ hmx_mm_batched_params_t *params,
                                                         ಪೂರ್ಣಾಂಕ dst_b2, ಪೂರ್ಣಾಂಕ dst_b3) {
    ಸ್ಥಿರಾಂಕ size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    ಸ್ಥಿರಾಂಕ size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    ಹಿಂತಿರುಗಿ params->ತೂಕ + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_addr_t hmx_mm_act_batch_addr(ಸ್ಥಿರಾಂಕ hmx_mm_batched_params_t *params,
                                                ಪೂರ್ಣಾಂಕ dst_b2, ಪೂರ್ಣಾಂಕ dst_b3) {
    ಹಿಂತಿರುಗಿ params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ *hmx_mm_dst_batch_ptr(ಸ್ಥಿರಾಂಕ hmx_mm_batched_params_t *params,
                                              ಪೂರ್ಣಾಂಕ dst_b2, ಪೂರ್ಣಾಂಕ dst_b3) {
    ಹಿಂತಿರುಗಿ (ದಶಮಾಂಶ *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hmx_mm_batched_simple(ರಚನೆ htp_context *ctx,
                                 ಸ್ಥಿರಾಂಕ hmx_mm_batched_params_t *params,
                                 ಪೂರ್ಣಾಂಕ m_chunk, ಪೂರ್ಣಾಂಕ n_chunk, ಪೂರ್ಣಾಂಕ pipeline, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು, ಪೂರ್ಣಾಂಕ act_ಎಳೆಗಳು, ಪೂರ್ಣಾಂಕ vtcm_ಗಾತ್ರ,
                                 ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * act_ಎಳೆಗಳು_div, ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * k_div) {
    ಪೂರ್ಣಾಂಕ ret = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->ತೂಕ_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_ಬೈಟುಗಳು,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_ತೂಕ_batch_ದತ್ತಾಂಶ(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_ಗಾತ್ರ, params->ತೂಕ_stride,
                                params->ತೂಕ_type, params->k, params->dst_stride, params->bias_stride, params->dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
                                m_chunk, n_chunk, pipeline, n_ಎಳೆಗಳು, act_ಎಳೆಗಳು,
                                act_ಎಳೆಗಳು_div, k_div, params->tile_ಗಾತ್ರ, params->aligned_tile_ಗಾತ್ರ, vtcm_ಗಾತ್ರ);
        }
    }
    ಹಿಂತಿರುಗಿ ret;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hmx_mm_batched(ರಚನೆ htp_context *ctx, ಸ್ಥಿರಾಂಕ hmx_mm_batched_params_t *params,
                          ಪೂರ್ಣಾಂಕ m_chunk, ಪೂರ್ಣಾಂಕ n_chunk, ಪೂರ್ಣಾಂಕ pipeline, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು, ಪೂರ್ಣಾಂಕ act_ಎಳೆಗಳು,
                          ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * act_ಎಳೆಗಳು_div,
                          ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * k_div,
                          ಪೂರ್ಣಾಂಕ vtcm_ಗಾತ್ರ) {
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(params->ತೂಕ_type, params->k);
    ಆದರೆ (ಅಡ್ಡಸಾಲು_stride == 0) { ಹಿಂತಿರುಗಿ -1; }
    ಆದರೆ (params->act_stride < params->k || (params->tile_ಗಾತ್ರ == 0 && (size_t) params->ತೂಕ_stride < ಅಡ್ಡಸಾಲು_stride) || params->dst_stride < params->dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) { ಹಿಂತಿರುಗಿ -1; }
    ಆದರೆ (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { ಹಿಂತಿರುಗಿ -1; }
    ಆದರೆ (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { ಹಿಂತಿರುಗಿ -1; }
    // N (the weight row count) does not have to be 32-aligned:
    ಆದರೆ (params->k % 32 != 0) { ಹಿಂತಿರುಗಿ -1; }
    ಆದರೆ (!hex_is_aligned(params->dst, ಗಾತ್ರ(ದಶಮಾಂಶ))) { ಹಿಂತಿರುಗಿ -1; }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗುಂಪು_ಗಾತ್ರ = params->r2;
    ಸ್ಥಿರಾಂಕ size_t vtcm_budget  = ctx->vtcm_ಗಾತ್ರ;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool ಸತತ_grouped = (params->ತೂಕ_type == HTP_TYPE_F16 && params->tile_ಗಾತ್ರ == 0 && ಗುಂಪು_ಗಾತ್ರ > 1 && (size_t)vtcm_ಗಾತ್ರ <= vtcm_budget);
    ಆದರೆ (!ಸತತ_grouped) {
        ಹಿಂತಿರುಗಿ hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_ಎಳೆಗಳು, act_ಎಳೆಗಳು, vtcm_ಗಾತ್ರ, act_ಎಳೆಗಳು_div, k_div);
    }

    ರಚನೆ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು = m_chunk;
    size_t n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = n_chunk;

    ರಚನೆ htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, ಗುಂಪು_ಗಾತ್ರ, false, act_ಎಳೆಗಳು, 0, params->bias_ಬೈಟುಗಳು);

    ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು > vtcm_budget) {
        FARF(ಮೇಲಿನ_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        ಹಿಂತಿರುಗಿ hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_ಎಳೆಗಳು, act_ಎಳೆಗಳು, vtcm_ಗಾತ್ರ, act_ಎಳೆಗಳು_div, k_div);
    }

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = (uint8_t *) ctx->vtcm_ತಳ;
    __fp16  *vtcm_ತೂಕ     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_ತೂಕ[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_dst[0]);
    ಶೂನ್ಯ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(ಶೂನ್ಯ, ತಳ, L.off_scratch[0]);
    ಶೂನ್ಯ    *vtcm_scratch1   = VTCM_LAYOUT_PTR(ಶೂನ್ಯ, ತಳ, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_scales);
    ದಶಮಾಂಶ   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ದಶಮಾಂಶ, ತಳ, L.off_act_f32);

    ಸ್ಥಿರಾಂಕ bool has_bias      = (params->bias_ಬೈಟುಗಳು > 0 && params->bias_addr != 0);
    ದಶಮಾಂಶ   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(ದಶಮಾಂಶ, ತಳ, L.off_bias, has_bias);
    ಆದರೆ (has_bias) {
        dma_ಸರತಿ_ತಳ್ಳು(params->ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_bias, params->bias_addr), hex_align_ಮೇಲೆ(params->bias_ಬೈಟುಗಳು, 128), 0, params->bias_ಬೈಟುಗಳು, 1);
        dma_ಸರತಿ_ತೆಗೆ(params->ತೂಕ_dma);
    }

    hmx_init_ಸ್ತಂಭ_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(ಮೇಲಿನ_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, ಗುಂಪು_ಗಾತ್ರ, params->ne13,
            m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
            L.ಒಟ್ಟು_ಬೈಟುಗಳು, vtcm_budget);

    ಸ್ಥಿರಾಂಕ size_t fp16_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು   = (size_t) params->k * ಗಾತ್ರ(__fp16);
    ಸ್ಥಿರಾಂಕ size_t ತೂಕ_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = (size_t) params->ತೂಕ_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_ಕೆಲಸ_t ಕೆಲಸ;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b3 = 0; b3 < params->ne13; ++b3) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b2_ತಳ = 0; b2_ತಳ < params->ne12; b2_ತಳ += ಗುಂಪು_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ dma_addr_t ತೂಕ_ಗುಂಪು = hmx_mm_ತೂಕ_batch_ದತ್ತಾಂಶ(params, b2_ತಳ, b3);
            dma_ಸರತಿ * ತೂಕ_dma = params->ತೂಕ_dma;

            ಗಾಗಿ (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು) {
                ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು = hex_smin((size_t) params->m - mr, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು);
                ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲು_tiles = hmx_ceil_div((ಪೂರ್ಣಾಂಕ) n_ಅಡ್ಡಸಾಲುಗಳು, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < ಗುಂಪು_ಗಾತ್ರ; ++g) {
                    ಸ್ಥಿರಾಂಕ dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_ತಳ + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_ಗಾತ್ರ;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_ತಲೆ_stride;
                    ರಚನೆ activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_ಅಡ್ಡಸಾಲುಗಳು = (ಪೂರ್ಣಾಂಕ) n_ಅಡ್ಡಸಾಲುಗಳು,
                        .k_ಖಂಡ = params->k,
                        .k_stride = params->act_stride,
                        .n_ಎಳೆಗಳು = act_ಎಳೆಗಳು,
                        .act_ಎಳೆಗಳು_div = act_ಎಳೆಗಳು_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_ಬೈಟುಗಳು = L.act_f32_ಬೈಟುಗಳು,
                        .act_elem_ಗಾತ್ರ = params->act_elem_ಗಾತ್ರ,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮೊದಲ = hex_smin((size_t) params->n, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_scratch0, ತೂಕ_ಗುಂಪು),
                                      fp16_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ತೂಕ_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, fp16_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮೊದಲ);
                }
                ಆದರೆ (n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ < (size_t) params->n) {
                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಎರಡನೇ = hex_smin((size_t) params->n - n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    ಸ್ಥಿರಾಂಕ dma_addr_t ಎರಡನೇ_ತೂಕ_chunk = ತೂಕ_ಗುಂಪು + n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ತೂಕ_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
                    dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_scratch1, ಎರಡನೇ_ತೂಕ_chunk),
                                      fp16_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ತೂಕ_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, fp16_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಎರಡನೇ);
                }

                ಗಾಗಿ (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) {
                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ      = hex_smin((size_t) params->n - nc, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_tiles = hmx_ceil_div((ಪೂರ್ಣಾಂಕ) n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                    {
                        ಶೂನ್ಯ * curr_raw = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(ತೂಕ_dma).dst;

                        ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_tiled = hex_align_ಮೇಲೆ(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                        hmx_interleave_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_tiles(vtcm_ತೂಕ, (ಸ್ಥಿರಾಂಕ __fp16 *) curr_raw, (uint32_t) n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, params->k, params->k, 0, (uint32_t) n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_tiled);

                        ಸ್ಥಿರಾಂಕ size_t nc_ಮುಂದಿನ = nc + n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * 2;
                        ಆದರೆ (nc_ಮುಂದಿನ < (size_t) params->n) {
                            ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ = hex_smin((size_t) params->n - nc_ಮುಂದಿನ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                            ಸ್ಥಿರಾಂಕ dma_addr_t ಮುಂದಿನ_ತೂಕ_chunk = ತೂಕ_ಗುಂಪು + nc_ಮುಂದಿನ * ತೂಕ_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;

                            dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(curr_raw, ಮುಂದಿನ_ತೂಕ_chunk),
                                              fp16_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ತೂಕ_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, fp16_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < ಗುಂಪು_ಗಾತ್ರ; ++g) {
                        {
                            ಸ್ಥಿರಾಂಕ __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_ತಲೆ_stride;
                            hmx_matmul_ಕೆಲಸ_init(&ಕೆಲಸ, vtcm_output, vtcm_act_g, vtcm_ತೂಕ, vtcm_scales, n_ಅಡ್ಡಸಾಲು_tiles, n_ಲಂಬಸಾಲು_tiles, params->k / 32);
                            hmx_ಸರತಿ_ತಳ್ಳು(ctx->hmx_ಸರತಿ, hmx_ಸರತಿ_make_desc(hmx_matmul_ಕೆಲಸಗಾರ_fn, &ಕೆಲಸ));
                            hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);
                        }

                        {
                            ದಶಮಾಂಶ *output = hmx_mm_dst_batch_ptr(params, b2_ತಳ + g, b3) + mr * params->dst_stride + nc;
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : ಖಾಲಿ;
                            ಪೂರ್ಣಾಂಕ chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = params->dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ - (ಪೂರ್ಣಾಂಕ)nc;
                            ಆದರೆ (chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (ಪೂರ್ಣಾಂಕ) n_ಅಡ್ಡಸಾಲುಗಳು, (ಪೂರ್ಣಾಂಕ) n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
                                                               params->dst_stride, params->bias_stride, chunk_dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n_ಎಳೆಗಳು);
                            }
                        }
                    }
                }
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_activation_chunk_gathered_threaded(
            ರಚನೆ htp_context *ctx,
            __fp16 *dst,
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src,
            ಪೂರ್ಣಾಂಕ ಆರಂಭ_ಅಡ್ಡಸಾಲು,
            ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು,
            ಪೂರ್ಣಾಂಕ k_ಖಂಡ,
            ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping *ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
            ಪೂರ್ಣಾಂಕ cur_a,
            ಪೂರ್ಣಾಂಕ mapping_stride,
            ಪೂರ್ಣಾಂಕ ne11,
            size_t nb11,
            size_t nb12,
            ಪೂರ್ಣಾಂಕ cne1,
            ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು,
            ಪೂರ್ಣಾಂಕ k_valid) {
    ಆದರೆ (n_ಅಡ್ಡಸಾಲುಗಳು <= 0) ಹಿಂತಿರುಗಿ;
    ಪೂರ್ಣಾಂಕ chunks_ಪ್ರತಿ_thread = hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, n_ಎಳೆಗಳು);
    chunks_ಪ್ರತಿ_thread = hex_align_ಮೇಲೆ(chunks_ಪ್ರತಿ_thread, 2);

    ಪೂರ್ಣಾಂಕ actual_ಎಳೆಗಳು = hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, chunks_ಪ್ರತಿ_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_ಎಳೆಗಳು,
        .n_tot_chunks      = n_ಅಡ್ಡಸಾಲುಗಳು,
        .n_chunks_ಪ್ರತಿ_task = chunks_ಪ್ರತಿ_thread,
        .k_ಖಂಡ           = k_ಖಂಡ,
        .ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು       = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_ಮೌಲ್ಯಗಳು(ne11) : (ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .ಆರಂಭ_ಅಡ್ಡಸಾಲು         = ಆರಂಭ_ಅಡ್ಡಸಾಲು,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    ಕೆಲಸಗಾರ_callback_t ಕೆಲಸಗಾರ_fn = ne11 == 1 ? transfer_activation_chunk_gathered_ಕೆಲಸಗಾರ_flat_fn :
                                              transfer_activation_chunk_gathered_ಕೆಲಸಗಾರ_fn;

    ಆದರೆ (actual_ಎಳೆಗಳು <= 1) {
        ಕೆಲಸಗಾರ_fn(1, 0, &state);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಕೆಲಸಗಾರ_pool_ಸತತ_func(ctx->ಕೆಲಸಗಾರ_pool, ಕೆಲಸಗಾರ_fn, &state, actual_ಎಳೆಗಳು);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ transfer_output_chunk_scattered_threaded(
            ರಚನೆ htp_context *ctx,
            ದಶಮಾಂಶ *dst,
            ಸ್ಥಿರಾಂಕ __fp16 *vtcm_src,
            ಪೂರ್ಣಾಂಕ ಆರಂಭ_ಅಡ್ಡಸಾಲು,
            ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು,
            ಪೂರ್ಣಾಂಕ n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
            ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping *ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
            ಪೂರ್ಣಾಂಕ cur_a,
            ಪೂರ್ಣಾಂಕ mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            ಪೂರ್ಣಾಂಕ cne1,
            ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    ಆದರೆ (n_ಅಡ್ಡಸಾಲುಗಳು <= 0) ಹಿಂತಿರುಗಿ;
    ಪೂರ್ಣಾಂಕ chunks_ಪ್ರತಿ_thread = hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, n_ಎಳೆಗಳು);
    chunks_ಪ್ರತಿ_thread = hex_align_ಮೇಲೆ(chunks_ಪ್ರತಿ_thread, 2);

    ಪೂರ್ಣಾಂಕ actual_ಎಳೆಗಳು = hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, chunks_ಪ್ರತಿ_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_ಎಳೆಗಳು,
        .n_tot_chunks      = n_ಅಡ್ಡಸಾಲುಗಳು,
        .n_chunks_ಪ್ರತಿ_task = chunks_ಪ್ರತಿ_thread,
        .n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ            = n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
        .ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು       = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .ಆರಂಭ_ಅಡ್ಡಸಾಲು         = ಆರಂಭ_ಅಡ್ಡಸಾಲು,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    ಆದರೆ (actual_ಎಳೆಗಳು <= 1) {
        transfer_output_chunk_scattered_ಕೆಲಸಗಾರ_fn(1, 0, &state);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಕೆಲಸಗಾರ_pool_ಸತತ_func(ctx->ಕೆಲಸಗಾರ_pool, transfer_output_chunk_scattered_ಕೆಲಸಗಾರ_fn, &state, actual_ಎಳೆಗಳು);
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hmx_mm_id_2d_f32(ರಚನೆ htp_context *ctx,
                                         dma_ಸರತಿ *ತೂಕ_dma,
                                         ದಶಮಾಂಶ *ಸೀಮಿತ dst,
                                         ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *activation,
                                         dma_addr_t ತೂಕ,
                                         ಪೂರ್ಣಾಂಕ m, ಪೂರ್ಣಾಂಕ k, ಪೂರ್ಣಾಂಕ n,
                                         ಪೂರ್ಣಾಂಕ k_valid,
                                         ಪೂರ್ಣಾಂಕ ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         ಪೂರ್ಣಾಂಕ ತೂಕ_stride,
                                         ಪೂರ್ಣಾಂಕ ತೂಕ_type,
                                         ಪೂರ್ಣಾಂಕ tile_ಗಾತ್ರ,
                                         ಪೂರ್ಣಾಂಕ aligned_tile_ಗಾತ್ರ,
                                         ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping *ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
                                         ಪೂರ್ಣಾಂಕ cur_a,
                                         ಪೂರ್ಣಾಂಕ mapping_stride,
                                         ಪೂರ್ಣಾಂಕ m_ಆರಂಭ,
                                         ಪೂರ್ಣಾಂಕ m_ಕೊನೆ,
                                         ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    ರಚನೆ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cne1 = m;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m_core = m_ಕೊನೆ - m_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m_core_padded = hex_align_ಮೇಲೆ(m_core > 0 ? m_core : 1, 32);

    ಆದರೆ (k % 32 != 0 || n % 32 != 0) { ಹಿಂತಿರುಗಿ -1; }
    ಆದರೆ (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { ಹಿಂತಿರುಗಿ -1; }

    size_t ಅಡ್ಡಸಾಲು_stride = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ತೂಕ_type, k);
    ಆದರೆ (ಅಡ್ಡಸಾಲು_stride == 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_k_tiles = k / HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    ಸ್ಥಿರಾಂಕ uint32_t n_ತೂಕ_tiles = k / htp_tensor_get_tile_k(ತೂಕ_type);
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು n_k_tiles_div = init_fastdiv_ಮೌಲ್ಯಗಳು(n_k_tiles);

    ಸ್ಥಿರಾಂಕ size_t vec_dot_ಗಾತ್ರ = k * ಗಾತ್ರ(__fp16);
    ಸ್ಥಿರಾಂಕ size_t vtcm_budget  = ctx->vtcm_ಗಾತ್ರ;
    size_t vtcm_ಬಳಸಿದ = 0;

    ಸ್ಥಿರಾಂಕ bool is_tiled = (tile_ಗಾತ್ರ > 0);
    tiled_dequant_task_fn_t dequant_ಕೆಲಸಗಾರ_fn = hmx_get_dequant_ಕೆಲಸಗಾರ_fn(ತೂಕ_type, is_tiled);

    ಸ್ಥಿರಾಂಕ uint32_t dma_dst_stride  = is_tiled ? aligned_tile_ಗಾತ್ರ : ಅಡ್ಡಸಾಲು_stride;
    ಸ್ಥಿರಾಂಕ uint32_t dma_src_stride  = is_tiled ? tile_ಗಾತ್ರ : ತೂಕ_stride;
    ಸ್ಥಿರಾಂಕ uint32_t dma_ಅಗಲ_ಬೈಟುಗಳು = is_tiled ? tile_ಗಾತ್ರ : ಅಡ್ಡಸಾಲು_stride;

    ಸ್ಥಿರಾಂಕ size_t qweight_ಅಡ್ಡಸಾಲು_stride = is_tiled ? (size_t)(n_ತೂಕ_tiles * aligned_tile_ಗಾತ್ರ) / 32 : 0;
    ಸ್ಥಿರಾಂಕ size_t ತೂಕ_ಅಡ್ಡಸಾಲು_stride = is_tiled ? qweight_ಅಡ್ಡಸಾಲು_stride : ಅಡ್ಡಸಾಲು_stride;

    size_t ಗಾತ್ರ_ಪ್ರತಿ_n = 0, ಗಾತ್ರ_ಪ್ರತಿ_m = 0, ಗಾತ್ರ_ಪ್ರತಿ_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(ತೂಕ_type, k, /*pipeline=*/false, aligned_tile_ಗಾತ್ರ,
                                  &ಗಾತ್ರ_ಪ್ರತಿ_n, &ಗಾತ್ರ_ಪ್ರತಿ_m, &ಗಾತ್ರ_ಪ್ರತಿ_mn);

    ಸ್ಥಿರಾಂಕ size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು = 0, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = 0;
    ಆದರೆ (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, ಗಾತ್ರ_ಪ್ರತಿ_n, ಗಾತ್ರ_ಪ್ರತಿ_m, ಗಾತ್ರ_ಪ್ರತಿ_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_ವೆಚ್ಚ_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_ವೆಚ್ಚ_A_CONVERT, &m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು, &n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, &vtcm_ಬಳಸಿದ)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ size_t ತೂಕ_area_ಗಾತ್ರ = hex_align_ಮೇಲೆ(n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ತೂಕ_ಅಡ್ಡಸಾಲು_stride, HTP_MM_HMX_TILE_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ size_t act_area_ಗಾತ್ರ    = hex_align_ಮೇಲೆ(m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು * vec_dot_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ size_t output_area_ಗಾತ್ರ = hex_align_ಮೇಲೆ(m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ಗಾತ್ರ(__fp16), HTP_MM_HMX_TILE_ಗಾತ್ರ);

    size_t scratch0_ಗಾತ್ರ = hex_align_ಮೇಲೆ(n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * vec_dot_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_ತಳ;
    __fp16  *vtcm_ತೂಕ   = ತೂಕ_area_ಗಾತ್ರ ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, ತೂಕ_area_ಗಾತ್ರ) : ಖಾಲಿ;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_ಗಾತ್ರ);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_ಗಾತ್ರ);
    ಶೂನ್ಯ    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_ಗಾತ್ರ);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_ಬಳಸಿದ = vtcm_ptr - (uint8_t *) ctx->vtcm_ತಳ;
    ಆದರೆ (vtcm_ಬಳಸಿದ > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_ಬಳಸಿದ, vtcm_budget);
        ಹಿಂತಿರುಗಿ -1;
    }

    hmx_init_ಸ್ತಂಭ_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_ಕೆಲಸ_t ಕೆಲಸ;
    ಶೂನ್ಯ *vtcm_raw_bufs[2] = { vtcm_ತೂಕ, dequant_ಕೆಲಸಗಾರ_fn ? vtcm_ತೂಕ : vtcm_scratch0 };
    ಶೂನ್ಯ *vtcm_ತೂಕ_bufs[2] = { dequant_ಕೆಲಸಗಾರ_fn ? vtcm_scratch0 : vtcm_ತೂಕ, vtcm_scratch0 };
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

    ಗಾಗಿ (size_t mr = (size_t) m_ಆರಂಭ; mr < (size_t) m_ಕೊನೆ; mr += m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು) {
        ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು = hex_smin((size_t) m_ಕೊನೆ - mr, m_chunk_n_ಅಡ್ಡಸಾಲುಗಳು);
        ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲು_tiles = hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (ಪೂರ್ಣಾಂಕ) mr, (ಪೂರ್ಣಾಂಕ) n_ಅಡ್ಡಸಾಲುಗಳು, k,
            ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_ಎಳೆಗಳು, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        ಆದರೆ (n > 0) {
            ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = hex_smin((size_t) n, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
            ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[0], ತೂಕ),
                           dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ);
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_chunk_cnt; ++i) {
            ಸ್ಥಿರಾಂಕ size_t nc = (size_t) i * n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = hex_smin((size_t) n - nc, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
            ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_tiles = hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

            // A: Wait for weight DMA
            ಶೂನ್ಯ * curr_raw = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(ತೂಕ_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_ತೂಕ_chunk_ಗಮ್ಯ_fp16_tiles(
                ctx, vtcm_ತೂಕ_bufs[i % 2], curr_raw,
                n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, k, ಅಡ್ಡಸಾಲು_stride, ತೂಕ_type,
                n_k_tiles, n_k_tiles_div, dequant_ಕೆಲಸಗಾರ_fn, n_ಎಳೆಗಳು
            );

            // Start weight DMA for the next chunk early
            ಸ್ಥಿರಾಂಕ size_t nc_ಮುಂದಿನ = nc + n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            ಆದರೆ (nc_ಮುಂದಿನ < (size_t) n) {
                ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ = hex_smin((size_t) n - nc_ಮುಂದಿನ, n_chunk_n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                ಸ್ಥಿರಾಂಕ uint32_t ಎತ್ತರ_ಮುಂದಿನ = is_tiled ? hmx_ceil_div(n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ, 32) * n_ತೂಕ_tiles : n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಮುಂದಿನ;
                dma_ಸರತಿ_ತಳ್ಳು(ತೂಕ_dma, dma_make_ದತ್ತಾಂಶ(vtcm_raw_bufs[(i + 1) % 2], ತೂಕ + nc_ಮುಂದಿನ * ತೂಕ_stride),
                               dma_dst_stride, dma_src_stride, dma_ಅಗಲ_ಬೈಟುಗಳು, ಎತ್ತರ_ಮುಂದಿನ);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_ಕೆಲಸ_init(&ಕೆಲಸ, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_ತೂಕ_bufs[i % 2], vtcm_scales, n_ಅಡ್ಡಸಾಲು_tiles, n_ಲಂಬಸಾಲು_tiles, k / HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
            hmx_ಸರತಿ_ತಳ್ಳು(ctx->hmx_ಸರತಿ, hmx_ಸರತಿ_make_desc(hmx_matmul_ಕೆಲಸಗಾರ_fn, &ಕೆಲಸ));
            hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (ಪೂರ್ಣಾಂಕ) mr, (ಪೂರ್ಣಾಂಕ) n_ಅಡ್ಡಸಾಲುಗಳು, (ಪೂರ್ಣಾಂಕ) n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
                ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_ಎಳೆಗಳು);
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}

// --- Dispatchers and Public Entry Points ---

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hmx_mm_op_matmul(ರಚನೆ htp_ops_context * octx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    ಪೂರ್ಣಾಂಕ k = (ಪೂರ್ಣಾಂಕ) src0->ne[0];
    ಪೂರ್ಣಾಂಕ n = (ಪೂರ್ಣಾಂಕ) src0->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m_ಒಟ್ಟು    = (ಪೂರ್ಣಾಂಕ) act->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t act_elem_ಗಾತ್ರ = (act->type == HTP_TYPE_F16) ? ಗಾತ್ರ(__fp16) : ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ act_stride = (ಪೂರ್ಣಾಂಕ) (act->nb[1] / act_elem_ಗಾತ್ರ);

    ಪೂರ್ಣಾಂಕ m_ಆರಂಭ = 0;
    ಪೂರ್ಣಾಂಕ m_ಅಡ್ಡಸಾಲುಗಳು  = m_ಒಟ್ಟು;
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        bool can_ವಿಭಜಿಸು = htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, ಗಾತ್ರ(ದಶಮಾಂಶ)) &&
                         htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(act, act_elem_ಗಾತ್ರ);
        ಆದರೆ (src2 && src2->ne[1] > 1 && !htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(src2, ಗಾತ್ರ(ದಶಮಾಂಶ))) {
            can_ವಿಭಜಿಸು = false;
        }
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ((uint32_t) m_ಒಟ್ಟು, can_ವಿಭಜಿಸು ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        m_ಆರಂಭ = (ಪೂರ್ಣಾಂಕ) range.ಆರಂಭ;
        m_ಅಡ್ಡಸಾಲುಗಳು  = (ಪೂರ್ಣಾಂಕ) range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (m_ಅಡ್ಡಸಾಲುಗಳು == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_ಬೈಟುಗಳು = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    ಆದರೆ (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ));
        src2_addr = src2->ದತ್ತಾಂಶ + m_ಆರಂಭ * src2_stride * ಗಾತ್ರ(ದಶಮಾಂಶ);
        src2_ಬೈಟುಗಳು = (size_t) kparams->vtcm_bias_ಗಾತ್ರ;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dst_stride = (ಪೂರ್ಣಾಂಕ)(dst->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ       * dst_ptr = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ + m_ಆರಂಭ * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ಸ್ಥಿರಾಂಕ dma_addr_t act_addr = act->ದತ್ತಾಂಶ + (size_t) m_ಆರಂಭ * act_stride * act_elem_ಗಾತ್ರ;

    ಪೂರ್ಣಾಂಕ ret = -1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು = kparams->n_ಎಳೆಗಳು;
    ಆದರೆ (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        hmx_mm_batched_params_t batch_params = {
            .dst               = dst_ptr,
            .bias_addr         = src2_addr,
            .bias_ಬೈಟುಗಳು        = src2_ಬೈಟುಗಳು,
            .act_dma_addr      = act_addr,
            .ತೂಕ            = src0->ದತ್ತಾಂಶ,
            .ತೂಕ_dma        = octx->ctx->dma[0],
            .m                 = m_ಅಡ್ಡಸಾಲುಗಳು,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_ಗಾತ್ರ     = act_elem_ಗಾತ್ರ,
            .ತೂಕ_stride     = (ಪೂರ್ಣಾಂಕ) src0->nb[1],
            .ತೂಕ_type       = (ಪೂರ್ಣಾಂಕ) src0->type,
            .dst_stride        = dst_stride,
            .dst_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ          = (ಪೂರ್ಣಾಂಕ) dst->ne[0],
            .bias_stride       = src2_stride,
            .tile_ಗಾತ್ರ         = kparams->tile_ಗಾತ್ರ,
            .aligned_tile_ಗಾತ್ರ = kparams->aligned_tile_ಗಾತ್ರ,
            .ne02              = ne02,
            .ne03              = ne03,
            .ne12              = ne12,
            .ne13              = ne13,
            .src0_nb2          = src0->nb[2],
            .src0_nb3          = src0->nb[3],
            .act_nb2           = act->nb[2],
            .act_nb3           = act->nb[3],
            .dst_nb2           = dst->nb[2],
            .dst_nb3           = dst->nb[3],
            .bias_nb2          = src2_nb2,
            .bias_nb3          = src2_nb3,
            .r2                = (ne02 > 0) ? (ne12 / ne02) : 1,
            .r3                = (ne03 > 0) ? (ne13 / ne03) : 1,
            .div_r2            = kparams->div_r2,
            .div_r3            = kparams->div_r3,
        };
        ret = hmx_mm_batched(octx->ctx, &batch_params,
                             kparams->m_chunk, kparams->n_chunk,
                             kparams->pipeline, n_ಎಳೆಗಳು,
                             kparams->n_act_ಎಳೆಗಳು,
                             &kparams->div_n_act_ಎಳೆಗಳು,
                             &kparams->div_ne00_padded,
                             kparams->vtcm_ಗಾತ್ರ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_ಬೈಟುಗಳು,
            act_addr, src0->ದತ್ತಾಂಶ,
            m_ಅಡ್ಡಸಾಲುಗಳು, k, n, act_stride, act_elem_ಗಾತ್ರ, (ಪೂರ್ಣಾಂಕ) src0->nb[1], (ಪೂರ್ಣಾಂಕ) src0->type, (ಪೂರ್ಣಾಂಕ) act->ne[0],
            dst_stride, src2_stride, (ಪೂರ್ಣಾಂಕ)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_ಎಳೆಗಳು,
            kparams->n_act_ಎಳೆಗಳು,
            &kparams->div_n_act_ಎಳೆಗಳು,
            &kparams->div_ne00_padded,
            kparams->tile_ಗಾತ್ರ, kparams->aligned_tile_ಗಾತ್ರ, kparams->vtcm_ಗಾತ್ರ
        );
    }

    ಆದರೆ (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR;
    }
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_mm_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(ರಚನೆ htp_tensor * c, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ op_matmul_impl(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = htp_mm_init_context(octx, kparams);
    ಆದರೆ (ಸ್ಥಿತಿ_2 != HTP_ಸ್ಥಿತಿ_OK) {
        ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
    }

    ಆದರೆ (kparams->n_hmx) {
        ಹಿಂತಿರುಗಿ hmx_mm_op_matmul(octx, kparams);
    }

    ಹಿಂತಿರುಗಿ hvx_mm_matmul(octx);
}

ಪೂರ್ಣಾಂಕ op_matmul(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;

    ಆದರೆ (kparams->collapse) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * act  = octx->src[1];
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;
        ಸ್ಥಿರಾಂಕ uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ಸ್ಥಿರಾಂಕ uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        ರಚನೆ htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        ರಚನೆ htp_tensor src2_collapsed;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src2 = octx->src[2];
        ಆದರೆ (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ಸ್ಥಿರಾಂಕ uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        ಆದರೆ (src2) {
            octx->src[2] = src2;
        }
        ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
    }

    ಹಿಂತಿರುಗಿ op_matmul_impl(octx);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hmx_mm_op_matmul_id(
    ರಚನೆ htp_ops_context * octx,
    ರಚನೆ htp_mm_context * mmctx
) {
    ಸ್ಥಿರಾಂಕ uint32_t *                ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts = mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts;
    ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು       = mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು;
    htp_matmul_tensors_preamble;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ids = octx->src[2]->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_as  = ne02;

    ಸ್ಥಿರಾಂಕ bool mdev_ವಿಭಜಿಸು = (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) && htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1 && !mdev_ವಿಭಜಿಸು && octx->ctx->mdev.idx > 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }
    uint32_t n_ಸಕ್ರಿಯ = 0;
    ಆದರೆ (mdev_ವಿಭಜಿಸು) {
        ಗಾಗಿ (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            ಆದರೆ (ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[a] > 0) n_ಸಕ್ರಿಯ++;
        }
    }
    ಸ್ಥಿರಾಂಕ bool expert_ವಿಭಜಿಸು = mdev_ವಿಭಜಿಸು && (n_ಸಕ್ರಿಯ >= octx->ctx->mdev.ಎಣಿಕೆ_2);

    uint32_t ಗುರಿ_dev = 0;
    ಗಾಗಿ (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ಸ್ಥಿರಾಂಕ int32_t cne1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a];
        ಆದರೆ (cne1 == 0) ಮುಂದುವರಿ;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m_padded = hex_align_ಮೇಲೆ(cne1, 32);
        ಪೂರ್ಣಾಂಕ m_ಆರಂಭ = 0, m_ಕೊನೆ = m_padded;
        ಆದರೆ (expert_ವಿಭಜಿಸು) {
            ಸ್ಥಿರಾಂಕ bool my_expert = (ಗುರಿ_dev == octx->ctx->mdev.idx);
            ಆದರೆ (++ಗುರಿ_dev == octx->ctx->mdev.ಎಣಿಕೆ_2) {
                ಗುರಿ_dev = 0;
            }
            ಆದರೆ (!my_expert) ಮುಂದುವರಿ;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mdev_ವಿಭಜಿಸು) {
            ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = (uint32_t) cne1 >= octx->ctx->mdev.ಎಣಿಕೆ_2;
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ((uint32_t) m_padded, can_ವಿಭಜಿಸು ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            m_ಆರಂಭ = (ಪೂರ್ಣಾಂಕ) range.ಆರಂಭ;
            m_ಕೊನೆ   = (ಪೂರ್ಣಾಂಕ) (range.ಆರಂಭ + range.ಎಣಿಕೆ_2);
        }
        ಆದರೆ (m_ಆರಂಭ >= m_ಕೊನೆ) ಮುಂದುವರಿ;

        ಪೂರ್ಣಾಂಕ ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (ದಶಮಾಂಶ*) dst->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ*) act->ದತ್ತಾಂಶ,
                                   src0->ದತ್ತಾಂಶ + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (ಪೂರ್ಣಾಂಕ) src0->nb[1], (ಪೂರ್ಣಾಂಕ) src0->type,
                                   kparams->tile_ಗಾತ್ರ, kparams->aligned_tile_ಗಾತ್ರ,
                                   ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, cur_a, mmctx->mapping_stride,
                                   m_ಆರಂಭ, m_ಕೊನೆ, (ಪೂರ್ಣಾಂಕ) octx->n_ಎಳೆಗಳು);
        ಆದರೆ (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hvx_mm_matmul_id(
    ರಚನೆ htp_ops_context * octx,
    ರಚನೆ htp_mm_context * mmctx,
    work_ಸರತಿ_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded = mmctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded;
    ಸ್ಥಿರಾಂಕ uint32_t act_nrows            = mmctx->act_nrows;

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, 0);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ ids = octx->src[2];
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb01;

    ಸ್ಥಿರಾಂಕ uint32_t qk = QK_Q8_0_TILED;
    ಸ್ಥಿರಾಂಕ uint32_t nb = (ne10 + qk - 1) / qk;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_nb = act_nrows * nb;

    work_ಸರತಿ_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ಆದರೆ (act_nrows < octx->n_ಎಳೆಗಳು) {
        n_quant_tasks = MIN(ಒಟ್ಟು_nb, octx->n_ಎಳೆಗಳು);
        quant_task_func = htp_mm_act_quant_ಖಂಡ_func(src0->type, act->type);
        ಗಾಗಿ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_ಮೊದಲ = (ಒಟ್ಟು_nb * ith) / n_quant_tasks;
            uint32_t ib_ಕೊನೆಯ  = (ಒಟ್ಟು_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_ಮೊದಲ[ith] = ib_ಮೊದಲ;
            mmctx->quant_ib_ಕೊನೆಯ[ith]  = ib_ಕೊನೆಯ;
            mmctx->quant_r[ith]        = ib_ಮೊದಲ / nb;
            mmctx->quant_c[ith]        = ib_ಮೊದಲ % nb;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        n_quant_tasks = MIN(act_nrows, octx->n_ಎಳೆಗಳು);
        quant_task_func = htp_mm_act_quant_ಅಡ್ಡಸಾಲು_func(src0->type, act->type);
    }
    size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src0->type, ne10);

    ರಚನೆ htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_ಎಳೆಗಳು,
                                 0, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0, kparams->n_prefetch, true, false);

    ಸ್ಥಿರಾಂಕ size_t vtcm_ಗಾತ್ರ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;

    FARF(ಮೇಲಿನ_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_ಬೈಟುಗಳು, L.act_ಬೈಟುಗಳು, L.dst_ಬೈಟುಗಳು, vtcm_ಗಾತ್ರ);

    FARF(ಮೇಲಿನ_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ದತ್ತಾಂಶ,
         act->ದತ್ತಾಂಶ, dst->ದತ್ತಾಂಶ);

    // Make sure the reserved vtcm size is sufficient
    ಆದರೆ (octx->ctx->vtcm_ಗಾತ್ರ < vtcm_ಗಾತ್ರ) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_ಗಾತ್ರ, vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_src0);
    mmctx->vtcm_bias     = ಖಾಲಿ;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded;
    mmctx->vtcm_act_stride     = act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    mmctx->vtcm_act_raw_stride = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * ಗಾತ್ರ(ದಶಮಾಂಶ), QK_Q8_0_TILED * ಗಾತ್ರ(ದಶಮಾಂಶ));

    mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = fastdiv(L.src0_ಬೈಟುಗಳು, &octx->n_ಎಳೆಗಳು_div);
    mmctx->vtcm_act_ಗಾತ್ರ_ಪ್ರತಿ_thread  = L.act_ಬೈಟುಗಳು;
    mmctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = fastdiv(L.dst_ಬೈಟುಗಳು, &octx->n_ಎಳೆಗಳು_div);

    mmctx->cur_m_ಆರಂಭ = 0;
    mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, quant_task_func, mmctx, n_quant_tasks);

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, hvx_mmid_task_func, mmctx, octx->n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hmx_mm_op_matmul_id_nx(
    ರಚನೆ htp_ops_context * octx,
    ರಚನೆ htp_mm_context * mmctx
) {
    ಸ್ಥಿರಾಂಕ uint32_t *                ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts = mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts;
    ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು       = mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_weights = kparams->n_weights;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ act  = octx->src[n_weights];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_as = src0->ne[2];

    bool mdev_ವಿಭಜಿಸು = (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1);
    ಗಾಗಿ (uint32_t p = 0; p < n_weights && mdev_ವಿಭಜಿಸು; ++p) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst = octx->dsts[p];
        mdev_ವಿಭಜಿಸು = htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, ಗಾತ್ರ(ದಶಮಾಂಶ));
    }
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1 && !mdev_ವಿಭಜಿಸು && octx->ctx->mdev.idx > 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }
    uint32_t n_ಸಕ್ರಿಯ = 0;
    ಆದರೆ (mdev_ವಿಭಜಿಸು) {
        ಗಾಗಿ (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            ಆದರೆ (ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[a] > 0) n_ಸಕ್ರಿಯ++;
        }
    }
    ಸ್ಥಿರಾಂಕ bool expert_ವಿಭಜಿಸು = mdev_ವಿಭಜಿಸು && (n_ಸಕ್ರಿಯ >= octx->ctx->mdev.ಎಣಿಕೆ_2);

    uint32_t ಗುರಿ_dev = 0;
    ಗಾಗಿ (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ಸ್ಥಿರಾಂಕ int32_t cne1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a];
        ಆದರೆ (cne1 == 0) ಮುಂದುವರಿ;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m_padded = hex_align_ಮೇಲೆ(cne1, 32);
        ಪೂರ್ಣಾಂಕ m_ಆರಂಭ = 0, m_ಕೊನೆ = m_padded;
        ಆದರೆ (expert_ವಿಭಜಿಸು) {
            ಸ್ಥಿರಾಂಕ bool my_expert = (ಗುರಿ_dev == octx->ctx->mdev.idx);
            ಆದರೆ (++ಗುರಿ_dev == octx->ctx->mdev.ಎಣಿಕೆ_2) {
                ಗುರಿ_dev = 0;
            }
            ಆದರೆ (!my_expert) ಮುಂದುವರಿ;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mdev_ವಿಭಜಿಸು) {
            ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = (uint32_t) cne1 >= octx->ctx->mdev.ಎಣಿಕೆ_2;
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ((uint32_t) m_padded, can_ವಿಭಜಿಸು ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            m_ಆರಂಭ = (ಪೂರ್ಣಾಂಕ) range.ಆರಂಭ;
            m_ಕೊನೆ   = (ಪೂರ್ಣಾಂಕ) (range.ಆರಂಭ + range.ಎಣಿಕೆ_2);
        }
        ಆದರೆ (m_ಆರಂಭ >= m_ಕೊನೆ) ಮುಂದುವರಿ;

        ಗಾಗಿ (uint32_t p = 0; p < n_weights; ++p) {
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src_w = octx->src[p];
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst   = octx->dsts[p];
            ಆದರೆ (!src_w || !dst) ಮುಂದುವರಿ;

            ಪೂರ್ಣಾಂಕ ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (ದಶಮಾಂಶ*) dst->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ*) act->ದತ್ತಾಂಶ,
                                       src_w->ದತ್ತಾಂಶ + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (ಪೂರ್ಣಾಂಕ) src_w->nb[1], (ಪೂರ್ಣಾಂಕ) src_w->type,
                                       kparams->tile_ಗಾತ್ರ, kparams->aligned_tile_ಗಾತ್ರ,
                                       ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, cur_a, mmctx->mapping_stride,
                                       m_ಆರಂಭ, m_ಕೊನೆ, (ಪೂರ್ಣಾಂಕ) octx->n_ಎಳೆಗಳು);
            ಆದರೆ (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
            }
        }
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ hvx_mm_matmul_id_nx(
    ರಚನೆ htp_ops_context * octx,
    ರಚನೆ htp_mm_context * mmctx,
    work_ಸರತಿ_func_t hvx_mmid_task_func
) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded = mmctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded;
    ಸ್ಥಿರಾಂಕ uint32_t act_nrows            = mmctx->act_nrows;

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, 0);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_weights = kparams->n_weights;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ act  = octx->src[n_weights];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ ids  = octx->src[n_weights + 1];
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0->nb[1];

    ಸ್ಥಿರಾಂಕ uint32_t qk = QK_Q8_0_TILED;
    ಸ್ಥಿರಾಂಕ uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_nb = act_nrows * nb;

    work_ಸರತಿ_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ಆದರೆ (act_nrows < octx->n_ಎಳೆಗಳು) {
        n_quant_tasks = MIN(ಒಟ್ಟು_nb, octx->n_ಎಳೆಗಳು);
        quant_task_func = htp_mm_act_quant_ಖಂಡ_func(src0->type, act->type);
        ಗಾಗಿ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_ಮೊದಲ = (ಒಟ್ಟು_nb * ith) / n_quant_tasks;
            uint32_t ib_ಕೊನೆಯ  = (ಒಟ್ಟು_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_ಮೊದಲ[ith] = ib_ಮೊದಲ;
            mmctx->quant_ib_ಕೊನೆಯ[ith]  = ib_ಕೊನೆಯ;
            mmctx->quant_r[ith]        = ib_ಮೊದಲ / nb;
            mmctx->quant_c[ith]        = ib_ಮೊದಲ % nb;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        n_quant_tasks = MIN(act_nrows, octx->n_ಎಳೆಗಳು);
        quant_task_func = htp_mm_act_quant_ಅಡ್ಡಸಾಲು_func(src0->type, act->type);
    }
    size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src0->type, act->ne[0]);

    ರಚನೆ htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_ಎಳೆಗಳು,
                                 0, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0, kparams->n_prefetch, true, false);

    ಸ್ಥಿರಾಂಕ size_t vtcm_ಗಾತ್ರ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;

    ಆದರೆ (octx->ctx->vtcm_ಗಾತ್ರ < vtcm_ಗಾತ್ರ) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_ಗಾತ್ರ, vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    mmctx->vtcm_act_raw_stride = hex_ಸುತ್ತು_ಮೇಲೆ(act->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ), QK_Q8_0_TILED * ಗಾತ್ರ(ದಶಮಾಂಶ));

    mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = fastdiv(L.src0_ಬೈಟುಗಳು, &octx->n_ಎಳೆಗಳು_div);
    mmctx->vtcm_act_ಗಾತ್ರ_ಪ್ರತಿ_thread  = L.act_ಬೈಟುಗಳು;
    mmctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = fastdiv(L.dst_ಬೈಟುಗಳು, &octx->n_ಎಳೆಗಳು_div);

    mmctx->cur_m_ಆರಂಭ = 0;
    mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು  = act_nrows;

    FARF(ಮೇಲಿನ_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.ಒಟ್ಟು_ಬೈಟುಗಳು, octx->ctx->vtcm_ಗಾತ್ರ, octx->n_ಎಳೆಗಳು);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, quant_task_func, mmctx, n_quant_tasks);

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, hvx_mmid_task_func, mmctx, octx->n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಪರಿಶೀಲನೆ_expert_ids_n(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ids,
    ಸ್ಥಿರಾಂಕ uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
    uint32_t mapping_stride
) {
    ಸ್ಥಿರಾಂಕ size_t ids_nb1 = ids->nb[1];
    ಸ್ಥಿರಾಂಕ uint8_t * ids_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ uint8_t *) ids->ದತ್ತಾಂಶ;

    ಗಾಗಿ (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ಸ್ಥಿರಾಂಕ int32_t * ಅಡ್ಡಸಾಲು_ptr = (ಸ್ಥಿರಾಂಕ int32_t *) (ids_ದತ್ತಾಂಶ + iid1 * ids_nb1);
        ಗಾಗಿ (uint32_t id = 0; id < n_ids; ++id) {
            ಸ್ಥಿರಾಂಕ int32_t i02 = ಅಡ್ಡಸಾಲು_ptr[id];
            ಆದರೆ (i02 < 0) {
                ಮುಂದುವರಿ;
            }
            ಪ್ರತಿಜ್ಞಾ(i02 < n_as);

            ಆದರೆ (ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು) {
                ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[i02 * mapping_stride + counts[i02]] = (ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಪರಿಶೀಲನೆ_expert_ids(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
    uint32_t mapping_stride
) {
    ಸ್ಥಿರಾಂಕ size_t ids_nb0 = ids->nb[0];

    ಆದರೆ (ids_nb0 == 4) {
        ಆಯ್ಕೆ (n_ids) {
            ಸ್ಥಿತಿ 8:  ಪರಿಶೀಲನೆ_expert_ids_n(ids, 8,     n_as, counts, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, mapping_stride); ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 4:  ಪರಿಶೀಲನೆ_expert_ids_n(ids, 4,     n_as, counts, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, mapping_stride); ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 2:  ಪರಿಶೀಲನೆ_expert_ids_n(ids, 2,     n_as, counts, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, mapping_stride); ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ: ಪರಿಶೀಲನೆ_expert_ids_n(ids, n_ids, n_as, counts, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, mapping_stride); ನಿಲ್ಲಿಸು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Strided fallback
        ಸ್ಥಿರಾಂಕ size_t ids_nb1 = ids->nb[1];
        ಸ್ಥಿರಾಂಕ uint8_t * ids_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ uint8_t *) ids->ದತ್ತಾಂಶ;
        ಗಾಗಿ (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ಸ್ಥಿರಾಂಕ int32_t * ಅಡ್ಡಸಾಲು_ptr = (ಸ್ಥಿರಾಂಕ int32_t *) (ids_ದತ್ತಾಂಶ + iid1 * ids_nb1);
            ಗಾಗಿ (uint32_t id = 0; id < n_ids; ++id) {
                ಸ್ಥಿರಾಂಕ int32_t i02 = *(ಸ್ಥಿರಾಂಕ int32_t *) ((ಸ್ಥಿರಾಂಕ uint8_t *) ಅಡ್ಡಸಾಲು_ptr + id * ids_nb0);
                ಆದರೆ (i02 < 0) {
                    ಮುಂದುವರಿ;
                }
                ಪ್ರತಿಜ್ಞಾ(i02 < n_as);

                ಆದರೆ (ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು) {
                    ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[i02 * mapping_stride + counts[i02]] = (ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

ಪೂರ್ಣಾಂಕ op_matmul_id(ರಚನೆ htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ರಚನೆ htp_mm_context mmctx_struct = {0};
    ರಚನೆ htp_mm_context * mmctx = &mmctx_struct;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = htp_mm_init_context(octx, kparams);
    ಆದರೆ (ಸ್ಥಿತಿ_2 != HTP_ಸ್ಥಿತಿ_OK) {
        ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ ids = octx->src[2];
    ಆದರೆ (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = nb01;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = nb1;

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = ne01;  // per expert
    ಸ್ಥಿರಾಂಕ uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ids = ids->ne[0];  // n_expert_used
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_ತಳ;
    uint32_t   mapping_stride    = 1;
    uint32_t * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts = (uint32_t *) mapping_buf;
    ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು = ಖಾಲಿ;

    ಆದರೆ (act_nrows > 1) {
        ಸ್ಥಿರಾಂಕ size_t ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts_ಗಾತ್ರ = n_as * ಗಾತ್ರ(uint32_t);
        ಪ್ರತಿಜ್ಞಾ(octx->ctx->ddr_spad_ಗಾತ್ರ >= ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts_ಗಾತ್ರ);

        hex_l2fetch_ಖಂಡ((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ids->ದತ್ತಾಂಶ, ids->ne[1] * ids->nb[1]);

        memset(ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, 0, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts_ಗಾತ್ರ);
        ಪರಿಶೀಲನೆ_expert_ids(ids, n_ids, n_as, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, ಖಾಲಿ, 0);

        uint32_t max_ಎಣಿಕೆ = hvx_reduce_max_i32((ಸ್ಥಿರಾಂಕ uint8_t *) ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, n_as);
        mapping_stride = max_ಎಣಿಕೆ > 0 ? max_ಎಣಿಕೆ : 1;

        size_t ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_map_ಗಾತ್ರ  = n_as * mapping_stride * ಗಾತ್ರ(ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping);
        ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು_map_ಗಾತ್ರ = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts_ಗಾತ್ರ + ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_map_ಗಾತ್ರ;

        ಆದರೆ (ಒಟ್ಟು_map_ಗಾತ್ರ > octx->ctx->ddr_spad_ಗಾತ್ರ) {
            mapping_buf = memalign(128, ಒಟ್ಟು_map_ಗಾತ್ರ);
            ಆದರೆ (!mapping_buf) {
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR;
            }
        }

        ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts = (uint32_t *) mapping_buf;
        ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು       = (ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping *) (mapping_buf + ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts_ಗಾತ್ರ);

        memset(ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, 0, n_as * ಗಾತ್ರ(uint32_t));
        ಪರಿಶೀಲನೆ_expert_ids(ids, n_ids, n_as, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, mapping_stride);
    }

    mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts    = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts;
    mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು          = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_ಆರಂಭ          = 0;
    mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ಪೂರ್ಣಾಂಕ s;
    ಆದರೆ (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } ಇಲ್ಲದಿದ್ದರೆ {
        uint32_t src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
        uint32_t src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ   = src0_nrows;
        ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
            ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, ಗಾತ್ರ(ದಶಮಾಂಶ));
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(src0_nrows, can_ವಿಭಜಿಸು ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
            src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ   = range.ಆರಂಭ + range.ಎಣಿಕೆ_2;
        }

        ಆದರೆ (src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ >= src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
            ಆದರೆ (mapping_buf != octx->ctx->ddr_spad_ತಳ) {
                ಬಿಡುಗಡೆ(mapping_buf);
            }
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
        }

        ಸ್ಥಿರಾಂಕ uint32_t nrows = src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ - src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;
        mmctx->src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ = src0_ಅಡ್ಡಸಾಲು_ಆರಂಭ;
        mmctx->src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ   = src0_ಅಡ್ಡಸಾಲು_ಕೊನೆ;

        mmctx->src0_nrows_ಪ್ರತಿ_thread = fastdiv(nrows + octx->n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);
        mmctx->src0_nrows_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ(mmctx->src0_nrows_ಪ್ರತಿ_thread, 32);

        ಆದರೆ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } ಇಲ್ಲದಿದ್ದರೆ {
            s = HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
    }

    ಆದರೆ (mapping_buf != octx->ctx->ddr_spad_ತಳ) {
        ಬಿಡುಗಡೆ(mapping_buf);
    }

    ಹಿಂತಿರುಗಿ s;
}

ಪೂರ್ಣಾಂಕ op_matmul_id_nx(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;
    ರಚನೆ htp_mm_context mmctx_struct = {0};
    ರಚನೆ htp_mm_context * mmctx = &mmctx_struct;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = htp_mm_init_context(octx, kparams);
    ಆದರೆ (ಸ್ಥಿತಿ_2 != HTP_ಸ್ಥಿತಿ_OK) {
        ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    ಸ್ಥಿರಾಂಕ uint32_t n_weights = kparams->n_weights;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ act  = octx->src[n_weights];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ ids  = octx->src[n_weights + 1];
    ಆದರೆ (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
    ಗಾಗಿ (uint32_t p = 0; p < n_weights; p++) {
        ಆದರೆ (htp_tensor_is_extended(octx->dsts[p])) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);

    ಸ್ಥಿರಾಂಕ uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ids = ids->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_ತಳ;
    uint32_t   mapping_stride    = 1;
    uint32_t * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts = (uint32_t *) mapping_buf;
    ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು = ಖಾಲಿ;

    ಆದರೆ (act_nrows > 1) {
        ಸ್ಥಿರಾಂಕ size_t ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts_ಗಾತ್ರ = n_as * ಗಾತ್ರ(uint32_t);
        ಪ್ರತಿಜ್ಞಾ(octx->ctx->ddr_spad_ಗಾತ್ರ >= ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts_ಗಾತ್ರ);

        hex_l2fetch_ಖಂಡ((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ids->ದತ್ತಾಂಶ, ids->ne[1] * ids->nb[1]);

        memset(ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, 0, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts_ಗಾತ್ರ);
        ಪರಿಶೀಲನೆ_expert_ids(ids, n_ids, n_as, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, ಖಾಲಿ, 0);

        uint32_t max_ಎಣಿಕೆ = hvx_reduce_max_i32((ಸ್ಥಿರಾಂಕ uint8_t *) ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, n_as);
        mapping_stride = max_ಎಣಿಕೆ > 0 ? max_ಎಣಿಕೆ : 1;

        size_t ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_map_ಗಾತ್ರ  = n_as * mapping_stride * ಗಾತ್ರ(ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping);
        ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು_map_ಗಾತ್ರ = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts_ಗಾತ್ರ + ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_map_ಗಾತ್ರ;

        ಆದರೆ (ಒಟ್ಟು_map_ಗಾತ್ರ > octx->ctx->ddr_spad_ಗಾತ್ರ) {
            mapping_buf = memalign(128, ಒಟ್ಟು_map_ಗಾತ್ರ);
            ಆದರೆ (!mapping_buf) {
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR;
            }
        }

        ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts = (uint32_t *) mapping_buf;
        ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು       = (ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping *) (mapping_buf + ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts_ಗಾತ್ರ);

        memset(ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, 0, n_as * ಗಾತ್ರ(uint32_t));
        ಪರಿಶೀಲನೆ_expert_ids(ids, n_ids, n_as, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, mapping_stride);
    }

    mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts    = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts;
    mmctx->ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು          = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_ಆರಂಭ          = 0;
    mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ಪೂರ್ಣಾಂಕ s;
    ಆದರೆ (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } ಇಲ್ಲದಿದ್ದರೆ {
            s = HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
    }

    ಆದರೆ (mapping_buf != octx->ctx->ddr_spad_ತಳ) {
        ಬಿಡುಗಡೆ(mapping_buf);
    }

    ಹಿಂತಿರುಗಿ s;
}
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ op_matmul_nx_impl(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = htp_mm_init_context(octx, kparams);
    ಆದರೆ (ಸ್ಥಿತಿ_2 != HTP_ಸ್ಥಿತಿ_OK) {
        ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
    }

    ಆದರೆ (kparams->n_hmx) {
        ಹಿಂತಿರುಗಿ hmx_mm_nx_2d_f32(octx, kparams);
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, 0);

    ಸ್ಥಿರಾಂಕ uint32_t n_weights = kparams->n_weights;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0]; // first weight
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ act  = octx->src[n_weights]; // activation x

    ರಚನೆ htp_mm_context mmctx_struct = {0};
    ರಚನೆ htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    ಸ್ಥಿರಾಂಕ uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_ಆರಂಭ = 0;
    mmctx->cur_m_ಅಡ್ಡಸಾಲುಗಳು  = act_nrows;

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0->nb[1];

    ಆದರೆ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t qk = QK_Q8_0_TILED;
    ಸ್ಥಿರಾಂಕ uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_nb = act_nrows * nb;

    ಕೆಲಸಗಾರ_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ಆದರೆ (act_nrows < octx->n_ಎಳೆಗಳು) {
        n_quant_tasks = MIN(ಒಟ್ಟು_nb, octx->n_ಎಳೆಗಳು);
        quant_task_func = htp_mm_act_quant_ಖಂಡ_func(src0->type, act->type);
        ಗಾಗಿ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_ಮೊದಲ = (ಒಟ್ಟು_nb * ith) / n_quant_tasks;
            uint32_t ib_ಕೊನೆಯ  = (ಒಟ್ಟು_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_ಮೊದಲ[ith] = ib_ಮೊದಲ;
            mmctx->quant_ib_ಕೊನೆಯ[ith]  = ib_ಕೊನೆಯ;
            mmctx->quant_r[ith]        = ib_ಮೊದಲ / nb;
            mmctx->quant_c[ith]        = ib_ಮೊದಲ % nb;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        n_quant_tasks = MIN(act_nrows, octx->n_ಎಳೆಗಳು);
        quant_task_func = htp_mm_act_quant_ಅಡ್ಡಸಾಲು_func(src0->type, act->type);
    }

    ಸ್ಥಿರಾಂಕ size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src0->type, act->ne[0]);

    ರಚನೆ htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_ಎಳೆಗಳು,
                                 0, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0, kparams->n_prefetch, false, true);

    ಸ್ಥಿರಾಂಕ size_t vtcm_ಗಾತ್ರ = L.ಒಟ್ಟು_ಬೈಟುಗಳು;

    ಆದರೆ (octx->ctx->vtcm_ಗಾತ್ರ < vtcm_ಗಾತ್ರ) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_ಗಾತ್ರ, vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    mmctx->vtcm_act_raw_stride = hex_ಸುತ್ತು_ಮೇಲೆ(act->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ), QK_Q8_0_TILED * ಗಾತ್ರ(ದಶಮಾಂಶ));

    mmctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = fastdiv(L.src0_ಬೈಟುಗಳು, &octx->n_ಎಳೆಗಳು_div);
    mmctx->vtcm_act_ಗಾತ್ರ_ಪ್ರತಿ_thread  = L.act_ಬೈಟುಗಳು;
    mmctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = fastdiv(L.dst_ಬೈಟುಗಳು, &octx->n_ಎಳೆಗಳು_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, quant_task_func, mmctx, n_quant_tasks);

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, hvx_mm_nx_2d_tiled, mmctx, octx->n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_matmul_nx(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) octx->kernel_params;

    ಆದರೆ (kparams->collapse) {
        ಸ್ಥಿರಾಂಕ uint32_t n_weights = kparams->n_weights;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * act = octx->src[n_weights];
        ಸ್ಥಿರಾಂಕ uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ರಚನೆ htp_tensor act_collapsed;
        ರಚನೆ htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        ಗಾಗಿ (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * d = octx->dsts[p];
            ಸ್ಥಿರಾಂಕ uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_ಅಡ್ಡಸಾಲುಗಳು(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        ಗಾಗಿ (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
    }

    ಹಿಂತಿರುಗಿ op_matmul_nx_impl(octx);
}
