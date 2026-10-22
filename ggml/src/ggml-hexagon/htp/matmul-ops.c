#ਹਦਾਇਤ clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_compute_res.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdatomic.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-dump.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-arith.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-reduce.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "matmul-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਦਸ਼ਮਲਵ        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_ਬਾਈਟਾਂ;
    dma_addr_t    act_dma_addr;
    dma_addr_t    ਭਾਰ;
    dma_ਕਤਾਰ *   ਭਾਰ_dma;
    ਪੂਰਨ           m;
    ਪੂਰਨ           k;
    ਪੂਰਨ           n;
    ਪੂਰਨ           act_stride;
    uint32_t      act_elem_ਮਾਪ;  // 4=F32 src1, 2=F16 src1
    ਪੂਰਨ           ਭਾਰ_stride;
    ਪੂਰਨ           ਭਾਰ_type;
    ਪੂਰਨ           dst_stride;
    ਪੂਰਨ           dst_ਖਾਨੇ_ਗਿਣਤੀ;
    uint32_t      bias_stride;
    ਪੂਰਨ           tile_ਮਾਪ;
    ਪੂਰਨ           aligned_tile_ਮਾਪ;
    ਪੂਰਨ           ne02;
    ਪੂਰਨ           ne03;
    ਪੂਰਨ           ne12;
    ਪੂਰਨ           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    ਪੂਰਨ           r2;
    ਪੂਰਨ           r3;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_r2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_r3;
} hmx_mm_batched_params_t;

ਸਥਿਰ bool htp_matmul_has_extended_ਭਾਰ(ਅਚਲ ਬਣਤਰ htp_ops_context * octx, uint32_t n_weights) {
    ਲਈ (uint32_t i = 0; i < n_weights; ++i) {
        ਜੇ (htp_tensor_is_extended(octx->src[i])) {
            ਵਾਪਸ true;
        }
    }
    ਵਾਪਸ false;
}

ਬਣਤਰ htp_mm_context {
    ਅਚਲ ਅੱਖਰ * type;
    ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * act;

    ਖਾਲੀ (*vec_dot_1x1)(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s0,
         ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx0,
         ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy0);

    ਖਾਲੀ (*vec_dot_2x1)(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s0,
         ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx1,
         ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy0);

    ਖਾਲੀ (*vec_dot_2x2)(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s1,
         ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx1,
         ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy1);

    ਖਾਲੀ (*vec_dot_32x1)(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s,
         ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx,
         ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy, uint32_t valid_ਪੰਕਤੀਆਂ,
         ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ sz);

    ਖਾਲੀ (*vec_dot_32x2)(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s1,
         ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx,
         ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy1, uint32_t valid_ਪੰਕਤੀਆਂ,
         ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ sz0, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ sz1);

    // Precomputed values
    uint32_t src0_nrows_ਪ੍ਰਤੀ_thread;
    uint32_t src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t src0_ਪੰਕਤੀ_ਸਿਰਾ;
    uint32_t src0_ਪੰਕਤੀ_ਮਾਪ_padded;
    uint32_t act_nrows;
    uint32_t cur_m_ਸ਼ੁਰੂ;
    uint32_t cur_m_ਪੰਕਤੀਆਂ;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ mm_div_ne12_ne1;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ mm_div_ne1;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ mm_div_r2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ mm_div_r3;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_ਪਹਿਲਾ[WORK_ਕਤਾਰ_MAX_N_ਧਾਗੇ];
    uint32_t          quant_ib_ਆਖਰੀ[WORK_ਕਤਾਰ_MAX_N_ਧਾਗੇ];
    uint32_t          quant_r[WORK_ਕਤਾਰ_MAX_N_ਧਾਗੇ];
    uint32_t          quant_c[WORK_ਕਤਾਰ_MAX_N_ਧਾਗੇ];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ਅਚਲ uint32_t * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts;
    ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ;
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
    uint32_t vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t vtcm_act_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;
};

ਸਥਿਰ ਪੂਰਨ htp_mm_init_context(
    ਬਣਤਰ htp_ops_context * octx,
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams
) {
    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, (uint32_t) kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਜੇ (kparams->n_hmx) {
        ਜੇ (kparams->n_act_ਧਾਗੇ <= 0 || kparams->n_act_ਧਾਗੇ > (int32_t) octx->n_ਧਾਗੇ) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
        }
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
ਸਥਿਰ ਅਚਲ uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
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
ਸਥਿਰ ਅਚਲ uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

ਸਥਿਰ ਅਚਲ uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_matmul_tensors_preamble                                 \
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0];         \
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src1 = octx->src[1];         \
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src2 = octx->src[2];         \
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ  dst = octx->dst;            \
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ  act = src1;                 \
                                                                    \
    ਅਚਲ uint32_t ne00 = src0->ne[0];                              \
    ਅਚਲ uint32_t ne01 = src0->ne[1];                              \
    ਅਚਲ uint32_t ne02 = src0->ne[2];                              \
    ਅਚਲ uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    ਅਚਲ uint32_t ne10 = src1->ne[0];                              \
    ਅਚਲ uint32_t ne11 = src1->ne[1];                              \
    ਅਚਲ uint32_t ne12 = src1->ne[2];                              \
    ਅਚਲ uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    ਅਚਲ uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    ਅਚਲ uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    ਅਚਲ uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    ਅਚਲ uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    ਅਚਲ uint32_t ne0 = dst->ne[0];                                \
    ਅਚਲ uint32_t ne1 = dst->ne[1];                                \
    ਅਚਲ uint32_t ne2 = dst->ne[2];                                \
    ਅਚਲ uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    ਅਚਲ uint32_t nb00 = src0->nb[0];                              \
    ਅਚਲ uint32_t nb01 = src0->nb[1];                              \
    ਅਚਲ uint32_t nb02 = src0->nb[2];                              \
    ਅਚਲ uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    ਅਚਲ uint32_t nb10 = src1->nb[0];                              \
    ਅਚਲ uint32_t nb11 = src1->nb[1];                              \
    ਅਚਲ uint32_t nb12 = src1->nb[2];                              \
    ਅਚਲ uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    ਅਚਲ uint32_t nb0 = dst->nb[0];                                \
    ਅਚਲ uint32_t nb1 = dst->nb[1];                                \
    ਅਚਲ uint32_t nb2 = dst->nb[2];                                \
    ਅਚਲ uint32_t nb3 = dst->nb[3];

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_matmul_preamble                                         \
    ਬਣਤਰ htp_mm_context * mmctx  = ਅੰਕੜੇ;                          \
    ਬਣਤਰ htp_ops_context * octx  = mmctx->octx;                   \
    dma_ਕਤਾਰ * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = mmctx->src0_nrows_ਪ੍ਰਤੀ_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-mm-kernels-tiled.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-mm-kernels-float.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-mm-kernels-tiled.h"

ਸਥਿਰ ਖਾਲੀ hvx_mm_2d_tiled(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_matmul_preamble;

    ਅਚਲ uint32_t src0_nrows = mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ - mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    ਅਚਲ uint32_t act_nrows = mmctx->cur_m_ਪੰਕਤੀਆਂ ? mmctx->cur_m_ਪੰਕਤੀਆਂ : (ne11 * ne12 * ne13);
    ਅਚਲ uint32_t cur_m_ਸ਼ੁਰੂ = mmctx->cur_m_ਸ਼ੁਰੂ;

    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ  = mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ    = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = nb1;
    ਅਚਲ size_t act_ਪੰਕਤੀ_ਮਾਪ = nb11;
    ਅਚਲ size_t act_stride = mmctx->vtcm_act_stride;
    ਅਚਲ size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * ਸੀਮਿਤ vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  * ith;
    uint8_t * ਸੀਮਿਤ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * ਸੀਮਿਤ act_ਅੰਕੜੇ = mmctx->vtcm_act;

    ਅਚਲ dma_addr_t src0_ਪੰਕਤੀ = src0->ਅੰਕੜੇ;

    ਅਚਲ uint32_t tile_ਮਾਪ = htp_tensor_get_tile_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_ਪੰਕਤੀ_stride = n_k_tiles_w * tile_ਮਾਪ;
    uint32_t tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned = n_k_tiles_w * aligned_tile_ਮਾਪ;

    uint32_t ct_ਸ਼ੁਰੂ = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ / 32;
    uint32_t ct_ਸਿਰਾ   = (src0_ਸਿਰਾ_ਪੰਕਤੀ + 31) / 32;

    uint32_t ਧੱਕੋ_ct = ct_ਸ਼ੁਰੂ;
    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਲਈ (uint32_t d = 0; d < n_prefetch && ਧੱਕੋ_ct < ct_ਸਿਰਾ; d++, ਧੱਕੋ_ct++) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + d * tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned,
                           src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride), aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
        }
    }

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਲਈ (uint32_t ct = ct_ਸ਼ੁਰੂ; ct < ct_ਸਿਰਾ; ct++) {
        ਅਚਲ uint8_t * w_tile = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        ਪੂਰਨ valid_ਪੰਕਤੀਆਂ = (ਪੂਰਨ)ne0 - (ਪੂਰਨ)(ct * 32);
        valid_ਪੰਕਤੀਆਂ = MIN(32, MAX(0, valid_ਪੰਕਤੀਆਂ));

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        ਲਈ (; ir1 + 1 < act_nrows; ir1 += 2) {
            ਅਚਲ uint8_t * ਸੀਮਿਤ act_col0 = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (ir1+0) * act_stride);
            ਅਚਲ uint8_t * ਸੀਮਿਤ act_col1 = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (ir1+1) * act_stride);
            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_row0 = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ((cur_m_ਸ਼ੁਰੂ + ir1+0) * dst_ਪੰਕਤੀ_ਮਾਪ));
            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_row1 = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ((cur_m_ਸ਼ੁਰੂ + ir1+1) * dst_ਪੰਕਤੀ_ਮਾਪ));

            ਦਸ਼ਮਲਵ * dst_ptr0 = &dst_row0[ct * 32];
            ਦਸ਼ਮਲਵ * dst_ptr1 = &dst_row1[ct * 32];

            ਅਚਲ ਦਸ਼ਮਲਵ * src2_ptr0 = ਨਿਰਾ;
            ਅਚਲ ਦਸ਼ਮਲਵ * src2_ptr1 = ਨਿਰਾ;
            ਜੇ (src2) {
                ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src2_row0 = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src2->ਅੰਕੜੇ + ((cur_m_ਸ਼ੁਰੂ + ir1+0) * src2_stride));
                ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src2_row1 = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src2->ਅੰਕੜੇ + ((cur_m_ਸ਼ੁਰੂ + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_ਪੰਕਤੀਆਂ, src2_ptr0, src2_ptr1);
        }

        ਲਈ (; ir1 < act_nrows; ++ir1) {
            ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + ir1 * act_stride);
            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ          = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ((cur_m_ਸ਼ੁਰੂ + ir1) * dst_ਪੰਕਤੀ_ਮਾਪ));
            ਦਸ਼ਮਲਵ * dst_ptr = &dst_ਪੰਕਤੀ[ct * 32];

            ਅਚਲ ਦਸ਼ਮਲਵ * src2_ptr = ਨਿਰਾ;
            ਜੇ (src2) {
                ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src2_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src2->ਅੰਕੜੇ + ((cur_m_ਸ਼ੁਰੂ + ir1) * src2_stride));
                src2_ptr = &src2_ਪੰਕਤੀ[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ਖਾਨਾ, valid_ਪੰਕਤੀਆਂ, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        ਜੇ (ਧੱਕੋ_ct < ct_ਸਿਰਾ) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(w_tile, src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                           aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
            ਧੱਕੋ_ct++;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_mv_2d_tiled(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_matmul_preamble;

    ਅਚਲ uint32_t src0_nrows = mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ - mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;

    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ  = mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ    = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = nb1;
    ਅਚਲ size_t act_ਪੰਕਤੀ_ਮਾਪ = nb11;
    ਅਚਲ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * act_ਅੰਕੜੇ = mmctx->vtcm_act;

    ਦਸ਼ਮਲਵ * tmp = (ਦਸ਼ਮਲਵ *) vtcm_dst_ptr;

    ਅਚਲ dma_addr_t src0_ਪੰਕਤੀ = src0->ਅੰਕੜੇ;

    ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) act_ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਖਾਨਾ          = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਅਚਲ uint32_t tile_ਮਾਪ = htp_tensor_get_tile_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_ਪੰਕਤੀ_stride = n_k_tiles_w * tile_ਮਾਪ;
    uint32_t tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned = n_k_tiles_w * aligned_tile_ਮਾਪ;

    uint32_t ct_ਸ਼ੁਰੂ = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ / 32;
    uint32_t ct_ਸਿਰਾ   = (src0_ਸਿਰਾ_ਪੰਕਤੀ + 31) / 32;

    uint32_t ਧੱਕੋ_ct = ct_ਸ਼ੁਰੂ;
    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਜੇ (src2) {
            ਦਸ਼ਮਲਵ * vtcm_bias_ptr = (ਦਸ਼ਮਲਵ *) mmctx->vtcm_bias + src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
            ਅਚਲ dma_addr_t src2_addr = src2->ਅੰਕੜੇ + src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            ਪੂਰਨ ਟੁਕੜਾ_ਮਾਪ = (ਪੂਰਨ)MIN(src0_ਸਿਰਾ_ਪੰਕਤੀ, ne0) - (ਪੂਰਨ)src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
            ਜੇ (ਟੁਕੜਾ_ਮਾਪ > 0) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_bias_ptr, src2_addr),
                               ਟੁਕੜਾ_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਟੁਕੜਾ_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਟੁਕੜਾ_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 1);
                dma_ਕਤਾਰ_ਕੱਢੋ_nowait(dma_q);
            }
        }
        ਲਈ (uint32_t d = 0; d < n_prefetch && ਧੱਕੋ_ct < ct_ਸਿਰਾ; d++, ਧੱਕੋ_ct++) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + d * tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned,
                           src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride), aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
        }
    }

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਲਈ (uint32_t ct = ct_ਸ਼ੁਰੂ; ct < ct_ਸਿਰਾ; ct++) {
        ਅਚਲ uint8_t * w_tile = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        ਦਸ਼ਮਲਵ * dst_ptr = &tmp[ct * 32 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ];
        ਪੂਰਨ valid_ਪੰਕਤੀਆਂ = (ਪੂਰਨ)ne0 - (ਪੂਰਨ)(ct * 32);
        valid_ਪੰਕਤੀਆਂ = MIN(32, MAX(0, valid_ਪੰਕਤੀਆਂ));

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ਖਾਨਾ, valid_ਪੰਕਤੀਆਂ, ਨਿਰਾ);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        ਜੇ (ਧੱਕੋ_ct < ct_ਸਿਰਾ) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(w_tile, src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                           aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
            ਧੱਕੋ_ct++;
        }
    }

    ਪੂਰਨ copy_cnt = (ਪੂਰਨ)MIN(src0_ਸਿਰਾ_ਪੰਕਤੀ, ne0) - (ਪੂਰਨ)src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਜੇ (copy_cnt > 0) {
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ct_ਸਿਰਾ);
        ਜੇ (src2) {
            hvx_ਜੋੜੋ_f32_uuu((uint8_t *) &dst_ਖਾਨਾ[src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ],
                            (ਅਚਲ uint8_t *) tmp,
                            (ਅਚਲ uint8_t *) ((ਅਚਲ ਦਸ਼ਮਲਵ *) mmctx->vtcm_bias + src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ),
                            copy_cnt);
        } ਨਹੀਂਤਾਂ {
            hvx_copy_f32_ua((uint8_t *) &dst_ਖਾਨਾ[src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_ਸਿਰਾ);
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_mm_nx_2d_tiled(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_mm_context * mmctx = ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = mmctx->octx;
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_weights = kparams->n_weights;

    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ act = octx->src[n_weights]; /* x */
    ਅਚਲ uint32_t ne10 = act->ne[0];
    ਅਚਲ uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ਅਚਲ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * ਸੀਮਿਤ vtcm_ਭਾਰ_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * ਸੀਮਿਤ act_ਅੰਕੜੇ        = mmctx->vtcm_act;

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ਲਈ (uint32_t widx = 0; widx < n_weights; widx++) {
        ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src_w = octx->src[widx];
        ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst   = octx->dsts[widx];
        ਜੇ (!src_w || !dst) ਜਾਰੀ;
        dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

        ਅਚਲ uint32_t tile_ਮਾਪ = htp_tensor_get_tile_ਮਾਪ(src_w->type);
        ਅਚਲ uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(src_w->type);
        ਅਚਲ uint32_t ne00 = src_w->ne[0];
        ਅਚਲ uint32_t ne01 = src_w->ne[1];
        ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ = dst->nb[1];
        ਅਚਲ dma_addr_t src_w_ਪੰਕਤੀ = src_w->ਅੰਕੜੇ;

        ਅਚਲ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ਅਚਲ uint32_t tile_ਪੰਕਤੀ_stride = n_k_tiles_w * tile_ਮਾਪ;
        ਅਚਲ uint32_t tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned = n_k_tiles_w * aligned_tile_ਮਾਪ;

        uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = 0;
        uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = ne01;
        ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
            ਅਚਲ bool can_ਵੰਡੋ = htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ne01, can_ਵੰਡੋ ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
            src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = range.ਸ਼ੁਰੂ;
            src0_ਸਿਰਾ_ਪੰਕਤੀ   = range.ਸ਼ੁਰੂ + range.ਗਿਣਤੀ_2;
        }

        ਅਚਲ uint32_t nrows = src0_ਸਿਰਾ_ਪੰਕਤੀ - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
        uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + nth - 1, &octx->n_ਧਾਗੇ_div);
        src0_nrows_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ(src0_nrows_ਪ੍ਰਤੀ_thread, 32);

        ਅਚਲ uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
        ਅਚਲ uint32_t ਸਿਰਾ_ਪੰਕਤੀ   = MIN(ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, src0_ਸਿਰਾ_ਪੰਕਤੀ);
        ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਸਿਰਾ_ਪੰਕਤੀ) ਜਾਰੀ;

        uint32_t ct_ਸ਼ੁਰੂ = ਸ਼ੁਰੂ_ਪੰਕਤੀ / 32;
        uint32_t ct_ਸਿਰਾ   = (ਸਿਰਾ_ਪੰਕਤੀ + 31) / 32;

        uint32_t ਧੱਕੋ_ct = ct_ਸ਼ੁਰੂ;
        ਲਈ (uint32_t d = 0; d < n_prefetch && ਧੱਕੋ_ct < ct_ਸਿਰਾ; d++, ਧੱਕੋ_ct++) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_ਭਾਰ_ptr + d * tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned,
                            src_w_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride), aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
        }

        ਲਈ (uint32_t ct = ct_ਸ਼ੁਰੂ; ct < ct_ਸਿਰਾ; ct++) {
            ਅਚਲ uint8_t * w_tile = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
            ਪੂਰਨ valid_ਪੰਕਤੀਆਂ = (ਪੂਰਨ)dst->ne[0] - (ਪੂਰਨ)(ct * 32);
            valid_ਪੰਕਤੀਆਂ = MIN(32, MAX(0, valid_ਪੰਕਤੀਆਂ));

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            ਲਈ (; ir1 + 1 < act_nrows; ir1 += 2) {
                ਅਚਲ uint8_t * ਸੀਮਿਤ act_col0 = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (ir1+0) * act_stride);
                ਅਚਲ uint8_t * ਸੀਮਿਤ act_col1 = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (ir1+1) * act_stride);

                ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_row0 = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ((ir1+0) * dst_ਪੰਕਤੀ_ਮਾਪ));
                ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_row1 = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ((ir1+1) * dst_ਪੰਕਤੀ_ਮਾਪ));
                ਦਸ਼ਮਲਵ * dst_ptr0 = &dst_row0[ct * 32];
                ਦਸ਼ਮਲਵ * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_ਪੰਕਤੀਆਂ, ਨਿਰਾ, ਨਿਰਾ);
            }

            ਲਈ (; ir1 < act_nrows; ++ir1) {
                ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + ir1 * act_stride);
                ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + (ir1 * dst_ਪੰਕਤੀ_ਮਾਪ));
                ਦਸ਼ਮਲਵ * dst_ptr = &dst_ਪੰਕਤੀ[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ਖਾਨਾ, valid_ਪੰਕਤੀਆਂ, ਨਿਰਾ);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ਜੇ (ਧੱਕੋ_ct < ct_ਸਿਰਾ) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(w_tile, src_w_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                                aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
                ਧੱਕੋ_ct++;
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_mm_transfer_act_dma(
    ਬਣਤਰ htp_ops_context * octx,
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams,
    ਅਚਲ ਬਣਤਰ htp_tensor * act,
    uint8_t * dst_ਅਧਾਰ,
    size_t dst_ਪੰਕਤੀ_ਮਾਪ,
    uint32_t m_ਸ਼ੁਰੂ,
    uint32_t m_ਪੰਕਤੀਆਂ
) {
    ਜੇ (m_ਪੰਕਤੀਆਂ == 0) {
        ਵਾਪਸ;
    }

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[0];
    ਅਚਲ uint32_t ne0 = act->ne[0];
    ਅਚਲ size_t elem_ਮਾਪ = (act->type == HTP_TYPE_F16) ? ਆਕਾਰ(__fp16) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t ਪੰਕਤੀ_ਬਾਈਟਾਂ = ne0 * elem_ਮਾਪ;
    ਅਚਲ size_t act_nb1 = act->nb[1];
    ਅਚਲ dma_addr_t act_ਅਧਾਰ = act->ਅੰਕੜੇ;

    ਅਚਲ bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    ਜੇ (is_contiguous) {
        ਅਚਲ dma_addr_t src_addr = act_ਅਧਾਰ + m_ਸ਼ੁਰੂ * act_nb1;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_ਅਧਾਰ, src_addr),
                       dst_ਪੰਕਤੀ_ਮਾਪ, act_nb1, ਪੰਕਤੀ_ਬਾਈਟਾਂ, m_ਪੰਕਤੀਆਂ);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ਅਚਲ bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        ਲਈ (uint32_t ir = 0; ir < m_ਪੰਕਤੀਆਂ; ++ir) {
            ਅਚਲ uint32_t ir1 = m_ਸ਼ੁਰੂ + ir;
            uint32_t i11, i12, i13;
            ਜੇ (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                ਅਚਲ uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } ਨਹੀਂਤਾਂ {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                ਅਚਲ uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            ਅਚਲ dma_addr_t ਪੰਕਤੀ_src = act_ਅਧਾਰ + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * ਪੰਕਤੀ_dst = dst_ਅਧਾਰ + ir * dst_ਪੰਕਤੀ_ਮਾਪ;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ਪੰਕਤੀ_dst, ਪੰਕਤੀ_src),
                           dst_ਪੰਕਤੀ_ਮਾਪ, act_nb1, ਪੰਕਤੀ_ਬਾਈਟਾਂ, 1);
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        }
    }

    ਜੇ (dst_ਪੰਕਤੀ_ਮਾਪ > ਪੰਕਤੀ_ਬਾਈਟਾਂ) {
        ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_ਸ਼ੁਰੂ);
        ਅਚਲ uint32_t pad_elems = (dst_ਪੰਕਤੀ_ਮਾਪ - ਪੰਕਤੀ_ਬਾਈਟਾਂ) / elem_ਮਾਪ;
        ਜੇ (elem_ਮਾਪ == ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
            ਲਈ (uint32_t ir = 0; ir < m_ਪੰਕਤੀਆਂ; ++ir) {
                hvx_splat_f32_u(dst_ਅਧਾਰ + ir * dst_ਪੰਕਤੀ_ਮਾਪ + ਪੰਕਤੀ_ਬਾਈਟਾਂ, 0.0f, pad_elems);
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (uint32_t ir = 0; ir < m_ਪੰਕਤੀਆਂ; ++ir) {
                hvx_splat_f16_u(dst_ਅਧਾਰ + ir * dst_ਪੰਕਤੀ_ਮਾਪ + ਪੰਕਤੀ_ਬਾਈਟਾਂ, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_ਸ਼ੁਰੂ);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ QUANTIZE_IMPL(name, log_name, kernel_fn, dst_ਪੰਕਤੀ_ਮਾਪ_ਵਿਅੰਜਕ)                                        \
ਸਥਿਰ ਖਾਲੀ name(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {                                        \
    (ਖਾਲੀ) nth;                                                                                            \
    ਬਣਤਰ htp_mm_context * mmctx = ਅੰਕੜੇ;                                                                  \
    ਬਣਤਰ htp_ops_context * octx = mmctx->octx;                                                           \
    ਅਚਲ ਬਣਤਰ htp_tensor * src = mmctx->act;                                                            \
    ਅਚਲ uint32_t ne0 = src->ne[0];                                                                       \
    ਅਚਲ uint32_t nrows = mmctx->cur_m_ਪੰਕਤੀਆਂ ? mmctx->cur_m_ਪੰਕਤੀਆਂ : mmctx->act_nrows;                       \
    ਅਚਲ uint32_t nrows_ਪ੍ਰਤੀ_thread = mmctx->n_quant_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;                                      \
                                                                                                           \
    ਅਚਲ uint32_t ir_ਪਹਿਲਾ = nrows_ਪ੍ਰਤੀ_thread * ith;                                                      \
    ਜੇ (ir_ਪਹਿਲਾ >= nrows) {                                                                               \
        ਵਾਪਸ;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_ਪਹਿਲਾ);                                        \
                                                                                                           \
    uint8_t * ਸੀਮਿਤ dst = mmctx->vtcm_act;                                                              \
    ਅਚਲ uint32_t ir_ਆਖਰੀ = MIN(ir_ਪਹਿਲਾ + nrows_ਪ੍ਰਤੀ_thread, nrows);                                      \
    ਅਚਲ size_t raw_ਪੰਕਤੀ_ਮਾਪ = mmctx->vtcm_act_raw_stride;                                                \
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ = (dst_ਪੰਕਤੀ_ਮਾਪ_ਵਿਅੰਜਕ);                                                       \
                                                                                                           \
    ਅਚਲ uint8_t * ਸੀਮਿਤ src_ਅੰਕੜੇ = (ਅਚਲ uint8_t *) mmctx->vtcm_act_raw + (raw_ਪੰਕਤੀ_ਮਾਪ * ir_ਪਹਿਲਾ); \
    uint8_t * ਸੀਮਿਤ dst_ਅੰਕੜੇ = (uint8_t *) dst + (dst_ਪੰਕਤੀ_ਮਾਪ * ir_ਪਹਿਲਾ);                             \
    kernel_fn(src_ਅੰਕੜੇ, dst_ਅੰਕੜੇ, ਨਿਰਾ, ne0, ir_ਆਖਰੀ - ir_ਪਹਿਲਾ, raw_ਪੰਕਤੀ_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_ਪਹਿਲਾ);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled,     "quantize-f32-q8_0_tiled",     quantize_f32_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_ਪੰਕਤੀ_ਮਾਪ(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled,     "quantize-f32-q8_1_tiled",     quantize_f32_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_ਪੰਕਤੀ_ਮਾਪ(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_ਪੰਕਤੀ_ਮਾਪ(ne0))
QUANTIZE_IMPL(quantize_f32_f16_tiled,      "quantize-f32-f16_tiled",      quantize_f32_f16_tiled_kernel,      htp_mm_f16_tiled_ਪੰਕਤੀ_ਮਾਪ(ne0))
QUANTIZE_IMPL(quantize_f16_f16_tiled,      "quantize-f16-f16_tiled",      quantize_f16_f16_tiled_kernel,      htp_mm_f16_tiled_ਪੰਕਤੀ_ਮਾਪ(ne0))
QUANTIZE_IMPL(quantize_f32_f32,            "quantize-f32-f32",            quantize_f32_f32_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,            "quantize-f32-f16",            quantize_f32_f16_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,            "quantize-f16-f16",            quantize_f16_f16_kernel,            mmctx->vtcm_act_stride)

ਸਥਿਰ ਖਾਲੀ quantize_f32_q8_0_tiled_ਖੰਡ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਬਣਤਰ htp_mm_context * mmctx = ਅੰਕੜੇ;
    ਜੇ (mmctx->quant_ib_ਪਹਿਲਾ[ith] >= mmctx->quant_ib_ਆਖਰੀ[ith]) {
        ਵਾਪਸ;
    }
    ਬਣਤਰ htp_ops_context * octx = mmctx->octx;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ਪਹਿਲਾ[ith]);

    ਅਚਲ ਬਣਤਰ htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_ਖੰਡ_kernel(
        (ਅਚਲ ਦਸ਼ਮਲਵ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ਨਿਰਾ,
        src->ne[0],
        mmctx->quant_ib_ਪਹਿਲਾ[ith],
        mmctx->quant_ib_ਆਖਰੀ[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_ਪੰਕਤੀ_ਮਾਪ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ਪਹਿਲਾ[ith]);
}

ਸਥਿਰ ਖਾਲੀ quantize_f32_q8_1_tiled_ਖੰਡ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਬਣਤਰ htp_mm_context * mmctx = ਅੰਕੜੇ;
    ਜੇ (mmctx->quant_ib_ਪਹਿਲਾ[ith] >= mmctx->quant_ib_ਆਖਰੀ[ith]) {
        ਵਾਪਸ;
    }
    ਬਣਤਰ htp_ops_context * octx = mmctx->octx;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ਪਹਿਲਾ[ith]);

    ਅਚਲ ਬਣਤਰ htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_ਖੰਡ_kernel(
        (ਅਚਲ ਦਸ਼ਮਲਵ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ਨਿਰਾ,
        src->ne[0],
        mmctx->quant_ib_ਪਹਿਲਾ[ith],
        mmctx->quant_ib_ਆਖਰੀ[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_ਪੰਕਤੀ_ਮਾਪ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ਪਹਿਲਾ[ith]);
}

ਸਥਿਰ ਖਾਲੀ quantize_f32_q8_1_s16_tiled_ਖੰਡ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਬਣਤਰ htp_mm_context * mmctx = ਅੰਕੜੇ;
    ਜੇ (mmctx->quant_ib_ਪਹਿਲਾ[ith] >= mmctx->quant_ib_ਆਖਰੀ[ith]) {
        ਵਾਪਸ;
    }
    ਬਣਤਰ htp_ops_context * octx = mmctx->octx;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ਪਹਿਲਾ[ith]);

    ਅਚਲ ਬਣਤਰ htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_ਖੰਡ_kernel(
        (ਅਚਲ ਦਸ਼ਮਲਵ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ਨਿਰਾ,
        src->ne[0],
        mmctx->quant_ib_ਪਹਿਲਾ[ith],
        mmctx->quant_ib_ਆਖਰੀ[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_ਪੰਕਤੀ_ਮਾਪ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ਪਹਿਲਾ[ith]);
}

ਸਥਿਰ ਖਾਲੀ quantize_f32_f16_tiled_ਖੰਡ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਬਣਤਰ htp_mm_context * mmctx = ਅੰਕੜੇ;
    ਜੇ (mmctx->quant_ib_ਪਹਿਲਾ[ith] >= mmctx->quant_ib_ਆਖਰੀ[ith]) {
        ਵਾਪਸ;
    }
    ਬਣਤਰ htp_ops_context * octx = mmctx->octx;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ਪਹਿਲਾ[ith]);

    ਅਚਲ ਬਣਤਰ htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_ਖੰਡ_kernel(
        (ਅਚਲ ਦਸ਼ਮਲਵ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ਨਿਰਾ,
        src->ne[0],
        mmctx->quant_ib_ਪਹਿਲਾ[ith],
        mmctx->quant_ib_ਆਖਰੀ[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_ਪੰਕਤੀ_ਮਾਪ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ਪਹਿਲਾ[ith]);
}

ਸਥਿਰ ਖਾਲੀ quantize_f16_f16_tiled_ਖੰਡ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਬਣਤਰ htp_mm_context * mmctx = ਅੰਕੜੇ;
    ਜੇ (mmctx->quant_ib_ਪਹਿਲਾ[ith] >= mmctx->quant_ib_ਆਖਰੀ[ith]) {
        ਵਾਪਸ;
    }
    ਬਣਤਰ htp_ops_context * octx = mmctx->octx;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ਪਹਿਲਾ[ith]);

    ਅਚਲ ਬਣਤਰ htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_ਖੰਡ_kernel(
        (ਅਚਲ __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ਨਿਰਾ,
        src->ne[0],
        mmctx->quant_ib_ਪਹਿਲਾ[ith],
        mmctx->quant_ib_ਆਖਰੀ[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_ਪੰਕਤੀ_ਮਾਪ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_ਪਹਿਲਾ[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
ਸਥਿਰ ਸਤਰਅੰਦਰ ਕਾਮਾ_callback_t htp_mm_act_quant_ਪੰਕਤੀ_func(ਪੂਰਨ ਭਾਰ_type, ਪੂਰਨ act_type) {
    ਜੇ (ਭਾਰ_type == HTP_TYPE_F16) {
        ਵਾਪਸ (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    ਜੇ (ਭਾਰ_type == HTP_TYPE_Q2_K) {
        ਵਾਪਸ quantize_f32_q8_1_s16_tiled;
    }
    ਵਾਪਸ htp_mm_ਭਾਰ_has_ਵਿੱਥ(ਭਾਰ_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਕਾਮਾ_callback_t htp_mm_act_quant_ਖੰਡ_func(ਪੂਰਨ ਭਾਰ_type, ਪੂਰਨ act_type) {
    ਜੇ (ਭਾਰ_type == HTP_TYPE_F16) {
        ਵਾਪਸ (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_ਖੰਡ : quantize_f32_f16_tiled_ਖੰਡ;
    }
    ਜੇ (ਭਾਰ_type == HTP_TYPE_Q2_K) {
        ਵਾਪਸ quantize_f32_q8_1_s16_tiled_ਖੰਡ;
    }
    ਵਾਪਸ htp_mm_ਭਾਰ_has_ਵਿੱਥ(ਭਾਰ_type) ? quantize_f32_q8_1_tiled_ਖੰਡ : quantize_f32_q8_0_tiled_ਖੰਡ;
}

ਸਥਿਰ ਖਾਲੀ hvx_mm_4d_tiled(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_matmul_preamble;

    ਅਚਲ uint32_t src0_nrows = mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ - mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    ਅਚਲ uint32_t cur_m_ਪੰਕਤੀਆਂ = mmctx->cur_m_ਪੰਕਤੀਆਂ ? mmctx->cur_m_ਪੰਕਤੀਆਂ : (ne11 * ne12 * ne13);
    ਅਚਲ uint32_t cur_m_ਸ਼ੁਰੂ = mmctx->cur_m_ਸ਼ੁਰੂ;

    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ  = mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ    = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ = nb1;
    ਅਚਲ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * ਸੀਮਿਤ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * ਸੀਮਿਤ act_ਅੰਕੜੇ = mmctx->vtcm_act;

    ਅਚਲ uint32_t tile_ਮਾਪ = htp_tensor_get_tile_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(src0->type);

    ਅਚਲ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ਅਚਲ uint32_t tile_ਪੰਕਤੀ_stride = n_k_tiles_w * tile_ਮਾਪ;
    ਅਚਲ uint32_t tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned = n_k_tiles_w * aligned_tile_ਮਾਪ;
    ਅਚਲ uint32_t src0_ਟੁਕੜਾ_stride = ((ne01 + 31) / 32) * tile_ਪੰਕਤੀ_stride;

    ਅਚਲ uint32_t ct_ਸ਼ੁਰੂ = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ / 32;
    ਅਚਲ uint32_t ct_ਸਿਰਾ   = (src0_ਸਿਰਾ_ਪੰਕਤੀ + 31) / 32;

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ || cur_m_ਪੰਕਤੀਆਂ == 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_batches = ne12 * ne13;
    ਅਚਲ uint32_t b_ਸ਼ੁਰੂ = fastdiv(cur_m_ਸ਼ੁਰੂ, &kparams->div_ne1);
    uint32_t b_ਸਿਰਾ = fastdiv(cur_m_ਸ਼ੁਰੂ + cur_m_ਪੰਕਤੀਆਂ + ne11 - 1, &kparams->div_ne1);
    b_ਸਿਰਾ = MIN(b_ਸਿਰਾ, ਕੁੱਲ_batches);

    uint32_t b_grp_ਸ਼ੁਰੂ = b_ਸ਼ੁਰੂ;
    ਜਦੋਂ (b_grp_ਸ਼ੁਰੂ < b_ਸਿਰਾ) {
        ਅਚਲ uint32_t b3 = fastdiv(b_grp_ਸ਼ੁਰੂ, &kparams->div_ne12);
        ਅਚਲ uint32_t b2 = b_grp_ਸ਼ੁਰੂ - b3 * ne12;
        ਅਚਲ uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ਅਚਲ uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_ਸਿਰਾ = b_grp_ਸ਼ੁਰੂ + 1;
        ਜਦੋਂ (b_grp_ਸਿਰਾ < b_ਸਿਰਾ) {
            ਅਚਲ uint32_t cur_b3 = fastdiv(b_grp_ਸਿਰਾ, &kparams->div_ne12);
            ਅਚਲ uint32_t cur_b2 = b_grp_ਸਿਰਾ - cur_b3 * ne12;
            ਅਚਲ uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ਅਚਲ uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            ਜੇ (cur_i02 != i02 || cur_i03 != i03) {
                ਰੋਕੋ;
            }
            b_grp_ਸਿਰਾ++;
        }

        ਅਚਲ uint32_t ਟੁਕੜਾ_idx = i03 * ne02 + i02;
        ਅਚਲ dma_addr_t src0_ਟੁਕੜਾ = src0->ਅੰਕੜੇ + (size_t) ਟੁਕੜਾ_idx * src0_ਟੁਕੜਾ_stride;

        uint32_t ਧੱਕੋ_ct = ct_ਸ਼ੁਰੂ;
        ਲਈ (uint32_t d = 0; d < n_prefetch && ਧੱਕੋ_ct < ct_ਸਿਰਾ; d++, ਧੱਕੋ_ct++) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + d * tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned,
                           src0_ਟੁਕੜਾ + (size_t) ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                           aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
        }

        ਲਈ (uint32_t ct = ct_ਸ਼ੁਰੂ; ct < ct_ਸਿਰਾ; ct++) {
            ਅਚਲ uint8_t * w_tile = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

            ਪੂਰਨ valid_ਪੰਕਤੀਆਂ = (ਪੂਰਨ)ne0 - (ਪੂਰਨ)(ct * 32);
            valid_ਪੰਕਤੀਆਂ = MIN(32, MAX(0, valid_ਪੰਕਤੀਆਂ));

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            ਲਈ (uint32_t b = b_grp_ਸ਼ੁਰੂ; b < b_grp_ਸਿਰਾ; b++) {
                ਅਚਲ uint32_t b_m_ਸ਼ੁਰੂ = b * ne11;
                ਅਚਲ uint32_t m_ਪਹਿਲਾ = MAX(cur_m_ਸ਼ੁਰੂ, b_m_ਸ਼ੁਰੂ);
                ਅਚਲ uint32_t m_ਆਖਰੀ  = MIN(cur_m_ਸ਼ੁਰੂ + cur_m_ਪੰਕਤੀਆਂ, b_m_ਸ਼ੁਰੂ + ne11);
                ਜੇ (m_ਪਹਿਲਾ >= m_ਆਖਰੀ) ਜਾਰੀ;

                ਅਚਲ uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ਅਚਲ uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_ਅਧਾਰ = (uint8_t *) dst->ਅੰਕੜੇ + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ਅਚਲ uint32_t chunk_m_ਵਿੱਥ = m_ਪਹਿਲਾ - cur_m_ਸ਼ੁਰੂ;
                ਅਚਲ uint32_t dst_m_ਵਿੱਥ   = m_ਪਹਿਲਾ - b_m_ਸ਼ੁਰੂ;
                ਅਚਲ uint32_t batch_nrows    = m_ਆਖਰੀ - m_ਪਹਿਲਾ;

                uint32_t ir1 = 0;
                ਲਈ (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ਅਚਲ uint8_t * ਸੀਮਿਤ act_col0 = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (chunk_m_ਵਿੱਥ + ir1 + 0) * act_stride);
                    ਅਚਲ uint8_t * ਸੀਮਿਤ act_col1 = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (chunk_m_ਵਿੱਥ + ir1 + 1) * act_stride);
                    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_row0 = (ਦਸ਼ਮਲਵ *) (dst_batch_ਅਧਾਰ + (dst_m_ਵਿੱਥ + ir1 + 0) * dst_ਪੰਕਤੀ_ਮਾਪ);
                    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_row1 = (ਦਸ਼ਮਲਵ *) (dst_batch_ਅਧਾਰ + (dst_m_ਵਿੱਥ + ir1 + 1) * dst_ਪੰਕਤੀ_ਮਾਪ);
                    ਦਸ਼ਮਲਵ * dst_ptr0 = &dst_row0[ct * 32];
                    ਦਸ਼ਮਲਵ * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_ਪੰਕਤੀਆਂ, ਨਿਰਾ, ਨਿਰਾ);
                }
                ਲਈ (; ir1 < batch_nrows; ++ir1) {
                    ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (chunk_m_ਵਿੱਥ + ir1) * act_stride);
                    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) (dst_batch_ਅਧਾਰ + (dst_m_ਵਿੱਥ + ir1) * dst_ਪੰਕਤੀ_ਮਾਪ);
                    ਦਸ਼ਮਲਵ * dst_ptr = &dst_ਪੰਕਤੀ[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ਖਾਨਾ, valid_ਪੰਕਤੀਆਂ, ਨਿਰਾ);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ਜੇ (ਧੱਕੋ_ct < ct_ਸਿਰਾ) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(w_tile, src0_ਟੁਕੜਾ + (size_t) ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                               aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
                ਧੱਕੋ_ct++;
            }
        }
        b_grp_ਸ਼ੁਰੂ = b_grp_ਸਿਰਾ;
    }
    ਜੇ (src2) {
        hvx_tensor_ਜੋੜੋ_f32_grid(dst, src2, cur_m_ਸ਼ੁਰੂ, cur_m_ਸ਼ੁਰੂ + cur_m_ਪੰਕਤੀਆਂ, src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, src0_ਸਿਰਾ_ਪੰਕਤੀ, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_mm_2d_flat(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_matmul_preamble;

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ਅਚਲ uint32_t prefetch_ਨਕਾਬ = n_prefetch - 1;

    ਅਚਲ uint32_t src0_nrows = mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ - mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;  // src0 rows
    ਅਚਲ uint32_t act_nrows = mmctx->cur_m_ਪੰਕਤੀਆਂ ? mmctx->cur_m_ਪੰਕਤੀਆਂ : mmctx->act_nrows;
    ਅਚਲ uint32_t cur_m_ਸ਼ੁਰੂ = mmctx->cur_m_ਸ਼ੁਰੂ;

    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ  = mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ    = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ);
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ_x2 = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + ((src0_ਸਿਰਾ_ਪੰਕਤੀ - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) & ~1U);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = nb1;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = nb01;
    ਅਚਲ size_t act_ਪੰਕਤੀ_ਮਾਪ = nb11;

    ਅਚਲ size_t src0_stride = mmctx->vtcm_src0_stride;
    ਅਚਲ size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * ਸੀਮਿਤ vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  * ith;
    uint8_t * ਸੀਮਿਤ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * ਸੀਮਿਤ act_ਅੰਕੜੇ      = mmctx->vtcm_act;

    ਅਚਲ dma_addr_t src0_ਪੰਕਤੀ = src0->ਅੰਕੜੇ;


    // Prefill vtcm with src0 rows
    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਲਈ (uint32_t ir0 = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir0 < src0_ਸਿਰਾ_ਪੰਕਤੀ_x2; ir0 += 2) {
            ਅਚਲ ਪੂਰਨ is0 = (ir0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ);
            ਜੇ (is0 >= (ਪੂਰਨ)n_prefetch) {
                ਰੋਕੋ;
            }
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + is0 * src0_stride, src0_ਪੰਕਤੀ + ir0 * src0_ਪੰਕਤੀ_ਮਾਪ),
                           src0_stride, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 2);
        }
    }

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    // Process src0 rows
    ਲਈ (uint32_t ir0 = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir0 < src0_ਸਿਰਾ_ਪੰਕਤੀ_x2; ir0 += 2) {
        ਅਚਲ uint8_t * ss0 = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        ਲਈ (; ir1 + 1 < act_nrows; ir1 += 2) {
            ਅਚਲ uint8_t * ਸੀਮਿਤ act_col0 = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (ir1+0) * act_stride);
            ਅਚਲ uint8_t * ਸੀਮਿਤ act_col1 = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (ir1+1) * act_stride);
            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_row0 = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ((cur_m_ਸ਼ੁਰੂ + ir1+0) * dst_ਪੰਕਤੀ_ਮਾਪ));
            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_row1 = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ((cur_m_ਸ਼ੁਰੂ + ir1+1) * dst_ਪੰਕਤੀ_ਮਾਪ));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        ਲਈ (; ir1 < act_nrows; ++ir1) {
            ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + ir1 * act_stride);
            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ          = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ((cur_m_ਸ਼ੁਰੂ + ir1) * dst_ਪੰਕਤੀ_ਮਾਪ));
            mmctx->vec_dot_2x1(ne00, &dst_ਪੰਕਤੀ[ir0], ss0, ss0 + src0_stride, act_ਖਾਨਾ);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ਅਚਲ ਪੂਰਨ pr0 = (ir0 + n_prefetch);
        ਅਚਲ ਪੂਰਨ is0 = (pr0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) & prefetch_ਨਕਾਬ;
        ਜੇ (pr0 < src0_ਸਿਰਾ_ਪੰਕਤੀ_x2) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + is0 * src0_stride, src0_ਪੰਕਤੀ + pr0 * src0_ਪੰਕਤੀ_ਮਾਪ),
                           src0_stride, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 2);
        }
    }

    // Process the last row (if any)
    ਜੇ (src0_ਸਿਰਾ_ਪੰਕਤੀ != src0_ਸਿਰਾ_ਪੰਕਤੀ_x2) {
        uint32_t  ir0 = src0_ਸਿਰਾ_ਪੰਕਤੀ_x2;
        ਅਚਲ ਪੂਰਨ is0 = (ir0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) & prefetch_ਨਕਾਬ;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + is0 * src0_stride, src0_ਪੰਕਤੀ + ir0 * src0_ਪੰਕਤੀ_ਮਾਪ),
                       src0_stride, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 1);
        ਅਚਲ uint8_t * ss0 = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #ਹਦਾਇਤ unroll(2)
        ਲਈ (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + ir1 * act_stride);
            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ          = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ((cur_m_ਸ਼ੁਰੂ + ir1) * dst_ਪੰਕਤੀ_ਮਾਪ));
            mmctx->vec_dot_1x1(ne00, &dst_ਪੰਕਤੀ[ir0], ss0, act_ਖਾਨਾ);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    ਜੇ (src2) {
        hvx_tensor_ਜੋੜੋ_f32_grid(dst, src2, cur_m_ਸ਼ੁਰੂ, cur_m_ਸ਼ੁਰੂ + act_nrows, src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, src0_ਸਿਰਾ_ਪੰਕਤੀ, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_mv_2d_flat(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_matmul_preamble;

    ਅਚਲ uint32_t src0_nrows = mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ - mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;

    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ  = mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ    = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = nb1;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = nb01;
    ਅਚਲ size_t act_ਪੰਕਤੀ_ਮਾਪ  = nb11;

    ਅਚਲ size_t src0_stride = mmctx->vtcm_src0_stride;
    ਅਚਲ size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * act_ਅੰਕੜੇ      = mmctx->vtcm_act;

    ਦਸ਼ਮਲਵ * tmp = (ਦਸ਼ਮਲਵ *) vtcm_dst_ptr;

    ਅਚਲ dma_addr_t src0_ਪੰਕਤੀ = src0->ਅੰਕੜੇ;
    ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) act_ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਖਾਨਾ         = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ_x2 = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + ((src0_ਸਿਰਾ_ਪੰਕਤੀ - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) & ~1U);

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ਅਚਲ uint32_t prefetch_ਨਕਾਬ = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਜੇ (src2) {
            ਦਸ਼ਮਲਵ * vtcm_bias_ptr = (ਦਸ਼ਮਲਵ *) mmctx->vtcm_bias + src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
            ਅਚਲ dma_addr_t src2_addr = src2->ਅੰਕੜੇ + src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            ਪੂਰਨ ਟੁਕੜਾ_ਮਾਪ = (ਪੂਰਨ)src0_ਸਿਰਾ_ਪੰਕਤੀ - (ਪੂਰਨ)src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
            ਜੇ (ਟੁਕੜਾ_ਮਾਪ > 0) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_bias_ptr, src2_addr),
                               ਟੁਕੜਾ_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਟੁਕੜਾ_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਟੁਕੜਾ_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 1);
                dma_ਕਤਾਰ_ਕੱਢੋ_nowait(dma_q);
            }
        }
        ਲਈ (uint32_t ir0 = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir0 < src0_ਸਿਰਾ_ਪੰਕਤੀ_x2; ir0 += 2) {
            ਅਚਲ uint32_t is0 = (ir0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ);
            ਜੇ (is0 >= n_prefetch) {
                ਰੋਕੋ;
            }
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + is0 * src0_stride, src0_ਪੰਕਤੀ + ir0 * src0_ਪੰਕਤੀ_ਮਾਪ),
                           src0_stride, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 2);
        }
    }


    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    // Process src0 rows
    ਲਈ (uint32_t ir0 = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir0 < src0_ਸਿਰਾ_ਪੰਕਤੀ_x2; ir0 += 2) {
        ਅਚਲ uint8_t * ss0 = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ], ss0, ss0 + src0_stride, act_ਖਾਨਾ);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ਅਚਲ uint32_t pr0 = (ir0 + n_prefetch);
        ਅਚਲ uint32_t is0 = (pr0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) & prefetch_ਨਕਾਬ;
        ਜੇ (pr0 < src0_ਸਿਰਾ_ਪੰਕਤੀ_x2) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + is0 * src0_stride, src0_ਪੰਕਤੀ + pr0 * src0_ਪੰਕਤੀ_ਮਾਪ),
                           src0_stride, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 2);
        }
    }

    // Process the last row (if any)
    ਜੇ (src0_ਸਿਰਾ_ਪੰਕਤੀ != src0_ਸਿਰਾ_ਪੰਕਤੀ_x2) {
        ਅਚਲ uint32_t ir0 = src0_ਸਿਰਾ_ਪੰਕਤੀ_x2;
        ਅਚਲ uint32_t is0 = (ir0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) & prefetch_ਨਕਾਬ;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + is0 * src0_stride, src0_ਪੰਕਤੀ + ir0 * src0_ਪੰਕਤੀ_ਮਾਪ),
                       src0_stride, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 1);
        ਅਚਲ uint8_t * ss0 = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ], ss0, act_ਖਾਨਾ);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    ਪੂਰਨ copy_cnt = src0_ਸਿਰਾ_ਪੰਕਤੀ - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਜੇ (copy_cnt > 0) {
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, src0_ਸਿਰਾ_ਪੰਕਤੀ);
        ਜੇ (src2) {
            hvx_ਜੋੜੋ_f32_uuu((uint8_t *) &dst_ਖਾਨਾ[src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ],
                            (ਅਚਲ uint8_t *) tmp,
                            (ਅਚਲ uint8_t *) ((ਅਚਲ ਦਸ਼ਮਲਵ *) mmctx->vtcm_bias + src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ),
                            copy_cnt);
        } ਨਹੀਂਤਾਂ {
            hvx_copy_f32_ua((uint8_t *) &dst_ਖਾਨਾ[src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_ਸਿਰਾ_ਪੰਕਤੀ);
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_mm_4d_flat(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_matmul_preamble;

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ਅਚਲ uint32_t prefetch_ਨਕਾਬ = n_prefetch - 1;

    ਅਚਲ uint32_t src0_nrows = mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ - mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    ਅਚਲ uint32_t cur_m_ਪੰਕਤੀਆਂ = mmctx->cur_m_ਪੰਕਤੀਆਂ ? mmctx->cur_m_ਪੰਕਤੀਆਂ : (ne11 * ne12 * ne13);
    ਅਚਲ uint32_t cur_m_ਸ਼ੁਰੂ = mmctx->cur_m_ਸ਼ੁਰੂ;

    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ  = mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ    = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ);
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ_x2 = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + ((src0_ਸਿਰਾ_ਪੰਕਤੀ - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) & ~1U);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = nb1;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = nb01;
    ਅਚਲ size_t src0_stride = mmctx->vtcm_src0_stride;
    ਅਚਲ size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * ਸੀਮਿਤ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * ਸੀਮਿਤ act_ਅੰਕੜੇ      = mmctx->vtcm_act;


    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ || cur_m_ਪੰਕਤੀਆਂ == 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_batches = ne12 * ne13;
    ਅਚਲ uint32_t b_ਸ਼ੁਰੂ = fastdiv(cur_m_ਸ਼ੁਰੂ, &kparams->div_ne1);
    uint32_t b_ਸਿਰਾ = fastdiv(cur_m_ਸ਼ੁਰੂ + cur_m_ਪੰਕਤੀਆਂ + ne11 - 1, &kparams->div_ne1);
    b_ਸਿਰਾ = MIN(b_ਸਿਰਾ, ਕੁੱਲ_batches);

    uint32_t b_grp_ਸ਼ੁਰੂ = b_ਸ਼ੁਰੂ;
    ਜਦੋਂ (b_grp_ਸ਼ੁਰੂ < b_ਸਿਰਾ) {
        ਅਚਲ uint32_t b3 = fastdiv(b_grp_ਸ਼ੁਰੂ, &kparams->div_ne12);
        ਅਚਲ uint32_t b2 = b_grp_ਸ਼ੁਰੂ - b3 * ne12;
        ਅਚਲ uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ਅਚਲ uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_ਸਿਰਾ = b_grp_ਸ਼ੁਰੂ + 1;
        ਜਦੋਂ (b_grp_ਸਿਰਾ < b_ਸਿਰਾ) {
            ਅਚਲ uint32_t cur_b3 = fastdiv(b_grp_ਸਿਰਾ, &kparams->div_ne12);
            ਅਚਲ uint32_t cur_b2 = b_grp_ਸਿਰਾ - cur_b3 * ne12;
            ਅਚਲ uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ਅਚਲ uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            ਜੇ (cur_i02 != i02 || cur_i03 != i03) {
                ਰੋਕੋ;
            }
            b_grp_ਸਿਰਾ++;
        }

        ਅਚਲ dma_addr_t src0_ਪੰਕਤੀ = src0->ਅੰਕੜੇ + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        ਲਈ (uint32_t ir0 = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir0 < src0_ਸਿਰਾ_ਪੰਕਤੀ_x2; ir0 += 2) {
            ਅਚਲ ਪੂਰਨ is0 = (ir0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ);
            ਜੇ (is0 >= (ਪੂਰਨ)n_prefetch) {
                ਰੋਕੋ;
            }
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + is0 * src0_stride, src0_ਪੰਕਤੀ + (size_t) ir0 * src0_ਪੰਕਤੀ_ਮਾਪ),
                           src0_stride, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 2);
        }

        ਲਈ (uint32_t ir0 = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir0 < src0_ਸਿਰਾ_ਪੰਕਤੀ_x2; ir0 += 2) {
            ਅਚਲ uint8_t * ss0 = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            ਲਈ (uint32_t b = b_grp_ਸ਼ੁਰੂ; b < b_grp_ਸਿਰਾ; b++) {
                ਅਚਲ uint32_t b_m_ਸ਼ੁਰੂ = b * ne11;
                ਅਚਲ uint32_t m_ਪਹਿਲਾ = MAX(cur_m_ਸ਼ੁਰੂ, b_m_ਸ਼ੁਰੂ);
                ਅਚਲ uint32_t m_ਆਖਰੀ  = MIN(cur_m_ਸ਼ੁਰੂ + cur_m_ਪੰਕਤੀਆਂ, b_m_ਸ਼ੁਰੂ + ne11);
                ਜੇ (m_ਪਹਿਲਾ >= m_ਆਖਰੀ) ਜਾਰੀ;

                ਅਚਲ uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ਅਚਲ uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_ਅਧਾਰ = (uint8_t *) dst->ਅੰਕੜੇ + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ਅਚਲ uint32_t chunk_m_ਵਿੱਥ = m_ਪਹਿਲਾ - cur_m_ਸ਼ੁਰੂ;
                ਅਚਲ uint32_t dst_m_ਵਿੱਥ   = m_ਪਹਿਲਾ - b_m_ਸ਼ੁਰੂ;
                ਅਚਲ uint32_t batch_nrows    = m_ਆਖਰੀ - m_ਪਹਿਲਾ;

                uint32_t ir1 = 0;
                ਲਈ (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ਅਚਲ uint8_t * ਸੀਮਿਤ act_col0 = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (chunk_m_ਵਿੱਥ + ir1 + 0) * act_stride);
                    ਅਚਲ uint8_t * ਸੀਮਿਤ act_col1 = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (chunk_m_ਵਿੱਥ + ir1 + 1) * act_stride);
                    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_row0 = (ਦਸ਼ਮਲਵ *) (dst_batch_ਅਧਾਰ + (dst_m_ਵਿੱਥ + ir1 + 0) * dst_ਪੰਕਤੀ_ਮਾਪ);
                    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_row1 = (ਦਸ਼ਮਲਵ *) (dst_batch_ਅਧਾਰ + (dst_m_ਵਿੱਥ + ir1 + 1) * dst_ਪੰਕਤੀ_ਮਾਪ);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                ਲਈ (; ir1 < batch_nrows; ++ir1) {
                    ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (chunk_m_ਵਿੱਥ + ir1) * act_stride);
                    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ          = (ਦਸ਼ਮਲਵ *) (dst_batch_ਅਧਾਰ + (dst_m_ਵਿੱਥ + ir1) * dst_ਪੰਕਤੀ_ਮਾਪ);
                    mmctx->vec_dot_2x1(ne00, &dst_ਪੰਕਤੀ[ir0], ss0, ss0 + src0_stride, act_ਖਾਨਾ);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ਅਚਲ ਪੂਰਨ pr0 = (ir0 + n_prefetch);
            ਅਚਲ ਪੂਰਨ is0 = (pr0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) & prefetch_ਨਕਾਬ;
            ਜੇ (pr0 < src0_ਸਿਰਾ_ਪੰਕਤੀ_x2) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + is0 * src0_stride, src0_ਪੰਕਤੀ + (size_t) pr0 * src0_ਪੰਕਤੀ_ਮਾਪ),
                               src0_stride, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 2);
            }
        }

        ਜੇ (src0_ਸਿਰਾ_ਪੰਕਤੀ != src0_ਸਿਰਾ_ਪੰਕਤੀ_x2) {
            uint32_t ir0 = src0_ਸਿਰਾ_ਪੰਕਤੀ_x2;
            ਅਚਲ ਪੂਰਨ is0 = (ir0 - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) & prefetch_ਨਕਾਬ;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + is0 * src0_stride, src0_ਪੰਕਤੀ + (size_t) ir0 * src0_ਪੰਕਤੀ_ਮਾਪ),
                           src0_stride, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 1);
            ਅਚਲ uint8_t * ss0 = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            ਲਈ (uint32_t b = b_grp_ਸ਼ੁਰੂ; b < b_grp_ਸਿਰਾ; b++) {
                ਅਚਲ uint32_t b_m_ਸ਼ੁਰੂ = b * ne11;
                ਅਚਲ uint32_t m_ਪਹਿਲਾ = MAX(cur_m_ਸ਼ੁਰੂ, b_m_ਸ਼ੁਰੂ);
                ਅਚਲ uint32_t m_ਆਖਰੀ  = MIN(cur_m_ਸ਼ੁਰੂ + cur_m_ਪੰਕਤੀਆਂ, b_m_ਸ਼ੁਰੂ + ne11);
                ਜੇ (m_ਪਹਿਲਾ >= m_ਆਖਰੀ) ਜਾਰੀ;

                ਅਚਲ uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ਅਚਲ uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_ਅਧਾਰ = (uint8_t *) dst->ਅੰਕੜੇ + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ਅਚਲ uint32_t chunk_m_ਵਿੱਥ = m_ਪਹਿਲਾ - cur_m_ਸ਼ੁਰੂ;
                ਅਚਲ uint32_t dst_m_ਵਿੱਥ   = m_ਪਹਿਲਾ - b_m_ਸ਼ੁਰੂ;
                ਅਚਲ uint32_t batch_nrows    = m_ਆਖਰੀ - m_ਪਹਿਲਾ;

                ਲਈ (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (chunk_m_ਵਿੱਥ + ir1) * act_stride);
                    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ          = (ਦਸ਼ਮਲਵ *) (dst_batch_ਅਧਾਰ + (dst_m_ਵਿੱਥ + ir1) * dst_ਪੰਕਤੀ_ਮਾਪ);
                    mmctx->vec_dot_1x1(ne00, &dst_ਪੰਕਤੀ[ir0], ss0, act_ਖਾਨਾ);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_ਸ਼ੁਰੂ = b_grp_ਸਿਰਾ;
    }

    ਜੇ (src2) {
        hvx_tensor_ਜੋੜੋ_f32_grid(dst, src2, cur_m_ਸ਼ੁਰੂ, cur_m_ਸ਼ੁਰੂ + cur_m_ਪੰਕਤੀਆਂ, src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, src0_ਸਿਰਾ_ਪੰਕਤੀ, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(ਪੰਕਤੀ_id, i1) ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[(ਪੰਕਤੀ_id) * mmctx->mapping_stride + (i1)]

ਸਥਿਰ ਖਾਲੀ hvx_mm_id_tiled(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_matmul_preamble;

    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ ids = octx->src[2];

    ਅਚਲ uint32_t src0_nrows      = mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ - mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;  // src0 rows per expert
    ਅਚਲ uint32_t act_nrows       = ne11;
    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ  = mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ    = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ);


    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ਅਚਲ uint32_t n_ids = ids->ne[0];  // n_expert_used
    ਅਚਲ uint32_t n_as  = ne02;        // n_expert

    ਅਚਲ uint32_t *                ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts;
    ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ       = mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ;

    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = nb1;
    ਅਚਲ size_t act_ਪੰਕਤੀ_ਮਾਪ  = htp_mm_q8_0_tiled_ਪੰਕਤੀ_ਮਾਪ(ne10);

    ਅਚਲ size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * ਸੀਮਿਤ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * ਸੀਮਿਤ act_ਅੰਕੜੇ = mmctx->vtcm_act;

    ਲਈ (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ਅਚਲ int32_t cne1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a];
        ਜੇ (cne1 == 0) {
            ਜਾਰੀ;
        }

        ਅਚਲ dma_addr_t src0_ਪੰਕਤੀ = src0->ਅੰਕੜੇ + cur_a * nb02;

        ਅਚਲ uint32_t tile_ਮਾਪ = htp_tensor_get_tile_ਮਾਪ(src0->type);
        ਅਚਲ uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(src0->type);
        ਅਚਲ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ਅਚਲ uint32_t tile_ਪੰਕਤੀ_stride = n_k_tiles_w * tile_ਮਾਪ;
        ਅਚਲ uint32_t tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned = n_k_tiles_w * aligned_tile_ਮਾਪ;

        ਅਚਲ uint32_t ct_ਸ਼ੁਰੂ = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ / 32;
        ਅਚਲ uint32_t ct_ਸਿਰਾ   = (src0_ਸਿਰਾ_ਪੰਕਤੀ + 31) / 32;

        uint32_t ਧੱਕੋ_ct = ct_ਸ਼ੁਰੂ;
        ਲਈ (uint32_t d = 0; d < n_prefetch && ਧੱਕੋ_ct < ct_ਸਿਰਾ; d++, ਧੱਕੋ_ct++) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + d * tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned, src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                           aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
        }

        ਲਈ (uint32_t ct = ct_ਸ਼ੁਰੂ; ct < ct_ਸਿਰਾ; ct++) {
            ਅਚਲ uint8_t * w_tile = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

            ਪੂਰਨ valid_ਪੰਕਤੀਆਂ = (ਪੂਰਨ)ne0 - (ਪੂਰਨ)(ct * 32);
            valid_ਪੰਕਤੀਆਂ = MIN(32, MAX(0, valid_ਪੰਕਤੀਆਂ));

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            ਲਈ (uint32_t cid = 0; cid < cne1; ++cid) {
                ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, cid);
                ਅਚਲ ਪੂਰਨ               rm1         = ਪੰਕਤੀ_mapping.i1;  // expert idx
                ਅਚਲ ਪੂਰਨ               rm2         = ਪੰਕਤੀ_mapping.i2;  // token idx

                ਅਚਲ uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (ir1 + rm2 * ne11 + 0) * act_stride);
                ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_ਪੰਕਤੀ[ct * 32], w_tile, act_ਖਾਨਾ, valid_ਪੰਕਤੀਆਂ, ਨਿਰਾ);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ਜੇ (ਧੱਕੋ_ct < ct_ਸਿਰਾ) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(w_tile, src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                               aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
                ਧੱਕੋ_ct++;
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_mv_id_tiled(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_matmul_preamble;

    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ ids = octx->src[2];

    ਅਚਲ uint32_t src0_nrows      = mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ - mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;  // src0 rows per expert
    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ  = mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ    = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ);


    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ਪ੍ਰਤਿਜ੍ਞਾ(ne13 % ne03 == 0);

    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = nb1;
    ਅਚਲ size_t act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_q8_0_tiled_ਪੰਕਤੀ_ਮਾਪ(ne10);

    ਅਚਲ uint32_t n_aids = src2->ne[0];  // num activated experts
    ਅਚਲ uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * ਸੀਮਿਤ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * ਸੀਮਿਤ act_ਅੰਕੜੇ = mmctx->vtcm_act;

    ਲਈ (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ਅਚਲ int32_t eid = *(ਅਚਲ int32_t *) ((ਅਚਲ uint8_t *) src2->ਅੰਕੜੇ + ie1 * src2->nb[0]);
        ਜੇ (eid < 0) {
            ਜਾਰੀ;
        }
        ਪ੍ਰਤਿਜ੍ਞਾ(eid < (int32_t) n_ids);

        ਅਚਲ dma_addr_t src0_ਪੰਕਤੀ = src0->ਅੰਕੜੇ + eid * nb02;
        ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) act_ਅੰਕੜੇ;
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ          = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ie1 * nb1);

        ਅਚਲ uint32_t tile_ਮਾਪ = htp_tensor_get_tile_ਮਾਪ(src0->type);
        ਅਚਲ uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(src0->type);
        ਅਚਲ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ਅਚਲ uint32_t tile_ਪੰਕਤੀ_stride = n_k_tiles_w * tile_ਮਾਪ;
        ਅਚਲ uint32_t tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned = n_k_tiles_w * aligned_tile_ਮਾਪ;

        ਅਚਲ uint32_t ct_ਸ਼ੁਰੂ = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ / 32;
        ਅਚਲ uint32_t ct_ਸਿਰਾ   = (src0_ਸਿਰਾ_ਪੰਕਤੀ + 31) / 32;

        uint32_t ਧੱਕੋ_ct = ct_ਸ਼ੁਰੂ;
        ਲਈ (uint32_t d = 0; d < n_prefetch && ਧੱਕੋ_ct < ct_ਸਿਰਾ; d++, ਧੱਕੋ_ct++) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + d * tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned, src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                           aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
        }

        ਲਈ (uint32_t ct = ct_ਸ਼ੁਰੂ; ct < ct_ਸਿਰਾ; ct++) {
            ਅਚਲ uint8_t * w_tile = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

            ਪੂਰਨ valid_ਪੰਕਤੀਆਂ = (ਪੂਰਨ)ne0 - (ਪੂਰਨ)(ct * 32);
            valid_ਪੰਕਤੀਆਂ = MIN(32, MAX(0, valid_ਪੰਕਤੀਆਂ));

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_ਪੰਕਤੀ[ct * 32], w_tile, act_ਖਾਨਾ, valid_ਪੰਕਤੀਆਂ, ਨਿਰਾ);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ਜੇ (ਧੱਕੋ_ct < ct_ਸਿਰਾ) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(w_tile, src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                               aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
                ਧੱਕੋ_ct++;
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_mv_id_nx_tiled(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_mm_context * mmctx = (ਬਣਤਰ htp_mm_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = mmctx->octx;
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_weights      = kparams->n_weights;
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ act  = octx->src[n_weights];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ ids  = octx->src[n_weights + 1];


    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ਅਚਲ uint32_t n_aids = ids->ne[0];
    ਅਚਲ uint32_t n_ids  = src0->ne[2];

    uint8_t * ਸੀਮਿਤ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * ਸੀਮਿਤ act_ਅੰਕੜੇ = mmctx->vtcm_act;

    ਲਈ (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ਅਚਲ int32_t eid = *(ਅਚਲ int32_t *) ((ਅਚਲ uint8_t *) ids->ਅੰਕੜੇ + ie1 * ids->nb[0]);
        ਜੇ (eid < 0) ਜਾਰੀ;
        ਪ੍ਰਤਿਜ੍ਞਾ(eid < (int32_t) n_ids);

        ਲਈ (uint32_t p = 0; p < n_weights; ++p) {
            ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src_w = octx->src[p];
            ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst   = octx->dsts[p];
            ਜੇ (!src_w || !dst) ਜਾਰੀ;
            dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

            ਅਚਲ uint32_t ne01 = src_w->ne[1];
            uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = 0;
            uint32_t ਸਿਰਾ_ਪੰਕਤੀ   = ne01;
            ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
                ਅਚਲ bool can_ਵੰਡੋ = htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ne01, can_ਵੰਡੋ ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
                ਸ਼ੁਰੂ_ਪੰਕਤੀ = range.ਸ਼ੁਰੂ;
                ਸਿਰਾ_ਪੰਕਤੀ   = range.ਸ਼ੁਰੂ + range.ਗਿਣਤੀ_2;
            }

            ਅਚਲ uint32_t nrows = ਸਿਰਾ_ਪੰਕਤੀ - ਸ਼ੁਰੂ_ਪੰਕਤੀ;
            uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + nth - 1, &octx->n_ਧਾਗੇ_div);
            src0_nrows_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ(src0_nrows_ਪ੍ਰਤੀ_thread, 32);

            ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
            ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, ਸਿਰਾ_ਪੰਕਤੀ);
            ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) ਜਾਰੀ;

            ਅਚਲ dma_addr_t src0_ਪੰਕਤੀ = src_w->ਅੰਕੜੇ + eid * src_w->nb[2];
            ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) act_ਅੰਕੜੇ;
            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + ie1 * dst->nb[1]);

            ਅਚਲ uint32_t tile_ਮਾਪ = htp_tensor_get_tile_ਮਾਪ(src_w->type);
            ਅਚਲ uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(src_w->type);
            ਅਚਲ uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ਅਚਲ uint32_t tile_ਪੰਕਤੀ_stride = n_k_tiles_w * tile_ਮਾਪ;
            ਅਚਲ uint32_t tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned = n_k_tiles_w * aligned_tile_ਮਾਪ;

            ਅਚਲ uint32_t ct_ਸ਼ੁਰੂ = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ / 32;
            ਅਚਲ uint32_t ct_ਸਿਰਾ   = (src0_ਸਿਰਾ_ਪੰਕਤੀ + 31) / 32;

            uint32_t ਧੱਕੋ_ct = ct_ਸ਼ੁਰੂ;
            ਲਈ (uint32_t d = 0; d < n_prefetch && ਧੱਕੋ_ct < ct_ਸਿਰਾ; d++, ਧੱਕੋ_ct++) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + d * tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned, src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                               aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
            }

            ਲਈ (uint32_t ct = ct_ਸ਼ੁਰੂ; ct < ct_ਸਿਰਾ; ct++) {
                ਅਚਲ uint8_t * w_tile = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

                ਪੂਰਨ valid_ਪੰਕਤੀਆਂ = (ਪੂਰਨ)dst->ne[0] - (ਪੂਰਨ)(ct * 32);
                valid_ਪੰਕਤੀਆਂ = MIN(32, MAX(0, valid_ਪੰਕਤੀਆਂ));

                htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_ਪੰਕਤੀ[ct * 32], w_tile, act_ਖਾਨਾ, valid_ਪੰਕਤੀਆਂ, ਨਿਰਾ);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                ਜੇ (ਧੱਕੋ_ct < ct_ਸਿਰਾ) {
                    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(w_tile, src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                                    aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
                    ਧੱਕੋ_ct++;
                }
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_mm_id_nx_tiled(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_mm_context * mmctx = (ਬਣਤਰ htp_mm_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = mmctx->octx;
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_weights      = kparams->n_weights;
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ act  = octx->src[n_weights];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ ids  = octx->src[n_weights + 1];


    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ uint32_t n_prefetch = kparams->n_prefetch;
    ਪ੍ਰਤਿਜ੍ਞਾ(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ਅਚਲ uint32_t n_as = src0->ne[2];

    ਅਚਲ uint32_t * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts;
    ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ = mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ;

    ਅਚਲ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * ਸੀਮਿਤ vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * ith;
    uint8_t * ਸੀਮਿਤ act_ਅੰਕੜੇ = mmctx->vtcm_act;

    ਲਈ (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ਅਚਲ int32_t cne1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a];
        ਜੇ (cne1 == 0) ਜਾਰੀ;

        ਲਈ (uint32_t p = 0; p < n_weights; ++p) {
            ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src_w = octx->src[p];
            ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst   = octx->dsts[p];
            ਜੇ (!src_w || !dst) ਜਾਰੀ;
            dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

            ਅਚਲ uint32_t ne01 = src_w->ne[1];
            uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = 0;
            uint32_t ਸਿਰਾ_ਪੰਕਤੀ   = ne01;
            ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
                ਅਚਲ bool can_ਵੰਡੋ = htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ne01, can_ਵੰਡੋ ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
                ਸ਼ੁਰੂ_ਪੰਕਤੀ = range.ਸ਼ੁਰੂ;
                ਸਿਰਾ_ਪੰਕਤੀ   = range.ਸ਼ੁਰੂ + range.ਗਿਣਤੀ_2;
            }

            ਅਚਲ uint32_t nrows = ਸਿਰਾ_ਪੰਕਤੀ - ਸ਼ੁਰੂ_ਪੰਕਤੀ;
            uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + nth - 1, &octx->n_ਧਾਗੇ_div);
            src0_nrows_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ(src0_nrows_ਪ੍ਰਤੀ_thread, 32);

            ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
            ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, ਸਿਰਾ_ਪੰਕਤੀ);
            ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) ਜਾਰੀ;

            ਅਚਲ dma_addr_t src0_ਪੰਕਤੀ = src_w->ਅੰਕੜੇ + cur_a * src_w->nb[2];

            ਅਚਲ uint32_t tile_ਮਾਪ = htp_tensor_get_tile_ਮਾਪ(src_w->type);
            ਅਚਲ uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(src_w->type);
            ਅਚਲ uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ਅਚਲ uint32_t tile_ਪੰਕਤੀ_stride = n_k_tiles_w * tile_ਮਾਪ;
            ਅਚਲ uint32_t tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned = n_k_tiles_w * aligned_tile_ਮਾਪ;

            ਅਚਲ uint32_t ct_ਸ਼ੁਰੂ = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ / 32;
            ਅਚਲ uint32_t ct_ਸਿਰਾ   = (src0_ਸਿਰਾ_ਪੰਕਤੀ + 31) / 32;

            uint32_t ਧੱਕੋ_ct = ct_ਸ਼ੁਰੂ;
            ਲਈ (uint32_t d = 0; d < n_prefetch && ਧੱਕੋ_ct < ct_ਸਿਰਾ; d++, ਧੱਕੋ_ct++) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src0_ptr + d * tile_ਪੰਕਤੀ_transfer_ਮਾਪ_aligned, src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                               aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
            }

            ਲਈ (uint32_t ct = ct_ਸ਼ੁਰੂ; ct < ct_ਸਿਰਾ; ct++) {
                ਅਚਲ uint8_t * w_tile = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

                ਪੂਰਨ valid_ਪੰਕਤੀਆਂ = (ਪੂਰਨ)dst->ne[0] - (ਪੂਰਨ)(ct * 32);
                valid_ਪੰਕਤੀਆਂ = MIN(32, MAX(0, valid_ਪੰਕਤੀਆਂ));

                htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                ਲਈ (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, cid);
                    ਅਚਲ ਪੂਰਨ rm1 = ਪੰਕਤੀ_mapping.i1;
                    ਅਚਲ ਪੂਰਨ rm2 = ਪੰਕਤੀ_mapping.i2;

                    ਅਚਲ uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ਅਚਲ uint8_t * ਸੀਮਿਤ act_ਖਾਨਾ = (ਅਚਲ uint8_t *) (act_ਅੰਕੜੇ + (ir1 + rm2 * act->ne[1]) * act_stride);
                    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) (dst->ਅੰਕੜੇ + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_ਪੰਕਤੀ[ct * 32], w_tile, act_ਖਾਨਾ, valid_ਪੰਕਤੀਆਂ, ਨਿਰਾ);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                ਜੇ (ਧੱਕੋ_ct < ct_ਸਿਰਾ) {
                    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(w_tile, src0_ਪੰਕਤੀ + ਧੱਕੋ_ct * tile_ਪੰਕਤੀ_stride),
                                    aligned_tile_ਮਾਪ, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles_w);
                    ਧੱਕੋ_ct++;
                }
            }
        }
    }
}

ਸਥਿਰ ਪੂਰਨ hvx_mm_init_tiled_vec_dot(ਬਣਤਰ htp_mm_context * mmctx, ਗਿਣਤੀ htp_ਅੰਕੜੇ_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_Q4_1:
        ਹਾਲਤ HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            ਵਾਪਸ 0;
        ਹਾਲਤ HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            ਵਾਪਸ 0;
        ਮੂਲ:
            ਵਾਪਸ -1;
    }
}

ਸਥਿਰ ਪੂਰਨ hvx_mm_matmul(ਬਣਤਰ htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, 0);

    ਬਣਤਰ htp_mm_context mmctx_struct = {0};
    ਬਣਤਰ htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;

    ਅਚਲ uint32_t src0_nrows = ne01;
    ਅਚਲ uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t src0_ਪੰਕਤੀ_ਸਿਰਾ   = src0_nrows;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ bool can_ਵੰਡੋ = htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(src0_nrows, can_ਵੰਡੋ ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        src0_ਪੰਕਤੀ_ਸਿਰਾ   = range.ਸ਼ੁਰੂ + range.ਗਿਣਤੀ_2;
    }

    ਜੇ (src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ >= src0_ਪੰਕਤੀ_ਸਿਰਾ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t nrows = src0_ਪੰਕਤੀ_ਸਿਰਾ - src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ = src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ   = src0_ਪੰਕਤੀ_ਸਿਰਾ;

    ਅਚਲ bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_ਪ੍ਰਤੀ_thread  = fastdiv(nrows + octx->n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);
    ਜੇ (is_tiled) {
        mmctx->src0_nrows_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ(mmctx->src0_nrows_ਪ੍ਰਤੀ_thread, 32);
    } ਨਹੀਂਤਾਂ {
        mmctx->src0_nrows_ਪ੍ਰਤੀ_thread += (mmctx->src0_nrows_ਪ੍ਰਤੀ_thread & 1); // round up to even
    }

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = nb01;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = nb1;
    size_t       act_ਪੰਕਤੀ_ਮਾਪ  = nb11;

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਪੰਕਤੀ_ਮਾਪ, 128);

    ਕਾਮਾ_callback_t quant_task_func;
    ਕਾਮਾ_callback_t matmul_ਕੰਮ_func;
    uint32_t n_quant_tasks = 1;
    ਅਚਲ bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    ਜੇ (is_batched) {
        matmul_ਕੰਮ_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } ਨਹੀਂਤਾਂ ਜੇ (act_nrows > 1) {
        matmul_ਕੰਮ_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } ਨਹੀਂਤਾਂ {
        matmul_ਕੰਮ_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    bool need_quant = true;

    ਚੋਣ (kparams->kernel_type) {
        ਹਾਲਤ HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            act_ਪੰਕਤੀ_ਮਾਪ           = kparams->act_ਪੰਕਤੀ_ਮਾਪ;
            ਰੋਕੋ;

        ਹਾਲਤ HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = ਨਿਰਾ;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_ਪੰਕਤੀ_ਮਾਪ           = kparams->act_ਪੰਕਤੀ_ਮਾਪ;
            ਰੋਕੋ;

        ਹਾਲਤ HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = ਨਿਰਾ;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_ਪੰਕਤੀ_ਮਾਪ           = kparams->act_ਪੰਕਤੀ_ਮਾਪ;
            ਰੋਕੋ;

        ਹਾਲਤ HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ:
        ਹਾਲਤ HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ:
        ਮੂਲ: {
            ਜੇ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
            }

            ਅਚਲ uint32_t qk = QK_Q8_0_TILED;
            ਅਚਲ uint32_t nb = (ne10 + qk - 1) / qk;
            ਅਚਲ uint32_t ਕੁੱਲ_nb = act_nrows * nb;

            ਜੇ (act_nrows < octx->n_ਧਾਗੇ && !is_batched) {
                n_quant_tasks = MIN(ਕੁੱਲ_nb, octx->n_ਧਾਗੇ);
                quant_task_func = htp_mm_act_quant_ਖੰਡ_func(src0->type, act->type);
                ਲਈ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_ਪਹਿਲਾ = (ਕੁੱਲ_nb * ith) / n_quant_tasks;
                    uint32_t ib_ਆਖਰੀ  = (ਕੁੱਲ_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_ਪਹਿਲਾ[ith] = ib_ਪਹਿਲਾ;
                    mmctx->quant_ib_ਆਖਰੀ[ith]  = ib_ਆਖਰੀ;
                    mmctx->quant_r[ith]        = ib_ਪਹਿਲਾ / nb;
                    mmctx->quant_c[ith]        = ib_ਪਹਿਲਾ % nb;
                }
            } ਨਹੀਂਤਾਂ {
                n_quant_tasks = MIN(act_nrows, octx->n_ਧਾਗੇ);
                quant_task_func = htp_mm_act_quant_ਪੰਕਤੀ_func(src0->type, act->type);
            }
            act_ਪੰਕਤੀ_ਮਾਪ = kparams->act_ਪੰਕਤੀ_ਮਾਪ;
            ਰੋਕੋ;
        }
    }

    ਅਚਲ uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ਅਚਲ uint32_t m_layout_ਪੰਕਤੀਆਂ = m_chunk;

    ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_ਪੰਕਤੀਆਂ, octx->n_ਧਾਗੇ,
                                 dst_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, act_ਪੰਕਤੀ_ਮਾਪ, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    ਜੇ (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ) {
        mmctx->vtcm_act_ਮਾਪ_ਪ੍ਰਤੀ_thread = L.act_ਬਾਈਟਾਂ;
    } ਨਹੀਂਤਾਂ {
        mmctx->vtcm_act_ਮਾਪ_ਪ੍ਰਤੀ_thread = fastdiv(L.act_ਬਾਈਟਾਂ, &octx->n_ਧਾਗੇ_div);
    }

    mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = fastdiv(L.src0_ਬਾਈਟਾਂ, &octx->n_ਧਾਗੇ_div);
    mmctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = fastdiv(L.dst_ਬਾਈਟਾਂ, &octx->n_ਧਾਗੇ_div);

    ਅਚਲ size_t vtcm_ਮਾਪ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;

    FARF(ਉੱਚਾ_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_ਬਾਈਟਾਂ, L.act_ਬਾਈਟਾਂ, L.dst_ਬਾਈਟਾਂ, vtcm_ਮਾਪ);

    FARF(ਉੱਚਾ_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->ਅੰਕੜੇ, act->ਅੰਕੜੇ, dst->ਅੰਕੜੇ);

    ਜੇ (octx->ctx->vtcm_ਮਾਪ < vtcm_ਮਾਪ) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_ਮਾਪ, vtcm_ਮਾਪ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    uint8_t * ਅਚਲ ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_ਪੰਕਤੀ_ਮਾਪ_padded;
    mmctx->vtcm_act_stride  = act_ਪੰਕਤੀ_ਮਾਪ;
    ਜੇ (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ) {
        mmctx->vtcm_act_raw_stride = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), QK_Q8_0_TILED * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    } ਨਹੀਂਤਾਂ ਜੇ (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
    } ਨਹੀਂਤਾਂ {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ਜੇ (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        ਲਈ (uint32_t m_ਸ਼ੁਰੂ = 0; m_ਸ਼ੁਰੂ < act_nrows; m_ਸ਼ੁਰੂ += m_chunk) {
            ਅਚਲ uint32_t cur_m_ਪੰਕਤੀਆਂ = MIN(act_nrows - m_ਸ਼ੁਰੂ, m_chunk);
            mmctx->cur_m_ਸ਼ੁਰੂ = m_ਸ਼ੁਰੂ;
            mmctx->cur_m_ਪੰਕਤੀਆਂ  = cur_m_ਪੰਕਤੀਆਂ;

            ਜੇ (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_ਸ਼ੁਰੂ, cur_m_ਪੰਕਤੀਆਂ);

                ਅਚਲ uint32_t qk = QK_Q8_0_TILED;
                ਅਚਲ uint32_t nb = (ne10 + qk - 1) / qk;
                ਅਚਲ uint32_t ਕੁੱਲ_nb = cur_m_ਪੰਕਤੀਆਂ * nb;
                uint32_t quant_tasks;
                work_ਕਤਾਰ_func_t q_func;
                ਜੇ (cur_m_ਪੰਕਤੀਆਂ < octx->n_ਧਾਗੇ && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ)) {
                    quant_tasks = MIN(ਕੁੱਲ_nb, octx->n_ਧਾਗੇ);
                    q_func = htp_mm_act_quant_ਖੰਡ_func(src0->type, act->type);
                    ਲਈ (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_ਪਹਿਲਾ = (ਕੁੱਲ_nb * ith) / quant_tasks;
                        uint32_t ib_ਆਖਰੀ  = (ਕੁੱਲ_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_ਪਹਿਲਾ[ith] = ib_ਪਹਿਲਾ;
                        mmctx->quant_ib_ਆਖਰੀ[ith]  = ib_ਆਖਰੀ;
                        mmctx->quant_r[ith]        = ib_ਪਹਿਲਾ / nb;
                        mmctx->quant_c[ith]        = ib_ਪਹਿਲਾ % nb;
                    }
                } ਨਹੀਂਤਾਂ {
                    quant_tasks = MIN(cur_m_ਪੰਕਤੀਆਂ, octx->n_ਧਾਗੇ);
                    q_func = quant_task_func;
                    mmctx->n_quant_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (cur_m_ਪੰਕਤੀਆਂ + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, q_func, mmctx, quant_tasks);
            } ਨਹੀਂਤਾਂ {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_ਸ਼ੁਰੂ, cur_m_ਪੰਕਤੀਆਂ);
            }

            work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, matmul_ਕੰਮ_func, mmctx, octx->n_ਧਾਗੇ);
        }
    } ਨਹੀਂਤਾਂ {
        mmctx->cur_m_ਸ਼ੁਰੂ = 0;
        mmctx->cur_m_ਪੰਕਤੀਆਂ  = act_nrows;

        ਜੇ (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, quant_task_func, mmctx, n_quant_tasks);
        } ਨਹੀਂਤਾਂ {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, matmul_ਕੰਮ_func, mmctx, octx->n_ਧਾਗੇ);
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਕਾਮਾ_loop(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ task_id = i; task_id < (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) state->n_tasks; task_id += n) {
        ਪੂਰਨ ਸ਼ੁਰੂ = task_id * state->n_tiles_ਪ੍ਰਤੀ_task;
        ਪੂਰਨ ਸਿਰਾ   = hex_smin(ਸ਼ੁਰੂ + state->n_tiles_ਪ੍ਰਤੀ_task, state->n_tot_tiles);
        state->task_fn(state, ਸ਼ੁਰੂ, ਸਿਰਾ);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

ਸਥਿਰ ਖਾਲੀ transfer_output_chunk_ਕਾਮਾ_fn(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ *ਅੰਕੜੇ) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) ਅੰਕੜੇ;

    ਬਣਤਰ htp_thread_trace * tr = &st->traces[i];

    ਪੂਰਨ ਸ਼ੁਰੂ_chunk_idx = i * st->n_chunks_ਪ੍ਰਤੀ_task;
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_O_PROC, ਸ਼ੁਰੂ_chunk_idx);

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ task_id = i; task_id < (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)st->n_tasks; task_id += n) {
        ਪੂਰਨ    chunk_idx  = task_id * st->n_chunks_ਪ੍ਰਤੀ_task;
        size_t chunk_ਮਾਪ = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_ਪ੍ਰਤੀ_task);

        ਦਸ਼ਮਲਵ        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ਅਚਲ ਦਸ਼ਮਲਵ  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : ਨਿਰਾ;
        transfer_output_chunk_fp16_ਮੰਜ਼ਿਲ_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_ਮਾਪ, st->n_ਖਾਨੇ_ਗਿਣਤੀ, st->dst_stride, st->src2_stride, st->dst_ਖਾਨੇ_ਗਿਣਤੀ);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, ਸ਼ੁਰੂ_chunk_idx);
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਣਤਰ htp_context            * ctx;
    ਬਣਤਰ htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    ਦਸ਼ਮਲਵ                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_ਪ੍ਰਤੀ_task;
    uint32_t                        k_ਖੰਡ;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    uint32_t                        dma_ਕਦਮ_ਪੰਕਤੀਆਂ;
    uint32_t                        dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ;
    uint32_t                        act_elem_ਮਾਪ;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਣਤਰ htp_context            * ctx;
    ਬਣਤਰ htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    ਦਸ਼ਮਲਵ                         * vtcm_f32_act;
    uint32_t                        n_ਪੰਕਤੀਆਂ;
    uint32_t                        k_ਖੰਡ;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_ਖਾਨਾ_chunks;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ           n_ਧਾਗੇ_div;
    size_t                          vtcm_f32_act_ਬਾਈਟਾਂ;
    uint32_t                        dma_ਕਦਮ_ਪੰਕਤੀਆਂ;
    uint32_t                        dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ;
} activation_transfer_ਖਾਨਾ_chunk_state_t;

ਸਥਿਰ ਖਾਲੀ transfer_activation_chunk_fp32_ਮੰਜ਼ਿਲ_fp16_dma_pipelined_ਖਾਨਾ_chunk(
        dma_ਕਤਾਰ *dma_q,
        __fp16 *ਸੀਮਿਤ vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_ਪੰਕਤੀਆਂ,
        uint32_t k_ਖੰਡ,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_ਪਹਿਲਾ,
        uint32_t c_len,
        ਦਸ਼ਮਲਵ *thread_f32_act,
        ਬਣਤਰ htp_thread_trace *tr,
        uint32_t dma_ਕਦਮ_ਪੰਕਤੀਆਂ,
        uint32_t dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ) {

    ਅਚਲ uint32_t R = dma_ਕਦਮ_ਪੰਕਤੀਆਂ;
    ਅਚਲ uint32_t n_ਪੰਕਤੀਆਂ_padded = hex_align_ਉੱਪਰ(n_ਪੰਕਤੀਆਂ, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);

    ਅਚਲ uint32_t n_ਕਦਮਾਂ = n_ਪੰਕਤੀਆਂ_padded >> dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ;

    // Push step 0
    ਜੇ (n_ਕਦਮਾਂ > 0 && n_ਪੰਕਤੀਆਂ > 0) {
        uint32_t nrows_ਮੰਜ਼ਿਲ_fetch = hex_smin(n_ਪੰਕਤੀਆਂ, R);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(thread_f32_act, act_dma_addr + (size_t) c_ਪਹਿਲਾ * ਆਕਾਰ(ਦਸ਼ਮਲਵ)),
                       c_len * ਆਕਾਰ(ਦਸ਼ਮਲਵ), k_stride * ਆਕਾਰ(ਦਸ਼ਮਲਵ), k_chunk_valid * ਆਕਾਰ(ਦਸ਼ਮਲਵ), nrows_ਮੰਜ਼ਿਲ_fetch);
    }
    // Push step 1
    ਜੇ (n_ਕਦਮਾਂ > 1) {
        uint32_t ਅਗਲਾ_r = R * 1;
        ਜੇ (ਅਗਲਾ_r < n_ਪੰਕਤੀਆਂ) {
            uint32_t nrows_ਮੰਜ਼ਿਲ_fetch = hex_smin(n_ਪੰਕਤੀਆਂ - ਅਗਲਾ_r, R);
            ਦਸ਼ਮਲਵ *ਅਗਲਾ_buf = thread_f32_act + 1 * R * c_len;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ਅਗਲਾ_buf, act_dma_addr + ((size_t) ਅਗਲਾ_r * k_stride + c_ਪਹਿਲਾ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ)),
                           c_len * ਆਕਾਰ(ਦਸ਼ਮਲਵ), k_stride * ਆਕਾਰ(ਦਸ਼ਮਲਵ), k_chunk_valid * ਆਕਾਰ(ਦਸ਼ਮਲਵ), nrows_ਮੰਜ਼ਿਲ_fetch);
        }
    }
    ਲਈ (uint32_t s = 0; s < n_ਕਦਮਾਂ; ++s) {
        uint32_t r = s << dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ;
        ਦਸ਼ਮਲਵ *curr_buf = thread_f32_act;

        ਜੇ (r < n_ਪੰਕਤੀਆਂ) {
            curr_buf = (ਦਸ਼ਮਲਵ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
        }

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        ਲਈ (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t ਪੰਕਤੀ_idx = r + (p << 1);
            ਦਸ਼ਮਲਵ *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((ਪੰਕਤੀ_idx + 0) < n_ਪੰਕਤੀਆਂ);
            bool r1_valid = ((ਪੰਕਤੀ_idx + 1) < n_ਪੰਕਤੀਆਂ);

            transfer_activation_ਪੰਕਤੀ_pair_fp32_ਮੰਜ਼ਿਲ_fp16_ਖਾਨਾ_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, ਪੰਕਤੀ_idx, k_ਖੰਡ, c_ਪਹਿਲਾ, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t ਅਗਲਾ_s = s + 2;
        uint32_t ਅਗਲਾ_r = ਅਗਲਾ_s << dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ;
        ਜੇ (ਅਗਲਾ_r < n_ਪੰਕਤੀਆਂ) {
            uint32_t nrows_ਮੰਜ਼ਿਲ_fetch = hex_smin(n_ਪੰਕਤੀਆਂ - ਅਗਲਾ_r, R);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(curr_buf, act_dma_addr + ((size_t) ਅਗਲਾ_r * k_stride + c_ਪਹਿਲਾ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ)),
                           c_len * ਆਕਾਰ(ਦਸ਼ਮਲਵ), k_stride * ਆਕਾਰ(ਦਸ਼ਮਲਵ), k_chunk_valid * ਆਕਾਰ(ਦਸ਼ਮਲਵ), nrows_ਮੰਜ਼ਿਲ_fetch);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ transfer_activation_chunk_ਖਾਨਾ_chunk_ਕਾਮਾ_fn(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ *ਅੰਕੜੇ) {
    activation_transfer_ਖਾਨਾ_chunk_state_t *st = (activation_transfer_ਖਾਨਾ_chunk_state_t *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &st->traces[i];

    uint32_t n_ਖੰਡਾਂ = st->k_ਖੰਡ / 32;
    uint32_t b_ਪਹਿਲਾ = fastdiv(n_ਖੰਡਾਂ * i, &st->n_ਧਾਗੇ_div);
    uint32_t b_ਆਖਰੀ  = fastdiv(n_ਖੰਡਾਂ * (i + 1), &st->n_ਧਾਗੇ_div);
    uint32_t c_ਪਹਿਲਾ = b_ਪਹਿਲਾ * 32;
    uint32_t c_ਆਖਰੀ = b_ਆਖਰੀ * 32;
    uint32_t c_len = c_ਆਖਰੀ - c_ਪਹਿਲਾ;

    ਜੇ (c_len == 0) {
        ਵਾਪਸ;
    }

    uint32_t k_chunk_valid = 0;
    ਜੇ (st->k_valid > c_ਪਹਿਲਾ) {
        k_chunk_valid = hex_smin(st->k_valid, c_ਆਖਰੀ) - c_ਪਹਿਲਾ;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_ਬਾਈਟਾਂ = hex_align_ਥੱਲੇ(fastdiv(st->vtcm_f32_act_ਬਾਈਟਾਂ, &st->n_ਧਾਗੇ_div), 128);
    ਦਸ਼ਮਲਵ *thread_f32_act = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)st->vtcm_f32_act + i * thread_scratch_ਬਾਈਟਾਂ);

    transfer_activation_chunk_fp32_ਮੰਜ਼ਿਲ_fp16_dma_pipelined_ਖਾਨਾ_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_ਪੰਕਤੀਆਂ, st->k_ਖੰਡ, st->k_stride, k_chunk_valid,
        c_ਪਹਿਲਾ, c_len, thread_f32_act, tr, st->dma_ਕਦਮ_ਪੰਕਤੀਆਂ, st->dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ
    );
}

ਸਥਿਰ ਖਾਲੀ transfer_activation_chunk_ਮੰਜ਼ਿਲ_fp16_dma_pipelined(
        dma_ਕਤਾਰ *dma_q,
        __fp16 *ਸੀਮਿਤ vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_ਪੰਕਤੀਆਂ,
        uint32_t k_ਖੰਡ,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_ਮਾਪ,
        ਬਣਤਰ htp_thread_trace *tr,
        uint32_t dma_ਕਦਮ_ਪੰਕਤੀਆਂ,
        uint32_t dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ) {

    ਅਚਲ uint32_t R = dma_ਕਦਮ_ਪੰਕਤੀਆਂ;
    ਅਚਲ uint32_t n_ਪੰਕਤੀਆਂ_padded = hex_align_ਉੱਪਰ(n_ਪੰਕਤੀਆਂ, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);

    ਅਚਲ uint32_t n_ਕਦਮਾਂ = n_ਪੰਕਤੀਆਂ_padded >> dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ;

    ਅਚਲ bool act_is_f16 = (act_elem_ਮਾਪ == ਆਕਾਰ(__fp16));
    ਅਚਲ size_t ਪੰਕਤੀ_ਬਾਈਟਾਂ = (size_t) k_ਖੰਡ * act_elem_ਮਾਪ;  // staging row (DMA dst stride)
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ = (size_t) k_stride * act_elem_ਮਾਪ;  // DDR row (DMA src stride)
    ਅਚਲ size_t ਚੌੜਾਈ_ਬਾਈਟਾਂ = (size_t) k_valid * act_elem_ਮਾਪ;

    // Push step 0
    ਜੇ (n_ਕਦਮਾਂ > 0 && n_ਪੰਕਤੀਆਂ > 0) {
        uint32_t nrows_ਮੰਜ਼ਿਲ_fetch = hex_smin(n_ਪੰਕਤੀਆਂ, R);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(thread_act, act_dma_addr),
                       ਪੰਕਤੀ_ਬਾਈਟਾਂ, src_ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਚੌੜਾਈ_ਬਾਈਟਾਂ, nrows_ਮੰਜ਼ਿਲ_fetch);
    }
    // Push step 1 (if valid)
    ਜੇ (n_ਕਦਮਾਂ > 1) {
        uint32_t ਅਗਲਾ_r = R * 1;
        ਜੇ (ਅਗਲਾ_r < n_ਪੰਕਤੀਆਂ) {
            uint32_t nrows_ਮੰਜ਼ਿਲ_fetch = hex_smin(n_ਪੰਕਤੀਆਂ - ਅਗਲਾ_r, R);
            uint8_t *ਅਗਲਾ_buf = thread_act + 1 * R * ਪੰਕਤੀ_ਬਾਈਟਾਂ;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ਅਗਲਾ_buf, act_dma_addr + (size_t) ਅਗਲਾ_r * src_ਪੰਕਤੀ_ਬਾਈਟਾਂ),
                           ਪੰਕਤੀ_ਬਾਈਟਾਂ, src_ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਚੌੜਾਈ_ਬਾਈਟਾਂ, nrows_ਮੰਜ਼ਿਲ_fetch);
        }
    }
    ਲਈ (uint32_t s = 0; s < n_ਕਦਮਾਂ; ++s) {
        uint32_t r = s << dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ;
        uint8_t *curr_buf = thread_act;

        ਜੇ (r < n_ਪੰਕਤੀਆਂ) {
            curr_buf = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
        }

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        ਜੇ (act_is_f16) {
            ਲਈ (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t ਪੰਕਤੀ_idx = r + (p << 1);
                ਅਚਲ __fp16 *pair_buf = (ਅਚਲ __fp16 *) (curr_buf + (p << 1) * ਪੰਕਤੀ_ਬਾਈਟਾਂ);
                bool r0_valid = ((ਪੰਕਤੀ_idx + 0) < n_ਪੰਕਤੀਆਂ);
                bool r1_valid = ((ਪੰਕਤੀ_idx + 1) < n_ਪੰਕਤੀਆਂ);

                transfer_activation_ਪੰਕਤੀ_pair_f16_ਮੰਜ਼ਿਲ_f16(vtcm_dst, pair_buf, pair_buf + k_ਖੰਡ, ਪੰਕਤੀ_idx, k_ਖੰਡ, k_valid, r0_valid, r1_valid);
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t ਪੰਕਤੀ_idx = r + (p << 1);
                ਅਚਲ ਦਸ਼ਮਲਵ *pair_buf = (ਅਚਲ ਦਸ਼ਮਲਵ *) (curr_buf + (p << 1) * ਪੰਕਤੀ_ਬਾਈਟਾਂ);
                bool r0_valid = ((ਪੰਕਤੀ_idx + 0) < n_ਪੰਕਤੀਆਂ);
                bool r1_valid = ((ਪੰਕਤੀ_idx + 1) < n_ਪੰਕਤੀਆਂ);

                transfer_activation_ਪੰਕਤੀ_pair_fp32_ਮੰਜ਼ਿਲ_fp16(vtcm_dst, pair_buf, pair_buf + k_ਖੰਡ, ਪੰਕਤੀ_idx, k_ਖੰਡ, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t ਅਗਲਾ_s = s + 2;
        uint32_t ਅਗਲਾ_r = ਅਗਲਾ_s << dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ;
        ਜੇ (ਅਗਲਾ_r < n_ਪੰਕਤੀਆਂ) {
            uint32_t nrows_ਮੰਜ਼ਿਲ_fetch = hex_smin(n_ਪੰਕਤੀਆਂ - ਅਗਲਾ_r, R);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(curr_buf, act_dma_addr + (size_t) ਅਗਲਾ_r * src_ਪੰਕਤੀ_ਬਾਈਟਾਂ),
                           ਪੰਕਤੀ_ਬਾਈਟਾਂ, src_ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਚੌੜਾਈ_ਬਾਈਟਾਂ, nrows_ਮੰਜ਼ਿਲ_fetch);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ transfer_activation_chunk_ਕਾਮਾ_fn(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ *ਅੰਕੜੇ) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) ਅੰਕੜੇ;

    ਬਣਤਰ htp_thread_trace * tr = &st->traces[i];

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ task_id = i; task_id < (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)st->n_tasks; task_id += n) {
        ਪੂਰਨ    chunk_idx  = task_id * st->n_chunks_ਪ੍ਰਤੀ_task;
        size_t chunk_ਮਾਪ = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_ਪ੍ਰਤੀ_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_ਖੰਡ;
        ਅਚਲ dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_ਮਾਪ;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
        transfer_activation_chunk_ਮੰਜ਼ਿਲ_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_ਮਾਪ, st->k_ਖੰਡ, st->k_stride, st->k_valid, thread_act, st->act_elem_ਮਾਪ, tr, st->dma_ਕਦਮ_ਪੰਕਤੀਆਂ, st->dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ
        );
    }
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਣਤਰ htp_thread_trace       * traces;
    ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ;
    __fp16                        * dst;
    ਅਚਲ ਦਸ਼ਮਲਵ                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_ਪ੍ਰਤੀ_task;
    uint32_t                        k_ਖੰਡ;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਣਤਰ htp_thread_trace       * traces;
    ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ;
    ਅਚਲ __fp16                  * vtcm_src;
    ਦਸ਼ਮਲਵ                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_ਪ੍ਰਤੀ_task;
    uint32_t                        n_ਖਾਨੇ_ਗਿਣਤੀ;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

ਸਥਿਰ ਖਾਲੀ transfer_activation_chunk_gathered_ਕਾਮਾ_fn(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ *ਅੰਕੜੇ) {
    activation_transfer_gathered_task_state_t *st = ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &st->traces[i];
    ਪੂਰਨ chunk_idx      = i;
    ਪੂਰਨ chunk_ਮਾਪ     = st->n_chunks_ਪ੍ਰਤੀ_task;
    ਪੂਰਨ vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ = chunk_idx * chunk_ਮਾਪ;
    ਪੂਰਨ ਸ਼ੁਰੂ_ਪੰਕਤੀ      = st->ਸ਼ੁਰੂ_ਪੰਕਤੀ + vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਪੂਰਨ n_ਪੰਕਤੀਆਂ         = hex_smin(st->cne1 - ਸ਼ੁਰੂ_ਪੰਕਤੀ, chunk_ਮਾਪ);
    ਜੇ (n_ਪੰਕਤੀਆਂ > 0) {
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_ਮੰਜ਼ਿਲ_fp16_gathered(
            st->dst, st->src, ਸ਼ੁਰੂ_ਪੰਕਤੀ, vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ, n_ਪੰਕਤੀਆਂ, st->k_ਖੰਡ,
            st->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ਸਥਿਰ ਖਾਲੀ transfer_activation_chunk_gathered_ਕਾਮਾ_flat_fn(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ *ਅੰਕੜੇ) {
    activation_transfer_gathered_task_state_t *st = ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &st->traces[i];
    ਪੂਰਨ chunk_idx = i;
    ਪੂਰਨ chunk_ਮਾਪ = st->n_chunks_ਪ੍ਰਤੀ_task;
    ਪੂਰਨ vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ = chunk_idx * chunk_ਮਾਪ;
    ਪੂਰਨ ਸ਼ੁਰੂ_ਪੰਕਤੀ = st->ਸ਼ੁਰੂ_ਪੰਕਤੀ + vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਪੂਰਨ n_ਪੰਕਤੀਆਂ = hex_smin(st->cne1 - ਸ਼ੁਰੂ_ਪੰਕਤੀ, chunk_ਮਾਪ);
    ਜੇ (n_ਪੰਕਤੀਆਂ > 0) {
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_ਮੰਜ਼ਿਲ_fp16_gathered_flat(
            st->dst, st->src, ਸ਼ੁਰੂ_ਪੰਕਤੀ, vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ, n_ਪੰਕਤੀਆਂ, st->k_ਖੰਡ,
            st->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ਸਥਿਰ ਖਾਲੀ transfer_output_chunk_scattered_ਕਾਮਾ_fn(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ *ਅੰਕੜੇ) {
    output_transfer_scattered_task_state_t *st = ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &st->traces[i];
    ਪੂਰਨ chunk_idx = i;
    ਪੂਰਨ chunk_ਮਾਪ = st->n_chunks_ਪ੍ਰਤੀ_task;
    ਪੂਰਨ vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ = chunk_idx * chunk_ਮਾਪ;
    ਪੂਰਨ ਸ਼ੁਰੂ_ਪੰਕਤੀ = st->ਸ਼ੁਰੂ_ਪੰਕਤੀ + vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਪੂਰਨ n_ਪੰਕਤੀਆਂ = hex_smin(st->cne1 - ਸ਼ੁਰੂ_ਪੰਕਤੀ, chunk_ਮਾਪ);
    ਜੇ (n_ਪੰਕਤੀਆਂ > 0) {
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_ਮੰਜ਼ਿਲ_fp32_scattered(
            st->dst, st->vtcm_src, ਸ਼ੁਰੂ_ਪੰਕਤੀ, vtcm_ਸ਼ੁਰੂ_ਪੰਕਤੀ, n_ਪੰਕਤੀਆਂ, st->n_ਖਾਨੇ_ਗਿਣਤੀ,
            st->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

ਸਥਿਰ ਸਤਰਅੰਦਰ tiled_dequant_task_fn_t hmx_get_dequant_ਕਾਮਾ_fn(ਪੂਰਨ ਭਾਰ_type, bool is_tiled) {
    ਚੋਣ (ਭਾਰ_type) {
        ਹਾਲਤ HTP_TYPE_Q4_0:   ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q4_0;
        ਹਾਲਤ HTP_TYPE_IQ4_NL: ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_iq4_nl;
        ਹਾਲਤ HTP_TYPE_Q4_1:
        ਹਾਲਤ HTP_TYPE_Q4_K:   ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q4_1;
        ਹਾਲਤ HTP_TYPE_MXFP4:  ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_mxfp4;
        ਹਾਲਤ HTP_TYPE_Q8_0:   ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q8_0;
        ਹਾਲਤ HTP_TYPE_Q5_K:   ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q5_k;
        ਹਾਲਤ HTP_TYPE_Q6_K:   ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q6_k;
        ਹਾਲਤ HTP_TYPE_Q3_K:   ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q3_k;
        ਹਾਲਤ HTP_TYPE_Q2_K:   ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q2_k;
        ਹਾਲਤ HTP_TYPE_Q1_0:   ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q1_0;
        ਹਾਲਤ HTP_TYPE_Q2_0:   ਵਾਪਸ dequantize_tiled_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_task_q2_0;
        ਹਾਲਤ HTP_TYPE_F16:    ਵਾਪਸ is_tiled ? ਨਿਰਾ : convert_f16_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_tiles_task;
        ਹਾਲਤ HTP_TYPE_BF16:   ਵਾਪਸ convert_bf16_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_tiles_task;
        ਹਾਲਤ HTP_TYPE_F32:    ਵਾਪਸ quantize_f32_ਭਾਰ_ਮੰਜ਼ਿਲ_fp16_tiles_task;
        ਮੂਲ:              ਵਾਪਸ ਨਿਰਾ;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_tiled_ਭਾਰ_chunk_ਮੰਜ਼ਿਲ_fp16_tiles(
        ਬਣਤਰ htp_context *ctx, __fp16 *vtcm_dst,
        ਅਚਲ ਖਾਲੀ *ਭਾਰ_src_ddr,
        ਪੂਰਨ n_ਖਾਨੇ_ਗਿਣਤੀ, ਪੂਰਨ k_ਖੰਡ,
        size_t ਪੰਕਤੀ_stride, ਪੂਰਨ ਭਾਰ_type,
        ਪੂਰਨ n_k_tiles, ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_ਕਾਮਾ_fn, ਪੂਰਨ n_ਧਾਗੇ) {

    ਜੇ (!dequant_ਕਾਮਾ_fn) {
        ਵਾਪਸ;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(k_ਖੰਡ % HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ == 0);

    ਅਚਲ size_t n_ਖਾਨਾ_tiles = hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ, HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);
    ਅਚਲ uint32_t n_ਭਾਰ_tiles = k_ਖੰਡ / htp_tensor_get_tile_k(ਭਾਰ_type);
    ਅਚਲ size_t n_tot_tiles = n_ਖਾਨਾ_tiles * n_ਭਾਰ_tiles;

    size_t n_tiles_ਪ੍ਰਤੀ_task = (n_ਧਾਗੇ == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_ਧਾਗੇ);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_ਪ੍ਰਤੀ_task - 1) / n_tiles_ਪ੍ਰਤੀ_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_ਪ੍ਰਤੀ_task = n_tiles_ਪ੍ਰਤੀ_task;
    state.dst              = vtcm_dst;
    state.src              = (ਅਚਲ uint8_t *)ਭਾਰ_src_ddr;
    state.task_fn          = dequant_ਕਾਮਾ_fn;
    state.n_ਖਾਨੇ_ਗਿਣਤੀ           = n_ਖਾਨੇ_ਗਿਣਤੀ;
    state.k_ਖੰਡ          = k_ਖੰਡ;
    state.ਪੰਕਤੀ_stride       = ਪੰਕਤੀ_stride;
    state.ਭਾਰ_type      = ਭਾਰ_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_ਭਾਰ_tiles   = n_ਭਾਰ_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_ਮਾਪ = htp_tensor_get_tile_ਮਾਪ(ਭਾਰ_type);
    state.aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(ਭਾਰ_type);

    ਜੇ (state.n_tasks == 1 || n_ਧਾਗੇ == 1) {
        dequantize_tiled_ਕਾਮਾ_loop(1, 0, &state);
    } ਨਹੀਂਤਾਂ {
        ਪੂਰਨ n_tasks = hex_smin((ਪੂਰਨ) state.n_tasks, n_ਧਾਗੇ);
        ਕਾਮਾ_pool_ਲੜੀਵਾਰ_func(ctx->ਕਾਮਾ_pool, dequantize_tiled_ਕਾਮਾ_loop, &state, n_tasks);
    }
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਣਤਰ htp_context      * ctx;
    ਬਣਤਰ htp_thread_trace * traces;
    ਦਸ਼ਮਲਵ                   * dst;
    ਅਚਲ __fp16            * vtcm_src;
    ਅਚਲ ਦਸ਼ਮਲਵ             * src2;
    uint32_t                  n_ਪੰਕਤੀਆਂ;
    uint32_t                  n_ਖਾਨੇ_ਗਿਣਤੀ;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_ਖਾਨੇ_ਗਿਣਤੀ;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ     n_ਧਾਗੇ_div;
} output_transfer_ਖਾਨਾ_chunk_state_t;

ਸਥਿਰ ਖਾਲੀ transfer_output_chunk_ਖਾਨਾ_chunk_ਕਾਮਾ_fn(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ *ਅੰਕੜੇ) {
    (ਖਾਲੀ) n;
    output_transfer_ਖਾਨਾ_chunk_state_t *st = (output_transfer_ਖਾਨਾ_chunk_state_t *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_ਖੰਡਾਂ = hmx_ceil_div(st->n_ਖਾਨੇ_ਗਿਣਤੀ, 32);
    uint32_t b_ਪਹਿਲਾ  = fastdiv(n_ਖੰਡਾਂ * i, &st->n_ਧਾਗੇ_div);
    uint32_t b_ਆਖਰੀ   = fastdiv(n_ਖੰਡਾਂ * (i + 1), &st->n_ਧਾਗੇ_div);
    uint32_t c_ਪਹਿਲਾ  = b_ਪਹਿਲਾ * 32;
    uint32_t c_ਆਖਰੀ   = b_ਆਖਰੀ * 32;
    uint32_t c_len    = c_ਆਖਰੀ - c_ਪਹਿਲਾ;

    ਜੇ (c_len == 0) ਵਾਪਸ;

    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_O_PROC, c_ਪਹਿਲਾ);

    ਅਚਲ __fp16 *vtcm_src = st->vtcm_src + b_ਪਹਿਲਾ * HTP_MM_HMX_TILE_N_ELMS;
    ਅਚਲ ਦਸ਼ਮਲਵ *src2      = st->src2 ? (st->src2 + c_ਪਹਿਲਾ) : ਨਿਰਾ;
    ਦਸ਼ਮਲਵ *dst             = st->dst + c_ਪਹਿਲਾ;

    ਪੂਰਨ chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ = (ਪੂਰਨ)st->dst_ਖਾਨੇ_ਗਿਣਤੀ - (ਪੂਰਨ)c_ਪਹਿਲਾ;
    ਜੇ (chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ > 0) {
        transfer_output_chunk_fp16_ਮੰਜ਼ਿਲ_fp32_ਖਾਨਾ_chunk(
            dst, src2, vtcm_src, 0, st->n_ਪੰਕਤੀਆਂ, c_len, st->n_ਖਾਨੇ_ਗਿਣਤੀ,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_ਪਹਿਲਾ);
}

ਸਥਿਰ ਖਾਲੀ transfer_output_chunk_threaded(ਬਣਤਰ htp_context *ctx, ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਦਸ਼ਮਲਵ *src2, ਅਚਲ __fp16 *vtcm_src,
                                              ਪੂਰਨ n_ਪੰਕਤੀਆਂ, ਪੂਰਨ n_ਖਾਨੇ_ਗਿਣਤੀ, ਪੂਰਨ dst_stride, uint32_t src2_stride, ਪੂਰਨ dst_ਖਾਨੇ_ਗਿਣਤੀ, ਪੂਰਨ n_ਧਾਗੇ) {
    ਜੇ (n_ਪੰਕਤੀਆਂ <= 0) ਵਾਪਸ;

    uint32_t n_ਖੰਡਾਂ = hmx_ceil_div((uint32_t) n_ਖਾਨੇ_ਗਿਣਤੀ, 32);
    ਜੇ (n_ਧਾਗੇ > 1 && n_ਖੰਡਾਂ >= (uint32_t)n_ਧਾਗੇ) {
        ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ n_ਧਾਗੇ_div = (n_ਧਾਗੇ == (ਪੂਰਨ)ctx->n_ਧਾਗੇ) ? ctx->n_ਧਾਗੇ_div : init_fastdiv_ਮੁੱਲਾਂ(n_ਧਾਗੇ);
        output_transfer_ਖਾਨਾ_chunk_state_t ਖਾਨਾ_state;
        ਖਾਨਾ_state.dst = dst;
        ਖਾਨਾ_state.src2 = src2;
        ਖਾਨਾ_state.vtcm_src = vtcm_src;
        ਖਾਨਾ_state.n_ਪੰਕਤੀਆਂ = (uint32_t)n_ਪੰਕਤੀਆਂ;
        ਖਾਨਾ_state.n_ਖਾਨੇ_ਗਿਣਤੀ = (uint32_t)n_ਖਾਨੇ_ਗਿਣਤੀ;
        ਖਾਨਾ_state.dst_stride = (uint32_t)dst_stride;
        ਖਾਨਾ_state.src2_stride = src2_stride;
        ਖਾਨਾ_state.dst_ਖਾਨੇ_ਗਿਣਤੀ = (uint32_t)dst_ਖਾਨੇ_ਗਿਣਤੀ;
        ਖਾਨਾ_state.n_ਧਾਗੇ_div = n_ਧਾਗੇ_div;
        ਖਾਨਾ_state.traces = ctx->trace;
        ਖਾਨਾ_state.ctx = ctx;

        ਕਾਮਾ_pool_ਲੜੀਵਾਰ_func(ctx->ਕਾਮਾ_pool, transfer_output_chunk_ਖਾਨਾ_chunk_ਕਾਮਾ_fn, &ਖਾਨਾ_state, n_ਧਾਗੇ);
        ਵਾਪਸ;
    }

    size_t n_tot_chunks      = n_ਪੰਕਤੀਆਂ;
    size_t n_chunks_ਪ੍ਰਤੀ_task = (n_ਧਾਗੇ == 1) ? n_tot_chunks : hmx_ceil_div(n_ਪੰਕਤੀਆਂ, n_ਧਾਗੇ);
    n_chunks_ਪ੍ਰਤੀ_task        = hex_align_ਉੱਪਰ(n_chunks_ਪ੍ਰਤੀ_task, 2);

    ਪੂਰਨ actual_ਧਾਗੇ = hmx_ceil_div(n_ਪੰਕਤੀਆਂ, n_chunks_ਪ੍ਰਤੀ_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_ਧਾਗੇ;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_ਪ੍ਰਤੀ_task = n_chunks_ਪ੍ਰਤੀ_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_ਖਾਨੇ_ਗਿਣਤੀ            = n_ਖਾਨੇ_ਗਿਣਤੀ;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_ਖਾਨੇ_ਗਿਣਤੀ          = dst_ਖਾਨੇ_ਗਿਣਤੀ;
    state.traces            = ctx->trace;

    ਜੇ (actual_ਧਾਗੇ <= 1) {
        transfer_output_chunk_ਕਾਮਾ_fn(1, 0, &state);
    } ਨਹੀਂਤਾਂ {
        ਕਾਮਾ_pool_ਲੜੀਵਾਰ_func(ctx->ਕਾਮਾ_pool, transfer_output_chunk_ਕਾਮਾ_fn, &state, actual_ਧਾਗੇ);
    }
}

ਬਣਤਰ activation_transfer_params {
    ਬਣਤਰ htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    ਪੂਰਨ                           n_ਪੰਕਤੀਆਂ;
    ਪੂਰਨ                           k_ਖੰਡ;
    ਪੂਰਨ                           k_stride;
    ਪੂਰਨ                           n_ਧਾਗੇ;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * act_ਧਾਗੇ_div;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * k_div;
    ਪੂਰਨ                           k_valid;
    ਦਸ਼ਮਲਵ *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_ਬਾਈਟਾਂ;
    uint32_t                      act_elem_ਮਾਪ;  // 4=F32 src1, 2=F16 src1
};

ਸਥਿਰ ਖਾਲੀ transfer_activation_chunk_threaded(ਅਚਲ ਬਣਤਰ activation_transfer_params * params) {
    ਬਣਤਰ htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ਅਚਲ dma_addr_t              act_dma_addr       = params->act_dma_addr;
    ਪੂਰਨ                           n_ਪੰਕਤੀਆਂ             = params->n_ਪੰਕਤੀਆਂ;
    ਪੂਰਨ                           k_ਖੰਡ            = params->k_ਖੰਡ;
    ਪੂਰਨ                           k_stride           = params->k_stride;
    ਪੂਰਨ                           n_ਧਾਗੇ          = params->n_ਧਾਗੇ;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * act_ਧਾਗੇ_div    = params->act_ਧਾਗੇ_div;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * k_div              = params->k_div;
    ਪੂਰਨ                           k_valid            = params->k_valid;
    ਦਸ਼ਮਲਵ *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_ਬਾਈਟਾਂ = params->vtcm_f32_act_ਬਾਈਟਾਂ;
    // element size of the activation rows (4 = F32, 2 = F16).
    ਅਚਲ uint32_t                act_elem_ਮਾਪ      = params->act_elem_ਮਾਪ ? params->act_elem_ਮਾਪ : (uint32_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਜੇ (n_ਪੰਕਤੀਆਂ <= 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ size_t n_tasks = (n_ਪੰਕਤੀਆਂ + 31) >> 5;
    ਜੇ (act_elem_ਮਾਪ == ਆਕਾਰ(ਦਸ਼ਮਲਵ) && n_ਧਾਗੇ > 1 && k_ਖੰਡ > 32 && n_tasks < (size_t) n_ਧਾਗੇ) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_ਕਦਮ_ਪੰਕਤੀਆਂ = 2;
        uint32_t dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ = 1;
        ਜੇ (vtcm_f32_act_ਬਾਈਟਾਂ > 0) {
            size_t thread_scratch_ਬਾਈਟਾਂ = hex_align_ਥੱਲੇ(fastdiv(vtcm_f32_act_ਬਾਈਟਾਂ, act_ਧਾਗੇ_div), 128);
            size_t thread_scratch_elements = thread_scratch_ਬਾਈਟਾਂ / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            size_t dma_ਕਦਮ_ਪੰਕਤੀਆਂ_max = fastdiv(thread_scratch_elements / 2, k_div);
            ਜੇ (dma_ਕਦਮ_ਪੰਕਤੀਆਂ_max >= 4) {
                dma_ਕਦਮ_ਪੰਕਤੀਆਂ = 4;
                dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ = 2;
            }
        }

        activation_transfer_ਖਾਨਾ_chunk_state_t ਖਾਨਾ_state;
        ਖਾਨਾ_state.dst = dst;
        ਖਾਨਾ_state.act_dma_addr = act_dma_addr;
        ਖਾਨਾ_state.n_ਪੰਕਤੀਆਂ = n_ਪੰਕਤੀਆਂ;
        ਖਾਨਾ_state.k_ਖੰਡ = k_ਖੰਡ;
        ਖਾਨਾ_state.k_stride = k_stride;
        ਖਾਨਾ_state.k_valid = k_valid;
        ਖਾਨਾ_state.n_ਖਾਨਾ_chunks = n_ਧਾਗੇ;
        ਖਾਨਾ_state.n_ਧਾਗੇ_div = *act_ਧਾਗੇ_div;
        ਖਾਨਾ_state.vtcm_f32_act = vtcm_f32_act;
        ਖਾਨਾ_state.vtcm_f32_act_ਬਾਈਟਾਂ = vtcm_f32_act_ਬਾਈਟਾਂ;
        ਖਾਨਾ_state.traces = ctx->trace;
        ਖਾਨਾ_state.ctx = ctx;
        ਖਾਨਾ_state.dma_ਕਦਮ_ਪੰਕਤੀਆਂ = dma_ਕਦਮ_ਪੰਕਤੀਆਂ;
        ਖਾਨਾ_state.dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ = dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ;

        ਕਾਮਾ_pool_ਲੜੀਵਾਰ_func(ctx->ਕਾਮਾ_pool, transfer_activation_chunk_ਖਾਨਾ_chunk_ਕਾਮਾ_fn, &ਖਾਨਾ_state, n_ਧਾਗੇ);
        ਵਾਪਸ;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(k_ਖੰਡ % HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ == 0 && k_stride % HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ == 0);

    size_t n_tot_chunks      = n_ਪੰਕਤੀਆਂ;
    size_t n_chunks_ਪ੍ਰਤੀ_task = (n_ਧਾਗੇ == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_ਧਾਗੇ == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
    state.n_tot_chunks       = n_tot_chunks;
    state.n_chunks_ਪ੍ਰਤੀ_task  = n_chunks_ਪ੍ਰਤੀ_task;
    state.dst                = dst;
    state.act_dma_addr       = act_dma_addr;
    state.k_ਖੰਡ            = k_ਖੰਡ;
    state.k_stride           = k_stride;
    state.k_valid            = k_valid;
    state.traces             = ctx->trace;
    state.ctx                = ctx;
    state.vtcm_f32_act       = vtcm_f32_act;
    state.act_elem_ਮਾਪ = act_elem_ਮਾਪ;

    state.vtcm_f32_act_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = hex_align_ਥੱਲੇ(fastdiv(vtcm_f32_act_ਬਾਈਟਾਂ, act_ਧਾਗੇ_div), 128);

    uint32_t dma_ਕਦਮ_ਪੰਕਤੀਆਂ = 2;
    uint32_t dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ = 1;
    ਜੇ (state.vtcm_f32_act_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        size_t dma_ਕਦਮ_ਪੰਕਤੀਆਂ_max = fastdiv(thread_scratch_elements / 2, k_div);
        ਜੇ (dma_ਕਦਮ_ਪੰਕਤੀਆਂ_max >= 4) {
            dma_ਕਦਮ_ਪੰਕਤੀਆਂ = 4;
            dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ = 2;
        }
    }
    state.dma_ਕਦਮ_ਪੰਕਤੀਆਂ      = dma_ਕਦਮ_ਪੰਕਤੀਆਂ;
    state.dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ = dma_ਕਦਮ_ਪੰਕਤੀਆਂ_ਖਿਸਕਾਓ;

    ਪੂਰਨ ਸਰਗਰਮ_ਧਾਗੇ = hex_smin(n_ਧਾਗੇ, (ਪੂਰਨ)state.n_tasks);
    ਜੇ (state.n_tasks == 1 || n_ਧਾਗੇ == 1) {
        transfer_activation_chunk_ਕਾਮਾ_fn(1, 0, &state);
    } ਨਹੀਂਤਾਂ {
        ਕਾਮਾ_pool_ਲੜੀਵਾਰ_func(ctx->ਕਾਮਾ_pool, transfer_activation_chunk_ਕਾਮਾ_fn, &state, ਸਰਗਰਮ_ਧਾਗੇ);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    __fp16 *       output;
    ਅਚਲ __fp16 * activation;
    ਅਚਲ __fp16 * ਭਾਰ;
    ਅਚਲ __fp16 * scales;
    uint32_t       n_ਪੰਕਤੀ_tiles;
    uint32_t       n_ਖਾਨਾ_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_ਕੰਮ_t;

ਸਥਿਰ ਖਾਲੀ hmx_matmul_ਕਾਮਾ_fn(ਖਾਲੀ * ਅੰਕੜੇ) {
    hmx_matmul_ਕੰਮ_t * ਕੰਮ = (hmx_matmul_ਕੰਮ_t *) ਅੰਕੜੇ;
    FARF(ਉੱਚਾ_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", ਕੰਮ->n_ਪੰਕਤੀ_tiles, ਕੰਮ->n_ਖਾਨਾ_tiles, ਕੰਮ->n_dot_tiles);
    core_dot_chunk_fp16(ਕੰਮ->output, ਕੰਮ->activation, ਕੰਮ->ਭਾਰ, ਕੰਮ->scales, ਕੰਮ->n_ਪੰਕਤੀ_tiles, ਕੰਮ->n_ਖਾਨਾ_tiles, ਕੰਮ->n_dot_tiles);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_matmul_ਕੰਮ_init(hmx_matmul_ਕੰਮ_t * ਕੰਮ,
                                       __fp16 *           output,
                                       ਅਚਲ __fp16 *     activation,
                                       ਅਚਲ __fp16 *     ਭਾਰ,
                                       ਅਚਲ __fp16 *     scales,
                                       uint32_t           n_ਪੰਕਤੀ_tiles,
                                       uint32_t           n_ਖਾਨਾ_tiles,
                                       uint32_t           n_dot_tiles) {
    ਕੰਮ->output      = output;
    ਕੰਮ->activation  = activation;
    ਕੰਮ->ਭਾਰ      = ਭਾਰ;
    ਕੰਮ->scales      = scales;
    ਕੰਮ->n_ਪੰਕਤੀ_tiles = n_ਪੰਕਤੀ_tiles;
    ਕੰਮ->n_ਖਾਨਾ_tiles = n_ਖਾਨਾ_tiles;
    ਕੰਮ->n_dot_tiles = n_dot_tiles;
}

ਸਥਿਰ ਪੂਰਨ hmx_mm_2d_f32(ਬਣਤਰ htp_context *ctx,
                                  dma_ਕਤਾਰ *ਭਾਰ_dma,
                                  ਦਸ਼ਮਲਵ *ਸੀਮਿਤ dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_ਬਾਈਟਾਂ,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t ਭਾਰ,
                                  ਪੂਰਨ m, ਪੂਰਨ k, ਪੂਰਨ n,
                                  ਪੂਰਨ act_stride,
                                  uint32_t act_elem_ਮਾਪ,  // 4=F32 src1, 2=F16 src1
                                  ਪੂਰਨ ਭਾਰ_stride,
                                  ਪੂਰਨ ਭਾਰ_type,
                                  ਪੂਰਨ k_valid,
                                  ਪੂਰਨ dst_stride,
                                  uint32_t src2_stride,
                                  ਪੂਰਨ dst_ਖਾਨੇ_ਗਿਣਤੀ,
                                  ਪੂਰਨ m_chunk,
                                  ਪੂਰਨ n_chunk,
                                  ਪੂਰਨ pipeline,
                                  ਪੂਰਨ n_ਧਾਗੇ,
                                  ਪੂਰਨ act_ਧਾਗੇ,
                                  ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * act_ਧਾਗੇ_div,
                                  ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * k_div,
                                  ਪੂਰਨ tile_ਮਾਪ,
                                  ਪੂਰਨ aligned_tile_ਮਾਪ,
                                  ਪੂਰਨ vtcm_ਮਾਪ) {
    ਬਣਤਰ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, 0);

    ਅਚਲ bool is_tiled = (tile_ਮਾਪ > 0);

    ਜੇ (k % 32 != 0) {
        ਵਾਪਸ -1;
    }
    ਜੇ (!hex_is_aligned(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ))) { ਵਾਪਸ -1; }

    size_t ਪੰਕਤੀ_stride = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(ਭਾਰ_type, k);
    ਜੇ (ਪੰਕਤੀ_stride == 0) {
        ਵਾਪਸ -1;
    }

    tiled_dequant_task_fn_t dequant_ਕਾਮਾ_fn = hmx_get_dequant_ਕਾਮਾ_fn(ਭਾਰ_type, is_tiled);

    ਅਚਲ ਪੂਰਨ n_k_tiles = k / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
    ਅਚਲ ਪੂਰਨ n_ਭਾਰ_tiles = k / htp_tensor_get_tile_k(ਭਾਰ_type);
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ n_k_tiles_div = init_fastdiv_ਮੁੱਲਾਂ(n_k_tiles);

    ਅਚਲ size_t vec_dot_ਮਾਪ = k * ਆਕਾਰ(__fp16);
    ਅਚਲ size_t vtcm_budget  = ctx->vtcm_ਮਾਪ;

    ਅਚਲ uint32_t dma_dst_stride  = is_tiled ? aligned_tile_ਮਾਪ : ਪੰਕਤੀ_stride;
    ਅਚਲ uint32_t dma_src_stride  = is_tiled ? tile_ਮਾਪ : ਭਾਰ_stride;
    ਅਚਲ uint32_t dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ = is_tiled ? tile_ਮਾਪ : ਪੰਕਤੀ_stride;

    size_t m_chunk_n_ਪੰਕਤੀਆਂ = m_chunk;
    size_t n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ = n_chunk;
    size_t vtcm_ਵਰਤਿਆ      = vtcm_ਮਾਪ;

    ਬਣਤਰ htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, ਭਾਰ_type, k, m_chunk_n_ਪੰਕਤੀਆਂ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, 1, pipeline, act_ਧਾਗੇ, aligned_tile_ਮਾਪ, src2_ਬਾਈਟਾਂ);

    vtcm_ਵਰਤਿਆ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;
    ਜੇ (vtcm_ਵਰਤਿਆ > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_ਵਰਤਿਆ, vtcm_budget, m, k, n, m_chunk_n_ਪੰਕਤੀਆਂ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
        ਵਾਪਸ -1;
    }

    uint8_t * ਅਚਲ ਅਧਾਰ = (uint8_t *) ctx->vtcm_ਅਧਾਰ;
    __fp16  *vtcm_ਭਾਰ_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_ਭਾਰ[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ਅਧਾਰ, L.off_ਭਾਰ[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_act);
    ਦਸ਼ਮਲਵ   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, ਅਧਾਰ, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_dst[0]);
    ਖਾਲੀ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(ਖਾਲੀ, ਅਧਾਰ, L.off_scratch[0]);
    ਖਾਲੀ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(ਖਾਲੀ, ਅਧਾਰ, L.off_scratch[1], pipeline);
    ਖਾਲੀ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(ਖਾਲੀ, ਅਧਾਰ, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_scales);

    hmx_init_ਕਾਲਮ_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ਅਚਲ bool has_bias      = (src2_ਬਾਈਟਾਂ > 0 && src2_addr != 0);
    ਦਸ਼ਮਲਵ   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(ਦਸ਼ਮਲਵ, ਅਧਾਰ, L.off_bias, has_bias);
    ਜੇ (has_bias) {
        dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_bias, src2_addr), hex_align_ਉੱਪਰ(src2_ਬਾਈਟਾਂ, 128), 0, src2_ਬਾਈਟਾਂ, 1);
        dma_ਕਤਾਰ_ਕੱਢੋ(ਭਾਰ_dma);
    }

    FARF(ਉੱਚਾ_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, ਭਾਰ_type, m_chunk_n_ਪੰਕਤੀਆਂ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, vtcm_ਵਰਤਿਆ, vtcm_budget);

    ਪੂਰਨ n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ਜੇ (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_ਕੰਮ_t ਕੰਮ_slots[2];  // persistent double-buffered job descriptors
        ਖਾਲੀ *vtcm_raw_bufs[4] = {
            vtcm_ਭਾਰ_raw[0],
            vtcm_ਭਾਰ_raw[1],
            dequant_ਕਾਮਾ_fn ? vtcm_ਭਾਰ_raw[0] : vtcm_scratch0,
            dequant_ਕਾਮਾ_fn ? vtcm_ਭਾਰ_raw[1] : vtcm_scratch1,
        };
        ਖਾਲੀ *vtcm_ਭਾਰ_bufs[4] = {
            dequant_ਕਾਮਾ_fn ? vtcm_scratch0 : vtcm_ਭਾਰ_raw[0],
            dequant_ਕਾਮਾ_fn ? vtcm_scratch1 : vtcm_ਭਾਰ_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        ਖਾਲੀ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        ਲਈ (size_t mr = 0; mr < m; mr += m_chunk_n_ਪੰਕਤੀਆਂ) {
            ਅਚਲ size_t n_ਪੰਕਤੀਆਂ = hex_smin(m - mr, m_chunk_n_ਪੰਕਤੀਆਂ);

            ਬਣਤਰ activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ਮਾਪ,  // byte offset (F16/F32)
                .n_ਪੰਕਤੀਆਂ = (ਪੂਰਨ) n_ਪੰਕਤੀਆਂ,
                .k_ਖੰਡ = k,
                .k_stride = act_stride,
                .n_ਧਾਗੇ = act_ਧਾਗੇ,
                .act_ਧਾਗੇ_div = act_ਧਾਗੇ_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_ਬਾਈਟਾਂ = L.act_f32_ਬਾਈਟਾਂ,
                .act_elem_ਮਾਪ = act_elem_ਮਾਪ,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            ਅਚਲ size_t   n_ਖਾਨੇ_ਗਿਣਤੀ_A0 = hex_smin(n - 0 * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
            ਅਚਲ uint32_t ਉਚਾਈ_A0 = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ_A0, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ_A0;
            dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[0], ਭਾਰ),
                           dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ_A0);

            ਜੇ (1 < n_chunk_cnt) {
                ਅਚਲ size_t   n_ਖਾਨੇ_ਗਿਣਤੀ_A1 = hex_smin(n - 1 * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                ਅਚਲ uint32_t ਉਚਾਈ_A1 = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ_A1, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ_A1;
                dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[1], ਭਾਰ + n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ * ਭਾਰ_stride),
                               dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            ਲਈ (ਪੂਰਨ i = 0; i < n_chunk_cnt; ++i) {
                ਅਚਲ size_t nc    = i * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
                ਅਚਲ size_t nc_p2 = nc + 2 * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;

                ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ    = hex_smin(n - nc, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_p2 = hex_smin(n - nc_p2, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);

                // 1. pop A_i
                ਖਾਲੀ * curr_raw = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(ਭਾਰ_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_ਭਾਰ_chunk_ਮੰਜ਼ਿਲ_fp16_tiles(
                    ctx, vtcm_ਭਾਰ_bufs[i % 4], curr_raw,
                    n_ਖਾਨੇ_ਗਿਣਤੀ, k, ਪੰਕਤੀ_stride, ਭਾਰ_type,
                    n_k_tiles, n_k_tiles_div, dequant_ਕਾਮਾ_fn, n_ਧਾਗੇ);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                ਜੇ (i + 2 < n_chunk_cnt) {
                    ਅਚਲ uint32_t ਉਚਾਈ_p2 = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ_p2, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ_p2;
                    dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[(i + 2) % 4], ਭਾਰ + nc_p2 * ਭਾਰ_stride),
                                   dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ_p2);
                }

                // 4. submit C_i
                hmx_matmul_ਕੰਮ_init(&ਕੰਮ_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_ਭਾਰ_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_ਪੰਕਤੀਆਂ, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ),
                                    hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ, HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ), k / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
                hmx_ਕਤਾਰ_ਧੱਕੋ(ctx->hmx_ਕਤਾਰ, hmx_ਕਤਾਰ_make_desc(hmx_matmul_ਕਾਮਾ_fn, &ਕੰਮ_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                ਜੇ (i > 0) {
                    hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);
                    ਅਚਲ size_t nc_prev = (i - 1) * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
                    ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_prev = hex_smin(n - nc_prev, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                    ਦਸ਼ਮਲਵ *output_chunk = dst + (mr * dst_stride + nc_prev);
                    ਅਚਲ ਦਸ਼ਮਲਵ *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : ਨਿਰਾ;
                    ਪੂਰਨ chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ = dst_ਖਾਨੇ_ਗਿਣਤੀ - (ਪੂਰਨ)nc_prev;
                    ਜੇ (chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_ਪੰਕਤੀਆਂ, n_ਖਾਨੇ_ਗਿਣਤੀ_prev, dst_stride, src2_stride, chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ, n_ਧਾਗੇ);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);
            ਅਚਲ size_t nc_ਆਖਰੀ = (n_chunk_cnt - 1) * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
            ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_ਆਖਰੀ = hex_smin(n - nc_ਆਖਰੀ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
            ਦਸ਼ਮਲਵ *output_chunk = dst + (mr * dst_stride + nc_ਆਖਰੀ);
            ਅਚਲ ਦਸ਼ਮਲਵ *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_ਆਖਰੀ) : ਨਿਰਾ;
            ਪੂਰਨ chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ = dst_ਖਾਨੇ_ਗਿਣਤੀ - (ਪੂਰਨ)nc_ਆਖਰੀ;
            ਜੇ (chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_ਪੰਕਤੀਆਂ, n_ਖਾਨੇ_ਗਿਣਤੀ_ਆਖਰੀ, dst_stride, src2_stride, chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ, n_ਧਾਗੇ);
            }
        }
    } ਨਹੀਂਤਾਂ {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_ਕੰਮ_t ਕੰਮ;
        ਖਾਲੀ *vtcm_raw_bufs[2] = { vtcm_ਭਾਰ_raw[0], dequant_ਕਾਮਾ_fn ? vtcm_ਭਾਰ_raw[0] : vtcm_scratch0 };
        ਖਾਲੀ *vtcm_ਭਾਰ_bufs[2] = { dequant_ਕਾਮਾ_fn ? vtcm_scratch0 : vtcm_ਭਾਰ_raw[0], vtcm_scratch0 };

        ਲਈ (size_t mr = 0; mr < m; mr += m_chunk_n_ਪੰਕਤੀਆਂ) {
            ਅਚਲ size_t n_ਪੰਕਤੀਆਂ = hex_smin(m - mr, m_chunk_n_ਪੰਕਤੀਆਂ);

            ਬਣਤਰ activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ਮਾਪ,  // byte offset (F16/F32)
                .n_ਪੰਕਤੀਆਂ = (ਪੂਰਨ) n_ਪੰਕਤੀਆਂ,
                .k_ਖੰਡ = k,
                .k_stride = act_stride,
                .n_ਧਾਗੇ = act_ਧਾਗੇ,
                .act_ਧਾਗੇ_div = act_ਧਾਗੇ_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_ਬਾਈਟਾਂ = L.act_f32_ਬਾਈਟਾਂ,
                .act_elem_ਮਾਪ = act_elem_ਮਾਪ,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            ਜੇ (n > 0) {
                ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ = hex_smin(n, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                ਅਚਲ uint32_t ਉਚਾਈ = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ;
                dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[0], ਭਾਰ), dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ);
            }

            ਲਈ (ਪੂਰਨ i = 0; i < n_chunk_cnt; ++i) {
                ਅਚਲ size_t nc = (size_t) i * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
                ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ = hex_smin(n - nc, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                ਅਚਲ size_t n_ਪੰਕਤੀ_tiles = hmx_ceil_div(n_ਪੰਕਤੀਆਂ, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
                ਅਚਲ size_t n_ਖਾਨਾ_tiles = hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ, HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);

                // A: Wait for weight DMA
                ਖਾਲੀ * curr_raw = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(ਭਾਰ_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_ਭਾਰ_chunk_ਮੰਜ਼ਿਲ_fp16_tiles(
                    ctx, vtcm_ਭਾਰ_bufs[i % 2], curr_raw,
                    n_ਖਾਨੇ_ਗਿਣਤੀ, k, ਪੰਕਤੀ_stride, ਭਾਰ_type,
                    n_k_tiles, n_k_tiles_div, dequant_ਕਾਮਾ_fn, n_ਧਾਗੇ);

                // Start weight DMA for the next chunk early
                ਅਚਲ size_t nc_ਅਗਲਾ = nc + n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
                ਜੇ (nc_ਅਗਲਾ < n) {
                    ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ = hex_smin(n - nc_ਅਗਲਾ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                    ਅਚਲ uint32_t ਉਚਾਈ_ਅਗਲਾ = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ;
                    dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[(i + 1) % 2], ਭਾਰ + nc_ਅਗਲਾ * ਭਾਰ_stride), dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ_ਅਗਲਾ);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_ਕੰਮ_init(&ਕੰਮ, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_ਭਾਰ_bufs[i % 2], vtcm_scales, n_ਪੰਕਤੀ_tiles, n_ਖਾਨਾ_tiles, k / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
                hmx_ਕਤਾਰ_ਧੱਕੋ(ctx->hmx_ਕਤਾਰ, hmx_ਕਤਾਰ_make_desc(hmx_matmul_ਕਾਮਾ_fn, &ਕੰਮ));
                hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);

                // D: Output Store
                ਦਸ਼ਮਲਵ *output_chunk = dst + (mr * dst_stride + nc);
                ਅਚਲ ਦਸ਼ਮਲਵ *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : ਨਿਰਾ;
                ਪੂਰਨ chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ = dst_ਖਾਨੇ_ਗਿਣਤੀ - (ਪੂਰਨ)nc;
                ਜੇ (chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_ਪੰਕਤੀਆਂ, n_ਖਾਨੇ_ਗਿਣਤੀ, dst_stride, src2_stride, chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ, n_ਧਾਗੇ);
                }
            }
        }
    }

    ਵਾਪਸ 0;
}

ਸਥਿਰ ਪੂਰਨ hmx_mm_nx_2d_f32(ਬਣਤਰ htp_ops_context * octx, ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams) {
    ਬਣਤਰ htp_context * ctx = octx->ctx;
    ਬਣਤਰ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, 0);

    ਅਚਲ uint32_t n_weights = kparams->n_weights;
    ਜੇ (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ act  = octx->src[n_weights];

    ਅਚਲ ਪੂਰਨ ਭਾਰ_type = (ਪੂਰਨ) src0->type;
    ਅਚਲ ਪੂਰਨ k           = (ਪੂਰਨ) act->ne[0];
    ਅਚਲ ਪੂਰਨ k_valid     = (ਪੂਰਨ) act->ne[0];
    ਅਚਲ ਪੂਰਨ m           = (ਪੂਰਨ) (act->ne[1] * act->ne[2] * act->ne[3]);
    ਅਚਲ uint32_t act_elem_ਮਾਪ = (act->type == HTP_TYPE_F16) ? ਆਕਾਰ(__fp16) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ ਪੂਰਨ act_stride  = (ਪੂਰਨ) (act->nb[1] / act_elem_ਮਾਪ);
    ਅਚਲ dma_addr_t act_dma_addr = act->ਅੰਕੜੇ;

    ਜੇ (k % 32 != 0) { ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT; }

    size_t ਪੰਕਤੀ_stride = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(ਭਾਰ_type, k);
    ਜੇ (ਪੰਕਤੀ_stride == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ ਪੂਰਨ n_k_tiles = k / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
    ਅਚਲ ਪੂਰਨ n_ਭਾਰ_tiles = k / htp_tensor_get_tile_k(ਭਾਰ_type);
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ n_k_tiles_div = init_fastdiv_ਮੁੱਲਾਂ(n_k_tiles);

    ਅਚਲ ਪੂਰਨ tile_ਮਾਪ       = kparams->tile_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ aligned_tile_ਮਾਪ = kparams->aligned_tile_ਮਾਪ;

    ਅਚਲ bool is_tiled       = (tile_ਮਾਪ > 0);
    tiled_dequant_task_fn_t dequant_ਕਾਮਾ_fn = hmx_get_dequant_ਕਾਮਾ_fn(ਭਾਰ_type, is_tiled);
    ਅਚਲ size_t vtcm_budget  = ctx->vtcm_ਮਾਪ;

    ਅਚਲ ਪੂਰਨ m_chunk_n_ਪੰਕਤੀਆਂ  = kparams->m_chunk;
    ਅਚਲ ਪੂਰਨ n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ  = kparams->n_chunk;
    ਅਚਲ ਪੂਰਨ pipeline        = kparams->pipeline;
    ਅਚਲ ਪੂਰਨ n_ਧਾਗੇ       = octx->n_ਧਾਗੇ;
    ਅਚਲ ਪੂਰਨ act_ਧਾਗੇ     = kparams->n_act_ਧਾਗੇ;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * act_ਧਾਗੇ_div = &kparams->div_n_act_ਧਾਗੇ;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * k_div           = &kparams->div_ne00_padded;

    ਅਚਲ uint32_t dma_dst_stride  = is_tiled ? aligned_tile_ਮਾਪ : ਪੰਕਤੀ_stride;
    ਅਚਲ uint32_t dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ = is_tiled ? tile_ਮਾਪ : ਪੰਕਤੀ_stride;

    ਬਣਤਰ htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, ਭਾਰ_type, k, m_chunk_n_ਪੰਕਤੀਆਂ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, 1, pipeline, act_ਧਾਗੇ, aligned_tile_ਮਾਪ, 0);

    ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.ਕੁੱਲ_ਬਾਈਟਾਂ, vtcm_budget, m, k, m_chunk_n_ਪੰਕਤੀਆਂ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    uint8_t * ਅਚਲ ਅਧਾਰ = (uint8_t *) ctx->vtcm_ਅਧਾਰ;
    __fp16  *vtcm_ਭਾਰ_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_ਭਾਰ[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ਅਧਾਰ, L.off_ਭਾਰ[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_act);
    ਦਸ਼ਮਲਵ   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, ਅਧਾਰ, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_dst[0]);
    ਖਾਲੀ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(ਖਾਲੀ, ਅਧਾਰ, L.off_scratch[0]);
    ਖਾਲੀ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(ਖਾਲੀ, ਅਧਾਰ, L.off_scratch[1], pipeline);
    ਖਾਲੀ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(ਖਾਲੀ, ਅਧਾਰ, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_scales);

    hmx_init_ਕਾਲਮ_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ਪੂਰਨ m_ਸ਼ੁਰੂ = 0;
    ਪੂਰਨ m_ਪੰਕਤੀਆਂ  = m;
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        bool can_ਵੰਡੋ = htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(act, act_elem_ਮਾਪ);
        ਜੇ (can_ਵੰਡੋ) {
            ਲਈ (uint32_t p = 0; p < n_weights; ++p) {
                ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst = octx->dsts[p];
                ਜੇ (!htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ))) {
                    can_ਵੰਡੋ = false;
                    ਰੋਕੋ;
                }
            }
        }
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ((uint32_t) m, can_ਵੰਡੋ ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        m_ਸ਼ੁਰੂ = (ਪੂਰਨ) range.ਸ਼ੁਰੂ;
        m_ਪੰਕਤੀਆਂ  = (ਪੂਰਨ) range.ਗਿਣਤੀ_2;
    }

    ਜੇ (m_ਪੰਕਤੀਆਂ == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    FARF(ਉੱਚਾ_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_ਸ਼ੁਰੂ, m_ਸ਼ੁਰੂ + m_ਪੰਕਤੀਆਂ, k, ਭਾਰ_type, m_chunk_n_ਪੰਕਤੀਆਂ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, L.ਕੁੱਲ_ਬਾਈਟਾਂ, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ਅਚਲ size_t mr_ਸਿਰਾ = (size_t)(m_ਸ਼ੁਰੂ + m_ਪੰਕਤੀਆਂ);

    ਜੇ (pipeline) {
        hmx_matmul_ਕੰਮ_t ਕੰਮ_slots[2];
        ਖਾਲੀ *vtcm_raw_bufs[4] = {
            vtcm_ਭਾਰ_raw[0],
            vtcm_ਭਾਰ_raw[1],
            dequant_ਕਾਮਾ_fn ? vtcm_ਭਾਰ_raw[0] : vtcm_scratch0,
            dequant_ਕਾਮਾ_fn ? vtcm_ਭਾਰ_raw[1] : vtcm_scratch1,
        };
        ਖਾਲੀ *vtcm_ਭਾਰ_bufs[4] = {
            dequant_ਕਾਮਾ_fn ? vtcm_scratch0 : vtcm_ਭਾਰ_raw[0],
            dequant_ਕਾਮਾ_fn ? vtcm_scratch1 : vtcm_ਭਾਰ_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        ਖਾਲੀ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        ਲਈ (size_t mr = (size_t) m_ਸ਼ੁਰੂ; mr < mr_ਸਿਰਾ; mr += m_chunk_n_ਪੰਕਤੀਆਂ) {
            ਅਚਲ size_t n_ਪੰਕਤੀਆਂ = hex_smin(mr_ਸਿਰਾ - mr, m_chunk_n_ਪੰਕਤੀਆਂ);

            ਬਣਤਰ activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ਮਾਪ,
                .n_ਪੰਕਤੀਆਂ = (ਪੂਰਨ) n_ਪੰਕਤੀਆਂ,
                .k_ਖੰਡ = k,
                .k_stride = act_stride,
                .n_ਧਾਗੇ = act_ਧਾਗੇ,
                .act_ਧਾਗੇ_div = act_ਧਾਗੇ_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_ਬਾਈਟਾਂ = L.act_f32_ਬਾਈਟਾਂ,
                .act_elem_ਮਾਪ = act_elem_ਮਾਪ,
            };
            transfer_activation_chunk_threaded(&act_params);

            ਲਈ (uint32_t p = 0; p < n_weights; p++) {
                ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src_w = octx->src[p];
                ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst   = octx->dsts[p];
                ਜੇ (!src_w || !dst) ਜਾਰੀ;

                ਅਚਲ dma_addr_t ਭਾਰ       = src_w->ਅੰਕੜੇ;
                dma_ਕਤਾਰ * ਭਾਰ_dma      = octx->ctx->dma[0];
                ਦਸ਼ਮਲਵ * dst_ptr            = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
                ਅਚਲ size_t n             = src_w->ne[1];
                ਜੇ (n == 0) ਜਾਰੀ;
                ਅਚਲ size_t ਭਾਰ_stride = src_w->nb[1];
                ਅਚਲ size_t dst_stride    = dst->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
                ਅਚਲ ਪੂਰਨ dst_ਖਾਨੇ_ਗਿਣਤੀ         = (ਪੂਰਨ) dst->ne[0];
                ਅਚਲ ਪੂਰਨ n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);

                ਅਚਲ uint32_t dma_src_stride = is_tiled ? tile_ਮਾਪ : ਭਾਰ_stride;

                ਅਚਲ size_t   n_ਖਾਨੇ_ਗਿਣਤੀ_A0 = hex_smin(n - 0 * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                ਅਚਲ uint32_t ਉਚਾਈ_A0 = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ_A0, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ_A0;
                dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[0], ਭਾਰ),
                               dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ_A0);

                ਜੇ (1 < n_chunk_cnt) {
                    ਅਚਲ size_t   n_ਖਾਨੇ_ਗਿਣਤੀ_A1 = hex_smin(n - 1 * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                    ਅਚਲ uint32_t ਉਚਾਈ_A1 = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ_A1, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ_A1;
                    dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[1], ਭਾਰ + n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ * ਭਾਰ_stride),
                                   dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ_A1);
                }

                ਲਈ (ਪੂਰਨ i = 0; i < n_chunk_cnt; ++i) {
                    ਅਚਲ size_t nc    = i * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
                    ਅਚਲ size_t nc_p2 = nc + 2 * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;

                    ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ    = hex_smin(n - nc, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                    ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_p2 = hex_smin(n - nc_p2, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);

                    ਖਾਲੀ * curr_raw = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(ਭਾਰ_dma).dst;

                    dequantize_tiled_ਭਾਰ_chunk_ਮੰਜ਼ਿਲ_fp16_tiles(
                        ctx, vtcm_ਭਾਰ_bufs[i % 4], curr_raw,
                        n_ਖਾਨੇ_ਗਿਣਤੀ, k, ਪੰਕਤੀ_stride, ਭਾਰ_type,
                        n_k_tiles, n_k_tiles_div, dequant_ਕਾਮਾ_fn, n_ਧਾਗੇ);

                    ਜੇ (i + 2 < n_chunk_cnt) {
                        ਅਚਲ uint32_t ਉਚਾਈ_p2 = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ_p2, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ_p2;
                        dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[(i + 2) % 4], ਭਾਰ + nc_p2 * ਭਾਰ_stride),
                                       dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ_p2);
                    }

                    hmx_matmul_ਕੰਮ_init(&ਕੰਮ_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_ਭਾਰ_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_ਪੰਕਤੀਆਂ, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ),
                                        hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ, HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ), k / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
                    hmx_ਕਤਾਰ_ਧੱਕੋ(ctx->hmx_ਕਤਾਰ, hmx_ਕਤਾਰ_make_desc(hmx_matmul_ਕਾਮਾ_fn, &ਕੰਮ_slots[i % 2]));

                    ਜੇ (i > 0) {
                        hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);
                        ਅਚਲ size_t nc_prev = (i - 1) * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
                        ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_prev = hex_smin(n - nc_prev, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                        ਦਸ਼ਮਲਵ *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        ਪੂਰਨ chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ = dst_ਖਾਨੇ_ਗਿਣਤੀ - (ਪੂਰਨ)nc_prev;
                        ਜੇ (chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, ਨਿਰਾ, vtcm_output_bufs[(i - 1) % 2], n_ਪੰਕਤੀਆਂ, n_ਖਾਨੇ_ਗਿਣਤੀ_prev, dst_stride, 0, chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ, n_ਧਾਗੇ);
                        }
                    }
                }

                hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);
                ਅਚਲ size_t nc_ਆਖਰੀ = (n_chunk_cnt - 1) * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
                ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_ਆਖਰੀ = hex_smin(n - nc_ਆਖਰੀ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                ਦਸ਼ਮਲਵ *output_chunk = dst_ptr + (mr * dst_stride + nc_ਆਖਰੀ);
                ਪੂਰਨ chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ = dst_ਖਾਨੇ_ਗਿਣਤੀ - (ਪੂਰਨ)nc_ਆਖਰੀ;
                ਜੇ (chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, ਨਿਰਾ, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_ਪੰਕਤੀਆਂ, n_ਖਾਨੇ_ਗਿਣਤੀ_ਆਖਰੀ, dst_stride, 0, chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ, n_ਧਾਗੇ);
                }
            }
        }
    } ਨਹੀਂਤਾਂ {
        hmx_matmul_ਕੰਮ_t ਕੰਮ;
        ਖਾਲੀ *vtcm_raw_bufs[2] = { vtcm_ਭਾਰ_raw[0], dequant_ਕਾਮਾ_fn ? vtcm_ਭਾਰ_raw[0] : vtcm_scratch0 };
        ਖਾਲੀ *vtcm_ਭਾਰ_bufs[2] = { dequant_ਕਾਮਾ_fn ? vtcm_scratch0 : vtcm_ਭਾਰ_raw[0], vtcm_scratch0 };

        ਲਈ (size_t mr = (size_t) m_ਸ਼ੁਰੂ; mr < mr_ਸਿਰਾ; mr += m_chunk_n_ਪੰਕਤੀਆਂ) {
            ਅਚਲ size_t n_ਪੰਕਤੀਆਂ = hex_smin(mr_ਸਿਰਾ - mr, m_chunk_n_ਪੰਕਤੀਆਂ);

            ਬਣਤਰ activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ਮਾਪ,
                .n_ਪੰਕਤੀਆਂ = (ਪੂਰਨ) n_ਪੰਕਤੀਆਂ,
                .k_ਖੰਡ = k,
                .k_stride = act_stride,
                .n_ਧਾਗੇ = act_ਧਾਗੇ,
                .act_ਧਾਗੇ_div = act_ਧਾਗੇ_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_ਬਾਈਟਾਂ = L.act_f32_ਬਾਈਟਾਂ,
                .act_elem_ਮਾਪ = act_elem_ਮਾਪ,
            };
            transfer_activation_chunk_threaded(&act_params);

            ਲਈ (uint32_t p = 0; p < n_weights; p++) {
                ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src_w = octx->src[p];
                ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst   = octx->dsts[p];
                ਜੇ (!src_w || !dst) ਜਾਰੀ;

                ਅਚਲ dma_addr_t ਭਾਰ       = src_w->ਅੰਕੜੇ;
                dma_ਕਤਾਰ * ਭਾਰ_dma      = octx->ctx->dma[0];
                ਦਸ਼ਮਲਵ * dst_ptr            = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
                ਅਚਲ size_t n             = src_w->ne[1];
                ਜੇ (n == 0) ਜਾਰੀ;
                ਅਚਲ size_t ਭਾਰ_stride = src_w->nb[1];
                ਅਚਲ size_t dst_stride    = dst->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
                ਅਚਲ ਪੂਰਨ dst_ਖਾਨੇ_ਗਿਣਤੀ         = (ਪੂਰਨ) dst->ne[0];
                ਅਚਲ ਪੂਰਨ n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);

                ਅਚਲ uint32_t dma_src_stride = is_tiled ? tile_ਮਾਪ : ਭਾਰ_stride;

                ਜੇ (n > 0) {
                    ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ = hex_smin(n, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                    ਅਚਲ uint32_t ਉਚਾਈ = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ;
                    dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[0], ਭਾਰ), dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ);
                }

                ਲਈ (ਪੂਰਨ i = 0; i < n_chunk_cnt; ++i) {
                    ਅਚਲ size_t nc = (size_t) i * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
                    ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ = hex_smin(n - nc, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                    ਅਚਲ size_t n_ਪੰਕਤੀ_tiles = hmx_ceil_div(n_ਪੰਕਤੀਆਂ, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
                    ਅਚਲ size_t n_ਖਾਨਾ_tiles = hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ, HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);

                    ਖਾਲੀ * curr_raw = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(ਭਾਰ_dma).dst;

                    dequantize_tiled_ਭਾਰ_chunk_ਮੰਜ਼ਿਲ_fp16_tiles(
                        ctx, vtcm_ਭਾਰ_bufs[i % 2], curr_raw,
                        n_ਖਾਨੇ_ਗਿਣਤੀ, k, ਪੰਕਤੀ_stride, ਭਾਰ_type,
                        n_k_tiles, n_k_tiles_div, dequant_ਕਾਮਾ_fn, n_ਧਾਗੇ);

                    ਅਚਲ size_t nc_ਅਗਲਾ = nc + n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
                    ਜੇ (nc_ਅਗਲਾ < n) {
                        ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ = hex_smin(n - nc_ਅਗਲਾ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                        ਅਚਲ uint32_t ਉਚਾਈ_ਅਗਲਾ = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ;
                        dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[(i + 1) % 2], ਭਾਰ + nc_ਅਗਲਾ * ਭਾਰ_stride), dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ_ਅਗਲਾ);
                    }

                    hmx_matmul_ਕੰਮ_init(&ਕੰਮ, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_ਭਾਰ_bufs[i % 2], vtcm_scales, n_ਪੰਕਤੀ_tiles, n_ਖਾਨਾ_tiles, k / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
                    hmx_ਕਤਾਰ_ਧੱਕੋ(ctx->hmx_ਕਤਾਰ, hmx_ਕਤਾਰ_make_desc(hmx_matmul_ਕਾਮਾ_fn, &ਕੰਮ));
                    hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);

                    ਦਸ਼ਮਲਵ *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    ਪੂਰਨ chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ = dst_ਖਾਨੇ_ਗਿਣਤੀ - (ਪੂਰਨ)nc;
                    ਜੇ (chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, ਨਿਰਾ, vtcm_output, n_ਪੰਕਤੀਆਂ, n_ਖਾਨੇ_ਗਿਣਤੀ, dst_stride, 0, chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ, n_ਧਾਗੇ);
                    }
                }
            }
        }
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_addr_t hmx_mm_ਭਾਰ_batch_ਅੰਕੜੇ(ਅਚਲ hmx_mm_batched_params_t *params,
                                                         ਪੂਰਨ dst_b2, ਪੂਰਨ dst_b3) {
    ਅਚਲ size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    ਅਚਲ size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    ਵਾਪਸ params->ਭਾਰ + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_addr_t hmx_mm_act_batch_addr(ਅਚਲ hmx_mm_batched_params_t *params,
                                                ਪੂਰਨ dst_b2, ਪੂਰਨ dst_b3) {
    ਵਾਪਸ params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ *hmx_mm_dst_batch_ptr(ਅਚਲ hmx_mm_batched_params_t *params,
                                              ਪੂਰਨ dst_b2, ਪੂਰਨ dst_b3) {
    ਵਾਪਸ (ਦਸ਼ਮਲਵ *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

ਸਥਿਰ ਪੂਰਨ hmx_mm_batched_simple(ਬਣਤਰ htp_context *ctx,
                                 ਅਚਲ hmx_mm_batched_params_t *params,
                                 ਪੂਰਨ m_chunk, ਪੂਰਨ n_chunk, ਪੂਰਨ pipeline, ਪੂਰਨ n_ਧਾਗੇ, ਪੂਰਨ act_ਧਾਗੇ, ਪੂਰਨ vtcm_ਮਾਪ,
                                 ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * act_ਧਾਗੇ_div, ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * k_div) {
    ਪੂਰਨ ret = 0;
    ਲਈ (ਪੂਰਨ b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        ਲਈ (ਪੂਰਨ b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->ਭਾਰ_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_ਬਾਈਟਾਂ,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_ਭਾਰ_batch_ਅੰਕੜੇ(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_ਮਾਪ, params->ਭਾਰ_stride,
                                params->ਭਾਰ_type, params->k, params->dst_stride, params->bias_stride, params->dst_ਖਾਨੇ_ਗਿਣਤੀ,
                                m_chunk, n_chunk, pipeline, n_ਧਾਗੇ, act_ਧਾਗੇ,
                                act_ਧਾਗੇ_div, k_div, params->tile_ਮਾਪ, params->aligned_tile_ਮਾਪ, vtcm_ਮਾਪ);
        }
    }
    ਵਾਪਸ ret;
}

ਸਥਿਰ ਪੂਰਨ hmx_mm_batched(ਬਣਤਰ htp_context *ctx, ਅਚਲ hmx_mm_batched_params_t *params,
                          ਪੂਰਨ m_chunk, ਪੂਰਨ n_chunk, ਪੂਰਨ pipeline, ਪੂਰਨ n_ਧਾਗੇ, ਪੂਰਨ act_ਧਾਗੇ,
                          ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * act_ਧਾਗੇ_div,
                          ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * k_div,
                          ਪੂਰਨ vtcm_ਮਾਪ) {
    ਅਚਲ size_t ਪੰਕਤੀ_stride = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(params->ਭਾਰ_type, params->k);
    ਜੇ (ਪੰਕਤੀ_stride == 0) { ਵਾਪਸ -1; }
    ਜੇ (params->act_stride < params->k || (params->tile_ਮਾਪ == 0 && (size_t) params->ਭਾਰ_stride < ਪੰਕਤੀ_stride) || params->dst_stride < params->dst_ਖਾਨੇ_ਗਿਣਤੀ) { ਵਾਪਸ -1; }
    ਜੇ (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { ਵਾਪਸ -1; }
    ਜੇ (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { ਵਾਪਸ -1; }
    // N (the weight row count) does not have to be 32-aligned:
    ਜੇ (params->k % 32 != 0) { ਵਾਪਸ -1; }
    ਜੇ (!hex_is_aligned(params->dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ))) { ਵਾਪਸ -1; }

    ਅਚਲ ਪੂਰਨ ਸਮੂਹ_ਮਾਪ = params->r2;
    ਅਚਲ size_t vtcm_budget  = ctx->vtcm_ਮਾਪ;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool ਲੜੀਵਾਰ_grouped = (params->ਭਾਰ_type == HTP_TYPE_F16 && params->tile_ਮਾਪ == 0 && ਸਮੂਹ_ਮਾਪ > 1 && (size_t)vtcm_ਮਾਪ <= vtcm_budget);
    ਜੇ (!ਲੜੀਵਾਰ_grouped) {
        ਵਾਪਸ hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_ਧਾਗੇ, act_ਧਾਗੇ, vtcm_ਮਾਪ, act_ਧਾਗੇ_div, k_div);
    }

    ਬਣਤਰ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_ਪੰਕਤੀਆਂ = m_chunk;
    size_t n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ = n_chunk;

    ਬਣਤਰ htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_ਪੰਕਤੀਆਂ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, ਸਮੂਹ_ਮਾਪ, false, act_ਧਾਗੇ, 0, params->bias_ਬਾਈਟਾਂ);

    ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ > vtcm_budget) {
        FARF(ਉੱਚਾ_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        ਵਾਪਸ hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_ਧਾਗੇ, act_ਧਾਗੇ, vtcm_ਮਾਪ, act_ਧਾਗੇ_div, k_div);
    }

    uint8_t * ਅਚਲ ਅਧਾਰ = (uint8_t *) ctx->vtcm_ਅਧਾਰ;
    __fp16  *vtcm_ਭਾਰ     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_ਭਾਰ[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_dst[0]);
    ਖਾਲੀ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(ਖਾਲੀ, ਅਧਾਰ, L.off_scratch[0]);
    ਖਾਲੀ    *vtcm_scratch1   = VTCM_LAYOUT_PTR(ਖਾਲੀ, ਅਧਾਰ, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_scales);
    ਦਸ਼ਮਲਵ   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ਦਸ਼ਮਲਵ, ਅਧਾਰ, L.off_act_f32);

    ਅਚਲ bool has_bias      = (params->bias_ਬਾਈਟਾਂ > 0 && params->bias_addr != 0);
    ਦਸ਼ਮਲਵ   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(ਦਸ਼ਮਲਵ, ਅਧਾਰ, L.off_bias, has_bias);
    ਜੇ (has_bias) {
        dma_ਕਤਾਰ_ਧੱਕੋ(params->ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_bias, params->bias_addr), hex_align_ਉੱਪਰ(params->bias_ਬਾਈਟਾਂ, 128), 0, params->bias_ਬਾਈਟਾਂ, 1);
        dma_ਕਤਾਰ_ਕੱਢੋ(params->ਭਾਰ_dma);
    }

    hmx_init_ਕਾਲਮ_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(ਉੱਚਾ_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, ਸਮੂਹ_ਮਾਪ, params->ne13,
            m_chunk_n_ਪੰਕਤੀਆਂ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ,
            L.ਕੁੱਲ_ਬਾਈਟਾਂ, vtcm_budget);

    ਅਚਲ size_t fp16_ਪੰਕਤੀ_ਬਾਈਟਾਂ   = (size_t) params->k * ਆਕਾਰ(__fp16);
    ਅਚਲ size_t ਭਾਰ_ਪੰਕਤੀ_ਬਾਈਟਾਂ = (size_t) params->ਭਾਰ_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_ਕੰਮ_t ਕੰਮ;

    ਲਈ (ਪੂਰਨ b3 = 0; b3 < params->ne13; ++b3) {
        ਲਈ (ਪੂਰਨ b2_ਅਧਾਰ = 0; b2_ਅਧਾਰ < params->ne12; b2_ਅਧਾਰ += ਸਮੂਹ_ਮਾਪ) {
            ਅਚਲ dma_addr_t ਭਾਰ_ਸਮੂਹ = hmx_mm_ਭਾਰ_batch_ਅੰਕੜੇ(params, b2_ਅਧਾਰ, b3);
            dma_ਕਤਾਰ * ਭਾਰ_dma = params->ਭਾਰ_dma;

            ਲਈ (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_ਪੰਕਤੀਆਂ) {
                ਅਚਲ size_t n_ਪੰਕਤੀਆਂ = hex_smin((size_t) params->m - mr, m_chunk_n_ਪੰਕਤੀਆਂ);
                ਅਚਲ size_t n_ਪੰਕਤੀ_tiles = hmx_ceil_div((ਪੂਰਨ) n_ਪੰਕਤੀਆਂ, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                ਲਈ (ਪੂਰਨ g = 0; g < ਸਮੂਹ_ਮਾਪ; ++g) {
                    ਅਚਲ dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_ਅਧਾਰ + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_ਮਾਪ;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_ਸਿਰ_stride;
                    ਬਣਤਰ activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_ਪੰਕਤੀਆਂ = (ਪੂਰਨ) n_ਪੰਕਤੀਆਂ,
                        .k_ਖੰਡ = params->k,
                        .k_stride = params->act_stride,
                        .n_ਧਾਗੇ = act_ਧਾਗੇ,
                        .act_ਧਾਗੇ_div = act_ਧਾਗੇ_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_ਬਾਈਟਾਂ = L.act_f32_ਬਾਈਟਾਂ,
                        .act_elem_ਮਾਪ = params->act_elem_ਮਾਪ,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_ਪਹਿਲਾ = hex_smin((size_t) params->n, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                    dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_scratch0, ਭਾਰ_ਸਮੂਹ),
                                      fp16_ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਭਾਰ_ਪੰਕਤੀ_ਬਾਈਟਾਂ, fp16_ਪੰਕਤੀ_ਬਾਈਟਾਂ, n_ਖਾਨੇ_ਗਿਣਤੀ_ਪਹਿਲਾ);
                }
                ਜੇ (n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ < (size_t) params->n) {
                    ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_ਦੂਜਾ = hex_smin((size_t) params->n - n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                    ਅਚਲ dma_addr_t ਦੂਜਾ_ਭਾਰ_chunk = ਭਾਰ_ਸਮੂਹ + n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ * ਭਾਰ_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
                    dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_scratch1, ਦੂਜਾ_ਭਾਰ_chunk),
                                      fp16_ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਭਾਰ_ਪੰਕਤੀ_ਬਾਈਟਾਂ, fp16_ਪੰਕਤੀ_ਬਾਈਟਾਂ, n_ਖਾਨੇ_ਗਿਣਤੀ_ਦੂਜਾ);
                }

                ਲਈ (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ) {
                    ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ      = hex_smin((size_t) params->n - nc, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                    ਅਚਲ size_t n_ਖਾਨਾ_tiles = hmx_ceil_div((ਪੂਰਨ) n_ਖਾਨੇ_ਗਿਣਤੀ, HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);

                    {
                        ਖਾਲੀ * curr_raw = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(ਭਾਰ_dma).dst;

                        ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_tiled = hex_align_ਉੱਪਰ(n_ਖਾਨੇ_ਗਿਣਤੀ, HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);

                        hmx_interleave_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_tiles(vtcm_ਭਾਰ, (ਅਚਲ __fp16 *) curr_raw, (uint32_t) n_ਖਾਨੇ_ਗਿਣਤੀ, params->k, params->k, 0, (uint32_t) n_ਖਾਨੇ_ਗਿਣਤੀ_tiled);

                        ਅਚਲ size_t nc_ਅਗਲਾ = nc + n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ * 2;
                        ਜੇ (nc_ਅਗਲਾ < (size_t) params->n) {
                            ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ = hex_smin((size_t) params->n - nc_ਅਗਲਾ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                            ਅਚਲ dma_addr_t ਅਗਲਾ_ਭਾਰ_chunk = ਭਾਰ_ਸਮੂਹ + nc_ਅਗਲਾ * ਭਾਰ_ਪੰਕਤੀ_ਬਾਈਟਾਂ;

                            dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(curr_raw, ਅਗਲਾ_ਭਾਰ_chunk),
                                              fp16_ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਭਾਰ_ਪੰਕਤੀ_ਬਾਈਟਾਂ, fp16_ਪੰਕਤੀ_ਬਾਈਟਾਂ, n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    ਲਈ (ਪੂਰਨ g = 0; g < ਸਮੂਹ_ਮਾਪ; ++g) {
                        {
                            ਅਚਲ __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_ਸਿਰ_stride;
                            hmx_matmul_ਕੰਮ_init(&ਕੰਮ, vtcm_output, vtcm_act_g, vtcm_ਭਾਰ, vtcm_scales, n_ਪੰਕਤੀ_tiles, n_ਖਾਨਾ_tiles, params->k / 32);
                            hmx_ਕਤਾਰ_ਧੱਕੋ(ctx->hmx_ਕਤਾਰ, hmx_ਕਤਾਰ_make_desc(hmx_matmul_ਕਾਮਾ_fn, &ਕੰਮ));
                            hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);
                        }

                        {
                            ਦਸ਼ਮਲਵ *output = hmx_mm_dst_batch_ptr(params, b2_ਅਧਾਰ + g, b3) + mr * params->dst_stride + nc;
                            ਅਚਲ ਦਸ਼ਮਲਵ *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : ਨਿਰਾ;
                            ਪੂਰਨ chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ = params->dst_ਖਾਨੇ_ਗਿਣਤੀ - (ਪੂਰਨ)nc;
                            ਜੇ (chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (ਪੂਰਨ) n_ਪੰਕਤੀਆਂ, (ਪੂਰਨ) n_ਖਾਨੇ_ਗਿਣਤੀ,
                                                               params->dst_stride, params->bias_stride, chunk_dst_ਖਾਨੇ_ਗਿਣਤੀ, n_ਧਾਗੇ);
                            }
                        }
                    }
                }
            }
        }
    }

    ਵਾਪਸ 0;
}

ਸਥਿਰ ਖਾਲੀ transfer_activation_chunk_gathered_threaded(
            ਬਣਤਰ htp_context *ctx,
            __fp16 *dst,
            ਅਚਲ ਦਸ਼ਮਲਵ *src,
            ਪੂਰਨ ਸ਼ੁਰੂ_ਪੰਕਤੀ,
            ਪੂਰਨ n_ਪੰਕਤੀਆਂ,
            ਪੂਰਨ k_ਖੰਡ,
            ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping *ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
            ਪੂਰਨ cur_a,
            ਪੂਰਨ mapping_stride,
            ਪੂਰਨ ne11,
            size_t nb11,
            size_t nb12,
            ਪੂਰਨ cne1,
            ਪੂਰਨ n_ਧਾਗੇ,
            ਪੂਰਨ k_valid) {
    ਜੇ (n_ਪੰਕਤੀਆਂ <= 0) ਵਾਪਸ;
    ਪੂਰਨ chunks_ਪ੍ਰਤੀ_thread = hmx_ceil_div(n_ਪੰਕਤੀਆਂ, n_ਧਾਗੇ);
    chunks_ਪ੍ਰਤੀ_thread = hex_align_ਉੱਪਰ(chunks_ਪ੍ਰਤੀ_thread, 2);

    ਪੂਰਨ actual_ਧਾਗੇ = hmx_ceil_div(n_ਪੰਕਤੀਆਂ, chunks_ਪ੍ਰਤੀ_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_ਧਾਗੇ,
        .n_tot_chunks      = n_ਪੰਕਤੀਆਂ,
        .n_chunks_ਪ੍ਰਤੀ_task = chunks_ਪ੍ਰਤੀ_thread,
        .k_ਖੰਡ           = k_ਖੰਡ,
        .ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ       = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_ਮੁੱਲਾਂ(ne11) : (ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .ਸ਼ੁਰੂ_ਪੰਕਤੀ         = ਸ਼ੁਰੂ_ਪੰਕਤੀ,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    ਕਾਮਾ_callback_t ਕਾਮਾ_fn = ne11 == 1 ? transfer_activation_chunk_gathered_ਕਾਮਾ_flat_fn :
                                              transfer_activation_chunk_gathered_ਕਾਮਾ_fn;

    ਜੇ (actual_ਧਾਗੇ <= 1) {
        ਕਾਮਾ_fn(1, 0, &state);
    } ਨਹੀਂਤਾਂ {
        ਕਾਮਾ_pool_ਲੜੀਵਾਰ_func(ctx->ਕਾਮਾ_pool, ਕਾਮਾ_fn, &state, actual_ਧਾਗੇ);
    }
}

ਸਥਿਰ ਖਾਲੀ transfer_output_chunk_scattered_threaded(
            ਬਣਤਰ htp_context *ctx,
            ਦਸ਼ਮਲਵ *dst,
            ਅਚਲ __fp16 *vtcm_src,
            ਪੂਰਨ ਸ਼ੁਰੂ_ਪੰਕਤੀ,
            ਪੂਰਨ n_ਪੰਕਤੀਆਂ,
            ਪੂਰਨ n_ਖਾਨੇ_ਗਿਣਤੀ,
            ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping *ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
            ਪੂਰਨ cur_a,
            ਪੂਰਨ mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            ਪੂਰਨ cne1,
            ਪੂਰਨ n_ਧਾਗੇ) {
    ਜੇ (n_ਪੰਕਤੀਆਂ <= 0) ਵਾਪਸ;
    ਪੂਰਨ chunks_ਪ੍ਰਤੀ_thread = hmx_ceil_div(n_ਪੰਕਤੀਆਂ, n_ਧਾਗੇ);
    chunks_ਪ੍ਰਤੀ_thread = hex_align_ਉੱਪਰ(chunks_ਪ੍ਰਤੀ_thread, 2);

    ਪੂਰਨ actual_ਧਾਗੇ = hmx_ceil_div(n_ਪੰਕਤੀਆਂ, chunks_ਪ੍ਰਤੀ_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_ਧਾਗੇ,
        .n_tot_chunks      = n_ਪੰਕਤੀਆਂ,
        .n_chunks_ਪ੍ਰਤੀ_task = chunks_ਪ੍ਰਤੀ_thread,
        .n_ਖਾਨੇ_ਗਿਣਤੀ            = n_ਖਾਨੇ_ਗਿਣਤੀ,
        .ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ       = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .ਸ਼ੁਰੂ_ਪੰਕਤੀ         = ਸ਼ੁਰੂ_ਪੰਕਤੀ,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    ਜੇ (actual_ਧਾਗੇ <= 1) {
        transfer_output_chunk_scattered_ਕਾਮਾ_fn(1, 0, &state);
    } ਨਹੀਂਤਾਂ {
        ਕਾਮਾ_pool_ਲੜੀਵਾਰ_func(ctx->ਕਾਮਾ_pool, transfer_output_chunk_scattered_ਕਾਮਾ_fn, &state, actual_ਧਾਗੇ);
    }
}

ਸਥਿਰ ਪੂਰਨ hmx_mm_id_2d_f32(ਬਣਤਰ htp_context *ctx,
                                         dma_ਕਤਾਰ *ਭਾਰ_dma,
                                         ਦਸ਼ਮਲਵ *ਸੀਮਿਤ dst,
                                         ਅਚਲ ਦਸ਼ਮਲਵ *activation,
                                         dma_addr_t ਭਾਰ,
                                         ਪੂਰਨ m, ਪੂਰਨ k, ਪੂਰਨ n,
                                         ਪੂਰਨ k_valid,
                                         ਪੂਰਨ ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         ਪੂਰਨ ਭਾਰ_stride,
                                         ਪੂਰਨ ਭਾਰ_type,
                                         ਪੂਰਨ tile_ਮਾਪ,
                                         ਪੂਰਨ aligned_tile_ਮਾਪ,
                                         ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping *ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
                                         ਪੂਰਨ cur_a,
                                         ਪੂਰਨ mapping_stride,
                                         ਪੂਰਨ m_ਸ਼ੁਰੂ,
                                         ਪੂਰਨ m_ਸਿਰਾ,
                                         ਪੂਰਨ n_ਧਾਗੇ) {
    ਬਣਤਰ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, 0);

    ਅਚਲ ਪੂਰਨ cne1 = m;
    ਅਚਲ ਪੂਰਨ m_core = m_ਸਿਰਾ - m_ਸ਼ੁਰੂ;
    ਅਚਲ ਪੂਰਨ m_core_padded = hex_align_ਉੱਪਰ(m_core > 0 ? m_core : 1, 32);

    ਜੇ (k % 32 != 0 || n % 32 != 0) { ਵਾਪਸ -1; }
    ਜੇ (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { ਵਾਪਸ -1; }

    size_t ਪੰਕਤੀ_stride = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(ਭਾਰ_type, k);
    ਜੇ (ਪੰਕਤੀ_stride == 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਪੂਰਨ n_k_tiles = k / HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
    ਅਚਲ uint32_t n_ਭਾਰ_tiles = k / htp_tensor_get_tile_k(ਭਾਰ_type);
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ n_k_tiles_div = init_fastdiv_ਮੁੱਲਾਂ(n_k_tiles);

    ਅਚਲ size_t vec_dot_ਮਾਪ = k * ਆਕਾਰ(__fp16);
    ਅਚਲ size_t vtcm_budget  = ctx->vtcm_ਮਾਪ;
    size_t vtcm_ਵਰਤਿਆ = 0;

    ਅਚਲ bool is_tiled = (tile_ਮਾਪ > 0);
    tiled_dequant_task_fn_t dequant_ਕਾਮਾ_fn = hmx_get_dequant_ਕਾਮਾ_fn(ਭਾਰ_type, is_tiled);

    ਅਚਲ uint32_t dma_dst_stride  = is_tiled ? aligned_tile_ਮਾਪ : ਪੰਕਤੀ_stride;
    ਅਚਲ uint32_t dma_src_stride  = is_tiled ? tile_ਮਾਪ : ਭਾਰ_stride;
    ਅਚਲ uint32_t dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ = is_tiled ? tile_ਮਾਪ : ਪੰਕਤੀ_stride;

    ਅਚਲ size_t qweight_ਪੰਕਤੀ_stride = is_tiled ? (size_t)(n_ਭਾਰ_tiles * aligned_tile_ਮਾਪ) / 32 : 0;
    ਅਚਲ size_t ਭਾਰ_ਪੰਕਤੀ_stride = is_tiled ? qweight_ਪੰਕਤੀ_stride : ਪੰਕਤੀ_stride;

    size_t ਮਾਪ_ਪ੍ਰਤੀ_n = 0, ਮਾਪ_ਪ੍ਰਤੀ_m = 0, ਮਾਪ_ਪ੍ਰਤੀ_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(ਭਾਰ_type, k, /*pipeline=*/false, aligned_tile_ਮਾਪ,
                                  &ਮਾਪ_ਪ੍ਰਤੀ_n, &ਮਾਪ_ਪ੍ਰਤੀ_m, &ਮਾਪ_ਪ੍ਰਤੀ_mn);

    ਅਚਲ size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_ਪੰਕਤੀਆਂ = 0, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ = 0;
    ਜੇ (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, ਮਾਪ_ਪ੍ਰਤੀ_n, ਮਾਪ_ਪ੍ਰਤੀ_m, ਮਾਪ_ਪ੍ਰਤੀ_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_ਲਾਗਤ_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_ਲਾਗਤ_A_CONVERT, &m_chunk_n_ਪੰਕਤੀਆਂ, &n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ, &vtcm_ਵਰਤਿਆ)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        ਵਾਪਸ -1;
    }

    ਅਚਲ size_t ਭਾਰ_area_ਮਾਪ = hex_align_ਉੱਪਰ(n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ * ਭਾਰ_ਪੰਕਤੀ_stride, HTP_MM_HMX_TILE_ਮਾਪ);
    ਅਚਲ size_t act_area_ਮਾਪ    = hex_align_ਉੱਪਰ(m_chunk_n_ਪੰਕਤੀਆਂ * vec_dot_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ);
    ਅਚਲ size_t output_area_ਮਾਪ = hex_align_ਉੱਪਰ(m_chunk_n_ਪੰਕਤੀਆਂ * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ * ਆਕਾਰ(__fp16), HTP_MM_HMX_TILE_ਮਾਪ);

    size_t scratch0_ਮਾਪ = hex_align_ਉੱਪਰ(n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ * vec_dot_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_ਅਧਾਰ;
    __fp16  *vtcm_ਭਾਰ   = ਭਾਰ_area_ਮਾਪ ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, ਭਾਰ_area_ਮਾਪ) : ਨਿਰਾ;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_ਮਾਪ);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_ਮਾਪ);
    ਖਾਲੀ    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_ਮਾਪ);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_ਵਰਤਿਆ = vtcm_ptr - (uint8_t *) ctx->vtcm_ਅਧਾਰ;
    ਜੇ (vtcm_ਵਰਤਿਆ > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_ਵਰਤਿਆ, vtcm_budget);
        ਵਾਪਸ -1;
    }

    hmx_init_ਕਾਲਮ_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_ਕੰਮ_t ਕੰਮ;
    ਖਾਲੀ *vtcm_raw_bufs[2] = { vtcm_ਭਾਰ, dequant_ਕਾਮਾ_fn ? vtcm_ਭਾਰ : vtcm_scratch0 };
    ਖਾਲੀ *vtcm_ਭਾਰ_bufs[2] = { dequant_ਕਾਮਾ_fn ? vtcm_scratch0 : vtcm_ਭਾਰ, vtcm_scratch0 };
    ਅਚਲ ਪੂਰਨ n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);

    ਲਈ (size_t mr = (size_t) m_ਸ਼ੁਰੂ; mr < (size_t) m_ਸਿਰਾ; mr += m_chunk_n_ਪੰਕਤੀਆਂ) {
        ਅਚਲ size_t n_ਪੰਕਤੀਆਂ = hex_smin((size_t) m_ਸਿਰਾ - mr, m_chunk_n_ਪੰਕਤੀਆਂ);
        ਅਚਲ size_t n_ਪੰਕਤੀ_tiles = hmx_ceil_div(n_ਪੰਕਤੀਆਂ, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (ਪੂਰਨ) mr, (ਪੂਰਨ) n_ਪੰਕਤੀਆਂ, k,
            ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_ਧਾਗੇ, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        ਜੇ (n > 0) {
            ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ = hex_smin((size_t) n, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
            ਅਚਲ uint32_t ਉਚਾਈ = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ;
            dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[0], ਭਾਰ),
                           dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ);
        }

        ਲਈ (ਪੂਰਨ i = 0; i < n_chunk_cnt; ++i) {
            ਅਚਲ size_t nc = (size_t) i * n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
            ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ = hex_smin((size_t) n - nc, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
            ਅਚਲ size_t n_ਖਾਨਾ_tiles = hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ, HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);

            // A: Wait for weight DMA
            ਖਾਲੀ * curr_raw = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(ਭਾਰ_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_ਭਾਰ_chunk_ਮੰਜ਼ਿਲ_fp16_tiles(
                ctx, vtcm_ਭਾਰ_bufs[i % 2], curr_raw,
                n_ਖਾਨੇ_ਗਿਣਤੀ, k, ਪੰਕਤੀ_stride, ਭਾਰ_type,
                n_k_tiles, n_k_tiles_div, dequant_ਕਾਮਾ_fn, n_ਧਾਗੇ
            );

            // Start weight DMA for the next chunk early
            ਅਚਲ size_t nc_ਅਗਲਾ = nc + n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ;
            ਜੇ (nc_ਅਗਲਾ < (size_t) n) {
                ਅਚਲ size_t n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ = hex_smin((size_t) n - nc_ਅਗਲਾ, n_chunk_n_ਖਾਨੇ_ਗਿਣਤੀ);
                ਅਚਲ uint32_t ਉਚਾਈ_ਅਗਲਾ = is_tiled ? hmx_ceil_div(n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ, 32) * n_ਭਾਰ_tiles : n_ਖਾਨੇ_ਗਿਣਤੀ_ਅਗਲਾ;
                dma_ਕਤਾਰ_ਧੱਕੋ(ਭਾਰ_dma, dma_make_ਅੰਕੜੇ(vtcm_raw_bufs[(i + 1) % 2], ਭਾਰ + nc_ਅਗਲਾ * ਭਾਰ_stride),
                               dma_dst_stride, dma_src_stride, dma_ਚੌੜਾਈ_ਬਾਈਟਾਂ, ਉਚਾਈ_ਅਗਲਾ);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_ਕੰਮ_init(&ਕੰਮ, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_ਭਾਰ_bufs[i % 2], vtcm_scales, n_ਪੰਕਤੀ_tiles, n_ਖਾਨਾ_tiles, k / HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
            hmx_ਕਤਾਰ_ਧੱਕੋ(ctx->hmx_ਕਤਾਰ, hmx_ਕਤਾਰ_make_desc(hmx_matmul_ਕਾਮਾ_fn, &ਕੰਮ));
            hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (ਪੂਰਨ) mr, (ਪੂਰਨ) n_ਪੰਕਤੀਆਂ, (ਪੂਰਨ) n_ਖਾਨੇ_ਗਿਣਤੀ,
                ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_ਧਾਗੇ);
        }
    }

    ਵਾਪਸ 0;
}

// --- Dispatchers and Public Entry Points ---

ਸਥਿਰ ਪੂਰਨ hmx_mm_op_matmul(ਬਣਤਰ htp_ops_context * octx, ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    ਪੂਰਨ k = (ਪੂਰਨ) src0->ne[0];
    ਪੂਰਨ n = (ਪੂਰਨ) src0->ne[1];
    ਅਚਲ ਪੂਰਨ m_ਕੁੱਲ    = (ਪੂਰਨ) act->ne[1];
    ਅਚਲ uint32_t act_elem_ਮਾਪ = (act->type == HTP_TYPE_F16) ? ਆਕਾਰ(__fp16) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ ਪੂਰਨ act_stride = (ਪੂਰਨ) (act->nb[1] / act_elem_ਮਾਪ);

    ਪੂਰਨ m_ਸ਼ੁਰੂ = 0;
    ਪੂਰਨ m_ਪੰਕਤੀਆਂ  = m_ਕੁੱਲ;
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        bool can_ਵੰਡੋ = htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ)) &&
                         htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(act, act_elem_ਮਾਪ);
        ਜੇ (src2 && src2->ne[1] > 1 && !htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(src2, ਆਕਾਰ(ਦਸ਼ਮਲਵ))) {
            can_ਵੰਡੋ = false;
        }
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ((uint32_t) m_ਕੁੱਲ, can_ਵੰਡੋ ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        m_ਸ਼ੁਰੂ = (ਪੂਰਨ) range.ਸ਼ੁਰੂ;
        m_ਪੰਕਤੀਆਂ  = (ਪੂਰਨ) range.ਗਿਣਤੀ_2;
    }

    ਜੇ (m_ਪੰਕਤੀਆਂ == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_ਬਾਈਟਾਂ = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    ਜੇ (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        src2_addr = src2->ਅੰਕੜੇ + m_ਸ਼ੁਰੂ * src2_stride * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        src2_ਬਾਈਟਾਂ = (size_t) kparams->vtcm_bias_ਮਾਪ;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ਅਚਲ ਪੂਰਨ dst_stride = (ਪੂਰਨ)(dst->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ       * dst_ptr = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ + m_ਸ਼ੁਰੂ * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ਅਚਲ dma_addr_t act_addr = act->ਅੰਕੜੇ + (size_t) m_ਸ਼ੁਰੂ * act_stride * act_elem_ਮਾਪ;

    ਪੂਰਨ ret = -1;
    ਅਚਲ ਪੂਰਨ n_ਧਾਗੇ = kparams->n_ਧਾਗੇ;
    ਜੇ (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        hmx_mm_batched_params_t batch_params = {
            .dst               = dst_ptr,
            .bias_addr         = src2_addr,
            .bias_ਬਾਈਟਾਂ        = src2_ਬਾਈਟਾਂ,
            .act_dma_addr      = act_addr,
            .ਭਾਰ            = src0->ਅੰਕੜੇ,
            .ਭਾਰ_dma        = octx->ctx->dma[0],
            .m                 = m_ਪੰਕਤੀਆਂ,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_ਮਾਪ     = act_elem_ਮਾਪ,
            .ਭਾਰ_stride     = (ਪੂਰਨ) src0->nb[1],
            .ਭਾਰ_type       = (ਪੂਰਨ) src0->type,
            .dst_stride        = dst_stride,
            .dst_ਖਾਨੇ_ਗਿਣਤੀ          = (ਪੂਰਨ) dst->ne[0],
            .bias_stride       = src2_stride,
            .tile_ਮਾਪ         = kparams->tile_ਮਾਪ,
            .aligned_tile_ਮਾਪ = kparams->aligned_tile_ਮਾਪ,
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
                             kparams->pipeline, n_ਧਾਗੇ,
                             kparams->n_act_ਧਾਗੇ,
                             &kparams->div_n_act_ਧਾਗੇ,
                             &kparams->div_ne00_padded,
                             kparams->vtcm_ਮਾਪ);
    } ਨਹੀਂਤਾਂ {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_ਬਾਈਟਾਂ,
            act_addr, src0->ਅੰਕੜੇ,
            m_ਪੰਕਤੀਆਂ, k, n, act_stride, act_elem_ਮਾਪ, (ਪੂਰਨ) src0->nb[1], (ਪੂਰਨ) src0->type, (ਪੂਰਨ) act->ne[0],
            dst_stride, src2_stride, (ਪੂਰਨ)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_ਧਾਗੇ,
            kparams->n_act_ਧਾਗੇ,
            &kparams->div_n_act_ਧਾਗੇ,
            &kparams->div_ne00_padded,
            kparams->tile_ਮਾਪ, kparams->aligned_tile_ਮਾਪ, kparams->vtcm_ਮਾਪ
        );
    }

    ਜੇ (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERR;
    }
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_mm_tensor_collapse_ਪੰਕਤੀਆਂ(ਬਣਤਰ htp_tensor * c, ਅਚਲ ਬਣਤਰ htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

ਸਥਿਰ ਪੂਰਨ op_matmul_impl(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;

    ਅਚਲ ਪੂਰਨ ਸਥਿਤੀ = htp_mm_init_context(octx, kparams);
    ਜੇ (ਸਥਿਤੀ != HTP_ਸਥਿਤੀ_OK) {
        ਵਾਪਸ ਸਥਿਤੀ;
    }

    ਜੇ (kparams->n_hmx) {
        ਵਾਪਸ hmx_mm_op_matmul(octx, kparams);
    }

    ਵਾਪਸ hvx_mm_matmul(octx);
}

ਪੂਰਨ op_matmul(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;

    ਜੇ (kparams->collapse) {
        ਅਚਲ ਬਣਤਰ htp_tensor * act  = octx->src[1];
        ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;
        ਅਚਲ uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ਅਚਲ uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        ਬਣਤਰ htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_ਪੰਕਤੀਆਂ(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_ਪੰਕਤੀਆਂ(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        ਬਣਤਰ htp_tensor src2_collapsed;
        ਅਚਲ ਬਣਤਰ htp_tensor * src2 = octx->src[2];
        ਜੇ (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ਅਚਲ uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_ਪੰਕਤੀਆਂ(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        ਅਚਲ ਪੂਰਨ ਸਥਿਤੀ = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        ਜੇ (src2) {
            octx->src[2] = src2;
        }
        ਵਾਪਸ ਸਥਿਤੀ;
    }

    ਵਾਪਸ op_matmul_impl(octx);
}

ਸਥਿਰ ਪੂਰਨ hmx_mm_op_matmul_id(
    ਬਣਤਰ htp_ops_context * octx,
    ਬਣਤਰ htp_mm_context * mmctx
) {
    ਅਚਲ uint32_t *                ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts;
    ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ       = mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ;
    htp_matmul_tensors_preamble;
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ ਪੂਰਨ n_ids = octx->src[2]->ne[0];
    ਅਚਲ ਪੂਰਨ n_as  = ne02;

    ਅਚਲ bool mdev_ਵੰਡੋ = (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) && htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1 && !mdev_ਵੰਡੋ && octx->ctx->mdev.idx > 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }
    uint32_t n_ਸਰਗਰਮ = 0;
    ਜੇ (mdev_ਵੰਡੋ) {
        ਲਈ (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            ਜੇ (ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[a] > 0) n_ਸਰਗਰਮ++;
        }
    }
    ਅਚਲ bool expert_ਵੰਡੋ = mdev_ਵੰਡੋ && (n_ਸਰਗਰਮ >= octx->ctx->mdev.ਗਿਣਤੀ_2);

    uint32_t ਨਿਸ਼ਾਨਾ_dev = 0;
    ਲਈ (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ਅਚਲ int32_t cne1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a];
        ਜੇ (cne1 == 0) ਜਾਰੀ;

        ਅਚਲ ਪੂਰਨ m_padded = hex_align_ਉੱਪਰ(cne1, 32);
        ਪੂਰਨ m_ਸ਼ੁਰੂ = 0, m_ਸਿਰਾ = m_padded;
        ਜੇ (expert_ਵੰਡੋ) {
            ਅਚਲ bool my_expert = (ਨਿਸ਼ਾਨਾ_dev == octx->ctx->mdev.idx);
            ਜੇ (++ਨਿਸ਼ਾਨਾ_dev == octx->ctx->mdev.ਗਿਣਤੀ_2) {
                ਨਿਸ਼ਾਨਾ_dev = 0;
            }
            ਜੇ (!my_expert) ਜਾਰੀ;
        } ਨਹੀਂਤਾਂ ਜੇ (mdev_ਵੰਡੋ) {
            ਅਚਲ bool can_ਵੰਡੋ = (uint32_t) cne1 >= octx->ctx->mdev.ਗਿਣਤੀ_2;
            ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ((uint32_t) m_padded, can_ਵੰਡੋ ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
            m_ਸ਼ੁਰੂ = (ਪੂਰਨ) range.ਸ਼ੁਰੂ;
            m_ਸਿਰਾ   = (ਪੂਰਨ) (range.ਸ਼ੁਰੂ + range.ਗਿਣਤੀ_2);
        }
        ਜੇ (m_ਸ਼ੁਰੂ >= m_ਸਿਰਾ) ਜਾਰੀ;

        ਪੂਰਨ ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (ਦਸ਼ਮਲਵ*) dst->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ*) act->ਅੰਕੜੇ,
                                   src0->ਅੰਕੜੇ + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (ਪੂਰਨ) src0->nb[1], (ਪੂਰਨ) src0->type,
                                   kparams->tile_ਮਾਪ, kparams->aligned_tile_ਮਾਪ,
                                   ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, cur_a, mmctx->mapping_stride,
                                   m_ਸ਼ੁਰੂ, m_ਸਿਰਾ, (ਪੂਰਨ) octx->n_ਧਾਗੇ);
        ਜੇ (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਪੂਰਨ hvx_mm_matmul_id(
    ਬਣਤਰ htp_ops_context * octx,
    ਬਣਤਰ htp_mm_context * mmctx,
    work_ਕਤਾਰ_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ਅਚਲ uint32_t src0_ਪੰਕਤੀ_ਮਾਪ_padded = mmctx->src0_ਪੰਕਤੀ_ਮਾਪ_padded;
    ਅਚਲ uint32_t act_nrows            = mmctx->act_nrows;

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, 0);

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ ids = octx->src[2];
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = nb01;

    ਅਚਲ uint32_t qk = QK_Q8_0_TILED;
    ਅਚਲ uint32_t nb = (ne10 + qk - 1) / qk;
    ਅਚਲ uint32_t ਕੁੱਲ_nb = act_nrows * nb;

    work_ਕਤਾਰ_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ਜੇ (act_nrows < octx->n_ਧਾਗੇ) {
        n_quant_tasks = MIN(ਕੁੱਲ_nb, octx->n_ਧਾਗੇ);
        quant_task_func = htp_mm_act_quant_ਖੰਡ_func(src0->type, act->type);
        ਲਈ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_ਪਹਿਲਾ = (ਕੁੱਲ_nb * ith) / n_quant_tasks;
            uint32_t ib_ਆਖਰੀ  = (ਕੁੱਲ_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_ਪਹਿਲਾ[ith] = ib_ਪਹਿਲਾ;
            mmctx->quant_ib_ਆਖਰੀ[ith]  = ib_ਆਖਰੀ;
            mmctx->quant_r[ith]        = ib_ਪਹਿਲਾ / nb;
            mmctx->quant_c[ith]        = ib_ਪਹਿਲਾ % nb;
        }
    } ਨਹੀਂਤਾਂ {
        n_quant_tasks = MIN(act_nrows, octx->n_ਧਾਗੇ);
        quant_task_func = htp_mm_act_quant_ਪੰਕਤੀ_func(src0->type, act->type);
    }
    size_t act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(src0->type, ne10);

    ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_ਧਾਗੇ,
                                 0, src0_ਪੰਕਤੀ_ਮਾਪ, act_ਪੰਕਤੀ_ਮਾਪ, 0, kparams->n_prefetch, true, false);

    ਅਚਲ size_t vtcm_ਮਾਪ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;

    FARF(ਉੱਚਾ_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_ਬਾਈਟਾਂ, L.act_ਬਾਈਟਾਂ, L.dst_ਬਾਈਟਾਂ, vtcm_ਮਾਪ);

    FARF(ਉੱਚਾ_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ਅੰਕੜੇ,
         act->ਅੰਕੜੇ, dst->ਅੰਕੜੇ);

    // Make sure the reserved vtcm size is sufficient
    ਜੇ (octx->ctx->vtcm_ਮਾਪ < vtcm_ਮਾਪ) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_ਮਾਪ, vtcm_ਮਾਪ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    uint8_t * ਅਚਲ ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_src0);
    mmctx->vtcm_bias     = ਨਿਰਾ;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_ਪੰਕਤੀ_ਮਾਪ_padded;
    mmctx->vtcm_act_stride     = act_ਪੰਕਤੀ_ਮਾਪ;
    mmctx->vtcm_act_raw_stride = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), QK_Q8_0_TILED * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = fastdiv(L.src0_ਬਾਈਟਾਂ, &octx->n_ਧਾਗੇ_div);
    mmctx->vtcm_act_ਮਾਪ_ਪ੍ਰਤੀ_thread  = L.act_ਬਾਈਟਾਂ;
    mmctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = fastdiv(L.dst_ਬਾਈਟਾਂ, &octx->n_ਧਾਗੇ_div);

    mmctx->cur_m_ਸ਼ੁਰੂ = 0;
    mmctx->cur_m_ਪੰਕਤੀਆਂ  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, quant_task_func, mmctx, n_quant_tasks);

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, hvx_mmid_task_func, mmctx, octx->n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਪੂਰਨ hmx_mm_op_matmul_id_nx(
    ਬਣਤਰ htp_ops_context * octx,
    ਬਣਤਰ htp_mm_context * mmctx
) {
    ਅਚਲ uint32_t *                ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts;
    ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ       = mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ;
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_weights = kparams->n_weights;
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ act  = octx->src[n_weights];
    ਅਚਲ ਪੂਰਨ n_as = src0->ne[2];

    bool mdev_ਵੰਡੋ = (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1);
    ਲਈ (uint32_t p = 0; p < n_weights && mdev_ਵੰਡੋ; ++p) {
        ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst = octx->dsts[p];
        mdev_ਵੰਡੋ = htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    }
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1 && !mdev_ਵੰਡੋ && octx->ctx->mdev.idx > 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }
    uint32_t n_ਸਰਗਰਮ = 0;
    ਜੇ (mdev_ਵੰਡੋ) {
        ਲਈ (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            ਜੇ (ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[a] > 0) n_ਸਰਗਰਮ++;
        }
    }
    ਅਚਲ bool expert_ਵੰਡੋ = mdev_ਵੰਡੋ && (n_ਸਰਗਰਮ >= octx->ctx->mdev.ਗਿਣਤੀ_2);

    uint32_t ਨਿਸ਼ਾਨਾ_dev = 0;
    ਲਈ (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ਅਚਲ int32_t cne1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a];
        ਜੇ (cne1 == 0) ਜਾਰੀ;

        ਅਚਲ ਪੂਰਨ m_padded = hex_align_ਉੱਪਰ(cne1, 32);
        ਪੂਰਨ m_ਸ਼ੁਰੂ = 0, m_ਸਿਰਾ = m_padded;
        ਜੇ (expert_ਵੰਡੋ) {
            ਅਚਲ bool my_expert = (ਨਿਸ਼ਾਨਾ_dev == octx->ctx->mdev.idx);
            ਜੇ (++ਨਿਸ਼ਾਨਾ_dev == octx->ctx->mdev.ਗਿਣਤੀ_2) {
                ਨਿਸ਼ਾਨਾ_dev = 0;
            }
            ਜੇ (!my_expert) ਜਾਰੀ;
        } ਨਹੀਂਤਾਂ ਜੇ (mdev_ਵੰਡੋ) {
            ਅਚਲ bool can_ਵੰਡੋ = (uint32_t) cne1 >= octx->ctx->mdev.ਗਿਣਤੀ_2;
            ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ((uint32_t) m_padded, can_ਵੰਡੋ ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
            m_ਸ਼ੁਰੂ = (ਪੂਰਨ) range.ਸ਼ੁਰੂ;
            m_ਸਿਰਾ   = (ਪੂਰਨ) (range.ਸ਼ੁਰੂ + range.ਗਿਣਤੀ_2);
        }
        ਜੇ (m_ਸ਼ੁਰੂ >= m_ਸਿਰਾ) ਜਾਰੀ;

        ਲਈ (uint32_t p = 0; p < n_weights; ++p) {
            ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src_w = octx->src[p];
            ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst   = octx->dsts[p];
            ਜੇ (!src_w || !dst) ਜਾਰੀ;

            ਪੂਰਨ ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (ਦਸ਼ਮਲਵ*) dst->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ*) act->ਅੰਕੜੇ,
                                       src_w->ਅੰਕੜੇ + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (ਪੂਰਨ) src_w->nb[1], (ਪੂਰਨ) src_w->type,
                                       kparams->tile_ਮਾਪ, kparams->aligned_tile_ਮਾਪ,
                                       ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, cur_a, mmctx->mapping_stride,
                                       m_ਸ਼ੁਰੂ, m_ਸਿਰਾ, (ਪੂਰਨ) octx->n_ਧਾਗੇ);
            ਜੇ (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
            }
        }
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਪੂਰਨ hvx_mm_matmul_id_nx(
    ਬਣਤਰ htp_ops_context * octx,
    ਬਣਤਰ htp_mm_context * mmctx,
    work_ਕਤਾਰ_func_t hvx_mmid_task_func
) {
    ਅਚਲ uint32_t src0_ਪੰਕਤੀ_ਮਾਪ_padded = mmctx->src0_ਪੰਕਤੀ_ਮਾਪ_padded;
    ਅਚਲ uint32_t act_nrows            = mmctx->act_nrows;

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, 0);

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_weights = kparams->n_weights;
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ act  = octx->src[n_weights];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ ids  = octx->src[n_weights + 1];
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = src0->nb[1];

    ਅਚਲ uint32_t qk = QK_Q8_0_TILED;
    ਅਚਲ uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ਅਚਲ uint32_t ਕੁੱਲ_nb = act_nrows * nb;

    work_ਕਤਾਰ_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ਜੇ (act_nrows < octx->n_ਧਾਗੇ) {
        n_quant_tasks = MIN(ਕੁੱਲ_nb, octx->n_ਧਾਗੇ);
        quant_task_func = htp_mm_act_quant_ਖੰਡ_func(src0->type, act->type);
        ਲਈ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_ਪਹਿਲਾ = (ਕੁੱਲ_nb * ith) / n_quant_tasks;
            uint32_t ib_ਆਖਰੀ  = (ਕੁੱਲ_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_ਪਹਿਲਾ[ith] = ib_ਪਹਿਲਾ;
            mmctx->quant_ib_ਆਖਰੀ[ith]  = ib_ਆਖਰੀ;
            mmctx->quant_r[ith]        = ib_ਪਹਿਲਾ / nb;
            mmctx->quant_c[ith]        = ib_ਪਹਿਲਾ % nb;
        }
    } ਨਹੀਂਤਾਂ {
        n_quant_tasks = MIN(act_nrows, octx->n_ਧਾਗੇ);
        quant_task_func = htp_mm_act_quant_ਪੰਕਤੀ_func(src0->type, act->type);
    }
    size_t act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(src0->type, act->ne[0]);

    ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_ਧਾਗੇ,
                                 0, src0_ਪੰਕਤੀ_ਮਾਪ, act_ਪੰਕਤੀ_ਮਾਪ, 0, kparams->n_prefetch, true, false);

    ਅਚਲ size_t vtcm_ਮਾਪ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;

    ਜੇ (octx->ctx->vtcm_ਮਾਪ < vtcm_ਮਾਪ) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_ਮਾਪ, vtcm_ਮਾਪ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    uint8_t * ਅਚਲ ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_ਪੰਕਤੀ_ਮਾਪ;
    mmctx->vtcm_act_raw_stride = hex_ਗੇੜਾ_ਉੱਪਰ(act->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ), QK_Q8_0_TILED * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = fastdiv(L.src0_ਬਾਈਟਾਂ, &octx->n_ਧਾਗੇ_div);
    mmctx->vtcm_act_ਮਾਪ_ਪ੍ਰਤੀ_thread  = L.act_ਬਾਈਟਾਂ;
    mmctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = fastdiv(L.dst_ਬਾਈਟਾਂ, &octx->n_ਧਾਗੇ_div);

    mmctx->cur_m_ਸ਼ੁਰੂ = 0;
    mmctx->cur_m_ਪੰਕਤੀਆਂ  = act_nrows;

    FARF(ਉੱਚਾ_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.ਕੁੱਲ_ਬਾਈਟਾਂ, octx->ctx->vtcm_ਮਾਪ, octx->n_ਧਾਗੇ);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, quant_task_func, mmctx, n_quant_tasks);

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, hvx_mmid_task_func, mmctx, octx->n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਜਾਂਚ_expert_ids_n(
    ਅਚਲ ਬਣਤਰ htp_tensor * ids,
    ਅਚਲ uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
    uint32_t mapping_stride
) {
    ਅਚਲ size_t ids_nb1 = ids->nb[1];
    ਅਚਲ uint8_t * ids_ਅੰਕੜੇ = (ਅਚਲ uint8_t *) ids->ਅੰਕੜੇ;

    ਲਈ (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ਅਚਲ int32_t * ਪੰਕਤੀ_ptr = (ਅਚਲ int32_t *) (ids_ਅੰਕੜੇ + iid1 * ids_nb1);
        ਲਈ (uint32_t id = 0; id < n_ids; ++id) {
            ਅਚਲ int32_t i02 = ਪੰਕਤੀ_ptr[id];
            ਜੇ (i02 < 0) {
                ਜਾਰੀ;
            }
            ਪ੍ਰਤਿਜ੍ਞਾ(i02 < n_as);

            ਜੇ (ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ) {
                ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[i02 * mapping_stride + counts[i02]] = (ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਜਾਂਚ_expert_ids(
    ਅਚਲ ਬਣਤਰ htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
    uint32_t mapping_stride
) {
    ਅਚਲ size_t ids_nb0 = ids->nb[0];

    ਜੇ (ids_nb0 == 4) {
        ਚੋਣ (n_ids) {
            ਹਾਲਤ 8:  ਜਾਂਚ_expert_ids_n(ids, 8,     n_as, counts, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, mapping_stride); ਰੋਕੋ;
            ਹਾਲਤ 4:  ਜਾਂਚ_expert_ids_n(ids, 4,     n_as, counts, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, mapping_stride); ਰੋਕੋ;
            ਹਾਲਤ 2:  ਜਾਂਚ_expert_ids_n(ids, 2,     n_as, counts, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, mapping_stride); ਰੋਕੋ;
            ਮੂਲ: ਜਾਂਚ_expert_ids_n(ids, n_ids, n_as, counts, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, mapping_stride); ਰੋਕੋ;
        }
    } ਨਹੀਂਤਾਂ {
        // Strided fallback
        ਅਚਲ size_t ids_nb1 = ids->nb[1];
        ਅਚਲ uint8_t * ids_ਅੰਕੜੇ = (ਅਚਲ uint8_t *) ids->ਅੰਕੜੇ;
        ਲਈ (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ਅਚਲ int32_t * ਪੰਕਤੀ_ptr = (ਅਚਲ int32_t *) (ids_ਅੰਕੜੇ + iid1 * ids_nb1);
            ਲਈ (uint32_t id = 0; id < n_ids; ++id) {
                ਅਚਲ int32_t i02 = *(ਅਚਲ int32_t *) ((ਅਚਲ uint8_t *) ਪੰਕਤੀ_ptr + id * ids_nb0);
                ਜੇ (i02 < 0) {
                    ਜਾਰੀ;
                }
                ਪ੍ਰਤਿਜ੍ਞਾ(i02 < n_as);

                ਜੇ (ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ) {
                    ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[i02 * mapping_stride + counts[i02]] = (ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

ਪੂਰਨ op_matmul_id(ਬਣਤਰ htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਬਣਤਰ htp_mm_context mmctx_struct = {0};
    ਬਣਤਰ htp_mm_context * mmctx = &mmctx_struct;

    ਅਚਲ ਪੂਰਨ ਸਥਿਤੀ = htp_mm_init_context(octx, kparams);
    ਜੇ (ਸਥਿਤੀ != HTP_ਸਥਿਤੀ_OK) {
        ਵਾਪਸ ਸਥਿਤੀ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ ids = octx->src[2];
    ਜੇ (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = nb01;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = nb1;

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਪੰਕਤੀ_ਮਾਪ, 128);

    ਅਚਲ uint32_t src0_nrows = ne01;  // per expert
    ਅਚਲ uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ਅਚਲ ਪੂਰਨ n_ids = ids->ne[0];  // n_expert_used
    ਅਚਲ ਪੂਰਨ n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_ਅਧਾਰ;
    uint32_t   mapping_stride    = 1;
    uint32_t * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = (uint32_t *) mapping_buf;
    ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ = ਨਿਰਾ;

    ਜੇ (act_nrows > 1) {
        ਅਚਲ size_t ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts_ਮਾਪ = n_as * ਆਕਾਰ(uint32_t);
        ਪ੍ਰਤਿਜ੍ਞਾ(octx->ctx->ddr_spad_ਮਾਪ >= ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts_ਮਾਪ);

        hex_l2fetch_ਖੰਡ((ਅਚਲ ਖਾਲੀ *) ids->ਅੰਕੜੇ, ids->ne[1] * ids->nb[1]);

        memset(ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, 0, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts_ਮਾਪ);
        ਜਾਂਚ_expert_ids(ids, n_ids, n_as, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, ਨਿਰਾ, 0);

        uint32_t max_ਗਿਣਤੀ = hvx_reduce_max_i32((ਅਚਲ uint8_t *) ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, n_as);
        mapping_stride = max_ਗਿਣਤੀ > 0 ? max_ਗਿਣਤੀ : 1;

        size_t ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_map_ਮਾਪ  = n_as * mapping_stride * ਆਕਾਰ(ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping);
        ਅਚਲ size_t ਕੁੱਲ_map_ਮਾਪ = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts_ਮਾਪ + ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_map_ਮਾਪ;

        ਜੇ (ਕੁੱਲ_map_ਮਾਪ > octx->ctx->ddr_spad_ਮਾਪ) {
            mapping_buf = memalign(128, ਕੁੱਲ_map_ਮਾਪ);
            ਜੇ (!mapping_buf) {
                ਵਾਪਸ HTP_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERR;
            }
        }

        ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = (uint32_t *) mapping_buf;
        ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ       = (ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping *) (mapping_buf + ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts_ਮਾਪ);

        memset(ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, 0, n_as * ਆਕਾਰ(uint32_t));
        ਜਾਂਚ_expert_ids(ids, n_ids, n_as, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, mapping_stride);
    }

    mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts    = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts;
    mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ          = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_ਪੰਕਤੀ_ਮਾਪ_padded = src0_ਪੰਕਤੀ_ਮਾਪ_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_ਸ਼ੁਰੂ          = 0;
    mmctx->cur_m_ਪੰਕਤੀਆਂ           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ਪੂਰਨ s;
    ਜੇ (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } ਨਹੀਂਤਾਂ {
        uint32_t src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
        uint32_t src0_ਪੰਕਤੀ_ਸਿਰਾ   = src0_nrows;
        ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
            ਅਚਲ bool can_ਵੰਡੋ = htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(src0_nrows, can_ਵੰਡੋ ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
            src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
            src0_ਪੰਕਤੀ_ਸਿਰਾ   = range.ਸ਼ੁਰੂ + range.ਗਿਣਤੀ_2;
        }

        ਜੇ (src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ >= src0_ਪੰਕਤੀ_ਸਿਰਾ) {
            ਜੇ (mapping_buf != octx->ctx->ddr_spad_ਅਧਾਰ) {
                ਮੁਕਤ(mapping_buf);
            }
            ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
        }

        ਅਚਲ uint32_t nrows = src0_ਪੰਕਤੀ_ਸਿਰਾ - src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;
        mmctx->src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ = src0_ਪੰਕਤੀ_ਸ਼ੁਰੂ;
        mmctx->src0_ਪੰਕਤੀ_ਸਿਰਾ   = src0_ਪੰਕਤੀ_ਸਿਰਾ;

        mmctx->src0_nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + octx->n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);
        mmctx->src0_nrows_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ(mmctx->src0_nrows_ਪ੍ਰਤੀ_thread, 32);

        ਜੇ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } ਨਹੀਂਤਾਂ {
            s = HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
    }

    ਜੇ (mapping_buf != octx->ctx->ddr_spad_ਅਧਾਰ) {
        ਮੁਕਤ(mapping_buf);
    }

    ਵਾਪਸ s;
}

ਪੂਰਨ op_matmul_id_nx(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;
    ਬਣਤਰ htp_mm_context mmctx_struct = {0};
    ਬਣਤਰ htp_mm_context * mmctx = &mmctx_struct;

    ਅਚਲ ਪੂਰਨ ਸਥਿਤੀ = htp_mm_init_context(octx, kparams);
    ਜੇ (ਸਥਿਤੀ != HTP_ਸਥਿਤੀ_OK) {
        ਵਾਪਸ ਸਥਿਤੀ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    ਅਚਲ uint32_t n_weights = kparams->n_weights;
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ act  = octx->src[n_weights];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ ids  = octx->src[n_weights + 1];
    ਜੇ (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
    ਲਈ (uint32_t p = 0; p < n_weights; p++) {
        ਜੇ (htp_tensor_is_extended(octx->dsts[p])) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = src0->nb[1];
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਪੰਕਤੀ_ਮਾਪ, 128);

    ਅਚਲ uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ਅਚਲ ਪੂਰਨ n_ids = ids->ne[0];
    ਅਚਲ ਪੂਰਨ n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_ਅਧਾਰ;
    uint32_t   mapping_stride    = 1;
    uint32_t * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = (uint32_t *) mapping_buf;
    ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ = ਨਿਰਾ;

    ਜੇ (act_nrows > 1) {
        ਅਚਲ size_t ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts_ਮਾਪ = n_as * ਆਕਾਰ(uint32_t);
        ਪ੍ਰਤਿਜ੍ਞਾ(octx->ctx->ddr_spad_ਮਾਪ >= ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts_ਮਾਪ);

        hex_l2fetch_ਖੰਡ((ਅਚਲ ਖਾਲੀ *) ids->ਅੰਕੜੇ, ids->ne[1] * ids->nb[1]);

        memset(ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, 0, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts_ਮਾਪ);
        ਜਾਂਚ_expert_ids(ids, n_ids, n_as, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, ਨਿਰਾ, 0);

        uint32_t max_ਗਿਣਤੀ = hvx_reduce_max_i32((ਅਚਲ uint8_t *) ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, n_as);
        mapping_stride = max_ਗਿਣਤੀ > 0 ? max_ਗਿਣਤੀ : 1;

        size_t ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_map_ਮਾਪ  = n_as * mapping_stride * ਆਕਾਰ(ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping);
        ਅਚਲ size_t ਕੁੱਲ_map_ਮਾਪ = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts_ਮਾਪ + ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_map_ਮਾਪ;

        ਜੇ (ਕੁੱਲ_map_ਮਾਪ > octx->ctx->ddr_spad_ਮਾਪ) {
            mapping_buf = memalign(128, ਕੁੱਲ_map_ਮਾਪ);
            ਜੇ (!mapping_buf) {
                ਵਾਪਸ HTP_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERR;
            }
        }

        ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = (uint32_t *) mapping_buf;
        ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ       = (ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping *) (mapping_buf + ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts_ਮਾਪ);

        memset(ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, 0, n_as * ਆਕਾਰ(uint32_t));
        ਜਾਂਚ_expert_ids(ids, n_ids, n_as, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, mapping_stride);
    }

    mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts    = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts;
    mmctx->ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ          = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_ਪੰਕਤੀ_ਮਾਪ_padded = src0_ਪੰਕਤੀ_ਮਾਪ_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_ਸ਼ੁਰੂ          = 0;
    mmctx->cur_m_ਪੰਕਤੀਆਂ           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ਪੂਰਨ s;
    ਜੇ (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } ਨਹੀਂਤਾਂ {
        ਜੇ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } ਨਹੀਂਤਾਂ {
            s = HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
    }

    ਜੇ (mapping_buf != octx->ctx->ddr_spad_ਅਧਾਰ) {
        ਮੁਕਤ(mapping_buf);
    }

    ਵਾਪਸ s;
}
ਸਥਿਰ ਪੂਰਨ op_matmul_nx_impl(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;

    ਅਚਲ ਪੂਰਨ ਸਥਿਤੀ = htp_mm_init_context(octx, kparams);
    ਜੇ (ਸਥਿਤੀ != HTP_ਸਥਿਤੀ_OK) {
        ਵਾਪਸ ਸਥਿਤੀ;
    }

    ਜੇ (kparams->n_hmx) {
        ਵਾਪਸ hmx_mm_nx_2d_f32(octx, kparams);
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, 0);

    ਅਚਲ uint32_t n_weights = kparams->n_weights;

    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0]; // first weight
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ act  = octx->src[n_weights]; // activation x

    ਬਣਤਰ htp_mm_context mmctx_struct = {0};
    ਬਣਤਰ htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    ਅਚਲ uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_ਸ਼ੁਰੂ = 0;
    mmctx->cur_m_ਪੰਕਤੀਆਂ  = act_nrows;

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = src0->nb[1];

    ਜੇ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t qk = QK_Q8_0_TILED;
    ਅਚਲ uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ਅਚਲ uint32_t ਕੁੱਲ_nb = act_nrows * nb;

    ਕਾਮਾ_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ਜੇ (act_nrows < octx->n_ਧਾਗੇ) {
        n_quant_tasks = MIN(ਕੁੱਲ_nb, octx->n_ਧਾਗੇ);
        quant_task_func = htp_mm_act_quant_ਖੰਡ_func(src0->type, act->type);
        ਲਈ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_ਪਹਿਲਾ = (ਕੁੱਲ_nb * ith) / n_quant_tasks;
            uint32_t ib_ਆਖਰੀ  = (ਕੁੱਲ_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_ਪਹਿਲਾ[ith] = ib_ਪਹਿਲਾ;
            mmctx->quant_ib_ਆਖਰੀ[ith]  = ib_ਆਖਰੀ;
            mmctx->quant_r[ith]        = ib_ਪਹਿਲਾ / nb;
            mmctx->quant_c[ith]        = ib_ਪਹਿਲਾ % nb;
        }
    } ਨਹੀਂਤਾਂ {
        n_quant_tasks = MIN(act_nrows, octx->n_ਧਾਗੇ);
        quant_task_func = htp_mm_act_quant_ਪੰਕਤੀ_func(src0->type, act->type);
    }

    ਅਚਲ size_t act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(src0->type, act->ne[0]);

    ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_ਧਾਗੇ,
                                 0, src0_ਪੰਕਤੀ_ਮਾਪ, act_ਪੰਕਤੀ_ਮਾਪ, 0, kparams->n_prefetch, false, true);

    ਅਚਲ size_t vtcm_ਮਾਪ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;

    ਜੇ (octx->ctx->vtcm_ਮਾਪ < vtcm_ਮਾਪ) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_ਮਾਪ, vtcm_ਮਾਪ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    uint8_t * ਅਚਲ ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_ਪੰਕਤੀ_ਮਾਪ;
    mmctx->vtcm_act_raw_stride = hex_ਗੇੜਾ_ਉੱਪਰ(act->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ), QK_Q8_0_TILED * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    mmctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = fastdiv(L.src0_ਬਾਈਟਾਂ, &octx->n_ਧਾਗੇ_div);
    mmctx->vtcm_act_ਮਾਪ_ਪ੍ਰਤੀ_thread  = L.act_ਬਾਈਟਾਂ;
    mmctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = fastdiv(L.dst_ਬਾਈਟਾਂ, &octx->n_ਧਾਗੇ_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, quant_task_func, mmctx, n_quant_tasks);

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, hvx_mm_nx_2d_tiled, mmctx, octx->n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_matmul_nx(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) octx->kernel_params;

    ਜੇ (kparams->collapse) {
        ਅਚਲ uint32_t n_weights = kparams->n_weights;
        ਅਚਲ ਬਣਤਰ htp_tensor * act = octx->src[n_weights];
        ਅਚਲ uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ਬਣਤਰ htp_tensor act_collapsed;
        ਬਣਤਰ htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        ਅਚਲ ਬਣਤਰ htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_ਪੰਕਤੀਆਂ(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        ਲਈ (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ਅਚਲ ਬਣਤਰ htp_tensor * d = octx->dsts[p];
            ਅਚਲ uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_ਪੰਕਤੀਆਂ(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        ਅਚਲ ਪੂਰਨ ਸਥਿਤੀ = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        ਲਈ (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        ਵਾਪਸ ਸਥਿਤੀ;
    }

    ਵਾਪਸ op_matmul_nx_impl(octx);
}
