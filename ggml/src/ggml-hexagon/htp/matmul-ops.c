#ہدایت clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>
#ضمن <HAP_compute_res.h>

#ضمن <math.h>
#ضمن <string.h>
#ضمن <stdatomic.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"
#ضمن "hvx-dump.h"
#ضمن "hvx-arith.h"
#ضمن "hvx-reduce.h"

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "matmul-ops.h"
#ضمن "htp-vtcm.h"

تعريفنوع بنية {
    عائم        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_بائٹاں;
    dma_addr_t    act_dma_addr;
    dma_addr_t    بھار;
    dma_قطار *   بھار_dma;
    صحيح           m;
    صحيح           k;
    صحيح           n;
    صحيح           act_stride;
    uint32_t      act_elem_ناپ;  // 4=F32 src1, 2=F16 src1
    صحيح           بھار_stride;
    صحيح           بھار_type;
    صحيح           dst_stride;
    صحيح           dst_خانے_گنتی;
    uint32_t      bias_stride;
    صحيح           tile_ناپ;
    صحيح           aligned_tile_ناپ;
    صحيح           ne02;
    صحيح           ne03;
    صحيح           ne12;
    صحيح           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    صحيح           r2;
    صحيح           r3;
    بنية fastdiv_اقدار div_r2;
    بنية fastdiv_اقدار div_r3;
} hmx_mm_batched_params_t;

ساكن bool htp_matmul_has_extended_بھار(ثابت بنية htp_ops_context * octx, uint32_t n_weights) {
    لكل (uint32_t i = 0; i < n_weights; ++i) {
        إذا (htp_tensor_is_extended(octx->src[i])) {
            أرجع true;
        }
    }
    أرجع false;
}

بنية htp_mm_context {
    ثابت حرف * type;
    بنية htp_ops_context * octx;
    ثابت بنية htp_tensor * act;

    فراغ (*vec_dot_1x1)(ثابت uint32_t n, عائم * محدود s0,
         ثابت فراغ * محدود vx0,
         ثابت فراغ * محدود vy0);

    فراغ (*vec_dot_2x1)(ثابت uint32_t n, عائم * محدود s0,
         ثابت فراغ * محدود vx0, ثابت فراغ * محدود vx1,
         ثابت فراغ * محدود vy0);

    فراغ (*vec_dot_2x2)(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1,
         ثابت فراغ * محدود vx0, ثابت فراغ * محدود vx1,
         ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1);

    فراغ (*vec_dot_32x1)(ثابت uint32_t n, عائم * محدود s,
         ثابت فراغ * محدود vx,
         ثابت فراغ * محدود vy, uint32_t valid_صفاں,
         ثابت عائم * محدود sz);

    فراغ (*vec_dot_32x2)(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1,
         ثابت فراغ * محدود vx,
         ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں,
         ثابت عائم * محدود sz0, ثابت عائم * محدود sz1);

    // Precomputed values
    uint32_t src0_nrows_فی_thread;
    uint32_t src0_صف_شروع;
    uint32_t src0_صف_سرا;
    uint32_t src0_صف_ناپ_padded;
    uint32_t act_nrows;
    uint32_t cur_m_شروع;
    uint32_t cur_m_صفاں;

    بنية fastdiv_اقدار mm_div_ne12_ne1;
    بنية fastdiv_اقدار mm_div_ne1;
    بنية fastdiv_اقدار mm_div_r2;
    بنية fastdiv_اقدار mm_div_r3;
    بنية fastdiv_اقدار mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_پہلا[WORK_قطار_MAX_N_دھاگے];
    uint32_t          quant_ib_آخری[WORK_قطار_MAX_N_دھاگے];
    uint32_t          quant_r[WORK_قطار_MAX_N_دھاگے];
    uint32_t          quant_c[WORK_قطار_MAX_N_دھاگے];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_صفاں_فی_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ثابت uint32_t * میٹرکس_صف_counts;
    ثابت بنية mmid_صف_mapping * میٹرکس_صفاں;
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
    uint32_t vtcm_src0_ناپ_فی_thread;
    uint32_t vtcm_act_ناپ_فی_thread;
    uint32_t vtcm_dst_ناپ_فی_thread;
};

ساكن صحيح htp_mm_init_context(
    بنية htp_ops_context * octx,
    ثابت بنية htp_mm_kernel_params * kparams
) {
    إذا (!htp_ops_context_سیٹ_n_دھاگے(octx, (uint32_t) kparams->n_دھاگے)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    إذا (kparams->n_hmx) {
        إذا (kparams->n_act_دھاگے <= 0 || kparams->n_act_دھاگے > (int32_t) octx->n_دھاگے) {
            أرجع HTP_حالت_INVAL_PARAMS;
        }
    }

    أرجع HTP_حالت_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
ساكن ثابت uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
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
ساكن ثابت uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

ساكن ثابت uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#تعریف_کرو htp_matmul_tensors_preamble                                 \
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];         \
    ثابت بنية htp_tensor * محدود src1 = octx->src[1];         \
    ثابت بنية htp_tensor * محدود src2 = octx->src[2];         \
    ثابت بنية htp_tensor * محدود  dst = octx->dst;            \
    ثابت بنية htp_tensor * محدود  act = src1;                 \
                                                                    \
    ثابت uint32_t ne00 = src0->ne[0];                              \
    ثابت uint32_t ne01 = src0->ne[1];                              \
    ثابت uint32_t ne02 = src0->ne[2];                              \
    ثابت uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    ثابت uint32_t ne10 = src1->ne[0];                              \
    ثابت uint32_t ne11 = src1->ne[1];                              \
    ثابت uint32_t ne12 = src1->ne[2];                              \
    ثابت uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    ثابت uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    ثابت uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    ثابت uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    ثابت uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    ثابت uint32_t ne0 = dst->ne[0];                                \
    ثابت uint32_t ne1 = dst->ne[1];                                \
    ثابت uint32_t ne2 = dst->ne[2];                                \
    ثابت uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    ثابت uint32_t nb00 = src0->nb[0];                              \
    ثابت uint32_t nb01 = src0->nb[1];                              \
    ثابت uint32_t nb02 = src0->nb[2];                              \
    ثابت uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    ثابت uint32_t nb10 = src1->nb[0];                              \
    ثابت uint32_t nb11 = src1->nb[1];                              \
    ثابت uint32_t nb12 = src1->nb[2];                              \
    ثابت uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    ثابت uint32_t nb0 = dst->nb[0];                                \
    ثابت uint32_t nb1 = dst->nb[1];                                \
    ثابت uint32_t nb2 = dst->nb[2];                                \
    ثابت uint32_t nb3 = dst->nb[3];

#تعریف_کرو htp_matmul_preamble                                         \
    بنية htp_mm_context * mmctx  = انکڑے;                          \
    بنية htp_ops_context * octx  = mmctx->octx;                   \
    dma_قطار * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_فی_thread = mmctx->src0_nrows_فی_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#ضمن "hvx-mm-kernels-tiled.h"
#ضمن "hvx-mm-kernels-float.h"
#ضمن "hmx-mm-kernels-tiled.h"

ساكن فراغ hvx_mm_2d_tiled(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_سرا - mmctx->src0_صف_شروع;
    ثابت uint32_t act_nrows = mmctx->cur_m_صفاں ? mmctx->cur_m_صفاں : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_شروع = mmctx->cur_m_شروع;

    ثابت uint32_t src0_شروع_صف  = mmctx->src0_صف_شروع + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_سرا_صف    = MIN(src0_شروع_صف + src0_nrows_فی_thread, mmctx->src0_صف_سرا);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_صف_ناپ  = nb1;
    ثابت size_t act_صف_ناپ = nb11;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;
    ثابت size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * محدود vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_ناپ_فی_thread  * ith;
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * محدود act_انکڑے = mmctx->vtcm_act;

    ثابت dma_addr_t src0_صف = src0->انکڑے;

    ثابت uint32_t tile_ناپ = htp_tensor_get_tile_ناپ(src0->type);
    ثابت uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_صف_stride = n_k_tiles_w * tile_ناپ;
    uint32_t tile_صف_transfer_ناپ_aligned = n_k_tiles_w * aligned_tile_ناپ;

    uint32_t ct_شروع = src0_شروع_صف / 32;
    uint32_t ct_سرا   = (src0_سرا_صف + 31) / 32;

    uint32_t ڈالو_ct = ct_شروع;
    إذا (src0_شروع_صف < src0_سرا_صف) {
        لكل (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_سرا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + d * tile_صف_transfer_ناپ_aligned,
                           src0_صف + ڈالو_ct * tile_صف_stride), aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
        }
    }

    إذا (src0_شروع_صف >= src0_سرا_صف) {
        أرجع;
    }

    لكل (uint32_t ct = ct_شروع; ct < ct_سرا; ct++) {
        ثابت uint8_t * w_tile = (فراغ *) dma_قطار_نکالو(dma_q).dst;

        صحيح valid_صفاں = (صحيح)ne0 - (صحيح)(ct * 32);
        valid_صفاں = MIN(32, MAX(0, valid_صفاں));

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
            ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_انکڑے + (ir1+0) * act_stride);
            ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_انکڑے + (ir1+1) * act_stride);
            عائم * محدود dst_row0 = (عائم *) (dst->انکڑے + ((cur_m_شروع + ir1+0) * dst_صف_ناپ));
            عائم * محدود dst_row1 = (عائم *) (dst->انکڑے + ((cur_m_شروع + ir1+1) * dst_صف_ناپ));

            عائم * dst_ptr0 = &dst_row0[ct * 32];
            عائم * dst_ptr1 = &dst_row1[ct * 32];

            ثابت عائم * src2_ptr0 = عدم;
            ثابت عائم * src2_ptr1 = عدم;
            إذا (src2) {
                ثابت عائم * محدود src2_row0 = (ثابت عائم *) ((ثابت uint8_t *) src2->انکڑے + ((cur_m_شروع + ir1+0) * src2_stride));
                ثابت عائم * محدود src2_row1 = (ثابت عائم *) ((ثابت uint8_t *) src2->انکڑے + ((cur_m_شروع + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_صفاں, src2_ptr0, src2_ptr1);
        }

        لكل (; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) (act_انکڑے + ir1 * act_stride);
            عائم * محدود dst_صف          = (عائم *) (dst->انکڑے + ((cur_m_شروع + ir1) * dst_صف_ناپ));
            عائم * dst_ptr = &dst_صف[ct * 32];

            ثابت عائم * src2_ptr = عدم;
            إذا (src2) {
                ثابت عائم * محدود src2_صف = (ثابت عائم *) ((ثابت uint8_t *) src2->انکڑے + ((cur_m_شروع + ir1) * src2_stride));
                src2_ptr = &src2_صف[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_خانہ, valid_صفاں, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        إذا (ڈالو_ct < ct_سرا) {
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                           aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
            ڈالو_ct++;
        }
    }
}

ساكن فراغ hvx_mv_2d_tiled(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_سرا - mmctx->src0_صف_شروع;

    ثابت uint32_t src0_شروع_صف  = mmctx->src0_صف_شروع + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_سرا_صف    = MIN(src0_شروع_صف + src0_nrows_فی_thread, mmctx->src0_صف_سرا);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_صف_ناپ  = nb1;
    ثابت size_t act_صف_ناپ = nb11;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_ناپ_فی_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * act_انکڑے = mmctx->vtcm_act;

    عائم * tmp = (عائم *) vtcm_dst_ptr;

    ثابت dma_addr_t src0_صف = src0->انکڑے;

    ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) act_انکڑے;
    عائم * محدود dst_خانہ          = (عائم *) dst->انکڑے;

    ثابت uint32_t tile_ناپ = htp_tensor_get_tile_ناپ(src0->type);
    ثابت uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_صف_stride = n_k_tiles_w * tile_ناپ;
    uint32_t tile_صف_transfer_ناپ_aligned = n_k_tiles_w * aligned_tile_ناپ;

    uint32_t ct_شروع = src0_شروع_صف / 32;
    uint32_t ct_سرا   = (src0_سرا_صف + 31) / 32;

    uint32_t ڈالو_ct = ct_شروع;
    إذا (src0_شروع_صف < src0_سرا_صف) {
        إذا (src2) {
            عائم * vtcm_bias_ptr = (عائم *) mmctx->vtcm_bias + src0_شروع_صف;
            ثابت dma_addr_t src2_addr = src2->انکڑے + src0_شروع_صف * حجم(عائم);
            صحيح ٹکڑا_ناپ = (صحيح)MIN(src0_سرا_صف, ne0) - (صحيح)src0_شروع_صف;
            إذا (ٹکڑا_ناپ > 0) {
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_bias_ptr, src2_addr),
                               ٹکڑا_ناپ * حجم(عائم), ٹکڑا_ناپ * حجم(عائم), ٹکڑا_ناپ * حجم(عائم), 1);
                dma_قطار_نکالو_nowait(dma_q);
            }
        }
        لكل (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_سرا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + d * tile_صف_transfer_ناپ_aligned,
                           src0_صف + ڈالو_ct * tile_صف_stride), aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
        }
    }

    إذا (src0_شروع_صف >= src0_سرا_صف) {
        أرجع;
    }

    لكل (uint32_t ct = ct_شروع; ct < ct_سرا; ct++) {
        ثابت uint8_t * w_tile = (فراغ *) dma_قطار_نکالو(dma_q).dst;

        عائم * dst_ptr = &tmp[ct * 32 - src0_شروع_صف];
        صحيح valid_صفاں = (صحيح)ne0 - (صحيح)(ct * 32);
        valid_صفاں = MIN(32, MAX(0, valid_صفاں));

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_خانہ, valid_صفاں, عدم);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        إذا (ڈالو_ct < ct_سرا) {
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                           aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
            ڈالو_ct++;
        }
    }

    صحيح copy_cnt = (صحيح)MIN(src0_سرا_صف, ne0) - (صحيح)src0_شروع_صف;
    إذا (copy_cnt > 0) {
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct_سرا);
        إذا (src2) {
            hvx_جوڑو_f32_uuu((uint8_t *) &dst_خانہ[src0_شروع_صف],
                            (ثابت uint8_t *) tmp,
                            (ثابت uint8_t *) ((ثابت عائم *) mmctx->vtcm_bias + src0_شروع_صف),
                            copy_cnt);
        } وإلا {
            hvx_copy_f32_ua((uint8_t *) &dst_خانہ[src0_شروع_صف], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_سرا);
    }
}

ساكن فراغ hvx_mm_nx_2d_tiled(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية htp_mm_context * mmctx = انکڑے;
    بنية htp_ops_context * octx = mmctx->octx;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;

    ثابت بنية htp_tensor * محدود act = octx->src[n_weights]; /* x */
    ثابت uint32_t ne10 = act->ne[0];
    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_بھار_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * محدود act_انکڑے        = mmctx->vtcm_act;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    لكل (uint32_t widx = 0; widx < n_weights; widx++) {
        ثابت بنية htp_tensor * محدود src_w = octx->src[widx];
        ثابت بنية htp_tensor * محدود dst   = octx->dsts[widx];
        إذا (!src_w || !dst) استمر;
        dma_قطار * dma_q = octx->ctx->dma[ith];

        ثابت uint32_t tile_ناپ = htp_tensor_get_tile_ناپ(src_w->type);
        ثابت uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(src_w->type);
        ثابت uint32_t ne00 = src_w->ne[0];
        ثابت uint32_t ne01 = src_w->ne[1];
        ثابت size_t dst_صف_ناپ = dst->nb[1];
        ثابت dma_addr_t src_w_صف = src_w->انکڑے;

        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_ناپ;
        ثابت uint32_t tile_صف_transfer_ناپ_aligned = n_k_tiles_w * aligned_tile_ناپ;

        uint32_t src0_شروع_صف = 0;
        uint32_t src0_سرا_صف   = ne01;
        إذا (octx->ctx->mdev.گنتی > 1) {
            ثابت bool can_ونڈو = htp_tensor_can_صف_حصے_کرو(dst, حجم(عائم));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(ne01, can_ونڈو ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            src0_شروع_صف = range.شروع;
            src0_سرا_صف   = range.شروع + range.گنتی;
        }

        ثابت uint32_t nrows = src0_سرا_صف - src0_شروع_صف;
        uint32_t src0_nrows_فی_thread = fastdiv(nrows + nth - 1, &octx->n_دھاگے_div);
        src0_nrows_فی_thread = hex_گیڑا_اتے(src0_nrows_فی_thread, 32);

        ثابت uint32_t شروع_صف = src0_شروع_صف + src0_nrows_فی_thread * ith;
        ثابت uint32_t سرا_صف   = MIN(شروع_صف + src0_nrows_فی_thread, src0_سرا_صف);
        إذا (شروع_صف >= سرا_صف) استمر;

        uint32_t ct_شروع = شروع_صف / 32;
        uint32_t ct_سرا   = (سرا_صف + 31) / 32;

        uint32_t ڈالو_ct = ct_شروع;
        لكل (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_سرا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_بھار_ptr + d * tile_صف_transfer_ناپ_aligned,
                            src_w_صف + ڈالو_ct * tile_صف_stride), aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_شروع; ct < ct_سرا; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_قطار_نکالو(dma_q).dst;
            صحيح valid_صفاں = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
            valid_صفاں = MIN(32, MAX(0, valid_صفاں));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
                ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_انکڑے + (ir1+0) * act_stride);
                ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_انکڑے + (ir1+1) * act_stride);

                عائم * محدود dst_row0 = (عائم *) (dst->انکڑے + ((ir1+0) * dst_صف_ناپ));
                عائم * محدود dst_row1 = (عائم *) (dst->انکڑے + ((ir1+1) * dst_صف_ناپ));
                عائم * dst_ptr0 = &dst_row0[ct * 32];
                عائم * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_صفاں, عدم, عدم);
            }

            لكل (; ir1 < act_nrows; ++ir1) {
                ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) (act_انکڑے + ir1 * act_stride);
                عائم * محدود dst_صف = (عائم *) (dst->انکڑے + (ir1 * dst_صف_ناپ));
                عائم * dst_ptr = &dst_صف[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_خانہ, valid_صفاں, عدم);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (ڈالو_ct < ct_سرا) {
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(w_tile, src_w_صف + ڈالو_ct * tile_صف_stride),
                                aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
                ڈالو_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mm_transfer_act_dma(
    بنية htp_ops_context * octx,
    ثابت بنية htp_mm_kernel_params * kparams,
    ثابت بنية htp_tensor * act,
    uint8_t * dst_ادھار,
    size_t dst_صف_ناپ,
    uint32_t m_شروع,
    uint32_t m_صفاں
) {
    إذا (m_صفاں == 0) {
        أرجع;
    }

    dma_قطار * dma_q = octx->ctx->dma[0];
    ثابت uint32_t ne0 = act->ne[0];
    ثابت size_t elem_ناپ = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت size_t صف_بائٹاں = ne0 * elem_ناپ;
    ثابت size_t act_nb1 = act->nb[1];
    ثابت dma_addr_t act_ادھار = act->انکڑے;

    ثابت bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    إذا (is_contiguous) {
        ثابت dma_addr_t src_addr = act_ادھار + m_شروع * act_nb1;
        dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst_ادھار, src_addr),
                       dst_صف_ناپ, act_nb1, صف_بائٹاں, m_صفاں);
        dma_قطار_نکالو(dma_q);
    } وإلا {
        ثابت uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ثابت bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        لكل (uint32_t ir = 0; ir < m_صفاں; ++ir) {
            ثابت uint32_t ir1 = m_شروع + ir;
            uint32_t i11, i12, i13;
            إذا (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                ثابت uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } وإلا {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                ثابت uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            ثابت dma_addr_t صف_src = act_ادھار + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * صف_dst = dst_ادھار + ir * dst_صف_ناپ;
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(صف_dst, صف_src),
                           dst_صف_ناپ, act_nb1, صف_بائٹاں, 1);
            dma_قطار_نکالو(dma_q);
        }
    }

    إذا (dst_صف_ناپ > صف_بائٹاں) {
        بنية htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_شروع);
        ثابت uint32_t pad_elems = (dst_صف_ناپ - صف_بائٹاں) / elem_ناپ;
        إذا (elem_ناپ == حجم(عائم)) {
            لكل (uint32_t ir = 0; ir < m_صفاں; ++ir) {
                hvx_splat_f32_u(dst_ادھار + ir * dst_صف_ناپ + صف_بائٹاں, 0.0f, pad_elems);
            }
        } وإلا {
            لكل (uint32_t ir = 0; ir < m_صفاں; ++ir) {
                hvx_splat_f16_u(dst_ادھار + ir * dst_صف_ناپ + صف_بائٹاں, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_شروع);
    }
}

#تعریف_کرو QUANTIZE_IMPL(name, log_name, kernel_fn, dst_صف_ناپ_اظہاریہ)                                        \
ساكن فراغ name(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {                                        \
    (فراغ) nth;                                                                                            \
    بنية htp_mm_context * mmctx = انکڑے;                                                                  \
    بنية htp_ops_context * octx = mmctx->octx;                                                           \
    ثابت بنية htp_tensor * src = mmctx->act;                                                            \
    ثابت uint32_t ne0 = src->ne[0];                                                                       \
    ثابت uint32_t nrows = mmctx->cur_m_صفاں ? mmctx->cur_m_صفاں : mmctx->act_nrows;                       \
    ثابت uint32_t nrows_فی_thread = mmctx->n_quant_صفاں_فی_thread;                                      \
                                                                                                           \
    ثابت uint32_t ir_پہلا = nrows_فی_thread * ith;                                                      \
    إذا (ir_پہلا >= nrows) {                                                                               \
        أرجع;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_پہلا);                                        \
                                                                                                           \
    uint8_t * محدود dst = mmctx->vtcm_act;                                                              \
    ثابت uint32_t ir_آخری = MIN(ir_پہلا + nrows_فی_thread, nrows);                                      \
    ثابت size_t raw_صف_ناپ = mmctx->vtcm_act_raw_stride;                                                \
    ثابت size_t dst_صف_ناپ = (dst_صف_ناپ_اظہاریہ);                                                       \
                                                                                                           \
    ثابت uint8_t * محدود src_انکڑے = (ثابت uint8_t *) mmctx->vtcm_act_raw + (raw_صف_ناپ * ir_پہلا); \
    uint8_t * محدود dst_انکڑے = (uint8_t *) dst + (dst_صف_ناپ * ir_پہلا);                             \
    kernel_fn(src_انکڑے, dst_انکڑے, عدم, ne0, ir_آخری - ir_پہلا, raw_صف_ناپ, dst_صف_ناپ);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_پہلا);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled,     "quantize-f32-q8_0_tiled",     quantize_f32_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_صف_ناپ(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled,     "quantize-f32-q8_1_tiled",     quantize_f32_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_صف_ناپ(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_صف_ناپ(ne0))
QUANTIZE_IMPL(quantize_f32_f16_tiled,      "quantize-f32-f16_tiled",      quantize_f32_f16_tiled_kernel,      htp_mm_f16_tiled_صف_ناپ(ne0))
QUANTIZE_IMPL(quantize_f16_f16_tiled,      "quantize-f16-f16_tiled",      quantize_f16_f16_tiled_kernel,      htp_mm_f16_tiled_صف_ناپ(ne0))
QUANTIZE_IMPL(quantize_f32_f32,            "quantize-f32-f32",            quantize_f32_f32_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,            "quantize-f32-f16",            quantize_f32_f16_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,            "quantize-f16-f16",            quantize_f16_f16_kernel,            mmctx->vtcm_act_stride)

ساكن فراغ quantize_f32_q8_0_tiled_بلاک(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = انکڑے;
    إذا (mmctx->quant_ib_پہلا[ith] >= mmctx->quant_ib_آخری[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_بلاک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_پہلا[ith],
        mmctx->quant_ib_آخری[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_صف_ناپ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);
}

ساكن فراغ quantize_f32_q8_1_tiled_بلاک(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = انکڑے;
    إذا (mmctx->quant_ib_پہلا[ith] >= mmctx->quant_ib_آخری[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_بلاک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_پہلا[ith],
        mmctx->quant_ib_آخری[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_صف_ناپ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);
}

ساكن فراغ quantize_f32_q8_1_s16_tiled_بلاک(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = انکڑے;
    إذا (mmctx->quant_ib_پہلا[ith] >= mmctx->quant_ib_آخری[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_بلاک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_پہلا[ith],
        mmctx->quant_ib_آخری[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_صف_ناپ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);
}

ساكن فراغ quantize_f32_f16_tiled_بلاک(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = انکڑے;
    إذا (mmctx->quant_ib_پہلا[ith] >= mmctx->quant_ib_آخری[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_بلاک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_پہلا[ith],
        mmctx->quant_ib_آخری[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_صف_ناپ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);
}

ساكن فراغ quantize_f16_f16_tiled_بلاک(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = انکڑے;
    إذا (mmctx->quant_ib_پہلا[ith] >= mmctx->quant_ib_آخری[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_بلاک_kernel(
        (ثابت __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_پہلا[ith],
        mmctx->quant_ib_آخری[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_صف_ناپ(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
ساكن مضمن کامہ_callback_t htp_mm_act_quant_صف_func(صحيح بھار_type, صحيح act_type) {
    إذا (بھار_type == HTP_TYPE_F16) {
        أرجع (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    إذا (بھار_type == HTP_TYPE_Q2_K) {
        أرجع quantize_f32_q8_1_s16_tiled;
    }
    أرجع htp_mm_بھار_has_وتھ(بھار_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

ساكن مضمن کامہ_callback_t htp_mm_act_quant_بلاک_func(صحيح بھار_type, صحيح act_type) {
    إذا (بھار_type == HTP_TYPE_F16) {
        أرجع (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_بلاک : quantize_f32_f16_tiled_بلاک;
    }
    إذا (بھار_type == HTP_TYPE_Q2_K) {
        أرجع quantize_f32_q8_1_s16_tiled_بلاک;
    }
    أرجع htp_mm_بھار_has_وتھ(بھار_type) ? quantize_f32_q8_1_tiled_بلاک : quantize_f32_q8_0_tiled_بلاک;
}

ساكن فراغ hvx_mm_4d_tiled(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_سرا - mmctx->src0_صف_شروع;
    ثابت uint32_t cur_m_صفاں = mmctx->cur_m_صفاں ? mmctx->cur_m_صفاں : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_شروع = mmctx->cur_m_شروع;

    ثابت uint32_t src0_شروع_صف  = mmctx->src0_صف_شروع + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_سرا_صف    = MIN(src0_شروع_صف + src0_nrows_فی_thread, mmctx->src0_صف_سرا);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_صف_ناپ = nb1;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * محدود act_انکڑے = mmctx->vtcm_act;

    ثابت uint32_t tile_ناپ = htp_tensor_get_tile_ناپ(src0->type);
    ثابت uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(src0->type);

    ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_ناپ;
    ثابت uint32_t tile_صف_transfer_ناپ_aligned = n_k_tiles_w * aligned_tile_ناپ;
    ثابت uint32_t src0_ٹکڑا_stride = ((ne01 + 31) / 32) * tile_صف_stride;

    ثابت uint32_t ct_شروع = src0_شروع_صف / 32;
    ثابت uint32_t ct_سرا   = (src0_سرا_صف + 31) / 32;

    إذا (src0_شروع_صف >= src0_سرا_صف || cur_m_صفاں == 0) {
        أرجع;
    }

    ثابت uint32_t کل_batches = ne12 * ne13;
    ثابت uint32_t b_شروع = fastdiv(cur_m_شروع, &kparams->div_ne1);
    uint32_t b_سرا = fastdiv(cur_m_شروع + cur_m_صفاں + ne11 - 1, &kparams->div_ne1);
    b_سرا = MIN(b_سرا, کل_batches);

    uint32_t b_grp_شروع = b_شروع;
    طالما (b_grp_شروع < b_سرا) {
        ثابت uint32_t b3 = fastdiv(b_grp_شروع, &kparams->div_ne12);
        ثابت uint32_t b2 = b_grp_شروع - b3 * ne12;
        ثابت uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ثابت uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_سرا = b_grp_شروع + 1;
        طالما (b_grp_سرا < b_سرا) {
            ثابت uint32_t cur_b3 = fastdiv(b_grp_سرا, &kparams->div_ne12);
            ثابت uint32_t cur_b2 = b_grp_سرا - cur_b3 * ne12;
            ثابت uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ثابت uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            إذا (cur_i02 != i02 || cur_i03 != i03) {
                اكسر;
            }
            b_grp_سرا++;
        }

        ثابت uint32_t ٹکڑا_idx = i03 * ne02 + i02;
        ثابت dma_addr_t src0_ٹکڑا = src0->انکڑے + (size_t) ٹکڑا_idx * src0_ٹکڑا_stride;

        uint32_t ڈالو_ct = ct_شروع;
        لكل (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_سرا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + d * tile_صف_transfer_ناپ_aligned,
                           src0_ٹکڑا + (size_t) ڈالو_ct * tile_صف_stride),
                           aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_شروع; ct < ct_سرا; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_قطار_نکالو(dma_q).dst;

            صحيح valid_صفاں = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_صفاں = MIN(32, MAX(0, valid_صفاں));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            لكل (uint32_t b = b_grp_شروع; b < b_grp_سرا; b++) {
                ثابت uint32_t b_m_شروع = b * ne11;
                ثابت uint32_t m_پہلا = MAX(cur_m_شروع, b_m_شروع);
                ثابت uint32_t m_آخری  = MIN(cur_m_شروع + cur_m_صفاں, b_m_شروع + ne11);
                إذا (m_پہلا >= m_آخری) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_ادھار = (uint8_t *) dst->انکڑے + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_وتھ = m_پہلا - cur_m_شروع;
                ثابت uint32_t dst_m_وتھ   = m_پہلا - b_m_شروع;
                ثابت uint32_t batch_nrows    = m_آخری - m_پہلا;

                uint32_t ir1 = 0;
                لكل (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_انکڑے + (chunk_m_وتھ + ir1 + 0) * act_stride);
                    ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_انکڑے + (chunk_m_وتھ + ir1 + 1) * act_stride);
                    عائم * محدود dst_row0 = (عائم *) (dst_batch_ادھار + (dst_m_وتھ + ir1 + 0) * dst_صف_ناپ);
                    عائم * محدود dst_row1 = (عائم *) (dst_batch_ادھار + (dst_m_وتھ + ir1 + 1) * dst_صف_ناپ);
                    عائم * dst_ptr0 = &dst_row0[ct * 32];
                    عائم * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_صفاں, عدم, عدم);
                }
                لكل (; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) (act_انکڑے + (chunk_m_وتھ + ir1) * act_stride);
                    عائم * محدود dst_صف = (عائم *) (dst_batch_ادھار + (dst_m_وتھ + ir1) * dst_صف_ناپ);
                    عائم * dst_ptr = &dst_صف[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_خانہ, valid_صفاں, عدم);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (ڈالو_ct < ct_سرا) {
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(w_tile, src0_ٹکڑا + (size_t) ڈالو_ct * tile_صف_stride),
                               aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
                ڈالو_ct++;
            }
        }
        b_grp_شروع = b_grp_سرا;
    }
    إذا (src2) {
        hvx_tensor_جوڑو_f32_grid(dst, src2, cur_m_شروع, cur_m_شروع + cur_m_صفاں, src0_شروع_صف, src0_سرا_صف, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ساكن فراغ hvx_mm_2d_flat(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_matmul_preamble;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_نقاب = n_prefetch - 1;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_سرا - mmctx->src0_صف_شروع;  // src0 rows
    ثابت uint32_t act_nrows = mmctx->cur_m_صفاں ? mmctx->cur_m_صفاں : mmctx->act_nrows;
    ثابت uint32_t cur_m_شروع = mmctx->cur_m_شروع;

    ثابت uint32_t src0_شروع_صف  = mmctx->src0_صف_شروع + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_سرا_صف    = MIN(src0_شروع_صف + src0_nrows_فی_thread, mmctx->src0_صف_سرا);
    ثابت uint32_t src0_سرا_صف_x2 = src0_شروع_صف + ((src0_سرا_صف - src0_شروع_صف) & ~1U);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_صف_ناپ  = nb1;
    ثابت size_t src0_صف_ناپ = nb01;
    ثابت size_t act_صف_ناپ = nb11;

    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_ناپ_فی_thread  * ith;
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * محدود act_انکڑے      = mmctx->vtcm_act;

    ثابت dma_addr_t src0_صف = src0->انکڑے;


    // Prefill vtcm with src0 rows
    إذا (src0_شروع_صف < src0_سرا_صف) {
        لكل (uint32_t ir0 = src0_شروع_صف; ir0 < src0_سرا_صف_x2; ir0 += 2) {
            ثابت صحيح is0 = (ir0 - src0_شروع_صف);
            إذا (is0 >= (صحيح)n_prefetch) {
                اكسر;
            }
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + is0 * src0_stride, src0_صف + ir0 * src0_صف_ناپ),
                           src0_stride, src0_صف_ناپ, src0_صف_ناپ, 2);
        }
    }

    إذا (src0_شروع_صف >= src0_سرا_صف) {
        أرجع;
    }

    // Process src0 rows
    لكل (uint32_t ir0 = src0_شروع_صف; ir0 < src0_سرا_صف_x2; ir0 += 2) {
        ثابت uint8_t * ss0 = (فراغ *) dma_قطار_نکالو(dma_q).dst;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
            ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_انکڑے + (ir1+0) * act_stride);
            ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_انکڑے + (ir1+1) * act_stride);
            عائم * محدود dst_row0 = (عائم *) (dst->انکڑے + ((cur_m_شروع + ir1+0) * dst_صف_ناپ));
            عائم * محدود dst_row1 = (عائم *) (dst->انکڑے + ((cur_m_شروع + ir1+1) * dst_صف_ناپ));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        لكل (; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) (act_انکڑے + ir1 * act_stride);
            عائم * محدود dst_صف          = (عائم *) (dst->انکڑے + ((cur_m_شروع + ir1) * dst_صف_ناپ));
            mmctx->vec_dot_2x1(ne00, &dst_صف[ir0], ss0, ss0 + src0_stride, act_خانہ);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ثابت صحيح pr0 = (ir0 + n_prefetch);
        ثابت صحيح is0 = (pr0 - src0_شروع_صف) & prefetch_نقاب;
        إذا (pr0 < src0_سرا_صف_x2) {
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + is0 * src0_stride, src0_صف + pr0 * src0_صف_ناپ),
                           src0_stride, src0_صف_ناپ, src0_صف_ناپ, 2);
        }
    }

    // Process the last row (if any)
    إذا (src0_سرا_صف != src0_سرا_صف_x2) {
        uint32_t  ir0 = src0_سرا_صف_x2;
        ثابت صحيح is0 = (ir0 - src0_شروع_صف) & prefetch_نقاب;
        dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + is0 * src0_stride, src0_صف + ir0 * src0_صف_ناپ),
                       src0_stride, src0_صف_ناپ, src0_صف_ناپ, 1);
        ثابت uint8_t * ss0 = (فراغ *) dma_قطار_نکالو(dma_q).dst;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #ہدایت unroll(2)
        لكل (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) (act_انکڑے + ir1 * act_stride);
            عائم * محدود dst_صف          = (عائم *) (dst->انکڑے + ((cur_m_شروع + ir1) * dst_صف_ناپ));
            mmctx->vec_dot_1x1(ne00, &dst_صف[ir0], ss0, act_خانہ);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    إذا (src2) {
        hvx_tensor_جوڑو_f32_grid(dst, src2, cur_m_شروع, cur_m_شروع + act_nrows, src0_شروع_صف, src0_سرا_صف, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ساكن فراغ hvx_mv_2d_flat(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_سرا - mmctx->src0_صف_شروع;

    ثابت uint32_t src0_شروع_صف  = mmctx->src0_صف_شروع + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_سرا_صف    = MIN(src0_شروع_صف + src0_nrows_فی_thread, mmctx->src0_صف_سرا);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_صف_ناپ  = nb1;
    ثابت size_t src0_صف_ناپ = nb01;
    ثابت size_t act_صف_ناپ  = nb11;

    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_ناپ_فی_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * act_انکڑے      = mmctx->vtcm_act;

    عائم * tmp = (عائم *) vtcm_dst_ptr;

    ثابت dma_addr_t src0_صف = src0->انکڑے;
    ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) act_انکڑے;
    عائم * محدود dst_خانہ         = (عائم *) dst->انکڑے;

    ثابت uint32_t src0_سرا_صف_x2 = src0_شروع_صف + ((src0_سرا_صف - src0_شروع_صف) & ~1U);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_نقاب = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    إذا (src0_شروع_صف < src0_سرا_صف) {
        إذا (src2) {
            عائم * vtcm_bias_ptr = (عائم *) mmctx->vtcm_bias + src0_شروع_صف;
            ثابت dma_addr_t src2_addr = src2->انکڑے + src0_شروع_صف * حجم(عائم);
            صحيح ٹکڑا_ناپ = (صحيح)src0_سرا_صف - (صحيح)src0_شروع_صف;
            إذا (ٹکڑا_ناپ > 0) {
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_bias_ptr, src2_addr),
                               ٹکڑا_ناپ * حجم(عائم), ٹکڑا_ناپ * حجم(عائم), ٹکڑا_ناپ * حجم(عائم), 1);
                dma_قطار_نکالو_nowait(dma_q);
            }
        }
        لكل (uint32_t ir0 = src0_شروع_صف; ir0 < src0_سرا_صف_x2; ir0 += 2) {
            ثابت uint32_t is0 = (ir0 - src0_شروع_صف);
            إذا (is0 >= n_prefetch) {
                اكسر;
            }
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + is0 * src0_stride, src0_صف + ir0 * src0_صف_ناپ),
                           src0_stride, src0_صف_ناپ, src0_صف_ناپ, 2);
        }
    }


    إذا (src0_شروع_صف >= src0_سرا_صف) {
        أرجع;
    }

    // Process src0 rows
    لكل (uint32_t ir0 = src0_شروع_صف; ir0 < src0_سرا_صف_x2; ir0 += 2) {
        ثابت uint8_t * ss0 = (فراغ *) dma_قطار_نکالو(dma_q).dst;
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_شروع_صف], ss0, ss0 + src0_stride, act_خانہ);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ثابت uint32_t pr0 = (ir0 + n_prefetch);
        ثابت uint32_t is0 = (pr0 - src0_شروع_صف) & prefetch_نقاب;
        إذا (pr0 < src0_سرا_صف_x2) {
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + is0 * src0_stride, src0_صف + pr0 * src0_صف_ناپ),
                           src0_stride, src0_صف_ناپ, src0_صف_ناپ, 2);
        }
    }

    // Process the last row (if any)
    إذا (src0_سرا_صف != src0_سرا_صف_x2) {
        ثابت uint32_t ir0 = src0_سرا_صف_x2;
        ثابت uint32_t is0 = (ir0 - src0_شروع_صف) & prefetch_نقاب;
        dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + is0 * src0_stride, src0_صف + ir0 * src0_صف_ناپ),
                       src0_stride, src0_صف_ناپ, src0_صف_ناپ, 1);
        ثابت uint8_t * ss0 = (فراغ *) dma_قطار_نکالو(dma_q).dst;
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_شروع_صف], ss0, act_خانہ);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    صحيح copy_cnt = src0_سرا_صف - src0_شروع_صف;
    إذا (copy_cnt > 0) {
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, src0_سرا_صف);
        إذا (src2) {
            hvx_جوڑو_f32_uuu((uint8_t *) &dst_خانہ[src0_شروع_صف],
                            (ثابت uint8_t *) tmp,
                            (ثابت uint8_t *) ((ثابت عائم *) mmctx->vtcm_bias + src0_شروع_صف),
                            copy_cnt);
        } وإلا {
            hvx_copy_f32_ua((uint8_t *) &dst_خانہ[src0_شروع_صف], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_سرا_صف);
    }
}

ساكن فراغ hvx_mm_4d_flat(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_matmul_preamble;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_نقاب = n_prefetch - 1;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_سرا - mmctx->src0_صف_شروع;
    ثابت uint32_t cur_m_صفاں = mmctx->cur_m_صفاں ? mmctx->cur_m_صفاں : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_شروع = mmctx->cur_m_شروع;

    ثابت uint32_t src0_شروع_صف  = mmctx->src0_صف_شروع + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_سرا_صف    = MIN(src0_شروع_صف + src0_nrows_فی_thread, mmctx->src0_صف_سرا);
    ثابت uint32_t src0_سرا_صف_x2 = src0_شروع_صف + ((src0_سرا_صف - src0_شروع_صف) & ~1U);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_صف_ناپ  = nb1;
    ثابت size_t src0_صف_ناپ = nb01;
    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * محدود act_انکڑے      = mmctx->vtcm_act;


    إذا (src0_شروع_صف >= src0_سرا_صف || cur_m_صفاں == 0) {
        أرجع;
    }

    ثابت uint32_t کل_batches = ne12 * ne13;
    ثابت uint32_t b_شروع = fastdiv(cur_m_شروع, &kparams->div_ne1);
    uint32_t b_سرا = fastdiv(cur_m_شروع + cur_m_صفاں + ne11 - 1, &kparams->div_ne1);
    b_سرا = MIN(b_سرا, کل_batches);

    uint32_t b_grp_شروع = b_شروع;
    طالما (b_grp_شروع < b_سرا) {
        ثابت uint32_t b3 = fastdiv(b_grp_شروع, &kparams->div_ne12);
        ثابت uint32_t b2 = b_grp_شروع - b3 * ne12;
        ثابت uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ثابت uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_سرا = b_grp_شروع + 1;
        طالما (b_grp_سرا < b_سرا) {
            ثابت uint32_t cur_b3 = fastdiv(b_grp_سرا, &kparams->div_ne12);
            ثابت uint32_t cur_b2 = b_grp_سرا - cur_b3 * ne12;
            ثابت uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ثابت uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            إذا (cur_i02 != i02 || cur_i03 != i03) {
                اكسر;
            }
            b_grp_سرا++;
        }

        ثابت dma_addr_t src0_صف = src0->انکڑے + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        لكل (uint32_t ir0 = src0_شروع_صف; ir0 < src0_سرا_صف_x2; ir0 += 2) {
            ثابت صحيح is0 = (ir0 - src0_شروع_صف);
            إذا (is0 >= (صحيح)n_prefetch) {
                اكسر;
            }
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + is0 * src0_stride, src0_صف + (size_t) ir0 * src0_صف_ناپ),
                           src0_stride, src0_صف_ناپ, src0_صف_ناپ, 2);
        }

        لكل (uint32_t ir0 = src0_شروع_صف; ir0 < src0_سرا_صف_x2; ir0 += 2) {
            ثابت uint8_t * ss0 = (فراغ *) dma_قطار_نکالو(dma_q).dst;

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            لكل (uint32_t b = b_grp_شروع; b < b_grp_سرا; b++) {
                ثابت uint32_t b_m_شروع = b * ne11;
                ثابت uint32_t m_پہلا = MAX(cur_m_شروع, b_m_شروع);
                ثابت uint32_t m_آخری  = MIN(cur_m_شروع + cur_m_صفاں, b_m_شروع + ne11);
                إذا (m_پہلا >= m_آخری) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_ادھار = (uint8_t *) dst->انکڑے + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_وتھ = m_پہلا - cur_m_شروع;
                ثابت uint32_t dst_m_وتھ   = m_پہلا - b_m_شروع;
                ثابت uint32_t batch_nrows    = m_آخری - m_پہلا;

                uint32_t ir1 = 0;
                لكل (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_انکڑے + (chunk_m_وتھ + ir1 + 0) * act_stride);
                    ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_انکڑے + (chunk_m_وتھ + ir1 + 1) * act_stride);
                    عائم * محدود dst_row0 = (عائم *) (dst_batch_ادھار + (dst_m_وتھ + ir1 + 0) * dst_صف_ناپ);
                    عائم * محدود dst_row1 = (عائم *) (dst_batch_ادھار + (dst_m_وتھ + ir1 + 1) * dst_صف_ناپ);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                لكل (; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) (act_انکڑے + (chunk_m_وتھ + ir1) * act_stride);
                    عائم * محدود dst_صف          = (عائم *) (dst_batch_ادھار + (dst_m_وتھ + ir1) * dst_صف_ناپ);
                    mmctx->vec_dot_2x1(ne00, &dst_صف[ir0], ss0, ss0 + src0_stride, act_خانہ);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ثابت صحيح pr0 = (ir0 + n_prefetch);
            ثابت صحيح is0 = (pr0 - src0_شروع_صف) & prefetch_نقاب;
            إذا (pr0 < src0_سرا_صف_x2) {
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + is0 * src0_stride, src0_صف + (size_t) pr0 * src0_صف_ناپ),
                               src0_stride, src0_صف_ناپ, src0_صف_ناپ, 2);
            }
        }

        إذا (src0_سرا_صف != src0_سرا_صف_x2) {
            uint32_t ir0 = src0_سرا_صف_x2;
            ثابت صحيح is0 = (ir0 - src0_شروع_صف) & prefetch_نقاب;
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + is0 * src0_stride, src0_صف + (size_t) ir0 * src0_صف_ناپ),
                           src0_stride, src0_صف_ناپ, src0_صف_ناپ, 1);
            ثابت uint8_t * ss0 = (فراغ *) dma_قطار_نکالو(dma_q).dst;

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            لكل (uint32_t b = b_grp_شروع; b < b_grp_سرا; b++) {
                ثابت uint32_t b_m_شروع = b * ne11;
                ثابت uint32_t m_پہلا = MAX(cur_m_شروع, b_m_شروع);
                ثابت uint32_t m_آخری  = MIN(cur_m_شروع + cur_m_صفاں, b_m_شروع + ne11);
                إذا (m_پہلا >= m_آخری) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_ادھار = (uint8_t *) dst->انکڑے + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_وتھ = m_پہلا - cur_m_شروع;
                ثابت uint32_t dst_m_وتھ   = m_پہلا - b_m_شروع;
                ثابت uint32_t batch_nrows    = m_آخری - m_پہلا;

                لكل (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) (act_انکڑے + (chunk_m_وتھ + ir1) * act_stride);
                    عائم * محدود dst_صف          = (عائم *) (dst_batch_ادھار + (dst_m_وتھ + ir1) * dst_صف_ناپ);
                    mmctx->vec_dot_1x1(ne00, &dst_صف[ir0], ss0, act_خانہ);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_شروع = b_grp_سرا;
    }

    إذا (src2) {
        hvx_tensor_جوڑو_f32_grid(dst, src2, cur_m_شروع, cur_m_شروع + cur_m_صفاں, src0_شروع_صف, src0_سرا_صف, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#تعریف_کرو MMID_میٹرکس_صف(صف_id, i1) میٹرکس_صفاں[(صف_id) * mmctx->mapping_stride + (i1)]

ساكن فراغ hvx_mm_id_tiled(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_matmul_preamble;

    ثابت بنية htp_tensor * محدود ids = octx->src[2];

    ثابت uint32_t src0_nrows      = mmctx->src0_صف_سرا - mmctx->src0_صف_شروع;  // src0 rows per expert
    ثابت uint32_t act_nrows       = ne11;
    ثابت uint32_t src0_شروع_صف  = mmctx->src0_صف_شروع + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_سرا_صف    = MIN(src0_شروع_صف + src0_nrows_فی_thread, mmctx->src0_صف_سرا);


    إذا (src0_شروع_صف >= src0_سرا_صف) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_ids = ids->ne[0];  // n_expert_used
    ثابت uint32_t n_as  = ne02;        // n_expert

    ثابت uint32_t *                میٹرکس_صف_counts = mmctx->میٹرکس_صف_counts;
    ثابت بنية mmid_صف_mapping * میٹرکس_صفاں       = mmctx->میٹرکس_صفاں;

    ثابت size_t dst_صف_ناپ  = nb1;
    ثابت size_t act_صف_ناپ  = htp_mm_q8_0_tiled_صف_ناپ(ne10);

    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * محدود act_انکڑے = mmctx->vtcm_act;

    لكل (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int32_t cne1 = میٹرکس_صف_counts[cur_a];
        إذا (cne1 == 0) {
            استمر;
        }

        ثابت dma_addr_t src0_صف = src0->انکڑے + cur_a * nb02;

        ثابت uint32_t tile_ناپ = htp_tensor_get_tile_ناپ(src0->type);
        ثابت uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(src0->type);
        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_ناپ;
        ثابت uint32_t tile_صف_transfer_ناپ_aligned = n_k_tiles_w * aligned_tile_ناپ;

        ثابت uint32_t ct_شروع = src0_شروع_صف / 32;
        ثابت uint32_t ct_سرا   = (src0_سرا_صف + 31) / 32;

        uint32_t ڈالو_ct = ct_شروع;
        لكل (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_سرا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + d * tile_صف_transfer_ناپ_aligned, src0_صف + ڈالو_ct * tile_صف_stride),
                           aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_شروع; ct < ct_سرا; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_قطار_نکالو(dma_q).dst;

            صحيح valid_صفاں = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_صفاں = MIN(32, MAX(0, valid_صفاں));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            لكل (uint32_t cid = 0; cid < cne1; ++cid) {
                بنية mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, cid);
                ثابت صحيح               rm1         = صف_mapping.i1;  // expert idx
                ثابت صحيح               rm2         = صف_mapping.i2;  // token idx

                ثابت uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) (act_انکڑے + (ir1 + rm2 * ne11 + 0) * act_stride);
                عائم * محدود dst_صف = (عائم *) (dst->انکڑے + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_صف[ct * 32], w_tile, act_خانہ, valid_صفاں, عدم);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (ڈالو_ct < ct_سرا) {
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                               aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
                ڈالو_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mv_id_tiled(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_matmul_preamble;

    ثابت بنية htp_tensor * محدود ids = octx->src[2];

    ثابت uint32_t src0_nrows      = mmctx->src0_صف_سرا - mmctx->src0_صف_شروع;  // src0 rows per expert
    ثابت uint32_t src0_شروع_صف  = mmctx->src0_صف_شروع + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_سرا_صف    = MIN(src0_شروع_صف + src0_nrows_فی_thread, mmctx->src0_صف_سرا);


    إذا (src0_شروع_صف >= src0_سرا_صف) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    أكد(ne13 % ne03 == 0);

    ثابت size_t dst_صف_ناپ  = nb1;
    ثابت size_t act_صف_ناپ = htp_mm_q8_0_tiled_صف_ناپ(ne10);

    ثابت uint32_t n_aids = src2->ne[0];  // num activated experts
    ثابت uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * محدود act_انکڑے = mmctx->vtcm_act;

    لكل (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ثابت int32_t eid = *(ثابت int32_t *) ((ثابت uint8_t *) src2->انکڑے + ie1 * src2->nb[0]);
        إذا (eid < 0) {
            استمر;
        }
        أكد(eid < (int32_t) n_ids);

        ثابت dma_addr_t src0_صف = src0->انکڑے + eid * nb02;
        ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) act_انکڑے;
        عائم * محدود dst_صف          = (عائم *) (dst->انکڑے + ie1 * nb1);

        ثابت uint32_t tile_ناپ = htp_tensor_get_tile_ناپ(src0->type);
        ثابت uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(src0->type);
        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_ناپ;
        ثابت uint32_t tile_صف_transfer_ناپ_aligned = n_k_tiles_w * aligned_tile_ناپ;

        ثابت uint32_t ct_شروع = src0_شروع_صف / 32;
        ثابت uint32_t ct_سرا   = (src0_سرا_صف + 31) / 32;

        uint32_t ڈالو_ct = ct_شروع;
        لكل (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_سرا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + d * tile_صف_transfer_ناپ_aligned, src0_صف + ڈالو_ct * tile_صف_stride),
                           aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_شروع; ct < ct_سرا; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_قطار_نکالو(dma_q).dst;

            صحيح valid_صفاں = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_صفاں = MIN(32, MAX(0, valid_صفاں));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_صف[ct * 32], w_tile, act_خانہ, valid_صفاں, عدم);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (ڈالو_ct < ct_سرا) {
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                               aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
                ڈالو_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mv_id_nx_tiled(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية htp_mm_context * mmctx = (بنية htp_mm_context *) انکڑے;
    بنية htp_ops_context * octx = mmctx->octx;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights      = kparams->n_weights;
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];
    ثابت بنية htp_tensor * محدود ids  = octx->src[n_weights + 1];


    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_aids = ids->ne[0];
    ثابت uint32_t n_ids  = src0->ne[2];

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * محدود act_انکڑے = mmctx->vtcm_act;

    لكل (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ثابت int32_t eid = *(ثابت int32_t *) ((ثابت uint8_t *) ids->انکڑے + ie1 * ids->nb[0]);
        إذا (eid < 0) استمر;
        أكد(eid < (int32_t) n_ids);

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * محدود src_w = octx->src[p];
            ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
            إذا (!src_w || !dst) استمر;
            dma_قطار * dma_q = octx->ctx->dma[ith];

            ثابت uint32_t ne01 = src_w->ne[1];
            uint32_t شروع_صف = 0;
            uint32_t سرا_صف   = ne01;
            إذا (octx->ctx->mdev.گنتی > 1) {
                ثابت bool can_ونڈو = htp_tensor_can_صف_حصے_کرو(dst, حجم(عائم));
                ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(ne01, can_ونڈو ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
                شروع_صف = range.شروع;
                سرا_صف   = range.شروع + range.گنتی;
            }

            ثابت uint32_t nrows = سرا_صف - شروع_صف;
            uint32_t src0_nrows_فی_thread = fastdiv(nrows + nth - 1, &octx->n_دھاگے_div);
            src0_nrows_فی_thread = hex_گیڑا_اتے(src0_nrows_فی_thread, 32);

            ثابت uint32_t src0_شروع_صف = شروع_صف + src0_nrows_فی_thread * ith;
            ثابت uint32_t src0_سرا_صف   = MIN(src0_شروع_صف + src0_nrows_فی_thread, سرا_صف);
            إذا (src0_شروع_صف >= src0_سرا_صف) استمر;

            ثابت dma_addr_t src0_صف = src_w->انکڑے + eid * src_w->nb[2];
            ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) act_انکڑے;
            عائم * محدود dst_صف = (عائم *) (dst->انکڑے + ie1 * dst->nb[1]);

            ثابت uint32_t tile_ناپ = htp_tensor_get_tile_ناپ(src_w->type);
            ثابت uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(src_w->type);
            ثابت uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_ناپ;
            ثابت uint32_t tile_صف_transfer_ناپ_aligned = n_k_tiles_w * aligned_tile_ناپ;

            ثابت uint32_t ct_شروع = src0_شروع_صف / 32;
            ثابت uint32_t ct_سرا   = (src0_سرا_صف + 31) / 32;

            uint32_t ڈالو_ct = ct_شروع;
            لكل (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_سرا; d++, ڈالو_ct++) {
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + d * tile_صف_transfer_ناپ_aligned, src0_صف + ڈالو_ct * tile_صف_stride),
                               aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
            }

            لكل (uint32_t ct = ct_شروع; ct < ct_سرا; ct++) {
                ثابت uint8_t * w_tile = (فراغ *) dma_قطار_نکالو(dma_q).dst;

                صحيح valid_صفاں = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
                valid_صفاں = MIN(32, MAX(0, valid_صفاں));

                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_صف[ct * 32], w_tile, act_خانہ, valid_صفاں, عدم);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                إذا (ڈالو_ct < ct_سرا) {
                    dma_قطار_ڈالو(dma_q, dma_make_انکڑے(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                                    aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
                    ڈالو_ct++;
                }
            }
        }
    }
}

ساكن فراغ hvx_mm_id_nx_tiled(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية htp_mm_context * mmctx = (بنية htp_mm_context *) انکڑے;
    بنية htp_ops_context * octx = mmctx->octx;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights      = kparams->n_weights;
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];
    ثابت بنية htp_tensor * محدود ids  = octx->src[n_weights + 1];


    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_as = src0->ne[2];

    ثابت uint32_t * میٹرکس_صف_counts = mmctx->میٹرکس_صف_counts;
    ثابت بنية mmid_صف_mapping * میٹرکس_صفاں = mmctx->میٹرکس_صفاں;

    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_ناپ_فی_thread * ith;
    uint8_t * محدود act_انکڑے = mmctx->vtcm_act;

    لكل (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int32_t cne1 = میٹرکس_صف_counts[cur_a];
        إذا (cne1 == 0) استمر;

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * محدود src_w = octx->src[p];
            ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
            إذا (!src_w || !dst) استمر;
            dma_قطار * dma_q = octx->ctx->dma[ith];

            ثابت uint32_t ne01 = src_w->ne[1];
            uint32_t شروع_صف = 0;
            uint32_t سرا_صف   = ne01;
            إذا (octx->ctx->mdev.گنتی > 1) {
                ثابت bool can_ونڈو = htp_tensor_can_صف_حصے_کرو(dst, حجم(عائم));
                ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(ne01, can_ونڈو ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
                شروع_صف = range.شروع;
                سرا_صف   = range.شروع + range.گنتی;
            }

            ثابت uint32_t nrows = سرا_صف - شروع_صف;
            uint32_t src0_nrows_فی_thread = fastdiv(nrows + nth - 1, &octx->n_دھاگے_div);
            src0_nrows_فی_thread = hex_گیڑا_اتے(src0_nrows_فی_thread, 32);

            ثابت uint32_t src0_شروع_صف = شروع_صف + src0_nrows_فی_thread * ith;
            ثابت uint32_t src0_سرا_صف   = MIN(src0_شروع_صف + src0_nrows_فی_thread, سرا_صف);
            إذا (src0_شروع_صف >= src0_سرا_صف) استمر;

            ثابت dma_addr_t src0_صف = src_w->انکڑے + cur_a * src_w->nb[2];

            ثابت uint32_t tile_ناپ = htp_tensor_get_tile_ناپ(src_w->type);
            ثابت uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(src_w->type);
            ثابت uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_ناپ;
            ثابت uint32_t tile_صف_transfer_ناپ_aligned = n_k_tiles_w * aligned_tile_ناپ;

            ثابت uint32_t ct_شروع = src0_شروع_صف / 32;
            ثابت uint32_t ct_سرا   = (src0_سرا_صف + 31) / 32;

            uint32_t ڈالو_ct = ct_شروع;
            لكل (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_سرا; d++, ڈالو_ct++) {
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_src0_ptr + d * tile_صف_transfer_ناپ_aligned, src0_صف + ڈالو_ct * tile_صف_stride),
                               aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
            }

            لكل (uint32_t ct = ct_شروع; ct < ct_سرا; ct++) {
                ثابت uint8_t * w_tile = (فراغ *) dma_قطار_نکالو(dma_q).dst;

                صحيح valid_صفاں = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
                valid_صفاں = MIN(32, MAX(0, valid_صفاں));

                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                لكل (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    بنية mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, cid);
                    ثابت صحيح rm1 = صف_mapping.i1;
                    ثابت صحيح rm2 = صف_mapping.i2;

                    ثابت uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ثابت uint8_t * محدود act_خانہ = (ثابت uint8_t *) (act_انکڑے + (ir1 + rm2 * act->ne[1]) * act_stride);
                    عائم * محدود dst_صف = (عائم *) (dst->انکڑے + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_صف[ct * 32], w_tile, act_خانہ, valid_صفاں, عدم);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                إذا (ڈالو_ct < ct_سرا) {
                    dma_قطار_ڈالو(dma_q, dma_make_انکڑے(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                                    aligned_tile_ناپ, tile_ناپ, tile_ناپ, n_k_tiles_w);
                    ڈالو_ct++;
                }
            }
        }
    }
}

ساكن صحيح hvx_mm_init_tiled_vec_dot(بنية htp_mm_context * mmctx, تعداد htp_انکڑے_type type) {
    بدل (type) {
        حالة HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            أرجع 0;
        حالة HTP_TYPE_Q4_1:
        حالة HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            أرجع 0;
        حالة HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            أرجع 0;
        حالة HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            أرجع 0;
        حالة HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            أرجع 0;
        حالة HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            أرجع 0;
        حالة HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            أرجع 0;
        حالة HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            أرجع 0;
        حالة HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            أرجع 0;
        حالة HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            أرجع 0;
        حالة HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            أرجع 0;
        حالة HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            أرجع 0;
        افتراضي:
            أرجع -1;
    }
}

ساكن صحيح hvx_mm_matmul(بنية htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    ثابت uint32_t src0_nrows = ne01;
    ثابت uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_صف_شروع = 0;
    uint32_t src0_صف_سرا   = src0_nrows;

    إذا (octx->ctx->mdev.گنتی > 1) {
        ثابت bool can_ونڈو = htp_tensor_can_صف_حصے_کرو(dst, حجم(عائم));
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(src0_nrows, can_ونڈو ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        src0_صف_شروع = range.شروع;
        src0_صف_سرا   = range.شروع + range.گنتی;
    }

    إذا (src0_صف_شروع >= src0_صف_سرا) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t nrows = src0_صف_سرا - src0_صف_شروع;
    mmctx->src0_صف_شروع = src0_صف_شروع;
    mmctx->src0_صف_سرا   = src0_صف_سرا;

    ثابت bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_فی_thread  = fastdiv(nrows + octx->n_دھاگے - 1, &octx->n_دھاگے_div);
    إذا (is_tiled) {
        mmctx->src0_nrows_فی_thread = hex_گیڑا_اتے(mmctx->src0_nrows_فی_thread, 32);
    } وإلا {
        mmctx->src0_nrows_فی_thread += (mmctx->src0_nrows_فی_thread & 1); // round up to even
    }

    ثابت size_t src0_صف_ناپ = nb01;
    ثابت size_t dst_صف_ناپ  = nb1;
    size_t       act_صف_ناپ  = nb11;

    ثابت size_t src0_صف_ناپ_padded = hex_گیڑا_اتے(src0_صف_ناپ, 128);

    کامہ_callback_t quant_task_func;
    کامہ_callback_t matmul_کم_func;
    uint32_t n_quant_tasks = 1;
    ثابت bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    إذا (is_batched) {
        matmul_کم_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } وإلا إذا (act_nrows > 1) {
        matmul_کم_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } وإلا {
        matmul_کم_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    bool need_quant = true;

    بدل (kparams->kernel_type) {
        حالة HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            act_صف_ناپ           = kparams->act_صف_ناپ;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = عدم;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_صف_ناپ           = kparams->act_صف_ناپ;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = عدم;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_صف_ناپ           = kparams->act_صف_ناپ;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_QUANT_بلاک:
        حالة HTP_MM_KERNEL_HVX_QUANT_صف:
        افتراضي: {
            إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                أرجع HTP_حالت_NO_SUPPORT;
            }

            ثابت uint32_t qk = QK_Q8_0_TILED;
            ثابت uint32_t nb = (ne10 + qk - 1) / qk;
            ثابت uint32_t کل_nb = act_nrows * nb;

            إذا (act_nrows < octx->n_دھاگے && !is_batched) {
                n_quant_tasks = MIN(کل_nb, octx->n_دھاگے);
                quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
                لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_پہلا = (کل_nb * ith) / n_quant_tasks;
                    uint32_t ib_آخری  = (کل_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_پہلا[ith] = ib_پہلا;
                    mmctx->quant_ib_آخری[ith]  = ib_آخری;
                    mmctx->quant_r[ith]        = ib_پہلا / nb;
                    mmctx->quant_c[ith]        = ib_پہلا % nb;
                }
            } وإلا {
                n_quant_tasks = MIN(act_nrows, octx->n_دھاگے);
                quant_task_func = htp_mm_act_quant_صف_func(src0->type, act->type);
            }
            act_صف_ناپ = kparams->act_صف_ناپ;
            اكسر;
        }
    }

    ثابت uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ثابت uint32_t m_layout_صفاں = m_chunk;

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_صفاں, octx->n_دھاگے,
                                 dst_صف_ناپ, src0_صف_ناپ, act_صف_ناپ, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    إذا (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
        mmctx->vtcm_act_ناپ_فی_thread = L.act_بائٹاں;
    } وإلا {
        mmctx->vtcm_act_ناپ_فی_thread = fastdiv(L.act_بائٹاں, &octx->n_دھاگے_div);
    }

    mmctx->vtcm_src0_ناپ_فی_thread = fastdiv(L.src0_بائٹاں, &octx->n_دھاگے_div);
    mmctx->vtcm_dst_ناپ_فی_thread  = fastdiv(L.dst_بائٹاں, &octx->n_دھاگے_div);

    ثابت size_t vtcm_ناپ = L.کل_بائٹاں;

    FARF(بلند_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_بائٹاں, L.act_بائٹاں, L.dst_بائٹاں, vtcm_ناپ);

    FARF(بلند_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->انکڑے, act->انکڑے, dst->انکڑے);

    إذا (octx->ctx->vtcm_ناپ < vtcm_ناپ) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_ناپ, vtcm_ناپ);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت ادھار = (uint8_t *) octx->ctx->vtcm_ادھار;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_صف_ناپ_padded;
    mmctx->vtcm_act_stride  = act_صف_ناپ;
    إذا (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف) {
        mmctx->vtcm_act_raw_stride = hex_گیڑا_اتے(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
    } وإلا إذا (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_گیڑا_اتے(ne10 * حجم(عائم), 128);
    } وإلا {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    إذا (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        لكل (uint32_t m_شروع = 0; m_شروع < act_nrows; m_شروع += m_chunk) {
            ثابت uint32_t cur_m_صفاں = MIN(act_nrows - m_شروع, m_chunk);
            mmctx->cur_m_شروع = m_شروع;
            mmctx->cur_m_صفاں  = cur_m_صفاں;

            إذا (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_شروع, cur_m_صفاں);

                ثابت uint32_t qk = QK_Q8_0_TILED;
                ثابت uint32_t nb = (ne10 + qk - 1) / qk;
                ثابت uint32_t کل_nb = cur_m_صفاں * nb;
                uint32_t quant_tasks;
                work_قطار_func_t q_func;
                إذا (cur_m_صفاں < octx->n_دھاگے && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف)) {
                    quant_tasks = MIN(کل_nb, octx->n_دھاگے);
                    q_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
                    لكل (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_پہلا = (کل_nb * ith) / quant_tasks;
                        uint32_t ib_آخری  = (کل_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_پہلا[ith] = ib_پہلا;
                        mmctx->quant_ib_آخری[ith]  = ib_آخری;
                        mmctx->quant_r[ith]        = ib_پہلا / nb;
                        mmctx->quant_c[ith]        = ib_پہلا % nb;
                    }
                } وإلا {
                    quant_tasks = MIN(cur_m_صفاں, octx->n_دھاگے);
                    q_func = quant_task_func;
                    mmctx->n_quant_صفاں_فی_thread = (cur_m_صفاں + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_قطار_لگاتار(octx->ctx->work_قطار, q_func, mmctx, quant_tasks);
            } وإلا {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_شروع, cur_m_صفاں);
            }

            work_قطار_لگاتار(octx->ctx->work_قطار, matmul_کم_func, mmctx, octx->n_دھاگے);
        }
    } وإلا {
        mmctx->cur_m_شروع = 0;
        mmctx->cur_m_صفاں  = act_nrows;

        إذا (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_صفاں_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_قطار_لگاتار(octx->ctx->work_قطار, quant_task_func, mmctx, n_quant_tasks);
        } وإلا {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_قطار_لگاتار(octx->ctx->work_قطار, matmul_کم_func, mmctx, octx->n_دھاگے);
    }

    أرجع HTP_حالت_OK;
}

ساكن فراغ dequantize_tiled_کامہ_loop(بنانشانی صحيح n, بنانشانی صحيح i, فراغ * انکڑے) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) انکڑے;
    بنية htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    لكل (بنانشانی صحيح task_id = i; task_id < (بنانشانی صحيح) state->n_tasks; task_id += n) {
        صحيح شروع = task_id * state->n_tiles_فی_task;
        صحيح سرا   = hex_smin(شروع + state->n_tiles_فی_task, state->n_tot_tiles);
        state->task_fn(state, شروع, سرا);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

ساكن فراغ transfer_output_chunk_کامہ_fn(بنانشانی صحيح n, بنانشانی صحيح i, فراغ *انکڑے) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) انکڑے;

    بنية htp_thread_trace * tr = &st->traces[i];

    صحيح شروع_chunk_idx = i * st->n_chunks_فی_task;
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_O_PROC, شروع_chunk_idx);

    لكل (بنانشانی صحيح task_id = i; task_id < (بنانشانی صحيح)st->n_tasks; task_id += n) {
        صحيح    chunk_idx  = task_id * st->n_chunks_فی_task;
        size_t chunk_ناپ = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_فی_task);

        عائم        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ثابت عائم  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : عدم;
        transfer_output_chunk_fp16_منزل_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_ناپ, st->n_خانے_گنتی, st->dst_stride, st->src2_stride, st->dst_خانے_گنتی);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, شروع_chunk_idx);
}

تعريفنوع بنية {
    بنية htp_context            * ctx;
    بنية htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    عائم                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_فی_task;
    uint32_t                        k_بلاک;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_بائٹاں_فی_thread;
    uint32_t                        dma_قدم_صفاں;
    uint32_t                        dma_قدم_صفاں_کھسکاؤ;
    uint32_t                        act_elem_ناپ;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

تعريفنوع بنية {
    بنية htp_context            * ctx;
    بنية htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    عائم                         * vtcm_f32_act;
    uint32_t                        n_صفاں;
    uint32_t                        k_بلاک;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_خانہ_chunks;
    بنية fastdiv_اقدار           n_دھاگے_div;
    size_t                          vtcm_f32_act_بائٹاں;
    uint32_t                        dma_قدم_صفاں;
    uint32_t                        dma_قدم_صفاں_کھسکاؤ;
} activation_transfer_خانہ_chunk_state_t;

ساكن فراغ transfer_activation_chunk_fp32_منزل_fp16_dma_pipelined_خانہ_chunk(
        dma_قطار *dma_q,
        __fp16 *محدود vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_صفاں,
        uint32_t k_بلاک,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_پہلا,
        uint32_t c_len,
        عائم *thread_f32_act,
        بنية htp_thread_trace *tr,
        uint32_t dma_قدم_صفاں,
        uint32_t dma_قدم_صفاں_کھسکاؤ) {

    ثابت uint32_t R = dma_قدم_صفاں;
    ثابت uint32_t n_صفاں_padded = hex_align_اتے(n_صفاں, HTP_MM_HMX_TILE_N_صفاں);

    ثابت uint32_t n_قدماں = n_صفاں_padded >> dma_قدم_صفاں_کھسکاؤ;

    // Push step 0
    إذا (n_قدماں > 0 && n_صفاں > 0) {
        uint32_t nrows_منزل_fetch = hex_smin(n_صفاں, R);
        dma_قطار_ڈالو(dma_q, dma_make_انکڑے(thread_f32_act, act_dma_addr + (size_t) c_پہلا * حجم(عائم)),
                       c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_منزل_fetch);
    }
    // Push step 1
    إذا (n_قدماں > 1) {
        uint32_t اگلا_r = R * 1;
        إذا (اگلا_r < n_صفاں) {
            uint32_t nrows_منزل_fetch = hex_smin(n_صفاں - اگلا_r, R);
            عائم *اگلا_buf = thread_f32_act + 1 * R * c_len;
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(اگلا_buf, act_dma_addr + ((size_t) اگلا_r * k_stride + c_پہلا) * حجم(عائم)),
                           c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_منزل_fetch);
        }
    }
    لكل (uint32_t s = 0; s < n_قدماں; ++s) {
        uint32_t r = s << dma_قدم_صفاں_کھسکاؤ;
        عائم *curr_buf = thread_f32_act;

        إذا (r < n_صفاں) {
            curr_buf = (عائم *) dma_قطار_نکالو(dma_q).dst;
        }

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        لكل (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t صف_idx = r + (p << 1);
            عائم *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((صف_idx + 0) < n_صفاں);
            bool r1_valid = ((صف_idx + 1) < n_صفاں);

            transfer_activation_صف_pair_fp32_منزل_fp16_خانہ_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, صف_idx, k_بلاک, c_پہلا, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t اگلا_s = s + 2;
        uint32_t اگلا_r = اگلا_s << dma_قدم_صفاں_کھسکاؤ;
        إذا (اگلا_r < n_صفاں) {
            uint32_t nrows_منزل_fetch = hex_smin(n_صفاں - اگلا_r, R);
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(curr_buf, act_dma_addr + ((size_t) اگلا_r * k_stride + c_پہلا) * حجم(عائم)),
                           c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_منزل_fetch);
        }
    }
}

ساكن فراغ transfer_activation_chunk_خانہ_chunk_کامہ_fn(بنانشانی صحيح n, بنانشانی صحيح i, فراغ *انکڑے) {
    activation_transfer_خانہ_chunk_state_t *st = (activation_transfer_خانہ_chunk_state_t *) انکڑے;
    بنية htp_thread_trace * tr = &st->traces[i];

    uint32_t n_بلاکاں = st->k_بلاک / 32;
    uint32_t b_پہلا = fastdiv(n_بلاکاں * i, &st->n_دھاگے_div);
    uint32_t b_آخری  = fastdiv(n_بلاکاں * (i + 1), &st->n_دھاگے_div);
    uint32_t c_پہلا = b_پہلا * 32;
    uint32_t c_آخری = b_آخری * 32;
    uint32_t c_len = c_آخری - c_پہلا;

    إذا (c_len == 0) {
        أرجع;
    }

    uint32_t k_chunk_valid = 0;
    إذا (st->k_valid > c_پہلا) {
        k_chunk_valid = hex_smin(st->k_valid, c_آخری) - c_پہلا;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_بائٹاں = hex_align_تھلے(fastdiv(st->vtcm_f32_act_بائٹاں, &st->n_دھاگے_div), 128);
    عائم *thread_f32_act = (عائم *)((حرف *)st->vtcm_f32_act + i * thread_scratch_بائٹاں);

    transfer_activation_chunk_fp32_منزل_fp16_dma_pipelined_خانہ_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_صفاں, st->k_بلاک, st->k_stride, k_chunk_valid,
        c_پہلا, c_len, thread_f32_act, tr, st->dma_قدم_صفاں, st->dma_قدم_صفاں_کھسکاؤ
    );
}

ساكن فراغ transfer_activation_chunk_منزل_fp16_dma_pipelined(
        dma_قطار *dma_q,
        __fp16 *محدود vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_صفاں,
        uint32_t k_بلاک,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_ناپ,
        بنية htp_thread_trace *tr,
        uint32_t dma_قدم_صفاں,
        uint32_t dma_قدم_صفاں_کھسکاؤ) {

    ثابت uint32_t R = dma_قدم_صفاں;
    ثابت uint32_t n_صفاں_padded = hex_align_اتے(n_صفاں, HTP_MM_HMX_TILE_N_صفاں);

    ثابت uint32_t n_قدماں = n_صفاں_padded >> dma_قدم_صفاں_کھسکاؤ;

    ثابت bool act_is_f16 = (act_elem_ناپ == حجم(__fp16));
    ثابت size_t صف_بائٹاں = (size_t) k_بلاک * act_elem_ناپ;  // staging row (DMA dst stride)
    ثابت size_t src_صف_بائٹاں = (size_t) k_stride * act_elem_ناپ;  // DDR row (DMA src stride)
    ثابت size_t چوڑائی_بائٹاں = (size_t) k_valid * act_elem_ناپ;

    // Push step 0
    إذا (n_قدماں > 0 && n_صفاں > 0) {
        uint32_t nrows_منزل_fetch = hex_smin(n_صفاں, R);
        dma_قطار_ڈالو(dma_q, dma_make_انکڑے(thread_act, act_dma_addr),
                       صف_بائٹاں, src_صف_بائٹاں, چوڑائی_بائٹاں, nrows_منزل_fetch);
    }
    // Push step 1 (if valid)
    إذا (n_قدماں > 1) {
        uint32_t اگلا_r = R * 1;
        إذا (اگلا_r < n_صفاں) {
            uint32_t nrows_منزل_fetch = hex_smin(n_صفاں - اگلا_r, R);
            uint8_t *اگلا_buf = thread_act + 1 * R * صف_بائٹاں;
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(اگلا_buf, act_dma_addr + (size_t) اگلا_r * src_صف_بائٹاں),
                           صف_بائٹاں, src_صف_بائٹاں, چوڑائی_بائٹاں, nrows_منزل_fetch);
        }
    }
    لكل (uint32_t s = 0; s < n_قدماں; ++s) {
        uint32_t r = s << dma_قدم_صفاں_کھسکاؤ;
        uint8_t *curr_buf = thread_act;

        إذا (r < n_صفاں) {
            curr_buf = (uint8_t *) dma_قطار_نکالو(dma_q).dst;
        }

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        إذا (act_is_f16) {
            لكل (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t صف_idx = r + (p << 1);
                ثابت __fp16 *pair_buf = (ثابت __fp16 *) (curr_buf + (p << 1) * صف_بائٹاں);
                bool r0_valid = ((صف_idx + 0) < n_صفاں);
                bool r1_valid = ((صف_idx + 1) < n_صفاں);

                transfer_activation_صف_pair_f16_منزل_f16(vtcm_dst, pair_buf, pair_buf + k_بلاک, صف_idx, k_بلاک, k_valid, r0_valid, r1_valid);
            }
        } وإلا {
            لكل (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t صف_idx = r + (p << 1);
                ثابت عائم *pair_buf = (ثابت عائم *) (curr_buf + (p << 1) * صف_بائٹاں);
                bool r0_valid = ((صف_idx + 0) < n_صفاں);
                bool r1_valid = ((صف_idx + 1) < n_صفاں);

                transfer_activation_صف_pair_fp32_منزل_fp16(vtcm_dst, pair_buf, pair_buf + k_بلاک, صف_idx, k_بلاک, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t اگلا_s = s + 2;
        uint32_t اگلا_r = اگلا_s << dma_قدم_صفاں_کھسکاؤ;
        إذا (اگلا_r < n_صفاں) {
            uint32_t nrows_منزل_fetch = hex_smin(n_صفاں - اگلا_r, R);
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(curr_buf, act_dma_addr + (size_t) اگلا_r * src_صف_بائٹاں),
                           صف_بائٹاں, src_صف_بائٹاں, چوڑائی_بائٹاں, nrows_منزل_fetch);
        }
    }
}

ساكن فراغ transfer_activation_chunk_کامہ_fn(بنانشانی صحيح n, بنانشانی صحيح i, فراغ *انکڑے) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) انکڑے;

    بنية htp_thread_trace * tr = &st->traces[i];

    لكل (بنانشانی صحيح task_id = i; task_id < (بنانشانی صحيح)st->n_tasks; task_id += n) {
        صحيح    chunk_idx  = task_id * st->n_chunks_فی_task;
        size_t chunk_ناپ = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_فی_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_بلاک;
        ثابت dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_ناپ;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_بائٹاں_فی_thread;
        transfer_activation_chunk_منزل_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_ناپ, st->k_بلاک, st->k_stride, st->k_valid, thread_act, st->act_elem_ناپ, tr, st->dma_قدم_صفاں, st->dma_قدم_صفاں_کھسکاؤ
        );
    }
}

تعريفنوع بنية {
    بنية htp_thread_trace       * traces;
    ثابت بنية mmid_صف_mapping * میٹرکس_صفاں;
    __fp16                        * dst;
    ثابت عائم                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_فی_task;
    uint32_t                        k_بلاک;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    بنية fastdiv_اقدار           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        شروع_صف;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

تعريفنوع بنية {
    بنية htp_thread_trace       * traces;
    ثابت بنية mmid_صف_mapping * میٹرکس_صفاں;
    ثابت __fp16                  * vtcm_src;
    عائم                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_فی_task;
    uint32_t                        n_خانے_گنتی;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        شروع_صف;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

ساكن فراغ transfer_activation_chunk_gathered_کامہ_fn(بنانشانی صحيح n, بنانشانی صحيح i, فراغ *انکڑے) {
    activation_transfer_gathered_task_state_t *st = انکڑے;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx      = i;
    صحيح chunk_ناپ     = st->n_chunks_فی_task;
    صحيح vtcm_شروع_صف = chunk_idx * chunk_ناپ;
    صحيح شروع_صف      = st->شروع_صف + vtcm_شروع_صف;
    صحيح n_صفاں         = hex_smin(st->cne1 - شروع_صف, chunk_ناپ);
    إذا (n_صفاں > 0) {
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_منزل_fp16_gathered(
            st->dst, st->src, شروع_صف, vtcm_شروع_صف, n_صفاں, st->k_بلاک,
            st->میٹرکس_صفاں, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ساكن فراغ transfer_activation_chunk_gathered_کامہ_flat_fn(بنانشانی صحيح n, بنانشانی صحيح i, فراغ *انکڑے) {
    activation_transfer_gathered_task_state_t *st = انکڑے;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx = i;
    صحيح chunk_ناپ = st->n_chunks_فی_task;
    صحيح vtcm_شروع_صف = chunk_idx * chunk_ناپ;
    صحيح شروع_صف = st->شروع_صف + vtcm_شروع_صف;
    صحيح n_صفاں = hex_smin(st->cne1 - شروع_صف, chunk_ناپ);
    إذا (n_صفاں > 0) {
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_منزل_fp16_gathered_flat(
            st->dst, st->src, شروع_صف, vtcm_شروع_صف, n_صفاں, st->k_بلاک,
            st->میٹرکس_صفاں, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ساكن فراغ transfer_output_chunk_scattered_کامہ_fn(بنانشانی صحيح n, بنانشانی صحيح i, فراغ *انکڑے) {
    output_transfer_scattered_task_state_t *st = انکڑے;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx = i;
    صحيح chunk_ناپ = st->n_chunks_فی_task;
    صحيح vtcm_شروع_صف = chunk_idx * chunk_ناپ;
    صحيح شروع_صف = st->شروع_صف + vtcm_شروع_صف;
    صحيح n_صفاں = hex_smin(st->cne1 - شروع_صف, chunk_ناپ);
    إذا (n_صفاں > 0) {
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_منزل_fp32_scattered(
            st->dst, st->vtcm_src, شروع_صف, vtcm_شروع_صف, n_صفاں, st->n_خانے_گنتی,
            st->میٹرکس_صفاں, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

ساكن مضمن tiled_dequant_task_fn_t hmx_get_dequant_کامہ_fn(صحيح بھار_type, bool is_tiled) {
    بدل (بھار_type) {
        حالة HTP_TYPE_Q4_0:   أرجع dequantize_tiled_بھار_منزل_fp16_task_q4_0;
        حالة HTP_TYPE_IQ4_NL: أرجع dequantize_tiled_بھار_منزل_fp16_task_iq4_nl;
        حالة HTP_TYPE_Q4_1:
        حالة HTP_TYPE_Q4_K:   أرجع dequantize_tiled_بھار_منزل_fp16_task_q4_1;
        حالة HTP_TYPE_MXFP4:  أرجع dequantize_tiled_بھار_منزل_fp16_task_mxfp4;
        حالة HTP_TYPE_Q8_0:   أرجع dequantize_tiled_بھار_منزل_fp16_task_q8_0;
        حالة HTP_TYPE_Q5_K:   أرجع dequantize_tiled_بھار_منزل_fp16_task_q5_k;
        حالة HTP_TYPE_Q6_K:   أرجع dequantize_tiled_بھار_منزل_fp16_task_q6_k;
        حالة HTP_TYPE_Q3_K:   أرجع dequantize_tiled_بھار_منزل_fp16_task_q3_k;
        حالة HTP_TYPE_Q2_K:   أرجع dequantize_tiled_بھار_منزل_fp16_task_q2_k;
        حالة HTP_TYPE_Q1_0:   أرجع dequantize_tiled_بھار_منزل_fp16_task_q1_0;
        حالة HTP_TYPE_Q2_0:   أرجع dequantize_tiled_بھار_منزل_fp16_task_q2_0;
        حالة HTP_TYPE_F16:    أرجع is_tiled ? عدم : convert_f16_بھار_منزل_fp16_tiles_task;
        حالة HTP_TYPE_BF16:   أرجع convert_bf16_بھار_منزل_fp16_tiles_task;
        حالة HTP_TYPE_F32:    أرجع quantize_f32_بھار_منزل_fp16_tiles_task;
        افتراضي:              أرجع عدم;
    }
}

ساكن فراغ dequantize_tiled_بھار_chunk_منزل_fp16_tiles(
        بنية htp_context *ctx, __fp16 *vtcm_dst,
        ثابت فراغ *بھار_src_ddr,
        صحيح n_خانے_گنتی, صحيح k_بلاک,
        size_t صف_stride, صحيح بھار_type,
        صحيح n_k_tiles, بنية fastdiv_اقدار n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_کامہ_fn, صحيح n_دھاگے) {

    إذا (!dequant_کامہ_fn) {
        أرجع;
    }

    أكد(k_بلاک % HTP_MM_HMX_TILE_N_خانے_گنتی == 0);

    ثابت size_t n_خانہ_tiles = hmx_ceil_div(n_خانے_گنتی, HTP_MM_HMX_TILE_N_خانے_گنتی);
    ثابت uint32_t n_بھار_tiles = k_بلاک / htp_tensor_get_tile_k(بھار_type);
    ثابت size_t n_tot_tiles = n_خانہ_tiles * n_بھار_tiles;

    size_t n_tiles_فی_task = (n_دھاگے == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_دھاگے);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_فی_task - 1) / n_tiles_فی_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_فی_task = n_tiles_فی_task;
    state.dst              = vtcm_dst;
    state.src              = (ثابت uint8_t *)بھار_src_ddr;
    state.task_fn          = dequant_کامہ_fn;
    state.n_خانے_گنتی           = n_خانے_گنتی;
    state.k_بلاک          = k_بلاک;
    state.صف_stride       = صف_stride;
    state.بھار_type      = بھار_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_بھار_tiles   = n_بھار_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_ناپ = htp_tensor_get_tile_ناپ(بھار_type);
    state.aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(بھار_type);

    إذا (state.n_tasks == 1 || n_دھاگے == 1) {
        dequantize_tiled_کامہ_loop(1, 0, &state);
    } وإلا {
        صحيح n_tasks = hex_smin((صحيح) state.n_tasks, n_دھاگے);
        کامہ_pool_لگاتار_func(ctx->کامہ_pool, dequantize_tiled_کامہ_loop, &state, n_tasks);
    }
}

تعريفنوع بنية {
    بنية htp_context      * ctx;
    بنية htp_thread_trace * traces;
    عائم                   * dst;
    ثابت __fp16            * vtcm_src;
    ثابت عائم             * src2;
    uint32_t                  n_صفاں;
    uint32_t                  n_خانے_گنتی;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_خانے_گنتی;
    بنية fastdiv_اقدار     n_دھاگے_div;
} output_transfer_خانہ_chunk_state_t;

ساكن فراغ transfer_output_chunk_خانہ_chunk_کامہ_fn(بنانشانی صحيح n, بنانشانی صحيح i, فراغ *انکڑے) {
    (فراغ) n;
    output_transfer_خانہ_chunk_state_t *st = (output_transfer_خانہ_chunk_state_t *) انکڑے;
    بنية htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_بلاکاں = hmx_ceil_div(st->n_خانے_گنتی, 32);
    uint32_t b_پہلا  = fastdiv(n_بلاکاں * i, &st->n_دھاگے_div);
    uint32_t b_آخری   = fastdiv(n_بلاکاں * (i + 1), &st->n_دھاگے_div);
    uint32_t c_پہلا  = b_پہلا * 32;
    uint32_t c_آخری   = b_آخری * 32;
    uint32_t c_len    = c_آخری - c_پہلا;

    إذا (c_len == 0) أرجع;

    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_O_PROC, c_پہلا);

    ثابت __fp16 *vtcm_src = st->vtcm_src + b_پہلا * HTP_MM_HMX_TILE_N_ELMS;
    ثابت عائم *src2      = st->src2 ? (st->src2 + c_پہلا) : عدم;
    عائم *dst             = st->dst + c_پہلا;

    صحيح chunk_dst_خانے_گنتی = (صحيح)st->dst_خانے_گنتی - (صحيح)c_پہلا;
    إذا (chunk_dst_خانے_گنتی > 0) {
        transfer_output_chunk_fp16_منزل_fp32_خانہ_chunk(
            dst, src2, vtcm_src, 0, st->n_صفاں, c_len, st->n_خانے_گنتی,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_خانے_گنتی
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_پہلا);
}

ساكن فراغ transfer_output_chunk_threaded(بنية htp_context *ctx, عائم *dst, ثابت عائم *src2, ثابت __fp16 *vtcm_src,
                                              صحيح n_صفاں, صحيح n_خانے_گنتی, صحيح dst_stride, uint32_t src2_stride, صحيح dst_خانے_گنتی, صحيح n_دھاگے) {
    إذا (n_صفاں <= 0) أرجع;

    uint32_t n_بلاکاں = hmx_ceil_div((uint32_t) n_خانے_گنتی, 32);
    إذا (n_دھاگے > 1 && n_بلاکاں >= (uint32_t)n_دھاگے) {
        بنية fastdiv_اقدار n_دھاگے_div = (n_دھاگے == (صحيح)ctx->n_دھاگے) ? ctx->n_دھاگے_div : init_fastdiv_اقدار(n_دھاگے);
        output_transfer_خانہ_chunk_state_t خانہ_state;
        خانہ_state.dst = dst;
        خانہ_state.src2 = src2;
        خانہ_state.vtcm_src = vtcm_src;
        خانہ_state.n_صفاں = (uint32_t)n_صفاں;
        خانہ_state.n_خانے_گنتی = (uint32_t)n_خانے_گنتی;
        خانہ_state.dst_stride = (uint32_t)dst_stride;
        خانہ_state.src2_stride = src2_stride;
        خانہ_state.dst_خانے_گنتی = (uint32_t)dst_خانے_گنتی;
        خانہ_state.n_دھاگے_div = n_دھاگے_div;
        خانہ_state.traces = ctx->trace;
        خانہ_state.ctx = ctx;

        کامہ_pool_لگاتار_func(ctx->کامہ_pool, transfer_output_chunk_خانہ_chunk_کامہ_fn, &خانہ_state, n_دھاگے);
        أرجع;
    }

    size_t n_tot_chunks      = n_صفاں;
    size_t n_chunks_فی_task = (n_دھاگے == 1) ? n_tot_chunks : hmx_ceil_div(n_صفاں, n_دھاگے);
    n_chunks_فی_task        = hex_align_اتے(n_chunks_فی_task, 2);

    صحيح actual_دھاگے = hmx_ceil_div(n_صفاں, n_chunks_فی_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_دھاگے;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_فی_task = n_chunks_فی_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_خانے_گنتی            = n_خانے_گنتی;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_خانے_گنتی          = dst_خانے_گنتی;
    state.traces            = ctx->trace;

    إذا (actual_دھاگے <= 1) {
        transfer_output_chunk_کامہ_fn(1, 0, &state);
    } وإلا {
        کامہ_pool_لگاتار_func(ctx->کامہ_pool, transfer_output_chunk_کامہ_fn, &state, actual_دھاگے);
    }
}

بنية activation_transfer_params {
    بنية htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    صحيح                           n_صفاں;
    صحيح                           k_بلاک;
    صحيح                           k_stride;
    صحيح                           n_دھاگے;
    ثابت بنية fastdiv_اقدار * act_دھاگے_div;
    ثابت بنية fastdiv_اقدار * k_div;
    صحيح                           k_valid;
    عائم *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_بائٹاں;
    uint32_t                      act_elem_ناپ;  // 4=F32 src1, 2=F16 src1
};

ساكن فراغ transfer_activation_chunk_threaded(ثابت بنية activation_transfer_params * params) {
    بنية htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ثابت dma_addr_t              act_dma_addr       = params->act_dma_addr;
    صحيح                           n_صفاں             = params->n_صفاں;
    صحيح                           k_بلاک            = params->k_بلاک;
    صحيح                           k_stride           = params->k_stride;
    صحيح                           n_دھاگے          = params->n_دھاگے;
    ثابت بنية fastdiv_اقدار * act_دھاگے_div    = params->act_دھاگے_div;
    ثابت بنية fastdiv_اقدار * k_div              = params->k_div;
    صحيح                           k_valid            = params->k_valid;
    عائم *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_بائٹاں = params->vtcm_f32_act_بائٹاں;
    // element size of the activation rows (4 = F32, 2 = F16).
    ثابت uint32_t                act_elem_ناپ      = params->act_elem_ناپ ? params->act_elem_ناپ : (uint32_t) حجم(عائم);

    إذا (n_صفاں <= 0) {
        أرجع;
    }

    ثابت size_t n_tasks = (n_صفاں + 31) >> 5;
    إذا (act_elem_ناپ == حجم(عائم) && n_دھاگے > 1 && k_بلاک > 32 && n_tasks < (size_t) n_دھاگے) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_قدم_صفاں = 2;
        uint32_t dma_قدم_صفاں_کھسکاؤ = 1;
        إذا (vtcm_f32_act_بائٹاں > 0) {
            size_t thread_scratch_بائٹاں = hex_align_تھلے(fastdiv(vtcm_f32_act_بائٹاں, act_دھاگے_div), 128);
            size_t thread_scratch_elements = thread_scratch_بائٹاں / حجم(عائم);
            size_t dma_قدم_صفاں_max = fastdiv(thread_scratch_elements / 2, k_div);
            إذا (dma_قدم_صفاں_max >= 4) {
                dma_قدم_صفاں = 4;
                dma_قدم_صفاں_کھسکاؤ = 2;
            }
        }

        activation_transfer_خانہ_chunk_state_t خانہ_state;
        خانہ_state.dst = dst;
        خانہ_state.act_dma_addr = act_dma_addr;
        خانہ_state.n_صفاں = n_صفاں;
        خانہ_state.k_بلاک = k_بلاک;
        خانہ_state.k_stride = k_stride;
        خانہ_state.k_valid = k_valid;
        خانہ_state.n_خانہ_chunks = n_دھاگے;
        خانہ_state.n_دھاگے_div = *act_دھاگے_div;
        خانہ_state.vtcm_f32_act = vtcm_f32_act;
        خانہ_state.vtcm_f32_act_بائٹاں = vtcm_f32_act_بائٹاں;
        خانہ_state.traces = ctx->trace;
        خانہ_state.ctx = ctx;
        خانہ_state.dma_قدم_صفاں = dma_قدم_صفاں;
        خانہ_state.dma_قدم_صفاں_کھسکاؤ = dma_قدم_صفاں_کھسکاؤ;

        کامہ_pool_لگاتار_func(ctx->کامہ_pool, transfer_activation_chunk_خانہ_chunk_کامہ_fn, &خانہ_state, n_دھاگے);
        أرجع;
    }

    أكد(k_بلاک % HTP_MM_HMX_TILE_N_خانے_گنتی == 0 && k_stride % HTP_MM_HMX_TILE_N_خانے_گنتی == 0);

    size_t n_tot_chunks      = n_صفاں;
    size_t n_chunks_فی_task = (n_دھاگے == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_دھاگے == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
    state.n_tot_chunks       = n_tot_chunks;
    state.n_chunks_فی_task  = n_chunks_فی_task;
    state.dst                = dst;
    state.act_dma_addr       = act_dma_addr;
    state.k_بلاک            = k_بلاک;
    state.k_stride           = k_stride;
    state.k_valid            = k_valid;
    state.traces             = ctx->trace;
    state.ctx                = ctx;
    state.vtcm_f32_act       = vtcm_f32_act;
    state.act_elem_ناپ = act_elem_ناپ;

    state.vtcm_f32_act_بائٹاں_فی_thread = hex_align_تھلے(fastdiv(vtcm_f32_act_بائٹاں, act_دھاگے_div), 128);

    uint32_t dma_قدم_صفاں = 2;
    uint32_t dma_قدم_صفاں_کھسکاؤ = 1;
    إذا (state.vtcm_f32_act_بائٹاں_فی_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_بائٹاں_فی_thread / حجم(عائم);
        size_t dma_قدم_صفاں_max = fastdiv(thread_scratch_elements / 2, k_div);
        إذا (dma_قدم_صفاں_max >= 4) {
            dma_قدم_صفاں = 4;
            dma_قدم_صفاں_کھسکاؤ = 2;
        }
    }
    state.dma_قدم_صفاں      = dma_قدم_صفاں;
    state.dma_قدم_صفاں_کھسکاؤ = dma_قدم_صفاں_کھسکاؤ;

    صحيح سرگرم_دھاگے = hex_smin(n_دھاگے, (صحيح)state.n_tasks);
    إذا (state.n_tasks == 1 || n_دھاگے == 1) {
        transfer_activation_chunk_کامہ_fn(1, 0, &state);
    } وإلا {
        کامہ_pool_لگاتار_func(ctx->کامہ_pool, transfer_activation_chunk_کامہ_fn, &state, سرگرم_دھاگے);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

تعريفنوع بنية {
    __fp16 *       output;
    ثابت __fp16 * activation;
    ثابت __fp16 * بھار;
    ثابت __fp16 * scales;
    uint32_t       n_صف_tiles;
    uint32_t       n_خانہ_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_کم_t;

ساكن فراغ hmx_matmul_کامہ_fn(فراغ * انکڑے) {
    hmx_matmul_کم_t * کم_2 = (hmx_matmul_کم_t *) انکڑے;
    FARF(بلند_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", کم_2->n_صف_tiles, کم_2->n_خانہ_tiles, کم_2->n_dot_tiles);
    core_dot_chunk_fp16(کم_2->output, کم_2->activation, کم_2->بھار, کم_2->scales, کم_2->n_صف_tiles, کم_2->n_خانہ_tiles, کم_2->n_dot_tiles);
}

ساكن مضمن فراغ hmx_matmul_کم_init(hmx_matmul_کم_t * کم_2,
                                       __fp16 *           output,
                                       ثابت __fp16 *     activation,
                                       ثابت __fp16 *     بھار,
                                       ثابت __fp16 *     scales,
                                       uint32_t           n_صف_tiles,
                                       uint32_t           n_خانہ_tiles,
                                       uint32_t           n_dot_tiles) {
    کم_2->output      = output;
    کم_2->activation  = activation;
    کم_2->بھار      = بھار;
    کم_2->scales      = scales;
    کم_2->n_صف_tiles = n_صف_tiles;
    کم_2->n_خانہ_tiles = n_خانہ_tiles;
    کم_2->n_dot_tiles = n_dot_tiles;
}

ساكن صحيح hmx_mm_2d_f32(بنية htp_context *ctx,
                                  dma_قطار *بھار_dma,
                                  عائم *محدود dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_بائٹاں,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t بھار,
                                  صحيح m, صحيح k, صحيح n,
                                  صحيح act_stride,
                                  uint32_t act_elem_ناپ,  // 4=F32 src1, 2=F16 src1
                                  صحيح بھار_stride,
                                  صحيح بھار_type,
                                  صحيح k_valid,
                                  صحيح dst_stride,
                                  uint32_t src2_stride,
                                  صحيح dst_خانے_گنتی,
                                  صحيح m_chunk,
                                  صحيح n_chunk,
                                  صحيح pipeline,
                                  صحيح n_دھاگے,
                                  صحيح act_دھاگے,
                                  ثابت بنية fastdiv_اقدار * act_دھاگے_div,
                                  ثابت بنية fastdiv_اقدار * k_div,
                                  صحيح tile_ناپ,
                                  صحيح aligned_tile_ناپ,
                                  صحيح vtcm_ناپ) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت bool is_tiled = (tile_ناپ > 0);

    إذا (k % 32 != 0) {
        أرجع -1;
    }
    إذا (!hex_is_aligned(dst, حجم(عائم))) { أرجع -1; }

    size_t صف_stride = htp_tensor_get_صف_ناپ(بھار_type, k);
    إذا (صف_stride == 0) {
        أرجع -1;
    }

    tiled_dequant_task_fn_t dequant_کامہ_fn = hmx_get_dequant_کامہ_fn(بھار_type, is_tiled);

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_خانے_گنتی;
    ثابت صحيح n_بھار_tiles = k / htp_tensor_get_tile_k(بھار_type);
    ثابت بنية fastdiv_اقدار n_k_tiles_div = init_fastdiv_اقدار(n_k_tiles);

    ثابت size_t vec_dot_ناپ = k * حجم(__fp16);
    ثابت size_t vtcm_budget  = ctx->vtcm_ناپ;

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_ناپ : صف_stride;
    ثابت uint32_t dma_src_stride  = is_tiled ? tile_ناپ : بھار_stride;
    ثابت uint32_t dma_چوڑائی_بائٹاں = is_tiled ? tile_ناپ : صف_stride;

    size_t m_chunk_n_صفاں = m_chunk;
    size_t n_chunk_n_خانے_گنتی = n_chunk;
    size_t vtcm_ورتیا      = vtcm_ناپ;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, بھار_type, k, m_chunk_n_صفاں, n_chunk_n_خانے_گنتی, 1, pipeline, act_دھاگے, aligned_tile_ناپ, src2_بائٹاں);

    vtcm_ورتیا = L.کل_بائٹاں;
    إذا (vtcm_ورتیا > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_ورتیا, vtcm_budget, m, k, n, m_chunk_n_صفاں, n_chunk_n_خانے_گنتی);
        أرجع -1;
    }

    uint8_t * ثابت ادھار = (uint8_t *) ctx->vtcm_ادھار;
    __fp16  *vtcm_بھار_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_بھار[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ادھار, L.off_بھار[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_act);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, ادھار, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, ادھار, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, ادھار, L.off_scratch[1], pipeline);
    فراغ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, ادھار, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_scales);

    hmx_init_کالم_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ثابت bool has_bias      = (src2_بائٹاں > 0 && src2_addr != 0);
    عائم   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(عائم, ادھار, L.off_bias, has_bias);
    إذا (has_bias) {
        dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_bias, src2_addr), hex_align_اتے(src2_بائٹاں, 128), 0, src2_بائٹاں, 1);
        dma_قطار_نکالو(بھار_dma);
    }

    FARF(بلند_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, بھار_type, m_chunk_n_صفاں, n_chunk_n_خانے_گنتی, vtcm_ورتیا, vtcm_budget);

    صحيح n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_خانے_گنتی);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    إذا (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_کم_t کم_slots[2];  // persistent double-buffered job descriptors
        فراغ *vtcm_raw_bufs[4] = {
            vtcm_بھار_raw[0],
            vtcm_بھار_raw[1],
            dequant_کامہ_fn ? vtcm_بھار_raw[0] : vtcm_scratch0,
            dequant_کامہ_fn ? vtcm_بھار_raw[1] : vtcm_scratch1,
        };
        فراغ *vtcm_بھار_bufs[4] = {
            dequant_کامہ_fn ? vtcm_scratch0 : vtcm_بھار_raw[0],
            dequant_کامہ_fn ? vtcm_scratch1 : vtcm_بھار_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        فراغ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        لكل (size_t mr = 0; mr < m; mr += m_chunk_n_صفاں) {
            ثابت size_t n_صفاں = hex_smin(m - mr, m_chunk_n_صفاں);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ناپ,  // byte offset (F16/F32)
                .n_صفاں = (صحيح) n_صفاں,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_دھاگے = act_دھاگے,
                .act_دھاگے_div = act_دھاگے_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بائٹاں = L.act_f32_بائٹاں,
                .act_elem_ناپ = act_elem_ناپ,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            ثابت size_t   n_خانے_گنتی_A0 = hex_smin(n - 0 * n_chunk_n_خانے_گنتی, n_chunk_n_خانے_گنتی);
            ثابت uint32_t اچائی_A0 = is_tiled ? hmx_ceil_div(n_خانے_گنتی_A0, 32) * n_بھار_tiles : n_خانے_گنتی_A0;
            dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[0], بھار),
                           dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی_A0);

            إذا (1 < n_chunk_cnt) {
                ثابت size_t   n_خانے_گنتی_A1 = hex_smin(n - 1 * n_chunk_n_خانے_گنتی, n_chunk_n_خانے_گنتی);
                ثابت uint32_t اچائی_A1 = is_tiled ? hmx_ceil_div(n_خانے_گنتی_A1, 32) * n_بھار_tiles : n_خانے_گنتی_A1;
                dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[1], بھار + n_chunk_n_خانے_گنتی * بھار_stride),
                               dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                ثابت size_t nc    = i * n_chunk_n_خانے_گنتی;
                ثابت size_t nc_p2 = nc + 2 * n_chunk_n_خانے_گنتی;

                ثابت size_t n_خانے_گنتی    = hex_smin(n - nc, n_chunk_n_خانے_گنتی);
                ثابت size_t n_خانے_گنتی_p2 = hex_smin(n - nc_p2, n_chunk_n_خانے_گنتی);

                // 1. pop A_i
                فراغ * curr_raw = (فراغ *) dma_قطار_نکالو(بھار_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_بھار_chunk_منزل_fp16_tiles(
                    ctx, vtcm_بھار_bufs[i % 4], curr_raw,
                    n_خانے_گنتی, k, صف_stride, بھار_type,
                    n_k_tiles, n_k_tiles_div, dequant_کامہ_fn, n_دھاگے);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                إذا (i + 2 < n_chunk_cnt) {
                    ثابت uint32_t اچائی_p2 = is_tiled ? hmx_ceil_div(n_خانے_گنتی_p2, 32) * n_بھار_tiles : n_خانے_گنتی_p2;
                    dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[(i + 2) % 4], بھار + nc_p2 * بھار_stride),
                                   dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی_p2);
                }

                // 4. submit C_i
                hmx_matmul_کم_init(&کم_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_بھار_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_صفاں, HTP_MM_HMX_TILE_N_صفاں),
                                    hmx_ceil_div(n_خانے_گنتی, HTP_MM_HMX_TILE_N_خانے_گنتی), k / HTP_MM_HMX_TILE_N_صفاں);
                hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کامہ_fn, &کم_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                إذا (i > 0) {
                    hmx_قطار_نکالو(ctx->hmx_قطار);
                    ثابت size_t nc_prev = (i - 1) * n_chunk_n_خانے_گنتی;
                    ثابت size_t n_خانے_گنتی_prev = hex_smin(n - nc_prev, n_chunk_n_خانے_گنتی);
                    عائم *output_chunk = dst + (mr * dst_stride + nc_prev);
                    ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : عدم;
                    صحيح chunk_dst_خانے_گنتی = dst_خانے_گنتی - (صحيح)nc_prev;
                    إذا (chunk_dst_خانے_گنتی > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_صفاں, n_خانے_گنتی_prev, dst_stride, src2_stride, chunk_dst_خانے_گنتی, n_دھاگے);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_قطار_نکالو(ctx->hmx_قطار);
            ثابت size_t nc_آخری = (n_chunk_cnt - 1) * n_chunk_n_خانے_گنتی;
            ثابت size_t n_خانے_گنتی_آخری = hex_smin(n - nc_آخری, n_chunk_n_خانے_گنتی);
            عائم *output_chunk = dst + (mr * dst_stride + nc_آخری);
            ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_آخری) : عدم;
            صحيح chunk_dst_خانے_گنتی = dst_خانے_گنتی - (صحيح)nc_آخری;
            إذا (chunk_dst_خانے_گنتی > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_صفاں, n_خانے_گنتی_آخری, dst_stride, src2_stride, chunk_dst_خانے_گنتی, n_دھاگے);
            }
        }
    } وإلا {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_کم_t کم_2;
        فراغ *vtcm_raw_bufs[2] = { vtcm_بھار_raw[0], dequant_کامہ_fn ? vtcm_بھار_raw[0] : vtcm_scratch0 };
        فراغ *vtcm_بھار_bufs[2] = { dequant_کامہ_fn ? vtcm_scratch0 : vtcm_بھار_raw[0], vtcm_scratch0 };

        لكل (size_t mr = 0; mr < m; mr += m_chunk_n_صفاں) {
            ثابت size_t n_صفاں = hex_smin(m - mr, m_chunk_n_صفاں);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ناپ,  // byte offset (F16/F32)
                .n_صفاں = (صحيح) n_صفاں,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_دھاگے = act_دھاگے,
                .act_دھاگے_div = act_دھاگے_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بائٹاں = L.act_f32_بائٹاں,
                .act_elem_ناپ = act_elem_ناپ,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            إذا (n > 0) {
                ثابت size_t n_خانے_گنتی = hex_smin(n, n_chunk_n_خانے_گنتی);
                ثابت uint32_t اچائی = is_tiled ? hmx_ceil_div(n_خانے_گنتی, 32) * n_بھار_tiles : n_خانے_گنتی;
                dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[0], بھار), dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی);
            }

            لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                ثابت size_t nc = (size_t) i * n_chunk_n_خانے_گنتی;
                ثابت size_t n_خانے_گنتی = hex_smin(n - nc, n_chunk_n_خانے_گنتی);
                ثابت size_t n_صف_tiles = hmx_ceil_div(n_صفاں, HTP_MM_HMX_TILE_N_صفاں);
                ثابت size_t n_خانہ_tiles = hmx_ceil_div(n_خانے_گنتی, HTP_MM_HMX_TILE_N_خانے_گنتی);

                // A: Wait for weight DMA
                فراغ * curr_raw = (فراغ *) dma_قطار_نکالو(بھار_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_بھار_chunk_منزل_fp16_tiles(
                    ctx, vtcm_بھار_bufs[i % 2], curr_raw,
                    n_خانے_گنتی, k, صف_stride, بھار_type,
                    n_k_tiles, n_k_tiles_div, dequant_کامہ_fn, n_دھاگے);

                // Start weight DMA for the next chunk early
                ثابت size_t nc_اگلا = nc + n_chunk_n_خانے_گنتی;
                إذا (nc_اگلا < n) {
                    ثابت size_t n_خانے_گنتی_اگلا = hex_smin(n - nc_اگلا, n_chunk_n_خانے_گنتی);
                    ثابت uint32_t اچائی_اگلا = is_tiled ? hmx_ceil_div(n_خانے_گنتی_اگلا, 32) * n_بھار_tiles : n_خانے_گنتی_اگلا;
                    dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[(i + 1) % 2], بھار + nc_اگلا * بھار_stride), dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی_اگلا);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_کم_init(&کم_2, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_بھار_bufs[i % 2], vtcm_scales, n_صف_tiles, n_خانہ_tiles, k / HTP_MM_HMX_TILE_N_صفاں);
                hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کامہ_fn, &کم_2));
                hmx_قطار_نکالو(ctx->hmx_قطار);

                // D: Output Store
                عائم *output_chunk = dst + (mr * dst_stride + nc);
                ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : عدم;
                صحيح chunk_dst_خانے_گنتی = dst_خانے_گنتی - (صحيح)nc;
                إذا (chunk_dst_خانے_گنتی > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_صفاں, n_خانے_گنتی, dst_stride, src2_stride, chunk_dst_خانے_گنتی, n_دھاگے);
                }
            }
        }
    }

    أرجع 0;
}

ساكن صحيح hmx_mm_nx_2d_f32(بنية htp_ops_context * octx, ثابت بنية htp_mm_kernel_params * kparams) {
    بنية htp_context * ctx = octx->ctx;
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت uint32_t n_weights = kparams->n_weights;
    إذا (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];

    ثابت صحيح بھار_type = (صحيح) src0->type;
    ثابت صحيح k           = (صحيح) act->ne[0];
    ثابت صحيح k_valid     = (صحيح) act->ne[0];
    ثابت صحيح m           = (صحيح) (act->ne[1] * act->ne[2] * act->ne[3]);
    ثابت uint32_t act_elem_ناپ = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت صحيح act_stride  = (صحيح) (act->nb[1] / act_elem_ناپ);
    ثابت dma_addr_t act_dma_addr = act->انکڑے;

    إذا (k % 32 != 0) { أرجع HTP_حالت_NO_SUPPORT; }

    size_t صف_stride = htp_tensor_get_صف_ناپ(بھار_type, k);
    إذا (صف_stride == 0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_خانے_گنتی;
    ثابت صحيح n_بھار_tiles = k / htp_tensor_get_tile_k(بھار_type);
    ثابت بنية fastdiv_اقدار n_k_tiles_div = init_fastdiv_اقدار(n_k_tiles);

    ثابت صحيح tile_ناپ       = kparams->tile_ناپ;
    ثابت صحيح aligned_tile_ناپ = kparams->aligned_tile_ناپ;

    ثابت bool is_tiled       = (tile_ناپ > 0);
    tiled_dequant_task_fn_t dequant_کامہ_fn = hmx_get_dequant_کامہ_fn(بھار_type, is_tiled);
    ثابت size_t vtcm_budget  = ctx->vtcm_ناپ;

    ثابت صحيح m_chunk_n_صفاں  = kparams->m_chunk;
    ثابت صحيح n_chunk_n_خانے_گنتی  = kparams->n_chunk;
    ثابت صحيح pipeline        = kparams->pipeline;
    ثابت صحيح n_دھاگے       = octx->n_دھاگے;
    ثابت صحيح act_دھاگے     = kparams->n_act_دھاگے;
    ثابت بنية fastdiv_اقدار * act_دھاگے_div = &kparams->div_n_act_دھاگے;
    ثابت بنية fastdiv_اقدار * k_div           = &kparams->div_ne00_padded;

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_ناپ : صف_stride;
    ثابت uint32_t dma_چوڑائی_بائٹاں = is_tiled ? tile_ناپ : صف_stride;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, بھار_type, k, m_chunk_n_صفاں, n_chunk_n_خانے_گنتی, 1, pipeline, act_دھاگے, aligned_tile_ناپ, 0);

    إذا (L.کل_بائٹاں > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.کل_بائٹاں, vtcm_budget, m, k, m_chunk_n_صفاں, n_chunk_n_خانے_گنتی);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت ادھار = (uint8_t *) ctx->vtcm_ادھار;
    __fp16  *vtcm_بھار_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_بھار[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ادھار, L.off_بھار[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_act);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, ادھار, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, ادھار, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, ادھار, L.off_scratch[1], pipeline);
    فراغ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, ادھار, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_scales);

    hmx_init_کالم_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    صحيح m_شروع = 0;
    صحيح m_صفاں  = m;
    إذا (octx->ctx->mdev.گنتی > 1) {
        bool can_ونڈو = htp_tensor_can_صف_حصے_کرو(act, act_elem_ناپ);
        إذا (can_ونڈو) {
            لكل (uint32_t p = 0; p < n_weights; ++p) {
                ثابت بنية htp_tensor * محدود dst = octx->dsts[p];
                إذا (!htp_tensor_can_صف_حصے_کرو(dst, حجم(عائم))) {
                    can_ونڈو = false;
                    اكسر;
                }
            }
        }
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو((uint32_t) m, can_ونڈو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        m_شروع = (صحيح) range.شروع;
        m_صفاں  = (صحيح) range.گنتی;
    }

    إذا (m_صفاں == 0) {
        أرجع HTP_حالت_OK;
    }

    FARF(بلند_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_شروع, m_شروع + m_صفاں, k, بھار_type, m_chunk_n_صفاں, n_chunk_n_خانے_گنتی, L.کل_بائٹاں, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت size_t mr_سرا = (size_t)(m_شروع + m_صفاں);

    إذا (pipeline) {
        hmx_matmul_کم_t کم_slots[2];
        فراغ *vtcm_raw_bufs[4] = {
            vtcm_بھار_raw[0],
            vtcm_بھار_raw[1],
            dequant_کامہ_fn ? vtcm_بھار_raw[0] : vtcm_scratch0,
            dequant_کامہ_fn ? vtcm_بھار_raw[1] : vtcm_scratch1,
        };
        فراغ *vtcm_بھار_bufs[4] = {
            dequant_کامہ_fn ? vtcm_scratch0 : vtcm_بھار_raw[0],
            dequant_کامہ_fn ? vtcm_scratch1 : vtcm_بھار_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        فراغ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        لكل (size_t mr = (size_t) m_شروع; mr < mr_سرا; mr += m_chunk_n_صفاں) {
            ثابت size_t n_صفاں = hex_smin(mr_سرا - mr, m_chunk_n_صفاں);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ناپ,
                .n_صفاں = (صحيح) n_صفاں,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_دھاگے = act_دھاگے,
                .act_دھاگے_div = act_دھاگے_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بائٹاں = L.act_f32_بائٹاں,
                .act_elem_ناپ = act_elem_ناپ,
            };
            transfer_activation_chunk_threaded(&act_params);

            لكل (uint32_t p = 0; p < n_weights; p++) {
                ثابت بنية htp_tensor * محدود src_w = octx->src[p];
                ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
                إذا (!src_w || !dst) استمر;

                ثابت dma_addr_t بھار       = src_w->انکڑے;
                dma_قطار * بھار_dma      = octx->ctx->dma[0];
                عائم * dst_ptr            = (عائم *) dst->انکڑے;
                ثابت size_t n             = src_w->ne[1];
                إذا (n == 0) استمر;
                ثابت size_t بھار_stride = src_w->nb[1];
                ثابت size_t dst_stride    = dst->nb[1] / حجم(عائم);
                ثابت صحيح dst_خانے_گنتی         = (صحيح) dst->ne[0];
                ثابت صحيح n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_خانے_گنتی);

                ثابت uint32_t dma_src_stride = is_tiled ? tile_ناپ : بھار_stride;

                ثابت size_t   n_خانے_گنتی_A0 = hex_smin(n - 0 * n_chunk_n_خانے_گنتی, n_chunk_n_خانے_گنتی);
                ثابت uint32_t اچائی_A0 = is_tiled ? hmx_ceil_div(n_خانے_گنتی_A0, 32) * n_بھار_tiles : n_خانے_گنتی_A0;
                dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[0], بھار),
                               dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی_A0);

                إذا (1 < n_chunk_cnt) {
                    ثابت size_t   n_خانے_گنتی_A1 = hex_smin(n - 1 * n_chunk_n_خانے_گنتی, n_chunk_n_خانے_گنتی);
                    ثابت uint32_t اچائی_A1 = is_tiled ? hmx_ceil_div(n_خانے_گنتی_A1, 32) * n_بھار_tiles : n_خانے_گنتی_A1;
                    dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[1], بھار + n_chunk_n_خانے_گنتی * بھار_stride),
                                   dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی_A1);
                }

                لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                    ثابت size_t nc    = i * n_chunk_n_خانے_گنتی;
                    ثابت size_t nc_p2 = nc + 2 * n_chunk_n_خانے_گنتی;

                    ثابت size_t n_خانے_گنتی    = hex_smin(n - nc, n_chunk_n_خانے_گنتی);
                    ثابت size_t n_خانے_گنتی_p2 = hex_smin(n - nc_p2, n_chunk_n_خانے_گنتی);

                    فراغ * curr_raw = (فراغ *) dma_قطار_نکالو(بھار_dma).dst;

                    dequantize_tiled_بھار_chunk_منزل_fp16_tiles(
                        ctx, vtcm_بھار_bufs[i % 4], curr_raw,
                        n_خانے_گنتی, k, صف_stride, بھار_type,
                        n_k_tiles, n_k_tiles_div, dequant_کامہ_fn, n_دھاگے);

                    إذا (i + 2 < n_chunk_cnt) {
                        ثابت uint32_t اچائی_p2 = is_tiled ? hmx_ceil_div(n_خانے_گنتی_p2, 32) * n_بھار_tiles : n_خانے_گنتی_p2;
                        dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[(i + 2) % 4], بھار + nc_p2 * بھار_stride),
                                       dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی_p2);
                    }

                    hmx_matmul_کم_init(&کم_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_بھار_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_صفاں, HTP_MM_HMX_TILE_N_صفاں),
                                        hmx_ceil_div(n_خانے_گنتی, HTP_MM_HMX_TILE_N_خانے_گنتی), k / HTP_MM_HMX_TILE_N_صفاں);
                    hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کامہ_fn, &کم_slots[i % 2]));

                    إذا (i > 0) {
                        hmx_قطار_نکالو(ctx->hmx_قطار);
                        ثابت size_t nc_prev = (i - 1) * n_chunk_n_خانے_گنتی;
                        ثابت size_t n_خانے_گنتی_prev = hex_smin(n - nc_prev, n_chunk_n_خانے_گنتی);
                        عائم *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        صحيح chunk_dst_خانے_گنتی = dst_خانے_گنتی - (صحيح)nc_prev;
                        إذا (chunk_dst_خانے_گنتی > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output_bufs[(i - 1) % 2], n_صفاں, n_خانے_گنتی_prev, dst_stride, 0, chunk_dst_خانے_گنتی, n_دھاگے);
                        }
                    }
                }

                hmx_قطار_نکالو(ctx->hmx_قطار);
                ثابت size_t nc_آخری = (n_chunk_cnt - 1) * n_chunk_n_خانے_گنتی;
                ثابت size_t n_خانے_گنتی_آخری = hex_smin(n - nc_آخری, n_chunk_n_خانے_گنتی);
                عائم *output_chunk = dst_ptr + (mr * dst_stride + nc_آخری);
                صحيح chunk_dst_خانے_گنتی = dst_خانے_گنتی - (صحيح)nc_آخری;
                إذا (chunk_dst_خانے_گنتی > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_صفاں, n_خانے_گنتی_آخری, dst_stride, 0, chunk_dst_خانے_گنتی, n_دھاگے);
                }
            }
        }
    } وإلا {
        hmx_matmul_کم_t کم_2;
        فراغ *vtcm_raw_bufs[2] = { vtcm_بھار_raw[0], dequant_کامہ_fn ? vtcm_بھار_raw[0] : vtcm_scratch0 };
        فراغ *vtcm_بھار_bufs[2] = { dequant_کامہ_fn ? vtcm_scratch0 : vtcm_بھار_raw[0], vtcm_scratch0 };

        لكل (size_t mr = (size_t) m_شروع; mr < mr_سرا; mr += m_chunk_n_صفاں) {
            ثابت size_t n_صفاں = hex_smin(mr_سرا - mr, m_chunk_n_صفاں);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_ناپ,
                .n_صفاں = (صحيح) n_صفاں,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_دھاگے = act_دھاگے,
                .act_دھاگے_div = act_دھاگے_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بائٹاں = L.act_f32_بائٹاں,
                .act_elem_ناپ = act_elem_ناپ,
            };
            transfer_activation_chunk_threaded(&act_params);

            لكل (uint32_t p = 0; p < n_weights; p++) {
                ثابت بنية htp_tensor * محدود src_w = octx->src[p];
                ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
                إذا (!src_w || !dst) استمر;

                ثابت dma_addr_t بھار       = src_w->انکڑے;
                dma_قطار * بھار_dma      = octx->ctx->dma[0];
                عائم * dst_ptr            = (عائم *) dst->انکڑے;
                ثابت size_t n             = src_w->ne[1];
                إذا (n == 0) استمر;
                ثابت size_t بھار_stride = src_w->nb[1];
                ثابت size_t dst_stride    = dst->nb[1] / حجم(عائم);
                ثابت صحيح dst_خانے_گنتی         = (صحيح) dst->ne[0];
                ثابت صحيح n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_خانے_گنتی);

                ثابت uint32_t dma_src_stride = is_tiled ? tile_ناپ : بھار_stride;

                إذا (n > 0) {
                    ثابت size_t n_خانے_گنتی = hex_smin(n, n_chunk_n_خانے_گنتی);
                    ثابت uint32_t اچائی = is_tiled ? hmx_ceil_div(n_خانے_گنتی, 32) * n_بھار_tiles : n_خانے_گنتی;
                    dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[0], بھار), dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی);
                }

                لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                    ثابت size_t nc = (size_t) i * n_chunk_n_خانے_گنتی;
                    ثابت size_t n_خانے_گنتی = hex_smin(n - nc, n_chunk_n_خانے_گنتی);
                    ثابت size_t n_صف_tiles = hmx_ceil_div(n_صفاں, HTP_MM_HMX_TILE_N_صفاں);
                    ثابت size_t n_خانہ_tiles = hmx_ceil_div(n_خانے_گنتی, HTP_MM_HMX_TILE_N_خانے_گنتی);

                    فراغ * curr_raw = (فراغ *) dma_قطار_نکالو(بھار_dma).dst;

                    dequantize_tiled_بھار_chunk_منزل_fp16_tiles(
                        ctx, vtcm_بھار_bufs[i % 2], curr_raw,
                        n_خانے_گنتی, k, صف_stride, بھار_type,
                        n_k_tiles, n_k_tiles_div, dequant_کامہ_fn, n_دھاگے);

                    ثابت size_t nc_اگلا = nc + n_chunk_n_خانے_گنتی;
                    إذا (nc_اگلا < n) {
                        ثابت size_t n_خانے_گنتی_اگلا = hex_smin(n - nc_اگلا, n_chunk_n_خانے_گنتی);
                        ثابت uint32_t اچائی_اگلا = is_tiled ? hmx_ceil_div(n_خانے_گنتی_اگلا, 32) * n_بھار_tiles : n_خانے_گنتی_اگلا;
                        dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[(i + 1) % 2], بھار + nc_اگلا * بھار_stride), dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی_اگلا);
                    }

                    hmx_matmul_کم_init(&کم_2, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_بھار_bufs[i % 2], vtcm_scales, n_صف_tiles, n_خانہ_tiles, k / HTP_MM_HMX_TILE_N_صفاں);
                    hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کامہ_fn, &کم_2));
                    hmx_قطار_نکالو(ctx->hmx_قطار);

                    عائم *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    صحيح chunk_dst_خانے_گنتی = dst_خانے_گنتی - (صحيح)nc;
                    إذا (chunk_dst_خانے_گنتی > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output, n_صفاں, n_خانے_گنتی, dst_stride, 0, chunk_dst_خانے_گنتی, n_دھاگے);
                    }
                }
            }
        }
    }

    أرجع HTP_حالت_OK;
}

ساكن مضمن dma_addr_t hmx_mm_بھار_batch_انکڑے(ثابت hmx_mm_batched_params_t *params,
                                                         صحيح dst_b2, صحيح dst_b3) {
    ثابت size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    ثابت size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    أرجع params->بھار + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

ساكن مضمن dma_addr_t hmx_mm_act_batch_addr(ثابت hmx_mm_batched_params_t *params,
                                                صحيح dst_b2, صحيح dst_b3) {
    أرجع params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

ساكن مضمن عائم *hmx_mm_dst_batch_ptr(ثابت hmx_mm_batched_params_t *params,
                                              صحيح dst_b2, صحيح dst_b3) {
    أرجع (عائم *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

ساكن صحيح hmx_mm_batched_simple(بنية htp_context *ctx,
                                 ثابت hmx_mm_batched_params_t *params,
                                 صحيح m_chunk, صحيح n_chunk, صحيح pipeline, صحيح n_دھاگے, صحيح act_دھاگے, صحيح vtcm_ناپ,
                                 ثابت بنية fastdiv_اقدار * act_دھاگے_div, ثابت بنية fastdiv_اقدار * k_div) {
    صحيح ret = 0;
    لكل (صحيح b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        لكل (صحيح b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->بھار_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_بائٹاں,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_بھار_batch_انکڑے(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_ناپ, params->بھار_stride,
                                params->بھار_type, params->k, params->dst_stride, params->bias_stride, params->dst_خانے_گنتی,
                                m_chunk, n_chunk, pipeline, n_دھاگے, act_دھاگے,
                                act_دھاگے_div, k_div, params->tile_ناپ, params->aligned_tile_ناپ, vtcm_ناپ);
        }
    }
    أرجع ret;
}

ساكن صحيح hmx_mm_batched(بنية htp_context *ctx, ثابت hmx_mm_batched_params_t *params,
                          صحيح m_chunk, صحيح n_chunk, صحيح pipeline, صحيح n_دھاگے, صحيح act_دھاگے,
                          ثابت بنية fastdiv_اقدار * act_دھاگے_div,
                          ثابت بنية fastdiv_اقدار * k_div,
                          صحيح vtcm_ناپ) {
    ثابت size_t صف_stride = htp_tensor_get_صف_ناپ(params->بھار_type, params->k);
    إذا (صف_stride == 0) { أرجع -1; }
    إذا (params->act_stride < params->k || (params->tile_ناپ == 0 && (size_t) params->بھار_stride < صف_stride) || params->dst_stride < params->dst_خانے_گنتی) { أرجع -1; }
    إذا (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { أرجع -1; }
    إذا (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { أرجع -1; }
    // N (the weight row count) does not have to be 32-aligned:
    إذا (params->k % 32 != 0) { أرجع -1; }
    إذا (!hex_is_aligned(params->dst, حجم(عائم))) { أرجع -1; }

    ثابت صحيح گروہ_ناپ = params->r2;
    ثابت size_t vtcm_budget  = ctx->vtcm_ناپ;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool لگاتار_grouped = (params->بھار_type == HTP_TYPE_F16 && params->tile_ناپ == 0 && گروہ_ناپ > 1 && (size_t)vtcm_ناپ <= vtcm_budget);
    إذا (!لگاتار_grouped) {
        أرجع hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_دھاگے, act_دھاگے, vtcm_ناپ, act_دھاگے_div, k_div);
    }

    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_صفاں = m_chunk;
    size_t n_chunk_n_خانے_گنتی = n_chunk;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_صفاں, n_chunk_n_خانے_گنتی, گروہ_ناپ, false, act_دھاگے, 0, params->bias_بائٹاں);

    إذا (L.کل_بائٹاں > vtcm_budget) {
        FARF(بلند_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        أرجع hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_دھاگے, act_دھاگے, vtcm_ناپ, act_دھاگے_div, k_div);
    }

    uint8_t * ثابت ادھار = (uint8_t *) ctx->vtcm_ادھار;
    __fp16  *vtcm_بھار     = VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_بھار[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, ادھار, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR(فراغ, ادھار, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, ادھار, L.off_scales);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, ادھار, L.off_act_f32);

    ثابت bool has_bias      = (params->bias_بائٹاں > 0 && params->bias_addr != 0);
    عائم   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(عائم, ادھار, L.off_bias, has_bias);
    إذا (has_bias) {
        dma_قطار_ڈالو(params->بھار_dma, dma_make_انکڑے(vtcm_bias, params->bias_addr), hex_align_اتے(params->bias_بائٹاں, 128), 0, params->bias_بائٹاں, 1);
        dma_قطار_نکالو(params->بھار_dma);
    }

    hmx_init_کالم_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(بلند_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, گروہ_ناپ, params->ne13,
            m_chunk_n_صفاں, n_chunk_n_خانے_گنتی,
            L.کل_بائٹاں, vtcm_budget);

    ثابت size_t fp16_صف_بائٹاں   = (size_t) params->k * حجم(__fp16);
    ثابت size_t بھار_صف_بائٹاں = (size_t) params->بھار_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_کم_t کم_2;

    لكل (صحيح b3 = 0; b3 < params->ne13; ++b3) {
        لكل (صحيح b2_ادھار = 0; b2_ادھار < params->ne12; b2_ادھار += گروہ_ناپ) {
            ثابت dma_addr_t بھار_گروہ = hmx_mm_بھار_batch_انکڑے(params, b2_ادھار, b3);
            dma_قطار * بھار_dma = params->بھار_dma;

            لكل (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_صفاں) {
                ثابت size_t n_صفاں = hex_smin((size_t) params->m - mr, m_chunk_n_صفاں);
                ثابت size_t n_صف_tiles = hmx_ceil_div((صحيح) n_صفاں, HTP_MM_HMX_TILE_N_صفاں);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                لكل (صحيح g = 0; g < گروہ_ناپ; ++g) {
                    ثابت dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_ادھار + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_ناپ;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_سر_stride;
                    بنية activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_صفاں = (صحيح) n_صفاں,
                        .k_بلاک = params->k,
                        .k_stride = params->act_stride,
                        .n_دھاگے = act_دھاگے,
                        .act_دھاگے_div = act_دھاگے_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_بائٹاں = L.act_f32_بائٹاں,
                        .act_elem_ناپ = params->act_elem_ناپ,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    ثابت size_t n_خانے_گنتی_پہلا = hex_smin((size_t) params->n, n_chunk_n_خانے_گنتی);
                    dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_scratch0, بھار_گروہ),
                                      fp16_صف_بائٹاں, بھار_صف_بائٹاں, fp16_صف_بائٹاں, n_خانے_گنتی_پہلا);
                }
                إذا (n_chunk_n_خانے_گنتی < (size_t) params->n) {
                    ثابت size_t n_خانے_گنتی_دوجا = hex_smin((size_t) params->n - n_chunk_n_خانے_گنتی, n_chunk_n_خانے_گنتی);
                    ثابت dma_addr_t دوجا_بھار_chunk = بھار_گروہ + n_chunk_n_خانے_گنتی * بھار_صف_بائٹاں;
                    dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_scratch1, دوجا_بھار_chunk),
                                      fp16_صف_بائٹاں, بھار_صف_بائٹاں, fp16_صف_بائٹاں, n_خانے_گنتی_دوجا);
                }

                لكل (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_خانے_گنتی) {
                    ثابت size_t n_خانے_گنتی      = hex_smin((size_t) params->n - nc, n_chunk_n_خانے_گنتی);
                    ثابت size_t n_خانہ_tiles = hmx_ceil_div((صحيح) n_خانے_گنتی, HTP_MM_HMX_TILE_N_خانے_گنتی);

                    {
                        فراغ * curr_raw = (فراغ *) dma_قطار_نکالو(بھار_dma).dst;

                        ثابت size_t n_خانے_گنتی_tiled = hex_align_اتے(n_خانے_گنتی, HTP_MM_HMX_TILE_N_خانے_گنتی);

                        hmx_interleave_صفاں_منزل_tiles(vtcm_بھار, (ثابت __fp16 *) curr_raw, (uint32_t) n_خانے_گنتی, params->k, params->k, 0, (uint32_t) n_خانے_گنتی_tiled);

                        ثابت size_t nc_اگلا = nc + n_chunk_n_خانے_گنتی * 2;
                        إذا (nc_اگلا < (size_t) params->n) {
                            ثابت size_t n_خانے_گنتی_اگلا = hex_smin((size_t) params->n - nc_اگلا, n_chunk_n_خانے_گنتی);
                            ثابت dma_addr_t اگلا_بھار_chunk = بھار_گروہ + nc_اگلا * بھار_صف_بائٹاں;

                            dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(curr_raw, اگلا_بھار_chunk),
                                              fp16_صف_بائٹاں, بھار_صف_بائٹاں, fp16_صف_بائٹاں, n_خانے_گنتی_اگلا);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    لكل (صحيح g = 0; g < گروہ_ناپ; ++g) {
                        {
                            ثابت __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_سر_stride;
                            hmx_matmul_کم_init(&کم_2, vtcm_output, vtcm_act_g, vtcm_بھار, vtcm_scales, n_صف_tiles, n_خانہ_tiles, params->k / 32);
                            hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کامہ_fn, &کم_2));
                            hmx_قطار_نکالو(ctx->hmx_قطار);
                        }

                        {
                            عائم *output = hmx_mm_dst_batch_ptr(params, b2_ادھار + g, b3) + mr * params->dst_stride + nc;
                            ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : عدم;
                            صحيح chunk_dst_خانے_گنتی = params->dst_خانے_گنتی - (صحيح)nc;
                            إذا (chunk_dst_خانے_گنتی > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (صحيح) n_صفاں, (صحيح) n_خانے_گنتی,
                                                               params->dst_stride, params->bias_stride, chunk_dst_خانے_گنتی, n_دھاگے);
                            }
                        }
                    }
                }
            }
        }
    }

    أرجع 0;
}

ساكن فراغ transfer_activation_chunk_gathered_threaded(
            بنية htp_context *ctx,
            __fp16 *dst,
            ثابت عائم *src,
            صحيح شروع_صف,
            صحيح n_صفاں,
            صحيح k_بلاک,
            ثابت بنية mmid_صف_mapping *میٹرکس_صفاں,
            صحيح cur_a,
            صحيح mapping_stride,
            صحيح ne11,
            size_t nb11,
            size_t nb12,
            صحيح cne1,
            صحيح n_دھاگے,
            صحيح k_valid) {
    إذا (n_صفاں <= 0) أرجع;
    صحيح chunks_فی_thread = hmx_ceil_div(n_صفاں, n_دھاگے);
    chunks_فی_thread = hex_align_اتے(chunks_فی_thread, 2);

    صحيح actual_دھاگے = hmx_ceil_div(n_صفاں, chunks_فی_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_دھاگے,
        .n_tot_chunks      = n_صفاں,
        .n_chunks_فی_task = chunks_فی_thread,
        .k_بلاک           = k_بلاک,
        .میٹرکس_صفاں       = میٹرکس_صفاں,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_اقدار(ne11) : (بنية fastdiv_اقدار){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .شروع_صف         = شروع_صف,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    کامہ_callback_t کامہ_fn = ne11 == 1 ? transfer_activation_chunk_gathered_کامہ_flat_fn :
                                              transfer_activation_chunk_gathered_کامہ_fn;

    إذا (actual_دھاگے <= 1) {
        کامہ_fn(1, 0, &state);
    } وإلا {
        کامہ_pool_لگاتار_func(ctx->کامہ_pool, کامہ_fn, &state, actual_دھاگے);
    }
}

ساكن فراغ transfer_output_chunk_scattered_threaded(
            بنية htp_context *ctx,
            عائم *dst,
            ثابت __fp16 *vtcm_src,
            صحيح شروع_صف,
            صحيح n_صفاں,
            صحيح n_خانے_گنتی,
            ثابت بنية mmid_صف_mapping *میٹرکس_صفاں,
            صحيح cur_a,
            صحيح mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            صحيح cne1,
            صحيح n_دھاگے) {
    إذا (n_صفاں <= 0) أرجع;
    صحيح chunks_فی_thread = hmx_ceil_div(n_صفاں, n_دھاگے);
    chunks_فی_thread = hex_align_اتے(chunks_فی_thread, 2);

    صحيح actual_دھاگے = hmx_ceil_div(n_صفاں, chunks_فی_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_دھاگے,
        .n_tot_chunks      = n_صفاں,
        .n_chunks_فی_task = chunks_فی_thread,
        .n_خانے_گنتی            = n_خانے_گنتی,
        .میٹرکس_صفاں       = میٹرکس_صفاں,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .شروع_صف         = شروع_صف,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    إذا (actual_دھاگے <= 1) {
        transfer_output_chunk_scattered_کامہ_fn(1, 0, &state);
    } وإلا {
        کامہ_pool_لگاتار_func(ctx->کامہ_pool, transfer_output_chunk_scattered_کامہ_fn, &state, actual_دھاگے);
    }
}

ساكن صحيح hmx_mm_id_2d_f32(بنية htp_context *ctx,
                                         dma_قطار *بھار_dma,
                                         عائم *محدود dst,
                                         ثابت عائم *activation,
                                         dma_addr_t بھار,
                                         صحيح m, صحيح k, صحيح n,
                                         صحيح k_valid,
                                         صحيح ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         صحيح بھار_stride,
                                         صحيح بھار_type,
                                         صحيح tile_ناپ,
                                         صحيح aligned_tile_ناپ,
                                         ثابت بنية mmid_صف_mapping *میٹرکس_صفاں,
                                         صحيح cur_a,
                                         صحيح mapping_stride,
                                         صحيح m_شروع,
                                         صحيح m_سرا,
                                         صحيح n_دھاگے) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت صحيح cne1 = m;
    ثابت صحيح m_core = m_سرا - m_شروع;
    ثابت صحيح m_core_padded = hex_align_اتے(m_core > 0 ? m_core : 1, 32);

    إذا (k % 32 != 0 || n % 32 != 0) { أرجع -1; }
    إذا (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { أرجع -1; }

    size_t صف_stride = htp_tensor_get_صف_ناپ(بھار_type, k);
    إذا (صف_stride == 0) {
        أرجع -1;
    }

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_خانے_گنتی;
    ثابت uint32_t n_بھار_tiles = k / htp_tensor_get_tile_k(بھار_type);
    ثابت بنية fastdiv_اقدار n_k_tiles_div = init_fastdiv_اقدار(n_k_tiles);

    ثابت size_t vec_dot_ناپ = k * حجم(__fp16);
    ثابت size_t vtcm_budget  = ctx->vtcm_ناپ;
    size_t vtcm_ورتیا = 0;

    ثابت bool is_tiled = (tile_ناپ > 0);
    tiled_dequant_task_fn_t dequant_کامہ_fn = hmx_get_dequant_کامہ_fn(بھار_type, is_tiled);

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_ناپ : صف_stride;
    ثابت uint32_t dma_src_stride  = is_tiled ? tile_ناپ : بھار_stride;
    ثابت uint32_t dma_چوڑائی_بائٹاں = is_tiled ? tile_ناپ : صف_stride;

    ثابت size_t qweight_صف_stride = is_tiled ? (size_t)(n_بھار_tiles * aligned_tile_ناپ) / 32 : 0;
    ثابت size_t بھار_صف_stride = is_tiled ? qweight_صف_stride : صف_stride;

    size_t ناپ_فی_n = 0, ناپ_فی_m = 0, ناپ_فی_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(بھار_type, k, /*pipeline=*/false, aligned_tile_ناپ,
                                  &ناپ_فی_n, &ناپ_فی_m, &ناپ_فی_mn);

    ثابت size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_صفاں = 0, n_chunk_n_خانے_گنتی = 0;
    إذا (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, ناپ_فی_n, ناپ_فی_m, ناپ_فی_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_لاگت_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_لاگت_A_CONVERT, &m_chunk_n_صفاں, &n_chunk_n_خانے_گنتی, &vtcm_ورتیا)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        أرجع -1;
    }

    ثابت size_t بھار_area_ناپ = hex_align_اتے(n_chunk_n_خانے_گنتی * بھار_صف_stride, HTP_MM_HMX_TILE_ناپ);
    ثابت size_t act_area_ناپ    = hex_align_اتے(m_chunk_n_صفاں * vec_dot_ناپ, HTP_MM_HMX_TILE_ناپ);
    ثابت size_t output_area_ناپ = hex_align_اتے(m_chunk_n_صفاں * n_chunk_n_خانے_گنتی * حجم(__fp16), HTP_MM_HMX_TILE_ناپ);

    size_t scratch0_ناپ = hex_align_اتے(n_chunk_n_خانے_گنتی * vec_dot_ناپ, HTP_MM_HMX_TILE_ناپ);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_ادھار;
    __fp16  *vtcm_بھار   = بھار_area_ناپ ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, بھار_area_ناپ) : عدم;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_ناپ);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_ناپ);
    فراغ    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_ناپ);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_ورتیا = vtcm_ptr - (uint8_t *) ctx->vtcm_ادھار;
    إذا (vtcm_ورتیا > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_ورتیا, vtcm_budget);
        أرجع -1;
    }

    hmx_init_کالم_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_کم_t کم_2;
    فراغ *vtcm_raw_bufs[2] = { vtcm_بھار, dequant_کامہ_fn ? vtcm_بھار : vtcm_scratch0 };
    فراغ *vtcm_بھار_bufs[2] = { dequant_کامہ_fn ? vtcm_scratch0 : vtcm_بھار, vtcm_scratch0 };
    ثابت صحيح n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_خانے_گنتی);

    لكل (size_t mr = (size_t) m_شروع; mr < (size_t) m_سرا; mr += m_chunk_n_صفاں) {
        ثابت size_t n_صفاں = hex_smin((size_t) m_سرا - mr, m_chunk_n_صفاں);
        ثابت size_t n_صف_tiles = hmx_ceil_div(n_صفاں, HTP_MM_HMX_TILE_N_صفاں);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (صحيح) mr, (صحيح) n_صفاں, k,
            میٹرکس_صفاں, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_دھاگے, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        إذا (n > 0) {
            ثابت size_t n_خانے_گنتی = hex_smin((size_t) n, n_chunk_n_خانے_گنتی);
            ثابت uint32_t اچائی = is_tiled ? hmx_ceil_div(n_خانے_گنتی, 32) * n_بھار_tiles : n_خانے_گنتی;
            dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[0], بھار),
                           dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی);
        }

        لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
            ثابت size_t nc = (size_t) i * n_chunk_n_خانے_گنتی;
            ثابت size_t n_خانے_گنتی = hex_smin((size_t) n - nc, n_chunk_n_خانے_گنتی);
            ثابت size_t n_خانہ_tiles = hmx_ceil_div(n_خانے_گنتی, HTP_MM_HMX_TILE_N_خانے_گنتی);

            // A: Wait for weight DMA
            فراغ * curr_raw = (فراغ *) dma_قطار_نکالو(بھار_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_بھار_chunk_منزل_fp16_tiles(
                ctx, vtcm_بھار_bufs[i % 2], curr_raw,
                n_خانے_گنتی, k, صف_stride, بھار_type,
                n_k_tiles, n_k_tiles_div, dequant_کامہ_fn, n_دھاگے
            );

            // Start weight DMA for the next chunk early
            ثابت size_t nc_اگلا = nc + n_chunk_n_خانے_گنتی;
            إذا (nc_اگلا < (size_t) n) {
                ثابت size_t n_خانے_گنتی_اگلا = hex_smin((size_t) n - nc_اگلا, n_chunk_n_خانے_گنتی);
                ثابت uint32_t اچائی_اگلا = is_tiled ? hmx_ceil_div(n_خانے_گنتی_اگلا, 32) * n_بھار_tiles : n_خانے_گنتی_اگلا;
                dma_قطار_ڈالو(بھار_dma, dma_make_انکڑے(vtcm_raw_bufs[(i + 1) % 2], بھار + nc_اگلا * بھار_stride),
                               dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹاں, اچائی_اگلا);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_کم_init(&کم_2, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_بھار_bufs[i % 2], vtcm_scales, n_صف_tiles, n_خانہ_tiles, k / HTP_MM_HMX_TILE_N_صفاں);
            hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کامہ_fn, &کم_2));
            hmx_قطار_نکالو(ctx->hmx_قطار);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (صحيح) mr, (صحيح) n_صفاں, (صحيح) n_خانے_گنتی,
                میٹرکس_صفاں, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_دھاگے);
        }
    }

    أرجع 0;
}

// --- Dispatchers and Public Entry Points ---

ساكن صحيح hmx_mm_op_matmul(بنية htp_ops_context * octx, ثابت بنية htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    صحيح k = (صحيح) src0->ne[0];
    صحيح n = (صحيح) src0->ne[1];
    ثابت صحيح m_کل    = (صحيح) act->ne[1];
    ثابت uint32_t act_elem_ناپ = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت صحيح act_stride = (صحيح) (act->nb[1] / act_elem_ناپ);

    صحيح m_شروع = 0;
    صحيح m_صفاں  = m_کل;
    إذا (octx->ctx->mdev.گنتی > 1) {
        bool can_ونڈو = htp_tensor_can_صف_حصے_کرو(dst, حجم(عائم)) &&
                         htp_tensor_can_صف_حصے_کرو(act, act_elem_ناپ);
        إذا (src2 && src2->ne[1] > 1 && !htp_tensor_can_صف_حصے_کرو(src2, حجم(عائم))) {
            can_ونڈو = false;
        }
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو((uint32_t) m_کل, can_ونڈو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        m_شروع = (صحيح) range.شروع;
        m_صفاں  = (صحيح) range.گنتی;
    }

    إذا (m_صفاں == 0) {
        أرجع HTP_حالت_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_بائٹاں = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    إذا (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / حجم(عائم));
        src2_addr = src2->انکڑے + m_شروع * src2_stride * حجم(عائم);
        src2_بائٹاں = (size_t) kparams->vtcm_bias_ناپ;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ثابت صحيح dst_stride = (صحيح)(dst->nb[1] / حجم(عائم));
    عائم       * dst_ptr = (عائم *) dst->انکڑے + m_شروع * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ثابت dma_addr_t act_addr = act->انکڑے + (size_t) m_شروع * act_stride * act_elem_ناپ;

    صحيح ret = -1;
    ثابت صحيح n_دھاگے = kparams->n_دھاگے;
    إذا (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        hmx_mm_batched_params_t batch_params = {
            .dst               = dst_ptr,
            .bias_addr         = src2_addr,
            .bias_بائٹاں        = src2_بائٹاں,
            .act_dma_addr      = act_addr,
            .بھار            = src0->انکڑے,
            .بھار_dma        = octx->ctx->dma[0],
            .m                 = m_صفاں,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_ناپ     = act_elem_ناپ,
            .بھار_stride     = (صحيح) src0->nb[1],
            .بھار_type       = (صحيح) src0->type,
            .dst_stride        = dst_stride,
            .dst_خانے_گنتی          = (صحيح) dst->ne[0],
            .bias_stride       = src2_stride,
            .tile_ناپ         = kparams->tile_ناپ,
            .aligned_tile_ناپ = kparams->aligned_tile_ناپ,
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
                             kparams->pipeline, n_دھاگے,
                             kparams->n_act_دھاگے,
                             &kparams->div_n_act_دھاگے,
                             &kparams->div_ne00_padded,
                             kparams->vtcm_ناپ);
    } وإلا {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_بائٹاں,
            act_addr, src0->انکڑے,
            m_صفاں, k, n, act_stride, act_elem_ناپ, (صحيح) src0->nb[1], (صحيح) src0->type, (صحيح) act->ne[0],
            dst_stride, src2_stride, (صحيح)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_دھاگے,
            kparams->n_act_دھاگے,
            &kparams->div_n_act_دھاگے,
            &kparams->div_ne00_padded,
            kparams->tile_ناپ, kparams->aligned_tile_ناپ, kparams->vtcm_ناپ
        );
    }

    إذا (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        أرجع HTP_حالت_اندرونی_ERR;
    }
    أرجع HTP_حالت_OK;
}

ساكن مضمن فراغ htp_mm_tensor_collapse_صفاں(بنية htp_tensor * c, ثابت بنية htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

ساكن صحيح op_matmul_impl(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    ثابت صحيح حالت = htp_mm_init_context(octx, kparams);
    إذا (حالت != HTP_حالت_OK) {
        أرجع حالت;
    }

    إذا (kparams->n_hmx) {
        أرجع hmx_mm_op_matmul(octx, kparams);
    }

    أرجع hvx_mm_matmul(octx);
}

صحيح op_matmul(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    إذا (kparams->collapse) {
        ثابت بنية htp_tensor * act  = octx->src[1];
        ثابت بنية htp_tensor * dst  = octx->dst;
        ثابت uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ثابت uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        بنية htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_صفاں(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_صفاں(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        بنية htp_tensor src2_collapsed;
        ثابت بنية htp_tensor * src2 = octx->src[2];
        إذا (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ثابت uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_صفاں(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        ثابت صحيح حالت = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        إذا (src2) {
            octx->src[2] = src2;
        }
        أرجع حالت;
    }

    أرجع op_matmul_impl(octx);
}

ساكن صحيح hmx_mm_op_matmul_id(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx
) {
    ثابت uint32_t *                میٹرکس_صف_counts = mmctx->میٹرکس_صف_counts;
    ثابت بنية mmid_صف_mapping * میٹرکس_صفاں       = mmctx->میٹرکس_صفاں;
    htp_matmul_tensors_preamble;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت صحيح n_ids = octx->src[2]->ne[0];
    ثابت صحيح n_as  = ne02;

    ثابت bool mdev_ونڈو = (octx->ctx->mdev.گنتی > 1) && htp_tensor_can_صف_حصے_کرو(dst, حجم(عائم));
    إذا (octx->ctx->mdev.گنتی > 1 && !mdev_ونڈو && octx->ctx->mdev.idx > 0) {
        أرجع HTP_حالت_OK;
    }
    uint32_t n_سرگرم = 0;
    إذا (mdev_ونڈو) {
        لكل (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            إذا (میٹرکس_صف_counts[a] > 0) n_سرگرم++;
        }
    }
    ثابت bool expert_ونڈو = mdev_ونڈو && (n_سرگرم >= octx->ctx->mdev.گنتی);

    uint32_t نشانہ_dev = 0;
    لكل (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ثابت int32_t cne1 = میٹرکس_صف_counts[cur_a];
        إذا (cne1 == 0) استمر;

        ثابت صحيح m_padded = hex_align_اتے(cne1, 32);
        صحيح m_شروع = 0, m_سرا = m_padded;
        إذا (expert_ونڈو) {
            ثابت bool my_expert = (نشانہ_dev == octx->ctx->mdev.idx);
            إذا (++نشانہ_dev == octx->ctx->mdev.گنتی) {
                نشانہ_dev = 0;
            }
            إذا (!my_expert) استمر;
        } وإلا إذا (mdev_ونڈو) {
            ثابت bool can_ونڈو = (uint32_t) cne1 >= octx->ctx->mdev.گنتی;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو((uint32_t) m_padded, can_ونڈو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            m_شروع = (صحيح) range.شروع;
            m_سرا   = (صحيح) (range.شروع + range.گنتی);
        }
        إذا (m_شروع >= m_سرا) استمر;

        صحيح ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (عائم*) dst->انکڑے, (عائم*) act->انکڑے,
                                   src0->انکڑے + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (صحيح) src0->nb[1], (صحيح) src0->type,
                                   kparams->tile_ناپ, kparams->aligned_tile_ناپ,
                                   میٹرکس_صفاں, cur_a, mmctx->mapping_stride,
                                   m_شروع, m_سرا, (صحيح) octx->n_دھاگے);
        إذا (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            أرجع HTP_حالت_NO_SUPPORT;
        }
    }

    أرجع HTP_حالت_OK;
}

ساكن صحيح hvx_mm_matmul_id(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx,
    work_قطار_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ثابت uint32_t src0_صف_ناپ_padded = mmctx->src0_صف_ناپ_padded;
    ثابت uint32_t act_nrows            = mmctx->act_nrows;

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت بنية htp_tensor * محدود ids = octx->src[2];
    ثابت size_t src0_صف_ناپ = nb01;

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (ne10 + qk - 1) / qk;
    ثابت uint32_t کل_nb = act_nrows * nb;

    work_قطار_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    إذا (act_nrows < octx->n_دھاگے) {
        n_quant_tasks = MIN(کل_nb, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_پہلا = (کل_nb * ith) / n_quant_tasks;
            uint32_t ib_آخری  = (کل_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_پہلا[ith] = ib_پہلا;
            mmctx->quant_ib_آخری[ith]  = ib_آخری;
            mmctx->quant_r[ith]        = ib_پہلا / nb;
            mmctx->quant_c[ith]        = ib_پہلا % nb;
        }
    } وإلا {
        n_quant_tasks = MIN(act_nrows, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_صف_func(src0->type, act->type);
    }
    size_t act_صف_ناپ = htp_mm_tiled_act_صف_ناپ(src0->type, ne10);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_دھاگے,
                                 0, src0_صف_ناپ, act_صف_ناپ, 0, kparams->n_prefetch, true, false);

    ثابت size_t vtcm_ناپ = L.کل_بائٹاں;

    FARF(بلند_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_بائٹاں, L.act_بائٹاں, L.dst_بائٹاں, vtcm_ناپ);

    FARF(بلند_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->انکڑے,
         act->انکڑے, dst->انکڑے);

    // Make sure the reserved vtcm size is sufficient
    إذا (octx->ctx->vtcm_ناپ < vtcm_ناپ) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_ناپ, vtcm_ناپ);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت ادھار = (uint8_t *) octx->ctx->vtcm_ادھار;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_src0);
    mmctx->vtcm_bias     = عدم;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_صف_ناپ_padded;
    mmctx->vtcm_act_stride     = act_صف_ناپ;
    mmctx->vtcm_act_raw_stride = hex_گیڑا_اتے(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_ناپ_فی_thread = fastdiv(L.src0_بائٹاں, &octx->n_دھاگے_div);
    mmctx->vtcm_act_ناپ_فی_thread  = L.act_بائٹاں;
    mmctx->vtcm_dst_ناپ_فی_thread  = fastdiv(L.dst_بائٹاں, &octx->n_دھاگے_div);

    mmctx->cur_m_شروع = 0;
    mmctx->cur_m_صفاں  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_صفاں_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_قطار_لگاتار(octx->ctx->work_قطار, quant_task_func, mmctx, n_quant_tasks);

    work_قطار_لگاتار(octx->ctx->work_قطار, hvx_mmid_task_func, mmctx, octx->n_دھاگے);

    أرجع HTP_حالت_OK;
}

ساكن صحيح hmx_mm_op_matmul_id_nx(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx
) {
    ثابت uint32_t *                میٹرکس_صف_counts = mmctx->میٹرکس_صف_counts;
    ثابت بنية mmid_صف_mapping * میٹرکس_صفاں       = mmctx->میٹرکس_صفاں;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];
    ثابت صحيح n_as = src0->ne[2];

    bool mdev_ونڈو = (octx->ctx->mdev.گنتی > 1);
    لكل (uint32_t p = 0; p < n_weights && mdev_ونڈو; ++p) {
        ثابت بنية htp_tensor * محدود dst = octx->dsts[p];
        mdev_ونڈو = htp_tensor_can_صف_حصے_کرو(dst, حجم(عائم));
    }
    إذا (octx->ctx->mdev.گنتی > 1 && !mdev_ونڈو && octx->ctx->mdev.idx > 0) {
        أرجع HTP_حالت_OK;
    }
    uint32_t n_سرگرم = 0;
    إذا (mdev_ونڈو) {
        لكل (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            إذا (میٹرکس_صف_counts[a] > 0) n_سرگرم++;
        }
    }
    ثابت bool expert_ونڈو = mdev_ونڈو && (n_سرگرم >= octx->ctx->mdev.گنتی);

    uint32_t نشانہ_dev = 0;
    لكل (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ثابت int32_t cne1 = میٹرکس_صف_counts[cur_a];
        إذا (cne1 == 0) استمر;

        ثابت صحيح m_padded = hex_align_اتے(cne1, 32);
        صحيح m_شروع = 0, m_سرا = m_padded;
        إذا (expert_ونڈو) {
            ثابت bool my_expert = (نشانہ_dev == octx->ctx->mdev.idx);
            إذا (++نشانہ_dev == octx->ctx->mdev.گنتی) {
                نشانہ_dev = 0;
            }
            إذا (!my_expert) استمر;
        } وإلا إذا (mdev_ونڈو) {
            ثابت bool can_ونڈو = (uint32_t) cne1 >= octx->ctx->mdev.گنتی;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو((uint32_t) m_padded, can_ونڈو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            m_شروع = (صحيح) range.شروع;
            m_سرا   = (صحيح) (range.شروع + range.گنتی);
        }
        إذا (m_شروع >= m_سرا) استمر;

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * محدود src_w = octx->src[p];
            ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
            إذا (!src_w || !dst) استمر;

            صحيح ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (عائم*) dst->انکڑے, (عائم*) act->انکڑے,
                                       src_w->انکڑے + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (صحيح) src_w->nb[1], (صحيح) src_w->type,
                                       kparams->tile_ناپ, kparams->aligned_tile_ناپ,
                                       میٹرکس_صفاں, cur_a, mmctx->mapping_stride,
                                       m_شروع, m_سرا, (صحيح) octx->n_دھاگے);
            إذا (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                أرجع HTP_حالت_NO_SUPPORT;
            }
        }
    }

    أرجع HTP_حالت_OK;
}

ساكن صحيح hvx_mm_matmul_id_nx(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx,
    work_قطار_func_t hvx_mmid_task_func
) {
    ثابت uint32_t src0_صف_ناپ_padded = mmctx->src0_صف_ناپ_padded;
    ثابت uint32_t act_nrows            = mmctx->act_nrows;

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];
    ثابت بنية htp_tensor * محدود ids  = octx->src[n_weights + 1];
    ثابت size_t src0_صف_ناپ = src0->nb[1];

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ثابت uint32_t کل_nb = act_nrows * nb;

    work_قطار_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    إذا (act_nrows < octx->n_دھاگے) {
        n_quant_tasks = MIN(کل_nb, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_پہلا = (کل_nb * ith) / n_quant_tasks;
            uint32_t ib_آخری  = (کل_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_پہلا[ith] = ib_پہلا;
            mmctx->quant_ib_آخری[ith]  = ib_آخری;
            mmctx->quant_r[ith]        = ib_پہلا / nb;
            mmctx->quant_c[ith]        = ib_پہلا % nb;
        }
    } وإلا {
        n_quant_tasks = MIN(act_nrows, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_صف_func(src0->type, act->type);
    }
    size_t act_صف_ناپ = htp_mm_tiled_act_صف_ناپ(src0->type, act->ne[0]);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_دھاگے,
                                 0, src0_صف_ناپ, act_صف_ناپ, 0, kparams->n_prefetch, true, false);

    ثابت size_t vtcm_ناپ = L.کل_بائٹاں;

    إذا (octx->ctx->vtcm_ناپ < vtcm_ناپ) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_ناپ, vtcm_ناپ);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت ادھار = (uint8_t *) octx->ctx->vtcm_ادھار;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_صف_ناپ;
    mmctx->vtcm_act_raw_stride = hex_گیڑا_اتے(act->ne[0] * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_ناپ_فی_thread = fastdiv(L.src0_بائٹاں, &octx->n_دھاگے_div);
    mmctx->vtcm_act_ناپ_فی_thread  = L.act_بائٹاں;
    mmctx->vtcm_dst_ناپ_فی_thread  = fastdiv(L.dst_بائٹاں, &octx->n_دھاگے_div);

    mmctx->cur_m_شروع = 0;
    mmctx->cur_m_صفاں  = act_nrows;

    FARF(بلند_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.کل_بائٹاں, octx->ctx->vtcm_ناپ, octx->n_دھاگے);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_صفاں_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_قطار_لگاتار(octx->ctx->work_قطار, quant_task_func, mmctx, n_quant_tasks);

    work_قطار_لگاتار(octx->ctx->work_قطار, hvx_mmid_task_func, mmctx, octx->n_دھاگے);

    أرجع HTP_حالت_OK;
}

ساكن مضمن فراغ جانچ_expert_ids_n(
    ثابت بنية htp_tensor * ids,
    ثابت uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    بنية mmid_صف_mapping * میٹرکس_صفاں,
    uint32_t mapping_stride
) {
    ثابت size_t ids_nb1 = ids->nb[1];
    ثابت uint8_t * ids_انکڑے = (ثابت uint8_t *) ids->انکڑے;

    لكل (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ثابت int32_t * صف_ptr = (ثابت int32_t *) (ids_انکڑے + iid1 * ids_nb1);
        لكل (uint32_t id = 0; id < n_ids; ++id) {
            ثابت int32_t i02 = صف_ptr[id];
            إذا (i02 < 0) {
                استمر;
            }
            أكد(i02 < n_as);

            إذا (میٹرکس_صفاں) {
                میٹرکس_صفاں[i02 * mapping_stride + counts[i02]] = (بنية mmid_صف_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

ساكن مضمن فراغ جانچ_expert_ids(
    ثابت بنية htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    بنية mmid_صف_mapping * میٹرکس_صفاں,
    uint32_t mapping_stride
) {
    ثابت size_t ids_nb0 = ids->nb[0];

    إذا (ids_nb0 == 4) {
        بدل (n_ids) {
            حالة 8:  جانچ_expert_ids_n(ids, 8,     n_as, counts, میٹرکس_صفاں, mapping_stride); اكسر;
            حالة 4:  جانچ_expert_ids_n(ids, 4,     n_as, counts, میٹرکس_صفاں, mapping_stride); اكسر;
            حالة 2:  جانچ_expert_ids_n(ids, 2,     n_as, counts, میٹرکس_صفاں, mapping_stride); اكسر;
            افتراضي: جانچ_expert_ids_n(ids, n_ids, n_as, counts, میٹرکس_صفاں, mapping_stride); اكسر;
        }
    } وإلا {
        // Strided fallback
        ثابت size_t ids_nb1 = ids->nb[1];
        ثابت uint8_t * ids_انکڑے = (ثابت uint8_t *) ids->انکڑے;
        لكل (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ثابت int32_t * صف_ptr = (ثابت int32_t *) (ids_انکڑے + iid1 * ids_nb1);
            لكل (uint32_t id = 0; id < n_ids; ++id) {
                ثابت int32_t i02 = *(ثابت int32_t *) ((ثابت uint8_t *) صف_ptr + id * ids_nb0);
                إذا (i02 < 0) {
                    استمر;
                }
                أكد(i02 < n_as);

                إذا (میٹرکس_صفاں) {
                    میٹرکس_صفاں[i02 * mapping_stride + counts[i02]] = (بنية mmid_صف_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

صحيح op_matmul_id(بنية htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;

    ثابت صحيح حالت = htp_mm_init_context(octx, kparams);
    إذا (حالت != HTP_حالت_OK) {
        أرجع حالت;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ثابت بنية htp_tensor * محدود ids = octx->src[2];
    إذا (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت size_t src0_صف_ناپ = nb01;
    ثابت size_t dst_صف_ناپ  = nb1;

    ثابت size_t src0_صف_ناپ_padded = hex_گیڑا_اتے(src0_صف_ناپ, 128);

    ثابت uint32_t src0_nrows = ne01;  // per expert
    ثابت uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ثابت صحيح n_ids = ids->ne[0];  // n_expert_used
    ثابت صحيح n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_ادھار;
    uint32_t   mapping_stride    = 1;
    uint32_t * میٹرکس_صف_counts = (uint32_t *) mapping_buf;
    بنية mmid_صف_mapping * میٹرکس_صفاں = عدم;

    إذا (act_nrows > 1) {
        ثابت size_t میٹرکس_صف_counts_ناپ = n_as * حجم(uint32_t);
        أكد(octx->ctx->ddr_spad_ناپ >= میٹرکس_صف_counts_ناپ);

        hex_l2fetch_بلاک((ثابت فراغ *) ids->انکڑے, ids->ne[1] * ids->nb[1]);

        memset(میٹرکس_صف_counts, 0, میٹرکس_صف_counts_ناپ);
        جانچ_expert_ids(ids, n_ids, n_as, میٹرکس_صف_counts, عدم, 0);

        uint32_t max_گنتی = hvx_reduce_max_i32((ثابت uint8_t *) میٹرکس_صف_counts, n_as);
        mapping_stride = max_گنتی > 0 ? max_گنتی : 1;

        size_t میٹرکس_صف_map_ناپ  = n_as * mapping_stride * حجم(بنية mmid_صف_mapping);
        ثابت size_t کل_map_ناپ = میٹرکس_صف_counts_ناپ + میٹرکس_صف_map_ناپ;

        إذا (کل_map_ناپ > octx->ctx->ddr_spad_ناپ) {
            mapping_buf = memalign(128, کل_map_ناپ);
            إذا (!mapping_buf) {
                أرجع HTP_حالت_اندرونی_ERR;
            }
        }

        میٹرکس_صف_counts = (uint32_t *) mapping_buf;
        میٹرکس_صفاں       = (بنية mmid_صف_mapping *) (mapping_buf + میٹرکس_صف_counts_ناپ);

        memset(میٹرکس_صف_counts, 0, n_as * حجم(uint32_t));
        جانچ_expert_ids(ids, n_ids, n_as, میٹرکس_صف_counts, میٹرکس_صفاں, mapping_stride);
    }

    mmctx->میٹرکس_صف_counts    = میٹرکس_صف_counts;
    mmctx->میٹرکس_صفاں          = میٹرکس_صفاں;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_صف_ناپ_padded = src0_صف_ناپ_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_شروع          = 0;
    mmctx->cur_m_صفاں           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    صحيح s;
    إذا (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } وإلا {
        uint32_t src0_صف_شروع = 0;
        uint32_t src0_صف_سرا   = src0_nrows;
        إذا (octx->ctx->mdev.گنتی > 1) {
            ثابت bool can_ونڈو = htp_tensor_can_صف_حصے_کرو(dst, حجم(عائم));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(src0_nrows, can_ونڈو ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            src0_صف_شروع = range.شروع;
            src0_صف_سرا   = range.شروع + range.گنتی;
        }

        إذا (src0_صف_شروع >= src0_صف_سرا) {
            إذا (mapping_buf != octx->ctx->ddr_spad_ادھار) {
                حرر(mapping_buf);
            }
            أرجع HTP_حالت_OK;
        }

        ثابت uint32_t nrows = src0_صف_سرا - src0_صف_شروع;
        mmctx->src0_صف_شروع = src0_صف_شروع;
        mmctx->src0_صف_سرا   = src0_صف_سرا;

        mmctx->src0_nrows_فی_thread = fastdiv(nrows + octx->n_دھاگے - 1, &octx->n_دھاگے_div);
        mmctx->src0_nrows_فی_thread = hex_گیڑا_اتے(mmctx->src0_nrows_فی_thread, 32);

        إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } وإلا {
            s = HTP_حالت_NO_SUPPORT;
        }
    }

    إذا (mapping_buf != octx->ctx->ddr_spad_ادھار) {
        حرر(mapping_buf);
    }

    أرجع s;
}

صحيح op_matmul_id_nx(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;

    ثابت صحيح حالت = htp_mm_init_context(octx, kparams);
    إذا (حالت != HTP_حالت_OK) {
        أرجع حالت;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];
    ثابت بنية htp_tensor * محدود ids  = octx->src[n_weights + 1];
    إذا (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }
    لكل (uint32_t p = 0; p < n_weights; p++) {
        إذا (htp_tensor_is_extended(octx->dsts[p])) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    ثابت size_t src0_صف_ناپ = src0->nb[1];
    ثابت size_t src0_صف_ناپ_padded = hex_گیڑا_اتے(src0_صف_ناپ, 128);

    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ثابت صحيح n_ids = ids->ne[0];
    ثابت صحيح n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_ادھار;
    uint32_t   mapping_stride    = 1;
    uint32_t * میٹرکس_صف_counts = (uint32_t *) mapping_buf;
    بنية mmid_صف_mapping * میٹرکس_صفاں = عدم;

    إذا (act_nrows > 1) {
        ثابت size_t میٹرکس_صف_counts_ناپ = n_as * حجم(uint32_t);
        أكد(octx->ctx->ddr_spad_ناپ >= میٹرکس_صف_counts_ناپ);

        hex_l2fetch_بلاک((ثابت فراغ *) ids->انکڑے, ids->ne[1] * ids->nb[1]);

        memset(میٹرکس_صف_counts, 0, میٹرکس_صف_counts_ناپ);
        جانچ_expert_ids(ids, n_ids, n_as, میٹرکس_صف_counts, عدم, 0);

        uint32_t max_گنتی = hvx_reduce_max_i32((ثابت uint8_t *) میٹرکس_صف_counts, n_as);
        mapping_stride = max_گنتی > 0 ? max_گنتی : 1;

        size_t میٹرکس_صف_map_ناپ  = n_as * mapping_stride * حجم(بنية mmid_صف_mapping);
        ثابت size_t کل_map_ناپ = میٹرکس_صف_counts_ناپ + میٹرکس_صف_map_ناپ;

        إذا (کل_map_ناپ > octx->ctx->ddr_spad_ناپ) {
            mapping_buf = memalign(128, کل_map_ناپ);
            إذا (!mapping_buf) {
                أرجع HTP_حالت_اندرونی_ERR;
            }
        }

        میٹرکس_صف_counts = (uint32_t *) mapping_buf;
        میٹرکس_صفاں       = (بنية mmid_صف_mapping *) (mapping_buf + میٹرکس_صف_counts_ناپ);

        memset(میٹرکس_صف_counts, 0, n_as * حجم(uint32_t));
        جانچ_expert_ids(ids, n_ids, n_as, میٹرکس_صف_counts, میٹرکس_صفاں, mapping_stride);
    }

    mmctx->میٹرکس_صف_counts    = میٹرکس_صف_counts;
    mmctx->میٹرکس_صفاں          = میٹرکس_صفاں;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_صف_ناپ_padded = src0_صف_ناپ_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_شروع          = 0;
    mmctx->cur_m_صفاں           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    صحيح s;
    إذا (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } وإلا {
        إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } وإلا {
            s = HTP_حالت_NO_SUPPORT;
        }
    }

    إذا (mapping_buf != octx->ctx->ddr_spad_ادھار) {
        حرر(mapping_buf);
    }

    أرجع s;
}
ساكن صحيح op_matmul_nx_impl(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    ثابت صحيح حالت = htp_mm_init_context(octx, kparams);
    إذا (حالت != HTP_حالت_OK) {
        أرجع حالت;
    }

    إذا (kparams->n_hmx) {
        أرجع hmx_mm_nx_2d_f32(octx, kparams);
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت uint32_t n_weights = kparams->n_weights;

    ثابت بنية htp_tensor * محدود src0 = octx->src[0]; // first weight
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights]; // activation x

    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_شروع = 0;
    mmctx->cur_m_صفاں  = act_nrows;

    ثابت size_t src0_صف_ناپ = src0->nb[1];

    إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ثابت uint32_t کل_nb = act_nrows * nb;

    کامہ_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    إذا (act_nrows < octx->n_دھاگے) {
        n_quant_tasks = MIN(کل_nb, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_پہلا = (کل_nb * ith) / n_quant_tasks;
            uint32_t ib_آخری  = (کل_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_پہلا[ith] = ib_پہلا;
            mmctx->quant_ib_آخری[ith]  = ib_آخری;
            mmctx->quant_r[ith]        = ib_پہلا / nb;
            mmctx->quant_c[ith]        = ib_پہلا % nb;
        }
    } وإلا {
        n_quant_tasks = MIN(act_nrows, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_صف_func(src0->type, act->type);
    }

    ثابت size_t act_صف_ناپ = htp_mm_tiled_act_صف_ناپ(src0->type, act->ne[0]);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_دھاگے,
                                 0, src0_صف_ناپ, act_صف_ناپ, 0, kparams->n_prefetch, false, true);

    ثابت size_t vtcm_ناپ = L.کل_بائٹاں;

    إذا (octx->ctx->vtcm_ناپ < vtcm_ناپ) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_ناپ, vtcm_ناپ);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت ادھار = (uint8_t *) octx->ctx->vtcm_ادھار;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_صف_ناپ;
    mmctx->vtcm_act_raw_stride = hex_گیڑا_اتے(act->ne[0] * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_ناپ_فی_thread = fastdiv(L.src0_بائٹاں, &octx->n_دھاگے_div);
    mmctx->vtcm_act_ناپ_فی_thread  = L.act_بائٹاں;
    mmctx->vtcm_dst_ناپ_فی_thread  = fastdiv(L.dst_بائٹاں, &octx->n_دھاگے_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_صفاں_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_قطار_لگاتار(octx->ctx->work_قطار, quant_task_func, mmctx, n_quant_tasks);

    work_قطار_لگاتار(octx->ctx->work_قطار, hvx_mm_nx_2d_tiled, mmctx, octx->n_دھاگے);

    أرجع HTP_حالت_OK;
}

صحيح op_matmul_nx(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    إذا (kparams->collapse) {
        ثابت uint32_t n_weights = kparams->n_weights;
        ثابت بنية htp_tensor * act = octx->src[n_weights];
        ثابت uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        بنية htp_tensor act_collapsed;
        بنية htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        ثابت بنية htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_صفاں(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        لكل (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ثابت بنية htp_tensor * d = octx->dsts[p];
            ثابت uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_صفاں(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        ثابت صحيح حالت = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        لكل (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        أرجع حالت;
    }

    أرجع op_matmul_nx_impl(octx);
}
