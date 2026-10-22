#لارښوونه clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

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

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "matmul-ops.h"
#ضمن "htp-vtcm.h"

تعريفنوع بنية {
    عائم        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_بایټونه;
    dma_addr_t    act_dma_addr;
    dma_addr_t    وزن;
    dma_کتار *   وزن_dma;
    صحيح           m;
    صحيح           k;
    صحيح           n;
    صحيح           act_stride;
    uint32_t      act_elem_اندازه;  // 4=F32 src1, 2=F16 src1
    صحيح           وزن_stride;
    صحيح           وزن_type;
    صحيح           dst_stride;
    صحيح           dst_ستون_شمېر;
    uint32_t      bias_stride;
    صحيح           tile_اندازه;
    صحيح           aligned_tile_اندازه;
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
    بنية fastdiv_ارزښتونه div_r2;
    بنية fastdiv_ارزښتونه div_r3;
} hmx_mm_batched_params_t;

ساكن bool htp_matmul_has_extended_وزن(ثابت بنية htp_ops_context * octx, uint32_t n_weights) {
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
         ثابت فراغ * محدود vy, uint32_t valid_لیکې,
         ثابت عائم * محدود sz);

    فراغ (*vec_dot_32x2)(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1,
         ثابت فراغ * محدود vx,
         ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_لیکې,
         ثابت عائم * محدود sz0, ثابت عائم * محدود sz1);

    // Precomputed values
    uint32_t src0_nrows_فی_thread;
    uint32_t src0_لیکه_پیل;
    uint32_t src0_لیکه_پای_ټکی;
    uint32_t src0_لیکه_اندازه_padded;
    uint32_t act_nrows;
    uint32_t cur_m_پیل;
    uint32_t cur_m_لیکې;

    بنية fastdiv_ارزښتونه mm_div_ne12_ne1;
    بنية fastdiv_ارزښتونه mm_div_ne1;
    بنية fastdiv_ارزښتونه mm_div_r2;
    بنية fastdiv_ارزښتونه mm_div_r3;
    بنية fastdiv_ارزښتونه mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_لومړی[WORK_کتار_MAX_N_تارونه];
    uint32_t          quant_ib_وروستی[WORK_کتار_MAX_N_تارونه];
    uint32_t          quant_r[WORK_کتار_MAX_N_تارونه];
    uint32_t          quant_c[WORK_کتار_MAX_N_تارونه];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_لیکې_فی_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ثابت uint32_t * ماتریکس_لیکه_counts;
    ثابت بنية mmid_لیکه_mapping * ماتریکس_لیکې;
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
    uint32_t vtcm_src0_اندازه_فی_thread;
    uint32_t vtcm_act_اندازه_فی_thread;
    uint32_t vtcm_dst_اندازه_فی_thread;
};

ساكن صحيح htp_mm_init_context(
    بنية htp_ops_context * octx,
    ثابت بنية htp_mm_kernel_params * kparams
) {
    إذا (!htp_ops_context_سېټ_n_تارونه(octx, (uint32_t) kparams->n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    إذا (kparams->n_hmx) {
        إذا (kparams->n_act_تارونه <= 0 || kparams->n_act_تارونه > (int32_t) octx->n_تارونه) {
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

#تعریف_کړه htp_matmul_tensors_preamble                                 \
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

#تعریف_کړه htp_matmul_preamble                                         \
    بنية htp_mm_context * mmctx  = معلومات;                          \
    بنية htp_ops_context * octx  = mmctx->octx;                   \
    dma_کتار * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_فی_thread = mmctx->src0_nrows_فی_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#ضمن "hvx-mm-kernels-tiled.h"
#ضمن "hvx-mm-kernels-float.h"
#ضمن "hmx-mm-kernels-tiled.h"

ساكن فراغ hvx_mm_2d_tiled(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_لیکه_پای_ټکی - mmctx->src0_لیکه_پیل;
    ثابت uint32_t act_nrows = mmctx->cur_m_لیکې ? mmctx->cur_m_لیکې : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_پیل = mmctx->cur_m_پیل;

    ثابت uint32_t src0_پیل_لیکه  = mmctx->src0_لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه    = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, mmctx->src0_لیکه_پای_ټکی);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_لیکه_اندازه  = nb1;
    ثابت size_t act_لیکه_اندازه = nb11;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;
    ثابت size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * محدود vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_اندازه_فی_thread  * ith;
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * محدود act_معلومات = mmctx->vtcm_act;

    ثابت dma_addr_t src0_لیکه = src0->معلومات;

    ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src0->type);
    ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_لیکه_stride = n_k_tiles_w * tile_اندازه;
    uint32_t tile_لیکه_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

    uint32_t ct_پیل = src0_پیل_لیکه / 32;
    uint32_t ct_پای_ټکی   = (src0_پای_ټکی_لیکه + 31) / 32;

    uint32_t ټېل_ct = ct_پیل;
    إذا (src0_پیل_لیکه < src0_پای_ټکی_لیکه) {
        لكل (uint32_t d = 0; d < n_prefetch && ټېل_ct < ct_پای_ټکی; d++, ټېل_ct++) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + d * tile_لیکه_transfer_اندازه_aligned,
                           src0_لیکه + ټېل_ct * tile_لیکه_stride), aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }
    }

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    لكل (uint32_t ct = ct_پیل; ct < ct_پای_ټکی; ct++) {
        ثابت uint8_t * w_tile = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

        صحيح valid_لیکې = (صحيح)ne0 - (صحيح)(ct * 32);
        valid_لیکې = MIN(32, MAX(0, valid_لیکې));

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
            ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_معلومات + (ir1+0) * act_stride);
            ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_معلومات + (ir1+1) * act_stride);
            عائم * محدود dst_row0 = (عائم *) (dst->معلومات + ((cur_m_پیل + ir1+0) * dst_لیکه_اندازه));
            عائم * محدود dst_row1 = (عائم *) (dst->معلومات + ((cur_m_پیل + ir1+1) * dst_لیکه_اندازه));

            عائم * dst_ptr0 = &dst_row0[ct * 32];
            عائم * dst_ptr1 = &dst_row1[ct * 32];

            ثابت عائم * src2_ptr0 = عدم;
            ثابت عائم * src2_ptr1 = عدم;
            إذا (src2) {
                ثابت عائم * محدود src2_row0 = (ثابت عائم *) ((ثابت uint8_t *) src2->معلومات + ((cur_m_پیل + ir1+0) * src2_stride));
                ثابت عائم * محدود src2_row1 = (ثابت عائم *) ((ثابت uint8_t *) src2->معلومات + ((cur_m_پیل + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_لیکې, src2_ptr0, src2_ptr1);
        }

        لكل (; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_معلومات + ir1 * act_stride);
            عائم * محدود dst_لیکه          = (عائم *) (dst->معلومات + ((cur_m_پیل + ir1) * dst_لیکه_اندازه));
            عائم * dst_ptr = &dst_لیکه[ct * 32];

            ثابت عائم * src2_ptr = عدم;
            إذا (src2) {
                ثابت عائم * محدود src2_لیکه = (ثابت عائم *) ((ثابت uint8_t *) src2->معلومات + ((cur_m_پیل + ir1) * src2_stride));
                src2_ptr = &src2_لیکه[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_لیکې, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        إذا (ټېل_ct < ct_پای_ټکی) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(w_tile, src0_لیکه + ټېل_ct * tile_لیکه_stride),
                           aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
            ټېل_ct++;
        }
    }
}

ساكن فراغ hvx_mv_2d_tiled(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_لیکه_پای_ټکی - mmctx->src0_لیکه_پیل;

    ثابت uint32_t src0_پیل_لیکه  = mmctx->src0_لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه    = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, mmctx->src0_لیکه_پای_ټکی);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_لیکه_اندازه  = nb1;
    ثابت size_t act_لیکه_اندازه = nb11;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_اندازه_فی_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * act_معلومات = mmctx->vtcm_act;

    عائم * tmp = (عائم *) vtcm_dst_ptr;

    ثابت dma_addr_t src0_لیکه = src0->معلومات;

    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_معلومات;
    عائم * محدود dst_ستون          = (عائم *) dst->معلومات;

    ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src0->type);
    ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_لیکه_stride = n_k_tiles_w * tile_اندازه;
    uint32_t tile_لیکه_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

    uint32_t ct_پیل = src0_پیل_لیکه / 32;
    uint32_t ct_پای_ټکی   = (src0_پای_ټکی_لیکه + 31) / 32;

    uint32_t ټېل_ct = ct_پیل;
    إذا (src0_پیل_لیکه < src0_پای_ټکی_لیکه) {
        إذا (src2) {
            عائم * vtcm_bias_ptr = (عائم *) mmctx->vtcm_bias + src0_پیل_لیکه;
            ثابت dma_addr_t src2_addr = src2->معلومات + src0_پیل_لیکه * حجم(عائم);
            صحيح ټوټه_اندازه = (صحيح)MIN(src0_پای_ټکی_لیکه, ne0) - (صحيح)src0_پیل_لیکه;
            إذا (ټوټه_اندازه > 0) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_bias_ptr, src2_addr),
                               ټوټه_اندازه * حجم(عائم), ټوټه_اندازه * حجم(عائم), ټوټه_اندازه * حجم(عائم), 1);
                dma_کتار_راوباسه_nowait(dma_q);
            }
        }
        لكل (uint32_t d = 0; d < n_prefetch && ټېل_ct < ct_پای_ټکی; d++, ټېل_ct++) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + d * tile_لیکه_transfer_اندازه_aligned,
                           src0_لیکه + ټېل_ct * tile_لیکه_stride), aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }
    }

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    لكل (uint32_t ct = ct_پیل; ct < ct_پای_ټکی; ct++) {
        ثابت uint8_t * w_tile = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

        عائم * dst_ptr = &tmp[ct * 32 - src0_پیل_لیکه];
        صحيح valid_لیکې = (صحيح)ne0 - (صحيح)(ct * 32);
        valid_لیکې = MIN(32, MAX(0, valid_لیکې));

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_لیکې, عدم);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        إذا (ټېل_ct < ct_پای_ټکی) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(w_tile, src0_لیکه + ټېل_ct * tile_لیکه_stride),
                           aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
            ټېل_ct++;
        }
    }

    صحيح copy_cnt = (صحيح)MIN(src0_پای_ټکی_لیکه, ne0) - (صحيح)src0_پیل_لیکه;
    إذا (copy_cnt > 0) {
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ct_پای_ټکی);
        إذا (src2) {
            hvx_جمع_کړه_f32_uuu((uint8_t *) &dst_ستون[src0_پیل_لیکه],
                            (ثابت uint8_t *) tmp,
                            (ثابت uint8_t *) ((ثابت عائم *) mmctx->vtcm_bias + src0_پیل_لیکه),
                            copy_cnt);
        } وإلا {
            hvx_copy_f32_ua((uint8_t *) &dst_ستون[src0_پیل_لیکه], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_پای_ټکی);
    }
}

ساكن فراغ hvx_mm_nx_2d_tiled(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_mm_context * mmctx = معلومات;
    بنية htp_ops_context * octx = mmctx->octx;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;

    ثابت بنية htp_tensor * محدود act = octx->src[n_weights]; /* x */
    ثابت uint32_t ne10 = act->ne[0];
    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_وزن_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * محدود act_معلومات        = mmctx->vtcm_act;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    لكل (uint32_t widx = 0; widx < n_weights; widx++) {
        ثابت بنية htp_tensor * محدود src_w = octx->src[widx];
        ثابت بنية htp_tensor * محدود dst   = octx->dsts[widx];
        إذا (!src_w || !dst) استمر;
        dma_کتار * dma_q = octx->ctx->dma[ith];

        ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src_w->type);
        ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src_w->type);
        ثابت uint32_t ne00 = src_w->ne[0];
        ثابت uint32_t ne01 = src_w->ne[1];
        ثابت size_t dst_لیکه_اندازه = dst->nb[1];
        ثابت dma_addr_t src_w_لیکه = src_w->معلومات;

        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ثابت uint32_t tile_لیکه_stride = n_k_tiles_w * tile_اندازه;
        ثابت uint32_t tile_لیکه_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

        uint32_t src0_پیل_لیکه = 0;
        uint32_t src0_پای_ټکی_لیکه   = ne01;
        إذا (octx->ctx->mdev.شمېر > 1) {
            ثابت bool can_وېشل = htp_tensor_can_لیکه_برخه_کول(dst, حجم(عائم));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(ne01, can_وېشل ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
            src0_پیل_لیکه = range.پیل;
            src0_پای_ټکی_لیکه   = range.پیل + range.شمېر;
        }

        ثابت uint32_t nrows = src0_پای_ټکی_لیکه - src0_پیل_لیکه;
        uint32_t src0_nrows_فی_thread = fastdiv(nrows + nth - 1, &octx->n_تارونه_div);
        src0_nrows_فی_thread = hex_پړاو_پورته(src0_nrows_فی_thread, 32);

        ثابت uint32_t پیل_لیکه = src0_پیل_لیکه + src0_nrows_فی_thread * ith;
        ثابت uint32_t پای_ټکی_لیکه   = MIN(پیل_لیکه + src0_nrows_فی_thread, src0_پای_ټکی_لیکه);
        إذا (پیل_لیکه >= پای_ټکی_لیکه) استمر;

        uint32_t ct_پیل = پیل_لیکه / 32;
        uint32_t ct_پای_ټکی   = (پای_ټکی_لیکه + 31) / 32;

        uint32_t ټېل_ct = ct_پیل;
        لكل (uint32_t d = 0; d < n_prefetch && ټېل_ct < ct_پای_ټکی; d++, ټېل_ct++) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_وزن_ptr + d * tile_لیکه_transfer_اندازه_aligned,
                            src_w_لیکه + ټېل_ct * tile_لیکه_stride), aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_پیل; ct < ct_پای_ټکی; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_کتار_راوباسه(dma_q).dst;
            صحيح valid_لیکې = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
            valid_لیکې = MIN(32, MAX(0, valid_لیکې));

            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
                ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_معلومات + (ir1+0) * act_stride);
                ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_معلومات + (ir1+1) * act_stride);

                عائم * محدود dst_row0 = (عائم *) (dst->معلومات + ((ir1+0) * dst_لیکه_اندازه));
                عائم * محدود dst_row1 = (عائم *) (dst->معلومات + ((ir1+1) * dst_لیکه_اندازه));
                عائم * dst_ptr0 = &dst_row0[ct * 32];
                عائم * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_لیکې, عدم, عدم);
            }

            لكل (; ir1 < act_nrows; ++ir1) {
                ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_معلومات + ir1 * act_stride);
                عائم * محدود dst_لیکه = (عائم *) (dst->معلومات + (ir1 * dst_لیکه_اندازه));
                عائم * dst_ptr = &dst_لیکه[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_لیکې, عدم);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (ټېل_ct < ct_پای_ټکی) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(w_tile, src_w_لیکه + ټېل_ct * tile_لیکه_stride),
                                aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                ټېل_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mm_transfer_act_dma(
    بنية htp_ops_context * octx,
    ثابت بنية htp_mm_kernel_params * kparams,
    ثابت بنية htp_tensor * act,
    uint8_t * dst_بنسټ,
    size_t dst_لیکه_اندازه,
    uint32_t m_پیل,
    uint32_t m_لیکې
) {
    إذا (m_لیکې == 0) {
        أرجع;
    }

    dma_کتار * dma_q = octx->ctx->dma[0];
    ثابت uint32_t ne0 = act->ne[0];
    ثابت size_t elem_اندازه = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت size_t لیکه_بایټونه = ne0 * elem_اندازه;
    ثابت size_t act_nb1 = act->nb[1];
    ثابت dma_addr_t act_بنسټ = act->معلومات;

    ثابت bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    إذا (is_contiguous) {
        ثابت dma_addr_t src_addr = act_بنسټ + m_پیل * act_nb1;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_بنسټ, src_addr),
                       dst_لیکه_اندازه, act_nb1, لیکه_بایټونه, m_لیکې);
        dma_کتار_راوباسه(dma_q);
    } وإلا {
        ثابت uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ثابت bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        لكل (uint32_t ir = 0; ir < m_لیکې; ++ir) {
            ثابت uint32_t ir1 = m_پیل + ir;
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
            ثابت dma_addr_t لیکه_src = act_بنسټ + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * لیکه_dst = dst_بنسټ + ir * dst_لیکه_اندازه;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(لیکه_dst, لیکه_src),
                           dst_لیکه_اندازه, act_nb1, لیکه_بایټونه, 1);
            dma_کتار_راوباسه(dma_q);
        }
    }

    إذا (dst_لیکه_اندازه > لیکه_بایټونه) {
        بنية htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_پیل);
        ثابت uint32_t pad_elems = (dst_لیکه_اندازه - لیکه_بایټونه) / elem_اندازه;
        إذا (elem_اندازه == حجم(عائم)) {
            لكل (uint32_t ir = 0; ir < m_لیکې; ++ir) {
                hvx_splat_f32_u(dst_بنسټ + ir * dst_لیکه_اندازه + لیکه_بایټونه, 0.0f, pad_elems);
            }
        } وإلا {
            لكل (uint32_t ir = 0; ir < m_لیکې; ++ir) {
                hvx_splat_f16_u(dst_بنسټ + ir * dst_لیکه_اندازه + لیکه_بایټونه, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_پیل);
    }
}

#تعریف_کړه QUANTIZE_IMPL(name, log_name, kernel_fn, dst_لیکه_اندازه_بیان)                                        \
ساكن فراغ name(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {                                        \
    (فراغ) nth;                                                                                            \
    بنية htp_mm_context * mmctx = معلومات;                                                                  \
    بنية htp_ops_context * octx = mmctx->octx;                                                           \
    ثابت بنية htp_tensor * src = mmctx->act;                                                            \
    ثابت uint32_t ne0 = src->ne[0];                                                                       \
    ثابت uint32_t nrows = mmctx->cur_m_لیکې ? mmctx->cur_m_لیکې : mmctx->act_nrows;                       \
    ثابت uint32_t nrows_فی_thread = mmctx->n_quant_لیکې_فی_thread;                                      \
                                                                                                           \
    ثابت uint32_t ir_لومړی = nrows_فی_thread * ith;                                                      \
    إذا (ir_لومړی >= nrows) {                                                                               \
        أرجع;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_لومړی);                                        \
                                                                                                           \
    uint8_t * محدود dst = mmctx->vtcm_act;                                                              \
    ثابت uint32_t ir_وروستی = MIN(ir_لومړی + nrows_فی_thread, nrows);                                      \
    ثابت size_t raw_لیکه_اندازه = mmctx->vtcm_act_raw_stride;                                                \
    ثابت size_t dst_لیکه_اندازه = (dst_لیکه_اندازه_بیان);                                                       \
                                                                                                           \
    ثابت uint8_t * محدود src_معلومات = (ثابت uint8_t *) mmctx->vtcm_act_raw + (raw_لیکه_اندازه * ir_لومړی); \
    uint8_t * محدود dst_معلومات = (uint8_t *) dst + (dst_لیکه_اندازه * ir_لومړی);                             \
    kernel_fn(src_معلومات, dst_معلومات, عدم, ne0, ir_وروستی - ir_لومړی, raw_لیکه_اندازه, dst_لیکه_اندازه);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_لومړی);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled,     "quantize-f32-q8_0_tiled",     quantize_f32_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_لیکه_اندازه(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled,     "quantize-f32-q8_1_tiled",     quantize_f32_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_لیکه_اندازه(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_لیکه_اندازه(ne0))
QUANTIZE_IMPL(quantize_f32_f16_tiled,      "quantize-f32-f16_tiled",      quantize_f32_f16_tiled_kernel,      htp_mm_f16_tiled_لیکه_اندازه(ne0))
QUANTIZE_IMPL(quantize_f16_f16_tiled,      "quantize-f16-f16_tiled",      quantize_f16_f16_tiled_kernel,      htp_mm_f16_tiled_لیکه_اندازه(ne0))
QUANTIZE_IMPL(quantize_f32_f32,            "quantize-f32-f32",            quantize_f32_f32_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,            "quantize-f32-f16",            quantize_f32_f16_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,            "quantize-f16-f16",            quantize_f16_f16_kernel,            mmctx->vtcm_act_stride)

ساكن فراغ quantize_f32_q8_0_tiled_بلاک(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = معلومات;
    إذا (mmctx->quant_ib_لومړی[ith] >= mmctx->quant_ib_وروستی[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_لومړی[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_بلاک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_لومړی[ith],
        mmctx->quant_ib_وروستی[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_لیکه_اندازه(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_لومړی[ith]);
}

ساكن فراغ quantize_f32_q8_1_tiled_بلاک(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = معلومات;
    إذا (mmctx->quant_ib_لومړی[ith] >= mmctx->quant_ib_وروستی[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_لومړی[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_بلاک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_لومړی[ith],
        mmctx->quant_ib_وروستی[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_لیکه_اندازه(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_لومړی[ith]);
}

ساكن فراغ quantize_f32_q8_1_s16_tiled_بلاک(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = معلومات;
    إذا (mmctx->quant_ib_لومړی[ith] >= mmctx->quant_ib_وروستی[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_لومړی[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_بلاک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_لومړی[ith],
        mmctx->quant_ib_وروستی[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_لیکه_اندازه(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_لومړی[ith]);
}

ساكن فراغ quantize_f32_f16_tiled_بلاک(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = معلومات;
    إذا (mmctx->quant_ib_لومړی[ith] >= mmctx->quant_ib_وروستی[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_لومړی[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_بلاک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_لومړی[ith],
        mmctx->quant_ib_وروستی[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_لیکه_اندازه(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_لومړی[ith]);
}

ساكن فراغ quantize_f16_f16_tiled_بلاک(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = معلومات;
    إذا (mmctx->quant_ib_لومړی[ith] >= mmctx->quant_ib_وروستی[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_لومړی[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_بلاک_kernel(
        (ثابت __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_لومړی[ith],
        mmctx->quant_ib_وروستی[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_لیکه_اندازه(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_لومړی[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
ساكن مضمن کارګر_callback_t htp_mm_act_quant_لیکه_func(صحيح وزن_type, صحيح act_type) {
    إذا (وزن_type == HTP_TYPE_F16) {
        أرجع (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    إذا (وزن_type == HTP_TYPE_Q2_K) {
        أرجع quantize_f32_q8_1_s16_tiled;
    }
    أرجع htp_mm_وزن_has_دننه_واټن(وزن_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

ساكن مضمن کارګر_callback_t htp_mm_act_quant_بلاک_func(صحيح وزن_type, صحيح act_type) {
    إذا (وزن_type == HTP_TYPE_F16) {
        أرجع (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_بلاک : quantize_f32_f16_tiled_بلاک;
    }
    إذا (وزن_type == HTP_TYPE_Q2_K) {
        أرجع quantize_f32_q8_1_s16_tiled_بلاک;
    }
    أرجع htp_mm_وزن_has_دننه_واټن(وزن_type) ? quantize_f32_q8_1_tiled_بلاک : quantize_f32_q8_0_tiled_بلاک;
}

ساكن فراغ hvx_mm_4d_tiled(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_لیکه_پای_ټکی - mmctx->src0_لیکه_پیل;
    ثابت uint32_t cur_m_لیکې = mmctx->cur_m_لیکې ? mmctx->cur_m_لیکې : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_پیل = mmctx->cur_m_پیل;

    ثابت uint32_t src0_پیل_لیکه  = mmctx->src0_لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه    = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, mmctx->src0_لیکه_پای_ټکی);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_لیکه_اندازه = nb1;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * محدود act_معلومات = mmctx->vtcm_act;

    ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src0->type);
    ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src0->type);

    ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ثابت uint32_t tile_لیکه_stride = n_k_tiles_w * tile_اندازه;
    ثابت uint32_t tile_لیکه_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;
    ثابت uint32_t src0_ټوټه_stride = ((ne01 + 31) / 32) * tile_لیکه_stride;

    ثابت uint32_t ct_پیل = src0_پیل_لیکه / 32;
    ثابت uint32_t ct_پای_ټکی   = (src0_پای_ټکی_لیکه + 31) / 32;

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه || cur_m_لیکې == 0) {
        أرجع;
    }

    ثابت uint32_t ټول_batches = ne12 * ne13;
    ثابت uint32_t b_پیل = fastdiv(cur_m_پیل, &kparams->div_ne1);
    uint32_t b_پای_ټکی = fastdiv(cur_m_پیل + cur_m_لیکې + ne11 - 1, &kparams->div_ne1);
    b_پای_ټکی = MIN(b_پای_ټکی, ټول_batches);

    uint32_t b_grp_پیل = b_پیل;
    طالما (b_grp_پیل < b_پای_ټکی) {
        ثابت uint32_t b3 = fastdiv(b_grp_پیل, &kparams->div_ne12);
        ثابت uint32_t b2 = b_grp_پیل - b3 * ne12;
        ثابت uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ثابت uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_پای_ټکی = b_grp_پیل + 1;
        طالما (b_grp_پای_ټکی < b_پای_ټکی) {
            ثابت uint32_t cur_b3 = fastdiv(b_grp_پای_ټکی, &kparams->div_ne12);
            ثابت uint32_t cur_b2 = b_grp_پای_ټکی - cur_b3 * ne12;
            ثابت uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ثابت uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            إذا (cur_i02 != i02 || cur_i03 != i03) {
                اكسر;
            }
            b_grp_پای_ټکی++;
        }

        ثابت uint32_t ټوټه_idx = i03 * ne02 + i02;
        ثابت dma_addr_t src0_ټوټه = src0->معلومات + (size_t) ټوټه_idx * src0_ټوټه_stride;

        uint32_t ټېل_ct = ct_پیل;
        لكل (uint32_t d = 0; d < n_prefetch && ټېل_ct < ct_پای_ټکی; d++, ټېل_ct++) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + d * tile_لیکه_transfer_اندازه_aligned,
                           src0_ټوټه + (size_t) ټېل_ct * tile_لیکه_stride),
                           aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_پیل; ct < ct_پای_ټکی; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

            صحيح valid_لیکې = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_لیکې = MIN(32, MAX(0, valid_لیکې));

            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            لكل (uint32_t b = b_grp_پیل; b < b_grp_پای_ټکی; b++) {
                ثابت uint32_t b_m_پیل = b * ne11;
                ثابت uint32_t m_لومړی = MAX(cur_m_پیل, b_m_پیل);
                ثابت uint32_t m_وروستی  = MIN(cur_m_پیل + cur_m_لیکې, b_m_پیل + ne11);
                إذا (m_لومړی >= m_وروستی) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_بنسټ = (uint8_t *) dst->معلومات + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_دننه_واټن = m_لومړی - cur_m_پیل;
                ثابت uint32_t dst_m_دننه_واټن   = m_لومړی - b_m_پیل;
                ثابت uint32_t batch_nrows    = m_وروستی - m_لومړی;

                uint32_t ir1 = 0;
                لكل (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_معلومات + (chunk_m_دننه_واټن + ir1 + 0) * act_stride);
                    ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_معلومات + (chunk_m_دننه_واټن + ir1 + 1) * act_stride);
                    عائم * محدود dst_row0 = (عائم *) (dst_batch_بنسټ + (dst_m_دننه_واټن + ir1 + 0) * dst_لیکه_اندازه);
                    عائم * محدود dst_row1 = (عائم *) (dst_batch_بنسټ + (dst_m_دننه_واټن + ir1 + 1) * dst_لیکه_اندازه);
                    عائم * dst_ptr0 = &dst_row0[ct * 32];
                    عائم * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_لیکې, عدم, عدم);
                }
                لكل (; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_معلومات + (chunk_m_دننه_واټن + ir1) * act_stride);
                    عائم * محدود dst_لیکه = (عائم *) (dst_batch_بنسټ + (dst_m_دننه_واټن + ir1) * dst_لیکه_اندازه);
                    عائم * dst_ptr = &dst_لیکه[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_لیکې, عدم);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (ټېل_ct < ct_پای_ټکی) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(w_tile, src0_ټوټه + (size_t) ټېل_ct * tile_لیکه_stride),
                               aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                ټېل_ct++;
            }
        }
        b_grp_پیل = b_grp_پای_ټکی;
    }
    إذا (src2) {
        hvx_tensor_جمع_کړه_f32_grid(dst, src2, cur_m_پیل, cur_m_پیل + cur_m_لیکې, src0_پیل_لیکه, src0_پای_ټکی_لیکه, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ساكن فراغ hvx_mm_2d_flat(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_matmul_preamble;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_ماسک = n_prefetch - 1;

    ثابت uint32_t src0_nrows = mmctx->src0_لیکه_پای_ټکی - mmctx->src0_لیکه_پیل;  // src0 rows
    ثابت uint32_t act_nrows = mmctx->cur_m_لیکې ? mmctx->cur_m_لیکې : mmctx->act_nrows;
    ثابت uint32_t cur_m_پیل = mmctx->cur_m_پیل;

    ثابت uint32_t src0_پیل_لیکه  = mmctx->src0_لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه    = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, mmctx->src0_لیکه_پای_ټکی);
    ثابت uint32_t src0_پای_ټکی_لیکه_x2 = src0_پیل_لیکه + ((src0_پای_ټکی_لیکه - src0_پیل_لیکه) & ~1U);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_لیکه_اندازه  = nb1;
    ثابت size_t src0_لیکه_اندازه = nb01;
    ثابت size_t act_لیکه_اندازه = nb11;

    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_اندازه_فی_thread  * ith;
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * محدود act_معلومات      = mmctx->vtcm_act;

    ثابت dma_addr_t src0_لیکه = src0->معلومات;


    // Prefill vtcm with src0 rows
    إذا (src0_پیل_لیکه < src0_پای_ټکی_لیکه) {
        لكل (uint32_t ir0 = src0_پیل_لیکه; ir0 < src0_پای_ټکی_لیکه_x2; ir0 += 2) {
            ثابت صحيح is0 = (ir0 - src0_پیل_لیکه);
            إذا (is0 >= (صحيح)n_prefetch) {
                اكسر;
            }
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + is0 * src0_stride, src0_لیکه + ir0 * src0_لیکه_اندازه),
                           src0_stride, src0_لیکه_اندازه, src0_لیکه_اندازه, 2);
        }
    }

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    // Process src0 rows
    لكل (uint32_t ir0 = src0_پیل_لیکه; ir0 < src0_پای_ټکی_لیکه_x2; ir0 += 2) {
        ثابت uint8_t * ss0 = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
            ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_معلومات + (ir1+0) * act_stride);
            ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_معلومات + (ir1+1) * act_stride);
            عائم * محدود dst_row0 = (عائم *) (dst->معلومات + ((cur_m_پیل + ir1+0) * dst_لیکه_اندازه));
            عائم * محدود dst_row1 = (عائم *) (dst->معلومات + ((cur_m_پیل + ir1+1) * dst_لیکه_اندازه));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        لكل (; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_معلومات + ir1 * act_stride);
            عائم * محدود dst_لیکه          = (عائم *) (dst->معلومات + ((cur_m_پیل + ir1) * dst_لیکه_اندازه));
            mmctx->vec_dot_2x1(ne00, &dst_لیکه[ir0], ss0, ss0 + src0_stride, act_ستون);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ثابت صحيح pr0 = (ir0 + n_prefetch);
        ثابت صحيح is0 = (pr0 - src0_پیل_لیکه) & prefetch_ماسک;
        إذا (pr0 < src0_پای_ټکی_لیکه_x2) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + is0 * src0_stride, src0_لیکه + pr0 * src0_لیکه_اندازه),
                           src0_stride, src0_لیکه_اندازه, src0_لیکه_اندازه, 2);
        }
    }

    // Process the last row (if any)
    إذا (src0_پای_ټکی_لیکه != src0_پای_ټکی_لیکه_x2) {
        uint32_t  ir0 = src0_پای_ټکی_لیکه_x2;
        ثابت صحيح is0 = (ir0 - src0_پیل_لیکه) & prefetch_ماسک;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + is0 * src0_stride, src0_لیکه + ir0 * src0_لیکه_اندازه),
                       src0_stride, src0_لیکه_اندازه, src0_لیکه_اندازه, 1);
        ثابت uint8_t * ss0 = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #لارښوونه unroll(2)
        لكل (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_معلومات + ir1 * act_stride);
            عائم * محدود dst_لیکه          = (عائم *) (dst->معلومات + ((cur_m_پیل + ir1) * dst_لیکه_اندازه));
            mmctx->vec_dot_1x1(ne00, &dst_لیکه[ir0], ss0, act_ستون);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    إذا (src2) {
        hvx_tensor_جمع_کړه_f32_grid(dst, src2, cur_m_پیل, cur_m_پیل + act_nrows, src0_پیل_لیکه, src0_پای_ټکی_لیکه, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ساكن فراغ hvx_mv_2d_flat(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_لیکه_پای_ټکی - mmctx->src0_لیکه_پیل;

    ثابت uint32_t src0_پیل_لیکه  = mmctx->src0_لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه    = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, mmctx->src0_لیکه_پای_ټکی);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_لیکه_اندازه  = nb1;
    ثابت size_t src0_لیکه_اندازه = nb01;
    ثابت size_t act_لیکه_اندازه  = nb11;

    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_اندازه_فی_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * act_معلومات      = mmctx->vtcm_act;

    عائم * tmp = (عائم *) vtcm_dst_ptr;

    ثابت dma_addr_t src0_لیکه = src0->معلومات;
    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_معلومات;
    عائم * محدود dst_ستون         = (عائم *) dst->معلومات;

    ثابت uint32_t src0_پای_ټکی_لیکه_x2 = src0_پیل_لیکه + ((src0_پای_ټکی_لیکه - src0_پیل_لیکه) & ~1U);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_ماسک = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    إذا (src0_پیل_لیکه < src0_پای_ټکی_لیکه) {
        إذا (src2) {
            عائم * vtcm_bias_ptr = (عائم *) mmctx->vtcm_bias + src0_پیل_لیکه;
            ثابت dma_addr_t src2_addr = src2->معلومات + src0_پیل_لیکه * حجم(عائم);
            صحيح ټوټه_اندازه = (صحيح)src0_پای_ټکی_لیکه - (صحيح)src0_پیل_لیکه;
            إذا (ټوټه_اندازه > 0) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_bias_ptr, src2_addr),
                               ټوټه_اندازه * حجم(عائم), ټوټه_اندازه * حجم(عائم), ټوټه_اندازه * حجم(عائم), 1);
                dma_کتار_راوباسه_nowait(dma_q);
            }
        }
        لكل (uint32_t ir0 = src0_پیل_لیکه; ir0 < src0_پای_ټکی_لیکه_x2; ir0 += 2) {
            ثابت uint32_t is0 = (ir0 - src0_پیل_لیکه);
            إذا (is0 >= n_prefetch) {
                اكسر;
            }
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + is0 * src0_stride, src0_لیکه + ir0 * src0_لیکه_اندازه),
                           src0_stride, src0_لیکه_اندازه, src0_لیکه_اندازه, 2);
        }
    }


    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    // Process src0 rows
    لكل (uint32_t ir0 = src0_پیل_لیکه; ir0 < src0_پای_ټکی_لیکه_x2; ir0 += 2) {
        ثابت uint8_t * ss0 = (فراغ *) dma_کتار_راوباسه(dma_q).dst;
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_پیل_لیکه], ss0, ss0 + src0_stride, act_ستون);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ثابت uint32_t pr0 = (ir0 + n_prefetch);
        ثابت uint32_t is0 = (pr0 - src0_پیل_لیکه) & prefetch_ماسک;
        إذا (pr0 < src0_پای_ټکی_لیکه_x2) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + is0 * src0_stride, src0_لیکه + pr0 * src0_لیکه_اندازه),
                           src0_stride, src0_لیکه_اندازه, src0_لیکه_اندازه, 2);
        }
    }

    // Process the last row (if any)
    إذا (src0_پای_ټکی_لیکه != src0_پای_ټکی_لیکه_x2) {
        ثابت uint32_t ir0 = src0_پای_ټکی_لیکه_x2;
        ثابت uint32_t is0 = (ir0 - src0_پیل_لیکه) & prefetch_ماسک;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + is0 * src0_stride, src0_لیکه + ir0 * src0_لیکه_اندازه),
                       src0_stride, src0_لیکه_اندازه, src0_لیکه_اندازه, 1);
        ثابت uint8_t * ss0 = (فراغ *) dma_کتار_راوباسه(dma_q).dst;
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_پیل_لیکه], ss0, act_ستون);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    صحيح copy_cnt = src0_پای_ټکی_لیکه - src0_پیل_لیکه;
    إذا (copy_cnt > 0) {
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, src0_پای_ټکی_لیکه);
        إذا (src2) {
            hvx_جمع_کړه_f32_uuu((uint8_t *) &dst_ستون[src0_پیل_لیکه],
                            (ثابت uint8_t *) tmp,
                            (ثابت uint8_t *) ((ثابت عائم *) mmctx->vtcm_bias + src0_پیل_لیکه),
                            copy_cnt);
        } وإلا {
            hvx_copy_f32_ua((uint8_t *) &dst_ستون[src0_پیل_لیکه], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_پای_ټکی_لیکه);
    }
}

ساكن فراغ hvx_mm_4d_flat(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_matmul_preamble;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_ماسک = n_prefetch - 1;

    ثابت uint32_t src0_nrows = mmctx->src0_لیکه_پای_ټکی - mmctx->src0_لیکه_پیل;
    ثابت uint32_t cur_m_لیکې = mmctx->cur_m_لیکې ? mmctx->cur_m_لیکې : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_پیل = mmctx->cur_m_پیل;

    ثابت uint32_t src0_پیل_لیکه  = mmctx->src0_لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه    = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, mmctx->src0_لیکه_پای_ټکی);
    ثابت uint32_t src0_پای_ټکی_لیکه_x2 = src0_پیل_لیکه + ((src0_پای_ټکی_لیکه - src0_پیل_لیکه) & ~1U);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_لیکه_اندازه  = nb1;
    ثابت size_t src0_لیکه_اندازه = nb01;
    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * محدود act_معلومات      = mmctx->vtcm_act;


    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه || cur_m_لیکې == 0) {
        أرجع;
    }

    ثابت uint32_t ټول_batches = ne12 * ne13;
    ثابت uint32_t b_پیل = fastdiv(cur_m_پیل, &kparams->div_ne1);
    uint32_t b_پای_ټکی = fastdiv(cur_m_پیل + cur_m_لیکې + ne11 - 1, &kparams->div_ne1);
    b_پای_ټکی = MIN(b_پای_ټکی, ټول_batches);

    uint32_t b_grp_پیل = b_پیل;
    طالما (b_grp_پیل < b_پای_ټکی) {
        ثابت uint32_t b3 = fastdiv(b_grp_پیل, &kparams->div_ne12);
        ثابت uint32_t b2 = b_grp_پیل - b3 * ne12;
        ثابت uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ثابت uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_پای_ټکی = b_grp_پیل + 1;
        طالما (b_grp_پای_ټکی < b_پای_ټکی) {
            ثابت uint32_t cur_b3 = fastdiv(b_grp_پای_ټکی, &kparams->div_ne12);
            ثابت uint32_t cur_b2 = b_grp_پای_ټکی - cur_b3 * ne12;
            ثابت uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ثابت uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            إذا (cur_i02 != i02 || cur_i03 != i03) {
                اكسر;
            }
            b_grp_پای_ټکی++;
        }

        ثابت dma_addr_t src0_لیکه = src0->معلومات + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        لكل (uint32_t ir0 = src0_پیل_لیکه; ir0 < src0_پای_ټکی_لیکه_x2; ir0 += 2) {
            ثابت صحيح is0 = (ir0 - src0_پیل_لیکه);
            إذا (is0 >= (صحيح)n_prefetch) {
                اكسر;
            }
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + is0 * src0_stride, src0_لیکه + (size_t) ir0 * src0_لیکه_اندازه),
                           src0_stride, src0_لیکه_اندازه, src0_لیکه_اندازه, 2);
        }

        لكل (uint32_t ir0 = src0_پیل_لیکه; ir0 < src0_پای_ټکی_لیکه_x2; ir0 += 2) {
            ثابت uint8_t * ss0 = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            لكل (uint32_t b = b_grp_پیل; b < b_grp_پای_ټکی; b++) {
                ثابت uint32_t b_m_پیل = b * ne11;
                ثابت uint32_t m_لومړی = MAX(cur_m_پیل, b_m_پیل);
                ثابت uint32_t m_وروستی  = MIN(cur_m_پیل + cur_m_لیکې, b_m_پیل + ne11);
                إذا (m_لومړی >= m_وروستی) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_بنسټ = (uint8_t *) dst->معلومات + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_دننه_واټن = m_لومړی - cur_m_پیل;
                ثابت uint32_t dst_m_دننه_واټن   = m_لومړی - b_m_پیل;
                ثابت uint32_t batch_nrows    = m_وروستی - m_لومړی;

                uint32_t ir1 = 0;
                لكل (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_معلومات + (chunk_m_دننه_واټن + ir1 + 0) * act_stride);
                    ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_معلومات + (chunk_m_دننه_واټن + ir1 + 1) * act_stride);
                    عائم * محدود dst_row0 = (عائم *) (dst_batch_بنسټ + (dst_m_دننه_واټن + ir1 + 0) * dst_لیکه_اندازه);
                    عائم * محدود dst_row1 = (عائم *) (dst_batch_بنسټ + (dst_m_دننه_واټن + ir1 + 1) * dst_لیکه_اندازه);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                لكل (; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_معلومات + (chunk_m_دننه_واټن + ir1) * act_stride);
                    عائم * محدود dst_لیکه          = (عائم *) (dst_batch_بنسټ + (dst_m_دننه_واټن + ir1) * dst_لیکه_اندازه);
                    mmctx->vec_dot_2x1(ne00, &dst_لیکه[ir0], ss0, ss0 + src0_stride, act_ستون);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ثابت صحيح pr0 = (ir0 + n_prefetch);
            ثابت صحيح is0 = (pr0 - src0_پیل_لیکه) & prefetch_ماسک;
            إذا (pr0 < src0_پای_ټکی_لیکه_x2) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + is0 * src0_stride, src0_لیکه + (size_t) pr0 * src0_لیکه_اندازه),
                               src0_stride, src0_لیکه_اندازه, src0_لیکه_اندازه, 2);
            }
        }

        إذا (src0_پای_ټکی_لیکه != src0_پای_ټکی_لیکه_x2) {
            uint32_t ir0 = src0_پای_ټکی_لیکه_x2;
            ثابت صحيح is0 = (ir0 - src0_پیل_لیکه) & prefetch_ماسک;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + is0 * src0_stride, src0_لیکه + (size_t) ir0 * src0_لیکه_اندازه),
                           src0_stride, src0_لیکه_اندازه, src0_لیکه_اندازه, 1);
            ثابت uint8_t * ss0 = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            لكل (uint32_t b = b_grp_پیل; b < b_grp_پای_ټکی; b++) {
                ثابت uint32_t b_m_پیل = b * ne11;
                ثابت uint32_t m_لومړی = MAX(cur_m_پیل, b_m_پیل);
                ثابت uint32_t m_وروستی  = MIN(cur_m_پیل + cur_m_لیکې, b_m_پیل + ne11);
                إذا (m_لومړی >= m_وروستی) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_بنسټ = (uint8_t *) dst->معلومات + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_دننه_واټن = m_لومړی - cur_m_پیل;
                ثابت uint32_t dst_m_دننه_واټن   = m_لومړی - b_m_پیل;
                ثابت uint32_t batch_nrows    = m_وروستی - m_لومړی;

                لكل (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_معلومات + (chunk_m_دننه_واټن + ir1) * act_stride);
                    عائم * محدود dst_لیکه          = (عائم *) (dst_batch_بنسټ + (dst_m_دننه_واټن + ir1) * dst_لیکه_اندازه);
                    mmctx->vec_dot_1x1(ne00, &dst_لیکه[ir0], ss0, act_ستون);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_پیل = b_grp_پای_ټکی;
    }

    إذا (src2) {
        hvx_tensor_جمع_کړه_f32_grid(dst, src2, cur_m_پیل, cur_m_پیل + cur_m_لیکې, src0_پیل_لیکه, src0_پای_ټکی_لیکه, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#تعریف_کړه MMID_ماتریکس_لیکه(لیکه_id, i1) ماتریکس_لیکې[(لیکه_id) * mmctx->mapping_stride + (i1)]

ساكن فراغ hvx_mm_id_tiled(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_matmul_preamble;

    ثابت بنية htp_tensor * محدود ids = octx->src[2];

    ثابت uint32_t src0_nrows      = mmctx->src0_لیکه_پای_ټکی - mmctx->src0_لیکه_پیل;  // src0 rows per expert
    ثابت uint32_t act_nrows       = ne11;
    ثابت uint32_t src0_پیل_لیکه  = mmctx->src0_لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه    = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, mmctx->src0_لیکه_پای_ټکی);


    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_ids = ids->ne[0];  // n_expert_used
    ثابت uint32_t n_as  = ne02;        // n_expert

    ثابت uint32_t *                ماتریکس_لیکه_counts = mmctx->ماتریکس_لیکه_counts;
    ثابت بنية mmid_لیکه_mapping * ماتریکس_لیکې       = mmctx->ماتریکس_لیکې;

    ثابت size_t dst_لیکه_اندازه  = nb1;
    ثابت size_t act_لیکه_اندازه  = htp_mm_q8_0_tiled_لیکه_اندازه(ne10);

    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * محدود act_معلومات = mmctx->vtcm_act;

    لكل (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int32_t cne1 = ماتریکس_لیکه_counts[cur_a];
        إذا (cne1 == 0) {
            استمر;
        }

        ثابت dma_addr_t src0_لیکه = src0->معلومات + cur_a * nb02;

        ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src0->type);
        ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src0->type);
        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ثابت uint32_t tile_لیکه_stride = n_k_tiles_w * tile_اندازه;
        ثابت uint32_t tile_لیکه_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

        ثابت uint32_t ct_پیل = src0_پیل_لیکه / 32;
        ثابت uint32_t ct_پای_ټکی   = (src0_پای_ټکی_لیکه + 31) / 32;

        uint32_t ټېل_ct = ct_پیل;
        لكل (uint32_t d = 0; d < n_prefetch && ټېل_ct < ct_پای_ټکی; d++, ټېل_ct++) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + d * tile_لیکه_transfer_اندازه_aligned, src0_لیکه + ټېل_ct * tile_لیکه_stride),
                           aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_پیل; ct < ct_پای_ټکی; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

            صحيح valid_لیکې = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_لیکې = MIN(32, MAX(0, valid_لیکې));

            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            لكل (uint32_t cid = 0; cid < cne1; ++cid) {
                بنية mmid_لیکه_mapping لیکه_mapping = MMID_ماتریکس_لیکه(cur_a, cid);
                ثابت صحيح               rm1         = لیکه_mapping.i1;  // expert idx
                ثابت صحيح               rm2         = لیکه_mapping.i2;  // token idx

                ثابت uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_معلومات + (ir1 + rm2 * ne11 + 0) * act_stride);
                عائم * محدود dst_لیکه = (عائم *) (dst->معلومات + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_لیکه[ct * 32], w_tile, act_ستون, valid_لیکې, عدم);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (ټېل_ct < ct_پای_ټکی) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(w_tile, src0_لیکه + ټېل_ct * tile_لیکه_stride),
                               aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                ټېل_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mv_id_tiled(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_matmul_preamble;

    ثابت بنية htp_tensor * محدود ids = octx->src[2];

    ثابت uint32_t src0_nrows      = mmctx->src0_لیکه_پای_ټکی - mmctx->src0_لیکه_پیل;  // src0 rows per expert
    ثابت uint32_t src0_پیل_لیکه  = mmctx->src0_لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه    = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, mmctx->src0_لیکه_پای_ټکی);


    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    أكد(ne13 % ne03 == 0);

    ثابت size_t dst_لیکه_اندازه  = nb1;
    ثابت size_t act_لیکه_اندازه = htp_mm_q8_0_tiled_لیکه_اندازه(ne10);

    ثابت uint32_t n_aids = src2->ne[0];  // num activated experts
    ثابت uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * محدود act_معلومات = mmctx->vtcm_act;

    لكل (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ثابت int32_t eid = *(ثابت int32_t *) ((ثابت uint8_t *) src2->معلومات + ie1 * src2->nb[0]);
        إذا (eid < 0) {
            استمر;
        }
        أكد(eid < (int32_t) n_ids);

        ثابت dma_addr_t src0_لیکه = src0->معلومات + eid * nb02;
        ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_معلومات;
        عائم * محدود dst_لیکه          = (عائم *) (dst->معلومات + ie1 * nb1);

        ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src0->type);
        ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src0->type);
        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ثابت uint32_t tile_لیکه_stride = n_k_tiles_w * tile_اندازه;
        ثابت uint32_t tile_لیکه_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

        ثابت uint32_t ct_پیل = src0_پیل_لیکه / 32;
        ثابت uint32_t ct_پای_ټکی   = (src0_پای_ټکی_لیکه + 31) / 32;

        uint32_t ټېل_ct = ct_پیل;
        لكل (uint32_t d = 0; d < n_prefetch && ټېل_ct < ct_پای_ټکی; d++, ټېل_ct++) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + d * tile_لیکه_transfer_اندازه_aligned, src0_لیکه + ټېل_ct * tile_لیکه_stride),
                           aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_پیل; ct < ct_پای_ټکی; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

            صحيح valid_لیکې = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_لیکې = MIN(32, MAX(0, valid_لیکې));

            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_لیکه[ct * 32], w_tile, act_ستون, valid_لیکې, عدم);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (ټېل_ct < ct_پای_ټکی) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(w_tile, src0_لیکه + ټېل_ct * tile_لیکه_stride),
                               aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                ټېل_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mv_id_nx_tiled(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_mm_context * mmctx = (بنية htp_mm_context *) معلومات;
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

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * محدود act_معلومات = mmctx->vtcm_act;

    لكل (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ثابت int32_t eid = *(ثابت int32_t *) ((ثابت uint8_t *) ids->معلومات + ie1 * ids->nb[0]);
        إذا (eid < 0) استمر;
        أكد(eid < (int32_t) n_ids);

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * محدود src_w = octx->src[p];
            ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
            إذا (!src_w || !dst) استمر;
            dma_کتار * dma_q = octx->ctx->dma[ith];

            ثابت uint32_t ne01 = src_w->ne[1];
            uint32_t پیل_لیکه = 0;
            uint32_t پای_ټکی_لیکه   = ne01;
            إذا (octx->ctx->mdev.شمېر > 1) {
                ثابت bool can_وېشل = htp_tensor_can_لیکه_برخه_کول(dst, حجم(عائم));
                ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(ne01, can_وېشل ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
                پیل_لیکه = range.پیل;
                پای_ټکی_لیکه   = range.پیل + range.شمېر;
            }

            ثابت uint32_t nrows = پای_ټکی_لیکه - پیل_لیکه;
            uint32_t src0_nrows_فی_thread = fastdiv(nrows + nth - 1, &octx->n_تارونه_div);
            src0_nrows_فی_thread = hex_پړاو_پورته(src0_nrows_فی_thread, 32);

            ثابت uint32_t src0_پیل_لیکه = پیل_لیکه + src0_nrows_فی_thread * ith;
            ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, پای_ټکی_لیکه);
            إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) استمر;

            ثابت dma_addr_t src0_لیکه = src_w->معلومات + eid * src_w->nb[2];
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_معلومات;
            عائم * محدود dst_لیکه = (عائم *) (dst->معلومات + ie1 * dst->nb[1]);

            ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src_w->type);
            ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src_w->type);
            ثابت uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ثابت uint32_t tile_لیکه_stride = n_k_tiles_w * tile_اندازه;
            ثابت uint32_t tile_لیکه_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

            ثابت uint32_t ct_پیل = src0_پیل_لیکه / 32;
            ثابت uint32_t ct_پای_ټکی   = (src0_پای_ټکی_لیکه + 31) / 32;

            uint32_t ټېل_ct = ct_پیل;
            لكل (uint32_t d = 0; d < n_prefetch && ټېل_ct < ct_پای_ټکی; d++, ټېل_ct++) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + d * tile_لیکه_transfer_اندازه_aligned, src0_لیکه + ټېل_ct * tile_لیکه_stride),
                               aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
            }

            لكل (uint32_t ct = ct_پیل; ct < ct_پای_ټکی; ct++) {
                ثابت uint8_t * w_tile = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

                صحيح valid_لیکې = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
                valid_لیکې = MIN(32, MAX(0, valid_لیکې));

                htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_لیکه[ct * 32], w_tile, act_ستون, valid_لیکې, عدم);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                إذا (ټېل_ct < ct_پای_ټکی) {
                    dma_کتار_ټېل(dma_q, dma_make_معلومات(w_tile, src0_لیکه + ټېل_ct * tile_لیکه_stride),
                                    aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                    ټېل_ct++;
                }
            }
        }
    }
}

ساكن فراغ hvx_mm_id_nx_tiled(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_mm_context * mmctx = (بنية htp_mm_context *) معلومات;
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

    ثابت uint32_t * ماتریکس_لیکه_counts = mmctx->ماتریکس_لیکه_counts;
    ثابت بنية mmid_لیکه_mapping * ماتریکس_لیکې = mmctx->ماتریکس_لیکې;

    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_فی_thread * ith;
    uint8_t * محدود act_معلومات = mmctx->vtcm_act;

    لكل (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int32_t cne1 = ماتریکس_لیکه_counts[cur_a];
        إذا (cne1 == 0) استمر;

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * محدود src_w = octx->src[p];
            ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
            إذا (!src_w || !dst) استمر;
            dma_کتار * dma_q = octx->ctx->dma[ith];

            ثابت uint32_t ne01 = src_w->ne[1];
            uint32_t پیل_لیکه = 0;
            uint32_t پای_ټکی_لیکه   = ne01;
            إذا (octx->ctx->mdev.شمېر > 1) {
                ثابت bool can_وېشل = htp_tensor_can_لیکه_برخه_کول(dst, حجم(عائم));
                ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(ne01, can_وېشل ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
                پیل_لیکه = range.پیل;
                پای_ټکی_لیکه   = range.پیل + range.شمېر;
            }

            ثابت uint32_t nrows = پای_ټکی_لیکه - پیل_لیکه;
            uint32_t src0_nrows_فی_thread = fastdiv(nrows + nth - 1, &octx->n_تارونه_div);
            src0_nrows_فی_thread = hex_پړاو_پورته(src0_nrows_فی_thread, 32);

            ثابت uint32_t src0_پیل_لیکه = پیل_لیکه + src0_nrows_فی_thread * ith;
            ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, پای_ټکی_لیکه);
            إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) استمر;

            ثابت dma_addr_t src0_لیکه = src_w->معلومات + cur_a * src_w->nb[2];

            ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src_w->type);
            ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src_w->type);
            ثابت uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ثابت uint32_t tile_لیکه_stride = n_k_tiles_w * tile_اندازه;
            ثابت uint32_t tile_لیکه_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

            ثابت uint32_t ct_پیل = src0_پیل_لیکه / 32;
            ثابت uint32_t ct_پای_ټکی   = (src0_پای_ټکی_لیکه + 31) / 32;

            uint32_t ټېل_ct = ct_پیل;
            لكل (uint32_t d = 0; d < n_prefetch && ټېل_ct < ct_پای_ټکی; d++, ټېل_ct++) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src0_ptr + d * tile_لیکه_transfer_اندازه_aligned, src0_لیکه + ټېل_ct * tile_لیکه_stride),
                               aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
            }

            لكل (uint32_t ct = ct_پیل; ct < ct_پای_ټکی; ct++) {
                ثابت uint8_t * w_tile = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

                صحيح valid_لیکې = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
                valid_لیکې = MIN(32, MAX(0, valid_لیکې));

                htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                لكل (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    بنية mmid_لیکه_mapping لیکه_mapping = MMID_ماتریکس_لیکه(cur_a, cid);
                    ثابت صحيح rm1 = لیکه_mapping.i1;
                    ثابت صحيح rm2 = لیکه_mapping.i2;

                    ثابت uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_معلومات + (ir1 + rm2 * act->ne[1]) * act_stride);
                    عائم * محدود dst_لیکه = (عائم *) (dst->معلومات + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_لیکه[ct * 32], w_tile, act_ستون, valid_لیکې, عدم);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                إذا (ټېل_ct < ct_پای_ټکی) {
                    dma_کتار_ټېل(dma_q, dma_make_معلومات(w_tile, src0_لیکه + ټېل_ct * tile_لیکه_stride),
                                    aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                    ټېل_ct++;
                }
            }
        }
    }
}

ساكن صحيح hvx_mm_init_tiled_vec_dot(بنية htp_mm_context * mmctx, تعداد htp_معلومات_type type) {
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
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, 0);

    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    ثابت uint32_t src0_nrows = ne01;
    ثابت uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_لیکه_پیل = 0;
    uint32_t src0_لیکه_پای_ټکی   = src0_nrows;

    إذا (octx->ctx->mdev.شمېر > 1) {
        ثابت bool can_وېشل = htp_tensor_can_لیکه_برخه_کول(dst, حجم(عائم));
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(src0_nrows, can_وېشل ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        src0_لیکه_پیل = range.پیل;
        src0_لیکه_پای_ټکی   = range.پیل + range.شمېر;
    }

    إذا (src0_لیکه_پیل >= src0_لیکه_پای_ټکی) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t nrows = src0_لیکه_پای_ټکی - src0_لیکه_پیل;
    mmctx->src0_لیکه_پیل = src0_لیکه_پیل;
    mmctx->src0_لیکه_پای_ټکی   = src0_لیکه_پای_ټکی;

    ثابت bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_فی_thread  = fastdiv(nrows + octx->n_تارونه - 1, &octx->n_تارونه_div);
    إذا (is_tiled) {
        mmctx->src0_nrows_فی_thread = hex_پړاو_پورته(mmctx->src0_nrows_فی_thread, 32);
    } وإلا {
        mmctx->src0_nrows_فی_thread += (mmctx->src0_nrows_فی_thread & 1); // round up to even
    }

    ثابت size_t src0_لیکه_اندازه = nb01;
    ثابت size_t dst_لیکه_اندازه  = nb1;
    size_t       act_لیکه_اندازه  = nb11;

    ثابت size_t src0_لیکه_اندازه_padded = hex_پړاو_پورته(src0_لیکه_اندازه, 128);

    کارګر_callback_t quant_task_func;
    کارګر_callback_t matmul_دنده_func;
    uint32_t n_quant_tasks = 1;
    ثابت bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    إذا (is_batched) {
        matmul_دنده_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } وإلا إذا (act_nrows > 1) {
        matmul_دنده_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } وإلا {
        matmul_دنده_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
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
            act_لیکه_اندازه           = kparams->act_لیکه_اندازه;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = عدم;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_لیکه_اندازه           = kparams->act_لیکه_اندازه;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = عدم;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_لیکه_اندازه           = kparams->act_لیکه_اندازه;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_QUANT_بلاک:
        حالة HTP_MM_KERNEL_HVX_QUANT_لیکه:
        افتراضي: {
            إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                أرجع HTP_حالت_NO_SUPPORT;
            }

            ثابت uint32_t qk = QK_Q8_0_TILED;
            ثابت uint32_t nb = (ne10 + qk - 1) / qk;
            ثابت uint32_t ټول_nb = act_nrows * nb;

            إذا (act_nrows < octx->n_تارونه && !is_batched) {
                n_quant_tasks = MIN(ټول_nb, octx->n_تارونه);
                quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
                لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_لومړی = (ټول_nb * ith) / n_quant_tasks;
                    uint32_t ib_وروستی  = (ټول_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_لومړی[ith] = ib_لومړی;
                    mmctx->quant_ib_وروستی[ith]  = ib_وروستی;
                    mmctx->quant_r[ith]        = ib_لومړی / nb;
                    mmctx->quant_c[ith]        = ib_لومړی % nb;
                }
            } وإلا {
                n_quant_tasks = MIN(act_nrows, octx->n_تارونه);
                quant_task_func = htp_mm_act_quant_لیکه_func(src0->type, act->type);
            }
            act_لیکه_اندازه = kparams->act_لیکه_اندازه;
            اكسر;
        }
    }

    ثابت uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ثابت uint32_t m_layout_لیکې = m_chunk;

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_لیکې, octx->n_تارونه,
                                 dst_لیکه_اندازه, src0_لیکه_اندازه, act_لیکه_اندازه, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    إذا (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_لیکه ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
        mmctx->vtcm_act_اندازه_فی_thread = L.act_بایټونه;
    } وإلا {
        mmctx->vtcm_act_اندازه_فی_thread = fastdiv(L.act_بایټونه, &octx->n_تارونه_div);
    }

    mmctx->vtcm_src0_اندازه_فی_thread = fastdiv(L.src0_بایټونه, &octx->n_تارونه_div);
    mmctx->vtcm_dst_اندازه_فی_thread  = fastdiv(L.dst_بایټونه, &octx->n_تارونه_div);

    ثابت size_t vtcm_اندازه = L.ټول_بایټونه;

    FARF(لوړ_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_بایټونه, L.act_بایټونه, L.dst_بایټونه, vtcm_اندازه);

    FARF(لوړ_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->معلومات, act->معلومات, dst->معلومات);

    إذا (octx->ctx->vtcm_اندازه < vtcm_اندازه) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_اندازه, vtcm_اندازه);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_لیکه_اندازه_padded;
    mmctx->vtcm_act_stride  = act_لیکه_اندازه;
    إذا (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_لیکه) {
        mmctx->vtcm_act_raw_stride = hex_پړاو_پورته(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
    } وإلا إذا (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_پړاو_پورته(ne10 * حجم(عائم), 128);
    } وإلا {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    إذا (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        لكل (uint32_t m_پیل = 0; m_پیل < act_nrows; m_پیل += m_chunk) {
            ثابت uint32_t cur_m_لیکې = MIN(act_nrows - m_پیل, m_chunk);
            mmctx->cur_m_پیل = m_پیل;
            mmctx->cur_m_لیکې  = cur_m_لیکې;

            إذا (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_پیل, cur_m_لیکې);

                ثابت uint32_t qk = QK_Q8_0_TILED;
                ثابت uint32_t nb = (ne10 + qk - 1) / qk;
                ثابت uint32_t ټول_nb = cur_m_لیکې * nb;
                uint32_t quant_tasks;
                work_کتار_func_t q_func;
                إذا (cur_m_لیکې < octx->n_تارونه && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_لیکه)) {
                    quant_tasks = MIN(ټول_nb, octx->n_تارونه);
                    q_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
                    لكل (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_لومړی = (ټول_nb * ith) / quant_tasks;
                        uint32_t ib_وروستی  = (ټول_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_لومړی[ith] = ib_لومړی;
                        mmctx->quant_ib_وروستی[ith]  = ib_وروستی;
                        mmctx->quant_r[ith]        = ib_لومړی / nb;
                        mmctx->quant_c[ith]        = ib_لومړی % nb;
                    }
                } وإلا {
                    quant_tasks = MIN(cur_m_لیکې, octx->n_تارونه);
                    q_func = quant_task_func;
                    mmctx->n_quant_لیکې_فی_thread = (cur_m_لیکې + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_کتار_پرلپسې(octx->ctx->work_کتار, q_func, mmctx, quant_tasks);
            } وإلا {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_پیل, cur_m_لیکې);
            }

            work_کتار_پرلپسې(octx->ctx->work_کتار, matmul_دنده_func, mmctx, octx->n_تارونه);
        }
    } وإلا {
        mmctx->cur_m_پیل = 0;
        mmctx->cur_m_لیکې  = act_nrows;

        إذا (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_لیکې_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_کتار_پرلپسې(octx->ctx->work_کتار, quant_task_func, mmctx, n_quant_tasks);
        } وإلا {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_کتار_پرلپسې(octx->ctx->work_کتار, matmul_دنده_func, mmctx, octx->n_تارونه);
    }

    أرجع HTP_حالت_OK;
}

ساكن فراغ dequantize_tiled_کارګر_loop(بینښه صحيح n, بینښه صحيح i, فراغ * معلومات) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) معلومات;
    بنية htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    لكل (بینښه صحيح task_id = i; task_id < (بینښه صحيح) state->n_tasks; task_id += n) {
        صحيح پیل = task_id * state->n_tiles_فی_task;
        صحيح پای_ټکی   = hex_smin(پیل + state->n_tiles_فی_task, state->n_tot_tiles);
        state->task_fn(state, پیل, پای_ټکی);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

ساكن فراغ transfer_output_chunk_کارګر_fn(بینښه صحيح n, بینښه صحيح i, فراغ *معلومات) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) معلومات;

    بنية htp_thread_trace * tr = &st->traces[i];

    صحيح پیل_chunk_idx = i * st->n_chunks_فی_task;
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_O_PROC, پیل_chunk_idx);

    لكل (بینښه صحيح task_id = i; task_id < (بینښه صحيح)st->n_tasks; task_id += n) {
        صحيح    chunk_idx  = task_id * st->n_chunks_فی_task;
        size_t chunk_اندازه = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_فی_task);

        عائم        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ثابت عائم  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : عدم;
        transfer_output_chunk_fp16_منزل_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_اندازه, st->n_ستون_شمېر, st->dst_stride, st->src2_stride, st->dst_ستون_شمېر);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, پیل_chunk_idx);
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
    size_t                          vtcm_f32_act_بایټونه_فی_thread;
    uint32_t                        dma_ګام_لیکې;
    uint32_t                        dma_ګام_لیکې_ښويدنه;
    uint32_t                        act_elem_اندازه;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

تعريفنوع بنية {
    بنية htp_context            * ctx;
    بنية htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    عائم                         * vtcm_f32_act;
    uint32_t                        n_لیکې;
    uint32_t                        k_بلاک;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_ستون_chunks;
    بنية fastdiv_ارزښتونه           n_تارونه_div;
    size_t                          vtcm_f32_act_بایټونه;
    uint32_t                        dma_ګام_لیکې;
    uint32_t                        dma_ګام_لیکې_ښويدنه;
} activation_transfer_ستون_chunk_state_t;

ساكن فراغ transfer_activation_chunk_fp32_منزل_fp16_dma_pipelined_ستون_chunk(
        dma_کتار *dma_q,
        __fp16 *محدود vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_لیکې,
        uint32_t k_بلاک,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_لومړی,
        uint32_t c_len,
        عائم *thread_f32_act,
        بنية htp_thread_trace *tr,
        uint32_t dma_ګام_لیکې,
        uint32_t dma_ګام_لیکې_ښويدنه) {

    ثابت uint32_t R = dma_ګام_لیکې;
    ثابت uint32_t n_لیکې_padded = hex_align_پورته(n_لیکې, HTP_MM_HMX_TILE_N_لیکې);

    ثابت uint32_t n_ګامونه = n_لیکې_padded >> dma_ګام_لیکې_ښويدنه;

    // Push step 0
    إذا (n_ګامونه > 0 && n_لیکې > 0) {
        uint32_t nrows_منزل_fetch = hex_smin(n_لیکې, R);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(thread_f32_act, act_dma_addr + (size_t) c_لومړی * حجم(عائم)),
                       c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_منزل_fetch);
    }
    // Push step 1
    إذا (n_ګامونه > 1) {
        uint32_t بل_r = R * 1;
        إذا (بل_r < n_لیکې) {
            uint32_t nrows_منزل_fetch = hex_smin(n_لیکې - بل_r, R);
            عائم *بل_buf = thread_f32_act + 1 * R * c_len;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(بل_buf, act_dma_addr + ((size_t) بل_r * k_stride + c_لومړی) * حجم(عائم)),
                           c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_منزل_fetch);
        }
    }
    لكل (uint32_t s = 0; s < n_ګامونه; ++s) {
        uint32_t r = s << dma_ګام_لیکې_ښويدنه;
        عائم *curr_buf = thread_f32_act;

        إذا (r < n_لیکې) {
            curr_buf = (عائم *) dma_کتار_راوباسه(dma_q).dst;
        }

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        لكل (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t لیکه_idx = r + (p << 1);
            عائم *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((لیکه_idx + 0) < n_لیکې);
            bool r1_valid = ((لیکه_idx + 1) < n_لیکې);

            transfer_activation_لیکه_pair_fp32_منزل_fp16_ستون_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, لیکه_idx, k_بلاک, c_لومړی, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t بل_s = s + 2;
        uint32_t بل_r = بل_s << dma_ګام_لیکې_ښويدنه;
        إذا (بل_r < n_لیکې) {
            uint32_t nrows_منزل_fetch = hex_smin(n_لیکې - بل_r, R);
            dma_کتار_ټېل(dma_q, dma_make_معلومات(curr_buf, act_dma_addr + ((size_t) بل_r * k_stride + c_لومړی) * حجم(عائم)),
                           c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_منزل_fetch);
        }
    }
}

ساكن فراغ transfer_activation_chunk_ستون_chunk_کارګر_fn(بینښه صحيح n, بینښه صحيح i, فراغ *معلومات) {
    activation_transfer_ستون_chunk_state_t *st = (activation_transfer_ستون_chunk_state_t *) معلومات;
    بنية htp_thread_trace * tr = &st->traces[i];

    uint32_t n_بلاکونه = st->k_بلاک / 32;
    uint32_t b_لومړی = fastdiv(n_بلاکونه * i, &st->n_تارونه_div);
    uint32_t b_وروستی  = fastdiv(n_بلاکونه * (i + 1), &st->n_تارونه_div);
    uint32_t c_لومړی = b_لومړی * 32;
    uint32_t c_وروستی = b_وروستی * 32;
    uint32_t c_len = c_وروستی - c_لومړی;

    إذا (c_len == 0) {
        أرجع;
    }

    uint32_t k_chunk_valid = 0;
    إذا (st->k_valid > c_لومړی) {
        k_chunk_valid = hex_smin(st->k_valid, c_وروستی) - c_لومړی;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_بایټونه = hex_align_ښکته(fastdiv(st->vtcm_f32_act_بایټونه, &st->n_تارونه_div), 128);
    عائم *thread_f32_act = (عائم *)((حرف *)st->vtcm_f32_act + i * thread_scratch_بایټونه);

    transfer_activation_chunk_fp32_منزل_fp16_dma_pipelined_ستون_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_لیکې, st->k_بلاک, st->k_stride, k_chunk_valid,
        c_لومړی, c_len, thread_f32_act, tr, st->dma_ګام_لیکې, st->dma_ګام_لیکې_ښويدنه
    );
}

ساكن فراغ transfer_activation_chunk_منزل_fp16_dma_pipelined(
        dma_کتار *dma_q,
        __fp16 *محدود vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_لیکې,
        uint32_t k_بلاک,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_اندازه,
        بنية htp_thread_trace *tr,
        uint32_t dma_ګام_لیکې,
        uint32_t dma_ګام_لیکې_ښويدنه) {

    ثابت uint32_t R = dma_ګام_لیکې;
    ثابت uint32_t n_لیکې_padded = hex_align_پورته(n_لیکې, HTP_MM_HMX_TILE_N_لیکې);

    ثابت uint32_t n_ګامونه = n_لیکې_padded >> dma_ګام_لیکې_ښويدنه;

    ثابت bool act_is_f16 = (act_elem_اندازه == حجم(__fp16));
    ثابت size_t لیکه_بایټونه = (size_t) k_بلاک * act_elem_اندازه;  // staging row (DMA dst stride)
    ثابت size_t src_لیکه_بایټونه = (size_t) k_stride * act_elem_اندازه;  // DDR row (DMA src stride)
    ثابت size_t پلنوالی_بایټونه = (size_t) k_valid * act_elem_اندازه;

    // Push step 0
    إذا (n_ګامونه > 0 && n_لیکې > 0) {
        uint32_t nrows_منزل_fetch = hex_smin(n_لیکې, R);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(thread_act, act_dma_addr),
                       لیکه_بایټونه, src_لیکه_بایټونه, پلنوالی_بایټونه, nrows_منزل_fetch);
    }
    // Push step 1 (if valid)
    إذا (n_ګامونه > 1) {
        uint32_t بل_r = R * 1;
        إذا (بل_r < n_لیکې) {
            uint32_t nrows_منزل_fetch = hex_smin(n_لیکې - بل_r, R);
            uint8_t *بل_buf = thread_act + 1 * R * لیکه_بایټونه;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(بل_buf, act_dma_addr + (size_t) بل_r * src_لیکه_بایټونه),
                           لیکه_بایټونه, src_لیکه_بایټونه, پلنوالی_بایټونه, nrows_منزل_fetch);
        }
    }
    لكل (uint32_t s = 0; s < n_ګامونه; ++s) {
        uint32_t r = s << dma_ګام_لیکې_ښويدنه;
        uint8_t *curr_buf = thread_act;

        إذا (r < n_لیکې) {
            curr_buf = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;
        }

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        إذا (act_is_f16) {
            لكل (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t لیکه_idx = r + (p << 1);
                ثابت __fp16 *pair_buf = (ثابت __fp16 *) (curr_buf + (p << 1) * لیکه_بایټونه);
                bool r0_valid = ((لیکه_idx + 0) < n_لیکې);
                bool r1_valid = ((لیکه_idx + 1) < n_لیکې);

                transfer_activation_لیکه_pair_f16_منزل_f16(vtcm_dst, pair_buf, pair_buf + k_بلاک, لیکه_idx, k_بلاک, k_valid, r0_valid, r1_valid);
            }
        } وإلا {
            لكل (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t لیکه_idx = r + (p << 1);
                ثابت عائم *pair_buf = (ثابت عائم *) (curr_buf + (p << 1) * لیکه_بایټونه);
                bool r0_valid = ((لیکه_idx + 0) < n_لیکې);
                bool r1_valid = ((لیکه_idx + 1) < n_لیکې);

                transfer_activation_لیکه_pair_fp32_منزل_fp16(vtcm_dst, pair_buf, pair_buf + k_بلاک, لیکه_idx, k_بلاک, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t بل_s = s + 2;
        uint32_t بل_r = بل_s << dma_ګام_لیکې_ښويدنه;
        إذا (بل_r < n_لیکې) {
            uint32_t nrows_منزل_fetch = hex_smin(n_لیکې - بل_r, R);
            dma_کتار_ټېل(dma_q, dma_make_معلومات(curr_buf, act_dma_addr + (size_t) بل_r * src_لیکه_بایټونه),
                           لیکه_بایټونه, src_لیکه_بایټونه, پلنوالی_بایټونه, nrows_منزل_fetch);
        }
    }
}

ساكن فراغ transfer_activation_chunk_کارګر_fn(بینښه صحيح n, بینښه صحيح i, فراغ *معلومات) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) معلومات;

    بنية htp_thread_trace * tr = &st->traces[i];

    لكل (بینښه صحيح task_id = i; task_id < (بینښه صحيح)st->n_tasks; task_id += n) {
        صحيح    chunk_idx  = task_id * st->n_chunks_فی_task;
        size_t chunk_اندازه = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_فی_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_بلاک;
        ثابت dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_اندازه;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_بایټونه_فی_thread;
        transfer_activation_chunk_منزل_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_اندازه, st->k_بلاک, st->k_stride, st->k_valid, thread_act, st->act_elem_اندازه, tr, st->dma_ګام_لیکې, st->dma_ګام_لیکې_ښويدنه
        );
    }
}

تعريفنوع بنية {
    بنية htp_thread_trace       * traces;
    ثابت بنية mmid_لیکه_mapping * ماتریکس_لیکې;
    __fp16                        * dst;
    ثابت عائم                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_فی_task;
    uint32_t                        k_بلاک;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    بنية fastdiv_ارزښتونه           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        پیل_لیکه;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

تعريفنوع بنية {
    بنية htp_thread_trace       * traces;
    ثابت بنية mmid_لیکه_mapping * ماتریکس_لیکې;
    ثابت __fp16                  * vtcm_src;
    عائم                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_فی_task;
    uint32_t                        n_ستون_شمېر;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        پیل_لیکه;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

ساكن فراغ transfer_activation_chunk_gathered_کارګر_fn(بینښه صحيح n, بینښه صحيح i, فراغ *معلومات) {
    activation_transfer_gathered_task_state_t *st = معلومات;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx      = i;
    صحيح chunk_اندازه     = st->n_chunks_فی_task;
    صحيح vtcm_پیل_لیکه = chunk_idx * chunk_اندازه;
    صحيح پیل_لیکه      = st->پیل_لیکه + vtcm_پیل_لیکه;
    صحيح n_لیکې         = hex_smin(st->cne1 - پیل_لیکه, chunk_اندازه);
    إذا (n_لیکې > 0) {
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_منزل_fp16_gathered(
            st->dst, st->src, پیل_لیکه, vtcm_پیل_لیکه, n_لیکې, st->k_بلاک,
            st->ماتریکس_لیکې, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ساكن فراغ transfer_activation_chunk_gathered_کارګر_flat_fn(بینښه صحيح n, بینښه صحيح i, فراغ *معلومات) {
    activation_transfer_gathered_task_state_t *st = معلومات;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx = i;
    صحيح chunk_اندازه = st->n_chunks_فی_task;
    صحيح vtcm_پیل_لیکه = chunk_idx * chunk_اندازه;
    صحيح پیل_لیکه = st->پیل_لیکه + vtcm_پیل_لیکه;
    صحيح n_لیکې = hex_smin(st->cne1 - پیل_لیکه, chunk_اندازه);
    إذا (n_لیکې > 0) {
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_منزل_fp16_gathered_flat(
            st->dst, st->src, پیل_لیکه, vtcm_پیل_لیکه, n_لیکې, st->k_بلاک,
            st->ماتریکس_لیکې, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ساكن فراغ transfer_output_chunk_scattered_کارګر_fn(بینښه صحيح n, بینښه صحيح i, فراغ *معلومات) {
    output_transfer_scattered_task_state_t *st = معلومات;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx = i;
    صحيح chunk_اندازه = st->n_chunks_فی_task;
    صحيح vtcm_پیل_لیکه = chunk_idx * chunk_اندازه;
    صحيح پیل_لیکه = st->پیل_لیکه + vtcm_پیل_لیکه;
    صحيح n_لیکې = hex_smin(st->cne1 - پیل_لیکه, chunk_اندازه);
    إذا (n_لیکې > 0) {
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_منزل_fp32_scattered(
            st->dst, st->vtcm_src, پیل_لیکه, vtcm_پیل_لیکه, n_لیکې, st->n_ستون_شمېر,
            st->ماتریکس_لیکې, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

ساكن مضمن tiled_dequant_task_fn_t hmx_get_dequant_کارګر_fn(صحيح وزن_type, bool is_tiled) {
    بدل (وزن_type) {
        حالة HTP_TYPE_Q4_0:   أرجع dequantize_tiled_وزن_منزل_fp16_task_q4_0;
        حالة HTP_TYPE_IQ4_NL: أرجع dequantize_tiled_وزن_منزل_fp16_task_iq4_nl;
        حالة HTP_TYPE_Q4_1:
        حالة HTP_TYPE_Q4_K:   أرجع dequantize_tiled_وزن_منزل_fp16_task_q4_1;
        حالة HTP_TYPE_MXFP4:  أرجع dequantize_tiled_وزن_منزل_fp16_task_mxfp4;
        حالة HTP_TYPE_Q8_0:   أرجع dequantize_tiled_وزن_منزل_fp16_task_q8_0;
        حالة HTP_TYPE_Q5_K:   أرجع dequantize_tiled_وزن_منزل_fp16_task_q5_k;
        حالة HTP_TYPE_Q6_K:   أرجع dequantize_tiled_وزن_منزل_fp16_task_q6_k;
        حالة HTP_TYPE_Q3_K:   أرجع dequantize_tiled_وزن_منزل_fp16_task_q3_k;
        حالة HTP_TYPE_Q2_K:   أرجع dequantize_tiled_وزن_منزل_fp16_task_q2_k;
        حالة HTP_TYPE_Q1_0:   أرجع dequantize_tiled_وزن_منزل_fp16_task_q1_0;
        حالة HTP_TYPE_Q2_0:   أرجع dequantize_tiled_وزن_منزل_fp16_task_q2_0;
        حالة HTP_TYPE_F16:    أرجع is_tiled ? عدم : convert_f16_وزن_منزل_fp16_tiles_task;
        حالة HTP_TYPE_BF16:   أرجع convert_bf16_وزن_منزل_fp16_tiles_task;
        حالة HTP_TYPE_F32:    أرجع quantize_f32_وزن_منزل_fp16_tiles_task;
        افتراضي:              أرجع عدم;
    }
}

ساكن فراغ dequantize_tiled_وزن_chunk_منزل_fp16_tiles(
        بنية htp_context *ctx, __fp16 *vtcm_dst,
        ثابت فراغ *وزن_src_ddr,
        صحيح n_ستون_شمېر, صحيح k_بلاک,
        size_t لیکه_stride, صحيح وزن_type,
        صحيح n_k_tiles, بنية fastdiv_ارزښتونه n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_کارګر_fn, صحيح n_تارونه) {

    إذا (!dequant_کارګر_fn) {
        أرجع;
    }

    أكد(k_بلاک % HTP_MM_HMX_TILE_N_ستون_شمېر == 0);

    ثابت size_t n_ستون_tiles = hmx_ceil_div(n_ستون_شمېر, HTP_MM_HMX_TILE_N_ستون_شمېر);
    ثابت uint32_t n_وزن_tiles = k_بلاک / htp_tensor_get_tile_k(وزن_type);
    ثابت size_t n_tot_tiles = n_ستون_tiles * n_وزن_tiles;

    size_t n_tiles_فی_task = (n_تارونه == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_تارونه);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_فی_task - 1) / n_tiles_فی_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_فی_task = n_tiles_فی_task;
    state.dst              = vtcm_dst;
    state.src              = (ثابت uint8_t *)وزن_src_ddr;
    state.task_fn          = dequant_کارګر_fn;
    state.n_ستون_شمېر           = n_ستون_شمېر;
    state.k_بلاک          = k_بلاک;
    state.لیکه_stride       = لیکه_stride;
    state.وزن_type      = وزن_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_وزن_tiles   = n_وزن_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_اندازه = htp_tensor_get_tile_اندازه(وزن_type);
    state.aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(وزن_type);

    إذا (state.n_tasks == 1 || n_تارونه == 1) {
        dequantize_tiled_کارګر_loop(1, 0, &state);
    } وإلا {
        صحيح n_tasks = hex_smin((صحيح) state.n_tasks, n_تارونه);
        کارګر_pool_پرلپسې_func(ctx->کارګر_pool, dequantize_tiled_کارګر_loop, &state, n_tasks);
    }
}

تعريفنوع بنية {
    بنية htp_context      * ctx;
    بنية htp_thread_trace * traces;
    عائم                   * dst;
    ثابت __fp16            * vtcm_src;
    ثابت عائم             * src2;
    uint32_t                  n_لیکې;
    uint32_t                  n_ستون_شمېر;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_ستون_شمېر;
    بنية fastdiv_ارزښتونه     n_تارونه_div;
} output_transfer_ستون_chunk_state_t;

ساكن فراغ transfer_output_chunk_ستون_chunk_کارګر_fn(بینښه صحيح n, بینښه صحيح i, فراغ *معلومات) {
    (فراغ) n;
    output_transfer_ستون_chunk_state_t *st = (output_transfer_ستون_chunk_state_t *) معلومات;
    بنية htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_بلاکونه = hmx_ceil_div(st->n_ستون_شمېر, 32);
    uint32_t b_لومړی  = fastdiv(n_بلاکونه * i, &st->n_تارونه_div);
    uint32_t b_وروستی   = fastdiv(n_بلاکونه * (i + 1), &st->n_تارونه_div);
    uint32_t c_لومړی  = b_لومړی * 32;
    uint32_t c_وروستی   = b_وروستی * 32;
    uint32_t c_len    = c_وروستی - c_لومړی;

    إذا (c_len == 0) أرجع;

    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_O_PROC, c_لومړی);

    ثابت __fp16 *vtcm_src = st->vtcm_src + b_لومړی * HTP_MM_HMX_TILE_N_ELMS;
    ثابت عائم *src2      = st->src2 ? (st->src2 + c_لومړی) : عدم;
    عائم *dst             = st->dst + c_لومړی;

    صحيح chunk_dst_ستون_شمېر = (صحيح)st->dst_ستون_شمېر - (صحيح)c_لومړی;
    إذا (chunk_dst_ستون_شمېر > 0) {
        transfer_output_chunk_fp16_منزل_fp32_ستون_chunk(
            dst, src2, vtcm_src, 0, st->n_لیکې, c_len, st->n_ستون_شمېر,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_ستون_شمېر
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_لومړی);
}

ساكن فراغ transfer_output_chunk_threaded(بنية htp_context *ctx, عائم *dst, ثابت عائم *src2, ثابت __fp16 *vtcm_src,
                                              صحيح n_لیکې, صحيح n_ستون_شمېر, صحيح dst_stride, uint32_t src2_stride, صحيح dst_ستون_شمېر, صحيح n_تارونه) {
    إذا (n_لیکې <= 0) أرجع;

    uint32_t n_بلاکونه = hmx_ceil_div((uint32_t) n_ستون_شمېر, 32);
    إذا (n_تارونه > 1 && n_بلاکونه >= (uint32_t)n_تارونه) {
        بنية fastdiv_ارزښتونه n_تارونه_div = (n_تارونه == (صحيح)ctx->n_تارونه) ? ctx->n_تارونه_div : init_fastdiv_ارزښتونه(n_تارونه);
        output_transfer_ستون_chunk_state_t ستون_state;
        ستون_state.dst = dst;
        ستون_state.src2 = src2;
        ستون_state.vtcm_src = vtcm_src;
        ستون_state.n_لیکې = (uint32_t)n_لیکې;
        ستون_state.n_ستون_شمېر = (uint32_t)n_ستون_شمېر;
        ستون_state.dst_stride = (uint32_t)dst_stride;
        ستون_state.src2_stride = src2_stride;
        ستون_state.dst_ستون_شمېر = (uint32_t)dst_ستون_شمېر;
        ستون_state.n_تارونه_div = n_تارونه_div;
        ستون_state.traces = ctx->trace;
        ستون_state.ctx = ctx;

        کارګر_pool_پرلپسې_func(ctx->کارګر_pool, transfer_output_chunk_ستون_chunk_کارګر_fn, &ستون_state, n_تارونه);
        أرجع;
    }

    size_t n_tot_chunks      = n_لیکې;
    size_t n_chunks_فی_task = (n_تارونه == 1) ? n_tot_chunks : hmx_ceil_div(n_لیکې, n_تارونه);
    n_chunks_فی_task        = hex_align_پورته(n_chunks_فی_task, 2);

    صحيح actual_تارونه = hmx_ceil_div(n_لیکې, n_chunks_فی_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_تارونه;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_فی_task = n_chunks_فی_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_ستون_شمېر            = n_ستون_شمېر;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_ستون_شمېر          = dst_ستون_شمېر;
    state.traces            = ctx->trace;

    إذا (actual_تارونه <= 1) {
        transfer_output_chunk_کارګر_fn(1, 0, &state);
    } وإلا {
        کارګر_pool_پرلپسې_func(ctx->کارګر_pool, transfer_output_chunk_کارګر_fn, &state, actual_تارونه);
    }
}

بنية activation_transfer_params {
    بنية htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    صحيح                           n_لیکې;
    صحيح                           k_بلاک;
    صحيح                           k_stride;
    صحيح                           n_تارونه;
    ثابت بنية fastdiv_ارزښتونه * act_تارونه_div;
    ثابت بنية fastdiv_ارزښتونه * k_div;
    صحيح                           k_valid;
    عائم *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_بایټونه;
    uint32_t                      act_elem_اندازه;  // 4=F32 src1, 2=F16 src1
};

ساكن فراغ transfer_activation_chunk_threaded(ثابت بنية activation_transfer_params * params) {
    بنية htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ثابت dma_addr_t              act_dma_addr       = params->act_dma_addr;
    صحيح                           n_لیکې             = params->n_لیکې;
    صحيح                           k_بلاک            = params->k_بلاک;
    صحيح                           k_stride           = params->k_stride;
    صحيح                           n_تارونه          = params->n_تارونه;
    ثابت بنية fastdiv_ارزښتونه * act_تارونه_div    = params->act_تارونه_div;
    ثابت بنية fastdiv_ارزښتونه * k_div              = params->k_div;
    صحيح                           k_valid            = params->k_valid;
    عائم *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_بایټونه = params->vtcm_f32_act_بایټونه;
    // element size of the activation rows (4 = F32, 2 = F16).
    ثابت uint32_t                act_elem_اندازه      = params->act_elem_اندازه ? params->act_elem_اندازه : (uint32_t) حجم(عائم);

    إذا (n_لیکې <= 0) {
        أرجع;
    }

    ثابت size_t n_tasks = (n_لیکې + 31) >> 5;
    إذا (act_elem_اندازه == حجم(عائم) && n_تارونه > 1 && k_بلاک > 32 && n_tasks < (size_t) n_تارونه) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_ګام_لیکې = 2;
        uint32_t dma_ګام_لیکې_ښويدنه = 1;
        إذا (vtcm_f32_act_بایټونه > 0) {
            size_t thread_scratch_بایټونه = hex_align_ښکته(fastdiv(vtcm_f32_act_بایټونه, act_تارونه_div), 128);
            size_t thread_scratch_elements = thread_scratch_بایټونه / حجم(عائم);
            size_t dma_ګام_لیکې_max = fastdiv(thread_scratch_elements / 2, k_div);
            إذا (dma_ګام_لیکې_max >= 4) {
                dma_ګام_لیکې = 4;
                dma_ګام_لیکې_ښويدنه = 2;
            }
        }

        activation_transfer_ستون_chunk_state_t ستون_state;
        ستون_state.dst = dst;
        ستون_state.act_dma_addr = act_dma_addr;
        ستون_state.n_لیکې = n_لیکې;
        ستون_state.k_بلاک = k_بلاک;
        ستون_state.k_stride = k_stride;
        ستون_state.k_valid = k_valid;
        ستون_state.n_ستون_chunks = n_تارونه;
        ستون_state.n_تارونه_div = *act_تارونه_div;
        ستون_state.vtcm_f32_act = vtcm_f32_act;
        ستون_state.vtcm_f32_act_بایټونه = vtcm_f32_act_بایټونه;
        ستون_state.traces = ctx->trace;
        ستون_state.ctx = ctx;
        ستون_state.dma_ګام_لیکې = dma_ګام_لیکې;
        ستون_state.dma_ګام_لیکې_ښويدنه = dma_ګام_لیکې_ښويدنه;

        کارګر_pool_پرلپسې_func(ctx->کارګر_pool, transfer_activation_chunk_ستون_chunk_کارګر_fn, &ستون_state, n_تارونه);
        أرجع;
    }

    أكد(k_بلاک % HTP_MM_HMX_TILE_N_ستون_شمېر == 0 && k_stride % HTP_MM_HMX_TILE_N_ستون_شمېر == 0);

    size_t n_tot_chunks      = n_لیکې;
    size_t n_chunks_فی_task = (n_تارونه == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_تارونه == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
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
    state.act_elem_اندازه = act_elem_اندازه;

    state.vtcm_f32_act_بایټونه_فی_thread = hex_align_ښکته(fastdiv(vtcm_f32_act_بایټونه, act_تارونه_div), 128);

    uint32_t dma_ګام_لیکې = 2;
    uint32_t dma_ګام_لیکې_ښويدنه = 1;
    إذا (state.vtcm_f32_act_بایټونه_فی_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_بایټونه_فی_thread / حجم(عائم);
        size_t dma_ګام_لیکې_max = fastdiv(thread_scratch_elements / 2, k_div);
        إذا (dma_ګام_لیکې_max >= 4) {
            dma_ګام_لیکې = 4;
            dma_ګام_لیکې_ښويدنه = 2;
        }
    }
    state.dma_ګام_لیکې      = dma_ګام_لیکې;
    state.dma_ګام_لیکې_ښويدنه = dma_ګام_لیکې_ښويدنه;

    صحيح فعال_تارونه = hex_smin(n_تارونه, (صحيح)state.n_tasks);
    إذا (state.n_tasks == 1 || n_تارونه == 1) {
        transfer_activation_chunk_کارګر_fn(1, 0, &state);
    } وإلا {
        کارګر_pool_پرلپسې_func(ctx->کارګر_pool, transfer_activation_chunk_کارګر_fn, &state, فعال_تارونه);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

تعريفنوع بنية {
    __fp16 *       output;
    ثابت __fp16 * activation;
    ثابت __fp16 * وزن;
    ثابت __fp16 * scales;
    uint32_t       n_لیکه_tiles;
    uint32_t       n_ستون_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_دنده_t;

ساكن فراغ hmx_matmul_کارګر_fn(فراغ * معلومات) {
    hmx_matmul_دنده_t * دنده = (hmx_matmul_دنده_t *) معلومات;
    FARF(لوړ_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", دنده->n_لیکه_tiles, دنده->n_ستون_tiles, دنده->n_dot_tiles);
    core_dot_chunk_fp16(دنده->output, دنده->activation, دنده->وزن, دنده->scales, دنده->n_لیکه_tiles, دنده->n_ستون_tiles, دنده->n_dot_tiles);
}

ساكن مضمن فراغ hmx_matmul_دنده_init(hmx_matmul_دنده_t * دنده,
                                       __fp16 *           output,
                                       ثابت __fp16 *     activation,
                                       ثابت __fp16 *     وزن,
                                       ثابت __fp16 *     scales,
                                       uint32_t           n_لیکه_tiles,
                                       uint32_t           n_ستون_tiles,
                                       uint32_t           n_dot_tiles) {
    دنده->output      = output;
    دنده->activation  = activation;
    دنده->وزن      = وزن;
    دنده->scales      = scales;
    دنده->n_لیکه_tiles = n_لیکه_tiles;
    دنده->n_ستون_tiles = n_ستون_tiles;
    دنده->n_dot_tiles = n_dot_tiles;
}

ساكن صحيح hmx_mm_2d_f32(بنية htp_context *ctx,
                                  dma_کتار *وزن_dma,
                                  عائم *محدود dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_بایټونه,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t وزن,
                                  صحيح m, صحيح k, صحيح n,
                                  صحيح act_stride,
                                  uint32_t act_elem_اندازه,  // 4=F32 src1, 2=F16 src1
                                  صحيح وزن_stride,
                                  صحيح وزن_type,
                                  صحيح k_valid,
                                  صحيح dst_stride,
                                  uint32_t src2_stride,
                                  صحيح dst_ستون_شمېر,
                                  صحيح m_chunk,
                                  صحيح n_chunk,
                                  صحيح pipeline,
                                  صحيح n_تارونه,
                                  صحيح act_تارونه,
                                  ثابت بنية fastdiv_ارزښتونه * act_تارونه_div,
                                  ثابت بنية fastdiv_ارزښتونه * k_div,
                                  صحيح tile_اندازه,
                                  صحيح aligned_tile_اندازه,
                                  صحيح vtcm_اندازه) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت bool is_tiled = (tile_اندازه > 0);

    إذا (k % 32 != 0) {
        أرجع -1;
    }
    إذا (!hex_is_aligned(dst, حجم(عائم))) { أرجع -1; }

    size_t لیکه_stride = htp_tensor_get_لیکه_اندازه(وزن_type, k);
    إذا (لیکه_stride == 0) {
        أرجع -1;
    }

    tiled_dequant_task_fn_t dequant_کارګر_fn = hmx_get_dequant_کارګر_fn(وزن_type, is_tiled);

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_ستون_شمېر;
    ثابت صحيح n_وزن_tiles = k / htp_tensor_get_tile_k(وزن_type);
    ثابت بنية fastdiv_ارزښتونه n_k_tiles_div = init_fastdiv_ارزښتونه(n_k_tiles);

    ثابت size_t vec_dot_اندازه = k * حجم(__fp16);
    ثابت size_t vtcm_budget  = ctx->vtcm_اندازه;

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_اندازه : لیکه_stride;
    ثابت uint32_t dma_src_stride  = is_tiled ? tile_اندازه : وزن_stride;
    ثابت uint32_t dma_پلنوالی_بایټونه = is_tiled ? tile_اندازه : لیکه_stride;

    size_t m_chunk_n_لیکې = m_chunk;
    size_t n_chunk_n_ستون_شمېر = n_chunk;
    size_t vtcm_کارول_شوی      = vtcm_اندازه;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, وزن_type, k, m_chunk_n_لیکې, n_chunk_n_ستون_شمېر, 1, pipeline, act_تارونه, aligned_tile_اندازه, src2_بایټونه);

    vtcm_کارول_شوی = L.ټول_بایټونه;
    إذا (vtcm_کارول_شوی > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_کارول_شوی, vtcm_budget, m, k, n, m_chunk_n_لیکې, n_chunk_n_ستون_شمېر);
        أرجع -1;
    }

    uint8_t * ثابت بنسټ = (uint8_t *) ctx->vtcm_بنسټ;
    __fp16  *vtcm_وزن_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_وزن[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنسټ, L.off_وزن[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_act);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, بنسټ, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, بنسټ, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, بنسټ, L.off_scratch[1], pipeline);
    فراغ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, بنسټ, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_scales);

    hmx_init_ستنه_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ثابت bool has_bias      = (src2_بایټونه > 0 && src2_addr != 0);
    عائم   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(عائم, بنسټ, L.off_bias, has_bias);
    إذا (has_bias) {
        dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_bias, src2_addr), hex_align_پورته(src2_بایټونه, 128), 0, src2_بایټونه, 1);
        dma_کتار_راوباسه(وزن_dma);
    }

    FARF(لوړ_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, وزن_type, m_chunk_n_لیکې, n_chunk_n_ستون_شمېر, vtcm_کارول_شوی, vtcm_budget);

    صحيح n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_ستون_شمېر);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    إذا (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_دنده_t دنده_slots[2];  // persistent double-buffered job descriptors
        فراغ *vtcm_raw_bufs[4] = {
            vtcm_وزن_raw[0],
            vtcm_وزن_raw[1],
            dequant_کارګر_fn ? vtcm_وزن_raw[0] : vtcm_scratch0,
            dequant_کارګر_fn ? vtcm_وزن_raw[1] : vtcm_scratch1,
        };
        فراغ *vtcm_وزن_bufs[4] = {
            dequant_کارګر_fn ? vtcm_scratch0 : vtcm_وزن_raw[0],
            dequant_کارګر_fn ? vtcm_scratch1 : vtcm_وزن_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        فراغ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        لكل (size_t mr = 0; mr < m; mr += m_chunk_n_لیکې) {
            ثابت size_t n_لیکې = hex_smin(m - mr, m_chunk_n_لیکې);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_اندازه,  // byte offset (F16/F32)
                .n_لیکې = (صحيح) n_لیکې,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_تارونه = act_تارونه,
                .act_تارونه_div = act_تارونه_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بایټونه = L.act_f32_بایټونه,
                .act_elem_اندازه = act_elem_اندازه,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            ثابت size_t   n_ستون_شمېر_A0 = hex_smin(n - 0 * n_chunk_n_ستون_شمېر, n_chunk_n_ستون_شمېر);
            ثابت uint32_t لوړوالی_A0 = is_tiled ? hmx_ceil_div(n_ستون_شمېر_A0, 32) * n_وزن_tiles : n_ستون_شمېر_A0;
            dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[0], وزن),
                           dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی_A0);

            إذا (1 < n_chunk_cnt) {
                ثابت size_t   n_ستون_شمېر_A1 = hex_smin(n - 1 * n_chunk_n_ستون_شمېر, n_chunk_n_ستون_شمېر);
                ثابت uint32_t لوړوالی_A1 = is_tiled ? hmx_ceil_div(n_ستون_شمېر_A1, 32) * n_وزن_tiles : n_ستون_شمېر_A1;
                dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[1], وزن + n_chunk_n_ستون_شمېر * وزن_stride),
                               dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                ثابت size_t nc    = i * n_chunk_n_ستون_شمېر;
                ثابت size_t nc_p2 = nc + 2 * n_chunk_n_ستون_شمېر;

                ثابت size_t n_ستون_شمېر    = hex_smin(n - nc, n_chunk_n_ستون_شمېر);
                ثابت size_t n_ستون_شمېر_p2 = hex_smin(n - nc_p2, n_chunk_n_ستون_شمېر);

                // 1. pop A_i
                فراغ * curr_raw = (فراغ *) dma_کتار_راوباسه(وزن_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_وزن_chunk_منزل_fp16_tiles(
                    ctx, vtcm_وزن_bufs[i % 4], curr_raw,
                    n_ستون_شمېر, k, لیکه_stride, وزن_type,
                    n_k_tiles, n_k_tiles_div, dequant_کارګر_fn, n_تارونه);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                إذا (i + 2 < n_chunk_cnt) {
                    ثابت uint32_t لوړوالی_p2 = is_tiled ? hmx_ceil_div(n_ستون_شمېر_p2, 32) * n_وزن_tiles : n_ستون_شمېر_p2;
                    dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[(i + 2) % 4], وزن + nc_p2 * وزن_stride),
                                   dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی_p2);
                }

                // 4. submit C_i
                hmx_matmul_دنده_init(&دنده_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_لیکې, HTP_MM_HMX_TILE_N_لیکې),
                                    hmx_ceil_div(n_ستون_شمېر, HTP_MM_HMX_TILE_N_ستون_شمېر), k / HTP_MM_HMX_TILE_N_لیکې);
                hmx_کتار_ټېل(ctx->hmx_کتار, hmx_کتار_make_desc(hmx_matmul_کارګر_fn, &دنده_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                إذا (i > 0) {
                    hmx_کتار_راوباسه(ctx->hmx_کتار);
                    ثابت size_t nc_prev = (i - 1) * n_chunk_n_ستون_شمېر;
                    ثابت size_t n_ستون_شمېر_prev = hex_smin(n - nc_prev, n_chunk_n_ستون_شمېر);
                    عائم *output_chunk = dst + (mr * dst_stride + nc_prev);
                    ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : عدم;
                    صحيح chunk_dst_ستون_شمېر = dst_ستون_شمېر - (صحيح)nc_prev;
                    إذا (chunk_dst_ستون_شمېر > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_لیکې, n_ستون_شمېر_prev, dst_stride, src2_stride, chunk_dst_ستون_شمېر, n_تارونه);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_کتار_راوباسه(ctx->hmx_کتار);
            ثابت size_t nc_وروستی = (n_chunk_cnt - 1) * n_chunk_n_ستون_شمېر;
            ثابت size_t n_ستون_شمېر_وروستی = hex_smin(n - nc_وروستی, n_chunk_n_ستون_شمېر);
            عائم *output_chunk = dst + (mr * dst_stride + nc_وروستی);
            ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_وروستی) : عدم;
            صحيح chunk_dst_ستون_شمېر = dst_ستون_شمېر - (صحيح)nc_وروستی;
            إذا (chunk_dst_ستون_شمېر > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_لیکې, n_ستون_شمېر_وروستی, dst_stride, src2_stride, chunk_dst_ستون_شمېر, n_تارونه);
            }
        }
    } وإلا {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_دنده_t دنده;
        فراغ *vtcm_raw_bufs[2] = { vtcm_وزن_raw[0], dequant_کارګر_fn ? vtcm_وزن_raw[0] : vtcm_scratch0 };
        فراغ *vtcm_وزن_bufs[2] = { dequant_کارګر_fn ? vtcm_scratch0 : vtcm_وزن_raw[0], vtcm_scratch0 };

        لكل (size_t mr = 0; mr < m; mr += m_chunk_n_لیکې) {
            ثابت size_t n_لیکې = hex_smin(m - mr, m_chunk_n_لیکې);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_اندازه,  // byte offset (F16/F32)
                .n_لیکې = (صحيح) n_لیکې,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_تارونه = act_تارونه,
                .act_تارونه_div = act_تارونه_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بایټونه = L.act_f32_بایټونه,
                .act_elem_اندازه = act_elem_اندازه,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            إذا (n > 0) {
                ثابت size_t n_ستون_شمېر = hex_smin(n, n_chunk_n_ستون_شمېر);
                ثابت uint32_t لوړوالی = is_tiled ? hmx_ceil_div(n_ستون_شمېر, 32) * n_وزن_tiles : n_ستون_شمېر;
                dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[0], وزن), dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی);
            }

            لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                ثابت size_t nc = (size_t) i * n_chunk_n_ستون_شمېر;
                ثابت size_t n_ستون_شمېر = hex_smin(n - nc, n_chunk_n_ستون_شمېر);
                ثابت size_t n_لیکه_tiles = hmx_ceil_div(n_لیکې, HTP_MM_HMX_TILE_N_لیکې);
                ثابت size_t n_ستون_tiles = hmx_ceil_div(n_ستون_شمېر, HTP_MM_HMX_TILE_N_ستون_شمېر);

                // A: Wait for weight DMA
                فراغ * curr_raw = (فراغ *) dma_کتار_راوباسه(وزن_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_وزن_chunk_منزل_fp16_tiles(
                    ctx, vtcm_وزن_bufs[i % 2], curr_raw,
                    n_ستون_شمېر, k, لیکه_stride, وزن_type,
                    n_k_tiles, n_k_tiles_div, dequant_کارګر_fn, n_تارونه);

                // Start weight DMA for the next chunk early
                ثابت size_t nc_بل = nc + n_chunk_n_ستون_شمېر;
                إذا (nc_بل < n) {
                    ثابت size_t n_ستون_شمېر_بل = hex_smin(n - nc_بل, n_chunk_n_ستون_شمېر);
                    ثابت uint32_t لوړوالی_بل = is_tiled ? hmx_ceil_div(n_ستون_شمېر_بل, 32) * n_وزن_tiles : n_ستون_شمېر_بل;
                    dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[(i + 1) % 2], وزن + nc_بل * وزن_stride), dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی_بل);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_دنده_init(&دنده, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 2], vtcm_scales, n_لیکه_tiles, n_ستون_tiles, k / HTP_MM_HMX_TILE_N_لیکې);
                hmx_کتار_ټېل(ctx->hmx_کتار, hmx_کتار_make_desc(hmx_matmul_کارګر_fn, &دنده));
                hmx_کتار_راوباسه(ctx->hmx_کتار);

                // D: Output Store
                عائم *output_chunk = dst + (mr * dst_stride + nc);
                ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : عدم;
                صحيح chunk_dst_ستون_شمېر = dst_ستون_شمېر - (صحيح)nc;
                إذا (chunk_dst_ستون_شمېر > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_لیکې, n_ستون_شمېر, dst_stride, src2_stride, chunk_dst_ستون_شمېر, n_تارونه);
                }
            }
        }
    }

    أرجع 0;
}

ساكن صحيح hmx_mm_nx_2d_f32(بنية htp_ops_context * octx, ثابت بنية htp_mm_kernel_params * kparams) {
    بنية htp_context * ctx = octx->ctx;
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت uint32_t n_weights = kparams->n_weights;
    إذا (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];

    ثابت صحيح وزن_type = (صحيح) src0->type;
    ثابت صحيح k           = (صحيح) act->ne[0];
    ثابت صحيح k_valid     = (صحيح) act->ne[0];
    ثابت صحيح m           = (صحيح) (act->ne[1] * act->ne[2] * act->ne[3]);
    ثابت uint32_t act_elem_اندازه = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت صحيح act_stride  = (صحيح) (act->nb[1] / act_elem_اندازه);
    ثابت dma_addr_t act_dma_addr = act->معلومات;

    إذا (k % 32 != 0) { أرجع HTP_حالت_NO_SUPPORT; }

    size_t لیکه_stride = htp_tensor_get_لیکه_اندازه(وزن_type, k);
    إذا (لیکه_stride == 0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_ستون_شمېر;
    ثابت صحيح n_وزن_tiles = k / htp_tensor_get_tile_k(وزن_type);
    ثابت بنية fastdiv_ارزښتونه n_k_tiles_div = init_fastdiv_ارزښتونه(n_k_tiles);

    ثابت صحيح tile_اندازه       = kparams->tile_اندازه;
    ثابت صحيح aligned_tile_اندازه = kparams->aligned_tile_اندازه;

    ثابت bool is_tiled       = (tile_اندازه > 0);
    tiled_dequant_task_fn_t dequant_کارګر_fn = hmx_get_dequant_کارګر_fn(وزن_type, is_tiled);
    ثابت size_t vtcm_budget  = ctx->vtcm_اندازه;

    ثابت صحيح m_chunk_n_لیکې  = kparams->m_chunk;
    ثابت صحيح n_chunk_n_ستون_شمېر  = kparams->n_chunk;
    ثابت صحيح pipeline        = kparams->pipeline;
    ثابت صحيح n_تارونه       = octx->n_تارونه;
    ثابت صحيح act_تارونه     = kparams->n_act_تارونه;
    ثابت بنية fastdiv_ارزښتونه * act_تارونه_div = &kparams->div_n_act_تارونه;
    ثابت بنية fastdiv_ارزښتونه * k_div           = &kparams->div_ne00_padded;

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_اندازه : لیکه_stride;
    ثابت uint32_t dma_پلنوالی_بایټونه = is_tiled ? tile_اندازه : لیکه_stride;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, وزن_type, k, m_chunk_n_لیکې, n_chunk_n_ستون_شمېر, 1, pipeline, act_تارونه, aligned_tile_اندازه, 0);

    إذا (L.ټول_بایټونه > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.ټول_بایټونه, vtcm_budget, m, k, m_chunk_n_لیکې, n_chunk_n_ستون_شمېر);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنسټ = (uint8_t *) ctx->vtcm_بنسټ;
    __fp16  *vtcm_وزن_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_وزن[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنسټ, L.off_وزن[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_act);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, بنسټ, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, بنسټ, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, بنسټ, L.off_scratch[1], pipeline);
    فراغ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, بنسټ, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_scales);

    hmx_init_ستنه_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    صحيح m_پیل = 0;
    صحيح m_لیکې  = m;
    إذا (octx->ctx->mdev.شمېر > 1) {
        bool can_وېشل = htp_tensor_can_لیکه_برخه_کول(act, act_elem_اندازه);
        إذا (can_وېشل) {
            لكل (uint32_t p = 0; p < n_weights; ++p) {
                ثابت بنية htp_tensor * محدود dst = octx->dsts[p];
                إذا (!htp_tensor_can_لیکه_برخه_کول(dst, حجم(عائم))) {
                    can_وېشل = false;
                    اكسر;
                }
            }
        }
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول((uint32_t) m, can_وېشل ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        m_پیل = (صحيح) range.پیل;
        m_لیکې  = (صحيح) range.شمېر;
    }

    إذا (m_لیکې == 0) {
        أرجع HTP_حالت_OK;
    }

    FARF(لوړ_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_پیل, m_پیل + m_لیکې, k, وزن_type, m_chunk_n_لیکې, n_chunk_n_ستون_شمېر, L.ټول_بایټونه, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت size_t mr_پای_ټکی = (size_t)(m_پیل + m_لیکې);

    إذا (pipeline) {
        hmx_matmul_دنده_t دنده_slots[2];
        فراغ *vtcm_raw_bufs[4] = {
            vtcm_وزن_raw[0],
            vtcm_وزن_raw[1],
            dequant_کارګر_fn ? vtcm_وزن_raw[0] : vtcm_scratch0,
            dequant_کارګر_fn ? vtcm_وزن_raw[1] : vtcm_scratch1,
        };
        فراغ *vtcm_وزن_bufs[4] = {
            dequant_کارګر_fn ? vtcm_scratch0 : vtcm_وزن_raw[0],
            dequant_کارګر_fn ? vtcm_scratch1 : vtcm_وزن_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        فراغ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        لكل (size_t mr = (size_t) m_پیل; mr < mr_پای_ټکی; mr += m_chunk_n_لیکې) {
            ثابت size_t n_لیکې = hex_smin(mr_پای_ټکی - mr, m_chunk_n_لیکې);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_اندازه,
                .n_لیکې = (صحيح) n_لیکې,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_تارونه = act_تارونه,
                .act_تارونه_div = act_تارونه_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بایټونه = L.act_f32_بایټونه,
                .act_elem_اندازه = act_elem_اندازه,
            };
            transfer_activation_chunk_threaded(&act_params);

            لكل (uint32_t p = 0; p < n_weights; p++) {
                ثابت بنية htp_tensor * محدود src_w = octx->src[p];
                ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
                إذا (!src_w || !dst) استمر;

                ثابت dma_addr_t وزن       = src_w->معلومات;
                dma_کتار * وزن_dma      = octx->ctx->dma[0];
                عائم * dst_ptr            = (عائم *) dst->معلومات;
                ثابت size_t n             = src_w->ne[1];
                إذا (n == 0) استمر;
                ثابت size_t وزن_stride = src_w->nb[1];
                ثابت size_t dst_stride    = dst->nb[1] / حجم(عائم);
                ثابت صحيح dst_ستون_شمېر         = (صحيح) dst->ne[0];
                ثابت صحيح n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_ستون_شمېر);

                ثابت uint32_t dma_src_stride = is_tiled ? tile_اندازه : وزن_stride;

                ثابت size_t   n_ستون_شمېر_A0 = hex_smin(n - 0 * n_chunk_n_ستون_شمېر, n_chunk_n_ستون_شمېر);
                ثابت uint32_t لوړوالی_A0 = is_tiled ? hmx_ceil_div(n_ستون_شمېر_A0, 32) * n_وزن_tiles : n_ستون_شمېر_A0;
                dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[0], وزن),
                               dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی_A0);

                إذا (1 < n_chunk_cnt) {
                    ثابت size_t   n_ستون_شمېر_A1 = hex_smin(n - 1 * n_chunk_n_ستون_شمېر, n_chunk_n_ستون_شمېر);
                    ثابت uint32_t لوړوالی_A1 = is_tiled ? hmx_ceil_div(n_ستون_شمېر_A1, 32) * n_وزن_tiles : n_ستون_شمېر_A1;
                    dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[1], وزن + n_chunk_n_ستون_شمېر * وزن_stride),
                                   dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی_A1);
                }

                لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                    ثابت size_t nc    = i * n_chunk_n_ستون_شمېر;
                    ثابت size_t nc_p2 = nc + 2 * n_chunk_n_ستون_شمېر;

                    ثابت size_t n_ستون_شمېر    = hex_smin(n - nc, n_chunk_n_ستون_شمېر);
                    ثابت size_t n_ستون_شمېر_p2 = hex_smin(n - nc_p2, n_chunk_n_ستون_شمېر);

                    فراغ * curr_raw = (فراغ *) dma_کتار_راوباسه(وزن_dma).dst;

                    dequantize_tiled_وزن_chunk_منزل_fp16_tiles(
                        ctx, vtcm_وزن_bufs[i % 4], curr_raw,
                        n_ستون_شمېر, k, لیکه_stride, وزن_type,
                        n_k_tiles, n_k_tiles_div, dequant_کارګر_fn, n_تارونه);

                    إذا (i + 2 < n_chunk_cnt) {
                        ثابت uint32_t لوړوالی_p2 = is_tiled ? hmx_ceil_div(n_ستون_شمېر_p2, 32) * n_وزن_tiles : n_ستون_شمېر_p2;
                        dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[(i + 2) % 4], وزن + nc_p2 * وزن_stride),
                                       dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی_p2);
                    }

                    hmx_matmul_دنده_init(&دنده_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_لیکې, HTP_MM_HMX_TILE_N_لیکې),
                                        hmx_ceil_div(n_ستون_شمېر, HTP_MM_HMX_TILE_N_ستون_شمېر), k / HTP_MM_HMX_TILE_N_لیکې);
                    hmx_کتار_ټېل(ctx->hmx_کتار, hmx_کتار_make_desc(hmx_matmul_کارګر_fn, &دنده_slots[i % 2]));

                    إذا (i > 0) {
                        hmx_کتار_راوباسه(ctx->hmx_کتار);
                        ثابت size_t nc_prev = (i - 1) * n_chunk_n_ستون_شمېر;
                        ثابت size_t n_ستون_شمېر_prev = hex_smin(n - nc_prev, n_chunk_n_ستون_شمېر);
                        عائم *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        صحيح chunk_dst_ستون_شمېر = dst_ستون_شمېر - (صحيح)nc_prev;
                        إذا (chunk_dst_ستون_شمېر > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output_bufs[(i - 1) % 2], n_لیکې, n_ستون_شمېر_prev, dst_stride, 0, chunk_dst_ستون_شمېر, n_تارونه);
                        }
                    }
                }

                hmx_کتار_راوباسه(ctx->hmx_کتار);
                ثابت size_t nc_وروستی = (n_chunk_cnt - 1) * n_chunk_n_ستون_شمېر;
                ثابت size_t n_ستون_شمېر_وروستی = hex_smin(n - nc_وروستی, n_chunk_n_ستون_شمېر);
                عائم *output_chunk = dst_ptr + (mr * dst_stride + nc_وروستی);
                صحيح chunk_dst_ستون_شمېر = dst_ستون_شمېر - (صحيح)nc_وروستی;
                إذا (chunk_dst_ستون_شمېر > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_لیکې, n_ستون_شمېر_وروستی, dst_stride, 0, chunk_dst_ستون_شمېر, n_تارونه);
                }
            }
        }
    } وإلا {
        hmx_matmul_دنده_t دنده;
        فراغ *vtcm_raw_bufs[2] = { vtcm_وزن_raw[0], dequant_کارګر_fn ? vtcm_وزن_raw[0] : vtcm_scratch0 };
        فراغ *vtcm_وزن_bufs[2] = { dequant_کارګر_fn ? vtcm_scratch0 : vtcm_وزن_raw[0], vtcm_scratch0 };

        لكل (size_t mr = (size_t) m_پیل; mr < mr_پای_ټکی; mr += m_chunk_n_لیکې) {
            ثابت size_t n_لیکې = hex_smin(mr_پای_ټکی - mr, m_chunk_n_لیکې);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_اندازه,
                .n_لیکې = (صحيح) n_لیکې,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_تارونه = act_تارونه,
                .act_تارونه_div = act_تارونه_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بایټونه = L.act_f32_بایټونه,
                .act_elem_اندازه = act_elem_اندازه,
            };
            transfer_activation_chunk_threaded(&act_params);

            لكل (uint32_t p = 0; p < n_weights; p++) {
                ثابت بنية htp_tensor * محدود src_w = octx->src[p];
                ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
                إذا (!src_w || !dst) استمر;

                ثابت dma_addr_t وزن       = src_w->معلومات;
                dma_کتار * وزن_dma      = octx->ctx->dma[0];
                عائم * dst_ptr            = (عائم *) dst->معلومات;
                ثابت size_t n             = src_w->ne[1];
                إذا (n == 0) استمر;
                ثابت size_t وزن_stride = src_w->nb[1];
                ثابت size_t dst_stride    = dst->nb[1] / حجم(عائم);
                ثابت صحيح dst_ستون_شمېر         = (صحيح) dst->ne[0];
                ثابت صحيح n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_ستون_شمېر);

                ثابت uint32_t dma_src_stride = is_tiled ? tile_اندازه : وزن_stride;

                إذا (n > 0) {
                    ثابت size_t n_ستون_شمېر = hex_smin(n, n_chunk_n_ستون_شمېر);
                    ثابت uint32_t لوړوالی = is_tiled ? hmx_ceil_div(n_ستون_شمېر, 32) * n_وزن_tiles : n_ستون_شمېر;
                    dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[0], وزن), dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی);
                }

                لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                    ثابت size_t nc = (size_t) i * n_chunk_n_ستون_شمېر;
                    ثابت size_t n_ستون_شمېر = hex_smin(n - nc, n_chunk_n_ستون_شمېر);
                    ثابت size_t n_لیکه_tiles = hmx_ceil_div(n_لیکې, HTP_MM_HMX_TILE_N_لیکې);
                    ثابت size_t n_ستون_tiles = hmx_ceil_div(n_ستون_شمېر, HTP_MM_HMX_TILE_N_ستون_شمېر);

                    فراغ * curr_raw = (فراغ *) dma_کتار_راوباسه(وزن_dma).dst;

                    dequantize_tiled_وزن_chunk_منزل_fp16_tiles(
                        ctx, vtcm_وزن_bufs[i % 2], curr_raw,
                        n_ستون_شمېر, k, لیکه_stride, وزن_type,
                        n_k_tiles, n_k_tiles_div, dequant_کارګر_fn, n_تارونه);

                    ثابت size_t nc_بل = nc + n_chunk_n_ستون_شمېر;
                    إذا (nc_بل < n) {
                        ثابت size_t n_ستون_شمېر_بل = hex_smin(n - nc_بل, n_chunk_n_ستون_شمېر);
                        ثابت uint32_t لوړوالی_بل = is_tiled ? hmx_ceil_div(n_ستون_شمېر_بل, 32) * n_وزن_tiles : n_ستون_شمېر_بل;
                        dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[(i + 1) % 2], وزن + nc_بل * وزن_stride), dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی_بل);
                    }

                    hmx_matmul_دنده_init(&دنده, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 2], vtcm_scales, n_لیکه_tiles, n_ستون_tiles, k / HTP_MM_HMX_TILE_N_لیکې);
                    hmx_کتار_ټېل(ctx->hmx_کتار, hmx_کتار_make_desc(hmx_matmul_کارګر_fn, &دنده));
                    hmx_کتار_راوباسه(ctx->hmx_کتار);

                    عائم *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    صحيح chunk_dst_ستون_شمېر = dst_ستون_شمېر - (صحيح)nc;
                    إذا (chunk_dst_ستون_شمېر > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output, n_لیکې, n_ستون_شمېر, dst_stride, 0, chunk_dst_ستون_شمېر, n_تارونه);
                    }
                }
            }
        }
    }

    أرجع HTP_حالت_OK;
}

ساكن مضمن dma_addr_t hmx_mm_وزن_batch_معلومات(ثابت hmx_mm_batched_params_t *params,
                                                         صحيح dst_b2, صحيح dst_b3) {
    ثابت size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    ثابت size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    أرجع params->وزن + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
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
                                 صحيح m_chunk, صحيح n_chunk, صحيح pipeline, صحيح n_تارونه, صحيح act_تارونه, صحيح vtcm_اندازه,
                                 ثابت بنية fastdiv_ارزښتونه * act_تارونه_div, ثابت بنية fastdiv_ارزښتونه * k_div) {
    صحيح ret = 0;
    لكل (صحيح b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        لكل (صحيح b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->وزن_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_بایټونه,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_وزن_batch_معلومات(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_اندازه, params->وزن_stride,
                                params->وزن_type, params->k, params->dst_stride, params->bias_stride, params->dst_ستون_شمېر,
                                m_chunk, n_chunk, pipeline, n_تارونه, act_تارونه,
                                act_تارونه_div, k_div, params->tile_اندازه, params->aligned_tile_اندازه, vtcm_اندازه);
        }
    }
    أرجع ret;
}

ساكن صحيح hmx_mm_batched(بنية htp_context *ctx, ثابت hmx_mm_batched_params_t *params,
                          صحيح m_chunk, صحيح n_chunk, صحيح pipeline, صحيح n_تارونه, صحيح act_تارونه,
                          ثابت بنية fastdiv_ارزښتونه * act_تارونه_div,
                          ثابت بنية fastdiv_ارزښتونه * k_div,
                          صحيح vtcm_اندازه) {
    ثابت size_t لیکه_stride = htp_tensor_get_لیکه_اندازه(params->وزن_type, params->k);
    إذا (لیکه_stride == 0) { أرجع -1; }
    إذا (params->act_stride < params->k || (params->tile_اندازه == 0 && (size_t) params->وزن_stride < لیکه_stride) || params->dst_stride < params->dst_ستون_شمېر) { أرجع -1; }
    إذا (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { أرجع -1; }
    إذا (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { أرجع -1; }
    // N (the weight row count) does not have to be 32-aligned:
    إذا (params->k % 32 != 0) { أرجع -1; }
    إذا (!hex_is_aligned(params->dst, حجم(عائم))) { أرجع -1; }

    ثابت صحيح ډله_اندازه = params->r2;
    ثابت size_t vtcm_budget  = ctx->vtcm_اندازه;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool پرلپسې_grouped = (params->وزن_type == HTP_TYPE_F16 && params->tile_اندازه == 0 && ډله_اندازه > 1 && (size_t)vtcm_اندازه <= vtcm_budget);
    إذا (!پرلپسې_grouped) {
        أرجع hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_تارونه, act_تارونه, vtcm_اندازه, act_تارونه_div, k_div);
    }

    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_لیکې = m_chunk;
    size_t n_chunk_n_ستون_شمېر = n_chunk;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_لیکې, n_chunk_n_ستون_شمېر, ډله_اندازه, false, act_تارونه, 0, params->bias_بایټونه);

    إذا (L.ټول_بایټونه > vtcm_budget) {
        FARF(لوړ_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        أرجع hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_تارونه, act_تارونه, vtcm_اندازه, act_تارونه_div, k_div);
    }

    uint8_t * ثابت بنسټ = (uint8_t *) ctx->vtcm_بنسټ;
    __fp16  *vtcm_وزن     = VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_وزن[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, بنسټ, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR(فراغ, بنسټ, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, بنسټ, L.off_scales);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, بنسټ, L.off_act_f32);

    ثابت bool has_bias      = (params->bias_بایټونه > 0 && params->bias_addr != 0);
    عائم   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(عائم, بنسټ, L.off_bias, has_bias);
    إذا (has_bias) {
        dma_کتار_ټېل(params->وزن_dma, dma_make_معلومات(vtcm_bias, params->bias_addr), hex_align_پورته(params->bias_بایټونه, 128), 0, params->bias_بایټونه, 1);
        dma_کتار_راوباسه(params->وزن_dma);
    }

    hmx_init_ستنه_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(لوړ_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, ډله_اندازه, params->ne13,
            m_chunk_n_لیکې, n_chunk_n_ستون_شمېر,
            L.ټول_بایټونه, vtcm_budget);

    ثابت size_t fp16_لیکه_بایټونه   = (size_t) params->k * حجم(__fp16);
    ثابت size_t وزن_لیکه_بایټونه = (size_t) params->وزن_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_دنده_t دنده;

    لكل (صحيح b3 = 0; b3 < params->ne13; ++b3) {
        لكل (صحيح b2_بنسټ = 0; b2_بنسټ < params->ne12; b2_بنسټ += ډله_اندازه) {
            ثابت dma_addr_t وزن_ډله = hmx_mm_وزن_batch_معلومات(params, b2_بنسټ, b3);
            dma_کتار * وزن_dma = params->وزن_dma;

            لكل (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_لیکې) {
                ثابت size_t n_لیکې = hex_smin((size_t) params->m - mr, m_chunk_n_لیکې);
                ثابت size_t n_لیکه_tiles = hmx_ceil_div((صحيح) n_لیکې, HTP_MM_HMX_TILE_N_لیکې);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                لكل (صحيح g = 0; g < ډله_اندازه; ++g) {
                    ثابت dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_بنسټ + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_اندازه;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_سر_stride;
                    بنية activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_لیکې = (صحيح) n_لیکې,
                        .k_بلاک = params->k,
                        .k_stride = params->act_stride,
                        .n_تارونه = act_تارونه,
                        .act_تارونه_div = act_تارونه_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_بایټونه = L.act_f32_بایټونه,
                        .act_elem_اندازه = params->act_elem_اندازه,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    ثابت size_t n_ستون_شمېر_لومړی = hex_smin((size_t) params->n, n_chunk_n_ستون_شمېر);
                    dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_scratch0, وزن_ډله),
                                      fp16_لیکه_بایټونه, وزن_لیکه_بایټونه, fp16_لیکه_بایټونه, n_ستون_شمېر_لومړی);
                }
                إذا (n_chunk_n_ستون_شمېر < (size_t) params->n) {
                    ثابت size_t n_ستون_شمېر_دوهم = hex_smin((size_t) params->n - n_chunk_n_ستون_شمېر, n_chunk_n_ستون_شمېر);
                    ثابت dma_addr_t دوهم_وزن_chunk = وزن_ډله + n_chunk_n_ستون_شمېر * وزن_لیکه_بایټونه;
                    dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_scratch1, دوهم_وزن_chunk),
                                      fp16_لیکه_بایټونه, وزن_لیکه_بایټونه, fp16_لیکه_بایټونه, n_ستون_شمېر_دوهم);
                }

                لكل (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_ستون_شمېر) {
                    ثابت size_t n_ستون_شمېر      = hex_smin((size_t) params->n - nc, n_chunk_n_ستون_شمېر);
                    ثابت size_t n_ستون_tiles = hmx_ceil_div((صحيح) n_ستون_شمېر, HTP_MM_HMX_TILE_N_ستون_شمېر);

                    {
                        فراغ * curr_raw = (فراغ *) dma_کتار_راوباسه(وزن_dma).dst;

                        ثابت size_t n_ستون_شمېر_tiled = hex_align_پورته(n_ستون_شمېر, HTP_MM_HMX_TILE_N_ستون_شمېر);

                        hmx_interleave_لیکې_منزل_tiles(vtcm_وزن, (ثابت __fp16 *) curr_raw, (uint32_t) n_ستون_شمېر, params->k, params->k, 0, (uint32_t) n_ستون_شمېر_tiled);

                        ثابت size_t nc_بل = nc + n_chunk_n_ستون_شمېر * 2;
                        إذا (nc_بل < (size_t) params->n) {
                            ثابت size_t n_ستون_شمېر_بل = hex_smin((size_t) params->n - nc_بل, n_chunk_n_ستون_شمېر);
                            ثابت dma_addr_t بل_وزن_chunk = وزن_ډله + nc_بل * وزن_لیکه_بایټونه;

                            dma_کتار_ټېل(وزن_dma, dma_make_معلومات(curr_raw, بل_وزن_chunk),
                                              fp16_لیکه_بایټونه, وزن_لیکه_بایټونه, fp16_لیکه_بایټونه, n_ستون_شمېر_بل);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    لكل (صحيح g = 0; g < ډله_اندازه; ++g) {
                        {
                            ثابت __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_سر_stride;
                            hmx_matmul_دنده_init(&دنده, vtcm_output, vtcm_act_g, vtcm_وزن, vtcm_scales, n_لیکه_tiles, n_ستون_tiles, params->k / 32);
                            hmx_کتار_ټېل(ctx->hmx_کتار, hmx_کتار_make_desc(hmx_matmul_کارګر_fn, &دنده));
                            hmx_کتار_راوباسه(ctx->hmx_کتار);
                        }

                        {
                            عائم *output = hmx_mm_dst_batch_ptr(params, b2_بنسټ + g, b3) + mr * params->dst_stride + nc;
                            ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : عدم;
                            صحيح chunk_dst_ستون_شمېر = params->dst_ستون_شمېر - (صحيح)nc;
                            إذا (chunk_dst_ستون_شمېر > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (صحيح) n_لیکې, (صحيح) n_ستون_شمېر,
                                                               params->dst_stride, params->bias_stride, chunk_dst_ستون_شمېر, n_تارونه);
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
            صحيح پیل_لیکه,
            صحيح n_لیکې,
            صحيح k_بلاک,
            ثابت بنية mmid_لیکه_mapping *ماتریکس_لیکې,
            صحيح cur_a,
            صحيح mapping_stride,
            صحيح ne11,
            size_t nb11,
            size_t nb12,
            صحيح cne1,
            صحيح n_تارونه,
            صحيح k_valid) {
    إذا (n_لیکې <= 0) أرجع;
    صحيح chunks_فی_thread = hmx_ceil_div(n_لیکې, n_تارونه);
    chunks_فی_thread = hex_align_پورته(chunks_فی_thread, 2);

    صحيح actual_تارونه = hmx_ceil_div(n_لیکې, chunks_فی_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_تارونه,
        .n_tot_chunks      = n_لیکې,
        .n_chunks_فی_task = chunks_فی_thread,
        .k_بلاک           = k_بلاک,
        .ماتریکس_لیکې       = ماتریکس_لیکې,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_ارزښتونه(ne11) : (بنية fastdiv_ارزښتونه){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .پیل_لیکه         = پیل_لیکه,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    کارګر_callback_t کارګر_fn = ne11 == 1 ? transfer_activation_chunk_gathered_کارګر_flat_fn :
                                              transfer_activation_chunk_gathered_کارګر_fn;

    إذا (actual_تارونه <= 1) {
        کارګر_fn(1, 0, &state);
    } وإلا {
        کارګر_pool_پرلپسې_func(ctx->کارګر_pool, کارګر_fn, &state, actual_تارونه);
    }
}

ساكن فراغ transfer_output_chunk_scattered_threaded(
            بنية htp_context *ctx,
            عائم *dst,
            ثابت __fp16 *vtcm_src,
            صحيح پیل_لیکه,
            صحيح n_لیکې,
            صحيح n_ستون_شمېر,
            ثابت بنية mmid_لیکه_mapping *ماتریکس_لیکې,
            صحيح cur_a,
            صحيح mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            صحيح cne1,
            صحيح n_تارونه) {
    إذا (n_لیکې <= 0) أرجع;
    صحيح chunks_فی_thread = hmx_ceil_div(n_لیکې, n_تارونه);
    chunks_فی_thread = hex_align_پورته(chunks_فی_thread, 2);

    صحيح actual_تارونه = hmx_ceil_div(n_لیکې, chunks_فی_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_تارونه,
        .n_tot_chunks      = n_لیکې,
        .n_chunks_فی_task = chunks_فی_thread,
        .n_ستون_شمېر            = n_ستون_شمېر,
        .ماتریکس_لیکې       = ماتریکس_لیکې,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .پیل_لیکه         = پیل_لیکه,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    إذا (actual_تارونه <= 1) {
        transfer_output_chunk_scattered_کارګر_fn(1, 0, &state);
    } وإلا {
        کارګر_pool_پرلپسې_func(ctx->کارګر_pool, transfer_output_chunk_scattered_کارګر_fn, &state, actual_تارونه);
    }
}

ساكن صحيح hmx_mm_id_2d_f32(بنية htp_context *ctx,
                                         dma_کتار *وزن_dma,
                                         عائم *محدود dst,
                                         ثابت عائم *activation,
                                         dma_addr_t وزن,
                                         صحيح m, صحيح k, صحيح n,
                                         صحيح k_valid,
                                         صحيح ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         صحيح وزن_stride,
                                         صحيح وزن_type,
                                         صحيح tile_اندازه,
                                         صحيح aligned_tile_اندازه,
                                         ثابت بنية mmid_لیکه_mapping *ماتریکس_لیکې,
                                         صحيح cur_a,
                                         صحيح mapping_stride,
                                         صحيح m_پیل,
                                         صحيح m_پای_ټکی,
                                         صحيح n_تارونه) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت صحيح cne1 = m;
    ثابت صحيح m_core = m_پای_ټکی - m_پیل;
    ثابت صحيح m_core_padded = hex_align_پورته(m_core > 0 ? m_core : 1, 32);

    إذا (k % 32 != 0 || n % 32 != 0) { أرجع -1; }
    إذا (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { أرجع -1; }

    size_t لیکه_stride = htp_tensor_get_لیکه_اندازه(وزن_type, k);
    إذا (لیکه_stride == 0) {
        أرجع -1;
    }

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_ستون_شمېر;
    ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(وزن_type);
    ثابت بنية fastdiv_ارزښتونه n_k_tiles_div = init_fastdiv_ارزښتونه(n_k_tiles);

    ثابت size_t vec_dot_اندازه = k * حجم(__fp16);
    ثابت size_t vtcm_budget  = ctx->vtcm_اندازه;
    size_t vtcm_کارول_شوی = 0;

    ثابت bool is_tiled = (tile_اندازه > 0);
    tiled_dequant_task_fn_t dequant_کارګر_fn = hmx_get_dequant_کارګر_fn(وزن_type, is_tiled);

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_اندازه : لیکه_stride;
    ثابت uint32_t dma_src_stride  = is_tiled ? tile_اندازه : وزن_stride;
    ثابت uint32_t dma_پلنوالی_بایټونه = is_tiled ? tile_اندازه : لیکه_stride;

    ثابت size_t qweight_لیکه_stride = is_tiled ? (size_t)(n_وزن_tiles * aligned_tile_اندازه) / 32 : 0;
    ثابت size_t وزن_لیکه_stride = is_tiled ? qweight_لیکه_stride : لیکه_stride;

    size_t اندازه_فی_n = 0, اندازه_فی_m = 0, اندازه_فی_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(وزن_type, k, /*pipeline=*/false, aligned_tile_اندازه,
                                  &اندازه_فی_n, &اندازه_فی_m, &اندازه_فی_mn);

    ثابت size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_لیکې = 0, n_chunk_n_ستون_شمېر = 0;
    إذا (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, اندازه_فی_n, اندازه_فی_m, اندازه_فی_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_لګښت_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_لګښت_A_CONVERT, &m_chunk_n_لیکې, &n_chunk_n_ستون_شمېر, &vtcm_کارول_شوی)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        أرجع -1;
    }

    ثابت size_t وزن_area_اندازه = hex_align_پورته(n_chunk_n_ستون_شمېر * وزن_لیکه_stride, HTP_MM_HMX_TILE_اندازه);
    ثابت size_t act_area_اندازه    = hex_align_پورته(m_chunk_n_لیکې * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);
    ثابت size_t output_area_اندازه = hex_align_پورته(m_chunk_n_لیکې * n_chunk_n_ستون_شمېر * حجم(__fp16), HTP_MM_HMX_TILE_اندازه);

    size_t scratch0_اندازه = hex_align_پورته(n_chunk_n_ستون_شمېر * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_بنسټ;
    __fp16  *vtcm_وزن   = وزن_area_اندازه ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, وزن_area_اندازه) : عدم;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_اندازه);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_اندازه);
    فراغ    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_اندازه);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_کارول_شوی = vtcm_ptr - (uint8_t *) ctx->vtcm_بنسټ;
    إذا (vtcm_کارول_شوی > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_کارول_شوی, vtcm_budget);
        أرجع -1;
    }

    hmx_init_ستنه_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_دنده_t دنده;
    فراغ *vtcm_raw_bufs[2] = { vtcm_وزن, dequant_کارګر_fn ? vtcm_وزن : vtcm_scratch0 };
    فراغ *vtcm_وزن_bufs[2] = { dequant_کارګر_fn ? vtcm_scratch0 : vtcm_وزن, vtcm_scratch0 };
    ثابت صحيح n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_ستون_شمېر);

    لكل (size_t mr = (size_t) m_پیل; mr < (size_t) m_پای_ټکی; mr += m_chunk_n_لیکې) {
        ثابت size_t n_لیکې = hex_smin((size_t) m_پای_ټکی - mr, m_chunk_n_لیکې);
        ثابت size_t n_لیکه_tiles = hmx_ceil_div(n_لیکې, HTP_MM_HMX_TILE_N_لیکې);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (صحيح) mr, (صحيح) n_لیکې, k,
            ماتریکس_لیکې, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_تارونه, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        إذا (n > 0) {
            ثابت size_t n_ستون_شمېر = hex_smin((size_t) n, n_chunk_n_ستون_شمېر);
            ثابت uint32_t لوړوالی = is_tiled ? hmx_ceil_div(n_ستون_شمېر, 32) * n_وزن_tiles : n_ستون_شمېر;
            dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[0], وزن),
                           dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی);
        }

        لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
            ثابت size_t nc = (size_t) i * n_chunk_n_ستون_شمېر;
            ثابت size_t n_ستون_شمېر = hex_smin((size_t) n - nc, n_chunk_n_ستون_شمېر);
            ثابت size_t n_ستون_tiles = hmx_ceil_div(n_ستون_شمېر, HTP_MM_HMX_TILE_N_ستون_شمېر);

            // A: Wait for weight DMA
            فراغ * curr_raw = (فراغ *) dma_کتار_راوباسه(وزن_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_وزن_chunk_منزل_fp16_tiles(
                ctx, vtcm_وزن_bufs[i % 2], curr_raw,
                n_ستون_شمېر, k, لیکه_stride, وزن_type,
                n_k_tiles, n_k_tiles_div, dequant_کارګر_fn, n_تارونه
            );

            // Start weight DMA for the next chunk early
            ثابت size_t nc_بل = nc + n_chunk_n_ستون_شمېر;
            إذا (nc_بل < (size_t) n) {
                ثابت size_t n_ستون_شمېر_بل = hex_smin((size_t) n - nc_بل, n_chunk_n_ستون_شمېر);
                ثابت uint32_t لوړوالی_بل = is_tiled ? hmx_ceil_div(n_ستون_شمېر_بل, 32) * n_وزن_tiles : n_ستون_شمېر_بل;
                dma_کتار_ټېل(وزن_dma, dma_make_معلومات(vtcm_raw_bufs[(i + 1) % 2], وزن + nc_بل * وزن_stride),
                               dma_dst_stride, dma_src_stride, dma_پلنوالی_بایټونه, لوړوالی_بل);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_دنده_init(&دنده, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 2], vtcm_scales, n_لیکه_tiles, n_ستون_tiles, k / HTP_MM_HMX_TILE_N_لیکې);
            hmx_کتار_ټېل(ctx->hmx_کتار, hmx_کتار_make_desc(hmx_matmul_کارګر_fn, &دنده));
            hmx_کتار_راوباسه(ctx->hmx_کتار);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (صحيح) mr, (صحيح) n_لیکې, (صحيح) n_ستون_شمېر,
                ماتریکس_لیکې, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_تارونه);
        }
    }

    أرجع 0;
}

// --- Dispatchers and Public Entry Points ---

ساكن صحيح hmx_mm_op_matmul(بنية htp_ops_context * octx, ثابت بنية htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    صحيح k = (صحيح) src0->ne[0];
    صحيح n = (صحيح) src0->ne[1];
    ثابت صحيح m_ټول    = (صحيح) act->ne[1];
    ثابت uint32_t act_elem_اندازه = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت صحيح act_stride = (صحيح) (act->nb[1] / act_elem_اندازه);

    صحيح m_پیل = 0;
    صحيح m_لیکې  = m_ټول;
    إذا (octx->ctx->mdev.شمېر > 1) {
        bool can_وېشل = htp_tensor_can_لیکه_برخه_کول(dst, حجم(عائم)) &&
                         htp_tensor_can_لیکه_برخه_کول(act, act_elem_اندازه);
        إذا (src2 && src2->ne[1] > 1 && !htp_tensor_can_لیکه_برخه_کول(src2, حجم(عائم))) {
            can_وېشل = false;
        }
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول((uint32_t) m_ټول, can_وېشل ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        m_پیل = (صحيح) range.پیل;
        m_لیکې  = (صحيح) range.شمېر;
    }

    إذا (m_لیکې == 0) {
        أرجع HTP_حالت_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_بایټونه = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    إذا (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / حجم(عائم));
        src2_addr = src2->معلومات + m_پیل * src2_stride * حجم(عائم);
        src2_بایټونه = (size_t) kparams->vtcm_bias_اندازه;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ثابت صحيح dst_stride = (صحيح)(dst->nb[1] / حجم(عائم));
    عائم       * dst_ptr = (عائم *) dst->معلومات + m_پیل * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ثابت dma_addr_t act_addr = act->معلومات + (size_t) m_پیل * act_stride * act_elem_اندازه;

    صحيح ret = -1;
    ثابت صحيح n_تارونه = kparams->n_تارونه;
    إذا (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        hmx_mm_batched_params_t batch_params = {
            .dst               = dst_ptr,
            .bias_addr         = src2_addr,
            .bias_بایټونه        = src2_بایټونه,
            .act_dma_addr      = act_addr,
            .وزن            = src0->معلومات,
            .وزن_dma        = octx->ctx->dma[0],
            .m                 = m_لیکې,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_اندازه     = act_elem_اندازه,
            .وزن_stride     = (صحيح) src0->nb[1],
            .وزن_type       = (صحيح) src0->type,
            .dst_stride        = dst_stride,
            .dst_ستون_شمېر          = (صحيح) dst->ne[0],
            .bias_stride       = src2_stride,
            .tile_اندازه         = kparams->tile_اندازه,
            .aligned_tile_اندازه = kparams->aligned_tile_اندازه,
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
                             kparams->pipeline, n_تارونه,
                             kparams->n_act_تارونه,
                             &kparams->div_n_act_تارونه,
                             &kparams->div_ne00_padded,
                             kparams->vtcm_اندازه);
    } وإلا {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_بایټونه,
            act_addr, src0->معلومات,
            m_لیکې, k, n, act_stride, act_elem_اندازه, (صحيح) src0->nb[1], (صحيح) src0->type, (صحيح) act->ne[0],
            dst_stride, src2_stride, (صحيح)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_تارونه,
            kparams->n_act_تارونه,
            &kparams->div_n_act_تارونه,
            &kparams->div_ne00_padded,
            kparams->tile_اندازه, kparams->aligned_tile_اندازه, kparams->vtcm_اندازه
        );
    }

    إذا (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        أرجع HTP_حالت_دننی_ERR;
    }
    أرجع HTP_حالت_OK;
}

ساكن مضمن فراغ htp_mm_tensor_collapse_لیکې(بنية htp_tensor * c, ثابت بنية htp_tensor * t, uint32_t stride) {
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
        htp_mm_tensor_collapse_لیکې(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_لیکې(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        بنية htp_tensor src2_collapsed;
        ثابت بنية htp_tensor * src2 = octx->src[2];
        إذا (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ثابت uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_لیکې(&src2_collapsed, src2, s2);
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
    ثابت uint32_t *                ماتریکس_لیکه_counts = mmctx->ماتریکس_لیکه_counts;
    ثابت بنية mmid_لیکه_mapping * ماتریکس_لیکې       = mmctx->ماتریکس_لیکې;
    htp_matmul_tensors_preamble;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت صحيح n_ids = octx->src[2]->ne[0];
    ثابت صحيح n_as  = ne02;

    ثابت bool mdev_وېشل = (octx->ctx->mdev.شمېر > 1) && htp_tensor_can_لیکه_برخه_کول(dst, حجم(عائم));
    إذا (octx->ctx->mdev.شمېر > 1 && !mdev_وېشل && octx->ctx->mdev.idx > 0) {
        أرجع HTP_حالت_OK;
    }
    uint32_t n_فعال = 0;
    إذا (mdev_وېشل) {
        لكل (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            إذا (ماتریکس_لیکه_counts[a] > 0) n_فعال++;
        }
    }
    ثابت bool expert_وېشل = mdev_وېشل && (n_فعال >= octx->ctx->mdev.شمېر);

    uint32_t موخه_dev = 0;
    لكل (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ثابت int32_t cne1 = ماتریکس_لیکه_counts[cur_a];
        إذا (cne1 == 0) استمر;

        ثابت صحيح m_padded = hex_align_پورته(cne1, 32);
        صحيح m_پیل = 0, m_پای_ټکی = m_padded;
        إذا (expert_وېشل) {
            ثابت bool my_expert = (موخه_dev == octx->ctx->mdev.idx);
            إذا (++موخه_dev == octx->ctx->mdev.شمېر) {
                موخه_dev = 0;
            }
            إذا (!my_expert) استمر;
        } وإلا إذا (mdev_وېشل) {
            ثابت bool can_وېشل = (uint32_t) cne1 >= octx->ctx->mdev.شمېر;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول((uint32_t) m_padded, can_وېشل ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
            m_پیل = (صحيح) range.پیل;
            m_پای_ټکی   = (صحيح) (range.پیل + range.شمېر);
        }
        إذا (m_پیل >= m_پای_ټکی) استمر;

        صحيح ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (عائم*) dst->معلومات, (عائم*) act->معلومات,
                                   src0->معلومات + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (صحيح) src0->nb[1], (صحيح) src0->type,
                                   kparams->tile_اندازه, kparams->aligned_tile_اندازه,
                                   ماتریکس_لیکې, cur_a, mmctx->mapping_stride,
                                   m_پیل, m_پای_ټکی, (صحيح) octx->n_تارونه);
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
    work_کتار_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ثابت uint32_t src0_لیکه_اندازه_padded = mmctx->src0_لیکه_اندازه_padded;
    ثابت uint32_t act_nrows            = mmctx->act_nrows;

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت بنية htp_tensor * محدود ids = octx->src[2];
    ثابت size_t src0_لیکه_اندازه = nb01;

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (ne10 + qk - 1) / qk;
    ثابت uint32_t ټول_nb = act_nrows * nb;

    work_کتار_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    إذا (act_nrows < octx->n_تارونه) {
        n_quant_tasks = MIN(ټول_nb, octx->n_تارونه);
        quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_لومړی = (ټول_nb * ith) / n_quant_tasks;
            uint32_t ib_وروستی  = (ټول_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_لومړی[ith] = ib_لومړی;
            mmctx->quant_ib_وروستی[ith]  = ib_وروستی;
            mmctx->quant_r[ith]        = ib_لومړی / nb;
            mmctx->quant_c[ith]        = ib_لومړی % nb;
        }
    } وإلا {
        n_quant_tasks = MIN(act_nrows, octx->n_تارونه);
        quant_task_func = htp_mm_act_quant_لیکه_func(src0->type, act->type);
    }
    size_t act_لیکه_اندازه = htp_mm_tiled_act_لیکه_اندازه(src0->type, ne10);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_تارونه,
                                 0, src0_لیکه_اندازه, act_لیکه_اندازه, 0, kparams->n_prefetch, true, false);

    ثابت size_t vtcm_اندازه = L.ټول_بایټونه;

    FARF(لوړ_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_بایټونه, L.act_بایټونه, L.dst_بایټونه, vtcm_اندازه);

    FARF(لوړ_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->معلومات,
         act->معلومات, dst->معلومات);

    // Make sure the reserved vtcm size is sufficient
    إذا (octx->ctx->vtcm_اندازه < vtcm_اندازه) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_اندازه, vtcm_اندازه);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_src0);
    mmctx->vtcm_bias     = عدم;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_لیکه_اندازه_padded;
    mmctx->vtcm_act_stride     = act_لیکه_اندازه;
    mmctx->vtcm_act_raw_stride = hex_پړاو_پورته(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_اندازه_فی_thread = fastdiv(L.src0_بایټونه, &octx->n_تارونه_div);
    mmctx->vtcm_act_اندازه_فی_thread  = L.act_بایټونه;
    mmctx->vtcm_dst_اندازه_فی_thread  = fastdiv(L.dst_بایټونه, &octx->n_تارونه_div);

    mmctx->cur_m_پیل = 0;
    mmctx->cur_m_لیکې  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_لیکې_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_کتار_پرلپسې(octx->ctx->work_کتار, quant_task_func, mmctx, n_quant_tasks);

    work_کتار_پرلپسې(octx->ctx->work_کتار, hvx_mmid_task_func, mmctx, octx->n_تارونه);

    أرجع HTP_حالت_OK;
}

ساكن صحيح hmx_mm_op_matmul_id_nx(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx
) {
    ثابت uint32_t *                ماتریکس_لیکه_counts = mmctx->ماتریکس_لیکه_counts;
    ثابت بنية mmid_لیکه_mapping * ماتریکس_لیکې       = mmctx->ماتریکس_لیکې;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];
    ثابت صحيح n_as = src0->ne[2];

    bool mdev_وېشل = (octx->ctx->mdev.شمېر > 1);
    لكل (uint32_t p = 0; p < n_weights && mdev_وېشل; ++p) {
        ثابت بنية htp_tensor * محدود dst = octx->dsts[p];
        mdev_وېشل = htp_tensor_can_لیکه_برخه_کول(dst, حجم(عائم));
    }
    إذا (octx->ctx->mdev.شمېر > 1 && !mdev_وېشل && octx->ctx->mdev.idx > 0) {
        أرجع HTP_حالت_OK;
    }
    uint32_t n_فعال = 0;
    إذا (mdev_وېشل) {
        لكل (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            إذا (ماتریکس_لیکه_counts[a] > 0) n_فعال++;
        }
    }
    ثابت bool expert_وېشل = mdev_وېشل && (n_فعال >= octx->ctx->mdev.شمېر);

    uint32_t موخه_dev = 0;
    لكل (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ثابت int32_t cne1 = ماتریکس_لیکه_counts[cur_a];
        إذا (cne1 == 0) استمر;

        ثابت صحيح m_padded = hex_align_پورته(cne1, 32);
        صحيح m_پیل = 0, m_پای_ټکی = m_padded;
        إذا (expert_وېشل) {
            ثابت bool my_expert = (موخه_dev == octx->ctx->mdev.idx);
            إذا (++موخه_dev == octx->ctx->mdev.شمېر) {
                موخه_dev = 0;
            }
            إذا (!my_expert) استمر;
        } وإلا إذا (mdev_وېشل) {
            ثابت bool can_وېشل = (uint32_t) cne1 >= octx->ctx->mdev.شمېر;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول((uint32_t) m_padded, can_وېشل ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
            m_پیل = (صحيح) range.پیل;
            m_پای_ټکی   = (صحيح) (range.پیل + range.شمېر);
        }
        إذا (m_پیل >= m_پای_ټکی) استمر;

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * محدود src_w = octx->src[p];
            ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
            إذا (!src_w || !dst) استمر;

            صحيح ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (عائم*) dst->معلومات, (عائم*) act->معلومات,
                                       src_w->معلومات + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (صحيح) src_w->nb[1], (صحيح) src_w->type,
                                       kparams->tile_اندازه, kparams->aligned_tile_اندازه,
                                       ماتریکس_لیکې, cur_a, mmctx->mapping_stride,
                                       m_پیل, m_پای_ټکی, (صحيح) octx->n_تارونه);
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
    work_کتار_func_t hvx_mmid_task_func
) {
    ثابت uint32_t src0_لیکه_اندازه_padded = mmctx->src0_لیکه_اندازه_padded;
    ثابت uint32_t act_nrows            = mmctx->act_nrows;

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];
    ثابت بنية htp_tensor * محدود ids  = octx->src[n_weights + 1];
    ثابت size_t src0_لیکه_اندازه = src0->nb[1];

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ثابت uint32_t ټول_nb = act_nrows * nb;

    work_کتار_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    إذا (act_nrows < octx->n_تارونه) {
        n_quant_tasks = MIN(ټول_nb, octx->n_تارونه);
        quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_لومړی = (ټول_nb * ith) / n_quant_tasks;
            uint32_t ib_وروستی  = (ټول_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_لومړی[ith] = ib_لومړی;
            mmctx->quant_ib_وروستی[ith]  = ib_وروستی;
            mmctx->quant_r[ith]        = ib_لومړی / nb;
            mmctx->quant_c[ith]        = ib_لومړی % nb;
        }
    } وإلا {
        n_quant_tasks = MIN(act_nrows, octx->n_تارونه);
        quant_task_func = htp_mm_act_quant_لیکه_func(src0->type, act->type);
    }
    size_t act_لیکه_اندازه = htp_mm_tiled_act_لیکه_اندازه(src0->type, act->ne[0]);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_تارونه,
                                 0, src0_لیکه_اندازه, act_لیکه_اندازه, 0, kparams->n_prefetch, true, false);

    ثابت size_t vtcm_اندازه = L.ټول_بایټونه;

    إذا (octx->ctx->vtcm_اندازه < vtcm_اندازه) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_اندازه, vtcm_اندازه);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_لیکه_اندازه;
    mmctx->vtcm_act_raw_stride = hex_پړاو_پورته(act->ne[0] * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_اندازه_فی_thread = fastdiv(L.src0_بایټونه, &octx->n_تارونه_div);
    mmctx->vtcm_act_اندازه_فی_thread  = L.act_بایټونه;
    mmctx->vtcm_dst_اندازه_فی_thread  = fastdiv(L.dst_بایټونه, &octx->n_تارونه_div);

    mmctx->cur_m_پیل = 0;
    mmctx->cur_m_لیکې  = act_nrows;

    FARF(لوړ_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.ټول_بایټونه, octx->ctx->vtcm_اندازه, octx->n_تارونه);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_لیکې_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_کتار_پرلپسې(octx->ctx->work_کتار, quant_task_func, mmctx, n_quant_tasks);

    work_کتار_پرلپسې(octx->ctx->work_کتار, hvx_mmid_task_func, mmctx, octx->n_تارونه);

    أرجع HTP_حالت_OK;
}

ساكن مضمن فراغ لټون_expert_ids_n(
    ثابت بنية htp_tensor * ids,
    ثابت uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    بنية mmid_لیکه_mapping * ماتریکس_لیکې,
    uint32_t mapping_stride
) {
    ثابت size_t ids_nb1 = ids->nb[1];
    ثابت uint8_t * ids_معلومات = (ثابت uint8_t *) ids->معلومات;

    لكل (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ثابت int32_t * لیکه_ptr = (ثابت int32_t *) (ids_معلومات + iid1 * ids_nb1);
        لكل (uint32_t id = 0; id < n_ids; ++id) {
            ثابت int32_t i02 = لیکه_ptr[id];
            إذا (i02 < 0) {
                استمر;
            }
            أكد(i02 < n_as);

            إذا (ماتریکس_لیکې) {
                ماتریکس_لیکې[i02 * mapping_stride + counts[i02]] = (بنية mmid_لیکه_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

ساكن مضمن فراغ لټون_expert_ids(
    ثابت بنية htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    بنية mmid_لیکه_mapping * ماتریکس_لیکې,
    uint32_t mapping_stride
) {
    ثابت size_t ids_nb0 = ids->nb[0];

    إذا (ids_nb0 == 4) {
        بدل (n_ids) {
            حالة 8:  لټون_expert_ids_n(ids, 8,     n_as, counts, ماتریکس_لیکې, mapping_stride); اكسر;
            حالة 4:  لټون_expert_ids_n(ids, 4,     n_as, counts, ماتریکس_لیکې, mapping_stride); اكسر;
            حالة 2:  لټون_expert_ids_n(ids, 2,     n_as, counts, ماتریکس_لیکې, mapping_stride); اكسر;
            افتراضي: لټون_expert_ids_n(ids, n_ids, n_as, counts, ماتریکس_لیکې, mapping_stride); اكسر;
        }
    } وإلا {
        // Strided fallback
        ثابت size_t ids_nb1 = ids->nb[1];
        ثابت uint8_t * ids_معلومات = (ثابت uint8_t *) ids->معلومات;
        لكل (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ثابت int32_t * لیکه_ptr = (ثابت int32_t *) (ids_معلومات + iid1 * ids_nb1);
            لكل (uint32_t id = 0; id < n_ids; ++id) {
                ثابت int32_t i02 = *(ثابت int32_t *) ((ثابت uint8_t *) لیکه_ptr + id * ids_nb0);
                إذا (i02 < 0) {
                    استمر;
                }
                أكد(i02 < n_as);

                إذا (ماتریکس_لیکې) {
                    ماتریکس_لیکې[i02 * mapping_stride + counts[i02]] = (بنية mmid_لیکه_mapping) { id, iid1 };
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
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ثابت بنية htp_tensor * محدود ids = octx->src[2];
    إذا (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت size_t src0_لیکه_اندازه = nb01;
    ثابت size_t dst_لیکه_اندازه  = nb1;

    ثابت size_t src0_لیکه_اندازه_padded = hex_پړاو_پورته(src0_لیکه_اندازه, 128);

    ثابت uint32_t src0_nrows = ne01;  // per expert
    ثابت uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ثابت صحيح n_ids = ids->ne[0];  // n_expert_used
    ثابت صحيح n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_بنسټ;
    uint32_t   mapping_stride    = 1;
    uint32_t * ماتریکس_لیکه_counts = (uint32_t *) mapping_buf;
    بنية mmid_لیکه_mapping * ماتریکس_لیکې = عدم;

    إذا (act_nrows > 1) {
        ثابت size_t ماتریکس_لیکه_counts_اندازه = n_as * حجم(uint32_t);
        أكد(octx->ctx->ddr_spad_اندازه >= ماتریکس_لیکه_counts_اندازه);

        hex_l2fetch_بلاک((ثابت فراغ *) ids->معلومات, ids->ne[1] * ids->nb[1]);

        memset(ماتریکس_لیکه_counts, 0, ماتریکس_لیکه_counts_اندازه);
        لټون_expert_ids(ids, n_ids, n_as, ماتریکس_لیکه_counts, عدم, 0);

        uint32_t max_شمېر = hvx_reduce_max_i32((ثابت uint8_t *) ماتریکس_لیکه_counts, n_as);
        mapping_stride = max_شمېر > 0 ? max_شمېر : 1;

        size_t ماتریکس_لیکه_map_اندازه  = n_as * mapping_stride * حجم(بنية mmid_لیکه_mapping);
        ثابت size_t ټول_map_اندازه = ماتریکس_لیکه_counts_اندازه + ماتریکس_لیکه_map_اندازه;

        إذا (ټول_map_اندازه > octx->ctx->ddr_spad_اندازه) {
            mapping_buf = memalign(128, ټول_map_اندازه);
            إذا (!mapping_buf) {
                أرجع HTP_حالت_دننی_ERR;
            }
        }

        ماتریکس_لیکه_counts = (uint32_t *) mapping_buf;
        ماتریکس_لیکې       = (بنية mmid_لیکه_mapping *) (mapping_buf + ماتریکس_لیکه_counts_اندازه);

        memset(ماتریکس_لیکه_counts, 0, n_as * حجم(uint32_t));
        لټون_expert_ids(ids, n_ids, n_as, ماتریکس_لیکه_counts, ماتریکس_لیکې, mapping_stride);
    }

    mmctx->ماتریکس_لیکه_counts    = ماتریکس_لیکه_counts;
    mmctx->ماتریکس_لیکې          = ماتریکس_لیکې;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_لیکه_اندازه_padded = src0_لیکه_اندازه_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_پیل          = 0;
    mmctx->cur_m_لیکې           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    صحيح s;
    إذا (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } وإلا {
        uint32_t src0_لیکه_پیل = 0;
        uint32_t src0_لیکه_پای_ټکی   = src0_nrows;
        إذا (octx->ctx->mdev.شمېر > 1) {
            ثابت bool can_وېشل = htp_tensor_can_لیکه_برخه_کول(dst, حجم(عائم));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(src0_nrows, can_وېشل ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
            src0_لیکه_پیل = range.پیل;
            src0_لیکه_پای_ټکی   = range.پیل + range.شمېر;
        }

        إذا (src0_لیکه_پیل >= src0_لیکه_پای_ټکی) {
            إذا (mapping_buf != octx->ctx->ddr_spad_بنسټ) {
                حرر(mapping_buf);
            }
            أرجع HTP_حالت_OK;
        }

        ثابت uint32_t nrows = src0_لیکه_پای_ټکی - src0_لیکه_پیل;
        mmctx->src0_لیکه_پیل = src0_لیکه_پیل;
        mmctx->src0_لیکه_پای_ټکی   = src0_لیکه_پای_ټکی;

        mmctx->src0_nrows_فی_thread = fastdiv(nrows + octx->n_تارونه - 1, &octx->n_تارونه_div);
        mmctx->src0_nrows_فی_thread = hex_پړاو_پورته(mmctx->src0_nrows_فی_thread, 32);

        إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } وإلا {
            s = HTP_حالت_NO_SUPPORT;
        }
    }

    إذا (mapping_buf != octx->ctx->ddr_spad_بنسټ) {
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
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, 0);

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

    ثابت size_t src0_لیکه_اندازه = src0->nb[1];
    ثابت size_t src0_لیکه_اندازه_padded = hex_پړاو_پورته(src0_لیکه_اندازه, 128);

    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ثابت صحيح n_ids = ids->ne[0];
    ثابت صحيح n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_بنسټ;
    uint32_t   mapping_stride    = 1;
    uint32_t * ماتریکس_لیکه_counts = (uint32_t *) mapping_buf;
    بنية mmid_لیکه_mapping * ماتریکس_لیکې = عدم;

    إذا (act_nrows > 1) {
        ثابت size_t ماتریکس_لیکه_counts_اندازه = n_as * حجم(uint32_t);
        أكد(octx->ctx->ddr_spad_اندازه >= ماتریکس_لیکه_counts_اندازه);

        hex_l2fetch_بلاک((ثابت فراغ *) ids->معلومات, ids->ne[1] * ids->nb[1]);

        memset(ماتریکس_لیکه_counts, 0, ماتریکس_لیکه_counts_اندازه);
        لټون_expert_ids(ids, n_ids, n_as, ماتریکس_لیکه_counts, عدم, 0);

        uint32_t max_شمېر = hvx_reduce_max_i32((ثابت uint8_t *) ماتریکس_لیکه_counts, n_as);
        mapping_stride = max_شمېر > 0 ? max_شمېر : 1;

        size_t ماتریکس_لیکه_map_اندازه  = n_as * mapping_stride * حجم(بنية mmid_لیکه_mapping);
        ثابت size_t ټول_map_اندازه = ماتریکس_لیکه_counts_اندازه + ماتریکس_لیکه_map_اندازه;

        إذا (ټول_map_اندازه > octx->ctx->ddr_spad_اندازه) {
            mapping_buf = memalign(128, ټول_map_اندازه);
            إذا (!mapping_buf) {
                أرجع HTP_حالت_دننی_ERR;
            }
        }

        ماتریکس_لیکه_counts = (uint32_t *) mapping_buf;
        ماتریکس_لیکې       = (بنية mmid_لیکه_mapping *) (mapping_buf + ماتریکس_لیکه_counts_اندازه);

        memset(ماتریکس_لیکه_counts, 0, n_as * حجم(uint32_t));
        لټون_expert_ids(ids, n_ids, n_as, ماتریکس_لیکه_counts, ماتریکس_لیکې, mapping_stride);
    }

    mmctx->ماتریکس_لیکه_counts    = ماتریکس_لیکه_counts;
    mmctx->ماتریکس_لیکې          = ماتریکس_لیکې;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_لیکه_اندازه_padded = src0_لیکه_اندازه_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_پیل          = 0;
    mmctx->cur_m_لیکې           = act_nrows;

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

    إذا (mapping_buf != octx->ctx->ddr_spad_بنسټ) {
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
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت uint32_t n_weights = kparams->n_weights;

    ثابت بنية htp_tensor * محدود src0 = octx->src[0]; // first weight
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights]; // activation x

    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_پیل = 0;
    mmctx->cur_m_لیکې  = act_nrows;

    ثابت size_t src0_لیکه_اندازه = src0->nb[1];

    إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ثابت uint32_t ټول_nb = act_nrows * nb;

    کارګر_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    إذا (act_nrows < octx->n_تارونه) {
        n_quant_tasks = MIN(ټول_nb, octx->n_تارونه);
        quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_لومړی = (ټول_nb * ith) / n_quant_tasks;
            uint32_t ib_وروستی  = (ټول_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_لومړی[ith] = ib_لومړی;
            mmctx->quant_ib_وروستی[ith]  = ib_وروستی;
            mmctx->quant_r[ith]        = ib_لومړی / nb;
            mmctx->quant_c[ith]        = ib_لومړی % nb;
        }
    } وإلا {
        n_quant_tasks = MIN(act_nrows, octx->n_تارونه);
        quant_task_func = htp_mm_act_quant_لیکه_func(src0->type, act->type);
    }

    ثابت size_t act_لیکه_اندازه = htp_mm_tiled_act_لیکه_اندازه(src0->type, act->ne[0]);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_تارونه,
                                 0, src0_لیکه_اندازه, act_لیکه_اندازه, 0, kparams->n_prefetch, false, true);

    ثابت size_t vtcm_اندازه = L.ټول_بایټونه;

    إذا (octx->ctx->vtcm_اندازه < vtcm_اندازه) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_اندازه, vtcm_اندازه);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_لیکه_اندازه;
    mmctx->vtcm_act_raw_stride = hex_پړاو_پورته(act->ne[0] * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_اندازه_فی_thread = fastdiv(L.src0_بایټونه, &octx->n_تارونه_div);
    mmctx->vtcm_act_اندازه_فی_thread  = L.act_بایټونه;
    mmctx->vtcm_dst_اندازه_فی_thread  = fastdiv(L.dst_بایټونه, &octx->n_تارونه_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_لیکې_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_کتار_پرلپسې(octx->ctx->work_کتار, quant_task_func, mmctx, n_quant_tasks);

    work_کتار_پرلپسې(octx->ctx->work_کتار, hvx_mm_nx_2d_tiled, mmctx, octx->n_تارونه);

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

        htp_mm_tensor_collapse_لیکې(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        لكل (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ثابت بنية htp_tensor * d = octx->dsts[p];
            ثابت uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_لیکې(&dsts_collapsed[p], d, sd);
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
