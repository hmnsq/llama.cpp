#رهنمود clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

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

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "matmul-ops.h"
#ضمن "htp-vtcm.h"

تعريفنوع بنية {
    عائم        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_بایتها;
    dma_addr_t    act_dma_addr;
    dma_addr_t    وزن;
    dma_صف *   وزن_dma;
    صحيح           m;
    صحيح           k;
    صحيح           n;
    صحيح           act_stride;
    uint32_t      act_elem_اندازه;  // 4=F32 src1, 2=F16 src1
    صحيح           وزن_stride;
    صحيح           وزن_type;
    صحيح           dst_stride;
    صحيح           dst_تعداد_ستون;
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
    بنية fastdiv_مقادیر div_r2;
    بنية fastdiv_مقادیر div_r3;
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
         ثابت فراغ * محدود vy, uint32_t valid_ردیفها,
         ثابت عائم * محدود sz);

    فراغ (*vec_dot_32x2)(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1,
         ثابت فراغ * محدود vx,
         ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_ردیفها,
         ثابت عائم * محدود sz0, ثابت عائم * محدود sz1);

    // Precomputed values
    uint32_t src0_nrows_در_هر_thread;
    uint32_t src0_ردیف_شروع;
    uint32_t src0_ردیف_انتها;
    uint32_t src0_ردیف_اندازه_padded;
    uint32_t act_nrows;
    uint32_t cur_m_شروع;
    uint32_t cur_m_ردیفها;

    بنية fastdiv_مقادیر mm_div_ne12_ne1;
    بنية fastdiv_مقادیر mm_div_ne1;
    بنية fastdiv_مقادیر mm_div_r2;
    بنية fastdiv_مقادیر mm_div_r3;
    بنية fastdiv_مقادیر mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_اولی[WORK_صف_MAX_N_نخها];
    uint32_t          quant_ib_آخرین[WORK_صف_MAX_N_نخها];
    uint32_t          quant_r[WORK_صف_MAX_N_نخها];
    uint32_t          quant_c[WORK_صف_MAX_N_نخها];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_ردیفها_در_هر_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ثابت uint32_t * ماتریس_ردیف_counts;
    ثابت بنية mmid_ردیف_mapping * ماتریس_ردیفها;
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
    uint32_t vtcm_src0_اندازه_در_هر_thread;
    uint32_t vtcm_act_اندازه_در_هر_thread;
    uint32_t vtcm_dst_اندازه_در_هر_thread;
};

ساكن صحيح htp_mm_init_context(
    بنية htp_ops_context * octx,
    ثابت بنية htp_mm_kernel_params * kparams
) {
    إذا (!htp_ops_context_مجموعه_n_نخها(octx, (uint32_t) kparams->n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    إذا (kparams->n_hmx) {
        إذا (kparams->n_act_نخها <= 0 || kparams->n_act_نخها > (int32_t) octx->n_نخها) {
            أرجع HTP_وضعیت_INVAL_PARAMS;
        }
    }

    أرجع HTP_وضعیت_OK;
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

#تعریف_کن htp_matmul_tensors_preamble                                 \
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

#تعریف_کن htp_matmul_preamble                                         \
    بنية htp_mm_context * mmctx  = داده;                          \
    بنية htp_ops_context * octx  = mmctx->octx;                   \
    dma_صف * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_در_هر_thread = mmctx->src0_nrows_در_هر_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#ضمن "hvx-mm-kernels-tiled.h"
#ضمن "hvx-mm-kernels-float.h"
#ضمن "hmx-mm-kernels-tiled.h"

ساكن فراغ hvx_mm_2d_tiled(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_ردیف_انتها - mmctx->src0_ردیف_شروع;
    ثابت uint32_t act_nrows = mmctx->cur_m_ردیفها ? mmctx->cur_m_ردیفها : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_شروع = mmctx->cur_m_شروع;

    ثابت uint32_t src0_شروع_ردیف  = mmctx->src0_ردیف_شروع + src0_nrows_در_هر_thread * ith;
    ثابت uint32_t src0_انتها_ردیف    = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, mmctx->src0_ردیف_انتها);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_ردیف_اندازه  = nb1;
    ثابت size_t act_ردیف_اندازه = nb11;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;
    ثابت size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * محدود vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_اندازه_در_هر_thread  * ith;
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * محدود act_داده = mmctx->vtcm_act;

    ثابت dma_addr_t src0_ردیف = src0->داده;

    ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src0->type);
    ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_ردیف_stride = n_k_tiles_w * tile_اندازه;
    uint32_t tile_ردیف_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

    uint32_t ct_شروع = src0_شروع_ردیف / 32;
    uint32_t ct_انتها   = (src0_انتها_ردیف + 31) / 32;

    uint32_t افزودن_ct = ct_شروع;
    إذا (src0_شروع_ردیف < src0_انتها_ردیف) {
        لكل (uint32_t d = 0; d < n_prefetch && افزودن_ct < ct_انتها; d++, افزودن_ct++) {
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + d * tile_ردیف_transfer_اندازه_aligned,
                           src0_ردیف + افزودن_ct * tile_ردیف_stride), aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }
    }

    إذا (src0_شروع_ردیف >= src0_انتها_ردیف) {
        أرجع;
    }

    لكل (uint32_t ct = ct_شروع; ct < ct_انتها; ct++) {
        ثابت uint8_t * w_tile = (فراغ *) dma_صف_برداشت(dma_q).dst;

        صحيح valid_ردیفها = (صحيح)ne0 - (صحيح)(ct * 32);
        valid_ردیفها = MIN(32, MAX(0, valid_ردیفها));

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
            ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_داده + (ir1+0) * act_stride);
            ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_داده + (ir1+1) * act_stride);
            عائم * محدود dst_row0 = (عائم *) (dst->داده + ((cur_m_شروع + ir1+0) * dst_ردیف_اندازه));
            عائم * محدود dst_row1 = (عائم *) (dst->داده + ((cur_m_شروع + ir1+1) * dst_ردیف_اندازه));

            عائم * dst_ptr0 = &dst_row0[ct * 32];
            عائم * dst_ptr1 = &dst_row1[ct * 32];

            ثابت عائم * src2_ptr0 = عدم;
            ثابت عائم * src2_ptr1 = عدم;
            إذا (src2) {
                ثابت عائم * محدود src2_row0 = (ثابت عائم *) ((ثابت uint8_t *) src2->داده + ((cur_m_شروع + ir1+0) * src2_stride));
                ثابت عائم * محدود src2_row1 = (ثابت عائم *) ((ثابت uint8_t *) src2->داده + ((cur_m_شروع + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_ردیفها, src2_ptr0, src2_ptr1);
        }

        لكل (; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_داده + ir1 * act_stride);
            عائم * محدود dst_ردیف          = (عائم *) (dst->داده + ((cur_m_شروع + ir1) * dst_ردیف_اندازه));
            عائم * dst_ptr = &dst_ردیف[ct * 32];

            ثابت عائم * src2_ptr = عدم;
            إذا (src2) {
                ثابت عائم * محدود src2_ردیف = (ثابت عائم *) ((ثابت uint8_t *) src2->داده + ((cur_m_شروع + ir1) * src2_stride));
                src2_ptr = &src2_ردیف[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_ردیفها, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        إذا (افزودن_ct < ct_انتها) {
            dma_صف_افزودن(dma_q, dma_make_داده(w_tile, src0_ردیف + افزودن_ct * tile_ردیف_stride),
                           aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
            افزودن_ct++;
        }
    }
}

ساكن فراغ hvx_mv_2d_tiled(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_ردیف_انتها - mmctx->src0_ردیف_شروع;

    ثابت uint32_t src0_شروع_ردیف  = mmctx->src0_ردیف_شروع + src0_nrows_در_هر_thread * ith;
    ثابت uint32_t src0_انتها_ردیف    = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, mmctx->src0_ردیف_انتها);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_ردیف_اندازه  = nb1;
    ثابت size_t act_ردیف_اندازه = nb11;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_اندازه_در_هر_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * act_داده = mmctx->vtcm_act;

    عائم * tmp = (عائم *) vtcm_dst_ptr;

    ثابت dma_addr_t src0_ردیف = src0->داده;

    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_داده;
    عائم * محدود dst_ستون          = (عائم *) dst->داده;

    ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src0->type);
    ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_ردیف_stride = n_k_tiles_w * tile_اندازه;
    uint32_t tile_ردیف_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

    uint32_t ct_شروع = src0_شروع_ردیف / 32;
    uint32_t ct_انتها   = (src0_انتها_ردیف + 31) / 32;

    uint32_t افزودن_ct = ct_شروع;
    إذا (src0_شروع_ردیف < src0_انتها_ردیف) {
        إذا (src2) {
            عائم * vtcm_bias_ptr = (عائم *) mmctx->vtcm_bias + src0_شروع_ردیف;
            ثابت dma_addr_t src2_addr = src2->داده + src0_شروع_ردیف * حجم(عائم);
            صحيح برش_اندازه = (صحيح)MIN(src0_انتها_ردیف, ne0) - (صحيح)src0_شروع_ردیف;
            إذا (برش_اندازه > 0) {
                dma_صف_افزودن(dma_q, dma_make_داده(vtcm_bias_ptr, src2_addr),
                               برش_اندازه * حجم(عائم), برش_اندازه * حجم(عائم), برش_اندازه * حجم(عائم), 1);
                dma_صف_برداشت_nowait(dma_q);
            }
        }
        لكل (uint32_t d = 0; d < n_prefetch && افزودن_ct < ct_انتها; d++, افزودن_ct++) {
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + d * tile_ردیف_transfer_اندازه_aligned,
                           src0_ردیف + افزودن_ct * tile_ردیف_stride), aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }
    }

    إذا (src0_شروع_ردیف >= src0_انتها_ردیف) {
        أرجع;
    }

    لكل (uint32_t ct = ct_شروع; ct < ct_انتها; ct++) {
        ثابت uint8_t * w_tile = (فراغ *) dma_صف_برداشت(dma_q).dst;

        عائم * dst_ptr = &tmp[ct * 32 - src0_شروع_ردیف];
        صحيح valid_ردیفها = (صحيح)ne0 - (صحيح)(ct * 32);
        valid_ردیفها = MIN(32, MAX(0, valid_ردیفها));

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_ردیفها, عدم);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        إذا (افزودن_ct < ct_انتها) {
            dma_صف_افزودن(dma_q, dma_make_داده(w_tile, src0_ردیف + افزودن_ct * tile_ردیف_stride),
                           aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
            افزودن_ct++;
        }
    }

    صحيح copy_cnt = (صحيح)MIN(src0_انتها_ردیف, ne0) - (صحيح)src0_شروع_ردیف;
    إذا (copy_cnt > 0) {
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct_انتها);
        إذا (src2) {
            hvx_جمع_کن_f32_uuu((uint8_t *) &dst_ستون[src0_شروع_ردیف],
                            (ثابت uint8_t *) tmp,
                            (ثابت uint8_t *) ((ثابت عائم *) mmctx->vtcm_bias + src0_شروع_ردیف),
                            copy_cnt);
        } وإلا {
            hvx_copy_f32_ua((uint8_t *) &dst_ستون[src0_شروع_ردیف], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_انتها);
    }
}

ساكن فراغ hvx_mm_nx_2d_tiled(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_mm_context * mmctx = داده;
    بنية htp_ops_context * octx = mmctx->octx;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;

    ثابت بنية htp_tensor * محدود act = octx->src[n_weights]; /* x */
    ثابت uint32_t ne10 = act->ne[0];
    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_وزن_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * محدود act_داده        = mmctx->vtcm_act;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    لكل (uint32_t widx = 0; widx < n_weights; widx++) {
        ثابت بنية htp_tensor * محدود src_w = octx->src[widx];
        ثابت بنية htp_tensor * محدود dst   = octx->dsts[widx];
        إذا (!src_w || !dst) استمر;
        dma_صف * dma_q = octx->ctx->dma[ith];

        ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src_w->type);
        ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src_w->type);
        ثابت uint32_t ne00 = src_w->ne[0];
        ثابت uint32_t ne01 = src_w->ne[1];
        ثابت size_t dst_ردیف_اندازه = dst->nb[1];
        ثابت dma_addr_t src_w_ردیف = src_w->داده;

        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ثابت uint32_t tile_ردیف_stride = n_k_tiles_w * tile_اندازه;
        ثابت uint32_t tile_ردیف_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

        uint32_t src0_شروع_ردیف = 0;
        uint32_t src0_انتها_ردیف   = ne01;
        إذا (octx->ctx->mdev.شمار > 1) {
            ثابت bool can_تقسیم = htp_tensor_can_ردیف_افراز(dst, حجم(عائم));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(ne01, can_تقسیم ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
            src0_شروع_ردیف = range.شروع;
            src0_انتها_ردیف   = range.شروع + range.شمار;
        }

        ثابت uint32_t nrows = src0_انتها_ردیف - src0_شروع_ردیف;
        uint32_t src0_nrows_در_هر_thread = fastdiv(nrows + nth - 1, &octx->n_نخها_div);
        src0_nrows_در_هر_thread = hex_دور_به_بالا(src0_nrows_در_هر_thread, 32);

        ثابت uint32_t شروع_ردیف = src0_شروع_ردیف + src0_nrows_در_هر_thread * ith;
        ثابت uint32_t انتها_ردیف   = MIN(شروع_ردیف + src0_nrows_در_هر_thread, src0_انتها_ردیف);
        إذا (شروع_ردیف >= انتها_ردیف) استمر;

        uint32_t ct_شروع = شروع_ردیف / 32;
        uint32_t ct_انتها   = (انتها_ردیف + 31) / 32;

        uint32_t افزودن_ct = ct_شروع;
        لكل (uint32_t d = 0; d < n_prefetch && افزودن_ct < ct_انتها; d++, افزودن_ct++) {
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_وزن_ptr + d * tile_ردیف_transfer_اندازه_aligned,
                            src_w_ردیف + افزودن_ct * tile_ردیف_stride), aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_شروع; ct < ct_انتها; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_صف_برداشت(dma_q).dst;
            صحيح valid_ردیفها = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
            valid_ردیفها = MIN(32, MAX(0, valid_ردیفها));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
                ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_داده + (ir1+0) * act_stride);
                ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_داده + (ir1+1) * act_stride);

                عائم * محدود dst_row0 = (عائم *) (dst->داده + ((ir1+0) * dst_ردیف_اندازه));
                عائم * محدود dst_row1 = (عائم *) (dst->داده + ((ir1+1) * dst_ردیف_اندازه));
                عائم * dst_ptr0 = &dst_row0[ct * 32];
                عائم * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_ردیفها, عدم, عدم);
            }

            لكل (; ir1 < act_nrows; ++ir1) {
                ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_داده + ir1 * act_stride);
                عائم * محدود dst_ردیف = (عائم *) (dst->داده + (ir1 * dst_ردیف_اندازه));
                عائم * dst_ptr = &dst_ردیف[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_ردیفها, عدم);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (افزودن_ct < ct_انتها) {
                dma_صف_افزودن(dma_q, dma_make_داده(w_tile, src_w_ردیف + افزودن_ct * tile_ردیف_stride),
                                aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                افزودن_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mm_transfer_act_dma(
    بنية htp_ops_context * octx,
    ثابت بنية htp_mm_kernel_params * kparams,
    ثابت بنية htp_tensor * act,
    uint8_t * dst_پایه,
    size_t dst_ردیف_اندازه,
    uint32_t m_شروع,
    uint32_t m_ردیفها
) {
    إذا (m_ردیفها == 0) {
        أرجع;
    }

    dma_صف * dma_q = octx->ctx->dma[0];
    ثابت uint32_t ne0 = act->ne[0];
    ثابت size_t elem_اندازه = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت size_t ردیف_بایتها = ne0 * elem_اندازه;
    ثابت size_t act_nb1 = act->nb[1];
    ثابت dma_addr_t act_پایه = act->داده;

    ثابت bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    إذا (is_contiguous) {
        ثابت dma_addr_t src_addr = act_پایه + m_شروع * act_nb1;
        dma_صف_افزودن(dma_q, dma_make_داده(dst_پایه, src_addr),
                       dst_ردیف_اندازه, act_nb1, ردیف_بایتها, m_ردیفها);
        dma_صف_برداشت(dma_q);
    } وإلا {
        ثابت uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ثابت bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        لكل (uint32_t ir = 0; ir < m_ردیفها; ++ir) {
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
            ثابت dma_addr_t ردیف_src = act_پایه + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * ردیف_dst = dst_پایه + ir * dst_ردیف_اندازه;
            dma_صف_افزودن(dma_q, dma_make_داده(ردیف_dst, ردیف_src),
                           dst_ردیف_اندازه, act_nb1, ردیف_بایتها, 1);
            dma_صف_برداشت(dma_q);
        }
    }

    إذا (dst_ردیف_اندازه > ردیف_بایتها) {
        بنية htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_شروع);
        ثابت uint32_t pad_elems = (dst_ردیف_اندازه - ردیف_بایتها) / elem_اندازه;
        إذا (elem_اندازه == حجم(عائم)) {
            لكل (uint32_t ir = 0; ir < m_ردیفها; ++ir) {
                hvx_splat_f32_u(dst_پایه + ir * dst_ردیف_اندازه + ردیف_بایتها, 0.0f, pad_elems);
            }
        } وإلا {
            لكل (uint32_t ir = 0; ir < m_ردیفها; ++ir) {
                hvx_splat_f16_u(dst_پایه + ir * dst_ردیف_اندازه + ردیف_بایتها, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_شروع);
    }
}

#تعریف_کن QUANTIZE_IMPL(name, log_name, kernel_fn, dst_ردیف_اندازه_عبارت)                                        \
ساكن فراغ name(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {                                        \
    (فراغ) nth;                                                                                            \
    بنية htp_mm_context * mmctx = داده;                                                                  \
    بنية htp_ops_context * octx = mmctx->octx;                                                           \
    ثابت بنية htp_tensor * src = mmctx->act;                                                            \
    ثابت uint32_t ne0 = src->ne[0];                                                                       \
    ثابت uint32_t nrows = mmctx->cur_m_ردیفها ? mmctx->cur_m_ردیفها : mmctx->act_nrows;                       \
    ثابت uint32_t nrows_در_هر_thread = mmctx->n_quant_ردیفها_در_هر_thread;                                      \
                                                                                                           \
    ثابت uint32_t ir_اولی = nrows_در_هر_thread * ith;                                                      \
    إذا (ir_اولی >= nrows) {                                                                               \
        أرجع;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_اولی);                                        \
                                                                                                           \
    uint8_t * محدود dst = mmctx->vtcm_act;                                                              \
    ثابت uint32_t ir_آخرین = MIN(ir_اولی + nrows_در_هر_thread, nrows);                                      \
    ثابت size_t raw_ردیف_اندازه = mmctx->vtcm_act_raw_stride;                                                \
    ثابت size_t dst_ردیف_اندازه = (dst_ردیف_اندازه_عبارت);                                                       \
                                                                                                           \
    ثابت uint8_t * محدود src_داده = (ثابت uint8_t *) mmctx->vtcm_act_raw + (raw_ردیف_اندازه * ir_اولی); \
    uint8_t * محدود dst_داده = (uint8_t *) dst + (dst_ردیف_اندازه * ir_اولی);                             \
    kernel_fn(src_داده, dst_داده, عدم, ne0, ir_آخرین - ir_اولی, raw_ردیف_اندازه, dst_ردیف_اندازه);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_اولی);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled,     "quantize-f32-q8_0_tiled",     quantize_f32_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_ردیف_اندازه(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled,     "quantize-f32-q8_1_tiled",     quantize_f32_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_ردیف_اندازه(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_ردیف_اندازه(ne0))
QUANTIZE_IMPL(quantize_f32_f16_tiled,      "quantize-f32-f16_tiled",      quantize_f32_f16_tiled_kernel,      htp_mm_f16_tiled_ردیف_اندازه(ne0))
QUANTIZE_IMPL(quantize_f16_f16_tiled,      "quantize-f16-f16_tiled",      quantize_f16_f16_tiled_kernel,      htp_mm_f16_tiled_ردیف_اندازه(ne0))
QUANTIZE_IMPL(quantize_f32_f32,            "quantize-f32-f32",            quantize_f32_f32_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,            "quantize-f32-f16",            quantize_f32_f16_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,            "quantize-f16-f16",            quantize_f16_f16_kernel,            mmctx->vtcm_act_stride)

ساكن فراغ quantize_f32_q8_0_tiled_بلوک(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = داده;
    إذا (mmctx->quant_ib_اولی[ith] >= mmctx->quant_ib_آخرین[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_اولی[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_بلوک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_اولی[ith],
        mmctx->quant_ib_آخرین[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_ردیف_اندازه(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_اولی[ith]);
}

ساكن فراغ quantize_f32_q8_1_tiled_بلوک(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = داده;
    إذا (mmctx->quant_ib_اولی[ith] >= mmctx->quant_ib_آخرین[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_اولی[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_بلوک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_اولی[ith],
        mmctx->quant_ib_آخرین[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_ردیف_اندازه(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_اولی[ith]);
}

ساكن فراغ quantize_f32_q8_1_s16_tiled_بلوک(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = داده;
    إذا (mmctx->quant_ib_اولی[ith] >= mmctx->quant_ib_آخرین[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_اولی[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_بلوک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_اولی[ith],
        mmctx->quant_ib_آخرین[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_ردیف_اندازه(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_اولی[ith]);
}

ساكن فراغ quantize_f32_f16_tiled_بلوک(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = داده;
    إذا (mmctx->quant_ib_اولی[ith] >= mmctx->quant_ib_آخرین[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_اولی[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_بلوک_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_اولی[ith],
        mmctx->quant_ib_آخرین[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_ردیف_اندازه(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_اولی[ith]);
}

ساكن فراغ quantize_f16_f16_tiled_بلوک(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = داده;
    إذا (mmctx->quant_ib_اولی[ith] >= mmctx->quant_ib_آخرین[ith]) {
        أرجع;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_اولی[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_بلوک_kernel(
        (ثابت __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_اولی[ith],
        mmctx->quant_ib_آخرین[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_ردیف_اندازه(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_اولی[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
ساكن مضمن کارگر_callback_t htp_mm_act_quant_ردیف_func(صحيح وزن_type, صحيح act_type) {
    إذا (وزن_type == HTP_TYPE_F16) {
        أرجع (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    إذا (وزن_type == HTP_TYPE_Q2_K) {
        أرجع quantize_f32_q8_1_s16_tiled;
    }
    أرجع htp_mm_وزن_has_جابه_جایی(وزن_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

ساكن مضمن کارگر_callback_t htp_mm_act_quant_بلوک_func(صحيح وزن_type, صحيح act_type) {
    إذا (وزن_type == HTP_TYPE_F16) {
        أرجع (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_بلوک : quantize_f32_f16_tiled_بلوک;
    }
    إذا (وزن_type == HTP_TYPE_Q2_K) {
        أرجع quantize_f32_q8_1_s16_tiled_بلوک;
    }
    أرجع htp_mm_وزن_has_جابه_جایی(وزن_type) ? quantize_f32_q8_1_tiled_بلوک : quantize_f32_q8_0_tiled_بلوک;
}

ساكن فراغ hvx_mm_4d_tiled(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_ردیف_انتها - mmctx->src0_ردیف_شروع;
    ثابت uint32_t cur_m_ردیفها = mmctx->cur_m_ردیفها ? mmctx->cur_m_ردیفها : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_شروع = mmctx->cur_m_شروع;

    ثابت uint32_t src0_شروع_ردیف  = mmctx->src0_ردیف_شروع + src0_nrows_در_هر_thread * ith;
    ثابت uint32_t src0_انتها_ردیف    = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, mmctx->src0_ردیف_انتها);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_ردیف_اندازه = nb1;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * محدود act_داده = mmctx->vtcm_act;

    ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src0->type);
    ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src0->type);

    ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ثابت uint32_t tile_ردیف_stride = n_k_tiles_w * tile_اندازه;
    ثابت uint32_t tile_ردیف_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;
    ثابت uint32_t src0_برش_stride = ((ne01 + 31) / 32) * tile_ردیف_stride;

    ثابت uint32_t ct_شروع = src0_شروع_ردیف / 32;
    ثابت uint32_t ct_انتها   = (src0_انتها_ردیف + 31) / 32;

    إذا (src0_شروع_ردیف >= src0_انتها_ردیف || cur_m_ردیفها == 0) {
        أرجع;
    }

    ثابت uint32_t مجموع_batches = ne12 * ne13;
    ثابت uint32_t b_شروع = fastdiv(cur_m_شروع, &kparams->div_ne1);
    uint32_t b_انتها = fastdiv(cur_m_شروع + cur_m_ردیفها + ne11 - 1, &kparams->div_ne1);
    b_انتها = MIN(b_انتها, مجموع_batches);

    uint32_t b_grp_شروع = b_شروع;
    طالما (b_grp_شروع < b_انتها) {
        ثابت uint32_t b3 = fastdiv(b_grp_شروع, &kparams->div_ne12);
        ثابت uint32_t b2 = b_grp_شروع - b3 * ne12;
        ثابت uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ثابت uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_انتها = b_grp_شروع + 1;
        طالما (b_grp_انتها < b_انتها) {
            ثابت uint32_t cur_b3 = fastdiv(b_grp_انتها, &kparams->div_ne12);
            ثابت uint32_t cur_b2 = b_grp_انتها - cur_b3 * ne12;
            ثابت uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ثابت uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            إذا (cur_i02 != i02 || cur_i03 != i03) {
                اكسر;
            }
            b_grp_انتها++;
        }

        ثابت uint32_t برش_idx = i03 * ne02 + i02;
        ثابت dma_addr_t src0_برش = src0->داده + (size_t) برش_idx * src0_برش_stride;

        uint32_t افزودن_ct = ct_شروع;
        لكل (uint32_t d = 0; d < n_prefetch && افزودن_ct < ct_انتها; d++, افزودن_ct++) {
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + d * tile_ردیف_transfer_اندازه_aligned,
                           src0_برش + (size_t) افزودن_ct * tile_ردیف_stride),
                           aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_شروع; ct < ct_انتها; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_صف_برداشت(dma_q).dst;

            صحيح valid_ردیفها = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_ردیفها = MIN(32, MAX(0, valid_ردیفها));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            لكل (uint32_t b = b_grp_شروع; b < b_grp_انتها; b++) {
                ثابت uint32_t b_m_شروع = b * ne11;
                ثابت uint32_t m_اولی = MAX(cur_m_شروع, b_m_شروع);
                ثابت uint32_t m_آخرین  = MIN(cur_m_شروع + cur_m_ردیفها, b_m_شروع + ne11);
                إذا (m_اولی >= m_آخرین) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_پایه = (uint8_t *) dst->داده + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_جابه_جایی = m_اولی - cur_m_شروع;
                ثابت uint32_t dst_m_جابه_جایی   = m_اولی - b_m_شروع;
                ثابت uint32_t batch_nrows    = m_آخرین - m_اولی;

                uint32_t ir1 = 0;
                لكل (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_داده + (chunk_m_جابه_جایی + ir1 + 0) * act_stride);
                    ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_داده + (chunk_m_جابه_جایی + ir1 + 1) * act_stride);
                    عائم * محدود dst_row0 = (عائم *) (dst_batch_پایه + (dst_m_جابه_جایی + ir1 + 0) * dst_ردیف_اندازه);
                    عائم * محدود dst_row1 = (عائم *) (dst_batch_پایه + (dst_m_جابه_جایی + ir1 + 1) * dst_ردیف_اندازه);
                    عائم * dst_ptr0 = &dst_row0[ct * 32];
                    عائم * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_ردیفها, عدم, عدم);
                }
                لكل (; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_داده + (chunk_m_جابه_جایی + ir1) * act_stride);
                    عائم * محدود dst_ردیف = (عائم *) (dst_batch_پایه + (dst_m_جابه_جایی + ir1) * dst_ردیف_اندازه);
                    عائم * dst_ptr = &dst_ردیف[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_ردیفها, عدم);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (افزودن_ct < ct_انتها) {
                dma_صف_افزودن(dma_q, dma_make_داده(w_tile, src0_برش + (size_t) افزودن_ct * tile_ردیف_stride),
                               aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                افزودن_ct++;
            }
        }
        b_grp_شروع = b_grp_انتها;
    }
    إذا (src2) {
        hvx_tensor_جمع_کن_f32_grid(dst, src2, cur_m_شروع, cur_m_شروع + cur_m_ردیفها, src0_شروع_ردیف, src0_انتها_ردیف, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ساكن فراغ hvx_mm_2d_flat(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_matmul_preamble;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_ماسک = n_prefetch - 1;

    ثابت uint32_t src0_nrows = mmctx->src0_ردیف_انتها - mmctx->src0_ردیف_شروع;  // src0 rows
    ثابت uint32_t act_nrows = mmctx->cur_m_ردیفها ? mmctx->cur_m_ردیفها : mmctx->act_nrows;
    ثابت uint32_t cur_m_شروع = mmctx->cur_m_شروع;

    ثابت uint32_t src0_شروع_ردیف  = mmctx->src0_ردیف_شروع + src0_nrows_در_هر_thread * ith;
    ثابت uint32_t src0_انتها_ردیف    = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, mmctx->src0_ردیف_انتها);
    ثابت uint32_t src0_انتها_ردیف_x2 = src0_شروع_ردیف + ((src0_انتها_ردیف - src0_شروع_ردیف) & ~1U);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_ردیف_اندازه  = nb1;
    ثابت size_t src0_ردیف_اندازه = nb01;
    ثابت size_t act_ردیف_اندازه = nb11;

    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_اندازه_در_هر_thread  * ith;
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * محدود act_داده      = mmctx->vtcm_act;

    ثابت dma_addr_t src0_ردیف = src0->داده;


    // Prefill vtcm with src0 rows
    إذا (src0_شروع_ردیف < src0_انتها_ردیف) {
        لكل (uint32_t ir0 = src0_شروع_ردیف; ir0 < src0_انتها_ردیف_x2; ir0 += 2) {
            ثابت صحيح is0 = (ir0 - src0_شروع_ردیف);
            إذا (is0 >= (صحيح)n_prefetch) {
                اكسر;
            }
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + is0 * src0_stride, src0_ردیف + ir0 * src0_ردیف_اندازه),
                           src0_stride, src0_ردیف_اندازه, src0_ردیف_اندازه, 2);
        }
    }

    إذا (src0_شروع_ردیف >= src0_انتها_ردیف) {
        أرجع;
    }

    // Process src0 rows
    لكل (uint32_t ir0 = src0_شروع_ردیف; ir0 < src0_انتها_ردیف_x2; ir0 += 2) {
        ثابت uint8_t * ss0 = (فراغ *) dma_صف_برداشت(dma_q).dst;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
            ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_داده + (ir1+0) * act_stride);
            ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_داده + (ir1+1) * act_stride);
            عائم * محدود dst_row0 = (عائم *) (dst->داده + ((cur_m_شروع + ir1+0) * dst_ردیف_اندازه));
            عائم * محدود dst_row1 = (عائم *) (dst->داده + ((cur_m_شروع + ir1+1) * dst_ردیف_اندازه));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        لكل (; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_داده + ir1 * act_stride);
            عائم * محدود dst_ردیف          = (عائم *) (dst->داده + ((cur_m_شروع + ir1) * dst_ردیف_اندازه));
            mmctx->vec_dot_2x1(ne00, &dst_ردیف[ir0], ss0, ss0 + src0_stride, act_ستون);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ثابت صحيح pr0 = (ir0 + n_prefetch);
        ثابت صحيح is0 = (pr0 - src0_شروع_ردیف) & prefetch_ماسک;
        إذا (pr0 < src0_انتها_ردیف_x2) {
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + is0 * src0_stride, src0_ردیف + pr0 * src0_ردیف_اندازه),
                           src0_stride, src0_ردیف_اندازه, src0_ردیف_اندازه, 2);
        }
    }

    // Process the last row (if any)
    إذا (src0_انتها_ردیف != src0_انتها_ردیف_x2) {
        uint32_t  ir0 = src0_انتها_ردیف_x2;
        ثابت صحيح is0 = (ir0 - src0_شروع_ردیف) & prefetch_ماسک;
        dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + is0 * src0_stride, src0_ردیف + ir0 * src0_ردیف_اندازه),
                       src0_stride, src0_ردیف_اندازه, src0_ردیف_اندازه, 1);
        ثابت uint8_t * ss0 = (فراغ *) dma_صف_برداشت(dma_q).dst;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #رهنمود unroll(2)
        لكل (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_داده + ir1 * act_stride);
            عائم * محدود dst_ردیف          = (عائم *) (dst->داده + ((cur_m_شروع + ir1) * dst_ردیف_اندازه));
            mmctx->vec_dot_1x1(ne00, &dst_ردیف[ir0], ss0, act_ستون);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    إذا (src2) {
        hvx_tensor_جمع_کن_f32_grid(dst, src2, cur_m_شروع, cur_m_شروع + act_nrows, src0_شروع_ردیف, src0_انتها_ردیف, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ساكن فراغ hvx_mv_2d_flat(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_ردیف_انتها - mmctx->src0_ردیف_شروع;

    ثابت uint32_t src0_شروع_ردیف  = mmctx->src0_ردیف_شروع + src0_nrows_در_هر_thread * ith;
    ثابت uint32_t src0_انتها_ردیف    = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, mmctx->src0_ردیف_انتها);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_ردیف_اندازه  = nb1;
    ثابت size_t src0_ردیف_اندازه = nb01;
    ثابت size_t act_ردیف_اندازه  = nb11;

    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_اندازه_در_هر_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * act_داده      = mmctx->vtcm_act;

    عائم * tmp = (عائم *) vtcm_dst_ptr;

    ثابت dma_addr_t src0_ردیف = src0->داده;
    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_داده;
    عائم * محدود dst_ستون         = (عائم *) dst->داده;

    ثابت uint32_t src0_انتها_ردیف_x2 = src0_شروع_ردیف + ((src0_انتها_ردیف - src0_شروع_ردیف) & ~1U);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_ماسک = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    إذا (src0_شروع_ردیف < src0_انتها_ردیف) {
        إذا (src2) {
            عائم * vtcm_bias_ptr = (عائم *) mmctx->vtcm_bias + src0_شروع_ردیف;
            ثابت dma_addr_t src2_addr = src2->داده + src0_شروع_ردیف * حجم(عائم);
            صحيح برش_اندازه = (صحيح)src0_انتها_ردیف - (صحيح)src0_شروع_ردیف;
            إذا (برش_اندازه > 0) {
                dma_صف_افزودن(dma_q, dma_make_داده(vtcm_bias_ptr, src2_addr),
                               برش_اندازه * حجم(عائم), برش_اندازه * حجم(عائم), برش_اندازه * حجم(عائم), 1);
                dma_صف_برداشت_nowait(dma_q);
            }
        }
        لكل (uint32_t ir0 = src0_شروع_ردیف; ir0 < src0_انتها_ردیف_x2; ir0 += 2) {
            ثابت uint32_t is0 = (ir0 - src0_شروع_ردیف);
            إذا (is0 >= n_prefetch) {
                اكسر;
            }
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + is0 * src0_stride, src0_ردیف + ir0 * src0_ردیف_اندازه),
                           src0_stride, src0_ردیف_اندازه, src0_ردیف_اندازه, 2);
        }
    }


    إذا (src0_شروع_ردیف >= src0_انتها_ردیف) {
        أرجع;
    }

    // Process src0 rows
    لكل (uint32_t ir0 = src0_شروع_ردیف; ir0 < src0_انتها_ردیف_x2; ir0 += 2) {
        ثابت uint8_t * ss0 = (فراغ *) dma_صف_برداشت(dma_q).dst;
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_شروع_ردیف], ss0, ss0 + src0_stride, act_ستون);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ثابت uint32_t pr0 = (ir0 + n_prefetch);
        ثابت uint32_t is0 = (pr0 - src0_شروع_ردیف) & prefetch_ماسک;
        إذا (pr0 < src0_انتها_ردیف_x2) {
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + is0 * src0_stride, src0_ردیف + pr0 * src0_ردیف_اندازه),
                           src0_stride, src0_ردیف_اندازه, src0_ردیف_اندازه, 2);
        }
    }

    // Process the last row (if any)
    إذا (src0_انتها_ردیف != src0_انتها_ردیف_x2) {
        ثابت uint32_t ir0 = src0_انتها_ردیف_x2;
        ثابت uint32_t is0 = (ir0 - src0_شروع_ردیف) & prefetch_ماسک;
        dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + is0 * src0_stride, src0_ردیف + ir0 * src0_ردیف_اندازه),
                       src0_stride, src0_ردیف_اندازه, src0_ردیف_اندازه, 1);
        ثابت uint8_t * ss0 = (فراغ *) dma_صف_برداشت(dma_q).dst;
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_شروع_ردیف], ss0, act_ستون);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    صحيح copy_cnt = src0_انتها_ردیف - src0_شروع_ردیف;
    إذا (copy_cnt > 0) {
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, src0_انتها_ردیف);
        إذا (src2) {
            hvx_جمع_کن_f32_uuu((uint8_t *) &dst_ستون[src0_شروع_ردیف],
                            (ثابت uint8_t *) tmp,
                            (ثابت uint8_t *) ((ثابت عائم *) mmctx->vtcm_bias + src0_شروع_ردیف),
                            copy_cnt);
        } وإلا {
            hvx_copy_f32_ua((uint8_t *) &dst_ستون[src0_شروع_ردیف], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_انتها_ردیف);
    }
}

ساكن فراغ hvx_mm_4d_flat(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_matmul_preamble;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_ماسک = n_prefetch - 1;

    ثابت uint32_t src0_nrows = mmctx->src0_ردیف_انتها - mmctx->src0_ردیف_شروع;
    ثابت uint32_t cur_m_ردیفها = mmctx->cur_m_ردیفها ? mmctx->cur_m_ردیفها : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_شروع = mmctx->cur_m_شروع;

    ثابت uint32_t src0_شروع_ردیف  = mmctx->src0_ردیف_شروع + src0_nrows_در_هر_thread * ith;
    ثابت uint32_t src0_انتها_ردیف    = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, mmctx->src0_ردیف_انتها);
    ثابت uint32_t src0_انتها_ردیف_x2 = src0_شروع_ردیف + ((src0_انتها_ردیف - src0_شروع_ردیف) & ~1U);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_ردیف_اندازه  = nb1;
    ثابت size_t src0_ردیف_اندازه = nb01;
    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * محدود act_داده      = mmctx->vtcm_act;


    إذا (src0_شروع_ردیف >= src0_انتها_ردیف || cur_m_ردیفها == 0) {
        أرجع;
    }

    ثابت uint32_t مجموع_batches = ne12 * ne13;
    ثابت uint32_t b_شروع = fastdiv(cur_m_شروع, &kparams->div_ne1);
    uint32_t b_انتها = fastdiv(cur_m_شروع + cur_m_ردیفها + ne11 - 1, &kparams->div_ne1);
    b_انتها = MIN(b_انتها, مجموع_batches);

    uint32_t b_grp_شروع = b_شروع;
    طالما (b_grp_شروع < b_انتها) {
        ثابت uint32_t b3 = fastdiv(b_grp_شروع, &kparams->div_ne12);
        ثابت uint32_t b2 = b_grp_شروع - b3 * ne12;
        ثابت uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ثابت uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_انتها = b_grp_شروع + 1;
        طالما (b_grp_انتها < b_انتها) {
            ثابت uint32_t cur_b3 = fastdiv(b_grp_انتها, &kparams->div_ne12);
            ثابت uint32_t cur_b2 = b_grp_انتها - cur_b3 * ne12;
            ثابت uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ثابت uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            إذا (cur_i02 != i02 || cur_i03 != i03) {
                اكسر;
            }
            b_grp_انتها++;
        }

        ثابت dma_addr_t src0_ردیف = src0->داده + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        لكل (uint32_t ir0 = src0_شروع_ردیف; ir0 < src0_انتها_ردیف_x2; ir0 += 2) {
            ثابت صحيح is0 = (ir0 - src0_شروع_ردیف);
            إذا (is0 >= (صحيح)n_prefetch) {
                اكسر;
            }
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + is0 * src0_stride, src0_ردیف + (size_t) ir0 * src0_ردیف_اندازه),
                           src0_stride, src0_ردیف_اندازه, src0_ردیف_اندازه, 2);
        }

        لكل (uint32_t ir0 = src0_شروع_ردیف; ir0 < src0_انتها_ردیف_x2; ir0 += 2) {
            ثابت uint8_t * ss0 = (فراغ *) dma_صف_برداشت(dma_q).dst;

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            لكل (uint32_t b = b_grp_شروع; b < b_grp_انتها; b++) {
                ثابت uint32_t b_m_شروع = b * ne11;
                ثابت uint32_t m_اولی = MAX(cur_m_شروع, b_m_شروع);
                ثابت uint32_t m_آخرین  = MIN(cur_m_شروع + cur_m_ردیفها, b_m_شروع + ne11);
                إذا (m_اولی >= m_آخرین) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_پایه = (uint8_t *) dst->داده + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_جابه_جایی = m_اولی - cur_m_شروع;
                ثابت uint32_t dst_m_جابه_جایی   = m_اولی - b_m_شروع;
                ثابت uint32_t batch_nrows    = m_آخرین - m_اولی;

                uint32_t ir1 = 0;
                لكل (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_داده + (chunk_m_جابه_جایی + ir1 + 0) * act_stride);
                    ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_داده + (chunk_m_جابه_جایی + ir1 + 1) * act_stride);
                    عائم * محدود dst_row0 = (عائم *) (dst_batch_پایه + (dst_m_جابه_جایی + ir1 + 0) * dst_ردیف_اندازه);
                    عائم * محدود dst_row1 = (عائم *) (dst_batch_پایه + (dst_m_جابه_جایی + ir1 + 1) * dst_ردیف_اندازه);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                لكل (; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_داده + (chunk_m_جابه_جایی + ir1) * act_stride);
                    عائم * محدود dst_ردیف          = (عائم *) (dst_batch_پایه + (dst_m_جابه_جایی + ir1) * dst_ردیف_اندازه);
                    mmctx->vec_dot_2x1(ne00, &dst_ردیف[ir0], ss0, ss0 + src0_stride, act_ستون);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ثابت صحيح pr0 = (ir0 + n_prefetch);
            ثابت صحيح is0 = (pr0 - src0_شروع_ردیف) & prefetch_ماسک;
            إذا (pr0 < src0_انتها_ردیف_x2) {
                dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + is0 * src0_stride, src0_ردیف + (size_t) pr0 * src0_ردیف_اندازه),
                               src0_stride, src0_ردیف_اندازه, src0_ردیف_اندازه, 2);
            }
        }

        إذا (src0_انتها_ردیف != src0_انتها_ردیف_x2) {
            uint32_t ir0 = src0_انتها_ردیف_x2;
            ثابت صحيح is0 = (ir0 - src0_شروع_ردیف) & prefetch_ماسک;
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + is0 * src0_stride, src0_ردیف + (size_t) ir0 * src0_ردیف_اندازه),
                           src0_stride, src0_ردیف_اندازه, src0_ردیف_اندازه, 1);
            ثابت uint8_t * ss0 = (فراغ *) dma_صف_برداشت(dma_q).dst;

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            لكل (uint32_t b = b_grp_شروع; b < b_grp_انتها; b++) {
                ثابت uint32_t b_m_شروع = b * ne11;
                ثابت uint32_t m_اولی = MAX(cur_m_شروع, b_m_شروع);
                ثابت uint32_t m_آخرین  = MIN(cur_m_شروع + cur_m_ردیفها, b_m_شروع + ne11);
                إذا (m_اولی >= m_آخرین) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_پایه = (uint8_t *) dst->داده + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_جابه_جایی = m_اولی - cur_m_شروع;
                ثابت uint32_t dst_m_جابه_جایی   = m_اولی - b_m_شروع;
                ثابت uint32_t batch_nrows    = m_آخرین - m_اولی;

                لكل (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_داده + (chunk_m_جابه_جایی + ir1) * act_stride);
                    عائم * محدود dst_ردیف          = (عائم *) (dst_batch_پایه + (dst_m_جابه_جایی + ir1) * dst_ردیف_اندازه);
                    mmctx->vec_dot_1x1(ne00, &dst_ردیف[ir0], ss0, act_ستون);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_شروع = b_grp_انتها;
    }

    إذا (src2) {
        hvx_tensor_جمع_کن_f32_grid(dst, src2, cur_m_شروع, cur_m_شروع + cur_m_ردیفها, src0_شروع_ردیف, src0_انتها_ردیف, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#تعریف_کن MMID_ماتریس_ردیف(ردیف_id, i1) ماتریس_ردیفها[(ردیف_id) * mmctx->mapping_stride + (i1)]

ساكن فراغ hvx_mm_id_tiled(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_matmul_preamble;

    ثابت بنية htp_tensor * محدود ids = octx->src[2];

    ثابت uint32_t src0_nrows      = mmctx->src0_ردیف_انتها - mmctx->src0_ردیف_شروع;  // src0 rows per expert
    ثابت uint32_t act_nrows       = ne11;
    ثابت uint32_t src0_شروع_ردیف  = mmctx->src0_ردیف_شروع + src0_nrows_در_هر_thread * ith;
    ثابت uint32_t src0_انتها_ردیف    = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, mmctx->src0_ردیف_انتها);


    إذا (src0_شروع_ردیف >= src0_انتها_ردیف) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_ids = ids->ne[0];  // n_expert_used
    ثابت uint32_t n_as  = ne02;        // n_expert

    ثابت uint32_t *                ماتریس_ردیف_counts = mmctx->ماتریس_ردیف_counts;
    ثابت بنية mmid_ردیف_mapping * ماتریس_ردیفها       = mmctx->ماتریس_ردیفها;

    ثابت size_t dst_ردیف_اندازه  = nb1;
    ثابت size_t act_ردیف_اندازه  = htp_mm_q8_0_tiled_ردیف_اندازه(ne10);

    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * محدود act_داده = mmctx->vtcm_act;

    لكل (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int32_t cne1 = ماتریس_ردیف_counts[cur_a];
        إذا (cne1 == 0) {
            استمر;
        }

        ثابت dma_addr_t src0_ردیف = src0->داده + cur_a * nb02;

        ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src0->type);
        ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src0->type);
        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ثابت uint32_t tile_ردیف_stride = n_k_tiles_w * tile_اندازه;
        ثابت uint32_t tile_ردیف_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

        ثابت uint32_t ct_شروع = src0_شروع_ردیف / 32;
        ثابت uint32_t ct_انتها   = (src0_انتها_ردیف + 31) / 32;

        uint32_t افزودن_ct = ct_شروع;
        لكل (uint32_t d = 0; d < n_prefetch && افزودن_ct < ct_انتها; d++, افزودن_ct++) {
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + d * tile_ردیف_transfer_اندازه_aligned, src0_ردیف + افزودن_ct * tile_ردیف_stride),
                           aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_شروع; ct < ct_انتها; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_صف_برداشت(dma_q).dst;

            صحيح valid_ردیفها = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_ردیفها = MIN(32, MAX(0, valid_ردیفها));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            لكل (uint32_t cid = 0; cid < cne1; ++cid) {
                بنية mmid_ردیف_mapping ردیف_mapping = MMID_ماتریس_ردیف(cur_a, cid);
                ثابت صحيح               rm1         = ردیف_mapping.i1;  // expert idx
                ثابت صحيح               rm2         = ردیف_mapping.i2;  // token idx

                ثابت uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_داده + (ir1 + rm2 * ne11 + 0) * act_stride);
                عائم * محدود dst_ردیف = (عائم *) (dst->داده + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_ردیف[ct * 32], w_tile, act_ستون, valid_ردیفها, عدم);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (افزودن_ct < ct_انتها) {
                dma_صف_افزودن(dma_q, dma_make_داده(w_tile, src0_ردیف + افزودن_ct * tile_ردیف_stride),
                               aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                افزودن_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mv_id_tiled(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_matmul_preamble;

    ثابت بنية htp_tensor * محدود ids = octx->src[2];

    ثابت uint32_t src0_nrows      = mmctx->src0_ردیف_انتها - mmctx->src0_ردیف_شروع;  // src0 rows per expert
    ثابت uint32_t src0_شروع_ردیف  = mmctx->src0_ردیف_شروع + src0_nrows_در_هر_thread * ith;
    ثابت uint32_t src0_انتها_ردیف    = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, mmctx->src0_ردیف_انتها);


    إذا (src0_شروع_ردیف >= src0_انتها_ردیف) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    أكد(ne13 % ne03 == 0);

    ثابت size_t dst_ردیف_اندازه  = nb1;
    ثابت size_t act_ردیف_اندازه = htp_mm_q8_0_tiled_ردیف_اندازه(ne10);

    ثابت uint32_t n_aids = src2->ne[0];  // num activated experts
    ثابت uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * محدود act_داده = mmctx->vtcm_act;

    لكل (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ثابت int32_t eid = *(ثابت int32_t *) ((ثابت uint8_t *) src2->داده + ie1 * src2->nb[0]);
        إذا (eid < 0) {
            استمر;
        }
        أكد(eid < (int32_t) n_ids);

        ثابت dma_addr_t src0_ردیف = src0->داده + eid * nb02;
        ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_داده;
        عائم * محدود dst_ردیف          = (عائم *) (dst->داده + ie1 * nb1);

        ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src0->type);
        ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src0->type);
        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ثابت uint32_t tile_ردیف_stride = n_k_tiles_w * tile_اندازه;
        ثابت uint32_t tile_ردیف_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

        ثابت uint32_t ct_شروع = src0_شروع_ردیف / 32;
        ثابت uint32_t ct_انتها   = (src0_انتها_ردیف + 31) / 32;

        uint32_t افزودن_ct = ct_شروع;
        لكل (uint32_t d = 0; d < n_prefetch && افزودن_ct < ct_انتها; d++, افزودن_ct++) {
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + d * tile_ردیف_transfer_اندازه_aligned, src0_ردیف + افزودن_ct * tile_ردیف_stride),
                           aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_شروع; ct < ct_انتها; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_صف_برداشت(dma_q).dst;

            صحيح valid_ردیفها = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_ردیفها = MIN(32, MAX(0, valid_ردیفها));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_ردیف[ct * 32], w_tile, act_ستون, valid_ردیفها, عدم);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            إذا (افزودن_ct < ct_انتها) {
                dma_صف_افزودن(dma_q, dma_make_داده(w_tile, src0_ردیف + افزودن_ct * tile_ردیف_stride),
                               aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                افزودن_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mv_id_nx_tiled(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_mm_context * mmctx = (بنية htp_mm_context *) داده;
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

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * محدود act_داده = mmctx->vtcm_act;

    لكل (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ثابت int32_t eid = *(ثابت int32_t *) ((ثابت uint8_t *) ids->داده + ie1 * ids->nb[0]);
        إذا (eid < 0) استمر;
        أكد(eid < (int32_t) n_ids);

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * محدود src_w = octx->src[p];
            ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
            إذا (!src_w || !dst) استمر;
            dma_صف * dma_q = octx->ctx->dma[ith];

            ثابت uint32_t ne01 = src_w->ne[1];
            uint32_t شروع_ردیف = 0;
            uint32_t انتها_ردیف   = ne01;
            إذا (octx->ctx->mdev.شمار > 1) {
                ثابت bool can_تقسیم = htp_tensor_can_ردیف_افراز(dst, حجم(عائم));
                ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(ne01, can_تقسیم ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
                شروع_ردیف = range.شروع;
                انتها_ردیف   = range.شروع + range.شمار;
            }

            ثابت uint32_t nrows = انتها_ردیف - شروع_ردیف;
            uint32_t src0_nrows_در_هر_thread = fastdiv(nrows + nth - 1, &octx->n_نخها_div);
            src0_nrows_در_هر_thread = hex_دور_به_بالا(src0_nrows_در_هر_thread, 32);

            ثابت uint32_t src0_شروع_ردیف = شروع_ردیف + src0_nrows_در_هر_thread * ith;
            ثابت uint32_t src0_انتها_ردیف   = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, انتها_ردیف);
            إذا (src0_شروع_ردیف >= src0_انتها_ردیف) استمر;

            ثابت dma_addr_t src0_ردیف = src_w->داده + eid * src_w->nb[2];
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_داده;
            عائم * محدود dst_ردیف = (عائم *) (dst->داده + ie1 * dst->nb[1]);

            ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src_w->type);
            ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src_w->type);
            ثابت uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ثابت uint32_t tile_ردیف_stride = n_k_tiles_w * tile_اندازه;
            ثابت uint32_t tile_ردیف_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

            ثابت uint32_t ct_شروع = src0_شروع_ردیف / 32;
            ثابت uint32_t ct_انتها   = (src0_انتها_ردیف + 31) / 32;

            uint32_t افزودن_ct = ct_شروع;
            لكل (uint32_t d = 0; d < n_prefetch && افزودن_ct < ct_انتها; d++, افزودن_ct++) {
                dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + d * tile_ردیف_transfer_اندازه_aligned, src0_ردیف + افزودن_ct * tile_ردیف_stride),
                               aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
            }

            لكل (uint32_t ct = ct_شروع; ct < ct_انتها; ct++) {
                ثابت uint8_t * w_tile = (فراغ *) dma_صف_برداشت(dma_q).dst;

                صحيح valid_ردیفها = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
                valid_ردیفها = MIN(32, MAX(0, valid_ردیفها));

                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_ردیف[ct * 32], w_tile, act_ستون, valid_ردیفها, عدم);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                إذا (افزودن_ct < ct_انتها) {
                    dma_صف_افزودن(dma_q, dma_make_داده(w_tile, src0_ردیف + افزودن_ct * tile_ردیف_stride),
                                    aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                    افزودن_ct++;
                }
            }
        }
    }
}

ساكن فراغ hvx_mm_id_nx_tiled(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_mm_context * mmctx = (بنية htp_mm_context *) داده;
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

    ثابت uint32_t * ماتریس_ردیف_counts = mmctx->ماتریس_ردیف_counts;
    ثابت بنية mmid_ردیف_mapping * ماتریس_ردیفها = mmctx->ماتریس_ردیفها;

    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_اندازه_در_هر_thread * ith;
    uint8_t * محدود act_داده = mmctx->vtcm_act;

    لكل (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int32_t cne1 = ماتریس_ردیف_counts[cur_a];
        إذا (cne1 == 0) استمر;

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * محدود src_w = octx->src[p];
            ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
            إذا (!src_w || !dst) استمر;
            dma_صف * dma_q = octx->ctx->dma[ith];

            ثابت uint32_t ne01 = src_w->ne[1];
            uint32_t شروع_ردیف = 0;
            uint32_t انتها_ردیف   = ne01;
            إذا (octx->ctx->mdev.شمار > 1) {
                ثابت bool can_تقسیم = htp_tensor_can_ردیف_افراز(dst, حجم(عائم));
                ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(ne01, can_تقسیم ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
                شروع_ردیف = range.شروع;
                انتها_ردیف   = range.شروع + range.شمار;
            }

            ثابت uint32_t nrows = انتها_ردیف - شروع_ردیف;
            uint32_t src0_nrows_در_هر_thread = fastdiv(nrows + nth - 1, &octx->n_نخها_div);
            src0_nrows_در_هر_thread = hex_دور_به_بالا(src0_nrows_در_هر_thread, 32);

            ثابت uint32_t src0_شروع_ردیف = شروع_ردیف + src0_nrows_در_هر_thread * ith;
            ثابت uint32_t src0_انتها_ردیف   = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, انتها_ردیف);
            إذا (src0_شروع_ردیف >= src0_انتها_ردیف) استمر;

            ثابت dma_addr_t src0_ردیف = src_w->داده + cur_a * src_w->nb[2];

            ثابت uint32_t tile_اندازه = htp_tensor_get_tile_اندازه(src_w->type);
            ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(src_w->type);
            ثابت uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ثابت uint32_t tile_ردیف_stride = n_k_tiles_w * tile_اندازه;
            ثابت uint32_t tile_ردیف_transfer_اندازه_aligned = n_k_tiles_w * aligned_tile_اندازه;

            ثابت uint32_t ct_شروع = src0_شروع_ردیف / 32;
            ثابت uint32_t ct_انتها   = (src0_انتها_ردیف + 31) / 32;

            uint32_t افزودن_ct = ct_شروع;
            لكل (uint32_t d = 0; d < n_prefetch && افزودن_ct < ct_انتها; d++, افزودن_ct++) {
                dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src0_ptr + d * tile_ردیف_transfer_اندازه_aligned, src0_ردیف + افزودن_ct * tile_ردیف_stride),
                               aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
            }

            لكل (uint32_t ct = ct_شروع; ct < ct_انتها; ct++) {
                ثابت uint8_t * w_tile = (فراغ *) dma_صف_برداشت(dma_q).dst;

                صحيح valid_ردیفها = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
                valid_ردیفها = MIN(32, MAX(0, valid_ردیفها));

                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                لكل (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    بنية mmid_ردیف_mapping ردیف_mapping = MMID_ماتریس_ردیف(cur_a, cid);
                    ثابت صحيح rm1 = ردیف_mapping.i1;
                    ثابت صحيح rm2 = ردیف_mapping.i2;

                    ثابت uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_داده + (ir1 + rm2 * act->ne[1]) * act_stride);
                    عائم * محدود dst_ردیف = (عائم *) (dst->داده + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_ردیف[ct * 32], w_tile, act_ستون, valid_ردیفها, عدم);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                إذا (افزودن_ct < ct_انتها) {
                    dma_صف_افزودن(dma_q, dma_make_داده(w_tile, src0_ردیف + افزودن_ct * tile_ردیف_stride),
                                    aligned_tile_اندازه, tile_اندازه, tile_اندازه, n_k_tiles_w);
                    افزودن_ct++;
                }
            }
        }
    }
}

ساكن صحيح hvx_mm_init_tiled_vec_dot(بنية htp_mm_context * mmctx, تعداد htp_داده_type type) {
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

    uint32_t src0_ردیف_شروع = 0;
    uint32_t src0_ردیف_انتها   = src0_nrows;

    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت bool can_تقسیم = htp_tensor_can_ردیف_افراز(dst, حجم(عائم));
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(src0_nrows, can_تقسیم ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        src0_ردیف_شروع = range.شروع;
        src0_ردیف_انتها   = range.شروع + range.شمار;
    }

    إذا (src0_ردیف_شروع >= src0_ردیف_انتها) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t nrows = src0_ردیف_انتها - src0_ردیف_شروع;
    mmctx->src0_ردیف_شروع = src0_ردیف_شروع;
    mmctx->src0_ردیف_انتها   = src0_ردیف_انتها;

    ثابت bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_در_هر_thread  = fastdiv(nrows + octx->n_نخها - 1, &octx->n_نخها_div);
    إذا (is_tiled) {
        mmctx->src0_nrows_در_هر_thread = hex_دور_به_بالا(mmctx->src0_nrows_در_هر_thread, 32);
    } وإلا {
        mmctx->src0_nrows_در_هر_thread += (mmctx->src0_nrows_در_هر_thread & 1); // round up to even
    }

    ثابت size_t src0_ردیف_اندازه = nb01;
    ثابت size_t dst_ردیف_اندازه  = nb1;
    size_t       act_ردیف_اندازه  = nb11;

    ثابت size_t src0_ردیف_اندازه_padded = hex_دور_به_بالا(src0_ردیف_اندازه, 128);

    کارگر_callback_t quant_task_func;
    کارگر_callback_t matmul_کار_func;
    uint32_t n_quant_tasks = 1;
    ثابت bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    إذا (is_batched) {
        matmul_کار_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } وإلا إذا (act_nrows > 1) {
        matmul_کار_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } وإلا {
        matmul_کار_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
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
            act_ردیف_اندازه           = kparams->act_ردیف_اندازه;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = عدم;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_ردیف_اندازه           = kparams->act_ردیف_اندازه;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = عدم;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_ردیف_اندازه           = kparams->act_ردیف_اندازه;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_QUANT_بلوک:
        حالة HTP_MM_KERNEL_HVX_QUANT_ردیف:
        افتراضي: {
            إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                أرجع HTP_وضعیت_NO_SUPPORT;
            }

            ثابت uint32_t qk = QK_Q8_0_TILED;
            ثابت uint32_t nb = (ne10 + qk - 1) / qk;
            ثابت uint32_t مجموع_nb = act_nrows * nb;

            إذا (act_nrows < octx->n_نخها && !is_batched) {
                n_quant_tasks = MIN(مجموع_nb, octx->n_نخها);
                quant_task_func = htp_mm_act_quant_بلوک_func(src0->type, act->type);
                لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_اولی = (مجموع_nb * ith) / n_quant_tasks;
                    uint32_t ib_آخرین  = (مجموع_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_اولی[ith] = ib_اولی;
                    mmctx->quant_ib_آخرین[ith]  = ib_آخرین;
                    mmctx->quant_r[ith]        = ib_اولی / nb;
                    mmctx->quant_c[ith]        = ib_اولی % nb;
                }
            } وإلا {
                n_quant_tasks = MIN(act_nrows, octx->n_نخها);
                quant_task_func = htp_mm_act_quant_ردیف_func(src0->type, act->type);
            }
            act_ردیف_اندازه = kparams->act_ردیف_اندازه;
            اكسر;
        }
    }

    ثابت uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ثابت uint32_t m_layout_ردیفها = m_chunk;

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_ردیفها, octx->n_نخها,
                                 dst_ردیف_اندازه, src0_ردیف_اندازه, act_ردیف_اندازه, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    إذا (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ردیف ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلوک) {
        mmctx->vtcm_act_اندازه_در_هر_thread = L.act_بایتها;
    } وإلا {
        mmctx->vtcm_act_اندازه_در_هر_thread = fastdiv(L.act_بایتها, &octx->n_نخها_div);
    }

    mmctx->vtcm_src0_اندازه_در_هر_thread = fastdiv(L.src0_بایتها, &octx->n_نخها_div);
    mmctx->vtcm_dst_اندازه_در_هر_thread  = fastdiv(L.dst_بایتها, &octx->n_نخها_div);

    ثابت size_t vtcm_اندازه = L.مجموع_بایتها;

    FARF(بالا_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_بایتها, L.act_بایتها, L.dst_بایتها, vtcm_اندازه);

    FARF(بالا_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->داده, act->داده, dst->داده);

    إذا (octx->ctx->vtcm_اندازه < vtcm_اندازه) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_اندازه, vtcm_اندازه);
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت پایه = (uint8_t *) octx->ctx->vtcm_پایه;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_ردیف_اندازه_padded;
    mmctx->vtcm_act_stride  = act_ردیف_اندازه;
    إذا (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلوک || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ردیف) {
        mmctx->vtcm_act_raw_stride = hex_دور_به_بالا(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
    } وإلا إذا (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_دور_به_بالا(ne10 * حجم(عائم), 128);
    } وإلا {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    إذا (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        لكل (uint32_t m_شروع = 0; m_شروع < act_nrows; m_شروع += m_chunk) {
            ثابت uint32_t cur_m_ردیفها = MIN(act_nrows - m_شروع, m_chunk);
            mmctx->cur_m_شروع = m_شروع;
            mmctx->cur_m_ردیفها  = cur_m_ردیفها;

            إذا (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_شروع, cur_m_ردیفها);

                ثابت uint32_t qk = QK_Q8_0_TILED;
                ثابت uint32_t nb = (ne10 + qk - 1) / qk;
                ثابت uint32_t مجموع_nb = cur_m_ردیفها * nb;
                uint32_t quant_tasks;
                work_صف_func_t q_func;
                إذا (cur_m_ردیفها < octx->n_نخها && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلوک || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ردیف)) {
                    quant_tasks = MIN(مجموع_nb, octx->n_نخها);
                    q_func = htp_mm_act_quant_بلوک_func(src0->type, act->type);
                    لكل (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_اولی = (مجموع_nb * ith) / quant_tasks;
                        uint32_t ib_آخرین  = (مجموع_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_اولی[ith] = ib_اولی;
                        mmctx->quant_ib_آخرین[ith]  = ib_آخرین;
                        mmctx->quant_r[ith]        = ib_اولی / nb;
                        mmctx->quant_c[ith]        = ib_اولی % nb;
                    }
                } وإلا {
                    quant_tasks = MIN(cur_m_ردیفها, octx->n_نخها);
                    q_func = quant_task_func;
                    mmctx->n_quant_ردیفها_در_هر_thread = (cur_m_ردیفها + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_صف_پیاپی(octx->ctx->work_صف, q_func, mmctx, quant_tasks);
            } وإلا {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_شروع, cur_m_ردیفها);
            }

            work_صف_پیاپی(octx->ctx->work_صف, matmul_کار_func, mmctx, octx->n_نخها);
        }
    } وإلا {
        mmctx->cur_m_شروع = 0;
        mmctx->cur_m_ردیفها  = act_nrows;

        إذا (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_ردیفها_در_هر_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_صف_پیاپی(octx->ctx->work_صف, quant_task_func, mmctx, n_quant_tasks);
        } وإلا {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_صف_پیاپی(octx->ctx->work_صف, matmul_کار_func, mmctx, octx->n_نخها);
    }

    أرجع HTP_وضعیت_OK;
}

ساكن فراغ dequantize_tiled_کارگر_loop(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) داده;
    بنية htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    لكل (بیعلامت صحيح task_id = i; task_id < (بیعلامت صحيح) state->n_tasks; task_id += n) {
        صحيح شروع = task_id * state->n_tiles_در_هر_task;
        صحيح انتها   = hex_smin(شروع + state->n_tiles_در_هر_task, state->n_tot_tiles);
        state->task_fn(state, شروع, انتها);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

ساكن فراغ transfer_output_chunk_کارگر_fn(بیعلامت صحيح n, بیعلامت صحيح i, فراغ *داده) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) داده;

    بنية htp_thread_trace * tr = &st->traces[i];

    صحيح شروع_chunk_idx = i * st->n_chunks_در_هر_task;
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_O_PROC, شروع_chunk_idx);

    لكل (بیعلامت صحيح task_id = i; task_id < (بیعلامت صحيح)st->n_tasks; task_id += n) {
        صحيح    chunk_idx  = task_id * st->n_chunks_در_هر_task;
        size_t chunk_اندازه = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_در_هر_task);

        عائم        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ثابت عائم  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : عدم;
        transfer_output_chunk_fp16_مقصد_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_اندازه, st->n_تعداد_ستون, st->dst_stride, st->src2_stride, st->dst_تعداد_ستون);
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
    uint32_t                        n_chunks_در_هر_task;
    uint32_t                        k_بلوک;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_بایتها_در_هر_thread;
    uint32_t                        dma_گام_ردیفها;
    uint32_t                        dma_گام_ردیفها_لغزش;
    uint32_t                        act_elem_اندازه;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

تعريفنوع بنية {
    بنية htp_context            * ctx;
    بنية htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    عائم                         * vtcm_f32_act;
    uint32_t                        n_ردیفها;
    uint32_t                        k_بلوک;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_ستون_chunks;
    بنية fastdiv_مقادیر           n_نخها_div;
    size_t                          vtcm_f32_act_بایتها;
    uint32_t                        dma_گام_ردیفها;
    uint32_t                        dma_گام_ردیفها_لغزش;
} activation_transfer_ستون_chunk_state_t;

ساكن فراغ transfer_activation_chunk_fp32_مقصد_fp16_dma_pipelined_ستون_chunk(
        dma_صف *dma_q,
        __fp16 *محدود vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_ردیفها,
        uint32_t k_بلوک,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_اولی,
        uint32_t c_len,
        عائم *thread_f32_act,
        بنية htp_thread_trace *tr,
        uint32_t dma_گام_ردیفها,
        uint32_t dma_گام_ردیفها_لغزش) {

    ثابت uint32_t R = dma_گام_ردیفها;
    ثابت uint32_t n_ردیفها_padded = hex_align_به_بالا(n_ردیفها, HTP_MM_HMX_TILE_N_ردیفها);

    ثابت uint32_t n_گامها = n_ردیفها_padded >> dma_گام_ردیفها_لغزش;

    // Push step 0
    إذا (n_گامها > 0 && n_ردیفها > 0) {
        uint32_t nrows_مقصد_fetch = hex_smin(n_ردیفها, R);
        dma_صف_افزودن(dma_q, dma_make_داده(thread_f32_act, act_dma_addr + (size_t) c_اولی * حجم(عائم)),
                       c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_مقصد_fetch);
    }
    // Push step 1
    إذا (n_گامها > 1) {
        uint32_t بعدی_r = R * 1;
        إذا (بعدی_r < n_ردیفها) {
            uint32_t nrows_مقصد_fetch = hex_smin(n_ردیفها - بعدی_r, R);
            عائم *بعدی_buf = thread_f32_act + 1 * R * c_len;
            dma_صف_افزودن(dma_q, dma_make_داده(بعدی_buf, act_dma_addr + ((size_t) بعدی_r * k_stride + c_اولی) * حجم(عائم)),
                           c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_مقصد_fetch);
        }
    }
    لكل (uint32_t s = 0; s < n_گامها; ++s) {
        uint32_t r = s << dma_گام_ردیفها_لغزش;
        عائم *curr_buf = thread_f32_act;

        إذا (r < n_ردیفها) {
            curr_buf = (عائم *) dma_صف_برداشت(dma_q).dst;
        }

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        لكل (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t ردیف_idx = r + (p << 1);
            عائم *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((ردیف_idx + 0) < n_ردیفها);
            bool r1_valid = ((ردیف_idx + 1) < n_ردیفها);

            transfer_activation_ردیف_pair_fp32_مقصد_fp16_ستون_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, ردیف_idx, k_بلوک, c_اولی, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t بعدی_s = s + 2;
        uint32_t بعدی_r = بعدی_s << dma_گام_ردیفها_لغزش;
        إذا (بعدی_r < n_ردیفها) {
            uint32_t nrows_مقصد_fetch = hex_smin(n_ردیفها - بعدی_r, R);
            dma_صف_افزودن(dma_q, dma_make_داده(curr_buf, act_dma_addr + ((size_t) بعدی_r * k_stride + c_اولی) * حجم(عائم)),
                           c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_مقصد_fetch);
        }
    }
}

ساكن فراغ transfer_activation_chunk_ستون_chunk_کارگر_fn(بیعلامت صحيح n, بیعلامت صحيح i, فراغ *داده) {
    activation_transfer_ستون_chunk_state_t *st = (activation_transfer_ستون_chunk_state_t *) داده;
    بنية htp_thread_trace * tr = &st->traces[i];

    uint32_t n_بلوکها = st->k_بلوک / 32;
    uint32_t b_اولی = fastdiv(n_بلوکها * i, &st->n_نخها_div);
    uint32_t b_آخرین  = fastdiv(n_بلوکها * (i + 1), &st->n_نخها_div);
    uint32_t c_اولی = b_اولی * 32;
    uint32_t c_آخرین = b_آخرین * 32;
    uint32_t c_len = c_آخرین - c_اولی;

    إذا (c_len == 0) {
        أرجع;
    }

    uint32_t k_chunk_valid = 0;
    إذا (st->k_valid > c_اولی) {
        k_chunk_valid = hex_smin(st->k_valid, c_آخرین) - c_اولی;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_بایتها = hex_align_پایین(fastdiv(st->vtcm_f32_act_بایتها, &st->n_نخها_div), 128);
    عائم *thread_f32_act = (عائم *)((حرف *)st->vtcm_f32_act + i * thread_scratch_بایتها);

    transfer_activation_chunk_fp32_مقصد_fp16_dma_pipelined_ستون_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_ردیفها, st->k_بلوک, st->k_stride, k_chunk_valid,
        c_اولی, c_len, thread_f32_act, tr, st->dma_گام_ردیفها, st->dma_گام_ردیفها_لغزش
    );
}

ساكن فراغ transfer_activation_chunk_مقصد_fp16_dma_pipelined(
        dma_صف *dma_q,
        __fp16 *محدود vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_ردیفها,
        uint32_t k_بلوک,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_اندازه,
        بنية htp_thread_trace *tr,
        uint32_t dma_گام_ردیفها,
        uint32_t dma_گام_ردیفها_لغزش) {

    ثابت uint32_t R = dma_گام_ردیفها;
    ثابت uint32_t n_ردیفها_padded = hex_align_به_بالا(n_ردیفها, HTP_MM_HMX_TILE_N_ردیفها);

    ثابت uint32_t n_گامها = n_ردیفها_padded >> dma_گام_ردیفها_لغزش;

    ثابت bool act_is_f16 = (act_elem_اندازه == حجم(__fp16));
    ثابت size_t ردیف_بایتها = (size_t) k_بلوک * act_elem_اندازه;  // staging row (DMA dst stride)
    ثابت size_t src_ردیف_بایتها = (size_t) k_stride * act_elem_اندازه;  // DDR row (DMA src stride)
    ثابت size_t عرض_بایتها = (size_t) k_valid * act_elem_اندازه;

    // Push step 0
    إذا (n_گامها > 0 && n_ردیفها > 0) {
        uint32_t nrows_مقصد_fetch = hex_smin(n_ردیفها, R);
        dma_صف_افزودن(dma_q, dma_make_داده(thread_act, act_dma_addr),
                       ردیف_بایتها, src_ردیف_بایتها, عرض_بایتها, nrows_مقصد_fetch);
    }
    // Push step 1 (if valid)
    إذا (n_گامها > 1) {
        uint32_t بعدی_r = R * 1;
        إذا (بعدی_r < n_ردیفها) {
            uint32_t nrows_مقصد_fetch = hex_smin(n_ردیفها - بعدی_r, R);
            uint8_t *بعدی_buf = thread_act + 1 * R * ردیف_بایتها;
            dma_صف_افزودن(dma_q, dma_make_داده(بعدی_buf, act_dma_addr + (size_t) بعدی_r * src_ردیف_بایتها),
                           ردیف_بایتها, src_ردیف_بایتها, عرض_بایتها, nrows_مقصد_fetch);
        }
    }
    لكل (uint32_t s = 0; s < n_گامها; ++s) {
        uint32_t r = s << dma_گام_ردیفها_لغزش;
        uint8_t *curr_buf = thread_act;

        إذا (r < n_ردیفها) {
            curr_buf = (uint8_t *) dma_صف_برداشت(dma_q).dst;
        }

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        إذا (act_is_f16) {
            لكل (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t ردیف_idx = r + (p << 1);
                ثابت __fp16 *pair_buf = (ثابت __fp16 *) (curr_buf + (p << 1) * ردیف_بایتها);
                bool r0_valid = ((ردیف_idx + 0) < n_ردیفها);
                bool r1_valid = ((ردیف_idx + 1) < n_ردیفها);

                transfer_activation_ردیف_pair_f16_مقصد_f16(vtcm_dst, pair_buf, pair_buf + k_بلوک, ردیف_idx, k_بلوک, k_valid, r0_valid, r1_valid);
            }
        } وإلا {
            لكل (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t ردیف_idx = r + (p << 1);
                ثابت عائم *pair_buf = (ثابت عائم *) (curr_buf + (p << 1) * ردیف_بایتها);
                bool r0_valid = ((ردیف_idx + 0) < n_ردیفها);
                bool r1_valid = ((ردیف_idx + 1) < n_ردیفها);

                transfer_activation_ردیف_pair_fp32_مقصد_fp16(vtcm_dst, pair_buf, pair_buf + k_بلوک, ردیف_idx, k_بلوک, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t بعدی_s = s + 2;
        uint32_t بعدی_r = بعدی_s << dma_گام_ردیفها_لغزش;
        إذا (بعدی_r < n_ردیفها) {
            uint32_t nrows_مقصد_fetch = hex_smin(n_ردیفها - بعدی_r, R);
            dma_صف_افزودن(dma_q, dma_make_داده(curr_buf, act_dma_addr + (size_t) بعدی_r * src_ردیف_بایتها),
                           ردیف_بایتها, src_ردیف_بایتها, عرض_بایتها, nrows_مقصد_fetch);
        }
    }
}

ساكن فراغ transfer_activation_chunk_کارگر_fn(بیعلامت صحيح n, بیعلامت صحيح i, فراغ *داده) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) داده;

    بنية htp_thread_trace * tr = &st->traces[i];

    لكل (بیعلامت صحيح task_id = i; task_id < (بیعلامت صحيح)st->n_tasks; task_id += n) {
        صحيح    chunk_idx  = task_id * st->n_chunks_در_هر_task;
        size_t chunk_اندازه = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_در_هر_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_بلوک;
        ثابت dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_اندازه;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_بایتها_در_هر_thread;
        transfer_activation_chunk_مقصد_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_اندازه, st->k_بلوک, st->k_stride, st->k_valid, thread_act, st->act_elem_اندازه, tr, st->dma_گام_ردیفها, st->dma_گام_ردیفها_لغزش
        );
    }
}

تعريفنوع بنية {
    بنية htp_thread_trace       * traces;
    ثابت بنية mmid_ردیف_mapping * ماتریس_ردیفها;
    __fp16                        * dst;
    ثابت عائم                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_در_هر_task;
    uint32_t                        k_بلوک;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    بنية fastdiv_مقادیر           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        شروع_ردیف;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

تعريفنوع بنية {
    بنية htp_thread_trace       * traces;
    ثابت بنية mmid_ردیف_mapping * ماتریس_ردیفها;
    ثابت __fp16                  * vtcm_src;
    عائم                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_در_هر_task;
    uint32_t                        n_تعداد_ستون;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        شروع_ردیف;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

ساكن فراغ transfer_activation_chunk_gathered_کارگر_fn(بیعلامت صحيح n, بیعلامت صحيح i, فراغ *داده) {
    activation_transfer_gathered_task_state_t *st = داده;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx      = i;
    صحيح chunk_اندازه     = st->n_chunks_در_هر_task;
    صحيح vtcm_شروع_ردیف = chunk_idx * chunk_اندازه;
    صحيح شروع_ردیف      = st->شروع_ردیف + vtcm_شروع_ردیف;
    صحيح n_ردیفها         = hex_smin(st->cne1 - شروع_ردیف, chunk_اندازه);
    إذا (n_ردیفها > 0) {
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_مقصد_fp16_gathered(
            st->dst, st->src, شروع_ردیف, vtcm_شروع_ردیف, n_ردیفها, st->k_بلوک,
            st->ماتریس_ردیفها, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ساكن فراغ transfer_activation_chunk_gathered_کارگر_flat_fn(بیعلامت صحيح n, بیعلامت صحيح i, فراغ *داده) {
    activation_transfer_gathered_task_state_t *st = داده;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx = i;
    صحيح chunk_اندازه = st->n_chunks_در_هر_task;
    صحيح vtcm_شروع_ردیف = chunk_idx * chunk_اندازه;
    صحيح شروع_ردیف = st->شروع_ردیف + vtcm_شروع_ردیف;
    صحيح n_ردیفها = hex_smin(st->cne1 - شروع_ردیف, chunk_اندازه);
    إذا (n_ردیفها > 0) {
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_مقصد_fp16_gathered_flat(
            st->dst, st->src, شروع_ردیف, vtcm_شروع_ردیف, n_ردیفها, st->k_بلوک,
            st->ماتریس_ردیفها, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ساكن فراغ transfer_output_chunk_scattered_کارگر_fn(بیعلامت صحيح n, بیعلامت صحيح i, فراغ *داده) {
    output_transfer_scattered_task_state_t *st = داده;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx = i;
    صحيح chunk_اندازه = st->n_chunks_در_هر_task;
    صحيح vtcm_شروع_ردیف = chunk_idx * chunk_اندازه;
    صحيح شروع_ردیف = st->شروع_ردیف + vtcm_شروع_ردیف;
    صحيح n_ردیفها = hex_smin(st->cne1 - شروع_ردیف, chunk_اندازه);
    إذا (n_ردیفها > 0) {
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_مقصد_fp32_scattered(
            st->dst, st->vtcm_src, شروع_ردیف, vtcm_شروع_ردیف, n_ردیفها, st->n_تعداد_ستون,
            st->ماتریس_ردیفها, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

ساكن مضمن tiled_dequant_task_fn_t hmx_get_dequant_کارگر_fn(صحيح وزن_type, bool is_tiled) {
    بدل (وزن_type) {
        حالة HTP_TYPE_Q4_0:   أرجع dequantize_tiled_وزن_مقصد_fp16_task_q4_0;
        حالة HTP_TYPE_IQ4_NL: أرجع dequantize_tiled_وزن_مقصد_fp16_task_iq4_nl;
        حالة HTP_TYPE_Q4_1:
        حالة HTP_TYPE_Q4_K:   أرجع dequantize_tiled_وزن_مقصد_fp16_task_q4_1;
        حالة HTP_TYPE_MXFP4:  أرجع dequantize_tiled_وزن_مقصد_fp16_task_mxfp4;
        حالة HTP_TYPE_Q8_0:   أرجع dequantize_tiled_وزن_مقصد_fp16_task_q8_0;
        حالة HTP_TYPE_Q5_K:   أرجع dequantize_tiled_وزن_مقصد_fp16_task_q5_k;
        حالة HTP_TYPE_Q6_K:   أرجع dequantize_tiled_وزن_مقصد_fp16_task_q6_k;
        حالة HTP_TYPE_Q3_K:   أرجع dequantize_tiled_وزن_مقصد_fp16_task_q3_k;
        حالة HTP_TYPE_Q2_K:   أرجع dequantize_tiled_وزن_مقصد_fp16_task_q2_k;
        حالة HTP_TYPE_Q1_0:   أرجع dequantize_tiled_وزن_مقصد_fp16_task_q1_0;
        حالة HTP_TYPE_Q2_0:   أرجع dequantize_tiled_وزن_مقصد_fp16_task_q2_0;
        حالة HTP_TYPE_F16:    أرجع is_tiled ? عدم : convert_f16_وزن_مقصد_fp16_tiles_task;
        حالة HTP_TYPE_BF16:   أرجع convert_bf16_وزن_مقصد_fp16_tiles_task;
        حالة HTP_TYPE_F32:    أرجع quantize_f32_وزن_مقصد_fp16_tiles_task;
        افتراضي:              أرجع عدم;
    }
}

ساكن فراغ dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
        بنية htp_context *ctx, __fp16 *vtcm_dst,
        ثابت فراغ *وزن_src_ddr,
        صحيح n_تعداد_ستون, صحيح k_بلوک,
        size_t ردیف_stride, صحيح وزن_type,
        صحيح n_k_tiles, بنية fastdiv_مقادیر n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_کارگر_fn, صحيح n_نخها) {

    إذا (!dequant_کارگر_fn) {
        أرجع;
    }

    أكد(k_بلوک % HTP_MM_HMX_TILE_N_تعداد_ستون == 0);

    ثابت size_t n_ستون_tiles = hmx_ceil_div(n_تعداد_ستون, HTP_MM_HMX_TILE_N_تعداد_ستون);
    ثابت uint32_t n_وزن_tiles = k_بلوک / htp_tensor_get_tile_k(وزن_type);
    ثابت size_t n_tot_tiles = n_ستون_tiles * n_وزن_tiles;

    size_t n_tiles_در_هر_task = (n_نخها == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_نخها);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_در_هر_task - 1) / n_tiles_در_هر_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_در_هر_task = n_tiles_در_هر_task;
    state.dst              = vtcm_dst;
    state.src              = (ثابت uint8_t *)وزن_src_ddr;
    state.task_fn          = dequant_کارگر_fn;
    state.n_تعداد_ستون           = n_تعداد_ستون;
    state.k_بلوک          = k_بلوک;
    state.ردیف_stride       = ردیف_stride;
    state.وزن_type      = وزن_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_وزن_tiles   = n_وزن_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_اندازه = htp_tensor_get_tile_اندازه(وزن_type);
    state.aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(وزن_type);

    إذا (state.n_tasks == 1 || n_نخها == 1) {
        dequantize_tiled_کارگر_loop(1, 0, &state);
    } وإلا {
        صحيح n_tasks = hex_smin((صحيح) state.n_tasks, n_نخها);
        کارگر_pool_پیاپی_func(ctx->کارگر_pool, dequantize_tiled_کارگر_loop, &state, n_tasks);
    }
}

تعريفنوع بنية {
    بنية htp_context      * ctx;
    بنية htp_thread_trace * traces;
    عائم                   * dst;
    ثابت __fp16            * vtcm_src;
    ثابت عائم             * src2;
    uint32_t                  n_ردیفها;
    uint32_t                  n_تعداد_ستون;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_تعداد_ستون;
    بنية fastdiv_مقادیر     n_نخها_div;
} output_transfer_ستون_chunk_state_t;

ساكن فراغ transfer_output_chunk_ستون_chunk_کارگر_fn(بیعلامت صحيح n, بیعلامت صحيح i, فراغ *داده) {
    (فراغ) n;
    output_transfer_ستون_chunk_state_t *st = (output_transfer_ستون_chunk_state_t *) داده;
    بنية htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_بلوکها = hmx_ceil_div(st->n_تعداد_ستون, 32);
    uint32_t b_اولی  = fastdiv(n_بلوکها * i, &st->n_نخها_div);
    uint32_t b_آخرین   = fastdiv(n_بلوکها * (i + 1), &st->n_نخها_div);
    uint32_t c_اولی  = b_اولی * 32;
    uint32_t c_آخرین   = b_آخرین * 32;
    uint32_t c_len    = c_آخرین - c_اولی;

    إذا (c_len == 0) أرجع;

    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_O_PROC, c_اولی);

    ثابت __fp16 *vtcm_src = st->vtcm_src + b_اولی * HTP_MM_HMX_TILE_N_ELMS;
    ثابت عائم *src2      = st->src2 ? (st->src2 + c_اولی) : عدم;
    عائم *dst             = st->dst + c_اولی;

    صحيح chunk_dst_تعداد_ستون = (صحيح)st->dst_تعداد_ستون - (صحيح)c_اولی;
    إذا (chunk_dst_تعداد_ستون > 0) {
        transfer_output_chunk_fp16_مقصد_fp32_ستون_chunk(
            dst, src2, vtcm_src, 0, st->n_ردیفها, c_len, st->n_تعداد_ستون,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_تعداد_ستون
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_اولی);
}

ساكن فراغ transfer_output_chunk_threaded(بنية htp_context *ctx, عائم *dst, ثابت عائم *src2, ثابت __fp16 *vtcm_src,
                                              صحيح n_ردیفها, صحيح n_تعداد_ستون, صحيح dst_stride, uint32_t src2_stride, صحيح dst_تعداد_ستون, صحيح n_نخها) {
    إذا (n_ردیفها <= 0) أرجع;

    uint32_t n_بلوکها = hmx_ceil_div((uint32_t) n_تعداد_ستون, 32);
    إذا (n_نخها > 1 && n_بلوکها >= (uint32_t)n_نخها) {
        بنية fastdiv_مقادیر n_نخها_div = (n_نخها == (صحيح)ctx->n_نخها) ? ctx->n_نخها_div : init_fastdiv_مقادیر(n_نخها);
        output_transfer_ستون_chunk_state_t ستون_state;
        ستون_state.dst = dst;
        ستون_state.src2 = src2;
        ستون_state.vtcm_src = vtcm_src;
        ستون_state.n_ردیفها = (uint32_t)n_ردیفها;
        ستون_state.n_تعداد_ستون = (uint32_t)n_تعداد_ستون;
        ستون_state.dst_stride = (uint32_t)dst_stride;
        ستون_state.src2_stride = src2_stride;
        ستون_state.dst_تعداد_ستون = (uint32_t)dst_تعداد_ستون;
        ستون_state.n_نخها_div = n_نخها_div;
        ستون_state.traces = ctx->trace;
        ستون_state.ctx = ctx;

        کارگر_pool_پیاپی_func(ctx->کارگر_pool, transfer_output_chunk_ستون_chunk_کارگر_fn, &ستون_state, n_نخها);
        أرجع;
    }

    size_t n_tot_chunks      = n_ردیفها;
    size_t n_chunks_در_هر_task = (n_نخها == 1) ? n_tot_chunks : hmx_ceil_div(n_ردیفها, n_نخها);
    n_chunks_در_هر_task        = hex_align_به_بالا(n_chunks_در_هر_task, 2);

    صحيح actual_نخها = hmx_ceil_div(n_ردیفها, n_chunks_در_هر_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_نخها;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_در_هر_task = n_chunks_در_هر_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_تعداد_ستون            = n_تعداد_ستون;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_تعداد_ستون          = dst_تعداد_ستون;
    state.traces            = ctx->trace;

    إذا (actual_نخها <= 1) {
        transfer_output_chunk_کارگر_fn(1, 0, &state);
    } وإلا {
        کارگر_pool_پیاپی_func(ctx->کارگر_pool, transfer_output_chunk_کارگر_fn, &state, actual_نخها);
    }
}

بنية activation_transfer_params {
    بنية htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    صحيح                           n_ردیفها;
    صحيح                           k_بلوک;
    صحيح                           k_stride;
    صحيح                           n_نخها;
    ثابت بنية fastdiv_مقادیر * act_نخها_div;
    ثابت بنية fastdiv_مقادیر * k_div;
    صحيح                           k_valid;
    عائم *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_بایتها;
    uint32_t                      act_elem_اندازه;  // 4=F32 src1, 2=F16 src1
};

ساكن فراغ transfer_activation_chunk_threaded(ثابت بنية activation_transfer_params * params) {
    بنية htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ثابت dma_addr_t              act_dma_addr       = params->act_dma_addr;
    صحيح                           n_ردیفها             = params->n_ردیفها;
    صحيح                           k_بلوک            = params->k_بلوک;
    صحيح                           k_stride           = params->k_stride;
    صحيح                           n_نخها          = params->n_نخها;
    ثابت بنية fastdiv_مقادیر * act_نخها_div    = params->act_نخها_div;
    ثابت بنية fastdiv_مقادیر * k_div              = params->k_div;
    صحيح                           k_valid            = params->k_valid;
    عائم *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_بایتها = params->vtcm_f32_act_بایتها;
    // element size of the activation rows (4 = F32, 2 = F16).
    ثابت uint32_t                act_elem_اندازه      = params->act_elem_اندازه ? params->act_elem_اندازه : (uint32_t) حجم(عائم);

    إذا (n_ردیفها <= 0) {
        أرجع;
    }

    ثابت size_t n_tasks = (n_ردیفها + 31) >> 5;
    إذا (act_elem_اندازه == حجم(عائم) && n_نخها > 1 && k_بلوک > 32 && n_tasks < (size_t) n_نخها) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_گام_ردیفها = 2;
        uint32_t dma_گام_ردیفها_لغزش = 1;
        إذا (vtcm_f32_act_بایتها > 0) {
            size_t thread_scratch_بایتها = hex_align_پایین(fastdiv(vtcm_f32_act_بایتها, act_نخها_div), 128);
            size_t thread_scratch_elements = thread_scratch_بایتها / حجم(عائم);
            size_t dma_گام_ردیفها_max = fastdiv(thread_scratch_elements / 2, k_div);
            إذا (dma_گام_ردیفها_max >= 4) {
                dma_گام_ردیفها = 4;
                dma_گام_ردیفها_لغزش = 2;
            }
        }

        activation_transfer_ستون_chunk_state_t ستون_state;
        ستون_state.dst = dst;
        ستون_state.act_dma_addr = act_dma_addr;
        ستون_state.n_ردیفها = n_ردیفها;
        ستون_state.k_بلوک = k_بلوک;
        ستون_state.k_stride = k_stride;
        ستون_state.k_valid = k_valid;
        ستون_state.n_ستون_chunks = n_نخها;
        ستون_state.n_نخها_div = *act_نخها_div;
        ستون_state.vtcm_f32_act = vtcm_f32_act;
        ستون_state.vtcm_f32_act_بایتها = vtcm_f32_act_بایتها;
        ستون_state.traces = ctx->trace;
        ستون_state.ctx = ctx;
        ستون_state.dma_گام_ردیفها = dma_گام_ردیفها;
        ستون_state.dma_گام_ردیفها_لغزش = dma_گام_ردیفها_لغزش;

        کارگر_pool_پیاپی_func(ctx->کارگر_pool, transfer_activation_chunk_ستون_chunk_کارگر_fn, &ستون_state, n_نخها);
        أرجع;
    }

    أكد(k_بلوک % HTP_MM_HMX_TILE_N_تعداد_ستون == 0 && k_stride % HTP_MM_HMX_TILE_N_تعداد_ستون == 0);

    size_t n_tot_chunks      = n_ردیفها;
    size_t n_chunks_در_هر_task = (n_نخها == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_نخها == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
    state.n_tot_chunks       = n_tot_chunks;
    state.n_chunks_در_هر_task  = n_chunks_در_هر_task;
    state.dst                = dst;
    state.act_dma_addr       = act_dma_addr;
    state.k_بلوک            = k_بلوک;
    state.k_stride           = k_stride;
    state.k_valid            = k_valid;
    state.traces             = ctx->trace;
    state.ctx                = ctx;
    state.vtcm_f32_act       = vtcm_f32_act;
    state.act_elem_اندازه = act_elem_اندازه;

    state.vtcm_f32_act_بایتها_در_هر_thread = hex_align_پایین(fastdiv(vtcm_f32_act_بایتها, act_نخها_div), 128);

    uint32_t dma_گام_ردیفها = 2;
    uint32_t dma_گام_ردیفها_لغزش = 1;
    إذا (state.vtcm_f32_act_بایتها_در_هر_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_بایتها_در_هر_thread / حجم(عائم);
        size_t dma_گام_ردیفها_max = fastdiv(thread_scratch_elements / 2, k_div);
        إذا (dma_گام_ردیفها_max >= 4) {
            dma_گام_ردیفها = 4;
            dma_گام_ردیفها_لغزش = 2;
        }
    }
    state.dma_گام_ردیفها      = dma_گام_ردیفها;
    state.dma_گام_ردیفها_لغزش = dma_گام_ردیفها_لغزش;

    صحيح فعال_نخها = hex_smin(n_نخها, (صحيح)state.n_tasks);
    إذا (state.n_tasks == 1 || n_نخها == 1) {
        transfer_activation_chunk_کارگر_fn(1, 0, &state);
    } وإلا {
        کارگر_pool_پیاپی_func(ctx->کارگر_pool, transfer_activation_chunk_کارگر_fn, &state, فعال_نخها);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

تعريفنوع بنية {
    __fp16 *       output;
    ثابت __fp16 * activation;
    ثابت __fp16 * وزن;
    ثابت __fp16 * scales;
    uint32_t       n_ردیف_tiles;
    uint32_t       n_ستون_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_کار_t;

ساكن فراغ hmx_matmul_کارگر_fn(فراغ * داده) {
    hmx_matmul_کار_t * کار = (hmx_matmul_کار_t *) داده;
    FARF(بالا_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", کار->n_ردیف_tiles, کار->n_ستون_tiles, کار->n_dot_tiles);
    core_dot_chunk_fp16(کار->output, کار->activation, کار->وزن, کار->scales, کار->n_ردیف_tiles, کار->n_ستون_tiles, کار->n_dot_tiles);
}

ساكن مضمن فراغ hmx_matmul_کار_init(hmx_matmul_کار_t * کار,
                                       __fp16 *           output,
                                       ثابت __fp16 *     activation,
                                       ثابت __fp16 *     وزن,
                                       ثابت __fp16 *     scales,
                                       uint32_t           n_ردیف_tiles,
                                       uint32_t           n_ستون_tiles,
                                       uint32_t           n_dot_tiles) {
    کار->output      = output;
    کار->activation  = activation;
    کار->وزن      = وزن;
    کار->scales      = scales;
    کار->n_ردیف_tiles = n_ردیف_tiles;
    کار->n_ستون_tiles = n_ستون_tiles;
    کار->n_dot_tiles = n_dot_tiles;
}

ساكن صحيح hmx_mm_2d_f32(بنية htp_context *ctx,
                                  dma_صف *وزن_dma,
                                  عائم *محدود dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_بایتها,
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
                                  صحيح dst_تعداد_ستون,
                                  صحيح m_chunk,
                                  صحيح n_chunk,
                                  صحيح pipeline,
                                  صحيح n_نخها,
                                  صحيح act_نخها,
                                  ثابت بنية fastdiv_مقادیر * act_نخها_div,
                                  ثابت بنية fastdiv_مقادیر * k_div,
                                  صحيح tile_اندازه,
                                  صحيح aligned_tile_اندازه,
                                  صحيح vtcm_اندازه) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت bool is_tiled = (tile_اندازه > 0);

    إذا (k % 32 != 0) {
        أرجع -1;
    }
    إذا (!hex_is_aligned(dst, حجم(عائم))) { أرجع -1; }

    size_t ردیف_stride = htp_tensor_get_ردیف_اندازه(وزن_type, k);
    إذا (ردیف_stride == 0) {
        أرجع -1;
    }

    tiled_dequant_task_fn_t dequant_کارگر_fn = hmx_get_dequant_کارگر_fn(وزن_type, is_tiled);

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_تعداد_ستون;
    ثابت صحيح n_وزن_tiles = k / htp_tensor_get_tile_k(وزن_type);
    ثابت بنية fastdiv_مقادیر n_k_tiles_div = init_fastdiv_مقادیر(n_k_tiles);

    ثابت size_t vec_dot_اندازه = k * حجم(__fp16);
    ثابت size_t vtcm_budget  = ctx->vtcm_اندازه;

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_اندازه : ردیف_stride;
    ثابت uint32_t dma_src_stride  = is_tiled ? tile_اندازه : وزن_stride;
    ثابت uint32_t dma_عرض_بایتها = is_tiled ? tile_اندازه : ردیف_stride;

    size_t m_chunk_n_ردیفها = m_chunk;
    size_t n_chunk_n_تعداد_ستون = n_chunk;
    size_t vtcm_استفاده_شده      = vtcm_اندازه;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, وزن_type, k, m_chunk_n_ردیفها, n_chunk_n_تعداد_ستون, 1, pipeline, act_نخها, aligned_tile_اندازه, src2_بایتها);

    vtcm_استفاده_شده = L.مجموع_بایتها;
    إذا (vtcm_استفاده_شده > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_استفاده_شده, vtcm_budget, m, k, n, m_chunk_n_ردیفها, n_chunk_n_تعداد_ستون);
        أرجع -1;
    }

    uint8_t * ثابت پایه = (uint8_t *) ctx->vtcm_پایه;
    __fp16  *vtcm_وزن_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, پایه, L.off_وزن[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, پایه, L.off_وزن[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_act);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, پایه, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, پایه, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, پایه, L.off_scratch[1], pipeline);
    فراغ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, پایه, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_scales);

    hmx_init_ستونی_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ثابت bool has_bias      = (src2_بایتها > 0 && src2_addr != 0);
    عائم   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(عائم, پایه, L.off_bias, has_bias);
    إذا (has_bias) {
        dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_bias, src2_addr), hex_align_به_بالا(src2_بایتها, 128), 0, src2_بایتها, 1);
        dma_صف_برداشت(وزن_dma);
    }

    FARF(بالا_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, وزن_type, m_chunk_n_ردیفها, n_chunk_n_تعداد_ستون, vtcm_استفاده_شده, vtcm_budget);

    صحيح n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_تعداد_ستون);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    إذا (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_کار_t کار_slots[2];  // persistent double-buffered job descriptors
        فراغ *vtcm_raw_bufs[4] = {
            vtcm_وزن_raw[0],
            vtcm_وزن_raw[1],
            dequant_کارگر_fn ? vtcm_وزن_raw[0] : vtcm_scratch0,
            dequant_کارگر_fn ? vtcm_وزن_raw[1] : vtcm_scratch1,
        };
        فراغ *vtcm_وزن_bufs[4] = {
            dequant_کارگر_fn ? vtcm_scratch0 : vtcm_وزن_raw[0],
            dequant_کارگر_fn ? vtcm_scratch1 : vtcm_وزن_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        فراغ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        لكل (size_t mr = 0; mr < m; mr += m_chunk_n_ردیفها) {
            ثابت size_t n_ردیفها = hex_smin(m - mr, m_chunk_n_ردیفها);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_اندازه,  // byte offset (F16/F32)
                .n_ردیفها = (صحيح) n_ردیفها,
                .k_بلوک = k,
                .k_stride = act_stride,
                .n_نخها = act_نخها,
                .act_نخها_div = act_نخها_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بایتها = L.act_f32_بایتها,
                .act_elem_اندازه = act_elem_اندازه,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            ثابت size_t   n_تعداد_ستون_A0 = hex_smin(n - 0 * n_chunk_n_تعداد_ستون, n_chunk_n_تعداد_ستون);
            ثابت uint32_t ارتفاع_A0 = is_tiled ? hmx_ceil_div(n_تعداد_ستون_A0, 32) * n_وزن_tiles : n_تعداد_ستون_A0;
            dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[0], وزن),
                           dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع_A0);

            إذا (1 < n_chunk_cnt) {
                ثابت size_t   n_تعداد_ستون_A1 = hex_smin(n - 1 * n_chunk_n_تعداد_ستون, n_chunk_n_تعداد_ستون);
                ثابت uint32_t ارتفاع_A1 = is_tiled ? hmx_ceil_div(n_تعداد_ستون_A1, 32) * n_وزن_tiles : n_تعداد_ستون_A1;
                dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[1], وزن + n_chunk_n_تعداد_ستون * وزن_stride),
                               dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                ثابت size_t nc    = i * n_chunk_n_تعداد_ستون;
                ثابت size_t nc_p2 = nc + 2 * n_chunk_n_تعداد_ستون;

                ثابت size_t n_تعداد_ستون    = hex_smin(n - nc, n_chunk_n_تعداد_ستون);
                ثابت size_t n_تعداد_ستون_p2 = hex_smin(n - nc_p2, n_chunk_n_تعداد_ستون);

                // 1. pop A_i
                فراغ * curr_raw = (فراغ *) dma_صف_برداشت(وزن_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
                    ctx, vtcm_وزن_bufs[i % 4], curr_raw,
                    n_تعداد_ستون, k, ردیف_stride, وزن_type,
                    n_k_tiles, n_k_tiles_div, dequant_کارگر_fn, n_نخها);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                إذا (i + 2 < n_chunk_cnt) {
                    ثابت uint32_t ارتفاع_p2 = is_tiled ? hmx_ceil_div(n_تعداد_ستون_p2, 32) * n_وزن_tiles : n_تعداد_ستون_p2;
                    dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[(i + 2) % 4], وزن + nc_p2 * وزن_stride),
                                   dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع_p2);
                }

                // 4. submit C_i
                hmx_matmul_کار_init(&کار_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_ردیفها, HTP_MM_HMX_TILE_N_ردیفها),
                                    hmx_ceil_div(n_تعداد_ستون, HTP_MM_HMX_TILE_N_تعداد_ستون), k / HTP_MM_HMX_TILE_N_ردیفها);
                hmx_صف_افزودن(ctx->hmx_صف, hmx_صف_make_desc(hmx_matmul_کارگر_fn, &کار_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                إذا (i > 0) {
                    hmx_صف_برداشت(ctx->hmx_صف);
                    ثابت size_t nc_prev = (i - 1) * n_chunk_n_تعداد_ستون;
                    ثابت size_t n_تعداد_ستون_prev = hex_smin(n - nc_prev, n_chunk_n_تعداد_ستون);
                    عائم *output_chunk = dst + (mr * dst_stride + nc_prev);
                    ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : عدم;
                    صحيح chunk_dst_تعداد_ستون = dst_تعداد_ستون - (صحيح)nc_prev;
                    إذا (chunk_dst_تعداد_ستون > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_ردیفها, n_تعداد_ستون_prev, dst_stride, src2_stride, chunk_dst_تعداد_ستون, n_نخها);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_صف_برداشت(ctx->hmx_صف);
            ثابت size_t nc_آخرین = (n_chunk_cnt - 1) * n_chunk_n_تعداد_ستون;
            ثابت size_t n_تعداد_ستون_آخرین = hex_smin(n - nc_آخرین, n_chunk_n_تعداد_ستون);
            عائم *output_chunk = dst + (mr * dst_stride + nc_آخرین);
            ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_آخرین) : عدم;
            صحيح chunk_dst_تعداد_ستون = dst_تعداد_ستون - (صحيح)nc_آخرین;
            إذا (chunk_dst_تعداد_ستون > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_ردیفها, n_تعداد_ستون_آخرین, dst_stride, src2_stride, chunk_dst_تعداد_ستون, n_نخها);
            }
        }
    } وإلا {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_کار_t کار;
        فراغ *vtcm_raw_bufs[2] = { vtcm_وزن_raw[0], dequant_کارگر_fn ? vtcm_وزن_raw[0] : vtcm_scratch0 };
        فراغ *vtcm_وزن_bufs[2] = { dequant_کارگر_fn ? vtcm_scratch0 : vtcm_وزن_raw[0], vtcm_scratch0 };

        لكل (size_t mr = 0; mr < m; mr += m_chunk_n_ردیفها) {
            ثابت size_t n_ردیفها = hex_smin(m - mr, m_chunk_n_ردیفها);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_اندازه,  // byte offset (F16/F32)
                .n_ردیفها = (صحيح) n_ردیفها,
                .k_بلوک = k,
                .k_stride = act_stride,
                .n_نخها = act_نخها,
                .act_نخها_div = act_نخها_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بایتها = L.act_f32_بایتها,
                .act_elem_اندازه = act_elem_اندازه,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            إذا (n > 0) {
                ثابت size_t n_تعداد_ستون = hex_smin(n, n_chunk_n_تعداد_ستون);
                ثابت uint32_t ارتفاع = is_tiled ? hmx_ceil_div(n_تعداد_ستون, 32) * n_وزن_tiles : n_تعداد_ستون;
                dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[0], وزن), dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع);
            }

            لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                ثابت size_t nc = (size_t) i * n_chunk_n_تعداد_ستون;
                ثابت size_t n_تعداد_ستون = hex_smin(n - nc, n_chunk_n_تعداد_ستون);
                ثابت size_t n_ردیف_tiles = hmx_ceil_div(n_ردیفها, HTP_MM_HMX_TILE_N_ردیفها);
                ثابت size_t n_ستون_tiles = hmx_ceil_div(n_تعداد_ستون, HTP_MM_HMX_TILE_N_تعداد_ستون);

                // A: Wait for weight DMA
                فراغ * curr_raw = (فراغ *) dma_صف_برداشت(وزن_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
                    ctx, vtcm_وزن_bufs[i % 2], curr_raw,
                    n_تعداد_ستون, k, ردیف_stride, وزن_type,
                    n_k_tiles, n_k_tiles_div, dequant_کارگر_fn, n_نخها);

                // Start weight DMA for the next chunk early
                ثابت size_t nc_بعدی = nc + n_chunk_n_تعداد_ستون;
                إذا (nc_بعدی < n) {
                    ثابت size_t n_تعداد_ستون_بعدی = hex_smin(n - nc_بعدی, n_chunk_n_تعداد_ستون);
                    ثابت uint32_t ارتفاع_بعدی = is_tiled ? hmx_ceil_div(n_تعداد_ستون_بعدی, 32) * n_وزن_tiles : n_تعداد_ستون_بعدی;
                    dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[(i + 1) % 2], وزن + nc_بعدی * وزن_stride), dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع_بعدی);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_کار_init(&کار, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 2], vtcm_scales, n_ردیف_tiles, n_ستون_tiles, k / HTP_MM_HMX_TILE_N_ردیفها);
                hmx_صف_افزودن(ctx->hmx_صف, hmx_صف_make_desc(hmx_matmul_کارگر_fn, &کار));
                hmx_صف_برداشت(ctx->hmx_صف);

                // D: Output Store
                عائم *output_chunk = dst + (mr * dst_stride + nc);
                ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : عدم;
                صحيح chunk_dst_تعداد_ستون = dst_تعداد_ستون - (صحيح)nc;
                إذا (chunk_dst_تعداد_ستون > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_ردیفها, n_تعداد_ستون, dst_stride, src2_stride, chunk_dst_تعداد_ستون, n_نخها);
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
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];

    ثابت صحيح وزن_type = (صحيح) src0->type;
    ثابت صحيح k           = (صحيح) act->ne[0];
    ثابت صحيح k_valid     = (صحيح) act->ne[0];
    ثابت صحيح m           = (صحيح) (act->ne[1] * act->ne[2] * act->ne[3]);
    ثابت uint32_t act_elem_اندازه = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت صحيح act_stride  = (صحيح) (act->nb[1] / act_elem_اندازه);
    ثابت dma_addr_t act_dma_addr = act->داده;

    إذا (k % 32 != 0) { أرجع HTP_وضعیت_NO_SUPPORT; }

    size_t ردیف_stride = htp_tensor_get_ردیف_اندازه(وزن_type, k);
    إذا (ردیف_stride == 0) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_تعداد_ستون;
    ثابت صحيح n_وزن_tiles = k / htp_tensor_get_tile_k(وزن_type);
    ثابت بنية fastdiv_مقادیر n_k_tiles_div = init_fastdiv_مقادیر(n_k_tiles);

    ثابت صحيح tile_اندازه       = kparams->tile_اندازه;
    ثابت صحيح aligned_tile_اندازه = kparams->aligned_tile_اندازه;

    ثابت bool is_tiled       = (tile_اندازه > 0);
    tiled_dequant_task_fn_t dequant_کارگر_fn = hmx_get_dequant_کارگر_fn(وزن_type, is_tiled);
    ثابت size_t vtcm_budget  = ctx->vtcm_اندازه;

    ثابت صحيح m_chunk_n_ردیفها  = kparams->m_chunk;
    ثابت صحيح n_chunk_n_تعداد_ستون  = kparams->n_chunk;
    ثابت صحيح pipeline        = kparams->pipeline;
    ثابت صحيح n_نخها       = octx->n_نخها;
    ثابت صحيح act_نخها     = kparams->n_act_نخها;
    ثابت بنية fastdiv_مقادیر * act_نخها_div = &kparams->div_n_act_نخها;
    ثابت بنية fastdiv_مقادیر * k_div           = &kparams->div_ne00_padded;

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_اندازه : ردیف_stride;
    ثابت uint32_t dma_عرض_بایتها = is_tiled ? tile_اندازه : ردیف_stride;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, وزن_type, k, m_chunk_n_ردیفها, n_chunk_n_تعداد_ستون, 1, pipeline, act_نخها, aligned_tile_اندازه, 0);

    إذا (L.مجموع_بایتها > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.مجموع_بایتها, vtcm_budget, m, k, m_chunk_n_ردیفها, n_chunk_n_تعداد_ستون);
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت پایه = (uint8_t *) ctx->vtcm_پایه;
    __fp16  *vtcm_وزن_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, پایه, L.off_وزن[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, پایه, L.off_وزن[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_act);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, پایه, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, پایه, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, پایه, L.off_scratch[1], pipeline);
    فراغ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, پایه, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_scales);

    hmx_init_ستونی_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    صحيح m_شروع = 0;
    صحيح m_ردیفها  = m;
    إذا (octx->ctx->mdev.شمار > 1) {
        bool can_تقسیم = htp_tensor_can_ردیف_افراز(act, act_elem_اندازه);
        إذا (can_تقسیم) {
            لكل (uint32_t p = 0; p < n_weights; ++p) {
                ثابت بنية htp_tensor * محدود dst = octx->dsts[p];
                إذا (!htp_tensor_can_ردیف_افراز(dst, حجم(عائم))) {
                    can_تقسیم = false;
                    اكسر;
                }
            }
        }
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز((uint32_t) m, can_تقسیم ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        m_شروع = (صحيح) range.شروع;
        m_ردیفها  = (صحيح) range.شمار;
    }

    إذا (m_ردیفها == 0) {
        أرجع HTP_وضعیت_OK;
    }

    FARF(بالا_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_شروع, m_شروع + m_ردیفها, k, وزن_type, m_chunk_n_ردیفها, n_chunk_n_تعداد_ستون, L.مجموع_بایتها, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت size_t mr_انتها = (size_t)(m_شروع + m_ردیفها);

    إذا (pipeline) {
        hmx_matmul_کار_t کار_slots[2];
        فراغ *vtcm_raw_bufs[4] = {
            vtcm_وزن_raw[0],
            vtcm_وزن_raw[1],
            dequant_کارگر_fn ? vtcm_وزن_raw[0] : vtcm_scratch0,
            dequant_کارگر_fn ? vtcm_وزن_raw[1] : vtcm_scratch1,
        };
        فراغ *vtcm_وزن_bufs[4] = {
            dequant_کارگر_fn ? vtcm_scratch0 : vtcm_وزن_raw[0],
            dequant_کارگر_fn ? vtcm_scratch1 : vtcm_وزن_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        فراغ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        لكل (size_t mr = (size_t) m_شروع; mr < mr_انتها; mr += m_chunk_n_ردیفها) {
            ثابت size_t n_ردیفها = hex_smin(mr_انتها - mr, m_chunk_n_ردیفها);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_اندازه,
                .n_ردیفها = (صحيح) n_ردیفها,
                .k_بلوک = k,
                .k_stride = act_stride,
                .n_نخها = act_نخها,
                .act_نخها_div = act_نخها_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بایتها = L.act_f32_بایتها,
                .act_elem_اندازه = act_elem_اندازه,
            };
            transfer_activation_chunk_threaded(&act_params);

            لكل (uint32_t p = 0; p < n_weights; p++) {
                ثابت بنية htp_tensor * محدود src_w = octx->src[p];
                ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
                إذا (!src_w || !dst) استمر;

                ثابت dma_addr_t وزن       = src_w->داده;
                dma_صف * وزن_dma      = octx->ctx->dma[0];
                عائم * dst_ptr            = (عائم *) dst->داده;
                ثابت size_t n             = src_w->ne[1];
                إذا (n == 0) استمر;
                ثابت size_t وزن_stride = src_w->nb[1];
                ثابت size_t dst_stride    = dst->nb[1] / حجم(عائم);
                ثابت صحيح dst_تعداد_ستون         = (صحيح) dst->ne[0];
                ثابت صحيح n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_تعداد_ستون);

                ثابت uint32_t dma_src_stride = is_tiled ? tile_اندازه : وزن_stride;

                ثابت size_t   n_تعداد_ستون_A0 = hex_smin(n - 0 * n_chunk_n_تعداد_ستون, n_chunk_n_تعداد_ستون);
                ثابت uint32_t ارتفاع_A0 = is_tiled ? hmx_ceil_div(n_تعداد_ستون_A0, 32) * n_وزن_tiles : n_تعداد_ستون_A0;
                dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[0], وزن),
                               dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع_A0);

                إذا (1 < n_chunk_cnt) {
                    ثابت size_t   n_تعداد_ستون_A1 = hex_smin(n - 1 * n_chunk_n_تعداد_ستون, n_chunk_n_تعداد_ستون);
                    ثابت uint32_t ارتفاع_A1 = is_tiled ? hmx_ceil_div(n_تعداد_ستون_A1, 32) * n_وزن_tiles : n_تعداد_ستون_A1;
                    dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[1], وزن + n_chunk_n_تعداد_ستون * وزن_stride),
                                   dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع_A1);
                }

                لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                    ثابت size_t nc    = i * n_chunk_n_تعداد_ستون;
                    ثابت size_t nc_p2 = nc + 2 * n_chunk_n_تعداد_ستون;

                    ثابت size_t n_تعداد_ستون    = hex_smin(n - nc, n_chunk_n_تعداد_ستون);
                    ثابت size_t n_تعداد_ستون_p2 = hex_smin(n - nc_p2, n_chunk_n_تعداد_ستون);

                    فراغ * curr_raw = (فراغ *) dma_صف_برداشت(وزن_dma).dst;

                    dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
                        ctx, vtcm_وزن_bufs[i % 4], curr_raw,
                        n_تعداد_ستون, k, ردیف_stride, وزن_type,
                        n_k_tiles, n_k_tiles_div, dequant_کارگر_fn, n_نخها);

                    إذا (i + 2 < n_chunk_cnt) {
                        ثابت uint32_t ارتفاع_p2 = is_tiled ? hmx_ceil_div(n_تعداد_ستون_p2, 32) * n_وزن_tiles : n_تعداد_ستون_p2;
                        dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[(i + 2) % 4], وزن + nc_p2 * وزن_stride),
                                       dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع_p2);
                    }

                    hmx_matmul_کار_init(&کار_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_ردیفها, HTP_MM_HMX_TILE_N_ردیفها),
                                        hmx_ceil_div(n_تعداد_ستون, HTP_MM_HMX_TILE_N_تعداد_ستون), k / HTP_MM_HMX_TILE_N_ردیفها);
                    hmx_صف_افزودن(ctx->hmx_صف, hmx_صف_make_desc(hmx_matmul_کارگر_fn, &کار_slots[i % 2]));

                    إذا (i > 0) {
                        hmx_صف_برداشت(ctx->hmx_صف);
                        ثابت size_t nc_prev = (i - 1) * n_chunk_n_تعداد_ستون;
                        ثابت size_t n_تعداد_ستون_prev = hex_smin(n - nc_prev, n_chunk_n_تعداد_ستون);
                        عائم *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        صحيح chunk_dst_تعداد_ستون = dst_تعداد_ستون - (صحيح)nc_prev;
                        إذا (chunk_dst_تعداد_ستون > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output_bufs[(i - 1) % 2], n_ردیفها, n_تعداد_ستون_prev, dst_stride, 0, chunk_dst_تعداد_ستون, n_نخها);
                        }
                    }
                }

                hmx_صف_برداشت(ctx->hmx_صف);
                ثابت size_t nc_آخرین = (n_chunk_cnt - 1) * n_chunk_n_تعداد_ستون;
                ثابت size_t n_تعداد_ستون_آخرین = hex_smin(n - nc_آخرین, n_chunk_n_تعداد_ستون);
                عائم *output_chunk = dst_ptr + (mr * dst_stride + nc_آخرین);
                صحيح chunk_dst_تعداد_ستون = dst_تعداد_ستون - (صحيح)nc_آخرین;
                إذا (chunk_dst_تعداد_ستون > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_ردیفها, n_تعداد_ستون_آخرین, dst_stride, 0, chunk_dst_تعداد_ستون, n_نخها);
                }
            }
        }
    } وإلا {
        hmx_matmul_کار_t کار;
        فراغ *vtcm_raw_bufs[2] = { vtcm_وزن_raw[0], dequant_کارگر_fn ? vtcm_وزن_raw[0] : vtcm_scratch0 };
        فراغ *vtcm_وزن_bufs[2] = { dequant_کارگر_fn ? vtcm_scratch0 : vtcm_وزن_raw[0], vtcm_scratch0 };

        لكل (size_t mr = (size_t) m_شروع; mr < mr_انتها; mr += m_chunk_n_ردیفها) {
            ثابت size_t n_ردیفها = hex_smin(mr_انتها - mr, m_chunk_n_ردیفها);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_اندازه,
                .n_ردیفها = (صحيح) n_ردیفها,
                .k_بلوک = k,
                .k_stride = act_stride,
                .n_نخها = act_نخها,
                .act_نخها_div = act_نخها_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بایتها = L.act_f32_بایتها,
                .act_elem_اندازه = act_elem_اندازه,
            };
            transfer_activation_chunk_threaded(&act_params);

            لكل (uint32_t p = 0; p < n_weights; p++) {
                ثابت بنية htp_tensor * محدود src_w = octx->src[p];
                ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
                إذا (!src_w || !dst) استمر;

                ثابت dma_addr_t وزن       = src_w->داده;
                dma_صف * وزن_dma      = octx->ctx->dma[0];
                عائم * dst_ptr            = (عائم *) dst->داده;
                ثابت size_t n             = src_w->ne[1];
                إذا (n == 0) استمر;
                ثابت size_t وزن_stride = src_w->nb[1];
                ثابت size_t dst_stride    = dst->nb[1] / حجم(عائم);
                ثابت صحيح dst_تعداد_ستون         = (صحيح) dst->ne[0];
                ثابت صحيح n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_تعداد_ستون);

                ثابت uint32_t dma_src_stride = is_tiled ? tile_اندازه : وزن_stride;

                إذا (n > 0) {
                    ثابت size_t n_تعداد_ستون = hex_smin(n, n_chunk_n_تعداد_ستون);
                    ثابت uint32_t ارتفاع = is_tiled ? hmx_ceil_div(n_تعداد_ستون, 32) * n_وزن_tiles : n_تعداد_ستون;
                    dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[0], وزن), dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع);
                }

                لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                    ثابت size_t nc = (size_t) i * n_chunk_n_تعداد_ستون;
                    ثابت size_t n_تعداد_ستون = hex_smin(n - nc, n_chunk_n_تعداد_ستون);
                    ثابت size_t n_ردیف_tiles = hmx_ceil_div(n_ردیفها, HTP_MM_HMX_TILE_N_ردیفها);
                    ثابت size_t n_ستون_tiles = hmx_ceil_div(n_تعداد_ستون, HTP_MM_HMX_TILE_N_تعداد_ستون);

                    فراغ * curr_raw = (فراغ *) dma_صف_برداشت(وزن_dma).dst;

                    dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
                        ctx, vtcm_وزن_bufs[i % 2], curr_raw,
                        n_تعداد_ستون, k, ردیف_stride, وزن_type,
                        n_k_tiles, n_k_tiles_div, dequant_کارگر_fn, n_نخها);

                    ثابت size_t nc_بعدی = nc + n_chunk_n_تعداد_ستون;
                    إذا (nc_بعدی < n) {
                        ثابت size_t n_تعداد_ستون_بعدی = hex_smin(n - nc_بعدی, n_chunk_n_تعداد_ستون);
                        ثابت uint32_t ارتفاع_بعدی = is_tiled ? hmx_ceil_div(n_تعداد_ستون_بعدی, 32) * n_وزن_tiles : n_تعداد_ستون_بعدی;
                        dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[(i + 1) % 2], وزن + nc_بعدی * وزن_stride), dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع_بعدی);
                    }

                    hmx_matmul_کار_init(&کار, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 2], vtcm_scales, n_ردیف_tiles, n_ستون_tiles, k / HTP_MM_HMX_TILE_N_ردیفها);
                    hmx_صف_افزودن(ctx->hmx_صف, hmx_صف_make_desc(hmx_matmul_کارگر_fn, &کار));
                    hmx_صف_برداشت(ctx->hmx_صف);

                    عائم *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    صحيح chunk_dst_تعداد_ستون = dst_تعداد_ستون - (صحيح)nc;
                    إذا (chunk_dst_تعداد_ستون > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output, n_ردیفها, n_تعداد_ستون, dst_stride, 0, chunk_dst_تعداد_ستون, n_نخها);
                    }
                }
            }
        }
    }

    أرجع HTP_وضعیت_OK;
}

ساكن مضمن dma_addr_t hmx_mm_وزن_batch_داده(ثابت hmx_mm_batched_params_t *params,
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
                                 صحيح m_chunk, صحيح n_chunk, صحيح pipeline, صحيح n_نخها, صحيح act_نخها, صحيح vtcm_اندازه,
                                 ثابت بنية fastdiv_مقادیر * act_نخها_div, ثابت بنية fastdiv_مقادیر * k_div) {
    صحيح ret = 0;
    لكل (صحيح b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        لكل (صحيح b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->وزن_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_بایتها,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_وزن_batch_داده(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_اندازه, params->وزن_stride,
                                params->وزن_type, params->k, params->dst_stride, params->bias_stride, params->dst_تعداد_ستون,
                                m_chunk, n_chunk, pipeline, n_نخها, act_نخها,
                                act_نخها_div, k_div, params->tile_اندازه, params->aligned_tile_اندازه, vtcm_اندازه);
        }
    }
    أرجع ret;
}

ساكن صحيح hmx_mm_batched(بنية htp_context *ctx, ثابت hmx_mm_batched_params_t *params,
                          صحيح m_chunk, صحيح n_chunk, صحيح pipeline, صحيح n_نخها, صحيح act_نخها,
                          ثابت بنية fastdiv_مقادیر * act_نخها_div,
                          ثابت بنية fastdiv_مقادیر * k_div,
                          صحيح vtcm_اندازه) {
    ثابت size_t ردیف_stride = htp_tensor_get_ردیف_اندازه(params->وزن_type, params->k);
    إذا (ردیف_stride == 0) { أرجع -1; }
    إذا (params->act_stride < params->k || (params->tile_اندازه == 0 && (size_t) params->وزن_stride < ردیف_stride) || params->dst_stride < params->dst_تعداد_ستون) { أرجع -1; }
    إذا (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { أرجع -1; }
    إذا (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { أرجع -1; }
    // N (the weight row count) does not have to be 32-aligned:
    إذا (params->k % 32 != 0) { أرجع -1; }
    إذا (!hex_is_aligned(params->dst, حجم(عائم))) { أرجع -1; }

    ثابت صحيح گروه_اندازه = params->r2;
    ثابت size_t vtcm_budget  = ctx->vtcm_اندازه;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool پیاپی_grouped = (params->وزن_type == HTP_TYPE_F16 && params->tile_اندازه == 0 && گروه_اندازه > 1 && (size_t)vtcm_اندازه <= vtcm_budget);
    إذا (!پیاپی_grouped) {
        أرجع hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_نخها, act_نخها, vtcm_اندازه, act_نخها_div, k_div);
    }

    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_ردیفها = m_chunk;
    size_t n_chunk_n_تعداد_ستون = n_chunk;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_ردیفها, n_chunk_n_تعداد_ستون, گروه_اندازه, false, act_نخها, 0, params->bias_بایتها);

    إذا (L.مجموع_بایتها > vtcm_budget) {
        FARF(بالا_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        أرجع hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_نخها, act_نخها, vtcm_اندازه, act_نخها_div, k_div);
    }

    uint8_t * ثابت پایه = (uint8_t *) ctx->vtcm_پایه;
    __fp16  *vtcm_وزن     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_وزن[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, پایه, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR(فراغ, پایه, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_scales);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, پایه, L.off_act_f32);

    ثابت bool has_bias      = (params->bias_بایتها > 0 && params->bias_addr != 0);
    عائم   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(عائم, پایه, L.off_bias, has_bias);
    إذا (has_bias) {
        dma_صف_افزودن(params->وزن_dma, dma_make_داده(vtcm_bias, params->bias_addr), hex_align_به_بالا(params->bias_بایتها, 128), 0, params->bias_بایتها, 1);
        dma_صف_برداشت(params->وزن_dma);
    }

    hmx_init_ستونی_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(بالا_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, گروه_اندازه, params->ne13,
            m_chunk_n_ردیفها, n_chunk_n_تعداد_ستون,
            L.مجموع_بایتها, vtcm_budget);

    ثابت size_t fp16_ردیف_بایتها   = (size_t) params->k * حجم(__fp16);
    ثابت size_t وزن_ردیف_بایتها = (size_t) params->وزن_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_کار_t کار;

    لكل (صحيح b3 = 0; b3 < params->ne13; ++b3) {
        لكل (صحيح b2_پایه = 0; b2_پایه < params->ne12; b2_پایه += گروه_اندازه) {
            ثابت dma_addr_t وزن_گروه = hmx_mm_وزن_batch_داده(params, b2_پایه, b3);
            dma_صف * وزن_dma = params->وزن_dma;

            لكل (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_ردیفها) {
                ثابت size_t n_ردیفها = hex_smin((size_t) params->m - mr, m_chunk_n_ردیفها);
                ثابت size_t n_ردیف_tiles = hmx_ceil_div((صحيح) n_ردیفها, HTP_MM_HMX_TILE_N_ردیفها);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                لكل (صحيح g = 0; g < گروه_اندازه; ++g) {
                    ثابت dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_پایه + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_اندازه;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_سر_stride;
                    بنية activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_ردیفها = (صحيح) n_ردیفها,
                        .k_بلوک = params->k,
                        .k_stride = params->act_stride,
                        .n_نخها = act_نخها,
                        .act_نخها_div = act_نخها_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_بایتها = L.act_f32_بایتها,
                        .act_elem_اندازه = params->act_elem_اندازه,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    ثابت size_t n_تعداد_ستون_اولی = hex_smin((size_t) params->n, n_chunk_n_تعداد_ستون);
                    dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_scratch0, وزن_گروه),
                                      fp16_ردیف_بایتها, وزن_ردیف_بایتها, fp16_ردیف_بایتها, n_تعداد_ستون_اولی);
                }
                إذا (n_chunk_n_تعداد_ستون < (size_t) params->n) {
                    ثابت size_t n_تعداد_ستون_دومی = hex_smin((size_t) params->n - n_chunk_n_تعداد_ستون, n_chunk_n_تعداد_ستون);
                    ثابت dma_addr_t دومی_وزن_chunk = وزن_گروه + n_chunk_n_تعداد_ستون * وزن_ردیف_بایتها;
                    dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_scratch1, دومی_وزن_chunk),
                                      fp16_ردیف_بایتها, وزن_ردیف_بایتها, fp16_ردیف_بایتها, n_تعداد_ستون_دومی);
                }

                لكل (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_تعداد_ستون) {
                    ثابت size_t n_تعداد_ستون      = hex_smin((size_t) params->n - nc, n_chunk_n_تعداد_ستون);
                    ثابت size_t n_ستون_tiles = hmx_ceil_div((صحيح) n_تعداد_ستون, HTP_MM_HMX_TILE_N_تعداد_ستون);

                    {
                        فراغ * curr_raw = (فراغ *) dma_صف_برداشت(وزن_dma).dst;

                        ثابت size_t n_تعداد_ستون_tiled = hex_align_به_بالا(n_تعداد_ستون, HTP_MM_HMX_TILE_N_تعداد_ستون);

                        hmx_interleave_ردیفها_مقصد_tiles(vtcm_وزن, (ثابت __fp16 *) curr_raw, (uint32_t) n_تعداد_ستون, params->k, params->k, 0, (uint32_t) n_تعداد_ستون_tiled);

                        ثابت size_t nc_بعدی = nc + n_chunk_n_تعداد_ستون * 2;
                        إذا (nc_بعدی < (size_t) params->n) {
                            ثابت size_t n_تعداد_ستون_بعدی = hex_smin((size_t) params->n - nc_بعدی, n_chunk_n_تعداد_ستون);
                            ثابت dma_addr_t بعدی_وزن_chunk = وزن_گروه + nc_بعدی * وزن_ردیف_بایتها;

                            dma_صف_افزودن(وزن_dma, dma_make_داده(curr_raw, بعدی_وزن_chunk),
                                              fp16_ردیف_بایتها, وزن_ردیف_بایتها, fp16_ردیف_بایتها, n_تعداد_ستون_بعدی);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    لكل (صحيح g = 0; g < گروه_اندازه; ++g) {
                        {
                            ثابت __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_سر_stride;
                            hmx_matmul_کار_init(&کار, vtcm_output, vtcm_act_g, vtcm_وزن, vtcm_scales, n_ردیف_tiles, n_ستون_tiles, params->k / 32);
                            hmx_صف_افزودن(ctx->hmx_صف, hmx_صف_make_desc(hmx_matmul_کارگر_fn, &کار));
                            hmx_صف_برداشت(ctx->hmx_صف);
                        }

                        {
                            عائم *output = hmx_mm_dst_batch_ptr(params, b2_پایه + g, b3) + mr * params->dst_stride + nc;
                            ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : عدم;
                            صحيح chunk_dst_تعداد_ستون = params->dst_تعداد_ستون - (صحيح)nc;
                            إذا (chunk_dst_تعداد_ستون > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (صحيح) n_ردیفها, (صحيح) n_تعداد_ستون,
                                                               params->dst_stride, params->bias_stride, chunk_dst_تعداد_ستون, n_نخها);
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
            صحيح شروع_ردیف,
            صحيح n_ردیفها,
            صحيح k_بلوک,
            ثابت بنية mmid_ردیف_mapping *ماتریس_ردیفها,
            صحيح cur_a,
            صحيح mapping_stride,
            صحيح ne11,
            size_t nb11,
            size_t nb12,
            صحيح cne1,
            صحيح n_نخها,
            صحيح k_valid) {
    إذا (n_ردیفها <= 0) أرجع;
    صحيح chunks_در_هر_thread = hmx_ceil_div(n_ردیفها, n_نخها);
    chunks_در_هر_thread = hex_align_به_بالا(chunks_در_هر_thread, 2);

    صحيح actual_نخها = hmx_ceil_div(n_ردیفها, chunks_در_هر_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_نخها,
        .n_tot_chunks      = n_ردیفها,
        .n_chunks_در_هر_task = chunks_در_هر_thread,
        .k_بلوک           = k_بلوک,
        .ماتریس_ردیفها       = ماتریس_ردیفها,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_مقادیر(ne11) : (بنية fastdiv_مقادیر){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .شروع_ردیف         = شروع_ردیف,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    کارگر_callback_t کارگر_fn = ne11 == 1 ? transfer_activation_chunk_gathered_کارگر_flat_fn :
                                              transfer_activation_chunk_gathered_کارگر_fn;

    إذا (actual_نخها <= 1) {
        کارگر_fn(1, 0, &state);
    } وإلا {
        کارگر_pool_پیاپی_func(ctx->کارگر_pool, کارگر_fn, &state, actual_نخها);
    }
}

ساكن فراغ transfer_output_chunk_scattered_threaded(
            بنية htp_context *ctx,
            عائم *dst,
            ثابت __fp16 *vtcm_src,
            صحيح شروع_ردیف,
            صحيح n_ردیفها,
            صحيح n_تعداد_ستون,
            ثابت بنية mmid_ردیف_mapping *ماتریس_ردیفها,
            صحيح cur_a,
            صحيح mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            صحيح cne1,
            صحيح n_نخها) {
    إذا (n_ردیفها <= 0) أرجع;
    صحيح chunks_در_هر_thread = hmx_ceil_div(n_ردیفها, n_نخها);
    chunks_در_هر_thread = hex_align_به_بالا(chunks_در_هر_thread, 2);

    صحيح actual_نخها = hmx_ceil_div(n_ردیفها, chunks_در_هر_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_نخها,
        .n_tot_chunks      = n_ردیفها,
        .n_chunks_در_هر_task = chunks_در_هر_thread,
        .n_تعداد_ستون            = n_تعداد_ستون,
        .ماتریس_ردیفها       = ماتریس_ردیفها,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .شروع_ردیف         = شروع_ردیف,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    إذا (actual_نخها <= 1) {
        transfer_output_chunk_scattered_کارگر_fn(1, 0, &state);
    } وإلا {
        کارگر_pool_پیاپی_func(ctx->کارگر_pool, transfer_output_chunk_scattered_کارگر_fn, &state, actual_نخها);
    }
}

ساكن صحيح hmx_mm_id_2d_f32(بنية htp_context *ctx,
                                         dma_صف *وزن_dma,
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
                                         ثابت بنية mmid_ردیف_mapping *ماتریس_ردیفها,
                                         صحيح cur_a,
                                         صحيح mapping_stride,
                                         صحيح m_شروع,
                                         صحيح m_انتها,
                                         صحيح n_نخها) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت صحيح cne1 = m;
    ثابت صحيح m_core = m_انتها - m_شروع;
    ثابت صحيح m_core_padded = hex_align_به_بالا(m_core > 0 ? m_core : 1, 32);

    إذا (k % 32 != 0 || n % 32 != 0) { أرجع -1; }
    إذا (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { أرجع -1; }

    size_t ردیف_stride = htp_tensor_get_ردیف_اندازه(وزن_type, k);
    إذا (ردیف_stride == 0) {
        أرجع -1;
    }

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_تعداد_ستون;
    ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(وزن_type);
    ثابت بنية fastdiv_مقادیر n_k_tiles_div = init_fastdiv_مقادیر(n_k_tiles);

    ثابت size_t vec_dot_اندازه = k * حجم(__fp16);
    ثابت size_t vtcm_budget  = ctx->vtcm_اندازه;
    size_t vtcm_استفاده_شده = 0;

    ثابت bool is_tiled = (tile_اندازه > 0);
    tiled_dequant_task_fn_t dequant_کارگر_fn = hmx_get_dequant_کارگر_fn(وزن_type, is_tiled);

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_اندازه : ردیف_stride;
    ثابت uint32_t dma_src_stride  = is_tiled ? tile_اندازه : وزن_stride;
    ثابت uint32_t dma_عرض_بایتها = is_tiled ? tile_اندازه : ردیف_stride;

    ثابت size_t qweight_ردیف_stride = is_tiled ? (size_t)(n_وزن_tiles * aligned_tile_اندازه) / 32 : 0;
    ثابت size_t وزن_ردیف_stride = is_tiled ? qweight_ردیف_stride : ردیف_stride;

    size_t اندازه_در_هر_n = 0, اندازه_در_هر_m = 0, اندازه_در_هر_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(وزن_type, k, /*pipeline=*/false, aligned_tile_اندازه,
                                  &اندازه_در_هر_n, &اندازه_در_هر_m, &اندازه_در_هر_mn);

    ثابت size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_ردیفها = 0, n_chunk_n_تعداد_ستون = 0;
    إذا (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, اندازه_در_هر_n, اندازه_در_هر_m, اندازه_در_هر_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_هزینه_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_هزینه_A_CONVERT, &m_chunk_n_ردیفها, &n_chunk_n_تعداد_ستون, &vtcm_استفاده_شده)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        أرجع -1;
    }

    ثابت size_t وزن_area_اندازه = hex_align_به_بالا(n_chunk_n_تعداد_ستون * وزن_ردیف_stride, HTP_MM_HMX_TILE_اندازه);
    ثابت size_t act_area_اندازه    = hex_align_به_بالا(m_chunk_n_ردیفها * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);
    ثابت size_t output_area_اندازه = hex_align_به_بالا(m_chunk_n_ردیفها * n_chunk_n_تعداد_ستون * حجم(__fp16), HTP_MM_HMX_TILE_اندازه);

    size_t scratch0_اندازه = hex_align_به_بالا(n_chunk_n_تعداد_ستون * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_پایه;
    __fp16  *vtcm_وزن   = وزن_area_اندازه ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, وزن_area_اندازه) : عدم;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_اندازه);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_اندازه);
    فراغ    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_اندازه);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_استفاده_شده = vtcm_ptr - (uint8_t *) ctx->vtcm_پایه;
    إذا (vtcm_استفاده_شده > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_استفاده_شده, vtcm_budget);
        أرجع -1;
    }

    hmx_init_ستونی_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_کار_t کار;
    فراغ *vtcm_raw_bufs[2] = { vtcm_وزن, dequant_کارگر_fn ? vtcm_وزن : vtcm_scratch0 };
    فراغ *vtcm_وزن_bufs[2] = { dequant_کارگر_fn ? vtcm_scratch0 : vtcm_وزن, vtcm_scratch0 };
    ثابت صحيح n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_تعداد_ستون);

    لكل (size_t mr = (size_t) m_شروع; mr < (size_t) m_انتها; mr += m_chunk_n_ردیفها) {
        ثابت size_t n_ردیفها = hex_smin((size_t) m_انتها - mr, m_chunk_n_ردیفها);
        ثابت size_t n_ردیف_tiles = hmx_ceil_div(n_ردیفها, HTP_MM_HMX_TILE_N_ردیفها);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (صحيح) mr, (صحيح) n_ردیفها, k,
            ماتریس_ردیفها, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_نخها, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        إذا (n > 0) {
            ثابت size_t n_تعداد_ستون = hex_smin((size_t) n, n_chunk_n_تعداد_ستون);
            ثابت uint32_t ارتفاع = is_tiled ? hmx_ceil_div(n_تعداد_ستون, 32) * n_وزن_tiles : n_تعداد_ستون;
            dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[0], وزن),
                           dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع);
        }

        لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
            ثابت size_t nc = (size_t) i * n_chunk_n_تعداد_ستون;
            ثابت size_t n_تعداد_ستون = hex_smin((size_t) n - nc, n_chunk_n_تعداد_ستون);
            ثابت size_t n_ستون_tiles = hmx_ceil_div(n_تعداد_ستون, HTP_MM_HMX_TILE_N_تعداد_ستون);

            // A: Wait for weight DMA
            فراغ * curr_raw = (فراغ *) dma_صف_برداشت(وزن_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
                ctx, vtcm_وزن_bufs[i % 2], curr_raw,
                n_تعداد_ستون, k, ردیف_stride, وزن_type,
                n_k_tiles, n_k_tiles_div, dequant_کارگر_fn, n_نخها
            );

            // Start weight DMA for the next chunk early
            ثابت size_t nc_بعدی = nc + n_chunk_n_تعداد_ستون;
            إذا (nc_بعدی < (size_t) n) {
                ثابت size_t n_تعداد_ستون_بعدی = hex_smin((size_t) n - nc_بعدی, n_chunk_n_تعداد_ستون);
                ثابت uint32_t ارتفاع_بعدی = is_tiled ? hmx_ceil_div(n_تعداد_ستون_بعدی, 32) * n_وزن_tiles : n_تعداد_ستون_بعدی;
                dma_صف_افزودن(وزن_dma, dma_make_داده(vtcm_raw_bufs[(i + 1) % 2], وزن + nc_بعدی * وزن_stride),
                               dma_dst_stride, dma_src_stride, dma_عرض_بایتها, ارتفاع_بعدی);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_کار_init(&کار, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 2], vtcm_scales, n_ردیف_tiles, n_ستون_tiles, k / HTP_MM_HMX_TILE_N_ردیفها);
            hmx_صف_افزودن(ctx->hmx_صف, hmx_صف_make_desc(hmx_matmul_کارگر_fn, &کار));
            hmx_صف_برداشت(ctx->hmx_صف);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (صحيح) mr, (صحيح) n_ردیفها, (صحيح) n_تعداد_ستون,
                ماتریس_ردیفها, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_نخها);
        }
    }

    أرجع 0;
}

// --- Dispatchers and Public Entry Points ---

ساكن صحيح hmx_mm_op_matmul(بنية htp_ops_context * octx, ثابت بنية htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    صحيح k = (صحيح) src0->ne[0];
    صحيح n = (صحيح) src0->ne[1];
    ثابت صحيح m_مجموع    = (صحيح) act->ne[1];
    ثابت uint32_t act_elem_اندازه = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت صحيح act_stride = (صحيح) (act->nb[1] / act_elem_اندازه);

    صحيح m_شروع = 0;
    صحيح m_ردیفها  = m_مجموع;
    إذا (octx->ctx->mdev.شمار > 1) {
        bool can_تقسیم = htp_tensor_can_ردیف_افراز(dst, حجم(عائم)) &&
                         htp_tensor_can_ردیف_افراز(act, act_elem_اندازه);
        إذا (src2 && src2->ne[1] > 1 && !htp_tensor_can_ردیف_افراز(src2, حجم(عائم))) {
            can_تقسیم = false;
        }
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز((uint32_t) m_مجموع, can_تقسیم ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        m_شروع = (صحيح) range.شروع;
        m_ردیفها  = (صحيح) range.شمار;
    }

    إذا (m_ردیفها == 0) {
        أرجع HTP_وضعیت_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_بایتها = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    إذا (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / حجم(عائم));
        src2_addr = src2->داده + m_شروع * src2_stride * حجم(عائم);
        src2_بایتها = (size_t) kparams->vtcm_bias_اندازه;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ثابت صحيح dst_stride = (صحيح)(dst->nb[1] / حجم(عائم));
    عائم       * dst_ptr = (عائم *) dst->داده + m_شروع * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ثابت dma_addr_t act_addr = act->داده + (size_t) m_شروع * act_stride * act_elem_اندازه;

    صحيح ret = -1;
    ثابت صحيح n_نخها = kparams->n_نخها;
    إذا (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        hmx_mm_batched_params_t batch_params = {
            .dst               = dst_ptr,
            .bias_addr         = src2_addr,
            .bias_بایتها        = src2_بایتها,
            .act_dma_addr      = act_addr,
            .وزن            = src0->داده,
            .وزن_dma        = octx->ctx->dma[0],
            .m                 = m_ردیفها,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_اندازه     = act_elem_اندازه,
            .وزن_stride     = (صحيح) src0->nb[1],
            .وزن_type       = (صحيح) src0->type,
            .dst_stride        = dst_stride,
            .dst_تعداد_ستون          = (صحيح) dst->ne[0],
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
                             kparams->pipeline, n_نخها,
                             kparams->n_act_نخها,
                             &kparams->div_n_act_نخها,
                             &kparams->div_ne00_padded,
                             kparams->vtcm_اندازه);
    } وإلا {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_بایتها,
            act_addr, src0->داده,
            m_ردیفها, k, n, act_stride, act_elem_اندازه, (صحيح) src0->nb[1], (صحيح) src0->type, (صحيح) act->ne[0],
            dst_stride, src2_stride, (صحيح)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_نخها,
            kparams->n_act_نخها,
            &kparams->div_n_act_نخها,
            &kparams->div_ne00_padded,
            kparams->tile_اندازه, kparams->aligned_tile_اندازه, kparams->vtcm_اندازه
        );
    }

    إذا (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        أرجع HTP_وضعیت_داخلی_ERR;
    }
    أرجع HTP_وضعیت_OK;
}

ساكن مضمن فراغ htp_mm_tensor_collapse_ردیفها(بنية htp_tensor * c, ثابت بنية htp_tensor * t, uint32_t stride) {
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

    ثابت صحيح وضعیت = htp_mm_init_context(octx, kparams);
    إذا (وضعیت != HTP_وضعیت_OK) {
        أرجع وضعیت;
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
        htp_mm_tensor_collapse_ردیفها(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_ردیفها(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        بنية htp_tensor src2_collapsed;
        ثابت بنية htp_tensor * src2 = octx->src[2];
        إذا (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ثابت uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_ردیفها(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        ثابت صحيح وضعیت = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        إذا (src2) {
            octx->src[2] = src2;
        }
        أرجع وضعیت;
    }

    أرجع op_matmul_impl(octx);
}

ساكن صحيح hmx_mm_op_matmul_id(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx
) {
    ثابت uint32_t *                ماتریس_ردیف_counts = mmctx->ماتریس_ردیف_counts;
    ثابت بنية mmid_ردیف_mapping * ماتریس_ردیفها       = mmctx->ماتریس_ردیفها;
    htp_matmul_tensors_preamble;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت صحيح n_ids = octx->src[2]->ne[0];
    ثابت صحيح n_as  = ne02;

    ثابت bool mdev_تقسیم = (octx->ctx->mdev.شمار > 1) && htp_tensor_can_ردیف_افراز(dst, حجم(عائم));
    إذا (octx->ctx->mdev.شمار > 1 && !mdev_تقسیم && octx->ctx->mdev.idx > 0) {
        أرجع HTP_وضعیت_OK;
    }
    uint32_t n_فعال = 0;
    إذا (mdev_تقسیم) {
        لكل (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            إذا (ماتریس_ردیف_counts[a] > 0) n_فعال++;
        }
    }
    ثابت bool expert_تقسیم = mdev_تقسیم && (n_فعال >= octx->ctx->mdev.شمار);

    uint32_t هدف_dev = 0;
    لكل (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ثابت int32_t cne1 = ماتریس_ردیف_counts[cur_a];
        إذا (cne1 == 0) استمر;

        ثابت صحيح m_padded = hex_align_به_بالا(cne1, 32);
        صحيح m_شروع = 0, m_انتها = m_padded;
        إذا (expert_تقسیم) {
            ثابت bool my_expert = (هدف_dev == octx->ctx->mdev.idx);
            إذا (++هدف_dev == octx->ctx->mdev.شمار) {
                هدف_dev = 0;
            }
            إذا (!my_expert) استمر;
        } وإلا إذا (mdev_تقسیم) {
            ثابت bool can_تقسیم = (uint32_t) cne1 >= octx->ctx->mdev.شمار;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز((uint32_t) m_padded, can_تقسیم ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
            m_شروع = (صحيح) range.شروع;
            m_انتها   = (صحيح) (range.شروع + range.شمار);
        }
        إذا (m_شروع >= m_انتها) استمر;

        صحيح ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (عائم*) dst->داده, (عائم*) act->داده,
                                   src0->داده + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (صحيح) src0->nb[1], (صحيح) src0->type,
                                   kparams->tile_اندازه, kparams->aligned_tile_اندازه,
                                   ماتریس_ردیفها, cur_a, mmctx->mapping_stride,
                                   m_شروع, m_انتها, (صحيح) octx->n_نخها);
        إذا (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            أرجع HTP_وضعیت_NO_SUPPORT;
        }
    }

    أرجع HTP_وضعیت_OK;
}

ساكن صحيح hvx_mm_matmul_id(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx,
    work_صف_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ثابت uint32_t src0_ردیف_اندازه_padded = mmctx->src0_ردیف_اندازه_padded;
    ثابت uint32_t act_nrows            = mmctx->act_nrows;

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت بنية htp_tensor * محدود ids = octx->src[2];
    ثابت size_t src0_ردیف_اندازه = nb01;

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (ne10 + qk - 1) / qk;
    ثابت uint32_t مجموع_nb = act_nrows * nb;

    work_صف_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    إذا (act_nrows < octx->n_نخها) {
        n_quant_tasks = MIN(مجموع_nb, octx->n_نخها);
        quant_task_func = htp_mm_act_quant_بلوک_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_اولی = (مجموع_nb * ith) / n_quant_tasks;
            uint32_t ib_آخرین  = (مجموع_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_اولی[ith] = ib_اولی;
            mmctx->quant_ib_آخرین[ith]  = ib_آخرین;
            mmctx->quant_r[ith]        = ib_اولی / nb;
            mmctx->quant_c[ith]        = ib_اولی % nb;
        }
    } وإلا {
        n_quant_tasks = MIN(act_nrows, octx->n_نخها);
        quant_task_func = htp_mm_act_quant_ردیف_func(src0->type, act->type);
    }
    size_t act_ردیف_اندازه = htp_mm_tiled_act_ردیف_اندازه(src0->type, ne10);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_نخها,
                                 0, src0_ردیف_اندازه, act_ردیف_اندازه, 0, kparams->n_prefetch, true, false);

    ثابت size_t vtcm_اندازه = L.مجموع_بایتها;

    FARF(بالا_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_بایتها, L.act_بایتها, L.dst_بایتها, vtcm_اندازه);

    FARF(بالا_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->داده,
         act->داده, dst->داده);

    // Make sure the reserved vtcm size is sufficient
    إذا (octx->ctx->vtcm_اندازه < vtcm_اندازه) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_اندازه, vtcm_اندازه);
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت پایه = (uint8_t *) octx->ctx->vtcm_پایه;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_src0);
    mmctx->vtcm_bias     = عدم;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_ردیف_اندازه_padded;
    mmctx->vtcm_act_stride     = act_ردیف_اندازه;
    mmctx->vtcm_act_raw_stride = hex_دور_به_بالا(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_اندازه_در_هر_thread = fastdiv(L.src0_بایتها, &octx->n_نخها_div);
    mmctx->vtcm_act_اندازه_در_هر_thread  = L.act_بایتها;
    mmctx->vtcm_dst_اندازه_در_هر_thread  = fastdiv(L.dst_بایتها, &octx->n_نخها_div);

    mmctx->cur_m_شروع = 0;
    mmctx->cur_m_ردیفها  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_ردیفها_در_هر_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_صف_پیاپی(octx->ctx->work_صف, quant_task_func, mmctx, n_quant_tasks);

    work_صف_پیاپی(octx->ctx->work_صف, hvx_mmid_task_func, mmctx, octx->n_نخها);

    أرجع HTP_وضعیت_OK;
}

ساكن صحيح hmx_mm_op_matmul_id_nx(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx
) {
    ثابت uint32_t *                ماتریس_ردیف_counts = mmctx->ماتریس_ردیف_counts;
    ثابت بنية mmid_ردیف_mapping * ماتریس_ردیفها       = mmctx->ماتریس_ردیفها;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];
    ثابت صحيح n_as = src0->ne[2];

    bool mdev_تقسیم = (octx->ctx->mdev.شمار > 1);
    لكل (uint32_t p = 0; p < n_weights && mdev_تقسیم; ++p) {
        ثابت بنية htp_tensor * محدود dst = octx->dsts[p];
        mdev_تقسیم = htp_tensor_can_ردیف_افراز(dst, حجم(عائم));
    }
    إذا (octx->ctx->mdev.شمار > 1 && !mdev_تقسیم && octx->ctx->mdev.idx > 0) {
        أرجع HTP_وضعیت_OK;
    }
    uint32_t n_فعال = 0;
    إذا (mdev_تقسیم) {
        لكل (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            إذا (ماتریس_ردیف_counts[a] > 0) n_فعال++;
        }
    }
    ثابت bool expert_تقسیم = mdev_تقسیم && (n_فعال >= octx->ctx->mdev.شمار);

    uint32_t هدف_dev = 0;
    لكل (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ثابت int32_t cne1 = ماتریس_ردیف_counts[cur_a];
        إذا (cne1 == 0) استمر;

        ثابت صحيح m_padded = hex_align_به_بالا(cne1, 32);
        صحيح m_شروع = 0, m_انتها = m_padded;
        إذا (expert_تقسیم) {
            ثابت bool my_expert = (هدف_dev == octx->ctx->mdev.idx);
            إذا (++هدف_dev == octx->ctx->mdev.شمار) {
                هدف_dev = 0;
            }
            إذا (!my_expert) استمر;
        } وإلا إذا (mdev_تقسیم) {
            ثابت bool can_تقسیم = (uint32_t) cne1 >= octx->ctx->mdev.شمار;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز((uint32_t) m_padded, can_تقسیم ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
            m_شروع = (صحيح) range.شروع;
            m_انتها   = (صحيح) (range.شروع + range.شمار);
        }
        إذا (m_شروع >= m_انتها) استمر;

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * محدود src_w = octx->src[p];
            ثابت بنية htp_tensor * محدود dst   = octx->dsts[p];
            إذا (!src_w || !dst) استمر;

            صحيح ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (عائم*) dst->داده, (عائم*) act->داده,
                                       src_w->داده + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (صحيح) src_w->nb[1], (صحيح) src_w->type,
                                       kparams->tile_اندازه, kparams->aligned_tile_اندازه,
                                       ماتریس_ردیفها, cur_a, mmctx->mapping_stride,
                                       m_شروع, m_انتها, (صحيح) octx->n_نخها);
            إذا (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                أرجع HTP_وضعیت_NO_SUPPORT;
            }
        }
    }

    أرجع HTP_وضعیت_OK;
}

ساكن صحيح hvx_mm_matmul_id_nx(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx,
    work_صف_func_t hvx_mmid_task_func
) {
    ثابت uint32_t src0_ردیف_اندازه_padded = mmctx->src0_ردیف_اندازه_padded;
    ثابت uint32_t act_nrows            = mmctx->act_nrows;

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];
    ثابت بنية htp_tensor * محدود ids  = octx->src[n_weights + 1];
    ثابت size_t src0_ردیف_اندازه = src0->nb[1];

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ثابت uint32_t مجموع_nb = act_nrows * nb;

    work_صف_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    إذا (act_nrows < octx->n_نخها) {
        n_quant_tasks = MIN(مجموع_nb, octx->n_نخها);
        quant_task_func = htp_mm_act_quant_بلوک_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_اولی = (مجموع_nb * ith) / n_quant_tasks;
            uint32_t ib_آخرین  = (مجموع_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_اولی[ith] = ib_اولی;
            mmctx->quant_ib_آخرین[ith]  = ib_آخرین;
            mmctx->quant_r[ith]        = ib_اولی / nb;
            mmctx->quant_c[ith]        = ib_اولی % nb;
        }
    } وإلا {
        n_quant_tasks = MIN(act_nrows, octx->n_نخها);
        quant_task_func = htp_mm_act_quant_ردیف_func(src0->type, act->type);
    }
    size_t act_ردیف_اندازه = htp_mm_tiled_act_ردیف_اندازه(src0->type, act->ne[0]);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_نخها,
                                 0, src0_ردیف_اندازه, act_ردیف_اندازه, 0, kparams->n_prefetch, true, false);

    ثابت size_t vtcm_اندازه = L.مجموع_بایتها;

    إذا (octx->ctx->vtcm_اندازه < vtcm_اندازه) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_اندازه, vtcm_اندازه);
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت پایه = (uint8_t *) octx->ctx->vtcm_پایه;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_ردیف_اندازه;
    mmctx->vtcm_act_raw_stride = hex_دور_به_بالا(act->ne[0] * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_اندازه_در_هر_thread = fastdiv(L.src0_بایتها, &octx->n_نخها_div);
    mmctx->vtcm_act_اندازه_در_هر_thread  = L.act_بایتها;
    mmctx->vtcm_dst_اندازه_در_هر_thread  = fastdiv(L.dst_بایتها, &octx->n_نخها_div);

    mmctx->cur_m_شروع = 0;
    mmctx->cur_m_ردیفها  = act_nrows;

    FARF(بالا_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.مجموع_بایتها, octx->ctx->vtcm_اندازه, octx->n_نخها);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_ردیفها_در_هر_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_صف_پیاپی(octx->ctx->work_صف, quant_task_func, mmctx, n_quant_tasks);

    work_صف_پیاپی(octx->ctx->work_صف, hvx_mmid_task_func, mmctx, octx->n_نخها);

    أرجع HTP_وضعیت_OK;
}

ساكن مضمن فراغ پیمایش_expert_ids_n(
    ثابت بنية htp_tensor * ids,
    ثابت uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    بنية mmid_ردیف_mapping * ماتریس_ردیفها,
    uint32_t mapping_stride
) {
    ثابت size_t ids_nb1 = ids->nb[1];
    ثابت uint8_t * ids_داده = (ثابت uint8_t *) ids->داده;

    لكل (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ثابت int32_t * ردیف_ptr = (ثابت int32_t *) (ids_داده + iid1 * ids_nb1);
        لكل (uint32_t id = 0; id < n_ids; ++id) {
            ثابت int32_t i02 = ردیف_ptr[id];
            إذا (i02 < 0) {
                استمر;
            }
            أكد(i02 < n_as);

            إذا (ماتریس_ردیفها) {
                ماتریس_ردیفها[i02 * mapping_stride + counts[i02]] = (بنية mmid_ردیف_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

ساكن مضمن فراغ پیمایش_expert_ids(
    ثابت بنية htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    بنية mmid_ردیف_mapping * ماتریس_ردیفها,
    uint32_t mapping_stride
) {
    ثابت size_t ids_nb0 = ids->nb[0];

    إذا (ids_nb0 == 4) {
        بدل (n_ids) {
            حالة 8:  پیمایش_expert_ids_n(ids, 8,     n_as, counts, ماتریس_ردیفها, mapping_stride); اكسر;
            حالة 4:  پیمایش_expert_ids_n(ids, 4,     n_as, counts, ماتریس_ردیفها, mapping_stride); اكسر;
            حالة 2:  پیمایش_expert_ids_n(ids, 2,     n_as, counts, ماتریس_ردیفها, mapping_stride); اكسر;
            افتراضي: پیمایش_expert_ids_n(ids, n_ids, n_as, counts, ماتریس_ردیفها, mapping_stride); اكسر;
        }
    } وإلا {
        // Strided fallback
        ثابت size_t ids_nb1 = ids->nb[1];
        ثابت uint8_t * ids_داده = (ثابت uint8_t *) ids->داده;
        لكل (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ثابت int32_t * ردیف_ptr = (ثابت int32_t *) (ids_داده + iid1 * ids_nb1);
            لكل (uint32_t id = 0; id < n_ids; ++id) {
                ثابت int32_t i02 = *(ثابت int32_t *) ((ثابت uint8_t *) ردیف_ptr + id * ids_nb0);
                إذا (i02 < 0) {
                    استمر;
                }
                أكد(i02 < n_as);

                إذا (ماتریس_ردیفها) {
                    ماتریس_ردیفها[i02 * mapping_stride + counts[i02]] = (بنية mmid_ردیف_mapping) { id, iid1 };
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

    ثابت صحيح وضعیت = htp_mm_init_context(octx, kparams);
    إذا (وضعیت != HTP_وضعیت_OK) {
        أرجع وضعیت;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ثابت بنية htp_tensor * محدود ids = octx->src[2];
    إذا (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت size_t src0_ردیف_اندازه = nb01;
    ثابت size_t dst_ردیف_اندازه  = nb1;

    ثابت size_t src0_ردیف_اندازه_padded = hex_دور_به_بالا(src0_ردیف_اندازه, 128);

    ثابت uint32_t src0_nrows = ne01;  // per expert
    ثابت uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ثابت صحيح n_ids = ids->ne[0];  // n_expert_used
    ثابت صحيح n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_پایه;
    uint32_t   mapping_stride    = 1;
    uint32_t * ماتریس_ردیف_counts = (uint32_t *) mapping_buf;
    بنية mmid_ردیف_mapping * ماتریس_ردیفها = عدم;

    إذا (act_nrows > 1) {
        ثابت size_t ماتریس_ردیف_counts_اندازه = n_as * حجم(uint32_t);
        أكد(octx->ctx->ddr_spad_اندازه >= ماتریس_ردیف_counts_اندازه);

        hex_l2fetch_بلوک((ثابت فراغ *) ids->داده, ids->ne[1] * ids->nb[1]);

        memset(ماتریس_ردیف_counts, 0, ماتریس_ردیف_counts_اندازه);
        پیمایش_expert_ids(ids, n_ids, n_as, ماتریس_ردیف_counts, عدم, 0);

        uint32_t max_شمار = hvx_reduce_max_i32((ثابت uint8_t *) ماتریس_ردیف_counts, n_as);
        mapping_stride = max_شمار > 0 ? max_شمار : 1;

        size_t ماتریس_ردیف_map_اندازه  = n_as * mapping_stride * حجم(بنية mmid_ردیف_mapping);
        ثابت size_t مجموع_map_اندازه = ماتریس_ردیف_counts_اندازه + ماتریس_ردیف_map_اندازه;

        إذا (مجموع_map_اندازه > octx->ctx->ddr_spad_اندازه) {
            mapping_buf = memalign(128, مجموع_map_اندازه);
            إذا (!mapping_buf) {
                أرجع HTP_وضعیت_داخلی_ERR;
            }
        }

        ماتریس_ردیف_counts = (uint32_t *) mapping_buf;
        ماتریس_ردیفها       = (بنية mmid_ردیف_mapping *) (mapping_buf + ماتریس_ردیف_counts_اندازه);

        memset(ماتریس_ردیف_counts, 0, n_as * حجم(uint32_t));
        پیمایش_expert_ids(ids, n_ids, n_as, ماتریس_ردیف_counts, ماتریس_ردیفها, mapping_stride);
    }

    mmctx->ماتریس_ردیف_counts    = ماتریس_ردیف_counts;
    mmctx->ماتریس_ردیفها          = ماتریس_ردیفها;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_ردیف_اندازه_padded = src0_ردیف_اندازه_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_شروع          = 0;
    mmctx->cur_m_ردیفها           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    صحيح s;
    إذا (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } وإلا {
        uint32_t src0_ردیف_شروع = 0;
        uint32_t src0_ردیف_انتها   = src0_nrows;
        إذا (octx->ctx->mdev.شمار > 1) {
            ثابت bool can_تقسیم = htp_tensor_can_ردیف_افراز(dst, حجم(عائم));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(src0_nrows, can_تقسیم ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
            src0_ردیف_شروع = range.شروع;
            src0_ردیف_انتها   = range.شروع + range.شمار;
        }

        إذا (src0_ردیف_شروع >= src0_ردیف_انتها) {
            إذا (mapping_buf != octx->ctx->ddr_spad_پایه) {
                حرر(mapping_buf);
            }
            أرجع HTP_وضعیت_OK;
        }

        ثابت uint32_t nrows = src0_ردیف_انتها - src0_ردیف_شروع;
        mmctx->src0_ردیف_شروع = src0_ردیف_شروع;
        mmctx->src0_ردیف_انتها   = src0_ردیف_انتها;

        mmctx->src0_nrows_در_هر_thread = fastdiv(nrows + octx->n_نخها - 1, &octx->n_نخها_div);
        mmctx->src0_nrows_در_هر_thread = hex_دور_به_بالا(mmctx->src0_nrows_در_هر_thread, 32);

        إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } وإلا {
            s = HTP_وضعیت_NO_SUPPORT;
        }
    }

    إذا (mapping_buf != octx->ctx->ddr_spad_پایه) {
        حرر(mapping_buf);
    }

    أرجع s;
}

صحيح op_matmul_id_nx(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;

    ثابت صحيح وضعیت = htp_mm_init_context(octx, kparams);
    إذا (وضعیت != HTP_وضعیت_OK) {
        أرجع وضعیت;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * محدود src0 = octx->src[0];
    ثابت بنية htp_tensor * محدود act  = octx->src[n_weights];
    ثابت بنية htp_tensor * محدود ids  = octx->src[n_weights + 1];
    إذا (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }
    لكل (uint32_t p = 0; p < n_weights; p++) {
        إذا (htp_tensor_is_extended(octx->dsts[p])) {
            أرجع HTP_وضعیت_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    ثابت size_t src0_ردیف_اندازه = src0->nb[1];
    ثابت size_t src0_ردیف_اندازه_padded = hex_دور_به_بالا(src0_ردیف_اندازه, 128);

    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ثابت صحيح n_ids = ids->ne[0];
    ثابت صحيح n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_پایه;
    uint32_t   mapping_stride    = 1;
    uint32_t * ماتریس_ردیف_counts = (uint32_t *) mapping_buf;
    بنية mmid_ردیف_mapping * ماتریس_ردیفها = عدم;

    إذا (act_nrows > 1) {
        ثابت size_t ماتریس_ردیف_counts_اندازه = n_as * حجم(uint32_t);
        أكد(octx->ctx->ddr_spad_اندازه >= ماتریس_ردیف_counts_اندازه);

        hex_l2fetch_بلوک((ثابت فراغ *) ids->داده, ids->ne[1] * ids->nb[1]);

        memset(ماتریس_ردیف_counts, 0, ماتریس_ردیف_counts_اندازه);
        پیمایش_expert_ids(ids, n_ids, n_as, ماتریس_ردیف_counts, عدم, 0);

        uint32_t max_شمار = hvx_reduce_max_i32((ثابت uint8_t *) ماتریس_ردیف_counts, n_as);
        mapping_stride = max_شمار > 0 ? max_شمار : 1;

        size_t ماتریس_ردیف_map_اندازه  = n_as * mapping_stride * حجم(بنية mmid_ردیف_mapping);
        ثابت size_t مجموع_map_اندازه = ماتریس_ردیف_counts_اندازه + ماتریس_ردیف_map_اندازه;

        إذا (مجموع_map_اندازه > octx->ctx->ddr_spad_اندازه) {
            mapping_buf = memalign(128, مجموع_map_اندازه);
            إذا (!mapping_buf) {
                أرجع HTP_وضعیت_داخلی_ERR;
            }
        }

        ماتریس_ردیف_counts = (uint32_t *) mapping_buf;
        ماتریس_ردیفها       = (بنية mmid_ردیف_mapping *) (mapping_buf + ماتریس_ردیف_counts_اندازه);

        memset(ماتریس_ردیف_counts, 0, n_as * حجم(uint32_t));
        پیمایش_expert_ids(ids, n_ids, n_as, ماتریس_ردیف_counts, ماتریس_ردیفها, mapping_stride);
    }

    mmctx->ماتریس_ردیف_counts    = ماتریس_ردیف_counts;
    mmctx->ماتریس_ردیفها          = ماتریس_ردیفها;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_ردیف_اندازه_padded = src0_ردیف_اندازه_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_شروع          = 0;
    mmctx->cur_m_ردیفها           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    صحيح s;
    إذا (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } وإلا {
        إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } وإلا {
            s = HTP_وضعیت_NO_SUPPORT;
        }
    }

    إذا (mapping_buf != octx->ctx->ddr_spad_پایه) {
        حرر(mapping_buf);
    }

    أرجع s;
}
ساكن صحيح op_matmul_nx_impl(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    ثابت صحيح وضعیت = htp_mm_init_context(octx, kparams);
    إذا (وضعیت != HTP_وضعیت_OK) {
        أرجع وضعیت;
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
    mmctx->cur_m_ردیفها  = act_nrows;

    ثابت size_t src0_ردیف_اندازه = src0->nb[1];

    إذا (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ثابت uint32_t مجموع_nb = act_nrows * nb;

    کارگر_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    إذا (act_nrows < octx->n_نخها) {
        n_quant_tasks = MIN(مجموع_nb, octx->n_نخها);
        quant_task_func = htp_mm_act_quant_بلوک_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_اولی = (مجموع_nb * ith) / n_quant_tasks;
            uint32_t ib_آخرین  = (مجموع_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_اولی[ith] = ib_اولی;
            mmctx->quant_ib_آخرین[ith]  = ib_آخرین;
            mmctx->quant_r[ith]        = ib_اولی / nb;
            mmctx->quant_c[ith]        = ib_اولی % nb;
        }
    } وإلا {
        n_quant_tasks = MIN(act_nrows, octx->n_نخها);
        quant_task_func = htp_mm_act_quant_ردیف_func(src0->type, act->type);
    }

    ثابت size_t act_ردیف_اندازه = htp_mm_tiled_act_ردیف_اندازه(src0->type, act->ne[0]);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_نخها,
                                 0, src0_ردیف_اندازه, act_ردیف_اندازه, 0, kparams->n_prefetch, false, true);

    ثابت size_t vtcm_اندازه = L.مجموع_بایتها;

    إذا (octx->ctx->vtcm_اندازه < vtcm_اندازه) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_اندازه, vtcm_اندازه);
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت پایه = (uint8_t *) octx->ctx->vtcm_پایه;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_ردیف_اندازه;
    mmctx->vtcm_act_raw_stride = hex_دور_به_بالا(act->ne[0] * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_اندازه_در_هر_thread = fastdiv(L.src0_بایتها, &octx->n_نخها_div);
    mmctx->vtcm_act_اندازه_در_هر_thread  = L.act_بایتها;
    mmctx->vtcm_dst_اندازه_در_هر_thread  = fastdiv(L.dst_بایتها, &octx->n_نخها_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_ردیفها_در_هر_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_صف_پیاپی(octx->ctx->work_صف, quant_task_func, mmctx, n_quant_tasks);

    work_صف_پیاپی(octx->ctx->work_صف, hvx_mm_nx_2d_tiled, mmctx, octx->n_نخها);

    أرجع HTP_وضعیت_OK;
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

        htp_mm_tensor_collapse_ردیفها(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        لكل (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ثابت بنية htp_tensor * d = octx->dsts[p];
            ثابت uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_ردیفها(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        ثابت صحيح وضعیت = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        لكل (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        أرجع وضعیت;
    }

    أرجع op_matmul_nx_impl(octx);
}
