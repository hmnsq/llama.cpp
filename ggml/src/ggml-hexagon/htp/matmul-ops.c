#ہدایت clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>
#شامل <HAP_compute_res.h>

#شامل <math.h>
#شامل <string.h>
#شامل <stdatomic.h>

#شامل "dma-queue.h"
#شامل "hvx-utils.h"
#شامل "hvx-dump.h"
#شامل "hvx-arith.h"
#شامل "hvx-reduce.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "matmul-ops.h"
#شامل "htp-vtcm.h"

قسمتعریف ساخت {
    اعشاری        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_بائٹیں;
    dma_addr_t    act_dma_addr;
    dma_addr_t    وزن;
    dma_قطار *   وزن_dma;
    صحیح           m;
    صحیح           k;
    صحیح           n;
    صحیح           act_stride;
    uint32_t      act_elem_سائز;  // 4=F32 src1, 2=F16 src1
    صحیح           وزن_stride;
    صحیح           وزن_type;
    صحیح           dst_stride;
    صحیح           dst_ستون_تعداد;
    uint32_t      bias_stride;
    صحیح           tile_سائز;
    صحیح           aligned_tile_سائز;
    صحیح           ne02;
    صحیح           ne03;
    صحیح           ne12;
    صحیح           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    صحیح           r2;
    صحیح           r3;
    ساخت fastdiv_اقدار div_r2;
    ساخت fastdiv_اقدار div_r3;
} hmx_mm_batched_params_t;

ساکن bool htp_matmul_has_extended_وزن(ثابت ساخت htp_ops_context * octx, uint32_t n_weights) {
    برائے (uint32_t i = 0; i < n_weights; ++i) {
        اگر (htp_tensor_is_extended(octx->src[i])) {
            واپس true;
        }
    }
    واپس false;
}

ساخت htp_mm_context {
    ثابت حرف * type;
    ساخت htp_ops_context * octx;
    ثابت ساخت htp_tensor * act;

    خالی (*vec_dot_1x1)(ثابت uint32_t n, اعشاری * محدود s0,
         ثابت خالی * محدود vx0,
         ثابت خالی * محدود vy0);

    خالی (*vec_dot_2x1)(ثابت uint32_t n, اعشاری * محدود s0,
         ثابت خالی * محدود vx0, ثابت خالی * محدود vx1,
         ثابت خالی * محدود vy0);

    خالی (*vec_dot_2x2)(ثابت uint32_t n, اعشاری * محدود s0, اعشاری * محدود s1,
         ثابت خالی * محدود vx0, ثابت خالی * محدود vx1,
         ثابت خالی * محدود vy0, ثابت خالی * محدود vy1);

    خالی (*vec_dot_32x1)(ثابت uint32_t n, اعشاری * محدود s,
         ثابت خالی * محدود vx,
         ثابت خالی * محدود vy, uint32_t valid_صفیں,
         ثابت اعشاری * محدود sz);

    خالی (*vec_dot_32x2)(ثابت uint32_t n, اعشاری * محدود s0, اعشاری * محدود s1,
         ثابت خالی * محدود vx,
         ثابت خالی * محدود vy0, ثابت خالی * محدود vy1, uint32_t valid_صفیں,
         ثابت اعشاری * محدود sz0, ثابت اعشاری * محدود sz1);

    // Precomputed values
    uint32_t src0_nrows_فی_thread;
    uint32_t src0_صف_آغاز;
    uint32_t src0_صف_انتہا;
    uint32_t src0_صف_سائز_padded;
    uint32_t act_nrows;
    uint32_t cur_m_آغاز;
    uint32_t cur_m_صفیں;

    ساخت fastdiv_اقدار mm_div_ne12_ne1;
    ساخت fastdiv_اقدار mm_div_ne1;
    ساخت fastdiv_اقدار mm_div_r2;
    ساخت fastdiv_اقدار mm_div_r3;
    ساخت fastdiv_اقدار mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_پہلا[WORK_قطار_MAX_N_دھاگے];
    uint32_t          quant_ib_آخری[WORK_قطار_MAX_N_دھاگے];
    uint32_t          quant_r[WORK_قطار_MAX_N_دھاگے];
    uint32_t          quant_c[WORK_قطار_MAX_N_دھاگے];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_صفیں_فی_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ثابت uint32_t * میٹرکس_صف_counts;
    ثابت ساخت mmid_صف_mapping * میٹرکس_صفیں;
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
    uint32_t vtcm_src0_سائز_فی_thread;
    uint32_t vtcm_act_سائز_فی_thread;
    uint32_t vtcm_dst_سائز_فی_thread;
};

ساکن صحیح htp_mm_init_context(
    ساخت htp_ops_context * octx,
    ثابت ساخت htp_mm_kernel_params * kparams
) {
    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, (uint32_t) kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    اگر (kparams->n_hmx) {
        اگر (kparams->n_act_دھاگے <= 0 || kparams->n_act_دھاگے > (int32_t) octx->n_دھاگے) {
            واپس HTP_حالت_INVAL_PARAMS;
        }
    }

    واپس HTP_حالت_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
ساکن ثابت uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
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
ساکن ثابت uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

ساکن ثابت uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#تعریف_کرو htp_matmul_tensors_preamble                                 \
    ثابت ساخت htp_tensor * محدود src0 = octx->src[0];         \
    ثابت ساخت htp_tensor * محدود src1 = octx->src[1];         \
    ثابت ساخت htp_tensor * محدود src2 = octx->src[2];         \
    ثابت ساخت htp_tensor * محدود  dst = octx->dst;            \
    ثابت ساخت htp_tensor * محدود  act = src1;                 \
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
    ساخت htp_mm_context * mmctx  = ڈیٹا;                          \
    ساخت htp_ops_context * octx  = mmctx->octx;                   \
    dma_قطار * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_فی_thread = mmctx->src0_nrows_فی_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#شامل "hvx-mm-kernels-tiled.h"
#شامل "hvx-mm-kernels-float.h"
#شامل "hmx-mm-kernels-tiled.h"

ساکن خالی hvx_mm_2d_tiled(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_انتہا - mmctx->src0_صف_آغاز;
    ثابت uint32_t act_nrows = mmctx->cur_m_صفیں ? mmctx->cur_m_صفیں : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_آغاز = mmctx->cur_m_آغاز;

    ثابت uint32_t src0_آغاز_صف  = mmctx->src0_صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف    = MIN(src0_آغاز_صف + src0_nrows_فی_thread, mmctx->src0_صف_انتہا);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_صف_سائز  = nb1;
    ثابت size_t act_صف_سائز = nb11;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;
    ثابت size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * محدود vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_سائز_فی_thread  * ith;
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * محدود act_ڈیٹا = mmctx->vtcm_act;

    ثابت dma_addr_t src0_صف = src0->ڈیٹا;

    ثابت uint32_t tile_سائز = htp_tensor_get_tile_سائز(src0->type);
    ثابت uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_صف_stride = n_k_tiles_w * tile_سائز;
    uint32_t tile_صف_transfer_سائز_aligned = n_k_tiles_w * aligned_tile_سائز;

    uint32_t ct_آغاز = src0_آغاز_صف / 32;
    uint32_t ct_انتہا   = (src0_انتہا_صف + 31) / 32;

    uint32_t ڈالو_ct = ct_آغاز;
    اگر (src0_آغاز_صف < src0_انتہا_صف) {
        برائے (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_انتہا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + d * tile_صف_transfer_سائز_aligned,
                           src0_صف + ڈالو_ct * tile_صف_stride), aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
        }
    }

    اگر (src0_آغاز_صف >= src0_انتہا_صف) {
        واپس;
    }

    برائے (uint32_t ct = ct_آغاز; ct < ct_انتہا; ct++) {
        ثابت uint8_t * w_tile = (خالی *) dma_قطار_نکالو(dma_q).dst;

        صحیح valid_صفیں = (صحیح)ne0 - (صحیح)(ct * 32);
        valid_صفیں = MIN(32, MAX(0, valid_صفیں));

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        برائے (; ir1 + 1 < act_nrows; ir1 += 2) {
            ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_ڈیٹا + (ir1+0) * act_stride);
            ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_ڈیٹا + (ir1+1) * act_stride);
            اعشاری * محدود dst_row0 = (اعشاری *) (dst->ڈیٹا + ((cur_m_آغاز + ir1+0) * dst_صف_سائز));
            اعشاری * محدود dst_row1 = (اعشاری *) (dst->ڈیٹا + ((cur_m_آغاز + ir1+1) * dst_صف_سائز));

            اعشاری * dst_ptr0 = &dst_row0[ct * 32];
            اعشاری * dst_ptr1 = &dst_row1[ct * 32];

            ثابت اعشاری * src2_ptr0 = تہی;
            ثابت اعشاری * src2_ptr1 = تہی;
            اگر (src2) {
                ثابت اعشاری * محدود src2_row0 = (ثابت اعشاری *) ((ثابت uint8_t *) src2->ڈیٹا + ((cur_m_آغاز + ir1+0) * src2_stride));
                ثابت اعشاری * محدود src2_row1 = (ثابت اعشاری *) ((ثابت uint8_t *) src2->ڈیٹا + ((cur_m_آغاز + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_صفیں, src2_ptr0, src2_ptr1);
        }

        برائے (; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_ڈیٹا + ir1 * act_stride);
            اعشاری * محدود dst_صف          = (اعشاری *) (dst->ڈیٹا + ((cur_m_آغاز + ir1) * dst_صف_سائز));
            اعشاری * dst_ptr = &dst_صف[ct * 32];

            ثابت اعشاری * src2_ptr = تہی;
            اگر (src2) {
                ثابت اعشاری * محدود src2_صف = (ثابت اعشاری *) ((ثابت uint8_t *) src2->ڈیٹا + ((cur_m_آغاز + ir1) * src2_stride));
                src2_ptr = &src2_صف[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_صفیں, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        اگر (ڈالو_ct < ct_انتہا) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                           aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
            ڈالو_ct++;
        }
    }
}

ساکن خالی hvx_mv_2d_tiled(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_انتہا - mmctx->src0_صف_آغاز;

    ثابت uint32_t src0_آغاز_صف  = mmctx->src0_صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف    = MIN(src0_آغاز_صف + src0_nrows_فی_thread, mmctx->src0_صف_انتہا);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_صف_سائز  = nb1;
    ثابت size_t act_صف_سائز = nb11;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_سائز_فی_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * act_ڈیٹا = mmctx->vtcm_act;

    اعشاری * tmp = (اعشاری *) vtcm_dst_ptr;

    ثابت dma_addr_t src0_صف = src0->ڈیٹا;

    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_ڈیٹا;
    اعشاری * محدود dst_ستون          = (اعشاری *) dst->ڈیٹا;

    ثابت uint32_t tile_سائز = htp_tensor_get_tile_سائز(src0->type);
    ثابت uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_صف_stride = n_k_tiles_w * tile_سائز;
    uint32_t tile_صف_transfer_سائز_aligned = n_k_tiles_w * aligned_tile_سائز;

    uint32_t ct_آغاز = src0_آغاز_صف / 32;
    uint32_t ct_انتہا   = (src0_انتہا_صف + 31) / 32;

    uint32_t ڈالو_ct = ct_آغاز;
    اگر (src0_آغاز_صف < src0_انتہا_صف) {
        اگر (src2) {
            اعشاری * vtcm_bias_ptr = (اعشاری *) mmctx->vtcm_bias + src0_آغاز_صف;
            ثابت dma_addr_t src2_addr = src2->ڈیٹا + src0_آغاز_صف * حجم(اعشاری);
            صحیح قاش_سائز = (صحیح)MIN(src0_انتہا_صف, ne0) - (صحیح)src0_آغاز_صف;
            اگر (قاش_سائز > 0) {
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_bias_ptr, src2_addr),
                               قاش_سائز * حجم(اعشاری), قاش_سائز * حجم(اعشاری), قاش_سائز * حجم(اعشاری), 1);
                dma_قطار_نکالو_nowait(dma_q);
            }
        }
        برائے (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_انتہا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + d * tile_صف_transfer_سائز_aligned,
                           src0_صف + ڈالو_ct * tile_صف_stride), aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
        }
    }

    اگر (src0_آغاز_صف >= src0_انتہا_صف) {
        واپس;
    }

    برائے (uint32_t ct = ct_آغاز; ct < ct_انتہا; ct++) {
        ثابت uint8_t * w_tile = (خالی *) dma_قطار_نکالو(dma_q).dst;

        اعشاری * dst_ptr = &tmp[ct * 32 - src0_آغاز_صف];
        صحیح valid_صفیں = (صحیح)ne0 - (صحیح)(ct * 32);
        valid_صفیں = MIN(32, MAX(0, valid_صفیں));

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_صفیں, تہی);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        اگر (ڈالو_ct < ct_انتہا) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                           aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
            ڈالو_ct++;
        }
    }

    صحیح copy_cnt = (صحیح)MIN(src0_انتہا_صف, ne0) - (صحیح)src0_آغاز_صف;
    اگر (copy_cnt > 0) {
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ct_انتہا);
        اگر (src2) {
            hvx_جوڑو_f32_uuu((uint8_t *) &dst_ستون[src0_آغاز_صف],
                            (ثابت uint8_t *) tmp,
                            (ثابت uint8_t *) ((ثابت اعشاری *) mmctx->vtcm_bias + src0_آغاز_صف),
                            copy_cnt);
        } ورنہ {
            hvx_copy_f32_ua((uint8_t *) &dst_ستون[src0_آغاز_صف], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_انتہا);
    }
}

ساکن خالی hvx_mm_nx_2d_tiled(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_mm_context * mmctx = ڈیٹا;
    ساخت htp_ops_context * octx = mmctx->octx;
    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;

    ثابت ساخت htp_tensor * محدود act = octx->src[n_weights]; /* x */
    ثابت uint32_t ne10 = act->ne[0];
    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_وزن_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * محدود act_ڈیٹا        = mmctx->vtcm_act;

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    برائے (uint32_t widx = 0; widx < n_weights; widx++) {
        ثابت ساخت htp_tensor * محدود src_w = octx->src[widx];
        ثابت ساخت htp_tensor * محدود dst   = octx->dsts[widx];
        اگر (!src_w || !dst) جاری;
        dma_قطار * dma_q = octx->ctx->dma[ith];

        ثابت uint32_t tile_سائز = htp_tensor_get_tile_سائز(src_w->type);
        ثابت uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(src_w->type);
        ثابت uint32_t ne00 = src_w->ne[0];
        ثابت uint32_t ne01 = src_w->ne[1];
        ثابت size_t dst_صف_سائز = dst->nb[1];
        ثابت dma_addr_t src_w_صف = src_w->ڈیٹا;

        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_سائز;
        ثابت uint32_t tile_صف_transfer_سائز_aligned = n_k_tiles_w * aligned_tile_سائز;

        uint32_t src0_آغاز_صف = 0;
        uint32_t src0_انتہا_صف   = ne01;
        اگر (octx->ctx->mdev.گنتی > 1) {
            ثابت bool can_بانٹو = htp_tensor_can_صف_تقسیم(dst, حجم(اعشاری));
            ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(ne01, can_بانٹو ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            src0_آغاز_صف = range.آغاز;
            src0_انتہا_صف   = range.آغاز + range.گنتی;
        }

        ثابت uint32_t nrows = src0_انتہا_صف - src0_آغاز_صف;
        uint32_t src0_nrows_فی_thread = fastdiv(nrows + nth - 1, &octx->n_دھاگے_div);
        src0_nrows_فی_thread = hex_باری_اوپر(src0_nrows_فی_thread, 32);

        ثابت uint32_t آغاز_صف = src0_آغاز_صف + src0_nrows_فی_thread * ith;
        ثابت uint32_t انتہا_صف   = MIN(آغاز_صف + src0_nrows_فی_thread, src0_انتہا_صف);
        اگر (آغاز_صف >= انتہا_صف) جاری;

        uint32_t ct_آغاز = آغاز_صف / 32;
        uint32_t ct_انتہا   = (انتہا_صف + 31) / 32;

        uint32_t ڈالو_ct = ct_آغاز;
        برائے (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_انتہا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_وزن_ptr + d * tile_صف_transfer_سائز_aligned,
                            src_w_صف + ڈالو_ct * tile_صف_stride), aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
        }

        برائے (uint32_t ct = ct_آغاز; ct < ct_انتہا; ct++) {
            ثابت uint8_t * w_tile = (خالی *) dma_قطار_نکالو(dma_q).dst;
            صحیح valid_صفیں = (صحیح)dst->ne[0] - (صحیح)(ct * 32);
            valid_صفیں = MIN(32, MAX(0, valid_صفیں));

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            برائے (; ir1 + 1 < act_nrows; ir1 += 2) {
                ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_ڈیٹا + (ir1+0) * act_stride);
                ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_ڈیٹا + (ir1+1) * act_stride);

                اعشاری * محدود dst_row0 = (اعشاری *) (dst->ڈیٹا + ((ir1+0) * dst_صف_سائز));
                اعشاری * محدود dst_row1 = (اعشاری *) (dst->ڈیٹا + ((ir1+1) * dst_صف_سائز));
                اعشاری * dst_ptr0 = &dst_row0[ct * 32];
                اعشاری * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_صفیں, تہی, تہی);
            }

            برائے (; ir1 < act_nrows; ++ir1) {
                ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_ڈیٹا + ir1 * act_stride);
                اعشاری * محدود dst_صف = (اعشاری *) (dst->ڈیٹا + (ir1 * dst_صف_سائز));
                اعشاری * dst_ptr = &dst_صف[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_صفیں, تہی);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            اگر (ڈالو_ct < ct_انتہا) {
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(w_tile, src_w_صف + ڈالو_ct * tile_صف_stride),
                                aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
                ڈالو_ct++;
            }
        }
    }
}

ساکن خالی hvx_mm_transfer_act_dma(
    ساخت htp_ops_context * octx,
    ثابت ساخت htp_mm_kernel_params * kparams,
    ثابت ساخت htp_tensor * act,
    uint8_t * dst_بنیاد,
    size_t dst_صف_سائز,
    uint32_t m_آغاز,
    uint32_t m_صفیں
) {
    اگر (m_صفیں == 0) {
        واپس;
    }

    dma_قطار * dma_q = octx->ctx->dma[0];
    ثابت uint32_t ne0 = act->ne[0];
    ثابت size_t elem_سائز = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(اعشاری);
    ثابت size_t صف_بائٹیں = ne0 * elem_سائز;
    ثابت size_t act_nb1 = act->nb[1];
    ثابت dma_addr_t act_بنیاد = act->ڈیٹا;

    ثابت bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    اگر (is_contiguous) {
        ثابت dma_addr_t src_addr = act_بنیاد + m_آغاز * act_nb1;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_بنیاد, src_addr),
                       dst_صف_سائز, act_nb1, صف_بائٹیں, m_صفیں);
        dma_قطار_نکالو(dma_q);
    } ورنہ {
        ثابت uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ثابت bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        برائے (uint32_t ir = 0; ir < m_صفیں; ++ir) {
            ثابت uint32_t ir1 = m_آغاز + ir;
            uint32_t i11, i12, i13;
            اگر (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                ثابت uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } ورنہ {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                ثابت uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            ثابت dma_addr_t صف_src = act_بنیاد + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * صف_dst = dst_بنیاد + ir * dst_صف_سائز;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(صف_dst, صف_src),
                           dst_صف_سائز, act_nb1, صف_بائٹیں, 1);
            dma_قطار_نکالو(dma_q);
        }
    }

    اگر (dst_صف_سائز > صف_بائٹیں) {
        ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_آغاز);
        ثابت uint32_t pad_elems = (dst_صف_سائز - صف_بائٹیں) / elem_سائز;
        اگر (elem_سائز == حجم(اعشاری)) {
            برائے (uint32_t ir = 0; ir < m_صفیں; ++ir) {
                hvx_splat_f32_u(dst_بنیاد + ir * dst_صف_سائز + صف_بائٹیں, 0.0f, pad_elems);
            }
        } ورنہ {
            برائے (uint32_t ir = 0; ir < m_صفیں; ++ir) {
                hvx_splat_f16_u(dst_بنیاد + ir * dst_صف_سائز + صف_بائٹیں, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_آغاز);
    }
}

#تعریف_کرو QUANTIZE_IMPL(name, log_name, kernel_fn, dst_صف_سائز_اظہاریہ)                                        \
ساکن خالی name(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {                                        \
    (خالی) nth;                                                                                            \
    ساخت htp_mm_context * mmctx = ڈیٹا;                                                                  \
    ساخت htp_ops_context * octx = mmctx->octx;                                                           \
    ثابت ساخت htp_tensor * src = mmctx->act;                                                            \
    ثابت uint32_t ne0 = src->ne[0];                                                                       \
    ثابت uint32_t nrows = mmctx->cur_m_صفیں ? mmctx->cur_m_صفیں : mmctx->act_nrows;                       \
    ثابت uint32_t nrows_فی_thread = mmctx->n_quant_صفیں_فی_thread;                                      \
                                                                                                           \
    ثابت uint32_t ir_پہلا = nrows_فی_thread * ith;                                                      \
    اگر (ir_پہلا >= nrows) {                                                                               \
        واپس;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_پہلا);                                        \
                                                                                                           \
    uint8_t * محدود dst = mmctx->vtcm_act;                                                              \
    ثابت uint32_t ir_آخری = MIN(ir_پہلا + nrows_فی_thread, nrows);                                      \
    ثابت size_t raw_صف_سائز = mmctx->vtcm_act_raw_stride;                                                \
    ثابت size_t dst_صف_سائز = (dst_صف_سائز_اظہاریہ);                                                       \
                                                                                                           \
    ثابت uint8_t * محدود src_ڈیٹا = (ثابت uint8_t *) mmctx->vtcm_act_raw + (raw_صف_سائز * ir_پہلا); \
    uint8_t * محدود dst_ڈیٹا = (uint8_t *) dst + (dst_صف_سائز * ir_پہلا);                             \
    kernel_fn(src_ڈیٹا, dst_ڈیٹا, تہی, ne0, ir_آخری - ir_پہلا, raw_صف_سائز, dst_صف_سائز);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_پہلا);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled,     "quantize-f32-q8_0_tiled",     quantize_f32_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_صف_سائز(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled,     "quantize-f32-q8_1_tiled",     quantize_f32_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_صف_سائز(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_صف_سائز(ne0))
QUANTIZE_IMPL(quantize_f32_f16_tiled,      "quantize-f32-f16_tiled",      quantize_f32_f16_tiled_kernel,      htp_mm_f16_tiled_صف_سائز(ne0))
QUANTIZE_IMPL(quantize_f16_f16_tiled,      "quantize-f16-f16_tiled",      quantize_f16_f16_tiled_kernel,      htp_mm_f16_tiled_صف_سائز(ne0))
QUANTIZE_IMPL(quantize_f32_f32,            "quantize-f32-f32",            quantize_f32_f32_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,            "quantize-f32-f16",            quantize_f32_f16_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,            "quantize-f16-f16",            quantize_f16_f16_kernel,            mmctx->vtcm_act_stride)

ساکن خالی quantize_f32_q8_0_tiled_بلاک(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    (خالی) nth;
    ساخت htp_mm_context * mmctx = ڈیٹا;
    اگر (mmctx->quant_ib_پہلا[ith] >= mmctx->quant_ib_آخری[ith]) {
        واپس;
    }
    ساخت htp_ops_context * octx = mmctx->octx;
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);

    ثابت ساخت htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_بلاک_kernel(
        (ثابت اعشاری *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        تہی,
        src->ne[0],
        mmctx->quant_ib_پہلا[ith],
        mmctx->quant_ib_آخری[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_صف_سائز(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);
}

ساکن خالی quantize_f32_q8_1_tiled_بلاک(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    (خالی) nth;
    ساخت htp_mm_context * mmctx = ڈیٹا;
    اگر (mmctx->quant_ib_پہلا[ith] >= mmctx->quant_ib_آخری[ith]) {
        واپس;
    }
    ساخت htp_ops_context * octx = mmctx->octx;
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);

    ثابت ساخت htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_بلاک_kernel(
        (ثابت اعشاری *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        تہی,
        src->ne[0],
        mmctx->quant_ib_پہلا[ith],
        mmctx->quant_ib_آخری[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_صف_سائز(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);
}

ساکن خالی quantize_f32_q8_1_s16_tiled_بلاک(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    (خالی) nth;
    ساخت htp_mm_context * mmctx = ڈیٹا;
    اگر (mmctx->quant_ib_پہلا[ith] >= mmctx->quant_ib_آخری[ith]) {
        واپس;
    }
    ساخت htp_ops_context * octx = mmctx->octx;
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);

    ثابت ساخت htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_بلاک_kernel(
        (ثابت اعشاری *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        تہی,
        src->ne[0],
        mmctx->quant_ib_پہلا[ith],
        mmctx->quant_ib_آخری[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_صف_سائز(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);
}

ساکن خالی quantize_f32_f16_tiled_بلاک(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    (خالی) nth;
    ساخت htp_mm_context * mmctx = ڈیٹا;
    اگر (mmctx->quant_ib_پہلا[ith] >= mmctx->quant_ib_آخری[ith]) {
        واپس;
    }
    ساخت htp_ops_context * octx = mmctx->octx;
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);

    ثابت ساخت htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_بلاک_kernel(
        (ثابت اعشاری *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        تہی,
        src->ne[0],
        mmctx->quant_ib_پہلا[ith],
        mmctx->quant_ib_آخری[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_صف_سائز(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);
}

ساکن خالی quantize_f16_f16_tiled_بلاک(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    (خالی) nth;
    ساخت htp_mm_context * mmctx = ڈیٹا;
    اگر (mmctx->quant_ib_پہلا[ith] >= mmctx->quant_ib_آخری[ith]) {
        واپس;
    }
    ساخت htp_ops_context * octx = mmctx->octx;
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);

    ثابت ساخت htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_بلاک_kernel(
        (ثابت __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        تہی,
        src->ne[0],
        mmctx->quant_ib_پہلا[ith],
        mmctx->quant_ib_آخری[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_صف_سائز(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_پہلا[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
ساکن اندرونی کارکن_callback_t htp_mm_act_quant_صف_func(صحیح وزن_type, صحیح act_type) {
    اگر (وزن_type == HTP_TYPE_F16) {
        واپس (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    اگر (وزن_type == HTP_TYPE_Q2_K) {
        واپس quantize_f32_q8_1_s16_tiled;
    }
    واپس htp_mm_وزن_has_انحراف(وزن_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

ساکن اندرونی کارکن_callback_t htp_mm_act_quant_بلاک_func(صحیح وزن_type, صحیح act_type) {
    اگر (وزن_type == HTP_TYPE_F16) {
        واپس (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_بلاک : quantize_f32_f16_tiled_بلاک;
    }
    اگر (وزن_type == HTP_TYPE_Q2_K) {
        واپس quantize_f32_q8_1_s16_tiled_بلاک;
    }
    واپس htp_mm_وزن_has_انحراف(وزن_type) ? quantize_f32_q8_1_tiled_بلاک : quantize_f32_q8_0_tiled_بلاک;
}

ساکن خالی hvx_mm_4d_tiled(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_انتہا - mmctx->src0_صف_آغاز;
    ثابت uint32_t cur_m_صفیں = mmctx->cur_m_صفیں ? mmctx->cur_m_صفیں : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_آغاز = mmctx->cur_m_آغاز;

    ثابت uint32_t src0_آغاز_صف  = mmctx->src0_صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف    = MIN(src0_آغاز_صف + src0_nrows_فی_thread, mmctx->src0_صف_انتہا);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_صف_سائز = nb1;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * محدود act_ڈیٹا = mmctx->vtcm_act;

    ثابت uint32_t tile_سائز = htp_tensor_get_tile_سائز(src0->type);
    ثابت uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(src0->type);

    ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_سائز;
    ثابت uint32_t tile_صف_transfer_سائز_aligned = n_k_tiles_w * aligned_tile_سائز;
    ثابت uint32_t src0_قاش_stride = ((ne01 + 31) / 32) * tile_صف_stride;

    ثابت uint32_t ct_آغاز = src0_آغاز_صف / 32;
    ثابت uint32_t ct_انتہا   = (src0_انتہا_صف + 31) / 32;

    اگر (src0_آغاز_صف >= src0_انتہا_صف || cur_m_صفیں == 0) {
        واپس;
    }

    ثابت uint32_t کل_batches = ne12 * ne13;
    ثابت uint32_t b_آغاز = fastdiv(cur_m_آغاز, &kparams->div_ne1);
    uint32_t b_انتہا = fastdiv(cur_m_آغاز + cur_m_صفیں + ne11 - 1, &kparams->div_ne1);
    b_انتہا = MIN(b_انتہا, کل_batches);

    uint32_t b_grp_آغاز = b_آغاز;
    جب (b_grp_آغاز < b_انتہا) {
        ثابت uint32_t b3 = fastdiv(b_grp_آغاز, &kparams->div_ne12);
        ثابت uint32_t b2 = b_grp_آغاز - b3 * ne12;
        ثابت uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ثابت uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_انتہا = b_grp_آغاز + 1;
        جب (b_grp_انتہا < b_انتہا) {
            ثابت uint32_t cur_b3 = fastdiv(b_grp_انتہا, &kparams->div_ne12);
            ثابت uint32_t cur_b2 = b_grp_انتہا - cur_b3 * ne12;
            ثابت uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ثابت uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            اگر (cur_i02 != i02 || cur_i03 != i03) {
                توڑو;
            }
            b_grp_انتہا++;
        }

        ثابت uint32_t قاش_idx = i03 * ne02 + i02;
        ثابت dma_addr_t src0_قاش = src0->ڈیٹا + (size_t) قاش_idx * src0_قاش_stride;

        uint32_t ڈالو_ct = ct_آغاز;
        برائے (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_انتہا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + d * tile_صف_transfer_سائز_aligned,
                           src0_قاش + (size_t) ڈالو_ct * tile_صف_stride),
                           aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
        }

        برائے (uint32_t ct = ct_آغاز; ct < ct_انتہا; ct++) {
            ثابت uint8_t * w_tile = (خالی *) dma_قطار_نکالو(dma_q).dst;

            صحیح valid_صفیں = (صحیح)ne0 - (صحیح)(ct * 32);
            valid_صفیں = MIN(32, MAX(0, valid_صفیں));

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            برائے (uint32_t b = b_grp_آغاز; b < b_grp_انتہا; b++) {
                ثابت uint32_t b_m_آغاز = b * ne11;
                ثابت uint32_t m_پہلا = MAX(cur_m_آغاز, b_m_آغاز);
                ثابت uint32_t m_آخری  = MIN(cur_m_آغاز + cur_m_صفیں, b_m_آغاز + ne11);
                اگر (m_پہلا >= m_آخری) جاری;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_بنیاد = (uint8_t *) dst->ڈیٹا + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_انحراف = m_پہلا - cur_m_آغاز;
                ثابت uint32_t dst_m_انحراف   = m_پہلا - b_m_آغاز;
                ثابت uint32_t batch_nrows    = m_آخری - m_پہلا;

                uint32_t ir1 = 0;
                برائے (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_ڈیٹا + (chunk_m_انحراف + ir1 + 0) * act_stride);
                    ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_ڈیٹا + (chunk_m_انحراف + ir1 + 1) * act_stride);
                    اعشاری * محدود dst_row0 = (اعشاری *) (dst_batch_بنیاد + (dst_m_انحراف + ir1 + 0) * dst_صف_سائز);
                    اعشاری * محدود dst_row1 = (اعشاری *) (dst_batch_بنیاد + (dst_m_انحراف + ir1 + 1) * dst_صف_سائز);
                    اعشاری * dst_ptr0 = &dst_row0[ct * 32];
                    اعشاری * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_صفیں, تہی, تہی);
                }
                برائے (; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_ڈیٹا + (chunk_m_انحراف + ir1) * act_stride);
                    اعشاری * محدود dst_صف = (اعشاری *) (dst_batch_بنیاد + (dst_m_انحراف + ir1) * dst_صف_سائز);
                    اعشاری * dst_ptr = &dst_صف[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_ستون, valid_صفیں, تہی);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            اگر (ڈالو_ct < ct_انتہا) {
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(w_tile, src0_قاش + (size_t) ڈالو_ct * tile_صف_stride),
                               aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
                ڈالو_ct++;
            }
        }
        b_grp_آغاز = b_grp_انتہا;
    }
    اگر (src2) {
        hvx_tensor_جوڑو_f32_grid(dst, src2, cur_m_آغاز, cur_m_آغاز + cur_m_صفیں, src0_آغاز_صف, src0_انتہا_صف, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ساکن خالی hvx_mm_2d_flat(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_matmul_preamble;

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_نقاب = n_prefetch - 1;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_انتہا - mmctx->src0_صف_آغاز;  // src0 rows
    ثابت uint32_t act_nrows = mmctx->cur_m_صفیں ? mmctx->cur_m_صفیں : mmctx->act_nrows;
    ثابت uint32_t cur_m_آغاز = mmctx->cur_m_آغاز;

    ثابت uint32_t src0_آغاز_صف  = mmctx->src0_صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف    = MIN(src0_آغاز_صف + src0_nrows_فی_thread, mmctx->src0_صف_انتہا);
    ثابت uint32_t src0_انتہا_صف_x2 = src0_آغاز_صف + ((src0_انتہا_صف - src0_آغاز_صف) & ~1U);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_صف_سائز  = nb1;
    ثابت size_t src0_صف_سائز = nb01;
    ثابت size_t act_صف_سائز = nb11;

    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_سائز_فی_thread  * ith;
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * محدود act_ڈیٹا      = mmctx->vtcm_act;

    ثابت dma_addr_t src0_صف = src0->ڈیٹا;


    // Prefill vtcm with src0 rows
    اگر (src0_آغاز_صف < src0_انتہا_صف) {
        برائے (uint32_t ir0 = src0_آغاز_صف; ir0 < src0_انتہا_صف_x2; ir0 += 2) {
            ثابت صحیح is0 = (ir0 - src0_آغاز_صف);
            اگر (is0 >= (صحیح)n_prefetch) {
                توڑو;
            }
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + is0 * src0_stride, src0_صف + ir0 * src0_صف_سائز),
                           src0_stride, src0_صف_سائز, src0_صف_سائز, 2);
        }
    }

    اگر (src0_آغاز_صف >= src0_انتہا_صف) {
        واپس;
    }

    // Process src0 rows
    برائے (uint32_t ir0 = src0_آغاز_صف; ir0 < src0_انتہا_صف_x2; ir0 += 2) {
        ثابت uint8_t * ss0 = (خالی *) dma_قطار_نکالو(dma_q).dst;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        برائے (; ir1 + 1 < act_nrows; ir1 += 2) {
            ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_ڈیٹا + (ir1+0) * act_stride);
            ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_ڈیٹا + (ir1+1) * act_stride);
            اعشاری * محدود dst_row0 = (اعشاری *) (dst->ڈیٹا + ((cur_m_آغاز + ir1+0) * dst_صف_سائز));
            اعشاری * محدود dst_row1 = (اعشاری *) (dst->ڈیٹا + ((cur_m_آغاز + ir1+1) * dst_صف_سائز));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        برائے (; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_ڈیٹا + ir1 * act_stride);
            اعشاری * محدود dst_صف          = (اعشاری *) (dst->ڈیٹا + ((cur_m_آغاز + ir1) * dst_صف_سائز));
            mmctx->vec_dot_2x1(ne00, &dst_صف[ir0], ss0, ss0 + src0_stride, act_ستون);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ثابت صحیح pr0 = (ir0 + n_prefetch);
        ثابت صحیح is0 = (pr0 - src0_آغاز_صف) & prefetch_نقاب;
        اگر (pr0 < src0_انتہا_صف_x2) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + is0 * src0_stride, src0_صف + pr0 * src0_صف_سائز),
                           src0_stride, src0_صف_سائز, src0_صف_سائز, 2);
        }
    }

    // Process the last row (if any)
    اگر (src0_انتہا_صف != src0_انتہا_صف_x2) {
        uint32_t  ir0 = src0_انتہا_صف_x2;
        ثابت صحیح is0 = (ir0 - src0_آغاز_صف) & prefetch_نقاب;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + is0 * src0_stride, src0_صف + ir0 * src0_صف_سائز),
                       src0_stride, src0_صف_سائز, src0_صف_سائز, 1);
        ثابت uint8_t * ss0 = (خالی *) dma_قطار_نکالو(dma_q).dst;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #ہدایت unroll(2)
        برائے (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_ڈیٹا + ir1 * act_stride);
            اعشاری * محدود dst_صف          = (اعشاری *) (dst->ڈیٹا + ((cur_m_آغاز + ir1) * dst_صف_سائز));
            mmctx->vec_dot_1x1(ne00, &dst_صف[ir0], ss0, act_ستون);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    اگر (src2) {
        hvx_tensor_جوڑو_f32_grid(dst, src2, cur_m_آغاز, cur_m_آغاز + act_nrows, src0_آغاز_صف, src0_انتہا_صف, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ساکن خالی hvx_mv_2d_flat(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_انتہا - mmctx->src0_صف_آغاز;

    ثابت uint32_t src0_آغاز_صف  = mmctx->src0_صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف    = MIN(src0_آغاز_صف + src0_nrows_فی_thread, mmctx->src0_صف_انتہا);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_صف_سائز  = nb1;
    ثابت size_t src0_صف_سائز = nb01;
    ثابت size_t act_صف_سائز  = nb11;

    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_سائز_فی_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * act_ڈیٹا      = mmctx->vtcm_act;

    اعشاری * tmp = (اعشاری *) vtcm_dst_ptr;

    ثابت dma_addr_t src0_صف = src0->ڈیٹا;
    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_ڈیٹا;
    اعشاری * محدود dst_ستون         = (اعشاری *) dst->ڈیٹا;

    ثابت uint32_t src0_انتہا_صف_x2 = src0_آغاز_صف + ((src0_انتہا_صف - src0_آغاز_صف) & ~1U);

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_نقاب = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    اگر (src0_آغاز_صف < src0_انتہا_صف) {
        اگر (src2) {
            اعشاری * vtcm_bias_ptr = (اعشاری *) mmctx->vtcm_bias + src0_آغاز_صف;
            ثابت dma_addr_t src2_addr = src2->ڈیٹا + src0_آغاز_صف * حجم(اعشاری);
            صحیح قاش_سائز = (صحیح)src0_انتہا_صف - (صحیح)src0_آغاز_صف;
            اگر (قاش_سائز > 0) {
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_bias_ptr, src2_addr),
                               قاش_سائز * حجم(اعشاری), قاش_سائز * حجم(اعشاری), قاش_سائز * حجم(اعشاری), 1);
                dma_قطار_نکالو_nowait(dma_q);
            }
        }
        برائے (uint32_t ir0 = src0_آغاز_صف; ir0 < src0_انتہا_صف_x2; ir0 += 2) {
            ثابت uint32_t is0 = (ir0 - src0_آغاز_صف);
            اگر (is0 >= n_prefetch) {
                توڑو;
            }
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + is0 * src0_stride, src0_صف + ir0 * src0_صف_سائز),
                           src0_stride, src0_صف_سائز, src0_صف_سائز, 2);
        }
    }


    اگر (src0_آغاز_صف >= src0_انتہا_صف) {
        واپس;
    }

    // Process src0 rows
    برائے (uint32_t ir0 = src0_آغاز_صف; ir0 < src0_انتہا_صف_x2; ir0 += 2) {
        ثابت uint8_t * ss0 = (خالی *) dma_قطار_نکالو(dma_q).dst;
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_آغاز_صف], ss0, ss0 + src0_stride, act_ستون);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ثابت uint32_t pr0 = (ir0 + n_prefetch);
        ثابت uint32_t is0 = (pr0 - src0_آغاز_صف) & prefetch_نقاب;
        اگر (pr0 < src0_انتہا_صف_x2) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + is0 * src0_stride, src0_صف + pr0 * src0_صف_سائز),
                           src0_stride, src0_صف_سائز, src0_صف_سائز, 2);
        }
    }

    // Process the last row (if any)
    اگر (src0_انتہا_صف != src0_انتہا_صف_x2) {
        ثابت uint32_t ir0 = src0_انتہا_صف_x2;
        ثابت uint32_t is0 = (ir0 - src0_آغاز_صف) & prefetch_نقاب;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + is0 * src0_stride, src0_صف + ir0 * src0_صف_سائز),
                       src0_stride, src0_صف_سائز, src0_صف_سائز, 1);
        ثابت uint8_t * ss0 = (خالی *) dma_قطار_نکالو(dma_q).dst;
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_آغاز_صف], ss0, act_ستون);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    صحیح copy_cnt = src0_انتہا_صف - src0_آغاز_صف;
    اگر (copy_cnt > 0) {
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, src0_انتہا_صف);
        اگر (src2) {
            hvx_جوڑو_f32_uuu((uint8_t *) &dst_ستون[src0_آغاز_صف],
                            (ثابت uint8_t *) tmp,
                            (ثابت uint8_t *) ((ثابت اعشاری *) mmctx->vtcm_bias + src0_آغاز_صف),
                            copy_cnt);
        } ورنہ {
            hvx_copy_f32_ua((uint8_t *) &dst_ستون[src0_آغاز_صف], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_انتہا_صف);
    }
}

ساکن خالی hvx_mm_4d_flat(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_matmul_preamble;

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_نقاب = n_prefetch - 1;

    ثابت uint32_t src0_nrows = mmctx->src0_صف_انتہا - mmctx->src0_صف_آغاز;
    ثابت uint32_t cur_m_صفیں = mmctx->cur_m_صفیں ? mmctx->cur_m_صفیں : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_آغاز = mmctx->cur_m_آغاز;

    ثابت uint32_t src0_آغاز_صف  = mmctx->src0_صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف    = MIN(src0_آغاز_صف + src0_nrows_فی_thread, mmctx->src0_صف_انتہا);
    ثابت uint32_t src0_انتہا_صف_x2 = src0_آغاز_صف + ((src0_انتہا_صف - src0_آغاز_صف) & ~1U);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_صف_سائز  = nb1;
    ثابت size_t src0_صف_سائز = nb01;
    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * محدود act_ڈیٹا      = mmctx->vtcm_act;


    اگر (src0_آغاز_صف >= src0_انتہا_صف || cur_m_صفیں == 0) {
        واپس;
    }

    ثابت uint32_t کل_batches = ne12 * ne13;
    ثابت uint32_t b_آغاز = fastdiv(cur_m_آغاز, &kparams->div_ne1);
    uint32_t b_انتہا = fastdiv(cur_m_آغاز + cur_m_صفیں + ne11 - 1, &kparams->div_ne1);
    b_انتہا = MIN(b_انتہا, کل_batches);

    uint32_t b_grp_آغاز = b_آغاز;
    جب (b_grp_آغاز < b_انتہا) {
        ثابت uint32_t b3 = fastdiv(b_grp_آغاز, &kparams->div_ne12);
        ثابت uint32_t b2 = b_grp_آغاز - b3 * ne12;
        ثابت uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ثابت uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_انتہا = b_grp_آغاز + 1;
        جب (b_grp_انتہا < b_انتہا) {
            ثابت uint32_t cur_b3 = fastdiv(b_grp_انتہا, &kparams->div_ne12);
            ثابت uint32_t cur_b2 = b_grp_انتہا - cur_b3 * ne12;
            ثابت uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ثابت uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            اگر (cur_i02 != i02 || cur_i03 != i03) {
                توڑو;
            }
            b_grp_انتہا++;
        }

        ثابت dma_addr_t src0_صف = src0->ڈیٹا + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        برائے (uint32_t ir0 = src0_آغاز_صف; ir0 < src0_انتہا_صف_x2; ir0 += 2) {
            ثابت صحیح is0 = (ir0 - src0_آغاز_صف);
            اگر (is0 >= (صحیح)n_prefetch) {
                توڑو;
            }
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + is0 * src0_stride, src0_صف + (size_t) ir0 * src0_صف_سائز),
                           src0_stride, src0_صف_سائز, src0_صف_سائز, 2);
        }

        برائے (uint32_t ir0 = src0_آغاز_صف; ir0 < src0_انتہا_صف_x2; ir0 += 2) {
            ثابت uint8_t * ss0 = (خالی *) dma_قطار_نکالو(dma_q).dst;

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            برائے (uint32_t b = b_grp_آغاز; b < b_grp_انتہا; b++) {
                ثابت uint32_t b_m_آغاز = b * ne11;
                ثابت uint32_t m_پہلا = MAX(cur_m_آغاز, b_m_آغاز);
                ثابت uint32_t m_آخری  = MIN(cur_m_آغاز + cur_m_صفیں, b_m_آغاز + ne11);
                اگر (m_پہلا >= m_آخری) جاری;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_بنیاد = (uint8_t *) dst->ڈیٹا + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_انحراف = m_پہلا - cur_m_آغاز;
                ثابت uint32_t dst_m_انحراف   = m_پہلا - b_m_آغاز;
                ثابت uint32_t batch_nrows    = m_آخری - m_پہلا;

                uint32_t ir1 = 0;
                برائے (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ثابت uint8_t * محدود act_col0 = (ثابت uint8_t *) (act_ڈیٹا + (chunk_m_انحراف + ir1 + 0) * act_stride);
                    ثابت uint8_t * محدود act_col1 = (ثابت uint8_t *) (act_ڈیٹا + (chunk_m_انحراف + ir1 + 1) * act_stride);
                    اعشاری * محدود dst_row0 = (اعشاری *) (dst_batch_بنیاد + (dst_m_انحراف + ir1 + 0) * dst_صف_سائز);
                    اعشاری * محدود dst_row1 = (اعشاری *) (dst_batch_بنیاد + (dst_m_انحراف + ir1 + 1) * dst_صف_سائز);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                برائے (; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_ڈیٹا + (chunk_m_انحراف + ir1) * act_stride);
                    اعشاری * محدود dst_صف          = (اعشاری *) (dst_batch_بنیاد + (dst_m_انحراف + ir1) * dst_صف_سائز);
                    mmctx->vec_dot_2x1(ne00, &dst_صف[ir0], ss0, ss0 + src0_stride, act_ستون);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ثابت صحیح pr0 = (ir0 + n_prefetch);
            ثابت صحیح is0 = (pr0 - src0_آغاز_صف) & prefetch_نقاب;
            اگر (pr0 < src0_انتہا_صف_x2) {
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + is0 * src0_stride, src0_صف + (size_t) pr0 * src0_صف_سائز),
                               src0_stride, src0_صف_سائز, src0_صف_سائز, 2);
            }
        }

        اگر (src0_انتہا_صف != src0_انتہا_صف_x2) {
            uint32_t ir0 = src0_انتہا_صف_x2;
            ثابت صحیح is0 = (ir0 - src0_آغاز_صف) & prefetch_نقاب;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + is0 * src0_stride, src0_صف + (size_t) ir0 * src0_صف_سائز),
                           src0_stride, src0_صف_سائز, src0_صف_سائز, 1);
            ثابت uint8_t * ss0 = (خالی *) dma_قطار_نکالو(dma_q).dst;

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            برائے (uint32_t b = b_grp_آغاز; b < b_grp_انتہا; b++) {
                ثابت uint32_t b_m_آغاز = b * ne11;
                ثابت uint32_t m_پہلا = MAX(cur_m_آغاز, b_m_آغاز);
                ثابت uint32_t m_آخری  = MIN(cur_m_آغاز + cur_m_صفیں, b_m_آغاز + ne11);
                اگر (m_پہلا >= m_آخری) جاری;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_بنیاد = (uint8_t *) dst->ڈیٹا + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_انحراف = m_پہلا - cur_m_آغاز;
                ثابت uint32_t dst_m_انحراف   = m_پہلا - b_m_آغاز;
                ثابت uint32_t batch_nrows    = m_آخری - m_پہلا;

                برائے (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_ڈیٹا + (chunk_m_انحراف + ir1) * act_stride);
                    اعشاری * محدود dst_صف          = (اعشاری *) (dst_batch_بنیاد + (dst_m_انحراف + ir1) * dst_صف_سائز);
                    mmctx->vec_dot_1x1(ne00, &dst_صف[ir0], ss0, act_ستون);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_آغاز = b_grp_انتہا;
    }

    اگر (src2) {
        hvx_tensor_جوڑو_f32_grid(dst, src2, cur_m_آغاز, cur_m_آغاز + cur_m_صفیں, src0_آغاز_صف, src0_انتہا_صف, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#تعریف_کرو MMID_میٹرکس_صف(صف_id, i1) میٹرکس_صفیں[(صف_id) * mmctx->mapping_stride + (i1)]

ساکن خالی hvx_mm_id_tiled(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_matmul_preamble;

    ثابت ساخت htp_tensor * محدود ids = octx->src[2];

    ثابت uint32_t src0_nrows      = mmctx->src0_صف_انتہا - mmctx->src0_صف_آغاز;  // src0 rows per expert
    ثابت uint32_t act_nrows       = ne11;
    ثابت uint32_t src0_آغاز_صف  = mmctx->src0_صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف    = MIN(src0_آغاز_صف + src0_nrows_فی_thread, mmctx->src0_صف_انتہا);


    اگر (src0_آغاز_صف >= src0_انتہا_صف) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_ids = ids->ne[0];  // n_expert_used
    ثابت uint32_t n_as  = ne02;        // n_expert

    ثابت uint32_t *                میٹرکس_صف_counts = mmctx->میٹرکس_صف_counts;
    ثابت ساخت mmid_صف_mapping * میٹرکس_صفیں       = mmctx->میٹرکس_صفیں;

    ثابت size_t dst_صف_سائز  = nb1;
    ثابت size_t act_صف_سائز  = htp_mm_q8_0_tiled_صف_سائز(ne10);

    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * محدود act_ڈیٹا = mmctx->vtcm_act;

    برائے (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int32_t cne1 = میٹرکس_صف_counts[cur_a];
        اگر (cne1 == 0) {
            جاری;
        }

        ثابت dma_addr_t src0_صف = src0->ڈیٹا + cur_a * nb02;

        ثابت uint32_t tile_سائز = htp_tensor_get_tile_سائز(src0->type);
        ثابت uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(src0->type);
        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_سائز;
        ثابت uint32_t tile_صف_transfer_سائز_aligned = n_k_tiles_w * aligned_tile_سائز;

        ثابت uint32_t ct_آغاز = src0_آغاز_صف / 32;
        ثابت uint32_t ct_انتہا   = (src0_انتہا_صف + 31) / 32;

        uint32_t ڈالو_ct = ct_آغاز;
        برائے (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_انتہا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + d * tile_صف_transfer_سائز_aligned, src0_صف + ڈالو_ct * tile_صف_stride),
                           aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
        }

        برائے (uint32_t ct = ct_آغاز; ct < ct_انتہا; ct++) {
            ثابت uint8_t * w_tile = (خالی *) dma_قطار_نکالو(dma_q).dst;

            صحیح valid_صفیں = (صحیح)ne0 - (صحیح)(ct * 32);
            valid_صفیں = MIN(32, MAX(0, valid_صفیں));

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            برائے (uint32_t cid = 0; cid < cne1; ++cid) {
                ساخت mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, cid);
                ثابت صحیح               rm1         = صف_mapping.i1;  // expert idx
                ثابت صحیح               rm2         = صف_mapping.i2;  // token idx

                ثابت uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_ڈیٹا + (ir1 + rm2 * ne11 + 0) * act_stride);
                اعشاری * محدود dst_صف = (اعشاری *) (dst->ڈیٹا + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_صف[ct * 32], w_tile, act_ستون, valid_صفیں, تہی);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            اگر (ڈالو_ct < ct_انتہا) {
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                               aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
                ڈالو_ct++;
            }
        }
    }
}

ساکن خالی hvx_mv_id_tiled(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_matmul_preamble;

    ثابت ساخت htp_tensor * محدود ids = octx->src[2];

    ثابت uint32_t src0_nrows      = mmctx->src0_صف_انتہا - mmctx->src0_صف_آغاز;  // src0 rows per expert
    ثابت uint32_t src0_آغاز_صف  = mmctx->src0_صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف    = MIN(src0_آغاز_صف + src0_nrows_فی_thread, mmctx->src0_صف_انتہا);


    اگر (src0_آغاز_صف >= src0_انتہا_صف) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    أكد(ne13 % ne03 == 0);

    ثابت size_t dst_صف_سائز  = nb1;
    ثابت size_t act_صف_سائز = htp_mm_q8_0_tiled_صف_سائز(ne10);

    ثابت uint32_t n_aids = src2->ne[0];  // num activated experts
    ثابت uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * محدود act_ڈیٹا = mmctx->vtcm_act;

    برائے (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ثابت int32_t eid = *(ثابت int32_t *) ((ثابت uint8_t *) src2->ڈیٹا + ie1 * src2->nb[0]);
        اگر (eid < 0) {
            جاری;
        }
        أكد(eid < (int32_t) n_ids);

        ثابت dma_addr_t src0_صف = src0->ڈیٹا + eid * nb02;
        ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_ڈیٹا;
        اعشاری * محدود dst_صف          = (اعشاری *) (dst->ڈیٹا + ie1 * nb1);

        ثابت uint32_t tile_سائز = htp_tensor_get_tile_سائز(src0->type);
        ثابت uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(src0->type);
        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_سائز;
        ثابت uint32_t tile_صف_transfer_سائز_aligned = n_k_tiles_w * aligned_tile_سائز;

        ثابت uint32_t ct_آغاز = src0_آغاز_صف / 32;
        ثابت uint32_t ct_انتہا   = (src0_انتہا_صف + 31) / 32;

        uint32_t ڈالو_ct = ct_آغاز;
        برائے (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_انتہا; d++, ڈالو_ct++) {
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + d * tile_صف_transfer_سائز_aligned, src0_صف + ڈالو_ct * tile_صف_stride),
                           aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
        }

        برائے (uint32_t ct = ct_آغاز; ct < ct_انتہا; ct++) {
            ثابت uint8_t * w_tile = (خالی *) dma_قطار_نکالو(dma_q).dst;

            صحیح valid_صفیں = (صحیح)ne0 - (صحیح)(ct * 32);
            valid_صفیں = MIN(32, MAX(0, valid_صفیں));

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_صف[ct * 32], w_tile, act_ستون, valid_صفیں, تہی);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            اگر (ڈالو_ct < ct_انتہا) {
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                               aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
                ڈالو_ct++;
            }
        }
    }
}

ساکن خالی hvx_mv_id_nx_tiled(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_mm_context * mmctx = (ساخت htp_mm_context *) ڈیٹا;
    ساخت htp_ops_context * octx = mmctx->octx;
    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights      = kparams->n_weights;
    ثابت ساخت htp_tensor * محدود src0 = octx->src[0];
    ثابت ساخت htp_tensor * محدود act  = octx->src[n_weights];
    ثابت ساخت htp_tensor * محدود ids  = octx->src[n_weights + 1];


    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_aids = ids->ne[0];
    ثابت uint32_t n_ids  = src0->ne[2];

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * محدود act_ڈیٹا = mmctx->vtcm_act;

    برائے (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ثابت int32_t eid = *(ثابت int32_t *) ((ثابت uint8_t *) ids->ڈیٹا + ie1 * ids->nb[0]);
        اگر (eid < 0) جاری;
        أكد(eid < (int32_t) n_ids);

        برائے (uint32_t p = 0; p < n_weights; ++p) {
            ثابت ساخت htp_tensor * محدود src_w = octx->src[p];
            ثابت ساخت htp_tensor * محدود dst   = octx->dsts[p];
            اگر (!src_w || !dst) جاری;
            dma_قطار * dma_q = octx->ctx->dma[ith];

            ثابت uint32_t ne01 = src_w->ne[1];
            uint32_t آغاز_صف = 0;
            uint32_t انتہا_صف   = ne01;
            اگر (octx->ctx->mdev.گنتی > 1) {
                ثابت bool can_بانٹو = htp_tensor_can_صف_تقسیم(dst, حجم(اعشاری));
                ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(ne01, can_بانٹو ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
                آغاز_صف = range.آغاز;
                انتہا_صف   = range.آغاز + range.گنتی;
            }

            ثابت uint32_t nrows = انتہا_صف - آغاز_صف;
            uint32_t src0_nrows_فی_thread = fastdiv(nrows + nth - 1, &octx->n_دھاگے_div);
            src0_nrows_فی_thread = hex_باری_اوپر(src0_nrows_فی_thread, 32);

            ثابت uint32_t src0_آغاز_صف = آغاز_صف + src0_nrows_فی_thread * ith;
            ثابت uint32_t src0_انتہا_صف   = MIN(src0_آغاز_صف + src0_nrows_فی_thread, انتہا_صف);
            اگر (src0_آغاز_صف >= src0_انتہا_صف) جاری;

            ثابت dma_addr_t src0_صف = src_w->ڈیٹا + eid * src_w->nb[2];
            ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) act_ڈیٹا;
            اعشاری * محدود dst_صف = (اعشاری *) (dst->ڈیٹا + ie1 * dst->nb[1]);

            ثابت uint32_t tile_سائز = htp_tensor_get_tile_سائز(src_w->type);
            ثابت uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(src_w->type);
            ثابت uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_سائز;
            ثابت uint32_t tile_صف_transfer_سائز_aligned = n_k_tiles_w * aligned_tile_سائز;

            ثابت uint32_t ct_آغاز = src0_آغاز_صف / 32;
            ثابت uint32_t ct_انتہا   = (src0_انتہا_صف + 31) / 32;

            uint32_t ڈالو_ct = ct_آغاز;
            برائے (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_انتہا; d++, ڈالو_ct++) {
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + d * tile_صف_transfer_سائز_aligned, src0_صف + ڈالو_ct * tile_صف_stride),
                               aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
            }

            برائے (uint32_t ct = ct_آغاز; ct < ct_انتہا; ct++) {
                ثابت uint8_t * w_tile = (خالی *) dma_قطار_نکالو(dma_q).dst;

                صحیح valid_صفیں = (صحیح)dst->ne[0] - (صحیح)(ct * 32);
                valid_صفیں = MIN(32, MAX(0, valid_صفیں));

                htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_صف[ct * 32], w_tile, act_ستون, valid_صفیں, تہی);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                اگر (ڈالو_ct < ct_انتہا) {
                    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                                    aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
                    ڈالو_ct++;
                }
            }
        }
    }
}

ساکن خالی hvx_mm_id_nx_tiled(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_mm_context * mmctx = (ساخت htp_mm_context *) ڈیٹا;
    ساخت htp_ops_context * octx = mmctx->octx;
    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights      = kparams->n_weights;
    ثابت ساخت htp_tensor * محدود src0 = octx->src[0];
    ثابت ساخت htp_tensor * محدود act  = octx->src[n_weights];
    ثابت ساخت htp_tensor * محدود ids  = octx->src[n_weights + 1];


    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_as = src0->ne[2];

    ثابت uint32_t * میٹرکس_صف_counts = mmctx->میٹرکس_صف_counts;
    ثابت ساخت mmid_صف_mapping * میٹرکس_صفیں = mmctx->میٹرکس_صفیں;

    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * محدود vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_سائز_فی_thread * ith;
    uint8_t * محدود act_ڈیٹا = mmctx->vtcm_act;

    برائے (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int32_t cne1 = میٹرکس_صف_counts[cur_a];
        اگر (cne1 == 0) جاری;

        برائے (uint32_t p = 0; p < n_weights; ++p) {
            ثابت ساخت htp_tensor * محدود src_w = octx->src[p];
            ثابت ساخت htp_tensor * محدود dst   = octx->dsts[p];
            اگر (!src_w || !dst) جاری;
            dma_قطار * dma_q = octx->ctx->dma[ith];

            ثابت uint32_t ne01 = src_w->ne[1];
            uint32_t آغاز_صف = 0;
            uint32_t انتہا_صف   = ne01;
            اگر (octx->ctx->mdev.گنتی > 1) {
                ثابت bool can_بانٹو = htp_tensor_can_صف_تقسیم(dst, حجم(اعشاری));
                ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(ne01, can_بانٹو ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
                آغاز_صف = range.آغاز;
                انتہا_صف   = range.آغاز + range.گنتی;
            }

            ثابت uint32_t nrows = انتہا_صف - آغاز_صف;
            uint32_t src0_nrows_فی_thread = fastdiv(nrows + nth - 1, &octx->n_دھاگے_div);
            src0_nrows_فی_thread = hex_باری_اوپر(src0_nrows_فی_thread, 32);

            ثابت uint32_t src0_آغاز_صف = آغاز_صف + src0_nrows_فی_thread * ith;
            ثابت uint32_t src0_انتہا_صف   = MIN(src0_آغاز_صف + src0_nrows_فی_thread, انتہا_صف);
            اگر (src0_آغاز_صف >= src0_انتہا_صف) جاری;

            ثابت dma_addr_t src0_صف = src_w->ڈیٹا + cur_a * src_w->nb[2];

            ثابت uint32_t tile_سائز = htp_tensor_get_tile_سائز(src_w->type);
            ثابت uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(src_w->type);
            ثابت uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ثابت uint32_t tile_صف_stride = n_k_tiles_w * tile_سائز;
            ثابت uint32_t tile_صف_transfer_سائز_aligned = n_k_tiles_w * aligned_tile_سائز;

            ثابت uint32_t ct_آغاز = src0_آغاز_صف / 32;
            ثابت uint32_t ct_انتہا   = (src0_انتہا_صف + 31) / 32;

            uint32_t ڈالو_ct = ct_آغاز;
            برائے (uint32_t d = 0; d < n_prefetch && ڈالو_ct < ct_انتہا; d++, ڈالو_ct++) {
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src0_ptr + d * tile_صف_transfer_سائز_aligned, src0_صف + ڈالو_ct * tile_صف_stride),
                               aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
            }

            برائے (uint32_t ct = ct_آغاز; ct < ct_انتہا; ct++) {
                ثابت uint8_t * w_tile = (خالی *) dma_قطار_نکالو(dma_q).dst;

                صحیح valid_صفیں = (صحیح)dst->ne[0] - (صحیح)(ct * 32);
                valid_صفیں = MIN(32, MAX(0, valid_صفیں));

                htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                برائے (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    ساخت mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, cid);
                    ثابت صحیح rm1 = صف_mapping.i1;
                    ثابت صحیح rm2 = صف_mapping.i2;

                    ثابت uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ثابت uint8_t * محدود act_ستون = (ثابت uint8_t *) (act_ڈیٹا + (ir1 + rm2 * act->ne[1]) * act_stride);
                    اعشاری * محدود dst_صف = (اعشاری *) (dst->ڈیٹا + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_صف[ct * 32], w_tile, act_ستون, valid_صفیں, تہی);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                اگر (ڈالو_ct < ct_انتہا) {
                    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(w_tile, src0_صف + ڈالو_ct * tile_صف_stride),
                                    aligned_tile_سائز, tile_سائز, tile_سائز, n_k_tiles_w);
                    ڈالو_ct++;
                }
            }
        }
    }
}

ساکن صحیح hvx_mm_init_tiled_vec_dot(ساخت htp_mm_context * mmctx, شمار htp_ڈیٹا_type type) {
    انتخاب (type) {
        حالت HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            واپس 0;
        حالت HTP_TYPE_Q4_1:
        حالت HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            واپس 0;
        حالت HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            واپس 0;
        حالت HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            واپس 0;
        حالت HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            واپس 0;
        حالت HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            واپس 0;
        حالت HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            واپس 0;
        حالت HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            واپس 0;
        حالت HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            واپس 0;
        حالت HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            واپس 0;
        حالت HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            واپس 0;
        حالت HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            واپس 0;
        طےشدہ:
            واپس -1;
    }
}

ساکن صحیح hvx_mm_matmul(ساخت htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, 0);

    ساخت htp_mm_context mmctx_struct = {0};
    ساخت htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;

    ثابت uint32_t src0_nrows = ne01;
    ثابت uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_صف_آغاز = 0;
    uint32_t src0_صف_انتہا   = src0_nrows;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت bool can_بانٹو = htp_tensor_can_صف_تقسیم(dst, حجم(اعشاری));
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(src0_nrows, can_بانٹو ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        src0_صف_آغاز = range.آغاز;
        src0_صف_انتہا   = range.آغاز + range.گنتی;
    }

    اگر (src0_صف_آغاز >= src0_صف_انتہا) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t nrows = src0_صف_انتہا - src0_صف_آغاز;
    mmctx->src0_صف_آغاز = src0_صف_آغاز;
    mmctx->src0_صف_انتہا   = src0_صف_انتہا;

    ثابت bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_فی_thread  = fastdiv(nrows + octx->n_دھاگے - 1, &octx->n_دھاگے_div);
    اگر (is_tiled) {
        mmctx->src0_nrows_فی_thread = hex_باری_اوپر(mmctx->src0_nrows_فی_thread, 32);
    } ورنہ {
        mmctx->src0_nrows_فی_thread += (mmctx->src0_nrows_فی_thread & 1); // round up to even
    }

    ثابت size_t src0_صف_سائز = nb01;
    ثابت size_t dst_صف_سائز  = nb1;
    size_t       act_صف_سائز  = nb11;

    ثابت size_t src0_صف_سائز_padded = hex_باری_اوپر(src0_صف_سائز, 128);

    کارکن_callback_t quant_task_func;
    کارکن_callback_t matmul_کام_func;
    uint32_t n_quant_tasks = 1;
    ثابت bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    اگر (is_batched) {
        matmul_کام_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } ورنہ اگر (act_nrows > 1) {
        matmul_کام_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } ورنہ {
        matmul_کام_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    bool need_quant = true;

    انتخاب (kparams->kernel_type) {
        حالت HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            act_صف_سائز           = kparams->act_صف_سائز;
            توڑو;

        حالت HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = تہی;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_صف_سائز           = kparams->act_صف_سائز;
            توڑو;

        حالت HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = تہی;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_صف_سائز           = kparams->act_صف_سائز;
            توڑو;

        حالت HTP_MM_KERNEL_HVX_QUANT_بلاک:
        حالت HTP_MM_KERNEL_HVX_QUANT_صف:
        طےشدہ: {
            اگر (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                واپس HTP_حالت_NO_SUPPORT;
            }

            ثابت uint32_t qk = QK_Q8_0_TILED;
            ثابت uint32_t nb = (ne10 + qk - 1) / qk;
            ثابت uint32_t کل_nb = act_nrows * nb;

            اگر (act_nrows < octx->n_دھاگے && !is_batched) {
                n_quant_tasks = MIN(کل_nb, octx->n_دھاگے);
                quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
                برائے (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_پہلا = (کل_nb * ith) / n_quant_tasks;
                    uint32_t ib_آخری  = (کل_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_پہلا[ith] = ib_پہلا;
                    mmctx->quant_ib_آخری[ith]  = ib_آخری;
                    mmctx->quant_r[ith]        = ib_پہلا / nb;
                    mmctx->quant_c[ith]        = ib_پہلا % nb;
                }
            } ورنہ {
                n_quant_tasks = MIN(act_nrows, octx->n_دھاگے);
                quant_task_func = htp_mm_act_quant_صف_func(src0->type, act->type);
            }
            act_صف_سائز = kparams->act_صف_سائز;
            توڑو;
        }
    }

    ثابت uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ثابت uint32_t m_layout_صفیں = m_chunk;

    ساخت htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_صفیں, octx->n_دھاگے,
                                 dst_صف_سائز, src0_صف_سائز, act_صف_سائز, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    اگر (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
        mmctx->vtcm_act_سائز_فی_thread = L.act_بائٹیں;
    } ورنہ {
        mmctx->vtcm_act_سائز_فی_thread = fastdiv(L.act_بائٹیں, &octx->n_دھاگے_div);
    }

    mmctx->vtcm_src0_سائز_فی_thread = fastdiv(L.src0_بائٹیں, &octx->n_دھاگے_div);
    mmctx->vtcm_dst_سائز_فی_thread  = fastdiv(L.dst_بائٹیں, &octx->n_دھاگے_div);

    ثابت size_t vtcm_سائز = L.کل_بائٹیں;

    FARF(بلند_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_بائٹیں, L.act_بائٹیں, L.dst_بائٹیں, vtcm_سائز);

    FARF(بلند_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->ڈیٹا, act->ڈیٹا, dst->ڈیٹا);

    اگر (octx->ctx->vtcm_سائز < vtcm_سائز) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_سائز, vtcm_سائز);
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_صف_سائز_padded;
    mmctx->vtcm_act_stride  = act_صف_سائز;
    اگر (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف) {
        mmctx->vtcm_act_raw_stride = hex_باری_اوپر(ne10 * حجم(اعشاری), QK_Q8_0_TILED * حجم(اعشاری));
    } ورنہ اگر (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_باری_اوپر(ne10 * حجم(اعشاری), 128);
    } ورنہ {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    اگر (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        برائے (uint32_t m_آغاز = 0; m_آغاز < act_nrows; m_آغاز += m_chunk) {
            ثابت uint32_t cur_m_صفیں = MIN(act_nrows - m_آغاز, m_chunk);
            mmctx->cur_m_آغاز = m_آغاز;
            mmctx->cur_m_صفیں  = cur_m_صفیں;

            اگر (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_آغاز, cur_m_صفیں);

                ثابت uint32_t qk = QK_Q8_0_TILED;
                ثابت uint32_t nb = (ne10 + qk - 1) / qk;
                ثابت uint32_t کل_nb = cur_m_صفیں * nb;
                uint32_t quant_tasks;
                work_قطار_func_t q_func;
                اگر (cur_m_صفیں < octx->n_دھاگے && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف)) {
                    quant_tasks = MIN(کل_nb, octx->n_دھاگے);
                    q_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
                    برائے (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_پہلا = (کل_nb * ith) / quant_tasks;
                        uint32_t ib_آخری  = (کل_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_پہلا[ith] = ib_پہلا;
                        mmctx->quant_ib_آخری[ith]  = ib_آخری;
                        mmctx->quant_r[ith]        = ib_پہلا / nb;
                        mmctx->quant_c[ith]        = ib_پہلا % nb;
                    }
                } ورنہ {
                    quant_tasks = MIN(cur_m_صفیں, octx->n_دھاگے);
                    q_func = quant_task_func;
                    mmctx->n_quant_صفیں_فی_thread = (cur_m_صفیں + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_قطار_تسلسل(octx->ctx->work_قطار, q_func, mmctx, quant_tasks);
            } ورنہ {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_آغاز, cur_m_صفیں);
            }

            work_قطار_تسلسل(octx->ctx->work_قطار, matmul_کام_func, mmctx, octx->n_دھاگے);
        }
    } ورنہ {
        mmctx->cur_m_آغاز = 0;
        mmctx->cur_m_صفیں  = act_nrows;

        اگر (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_صفیں_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_قطار_تسلسل(octx->ctx->work_قطار, quant_task_func, mmctx, n_quant_tasks);
        } ورنہ {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_قطار_تسلسل(octx->ctx->work_قطار, matmul_کام_func, mmctx, octx->n_دھاگے);
    }

    واپس HTP_حالت_OK;
}

ساکن خالی dequantize_tiled_کارکن_loop(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) ڈیٹا;
    ساخت htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    برائے (غیرعلامتی صحیح task_id = i; task_id < (غیرعلامتی صحیح) state->n_tasks; task_id += n) {
        صحیح آغاز = task_id * state->n_tiles_فی_task;
        صحیح انتہا   = hex_smin(آغاز + state->n_tiles_فی_task, state->n_tot_tiles);
        state->task_fn(state, آغاز, انتہا);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

ساکن خالی transfer_output_chunk_کارکن_fn(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی *ڈیٹا) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) ڈیٹا;

    ساخت htp_thread_trace * tr = &st->traces[i];

    صحیح آغاز_chunk_idx = i * st->n_chunks_فی_task;
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_O_PROC, آغاز_chunk_idx);

    برائے (غیرعلامتی صحیح task_id = i; task_id < (غیرعلامتی صحیح)st->n_tasks; task_id += n) {
        صحیح    chunk_idx  = task_id * st->n_chunks_فی_task;
        size_t chunk_سائز = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_فی_task);

        اعشاری        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ثابت اعشاری  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : تہی;
        transfer_output_chunk_fp16_مقصد_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_سائز, st->n_ستون_تعداد, st->dst_stride, st->src2_stride, st->dst_ستون_تعداد);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, آغاز_chunk_idx);
}

قسمتعریف ساخت {
    ساخت htp_context            * ctx;
    ساخت htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    اعشاری                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_فی_task;
    uint32_t                        k_بلاک;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_بائٹیں_فی_thread;
    uint32_t                        dma_قدم_صفیں;
    uint32_t                        dma_قدم_صفیں_سرکاؤ;
    uint32_t                        act_elem_سائز;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

قسمتعریف ساخت {
    ساخت htp_context            * ctx;
    ساخت htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    اعشاری                         * vtcm_f32_act;
    uint32_t                        n_صفیں;
    uint32_t                        k_بلاک;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_ستون_chunks;
    ساخت fastdiv_اقدار           n_دھاگے_div;
    size_t                          vtcm_f32_act_بائٹیں;
    uint32_t                        dma_قدم_صفیں;
    uint32_t                        dma_قدم_صفیں_سرکاؤ;
} activation_transfer_ستون_chunk_state_t;

ساکن خالی transfer_activation_chunk_fp32_مقصد_fp16_dma_pipelined_ستون_chunk(
        dma_قطار *dma_q,
        __fp16 *محدود vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_صفیں,
        uint32_t k_بلاک,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_پہلا,
        uint32_t c_len,
        اعشاری *thread_f32_act,
        ساخت htp_thread_trace *tr,
        uint32_t dma_قدم_صفیں,
        uint32_t dma_قدم_صفیں_سرکاؤ) {

    ثابت uint32_t R = dma_قدم_صفیں;
    ثابت uint32_t n_صفیں_padded = hex_align_اوپر(n_صفیں, HTP_MM_HMX_TILE_N_صفیں);

    ثابت uint32_t n_اقدام = n_صفیں_padded >> dma_قدم_صفیں_سرکاؤ;

    // Push step 0
    اگر (n_اقدام > 0 && n_صفیں > 0) {
        uint32_t nrows_مقصد_fetch = hex_smin(n_صفیں, R);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(thread_f32_act, act_dma_addr + (size_t) c_پہلا * حجم(اعشاری)),
                       c_len * حجم(اعشاری), k_stride * حجم(اعشاری), k_chunk_valid * حجم(اعشاری), nrows_مقصد_fetch);
    }
    // Push step 1
    اگر (n_اقدام > 1) {
        uint32_t اگلا_r = R * 1;
        اگر (اگلا_r < n_صفیں) {
            uint32_t nrows_مقصد_fetch = hex_smin(n_صفیں - اگلا_r, R);
            اعشاری *اگلا_buf = thread_f32_act + 1 * R * c_len;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(اگلا_buf, act_dma_addr + ((size_t) اگلا_r * k_stride + c_پہلا) * حجم(اعشاری)),
                           c_len * حجم(اعشاری), k_stride * حجم(اعشاری), k_chunk_valid * حجم(اعشاری), nrows_مقصد_fetch);
        }
    }
    برائے (uint32_t s = 0; s < n_اقدام; ++s) {
        uint32_t r = s << dma_قدم_صفیں_سرکاؤ;
        اعشاری *curr_buf = thread_f32_act;

        اگر (r < n_صفیں) {
            curr_buf = (اعشاری *) dma_قطار_نکالو(dma_q).dst;
        }

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        برائے (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t صف_idx = r + (p << 1);
            اعشاری *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((صف_idx + 0) < n_صفیں);
            bool r1_valid = ((صف_idx + 1) < n_صفیں);

            transfer_activation_صف_pair_fp32_مقصد_fp16_ستون_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, صف_idx, k_بلاک, c_پہلا, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t اگلا_s = s + 2;
        uint32_t اگلا_r = اگلا_s << dma_قدم_صفیں_سرکاؤ;
        اگر (اگلا_r < n_صفیں) {
            uint32_t nrows_مقصد_fetch = hex_smin(n_صفیں - اگلا_r, R);
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(curr_buf, act_dma_addr + ((size_t) اگلا_r * k_stride + c_پہلا) * حجم(اعشاری)),
                           c_len * حجم(اعشاری), k_stride * حجم(اعشاری), k_chunk_valid * حجم(اعشاری), nrows_مقصد_fetch);
        }
    }
}

ساکن خالی transfer_activation_chunk_ستون_chunk_کارکن_fn(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی *ڈیٹا) {
    activation_transfer_ستون_chunk_state_t *st = (activation_transfer_ستون_chunk_state_t *) ڈیٹا;
    ساخت htp_thread_trace * tr = &st->traces[i];

    uint32_t n_بلاکیں = st->k_بلاک / 32;
    uint32_t b_پہلا = fastdiv(n_بلاکیں * i, &st->n_دھاگے_div);
    uint32_t b_آخری  = fastdiv(n_بلاکیں * (i + 1), &st->n_دھاگے_div);
    uint32_t c_پہلا = b_پہلا * 32;
    uint32_t c_آخری = b_آخری * 32;
    uint32_t c_len = c_آخری - c_پہلا;

    اگر (c_len == 0) {
        واپس;
    }

    uint32_t k_chunk_valid = 0;
    اگر (st->k_valid > c_پہلا) {
        k_chunk_valid = hex_smin(st->k_valid, c_آخری) - c_پہلا;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_بائٹیں = hex_align_نیچے_کو(fastdiv(st->vtcm_f32_act_بائٹیں, &st->n_دھاگے_div), 128);
    اعشاری *thread_f32_act = (اعشاری *)((حرف *)st->vtcm_f32_act + i * thread_scratch_بائٹیں);

    transfer_activation_chunk_fp32_مقصد_fp16_dma_pipelined_ستون_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_صفیں, st->k_بلاک, st->k_stride, k_chunk_valid,
        c_پہلا, c_len, thread_f32_act, tr, st->dma_قدم_صفیں, st->dma_قدم_صفیں_سرکاؤ
    );
}

ساکن خالی transfer_activation_chunk_مقصد_fp16_dma_pipelined(
        dma_قطار *dma_q,
        __fp16 *محدود vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_صفیں,
        uint32_t k_بلاک,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_سائز,
        ساخت htp_thread_trace *tr,
        uint32_t dma_قدم_صفیں,
        uint32_t dma_قدم_صفیں_سرکاؤ) {

    ثابت uint32_t R = dma_قدم_صفیں;
    ثابت uint32_t n_صفیں_padded = hex_align_اوپر(n_صفیں, HTP_MM_HMX_TILE_N_صفیں);

    ثابت uint32_t n_اقدام = n_صفیں_padded >> dma_قدم_صفیں_سرکاؤ;

    ثابت bool act_is_f16 = (act_elem_سائز == حجم(__fp16));
    ثابت size_t صف_بائٹیں = (size_t) k_بلاک * act_elem_سائز;  // staging row (DMA dst stride)
    ثابت size_t src_صف_بائٹیں = (size_t) k_stride * act_elem_سائز;  // DDR row (DMA src stride)
    ثابت size_t چوڑائی_بائٹیں = (size_t) k_valid * act_elem_سائز;

    // Push step 0
    اگر (n_اقدام > 0 && n_صفیں > 0) {
        uint32_t nrows_مقصد_fetch = hex_smin(n_صفیں, R);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(thread_act, act_dma_addr),
                       صف_بائٹیں, src_صف_بائٹیں, چوڑائی_بائٹیں, nrows_مقصد_fetch);
    }
    // Push step 1 (if valid)
    اگر (n_اقدام > 1) {
        uint32_t اگلا_r = R * 1;
        اگر (اگلا_r < n_صفیں) {
            uint32_t nrows_مقصد_fetch = hex_smin(n_صفیں - اگلا_r, R);
            uint8_t *اگلا_buf = thread_act + 1 * R * صف_بائٹیں;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(اگلا_buf, act_dma_addr + (size_t) اگلا_r * src_صف_بائٹیں),
                           صف_بائٹیں, src_صف_بائٹیں, چوڑائی_بائٹیں, nrows_مقصد_fetch);
        }
    }
    برائے (uint32_t s = 0; s < n_اقدام; ++s) {
        uint32_t r = s << dma_قدم_صفیں_سرکاؤ;
        uint8_t *curr_buf = thread_act;

        اگر (r < n_صفیں) {
            curr_buf = (uint8_t *) dma_قطار_نکالو(dma_q).dst;
        }

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        اگر (act_is_f16) {
            برائے (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t صف_idx = r + (p << 1);
                ثابت __fp16 *pair_buf = (ثابت __fp16 *) (curr_buf + (p << 1) * صف_بائٹیں);
                bool r0_valid = ((صف_idx + 0) < n_صفیں);
                bool r1_valid = ((صف_idx + 1) < n_صفیں);

                transfer_activation_صف_pair_f16_مقصد_f16(vtcm_dst, pair_buf, pair_buf + k_بلاک, صف_idx, k_بلاک, k_valid, r0_valid, r1_valid);
            }
        } ورنہ {
            برائے (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t صف_idx = r + (p << 1);
                ثابت اعشاری *pair_buf = (ثابت اعشاری *) (curr_buf + (p << 1) * صف_بائٹیں);
                bool r0_valid = ((صف_idx + 0) < n_صفیں);
                bool r1_valid = ((صف_idx + 1) < n_صفیں);

                transfer_activation_صف_pair_fp32_مقصد_fp16(vtcm_dst, pair_buf, pair_buf + k_بلاک, صف_idx, k_بلاک, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t اگلا_s = s + 2;
        uint32_t اگلا_r = اگلا_s << dma_قدم_صفیں_سرکاؤ;
        اگر (اگلا_r < n_صفیں) {
            uint32_t nrows_مقصد_fetch = hex_smin(n_صفیں - اگلا_r, R);
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(curr_buf, act_dma_addr + (size_t) اگلا_r * src_صف_بائٹیں),
                           صف_بائٹیں, src_صف_بائٹیں, چوڑائی_بائٹیں, nrows_مقصد_fetch);
        }
    }
}

ساکن خالی transfer_activation_chunk_کارکن_fn(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی *ڈیٹا) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) ڈیٹا;

    ساخت htp_thread_trace * tr = &st->traces[i];

    برائے (غیرعلامتی صحیح task_id = i; task_id < (غیرعلامتی صحیح)st->n_tasks; task_id += n) {
        صحیح    chunk_idx  = task_id * st->n_chunks_فی_task;
        size_t chunk_سائز = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_فی_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_بلاک;
        ثابت dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_سائز;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_بائٹیں_فی_thread;
        transfer_activation_chunk_مقصد_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_سائز, st->k_بلاک, st->k_stride, st->k_valid, thread_act, st->act_elem_سائز, tr, st->dma_قدم_صفیں, st->dma_قدم_صفیں_سرکاؤ
        );
    }
}

قسمتعریف ساخت {
    ساخت htp_thread_trace       * traces;
    ثابت ساخت mmid_صف_mapping * میٹرکس_صفیں;
    __fp16                        * dst;
    ثابت اعشاری                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_فی_task;
    uint32_t                        k_بلاک;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    ساخت fastdiv_اقدار           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        آغاز_صف;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

قسمتعریف ساخت {
    ساخت htp_thread_trace       * traces;
    ثابت ساخت mmid_صف_mapping * میٹرکس_صفیں;
    ثابت __fp16                  * vtcm_src;
    اعشاری                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_فی_task;
    uint32_t                        n_ستون_تعداد;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        آغاز_صف;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

ساکن خالی transfer_activation_chunk_gathered_کارکن_fn(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی *ڈیٹا) {
    activation_transfer_gathered_task_state_t *st = ڈیٹا;
    ساخت htp_thread_trace * tr = &st->traces[i];
    صحیح chunk_idx      = i;
    صحیح chunk_سائز     = st->n_chunks_فی_task;
    صحیح vtcm_آغاز_صف = chunk_idx * chunk_سائز;
    صحیح آغاز_صف      = st->آغاز_صف + vtcm_آغاز_صف;
    صحیح n_صفیں         = hex_smin(st->cne1 - آغاز_صف, chunk_سائز);
    اگر (n_صفیں > 0) {
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_مقصد_fp16_gathered(
            st->dst, st->src, آغاز_صف, vtcm_آغاز_صف, n_صفیں, st->k_بلاک,
            st->میٹرکس_صفیں, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ساکن خالی transfer_activation_chunk_gathered_کارکن_flat_fn(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی *ڈیٹا) {
    activation_transfer_gathered_task_state_t *st = ڈیٹا;
    ساخت htp_thread_trace * tr = &st->traces[i];
    صحیح chunk_idx = i;
    صحیح chunk_سائز = st->n_chunks_فی_task;
    صحیح vtcm_آغاز_صف = chunk_idx * chunk_سائز;
    صحیح آغاز_صف = st->آغاز_صف + vtcm_آغاز_صف;
    صحیح n_صفیں = hex_smin(st->cne1 - آغاز_صف, chunk_سائز);
    اگر (n_صفیں > 0) {
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_مقصد_fp16_gathered_flat(
            st->dst, st->src, آغاز_صف, vtcm_آغاز_صف, n_صفیں, st->k_بلاک,
            st->میٹرکس_صفیں, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ساکن خالی transfer_output_chunk_scattered_کارکن_fn(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی *ڈیٹا) {
    output_transfer_scattered_task_state_t *st = ڈیٹا;
    ساخت htp_thread_trace * tr = &st->traces[i];
    صحیح chunk_idx = i;
    صحیح chunk_سائز = st->n_chunks_فی_task;
    صحیح vtcm_آغاز_صف = chunk_idx * chunk_سائز;
    صحیح آغاز_صف = st->آغاز_صف + vtcm_آغاز_صف;
    صحیح n_صفیں = hex_smin(st->cne1 - آغاز_صف, chunk_سائز);
    اگر (n_صفیں > 0) {
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_مقصد_fp32_scattered(
            st->dst, st->vtcm_src, آغاز_صف, vtcm_آغاز_صف, n_صفیں, st->n_ستون_تعداد,
            st->میٹرکس_صفیں, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

ساکن اندرونی tiled_dequant_task_fn_t hmx_get_dequant_کارکن_fn(صحیح وزن_type, bool is_tiled) {
    انتخاب (وزن_type) {
        حالت HTP_TYPE_Q4_0:   واپس dequantize_tiled_وزن_مقصد_fp16_task_q4_0;
        حالت HTP_TYPE_IQ4_NL: واپس dequantize_tiled_وزن_مقصد_fp16_task_iq4_nl;
        حالت HTP_TYPE_Q4_1:
        حالت HTP_TYPE_Q4_K:   واپس dequantize_tiled_وزن_مقصد_fp16_task_q4_1;
        حالت HTP_TYPE_MXFP4:  واپس dequantize_tiled_وزن_مقصد_fp16_task_mxfp4;
        حالت HTP_TYPE_Q8_0:   واپس dequantize_tiled_وزن_مقصد_fp16_task_q8_0;
        حالت HTP_TYPE_Q5_K:   واپس dequantize_tiled_وزن_مقصد_fp16_task_q5_k;
        حالت HTP_TYPE_Q6_K:   واپس dequantize_tiled_وزن_مقصد_fp16_task_q6_k;
        حالت HTP_TYPE_Q3_K:   واپس dequantize_tiled_وزن_مقصد_fp16_task_q3_k;
        حالت HTP_TYPE_Q2_K:   واپس dequantize_tiled_وزن_مقصد_fp16_task_q2_k;
        حالت HTP_TYPE_Q1_0:   واپس dequantize_tiled_وزن_مقصد_fp16_task_q1_0;
        حالت HTP_TYPE_Q2_0:   واپس dequantize_tiled_وزن_مقصد_fp16_task_q2_0;
        حالت HTP_TYPE_F16:    واپس is_tiled ? تہی : convert_f16_وزن_مقصد_fp16_tiles_task;
        حالت HTP_TYPE_BF16:   واپس convert_bf16_وزن_مقصد_fp16_tiles_task;
        حالت HTP_TYPE_F32:    واپس quantize_f32_وزن_مقصد_fp16_tiles_task;
        طےشدہ:              واپس تہی;
    }
}

ساکن خالی dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
        ساخت htp_context *ctx, __fp16 *vtcm_dst,
        ثابت خالی *وزن_src_ddr,
        صحیح n_ستون_تعداد, صحیح k_بلاک,
        size_t صف_stride, صحیح وزن_type,
        صحیح n_k_tiles, ساخت fastdiv_اقدار n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_کارکن_fn, صحیح n_دھاگے) {

    اگر (!dequant_کارکن_fn) {
        واپس;
    }

    أكد(k_بلاک % HTP_MM_HMX_TILE_N_ستون_تعداد == 0);

    ثابت size_t n_ستون_tiles = hmx_ceil_div(n_ستون_تعداد, HTP_MM_HMX_TILE_N_ستون_تعداد);
    ثابت uint32_t n_وزن_tiles = k_بلاک / htp_tensor_get_tile_k(وزن_type);
    ثابت size_t n_tot_tiles = n_ستون_tiles * n_وزن_tiles;

    size_t n_tiles_فی_task = (n_دھاگے == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_دھاگے);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_فی_task - 1) / n_tiles_فی_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_فی_task = n_tiles_فی_task;
    state.dst              = vtcm_dst;
    state.src              = (ثابت uint8_t *)وزن_src_ddr;
    state.task_fn          = dequant_کارکن_fn;
    state.n_ستون_تعداد           = n_ستون_تعداد;
    state.k_بلاک          = k_بلاک;
    state.صف_stride       = صف_stride;
    state.وزن_type      = وزن_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_وزن_tiles   = n_وزن_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_سائز = htp_tensor_get_tile_سائز(وزن_type);
    state.aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(وزن_type);

    اگر (state.n_tasks == 1 || n_دھاگے == 1) {
        dequantize_tiled_کارکن_loop(1, 0, &state);
    } ورنہ {
        صحیح n_tasks = hex_smin((صحیح) state.n_tasks, n_دھاگے);
        کارکن_pool_تسلسل_func(ctx->کارکن_pool, dequantize_tiled_کارکن_loop, &state, n_tasks);
    }
}

قسمتعریف ساخت {
    ساخت htp_context      * ctx;
    ساخت htp_thread_trace * traces;
    اعشاری                   * dst;
    ثابت __fp16            * vtcm_src;
    ثابت اعشاری             * src2;
    uint32_t                  n_صفیں;
    uint32_t                  n_ستون_تعداد;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_ستون_تعداد;
    ساخت fastdiv_اقدار     n_دھاگے_div;
} output_transfer_ستون_chunk_state_t;

ساکن خالی transfer_output_chunk_ستون_chunk_کارکن_fn(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی *ڈیٹا) {
    (خالی) n;
    output_transfer_ستون_chunk_state_t *st = (output_transfer_ستون_chunk_state_t *) ڈیٹا;
    ساخت htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_بلاکیں = hmx_ceil_div(st->n_ستون_تعداد, 32);
    uint32_t b_پہلا  = fastdiv(n_بلاکیں * i, &st->n_دھاگے_div);
    uint32_t b_آخری   = fastdiv(n_بلاکیں * (i + 1), &st->n_دھاگے_div);
    uint32_t c_پہلا  = b_پہلا * 32;
    uint32_t c_آخری   = b_آخری * 32;
    uint32_t c_len    = c_آخری - c_پہلا;

    اگر (c_len == 0) واپس;

    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_O_PROC, c_پہلا);

    ثابت __fp16 *vtcm_src = st->vtcm_src + b_پہلا * HTP_MM_HMX_TILE_N_ELMS;
    ثابت اعشاری *src2      = st->src2 ? (st->src2 + c_پہلا) : تہی;
    اعشاری *dst             = st->dst + c_پہلا;

    صحیح chunk_dst_ستون_تعداد = (صحیح)st->dst_ستون_تعداد - (صحیح)c_پہلا;
    اگر (chunk_dst_ستون_تعداد > 0) {
        transfer_output_chunk_fp16_مقصد_fp32_ستون_chunk(
            dst, src2, vtcm_src, 0, st->n_صفیں, c_len, st->n_ستون_تعداد,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_ستون_تعداد
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_پہلا);
}

ساکن خالی transfer_output_chunk_threaded(ساخت htp_context *ctx, اعشاری *dst, ثابت اعشاری *src2, ثابت __fp16 *vtcm_src,
                                              صحیح n_صفیں, صحیح n_ستون_تعداد, صحیح dst_stride, uint32_t src2_stride, صحیح dst_ستون_تعداد, صحیح n_دھاگے) {
    اگر (n_صفیں <= 0) واپس;

    uint32_t n_بلاکیں = hmx_ceil_div((uint32_t) n_ستون_تعداد, 32);
    اگر (n_دھاگے > 1 && n_بلاکیں >= (uint32_t)n_دھاگے) {
        ساخت fastdiv_اقدار n_دھاگے_div = (n_دھاگے == (صحیح)ctx->n_دھاگے) ? ctx->n_دھاگے_div : init_fastdiv_اقدار(n_دھاگے);
        output_transfer_ستون_chunk_state_t ستون_state;
        ستون_state.dst = dst;
        ستون_state.src2 = src2;
        ستون_state.vtcm_src = vtcm_src;
        ستون_state.n_صفیں = (uint32_t)n_صفیں;
        ستون_state.n_ستون_تعداد = (uint32_t)n_ستون_تعداد;
        ستون_state.dst_stride = (uint32_t)dst_stride;
        ستون_state.src2_stride = src2_stride;
        ستون_state.dst_ستون_تعداد = (uint32_t)dst_ستون_تعداد;
        ستون_state.n_دھاگے_div = n_دھاگے_div;
        ستون_state.traces = ctx->trace;
        ستون_state.ctx = ctx;

        کارکن_pool_تسلسل_func(ctx->کارکن_pool, transfer_output_chunk_ستون_chunk_کارکن_fn, &ستون_state, n_دھاگے);
        واپس;
    }

    size_t n_tot_chunks      = n_صفیں;
    size_t n_chunks_فی_task = (n_دھاگے == 1) ? n_tot_chunks : hmx_ceil_div(n_صفیں, n_دھاگے);
    n_chunks_فی_task        = hex_align_اوپر(n_chunks_فی_task, 2);

    صحیح actual_دھاگے = hmx_ceil_div(n_صفیں, n_chunks_فی_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_دھاگے;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_فی_task = n_chunks_فی_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_ستون_تعداد            = n_ستون_تعداد;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_ستون_تعداد          = dst_ستون_تعداد;
    state.traces            = ctx->trace;

    اگر (actual_دھاگے <= 1) {
        transfer_output_chunk_کارکن_fn(1, 0, &state);
    } ورنہ {
        کارکن_pool_تسلسل_func(ctx->کارکن_pool, transfer_output_chunk_کارکن_fn, &state, actual_دھاگے);
    }
}

ساخت activation_transfer_params {
    ساخت htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    صحیح                           n_صفیں;
    صحیح                           k_بلاک;
    صحیح                           k_stride;
    صحیح                           n_دھاگے;
    ثابت ساخت fastdiv_اقدار * act_دھاگے_div;
    ثابت ساخت fastdiv_اقدار * k_div;
    صحیح                           k_valid;
    اعشاری *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_بائٹیں;
    uint32_t                      act_elem_سائز;  // 4=F32 src1, 2=F16 src1
};

ساکن خالی transfer_activation_chunk_threaded(ثابت ساخت activation_transfer_params * params) {
    ساخت htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ثابت dma_addr_t              act_dma_addr       = params->act_dma_addr;
    صحیح                           n_صفیں             = params->n_صفیں;
    صحیح                           k_بلاک            = params->k_بلاک;
    صحیح                           k_stride           = params->k_stride;
    صحیح                           n_دھاگے          = params->n_دھاگے;
    ثابت ساخت fastdiv_اقدار * act_دھاگے_div    = params->act_دھاگے_div;
    ثابت ساخت fastdiv_اقدار * k_div              = params->k_div;
    صحیح                           k_valid            = params->k_valid;
    اعشاری *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_بائٹیں = params->vtcm_f32_act_بائٹیں;
    // element size of the activation rows (4 = F32, 2 = F16).
    ثابت uint32_t                act_elem_سائز      = params->act_elem_سائز ? params->act_elem_سائز : (uint32_t) حجم(اعشاری);

    اگر (n_صفیں <= 0) {
        واپس;
    }

    ثابت size_t n_tasks = (n_صفیں + 31) >> 5;
    اگر (act_elem_سائز == حجم(اعشاری) && n_دھاگے > 1 && k_بلاک > 32 && n_tasks < (size_t) n_دھاگے) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_قدم_صفیں = 2;
        uint32_t dma_قدم_صفیں_سرکاؤ = 1;
        اگر (vtcm_f32_act_بائٹیں > 0) {
            size_t thread_scratch_بائٹیں = hex_align_نیچے_کو(fastdiv(vtcm_f32_act_بائٹیں, act_دھاگے_div), 128);
            size_t thread_scratch_elements = thread_scratch_بائٹیں / حجم(اعشاری);
            size_t dma_قدم_صفیں_max = fastdiv(thread_scratch_elements / 2, k_div);
            اگر (dma_قدم_صفیں_max >= 4) {
                dma_قدم_صفیں = 4;
                dma_قدم_صفیں_سرکاؤ = 2;
            }
        }

        activation_transfer_ستون_chunk_state_t ستون_state;
        ستون_state.dst = dst;
        ستون_state.act_dma_addr = act_dma_addr;
        ستون_state.n_صفیں = n_صفیں;
        ستون_state.k_بلاک = k_بلاک;
        ستون_state.k_stride = k_stride;
        ستون_state.k_valid = k_valid;
        ستون_state.n_ستون_chunks = n_دھاگے;
        ستون_state.n_دھاگے_div = *act_دھاگے_div;
        ستون_state.vtcm_f32_act = vtcm_f32_act;
        ستون_state.vtcm_f32_act_بائٹیں = vtcm_f32_act_بائٹیں;
        ستون_state.traces = ctx->trace;
        ستون_state.ctx = ctx;
        ستون_state.dma_قدم_صفیں = dma_قدم_صفیں;
        ستون_state.dma_قدم_صفیں_سرکاؤ = dma_قدم_صفیں_سرکاؤ;

        کارکن_pool_تسلسل_func(ctx->کارکن_pool, transfer_activation_chunk_ستون_chunk_کارکن_fn, &ستون_state, n_دھاگے);
        واپس;
    }

    أكد(k_بلاک % HTP_MM_HMX_TILE_N_ستون_تعداد == 0 && k_stride % HTP_MM_HMX_TILE_N_ستون_تعداد == 0);

    size_t n_tot_chunks      = n_صفیں;
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
    state.act_elem_سائز = act_elem_سائز;

    state.vtcm_f32_act_بائٹیں_فی_thread = hex_align_نیچے_کو(fastdiv(vtcm_f32_act_بائٹیں, act_دھاگے_div), 128);

    uint32_t dma_قدم_صفیں = 2;
    uint32_t dma_قدم_صفیں_سرکاؤ = 1;
    اگر (state.vtcm_f32_act_بائٹیں_فی_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_بائٹیں_فی_thread / حجم(اعشاری);
        size_t dma_قدم_صفیں_max = fastdiv(thread_scratch_elements / 2, k_div);
        اگر (dma_قدم_صفیں_max >= 4) {
            dma_قدم_صفیں = 4;
            dma_قدم_صفیں_سرکاؤ = 2;
        }
    }
    state.dma_قدم_صفیں      = dma_قدم_صفیں;
    state.dma_قدم_صفیں_سرکاؤ = dma_قدم_صفیں_سرکاؤ;

    صحیح فعال_دھاگے = hex_smin(n_دھاگے, (صحیح)state.n_tasks);
    اگر (state.n_tasks == 1 || n_دھاگے == 1) {
        transfer_activation_chunk_کارکن_fn(1, 0, &state);
    } ورنہ {
        کارکن_pool_تسلسل_func(ctx->کارکن_pool, transfer_activation_chunk_کارکن_fn, &state, فعال_دھاگے);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

قسمتعریف ساخت {
    __fp16 *       output;
    ثابت __fp16 * activation;
    ثابت __fp16 * وزن;
    ثابت __fp16 * scales;
    uint32_t       n_صف_tiles;
    uint32_t       n_ستون_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_کام_t;

ساکن خالی hmx_matmul_کارکن_fn(خالی * ڈیٹا) {
    hmx_matmul_کام_t * کام = (hmx_matmul_کام_t *) ڈیٹا;
    FARF(بلند_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", کام->n_صف_tiles, کام->n_ستون_tiles, کام->n_dot_tiles);
    core_dot_chunk_fp16(کام->output, کام->activation, کام->وزن, کام->scales, کام->n_صف_tiles, کام->n_ستون_tiles, کام->n_dot_tiles);
}

ساکن اندرونی خالی hmx_matmul_کام_init(hmx_matmul_کام_t * کام,
                                       __fp16 *           output,
                                       ثابت __fp16 *     activation,
                                       ثابت __fp16 *     وزن,
                                       ثابت __fp16 *     scales,
                                       uint32_t           n_صف_tiles,
                                       uint32_t           n_ستون_tiles,
                                       uint32_t           n_dot_tiles) {
    کام->output      = output;
    کام->activation  = activation;
    کام->وزن      = وزن;
    کام->scales      = scales;
    کام->n_صف_tiles = n_صف_tiles;
    کام->n_ستون_tiles = n_ستون_tiles;
    کام->n_dot_tiles = n_dot_tiles;
}

ساکن صحیح hmx_mm_2d_f32(ساخت htp_context *ctx,
                                  dma_قطار *وزن_dma,
                                  اعشاری *محدود dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_بائٹیں,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t وزن,
                                  صحیح m, صحیح k, صحیح n,
                                  صحیح act_stride,
                                  uint32_t act_elem_سائز,  // 4=F32 src1, 2=F16 src1
                                  صحیح وزن_stride,
                                  صحیح وزن_type,
                                  صحیح k_valid,
                                  صحیح dst_stride,
                                  uint32_t src2_stride,
                                  صحیح dst_ستون_تعداد,
                                  صحیح m_chunk,
                                  صحیح n_chunk,
                                  صحیح pipeline,
                                  صحیح n_دھاگے,
                                  صحیح act_دھاگے,
                                  ثابت ساخت fastdiv_اقدار * act_دھاگے_div,
                                  ثابت ساخت fastdiv_اقدار * k_div,
                                  صحیح tile_سائز,
                                  صحیح aligned_tile_سائز,
                                  صحیح vtcm_سائز) {
    ساخت htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت bool is_tiled = (tile_سائز > 0);

    اگر (k % 32 != 0) {
        واپس -1;
    }
    اگر (!hex_is_aligned(dst, حجم(اعشاری))) { واپس -1; }

    size_t صف_stride = htp_tensor_get_صف_سائز(وزن_type, k);
    اگر (صف_stride == 0) {
        واپس -1;
    }

    tiled_dequant_task_fn_t dequant_کارکن_fn = hmx_get_dequant_کارکن_fn(وزن_type, is_tiled);

    ثابت صحیح n_k_tiles = k / HTP_MM_HMX_TILE_N_ستون_تعداد;
    ثابت صحیح n_وزن_tiles = k / htp_tensor_get_tile_k(وزن_type);
    ثابت ساخت fastdiv_اقدار n_k_tiles_div = init_fastdiv_اقدار(n_k_tiles);

    ثابت size_t vec_dot_سائز = k * حجم(__fp16);
    ثابت size_t vtcm_budget  = ctx->vtcm_سائز;

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_سائز : صف_stride;
    ثابت uint32_t dma_src_stride  = is_tiled ? tile_سائز : وزن_stride;
    ثابت uint32_t dma_چوڑائی_بائٹیں = is_tiled ? tile_سائز : صف_stride;

    size_t m_chunk_n_صفیں = m_chunk;
    size_t n_chunk_n_ستون_تعداد = n_chunk;
    size_t vtcm_مستعمل      = vtcm_سائز;

    ساخت htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, وزن_type, k, m_chunk_n_صفیں, n_chunk_n_ستون_تعداد, 1, pipeline, act_دھاگے, aligned_tile_سائز, src2_بائٹیں);

    vtcm_مستعمل = L.کل_بائٹیں;
    اگر (vtcm_مستعمل > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_مستعمل, vtcm_budget, m, k, n, m_chunk_n_صفیں, n_chunk_n_ستون_تعداد);
        واپس -1;
    }

    uint8_t * ثابت بنیاد = (uint8_t *) ctx->vtcm_بنیاد;
    __fp16  *vtcm_وزن_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_وزن[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنیاد, L.off_وزن[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_act);
    اعشاری   *vtcm_f32_act    = VTCM_LAYOUT_PTR(اعشاری, بنیاد, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_dst[0]);
    خالی    *vtcm_scratch0   = VTCM_LAYOUT_PTR(خالی, بنیاد, L.off_scratch[0]);
    خالی    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(خالی, بنیاد, L.off_scratch[1], pipeline);
    خالی    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(خالی, بنیاد, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_scales);

    hmx_init_عمود_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ثابت bool has_bias      = (src2_بائٹیں > 0 && src2_addr != 0);
    اعشاری   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(اعشاری, بنیاد, L.off_bias, has_bias);
    اگر (has_bias) {
        dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_bias, src2_addr), hex_align_اوپر(src2_بائٹیں, 128), 0, src2_بائٹیں, 1);
        dma_قطار_نکالو(وزن_dma);
    }

    FARF(بلند_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, وزن_type, m_chunk_n_صفیں, n_chunk_n_ستون_تعداد, vtcm_مستعمل, vtcm_budget);

    صحیح n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_ستون_تعداد);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    اگر (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_کام_t کام_slots[2];  // persistent double-buffered job descriptors
        خالی *vtcm_raw_bufs[4] = {
            vtcm_وزن_raw[0],
            vtcm_وزن_raw[1],
            dequant_کارکن_fn ? vtcm_وزن_raw[0] : vtcm_scratch0,
            dequant_کارکن_fn ? vtcm_وزن_raw[1] : vtcm_scratch1,
        };
        خالی *vtcm_وزن_bufs[4] = {
            dequant_کارکن_fn ? vtcm_scratch0 : vtcm_وزن_raw[0],
            dequant_کارکن_fn ? vtcm_scratch1 : vtcm_وزن_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        خالی *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        برائے (size_t mr = 0; mr < m; mr += m_chunk_n_صفیں) {
            ثابت size_t n_صفیں = hex_smin(m - mr, m_chunk_n_صفیں);

            ساخت activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_سائز,  // byte offset (F16/F32)
                .n_صفیں = (صحیح) n_صفیں,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_دھاگے = act_دھاگے,
                .act_دھاگے_div = act_دھاگے_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بائٹیں = L.act_f32_بائٹیں,
                .act_elem_سائز = act_elem_سائز,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            ثابت size_t   n_ستون_تعداد_A0 = hex_smin(n - 0 * n_chunk_n_ستون_تعداد, n_chunk_n_ستون_تعداد);
            ثابت uint32_t اونچائی_A0 = is_tiled ? hmx_ceil_div(n_ستون_تعداد_A0, 32) * n_وزن_tiles : n_ستون_تعداد_A0;
            dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[0], وزن),
                           dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی_A0);

            اگر (1 < n_chunk_cnt) {
                ثابت size_t   n_ستون_تعداد_A1 = hex_smin(n - 1 * n_chunk_n_ستون_تعداد, n_chunk_n_ستون_تعداد);
                ثابت uint32_t اونچائی_A1 = is_tiled ? hmx_ceil_div(n_ستون_تعداد_A1, 32) * n_وزن_tiles : n_ستون_تعداد_A1;
                dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[1], وزن + n_chunk_n_ستون_تعداد * وزن_stride),
                               dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            برائے (صحیح i = 0; i < n_chunk_cnt; ++i) {
                ثابت size_t nc    = i * n_chunk_n_ستون_تعداد;
                ثابت size_t nc_p2 = nc + 2 * n_chunk_n_ستون_تعداد;

                ثابت size_t n_ستون_تعداد    = hex_smin(n - nc, n_chunk_n_ستون_تعداد);
                ثابت size_t n_ستون_تعداد_p2 = hex_smin(n - nc_p2, n_chunk_n_ستون_تعداد);

                // 1. pop A_i
                خالی * curr_raw = (خالی *) dma_قطار_نکالو(وزن_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
                    ctx, vtcm_وزن_bufs[i % 4], curr_raw,
                    n_ستون_تعداد, k, صف_stride, وزن_type,
                    n_k_tiles, n_k_tiles_div, dequant_کارکن_fn, n_دھاگے);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                اگر (i + 2 < n_chunk_cnt) {
                    ثابت uint32_t اونچائی_p2 = is_tiled ? hmx_ceil_div(n_ستون_تعداد_p2, 32) * n_وزن_tiles : n_ستون_تعداد_p2;
                    dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[(i + 2) % 4], وزن + nc_p2 * وزن_stride),
                                   dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی_p2);
                }

                // 4. submit C_i
                hmx_matmul_کام_init(&کام_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_صفیں, HTP_MM_HMX_TILE_N_صفیں),
                                    hmx_ceil_div(n_ستون_تعداد, HTP_MM_HMX_TILE_N_ستون_تعداد), k / HTP_MM_HMX_TILE_N_صفیں);
                hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کارکن_fn, &کام_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                اگر (i > 0) {
                    hmx_قطار_نکالو(ctx->hmx_قطار);
                    ثابت size_t nc_پچھلا = (i - 1) * n_chunk_n_ستون_تعداد;
                    ثابت size_t n_ستون_تعداد_پچھلا = hex_smin(n - nc_پچھلا, n_chunk_n_ستون_تعداد);
                    اعشاری *output_chunk = dst + (mr * dst_stride + nc_پچھلا);
                    ثابت اعشاری *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_پچھلا) : تہی;
                    صحیح chunk_dst_ستون_تعداد = dst_ستون_تعداد - (صحیح)nc_پچھلا;
                    اگر (chunk_dst_ستون_تعداد > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_صفیں, n_ستون_تعداد_پچھلا, dst_stride, src2_stride, chunk_dst_ستون_تعداد, n_دھاگے);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_قطار_نکالو(ctx->hmx_قطار);
            ثابت size_t nc_آخری = (n_chunk_cnt - 1) * n_chunk_n_ستون_تعداد;
            ثابت size_t n_ستون_تعداد_آخری = hex_smin(n - nc_آخری, n_chunk_n_ستون_تعداد);
            اعشاری *output_chunk = dst + (mr * dst_stride + nc_آخری);
            ثابت اعشاری *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_آخری) : تہی;
            صحیح chunk_dst_ستون_تعداد = dst_ستون_تعداد - (صحیح)nc_آخری;
            اگر (chunk_dst_ستون_تعداد > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_صفیں, n_ستون_تعداد_آخری, dst_stride, src2_stride, chunk_dst_ستون_تعداد, n_دھاگے);
            }
        }
    } ورنہ {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_کام_t کام;
        خالی *vtcm_raw_bufs[2] = { vtcm_وزن_raw[0], dequant_کارکن_fn ? vtcm_وزن_raw[0] : vtcm_scratch0 };
        خالی *vtcm_وزن_bufs[2] = { dequant_کارکن_fn ? vtcm_scratch0 : vtcm_وزن_raw[0], vtcm_scratch0 };

        برائے (size_t mr = 0; mr < m; mr += m_chunk_n_صفیں) {
            ثابت size_t n_صفیں = hex_smin(m - mr, m_chunk_n_صفیں);

            ساخت activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_سائز,  // byte offset (F16/F32)
                .n_صفیں = (صحیح) n_صفیں,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_دھاگے = act_دھاگے,
                .act_دھاگے_div = act_دھاگے_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بائٹیں = L.act_f32_بائٹیں,
                .act_elem_سائز = act_elem_سائز,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            اگر (n > 0) {
                ثابت size_t n_ستون_تعداد = hex_smin(n, n_chunk_n_ستون_تعداد);
                ثابت uint32_t اونچائی = is_tiled ? hmx_ceil_div(n_ستون_تعداد, 32) * n_وزن_tiles : n_ستون_تعداد;
                dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[0], وزن), dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی);
            }

            برائے (صحیح i = 0; i < n_chunk_cnt; ++i) {
                ثابت size_t nc = (size_t) i * n_chunk_n_ستون_تعداد;
                ثابت size_t n_ستون_تعداد = hex_smin(n - nc, n_chunk_n_ستون_تعداد);
                ثابت size_t n_صف_tiles = hmx_ceil_div(n_صفیں, HTP_MM_HMX_TILE_N_صفیں);
                ثابت size_t n_ستون_tiles = hmx_ceil_div(n_ستون_تعداد, HTP_MM_HMX_TILE_N_ستون_تعداد);

                // A: Wait for weight DMA
                خالی * curr_raw = (خالی *) dma_قطار_نکالو(وزن_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
                    ctx, vtcm_وزن_bufs[i % 2], curr_raw,
                    n_ستون_تعداد, k, صف_stride, وزن_type,
                    n_k_tiles, n_k_tiles_div, dequant_کارکن_fn, n_دھاگے);

                // Start weight DMA for the next chunk early
                ثابت size_t nc_اگلا = nc + n_chunk_n_ستون_تعداد;
                اگر (nc_اگلا < n) {
                    ثابت size_t n_ستون_تعداد_اگلا = hex_smin(n - nc_اگلا, n_chunk_n_ستون_تعداد);
                    ثابت uint32_t اونچائی_اگلا = is_tiled ? hmx_ceil_div(n_ستون_تعداد_اگلا, 32) * n_وزن_tiles : n_ستون_تعداد_اگلا;
                    dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[(i + 1) % 2], وزن + nc_اگلا * وزن_stride), dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی_اگلا);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_کام_init(&کام, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 2], vtcm_scales, n_صف_tiles, n_ستون_tiles, k / HTP_MM_HMX_TILE_N_صفیں);
                hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کارکن_fn, &کام));
                hmx_قطار_نکالو(ctx->hmx_قطار);

                // D: Output Store
                اعشاری *output_chunk = dst + (mr * dst_stride + nc);
                ثابت اعشاری *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : تہی;
                صحیح chunk_dst_ستون_تعداد = dst_ستون_تعداد - (صحیح)nc;
                اگر (chunk_dst_ستون_تعداد > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_صفیں, n_ستون_تعداد, dst_stride, src2_stride, chunk_dst_ستون_تعداد, n_دھاگے);
                }
            }
        }
    }

    واپس 0;
}

ساکن صحیح hmx_mm_nx_2d_f32(ساخت htp_ops_context * octx, ثابت ساخت htp_mm_kernel_params * kparams) {
    ساخت htp_context * ctx = octx->ctx;
    ساخت htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت uint32_t n_weights = kparams->n_weights;
    اگر (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    ثابت ساخت htp_tensor * محدود src0 = octx->src[0];
    ثابت ساخت htp_tensor * محدود act  = octx->src[n_weights];

    ثابت صحیح وزن_type = (صحیح) src0->type;
    ثابت صحیح k           = (صحیح) act->ne[0];
    ثابت صحیح k_valid     = (صحیح) act->ne[0];
    ثابت صحیح m           = (صحیح) (act->ne[1] * act->ne[2] * act->ne[3]);
    ثابت uint32_t act_elem_سائز = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(اعشاری);
    ثابت صحیح act_stride  = (صحیح) (act->nb[1] / act_elem_سائز);
    ثابت dma_addr_t act_dma_addr = act->ڈیٹا;

    اگر (k % 32 != 0) { واپس HTP_حالت_NO_SUPPORT; }

    size_t صف_stride = htp_tensor_get_صف_سائز(وزن_type, k);
    اگر (صف_stride == 0) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت صحیح n_k_tiles = k / HTP_MM_HMX_TILE_N_ستون_تعداد;
    ثابت صحیح n_وزن_tiles = k / htp_tensor_get_tile_k(وزن_type);
    ثابت ساخت fastdiv_اقدار n_k_tiles_div = init_fastdiv_اقدار(n_k_tiles);

    ثابت صحیح tile_سائز       = kparams->tile_سائز;
    ثابت صحیح aligned_tile_سائز = kparams->aligned_tile_سائز;

    ثابت bool is_tiled       = (tile_سائز > 0);
    tiled_dequant_task_fn_t dequant_کارکن_fn = hmx_get_dequant_کارکن_fn(وزن_type, is_tiled);
    ثابت size_t vtcm_budget  = ctx->vtcm_سائز;

    ثابت صحیح m_chunk_n_صفیں  = kparams->m_chunk;
    ثابت صحیح n_chunk_n_ستون_تعداد  = kparams->n_chunk;
    ثابت صحیح pipeline        = kparams->pipeline;
    ثابت صحیح n_دھاگے       = octx->n_دھاگے;
    ثابت صحیح act_دھاگے     = kparams->n_act_دھاگے;
    ثابت ساخت fastdiv_اقدار * act_دھاگے_div = &kparams->div_n_act_دھاگے;
    ثابت ساخت fastdiv_اقدار * k_div           = &kparams->div_ne00_padded;

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_سائز : صف_stride;
    ثابت uint32_t dma_چوڑائی_بائٹیں = is_tiled ? tile_سائز : صف_stride;

    ساخت htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, وزن_type, k, m_chunk_n_صفیں, n_chunk_n_ستون_تعداد, 1, pipeline, act_دھاگے, aligned_tile_سائز, 0);

    اگر (L.کل_بائٹیں > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.کل_بائٹیں, vtcm_budget, m, k, m_chunk_n_صفیں, n_chunk_n_ستون_تعداد);
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنیاد = (uint8_t *) ctx->vtcm_بنیاد;
    __fp16  *vtcm_وزن_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_وزن[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنیاد, L.off_وزن[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_act);
    اعشاری   *vtcm_f32_act    = VTCM_LAYOUT_PTR(اعشاری, بنیاد, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_dst[0]);
    خالی    *vtcm_scratch0   = VTCM_LAYOUT_PTR(خالی, بنیاد, L.off_scratch[0]);
    خالی    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(خالی, بنیاد, L.off_scratch[1], pipeline);
    خالی    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(خالی, بنیاد, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_scales);

    hmx_init_عمود_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    صحیح m_آغاز = 0;
    صحیح m_صفیں  = m;
    اگر (octx->ctx->mdev.گنتی > 1) {
        bool can_بانٹو = htp_tensor_can_صف_تقسیم(act, act_elem_سائز);
        اگر (can_بانٹو) {
            برائے (uint32_t p = 0; p < n_weights; ++p) {
                ثابت ساخت htp_tensor * محدود dst = octx->dsts[p];
                اگر (!htp_tensor_can_صف_تقسیم(dst, حجم(اعشاری))) {
                    can_بانٹو = false;
                    توڑو;
                }
            }
        }
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم((uint32_t) m, can_بانٹو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        m_آغاز = (صحیح) range.آغاز;
        m_صفیں  = (صحیح) range.گنتی;
    }

    اگر (m_صفیں == 0) {
        واپس HTP_حالت_OK;
    }

    FARF(بلند_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_آغاز, m_آغاز + m_صفیں, k, وزن_type, m_chunk_n_صفیں, n_chunk_n_ستون_تعداد, L.کل_بائٹیں, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت size_t mr_انتہا = (size_t)(m_آغاز + m_صفیں);

    اگر (pipeline) {
        hmx_matmul_کام_t کام_slots[2];
        خالی *vtcm_raw_bufs[4] = {
            vtcm_وزن_raw[0],
            vtcm_وزن_raw[1],
            dequant_کارکن_fn ? vtcm_وزن_raw[0] : vtcm_scratch0,
            dequant_کارکن_fn ? vtcm_وزن_raw[1] : vtcm_scratch1,
        };
        خالی *vtcm_وزن_bufs[4] = {
            dequant_کارکن_fn ? vtcm_scratch0 : vtcm_وزن_raw[0],
            dequant_کارکن_fn ? vtcm_scratch1 : vtcm_وزن_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        خالی *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        برائے (size_t mr = (size_t) m_آغاز; mr < mr_انتہا; mr += m_chunk_n_صفیں) {
            ثابت size_t n_صفیں = hex_smin(mr_انتہا - mr, m_chunk_n_صفیں);

            ساخت activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_سائز,
                .n_صفیں = (صحیح) n_صفیں,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_دھاگے = act_دھاگے,
                .act_دھاگے_div = act_دھاگے_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بائٹیں = L.act_f32_بائٹیں,
                .act_elem_سائز = act_elem_سائز,
            };
            transfer_activation_chunk_threaded(&act_params);

            برائے (uint32_t p = 0; p < n_weights; p++) {
                ثابت ساخت htp_tensor * محدود src_w = octx->src[p];
                ثابت ساخت htp_tensor * محدود dst   = octx->dsts[p];
                اگر (!src_w || !dst) جاری;

                ثابت dma_addr_t وزن       = src_w->ڈیٹا;
                dma_قطار * وزن_dma      = octx->ctx->dma[0];
                اعشاری * dst_ptr            = (اعشاری *) dst->ڈیٹا;
                ثابت size_t n             = src_w->ne[1];
                اگر (n == 0) جاری;
                ثابت size_t وزن_stride = src_w->nb[1];
                ثابت size_t dst_stride    = dst->nb[1] / حجم(اعشاری);
                ثابت صحیح dst_ستون_تعداد         = (صحیح) dst->ne[0];
                ثابت صحیح n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_ستون_تعداد);

                ثابت uint32_t dma_src_stride = is_tiled ? tile_سائز : وزن_stride;

                ثابت size_t   n_ستون_تعداد_A0 = hex_smin(n - 0 * n_chunk_n_ستون_تعداد, n_chunk_n_ستون_تعداد);
                ثابت uint32_t اونچائی_A0 = is_tiled ? hmx_ceil_div(n_ستون_تعداد_A0, 32) * n_وزن_tiles : n_ستون_تعداد_A0;
                dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[0], وزن),
                               dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی_A0);

                اگر (1 < n_chunk_cnt) {
                    ثابت size_t   n_ستون_تعداد_A1 = hex_smin(n - 1 * n_chunk_n_ستون_تعداد, n_chunk_n_ستون_تعداد);
                    ثابت uint32_t اونچائی_A1 = is_tiled ? hmx_ceil_div(n_ستون_تعداد_A1, 32) * n_وزن_tiles : n_ستون_تعداد_A1;
                    dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[1], وزن + n_chunk_n_ستون_تعداد * وزن_stride),
                                   dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی_A1);
                }

                برائے (صحیح i = 0; i < n_chunk_cnt; ++i) {
                    ثابت size_t nc    = i * n_chunk_n_ستون_تعداد;
                    ثابت size_t nc_p2 = nc + 2 * n_chunk_n_ستون_تعداد;

                    ثابت size_t n_ستون_تعداد    = hex_smin(n - nc, n_chunk_n_ستون_تعداد);
                    ثابت size_t n_ستون_تعداد_p2 = hex_smin(n - nc_p2, n_chunk_n_ستون_تعداد);

                    خالی * curr_raw = (خالی *) dma_قطار_نکالو(وزن_dma).dst;

                    dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
                        ctx, vtcm_وزن_bufs[i % 4], curr_raw,
                        n_ستون_تعداد, k, صف_stride, وزن_type,
                        n_k_tiles, n_k_tiles_div, dequant_کارکن_fn, n_دھاگے);

                    اگر (i + 2 < n_chunk_cnt) {
                        ثابت uint32_t اونچائی_p2 = is_tiled ? hmx_ceil_div(n_ستون_تعداد_p2, 32) * n_وزن_tiles : n_ستون_تعداد_p2;
                        dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[(i + 2) % 4], وزن + nc_p2 * وزن_stride),
                                       dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی_p2);
                    }

                    hmx_matmul_کام_init(&کام_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_صفیں, HTP_MM_HMX_TILE_N_صفیں),
                                        hmx_ceil_div(n_ستون_تعداد, HTP_MM_HMX_TILE_N_ستون_تعداد), k / HTP_MM_HMX_TILE_N_صفیں);
                    hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کارکن_fn, &کام_slots[i % 2]));

                    اگر (i > 0) {
                        hmx_قطار_نکالو(ctx->hmx_قطار);
                        ثابت size_t nc_پچھلا = (i - 1) * n_chunk_n_ستون_تعداد;
                        ثابت size_t n_ستون_تعداد_پچھلا = hex_smin(n - nc_پچھلا, n_chunk_n_ستون_تعداد);
                        اعشاری *output_chunk = dst_ptr + (mr * dst_stride + nc_پچھلا);
                        صحیح chunk_dst_ستون_تعداد = dst_ستون_تعداد - (صحیح)nc_پچھلا;
                        اگر (chunk_dst_ستون_تعداد > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, تہی, vtcm_output_bufs[(i - 1) % 2], n_صفیں, n_ستون_تعداد_پچھلا, dst_stride, 0, chunk_dst_ستون_تعداد, n_دھاگے);
                        }
                    }
                }

                hmx_قطار_نکالو(ctx->hmx_قطار);
                ثابت size_t nc_آخری = (n_chunk_cnt - 1) * n_chunk_n_ستون_تعداد;
                ثابت size_t n_ستون_تعداد_آخری = hex_smin(n - nc_آخری, n_chunk_n_ستون_تعداد);
                اعشاری *output_chunk = dst_ptr + (mr * dst_stride + nc_آخری);
                صحیح chunk_dst_ستون_تعداد = dst_ستون_تعداد - (صحیح)nc_آخری;
                اگر (chunk_dst_ستون_تعداد > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, تہی, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_صفیں, n_ستون_تعداد_آخری, dst_stride, 0, chunk_dst_ستون_تعداد, n_دھاگے);
                }
            }
        }
    } ورنہ {
        hmx_matmul_کام_t کام;
        خالی *vtcm_raw_bufs[2] = { vtcm_وزن_raw[0], dequant_کارکن_fn ? vtcm_وزن_raw[0] : vtcm_scratch0 };
        خالی *vtcm_وزن_bufs[2] = { dequant_کارکن_fn ? vtcm_scratch0 : vtcm_وزن_raw[0], vtcm_scratch0 };

        برائے (size_t mr = (size_t) m_آغاز; mr < mr_انتہا; mr += m_chunk_n_صفیں) {
            ثابت size_t n_صفیں = hex_smin(mr_انتہا - mr, m_chunk_n_صفیں);

            ساخت activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_سائز,
                .n_صفیں = (صحیح) n_صفیں,
                .k_بلاک = k,
                .k_stride = act_stride,
                .n_دھاگے = act_دھاگے,
                .act_دھاگے_div = act_دھاگے_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_بائٹیں = L.act_f32_بائٹیں,
                .act_elem_سائز = act_elem_سائز,
            };
            transfer_activation_chunk_threaded(&act_params);

            برائے (uint32_t p = 0; p < n_weights; p++) {
                ثابت ساخت htp_tensor * محدود src_w = octx->src[p];
                ثابت ساخت htp_tensor * محدود dst   = octx->dsts[p];
                اگر (!src_w || !dst) جاری;

                ثابت dma_addr_t وزن       = src_w->ڈیٹا;
                dma_قطار * وزن_dma      = octx->ctx->dma[0];
                اعشاری * dst_ptr            = (اعشاری *) dst->ڈیٹا;
                ثابت size_t n             = src_w->ne[1];
                اگر (n == 0) جاری;
                ثابت size_t وزن_stride = src_w->nb[1];
                ثابت size_t dst_stride    = dst->nb[1] / حجم(اعشاری);
                ثابت صحیح dst_ستون_تعداد         = (صحیح) dst->ne[0];
                ثابت صحیح n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_ستون_تعداد);

                ثابت uint32_t dma_src_stride = is_tiled ? tile_سائز : وزن_stride;

                اگر (n > 0) {
                    ثابت size_t n_ستون_تعداد = hex_smin(n, n_chunk_n_ستون_تعداد);
                    ثابت uint32_t اونچائی = is_tiled ? hmx_ceil_div(n_ستون_تعداد, 32) * n_وزن_tiles : n_ستون_تعداد;
                    dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[0], وزن), dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی);
                }

                برائے (صحیح i = 0; i < n_chunk_cnt; ++i) {
                    ثابت size_t nc = (size_t) i * n_chunk_n_ستون_تعداد;
                    ثابت size_t n_ستون_تعداد = hex_smin(n - nc, n_chunk_n_ستون_تعداد);
                    ثابت size_t n_صف_tiles = hmx_ceil_div(n_صفیں, HTP_MM_HMX_TILE_N_صفیں);
                    ثابت size_t n_ستون_tiles = hmx_ceil_div(n_ستون_تعداد, HTP_MM_HMX_TILE_N_ستون_تعداد);

                    خالی * curr_raw = (خالی *) dma_قطار_نکالو(وزن_dma).dst;

                    dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
                        ctx, vtcm_وزن_bufs[i % 2], curr_raw,
                        n_ستون_تعداد, k, صف_stride, وزن_type,
                        n_k_tiles, n_k_tiles_div, dequant_کارکن_fn, n_دھاگے);

                    ثابت size_t nc_اگلا = nc + n_chunk_n_ستون_تعداد;
                    اگر (nc_اگلا < n) {
                        ثابت size_t n_ستون_تعداد_اگلا = hex_smin(n - nc_اگلا, n_chunk_n_ستون_تعداد);
                        ثابت uint32_t اونچائی_اگلا = is_tiled ? hmx_ceil_div(n_ستون_تعداد_اگلا, 32) * n_وزن_tiles : n_ستون_تعداد_اگلا;
                        dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[(i + 1) % 2], وزن + nc_اگلا * وزن_stride), dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی_اگلا);
                    }

                    hmx_matmul_کام_init(&کام, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 2], vtcm_scales, n_صف_tiles, n_ستون_tiles, k / HTP_MM_HMX_TILE_N_صفیں);
                    hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کارکن_fn, &کام));
                    hmx_قطار_نکالو(ctx->hmx_قطار);

                    اعشاری *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    صحیح chunk_dst_ستون_تعداد = dst_ستون_تعداد - (صحیح)nc;
                    اگر (chunk_dst_ستون_تعداد > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, تہی, vtcm_output, n_صفیں, n_ستون_تعداد, dst_stride, 0, chunk_dst_ستون_تعداد, n_دھاگے);
                    }
                }
            }
        }
    }

    واپس HTP_حالت_OK;
}

ساکن اندرونی dma_addr_t hmx_mm_وزن_batch_ڈیٹا(ثابت hmx_mm_batched_params_t *params,
                                                         صحیح dst_b2, صحیح dst_b3) {
    ثابت size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    ثابت size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    واپس params->وزن + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

ساکن اندرونی dma_addr_t hmx_mm_act_batch_addr(ثابت hmx_mm_batched_params_t *params,
                                                صحیح dst_b2, صحیح dst_b3) {
    واپس params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

ساکن اندرونی اعشاری *hmx_mm_dst_batch_ptr(ثابت hmx_mm_batched_params_t *params,
                                              صحیح dst_b2, صحیح dst_b3) {
    واپس (اعشاری *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

ساکن صحیح hmx_mm_batched_simple(ساخت htp_context *ctx,
                                 ثابت hmx_mm_batched_params_t *params,
                                 صحیح m_chunk, صحیح n_chunk, صحیح pipeline, صحیح n_دھاگے, صحیح act_دھاگے, صحیح vtcm_سائز,
                                 ثابت ساخت fastdiv_اقدار * act_دھاگے_div, ثابت ساخت fastdiv_اقدار * k_div) {
    صحیح ret = 0;
    برائے (صحیح b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        برائے (صحیح b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->وزن_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_بائٹیں,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_وزن_batch_ڈیٹا(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_سائز, params->وزن_stride,
                                params->وزن_type, params->k, params->dst_stride, params->bias_stride, params->dst_ستون_تعداد,
                                m_chunk, n_chunk, pipeline, n_دھاگے, act_دھاگے,
                                act_دھاگے_div, k_div, params->tile_سائز, params->aligned_tile_سائز, vtcm_سائز);
        }
    }
    واپس ret;
}

ساکن صحیح hmx_mm_batched(ساخت htp_context *ctx, ثابت hmx_mm_batched_params_t *params,
                          صحیح m_chunk, صحیح n_chunk, صحیح pipeline, صحیح n_دھاگے, صحیح act_دھاگے,
                          ثابت ساخت fastdiv_اقدار * act_دھاگے_div,
                          ثابت ساخت fastdiv_اقدار * k_div,
                          صحیح vtcm_سائز) {
    ثابت size_t صف_stride = htp_tensor_get_صف_سائز(params->وزن_type, params->k);
    اگر (صف_stride == 0) { واپس -1; }
    اگر (params->act_stride < params->k || (params->tile_سائز == 0 && (size_t) params->وزن_stride < صف_stride) || params->dst_stride < params->dst_ستون_تعداد) { واپس -1; }
    اگر (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { واپس -1; }
    اگر (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { واپس -1; }
    // N (the weight row count) does not have to be 32-aligned:
    اگر (params->k % 32 != 0) { واپس -1; }
    اگر (!hex_is_aligned(params->dst, حجم(اعشاری))) { واپس -1; }

    ثابت صحیح گروہ_سائز = params->r2;
    ثابت size_t vtcm_budget  = ctx->vtcm_سائز;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool تسلسل_grouped = (params->وزن_type == HTP_TYPE_F16 && params->tile_سائز == 0 && گروہ_سائز > 1 && (size_t)vtcm_سائز <= vtcm_budget);
    اگر (!تسلسل_grouped) {
        واپس hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_دھاگے, act_دھاگے, vtcm_سائز, act_دھاگے_div, k_div);
    }

    ساخت htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_صفیں = m_chunk;
    size_t n_chunk_n_ستون_تعداد = n_chunk;

    ساخت htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_صفیں, n_chunk_n_ستون_تعداد, گروہ_سائز, false, act_دھاگے, 0, params->bias_بائٹیں);

    اگر (L.کل_بائٹیں > vtcm_budget) {
        FARF(بلند_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        واپس hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_دھاگے, act_دھاگے, vtcm_سائز, act_دھاگے_div, k_div);
    }

    uint8_t * ثابت بنیاد = (uint8_t *) ctx->vtcm_بنیاد;
    __fp16  *vtcm_وزن     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_وزن[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_dst[0]);
    خالی    *vtcm_scratch0   = VTCM_LAYOUT_PTR(خالی, بنیاد, L.off_scratch[0]);
    خالی    *vtcm_scratch1   = VTCM_LAYOUT_PTR(خالی, بنیاد, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_scales);
    اعشاری   *vtcm_f32_act    = VTCM_LAYOUT_PTR(اعشاری, بنیاد, L.off_act_f32);

    ثابت bool has_bias      = (params->bias_بائٹیں > 0 && params->bias_addr != 0);
    اعشاری   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(اعشاری, بنیاد, L.off_bias, has_bias);
    اگر (has_bias) {
        dma_قطار_ڈالو(params->وزن_dma, dma_make_ڈیٹا(vtcm_bias, params->bias_addr), hex_align_اوپر(params->bias_بائٹیں, 128), 0, params->bias_بائٹیں, 1);
        dma_قطار_نکالو(params->وزن_dma);
    }

    hmx_init_عمود_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(بلند_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, گروہ_سائز, params->ne13,
            m_chunk_n_صفیں, n_chunk_n_ستون_تعداد,
            L.کل_بائٹیں, vtcm_budget);

    ثابت size_t fp16_صف_بائٹیں   = (size_t) params->k * حجم(__fp16);
    ثابت size_t وزن_صف_بائٹیں = (size_t) params->وزن_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_کام_t کام;

    برائے (صحیح b3 = 0; b3 < params->ne13; ++b3) {
        برائے (صحیح b2_بنیاد = 0; b2_بنیاد < params->ne12; b2_بنیاد += گروہ_سائز) {
            ثابت dma_addr_t وزن_گروہ = hmx_mm_وزن_batch_ڈیٹا(params, b2_بنیاد, b3);
            dma_قطار * وزن_dma = params->وزن_dma;

            برائے (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_صفیں) {
                ثابت size_t n_صفیں = hex_smin((size_t) params->m - mr, m_chunk_n_صفیں);
                ثابت size_t n_صف_tiles = hmx_ceil_div((صحیح) n_صفیں, HTP_MM_HMX_TILE_N_صفیں);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                برائے (صحیح g = 0; g < گروہ_سائز; ++g) {
                    ثابت dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_بنیاد + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_سائز;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_سر_stride;
                    ساخت activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_صفیں = (صحیح) n_صفیں,
                        .k_بلاک = params->k,
                        .k_stride = params->act_stride,
                        .n_دھاگے = act_دھاگے,
                        .act_دھاگے_div = act_دھاگے_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_بائٹیں = L.act_f32_بائٹیں,
                        .act_elem_سائز = params->act_elem_سائز,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    ثابت size_t n_ستون_تعداد_پہلا = hex_smin((size_t) params->n, n_chunk_n_ستون_تعداد);
                    dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_scratch0, وزن_گروہ),
                                      fp16_صف_بائٹیں, وزن_صف_بائٹیں, fp16_صف_بائٹیں, n_ستون_تعداد_پہلا);
                }
                اگر (n_chunk_n_ستون_تعداد < (size_t) params->n) {
                    ثابت size_t n_ستون_تعداد_دوسرا = hex_smin((size_t) params->n - n_chunk_n_ستون_تعداد, n_chunk_n_ستون_تعداد);
                    ثابت dma_addr_t دوسرا_وزن_chunk = وزن_گروہ + n_chunk_n_ستون_تعداد * وزن_صف_بائٹیں;
                    dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_scratch1, دوسرا_وزن_chunk),
                                      fp16_صف_بائٹیں, وزن_صف_بائٹیں, fp16_صف_بائٹیں, n_ستون_تعداد_دوسرا);
                }

                برائے (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_ستون_تعداد) {
                    ثابت size_t n_ستون_تعداد      = hex_smin((size_t) params->n - nc, n_chunk_n_ستون_تعداد);
                    ثابت size_t n_ستون_tiles = hmx_ceil_div((صحیح) n_ستون_تعداد, HTP_MM_HMX_TILE_N_ستون_تعداد);

                    {
                        خالی * curr_raw = (خالی *) dma_قطار_نکالو(وزن_dma).dst;

                        ثابت size_t n_ستون_تعداد_tiled = hex_align_اوپر(n_ستون_تعداد, HTP_MM_HMX_TILE_N_ستون_تعداد);

                        hmx_interleave_صفیں_مقصد_tiles(vtcm_وزن, (ثابت __fp16 *) curr_raw, (uint32_t) n_ستون_تعداد, params->k, params->k, 0, (uint32_t) n_ستون_تعداد_tiled);

                        ثابت size_t nc_اگلا = nc + n_chunk_n_ستون_تعداد * 2;
                        اگر (nc_اگلا < (size_t) params->n) {
                            ثابت size_t n_ستون_تعداد_اگلا = hex_smin((size_t) params->n - nc_اگلا, n_chunk_n_ستون_تعداد);
                            ثابت dma_addr_t اگلا_وزن_chunk = وزن_گروہ + nc_اگلا * وزن_صف_بائٹیں;

                            dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(curr_raw, اگلا_وزن_chunk),
                                              fp16_صف_بائٹیں, وزن_صف_بائٹیں, fp16_صف_بائٹیں, n_ستون_تعداد_اگلا);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    برائے (صحیح g = 0; g < گروہ_سائز; ++g) {
                        {
                            ثابت __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_سر_stride;
                            hmx_matmul_کام_init(&کام, vtcm_output, vtcm_act_g, vtcm_وزن, vtcm_scales, n_صف_tiles, n_ستون_tiles, params->k / 32);
                            hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کارکن_fn, &کام));
                            hmx_قطار_نکالو(ctx->hmx_قطار);
                        }

                        {
                            اعشاری *output = hmx_mm_dst_batch_ptr(params, b2_بنیاد + g, b3) + mr * params->dst_stride + nc;
                            ثابت اعشاری *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : تہی;
                            صحیح chunk_dst_ستون_تعداد = params->dst_ستون_تعداد - (صحیح)nc;
                            اگر (chunk_dst_ستون_تعداد > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (صحیح) n_صفیں, (صحیح) n_ستون_تعداد,
                                                               params->dst_stride, params->bias_stride, chunk_dst_ستون_تعداد, n_دھاگے);
                            }
                        }
                    }
                }
            }
        }
    }

    واپس 0;
}

ساکن خالی transfer_activation_chunk_gathered_threaded(
            ساخت htp_context *ctx,
            __fp16 *dst,
            ثابت اعشاری *src,
            صحیح آغاز_صف,
            صحیح n_صفیں,
            صحیح k_بلاک,
            ثابت ساخت mmid_صف_mapping *میٹرکس_صفیں,
            صحیح cur_a,
            صحیح mapping_stride,
            صحیح ne11,
            size_t nb11,
            size_t nb12,
            صحیح cne1,
            صحیح n_دھاگے,
            صحیح k_valid) {
    اگر (n_صفیں <= 0) واپس;
    صحیح chunks_فی_thread = hmx_ceil_div(n_صفیں, n_دھاگے);
    chunks_فی_thread = hex_align_اوپر(chunks_فی_thread, 2);

    صحیح actual_دھاگے = hmx_ceil_div(n_صفیں, chunks_فی_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_دھاگے,
        .n_tot_chunks      = n_صفیں,
        .n_chunks_فی_task = chunks_فی_thread,
        .k_بلاک           = k_بلاک,
        .میٹرکس_صفیں       = میٹرکس_صفیں,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_اقدار(ne11) : (ساخت fastdiv_اقدار){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .آغاز_صف         = آغاز_صف,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    کارکن_callback_t کارکن_fn = ne11 == 1 ? transfer_activation_chunk_gathered_کارکن_flat_fn :
                                              transfer_activation_chunk_gathered_کارکن_fn;

    اگر (actual_دھاگے <= 1) {
        کارکن_fn(1, 0, &state);
    } ورنہ {
        کارکن_pool_تسلسل_func(ctx->کارکن_pool, کارکن_fn, &state, actual_دھاگے);
    }
}

ساکن خالی transfer_output_chunk_scattered_threaded(
            ساخت htp_context *ctx,
            اعشاری *dst,
            ثابت __fp16 *vtcm_src,
            صحیح آغاز_صف,
            صحیح n_صفیں,
            صحیح n_ستون_تعداد,
            ثابت ساخت mmid_صف_mapping *میٹرکس_صفیں,
            صحیح cur_a,
            صحیح mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            صحیح cne1,
            صحیح n_دھاگے) {
    اگر (n_صفیں <= 0) واپس;
    صحیح chunks_فی_thread = hmx_ceil_div(n_صفیں, n_دھاگے);
    chunks_فی_thread = hex_align_اوپر(chunks_فی_thread, 2);

    صحیح actual_دھاگے = hmx_ceil_div(n_صفیں, chunks_فی_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_دھاگے,
        .n_tot_chunks      = n_صفیں,
        .n_chunks_فی_task = chunks_فی_thread,
        .n_ستون_تعداد            = n_ستون_تعداد,
        .میٹرکس_صفیں       = میٹرکس_صفیں,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .آغاز_صف         = آغاز_صف,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    اگر (actual_دھاگے <= 1) {
        transfer_output_chunk_scattered_کارکن_fn(1, 0, &state);
    } ورنہ {
        کارکن_pool_تسلسل_func(ctx->کارکن_pool, transfer_output_chunk_scattered_کارکن_fn, &state, actual_دھاگے);
    }
}

ساکن صحیح hmx_mm_id_2d_f32(ساخت htp_context *ctx,
                                         dma_قطار *وزن_dma,
                                         اعشاری *محدود dst,
                                         ثابت اعشاری *activation,
                                         dma_addr_t وزن,
                                         صحیح m, صحیح k, صحیح n,
                                         صحیح k_valid,
                                         صحیح ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         صحیح وزن_stride,
                                         صحیح وزن_type,
                                         صحیح tile_سائز,
                                         صحیح aligned_tile_سائز,
                                         ثابت ساخت mmid_صف_mapping *میٹرکس_صفیں,
                                         صحیح cur_a,
                                         صحیح mapping_stride,
                                         صحیح m_آغاز,
                                         صحیح m_انتہا,
                                         صحیح n_دھاگے) {
    ساخت htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت صحیح cne1 = m;
    ثابت صحیح m_core = m_انتہا - m_آغاز;
    ثابت صحیح m_core_padded = hex_align_اوپر(m_core > 0 ? m_core : 1, 32);

    اگر (k % 32 != 0 || n % 32 != 0) { واپس -1; }
    اگر (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { واپس -1; }

    size_t صف_stride = htp_tensor_get_صف_سائز(وزن_type, k);
    اگر (صف_stride == 0) {
        واپس -1;
    }

    ثابت صحیح n_k_tiles = k / HTP_MM_HMX_TILE_N_ستون_تعداد;
    ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(وزن_type);
    ثابت ساخت fastdiv_اقدار n_k_tiles_div = init_fastdiv_اقدار(n_k_tiles);

    ثابت size_t vec_dot_سائز = k * حجم(__fp16);
    ثابت size_t vtcm_budget  = ctx->vtcm_سائز;
    size_t vtcm_مستعمل = 0;

    ثابت bool is_tiled = (tile_سائز > 0);
    tiled_dequant_task_fn_t dequant_کارکن_fn = hmx_get_dequant_کارکن_fn(وزن_type, is_tiled);

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_سائز : صف_stride;
    ثابت uint32_t dma_src_stride  = is_tiled ? tile_سائز : وزن_stride;
    ثابت uint32_t dma_چوڑائی_بائٹیں = is_tiled ? tile_سائز : صف_stride;

    ثابت size_t qweight_صف_stride = is_tiled ? (size_t)(n_وزن_tiles * aligned_tile_سائز) / 32 : 0;
    ثابت size_t وزن_صف_stride = is_tiled ? qweight_صف_stride : صف_stride;

    size_t سائز_فی_n = 0, سائز_فی_m = 0, سائز_فی_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(وزن_type, k, /*pipeline=*/false, aligned_tile_سائز,
                                  &سائز_فی_n, &سائز_فی_m, &سائز_فی_mn);

    ثابت size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_صفیں = 0, n_chunk_n_ستون_تعداد = 0;
    اگر (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, سائز_فی_n, سائز_فی_m, سائز_فی_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_لاگت_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_لاگت_A_CONVERT, &m_chunk_n_صفیں, &n_chunk_n_ستون_تعداد, &vtcm_مستعمل)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        واپس -1;
    }

    ثابت size_t وزن_area_سائز = hex_align_اوپر(n_chunk_n_ستون_تعداد * وزن_صف_stride, HTP_MM_HMX_TILE_سائز);
    ثابت size_t act_area_سائز    = hex_align_اوپر(m_chunk_n_صفیں * vec_dot_سائز, HTP_MM_HMX_TILE_سائز);
    ثابت size_t output_area_سائز = hex_align_اوپر(m_chunk_n_صفیں * n_chunk_n_ستون_تعداد * حجم(__fp16), HTP_MM_HMX_TILE_سائز);

    size_t scratch0_سائز = hex_align_اوپر(n_chunk_n_ستون_تعداد * vec_dot_سائز, HTP_MM_HMX_TILE_سائز);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_بنیاد;
    __fp16  *vtcm_وزن   = وزن_area_سائز ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, وزن_area_سائز) : تہی;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_سائز);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_سائز);
    خالی    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_سائز);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_مستعمل = vtcm_ptr - (uint8_t *) ctx->vtcm_بنیاد;
    اگر (vtcm_مستعمل > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_مستعمل, vtcm_budget);
        واپس -1;
    }

    hmx_init_عمود_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_کام_t کام;
    خالی *vtcm_raw_bufs[2] = { vtcm_وزن, dequant_کارکن_fn ? vtcm_وزن : vtcm_scratch0 };
    خالی *vtcm_وزن_bufs[2] = { dequant_کارکن_fn ? vtcm_scratch0 : vtcm_وزن, vtcm_scratch0 };
    ثابت صحیح n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_ستون_تعداد);

    برائے (size_t mr = (size_t) m_آغاز; mr < (size_t) m_انتہا; mr += m_chunk_n_صفیں) {
        ثابت size_t n_صفیں = hex_smin((size_t) m_انتہا - mr, m_chunk_n_صفیں);
        ثابت size_t n_صف_tiles = hmx_ceil_div(n_صفیں, HTP_MM_HMX_TILE_N_صفیں);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (صحیح) mr, (صحیح) n_صفیں, k,
            میٹرکس_صفیں, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_دھاگے, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        اگر (n > 0) {
            ثابت size_t n_ستون_تعداد = hex_smin((size_t) n, n_chunk_n_ستون_تعداد);
            ثابت uint32_t اونچائی = is_tiled ? hmx_ceil_div(n_ستون_تعداد, 32) * n_وزن_tiles : n_ستون_تعداد;
            dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[0], وزن),
                           dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی);
        }

        برائے (صحیح i = 0; i < n_chunk_cnt; ++i) {
            ثابت size_t nc = (size_t) i * n_chunk_n_ستون_تعداد;
            ثابت size_t n_ستون_تعداد = hex_smin((size_t) n - nc, n_chunk_n_ستون_تعداد);
            ثابت size_t n_ستون_tiles = hmx_ceil_div(n_ستون_تعداد, HTP_MM_HMX_TILE_N_ستون_تعداد);

            // A: Wait for weight DMA
            خالی * curr_raw = (خالی *) dma_قطار_نکالو(وزن_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_وزن_chunk_مقصد_fp16_tiles(
                ctx, vtcm_وزن_bufs[i % 2], curr_raw,
                n_ستون_تعداد, k, صف_stride, وزن_type,
                n_k_tiles, n_k_tiles_div, dequant_کارکن_fn, n_دھاگے
            );

            // Start weight DMA for the next chunk early
            ثابت size_t nc_اگلا = nc + n_chunk_n_ستون_تعداد;
            اگر (nc_اگلا < (size_t) n) {
                ثابت size_t n_ستون_تعداد_اگلا = hex_smin((size_t) n - nc_اگلا, n_chunk_n_ستون_تعداد);
                ثابت uint32_t اونچائی_اگلا = is_tiled ? hmx_ceil_div(n_ستون_تعداد_اگلا, 32) * n_وزن_tiles : n_ستون_تعداد_اگلا;
                dma_قطار_ڈالو(وزن_dma, dma_make_ڈیٹا(vtcm_raw_bufs[(i + 1) % 2], وزن + nc_اگلا * وزن_stride),
                               dma_dst_stride, dma_src_stride, dma_چوڑائی_بائٹیں, اونچائی_اگلا);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_کام_init(&کام, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_وزن_bufs[i % 2], vtcm_scales, n_صف_tiles, n_ستون_tiles, k / HTP_MM_HMX_TILE_N_صفیں);
            hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_matmul_کارکن_fn, &کام));
            hmx_قطار_نکالو(ctx->hmx_قطار);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (صحیح) mr, (صحیح) n_صفیں, (صحیح) n_ستون_تعداد,
                میٹرکس_صفیں, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_دھاگے);
        }
    }

    واپس 0;
}

// --- Dispatchers and Public Entry Points ---

ساکن صحیح hmx_mm_op_matmul(ساخت htp_ops_context * octx, ثابت ساخت htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    صحیح k = (صحیح) src0->ne[0];
    صحیح n = (صحیح) src0->ne[1];
    ثابت صحیح m_کل    = (صحیح) act->ne[1];
    ثابت uint32_t act_elem_سائز = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(اعشاری);
    ثابت صحیح act_stride = (صحیح) (act->nb[1] / act_elem_سائز);

    صحیح m_آغاز = 0;
    صحیح m_صفیں  = m_کل;
    اگر (octx->ctx->mdev.گنتی > 1) {
        bool can_بانٹو = htp_tensor_can_صف_تقسیم(dst, حجم(اعشاری)) &&
                         htp_tensor_can_صف_تقسیم(act, act_elem_سائز);
        اگر (src2 && src2->ne[1] > 1 && !htp_tensor_can_صف_تقسیم(src2, حجم(اعشاری))) {
            can_بانٹو = false;
        }
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم((uint32_t) m_کل, can_بانٹو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        m_آغاز = (صحیح) range.آغاز;
        m_صفیں  = (صحیح) range.گنتی;
    }

    اگر (m_صفیں == 0) {
        واپس HTP_حالت_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_بائٹیں = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    اگر (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / حجم(اعشاری));
        src2_addr = src2->ڈیٹا + m_آغاز * src2_stride * حجم(اعشاری);
        src2_بائٹیں = (size_t) kparams->vtcm_bias_سائز;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ثابت صحیح dst_stride = (صحیح)(dst->nb[1] / حجم(اعشاری));
    اعشاری       * dst_ptr = (اعشاری *) dst->ڈیٹا + m_آغاز * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ثابت dma_addr_t act_addr = act->ڈیٹا + (size_t) m_آغاز * act_stride * act_elem_سائز;

    صحیح ret = -1;
    ثابت صحیح n_دھاگے = kparams->n_دھاگے;
    اگر (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        hmx_mm_batched_params_t batch_params = {
            .dst               = dst_ptr,
            .bias_addr         = src2_addr,
            .bias_بائٹیں        = src2_بائٹیں,
            .act_dma_addr      = act_addr,
            .وزن            = src0->ڈیٹا,
            .وزن_dma        = octx->ctx->dma[0],
            .m                 = m_صفیں,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_سائز     = act_elem_سائز,
            .وزن_stride     = (صحیح) src0->nb[1],
            .وزن_type       = (صحیح) src0->type,
            .dst_stride        = dst_stride,
            .dst_ستون_تعداد          = (صحیح) dst->ne[0],
            .bias_stride       = src2_stride,
            .tile_سائز         = kparams->tile_سائز,
            .aligned_tile_سائز = kparams->aligned_tile_سائز,
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
                             kparams->vtcm_سائز);
    } ورنہ {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_بائٹیں,
            act_addr, src0->ڈیٹا,
            m_صفیں, k, n, act_stride, act_elem_سائز, (صحیح) src0->nb[1], (صحیح) src0->type, (صحیح) act->ne[0],
            dst_stride, src2_stride, (صحیح)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_دھاگے,
            kparams->n_act_دھاگے,
            &kparams->div_n_act_دھاگے,
            &kparams->div_ne00_padded,
            kparams->tile_سائز, kparams->aligned_tile_سائز, kparams->vtcm_سائز
        );
    }

    اگر (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        واپس HTP_حالت_اندرونی_ERR;
    }
    واپس HTP_حالت_OK;
}

ساکن اندرونی خالی htp_mm_tensor_collapse_صفیں(ساخت htp_tensor * c, ثابت ساخت htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

ساکن صحیح op_matmul_impl(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;

    ثابت صحیح حالت_2 = htp_mm_init_context(octx, kparams);
    اگر (حالت_2 != HTP_حالت_OK) {
        واپس حالت_2;
    }

    اگر (kparams->n_hmx) {
        واپس hmx_mm_op_matmul(octx, kparams);
    }

    واپس hvx_mm_matmul(octx);
}

صحیح op_matmul(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;

    اگر (kparams->collapse) {
        ثابت ساخت htp_tensor * act  = octx->src[1];
        ثابت ساخت htp_tensor * dst  = octx->dst;
        ثابت uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ثابت uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        ساخت htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_صفیں(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_صفیں(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        ساخت htp_tensor src2_collapsed;
        ثابت ساخت htp_tensor * src2 = octx->src[2];
        اگر (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ثابت uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_صفیں(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        ثابت صحیح حالت_2 = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        اگر (src2) {
            octx->src[2] = src2;
        }
        واپس حالت_2;
    }

    واپس op_matmul_impl(octx);
}

ساکن صحیح hmx_mm_op_matmul_id(
    ساخت htp_ops_context * octx,
    ساخت htp_mm_context * mmctx
) {
    ثابت uint32_t *                میٹرکس_صف_counts = mmctx->میٹرکس_صف_counts;
    ثابت ساخت mmid_صف_mapping * میٹرکس_صفیں       = mmctx->میٹرکس_صفیں;
    htp_matmul_tensors_preamble;
    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت صحیح n_ids = octx->src[2]->ne[0];
    ثابت صحیح n_as  = ne02;

    ثابت bool mdev_بانٹو = (octx->ctx->mdev.گنتی > 1) && htp_tensor_can_صف_تقسیم(dst, حجم(اعشاری));
    اگر (octx->ctx->mdev.گنتی > 1 && !mdev_بانٹو && octx->ctx->mdev.idx > 0) {
        واپس HTP_حالت_OK;
    }
    uint32_t n_فعال = 0;
    اگر (mdev_بانٹو) {
        برائے (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            اگر (میٹرکس_صف_counts[a] > 0) n_فعال++;
        }
    }
    ثابت bool expert_بانٹو = mdev_بانٹو && (n_فعال >= octx->ctx->mdev.گنتی);

    uint32_t ہدف_dev = 0;
    برائے (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ثابت int32_t cne1 = میٹرکس_صف_counts[cur_a];
        اگر (cne1 == 0) جاری;

        ثابت صحیح m_padded = hex_align_اوپر(cne1, 32);
        صحیح m_آغاز = 0, m_انتہا = m_padded;
        اگر (expert_بانٹو) {
            ثابت bool my_expert = (ہدف_dev == octx->ctx->mdev.idx);
            اگر (++ہدف_dev == octx->ctx->mdev.گنتی) {
                ہدف_dev = 0;
            }
            اگر (!my_expert) جاری;
        } ورنہ اگر (mdev_بانٹو) {
            ثابت bool can_بانٹو = (uint32_t) cne1 >= octx->ctx->mdev.گنتی;
            ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم((uint32_t) m_padded, can_بانٹو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            m_آغاز = (صحیح) range.آغاز;
            m_انتہا   = (صحیح) (range.آغاز + range.گنتی);
        }
        اگر (m_آغاز >= m_انتہا) جاری;

        صحیح ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (اعشاری*) dst->ڈیٹا, (اعشاری*) act->ڈیٹا,
                                   src0->ڈیٹا + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (صحیح) src0->nb[1], (صحیح) src0->type,
                                   kparams->tile_سائز, kparams->aligned_tile_سائز,
                                   میٹرکس_صفیں, cur_a, mmctx->mapping_stride,
                                   m_آغاز, m_انتہا, (صحیح) octx->n_دھاگے);
        اگر (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            واپس HTP_حالت_NO_SUPPORT;
        }
    }

    واپس HTP_حالت_OK;
}

ساکن صحیح hvx_mm_matmul_id(
    ساخت htp_ops_context * octx,
    ساخت htp_mm_context * mmctx,
    work_قطار_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ثابت uint32_t src0_صف_سائز_padded = mmctx->src0_صف_سائز_padded;
    ثابت uint32_t act_nrows            = mmctx->act_nrows;

    ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت ساخت htp_tensor * محدود ids = octx->src[2];
    ثابت size_t src0_صف_سائز = nb01;

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (ne10 + qk - 1) / qk;
    ثابت uint32_t کل_nb = act_nrows * nb;

    work_قطار_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    اگر (act_nrows < octx->n_دھاگے) {
        n_quant_tasks = MIN(کل_nb, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
        برائے (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_پہلا = (کل_nb * ith) / n_quant_tasks;
            uint32_t ib_آخری  = (کل_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_پہلا[ith] = ib_پہلا;
            mmctx->quant_ib_آخری[ith]  = ib_آخری;
            mmctx->quant_r[ith]        = ib_پہلا / nb;
            mmctx->quant_c[ith]        = ib_پہلا % nb;
        }
    } ورنہ {
        n_quant_tasks = MIN(act_nrows, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_صف_func(src0->type, act->type);
    }
    size_t act_صف_سائز = htp_mm_tiled_act_صف_سائز(src0->type, ne10);

    ساخت htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_دھاگے,
                                 0, src0_صف_سائز, act_صف_سائز, 0, kparams->n_prefetch, true, false);

    ثابت size_t vtcm_سائز = L.کل_بائٹیں;

    FARF(بلند_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_بائٹیں, L.act_بائٹیں, L.dst_بائٹیں, vtcm_سائز);

    FARF(بلند_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ڈیٹا,
         act->ڈیٹا, dst->ڈیٹا);

    // Make sure the reserved vtcm size is sufficient
    اگر (octx->ctx->vtcm_سائز < vtcm_سائز) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_سائز, vtcm_سائز);
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_src0);
    mmctx->vtcm_bias     = تہی;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_صف_سائز_padded;
    mmctx->vtcm_act_stride     = act_صف_سائز;
    mmctx->vtcm_act_raw_stride = hex_باری_اوپر(ne10 * حجم(اعشاری), QK_Q8_0_TILED * حجم(اعشاری));

    mmctx->vtcm_src0_سائز_فی_thread = fastdiv(L.src0_بائٹیں, &octx->n_دھاگے_div);
    mmctx->vtcm_act_سائز_فی_thread  = L.act_بائٹیں;
    mmctx->vtcm_dst_سائز_فی_thread  = fastdiv(L.dst_بائٹیں, &octx->n_دھاگے_div);

    mmctx->cur_m_آغاز = 0;
    mmctx->cur_m_صفیں  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_صفیں_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_قطار_تسلسل(octx->ctx->work_قطار, quant_task_func, mmctx, n_quant_tasks);

    work_قطار_تسلسل(octx->ctx->work_قطار, hvx_mmid_task_func, mmctx, octx->n_دھاگے);

    واپس HTP_حالت_OK;
}

ساکن صحیح hmx_mm_op_matmul_id_nx(
    ساخت htp_ops_context * octx,
    ساخت htp_mm_context * mmctx
) {
    ثابت uint32_t *                میٹرکس_صف_counts = mmctx->میٹرکس_صف_counts;
    ثابت ساخت mmid_صف_mapping * میٹرکس_صفیں       = mmctx->میٹرکس_صفیں;
    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت ساخت htp_tensor * محدود src0 = octx->src[0];
    ثابت ساخت htp_tensor * محدود act  = octx->src[n_weights];
    ثابت صحیح n_as = src0->ne[2];

    bool mdev_بانٹو = (octx->ctx->mdev.گنتی > 1);
    برائے (uint32_t p = 0; p < n_weights && mdev_بانٹو; ++p) {
        ثابت ساخت htp_tensor * محدود dst = octx->dsts[p];
        mdev_بانٹو = htp_tensor_can_صف_تقسیم(dst, حجم(اعشاری));
    }
    اگر (octx->ctx->mdev.گنتی > 1 && !mdev_بانٹو && octx->ctx->mdev.idx > 0) {
        واپس HTP_حالت_OK;
    }
    uint32_t n_فعال = 0;
    اگر (mdev_بانٹو) {
        برائے (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            اگر (میٹرکس_صف_counts[a] > 0) n_فعال++;
        }
    }
    ثابت bool expert_بانٹو = mdev_بانٹو && (n_فعال >= octx->ctx->mdev.گنتی);

    uint32_t ہدف_dev = 0;
    برائے (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ثابت int32_t cne1 = میٹرکس_صف_counts[cur_a];
        اگر (cne1 == 0) جاری;

        ثابت صحیح m_padded = hex_align_اوپر(cne1, 32);
        صحیح m_آغاز = 0, m_انتہا = m_padded;
        اگر (expert_بانٹو) {
            ثابت bool my_expert = (ہدف_dev == octx->ctx->mdev.idx);
            اگر (++ہدف_dev == octx->ctx->mdev.گنتی) {
                ہدف_dev = 0;
            }
            اگر (!my_expert) جاری;
        } ورنہ اگر (mdev_بانٹو) {
            ثابت bool can_بانٹو = (uint32_t) cne1 >= octx->ctx->mdev.گنتی;
            ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم((uint32_t) m_padded, can_بانٹو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            m_آغاز = (صحیح) range.آغاز;
            m_انتہا   = (صحیح) (range.آغاز + range.گنتی);
        }
        اگر (m_آغاز >= m_انتہا) جاری;

        برائے (uint32_t p = 0; p < n_weights; ++p) {
            ثابت ساخت htp_tensor * محدود src_w = octx->src[p];
            ثابت ساخت htp_tensor * محدود dst   = octx->dsts[p];
            اگر (!src_w || !dst) جاری;

            صحیح ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (اعشاری*) dst->ڈیٹا, (اعشاری*) act->ڈیٹا,
                                       src_w->ڈیٹا + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (صحیح) src_w->nb[1], (صحیح) src_w->type,
                                       kparams->tile_سائز, kparams->aligned_tile_سائز,
                                       میٹرکس_صفیں, cur_a, mmctx->mapping_stride,
                                       m_آغاز, m_انتہا, (صحیح) octx->n_دھاگے);
            اگر (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                واپس HTP_حالت_NO_SUPPORT;
            }
        }
    }

    واپس HTP_حالت_OK;
}

ساکن صحیح hvx_mm_matmul_id_nx(
    ساخت htp_ops_context * octx,
    ساخت htp_mm_context * mmctx,
    work_قطار_func_t hvx_mmid_task_func
) {
    ثابت uint32_t src0_صف_سائز_padded = mmctx->src0_صف_سائز_padded;
    ثابت uint32_t act_nrows            = mmctx->act_nrows;

    ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت ساخت htp_tensor * محدود src0 = octx->src[0];
    ثابت ساخت htp_tensor * محدود act  = octx->src[n_weights];
    ثابت ساخت htp_tensor * محدود ids  = octx->src[n_weights + 1];
    ثابت size_t src0_صف_سائز = src0->nb[1];

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ثابت uint32_t کل_nb = act_nrows * nb;

    work_قطار_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    اگر (act_nrows < octx->n_دھاگے) {
        n_quant_tasks = MIN(کل_nb, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
        برائے (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_پہلا = (کل_nb * ith) / n_quant_tasks;
            uint32_t ib_آخری  = (کل_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_پہلا[ith] = ib_پہلا;
            mmctx->quant_ib_آخری[ith]  = ib_آخری;
            mmctx->quant_r[ith]        = ib_پہلا / nb;
            mmctx->quant_c[ith]        = ib_پہلا % nb;
        }
    } ورنہ {
        n_quant_tasks = MIN(act_nrows, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_صف_func(src0->type, act->type);
    }
    size_t act_صف_سائز = htp_mm_tiled_act_صف_سائز(src0->type, act->ne[0]);

    ساخت htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_دھاگے,
                                 0, src0_صف_سائز, act_صف_سائز, 0, kparams->n_prefetch, true, false);

    ثابت size_t vtcm_سائز = L.کل_بائٹیں;

    اگر (octx->ctx->vtcm_سائز < vtcm_سائز) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_سائز, vtcm_سائز);
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_صف_سائز;
    mmctx->vtcm_act_raw_stride = hex_باری_اوپر(act->ne[0] * حجم(اعشاری), QK_Q8_0_TILED * حجم(اعشاری));

    mmctx->vtcm_src0_سائز_فی_thread = fastdiv(L.src0_بائٹیں, &octx->n_دھاگے_div);
    mmctx->vtcm_act_سائز_فی_thread  = L.act_بائٹیں;
    mmctx->vtcm_dst_سائز_فی_thread  = fastdiv(L.dst_بائٹیں, &octx->n_دھاگے_div);

    mmctx->cur_m_آغاز = 0;
    mmctx->cur_m_صفیں  = act_nrows;

    FARF(بلند_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.کل_بائٹیں, octx->ctx->vtcm_سائز, octx->n_دھاگے);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_صفیں_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_قطار_تسلسل(octx->ctx->work_قطار, quant_task_func, mmctx, n_quant_tasks);

    work_قطار_تسلسل(octx->ctx->work_قطار, hvx_mmid_task_func, mmctx, octx->n_دھاگے);

    واپس HTP_حالت_OK;
}

ساکن اندرونی خالی جائزہ_expert_ids_n(
    ثابت ساخت htp_tensor * ids,
    ثابت uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    ساخت mmid_صف_mapping * میٹرکس_صفیں,
    uint32_t mapping_stride
) {
    ثابت size_t ids_nb1 = ids->nb[1];
    ثابت uint8_t * ids_ڈیٹا = (ثابت uint8_t *) ids->ڈیٹا;

    برائے (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ثابت int32_t * صف_ptr = (ثابت int32_t *) (ids_ڈیٹا + iid1 * ids_nb1);
        برائے (uint32_t id = 0; id < n_ids; ++id) {
            ثابت int32_t i02 = صف_ptr[id];
            اگر (i02 < 0) {
                جاری;
            }
            أكد(i02 < n_as);

            اگر (میٹرکس_صفیں) {
                میٹرکس_صفیں[i02 * mapping_stride + counts[i02]] = (ساخت mmid_صف_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

ساکن اندرونی خالی جائزہ_expert_ids(
    ثابت ساخت htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    ساخت mmid_صف_mapping * میٹرکس_صفیں,
    uint32_t mapping_stride
) {
    ثابت size_t ids_nb0 = ids->nb[0];

    اگر (ids_nb0 == 4) {
        انتخاب (n_ids) {
            حالت 8:  جائزہ_expert_ids_n(ids, 8,     n_as, counts, میٹرکس_صفیں, mapping_stride); توڑو;
            حالت 4:  جائزہ_expert_ids_n(ids, 4,     n_as, counts, میٹرکس_صفیں, mapping_stride); توڑو;
            حالت 2:  جائزہ_expert_ids_n(ids, 2,     n_as, counts, میٹرکس_صفیں, mapping_stride); توڑو;
            طےشدہ: جائزہ_expert_ids_n(ids, n_ids, n_as, counts, میٹرکس_صفیں, mapping_stride); توڑو;
        }
    } ورنہ {
        // Strided fallback
        ثابت size_t ids_nb1 = ids->nb[1];
        ثابت uint8_t * ids_ڈیٹا = (ثابت uint8_t *) ids->ڈیٹا;
        برائے (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ثابت int32_t * صف_ptr = (ثابت int32_t *) (ids_ڈیٹا + iid1 * ids_nb1);
            برائے (uint32_t id = 0; id < n_ids; ++id) {
                ثابت int32_t i02 = *(ثابت int32_t *) ((ثابت uint8_t *) صف_ptr + id * ids_nb0);
                اگر (i02 < 0) {
                    جاری;
                }
                أكد(i02 < n_as);

                اگر (میٹرکس_صفیں) {
                    میٹرکس_صفیں[i02 * mapping_stride + counts[i02]] = (ساخت mmid_صف_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

صحیح op_matmul_id(ساخت htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ساخت htp_mm_context mmctx_struct = {0};
    ساخت htp_mm_context * mmctx = &mmctx_struct;

    ثابت صحیح حالت_2 = htp_mm_init_context(octx, kparams);
    اگر (حالت_2 != HTP_حالت_OK) {
        واپس حالت_2;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ثابت ساخت htp_tensor * محدود ids = octx->src[2];
    اگر (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت size_t src0_صف_سائز = nb01;
    ثابت size_t dst_صف_سائز  = nb1;

    ثابت size_t src0_صف_سائز_padded = hex_باری_اوپر(src0_صف_سائز, 128);

    ثابت uint32_t src0_nrows = ne01;  // per expert
    ثابت uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ثابت صحیح n_ids = ids->ne[0];  // n_expert_used
    ثابت صحیح n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_بنیاد;
    uint32_t   mapping_stride    = 1;
    uint32_t * میٹرکس_صف_counts = (uint32_t *) mapping_buf;
    ساخت mmid_صف_mapping * میٹرکس_صفیں = تہی;

    اگر (act_nrows > 1) {
        ثابت size_t میٹرکس_صف_counts_سائز = n_as * حجم(uint32_t);
        أكد(octx->ctx->ddr_spad_سائز >= میٹرکس_صف_counts_سائز);

        hex_l2fetch_بلاک((ثابت خالی *) ids->ڈیٹا, ids->ne[1] * ids->nb[1]);

        memset(میٹرکس_صف_counts, 0, میٹرکس_صف_counts_سائز);
        جائزہ_expert_ids(ids, n_ids, n_as, میٹرکس_صف_counts, تہی, 0);

        uint32_t max_گنتی = hvx_reduce_max_i32((ثابت uint8_t *) میٹرکس_صف_counts, n_as);
        mapping_stride = max_گنتی > 0 ? max_گنتی : 1;

        size_t میٹرکس_صف_map_سائز  = n_as * mapping_stride * حجم(ساخت mmid_صف_mapping);
        ثابت size_t کل_map_سائز = میٹرکس_صف_counts_سائز + میٹرکس_صف_map_سائز;

        اگر (کل_map_سائز > octx->ctx->ddr_spad_سائز) {
            mapping_buf = memalign(128, کل_map_سائز);
            اگر (!mapping_buf) {
                واپس HTP_حالت_اندرونی_ERR;
            }
        }

        میٹرکس_صف_counts = (uint32_t *) mapping_buf;
        میٹرکس_صفیں       = (ساخت mmid_صف_mapping *) (mapping_buf + میٹرکس_صف_counts_سائز);

        memset(میٹرکس_صف_counts, 0, n_as * حجم(uint32_t));
        جائزہ_expert_ids(ids, n_ids, n_as, میٹرکس_صف_counts, میٹرکس_صفیں, mapping_stride);
    }

    mmctx->میٹرکس_صف_counts    = میٹرکس_صف_counts;
    mmctx->میٹرکس_صفیں          = میٹرکس_صفیں;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_صف_سائز_padded = src0_صف_سائز_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_آغاز          = 0;
    mmctx->cur_m_صفیں           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    صحیح s;
    اگر (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } ورنہ {
        uint32_t src0_صف_آغاز = 0;
        uint32_t src0_صف_انتہا   = src0_nrows;
        اگر (octx->ctx->mdev.گنتی > 1) {
            ثابت bool can_بانٹو = htp_tensor_can_صف_تقسیم(dst, حجم(اعشاری));
            ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(src0_nrows, can_بانٹو ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            src0_صف_آغاز = range.آغاز;
            src0_صف_انتہا   = range.آغاز + range.گنتی;
        }

        اگر (src0_صف_آغاز >= src0_صف_انتہا) {
            اگر (mapping_buf != octx->ctx->ddr_spad_بنیاد) {
                آزاد(mapping_buf);
            }
            واپس HTP_حالت_OK;
        }

        ثابت uint32_t nrows = src0_صف_انتہا - src0_صف_آغاز;
        mmctx->src0_صف_آغاز = src0_صف_آغاز;
        mmctx->src0_صف_انتہا   = src0_صف_انتہا;

        mmctx->src0_nrows_فی_thread = fastdiv(nrows + octx->n_دھاگے - 1, &octx->n_دھاگے_div);
        mmctx->src0_nrows_فی_thread = hex_باری_اوپر(mmctx->src0_nrows_فی_thread, 32);

        اگر (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } ورنہ {
            s = HTP_حالت_NO_SUPPORT;
        }
    }

    اگر (mapping_buf != octx->ctx->ddr_spad_بنیاد) {
        آزاد(mapping_buf);
    }

    واپس s;
}

صحیح op_matmul_id_nx(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;
    ساخت htp_mm_context mmctx_struct = {0};
    ساخت htp_mm_context * mmctx = &mmctx_struct;

    ثابت صحیح حالت_2 = htp_mm_init_context(octx, kparams);
    اگر (حالت_2 != HTP_حالت_OK) {
        واپس حالت_2;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت ساخت htp_tensor * محدود src0 = octx->src[0];
    ثابت ساخت htp_tensor * محدود act  = octx->src[n_weights];
    ثابت ساخت htp_tensor * محدود ids  = octx->src[n_weights + 1];
    اگر (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        واپس HTP_حالت_NO_SUPPORT;
    }
    برائے (uint32_t p = 0; p < n_weights; p++) {
        اگر (htp_tensor_is_extended(octx->dsts[p])) {
            واپس HTP_حالت_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    ثابت size_t src0_صف_سائز = src0->nb[1];
    ثابت size_t src0_صف_سائز_padded = hex_باری_اوپر(src0_صف_سائز, 128);

    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ثابت صحیح n_ids = ids->ne[0];
    ثابت صحیح n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_بنیاد;
    uint32_t   mapping_stride    = 1;
    uint32_t * میٹرکس_صف_counts = (uint32_t *) mapping_buf;
    ساخت mmid_صف_mapping * میٹرکس_صفیں = تہی;

    اگر (act_nrows > 1) {
        ثابت size_t میٹرکس_صف_counts_سائز = n_as * حجم(uint32_t);
        أكد(octx->ctx->ddr_spad_سائز >= میٹرکس_صف_counts_سائز);

        hex_l2fetch_بلاک((ثابت خالی *) ids->ڈیٹا, ids->ne[1] * ids->nb[1]);

        memset(میٹرکس_صف_counts, 0, میٹرکس_صف_counts_سائز);
        جائزہ_expert_ids(ids, n_ids, n_as, میٹرکس_صف_counts, تہی, 0);

        uint32_t max_گنتی = hvx_reduce_max_i32((ثابت uint8_t *) میٹرکس_صف_counts, n_as);
        mapping_stride = max_گنتی > 0 ? max_گنتی : 1;

        size_t میٹرکس_صف_map_سائز  = n_as * mapping_stride * حجم(ساخت mmid_صف_mapping);
        ثابت size_t کل_map_سائز = میٹرکس_صف_counts_سائز + میٹرکس_صف_map_سائز;

        اگر (کل_map_سائز > octx->ctx->ddr_spad_سائز) {
            mapping_buf = memalign(128, کل_map_سائز);
            اگر (!mapping_buf) {
                واپس HTP_حالت_اندرونی_ERR;
            }
        }

        میٹرکس_صف_counts = (uint32_t *) mapping_buf;
        میٹرکس_صفیں       = (ساخت mmid_صف_mapping *) (mapping_buf + میٹرکس_صف_counts_سائز);

        memset(میٹرکس_صف_counts, 0, n_as * حجم(uint32_t));
        جائزہ_expert_ids(ids, n_ids, n_as, میٹرکس_صف_counts, میٹرکس_صفیں, mapping_stride);
    }

    mmctx->میٹرکس_صف_counts    = میٹرکس_صف_counts;
    mmctx->میٹرکس_صفیں          = میٹرکس_صفیں;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_صف_سائز_padded = src0_صف_سائز_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_آغاز          = 0;
    mmctx->cur_m_صفیں           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    صحیح s;
    اگر (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } ورنہ {
        اگر (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } ورنہ {
            s = HTP_حالت_NO_SUPPORT;
        }
    }

    اگر (mapping_buf != octx->ctx->ddr_spad_بنیاد) {
        آزاد(mapping_buf);
    }

    واپس s;
}
ساکن صحیح op_matmul_nx_impl(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;

    ثابت صحیح حالت_2 = htp_mm_init_context(octx, kparams);
    اگر (حالت_2 != HTP_حالت_OK) {
        واپس حالت_2;
    }

    اگر (kparams->n_hmx) {
        واپس hmx_mm_nx_2d_f32(octx, kparams);
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت uint32_t n_weights = kparams->n_weights;

    ثابت ساخت htp_tensor * محدود src0 = octx->src[0]; // first weight
    ثابت ساخت htp_tensor * محدود act  = octx->src[n_weights]; // activation x

    ساخت htp_mm_context mmctx_struct = {0};
    ساخت htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_آغاز = 0;
    mmctx->cur_m_صفیں  = act_nrows;

    ثابت size_t src0_صف_سائز = src0->nb[1];

    اگر (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ثابت uint32_t کل_nb = act_nrows * nb;

    کارکن_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    اگر (act_nrows < octx->n_دھاگے) {
        n_quant_tasks = MIN(کل_nb, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_بلاک_func(src0->type, act->type);
        برائے (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_پہلا = (کل_nb * ith) / n_quant_tasks;
            uint32_t ib_آخری  = (کل_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_پہلا[ith] = ib_پہلا;
            mmctx->quant_ib_آخری[ith]  = ib_آخری;
            mmctx->quant_r[ith]        = ib_پہلا / nb;
            mmctx->quant_c[ith]        = ib_پہلا % nb;
        }
    } ورنہ {
        n_quant_tasks = MIN(act_nrows, octx->n_دھاگے);
        quant_task_func = htp_mm_act_quant_صف_func(src0->type, act->type);
    }

    ثابت size_t act_صف_سائز = htp_mm_tiled_act_صف_سائز(src0->type, act->ne[0]);

    ساخت htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_دھاگے,
                                 0, src0_صف_سائز, act_صف_سائز, 0, kparams->n_prefetch, false, true);

    ثابت size_t vtcm_سائز = L.کل_بائٹیں;

    اگر (octx->ctx->vtcm_سائز < vtcm_سائز) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_سائز, vtcm_سائز);
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_صف_سائز;
    mmctx->vtcm_act_raw_stride = hex_باری_اوپر(act->ne[0] * حجم(اعشاری), QK_Q8_0_TILED * حجم(اعشاری));

    mmctx->vtcm_src0_سائز_فی_thread = fastdiv(L.src0_بائٹیں, &octx->n_دھاگے_div);
    mmctx->vtcm_act_سائز_فی_thread  = L.act_بائٹیں;
    mmctx->vtcm_dst_سائز_فی_thread  = fastdiv(L.dst_بائٹیں, &octx->n_دھاگے_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_صفیں_فی_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_قطار_تسلسل(octx->ctx->work_قطار, quant_task_func, mmctx, n_quant_tasks);

    work_قطار_تسلسل(octx->ctx->work_قطار, hvx_mm_nx_2d_tiled, mmctx, octx->n_دھاگے);

    واپس HTP_حالت_OK;
}

صحیح op_matmul_nx(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_mm_kernel_params * kparams = (ثابت ساخت htp_mm_kernel_params *) octx->kernel_params;

    اگر (kparams->collapse) {
        ثابت uint32_t n_weights = kparams->n_weights;
        ثابت ساخت htp_tensor * act = octx->src[n_weights];
        ثابت uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ساخت htp_tensor act_collapsed;
        ساخت htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        ثابت ساخت htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_صفیں(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        برائے (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ثابت ساخت htp_tensor * d = octx->dsts[p];
            ثابت uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_صفیں(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        ثابت صحیح حالت_2 = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        برائے (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        واپس حالت_2;
    }

    واپس op_matmul_nx_impl(octx);
}
