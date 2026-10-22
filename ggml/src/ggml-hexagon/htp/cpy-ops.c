#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>
#অন্তর্ভাব <qurt_memory.h>

#অন্তর্ভাব <math.h>
#অন্তর্ভাব <string.h>

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "cpy-ops.h"
#অন্তর্ভাব "dma-copy.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-fence.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "hvx-utils.h"
#অন্তর্ভাব "gather-rows.h"

কাঠামো htp_copy_context {
    কাঠামো htp_ops_context *              octx;
    ধ্রুবক কাঠামো htp_copy_kernel_params * kparams;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t src0_nrows_per_thread;

    uint32_t elem_start;
    uint32_t nelem;
    uint32_t elem_per_thread;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#define cpy_preamble                              \
    ধ্রুবক কাঠামো htp_tensor *src0 = octx->src[0]; \
    ধ্রুবক কাঠামো htp_tensor *dst  = octx->dst;    \
                                                  \
    ধ্রুবক uint32_t ne00 = src0->ne[0];            \
    ধ্রুবক uint32_t ne01 = src0->ne[1];            \
    ধ্রুবক uint32_t ne02 = src0->ne[2];            \
    ধ্রুবক uint32_t ne03 = src0->ne[3];            \
                                                  \
    ধ্রুবক uint32_t nb00 = src0->nb[0];            \
    ধ্রুবক uint32_t nb01 = src0->nb[1];            \
    ধ্রুবক uint32_t nb02 = src0->nb[2];            \
    ধ্রুবক uint32_t nb03 = src0->nb[3];            \
                                                  \
    ধ্রুবক uint32_t  ne0 = dst->ne[0];             \
    ধ্রুবক uint32_t  ne1 = dst->ne[1];             \
    ধ্রুবক uint32_t  ne2 = dst->ne[2];             \
    ধ্রুবক uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ধ্রুবক uint32_t  nb0 = dst->nb[0];             \
    ধ্রুবক uint32_t  nb1 = dst->nb[1];             \
    ধ্রুবক uint32_t  nb2 = dst->nb[2];             \
    ধ্রুবক uint32_t  nb3 = dst->nb[3];

#define DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_আকার)                                                \
স্থির শূন্য cpy_thread_##NAME##_reshape(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {            \
    কাঠামো htp_copy_context * ct = (কাঠামো htp_copy_context *) তথ্য;                                  \
    কাঠামো htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    ধ্রুবক uint32_t th_nelem = ct->elem_per_thread;                                                    \
    ধ্রুবক uint32_t th_start = ct->elem_start + ith * th_nelem;                                        \
    ধ্রুবক uint32_t th_end   = MIN(th_start + th_nelem, ct->elem_start + ct->nelem);                   \
    যদি (th_start >= th_end) ফেরত;                                                                   \
                                                                                                      \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    ধ্রুবক uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    ধ্রুবক uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    ধ্রুবক uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    ধ্রুবক uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    ধ্রুবক কাঠামো htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
    uint32_t e = th_start;                                                                            \
    uint32_t i13 = fastdiv(e, &rsh->div_ne2_ne1_ne0);                                                 \
    uint32_t rem = e - i13 * ne2_ne1_ne0;                                                             \
    uint32_t i12 = fastdiv(rem, &rsh->div_ne1_ne0);                                                   \
    uint32_t rem2 = rem - i12 * ne1_ne0;                                                              \
    uint32_t i11 = fastdiv(rem2, &rsh->div_ne0);                                                      \
    uint32_t i10 = rem2 - i11 * ne0;                                                                  \
                                                                                                      \
    uint32_t i03 = fastdiv(e, &rsh->div_ne02_ne01_ne00);                                              \
    uint32_t rem_s = e - i03 * ne02_ne01_ne00;                                                        \
    uint32_t i02 = fastdiv(rem_s, &rsh->div_ne01_ne00);                                               \
    uint32_t rem2_s = rem_s - i02 * ne01_ne00;                                                        \
    uint32_t i01 = fastdiv(rem2_s, &rsh->div_ne00);                                                   \
    uint32_t i00 = rem2_s - i01 * ne00;                                                               \
                                                                                                      \
    dma_addr_t dst_addr  = dst->তথ্য  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->তথ্য + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    ধ্রুবক bool rows_contig = (nb00 == ELEM_আকার) && (nb0 == ELEM_আকার);                               \
                                                                                                      \
    যখন (e < th_end) {                                                                              \
        ধ্রুবক uint32_t run = MIN(MIN(ne00 - i00, ne0 - i10), th_end - e);                             \
        যদি (rows_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, run * ELEM_আকার);             \
        } অন্যথা {                                                                                      \
            dma_cpy_ঠেলো_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_আকার, run);           \
        }                                                                                             \
        e += run;                                                                                     \
                                                                                                      \
        dst_addr += run * nb0;                                                                        \
        i10      += run;                                                                              \
        যদি (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            যদি (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                যদি (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->তথ্য + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += run * nb00;                                                                      \
        i00       += run;                                                                             \
        যদি (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            যদি (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                যদি (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->তথ্য + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_queue_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  দশমিক, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

কাঠামো cpy_row_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

স্থির অন্তঃরেখ শূন্য cpy_cursor_advance(কাঠামো cpy_row_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
    c->r   += n;
    c->i01 += n;
    যদি (c->i01 == ne01) {
        c->i01 = 0;
        যদি (++c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

#define DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
স্থির শূন্য cpy_thread_##NAME##_sameshape(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) { \
    কাঠামো htp_copy_context * ct = (কাঠামো htp_copy_context *) তথ্য;                         \
    কাঠামো htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    ধ্রুবক uint32_t dr  = ct->src0_nrows_per_thread;                                          \
    ধ্রুবক uint32_t ir0 = ct->row_start + dr * ith;                                           \
    ধ্রুবক uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);                           \
    যদি (ir0 >= ir1) ফেরত;                                                                  \
                                                                                             \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                 \
    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    ধ্রুবক কাঠামো htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    ধ্রুবক uint32_t src0_buf_আকার = cvt->src0_buf_আকার;                                       \
    ধ্রুবক uint32_t dst_buf_আকার  = cvt->dst_buf_আকার;                                        \
    ধ্রুবক uint32_t src0_row_stride = cvt->src0_row_stride;                                   \
    ধ্রুবক uint32_t dst_row_stride  = cvt->dst_row_stride;                                    \
    ধ্রুবক uint32_t blk_rows        = cvt->blk_rows;                                          \
    uint8_t * vtcm_src0_base = ct->vtcm_src0 + ith * cvt->spad0_আকার_per_thread;             \
    uint8_t * vtcm_dst_base  = ct->vtcm_dst  + ith * cvt->spad1_আকার_per_thread;             \
    ধ্রুবক uint32_t src0_row_আকার = ne00 * ct->kparams->src0_type_আকার;                       \
    ধ্রুবক uint32_t dst_row_আকার  = ne00 * ct->kparams->dst_type_আকার;                        \
                                                                                             \
    ধ্রুবক uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    কাঠামো cpy_row_cursor f = { ir0, i01, i02, i03 };                                        \
    কাঠামো cpy_row_cursor c = f;                                                             \
                                                                                             \
    জন্য (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_base + b * src0_buf_আকার;                             \
        uint8_t * dst_spad = vtcm_dst_base  + b * dst_buf_আকার;                              \
        ধ্রুবক uint32_t nr = MIN(blk_rows, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->তথ্য + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst->তথ্য, dst_spad),                            \
                       dst_row_আকার, dst_buf_আকার, dst_row_আকার, 0);                         \
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(src_spad, f_src0_addr),                          \
                       src0_row_stride, nb01, src0_row_আকার, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    জন্য (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        ধ্রুবক uint32_t nr = MIN(blk_rows, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_queue_বের_করো(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_queue_বের_করো(dma_q).dst;               \
                                                                                             \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        জন্য (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_row_stride, src_spad + r * src0_row_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->তথ্য + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(c_dst_addr, dst_spad),                           \
                       nb1, dst_row_stride, dst_row_আকার, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        যদি (f.r < ir1) {                                                                     \
            ধ্রুবক uint32_t fnr = MIN(blk_rows, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->তথ্য + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(src_spad, f_src0_addr),                      \
                           src0_row_stride, nb01, src0_row_আকার, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_queue_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

স্থির পূর্ণসংখ্যা cpy_scalar(কাঠামো htp_ops_context * octx, ধ্রুবক কাঠামো htp_copy_kernel_params * kparams) {
    যদি (octx->ctx->mdev.গুনতি > 1 && octx->ctx->mdev.idx > 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    যদি (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->তথ্য, src0->তথ্য, kparams->src0_type_আকার);
        dma_queue_flush(octx->ctx->dma[0]);
        ফেরত HTP_STATUS_OK;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_base;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    ধ্রুবক uint32_t s_আকার = kparams->src0_type_আকার;
    ধ্রুবক uint32_t d_আকার = kparams->dst_type_আকার;

    dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s_vtcm, src0->তথ্য), s_আকার, s_আকার, s_আকার, 1);
    dma_queue_বের_করো(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_base;
    uint8_t * d_ptr = s_ptr + VLEN;

    ধ্রুবক HVX_Vector v_src = hvx_vmem(s_ptr);
    HVX_Vector v_dst;

    যদি (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } অন্যথা যদি (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } অন্যথা যদি (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_to_f16(v_src, v_src);
    } অন্যথা যদি (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src));
    } অন্যথা {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst->তথ্য, d_vtcm), d_আকার, d_আকার, d_আকার, 1);
    dma_queue_flush(dma_q);
    ফেরত HTP_STATUS_OK;
}

স্থির পূর্ণসংখ্যা cpy_1d_contig(কাঠামো htp_ops_context * octx, ধ্রুবক কাঠামো htp_copy_kernel_params * kparams) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->যোগ_elems;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->যোগ_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        elem_start = range.start;
        nelem      = range.গুনতি;
    }

    যদি (nelem > 0) {
        dma_queue * q = octx->ctx->dma[0];
        ধ্রুবক uint32_t type_আকার = kparams->src0_type_আকার;
        dma_addr_t dst_addr  = dst->তথ্য  + elem_start * type_আকার;
        dma_addr_t src0_addr = src0->তথ্য + elem_start * type_আকার;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_আকার);
        dma_queue_flush(q);
    }

    ফেরত HTP_STATUS_OK;
}

স্থির পূর্ণসংখ্যা cpy_sameshape_sametype(কাঠামো htp_ops_context * octx, ধ্রুবক কাঠামো htp_copy_kernel_params * kparams) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->যোগ_rows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->যোগ_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows > 0) {
        dma_queue * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_আকার, row_start, nrows);
        dma_queue_flush(q);
    }

    ফেরত HTP_STATUS_OK;
}

স্থির পূর্ণসংখ্যা cpy_sameshape_convert(কাঠামো htp_ops_context * octx, ধ্রুবক কাঠামো htp_copy_kernel_params * kparams) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    যদি (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->যোগ_rows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->যোগ_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    যদি (kparams->vtcm_আকার > octx->ctx->vtcm_আকার) {
        ফেরত HTP_STATUS_VTCM_TOO_SMALL;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;
    ধ্রুবক কাঠামো htp_copy_convert_params * cvt = &kparams->u.convert;

    কাঠামো htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.row_start             = row_start;
    ct.nrows                 = nrows;
    ct.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    ct.vtcm_src0 = vtcm_base;
    ct.vtcm_dst  = vtcm_base + (size_t) n_threads * cvt->spad0_আকার_per_thread;

    work_queue_func_t copy_fun = ফাঁকা;
    যদি (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } অন্যথা যদি (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } অন্যথা যদি (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } অন্যথা যদি (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } অন্যথা {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    ফেরত HTP_STATUS_OK;
}

স্থির পূর্ণসংখ্যা cpy_reshape(কাঠামো htp_ops_context * octx, ধ্রুবক কাঠামো htp_copy_kernel_params * kparams) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    যদি (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->যোগ_elems;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->যোগ_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        elem_start = range.start;
        nelem      = range.গুনতি;
    }

    যদি (nelem == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;

    কাঠামো htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_start      = elem_start;
    ct.nelem           = nelem;
    ct.elem_per_thread = fastdiv(nelem + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t copy_fun = ফাঁকা;
    নির্বাচন (src0->type) {
        ক্ষেত্র HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; বিরতি;
        ক্ষেত্র HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; বিরতি;
        ক্ষেত্র HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; বিরতি;
        পূর্বনির্ধারিত: ফেরত HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    ফেরত HTP_STATUS_OK;
}

পূর্ণসংখ্যা op_cpy(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_copy_kernel_params * kparams = (ধ্রুবক কাঠামো htp_copy_kernel_params *) octx->kernel_params;
    পূর্ণসংখ্যা status = HTP_STATUS_OK;

    নির্বাচন (kparams->kernel_type) {
        ক্ষেত্র HTP_COPY_KERNEL_SCALAR:
            status = cpy_scalar(octx, kparams);
            বিরতি;
        ক্ষেত্র HTP_COPY_KERNEL_1D_CONTIG:
            status = cpy_1d_contig(octx, kparams);
            বিরতি;
        ক্ষেত্র HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            status = cpy_sameshape_sametype(octx, kparams);
            বিরতি;
        ক্ষেত্র HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            status = cpy_sameshape_convert(octx, kparams);
            বিরতি;
        ক্ষেত্র HTP_COPY_KERNEL_RESHAPE:
            status = cpy_reshape(octx, kparams);
            বিরতি;
        ক্ষেত্র HTP_COPY_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, octx->src[0], ফাঁকা, &kparams->u.gather, kparams->n_threads);
            বিরতি;
        পূর্বনির্ধারিত:
            status = HTP_STATUS_NO_SUPPORT;
            বিরতি;
    }

    htp_ops_context_set_status(octx, status);

    যদি (octx->op == HTP_OP_CPY_FENCE) {
        যদি (octx->ctx->mdev.গুনতি > 1) {
            htp_mdev_group_barrier(octx);
        }

        যদি (octx->ctx->mdev.idx == 0) {
            ধ্রুবক কাঠামো htp_tensor * sync = octx->src[1];
            ধ্রুবক uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->তথ্য;
            htp_fence_write(sync_fence, seq, octx->status);

            FARF(উপরের_2, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        }
    }

    ফেরত octx->status;
}
