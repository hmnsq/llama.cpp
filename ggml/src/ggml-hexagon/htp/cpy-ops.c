#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>
#شامل <qurt_memory.h>

#شامل <math.h>
#شامل <string.h>

#define GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "cpy-ops.h"
#شامل "dma-copy.h"
#شامل "htp-ctx.h"
#شامل "htp-fence.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hvx-utils.h"
#شامل "gather-rows.h"

ساختار htp_copy_context {
    ساختار htp_ops_context *              octx;
    ثابت ساختار htp_copy_kernel_params * kparams;

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
    ثابت ساختار htp_tensor *src0 = octx->src[0]; \
    ثابت ساختار htp_tensor *dst  = octx->dst;    \
                                                  \
    ثابت uint32_t ne00 = src0->ne[0];            \
    ثابت uint32_t ne01 = src0->ne[1];            \
    ثابت uint32_t ne02 = src0->ne[2];            \
    ثابت uint32_t ne03 = src0->ne[3];            \
                                                  \
    ثابت uint32_t nb00 = src0->nb[0];            \
    ثابت uint32_t nb01 = src0->nb[1];            \
    ثابت uint32_t nb02 = src0->nb[2];            \
    ثابت uint32_t nb03 = src0->nb[3];            \
                                                  \
    ثابت uint32_t  ne0 = dst->ne[0];             \
    ثابت uint32_t  ne1 = dst->ne[1];             \
    ثابت uint32_t  ne2 = dst->ne[2];             \
    ثابت uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ثابت uint32_t  nb0 = dst->nb[0];             \
    ثابت uint32_t  nb1 = dst->nb[1];             \
    ثابت uint32_t  nb2 = dst->nb[2];             \
    ثابت uint32_t  nb3 = dst->nb[3];

#define DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_SIZE)                                                \
ایستا تهی cpy_thread_##NAME##_reshape(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {            \
    ساختار htp_copy_context * ct = (ساختار htp_copy_context *) data;                                  \
    ساختار htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    ثابت uint32_t th_nelem = ct->elem_per_thread;                                                    \
    ثابت uint32_t th_start = ct->elem_start + ith * th_nelem;                                        \
    ثابت uint32_t th_end   = MIN(th_start + th_nelem, ct->elem_start + ct->nelem);                   \
    اگر (th_start >= th_end) بازگشت;                                                                   \
                                                                                                      \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    ثابت uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    ثابت uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    ثابت uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    ثابت uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    ثابت ساختار htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
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
    dma_addr_t dst_addr  = dst->data  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    ثابت bool rows_contig = (nb00 == ELEM_SIZE) && (nb0 == ELEM_SIZE);                               \
                                                                                                      \
    وقتی (e < th_end) {                                                                              \
        ثابت uint32_t run = MIN(MIN(ne00 - i00, ne0 - i10), th_end - e);                             \
        اگر (rows_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, run * ELEM_SIZE);             \
        } وگرنه {                                                                                      \
            dma_cpy_افزودن_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_SIZE, run);           \
        }                                                                                             \
        e += run;                                                                                     \
                                                                                                      \
        dst_addr += run * nb0;                                                                        \
        i10      += run;                                                                              \
        اگر (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            اگر (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                اگر (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->data + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += run * nb00;                                                                      \
        i00       += run;                                                                             \
        اگر (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            اگر (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                اگر (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->data + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_queue_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  اعشاری, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

ساختار cpy_row_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ایستا درونخطی تهی cpy_cursor_advance(ساختار cpy_row_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
    c->r   += n;
    c->i01 += n;
    اگر (c->i01 == ne01) {
        c->i01 = 0;
        اگر (++c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

#define DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
ایستا تهی cpy_thread_##NAME##_sameshape(unsigned صحیح nth, unsigned صحیح ith, تهی * data) { \
    ساختار htp_copy_context * ct = (ساختار htp_copy_context *) data;                         \
    ساختار htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    ثابت uint32_t dr  = ct->src0_nrows_per_thread;                                          \
    ثابت uint32_t ir0 = ct->row_start + dr * ith;                                           \
    ثابت uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);                           \
    اگر (ir0 >= ir1) بازگشت;                                                                  \
                                                                                             \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                 \
    ساختار htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    ثابت ساختار htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    ثابت uint32_t src0_buf_size = cvt->src0_buf_size;                                       \
    ثابت uint32_t dst_buf_size  = cvt->dst_buf_size;                                        \
    ثابت uint32_t src0_row_stride = cvt->src0_row_stride;                                   \
    ثابت uint32_t dst_row_stride  = cvt->dst_row_stride;                                    \
    ثابت uint32_t blk_rows        = cvt->blk_rows;                                          \
    uint8_t * vtcm_src0_base = ct->vtcm_src0 + ith * cvt->spad0_size_per_thread;             \
    uint8_t * vtcm_dst_base  = ct->vtcm_dst  + ith * cvt->spad1_size_per_thread;             \
    ثابت uint32_t src0_row_size = ne00 * ct->kparams->src0_type_size;                       \
    ثابت uint32_t dst_row_size  = ne00 * ct->kparams->dst_type_size;                        \
                                                                                             \
    ثابت uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    ساختار cpy_row_cursor f = { ir0, i01, i02, i03 };                                        \
    ساختار cpy_row_cursor c = f;                                                             \
                                                                                             \
    برای (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_base + b * src0_buf_size;                             \
        uint8_t * dst_spad = vtcm_dst_base  + b * dst_buf_size;                              \
        ثابت uint32_t nr = MIN(blk_rows, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->data + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_queue_افزودن(dma_q, dma_make_data(dst->data, dst_spad),                            \
                       dst_row_size, dst_buf_size, dst_row_size, 0);                         \
        dma_queue_افزودن(dma_q, dma_make_data(src_spad, f_src0_addr),                          \
                       src0_row_stride, nb01, src0_row_size, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    برای (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        ثابت uint32_t nr = MIN(blk_rows, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_queue_برداشت(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_queue_برداشت(dma_q).dst;               \
                                                                                             \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        برای (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_row_stride, src_spad + r * src0_row_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->data + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_queue_افزودن(dma_q, dma_make_data(c_dst_addr, dst_spad),                           \
                       nb1, dst_row_stride, dst_row_size, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        اگر (f.r < ir1) {                                                                     \
            ثابت uint32_t fnr = MIN(blk_rows, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->data + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_queue_افزودن(dma_q, dma_make_data(src_spad, f_src0_addr),                      \
                           src0_row_stride, nb01, src0_row_size, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_queue_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

ایستا صحیح cpy_scalar(ساختار htp_ops_context * octx, ثابت ساختار htp_copy_kernel_params * kparams) {
    اگر (octx->ctx->mdev.شمار > 1 && octx->ctx->mdev.idx > 0) {
        بازگشت HTP_STATUS_OK;
    }

    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    اگر (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->data, src0->data, kparams->src0_type_size);
        dma_queue_flush(octx->ctx->dma[0]);
        بازگشت HTP_STATUS_OK;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_base;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    ثابت uint32_t s_size = kparams->src0_type_size;
    ثابت uint32_t d_size = kparams->dst_type_size;

    dma_queue_افزودن(dma_q, dma_make_data(s_vtcm, src0->data), s_size, s_size, s_size, 1);
    dma_queue_برداشت(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_base;
    uint8_t * d_ptr = s_ptr + VLEN;

    ثابت HVX_Vector v_src = hvx_vmem(s_ptr);
    HVX_Vector v_dst;

    اگر (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } وگرنه اگر (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } وگرنه اگر (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_to_f16(v_src, v_src);
    } وگرنه اگر (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src));
    } وگرنه {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_queue_افزودن(dma_q, dma_make_data(dst->data, d_vtcm), d_size, d_size, d_size, 1);
    dma_queue_flush(dma_q);
    بازگشت HTP_STATUS_OK;
}

ایستا صحیح cpy_1d_contig(ساختار htp_ops_context * octx, ثابت ساختار htp_copy_kernel_params * kparams) {
    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->total_elems;

    اگر (octx->ctx->mdev.شمار > 1) {
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        elem_start = range.start;
        nelem      = range.شمار;
    }

    اگر (nelem > 0) {
        dma_queue * q = octx->ctx->dma[0];
        ثابت uint32_t type_size = kparams->src0_type_size;
        dma_addr_t dst_addr  = dst->data  + elem_start * type_size;
        dma_addr_t src0_addr = src0->data + elem_start * type_size;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_size);
        dma_queue_flush(q);
    }

    بازگشت HTP_STATUS_OK;
}

ایستا صحیح cpy_sameshape_sametype(ساختار htp_ops_context * octx, ثابت ساختار htp_copy_kernel_params * kparams) {
    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->total_rows;

    اگر (octx->ctx->mdev.شمار > 1) {
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        row_start = range.start;
        nrows     = range.شمار;
    }

    اگر (nrows > 0) {
        dma_queue * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_size, row_start, nrows);
        dma_queue_flush(q);
    }

    بازگشت HTP_STATUS_OK;
}

ایستا صحیح cpy_sameshape_convert(ساختار htp_ops_context * octx, ثابت ساختار htp_copy_kernel_params * kparams) {
    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    اگر (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        بازگشت HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->total_rows;

    اگر (octx->ctx->mdev.شمار > 1) {
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        row_start = range.start;
        nrows     = range.شمار;
    }

    اگر (nrows == 0) {
        بازگشت HTP_STATUS_OK;
    }

    اگر (kparams->vtcm_size > octx->ctx->vtcm_size) {
        بازگشت HTP_STATUS_VTCM_TOO_SMALL;
    }

    ثابت uint32_t n_threads = octx->n_threads;
    ثابت ساختار htp_copy_convert_params * cvt = &kparams->u.convert;

    ساختار htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.row_start             = row_start;
    ct.nrows                 = nrows;
    ct.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    ct.vtcm_src0 = vtcm_base;
    ct.vtcm_dst  = vtcm_base + (size_t) n_threads * cvt->spad0_size_per_thread;

    work_queue_func_t copy_fun = پوچ;
    اگر (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } وگرنه اگر (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } وگرنه اگر (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } وگرنه اگر (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } وگرنه {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    بازگشت HTP_STATUS_OK;
}

ایستا صحیح cpy_reshape(ساختار htp_ops_context * octx, ثابت ساختار htp_copy_kernel_params * kparams) {
    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    اگر (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        بازگشت HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->total_elems;

    اگر (octx->ctx->mdev.شمار > 1) {
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        elem_start = range.start;
        nelem      = range.شمار;
    }

    اگر (nelem == 0) {
        بازگشت HTP_STATUS_OK;
    }

    ثابت uint32_t n_threads = octx->n_threads;

    ساختار htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_start      = elem_start;
    ct.nelem           = nelem;
    ct.elem_per_thread = fastdiv(nelem + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t copy_fun = پوچ;
    انتخاب (src0->type) {
        حالت HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; بشکن;
        حالت HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; بشکن;
        حالت HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; بشکن;
        پیشفرض: بازگشت HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    بازگشت HTP_STATUS_OK;
}

صحیح op_cpy(ساختار htp_ops_context * octx) {
    ثابت ساختار htp_copy_kernel_params * kparams = (ثابت ساختار htp_copy_kernel_params *) octx->kernel_params;
    صحیح status = HTP_STATUS_OK;

    انتخاب (kparams->kernel_type) {
        حالت HTP_COPY_KERNEL_SCALAR:
            status = cpy_scalar(octx, kparams);
            بشکن;
        حالت HTP_COPY_KERNEL_1D_CONTIG:
            status = cpy_1d_contig(octx, kparams);
            بشکن;
        حالت HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            status = cpy_sameshape_sametype(octx, kparams);
            بشکن;
        حالت HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            status = cpy_sameshape_convert(octx, kparams);
            بشکن;
        حالت HTP_COPY_KERNEL_RESHAPE:
            status = cpy_reshape(octx, kparams);
            بشکن;
        حالت HTP_COPY_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, octx->src[0], پوچ, &kparams->u.gather, kparams->n_threads);
            بشکن;
        پیشفرض:
            status = HTP_STATUS_NO_SUPPORT;
            بشکن;
    }

    htp_ops_context_set_status(octx, status);

    اگر (octx->op == HTP_OP_CPY_FENCE) {
        اگر (octx->ctx->mdev.شمار > 1) {
            htp_mdev_group_barrier(octx);
        }

        اگر (octx->ctx->mdev.idx == 0) {
            ثابت ساختار htp_tensor * sync = octx->src[1];
            ثابت uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;
            htp_fence_write(sync_fence, seq, octx->status);

            FARF(بالا_2, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        }
    }

    بازگشت octx->status;
}
