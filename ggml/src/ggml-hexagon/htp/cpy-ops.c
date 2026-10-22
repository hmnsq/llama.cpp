#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>
#شامل <qurt_memory.h>

#شامل <math.h>
#شامل <string.h>

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "cpy-ops.h"
#شامل "dma-copy.h"
#شامل "htp-ctx.h"
#شامل "htp-fence.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hvx-utils.h"
#شامل "gather-rows.h"

ساخت htp_copy_context {
    ساخت htp_ops_context *              octx;
    ثابت ساخت htp_copy_kernel_params * kparams;

    uint32_t صف_آغاز;
    uint32_t nrows;
    uint32_t src0_nrows_فی_thread;

    uint32_t elem_آغاز;
    uint32_t nelem;
    uint32_t elem_فی_thread;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#تعریف_کرو cpy_preamble                              \
    ثابت ساخت htp_tensor *src0 = octx->src[0]; \
    ثابت ساخت htp_tensor *dst  = octx->dst;    \
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

#تعریف_کرو DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_سائز)                                                \
ساکن خالی cpy_thread_##NAME##_reshape(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {            \
    ساخت htp_copy_context * ct = (ساخت htp_copy_context *) ڈیٹا;                                  \
    ساخت htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    ثابت uint32_t th_nelem = ct->elem_فی_thread;                                                    \
    ثابت uint32_t th_آغاز = ct->elem_آغاز + ith * th_nelem;                                        \
    ثابت uint32_t th_انتہا   = MIN(th_آغاز + th_nelem, ct->elem_آغاز + ct->nelem);                   \
    اگر (th_آغاز >= th_انتہا) واپس;                                                                   \
                                                                                                      \
    dma_قطار * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    ثابت uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    ثابت uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    ثابت uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    ثابت uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    ثابت ساخت htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
    uint32_t e = th_آغاز;                                                                            \
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
    dma_addr_t dst_addr  = dst->ڈیٹا  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->ڈیٹا + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    ثابت bool صفیں_contig = (nb00 == ELEM_سائز) && (nb0 == ELEM_سائز);                               \
                                                                                                      \
    جب (e < th_انتہا) {                                                                              \
        ثابت uint32_t تسلسل = MIN(MIN(ne00 - i00, ne0 - i10), th_انتہا - e);                             \
        اگر (صفیں_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, تسلسل * ELEM_سائز);             \
        } ورنہ {                                                                                      \
            dma_cpy_ڈالو_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_سائز, تسلسل);           \
        }                                                                                             \
        e += تسلسل;                                                                                     \
                                                                                                      \
        dst_addr += تسلسل * nb0;                                                                        \
        i10      += تسلسل;                                                                              \
        اگر (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            اگر (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                اگر (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->ڈیٹا + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += تسلسل * nb00;                                                                      \
        i00       += تسلسل;                                                                             \
        اگر (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            اگر (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                اگر (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_قطار_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  اعشاری, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

ساخت cpy_صف_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ساکن اندرونی خالی cpy_cursor_advance(ساخت cpy_صف_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
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

#تعریف_کرو DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
ساکن خالی cpy_thread_##NAME##_sameshape(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) { \
    ساخت htp_copy_context * ct = (ساخت htp_copy_context *) ڈیٹا;                         \
    ساخت htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    ثابت uint32_t dr  = ct->src0_nrows_فی_thread;                                          \
    ثابت uint32_t ir0 = ct->صف_آغاز + dr * ith;                                           \
    ثابت uint32_t ir1 = MIN(ir0 + dr, ct->صف_آغاز + ct->nrows);                           \
    اگر (ir0 >= ir1) واپس;                                                                  \
                                                                                             \
    dma_قطار * dma_q = octx->ctx->dma[ith];                                                 \
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    ثابت ساخت htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    ثابت uint32_t src0_buf_سائز = cvt->src0_buf_سائز;                                       \
    ثابت uint32_t dst_buf_سائز  = cvt->dst_buf_سائز;                                        \
    ثابت uint32_t src0_صف_stride = cvt->src0_صف_stride;                                   \
    ثابت uint32_t dst_صف_stride  = cvt->dst_صف_stride;                                    \
    ثابت uint32_t blk_صفیں        = cvt->blk_صفیں;                                          \
    uint8_t * vtcm_src0_بنیاد = ct->vtcm_src0 + ith * cvt->spad0_سائز_فی_thread;             \
    uint8_t * vtcm_dst_بنیاد  = ct->vtcm_dst  + ith * cvt->spad1_سائز_فی_thread;             \
    ثابت uint32_t src0_صف_سائز = ne00 * ct->kparams->src0_type_سائز;                       \
    ثابت uint32_t dst_صف_سائز  = ne00 * ct->kparams->dst_type_سائز;                        \
                                                                                             \
    ثابت uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    ساخت cpy_صف_cursor f = { ir0, i01, i02, i03 };                                        \
    ساخت cpy_صف_cursor c = f;                                                             \
                                                                                             \
    برائے (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_بنیاد + b * src0_buf_سائز;                             \
        uint8_t * dst_spad = vtcm_dst_بنیاد  + b * dst_buf_سائز;                              \
        ثابت uint32_t nr = MIN(blk_صفیں, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->ڈیٹا + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst->ڈیٹا, dst_spad),                            \
                       dst_صف_سائز, dst_buf_سائز, dst_صف_سائز, 0);                         \
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(src_spad, f_src0_addr),                          \
                       src0_صف_stride, nb01, src0_صف_سائز, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    برائے (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        ثابت uint32_t nr = MIN(blk_صفیں, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_قطار_نکالو(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_قطار_نکالو(dma_q).dst;               \
                                                                                             \
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        برائے (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_صف_stride, src_spad + r * src0_صف_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->ڈیٹا + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(c_dst_addr, dst_spad),                           \
                       nb1, dst_صف_stride, dst_صف_سائز, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        اگر (f.r < ir1) {                                                                     \
            ثابت uint32_t fnr = MIN(blk_صفیں, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->ڈیٹا + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(src_spad, f_src0_addr),                      \
                           src0_صف_stride, nb01, src0_صف_سائز, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_قطار_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

ساکن صحیح cpy_scalar(ساخت htp_ops_context * octx, ثابت ساخت htp_copy_kernel_params * kparams) {
    اگر (octx->ctx->mdev.گنتی > 1 && octx->ctx->mdev.idx > 0) {
        واپس HTP_حالت_OK;
    }

    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    اگر (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->ڈیٹا, src0->ڈیٹا, kparams->src0_type_سائز);
        dma_قطار_flush(octx->ctx->dma[0]);
        واپس HTP_حالت_OK;
    }

    dma_قطار * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_بنیاد;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    ثابت uint32_t s_سائز = kparams->src0_type_سائز;
    ثابت uint32_t d_سائز = kparams->dst_type_سائز;

    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_vtcm, src0->ڈیٹا), s_سائز, s_سائز, s_سائز, 1);
    dma_قطار_نکالو(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_بنیاد;
    uint8_t * d_ptr = s_ptr + VLEN;

    ثابت HVX_سمتیہ v_src = hvx_vmem(s_ptr);
    HVX_سمتیہ v_dst;

    اگر (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } ورنہ اگر (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } ورنہ اگر (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_مقصد_f16(v_src, v_src);
    } ورنہ اگر (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(v_src));
    } ورنہ {
        واپس HTP_حالت_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst->ڈیٹا, d_vtcm), d_سائز, d_سائز, d_سائز, 1);
    dma_قطار_flush(dma_q);
    واپس HTP_حالت_OK;
}

ساکن صحیح cpy_1d_contig(ساخت htp_ops_context * octx, ثابت ساخت htp_copy_kernel_params * kparams) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    uint32_t elem_آغاز = 0;
    uint32_t nelem      = kparams->کل_elems;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            kparams->کل_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        elem_آغاز = range.آغاز;
        nelem      = range.گنتی;
    }

    اگر (nelem > 0) {
        dma_قطار * q = octx->ctx->dma[0];
        ثابت uint32_t type_سائز = kparams->src0_type_سائز;
        dma_addr_t dst_addr  = dst->ڈیٹا  + elem_آغاز * type_سائز;
        dma_addr_t src0_addr = src0->ڈیٹا + elem_آغاز * type_سائز;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_سائز);
        dma_قطار_flush(q);
    }

    واپس HTP_حالت_OK;
}

ساکن صحیح cpy_sameshape_sametype(ساخت htp_ops_context * octx, ثابت ساخت htp_copy_kernel_params * kparams) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = kparams->کل_صفیں;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            kparams->کل_صفیں, 1, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows > 0) {
        dma_قطار * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_سائز, صف_آغاز, nrows);
        dma_قطار_flush(q);
    }

    واپس HTP_حالت_OK;
}

ساکن صحیح cpy_sameshape_convert(ساخت htp_ops_context * octx, ثابت ساخت htp_copy_kernel_params * kparams) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = kparams->کل_صفیں;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            kparams->کل_صفیں, 1, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    اگر (kparams->vtcm_سائز > octx->ctx->vtcm_سائز) {
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;
    ثابت ساخت htp_copy_convert_params * cvt = &kparams->u.convert;

    ساخت htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.صف_آغاز             = صف_آغاز;
    ct.nrows                 = nrows;
    ct.src0_nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);

    uint8_t * vtcm_بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;
    ct.vtcm_src0 = vtcm_بنیاد;
    ct.vtcm_dst  = vtcm_بنیاد + (size_t) n_دھاگے * cvt->spad0_سائز_فی_thread;

    work_قطار_func_t copy_fun = تہی;
    اگر (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } ورنہ اگر (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } ورنہ اگر (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } ورنہ اگر (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } ورنہ {
        واپس HTP_حالت_NO_SUPPORT;
    }

    work_قطار_تسلسل(octx->ctx->work_قطار, copy_fun, &ct, n_دھاگے);
    واپس HTP_حالت_OK;
}

ساکن صحیح cpy_reshape(ساخت htp_ops_context * octx, ثابت ساخت htp_copy_kernel_params * kparams) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    uint32_t elem_آغاز = 0;
    uint32_t nelem      = kparams->کل_elems;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            kparams->کل_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        elem_آغاز = range.آغاز;
        nelem      = range.گنتی;
    }

    اگر (nelem == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    ساخت htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_آغاز      = elem_آغاز;
    ct.nelem           = nelem;
    ct.elem_فی_thread = fastdiv(nelem + n_دھاگے - 1, &octx->n_دھاگے_div);

    work_قطار_func_t copy_fun = تہی;
    انتخاب (src0->type) {
        حالت HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; توڑو;
        حالت HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; توڑو;
        حالت HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; توڑو;
        طےشدہ: واپس HTP_حالت_NO_SUPPORT;
    }

    work_قطار_تسلسل(octx->ctx->work_قطار, copy_fun, &ct, n_دھاگے);
    واپس HTP_حالت_OK;
}

صحیح op_cpy(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_copy_kernel_params * kparams = (ثابت ساخت htp_copy_kernel_params *) octx->kernel_params;
    صحیح حالت_2 = HTP_حالت_OK;

    انتخاب (kparams->kernel_type) {
        حالت HTP_COPY_KERNEL_SCALAR:
            حالت_2 = cpy_scalar(octx, kparams);
            توڑو;
        حالت HTP_COPY_KERNEL_1D_CONTIG:
            حالت_2 = cpy_1d_contig(octx, kparams);
            توڑو;
        حالت HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            حالت_2 = cpy_sameshape_sametype(octx, kparams);
            توڑو;
        حالت HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            حالت_2 = cpy_sameshape_convert(octx, kparams);
            توڑو;
        حالت HTP_COPY_KERNEL_RESHAPE:
            حالت_2 = cpy_reshape(octx, kparams);
            توڑو;
        حالت HTP_COPY_KERNEL_GATHER_صفیں:
            حالت_2 = htp_gather_صفیں(octx, octx->src[0], تہی, &kparams->u.gather, kparams->n_دھاگے);
            توڑو;
        طےشدہ:
            حالت_2 = HTP_حالت_NO_SUPPORT;
            توڑو;
    }

    htp_ops_context_سیٹ_حالت(octx, حالت_2);

    اگر (octx->op == HTP_OP_CPY_FENCE) {
        اگر (octx->ctx->mdev.گنتی > 1) {
            htp_mdev_گروہ_barrier(octx);
        }

        اگر (octx->ctx->mdev.idx == 0) {
            ثابت ساخت htp_tensor * sync = octx->src[1];
            ثابت uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->ڈیٹا;
            htp_fence_write(sync_fence, seq, octx->حالت_2);

            FARF(بلند_2, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->حالت_2);
        }
    }

    واپس octx->حالت_2;
}
