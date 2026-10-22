#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>
#ضمن <qurt_memory.h>

#ضمن <math.h>
#ضمن <string.h>

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "cpy-ops.h"
#ضمن "dma-copy.h"
#ضمن "htp-ctx.h"
#ضمن "htp-fence.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-utils.h"
#ضمن "gather-rows.h"

بنية htp_copy_context {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_copy_kernel_params * kparams;

    uint32_t ردیف_شروع;
    uint32_t nrows;
    uint32_t src0_nrows_در_هر_thread;

    uint32_t elem_شروع;
    uint32_t nelem;
    uint32_t elem_در_هر_thread;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#تعریف_کن cpy_preamble                              \
    ثابت بنية htp_tensor *src0 = octx->src[0]; \
    ثابت بنية htp_tensor *dst  = octx->dst;    \
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

#تعریف_کن DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_اندازه)                                                \
ساكن فراغ cpy_thread_##NAME##_reshape(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {            \
    بنية htp_copy_context * ct = (بنية htp_copy_context *) داده;                                  \
    بنية htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    ثابت uint32_t th_nelem = ct->elem_در_هر_thread;                                                    \
    ثابت uint32_t th_شروع = ct->elem_شروع + ith * th_nelem;                                        \
    ثابت uint32_t th_انتها   = MIN(th_شروع + th_nelem, ct->elem_شروع + ct->nelem);                   \
    إذا (th_شروع >= th_انتها) أرجع;                                                                   \
                                                                                                      \
    dma_صف * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    ثابت uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    ثابت uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    ثابت uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    ثابت uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    ثابت بنية htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
    uint32_t e = th_شروع;                                                                            \
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
    dma_addr_t dst_addr  = dst->داده  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->داده + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    ثابت bool ردیفها_contig = (nb00 == ELEM_اندازه) && (nb0 == ELEM_اندازه);                               \
                                                                                                      \
    طالما (e < th_انتها) {                                                                              \
        ثابت uint32_t پیاپی = MIN(MIN(ne00 - i00, ne0 - i10), th_انتها - e);                             \
        إذا (ردیفها_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, پیاپی * ELEM_اندازه);             \
        } وإلا {                                                                                      \
            dma_cpy_افزودن_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_اندازه, پیاپی);           \
        }                                                                                             \
        e += پیاپی;                                                                                     \
                                                                                                      \
        dst_addr += پیاپی * nb0;                                                                        \
        i10      += پیاپی;                                                                              \
        إذا (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            إذا (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                إذا (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->داده + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += پیاپی * nb00;                                                                      \
        i00       += پیاپی;                                                                             \
        إذا (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            إذا (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                إذا (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->داده + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_صف_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  عائم, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

بنية cpy_ردیف_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ساكن مضمن فراغ cpy_cursor_advance(بنية cpy_ردیف_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
    c->r   += n;
    c->i01 += n;
    إذا (c->i01 == ne01) {
        c->i01 = 0;
        إذا (++c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

#تعریف_کن DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
ساكن فراغ cpy_thread_##NAME##_sameshape(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) { \
    بنية htp_copy_context * ct = (بنية htp_copy_context *) داده;                         \
    بنية htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    ثابت uint32_t dr  = ct->src0_nrows_در_هر_thread;                                          \
    ثابت uint32_t ir0 = ct->ردیف_شروع + dr * ith;                                           \
    ثابت uint32_t ir1 = MIN(ir0 + dr, ct->ردیف_شروع + ct->nrows);                           \
    إذا (ir0 >= ir1) أرجع;                                                                  \
                                                                                             \
    dma_صف * dma_q = octx->ctx->dma[ith];                                                 \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    ثابت بنية htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    ثابت uint32_t src0_buf_اندازه = cvt->src0_buf_اندازه;                                       \
    ثابت uint32_t dst_buf_اندازه  = cvt->dst_buf_اندازه;                                        \
    ثابت uint32_t src0_ردیف_stride = cvt->src0_ردیف_stride;                                   \
    ثابت uint32_t dst_ردیف_stride  = cvt->dst_ردیف_stride;                                    \
    ثابت uint32_t blk_ردیفها        = cvt->blk_ردیفها;                                          \
    uint8_t * vtcm_src0_پایه = ct->vtcm_src0 + ith * cvt->spad0_اندازه_در_هر_thread;             \
    uint8_t * vtcm_dst_پایه  = ct->vtcm_dst  + ith * cvt->spad1_اندازه_در_هر_thread;             \
    ثابت uint32_t src0_ردیف_اندازه = ne00 * ct->kparams->src0_type_اندازه;                       \
    ثابت uint32_t dst_ردیف_اندازه  = ne00 * ct->kparams->dst_type_اندازه;                        \
                                                                                             \
    ثابت uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    بنية cpy_ردیف_cursor f = { ir0, i01, i02, i03 };                                        \
    بنية cpy_ردیف_cursor c = f;                                                             \
                                                                                             \
    لكل (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_پایه + b * src0_buf_اندازه;                             \
        uint8_t * dst_spad = vtcm_dst_پایه  + b * dst_buf_اندازه;                              \
        ثابت uint32_t nr = MIN(blk_ردیفها, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->داده + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_صف_افزودن(dma_q, dma_make_داده(dst->داده, dst_spad),                            \
                       dst_ردیف_اندازه, dst_buf_اندازه, dst_ردیف_اندازه, 0);                         \
        dma_صف_افزودن(dma_q, dma_make_داده(src_spad, f_src0_addr),                          \
                       src0_ردیف_stride, nb01, src0_ردیف_اندازه, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    لكل (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        ثابت uint32_t nr = MIN(blk_ردیفها, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_صف_برداشت(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_صف_برداشت(dma_q).dst;               \
                                                                                             \
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        لكل (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_ردیف_stride, src_spad + r * src0_ردیف_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->داده + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_صف_افزودن(dma_q, dma_make_داده(c_dst_addr, dst_spad),                           \
                       nb1, dst_ردیف_stride, dst_ردیف_اندازه, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        إذا (f.r < ir1) {                                                                     \
            ثابت uint32_t fnr = MIN(blk_ردیفها, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->داده + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_صف_افزودن(dma_q, dma_make_داده(src_spad, f_src0_addr),                      \
                           src0_ردیف_stride, nb01, src0_ردیف_اندازه, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_صف_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

ساكن صحيح cpy_scalar(بنية htp_ops_context * octx, ثابت بنية htp_copy_kernel_params * kparams) {
    إذا (octx->ctx->mdev.شمار > 1 && octx->ctx->mdev.idx > 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->داده, src0->داده, kparams->src0_type_اندازه);
        dma_صف_flush(octx->ctx->dma[0]);
        أرجع HTP_وضعیت_OK;
    }

    dma_صف * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_پایه;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    ثابت uint32_t s_اندازه = kparams->src0_type_اندازه;
    ثابت uint32_t d_اندازه = kparams->dst_type_اندازه;

    dma_صف_افزودن(dma_q, dma_make_داده(s_vtcm, src0->داده), s_اندازه, s_اندازه, s_اندازه, 1);
    dma_صف_برداشت(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_پایه;
    uint8_t * d_ptr = s_ptr + VLEN;

    ثابت HVX_وکتور v_src = hvx_vmem(s_ptr);
    HVX_وکتور v_dst;

    إذا (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } وإلا إذا (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } وإلا إذا (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_مقصد_f16(v_src, v_src);
    } وإلا إذا (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(v_src));
    } وإلا {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_صف_افزودن(dma_q, dma_make_داده(dst->داده, d_vtcm), d_اندازه, d_اندازه, d_اندازه, 1);
    dma_صف_flush(dma_q);
    أرجع HTP_وضعیت_OK;
}

ساكن صحيح cpy_1d_contig(بنية htp_ops_context * octx, ثابت بنية htp_copy_kernel_params * kparams) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    uint32_t elem_شروع = 0;
    uint32_t nelem      = kparams->مجموع_elems;

    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(
            kparams->مجموع_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        elem_شروع = range.شروع;
        nelem      = range.شمار;
    }

    إذا (nelem > 0) {
        dma_صف * q = octx->ctx->dma[0];
        ثابت uint32_t type_اندازه = kparams->src0_type_اندازه;
        dma_addr_t dst_addr  = dst->داده  + elem_شروع * type_اندازه;
        dma_addr_t src0_addr = src0->داده + elem_شروع * type_اندازه;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_اندازه);
        dma_صف_flush(q);
    }

    أرجع HTP_وضعیت_OK;
}

ساكن صحيح cpy_sameshape_sametype(بنية htp_ops_context * octx, ثابت بنية htp_copy_kernel_params * kparams) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = kparams->مجموع_ردیفها;

    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(
            kparams->مجموع_ردیفها, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows > 0) {
        dma_صف * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_اندازه, ردیف_شروع, nrows);
        dma_صف_flush(q);
    }

    أرجع HTP_وضعیت_OK;
}

ساكن صحيح cpy_sameshape_convert(بنية htp_ops_context * octx, ثابت بنية htp_copy_kernel_params * kparams) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (!htp_ops_context_مجموعه_n_نخها(octx, kparams->n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = kparams->مجموع_ردیفها;

    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(
            kparams->مجموع_ردیفها, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    إذا (kparams->vtcm_اندازه > octx->ctx->vtcm_اندازه) {
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;
    ثابت بنية htp_copy_convert_params * cvt = &kparams->u.convert;

    بنية htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.ردیف_شروع             = ردیف_شروع;
    ct.nrows                 = nrows;
    ct.src0_nrows_در_هر_thread = fastdiv(nrows + n_نخها - 1, &octx->n_نخها_div);

    uint8_t * vtcm_پایه = (uint8_t *) octx->ctx->vtcm_پایه;
    ct.vtcm_src0 = vtcm_پایه;
    ct.vtcm_dst  = vtcm_پایه + (size_t) n_نخها * cvt->spad0_اندازه_در_هر_thread;

    work_صف_func_t copy_fun = عدم;
    إذا (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } وإلا إذا (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } وإلا إذا (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } وإلا إذا (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } وإلا {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    work_صف_پیاپی(octx->ctx->work_صف, copy_fun, &ct, n_نخها);
    أرجع HTP_وضعیت_OK;
}

ساكن صحيح cpy_reshape(بنية htp_ops_context * octx, ثابت بنية htp_copy_kernel_params * kparams) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (!htp_ops_context_مجموعه_n_نخها(octx, kparams->n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    uint32_t elem_شروع = 0;
    uint32_t nelem      = kparams->مجموع_elems;

    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(
            kparams->مجموع_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        elem_شروع = range.شروع;
        nelem      = range.شمار;
    }

    إذا (nelem == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;

    بنية htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_شروع      = elem_شروع;
    ct.nelem           = nelem;
    ct.elem_در_هر_thread = fastdiv(nelem + n_نخها - 1, &octx->n_نخها_div);

    work_صف_func_t copy_fun = عدم;
    بدل (src0->type) {
        حالة HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; اكسر;
        حالة HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; اكسر;
        حالة HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; اكسر;
        افتراضي: أرجع HTP_وضعیت_NO_SUPPORT;
    }

    work_صف_پیاپی(octx->ctx->work_صف, copy_fun, &ct, n_نخها);
    أرجع HTP_وضعیت_OK;
}

صحيح op_cpy(بنية htp_ops_context * octx) {
    ثابت بنية htp_copy_kernel_params * kparams = (ثابت بنية htp_copy_kernel_params *) octx->kernel_params;
    صحيح وضعیت = HTP_وضعیت_OK;

    بدل (kparams->kernel_type) {
        حالة HTP_COPY_KERNEL_SCALAR:
            وضعیت = cpy_scalar(octx, kparams);
            اكسر;
        حالة HTP_COPY_KERNEL_1D_CONTIG:
            وضعیت = cpy_1d_contig(octx, kparams);
            اكسر;
        حالة HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            وضعیت = cpy_sameshape_sametype(octx, kparams);
            اكسر;
        حالة HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            وضعیت = cpy_sameshape_convert(octx, kparams);
            اكسر;
        حالة HTP_COPY_KERNEL_RESHAPE:
            وضعیت = cpy_reshape(octx, kparams);
            اكسر;
        حالة HTP_COPY_KERNEL_GATHER_ردیفها:
            وضعیت = htp_gather_ردیفها(octx, octx->src[0], عدم, &kparams->u.gather, kparams->n_نخها);
            اكسر;
        افتراضي:
            وضعیت = HTP_وضعیت_NO_SUPPORT;
            اكسر;
    }

    htp_ops_context_مجموعه_وضعیت(octx, وضعیت);

    إذا (octx->op == HTP_OP_CPY_FENCE) {
        إذا (octx->ctx->mdev.شمار > 1) {
            htp_mdev_گروه_barrier(octx);
        }

        إذا (octx->ctx->mdev.idx == 0) {
            ثابت بنية htp_tensor * sync = octx->src[1];
            ثابت uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->داده;
            htp_fence_write(sync_fence, seq, octx->وضعیت);

            FARF(بالا_2, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->وضعیت);
        }
    }

    أرجع octx->وضعیت;
}
