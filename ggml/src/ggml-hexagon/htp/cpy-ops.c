#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>
#ضمن <qurt_memory.h>

#ضمن <math.h>
#ضمن <string.h>

#وصف_ڪريو GGML_COMMON_DECL_C
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

    uint32_t صف_شروعات;
    uint32_t nrows;
    uint32_t src0_nrows_في_thread;

    uint32_t elem_شروعات;
    uint32_t nelem;
    uint32_t elem_في_thread;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#وصف_ڪريو cpy_preamble                              \
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

#وصف_ڪريو DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_سائيز)                                                \
ساكن فراغ cpy_thread_##NAME##_reshape(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {            \
    بنية htp_copy_context * ct = (بنية htp_copy_context *) معلومات;                                  \
    بنية htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    ثابت uint32_t th_nelem = ct->elem_في_thread;                                                    \
    ثابت uint32_t th_شروعات = ct->elem_شروعات + ith * th_nelem;                                        \
    ثابت uint32_t th_پڇاڙي   = MIN(th_شروعات + th_nelem, ct->elem_شروعات + ct->nelem);                   \
    إذا (th_شروعات >= th_پڇاڙي) أرجع;                                                                   \
                                                                                                      \
    dma_قطار * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    ثابت uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    ثابت uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    ثابت uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    ثابت uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    ثابت بنية htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
    uint32_t e = th_شروعات;                                                                            \
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
    dma_addr_t dst_addr  = dst->معلومات  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->معلومات + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    ثابت bool صفون_contig = (nb00 == ELEM_سائيز) && (nb0 == ELEM_سائيز);                               \
                                                                                                      \
    طالما (e < th_پڇاڙي) {                                                                              \
        ثابت uint32_t لاڳيتو = MIN(MIN(ne00 - i00, ne0 - i10), th_پڇاڙي - e);                             \
        إذا (صفون_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, لاڳيتو * ELEM_سائيز);             \
        } وإلا {                                                                                      \
            dma_cpy_ڌڪو_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_سائيز, لاڳيتو);           \
        }                                                                                             \
        e += لاڳيتو;                                                                                     \
                                                                                                      \
        dst_addr += لاڳيتو * nb0;                                                                        \
        i10      += لاڳيتو;                                                                              \
        إذا (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            إذا (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                إذا (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->معلومات + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += لاڳيتو * nb00;                                                                      \
        i00       += لاڳيتو;                                                                             \
        إذا (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            إذا (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                إذا (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_قطار_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  عائم, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

بنية cpy_صف_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ساكن مضمن فراغ cpy_cursor_advance(بنية cpy_صف_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
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

#وصف_ڪريو DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
ساكن فراغ cpy_thread_##NAME##_sameshape(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) { \
    بنية htp_copy_context * ct = (بنية htp_copy_context *) معلومات;                         \
    بنية htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    ثابت uint32_t dr  = ct->src0_nrows_في_thread;                                          \
    ثابت uint32_t ir0 = ct->صف_شروعات + dr * ith;                                           \
    ثابت uint32_t ir1 = MIN(ir0 + dr, ct->صف_شروعات + ct->nrows);                           \
    إذا (ir0 >= ir1) أرجع;                                                                  \
                                                                                             \
    dma_قطار * dma_q = octx->ctx->dma[ith];                                                 \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    ثابت بنية htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    ثابت uint32_t src0_buf_سائيز = cvt->src0_buf_سائيز;                                       \
    ثابت uint32_t dst_buf_سائيز  = cvt->dst_buf_سائيز;                                        \
    ثابت uint32_t src0_صف_stride = cvt->src0_صف_stride;                                   \
    ثابت uint32_t dst_صف_stride  = cvt->dst_صف_stride;                                    \
    ثابت uint32_t blk_صفون        = cvt->blk_صفون;                                          \
    uint8_t * vtcm_src0_بنياد = ct->vtcm_src0 + ith * cvt->spad0_سائيز_في_thread;             \
    uint8_t * vtcm_dst_بنياد  = ct->vtcm_dst  + ith * cvt->spad1_سائيز_في_thread;             \
    ثابت uint32_t src0_صف_سائيز = ne00 * ct->kparams->src0_type_سائيز;                       \
    ثابت uint32_t dst_صف_سائيز  = ne00 * ct->kparams->dst_type_سائيز;                        \
                                                                                             \
    ثابت uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    بنية cpy_صف_cursor f = { ir0, i01, i02, i03 };                                        \
    بنية cpy_صف_cursor c = f;                                                             \
                                                                                             \
    لكل (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_بنياد + b * src0_buf_سائيز;                             \
        uint8_t * dst_spad = vtcm_dst_بنياد  + b * dst_buf_سائيز;                              \
        ثابت uint32_t nr = MIN(blk_صفون, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->معلومات + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst->معلومات, dst_spad),                            \
                       dst_صف_سائيز, dst_buf_سائيز, dst_صف_سائيز, 0);                         \
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(src_spad, f_src0_addr),                          \
                       src0_صف_stride, nb01, src0_صف_سائيز, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    لكل (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        ثابت uint32_t nr = MIN(blk_صفون, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_قطار_ڪڍو(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_قطار_ڪڍو(dma_q).dst;               \
                                                                                             \
        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        لكل (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_صف_stride, src_spad + r * src0_صف_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->معلومات + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(c_dst_addr, dst_spad),                           \
                       nb1, dst_صف_stride, dst_صف_سائيز, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        إذا (f.r < ir1) {                                                                     \
            ثابت uint32_t fnr = MIN(blk_صفون, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->معلومات + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(src_spad, f_src0_addr),                      \
                           src0_صف_stride, nb01, src0_صف_سائيز, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_قطار_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

ساكن صحيح cpy_scalar(بنية htp_ops_context * octx, ثابت بنية htp_copy_kernel_params * kparams) {
    إذا (octx->ctx->mdev.ڳڻپ > 1 && octx->ctx->mdev.idx > 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->معلومات, src0->معلومات, kparams->src0_type_سائيز);
        dma_قطار_flush(octx->ctx->dma[0]);
        أرجع HTP_حالت_OK;
    }

    dma_قطار * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_بنياد;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    ثابت uint32_t s_سائيز = kparams->src0_type_سائيز;
    ثابت uint32_t d_سائيز = kparams->dst_type_سائيز;

    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_vtcm, src0->معلومات), s_سائيز, s_سائيز, s_سائيز, 1);
    dma_قطار_ڪڍو(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_بنياد;
    uint8_t * d_ptr = s_ptr + VLEN;

    ثابت HVX_ويڪٽر v_src = hvx_vmem(s_ptr);
    HVX_ويڪٽر v_dst;

    إذا (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } وإلا إذا (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } وإلا إذا (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_منزل_f16(v_src, v_src);
    } وإلا إذا (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_منزل_f32(v_src));
    } وإلا {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst->معلومات, d_vtcm), d_سائيز, d_سائيز, d_سائيز, 1);
    dma_قطار_flush(dma_q);
    أرجع HTP_حالت_OK;
}

ساكن صحيح cpy_1d_contig(بنية htp_ops_context * octx, ثابت بنية htp_copy_kernel_params * kparams) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    uint32_t elem_شروعات = 0;
    uint32_t nelem      = kparams->ڪل_elems;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(
            kparams->ڪل_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        elem_شروعات = range.شروعات;
        nelem      = range.ڳڻپ;
    }

    إذا (nelem > 0) {
        dma_قطار * q = octx->ctx->dma[0];
        ثابت uint32_t type_سائيز = kparams->src0_type_سائيز;
        dma_addr_t dst_addr  = dst->معلومات  + elem_شروعات * type_سائيز;
        dma_addr_t src0_addr = src0->معلومات + elem_شروعات * type_سائيز;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_سائيز);
        dma_قطار_flush(q);
    }

    أرجع HTP_حالت_OK;
}

ساكن صحيح cpy_sameshape_sametype(بنية htp_ops_context * octx, ثابت بنية htp_copy_kernel_params * kparams) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    uint32_t صف_شروعات = 0;
    uint32_t nrows     = kparams->ڪل_صفون;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(
            kparams->ڪل_صفون, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    }

    إذا (nrows > 0) {
        dma_قطار * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_سائيز, صف_شروعات, nrows);
        dma_قطار_flush(q);
    }

    أرجع HTP_حالت_OK;
}

ساكن صحيح cpy_sameshape_convert(بنية htp_ops_context * octx, ثابت بنية htp_copy_kernel_params * kparams) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (!htp_ops_context_سيٽ_n_ڌاڳا(octx, kparams->n_ڌاڳا)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    uint32_t صف_شروعات = 0;
    uint32_t nrows     = kparams->ڪل_صفون;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(
            kparams->ڪل_صفون, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (kparams->vtcm_سائيز > octx->ctx->vtcm_سائيز) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;
    ثابت بنية htp_copy_convert_params * cvt = &kparams->u.convert;

    بنية htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.صف_شروعات             = صف_شروعات;
    ct.nrows                 = nrows;
    ct.src0_nrows_في_thread = fastdiv(nrows + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);

    uint8_t * vtcm_بنياد = (uint8_t *) octx->ctx->vtcm_بنياد;
    ct.vtcm_src0 = vtcm_بنياد;
    ct.vtcm_dst  = vtcm_بنياد + (size_t) n_ڌاڳا * cvt->spad0_سائيز_في_thread;

    work_قطار_func_t copy_fun = عدم;
    إذا (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } وإلا إذا (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } وإلا إذا (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } وإلا إذا (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } وإلا {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    work_قطار_لاڳيتو(octx->ctx->work_قطار, copy_fun, &ct, n_ڌاڳا);
    أرجع HTP_حالت_OK;
}

ساكن صحيح cpy_reshape(بنية htp_ops_context * octx, ثابت بنية htp_copy_kernel_params * kparams) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (!htp_ops_context_سيٽ_n_ڌاڳا(octx, kparams->n_ڌاڳا)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    uint32_t elem_شروعات = 0;
    uint32_t nelem      = kparams->ڪل_elems;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(
            kparams->ڪل_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        elem_شروعات = range.شروعات;
        nelem      = range.ڳڻپ;
    }

    إذا (nelem == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

    بنية htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_شروعات      = elem_شروعات;
    ct.nelem           = nelem;
    ct.elem_في_thread = fastdiv(nelem + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);

    work_قطار_func_t copy_fun = عدم;
    بدل (src0->type) {
        حالة HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; اكسر;
        حالة HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; اكسر;
        حالة HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; اكسر;
        افتراضي: أرجع HTP_حالت_NO_SUPPORT;
    }

    work_قطار_لاڳيتو(octx->ctx->work_قطار, copy_fun, &ct, n_ڌاڳا);
    أرجع HTP_حالت_OK;
}

صحيح op_cpy(بنية htp_ops_context * octx) {
    ثابت بنية htp_copy_kernel_params * kparams = (ثابت بنية htp_copy_kernel_params *) octx->kernel_params;
    صحيح حالت = HTP_حالت_OK;

    بدل (kparams->kernel_type) {
        حالة HTP_COPY_KERNEL_SCALAR:
            حالت = cpy_scalar(octx, kparams);
            اكسر;
        حالة HTP_COPY_KERNEL_1D_CONTIG:
            حالت = cpy_1d_contig(octx, kparams);
            اكسر;
        حالة HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            حالت = cpy_sameshape_sametype(octx, kparams);
            اكسر;
        حالة HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            حالت = cpy_sameshape_convert(octx, kparams);
            اكسر;
        حالة HTP_COPY_KERNEL_RESHAPE:
            حالت = cpy_reshape(octx, kparams);
            اكسر;
        حالة HTP_COPY_KERNEL_GATHER_صفون:
            حالت = htp_gather_صفون(octx, octx->src[0], عدم, &kparams->u.gather, kparams->n_ڌاڳا);
            اكسر;
        افتراضي:
            حالت = HTP_حالت_NO_SUPPORT;
            اكسر;
    }

    htp_ops_context_سيٽ_حالت(octx, حالت);

    إذا (octx->op == HTP_OP_CPY_FENCE) {
        إذا (octx->ctx->mdev.ڳڻپ > 1) {
            htp_mdev_گروهه_barrier(octx);
        }

        إذا (octx->ctx->mdev.idx == 0) {
            ثابت بنية htp_tensor * sync = octx->src[1];
            ثابت uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->معلومات;
            htp_fence_write(sync_fence, seq, octx->حالت);

            FARF(مٿاهون_2, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->حالت);
        }
    }

    أرجع octx->حالت;
}
