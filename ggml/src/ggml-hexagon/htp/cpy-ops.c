#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_memory.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "cpy-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-copy.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-fence.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "gather-rows.h"

ਬਣਤਰ htp_copy_context {
    ਬਣਤਰ htp_ops_context *              octx;
    ਅਚਲ ਬਣਤਰ htp_copy_kernel_params * kparams;

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t nrows;
    uint32_t src0_nrows_ਪ੍ਰਤੀ_thread;

    uint32_t elem_ਸ਼ੁਰੂ;
    uint32_t nelem;
    uint32_t elem_ਪ੍ਰਤੀ_thread;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cpy_preamble                              \
    ਅਚਲ ਬਣਤਰ htp_tensor *src0 = octx->src[0]; \
    ਅਚਲ ਬਣਤਰ htp_tensor *dst  = octx->dst;    \
                                                  \
    ਅਚਲ uint32_t ne00 = src0->ne[0];            \
    ਅਚਲ uint32_t ne01 = src0->ne[1];            \
    ਅਚਲ uint32_t ne02 = src0->ne[2];            \
    ਅਚਲ uint32_t ne03 = src0->ne[3];            \
                                                  \
    ਅਚਲ uint32_t nb00 = src0->nb[0];            \
    ਅਚਲ uint32_t nb01 = src0->nb[1];            \
    ਅਚਲ uint32_t nb02 = src0->nb[2];            \
    ਅਚਲ uint32_t nb03 = src0->nb[3];            \
                                                  \
    ਅਚਲ uint32_t  ne0 = dst->ne[0];             \
    ਅਚਲ uint32_t  ne1 = dst->ne[1];             \
    ਅਚਲ uint32_t  ne2 = dst->ne[2];             \
    ਅਚਲ uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ਅਚਲ uint32_t  nb0 = dst->nb[0];             \
    ਅਚਲ uint32_t  nb1 = dst->nb[1];             \
    ਅਚਲ uint32_t  nb2 = dst->nb[2];             \
    ਅਚਲ uint32_t  nb3 = dst->nb[3];

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_ਮਾਪ)                                                \
ਸਥਿਰ ਖਾਲੀ cpy_thread_##NAME##_reshape(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {            \
    ਬਣਤਰ htp_copy_context * ct = (ਬਣਤਰ htp_copy_context *) ਅੰਕੜੇ;                                  \
    ਬਣਤਰ htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    ਅਚਲ uint32_t th_nelem = ct->elem_ਪ੍ਰਤੀ_thread;                                                    \
    ਅਚਲ uint32_t th_ਸ਼ੁਰੂ = ct->elem_ਸ਼ੁਰੂ + ith * th_nelem;                                        \
    ਅਚਲ uint32_t th_ਸਿਰਾ   = MIN(th_ਸ਼ੁਰੂ + th_nelem, ct->elem_ਸ਼ੁਰੂ + ct->nelem);                   \
    ਜੇ (th_ਸ਼ੁਰੂ >= th_ਸਿਰਾ) ਵਾਪਸ;                                                                   \
                                                                                                      \
    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    ਅਚਲ uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    ਅਚਲ uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    ਅਚਲ uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    ਅਚਲ uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    ਅਚਲ ਬਣਤਰ htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
    uint32_t e = th_ਸ਼ੁਰੂ;                                                                            \
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
    dma_addr_t dst_addr  = dst->ਅੰਕੜੇ  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->ਅੰਕੜੇ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    ਅਚਲ bool ਪੰਕਤੀਆਂ_contig = (nb00 == ELEM_ਮਾਪ) && (nb0 == ELEM_ਮਾਪ);                               \
                                                                                                      \
    ਜਦੋਂ (e < th_ਸਿਰਾ) {                                                                              \
        ਅਚਲ uint32_t ਲੜੀਵਾਰ = MIN(MIN(ne00 - i00, ne0 - i10), th_ਸਿਰਾ - e);                             \
        ਜੇ (ਪੰਕਤੀਆਂ_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, ਲੜੀਵਾਰ * ELEM_ਮਾਪ);             \
        } ਨਹੀਂਤਾਂ {                                                                                      \
            dma_cpy_ਧੱਕੋ_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_ਮਾਪ, ਲੜੀਵਾਰ);           \
        }                                                                                             \
        e += ਲੜੀਵਾਰ;                                                                                     \
                                                                                                      \
        dst_addr += ਲੜੀਵਾਰ * nb0;                                                                        \
        i10      += ਲੜੀਵਾਰ;                                                                              \
        ਜੇ (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            ਜੇ (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                ਜੇ (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->ਅੰਕੜੇ + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += ਲੜੀਵਾਰ * nb00;                                                                      \
        i00       += ਲੜੀਵਾਰ;                                                                             \
        ਜੇ (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            ਜੇ (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                ਜੇ (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_ਕਤਾਰ_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  ਦਸ਼ਮਲਵ, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

ਬਣਤਰ cpy_ਪੰਕਤੀ_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_cursor_advance(ਬਣਤਰ cpy_ਪੰਕਤੀ_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
    c->r   += n;
    c->i01 += n;
    ਜੇ (c->i01 == ne01) {
        c->i01 = 0;
        ਜੇ (++c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
ਸਥਿਰ ਖਾਲੀ cpy_thread_##NAME##_sameshape(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) { \
    ਬਣਤਰ htp_copy_context * ct = (ਬਣਤਰ htp_copy_context *) ਅੰਕੜੇ;                         \
    ਬਣਤਰ htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    ਅਚਲ uint32_t dr  = ct->src0_nrows_ਪ੍ਰਤੀ_thread;                                          \
    ਅਚਲ uint32_t ir0 = ct->ਪੰਕਤੀ_ਸ਼ੁਰੂ + dr * ith;                                           \
    ਅਚਲ uint32_t ir1 = MIN(ir0 + dr, ct->ਪੰਕਤੀ_ਸ਼ੁਰੂ + ct->nrows);                           \
    ਜੇ (ir0 >= ir1) ਵਾਪਸ;                                                                  \
                                                                                             \
    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];                                                 \
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    ਅਚਲ ਬਣਤਰ htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    ਅਚਲ uint32_t src0_buf_ਮਾਪ = cvt->src0_buf_ਮਾਪ;                                       \
    ਅਚਲ uint32_t dst_buf_ਮਾਪ  = cvt->dst_buf_ਮਾਪ;                                        \
    ਅਚਲ uint32_t src0_ਪੰਕਤੀ_stride = cvt->src0_ਪੰਕਤੀ_stride;                                   \
    ਅਚਲ uint32_t dst_ਪੰਕਤੀ_stride  = cvt->dst_ਪੰਕਤੀ_stride;                                    \
    ਅਚਲ uint32_t blk_ਪੰਕਤੀਆਂ        = cvt->blk_ਪੰਕਤੀਆਂ;                                          \
    uint8_t * vtcm_src0_ਅਧਾਰ = ct->vtcm_src0 + ith * cvt->spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;             \
    uint8_t * vtcm_dst_ਅਧਾਰ  = ct->vtcm_dst  + ith * cvt->spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread;             \
    ਅਚਲ uint32_t src0_ਪੰਕਤੀ_ਮਾਪ = ne00 * ct->kparams->src0_type_ਮਾਪ;                       \
    ਅਚਲ uint32_t dst_ਪੰਕਤੀ_ਮਾਪ  = ne00 * ct->kparams->dst_type_ਮਾਪ;                        \
                                                                                             \
    ਅਚਲ uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    ਬਣਤਰ cpy_ਪੰਕਤੀ_cursor f = { ir0, i01, i02, i03 };                                        \
    ਬਣਤਰ cpy_ਪੰਕਤੀ_cursor c = f;                                                             \
                                                                                             \
    ਲਈ (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_ਅਧਾਰ + b * src0_buf_ਮਾਪ;                             \
        uint8_t * dst_spad = vtcm_dst_ਅਧਾਰ  + b * dst_buf_ਮਾਪ;                              \
        ਅਚਲ uint32_t nr = MIN(blk_ਪੰਕਤੀਆਂ, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->ਅੰਕੜੇ + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst->ਅੰਕੜੇ, dst_spad),                            \
                       dst_ਪੰਕਤੀ_ਮਾਪ, dst_buf_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ, 0);                         \
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(src_spad, f_src0_addr),                          \
                       src0_ਪੰਕਤੀ_stride, nb01, src0_ਪੰਕਤੀ_ਮਾਪ, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    ਲਈ (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        ਅਚਲ uint32_t nr = MIN(blk_ਪੰਕਤੀਆਂ, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;               \
                                                                                             \
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        ਲਈ (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_ਪੰਕਤੀ_stride, src_spad + r * src0_ਪੰਕਤੀ_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->ਅੰਕੜੇ + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(c_dst_addr, dst_spad),                           \
                       nb1, dst_ਪੰਕਤੀ_stride, dst_ਪੰਕਤੀ_ਮਾਪ, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        ਜੇ (f.r < ir1) {                                                                     \
            ਅਚਲ uint32_t fnr = MIN(blk_ਪੰਕਤੀਆਂ, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->ਅੰਕੜੇ + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(src_spad, f_src0_addr),                      \
                           src0_ਪੰਕਤੀ_stride, nb01, src0_ਪੰਕਤੀ_ਮਾਪ, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_ਕਤਾਰ_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

ਸਥਿਰ ਪੂਰਨ cpy_scalar(ਬਣਤਰ htp_ops_context * octx, ਅਚਲ ਬਣਤਰ htp_copy_kernel_params * kparams) {
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1 && octx->ctx->mdev.idx > 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਜੇ (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->ਅੰਕੜੇ, src0->ਅੰਕੜੇ, kparams->src0_type_ਮਾਪ);
        dma_ਕਤਾਰ_flush(octx->ctx->dma[0]);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_ਅਧਾਰ;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    ਅਚਲ uint32_t s_ਮਾਪ = kparams->src0_type_ਮਾਪ;
    ਅਚਲ uint32_t d_ਮਾਪ = kparams->dst_type_ਮਾਪ;

    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_vtcm, src0->ਅੰਕੜੇ), s_ਮਾਪ, s_ਮਾਪ, s_ਮਾਪ, 1);
    dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    uint8_t * d_ptr = s_ptr + VLEN;

    ਅਚਲ HVX_ਸਦਿਸ਼ v_src = hvx_vmem(s_ptr);
    HVX_ਸਦਿਸ਼ v_dst;

    ਜੇ (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v_src, v_src);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_src));
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst->ਅੰਕੜੇ, d_vtcm), d_ਮਾਪ, d_ਮਾਪ, d_ਮਾਪ, 1);
    dma_ਕਤਾਰ_flush(dma_q);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਪੂਰਨ cpy_1d_contig(ਬਣਤਰ htp_ops_context * octx, ਅਚਲ ਬਣਤਰ htp_copy_kernel_params * kparams) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    uint32_t elem_ਸ਼ੁਰੂ = 0;
    uint32_t nelem      = kparams->ਕੁੱਲ_elems;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            kparams->ਕੁੱਲ_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        elem_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nelem      = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nelem > 0) {
        dma_ਕਤਾਰ * q = octx->ctx->dma[0];
        ਅਚਲ uint32_t type_ਮਾਪ = kparams->src0_type_ਮਾਪ;
        dma_addr_t dst_addr  = dst->ਅੰਕੜੇ  + elem_ਸ਼ੁਰੂ * type_ਮਾਪ;
        dma_addr_t src0_addr = src0->ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ * type_ਮਾਪ;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_ਮਾਪ);
        dma_ਕਤਾਰ_flush(q);
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਪੂਰਨ cpy_sameshape_sametype(ਬਣਤਰ htp_ops_context * octx, ਅਚਲ ਬਣਤਰ htp_copy_kernel_params * kparams) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows > 0) {
        dma_ਕਤਾਰ * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_ਮਾਪ, ਪੰਕਤੀ_ਸ਼ੁਰੂ, nrows);
        dma_ਕਤਾਰ_flush(q);
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਪੂਰਨ cpy_sameshape_convert(ਬਣਤਰ htp_ops_context * octx, ਅਚਲ ਬਣਤਰ htp_copy_kernel_params * kparams) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਜੇ (kparams->vtcm_ਮਾਪ > octx->ctx->vtcm_ਮਾਪ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;
    ਅਚਲ ਬਣਤਰ htp_copy_convert_params * cvt = &kparams->u.convert;

    ਬਣਤਰ htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.ਪੰਕਤੀ_ਸ਼ੁਰੂ             = ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    ct.nrows                 = nrows;
    ct.src0_nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);

    uint8_t * vtcm_ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    ct.vtcm_src0 = vtcm_ਅਧਾਰ;
    ct.vtcm_dst  = vtcm_ਅਧਾਰ + (size_t) n_ਧਾਗੇ * cvt->spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;

    work_ਕਤਾਰ_func_t copy_fun = ਨਿਰਾ;
    ਜੇ (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } ਨਹੀਂਤਾਂ ਜੇ (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } ਨਹੀਂਤਾਂ ਜੇ (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } ਨਹੀਂਤਾਂ ਜੇ (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, copy_fun, &ct, n_ਧਾਗੇ);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਪੂਰਨ cpy_reshape(ਬਣਤਰ htp_ops_context * octx, ਅਚਲ ਬਣਤਰ htp_copy_kernel_params * kparams) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    uint32_t elem_ਸ਼ੁਰੂ = 0;
    uint32_t nelem      = kparams->ਕੁੱਲ_elems;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            kparams->ਕੁੱਲ_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        elem_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nelem      = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nelem == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    ਬਣਤਰ htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_ਸ਼ੁਰੂ      = elem_ਸ਼ੁਰੂ;
    ct.nelem           = nelem;
    ct.elem_ਪ੍ਰਤੀ_thread = fastdiv(nelem + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);

    work_ਕਤਾਰ_func_t copy_fun = ਨਿਰਾ;
    ਚੋਣ (src0->type) {
        ਹਾਲਤ HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; ਰੋਕੋ;
        ਹਾਲਤ HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; ਰੋਕੋ;
        ਹਾਲਤ HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; ਰੋਕੋ;
        ਮੂਲ: ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, copy_fun, &ct, n_ਧਾਗੇ);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_cpy(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_copy_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_copy_kernel_params *) octx->kernel_params;
    ਪੂਰਨ ਸਥਿਤੀ = HTP_ਸਥਿਤੀ_OK;

    ਚੋਣ (kparams->kernel_type) {
        ਹਾਲਤ HTP_COPY_KERNEL_SCALAR:
            ਸਥਿਤੀ = cpy_scalar(octx, kparams);
            ਰੋਕੋ;
        ਹਾਲਤ HTP_COPY_KERNEL_1D_CONTIG:
            ਸਥਿਤੀ = cpy_1d_contig(octx, kparams);
            ਰੋਕੋ;
        ਹਾਲਤ HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            ਸਥਿਤੀ = cpy_sameshape_sametype(octx, kparams);
            ਰੋਕੋ;
        ਹਾਲਤ HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            ਸਥਿਤੀ = cpy_sameshape_convert(octx, kparams);
            ਰੋਕੋ;
        ਹਾਲਤ HTP_COPY_KERNEL_RESHAPE:
            ਸਥਿਤੀ = cpy_reshape(octx, kparams);
            ਰੋਕੋ;
        ਹਾਲਤ HTP_COPY_KERNEL_GATHER_ਪੰਕਤੀਆਂ:
            ਸਥਿਤੀ = htp_gather_ਪੰਕਤੀਆਂ(octx, octx->src[0], ਨਿਰਾ, &kparams->u.gather, kparams->n_ਧਾਗੇ);
            ਰੋਕੋ;
        ਮੂਲ:
            ਸਥਿਤੀ = HTP_ਸਥਿਤੀ_NO_SUPPORT;
            ਰੋਕੋ;
    }

    htp_ops_context_ਸੈੱਟ_ਸਥਿਤੀ(octx, ਸਥਿਤੀ);

    ਜੇ (octx->op == HTP_OP_CPY_FENCE) {
        ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
            htp_mdev_ਸਮੂਹ_barrier(octx);
        }

        ਜੇ (octx->ctx->mdev.idx == 0) {
            ਅਚਲ ਬਣਤਰ htp_tensor * sync = octx->src[1];
            ਅਚਲ uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->ਅੰਕੜੇ;
            htp_fence_write(sync_fence, seq, octx->ਸਥਿਤੀ);

            FARF(ਉੱਚਾ_2, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->ਸਥਿਤੀ);
        }
    }

    ਵਾਪਸ octx->ਸਥਿਤੀ;
}
