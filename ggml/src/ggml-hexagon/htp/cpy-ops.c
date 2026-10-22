#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>
#ಅನ್ತರ್ಭಾವ <qurt_memory.h>

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "cpy-ops.h"
#ಅನ್ತರ್ಭಾವ "dma-copy.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-fence.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "gather-rows.h"

ರಚನೆ htp_copy_context {
    ರಚನೆ htp_ops_context *              octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_kernel_params * kparams;

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t nrows;
    uint32_t src0_nrows_ಪ್ರತಿ_thread;

    uint32_t elem_ಆರಂಭ;
    uint32_t nelem;
    uint32_t elem_ಪ್ರತಿ_thread;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#ವ್ಯಾಖ್ಯಾನಿಸು cpy_preamble                              \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor *src0 = octx->src[0]; \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor *dst  = octx->dst;    \
                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];            \
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src0->ne[1];            \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];            \
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3];            \
                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t nb00 = src0->nb[0];            \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src0->nb[1];            \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2];            \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3];            \
                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t  ne0 = dst->ne[0];             \
    ಸ್ಥಿರಾಂಕ uint32_t  ne1 = dst->ne[1];             \
    ಸ್ಥಿರಾಂಕ uint32_t  ne2 = dst->ne[2];             \
    ಸ್ಥಿರಾಂಕ uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t  nb0 = dst->nb[0];             \
    ಸ್ಥಿರಾಂಕ uint32_t  nb1 = dst->nb[1];             \
    ಸ್ಥಿರಾಂಕ uint32_t  nb2 = dst->nb[2];             \
    ಸ್ಥಿರಾಂಕ uint32_t  nb3 = dst->nb[3];

#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_ಗಾತ್ರ)                                                \
ಸ್ಥಿರ ಶೂನ್ಯ cpy_thread_##NAME##_reshape(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {            \
    ರಚನೆ htp_copy_context * ct = (ರಚನೆ htp_copy_context *) ದತ್ತಾಂಶ;                                  \
    ರಚನೆ htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    ಸ್ಥಿರಾಂಕ uint32_t th_nelem = ct->elem_ಪ್ರತಿ_thread;                                                    \
    ಸ್ಥಿರಾಂಕ uint32_t th_ಆರಂಭ = ct->elem_ಆರಂಭ + ith * th_nelem;                                        \
    ಸ್ಥಿರಾಂಕ uint32_t th_ಕೊನೆ   = MIN(th_ಆರಂಭ + th_nelem, ct->elem_ಆರಂಭ + ct->nelem);                   \
    ಆದರೆ (th_ಆರಂಭ >= th_ಕೊನೆ) ಹಿಂತಿರುಗಿ;                                                                   \
                                                                                                      \
    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    ಸ್ಥಿರಾಂಕ uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    ಸ್ಥಿರಾಂಕ uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    ಸ್ಥಿರಾಂಕ uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    ಸ್ಥಿರಾಂಕ uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
    uint32_t e = th_ಆರಂಭ;                                                                            \
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
    dma_addr_t dst_addr  = dst->ದತ್ತಾಂಶ  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->ದತ್ತಾಂಶ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    ಸ್ಥಿರಾಂಕ bool ಅಡ್ಡಸಾಲುಗಳು_contig = (nb00 == ELEM_ಗಾತ್ರ) && (nb0 == ELEM_ಗಾತ್ರ);                               \
                                                                                                      \
    ಆಗ (e < th_ಕೊನೆ) {                                                                              \
        ಸ್ಥಿರಾಂಕ uint32_t ಸತತ = MIN(MIN(ne00 - i00, ne0 - i10), th_ಕೊನೆ - e);                             \
        ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, ಸತತ * ELEM_ಗಾತ್ರ);             \
        } ಇಲ್ಲದಿದ್ದರೆ {                                                                                      \
            dma_cpy_ತಳ್ಳು_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_ಗಾತ್ರ, ಸತತ);           \
        }                                                                                             \
        e += ಸತತ;                                                                                     \
                                                                                                      \
        dst_addr += ಸತತ * nb0;                                                                        \
        i10      += ಸತತ;                                                                              \
        ಆದರೆ (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            ಆದರೆ (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                ಆದರೆ (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->ದತ್ತಾಂಶ + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += ಸತತ * nb00;                                                                      \
        i00       += ಸತತ;                                                                             \
        ಆದರೆ (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            ಆದರೆ (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                ಆದರೆ (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_ಸರತಿ_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  ದಶಮಾಂಶ, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

ರಚನೆ cpy_ಅಡ್ಡಸಾಲು_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_cursor_advance(ರಚನೆ cpy_ಅಡ್ಡಸಾಲು_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
    c->r   += n;
    c->i01 += n;
    ಆದರೆ (c->i01 == ne01) {
        c->i01 = 0;
        ಆದರೆ (++c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
ಸ್ಥಿರ ಶೂನ್ಯ cpy_thread_##NAME##_sameshape(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) { \
    ರಚನೆ htp_copy_context * ct = (ರಚನೆ htp_copy_context *) ದತ್ತಾಂಶ;                         \
    ರಚನೆ htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    ಸ್ಥಿರಾಂಕ uint32_t dr  = ct->src0_nrows_ಪ್ರತಿ_thread;                                          \
    ಸ್ಥಿರಾಂಕ uint32_t ir0 = ct->ಅಡ್ಡಸಾಲು_ಆರಂಭ + dr * ith;                                           \
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + dr, ct->ಅಡ್ಡಸಾಲು_ಆರಂಭ + ct->nrows);                           \
    ಆದರೆ (ir0 >= ir1) ಹಿಂತಿರುಗಿ;                                                                  \
                                                                                             \
    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];                                                 \
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    ಸ್ಥಿರಾಂಕ uint32_t src0_buf_ಗಾತ್ರ = cvt->src0_buf_ಗಾತ್ರ;                                       \
    ಸ್ಥಿರಾಂಕ uint32_t dst_buf_ಗಾತ್ರ  = cvt->dst_buf_ಗಾತ್ರ;                                        \
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಅಡ್ಡಸಾಲು_stride = cvt->src0_ಅಡ್ಡಸಾಲು_stride;                                   \
    ಸ್ಥಿರಾಂಕ uint32_t dst_ಅಡ್ಡಸಾಲು_stride  = cvt->dst_ಅಡ್ಡಸಾಲು_stride;                                    \
    ಸ್ಥಿರಾಂಕ uint32_t blk_ಅಡ್ಡಸಾಲುಗಳು        = cvt->blk_ಅಡ್ಡಸಾಲುಗಳು;                                          \
    uint8_t * vtcm_src0_ತಳ = ct->vtcm_src0 + ith * cvt->spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;             \
    uint8_t * vtcm_dst_ತಳ  = ct->vtcm_dst  + ith * cvt->spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread;             \
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne00 * ct->kparams->src0_type_ಗಾತ್ರ;                       \
    ಸ್ಥಿರಾಂಕ uint32_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = ne00 * ct->kparams->dst_type_ಗಾತ್ರ;                        \
                                                                                             \
    ಸ್ಥಿರಾಂಕ uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    ರಚನೆ cpy_ಅಡ್ಡಸಾಲು_cursor f = { ir0, i01, i02, i03 };                                        \
    ರಚನೆ cpy_ಅಡ್ಡಸಾಲು_cursor c = f;                                                             \
                                                                                             \
    ಗಾಗಿ (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_ತಳ + b * src0_buf_ಗಾತ್ರ;                             \
        uint8_t * dst_spad = vtcm_dst_ತಳ  + b * dst_buf_ಗಾತ್ರ;                              \
        ಸ್ಥಿರಾಂಕ uint32_t nr = MIN(blk_ಅಡ್ಡಸಾಲುಗಳು, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->ದತ್ತಾಂಶ + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst->ದತ್ತಾಂಶ, dst_spad),                            \
                       dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_buf_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);                         \
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(src_spad, f_src0_addr),                          \
                       src0_ಅಡ್ಡಸಾಲು_stride, nb01, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    ಗಾಗಿ (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        ಸ್ಥಿರಾಂಕ uint32_t nr = MIN(blk_ಅಡ್ಡಸಾಲುಗಳು, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;               \
                                                                                             \
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        ಗಾಗಿ (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_ಅಡ್ಡಸಾಲು_stride, src_spad + r * src0_ಅಡ್ಡಸಾಲು_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->ದತ್ತಾಂಶ + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(c_dst_addr, dst_spad),                           \
                       nb1, dst_ಅಡ್ಡಸಾಲು_stride, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        ಆದರೆ (f.r < ir1) {                                                                     \
            ಸ್ಥಿರಾಂಕ uint32_t fnr = MIN(blk_ಅಡ್ಡಸಾಲುಗಳು, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->ದತ್ತಾಂಶ + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(src_spad, f_src0_addr),                      \
                           src0_ಅಡ್ಡಸಾಲು_stride, nb01, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_ಸರತಿ_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ cpy_scalar(ರಚನೆ htp_ops_context * octx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_kernel_params * kparams) {
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1 && octx->ctx->mdev.idx > 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಆದರೆ (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->ದತ್ತಾಂಶ, src0->ದತ್ತಾಂಶ, kparams->src0_type_ಗಾತ್ರ);
        dma_ಸರತಿ_flush(octx->ctx->dma[0]);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    dma_ಸರತಿ * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_ತಳ;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    ಸ್ಥಿರಾಂಕ uint32_t s_ಗಾತ್ರ = kparams->src0_type_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t d_ಗಾತ್ರ = kparams->dst_type_ಗಾತ್ರ;

    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_vtcm, src0->ದತ್ತಾಂಶ), s_ಗಾತ್ರ, s_ಗಾತ್ರ, s_ಗಾತ್ರ, 1);
    dma_ಸರತಿ_ತೆಗೆ(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_ತಳ;
    uint8_t * d_ptr = s_ptr + VLEN;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_src = hvx_vmem(s_ptr);
    HVX_ಸದಿಶ v_dst;

    ಆದರೆ (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_ಗಮ್ಯ_f16(v_src, v_src);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_ಗಮ್ಯ_f32(v_src));
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst->ದತ್ತಾಂಶ, d_vtcm), d_ಗಾತ್ರ, d_ಗಾತ್ರ, d_ಗಾತ್ರ, 1);
    dma_ಸರತಿ_flush(dma_q);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ cpy_1d_contig(ರಚನೆ htp_ops_context * octx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_kernel_params * kparams) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    uint32_t elem_ಆರಂಭ = 0;
    uint32_t nelem      = kparams->ಒಟ್ಟು_elems;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            kparams->ಒಟ್ಟು_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        elem_ಆರಂಭ = range.ಆರಂಭ;
        nelem      = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nelem > 0) {
        dma_ಸರತಿ * q = octx->ctx->dma[0];
        ಸ್ಥಿರಾಂಕ uint32_t type_ಗಾತ್ರ = kparams->src0_type_ಗಾತ್ರ;
        dma_addr_t dst_addr  = dst->ದತ್ತಾಂಶ  + elem_ಆರಂಭ * type_ಗಾತ್ರ;
        dma_addr_t src0_addr = src0->ದತ್ತಾಂಶ + elem_ಆರಂಭ * type_ಗಾತ್ರ;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_ಗಾತ್ರ);
        dma_ಸರತಿ_flush(q);
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ cpy_sameshape_sametype(ರಚನೆ htp_ops_context * octx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_kernel_params * kparams) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows > 0) {
        dma_ಸರತಿ * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_ಗಾತ್ರ, ಅಡ್ಡಸಾಲು_ಆರಂಭ, nrows);
        dma_ಸರತಿ_flush(q);
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ cpy_sameshape_convert(ರಚನೆ htp_ops_context * octx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_kernel_params * kparams) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಆದರೆ (kparams->vtcm_ಗಾತ್ರ > octx->ctx->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_convert_params * cvt = &kparams->u.convert;

    ರಚನೆ htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.ಅಡ್ಡಸಾಲು_ಆರಂಭ             = ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    ct.nrows                 = nrows;
    ct.src0_nrows_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);

    uint8_t * vtcm_ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;
    ct.vtcm_src0 = vtcm_ತಳ;
    ct.vtcm_dst  = vtcm_ತಳ + (size_t) n_ಎಳೆಗಳು * cvt->spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;

    work_ಸರತಿ_func_t copy_fun = ಖಾಲಿ;
    ಆದರೆ (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, copy_fun, &ct, n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ cpy_reshape(ರಚನೆ htp_ops_context * octx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_kernel_params * kparams) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    uint32_t elem_ಆರಂಭ = 0;
    uint32_t nelem      = kparams->ಒಟ್ಟು_elems;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            kparams->ಒಟ್ಟು_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        elem_ಆರಂಭ = range.ಆರಂಭ;
        nelem      = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nelem == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    ರಚನೆ htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_ಆರಂಭ      = elem_ಆರಂಭ;
    ct.nelem           = nelem;
    ct.elem_ಪ್ರತಿ_thread = fastdiv(nelem + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);

    work_ಸರತಿ_func_t copy_fun = ಖಾಲಿ;
    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, copy_fun, &ct, n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_cpy(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_copy_kernel_params *) octx->kernel_params;
    ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = HTP_ಸ್ಥಿತಿ_OK;

    ಆಯ್ಕೆ (kparams->kernel_type) {
        ಸ್ಥಿತಿ HTP_COPY_KERNEL_SCALAR:
            ಸ್ಥಿತಿ_2 = cpy_scalar(octx, kparams);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_COPY_KERNEL_1D_CONTIG:
            ಸ್ಥಿತಿ_2 = cpy_1d_contig(octx, kparams);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            ಸ್ಥಿತಿ_2 = cpy_sameshape_sametype(octx, kparams);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            ಸ್ಥಿತಿ_2 = cpy_sameshape_convert(octx, kparams);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_COPY_KERNEL_RESHAPE:
            ಸ್ಥಿತಿ_2 = cpy_reshape(octx, kparams);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_COPY_KERNEL_GATHER_ಅಡ್ಡಸಾಲುಗಳು:
            ಸ್ಥಿತಿ_2 = htp_gather_ಅಡ್ಡಸಾಲುಗಳು(octx, octx->src[0], ಖಾಲಿ, &kparams->u.gather, kparams->n_ಎಳೆಗಳು);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಸ್ಥಿತಿ_2 = HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
            ನಿಲ್ಲಿಸು;
    }

    htp_ops_context_ಗಣ_ಸ್ಥಿತಿ(octx, ಸ್ಥಿತಿ_2);

    ಆದರೆ (octx->op == HTP_OP_CPY_FENCE) {
        ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
            htp_mdev_ಗುಂಪು_barrier(octx);
        }

        ಆದರೆ (octx->ctx->mdev.idx == 0) {
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * sync = octx->src[1];
            ಸ್ಥಿರಾಂಕ uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->ದತ್ತಾಂಶ;
            htp_fence_write(sync_fence, seq, octx->ಸ್ಥಿತಿ_2);

            FARF(ಮೇಲಿನ_2, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->ಸ್ಥಿತಿ_2);
        }
    }

    ಹಿಂತಿರುಗಿ octx->ಸ್ಥಿತಿ_2;
}
