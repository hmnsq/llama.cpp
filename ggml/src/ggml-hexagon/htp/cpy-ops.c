#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>
#అన్తర్భావ <qurt_memory.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "cpy-ops.h"
#అన్తర్భావ "dma-copy.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-fence.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "gather-rows.h"

నిర్మాణం htp_copy_context {
    నిర్మాణం htp_ops_context *              octx;
    స్థిరరాశి నిర్మాణం htp_copy_kernel_params * kparams;

    uint32_t అడ్డువరుస_ప్రారంభం;
    uint32_t nrows;
    uint32_t src0_nrows_చొప్పున_thread;

    uint32_t elem_ప్రారంభం;
    uint32_t nelem;
    uint32_t elem_చొప్పున_thread;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#నిర్వచించు cpy_preamble                              \
    స్థిరరాశి నిర్మాణం htp_tensor *src0 = octx->src[0]; \
    స్థిరరాశి నిర్మాణం htp_tensor *dst  = octx->dst;    \
                                                  \
    స్థిరరాశి uint32_t ne00 = src0->ne[0];            \
    స్థిరరాశి uint32_t ne01 = src0->ne[1];            \
    స్థిరరాశి uint32_t ne02 = src0->ne[2];            \
    స్థిరరాశి uint32_t ne03 = src0->ne[3];            \
                                                  \
    స్థిరరాశి uint32_t nb00 = src0->nb[0];            \
    స్థిరరాశి uint32_t nb01 = src0->nb[1];            \
    స్థిరరాశి uint32_t nb02 = src0->nb[2];            \
    స్థిరరాశి uint32_t nb03 = src0->nb[3];            \
                                                  \
    స్థిరరాశి uint32_t  ne0 = dst->ne[0];             \
    స్థిరరాశి uint32_t  ne1 = dst->ne[1];             \
    స్థిరరాశి uint32_t  ne2 = dst->ne[2];             \
    స్థిరరాశి uint32_t  ne3 = dst->ne[3];             \
                                                  \
    స్థిరరాశి uint32_t  nb0 = dst->nb[0];             \
    స్థిరరాశి uint32_t  nb1 = dst->nb[1];             \
    స్థిరరాశి uint32_t  nb2 = dst->nb[2];             \
    స్థిరరాశి uint32_t  nb3 = dst->nb[3];

#నిర్వచించు DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_కొలత)                                                \
స్థిర శూన్య cpy_thread_##NAME##_reshape(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {            \
    నిర్మాణం htp_copy_context * ct = (నిర్మాణం htp_copy_context *) దత్తాంశం;                                  \
    నిర్మాణం htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    స్థిరరాశి uint32_t th_nelem = ct->elem_చొప్పున_thread;                                                    \
    స్థిరరాశి uint32_t th_ప్రారంభం = ct->elem_ప్రారంభం + ith * th_nelem;                                        \
    స్థిరరాశి uint32_t th_చివర   = MIN(th_ప్రారంభం + th_nelem, ct->elem_ప్రారంభం + ct->nelem);                   \
    ఒకవేళ (th_ప్రారంభం >= th_చివర) తిరిగి;                                                                   \
                                                                                                      \
    dma_వరుస * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    స్థిరరాశి uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    స్థిరరాశి uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    స్థిరరాశి uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    స్థిరరాశి uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    స్థిరరాశి నిర్మాణం htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
    uint32_t e = th_ప్రారంభం;                                                                            \
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
    dma_addr_t dst_addr  = dst->దత్తాంశం  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->దత్తాంశం + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    స్థిరరాశి bool అడ్డువరుసలు_contig = (nb00 == ELEM_కొలత) && (nb0 == ELEM_కొలత);                               \
                                                                                                      \
    అయితే (e < th_చివర) {                                                                              \
        స్థిరరాశి uint32_t నిరంతరం = MIN(MIN(ne00 - i00, ne0 - i10), th_చివర - e);                             \
        ఒకవేళ (అడ్డువరుసలు_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, నిరంతరం * ELEM_కొలత);             \
        } లేకపోతే {                                                                                      \
            dma_cpy_నెట్టు_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_కొలత, నిరంతరం);           \
        }                                                                                             \
        e += నిరంతరం;                                                                                     \
                                                                                                      \
        dst_addr += నిరంతరం * nb0;                                                                        \
        i10      += నిరంతరం;                                                                              \
        ఒకవేళ (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            ఒకవేళ (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                ఒకవేళ (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->దత్తాంశం + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += నిరంతరం * nb00;                                                                      \
        i00       += నిరంతరం;                                                                             \
        ఒకవేళ (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            ఒకవేళ (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                ఒకవేళ (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_వరుస_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  దశాంశం, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

నిర్మాణం cpy_అడ్డువరుస_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

స్థిర అంతఃవరుస శూన్య cpy_cursor_advance(నిర్మాణం cpy_అడ్డువరుస_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
    c->r   += n;
    c->i01 += n;
    ఒకవేళ (c->i01 == ne01) {
        c->i01 = 0;
        ఒకవేళ (++c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

#నిర్వచించు DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
స్థిర శూన్య cpy_thread_##NAME##_sameshape(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) { \
    నిర్మాణం htp_copy_context * ct = (నిర్మాణం htp_copy_context *) దత్తాంశం;                         \
    నిర్మాణం htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    స్థిరరాశి uint32_t dr  = ct->src0_nrows_చొప్పున_thread;                                          \
    స్థిరరాశి uint32_t ir0 = ct->అడ్డువరుస_ప్రారంభం + dr * ith;                                           \
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + dr, ct->అడ్డువరుస_ప్రారంభం + ct->nrows);                           \
    ఒకవేళ (ir0 >= ir1) తిరిగి;                                                                  \
                                                                                             \
    dma_వరుస * dma_q = octx->ctx->dma[ith];                                                 \
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    స్థిరరాశి నిర్మాణం htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    స్థిరరాశి uint32_t src0_buf_కొలత = cvt->src0_buf_కొలత;                                       \
    స్థిరరాశి uint32_t dst_buf_కొలత  = cvt->dst_buf_కొలత;                                        \
    స్థిరరాశి uint32_t src0_అడ్డువరుస_stride = cvt->src0_అడ్డువరుస_stride;                                   \
    స్థిరరాశి uint32_t dst_అడ్డువరుస_stride  = cvt->dst_అడ్డువరుస_stride;                                    \
    స్థిరరాశి uint32_t blk_అడ్డువరుసలు        = cvt->blk_అడ్డువరుసలు;                                          \
    uint8_t * vtcm_src0_పునాది = ct->vtcm_src0 + ith * cvt->spad0_కొలత_చొప్పున_thread;             \
    uint8_t * vtcm_dst_పునాది  = ct->vtcm_dst  + ith * cvt->spad1_కొలత_చొప్పున_thread;             \
    స్థిరరాశి uint32_t src0_అడ్డువరుస_కొలత = ne00 * ct->kparams->src0_type_కొలత;                       \
    స్థిరరాశి uint32_t dst_అడ్డువరుస_కొలత  = ne00 * ct->kparams->dst_type_కొలత;                        \
                                                                                             \
    స్థిరరాశి uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    నిర్మాణం cpy_అడ్డువరుస_cursor f = { ir0, i01, i02, i03 };                                        \
    నిర్మాణం cpy_అడ్డువరుస_cursor c = f;                                                             \
                                                                                             \
    కొరకు (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_పునాది + b * src0_buf_కొలత;                             \
        uint8_t * dst_spad = vtcm_dst_పునాది  + b * dst_buf_కొలత;                              \
        స్థిరరాశి uint32_t nr = MIN(blk_అడ్డువరుసలు, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->దత్తాంశం + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst->దత్తాంశం, dst_spad),                            \
                       dst_అడ్డువరుస_కొలత, dst_buf_కొలత, dst_అడ్డువరుస_కొలత, 0);                         \
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(src_spad, f_src0_addr),                          \
                       src0_అడ్డువరుస_stride, nb01, src0_అడ్డువరుస_కొలత, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    కొరకు (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        స్థిరరాశి uint32_t nr = MIN(blk_అడ్డువరుసలు, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_వరుస_తీయి(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_వరుస_తీయి(dma_q).dst;               \
                                                                                             \
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        కొరకు (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_అడ్డువరుస_stride, src_spad + r * src0_అడ్డువరుస_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->దత్తాంశం + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(c_dst_addr, dst_spad),                           \
                       nb1, dst_అడ్డువరుస_stride, dst_అడ్డువరుస_కొలత, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        ఒకవేళ (f.r < ir1) {                                                                     \
            స్థిరరాశి uint32_t fnr = MIN(blk_అడ్డువరుసలు, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->దత్తాంశం + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(src_spad, f_src0_addr),                      \
                           src0_అడ్డువరుస_stride, nb01, src0_అడ్డువరుస_కొలత, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_వరుస_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

స్థిర పూర్ణాంకం cpy_scalar(నిర్మాణం htp_ops_context * octx, స్థిరరాశి నిర్మాణం htp_copy_kernel_params * kparams) {
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1 && octx->ctx->mdev.idx > 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    ఒకవేళ (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->దత్తాంశం, src0->దత్తాంశం, kparams->src0_type_కొలత);
        dma_వరుస_flush(octx->ctx->dma[0]);
        తిరిగి HTP_స్థితి_OK;
    }

    dma_వరుస * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_పునాది;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    స్థిరరాశి uint32_t s_కొలత = kparams->src0_type_కొలత;
    స్థిరరాశి uint32_t d_కొలత = kparams->dst_type_కొలత;

    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_vtcm, src0->దత్తాంశం), s_కొలత, s_కొలత, s_కొలత, 1);
    dma_వరుస_తీయి(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_పునాది;
    uint8_t * d_ptr = s_ptr + VLEN;

    స్థిరరాశి HVX_సదిశ v_src = hvx_vmem(s_ptr);
    HVX_సదిశ v_dst;

    ఒకవేళ (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } లేకపోతే ఒకవేళ (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } లేకపోతే ఒకవేళ (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_గమ్యం_f16(v_src, v_src);
    } లేకపోతే ఒకవేళ (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(v_src));
    } లేకపోతే {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst->దత్తాంశం, d_vtcm), d_కొలత, d_కొలత, d_కొలత, 1);
    dma_వరుస_flush(dma_q);
    తిరిగి HTP_స్థితి_OK;
}

స్థిర పూర్ణాంకం cpy_1d_contig(నిర్మాణం htp_ops_context * octx, స్థిరరాశి నిర్మాణం htp_copy_kernel_params * kparams) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    uint32_t elem_ప్రారంభం = 0;
    uint32_t nelem      = kparams->మొత్తం_elems;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            kparams->మొత్తం_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        elem_ప్రారంభం = range.ప్రారంభం;
        nelem      = range.లెక్క;
    }

    ఒకవేళ (nelem > 0) {
        dma_వరుస * q = octx->ctx->dma[0];
        స్థిరరాశి uint32_t type_కొలత = kparams->src0_type_కొలత;
        dma_addr_t dst_addr  = dst->దత్తాంశం  + elem_ప్రారంభం * type_కొలత;
        dma_addr_t src0_addr = src0->దత్తాంశం + elem_ప్రారంభం * type_కొలత;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_కొలత);
        dma_వరుస_flush(q);
    }

    తిరిగి HTP_స్థితి_OK;
}

స్థిర పూర్ణాంకం cpy_sameshape_sametype(నిర్మాణం htp_ops_context * octx, స్థిరరాశి నిర్మాణం htp_copy_kernel_params * kparams) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = kparams->మొత్తం_అడ్డువరుసలు;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            kparams->మొత్తం_అడ్డువరుసలు, 1, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows > 0) {
        dma_వరుస * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_కొలత, అడ్డువరుస_ప్రారంభం, nrows);
        dma_వరుస_flush(q);
    }

    తిరిగి HTP_స్థితి_OK;
}

స్థిర పూర్ణాంకం cpy_sameshape_convert(నిర్మాణం htp_ops_context * octx, స్థిరరాశి నిర్మాణం htp_copy_kernel_params * kparams) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = kparams->మొత్తం_అడ్డువరుసలు;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            kparams->మొత్తం_అడ్డువరుసలు, 1, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    ఒకవేళ (kparams->vtcm_కొలత > octx->ctx->vtcm_కొలత) {
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;
    స్థిరరాశి నిర్మాణం htp_copy_convert_params * cvt = &kparams->u.convert;

    నిర్మాణం htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.అడ్డువరుస_ప్రారంభం             = అడ్డువరుస_ప్రారంభం;
    ct.nrows                 = nrows;
    ct.src0_nrows_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div);

    uint8_t * vtcm_పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;
    ct.vtcm_src0 = vtcm_పునాది;
    ct.vtcm_dst  = vtcm_పునాది + (size_t) n_దారాలు * cvt->spad0_కొలత_చొప్పున_thread;

    work_వరుస_func_t copy_fun = ఖాళీ;
    ఒకవేళ (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } లేకపోతే ఒకవేళ (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } లేకపోతే ఒకవేళ (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } లేకపోతే ఒకవేళ (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } లేకపోతే {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, copy_fun, &ct, n_దారాలు);
    తిరిగి HTP_స్థితి_OK;
}

స్థిర పూర్ణాంకం cpy_reshape(నిర్మాణం htp_ops_context * octx, స్థిరరాశి నిర్మాణం htp_copy_kernel_params * kparams) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    uint32_t elem_ప్రారంభం = 0;
    uint32_t nelem      = kparams->మొత్తం_elems;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            kparams->మొత్తం_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        elem_ప్రారంభం = range.ప్రారంభం;
        nelem      = range.లెక్క;
    }

    ఒకవేళ (nelem == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    నిర్మాణం htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_ప్రారంభం      = elem_ప్రారంభం;
    ct.nelem           = nelem;
    ct.elem_చొప్పున_thread = fastdiv(nelem + n_దారాలు - 1, &octx->n_దారాలు_div);

    work_వరుస_func_t copy_fun = ఖాళీ;
    ఎంపిక (src0->type) {
        స్థితి HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; ఆపు;
        స్థితి HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; ఆపు;
        స్థితి HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; ఆపు;
        అప్రమేయ: తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, copy_fun, &ct, n_దారాలు);
    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_cpy(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_copy_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_copy_kernel_params *) octx->kernel_params;
    పూర్ణాంకం స్థితి_2 = HTP_స్థితి_OK;

    ఎంపిక (kparams->kernel_type) {
        స్థితి HTP_COPY_KERNEL_SCALAR:
            స్థితి_2 = cpy_scalar(octx, kparams);
            ఆపు;
        స్థితి HTP_COPY_KERNEL_1D_CONTIG:
            స్థితి_2 = cpy_1d_contig(octx, kparams);
            ఆపు;
        స్థితి HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            స్థితి_2 = cpy_sameshape_sametype(octx, kparams);
            ఆపు;
        స్థితి HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            స్థితి_2 = cpy_sameshape_convert(octx, kparams);
            ఆపు;
        స్థితి HTP_COPY_KERNEL_RESHAPE:
            స్థితి_2 = cpy_reshape(octx, kparams);
            ఆపు;
        స్థితి HTP_COPY_KERNEL_GATHER_అడ్డువరుసలు:
            స్థితి_2 = htp_gather_అడ్డువరుసలు(octx, octx->src[0], ఖాళీ, &kparams->u.gather, kparams->n_దారాలు);
            ఆపు;
        అప్రమేయ:
            స్థితి_2 = HTP_స్థితి_NO_SUPPORT;
            ఆపు;
    }

    htp_ops_context_సమితి_స్థితి(octx, స్థితి_2);

    ఒకవేళ (octx->op == HTP_OP_CPY_FENCE) {
        ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
            htp_mdev_సమూహం_barrier(octx);
        }

        ఒకవేళ (octx->ctx->mdev.idx == 0) {
            స్థిరరాశి నిర్మాణం htp_tensor * sync = octx->src[1];
            స్థిరరాశి uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->దత్తాంశం;
            htp_fence_write(sync_fence, seq, octx->స్థితి_2);

            FARF(ఎగువ_2, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->స్థితి_2);
        }
    }

    తిరిగి octx->స్థితి_2;
}
