#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <hexagon_protos.h>
#ضمن <hexagon_types.h>
#ضمن <string.h>

#ضمن "hex-common.h"

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "hvx-utils.h"
#ضمن "dma-queue.h"
#ضمن "hex-profile.h"
#ضمن "htp-vtcm.h"
#ضمن "htp-tensor.h"
#ضمن "im2col-ops.h"

بنية htp_im2col_context {
    بنية htp_ops_context * octx;

    uint32_t pe_ردیف_پایه;                          // first output row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of output rows assigned to this dev (DMA path)
    uint32_t pe_ردیفها_در_هر_thread;                   // output rows per worker
    uint32_t pe_src_ردیف_بایتها;                     // one output row's source, rounded 256
    uint32_t pe_dst_ردیف_بایتها;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_اندازه_در_هر_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_اندازه_در_هر_thread;              // 2 * pe_dst_row_bytes

    uint32_t pe_owb;                               // output-col block size
    uint32_t pe_wb;                                // staged source window width
    uint32_t pe_nbpr;                              // blocks per row

    // Precomputed dimensions for worker threads
    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;

    // Per-thread initial coordinates (removes division from thread prologue)
    uint32_t thread_درون[HTP_MAX_NTHREADS];
    uint32_t thread_iod[HTP_MAX_NTHREADS];
    uint32_t thread_ioh[HTP_MAX_NTHREADS];

    فراغ * compute;
    uint32_t dst_elem;
};

تعريفنوع فراغ (*im2col_blocked_compute_fn_t)(
    ثابت بنية htp_im2col_context * ictx,
    فراغ * محدود dstb, ثابت عائم * محدود srcb,
    عائم * محدود scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

تعريفنوع فراغ (*im2col_patchembed_compute_fn_t)(
    ثابت بنية htp_im2col_context * ictx,
    فراغ * محدود dstb, ثابت عائم * محدود srcb, uint32_t ioh);

تعريفنوع فراغ (*im2col_3d_blocked_compute_fn_t)(
    ثابت بنية htp_im2col_context * ictx,
    فراغ * محدود dstb, ثابت عائم * محدود srcb,
    عائم * محدود scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

تعريفنوع فراغ (*im2col_3d_patchembed_compute_fn_t)(
    ثابت بنية htp_im2col_context * ictx,
    فراغ * محدود dstb, ثابت عائم * محدود srcb);

بنية im2col_cursor {
    uint32_t r;
    uint32_t درون;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

ساكن مضمن فراغ im2col_گام_ردیف(بنية im2col_cursor * c, ثابت بنية htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    إذا (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->درون++;
    }
}

ساكن مضمن فراغ im2col_گام_بلوک(بنية im2col_cursor * c, ثابت بنية htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    إذا (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        إذا (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->درون++;
        }
    }
}

ساكن فراغ im2col_blocked_dma_stage(بنية htp_im2col_context * ictx, dma_صف * dma_q,
                                     عائم * srcb, ثابت بنية im2col_cursor * c) {
    ثابت uint32_t درون = c->درون, ioh = c->ioh, c0 = c->c0;
    ثابت بنية htp_ops_context * octx = ictx->octx;
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    ثابت int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    ثابت int32_t d1 = octx->op_params[5];
    ثابت bool is_2D = (octx->op_params[6] == 1);
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KH = ictx->KH;
    ثابت uint32_t Wb = ictx->pe_wb;
    ثابت dma_addr_t src_داده = src1->داده;
    ثابت int32_t win0 = (int32_t) c0 * s0 - p0;
    ثابت size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    ثابت size_t batch_جابه_جایی        = is_2D ? (size_t) درون * src1->nb[3] : (size_t) درون * src1->nb[2];

    لكل (uint32_t ikh = 0; ikh < KH; ikh++) {
        ثابت int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        إذا (iih < 0 || iih >= (int32_t) IH) {
            dma_صف_افزودن(dma_q, dma_make_داده(srcb, 0), 0, 0, 0, 0);
            استمر;
        }
        ثابت int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        إذا (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        إذا (hi <= lo) {
            dma_صف_افزودن(dma_q, dma_make_داده(srcb, 0), 0, 0, 0, 0);
            استمر;
        }
        ثابت uint32_t cpw = (uint32_t) (hi - lo);
        عائم * vdst = srcb + (size_t) ikh * Wb + lo;
        ثابت dma_addr_t vsrc = src_داده + batch_جابه_جایی + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_صف_افزودن(dma_q, dma_make_داده(vdst, vsrc),
                       (size_t) KH * Wb * حجم(عائم), src_channel_stride,
                       cpw * حجم(عائم), IC);
    }
}

#تعریف_کن IM2_ستون_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
ساكن فراغ FNAME(                                                                               \
    ثابت بنية htp_im2col_context * ictx,                                                      \
    فراغ * محدود dstb_raw, ثابت عائم * محدود srcb,                                       \
    عائم * محدود scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * محدود dstb = (DST_CTYPE *) dstb_raw;                                          \
    ثابت بنية htp_ops_context * octx = ictx->octx;                                            \
    ثابت int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    ثابت int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    ثابت int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    ثابت uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    ثابت uint32_t Wb = ictx->pe_wb;                                                             \
    ثابت uint32_t patch_stride = ictx->patch_stride;                                            \
    ثابت int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    لكل (uint32_t j = 0; j < nb; j++) {                                                          \
        ثابت uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (size_t) j * patch_stride;                             \
        ثابت int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        لكل (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            ثابت int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            ثابت صحيح     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            لكل (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * بیرون_پیاپی = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                إذا (!okh) {                                                                      \
                    SPLAT_FN(بیرون_پیاپی, 0, KW);                                                    \
                    استمر;                                                                    \
                }                                                                                \
                ثابت عائم * vrow = srcb + (size_t) (iic * KH + ikh) * Wb;                      \
                إذا (d0 == 1) {                                                                   \
                    ثابت int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    إذا (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    إذا (hi <= lo) {                                                              \
                        SPLAT_FN(بیرون_پیاپی, 0, KW);                                                \
                    } وإلا {                                                                     \
                        إذا (lo > 0) {                                                            \
                            SPLAT_FN(بیرون_پیاپی, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (بیرون_پیاپی + lo),                                      \
                                (ثابت uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        إذا (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(بیرون_پیاپی + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    استمر;                                                                    \
                }                                                                                \
                لكل (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    ثابت uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    لكل (uint32_t c = 0; c < chunk; c++) {                                       \
                        ثابت int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (بیرون_پیاپی + ikw), (ثابت uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2_ستون_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2_ستون_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, عائم,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ساكن فراغ im2col_blocked_dma_thread(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    بنية htp_im2col_context * ictx      = (بنية htp_im2col_context *) داده;
    بنية htp_ops_context *    octx      = ictx->octx;
    بنية htp_thread_trace * محدود tr = &octx->ctx->trace[ith];
    ثابت uint32_t OW                     = ictx->OW;
    ثابت uint32_t KH                     = ictx->KH;
    ثابت uint32_t owb                    = ictx->pe_owb;
    ثابت uint32_t nbpr                   = ictx->pe_nbpr;
    ثابت uint32_t patch_stride           = ictx->patch_stride;
    ثابت uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    ثابت dma_addr_t dst_داده             = octx->dst->داده;
    dma_صف *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_پایه              = ictx->pe_vtcm_src + ith * ictx->pe_src_اندازه_در_هر_thread;
    uint8_t *      dstb_پایه              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_اندازه_در_هر_thread;
    عائم *        srcb2[2]               = { (عائم *) srcb_پایه, (عائم *) (srcb_پایه + ictx->pe_src_ردیف_بایتها) };
    فراغ *         dstb2[2]               = { dstb_پایه, dstb_پایه + ictx->pe_dst_ردیف_بایتها };
    عائم *        scratch                = (عائم *) (srcb_پایه + 2 * ictx->pe_src_ردیف_بایتها);
    ثابت uint32_t ردیف_انتها_max            = ictx->pe_ردیف_پایه + ictx->pe_nrows;
    ثابت uint32_t ردیف_شروع              = ictx->pe_ردیف_پایه + ictx->pe_ردیفها_در_هر_thread * ith;
    ثابت uint32_t ردیف_انتها                = MIN(ردیف_شروع + ictx->pe_ردیفها_در_هر_thread, ردیف_انتها_max);
    إذا (ردیف_شروع >= ردیف_انتها) {
        أرجع;
    }
    ثابت uint32_t nrows_local  = ردیف_انتها - ردیف_شروع;
    ثابت uint32_t مجموع_بلوکها = nrows_local * nbpr;
    إذا (مجموع_بلوکها == 0) {
        أرجع;
    }
    بنية im2col_cursor cur = {
        .r   = ردیف_شروع,
        .درون  = ictx->thread_درون[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    بنية im2col_cursor stage = cur;
    لكل (uint32_t i = 0; i < مجموع_بلوکها && i < 2; i++) {
        dma_صف_افزودن(dma_q, dma_make_داده(dst_داده, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_گام_بلوک(&stage, ictx);
    }
    لكل (uint32_t bi = 0; bi < مجموع_بلوکها; bi++) {
        ثابت uint32_t buf = bi & 1u;
        عائم * srcb = srcb2[buf];
        فراغ *  dstb = dstb2[buf];
        ثابت uint32_t nb = MIN(owb, OW - cur.c0);
        dma_صف_برداشت(dma_q);
        لكل (uint32_t k = 0; k < KH; k++) {
            dma_صف_برداشت(dma_q);
        }
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr = dst_داده + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_صف_افزودن(dma_q, dma_make_داده(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        إذا (bi + 2 < مجموع_بلوکها) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_گام_بلوک(&stage, ictx);
        }
        im2col_گام_بلوک(&cur, ictx);
    }
    dma_صف_flush(dma_q);
}

ساكن فراغ im2col_patchembed_dma_stage(بنية htp_im2col_context * ictx, dma_صف * dma_q,
                                        عائم * srcb, ثابت بنية im2col_cursor * c) {
    ثابت بنية htp_ops_context * octx = ictx->octx;
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KH = ictx->KH;
    ثابت uint32_t درون = c->درون, ioh = c->ioh;
    ثابت size_t src_channel_stride  = (size_t) IH * IW * حجم(عائم);
    ثابت size_t vtcm_channel_stride = (size_t) KH * IW * حجم(عائم);
    ثابت dma_addr_t src_داده        = octx->src[1]->داده;
    ثابت dma_addr_t vsrc            = src_داده + (size_t) (((درون * IC) * IH + ioh * KH) * IW) * حجم(عائم);

    dma_صف_افزودن(dma_q, dma_make_داده(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

ساكن فراغ im2col_patchembed_compute_f16(
    ثابت بنية htp_im2col_context * ictx,
    فراغ * محدود dstb_raw, ثابت عائم * محدود srcb, uint32_t ioh) {
    uint16_t * محدود dstb = (uint16_t *) dstb_raw;
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KH = ictx->KH, KW = ictx->KW;
    ثابت uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    لكل (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        لكل (uint32_t ikh = 0; ikh < KH; ikh++) {
            ثابت int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            لكل (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * بیرون_پیاپی = dst_patch + (size_t) (iic * KH + ikh) * KW;
                إذا (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(بیرون_پیاپی, 0, KW);
                    استمر;
                }
                ثابت عائم * src_پیاپی = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) بیرون_پیاپی, (ثابت uint8_t *) src_پیاپی, KW);
            }
        }
    }
}

ساكن فراغ im2col_patchembed_compute_f32(
    ثابت بنية htp_im2col_context * ictx,
    فراغ * محدود dstb_raw, ثابت عائم * محدود srcb, uint32_t ioh) {
    عائم * محدود dstb = (عائم *) dstb_raw;
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KH = ictx->KH, KW = ictx->KW;
    ثابت uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    لكل (uint32_t iow = 0; iow < OW; iow++) {
        عائم * dst_patch = dstb + (size_t) iow * patch_stride;
        لكل (uint32_t ikh = 0; ikh < KH; ikh++) {
            ثابت int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            لكل (uint32_t iic = 0; iic < IC; iic++) {
                عائم * بیرون_پیاپی = dst_patch + (size_t) (iic * KH + ikh) * KW;
                إذا (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(بیرون_پیاپی, 0, KW);
                    استمر;
                }
                ثابت عائم * src_پیاپی = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) بیرون_پیاپی, (ثابت uint8_t *) src_پیاپی, KW);
            }
        }
    }
}

ساكن فراغ im2col_patchembed_dma_thread(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    بنية htp_im2col_context * ictx        = (بنية htp_im2col_context *) داده;
    بنية htp_ops_context *    octx        = ictx->octx;
    بنية htp_thread_trace * محدود tr   = &octx->ctx->trace[ith];
    ثابت uint32_t OW                       = ictx->OW;
    ثابت uint32_t patch_stride             = ictx->patch_stride;
    ثابت uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    ثابت dma_addr_t dst_داده               = octx->dst->داده;
    dma_صف *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_پایه                = ictx->pe_vtcm_src + ith * ictx->pe_src_اندازه_در_هر_thread;
    uint8_t *      dstb_پایه                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_اندازه_در_هر_thread;
    عائم *        srcb2[2]                 = { (عائم *) srcb_پایه, (عائم *) (srcb_پایه + ictx->pe_src_ردیف_بایتها) };
    فراغ *         dstb2[2]                 = { dstb_پایه, dstb_پایه + ictx->pe_dst_ردیف_بایتها };
    ثابت uint32_t ردیف_انتها_max              = ictx->pe_ردیف_پایه + ictx->pe_nrows;
    ثابت uint32_t ردیف_شروع                = ictx->pe_ردیف_پایه + ictx->pe_ردیفها_در_هر_thread * ith;
    ثابت uint32_t ردیف_انتها                  = MIN(ردیف_شروع + ictx->pe_ردیفها_در_هر_thread, ردیف_انتها_max);
    إذا (ردیف_شروع >= ردیف_انتها) {
        أرجع;
    }
    ثابت uint32_t مجموع_ردیفها = ردیف_انتها - ردیف_شروع;
    إذا (مجموع_ردیفها == 0) {
        أرجع;
    }
    بنية im2col_cursor cur = {
        .r   = ردیف_شروع,
        .درون  = ictx->thread_درون[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    بنية im2col_cursor stage = cur;
    لكل (uint32_t i = 0; i < مجموع_ردیفها && i < 2; i++) {
        dma_صف_افزودن(dma_q, dma_make_داده(dst_داده, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_گام_ردیف(&stage, ictx);
    }
    لكل (uint32_t ri = 0; ri < مجموع_ردیفها; ri++) {
        ثابت uint32_t buf = ri & 1u;
        عائم * srcb = srcb2[buf];
        فراغ *  dstb = dstb2[buf];
        dma_صف_برداشت(dma_q);
        dma_صف_برداشت(dma_q);
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr_ردیف = dst_داده + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_صف_افزودن(dma_q, dma_make_داده(ddr_ردیف, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        إذا (ri + 2 < مجموع_ردیفها) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_گام_ردیف(&stage, ictx);
        }
        im2col_گام_ردیف(&cur, ictx);
    }
    dma_صف_flush(dma_q);
}

ساكن مضمن فراغ im2col_3d_گام_ردیف(بنية im2col_cursor * c, ثابت بنية htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    إذا (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        إذا (c->iod >= ictx->OD) {
            c->iod = 0;
            c->درون++;
        }
    }
}

ساكن مضمن فراغ im2col_3d_گام_بلوک(بنية im2col_cursor * c, ثابت بنية htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    إذا (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        إذا (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            إذا (c->iod >= ictx->OD) {
                c->iod = 0;
                c->درون++;
            }
        }
    }
}

ساكن فراغ im2col_3d_blocked_dma_stage(بنية htp_im2col_context * ictx, dma_صف * dma_q,
                                        عائم * srcb, ثابت بنية im2col_cursor * c) {
    ثابت uint32_t درون = c->درون, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    ثابت بنية htp_ops_context * octx = ictx->octx;
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    ثابت int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    ثابت int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    ثابت uint32_t IC = ictx->IC;
    ثابت uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KD = ictx->KD, KH = ictx->KH;
    ثابت uint32_t Wb = ictx->pe_wb;
    ثابت int32_t win0 = (int32_t) c0 * s0 - p0;
    ثابت dma_addr_t src_داده = src1->داده;
    ثابت size_t src_channel_stride = src1->nb[3];
    ثابت size_t vtcm_channel_stride = (size_t) KD * KH * Wb * حجم(عائم);
    ثابت size_t batch_جابه_جایی        = (size_t) (درون * IC) * src1->nb[3];

    لكل (uint32_t ikd = 0; ikd < KD; ikd++) {
        ثابت int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        ثابت bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        لكل (uint32_t ikh = 0; ikh < KH; ikh++) {
            ثابت int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            إذا (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_صف_افزودن(dma_q, dma_make_داده(srcb, 0), 0, 0, 0, 0);
                استمر;
            }
            ثابت int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            إذا (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            إذا (hi <= lo) {
                dma_صف_افزودن(dma_q, dma_make_داده(srcb, 0), 0, 0, 0, 0);
                استمر;
            }
            عائم * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            ثابت dma_addr_t vsrc = src_داده + batch_جابه_جایی +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_صف_افزودن(dma_q, dma_make_داده(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * حجم(عائم), IC);
        }
    }
}

#تعریف_کن IM2_ستون_3_D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
ساكن فراغ FNAME(ثابت بنية htp_im2col_context * ictx,                                            \
                  فراغ * محدود dstb_raw, ثابت عائم * محدود srcb,                             \
                  عائم * محدود scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * محدود dstb = (DST_CTYPE *) dstb_raw;                                              \
    ثابت بنية htp_ops_context * octx = ictx->octx;                                                \
    ثابت int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    ثابت int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    ثابت int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    ثابت uint32_t IC = ictx->IC;                                                                    \
    ثابت uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    ثابت uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    ثابت uint32_t patch_stride = ictx->patch_stride;                                                \
    ثابت uint32_t Wb = ictx->pe_wb;                                                                 \
    ثابت int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    لكل (uint32_t j = 0; j < nb; j++) {                                                              \
        ثابت uint32_t iow = c0 + j;                                                                 \
        ثابت int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (size_t) j * patch_stride;                                    \
        لكل (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            لكل (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                ثابت int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                لكل (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    ثابت int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * بیرون_پیاپی = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    إذا (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(بیرون_پیاپی, 0, KW);                                                    \
                        استمر;                                                                    \
                    }                                                                                \
                    ثابت عائم * vrow = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    إذا (d0 == 1) {                                                                   \
                        ثابت int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        إذا (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        إذا (hi <= lo) {                                                              \
                            SPLAT_FN(بیرون_پیاپی, 0, KW);                                                \
                        } وإلا {                                                                     \
                            إذا (lo > 0) {                                                            \
                                SPLAT_FN(بیرون_پیاپی, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (بیرون_پیاپی + lo),                                      \
                                    (ثابت uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            إذا (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(بیرون_پیاپی + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        استمر;                                                                    \
                    }                                                                                \
                    لكل (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        ثابت uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        لكل (uint32_t c = 0; c < chunk; c++) {                                       \
                            ثابت int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (بیرون_پیاپی + ikw), (ثابت uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2_ستون_3_D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2_ستون_3_D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, عائم,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ساكن فراغ im2col_3d_blocked_dma_thread(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    بنية htp_im2col_context * ictx        = (بنية htp_im2col_context *) داده;
    بنية htp_ops_context *    octx        = ictx->octx;
    بنية htp_thread_trace * محدود tr   = &octx->ctx->trace[ith];
    ثابت بنية htp_tensor * محدود dst  = octx->dst;
    ثابت uint32_t OW           = ictx->OW;
    ثابت uint32_t KD = ictx->KD, KH = ictx->KH;
    ثابت uint32_t kd_kh        = KD * KH;
    ثابت uint32_t patch_stride = ictx->patch_stride;
    ثابت uint32_t owb          = ictx->pe_owb;
    ثابت uint32_t nbpr         = ictx->pe_nbpr;
    ثابت uint32_t dst_elem     = ictx->dst_elem;
    ثابت dma_addr_t dst_داده   = dst->داده;
    dma_صف * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_پایه         = ictx->pe_vtcm_src + ith * ictx->pe_src_اندازه_در_هر_thread;
    uint8_t * dstb_پایه         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_اندازه_در_هر_thread;
    عائم * srcb2[2]            = { (عائم *) srcb_پایه, (عائم *) (srcb_پایه + ictx->pe_src_ردیف_بایتها) };
    فراغ * dstb2[2]             = { dstb_پایه, dstb_پایه + ictx->pe_dst_ردیف_بایتها };
    عائم * scratch             = (عائم *) (srcb_پایه + 2 * ictx->pe_src_ردیف_بایتها);
    ثابت uint32_t ردیف_انتها_max  = ictx->pe_ردیف_پایه + ictx->pe_nrows;
    ثابت uint32_t ردیف_شروع    = ictx->pe_ردیف_پایه + ictx->pe_ردیفها_در_هر_thread * ith;
    ثابت uint32_t ردیف_انتها      = MIN(ردیف_شروع + ictx->pe_ردیفها_در_هر_thread, ردیف_انتها_max);
    إذا (ردیف_شروع >= ردیف_انتها) {
        أرجع;
    }
    ثابت uint32_t nrows_local  = ردیف_انتها - ردیف_شروع;
    ثابت uint32_t مجموع_بلوکها = nrows_local * nbpr;
    إذا (مجموع_بلوکها == 0) {
        أرجع;
    }
    بنية im2col_cursor cur = {
        .r   = ردیف_شروع,
        .درون  = ictx->thread_درون[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    بنية im2col_cursor stage = cur;
    لكل (uint32_t i = 0; i < مجموع_بلوکها && i < 2; i++) {
        dma_صف_افزودن(dma_q, dma_make_داده(dst_داده, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_گام_بلوک(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    لكل (uint32_t bi = 0; bi < مجموع_بلوکها; bi++) {
        ثابت uint32_t buf = bi & 1u;
        عائم * srcb = srcb2[buf];
        فراغ *  dstb = dstb2[buf];
        ثابت uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_صف_برداشت(dma_q);
        لكل (uint32_t p = 0; p < kd_kh; p++) {
            dma_صف_برداشت(dma_q);
        }
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr = dst_داده + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_صف_افزودن(dma_q, dma_make_داده(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        إذا (bi + 2 < مجموع_بلوکها) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_گام_بلوک(&stage, ictx);
        }
        im2col_3d_گام_بلوک(&cur, ictx);
    }
    dma_صف_flush(dma_q);
}
ساكن فراغ im2col_3d_patchembed_dma_stage(بنية htp_im2col_context * ictx, dma_صف * dma_q,
                                           عائم * srcb, ثابت بنية im2col_cursor * c) {
    ثابت بنية htp_ops_context * octx = ictx->octx;
    ثابت uint32_t IC = ictx->IC;
    ثابت uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KD = ictx->KD, KH = ictx->KH;
    ثابت dma_addr_t src_داده = octx->src[1]->داده;
    ثابت uint32_t درون = c->درون, iod = c->iod, ioh = c->ioh;
    ثابت size_t src_channel_stride  = (size_t) ID * IH * IW * حجم(عائم);
    ثابت size_t vtcm_channel_stride = (size_t) KD * KH * IW * حجم(عائم);
    ثابت size_t ردیف_اندازه            = (size_t) KH * IW * حجم(عائم);

    لكل (uint32_t ikd = 0; ikd < KD; ikd++) {
        ثابت uint32_t iid = iod * KD + ikd;
        عائم * vdst = srcb + (size_t) ikd * KH * IW;
        ثابت dma_addr_t vsrc = src_داده + (size_t) (((درون * IC) * ID + iid) * IH + ioh * KH) * IW * حجم(عائم);
        dma_صف_افزودن(dma_q, dma_make_داده(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, ردیف_اندازه, IC);
    }
}
ساكن فراغ im2col_3d_patchembed_compute_f16(
    ثابت بنية htp_im2col_context * ictx,
    فراغ * محدود dstb_raw, ثابت عائم * محدود srcb) {
    uint16_t * محدود dstb = (uint16_t *) dstb_raw;
    ثابت uint32_t OW           = ictx->OW;
    ثابت uint32_t IC           = ictx->IC;
    ثابت uint32_t KD           = ictx->KD;
    ثابت uint32_t KH           = ictx->KH;
    ثابت uint32_t KW           = ictx->KW;
    ثابت uint32_t IW           = ictx->IW;
    ثابت uint32_t patch_stride = ictx->patch_stride;

    لكل (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        لكل (uint32_t iic = 0; iic < IC; iic++) {
            لكل (uint32_t ikd = 0; ikd < KD; ikd++) {
                لكل (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * بیرون_پیاپی = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ثابت عائم * src_پیاپی = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) بیرون_پیاپی, (ثابت uint8_t *) src_پیاپی, KW);
                }
            }
        }
    }
}

ساكن فراغ im2col_3d_patchembed_compute_f32(
    ثابت بنية htp_im2col_context * ictx,
    فراغ * محدود dstb_raw, ثابت عائم * محدود srcb) {
    عائم * محدود dstb = (عائم *) dstb_raw;
    ثابت uint32_t OW           = ictx->OW;
    ثابت uint32_t IC           = ictx->IC;
    ثابت uint32_t KD           = ictx->KD;
    ثابت uint32_t KH           = ictx->KH;
    ثابت uint32_t KW           = ictx->KW;
    ثابت uint32_t IW           = ictx->IW;
    ثابت uint32_t patch_stride = ictx->patch_stride;

    لكل (uint32_t iow = 0; iow < OW; iow++) {
        عائم * dst_patch = dstb + (size_t) iow * patch_stride;
        لكل (uint32_t iic = 0; iic < IC; iic++) {
            لكل (uint32_t ikd = 0; ikd < KD; ikd++) {
                لكل (uint32_t ikh = 0; ikh < KH; ikh++) {
                    عائم * بیرون_پیاپی = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ثابت عائم * src_پیاپی = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) بیرون_پیاپی, (ثابت uint8_t *) src_پیاپی, KW);
                }
            }
        }
    }
}

ساكن فراغ im2col_3d_patchembed_dma_thread(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    بنية htp_im2col_context * ictx        = (بنية htp_im2col_context *) داده;
    بنية htp_ops_context *    octx        = ictx->octx;
    بنية htp_thread_trace * محدود tr   = &octx->ctx->trace[ith];
    ثابت بنية htp_tensor * محدود dst  = octx->dst;
    ثابت uint32_t KD           = ictx->KD;
    ثابت uint32_t OW           = ictx->OW;
    ثابت uint32_t patch_stride = ictx->patch_stride;
    ثابت uint32_t dst_elem     = ictx->dst_elem;
    ثابت dma_addr_t dst_داده   = dst->داده;
    dma_صف * dma_q           = octx->ctx->dma[ith];
    ثابت uint32_t ردیف_انتها_max  = ictx->pe_ردیف_پایه + ictx->pe_nrows;
    ثابت uint32_t ردیف_شروع    = ictx->pe_ردیف_پایه + ictx->pe_ردیفها_در_هر_thread * ith;
    ثابت uint32_t ردیف_انتها      = MIN(ردیف_شروع + ictx->pe_ردیفها_در_هر_thread, ردیف_انتها_max);
    إذا (ردیف_شروع >= ردیف_انتها) {
        أرجع;
    }

    ثابت uint32_t nrows_local  = ردیف_انتها - ردیف_شروع;
    uint8_t * src_پایه          = ictx->pe_vtcm_src + ith * ictx->pe_src_اندازه_در_هر_thread;
    uint8_t * dstb_پایه         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_اندازه_در_هر_thread;
    عائم * srcb2[2]            = { (عائم *) src_پایه, (عائم *) (src_پایه + ictx->pe_src_ردیف_بایتها) };
    فراغ * dstb2[2]             = { dstb_پایه, dstb_پایه + ictx->pe_dst_ردیف_بایتها };
    بنية im2col_cursor cur = {
        .r   = ردیف_شروع,
        .درون  = ictx->thread_درون[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    بنية im2col_cursor stage = cur;
    لكل (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_صف_افزودن(dma_q, dma_make_داده(dst_داده, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_گام_ردیف(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    لكل (uint32_t idx = 0; idx < nrows_local; idx++) {
        ثابت uint32_t buf = idx & 1u;
        عائم * srcb = srcb2[buf];
        فراغ *  dstb = dstb2[buf];
        dma_صف_برداشت(dma_q);
        لكل (uint32_t p = 0; p < KD; p++) {
            dma_صف_برداشت(dma_q);
        }
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr_ردیف = dst_داده + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_صف_افزودن(dma_q, dma_make_داده(ddr_ردیف, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        إذا (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_گام_ردیف(&stage, ictx);
        }
        im2col_3d_گام_ردیف(&cur, ictx);
    }
    dma_صف_flush(dma_q);
}

صحيح op_im2col(بنية htp_ops_context * octx) {
    ثابت بنية htp_im2col_kernel_params * kparams = (ثابت بنية htp_im2col_kernel_params *) octx->kernel_params;
    إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_UNSUPPORTED) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (!htp_ops_context_مجموعه_n_نخها(octx, kparams->n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * dst = octx->dst;

    uint32_t ردیف_پایه = 0;
    uint32_t nrows    = kparams->مجموع_ردیفها;
    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت uint32_t ردیف_اندازه = dst->nb[2];
        ثابت uint32_t ردیفها_در_هر_chunk = (ردیف_اندازه > 0) ? (HEX_L2_سطر_اندازه / hex_بزرگترین_مقسوم_علیه_u32(ردیف_اندازه, HEX_L2_سطر_اندازه)) : 1;
        ثابت بنية htp_tensor_mdev_range range =
            htp_tensor_mdev_افراز(kparams->مجموع_ردیفها, htp_tensor_mdev_داده_aligned(dst) ? ردیفها_در_هر_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_پایه = range.شروع;
        nrows    = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    بنية htp_im2col_context ictx = { 0 };
    uint8_t * ثابت پایه = octx->ctx->vtcm_پایه;

    ثابت uint32_t pth = MIN(octx->n_نخها, nrows);
    ictx.octx                   = octx;
    ictx.pe_ردیف_پایه            = ردیف_پایه;
    ictx.pe_nrows               = nrows;
    ictx.pe_ردیفها_در_هر_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_ردیف_بایتها       = kparams->src_ردیف_بایتها;
    ictx.pe_dst_ردیف_بایتها       = kparams->dst_ردیف_بایتها;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, پایه, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, پایه, kparams->off_dst);
    ictx.pe_src_اندازه_در_هر_thread = kparams->src_بایتها_در_هر_thread;
    ictx.pe_dst_اندازه_در_هر_thread = kparams->dst_بایتها_در_هر_thread;
    ictx.pe_owb                 = kparams->owb;
    ictx.pe_wb                  = kparams->wb;
    ictx.pe_nbpr                = kparams->nbpr;
    ictx.N                      = kparams->N;
    ictx.IC                     = kparams->IC;
    ictx.ID                     = kparams->ID;
    ictx.IH                     = kparams->IH;
    ictx.IW                     = kparams->IW;
    ictx.KD                     = kparams->KD;
    ictx.KH                     = kparams->KH;
    ictx.KW                     = kparams->KW;
    ictx.OD                     = kparams->OD;
    ictx.OH                     = kparams->OH;
    ictx.OW                     = kparams->OW;
    ictx.patch_stride           = kparams->patch_stride;

    لكل (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = ردیف_پایه + ictx.pe_ردیفها_در_هر_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_درون[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? حجم(__fp16) : حجم(عائم);

    إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (فراغ *) im2col_patchembed_compute_f16
                                                 : (فراغ *) im2col_patchembed_compute_f32;
        work_صف_پیاپی(octx->ctx->work_صف, im2col_patchembed_dma_thread, &ictx, pth);
        أرجع HTP_وضعیت_OK;
    } وإلا إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (فراغ *) im2col_blocked_compute_f16
                                                 : (فراغ *) im2col_blocked_compute_f32;
        work_صف_پیاپی(octx->ctx->work_صف, im2col_blocked_dma_thread, &ictx, pth);
        أرجع HTP_وضعیت_OK;
    }

    أرجع HTP_وضعیت_NO_SUPPORT;
}

صحيح op_im2col_3d(بنية htp_ops_context * octx) {
    ثابت بنية htp_im2col_kernel_params * kparams =
        (ثابت بنية htp_im2col_kernel_params *) octx->kernel_params;
    إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_UNSUPPORTED) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (!htp_ops_context_مجموعه_n_نخها(octx, kparams->n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * dst = octx->dst;

    uint32_t ردیف_پایه = 0;
    uint32_t nrows    = kparams->مجموع_ردیفها;
    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت uint32_t ردیف_اندازه = dst->nb[2];
        ثابت uint32_t ردیفها_در_هر_chunk = (ردیف_اندازه > 0) ? (HEX_L2_سطر_اندازه / hex_بزرگترین_مقسوم_علیه_u32(ردیف_اندازه, HEX_L2_سطر_اندازه)) : 1;
        ثابت بنية htp_tensor_mdev_range range =
            htp_tensor_mdev_افراز(kparams->مجموع_ردیفها, htp_tensor_mdev_داده_aligned(dst) ? ردیفها_در_هر_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_پایه = range.شروع;
        nrows    = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    بنية htp_im2col_context ictx = { 0 };
    uint8_t * ثابت پایه = octx->ctx->vtcm_پایه;

    ثابت uint32_t pth = MIN(octx->n_نخها, nrows);
    ictx.octx                   = octx;
    ictx.pe_ردیف_پایه            = ردیف_پایه;
    ictx.pe_nrows               = nrows;
    ictx.pe_ردیفها_در_هر_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_ردیف_بایتها       = kparams->src_ردیف_بایتها;
    ictx.pe_dst_ردیف_بایتها       = kparams->dst_ردیف_بایتها;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, پایه, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, پایه, kparams->off_dst);
    ictx.pe_src_اندازه_در_هر_thread = kparams->src_بایتها_در_هر_thread;
    ictx.pe_dst_اندازه_در_هر_thread = kparams->dst_بایتها_در_هر_thread;
    ictx.pe_owb                 = kparams->owb;
    ictx.pe_wb                  = kparams->wb;
    ictx.pe_nbpr                = kparams->nbpr;
    ictx.N                      = kparams->N;
    ictx.IC                     = kparams->IC;
    ictx.ID                     = kparams->ID;
    ictx.IH                     = kparams->IH;
    ictx.IW                     = kparams->IW;
    ictx.KD                     = kparams->KD;
    ictx.KH                     = kparams->KH;
    ictx.KW                     = kparams->KW;
    ictx.OD                     = kparams->OD;
    ictx.OH                     = kparams->OH;
    ictx.OW                     = kparams->OW;
    ictx.patch_stride           = kparams->patch_stride;

    لكل (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = ردیف_پایه + ictx.pe_ردیفها_در_هر_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_درون[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? حجم(__fp16) : حجم(عائم);

    إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (فراغ *) im2col_3d_patchembed_compute_f16
                                                 : (فراغ *) im2col_3d_patchembed_compute_f32;
        work_صف_پیاپی(octx->ctx->work_صف, im2col_3d_patchembed_dma_thread, &ictx, pth);
        أرجع HTP_وضعیت_OK;
    } وإلا إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (فراغ *) im2col_3d_blocked_compute_f16
                                                 : (فراغ *) im2col_3d_blocked_compute_f32;
        work_صف_پیاپی(octx->ctx->work_صف, im2col_3d_blocked_dma_thread, &ictx, pth);
        أرجع HTP_وضعیت_OK;
    }

    أرجع HTP_وضعیت_NO_SUPPORT;
}
