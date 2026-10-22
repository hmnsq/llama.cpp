#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <hexagon_protos.h>
#ضمن <hexagon_types.h>
#ضمن <string.h>

#ضمن "hex-common.h"

#تعریف_کړه GGML_COMMON_DECL_C
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

    uint32_t pe_لیکه_بنسټ;                          // first output row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of output rows assigned to this dev (DMA path)
    uint32_t pe_لیکې_فی_thread;                   // output rows per worker
    uint32_t pe_src_لیکه_بایټونه;                     // one output row's source, rounded 256
    uint32_t pe_dst_لیکه_بایټونه;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_اندازه_فی_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_اندازه_فی_thread;              // 2 * pe_dst_row_bytes

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
    uint32_t thread_دننه[HTP_MAX_NTHREADS];
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
    uint32_t دننه;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

ساكن مضمن فراغ im2col_ګام_لیکه(بنية im2col_cursor * c, ثابت بنية htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    إذا (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->دننه++;
    }
}

ساكن مضمن فراغ im2col_ګام_بلاک(بنية im2col_cursor * c, ثابت بنية htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    إذا (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        إذا (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->دننه++;
        }
    }
}

ساكن فراغ im2col_blocked_dma_stage(بنية htp_im2col_context * ictx, dma_کتار * dma_q,
                                     عائم * srcb, ثابت بنية im2col_cursor * c) {
    ثابت uint32_t دننه = c->دننه, ioh = c->ioh, c0 = c->c0;
    ثابت بنية htp_ops_context * octx = ictx->octx;
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    ثابت int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    ثابت int32_t d1 = octx->op_params[5];
    ثابت bool is_2D = (octx->op_params[6] == 1);
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KH = ictx->KH;
    ثابت uint32_t Wb = ictx->pe_wb;
    ثابت dma_addr_t src_معلومات = src1->معلومات;
    ثابت int32_t win0 = (int32_t) c0 * s0 - p0;
    ثابت size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    ثابت size_t batch_دننه_واټن        = is_2D ? (size_t) دننه * src1->nb[3] : (size_t) دننه * src1->nb[2];

    لكل (uint32_t ikh = 0; ikh < KH; ikh++) {
        ثابت int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        إذا (iih < 0 || iih >= (int32_t) IH) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(srcb, 0), 0, 0, 0, 0);
            استمر;
        }
        ثابت int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        إذا (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        إذا (hi <= lo) {
            dma_کتار_ټېل(dma_q, dma_make_معلومات(srcb, 0), 0, 0, 0, 0);
            استمر;
        }
        ثابت uint32_t cpw = (uint32_t) (hi - lo);
        عائم * vdst = srcb + (size_t) ikh * Wb + lo;
        ثابت dma_addr_t vsrc = src_معلومات + batch_دننه_واټن + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_کتار_ټېل(dma_q, dma_make_معلومات(vdst, vsrc),
                       (size_t) KH * Wb * حجم(عائم), src_channel_stride,
                       cpw * حجم(عائم), IC);
    }
}

#تعریف_کړه IM2_ستون_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
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
                DST_CTYPE * بهر_پرلپسې = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                إذا (!okh) {                                                                      \
                    SPLAT_FN(بهر_پرلپسې, 0, KW);                                                    \
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
                        SPLAT_FN(بهر_پرلپسې, 0, KW);                                                \
                    } وإلا {                                                                     \
                        إذا (lo > 0) {                                                            \
                            SPLAT_FN(بهر_پرلپسې, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (بهر_پرلپسې + lo),                                      \
                                (ثابت uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        إذا (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(بهر_پرلپسې + hi, 0, (uint32_t) (KW - hi));                     \
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
                    COPY_FN((uint8_t *) (بهر_پرلپسې + ikw), (ثابت uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2_ستون_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2_ستون_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, عائم,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ساكن فراغ im2col_blocked_dma_thread(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية htp_im2col_context * ictx      = (بنية htp_im2col_context *) معلومات;
    بنية htp_ops_context *    octx      = ictx->octx;
    بنية htp_thread_trace * محدود tr = &octx->ctx->trace[ith];
    ثابت uint32_t OW                     = ictx->OW;
    ثابت uint32_t KH                     = ictx->KH;
    ثابت uint32_t owb                    = ictx->pe_owb;
    ثابت uint32_t nbpr                   = ictx->pe_nbpr;
    ثابت uint32_t patch_stride           = ictx->patch_stride;
    ثابت uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    ثابت dma_addr_t dst_معلومات             = octx->dst->معلومات;
    dma_کتار *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_بنسټ              = ictx->pe_vtcm_src + ith * ictx->pe_src_اندازه_فی_thread;
    uint8_t *      dstb_بنسټ              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_اندازه_فی_thread;
    عائم *        srcb2[2]               = { (عائم *) srcb_بنسټ, (عائم *) (srcb_بنسټ + ictx->pe_src_لیکه_بایټونه) };
    فراغ *         dstb2[2]               = { dstb_بنسټ, dstb_بنسټ + ictx->pe_dst_لیکه_بایټونه };
    عائم *        scratch                = (عائم *) (srcb_بنسټ + 2 * ictx->pe_src_لیکه_بایټونه);
    ثابت uint32_t لیکه_پای_ټکی_max            = ictx->pe_لیکه_بنسټ + ictx->pe_nrows;
    ثابت uint32_t لیکه_پیل              = ictx->pe_لیکه_بنسټ + ictx->pe_لیکې_فی_thread * ith;
    ثابت uint32_t لیکه_پای_ټکی                = MIN(لیکه_پیل + ictx->pe_لیکې_فی_thread, لیکه_پای_ټکی_max);
    إذا (لیکه_پیل >= لیکه_پای_ټکی) {
        أرجع;
    }
    ثابت uint32_t nrows_local  = لیکه_پای_ټکی - لیکه_پیل;
    ثابت uint32_t ټول_بلاکونه = nrows_local * nbpr;
    إذا (ټول_بلاکونه == 0) {
        أرجع;
    }
    بنية im2col_cursor cur = {
        .r   = لیکه_پیل,
        .دننه  = ictx->thread_دننه[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    بنية im2col_cursor stage = cur;
    لكل (uint32_t i = 0; i < ټول_بلاکونه && i < 2; i++) {
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_معلومات, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_ګام_بلاک(&stage, ictx);
    }
    لكل (uint32_t bi = 0; bi < ټول_بلاکونه; bi++) {
        ثابت uint32_t buf = bi & 1u;
        عائم * srcb = srcb2[buf];
        فراغ *  dstb = dstb2[buf];
        ثابت uint32_t nb = MIN(owb, OW - cur.c0);
        dma_کتار_راوباسه(dma_q);
        لكل (uint32_t k = 0; k < KH; k++) {
            dma_کتار_راوباسه(dma_q);
        }
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr = dst_معلومات + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        إذا (bi + 2 < ټول_بلاکونه) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_ګام_بلاک(&stage, ictx);
        }
        im2col_ګام_بلاک(&cur, ictx);
    }
    dma_کتار_flush(dma_q);
}

ساكن فراغ im2col_patchembed_dma_stage(بنية htp_im2col_context * ictx, dma_کتار * dma_q,
                                        عائم * srcb, ثابت بنية im2col_cursor * c) {
    ثابت بنية htp_ops_context * octx = ictx->octx;
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KH = ictx->KH;
    ثابت uint32_t دننه = c->دننه, ioh = c->ioh;
    ثابت size_t src_channel_stride  = (size_t) IH * IW * حجم(عائم);
    ثابت size_t vtcm_channel_stride = (size_t) KH * IW * حجم(عائم);
    ثابت dma_addr_t src_معلومات        = octx->src[1]->معلومات;
    ثابت dma_addr_t vsrc            = src_معلومات + (size_t) (((دننه * IC) * IH + ioh * KH) * IW) * حجم(عائم);

    dma_کتار_ټېل(dma_q, dma_make_معلومات(srcb, vsrc),
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
                uint16_t * بهر_پرلپسې = dst_patch + (size_t) (iic * KH + ikh) * KW;
                إذا (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(بهر_پرلپسې, 0, KW);
                    استمر;
                }
                ثابت عائم * src_پرلپسې = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) بهر_پرلپسې, (ثابت uint8_t *) src_پرلپسې, KW);
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
                عائم * بهر_پرلپسې = dst_patch + (size_t) (iic * KH + ikh) * KW;
                إذا (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(بهر_پرلپسې, 0, KW);
                    استمر;
                }
                ثابت عائم * src_پرلپسې = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) بهر_پرلپسې, (ثابت uint8_t *) src_پرلپسې, KW);
            }
        }
    }
}

ساكن فراغ im2col_patchembed_dma_thread(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية htp_im2col_context * ictx        = (بنية htp_im2col_context *) معلومات;
    بنية htp_ops_context *    octx        = ictx->octx;
    بنية htp_thread_trace * محدود tr   = &octx->ctx->trace[ith];
    ثابت uint32_t OW                       = ictx->OW;
    ثابت uint32_t patch_stride             = ictx->patch_stride;
    ثابت uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    ثابت dma_addr_t dst_معلومات               = octx->dst->معلومات;
    dma_کتار *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_بنسټ                = ictx->pe_vtcm_src + ith * ictx->pe_src_اندازه_فی_thread;
    uint8_t *      dstb_بنسټ                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_اندازه_فی_thread;
    عائم *        srcb2[2]                 = { (عائم *) srcb_بنسټ, (عائم *) (srcb_بنسټ + ictx->pe_src_لیکه_بایټونه) };
    فراغ *         dstb2[2]                 = { dstb_بنسټ, dstb_بنسټ + ictx->pe_dst_لیکه_بایټونه };
    ثابت uint32_t لیکه_پای_ټکی_max              = ictx->pe_لیکه_بنسټ + ictx->pe_nrows;
    ثابت uint32_t لیکه_پیل                = ictx->pe_لیکه_بنسټ + ictx->pe_لیکې_فی_thread * ith;
    ثابت uint32_t لیکه_پای_ټکی                  = MIN(لیکه_پیل + ictx->pe_لیکې_فی_thread, لیکه_پای_ټکی_max);
    إذا (لیکه_پیل >= لیکه_پای_ټکی) {
        أرجع;
    }
    ثابت uint32_t ټول_لیکې = لیکه_پای_ټکی - لیکه_پیل;
    إذا (ټول_لیکې == 0) {
        أرجع;
    }
    بنية im2col_cursor cur = {
        .r   = لیکه_پیل,
        .دننه  = ictx->thread_دننه[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    بنية im2col_cursor stage = cur;
    لكل (uint32_t i = 0; i < ټول_لیکې && i < 2; i++) {
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_معلومات, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_ګام_لیکه(&stage, ictx);
    }
    لكل (uint32_t ri = 0; ri < ټول_لیکې; ri++) {
        ثابت uint32_t buf = ri & 1u;
        عائم * srcb = srcb2[buf];
        فراغ *  dstb = dstb2[buf];
        dma_کتار_راوباسه(dma_q);
        dma_کتار_راوباسه(dma_q);
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr_لیکه = dst_معلومات + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(ddr_لیکه, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        إذا (ri + 2 < ټول_لیکې) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_ګام_لیکه(&stage, ictx);
        }
        im2col_ګام_لیکه(&cur, ictx);
    }
    dma_کتار_flush(dma_q);
}

ساكن مضمن فراغ im2col_3d_ګام_لیکه(بنية im2col_cursor * c, ثابت بنية htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    إذا (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        إذا (c->iod >= ictx->OD) {
            c->iod = 0;
            c->دننه++;
        }
    }
}

ساكن مضمن فراغ im2col_3d_ګام_بلاک(بنية im2col_cursor * c, ثابت بنية htp_im2col_context * ictx) {
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
                c->دننه++;
            }
        }
    }
}

ساكن فراغ im2col_3d_blocked_dma_stage(بنية htp_im2col_context * ictx, dma_کتار * dma_q,
                                        عائم * srcb, ثابت بنية im2col_cursor * c) {
    ثابت uint32_t دننه = c->دننه, iod = c->iod, ioh = c->ioh, c0 = c->c0;
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
    ثابت dma_addr_t src_معلومات = src1->معلومات;
    ثابت size_t src_channel_stride = src1->nb[3];
    ثابت size_t vtcm_channel_stride = (size_t) KD * KH * Wb * حجم(عائم);
    ثابت size_t batch_دننه_واټن        = (size_t) (دننه * IC) * src1->nb[3];

    لكل (uint32_t ikd = 0; ikd < KD; ikd++) {
        ثابت int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        ثابت bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        لكل (uint32_t ikh = 0; ikh < KH; ikh++) {
            ثابت int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            إذا (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(srcb, 0), 0, 0, 0, 0);
                استمر;
            }
            ثابت int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            إذا (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            إذا (hi <= lo) {
                dma_کتار_ټېل(dma_q, dma_make_معلومات(srcb, 0), 0, 0, 0, 0);
                استمر;
            }
            عائم * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            ثابت dma_addr_t vsrc = src_معلومات + batch_دننه_واټن +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * حجم(عائم), IC);
        }
    }
}

#تعریف_کړه IM2_ستون_3_D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
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
                    DST_CTYPE * بهر_پرلپسې = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    إذا (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(بهر_پرلپسې, 0, KW);                                                    \
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
                            SPLAT_FN(بهر_پرلپسې, 0, KW);                                                \
                        } وإلا {                                                                     \
                            إذا (lo > 0) {                                                            \
                                SPLAT_FN(بهر_پرلپسې, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (بهر_پرلپسې + lo),                                      \
                                    (ثابت uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            إذا (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(بهر_پرلپسې + hi, 0, (uint32_t) (KW - hi));                     \
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
                        COPY_FN((uint8_t *) (بهر_پرلپسې + ikw), (ثابت uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2_ستون_3_D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2_ستون_3_D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, عائم,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ساكن فراغ im2col_3d_blocked_dma_thread(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية htp_im2col_context * ictx        = (بنية htp_im2col_context *) معلومات;
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
    ثابت dma_addr_t dst_معلومات   = dst->معلومات;
    dma_کتار * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_بنسټ         = ictx->pe_vtcm_src + ith * ictx->pe_src_اندازه_فی_thread;
    uint8_t * dstb_بنسټ         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_اندازه_فی_thread;
    عائم * srcb2[2]            = { (عائم *) srcb_بنسټ, (عائم *) (srcb_بنسټ + ictx->pe_src_لیکه_بایټونه) };
    فراغ * dstb2[2]             = { dstb_بنسټ, dstb_بنسټ + ictx->pe_dst_لیکه_بایټونه };
    عائم * scratch             = (عائم *) (srcb_بنسټ + 2 * ictx->pe_src_لیکه_بایټونه);
    ثابت uint32_t لیکه_پای_ټکی_max  = ictx->pe_لیکه_بنسټ + ictx->pe_nrows;
    ثابت uint32_t لیکه_پیل    = ictx->pe_لیکه_بنسټ + ictx->pe_لیکې_فی_thread * ith;
    ثابت uint32_t لیکه_پای_ټکی      = MIN(لیکه_پیل + ictx->pe_لیکې_فی_thread, لیکه_پای_ټکی_max);
    إذا (لیکه_پیل >= لیکه_پای_ټکی) {
        أرجع;
    }
    ثابت uint32_t nrows_local  = لیکه_پای_ټکی - لیکه_پیل;
    ثابت uint32_t ټول_بلاکونه = nrows_local * nbpr;
    إذا (ټول_بلاکونه == 0) {
        أرجع;
    }
    بنية im2col_cursor cur = {
        .r   = لیکه_پیل,
        .دننه  = ictx->thread_دننه[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    بنية im2col_cursor stage = cur;
    لكل (uint32_t i = 0; i < ټول_بلاکونه && i < 2; i++) {
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_معلومات, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_ګام_بلاک(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    لكل (uint32_t bi = 0; bi < ټول_بلاکونه; bi++) {
        ثابت uint32_t buf = bi & 1u;
        عائم * srcb = srcb2[buf];
        فراغ *  dstb = dstb2[buf];
        ثابت uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_کتار_راوباسه(dma_q);
        لكل (uint32_t p = 0; p < kd_kh; p++) {
            dma_کتار_راوباسه(dma_q);
        }
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr = dst_معلومات + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        إذا (bi + 2 < ټول_بلاکونه) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_ګام_بلاک(&stage, ictx);
        }
        im2col_3d_ګام_بلاک(&cur, ictx);
    }
    dma_کتار_flush(dma_q);
}
ساكن فراغ im2col_3d_patchembed_dma_stage(بنية htp_im2col_context * ictx, dma_کتار * dma_q,
                                           عائم * srcb, ثابت بنية im2col_cursor * c) {
    ثابت بنية htp_ops_context * octx = ictx->octx;
    ثابت uint32_t IC = ictx->IC;
    ثابت uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KD = ictx->KD, KH = ictx->KH;
    ثابت dma_addr_t src_معلومات = octx->src[1]->معلومات;
    ثابت uint32_t دننه = c->دننه, iod = c->iod, ioh = c->ioh;
    ثابت size_t src_channel_stride  = (size_t) ID * IH * IW * حجم(عائم);
    ثابت size_t vtcm_channel_stride = (size_t) KD * KH * IW * حجم(عائم);
    ثابت size_t لیکه_اندازه            = (size_t) KH * IW * حجم(عائم);

    لكل (uint32_t ikd = 0; ikd < KD; ikd++) {
        ثابت uint32_t iid = iod * KD + ikd;
        عائم * vdst = srcb + (size_t) ikd * KH * IW;
        ثابت dma_addr_t vsrc = src_معلومات + (size_t) (((دننه * IC) * ID + iid) * IH + ioh * KH) * IW * حجم(عائم);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, لیکه_اندازه, IC);
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
                    uint16_t * بهر_پرلپسې = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ثابت عائم * src_پرلپسې = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) بهر_پرلپسې, (ثابت uint8_t *) src_پرلپسې, KW);
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
                    عائم * بهر_پرلپسې = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ثابت عائم * src_پرلپسې = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) بهر_پرلپسې, (ثابت uint8_t *) src_پرلپسې, KW);
                }
            }
        }
    }
}

ساكن فراغ im2col_3d_patchembed_dma_thread(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية htp_im2col_context * ictx        = (بنية htp_im2col_context *) معلومات;
    بنية htp_ops_context *    octx        = ictx->octx;
    بنية htp_thread_trace * محدود tr   = &octx->ctx->trace[ith];
    ثابت بنية htp_tensor * محدود dst  = octx->dst;
    ثابت uint32_t KD           = ictx->KD;
    ثابت uint32_t OW           = ictx->OW;
    ثابت uint32_t patch_stride = ictx->patch_stride;
    ثابت uint32_t dst_elem     = ictx->dst_elem;
    ثابت dma_addr_t dst_معلومات   = dst->معلومات;
    dma_کتار * dma_q           = octx->ctx->dma[ith];
    ثابت uint32_t لیکه_پای_ټکی_max  = ictx->pe_لیکه_بنسټ + ictx->pe_nrows;
    ثابت uint32_t لیکه_پیل    = ictx->pe_لیکه_بنسټ + ictx->pe_لیکې_فی_thread * ith;
    ثابت uint32_t لیکه_پای_ټکی      = MIN(لیکه_پیل + ictx->pe_لیکې_فی_thread, لیکه_پای_ټکی_max);
    إذا (لیکه_پیل >= لیکه_پای_ټکی) {
        أرجع;
    }

    ثابت uint32_t nrows_local  = لیکه_پای_ټکی - لیکه_پیل;
    uint8_t * src_بنسټ          = ictx->pe_vtcm_src + ith * ictx->pe_src_اندازه_فی_thread;
    uint8_t * dstb_بنسټ         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_اندازه_فی_thread;
    عائم * srcb2[2]            = { (عائم *) src_بنسټ, (عائم *) (src_بنسټ + ictx->pe_src_لیکه_بایټونه) };
    فراغ * dstb2[2]             = { dstb_بنسټ, dstb_بنسټ + ictx->pe_dst_لیکه_بایټونه };
    بنية im2col_cursor cur = {
        .r   = لیکه_پیل,
        .دننه  = ictx->thread_دننه[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    بنية im2col_cursor stage = cur;
    لكل (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_معلومات, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_ګام_لیکه(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    لكل (uint32_t idx = 0; idx < nrows_local; idx++) {
        ثابت uint32_t buf = idx & 1u;
        عائم * srcb = srcb2[buf];
        فراغ *  dstb = dstb2[buf];
        dma_کتار_راوباسه(dma_q);
        لكل (uint32_t p = 0; p < KD; p++) {
            dma_کتار_راوباسه(dma_q);
        }
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr_لیکه = dst_معلومات + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(ddr_لیکه, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        إذا (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_ګام_لیکه(&stage, ictx);
        }
        im2col_3d_ګام_لیکه(&cur, ictx);
    }
    dma_کتار_flush(dma_q);
}

صحيح op_im2col(بنية htp_ops_context * octx) {
    ثابت بنية htp_im2col_kernel_params * kparams = (ثابت بنية htp_im2col_kernel_params *) octx->kernel_params;
    إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_UNSUPPORTED) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (!htp_ops_context_سېټ_n_تارونه(octx, kparams->n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * dst = octx->dst;

    uint32_t لیکه_بنسټ = 0;
    uint32_t nrows    = kparams->ټول_لیکې;
    إذا (octx->ctx->mdev.شمېر > 1) {
        ثابت uint32_t لیکه_اندازه = dst->nb[2];
        ثابت uint32_t لیکې_فی_chunk = (لیکه_اندازه > 0) ? (HEX_L2_کرښه_اندازه / hex_لوی_ګډ_وېشونکی_u32(لیکه_اندازه, HEX_L2_کرښه_اندازه)) : 1;
        ثابت بنية htp_tensor_mdev_range range =
            htp_tensor_mdev_برخه_کول(kparams->ټول_لیکې, htp_tensor_mdev_معلومات_aligned(dst) ? لیکې_فی_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_بنسټ = range.پیل;
        nrows    = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    بنية htp_im2col_context ictx = { 0 };
    uint8_t * ثابت بنسټ = octx->ctx->vtcm_بنسټ;

    ثابت uint32_t pth = MIN(octx->n_تارونه, nrows);
    ictx.octx                   = octx;
    ictx.pe_لیکه_بنسټ            = لیکه_بنسټ;
    ictx.pe_nrows               = nrows;
    ictx.pe_لیکې_فی_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_لیکه_بایټونه       = kparams->src_لیکه_بایټونه;
    ictx.pe_dst_لیکه_بایټونه       = kparams->dst_لیکه_بایټونه;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, بنسټ, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, بنسټ, kparams->off_dst);
    ictx.pe_src_اندازه_فی_thread = kparams->src_بایټونه_فی_thread;
    ictx.pe_dst_اندازه_فی_thread = kparams->dst_بایټونه_فی_thread;
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
        uint32_t r = لیکه_بنسټ + ictx.pe_لیکې_فی_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_دننه[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? حجم(__fp16) : حجم(عائم);

    إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (فراغ *) im2col_patchembed_compute_f16
                                                 : (فراغ *) im2col_patchembed_compute_f32;
        work_کتار_پرلپسې(octx->ctx->work_کتار, im2col_patchembed_dma_thread, &ictx, pth);
        أرجع HTP_حالت_OK;
    } وإلا إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (فراغ *) im2col_blocked_compute_f16
                                                 : (فراغ *) im2col_blocked_compute_f32;
        work_کتار_پرلپسې(octx->ctx->work_کتار, im2col_blocked_dma_thread, &ictx, pth);
        أرجع HTP_حالت_OK;
    }

    أرجع HTP_حالت_NO_SUPPORT;
}

صحيح op_im2col_3d(بنية htp_ops_context * octx) {
    ثابت بنية htp_im2col_kernel_params * kparams =
        (ثابت بنية htp_im2col_kernel_params *) octx->kernel_params;
    إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_UNSUPPORTED) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (!htp_ops_context_سېټ_n_تارونه(octx, kparams->n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * dst = octx->dst;

    uint32_t لیکه_بنسټ = 0;
    uint32_t nrows    = kparams->ټول_لیکې;
    إذا (octx->ctx->mdev.شمېر > 1) {
        ثابت uint32_t لیکه_اندازه = dst->nb[2];
        ثابت uint32_t لیکې_فی_chunk = (لیکه_اندازه > 0) ? (HEX_L2_کرښه_اندازه / hex_لوی_ګډ_وېشونکی_u32(لیکه_اندازه, HEX_L2_کرښه_اندازه)) : 1;
        ثابت بنية htp_tensor_mdev_range range =
            htp_tensor_mdev_برخه_کول(kparams->ټول_لیکې, htp_tensor_mdev_معلومات_aligned(dst) ? لیکې_فی_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_بنسټ = range.پیل;
        nrows    = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    بنية htp_im2col_context ictx = { 0 };
    uint8_t * ثابت بنسټ = octx->ctx->vtcm_بنسټ;

    ثابت uint32_t pth = MIN(octx->n_تارونه, nrows);
    ictx.octx                   = octx;
    ictx.pe_لیکه_بنسټ            = لیکه_بنسټ;
    ictx.pe_nrows               = nrows;
    ictx.pe_لیکې_فی_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_لیکه_بایټونه       = kparams->src_لیکه_بایټونه;
    ictx.pe_dst_لیکه_بایټونه       = kparams->dst_لیکه_بایټونه;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, بنسټ, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, بنسټ, kparams->off_dst);
    ictx.pe_src_اندازه_فی_thread = kparams->src_بایټونه_فی_thread;
    ictx.pe_dst_اندازه_فی_thread = kparams->dst_بایټونه_فی_thread;
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
        uint32_t rem = لیکه_بنسټ + ictx.pe_لیکې_فی_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_دننه[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? حجم(__fp16) : حجم(عائم);

    إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (فراغ *) im2col_3d_patchembed_compute_f16
                                                 : (فراغ *) im2col_3d_patchembed_compute_f32;
        work_کتار_پرلپسې(octx->ctx->work_کتار, im2col_3d_patchembed_dma_thread, &ictx, pth);
        أرجع HTP_حالت_OK;
    } وإلا إذا (kparams->kernel_type == HTP_IM2_ستون_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (فراغ *) im2col_3d_blocked_compute_f16
                                                 : (فراغ *) im2col_3d_blocked_compute_f32;
        work_کتار_پرلپسې(octx->ctx->work_کتار, im2col_3d_blocked_dma_thread, &ictx, pth);
        أرجع HTP_حالت_OK;
    }

    أرجع HTP_حالت_NO_SUPPORT;
}
