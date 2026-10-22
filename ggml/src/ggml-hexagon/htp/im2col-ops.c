#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#include "hex-common.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-vtcm.h"
#include "htp-tensor.h"
#include "im2col-ops.h"

sa_nracanaa htp_im2col_context {
    sa_nracanaa htp_ops_context * octx;

    uint32_t pe_row_base;                          // first output row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of output rows assigned to this dev (DMA path)
    uint32_t pe_rows_per_thread;                   // output rows per worker
    uint32_t pe_src_row_bytes;                     // one output row's source, rounded 256
    uint32_t pe_dst_row_bytes;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_size_per_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_size_per_thread;              // 2 * pe_dst_row_bytes

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
    uint32_t thread_in[HTP_MAX_NTHREADS];
    uint32_t thread_iod[HTP_MAX_NTHREADS];
    uint32_t thread_ioh[HTP_MAX_NTHREADS];

    void * compute;
    uint32_t dst_elem;
};

typedef void (*im2col_blocked_compute_fn_t)(
    const sa_nracanaa htp_im2col_context * ictx,
    void * restrict dstb, const float * restrict srcb,
    float * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

typedef void (*im2col_patchembed_compute_fn_t)(
    const sa_nracanaa htp_im2col_context * ictx,
    void * restrict dstb, const float * restrict srcb, uint32_t ioh);

typedef void (*im2col_3d_blocked_compute_fn_t)(
    const sa_nracanaa htp_im2col_context * ictx,
    void * restrict dstb, const float * restrict srcb,
    float * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

typedef void (*im2col_3d_patchembed_compute_fn_t)(
    const sa_nracanaa htp_im2col_context * ictx,
    void * restrict dstb, const float * restrict srcb);

sa_nracanaa im2col_cursor {
    uint32_t r;
    uint32_t in;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

static anta_hstha void im2col_step_row(sa_nracanaa im2col_cursor * c, const sa_nracanaa htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    ᮚᮦᮔ᮪ (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->in++;
    }
}

static anta_hstha void im2col_step_block(sa_nracanaa im2col_cursor * c, const sa_nracanaa htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    ᮚᮦᮔ᮪ (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        ᮚᮦᮔ᮪ (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->in++;
        }
    }
}

static void im2col_blocked_dma_stage(sa_nracanaa htp_im2col_context * ictx, dma_queue * dma_q,
                                     float * srcb, const sa_nracanaa im2col_cursor * c) {
    const uint32_t in = c->in, ioh = c->ioh, c0 = c->c0;
    const sa_nracanaa htp_ops_context * octx = ictx->octx;
    const sa_nracanaa htp_tensor * src1 = octx->src[1];
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    const int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    const int32_t d1 = octx->op_params[5];
    const bool is_2D = (octx->op_params[6] == 1);
    const uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KH = ictx->KH;
    const uint32_t Wb = ictx->pe_wb;
    const dma_addr_t src_data = src1->data;
    const int32_t win0 = (int32_t) c0 * s0 - p0;
    const size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    const size_t batch_offset        = is_2D ? (size_t) in * src1->nb[3] : (size_t) in * src1->nb[2];

    for (uint32_t ikh = 0; ikh < KH; ikh++) {
        const int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        ᮚᮦᮔ᮪ (iih < 0 || iih >= (int32_t) IH) {
            dma_queue_push(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            continue;
        }
        const int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        ᮚᮦᮔ᮪ (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        ᮚᮦᮔ᮪ (hi <= lo) {
            dma_queue_push(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            continue;
        }
        const uint32_t cpw = (uint32_t) (hi - lo);
        float * vdst = srcb + (size_t) ikh * Wb + lo;
        const dma_addr_t vsrc = src_data + batch_offset + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_queue_push(dma_q, dma_make_data(vdst, vsrc),
                       (size_t) KH * Wb * parimaa_na(float), src_channel_stride,
                       cpw * parimaa_na(float), IC);
    }
}

#define IM2COL_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
static void FNAME(                                                                               \
    const sa_nracanaa htp_im2col_context * ictx,                                                      \
    void * restrict dstb_raw, const float * restrict srcb,                                       \
    float * restrict scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                          \
    const sa_nracanaa htp_ops_context * octx = ictx->octx;                                            \
    const int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    const int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    const int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    const uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    const uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    const uint32_t Wb = ictx->pe_wb;                                                             \
    const uint32_t patch_stride = ictx->patch_stride;                                            \
    const int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    for (uint32_t j = 0; j < nb; j++) {                                                          \
        const uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (size_t) j * patch_stride;                             \
        const int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            const int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            const puur_naa_nka     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            for (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                ᮚᮦᮔ᮪ (!okh) {                                                                      \
                    SPLAT_FN(out_run, 0, KW);                                                    \
                    continue;                                                                    \
                }                                                                                \
                const float * vrow = srcb + (size_t) (iic * KH + ikh) * Wb;                      \
                ᮚᮦᮔ᮪ (d0 == 1) {                                                                   \
                    const int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    ᮚᮦᮔ᮪ (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    ᮚᮦᮔ᮪ (hi <= lo) {                                                              \
                        SPLAT_FN(out_run, 0, KW);                                                \
                    } ᮜᮤᮚ {                                                                     \
                        ᮚᮦᮔ᮪ (lo > 0) {                                                            \
                            SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (out_run + lo),                                      \
                                (const uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        ᮚᮦᮔ᮪ (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    continue;                                                                    \
                }                                                                                \
                for (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    const uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    for (uint32_t c = 0; c < chunk; c++) {                                       \
                        const int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (out_run + ikw), (const uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, float,    hvx_copy_f32_uu,     hvx_splat_f32_u)

static void im2col_blocked_dma_thread(unsigned puur_naa_nka nth, unsigned puur_naa_nka ith, void * data) {
    (void) nth;
    sa_nracanaa htp_im2col_context * ictx      = (sa_nracanaa htp_im2col_context *) data;
    sa_nracanaa htp_ops_context *    octx      = ictx->octx;
    sa_nracanaa htp_thread_trace * restrict tr = &octx->ctx->trace[ith];
    const uint32_t OW                     = ictx->OW;
    const uint32_t KH                     = ictx->KH;
    const uint32_t owb                    = ictx->pe_owb;
    const uint32_t nbpr                   = ictx->pe_nbpr;
    const uint32_t patch_stride           = ictx->patch_stride;
    const uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    const dma_addr_t dst_data             = octx->dst->data;
    dma_queue *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_base              = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    float *        srcb2[2]               = { (float *) srcb_base, (float *) (srcb_base + ictx->pe_src_row_bytes) };
    void *         dstb2[2]               = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    float *        scratch                = (float *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    const uint32_t row_end_max            = ictx->pe_row_base + ictx->pe_nrows;
    const uint32_t row_start              = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    const uint32_t row_end                = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    ᮚᮦᮔ᮪ (row_start >= row_end) {
        ᮝᮜᮤ;
    }
    const uint32_t nrows_local  = row_end - row_start;
    const uint32_t total_blocks = nrows_local * nbpr;
    ᮚᮦᮔ᮪ (total_blocks == 0) {
        ᮝᮜᮤ;
    }
    sa_nracanaa im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    sa_nracanaa im2col_cursor stage = cur;
    for (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_push(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_block(&stage, ictx);
    }
    for (uint32_t bi = 0; bi < total_blocks; bi++) {
        const uint32_t buf = bi & 1u;
        float * srcb = srcb2[buf];
        void *  dstb = dstb2[buf];
        const uint32_t nb = MIN(owb, OW - cur.c0);
        dma_queue_pop(dma_q);
        for (uint32_t k = 0; k < KH; k++) {
            dma_queue_pop(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        const dma_addr_t ddr = dst_data + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_push(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        ᮚᮦᮔ᮪ (bi + 2 < total_blocks) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_block(&stage, ictx);
        }
        im2col_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

static void im2col_patchembed_dma_stage(sa_nracanaa htp_im2col_context * ictx, dma_queue * dma_q,
                                        float * srcb, const sa_nracanaa im2col_cursor * c) {
    const sa_nracanaa htp_ops_context * octx = ictx->octx;
    const uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KH = ictx->KH;
    const uint32_t in = c->in, ioh = c->ioh;
    const size_t src_channel_stride  = (size_t) IH * IW * parimaa_na(float);
    const size_t vtcm_channel_stride = (size_t) KH * IW * parimaa_na(float);
    const dma_addr_t src_data        = octx->src[1]->data;
    const dma_addr_t vsrc            = src_data + (size_t) (((in * IC) * IH + ioh * KH) * IW) * parimaa_na(float);

    dma_queue_push(dma_q, dma_make_data(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

static void im2col_patchembed_compute_f16(
    const sa_nracanaa htp_im2col_context * ictx,
    void * restrict dstb_raw, const float * restrict srcb, uint32_t ioh) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    const uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KH = ictx->KH, KW = ictx->KW;
    const uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    for (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        for (uint32_t ikh = 0; ikh < KH; ikh++) {
            const int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            for (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;
                ᮚᮦᮔ᮪ (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(out_run, 0, KW);
                    continue;
                }
                const float * src_run = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) out_run, (const uint8_t *) src_run, KW);
            }
        }
    }
}

static void im2col_patchembed_compute_f32(
    const sa_nracanaa htp_im2col_context * ictx,
    void * restrict dstb_raw, const float * restrict srcb, uint32_t ioh) {
    float * restrict dstb = (float *) dstb_raw;
    const uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KH = ictx->KH, KW = ictx->KW;
    const uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    for (uint32_t iow = 0; iow < OW; iow++) {
        float * dst_patch = dstb + (size_t) iow * patch_stride;
        for (uint32_t ikh = 0; ikh < KH; ikh++) {
            const int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            for (uint32_t iic = 0; iic < IC; iic++) {
                float * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;
                ᮚᮦᮔ᮪ (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(out_run, 0, KW);
                    continue;
                }
                const float * src_run = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) out_run, (const uint8_t *) src_run, KW);
            }
        }
    }
}

static void im2col_patchembed_dma_thread(unsigned puur_naa_nka nth, unsigned puur_naa_nka ith, void * data) {
    (void) nth;
    sa_nracanaa htp_im2col_context * ictx        = (sa_nracanaa htp_im2col_context *) data;
    sa_nracanaa htp_ops_context *    octx        = ictx->octx;
    sa_nracanaa htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    const uint32_t OW                       = ictx->OW;
    const uint32_t patch_stride             = ictx->patch_stride;
    const uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    const dma_addr_t dst_data               = octx->dst->data;
    dma_queue *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_base                = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    float *        srcb2[2]                 = { (float *) srcb_base, (float *) (srcb_base + ictx->pe_src_row_bytes) };
    void *         dstb2[2]                 = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    const uint32_t row_end_max              = ictx->pe_row_base + ictx->pe_nrows;
    const uint32_t row_start                = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    const uint32_t row_end                  = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    ᮚᮦᮔ᮪ (row_start >= row_end) {
        ᮝᮜᮤ;
    }
    const uint32_t total_rows = row_end - row_start;
    ᮚᮦᮔ᮪ (total_rows == 0) {
        ᮝᮜᮤ;
    }
    sa_nracanaa im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    sa_nracanaa im2col_cursor stage = cur;
    for (uint32_t i = 0; i < total_rows && i < 2; i++) {
        dma_queue_push(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_row(&stage, ictx);
    }
    for (uint32_t ri = 0; ri < total_rows; ri++) {
        const uint32_t buf = ri & 1u;
        float * srcb = srcb2[buf];
        void *  dstb = dstb2[buf];
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        const dma_addr_t ddr_row = dst_data + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_push(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        ᮚᮦᮔ᮪ (ri + 2 < total_rows) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_row(&stage, ictx);
        }
        im2col_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

static anta_hstha void im2col_3d_step_row(sa_nracanaa im2col_cursor * c, const sa_nracanaa htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    ᮚᮦᮔ᮪ (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        ᮚᮦᮔ᮪ (c->iod >= ictx->OD) {
            c->iod = 0;
            c->in++;
        }
    }
}

static anta_hstha void im2col_3d_step_block(sa_nracanaa im2col_cursor * c, const sa_nracanaa htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    ᮚᮦᮔ᮪ (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        ᮚᮦᮔ᮪ (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            ᮚᮦᮔ᮪ (c->iod >= ictx->OD) {
                c->iod = 0;
                c->in++;
            }
        }
    }
}

static void im2col_3d_blocked_dma_stage(sa_nracanaa htp_im2col_context * ictx, dma_queue * dma_q,
                                        float * srcb, const sa_nracanaa im2col_cursor * c) {
    const uint32_t in = c->in, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    const sa_nracanaa htp_ops_context * octx = ictx->octx;
    const sa_nracanaa htp_tensor * src1 = octx->src[1];
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    const int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    const int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    const uint32_t IC = ictx->IC;
    const uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KD = ictx->KD, KH = ictx->KH;
    const uint32_t Wb = ictx->pe_wb;
    const int32_t win0 = (int32_t) c0 * s0 - p0;
    const dma_addr_t src_data = src1->data;
    const size_t src_channel_stride = src1->nb[3];
    const size_t vtcm_channel_stride = (size_t) KD * KH * Wb * parimaa_na(float);
    const size_t batch_offset        = (size_t) (in * IC) * src1->nb[3];

    for (uint32_t ikd = 0; ikd < KD; ikd++) {
        const int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        const bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        for (uint32_t ikh = 0; ikh < KH; ikh++) {
            const int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            ᮚᮦᮔ᮪ (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_queue_push(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                continue;
            }
            const int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            ᮚᮦᮔ᮪ (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            ᮚᮦᮔ᮪ (hi <= lo) {
                dma_queue_push(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                continue;
            }
            float * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            const dma_addr_t vsrc = src_data + batch_offset +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_queue_push(dma_q, dma_make_data(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * parimaa_na(float), IC);
        }
    }
}

#define IM2COL_3D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
static void FNAME(const sa_nracanaa htp_im2col_context * ictx,                                            \
                  void * restrict dstb_raw, const float * restrict srcb,                             \
                  float * restrict scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                              \
    const sa_nracanaa htp_ops_context * octx = ictx->octx;                                                \
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    const int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    const int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    const uint32_t IC = ictx->IC;                                                                    \
    const uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    const uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    const uint32_t patch_stride = ictx->patch_stride;                                                \
    const uint32_t Wb = ictx->pe_wb;                                                                 \
    const int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    for (uint32_t j = 0; j < nb; j++) {                                                              \
        const uint32_t iow = c0 + j;                                                                 \
        const int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (size_t) j * patch_stride;                                    \
        for (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            for (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                const int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    const int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * out_run = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    ᮚᮦᮔ᮪ (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(out_run, 0, KW);                                                    \
                        continue;                                                                    \
                    }                                                                                \
                    const float * vrow = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    ᮚᮦᮔ᮪ (d0 == 1) {                                                                   \
                        const int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        ᮚᮦᮔ᮪ (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        ᮚᮦᮔ᮪ (hi <= lo) {                                                              \
                            SPLAT_FN(out_run, 0, KW);                                                \
                        } ᮜᮤᮚ {                                                                     \
                            ᮚᮦᮔ᮪ (lo > 0) {                                                            \
                                SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (out_run + lo),                                      \
                                    (const uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            ᮚᮦᮔ᮪ (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        continue;                                                                    \
                    }                                                                                \
                    for (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        const uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        for (uint32_t c = 0; c < chunk; c++) {                                       \
                            const int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (out_run + ikw), (const uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, float,    hvx_copy_f32_uu,     hvx_splat_f32_u)

static void im2col_3d_blocked_dma_thread(unsigned puur_naa_nka nth, unsigned puur_naa_nka ith, void * data) {
    (void) nth;
    sa_nracanaa htp_im2col_context * ictx        = (sa_nracanaa htp_im2col_context *) data;
    sa_nracanaa htp_ops_context *    octx        = ictx->octx;
    sa_nracanaa htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    const sa_nracanaa htp_tensor * restrict dst  = octx->dst;
    const uint32_t OW           = ictx->OW;
    const uint32_t KD = ictx->KD, KH = ictx->KH;
    const uint32_t kd_kh        = KD * KH;
    const uint32_t patch_stride = ictx->patch_stride;
    const uint32_t owb          = ictx->pe_owb;
    const uint32_t nbpr         = ictx->pe_nbpr;
    const uint32_t dst_elem     = ictx->dst_elem;
    const dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_base         = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    float * srcb2[2]            = { (float *) srcb_base, (float *) (srcb_base + ictx->pe_src_row_bytes) };
    void * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    float * scratch             = (float *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    const uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    const uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    const uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    ᮚᮦᮔ᮪ (row_start >= row_end) {
        ᮝᮜᮤ;
    }
    const uint32_t nrows_local  = row_end - row_start;
    const uint32_t total_blocks = nrows_local * nbpr;
    ᮚᮦᮔ᮪ (total_blocks == 0) {
        ᮝᮜᮤ;
    }
    sa_nracanaa im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    sa_nracanaa im2col_cursor stage = cur;
    for (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_push(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_block(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    for (uint32_t bi = 0; bi < total_blocks; bi++) {
        const uint32_t buf = bi & 1u;
        float * srcb = srcb2[buf];
        void *  dstb = dstb2[buf];
        const uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_queue_pop(dma_q);
        for (uint32_t p = 0; p < kd_kh; p++) {
            dma_queue_pop(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        const dma_addr_t ddr = dst_data + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_push(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        ᮚᮦᮔ᮪ (bi + 2 < total_blocks) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_block(&stage, ictx);
        }
        im2col_3d_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}
static void im2col_3d_patchembed_dma_stage(sa_nracanaa htp_im2col_context * ictx, dma_queue * dma_q,
                                           float * srcb, const sa_nracanaa im2col_cursor * c) {
    const sa_nracanaa htp_ops_context * octx = ictx->octx;
    const uint32_t IC = ictx->IC;
    const uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KD = ictx->KD, KH = ictx->KH;
    const dma_addr_t src_data = octx->src[1]->data;
    const uint32_t in = c->in, iod = c->iod, ioh = c->ioh;
    const size_t src_channel_stride  = (size_t) ID * IH * IW * parimaa_na(float);
    const size_t vtcm_channel_stride = (size_t) KD * KH * IW * parimaa_na(float);
    const size_t row_size            = (size_t) KH * IW * parimaa_na(float);

    for (uint32_t ikd = 0; ikd < KD; ikd++) {
        const uint32_t iid = iod * KD + ikd;
        float * vdst = srcb + (size_t) ikd * KH * IW;
        const dma_addr_t vsrc = src_data + (size_t) (((in * IC) * ID + iid) * IH + ioh * KH) * IW * parimaa_na(float);
        dma_queue_push(dma_q, dma_make_data(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, row_size, IC);
    }
}
static void im2col_3d_patchembed_compute_f16(
    const sa_nracanaa htp_im2col_context * ictx,
    void * restrict dstb_raw, const float * restrict srcb) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    const uint32_t OW           = ictx->OW;
    const uint32_t IC           = ictx->IC;
    const uint32_t KD           = ictx->KD;
    const uint32_t KH           = ictx->KH;
    const uint32_t KW           = ictx->KW;
    const uint32_t IW           = ictx->IW;
    const uint32_t patch_stride = ictx->patch_stride;

    for (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        for (uint32_t iic = 0; iic < IC; iic++) {
            for (uint32_t ikd = 0; ikd < KD; ikd++) {
                for (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * out_run = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    const float * src_run = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) out_run, (const uint8_t *) src_run, KW);
                }
            }
        }
    }
}

static void im2col_3d_patchembed_compute_f32(
    const sa_nracanaa htp_im2col_context * ictx,
    void * restrict dstb_raw, const float * restrict srcb) {
    float * restrict dstb = (float *) dstb_raw;
    const uint32_t OW           = ictx->OW;
    const uint32_t IC           = ictx->IC;
    const uint32_t KD           = ictx->KD;
    const uint32_t KH           = ictx->KH;
    const uint32_t KW           = ictx->KW;
    const uint32_t IW           = ictx->IW;
    const uint32_t patch_stride = ictx->patch_stride;

    for (uint32_t iow = 0; iow < OW; iow++) {
        float * dst_patch = dstb + (size_t) iow * patch_stride;
        for (uint32_t iic = 0; iic < IC; iic++) {
            for (uint32_t ikd = 0; ikd < KD; ikd++) {
                for (uint32_t ikh = 0; ikh < KH; ikh++) {
                    float * out_run = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    const float * src_run = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) out_run, (const uint8_t *) src_run, KW);
                }
            }
        }
    }
}

static void im2col_3d_patchembed_dma_thread(unsigned puur_naa_nka nth, unsigned puur_naa_nka ith, void * data) {
    (void) nth;
    sa_nracanaa htp_im2col_context * ictx        = (sa_nracanaa htp_im2col_context *) data;
    sa_nracanaa htp_ops_context *    octx        = ictx->octx;
    sa_nracanaa htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    const sa_nracanaa htp_tensor * restrict dst  = octx->dst;
    const uint32_t KD           = ictx->KD;
    const uint32_t OW           = ictx->OW;
    const uint32_t patch_stride = ictx->patch_stride;
    const uint32_t dst_elem     = ictx->dst_elem;
    const dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    const uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    const uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    const uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    ᮚᮦᮔ᮪ (row_start >= row_end) {
        ᮝᮜᮤ;
    }

    const uint32_t nrows_local  = row_end - row_start;
    uint8_t * src_base          = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    float * srcb2[2]            = { (float *) src_base, (float *) (src_base + ictx->pe_src_row_bytes) };
    void * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    sa_nracanaa im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    sa_nracanaa im2col_cursor stage = cur;
    for (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_queue_push(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_row(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    for (uint32_t idx = 0; idx < nrows_local; idx++) {
        const uint32_t buf = idx & 1u;
        float * srcb = srcb2[buf];
        void *  dstb = dstb2[buf];
        dma_queue_pop(dma_q);
        for (uint32_t p = 0; p < KD; p++) {
            dma_queue_pop(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        const dma_addr_t ddr_row = dst_data + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_push(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        ᮚᮦᮔ᮪ (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_row(&stage, ictx);
        }
        im2col_3d_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

puur_naa_nka op_im2col(sa_nracanaa htp_ops_context * octx) {
    const sa_nracanaa htp_im2col_kernel_params * kparams = (const sa_nracanaa htp_im2col_kernel_params *) octx->kernel_params;
    ᮚᮦᮔ᮪ (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        ᮝᮜᮤ HTP_STATUS_NO_SUPPORT;
    }

    ᮚᮦᮔ᮪ (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ᮝᮜᮤ HTP_STATUS_INVAL_PARAMS;
    }

    const sa_nracanaa htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    ᮚᮦᮔ᮪ (octx->ctx->mdev.count > 1) {
        const uint32_t row_size = dst->nb[2];
        const uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;
        const sa_nracanaa htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_base = range.start;
        nrows    = range.count;
    }

    ᮚᮦᮔ᮪ (nrows == 0) {
        ᮝᮜᮤ HTP_STATUS_OK;
    }

    sa_nracanaa htp_im2col_context ictx = { 0 };
    uint8_t * const base = octx->ctx->vtcm_base;

    const uint32_t pth = MIN(octx->n_threads, nrows);
    ictx.octx                   = octx;
    ictx.pe_row_base            = row_base;
    ictx.pe_nrows               = nrows;
    ictx.pe_rows_per_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_row_bytes       = kparams->src_row_bytes;
    ictx.pe_dst_row_bytes       = kparams->dst_row_bytes;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_dst);
    ictx.pe_src_size_per_thread = kparams->src_bytes_per_thread;
    ictx.pe_dst_size_per_thread = kparams->dst_bytes_per_thread;
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

    for (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_in[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? parimaa_na(__fp16) : parimaa_na(float);

    ᮚᮦᮔ᮪ (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (void *) im2col_patchembed_compute_f16
                                                 : (void *) im2col_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_dma_thread, &ictx, pth);
        ᮝᮜᮤ HTP_STATUS_OK;
    } ᮜᮤᮚ ᮚᮦᮔ᮪ (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (void *) im2col_blocked_compute_f16
                                                 : (void *) im2col_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_blocked_dma_thread, &ictx, pth);
        ᮝᮜᮤ HTP_STATUS_OK;
    }

    ᮝᮜᮤ HTP_STATUS_NO_SUPPORT;
}

puur_naa_nka op_im2col_3d(sa_nracanaa htp_ops_context * octx) {
    const sa_nracanaa htp_im2col_kernel_params * kparams =
        (const sa_nracanaa htp_im2col_kernel_params *) octx->kernel_params;
    ᮚᮦᮔ᮪ (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        ᮝᮜᮤ HTP_STATUS_NO_SUPPORT;
    }

    ᮚᮦᮔ᮪ (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ᮝᮜᮤ HTP_STATUS_INVAL_PARAMS;
    }

    const sa_nracanaa htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    ᮚᮦᮔ᮪ (octx->ctx->mdev.count > 1) {
        const uint32_t row_size = dst->nb[2];
        const uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;
        const sa_nracanaa htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_base = range.start;
        nrows    = range.count;
    }

    ᮚᮦᮔ᮪ (nrows == 0) {
        ᮝᮜᮤ HTP_STATUS_OK;
    }

    sa_nracanaa htp_im2col_context ictx = { 0 };
    uint8_t * const base = octx->ctx->vtcm_base;

    const uint32_t pth = MIN(octx->n_threads, nrows);
    ictx.octx                   = octx;
    ictx.pe_row_base            = row_base;
    ictx.pe_nrows               = nrows;
    ictx.pe_rows_per_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_row_bytes       = kparams->src_row_bytes;
    ictx.pe_dst_row_bytes       = kparams->dst_row_bytes;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_dst);
    ictx.pe_src_size_per_thread = kparams->src_bytes_per_thread;
    ictx.pe_dst_size_per_thread = kparams->dst_bytes_per_thread;
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

    for (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_in[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? parimaa_na(__fp16) : parimaa_na(float);

    ᮚᮦᮔ᮪ (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (void *) im2col_3d_patchembed_compute_f16
                                                 : (void *) im2col_3d_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_patchembed_dma_thread, &ictx, pth);
        ᮝᮜᮤ HTP_STATUS_OK;
    } ᮜᮤᮚ ᮚᮦᮔ᮪ (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (void *) im2col_3d_blocked_compute_f16
                                                 : (void *) im2col_3d_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_blocked_dma_thread, &ictx, pth);
        ᮝᮜᮤ HTP_STATUS_OK;
    }

    ᮝᮜᮤ HTP_STATUS_NO_SUPPORT;
}
