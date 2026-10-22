#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>
#অন্তর্ভাব <stdatomic.h>
#অন্তর্ভাব <math.h>
#অন্তর্ভাব <string.h>

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "hvx-utils.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "allreduce-ops.h"
#অন্তর্ভাব "htp-fence.h"

কাঠামো htp_allreduce_context {
    কাঠামো htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t n_dsts;
    uint32_t nelem;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t row_আকার_aligned;
    uint32_t rank_elem_start;
    uint32_t rank_nelem;
    uint32_t elems_per_thread;
    uint32_t block_elems;
    uint32_t vtcm_আকার_per_thread;
    bool     is_row_bcast;
    uint8_t * src_spad_base[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_base;
    uint8_t * res_spad_base;
};

#define DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD)                                         \
স্থির শূন্য allreduce_thread_dma_1d_##SUFFIX(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {                   \
    কাঠামো htp_allreduce_context * actx = (কাঠামো htp_allreduce_context *) তথ্য;                                  \
    কাঠামো htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    ধ্রুবক uint32_t n_ranks     = actx->n_ranks;                                                                   \
    ধ্রুবক uint32_t n_dsts      = actx->n_dsts;                                                                    \
    ধ্রুবক uint32_t block_elems = actx->block_elems;                                                               \
                                                                                                                  \
    ধ্রুবক uint32_t dr  = actx->elems_per_thread;                                                                  \
    ধ্রুবক uint32_t ir0 = actx->rank_elem_start + dr * ith;                                                        \
    ধ্রুবক uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_start + actx->rank_nelem);                                 \
    যদি (ir0 >= ir1) ফেরত;                                                                                       \
                                                                                                                  \
    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    ধ্রুবক size_t vtcm_thread_offset = ith * actx->vtcm_আকার_per_thread;                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + vtcm_thread_offset;                                           \
    uint8_t * res_spad_base = HAS_ADD ? (actx->res_spad_base + vtcm_thread_offset) : ফাঁকা;                        \
                                                                                                                  \
    ধ্রুবক size_t spad_half = actx->vtcm_আকার_per_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    পূর্ণসংখ্যা spad_idx = 0;                                                                                             \
                                                                                                                  \
    জন্য (পূর্ণসংখ্যা k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_bytes = cur_elems * আকার(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                  \
        জন্য (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->তথ্য + ir_prefetch * আকার(TYPE);                                  \
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 0);              \
        }                                                                                                         \
        জন্য (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + spad_idx * spad_half;                \
            ধ্রুবক dma_addr_t s_ddr = octx->src[s]->তথ্য + ir_prefetch * আকার(TYPE);                             \
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s_spad, s_ddr), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        যদি (HAS_ADD) {                                                                                            \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                              \
            ধ্রুবক dma_addr_t r_ddr = octx->src[2 * n_ranks]->তথ্য + ir_prefetch * আকার(TYPE);                   \
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(r_spad, r_ddr), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    পূর্ণসংখ্যা comp_spad_idx = 0;                                                                                        \
    জন্য (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir);                                                          \
        size_t   cur_bytes = cur_elems * আকার(TYPE);                                                            \
        জন্য (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_queue_বের_করো(dma_q);                                                                                 \
        }                                                                                                         \
        জন্য (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_queue_বের_করো(dma_q);                                                                                 \
        }                                                                                                         \
        যদি (HAS_ADD) {                                                                                            \
            dma_queue_বের_করো(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_base + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_base[0] + vtcm_thread_offset + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_base[1] + vtcm_thread_offset + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_ADD ? (res_spad_base + comp_spad_idx * spad_half) : ফাঁকা;                         \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_ADD_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        জন্য (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;          \
            HVX_ADD_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        যদি (HAS_ADD) {                                                                                            \
            HVX_ADD_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        জন্য (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->তথ্য + ir * আকার(TYPE);                                           \
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        যদি (ir_prefetch < ir1) {                                                                                  \
            uint32_t পরবর্তী_elems = MIN(block_elems, ir1 - ir_prefetch);                                            \
            size_t   পরবর্তী_bytes = পরবর্তী_elems * আকার(TYPE);                                                      \
            জন্য (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;       \
                ধ্রুবক dma_addr_t s_পরবর্তী = octx->src[s]->তথ্য + ir_prefetch * আকার(TYPE);                        \
                dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s_spad, s_পরবর্তী), পরবর্তী_bytes, পরবর্তী_bytes, পরবর্তী_bytes, 1);      \
            }                                                                                                     \
            যদি (HAS_ADD) {                                                                                        \
                uint8_t * r_spad_পরবর্তী = res_spad_base + comp_spad_idx * spad_half;                                \
                ধ্রুবক dma_addr_t r_পরবর্তী = octx->src[2 * n_ranks]->তথ্য + ir_prefetch * আকার(TYPE);              \
                dma_queue_ঠেলো(dma_q, dma_make_তথ্য(r_spad_পরবর্তী, r_পরবর্তী), পরবর্তী_bytes, পরবর্তী_bytes, পরবর্তী_bytes, 1); \
            }                                                                                                     \
            ir_prefetch += পরবর্তী_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_queue_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_add_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     দশমিক,  hvx_add_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f16, __fp16, hvx_add_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f32, দশমিক,  hvx_add_f32_aaa, 1)

#define DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD, IS_ROW_BCAST)                                                           \
স্থির শূন্য allreduce_thread_dma_2d_##SUFFIX(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {                                                   \
    কাঠামো htp_allreduce_context * actx = (কাঠামো htp_allreduce_context *) তথ্য;                                                                  \
    কাঠামো htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    ধ্রুবক uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    ধ্রুবক uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    ধ্রুবক uint32_t ne0              = actx->ne0;                                                                                                  \
    ধ্রুবক uint32_t block_rows       = actx->block_elems;                                                                                          \
    ধ্রুবক uint32_t row_আকার_aligned = actx->row_আকার_aligned;                                                                                     \
    ধ্রুবক uint32_t row_bytes        = ne0 * আকার(TYPE);                                                                                         \
                                                                                                                                                  \
    ধ্রুবক uint32_t dr  = actx->elems_per_thread;                                                                                                  \
    ধ্রুবক uint32_t r0  = actx->rank_elem_start + dr * ith;                                                                                        \
    ধ্রুবক uint32_t r1  = MIN(r0 + dr, actx->rank_elem_start + actx->rank_nelem);                                                                  \
    যদি (r0 >= r1) ফেরত;                                                                                                                         \
                                                                                                                                                  \
    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    ধ্রুবক size_t vtcm_thread_offset = ith * actx->vtcm_আকার_per_thread;                                                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + vtcm_thread_offset;                                                                           \
    uint8_t * res_spad_base = HAS_ADD ? (IS_ROW_BCAST ? actx->res_spad_base : (actx->res_spad_base + vtcm_thread_offset)) : ফাঁকা;                 \
                                                                                                                                                  \
    ধ্রুবক size_t spad_half = actx->vtcm_আকার_per_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    পূর্ণসংখ্যা spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    জন্য (পূর্ণসংখ্যা k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_rows = MIN(block_rows, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                                                  \
        জন্য (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->তথ্য + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(d_ddr, d_spad), octx->dsts[d]->nb[1], row_আকার_aligned, row_bytes, 0);                            \
        }                                                                                                                                         \
        জন্য (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + spad_idx * spad_half;                                                \
            ধ্রুবক dma_addr_t s_ddr = octx->src[s]->তথ্য + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s_spad, s_ddr), row_আকার_aligned, octx->src[s]->nb[1], row_bytes, cur_rows);                      \
        }                                                                                                                                         \
        যদি (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                                                              \
            ধ্রুবক dma_addr_t r_ddr = octx->src[2 * n_ranks]->তথ্য + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(r_spad, r_ddr), row_আকার_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, cur_rows);            \
        }                                                                                                                                         \
        r_prefetch += cur_rows;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    পূর্ণসংখ্যা comp_spad_idx = 0;                                                                                                                        \
    জন্য (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_rows = MIN(block_rows, r1 - r);                                                                                              \
        জন্য (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_queue_বের_করো(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        জন্য (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_queue_বের_করো(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        যদি (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            dma_queue_বের_করো(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_base + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_base[0] + vtcm_thread_offset + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_base[1] + vtcm_thread_offset + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_ADD && !IS_ROW_BCAST) ? (res_spad_base + comp_spad_idx * spad_half) : ফাঁকা;                                      \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        জন্য (uint32_t row = 0; row < cur_rows; row++) {                                                                                           \
            uint8_t * d_row = d_spad + row * row_আকার_aligned;                                                                                    \
            ধ্রুবক uint8_t * s0_row = s0_spad + row * row_আকার_aligned;                                                                            \
            ধ্রুবক uint8_t * s1_row = s1_spad + row * row_আকার_aligned;                                                                            \
            HVX_ADD_FN(d_row, s0_row, s1_row, ne0);                                                                                               \
            জন্য (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                ধ্রুবক uint8_t * ss_row = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half + row * row_আকার_aligned;        \
                HVX_ADD_FN(d_row, d_row, ss_row, ne0);                                                                                            \
            }                                                                                                                                     \
            যদি (HAS_ADD) {                                                                                                                        \
                ধ্রুবক uint8_t * res_row = IS_ROW_BCAST ? res_spad_base : (r_spad + row * row_আকার_aligned);                                       \
                HVX_ADD_FN(d_row, d_row, res_row, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        জন্য (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->তথ্য + r * octx->dsts[d]->nb[1];                                                                    \
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(d_ddr, d_spad), octx->dsts[d]->nb[1], row_আকার_aligned, row_bytes, cur_rows);                     \
        }                                                                                                                                         \
        যদি (r_prefetch < r1) {                                                                                                                    \
            uint32_t পরবর্তী_rows = MIN(block_rows, r1 - r_prefetch);                                                                                \
            জন্য (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;                                       \
                ধ্রুবক dma_addr_t s_পরবর্তী = octx->src[s]->তথ্য + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s_spad, s_পরবর্তী), row_আকার_aligned, octx->src[s]->nb[1], row_bytes, পরবর্তী_rows);                \
            }                                                                                                                                     \
            যদি (HAS_ADD && !IS_ROW_BCAST) {                                                                                                       \
                uint8_t * r_spad_পরবর্তী = res_spad_base + comp_spad_idx * spad_half;                                                                \
                ধ্রুবক dma_addr_t r_পরবর্তী = octx->src[2 * n_ranks]->তথ্য + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_queue_ঠেলো(dma_q, dma_make_তথ্য(r_spad_পরবর্তী, r_পরবর্তী), row_আকার_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, পরবর্তী_rows); \
            }                                                                                                                                     \
            r_prefetch += পরবর্তী_rows;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_rows;                                                                                                                            \
    }                                                                                                                                             \
    dma_queue_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_add_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           দশমিক,  hvx_add_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f16,       __fp16, hvx_add_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f32,       দশমিক,  hvx_add_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f16, __fp16, hvx_add_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f32, দশমিক,  hvx_add_f32_aaa, 1, 1)

স্থির পূর্ণসংখ্যা validate_allreduce(
    কাঠামো htp_ops_context * octx,
    ধ্রুবক কাঠামো htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    যদি (kparams->n_threads == 0 || (uint32_t) kparams->n_threads > octx->ctx->n_threads) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }
    octx->n_threads = (uint32_t) kparams->n_threads;

    যদি (kparams->vtcm_আকার_per_thread <= 0 || kparams->vtcm_আকার <= 0) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুবক bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    ধ্রুবক size_t n_vtcm_buffers = htp_allreduce_vtcm_বাফার_গুনতি(
        n_ranks, octx->n_threads, has_add, kparams->is_row_bcast != 0);
    ধ্রুবক size_t vtcm_আকার = n_vtcm_buffers * (size_t) kparams->vtcm_আকার_per_thread;
    যদি (vtcm_আকার != (size_t) kparams->vtcm_আকার) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }
    যদি (vtcm_আকার > octx->ctx->vtcm_আকার) {
        ফেরত HTP_STATUS_VTCM_TOO_SMALL;
    }

    যদি (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ফেরত HTP_STATUS_OK;
}

পূর্ণসংখ্যা op_allreduce(কাঠামো htp_ops_context * octx) {
    যদি (octx->ctx->mdev.গুনতি > 1 && octx->ctx->mdev.idx > 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক কাঠামো htp_allreduce_kernel_params * kparams = (ধ্রুবক কাঠামো htp_allreduce_kernel_params *) octx->kernel_params;
    ধ্রুবক কাঠামো htp_tensor * dst = octx->dst;

    ধ্রুবক uint32_t rank    = (uint32_t) kparams->rank;
    ধ্রুবক uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    যদি (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুবক uint32_t fence_seq_entry = (uint32_t) octx->op_params[0];
    ধ্রুবক uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    ধ্রুবক কাঠামো htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->তথ্য;

    ধ্রুবক পূর্ণসংখ্যা status = validate_allreduce(octx, kparams, n_ranks);
    যদি (status != HTP_STATUS_OK) {
        যদি (status == HTP_STATUS_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, status);
        ফেরত status;
    }

    ধ্রুবক bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    ধ্রুবক uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ধ্রুবক int32_t  mode  = kparams->mode;

    // 1. Entry Barrier: Synchronize all ranks before reading
    কাঠামো htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    htp_fence_write(my_fence, fence_seq_entry, octx->status);

    জন্য (uint32_t j = 0; j < n_ranks; j++) {
        যদি (j == rank) চালু;
        ধ্রুবক কাঠামো htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->তথ্য;
        uint64_t spins = 0;
        যখন (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            যদি ((int32_t)(peer_seq - fence_seq_entry) >= 0) {
                যদি (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                    ফেরত peer_status;
                }
                বিরতি;
            }
            যদি (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_entry, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                ফেরত HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm অস্থির ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    // 2. Multi-threaded Reduction across assigned rank chunk
    যদি (nelem > 0) {
        ধ্রুবক uint32_t n_threads            = (uint32_t) kparams->n_threads;
        ধ্রুবক uint32_t block_elems          = (uint32_t) kparams->block_elems;
        ধ্রুবক uint32_t elems_per_thread     = (uint32_t) kparams->elems_per_thread;
        ধ্রুবক uint32_t vtcm_আকার_per_thread = (uint32_t) kparams->vtcm_আকার_per_thread;

        কাঠামো htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.row_আকার_aligned     = (uint32_t) kparams->row_আকার_aligned;
        actx.rank_elem_start      = (uint32_t) kparams->rank_elem_start;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_per_thread     = elems_per_thread;
        actx.block_elems          = block_elems;
        actx.vtcm_আকার_per_thread = vtcm_আকার_per_thread;
        actx.is_row_bcast         = (kparams->is_row_bcast != 0);

        work_queue_func_t reduce_fun = ফাঁকা;
        নির্বাচন (kparams->kernel_type) {
            ক্ষেত্র HTP_ALLREDUCE_KERNEL_DMA_1D:
                যদি (has_add) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_add_f16 : allreduce_thread_dma_1d_add_f32;
                } অন্যথা {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                বিরতি;
            ক্ষেত্র HTP_ALLREDUCE_KERNEL_DMA_2D:
                যদি (has_add) {
                    যদি (kparams->is_row_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_bcast_f16 : allreduce_thread_dma_2d_add_bcast_f32;
                    } অন্যথা {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_f16 : allreduce_thread_dma_2d_add_f32;
                    }
                } অন্যথা {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                বিরতি;
            পূর্বনির্ধারিত:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_NO_SUPPORT);
                ফেরত HTP_STATUS_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
        জন্য (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_base[s] = vtcm_ptr;
            vtcm_ptr += n_threads * vtcm_আকার_per_thread;
        }
        actx.dst_spad_base = vtcm_ptr;
        vtcm_ptr += n_threads * vtcm_আকার_per_thread;
        যদি (has_add) {
            actx.res_spad_base = vtcm_ptr;
            vtcm_ptr += (actx.is_row_bcast ? 1 : n_threads) * vtcm_আকার_per_thread;
        }

        যদি (has_add && actx.is_row_bcast) {
            ধ্রুবক dma_addr_t r_ddr = octx->src[2 * n_ranks]->তথ্য;
            ধ্রুবক uint32_t row_bytes = actx.ne0 * (dst->type == HTP_TYPE_F16 ? আকার(__fp16) : আকার(দশমিক));
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(actx.res_spad_base, r_ddr), actx.row_আকার_aligned, 0, row_bytes, 1);
            dma_queue_বের_করো(dma_q);
        }

        work_queue_run(octx->ctx->work_queue, reduce_fun, &actx, n_threads);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    যদি (mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
        asm অস্থির ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->status);

    জন্য (uint32_t j = 0; j < n_ranks; j++) {
        যদি (j == rank) চালু;
        ধ্রুবক কাঠামো htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->তথ্য;
        uint64_t spins = 0;
        যখন (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            যদি ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                যদি (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    ফেরত peer_status;
                }
                বিরতি;
            }
            যদি (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                ফেরত HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm অস্থির ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    ফেরত octx->status;
}
