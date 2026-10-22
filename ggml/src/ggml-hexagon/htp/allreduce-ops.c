#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#અન્તર્ભાવ <HAP_farf.h>
#અન્તર્ભાવ <HAP_perf.h>
#અન્તર્ભાવ <stdatomic.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <string.h>

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "hvx-utils.h"
#અન્તર્ભાવ "htp-tensor.h"
#અન્તર્ભાવ "dma-queue.h"
#અન્તર્ભાવ "hex-profile.h"
#અન્તર્ભાવ "allreduce-ops.h"
#અન્તર્ભાવ "htp-fence.h"

સંરચના htp_allreduce_context {
    સંરચના htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t n_dsts;
    uint32_t nelem;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t row_size_aligned;
    uint32_t rank_elem_start;
    uint32_t rank_nelem;
    uint32_t elems_per_thread;
    uint32_t block_elems;
    uint32_t vtcm_size_per_thread;
    bool     is_row_bcast;
    uint8_t * src_spad_base[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_base;
    uint8_t * res_spad_base;
};

#define DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD)                                         \
સ્થિર શૂન્ય allreduce_thread_dma_1d_##SUFFIX(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {                   \
    સંરચના htp_allreduce_context * actx = (સંરચના htp_allreduce_context *) data;                                  \
    સંરચના htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    અચલ uint32_t n_ranks     = actx->n_ranks;                                                                   \
    અચલ uint32_t n_dsts      = actx->n_dsts;                                                                    \
    અચલ uint32_t block_elems = actx->block_elems;                                                               \
                                                                                                                  \
    અચલ uint32_t dr  = actx->elems_per_thread;                                                                  \
    અચલ uint32_t ir0 = actx->rank_elem_start + dr * ith;                                                        \
    અચલ uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_start + actx->rank_nelem);                                 \
    જો (ir0 >= ir1) પાછા;                                                                                       \
                                                                                                                  \
    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    અચલ size_t vtcm_thread_offset = ith * actx->vtcm_size_per_thread;                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + vtcm_thread_offset;                                           \
    uint8_t * res_spad_base = HAS_ADD ? (actx->res_spad_base + vtcm_thread_offset) : ખાલી;                        \
                                                                                                                  \
    અચલ size_t spad_half = actx->vtcm_size_per_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    પૂર્ણાંક spad_idx = 0;                                                                                             \
                                                                                                                  \
    માટે (પૂર્ણાંક k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_bytes = cur_elems * કદ(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                  \
        માટે (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + ir_prefetch * કદ(TYPE);                                  \
            dma_queue_ધકેલો(dma_q, dma_make_data(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 0);              \
        }                                                                                                         \
        માટે (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + spad_idx * spad_half;                \
            અચલ dma_addr_t s_ddr = octx->src[s]->data + ir_prefetch * કદ(TYPE);                             \
            dma_queue_ધકેલો(dma_q, dma_make_data(s_spad, s_ddr), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        જો (HAS_ADD) {                                                                                            \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                              \
            અચલ dma_addr_t r_ddr = octx->src[2 * n_ranks]->data + ir_prefetch * કદ(TYPE);                   \
            dma_queue_ધકેલો(dma_q, dma_make_data(r_spad, r_ddr), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    પૂર્ણાંક comp_spad_idx = 0;                                                                                        \
    માટે (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir);                                                          \
        size_t   cur_bytes = cur_elems * કદ(TYPE);                                                            \
        માટે (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_queue_કાઢો(dma_q);                                                                                 \
        }                                                                                                         \
        માટે (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_queue_કાઢો(dma_q);                                                                                 \
        }                                                                                                         \
        જો (HAS_ADD) {                                                                                            \
            dma_queue_કાઢો(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_base + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_base[0] + vtcm_thread_offset + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_base[1] + vtcm_thread_offset + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_ADD ? (res_spad_base + comp_spad_idx * spad_half) : ખાલી;                         \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_ADD_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        માટે (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;          \
            HVX_ADD_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        જો (HAS_ADD) {                                                                                            \
            HVX_ADD_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        માટે (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + ir * કદ(TYPE);                                           \
            dma_queue_ધકેલો(dma_q, dma_make_data(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        જો (ir_prefetch < ir1) {                                                                                  \
            uint32_t next_elems = MIN(block_elems, ir1 - ir_prefetch);                                            \
            size_t   next_bytes = next_elems * કદ(TYPE);                                                      \
            માટે (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;       \
                અચલ dma_addr_t s_next = octx->src[s]->data + ir_prefetch * કદ(TYPE);                        \
                dma_queue_ધકેલો(dma_q, dma_make_data(s_spad, s_next), next_bytes, next_bytes, next_bytes, 1);      \
            }                                                                                                     \
            જો (HAS_ADD) {                                                                                        \
                uint8_t * r_spad_next = res_spad_base + comp_spad_idx * spad_half;                                \
                અચલ dma_addr_t r_next = octx->src[2 * n_ranks]->data + ir_prefetch * કદ(TYPE);              \
                dma_queue_ધકેલો(dma_q, dma_make_data(r_spad_next, r_next), next_bytes, next_bytes, next_bytes, 1); \
            }                                                                                                     \
            ir_prefetch += next_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_queue_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_add_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     દશાંશ,  hvx_add_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f16, __fp16, hvx_add_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f32, દશાંશ,  hvx_add_f32_aaa, 1)

#define DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD, IS_ROW_BCAST)                                                           \
સ્થિર શૂન્ય allreduce_thread_dma_2d_##SUFFIX(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {                                                   \
    સંરચના htp_allreduce_context * actx = (સંરચના htp_allreduce_context *) data;                                                                  \
    સંરચના htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    અચલ uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    અચલ uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    અચલ uint32_t ne0              = actx->ne0;                                                                                                  \
    અચલ uint32_t block_rows       = actx->block_elems;                                                                                          \
    અચલ uint32_t row_size_aligned = actx->row_size_aligned;                                                                                     \
    અચલ uint32_t row_bytes        = ne0 * કદ(TYPE);                                                                                         \
                                                                                                                                                  \
    અચલ uint32_t dr  = actx->elems_per_thread;                                                                                                  \
    અચલ uint32_t r0  = actx->rank_elem_start + dr * ith;                                                                                        \
    અચલ uint32_t r1  = MIN(r0 + dr, actx->rank_elem_start + actx->rank_nelem);                                                                  \
    જો (r0 >= r1) પાછા;                                                                                                                         \
                                                                                                                                                  \
    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    અચલ size_t vtcm_thread_offset = ith * actx->vtcm_size_per_thread;                                                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + vtcm_thread_offset;                                                                           \
    uint8_t * res_spad_base = HAS_ADD ? (IS_ROW_BCAST ? actx->res_spad_base : (actx->res_spad_base + vtcm_thread_offset)) : ખાલી;                 \
                                                                                                                                                  \
    અચલ size_t spad_half = actx->vtcm_size_per_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    પૂર્ણાંક spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    માટે (પૂર્ણાંક k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_rows = MIN(block_rows, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                                                  \
        માટે (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_queue_ધકેલો(dma_q, dma_make_data(d_ddr, d_spad), octx->dsts[d]->nb[1], row_size_aligned, row_bytes, 0);                            \
        }                                                                                                                                         \
        માટે (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + spad_idx * spad_half;                                                \
            અચલ dma_addr_t s_ddr = octx->src[s]->data + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_queue_ધકેલો(dma_q, dma_make_data(s_spad, s_ddr), row_size_aligned, octx->src[s]->nb[1], row_bytes, cur_rows);                      \
        }                                                                                                                                         \
        જો (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                                                              \
            અચલ dma_addr_t r_ddr = octx->src[2 * n_ranks]->data + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_queue_ધકેલો(dma_q, dma_make_data(r_spad, r_ddr), row_size_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, cur_rows);            \
        }                                                                                                                                         \
        r_prefetch += cur_rows;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    પૂર્ણાંક comp_spad_idx = 0;                                                                                                                        \
    માટે (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_rows = MIN(block_rows, r1 - r);                                                                                              \
        માટે (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_queue_કાઢો(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        માટે (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_queue_કાઢો(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        જો (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            dma_queue_કાઢો(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_base + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_base[0] + vtcm_thread_offset + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_base[1] + vtcm_thread_offset + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_ADD && !IS_ROW_BCAST) ? (res_spad_base + comp_spad_idx * spad_half) : ખાલી;                                      \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        માટે (uint32_t row = 0; row < cur_rows; row++) {                                                                                           \
            uint8_t * d_row = d_spad + row * row_size_aligned;                                                                                    \
            અચલ uint8_t * s0_row = s0_spad + row * row_size_aligned;                                                                            \
            અચલ uint8_t * s1_row = s1_spad + row * row_size_aligned;                                                                            \
            HVX_ADD_FN(d_row, s0_row, s1_row, ne0);                                                                                               \
            માટે (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                અચલ uint8_t * ss_row = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half + row * row_size_aligned;        \
                HVX_ADD_FN(d_row, d_row, ss_row, ne0);                                                                                            \
            }                                                                                                                                     \
            જો (HAS_ADD) {                                                                                                                        \
                અચલ uint8_t * res_row = IS_ROW_BCAST ? res_spad_base : (r_spad + row * row_size_aligned);                                       \
                HVX_ADD_FN(d_row, d_row, res_row, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        માટે (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + r * octx->dsts[d]->nb[1];                                                                    \
            dma_queue_ધકેલો(dma_q, dma_make_data(d_ddr, d_spad), octx->dsts[d]->nb[1], row_size_aligned, row_bytes, cur_rows);                     \
        }                                                                                                                                         \
        જો (r_prefetch < r1) {                                                                                                                    \
            uint32_t next_rows = MIN(block_rows, r1 - r_prefetch);                                                                                \
            માટે (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;                                       \
                અચલ dma_addr_t s_next = octx->src[s]->data + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_queue_ધકેલો(dma_q, dma_make_data(s_spad, s_next), row_size_aligned, octx->src[s]->nb[1], row_bytes, next_rows);                \
            }                                                                                                                                     \
            જો (HAS_ADD && !IS_ROW_BCAST) {                                                                                                       \
                uint8_t * r_spad_next = res_spad_base + comp_spad_idx * spad_half;                                                                \
                અચલ dma_addr_t r_next = octx->src[2 * n_ranks]->data + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_queue_ધકેલો(dma_q, dma_make_data(r_spad_next, r_next), row_size_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, next_rows); \
            }                                                                                                                                     \
            r_prefetch += next_rows;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_rows;                                                                                                                            \
    }                                                                                                                                             \
    dma_queue_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_add_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           દશાંશ,  hvx_add_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f16,       __fp16, hvx_add_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f32,       દશાંશ,  hvx_add_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f16, __fp16, hvx_add_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f32, દશાંશ,  hvx_add_f32_aaa, 1, 1)

સ્થિર પૂર્ણાંક validate_allreduce(
    સંરચના htp_ops_context * octx,
    અચલ સંરચના htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    જો (kparams->n_threads == 0 || (uint32_t) kparams->n_threads > octx->ctx->n_threads) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }
    octx->n_threads = (uint32_t) kparams->n_threads;

    જો (kparams->vtcm_size_per_thread <= 0 || kparams->vtcm_size <= 0) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }

    અચલ bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    અચલ size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_ગણતરી(
        n_ranks, octx->n_threads, has_add, kparams->is_row_bcast != 0);
    અચલ size_t vtcm_size = n_vtcm_buffers * (size_t) kparams->vtcm_size_per_thread;
    જો (vtcm_size != (size_t) kparams->vtcm_size) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }
    જો (vtcm_size > octx->ctx->vtcm_size) {
        પાછા HTP_STATUS_VTCM_TOO_SMALL;
    }

    જો (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    પાછા HTP_STATUS_OK;
}

પૂર્ણાંક op_allreduce(સંરચના htp_ops_context * octx) {
    જો (octx->ctx->mdev.ગણતરી > 1 && octx->ctx->mdev.idx > 0) {
        પાછા HTP_STATUS_OK;
    }

    અચલ સંરચના htp_allreduce_kernel_params * kparams = (અચલ સંરચના htp_allreduce_kernel_params *) octx->kernel_params;
    અચલ સંરચના htp_tensor * dst = octx->dst;

    અચલ uint32_t rank    = (uint32_t) kparams->rank;
    અચલ uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    જો (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }

    અચલ uint32_t fence_seq_entry = (uint32_t) octx->op_params[0];
    અચલ uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    અચલ સંરચના htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->data;

    અચલ પૂર્ણાંક status = validate_allreduce(octx, kparams, n_ranks);
    જો (status != HTP_STATUS_OK) {
        જો (status == HTP_STATUS_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, status);
        પાછા status;
    }

    અચલ bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    અચલ uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    અચલ int32_t  mode  = kparams->mode;

    // 1. Entry Barrier: Synchronize all ranks before reading
    સંરચના htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    htp_fence_write(my_fence, fence_seq_entry, octx->status);

    માટે (uint32_t j = 0; j < n_ranks; j++) {
        જો (j == rank) ચાલુ;
        અચલ સંરચના htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        જ્યારે (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            જો ((int32_t)(peer_seq - fence_seq_entry) >= 0) {
                જો (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                    પાછા peer_status;
                }
                તોડો;
            }
            જો (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_entry, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                પાછા HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm અસ્થિર ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    // 2. Multi-threaded Reduction across assigned rank chunk
    જો (nelem > 0) {
        અચલ uint32_t n_threads            = (uint32_t) kparams->n_threads;
        અચલ uint32_t block_elems          = (uint32_t) kparams->block_elems;
        અચલ uint32_t elems_per_thread     = (uint32_t) kparams->elems_per_thread;
        અચલ uint32_t vtcm_size_per_thread = (uint32_t) kparams->vtcm_size_per_thread;

        સંરચના htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.row_size_aligned     = (uint32_t) kparams->row_size_aligned;
        actx.rank_elem_start      = (uint32_t) kparams->rank_elem_start;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_per_thread     = elems_per_thread;
        actx.block_elems          = block_elems;
        actx.vtcm_size_per_thread = vtcm_size_per_thread;
        actx.is_row_bcast         = (kparams->is_row_bcast != 0);

        work_queue_func_t reduce_fun = ખાલી;
        પસંદગી (kparams->kernel_type) {
            સ્થિતિ HTP_ALLREDUCE_KERNEL_DMA_1D:
                જો (has_add) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_add_f16 : allreduce_thread_dma_1d_add_f32;
                } નહીંતર {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                તોડો;
            સ્થિતિ HTP_ALLREDUCE_KERNEL_DMA_2D:
                જો (has_add) {
                    જો (kparams->is_row_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_bcast_f16 : allreduce_thread_dma_2d_add_bcast_f32;
                    } નહીંતર {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_f16 : allreduce_thread_dma_2d_add_f32;
                    }
                } નહીંતર {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                તોડો;
            મૂળભૂત:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_NO_SUPPORT);
                પાછા HTP_STATUS_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
        માટે (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_base[s] = vtcm_ptr;
            vtcm_ptr += n_threads * vtcm_size_per_thread;
        }
        actx.dst_spad_base = vtcm_ptr;
        vtcm_ptr += n_threads * vtcm_size_per_thread;
        જો (has_add) {
            actx.res_spad_base = vtcm_ptr;
            vtcm_ptr += (actx.is_row_bcast ? 1 : n_threads) * vtcm_size_per_thread;
        }

        જો (has_add && actx.is_row_bcast) {
            અચલ dma_addr_t r_ddr = octx->src[2 * n_ranks]->data;
            અચલ uint32_t row_bytes = actx.ne0 * (dst->type == HTP_TYPE_F16 ? કદ(__fp16) : કદ(દશાંશ));
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_ધકેલો(dma_q, dma_make_data(actx.res_spad_base, r_ddr), actx.row_size_aligned, 0, row_bytes, 1);
            dma_queue_કાઢો(dma_q);
        }

        work_queue_run(octx->ctx->work_queue, reduce_fun, &actx, n_threads);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    જો (mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
        asm અસ્થિર ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->status);

    માટે (uint32_t j = 0; j < n_ranks; j++) {
        જો (j == rank) ચાલુ;
        અચલ સંરચના htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        જ્યારે (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            જો ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                જો (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    પાછા peer_status;
                }
                તોડો;
            }
            જો (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                પાછા HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm અસ્થિર ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    પાછા octx->status;
}
