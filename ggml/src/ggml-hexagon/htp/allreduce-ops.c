#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <HAP_perf.h>
#അന്തര്ഭാവ <stdatomic.h>
#അന്തര്ഭാവ <math.h>
#അന്തര്ഭാവ <string.h>

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "hvx-utils.h"
#അന്തര്ഭാവ "htp-tensor.h"
#അന്തര്ഭാവ "dma-queue.h"
#അന്തര്ഭാവ "hex-profile.h"
#അന്തര്ഭാവ "allreduce-ops.h"
#അന്തര്ഭാവ "htp-fence.h"

ഘടന htp_allreduce_context {
    ഘടന htp_ops_context * octx;
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
സ്ഥിതം ശൂന്യം allreduce_thread_dma_1d_##SUFFIX(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {                   \
    ഘടന htp_allreduce_context * actx = (ഘടന htp_allreduce_context *) data;                                  \
    ഘടന htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    സ്ഥിരം uint32_t n_ranks     = actx->n_ranks;                                                                   \
    സ്ഥിരം uint32_t n_dsts      = actx->n_dsts;                                                                    \
    സ്ഥിരം uint32_t block_elems = actx->block_elems;                                                               \
                                                                                                                  \
    സ്ഥിരം uint32_t dr  = actx->elems_per_thread;                                                                  \
    സ്ഥിരം uint32_t ir0 = actx->rank_elem_start + dr * ith;                                                        \
    സ്ഥിരം uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_start + actx->rank_nelem);                                 \
    എങ്കിൽ (ir0 >= ir1) തിരികെ;                                                                                       \
                                                                                                                  \
    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    സ്ഥിരം size_t vtcm_thread_offset = ith * actx->vtcm_size_per_thread;                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + vtcm_thread_offset;                                           \
    uint8_t * res_spad_base = HAS_ADD ? (actx->res_spad_base + vtcm_thread_offset) : ശൂന്യ;                        \
                                                                                                                  \
    സ്ഥിരം size_t spad_half = actx->vtcm_size_per_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    പൂർണ്ണസംഖ്യ spad_idx = 0;                                                                                             \
                                                                                                                  \
    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_bytes = cur_elems * വലുപ്പം(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                  \
        വേണ്ടി (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + ir_prefetch * വലുപ്പം(TYPE);                                  \
            dma_queue_തള്ളുക(dma_q, dma_make_data(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 0);              \
        }                                                                                                         \
        വേണ്ടി (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + spad_idx * spad_half;                \
            സ്ഥിരം dma_addr_t s_ddr = octx->src[s]->data + ir_prefetch * വലുപ്പം(TYPE);                             \
            dma_queue_തള്ളുക(dma_q, dma_make_data(s_spad, s_ddr), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        എങ്കിൽ (HAS_ADD) {                                                                                            \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                              \
            സ്ഥിരം dma_addr_t r_ddr = octx->src[2 * n_ranks]->data + ir_prefetch * വലുപ്പം(TYPE);                   \
            dma_queue_തള്ളുക(dma_q, dma_make_data(r_spad, r_ddr), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    പൂർണ്ണസംഖ്യ comp_spad_idx = 0;                                                                                        \
    വേണ്ടി (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir);                                                          \
        size_t   cur_bytes = cur_elems * വലുപ്പം(TYPE);                                                            \
        വേണ്ടി (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_queue_എടുക്കുക(dma_q);                                                                                 \
        }                                                                                                         \
        വേണ്ടി (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_queue_എടുക്കുക(dma_q);                                                                                 \
        }                                                                                                         \
        എങ്കിൽ (HAS_ADD) {                                                                                            \
            dma_queue_എടുക്കുക(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_base + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_base[0] + vtcm_thread_offset + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_base[1] + vtcm_thread_offset + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_ADD ? (res_spad_base + comp_spad_idx * spad_half) : ശൂന്യ;                         \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_ADD_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        വേണ്ടി (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;          \
            HVX_ADD_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        എങ്കിൽ (HAS_ADD) {                                                                                            \
            HVX_ADD_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        വേണ്ടി (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + ir * വലുപ്പം(TYPE);                                           \
            dma_queue_തള്ളുക(dma_q, dma_make_data(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        എങ്കിൽ (ir_prefetch < ir1) {                                                                                  \
            uint32_t next_elems = MIN(block_elems, ir1 - ir_prefetch);                                            \
            size_t   next_bytes = next_elems * വലുപ്പം(TYPE);                                                      \
            വേണ്ടി (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;       \
                സ്ഥിരം dma_addr_t s_next = octx->src[s]->data + ir_prefetch * വലുപ്പം(TYPE);                        \
                dma_queue_തള്ളുക(dma_q, dma_make_data(s_spad, s_next), next_bytes, next_bytes, next_bytes, 1);      \
            }                                                                                                     \
            എങ്കിൽ (HAS_ADD) {                                                                                        \
                uint8_t * r_spad_next = res_spad_base + comp_spad_idx * spad_half;                                \
                സ്ഥിരം dma_addr_t r_next = octx->src[2 * n_ranks]->data + ir_prefetch * വലുപ്പം(TYPE);              \
                dma_queue_തള്ളുക(dma_q, dma_make_data(r_spad_next, r_next), next_bytes, next_bytes, next_bytes, 1); \
            }                                                                                                     \
            ir_prefetch += next_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_queue_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_add_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     ദശാംശം,  hvx_add_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f16, __fp16, hvx_add_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f32, ദശാംശം,  hvx_add_f32_aaa, 1)

#define DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD, IS_ROW_BCAST)                                                           \
സ്ഥിതം ശൂന്യം allreduce_thread_dma_2d_##SUFFIX(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {                                                   \
    ഘടന htp_allreduce_context * actx = (ഘടന htp_allreduce_context *) data;                                                                  \
    ഘടന htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    സ്ഥിരം uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    സ്ഥിരം uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    സ്ഥിരം uint32_t ne0              = actx->ne0;                                                                                                  \
    സ്ഥിരം uint32_t block_rows       = actx->block_elems;                                                                                          \
    സ്ഥിരം uint32_t row_size_aligned = actx->row_size_aligned;                                                                                     \
    സ്ഥിരം uint32_t row_bytes        = ne0 * വലുപ്പം(TYPE);                                                                                         \
                                                                                                                                                  \
    സ്ഥിരം uint32_t dr  = actx->elems_per_thread;                                                                                                  \
    സ്ഥിരം uint32_t r0  = actx->rank_elem_start + dr * ith;                                                                                        \
    സ്ഥിരം uint32_t r1  = MIN(r0 + dr, actx->rank_elem_start + actx->rank_nelem);                                                                  \
    എങ്കിൽ (r0 >= r1) തിരികെ;                                                                                                                         \
                                                                                                                                                  \
    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    സ്ഥിരം size_t vtcm_thread_offset = ith * actx->vtcm_size_per_thread;                                                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + vtcm_thread_offset;                                                                           \
    uint8_t * res_spad_base = HAS_ADD ? (IS_ROW_BCAST ? actx->res_spad_base : (actx->res_spad_base + vtcm_thread_offset)) : ശൂന്യ;                 \
                                                                                                                                                  \
    സ്ഥിരം size_t spad_half = actx->vtcm_size_per_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    പൂർണ്ണസംഖ്യ spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_rows = MIN(block_rows, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                                                  \
        വേണ്ടി (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_queue_തള്ളുക(dma_q, dma_make_data(d_ddr, d_spad), octx->dsts[d]->nb[1], row_size_aligned, row_bytes, 0);                            \
        }                                                                                                                                         \
        വേണ്ടി (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + spad_idx * spad_half;                                                \
            സ്ഥിരം dma_addr_t s_ddr = octx->src[s]->data + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_queue_തള്ളുക(dma_q, dma_make_data(s_spad, s_ddr), row_size_aligned, octx->src[s]->nb[1], row_bytes, cur_rows);                      \
        }                                                                                                                                         \
        എങ്കിൽ (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                                                              \
            സ്ഥിരം dma_addr_t r_ddr = octx->src[2 * n_ranks]->data + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_queue_തള്ളുക(dma_q, dma_make_data(r_spad, r_ddr), row_size_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, cur_rows);            \
        }                                                                                                                                         \
        r_prefetch += cur_rows;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    പൂർണ്ണസംഖ്യ comp_spad_idx = 0;                                                                                                                        \
    വേണ്ടി (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_rows = MIN(block_rows, r1 - r);                                                                                              \
        വേണ്ടി (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_queue_എടുക്കുക(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        വേണ്ടി (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_queue_എടുക്കുക(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        എങ്കിൽ (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            dma_queue_എടുക്കുക(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_base + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_base[0] + vtcm_thread_offset + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_base[1] + vtcm_thread_offset + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_ADD && !IS_ROW_BCAST) ? (res_spad_base + comp_spad_idx * spad_half) : ശൂന്യ;                                      \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        വേണ്ടി (uint32_t row = 0; row < cur_rows; row++) {                                                                                           \
            uint8_t * d_row = d_spad + row * row_size_aligned;                                                                                    \
            സ്ഥിരം uint8_t * s0_row = s0_spad + row * row_size_aligned;                                                                            \
            സ്ഥിരം uint8_t * s1_row = s1_spad + row * row_size_aligned;                                                                            \
            HVX_ADD_FN(d_row, s0_row, s1_row, ne0);                                                                                               \
            വേണ്ടി (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                സ്ഥിരം uint8_t * ss_row = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half + row * row_size_aligned;        \
                HVX_ADD_FN(d_row, d_row, ss_row, ne0);                                                                                            \
            }                                                                                                                                     \
            എങ്കിൽ (HAS_ADD) {                                                                                                                        \
                സ്ഥിരം uint8_t * res_row = IS_ROW_BCAST ? res_spad_base : (r_spad + row * row_size_aligned);                                       \
                HVX_ADD_FN(d_row, d_row, res_row, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        വേണ്ടി (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + r * octx->dsts[d]->nb[1];                                                                    \
            dma_queue_തള്ളുക(dma_q, dma_make_data(d_ddr, d_spad), octx->dsts[d]->nb[1], row_size_aligned, row_bytes, cur_rows);                     \
        }                                                                                                                                         \
        എങ്കിൽ (r_prefetch < r1) {                                                                                                                    \
            uint32_t next_rows = MIN(block_rows, r1 - r_prefetch);                                                                                \
            വേണ്ടി (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;                                       \
                സ്ഥിരം dma_addr_t s_next = octx->src[s]->data + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_queue_തള്ളുക(dma_q, dma_make_data(s_spad, s_next), row_size_aligned, octx->src[s]->nb[1], row_bytes, next_rows);                \
            }                                                                                                                                     \
            എങ്കിൽ (HAS_ADD && !IS_ROW_BCAST) {                                                                                                       \
                uint8_t * r_spad_next = res_spad_base + comp_spad_idx * spad_half;                                                                \
                സ്ഥിരം dma_addr_t r_next = octx->src[2 * n_ranks]->data + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_queue_തള്ളുക(dma_q, dma_make_data(r_spad_next, r_next), row_size_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, next_rows); \
            }                                                                                                                                     \
            r_prefetch += next_rows;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_rows;                                                                                                                            \
    }                                                                                                                                             \
    dma_queue_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_add_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           ദശാംശം,  hvx_add_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f16,       __fp16, hvx_add_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f32,       ദശാംശം,  hvx_add_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f16, __fp16, hvx_add_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f32, ദശാംശം,  hvx_add_f32_aaa, 1, 1)

സ്ഥിതം പൂർണ്ണസംഖ്യ validate_allreduce(
    ഘടന htp_ops_context * octx,
    സ്ഥിരം ഘടന htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    എങ്കിൽ (kparams->n_threads == 0 || (uint32_t) kparams->n_threads > octx->ctx->n_threads) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }
    octx->n_threads = (uint32_t) kparams->n_threads;

    എങ്കിൽ (kparams->vtcm_size_per_thread <= 0 || kparams->vtcm_size <= 0) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }

    സ്ഥിരം bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    സ്ഥിരം size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_എണ്ണം(
        n_ranks, octx->n_threads, has_add, kparams->is_row_bcast != 0);
    സ്ഥിരം size_t vtcm_size = n_vtcm_buffers * (size_t) kparams->vtcm_size_per_thread;
    എങ്കിൽ (vtcm_size != (size_t) kparams->vtcm_size) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }
    എങ്കിൽ (vtcm_size > octx->ctx->vtcm_size) {
        തിരികെ HTP_STATUS_VTCM_TOO_SMALL;
    }

    എങ്കിൽ (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    തിരികെ HTP_STATUS_OK;
}

പൂർണ്ണസംഖ്യ op_allreduce(ഘടന htp_ops_context * octx) {
    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1 && octx->ctx->mdev.idx > 0) {
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം ഘടന htp_allreduce_kernel_params * kparams = (സ്ഥിരം ഘടന htp_allreduce_kernel_params *) octx->kernel_params;
    സ്ഥിരം ഘടന htp_tensor * dst = octx->dst;

    സ്ഥിരം uint32_t rank    = (uint32_t) kparams->rank;
    സ്ഥിരം uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    എങ്കിൽ (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }

    സ്ഥിരം uint32_t fence_seq_entry = (uint32_t) octx->op_params[0];
    സ്ഥിരം uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    സ്ഥിരം ഘടന htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->data;

    സ്ഥിരം പൂർണ്ണസംഖ്യ status = validate_allreduce(octx, kparams, n_ranks);
    എങ്കിൽ (status != HTP_STATUS_OK) {
        എങ്കിൽ (status == HTP_STATUS_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, status);
        തിരികെ status;
    }

    സ്ഥിരം bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    സ്ഥിരം uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    സ്ഥിരം int32_t  mode  = kparams->mode;

    // 1. Entry Barrier: Synchronize all ranks before reading
    ഘടന htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    htp_fence_write(my_fence, fence_seq_entry, octx->status);

    വേണ്ടി (uint32_t j = 0; j < n_ranks; j++) {
        എങ്കിൽ (j == rank) തുടരുക;
        സ്ഥിരം ഘടന htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        ആകുമ്പോൾ (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            എങ്കിൽ ((int32_t)(peer_seq - fence_seq_entry) >= 0) {
                എങ്കിൽ (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                    തിരികെ peer_status;
                }
                നിർത്തുക;
            }
            എങ്കിൽ (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_entry, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                തിരികെ HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm അസ്ഥിര ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    // 2. Multi-threaded Reduction across assigned rank chunk
    എങ്കിൽ (nelem > 0) {
        സ്ഥിരം uint32_t n_threads            = (uint32_t) kparams->n_threads;
        സ്ഥിരം uint32_t block_elems          = (uint32_t) kparams->block_elems;
        സ്ഥിരം uint32_t elems_per_thread     = (uint32_t) kparams->elems_per_thread;
        സ്ഥിരം uint32_t vtcm_size_per_thread = (uint32_t) kparams->vtcm_size_per_thread;

        ഘടന htp_allreduce_context actx;
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

        work_queue_func_t reduce_fun = ശൂന്യ;
        തിരഞ്ഞെടുപ്പ് (kparams->kernel_type) {
            അവസ്ഥ HTP_ALLREDUCE_KERNEL_DMA_1D:
                എങ്കിൽ (has_add) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_add_f16 : allreduce_thread_dma_1d_add_f32;
                } അല്ലെങ്കിൽ {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                നിർത്തുക;
            അവസ്ഥ HTP_ALLREDUCE_KERNEL_DMA_2D:
                എങ്കിൽ (has_add) {
                    എങ്കിൽ (kparams->is_row_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_bcast_f16 : allreduce_thread_dma_2d_add_bcast_f32;
                    } അല്ലെങ്കിൽ {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_f16 : allreduce_thread_dma_2d_add_f32;
                    }
                } അല്ലെങ്കിൽ {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_NO_SUPPORT);
                തിരികെ HTP_STATUS_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
        വേണ്ടി (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_base[s] = vtcm_ptr;
            vtcm_ptr += n_threads * vtcm_size_per_thread;
        }
        actx.dst_spad_base = vtcm_ptr;
        vtcm_ptr += n_threads * vtcm_size_per_thread;
        എങ്കിൽ (has_add) {
            actx.res_spad_base = vtcm_ptr;
            vtcm_ptr += (actx.is_row_bcast ? 1 : n_threads) * vtcm_size_per_thread;
        }

        എങ്കിൽ (has_add && actx.is_row_bcast) {
            സ്ഥിരം dma_addr_t r_ddr = octx->src[2 * n_ranks]->data;
            സ്ഥിരം uint32_t row_bytes = actx.ne0 * (dst->type == HTP_TYPE_F16 ? വലുപ്പം(__fp16) : വലുപ്പം(ദശാംശം));
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_തള്ളുക(dma_q, dma_make_data(actx.res_spad_base, r_ddr), actx.row_size_aligned, 0, row_bytes, 1);
            dma_queue_എടുക്കുക(dma_q);
        }

        work_queue_run(octx->ctx->work_queue, reduce_fun, &actx, n_threads);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    എങ്കിൽ (mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
        asm അസ്ഥിര ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->status);

    വേണ്ടി (uint32_t j = 0; j < n_ranks; j++) {
        എങ്കിൽ (j == rank) തുടരുക;
        സ്ഥിരം ഘടന htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        ആകുമ്പോൾ (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            എങ്കിൽ ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                എങ്കിൽ (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    തിരികെ peer_status;
                }
                നിർത്തുക;
            }
            എങ്കിൽ (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                തിരികെ HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm അസ്ഥിര ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    തിരികെ octx->status;
}
