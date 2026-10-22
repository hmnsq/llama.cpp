#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>
#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "allreduce-ops.h"
#अन्तर्भाव "htp-fence.h"

संरचना htp_allreduce_context {
    संरचना htp_ops_context * octx;
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
स्थिर शून्य allreduce_thread_dma_1d_##SUFFIX(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, शून्य * data) {                   \
    संरचना htp_allreduce_context * actx = (संरचना htp_allreduce_context *) data;                                  \
    संरचना htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    स्थिरांक uint32_t n_ranks     = actx->n_ranks;                                                                   \
    स्थिरांक uint32_t n_dsts      = actx->n_dsts;                                                                    \
    स्थिरांक uint32_t block_elems = actx->block_elems;                                                               \
                                                                                                                  \
    स्थिरांक uint32_t dr  = actx->elems_per_thread;                                                                  \
    स्थिरांक uint32_t ir0 = actx->rank_elem_start + dr * ith;                                                        \
    स्थिरांक uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_start + actx->rank_nelem);                                 \
    जर (ir0 >= ir1) परत;                                                                                       \
                                                                                                                  \
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    स्थिरांक माप_प्रकार vtcm_thread_offset = ith * actx->vtcm_size_per_thread;                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + vtcm_thread_offset;                                           \
    uint8_t * res_spad_base = HAS_ADD ? (actx->res_spad_base + vtcm_thread_offset) : रिक्त;                        \
                                                                                                                  \
    स्थिरांक माप_प्रकार spad_half = actx->vtcm_size_per_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    पूर्णांक spad_idx = 0;                                                                                             \
                                                                                                                  \
    साठी (पूर्णांक k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir_prefetch);                                                 \
        माप_प्रकार   cur_bytes = cur_elems * आकार(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                  \
        साठी (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + ir_prefetch * आकार(TYPE);                                  \
            dma_queue_ढकला(dma_q, dma_make_data(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 0);              \
        }                                                                                                         \
        साठी (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + spad_idx * spad_half;                \
            स्थिरांक dma_addr_t s_ddr = octx->src[s]->data + ir_prefetch * आकार(TYPE);                             \
            dma_queue_ढकला(dma_q, dma_make_data(s_spad, s_ddr), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        जर (HAS_ADD) {                                                                                            \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                              \
            स्थिरांक dma_addr_t r_ddr = octx->src[2 * n_ranks]->data + ir_prefetch * आकार(TYPE);                   \
            dma_queue_ढकला(dma_q, dma_make_data(r_spad, r_ddr), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    पूर्णांक comp_spad_idx = 0;                                                                                        \
    साठी (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir);                                                          \
        माप_प्रकार   cur_bytes = cur_elems * आकार(TYPE);                                                            \
        साठी (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_queue_काढा(dma_q);                                                                                 \
        }                                                                                                         \
        साठी (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_queue_काढा(dma_q);                                                                                 \
        }                                                                                                         \
        जर (HAS_ADD) {                                                                                            \
            dma_queue_काढा(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_base + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_base[0] + vtcm_thread_offset + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_base[1] + vtcm_thread_offset + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_ADD ? (res_spad_base + comp_spad_idx * spad_half) : रिक्त;                         \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_ADD_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        साठी (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;          \
            HVX_ADD_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        जर (HAS_ADD) {                                                                                            \
            HVX_ADD_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        साठी (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + ir * आकार(TYPE);                                           \
            dma_queue_ढकला(dma_q, dma_make_data(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        जर (ir_prefetch < ir1) {                                                                                  \
            uint32_t next_elems = MIN(block_elems, ir1 - ir_prefetch);                                            \
            माप_प्रकार   next_bytes = next_elems * आकार(TYPE);                                                      \
            साठी (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;       \
                स्थिरांक dma_addr_t s_next = octx->src[s]->data + ir_prefetch * आकार(TYPE);                        \
                dma_queue_ढकला(dma_q, dma_make_data(s_spad, s_next), next_bytes, next_bytes, next_bytes, 1);      \
            }                                                                                                     \
            जर (HAS_ADD) {                                                                                        \
                uint8_t * r_spad_next = res_spad_base + comp_spad_idx * spad_half;                                \
                स्थिरांक dma_addr_t r_next = octx->src[2 * n_ranks]->data + ir_prefetch * आकार(TYPE);              \
                dma_queue_ढकला(dma_q, dma_make_data(r_spad_next, r_next), next_bytes, next_bytes, next_bytes, 1); \
            }                                                                                                     \
            ir_prefetch += next_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_queue_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_add_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     दशांश,  hvx_add_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f16, __fp16, hvx_add_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f32, दशांश,  hvx_add_f32_aaa, 1)

#define DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD, IS_ROW_BCAST)                                                           \
स्थिर शून्य allreduce_thread_dma_2d_##SUFFIX(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, शून्य * data) {                                                   \
    संरचना htp_allreduce_context * actx = (संरचना htp_allreduce_context *) data;                                                                  \
    संरचना htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    स्थिरांक uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    स्थिरांक uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    स्थिरांक uint32_t ne0              = actx->ne0;                                                                                                  \
    स्थिरांक uint32_t block_rows       = actx->block_elems;                                                                                          \
    स्थिरांक uint32_t row_size_aligned = actx->row_size_aligned;                                                                                     \
    स्थिरांक uint32_t row_bytes        = ne0 * आकार(TYPE);                                                                                         \
                                                                                                                                                  \
    स्थिरांक uint32_t dr  = actx->elems_per_thread;                                                                                                  \
    स्थिरांक uint32_t r0  = actx->rank_elem_start + dr * ith;                                                                                        \
    स्थिरांक uint32_t r1  = MIN(r0 + dr, actx->rank_elem_start + actx->rank_nelem);                                                                  \
    जर (r0 >= r1) परत;                                                                                                                         \
                                                                                                                                                  \
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    स्थिरांक माप_प्रकार vtcm_thread_offset = ith * actx->vtcm_size_per_thread;                                                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + vtcm_thread_offset;                                                                           \
    uint8_t * res_spad_base = HAS_ADD ? (IS_ROW_BCAST ? actx->res_spad_base : (actx->res_spad_base + vtcm_thread_offset)) : रिक्त;                 \
                                                                                                                                                  \
    स्थिरांक माप_प्रकार spad_half = actx->vtcm_size_per_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    पूर्णांक spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    साठी (पूर्णांक k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_rows = MIN(block_rows, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                                                  \
        साठी (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_queue_ढकला(dma_q, dma_make_data(d_ddr, d_spad), octx->dsts[d]->nb[1], row_size_aligned, row_bytes, 0);                            \
        }                                                                                                                                         \
        साठी (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + spad_idx * spad_half;                                                \
            स्थिरांक dma_addr_t s_ddr = octx->src[s]->data + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_queue_ढकला(dma_q, dma_make_data(s_spad, s_ddr), row_size_aligned, octx->src[s]->nb[1], row_bytes, cur_rows);                      \
        }                                                                                                                                         \
        जर (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                                                              \
            स्थिरांक dma_addr_t r_ddr = octx->src[2 * n_ranks]->data + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_queue_ढकला(dma_q, dma_make_data(r_spad, r_ddr), row_size_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, cur_rows);            \
        }                                                                                                                                         \
        r_prefetch += cur_rows;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    पूर्णांक comp_spad_idx = 0;                                                                                                                        \
    साठी (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_rows = MIN(block_rows, r1 - r);                                                                                              \
        साठी (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_queue_काढा(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        साठी (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_queue_काढा(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        जर (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            dma_queue_काढा(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_base + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_base[0] + vtcm_thread_offset + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_base[1] + vtcm_thread_offset + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_ADD && !IS_ROW_BCAST) ? (res_spad_base + comp_spad_idx * spad_half) : रिक्त;                                      \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        साठी (uint32_t row = 0; row < cur_rows; row++) {                                                                                           \
            uint8_t * d_row = d_spad + row * row_size_aligned;                                                                                    \
            स्थिरांक uint8_t * s0_row = s0_spad + row * row_size_aligned;                                                                            \
            स्थिरांक uint8_t * s1_row = s1_spad + row * row_size_aligned;                                                                            \
            HVX_ADD_FN(d_row, s0_row, s1_row, ne0);                                                                                               \
            साठी (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                स्थिरांक uint8_t * ss_row = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half + row * row_size_aligned;        \
                HVX_ADD_FN(d_row, d_row, ss_row, ne0);                                                                                            \
            }                                                                                                                                     \
            जर (HAS_ADD) {                                                                                                                        \
                स्थिरांक uint8_t * res_row = IS_ROW_BCAST ? res_spad_base : (r_spad + row * row_size_aligned);                                       \
                HVX_ADD_FN(d_row, d_row, res_row, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        साठी (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + r * octx->dsts[d]->nb[1];                                                                    \
            dma_queue_ढकला(dma_q, dma_make_data(d_ddr, d_spad), octx->dsts[d]->nb[1], row_size_aligned, row_bytes, cur_rows);                     \
        }                                                                                                                                         \
        जर (r_prefetch < r1) {                                                                                                                    \
            uint32_t next_rows = MIN(block_rows, r1 - r_prefetch);                                                                                \
            साठी (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;                                       \
                स्थिरांक dma_addr_t s_next = octx->src[s]->data + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_queue_ढकला(dma_q, dma_make_data(s_spad, s_next), row_size_aligned, octx->src[s]->nb[1], row_bytes, next_rows);                \
            }                                                                                                                                     \
            जर (HAS_ADD && !IS_ROW_BCAST) {                                                                                                       \
                uint8_t * r_spad_next = res_spad_base + comp_spad_idx * spad_half;                                                                \
                स्थिरांक dma_addr_t r_next = octx->src[2 * n_ranks]->data + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_queue_ढकला(dma_q, dma_make_data(r_spad_next, r_next), row_size_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, next_rows); \
            }                                                                                                                                     \
            r_prefetch += next_rows;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_rows;                                                                                                                            \
    }                                                                                                                                             \
    dma_queue_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_add_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           दशांश,  hvx_add_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f16,       __fp16, hvx_add_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f32,       दशांश,  hvx_add_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f16, __fp16, hvx_add_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f32, दशांश,  hvx_add_f32_aaa, 1, 1)

स्थिर पूर्णांक validate_allreduce(
    संरचना htp_ops_context * octx,
    स्थिरांक संरचना htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    जर (kparams->n_threads == 0 || (uint32_t) kparams->n_threads > octx->ctx->n_threads) {
        परत HTP_STATUS_INVAL_PARAMS;
    }
    octx->n_threads = (uint32_t) kparams->n_threads;

    जर (kparams->vtcm_size_per_thread <= 0 || kparams->vtcm_size <= 0) {
        परत HTP_STATUS_INVAL_PARAMS;
    }

    स्थिरांक bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    स्थिरांक माप_प्रकार n_vtcm_buffers = htp_allreduce_vtcm_buffer_मोजणी(
        n_ranks, octx->n_threads, has_add, kparams->is_row_bcast != 0);
    स्थिरांक माप_प्रकार vtcm_size = n_vtcm_buffers * (माप_प्रकार) kparams->vtcm_size_per_thread;
    जर (vtcm_size != (माप_प्रकार) kparams->vtcm_size) {
        परत HTP_STATUS_INVAL_PARAMS;
    }
    जर (vtcm_size > octx->ctx->vtcm_size) {
        परत HTP_STATUS_VTCM_TOO_SMALL;
    }

    जर (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    परत HTP_STATUS_OK;
}

पूर्णांक op_allreduce(संरचना htp_ops_context * octx) {
    जर (octx->ctx->mdev.मोजणी > 1 && octx->ctx->mdev.idx > 0) {
        परत HTP_STATUS_OK;
    }

    स्थिरांक संरचना htp_allreduce_kernel_params * kparams = (स्थिरांक संरचना htp_allreduce_kernel_params *) octx->kernel_params;
    स्थिरांक संरचना htp_tensor * dst = octx->dst;

    स्थिरांक uint32_t rank    = (uint32_t) kparams->rank;
    स्थिरांक uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    जर (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        परत HTP_STATUS_INVAL_PARAMS;
    }

    स्थिरांक uint32_t fence_seq_entry = (uint32_t) octx->op_params[0];
    स्थिरांक uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    स्थिरांक संरचना htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->data;

    स्थिरांक पूर्णांक status = validate_allreduce(octx, kparams, n_ranks);
    जर (status != HTP_STATUS_OK) {
        जर (status == HTP_STATUS_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, status);
        परत status;
    }

    स्थिरांक bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    स्थिरांक uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    स्थिरांक int32_t  mode  = kparams->mode;

    // 1. Entry Barrier: Synchronize all ranks before reading
    संरचना htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    htp_fence_write(my_fence, fence_seq_entry, octx->status);

    साठी (uint32_t j = 0; j < n_ranks; j++) {
        जर (j == rank) सुरूठेवा;
        स्थिरांक संरचना htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        पर्यंत (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            जर ((int32_t)(peer_seq - fence_seq_entry) >= 0) {
                जर (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                    परत peer_status;
                }
                थांबा;
            }
            जर (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_entry, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                परत HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    यंत्र अस्थिर ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    // 2. Multi-threaded Reduction across assigned rank chunk
    जर (nelem > 0) {
        स्थिरांक uint32_t n_threads            = (uint32_t) kparams->n_threads;
        स्थिरांक uint32_t block_elems          = (uint32_t) kparams->block_elems;
        स्थिरांक uint32_t elems_per_thread     = (uint32_t) kparams->elems_per_thread;
        स्थिरांक uint32_t vtcm_size_per_thread = (uint32_t) kparams->vtcm_size_per_thread;

        संरचना htp_allreduce_context actx;
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

        work_queue_func_t reduce_fun = रिक्त;
        निवड (kparams->kernel_type) {
            स्थिती HTP_ALLREDUCE_KERNEL_DMA_1D:
                जर (has_add) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_add_f16 : allreduce_thread_dma_1d_add_f32;
                } अन्यथा {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                थांबा;
            स्थिती HTP_ALLREDUCE_KERNEL_DMA_2D:
                जर (has_add) {
                    जर (kparams->is_row_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_bcast_f16 : allreduce_thread_dma_2d_add_bcast_f32;
                    } अन्यथा {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_f16 : allreduce_thread_dma_2d_add_f32;
                    }
                } अन्यथा {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                थांबा;
            पूर्वनिर्धारित:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_NO_SUPPORT);
                परत HTP_STATUS_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
        साठी (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_base[s] = vtcm_ptr;
            vtcm_ptr += n_threads * vtcm_size_per_thread;
        }
        actx.dst_spad_base = vtcm_ptr;
        vtcm_ptr += n_threads * vtcm_size_per_thread;
        जर (has_add) {
            actx.res_spad_base = vtcm_ptr;
            vtcm_ptr += (actx.is_row_bcast ? 1 : n_threads) * vtcm_size_per_thread;
        }

        जर (has_add && actx.is_row_bcast) {
            स्थिरांक dma_addr_t r_ddr = octx->src[2 * n_ranks]->data;
            स्थिरांक uint32_t row_bytes = actx.ne0 * (dst->type == HTP_TYPE_F16 ? आकार(__fp16) : आकार(दशांश));
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_ढकला(dma_q, dma_make_data(actx.res_spad_base, r_ddr), actx.row_size_aligned, 0, row_bytes, 1);
            dma_queue_काढा(dma_q);
        }

        work_queue_run(octx->ctx->work_queue, reduce_fun, &actx, n_threads);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    जर (mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
        यंत्र अस्थिर ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->status);

    साठी (uint32_t j = 0; j < n_ranks; j++) {
        जर (j == rank) सुरूठेवा;
        स्थिरांक संरचना htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        पर्यंत (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            जर ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                जर (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    परत peer_status;
                }
                थांबा;
            }
            जर (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                परत HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    यंत्र अस्थिर ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    परत octx->status;
}
