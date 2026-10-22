#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>
#समावेश <stdatomic.h>
#समावेश <math.h>
#समावेश <string.h>

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "hvx-utils.h"
#समावेश "htp-tensor.h"
#समावेश "dma-queue.h"
#समावेश "hex-profile.h"
#समावेश "allreduce-ops.h"
#समावेश "htp-fence.h"

काष्ठा htp_allreduce_context {
    काष्ठा htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t n_dsts;
    uint32_t nelem;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t row_आकार_aligned;
    uint32_t rank_elem_start;
    uint32_t rank_nelem;
    uint32_t elems_per_thread;
    uint32_t block_elems;
    uint32_t vtcm_आकार_per_thread;
    bool     is_row_bcast;
    uint8_t * src_spad_base[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_base;
    uint8_t * res_spad_base;
};

#घोषणा DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD)                                         \
अटल व्योम allreduce_thread_dma_1d_##SUFFIX(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {                   \
    काष्ठा htp_allreduce_context * actx = (काष्ठा htp_allreduce_context *) तथ्य;                                  \
    काष्ठा htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    स्थिर uint32_t n_ranks     = actx->n_ranks;                                                                   \
    स्थिर uint32_t n_dsts      = actx->n_dsts;                                                                    \
    स्थिर uint32_t block_elems = actx->block_elems;                                                               \
                                                                                                                  \
    स्थिर uint32_t dr  = actx->elems_per_thread;                                                                  \
    स्थिर uint32_t ir0 = actx->rank_elem_start + dr * ith;                                                        \
    स्थिर uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_start + actx->rank_nelem);                                 \
    अगर (ir0 >= ir1) वापस;                                                                                       \
                                                                                                                  \
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    स्थिर माप_प्रकार vtcm_thread_offset = ith * actx->vtcm_आकार_per_thread;                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + vtcm_thread_offset;                                           \
    uint8_t * res_spad_base = HAS_ADD ? (actx->res_spad_base + vtcm_thread_offset) : शून्य;                        \
                                                                                                                  \
    स्थिर माप_प्रकार spad_half = actx->vtcm_आकार_per_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    पूर्णांक spad_idx = 0;                                                                                             \
                                                                                                                  \
    क्रम (पूर्णांक k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir_prefetch);                                                 \
        माप_प्रकार   cur_bytes = cur_elems * माप(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                  \
        क्रम (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->तथ्य + ir_prefetch * माप(TYPE);                                  \
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 0);              \
        }                                                                                                         \
        क्रम (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + spad_idx * spad_half;                \
            स्थिर dma_addr_t s_ddr = octx->src[s]->तथ्य + ir_prefetch * माप(TYPE);                             \
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s_spad, s_ddr), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        अगर (HAS_ADD) {                                                                                            \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                              \
            स्थिर dma_addr_t r_ddr = octx->src[2 * n_ranks]->तथ्य + ir_prefetch * माप(TYPE);                   \
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(r_spad, r_ddr), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    पूर्णांक comp_spad_idx = 0;                                                                                        \
    क्रम (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir);                                                          \
        माप_प्रकार   cur_bytes = cur_elems * माप(TYPE);                                                            \
        क्रम (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_queue_निकालो(dma_q);                                                                                 \
        }                                                                                                         \
        क्रम (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_queue_निकालो(dma_q);                                                                                 \
        }                                                                                                         \
        अगर (HAS_ADD) {                                                                                            \
            dma_queue_निकालो(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_base + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_base[0] + vtcm_thread_offset + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_base[1] + vtcm_thread_offset + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_ADD ? (res_spad_base + comp_spad_idx * spad_half) : शून्य;                         \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_ADD_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        क्रम (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;          \
            HVX_ADD_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        अगर (HAS_ADD) {                                                                                            \
            HVX_ADD_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        क्रम (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->तथ्य + ir * माप(TYPE);                                           \
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 1);              \
        }                                                                                                         \
        अगर (ir_prefetch < ir1) {                                                                                  \
            uint32_t अग्र_elems = MIN(block_elems, ir1 - ir_prefetch);                                            \
            माप_प्रकार   अग्र_bytes = अग्र_elems * माप(TYPE);                                                      \
            क्रम (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;       \
                स्थिर dma_addr_t s_अग्र = octx->src[s]->तथ्य + ir_prefetch * माप(TYPE);                        \
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(s_spad, s_अग्र), अग्र_bytes, अग्र_bytes, अग्र_bytes, 1);      \
            }                                                                                                     \
            अगर (HAS_ADD) {                                                                                        \
                uint8_t * r_spad_अग्र = res_spad_base + comp_spad_idx * spad_half;                                \
                स्थिर dma_addr_t r_अग्र = octx->src[2 * n_ranks]->तथ्य + ir_prefetch * माप(TYPE);              \
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(r_spad_अग्र, r_अग्र), अग्र_bytes, अग्र_bytes, अग्र_bytes, 1); \
            }                                                                                                     \
            ir_prefetch += अग्र_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_queue_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_add_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     भग्न,  hvx_add_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f16, __fp16, hvx_add_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f32, भग्न,  hvx_add_f32_aaa, 1)

#घोषणा DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD, IS_ROW_BCAST)                                                           \
अटल व्योम allreduce_thread_dma_2d_##SUFFIX(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {                                                   \
    काष्ठा htp_allreduce_context * actx = (काष्ठा htp_allreduce_context *) तथ्य;                                                                  \
    काष्ठा htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    स्थिर uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    स्थिर uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    स्थिर uint32_t ne0              = actx->ne0;                                                                                                  \
    स्थिर uint32_t block_rows       = actx->block_elems;                                                                                          \
    स्थिर uint32_t row_आकार_aligned = actx->row_आकार_aligned;                                                                                     \
    स्थिर uint32_t row_bytes        = ne0 * माप(TYPE);                                                                                         \
                                                                                                                                                  \
    स्थिर uint32_t dr  = actx->elems_per_thread;                                                                                                  \
    स्थिर uint32_t r0  = actx->rank_elem_start + dr * ith;                                                                                        \
    स्थिर uint32_t r1  = MIN(r0 + dr, actx->rank_elem_start + actx->rank_nelem);                                                                  \
    अगर (r0 >= r1) वापस;                                                                                                                         \
                                                                                                                                                  \
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    स्थिर माप_प्रकार vtcm_thread_offset = ith * actx->vtcm_आकार_per_thread;                                                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + vtcm_thread_offset;                                                                           \
    uint8_t * res_spad_base = HAS_ADD ? (IS_ROW_BCAST ? actx->res_spad_base : (actx->res_spad_base + vtcm_thread_offset)) : शून्य;                 \
                                                                                                                                                  \
    स्थिर माप_प्रकार spad_half = actx->vtcm_आकार_per_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    पूर्णांक spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    क्रम (पूर्णांक k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_rows = MIN(block_rows, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                                                  \
        क्रम (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->तथ्य + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(d_ddr, d_spad), octx->dsts[d]->nb[1], row_आकार_aligned, row_bytes, 0);                            \
        }                                                                                                                                         \
        क्रम (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + spad_idx * spad_half;                                                \
            स्थिर dma_addr_t s_ddr = octx->src[s]->तथ्य + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s_spad, s_ddr), row_आकार_aligned, octx->src[s]->nb[1], row_bytes, cur_rows);                      \
        }                                                                                                                                         \
        अगर (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                                                              \
            स्थिर dma_addr_t r_ddr = octx->src[2 * n_ranks]->तथ्य + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(r_spad, r_ddr), row_आकार_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, cur_rows);            \
        }                                                                                                                                         \
        r_prefetch += cur_rows;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    पूर्णांक comp_spad_idx = 0;                                                                                                                        \
    क्रम (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_rows = MIN(block_rows, r1 - r);                                                                                              \
        क्रम (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_queue_निकालो(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        क्रम (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_queue_निकालो(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        अगर (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            dma_queue_निकालो(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_base + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_base[0] + vtcm_thread_offset + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_base[1] + vtcm_thread_offset + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_ADD && !IS_ROW_BCAST) ? (res_spad_base + comp_spad_idx * spad_half) : शून्य;                                      \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        क्रम (uint32_t row = 0; row < cur_rows; row++) {                                                                                           \
            uint8_t * d_row = d_spad + row * row_आकार_aligned;                                                                                    \
            स्थिर uint8_t * s0_row = s0_spad + row * row_आकार_aligned;                                                                            \
            स्थिर uint8_t * s1_row = s1_spad + row * row_आकार_aligned;                                                                            \
            HVX_ADD_FN(d_row, s0_row, s1_row, ne0);                                                                                               \
            क्रम (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                स्थिर uint8_t * ss_row = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half + row * row_आकार_aligned;        \
                HVX_ADD_FN(d_row, d_row, ss_row, ne0);                                                                                            \
            }                                                                                                                                     \
            अगर (HAS_ADD) {                                                                                                                        \
                स्थिर uint8_t * res_row = IS_ROW_BCAST ? res_spad_base : (r_spad + row * row_आकार_aligned);                                       \
                HVX_ADD_FN(d_row, d_row, res_row, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        क्रम (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->तथ्य + r * octx->dsts[d]->nb[1];                                                                    \
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(d_ddr, d_spad), octx->dsts[d]->nb[1], row_आकार_aligned, row_bytes, cur_rows);                     \
        }                                                                                                                                         \
        अगर (r_prefetch < r1) {                                                                                                                    \
            uint32_t अग्र_rows = MIN(block_rows, r1 - r_prefetch);                                                                                \
            क्रम (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_base[s] + vtcm_thread_offset + comp_spad_idx * spad_half;                                       \
                स्थिर dma_addr_t s_अग्र = octx->src[s]->तथ्य + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(s_spad, s_अग्र), row_आकार_aligned, octx->src[s]->nb[1], row_bytes, अग्र_rows);                \
            }                                                                                                                                     \
            अगर (HAS_ADD && !IS_ROW_BCAST) {                                                                                                       \
                uint8_t * r_spad_अग्र = res_spad_base + comp_spad_idx * spad_half;                                                                \
                स्थिर dma_addr_t r_अग्र = octx->src[2 * n_ranks]->तथ्य + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(r_spad_अग्र, r_अग्र), row_आकार_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, अग्र_rows); \
            }                                                                                                                                     \
            r_prefetch += अग्र_rows;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_rows;                                                                                                                            \
    }                                                                                                                                             \
    dma_queue_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_add_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           भग्न,  hvx_add_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f16,       __fp16, hvx_add_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f32,       भग्न,  hvx_add_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f16, __fp16, hvx_add_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f32, भग्न,  hvx_add_f32_aaa, 1, 1)

अटल पूर्णांक validate_allreduce(
    काष्ठा htp_ops_context * octx,
    स्थिर काष्ठा htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    अगर (kparams->n_threads == 0 || (uint32_t) kparams->n_threads > octx->ctx->n_threads) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }
    octx->n_threads = (uint32_t) kparams->n_threads;

    अगर (kparams->vtcm_आकार_per_thread <= 0 || kparams->vtcm_आकार <= 0) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    स्थिर माप_प्रकार n_vtcm_buffers = htp_allreduce_vtcm_बफर_गिनती(
        n_ranks, octx->n_threads, has_add, kparams->is_row_bcast != 0);
    स्थिर माप_प्रकार vtcm_आकार = n_vtcm_buffers * (माप_प्रकार) kparams->vtcm_आकार_per_thread;
    अगर (vtcm_आकार != (माप_प्रकार) kparams->vtcm_आकार) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }
    अगर (vtcm_आकार > octx->ctx->vtcm_आकार) {
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    अगर (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    वापस HTP_STATUS_OK;
}

पूर्णांक op_allreduce(काष्ठा htp_ops_context * octx) {
    अगर (octx->ctx->mdev.गिनती > 1 && octx->ctx->mdev.idx > 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर काष्ठा htp_allreduce_kernel_params * kparams = (स्थिर काष्ठा htp_allreduce_kernel_params *) octx->kernel_params;
    स्थिर काष्ठा htp_tensor * dst = octx->dst;

    स्थिर uint32_t rank    = (uint32_t) kparams->rank;
    स्थिर uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    अगर (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर uint32_t fence_seq_entry = (uint32_t) octx->op_params[0];
    स्थिर uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    स्थिर काष्ठा htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->तथ्य;

    स्थिर पूर्णांक status = validate_allreduce(octx, kparams, n_ranks);
    अगर (status != HTP_STATUS_OK) {
        अगर (status == HTP_STATUS_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, status);
        वापस status;
    }

    स्थिर bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    स्थिर uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    स्थिर int32_t  mode  = kparams->mode;

    // 1. Entry Barrier: Synchronize all ranks before reading
    काष्ठा htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    htp_fence_write(my_fence, fence_seq_entry, octx->status);

    क्रम (uint32_t j = 0; j < n_ranks; j++) {
        अगर (j == rank) जारी;
        स्थिर काष्ठा htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->तथ्य;
        uint64_t spins = 0;
        जबतक (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            अगर ((int32_t)(peer_seq - fence_seq_entry) >= 0) {
                अगर (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                    वापस peer_status;
                }
                अवरोध;
            }
            अगर (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_entry, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                वापस HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    यंत्र अस्थिर ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    // 2. Multi-threaded Reduction across assigned rank chunk
    अगर (nelem > 0) {
        स्थिर uint32_t n_threads            = (uint32_t) kparams->n_threads;
        स्थिर uint32_t block_elems          = (uint32_t) kparams->block_elems;
        स्थिर uint32_t elems_per_thread     = (uint32_t) kparams->elems_per_thread;
        स्थिर uint32_t vtcm_आकार_per_thread = (uint32_t) kparams->vtcm_आकार_per_thread;

        काष्ठा htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.row_आकार_aligned     = (uint32_t) kparams->row_आकार_aligned;
        actx.rank_elem_start      = (uint32_t) kparams->rank_elem_start;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_per_thread     = elems_per_thread;
        actx.block_elems          = block_elems;
        actx.vtcm_आकार_per_thread = vtcm_आकार_per_thread;
        actx.is_row_bcast         = (kparams->is_row_bcast != 0);

        work_queue_func_t reduce_fun = शून्य;
        चयन (kparams->kernel_type) {
            हाल HTP_ALLREDUCE_KERNEL_DMA_1D:
                अगर (has_add) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_add_f16 : allreduce_thread_dma_1d_add_f32;
                } अन्यथा {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                अवरोध;
            हाल HTP_ALLREDUCE_KERNEL_DMA_2D:
                अगर (has_add) {
                    अगर (kparams->is_row_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_bcast_f16 : allreduce_thread_dma_2d_add_bcast_f32;
                    } अन्यथा {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_f16 : allreduce_thread_dma_2d_add_f32;
                    }
                } अन्यथा {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                अवरोध;
            शेष:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_NO_SUPPORT);
                वापस HTP_STATUS_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
        क्रम (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_base[s] = vtcm_ptr;
            vtcm_ptr += n_threads * vtcm_आकार_per_thread;
        }
        actx.dst_spad_base = vtcm_ptr;
        vtcm_ptr += n_threads * vtcm_आकार_per_thread;
        अगर (has_add) {
            actx.res_spad_base = vtcm_ptr;
            vtcm_ptr += (actx.is_row_bcast ? 1 : n_threads) * vtcm_आकार_per_thread;
        }

        अगर (has_add && actx.is_row_bcast) {
            स्थिर dma_addr_t r_ddr = octx->src[2 * n_ranks]->तथ्य;
            स्थिर uint32_t row_bytes = actx.ne0 * (dst->type == HTP_TYPE_F16 ? माप(__fp16) : माप(भग्न));
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(actx.res_spad_base, r_ddr), actx.row_आकार_aligned, 0, row_bytes, 1);
            dma_queue_निकालो(dma_q);
        }

        work_queue_run(octx->ctx->work_queue, reduce_fun, &actx, n_threads);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    अगर (mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
        यंत्र अस्थिर ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->status);

    क्रम (uint32_t j = 0; j < n_ranks; j++) {
        अगर (j == rank) जारी;
        स्थिर काष्ठा htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->तथ्य;
        uint64_t spins = 0;
        जबतक (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            अगर ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                अगर (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    वापस peer_status;
                }
                अवरोध;
            }
            अगर (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                वापस HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    यंत्र अस्थिर ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    वापस octx->status;
}
