#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>
#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

#परिभाषय GGML_COMMON_DECL_C
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
    uint32_t row_आकार_aligned;
    uint32_t rank_elem_start;
    uint32_t rank_nelem;
    uint32_t elems_प्रत्येकम्_thread;
    uint32_t खण्डम्_elems;
    uint32_t vtcm_आकार_प्रत्येकम्_thread;
    bool     is_row_bcast;
    uint8_t * src_spad_आधारः[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_आधारः;
    uint8_t * res_spad_आधारः;
};

#परिभाषय DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_सङ्कलय_FN, HAS_सङ्कलय)                                         \
स्थिर शून्य allreduce_thread_dma_1d_##SUFFIX(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {                   \
    संरचना htp_allreduce_context * actx = (संरचना htp_allreduce_context *) दत्त;                                  \
    संरचना htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    ध्रुव uint32_t n_ranks     = actx->n_ranks;                                                                   \
    ध्रुव uint32_t n_dsts      = actx->n_dsts;                                                                    \
    ध्रुव uint32_t खण्डम्_elems = actx->खण्डम्_elems;                                                               \
                                                                                                                  \
    ध्रुव uint32_t dr  = actx->elems_प्रत्येकम्_thread;                                                                  \
    ध्रुव uint32_t ir0 = actx->rank_elem_start + dr * ith;                                                        \
    ध्रुव uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_start + actx->rank_nelem);                                 \
    यदि (ir0 >= ir1) प्रत्यागम;                                                                                       \
                                                                                                                  \
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    ध्रुव size_t vtcm_thread_विस्थापनम् = ith * actx->vtcm_आकार_प्रत्येकम्_thread;                                           \
    uint8_t * dst_spad_आधारः = actx->dst_spad_आधारः + vtcm_thread_विस्थापनम्;                                           \
    uint8_t * res_spad_आधारः = HAS_सङ्कलय ? (actx->res_spad_आधारः + vtcm_thread_विस्थापनम्) : रिक्त;                        \
                                                                                                                  \
    ध्रुव size_t spad_half = actx->vtcm_आकार_प्रत्येकम्_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    पूर्णाङ्क spad_idx = 0;                                                                                             \
                                                                                                                  \
    प्रति (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(खण्डम्_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_अष्टकानि = cur_elems * परिमाण(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_आधारः + spad_idx * spad_half;                                                  \
        प्रति (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->दत्त + ir_prefetch * परिमाण(TYPE);                                  \
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(d_ddr, d_spad), cur_अष्टकानि, cur_अष्टकानि, cur_अष्टकानि, 0);              \
        }                                                                                                         \
        प्रति (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_आधारः[s] + vtcm_thread_विस्थापनम् + spad_idx * spad_half;                \
            ध्रुव dma_addr_t s_ddr = octx->src[s]->दत्त + ir_prefetch * परिमाण(TYPE);                             \
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s_spad, s_ddr), cur_अष्टकानि, cur_अष्टकानि, cur_अष्टकानि, 1);              \
        }                                                                                                         \
        यदि (HAS_सङ्कलय) {                                                                                            \
            uint8_t * r_spad = res_spad_आधारः + spad_idx * spad_half;                                              \
            ध्रुव dma_addr_t r_ddr = octx->src[2 * n_ranks]->दत्त + ir_prefetch * परिमाण(TYPE);                   \
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(r_spad, r_ddr), cur_अष्टकानि, cur_अष्टकानि, cur_अष्टकानि, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    पूर्णाङ्क comp_spad_idx = 0;                                                                                        \
    प्रति (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(खण्डम्_elems, ir1 - ir);                                                          \
        size_t   cur_अष्टकानि = cur_elems * परिमाण(TYPE);                                                            \
        प्रति (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_queue_निष्कासय(dma_q);                                                                                 \
        }                                                                                                         \
        प्रति (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_queue_निष्कासय(dma_q);                                                                                 \
        }                                                                                                         \
        यदि (HAS_सङ्कलय) {                                                                                            \
            dma_queue_निष्कासय(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_आधारः + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_आधारः[0] + vtcm_thread_विस्थापनम् + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_आधारः[1] + vtcm_thread_विस्थापनम् + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_सङ्कलय ? (res_spad_आधारः + comp_spad_idx * spad_half) : रिक्त;                         \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_सङ्कलय_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        प्रति (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_आधारः[s] + vtcm_thread_विस्थापनम् + comp_spad_idx * spad_half;          \
            HVX_सङ्कलय_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        यदि (HAS_सङ्कलय) {                                                                                            \
            HVX_सङ्कलय_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        प्रति (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->दत्त + ir * परिमाण(TYPE);                                           \
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(d_ddr, d_spad), cur_अष्टकानि, cur_अष्टकानि, cur_अष्टकानि, 1);              \
        }                                                                                                         \
        यदि (ir_prefetch < ir1) {                                                                                  \
            uint32_t अग्र_elems = MIN(खण्डम्_elems, ir1 - ir_prefetch);                                            \
            size_t   अग्र_अष्टकानि = अग्र_elems * परिमाण(TYPE);                                                      \
            प्रति (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_आधारः[s] + vtcm_thread_विस्थापनम् + comp_spad_idx * spad_half;       \
                ध्रुव dma_addr_t s_अग्र = octx->src[s]->दत्त + ir_prefetch * परिमाण(TYPE);                        \
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s_spad, s_अग्र), अग्र_अष्टकानि, अग्र_अष्टकानि, अग्र_अष्टकानि, 1);      \
            }                                                                                                     \
            यदि (HAS_सङ्कलय) {                                                                                        \
                uint8_t * r_spad_अग्र = res_spad_आधारः + comp_spad_idx * spad_half;                                \
                ध्रुव dma_addr_t r_अग्र = octx->src[2 * n_ranks]->दत्त + ir_prefetch * परिमाण(TYPE);              \
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(r_spad_अग्र, r_अग्र), अग्र_अष्टकानि, अग्र_अष्टकानि, अग्र_अष्टकानि, 1); \
            }                                                                                                     \
            ir_prefetch += अग्र_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_queue_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_सङ्कलय_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     भिन्न,  hvx_सङ्कलय_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(सङ्कलय_f16, __fp16, hvx_सङ्कलय_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(सङ्कलय_f32, भिन्न,  hvx_सङ्कलय_f32_aaa, 1)

#परिभाषय DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_सङ्कलय_FN, HAS_सङ्कलय, IS_ROW_BCAST)                                                           \
स्थिर शून्य allreduce_thread_dma_2d_##SUFFIX(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {                                                   \
    संरचना htp_allreduce_context * actx = (संरचना htp_allreduce_context *) दत्त;                                                                  \
    संरचना htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    ध्रुव uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    ध्रुव uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    ध्रुव uint32_t ne0              = actx->ne0;                                                                                                  \
    ध्रुव uint32_t खण्डम्_पङ्क्तिकाः       = actx->खण्डम्_elems;                                                                                          \
    ध्रुव uint32_t row_आकार_aligned = actx->row_आकार_aligned;                                                                                     \
    ध्रुव uint32_t row_अष्टकानि        = ne0 * परिमाण(TYPE);                                                                                         \
                                                                                                                                                  \
    ध्रुव uint32_t dr  = actx->elems_प्रत्येकम्_thread;                                                                                                  \
    ध्रुव uint32_t r0  = actx->rank_elem_start + dr * ith;                                                                                        \
    ध्रुव uint32_t r1  = MIN(r0 + dr, actx->rank_elem_start + actx->rank_nelem);                                                                  \
    यदि (r0 >= r1) प्रत्यागम;                                                                                                                         \
                                                                                                                                                  \
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    ध्रुव size_t vtcm_thread_विस्थापनम् = ith * actx->vtcm_आकार_प्रत्येकम्_thread;                                                                           \
    uint8_t * dst_spad_आधारः = actx->dst_spad_आधारः + vtcm_thread_विस्थापनम्;                                                                           \
    uint8_t * res_spad_आधारः = HAS_सङ्कलय ? (IS_ROW_BCAST ? actx->res_spad_आधारः : (actx->res_spad_आधारः + vtcm_thread_विस्थापनम्)) : रिक्त;                 \
                                                                                                                                                  \
    ध्रुव size_t spad_half = actx->vtcm_आकार_प्रत्येकम्_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    पूर्णाङ्क spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    प्रति (पूर्णाङ्क k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_पङ्क्तिकाः = MIN(खण्डम्_पङ्क्तिकाः, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_आधारः + spad_idx * spad_half;                                                                                  \
        प्रति (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->दत्त + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(d_ddr, d_spad), octx->dsts[d]->nb[1], row_आकार_aligned, row_अष्टकानि, 0);                            \
        }                                                                                                                                         \
        प्रति (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_आधारः[s] + vtcm_thread_विस्थापनम् + spad_idx * spad_half;                                                \
            ध्रुव dma_addr_t s_ddr = octx->src[s]->दत्त + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s_spad, s_ddr), row_आकार_aligned, octx->src[s]->nb[1], row_अष्टकानि, cur_पङ्क्तिकाः);                      \
        }                                                                                                                                         \
        यदि (HAS_सङ्कलय && !IS_ROW_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_आधारः + spad_idx * spad_half;                                                                              \
            ध्रुव dma_addr_t r_ddr = octx->src[2 * n_ranks]->दत्त + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(r_spad, r_ddr), row_आकार_aligned, octx->src[2 * n_ranks]->nb[1], row_अष्टकानि, cur_पङ्क्तिकाः);            \
        }                                                                                                                                         \
        r_prefetch += cur_पङ्क्तिकाः;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    पूर्णाङ्क comp_spad_idx = 0;                                                                                                                        \
    प्रति (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_पङ्क्तिकाः = MIN(खण्डम्_पङ्क्तिकाः, r1 - r);                                                                                              \
        प्रति (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_queue_निष्कासय(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        प्रति (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_queue_निष्कासय(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        यदि (HAS_सङ्कलय && !IS_ROW_BCAST) {                                                                                                           \
            dma_queue_निष्कासय(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_आधारः + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_आधारः[0] + vtcm_thread_विस्थापनम् + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_आधारः[1] + vtcm_thread_विस्थापनम् + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_सङ्कलय && !IS_ROW_BCAST) ? (res_spad_आधारः + comp_spad_idx * spad_half) : रिक्त;                                      \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        प्रति (uint32_t row = 0; row < cur_पङ्क्तिकाः; row++) {                                                                                           \
            uint8_t * d_row = d_spad + row * row_आकार_aligned;                                                                                    \
            ध्रुव uint8_t * s0_row = s0_spad + row * row_आकार_aligned;                                                                            \
            ध्रुव uint8_t * s1_row = s1_spad + row * row_आकार_aligned;                                                                            \
            HVX_सङ्कलय_FN(d_row, s0_row, s1_row, ne0);                                                                                               \
            प्रति (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                ध्रुव uint8_t * ss_row = actx->src_spad_आधारः[s] + vtcm_thread_विस्थापनम् + comp_spad_idx * spad_half + row * row_आकार_aligned;        \
                HVX_सङ्कलय_FN(d_row, d_row, ss_row, ne0);                                                                                            \
            }                                                                                                                                     \
            यदि (HAS_सङ्कलय) {                                                                                                                        \
                ध्रुव uint8_t * res_row = IS_ROW_BCAST ? res_spad_आधारः : (r_spad + row * row_आकार_aligned);                                       \
                HVX_सङ्कलय_FN(d_row, d_row, res_row, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        प्रति (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->दत्त + r * octx->dsts[d]->nb[1];                                                                    \
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(d_ddr, d_spad), octx->dsts[d]->nb[1], row_आकार_aligned, row_अष्टकानि, cur_पङ्क्तिकाः);                     \
        }                                                                                                                                         \
        यदि (r_prefetch < r1) {                                                                                                                    \
            uint32_t अग्र_पङ्क्तिकाः = MIN(खण्डम्_पङ्क्तिकाः, r1 - r_prefetch);                                                                                \
            प्रति (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_आधारः[s] + vtcm_thread_विस्थापनम् + comp_spad_idx * spad_half;                                       \
                ध्रुव dma_addr_t s_अग्र = octx->src[s]->दत्त + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s_spad, s_अग्र), row_आकार_aligned, octx->src[s]->nb[1], row_अष्टकानि, अग्र_पङ्क्तिकाः);                \
            }                                                                                                                                     \
            यदि (HAS_सङ्कलय && !IS_ROW_BCAST) {                                                                                                       \
                uint8_t * r_spad_अग्र = res_spad_आधारः + comp_spad_idx * spad_half;                                                                \
                ध्रुव dma_addr_t r_अग्र = octx->src[2 * n_ranks]->दत्त + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(r_spad_अग्र, r_अग्र), row_आकार_aligned, octx->src[2 * n_ranks]->nb[1], row_अष्टकानि, अग्र_पङ्क्तिकाः); \
            }                                                                                                                                     \
            r_prefetch += अग्र_पङ्क्तिकाः;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_पङ्क्तिकाः;                                                                                                                            \
    }                                                                                                                                             \
    dma_queue_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_सङ्कलय_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           भिन्न,  hvx_सङ्कलय_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(सङ्कलय_f16,       __fp16, hvx_सङ्कलय_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(सङ्कलय_f32,       भिन्न,  hvx_सङ्कलय_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(सङ्कलय_bcast_f16, __fp16, hvx_सङ्कलय_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(सङ्कलय_bcast_f32, भिन्न,  hvx_सङ्कलय_f32_aaa, 1, 1)

स्थिर पूर्णाङ्क validate_allreduce(
    संरचना htp_ops_context * octx,
    ध्रुव संरचना htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    यदि (kparams->n_तन्तवः == 0 || (uint32_t) kparams->n_तन्तवः > octx->ctx->n_तन्तवः) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }
    octx->n_तन्तवः = (uint32_t) kparams->n_तन्तवः;

    यदि (kparams->vtcm_आकार_प्रत्येकम्_thread <= 0 || kparams->vtcm_आकार <= 0) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    ध्रुव bool has_सङ्कलय = (octx->op == HTP_OP_ALLREDUCE_सङ्कलय);
    ध्रुव size_t n_vtcm_buffers = htp_allreduce_vtcm_बफर_गणनम्(
        n_ranks, octx->n_तन्तवः, has_सङ्कलय, kparams->is_row_bcast != 0);
    ध्रुव size_t vtcm_आकार = n_vtcm_buffers * (size_t) kparams->vtcm_आकार_प्रत्येकम्_thread;
    यदि (vtcm_आकार != (size_t) kparams->vtcm_आकार) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }
    यदि (vtcm_आकार > octx->ctx->vtcm_आकार) {
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    यदि (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

पूर्णाङ्क op_allreduce(संरचना htp_ops_context * octx) {
    यदि (octx->ctx->mdev.गणनम् > 1 && octx->ctx->mdev.idx > 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव संरचना htp_allreduce_kernel_params * kparams = (ध्रुव संरचना htp_allreduce_kernel_params *) octx->kernel_params;
    ध्रुव संरचना htp_tensor * dst = octx->dst;

    ध्रुव uint32_t rank    = (uint32_t) kparams->rank;
    ध्रुव uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    यदि (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    ध्रुव uint32_t fence_seq_प्रविष्टिः = (uint32_t) octx->op_params[0];
    ध्रुव uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    ध्रुव संरचना htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->दत्त;

    ध्रुव पूर्णाङ्क स्थितिः = validate_allreduce(octx, kparams, n_ranks);
    यदि (स्थितिः != HTP_स्थितिः_OK) {
        यदि (स्थितिः == HTP_स्थितिः_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, स्थितिः);
        प्रत्यागम स्थितिः;
    }

    ध्रुव bool has_सङ्कलय = (octx->op == HTP_OP_ALLREDUCE_सङ्कलय);
    ध्रुव uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ध्रुव int32_t  बहुलकम्  = kparams->बहुलकम्;

    // 1. Entry Barrier: Synchronize all ranks before reading
    संरचना htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_प्रविष्टिः);

    htp_fence_write(my_fence, fence_seq_प्रविष्टिः, octx->स्थितिः);

    प्रति (uint32_t j = 0; j < n_ranks; j++) {
        यदि (j == rank) अनुवर्त;
        ध्रुव संरचना htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->दत्त;
        uint64_t spins = 0;
        यावत् (1) {
            uint32_t peer_seq;
            uint32_t peer_स्थितिः;
            htp_fence_read(peer_fence, &peer_seq, &peer_स्थितिः);
            यदि ((int32_t)(peer_seq - fence_seq_प्रविष्टिः) >= 0) {
                यदि (peer_स्थितिः > HTP_स्थितिः_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_स्थितिः);
                    htp_fence_write(my_fence, fence_seq_exit, peer_स्थितिः);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_प्रविष्टिः);
                    प्रत्यागम peer_स्थितिः;
                }
                भङ्ग;
            }
            यदि (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_प्रविष्टिः, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_स्थितिः_आन्तरिकम्_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_प्रविष्टिः);
                प्रत्यागम HTP_स्थितिः_आन्तरिकम्_ERR;
            }
            hex_pause();
        }
    }
    asm चञ्चल ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_प्रविष्टिः);

    // 2. Multi-threaded Reduction across assigned rank chunk
    यदि (nelem > 0) {
        ध्रुव uint32_t n_तन्तवः            = (uint32_t) kparams->n_तन्तवः;
        ध्रुव uint32_t खण्डम्_elems          = (uint32_t) kparams->खण्डम्_elems;
        ध्रुव uint32_t elems_प्रत्येकम्_thread     = (uint32_t) kparams->elems_प्रत्येकम्_thread;
        ध्रुव uint32_t vtcm_आकार_प्रत्येकम्_thread = (uint32_t) kparams->vtcm_आकार_प्रत्येकम्_thread;

        संरचना htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.row_आकार_aligned     = (uint32_t) kparams->row_आकार_aligned;
        actx.rank_elem_start      = (uint32_t) kparams->rank_elem_start;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_प्रत्येकम्_thread     = elems_प्रत्येकम्_thread;
        actx.खण्डम्_elems          = खण्डम्_elems;
        actx.vtcm_आकार_प्रत्येकम्_thread = vtcm_आकार_प्रत्येकम्_thread;
        actx.is_row_bcast         = (kparams->is_row_bcast != 0);

        work_queue_func_t reduce_fun = रिक्त;
        विकल्प (kparams->kernel_type) {
            पक्ष HTP_ALLREDUCE_KERNEL_DMA_1D:
                यदि (has_सङ्कलय) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_सङ्कलय_f16 : allreduce_thread_dma_1d_सङ्कलय_f32;
                } अन्यथा {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                भङ्ग;
            पक्ष HTP_ALLREDUCE_KERNEL_DMA_2D:
                यदि (has_सङ्कलय) {
                    यदि (kparams->is_row_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_सङ्कलय_bcast_f16 : allreduce_thread_dma_2d_सङ्कलय_bcast_f32;
                    } अन्यथा {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_सङ्कलय_f16 : allreduce_thread_dma_2d_सङ्कलय_f32;
                    }
                } अन्यथा {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                भङ्ग;
            स्वभाव:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_स्थितिः_NO_SUPPORT);
                प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_आधारः;
        प्रति (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_आधारः[s] = vtcm_ptr;
            vtcm_ptr += n_तन्तवः * vtcm_आकार_प्रत्येकम्_thread;
        }
        actx.dst_spad_आधारः = vtcm_ptr;
        vtcm_ptr += n_तन्तवः * vtcm_आकार_प्रत्येकम्_thread;
        यदि (has_सङ्कलय) {
            actx.res_spad_आधारः = vtcm_ptr;
            vtcm_ptr += (actx.is_row_bcast ? 1 : n_तन्तवः) * vtcm_आकार_प्रत्येकम्_thread;
        }

        यदि (has_सङ्कलय && actx.is_row_bcast) {
            ध्रुव dma_addr_t r_ddr = octx->src[2 * n_ranks]->दत्त;
            ध्रुव uint32_t row_अष्टकानि = actx.ne0 * (dst->type == HTP_TYPE_F16 ? परिमाण(__fp16) : परिमाण(भिन्न));
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(actx.res_spad_आधारः, r_ddr), actx.row_आकार_aligned, 0, row_अष्टकानि, 1);
            dma_queue_निष्कासय(dma_q);
        }

        work_queue_सततम्(octx->ctx->work_queue, reduce_fun, &actx, n_तन्तवः);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    यदि (बहुलकम् == HTP_ALLREDUCE_SHARDED_FANOUT) {
        asm चञ्चल ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->स्थितिः);

    प्रति (uint32_t j = 0; j < n_ranks; j++) {
        यदि (j == rank) अनुवर्त;
        ध्रुव संरचना htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->दत्त;
        uint64_t spins = 0;
        यावत् (1) {
            uint32_t peer_seq;
            uint32_t peer_स्थितिः;
            htp_fence_read(peer_fence, &peer_seq, &peer_स्थितिः);
            यदि ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                यदि (peer_स्थितिः > HTP_स्थितिः_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_स्थितिः);
                    htp_fence_write(my_fence, fence_seq_exit, peer_स्थितिः);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    प्रत्यागम peer_स्थितिः;
                }
                भङ्ग;
            }
            यदि (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_स्थितिः_आन्तरिकम्_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                प्रत्यागम HTP_स्थितिः_आन्तरिकम्_ERR;
            }
            hex_pause();
        }
    }
    asm चञ्चल ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    प्रत्यागम octx->स्थितिः;
}
