#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>
#అన్తర్భావ <stdatomic.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "allreduce-ops.h"
#అన్తర్భావ "htp-fence.h"

నిర్మాణం htp_allreduce_context {
    నిర్మాణం htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t n_dsts;
    uint32_t nelem;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t అడ్డువరుస_కొలత_aligned;
    uint32_t rank_elem_ప్రారంభం;
    uint32_t rank_nelem;
    uint32_t elems_చొప్పున_thread;
    uint32_t ఖండం_elems;
    uint32_t vtcm_కొలత_చొప్పున_thread;
    bool     is_అడ్డువరుస_bcast;
    uint8_t * src_spad_పునాది[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_పునాది;
    uint8_t * res_spad_పునాది;
};

#నిర్వచించు DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_కూడు_FN, HAS_కూడు)                                         \
స్థిర శూన్య allreduce_thread_dma_1d_##SUFFIX(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {                   \
    నిర్మాణం htp_allreduce_context * actx = (నిర్మాణం htp_allreduce_context *) దత్తాంశం;                                  \
    నిర్మాణం htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    స్థిరరాశి uint32_t n_ranks     = actx->n_ranks;                                                                   \
    స్థిరరాశి uint32_t n_dsts      = actx->n_dsts;                                                                    \
    స్థిరరాశి uint32_t ఖండం_elems = actx->ఖండం_elems;                                                               \
                                                                                                                  \
    స్థిరరాశి uint32_t dr  = actx->elems_చొప్పున_thread;                                                                  \
    స్థిరరాశి uint32_t ir0 = actx->rank_elem_ప్రారంభం + dr * ith;                                                        \
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_ప్రారంభం + actx->rank_nelem);                                 \
    ఒకవేళ (ir0 >= ir1) తిరిగి;                                                                                       \
                                                                                                                  \
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_వరుస * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    స్థిరరాశి size_t vtcm_thread_స్థానభ్రంశం = ith * actx->vtcm_కొలత_చొప్పున_thread;                                           \
    uint8_t * dst_spad_పునాది = actx->dst_spad_పునాది + vtcm_thread_స్థానభ్రంశం;                                           \
    uint8_t * res_spad_పునాది = HAS_కూడు ? (actx->res_spad_పునాది + vtcm_thread_స్థానభ్రంశం) : ఖాళీ;                        \
                                                                                                                  \
    స్థిరరాశి size_t spad_half = actx->vtcm_కొలత_చొప్పున_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    పూర్ణాంకం spad_idx = 0;                                                                                             \
                                                                                                                  \
    కొరకు (పూర్ణాంకం k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(ఖండం_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_బైట్లు = cur_elems * పరిమాణం(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_పునాది + spad_idx * spad_half;                                                  \
        కొరకు (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->దత్తాంశం + ir_prefetch * పరిమాణం(TYPE);                                  \
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(d_ddr, d_spad), cur_బైట్లు, cur_బైట్లు, cur_బైట్లు, 0);              \
        }                                                                                                         \
        కొరకు (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_పునాది[s] + vtcm_thread_స్థానభ్రంశం + spad_idx * spad_half;                \
            స్థిరరాశి dma_addr_t s_ddr = octx->src[s]->దత్తాంశం + ir_prefetch * పరిమాణం(TYPE);                             \
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_spad, s_ddr), cur_బైట్లు, cur_బైట్లు, cur_బైట్లు, 1);              \
        }                                                                                                         \
        ఒకవేళ (HAS_కూడు) {                                                                                            \
            uint8_t * r_spad = res_spad_పునాది + spad_idx * spad_half;                                              \
            స్థిరరాశి dma_addr_t r_ddr = octx->src[2 * n_ranks]->దత్తాంశం + ir_prefetch * పరిమాణం(TYPE);                   \
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(r_spad, r_ddr), cur_బైట్లు, cur_బైట్లు, cur_బైట్లు, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    పూర్ణాంకం comp_spad_idx = 0;                                                                                        \
    కొరకు (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(ఖండం_elems, ir1 - ir);                                                          \
        size_t   cur_బైట్లు = cur_elems * పరిమాణం(TYPE);                                                            \
        కొరకు (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_వరుస_తీయి(dma_q);                                                                                 \
        }                                                                                                         \
        కొరకు (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_వరుస_తీయి(dma_q);                                                                                 \
        }                                                                                                         \
        ఒకవేళ (HAS_కూడు) {                                                                                            \
            dma_వరుస_తీయి(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_పునాది + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_పునాది[0] + vtcm_thread_స్థానభ్రంశం + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_పునాది[1] + vtcm_thread_స్థానభ్రంశం + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_కూడు ? (res_spad_పునాది + comp_spad_idx * spad_half) : ఖాళీ;                         \
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_కూడు_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        కొరకు (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_పునాది[s] + vtcm_thread_స్థానభ్రంశం + comp_spad_idx * spad_half;          \
            HVX_కూడు_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        ఒకవేళ (HAS_కూడు) {                                                                                            \
            HVX_కూడు_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        కొరకు (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->దత్తాంశం + ir * పరిమాణం(TYPE);                                           \
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(d_ddr, d_spad), cur_బైట్లు, cur_బైట్లు, cur_బైట్లు, 1);              \
        }                                                                                                         \
        ఒకవేళ (ir_prefetch < ir1) {                                                                                  \
            uint32_t తదుపరి_elems = MIN(ఖండం_elems, ir1 - ir_prefetch);                                            \
            size_t   తదుపరి_బైట్లు = తదుపరి_elems * పరిమాణం(TYPE);                                                      \
            కొరకు (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_పునాది[s] + vtcm_thread_స్థానభ్రంశం + comp_spad_idx * spad_half;       \
                స్థిరరాశి dma_addr_t s_తదుపరి = octx->src[s]->దత్తాంశం + ir_prefetch * పరిమాణం(TYPE);                        \
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_spad, s_తదుపరి), తదుపరి_బైట్లు, తదుపరి_బైట్లు, తదుపరి_బైట్లు, 1);      \
            }                                                                                                     \
            ఒకవేళ (HAS_కూడు) {                                                                                        \
                uint8_t * r_spad_తదుపరి = res_spad_పునాది + comp_spad_idx * spad_half;                                \
                స్థిరరాశి dma_addr_t r_తదుపరి = octx->src[2 * n_ranks]->దత్తాంశం + ir_prefetch * పరిమాణం(TYPE);              \
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(r_spad_తదుపరి, r_తదుపరి), తదుపరి_బైట్లు, తదుపరి_బైట్లు, తదుపరి_బైట్లు, 1); \
            }                                                                                                     \
            ir_prefetch += తదుపరి_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_వరుస_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_కూడు_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     దశాంశం,  hvx_కూడు_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(కూడు_f16, __fp16, hvx_కూడు_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(కూడు_f32, దశాంశం,  hvx_కూడు_f32_aaa, 1)

#నిర్వచించు DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_కూడు_FN, HAS_కూడు, IS_అడ్డువరుస_BCAST)                                                           \
స్థిర శూన్య allreduce_thread_dma_2d_##SUFFIX(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {                                                   \
    నిర్మాణం htp_allreduce_context * actx = (నిర్మాణం htp_allreduce_context *) దత్తాంశం;                                                                  \
    నిర్మాణం htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    స్థిరరాశి uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    స్థిరరాశి uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    స్థిరరాశి uint32_t ne0              = actx->ne0;                                                                                                  \
    స్థిరరాశి uint32_t ఖండం_అడ్డువరుసలు       = actx->ఖండం_elems;                                                                                          \
    స్థిరరాశి uint32_t అడ్డువరుస_కొలత_aligned = actx->అడ్డువరుస_కొలత_aligned;                                                                                     \
    స్థిరరాశి uint32_t అడ్డువరుస_బైట్లు        = ne0 * పరిమాణం(TYPE);                                                                                         \
                                                                                                                                                  \
    స్థిరరాశి uint32_t dr  = actx->elems_చొప్పున_thread;                                                                                                  \
    స్థిరరాశి uint32_t r0  = actx->rank_elem_ప్రారంభం + dr * ith;                                                                                        \
    స్థిరరాశి uint32_t r1  = MIN(r0 + dr, actx->rank_elem_ప్రారంభం + actx->rank_nelem);                                                                  \
    ఒకవేళ (r0 >= r1) తిరిగి;                                                                                                                         \
                                                                                                                                                  \
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_వరుస * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    స్థిరరాశి size_t vtcm_thread_స్థానభ్రంశం = ith * actx->vtcm_కొలత_చొప్పున_thread;                                                                           \
    uint8_t * dst_spad_పునాది = actx->dst_spad_పునాది + vtcm_thread_స్థానభ్రంశం;                                                                           \
    uint8_t * res_spad_పునాది = HAS_కూడు ? (IS_అడ్డువరుస_BCAST ? actx->res_spad_పునాది : (actx->res_spad_పునాది + vtcm_thread_స్థానభ్రంశం)) : ఖాళీ;                 \
                                                                                                                                                  \
    స్థిరరాశి size_t spad_half = actx->vtcm_కొలత_చొప్పున_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    పూర్ణాంకం spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    కొరకు (పూర్ణాంకం k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_అడ్డువరుసలు = MIN(ఖండం_అడ్డువరుసలు, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_పునాది + spad_idx * spad_half;                                                                                  \
        కొరకు (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->దత్తాంశం + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(d_ddr, d_spad), octx->dsts[d]->nb[1], అడ్డువరుస_కొలత_aligned, అడ్డువరుస_బైట్లు, 0);                            \
        }                                                                                                                                         \
        కొరకు (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_పునాది[s] + vtcm_thread_స్థానభ్రంశం + spad_idx * spad_half;                                                \
            స్థిరరాశి dma_addr_t s_ddr = octx->src[s]->దత్తాంశం + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_spad, s_ddr), అడ్డువరుస_కొలత_aligned, octx->src[s]->nb[1], అడ్డువరుస_బైట్లు, cur_అడ్డువరుసలు);                      \
        }                                                                                                                                         \
        ఒకవేళ (HAS_కూడు && !IS_అడ్డువరుస_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_పునాది + spad_idx * spad_half;                                                                              \
            స్థిరరాశి dma_addr_t r_ddr = octx->src[2 * n_ranks]->దత్తాంశం + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(r_spad, r_ddr), అడ్డువరుస_కొలత_aligned, octx->src[2 * n_ranks]->nb[1], అడ్డువరుస_బైట్లు, cur_అడ్డువరుసలు);            \
        }                                                                                                                                         \
        r_prefetch += cur_అడ్డువరుసలు;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    పూర్ణాంకం comp_spad_idx = 0;                                                                                                                        \
    కొరకు (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_అడ్డువరుసలు = MIN(ఖండం_అడ్డువరుసలు, r1 - r);                                                                                              \
        కొరకు (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_వరుస_తీయి(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        కొరకు (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_వరుస_తీయి(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        ఒకవేళ (HAS_కూడు && !IS_అడ్డువరుస_BCAST) {                                                                                                           \
            dma_వరుస_తీయి(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_పునాది + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_పునాది[0] + vtcm_thread_స్థానభ్రంశం + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_పునాది[1] + vtcm_thread_స్థానభ్రంశం + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_కూడు && !IS_అడ్డువరుస_BCAST) ? (res_spad_పునాది + comp_spad_idx * spad_half) : ఖాళీ;                                      \
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        కొరకు (uint32_t అడ్డువరుస = 0; అడ్డువరుస < cur_అడ్డువరుసలు; అడ్డువరుస++) {                                                                                           \
            uint8_t * d_అడ్డువరుస = d_spad + అడ్డువరుస * అడ్డువరుస_కొలత_aligned;                                                                                    \
            స్థిరరాశి uint8_t * s0_అడ్డువరుస = s0_spad + అడ్డువరుస * అడ్డువరుస_కొలత_aligned;                                                                            \
            స్థిరరాశి uint8_t * s1_అడ్డువరుస = s1_spad + అడ్డువరుస * అడ్డువరుస_కొలత_aligned;                                                                            \
            HVX_కూడు_FN(d_అడ్డువరుస, s0_అడ్డువరుస, s1_అడ్డువరుస, ne0);                                                                                               \
            కొరకు (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                స్థిరరాశి uint8_t * ss_అడ్డువరుస = actx->src_spad_పునాది[s] + vtcm_thread_స్థానభ్రంశం + comp_spad_idx * spad_half + అడ్డువరుస * అడ్డువరుస_కొలత_aligned;        \
                HVX_కూడు_FN(d_అడ్డువరుస, d_అడ్డువరుస, ss_అడ్డువరుస, ne0);                                                                                            \
            }                                                                                                                                     \
            ఒకవేళ (HAS_కూడు) {                                                                                                                        \
                స్థిరరాశి uint8_t * res_అడ్డువరుస = IS_అడ్డువరుస_BCAST ? res_spad_పునాది : (r_spad + అడ్డువరుస * అడ్డువరుస_కొలత_aligned);                                       \
                HVX_కూడు_FN(d_అడ్డువరుస, d_అడ్డువరుస, res_అడ్డువరుస, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        కొరకు (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->దత్తాంశం + r * octx->dsts[d]->nb[1];                                                                    \
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(d_ddr, d_spad), octx->dsts[d]->nb[1], అడ్డువరుస_కొలత_aligned, అడ్డువరుస_బైట్లు, cur_అడ్డువరుసలు);                     \
        }                                                                                                                                         \
        ఒకవేళ (r_prefetch < r1) {                                                                                                                    \
            uint32_t తదుపరి_అడ్డువరుసలు = MIN(ఖండం_అడ్డువరుసలు, r1 - r_prefetch);                                                                                \
            కొరకు (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_పునాది[s] + vtcm_thread_స్థానభ్రంశం + comp_spad_idx * spad_half;                                       \
                స్థిరరాశి dma_addr_t s_తదుపరి = octx->src[s]->దత్తాంశం + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_spad, s_తదుపరి), అడ్డువరుస_కొలత_aligned, octx->src[s]->nb[1], అడ్డువరుస_బైట్లు, తదుపరి_అడ్డువరుసలు);                \
            }                                                                                                                                     \
            ఒకవేళ (HAS_కూడు && !IS_అడ్డువరుస_BCAST) {                                                                                                       \
                uint8_t * r_spad_తదుపరి = res_spad_పునాది + comp_spad_idx * spad_half;                                                                \
                స్థిరరాశి dma_addr_t r_తదుపరి = octx->src[2 * n_ranks]->దత్తాంశం + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(r_spad_తదుపరి, r_తదుపరి), అడ్డువరుస_కొలత_aligned, octx->src[2 * n_ranks]->nb[1], అడ్డువరుస_బైట్లు, తదుపరి_అడ్డువరుసలు); \
            }                                                                                                                                     \
            r_prefetch += తదుపరి_అడ్డువరుసలు;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_అడ్డువరుసలు;                                                                                                                            \
    }                                                                                                                                             \
    dma_వరుస_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_కూడు_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           దశాంశం,  hvx_కూడు_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(కూడు_f16,       __fp16, hvx_కూడు_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(కూడు_f32,       దశాంశం,  hvx_కూడు_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(కూడు_bcast_f16, __fp16, hvx_కూడు_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(కూడు_bcast_f32, దశాంశం,  hvx_కూడు_f32_aaa, 1, 1)

స్థిర పూర్ణాంకం validate_allreduce(
    నిర్మాణం htp_ops_context * octx,
    స్థిరరాశి నిర్మాణం htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    ఒకవేళ (kparams->n_దారాలు == 0 || (uint32_t) kparams->n_దారాలు > octx->ctx->n_దారాలు) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }
    octx->n_దారాలు = (uint32_t) kparams->n_దారాలు;

    ఒకవేళ (kparams->vtcm_కొలత_చొప్పున_thread <= 0 || kparams->vtcm_కొలత <= 0) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి bool has_కూడు = (octx->op == HTP_OP_ALLREDUCE_కూడు);
    స్థిరరాశి size_t n_vtcm_buffers = htp_allreduce_vtcm_బఫరు_లెక్క(
        n_ranks, octx->n_దారాలు, has_కూడు, kparams->is_అడ్డువరుస_bcast != 0);
    స్థిరరాశి size_t vtcm_కొలత = n_vtcm_buffers * (size_t) kparams->vtcm_కొలత_చొప్పున_thread;
    ఒకవేళ (vtcm_కొలత != (size_t) kparams->vtcm_కొలత) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }
    ఒకవేళ (vtcm_కొలత > octx->ctx->vtcm_కొలత) {
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    ఒకవేళ (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_allreduce(నిర్మాణం htp_ops_context * octx) {
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1 && octx->ctx->mdev.idx > 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి నిర్మాణం htp_allreduce_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_allreduce_kernel_params *) octx->kernel_params;
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;

    స్థిరరాశి uint32_t rank    = (uint32_t) kparams->rank;
    స్థిరరాశి uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    ఒకవేళ (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి uint32_t fence_seq_నమోదు = (uint32_t) octx->op_params[0];
    స్థిరరాశి uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    స్థిరరాశి నిర్మాణం htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->దత్తాంశం;

    స్థిరరాశి పూర్ణాంకం స్థితి_2 = validate_allreduce(octx, kparams, n_ranks);
    ఒకవేళ (స్థితి_2 != HTP_స్థితి_OK) {
        ఒకవేళ (స్థితి_2 == HTP_స్థితి_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, స్థితి_2);
        తిరిగి స్థితి_2;
    }

    స్థిరరాశి bool has_కూడు = (octx->op == HTP_OP_ALLREDUCE_కూడు);
    స్థిరరాశి uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    స్థిరరాశి int32_t  బహుళకం  = kparams->బహుళకం;

    // 1. Entry Barrier: Synchronize all ranks before reading
    నిర్మాణం htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_నమోదు);

    htp_fence_write(my_fence, fence_seq_నమోదు, octx->స్థితి_2);

    కొరకు (uint32_t j = 0; j < n_ranks; j++) {
        ఒకవేళ (j == rank) కొనసాగు;
        స్థిరరాశి నిర్మాణం htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->దత్తాంశం;
        uint64_t spins = 0;
        అయితే (1) {
            uint32_t peer_seq;
            uint32_t peer_స్థితి;
            htp_fence_read(peer_fence, &peer_seq, &peer_స్థితి);
            ఒకవేళ ((int32_t)(peer_seq - fence_seq_నమోదు) >= 0) {
                ఒకవేళ (peer_స్థితి > HTP_స్థితి_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_స్థితి);
                    htp_fence_write(my_fence, fence_seq_exit, peer_స్థితి);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_నమోదు);
                    తిరిగి peer_స్థితి;
                }
                ఆపు;
            }
            ఒకవేళ (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_నమోదు, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_స్థితి_అంతర్గత_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_నమోదు);
                తిరిగి HTP_స్థితి_అంతర్గత_ERR;
            }
            hex_pause();
        }
    }
    asm అస్థిర ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_నమోదు);

    // 2. Multi-threaded Reduction across assigned rank chunk
    ఒకవేళ (nelem > 0) {
        స్థిరరాశి uint32_t n_దారాలు            = (uint32_t) kparams->n_దారాలు;
        స్థిరరాశి uint32_t ఖండం_elems          = (uint32_t) kparams->ఖండం_elems;
        స్థిరరాశి uint32_t elems_చొప్పున_thread     = (uint32_t) kparams->elems_చొప్పున_thread;
        స్థిరరాశి uint32_t vtcm_కొలత_చొప్పున_thread = (uint32_t) kparams->vtcm_కొలత_చొప్పున_thread;

        నిర్మాణం htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.అడ్డువరుస_కొలత_aligned     = (uint32_t) kparams->అడ్డువరుస_కొలత_aligned;
        actx.rank_elem_ప్రారంభం      = (uint32_t) kparams->rank_elem_ప్రారంభం;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_చొప్పున_thread     = elems_చొప్పున_thread;
        actx.ఖండం_elems          = ఖండం_elems;
        actx.vtcm_కొలత_చొప్పున_thread = vtcm_కొలత_చొప్పున_thread;
        actx.is_అడ్డువరుస_bcast         = (kparams->is_అడ్డువరుస_bcast != 0);

        work_వరుస_func_t reduce_fun = ఖాళీ;
        ఎంపిక (kparams->kernel_type) {
            స్థితి HTP_ALLREDUCE_KERNEL_DMA_1D:
                ఒకవేళ (has_కూడు) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_కూడు_f16 : allreduce_thread_dma_1d_కూడు_f32;
                } లేకపోతే {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                ఆపు;
            స్థితి HTP_ALLREDUCE_KERNEL_DMA_2D:
                ఒకవేళ (has_కూడు) {
                    ఒకవేళ (kparams->is_అడ్డువరుస_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_కూడు_bcast_f16 : allreduce_thread_dma_2d_కూడు_bcast_f32;
                    } లేకపోతే {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_కూడు_f16 : allreduce_thread_dma_2d_కూడు_f32;
                    }
                } లేకపోతే {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                ఆపు;
            అప్రమేయ:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_స్థితి_NO_SUPPORT);
                తిరిగి HTP_స్థితి_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_పునాది;
        కొరకు (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_పునాది[s] = vtcm_ptr;
            vtcm_ptr += n_దారాలు * vtcm_కొలత_చొప్పున_thread;
        }
        actx.dst_spad_పునాది = vtcm_ptr;
        vtcm_ptr += n_దారాలు * vtcm_కొలత_చొప్పున_thread;
        ఒకవేళ (has_కూడు) {
            actx.res_spad_పునాది = vtcm_ptr;
            vtcm_ptr += (actx.is_అడ్డువరుస_bcast ? 1 : n_దారాలు) * vtcm_కొలత_చొప్పున_thread;
        }

        ఒకవేళ (has_కూడు && actx.is_అడ్డువరుస_bcast) {
            స్థిరరాశి dma_addr_t r_ddr = octx->src[2 * n_ranks]->దత్తాంశం;
            స్థిరరాశి uint32_t అడ్డువరుస_బైట్లు = actx.ne0 * (dst->type == HTP_TYPE_F16 ? పరిమాణం(__fp16) : పరిమాణం(దశాంశం));
            dma_వరుస * dma_q = octx->ctx->dma[0];
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(actx.res_spad_పునాది, r_ddr), actx.అడ్డువరుస_కొలత_aligned, 0, అడ్డువరుస_బైట్లు, 1);
            dma_వరుస_తీయి(dma_q);
        }

        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, reduce_fun, &actx, n_దారాలు);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_ప్రారంభం(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    ఒకవేళ (బహుళకం == HTP_ALLREDUCE_SHARDED_FANOUT) {
        asm అస్థిర ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->స్థితి_2);

    కొరకు (uint32_t j = 0; j < n_ranks; j++) {
        ఒకవేళ (j == rank) కొనసాగు;
        స్థిరరాశి నిర్మాణం htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->దత్తాంశం;
        uint64_t spins = 0;
        అయితే (1) {
            uint32_t peer_seq;
            uint32_t peer_స్థితి;
            htp_fence_read(peer_fence, &peer_seq, &peer_స్థితి);
            ఒకవేళ ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                ఒకవేళ (peer_స్థితి > HTP_స్థితి_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_స్థితి);
                    htp_fence_write(my_fence, fence_seq_exit, peer_స్థితి);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    తిరిగి peer_స్థితి;
                }
                ఆపు;
            }
            ఒకవేళ (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_స్థితి_అంతర్గత_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                తిరిగి HTP_స్థితి_అంతర్గత_ERR;
            }
            hex_pause();
        }
    }
    asm అస్థిర ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    తిరిగి octx->స్థితి_2;
}
