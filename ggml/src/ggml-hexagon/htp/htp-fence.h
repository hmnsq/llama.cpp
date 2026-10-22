#నిర్వచితం_కాకపోతే HTP_FENCE_H
#నిర్వచించు HTP_FENCE_H

#అన్తర్భావ <stdatomic.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ <HAP_farf.h>

#అన్తర్భావ "hex-utils.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-ctx.h"

స్థిర అంతఃవరుస atomic_uint * htp_mdev_fence_స్థానం(స్థిరరాశి శూన్య * fence_పునాది, uint32_t idx) {
    తిరిగి (atomic_uint *) ((స్థిరరాశి uint8_t *) fence_పునాది + (size_t) idx * HTP_FENCE_స్థానం_కొలత);
}

స్థిర అంతఃవరుస శూన్య htp_fence_write(శూన్య * fence_ptr, uint32_t seq, uint32_t స్థితి_2) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], స్థితి_2);
    atomic_store(&fence[0], seq);
    asm అస్థిర ("syncht" : : : "memory");
    Q6_dccleaninva_A((శూన్య *) fence);
}

స్థిర అంతఃవరుస శూన్య htp_fence_read(స్థిరరాశి శూన్య * fence_ptr, uint32_t * seq, uint32_t * స్థితి_2) {
    స్థిరరాశి atomic_uint * fence = (స్థిరరాశి atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((శూన్య *) fence);
    asm అస్థిర ("syncht" : : : "memory");
    *seq = atomic_ఎక్కించు(&fence[0]);
    *స్థితి_2 = atomic_ఎక్కించు(&fence[1]);
}

స్థిర అంతఃవరుస శూన్య htp_mdev_సమూహం_barrier(నిర్మాణం htp_ops_context * octx) {
    నిర్మాణం htp_context * ctx = octx->ctx;
    ఒకవేళ (ctx->mdev.లెక్క <= 1) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t seq = ++ctx->mdev.fence_seq;

    నిర్మాణం htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    స్థిరరాశి uint32_t mdev_idx   = ctx->mdev.idx;
    స్థిరరాశి uint32_t mdev_లెక్క = ctx->mdev.లెక్క;

    uint8_t * fence_పునాది   = ctx->mdev.fence_పునాది;
    atomic_uint * my_fence = htp_mdev_fence_స్థానం(fence_పునాది, mdev_idx);
    htp_fence_write(my_fence, seq, octx->స్థితి_2);

    కొరకు (uint32_t d = 0; d < mdev_లెక్క; d++) {
        ఒకవేళ (d == mdev_idx) కొనసాగు;
        atomic_uint * peer_fence = htp_mdev_fence_స్థానం(fence_పునాది, d);
        uint64_t spins = 0;
        అయితే (1) {
            uint32_t peer_seq;
            uint32_t peer_స్థితి;
            htp_fence_read(peer_fence, &peer_seq, &peer_స్థితి);
            ఒకవేళ ((int32_t)(peer_seq - seq) >= 0) {
                ఒకవేళ (peer_స్థితి > HTP_స్థితి_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_స్థితి, seq);
                    htp_ops_context_సమితి_స్థితి(octx, peer_స్థితి);
                }
                ఆపు;
            }
            ఒకవేళ (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            ఒకవేళ (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_సమితి_స్థితి(octx, HTP_స్థితి_అంతర్గత_ERR);
                ఆపు;
            }
            hex_pause();
        }
    }
    asm అస్థిర ("syncht" : : : "memory");

    ఒకవేళ (octx->స్థితి_2 > HTP_స్థితి_OK) {
        htp_fence_write(my_fence, seq, octx->స్థితి_2);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#అయితే_ముగింపు // HTP_FENCE_H
