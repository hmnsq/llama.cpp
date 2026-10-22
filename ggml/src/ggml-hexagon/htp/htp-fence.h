#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#ಅನ್ತರ್ಭಾವ <stdatomic.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ atomic_uint * htp_mdev_fence_slot(ಧ್ರುವ ಶೂನ್ಯ * fence_base, uint32_t idx) {
    ಪ್ರತ್ಯಾಗಮ (atomic_uint *) ((ಧ್ರುವ uint8_t *) fence_base + (size_t) idx * HTP_FENCE_SLOT_SIZE);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ htp_fence_write(ಶೂನ್ಯ * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    asm ಚಞ್ಚಲ ("syncht" : : : "memory");
    Q6_dccleaninva_A((ಶೂನ್ಯ *) fence);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ htp_fence_read(ಧ್ರುವ ಶೂನ್ಯ * fence_ptr, uint32_t * seq, uint32_t * status) {
    ಧ್ರುವ atomic_uint * fence = (ಧ್ರುವ atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((ಶೂನ್ಯ *) fence);
    asm ಚಞ್ಚಲ ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ htp_mdev_group_barrier(ಸಂರಚನಾ htp_ops_context * octx) {
    ಸಂರಚನಾ htp_context * ctx = octx->ctx;
    ಯದಿ (ctx->mdev.ಎಣಿಕೆ <= 1) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ uint32_t seq = ++ctx->mdev.fence_seq;

    ಸಂರಚನಾ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ಧ್ರುವ uint32_t mdev_idx   = ctx->mdev.idx;
    ಧ್ರುವ uint32_t mdev_ಎಣಿಕೆ = ctx->mdev.ಎಣಿಕೆ;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    ಪ್ರತಿ (uint32_t d = 0; d < mdev_ಎಣಿಕೆ; d++) {
        ಯದಿ (d == mdev_idx) ಅನುವರ್ತ;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        ಯಾವತ್ (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            ಯದಿ ((int32_t)(peer_seq - seq) >= 0) {
                ಯದಿ (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                ಭಙ್ಗ;
            }
            ಯದಿ (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            ಯದಿ (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                ಭಙ್ಗ;
            }
            hex_pause();
        }
    }
    asm ಚಞ್ಚಲ ("syncht" : : : "memory");

    ಯದಿ (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
