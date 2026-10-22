#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_FENCE_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_FENCE_H

#ಅನ್ತರ್ಭಾವ <stdatomic.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"

ಸ್ಥಿರ ಸಾಲೊಳಗೆ atomic_uint * htp_mdev_fence_ಸ್ಥಾನ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * fence_ತಳ, uint32_t idx) {
    ಹಿಂತಿರುಗಿ (atomic_uint *) ((ಸ್ಥಿರಾಂಕ uint8_t *) fence_ತಳ + (size_t) idx * HTP_FENCE_ಸ್ಥಾನ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_fence_write(ಶೂನ್ಯ * fence_ptr, uint32_t seq, uint32_t ಸ್ಥಿತಿ_2) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], ಸ್ಥಿತಿ_2);
    atomic_store(&fence[0], seq);
    asm ಅಸ್ಥಿರ ("syncht" : : : "memory");
    Q6_dccleaninva_A((ಶೂನ್ಯ *) fence);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_fence_read(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * fence_ptr, uint32_t * seq, uint32_t * ಸ್ಥಿತಿ_2) {
    ಸ್ಥಿರಾಂಕ atomic_uint * fence = (ಸ್ಥಿರಾಂಕ atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((ಶೂನ್ಯ *) fence);
    asm ಅಸ್ಥಿರ ("syncht" : : : "memory");
    *seq = atomic_ತುಂಬು(&fence[0]);
    *ಸ್ಥಿತಿ_2 = atomic_ತುಂಬು(&fence[1]);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_mdev_ಗುಂಪು_barrier(ರಚನೆ htp_ops_context * octx) {
    ರಚನೆ htp_context * ctx = octx->ctx;
    ಆದರೆ (ctx->mdev.ಎಣಿಕೆ_2 <= 1) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t seq = ++ctx->mdev.fence_seq;

    ರಚನೆ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ಸ್ಥಿರಾಂಕ uint32_t mdev_idx   = ctx->mdev.idx;
    ಸ್ಥಿರಾಂಕ uint32_t mdev_ಎಣಿಕೆ = ctx->mdev.ಎಣಿಕೆ_2;

    uint8_t * fence_ತಳ   = ctx->mdev.fence_ತಳ;
    atomic_uint * my_fence = htp_mdev_fence_ಸ್ಥಾನ(fence_ತಳ, mdev_idx);
    htp_fence_write(my_fence, seq, octx->ಸ್ಥಿತಿ_2);

    ಗಾಗಿ (uint32_t d = 0; d < mdev_ಎಣಿಕೆ; d++) {
        ಆದರೆ (d == mdev_idx) ಮುಂದುವರಿ;
        atomic_uint * peer_fence = htp_mdev_fence_ಸ್ಥಾನ(fence_ತಳ, d);
        uint64_t spins = 0;
        ಆಗ (1) {
            uint32_t peer_seq;
            uint32_t peer_ಸ್ಥಿತಿ;
            htp_fence_read(peer_fence, &peer_seq, &peer_ಸ್ಥಿತಿ);
            ಆದರೆ ((int32_t)(peer_seq - seq) >= 0) {
                ಆದರೆ (peer_ಸ್ಥಿತಿ > HTP_ಸ್ಥಿತಿ_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_ಸ್ಥಿತಿ, seq);
                    htp_ops_context_ಗಣ_ಸ್ಥಿತಿ(octx, peer_ಸ್ಥಿತಿ);
                }
                ನಿಲ್ಲಿಸು;
            }
            ಆದರೆ (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            ಆದರೆ (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_ಗಣ_ಸ್ಥಿತಿ(octx, HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR);
                ನಿಲ್ಲಿಸು;
            }
            hex_pause();
        }
    }
    asm ಅಸ್ಥಿರ ("syncht" : : : "memory");

    ಆದರೆ (octx->ಸ್ಥಿತಿ_2 > HTP_ಸ್ಥಿತಿ_OK) {
        htp_fence_write(my_fence, seq, octx->ಸ್ಥಿತಿ_2);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#ಆದರೆ_ಅಂತ್ಯ // HTP_FENCE_H
