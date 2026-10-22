#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_FENCE_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_FENCE_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdatomic.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"

ਸਥਿਰ ਸਤਰਅੰਦਰ atomic_uint * htp_mdev_fence_ਥਾਂ(ਅਚਲ ਖਾਲੀ * fence_ਅਧਾਰ, uint32_t idx) {
    ਵਾਪਸ (atomic_uint *) ((ਅਚਲ uint8_t *) fence_ਅਧਾਰ + (size_t) idx * HTP_FENCE_ਥਾਂ_ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_fence_write(ਖਾਲੀ * fence_ptr, uint32_t seq, uint32_t ਸਥਿਤੀ) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], ਸਥਿਤੀ);
    atomic_store(&fence[0], seq);
    asm ਅਸਥਿਰ ("syncht" : : : "memory");
    Q6_dccleaninva_A((ਖਾਲੀ *) fence);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_fence_read(ਅਚਲ ਖਾਲੀ * fence_ptr, uint32_t * seq, uint32_t * ਸਥਿਤੀ) {
    ਅਚਲ atomic_uint * fence = (ਅਚਲ atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((ਖਾਲੀ *) fence);
    asm ਅਸਥਿਰ ("syncht" : : : "memory");
    *seq = atomic_ਲੱਦੋ(&fence[0]);
    *ਸਥਿਤੀ = atomic_ਲੱਦੋ(&fence[1]);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_mdev_ਸਮੂਹ_barrier(ਬਣਤਰ htp_ops_context * octx) {
    ਬਣਤਰ htp_context * ctx = octx->ctx;
    ਜੇ (ctx->mdev.ਗਿਣਤੀ_2 <= 1) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t seq = ++ctx->mdev.fence_seq;

    ਬਣਤਰ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ਅਚਲ uint32_t mdev_idx   = ctx->mdev.idx;
    ਅਚਲ uint32_t mdev_ਗਿਣਤੀ = ctx->mdev.ਗਿਣਤੀ_2;

    uint8_t * fence_ਅਧਾਰ   = ctx->mdev.fence_ਅਧਾਰ;
    atomic_uint * my_fence = htp_mdev_fence_ਥਾਂ(fence_ਅਧਾਰ, mdev_idx);
    htp_fence_write(my_fence, seq, octx->ਸਥਿਤੀ);

    ਲਈ (uint32_t d = 0; d < mdev_ਗਿਣਤੀ; d++) {
        ਜੇ (d == mdev_idx) ਜਾਰੀ;
        atomic_uint * peer_fence = htp_mdev_fence_ਥਾਂ(fence_ਅਧਾਰ, d);
        uint64_t spins = 0;
        ਜਦੋਂ (1) {
            uint32_t peer_seq;
            uint32_t peer_ਸਥਿਤੀ;
            htp_fence_read(peer_fence, &peer_seq, &peer_ਸਥਿਤੀ);
            ਜੇ ((int32_t)(peer_seq - seq) >= 0) {
                ਜੇ (peer_ਸਥਿਤੀ > HTP_ਸਥਿਤੀ_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_ਸਥਿਤੀ, seq);
                    htp_ops_context_ਸੈੱਟ_ਸਥਿਤੀ(octx, peer_ਸਥਿਤੀ);
                }
                ਰੋਕੋ;
            }
            ਜੇ (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            ਜੇ (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_ਸੈੱਟ_ਸਥਿਤੀ(octx, HTP_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERR);
                ਰੋਕੋ;
            }
            hex_pause();
        }
    }
    asm ਅਸਥਿਰ ("syncht" : : : "memory");

    ਜੇ (octx->ਸਥਿਤੀ > HTP_ਸਥਿਤੀ_OK) {
        htp_fence_write(my_fence, seq, octx->ਸਥਿਤੀ);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#ਜੇ_ਅੰਤ // HTP_FENCE_H
