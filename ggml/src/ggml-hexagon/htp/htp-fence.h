#यदि_अपरिभाषितम् HTP_FENCE_H
#परिभाषय HTP_FENCE_H

#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव <HAP_farf.h>

#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-ctx.h"

स्थिर अन्तःस्थ atomic_uint * htp_mdev_fence_slot(ध्रुव शून्य * fence_आधारः, uint32_t idx) {
    प्रत्यागम (atomic_uint *) ((ध्रुव uint8_t *) fence_आधारः + (size_t) idx * HTP_FENCE_SLOT_आकार);
}

स्थिर अन्तःस्थ शून्य htp_fence_write(शून्य * fence_ptr, uint32_t seq, uint32_t स्थितिः) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], स्थितिः);
    atomic_store(&fence[0], seq);
    asm चञ्चल ("syncht" : : : "memory");
    Q6_dccleaninva_A((शून्य *) fence);
}

स्थिर अन्तःस्थ शून्य htp_fence_read(ध्रुव शून्य * fence_ptr, uint32_t * seq, uint32_t * स्थितिः) {
    ध्रुव atomic_uint * fence = (ध्रुव atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((शून्य *) fence);
    asm चञ्चल ("syncht" : : : "memory");
    *seq = atomic_आरोपय(&fence[0]);
    *स्थितिः = atomic_आरोपय(&fence[1]);
}

स्थिर अन्तःस्थ शून्य htp_mdev_गणः_barrier(संरचना htp_ops_context * octx) {
    संरचना htp_context * ctx = octx->ctx;
    यदि (ctx->mdev.गणनम् <= 1) {
        प्रत्यागम;
    }

    ध्रुव uint32_t seq = ++ctx->mdev.fence_seq;

    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ध्रुव uint32_t mdev_idx   = ctx->mdev.idx;
    ध्रुव uint32_t mdev_गणनम् = ctx->mdev.गणनम्;

    uint8_t * fence_आधारः   = ctx->mdev.fence_आधारः;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_आधारः, mdev_idx);
    htp_fence_write(my_fence, seq, octx->स्थितिः);

    प्रति (uint32_t d = 0; d < mdev_गणनम्; d++) {
        यदि (d == mdev_idx) अनुवर्त;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_आधारः, d);
        uint64_t spins = 0;
        यावत् (1) {
            uint32_t peer_seq;
            uint32_t peer_स्थितिः;
            htp_fence_read(peer_fence, &peer_seq, &peer_स्थितिः);
            यदि ((int32_t)(peer_seq - seq) >= 0) {
                यदि (peer_स्थितिः > HTP_स्थितिः_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_स्थितिः, seq);
                    htp_ops_context_समुच्चयः_स्थितिः(octx, peer_स्थितिः);
                }
                भङ्ग;
            }
            यदि (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            यदि (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_समुच्चयः_स्थितिः(octx, HTP_स्थितिः_आन्तरिकम्_ERR);
                भङ्ग;
            }
            hex_pause();
        }
    }
    asm चञ्चल ("syncht" : : : "memory");

    यदि (octx->स्थितिः > HTP_स्थितिः_OK) {
        htp_fence_write(my_fence, seq, octx->स्थितिः);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#यदि_अन्तः // HTP_FENCE_H
