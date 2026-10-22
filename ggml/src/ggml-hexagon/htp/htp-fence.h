#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव <HAP_farf.h>

#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-ctx.h"

स्थिर अन्तःस्थ atomic_uint * htp_mdev_fence_slot(अचर शून्य * fence_base, uint32_t idx) {
    फर्काऊ (atomic_uint *) ((अचर uint8_t *) fence_base + (माप_प्रकार) idx * HTP_FENCE_SLOT_SIZE);
}

स्थिर अन्तःस्थ शून्य htp_fence_write(शून्य * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    यंत्र चञ्चल ("syncht" : : : "memory");
    Q6_dccleaninva_A((शून्य *) fence);
}

स्थिर अन्तःस्थ शून्य htp_fence_read(अचर शून्य * fence_ptr, uint32_t * seq, uint32_t * status) {
    अचर atomic_uint * fence = (अचर atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((शून्य *) fence);
    यंत्र चञ्चल ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

स्थिर अन्तःस्थ शून्य htp_mdev_group_barrier(संरचना htp_ops_context * octx) {
    संरचना htp_context * ctx = octx->ctx;
    यदि (ctx->mdev.गन्ती <= 1) {
        फर्काऊ;
    }

    अचर uint32_t seq = ++ctx->mdev.fence_seq;

    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    अचर uint32_t mdev_idx   = ctx->mdev.idx;
    अचर uint32_t mdev_गन्ती = ctx->mdev.गन्ती;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    लागि (uint32_t d = 0; d < mdev_गन्ती; d++) {
        यदि (d == mdev_idx) जारी;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        जबसम्म (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            यदि ((int32_t)(peer_seq - seq) >= 0) {
                यदि (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                रोक;
            }
            यदि (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            यदि (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                रोक;
            }
            hex_pause();
        }
    }
    यंत्र चञ्चल ("syncht" : : : "memory");

    यदि (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
