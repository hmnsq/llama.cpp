#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#அந்தர்பாவ <stdatomic.h>
#அந்தர்பாவ <stdint.h>

#அந்தர்பாவ <HAP_farf.h>

#அந்தர்பாவ "hex-utils.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-ctx.h"

நிலையான உள்வரி atomic_uint * htp_mdev_fence_slot(மாறிலி வெறுமை * fence_base, uint32_t idx) {
    திரும்பு (atomic_uint *) ((மாறிலி uint8_t *) fence_base + (size_t) idx * HTP_FENCE_SLOT_SIZE);
}

நிலையான உள்வரி வெறுமை htp_fence_write(வெறுமை * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    asm நிலையற்ற ("syncht" : : : "memory");
    Q6_dccleaninva_A((வெறுமை *) fence);
}

நிலையான உள்வரி வெறுமை htp_fence_read(மாறிலி வெறுமை * fence_ptr, uint32_t * seq, uint32_t * status) {
    மாறிலி atomic_uint * fence = (மாறிலி atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((வெறுமை *) fence);
    asm நிலையற்ற ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

நிலையான உள்வரி வெறுமை htp_mdev_group_barrier(கட்டமைப்பு htp_ops_context * octx) {
    கட்டமைப்பு htp_context * ctx = octx->ctx;
    எனில் (ctx->mdev.மொத்தம் <= 1) {
        திரும்பு;
    }

    மாறிலி uint32_t seq = ++ctx->mdev.fence_seq;

    கட்டமைப்பு htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    மாறிலி uint32_t mdev_idx   = ctx->mdev.idx;
    மாறிலி uint32_t mdev_மொத்தம் = ctx->mdev.மொத்தம்;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    க்கு (uint32_t d = 0; d < mdev_மொத்தம்; d++) {
        எனில் (d == mdev_idx) தொடர்;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        போது (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            எனில் ((int32_t)(peer_seq - seq) >= 0) {
                எனில் (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                நிறுத்து;
            }
            எனில் (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            எனில் (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                நிறுத்து;
            }
            hex_pause();
        }
    }
    asm நிலையற்ற ("syncht" : : : "memory");

    எனில் (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
