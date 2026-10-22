#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#ଅନ୍ତର୍ଭାଵ <stdatomic.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>

#ଅନ୍ତର୍ଭାଵ "hex-utils.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"

ସ୍ଥିର ଧାଡ଼ିଭିତରେ atomic_uint * htp_mdev_fence_slot(ଅଚଳ ଶୂନ୍ୟ * fence_base, uint32_t idx) {
    ଫେରନ୍ତୁ (atomic_uint *) ((ଅଚଳ uint8_t *) fence_base + (size_t) idx * HTP_FENCE_SLOT_SIZE);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_fence_write(ଶୂନ୍ୟ * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    asm ଅସ୍ଥିର ("syncht" : : : "memory");
    Q6_dccleaninva_A((ଶୂନ୍ୟ *) fence);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_fence_read(ଅଚଳ ଶୂନ୍ୟ * fence_ptr, uint32_t * seq, uint32_t * status) {
    ଅଚଳ atomic_uint * fence = (ଅଚଳ atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((ଶୂନ୍ୟ *) fence);
    asm ଅସ୍ଥିର ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_mdev_group_barrier(ସଂରଚନା htp_ops_context * octx) {
    ସଂରଚନା htp_context * ctx = octx->ctx;
    ଯଦି (ctx->mdev.ଗଣନା_2 <= 1) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ uint32_t seq = ++ctx->mdev.fence_seq;

    ସଂରଚନା htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ଅଚଳ uint32_t mdev_idx   = ctx->mdev.idx;
    ଅଚଳ uint32_t mdev_ଗଣନା = ctx->mdev.ଗଣନା_2;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    ପାଇଁ (uint32_t d = 0; d < mdev_ଗଣନା; d++) {
        ଯଦି (d == mdev_idx) ଜାରି;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        ଯେତେବେଳେ (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            ଯଦି ((int32_t)(peer_seq - seq) >= 0) {
                ଯଦି (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                ଭାଙ୍ଗ;
            }
            ଯଦି (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            ଯଦି (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                ଭାଙ୍ଗ;
            }
            hex_pause();
        }
    }
    asm ଅସ୍ଥିର ("syncht" : : : "memory");

    ଯଦି (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
