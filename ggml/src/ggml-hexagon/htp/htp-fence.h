#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#شامل <stdatomic.h>
#شامل <stdint.h>

#شامل <HAP_farf.h>

#شامل "hex-utils.h"
#شامل "htp-ops.h"
#شامل "htp-ctx.h"

ایستا درونخطی atomic_uint * htp_mdev_fence_slot(ثابت تهی * fence_base, uint32_t idx) {
    بازگشت (atomic_uint *) ((ثابت uint8_t *) fence_base + (size_t) idx * HTP_FENCE_SLOT_SIZE);
}

ایستا درونخطی تهی htp_fence_write(تهی * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    asm فرار ("syncht" : : : "memory");
    Q6_dccleaninva_A((تهی *) fence);
}

ایستا درونخطی تهی htp_fence_read(ثابت تهی * fence_ptr, uint32_t * seq, uint32_t * status) {
    ثابت atomic_uint * fence = (ثابت atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((تهی *) fence);
    asm فرار ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

ایستا درونخطی تهی htp_mdev_group_barrier(ساختار htp_ops_context * octx) {
    ساختار htp_context * ctx = octx->ctx;
    اگر (ctx->mdev.شمار <= 1) {
        بازگشت;
    }

    ثابت uint32_t seq = ++ctx->mdev.fence_seq;

    ساختار htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ثابت uint32_t mdev_idx   = ctx->mdev.idx;
    ثابت uint32_t mdev_شمار = ctx->mdev.شمار;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    برای (uint32_t d = 0; d < mdev_شمار; d++) {
        اگر (d == mdev_idx) ادامه;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        وقتی (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            اگر ((int32_t)(peer_seq - seq) >= 0) {
                اگر (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                بشکن;
            }
            اگر (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            اگر (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                بشکن;
            }
            hex_pause();
        }
    }
    asm فرار ("syncht" : : : "memory");

    اگر (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
