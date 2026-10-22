#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#অন্তর্ভাব <stdatomic.h>
#অন্তর্ভাব <stdint.h>

#অন্তর্ভাব <HAP_farf.h>

#অন্তর্ভাব "hex-utils.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-ctx.h"

স্থির অন্তঃস্থ atomic_uint * htp_mdev_fence_slot(ধ্রুব শূন্য * fence_base, uint32_t idx) {
    প্রত্যাগম (atomic_uint *) ((ধ্রুব uint8_t *) fence_base + (size_t) idx * HTP_FENCE_SLOT_SIZE);
}

স্থির অন্তঃস্থ শূন্য htp_fence_write(শূন্য * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    asm চঞ্চল ("syncht" : : : "memory");
    Q6_dccleaninva_A((শূন্য *) fence);
}

স্থির অন্তঃস্থ শূন্য htp_fence_read(ধ্রুব শূন্য * fence_ptr, uint32_t * seq, uint32_t * status) {
    ধ্রুব atomic_uint * fence = (ধ্রুব atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((শূন্য *) fence);
    asm চঞ্চল ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

স্থির অন্তঃস্থ শূন্য htp_mdev_group_barrier(সংরচনা htp_ops_context * octx) {
    সংরচনা htp_context * ctx = octx->ctx;
    যদি (ctx->mdev.গুনতি <= 1) {
        প্রত্যাগম;
    }

    ধ্রুব uint32_t seq = ++ctx->mdev.fence_seq;

    সংরচনা htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ধ্রুব uint32_t mdev_idx   = ctx->mdev.idx;
    ধ্রুব uint32_t mdev_গুনতি = ctx->mdev.গুনতি;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    প্রতি (uint32_t d = 0; d < mdev_গুনতি; d++) {
        যদি (d == mdev_idx) অনুবর্ত;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        যাবত্ (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            যদি ((int32_t)(peer_seq - seq) >= 0) {
                যদি (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                ভঙ্গ;
            }
            যদি (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            যদি (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                ভঙ্গ;
            }
            hex_pause();
        }
    }
    asm চঞ্চল ("syncht" : : : "memory");

    যদি (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
