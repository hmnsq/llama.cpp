#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#અન્તર્ભાવ <stdatomic.h>
#અન્તર્ભાવ <stdint.h>

#અન્તર્ભાવ <HAP_farf.h>

#અન્તર્ભાવ "hex-utils.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp-ctx.h"

સ્થિર અંતઃરેખ atomic_uint * htp_mdev_fence_slot(અચલ શૂન્ય * fence_base, uint32_t idx) {
    પાછા (atomic_uint *) ((અચલ uint8_t *) fence_base + (size_t) idx * HTP_FENCE_SLOT_SIZE);
}

સ્થિર અંતઃરેખ શૂન્ય htp_fence_write(શૂન્ય * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    asm અસ્થિર ("syncht" : : : "memory");
    Q6_dccleaninva_A((શૂન્ય *) fence);
}

સ્થિર અંતઃરેખ શૂન્ય htp_fence_read(અચલ શૂન્ય * fence_ptr, uint32_t * seq, uint32_t * status) {
    અચલ atomic_uint * fence = (અચલ atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((શૂન્ય *) fence);
    asm અસ્થિર ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

સ્થિર અંતઃરેખ શૂન્ય htp_mdev_group_barrier(સંરચના htp_ops_context * octx) {
    સંરચના htp_context * ctx = octx->ctx;
    જો (ctx->mdev.ગણતરી <= 1) {
        પાછા;
    }

    અચલ uint32_t seq = ++ctx->mdev.fence_seq;

    સંરચના htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    અચલ uint32_t mdev_idx   = ctx->mdev.idx;
    અચલ uint32_t mdev_ગણતરી = ctx->mdev.ગણતરી;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    માટે (uint32_t d = 0; d < mdev_ગણતરી; d++) {
        જો (d == mdev_idx) ચાલુ;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        જ્યારે (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            જો ((int32_t)(peer_seq - seq) >= 0) {
                જો (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                તોડો;
            }
            જો (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            જો (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                તોડો;
            }
            hex_pause();
        }
    }
    asm અસ્થિર ("syncht" : : : "memory");

    જો (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
