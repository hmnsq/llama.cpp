#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#അന്തര്ഭാവ <stdatomic.h>
#അന്തര്ഭാവ <stdint.h>

#അന്തര്ഭാവ <HAP_farf.h>

#അന്തര്ഭാവ "hex-utils.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp-ctx.h"

സ്ഥിതം വരിയിൽ atomic_uint * htp_mdev_fence_slot(സ്ഥിരം ശൂന്യം * fence_base, uint32_t idx) {
    തിരികെ (atomic_uint *) ((സ്ഥിരം uint8_t *) fence_base + (size_t) idx * HTP_FENCE_SLOT_SIZE);
}

സ്ഥിതം വരിയിൽ ശൂന്യം htp_fence_write(ശൂന്യം * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    asm അസ്ഥിര ("syncht" : : : "memory");
    Q6_dccleaninva_A((ശൂന്യം *) fence);
}

സ്ഥിതം വരിയിൽ ശൂന്യം htp_fence_read(സ്ഥിരം ശൂന്യം * fence_ptr, uint32_t * seq, uint32_t * status) {
    സ്ഥിരം atomic_uint * fence = (സ്ഥിരം atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((ശൂന്യം *) fence);
    asm അസ്ഥിര ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

സ്ഥിതം വരിയിൽ ശൂന്യം htp_mdev_group_barrier(ഘടന htp_ops_context * octx) {
    ഘടന htp_context * ctx = octx->ctx;
    എങ്കിൽ (ctx->mdev.എണ്ണം <= 1) {
        തിരികെ;
    }

    സ്ഥിരം uint32_t seq = ++ctx->mdev.fence_seq;

    ഘടന htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    സ്ഥിരം uint32_t mdev_idx   = ctx->mdev.idx;
    സ്ഥിരം uint32_t mdev_എണ്ണം = ctx->mdev.എണ്ണം;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    വേണ്ടി (uint32_t d = 0; d < mdev_എണ്ണം; d++) {
        എങ്കിൽ (d == mdev_idx) തുടരുക;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        ആകുമ്പോൾ (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            എങ്കിൽ ((int32_t)(peer_seq - seq) >= 0) {
                എങ്കിൽ (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                നിർത്തുക;
            }
            എങ്കിൽ (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            എങ്കിൽ (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                നിർത്തുക;
            }
            hex_pause();
        }
    }
    asm അസ്ഥിര ("syncht" : : : "memory");

    എങ്കിൽ (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
