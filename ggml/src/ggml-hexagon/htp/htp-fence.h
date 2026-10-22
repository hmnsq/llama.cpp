#اگر_غیر_تعریف_شدہ HTP_FENCE_H
#تعریف_کرو HTP_FENCE_H

#شامل <stdatomic.h>
#شامل <stdint.h>

#شامل <HAP_farf.h>

#شامل "hex-utils.h"
#شامل "htp-ops.h"
#شامل "htp-ctx.h"

ساکن اندرونی atomic_uint * htp_mdev_fence_خانہ(ثابت خالی * fence_بنیاد, uint32_t idx) {
    واپس (atomic_uint *) ((ثابت uint8_t *) fence_بنیاد + (size_t) idx * HTP_FENCE_خانہ_سائز);
}

ساکن اندرونی خالی htp_fence_write(خالی * fence_ptr, uint32_t seq, uint32_t حالت_2) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], حالت_2);
    atomic_store(&fence[0], seq);
    asm متلون ("syncht" : : : "memory");
    Q6_dccleaninva_A((خالی *) fence);
}

ساکن اندرونی خالی htp_fence_read(ثابت خالی * fence_ptr, uint32_t * seq, uint32_t * حالت_2) {
    ثابت atomic_uint * fence = (ثابت atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((خالی *) fence);
    asm متلون ("syncht" : : : "memory");
    *seq = atomic_بھرو(&fence[0]);
    *حالت_2 = atomic_بھرو(&fence[1]);
}

ساکن اندرونی خالی htp_mdev_گروہ_barrier(ساخت htp_ops_context * octx) {
    ساخت htp_context * ctx = octx->ctx;
    اگر (ctx->mdev.گنتی <= 1) {
        واپس;
    }

    ثابت uint32_t seq = ++ctx->mdev.fence_seq;

    ساخت htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ثابت uint32_t mdev_idx   = ctx->mdev.idx;
    ثابت uint32_t mdev_گنتی = ctx->mdev.گنتی;

    uint8_t * fence_بنیاد   = ctx->mdev.fence_بنیاد;
    atomic_uint * my_fence = htp_mdev_fence_خانہ(fence_بنیاد, mdev_idx);
    htp_fence_write(my_fence, seq, octx->حالت_2);

    برائے (uint32_t d = 0; d < mdev_گنتی; d++) {
        اگر (d == mdev_idx) جاری;
        atomic_uint * peer_fence = htp_mdev_fence_خانہ(fence_بنیاد, d);
        uint64_t spins = 0;
        جب (1) {
            uint32_t peer_seq;
            uint32_t peer_حالت;
            htp_fence_read(peer_fence, &peer_seq, &peer_حالت);
            اگر ((int32_t)(peer_seq - seq) >= 0) {
                اگر (peer_حالت > HTP_حالت_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_حالت, seq);
                    htp_ops_context_سیٹ_حالت(octx, peer_حالت);
                }
                توڑو;
            }
            اگر (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            اگر (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_سیٹ_حالت(octx, HTP_حالت_اندرونی_ERR);
                توڑو;
            }
            hex_pause();
        }
    }
    asm متلون ("syncht" : : : "memory");

    اگر (octx->حالت_2 > HTP_حالت_OK) {
        htp_fence_write(my_fence, seq, octx->حالت_2);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#اگر_ختم // HTP_FENCE_H
