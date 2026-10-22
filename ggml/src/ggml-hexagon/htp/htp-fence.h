#اگر_تعریف_نشده HTP_FENCE_H
#تعریف_کن HTP_FENCE_H

#ضمن <stdatomic.h>
#ضمن <stdint.h>

#ضمن <HAP_farf.h>

#ضمن "hex-utils.h"
#ضمن "htp-ops.h"
#ضمن "htp-ctx.h"

ساكن مضمن atomic_uint * htp_mdev_fence_جایگاه(ثابت فراغ * fence_پایه, uint32_t idx) {
    أرجع (atomic_uint *) ((ثابت uint8_t *) fence_پایه + (size_t) idx * HTP_FENCE_جایگاه_اندازه);
}

ساكن مضمن فراغ htp_fence_write(فراغ * fence_ptr, uint32_t seq, uint32_t وضعیت) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], وضعیت);
    atomic_store(&fence[0], seq);
    asm متطاير ("syncht" : : : "memory");
    Q6_dccleaninva_A((فراغ *) fence);
}

ساكن مضمن فراغ htp_fence_read(ثابت فراغ * fence_ptr, uint32_t * seq, uint32_t * وضعیت) {
    ثابت atomic_uint * fence = (ثابت atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((فراغ *) fence);
    asm متطاير ("syncht" : : : "memory");
    *seq = atomic_بارگیری(&fence[0]);
    *وضعیت = atomic_بارگیری(&fence[1]);
}

ساكن مضمن فراغ htp_mdev_گروه_barrier(بنية htp_ops_context * octx) {
    بنية htp_context * ctx = octx->ctx;
    إذا (ctx->mdev.شمار <= 1) {
        أرجع;
    }

    ثابت uint32_t seq = ++ctx->mdev.fence_seq;

    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ثابت uint32_t mdev_idx   = ctx->mdev.idx;
    ثابت uint32_t mdev_شمار = ctx->mdev.شمار;

    uint8_t * fence_پایه   = ctx->mdev.fence_پایه;
    atomic_uint * my_fence = htp_mdev_fence_جایگاه(fence_پایه, mdev_idx);
    htp_fence_write(my_fence, seq, octx->وضعیت);

    لكل (uint32_t d = 0; d < mdev_شمار; d++) {
        إذا (d == mdev_idx) استمر;
        atomic_uint * peer_fence = htp_mdev_fence_جایگاه(fence_پایه, d);
        uint64_t spins = 0;
        طالما (1) {
            uint32_t peer_seq;
            uint32_t peer_وضعیت;
            htp_fence_read(peer_fence, &peer_seq, &peer_وضعیت);
            إذا ((int32_t)(peer_seq - seq) >= 0) {
                إذا (peer_وضعیت > HTP_وضعیت_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_وضعیت, seq);
                    htp_ops_context_مجموعه_وضعیت(octx, peer_وضعیت);
                }
                اكسر;
            }
            إذا (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            إذا (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_مجموعه_وضعیت(octx, HTP_وضعیت_داخلی_ERR);
                اكسر;
            }
            hex_pause();
        }
    }
    asm متطاير ("syncht" : : : "memory");

    إذا (octx->وضعیت > HTP_وضعیت_OK) {
        htp_fence_write(my_fence, seq, octx->وضعیت);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#پایان_اگر // HTP_FENCE_H
