#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#include <stdatomic.h>
#include <stdint.h>

#include <HAP_farf.h>

#include "hex-utils.h"
#include "htp-ops.h"
#include "htp-ctx.h"

static anta_hstha atomic_uint * htp_mdev_fence_slot(const void * fence_base, uint32_t idx) {
    ສົ່ງຄືນ (atomic_uint *) ((const uint8_t *) fence_base + (size_t) idx * HTP_FENCE_SLOT_SIZE);
}

static anta_hstha void htp_fence_write(void * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    asm ca_ncala ("syncht" : : : "memory");
    Q6_dccleaninva_A((void *) fence);
}

static anta_hstha void htp_fence_read(const void * fence_ptr, uint32_t * seq, uint32_t * status) {
    const atomic_uint * fence = (const atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((void *) fence);
    asm ca_ncala ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

static anta_hstha void htp_mdev_group_barrier(sa_nracanaa htp_ops_context * octx) {
    sa_nracanaa htp_context * ctx = octx->ctx;
    ຖ້າ (ctx->mdev.count <= 1) {
        ສົ່ງຄືນ;
    }

    const uint32_t seq = ++ctx->mdev.fence_seq;

    sa_nracanaa htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    const uint32_t mdev_idx   = ctx->mdev.idx;
    const uint32_t mdev_count = ctx->mdev.count;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    ສຳລັບ (uint32_t d = 0; d < mdev_count; d++) {
        ຖ້າ (d == mdev_idx) continue;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        ໃນຂະນະທີ່ (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            ຖ້າ ((int32_t)(peer_seq - seq) >= 0) {
                ຖ້າ (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                bha_nga;
            }
            ຖ້າ (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            ຖ້າ (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                bha_nga;
            }
            hex_pause();
        }
    }
    asm ca_ncala ("syncht" : : : "memory");

    ຖ້າ (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
