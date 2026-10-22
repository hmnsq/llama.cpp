#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव <HAP_farf.h>

#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-ctx.h"

स्थिर ओळीत atomic_uint * htp_mdev_fence_slot(स्थिरांक शून्य * fence_base, uint32_t idx) {
    परत (atomic_uint *) ((स्थिरांक uint8_t *) fence_base + (माप_प्रकार) idx * HTP_FENCE_SLOT_SIZE);
}

स्थिर ओळीत शून्य htp_fence_write(शून्य * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    यंत्र अस्थिर ("syncht" : : : "memory");
    Q6_dccleaninva_A((शून्य *) fence);
}

स्थिर ओळीत शून्य htp_fence_read(स्थिरांक शून्य * fence_ptr, uint32_t * seq, uint32_t * status) {
    स्थिरांक atomic_uint * fence = (स्थिरांक atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((शून्य *) fence);
    यंत्र अस्थिर ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

स्थिर ओळीत शून्य htp_mdev_group_barrier(संरचना htp_ops_context * octx) {
    संरचना htp_context * ctx = octx->ctx;
    जर (ctx->mdev.मोजणी <= 1) {
        परत;
    }

    स्थिरांक uint32_t seq = ++ctx->mdev.fence_seq;

    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    स्थिरांक uint32_t mdev_idx   = ctx->mdev.idx;
    स्थिरांक uint32_t mdev_मोजणी = ctx->mdev.मोजणी;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    साठी (uint32_t d = 0; d < mdev_मोजणी; d++) {
        जर (d == mdev_idx) सुरूठेवा;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        पर्यंत (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            जर ((int32_t)(peer_seq - seq) >= 0) {
                जर (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                थांबा;
            }
            जर (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            जर (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                थांबा;
            }
            hex_pause();
        }
    }
    यंत्र अस्थिर ("syncht" : : : "memory");

    जर (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
