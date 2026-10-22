#अगर_अघोषित HTP_FENCE_H
#घोषणा HTP_FENCE_H

#समावेश <stdatomic.h>
#समावेश <stdint.h>

#समावेश <HAP_farf.h>

#समावेश "hex-utils.h"
#समावेश "htp-ops.h"
#समावेश "htp-ctx.h"

अटल अंतरभूत atomic_uint * htp_mdev_fence_slot(स्थिर व्योम * fence_base, uint32_t idx) {
    वापस (atomic_uint *) ((स्थिर uint8_t *) fence_base + (माप_प्रकार) idx * HTP_FENCE_SLOT_आकार);
}

अटल अंतरभूत व्योम htp_fence_write(व्योम * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    यंत्र अस्थिर ("syncht" : : : "memory");
    Q6_dccleaninva_A((व्योम *) fence);
}

अटल अंतरभूत व्योम htp_fence_read(स्थिर व्योम * fence_ptr, uint32_t * seq, uint32_t * status) {
    स्थिर atomic_uint * fence = (स्थिर atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((व्योम *) fence);
    यंत्र अस्थिर ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

अटल अंतरभूत व्योम htp_mdev_group_barrier(काष्ठा htp_ops_context * octx) {
    काष्ठा htp_context * ctx = octx->ctx;
    अगर (ctx->mdev.गिनती <= 1) {
        वापस;
    }

    स्थिर uint32_t seq = ++ctx->mdev.fence_seq;

    काष्ठा htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    स्थिर uint32_t mdev_idx   = ctx->mdev.idx;
    स्थिर uint32_t mdev_गिनती = ctx->mdev.गिनती;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    क्रम (uint32_t d = 0; d < mdev_गिनती; d++) {
        अगर (d == mdev_idx) जारी;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        जबतक (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            अगर ((int32_t)(peer_seq - seq) >= 0) {
                अगर (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                अवरोध;
            }
            अगर (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            अगर (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                अवरोध;
            }
            hex_pause();
        }
    }
    यंत्र अस्थिर ("syncht" : : : "memory");

    अगर (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#पूर्ण_अगर // HTP_FENCE_H
