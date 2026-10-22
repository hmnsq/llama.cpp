#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <stdatomic.h>

#அந்தர்பாவ <hexagon_types.h>
#அந்தர்பாவ <qurt_thread.h>
#அந்தர்பாவ <qurt_futex.h>
#அந்தர்பாவ <HAP_farf.h>

#அந்தர்பாவ "hex-utils.h"
#அந்தர்பாவ "hex-profile.h"

#ifdef __cplusplus
வெளி "C" {
#endif

#எனில் __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_மொத்தம் 2000
#இல்லையெனில்
#define HMX_QUEUE_POLL_மொத்தம் 1
#endif

வகைவரை வெறுமை (*hmx_queue_func)(வெறுமை *);

// Dummy funcs used as signals
எண்ணிக்கை hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

கட்டமைப்பு hmx_queue_desc {
    hmx_queue_func   func;
    வெறுமை *           data;
    atomic_uint      done;
};

கட்டமைப்பு hmx_queue_s {
    கட்டமைப்பு hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned முழு     idx_எடு;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    வெறுமை *           அடுக்கு;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    கட்டமைப்பு htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

வகைவரை கட்டமைப்பு hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t அடுக்கு_size);
size_t      hmx_queue_alignof(வெறுமை);
hmx_queue_t hmx_queue_init(வெறுமை * ptr, size_t capacity, uint32_t அடுக்கு_size, uint32_t hap_rctx, கட்டமைப்பு htp_thread_trace * trace);
வெறுமை        hmx_queue_free(hmx_queue_t q);

நிலையான உள்வரி கட்டமைப்பு hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, வெறுமை * data) {
    கட்டமைப்பு hmx_queue_desc d = { func, data };
    திரும்பு d;
}

நிலையான உள்வரி bool hmx_queue_தள்ளு(hmx_queue_t q, கட்டமைப்பு hmx_queue_desc d) {
    unsigned முழு ir = atomic_load(&q->idx_read);
    unsigned முழு iw = atomic_load(&q->idx_write);

    எனில் (((iw + 1) & q->idx_mask) == ir) {
        FARF(உயர்வு_2, "hmx-queue-push: queue is full\n");
        திரும்பு false;
    }

    atomic_store(&d.done, 0);

    FARF(உயர்வு_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    திரும்பு true;
}

நிலையான உள்வரி bool hmx_queue_signal(hmx_queue_t q, எண்ணிக்கை hmx_queue_signal sig) {
    திரும்பு hmx_queue_தள்ளு(q, hmx_queue_make_desc((hmx_queue_func) sig, வெற்று));
}

நிலையான உள்வரி bool hmx_queue_empty(hmx_queue_t q) {
    திரும்பு q->idx_எடு == atomic_load(&q->idx_write);
}

நிலையான உள்வரி uint32_t hmx_queue_depth(hmx_queue_t q) {
    திரும்பு (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

நிலையான உள்வரி uint32_t hmx_queue_capacity(hmx_queue_t q) {
    திரும்பு q->capacity;
}

நிலையான உள்வரி கட்டமைப்பு hmx_queue_desc hmx_queue_எடு_one(hmx_queue_t q) {
    unsigned முழு ip = q->idx_எடு;
    unsigned முழு iw = atomic_load(&q->idx_write);

    கட்டமைப்பு hmx_queue_desc rd = { வெற்று, வெற்று };
    எனில் (ip == iw) {
        திரும்பு rd;
    }

    // Wait for desc to complete
    கட்டமைப்பு hmx_queue_desc * d = &q->desc[ip];
    போது (!atomic_load(&d->done)) {
        FARF(உயர்வு_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_எடு = (ip + 1) & q->idx_mask;

    FARF(உயர்வு_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    திரும்பு rd;
}

நிலையான உள்வரி கட்டமைப்பு hmx_queue_desc hmx_queue_எடு(hmx_queue_t q) {
    போது (1) {
        கட்டமைப்பு hmx_queue_desc d = hmx_queue_எடு_one(q);

        uint32_t sig = (uint32_t) d.func;
        எனில் (sig && sig <= HMX_QUEUE_KILL)
            தொடர்;

        திரும்பு d;
    }
}

நிலையான உள்வரி வெறுமை hmx_queue_flush(hmx_queue_t q) {
    போது (hmx_queue_எடு_one(q).func != வெற்று) ;
}

நிலையான உள்வரி வெறுமை hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

நிலையான உள்வரி வெறுமை hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
