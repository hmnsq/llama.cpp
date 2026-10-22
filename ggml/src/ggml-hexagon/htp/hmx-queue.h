#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdatomic.h>

#ಅನ್ತರ್ಭಾವ <hexagon_types.h>
#ಅನ್ತರ್ಭಾವ <qurt_thread.h>
#ಅನ್ತರ್ಭಾವ <qurt_futex.h>
#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

#ಯದಿ __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_ಎಣಿಕೆ 2000
#ಅನ್ಯಥಾ
#define HMX_QUEUE_POLL_ಎಣಿಕೆ 1
#endif

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*hmx_queue_func)(ಶೂನ್ಯ *);

// Dummy funcs used as signals
ಗಣನಾ hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

ಸಂರಚನಾ hmx_queue_desc {
    hmx_queue_func   func;
    ಶೂನ್ಯ *           data;
    atomic_uint      done;
};

ಸಂರಚನಾ hmx_queue_s {
    ಸಂರಚನಾ hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned ಪೂರ್ಣಾಙ್ಕ     idx_ತೆಗೆ;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    ಶೂನ್ಯ *           ರಾಶಿ;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    ಸಂರಚನಾ htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t ರಾಶಿ_size);
size_t      hmx_queue_alignof(ಶೂನ್ಯ);
hmx_queue_t hmx_queue_init(ಶೂನ್ಯ * ptr, size_t capacity, uint32_t ರಾಶಿ_size, uint32_t hap_rctx, ಸಂರಚನಾ htp_thread_trace * trace);
ಶೂನ್ಯ        hmx_queue_free(hmx_queue_t q);

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಸಂರಚನಾ hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ hmx_queue_desc d = { func, data };
    ಪ್ರತ್ಯಾಗಮ d;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool hmx_queue_ತಳ್ಳು(hmx_queue_t q, ಸಂರಚನಾ hmx_queue_desc d) {
    unsigned ಪೂರ್ಣಾಙ್ಕ ir = atomic_load(&q->idx_read);
    unsigned ಪೂರ್ಣಾಙ್ಕ iw = atomic_load(&q->idx_write);

    ಯದಿ (((iw + 1) & q->idx_mask) == ir) {
        FARF(ಮೇಲಿನ_2, "hmx-queue-push: queue is full\n");
        ಪ್ರತ್ಯಾಗಮ false;
    }

    atomic_store(&d.done, 0);

    FARF(ಮೇಲಿನ_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    ಪ್ರತ್ಯಾಗಮ true;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool hmx_queue_signal(hmx_queue_t q, ಗಣನಾ hmx_queue_signal sig) {
    ಪ್ರತ್ಯಾಗಮ hmx_queue_ತಳ್ಳು(q, hmx_queue_make_desc((hmx_queue_func) sig, ರಿಕ್ತ));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool hmx_queue_empty(hmx_queue_t q) {
    ಪ್ರತ್ಯಾಗಮ q->idx_ತೆಗೆ == atomic_load(&q->idx_write);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t hmx_queue_depth(hmx_queue_t q) {
    ಪ್ರತ್ಯಾಗಮ (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t hmx_queue_capacity(hmx_queue_t q) {
    ಪ್ರತ್ಯಾಗಮ q->capacity;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಸಂರಚನಾ hmx_queue_desc hmx_queue_ತೆಗೆ_one(hmx_queue_t q) {
    unsigned ಪೂರ್ಣಾಙ್ಕ ip = q->idx_ತೆಗೆ;
    unsigned ಪೂರ್ಣಾಙ್ಕ iw = atomic_load(&q->idx_write);

    ಸಂರಚನಾ hmx_queue_desc rd = { ರಿಕ್ತ, ರಿಕ್ತ };
    ಯದಿ (ip == iw) {
        ಪ್ರತ್ಯಾಗಮ rd;
    }

    // Wait for desc to complete
    ಸಂರಚನಾ hmx_queue_desc * d = &q->desc[ip];
    ಯಾವತ್ (!atomic_load(&d->done)) {
        FARF(ಮೇಲಿನ_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_ತೆಗೆ = (ip + 1) & q->idx_mask;

    FARF(ಮೇಲಿನ_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    ಪ್ರತ್ಯಾಗಮ rd;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಸಂರಚನಾ hmx_queue_desc hmx_queue_ತೆಗೆ(hmx_queue_t q) {
    ಯಾವತ್ (1) {
        ಸಂರಚನಾ hmx_queue_desc d = hmx_queue_ತೆಗೆ_one(q);

        uint32_t sig = (uint32_t) d.func;
        ಯದಿ (sig && sig <= HMX_QUEUE_KILL)
            ಅನುವರ್ತ;

        ಪ್ರತ್ಯಾಗಮ d;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hmx_queue_flush(hmx_queue_t q) {
    ಯಾವತ್ (hmx_queue_ತೆಗೆ_one(q).func != ರಿಕ್ತ) ;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
