#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#অন্তর্ভাৱ <stdbool.h>
#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <stdatomic.h>

#অন্তর্ভাৱ <hexagon_types.h>
#অন্তর্ভাৱ <qurt_thread.h>
#অন্তর্ভাৱ <qurt_futex.h>
#অন্তর্ভাৱ <HAP_farf.h>

#অন্তর্ভাৱ "hex-utils.h"
#অন্তর্ভাৱ "hex-profile.h"

#ifdef __cplusplus
বাহ্য "C" {
#endif

#যদি __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_গুনতি 2000
#অন্যথা
#define HMX_QUEUE_POLL_গুনতি 1
#endif

প্রকারনির্দেশ শূন্য (*hmx_queue_func)(শূন্য *);

// Dummy funcs used as signals
গণনা hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

সংরচনা hmx_queue_desc {
    hmx_queue_func   func;
    শূন্য *           data;
    atomic_uint      done;
};

সংরচনা hmx_queue_s {
    সংরচনা hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned পূর্ণাঙ্ক     idx_বেৰ_কৰো;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    শূন্য *           স্তূপ;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    সংরচনা htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

প্রকারনির্দেশ সংরচনা hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t স্তূপ_size);
size_t      hmx_queue_alignof(শূন্য);
hmx_queue_t hmx_queue_init(শূন্য * ptr, size_t capacity, uint32_t স্তূপ_size, uint32_t hap_rctx, সংরচনা htp_thread_trace * trace);
শূন্য        hmx_queue_free(hmx_queue_t q);

স্থির অন্তঃস্থ সংরচনা hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, শূন্য * data) {
    সংরচনা hmx_queue_desc d = { func, data };
    ঘূৰাই d;
}

স্থির অন্তঃস্থ bool hmx_queue_ঠেলো(hmx_queue_t q, সংরচনা hmx_queue_desc d) {
    unsigned পূর্ণাঙ্ক ir = atomic_load(&q->idx_read);
    unsigned পূর্ণাঙ্ক iw = atomic_load(&q->idx_write);

    যদি (((iw + 1) & q->idx_mask) == ir) {
        FARF(উপৰেৰ_2, "hmx-queue-push: queue is full\n");
        ঘূৰাই false;
    }

    atomic_store(&d.done, 0);

    FARF(উপৰেৰ_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    ঘূৰাই true;
}

স্থির অন্তঃস্থ bool hmx_queue_signal(hmx_queue_t q, গণনা hmx_queue_signal sig) {
    ঘূৰাই hmx_queue_ঠেলো(q, hmx_queue_make_desc((hmx_queue_func) sig, রিক্ত));
}

স্থির অন্তঃস্থ bool hmx_queue_empty(hmx_queue_t q) {
    ঘূৰাই q->idx_বেৰ_কৰো == atomic_load(&q->idx_write);
}

স্থির অন্তঃস্থ uint32_t hmx_queue_depth(hmx_queue_t q) {
    ঘূৰাই (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

স্থির অন্তঃস্থ uint32_t hmx_queue_capacity(hmx_queue_t q) {
    ঘূৰাই q->capacity;
}

স্থির অন্তঃস্থ সংরচনা hmx_queue_desc hmx_queue_বেৰ_কৰো_one(hmx_queue_t q) {
    unsigned পূর্ণাঙ্ক ip = q->idx_বেৰ_কৰো;
    unsigned পূর্ণাঙ্ক iw = atomic_load(&q->idx_write);

    সংরচনা hmx_queue_desc rd = { রিক্ত, রিক্ত };
    যদি (ip == iw) {
        ঘূৰাই rd;
    }

    // Wait for desc to complete
    সংরচনা hmx_queue_desc * d = &q->desc[ip];
    যেতিয়ালৈ (!atomic_load(&d->done)) {
        FARF(উপৰেৰ_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_বেৰ_কৰো = (ip + 1) & q->idx_mask;

    FARF(উপৰেৰ_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    ঘূৰাই rd;
}

স্থির অন্তঃস্থ সংরচনা hmx_queue_desc hmx_queue_বেৰ_কৰো(hmx_queue_t q) {
    যেতিয়ালৈ (1) {
        সংরচনা hmx_queue_desc d = hmx_queue_বেৰ_কৰো_one(q);

        uint32_t sig = (uint32_t) d.func;
        যদি (sig && sig <= HMX_QUEUE_KILL)
            অনুৱর্ত;

        ঘূৰাই d;
    }
}

স্থির অন্তঃস্থ শূন্য hmx_queue_flush(hmx_queue_t q) {
    যেতিয়ালৈ (hmx_queue_বেৰ_কৰো_one(q).func != রিক্ত) ;
}

স্থির অন্তঃস্থ শূন্য hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

স্থির অন্তঃস্থ শূন্য hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
