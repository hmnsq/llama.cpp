#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stdatomic.h>

#অন্তর্ভাব <hexagon_types.h>
#অন্তর্ভাব <qurt_thread.h>
#অন্তর্ভাব <qurt_futex.h>
#অন্তর্ভাব <HAP_farf.h>

#অন্তর্ভাব "hex-utils.h"
#অন্তর্ভাব "hex-profile.h"

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

#যদি __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_গুনতি 2000
#অন্যথা
#define HMX_QUEUE_POLL_গুনতি 1
#endif

প্রকারলেখ শূন্য (*hmx_queue_func)(শূন্য *);

// Dummy funcs used as signals
গণনা hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

কাঠামো hmx_queue_desc {
    hmx_queue_func   func;
    শূন্য *           তথ্য;
    atomic_uint      done;
};

কাঠামো hmx_queue_s {
    কাঠামো hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned পূর্ণসংখ্যা     idx_বের_করো;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    শূন্য *           স্তূপ;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    কাঠামো htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

প্রকারলেখ কাঠামো hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t স্তূপ_আকার);
size_t      hmx_queue_alignof(শূন্য);
hmx_queue_t hmx_queue_init(শূন্য * ptr, size_t capacity, uint32_t স্তূপ_আকার, uint32_t hap_rctx, কাঠামো htp_thread_trace * trace);
শূন্য        hmx_queue_free(hmx_queue_t q);

স্থির অন্তঃরেখ কাঠামো hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, শূন্য * তথ্য) {
    কাঠামো hmx_queue_desc d = { func, তথ্য };
    ফেরত d;
}

স্থির অন্তঃরেখ bool hmx_queue_ঠেলো(hmx_queue_t q, কাঠামো hmx_queue_desc d) {
    unsigned পূর্ণসংখ্যা ir = atomic_load(&q->idx_read);
    unsigned পূর্ণসংখ্যা iw = atomic_load(&q->idx_write);

    যদি (((iw + 1) & q->idx_mask) == ir) {
        FARF(উপরের_2, "hmx-queue-push: queue is full\n");
        ফেরত false;
    }

    atomic_store(&d.done, 0);

    FARF(উপরের_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.তথ্য);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    ফেরত true;
}

স্থির অন্তঃরেখ bool hmx_queue_signal(hmx_queue_t q, গণনা hmx_queue_signal sig) {
    ফেরত hmx_queue_ঠেলো(q, hmx_queue_make_desc((hmx_queue_func) sig, ফাঁকা));
}

স্থির অন্তঃরেখ bool hmx_queue_empty(hmx_queue_t q) {
    ফেরত q->idx_বের_করো == atomic_load(&q->idx_write);
}

স্থির অন্তঃরেখ uint32_t hmx_queue_depth(hmx_queue_t q) {
    ফেরত (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

স্থির অন্তঃরেখ uint32_t hmx_queue_capacity(hmx_queue_t q) {
    ফেরত q->capacity;
}

স্থির অন্তঃরেখ কাঠামো hmx_queue_desc hmx_queue_বের_করো_one(hmx_queue_t q) {
    unsigned পূর্ণসংখ্যা ip = q->idx_বের_করো;
    unsigned পূর্ণসংখ্যা iw = atomic_load(&q->idx_write);

    কাঠামো hmx_queue_desc rd = { ফাঁকা, ফাঁকা };
    যদি (ip == iw) {
        ফেরত rd;
    }

    // Wait for desc to complete
    কাঠামো hmx_queue_desc * d = &q->desc[ip];
    যখন (!atomic_load(&d->done)) {
        FARF(উপরের_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_বের_করো = (ip + 1) & q->idx_mask;

    FARF(উপরের_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.তথ্য);
    ফেরত rd;
}

স্থির অন্তঃরেখ কাঠামো hmx_queue_desc hmx_queue_বের_করো(hmx_queue_t q) {
    যখন (1) {
        কাঠামো hmx_queue_desc d = hmx_queue_বের_করো_one(q);

        uint32_t sig = (uint32_t) d.func;
        যদি (sig && sig <= HMX_QUEUE_KILL)
            চালু;

        ফেরত d;
    }
}

স্থির অন্তঃরেখ শূন্য hmx_queue_flush(hmx_queue_t q) {
    যখন (hmx_queue_বের_করো_one(q).func != ফাঁকা) ;
}

স্থির অন্তঃরেখ শূন্য hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

স্থির অন্তঃরেখ শূন্য hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
