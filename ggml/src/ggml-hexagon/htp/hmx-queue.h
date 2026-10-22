#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <stdatomic.h>

#شامل <hexagon_types.h>
#شامل <qurt_thread.h>
#شامل <qurt_futex.h>
#شامل <HAP_farf.h>

#شامل "hex-utils.h"
#شامل "hex-profile.h"

#ifdef __cplusplus
بیرونی "C" {
#endif

#اگر __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_شمار 2000
#وگرنه
#define HMX_QUEUE_POLL_شمار 1
#endif

تعریفنوع تهی (*hmx_queue_func)(تهی *);

// Dummy funcs used as signals
شمارش hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

ساختار hmx_queue_desc {
    hmx_queue_func   func;
    تهی *           data;
    atomic_uint      done;
};

ساختار hmx_queue_s {
    ساختار hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned صحیح     idx_برداشت;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    تهی *           پشته;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    ساختار htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

تعریفنوع ساختار hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t پشته_size);
size_t      hmx_queue_alignof(تهی);
hmx_queue_t hmx_queue_init(تهی * ptr, size_t capacity, uint32_t پشته_size, uint32_t hap_rctx, ساختار htp_thread_trace * trace);
تهی        hmx_queue_free(hmx_queue_t q);

ایستا درونخطی ساختار hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, تهی * data) {
    ساختار hmx_queue_desc d = { func, data };
    بازگشت d;
}

ایستا درونخطی bool hmx_queue_افزودن(hmx_queue_t q, ساختار hmx_queue_desc d) {
    unsigned صحیح ir = atomic_load(&q->idx_read);
    unsigned صحیح iw = atomic_load(&q->idx_write);

    اگر (((iw + 1) & q->idx_mask) == ir) {
        FARF(بالا_2, "hmx-queue-push: queue is full\n");
        بازگشت false;
    }

    atomic_store(&d.done, 0);

    FARF(بالا_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    بازگشت true;
}

ایستا درونخطی bool hmx_queue_signal(hmx_queue_t q, شمارش hmx_queue_signal sig) {
    بازگشت hmx_queue_افزودن(q, hmx_queue_make_desc((hmx_queue_func) sig, پوچ));
}

ایستا درونخطی bool hmx_queue_empty(hmx_queue_t q) {
    بازگشت q->idx_برداشت == atomic_load(&q->idx_write);
}

ایستا درونخطی uint32_t hmx_queue_depth(hmx_queue_t q) {
    بازگشت (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

ایستا درونخطی uint32_t hmx_queue_capacity(hmx_queue_t q) {
    بازگشت q->capacity;
}

ایستا درونخطی ساختار hmx_queue_desc hmx_queue_برداشت_one(hmx_queue_t q) {
    unsigned صحیح ip = q->idx_برداشت;
    unsigned صحیح iw = atomic_load(&q->idx_write);

    ساختار hmx_queue_desc rd = { پوچ, پوچ };
    اگر (ip == iw) {
        بازگشت rd;
    }

    // Wait for desc to complete
    ساختار hmx_queue_desc * d = &q->desc[ip];
    وقتی (!atomic_load(&d->done)) {
        FARF(بالا_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_برداشت = (ip + 1) & q->idx_mask;

    FARF(بالا_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    بازگشت rd;
}

ایستا درونخطی ساختار hmx_queue_desc hmx_queue_برداشت(hmx_queue_t q) {
    وقتی (1) {
        ساختار hmx_queue_desc d = hmx_queue_برداشت_one(q);

        uint32_t sig = (uint32_t) d.func;
        اگر (sig && sig <= HMX_QUEUE_KILL)
            ادامه;

        بازگشت d;
    }
}

ایستا درونخطی تهی hmx_queue_flush(hmx_queue_t q) {
    وقتی (hmx_queue_برداشت_one(q).func != پوچ) ;
}

ایستا درونخطی تهی hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

ایستا درونخطی تهی hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
