#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stdatomic.h>

#ଅନ୍ତର୍ଭାଵ <hexagon_types.h>
#ଅନ୍ତର୍ଭାଵ <qurt_thread.h>
#ଅନ୍ତର୍ଭାଵ <qurt_futex.h>
#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>

#ଅନ୍ତର୍ଭାଵ "hex-utils.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

#ଯଦି __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_ଗଣନା 2000
#ଅନ୍ୟଥା
#define HMX_QUEUE_POLL_ଗଣନା 1
#endif

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ (*hmx_queue_func)(ଶୂନ୍ୟ *);

// Dummy funcs used as signals
ଗଣନା hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

ସଂରଚନା hmx_queue_desc {
    hmx_queue_func   func;
    ଶୂନ୍ୟ *           data;
    atomic_uint      done;
};

ସଂରଚନା hmx_queue_s {
    ସଂରଚନା hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା     idx_କାଢ଼;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    ଶୂନ୍ୟ *           ଗଦା;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    ସଂରଚନା htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t ଗଦା_size);
size_t      hmx_queue_alignof(ଶୂନ୍ୟ);
hmx_queue_t hmx_queue_init(ଶୂନ୍ୟ * ptr, size_t capacity, uint32_t ଗଦା_size, uint32_t hap_rctx, ସଂରଚନା htp_thread_trace * trace);
ଶୂନ୍ୟ        hmx_queue_free(hmx_queue_t q);

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ସଂରଚନା hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, ଶୂନ୍ୟ * data) {
    ସଂରଚନା hmx_queue_desc d = { func, data };
    ଫେରନ୍ତୁ d;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool hmx_queue_ଠେଲ(hmx_queue_t q, ସଂରଚନା hmx_queue_desc d) {
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = atomic_load(&q->idx_read);
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା iw = atomic_load(&q->idx_write);

    ଯଦି (((iw + 1) & q->idx_mask) == ir) {
        FARF(ଉପର_2, "hmx-queue-push: queue is full\n");
        ଫେରନ୍ତୁ false;
    }

    atomic_store(&d.done, 0);

    FARF(ଉପର_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    ଫେରନ୍ତୁ true;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool hmx_queue_signal(hmx_queue_t q, ଗଣନା hmx_queue_signal sig) {
    ଫେରନ୍ତୁ hmx_queue_ଠେଲ(q, hmx_queue_make_desc((hmx_queue_func) sig, ଖାଲି));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool hmx_queue_empty(hmx_queue_t q) {
    ଫେରନ୍ତୁ q->idx_କାଢ଼ == atomic_load(&q->idx_write);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t hmx_queue_depth(hmx_queue_t q) {
    ଫେରନ୍ତୁ (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t hmx_queue_capacity(hmx_queue_t q) {
    ଫେରନ୍ତୁ q->capacity;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ସଂରଚନା hmx_queue_desc hmx_queue_କାଢ଼_one(hmx_queue_t q) {
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ip = q->idx_କାଢ଼;
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା iw = atomic_load(&q->idx_write);

    ସଂରଚନା hmx_queue_desc rd = { ଖାଲି, ଖାଲି };
    ଯଦି (ip == iw) {
        ଫେରନ୍ତୁ rd;
    }

    // Wait for desc to complete
    ସଂରଚନା hmx_queue_desc * d = &q->desc[ip];
    ଯେତେବେଳେ (!atomic_load(&d->done)) {
        FARF(ଉପର_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_କାଢ଼ = (ip + 1) & q->idx_mask;

    FARF(ଉପର_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    ଫେରନ୍ତୁ rd;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ସଂରଚନା hmx_queue_desc hmx_queue_କାଢ଼(hmx_queue_t q) {
    ଯେତେବେଳେ (1) {
        ସଂରଚନା hmx_queue_desc d = hmx_queue_କାଢ଼_one(q);

        uint32_t sig = (uint32_t) d.func;
        ଯଦି (sig && sig <= HMX_QUEUE_KILL)
            ଜାରି;

        ଫେରନ୍ତୁ d;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hmx_queue_flush(hmx_queue_t q) {
    ଯେତେବେଳେ (hmx_queue_କାଢ଼_one(q).func != ଖାଲି) ;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
