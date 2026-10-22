#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stdatomic.h>

#અન્તર્ભાવ <hexagon_types.h>
#અન્તર્ભાવ <qurt_thread.h>
#અન્તર્ભાવ <qurt_futex.h>
#અન્તર્ભાવ <HAP_farf.h>

#અન્તર્ભાવ "hex-utils.h"
#અન્તર્ભાવ "hex-profile.h"

#ifdef __cplusplus
બાહ્ય "C" {
#endif

#જો __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_ગણતરી 2000
#નહીંતર
#define HMX_QUEUE_POLL_ગણતરી 1
#endif

પ્રકારવ્યાખ્યા શૂન્ય (*hmx_queue_func)(શૂન્ય *);

// Dummy funcs used as signals
ગણના hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

સંરચના hmx_queue_desc {
    hmx_queue_func   func;
    શૂન્ય *           data;
    atomic_uint      done;
};

સંરચના hmx_queue_s {
    સંરચના hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned પૂર્ણાંક     idx_કાઢો;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    શૂન્ય *           ઢગલો;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    સંરચના htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

પ્રકારવ્યાખ્યા સંરચના hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t ઢગલો_size);
size_t      hmx_queue_alignof(શૂન્ય);
hmx_queue_t hmx_queue_init(શૂન્ય * ptr, size_t capacity, uint32_t ઢગલો_size, uint32_t hap_rctx, સંરચના htp_thread_trace * trace);
શૂન્ય        hmx_queue_free(hmx_queue_t q);

સ્થિર અંતઃરેખ સંરચના hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, શૂન્ય * data) {
    સંરચના hmx_queue_desc d = { func, data };
    પાછા d;
}

સ્થિર અંતઃરેખ bool hmx_queue_ધકેલો(hmx_queue_t q, સંરચના hmx_queue_desc d) {
    unsigned પૂર્ણાંક ir = atomic_load(&q->idx_read);
    unsigned પૂર્ણાંક iw = atomic_load(&q->idx_write);

    જો (((iw + 1) & q->idx_mask) == ir) {
        FARF(ઉપલું_2, "hmx-queue-push: queue is full\n");
        પાછા false;
    }

    atomic_store(&d.done, 0);

    FARF(ઉપલું_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    પાછા true;
}

સ્થિર અંતઃરેખ bool hmx_queue_signal(hmx_queue_t q, ગણના hmx_queue_signal sig) {
    પાછા hmx_queue_ધકેલો(q, hmx_queue_make_desc((hmx_queue_func) sig, ખાલી));
}

સ્થિર અંતઃરેખ bool hmx_queue_empty(hmx_queue_t q) {
    પાછા q->idx_કાઢો == atomic_load(&q->idx_write);
}

સ્થિર અંતઃરેખ uint32_t hmx_queue_depth(hmx_queue_t q) {
    પાછા (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

સ્થિર અંતઃરેખ uint32_t hmx_queue_capacity(hmx_queue_t q) {
    પાછા q->capacity;
}

સ્થિર અંતઃરેખ સંરચના hmx_queue_desc hmx_queue_કાઢો_one(hmx_queue_t q) {
    unsigned પૂર્ણાંક ip = q->idx_કાઢો;
    unsigned પૂર્ણાંક iw = atomic_load(&q->idx_write);

    સંરચના hmx_queue_desc rd = { ખાલી, ખાલી };
    જો (ip == iw) {
        પાછા rd;
    }

    // Wait for desc to complete
    સંરચના hmx_queue_desc * d = &q->desc[ip];
    જ્યારે (!atomic_load(&d->done)) {
        FARF(ઉપલું_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_કાઢો = (ip + 1) & q->idx_mask;

    FARF(ઉપલું_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    પાછા rd;
}

સ્થિર અંતઃરેખ સંરચના hmx_queue_desc hmx_queue_કાઢો(hmx_queue_t q) {
    જ્યારે (1) {
        સંરચના hmx_queue_desc d = hmx_queue_કાઢો_one(q);

        uint32_t sig = (uint32_t) d.func;
        જો (sig && sig <= HMX_QUEUE_KILL)
            ચાલુ;

        પાછા d;
    }
}

સ્થિર અંતઃરેખ શૂન્ય hmx_queue_flush(hmx_queue_t q) {
    જ્યારે (hmx_queue_કાઢો_one(q).func != ખાલી) ;
}

સ્થિર અંતઃરેખ શૂન્ય hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

સ્થિર અંતઃરેખ શૂન્ય hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
