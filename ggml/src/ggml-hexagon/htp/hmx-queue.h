#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdatomic.h>

#अन्तर्भाव <hexagon_types.h>
#अन्तर्भाव <qurt_thread.h>
#अन्तर्भाव <qurt_futex.h>
#अन्तर्भाव <HAP_farf.h>

#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "hex-profile.h"

#ifdef __cplusplus
बाह्य "C" {
#endif

#यदि __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_गिनती 2000
#अन्यथा
#define HMX_QUEUE_POLL_गिनती 1
#endif

प्रकारनिर्देश शून्य (*hmx_queue_func)(शून्य *);

// Dummy funcs used as signals
गणना hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

संरचना hmx_queue_desc {
    hmx_queue_func   func;
    शून्य *           data;
    atomic_uint      done;
};

संरचना hmx_queue_s {
    संरचना hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned पूर्णाङ्क     idx_निकालो;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    शून्य *           ढेर;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    संरचना htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

प्रकारनिर्देश संरचना hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t ढेर_size);
size_t      hmx_queue_alignof(शून्य);
hmx_queue_t hmx_queue_init(शून्य * ptr, size_t capacity, uint32_t ढेर_size, uint32_t hap_rctx, संरचना htp_thread_trace * trace);
शून्य        hmx_queue_free(hmx_queue_t q);

स्थिर अन्तःस्थ संरचना hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, शून्य * data) {
    संरचना hmx_queue_desc d = { func, data };
    प्रत्यागम d;
}

स्थिर अन्तःस्थ bool hmx_queue_धकेलो(hmx_queue_t q, संरचना hmx_queue_desc d) {
    unsigned पूर्णाङ्क ir = atomic_load(&q->idx_read);
    unsigned पूर्णाङ्क iw = atomic_load(&q->idx_write);

    यदि (((iw + 1) & q->idx_mask) == ir) {
        FARF(ऊपरी_2, "hmx-queue-push: queue is full\n");
        प्रत्यागम false;
    }

    atomic_store(&d.done, 0);

    FARF(ऊपरी_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    प्रत्यागम true;
}

स्थिर अन्तःस्थ bool hmx_queue_signal(hmx_queue_t q, गणना hmx_queue_signal sig) {
    प्रत्यागम hmx_queue_धकेलो(q, hmx_queue_make_desc((hmx_queue_func) sig, रिक्त));
}

स्थिर अन्तःस्थ bool hmx_queue_empty(hmx_queue_t q) {
    प्रत्यागम q->idx_निकालो == atomic_load(&q->idx_write);
}

स्थिर अन्तःस्थ uint32_t hmx_queue_depth(hmx_queue_t q) {
    प्रत्यागम (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

स्थिर अन्तःस्थ uint32_t hmx_queue_capacity(hmx_queue_t q) {
    प्रत्यागम q->capacity;
}

स्थिर अन्तःस्थ संरचना hmx_queue_desc hmx_queue_निकालो_one(hmx_queue_t q) {
    unsigned पूर्णाङ्क ip = q->idx_निकालो;
    unsigned पूर्णाङ्क iw = atomic_load(&q->idx_write);

    संरचना hmx_queue_desc rd = { रिक्त, रिक्त };
    यदि (ip == iw) {
        प्रत्यागम rd;
    }

    // Wait for desc to complete
    संरचना hmx_queue_desc * d = &q->desc[ip];
    यावत् (!atomic_load(&d->done)) {
        FARF(ऊपरी_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_निकालो = (ip + 1) & q->idx_mask;

    FARF(ऊपरी_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    प्रत्यागम rd;
}

स्थिर अन्तःस्थ संरचना hmx_queue_desc hmx_queue_निकालो(hmx_queue_t q) {
    यावत् (1) {
        संरचना hmx_queue_desc d = hmx_queue_निकालो_one(q);

        uint32_t sig = (uint32_t) d.func;
        यदि (sig && sig <= HMX_QUEUE_KILL)
            अनुवर्त;

        प्रत्यागम d;
    }
}

स्थिर अन्तःस्थ शून्य hmx_queue_flush(hmx_queue_t q) {
    यावत् (hmx_queue_निकालो_one(q).func != रिक्त) ;
}

स्थिर अन्तःस्थ शून्य hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

स्थिर अन्तःस्थ शून्य hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
