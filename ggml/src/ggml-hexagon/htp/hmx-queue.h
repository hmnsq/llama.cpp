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

#जर __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_मोजणी 2000
#अन्यथा
#define HMX_QUEUE_POLL_मोजणी 1
#endif

प्रकारव्याख्या शून्य (*hmx_queue_func)(शून्य *);

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
    अचिन्हित पूर्णांक     idx_काढा;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    शून्य *           चळत;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    संरचना htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

प्रकारव्याख्या संरचना hmx_queue_s * hmx_queue_t;

माप_प्रकार      hmx_queue_sizeof(माप_प्रकार capacity, uint32_t चळत_size);
माप_प्रकार      hmx_queue_alignof(शून्य);
hmx_queue_t hmx_queue_init(शून्य * ptr, माप_प्रकार capacity, uint32_t चळत_size, uint32_t hap_rctx, संरचना htp_thread_trace * trace);
शून्य        hmx_queue_free(hmx_queue_t q);

स्थिर ओळीत संरचना hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, शून्य * data) {
    संरचना hmx_queue_desc d = { func, data };
    परत d;
}

स्थिर ओळीत bool hmx_queue_ढकला(hmx_queue_t q, संरचना hmx_queue_desc d) {
    अचिन्हित पूर्णांक ir = atomic_load(&q->idx_read);
    अचिन्हित पूर्णांक iw = atomic_load(&q->idx_write);

    जर (((iw + 1) & q->idx_mask) == ir) {
        FARF(वरचा_2, "hmx-queue-push: queue is full\n");
        परत false;
    }

    atomic_store(&d.done, 0);

    FARF(वरचा_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    परत true;
}

स्थिर ओळीत bool hmx_queue_signal(hmx_queue_t q, गणना hmx_queue_signal sig) {
    परत hmx_queue_ढकला(q, hmx_queue_make_desc((hmx_queue_func) sig, रिक्त));
}

स्थिर ओळीत bool hmx_queue_empty(hmx_queue_t q) {
    परत q->idx_काढा == atomic_load(&q->idx_write);
}

स्थिर ओळीत uint32_t hmx_queue_depth(hmx_queue_t q) {
    परत (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

स्थिर ओळीत uint32_t hmx_queue_capacity(hmx_queue_t q) {
    परत q->capacity;
}

स्थिर ओळीत संरचना hmx_queue_desc hmx_queue_काढा_one(hmx_queue_t q) {
    अचिन्हित पूर्णांक ip = q->idx_काढा;
    अचिन्हित पूर्णांक iw = atomic_load(&q->idx_write);

    संरचना hmx_queue_desc rd = { रिक्त, रिक्त };
    जर (ip == iw) {
        परत rd;
    }

    // Wait for desc to complete
    संरचना hmx_queue_desc * d = &q->desc[ip];
    पर्यंत (!atomic_load(&d->done)) {
        FARF(वरचा_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_काढा = (ip + 1) & q->idx_mask;

    FARF(वरचा_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    परत rd;
}

स्थिर ओळीत संरचना hmx_queue_desc hmx_queue_काढा(hmx_queue_t q) {
    पर्यंत (1) {
        संरचना hmx_queue_desc d = hmx_queue_काढा_one(q);

        uint32_t sig = (uint32_t) d.func;
        जर (sig && sig <= HMX_QUEUE_KILL)
            सुरूठेवा;

        परत d;
    }
}

स्थिर ओळीत शून्य hmx_queue_flush(hmx_queue_t q) {
    पर्यंत (hmx_queue_काढा_one(q).func != रिक्त) ;
}

स्थिर ओळीत शून्य hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

स्थिर ओळीत शून्य hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
