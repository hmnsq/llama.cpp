#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <stdatomic.h>

#ضمن <hexagon_types.h>
#ضمن <qurt_thread.h>
#ضمن <qurt_futex.h>
#ضمن <HAP_farf.h>

#ضمن "hex-utils.h"
#ضمن "hex-profile.h"

#ifdef __cplusplus
خارجي "C" {
#endif

#إذا __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_کيراءن 2000
#وإلا
#define HMX_QUEUE_POLL_کيراءن 1
#endif

تعريفنوع فراغ (*hmx_queue_func)(فراغ *);

// Dummy funcs used as signals
تعداد hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

بنية hmx_queue_desc {
    hmx_queue_func   func;
    فراغ *           data;
    atomic_uint      done;
};

بنية hmx_queue_s {
    بنية hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned صحيح     idx_کلوار;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    فراغ *           تيندنن;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    بنية htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

تعريفنوع بنية hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t تيندنن_size);
size_t      hmx_queue_alignof(فراغ);
hmx_queue_t hmx_queue_init(فراغ * ptr, size_t capacity, uint32_t تيندنن_size, uint32_t hap_rctx, بنية htp_thread_trace * trace);
فراغ        hmx_queue_free(hmx_queue_t q);

ساكن مضمن بنية hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, فراغ * data) {
    بنية hmx_queue_desc d = { func, data };
    أرجع d;
}

ساكن مضمن bool hmx_queue_تولق(hmx_queue_t q, بنية hmx_queue_desc d) {
    unsigned صحيح ir = atomic_load(&q->idx_read);
    unsigned صحيح iw = atomic_load(&q->idx_write);

    إذا (((iw + 1) & q->idx_mask) == ir) {
        FARF(تيڠݢي_2, "hmx-queue-push: queue is full\n");
        أرجع false;
    }

    atomic_store(&d.done, 0);

    FARF(تيڠݢي_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    أرجع true;
}

ساكن مضمن bool hmx_queue_signal(hmx_queue_t q, تعداد hmx_queue_signal sig) {
    أرجع hmx_queue_تولق(q, hmx_queue_make_desc((hmx_queue_func) sig, عدم));
}

ساكن مضمن bool hmx_queue_empty(hmx_queue_t q) {
    أرجع q->idx_کلوار == atomic_load(&q->idx_write);
}

ساكن مضمن uint32_t hmx_queue_depth(hmx_queue_t q) {
    أرجع (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

ساكن مضمن uint32_t hmx_queue_capacity(hmx_queue_t q) {
    أرجع q->capacity;
}

ساكن مضمن بنية hmx_queue_desc hmx_queue_کلوار_one(hmx_queue_t q) {
    unsigned صحيح ip = q->idx_کلوار;
    unsigned صحيح iw = atomic_load(&q->idx_write);

    بنية hmx_queue_desc rd = { عدم, عدم };
    إذا (ip == iw) {
        أرجع rd;
    }

    // Wait for desc to complete
    بنية hmx_queue_desc * d = &q->desc[ip];
    طالما (!atomic_load(&d->done)) {
        FARF(تيڠݢي_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_کلوار = (ip + 1) & q->idx_mask;

    FARF(تيڠݢي_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    أرجع rd;
}

ساكن مضمن بنية hmx_queue_desc hmx_queue_کلوار(hmx_queue_t q) {
    طالما (1) {
        بنية hmx_queue_desc d = hmx_queue_کلوار_one(q);

        uint32_t sig = (uint32_t) d.func;
        إذا (sig && sig <= HMX_QUEUE_KILL)
            استمر;

        أرجع d;
    }
}

ساكن مضمن فراغ hmx_queue_flush(hmx_queue_t q) {
    طالما (hmx_queue_کلوار_one(q).func != عدم) ;
}

ساكن مضمن فراغ hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

ساكن مضمن فراغ hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
