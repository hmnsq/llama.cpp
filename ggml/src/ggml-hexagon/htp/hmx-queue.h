#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <stdatomic.h>

#അന്തര്ഭാവ <hexagon_types.h>
#അന്തര്ഭാവ <qurt_thread.h>
#അന്തര്ഭാവ <qurt_futex.h>
#അന്തര്ഭാവ <HAP_farf.h>

#അന്തര്ഭാവ "hex-utils.h"
#അന്തര്ഭാവ "hex-profile.h"

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

#എങ്കിൽ __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_എണ്ണം 2000
#അല്ലെങ്കിൽ
#define HMX_QUEUE_POLL_എണ്ണം 1
#endif

തരനിർവചനം ശൂന്യം (*hmx_queue_func)(ശൂന്യം *);

// Dummy funcs used as signals
എണ്ണൽ hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

ഘടന hmx_queue_desc {
    hmx_queue_func   func;
    ശൂന്യം *           data;
    atomic_uint      done;
};

ഘടന hmx_queue_s {
    ഘടന hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned പൂർണ്ണസംഖ്യ     idx_എടുക്കുക;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    ശൂന്യം *           അട്ടി;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    ഘടന htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

തരനിർവചനം ഘടന hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t അട്ടി_size);
size_t      hmx_queue_alignof(ശൂന്യം);
hmx_queue_t hmx_queue_init(ശൂന്യം * ptr, size_t capacity, uint32_t അട്ടി_size, uint32_t hap_rctx, ഘടന htp_thread_trace * trace);
ശൂന്യം        hmx_queue_free(hmx_queue_t q);

സ്ഥിതം വരിയിൽ ഘടന hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, ശൂന്യം * data) {
    ഘടന hmx_queue_desc d = { func, data };
    തിരികെ d;
}

സ്ഥിതം വരിയിൽ bool hmx_queue_തള്ളുക(hmx_queue_t q, ഘടന hmx_queue_desc d) {
    unsigned പൂർണ്ണസംഖ്യ ir = atomic_load(&q->idx_read);
    unsigned പൂർണ്ണസംഖ്യ iw = atomic_load(&q->idx_write);

    എങ്കിൽ (((iw + 1) & q->idx_mask) == ir) {
        FARF(മുകളിലെ_2, "hmx-queue-push: queue is full\n");
        തിരികെ false;
    }

    atomic_store(&d.done, 0);

    FARF(മുകളിലെ_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    തിരികെ true;
}

സ്ഥിതം വരിയിൽ bool hmx_queue_signal(hmx_queue_t q, എണ്ണൽ hmx_queue_signal sig) {
    തിരികെ hmx_queue_തള്ളുക(q, hmx_queue_make_desc((hmx_queue_func) sig, ശൂന്യ));
}

സ്ഥിതം വരിയിൽ bool hmx_queue_empty(hmx_queue_t q) {
    തിരികെ q->idx_എടുക്കുക == atomic_load(&q->idx_write);
}

സ്ഥിതം വരിയിൽ uint32_t hmx_queue_depth(hmx_queue_t q) {
    തിരികെ (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

സ്ഥിതം വരിയിൽ uint32_t hmx_queue_capacity(hmx_queue_t q) {
    തിരികെ q->capacity;
}

സ്ഥിതം വരിയിൽ ഘടന hmx_queue_desc hmx_queue_എടുക്കുക_one(hmx_queue_t q) {
    unsigned പൂർണ്ണസംഖ്യ ip = q->idx_എടുക്കുക;
    unsigned പൂർണ്ണസംഖ്യ iw = atomic_load(&q->idx_write);

    ഘടന hmx_queue_desc rd = { ശൂന്യ, ശൂന്യ };
    എങ്കിൽ (ip == iw) {
        തിരികെ rd;
    }

    // Wait for desc to complete
    ഘടന hmx_queue_desc * d = &q->desc[ip];
    ആകുമ്പോൾ (!atomic_load(&d->done)) {
        FARF(മുകളിലെ_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_എടുക്കുക = (ip + 1) & q->idx_mask;

    FARF(മുകളിലെ_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    തിരികെ rd;
}

സ്ഥിതം വരിയിൽ ഘടന hmx_queue_desc hmx_queue_എടുക്കുക(hmx_queue_t q) {
    ആകുമ്പോൾ (1) {
        ഘടന hmx_queue_desc d = hmx_queue_എടുക്കുക_one(q);

        uint32_t sig = (uint32_t) d.func;
        എങ്കിൽ (sig && sig <= HMX_QUEUE_KILL)
            തുടരുക;

        തിരികെ d;
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_queue_flush(hmx_queue_t q) {
    ആകുമ്പോൾ (hmx_queue_എടുക്കുക_one(q).func != ശൂന്യ) ;
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
