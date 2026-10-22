#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>

#include <hexagon_types.h>
#include <qurt_thread.h>
#include <qurt_futex.h>
#include <HAP_farf.h>

#include "hex-utils.h"
#include "hex-profile.h"

#ifdef __cplusplus
extern "C" {
#endif

#만약 __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_COUNT 2000
#아니면
#define HMX_QUEUE_POLL_COUNT 1
#endif

typedef void (*hmx_queue_func)(void *);

// Dummy funcs used as signals
enum hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

struct hmx_queue_desc {
    hmx_queue_func   func;
    void *           data;
    atomic_uint      done;
};

struct hmx_queue_s {
    struct hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned 정수     idx_pop;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    void *           stack;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    struct htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

typedef struct hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t stack_size);
size_t      hmx_queue_alignof(void);
hmx_queue_t hmx_queue_init(void * ptr, size_t capacity, uint32_t stack_size, uint32_t hap_rctx, struct htp_thread_trace * trace);
void        hmx_queue_free(hmx_queue_t q);

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };
    반환 d;
}

static inline bool hmx_queue_push(hmx_queue_t q, struct hmx_queue_desc d) {
    unsigned 정수 ir = atomic_load(&q->idx_read);
    unsigned 정수 iw = atomic_load(&q->idx_write);

    만약 (((iw + 1) & q->idx_mask) == ir) {
        FARF(HIGH, "hmx-queue-push: queue is full\n");
        반환 false;
    }

    atomic_store(&d.done, 0);

    FARF(HIGH, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    반환 true;
}

static inline bool hmx_queue_signal(hmx_queue_t q, enum hmx_queue_signal sig) {
    반환 hmx_queue_push(q, hmx_queue_make_desc((hmx_queue_func) sig, NULL));
}

static inline bool hmx_queue_empty(hmx_queue_t q) {
    반환 q->idx_pop == atomic_load(&q->idx_write);
}

static inline uint32_t hmx_queue_depth(hmx_queue_t q) {
    반환 (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

static inline uint32_t hmx_queue_capacity(hmx_queue_t q) {
    반환 q->capacity;
}

static inline struct hmx_queue_desc hmx_queue_pop_one(hmx_queue_t q) {
    unsigned 정수 ip = q->idx_pop;
    unsigned 정수 iw = atomic_load(&q->idx_write);

    struct hmx_queue_desc rd = { NULL, NULL };
    만약 (ip == iw) {
        반환 rd;
    }

    // Wait for desc to complete
    struct hmx_queue_desc * d = &q->desc[ip];
    동안 (!atomic_load(&d->done)) {
        FARF(HIGH, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_pop = (ip + 1) & q->idx_mask;

    FARF(HIGH, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    반환 rd;
}

static inline struct hmx_queue_desc hmx_queue_pop(hmx_queue_t q) {
    동안 (1) {
        struct hmx_queue_desc d = hmx_queue_pop_one(q);

        uint32_t sig = (uint32_t) d.func;
        만약 (sig && sig <= HMX_QUEUE_KILL)
            continue;

        반환 d;
    }
}

static inline void hmx_queue_flush(hmx_queue_t q) {
    동안 (hmx_queue_pop_one(q).func != NULL) ;
}

static inline void hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

static inline void hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
