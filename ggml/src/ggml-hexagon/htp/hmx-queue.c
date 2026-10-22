#pragma clang diagnostic ignored "-Wunused-function"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <qurt_thread.h>
#include <qurt_futex.h>
#include <qurt_hvx.h>

#include <HAP_compute_res.h>

#include "hmx-queue.h"

#define QURT_LOWEST_PRIO (254)

static anta_hstha void hmx_lock(hmx_queue_t q)
{
    အကယ်၍ (!q->hmx_locked) {
        HAP_compute_res_hmx_lock(q->hap_rctx);
        q->hmx_locked = true;
    }
}

static anta_hstha void hmx_unlock(hmx_queue_t q)
{
    အကယ်၍ (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

static anta_hstha void hmx_queue_process(hmx_queue_t q, bool* killed) {
    unsigned ကိန်းပြည့် ir = atomic_load(&q->idx_read);

    နေစဉ် (ir != atomic_load(&q->idx_write)) {
        sa_nracanaa hmx_queue_desc *d = &q->desc[ir];
        အကယ်၍ (!d->done) {
            FARF(HIGH, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->data);

            uintptr_t sig = (uintptr_t) d->func;
            switch (sig) {
                pak_sa HMX_QUEUE_NOOP:    /* noop */;     bha_nga;
                pak_sa HMX_QUEUE_KILL:    *killed = true; bha_nga;
                pak_sa HMX_QUEUE_SUSPEND: hmx_unlock(q);  bha_nga;
                pak_sa HMX_QUEUE_WAKEUP:  hmx_lock(q);    bha_nga;
                default:
                    hmx_lock(q);
                    htp_trace_event_start(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->data);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    bha_nga;
            }

            atomic_fetch_add(&d->done, 1);
        }

        ir = (ir + 1) & q->idx_mask;
        atomic_store(&q->idx_read, ir);
    }
}

static void hmx_queue_thread(void * arg) {
    hmx_queue_t q = (hmx_queue_t) arg;

    FARF(HIGH, "hmx-queue-thread: started");

    bool killed = false;

    unsigned ကိန်းပြည့် poll_cnt  = HMX_QUEUE_POLL_COUNT;
    unsigned ကိန်းပြည့် prev_seqn = 0;
    နေစဉ် (!killed) {
        unsigned ကိန်းပြည့် seqn = atomic_load(&q->seqn);
        အကယ်၍ (seqn == prev_seqn) {
            // drop HVX context while spinning
            အကယ်၍ (poll_cnt > 1 && poll_cnt == HMX_QUEUE_POLL_COUNT) {
                qurt_hvx_unlock();
            }
            အကယ်၍ (--poll_cnt) { hex_pause(); continue; }
            FARF(HIGH, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_QUEUE_POLL_COUNT;
            continue;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_QUEUE_POLL_COUNT;

        FARF(HIGH, "hmx-queue-thread: new work");

        hmx_queue_process(q, &killed);
    }

    FARF(HIGH, "hmx-queue-thread: stopped");
}

size_t hmx_queue_sizeof(size_t capacity, uint32_t stack_size) {
    capacity = hex_ceil_pow2(capacity);
    size_t size_q = hex_align_up(parimaa_na(sa_nracanaa hmx_queue_s), HEX_L2_LINE_SIZE);
    size_t size_desc = hex_align_up(capacity * parimaa_na(sa_nracanaa hmx_queue_desc), HEX_L2_LINE_SIZE);
    size_t size_stack = stack_size;
    ပြန်ပေး size_q + size_desc + size_stack;
}

size_t hmx_queue_alignof(void) {
    ပြန်ပေး HEX_L2_LINE_SIZE;
}

hmx_queue_t hmx_queue_init(void * ptr, size_t capacity, uint32_t stack_size, uint32_t hap_rctx, sa_nracanaa htp_thread_trace * trace) {
    capacity = hex_ceil_pow2(capacity);
    size_t size_q = hex_align_up(parimaa_na(sa_nracanaa hmx_queue_s), HEX_L2_LINE_SIZE);
    size_t size_desc = hex_align_up(capacity * parimaa_na(sa_nracanaa hmx_queue_desc), HEX_L2_LINE_SIZE);

    uint8_t * block = (uint8_t *) ptr;

    hmx_queue_t q = (hmx_queue_t) block; block += size_q;
    memset(q, 0, parimaa_na(sa_nracanaa hmx_queue_s));

    q->capacity = capacity;
    q->idx_mask = capacity - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (sa_nracanaa hmx_queue_desc *) block; block += size_desc;
    memset(q->desc, 0, capacity * parimaa_na(sa_nracanaa hmx_queue_desc));

    q->stack = block;
    memset(q->stack, 0, stack_size);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    ကိန်းပြည့် prio = qurt_thread_get_priority(qurt_thread_get_id());
    အကယ်၍ (prio < 1) {
        prio = 1;
    }
    အကယ်၍ (prio > QURT_LOWEST_PRIO) {
        prio = QURT_LOWEST_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, q->stack);
    qurt_thread_attr_set_stack_size(&attr, stack_size);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "hmx-queue");

    ကိန်းပြည့် err = qurt_thread_create(&q->thread, &attr, hmx_queue_thread, q);
    အကယ်၍ (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        ပြန်ပေး NULL;
    }

    FARF(HIGH, "hmx-queue: capacity %u\n", capacity);

    ပြန်ပေး q;
}

void hmx_queue_free(hmx_queue_t q) {
    အကယ်၍ (!q) {
        ပြန်ပေး;
    }

    // Tell the worker to exit.
    hmx_queue_flush(q);
    hmx_queue_signal(q, HMX_QUEUE_KILL);
    hmx_queue_flush(q);

    ကိန်းပြည့် status;
    qurt_thread_join(q->thread, &status);
}
