#pragma clang diagnostic ignored "-Wunused-function"

#ضمن <stdbool.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#ضمن <qurt_thread.h>
#ضمن <qurt_futex.h>
#ضمن <qurt_hvx.h>

#ضمن <HAP_compute_res.h>

#ضمن "hmx-queue.h"

#define QURT_LOWEST_PRIO (254)

ساكن مضمن فراغ hmx_lock(hmx_queue_t q)
{
    إذا (!q->hmx_locked) {
        HAP_compute_res_hmx_lock(q->hap_rctx);
        q->hmx_locked = true;
    }
}

ساكن مضمن فراغ hmx_unlock(hmx_queue_t q)
{
    إذا (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

ساكن مضمن فراغ hmx_queue_process(hmx_queue_t q, bool* killed) {
    unsigned صحيح ir = atomic_load(&q->idx_read);

    طالما (ir != atomic_load(&q->idx_write)) {
        بنية hmx_queue_desc *d = &q->desc[ir];
        إذا (!d->done) {
            FARF(يۇقىرى_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->data);

            uintptr_t sig = (uintptr_t) d->func;
            بدل (sig) {
                حالة HMX_QUEUE_NOOP:    /* noop */;     اكسر;
                حالة HMX_QUEUE_KILL:    *killed = true; اكسر;
                حالة HMX_QUEUE_SUSPEND: hmx_unlock(q);  اكسر;
                حالة HMX_QUEUE_WAKEUP:  hmx_lock(q);    اكسر;
                افتراضي:
                    hmx_lock(q);
                    htp_trace_event_start(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->data);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    اكسر;
            }

            atomic_fetch_add(&d->done, 1);
        }

        ir = (ir + 1) & q->idx_mask;
        atomic_store(&q->idx_read, ir);
    }
}

ساكن فراغ hmx_queue_thread(فراغ * arg) {
    hmx_queue_t q = (hmx_queue_t) arg;

    FARF(يۇقىرى_2, "hmx-queue-thread: started");

    bool killed = false;

    unsigned صحيح poll_cnt  = HMX_QUEUE_POLL_سانى;
    unsigned صحيح prev_seqn = 0;
    طالما (!killed) {
        unsigned صحيح seqn = atomic_load(&q->seqn);
        إذا (seqn == prev_seqn) {
            // drop HVX context while spinning
            إذا (poll_cnt > 1 && poll_cnt == HMX_QUEUE_POLL_سانى) {
                qurt_hvx_unlock();
            }
            إذا (--poll_cnt) { hex_pause(); استمر; }
            FARF(يۇقىرى_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_QUEUE_POLL_سانى;
            استمر;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_QUEUE_POLL_سانى;

        FARF(يۇقىرى_2, "hmx-queue-thread: new work");

        hmx_queue_process(q, &killed);
    }

    FARF(يۇقىرى_2, "hmx-queue-thread: stopped");
}

size_t hmx_queue_sizeof(size_t capacity, uint32_t دۆۋە_size) {
    capacity = hex_ceil_pow2(capacity);
    size_t size_q = hex_align_up(حجم(بنية hmx_queue_s), HEX_L2_قۇر_SIZE);
    size_t size_desc = hex_align_up(capacity * حجم(بنية hmx_queue_desc), HEX_L2_قۇر_SIZE);
    size_t size_دۆۋە = دۆۋە_size;
    أرجع size_q + size_desc + size_دۆۋە;
}

size_t hmx_queue_alignof(فراغ) {
    أرجع HEX_L2_قۇر_SIZE;
}

hmx_queue_t hmx_queue_init(فراغ * ptr, size_t capacity, uint32_t دۆۋە_size, uint32_t hap_rctx, بنية htp_thread_trace * trace) {
    capacity = hex_ceil_pow2(capacity);
    size_t size_q = hex_align_up(حجم(بنية hmx_queue_s), HEX_L2_قۇر_SIZE);
    size_t size_desc = hex_align_up(capacity * حجم(بنية hmx_queue_desc), HEX_L2_قۇر_SIZE);

    uint8_t * block = (uint8_t *) ptr;

    hmx_queue_t q = (hmx_queue_t) block; block += size_q;
    memset(q, 0, حجم(بنية hmx_queue_s));

    q->capacity = capacity;
    q->idx_mask = capacity - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (بنية hmx_queue_desc *) block; block += size_desc;
    memset(q->desc, 0, capacity * حجم(بنية hmx_queue_desc));

    q->دۆۋە = block;
    memset(q->دۆۋە, 0, دۆۋە_size);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    صحيح prio = qurt_thread_get_priority(qurt_thread_get_id());
    إذا (prio < 1) {
        prio = 1;
    }
    إذا (prio > QURT_LOWEST_PRIO) {
        prio = QURT_LOWEST_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_دۆۋە_addr(&attr, q->دۆۋە);
    qurt_thread_attr_set_دۆۋە_size(&attr, دۆۋە_size);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "hmx-queue");

    صحيح err = qurt_thread_create(&q->thread, &attr, hmx_queue_thread, q);
    إذا (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        أرجع عدم;
    }

    FARF(يۇقىرى_2, "hmx-queue: capacity %u\n", capacity);

    أرجع q;
}

فراغ hmx_queue_free(hmx_queue_t q) {
    إذا (!q) {
        أرجع;
    }

    // Tell the worker to exit.
    hmx_queue_flush(q);
    hmx_queue_signal(q, HMX_QUEUE_KILL);
    hmx_queue_flush(q);

    صحيح status;
    qurt_thread_join(q->thread, &status);
}
