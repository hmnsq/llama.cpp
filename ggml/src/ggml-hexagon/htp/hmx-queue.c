#pragma clang diagnostic ignored "-Wunused-function"

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव <qurt_thread.h>
#अन्तर्भाव <qurt_futex.h>
#अन्तर्भाव <qurt_hvx.h>

#अन्तर्भाव <HAP_compute_res.h>

#अन्तर्भाव "hmx-queue.h"

#define QURT_LOWEST_PRIO (254)

स्थिर अन्तःस्थ शून्य hmx_lock(hmx_queue_t q)
{
    यदि (!q->hmx_locked) {
        HAP_compute_res_hmx_lock(q->hap_rctx);
        q->hmx_locked = true;
    }
}

स्थिर अन्तःस्थ शून्य hmx_unlock(hmx_queue_t q)
{
    यदि (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

स्थिर अन्तःस्थ शून्य hmx_queue_process(hmx_queue_t q, bool* killed) {
    अचिन्हित पूर्णाङ्क ir = atomic_load(&q->idx_read);

    जबसम्म (ir != atomic_load(&q->idx_write)) {
        संरचना hmx_queue_desc *d = &q->desc[ir];
        यदि (!d->done) {
            FARF(माथिल्लो_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->data);

            uintptr_t sig = (uintptr_t) d->func;
            विकल्प (sig) {
                पक्ष HMX_QUEUE_NOOP:    /* noop */;     रोक;
                पक्ष HMX_QUEUE_KILL:    *killed = true; रोक;
                पक्ष HMX_QUEUE_SUSPEND: hmx_unlock(q);  रोक;
                पक्ष HMX_QUEUE_WAKEUP:  hmx_lock(q);    रोक;
                स्वभाव:
                    hmx_lock(q);
                    htp_trace_event_start(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->data);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    रोक;
            }

            atomic_fetch_add(&d->done, 1);
        }

        ir = (ir + 1) & q->idx_mask;
        atomic_store(&q->idx_read, ir);
    }
}

स्थिर शून्य hmx_queue_thread(शून्य * arg) {
    hmx_queue_t q = (hmx_queue_t) arg;

    FARF(माथिल्लो_2, "hmx-queue-thread: started");

    bool killed = false;

    अचिन्हित पूर्णाङ्क poll_cnt  = HMX_QUEUE_POLL_गन्ती;
    अचिन्हित पूर्णाङ्क prev_seqn = 0;
    जबसम्म (!killed) {
        अचिन्हित पूर्णाङ्क seqn = atomic_load(&q->seqn);
        यदि (seqn == prev_seqn) {
            // drop HVX context while spinning
            यदि (poll_cnt > 1 && poll_cnt == HMX_QUEUE_POLL_गन्ती) {
                qurt_hvx_unlock();
            }
            यदि (--poll_cnt) { hex_pause(); जारी; }
            FARF(माथिल्लो_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_QUEUE_POLL_गन्ती;
            जारी;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_QUEUE_POLL_गन्ती;

        FARF(माथिल्लो_2, "hmx-queue-thread: new work");

        hmx_queue_process(q, &killed);
    }

    FARF(माथिल्लो_2, "hmx-queue-thread: stopped");
}

माप_प्रकार hmx_queue_sizeof(माप_प्रकार capacity, uint32_t थाक_size) {
    capacity = hex_ceil_pow2(capacity);
    माप_प्रकार size_q = hex_align_up(परिमाण(संरचना hmx_queue_s), HEX_L2_पङ्क्ति_SIZE);
    माप_प्रकार size_desc = hex_align_up(capacity * परिमाण(संरचना hmx_queue_desc), HEX_L2_पङ्क्ति_SIZE);
    माप_प्रकार size_थाक = थाक_size;
    फर्काऊ size_q + size_desc + size_थाक;
}

माप_प्रकार hmx_queue_alignof(शून्य) {
    फर्काऊ HEX_L2_पङ्क्ति_SIZE;
}

hmx_queue_t hmx_queue_init(शून्य * ptr, माप_प्रकार capacity, uint32_t थाक_size, uint32_t hap_rctx, संरचना htp_thread_trace * trace) {
    capacity = hex_ceil_pow2(capacity);
    माप_प्रकार size_q = hex_align_up(परिमाण(संरचना hmx_queue_s), HEX_L2_पङ्क्ति_SIZE);
    माप_प्रकार size_desc = hex_align_up(capacity * परिमाण(संरचना hmx_queue_desc), HEX_L2_पङ्क्ति_SIZE);

    uint8_t * block = (uint8_t *) ptr;

    hmx_queue_t q = (hmx_queue_t) block; block += size_q;
    स_रखो(q, 0, परिमाण(संरचना hmx_queue_s));

    q->capacity = capacity;
    q->idx_mask = capacity - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (संरचना hmx_queue_desc *) block; block += size_desc;
    स_रखो(q->desc, 0, capacity * परिमाण(संरचना hmx_queue_desc));

    q->थाक = block;
    स_रखो(q->थाक, 0, थाक_size);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    पूर्णाङ्क prio = qurt_thread_get_priority(qurt_thread_get_id());
    यदि (prio < 1) {
        prio = 1;
    }
    यदि (prio > QURT_LOWEST_PRIO) {
        prio = QURT_LOWEST_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_थाक_addr(&attr, q->थाक);
    qurt_thread_attr_set_थाक_size(&attr, थाक_size);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "hmx-queue");

    पूर्णाङ्क err = qurt_thread_create(&q->thread, &attr, hmx_queue_thread, q);
    यदि (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        फर्काऊ रिक्त;
    }

    FARF(माथिल्लो_2, "hmx-queue: capacity %u\n", capacity);

    फर्काऊ q;
}

शून्य hmx_queue_free(hmx_queue_t q) {
    यदि (!q) {
        फर्काऊ;
    }

    // Tell the worker to exit.
    hmx_queue_flush(q);
    hmx_queue_signal(q, HMX_QUEUE_KILL);
    hmx_queue_flush(q);

    पूर्णाङ्क status;
    qurt_thread_join(q->thread, &status);
}
