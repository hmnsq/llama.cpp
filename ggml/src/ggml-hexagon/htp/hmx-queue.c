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
    जइ (!q->hmx_locked) {
        HAP_compute_res_hmx_lock(q->hap_rctx);
        q->hmx_locked = true;
    }
}

स्थिर अन्तःस्थ शून्य hmx_unlock(hmx_queue_t q)
{
    जइ (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

स्थिर अन्तःस्थ शून्य hmx_queue_process(hmx_queue_t q, bool* killed) {
    unsigned पूर्णाङ्क ir = atomic_load(&q->idx_read);

    जाव (ir != atomic_load(&q->idx_write)) {
        संरचना hmx_queue_desc *d = &q->desc[ir];
        जइ (!d->done) {
            FARF(उच्चः_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->data);

            uintptr_t sig = (uintptr_t) d->func;
            विकल्प (sig) {
                पक्ष HMX_QUEUE_NOOP:    /* noop */;     भङ्ग;
                पक्ष HMX_QUEUE_KILL:    *killed = true; भङ्ग;
                पक्ष HMX_QUEUE_SUSPEND: hmx_unlock(q);  भङ्ग;
                पक्ष HMX_QUEUE_WAKEUP:  hmx_lock(q);    भङ्ग;
                स्वभाव:
                    hmx_lock(q);
                    htp_trace_event_start(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->data);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    भङ्ग;
            }

            atomic_fetch_add(&d->done, 1);
        }

        ir = (ir + 1) & q->idx_mask;
        atomic_store(&q->idx_read, ir);
    }
}

स्थिर शून्य hmx_queue_thread(शून्य * arg) {
    hmx_queue_t q = (hmx_queue_t) arg;

    FARF(उच्चः_2, "hmx-queue-thread: started");

    bool killed = false;

    unsigned पूर्णाङ्क poll_cnt  = HMX_QUEUE_POLL_गणनम्;
    unsigned पूर्णाङ्क prev_seqn = 0;
    जाव (!killed) {
        unsigned पूर्णाङ्क seqn = atomic_load(&q->seqn);
        जइ (seqn == prev_seqn) {
            // drop HVX context while spinning
            जइ (poll_cnt > 1 && poll_cnt == HMX_QUEUE_POLL_गणनम्) {
                qurt_hvx_unlock();
            }
            जइ (--poll_cnt) { hex_pause(); अनुवर्त; }
            FARF(उच्चः_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_QUEUE_POLL_गणनम्;
            अनुवर्त;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_QUEUE_POLL_गणनम्;

        FARF(उच्चः_2, "hmx-queue-thread: new work");

        hmx_queue_process(q, &killed);
    }

    FARF(उच्चः_2, "hmx-queue-thread: stopped");
}

size_t hmx_queue_sizeof(size_t capacity, uint32_t राशिः_size) {
    capacity = hex_ceil_pow2(capacity);
    size_t size_q = hex_align_up(परिमाण(संरचना hmx_queue_s), HEX_L2_पङ्क्तिः_SIZE);
    size_t size_desc = hex_align_up(capacity * परिमाण(संरचना hmx_queue_desc), HEX_L2_पङ्क्तिः_SIZE);
    size_t size_राशिः = राशिः_size;
    प्रत्यागम size_q + size_desc + size_राशिः;
}

size_t hmx_queue_alignof(शून्य) {
    प्रत्यागम HEX_L2_पङ्क्तिः_SIZE;
}

hmx_queue_t hmx_queue_init(शून्य * ptr, size_t capacity, uint32_t राशिः_size, uint32_t hap_rctx, संरचना htp_thread_trace * trace) {
    capacity = hex_ceil_pow2(capacity);
    size_t size_q = hex_align_up(परिमाण(संरचना hmx_queue_s), HEX_L2_पङ्क्तिः_SIZE);
    size_t size_desc = hex_align_up(capacity * परिमाण(संरचना hmx_queue_desc), HEX_L2_पङ्क्तिः_SIZE);

    uint8_t * block = (uint8_t *) ptr;

    hmx_queue_t q = (hmx_queue_t) block; block += size_q;
    memset(q, 0, परिमाण(संरचना hmx_queue_s));

    q->capacity = capacity;
    q->idx_mask = capacity - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (संरचना hmx_queue_desc *) block; block += size_desc;
    memset(q->desc, 0, capacity * परिमाण(संरचना hmx_queue_desc));

    q->राशिः = block;
    memset(q->राशिः, 0, राशिः_size);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    पूर्णाङ्क prio = qurt_thread_get_priority(qurt_thread_get_id());
    जइ (prio < 1) {
        prio = 1;
    }
    जइ (prio > QURT_LOWEST_PRIO) {
        prio = QURT_LOWEST_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_राशिः_addr(&attr, q->राशिः);
    qurt_thread_attr_set_राशिः_size(&attr, राशिः_size);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "hmx-queue");

    पूर्णाङ्क err = qurt_thread_create(&q->thread, &attr, hmx_queue_thread, q);
    जइ (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        प्रत्यागम रिक्त;
    }

    FARF(उच्चः_2, "hmx-queue: capacity %u\n", capacity);

    प्रत्यागम q;
}

शून्य hmx_queue_free(hmx_queue_t q) {
    जइ (!q) {
        प्रत्यागम;
    }

    // Tell the worker to exit.
    hmx_queue_flush(q);
    hmx_queue_signal(q, HMX_QUEUE_KILL);
    hmx_queue_flush(q);

    पूर्णाङ्क status;
    qurt_thread_join(q->thread, &status);
}
