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

स्थिर ओळीत शून्य hmx_lock(hmx_queue_t q)
{
    जर (!q->hmx_locked) {
        HAP_compute_res_hmx_lock(q->hap_rctx);
        q->hmx_locked = true;
    }
}

स्थिर ओळीत शून्य hmx_unlock(hmx_queue_t q)
{
    जर (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

स्थिर ओळीत शून्य hmx_queue_process(hmx_queue_t q, bool* killed) {
    अचिन्हित पूर्णांक ir = atomic_load(&q->idx_read);

    पर्यंत (ir != atomic_load(&q->idx_write)) {
        संरचना hmx_queue_desc *d = &q->desc[ir];
        जर (!d->done) {
            FARF(वरचा_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->data);

            uintptr_t sig = (uintptr_t) d->func;
            निवड (sig) {
                स्थिती HMX_QUEUE_NOOP:    /* noop */;     थांबा;
                स्थिती HMX_QUEUE_KILL:    *killed = true; थांबा;
                स्थिती HMX_QUEUE_SUSPEND: hmx_unlock(q);  थांबा;
                स्थिती HMX_QUEUE_WAKEUP:  hmx_lock(q);    थांबा;
                पूर्वनिर्धारित:
                    hmx_lock(q);
                    htp_trace_event_start(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->data);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    थांबा;
            }

            atomic_fetch_add(&d->done, 1);
        }

        ir = (ir + 1) & q->idx_mask;
        atomic_store(&q->idx_read, ir);
    }
}

स्थिर शून्य hmx_queue_thread(शून्य * arg) {
    hmx_queue_t q = (hmx_queue_t) arg;

    FARF(वरचा_2, "hmx-queue-thread: started");

    bool killed = false;

    अचिन्हित पूर्णांक poll_cnt  = HMX_QUEUE_POLL_मोजणी;
    अचिन्हित पूर्णांक prev_seqn = 0;
    पर्यंत (!killed) {
        अचिन्हित पूर्णांक seqn = atomic_load(&q->seqn);
        जर (seqn == prev_seqn) {
            // drop HVX context while spinning
            जर (poll_cnt > 1 && poll_cnt == HMX_QUEUE_POLL_मोजणी) {
                qurt_hvx_unlock();
            }
            जर (--poll_cnt) { hex_pause(); सुरूठेवा; }
            FARF(वरचा_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_QUEUE_POLL_मोजणी;
            सुरूठेवा;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_QUEUE_POLL_मोजणी;

        FARF(वरचा_2, "hmx-queue-thread: new work");

        hmx_queue_process(q, &killed);
    }

    FARF(वरचा_2, "hmx-queue-thread: stopped");
}

माप_प्रकार hmx_queue_sizeof(माप_प्रकार capacity, uint32_t चळत_size) {
    capacity = hex_ceil_pow2(capacity);
    माप_प्रकार size_q = hex_align_up(आकार(संरचना hmx_queue_s), HEX_L2_ओळ_SIZE);
    माप_प्रकार size_desc = hex_align_up(capacity * आकार(संरचना hmx_queue_desc), HEX_L2_ओळ_SIZE);
    माप_प्रकार size_चळत = चळत_size;
    परत size_q + size_desc + size_चळत;
}

माप_प्रकार hmx_queue_alignof(शून्य) {
    परत HEX_L2_ओळ_SIZE;
}

hmx_queue_t hmx_queue_init(शून्य * ptr, माप_प्रकार capacity, uint32_t चळत_size, uint32_t hap_rctx, संरचना htp_thread_trace * trace) {
    capacity = hex_ceil_pow2(capacity);
    माप_प्रकार size_q = hex_align_up(आकार(संरचना hmx_queue_s), HEX_L2_ओळ_SIZE);
    माप_प्रकार size_desc = hex_align_up(capacity * आकार(संरचना hmx_queue_desc), HEX_L2_ओळ_SIZE);

    uint8_t * block = (uint8_t *) ptr;

    hmx_queue_t q = (hmx_queue_t) block; block += size_q;
    स_रखो(q, 0, आकार(संरचना hmx_queue_s));

    q->capacity = capacity;
    q->idx_mask = capacity - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (संरचना hmx_queue_desc *) block; block += size_desc;
    स_रखो(q->desc, 0, capacity * आकार(संरचना hmx_queue_desc));

    q->चळत = block;
    स_रखो(q->चळत, 0, चळत_size);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    पूर्णांक prio = qurt_thread_get_priority(qurt_thread_get_id());
    जर (prio < 1) {
        prio = 1;
    }
    जर (prio > QURT_LOWEST_PRIO) {
        prio = QURT_LOWEST_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_चळत_addr(&attr, q->चळत);
    qurt_thread_attr_set_चळत_size(&attr, चळत_size);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "hmx-queue");

    पूर्णांक err = qurt_thread_create(&q->thread, &attr, hmx_queue_thread, q);
    जर (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        परत रिक्त;
    }

    FARF(वरचा_2, "hmx-queue: capacity %u\n", capacity);

    परत q;
}

शून्य hmx_queue_free(hmx_queue_t q) {
    जर (!q) {
        परत;
    }

    // Tell the worker to exit.
    hmx_queue_flush(q);
    hmx_queue_signal(q, HMX_QUEUE_KILL);
    hmx_queue_flush(q);

    पूर्णांक status;
    qurt_thread_join(q->thread, &status);
}
