#आशय clang diagnostic ignored "-Wunused-function"

#समावेश <stdbool.h>
#समावेश <stdlib.h>
#समावेश <string.h>

#समावेश <qurt_thread.h>
#समावेश <qurt_futex.h>
#समावेश <qurt_hvx.h>

#समावेश <HAP_compute_res.h>

#समावेश "hmx-queue.h"

#घोषणा QURT_LOWEST_PRIO (254)

अटल अंतरभूत व्योम hmx_lock(hmx_queue_t q)
{
    अगर (!q->hmx_locked) {
        HAP_compute_res_hmx_lock(q->hap_rctx);
        q->hmx_locked = true;
    }
}

अटल अंतरभूत व्योम hmx_unlock(hmx_queue_t q)
{
    अगर (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

अटल अंतरभूत व्योम hmx_queue_process(hmx_queue_t q, bool* killed) {
    अचिन्हित पूर्णांक ir = atomic_load(&q->idx_read);

    जबतक (ir != atomic_load(&q->idx_write)) {
        काष्ठा hmx_queue_desc *d = &q->desc[ir];
        अगर (!d->done) {
            FARF(ऊपरी_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->तथ्य);

            uintptr_t sig = (uintptr_t) d->func;
            चयन (sig) {
                हाल HMX_QUEUE_NOOP:    /* noop */;     अवरोध;
                हाल HMX_QUEUE_KILL:    *killed = true; अवरोध;
                हाल HMX_QUEUE_SUSPEND: hmx_unlock(q);  अवरोध;
                हाल HMX_QUEUE_WAKEUP:  hmx_lock(q);    अवरोध;
                शेष:
                    hmx_lock(q);
                    htp_trace_event_start(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->तथ्य);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    अवरोध;
            }

            atomic_fetch_add(&d->done, 1);
        }

        ir = (ir + 1) & q->idx_mask;
        atomic_store(&q->idx_read, ir);
    }
}

अटल व्योम hmx_queue_thread(व्योम * arg) {
    hmx_queue_t q = (hmx_queue_t) arg;

    FARF(ऊपरी_2, "hmx-queue-thread: started");

    bool killed = false;

    अचिन्हित पूर्णांक poll_cnt  = HMX_QUEUE_POLL_गिनती;
    अचिन्हित पूर्णांक पूर्व_seqn = 0;
    जबतक (!killed) {
        अचिन्हित पूर्णांक seqn = atomic_load(&q->seqn);
        अगर (seqn == पूर्व_seqn) {
            // drop HVX context while spinning
            अगर (poll_cnt > 1 && poll_cnt == HMX_QUEUE_POLL_गिनती) {
                qurt_hvx_unlock();
            }
            अगर (--poll_cnt) { hex_pause(); जारी; }
            FARF(ऊपरी_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, पूर्व_seqn);
            poll_cnt = HMX_QUEUE_POLL_गिनती;
            जारी;
        }
        पूर्व_seqn = seqn;
        poll_cnt  = HMX_QUEUE_POLL_गिनती;

        FARF(ऊपरी_2, "hmx-queue-thread: new work");

        hmx_queue_process(q, &killed);
    }

    FARF(ऊपरी_2, "hmx-queue-thread: stopped");
}

माप_प्रकार hmx_queue_sizeof(माप_प्रकार capacity, uint32_t ढेर_आकार) {
    capacity = hex_ceil_pow2(capacity);
    माप_प्रकार आकार_q = hex_align_up(माप(काष्ठा hmx_queue_s), HEX_L2_पंक्ति_आकार);
    माप_प्रकार आकार_desc = hex_align_up(capacity * माप(काष्ठा hmx_queue_desc), HEX_L2_पंक्ति_आकार);
    माप_प्रकार आकार_ढेर = ढेर_आकार;
    वापस आकार_q + आकार_desc + आकार_ढेर;
}

माप_प्रकार hmx_queue_alignof(व्योम) {
    वापस HEX_L2_पंक्ति_आकार;
}

hmx_queue_t hmx_queue_init(व्योम * ptr, माप_प्रकार capacity, uint32_t ढेर_आकार, uint32_t hap_rctx, काष्ठा htp_thread_trace * trace) {
    capacity = hex_ceil_pow2(capacity);
    माप_प्रकार आकार_q = hex_align_up(माप(काष्ठा hmx_queue_s), HEX_L2_पंक्ति_आकार);
    माप_प्रकार आकार_desc = hex_align_up(capacity * माप(काष्ठा hmx_queue_desc), HEX_L2_पंक्ति_आकार);

    uint8_t * block = (uint8_t *) ptr;

    hmx_queue_t q = (hmx_queue_t) block; block += आकार_q;
    स_रखो(q, 0, माप(काष्ठा hmx_queue_s));

    q->capacity = capacity;
    q->idx_mask = capacity - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (काष्ठा hmx_queue_desc *) block; block += आकार_desc;
    स_रखो(q->desc, 0, capacity * माप(काष्ठा hmx_queue_desc));

    q->ढेर_2 = block;
    स_रखो(q->ढेर_2, 0, ढेर_आकार);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    पूर्णांक prio = qurt_thread_get_priority(qurt_thread_get_id());
    अगर (prio < 1) {
        prio = 1;
    }
    अगर (prio > QURT_LOWEST_PRIO) {
        prio = QURT_LOWEST_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_ढेर_addr(&attr, q->ढेर_2);
    qurt_thread_attr_set_ढेर_आकार(&attr, ढेर_आकार);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "hmx-queue");

    पूर्णांक err = qurt_thread_create(&q->thread, &attr, hmx_queue_thread, q);
    अगर (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        वापस शून्य;
    }

    FARF(ऊपरी_2, "hmx-queue: capacity %u\n", capacity);

    वापस q;
}

व्योम hmx_queue_free(hmx_queue_t q) {
    अगर (!q) {
        वापस;
    }

    // Tell the worker to exit.
    hmx_queue_flush(q);
    hmx_queue_signal(q, HMX_QUEUE_KILL);
    hmx_queue_flush(q);

    पूर्णांक status;
    qurt_thread_join(q->thread, &status);
}
