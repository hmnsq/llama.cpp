#निर्देशः clang diagnostic ignored "-Wunused-function"

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव <qurt_thread.h>
#अन्तर्भाव <qurt_futex.h>
#अन्तर्भाव <qurt_hvx.h>

#अन्तर्भाव <HAP_compute_res.h>

#अन्तर्भाव "hmx-queue.h"

#परिभाषय QURT_निम्नतमम्_PRIO (254)

स्थिर अन्तःस्थ शून्य hmx_तालकम्(hmx_queue_t q)
{
    यदि (!q->hmx_locked) {
        HAP_compute_res_hmx_तालकम्(q->hap_rctx);
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
    अचिह्नित पूर्णाङ्क ir = atomic_आरोपय(&q->idx_read);

    यावत् (ir != atomic_आरोपय(&q->idx_write)) {
        संरचना hmx_queue_desc *d = &q->desc[ir];
        यदि (!d->समाप्तम्) {
            FARF(उच्चः_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->दत्त);

            uintptr_t sig = (uintptr_t) d->func;
            विकल्प (sig) {
                पक्ष HMX_QUEUE_NOOP:    /* noop */;     भङ्ग;
                पक्ष HMX_QUEUE_KILL:    *killed = true; भङ्ग;
                पक्ष HMX_QUEUE_SUSPEND: hmx_unlock(q);  भङ्ग;
                पक्ष HMX_QUEUE_WAKEUP:  hmx_तालकम्(q);    भङ्ग;
                स्वभाव:
                    hmx_तालकम्(q);
                    htp_trace_event_start(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->दत्त);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    भङ्ग;
            }

            atomic_fetch_सङ्कलय(&d->समाप्तम्, 1);
        }

        ir = (ir + 1) & q->idx_आवरणम्;
        atomic_store(&q->idx_read, ir);
    }
}

स्थिर शून्य hmx_queue_thread(शून्य * arg) {
    hmx_queue_t q = (hmx_queue_t) arg;

    FARF(उच्चः_2, "hmx-queue-thread: started");

    bool killed = false;

    अचिह्नित पूर्णाङ्क poll_cnt  = HMX_QUEUE_POLL_गणनम्;
    अचिह्नित पूर्णाङ्क पूर्व_seqn = 0;
    यावत् (!killed) {
        अचिह्नित पूर्णाङ्क seqn = atomic_आरोपय(&q->seqn);
        यदि (seqn == पूर्व_seqn) {
            // drop HVX context while spinning
            यदि (poll_cnt > 1 && poll_cnt == HMX_QUEUE_POLL_गणनम्) {
                qurt_hvx_unlock();
            }
            यदि (--poll_cnt) { hex_pause(); अनुवर्त; }
            FARF(उच्चः_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, पूर्व_seqn);
            poll_cnt = HMX_QUEUE_POLL_गणनम्;
            अनुवर्त;
        }
        पूर्व_seqn = seqn;
        poll_cnt  = HMX_QUEUE_POLL_गणनम्;

        FARF(उच्चः_2, "hmx-queue-thread: new work");

        hmx_queue_process(q, &killed);
    }

    FARF(उच्चः_2, "hmx-queue-thread: stopped");
}

size_t hmx_queue_sizeof(size_t क्षमता, uint32_t राशिः_आकार) {
    क्षमता = hex_ceil_pow2(क्षमता);
    size_t आकार_q = hex_align_ऊर्ध्वम्(परिमाण(संरचना hmx_queue_s), HEX_L2_पङ्क्तिः_आकार);
    size_t आकार_desc = hex_align_ऊर्ध्वम्(क्षमता * परिमाण(संरचना hmx_queue_desc), HEX_L2_पङ्क्तिः_आकार);
    size_t आकार_राशिः = राशिः_आकार;
    प्रत्यागम आकार_q + आकार_desc + आकार_राशिः;
}

size_t hmx_queue_alignof(शून्य) {
    प्रत्यागम HEX_L2_पङ्क्तिः_आकार;
}

hmx_queue_t hmx_queue_init(शून्य * ptr, size_t क्षमता, uint32_t राशिः_आकार, uint32_t hap_rctx, संरचना htp_thread_trace * trace) {
    क्षमता = hex_ceil_pow2(क्षमता);
    size_t आकार_q = hex_align_ऊर्ध्वम्(परिमाण(संरचना hmx_queue_s), HEX_L2_पङ्क्तिः_आकार);
    size_t आकार_desc = hex_align_ऊर्ध्वम्(क्षमता * परिमाण(संरचना hmx_queue_desc), HEX_L2_पङ्क्तिः_आकार);

    uint8_t * खण्डम् = (uint8_t *) ptr;

    hmx_queue_t q = (hmx_queue_t) खण्डम्; खण्डम् += आकार_q;
    स्मृतिं_पूरय(q, 0, परिमाण(संरचना hmx_queue_s));

    q->क्षमता = क्षमता;
    q->idx_आवरणम् = क्षमता - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (संरचना hmx_queue_desc *) खण्डम्; खण्डम् += आकार_desc;
    स्मृतिं_पूरय(q->desc, 0, क्षमता * परिमाण(संरचना hmx_queue_desc));

    q->राशिः = खण्डम्;
    स्मृतिं_पूरय(q->राशिः, 0, राशिः_आकार);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    पूर्णाङ्क prio = qurt_thread_get_प्राथमिकता(qurt_thread_get_id());
    यदि (prio < 1) {
        prio = 1;
    }
    यदि (prio > QURT_निम्नतमम्_PRIO) {
        prio = QURT_निम्नतमम्_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_समुच्चयः_राशिः_addr(&attr, q->राशिः);
    qurt_thread_attr_समुच्चयः_राशिः_आकार(&attr, राशिः_आकार);
    qurt_thread_attr_समुच्चयः_प्राथमिकता(&attr, prio);
    qurt_thread_attr_समुच्चयः_name(&attr, "hmx-queue");

    पूर्णाङ्क err = qurt_thread_create(&q->thread, &attr, hmx_queue_thread, q);
    यदि (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        प्रत्यागम रिक्त;
    }

    FARF(उच्चः_2, "hmx-queue: capacity %u\n", क्षमता);

    प्रत्यागम q;
}

शून्य hmx_queue_free(hmx_queue_t q) {
    यदि (!q) {
        प्रत्यागम;
    }

    // Tell the worker to exit.
    hmx_queue_flush(q);
    hmx_queue_signal(q, HMX_QUEUE_KILL);
    hmx_queue_flush(q);

    पूर्णाङ्क स्थितिः;
    qurt_thread_join(q->thread, &स्थितिः);
}
