#సూచన clang diagnostic ignored "-Wunused-function"

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ <qurt_thread.h>
#అన్తర్భావ <qurt_futex.h>
#అన్తర్భావ <qurt_hvx.h>

#అన్తర్భావ <HAP_compute_res.h>

#అన్తర్భావ "hmx-queue.h"

#నిర్వచించు QURT_అత్యల్ప_PRIO (254)

స్థిర అంతఃవరుస శూన్య hmx_తాళం(hmx_వరుస_t q)
{
    ఒకవేళ (!q->hmx_locked) {
        HAP_compute_res_hmx_తాళం(q->hap_rctx);
        q->hmx_locked = true;
    }
}

స్థిర అంతఃవరుస శూన్య hmx_unlock(hmx_వరుస_t q)
{
    ఒకవేళ (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

స్థిర అంతఃవరుస శూన్య hmx_వరుస_process(hmx_వరుస_t q, bool* killed) {
    గుర్తులేని పూర్ణాంకం ir = atomic_ఎక్కించు(&q->idx_read);

    అయితే (ir != atomic_ఎక్కించు(&q->idx_write)) {
        నిర్మాణం hmx_వరుస_desc *d = &q->desc[ir];
        ఒకవేళ (!d->పూర్తి) {
            FARF(ఎగువ_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->దత్తాంశం);

            uintptr_t sig = (uintptr_t) d->func;
            ఎంపిక (sig) {
                స్థితి HMX_వరుస_NOOP:    /* noop */;     ఆపు;
                స్థితి HMX_వరుస_KILL:    *killed = true; ఆపు;
                స్థితి HMX_వరుస_SUSPEND: hmx_unlock(q);  ఆపు;
                స్థితి HMX_వరుస_WAKEUP:  hmx_తాళం(q);    ఆపు;
                అప్రమేయ:
                    hmx_తాళం(q);
                    htp_trace_event_ప్రారంభం(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->దత్తాంశం);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    ఆపు;
            }

            atomic_fetch_కూడు(&d->పూర్తి, 1);
        }

        ir = (ir + 1) & q->idx_ముసుగు;
        atomic_store(&q->idx_read, ir);
    }
}

స్థిర శూన్య hmx_వరుస_thread(శూన్య * arg) {
    hmx_వరుస_t q = (hmx_వరుస_t) arg;

    FARF(ఎగువ_2, "hmx-queue-thread: started");

    bool killed = false;

    గుర్తులేని పూర్ణాంకం poll_cnt  = HMX_వరుస_POLL_లెక్క;
    గుర్తులేని పూర్ణాంకం prev_seqn = 0;
    అయితే (!killed) {
        గుర్తులేని పూర్ణాంకం seqn = atomic_ఎక్కించు(&q->seqn);
        ఒకవేళ (seqn == prev_seqn) {
            // drop HVX context while spinning
            ఒకవేళ (poll_cnt > 1 && poll_cnt == HMX_వరుస_POLL_లెక్క) {
                qurt_hvx_unlock();
            }
            ఒకవేళ (--poll_cnt) { hex_pause(); కొనసాగు; }
            FARF(ఎగువ_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_వరుస_POLL_లెక్క;
            కొనసాగు;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_వరుస_POLL_లెక్క;

        FARF(ఎగువ_2, "hmx-queue-thread: new work");

        hmx_వరుస_process(q, &killed);
    }

    FARF(ఎగువ_2, "hmx-queue-thread: stopped");
}

size_t hmx_వరుస_sizeof(size_t సామర్థ్యం, uint32_t దొంతర_కొలత) {
    సామర్థ్యం = hex_ceil_pow2(సామర్థ్యం);
    size_t కొలత_q = hex_align_పైకి(పరిమాణం(నిర్మాణం hmx_వరుస_s), HEX_L2_పంక్తి_కొలత);
    size_t కొలత_desc = hex_align_పైకి(సామర్థ్యం * పరిమాణం(నిర్మాణం hmx_వరుస_desc), HEX_L2_పంక్తి_కొలత);
    size_t కొలత_దొంతర = దొంతర_కొలత;
    తిరిగి కొలత_q + కొలత_desc + కొలత_దొంతర;
}

size_t hmx_వరుస_alignof(శూన్య) {
    తిరిగి HEX_L2_పంక్తి_కొలత;
}

hmx_వరుస_t hmx_వరుస_init(శూన్య * ptr, size_t సామర్థ్యం, uint32_t దొంతర_కొలత, uint32_t hap_rctx, నిర్మాణం htp_thread_trace * trace) {
    సామర్థ్యం = hex_ceil_pow2(సామర్థ్యం);
    size_t కొలత_q = hex_align_పైకి(పరిమాణం(నిర్మాణం hmx_వరుస_s), HEX_L2_పంక్తి_కొలత);
    size_t కొలత_desc = hex_align_పైకి(సామర్థ్యం * పరిమాణం(నిర్మాణం hmx_వరుస_desc), HEX_L2_పంక్తి_కొలత);

    uint8_t * ఖండం = (uint8_t *) ptr;

    hmx_వరుస_t q = (hmx_వరుస_t) ఖండం; ఖండం += కొలత_q;
    memset(q, 0, పరిమాణం(నిర్మాణం hmx_వరుస_s));

    q->సామర్థ్యం = సామర్థ్యం;
    q->idx_ముసుగు = సామర్థ్యం - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (నిర్మాణం hmx_వరుస_desc *) ఖండం; ఖండం += కొలత_desc;
    memset(q->desc, 0, సామర్థ్యం * పరిమాణం(నిర్మాణం hmx_వరుస_desc));

    q->దొంతర = ఖండం;
    memset(q->దొంతర, 0, దొంతర_కొలత);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    పూర్ణాంకం prio = qurt_thread_get_ప్రాధాన్యత(qurt_thread_get_id());
    ఒకవేళ (prio < 1) {
        prio = 1;
    }
    ఒకవేళ (prio > QURT_అత్యల్ప_PRIO) {
        prio = QURT_అత్యల్ప_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_సమితి_దొంతర_addr(&attr, q->దొంతర);
    qurt_thread_attr_సమితి_దొంతర_కొలత(&attr, దొంతర_కొలత);
    qurt_thread_attr_సమితి_ప్రాధాన్యత(&attr, prio);
    qurt_thread_attr_సమితి_name(&attr, "hmx-queue");

    పూర్ణాంకం err = qurt_thread_create(&q->thread, &attr, hmx_వరుస_thread, q);
    ఒకవేళ (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        తిరిగి ఖాళీ;
    }

    FARF(ఎగువ_2, "hmx-queue: capacity %u\n", సామర్థ్యం);

    తిరిగి q;
}

శూన్య hmx_వరుస_free(hmx_వరుస_t q) {
    ఒకవేళ (!q) {
        తిరిగి;
    }

    // Tell the worker to exit.
    hmx_వరుస_flush(q);
    hmx_వరుస_signal(q, HMX_వరుస_KILL);
    hmx_వరుస_flush(q);

    పూర్ణాంకం స్థితి_2;
    qurt_thread_join(q->thread, &స్థితి_2);
}
