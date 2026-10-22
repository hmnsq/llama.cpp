#ہدایت clang diagnostic ignored "-Wunused-function"

#ضمن <stdbool.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#ضمن <qurt_thread.h>
#ضمن <qurt_futex.h>
#ضمن <qurt_hvx.h>

#ضمن <HAP_compute_res.h>

#ضمن "hmx-queue.h"

#تعریف_کرو QURT_سبھ_توں_نیواں_PRIO (254)

ساكن مضمن فراغ hmx_جندرا(hmx_قطار_t q)
{
    إذا (!q->hmx_locked) {
        HAP_compute_res_hmx_جندرا(q->hap_rctx);
        q->hmx_locked = true;
    }
}

ساكن مضمن فراغ hmx_unlock(hmx_قطار_t q)
{
    إذا (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

ساكن مضمن فراغ hmx_قطار_process(hmx_قطار_t q, bool* killed) {
    بنانشانی صحيح ir = atomic_لدو(&q->idx_read);

    طالما (ir != atomic_لدو(&q->idx_write)) {
        بنية hmx_قطار_desc *d = &q->desc[ir];
        إذا (!d->پورا) {
            FARF(بلند_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->انکڑے);

            uintptr_t sig = (uintptr_t) d->func;
            بدل (sig) {
                حالة HMX_قطار_NOOP:    /* noop */;     اكسر;
                حالة HMX_قطار_KILL:    *killed = true; اكسر;
                حالة HMX_قطار_SUSPEND: hmx_unlock(q);  اكسر;
                حالة HMX_قطار_WAKEUP:  hmx_جندرا(q);    اكسر;
                افتراضي:
                    hmx_جندرا(q);
                    htp_trace_event_شروع(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->انکڑے);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    اكسر;
            }

            atomic_fetch_جوڑو(&d->پورا, 1);
        }

        ir = (ir + 1) & q->idx_نقاب;
        atomic_store(&q->idx_read, ir);
    }
}

ساكن فراغ hmx_قطار_thread(فراغ * arg) {
    hmx_قطار_t q = (hmx_قطار_t) arg;

    FARF(بلند_2, "hmx-queue-thread: started");

    bool killed = false;

    بنانشانی صحيح poll_cnt  = HMX_قطار_POLL_گنتی;
    بنانشانی صحيح prev_seqn = 0;
    طالما (!killed) {
        بنانشانی صحيح seqn = atomic_لدو(&q->seqn);
        إذا (seqn == prev_seqn) {
            // drop HVX context while spinning
            إذا (poll_cnt > 1 && poll_cnt == HMX_قطار_POLL_گنتی) {
                qurt_hvx_unlock();
            }
            إذا (--poll_cnt) { hex_pause(); استمر; }
            FARF(بلند_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_قطار_POLL_گنتی;
            استمر;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_قطار_POLL_گنتی;

        FARF(بلند_2, "hmx-queue-thread: new work");

        hmx_قطار_process(q, &killed);
    }

    FARF(بلند_2, "hmx-queue-thread: stopped");
}

size_t hmx_قطار_sizeof(size_t سمرتھا, uint32_t ڈھیر_ناپ) {
    سمرتھا = hex_ceil_pow2(سمرتھا);
    size_t ناپ_q = hex_align_اتے(حجم(بنية hmx_قطار_s), HEX_L2_سطر_ناپ);
    size_t ناپ_desc = hex_align_اتے(سمرتھا * حجم(بنية hmx_قطار_desc), HEX_L2_سطر_ناپ);
    size_t ناپ_ڈھیر = ڈھیر_ناپ;
    أرجع ناپ_q + ناپ_desc + ناپ_ڈھیر;
}

size_t hmx_قطار_alignof(فراغ) {
    أرجع HEX_L2_سطر_ناپ;
}

hmx_قطار_t hmx_قطار_init(فراغ * ptr, size_t سمرتھا, uint32_t ڈھیر_ناپ, uint32_t hap_rctx, بنية htp_thread_trace * trace) {
    سمرتھا = hex_ceil_pow2(سمرتھا);
    size_t ناپ_q = hex_align_اتے(حجم(بنية hmx_قطار_s), HEX_L2_سطر_ناپ);
    size_t ناپ_desc = hex_align_اتے(سمرتھا * حجم(بنية hmx_قطار_desc), HEX_L2_سطر_ناپ);

    uint8_t * بلاک = (uint8_t *) ptr;

    hmx_قطار_t q = (hmx_قطار_t) بلاک; بلاک += ناپ_q;
    memset(q, 0, حجم(بنية hmx_قطار_s));

    q->سمرتھا = سمرتھا;
    q->idx_نقاب = سمرتھا - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (بنية hmx_قطار_desc *) بلاک; بلاک += ناپ_desc;
    memset(q->desc, 0, سمرتھا * حجم(بنية hmx_قطار_desc));

    q->ڈھیر = بلاک;
    memset(q->ڈھیر, 0, ڈھیر_ناپ);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    صحيح prio = qurt_thread_get_ترجیح(qurt_thread_get_id());
    إذا (prio < 1) {
        prio = 1;
    }
    إذا (prio > QURT_سبھ_توں_نیواں_PRIO) {
        prio = QURT_سبھ_توں_نیواں_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_سیٹ_ڈھیر_addr(&attr, q->ڈھیر);
    qurt_thread_attr_سیٹ_ڈھیر_ناپ(&attr, ڈھیر_ناپ);
    qurt_thread_attr_سیٹ_ترجیح(&attr, prio);
    qurt_thread_attr_سیٹ_name(&attr, "hmx-queue");

    صحيح err = qurt_thread_create(&q->thread, &attr, hmx_قطار_thread, q);
    إذا (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        أرجع عدم;
    }

    FARF(بلند_2, "hmx-queue: capacity %u\n", سمرتھا);

    أرجع q;
}

فراغ hmx_قطار_free(hmx_قطار_t q) {
    إذا (!q) {
        أرجع;
    }

    // Tell the worker to exit.
    hmx_قطار_flush(q);
    hmx_قطار_signal(q, HMX_قطار_KILL);
    hmx_قطار_flush(q);

    صحيح حالت;
    qurt_thread_join(q->thread, &حالت);
}
