#لارښوونه clang diagnostic ignored "-Wunused-function"

#ضمن <stdbool.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#ضمن <qurt_thread.h>
#ضمن <qurt_futex.h>
#ضمن <qurt_hvx.h>

#ضمن <HAP_compute_res.h>

#ضمن "hmx-queue.h"

#تعریف_کړه QURT_ټیټ_ترین_PRIO (254)

ساكن مضمن فراغ hmx_قلف(hmx_کتار_t q)
{
    إذا (!q->hmx_locked) {
        HAP_compute_res_hmx_قلف(q->hap_rctx);
        q->hmx_locked = true;
    }
}

ساكن مضمن فراغ hmx_unlock(hmx_کتار_t q)
{
    إذا (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

ساكن مضمن فراغ hmx_کتار_process(hmx_کتار_t q, bool* killed) {
    بینښه صحيح ir = atomic_بارول(&q->idx_read);

    طالما (ir != atomic_بارول(&q->idx_write)) {
        بنية hmx_کتار_desc *d = &q->desc[ir];
        إذا (!d->بشپړ) {
            FARF(لوړ_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->معلومات);

            uintptr_t sig = (uintptr_t) d->func;
            بدل (sig) {
                حالة HMX_کتار_NOOP:    /* noop */;     اكسر;
                حالة HMX_کتار_KILL:    *killed = true; اكسر;
                حالة HMX_کتار_SUSPEND: hmx_unlock(q);  اكسر;
                حالة HMX_کتار_WAKEUP:  hmx_قلف(q);    اكسر;
                افتراضي:
                    hmx_قلف(q);
                    htp_trace_event_پیل(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->معلومات);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    اكسر;
            }

            atomic_fetch_جمع_کړه(&d->بشپړ, 1);
        }

        ir = (ir + 1) & q->idx_ماسک;
        atomic_store(&q->idx_read, ir);
    }
}

ساكن فراغ hmx_کتار_thread(فراغ * arg) {
    hmx_کتار_t q = (hmx_کتار_t) arg;

    FARF(لوړ_2, "hmx-queue-thread: started");

    bool killed = false;

    بینښه صحيح poll_cnt  = HMX_کتار_POLL_شمېر;
    بینښه صحيح prev_seqn = 0;
    طالما (!killed) {
        بینښه صحيح seqn = atomic_بارول(&q->seqn);
        إذا (seqn == prev_seqn) {
            // drop HVX context while spinning
            إذا (poll_cnt > 1 && poll_cnt == HMX_کتار_POLL_شمېر) {
                qurt_hvx_unlock();
            }
            إذا (--poll_cnt) { hex_pause(); استمر; }
            FARF(لوړ_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_کتار_POLL_شمېر;
            استمر;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_کتار_POLL_شمېر;

        FARF(لوړ_2, "hmx-queue-thread: new work");

        hmx_کتار_process(q, &killed);
    }

    FARF(لوړ_2, "hmx-queue-thread: stopped");
}

size_t hmx_کتار_sizeof(size_t ظرفیت, uint32_t ډېری_اندازه) {
    ظرفیت = hex_ceil_pow2(ظرفیت);
    size_t اندازه_q = hex_align_پورته(حجم(بنية hmx_کتار_s), HEX_L2_کرښه_اندازه);
    size_t اندازه_desc = hex_align_پورته(ظرفیت * حجم(بنية hmx_کتار_desc), HEX_L2_کرښه_اندازه);
    size_t اندازه_ډېری = ډېری_اندازه;
    أرجع اندازه_q + اندازه_desc + اندازه_ډېری;
}

size_t hmx_کتار_alignof(فراغ) {
    أرجع HEX_L2_کرښه_اندازه;
}

hmx_کتار_t hmx_کتار_init(فراغ * ptr, size_t ظرفیت, uint32_t ډېری_اندازه, uint32_t hap_rctx, بنية htp_thread_trace * trace) {
    ظرفیت = hex_ceil_pow2(ظرفیت);
    size_t اندازه_q = hex_align_پورته(حجم(بنية hmx_کتار_s), HEX_L2_کرښه_اندازه);
    size_t اندازه_desc = hex_align_پورته(ظرفیت * حجم(بنية hmx_کتار_desc), HEX_L2_کرښه_اندازه);

    uint8_t * بلاک = (uint8_t *) ptr;

    hmx_کتار_t q = (hmx_کتار_t) بلاک; بلاک += اندازه_q;
    memset(q, 0, حجم(بنية hmx_کتار_s));

    q->ظرفیت = ظرفیت;
    q->idx_ماسک = ظرفیت - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (بنية hmx_کتار_desc *) بلاک; بلاک += اندازه_desc;
    memset(q->desc, 0, ظرفیت * حجم(بنية hmx_کتار_desc));

    q->ډېری = بلاک;
    memset(q->ډېری, 0, ډېری_اندازه);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    صحيح prio = qurt_thread_get_لومړیتوب(qurt_thread_get_id());
    إذا (prio < 1) {
        prio = 1;
    }
    إذا (prio > QURT_ټیټ_ترین_PRIO) {
        prio = QURT_ټیټ_ترین_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_سېټ_ډېری_addr(&attr, q->ډېری);
    qurt_thread_attr_سېټ_ډېری_اندازه(&attr, ډېری_اندازه);
    qurt_thread_attr_سېټ_لومړیتوب(&attr, prio);
    qurt_thread_attr_سېټ_name(&attr, "hmx-queue");

    صحيح err = qurt_thread_create(&q->thread, &attr, hmx_کتار_thread, q);
    إذا (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        أرجع عدم;
    }

    FARF(لوړ_2, "hmx-queue: capacity %u\n", ظرفیت);

    أرجع q;
}

فراغ hmx_کتار_free(hmx_کتار_t q) {
    إذا (!q) {
        أرجع;
    }

    // Tell the worker to exit.
    hmx_کتار_flush(q);
    hmx_کتار_signal(q, HMX_کتار_KILL);
    hmx_کتار_flush(q);

    صحيح حالت;
    qurt_thread_join(q->thread, &حالت);
}
