#رهنمود clang diagnostic ignored "-Wunused-function"

#ضمن <stdbool.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#ضمن <qurt_thread.h>
#ضمن <qurt_futex.h>
#ضمن <qurt_hvx.h>

#ضمن <HAP_compute_res.h>

#ضمن "hmx-queue.h"

#تعریف_کن QURT_پایینترین_PRIO (254)

ساكن مضمن فراغ hmx_قفل(hmx_صف_t q)
{
    إذا (!q->hmx_locked) {
        HAP_compute_res_hmx_قفل(q->hap_rctx);
        q->hmx_locked = true;
    }
}

ساكن مضمن فراغ hmx_unlock(hmx_صف_t q)
{
    إذا (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

ساكن مضمن فراغ hmx_صف_process(hmx_صف_t q, bool* killed) {
    بیعلامت صحيح ir = atomic_بارگیری(&q->idx_read);

    طالما (ir != atomic_بارگیری(&q->idx_write)) {
        بنية hmx_صف_desc *d = &q->desc[ir];
        إذا (!d->انجامشده) {
            FARF(بالا_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->داده);

            uintptr_t sig = (uintptr_t) d->func;
            بدل (sig) {
                حالة HMX_صف_NOOP:    /* noop */;     اكسر;
                حالة HMX_صف_KILL:    *killed = true; اكسر;
                حالة HMX_صف_SUSPEND: hmx_unlock(q);  اكسر;
                حالة HMX_صف_WAKEUP:  hmx_قفل(q);    اكسر;
                افتراضي:
                    hmx_قفل(q);
                    htp_trace_event_شروع(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->داده);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    اكسر;
            }

            atomic_fetch_جمع_کن(&d->انجامشده, 1);
        }

        ir = (ir + 1) & q->idx_ماسک;
        atomic_store(&q->idx_read, ir);
    }
}

ساكن فراغ hmx_صف_thread(فراغ * arg) {
    hmx_صف_t q = (hmx_صف_t) arg;

    FARF(بالا_2, "hmx-queue-thread: started");

    bool killed = false;

    بیعلامت صحيح poll_cnt  = HMX_صف_POLL_شمار;
    بیعلامت صحيح prev_seqn = 0;
    طالما (!killed) {
        بیعلامت صحيح seqn = atomic_بارگیری(&q->seqn);
        إذا (seqn == prev_seqn) {
            // drop HVX context while spinning
            إذا (poll_cnt > 1 && poll_cnt == HMX_صف_POLL_شمار) {
                qurt_hvx_unlock();
            }
            إذا (--poll_cnt) { hex_pause(); استمر; }
            FARF(بالا_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_صف_POLL_شمار;
            استمر;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_صف_POLL_شمار;

        FARF(بالا_2, "hmx-queue-thread: new work");

        hmx_صف_process(q, &killed);
    }

    FARF(بالا_2, "hmx-queue-thread: stopped");
}

size_t hmx_صف_sizeof(size_t ظرفیت, uint32_t پشته_اندازه) {
    ظرفیت = hex_ceil_pow2(ظرفیت);
    size_t اندازه_q = hex_align_به_بالا(حجم(بنية hmx_صف_s), HEX_L2_سطر_اندازه);
    size_t اندازه_desc = hex_align_به_بالا(ظرفیت * حجم(بنية hmx_صف_desc), HEX_L2_سطر_اندازه);
    size_t اندازه_پشته = پشته_اندازه;
    أرجع اندازه_q + اندازه_desc + اندازه_پشته;
}

size_t hmx_صف_alignof(فراغ) {
    أرجع HEX_L2_سطر_اندازه;
}

hmx_صف_t hmx_صف_init(فراغ * ptr, size_t ظرفیت, uint32_t پشته_اندازه, uint32_t hap_rctx, بنية htp_thread_trace * trace) {
    ظرفیت = hex_ceil_pow2(ظرفیت);
    size_t اندازه_q = hex_align_به_بالا(حجم(بنية hmx_صف_s), HEX_L2_سطر_اندازه);
    size_t اندازه_desc = hex_align_به_بالا(ظرفیت * حجم(بنية hmx_صف_desc), HEX_L2_سطر_اندازه);

    uint8_t * بلوک = (uint8_t *) ptr;

    hmx_صف_t q = (hmx_صف_t) بلوک; بلوک += اندازه_q;
    memset(q, 0, حجم(بنية hmx_صف_s));

    q->ظرفیت = ظرفیت;
    q->idx_ماسک = ظرفیت - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (بنية hmx_صف_desc *) بلوک; بلوک += اندازه_desc;
    memset(q->desc, 0, ظرفیت * حجم(بنية hmx_صف_desc));

    q->پشته = بلوک;
    memset(q->پشته, 0, پشته_اندازه);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    صحيح prio = qurt_thread_get_اولویت(qurt_thread_get_id());
    إذا (prio < 1) {
        prio = 1;
    }
    إذا (prio > QURT_پایینترین_PRIO) {
        prio = QURT_پایینترین_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_مجموعه_پشته_addr(&attr, q->پشته);
    qurt_thread_attr_مجموعه_پشته_اندازه(&attr, پشته_اندازه);
    qurt_thread_attr_مجموعه_اولویت(&attr, prio);
    qurt_thread_attr_مجموعه_name(&attr, "hmx-queue");

    صحيح err = qurt_thread_create(&q->thread, &attr, hmx_صف_thread, q);
    إذا (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        أرجع عدم;
    }

    FARF(بالا_2, "hmx-queue: capacity %u\n", ظرفیت);

    أرجع q;
}

فراغ hmx_صف_free(hmx_صف_t q) {
    إذا (!q) {
        أرجع;
    }

    // Tell the worker to exit.
    hmx_صف_flush(q);
    hmx_صف_signal(q, HMX_صف_KILL);
    hmx_صف_flush(q);

    صحيح وضعیت;
    qurt_thread_join(q->thread, &وضعیت);
}
