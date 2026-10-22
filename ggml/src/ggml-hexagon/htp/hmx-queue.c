#هدايت clang diagnostic ignored "-Wunused-function"

#ضمن <stdbool.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#ضمن <qurt_thread.h>
#ضمن <qurt_futex.h>
#ضمن <qurt_hvx.h>

#ضمن <HAP_compute_res.h>

#ضمن "hmx-queue.h"

#وصف_ڪريو QURT_سڀ_کان_هيٺ_PRIO (254)

ساكن مضمن فراغ hmx_تالو(hmx_قطار_t q)
{
    إذا (!q->hmx_locked) {
        HAP_compute_res_hmx_تالو(q->hap_rctx);
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
    بنانشان صحيح ir = atomic_ڀريو(&q->idx_read);

    طالما (ir != atomic_ڀريو(&q->idx_write)) {
        بنية hmx_قطار_desc *d = &q->desc[ir];
        إذا (!d->پورو) {
            FARF(مٿاهون_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->معلومات);

            uintptr_t sig = (uintptr_t) d->func;
            بدل (sig) {
                حالة HMX_قطار_NOOP:    /* noop */;     اكسر;
                حالة HMX_قطار_KILL:    *killed = true; اكسر;
                حالة HMX_قطار_SUSPEND: hmx_unlock(q);  اكسر;
                حالة HMX_قطار_WAKEUP:  hmx_تالو(q);    اكسر;
                افتراضي:
                    hmx_تالو(q);
                    htp_trace_event_شروعات(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->معلومات);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    اكسر;
            }

            atomic_fetch_جوڙيو(&d->پورو, 1);
        }

        ir = (ir + 1) & q->idx_نقاب;
        atomic_store(&q->idx_read, ir);
    }
}

ساكن فراغ hmx_قطار_thread(فراغ * arg) {
    hmx_قطار_t q = (hmx_قطار_t) arg;

    FARF(مٿاهون_2, "hmx-queue-thread: started");

    bool killed = false;

    بنانشان صحيح poll_cnt  = HMX_قطار_POLL_ڳڻپ;
    بنانشان صحيح prev_seqn = 0;
    طالما (!killed) {
        بنانشان صحيح seqn = atomic_ڀريو(&q->seqn);
        إذا (seqn == prev_seqn) {
            // drop HVX context while spinning
            إذا (poll_cnt > 1 && poll_cnt == HMX_قطار_POLL_ڳڻپ) {
                qurt_hvx_unlock();
            }
            إذا (--poll_cnt) { hex_pause(); استمر; }
            FARF(مٿاهون_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_قطار_POLL_ڳڻپ;
            استمر;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_قطار_POLL_ڳڻپ;

        FARF(مٿاهون_2, "hmx-queue-thread: new work");

        hmx_قطار_process(q, &killed);
    }

    FARF(مٿاهون_2, "hmx-queue-thread: stopped");
}

size_t hmx_قطار_sizeof(size_t صلاحيت, uint32_t ڍير_سائيز) {
    صلاحيت = hex_ceil_pow2(صلاحيت);
    size_t سائيز_q = hex_align_مٿي(حجم(بنية hmx_قطار_s), HEX_L2_سٽ_سائيز);
    size_t سائيز_desc = hex_align_مٿي(صلاحيت * حجم(بنية hmx_قطار_desc), HEX_L2_سٽ_سائيز);
    size_t سائيز_ڍير = ڍير_سائيز;
    أرجع سائيز_q + سائيز_desc + سائيز_ڍير;
}

size_t hmx_قطار_alignof(فراغ) {
    أرجع HEX_L2_سٽ_سائيز;
}

hmx_قطار_t hmx_قطار_init(فراغ * ptr, size_t صلاحيت, uint32_t ڍير_سائيز, uint32_t hap_rctx, بنية htp_thread_trace * trace) {
    صلاحيت = hex_ceil_pow2(صلاحيت);
    size_t سائيز_q = hex_align_مٿي(حجم(بنية hmx_قطار_s), HEX_L2_سٽ_سائيز);
    size_t سائيز_desc = hex_align_مٿي(صلاحيت * حجم(بنية hmx_قطار_desc), HEX_L2_سٽ_سائيز);

    uint8_t * بلاڪ = (uint8_t *) ptr;

    hmx_قطار_t q = (hmx_قطار_t) بلاڪ; بلاڪ += سائيز_q;
    memset(q, 0, حجم(بنية hmx_قطار_s));

    q->صلاحيت = صلاحيت;
    q->idx_نقاب = صلاحيت - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (بنية hmx_قطار_desc *) بلاڪ; بلاڪ += سائيز_desc;
    memset(q->desc, 0, صلاحيت * حجم(بنية hmx_قطار_desc));

    q->ڍير = بلاڪ;
    memset(q->ڍير, 0, ڍير_سائيز);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    صحيح prio = qurt_thread_get_ترجيح(qurt_thread_get_id());
    إذا (prio < 1) {
        prio = 1;
    }
    إذا (prio > QURT_سڀ_کان_هيٺ_PRIO) {
        prio = QURT_سڀ_کان_هيٺ_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_سيٽ_ڍير_addr(&attr, q->ڍير);
    qurt_thread_attr_سيٽ_ڍير_سائيز(&attr, ڍير_سائيز);
    qurt_thread_attr_سيٽ_ترجيح(&attr, prio);
    qurt_thread_attr_سيٽ_name(&attr, "hmx-queue");

    صحيح err = qurt_thread_create(&q->thread, &attr, hmx_قطار_thread, q);
    إذا (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        أرجع عدم;
    }

    FARF(مٿاهون_2, "hmx-queue: capacity %u\n", صلاحيت);

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
