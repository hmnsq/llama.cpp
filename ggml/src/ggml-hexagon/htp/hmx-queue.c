#ہدایت clang diagnostic ignored "-Wunused-function"

#شامل <stdbool.h>
#شامل <stdlib.h>
#شامل <string.h>

#شامل <qurt_thread.h>
#شامل <qurt_futex.h>
#شامل <qurt_hvx.h>

#شامل <HAP_compute_res.h>

#شامل "hmx-queue.h"

#تعریف_کرو QURT_سب_سے_نیچا_PRIO (254)

ساکن اندرونی خالی hmx_تالا(hmx_قطار_t q)
{
    اگر (!q->hmx_locked) {
        HAP_compute_res_hmx_تالا(q->hap_rctx);
        q->hmx_locked = true;
    }
}

ساکن اندرونی خالی hmx_unlock(hmx_قطار_t q)
{
    اگر (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

ساکن اندرونی خالی hmx_قطار_process(hmx_قطار_t q, bool* killed) {
    غیرعلامتی صحیح ir = atomic_بھرو(&q->idx_read);

    جب (ir != atomic_بھرو(&q->idx_write)) {
        ساخت hmx_قطار_desc *d = &q->desc[ir];
        اگر (!d->مکمل) {
            FARF(بلند_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->ڈیٹا);

            uintptr_t sig = (uintptr_t) d->func;
            انتخاب (sig) {
                حالت HMX_قطار_NOOP:    /* noop */;     توڑو;
                حالت HMX_قطار_KILL:    *killed = true; توڑو;
                حالت HMX_قطار_SUSPEND: hmx_unlock(q);  توڑو;
                حالت HMX_قطار_WAKEUP:  hmx_تالا(q);    توڑو;
                طےشدہ:
                    hmx_تالا(q);
                    htp_trace_event_آغاز(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->ڈیٹا);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    توڑو;
            }

            atomic_fetch_جوڑو(&d->مکمل, 1);
        }

        ir = (ir + 1) & q->idx_نقاب;
        atomic_store(&q->idx_read, ir);
    }
}

ساکن خالی hmx_قطار_thread(خالی * arg) {
    hmx_قطار_t q = (hmx_قطار_t) arg;

    FARF(بلند_2, "hmx-queue-thread: started");

    bool killed = false;

    غیرعلامتی صحیح poll_cnt  = HMX_قطار_POLL_گنتی;
    غیرعلامتی صحیح پچھلا_seqn = 0;
    جب (!killed) {
        غیرعلامتی صحیح seqn = atomic_بھرو(&q->seqn);
        اگر (seqn == پچھلا_seqn) {
            // drop HVX context while spinning
            اگر (poll_cnt > 1 && poll_cnt == HMX_قطار_POLL_گنتی) {
                qurt_hvx_unlock();
            }
            اگر (--poll_cnt) { hex_pause(); جاری; }
            FARF(بلند_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, پچھلا_seqn);
            poll_cnt = HMX_قطار_POLL_گنتی;
            جاری;
        }
        پچھلا_seqn = seqn;
        poll_cnt  = HMX_قطار_POLL_گنتی;

        FARF(بلند_2, "hmx-queue-thread: new work");

        hmx_قطار_process(q, &killed);
    }

    FARF(بلند_2, "hmx-queue-thread: stopped");
}

size_t hmx_قطار_sizeof(size_t استعداد, uint32_t ڈھیر_سائز) {
    استعداد = hex_ceil_pow2(استعداد);
    size_t سائز_q = hex_align_اوپر(حجم(ساخت hmx_قطار_s), HEX_L2_سطر_سائز);
    size_t سائز_desc = hex_align_اوپر(استعداد * حجم(ساخت hmx_قطار_desc), HEX_L2_سطر_سائز);
    size_t سائز_ڈھیر = ڈھیر_سائز;
    واپس سائز_q + سائز_desc + سائز_ڈھیر;
}

size_t hmx_قطار_alignof(خالی) {
    واپس HEX_L2_سطر_سائز;
}

hmx_قطار_t hmx_قطار_init(خالی * ptr, size_t استعداد, uint32_t ڈھیر_سائز, uint32_t hap_rctx, ساخت htp_thread_trace * trace) {
    استعداد = hex_ceil_pow2(استعداد);
    size_t سائز_q = hex_align_اوپر(حجم(ساخت hmx_قطار_s), HEX_L2_سطر_سائز);
    size_t سائز_desc = hex_align_اوپر(استعداد * حجم(ساخت hmx_قطار_desc), HEX_L2_سطر_سائز);

    uint8_t * بلاک = (uint8_t *) ptr;

    hmx_قطار_t q = (hmx_قطار_t) بلاک; بلاک += سائز_q;
    memset(q, 0, حجم(ساخت hmx_قطار_s));

    q->استعداد = استعداد;
    q->idx_نقاب = استعداد - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (ساخت hmx_قطار_desc *) بلاک; بلاک += سائز_desc;
    memset(q->desc, 0, استعداد * حجم(ساخت hmx_قطار_desc));

    q->ڈھیر = بلاک;
    memset(q->ڈھیر, 0, ڈھیر_سائز);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    صحیح prio = qurt_thread_get_ترجیح(qurt_thread_get_id());
    اگر (prio < 1) {
        prio = 1;
    }
    اگر (prio > QURT_سب_سے_نیچا_PRIO) {
        prio = QURT_سب_سے_نیچا_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_سیٹ_ڈھیر_addr(&attr, q->ڈھیر);
    qurt_thread_attr_سیٹ_ڈھیر_سائز(&attr, ڈھیر_سائز);
    qurt_thread_attr_سیٹ_ترجیح(&attr, prio);
    qurt_thread_attr_سیٹ_name(&attr, "hmx-queue");

    صحیح err = qurt_thread_create(&q->thread, &attr, hmx_قطار_thread, q);
    اگر (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        واپس تہی;
    }

    FARF(بلند_2, "hmx-queue: capacity %u\n", استعداد);

    واپس q;
}

خالی hmx_قطار_free(hmx_قطار_t q) {
    اگر (!q) {
        واپس;
    }

    // Tell the worker to exit.
    hmx_قطار_flush(q);
    hmx_قطار_signal(q, HMX_قطار_KILL);
    hmx_قطار_flush(q);

    صحیح حالت_2;
    qurt_thread_join(q->thread, &حالت_2);
}
