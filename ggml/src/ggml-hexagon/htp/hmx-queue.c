#pragma clang diagnostic ignored "-Wunused-function"

#شامل <stdbool.h>
#شامل <stdlib.h>
#شامل <string.h>

#شامل <qurt_thread.h>
#شامل <qurt_futex.h>
#شامل <qurt_hvx.h>

#شامل <HAP_compute_res.h>

#شامل "hmx-queue.h"

#define QURT_LOWEST_PRIO (254)

ایستا درونخطی تهی hmx_lock(hmx_queue_t q)
{
    اگر (!q->hmx_locked) {
        HAP_compute_res_hmx_lock(q->hap_rctx);
        q->hmx_locked = true;
    }
}

ایستا درونخطی تهی hmx_unlock(hmx_queue_t q)
{
    اگر (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

ایستا درونخطی تهی hmx_queue_process(hmx_queue_t q, bool* killed) {
    unsigned صحیح ir = atomic_load(&q->idx_read);

    وقتی (ir != atomic_load(&q->idx_write)) {
        ساختار hmx_queue_desc *d = &q->desc[ir];
        اگر (!d->done) {
            FARF(بالا_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->data);

            uintptr_t sig = (uintptr_t) d->func;
            انتخاب (sig) {
                حالت HMX_QUEUE_NOOP:    /* noop */;     بشکن;
                حالت HMX_QUEUE_KILL:    *killed = true; بشکن;
                حالت HMX_QUEUE_SUSPEND: hmx_unlock(q);  بشکن;
                حالت HMX_QUEUE_WAKEUP:  hmx_lock(q);    بشکن;
                پیشفرض:
                    hmx_lock(q);
                    htp_trace_event_start(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->data);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    بشکن;
            }

            atomic_fetch_add(&d->done, 1);
        }

        ir = (ir + 1) & q->idx_mask;
        atomic_store(&q->idx_read, ir);
    }
}

ایستا تهی hmx_queue_thread(تهی * arg) {
    hmx_queue_t q = (hmx_queue_t) arg;

    FARF(بالا_2, "hmx-queue-thread: started");

    bool killed = false;

    unsigned صحیح poll_cnt  = HMX_QUEUE_POLL_شمار;
    unsigned صحیح prev_seqn = 0;
    وقتی (!killed) {
        unsigned صحیح seqn = atomic_load(&q->seqn);
        اگر (seqn == prev_seqn) {
            // drop HVX context while spinning
            اگر (poll_cnt > 1 && poll_cnt == HMX_QUEUE_POLL_شمار) {
                qurt_hvx_unlock();
            }
            اگر (--poll_cnt) { hex_pause(); ادامه; }
            FARF(بالا_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_QUEUE_POLL_شمار;
            ادامه;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_QUEUE_POLL_شمار;

        FARF(بالا_2, "hmx-queue-thread: new work");

        hmx_queue_process(q, &killed);
    }

    FARF(بالا_2, "hmx-queue-thread: stopped");
}

size_t hmx_queue_sizeof(size_t capacity, uint32_t پشته_size) {
    capacity = hex_ceil_pow2(capacity);
    size_t size_q = hex_align_up(اندازه(ساختار hmx_queue_s), HEX_L2_سطر_SIZE);
    size_t size_desc = hex_align_up(capacity * اندازه(ساختار hmx_queue_desc), HEX_L2_سطر_SIZE);
    size_t size_پشته = پشته_size;
    بازگشت size_q + size_desc + size_پشته;
}

size_t hmx_queue_alignof(تهی) {
    بازگشت HEX_L2_سطر_SIZE;
}

hmx_queue_t hmx_queue_init(تهی * ptr, size_t capacity, uint32_t پشته_size, uint32_t hap_rctx, ساختار htp_thread_trace * trace) {
    capacity = hex_ceil_pow2(capacity);
    size_t size_q = hex_align_up(اندازه(ساختار hmx_queue_s), HEX_L2_سطر_SIZE);
    size_t size_desc = hex_align_up(capacity * اندازه(ساختار hmx_queue_desc), HEX_L2_سطر_SIZE);

    uint8_t * block = (uint8_t *) ptr;

    hmx_queue_t q = (hmx_queue_t) block; block += size_q;
    memset(q, 0, اندازه(ساختار hmx_queue_s));

    q->capacity = capacity;
    q->idx_mask = capacity - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (ساختار hmx_queue_desc *) block; block += size_desc;
    memset(q->desc, 0, capacity * اندازه(ساختار hmx_queue_desc));

    q->پشته = block;
    memset(q->پشته, 0, پشته_size);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    صحیح prio = qurt_thread_get_priority(qurt_thread_get_id());
    اگر (prio < 1) {
        prio = 1;
    }
    اگر (prio > QURT_LOWEST_PRIO) {
        prio = QURT_LOWEST_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_پشته_addr(&attr, q->پشته);
    qurt_thread_attr_set_پشته_size(&attr, پشته_size);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "hmx-queue");

    صحیح err = qurt_thread_create(&q->thread, &attr, hmx_queue_thread, q);
    اگر (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        بازگشت پوچ;
    }

    FARF(بالا_2, "hmx-queue: capacity %u\n", capacity);

    بازگشت q;
}

تهی hmx_queue_free(hmx_queue_t q) {
    اگر (!q) {
        بازگشت;
    }

    // Tell the worker to exit.
    hmx_queue_flush(q);
    hmx_queue_signal(q, HMX_QUEUE_KILL);
    hmx_queue_flush(q);

    صحیح status;
    qurt_thread_join(q->thread, &status);
}
