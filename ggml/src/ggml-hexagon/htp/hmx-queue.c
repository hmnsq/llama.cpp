#pragma clang diagnostic ignored "-Wunused-function"

#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <string.h>

#அந்தர்பாவ <qurt_thread.h>
#அந்தர்பாவ <qurt_futex.h>
#அந்தர்பாவ <qurt_hvx.h>

#அந்தர்பாவ <HAP_compute_res.h>

#அந்தர்பாவ "hmx-queue.h"

#define QURT_LOWEST_PRIO (254)

நிலையான உள்வரி வெறுமை hmx_lock(hmx_queue_t q)
{
    எனில் (!q->hmx_locked) {
        HAP_compute_res_hmx_lock(q->hap_rctx);
        q->hmx_locked = true;
    }
}

நிலையான உள்வரி வெறுமை hmx_unlock(hmx_queue_t q)
{
    எனில் (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

நிலையான உள்வரி வெறுமை hmx_queue_process(hmx_queue_t q, bool* killed) {
    unsigned முழு ir = atomic_load(&q->idx_read);

    போது (ir != atomic_load(&q->idx_write)) {
        கட்டமைப்பு hmx_queue_desc *d = &q->desc[ir];
        எனில் (!d->done) {
            FARF(உயர்வு_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->data);

            uintptr_t sig = (uintptr_t) d->func;
            தேர்வு (sig) {
                நிலை HMX_QUEUE_NOOP:    /* noop */;     நிறுத்து;
                நிலை HMX_QUEUE_KILL:    *killed = true; நிறுத்து;
                நிலை HMX_QUEUE_SUSPEND: hmx_unlock(q);  நிறுத்து;
                நிலை HMX_QUEUE_WAKEUP:  hmx_lock(q);    நிறுத்து;
                இயல்பு:
                    hmx_lock(q);
                    htp_trace_event_start(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->data);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    நிறுத்து;
            }

            atomic_fetch_add(&d->done, 1);
        }

        ir = (ir + 1) & q->idx_mask;
        atomic_store(&q->idx_read, ir);
    }
}

நிலையான வெறுமை hmx_queue_thread(வெறுமை * arg) {
    hmx_queue_t q = (hmx_queue_t) arg;

    FARF(உயர்வு_2, "hmx-queue-thread: started");

    bool killed = false;

    unsigned முழு poll_cnt  = HMX_QUEUE_POLL_மொத்தம்;
    unsigned முழு prev_seqn = 0;
    போது (!killed) {
        unsigned முழு seqn = atomic_load(&q->seqn);
        எனில் (seqn == prev_seqn) {
            // drop HVX context while spinning
            எனில் (poll_cnt > 1 && poll_cnt == HMX_QUEUE_POLL_மொத்தம்) {
                qurt_hvx_unlock();
            }
            எனில் (--poll_cnt) { hex_pause(); தொடர்; }
            FARF(உயர்வு_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_QUEUE_POLL_மொத்தம்;
            தொடர்;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_QUEUE_POLL_மொத்தம்;

        FARF(உயர்வு_2, "hmx-queue-thread: new work");

        hmx_queue_process(q, &killed);
    }

    FARF(உயர்வு_2, "hmx-queue-thread: stopped");
}

size_t hmx_queue_sizeof(size_t capacity, uint32_t அடுக்கு_size) {
    capacity = hex_ceil_pow2(capacity);
    size_t size_q = hex_align_up(அளவு(கட்டமைப்பு hmx_queue_s), HEX_L2_வரி_SIZE);
    size_t size_desc = hex_align_up(capacity * அளவு(கட்டமைப்பு hmx_queue_desc), HEX_L2_வரி_SIZE);
    size_t size_அடுக்கு = அடுக்கு_size;
    திரும்பு size_q + size_desc + size_அடுக்கு;
}

size_t hmx_queue_alignof(வெறுமை) {
    திரும்பு HEX_L2_வரி_SIZE;
}

hmx_queue_t hmx_queue_init(வெறுமை * ptr, size_t capacity, uint32_t அடுக்கு_size, uint32_t hap_rctx, கட்டமைப்பு htp_thread_trace * trace) {
    capacity = hex_ceil_pow2(capacity);
    size_t size_q = hex_align_up(அளவு(கட்டமைப்பு hmx_queue_s), HEX_L2_வரி_SIZE);
    size_t size_desc = hex_align_up(capacity * அளவு(கட்டமைப்பு hmx_queue_desc), HEX_L2_வரி_SIZE);

    uint8_t * block = (uint8_t *) ptr;

    hmx_queue_t q = (hmx_queue_t) block; block += size_q;
    memset(q, 0, அளவு(கட்டமைப்பு hmx_queue_s));

    q->capacity = capacity;
    q->idx_mask = capacity - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (கட்டமைப்பு hmx_queue_desc *) block; block += size_desc;
    memset(q->desc, 0, capacity * அளவு(கட்டமைப்பு hmx_queue_desc));

    q->அடுக்கு = block;
    memset(q->அடுக்கு, 0, அடுக்கு_size);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    முழு prio = qurt_thread_get_priority(qurt_thread_get_id());
    எனில் (prio < 1) {
        prio = 1;
    }
    எனில் (prio > QURT_LOWEST_PRIO) {
        prio = QURT_LOWEST_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_அடுக்கு_addr(&attr, q->அடுக்கு);
    qurt_thread_attr_set_அடுக்கு_size(&attr, அடுக்கு_size);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "hmx-queue");

    முழு err = qurt_thread_create(&q->thread, &attr, hmx_queue_thread, q);
    எனில் (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        திரும்பு வெற்று;
    }

    FARF(உயர்வு_2, "hmx-queue: capacity %u\n", capacity);

    திரும்பு q;
}

வெறுமை hmx_queue_free(hmx_queue_t q) {
    எனில் (!q) {
        திரும்பு;
    }

    // Tell the worker to exit.
    hmx_queue_flush(q);
    hmx_queue_signal(q, HMX_QUEUE_KILL);
    hmx_queue_flush(q);

    முழு status;
    qurt_thread_join(q->thread, &status);
}
