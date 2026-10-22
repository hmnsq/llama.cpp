#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_thread.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_futex.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_hvx.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_compute_res.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-queue.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ QURT_ਸਭ_ਤੋਂ_ਨੀਵਾਂ_PRIO (254)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_ਤਾਲਾ(hmx_ਕਤਾਰ_t q)
{
    ਜੇ (!q->hmx_locked) {
        HAP_compute_res_hmx_ਤਾਲਾ(q->hap_rctx);
        q->hmx_locked = true;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_unlock(hmx_ਕਤਾਰ_t q)
{
    ਜੇ (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_ਕਤਾਰ_process(hmx_ਕਤਾਰ_t q, bool* killed) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ir = atomic_ਲੱਦੋ(&q->idx_read);

    ਜਦੋਂ (ir != atomic_ਲੱਦੋ(&q->idx_write)) {
        ਬਣਤਰ hmx_ਕਤਾਰ_desc *d = &q->desc[ir];
        ਜੇ (!d->ਪੂਰਾ) {
            FARF(ਉੱਚਾ_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->ਅੰਕੜੇ);

            uintptr_t sig = (uintptr_t) d->func;
            ਚੋਣ (sig) {
                ਹਾਲਤ HMX_ਕਤਾਰ_NOOP:    /* noop */;     ਰੋਕੋ;
                ਹਾਲਤ HMX_ਕਤਾਰ_KILL:    *killed = true; ਰੋਕੋ;
                ਹਾਲਤ HMX_ਕਤਾਰ_SUSPEND: hmx_unlock(q);  ਰੋਕੋ;
                ਹਾਲਤ HMX_ਕਤਾਰ_WAKEUP:  hmx_ਤਾਲਾ(q);    ਰੋਕੋ;
                ਮੂਲ:
                    hmx_ਤਾਲਾ(q);
                    htp_trace_event_ਸ਼ੁਰੂ(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->ਅੰਕੜੇ);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    ਰੋਕੋ;
            }

            atomic_fetch_ਜੋੜੋ(&d->ਪੂਰਾ, 1);
        }

        ir = (ir + 1) & q->idx_ਨਕਾਬ;
        atomic_store(&q->idx_read, ir);
    }
}

ਸਥਿਰ ਖਾਲੀ hmx_ਕਤਾਰ_thread(ਖਾਲੀ * arg) {
    hmx_ਕਤਾਰ_t q = (hmx_ਕਤਾਰ_t) arg;

    FARF(ਉੱਚਾ_2, "hmx-queue-thread: started");

    bool killed = false;

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ poll_cnt  = HMX_ਕਤਾਰ_POLL_ਗਿਣਤੀ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ prev_seqn = 0;
    ਜਦੋਂ (!killed) {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ seqn = atomic_ਲੱਦੋ(&q->seqn);
        ਜੇ (seqn == prev_seqn) {
            // drop HVX context while spinning
            ਜੇ (poll_cnt > 1 && poll_cnt == HMX_ਕਤਾਰ_POLL_ਗਿਣਤੀ) {
                qurt_hvx_unlock();
            }
            ਜੇ (--poll_cnt) { hex_pause(); ਜਾਰੀ; }
            FARF(ਉੱਚਾ_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_ਕਤਾਰ_POLL_ਗਿਣਤੀ;
            ਜਾਰੀ;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_ਕਤਾਰ_POLL_ਗਿਣਤੀ;

        FARF(ਉੱਚਾ_2, "hmx-queue-thread: new work");

        hmx_ਕਤਾਰ_process(q, &killed);
    }

    FARF(ਉੱਚਾ_2, "hmx-queue-thread: stopped");
}

size_t hmx_ਕਤਾਰ_sizeof(size_t ਸਮਰੱਥਾ, uint32_t ਢੇਰ_ਮਾਪ) {
    ਸਮਰੱਥਾ = hex_ceil_pow2(ਸਮਰੱਥਾ);
    size_t ਮਾਪ_q = hex_align_ਉੱਪਰ(ਆਕਾਰ(ਬਣਤਰ hmx_ਕਤਾਰ_s), HEX_L2_ਸਤਰ_ਮਾਪ);
    size_t ਮਾਪ_desc = hex_align_ਉੱਪਰ(ਸਮਰੱਥਾ * ਆਕਾਰ(ਬਣਤਰ hmx_ਕਤਾਰ_desc), HEX_L2_ਸਤਰ_ਮਾਪ);
    size_t ਮਾਪ_ਢੇਰ = ਢੇਰ_ਮਾਪ;
    ਵਾਪਸ ਮਾਪ_q + ਮਾਪ_desc + ਮਾਪ_ਢੇਰ;
}

size_t hmx_ਕਤਾਰ_alignof(ਖਾਲੀ) {
    ਵਾਪਸ HEX_L2_ਸਤਰ_ਮਾਪ;
}

hmx_ਕਤਾਰ_t hmx_ਕਤਾਰ_init(ਖਾਲੀ * ptr, size_t ਸਮਰੱਥਾ, uint32_t ਢੇਰ_ਮਾਪ, uint32_t hap_rctx, ਬਣਤਰ htp_thread_trace * trace) {
    ਸਮਰੱਥਾ = hex_ceil_pow2(ਸਮਰੱਥਾ);
    size_t ਮਾਪ_q = hex_align_ਉੱਪਰ(ਆਕਾਰ(ਬਣਤਰ hmx_ਕਤਾਰ_s), HEX_L2_ਸਤਰ_ਮਾਪ);
    size_t ਮਾਪ_desc = hex_align_ਉੱਪਰ(ਸਮਰੱਥਾ * ਆਕਾਰ(ਬਣਤਰ hmx_ਕਤਾਰ_desc), HEX_L2_ਸਤਰ_ਮਾਪ);

    uint8_t * ਖੰਡ = (uint8_t *) ptr;

    hmx_ਕਤਾਰ_t q = (hmx_ਕਤਾਰ_t) ਖੰਡ; ਖੰਡ += ਮਾਪ_q;
    memset(q, 0, ਆਕਾਰ(ਬਣਤਰ hmx_ਕਤਾਰ_s));

    q->ਸਮਰੱਥਾ = ਸਮਰੱਥਾ;
    q->idx_ਨਕਾਬ = ਸਮਰੱਥਾ - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (ਬਣਤਰ hmx_ਕਤਾਰ_desc *) ਖੰਡ; ਖੰਡ += ਮਾਪ_desc;
    memset(q->desc, 0, ਸਮਰੱਥਾ * ਆਕਾਰ(ਬਣਤਰ hmx_ਕਤਾਰ_desc));

    q->ਢੇਰ = ਖੰਡ;
    memset(q->ਢੇਰ, 0, ਢੇਰ_ਮਾਪ);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    ਪੂਰਨ prio = qurt_thread_get_ਤਰਜੀਹ(qurt_thread_get_id());
    ਜੇ (prio < 1) {
        prio = 1;
    }
    ਜੇ (prio > QURT_ਸਭ_ਤੋਂ_ਨੀਵਾਂ_PRIO) {
        prio = QURT_ਸਭ_ਤੋਂ_ਨੀਵਾਂ_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_ਸੈੱਟ_ਢੇਰ_addr(&attr, q->ਢੇਰ);
    qurt_thread_attr_ਸੈੱਟ_ਢੇਰ_ਮਾਪ(&attr, ਢੇਰ_ਮਾਪ);
    qurt_thread_attr_ਸੈੱਟ_ਤਰਜੀਹ(&attr, prio);
    qurt_thread_attr_ਸੈੱਟ_name(&attr, "hmx-queue");

    ਪੂਰਨ err = qurt_thread_create(&q->thread, &attr, hmx_ਕਤਾਰ_thread, q);
    ਜੇ (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        ਵਾਪਸ ਨਿਰਾ;
    }

    FARF(ਉੱਚਾ_2, "hmx-queue: capacity %u\n", ਸਮਰੱਥਾ);

    ਵਾਪਸ q;
}

ਖਾਲੀ hmx_ਕਤਾਰ_free(hmx_ਕਤਾਰ_t q) {
    ਜੇ (!q) {
        ਵਾਪਸ;
    }

    // Tell the worker to exit.
    hmx_ਕਤਾਰ_flush(q);
    hmx_ਕਤਾਰ_signal(q, HMX_ਕਤਾਰ_KILL);
    hmx_ਕਤਾਰ_flush(q);

    ਪੂਰਨ ਸਥਿਤੀ;
    qurt_thread_join(q->thread, &ਸਥਿਤੀ);
}
