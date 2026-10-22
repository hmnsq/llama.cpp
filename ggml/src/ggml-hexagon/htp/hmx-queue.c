#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ <qurt_thread.h>
#ಅನ್ತರ್ಭಾವ <qurt_futex.h>
#ಅನ್ತರ್ಭಾವ <qurt_hvx.h>

#ಅನ್ತರ್ಭಾವ <HAP_compute_res.h>

#ಅನ್ತರ್ಭಾವ "hmx-queue.h"

#ವ್ಯಾಖ್ಯಾನಿಸು QURT_ಅತಿಕಡಿಮೆ_PRIO (254)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_ಬೀಗ(hmx_ಸರತಿ_t q)
{
    ಆದರೆ (!q->hmx_locked) {
        HAP_compute_res_hmx_ಬೀಗ(q->hap_rctx);
        q->hmx_locked = true;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_unlock(hmx_ಸರತಿ_t q)
{
    ಆದರೆ (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_ಸರತಿ_process(hmx_ಸರತಿ_t q, bool* killed) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ir = atomic_ತುಂಬು(&q->idx_read);

    ಆಗ (ir != atomic_ತುಂಬು(&q->idx_write)) {
        ರಚನೆ hmx_ಸರತಿ_desc *d = &q->desc[ir];
        ಆದರೆ (!d->ಮುಗಿದ) {
            FARF(ಮೇಲಿನ_2, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->ದತ್ತಾಂಶ);

            uintptr_t sig = (uintptr_t) d->func;
            ಆಯ್ಕೆ (sig) {
                ಸ್ಥಿತಿ HMX_ಸರತಿ_NOOP:    /* noop */;     ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HMX_ಸರತಿ_KILL:    *killed = true; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HMX_ಸರತಿ_SUSPEND: hmx_unlock(q);  ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HMX_ಸರತಿ_WAKEUP:  hmx_ಬೀಗ(q);    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    hmx_ಬೀಗ(q);
                    htp_trace_event_ಆರಂಭ(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    d->func(d->ದತ್ತಾಂಶ);
                    htp_trace_event_stop(q->trace, HTP_TRACE_EVT_HMX_COMP, ir);
                    ನಿಲ್ಲಿಸು;
            }

            atomic_fetch_ಕೂಡಿಸು(&d->ಮುಗಿದ, 1);
        }

        ir = (ir + 1) & q->idx_ಮುಖವಾಡ;
        atomic_store(&q->idx_read, ir);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hmx_ಸರತಿ_thread(ಶೂನ್ಯ * arg) {
    hmx_ಸರತಿ_t q = (hmx_ಸರತಿ_t) arg;

    FARF(ಮೇಲಿನ_2, "hmx-queue-thread: started");

    bool killed = false;

    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ poll_cnt  = HMX_ಸರತಿ_POLL_ಎಣಿಕೆ;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ prev_seqn = 0;
    ಆಗ (!killed) {
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ seqn = atomic_ತುಂಬು(&q->seqn);
        ಆದರೆ (seqn == prev_seqn) {
            // drop HVX context while spinning
            ಆದರೆ (poll_cnt > 1 && poll_cnt == HMX_ಸರತಿ_POLL_ಎಣಿಕೆ) {
                qurt_hvx_unlock();
            }
            ಆದರೆ (--poll_cnt) { hex_pause(); ಮುಂದುವರಿ; }
            FARF(ಮೇಲಿನ_2, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = HMX_ಸರತಿ_POLL_ಎಣಿಕೆ;
            ಮುಂದುವರಿ;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_ಸರತಿ_POLL_ಎಣಿಕೆ;

        FARF(ಮೇಲಿನ_2, "hmx-queue-thread: new work");

        hmx_ಸರತಿ_process(q, &killed);
    }

    FARF(ಮೇಲಿನ_2, "hmx-queue-thread: stopped");
}

size_t hmx_ಸರತಿ_sizeof(size_t ಸಾಮರ್ಥ್ಯ, uint32_t ರಾಶಿ_ಗಾತ್ರ) {
    ಸಾಮರ್ಥ್ಯ = hex_ceil_pow2(ಸಾಮರ್ಥ್ಯ);
    size_t ಗಾತ್ರ_q = hex_align_ಮೇಲೆ(ಗಾತ್ರ(ರಚನೆ hmx_ಸರತಿ_s), HEX_L2_ಸಾಲು_ಗಾತ್ರ);
    size_t ಗಾತ್ರ_desc = hex_align_ಮೇಲೆ(ಸಾಮರ್ಥ್ಯ * ಗಾತ್ರ(ರಚನೆ hmx_ಸರತಿ_desc), HEX_L2_ಸಾಲು_ಗಾತ್ರ);
    size_t ಗಾತ್ರ_ರಾಶಿ = ರಾಶಿ_ಗಾತ್ರ;
    ಹಿಂತಿರುಗಿ ಗಾತ್ರ_q + ಗಾತ್ರ_desc + ಗಾತ್ರ_ರಾಶಿ;
}

size_t hmx_ಸರತಿ_alignof(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ HEX_L2_ಸಾಲು_ಗಾತ್ರ;
}

hmx_ಸರತಿ_t hmx_ಸರತಿ_init(ಶೂನ್ಯ * ptr, size_t ಸಾಮರ್ಥ್ಯ, uint32_t ರಾಶಿ_ಗಾತ್ರ, uint32_t hap_rctx, ರಚನೆ htp_thread_trace * trace) {
    ಸಾಮರ್ಥ್ಯ = hex_ceil_pow2(ಸಾಮರ್ಥ್ಯ);
    size_t ಗಾತ್ರ_q = hex_align_ಮೇಲೆ(ಗಾತ್ರ(ರಚನೆ hmx_ಸರತಿ_s), HEX_L2_ಸಾಲು_ಗಾತ್ರ);
    size_t ಗಾತ್ರ_desc = hex_align_ಮೇಲೆ(ಸಾಮರ್ಥ್ಯ * ಗಾತ್ರ(ರಚನೆ hmx_ಸರತಿ_desc), HEX_L2_ಸಾಲು_ಗಾತ್ರ);

    uint8_t * ಖಂಡ = (uint8_t *) ptr;

    hmx_ಸರತಿ_t q = (hmx_ಸರತಿ_t) ಖಂಡ; ಖಂಡ += ಗಾತ್ರ_q;
    memset(q, 0, ಗಾತ್ರ(ರಚನೆ hmx_ಸರತಿ_s));

    q->ಸಾಮರ್ಥ್ಯ = ಸಾಮರ್ಥ್ಯ;
    q->idx_ಮುಖವಾಡ = ಸಾಮರ್ಥ್ಯ - 1;
    q->hap_rctx = hap_rctx;
    q->external_mem = true;

    q->desc = (ರಚನೆ hmx_ಸರತಿ_desc *) ಖಂಡ; ಖಂಡ += ಗಾತ್ರ_desc;
    memset(q->desc, 0, ಸಾಮರ್ಥ್ಯ * ಗಾತ್ರ(ರಚನೆ hmx_ಸರತಿ_desc));

    q->ರಾಶಿ = ಖಂಡ;
    memset(q->ರಾಶಿ, 0, ರಾಶಿ_ಗಾತ್ರ);

    q->trace = trace;

    // Match caller thread priority (same pattern as worker-pool.c).
    ಪೂರ್ಣಾಂಕ prio = qurt_thread_get_ಆದ್ಯತೆ(qurt_thread_get_id());
    ಆದರೆ (prio < 1) {
        prio = 1;
    }
    ಆದರೆ (prio > QURT_ಅತಿಕಡಿಮೆ_PRIO) {
        prio = QURT_ಅತಿಕಡಿಮೆ_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_ಗಣ_ರಾಶಿ_addr(&attr, q->ರಾಶಿ);
    qurt_thread_attr_ಗಣ_ರಾಶಿ_ಗಾತ್ರ(&attr, ರಾಶಿ_ಗಾತ್ರ);
    qurt_thread_attr_ಗಣ_ಆದ್ಯತೆ(&attr, prio);
    qurt_thread_attr_ಗಣ_name(&attr, "hmx-queue");

    ಪೂರ್ಣಾಂಕ err = qurt_thread_create(&q->thread, &attr, hmx_ಸರತಿ_thread, q);
    ಆದರೆ (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    FARF(ಮೇಲಿನ_2, "hmx-queue: capacity %u\n", ಸಾಮರ್ಥ್ಯ);

    ಹಿಂತಿರುಗಿ q;
}

ಶೂನ್ಯ hmx_ಸರತಿ_free(hmx_ಸರತಿ_t q) {
    ಆದರೆ (!q) {
        ಹಿಂತಿರುಗಿ;
    }

    // Tell the worker to exit.
    hmx_ಸರತಿ_flush(q);
    hmx_ಸರತಿ_signal(q, HMX_ಸರತಿ_KILL);
    hmx_ಸರತಿ_flush(q);

    ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2;
    qurt_thread_join(q->thread, &ಸ್ಥಿತಿ_2);
}
