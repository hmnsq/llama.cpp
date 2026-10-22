#ಅನ್ತರ್ಭಾವ "work-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-utils.h"

#ಅನ್ತರ್ಭಾವ <qurt.h>
#ಅನ್ತರ್ಭಾವ <qurt_hvx.h>

#ಅನ್ತರ್ಭಾವ <stdatomic.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "HAP_farf.h"

#ವ್ಯಾಖ್ಯಾನಿಸು ಅತಿಕಡಿಮೆ_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    work_ಸರತಿ_t  ಸರತಿ;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ  id;
} ಕೆಲಸಗಾರ_context_t;

ರಚನೆ work_ಸರತಿ_task_s {
    work_ಸರತಿ_func_t func;
    ಶೂನ್ಯ *            ದತ್ತಾಂಶ;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ      n_ಎಳೆಗಳು;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
ರಚನೆ work_ಸರತಿ_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_ಮುಖವಾಡ;
    uint32_t           ಸಾಮರ್ಥ್ಯ;

    qurt_thread_t      thread[WORK_ಸರತಿ_MAX_N_ಎಳೆಗಳು];   // thread ID's of the workers
    ಕೆಲಸಗಾರ_context_t   context[WORK_ಸರತಿ_MAX_N_ಎಳೆಗಳು];  // worker contexts
    ಶೂನ್ಯ *             ರಾಶಿ[WORK_ಸರತಿ_MAX_N_ಎಳೆಗಳು];    // thread stack pointers
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ       n_ಎಳೆಗಳು;                          // total threads (workers + main)
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ       n_workers;                          // number of active threads (just workers)

    atomic_uint        ಸಕ್ರಿಯ;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    ರಚನೆ work_ಸರತಿ_task_s ಸರತಿ[] __attribute__((aligned(HEX_L2_ಸಾಲು_ಗಾತ್ರ)));
};

ಸ್ಥಿರ ಶೂನ್ಯ work_ಸರತಿ_thread(ಶೂನ್ಯ * context) {
    ಕೆಲಸಗಾರ_context_t * me = (ಕೆಲಸಗಾರ_context_t *) context;
    work_ಸರತಿ_t       q  = me->ಸರತಿ;

    FARF(ಮೇಲಿನ_2, "work-queue: thread %u started", me->id);

    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ prev_seqn = 0;

    ಆಗ (!atomic_ತುಂಬು_explicit(&q->killed, memory_order_relaxed)) {
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ seqn = atomic_ತುಂಬು_explicit(&q->seqn, memory_order_acquire);
        ಆದರೆ (seqn == prev_seqn) {
            ಆದರೆ (atomic_ತುಂಬು_explicit(&q->ಸಕ್ರಿಯ, memory_order_relaxed)) {
                hex_pause();
            } ಇಲ್ಲದಿದ್ದರೆ {
                qurt_futex_wait(&q->ಸಕ್ರಿಯ, 0);
            }
            ಮುಂದುವರಿ;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ir = atomic_ತುಂಬು_explicit(&q->idx_read, memory_order_relaxed);
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ iw = atomic_ತುಂಬು_explicit(&q->idx_write, memory_order_relaxed);

        ಆಗ (ir != iw) {
            ರಚನೆ work_ಸರತಿ_task_s * task = &q->ಸರತಿ[ir];

            ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n = task->n_ಎಳೆಗಳು;
            ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = me->id;
            ಆದರೆ (i < n) {
                task->func(n, i, task->ದತ್ತಾಂಶ);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆಗ (atomic_ತುಂಬು_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_ಮುಖವಾಡ;
        }
    }

    FARF(ಮೇಲಿನ_2, "work-queue: thread %u stopped", me->id);
}

bool work_ಸರತಿ_ಸತತ_async(work_ಸರತಿ_t q, work_ಸರತಿ_func_t func, ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n) {
    ಆದರೆ (n > q->n_ಎಳೆಗಳು) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_ಎಳೆಗಳು);
        ಹಿಂತಿರುಗಿ false;
    }

    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ir = atomic_ತುಂಬು_explicit(&q->idx_read,  memory_order_relaxed);
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ iw = atomic_ತುಂಬು_explicit(&q->idx_write, memory_order_relaxed);

    ಆದರೆ (((iw + 1) & q->idx_ಮುಖವಾಡ) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        ಹಿಂತಿರುಗಿ false;
    }

    ರಚನೆ work_ಸರತಿ_task_s * task = &q->ಸರತಿ[iw];
    task->func      = func;
    task->ದತ್ತಾಂಶ      = ದತ್ತಾಂಶ;
    task->n_ಎಳೆಗಳು = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_ಮುಖವಾಡ, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_ಕೂಡಿಸು_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, ದತ್ತಾಂಶ);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    ಆಗ (atomic_ತುಂಬು_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_ಮುಖವಾಡ, memory_order_relaxed);

    ಹಿಂತಿರುಗಿ true;
}

size_t work_ಸರತಿ_sizeof(uint32_t n_ಎಳೆಗಳು, uint32_t ಸಾಮರ್ಥ್ಯ, uint32_t ರಾಶಿ_ಗಾತ್ರ) {
    ಸಾಮರ್ಥ್ಯ = hex_ceil_pow2(ಸಾಮರ್ಥ್ಯ);
    uint32_t n_workers = n_ಎಳೆಗಳು > 1 ? n_ಎಳೆಗಳು - 1 : 0;
    size_t ಗಾತ್ರ_stacks = ರಾಶಿ_ಗಾತ್ರ * n_workers;
    size_t ಗಾತ್ರ_q = hex_align_ಮೇಲೆ(ಗಾತ್ರ(ರಚನೆ work_ಸರತಿ_s) + ಸಾಮರ್ಥ್ಯ * ಗಾತ್ರ(ರಚನೆ work_ಸರತಿ_task_s), HEX_L2_ಸಾಲು_ಗಾತ್ರ);
    ಹಿಂತಿರುಗಿ ಗಾತ್ರ_stacks + ಗಾತ್ರ_q;
}

size_t work_ಸರತಿ_alignof(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ 4096;
}

work_ಸರತಿ_t work_ಸರತಿ_init(ಶೂನ್ಯ * ptr, uint32_t n_ಎಳೆಗಳು, uint32_t ಸಾಮರ್ಥ್ಯ, uint32_t ರಾಶಿ_ಗಾತ್ರ) {
    ಸಾಮರ್ಥ್ಯ = hex_ceil_pow2(ಸಾಮರ್ಥ್ಯ);
    uint32_t n_workers = n_ಎಳೆಗಳು > 1 ? n_ಎಳೆಗಳು - 1 : 0;
    ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ * mem_blob = (ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *) ptr;

    work_ಸರತಿ_t q = (work_ಸರತಿ_t) (mem_blob + ರಾಶಿ_ಗಾತ್ರ * n_workers);
    memset(q, 0, ಗಾತ್ರ(ರಚನೆ work_ಸರತಿ_s) + ಸಾಮರ್ಥ್ಯ * ಗಾತ್ರ(ರಚನೆ work_ಸರತಿ_task_s));

    q->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->ಸಾಮರ್ಥ್ಯ = ಸಾಮರ್ಥ್ಯ;

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < n_workers; i++) {
        q->ರಾಶಿ[i]  = mem_blob; mem_blob += ರಾಶಿ_ಗಾತ್ರ;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].ಸರತಿ = q;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->ಸಕ್ರಿಯ,    0);
    q->idx_ಮುಖವಾಡ  = ಸಾಮರ್ಥ್ಯ - 1;
    q->killed    = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < (ಪೂರ್ಣಾಂಕ) ಸಾಮರ್ಥ್ಯ; i++) {
        atomic_init(&q->ಸರತಿ[i].barrier, 0);
        q->ಸರತಿ[i].func      = ಖಾಲಿ;
        q->ಸರತಿ[i].ದತ್ತಾಂಶ      = ಖಾಲಿ;
        q->ಸರತಿ[i].n_ಎಳೆಗಳು = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < n_workers; i++) {
        qurt_thread_attr_ಗಣ_ರಾಶಿ_addr(&attr, q->ರಾಶಿ[i]);
        qurt_thread_attr_ಗಣ_ರಾಶಿ_ಗಾತ್ರ(&attr, ರಾಶಿ_ಗಾತ್ರ);

        ಅಕ್ಷರ thread_name[32];
        snprintf(thread_name, ಗಾತ್ರ(thread_name), "work-queue:%u", i);
        qurt_thread_attr_ಗಣ_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        ಪೂರ್ಣಾಂಕ prio = qurt_thread_get_ಆದ್ಯತೆ(qurt_thread_get_id());
        ಆದರೆ (prio < 1) {
            prio = 1;
        }
        ಆದರೆ (prio > ಅತಿಕಡಿಮೆ_USABLE_QURT_PRIO) {
            prio = ಅತಿಕಡಿಮೆ_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_ಗಣ_ಆದ್ಯತೆ(&attr, prio);

        ಪೂರ್ಣಾಂಕ err = qurt_thread_create(&q->thread[i], &attr, work_ಸರತಿ_thread, (ಶೂನ್ಯ *) &q->context[i]);
        ಆದರೆ (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_ಸರತಿ_free(q);
            ಹಿಂತಿರುಗಿ ಖಾಲಿ;
        }
    }

    ಹಿಂತಿರುಗಿ q;
}

ಶೂನ್ಯ work_ಸರತಿ_free(work_ಸರತಿ_t q) {
    ಆದರೆ (!q) { ಹಿಂತಿರುಗಿ; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->ಸಕ್ರಿಯ,   1, memory_order_release);
    atomic_fetch_ಕೂಡಿಸು_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->ಸಕ್ರಿಯ, q->n_workers);

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < q->n_workers; i++) {
        ಆದರೆ (q->thread[i]) {
            ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2;
            (ಶೂನ್ಯ) qurt_thread_join(q->thread[i], &ಸ್ಥಿತಿ_2);
        }
    }
}

ಶೂನ್ಯ work_ಸರತಿ_wakeup(work_ಸರತಿ_t q) {
    ಆದರೆ (!atomic_ತುಂಬು_explicit(&q->ಸಕ್ರಿಯ, memory_order_relaxed)) {
        atomic_store_explicit(&q->ಸಕ್ರಿಯ, 1, memory_order_release);
        qurt_futex_wake(&q->ಸಕ್ರಿಯ, q->n_workers);
    }
}

ಶೂನ್ಯ work_ಸರತಿ_suspend(work_ಸರತಿ_t q) {
    atomic_store_explicit(&q->ಸಕ್ರಿಯ, 0, memory_order_release);
}
