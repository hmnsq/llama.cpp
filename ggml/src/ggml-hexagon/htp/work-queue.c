#ਅਨ੍ਤਰ੍ਭਾਵ "work-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <qurt.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_hvx.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <stdatomic.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "HAP_farf.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਭ_ਤੋਂ_ਨੀਵਾਂ_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    work_ਕਤਾਰ_t  ਕਤਾਰ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ  id;
} ਕਾਮਾ_context_t;

ਬਣਤਰ work_ਕਤਾਰ_task_s {
    work_ਕਤਾਰ_func_t func;
    ਖਾਲੀ *            ਅੰਕੜੇ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ      n_ਧਾਗੇ;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
ਬਣਤਰ work_ਕਤਾਰ_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_ਨਕਾਬ;
    uint32_t           ਸਮਰੱਥਾ;

    qurt_thread_t      thread[WORK_ਕਤਾਰ_MAX_N_ਧਾਗੇ];   // thread ID's of the workers
    ਕਾਮਾ_context_t   context[WORK_ਕਤਾਰ_MAX_N_ਧਾਗੇ];  // worker contexts
    ਖਾਲੀ *             ਢੇਰ[WORK_ਕਤਾਰ_MAX_N_ਧਾਗੇ];    // thread stack pointers
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ       n_ਧਾਗੇ;                          // total threads (workers + main)
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ       n_workers;                          // number of active threads (just workers)

    atomic_uint        ਸਰਗਰਮ;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    ਬਣਤਰ work_ਕਤਾਰ_task_s ਕਤਾਰ[] __attribute__((aligned(HEX_L2_ਸਤਰ_ਮਾਪ)));
};

ਸਥਿਰ ਖਾਲੀ work_ਕਤਾਰ_thread(ਖਾਲੀ * context) {
    ਕਾਮਾ_context_t * me = (ਕਾਮਾ_context_t *) context;
    work_ਕਤਾਰ_t       q  = me->ਕਤਾਰ;

    FARF(ਉੱਚਾ_2, "work-queue: thread %u started", me->id);

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ prev_seqn = 0;

    ਜਦੋਂ (!atomic_ਲੱਦੋ_explicit(&q->killed, memory_order_relaxed)) {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ seqn = atomic_ਲੱਦੋ_explicit(&q->seqn, memory_order_acquire);
        ਜੇ (seqn == prev_seqn) {
            ਜੇ (atomic_ਲੱਦੋ_explicit(&q->ਸਰਗਰਮ, memory_order_relaxed)) {
                hex_pause();
            } ਨਹੀਂਤਾਂ {
                qurt_futex_wait(&q->ਸਰਗਰਮ, 0);
            }
            ਜਾਰੀ;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ir = atomic_ਲੱਦੋ_explicit(&q->idx_read, memory_order_relaxed);
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ iw = atomic_ਲੱਦੋ_explicit(&q->idx_write, memory_order_relaxed);

        ਜਦੋਂ (ir != iw) {
            ਬਣਤਰ work_ਕਤਾਰ_task_s * task = &q->ਕਤਾਰ[ir];

            ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n = task->n_ਧਾਗੇ;
            ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = me->id;
            ਜੇ (i < n) {
                task->func(n, i, task->ਅੰਕੜੇ);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } ਨਹੀਂਤਾਂ {
                ਜਦੋਂ (atomic_ਲੱਦੋ_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_ਨਕਾਬ;
        }
    }

    FARF(ਉੱਚਾ_2, "work-queue: thread %u stopped", me->id);
}

bool work_ਕਤਾਰ_ਲੜੀਵਾਰ_async(work_ਕਤਾਰ_t q, work_ਕਤਾਰ_func_t func, ਖਾਲੀ * ਅੰਕੜੇ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n) {
    ਜੇ (n > q->n_ਧਾਗੇ) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_ਧਾਗੇ);
        ਵਾਪਸ false;
    }

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ir = atomic_ਲੱਦੋ_explicit(&q->idx_read,  memory_order_relaxed);
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ iw = atomic_ਲੱਦੋ_explicit(&q->idx_write, memory_order_relaxed);

    ਜੇ (((iw + 1) & q->idx_ਨਕਾਬ) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        ਵਾਪਸ false;
    }

    ਬਣਤਰ work_ਕਤਾਰ_task_s * task = &q->ਕਤਾਰ[iw];
    task->func      = func;
    task->ਅੰਕੜੇ      = ਅੰਕੜੇ;
    task->n_ਧਾਗੇ = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_ਨਕਾਬ, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_ਜੋੜੋ_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, ਅੰਕੜੇ);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    ਜਦੋਂ (atomic_ਲੱਦੋ_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_ਨਕਾਬ, memory_order_relaxed);

    ਵਾਪਸ true;
}

size_t work_ਕਤਾਰ_sizeof(uint32_t n_ਧਾਗੇ, uint32_t ਸਮਰੱਥਾ, uint32_t ਢੇਰ_ਮਾਪ) {
    ਸਮਰੱਥਾ = hex_ceil_pow2(ਸਮਰੱਥਾ);
    uint32_t n_workers = n_ਧਾਗੇ > 1 ? n_ਧਾਗੇ - 1 : 0;
    size_t ਮਾਪ_stacks = ਢੇਰ_ਮਾਪ * n_workers;
    size_t ਮਾਪ_q = hex_align_ਉੱਪਰ(ਆਕਾਰ(ਬਣਤਰ work_ਕਤਾਰ_s) + ਸਮਰੱਥਾ * ਆਕਾਰ(ਬਣਤਰ work_ਕਤਾਰ_task_s), HEX_L2_ਸਤਰ_ਮਾਪ);
    ਵਾਪਸ ਮਾਪ_stacks + ਮਾਪ_q;
}

size_t work_ਕਤਾਰ_alignof(ਖਾਲੀ) {
    ਵਾਪਸ 4096;
}

work_ਕਤਾਰ_t work_ਕਤਾਰ_init(ਖਾਲੀ * ptr, uint32_t n_ਧਾਗੇ, uint32_t ਸਮਰੱਥਾ, uint32_t ਢੇਰ_ਮਾਪ) {
    ਸਮਰੱਥਾ = hex_ceil_pow2(ਸਮਰੱਥਾ);
    uint32_t n_workers = n_ਧਾਗੇ > 1 ? n_ਧਾਗੇ - 1 : 0;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ * mem_blob = (ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *) ptr;

    work_ਕਤਾਰ_t q = (work_ਕਤਾਰ_t) (mem_blob + ਢੇਰ_ਮਾਪ * n_workers);
    memset(q, 0, ਆਕਾਰ(ਬਣਤਰ work_ਕਤਾਰ_s) + ਸਮਰੱਥਾ * ਆਕਾਰ(ਬਣਤਰ work_ਕਤਾਰ_task_s));

    q->n_ਧਾਗੇ = n_ਧਾਗੇ;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->ਸਮਰੱਥਾ = ਸਮਰੱਥਾ;

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < n_workers; i++) {
        q->ਢੇਰ[i]  = mem_blob; mem_blob += ਢੇਰ_ਮਾਪ;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].ਕਤਾਰ = q;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->ਸਰਗਰਮ,    0);
    q->idx_ਨਕਾਬ  = ਸਮਰੱਥਾ - 1;
    q->killed    = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < (ਪੂਰਨ) ਸਮਰੱਥਾ; i++) {
        atomic_init(&q->ਕਤਾਰ[i].barrier, 0);
        q->ਕਤਾਰ[i].func      = ਨਿਰਾ;
        q->ਕਤਾਰ[i].ਅੰਕੜੇ      = ਨਿਰਾ;
        q->ਕਤਾਰ[i].n_ਧਾਗੇ = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < n_workers; i++) {
        qurt_thread_attr_ਸੈੱਟ_ਢੇਰ_addr(&attr, q->ਢੇਰ[i]);
        qurt_thread_attr_ਸੈੱਟ_ਢੇਰ_ਮਾਪ(&attr, ਢੇਰ_ਮਾਪ);

        ਅੱਖਰ thread_name[32];
        snprintf(thread_name, ਆਕਾਰ(thread_name), "work-queue:%u", i);
        qurt_thread_attr_ਸੈੱਟ_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        ਪੂਰਨ prio = qurt_thread_get_ਤਰਜੀਹ(qurt_thread_get_id());
        ਜੇ (prio < 1) {
            prio = 1;
        }
        ਜੇ (prio > ਸਭ_ਤੋਂ_ਨੀਵਾਂ_USABLE_QURT_PRIO) {
            prio = ਸਭ_ਤੋਂ_ਨੀਵਾਂ_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_ਸੈੱਟ_ਤਰਜੀਹ(&attr, prio);

        ਪੂਰਨ err = qurt_thread_create(&q->thread[i], &attr, work_ਕਤਾਰ_thread, (ਖਾਲੀ *) &q->context[i]);
        ਜੇ (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_ਕਤਾਰ_free(q);
            ਵਾਪਸ ਨਿਰਾ;
        }
    }

    ਵਾਪਸ q;
}

ਖਾਲੀ work_ਕਤਾਰ_free(work_ਕਤਾਰ_t q) {
    ਜੇ (!q) { ਵਾਪਸ; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->ਸਰਗਰਮ,   1, memory_order_release);
    atomic_fetch_ਜੋੜੋ_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->ਸਰਗਰਮ, q->n_workers);

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < q->n_workers; i++) {
        ਜੇ (q->thread[i]) {
            ਪੂਰਨ ਸਥਿਤੀ;
            (ਖਾਲੀ) qurt_thread_join(q->thread[i], &ਸਥਿਤੀ);
        }
    }
}

ਖਾਲੀ work_ਕਤਾਰ_wakeup(work_ਕਤਾਰ_t q) {
    ਜੇ (!atomic_ਲੱਦੋ_explicit(&q->ਸਰਗਰਮ, memory_order_relaxed)) {
        atomic_store_explicit(&q->ਸਰਗਰਮ, 1, memory_order_release);
        qurt_futex_wake(&q->ਸਰਗਰਮ, q->n_workers);
    }
}

ਖਾਲੀ work_ਕਤਾਰ_suspend(work_ਕਤਾਰ_t q) {
    atomic_store_explicit(&q->ਸਰਗਰਮ, 0, memory_order_release);
}
