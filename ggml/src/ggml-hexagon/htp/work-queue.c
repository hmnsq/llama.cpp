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

#define LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ {
    work_queue_t  queue;
    unsigned ಪೂರ್ಣಾಙ್ಕ  id;
} worker_context_t;

ಸಂರಚನಾ work_queue_task_s {
    work_queue_func_t func;
    ಶೂನ್ಯ *            data;
    unsigned ಪೂರ್ಣಾಙ್ಕ      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
ಸಂರಚನಾ work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    ಶೂನ್ಯ *             ರಾಶಿ[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    unsigned ಪೂರ್ಣಾಙ್ಕ       n_threads;                          // total threads (workers + main)
    unsigned ಪೂರ್ಣಾಙ್ಕ       n_workers;                          // number of active threads (just workers)

    atomic_uint        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    ಸಂರಚನಾ work_queue_task_s queue[] __attribute__((aligned(HEX_L2_ಸಾಲು_SIZE)));
};

ಸ್ಥಿರ ಶೂನ್ಯ work_queue_thread(ಶೂನ್ಯ * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(ಮೇಲಿನ_2, "work-queue: thread %u started", me->id);

    unsigned ಪೂರ್ಣಾಙ್ಕ prev_seqn = 0;

    ಯಾವತ್ (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned ಪೂರ್ಣಾಙ್ಕ seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        ಯದಿ (seqn == prev_seqn) {
            ಯದಿ (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } ಅನ್ಯಥಾ {
                qurt_futex_wait(&q->active, 0);
            }
            ಅನುವರ್ತ;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        unsigned ಪೂರ್ಣಾಙ್ಕ ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        unsigned ಪೂರ್ಣಾಙ್ಕ iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

        ಯಾವತ್ (ir != iw) {
            ಸಂರಚನಾ work_queue_task_s * task = &q->queue[ir];

            unsigned ಪೂರ್ಣಾಙ್ಕ n = task->n_threads;
            unsigned ಪೂರ್ಣಾಙ್ಕ i = me->id;
            ಯದಿ (i < n) {
                task->func(n, i, task->data);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } ಅನ್ಯಥಾ {
                ಯಾವತ್ (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(ಮೇಲಿನ_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, ಶೂನ್ಯ * data, unsigned ಪೂರ್ಣಾಙ್ಕ n) {
    ಯದಿ (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        ಪ್ರತ್ಯಾಗಮ false;
    }

    unsigned ಪೂರ್ಣಾಙ್ಕ ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    unsigned ಪೂರ್ಣಾಙ್ಕ iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    ಯದಿ (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        ಪ್ರತ್ಯಾಗಮ false;
    }

    ಸಂರಚನಾ work_queue_task_s * task = &q->queue[iw];
    task->func      = func;
    task->data      = data;
    task->n_threads = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_mask, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, data);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    ಯಾವತ್ (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    ಪ್ರತ್ಯಾಗಮ true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t ರಾಶಿ_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    size_t size_stacks = ರಾಶಿ_size * n_workers;
    size_t size_q = hex_align_up(ಪರಿಮಾಣ(ಸಂರಚನಾ work_queue_s) + capacity * ಪರಿಮಾಣ(ಸಂರಚನಾ work_queue_task_s), HEX_L2_ಸಾಲು_SIZE);
    ಪ್ರತ್ಯಾಗಮ size_stacks + size_q;
}

size_t work_queue_alignof(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ 4096;
}

work_queue_t work_queue_init(ಶೂನ್ಯ * ptr, uint32_t n_threads, uint32_t capacity, uint32_t ರಾಶಿ_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    unsigned ವರ್ಣ * mem_blob = (unsigned ವರ್ಣ *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + ರಾಶಿ_size * n_workers);
    memset(q, 0, ಪರಿಮಾಣ(ಸಂರಚನಾ work_queue_s) + capacity * ಪರಿಮಾಣ(ಸಂರಚನಾ work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_workers; i++) {
        q->ರಾಶಿ[i]  = mem_blob; mem_blob += ರಾಶಿ_size;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].queue = q;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->active,    0);
    q->idx_mask  = capacity - 1;
    q->killed    = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < (ಪೂರ್ಣಾಙ್ಕ) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = ರಿಕ್ತ;
        q->queue[i].data      = ರಿಕ್ತ;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_ರಾಶಿ_addr(&attr, q->ರಾಶಿ[i]);
        qurt_thread_attr_set_ರಾಶಿ_size(&attr, ರಾಶಿ_size);

        ವರ್ಣ thread_name[32];
        snprintf(thread_name, ಪರಿಮಾಣ(thread_name), "work-queue:%u", i);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        ಪೂರ್ಣಾಙ್ಕ prio = qurt_thread_get_priority(qurt_thread_get_id());
        ಯದಿ (prio < 1) {
            prio = 1;
        }
        ಯದಿ (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        ಪೂರ್ಣಾಙ್ಕ err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (ಶೂನ್ಯ *) &q->context[i]);
        ಯದಿ (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ q;
}

ಶೂನ್ಯ work_queue_free(work_queue_t q) {
    ಯದಿ (!q) { ಪ್ರತ್ಯಾಗಮ; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i = 0; i < q->n_workers; i++) {
        ಯದಿ (q->thread[i]) {
            ಪೂರ್ಣಾಙ್ಕ status;
            (ಶೂನ್ಯ) qurt_thread_join(q->thread[i], &status);
        }
    }
}

ಶೂನ್ಯ work_queue_wakeup(work_queue_t q) {
    ಯದಿ (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

ಶೂನ್ಯ work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
