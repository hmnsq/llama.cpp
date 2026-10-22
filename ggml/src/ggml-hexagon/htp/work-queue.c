#شامل "work-queue.h"
#شامل "hex-utils.h"

#شامل <qurt.h>
#شامل <qurt_hvx.h>

#شامل <stdatomic.h>
#شامل <stdint.h>
#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <string.h>

#شامل "HAP_farf.h"

#define LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
تعریفنوع ساختار {
    work_queue_t  queue;
    unsigned صحیح  id;
} worker_context_t;

ساختار work_queue_task_s {
    work_queue_func_t func;
    تهی *            data;
    unsigned صحیح      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
ساختار work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    تهی *             پشته[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    unsigned صحیح       n_threads;                          // total threads (workers + main)
    unsigned صحیح       n_workers;                          // number of active threads (just workers)

    atomic_uint        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    ساختار work_queue_task_s queue[] __attribute__((aligned(HEX_L2_سطر_SIZE)));
};

ایستا تهی work_queue_thread(تهی * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(بالا_2, "work-queue: thread %u started", me->id);

    unsigned صحیح prev_seqn = 0;

    وقتی (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned صحیح seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        اگر (seqn == prev_seqn) {
            اگر (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } وگرنه {
                qurt_futex_wait(&q->active, 0);
            }
            ادامه;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        unsigned صحیح ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        unsigned صحیح iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

        وقتی (ir != iw) {
            ساختار work_queue_task_s * task = &q->queue[ir];

            unsigned صحیح n = task->n_threads;
            unsigned صحیح i = me->id;
            اگر (i < n) {
                task->func(n, i, task->data);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } وگرنه {
                وقتی (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(بالا_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, تهی * data, unsigned صحیح n) {
    اگر (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        بازگشت false;
    }

    unsigned صحیح ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    unsigned صحیح iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    اگر (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        بازگشت false;
    }

    ساختار work_queue_task_s * task = &q->queue[iw];
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

    وقتی (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    بازگشت true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t پشته_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    size_t size_stacks = پشته_size * n_workers;
    size_t size_q = hex_align_up(اندازه(ساختار work_queue_s) + capacity * اندازه(ساختار work_queue_task_s), HEX_L2_سطر_SIZE);
    بازگشت size_stacks + size_q;
}

size_t work_queue_alignof(تهی) {
    بازگشت 4096;
}

work_queue_t work_queue_init(تهی * ptr, uint32_t n_threads, uint32_t capacity, uint32_t پشته_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    unsigned نویسه * mem_blob = (unsigned نویسه *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + پشته_size * n_workers);
    memset(q, 0, اندازه(ساختار work_queue_s) + capacity * اندازه(ساختار work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    برای (unsigned صحیح i = 0; i < n_workers; i++) {
        q->پشته[i]  = mem_blob; mem_blob += پشته_size;
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
    برای (صحیح i = 0; i < (صحیح) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = پوچ;
        q->queue[i].data      = پوچ;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    برای (unsigned صحیح i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_پشته_addr(&attr, q->پشته[i]);
        qurt_thread_attr_set_پشته_size(&attr, پشته_size);

        نویسه thread_name[32];
        snprintf(thread_name, اندازه(thread_name), "work-queue:%u", i);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        صحیح prio = qurt_thread_get_priority(qurt_thread_get_id());
        اگر (prio < 1) {
            prio = 1;
        }
        اگر (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        صحیح err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (تهی *) &q->context[i]);
        اگر (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            بازگشت پوچ;
        }
    }

    بازگشت q;
}

تهی work_queue_free(work_queue_t q) {
    اگر (!q) { بازگشت; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    برای (unsigned صحیح i = 0; i < q->n_workers; i++) {
        اگر (q->thread[i]) {
            صحیح status;
            (تهی) qurt_thread_join(q->thread[i], &status);
        }
    }
}

تهی work_queue_wakeup(work_queue_t q) {
    اگر (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

تهی work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
