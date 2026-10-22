#অন্তর্ভাব "work-queue.h"
#অন্তর্ভাব "hex-utils.h"

#অন্তর্ভাব <qurt.h>
#অন্তর্ভাব <qurt_hvx.h>

#অন্তর্ভাব <stdatomic.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <string.h>

#অন্তর্ভাব "HAP_farf.h"

#define LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
প্রকারনির্দেশ সংরচনা {
    work_queue_t  queue;
    unsigned পূর্ণাঙ্ক  id;
} worker_context_t;

সংরচনা work_queue_task_s {
    work_queue_func_t func;
    শূন্য *            data;
    unsigned পূর্ণাঙ্ক      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
সংরচনা work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    শূন্য *             স্তূপ[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    unsigned পূর্ণাঙ্ক       n_threads;                          // total threads (workers + main)
    unsigned পূর্ণাঙ্ক       n_workers;                          // number of active threads (just workers)

    atomic_uint        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    সংরচনা work_queue_task_s queue[] __attribute__((aligned(HEX_L2_সারি_SIZE)));
};

স্থির শূন্য work_queue_thread(শূন্য * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(উপরের_2, "work-queue: thread %u started", me->id);

    unsigned পূর্ণাঙ্ক prev_seqn = 0;

    যাবত্ (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned পূর্ণাঙ্ক seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        যদি (seqn == prev_seqn) {
            যদি (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } অন্যথা {
                qurt_futex_wait(&q->active, 0);
            }
            অনুবর্ত;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        unsigned পূর্ণাঙ্ক ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        unsigned পূর্ণাঙ্ক iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

        যাবত্ (ir != iw) {
            সংরচনা work_queue_task_s * task = &q->queue[ir];

            unsigned পূর্ণাঙ্ক n = task->n_threads;
            unsigned পূর্ণাঙ্ক i = me->id;
            যদি (i < n) {
                task->func(n, i, task->data);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } অন্যথা {
                যাবত্ (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(উপরের_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, শূন্য * data, unsigned পূর্ণাঙ্ক n) {
    যদি (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        প্রত্যাগম false;
    }

    unsigned পূর্ণাঙ্ক ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    unsigned পূর্ণাঙ্ক iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    যদি (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        প্রত্যাগম false;
    }

    সংরচনা work_queue_task_s * task = &q->queue[iw];
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

    যাবত্ (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    প্রত্যাগম true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t স্তূপ_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    size_t size_stacks = স্তূপ_size * n_workers;
    size_t size_q = hex_align_up(পরিমাণ(সংরচনা work_queue_s) + capacity * পরিমাণ(সংরচনা work_queue_task_s), HEX_L2_সারি_SIZE);
    প্রত্যাগম size_stacks + size_q;
}

size_t work_queue_alignof(শূন্য) {
    প্রত্যাগম 4096;
}

work_queue_t work_queue_init(শূন্য * ptr, uint32_t n_threads, uint32_t capacity, uint32_t স্তূপ_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    unsigned বর্ণ * mem_blob = (unsigned বর্ণ *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + স্তূপ_size * n_workers);
    memset(q, 0, পরিমাণ(সংরচনা work_queue_s) + capacity * পরিমাণ(সংরচনা work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    প্রতি (unsigned পূর্ণাঙ্ক i = 0; i < n_workers; i++) {
        q->স্তূপ[i]  = mem_blob; mem_blob += স্তূপ_size;
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
    প্রতি (পূর্ণাঙ্ক i = 0; i < (পূর্ণাঙ্ক) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = রিক্ত;
        q->queue[i].data      = রিক্ত;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    প্রতি (unsigned পূর্ণাঙ্ক i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_স্তূপ_addr(&attr, q->স্তূপ[i]);
        qurt_thread_attr_set_স্তূপ_size(&attr, স্তূপ_size);

        বর্ণ thread_name[32];
        snprintf(thread_name, পরিমাণ(thread_name), "work-queue:%u", i);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        পূর্ণাঙ্ক prio = qurt_thread_get_priority(qurt_thread_get_id());
        যদি (prio < 1) {
            prio = 1;
        }
        যদি (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        পূর্ণাঙ্ক err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (শূন্য *) &q->context[i]);
        যদি (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            প্রত্যাগম রিক্ত;
        }
    }

    প্রত্যাগম q;
}

শূন্য work_queue_free(work_queue_t q) {
    যদি (!q) { প্রত্যাগম; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    প্রতি (unsigned পূর্ণাঙ্ক i = 0; i < q->n_workers; i++) {
        যদি (q->thread[i]) {
            পূর্ণাঙ্ক status;
            (শূন্য) qurt_thread_join(q->thread[i], &status);
        }
    }
}

শূন্য work_queue_wakeup(work_queue_t q) {
    যদি (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

শূন্য work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
