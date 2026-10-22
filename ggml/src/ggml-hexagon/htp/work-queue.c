#ضمن "work-queue.h"
#ضمن "hex-utils.h"

#ضمن <qurt.h>
#ضمن <qurt_hvx.h>

#ضمن <stdatomic.h>
#ضمن <stdint.h>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#ضمن "HAP_farf.h"

#define LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
تعريفنوع بنية {
    work_queue_t  queue;
    unsigned صحيح  id;
} worker_context_t;

بنية work_queue_task_s {
    work_queue_func_t func;
    فراغ *            data;
    unsigned صحيح      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
بنية work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    فراغ *             کەڵەکە[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    unsigned صحيح       n_threads;                          // total threads (workers + main)
    unsigned صحيح       n_workers;                          // number of active threads (just workers)

    atomic_uint        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    بنية work_queue_task_s queue[] __attribute__((aligned(HEX_L2_هێڵ_SIZE)));
};

ساكن فراغ work_queue_thread(فراغ * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(بەرز_2, "work-queue: thread %u started", me->id);

    unsigned صحيح prev_seqn = 0;

    طالما (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned صحيح seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        إذا (seqn == prev_seqn) {
            إذا (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } وإلا {
                qurt_futex_wait(&q->active, 0);
            }
            استمر;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        unsigned صحيح ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        unsigned صحيح iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

        طالما (ir != iw) {
            بنية work_queue_task_s * task = &q->queue[ir];

            unsigned صحيح n = task->n_threads;
            unsigned صحيح i = me->id;
            إذا (i < n) {
                task->func(n, i, task->data);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } وإلا {
                طالما (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(بەرز_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, فراغ * data, unsigned صحيح n) {
    إذا (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        أرجع false;
    }

    unsigned صحيح ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    unsigned صحيح iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    إذا (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        أرجع false;
    }

    بنية work_queue_task_s * task = &q->queue[iw];
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

    طالما (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    أرجع true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t کەڵەکە_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    size_t size_stacks = کەڵەکە_size * n_workers;
    size_t size_q = hex_align_up(حجم(بنية work_queue_s) + capacity * حجم(بنية work_queue_task_s), HEX_L2_هێڵ_SIZE);
    أرجع size_stacks + size_q;
}

size_t work_queue_alignof(فراغ) {
    أرجع 4096;
}

work_queue_t work_queue_init(فراغ * ptr, uint32_t n_threads, uint32_t capacity, uint32_t کەڵەکە_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    unsigned حرف * mem_blob = (unsigned حرف *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + کەڵەکە_size * n_workers);
    memset(q, 0, حجم(بنية work_queue_s) + capacity * حجم(بنية work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    لكل (unsigned صحيح i = 0; i < n_workers; i++) {
        q->کەڵەکە[i]  = mem_blob; mem_blob += کەڵەکە_size;
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
    لكل (صحيح i = 0; i < (صحيح) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = عدم;
        q->queue[i].data      = عدم;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    لكل (unsigned صحيح i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_کەڵەکە_addr(&attr, q->کەڵەکە[i]);
        qurt_thread_attr_set_کەڵەکە_size(&attr, کەڵەکە_size);

        حرف thread_name[32];
        snprintf(thread_name, حجم(thread_name), "work-queue:%u", i);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        صحيح prio = qurt_thread_get_priority(qurt_thread_get_id());
        إذا (prio < 1) {
            prio = 1;
        }
        إذا (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        صحيح err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (فراغ *) &q->context[i]);
        إذا (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            أرجع عدم;
        }
    }

    أرجع q;
}

فراغ work_queue_free(work_queue_t q) {
    إذا (!q) { أرجع; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    لكل (unsigned صحيح i = 0; i < q->n_workers; i++) {
        إذا (q->thread[i]) {
            صحيح status;
            (فراغ) qurt_thread_join(q->thread[i], &status);
        }
    }
}

فراغ work_queue_wakeup(work_queue_t q) {
    إذا (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

فراغ work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
