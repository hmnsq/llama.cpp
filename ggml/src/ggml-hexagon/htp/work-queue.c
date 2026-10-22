#समावेश "work-queue.h"
#समावेश "hex-utils.h"

#समावेश <qurt.h>
#समावेश <qurt_hvx.h>

#समावेश <stdatomic.h>
#समावेश <stdint.h>
#समावेश <stdio.h>
#समावेश <stdlib.h>
#समावेश <string.h>

#समावेश "HAP_farf.h"

#घोषणा LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
प्रकार काष्ठा {
    work_queue_t  queue;
    अचिन्हित पूर्णांक  id;
} worker_context_t;

काष्ठा work_queue_task_s {
    work_queue_func_t func;
    व्योम *            तथ्य;
    अचिन्हित पूर्णांक      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
काष्ठा work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    व्योम *             ढेर_2[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    अचिन्हित पूर्णांक       n_threads;                          // total threads (workers + main)
    अचिन्हित पूर्णांक       n_workers;                          // number of active threads (just workers)

    atomic_uint        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    काष्ठा work_queue_task_s queue[] __attribute__((aligned(HEX_L2_पंक्ति_आकार)));
};

अटल व्योम work_queue_thread(व्योम * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(ऊपरी_2, "work-queue: thread %u started", me->id);

    अचिन्हित पूर्णांक पूर्व_seqn = 0;

    जबतक (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        अचिन्हित पूर्णांक seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        अगर (seqn == पूर्व_seqn) {
            अगर (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } अन्यथा {
                qurt_futex_wait(&q->active, 0);
            }
            जारी;
        }

        पूर्व_seqn = seqn;

        // Process all active tasks in the queue
        अचिन्हित पूर्णांक ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        अचिन्हित पूर्णांक iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

        जबतक (ir != iw) {
            काष्ठा work_queue_task_s * task = &q->queue[ir];

            अचिन्हित पूर्णांक n = task->n_threads;
            अचिन्हित पूर्णांक i = me->id;
            अगर (i < n) {
                task->func(n, i, task->तथ्य);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } अन्यथा {
                जबतक (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(ऊपरी_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, व्योम * तथ्य, अचिन्हित पूर्णांक n) {
    अगर (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        वापस false;
    }

    अचिन्हित पूर्णांक ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    अचिन्हित पूर्णांक iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    अगर (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        वापस false;
    }

    काष्ठा work_queue_task_s * task = &q->queue[iw];
    task->func      = func;
    task->तथ्य      = तथ्य;
    task->n_threads = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_mask, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, तथ्य);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    जबतक (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    वापस true;
}

माप_प्रकार work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t ढेर_आकार) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    माप_प्रकार आकार_stacks = ढेर_आकार * n_workers;
    माप_प्रकार आकार_q = hex_align_up(माप(काष्ठा work_queue_s) + capacity * माप(काष्ठा work_queue_task_s), HEX_L2_पंक्ति_आकार);
    वापस आकार_stacks + आकार_q;
}

माप_प्रकार work_queue_alignof(व्योम) {
    वापस 4096;
}

work_queue_t work_queue_init(व्योम * ptr, uint32_t n_threads, uint32_t capacity, uint32_t ढेर_आकार) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    अचिन्हित अक्षर * mem_blob = (अचिन्हित अक्षर *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + ढेर_आकार * n_workers);
    स_रखो(q, 0, माप(काष्ठा work_queue_s) + capacity * माप(काष्ठा work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    क्रम (अचिन्हित पूर्णांक i = 0; i < n_workers; i++) {
        q->ढेर_2[i]  = mem_blob; mem_blob += ढेर_आकार;
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
    क्रम (पूर्णांक i = 0; i < (पूर्णांक) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = शून्य;
        q->queue[i].तथ्य      = शून्य;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    क्रम (अचिन्हित पूर्णांक i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_ढेर_addr(&attr, q->ढेर_2[i]);
        qurt_thread_attr_set_ढेर_आकार(&attr, ढेर_आकार);

        अक्षर thread_name[32];
        snprintf(thread_name, माप(thread_name), "work-queue:%u", i);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        पूर्णांक prio = qurt_thread_get_priority(qurt_thread_get_id());
        अगर (prio < 1) {
            prio = 1;
        }
        अगर (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        पूर्णांक err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (व्योम *) &q->context[i]);
        अगर (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            वापस शून्य;
        }
    }

    वापस q;
}

व्योम work_queue_free(work_queue_t q) {
    अगर (!q) { वापस; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    क्रम (अचिन्हित पूर्णांक i = 0; i < q->n_workers; i++) {
        अगर (q->thread[i]) {
            पूर्णांक status;
            (व्योम) qurt_thread_join(q->thread[i], &status);
        }
    }
}

व्योम work_queue_wakeup(work_queue_t q) {
    अगर (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

व्योम work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
