#अन्तर्भाव "work-queue.h"
#अन्तर्भाव "hex-utils.h"

#अन्तर्भाव <qurt.h>
#अन्तर्भाव <qurt_hvx.h>

#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "HAP_farf.h"

#define LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
प्रकारनिर्देश संरचना {
    work_queue_t  queue;
    अचिन्हित पूर्णाङ्क  id;
} worker_context_t;

संरचना work_queue_task_s {
    work_queue_func_t func;
    शून्य *            data;
    अचिन्हित पूर्णाङ्क      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
संरचना work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    शून्य *             ढेर[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    अचिन्हित पूर्णाङ्क       n_threads;                          // total threads (workers + main)
    अचिन्हित पूर्णाङ्क       n_workers;                          // number of active threads (just workers)

    atomic_uint        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    संरचना work_queue_task_s queue[] __attribute__((aligned(HEX_L2_पंक्ति_SIZE)));
};

स्थिर शून्य work_queue_thread(शून्य * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(ऊपरी_2, "work-queue: thread %u started", me->id);

    अचिन्हित पूर्णाङ्क prev_seqn = 0;

    यावत् (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        अचिन्हित पूर्णाङ्क seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        यदि (seqn == prev_seqn) {
            यदि (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } अन्यथा {
                qurt_futex_wait(&q->active, 0);
            }
            अनुवर्त;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        अचिन्हित पूर्णाङ्क ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        अचिन्हित पूर्णाङ्क iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

        यावत् (ir != iw) {
            संरचना work_queue_task_s * task = &q->queue[ir];

            अचिन्हित पूर्णाङ्क n = task->n_threads;
            अचिन्हित पूर्णाङ्क i = me->id;
            यदि (i < n) {
                task->func(n, i, task->data);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } अन्यथा {
                यावत् (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(ऊपरी_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, शून्य * data, अचिन्हित पूर्णाङ्क n) {
    यदि (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        प्रत्यागम false;
    }

    अचिन्हित पूर्णाङ्क ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    अचिन्हित पूर्णाङ्क iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    यदि (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        प्रत्यागम false;
    }

    संरचना work_queue_task_s * task = &q->queue[iw];
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

    यावत् (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    प्रत्यागम true;
}

माप_प्रकार work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t ढेर_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    माप_प्रकार size_stacks = ढेर_size * n_workers;
    माप_प्रकार size_q = hex_align_up(परिमाण(संरचना work_queue_s) + capacity * परिमाण(संरचना work_queue_task_s), HEX_L2_पंक्ति_SIZE);
    प्रत्यागम size_stacks + size_q;
}

माप_प्रकार work_queue_alignof(शून्य) {
    प्रत्यागम 4096;
}

work_queue_t work_queue_init(शून्य * ptr, uint32_t n_threads, uint32_t capacity, uint32_t ढेर_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    अचिन्हित वर्ण * mem_blob = (अचिन्हित वर्ण *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + ढेर_size * n_workers);
    स_रखो(q, 0, परिमाण(संरचना work_queue_s) + capacity * परिमाण(संरचना work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    प्रति (अचिन्हित पूर्णाङ्क i = 0; i < n_workers; i++) {
        q->ढेर[i]  = mem_blob; mem_blob += ढेर_size;
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
    प्रति (पूर्णाङ्क i = 0; i < (पूर्णाङ्क) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = रिक्त;
        q->queue[i].data      = रिक्त;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    प्रति (अचिन्हित पूर्णाङ्क i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_ढेर_addr(&attr, q->ढेर[i]);
        qurt_thread_attr_set_ढेर_size(&attr, ढेर_size);

        वर्ण thread_name[32];
        snprintf(thread_name, परिमाण(thread_name), "work-queue:%u", i);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        पूर्णाङ्क prio = qurt_thread_get_priority(qurt_thread_get_id());
        यदि (prio < 1) {
            prio = 1;
        }
        यदि (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        पूर्णाङ्क err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (शून्य *) &q->context[i]);
        यदि (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            प्रत्यागम रिक्त;
        }
    }

    प्रत्यागम q;
}

शून्य work_queue_free(work_queue_t q) {
    यदि (!q) { प्रत्यागम; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    प्रति (अचिन्हित पूर्णाङ्क i = 0; i < q->n_workers; i++) {
        यदि (q->thread[i]) {
            पूर्णाङ्क status;
            (शून्य) qurt_thread_join(q->thread[i], &status);
        }
    }
}

शून्य work_queue_wakeup(work_queue_t q) {
    यदि (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

शून्य work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
