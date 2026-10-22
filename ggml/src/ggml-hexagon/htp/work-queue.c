#അന്തര്ഭാവ "work-queue.h"
#അന്തര്ഭാവ "hex-utils.h"

#അന്തര്ഭാവ <qurt.h>
#അന്തര്ഭാവ <qurt_hvx.h>

#അന്തര്ഭാവ <stdatomic.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <string.h>

#അന്തര്ഭാവ "HAP_farf.h"

#define LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
തരനിർവചനം ഘടന {
    work_queue_t  queue;
    unsigned പൂർണ്ണസംഖ്യ  id;
} worker_context_t;

ഘടന work_queue_task_s {
    work_queue_func_t func;
    ശൂന്യം *            data;
    unsigned പൂർണ്ണസംഖ്യ      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
ഘടന work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    ശൂന്യം *             അട്ടി[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    unsigned പൂർണ്ണസംഖ്യ       n_threads;                          // total threads (workers + main)
    unsigned പൂർണ്ണസംഖ്യ       n_workers;                          // number of active threads (just workers)

    atomic_uint        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    ഘടന work_queue_task_s queue[] __attribute__((aligned(HEX_L2_വരി_SIZE)));
};

സ്ഥിതം ശൂന്യം work_queue_thread(ശൂന്യം * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(മുകളിലെ_2, "work-queue: thread %u started", me->id);

    unsigned പൂർണ്ണസംഖ്യ prev_seqn = 0;

    ആകുമ്പോൾ (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned പൂർണ്ണസംഖ്യ seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        എങ്കിൽ (seqn == prev_seqn) {
            എങ്കിൽ (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } അല്ലെങ്കിൽ {
                qurt_futex_wait(&q->active, 0);
            }
            തുടരുക;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        unsigned പൂർണ്ണസംഖ്യ ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        unsigned പൂർണ്ണസംഖ്യ iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

        ആകുമ്പോൾ (ir != iw) {
            ഘടന work_queue_task_s * task = &q->queue[ir];

            unsigned പൂർണ്ണസംഖ്യ n = task->n_threads;
            unsigned പൂർണ്ണസംഖ്യ i = me->id;
            എങ്കിൽ (i < n) {
                task->func(n, i, task->data);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } അല്ലെങ്കിൽ {
                ആകുമ്പോൾ (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(മുകളിലെ_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, ശൂന്യം * data, unsigned പൂർണ്ണസംഖ്യ n) {
    എങ്കിൽ (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        തിരികെ false;
    }

    unsigned പൂർണ്ണസംഖ്യ ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    unsigned പൂർണ്ണസംഖ്യ iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    എങ്കിൽ (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        തിരികെ false;
    }

    ഘടന work_queue_task_s * task = &q->queue[iw];
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

    ആകുമ്പോൾ (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    തിരികെ true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t അട്ടി_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    size_t size_stacks = അട്ടി_size * n_workers;
    size_t size_q = hex_align_up(വലുപ്പം(ഘടന work_queue_s) + capacity * വലുപ്പം(ഘടന work_queue_task_s), HEX_L2_വരി_SIZE);
    തിരികെ size_stacks + size_q;
}

size_t work_queue_alignof(ശൂന്യം) {
    തിരികെ 4096;
}

work_queue_t work_queue_init(ശൂന്യം * ptr, uint32_t n_threads, uint32_t capacity, uint32_t അട്ടി_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    unsigned അക്ഷരം * mem_blob = (unsigned അക്ഷരം *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + അട്ടി_size * n_workers);
    memset(q, 0, വലുപ്പം(ഘടന work_queue_s) + capacity * വലുപ്പം(ഘടന work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i = 0; i < n_workers; i++) {
        q->അട്ടി[i]  = mem_blob; mem_blob += അട്ടി_size;
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < (പൂർണ്ണസംഖ്യ) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = ശൂന്യ;
        q->queue[i].data      = ശൂന്യ;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_അട്ടി_addr(&attr, q->അട്ടി[i]);
        qurt_thread_attr_set_അട്ടി_size(&attr, അട്ടി_size);

        അക്ഷരം thread_name[32];
        snprintf(thread_name, വലുപ്പം(thread_name), "work-queue:%u", i);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        പൂർണ്ണസംഖ്യ prio = qurt_thread_get_priority(qurt_thread_get_id());
        എങ്കിൽ (prio < 1) {
            prio = 1;
        }
        എങ്കിൽ (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        പൂർണ്ണസംഖ്യ err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (ശൂന്യം *) &q->context[i]);
        എങ്കിൽ (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            തിരികെ ശൂന്യ;
        }
    }

    തിരികെ q;
}

ശൂന്യം work_queue_free(work_queue_t q) {
    എങ്കിൽ (!q) { തിരികെ; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i = 0; i < q->n_workers; i++) {
        എങ്കിൽ (q->thread[i]) {
            പൂർണ്ണസംഖ്യ status;
            (ശൂന്യം) qurt_thread_join(q->thread[i], &status);
        }
    }
}

ശൂന്യം work_queue_wakeup(work_queue_t q) {
    എങ്കിൽ (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

ശൂന്യം work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
