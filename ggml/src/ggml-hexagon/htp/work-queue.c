#ଅନ୍ତର୍ଭାଵ "work-queue.h"
#ଅନ୍ତର୍ଭାଵ "hex-utils.h"

#ଅନ୍ତର୍ଭାଵ <qurt.h>
#ଅନ୍ତର୍ଭାଵ <qurt_hvx.h>

#ଅନ୍ତର୍ଭାଵ <stdatomic.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

#ଅନ୍ତର୍ଭାଵ "HAP_farf.h"

#define LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    work_queue_t  queue;
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା  id;
} worker_context_t;

ସଂରଚନା work_queue_task_s {
    work_queue_func_t func;
    ଶୂନ୍ୟ *            data;
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
ସଂରଚନା work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    ଶୂନ୍ୟ *             ଗଦା[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା       n_threads;                          // total threads (workers + main)
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା       n_workers;                          // number of active threads (just workers)

    atomic_uint        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    ସଂରଚନା work_queue_task_s queue[] __attribute__((aligned(HEX_L2_ଧାଡ଼ି_SIZE)));
};

ସ୍ଥିର ଶୂନ୍ୟ work_queue_thread(ଶୂନ୍ୟ * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(ଉପର_2, "work-queue: thread %u started", me->id);

    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା prev_seqn = 0;

    ଯେତେବେଳେ (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        ଯଦି (seqn == prev_seqn) {
            ଯଦି (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } ଅନ୍ୟଥା {
                qurt_futex_wait(&q->active, 0);
            }
            ଜାରି;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

        ଯେତେବେଳେ (ir != iw) {
            ସଂରଚନା work_queue_task_s * task = &q->queue[ir];

            unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = task->n_threads;
            unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = me->id;
            ଯଦି (i < n) {
                task->func(n, i, task->data);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } ଅନ୍ୟଥା {
                ଯେତେବେଳେ (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(ଉପର_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, ଶୂନ୍ୟ * data, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n) {
    ଯଦି (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        ଫେରନ୍ତୁ false;
    }

    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    ଯଦି (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        ଫେରନ୍ତୁ false;
    }

    ସଂରଚନା work_queue_task_s * task = &q->queue[iw];
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

    ଯେତେବେଳେ (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    ଫେରନ୍ତୁ true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t ଗଦା_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    size_t size_stacks = ଗଦା_size * n_workers;
    size_t size_q = hex_align_up(ଆକାର(ସଂରଚନା work_queue_s) + capacity * ଆକାର(ସଂରଚନା work_queue_task_s), HEX_L2_ଧାଡ଼ି_SIZE);
    ଫେରନ୍ତୁ size_stacks + size_q;
}

size_t work_queue_alignof(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ 4096;
}

work_queue_t work_queue_init(ଶୂନ୍ୟ * ptr, uint32_t n_threads, uint32_t capacity, uint32_t ଗଦା_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    unsigned ଅକ୍ଷର * mem_blob = (unsigned ଅକ୍ଷର *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + ଗଦା_size * n_workers);
    memset(q, 0, ଆକାର(ସଂରଚନା work_queue_s) + capacity * ଆକାର(ସଂରଚନା work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_workers; i++) {
        q->ଗଦା[i]  = mem_blob; mem_blob += ଗଦା_size;
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
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = ଖାଲି;
        q->queue[i].data      = ଖାଲି;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_ଗଦା_addr(&attr, q->ଗଦା[i]);
        qurt_thread_attr_set_ଗଦା_size(&attr, ଗଦା_size);

        ଅକ୍ଷର thread_name[32];
        snprintf(thread_name, ଆକାର(thread_name), "work-queue:%u", i);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା prio = qurt_thread_get_priority(qurt_thread_get_id());
        ଯଦି (prio < 1) {
            prio = 1;
        }
        ଯଦି (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (ଶୂନ୍ୟ *) &q->context[i]);
        ଯଦି (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            ଫେରନ୍ତୁ ଖାଲି;
        }
    }

    ଫେରନ୍ତୁ q;
}

ଶୂନ୍ୟ work_queue_free(work_queue_t q) {
    ଯଦି (!q) { ଫେରନ୍ତୁ; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < q->n_workers; i++) {
        ଯଦି (q->thread[i]) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା status;
            (ଶୂନ୍ୟ) qurt_thread_join(q->thread[i], &status);
        }
    }
}

ଶୂନ୍ୟ work_queue_wakeup(work_queue_t q) {
    ଯଦି (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

ଶୂନ୍ୟ work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
