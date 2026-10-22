#அந்தர்பாவ "work-queue.h"
#அந்தர்பாவ "hex-utils.h"

#அந்தர்பாவ <qurt.h>
#அந்தர்பாவ <qurt_hvx.h>

#அந்தர்பாவ <stdatomic.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <string.h>

#அந்தர்பாவ "HAP_farf.h"

#define LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
வகைவரை கட்டமைப்பு {
    work_queue_t  queue;
    unsigned முழு  id;
} worker_context_t;

கட்டமைப்பு work_queue_task_s {
    work_queue_func_t func;
    வெறுமை *            data;
    unsigned முழு      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
கட்டமைப்பு work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    வெறுமை *             அடுக்கு[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    unsigned முழு       n_threads;                          // total threads (workers + main)
    unsigned முழு       n_workers;                          // number of active threads (just workers)

    atomic_uint        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    கட்டமைப்பு work_queue_task_s queue[] __attribute__((aligned(HEX_L2_வரி_SIZE)));
};

நிலையான வெறுமை work_queue_thread(வெறுமை * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(உயர்வு_2, "work-queue: thread %u started", me->id);

    unsigned முழு prev_seqn = 0;

    போது (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned முழு seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        எனில் (seqn == prev_seqn) {
            எனில் (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } இல்லையெனில் {
                qurt_futex_wait(&q->active, 0);
            }
            தொடர்;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        unsigned முழு ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        unsigned முழு iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

        போது (ir != iw) {
            கட்டமைப்பு work_queue_task_s * task = &q->queue[ir];

            unsigned முழு n = task->n_threads;
            unsigned முழு i = me->id;
            எனில் (i < n) {
                task->func(n, i, task->data);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } இல்லையெனில் {
                போது (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(உயர்வு_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, வெறுமை * data, unsigned முழு n) {
    எனில் (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        திரும்பு false;
    }

    unsigned முழு ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    unsigned முழு iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    எனில் (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        திரும்பு false;
    }

    கட்டமைப்பு work_queue_task_s * task = &q->queue[iw];
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

    போது (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    திரும்பு true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t அடுக்கு_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    size_t size_stacks = அடுக்கு_size * n_workers;
    size_t size_q = hex_align_up(அளவு(கட்டமைப்பு work_queue_s) + capacity * அளவு(கட்டமைப்பு work_queue_task_s), HEX_L2_வரி_SIZE);
    திரும்பு size_stacks + size_q;
}

size_t work_queue_alignof(வெறுமை) {
    திரும்பு 4096;
}

work_queue_t work_queue_init(வெறுமை * ptr, uint32_t n_threads, uint32_t capacity, uint32_t அடுக்கு_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    unsigned எழுத்து * mem_blob = (unsigned எழுத்து *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + அடுக்கு_size * n_workers);
    memset(q, 0, அளவு(கட்டமைப்பு work_queue_s) + capacity * அளவு(கட்டமைப்பு work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    க்கு (unsigned முழு i = 0; i < n_workers; i++) {
        q->அடுக்கு[i]  = mem_blob; mem_blob += அடுக்கு_size;
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
    க்கு (முழு i = 0; i < (முழு) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = வெற்று;
        q->queue[i].data      = வெற்று;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    க்கு (unsigned முழு i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_அடுக்கு_addr(&attr, q->அடுக்கு[i]);
        qurt_thread_attr_set_அடுக்கு_size(&attr, அடுக்கு_size);

        எழுத்து thread_name[32];
        snprintf(thread_name, அளவு(thread_name), "work-queue:%u", i);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        முழு prio = qurt_thread_get_priority(qurt_thread_get_id());
        எனில் (prio < 1) {
            prio = 1;
        }
        எனில் (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        முழு err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (வெறுமை *) &q->context[i]);
        எனில் (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            திரும்பு வெற்று;
        }
    }

    திரும்பு q;
}

வெறுமை work_queue_free(work_queue_t q) {
    எனில் (!q) { திரும்பு; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    க்கு (unsigned முழு i = 0; i < q->n_workers; i++) {
        எனில் (q->thread[i]) {
            முழு status;
            (வெறுமை) qurt_thread_join(q->thread[i], &status);
        }
    }
}

வெறுமை work_queue_wakeup(work_queue_t q) {
    எனில் (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

வெறுமை work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
