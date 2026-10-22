#અન્તર્ભાવ "work-queue.h"
#અન્તર્ભાવ "hex-utils.h"

#અન્તર્ભાવ <qurt.h>
#અન્તર્ભાવ <qurt_hvx.h>

#અન્તર્ભાવ <stdatomic.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <string.h>

#અન્તર્ભાવ "HAP_farf.h"

#define LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
પ્રકારવ્યાખ્યા સંરચના {
    work_queue_t  queue;
    unsigned પૂર્ણાંક  id;
} worker_context_t;

સંરચના work_queue_task_s {
    work_queue_func_t func;
    શૂન્ય *            data;
    unsigned પૂર્ણાંક      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
સંરચના work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    શૂન્ય *             ઢગલો[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    unsigned પૂર્ણાંક       n_threads;                          // total threads (workers + main)
    unsigned પૂર્ણાંક       n_workers;                          // number of active threads (just workers)

    atomic_uint        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    સંરચના work_queue_task_s queue[] __attribute__((aligned(HEX_L2_લીટી_SIZE)));
};

સ્થિર શૂન્ય work_queue_thread(શૂન્ય * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(ઉપલું_2, "work-queue: thread %u started", me->id);

    unsigned પૂર્ણાંક prev_seqn = 0;

    જ્યારે (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned પૂર્ણાંક seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        જો (seqn == prev_seqn) {
            જો (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } નહીંતર {
                qurt_futex_wait(&q->active, 0);
            }
            ચાલુ;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        unsigned પૂર્ણાંક ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        unsigned પૂર્ણાંક iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

        જ્યારે (ir != iw) {
            સંરચના work_queue_task_s * task = &q->queue[ir];

            unsigned પૂર્ણાંક n = task->n_threads;
            unsigned પૂર્ણાંક i = me->id;
            જો (i < n) {
                task->func(n, i, task->data);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } નહીંતર {
                જ્યારે (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(ઉપલું_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, શૂન્ય * data, unsigned પૂર્ણાંક n) {
    જો (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        પાછા false;
    }

    unsigned પૂર્ણાંક ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    unsigned પૂર્ણાંક iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    જો (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        પાછા false;
    }

    સંરચના work_queue_task_s * task = &q->queue[iw];
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

    જ્યારે (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    પાછા true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t ઢગલો_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    size_t size_stacks = ઢગલો_size * n_workers;
    size_t size_q = hex_align_up(કદ(સંરચના work_queue_s) + capacity * કદ(સંરચના work_queue_task_s), HEX_L2_લીટી_SIZE);
    પાછા size_stacks + size_q;
}

size_t work_queue_alignof(શૂન્ય) {
    પાછા 4096;
}

work_queue_t work_queue_init(શૂન્ય * ptr, uint32_t n_threads, uint32_t capacity, uint32_t ઢગલો_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    unsigned અક્ષર * mem_blob = (unsigned અક્ષર *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + ઢગલો_size * n_workers);
    memset(q, 0, કદ(સંરચના work_queue_s) + capacity * કદ(સંરચના work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    માટે (unsigned પૂર્ણાંક i = 0; i < n_workers; i++) {
        q->ઢગલો[i]  = mem_blob; mem_blob += ઢગલો_size;
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
    માટે (પૂર્ણાંક i = 0; i < (પૂર્ણાંક) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = ખાલી;
        q->queue[i].data      = ખાલી;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    માટે (unsigned પૂર્ણાંક i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_ઢગલો_addr(&attr, q->ઢગલો[i]);
        qurt_thread_attr_set_ઢગલો_size(&attr, ઢગલો_size);

        અક્ષર thread_name[32];
        snprintf(thread_name, કદ(thread_name), "work-queue:%u", i);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        પૂર્ણાંક prio = qurt_thread_get_priority(qurt_thread_get_id());
        જો (prio < 1) {
            prio = 1;
        }
        જો (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        પૂર્ણાંક err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (શૂન્ય *) &q->context[i]);
        જો (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            પાછા ખાલી;
        }
    }

    પાછા q;
}

શૂન્ય work_queue_free(work_queue_t q) {
    જો (!q) { પાછા; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    માટે (unsigned પૂર્ણાંક i = 0; i < q->n_workers; i++) {
        જો (q->thread[i]) {
            પૂર્ણાંક status;
            (શૂન્ય) qurt_thread_join(q->thread[i], &status);
        }
    }
}

શૂન્ય work_queue_wakeup(work_queue_t q) {
    જો (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

શૂન્ય work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
