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

#परिभाषय निम्नतमम्_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
प्रकारनिर्देश संरचना {
    work_queue_t  queue;
    अचिह्नित पूर्णाङ्क  id;
} कार्यकर्ता_context_t;

संरचना work_queue_task_s {
    work_queue_func_t func;
    शून्य *            दत्त;
    अचिह्नित पूर्णाङ्क      n_तन्तवः;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
संरचना work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_आवरणम्;
    uint32_t           क्षमता;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_तन्तवः];   // thread ID's of the workers
    कार्यकर्ता_context_t   context[WORK_QUEUE_MAX_N_तन्तवः];  // worker contexts
    शून्य *             राशिः[WORK_QUEUE_MAX_N_तन्तवः];    // thread stack pointers
    अचिह्नित पूर्णाङ्क       n_तन्तवः;                          // total threads (workers + main)
    अचिह्नित पूर्णाङ्क       n_workers;                          // number of active threads (just workers)

    atomic_uint        सक्रिय;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    संरचना work_queue_task_s queue[] __attribute__((aligned(HEX_L2_पङ्क्तिः_आकार)));
};

स्थिर शून्य work_queue_thread(शून्य * context) {
    कार्यकर्ता_context_t * me = (कार्यकर्ता_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(उच्चः_2, "work-queue: thread %u started", me->id);

    अचिह्नित पूर्णाङ्क पूर्व_seqn = 0;

    यावत् (!atomic_आरोपय_explicit(&q->killed, memory_order_relaxed)) {
        अचिह्नित पूर्णाङ्क seqn = atomic_आरोपय_explicit(&q->seqn, memory_order_acquire);
        यदि (seqn == पूर्व_seqn) {
            यदि (atomic_आरोपय_explicit(&q->सक्रिय, memory_order_relaxed)) {
                hex_pause();
            } अन्यथा {
                qurt_futex_wait(&q->सक्रिय, 0);
            }
            अनुवर्त;
        }

        पूर्व_seqn = seqn;

        // Process all active tasks in the queue
        अचिह्नित पूर्णाङ्क ir = atomic_आरोपय_explicit(&q->idx_read, memory_order_relaxed);
        अचिह्नित पूर्णाङ्क iw = atomic_आरोपय_explicit(&q->idx_write, memory_order_relaxed);

        यावत् (ir != iw) {
            संरचना work_queue_task_s * task = &q->queue[ir];

            अचिह्नित पूर्णाङ्क n = task->n_तन्तवः;
            अचिह्नित पूर्णाङ्क i = me->id;
            यदि (i < n) {
                task->func(n, i, task->दत्त);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } अन्यथा {
                यावत् (atomic_आरोपय_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_आवरणम्;
        }
    }

    FARF(उच्चः_2, "work-queue: thread %u stopped", me->id);
}

bool work_queue_सततम्_async(work_queue_t q, work_queue_func_t func, शून्य * दत्त, अचिह्नित पूर्णाङ्क n) {
    यदि (n > q->n_तन्तवः) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_तन्तवः);
        प्रत्यागम false;
    }

    अचिह्नित पूर्णाङ्क ir = atomic_आरोपय_explicit(&q->idx_read,  memory_order_relaxed);
    अचिह्नित पूर्णाङ्क iw = atomic_आरोपय_explicit(&q->idx_write, memory_order_relaxed);

    यदि (((iw + 1) & q->idx_आवरणम्) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        प्रत्यागम false;
    }

    संरचना work_queue_task_s * task = &q->queue[iw];
    task->func      = func;
    task->दत्त      = दत्त;
    task->n_तन्तवः = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_आवरणम्, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_सङ्कलय_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, दत्त);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    यावत् (atomic_आरोपय_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_आवरणम्, memory_order_relaxed);

    प्रत्यागम true;
}

size_t work_queue_sizeof(uint32_t n_तन्तवः, uint32_t क्षमता, uint32_t राशिः_आकार) {
    क्षमता = hex_ceil_pow2(क्षमता);
    uint32_t n_workers = n_तन्तवः > 1 ? n_तन्तवः - 1 : 0;
    size_t आकार_stacks = राशिः_आकार * n_workers;
    size_t आकार_q = hex_align_ऊर्ध्वम्(परिमाण(संरचना work_queue_s) + क्षमता * परिमाण(संरचना work_queue_task_s), HEX_L2_पङ्क्तिः_आकार);
    प्रत्यागम आकार_stacks + आकार_q;
}

size_t work_queue_alignof(शून्य) {
    प्रत्यागम 4096;
}

work_queue_t work_queue_init(शून्य * ptr, uint32_t n_तन्तवः, uint32_t क्षमता, uint32_t राशिः_आकार) {
    क्षमता = hex_ceil_pow2(क्षमता);
    uint32_t n_workers = n_तन्तवः > 1 ? n_तन्तवः - 1 : 0;
    अचिह्नित वर्ण * mem_blob = (अचिह्नित वर्ण *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + राशिः_आकार * n_workers);
    स्मृतिं_पूरय(q, 0, परिमाण(संरचना work_queue_s) + क्षमता * परिमाण(संरचना work_queue_task_s));

    q->n_तन्तवः = n_तन्तवः;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->क्षमता = क्षमता;

    प्रति (अचिह्नित पूर्णाङ्क i = 0; i < n_workers; i++) {
        q->राशिः[i]  = mem_blob; mem_blob += राशिः_आकार;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].queue = q;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->सक्रिय,    0);
    q->idx_आवरणम्  = क्षमता - 1;
    q->killed    = 0;
    प्रति (पूर्णाङ्क i = 0; i < (पूर्णाङ्क) क्षमता; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = रिक्त;
        q->queue[i].दत्त      = रिक्त;
        q->queue[i].n_तन्तवः = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    प्रति (अचिह्नित पूर्णाङ्क i = 0; i < n_workers; i++) {
        qurt_thread_attr_समुच्चयः_राशिः_addr(&attr, q->राशिः[i]);
        qurt_thread_attr_समुच्चयः_राशिः_आकार(&attr, राशिः_आकार);

        वर्ण thread_name[32];
        snprintf(thread_name, परिमाण(thread_name), "work-queue:%u", i);
        qurt_thread_attr_समुच्चयः_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        पूर्णाङ्क prio = qurt_thread_get_प्राथमिकता(qurt_thread_get_id());
        यदि (prio < 1) {
            prio = 1;
        }
        यदि (prio > निम्नतमम्_USABLE_QURT_PRIO) {
            prio = निम्नतमम्_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_समुच्चयः_प्राथमिकता(&attr, prio);

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
    atomic_store_explicit(&q->सक्रिय,   1, memory_order_release);
    atomic_fetch_सङ्कलय_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->सक्रिय, q->n_workers);

    प्रति (अचिह्नित पूर्णाङ्क i = 0; i < q->n_workers; i++) {
        यदि (q->thread[i]) {
            पूर्णाङ्क स्थितिः;
            (शून्य) qurt_thread_join(q->thread[i], &स्थितिः);
        }
    }
}

शून्य work_queue_wakeup(work_queue_t q) {
    यदि (!atomic_आरोपय_explicit(&q->सक्रिय, memory_order_relaxed)) {
        atomic_store_explicit(&q->सक्रिय, 1, memory_order_release);
        qurt_futex_wake(&q->सक्रिय, q->n_workers);
    }
}

शून्य work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->सक्रिय, 0, memory_order_release);
}
