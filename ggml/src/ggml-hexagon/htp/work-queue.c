#అన్తర్భావ "work-queue.h"
#అన్తర్భావ "hex-utils.h"

#అన్తర్భావ <qurt.h>
#అన్తర్భావ <qurt_hvx.h>

#అన్తర్భావ <stdatomic.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "HAP_farf.h"

#నిర్వచించు అత్యల్ప_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
రకనిర్వచనం నిర్మాణం {
    work_వరుస_t  వరుస;
    గుర్తులేని పూర్ణాంకం  id;
} కార్మికుడు_context_t;

నిర్మాణం work_వరుస_task_s {
    work_వరుస_func_t func;
    శూన్య *            దత్తాంశం;
    గుర్తులేని పూర్ణాంకం      n_దారాలు;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
నిర్మాణం work_వరుస_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_ముసుగు;
    uint32_t           సామర్థ్యం;

    qurt_thread_t      thread[WORK_వరుస_MAX_N_దారాలు];   // thread ID's of the workers
    కార్మికుడు_context_t   context[WORK_వరుస_MAX_N_దారాలు];  // worker contexts
    శూన్య *             దొంతర[WORK_వరుస_MAX_N_దారాలు];    // thread stack pointers
    గుర్తులేని పూర్ణాంకం       n_దారాలు;                          // total threads (workers + main)
    గుర్తులేని పూర్ణాంకం       n_workers;                          // number of active threads (just workers)

    atomic_uint        చురుకైన;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    నిర్మాణం work_వరుస_task_s వరుస[] __attribute__((aligned(HEX_L2_పంక్తి_కొలత)));
};

స్థిర శూన్య work_వరుస_thread(శూన్య * context) {
    కార్మికుడు_context_t * me = (కార్మికుడు_context_t *) context;
    work_వరుస_t       q  = me->వరుస;

    FARF(ఎగువ_2, "work-queue: thread %u started", me->id);

    గుర్తులేని పూర్ణాంకం prev_seqn = 0;

    అయితే (!atomic_ఎక్కించు_explicit(&q->killed, memory_order_relaxed)) {
        గుర్తులేని పూర్ణాంకం seqn = atomic_ఎక్కించు_explicit(&q->seqn, memory_order_acquire);
        ఒకవేళ (seqn == prev_seqn) {
            ఒకవేళ (atomic_ఎక్కించు_explicit(&q->చురుకైన, memory_order_relaxed)) {
                hex_pause();
            } లేకపోతే {
                qurt_futex_wait(&q->చురుకైన, 0);
            }
            కొనసాగు;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        గుర్తులేని పూర్ణాంకం ir = atomic_ఎక్కించు_explicit(&q->idx_read, memory_order_relaxed);
        గుర్తులేని పూర్ణాంకం iw = atomic_ఎక్కించు_explicit(&q->idx_write, memory_order_relaxed);

        అయితే (ir != iw) {
            నిర్మాణం work_వరుస_task_s * task = &q->వరుస[ir];

            గుర్తులేని పూర్ణాంకం n = task->n_దారాలు;
            గుర్తులేని పూర్ణాంకం i = me->id;
            ఒకవేళ (i < n) {
                task->func(n, i, task->దత్తాంశం);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } లేకపోతే {
                అయితే (atomic_ఎక్కించు_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_ముసుగు;
        }
    }

    FARF(ఎగువ_2, "work-queue: thread %u stopped", me->id);
}

bool work_వరుస_నిరంతరం_async(work_వరుస_t q, work_వరుస_func_t func, శూన్య * దత్తాంశం, గుర్తులేని పూర్ణాంకం n) {
    ఒకవేళ (n > q->n_దారాలు) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_దారాలు);
        తిరిగి false;
    }

    గుర్తులేని పూర్ణాంకం ir = atomic_ఎక్కించు_explicit(&q->idx_read,  memory_order_relaxed);
    గుర్తులేని పూర్ణాంకం iw = atomic_ఎక్కించు_explicit(&q->idx_write, memory_order_relaxed);

    ఒకవేళ (((iw + 1) & q->idx_ముసుగు) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        తిరిగి false;
    }

    నిర్మాణం work_వరుస_task_s * task = &q->వరుస[iw];
    task->func      = func;
    task->దత్తాంశం      = దత్తాంశం;
    task->n_దారాలు = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_ముసుగు, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_కూడు_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, దత్తాంశం);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    అయితే (atomic_ఎక్కించు_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_ముసుగు, memory_order_relaxed);

    తిరిగి true;
}

size_t work_వరుస_sizeof(uint32_t n_దారాలు, uint32_t సామర్థ్యం, uint32_t దొంతర_కొలత) {
    సామర్థ్యం = hex_ceil_pow2(సామర్థ్యం);
    uint32_t n_workers = n_దారాలు > 1 ? n_దారాలు - 1 : 0;
    size_t కొలత_stacks = దొంతర_కొలత * n_workers;
    size_t కొలత_q = hex_align_పైకి(పరిమాణం(నిర్మాణం work_వరుస_s) + సామర్థ్యం * పరిమాణం(నిర్మాణం work_వరుస_task_s), HEX_L2_పంక్తి_కొలత);
    తిరిగి కొలత_stacks + కొలత_q;
}

size_t work_వరుస_alignof(శూన్య) {
    తిరిగి 4096;
}

work_వరుస_t work_వరుస_init(శూన్య * ptr, uint32_t n_దారాలు, uint32_t సామర్థ్యం, uint32_t దొంతర_కొలత) {
    సామర్థ్యం = hex_ceil_pow2(సామర్థ్యం);
    uint32_t n_workers = n_దారాలు > 1 ? n_దారాలు - 1 : 0;
    గుర్తులేని అక్షరం * mem_blob = (గుర్తులేని అక్షరం *) ptr;

    work_వరుస_t q = (work_వరుస_t) (mem_blob + దొంతర_కొలత * n_workers);
    memset(q, 0, పరిమాణం(నిర్మాణం work_వరుస_s) + సామర్థ్యం * పరిమాణం(నిర్మాణం work_వరుస_task_s));

    q->n_దారాలు = n_దారాలు;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->సామర్థ్యం = సామర్థ్యం;

    కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < n_workers; i++) {
        q->దొంతర[i]  = mem_blob; mem_blob += దొంతర_కొలత;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].వరుస = q;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->చురుకైన,    0);
    q->idx_ముసుగు  = సామర్థ్యం - 1;
    q->killed    = 0;
    కొరకు (పూర్ణాంకం i = 0; i < (పూర్ణాంకం) సామర్థ్యం; i++) {
        atomic_init(&q->వరుస[i].barrier, 0);
        q->వరుస[i].func      = ఖాళీ;
        q->వరుస[i].దత్తాంశం      = ఖాళీ;
        q->వరుస[i].n_దారాలు = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < n_workers; i++) {
        qurt_thread_attr_సమితి_దొంతర_addr(&attr, q->దొంతర[i]);
        qurt_thread_attr_సమితి_దొంతర_కొలత(&attr, దొంతర_కొలత);

        అక్షరం thread_name[32];
        snprintf(thread_name, పరిమాణం(thread_name), "work-queue:%u", i);
        qurt_thread_attr_సమితి_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        పూర్ణాంకం prio = qurt_thread_get_ప్రాధాన్యత(qurt_thread_get_id());
        ఒకవేళ (prio < 1) {
            prio = 1;
        }
        ఒకవేళ (prio > అత్యల్ప_USABLE_QURT_PRIO) {
            prio = అత్యల్ప_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_సమితి_ప్రాధాన్యత(&attr, prio);

        పూర్ణాంకం err = qurt_thread_create(&q->thread[i], &attr, work_వరుస_thread, (శూన్య *) &q->context[i]);
        ఒకవేళ (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_వరుస_free(q);
            తిరిగి ఖాళీ;
        }
    }

    తిరిగి q;
}

శూన్య work_వరుస_free(work_వరుస_t q) {
    ఒకవేళ (!q) { తిరిగి; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->చురుకైన,   1, memory_order_release);
    atomic_fetch_కూడు_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->చురుకైన, q->n_workers);

    కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < q->n_workers; i++) {
        ఒకవేళ (q->thread[i]) {
            పూర్ణాంకం స్థితి_2;
            (శూన్య) qurt_thread_join(q->thread[i], &స్థితి_2);
        }
    }
}

శూన్య work_వరుస_wakeup(work_వరుస_t q) {
    ఒకవేళ (!atomic_ఎక్కించు_explicit(&q->చురుకైన, memory_order_relaxed)) {
        atomic_store_explicit(&q->చురుకైన, 1, memory_order_release);
        qurt_futex_wake(&q->చురుకైన, q->n_workers);
    }
}

శూన్య work_వరుస_suspend(work_వరుస_t q) {
    atomic_store_explicit(&q->చురుకైన, 0, memory_order_release);
}
