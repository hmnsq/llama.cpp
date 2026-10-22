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

#وصف_ڪريو سڀ_کان_هيٺ_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
تعريفنوع بنية {
    work_قطار_t  قطار;
    بنانشان صحيح  id;
} ڪم_ڪندڙ_context_t;

بنية work_قطار_task_s {
    work_قطار_func_t func;
    فراغ *            معلومات;
    بنانشان صحيح      n_ڌاڳا;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
بنية work_قطار_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_نقاب;
    uint32_t           صلاحيت;

    qurt_thread_t      thread[WORK_قطار_MAX_N_ڌاڳا];   // thread ID's of the workers
    ڪم_ڪندڙ_context_t   context[WORK_قطار_MAX_N_ڌاڳا];  // worker contexts
    فراغ *             ڍير[WORK_قطار_MAX_N_ڌاڳا];    // thread stack pointers
    بنانشان صحيح       n_ڌاڳا;                          // total threads (workers + main)
    بنانشان صحيح       n_workers;                          // number of active threads (just workers)

    atomic_uint        سرگرم;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    بنية work_قطار_task_s قطار[] __attribute__((aligned(HEX_L2_سٽ_سائيز)));
};

ساكن فراغ work_قطار_thread(فراغ * context) {
    ڪم_ڪندڙ_context_t * me = (ڪم_ڪندڙ_context_t *) context;
    work_قطار_t       q  = me->قطار;

    FARF(مٿاهون_2, "work-queue: thread %u started", me->id);

    بنانشان صحيح prev_seqn = 0;

    طالما (!atomic_ڀريو_explicit(&q->killed, memory_order_relaxed)) {
        بنانشان صحيح seqn = atomic_ڀريو_explicit(&q->seqn, memory_order_acquire);
        إذا (seqn == prev_seqn) {
            إذا (atomic_ڀريو_explicit(&q->سرگرم, memory_order_relaxed)) {
                hex_pause();
            } وإلا {
                qurt_futex_wait(&q->سرگرم, 0);
            }
            استمر;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        بنانشان صحيح ir = atomic_ڀريو_explicit(&q->idx_read, memory_order_relaxed);
        بنانشان صحيح iw = atomic_ڀريو_explicit(&q->idx_write, memory_order_relaxed);

        طالما (ir != iw) {
            بنية work_قطار_task_s * task = &q->قطار[ir];

            بنانشان صحيح n = task->n_ڌاڳا;
            بنانشان صحيح i = me->id;
            إذا (i < n) {
                task->func(n, i, task->معلومات);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } وإلا {
                طالما (atomic_ڀريو_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_نقاب;
        }
    }

    FARF(مٿاهون_2, "work-queue: thread %u stopped", me->id);
}

bool work_قطار_لاڳيتو_async(work_قطار_t q, work_قطار_func_t func, فراغ * معلومات, بنانشان صحيح n) {
    إذا (n > q->n_ڌاڳا) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_ڌاڳا);
        أرجع false;
    }

    بنانشان صحيح ir = atomic_ڀريو_explicit(&q->idx_read,  memory_order_relaxed);
    بنانشان صحيح iw = atomic_ڀريو_explicit(&q->idx_write, memory_order_relaxed);

    إذا (((iw + 1) & q->idx_نقاب) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        أرجع false;
    }

    بنية work_قطار_task_s * task = &q->قطار[iw];
    task->func      = func;
    task->معلومات      = معلومات;
    task->n_ڌاڳا = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_نقاب, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_جوڙيو_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, معلومات);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    طالما (atomic_ڀريو_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_نقاب, memory_order_relaxed);

    أرجع true;
}

size_t work_قطار_sizeof(uint32_t n_ڌاڳا, uint32_t صلاحيت, uint32_t ڍير_سائيز) {
    صلاحيت = hex_ceil_pow2(صلاحيت);
    uint32_t n_workers = n_ڌاڳا > 1 ? n_ڌاڳا - 1 : 0;
    size_t سائيز_stacks = ڍير_سائيز * n_workers;
    size_t سائيز_q = hex_align_مٿي(حجم(بنية work_قطار_s) + صلاحيت * حجم(بنية work_قطار_task_s), HEX_L2_سٽ_سائيز);
    أرجع سائيز_stacks + سائيز_q;
}

size_t work_قطار_alignof(فراغ) {
    أرجع 4096;
}

work_قطار_t work_قطار_init(فراغ * ptr, uint32_t n_ڌاڳا, uint32_t صلاحيت, uint32_t ڍير_سائيز) {
    صلاحيت = hex_ceil_pow2(صلاحيت);
    uint32_t n_workers = n_ڌاڳا > 1 ? n_ڌاڳا - 1 : 0;
    بنانشان حرف * mem_blob = (بنانشان حرف *) ptr;

    work_قطار_t q = (work_قطار_t) (mem_blob + ڍير_سائيز * n_workers);
    memset(q, 0, حجم(بنية work_قطار_s) + صلاحيت * حجم(بنية work_قطار_task_s));

    q->n_ڌاڳا = n_ڌاڳا;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->صلاحيت = صلاحيت;

    لكل (بنانشان صحيح i = 0; i < n_workers; i++) {
        q->ڍير[i]  = mem_blob; mem_blob += ڍير_سائيز;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].قطار = q;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->سرگرم,    0);
    q->idx_نقاب  = صلاحيت - 1;
    q->killed    = 0;
    لكل (صحيح i = 0; i < (صحيح) صلاحيت; i++) {
        atomic_init(&q->قطار[i].barrier, 0);
        q->قطار[i].func      = عدم;
        q->قطار[i].معلومات      = عدم;
        q->قطار[i].n_ڌاڳا = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    لكل (بنانشان صحيح i = 0; i < n_workers; i++) {
        qurt_thread_attr_سيٽ_ڍير_addr(&attr, q->ڍير[i]);
        qurt_thread_attr_سيٽ_ڍير_سائيز(&attr, ڍير_سائيز);

        حرف thread_name[32];
        snprintf(thread_name, حجم(thread_name), "work-queue:%u", i);
        qurt_thread_attr_سيٽ_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        صحيح prio = qurt_thread_get_ترجيح(qurt_thread_get_id());
        إذا (prio < 1) {
            prio = 1;
        }
        إذا (prio > سڀ_کان_هيٺ_USABLE_QURT_PRIO) {
            prio = سڀ_کان_هيٺ_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_سيٽ_ترجيح(&attr, prio);

        صحيح err = qurt_thread_create(&q->thread[i], &attr, work_قطار_thread, (فراغ *) &q->context[i]);
        إذا (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_قطار_free(q);
            أرجع عدم;
        }
    }

    أرجع q;
}

فراغ work_قطار_free(work_قطار_t q) {
    إذا (!q) { أرجع; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->سرگرم,   1, memory_order_release);
    atomic_fetch_جوڙيو_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->سرگرم, q->n_workers);

    لكل (بنانشان صحيح i = 0; i < q->n_workers; i++) {
        إذا (q->thread[i]) {
            صحيح حالت;
            (فراغ) qurt_thread_join(q->thread[i], &حالت);
        }
    }
}

فراغ work_قطار_wakeup(work_قطار_t q) {
    إذا (!atomic_ڀريو_explicit(&q->سرگرم, memory_order_relaxed)) {
        atomic_store_explicit(&q->سرگرم, 1, memory_order_release);
        qurt_futex_wake(&q->سرگرم, q->n_workers);
    }
}

فراغ work_قطار_suspend(work_قطار_t q) {
    atomic_store_explicit(&q->سرگرم, 0, memory_order_release);
}
