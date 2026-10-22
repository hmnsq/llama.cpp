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

#تعریف_کرو سبھ_توں_نیواں_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
تعريفنوع بنية {
    work_قطار_t  قطار;
    بنانشانی صحيح  id;
} کامہ_context_t;

بنية work_قطار_task_s {
    work_قطار_func_t func;
    فراغ *            انکڑے;
    بنانشانی صحيح      n_دھاگے;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
بنية work_قطار_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_نقاب;
    uint32_t           سمرتھا;

    qurt_thread_t      thread[WORK_قطار_MAX_N_دھاگے];   // thread ID's of the workers
    کامہ_context_t   context[WORK_قطار_MAX_N_دھاگے];  // worker contexts
    فراغ *             ڈھیر[WORK_قطار_MAX_N_دھاگے];    // thread stack pointers
    بنانشانی صحيح       n_دھاگے;                          // total threads (workers + main)
    بنانشانی صحيح       n_workers;                          // number of active threads (just workers)

    atomic_uint        سرگرم;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    بنية work_قطار_task_s قطار[] __attribute__((aligned(HEX_L2_سطر_ناپ)));
};

ساكن فراغ work_قطار_thread(فراغ * context) {
    کامہ_context_t * me = (کامہ_context_t *) context;
    work_قطار_t       q  = me->قطار;

    FARF(بلند_2, "work-queue: thread %u started", me->id);

    بنانشانی صحيح prev_seqn = 0;

    طالما (!atomic_لدو_explicit(&q->killed, memory_order_relaxed)) {
        بنانشانی صحيح seqn = atomic_لدو_explicit(&q->seqn, memory_order_acquire);
        إذا (seqn == prev_seqn) {
            إذا (atomic_لدو_explicit(&q->سرگرم, memory_order_relaxed)) {
                hex_pause();
            } وإلا {
                qurt_futex_wait(&q->سرگرم, 0);
            }
            استمر;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        بنانشانی صحيح ir = atomic_لدو_explicit(&q->idx_read, memory_order_relaxed);
        بنانشانی صحيح iw = atomic_لدو_explicit(&q->idx_write, memory_order_relaxed);

        طالما (ir != iw) {
            بنية work_قطار_task_s * task = &q->قطار[ir];

            بنانشانی صحيح n = task->n_دھاگے;
            بنانشانی صحيح i = me->id;
            إذا (i < n) {
                task->func(n, i, task->انکڑے);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } وإلا {
                طالما (atomic_لدو_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_نقاب;
        }
    }

    FARF(بلند_2, "work-queue: thread %u stopped", me->id);
}

bool work_قطار_لگاتار_async(work_قطار_t q, work_قطار_func_t func, فراغ * انکڑے, بنانشانی صحيح n) {
    إذا (n > q->n_دھاگے) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_دھاگے);
        أرجع false;
    }

    بنانشانی صحيح ir = atomic_لدو_explicit(&q->idx_read,  memory_order_relaxed);
    بنانشانی صحيح iw = atomic_لدو_explicit(&q->idx_write, memory_order_relaxed);

    إذا (((iw + 1) & q->idx_نقاب) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        أرجع false;
    }

    بنية work_قطار_task_s * task = &q->قطار[iw];
    task->func      = func;
    task->انکڑے      = انکڑے;
    task->n_دھاگے = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_نقاب, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_جوڑو_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, انکڑے);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    طالما (atomic_لدو_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_نقاب, memory_order_relaxed);

    أرجع true;
}

size_t work_قطار_sizeof(uint32_t n_دھاگے, uint32_t سمرتھا, uint32_t ڈھیر_ناپ) {
    سمرتھا = hex_ceil_pow2(سمرتھا);
    uint32_t n_workers = n_دھاگے > 1 ? n_دھاگے - 1 : 0;
    size_t ناپ_stacks = ڈھیر_ناپ * n_workers;
    size_t ناپ_q = hex_align_اتے(حجم(بنية work_قطار_s) + سمرتھا * حجم(بنية work_قطار_task_s), HEX_L2_سطر_ناپ);
    أرجع ناپ_stacks + ناپ_q;
}

size_t work_قطار_alignof(فراغ) {
    أرجع 4096;
}

work_قطار_t work_قطار_init(فراغ * ptr, uint32_t n_دھاگے, uint32_t سمرتھا, uint32_t ڈھیر_ناپ) {
    سمرتھا = hex_ceil_pow2(سمرتھا);
    uint32_t n_workers = n_دھاگے > 1 ? n_دھاگے - 1 : 0;
    بنانشانی حرف * mem_blob = (بنانشانی حرف *) ptr;

    work_قطار_t q = (work_قطار_t) (mem_blob + ڈھیر_ناپ * n_workers);
    memset(q, 0, حجم(بنية work_قطار_s) + سمرتھا * حجم(بنية work_قطار_task_s));

    q->n_دھاگے = n_دھاگے;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->سمرتھا = سمرتھا;

    لكل (بنانشانی صحيح i = 0; i < n_workers; i++) {
        q->ڈھیر[i]  = mem_blob; mem_blob += ڈھیر_ناپ;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].قطار = q;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->سرگرم,    0);
    q->idx_نقاب  = سمرتھا - 1;
    q->killed    = 0;
    لكل (صحيح i = 0; i < (صحيح) سمرتھا; i++) {
        atomic_init(&q->قطار[i].barrier, 0);
        q->قطار[i].func      = عدم;
        q->قطار[i].انکڑے      = عدم;
        q->قطار[i].n_دھاگے = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    لكل (بنانشانی صحيح i = 0; i < n_workers; i++) {
        qurt_thread_attr_سیٹ_ڈھیر_addr(&attr, q->ڈھیر[i]);
        qurt_thread_attr_سیٹ_ڈھیر_ناپ(&attr, ڈھیر_ناپ);

        حرف thread_name[32];
        snprintf(thread_name, حجم(thread_name), "work-queue:%u", i);
        qurt_thread_attr_سیٹ_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        صحيح prio = qurt_thread_get_ترجیح(qurt_thread_get_id());
        إذا (prio < 1) {
            prio = 1;
        }
        إذا (prio > سبھ_توں_نیواں_USABLE_QURT_PRIO) {
            prio = سبھ_توں_نیواں_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_سیٹ_ترجیح(&attr, prio);

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
    atomic_fetch_جوڑو_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->سرگرم, q->n_workers);

    لكل (بنانشانی صحيح i = 0; i < q->n_workers; i++) {
        إذا (q->thread[i]) {
            صحيح حالت;
            (فراغ) qurt_thread_join(q->thread[i], &حالت);
        }
    }
}

فراغ work_قطار_wakeup(work_قطار_t q) {
    إذا (!atomic_لدو_explicit(&q->سرگرم, memory_order_relaxed)) {
        atomic_store_explicit(&q->سرگرم, 1, memory_order_release);
        qurt_futex_wake(&q->سرگرم, q->n_workers);
    }
}

فراغ work_قطار_suspend(work_قطار_t q) {
    atomic_store_explicit(&q->سرگرم, 0, memory_order_release);
}
