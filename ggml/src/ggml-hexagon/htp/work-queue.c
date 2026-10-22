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

#تعریف_کړه ټیټ_ترین_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
تعريفنوع بنية {
    work_کتار_t  کتار;
    بینښه صحيح  id;
} کارګر_context_t;

بنية work_کتار_task_s {
    work_کتار_func_t func;
    فراغ *            معلومات;
    بینښه صحيح      n_تارونه;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
بنية work_کتار_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_ماسک;
    uint32_t           ظرفیت;

    qurt_thread_t      thread[WORK_کتار_MAX_N_تارونه];   // thread ID's of the workers
    کارګر_context_t   context[WORK_کتار_MAX_N_تارونه];  // worker contexts
    فراغ *             ډېری[WORK_کتار_MAX_N_تارونه];    // thread stack pointers
    بینښه صحيح       n_تارونه;                          // total threads (workers + main)
    بینښه صحيح       n_workers;                          // number of active threads (just workers)

    atomic_uint        فعال;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    بنية work_کتار_task_s کتار[] __attribute__((aligned(HEX_L2_کرښه_اندازه)));
};

ساكن فراغ work_کتار_thread(فراغ * context) {
    کارګر_context_t * me = (کارګر_context_t *) context;
    work_کتار_t       q  = me->کتار;

    FARF(لوړ_2, "work-queue: thread %u started", me->id);

    بینښه صحيح prev_seqn = 0;

    طالما (!atomic_بارول_explicit(&q->killed, memory_order_relaxed)) {
        بینښه صحيح seqn = atomic_بارول_explicit(&q->seqn, memory_order_acquire);
        إذا (seqn == prev_seqn) {
            إذا (atomic_بارول_explicit(&q->فعال, memory_order_relaxed)) {
                hex_pause();
            } وإلا {
                qurt_futex_wait(&q->فعال, 0);
            }
            استمر;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        بینښه صحيح ir = atomic_بارول_explicit(&q->idx_read, memory_order_relaxed);
        بینښه صحيح iw = atomic_بارول_explicit(&q->idx_write, memory_order_relaxed);

        طالما (ir != iw) {
            بنية work_کتار_task_s * task = &q->کتار[ir];

            بینښه صحيح n = task->n_تارونه;
            بینښه صحيح i = me->id;
            إذا (i < n) {
                task->func(n, i, task->معلومات);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } وإلا {
                طالما (atomic_بارول_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_ماسک;
        }
    }

    FARF(لوړ_2, "work-queue: thread %u stopped", me->id);
}

bool work_کتار_پرلپسې_async(work_کتار_t q, work_کتار_func_t func, فراغ * معلومات, بینښه صحيح n) {
    إذا (n > q->n_تارونه) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_تارونه);
        أرجع false;
    }

    بینښه صحيح ir = atomic_بارول_explicit(&q->idx_read,  memory_order_relaxed);
    بینښه صحيح iw = atomic_بارول_explicit(&q->idx_write, memory_order_relaxed);

    إذا (((iw + 1) & q->idx_ماسک) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        أرجع false;
    }

    بنية work_کتار_task_s * task = &q->کتار[iw];
    task->func      = func;
    task->معلومات      = معلومات;
    task->n_تارونه = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_ماسک, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_جمع_کړه_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, معلومات);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    طالما (atomic_بارول_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_ماسک, memory_order_relaxed);

    أرجع true;
}

size_t work_کتار_sizeof(uint32_t n_تارونه, uint32_t ظرفیت, uint32_t ډېری_اندازه) {
    ظرفیت = hex_ceil_pow2(ظرفیت);
    uint32_t n_workers = n_تارونه > 1 ? n_تارونه - 1 : 0;
    size_t اندازه_stacks = ډېری_اندازه * n_workers;
    size_t اندازه_q = hex_align_پورته(حجم(بنية work_کتار_s) + ظرفیت * حجم(بنية work_کتار_task_s), HEX_L2_کرښه_اندازه);
    أرجع اندازه_stacks + اندازه_q;
}

size_t work_کتار_alignof(فراغ) {
    أرجع 4096;
}

work_کتار_t work_کتار_init(فراغ * ptr, uint32_t n_تارونه, uint32_t ظرفیت, uint32_t ډېری_اندازه) {
    ظرفیت = hex_ceil_pow2(ظرفیت);
    uint32_t n_workers = n_تارونه > 1 ? n_تارونه - 1 : 0;
    بینښه حرف * mem_blob = (بینښه حرف *) ptr;

    work_کتار_t q = (work_کتار_t) (mem_blob + ډېری_اندازه * n_workers);
    memset(q, 0, حجم(بنية work_کتار_s) + ظرفیت * حجم(بنية work_کتار_task_s));

    q->n_تارونه = n_تارونه;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->ظرفیت = ظرفیت;

    لكل (بینښه صحيح i = 0; i < n_workers; i++) {
        q->ډېری[i]  = mem_blob; mem_blob += ډېری_اندازه;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].کتار = q;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->فعال,    0);
    q->idx_ماسک  = ظرفیت - 1;
    q->killed    = 0;
    لكل (صحيح i = 0; i < (صحيح) ظرفیت; i++) {
        atomic_init(&q->کتار[i].barrier, 0);
        q->کتار[i].func      = عدم;
        q->کتار[i].معلومات      = عدم;
        q->کتار[i].n_تارونه = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    لكل (بینښه صحيح i = 0; i < n_workers; i++) {
        qurt_thread_attr_سېټ_ډېری_addr(&attr, q->ډېری[i]);
        qurt_thread_attr_سېټ_ډېری_اندازه(&attr, ډېری_اندازه);

        حرف thread_name[32];
        snprintf(thread_name, حجم(thread_name), "work-queue:%u", i);
        qurt_thread_attr_سېټ_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        صحيح prio = qurt_thread_get_لومړیتوب(qurt_thread_get_id());
        إذا (prio < 1) {
            prio = 1;
        }
        إذا (prio > ټیټ_ترین_USABLE_QURT_PRIO) {
            prio = ټیټ_ترین_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_سېټ_لومړیتوب(&attr, prio);

        صحيح err = qurt_thread_create(&q->thread[i], &attr, work_کتار_thread, (فراغ *) &q->context[i]);
        إذا (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_کتار_free(q);
            أرجع عدم;
        }
    }

    أرجع q;
}

فراغ work_کتار_free(work_کتار_t q) {
    إذا (!q) { أرجع; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->فعال,   1, memory_order_release);
    atomic_fetch_جمع_کړه_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->فعال, q->n_workers);

    لكل (بینښه صحيح i = 0; i < q->n_workers; i++) {
        إذا (q->thread[i]) {
            صحيح حالت;
            (فراغ) qurt_thread_join(q->thread[i], &حالت);
        }
    }
}

فراغ work_کتار_wakeup(work_کتار_t q) {
    إذا (!atomic_بارول_explicit(&q->فعال, memory_order_relaxed)) {
        atomic_store_explicit(&q->فعال, 1, memory_order_release);
        qurt_futex_wake(&q->فعال, q->n_workers);
    }
}

فراغ work_کتار_suspend(work_کتار_t q) {
    atomic_store_explicit(&q->فعال, 0, memory_order_release);
}
