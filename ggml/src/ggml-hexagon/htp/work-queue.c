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

#تعریف_کن پایینترین_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
تعريفنوع بنية {
    work_صف_t  صف;
    بیعلامت صحيح  id;
} کارگر_context_t;

بنية work_صف_task_s {
    work_صف_func_t func;
    فراغ *            داده;
    بیعلامت صحيح      n_نخها;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
بنية work_صف_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_ماسک;
    uint32_t           ظرفیت;

    qurt_thread_t      thread[WORK_صف_MAX_N_نخها];   // thread ID's of the workers
    کارگر_context_t   context[WORK_صف_MAX_N_نخها];  // worker contexts
    فراغ *             پشته[WORK_صف_MAX_N_نخها];    // thread stack pointers
    بیعلامت صحيح       n_نخها;                          // total threads (workers + main)
    بیعلامت صحيح       n_workers;                          // number of active threads (just workers)

    atomic_uint        فعال;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    بنية work_صف_task_s صف[] __attribute__((aligned(HEX_L2_سطر_اندازه)));
};

ساكن فراغ work_صف_thread(فراغ * context) {
    کارگر_context_t * me = (کارگر_context_t *) context;
    work_صف_t       q  = me->صف;

    FARF(بالا_2, "work-queue: thread %u started", me->id);

    بیعلامت صحيح prev_seqn = 0;

    طالما (!atomic_بارگیری_explicit(&q->killed, memory_order_relaxed)) {
        بیعلامت صحيح seqn = atomic_بارگیری_explicit(&q->seqn, memory_order_acquire);
        إذا (seqn == prev_seqn) {
            إذا (atomic_بارگیری_explicit(&q->فعال, memory_order_relaxed)) {
                hex_pause();
            } وإلا {
                qurt_futex_wait(&q->فعال, 0);
            }
            استمر;
        }

        prev_seqn = seqn;

        // Process all active tasks in the queue
        بیعلامت صحيح ir = atomic_بارگیری_explicit(&q->idx_read, memory_order_relaxed);
        بیعلامت صحيح iw = atomic_بارگیری_explicit(&q->idx_write, memory_order_relaxed);

        طالما (ir != iw) {
            بنية work_صف_task_s * task = &q->صف[ir];

            بیعلامت صحيح n = task->n_نخها;
            بیعلامت صحيح i = me->id;
            إذا (i < n) {
                task->func(n, i, task->داده);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } وإلا {
                طالما (atomic_بارگیری_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_ماسک;
        }
    }

    FARF(بالا_2, "work-queue: thread %u stopped", me->id);
}

bool work_صف_پیاپی_async(work_صف_t q, work_صف_func_t func, فراغ * داده, بیعلامت صحيح n) {
    إذا (n > q->n_نخها) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_نخها);
        أرجع false;
    }

    بیعلامت صحيح ir = atomic_بارگیری_explicit(&q->idx_read,  memory_order_relaxed);
    بیعلامت صحيح iw = atomic_بارگیری_explicit(&q->idx_write, memory_order_relaxed);

    إذا (((iw + 1) & q->idx_ماسک) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        أرجع false;
    }

    بنية work_صف_task_s * task = &q->صف[iw];
    task->func      = func;
    task->داده      = داده;
    task->n_نخها = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_ماسک, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_جمع_کن_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, داده);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    طالما (atomic_بارگیری_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_ماسک, memory_order_relaxed);

    أرجع true;
}

size_t work_صف_sizeof(uint32_t n_نخها, uint32_t ظرفیت, uint32_t پشته_اندازه) {
    ظرفیت = hex_ceil_pow2(ظرفیت);
    uint32_t n_workers = n_نخها > 1 ? n_نخها - 1 : 0;
    size_t اندازه_stacks = پشته_اندازه * n_workers;
    size_t اندازه_q = hex_align_به_بالا(حجم(بنية work_صف_s) + ظرفیت * حجم(بنية work_صف_task_s), HEX_L2_سطر_اندازه);
    أرجع اندازه_stacks + اندازه_q;
}

size_t work_صف_alignof(فراغ) {
    أرجع 4096;
}

work_صف_t work_صف_init(فراغ * ptr, uint32_t n_نخها, uint32_t ظرفیت, uint32_t پشته_اندازه) {
    ظرفیت = hex_ceil_pow2(ظرفیت);
    uint32_t n_workers = n_نخها > 1 ? n_نخها - 1 : 0;
    بیعلامت حرف * mem_blob = (بیعلامت حرف *) ptr;

    work_صف_t q = (work_صف_t) (mem_blob + پشته_اندازه * n_workers);
    memset(q, 0, حجم(بنية work_صف_s) + ظرفیت * حجم(بنية work_صف_task_s));

    q->n_نخها = n_نخها;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->ظرفیت = ظرفیت;

    لكل (بیعلامت صحيح i = 0; i < n_workers; i++) {
        q->پشته[i]  = mem_blob; mem_blob += پشته_اندازه;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].صف = q;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->فعال,    0);
    q->idx_ماسک  = ظرفیت - 1;
    q->killed    = 0;
    لكل (صحيح i = 0; i < (صحيح) ظرفیت; i++) {
        atomic_init(&q->صف[i].barrier, 0);
        q->صف[i].func      = عدم;
        q->صف[i].داده      = عدم;
        q->صف[i].n_نخها = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    لكل (بیعلامت صحيح i = 0; i < n_workers; i++) {
        qurt_thread_attr_مجموعه_پشته_addr(&attr, q->پشته[i]);
        qurt_thread_attr_مجموعه_پشته_اندازه(&attr, پشته_اندازه);

        حرف thread_name[32];
        snprintf(thread_name, حجم(thread_name), "work-queue:%u", i);
        qurt_thread_attr_مجموعه_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        صحيح prio = qurt_thread_get_اولویت(qurt_thread_get_id());
        إذا (prio < 1) {
            prio = 1;
        }
        إذا (prio > پایینترین_USABLE_QURT_PRIO) {
            prio = پایینترین_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_مجموعه_اولویت(&attr, prio);

        صحيح err = qurt_thread_create(&q->thread[i], &attr, work_صف_thread, (فراغ *) &q->context[i]);
        إذا (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_صف_free(q);
            أرجع عدم;
        }
    }

    أرجع q;
}

فراغ work_صف_free(work_صف_t q) {
    إذا (!q) { أرجع; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->فعال,   1, memory_order_release);
    atomic_fetch_جمع_کن_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->فعال, q->n_workers);

    لكل (بیعلامت صحيح i = 0; i < q->n_workers; i++) {
        إذا (q->thread[i]) {
            صحيح وضعیت;
            (فراغ) qurt_thread_join(q->thread[i], &وضعیت);
        }
    }
}

فراغ work_صف_wakeup(work_صف_t q) {
    إذا (!atomic_بارگیری_explicit(&q->فعال, memory_order_relaxed)) {
        atomic_store_explicit(&q->فعال, 1, memory_order_release);
        qurt_futex_wake(&q->فعال, q->n_workers);
    }
}

فراغ work_صف_suspend(work_صف_t q) {
    atomic_store_explicit(&q->فعال, 0, memory_order_release);
}
