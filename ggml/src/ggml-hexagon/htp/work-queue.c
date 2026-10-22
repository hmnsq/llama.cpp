#شامل "work-queue.h"
#شامل "hex-utils.h"

#شامل <qurt.h>
#شامل <qurt_hvx.h>

#شامل <stdatomic.h>
#شامل <stdint.h>
#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <string.h>

#شامل "HAP_farf.h"

#تعریف_کرو سب_سے_نیچا_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
قسمتعریف ساخت {
    work_قطار_t  قطار;
    غیرعلامتی صحیح  id;
} کارکن_context_t;

ساخت work_قطار_task_s {
    work_قطار_func_t func;
    خالی *            ڈیٹا;
    غیرعلامتی صحیح      n_دھاگے;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
ساخت work_قطار_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_نقاب;
    uint32_t           استعداد;

    qurt_thread_t      thread[WORK_قطار_MAX_N_دھاگے];   // thread ID's of the workers
    کارکن_context_t   context[WORK_قطار_MAX_N_دھاگے];  // worker contexts
    خالی *             ڈھیر[WORK_قطار_MAX_N_دھاگے];    // thread stack pointers
    غیرعلامتی صحیح       n_دھاگے;                          // total threads (workers + main)
    غیرعلامتی صحیح       n_workers;                          // number of active threads (just workers)

    atomic_uint        فعال;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    ساخت work_قطار_task_s قطار[] __attribute__((aligned(HEX_L2_سطر_سائز)));
};

ساکن خالی work_قطار_thread(خالی * context) {
    کارکن_context_t * me = (کارکن_context_t *) context;
    work_قطار_t       q  = me->قطار;

    FARF(بلند_2, "work-queue: thread %u started", me->id);

    غیرعلامتی صحیح پچھلا_seqn = 0;

    جب (!atomic_بھرو_explicit(&q->killed, memory_order_relaxed)) {
        غیرعلامتی صحیح seqn = atomic_بھرو_explicit(&q->seqn, memory_order_acquire);
        اگر (seqn == پچھلا_seqn) {
            اگر (atomic_بھرو_explicit(&q->فعال, memory_order_relaxed)) {
                hex_pause();
            } ورنہ {
                qurt_futex_wait(&q->فعال, 0);
            }
            جاری;
        }

        پچھلا_seqn = seqn;

        // Process all active tasks in the queue
        غیرعلامتی صحیح ir = atomic_بھرو_explicit(&q->idx_read, memory_order_relaxed);
        غیرعلامتی صحیح iw = atomic_بھرو_explicit(&q->idx_write, memory_order_relaxed);

        جب (ir != iw) {
            ساخت work_قطار_task_s * task = &q->قطار[ir];

            غیرعلامتی صحیح n = task->n_دھاگے;
            غیرعلامتی صحیح i = me->id;
            اگر (i < n) {
                task->func(n, i, task->ڈیٹا);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } ورنہ {
                جب (atomic_بھرو_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_نقاب;
        }
    }

    FARF(بلند_2, "work-queue: thread %u stopped", me->id);
}

bool work_قطار_تسلسل_async(work_قطار_t q, work_قطار_func_t func, خالی * ڈیٹا, غیرعلامتی صحیح n) {
    اگر (n > q->n_دھاگے) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_دھاگے);
        واپس false;
    }

    غیرعلامتی صحیح ir = atomic_بھرو_explicit(&q->idx_read,  memory_order_relaxed);
    غیرعلامتی صحیح iw = atomic_بھرو_explicit(&q->idx_write, memory_order_relaxed);

    اگر (((iw + 1) & q->idx_نقاب) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        واپس false;
    }

    ساخت work_قطار_task_s * task = &q->قطار[iw];
    task->func      = func;
    task->ڈیٹا      = ڈیٹا;
    task->n_دھاگے = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_نقاب, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_جوڑو_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(n, 0, ڈیٹا);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    جب (atomic_بھرو_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_نقاب, memory_order_relaxed);

    واپس true;
}

size_t work_قطار_sizeof(uint32_t n_دھاگے, uint32_t استعداد, uint32_t ڈھیر_سائز) {
    استعداد = hex_ceil_pow2(استعداد);
    uint32_t n_workers = n_دھاگے > 1 ? n_دھاگے - 1 : 0;
    size_t سائز_stacks = ڈھیر_سائز * n_workers;
    size_t سائز_q = hex_align_اوپر(حجم(ساخت work_قطار_s) + استعداد * حجم(ساخت work_قطار_task_s), HEX_L2_سطر_سائز);
    واپس سائز_stacks + سائز_q;
}

size_t work_قطار_alignof(خالی) {
    واپس 4096;
}

work_قطار_t work_قطار_init(خالی * ptr, uint32_t n_دھاگے, uint32_t استعداد, uint32_t ڈھیر_سائز) {
    استعداد = hex_ceil_pow2(استعداد);
    uint32_t n_workers = n_دھاگے > 1 ? n_دھاگے - 1 : 0;
    غیرعلامتی حرف * mem_blob = (غیرعلامتی حرف *) ptr;

    work_قطار_t q = (work_قطار_t) (mem_blob + ڈھیر_سائز * n_workers);
    memset(q, 0, حجم(ساخت work_قطار_s) + استعداد * حجم(ساخت work_قطار_task_s));

    q->n_دھاگے = n_دھاگے;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->استعداد = استعداد;

    برائے (غیرعلامتی صحیح i = 0; i < n_workers; i++) {
        q->ڈھیر[i]  = mem_blob; mem_blob += ڈھیر_سائز;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].قطار = q;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->فعال,    0);
    q->idx_نقاب  = استعداد - 1;
    q->killed    = 0;
    برائے (صحیح i = 0; i < (صحیح) استعداد; i++) {
        atomic_init(&q->قطار[i].barrier, 0);
        q->قطار[i].func      = تہی;
        q->قطار[i].ڈیٹا      = تہی;
        q->قطار[i].n_دھاگے = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    برائے (غیرعلامتی صحیح i = 0; i < n_workers; i++) {
        qurt_thread_attr_سیٹ_ڈھیر_addr(&attr, q->ڈھیر[i]);
        qurt_thread_attr_سیٹ_ڈھیر_سائز(&attr, ڈھیر_سائز);

        حرف thread_name[32];
        snprintf(thread_name, حجم(thread_name), "work-queue:%u", i);
        qurt_thread_attr_سیٹ_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        صحیح prio = qurt_thread_get_ترجیح(qurt_thread_get_id());
        اگر (prio < 1) {
            prio = 1;
        }
        اگر (prio > سب_سے_نیچا_USABLE_QURT_PRIO) {
            prio = سب_سے_نیچا_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_سیٹ_ترجیح(&attr, prio);

        صحیح err = qurt_thread_create(&q->thread[i], &attr, work_قطار_thread, (خالی *) &q->context[i]);
        اگر (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_قطار_free(q);
            واپس تہی;
        }
    }

    واپس q;
}

خالی work_قطار_free(work_قطار_t q) {
    اگر (!q) { واپس; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->فعال,   1, memory_order_release);
    atomic_fetch_جوڑو_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->فعال, q->n_workers);

    برائے (غیرعلامتی صحیح i = 0; i < q->n_workers; i++) {
        اگر (q->thread[i]) {
            صحیح حالت_2;
            (خالی) qurt_thread_join(q->thread[i], &حالت_2);
        }
    }
}

خالی work_قطار_wakeup(work_قطار_t q) {
    اگر (!atomic_بھرو_explicit(&q->فعال, memory_order_relaxed)) {
        atomic_store_explicit(&q->فعال, 1, memory_order_release);
        qurt_futex_wake(&q->فعال, q->n_workers);
    }
}

خالی work_قطار_suspend(work_قطار_t q) {
    atomic_store_explicit(&q->فعال, 0, memory_order_release);
}
