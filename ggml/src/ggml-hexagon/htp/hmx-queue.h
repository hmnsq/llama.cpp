#که_نه_تعریف_شوی HMX_کتار_H
#تعریف_کړه HMX_کتار_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <stdatomic.h>

#ضمن <hexagon_types.h>
#ضمن <qurt_thread.h>
#ضمن <qurt_futex.h>
#ضمن <HAP_farf.h>

#ضمن "hex-utils.h"
#ضمن "hex-profile.h"

#که_تعریف_شوی __cplusplus
خارجي "C" {
#د_که_پای

#إذا __HVX_ARCH__ > 79
#تعریف_کړه HMX_کتار_POLL_شمېر 2000
#وإلا
#تعریف_کړه HMX_کتار_POLL_شمېر 1
#د_که_پای

تعريفنوع فراغ (*hmx_کتار_func)(فراغ *);

// Dummy funcs used as signals
تعداد hmx_کتار_signal {
    HMX_کتار_NOOP = 0, // aka NULL
    HMX_کتار_WAKEUP,
    HMX_کتار_SUSPEND,
    HMX_کتار_KILL
};

بنية hmx_کتار_desc {
    hmx_کتار_func   func;
    فراغ *           معلومات;
    atomic_uint      بشپړ;
};

بنية hmx_کتار_s {
    بنية hmx_کتار_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    بینښه صحيح     idx_راوباسه;   // updated by producer (pop)
    uint32_t         idx_ماسک;
    uint32_t         ظرفیت;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    فراغ *           ډېری;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    بنية htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

تعريفنوع بنية hmx_کتار_s * hmx_کتار_t;

size_t      hmx_کتار_sizeof(size_t ظرفیت, uint32_t ډېری_اندازه);
size_t      hmx_کتار_alignof(فراغ);
hmx_کتار_t hmx_کتار_init(فراغ * ptr, size_t ظرفیت, uint32_t ډېری_اندازه, uint32_t hap_rctx, بنية htp_thread_trace * trace);
فراغ        hmx_کتار_free(hmx_کتار_t q);

ساكن مضمن بنية hmx_کتار_desc hmx_کتار_make_desc(hmx_کتار_func func, فراغ * معلومات) {
    بنية hmx_کتار_desc d = { func, معلومات };
    أرجع d;
}

ساكن مضمن bool hmx_کتار_ټېل(hmx_کتار_t q, بنية hmx_کتار_desc d) {
    بینښه صحيح ir = atomic_بارول(&q->idx_read);
    بینښه صحيح iw = atomic_بارول(&q->idx_write);

    إذا (((iw + 1) & q->idx_ماسک) == ir) {
        FARF(لوړ_2, "hmx-queue-push: queue is full\n");
        أرجع false;
    }

    atomic_store(&d.بشپړ, 0);

    FARF(لوړ_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.معلومات);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_ماسک);
    // wake up our thread
    atomic_fetch_جمع_کړه(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    أرجع true;
}

ساكن مضمن bool hmx_کتار_signal(hmx_کتار_t q, تعداد hmx_کتار_signal sig) {
    أرجع hmx_کتار_ټېل(q, hmx_کتار_make_desc((hmx_کتار_func) sig, عدم));
}

ساكن مضمن bool hmx_کتار_تشه(hmx_کتار_t q) {
    أرجع q->idx_راوباسه == atomic_بارول(&q->idx_write);
}

ساكن مضمن uint32_t hmx_کتار_ژوروالی(hmx_کتار_t q) {
    أرجع (atomic_بارول(&q->idx_write) - atomic_بارول(&q->idx_read)) & q->idx_ماسک;
}

ساكن مضمن uint32_t hmx_کتار_ظرفیت(hmx_کتار_t q) {
    أرجع q->ظرفیت;
}

ساكن مضمن بنية hmx_کتار_desc hmx_کتار_راوباسه_one(hmx_کتار_t q) {
    بینښه صحيح ip = q->idx_راوباسه;
    بینښه صحيح iw = atomic_بارول(&q->idx_write);

    بنية hmx_کتار_desc rd = { عدم, عدم };
    إذا (ip == iw) {
        أرجع rd;
    }

    // Wait for desc to complete
    بنية hmx_کتار_desc * d = &q->desc[ip];
    طالما (!atomic_بارول(&d->بشپړ)) {
        FARF(لوړ_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_راوباسه = (ip + 1) & q->idx_ماسک;

    FARF(لوړ_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.معلومات);
    أرجع rd;
}

ساكن مضمن بنية hmx_کتار_desc hmx_کتار_راوباسه(hmx_کتار_t q) {
    طالما (1) {
        بنية hmx_کتار_desc d = hmx_کتار_راوباسه_one(q);

        uint32_t sig = (uint32_t) d.func;
        إذا (sig && sig <= HMX_کتار_KILL)
            استمر;

        أرجع d;
    }
}

ساكن مضمن فراغ hmx_کتار_flush(hmx_کتار_t q) {
    طالما (hmx_کتار_راوباسه_one(q).func != عدم) ;
}

ساكن مضمن فراغ hmx_کتار_wakeup(hmx_کتار_t q) {
    hmx_کتار_signal(q, HMX_کتار_WAKEUP);
}

ساكن مضمن فراغ hmx_کتار_suspend(hmx_کتار_t q) {
    hmx_کتار_signal(q, HMX_کتار_SUSPEND);
}

#که_تعریف_شوی __cplusplus
}  // extern "C"
#د_که_پای

#د_که_پای /* HMX_QUEUE_H */
