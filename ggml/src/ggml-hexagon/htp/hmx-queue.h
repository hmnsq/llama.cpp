#اگر_تعریف_نشده HMX_صف_H
#تعریف_کن HMX_صف_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <stdatomic.h>

#ضمن <hexagon_types.h>
#ضمن <qurt_thread.h>
#ضمن <qurt_futex.h>
#ضمن <HAP_farf.h>

#ضمن "hex-utils.h"
#ضمن "hex-profile.h"

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

#إذا __HVX_ARCH__ > 79
#تعریف_کن HMX_صف_POLL_شمار 2000
#وإلا
#تعریف_کن HMX_صف_POLL_شمار 1
#پایان_اگر

تعريفنوع فراغ (*hmx_صف_func)(فراغ *);

// Dummy funcs used as signals
تعداد hmx_صف_signal {
    HMX_صف_NOOP = 0, // aka NULL
    HMX_صف_WAKEUP,
    HMX_صف_SUSPEND,
    HMX_صف_KILL
};

بنية hmx_صف_desc {
    hmx_صف_func   func;
    فراغ *           داده;
    atomic_uint      انجامشده;
};

بنية hmx_صف_s {
    بنية hmx_صف_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    بیعلامت صحيح     idx_برداشت;   // updated by producer (pop)
    uint32_t         idx_ماسک;
    uint32_t         ظرفیت;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    فراغ *           پشته;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    بنية htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

تعريفنوع بنية hmx_صف_s * hmx_صف_t;

size_t      hmx_صف_sizeof(size_t ظرفیت, uint32_t پشته_اندازه);
size_t      hmx_صف_alignof(فراغ);
hmx_صف_t hmx_صف_init(فراغ * ptr, size_t ظرفیت, uint32_t پشته_اندازه, uint32_t hap_rctx, بنية htp_thread_trace * trace);
فراغ        hmx_صف_free(hmx_صف_t q);

ساكن مضمن بنية hmx_صف_desc hmx_صف_make_desc(hmx_صف_func func, فراغ * داده) {
    بنية hmx_صف_desc d = { func, داده };
    أرجع d;
}

ساكن مضمن bool hmx_صف_افزودن(hmx_صف_t q, بنية hmx_صف_desc d) {
    بیعلامت صحيح ir = atomic_بارگیری(&q->idx_read);
    بیعلامت صحيح iw = atomic_بارگیری(&q->idx_write);

    إذا (((iw + 1) & q->idx_ماسک) == ir) {
        FARF(بالا_2, "hmx-queue-push: queue is full\n");
        أرجع false;
    }

    atomic_store(&d.انجامشده, 0);

    FARF(بالا_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.داده);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_ماسک);
    // wake up our thread
    atomic_fetch_جمع_کن(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    أرجع true;
}

ساكن مضمن bool hmx_صف_signal(hmx_صف_t q, تعداد hmx_صف_signal sig) {
    أرجع hmx_صف_افزودن(q, hmx_صف_make_desc((hmx_صف_func) sig, عدم));
}

ساكن مضمن bool hmx_صف_خالی(hmx_صف_t q) {
    أرجع q->idx_برداشت == atomic_بارگیری(&q->idx_write);
}

ساكن مضمن uint32_t hmx_صف_عمق(hmx_صف_t q) {
    أرجع (atomic_بارگیری(&q->idx_write) - atomic_بارگیری(&q->idx_read)) & q->idx_ماسک;
}

ساكن مضمن uint32_t hmx_صف_ظرفیت(hmx_صف_t q) {
    أرجع q->ظرفیت;
}

ساكن مضمن بنية hmx_صف_desc hmx_صف_برداشت_one(hmx_صف_t q) {
    بیعلامت صحيح ip = q->idx_برداشت;
    بیعلامت صحيح iw = atomic_بارگیری(&q->idx_write);

    بنية hmx_صف_desc rd = { عدم, عدم };
    إذا (ip == iw) {
        أرجع rd;
    }

    // Wait for desc to complete
    بنية hmx_صف_desc * d = &q->desc[ip];
    طالما (!atomic_بارگیری(&d->انجامشده)) {
        FARF(بالا_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_برداشت = (ip + 1) & q->idx_ماسک;

    FARF(بالا_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.داده);
    أرجع rd;
}

ساكن مضمن بنية hmx_صف_desc hmx_صف_برداشت(hmx_صف_t q) {
    طالما (1) {
        بنية hmx_صف_desc d = hmx_صف_برداشت_one(q);

        uint32_t sig = (uint32_t) d.func;
        إذا (sig && sig <= HMX_صف_KILL)
            استمر;

        أرجع d;
    }
}

ساكن مضمن فراغ hmx_صف_flush(hmx_صف_t q) {
    طالما (hmx_صف_برداشت_one(q).func != عدم) ;
}

ساكن مضمن فراغ hmx_صف_wakeup(hmx_صف_t q) {
    hmx_صف_signal(q, HMX_صف_WAKEUP);
}

ساكن مضمن فراغ hmx_صف_suspend(hmx_صف_t q) {
    hmx_صف_signal(q, HMX_صف_SUSPEND);
}

#اگر_تعریف_شده __cplusplus
}  // extern "C"
#پایان_اگر

#پایان_اگر /* HMX_QUEUE_H */
