#جے_بنا_تعریف HMX_قطار_H
#تعریف_کرو HMX_قطار_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <stdatomic.h>

#ضمن <hexagon_types.h>
#ضمن <qurt_thread.h>
#ضمن <qurt_futex.h>
#ضمن <HAP_farf.h>

#ضمن "hex-utils.h"
#ضمن "hex-profile.h"

#جے_تعریف_شدہ __cplusplus
خارجي "C" {
#جے_انت

#إذا __HVX_ARCH__ > 79
#تعریف_کرو HMX_قطار_POLL_گنتی 2000
#وإلا
#تعریف_کرو HMX_قطار_POLL_گنتی 1
#جے_انت

تعريفنوع فراغ (*hmx_قطار_func)(فراغ *);

// Dummy funcs used as signals
تعداد hmx_قطار_signal {
    HMX_قطار_NOOP = 0, // aka NULL
    HMX_قطار_WAKEUP,
    HMX_قطار_SUSPEND,
    HMX_قطار_KILL
};

بنية hmx_قطار_desc {
    hmx_قطار_func   func;
    فراغ *           انکڑے;
    atomic_uint      پورا;
};

بنية hmx_قطار_s {
    بنية hmx_قطار_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    بنانشانی صحيح     idx_نکالو;   // updated by producer (pop)
    uint32_t         idx_نقاب;
    uint32_t         سمرتھا;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    فراغ *           ڈھیر;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    بنية htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

تعريفنوع بنية hmx_قطار_s * hmx_قطار_t;

size_t      hmx_قطار_sizeof(size_t سمرتھا, uint32_t ڈھیر_ناپ);
size_t      hmx_قطار_alignof(فراغ);
hmx_قطار_t hmx_قطار_init(فراغ * ptr, size_t سمرتھا, uint32_t ڈھیر_ناپ, uint32_t hap_rctx, بنية htp_thread_trace * trace);
فراغ        hmx_قطار_free(hmx_قطار_t q);

ساكن مضمن بنية hmx_قطار_desc hmx_قطار_make_desc(hmx_قطار_func func, فراغ * انکڑے) {
    بنية hmx_قطار_desc d = { func, انکڑے };
    أرجع d;
}

ساكن مضمن bool hmx_قطار_ڈالو(hmx_قطار_t q, بنية hmx_قطار_desc d) {
    بنانشانی صحيح ir = atomic_لدو(&q->idx_read);
    بنانشانی صحيح iw = atomic_لدو(&q->idx_write);

    إذا (((iw + 1) & q->idx_نقاب) == ir) {
        FARF(بلند_2, "hmx-queue-push: queue is full\n");
        أرجع false;
    }

    atomic_store(&d.پورا, 0);

    FARF(بلند_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.انکڑے);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_نقاب);
    // wake up our thread
    atomic_fetch_جوڑو(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    أرجع true;
}

ساكن مضمن bool hmx_قطار_signal(hmx_قطار_t q, تعداد hmx_قطار_signal sig) {
    أرجع hmx_قطار_ڈالو(q, hmx_قطار_make_desc((hmx_قطار_func) sig, عدم));
}

ساكن مضمن bool hmx_قطار_خالی_تھاواں(hmx_قطار_t q) {
    أرجع q->idx_نکالو == atomic_لدو(&q->idx_write);
}

ساكن مضمن uint32_t hmx_قطار_ڈونگھائی(hmx_قطار_t q) {
    أرجع (atomic_لدو(&q->idx_write) - atomic_لدو(&q->idx_read)) & q->idx_نقاب;
}

ساكن مضمن uint32_t hmx_قطار_سمرتھا(hmx_قطار_t q) {
    أرجع q->سمرتھا;
}

ساكن مضمن بنية hmx_قطار_desc hmx_قطار_نکالو_one(hmx_قطار_t q) {
    بنانشانی صحيح ip = q->idx_نکالو;
    بنانشانی صحيح iw = atomic_لدو(&q->idx_write);

    بنية hmx_قطار_desc rd = { عدم, عدم };
    إذا (ip == iw) {
        أرجع rd;
    }

    // Wait for desc to complete
    بنية hmx_قطار_desc * d = &q->desc[ip];
    طالما (!atomic_لدو(&d->پورا)) {
        FARF(بلند_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_نکالو = (ip + 1) & q->idx_نقاب;

    FARF(بلند_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.انکڑے);
    أرجع rd;
}

ساكن مضمن بنية hmx_قطار_desc hmx_قطار_نکالو(hmx_قطار_t q) {
    طالما (1) {
        بنية hmx_قطار_desc d = hmx_قطار_نکالو_one(q);

        uint32_t sig = (uint32_t) d.func;
        إذا (sig && sig <= HMX_قطار_KILL)
            استمر;

        أرجع d;
    }
}

ساكن مضمن فراغ hmx_قطار_flush(hmx_قطار_t q) {
    طالما (hmx_قطار_نکالو_one(q).func != عدم) ;
}

ساكن مضمن فراغ hmx_قطار_wakeup(hmx_قطار_t q) {
    hmx_قطار_signal(q, HMX_قطار_WAKEUP);
}

ساكن مضمن فراغ hmx_قطار_suspend(hmx_قطار_t q) {
    hmx_قطار_signal(q, HMX_قطار_SUSPEND);
}

#جے_تعریف_شدہ __cplusplus
}  // extern "C"
#جے_انت

#جے_انت /* HMX_QUEUE_H */
