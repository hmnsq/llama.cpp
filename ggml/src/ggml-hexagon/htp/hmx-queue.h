#جيڪڏهن_وصف_نه_ٿيل HMX_قطار_H
#وصف_ڪريو HMX_قطار_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <stdatomic.h>

#ضمن <hexagon_types.h>
#ضمن <qurt_thread.h>
#ضمن <qurt_futex.h>
#ضمن <HAP_farf.h>

#ضمن "hex-utils.h"
#ضمن "hex-profile.h"

#جيڪڏهن_وصف_ٿيل __cplusplus
خارجي "C" {
#جيڪڏهن_پورو

#إذا __HVX_ARCH__ > 79
#وصف_ڪريو HMX_قطار_POLL_ڳڻپ 2000
#وإلا
#وصف_ڪريو HMX_قطار_POLL_ڳڻپ 1
#جيڪڏهن_پورو

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
    فراغ *           معلومات;
    atomic_uint      پورو;
};

بنية hmx_قطار_s {
    بنية hmx_قطار_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    بنانشان صحيح     idx_ڪڍو;   // updated by producer (pop)
    uint32_t         idx_نقاب;
    uint32_t         صلاحيت;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    فراغ *           ڍير;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    بنية htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

تعريفنوع بنية hmx_قطار_s * hmx_قطار_t;

size_t      hmx_قطار_sizeof(size_t صلاحيت, uint32_t ڍير_سائيز);
size_t      hmx_قطار_alignof(فراغ);
hmx_قطار_t hmx_قطار_init(فراغ * ptr, size_t صلاحيت, uint32_t ڍير_سائيز, uint32_t hap_rctx, بنية htp_thread_trace * trace);
فراغ        hmx_قطار_free(hmx_قطار_t q);

ساكن مضمن بنية hmx_قطار_desc hmx_قطار_make_desc(hmx_قطار_func func, فراغ * معلومات) {
    بنية hmx_قطار_desc d = { func, معلومات };
    أرجع d;
}

ساكن مضمن bool hmx_قطار_ڌڪو(hmx_قطار_t q, بنية hmx_قطار_desc d) {
    بنانشان صحيح ir = atomic_ڀريو(&q->idx_read);
    بنانشان صحيح iw = atomic_ڀريو(&q->idx_write);

    إذا (((iw + 1) & q->idx_نقاب) == ir) {
        FARF(مٿاهون_2, "hmx-queue-push: queue is full\n");
        أرجع false;
    }

    atomic_store(&d.پورو, 0);

    FARF(مٿاهون_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.معلومات);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_نقاب);
    // wake up our thread
    atomic_fetch_جوڙيو(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    أرجع true;
}

ساكن مضمن bool hmx_قطار_signal(hmx_قطار_t q, تعداد hmx_قطار_signal sig) {
    أرجع hmx_قطار_ڌڪو(q, hmx_قطار_make_desc((hmx_قطار_func) sig, عدم));
}

ساكن مضمن bool hmx_قطار_خالي_جايون(hmx_قطار_t q) {
    أرجع q->idx_ڪڍو == atomic_ڀريو(&q->idx_write);
}

ساكن مضمن uint32_t hmx_قطار_اونهائي(hmx_قطار_t q) {
    أرجع (atomic_ڀريو(&q->idx_write) - atomic_ڀريو(&q->idx_read)) & q->idx_نقاب;
}

ساكن مضمن uint32_t hmx_قطار_صلاحيت(hmx_قطار_t q) {
    أرجع q->صلاحيت;
}

ساكن مضمن بنية hmx_قطار_desc hmx_قطار_ڪڍو_one(hmx_قطار_t q) {
    بنانشان صحيح ip = q->idx_ڪڍو;
    بنانشان صحيح iw = atomic_ڀريو(&q->idx_write);

    بنية hmx_قطار_desc rd = { عدم, عدم };
    إذا (ip == iw) {
        أرجع rd;
    }

    // Wait for desc to complete
    بنية hmx_قطار_desc * d = &q->desc[ip];
    طالما (!atomic_ڀريو(&d->پورو)) {
        FARF(مٿاهون_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_ڪڍو = (ip + 1) & q->idx_نقاب;

    FARF(مٿاهون_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.معلومات);
    أرجع rd;
}

ساكن مضمن بنية hmx_قطار_desc hmx_قطار_ڪڍو(hmx_قطار_t q) {
    طالما (1) {
        بنية hmx_قطار_desc d = hmx_قطار_ڪڍو_one(q);

        uint32_t sig = (uint32_t) d.func;
        إذا (sig && sig <= HMX_قطار_KILL)
            استمر;

        أرجع d;
    }
}

ساكن مضمن فراغ hmx_قطار_flush(hmx_قطار_t q) {
    طالما (hmx_قطار_ڪڍو_one(q).func != عدم) ;
}

ساكن مضمن فراغ hmx_قطار_wakeup(hmx_قطار_t q) {
    hmx_قطار_signal(q, HMX_قطار_WAKEUP);
}

ساكن مضمن فراغ hmx_قطار_suspend(hmx_قطار_t q) {
    hmx_قطار_signal(q, HMX_قطار_SUSPEND);
}

#جيڪڏهن_وصف_ٿيل __cplusplus
}  // extern "C"
#جيڪڏهن_پورو

#جيڪڏهن_پورو /* HMX_QUEUE_H */
