#اگر_غیر_تعریف_شدہ HMX_قطار_H
#تعریف_کرو HMX_قطار_H

#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <stdatomic.h>

#شامل <hexagon_types.h>
#شامل <qurt_thread.h>
#شامل <qurt_futex.h>
#شامل <HAP_farf.h>

#شامل "hex-utils.h"
#شامل "hex-profile.h"

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

#اگر __HVX_ARCH__ > 79
#تعریف_کرو HMX_قطار_POLL_گنتی 2000
#ورنہ
#تعریف_کرو HMX_قطار_POLL_گنتی 1
#اگر_ختم

قسمتعریف خالی (*hmx_قطار_func)(خالی *);

// Dummy funcs used as signals
شمار hmx_قطار_signal {
    HMX_قطار_NOOP = 0, // aka NULL
    HMX_قطار_WAKEUP,
    HMX_قطار_SUSPEND,
    HMX_قطار_KILL
};

ساخت hmx_قطار_desc {
    hmx_قطار_func   func;
    خالی *           ڈیٹا;
    atomic_uint      مکمل;
};

ساخت hmx_قطار_s {
    ساخت hmx_قطار_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    غیرعلامتی صحیح     idx_نکالو;   // updated by producer (pop)
    uint32_t         idx_نقاب;
    uint32_t         استعداد;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    خالی *           ڈھیر;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    ساخت htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

قسمتعریف ساخت hmx_قطار_s * hmx_قطار_t;

size_t      hmx_قطار_sizeof(size_t استعداد, uint32_t ڈھیر_سائز);
size_t      hmx_قطار_alignof(خالی);
hmx_قطار_t hmx_قطار_init(خالی * ptr, size_t استعداد, uint32_t ڈھیر_سائز, uint32_t hap_rctx, ساخت htp_thread_trace * trace);
خالی        hmx_قطار_free(hmx_قطار_t q);

ساکن اندرونی ساخت hmx_قطار_desc hmx_قطار_make_desc(hmx_قطار_func func, خالی * ڈیٹا) {
    ساخت hmx_قطار_desc d = { func, ڈیٹا };
    واپس d;
}

ساکن اندرونی bool hmx_قطار_ڈالو(hmx_قطار_t q, ساخت hmx_قطار_desc d) {
    غیرعلامتی صحیح ir = atomic_بھرو(&q->idx_read);
    غیرعلامتی صحیح iw = atomic_بھرو(&q->idx_write);

    اگر (((iw + 1) & q->idx_نقاب) == ir) {
        FARF(بلند_2, "hmx-queue-push: queue is full\n");
        واپس false;
    }

    atomic_store(&d.مکمل, 0);

    FARF(بلند_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.ڈیٹا);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_نقاب);
    // wake up our thread
    atomic_fetch_جوڑو(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    واپس true;
}

ساکن اندرونی bool hmx_قطار_signal(hmx_قطار_t q, شمار hmx_قطار_signal sig) {
    واپس hmx_قطار_ڈالو(q, hmx_قطار_make_desc((hmx_قطار_func) sig, تہی));
}

ساکن اندرونی bool hmx_قطار_خالی_خانے(hmx_قطار_t q) {
    واپس q->idx_نکالو == atomic_بھرو(&q->idx_write);
}

ساکن اندرونی uint32_t hmx_قطار_گہرائی(hmx_قطار_t q) {
    واپس (atomic_بھرو(&q->idx_write) - atomic_بھرو(&q->idx_read)) & q->idx_نقاب;
}

ساکن اندرونی uint32_t hmx_قطار_استعداد(hmx_قطار_t q) {
    واپس q->استعداد;
}

ساکن اندرونی ساخت hmx_قطار_desc hmx_قطار_نکالو_one(hmx_قطار_t q) {
    غیرعلامتی صحیح ip = q->idx_نکالو;
    غیرعلامتی صحیح iw = atomic_بھرو(&q->idx_write);

    ساخت hmx_قطار_desc rd = { تہی, تہی };
    اگر (ip == iw) {
        واپس rd;
    }

    // Wait for desc to complete
    ساخت hmx_قطار_desc * d = &q->desc[ip];
    جب (!atomic_بھرو(&d->مکمل)) {
        FARF(بلند_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_نکالو = (ip + 1) & q->idx_نقاب;

    FARF(بلند_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.ڈیٹا);
    واپس rd;
}

ساکن اندرونی ساخت hmx_قطار_desc hmx_قطار_نکالو(hmx_قطار_t q) {
    جب (1) {
        ساخت hmx_قطار_desc d = hmx_قطار_نکالو_one(q);

        uint32_t sig = (uint32_t) d.func;
        اگر (sig && sig <= HMX_قطار_KILL)
            جاری;

        واپس d;
    }
}

ساکن اندرونی خالی hmx_قطار_flush(hmx_قطار_t q) {
    جب (hmx_قطار_نکالو_one(q).func != تہی) ;
}

ساکن اندرونی خالی hmx_قطار_wakeup(hmx_قطار_t q) {
    hmx_قطار_signal(q, HMX_قطار_WAKEUP);
}

ساکن اندرونی خالی hmx_قطار_suspend(hmx_قطار_t q) {
    hmx_قطار_signal(q, HMX_قطار_SUSPEND);
}

#اگر_تعریف_شدہ __cplusplus
}  // extern "C"
#اگر_ختم

#اگر_ختم /* HMX_QUEUE_H */
