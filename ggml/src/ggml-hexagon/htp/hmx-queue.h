#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HMX_ਕਤਾਰ_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_ਕਤਾਰ_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdatomic.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <hexagon_types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_thread.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_futex.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

#ਜੇ __HVX_ARCH__ > 79
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_ਕਤਾਰ_POLL_ਗਿਣਤੀ 2000
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_ਕਤਾਰ_POLL_ਗਿਣਤੀ 1
#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*hmx_ਕਤਾਰ_func)(ਖਾਲੀ *);

// Dummy funcs used as signals
ਗਿਣਤੀ hmx_ਕਤਾਰ_signal {
    HMX_ਕਤਾਰ_NOOP = 0, // aka NULL
    HMX_ਕਤਾਰ_WAKEUP,
    HMX_ਕਤਾਰ_SUSPEND,
    HMX_ਕਤਾਰ_KILL
};

ਬਣਤਰ hmx_ਕਤਾਰ_desc {
    hmx_ਕਤਾਰ_func   func;
    ਖਾਲੀ *           ਅੰਕੜੇ;
    atomic_uint      ਪੂਰਾ;
};

ਬਣਤਰ hmx_ਕਤਾਰ_s {
    ਬਣਤਰ hmx_ਕਤਾਰ_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ     idx_ਕੱਢੋ;   // updated by producer (pop)
    uint32_t         idx_ਨਕਾਬ;
    uint32_t         ਸਮਰੱਥਾ;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    ਖਾਲੀ *           ਢੇਰ;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    ਬਣਤਰ htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ hmx_ਕਤਾਰ_s * hmx_ਕਤਾਰ_t;

size_t      hmx_ਕਤਾਰ_sizeof(size_t ਸਮਰੱਥਾ, uint32_t ਢੇਰ_ਮਾਪ);
size_t      hmx_ਕਤਾਰ_alignof(ਖਾਲੀ);
hmx_ਕਤਾਰ_t hmx_ਕਤਾਰ_init(ਖਾਲੀ * ptr, size_t ਸਮਰੱਥਾ, uint32_t ਢੇਰ_ਮਾਪ, uint32_t hap_rctx, ਬਣਤਰ htp_thread_trace * trace);
ਖਾਲੀ        hmx_ਕਤਾਰ_free(hmx_ਕਤਾਰ_t q);

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਣਤਰ hmx_ਕਤਾਰ_desc hmx_ਕਤਾਰ_make_desc(hmx_ਕਤਾਰ_func func, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ hmx_ਕਤਾਰ_desc d = { func, ਅੰਕੜੇ };
    ਵਾਪਸ d;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool hmx_ਕਤਾਰ_ਧੱਕੋ(hmx_ਕਤਾਰ_t q, ਬਣਤਰ hmx_ਕਤਾਰ_desc d) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ir = atomic_ਲੱਦੋ(&q->idx_read);
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ iw = atomic_ਲੱਦੋ(&q->idx_write);

    ਜੇ (((iw + 1) & q->idx_ਨਕਾਬ) == ir) {
        FARF(ਉੱਚਾ_2, "hmx-queue-push: queue is full\n");
        ਵਾਪਸ false;
    }

    atomic_store(&d.ਪੂਰਾ, 0);

    FARF(ਉੱਚਾ_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.ਅੰਕੜੇ);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_ਨਕਾਬ);
    // wake up our thread
    atomic_fetch_ਜੋੜੋ(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    ਵਾਪਸ true;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool hmx_ਕਤਾਰ_signal(hmx_ਕਤਾਰ_t q, ਗਿਣਤੀ hmx_ਕਤਾਰ_signal sig) {
    ਵਾਪਸ hmx_ਕਤਾਰ_ਧੱਕੋ(q, hmx_ਕਤਾਰ_make_desc((hmx_ਕਤਾਰ_func) sig, ਨਿਰਾ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool hmx_ਕਤਾਰ_ਖਾਲੀ_ਥਾਵਾਂ(hmx_ਕਤਾਰ_t q) {
    ਵਾਪਸ q->idx_ਕੱਢੋ == atomic_ਲੱਦੋ(&q->idx_write);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t hmx_ਕਤਾਰ_ਡੂੰਘਾਈ(hmx_ਕਤਾਰ_t q) {
    ਵਾਪਸ (atomic_ਲੱਦੋ(&q->idx_write) - atomic_ਲੱਦੋ(&q->idx_read)) & q->idx_ਨਕਾਬ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t hmx_ਕਤਾਰ_ਸਮਰੱਥਾ(hmx_ਕਤਾਰ_t q) {
    ਵਾਪਸ q->ਸਮਰੱਥਾ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਣਤਰ hmx_ਕਤਾਰ_desc hmx_ਕਤਾਰ_ਕੱਢੋ_one(hmx_ਕਤਾਰ_t q) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ip = q->idx_ਕੱਢੋ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ iw = atomic_ਲੱਦੋ(&q->idx_write);

    ਬਣਤਰ hmx_ਕਤਾਰ_desc rd = { ਨਿਰਾ, ਨਿਰਾ };
    ਜੇ (ip == iw) {
        ਵਾਪਸ rd;
    }

    // Wait for desc to complete
    ਬਣਤਰ hmx_ਕਤਾਰ_desc * d = &q->desc[ip];
    ਜਦੋਂ (!atomic_ਲੱਦੋ(&d->ਪੂਰਾ)) {
        FARF(ਉੱਚਾ_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_ਕੱਢੋ = (ip + 1) & q->idx_ਨਕਾਬ;

    FARF(ਉੱਚਾ_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.ਅੰਕੜੇ);
    ਵਾਪਸ rd;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਣਤਰ hmx_ਕਤਾਰ_desc hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_ਕਤਾਰ_t q) {
    ਜਦੋਂ (1) {
        ਬਣਤਰ hmx_ਕਤਾਰ_desc d = hmx_ਕਤਾਰ_ਕੱਢੋ_one(q);

        uint32_t sig = (uint32_t) d.func;
        ਜੇ (sig && sig <= HMX_ਕਤਾਰ_KILL)
            ਜਾਰੀ;

        ਵਾਪਸ d;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_ਕਤਾਰ_flush(hmx_ਕਤਾਰ_t q) {
    ਜਦੋਂ (hmx_ਕਤਾਰ_ਕੱਢੋ_one(q).func != ਨਿਰਾ) ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_ਕਤਾਰ_wakeup(hmx_ਕਤਾਰ_t q) {
    hmx_ਕਤਾਰ_signal(q, HMX_ਕਤਾਰ_WAKEUP);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_ਕਤਾਰ_suspend(hmx_ਕਤਾਰ_t q) {
    hmx_ਕਤਾਰ_signal(q, HMX_ਕਤਾਰ_SUSPEND);
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}  // extern "C"
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ /* HMX_QUEUE_H */
