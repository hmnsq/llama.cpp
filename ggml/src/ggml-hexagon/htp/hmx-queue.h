#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HMX_ಸರತಿ_H
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_ಸರತಿ_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdatomic.h>

#ಅನ್ತರ್ಭಾವ <hexagon_types.h>
#ಅನ್ತರ್ಭಾವ <qurt_thread.h>
#ಅನ್ತರ್ಭಾವ <qurt_futex.h>
#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ __HVX_ARCH__ > 79
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_ಸರತಿ_POLL_ಎಣಿಕೆ 2000
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_ಸರತಿ_POLL_ಎಣಿಕೆ 1
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*hmx_ಸರತಿ_func)(ಶೂನ್ಯ *);

// Dummy funcs used as signals
ಎಣಿಕೆ hmx_ಸರತಿ_signal {
    HMX_ಸರತಿ_NOOP = 0, // aka NULL
    HMX_ಸರತಿ_WAKEUP,
    HMX_ಸರತಿ_SUSPEND,
    HMX_ಸರತಿ_KILL
};

ರಚನೆ hmx_ಸರತಿ_desc {
    hmx_ಸರತಿ_func   func;
    ಶೂನ್ಯ *           ದತ್ತಾಂಶ;
    atomic_uint      ಮುಗಿದ;
};

ರಚನೆ hmx_ಸರತಿ_s {
    ರಚನೆ hmx_ಸರತಿ_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ     idx_ತೆಗೆ;   // updated by producer (pop)
    uint32_t         idx_ಮುಖವಾಡ;
    uint32_t         ಸಾಮರ್ಥ್ಯ;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    ಶೂನ್ಯ *           ರಾಶಿ;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    ರಚನೆ htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ hmx_ಸರತಿ_s * hmx_ಸರತಿ_t;

size_t      hmx_ಸರತಿ_sizeof(size_t ಸಾಮರ್ಥ್ಯ, uint32_t ರಾಶಿ_ಗಾತ್ರ);
size_t      hmx_ಸರತಿ_alignof(ಶೂನ್ಯ);
hmx_ಸರತಿ_t hmx_ಸರತಿ_init(ಶೂನ್ಯ * ptr, size_t ಸಾಮರ್ಥ್ಯ, uint32_t ರಾಶಿ_ಗಾತ್ರ, uint32_t hap_rctx, ರಚನೆ htp_thread_trace * trace);
ಶೂನ್ಯ        hmx_ಸರತಿ_free(hmx_ಸರತಿ_t q);

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ರಚನೆ hmx_ಸರತಿ_desc hmx_ಸರತಿ_make_desc(hmx_ಸರತಿ_func func, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ hmx_ಸರತಿ_desc d = { func, ದತ್ತಾಂಶ };
    ಹಿಂತಿರುಗಿ d;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool hmx_ಸರತಿ_ತಳ್ಳು(hmx_ಸರತಿ_t q, ರಚನೆ hmx_ಸರತಿ_desc d) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ir = atomic_ತುಂಬು(&q->idx_read);
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ iw = atomic_ತುಂಬು(&q->idx_write);

    ಆದರೆ (((iw + 1) & q->idx_ಮುಖವಾಡ) == ir) {
        FARF(ಮೇಲಿನ_2, "hmx-queue-push: queue is full\n");
        ಹಿಂತಿರುಗಿ false;
    }

    atomic_store(&d.ಮುಗಿದ, 0);

    FARF(ಮೇಲಿನ_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.ದತ್ತಾಂಶ);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_ಮುಖವಾಡ);
    // wake up our thread
    atomic_fetch_ಕೂಡಿಸು(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    ಹಿಂತಿರುಗಿ true;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool hmx_ಸರತಿ_signal(hmx_ಸರತಿ_t q, ಎಣಿಕೆ hmx_ಸರತಿ_signal sig) {
    ಹಿಂತಿರುಗಿ hmx_ಸರತಿ_ತಳ್ಳು(q, hmx_ಸರತಿ_make_desc((hmx_ಸರತಿ_func) sig, ಖಾಲಿ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool hmx_ಸರತಿ_ಖಾಲಿಗಳು(hmx_ಸರತಿ_t q) {
    ಹಿಂತಿರುಗಿ q->idx_ತೆಗೆ == atomic_ತುಂಬು(&q->idx_write);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t hmx_ಸರತಿ_ಆಳ(hmx_ಸರತಿ_t q) {
    ಹಿಂತಿರುಗಿ (atomic_ತುಂಬು(&q->idx_write) - atomic_ತುಂಬು(&q->idx_read)) & q->idx_ಮುಖವಾಡ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t hmx_ಸರತಿ_ಸಾಮರ್ಥ್ಯ(hmx_ಸರತಿ_t q) {
    ಹಿಂತಿರುಗಿ q->ಸಾಮರ್ಥ್ಯ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ರಚನೆ hmx_ಸರತಿ_desc hmx_ಸರತಿ_ತೆಗೆ_one(hmx_ಸರತಿ_t q) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ip = q->idx_ತೆಗೆ;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ iw = atomic_ತುಂಬು(&q->idx_write);

    ರಚನೆ hmx_ಸರತಿ_desc rd = { ಖಾಲಿ, ಖಾಲಿ };
    ಆದರೆ (ip == iw) {
        ಹಿಂತಿರುಗಿ rd;
    }

    // Wait for desc to complete
    ರಚನೆ hmx_ಸರತಿ_desc * d = &q->desc[ip];
    ಆಗ (!atomic_ತುಂಬು(&d->ಮುಗಿದ)) {
        FARF(ಮೇಲಿನ_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_ತೆಗೆ = (ip + 1) & q->idx_ಮುಖವಾಡ;

    FARF(ಮೇಲಿನ_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.ದತ್ತಾಂಶ);
    ಹಿಂತಿರುಗಿ rd;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ರಚನೆ hmx_ಸರತಿ_desc hmx_ಸರತಿ_ತೆಗೆ(hmx_ಸರತಿ_t q) {
    ಆಗ (1) {
        ರಚನೆ hmx_ಸರತಿ_desc d = hmx_ಸರತಿ_ತೆಗೆ_one(q);

        uint32_t sig = (uint32_t) d.func;
        ಆದರೆ (sig && sig <= HMX_ಸರತಿ_KILL)
            ಮುಂದುವರಿ;

        ಹಿಂತಿರುಗಿ d;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_ಸರತಿ_flush(hmx_ಸರತಿ_t q) {
    ಆಗ (hmx_ಸರತಿ_ತೆಗೆ_one(q).func != ಖಾಲಿ) ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_ಸರತಿ_wakeup(hmx_ಸರತಿ_t q) {
    hmx_ಸರತಿ_signal(q, HMX_ಸರತಿ_WAKEUP);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_ಸರತಿ_suspend(hmx_ಸರತಿ_t q) {
    hmx_ಸರತಿ_signal(q, HMX_ಸರತಿ_SUSPEND);
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}  // extern "C"
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ /* HMX_QUEUE_H */
