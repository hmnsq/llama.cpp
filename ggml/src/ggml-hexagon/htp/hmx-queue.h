#నిర్వచితం_కాకపోతే HMX_వరుస_H
#నిర్వచించు HMX_వరుస_H

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stdatomic.h>

#అన్తర్భావ <hexagon_types.h>
#అన్తర్భావ <qurt_thread.h>
#అన్తర్భావ <qurt_futex.h>
#అన్తర్భావ <HAP_farf.h>

#అన్తర్భావ "hex-utils.h"
#అన్తర్భావ "hex-profile.h"

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

#ఒకవేళ __HVX_ARCH__ > 79
#నిర్వచించు HMX_వరుస_POLL_లెక్క 2000
#లేకపోతే
#నిర్వచించు HMX_వరుస_POLL_లెక్క 1
#అయితే_ముగింపు

రకనిర్వచనం శూన్య (*hmx_వరుస_func)(శూన్య *);

// Dummy funcs used as signals
గణన hmx_వరుస_signal {
    HMX_వరుస_NOOP = 0, // aka NULL
    HMX_వరుస_WAKEUP,
    HMX_వరుస_SUSPEND,
    HMX_వరుస_KILL
};

నిర్మాణం hmx_వరుస_desc {
    hmx_వరుస_func   func;
    శూన్య *           దత్తాంశం;
    atomic_uint      పూర్తి;
};

నిర్మాణం hmx_వరుస_s {
    నిర్మాణం hmx_వరుస_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    గుర్తులేని పూర్ణాంకం     idx_తీయి;   // updated by producer (pop)
    uint32_t         idx_ముసుగు;
    uint32_t         సామర్థ్యం;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    శూన్య *           దొంతర;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    నిర్మాణం htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

రకనిర్వచనం నిర్మాణం hmx_వరుస_s * hmx_వరుస_t;

size_t      hmx_వరుస_sizeof(size_t సామర్థ్యం, uint32_t దొంతర_కొలత);
size_t      hmx_వరుస_alignof(శూన్య);
hmx_వరుస_t hmx_వరుస_init(శూన్య * ptr, size_t సామర్థ్యం, uint32_t దొంతర_కొలత, uint32_t hap_rctx, నిర్మాణం htp_thread_trace * trace);
శూన్య        hmx_వరుస_free(hmx_వరుస_t q);

స్థిర అంతఃవరుస నిర్మాణం hmx_వరుస_desc hmx_వరుస_make_desc(hmx_వరుస_func func, శూన్య * దత్తాంశం) {
    నిర్మాణం hmx_వరుస_desc d = { func, దత్తాంశం };
    తిరిగి d;
}

స్థిర అంతఃవరుస bool hmx_వరుస_నెట్టు(hmx_వరుస_t q, నిర్మాణం hmx_వరుస_desc d) {
    గుర్తులేని పూర్ణాంకం ir = atomic_ఎక్కించు(&q->idx_read);
    గుర్తులేని పూర్ణాంకం iw = atomic_ఎక్కించు(&q->idx_write);

    ఒకవేళ (((iw + 1) & q->idx_ముసుగు) == ir) {
        FARF(ఎగువ_2, "hmx-queue-push: queue is full\n");
        తిరిగి false;
    }

    atomic_store(&d.పూర్తి, 0);

    FARF(ఎగువ_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.దత్తాంశం);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_ముసుగు);
    // wake up our thread
    atomic_fetch_కూడు(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    తిరిగి true;
}

స్థిర అంతఃవరుస bool hmx_వరుస_signal(hmx_వరుస_t q, గణన hmx_వరుస_signal sig) {
    తిరిగి hmx_వరుస_నెట్టు(q, hmx_వరుస_make_desc((hmx_వరుస_func) sig, ఖాళీ));
}

స్థిర అంతఃవరుస bool hmx_వరుస_ఖాళీలు(hmx_వరుస_t q) {
    తిరిగి q->idx_తీయి == atomic_ఎక్కించు(&q->idx_write);
}

స్థిర అంతఃవరుస uint32_t hmx_వరుస_లోతు(hmx_వరుస_t q) {
    తిరిగి (atomic_ఎక్కించు(&q->idx_write) - atomic_ఎక్కించు(&q->idx_read)) & q->idx_ముసుగు;
}

స్థిర అంతఃవరుస uint32_t hmx_వరుస_సామర్థ్యం(hmx_వరుస_t q) {
    తిరిగి q->సామర్థ్యం;
}

స్థిర అంతఃవరుస నిర్మాణం hmx_వరుస_desc hmx_వరుస_తీయి_one(hmx_వరుస_t q) {
    గుర్తులేని పూర్ణాంకం ip = q->idx_తీయి;
    గుర్తులేని పూర్ణాంకం iw = atomic_ఎక్కించు(&q->idx_write);

    నిర్మాణం hmx_వరుస_desc rd = { ఖాళీ, ఖాళీ };
    ఒకవేళ (ip == iw) {
        తిరిగి rd;
    }

    // Wait for desc to complete
    నిర్మాణం hmx_వరుస_desc * d = &q->desc[ip];
    అయితే (!atomic_ఎక్కించు(&d->పూర్తి)) {
        FARF(ఎగువ_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_తీయి = (ip + 1) & q->idx_ముసుగు;

    FARF(ఎగువ_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.దత్తాంశం);
    తిరిగి rd;
}

స్థిర అంతఃవరుస నిర్మాణం hmx_వరుస_desc hmx_వరుస_తీయి(hmx_వరుస_t q) {
    అయితే (1) {
        నిర్మాణం hmx_వరుస_desc d = hmx_వరుస_తీయి_one(q);

        uint32_t sig = (uint32_t) d.func;
        ఒకవేళ (sig && sig <= HMX_వరుస_KILL)
            కొనసాగు;

        తిరిగి d;
    }
}

స్థిర అంతఃవరుస శూన్య hmx_వరుస_flush(hmx_వరుస_t q) {
    అయితే (hmx_వరుస_తీయి_one(q).func != ఖాళీ) ;
}

స్థిర అంతఃవరుస శూన్య hmx_వరుస_wakeup(hmx_వరుస_t q) {
    hmx_వరుస_signal(q, HMX_వరుస_WAKEUP);
}

స్థిర అంతఃవరుస శూన్య hmx_వరుస_suspend(hmx_వరుస_t q) {
    hmx_వరుస_signal(q, HMX_వరుస_SUSPEND);
}

#నిర్వచితమైతే __cplusplus
}  // extern "C"
#అయితే_ముగింపు

#అయితే_ముగింపు /* HMX_QUEUE_H */
