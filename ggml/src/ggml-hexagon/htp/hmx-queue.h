#अगर_अघोषित HMX_QUEUE_H
#घोषणा HMX_QUEUE_H

#समावेश <stdbool.h>
#समावेश <stdint.h>
#समावेश <stdatomic.h>

#समावेश <hexagon_types.h>
#समावेश <qurt_thread.h>
#समावेश <qurt_futex.h>
#समावेश <HAP_farf.h>

#समावेश "hex-utils.h"
#समावेश "hex-profile.h"

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

#अगर __HVX_ARCH__ > 79
#घोषणा HMX_QUEUE_POLL_गिनती 2000
#अन्यथा
#घोषणा HMX_QUEUE_POLL_गिनती 1
#पूर्ण_अगर

प्रकार व्योम (*hmx_queue_func)(व्योम *);

// Dummy funcs used as signals
क्रमागत hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

काष्ठा hmx_queue_desc {
    hmx_queue_func   func;
    व्योम *           तथ्य;
    atomic_uint      done;
};

काष्ठा hmx_queue_s {
    काष्ठा hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    अचिन्हित पूर्णांक     idx_निकालो;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    व्योम *           ढेर_2;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    काष्ठा htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

प्रकार काष्ठा hmx_queue_s * hmx_queue_t;

माप_प्रकार      hmx_queue_sizeof(माप_प्रकार capacity, uint32_t ढेर_आकार);
माप_प्रकार      hmx_queue_alignof(व्योम);
hmx_queue_t hmx_queue_init(व्योम * ptr, माप_प्रकार capacity, uint32_t ढेर_आकार, uint32_t hap_rctx, काष्ठा htp_thread_trace * trace);
व्योम        hmx_queue_free(hmx_queue_t q);

अटल अंतरभूत काष्ठा hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, व्योम * तथ्य) {
    काष्ठा hmx_queue_desc d = { func, तथ्य };
    वापस d;
}

अटल अंतरभूत bool hmx_queue_धकेलो(hmx_queue_t q, काष्ठा hmx_queue_desc d) {
    अचिन्हित पूर्णांक ir = atomic_load(&q->idx_read);
    अचिन्हित पूर्णांक iw = atomic_load(&q->idx_write);

    अगर (((iw + 1) & q->idx_mask) == ir) {
        FARF(ऊपरी_2, "hmx-queue-push: queue is full\n");
        वापस false;
    }

    atomic_store(&d.done, 0);

    FARF(ऊपरी_2, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.तथ्य);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    वापस true;
}

अटल अंतरभूत bool hmx_queue_signal(hmx_queue_t q, क्रमागत hmx_queue_signal sig) {
    वापस hmx_queue_धकेलो(q, hmx_queue_make_desc((hmx_queue_func) sig, शून्य));
}

अटल अंतरभूत bool hmx_queue_empty(hmx_queue_t q) {
    वापस q->idx_निकालो == atomic_load(&q->idx_write);
}

अटल अंतरभूत uint32_t hmx_queue_depth(hmx_queue_t q) {
    वापस (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

अटल अंतरभूत uint32_t hmx_queue_capacity(hmx_queue_t q) {
    वापस q->capacity;
}

अटल अंतरभूत काष्ठा hmx_queue_desc hmx_queue_निकालो_one(hmx_queue_t q) {
    अचिन्हित पूर्णांक ip = q->idx_निकालो;
    अचिन्हित पूर्णांक iw = atomic_load(&q->idx_write);

    काष्ठा hmx_queue_desc rd = { शून्य, शून्य };
    अगर (ip == iw) {
        वापस rd;
    }

    // Wait for desc to complete
    काष्ठा hmx_queue_desc * d = &q->desc[ip];
    जबतक (!atomic_load(&d->done)) {
        FARF(ऊपरी_2, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_निकालो = (ip + 1) & q->idx_mask;

    FARF(ऊपरी_2, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.तथ्य);
    वापस rd;
}

अटल अंतरभूत काष्ठा hmx_queue_desc hmx_queue_निकालो(hmx_queue_t q) {
    जबतक (1) {
        काष्ठा hmx_queue_desc d = hmx_queue_निकालो_one(q);

        uint32_t sig = (uint32_t) d.func;
        अगर (sig && sig <= HMX_QUEUE_KILL)
            जारी;

        वापस d;
    }
}

अटल अंतरभूत व्योम hmx_queue_flush(hmx_queue_t q) {
    जबतक (hmx_queue_निकालो_one(q).func != शून्य) ;
}

अटल अंतरभूत व्योम hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

अटल अंतरभूत व्योम hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#अगर_घोषित __cplusplus
}  // extern "C"
#पूर्ण_अगर

#पूर्ण_अगर /* HMX_QUEUE_H */
