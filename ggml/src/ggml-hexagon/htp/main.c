#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#અન્તર્ભાવ <HAP_farf.h>
#અન્તર્ભાવ <HAP_perf.h>
#અન્તર્ભાવ <AEEStdErr.h>
#અન્તર્ભાવ <dspqueue.h>
#અન્તર્ભાવ <HAP_compute_res.h>
#અન્તર્ભાવ <HAP_etm_config.h>
#અન્તર્ભાવ <HAP_mem.h>
#અન્તર્ભાવ <HAP_power.h>
#અન્તર્ભાવ <HAP_ps.h>
#અન્તર્ભાવ <HAP_dcvs.h>
#અન્તર્ભાવ <qurt.h>
#અન્તર્ભાવ <qurt_thread.h>
#અન્તર્ભાવ <qurt_memory.h>
#અન્તર્ભાવ <remote.h>
#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <stdatomic.h>

#અન્તર્ભાવ "hex-utils.h"
#અન્તર્ભાવ "dma-queue.h"
#અન્તર્ભાવ "hmx-queue.h"

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "hex-bitmap.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "htp-dcache.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp_iface.h"
#અન્તર્ભાવ "work-queue.h"
#અન્તર્ભાવ "hex-profile.h"
#અન્તર્ભાવ "allreduce-ops.h"
#અન્તર્ભાવ "htp-fence.h"

#define HMX_QUEUE_CAPACITY     128
#define HMX_QUEUE_ઢગલો_SIZE   16384
#define WORK_QUEUE_CAPACITY    16
#define WORK_QUEUE_ઢગલો_SIZE  16384
#define MAIN_THREAD_ઢગલો_SIZE 32768

_Static_assert(WORK_QUEUE_MAX_N_THREADS >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

સંરચના htp_handle {
    સંરચના htp_context * ctx;
};

સ્થિર અંતઃરેખ uint64_t htp_mmap(uint32_t fd, uint64_t size, uint32_t flags) {
#જો __HVX_ARCH__ > 79
    જો (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, કદ(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = size;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        જો (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (unsigned long long) size);
            પાછા 0;
        }

        પાછા payload.mmap.dsp_va;
    }
#નહીંતર
    જો (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        પાછા 0;
    }
#endif

    જો (size > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (unsigned long long) size);
        પાછા 0;
    }

    શૂન્ય * va = (શૂન્ય *)-1;
    માટે (પૂર્ણાંક retry = 0; retry < 2; retry++) {
#જો __HVX_ARCH__ > 73
        va = HAP_mmap2(ખાલી, (size_t) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#નહીંતર
        જો (size > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (unsigned long long) size);
            abort();
        }
        va = HAP_mmap(ખાલી, (પૂર્ણાંક) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
        જો (va != (શૂન્ય *)-1 && va != ખાલી) {
            પાછા (uint64_t) (uintptr_t) va;
        }
        જો (retry == 0) {
            FARF(ઉપલું_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (unsigned long long) size);
        }
    }
    પાછા 0;
}

સ્થિર અંતઃરેખ શૂન્ય htp_munmap(uint64_t va, uint64_t size, uint32_t flags) {
#જો __HVX_ARCH__ > 79
    જો (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, કદ(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = size;
        HAP_mem_request(&payload);
        પાછા;
    }
#endif

#જો __HVX_ARCH__ > 73
    HAP_munmap2((શૂન્ય *) (uintptr_t) va, (size_t) size);
#નહીંતર
    HAP_munmap((શૂન્ય *) (uintptr_t) va, (પૂર્ણાંક) size);
#endif
}

AEEResult htp_iface_open(અચલ અક્ષર * uri, remote_handle64 * handle) {
    (શૂન્ય) uri;
    સંરચના htp_handle * h = calloc(1, કદ(*h));
    જો (h == ખાલી) {
        પાછા AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    પાછા AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    સંરચના htp_handle * h = (સંરચના htp_handle *) handle;
    જો (!h) {
        પાછા AEE_EBADPARM;
    }

    પૂર્ણાંક err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    જો (err) {
        જો (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } નહીંતર {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    પાછા err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, અચલ htp_iface_pmu_conf* pmu_conf) {
    સંરચના htp_handle * h = (સંરચના htp_handle *) handle;
    જો (!h || !h->ctx) {
        પાછા AEE_EBADPARM;
    }
    સંરચના htp_context * ctx = h->ctx;

    જો (mode == HTP_PROF_PMU) {
        અચલ uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        માટે (; i < HEX_અંક_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        માટે (i = 0; i < HEX_અંક_PMU_COUNTERS; i++) {
            cfg |= (((events[i] >> 8) & 3) << (i * 2));
        }

        FARF(ALWAYS, "Configuring PMU registers: evtcfg = 0x%x, evtcfg1 = 0x%x, pmucfg = 0x%x", evtcfg, evtcfg1, cfg);

        // Configure PMU registers
        qurt_pmu_set(QURT_PMUCFG,     cfg);
        qurt_pmu_set(QURT_PMUEVTCFG,  evtcfg);
        qurt_pmu_set(QURT_PMUEVTCFG1, evtcfg1);
        qurt_pmu_enable(1);
    }

    ctx->profiler = mode;

    પાછા AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    સંરચના htp_handle * h = (સંરચના htp_handle *) handle;
    જો (!h) {
        પાછા AEE_EBADPARM;
    }

    સંરચના htp_context * ctx = h->ctx;
    જો (ctx) {
        જો (ctx->dsp_queue) {
            FARF(ERROR, "Closing handle with queue still open");
            પાછા AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        માટે (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            જો (ctx->mmap[i].size) {
                htp_munmap(ctx->mmap[i].base, ctx->mmap[i].size, ctx->mmap[i].flags);
                ctx->mmap[i].size = 0;
                ctx->mmap[i].base = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        જો (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        જો (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        મુક્ત(ctx);
        h->ctx = ખાલી;
    }

    મુક્ત(h);
    પાછા AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t size) {
    સંરચના htp_handle * h = (સંરચના htp_handle *) handle;
    જો (!h || !h->ctx) {
        પાછા AEE_EBADPARM;
    }
    સંરચના htp_context * ctx = h->ctx;

    // See if we already have this mapping
    માટે (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        સંરચના htp_mmap *m = &ctx->mmap[i];
        જો (m->fd == fd) {
            પાછા AEE_SUCCESS;
        }
    }

    // Add new mapping
    માટે (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        સંરચના htp_mmap *m = &ctx->mmap[i];
        જો (!m->size) {
            FARF(ઉપલું_2, "mmap : fd %u size %llu", fd, (unsigned long long) size);
            uint64_t va = htp_mmap(fd, size, 0);
            જો (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (unsigned long long) size);
                પાછા AEE_EFAILED;
            }

            m->base   = va;
            m->fd     = fd;
            m->size   = size;
            m->flags  = 0;

            પાછા AEE_SUCCESS;
        }
    }

    પાછા AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    સંરચના htp_handle * h = (સંરચના htp_handle *) handle;
    જો (!h || !h->ctx) {
        પાછા AEE_EBADPARM;
    }
    સંરચના htp_context * ctx = h->ctx;

    માટે (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        સંરચના htp_mmap *m = &ctx->mmap[i];
        જો (fd < 0 || m->fd == fd) {
            FARF(ઉપલું_2, "unmmap : base 0x%llx fd %u size %llu", (unsigned long long) m->base, m->fd, (unsigned long long) m->size);
            htp_munmap(m->base, m->size, m->flags);
            m->size   = 0;
            m->base   = ખાલી;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    પાછા AEE_SUCCESS;
}

સ્થિર શૂન્ય vtcm_acquire(સંરચના htp_context * ctx) {
    જો (!atomic_load(&ctx->vtcm_valid)) {
        પૂર્ણાંક err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        જો (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (unsigned)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

સ્થિર શૂન્ય vtcm_release(સંરચના htp_context * ctx) {
    જો (atomic_load(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

સ્થિર પૂર્ણાંક vtcm_release_callback(unsigned પૂર્ણાંક rctx, શૂન્ય * state) {
    સંરચના htp_context * ctx = (સંરચના htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    પાછા 0;
}

સ્થિર unsigned પૂર્ણાંક vtcm_query_size(શૂન્ય) {
    unsigned પૂર્ણાંક vtcm_size = 8 * 1024 * 1024;  // 8MB default
    unsigned પૂર્ણાંક vtcm_avail;
    compute_res_vtcm_page_t total_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_size, &total_layout, &vtcm_avail, &avail_layout);
    પાછા vtcm_size;
}

સ્થિર પૂર્ણાંક vtcm_alloc(સંરચના htp_context * ctx) {
    unsigned પૂર્ણાંક vtcm_size = vtcm_query_size();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (શૂન્ય *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    જો (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_size);
        પાછા AEE_ENOMEMORY;
    }

    શૂન્ય * vtcm_ptr;
    જો (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_size);
        પાછા AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size          = vtcm_size;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    પાછા 0;
}

સ્થિર શૂન્ય vtcm_free(સંરચના htp_context * ctx) {
    જો (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

સ્થિર શૂન્ય htp_main_thread(શૂન્ય * context);
સ્થિર શૂન્ય htp_packet_callback(dspqueue_t queue, પૂર્ણાંક error, શૂન્ય * context);
સ્થિર શૂન્ય htp_error_callback(dspqueue_t queue, પૂર્ણાંક error, શૂન્ય * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
સંરચના htp_power_profile {
    boolean  dcvs_enable;
    uint32_t bus_min_corner;
    uint32_t bus_max_corner;
    uint32_t bus_target_corner;
    uint32_t core_min_corner;
    uint32_t core_max_corner;
    uint32_t core_target_corner;
    boolean  sleep_disable;
    uint32_t hmx_min_corner;    // only used when __HVX_ARCH__ >= 75
    uint32_t hmx_max_corner;
    uint32_t hmx_target_corner;
    uint32_t hmx_perf_mode;
};

#define HTP_POWER_PROFILE_ENTRY(min_c, max_c, hmx_c) \
    { \
        .dcvs_enable        = TRUE, \
        .bus_min_corner     = (min_c), \
        .bus_max_corner     = (max_c), \
        .bus_target_corner  = (max_c), \
        .core_min_corner    = (min_c), \
        .core_max_corner    = (max_c), \
        .core_target_corner = (max_c), \
        .sleep_disable      = TRUE, \
        .hmx_min_corner     = (hmx_c), \
        .hmx_max_corner     = (hmx_c), \
        .hmx_target_corner  = (hmx_c), \
        .hmx_perf_mode      = HAP_CLK_PERF_ઉપલું, \
    }

સ્થિર અચલ સંરચના htp_power_profile htp_power_profiles[HTP_POWER_MODE_ગણતરી] = {
    [HTP_POWER_MODE_નીચલું_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_MODE_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_MODE_ઉપલું_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_MODE_નીચલું_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_MODE_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_MODE_ઉપલું_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_SUSTAINED_ઉપલું_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_BURST] = {
        .dcvs_enable        = FALSE,
        .bus_min_corner     = HAP_DCVS_VCORNER_MAX,
        .bus_max_corner     = HAP_DCVS_VCORNER_MAX,
        .bus_target_corner  = HAP_DCVS_VCORNER_MAX,
        .core_min_corner    = HAP_DCVS_VCORNER_MAX,
        .core_max_corner    = HAP_DCVS_VCORNER_MAX,
        .core_target_corner = HAP_DCVS_VCORNER_MAX,
        .sleep_disable      = TRUE,
        .hmx_min_corner     = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_max_corner     = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_target_corner  = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_perf_mode      = HAP_CLK_PERF_ઉપલું,
    },
};

#undef HTP_POWER_PROFILE_ENTRY

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_mode) {
    સંરચના htp_handle * h = (સંરચના htp_handle *) handle;
    જો (!h) {
        પાછા AEE_EBADPARM;
    }

    જો (power_mode >= HTP_POWER_MODE_ગણતરી) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (unsigned) power_mode);
        power_mode = HTP_POWER_MODE_BURST;
    }
    અચલ સંરચના htp_power_profile * pwr = &htp_power_profiles[power_mode];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (unsigned) power_mode, (પૂર્ણાંક) pwr->dcvs_enable,
         (unsigned) pwr->bus_min_corner, (unsigned) pwr->bus_max_corner, (unsigned) pwr->bus_target_corner,
         (unsigned) pwr->core_min_corner, (unsigned) pwr->core_max_corner, (unsigned) pwr->core_target_corner,
         (પૂર્ણાંક) pwr->sleep_disable,
         (unsigned) pwr->hmx_min_corner, (unsigned) pwr->hmx_max_corner, (unsigned) pwr->hmx_target_corner);

    જો (h->ctx) {
        FARF(ERROR, "Queue already open");
        પાછા AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    પૂર્ણાંક fastrpc_tid  = qurt_thread_get_id();
    પૂર્ણાંક fastrpc_prio = qurt_thread_get_priority(fastrpc_tid);
    પૂર્ણાંક main_prio    = fastrpc_prio - 10;
    જો (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_queue = ખાલી;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    પૂર્ણાંક err = dspqueue_import(dsp_queue_id, ખાલી, ખાલી, (શૂન્ય *) h, &dsp_queue);
    જો (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(ઉપલું_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (શૂન્ય *) h, &dsp_queue);
    }

    જો (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        પાછા err;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    જો (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    જો (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    જો (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t offset = કદ(સંરચના htp_context);

    // 2. main_stack
    size_t offset_main_ઢગલો = 0;
    size_t size_main_ઢગલો   = 0;
    જો (!use_callbacks) {
        offset_main_ઢગલો = hex_align_up(offset, 4096);
        size_main_ઢગલો   = MAIN_THREAD_ઢગલો_SIZE;
        offset = offset_main_ઢગલો + size_main_ઢગલો;
    }

    // 3. work_queue
    size_t wq_align  = work_queue_alignof();
    size_t offset_wq = hex_align_up(offset, wq_align);
    size_t size_wq   = work_queue_sizeof(n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_ઢગલો_SIZE);
    offset = offset_wq + size_wq;

    // 4. dma_queue
    size_t dma_align = dma_queue_alignof();
    size_t offset_dma = hex_align_up(offset, dma_align);
    size_t size_dma = 0;
    માટે (uint32_t i = 0; i < n_hvx; i++) {
        size_dma  = hex_align_up(size_dma, dma_queue_alignof());
        size_dma += dma_queue_sizeof(256);
    }
    offset = offset_dma + size_dma;

    // 5. hmx_queue
    size_t offset_hmx = 0;
    size_t size_hmx = 0;
    જો (n_hmx) {
        size_t hmx_align = hmx_queue_alignof();
        offset_hmx = hex_align_up(offset, hmx_align);
        size_hmx   = hmx_queue_sizeof(HMX_QUEUE_CAPACITY, HMX_QUEUE_ઢગલો_SIZE);
        offset = offset_hmx + size_hmx;
    }

    size_t footprint = hex_align_up(offset, 128);

    શૂન્ય * block = memalign(4096, footprint);
    જો (!block) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_queue);
        પાછા AEE_ENOMEMORY;
    }
    memset(block, 0, footprint);

    h->ctx = (સંરચના htp_context *) block;
    સંરચના htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_queue   = dsp_queue;

    err = vtcm_alloc(ctx);
    જો (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        પાછા AEE_ENOMEMORY;
    }

    HAP_setFARFRuntimeLoggingParams(0xffff, ખાલી, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, કદ(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        જો ((err = HAP_power_set((શૂન્ય *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            પાછા err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, કદ(request));

        request.type                              = HAP_power_set_DCVS_v3;
        request.dcvs_v3.set_dcvs_enable           = TRUE;
        request.dcvs_v3.dcvs_enable               = pwr->dcvs_enable;
        request.dcvs_v3.set_bus_params            = TRUE;
        request.dcvs_v3.bus_params.min_corner     = pwr->bus_min_corner;
        request.dcvs_v3.bus_params.max_corner     = pwr->bus_max_corner;
        request.dcvs_v3.bus_params.target_corner  = pwr->bus_target_corner;
        request.dcvs_v3.set_core_params           = TRUE;
        request.dcvs_v3.core_params.min_corner    = pwr->core_min_corner;
        request.dcvs_v3.core_params.max_corner    = pwr->core_max_corner;
        request.dcvs_v3.core_params.target_corner = pwr->core_target_corner;
        request.dcvs_v3.set_sleep_disable         = TRUE;
        request.dcvs_v3.sleep_disable             = pwr->sleep_disable;

#જો (__HEXAGON_ARCH__ >= 79)
        HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
#endif
        જો ((err = HAP_power_set((શૂન્ય *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            પાછા err;
        }

        memset(&request, 0, કદ(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        જો ((err = HAP_power_set((શૂન્ય *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            પાછા err;
        }
    }

#જો __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, કદ(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_power     = TRUE;
        request.hmx_v2.power_up      = TRUE;
        request.hmx_v2.set_clock     = TRUE;
        request.hmx_v2.target_corner = pwr->hmx_target_corner;
        request.hmx_v2.min_corner    = pwr->hmx_min_corner;
        request.hmx_v2.max_corner    = pwr->hmx_max_corner;
        request.hmx_v2.perf_mode     = pwr->hmx_perf_mode;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((શૂન્ય *) ctx, &request);
        જો (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            પાછા err;
        }
    }
#નહીંતર
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, કદ(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((શૂન્ય *) ctx, &request);
        જો (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            પાછા err;
        }
    }
#endif

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_queue   = ખાલી;
    જો (n_hmx) {
        શૂન્ય * hmx_ptr = (શૂન્ય *) ((uintptr_t) block + offset_hmx);
        ctx->hmx_queue = hmx_queue_init(hmx_ptr, HMX_QUEUE_CAPACITY, HMX_QUEUE_ઢગલો_SIZE, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(ઉપલું_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_threads = n_hvx;
    ctx->n_threads_div = init_fastdiv_કિંમતો(ctx->n_threads);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) block + offset_dma);
    size_t size_dma_q = dma_queue_sizeof(256);

    માટે (પૂર્ણાંક i = 0; i < ctx->n_threads; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_up((uintptr_t) dma_ptr_curr, dma_queue_alignof());
        ctx->dma[i] = dma_queue_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += size_dma_q;
    }

    ctx->ddr_spad_size = 512 * 1024; // 512 KB
    ctx->ddr_spad_base = memalign(128, ctx->ddr_spad_size);

    શૂન્ય * wq_ptr = (શૂન્ય *) ((uintptr_t) block + offset_wq);
    ctx->work_queue = work_queue_init(wq_ptr, n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_ઢગલો_SIZE);

    ctx->main_ઢગલો = ખાલી;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    જો (!use_callbacks) {
        // Start main compute thread
        ctx->main_ઢગલો = (શૂન્ય *) ((uintptr_t) block + offset_main_ઢગલો);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_ઢગલો_addr(&attr, ctx->main_ઢગલો);
        qurt_thread_attr_set_ઢગલો_size(&attr, size_main_ઢગલો);
        qurt_thread_attr_set_priority(&attr, main_prio);
        qurt_thread_attr_set_name(&attr, "htp-main");

        પૂર્ણાંક err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        જો (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            પાછા AEE_ENOMEMORY;
        }
    }

    FARF(ઉપલું_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_size, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    પાછા AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    સંરચના htp_handle * h = (સંરચના htp_handle *) handle;
    જો (!h || !h->ctx) {
        પાછા AEE_EBADPARM;
    }
    સંરચના htp_context * ctx = h->ctx;

    જો (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        પૂર્ણાંક status;
        (શૂન્ય) qurt_thread_join(ctx->main_thread, &status);
        ctx->main_thread = 0;
    }

    પૂર્ણાંક err = dspqueue_close(ctx->dsp_queue); ctx->dsp_queue = ખાલી;
    જો (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (unsigned) err);
        પાછા err;
    }

    work_queue_free(ctx->work_queue);

    માટે (પૂર્ણાંક i = 0; i < ctx->n_threads; i++) {
        dma_queue_free(ctx->dma[i]);
    }

    જો (ctx->hmx_queue) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = ખાલી;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    જો (ctx->ddr_spad_base) {
        મુક્ત(ctx->ddr_spad_base);
        ctx->ddr_spad_base = ખાલી;
        ctx->ddr_spad_size = 0;
    }

    મુક્ત(ctx);
    h->ctx = ખાલી;

    પાછા AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_threads, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_size) {
    (શૂન્ય)handle;
    જો (!n_threads || !n_hvx || !n_hmx || !vtcm_size) {
        પાછા AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    જો (n_hvx_val > hw_threads.max_hthreads) {
        n_hvx_val = hw_threads.max_hthreads;
    }
    જો (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_threads = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_size = vtcm_query_size();

    પાછા AEE_SUCCESS;
}

સ્થિર શૂન્ય htp_error_callback(dspqueue_t queue, પૂર્ણાંક error, શૂન્ય * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (unsigned) error);
}

સંરચના profile_data {
    uint64_t usecs;
    uint64_t cycles_start;
    uint64_t cycles_stop;
    uint32_t pmu_counters[HEX_અંક_PMU_COUNTERS];
};

સ્થિર અંતઃરેખ શૂન્ય profile_start(uint32_t mode, સંરચના profile_data * d) {
    પસંદગી (mode) {
        સ્થિતિ HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        સ્થિતિ HTP_PROF_BASIC:
        સ્થિતિ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_ગણતરી();
            d->cycles_start = hex_get_cycles();
            તોડો;
        મૂળભૂત:
            તોડો;
    }
}

સ્થિર અંતઃરેખ શૂન્ય profile_stop(uint32_t mode, સંરચના profile_data * d) {
    uint32_t pmu_counters[HEX_અંક_PMU_COUNTERS];
    પસંદગી (mode) {
        સ્થિતિ HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            માટે (પૂર્ણાંક i = 0; i < HEX_અંક_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        સ્થિતિ HTP_PROF_BASIC:
        સ્થિતિ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_ગણતરી_to_us(HAP_perf_get_qtimer_ગણતરી() - d->usecs);
            d->cycles_stop = hex_get_cycles();
            તોડો;
        મૂળભૂત:
            તોડો;
    }
}

સ્થિર પૂર્ણાંક op_fence(સંરચના htp_ops_context * octx) {
    સંરચના htp_context *ctx = octx->ctx;
    સંરચના htp_thread_trace * tr = &ctx->trace[0];
    અચલ uint32_t seq  = (uint32_t) octx->op_params[0];
    અચલ uint32_t mode = (uint32_t) octx->op_params[1];

    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    અચલ સંરચના htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;

    જો (mode == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_group_barrier(octx);

        જો (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->status);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(ઉપલું_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        પાછા octx->status;
    }

    પૂર્ણાંક status = HTP_STATUS_OK;
    uint64_t spins = 0;
    જ્યારે (1) {
        uint32_t sync_seq;
        uint32_t sync_status;
        htp_fence_read(sync_fence, &sync_seq, &sync_status);
        જો ((int32_t)(sync_seq - seq) >= 0) {
            જો (sync_status > HTP_STATUS_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_status, sync_fence, seq);
                status = sync_status;
            }
            તોડો;
        }
        જો (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            status = HTP_STATUS_INTERNAL_ERR;
            તોડો;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(ઉપલું_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    પાછા status;
}

સ્થિર પૂર્ણાંક op_mdev_group(સંરચના htp_ops_context * octx) {
    સંરચના htp_context * ctx = octx->ctx;
    અચલ સંરચના htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.ગણતરી = (uint16_t) sync->ne[1];
    જો (ctx->mdev.ગણતરી > 1) {
        ctx->mdev.ગણતરી_div = init_fastdiv_કિંમતો(ctx->mdev.ગણતરી);
        ctx->mdev.fence_base = (uint8_t *) sync->data;
    }
    પાછા HTP_STATUS_OK;
}

સ્થિર પૂર્ણાંક execute_op(સંરચના htp_ops_context * octx) {
    પસંદગી (octx->op) {
        સ્થિતિ HTP_OP_MDEV_GROUP:
            પાછા op_mdev_group(octx);

        સ્થિતિ HTP_OP_FENCE:
            પાછા op_fence(octx);

        સ્થિતિ HTP_OP_ALLREDUCE:
        સ્થિતિ HTP_OP_ALLREDUCE_ADD:
            પાછા op_allreduce(octx);

        સ્થિતિ HTP_OP_MUL_MAT:
        સ્થિતિ HTP_OP_MUL_MAT_ADD:
            પાછા op_matmul(octx);

        સ્થિતિ HTP_OP_MUL_MAT_ID:
            પાછા op_matmul_id(octx);

        સ્થિતિ HTP_OP_MUL_MAT_ID_NX:
            પાછા op_matmul_id_nx(octx);

        સ્થિતિ HTP_OP_MUL_MAT_NX:
            પાછા op_matmul_nx(octx);

        સ્થિતિ HTP_OP_MUL:
        સ્થિતિ HTP_OP_ADD:
        સ્થિતિ HTP_OP_SUB:
        સ્થિતિ HTP_OP_DIV:
        સ્થિતિ HTP_OP_ADD_ID:
            પાછા op_દ્વિઅંકી(octx);

        સ્થિતિ HTP_OP_NORM:
        સ્થિતિ HTP_OP_RMS_NORM:
        સ્થિતિ HTP_OP_RMS_NORM_MUL:
        સ્થિતિ HTP_OP_SCALE:
        સ્થિતિ HTP_OP_CLAMP:
        સ્થિતિ HTP_OP_LEAKY_RELU:
        સ્થિતિ HTP_OP_SQR:
        સ્થિતિ HTP_OP_SQRT:
        સ્થિતિ HTP_OP_UNARY_SOFTPLUS:
        સ્થિતિ HTP_OP_UNARY_SIGMOID:
        સ્થિતિ HTP_OP_UNARY_SILU:
        સ્થિતિ HTP_OP_UNARY_GELU:
        સ્થિતિ HTP_OP_UNARY_GELU_QUICK:
        સ્થિતિ HTP_OP_UNARY_GELU_ERF:
        સ્થિતિ HTP_OP_UNARY_NEG:
        સ્થિતિ HTP_OP_UNARY_EXP:
        સ્થિતિ HTP_OP_UNARY_TANH:
        સ્થિતિ HTP_OP_UNARY_ABS:
        સ્થિતિ HTP_OP_UNARY_LOG:
        સ્થિતિ HTP_OP_UNARY_RELU:
        સ્થિતિ HTP_OP_UNARY_STEP:
        સ્થિતિ HTP_OP_L2_NORM:
        સ્થિતિ HTP_OP_HADAMARD:
            પાછા op_unary(octx);

        સ્થિતિ HTP_OP_GLU_SWIGLU:
        સ્થિતિ HTP_OP_GLU_SWIGLU_OAI:
        સ્થિતિ HTP_OP_GLU_SWIGLU_CLAMP:
        સ્થિતિ HTP_OP_GLU_GEGLU:
        સ્થિતિ HTP_OP_GLU_GEGLU_QUICK:
        સ્થિતિ HTP_OP_GLU_GEGLU_ERF:
            પાછા op_activations(octx);

        સ્થિતિ HTP_OP_SOFTMAX:
            પાછા op_softmax(octx);

        સ્થિતિ HTP_OP_ROPE:
            પાછા op_rope(octx);

        સ્થિતિ HTP_OP_FLASH_ATTN_EXT:
            પાછા op_flash_attn_ext(octx);

        સ્થિતિ HTP_OP_SET_ROWS:
            પાછા op_set_rows(octx);

        સ્થિતિ HTP_OP_GET_ROWS:
            પાછા op_get_rows(octx);

        સ્થિતિ HTP_OP_SUM:
            પાછા op_sum(octx);

        સ્થિતિ HTP_OP_SUM_ROWS:
            પાછા op_sum_rows(octx);

        સ્થિતિ HTP_OP_CPY:
        સ્થિતિ HTP_OP_CPY_FENCE:
            પાછા op_cpy(octx);

        સ્થિતિ HTP_OP_REPEAT:
            પાછા op_repeat(octx);

        સ્થિતિ HTP_OP_ARGSORT:
            પાછા op_argsort(octx);

        સ્થિતિ HTP_OP_ટોચ_K:
            પાછા op_ટોચ_k(octx);

        સ્થિતિ HTP_OP_ARGMAX:
            પાછા op_argmax(octx);

        સ્થિતિ HTP_OP_SSM_CONV:
            પાછા op_ssm_conv(octx);

        સ્થિતિ HTP_OP_CUMSUM:
            પાછા op_cumsum(octx);

        સ્થિતિ HTP_OP_FILL:
            પાછા op_fill(octx);

        સ્થિતિ HTP_OP_DIAG:
            પાછા op_diag(octx);

        સ્થિતિ HTP_OP_SOLVE_TRI:
            પાછા op_solve_tri(octx);

        સ્થિતિ HTP_OP_PAD:
            પાછા op_pad(octx);

        સ્થિતિ HTP_OP_IM2COL:
            પાછા op_im2col(octx);

        સ્થિતિ HTP_OP_IM2COL_3D:
            પાછા op_im2col_3d(octx);

        સ્થિતિ HTP_OP_ROLL:
            પાછા op_roll(octx);

        સ્થિતિ HTP_OP_POOL_2D:
            પાછા op_pool_2d(octx);

        સ્થિતિ HTP_OP_POOL_1D:
            પાછા op_pool_1d(octx);

        સ્થિતિ HTP_OP_CONCAT:
            પાછા op_concat(octx);

        સ્થિતિ HTP_OP_GATED_DELTA_NET:
            પાછા op_gated_delta_net(octx);

        સ્થિતિ HTP_OP_TRI:
            પાછા op_unary(octx);

        સ્થિતિ HTP_OP_INVALID:
            તોડો;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    પાછા HTP_STATUS_NO_SUPPORT;
}

સ્થિર અંતઃરેખ bool reuse_buf(સંરચના htp_context *ctx, uint32_t *m_reuse, સંરચના htp_buf_desc *b) {
    b->base = ખાલી;

    માટે (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        સંરચના htp_mmap *m = ctx->mmap + i;
        જો (m->size && m->fd == b->fd && m->flags == b->flags) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            પાછા true;
        }
    }

    પાછા false;
}

સ્થિર અંતઃરેખ શૂન્ય drop_mmap(સંરચના htp_context *ctx, સંરચના htp_mmap *m) {
    જો (m->size) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (unsigned long long) m->base, (unsigned long long) m->size);
        htp_munmap(m->base, m->size, m->flags);
        m->size = 0;
        m->base = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

સ્થિર અંતઃરેખ bool mmap_buf(સંરચના htp_context *ctx, સંરચના htp_buf_desc *b) {
    જો (b->base) પાછા true; // already mapped

    // find unused mapping
    માટે (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        સંરચના htp_mmap *m = &ctx->mmap[i];
        જો (!m->size) {
            uint64_t va = htp_mmap(b->fd, b->size, b->flags);
            જો (va == 0) {
                FARF(ઉપલું_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (unsigned long long) b->size);
                પાછા false;
            }

            m->base   = b->base = va;
            m->fd     = b->fd;
            m->size   = b->size;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (unsigned long long) m->base, (unsigned long long) m->size, m->flags);
            પાછા true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    પાછા false;
}

સ્થિર શૂન્ય prep_op_bufs(સંરચના htp_context *ctx, સંરચના htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    માટે (uint32_t i=0; i < n_bufs; i++) {
        સંરચના htp_buf_desc *b = bufs + i;
        જો (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } નહીંતર જો (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->size;
        }
        FARF(ઉપલું_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    જો (b_reuse == n_bufs) પાછા; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    માટે (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        જો (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].size;
        }
    }

    FARF(ઉપલું_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    જો ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        માટે (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            જો (!used && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    માટે (uint32_t i=0; i < n_bufs; i++) {
        સંરચના htp_buf_desc *b = bufs + i;
        જો (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            તોડો;
        }
        FARF(ઉપલું_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    જો (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(ઉપલું_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        માટે (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            જો (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        માટે (uint32_t i=0; i < n_bufs; i++) {
            સંરચના htp_buf_desc *b = bufs + i;
            જો (!(b->flags & HTP_BUF_EXTENDED)) {
                b->base = 0;
            }
            જો (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (unsigned long long) b->size);
                abort();
            }
            FARF(ઉપલું_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
        }
    }
}

સ્થિર શૂન્ય prep_tensor(સંરચના htp_context *ctx, સંરચના htp_buf_desc *bufs, સંરચના htp_tensor *tens, uint32_t idx, સંરચના htp_tensor *t) {
    uint64_t offset = t->data;
    uint32_t bi     = t->bi;

    t->data  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(ઉપલું_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (unsigned long long) offset, t->size, (unsigned long long) t->data,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

સ્થિર શૂન્ય prep_tensors(સંરચના htp_context *ctx, સંરચના htp_buf_desc *bufs, સંરચના htp_tensor *tens, uint32_t n_tens) {
    માટે (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

સ્થિર શૂન્ય mdev_group_init(સંરચના htp_context * ctx, અચલ સંરચના htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, કદ(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

સ્થિર પૂર્ણાંક proc_op_req(સંરચના htp_ops_context * octx, સંરચના htp_buf_desc * bufs, uint32_t n_bufs,
                       સંરચના htp_tensor * tens, uint32_t idx, સંરચના htp_op_desc * op) {
    memcpy(octx->op_params,     op->params, કદ(octx->op_params));
    memcpy(octx->kernel_params, op->kernel_params, કદ(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_threads     = octx->ctx->n_threads;
    octx->n_threads_div = octx->ctx->n_threads_div;

    FARF(ઉપલું_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    માટે (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        જો (src_idx == 0xffff) {
            octx->src[i]     = ખાલી;
            ચાલુ;
        }

        સંરચના htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(ઉપલું_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (unsigned long long) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    માટે (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        જો (dst_idx == 0xffff) {
            octx->dsts[i]    = ખાલી;
            ચાલુ;
        }
        સંરચના htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(ઉપલું_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (unsigned long long) dst->data, dst->size,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_group_barrier(octx);

    પૂર્ણાંક status = execute_op(octx);

    htp_ops_context_set_status(octx, status);

    octx->src0_spad.src = ખાલી;
    octx->src1_spad.src = ખાલી;
    octx->src2_spad.src = ખાલી;
    octx->src3_spad.src = ખાલી;
    octx->dst_spad.src  = ખાલી;

    પાછા octx->status;
}

સ્થિર શૂન્ય process_opbatch(સંરચના htp_context * ctx, અચલ સંરચના htp_opbatch_req * req, અચલ સંરચના dspqueue_buffer * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    પૂર્ણાંક err;

    અચલ uint32_t n_bufs = req->n_bufs;
    અચલ uint32_t n_tens = req->n_tensors;
    અચલ uint32_t n_ops  = req->n_ops;

    અચલ uint32_t b_size = કદ(સંરચના htp_buf_desc)  * n_bufs;
    અચલ uint32_t t_size = કદ(સંરચના htp_tensor)    * n_tens;
    અચલ uint32_t o_size = કદ(સંરચના htp_op_desc)   * n_ops;
    અચલ uint32_t p_size = કદ(સંરચના htp_prof_desc) * n_ops;
    અચલ uint32_t tr_size = (HTP_MAX_NTHREADS + 1) * req->n_traces * કદ(સંરચના htp_trace_desc);

    જો (dbuf->size < b_size + t_size + o_size + p_size + tr_size) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->size, b_size + t_size + o_size + p_size + tr_size);
        પાછા;
    }

    FARF(ઉપલું_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (unsigned long long) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->size, b_size, t_size, o_size);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    સંરચના htp_buf_desc* bufs = (સંરચના htp_buf_desc*)  m_ptr; m_ptr += b_size;
    સંરચના htp_tensor*   tens = (સંરચના htp_tensor*)    m_ptr; m_ptr += t_size;
    સંરચના htp_op_desc*   ops = (સંરચના htp_op_desc*)   m_ptr; m_ptr += o_size;
    સંરચના htp_prof_desc* pds = (સંરચના htp_prof_desc*) m_ptr;

    સંરચના profile_data batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, કદ(ctx->trace));
    જો (ctx->profiler == HTP_PROF_TRACE) {
        સંરચના htp_trace_desc * trace_events = (સંરચના htp_trace_desc *) (m_ptr + p_size);
        માટે (પૂર્ણાંક t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, કદ(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    સંરચના htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, કદ(*octx));
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
    octx->ctx           = ctx;

    mdev_group_init(ctx, req);

    work_queue_wakeup(ctx->work_queue);
    જો (ctx->hmx_queue) {
        hmx_queue_wakeup(ctx->hmx_queue);
    }

    પૂર્ણાંક op_status = HTP_STATUS_OK;
    octx->status  = HTP_STATUS_OK;
    માટે (uint32_t i = 0; i < n_ops; i++) {
        સંરચના profile_data prof;

        profile_start(ctx->profiler, &prof);

        op_status = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_set_status(octx, op_status);

        જો (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].cycles_start = prof.cycles_start;
            pds[i].cycles_stop = prof.cycles_stop;
            માટે (પૂર્ણાંક j = 0; j < HEX_અંક_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    જો (ctx->hmx_queue) {
        hmx_queue_suspend(ctx->hmx_queue);
        hmx_queue_flush(ctx->hmx_queue);
    }
    work_queue_suspend(ctx->work_queue);

    // Flush remaining dirty tensors at the end of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_mdev_group_barrier(octx);

    profile_stop(HTP_PROF_BASIC, &batch_prof);

    સંરચના htp_opbatch_rsp rsp;
    memset(&rsp, 0, કદ(rsp));
    rsp.seq          = req->seq;
    rsp.status       = octx->status;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.cycles_start = batch_prof.cycles_start;
    rsp.cycles_stop  = batch_prof.cycles_stop;

    જો (ctx->profiler == HTP_PROF_TRACE) {
        માટે (પૂર્ણાંક t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].ગણતરી;
        }
    }

    સંરચના dspqueue_buffer write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(queue, 0, 1, &write_dbuf, કદ(rsp), (અચલ uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    જો (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
    }
}

#define DSPQUEUE_READ_TIMEOUT_USEC 5000
#define DSPQUEUE_POLL_TIMEOUT_USEC 100
#define DSPQUEUE_POLL_ગણતરી        100

સ્થિર શૂન્ય process_ops(સંરચના htp_context * ctx) {
    dspqueue_t queue = ctx->dsp_queue;
    પૂર્ણાંક err;

    uint32_t poll_ગણતરી = DSPQUEUE_POLL_ગણતરી;

    vtcm_acquire(ctx);

    જ્યારે (!atomic_load(&ctx->vtcm_needs_release) && !atomic_load(&ctx->killed)) {
        સંરચના htp_opbatch_req req;
        uint32_t r_size = કદ(req);

        સંરચના dspqueue_buffer dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_size, &r_size, (uint8_t *) &req);
        જો (err == AEE_EWOULDBLOCK) {
            જો (--poll_ગણતરી) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                ચાલુ;
            }
            તોડો;
        }

        જો (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            તોડો;
        }

        જો (r_size < કદ(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_size, n_dbufs);
            ચાલુ;
        }

        // Reset poll count for valid requests
        poll_ગણતરી = DSPQUEUE_POLL_ગણતરી;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

સ્થિર શૂન્ય htp_packet_callback(dspqueue_t queue, પૂર્ણાંક error, શૂન્ય * context) {
    (શૂન્ય) queue;
    (શૂન્ય) error;
    સંરચના htp_handle * h = (સંરચના htp_handle *) context;
    જો (h && h->ctx) {
        process_ops(h->ctx);
    }
}

સ્થિર શૂન્ય htp_main_thread(શૂન્ય * context) {
    સંરચના htp_context * ctx = (સંરચના htp_context *) context;

    FARF(ઉપલું_2, "htp-main-thread: started");

    જ્યારે (!atomic_load(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t અંક_buffers = 0;
        uint32_t message_length = 0;

        પૂર્ણાંક err = dspqueue_peek(ctx->dsp_queue, &flags, &અંક_buffers, &message_length, 50000);
        જો (err == 0) {
            process_ops(ctx);
        } નહીંતર જો (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            ચાલુ;
        } નહીંતર {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (unsigned) err);
            તોડો;
        }
    }

    FARF(ઉપલું_2, "htp-main-thread: stopped");
}
