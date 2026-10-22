#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>
#अन्तर्भाव <AEEStdErr.h>
#अन्तर्भाव <dspqueue.h>
#अन्तर्भाव <HAP_compute_res.h>
#अन्तर्भाव <HAP_etm_config.h>
#अन्तर्भाव <HAP_mem.h>
#अन्तर्भाव <HAP_power.h>
#अन्तर्भाव <HAP_ps.h>
#अन्तर्भाव <HAP_dcvs.h>
#अन्तर्भाव <qurt.h>
#अन्तर्भाव <qurt_thread.h>
#अन्तर्भाव <qurt_memory.h>
#अन्तर्भाव <remote.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <stdatomic.h>

#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hmx-queue.h"

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "hex-bitmap.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-dcache.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp_iface.h"
#अन्तर्भाव "work-queue.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "allreduce-ops.h"
#अन्तर्भाव "htp-fence.h"

#define HMX_QUEUE_CAPACITY     128
#define HMX_QUEUE_चळत_SIZE   16384
#define WORK_QUEUE_CAPACITY    16
#define WORK_QUEUE_चळत_SIZE  16384
#define MAIN_THREAD_चळत_SIZE 32768

_Static_assert(WORK_QUEUE_MAX_N_THREADS >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

संरचना htp_handle {
    संरचना htp_context * ctx;
};

स्थिर ओळीत uint64_t htp_mmap(uint32_t fd, uint64_t size, uint32_t flags) {
#जर __HVX_ARCH__ > 79
    जर (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        स_रखो(&payload, 0, आकार(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = size;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        जर (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (अचिन्हित दीर्घ दीर्घ) size);
            परत 0;
        }

        परत payload.mmap.dsp_va;
    }
#अन्यथा
    जर (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        परत 0;
    }
#endif

    जर (size > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (अचिन्हित दीर्घ दीर्घ) size);
        परत 0;
    }

    शून्य * va = (शून्य *)-1;
    साठी (पूर्णांक retry = 0; retry < 2; retry++) {
#जर __HVX_ARCH__ > 73
        va = HAP_mmap2(रिक्त, (माप_प्रकार) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#अन्यथा
        जर (size > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (अचिन्हित दीर्घ दीर्घ) size);
            पात();
        }
        va = HAP_mmap(रिक्त, (पूर्णांक) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
        जर (va != (शून्य *)-1 && va != रिक्त) {
            परत (uint64_t) (uintptr_t) va;
        }
        जर (retry == 0) {
            FARF(वरचा_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (अचिन्हित दीर्घ दीर्घ) size);
        }
    }
    परत 0;
}

स्थिर ओळीत शून्य htp_munmap(uint64_t va, uint64_t size, uint32_t flags) {
#जर __HVX_ARCH__ > 79
    जर (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        स_रखो(&payload, 0, आकार(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = size;
        HAP_mem_request(&payload);
        परत;
    }
#endif

#जर __HVX_ARCH__ > 73
    HAP_munmap2((शून्य *) (uintptr_t) va, (माप_प्रकार) size);
#अन्यथा
    HAP_munmap((शून्य *) (uintptr_t) va, (पूर्णांक) size);
#endif
}

AEEResult htp_iface_open(स्थिरांक अक्षर * uri, remote_handle64 * handle) {
    (शून्य) uri;
    संरचना htp_handle * h = सुस्मृति(1, आकार(*h));
    जर (h == रिक्त) {
        परत AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    परत AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    संरचना htp_handle * h = (संरचना htp_handle *) handle;
    जर (!h) {
        परत AEE_EBADPARM;
    }

    पूर्णांक err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    जर (err) {
        जर (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } अन्यथा {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    परत err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, स्थिरांक htp_iface_pmu_conf* pmu_conf) {
    संरचना htp_handle * h = (संरचना htp_handle *) handle;
    जर (!h || !h->ctx) {
        परत AEE_EBADPARM;
    }
    संरचना htp_context * ctx = h->ctx;

    जर (mode == HTP_PROF_PMU) {
        स्थिरांक uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        साठी (; i < HEX_अंक_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        साठी (i = 0; i < HEX_अंक_PMU_COUNTERS; i++) {
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

    परत AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    संरचना htp_handle * h = (संरचना htp_handle *) handle;
    जर (!h) {
        परत AEE_EBADPARM;
    }

    संरचना htp_context * ctx = h->ctx;
    जर (ctx) {
        जर (ctx->dsp_queue) {
            FARF(ERROR, "Closing handle with queue still open");
            परत AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        साठी (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            जर (ctx->mmap[i].size) {
                htp_munmap(ctx->mmap[i].base, ctx->mmap[i].size, ctx->mmap[i].flags);
                ctx->mmap[i].size = 0;
                ctx->mmap[i].base = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        जर (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        जर (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        मुक्त(ctx);
        h->ctx = रिक्त;
    }

    मुक्त(h);
    परत AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t size) {
    संरचना htp_handle * h = (संरचना htp_handle *) handle;
    जर (!h || !h->ctx) {
        परत AEE_EBADPARM;
    }
    संरचना htp_context * ctx = h->ctx;

    // See if we already have this mapping
    साठी (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        संरचना htp_mmap *m = &ctx->mmap[i];
        जर (m->fd == fd) {
            परत AEE_SUCCESS;
        }
    }

    // Add new mapping
    साठी (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        संरचना htp_mmap *m = &ctx->mmap[i];
        जर (!m->size) {
            FARF(वरचा_2, "mmap : fd %u size %llu", fd, (अचिन्हित दीर्घ दीर्घ) size);
            uint64_t va = htp_mmap(fd, size, 0);
            जर (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (अचिन्हित दीर्घ दीर्घ) size);
                परत AEE_EFAILED;
            }

            m->base   = va;
            m->fd     = fd;
            m->size   = size;
            m->flags  = 0;

            परत AEE_SUCCESS;
        }
    }

    परत AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    संरचना htp_handle * h = (संरचना htp_handle *) handle;
    जर (!h || !h->ctx) {
        परत AEE_EBADPARM;
    }
    संरचना htp_context * ctx = h->ctx;

    साठी (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        संरचना htp_mmap *m = &ctx->mmap[i];
        जर (fd < 0 || m->fd == fd) {
            FARF(वरचा_2, "unmmap : base 0x%llx fd %u size %llu", (अचिन्हित दीर्घ दीर्घ) m->base, m->fd, (अचिन्हित दीर्घ दीर्घ) m->size);
            htp_munmap(m->base, m->size, m->flags);
            m->size   = 0;
            m->base   = रिक्त;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    परत AEE_SUCCESS;
}

स्थिर शून्य vtcm_acquire(संरचना htp_context * ctx) {
    जर (!atomic_load(&ctx->vtcm_valid)) {
        पूर्णांक err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        जर (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (अचिन्हित)err);
            पात();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

स्थिर शून्य vtcm_release(संरचना htp_context * ctx) {
    जर (atomic_load(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

स्थिर पूर्णांक vtcm_release_callback(अचिन्हित पूर्णांक rctx, शून्य * state) {
    संरचना htp_context * ctx = (संरचना htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    परत 0;
}

स्थिर अचिन्हित पूर्णांक vtcm_query_size(शून्य) {
    अचिन्हित पूर्णांक vtcm_size = 8 * 1024 * 1024;  // 8MB default
    अचिन्हित पूर्णांक vtcm_avail;
    compute_res_vtcm_page_t total_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_size, &total_layout, &vtcm_avail, &avail_layout);
    परत vtcm_size;
}

स्थिर पूर्णांक vtcm_alloc(संरचना htp_context * ctx) {
    अचिन्हित पूर्णांक vtcm_size = vtcm_query_size();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (शून्य *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    जर (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_size);
        परत AEE_ENOMEMORY;
    }

    शून्य * vtcm_ptr;
    जर (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_size);
        परत AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size          = vtcm_size;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    परत 0;
}

स्थिर शून्य vtcm_free(संरचना htp_context * ctx) {
    जर (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

स्थिर शून्य htp_main_thread(शून्य * context);
स्थिर शून्य htp_packet_callback(dspqueue_t queue, पूर्णांक error, शून्य * context);
स्थिर शून्य htp_error_callback(dspqueue_t queue, पूर्णांक error, शून्य * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
संरचना htp_power_profile {
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
        .hmx_perf_mode      = HAP_CLK_PERF_वरचा, \
    }

स्थिर स्थिरांक संरचना htp_power_profile htp_power_profiles[HTP_POWER_MODE_मोजणी] = {
    [HTP_POWER_MODE_खालचा_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_MODE_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_MODE_वरचा_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_MODE_खालचा_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_MODE_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_MODE_वरचा_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_SUSTAINED_वरचा_PERFORMANCE] =
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
        .hmx_perf_mode      = HAP_CLK_PERF_वरचा,
    },
};

#undef HTP_POWER_PROFILE_ENTRY

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_mode) {
    संरचना htp_handle * h = (संरचना htp_handle *) handle;
    जर (!h) {
        परत AEE_EBADPARM;
    }

    जर (power_mode >= HTP_POWER_MODE_मोजणी) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (अचिन्हित) power_mode);
        power_mode = HTP_POWER_MODE_BURST;
    }
    स्थिरांक संरचना htp_power_profile * pwr = &htp_power_profiles[power_mode];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (अचिन्हित) power_mode, (पूर्णांक) pwr->dcvs_enable,
         (अचिन्हित) pwr->bus_min_corner, (अचिन्हित) pwr->bus_max_corner, (अचिन्हित) pwr->bus_target_corner,
         (अचिन्हित) pwr->core_min_corner, (अचिन्हित) pwr->core_max_corner, (अचिन्हित) pwr->core_target_corner,
         (पूर्णांक) pwr->sleep_disable,
         (अचिन्हित) pwr->hmx_min_corner, (अचिन्हित) pwr->hmx_max_corner, (अचिन्हित) pwr->hmx_target_corner);

    जर (h->ctx) {
        FARF(ERROR, "Queue already open");
        परत AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    पूर्णांक fastrpc_tid  = qurt_thread_get_id();
    पूर्णांक fastrpc_prio = qurt_thread_get_priority(fastrpc_tid);
    पूर्णांक main_prio    = fastrpc_prio - 10;
    जर (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_queue = रिक्त;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    पूर्णांक err = dspqueue_import(dsp_queue_id, रिक्त, रिक्त, (शून्य *) h, &dsp_queue);
    जर (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(वरचा_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (शून्य *) h, &dsp_queue);
    }

    जर (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (अचिन्हित) err);
        परत err;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    जर (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    जर (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    जर (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    माप_प्रकार offset = आकार(संरचना htp_context);

    // 2. main_stack
    माप_प्रकार offset_main_चळत = 0;
    माप_प्रकार size_main_चळत   = 0;
    जर (!use_callbacks) {
        offset_main_चळत = hex_align_up(offset, 4096);
        size_main_चळत   = MAIN_THREAD_चळत_SIZE;
        offset = offset_main_चळत + size_main_चळत;
    }

    // 3. work_queue
    माप_प्रकार wq_align  = work_queue_alignof();
    माप_प्रकार offset_wq = hex_align_up(offset, wq_align);
    माप_प्रकार size_wq   = work_queue_sizeof(n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_चळत_SIZE);
    offset = offset_wq + size_wq;

    // 4. dma_queue
    माप_प्रकार dma_align = dma_queue_alignof();
    माप_प्रकार offset_dma = hex_align_up(offset, dma_align);
    माप_प्रकार size_dma = 0;
    साठी (uint32_t i = 0; i < n_hvx; i++) {
        size_dma  = hex_align_up(size_dma, dma_queue_alignof());
        size_dma += dma_queue_sizeof(256);
    }
    offset = offset_dma + size_dma;

    // 5. hmx_queue
    माप_प्रकार offset_hmx = 0;
    माप_प्रकार size_hmx = 0;
    जर (n_hmx) {
        माप_प्रकार hmx_align = hmx_queue_alignof();
        offset_hmx = hex_align_up(offset, hmx_align);
        size_hmx   = hmx_queue_sizeof(HMX_QUEUE_CAPACITY, HMX_QUEUE_चळत_SIZE);
        offset = offset_hmx + size_hmx;
    }

    माप_प्रकार footprint = hex_align_up(offset, 128);

    शून्य * block = memalign(4096, footprint);
    जर (!block) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_queue);
        परत AEE_ENOMEMORY;
    }
    स_रखो(block, 0, footprint);

    h->ctx = (संरचना htp_context *) block;
    संरचना htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_queue   = dsp_queue;

    err = vtcm_alloc(ctx);
    जर (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        परत AEE_ENOMEMORY;
    }

    HAP_setFARFRuntimeLoggingParams(0xffff, रिक्त, 0);

    // Set client class
    {
        HAP_power_request_t request;
        स_रखो(&request, 0, आकार(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        जर ((err = HAP_power_set((शून्य *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            परत err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        स_रखो(&request, 0, आकार(request));

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

#जर (__HEXAGON_ARCH__ >= 79)
        HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
#endif
        जर ((err = HAP_power_set((शून्य *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            परत err;
        }

        स_रखो(&request, 0, आकार(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        जर ((err = HAP_power_set((शून्य *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            परत err;
        }
    }

#जर __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        स_रखो(&request, 0, आकार(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_power     = TRUE;
        request.hmx_v2.power_up      = TRUE;
        request.hmx_v2.set_clock     = TRUE;
        request.hmx_v2.target_corner = pwr->hmx_target_corner;
        request.hmx_v2.min_corner    = pwr->hmx_min_corner;
        request.hmx_v2.max_corner    = pwr->hmx_max_corner;
        request.hmx_v2.perf_mode     = pwr->hmx_perf_mode;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((शून्य *) ctx, &request);
        जर (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            परत err;
        }
    }
#अन्यथा
    {
        // Power on HMX
        HAP_power_request_t request;
        स_रखो(&request, 0, आकार(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((शून्य *) ctx, &request);
        जर (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            परत err;
        }
    }
#endif

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_queue   = रिक्त;
    जर (n_hmx) {
        शून्य * hmx_ptr = (शून्य *) ((uintptr_t) block + offset_hmx);
        ctx->hmx_queue = hmx_queue_init(hmx_ptr, HMX_QUEUE_CAPACITY, HMX_QUEUE_चळत_SIZE, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(वरचा_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_threads = n_hvx;
    ctx->n_threads_div = init_fastdiv_मूल्ये(ctx->n_threads);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) block + offset_dma);
    माप_प्रकार size_dma_q = dma_queue_sizeof(256);

    साठी (पूर्णांक i = 0; i < ctx->n_threads; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_up((uintptr_t) dma_ptr_curr, dma_queue_alignof());
        ctx->dma[i] = dma_queue_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += size_dma_q;
    }

    ctx->ddr_spad_size = 512 * 1024; // 512 KB
    ctx->ddr_spad_base = memalign(128, ctx->ddr_spad_size);

    शून्य * wq_ptr = (शून्य *) ((uintptr_t) block + offset_wq);
    ctx->work_queue = work_queue_init(wq_ptr, n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_चळत_SIZE);

    ctx->main_चळत = रिक्त;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    जर (!use_callbacks) {
        // Start main compute thread
        ctx->main_चळत = (शून्य *) ((uintptr_t) block + offset_main_चळत);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_चळत_addr(&attr, ctx->main_चळत);
        qurt_thread_attr_set_चळत_size(&attr, size_main_चळत);
        qurt_thread_attr_set_priority(&attr, main_prio);
        qurt_thread_attr_set_name(&attr, "htp-main");

        पूर्णांक err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        जर (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            परत AEE_ENOMEMORY;
        }
    }

    FARF(वरचा_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_size, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    परत AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    संरचना htp_handle * h = (संरचना htp_handle *) handle;
    जर (!h || !h->ctx) {
        परत AEE_EBADPARM;
    }
    संरचना htp_context * ctx = h->ctx;

    जर (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        पूर्णांक status;
        (शून्य) qurt_thread_join(ctx->main_thread, &status);
        ctx->main_thread = 0;
    }

    पूर्णांक err = dspqueue_close(ctx->dsp_queue); ctx->dsp_queue = रिक्त;
    जर (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (अचिन्हित) err);
        परत err;
    }

    work_queue_free(ctx->work_queue);

    साठी (पूर्णांक i = 0; i < ctx->n_threads; i++) {
        dma_queue_free(ctx->dma[i]);
    }

    जर (ctx->hmx_queue) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = रिक्त;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    जर (ctx->ddr_spad_base) {
        मुक्त(ctx->ddr_spad_base);
        ctx->ddr_spad_base = रिक्त;
        ctx->ddr_spad_size = 0;
    }

    मुक्त(ctx);
    h->ctx = रिक्त;

    परत AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_threads, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_size) {
    (शून्य)handle;
    जर (!n_threads || !n_hvx || !n_hmx || !vtcm_size) {
        परत AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    जर (n_hvx_val > hw_threads.max_hthreads) {
        n_hvx_val = hw_threads.max_hthreads;
    }
    जर (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_threads = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_size = vtcm_query_size();

    परत AEE_SUCCESS;
}

स्थिर शून्य htp_error_callback(dspqueue_t queue, पूर्णांक error, शून्य * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (अचिन्हित) error);
}

संरचना profile_data {
    uint64_t usecs;
    uint64_t cycles_start;
    uint64_t cycles_stop;
    uint32_t pmu_counters[HEX_अंक_PMU_COUNTERS];
};

स्थिर ओळीत शून्य profile_start(uint32_t mode, संरचना profile_data * d) {
    निवड (mode) {
        स्थिती HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        स्थिती HTP_PROF_BASIC:
        स्थिती HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_मोजणी();
            d->cycles_start = hex_get_cycles();
            थांबा;
        पूर्वनिर्धारित:
            थांबा;
    }
}

स्थिर ओळीत शून्य profile_stop(uint32_t mode, संरचना profile_data * d) {
    uint32_t pmu_counters[HEX_अंक_PMU_COUNTERS];
    निवड (mode) {
        स्थिती HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            साठी (पूर्णांक i = 0; i < HEX_अंक_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        स्थिती HTP_PROF_BASIC:
        स्थिती HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_मोजणी_to_us(HAP_perf_get_qtimer_मोजणी() - d->usecs);
            d->cycles_stop = hex_get_cycles();
            थांबा;
        पूर्वनिर्धारित:
            थांबा;
    }
}

स्थिर पूर्णांक op_fence(संरचना htp_ops_context * octx) {
    संरचना htp_context *ctx = octx->ctx;
    संरचना htp_thread_trace * tr = &ctx->trace[0];
    स्थिरांक uint32_t seq  = (uint32_t) octx->op_params[0];
    स्थिरांक uint32_t mode = (uint32_t) octx->op_params[1];

    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    स्थिरांक संरचना htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;

    जर (mode == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_group_barrier(octx);

        जर (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->status);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(वरचा_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        परत octx->status;
    }

    पूर्णांक status = HTP_STATUS_OK;
    uint64_t spins = 0;
    पर्यंत (1) {
        uint32_t sync_seq;
        uint32_t sync_status;
        htp_fence_read(sync_fence, &sync_seq, &sync_status);
        जर ((int32_t)(sync_seq - seq) >= 0) {
            जर (sync_status > HTP_STATUS_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_status, sync_fence, seq);
                status = sync_status;
            }
            थांबा;
        }
        जर (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            status = HTP_STATUS_INTERNAL_ERR;
            थांबा;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(वरचा_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    परत status;
}

स्थिर पूर्णांक op_mdev_group(संरचना htp_ops_context * octx) {
    संरचना htp_context * ctx = octx->ctx;
    स्थिरांक संरचना htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.मोजणी = (uint16_t) sync->ne[1];
    जर (ctx->mdev.मोजणी > 1) {
        ctx->mdev.मोजणी_div = init_fastdiv_मूल्ये(ctx->mdev.मोजणी);
        ctx->mdev.fence_base = (uint8_t *) sync->data;
    }
    परत HTP_STATUS_OK;
}

स्थिर पूर्णांक execute_op(संरचना htp_ops_context * octx) {
    निवड (octx->op) {
        स्थिती HTP_OP_MDEV_GROUP:
            परत op_mdev_group(octx);

        स्थिती HTP_OP_FENCE:
            परत op_fence(octx);

        स्थिती HTP_OP_ALLREDUCE:
        स्थिती HTP_OP_ALLREDUCE_ADD:
            परत op_allreduce(octx);

        स्थिती HTP_OP_MUL_MAT:
        स्थिती HTP_OP_MUL_MAT_ADD:
            परत op_matmul(octx);

        स्थिती HTP_OP_MUL_MAT_ID:
            परत op_matmul_id(octx);

        स्थिती HTP_OP_MUL_MAT_ID_NX:
            परत op_matmul_id_nx(octx);

        स्थिती HTP_OP_MUL_MAT_NX:
            परत op_matmul_nx(octx);

        स्थिती HTP_OP_MUL:
        स्थिती HTP_OP_ADD:
        स्थिती HTP_OP_SUB:
        स्थिती HTP_OP_DIV:
        स्थिती HTP_OP_ADD_ID:
            परत op_द्विमान(octx);

        स्थिती HTP_OP_NORM:
        स्थिती HTP_OP_RMS_NORM:
        स्थिती HTP_OP_RMS_NORM_MUL:
        स्थिती HTP_OP_SCALE:
        स्थिती HTP_OP_CLAMP:
        स्थिती HTP_OP_LEAKY_RELU:
        स्थिती HTP_OP_SQR:
        स्थिती HTP_OP_SQRT:
        स्थिती HTP_OP_UNARY_SOFTPLUS:
        स्थिती HTP_OP_UNARY_SIGMOID:
        स्थिती HTP_OP_UNARY_SILU:
        स्थिती HTP_OP_UNARY_GELU:
        स्थिती HTP_OP_UNARY_GELU_QUICK:
        स्थिती HTP_OP_UNARY_GELU_ERF:
        स्थिती HTP_OP_UNARY_NEG:
        स्थिती HTP_OP_UNARY_EXP:
        स्थिती HTP_OP_UNARY_TANH:
        स्थिती HTP_OP_UNARY_ABS:
        स्थिती HTP_OP_UNARY_LOG:
        स्थिती HTP_OP_UNARY_RELU:
        स्थिती HTP_OP_UNARY_STEP:
        स्थिती HTP_OP_L2_NORM:
        स्थिती HTP_OP_HADAMARD:
            परत op_unary(octx);

        स्थिती HTP_OP_GLU_SWIGLU:
        स्थिती HTP_OP_GLU_SWIGLU_OAI:
        स्थिती HTP_OP_GLU_SWIGLU_CLAMP:
        स्थिती HTP_OP_GLU_GEGLU:
        स्थिती HTP_OP_GLU_GEGLU_QUICK:
        स्थिती HTP_OP_GLU_GEGLU_ERF:
            परत op_activations(octx);

        स्थिती HTP_OP_SOFTMAX:
            परत op_softmax(octx);

        स्थिती HTP_OP_ROPE:
            परत op_rope(octx);

        स्थिती HTP_OP_FLASH_ATTN_EXT:
            परत op_flash_attn_ext(octx);

        स्थिती HTP_OP_SET_ROWS:
            परत op_set_rows(octx);

        स्थिती HTP_OP_GET_ROWS:
            परत op_get_rows(octx);

        स्थिती HTP_OP_SUM:
            परत op_sum(octx);

        स्थिती HTP_OP_SUM_ROWS:
            परत op_sum_rows(octx);

        स्थिती HTP_OP_CPY:
        स्थिती HTP_OP_CPY_FENCE:
            परत op_cpy(octx);

        स्थिती HTP_OP_REPEAT:
            परत op_repeat(octx);

        स्थिती HTP_OP_ARGSORT:
            परत op_argsort(octx);

        स्थिती HTP_OP_शीर्ष_K:
            परत op_शीर्ष_k(octx);

        स्थिती HTP_OP_ARGMAX:
            परत op_argmax(octx);

        स्थिती HTP_OP_SSM_CONV:
            परत op_ssm_conv(octx);

        स्थिती HTP_OP_CUMSUM:
            परत op_cumsum(octx);

        स्थिती HTP_OP_FILL:
            परत op_fill(octx);

        स्थिती HTP_OP_DIAG:
            परत op_diag(octx);

        स्थिती HTP_OP_SOLVE_TRI:
            परत op_solve_tri(octx);

        स्थिती HTP_OP_PAD:
            परत op_pad(octx);

        स्थिती HTP_OP_IM2COL:
            परत op_im2col(octx);

        स्थिती HTP_OP_IM2COL_3D:
            परत op_im2col_3d(octx);

        स्थिती HTP_OP_ROLL:
            परत op_roll(octx);

        स्थिती HTP_OP_POOL_2D:
            परत op_pool_2d(octx);

        स्थिती HTP_OP_POOL_1D:
            परत op_pool_1d(octx);

        स्थिती HTP_OP_CONCAT:
            परत op_concat(octx);

        स्थिती HTP_OP_GATED_DELTA_NET:
            परत op_gated_delta_net(octx);

        स्थिती HTP_OP_TRI:
            परत op_unary(octx);

        स्थिती HTP_OP_INVALID:
            थांबा;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    परत HTP_STATUS_NO_SUPPORT;
}

स्थिर ओळीत bool reuse_buf(संरचना htp_context *ctx, uint32_t *m_reuse, संरचना htp_buf_desc *b) {
    b->base = रिक्त;

    साठी (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        संरचना htp_mmap *m = ctx->mmap + i;
        जर (m->size && m->fd == b->fd && m->flags == b->flags) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            परत true;
        }
    }

    परत false;
}

स्थिर ओळीत शून्य drop_mmap(संरचना htp_context *ctx, संरचना htp_mmap *m) {
    जर (m->size) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (अचिन्हित दीर्घ दीर्घ) m->base, (अचिन्हित दीर्घ दीर्घ) m->size);
        htp_munmap(m->base, m->size, m->flags);
        m->size = 0;
        m->base = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

स्थिर ओळीत bool mmap_buf(संरचना htp_context *ctx, संरचना htp_buf_desc *b) {
    जर (b->base) परत true; // already mapped

    // find unused mapping
    साठी (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        संरचना htp_mmap *m = &ctx->mmap[i];
        जर (!m->size) {
            uint64_t va = htp_mmap(b->fd, b->size, b->flags);
            जर (va == 0) {
                FARF(वरचा_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (अचिन्हित दीर्घ दीर्घ) b->size);
                परत false;
            }

            m->base   = b->base = va;
            m->fd     = b->fd;
            m->size   = b->size;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (अचिन्हित दीर्घ दीर्घ) m->base, (अचिन्हित दीर्घ दीर्घ) m->size, m->flags);
            परत true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    परत false;
}

स्थिर शून्य prep_op_bufs(संरचना htp_context *ctx, संरचना htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    साठी (uint32_t i=0; i < n_bufs; i++) {
        संरचना htp_buf_desc *b = bufs + i;
        जर (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } अन्यथा जर (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->size;
        }
        FARF(वरचा_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (अचिन्हित दीर्घ दीर्घ) b->base, (अचिन्हित दीर्घ दीर्घ) b->size, b->flags);
    }

    जर (b_reuse == n_bufs) परत; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    साठी (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        जर (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].size;
        }
    }

    FARF(वरचा_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (माप_प्रकार) m_vmem, (माप_प्रकार) e_vmem, (माप_प्रकार) ctx->max_vmem, n_bufs, b_reuse);

    जर ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        साठी (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            जर (!used && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    साठी (uint32_t i=0; i < n_bufs; i++) {
        संरचना htp_buf_desc *b = bufs + i;
        जर (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            थांबा;
        }
        FARF(वरचा_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (अचिन्हित दीर्घ दीर्घ) b->base, (अचिन्हित दीर्घ दीर्घ) b->size, b->flags);
    }

    जर (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(वरचा_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        साठी (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            जर (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        साठी (uint32_t i=0; i < n_bufs; i++) {
            संरचना htp_buf_desc *b = bufs + i;
            जर (!(b->flags & HTP_BUF_EXTENDED)) {
                b->base = 0;
            }
            जर (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (अचिन्हित दीर्घ दीर्घ) b->size);
                पात();
            }
            FARF(वरचा_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (अचिन्हित दीर्घ दीर्घ) b->base, (अचिन्हित दीर्घ दीर्घ) b->size, b->flags);
        }
    }
}

स्थिर शून्य prep_tensor(संरचना htp_context *ctx, संरचना htp_buf_desc *bufs, संरचना htp_tensor *tens, uint32_t idx, संरचना htp_tensor *t) {
    uint64_t offset = t->data;
    uint32_t bi     = t->bi;

    t->data  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(वरचा_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (अचिन्हित दीर्घ दीर्घ) offset, t->size, (अचिन्हित दीर्घ दीर्घ) t->data,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

स्थिर शून्य prep_tensors(संरचना htp_context *ctx, संरचना htp_buf_desc *bufs, संरचना htp_tensor *tens, uint32_t n_tens) {
    साठी (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

स्थिर शून्य mdev_group_init(संरचना htp_context * ctx, स्थिरांक संरचना htp_opbatch_req * req) {
    स_रखो(&ctx->mdev, 0, आकार(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

स्थिर पूर्णांक proc_op_req(संरचना htp_ops_context * octx, संरचना htp_buf_desc * bufs, uint32_t n_bufs,
                       संरचना htp_tensor * tens, uint32_t idx, संरचना htp_op_desc * op) {
    स_नकल(octx->op_params,     op->params, आकार(octx->op_params));
    स_नकल(octx->kernel_params, op->kernel_params, आकार(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_threads     = octx->ctx->n_threads;
    octx->n_threads_div = octx->ctx->n_threads_div;

    FARF(वरचा_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    साठी (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        जर (src_idx == 0xffff) {
            octx->src[i]     = रिक्त;
            सुरूठेवा;
        }

        संरचना htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(वरचा_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (अचिन्हित दीर्घ दीर्घ) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    साठी (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        जर (dst_idx == 0xffff) {
            octx->dsts[i]    = रिक्त;
            सुरूठेवा;
        }
        संरचना htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(वरचा_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (अचिन्हित दीर्घ दीर्घ) dst->data, dst->size,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_group_barrier(octx);

    पूर्णांक status = execute_op(octx);

    htp_ops_context_set_status(octx, status);

    octx->src0_spad.src = रिक्त;
    octx->src1_spad.src = रिक्त;
    octx->src2_spad.src = रिक्त;
    octx->src3_spad.src = रिक्त;
    octx->dst_spad.src  = रिक्त;

    परत octx->status;
}

स्थिर शून्य process_opbatch(संरचना htp_context * ctx, स्थिरांक संरचना htp_opbatch_req * req, स्थिरांक संरचना dspqueue_buffer * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    पूर्णांक err;

    स्थिरांक uint32_t n_bufs = req->n_bufs;
    स्थिरांक uint32_t n_tens = req->n_tensors;
    स्थिरांक uint32_t n_ops  = req->n_ops;

    स्थिरांक uint32_t b_size = आकार(संरचना htp_buf_desc)  * n_bufs;
    स्थिरांक uint32_t t_size = आकार(संरचना htp_tensor)    * n_tens;
    स्थिरांक uint32_t o_size = आकार(संरचना htp_op_desc)   * n_ops;
    स्थिरांक uint32_t p_size = आकार(संरचना htp_prof_desc) * n_ops;
    स्थिरांक uint32_t tr_size = (HTP_MAX_NTHREADS + 1) * req->n_traces * आकार(संरचना htp_trace_desc);

    जर (dbuf->size < b_size + t_size + o_size + p_size + tr_size) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->size, b_size + t_size + o_size + p_size + tr_size);
        परत;
    }

    FARF(वरचा_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (अचिन्हित दीर्घ दीर्घ) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->size, b_size, t_size, o_size);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    संरचना htp_buf_desc* bufs = (संरचना htp_buf_desc*)  m_ptr; m_ptr += b_size;
    संरचना htp_tensor*   tens = (संरचना htp_tensor*)    m_ptr; m_ptr += t_size;
    संरचना htp_op_desc*   ops = (संरचना htp_op_desc*)   m_ptr; m_ptr += o_size;
    संरचना htp_prof_desc* pds = (संरचना htp_prof_desc*) m_ptr;

    संरचना profile_data batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    स_रखो(ctx->trace, 0, आकार(ctx->trace));
    जर (ctx->profiler == HTP_PROF_TRACE) {
        संरचना htp_trace_desc * trace_events = (संरचना htp_trace_desc *) (m_ptr + p_size);
        साठी (पूर्णांक t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    स_रखो(ctx->dirty_ranges, 0, आकार(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    संरचना htp_ops_context *octx = &ctx->octx;
    स_रखो(octx, 0, आकार(*octx));
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
    octx->ctx           = ctx;

    mdev_group_init(ctx, req);

    work_queue_wakeup(ctx->work_queue);
    जर (ctx->hmx_queue) {
        hmx_queue_wakeup(ctx->hmx_queue);
    }

    पूर्णांक op_status = HTP_STATUS_OK;
    octx->status  = HTP_STATUS_OK;
    साठी (uint32_t i = 0; i < n_ops; i++) {
        संरचना profile_data prof;

        profile_start(ctx->profiler, &prof);

        op_status = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_set_status(octx, op_status);

        जर (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].cycles_start = prof.cycles_start;
            pds[i].cycles_stop = prof.cycles_stop;
            साठी (पूर्णांक j = 0; j < HEX_अंक_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    जर (ctx->hmx_queue) {
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

    संरचना htp_opbatch_rsp rsp;
    स_रखो(&rsp, 0, आकार(rsp));
    rsp.seq          = req->seq;
    rsp.status       = octx->status;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.cycles_start = batch_prof.cycles_start;
    rsp.cycles_stop  = batch_prof.cycles_stop;

    जर (ctx->profiler == HTP_PROF_TRACE) {
        साठी (पूर्णांक t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].मोजणी;
        }
    }

    संरचना dspqueue_buffer write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(queue, 0, 1, &write_dbuf, आकार(rsp), (स्थिरांक uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    जर (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (अचिन्हित) err);
    }
}

#define DSPQUEUE_READ_TIMEOUT_USEC 5000
#define DSPQUEUE_POLL_TIMEOUT_USEC 100
#define DSPQUEUE_POLL_मोजणी        100

स्थिर शून्य process_ops(संरचना htp_context * ctx) {
    dspqueue_t queue = ctx->dsp_queue;
    पूर्णांक err;

    uint32_t poll_मोजणी = DSPQUEUE_POLL_मोजणी;

    vtcm_acquire(ctx);

    पर्यंत (!atomic_load(&ctx->vtcm_needs_release) && !atomic_load(&ctx->killed)) {
        संरचना htp_opbatch_req req;
        uint32_t r_size = आकार(req);

        संरचना dspqueue_buffer dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_size, &r_size, (uint8_t *) &req);
        जर (err == AEE_EWOULDBLOCK) {
            जर (--poll_मोजणी) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                सुरूठेवा;
            }
            थांबा;
        }

        जर (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (अचिन्हित) err);
            थांबा;
        }

        जर (r_size < आकार(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_size, n_dbufs);
            सुरूठेवा;
        }

        // Reset poll count for valid requests
        poll_मोजणी = DSPQUEUE_POLL_मोजणी;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

स्थिर शून्य htp_packet_callback(dspqueue_t queue, पूर्णांक error, शून्य * context) {
    (शून्य) queue;
    (शून्य) error;
    संरचना htp_handle * h = (संरचना htp_handle *) context;
    जर (h && h->ctx) {
        process_ops(h->ctx);
    }
}

स्थिर शून्य htp_main_thread(शून्य * context) {
    संरचना htp_context * ctx = (संरचना htp_context *) context;

    FARF(वरचा_2, "htp-main-thread: started");

    पर्यंत (!atomic_load(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t अंक_buffers = 0;
        uint32_t message_length = 0;

        पूर्णांक err = dspqueue_peek(ctx->dsp_queue, &flags, &अंक_buffers, &message_length, 50000);
        जर (err == 0) {
            process_ops(ctx);
        } अन्यथा जर (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            सुरूठेवा;
        } अन्यथा {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (अचिन्हित) err);
            थांबा;
        }
    }

    FARF(वरचा_2, "htp-main-thread: stopped");
}
