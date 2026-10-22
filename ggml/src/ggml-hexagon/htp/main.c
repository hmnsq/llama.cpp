#आशय clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>
#समावेश <AEEStdErr.h>
#समावेश <dspqueue.h>
#समावेश <HAP_compute_res.h>
#समावेश <HAP_etm_config.h>
#समावेश <HAP_mem.h>
#समावेश <HAP_power.h>
#समावेश <HAP_ps.h>
#समावेश <HAP_dcvs.h>
#समावेश <qurt.h>
#समावेश <qurt_thread.h>
#समावेश <qurt_memory.h>
#समावेश <remote.h>
#समावेश <string.h>
#समावेश <stdatomic.h>

#समावेश "hex-utils.h"
#समावेश "dma-queue.h"
#समावेश "hmx-queue.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "hex-bitmap.h"
#समावेश "htp-ctx.h"
#समावेश "htp-dcache.h"
#समावेश "htp-ops.h"
#समावेश "htp_iface.h"
#समावेश "work-queue.h"
#समावेश "hex-profile.h"
#समावेश "allreduce-ops.h"
#समावेश "htp-fence.h"

#घोषणा HMX_QUEUE_CAPACITY     128
#घोषणा HMX_QUEUE_ढेर_आकार   16384
#घोषणा WORK_QUEUE_CAPACITY    16
#घोषणा WORK_QUEUE_ढेर_आकार  16384
#घोषणा MAIN_THREAD_ढेर_आकार 32768

_Static_assert(WORK_QUEUE_MAX_N_THREADS >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

काष्ठा htp_handle {
    काष्ठा htp_context * ctx;
};

अटल अंतरभूत uint64_t htp_mmap(uint32_t fd, uint64_t आकार, uint32_t flags) {
#अगर __HVX_ARCH__ > 79
    अगर (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        स_रखो(&payload, 0, माप(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = आकार;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        अगर (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (अचिन्हित दीर्घ दीर्घ) आकार);
            वापस 0;
        }

        वापस payload.mmap.dsp_va;
    }
#अन्यथा
    अगर (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        वापस 0;
    }
#पूर्ण_अगर

    अगर (आकार > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (अचिन्हित दीर्घ दीर्घ) आकार);
        वापस 0;
    }

    व्योम * va = (व्योम *)-1;
    क्रम (पूर्णांक retry = 0; retry < 2; retry++) {
#अगर __HVX_ARCH__ > 73
        va = HAP_mmap2(शून्य, (माप_प्रकार) आकार, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#अन्यथा
        अगर (आकार > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (अचिन्हित दीर्घ दीर्घ) आकार);
            पात();
        }
        va = HAP_mmap(शून्य, (पूर्णांक) आकार, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#पूर्ण_अगर
        अगर (va != (व्योम *)-1 && va != शून्य) {
            वापस (uint64_t) (uintptr_t) va;
        }
        अगर (retry == 0) {
            FARF(ऊपरी_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (अचिन्हित दीर्घ दीर्घ) आकार);
        }
    }
    वापस 0;
}

अटल अंतरभूत व्योम htp_munmap(uint64_t va, uint64_t आकार, uint32_t flags) {
#अगर __HVX_ARCH__ > 79
    अगर (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        स_रखो(&payload, 0, माप(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = आकार;
        HAP_mem_request(&payload);
        वापस;
    }
#पूर्ण_अगर

#अगर __HVX_ARCH__ > 73
    HAP_munmap2((व्योम *) (uintptr_t) va, (माप_प्रकार) आकार);
#अन्यथा
    HAP_munmap((व्योम *) (uintptr_t) va, (पूर्णांक) आकार);
#पूर्ण_अगर
}

AEEResult htp_iface_open(स्थिर अक्षर * uri, remote_handle64 * handle) {
    (व्योम) uri;
    काष्ठा htp_handle * h = सुस्मृति(1, माप(*h));
    अगर (h == शून्य) {
        वापस AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    वापस AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    काष्ठा htp_handle * h = (काष्ठा htp_handle *) handle;
    अगर (!h) {
        वापस AEE_EBADPARM;
    }

    पूर्णांक err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    अगर (err) {
        अगर (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } अन्यथा {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    वापस err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, स्थिर htp_iface_pmu_conf* pmu_conf) {
    काष्ठा htp_handle * h = (काष्ठा htp_handle *) handle;
    अगर (!h || !h->ctx) {
        वापस AEE_EBADPARM;
    }
    काष्ठा htp_context * ctx = h->ctx;

    अगर (mode == HTP_PROF_PMU) {
        स्थिर uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        क्रम (; i < HEX_अंक_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        क्रम (i = 0; i < HEX_अंक_PMU_COUNTERS; i++) {
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

    वापस AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    काष्ठा htp_handle * h = (काष्ठा htp_handle *) handle;
    अगर (!h) {
        वापस AEE_EBADPARM;
    }

    काष्ठा htp_context * ctx = h->ctx;
    अगर (ctx) {
        अगर (ctx->dsp_queue) {
            FARF(ERROR, "Closing handle with queue still open");
            वापस AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        क्रम (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            अगर (ctx->mmap[i].आकार) {
                htp_munmap(ctx->mmap[i].base, ctx->mmap[i].आकार, ctx->mmap[i].flags);
                ctx->mmap[i].आकार = 0;
                ctx->mmap[i].base = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        अगर (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        अगर (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        मुक्त(ctx);
        h->ctx = शून्य;
    }

    मुक्त(h);
    वापस AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t आकार) {
    काष्ठा htp_handle * h = (काष्ठा htp_handle *) handle;
    अगर (!h || !h->ctx) {
        वापस AEE_EBADPARM;
    }
    काष्ठा htp_context * ctx = h->ctx;

    // See if we already have this mapping
    क्रम (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        काष्ठा htp_mmap *m = &ctx->mmap[i];
        अगर (m->fd == fd) {
            वापस AEE_SUCCESS;
        }
    }

    // Add new mapping
    क्रम (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        काष्ठा htp_mmap *m = &ctx->mmap[i];
        अगर (!m->आकार) {
            FARF(ऊपरी_2, "mmap : fd %u size %llu", fd, (अचिन्हित दीर्घ दीर्घ) आकार);
            uint64_t va = htp_mmap(fd, आकार, 0);
            अगर (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (अचिन्हित दीर्घ दीर्घ) आकार);
                वापस AEE_EFAILED;
            }

            m->base   = va;
            m->fd     = fd;
            m->आकार   = आकार;
            m->flags  = 0;

            वापस AEE_SUCCESS;
        }
    }

    वापस AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    काष्ठा htp_handle * h = (काष्ठा htp_handle *) handle;
    अगर (!h || !h->ctx) {
        वापस AEE_EBADPARM;
    }
    काष्ठा htp_context * ctx = h->ctx;

    क्रम (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        काष्ठा htp_mmap *m = &ctx->mmap[i];
        अगर (fd < 0 || m->fd == fd) {
            FARF(ऊपरी_2, "unmmap : base 0x%llx fd %u size %llu", (अचिन्हित दीर्घ दीर्घ) m->base, m->fd, (अचिन्हित दीर्घ दीर्घ) m->आकार);
            htp_munmap(m->base, m->आकार, m->flags);
            m->आकार   = 0;
            m->base   = शून्य;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    वापस AEE_SUCCESS;
}

अटल व्योम vtcm_acquire(काष्ठा htp_context * ctx) {
    अगर (!atomic_load(&ctx->vtcm_valid)) {
        पूर्णांक err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        अगर (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (अचिन्हित)err);
            पात();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

अटल व्योम vtcm_release(काष्ठा htp_context * ctx) {
    अगर (atomic_load(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

अटल पूर्णांक vtcm_release_callback(अचिन्हित पूर्णांक rctx, व्योम * state) {
    काष्ठा htp_context * ctx = (काष्ठा htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    वापस 0;
}

अटल अचिन्हित पूर्णांक vtcm_query_आकार(व्योम) {
    अचिन्हित पूर्णांक vtcm_आकार = 8 * 1024 * 1024;  // 8MB default
    अचिन्हित पूर्णांक vtcm_avail;
    compute_res_vtcm_page_t योग_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_आकार, &योग_layout, &vtcm_avail, &avail_layout);
    वापस vtcm_आकार;
}

अटल पूर्णांक vtcm_alloc(काष्ठा htp_context * ctx) {
    अचिन्हित पूर्णांक vtcm_आकार = vtcm_query_आकार();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_आकार, vtcm_आकार, vtcm_आकार); // single page
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (व्योम *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    अगर (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_आकार);
        वापस AEE_ENOMEMORY;
    }

    व्योम * vtcm_ptr;
    अगर (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_आकार) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_आकार);
        वापस AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_आकार          = vtcm_आकार;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    वापस 0;
}

अटल व्योम vtcm_free(काष्ठा htp_context * ctx) {
    अगर (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

अटल व्योम htp_main_thread(व्योम * context);
अटल व्योम htp_packet_callback(dspqueue_t queue, पूर्णांक error, व्योम * context);
अटल व्योम htp_error_callback(dspqueue_t queue, पूर्णांक error, व्योम * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
काष्ठा htp_power_profile {
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

#घोषणा HTP_POWER_PROFILE_ENTRY(min_c, max_c, hmx_c) \
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
        .hmx_perf_mode      = HAP_CLK_PERF_ऊपरी, \
    }

अटल स्थिर काष्ठा htp_power_profile htp_power_profiles[HTP_POWER_MODE_गिनती] = {
    [HTP_POWER_MODE_निचला_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_MODE_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_MODE_ऊपरी_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_MODE_निचला_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_MODE_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_MODE_ऊपरी_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_SUSTAINED_ऊपरी_PERFORMANCE] =
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
        .hmx_perf_mode      = HAP_CLK_PERF_ऊपरी,
    },
};

#अघोषित HTP_POWER_PROFILE_ENTRY

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_mode) {
    काष्ठा htp_handle * h = (काष्ठा htp_handle *) handle;
    अगर (!h) {
        वापस AEE_EBADPARM;
    }

    अगर (power_mode >= HTP_POWER_MODE_गिनती) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (अचिन्हित) power_mode);
        power_mode = HTP_POWER_MODE_BURST;
    }
    स्थिर काष्ठा htp_power_profile * pwr = &htp_power_profiles[power_mode];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (अचिन्हित) power_mode, (पूर्णांक) pwr->dcvs_enable,
         (अचिन्हित) pwr->bus_min_corner, (अचिन्हित) pwr->bus_max_corner, (अचिन्हित) pwr->bus_target_corner,
         (अचिन्हित) pwr->core_min_corner, (अचिन्हित) pwr->core_max_corner, (अचिन्हित) pwr->core_target_corner,
         (पूर्णांक) pwr->sleep_disable,
         (अचिन्हित) pwr->hmx_min_corner, (अचिन्हित) pwr->hmx_max_corner, (अचिन्हित) pwr->hmx_target_corner);

    अगर (h->ctx) {
        FARF(ERROR, "Queue already open");
        वापस AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    पूर्णांक fastrpc_tid  = qurt_thread_get_id();
    पूर्णांक fastrpc_prio = qurt_thread_get_priority(fastrpc_tid);
    पूर्णांक main_prio    = fastrpc_prio - 10;
    अगर (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_queue = शून्य;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    पूर्णांक err = dspqueue_import(dsp_queue_id, शून्य, शून्य, (व्योम *) h, &dsp_queue);
    अगर (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(ऊपरी_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (व्योम *) h, &dsp_queue);
    }

    अगर (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (अचिन्हित) err);
        वापस err;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    अगर (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    अगर (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    अगर (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    माप_प्रकार offset = माप(काष्ठा htp_context);

    // 2. main_stack
    माप_प्रकार offset_main_ढेर = 0;
    माप_प्रकार आकार_main_ढेर   = 0;
    अगर (!use_callbacks) {
        offset_main_ढेर = hex_align_up(offset, 4096);
        आकार_main_ढेर   = MAIN_THREAD_ढेर_आकार;
        offset = offset_main_ढेर + आकार_main_ढेर;
    }

    // 3. work_queue
    माप_प्रकार wq_align  = work_queue_alignof();
    माप_प्रकार offset_wq = hex_align_up(offset, wq_align);
    माप_प्रकार आकार_wq   = work_queue_sizeof(n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_ढेर_आकार);
    offset = offset_wq + आकार_wq;

    // 4. dma_queue
    माप_प्रकार dma_align = dma_queue_alignof();
    माप_प्रकार offset_dma = hex_align_up(offset, dma_align);
    माप_प्रकार आकार_dma = 0;
    क्रम (uint32_t i = 0; i < n_hvx; i++) {
        आकार_dma  = hex_align_up(आकार_dma, dma_queue_alignof());
        आकार_dma += dma_queue_sizeof(256);
    }
    offset = offset_dma + आकार_dma;

    // 5. hmx_queue
    माप_प्रकार offset_hmx = 0;
    माप_प्रकार आकार_hmx = 0;
    अगर (n_hmx) {
        माप_प्रकार hmx_align = hmx_queue_alignof();
        offset_hmx = hex_align_up(offset, hmx_align);
        आकार_hmx   = hmx_queue_sizeof(HMX_QUEUE_CAPACITY, HMX_QUEUE_ढेर_आकार);
        offset = offset_hmx + आकार_hmx;
    }

    माप_प्रकार footprint = hex_align_up(offset, 128);

    व्योम * block = memalign(4096, footprint);
    अगर (!block) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_queue);
        वापस AEE_ENOMEMORY;
    }
    स_रखो(block, 0, footprint);

    h->ctx = (काष्ठा htp_context *) block;
    काष्ठा htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_queue   = dsp_queue;

    err = vtcm_alloc(ctx);
    अगर (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        वापस AEE_ENOMEMORY;
    }

    HAP_setFARFRuntimeLoggingParams(0xffff, शून्य, 0);

    // Set client class
    {
        HAP_power_request_t request;
        स_रखो(&request, 0, माप(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        अगर ((err = HAP_power_set((व्योम *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            वापस err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        स_रखो(&request, 0, माप(request));

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

#अगर (__HEXAGON_ARCH__ >= 79)
        HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
#पूर्ण_अगर
        अगर ((err = HAP_power_set((व्योम *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            वापस err;
        }

        स_रखो(&request, 0, माप(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        अगर ((err = HAP_power_set((व्योम *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            वापस err;
        }
    }

#अगर __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        स_रखो(&request, 0, माप(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_power     = TRUE;
        request.hmx_v2.power_up      = TRUE;
        request.hmx_v2.set_clock     = TRUE;
        request.hmx_v2.target_corner = pwr->hmx_target_corner;
        request.hmx_v2.min_corner    = pwr->hmx_min_corner;
        request.hmx_v2.max_corner    = pwr->hmx_max_corner;
        request.hmx_v2.perf_mode     = pwr->hmx_perf_mode;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((व्योम *) ctx, &request);
        अगर (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            वापस err;
        }
    }
#अन्यथा
    {
        // Power on HMX
        HAP_power_request_t request;
        स_रखो(&request, 0, माप(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((व्योम *) ctx, &request);
        अगर (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            वापस err;
        }
    }
#पूर्ण_अगर

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_queue   = शून्य;
    अगर (n_hmx) {
        व्योम * hmx_ptr = (व्योम *) ((uintptr_t) block + offset_hmx);
        ctx->hmx_queue = hmx_queue_init(hmx_ptr, HMX_QUEUE_CAPACITY, HMX_QUEUE_ढेर_आकार, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(ऊपरी_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_threads = n_hvx;
    ctx->n_threads_div = init_fastdiv_मूल्यावली(ctx->n_threads);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) block + offset_dma);
    माप_प्रकार आकार_dma_q = dma_queue_sizeof(256);

    क्रम (पूर्णांक i = 0; i < ctx->n_threads; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_up((uintptr_t) dma_ptr_curr, dma_queue_alignof());
        ctx->dma[i] = dma_queue_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += आकार_dma_q;
    }

    ctx->ddr_spad_आकार = 512 * 1024; // 512 KB
    ctx->ddr_spad_base = memalign(128, ctx->ddr_spad_आकार);

    व्योम * wq_ptr = (व्योम *) ((uintptr_t) block + offset_wq);
    ctx->work_queue = work_queue_init(wq_ptr, n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_ढेर_आकार);

    ctx->main_ढेर = शून्य;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    अगर (!use_callbacks) {
        // Start main compute thread
        ctx->main_ढेर = (व्योम *) ((uintptr_t) block + offset_main_ढेर);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_ढेर_addr(&attr, ctx->main_ढेर);
        qurt_thread_attr_set_ढेर_आकार(&attr, आकार_main_ढेर);
        qurt_thread_attr_set_priority(&attr, main_prio);
        qurt_thread_attr_set_name(&attr, "htp-main");

        पूर्णांक err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        अगर (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            वापस AEE_ENOMEMORY;
        }
    }

    FARF(ऊपरी_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_आकार, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    वापस AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    काष्ठा htp_handle * h = (काष्ठा htp_handle *) handle;
    अगर (!h || !h->ctx) {
        वापस AEE_EBADPARM;
    }
    काष्ठा htp_context * ctx = h->ctx;

    अगर (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        पूर्णांक status;
        (व्योम) qurt_thread_join(ctx->main_thread, &status);
        ctx->main_thread = 0;
    }

    पूर्णांक err = dspqueue_close(ctx->dsp_queue); ctx->dsp_queue = शून्य;
    अगर (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (अचिन्हित) err);
        वापस err;
    }

    work_queue_free(ctx->work_queue);

    क्रम (पूर्णांक i = 0; i < ctx->n_threads; i++) {
        dma_queue_free(ctx->dma[i]);
    }

    अगर (ctx->hmx_queue) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = शून्य;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    अगर (ctx->ddr_spad_base) {
        मुक्त(ctx->ddr_spad_base);
        ctx->ddr_spad_base = शून्य;
        ctx->ddr_spad_आकार = 0;
    }

    मुक्त(ctx);
    h->ctx = शून्य;

    वापस AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_threads, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_आकार) {
    (व्योम)handle;
    अगर (!n_threads || !n_hvx || !n_hmx || !vtcm_आकार) {
        वापस AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    अगर (n_hvx_val > hw_threads.max_hthreads) {
        n_hvx_val = hw_threads.max_hthreads;
    }
    अगर (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_threads = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_आकार = vtcm_query_आकार();

    वापस AEE_SUCCESS;
}

अटल व्योम htp_error_callback(dspqueue_t queue, पूर्णांक error, व्योम * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (अचिन्हित) error);
}

काष्ठा profile_तथ्य {
    uint64_t usecs;
    uint64_t cycles_start;
    uint64_t cycles_stop;
    uint32_t pmu_counters[HEX_अंक_PMU_COUNTERS];
};

अटल अंतरभूत व्योम profile_start(uint32_t mode, काष्ठा profile_तथ्य * d) {
    चयन (mode) {
        हाल HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        हाल HTP_PROF_BASIC:
        हाल HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_गिनती();
            d->cycles_start = hex_get_cycles();
            अवरोध;
        शेष:
            अवरोध;
    }
}

अटल अंतरभूत व्योम profile_stop(uint32_t mode, काष्ठा profile_तथ्य * d) {
    uint32_t pmu_counters[HEX_अंक_PMU_COUNTERS];
    चयन (mode) {
        हाल HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            क्रम (पूर्णांक i = 0; i < HEX_अंक_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        हाल HTP_PROF_BASIC:
        हाल HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_गिनती_to_us(HAP_perf_get_qtimer_गिनती() - d->usecs);
            d->cycles_stop = hex_get_cycles();
            अवरोध;
        शेष:
            अवरोध;
    }
}

अटल पूर्णांक op_fence(काष्ठा htp_ops_context * octx) {
    काष्ठा htp_context *ctx = octx->ctx;
    काष्ठा htp_thread_trace * tr = &ctx->trace[0];
    स्थिर uint32_t seq  = (uint32_t) octx->op_params[0];
    स्थिर uint32_t mode = (uint32_t) octx->op_params[1];

    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    स्थिर काष्ठा htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->तथ्य;

    अगर (mode == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_group_barrier(octx);

        अगर (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->status);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(ऊपरी_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        वापस octx->status;
    }

    पूर्णांक status = HTP_STATUS_OK;
    uint64_t spins = 0;
    जबतक (1) {
        uint32_t sync_seq;
        uint32_t sync_status;
        htp_fence_read(sync_fence, &sync_seq, &sync_status);
        अगर ((int32_t)(sync_seq - seq) >= 0) {
            अगर (sync_status > HTP_STATUS_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_status, sync_fence, seq);
                status = sync_status;
            }
            अवरोध;
        }
        अगर (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            status = HTP_STATUS_INTERNAL_ERR;
            अवरोध;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(ऊपरी_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    वापस status;
}

अटल पूर्णांक op_mdev_group(काष्ठा htp_ops_context * octx) {
    काष्ठा htp_context * ctx = octx->ctx;
    स्थिर काष्ठा htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.गिनती = (uint16_t) sync->ne[1];
    अगर (ctx->mdev.गिनती > 1) {
        ctx->mdev.गिनती_div = init_fastdiv_मूल्यावली(ctx->mdev.गिनती);
        ctx->mdev.fence_base = (uint8_t *) sync->तथ्य;
    }
    वापस HTP_STATUS_OK;
}

अटल पूर्णांक execute_op(काष्ठा htp_ops_context * octx) {
    चयन (octx->op) {
        हाल HTP_OP_MDEV_GROUP:
            वापस op_mdev_group(octx);

        हाल HTP_OP_FENCE:
            वापस op_fence(octx);

        हाल HTP_OP_ALLREDUCE:
        हाल HTP_OP_ALLREDUCE_ADD:
            वापस op_allreduce(octx);

        हाल HTP_OP_MUL_MAT:
        हाल HTP_OP_MUL_MAT_ADD:
            वापस op_matmul(octx);

        हाल HTP_OP_MUL_MAT_ID:
            वापस op_matmul_id(octx);

        हाल HTP_OP_MUL_MAT_ID_NX:
            वापस op_matmul_id_nx(octx);

        हाल HTP_OP_MUL_MAT_NX:
            वापस op_matmul_nx(octx);

        हाल HTP_OP_MUL:
        हाल HTP_OP_ADD:
        हाल HTP_OP_SUB:
        हाल HTP_OP_DIV:
        हाल HTP_OP_ADD_ID:
            वापस op_द्विआधारी(octx);

        हाल HTP_OP_NORM:
        हाल HTP_OP_RMS_NORM:
        हाल HTP_OP_RMS_NORM_MUL:
        हाल HTP_OP_SCALE:
        हाल HTP_OP_CLAMP:
        हाल HTP_OP_LEAKY_RELU:
        हाल HTP_OP_SQR:
        हाल HTP_OP_SQRT:
        हाल HTP_OP_UNARY_SOFTPLUS:
        हाल HTP_OP_UNARY_SIGMOID:
        हाल HTP_OP_UNARY_SILU:
        हाल HTP_OP_UNARY_GELU:
        हाल HTP_OP_UNARY_GELU_QUICK:
        हाल HTP_OP_UNARY_GELU_ERF:
        हाल HTP_OP_UNARY_NEG:
        हाल HTP_OP_UNARY_EXP:
        हाल HTP_OP_UNARY_TANH:
        हाल HTP_OP_UNARY_ABS:
        हाल HTP_OP_UNARY_LOG:
        हाल HTP_OP_UNARY_RELU:
        हाल HTP_OP_UNARY_STEP:
        हाल HTP_OP_L2_NORM:
        हाल HTP_OP_HADAMARD:
            वापस op_unary(octx);

        हाल HTP_OP_GLU_SWIGLU:
        हाल HTP_OP_GLU_SWIGLU_OAI:
        हाल HTP_OP_GLU_SWIGLU_CLAMP:
        हाल HTP_OP_GLU_GEGLU:
        हाल HTP_OP_GLU_GEGLU_QUICK:
        हाल HTP_OP_GLU_GEGLU_ERF:
            वापस op_activations(octx);

        हाल HTP_OP_SOFTMAX:
            वापस op_softmax(octx);

        हाल HTP_OP_ROPE:
            वापस op_rope(octx);

        हाल HTP_OP_FLASH_ATTN_EXT:
            वापस op_flash_attn_ext(octx);

        हाल HTP_OP_SET_ROWS:
            वापस op_set_rows(octx);

        हाल HTP_OP_GET_ROWS:
            वापस op_get_rows(octx);

        हाल HTP_OP_योग:
            वापस op_योग(octx);

        हाल HTP_OP_योग_ROWS:
            वापस op_योग_rows(octx);

        हाल HTP_OP_CPY:
        हाल HTP_OP_CPY_FENCE:
            वापस op_cpy(octx);

        हाल HTP_OP_REPEAT:
            वापस op_repeat(octx);

        हाल HTP_OP_ARGSORT:
            वापस op_argsort(octx);

        हाल HTP_OP_शीर्ष_K:
            वापस op_शीर्ष_k(octx);

        हाल HTP_OP_ARGMAX:
            वापस op_argmax(octx);

        हाल HTP_OP_SSM_CONV:
            वापस op_ssm_conv(octx);

        हाल HTP_OP_CUMSUM:
            वापस op_cumsum(octx);

        हाल HTP_OP_FILL:
            वापस op_fill(octx);

        हाल HTP_OP_DIAG:
            वापस op_diag(octx);

        हाल HTP_OP_SOLVE_TRI:
            वापस op_solve_tri(octx);

        हाल HTP_OP_PAD:
            वापस op_pad(octx);

        हाल HTP_OP_IM2COL:
            वापस op_im2col(octx);

        हाल HTP_OP_IM2COL_3D:
            वापस op_im2col_3d(octx);

        हाल HTP_OP_ROLL:
            वापस op_roll(octx);

        हाल HTP_OP_POOL_2D:
            वापस op_pool_2d(octx);

        हाल HTP_OP_POOL_1D:
            वापस op_pool_1d(octx);

        हाल HTP_OP_CONCAT:
            वापस op_concat(octx);

        हाल HTP_OP_GATED_DELTA_NET:
            वापस op_gated_delta_net(octx);

        हाल HTP_OP_TRI:
            वापस op_unary(octx);

        हाल HTP_OP_INVALID:
            अवरोध;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    वापस HTP_STATUS_NO_SUPPORT;
}

अटल अंतरभूत bool reuse_buf(काष्ठा htp_context *ctx, uint32_t *m_reuse, काष्ठा htp_buf_desc *b) {
    b->base = शून्य;

    क्रम (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        काष्ठा htp_mmap *m = ctx->mmap + i;
        अगर (m->आकार && m->fd == b->fd && m->flags == b->flags) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            वापस true;
        }
    }

    वापस false;
}

अटल अंतरभूत व्योम drop_mmap(काष्ठा htp_context *ctx, काष्ठा htp_mmap *m) {
    अगर (m->आकार) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (अचिन्हित दीर्घ दीर्घ) m->base, (अचिन्हित दीर्घ दीर्घ) m->आकार);
        htp_munmap(m->base, m->आकार, m->flags);
        m->आकार = 0;
        m->base = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

अटल अंतरभूत bool mmap_buf(काष्ठा htp_context *ctx, काष्ठा htp_buf_desc *b) {
    अगर (b->base) वापस true; // already mapped

    // find unused mapping
    क्रम (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        काष्ठा htp_mmap *m = &ctx->mmap[i];
        अगर (!m->आकार) {
            uint64_t va = htp_mmap(b->fd, b->आकार, b->flags);
            अगर (va == 0) {
                FARF(ऊपरी_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (अचिन्हित दीर्घ दीर्घ) b->आकार);
                वापस false;
            }

            m->base   = b->base = va;
            m->fd     = b->fd;
            m->आकार   = b->आकार;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (अचिन्हित दीर्घ दीर्घ) m->base, (अचिन्हित दीर्घ दीर्घ) m->आकार, m->flags);
            वापस true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    वापस false;
}

अटल व्योम prep_op_bufs(काष्ठा htp_context *ctx, काष्ठा htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    क्रम (uint32_t i=0; i < n_bufs; i++) {
        काष्ठा htp_buf_desc *b = bufs + i;
        अगर (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } अन्यथा अगर (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->आकार;
        }
        FARF(ऊपरी_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (अचिन्हित दीर्घ दीर्घ) b->base, (अचिन्हित दीर्घ दीर्घ) b->आकार, b->flags);
    }

    अगर (b_reuse == n_bufs) वापस; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    क्रम (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        अगर (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].आकार;
        }
    }

    FARF(ऊपरी_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (माप_प्रकार) m_vmem, (माप_प्रकार) e_vmem, (माप_प्रकार) ctx->max_vmem, n_bufs, b_reuse);

    अगर ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        क्रम (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            अगर (!used && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    क्रम (uint32_t i=0; i < n_bufs; i++) {
        काष्ठा htp_buf_desc *b = bufs + i;
        अगर (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            अवरोध;
        }
        FARF(ऊपरी_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (अचिन्हित दीर्घ दीर्घ) b->base, (अचिन्हित दीर्घ दीर्घ) b->आकार, b->flags);
    }

    अगर (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(ऊपरी_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        क्रम (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            अगर (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        क्रम (uint32_t i=0; i < n_bufs; i++) {
            काष्ठा htp_buf_desc *b = bufs + i;
            अगर (!(b->flags & HTP_BUF_EXTENDED)) {
                b->base = 0;
            }
            अगर (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (अचिन्हित दीर्घ दीर्घ) b->आकार);
                पात();
            }
            FARF(ऊपरी_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (अचिन्हित दीर्घ दीर्घ) b->base, (अचिन्हित दीर्घ दीर्घ) b->आकार, b->flags);
        }
    }
}

अटल व्योम prep_tensor(काष्ठा htp_context *ctx, काष्ठा htp_buf_desc *bufs, काष्ठा htp_tensor *tens, uint32_t idx, काष्ठा htp_tensor *t) {
    uint64_t offset = t->तथ्य;
    uint32_t bi     = t->bi;

    t->तथ्य  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(ऊपरी_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (अचिन्हित दीर्घ दीर्घ) offset, t->आकार, (अचिन्हित दीर्घ दीर्घ) t->तथ्य,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

अटल व्योम prep_tensors(काष्ठा htp_context *ctx, काष्ठा htp_buf_desc *bufs, काष्ठा htp_tensor *tens, uint32_t n_tens) {
    क्रम (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

अटल व्योम mdev_group_init(काष्ठा htp_context * ctx, स्थिर काष्ठा htp_opbatch_req * req) {
    स_रखो(&ctx->mdev, 0, माप(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

अटल पूर्णांक proc_op_req(काष्ठा htp_ops_context * octx, काष्ठा htp_buf_desc * bufs, uint32_t n_bufs,
                       काष्ठा htp_tensor * tens, uint32_t idx, काष्ठा htp_op_desc * op) {
    स_नकल(octx->op_params,     op->params, माप(octx->op_params));
    स_नकल(octx->kernel_params, op->kernel_params, माप(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_threads     = octx->ctx->n_threads;
    octx->n_threads_div = octx->ctx->n_threads_div;

    FARF(ऊपरी_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    क्रम (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        अगर (src_idx == 0xffff) {
            octx->src[i]     = शून्य;
            जारी;
        }

        काष्ठा htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(ऊपरी_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (अचिन्हित दीर्घ दीर्घ) src->तथ्य, src->आकार,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    क्रम (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        अगर (dst_idx == 0xffff) {
            octx->dsts[i]    = शून्य;
            जारी;
        }
        काष्ठा htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(ऊपरी_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (अचिन्हित दीर्घ दीर्घ) dst->तथ्य, dst->आकार,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_group_barrier(octx);

    पूर्णांक status = execute_op(octx);

    htp_ops_context_set_status(octx, status);

    octx->src0_spad.src = शून्य;
    octx->src1_spad.src = शून्य;
    octx->src2_spad.src = शून्य;
    octx->src3_spad.src = शून्य;
    octx->dst_spad.src  = शून्य;

    वापस octx->status;
}

अटल व्योम process_opbatch(काष्ठा htp_context * ctx, स्थिर काष्ठा htp_opbatch_req * req, स्थिर काष्ठा dspqueue_बफर * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    पूर्णांक err;

    स्थिर uint32_t n_bufs = req->n_bufs;
    स्थिर uint32_t n_tens = req->n_tensors;
    स्थिर uint32_t n_ops  = req->n_ops;

    स्थिर uint32_t b_आकार = माप(काष्ठा htp_buf_desc)  * n_bufs;
    स्थिर uint32_t t_आकार = माप(काष्ठा htp_tensor)    * n_tens;
    स्थिर uint32_t o_आकार = माप(काष्ठा htp_op_desc)   * n_ops;
    स्थिर uint32_t p_आकार = माप(काष्ठा htp_prof_desc) * n_ops;
    स्थिर uint32_t tr_आकार = (HTP_MAX_NTHREADS + 1) * req->n_traces * माप(काष्ठा htp_trace_desc);

    अगर (dbuf->आकार < b_आकार + t_आकार + o_आकार + p_आकार + tr_आकार) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->आकार, b_आकार + t_आकार + o_आकार + p_आकार + tr_आकार);
        वापस;
    }

    FARF(ऊपरी_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (अचिन्हित दीर्घ दीर्घ) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->आकार, b_आकार, t_आकार, o_आकार);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    काष्ठा htp_buf_desc* bufs = (काष्ठा htp_buf_desc*)  m_ptr; m_ptr += b_आकार;
    काष्ठा htp_tensor*   tens = (काष्ठा htp_tensor*)    m_ptr; m_ptr += t_आकार;
    काष्ठा htp_op_desc*   ops = (काष्ठा htp_op_desc*)   m_ptr; m_ptr += o_आकार;
    काष्ठा htp_prof_desc* pds = (काष्ठा htp_prof_desc*) m_ptr;

    काष्ठा profile_तथ्य batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    स_रखो(ctx->trace, 0, माप(ctx->trace));
    अगर (ctx->profiler == HTP_PROF_TRACE) {
        काष्ठा htp_trace_desc * trace_events = (काष्ठा htp_trace_desc *) (m_ptr + p_आकार);
        क्रम (पूर्णांक t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    स_रखो(ctx->dirty_ranges, 0, माप(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    काष्ठा htp_ops_context *octx = &ctx->octx;
    स_रखो(octx, 0, माप(*octx));
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
    octx->ctx           = ctx;

    mdev_group_init(ctx, req);

    work_queue_wakeup(ctx->work_queue);
    अगर (ctx->hmx_queue) {
        hmx_queue_wakeup(ctx->hmx_queue);
    }

    पूर्णांक op_status = HTP_STATUS_OK;
    octx->status  = HTP_STATUS_OK;
    क्रम (uint32_t i = 0; i < n_ops; i++) {
        काष्ठा profile_तथ्य prof;

        profile_start(ctx->profiler, &prof);

        op_status = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_set_status(octx, op_status);

        अगर (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].cycles_start = prof.cycles_start;
            pds[i].cycles_stop = prof.cycles_stop;
            क्रम (पूर्णांक j = 0; j < HEX_अंक_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    अगर (ctx->hmx_queue) {
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

    काष्ठा htp_opbatch_rsp rsp;
    स_रखो(&rsp, 0, माप(rsp));
    rsp.seq          = req->seq;
    rsp.status       = octx->status;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.cycles_start = batch_prof.cycles_start;
    rsp.cycles_stop  = batch_prof.cycles_stop;

    अगर (ctx->profiler == HTP_PROF_TRACE) {
        क्रम (पूर्णांक t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].गिनती;
        }
    }

    काष्ठा dspqueue_बफर write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_बफर_FLAG_FLUSH_SENDER | DSPQUEUE_बफर_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(queue, 0, 1, &write_dbuf, माप(rsp), (स्थिर uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    अगर (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (अचिन्हित) err);
    }
}

#घोषणा DSPQUEUE_READ_TIMEOUT_USEC 5000
#घोषणा DSPQUEUE_POLL_TIMEOUT_USEC 100
#घोषणा DSPQUEUE_POLL_गिनती        100

अटल व्योम process_ops(काष्ठा htp_context * ctx) {
    dspqueue_t queue = ctx->dsp_queue;
    पूर्णांक err;

    uint32_t poll_गिनती = DSPQUEUE_POLL_गिनती;

    vtcm_acquire(ctx);

    जबतक (!atomic_load(&ctx->vtcm_needs_release) && !atomic_load(&ctx->killed)) {
        काष्ठा htp_opbatch_req req;
        uint32_t r_आकार = माप(req);

        काष्ठा dspqueue_बफर dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_आकार, &r_आकार, (uint8_t *) &req);
        अगर (err == AEE_EWOULDBLOCK) {
            अगर (--poll_गिनती) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                जारी;
            }
            अवरोध;
        }

        अगर (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (अचिन्हित) err);
            अवरोध;
        }

        अगर (r_आकार < माप(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_आकार, n_dbufs);
            जारी;
        }

        // Reset poll count for valid requests
        poll_गिनती = DSPQUEUE_POLL_गिनती;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

अटल व्योम htp_packet_callback(dspqueue_t queue, पूर्णांक error, व्योम * context) {
    (व्योम) queue;
    (व्योम) error;
    काष्ठा htp_handle * h = (काष्ठा htp_handle *) context;
    अगर (h && h->ctx) {
        process_ops(h->ctx);
    }
}

अटल व्योम htp_main_thread(व्योम * context) {
    काष्ठा htp_context * ctx = (काष्ठा htp_context *) context;

    FARF(ऊपरी_2, "htp-main-thread: started");

    जबतक (!atomic_load(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t अंक_buffers = 0;
        uint32_t message_लंबाई = 0;

        पूर्णांक err = dspqueue_peek(ctx->dsp_queue, &flags, &अंक_buffers, &message_लंबाई, 50000);
        अगर (err == 0) {
            process_ops(ctx);
        } अन्यथा अगर (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            जारी;
        } अन्यथा {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (अचिन्हित) err);
            अवरोध;
        }
    }

    FARF(ऊपरी_2, "htp-main-thread: stopped");
}
