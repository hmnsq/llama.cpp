#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <HAP_perf.h>
#அந்தர்பாவ <AEEStdErr.h>
#அந்தர்பாவ <dspqueue.h>
#அந்தர்பாவ <HAP_compute_res.h>
#அந்தர்பாவ <HAP_etm_config.h>
#அந்தர்பாவ <HAP_mem.h>
#அந்தர்பாவ <HAP_power.h>
#அந்தர்பாவ <HAP_ps.h>
#அந்தர்பாவ <HAP_dcvs.h>
#அந்தர்பாவ <qurt.h>
#அந்தர்பாவ <qurt_thread.h>
#அந்தர்பாவ <qurt_memory.h>
#அந்தர்பாவ <remote.h>
#அந்தர்பாவ <string.h>
#அந்தர்பாவ <stdatomic.h>

#அந்தர்பாவ "hex-utils.h"
#அந்தர்பாவ "dma-queue.h"
#அந்தர்பாவ "hmx-queue.h"

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "hex-bitmap.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-dcache.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp_iface.h"
#அந்தர்பாவ "work-queue.h"
#அந்தர்பாவ "hex-profile.h"
#அந்தர்பாவ "allreduce-ops.h"
#அந்தர்பாவ "htp-fence.h"

#define HMX_QUEUE_CAPACITY     128
#define HMX_QUEUE_அடுக்கு_SIZE   16384
#define WORK_QUEUE_CAPACITY    16
#define WORK_QUEUE_அடுக்கு_SIZE  16384
#define MAIN_THREAD_அடுக்கு_SIZE 32768

_Static_assert(WORK_QUEUE_MAX_N_THREADS >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

கட்டமைப்பு htp_handle {
    கட்டமைப்பு htp_context * ctx;
};

நிலையான உள்வரி uint64_t htp_mmap(uint32_t fd, uint64_t size, uint32_t flags) {
#எனில் __HVX_ARCH__ > 79
    எனில் (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, அளவு(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = size;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        எனில் (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (unsigned long long) size);
            திரும்பு 0;
        }

        திரும்பு payload.mmap.dsp_va;
    }
#இல்லையெனில்
    எனில் (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        திரும்பு 0;
    }
#endif

    எனில் (size > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (unsigned long long) size);
        திரும்பு 0;
    }

    வெறுமை * va = (வெறுமை *)-1;
    க்கு (முழு retry = 0; retry < 2; retry++) {
#எனில் __HVX_ARCH__ > 73
        va = HAP_mmap2(வெற்று, (size_t) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#இல்லையெனில்
        எனில் (size > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (unsigned long long) size);
            abort();
        }
        va = HAP_mmap(வெற்று, (முழு) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
        எனில் (va != (வெறுமை *)-1 && va != வெற்று) {
            திரும்பு (uint64_t) (uintptr_t) va;
        }
        எனில் (retry == 0) {
            FARF(உயர்வு_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (unsigned long long) size);
        }
    }
    திரும்பு 0;
}

நிலையான உள்வரி வெறுமை htp_munmap(uint64_t va, uint64_t size, uint32_t flags) {
#எனில் __HVX_ARCH__ > 79
    எனில் (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, அளவு(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = size;
        HAP_mem_request(&payload);
        திரும்பு;
    }
#endif

#எனில் __HVX_ARCH__ > 73
    HAP_munmap2((வெறுமை *) (uintptr_t) va, (size_t) size);
#இல்லையெனில்
    HAP_munmap((வெறுமை *) (uintptr_t) va, (முழு) size);
#endif
}

AEEResult htp_iface_open(மாறிலி எழுத்து * uri, remote_handle64 * handle) {
    (வெறுமை) uri;
    கட்டமைப்பு htp_handle * h = calloc(1, அளவு(*h));
    எனில் (h == வெற்று) {
        திரும்பு AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    திரும்பு AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    கட்டமைப்பு htp_handle * h = (கட்டமைப்பு htp_handle *) handle;
    எனில் (!h) {
        திரும்பு AEE_EBADPARM;
    }

    முழு err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    எனில் (err) {
        எனில் (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } இல்லையெனில் {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    திரும்பு err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, மாறிலி htp_iface_pmu_conf* pmu_conf) {
    கட்டமைப்பு htp_handle * h = (கட்டமைப்பு htp_handle *) handle;
    எனில் (!h || !h->ctx) {
        திரும்பு AEE_EBADPARM;
    }
    கட்டமைப்பு htp_context * ctx = h->ctx;

    எனில் (mode == HTP_PROF_PMU) {
        மாறிலி uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        க்கு (; i < HEX_இலக்கம்_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        க்கு (i = 0; i < HEX_இலக்கம்_PMU_COUNTERS; i++) {
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

    திரும்பு AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    கட்டமைப்பு htp_handle * h = (கட்டமைப்பு htp_handle *) handle;
    எனில் (!h) {
        திரும்பு AEE_EBADPARM;
    }

    கட்டமைப்பு htp_context * ctx = h->ctx;
    எனில் (ctx) {
        எனில் (ctx->dsp_queue) {
            FARF(ERROR, "Closing handle with queue still open");
            திரும்பு AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        க்கு (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            எனில் (ctx->mmap[i].size) {
                htp_munmap(ctx->mmap[i].base, ctx->mmap[i].size, ctx->mmap[i].flags);
                ctx->mmap[i].size = 0;
                ctx->mmap[i].base = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        எனில் (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        எனில் (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        விடுவி(ctx);
        h->ctx = வெற்று;
    }

    விடுவி(h);
    திரும்பு AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t size) {
    கட்டமைப்பு htp_handle * h = (கட்டமைப்பு htp_handle *) handle;
    எனில் (!h || !h->ctx) {
        திரும்பு AEE_EBADPARM;
    }
    கட்டமைப்பு htp_context * ctx = h->ctx;

    // See if we already have this mapping
    க்கு (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        கட்டமைப்பு htp_mmap *m = &ctx->mmap[i];
        எனில் (m->fd == fd) {
            திரும்பு AEE_SUCCESS;
        }
    }

    // Add new mapping
    க்கு (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        கட்டமைப்பு htp_mmap *m = &ctx->mmap[i];
        எனில் (!m->size) {
            FARF(உயர்வு_2, "mmap : fd %u size %llu", fd, (unsigned long long) size);
            uint64_t va = htp_mmap(fd, size, 0);
            எனில் (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (unsigned long long) size);
                திரும்பு AEE_EFAILED;
            }

            m->base   = va;
            m->fd     = fd;
            m->size   = size;
            m->flags  = 0;

            திரும்பு AEE_SUCCESS;
        }
    }

    திரும்பு AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    கட்டமைப்பு htp_handle * h = (கட்டமைப்பு htp_handle *) handle;
    எனில் (!h || !h->ctx) {
        திரும்பு AEE_EBADPARM;
    }
    கட்டமைப்பு htp_context * ctx = h->ctx;

    க்கு (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        கட்டமைப்பு htp_mmap *m = &ctx->mmap[i];
        எனில் (fd < 0 || m->fd == fd) {
            FARF(உயர்வு_2, "unmmap : base 0x%llx fd %u size %llu", (unsigned long long) m->base, m->fd, (unsigned long long) m->size);
            htp_munmap(m->base, m->size, m->flags);
            m->size   = 0;
            m->base   = வெற்று;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    திரும்பு AEE_SUCCESS;
}

நிலையான வெறுமை vtcm_acquire(கட்டமைப்பு htp_context * ctx) {
    எனில் (!atomic_load(&ctx->vtcm_valid)) {
        முழு err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        எனில் (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (unsigned)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

நிலையான வெறுமை vtcm_release(கட்டமைப்பு htp_context * ctx) {
    எனில் (atomic_load(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

நிலையான முழு vtcm_release_callback(unsigned முழு rctx, வெறுமை * state) {
    கட்டமைப்பு htp_context * ctx = (கட்டமைப்பு htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    திரும்பு 0;
}

நிலையான unsigned முழு vtcm_query_size(வெறுமை) {
    unsigned முழு vtcm_size = 8 * 1024 * 1024;  // 8MB default
    unsigned முழு vtcm_avail;
    compute_res_vtcm_page_t total_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_size, &total_layout, &vtcm_avail, &avail_layout);
    திரும்பு vtcm_size;
}

நிலையான முழு vtcm_alloc(கட்டமைப்பு htp_context * ctx) {
    unsigned முழு vtcm_size = vtcm_query_size();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (வெறுமை *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    எனில் (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_size);
        திரும்பு AEE_ENOMEMORY;
    }

    வெறுமை * vtcm_ptr;
    எனில் (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_size);
        திரும்பு AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size          = vtcm_size;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    திரும்பு 0;
}

நிலையான வெறுமை vtcm_free(கட்டமைப்பு htp_context * ctx) {
    எனில் (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

நிலையான வெறுமை htp_main_thread(வெறுமை * context);
நிலையான வெறுமை htp_packet_callback(dspqueue_t queue, முழு error, வெறுமை * context);
நிலையான வெறுமை htp_error_callback(dspqueue_t queue, முழு error, வெறுமை * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
கட்டமைப்பு htp_power_profile {
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
        .hmx_perf_mode      = HAP_CLK_PERF_உயர்வு, \
    }

நிலையான மாறிலி கட்டமைப்பு htp_power_profile htp_power_profiles[HTP_POWER_MODE_மொத்தம்] = {
    [HTP_POWER_MODE_கீழ்_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_MODE_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_MODE_உயர்வு_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_MODE_கீழ்_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_MODE_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_MODE_உயர்வு_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_SUSTAINED_உயர்வு_PERFORMANCE] =
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
        .hmx_perf_mode      = HAP_CLK_PERF_உயர்வு,
    },
};

#undef HTP_POWER_PROFILE_ENTRY

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_mode) {
    கட்டமைப்பு htp_handle * h = (கட்டமைப்பு htp_handle *) handle;
    எனில் (!h) {
        திரும்பு AEE_EBADPARM;
    }

    எனில் (power_mode >= HTP_POWER_MODE_மொத்தம்) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (unsigned) power_mode);
        power_mode = HTP_POWER_MODE_BURST;
    }
    மாறிலி கட்டமைப்பு htp_power_profile * pwr = &htp_power_profiles[power_mode];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (unsigned) power_mode, (முழு) pwr->dcvs_enable,
         (unsigned) pwr->bus_min_corner, (unsigned) pwr->bus_max_corner, (unsigned) pwr->bus_target_corner,
         (unsigned) pwr->core_min_corner, (unsigned) pwr->core_max_corner, (unsigned) pwr->core_target_corner,
         (முழு) pwr->sleep_disable,
         (unsigned) pwr->hmx_min_corner, (unsigned) pwr->hmx_max_corner, (unsigned) pwr->hmx_target_corner);

    எனில் (h->ctx) {
        FARF(ERROR, "Queue already open");
        திரும்பு AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    முழு fastrpc_tid  = qurt_thread_get_id();
    முழு fastrpc_prio = qurt_thread_get_priority(fastrpc_tid);
    முழு main_prio    = fastrpc_prio - 10;
    எனில் (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_queue = வெற்று;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    முழு err = dspqueue_import(dsp_queue_id, வெற்று, வெற்று, (வெறுமை *) h, &dsp_queue);
    எனில் (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(உயர்வு_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (வெறுமை *) h, &dsp_queue);
    }

    எனில் (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        திரும்பு err;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    எனில் (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    எனில் (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    எனில் (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t offset = அளவு(கட்டமைப்பு htp_context);

    // 2. main_stack
    size_t offset_main_அடுக்கு = 0;
    size_t size_main_அடுக்கு   = 0;
    எனில் (!use_callbacks) {
        offset_main_அடுக்கு = hex_align_up(offset, 4096);
        size_main_அடுக்கு   = MAIN_THREAD_அடுக்கு_SIZE;
        offset = offset_main_அடுக்கு + size_main_அடுக்கு;
    }

    // 3. work_queue
    size_t wq_align  = work_queue_alignof();
    size_t offset_wq = hex_align_up(offset, wq_align);
    size_t size_wq   = work_queue_sizeof(n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_அடுக்கு_SIZE);
    offset = offset_wq + size_wq;

    // 4. dma_queue
    size_t dma_align = dma_queue_alignof();
    size_t offset_dma = hex_align_up(offset, dma_align);
    size_t size_dma = 0;
    க்கு (uint32_t i = 0; i < n_hvx; i++) {
        size_dma  = hex_align_up(size_dma, dma_queue_alignof());
        size_dma += dma_queue_sizeof(256);
    }
    offset = offset_dma + size_dma;

    // 5. hmx_queue
    size_t offset_hmx = 0;
    size_t size_hmx = 0;
    எனில் (n_hmx) {
        size_t hmx_align = hmx_queue_alignof();
        offset_hmx = hex_align_up(offset, hmx_align);
        size_hmx   = hmx_queue_sizeof(HMX_QUEUE_CAPACITY, HMX_QUEUE_அடுக்கு_SIZE);
        offset = offset_hmx + size_hmx;
    }

    size_t footprint = hex_align_up(offset, 128);

    வெறுமை * block = memalign(4096, footprint);
    எனில் (!block) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_queue);
        திரும்பு AEE_ENOMEMORY;
    }
    memset(block, 0, footprint);

    h->ctx = (கட்டமைப்பு htp_context *) block;
    கட்டமைப்பு htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_queue   = dsp_queue;

    err = vtcm_alloc(ctx);
    எனில் (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        திரும்பு AEE_ENOMEMORY;
    }

    HAP_setFARFRuntimeLoggingParams(0xffff, வெற்று, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, அளவு(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        எனில் ((err = HAP_power_set((வெறுமை *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            திரும்பு err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, அளவு(request));

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

#எனில் (__HEXAGON_ARCH__ >= 79)
        HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
#endif
        எனில் ((err = HAP_power_set((வெறுமை *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            திரும்பு err;
        }

        memset(&request, 0, அளவு(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        எனில் ((err = HAP_power_set((வெறுமை *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            திரும்பு err;
        }
    }

#எனில் __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, அளவு(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_power     = TRUE;
        request.hmx_v2.power_up      = TRUE;
        request.hmx_v2.set_clock     = TRUE;
        request.hmx_v2.target_corner = pwr->hmx_target_corner;
        request.hmx_v2.min_corner    = pwr->hmx_min_corner;
        request.hmx_v2.max_corner    = pwr->hmx_max_corner;
        request.hmx_v2.perf_mode     = pwr->hmx_perf_mode;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((வெறுமை *) ctx, &request);
        எனில் (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            திரும்பு err;
        }
    }
#இல்லையெனில்
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, அளவு(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((வெறுமை *) ctx, &request);
        எனில் (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            திரும்பு err;
        }
    }
#endif

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_queue   = வெற்று;
    எனில் (n_hmx) {
        வெறுமை * hmx_ptr = (வெறுமை *) ((uintptr_t) block + offset_hmx);
        ctx->hmx_queue = hmx_queue_init(hmx_ptr, HMX_QUEUE_CAPACITY, HMX_QUEUE_அடுக்கு_SIZE, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(உயர்வு_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_threads = n_hvx;
    ctx->n_threads_div = init_fastdiv_மதிப்புகள்(ctx->n_threads);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) block + offset_dma);
    size_t size_dma_q = dma_queue_sizeof(256);

    க்கு (முழு i = 0; i < ctx->n_threads; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_up((uintptr_t) dma_ptr_curr, dma_queue_alignof());
        ctx->dma[i] = dma_queue_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += size_dma_q;
    }

    ctx->ddr_spad_size = 512 * 1024; // 512 KB
    ctx->ddr_spad_base = memalign(128, ctx->ddr_spad_size);

    வெறுமை * wq_ptr = (வெறுமை *) ((uintptr_t) block + offset_wq);
    ctx->work_queue = work_queue_init(wq_ptr, n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_அடுக்கு_SIZE);

    ctx->main_அடுக்கு = வெற்று;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    எனில் (!use_callbacks) {
        // Start main compute thread
        ctx->main_அடுக்கு = (வெறுமை *) ((uintptr_t) block + offset_main_அடுக்கு);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_அடுக்கு_addr(&attr, ctx->main_அடுக்கு);
        qurt_thread_attr_set_அடுக்கு_size(&attr, size_main_அடுக்கு);
        qurt_thread_attr_set_priority(&attr, main_prio);
        qurt_thread_attr_set_name(&attr, "htp-main");

        முழு err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        எனில் (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            திரும்பு AEE_ENOMEMORY;
        }
    }

    FARF(உயர்வு_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_size, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    திரும்பு AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    கட்டமைப்பு htp_handle * h = (கட்டமைப்பு htp_handle *) handle;
    எனில் (!h || !h->ctx) {
        திரும்பு AEE_EBADPARM;
    }
    கட்டமைப்பு htp_context * ctx = h->ctx;

    எனில் (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        முழு status;
        (வெறுமை) qurt_thread_join(ctx->main_thread, &status);
        ctx->main_thread = 0;
    }

    முழு err = dspqueue_close(ctx->dsp_queue); ctx->dsp_queue = வெற்று;
    எனில் (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (unsigned) err);
        திரும்பு err;
    }

    work_queue_free(ctx->work_queue);

    க்கு (முழு i = 0; i < ctx->n_threads; i++) {
        dma_queue_free(ctx->dma[i]);
    }

    எனில் (ctx->hmx_queue) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = வெற்று;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    எனில் (ctx->ddr_spad_base) {
        விடுவி(ctx->ddr_spad_base);
        ctx->ddr_spad_base = வெற்று;
        ctx->ddr_spad_size = 0;
    }

    விடுவி(ctx);
    h->ctx = வெற்று;

    திரும்பு AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_threads, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_size) {
    (வெறுமை)handle;
    எனில் (!n_threads || !n_hvx || !n_hmx || !vtcm_size) {
        திரும்பு AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    எனில் (n_hvx_val > hw_threads.max_hthreads) {
        n_hvx_val = hw_threads.max_hthreads;
    }
    எனில் (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_threads = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_size = vtcm_query_size();

    திரும்பு AEE_SUCCESS;
}

நிலையான வெறுமை htp_error_callback(dspqueue_t queue, முழு error, வெறுமை * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (unsigned) error);
}

கட்டமைப்பு profile_data {
    uint64_t usecs;
    uint64_t cycles_start;
    uint64_t cycles_stop;
    uint32_t pmu_counters[HEX_இலக்கம்_PMU_COUNTERS];
};

நிலையான உள்வரி வெறுமை profile_start(uint32_t mode, கட்டமைப்பு profile_data * d) {
    தேர்வு (mode) {
        நிலை HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        நிலை HTP_PROF_BASIC:
        நிலை HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_மொத்தம்();
            d->cycles_start = hex_get_cycles();
            நிறுத்து;
        இயல்பு:
            நிறுத்து;
    }
}

நிலையான உள்வரி வெறுமை profile_stop(uint32_t mode, கட்டமைப்பு profile_data * d) {
    uint32_t pmu_counters[HEX_இலக்கம்_PMU_COUNTERS];
    தேர்வு (mode) {
        நிலை HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            க்கு (முழு i = 0; i < HEX_இலக்கம்_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        நிலை HTP_PROF_BASIC:
        நிலை HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_மொத்தம்_to_us(HAP_perf_get_qtimer_மொத்தம்() - d->usecs);
            d->cycles_stop = hex_get_cycles();
            நிறுத்து;
        இயல்பு:
            நிறுத்து;
    }
}

நிலையான முழு op_fence(கட்டமைப்பு htp_ops_context * octx) {
    கட்டமைப்பு htp_context *ctx = octx->ctx;
    கட்டமைப்பு htp_thread_trace * tr = &ctx->trace[0];
    மாறிலி uint32_t seq  = (uint32_t) octx->op_params[0];
    மாறிலி uint32_t mode = (uint32_t) octx->op_params[1];

    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    மாறிலி கட்டமைப்பு htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;

    எனில் (mode == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_group_barrier(octx);

        எனில் (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->status);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(உயர்வு_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        திரும்பு octx->status;
    }

    முழு status = HTP_STATUS_OK;
    uint64_t spins = 0;
    போது (1) {
        uint32_t sync_seq;
        uint32_t sync_status;
        htp_fence_read(sync_fence, &sync_seq, &sync_status);
        எனில் ((int32_t)(sync_seq - seq) >= 0) {
            எனில் (sync_status > HTP_STATUS_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_status, sync_fence, seq);
                status = sync_status;
            }
            நிறுத்து;
        }
        எனில் (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            status = HTP_STATUS_INTERNAL_ERR;
            நிறுத்து;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(உயர்வு_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    திரும்பு status;
}

நிலையான முழு op_mdev_group(கட்டமைப்பு htp_ops_context * octx) {
    கட்டமைப்பு htp_context * ctx = octx->ctx;
    மாறிலி கட்டமைப்பு htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.மொத்தம் = (uint16_t) sync->ne[1];
    எனில் (ctx->mdev.மொத்தம் > 1) {
        ctx->mdev.மொத்தம்_div = init_fastdiv_மதிப்புகள்(ctx->mdev.மொத்தம்);
        ctx->mdev.fence_base = (uint8_t *) sync->data;
    }
    திரும்பு HTP_STATUS_OK;
}

நிலையான முழு execute_op(கட்டமைப்பு htp_ops_context * octx) {
    தேர்வு (octx->op) {
        நிலை HTP_OP_MDEV_GROUP:
            திரும்பு op_mdev_group(octx);

        நிலை HTP_OP_FENCE:
            திரும்பு op_fence(octx);

        நிலை HTP_OP_ALLREDUCE:
        நிலை HTP_OP_ALLREDUCE_ADD:
            திரும்பு op_allreduce(octx);

        நிலை HTP_OP_MUL_MAT:
        நிலை HTP_OP_MUL_MAT_ADD:
            திரும்பு op_matmul(octx);

        நிலை HTP_OP_MUL_MAT_ID:
            திரும்பு op_matmul_id(octx);

        நிலை HTP_OP_MUL_MAT_ID_NX:
            திரும்பு op_matmul_id_nx(octx);

        நிலை HTP_OP_MUL_MAT_NX:
            திரும்பு op_matmul_nx(octx);

        நிலை HTP_OP_MUL:
        நிலை HTP_OP_ADD:
        நிலை HTP_OP_SUB:
        நிலை HTP_OP_DIV:
        நிலை HTP_OP_ADD_ID:
            திரும்பு op_இருமம்(octx);

        நிலை HTP_OP_NORM:
        நிலை HTP_OP_RMS_NORM:
        நிலை HTP_OP_RMS_NORM_MUL:
        நிலை HTP_OP_SCALE:
        நிலை HTP_OP_CLAMP:
        நிலை HTP_OP_LEAKY_RELU:
        நிலை HTP_OP_SQR:
        நிலை HTP_OP_SQRT:
        நிலை HTP_OP_UNARY_SOFTPLUS:
        நிலை HTP_OP_UNARY_SIGMOID:
        நிலை HTP_OP_UNARY_SILU:
        நிலை HTP_OP_UNARY_GELU:
        நிலை HTP_OP_UNARY_GELU_QUICK:
        நிலை HTP_OP_UNARY_GELU_ERF:
        நிலை HTP_OP_UNARY_NEG:
        நிலை HTP_OP_UNARY_EXP:
        நிலை HTP_OP_UNARY_TANH:
        நிலை HTP_OP_UNARY_ABS:
        நிலை HTP_OP_UNARY_LOG:
        நிலை HTP_OP_UNARY_RELU:
        நிலை HTP_OP_UNARY_STEP:
        நிலை HTP_OP_L2_NORM:
        நிலை HTP_OP_HADAMARD:
            திரும்பு op_unary(octx);

        நிலை HTP_OP_GLU_SWIGLU:
        நிலை HTP_OP_GLU_SWIGLU_OAI:
        நிலை HTP_OP_GLU_SWIGLU_CLAMP:
        நிலை HTP_OP_GLU_GEGLU:
        நிலை HTP_OP_GLU_GEGLU_QUICK:
        நிலை HTP_OP_GLU_GEGLU_ERF:
            திரும்பு op_activations(octx);

        நிலை HTP_OP_SOFTMAX:
            திரும்பு op_softmax(octx);

        நிலை HTP_OP_ROPE:
            திரும்பு op_rope(octx);

        நிலை HTP_OP_FLASH_ATTN_EXT:
            திரும்பு op_flash_attn_ext(octx);

        நிலை HTP_OP_SET_ROWS:
            திரும்பு op_set_rows(octx);

        நிலை HTP_OP_GET_ROWS:
            திரும்பு op_get_rows(octx);

        நிலை HTP_OP_SUM:
            திரும்பு op_sum(octx);

        நிலை HTP_OP_SUM_ROWS:
            திரும்பு op_sum_rows(octx);

        நிலை HTP_OP_CPY:
        நிலை HTP_OP_CPY_FENCE:
            திரும்பு op_cpy(octx);

        நிலை HTP_OP_REPEAT:
            திரும்பு op_repeat(octx);

        நிலை HTP_OP_ARGSORT:
            திரும்பு op_argsort(octx);

        நிலை HTP_OP_உச்சி_K:
            திரும்பு op_உச்சி_k(octx);

        நிலை HTP_OP_ARGMAX:
            திரும்பு op_argmax(octx);

        நிலை HTP_OP_SSM_CONV:
            திரும்பு op_ssm_conv(octx);

        நிலை HTP_OP_CUMSUM:
            திரும்பு op_cumsum(octx);

        நிலை HTP_OP_FILL:
            திரும்பு op_fill(octx);

        நிலை HTP_OP_DIAG:
            திரும்பு op_diag(octx);

        நிலை HTP_OP_SOLVE_TRI:
            திரும்பு op_solve_tri(octx);

        நிலை HTP_OP_PAD:
            திரும்பு op_pad(octx);

        நிலை HTP_OP_IM2COL:
            திரும்பு op_im2col(octx);

        நிலை HTP_OP_IM2COL_3D:
            திரும்பு op_im2col_3d(octx);

        நிலை HTP_OP_ROLL:
            திரும்பு op_roll(octx);

        நிலை HTP_OP_POOL_2D:
            திரும்பு op_pool_2d(octx);

        நிலை HTP_OP_POOL_1D:
            திரும்பு op_pool_1d(octx);

        நிலை HTP_OP_CONCAT:
            திரும்பு op_concat(octx);

        நிலை HTP_OP_GATED_DELTA_NET:
            திரும்பு op_gated_delta_net(octx);

        நிலை HTP_OP_TRI:
            திரும்பு op_unary(octx);

        நிலை HTP_OP_INVALID:
            நிறுத்து;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    திரும்பு HTP_STATUS_NO_SUPPORT;
}

நிலையான உள்வரி bool reuse_buf(கட்டமைப்பு htp_context *ctx, uint32_t *m_reuse, கட்டமைப்பு htp_buf_desc *b) {
    b->base = வெற்று;

    க்கு (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        கட்டமைப்பு htp_mmap *m = ctx->mmap + i;
        எனில் (m->size && m->fd == b->fd && m->flags == b->flags) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            திரும்பு true;
        }
    }

    திரும்பு false;
}

நிலையான உள்வரி வெறுமை drop_mmap(கட்டமைப்பு htp_context *ctx, கட்டமைப்பு htp_mmap *m) {
    எனில் (m->size) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (unsigned long long) m->base, (unsigned long long) m->size);
        htp_munmap(m->base, m->size, m->flags);
        m->size = 0;
        m->base = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

நிலையான உள்வரி bool mmap_buf(கட்டமைப்பு htp_context *ctx, கட்டமைப்பு htp_buf_desc *b) {
    எனில் (b->base) திரும்பு true; // already mapped

    // find unused mapping
    க்கு (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        கட்டமைப்பு htp_mmap *m = &ctx->mmap[i];
        எனில் (!m->size) {
            uint64_t va = htp_mmap(b->fd, b->size, b->flags);
            எனில் (va == 0) {
                FARF(உயர்வு_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (unsigned long long) b->size);
                திரும்பு false;
            }

            m->base   = b->base = va;
            m->fd     = b->fd;
            m->size   = b->size;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (unsigned long long) m->base, (unsigned long long) m->size, m->flags);
            திரும்பு true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    திரும்பு false;
}

நிலையான வெறுமை prep_op_bufs(கட்டமைப்பு htp_context *ctx, கட்டமைப்பு htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    க்கு (uint32_t i=0; i < n_bufs; i++) {
        கட்டமைப்பு htp_buf_desc *b = bufs + i;
        எனில் (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } இல்லையெனில் எனில் (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->size;
        }
        FARF(உயர்வு_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    எனில் (b_reuse == n_bufs) திரும்பு; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    க்கு (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        எனில் (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].size;
        }
    }

    FARF(உயர்வு_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    எனில் ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        க்கு (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            எனில் (!used && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    க்கு (uint32_t i=0; i < n_bufs; i++) {
        கட்டமைப்பு htp_buf_desc *b = bufs + i;
        எனில் (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            நிறுத்து;
        }
        FARF(உயர்வு_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    எனில் (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(உயர்வு_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        க்கு (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            எனில் (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        க்கு (uint32_t i=0; i < n_bufs; i++) {
            கட்டமைப்பு htp_buf_desc *b = bufs + i;
            எனில் (!(b->flags & HTP_BUF_EXTENDED)) {
                b->base = 0;
            }
            எனில் (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (unsigned long long) b->size);
                abort();
            }
            FARF(உயர்வு_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
        }
    }
}

நிலையான வெறுமை prep_tensor(கட்டமைப்பு htp_context *ctx, கட்டமைப்பு htp_buf_desc *bufs, கட்டமைப்பு htp_tensor *tens, uint32_t idx, கட்டமைப்பு htp_tensor *t) {
    uint64_t offset = t->data;
    uint32_t bi     = t->bi;

    t->data  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(உயர்வு_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (unsigned long long) offset, t->size, (unsigned long long) t->data,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

நிலையான வெறுமை prep_tensors(கட்டமைப்பு htp_context *ctx, கட்டமைப்பு htp_buf_desc *bufs, கட்டமைப்பு htp_tensor *tens, uint32_t n_tens) {
    க்கு (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

நிலையான வெறுமை mdev_group_init(கட்டமைப்பு htp_context * ctx, மாறிலி கட்டமைப்பு htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, அளவு(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

நிலையான முழு proc_op_req(கட்டமைப்பு htp_ops_context * octx, கட்டமைப்பு htp_buf_desc * bufs, uint32_t n_bufs,
                       கட்டமைப்பு htp_tensor * tens, uint32_t idx, கட்டமைப்பு htp_op_desc * op) {
    memcpy(octx->op_params,     op->params, அளவு(octx->op_params));
    memcpy(octx->kernel_params, op->kernel_params, அளவு(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_threads     = octx->ctx->n_threads;
    octx->n_threads_div = octx->ctx->n_threads_div;

    FARF(உயர்வு_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    க்கு (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        எனில் (src_idx == 0xffff) {
            octx->src[i]     = வெற்று;
            தொடர்;
        }

        கட்டமைப்பு htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(உயர்வு_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (unsigned long long) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    க்கு (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        எனில் (dst_idx == 0xffff) {
            octx->dsts[i]    = வெற்று;
            தொடர்;
        }
        கட்டமைப்பு htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(உயர்வு_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (unsigned long long) dst->data, dst->size,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_group_barrier(octx);

    முழு status = execute_op(octx);

    htp_ops_context_set_status(octx, status);

    octx->src0_spad.src = வெற்று;
    octx->src1_spad.src = வெற்று;
    octx->src2_spad.src = வெற்று;
    octx->src3_spad.src = வெற்று;
    octx->dst_spad.src  = வெற்று;

    திரும்பு octx->status;
}

நிலையான வெறுமை process_opbatch(கட்டமைப்பு htp_context * ctx, மாறிலி கட்டமைப்பு htp_opbatch_req * req, மாறிலி கட்டமைப்பு dspqueue_buffer * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    முழு err;

    மாறிலி uint32_t n_bufs = req->n_bufs;
    மாறிலி uint32_t n_tens = req->n_tensors;
    மாறிலி uint32_t n_ops  = req->n_ops;

    மாறிலி uint32_t b_size = அளவு(கட்டமைப்பு htp_buf_desc)  * n_bufs;
    மாறிலி uint32_t t_size = அளவு(கட்டமைப்பு htp_tensor)    * n_tens;
    மாறிலி uint32_t o_size = அளவு(கட்டமைப்பு htp_op_desc)   * n_ops;
    மாறிலி uint32_t p_size = அளவு(கட்டமைப்பு htp_prof_desc) * n_ops;
    மாறிலி uint32_t tr_size = (HTP_MAX_NTHREADS + 1) * req->n_traces * அளவு(கட்டமைப்பு htp_trace_desc);

    எனில் (dbuf->size < b_size + t_size + o_size + p_size + tr_size) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->size, b_size + t_size + o_size + p_size + tr_size);
        திரும்பு;
    }

    FARF(உயர்வு_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (unsigned long long) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->size, b_size, t_size, o_size);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    கட்டமைப்பு htp_buf_desc* bufs = (கட்டமைப்பு htp_buf_desc*)  m_ptr; m_ptr += b_size;
    கட்டமைப்பு htp_tensor*   tens = (கட்டமைப்பு htp_tensor*)    m_ptr; m_ptr += t_size;
    கட்டமைப்பு htp_op_desc*   ops = (கட்டமைப்பு htp_op_desc*)   m_ptr; m_ptr += o_size;
    கட்டமைப்பு htp_prof_desc* pds = (கட்டமைப்பு htp_prof_desc*) m_ptr;

    கட்டமைப்பு profile_data batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, அளவு(ctx->trace));
    எனில் (ctx->profiler == HTP_PROF_TRACE) {
        கட்டமைப்பு htp_trace_desc * trace_events = (கட்டமைப்பு htp_trace_desc *) (m_ptr + p_size);
        க்கு (முழு t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, அளவு(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    கட்டமைப்பு htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, அளவு(*octx));
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
    octx->ctx           = ctx;

    mdev_group_init(ctx, req);

    work_queue_wakeup(ctx->work_queue);
    எனில் (ctx->hmx_queue) {
        hmx_queue_wakeup(ctx->hmx_queue);
    }

    முழு op_status = HTP_STATUS_OK;
    octx->status  = HTP_STATUS_OK;
    க்கு (uint32_t i = 0; i < n_ops; i++) {
        கட்டமைப்பு profile_data prof;

        profile_start(ctx->profiler, &prof);

        op_status = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_set_status(octx, op_status);

        எனில் (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].cycles_start = prof.cycles_start;
            pds[i].cycles_stop = prof.cycles_stop;
            க்கு (முழு j = 0; j < HEX_இலக்கம்_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    எனில் (ctx->hmx_queue) {
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

    கட்டமைப்பு htp_opbatch_rsp rsp;
    memset(&rsp, 0, அளவு(rsp));
    rsp.seq          = req->seq;
    rsp.status       = octx->status;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.cycles_start = batch_prof.cycles_start;
    rsp.cycles_stop  = batch_prof.cycles_stop;

    எனில் (ctx->profiler == HTP_PROF_TRACE) {
        க்கு (முழு t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].மொத்தம்;
        }
    }

    கட்டமைப்பு dspqueue_buffer write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(queue, 0, 1, &write_dbuf, அளவு(rsp), (மாறிலி uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    எனில் (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
    }
}

#define DSPQUEUE_READ_TIMEOUT_USEC 5000
#define DSPQUEUE_POLL_TIMEOUT_USEC 100
#define DSPQUEUE_POLL_மொத்தம்        100

நிலையான வெறுமை process_ops(கட்டமைப்பு htp_context * ctx) {
    dspqueue_t queue = ctx->dsp_queue;
    முழு err;

    uint32_t poll_மொத்தம் = DSPQUEUE_POLL_மொத்தம்;

    vtcm_acquire(ctx);

    போது (!atomic_load(&ctx->vtcm_needs_release) && !atomic_load(&ctx->killed)) {
        கட்டமைப்பு htp_opbatch_req req;
        uint32_t r_size = அளவு(req);

        கட்டமைப்பு dspqueue_buffer dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_size, &r_size, (uint8_t *) &req);
        எனில் (err == AEE_EWOULDBLOCK) {
            எனில் (--poll_மொத்தம்) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                தொடர்;
            }
            நிறுத்து;
        }

        எனில் (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            நிறுத்து;
        }

        எனில் (r_size < அளவு(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_size, n_dbufs);
            தொடர்;
        }

        // Reset poll count for valid requests
        poll_மொத்தம் = DSPQUEUE_POLL_மொத்தம்;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

நிலையான வெறுமை htp_packet_callback(dspqueue_t queue, முழு error, வெறுமை * context) {
    (வெறுமை) queue;
    (வெறுமை) error;
    கட்டமைப்பு htp_handle * h = (கட்டமைப்பு htp_handle *) context;
    எனில் (h && h->ctx) {
        process_ops(h->ctx);
    }
}

நிலையான வெறுமை htp_main_thread(வெறுமை * context) {
    கட்டமைப்பு htp_context * ctx = (கட்டமைப்பு htp_context *) context;

    FARF(உயர்வு_2, "htp-main-thread: started");

    போது (!atomic_load(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t இலக்கம்_buffers = 0;
        uint32_t message_length = 0;

        முழு err = dspqueue_peek(ctx->dsp_queue, &flags, &இலக்கம்_buffers, &message_length, 50000);
        எனில் (err == 0) {
            process_ops(ctx);
        } இல்லையெனில் எனில் (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            தொடர்;
        } இல்லையெனில் {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (unsigned) err);
            நிறுத்து;
        }
    }

    FARF(உயர்வு_2, "htp-main-thread: stopped");
}
