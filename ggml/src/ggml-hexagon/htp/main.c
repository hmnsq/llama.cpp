#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>
#ಅನ್ತರ್ಭಾವ <AEEStdErr.h>
#ಅನ್ತರ್ಭಾವ <dspqueue.h>
#ಅನ್ತರ್ಭಾವ <HAP_compute_res.h>
#ಅನ್ತರ್ಭಾವ <HAP_etm_config.h>
#ಅನ್ತರ್ಭಾವ <HAP_mem.h>
#ಅನ್ತರ್ಭಾವ <HAP_power.h>
#ಅನ್ತರ್ಭಾವ <HAP_ps.h>
#ಅನ್ತರ್ಭಾವ <HAP_dcvs.h>
#ಅನ್ತರ್ಭಾವ <qurt.h>
#ಅನ್ತರ್ಭಾವ <qurt_thread.h>
#ಅನ್ತರ್ಭಾವ <qurt_memory.h>
#ಅನ್ತರ್ಭಾವ <remote.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <stdatomic.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hmx-queue.h"

#define GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "hex-bitmap.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-dcache.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp_iface.h"
#ಅನ್ತರ್ಭಾವ "work-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "allreduce-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-fence.h"

#define HMX_QUEUE_CAPACITY     128
#define HMX_QUEUE_ರಾಶಿ_SIZE   16384
#define WORK_QUEUE_CAPACITY    16
#define WORK_QUEUE_ರಾಶಿ_SIZE  16384
#define MAIN_THREAD_ರಾಶಿ_SIZE 32768

_Static_assert(WORK_QUEUE_MAX_N_THREADS >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

ಸಂರಚನಾ htp_handle {
    ಸಂರಚನಾ htp_context * ctx;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint64_t htp_mmap(uint32_t fd, uint64_t size, uint32_t flags) {
#ಯದಿ __HVX_ARCH__ > 79
    ಯದಿ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, ಪರಿಮಾಣ(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = size;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        ಯದಿ (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (unsigned long long) size);
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        ಪ್ರತ್ಯಾಗಮ payload.mmap.dsp_va;
    }
#ಅನ್ಯಥಾ
    ಯದಿ (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        ಪ್ರತ್ಯಾಗಮ 0;
    }
#endif

    ಯದಿ (size > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (unsigned long long) size);
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಶೂನ್ಯ * va = (ಶೂನ್ಯ *)-1;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ retry = 0; retry < 2; retry++) {
#ಯದಿ __HVX_ARCH__ > 73
        va = HAP_mmap2(ರಿಕ್ತ, (size_t) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#ಅನ್ಯಥಾ
        ಯದಿ (size > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (unsigned long long) size);
            abort();
        }
        va = HAP_mmap(ರಿಕ್ತ, (ಪೂರ್ಣಾಙ್ಕ) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
        ಯದಿ (va != (ಶೂನ್ಯ *)-1 && va != ರಿಕ್ತ) {
            ಪ್ರತ್ಯಾಗಮ (uint64_t) (uintptr_t) va;
        }
        ಯದಿ (retry == 0) {
            FARF(ಮೇಲಿನ_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (unsigned long long) size);
        }
    }
    ಪ್ರತ್ಯಾಗಮ 0;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ htp_munmap(uint64_t va, uint64_t size, uint32_t flags) {
#ಯದಿ __HVX_ARCH__ > 79
    ಯದಿ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, ಪರಿಮಾಣ(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = size;
        HAP_mem_request(&payload);
        ಪ್ರತ್ಯಾಗಮ;
    }
#endif

#ಯದಿ __HVX_ARCH__ > 73
    HAP_munmap2((ಶೂನ್ಯ *) (uintptr_t) va, (size_t) size);
#ಅನ್ಯಥಾ
    HAP_munmap((ಶೂನ್ಯ *) (uintptr_t) va, (ಪೂರ್ಣಾಙ್ಕ) size);
#endif
}

AEEResult htp_iface_open(ಧ್ರುವ ವರ್ಣ * uri, remote_handle64 * handle) {
    (ಶೂನ್ಯ) uri;
    ಸಂರಚನಾ htp_handle * h = calloc(1, ಪರಿಮಾಣ(*h));
    ಯದಿ (h == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    ಸಂರಚನಾ htp_handle * h = (ಸಂರಚನಾ htp_handle *) handle;
    ಯದಿ (!h) {
        ಪ್ರತ್ಯಾಗಮ AEE_EBADPARM;
    }

    ಪೂರ್ಣಾಙ್ಕ err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    ಯದಿ (err) {
        ಯದಿ (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } ಅನ್ಯಥಾ {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    ಪ್ರತ್ಯಾಗಮ err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, ಧ್ರುವ htp_iface_pmu_conf* pmu_conf) {
    ಸಂರಚನಾ htp_handle * h = (ಸಂರಚನಾ htp_handle *) handle;
    ಯದಿ (!h || !h->ctx) {
        ಪ್ರತ್ಯಾಗಮ AEE_EBADPARM;
    }
    ಸಂರಚನಾ htp_context * ctx = h->ctx;

    ಯದಿ (mode == HTP_PROF_PMU) {
        ಧ್ರುವ uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        ಪ್ರತಿ (; i < HEX_ಅಂಕೆ_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        ಪ್ರತಿ (i = 0; i < HEX_ಅಂಕೆ_PMU_COUNTERS; i++) {
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

    ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    ಸಂರಚನಾ htp_handle * h = (ಸಂರಚನಾ htp_handle *) handle;
    ಯದಿ (!h) {
        ಪ್ರತ್ಯಾಗಮ AEE_EBADPARM;
    }

    ಸಂರಚನಾ htp_context * ctx = h->ctx;
    ಯದಿ (ctx) {
        ಯದಿ (ctx->dsp_queue) {
            FARF(ERROR, "Closing handle with queue still open");
            ಪ್ರತ್ಯಾಗಮ AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        ಪ್ರತಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            ಯದಿ (ctx->mmap[i].size) {
                htp_munmap(ctx->mmap[i].base, ctx->mmap[i].size, ctx->mmap[i].flags);
                ctx->mmap[i].size = 0;
                ctx->mmap[i].base = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        ಯದಿ (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        ಯದಿ (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        ಮುಕ್ತ(ctx);
        h->ctx = ರಿಕ್ತ;
    }

    ಮುಕ್ತ(h);
    ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t size) {
    ಸಂರಚನಾ htp_handle * h = (ಸಂರಚನಾ htp_handle *) handle;
    ಯದಿ (!h || !h->ctx) {
        ಪ್ರತ್ಯಾಗಮ AEE_EBADPARM;
    }
    ಸಂರಚನಾ htp_context * ctx = h->ctx;

    // See if we already have this mapping
    ಪ್ರತಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ಸಂರಚನಾ htp_mmap *m = &ctx->mmap[i];
        ಯದಿ (m->fd == fd) {
            ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
        }
    }

    // Add new mapping
    ಪ್ರತಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ಸಂರಚನಾ htp_mmap *m = &ctx->mmap[i];
        ಯದಿ (!m->size) {
            FARF(ಮೇಲಿನ_2, "mmap : fd %u size %llu", fd, (unsigned long long) size);
            uint64_t va = htp_mmap(fd, size, 0);
            ಯದಿ (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (unsigned long long) size);
                ಪ್ರತ್ಯಾಗಮ AEE_EFAILED;
            }

            m->base   = va;
            m->fd     = fd;
            m->size   = size;
            m->flags  = 0;

            ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
        }
    }

    ಪ್ರತ್ಯಾಗಮ AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    ಸಂರಚನಾ htp_handle * h = (ಸಂರಚನಾ htp_handle *) handle;
    ಯದಿ (!h || !h->ctx) {
        ಪ್ರತ್ಯಾಗಮ AEE_EBADPARM;
    }
    ಸಂರಚನಾ htp_context * ctx = h->ctx;

    ಪ್ರತಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ಸಂರಚನಾ htp_mmap *m = &ctx->mmap[i];
        ಯದಿ (fd < 0 || m->fd == fd) {
            FARF(ಮೇಲಿನ_2, "unmmap : base 0x%llx fd %u size %llu", (unsigned long long) m->base, m->fd, (unsigned long long) m->size);
            htp_munmap(m->base, m->size, m->flags);
            m->size   = 0;
            m->base   = ರಿಕ್ತ;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ vtcm_acquire(ಸಂರಚನಾ htp_context * ctx) {
    ಯದಿ (!atomic_load(&ctx->vtcm_valid)) {
        ಪೂರ್ಣಾಙ್ಕ err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        ಯದಿ (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (unsigned)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ vtcm_release(ಸಂರಚನಾ htp_context * ctx) {
    ಯದಿ (atomic_load(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ vtcm_release_callback(unsigned ಪೂರ್ಣಾಙ್ಕ rctx, ಶೂನ್ಯ * state) {
    ಸಂರಚನಾ htp_context * ctx = (ಸಂರಚನಾ htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    ಪ್ರತ್ಯಾಗಮ 0;
}

ಸ್ಥಿರ unsigned ಪೂರ್ಣಾಙ್ಕ vtcm_query_size(ಶೂನ್ಯ) {
    unsigned ಪೂರ್ಣಾಙ್ಕ vtcm_size = 8 * 1024 * 1024;  // 8MB default
    unsigned ಪೂರ್ಣಾಙ್ಕ vtcm_avail;
    compute_res_vtcm_page_t total_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_size, &total_layout, &vtcm_avail, &avail_layout);
    ಪ್ರತ್ಯಾಗಮ vtcm_size;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ vtcm_alloc(ಸಂರಚನಾ htp_context * ctx) {
    unsigned ಪೂರ್ಣಾಙ್ಕ vtcm_size = vtcm_query_size();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (ಶೂನ್ಯ *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    ಯದಿ (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_size);
        ಪ್ರತ್ಯಾಗಮ AEE_ENOMEMORY;
    }

    ಶೂನ್ಯ * vtcm_ptr;
    ಯದಿ (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_size);
        ಪ್ರತ್ಯಾಗಮ AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size          = vtcm_size;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    ಪ್ರತ್ಯಾಗಮ 0;
}

ಸ್ಥಿರ ಶೂನ್ಯ vtcm_free(ಸಂರಚನಾ htp_context * ctx) {
    ಯದಿ (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ htp_main_thread(ಶೂನ್ಯ * context);
ಸ್ಥಿರ ಶೂನ್ಯ htp_packet_callback(dspqueue_t queue, ಪೂರ್ಣಾಙ್ಕ error, ಶೂನ್ಯ * context);
ಸ್ಥಿರ ಶೂನ್ಯ htp_error_callback(dspqueue_t queue, ಪೂರ್ಣಾಙ್ಕ error, ಶೂನ್ಯ * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
ಸಂರಚನಾ htp_power_profile {
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
        .hmx_perf_mode      = HAP_CLK_PERF_ಮೇಲಿನ, \
    }

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ htp_power_profile htp_power_profiles[HTP_POWER_MODE_ಎಣಿಕೆ] = {
    [HTP_POWER_MODE_ಕೆಳಗಿನ_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_MODE_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_MODE_ಮೇಲಿನ_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_MODE_ಕೆಳಗಿನ_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_MODE_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_MODE_ಮೇಲಿನ_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_SUSTAINED_ಮೇಲಿನ_PERFORMANCE] =
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
        .hmx_perf_mode      = HAP_CLK_PERF_ಮೇಲಿನ,
    },
};

#undef HTP_POWER_PROFILE_ENTRY

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_mode) {
    ಸಂರಚನಾ htp_handle * h = (ಸಂರಚನಾ htp_handle *) handle;
    ಯದಿ (!h) {
        ಪ್ರತ್ಯಾಗಮ AEE_EBADPARM;
    }

    ಯದಿ (power_mode >= HTP_POWER_MODE_ಎಣಿಕೆ) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (unsigned) power_mode);
        power_mode = HTP_POWER_MODE_BURST;
    }
    ಧ್ರುವ ಸಂರಚನಾ htp_power_profile * pwr = &htp_power_profiles[power_mode];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (unsigned) power_mode, (ಪೂರ್ಣಾಙ್ಕ) pwr->dcvs_enable,
         (unsigned) pwr->bus_min_corner, (unsigned) pwr->bus_max_corner, (unsigned) pwr->bus_target_corner,
         (unsigned) pwr->core_min_corner, (unsigned) pwr->core_max_corner, (unsigned) pwr->core_target_corner,
         (ಪೂರ್ಣಾಙ್ಕ) pwr->sleep_disable,
         (unsigned) pwr->hmx_min_corner, (unsigned) pwr->hmx_max_corner, (unsigned) pwr->hmx_target_corner);

    ಯದಿ (h->ctx) {
        FARF(ERROR, "Queue already open");
        ಪ್ರತ್ಯಾಗಮ AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    ಪೂರ್ಣಾಙ್ಕ fastrpc_tid  = qurt_thread_get_id();
    ಪೂರ್ಣಾಙ್ಕ fastrpc_prio = qurt_thread_get_priority(fastrpc_tid);
    ಪೂರ್ಣಾಙ್ಕ main_prio    = fastrpc_prio - 10;
    ಯದಿ (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_queue = ರಿಕ್ತ;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    ಪೂರ್ಣಾಙ್ಕ err = dspqueue_import(dsp_queue_id, ರಿಕ್ತ, ರಿಕ್ತ, (ಶೂನ್ಯ *) h, &dsp_queue);
    ಯದಿ (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(ಮೇಲಿನ_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (ಶೂನ್ಯ *) h, &dsp_queue);
    }

    ಯದಿ (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        ಪ್ರತ್ಯಾಗಮ err;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    ಯದಿ (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    ಯದಿ (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    ಯದಿ (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t offset = ಪರಿಮಾಣ(ಸಂರಚನಾ htp_context);

    // 2. main_stack
    size_t offset_main_ರಾಶಿ = 0;
    size_t size_main_ರಾಶಿ   = 0;
    ಯದಿ (!use_callbacks) {
        offset_main_ರಾಶಿ = hex_align_up(offset, 4096);
        size_main_ರಾಶಿ   = MAIN_THREAD_ರಾಶಿ_SIZE;
        offset = offset_main_ರಾಶಿ + size_main_ರಾಶಿ;
    }

    // 3. work_queue
    size_t wq_align  = work_queue_alignof();
    size_t offset_wq = hex_align_up(offset, wq_align);
    size_t size_wq   = work_queue_sizeof(n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_ರಾಶಿ_SIZE);
    offset = offset_wq + size_wq;

    // 4. dma_queue
    size_t dma_align = dma_queue_alignof();
    size_t offset_dma = hex_align_up(offset, dma_align);
    size_t size_dma = 0;
    ಪ್ರತಿ (uint32_t i = 0; i < n_hvx; i++) {
        size_dma  = hex_align_up(size_dma, dma_queue_alignof());
        size_dma += dma_queue_sizeof(256);
    }
    offset = offset_dma + size_dma;

    // 5. hmx_queue
    size_t offset_hmx = 0;
    size_t size_hmx = 0;
    ಯದಿ (n_hmx) {
        size_t hmx_align = hmx_queue_alignof();
        offset_hmx = hex_align_up(offset, hmx_align);
        size_hmx   = hmx_queue_sizeof(HMX_QUEUE_CAPACITY, HMX_QUEUE_ರಾಶಿ_SIZE);
        offset = offset_hmx + size_hmx;
    }

    size_t footprint = hex_align_up(offset, 128);

    ಶೂನ್ಯ * block = memalign(4096, footprint);
    ಯದಿ (!block) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_queue);
        ಪ್ರತ್ಯಾಗಮ AEE_ENOMEMORY;
    }
    memset(block, 0, footprint);

    h->ctx = (ಸಂರಚನಾ htp_context *) block;
    ಸಂರಚನಾ htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_queue   = dsp_queue;

    err = vtcm_alloc(ctx);
    ಯದಿ (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        ಪ್ರತ್ಯಾಗಮ AEE_ENOMEMORY;
    }

    HAP_setFARFRuntimeLoggingParams(0xffff, ರಿಕ್ತ, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, ಪರಿಮಾಣ(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        ಯದಿ ((err = HAP_power_set((ಶೂನ್ಯ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ಪ್ರತ್ಯಾಗಮ err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, ಪರಿಮಾಣ(request));

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

#ಯದಿ (__HEXAGON_ARCH__ >= 79)
        HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
#endif
        ಯದಿ ((err = HAP_power_set((ಶೂನ್ಯ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ಪ್ರತ್ಯಾಗಮ err;
        }

        memset(&request, 0, ಪರಿಮಾಣ(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        ಯದಿ ((err = HAP_power_set((ಶೂನ್ಯ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ಪ್ರತ್ಯಾಗಮ err;
        }
    }

#ಯದಿ __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, ಪರಿಮಾಣ(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_power     = TRUE;
        request.hmx_v2.power_up      = TRUE;
        request.hmx_v2.set_clock     = TRUE;
        request.hmx_v2.target_corner = pwr->hmx_target_corner;
        request.hmx_v2.min_corner    = pwr->hmx_min_corner;
        request.hmx_v2.max_corner    = pwr->hmx_max_corner;
        request.hmx_v2.perf_mode     = pwr->hmx_perf_mode;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((ಶೂನ್ಯ *) ctx, &request);
        ಯದಿ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            ಪ್ರತ್ಯಾಗಮ err;
        }
    }
#ಅನ್ಯಥಾ
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, ಪರಿಮಾಣ(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((ಶೂನ್ಯ *) ctx, &request);
        ಯದಿ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            ಪ್ರತ್ಯಾಗಮ err;
        }
    }
#endif

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_queue   = ರಿಕ್ತ;
    ಯದಿ (n_hmx) {
        ಶೂನ್ಯ * hmx_ptr = (ಶೂನ್ಯ *) ((uintptr_t) block + offset_hmx);
        ctx->hmx_queue = hmx_queue_init(hmx_ptr, HMX_QUEUE_CAPACITY, HMX_QUEUE_ರಾಶಿ_SIZE, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(ಮೇಲಿನ_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_threads = n_hvx;
    ctx->n_threads_div = init_fastdiv_ಮೌಲ್ಯಗಳು(ctx->n_threads);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) block + offset_dma);
    size_t size_dma_q = dma_queue_sizeof(256);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ctx->n_threads; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_up((uintptr_t) dma_ptr_curr, dma_queue_alignof());
        ctx->dma[i] = dma_queue_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += size_dma_q;
    }

    ctx->ddr_spad_size = 512 * 1024; // 512 KB
    ctx->ddr_spad_base = memalign(128, ctx->ddr_spad_size);

    ಶೂನ್ಯ * wq_ptr = (ಶೂನ್ಯ *) ((uintptr_t) block + offset_wq);
    ctx->work_queue = work_queue_init(wq_ptr, n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_ರಾಶಿ_SIZE);

    ctx->main_ರಾಶಿ = ರಿಕ್ತ;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    ಯದಿ (!use_callbacks) {
        // Start main compute thread
        ctx->main_ರಾಶಿ = (ಶೂನ್ಯ *) ((uintptr_t) block + offset_main_ರಾಶಿ);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_ರಾಶಿ_addr(&attr, ctx->main_ರಾಶಿ);
        qurt_thread_attr_set_ರಾಶಿ_size(&attr, size_main_ರಾಶಿ);
        qurt_thread_attr_set_priority(&attr, main_prio);
        qurt_thread_attr_set_name(&attr, "htp-main");

        ಪೂರ್ಣಾಙ್ಕ err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        ಯದಿ (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            ಪ್ರತ್ಯಾಗಮ AEE_ENOMEMORY;
        }
    }

    FARF(ಮೇಲಿನ_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_size, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    ಸಂರಚನಾ htp_handle * h = (ಸಂರಚನಾ htp_handle *) handle;
    ಯದಿ (!h || !h->ctx) {
        ಪ್ರತ್ಯಾಗಮ AEE_EBADPARM;
    }
    ಸಂರಚನಾ htp_context * ctx = h->ctx;

    ಯದಿ (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        ಪೂರ್ಣಾಙ್ಕ status;
        (ಶೂನ್ಯ) qurt_thread_join(ctx->main_thread, &status);
        ctx->main_thread = 0;
    }

    ಪೂರ್ಣಾಙ್ಕ err = dspqueue_close(ctx->dsp_queue); ctx->dsp_queue = ರಿಕ್ತ;
    ಯದಿ (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (unsigned) err);
        ಪ್ರತ್ಯಾಗಮ err;
    }

    work_queue_free(ctx->work_queue);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ctx->n_threads; i++) {
        dma_queue_free(ctx->dma[i]);
    }

    ಯದಿ (ctx->hmx_queue) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = ರಿಕ್ತ;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    ಯದಿ (ctx->ddr_spad_base) {
        ಮುಕ್ತ(ctx->ddr_spad_base);
        ctx->ddr_spad_base = ರಿಕ್ತ;
        ctx->ddr_spad_size = 0;
    }

    ಮುಕ್ತ(ctx);
    h->ctx = ರಿಕ್ತ;

    ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_threads, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_size) {
    (ಶೂನ್ಯ)handle;
    ಯದಿ (!n_threads || !n_hvx || !n_hmx || !vtcm_size) {
        ಪ್ರತ್ಯಾಗಮ AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    ಯದಿ (n_hvx_val > hw_threads.max_hthreads) {
        n_hvx_val = hw_threads.max_hthreads;
    }
    ಯದಿ (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_threads = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_size = vtcm_query_size();

    ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ htp_error_callback(dspqueue_t queue, ಪೂರ್ಣಾಙ್ಕ error, ಶೂನ್ಯ * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (unsigned) error);
}

ಸಂರಚನಾ profile_data {
    uint64_t usecs;
    uint64_t cycles_start;
    uint64_t cycles_stop;
    uint32_t pmu_counters[HEX_ಅಂಕೆ_PMU_COUNTERS];
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ profile_start(uint32_t mode, ಸಂರಚನಾ profile_data * d) {
    ವಿಕಲ್ಪ (mode) {
        ಪಕ್ಷ HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        ಪಕ್ಷ HTP_PROF_BASIC:
        ಪಕ್ಷ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_ಎಣಿಕೆ();
            d->cycles_start = hex_get_cycles();
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ profile_stop(uint32_t mode, ಸಂರಚನಾ profile_data * d) {
    uint32_t pmu_counters[HEX_ಅಂಕೆ_PMU_COUNTERS];
    ವಿಕಲ್ಪ (mode) {
        ಪಕ್ಷ HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < HEX_ಅಂಕೆ_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        ಪಕ್ಷ HTP_PROF_BASIC:
        ಪಕ್ಷ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_ಎಣಿಕೆ_to_us(HAP_perf_get_qtimer_ಎಣಿಕೆ() - d->usecs);
            d->cycles_stop = hex_get_cycles();
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ op_fence(ಸಂರಚನಾ htp_ops_context * octx) {
    ಸಂರಚನಾ htp_context *ctx = octx->ctx;
    ಸಂರಚನಾ htp_thread_trace * tr = &ctx->trace[0];
    ಧ್ರುವ uint32_t seq  = (uint32_t) octx->op_params[0];
    ಧ್ರುವ uint32_t mode = (uint32_t) octx->op_params[1];

    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;

    ಯದಿ (mode == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_group_barrier(octx);

        ಯದಿ (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->status);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(ಮೇಲಿನ_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        ಪ್ರತ್ಯಾಗಮ octx->status;
    }

    ಪೂರ್ಣಾಙ್ಕ status = HTP_STATUS_OK;
    uint64_t spins = 0;
    ಯಾವತ್ (1) {
        uint32_t sync_seq;
        uint32_t sync_status;
        htp_fence_read(sync_fence, &sync_seq, &sync_status);
        ಯದಿ ((int32_t)(sync_seq - seq) >= 0) {
            ಯದಿ (sync_status > HTP_STATUS_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_status, sync_fence, seq);
                status = sync_status;
            }
            ಭಙ್ಗ;
        }
        ಯದಿ (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            status = HTP_STATUS_INTERNAL_ERR;
            ಭಙ್ಗ;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(ಮೇಲಿನ_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    ಪ್ರತ್ಯಾಗಮ status;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ op_mdev_group(ಸಂರಚನಾ htp_ops_context * octx) {
    ಸಂರಚನಾ htp_context * ctx = octx->ctx;
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.ಎಣಿಕೆ = (uint16_t) sync->ne[1];
    ಯದಿ (ctx->mdev.ಎಣಿಕೆ > 1) {
        ctx->mdev.ಎಣಿಕೆ_div = init_fastdiv_ಮೌಲ್ಯಗಳು(ctx->mdev.ಎಣಿಕೆ);
        ctx->mdev.fence_base = (uint8_t *) sync->data;
    }
    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ execute_op(ಸಂರಚನಾ htp_ops_context * octx) {
    ವಿಕಲ್ಪ (octx->op) {
        ಪಕ್ಷ HTP_OP_MDEV_GROUP:
            ಪ್ರತ್ಯಾಗಮ op_mdev_group(octx);

        ಪಕ್ಷ HTP_OP_FENCE:
            ಪ್ರತ್ಯಾಗಮ op_fence(octx);

        ಪಕ್ಷ HTP_OP_ALLREDUCE:
        ಪಕ್ಷ HTP_OP_ALLREDUCE_ADD:
            ಪ್ರತ್ಯಾಗಮ op_allreduce(octx);

        ಪಕ್ಷ HTP_OP_MUL_MAT:
        ಪಕ್ಷ HTP_OP_MUL_MAT_ADD:
            ಪ್ರತ್ಯಾಗಮ op_matmul(octx);

        ಪಕ್ಷ HTP_OP_MUL_MAT_ID:
            ಪ್ರತ್ಯಾಗಮ op_matmul_id(octx);

        ಪಕ್ಷ HTP_OP_MUL_MAT_ID_NX:
            ಪ್ರತ್ಯಾಗಮ op_matmul_id_nx(octx);

        ಪಕ್ಷ HTP_OP_MUL_MAT_NX:
            ಪ್ರತ್ಯಾಗಮ op_matmul_nx(octx);

        ಪಕ್ಷ HTP_OP_MUL:
        ಪಕ್ಷ HTP_OP_ADD:
        ಪಕ್ಷ HTP_OP_SUB:
        ಪಕ್ಷ HTP_OP_DIV:
        ಪಕ್ಷ HTP_OP_ADD_ID:
            ಪ್ರತ್ಯಾಗಮ op_ದ್ವಿಮಾನ(octx);

        ಪಕ್ಷ HTP_OP_NORM:
        ಪಕ್ಷ HTP_OP_RMS_NORM:
        ಪಕ್ಷ HTP_OP_RMS_NORM_MUL:
        ಪಕ್ಷ HTP_OP_SCALE:
        ಪಕ್ಷ HTP_OP_CLAMP:
        ಪಕ್ಷ HTP_OP_LEAKY_RELU:
        ಪಕ್ಷ HTP_OP_SQR:
        ಪಕ್ಷ HTP_OP_SQRT:
        ಪಕ್ಷ HTP_OP_UNARY_SOFTPLUS:
        ಪಕ್ಷ HTP_OP_UNARY_SIGMOID:
        ಪಕ್ಷ HTP_OP_UNARY_SILU:
        ಪಕ್ಷ HTP_OP_UNARY_GELU:
        ಪಕ್ಷ HTP_OP_UNARY_GELU_QUICK:
        ಪಕ್ಷ HTP_OP_UNARY_GELU_ERF:
        ಪಕ್ಷ HTP_OP_UNARY_NEG:
        ಪಕ್ಷ HTP_OP_UNARY_EXP:
        ಪಕ್ಷ HTP_OP_UNARY_TANH:
        ಪಕ್ಷ HTP_OP_UNARY_ABS:
        ಪಕ್ಷ HTP_OP_UNARY_LOG:
        ಪಕ್ಷ HTP_OP_UNARY_RELU:
        ಪಕ್ಷ HTP_OP_UNARY_STEP:
        ಪಕ್ಷ HTP_OP_L2_NORM:
        ಪಕ್ಷ HTP_OP_HADAMARD:
            ಪ್ರತ್ಯಾಗಮ op_unary(octx);

        ಪಕ್ಷ HTP_OP_GLU_SWIGLU:
        ಪಕ್ಷ HTP_OP_GLU_SWIGLU_OAI:
        ಪಕ್ಷ HTP_OP_GLU_SWIGLU_CLAMP:
        ಪಕ್ಷ HTP_OP_GLU_GEGLU:
        ಪಕ್ಷ HTP_OP_GLU_GEGLU_QUICK:
        ಪಕ್ಷ HTP_OP_GLU_GEGLU_ERF:
            ಪ್ರತ್ಯಾಗಮ op_activations(octx);

        ಪಕ್ಷ HTP_OP_SOFTMAX:
            ಪ್ರತ್ಯಾಗಮ op_softmax(octx);

        ಪಕ್ಷ HTP_OP_ROPE:
            ಪ್ರತ್ಯಾಗಮ op_rope(octx);

        ಪಕ್ಷ HTP_OP_FLASH_ATTN_EXT:
            ಪ್ರತ್ಯಾಗಮ op_flash_attn_ext(octx);

        ಪಕ್ಷ HTP_OP_SET_ROWS:
            ಪ್ರತ್ಯಾಗಮ op_set_rows(octx);

        ಪಕ್ಷ HTP_OP_GET_ROWS:
            ಪ್ರತ್ಯಾಗಮ op_get_rows(octx);

        ಪಕ್ಷ HTP_OP_SUM:
            ಪ್ರತ್ಯಾಗಮ op_sum(octx);

        ಪಕ್ಷ HTP_OP_SUM_ROWS:
            ಪ್ರತ್ಯಾಗಮ op_sum_rows(octx);

        ಪಕ್ಷ HTP_OP_CPY:
        ಪಕ್ಷ HTP_OP_CPY_FENCE:
            ಪ್ರತ್ಯಾಗಮ op_cpy(octx);

        ಪಕ್ಷ HTP_OP_REPEAT:
            ಪ್ರತ್ಯಾಗಮ op_repeat(octx);

        ಪಕ್ಷ HTP_OP_ARGSORT:
            ಪ್ರತ್ಯಾಗಮ op_argsort(octx);

        ಪಕ್ಷ HTP_OP_ತುದಿ_K:
            ಪ್ರತ್ಯಾಗಮ op_ತುದಿ_k(octx);

        ಪಕ್ಷ HTP_OP_ARGMAX:
            ಪ್ರತ್ಯಾಗಮ op_argmax(octx);

        ಪಕ್ಷ HTP_OP_SSM_CONV:
            ಪ್ರತ್ಯಾಗಮ op_ssm_conv(octx);

        ಪಕ್ಷ HTP_OP_CUMSUM:
            ಪ್ರತ್ಯಾಗಮ op_cumsum(octx);

        ಪಕ್ಷ HTP_OP_FILL:
            ಪ್ರತ್ಯಾಗಮ op_fill(octx);

        ಪಕ್ಷ HTP_OP_DIAG:
            ಪ್ರತ್ಯಾಗಮ op_diag(octx);

        ಪಕ್ಷ HTP_OP_SOLVE_TRI:
            ಪ್ರತ್ಯಾಗಮ op_solve_tri(octx);

        ಪಕ್ಷ HTP_OP_PAD:
            ಪ್ರತ್ಯಾಗಮ op_pad(octx);

        ಪಕ್ಷ HTP_OP_IM2COL:
            ಪ್ರತ್ಯಾಗಮ op_im2col(octx);

        ಪಕ್ಷ HTP_OP_IM2COL_3D:
            ಪ್ರತ್ಯಾಗಮ op_im2col_3d(octx);

        ಪಕ್ಷ HTP_OP_ROLL:
            ಪ್ರತ್ಯಾಗಮ op_roll(octx);

        ಪಕ್ಷ HTP_OP_POOL_2D:
            ಪ್ರತ್ಯಾಗಮ op_pool_2d(octx);

        ಪಕ್ಷ HTP_OP_POOL_1D:
            ಪ್ರತ್ಯಾಗಮ op_pool_1d(octx);

        ಪಕ್ಷ HTP_OP_CONCAT:
            ಪ್ರತ್ಯಾಗಮ op_concat(octx);

        ಪಕ್ಷ HTP_OP_GATED_DELTA_NET:
            ಪ್ರತ್ಯಾಗಮ op_gated_delta_net(octx);

        ಪಕ್ಷ HTP_OP_TRI:
            ಪ್ರತ್ಯಾಗಮ op_unary(octx);

        ಪಕ್ಷ HTP_OP_INVALID:
            ಭಙ್ಗ;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool reuse_buf(ಸಂರಚನಾ htp_context *ctx, uint32_t *m_reuse, ಸಂರಚನಾ htp_buf_desc *b) {
    b->base = ರಿಕ್ತ;

    ಪ್ರತಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ಸಂರಚನಾ htp_mmap *m = ctx->mmap + i;
        ಯದಿ (m->size && m->fd == b->fd && m->flags == b->flags) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            ಪ್ರತ್ಯಾಗಮ true;
        }
    }

    ಪ್ರತ್ಯಾಗಮ false;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ drop_mmap(ಸಂರಚನಾ htp_context *ctx, ಸಂರಚನಾ htp_mmap *m) {
    ಯದಿ (m->size) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (unsigned long long) m->base, (unsigned long long) m->size);
        htp_munmap(m->base, m->size, m->flags);
        m->size = 0;
        m->base = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool mmap_buf(ಸಂರಚನಾ htp_context *ctx, ಸಂರಚನಾ htp_buf_desc *b) {
    ಯದಿ (b->base) ಪ್ರತ್ಯಾಗಮ true; // already mapped

    // find unused mapping
    ಪ್ರತಿ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        ಸಂರಚನಾ htp_mmap *m = &ctx->mmap[i];
        ಯದಿ (!m->size) {
            uint64_t va = htp_mmap(b->fd, b->size, b->flags);
            ಯದಿ (va == 0) {
                FARF(ಮೇಲಿನ_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (unsigned long long) b->size);
                ಪ್ರತ್ಯಾಗಮ false;
            }

            m->base   = b->base = va;
            m->fd     = b->fd;
            m->size   = b->size;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (unsigned long long) m->base, (unsigned long long) m->size, m->flags);
            ಪ್ರತ್ಯಾಗಮ true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    ಪ್ರತ್ಯಾಗಮ false;
}

ಸ್ಥಿರ ಶೂನ್ಯ prep_op_bufs(ಸಂರಚನಾ htp_context *ctx, ಸಂರಚನಾ htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    ಪ್ರತಿ (uint32_t i=0; i < n_bufs; i++) {
        ಸಂರಚನಾ htp_buf_desc *b = bufs + i;
        ಯದಿ (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } ಅನ್ಯಥಾ ಯದಿ (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->size;
        }
        FARF(ಮೇಲಿನ_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    ಯದಿ (b_reuse == n_bufs) ಪ್ರತ್ಯಾಗಮ; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    ಪ್ರತಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ಯದಿ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].size;
        }
    }

    FARF(ಮೇಲಿನ_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    ಯದಿ ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        ಪ್ರತಿ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            ಯದಿ (!used && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    ಪ್ರತಿ (uint32_t i=0; i < n_bufs; i++) {
        ಸಂರಚನಾ htp_buf_desc *b = bufs + i;
        ಯದಿ (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            ಭಙ್ಗ;
        }
        FARF(ಮೇಲಿನ_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    ಯದಿ (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(ಮೇಲಿನ_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        ಪ್ರತಿ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            ಯದಿ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        ಪ್ರತಿ (uint32_t i=0; i < n_bufs; i++) {
            ಸಂರಚನಾ htp_buf_desc *b = bufs + i;
            ಯದಿ (!(b->flags & HTP_BUF_EXTENDED)) {
                b->base = 0;
            }
            ಯದಿ (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (unsigned long long) b->size);
                abort();
            }
            FARF(ಮೇಲಿನ_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ prep_tensor(ಸಂರಚನಾ htp_context *ctx, ಸಂರಚನಾ htp_buf_desc *bufs, ಸಂರಚನಾ htp_tensor *tens, uint32_t idx, ಸಂರಚನಾ htp_tensor *t) {
    uint64_t offset = t->data;
    uint32_t bi     = t->bi;

    t->data  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(ಮೇಲಿನ_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (unsigned long long) offset, t->size, (unsigned long long) t->data,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

ಸ್ಥಿರ ಶೂನ್ಯ prep_tensors(ಸಂರಚನಾ htp_context *ctx, ಸಂರಚನಾ htp_buf_desc *bufs, ಸಂರಚನಾ htp_tensor *tens, uint32_t n_tens) {
    ಪ್ರತಿ (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mdev_group_init(ಸಂರಚನಾ htp_context * ctx, ಧ್ರುವ ಸಂರಚನಾ htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, ಪರಿಮಾಣ(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ proc_op_req(ಸಂರಚನಾ htp_ops_context * octx, ಸಂರಚನಾ htp_buf_desc * bufs, uint32_t n_bufs,
                       ಸಂರಚನಾ htp_tensor * tens, uint32_t idx, ಸಂರಚನಾ htp_op_desc * op) {
    memcpy(octx->op_params,     op->params, ಪರಿಮಾಣ(octx->op_params));
    memcpy(octx->kernel_params, op->kernel_params, ಪರಿಮಾಣ(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_threads     = octx->ctx->n_threads;
    octx->n_threads_div = octx->ctx->n_threads_div;

    FARF(ಮೇಲಿನ_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    ಪ್ರತಿ (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        ಯದಿ (src_idx == 0xffff) {
            octx->src[i]     = ರಿಕ್ತ;
            ಅನುವರ್ತ;
        }

        ಸಂರಚನಾ htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(ಮೇಲಿನ_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (unsigned long long) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    ಪ್ರತಿ (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        ಯದಿ (dst_idx == 0xffff) {
            octx->dsts[i]    = ರಿಕ್ತ;
            ಅನುವರ್ತ;
        }
        ಸಂರಚನಾ htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(ಮೇಲಿನ_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (unsigned long long) dst->data, dst->size,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_group_barrier(octx);

    ಪೂರ್ಣಾಙ್ಕ status = execute_op(octx);

    htp_ops_context_set_status(octx, status);

    octx->src0_spad.src = ರಿಕ್ತ;
    octx->src1_spad.src = ರಿಕ್ತ;
    octx->src2_spad.src = ರಿಕ್ತ;
    octx->src3_spad.src = ರಿಕ್ತ;
    octx->dst_spad.src  = ರಿಕ್ತ;

    ಪ್ರತ್ಯಾಗಮ octx->status;
}

ಸ್ಥಿರ ಶೂನ್ಯ process_opbatch(ಸಂರಚನಾ htp_context * ctx, ಧ್ರುವ ಸಂರಚನಾ htp_opbatch_req * req, ಧ್ರುವ ಸಂರಚನಾ dspqueue_buffer * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    ಪೂರ್ಣಾಙ್ಕ err;

    ಧ್ರುವ uint32_t n_bufs = req->n_bufs;
    ಧ್ರುವ uint32_t n_tens = req->n_tensors;
    ಧ್ರುವ uint32_t n_ops  = req->n_ops;

    ಧ್ರುವ uint32_t b_size = ಪರಿಮಾಣ(ಸಂರಚನಾ htp_buf_desc)  * n_bufs;
    ಧ್ರುವ uint32_t t_size = ಪರಿಮಾಣ(ಸಂರಚನಾ htp_tensor)    * n_tens;
    ಧ್ರುವ uint32_t o_size = ಪರಿಮಾಣ(ಸಂರಚನಾ htp_op_desc)   * n_ops;
    ಧ್ರುವ uint32_t p_size = ಪರಿಮಾಣ(ಸಂರಚನಾ htp_prof_desc) * n_ops;
    ಧ್ರುವ uint32_t tr_size = (HTP_MAX_NTHREADS + 1) * req->n_traces * ಪರಿಮಾಣ(ಸಂರಚನಾ htp_trace_desc);

    ಯದಿ (dbuf->size < b_size + t_size + o_size + p_size + tr_size) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->size, b_size + t_size + o_size + p_size + tr_size);
        ಪ್ರತ್ಯಾಗಮ;
    }

    FARF(ಮೇಲಿನ_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (unsigned long long) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->size, b_size, t_size, o_size);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    ಸಂರಚನಾ htp_buf_desc* bufs = (ಸಂರಚನಾ htp_buf_desc*)  m_ptr; m_ptr += b_size;
    ಸಂರಚನಾ htp_tensor*   tens = (ಸಂರಚನಾ htp_tensor*)    m_ptr; m_ptr += t_size;
    ಸಂರಚನಾ htp_op_desc*   ops = (ಸಂರಚನಾ htp_op_desc*)   m_ptr; m_ptr += o_size;
    ಸಂರಚನಾ htp_prof_desc* pds = (ಸಂರಚನಾ htp_prof_desc*) m_ptr;

    ಸಂರಚನಾ profile_data batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, ಪರಿಮಾಣ(ctx->trace));
    ಯದಿ (ctx->profiler == HTP_PROF_TRACE) {
        ಸಂರಚನಾ htp_trace_desc * trace_events = (ಸಂರಚನಾ htp_trace_desc *) (m_ptr + p_size);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, ಪರಿಮಾಣ(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    ಸಂರಚನಾ htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, ಪರಿಮಾಣ(*octx));
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
    octx->ctx           = ctx;

    mdev_group_init(ctx, req);

    work_queue_wakeup(ctx->work_queue);
    ಯದಿ (ctx->hmx_queue) {
        hmx_queue_wakeup(ctx->hmx_queue);
    }

    ಪೂರ್ಣಾಙ್ಕ op_status = HTP_STATUS_OK;
    octx->status  = HTP_STATUS_OK;
    ಪ್ರತಿ (uint32_t i = 0; i < n_ops; i++) {
        ಸಂರಚನಾ profile_data prof;

        profile_start(ctx->profiler, &prof);

        op_status = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_set_status(octx, op_status);

        ಯದಿ (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].cycles_start = prof.cycles_start;
            pds[i].cycles_stop = prof.cycles_stop;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < HEX_ಅಂಕೆ_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    ಯದಿ (ctx->hmx_queue) {
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

    ಸಂರಚನಾ htp_opbatch_rsp rsp;
    memset(&rsp, 0, ಪರಿಮಾಣ(rsp));
    rsp.seq          = req->seq;
    rsp.status       = octx->status;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.cycles_start = batch_prof.cycles_start;
    rsp.cycles_stop  = batch_prof.cycles_stop;

    ಯದಿ (ctx->profiler == HTP_PROF_TRACE) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].ಎಣಿಕೆ;
        }
    }

    ಸಂರಚನಾ dspqueue_buffer write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(queue, 0, 1, &write_dbuf, ಪರಿಮಾಣ(rsp), (ಧ್ರುವ uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    ಯದಿ (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
    }
}

#define DSPQUEUE_READ_TIMEOUT_USEC 5000
#define DSPQUEUE_POLL_TIMEOUT_USEC 100
#define DSPQUEUE_POLL_ಎಣಿಕೆ        100

ಸ್ಥಿರ ಶೂನ್ಯ process_ops(ಸಂರಚನಾ htp_context * ctx) {
    dspqueue_t queue = ctx->dsp_queue;
    ಪೂರ್ಣಾಙ್ಕ err;

    uint32_t poll_ಎಣಿಕೆ = DSPQUEUE_POLL_ಎಣಿಕೆ;

    vtcm_acquire(ctx);

    ಯಾವತ್ (!atomic_load(&ctx->vtcm_needs_release) && !atomic_load(&ctx->killed)) {
        ಸಂರಚನಾ htp_opbatch_req req;
        uint32_t r_size = ಪರಿಮಾಣ(req);

        ಸಂರಚನಾ dspqueue_buffer dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_size, &r_size, (uint8_t *) &req);
        ಯದಿ (err == AEE_EWOULDBLOCK) {
            ಯದಿ (--poll_ಎಣಿಕೆ) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                ಅನುವರ್ತ;
            }
            ಭಙ್ಗ;
        }

        ಯದಿ (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            ಭಙ್ಗ;
        }

        ಯದಿ (r_size < ಪರಿಮಾಣ(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_size, n_dbufs);
            ಅನುವರ್ತ;
        }

        // Reset poll count for valid requests
        poll_ಎಣಿಕೆ = DSPQUEUE_POLL_ಎಣಿಕೆ;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ htp_packet_callback(dspqueue_t queue, ಪೂರ್ಣಾಙ್ಕ error, ಶೂನ್ಯ * context) {
    (ಶೂನ್ಯ) queue;
    (ಶೂನ್ಯ) error;
    ಸಂರಚನಾ htp_handle * h = (ಸಂರಚನಾ htp_handle *) context;
    ಯದಿ (h && h->ctx) {
        process_ops(h->ctx);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ htp_main_thread(ಶೂನ್ಯ * context) {
    ಸಂರಚನಾ htp_context * ctx = (ಸಂರಚನಾ htp_context *) context;

    FARF(ಮೇಲಿನ_2, "htp-main-thread: started");

    ಯಾವತ್ (!atomic_load(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t ಅಂಕೆ_buffers = 0;
        uint32_t message_length = 0;

        ಪೂರ್ಣಾಙ್ಕ err = dspqueue_peek(ctx->dsp_queue, &flags, &ಅಂಕೆ_buffers, &message_length, 50000);
        ಯದಿ (err == 0) {
            process_ops(ctx);
        } ಅನ್ಯಥಾ ಯದಿ (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            ಅನುವರ್ತ;
        } ಅನ್ಯಥಾ {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (unsigned) err);
            ಭಙ್ಗ;
        }
    }

    FARF(ಮೇಲಿನ_2, "htp-main-thread: stopped");
}
