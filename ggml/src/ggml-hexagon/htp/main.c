#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <HAP_perf.h>
#അന്തര്ഭാവ <AEEStdErr.h>
#അന്തര്ഭാവ <dspqueue.h>
#അന്തര്ഭാവ <HAP_compute_res.h>
#അന്തര്ഭാവ <HAP_etm_config.h>
#അന്തര്ഭാവ <HAP_mem.h>
#അന്തര്ഭാവ <HAP_power.h>
#അന്തര്ഭാവ <HAP_ps.h>
#അന്തര്ഭാവ <HAP_dcvs.h>
#അന്തര്ഭാവ <qurt.h>
#അന്തര്ഭാവ <qurt_thread.h>
#അന്തര്ഭാവ <qurt_memory.h>
#അന്തര്ഭാവ <remote.h>
#അന്തര്ഭാവ <string.h>
#അന്തര്ഭാവ <stdatomic.h>

#അന്തര്ഭാവ "hex-utils.h"
#അന്തര്ഭാവ "dma-queue.h"
#അന്തര്ഭാവ "hmx-queue.h"

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "hex-bitmap.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "htp-dcache.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp_iface.h"
#അന്തര്ഭാവ "work-queue.h"
#അന്തര്ഭാവ "hex-profile.h"
#അന്തര്ഭാവ "allreduce-ops.h"
#അന്തര്ഭാവ "htp-fence.h"

#define HMX_QUEUE_CAPACITY     128
#define HMX_QUEUE_അട്ടി_SIZE   16384
#define WORK_QUEUE_CAPACITY    16
#define WORK_QUEUE_അട്ടി_SIZE  16384
#define MAIN_THREAD_അട്ടി_SIZE 32768

_Static_assert(WORK_QUEUE_MAX_N_THREADS >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

ഘടന htp_handle {
    ഘടന htp_context * ctx;
};

സ്ഥിതം വരിയിൽ uint64_t htp_mmap(uint32_t fd, uint64_t size, uint32_t flags) {
#എങ്കിൽ __HVX_ARCH__ > 79
    എങ്കിൽ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, വലുപ്പം(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = size;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        എങ്കിൽ (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (unsigned long long) size);
            തിരികെ 0;
        }

        തിരികെ payload.mmap.dsp_va;
    }
#അല്ലെങ്കിൽ
    എങ്കിൽ (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        തിരികെ 0;
    }
#endif

    എങ്കിൽ (size > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (unsigned long long) size);
        തിരികെ 0;
    }

    ശൂന്യം * va = (ശൂന്യം *)-1;
    വേണ്ടി (പൂർണ്ണസംഖ്യ retry = 0; retry < 2; retry++) {
#എങ്കിൽ __HVX_ARCH__ > 73
        va = HAP_mmap2(ശൂന്യ, (size_t) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#അല്ലെങ്കിൽ
        എങ്കിൽ (size > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (unsigned long long) size);
            abort();
        }
        va = HAP_mmap(ശൂന്യ, (പൂർണ്ണസംഖ്യ) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
        എങ്കിൽ (va != (ശൂന്യം *)-1 && va != ശൂന്യ) {
            തിരികെ (uint64_t) (uintptr_t) va;
        }
        എങ്കിൽ (retry == 0) {
            FARF(മുകളിലെ_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (unsigned long long) size);
        }
    }
    തിരികെ 0;
}

സ്ഥിതം വരിയിൽ ശൂന്യം htp_munmap(uint64_t va, uint64_t size, uint32_t flags) {
#എങ്കിൽ __HVX_ARCH__ > 79
    എങ്കിൽ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, വലുപ്പം(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = size;
        HAP_mem_request(&payload);
        തിരികെ;
    }
#endif

#എങ്കിൽ __HVX_ARCH__ > 73
    HAP_munmap2((ശൂന്യം *) (uintptr_t) va, (size_t) size);
#അല്ലെങ്കിൽ
    HAP_munmap((ശൂന്യം *) (uintptr_t) va, (പൂർണ്ണസംഖ്യ) size);
#endif
}

AEEResult htp_iface_open(സ്ഥിരം അക്ഷരം * uri, remote_handle64 * handle) {
    (ശൂന്യം) uri;
    ഘടന htp_handle * h = calloc(1, വലുപ്പം(*h));
    എങ്കിൽ (h == ശൂന്യ) {
        തിരികെ AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    തിരികെ AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    ഘടന htp_handle * h = (ഘടന htp_handle *) handle;
    എങ്കിൽ (!h) {
        തിരികെ AEE_EBADPARM;
    }

    പൂർണ്ണസംഖ്യ err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    എങ്കിൽ (err) {
        എങ്കിൽ (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } അല്ലെങ്കിൽ {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    തിരികെ err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, സ്ഥിരം htp_iface_pmu_conf* pmu_conf) {
    ഘടന htp_handle * h = (ഘടന htp_handle *) handle;
    എങ്കിൽ (!h || !h->ctx) {
        തിരികെ AEE_EBADPARM;
    }
    ഘടന htp_context * ctx = h->ctx;

    എങ്കിൽ (mode == HTP_PROF_PMU) {
        സ്ഥിരം uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        വേണ്ടി (; i < HEX_അക്കം_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        വേണ്ടി (i = 0; i < HEX_അക്കം_PMU_COUNTERS; i++) {
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

    തിരികെ AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    ഘടന htp_handle * h = (ഘടന htp_handle *) handle;
    എങ്കിൽ (!h) {
        തിരികെ AEE_EBADPARM;
    }

    ഘടന htp_context * ctx = h->ctx;
    എങ്കിൽ (ctx) {
        എങ്കിൽ (ctx->dsp_queue) {
            FARF(ERROR, "Closing handle with queue still open");
            തിരികെ AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        വേണ്ടി (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            എങ്കിൽ (ctx->mmap[i].size) {
                htp_munmap(ctx->mmap[i].base, ctx->mmap[i].size, ctx->mmap[i].flags);
                ctx->mmap[i].size = 0;
                ctx->mmap[i].base = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        എങ്കിൽ (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        എങ്കിൽ (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        സ്വതന്ത്രം(ctx);
        h->ctx = ശൂന്യ;
    }

    സ്വതന്ത്രം(h);
    തിരികെ AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t size) {
    ഘടന htp_handle * h = (ഘടന htp_handle *) handle;
    എങ്കിൽ (!h || !h->ctx) {
        തിരികെ AEE_EBADPARM;
    }
    ഘടന htp_context * ctx = h->ctx;

    // See if we already have this mapping
    വേണ്ടി (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ഘടന htp_mmap *m = &ctx->mmap[i];
        എങ്കിൽ (m->fd == fd) {
            തിരികെ AEE_SUCCESS;
        }
    }

    // Add new mapping
    വേണ്ടി (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ഘടന htp_mmap *m = &ctx->mmap[i];
        എങ്കിൽ (!m->size) {
            FARF(മുകളിലെ_2, "mmap : fd %u size %llu", fd, (unsigned long long) size);
            uint64_t va = htp_mmap(fd, size, 0);
            എങ്കിൽ (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (unsigned long long) size);
                തിരികെ AEE_EFAILED;
            }

            m->base   = va;
            m->fd     = fd;
            m->size   = size;
            m->flags  = 0;

            തിരികെ AEE_SUCCESS;
        }
    }

    തിരികെ AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    ഘടന htp_handle * h = (ഘടന htp_handle *) handle;
    എങ്കിൽ (!h || !h->ctx) {
        തിരികെ AEE_EBADPARM;
    }
    ഘടന htp_context * ctx = h->ctx;

    വേണ്ടി (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ഘടന htp_mmap *m = &ctx->mmap[i];
        എങ്കിൽ (fd < 0 || m->fd == fd) {
            FARF(മുകളിലെ_2, "unmmap : base 0x%llx fd %u size %llu", (unsigned long long) m->base, m->fd, (unsigned long long) m->size);
            htp_munmap(m->base, m->size, m->flags);
            m->size   = 0;
            m->base   = ശൂന്യ;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    തിരികെ AEE_SUCCESS;
}

സ്ഥിതം ശൂന്യം vtcm_acquire(ഘടന htp_context * ctx) {
    എങ്കിൽ (!atomic_load(&ctx->vtcm_valid)) {
        പൂർണ്ണസംഖ്യ err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        എങ്കിൽ (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (unsigned)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

സ്ഥിതം ശൂന്യം vtcm_release(ഘടന htp_context * ctx) {
    എങ്കിൽ (atomic_load(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

സ്ഥിതം പൂർണ്ണസംഖ്യ vtcm_release_callback(unsigned പൂർണ്ണസംഖ്യ rctx, ശൂന്യം * state) {
    ഘടന htp_context * ctx = (ഘടന htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    തിരികെ 0;
}

സ്ഥിതം unsigned പൂർണ്ണസംഖ്യ vtcm_query_size(ശൂന്യം) {
    unsigned പൂർണ്ണസംഖ്യ vtcm_size = 8 * 1024 * 1024;  // 8MB default
    unsigned പൂർണ്ണസംഖ്യ vtcm_avail;
    compute_res_vtcm_page_t total_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_size, &total_layout, &vtcm_avail, &avail_layout);
    തിരികെ vtcm_size;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ vtcm_alloc(ഘടന htp_context * ctx) {
    unsigned പൂർണ്ണസംഖ്യ vtcm_size = vtcm_query_size();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (ശൂന്യം *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    എങ്കിൽ (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_size);
        തിരികെ AEE_ENOMEMORY;
    }

    ശൂന്യം * vtcm_ptr;
    എങ്കിൽ (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_size);
        തിരികെ AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size          = vtcm_size;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    തിരികെ 0;
}

സ്ഥിതം ശൂന്യം vtcm_free(ഘടന htp_context * ctx) {
    എങ്കിൽ (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

സ്ഥിതം ശൂന്യം htp_main_thread(ശൂന്യം * context);
സ്ഥിതം ശൂന്യം htp_packet_callback(dspqueue_t queue, പൂർണ്ണസംഖ്യ error, ശൂന്യം * context);
സ്ഥിതം ശൂന്യം htp_error_callback(dspqueue_t queue, പൂർണ്ണസംഖ്യ error, ശൂന്യം * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
ഘടന htp_power_profile {
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
        .hmx_perf_mode      = HAP_CLK_PERF_മുകളിലെ, \
    }

സ്ഥിതം സ്ഥിരം ഘടന htp_power_profile htp_power_profiles[HTP_POWER_MODE_എണ്ണം] = {
    [HTP_POWER_MODE_താഴെയുള്ള_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_MODE_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_MODE_മുകളിലെ_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_MODE_താഴെയുള്ള_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_MODE_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_MODE_മുകളിലെ_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_SUSTAINED_മുകളിലെ_PERFORMANCE] =
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
        .hmx_perf_mode      = HAP_CLK_PERF_മുകളിലെ,
    },
};

#undef HTP_POWER_PROFILE_ENTRY

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_mode) {
    ഘടന htp_handle * h = (ഘടന htp_handle *) handle;
    എങ്കിൽ (!h) {
        തിരികെ AEE_EBADPARM;
    }

    എങ്കിൽ (power_mode >= HTP_POWER_MODE_എണ്ണം) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (unsigned) power_mode);
        power_mode = HTP_POWER_MODE_BURST;
    }
    സ്ഥിരം ഘടന htp_power_profile * pwr = &htp_power_profiles[power_mode];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (unsigned) power_mode, (പൂർണ്ണസംഖ്യ) pwr->dcvs_enable,
         (unsigned) pwr->bus_min_corner, (unsigned) pwr->bus_max_corner, (unsigned) pwr->bus_target_corner,
         (unsigned) pwr->core_min_corner, (unsigned) pwr->core_max_corner, (unsigned) pwr->core_target_corner,
         (പൂർണ്ണസംഖ്യ) pwr->sleep_disable,
         (unsigned) pwr->hmx_min_corner, (unsigned) pwr->hmx_max_corner, (unsigned) pwr->hmx_target_corner);

    എങ്കിൽ (h->ctx) {
        FARF(ERROR, "Queue already open");
        തിരികെ AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    പൂർണ്ണസംഖ്യ fastrpc_tid  = qurt_thread_get_id();
    പൂർണ്ണസംഖ്യ fastrpc_prio = qurt_thread_get_priority(fastrpc_tid);
    പൂർണ്ണസംഖ്യ main_prio    = fastrpc_prio - 10;
    എങ്കിൽ (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_queue = ശൂന്യ;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    പൂർണ്ണസംഖ്യ err = dspqueue_import(dsp_queue_id, ശൂന്യ, ശൂന്യ, (ശൂന്യം *) h, &dsp_queue);
    എങ്കിൽ (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(മുകളിലെ_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (ശൂന്യം *) h, &dsp_queue);
    }

    എങ്കിൽ (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        തിരികെ err;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    എങ്കിൽ (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    എങ്കിൽ (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    എങ്കിൽ (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t offset = വലുപ്പം(ഘടന htp_context);

    // 2. main_stack
    size_t offset_main_അട്ടി = 0;
    size_t size_main_അട്ടി   = 0;
    എങ്കിൽ (!use_callbacks) {
        offset_main_അട്ടി = hex_align_up(offset, 4096);
        size_main_അട്ടി   = MAIN_THREAD_അട്ടി_SIZE;
        offset = offset_main_അട്ടി + size_main_അട്ടി;
    }

    // 3. work_queue
    size_t wq_align  = work_queue_alignof();
    size_t offset_wq = hex_align_up(offset, wq_align);
    size_t size_wq   = work_queue_sizeof(n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_അട്ടി_SIZE);
    offset = offset_wq + size_wq;

    // 4. dma_queue
    size_t dma_align = dma_queue_alignof();
    size_t offset_dma = hex_align_up(offset, dma_align);
    size_t size_dma = 0;
    വേണ്ടി (uint32_t i = 0; i < n_hvx; i++) {
        size_dma  = hex_align_up(size_dma, dma_queue_alignof());
        size_dma += dma_queue_sizeof(256);
    }
    offset = offset_dma + size_dma;

    // 5. hmx_queue
    size_t offset_hmx = 0;
    size_t size_hmx = 0;
    എങ്കിൽ (n_hmx) {
        size_t hmx_align = hmx_queue_alignof();
        offset_hmx = hex_align_up(offset, hmx_align);
        size_hmx   = hmx_queue_sizeof(HMX_QUEUE_CAPACITY, HMX_QUEUE_അട്ടി_SIZE);
        offset = offset_hmx + size_hmx;
    }

    size_t footprint = hex_align_up(offset, 128);

    ശൂന്യം * block = memalign(4096, footprint);
    എങ്കിൽ (!block) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_queue);
        തിരികെ AEE_ENOMEMORY;
    }
    memset(block, 0, footprint);

    h->ctx = (ഘടന htp_context *) block;
    ഘടന htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_queue   = dsp_queue;

    err = vtcm_alloc(ctx);
    എങ്കിൽ (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        തിരികെ AEE_ENOMEMORY;
    }

    HAP_setFARFRuntimeLoggingParams(0xffff, ശൂന്യ, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, വലുപ്പം(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        എങ്കിൽ ((err = HAP_power_set((ശൂന്യം *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            തിരികെ err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, വലുപ്പം(request));

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

#എങ്കിൽ (__HEXAGON_ARCH__ >= 79)
        HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
#endif
        എങ്കിൽ ((err = HAP_power_set((ശൂന്യം *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            തിരികെ err;
        }

        memset(&request, 0, വലുപ്പം(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        എങ്കിൽ ((err = HAP_power_set((ശൂന്യം *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            തിരികെ err;
        }
    }

#എങ്കിൽ __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, വലുപ്പം(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_power     = TRUE;
        request.hmx_v2.power_up      = TRUE;
        request.hmx_v2.set_clock     = TRUE;
        request.hmx_v2.target_corner = pwr->hmx_target_corner;
        request.hmx_v2.min_corner    = pwr->hmx_min_corner;
        request.hmx_v2.max_corner    = pwr->hmx_max_corner;
        request.hmx_v2.perf_mode     = pwr->hmx_perf_mode;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((ശൂന്യം *) ctx, &request);
        എങ്കിൽ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            തിരികെ err;
        }
    }
#അല്ലെങ്കിൽ
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, വലുപ്പം(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((ശൂന്യം *) ctx, &request);
        എങ്കിൽ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            തിരികെ err;
        }
    }
#endif

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_queue   = ശൂന്യ;
    എങ്കിൽ (n_hmx) {
        ശൂന്യം * hmx_ptr = (ശൂന്യം *) ((uintptr_t) block + offset_hmx);
        ctx->hmx_queue = hmx_queue_init(hmx_ptr, HMX_QUEUE_CAPACITY, HMX_QUEUE_അട്ടി_SIZE, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(മുകളിലെ_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_threads = n_hvx;
    ctx->n_threads_div = init_fastdiv_മൂല്യങ്ങൾ(ctx->n_threads);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) block + offset_dma);
    size_t size_dma_q = dma_queue_sizeof(256);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ctx->n_threads; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_up((uintptr_t) dma_ptr_curr, dma_queue_alignof());
        ctx->dma[i] = dma_queue_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += size_dma_q;
    }

    ctx->ddr_spad_size = 512 * 1024; // 512 KB
    ctx->ddr_spad_base = memalign(128, ctx->ddr_spad_size);

    ശൂന്യം * wq_ptr = (ശൂന്യം *) ((uintptr_t) block + offset_wq);
    ctx->work_queue = work_queue_init(wq_ptr, n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_അട്ടി_SIZE);

    ctx->main_അട്ടി = ശൂന്യ;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    എങ്കിൽ (!use_callbacks) {
        // Start main compute thread
        ctx->main_അട്ടി = (ശൂന്യം *) ((uintptr_t) block + offset_main_അട്ടി);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_അട്ടി_addr(&attr, ctx->main_അട്ടി);
        qurt_thread_attr_set_അട്ടി_size(&attr, size_main_അട്ടി);
        qurt_thread_attr_set_priority(&attr, main_prio);
        qurt_thread_attr_set_name(&attr, "htp-main");

        പൂർണ്ണസംഖ്യ err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        എങ്കിൽ (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            തിരികെ AEE_ENOMEMORY;
        }
    }

    FARF(മുകളിലെ_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_size, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    തിരികെ AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    ഘടന htp_handle * h = (ഘടന htp_handle *) handle;
    എങ്കിൽ (!h || !h->ctx) {
        തിരികെ AEE_EBADPARM;
    }
    ഘടന htp_context * ctx = h->ctx;

    എങ്കിൽ (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        പൂർണ്ണസംഖ്യ status;
        (ശൂന്യം) qurt_thread_join(ctx->main_thread, &status);
        ctx->main_thread = 0;
    }

    പൂർണ്ണസംഖ്യ err = dspqueue_close(ctx->dsp_queue); ctx->dsp_queue = ശൂന്യ;
    എങ്കിൽ (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (unsigned) err);
        തിരികെ err;
    }

    work_queue_free(ctx->work_queue);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ctx->n_threads; i++) {
        dma_queue_free(ctx->dma[i]);
    }

    എങ്കിൽ (ctx->hmx_queue) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = ശൂന്യ;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    എങ്കിൽ (ctx->ddr_spad_base) {
        സ്വതന്ത്രം(ctx->ddr_spad_base);
        ctx->ddr_spad_base = ശൂന്യ;
        ctx->ddr_spad_size = 0;
    }

    സ്വതന്ത്രം(ctx);
    h->ctx = ശൂന്യ;

    തിരികെ AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_threads, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_size) {
    (ശൂന്യം)handle;
    എങ്കിൽ (!n_threads || !n_hvx || !n_hmx || !vtcm_size) {
        തിരികെ AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    എങ്കിൽ (n_hvx_val > hw_threads.max_hthreads) {
        n_hvx_val = hw_threads.max_hthreads;
    }
    എങ്കിൽ (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_threads = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_size = vtcm_query_size();

    തിരികെ AEE_SUCCESS;
}

സ്ഥിതം ശൂന്യം htp_error_callback(dspqueue_t queue, പൂർണ്ണസംഖ്യ error, ശൂന്യം * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (unsigned) error);
}

ഘടന profile_data {
    uint64_t usecs;
    uint64_t cycles_start;
    uint64_t cycles_stop;
    uint32_t pmu_counters[HEX_അക്കം_PMU_COUNTERS];
};

സ്ഥിതം വരിയിൽ ശൂന്യം profile_start(uint32_t mode, ഘടന profile_data * d) {
    തിരഞ്ഞെടുപ്പ് (mode) {
        അവസ്ഥ HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        അവസ്ഥ HTP_PROF_BASIC:
        അവസ്ഥ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_എണ്ണം();
            d->cycles_start = hex_get_cycles();
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            നിർത്തുക;
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം profile_stop(uint32_t mode, ഘടന profile_data * d) {
    uint32_t pmu_counters[HEX_അക്കം_PMU_COUNTERS];
    തിരഞ്ഞെടുപ്പ് (mode) {
        അവസ്ഥ HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < HEX_അക്കം_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        അവസ്ഥ HTP_PROF_BASIC:
        അവസ്ഥ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_എണ്ണം_to_us(HAP_perf_get_qtimer_എണ്ണം() - d->usecs);
            d->cycles_stop = hex_get_cycles();
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            നിർത്തുക;
    }
}

സ്ഥിതം പൂർണ്ണസംഖ്യ op_fence(ഘടന htp_ops_context * octx) {
    ഘടന htp_context *ctx = octx->ctx;
    ഘടന htp_thread_trace * tr = &ctx->trace[0];
    സ്ഥിരം uint32_t seq  = (uint32_t) octx->op_params[0];
    സ്ഥിരം uint32_t mode = (uint32_t) octx->op_params[1];

    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    സ്ഥിരം ഘടന htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;

    എങ്കിൽ (mode == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_group_barrier(octx);

        എങ്കിൽ (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->status);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(മുകളിലെ_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        തിരികെ octx->status;
    }

    പൂർണ്ണസംഖ്യ status = HTP_STATUS_OK;
    uint64_t spins = 0;
    ആകുമ്പോൾ (1) {
        uint32_t sync_seq;
        uint32_t sync_status;
        htp_fence_read(sync_fence, &sync_seq, &sync_status);
        എങ്കിൽ ((int32_t)(sync_seq - seq) >= 0) {
            എങ്കിൽ (sync_status > HTP_STATUS_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_status, sync_fence, seq);
                status = sync_status;
            }
            നിർത്തുക;
        }
        എങ്കിൽ (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            status = HTP_STATUS_INTERNAL_ERR;
            നിർത്തുക;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(മുകളിലെ_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    തിരികെ status;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ op_mdev_group(ഘടന htp_ops_context * octx) {
    ഘടന htp_context * ctx = octx->ctx;
    സ്ഥിരം ഘടന htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.എണ്ണം = (uint16_t) sync->ne[1];
    എങ്കിൽ (ctx->mdev.എണ്ണം > 1) {
        ctx->mdev.എണ്ണം_div = init_fastdiv_മൂല്യങ്ങൾ(ctx->mdev.എണ്ണം);
        ctx->mdev.fence_base = (uint8_t *) sync->data;
    }
    തിരികെ HTP_STATUS_OK;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ execute_op(ഘടന htp_ops_context * octx) {
    തിരഞ്ഞെടുപ്പ് (octx->op) {
        അവസ്ഥ HTP_OP_MDEV_GROUP:
            തിരികെ op_mdev_group(octx);

        അവസ്ഥ HTP_OP_FENCE:
            തിരികെ op_fence(octx);

        അവസ്ഥ HTP_OP_ALLREDUCE:
        അവസ്ഥ HTP_OP_ALLREDUCE_ADD:
            തിരികെ op_allreduce(octx);

        അവസ്ഥ HTP_OP_MUL_MAT:
        അവസ്ഥ HTP_OP_MUL_MAT_ADD:
            തിരികെ op_matmul(octx);

        അവസ്ഥ HTP_OP_MUL_MAT_ID:
            തിരികെ op_matmul_id(octx);

        അവസ്ഥ HTP_OP_MUL_MAT_ID_NX:
            തിരികെ op_matmul_id_nx(octx);

        അവസ്ഥ HTP_OP_MUL_MAT_NX:
            തിരികെ op_matmul_nx(octx);

        അവസ്ഥ HTP_OP_MUL:
        അവസ്ഥ HTP_OP_ADD:
        അവസ്ഥ HTP_OP_SUB:
        അവസ്ഥ HTP_OP_DIV:
        അവസ്ഥ HTP_OP_ADD_ID:
            തിരികെ op_ദ്വിമാന(octx);

        അവസ്ഥ HTP_OP_NORM:
        അവസ്ഥ HTP_OP_RMS_NORM:
        അവസ്ഥ HTP_OP_RMS_NORM_MUL:
        അവസ്ഥ HTP_OP_SCALE:
        അവസ്ഥ HTP_OP_CLAMP:
        അവസ്ഥ HTP_OP_LEAKY_RELU:
        അവസ്ഥ HTP_OP_SQR:
        അവസ്ഥ HTP_OP_SQRT:
        അവസ്ഥ HTP_OP_UNARY_SOFTPLUS:
        അവസ്ഥ HTP_OP_UNARY_SIGMOID:
        അവസ്ഥ HTP_OP_UNARY_SILU:
        അവസ്ഥ HTP_OP_UNARY_GELU:
        അവസ്ഥ HTP_OP_UNARY_GELU_QUICK:
        അവസ്ഥ HTP_OP_UNARY_GELU_ERF:
        അവസ്ഥ HTP_OP_UNARY_NEG:
        അവസ്ഥ HTP_OP_UNARY_EXP:
        അവസ്ഥ HTP_OP_UNARY_TANH:
        അവസ്ഥ HTP_OP_UNARY_ABS:
        അവസ്ഥ HTP_OP_UNARY_LOG:
        അവസ്ഥ HTP_OP_UNARY_RELU:
        അവസ്ഥ HTP_OP_UNARY_STEP:
        അവസ്ഥ HTP_OP_L2_NORM:
        അവസ്ഥ HTP_OP_HADAMARD:
            തിരികെ op_unary(octx);

        അവസ്ഥ HTP_OP_GLU_SWIGLU:
        അവസ്ഥ HTP_OP_GLU_SWIGLU_OAI:
        അവസ്ഥ HTP_OP_GLU_SWIGLU_CLAMP:
        അവസ്ഥ HTP_OP_GLU_GEGLU:
        അവസ്ഥ HTP_OP_GLU_GEGLU_QUICK:
        അവസ്ഥ HTP_OP_GLU_GEGLU_ERF:
            തിരികെ op_activations(octx);

        അവസ്ഥ HTP_OP_SOFTMAX:
            തിരികെ op_softmax(octx);

        അവസ്ഥ HTP_OP_ROPE:
            തിരികെ op_rope(octx);

        അവസ്ഥ HTP_OP_FLASH_ATTN_EXT:
            തിരികെ op_flash_attn_ext(octx);

        അവസ്ഥ HTP_OP_SET_ROWS:
            തിരികെ op_set_rows(octx);

        അവസ്ഥ HTP_OP_GET_ROWS:
            തിരികെ op_get_rows(octx);

        അവസ്ഥ HTP_OP_SUM:
            തിരികെ op_sum(octx);

        അവസ്ഥ HTP_OP_SUM_ROWS:
            തിരികെ op_sum_rows(octx);

        അവസ്ഥ HTP_OP_CPY:
        അവസ്ഥ HTP_OP_CPY_FENCE:
            തിരികെ op_cpy(octx);

        അവസ്ഥ HTP_OP_REPEAT:
            തിരികെ op_repeat(octx);

        അവസ്ഥ HTP_OP_ARGSORT:
            തിരികെ op_argsort(octx);

        അവസ്ഥ HTP_OP_മുകൾ_K:
            തിരികെ op_മുകൾ_k(octx);

        അവസ്ഥ HTP_OP_ARGMAX:
            തിരികെ op_argmax(octx);

        അവസ്ഥ HTP_OP_SSM_CONV:
            തിരികെ op_ssm_conv(octx);

        അവസ്ഥ HTP_OP_CUMSUM:
            തിരികെ op_cumsum(octx);

        അവസ്ഥ HTP_OP_FILL:
            തിരികെ op_fill(octx);

        അവസ്ഥ HTP_OP_DIAG:
            തിരികെ op_diag(octx);

        അവസ്ഥ HTP_OP_SOLVE_TRI:
            തിരികെ op_solve_tri(octx);

        അവസ്ഥ HTP_OP_PAD:
            തിരികെ op_pad(octx);

        അവസ്ഥ HTP_OP_IM2COL:
            തിരികെ op_im2col(octx);

        അവസ്ഥ HTP_OP_IM2COL_3D:
            തിരികെ op_im2col_3d(octx);

        അവസ്ഥ HTP_OP_ROLL:
            തിരികെ op_roll(octx);

        അവസ്ഥ HTP_OP_POOL_2D:
            തിരികെ op_pool_2d(octx);

        അവസ്ഥ HTP_OP_POOL_1D:
            തിരികെ op_pool_1d(octx);

        അവസ്ഥ HTP_OP_CONCAT:
            തിരികെ op_concat(octx);

        അവസ്ഥ HTP_OP_GATED_DELTA_NET:
            തിരികെ op_gated_delta_net(octx);

        അവസ്ഥ HTP_OP_TRI:
            തിരികെ op_unary(octx);

        അവസ്ഥ HTP_OP_INVALID:
            നിർത്തുക;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    തിരികെ HTP_STATUS_NO_SUPPORT;
}

സ്ഥിതം വരിയിൽ bool reuse_buf(ഘടന htp_context *ctx, uint32_t *m_reuse, ഘടന htp_buf_desc *b) {
    b->base = ശൂന്യ;

    വേണ്ടി (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ഘടന htp_mmap *m = ctx->mmap + i;
        എങ്കിൽ (m->size && m->fd == b->fd && m->flags == b->flags) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            തിരികെ true;
        }
    }

    തിരികെ false;
}

സ്ഥിതം വരിയിൽ ശൂന്യം drop_mmap(ഘടന htp_context *ctx, ഘടന htp_mmap *m) {
    എങ്കിൽ (m->size) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (unsigned long long) m->base, (unsigned long long) m->size);
        htp_munmap(m->base, m->size, m->flags);
        m->size = 0;
        m->base = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

സ്ഥിതം വരിയിൽ bool mmap_buf(ഘടന htp_context *ctx, ഘടന htp_buf_desc *b) {
    എങ്കിൽ (b->base) തിരികെ true; // already mapped

    // find unused mapping
    വേണ്ടി (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        ഘടന htp_mmap *m = &ctx->mmap[i];
        എങ്കിൽ (!m->size) {
            uint64_t va = htp_mmap(b->fd, b->size, b->flags);
            എങ്കിൽ (va == 0) {
                FARF(മുകളിലെ_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (unsigned long long) b->size);
                തിരികെ false;
            }

            m->base   = b->base = va;
            m->fd     = b->fd;
            m->size   = b->size;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (unsigned long long) m->base, (unsigned long long) m->size, m->flags);
            തിരികെ true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    തിരികെ false;
}

സ്ഥിതം ശൂന്യം prep_op_bufs(ഘടന htp_context *ctx, ഘടന htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    വേണ്ടി (uint32_t i=0; i < n_bufs; i++) {
        ഘടന htp_buf_desc *b = bufs + i;
        എങ്കിൽ (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } അല്ലെങ്കിൽ എങ്കിൽ (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->size;
        }
        FARF(മുകളിലെ_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    എങ്കിൽ (b_reuse == n_bufs) തിരികെ; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    വേണ്ടി (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        എങ്കിൽ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].size;
        }
    }

    FARF(മുകളിലെ_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    എങ്കിൽ ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        വേണ്ടി (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            എങ്കിൽ (!used && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    വേണ്ടി (uint32_t i=0; i < n_bufs; i++) {
        ഘടന htp_buf_desc *b = bufs + i;
        എങ്കിൽ (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            നിർത്തുക;
        }
        FARF(മുകളിലെ_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    എങ്കിൽ (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(മുകളിലെ_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        വേണ്ടി (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            എങ്കിൽ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        വേണ്ടി (uint32_t i=0; i < n_bufs; i++) {
            ഘടന htp_buf_desc *b = bufs + i;
            എങ്കിൽ (!(b->flags & HTP_BUF_EXTENDED)) {
                b->base = 0;
            }
            എങ്കിൽ (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (unsigned long long) b->size);
                abort();
            }
            FARF(മുകളിലെ_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
        }
    }
}

സ്ഥിതം ശൂന്യം prep_tensor(ഘടന htp_context *ctx, ഘടന htp_buf_desc *bufs, ഘടന htp_tensor *tens, uint32_t idx, ഘടന htp_tensor *t) {
    uint64_t offset = t->data;
    uint32_t bi     = t->bi;

    t->data  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(മുകളിലെ_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (unsigned long long) offset, t->size, (unsigned long long) t->data,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

സ്ഥിതം ശൂന്യം prep_tensors(ഘടന htp_context *ctx, ഘടന htp_buf_desc *bufs, ഘടന htp_tensor *tens, uint32_t n_tens) {
    വേണ്ടി (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

സ്ഥിതം ശൂന്യം mdev_group_init(ഘടന htp_context * ctx, സ്ഥിരം ഘടന htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, വലുപ്പം(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ proc_op_req(ഘടന htp_ops_context * octx, ഘടന htp_buf_desc * bufs, uint32_t n_bufs,
                       ഘടന htp_tensor * tens, uint32_t idx, ഘടന htp_op_desc * op) {
    memcpy(octx->op_params,     op->params, വലുപ്പം(octx->op_params));
    memcpy(octx->kernel_params, op->kernel_params, വലുപ്പം(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_threads     = octx->ctx->n_threads;
    octx->n_threads_div = octx->ctx->n_threads_div;

    FARF(മുകളിലെ_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    വേണ്ടി (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        എങ്കിൽ (src_idx == 0xffff) {
            octx->src[i]     = ശൂന്യ;
            തുടരുക;
        }

        ഘടന htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(മുകളിലെ_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (unsigned long long) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    വേണ്ടി (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        എങ്കിൽ (dst_idx == 0xffff) {
            octx->dsts[i]    = ശൂന്യ;
            തുടരുക;
        }
        ഘടന htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(മുകളിലെ_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (unsigned long long) dst->data, dst->size,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_group_barrier(octx);

    പൂർണ്ണസംഖ്യ status = execute_op(octx);

    htp_ops_context_set_status(octx, status);

    octx->src0_spad.src = ശൂന്യ;
    octx->src1_spad.src = ശൂന്യ;
    octx->src2_spad.src = ശൂന്യ;
    octx->src3_spad.src = ശൂന്യ;
    octx->dst_spad.src  = ശൂന്യ;

    തിരികെ octx->status;
}

സ്ഥിതം ശൂന്യം process_opbatch(ഘടന htp_context * ctx, സ്ഥിരം ഘടന htp_opbatch_req * req, സ്ഥിരം ഘടന dspqueue_buffer * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    പൂർണ്ണസംഖ്യ err;

    സ്ഥിരം uint32_t n_bufs = req->n_bufs;
    സ്ഥിരം uint32_t n_tens = req->n_tensors;
    സ്ഥിരം uint32_t n_ops  = req->n_ops;

    സ്ഥിരം uint32_t b_size = വലുപ്പം(ഘടന htp_buf_desc)  * n_bufs;
    സ്ഥിരം uint32_t t_size = വലുപ്പം(ഘടന htp_tensor)    * n_tens;
    സ്ഥിരം uint32_t o_size = വലുപ്പം(ഘടന htp_op_desc)   * n_ops;
    സ്ഥിരം uint32_t p_size = വലുപ്പം(ഘടന htp_prof_desc) * n_ops;
    സ്ഥിരം uint32_t tr_size = (HTP_MAX_NTHREADS + 1) * req->n_traces * വലുപ്പം(ഘടന htp_trace_desc);

    എങ്കിൽ (dbuf->size < b_size + t_size + o_size + p_size + tr_size) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->size, b_size + t_size + o_size + p_size + tr_size);
        തിരികെ;
    }

    FARF(മുകളിലെ_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (unsigned long long) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->size, b_size, t_size, o_size);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    ഘടന htp_buf_desc* bufs = (ഘടന htp_buf_desc*)  m_ptr; m_ptr += b_size;
    ഘടന htp_tensor*   tens = (ഘടന htp_tensor*)    m_ptr; m_ptr += t_size;
    ഘടന htp_op_desc*   ops = (ഘടന htp_op_desc*)   m_ptr; m_ptr += o_size;
    ഘടന htp_prof_desc* pds = (ഘടന htp_prof_desc*) m_ptr;

    ഘടന profile_data batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, വലുപ്പം(ctx->trace));
    എങ്കിൽ (ctx->profiler == HTP_PROF_TRACE) {
        ഘടന htp_trace_desc * trace_events = (ഘടന htp_trace_desc *) (m_ptr + p_size);
        വേണ്ടി (പൂർണ്ണസംഖ്യ t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, വലുപ്പം(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    ഘടന htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, വലുപ്പം(*octx));
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
    octx->ctx           = ctx;

    mdev_group_init(ctx, req);

    work_queue_wakeup(ctx->work_queue);
    എങ്കിൽ (ctx->hmx_queue) {
        hmx_queue_wakeup(ctx->hmx_queue);
    }

    പൂർണ്ണസംഖ്യ op_status = HTP_STATUS_OK;
    octx->status  = HTP_STATUS_OK;
    വേണ്ടി (uint32_t i = 0; i < n_ops; i++) {
        ഘടന profile_data prof;

        profile_start(ctx->profiler, &prof);

        op_status = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_set_status(octx, op_status);

        എങ്കിൽ (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].cycles_start = prof.cycles_start;
            pds[i].cycles_stop = prof.cycles_stop;
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < HEX_അക്കം_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    എങ്കിൽ (ctx->hmx_queue) {
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

    ഘടന htp_opbatch_rsp rsp;
    memset(&rsp, 0, വലുപ്പം(rsp));
    rsp.seq          = req->seq;
    rsp.status       = octx->status;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.cycles_start = batch_prof.cycles_start;
    rsp.cycles_stop  = batch_prof.cycles_stop;

    എങ്കിൽ (ctx->profiler == HTP_PROF_TRACE) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].എണ്ണം;
        }
    }

    ഘടന dspqueue_buffer write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(queue, 0, 1, &write_dbuf, വലുപ്പം(rsp), (സ്ഥിരം uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    എങ്കിൽ (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
    }
}

#define DSPQUEUE_READ_TIMEOUT_USEC 5000
#define DSPQUEUE_POLL_TIMEOUT_USEC 100
#define DSPQUEUE_POLL_എണ്ണം        100

സ്ഥിതം ശൂന്യം process_ops(ഘടന htp_context * ctx) {
    dspqueue_t queue = ctx->dsp_queue;
    പൂർണ്ണസംഖ്യ err;

    uint32_t poll_എണ്ണം = DSPQUEUE_POLL_എണ്ണം;

    vtcm_acquire(ctx);

    ആകുമ്പോൾ (!atomic_load(&ctx->vtcm_needs_release) && !atomic_load(&ctx->killed)) {
        ഘടന htp_opbatch_req req;
        uint32_t r_size = വലുപ്പം(req);

        ഘടന dspqueue_buffer dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_size, &r_size, (uint8_t *) &req);
        എങ്കിൽ (err == AEE_EWOULDBLOCK) {
            എങ്കിൽ (--poll_എണ്ണം) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                തുടരുക;
            }
            നിർത്തുക;
        }

        എങ്കിൽ (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            നിർത്തുക;
        }

        എങ്കിൽ (r_size < വലുപ്പം(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_size, n_dbufs);
            തുടരുക;
        }

        // Reset poll count for valid requests
        poll_എണ്ണം = DSPQUEUE_POLL_എണ്ണം;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

സ്ഥിതം ശൂന്യം htp_packet_callback(dspqueue_t queue, പൂർണ്ണസംഖ്യ error, ശൂന്യം * context) {
    (ശൂന്യം) queue;
    (ശൂന്യം) error;
    ഘടന htp_handle * h = (ഘടന htp_handle *) context;
    എങ്കിൽ (h && h->ctx) {
        process_ops(h->ctx);
    }
}

സ്ഥിതം ശൂന്യം htp_main_thread(ശൂന്യം * context) {
    ഘടന htp_context * ctx = (ഘടന htp_context *) context;

    FARF(മുകളിലെ_2, "htp-main-thread: started");

    ആകുമ്പോൾ (!atomic_load(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t അക്കം_buffers = 0;
        uint32_t message_length = 0;

        പൂർണ്ണസംഖ്യ err = dspqueue_peek(ctx->dsp_queue, &flags, &അക്കം_buffers, &message_length, 50000);
        എങ്കിൽ (err == 0) {
            process_ops(ctx);
        } അല്ലെങ്കിൽ എങ്കിൽ (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            തുടരുക;
        } അല്ലെങ്കിൽ {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (unsigned) err);
            നിർത്തുക;
        }
    }

    FARF(മുകളിലെ_2, "htp-main-thread: stopped");
}
