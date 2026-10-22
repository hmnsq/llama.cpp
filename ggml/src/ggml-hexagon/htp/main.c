#సూచన clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>
#అన్తర్భావ <AEEStdErr.h>
#అన్తర్భావ <dspqueue.h>
#అన్తర్భావ <HAP_compute_res.h>
#అన్తర్భావ <HAP_etm_config.h>
#అన్తర్భావ <HAP_mem.h>
#అన్తర్భావ <HAP_power.h>
#అన్తర్భావ <HAP_ps.h>
#అన్తర్భావ <HAP_dcvs.h>
#అన్తర్భావ <qurt.h>
#అన్తర్భావ <qurt_thread.h>
#అన్తర్భావ <qurt_memory.h>
#అన్తర్భావ <remote.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <stdatomic.h>

#అన్తర్భావ "hex-utils.h"
#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hmx-queue.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "hex-bitmap.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-dcache.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp_iface.h"
#అన్తర్భావ "work-queue.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "allreduce-ops.h"
#అన్తర్భావ "htp-fence.h"

#నిర్వచించు HMX_వరుస_సామర్థ్యం     128
#నిర్వచించు HMX_వరుస_దొంతర_కొలత   16384
#నిర్వచించు WORK_వరుస_సామర్థ్యం    16
#నిర్వచించు WORK_వరుస_దొంతర_కొలత  16384
#నిర్వచించు MAIN_THREAD_దొంతర_కొలత 32768

_Static_assert(WORK_వరుస_MAX_N_దారాలు >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

నిర్మాణం htp_handle {
    నిర్మాణం htp_context * ctx;
};

స్థిర అంతఃవరుస uint64_t htp_mmap(uint32_t fd, uint64_t కొలత, uint32_t flags) {
#ఒకవేళ __HVX_ARCH__ > 79
    ఒకవేళ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, పరిమాణం(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = కొలత;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        ఒకవేళ (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (గుర్తులేని దీర్ఘ దీర్ఘ) కొలత);
            తిరిగి 0;
        }

        తిరిగి payload.mmap.dsp_va;
    }
#లేకపోతే
    ఒకవేళ (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        తిరిగి 0;
    }
#అయితే_ముగింపు

    ఒకవేళ (కొలత > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (గుర్తులేని దీర్ఘ దీర్ఘ) కొలత);
        తిరిగి 0;
    }

    శూన్య * va = (శూన్య *)-1;
    కొరకు (పూర్ణాంకం retry = 0; retry < 2; retry++) {
#ఒకవేళ __HVX_ARCH__ > 73
        va = HAP_mmap2(ఖాళీ, (size_t) కొలత, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#లేకపోతే
        ఒకవేళ (కొలత > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (గుర్తులేని దీర్ఘ దీర్ఘ) కొలత);
            abort();
        }
        va = HAP_mmap(ఖాళీ, (పూర్ణాంకం) కొలత, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#అయితే_ముగింపు
        ఒకవేళ (va != (శూన్య *)-1 && va != ఖాళీ) {
            తిరిగి (uint64_t) (uintptr_t) va;
        }
        ఒకవేళ (retry == 0) {
            FARF(ఎగువ_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (గుర్తులేని దీర్ఘ దీర్ఘ) కొలత);
        }
    }
    తిరిగి 0;
}

స్థిర అంతఃవరుస శూన్య htp_munmap(uint64_t va, uint64_t కొలత, uint32_t flags) {
#ఒకవేళ __HVX_ARCH__ > 79
    ఒకవేళ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, పరిమాణం(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = కొలత;
        HAP_mem_request(&payload);
        తిరిగి;
    }
#అయితే_ముగింపు

#ఒకవేళ __HVX_ARCH__ > 73
    HAP_munmap2((శూన్య *) (uintptr_t) va, (size_t) కొలత);
#లేకపోతే
    HAP_munmap((శూన్య *) (uintptr_t) va, (పూర్ణాంకం) కొలత);
#అయితే_ముగింపు
}

AEEResult htp_iface_open(స్థిరరాశి అక్షరం * uri, remote_handle64 * handle) {
    (శూన్య) uri;
    నిర్మాణం htp_handle * h = సున్నా_జ్ఞాపకం_ఇవ్వు(1, పరిమాణం(*h));
    ఒకవేళ (h == ఖాళీ) {
        తిరిగి AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    తిరిగి AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    నిర్మాణం htp_handle * h = (నిర్మాణం htp_handle *) handle;
    ఒకవేళ (!h) {
        తిరిగి AEE_EBADPARM;
    }

    పూర్ణాంకం err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    ఒకవేళ (err) {
        ఒకవేళ (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } లేకపోతే {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    తిరిగి err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t బహుళకం, స్థిరరాశి htp_iface_pmu_conf* pmu_conf) {
    నిర్మాణం htp_handle * h = (నిర్మాణం htp_handle *) handle;
    ఒకవేళ (!h || !h->ctx) {
        తిరిగి AEE_EBADPARM;
    }
    నిర్మాణం htp_context * ctx = h->ctx;

    ఒకవేళ (బహుళకం == HTP_PROF_PMU) {
        స్థిరరాశి uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        కొరకు (; i < HEX_అంకె_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        కొరకు (i = 0; i < HEX_అంకె_PMU_COUNTERS; i++) {
            cfg |= (((events[i] >> 8) & 3) << (i * 2));
        }

        FARF(ALWAYS, "Configuring PMU registers: evtcfg = 0x%x, evtcfg1 = 0x%x, pmucfg = 0x%x", evtcfg, evtcfg1, cfg);

        // Configure PMU registers
        qurt_pmu_సమితి(QURT_PMUCFG,     cfg);
        qurt_pmu_సమితి(QURT_PMUEVTCFG,  evtcfg);
        qurt_pmu_సమితి(QURT_PMUEVTCFG1, evtcfg1);
        qurt_pmu_enable(1);
    }

    ctx->profiler = బహుళకం;

    తిరిగి AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    నిర్మాణం htp_handle * h = (నిర్మాణం htp_handle *) handle;
    ఒకవేళ (!h) {
        తిరిగి AEE_EBADPARM;
    }

    నిర్మాణం htp_context * ctx = h->ctx;
    ఒకవేళ (ctx) {
        ఒకవేళ (ctx->dsp_వరుస) {
            FARF(ERROR, "Closing handle with queue still open");
            తిరిగి AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        కొరకు (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            ఒకవేళ (ctx->mmap[i].కొలత) {
                htp_munmap(ctx->mmap[i].పునాది, ctx->mmap[i].కొలత, ctx->mmap[i].flags);
                ctx->mmap[i].కొలత = 0;
                ctx->mmap[i].పునాది = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        ఒకవేళ (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        ఒకవేళ (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        విడుదల(ctx);
        h->ctx = ఖాళీ;
    }

    విడుదల(h);
    తిరిగి AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t కొలత) {
    నిర్మాణం htp_handle * h = (నిర్మాణం htp_handle *) handle;
    ఒకవేళ (!h || !h->ctx) {
        తిరిగి AEE_EBADPARM;
    }
    నిర్మాణం htp_context * ctx = h->ctx;

    // See if we already have this mapping
    కొరకు (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        నిర్మాణం htp_mmap *m = &ctx->mmap[i];
        ఒకవేళ (m->fd == fd) {
            తిరిగి AEE_SUCCESS;
        }
    }

    // Add new mapping
    కొరకు (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        నిర్మాణం htp_mmap *m = &ctx->mmap[i];
        ఒకవేళ (!m->కొలత) {
            FARF(ఎగువ_2, "mmap : fd %u size %llu", fd, (గుర్తులేని దీర్ఘ దీర్ఘ) కొలత);
            uint64_t va = htp_mmap(fd, కొలత, 0);
            ఒకవేళ (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (గుర్తులేని దీర్ఘ దీర్ఘ) కొలత);
                తిరిగి AEE_EFAILED;
            }

            m->పునాది   = va;
            m->fd     = fd;
            m->కొలత   = కొలత;
            m->flags  = 0;

            తిరిగి AEE_SUCCESS;
        }
    }

    తిరిగి AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    నిర్మాణం htp_handle * h = (నిర్మాణం htp_handle *) handle;
    ఒకవేళ (!h || !h->ctx) {
        తిరిగి AEE_EBADPARM;
    }
    నిర్మాణం htp_context * ctx = h->ctx;

    కొరకు (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        నిర్మాణం htp_mmap *m = &ctx->mmap[i];
        ఒకవేళ (fd < 0 || m->fd == fd) {
            FARF(ఎగువ_2, "unmmap : base 0x%llx fd %u size %llu", (గుర్తులేని దీర్ఘ దీర్ఘ) m->పునాది, m->fd, (గుర్తులేని దీర్ఘ దీర్ఘ) m->కొలత);
            htp_munmap(m->పునాది, m->కొలత, m->flags);
            m->కొలత   = 0;
            m->పునాది   = ఖాళీ;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    తిరిగి AEE_SUCCESS;
}

స్థిర శూన్య vtcm_acquire(నిర్మాణం htp_context * ctx) {
    ఒకవేళ (!atomic_ఎక్కించు(&ctx->vtcm_valid)) {
        పూర్ణాంకం err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        ఒకవేళ (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (గుర్తులేని)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_ప్రాధాన్యత(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

స్థిర శూన్య vtcm_release(నిర్మాణం htp_context * ctx) {
    ఒకవేళ (atomic_ఎక్కించు(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

స్థిర పూర్ణాంకం vtcm_release_callback(గుర్తులేని పూర్ణాంకం rctx, శూన్య * state) {
    నిర్మాణం htp_context * ctx = (నిర్మాణం htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    తిరిగి 0;
}

స్థిర గుర్తులేని పూర్ణాంకం vtcm_query_కొలత(శూన్య) {
    గుర్తులేని పూర్ణాంకం vtcm_కొలత = 8 * 1024 * 1024;  // 8MB default
    గుర్తులేని పూర్ణాంకం vtcm_avail;
    compute_res_vtcm_పుట_t మొత్తం_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_కొలత, &మొత్తం_layout, &vtcm_avail, &avail_layout);
    తిరిగి vtcm_కొలత;
}

స్థిర పూర్ణాంకం vtcm_alloc(నిర్మాణం htp_context * ctx) {
    గుర్తులేని పూర్ణాంకం vtcm_కొలత = vtcm_query_కొలత();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_సమితి_serialize(&attr, 0);
    HAP_compute_res_attr_సమితి_cache_బహుళకం(&attr, 1);
    HAP_compute_res_attr_సమితి_vtcm_param_v2(&attr, vtcm_కొలత, vtcm_కొలత, vtcm_కొలత); // single page
    HAP_compute_res_attr_సమితి_release_callback(&attr, vtcm_release_callback, (శూన్య *) ctx);
    HAP_compute_res_attr_సమితి_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    ఒకవేళ (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_కొలత);
        తిరిగి AEE_ENOMEMORY;
    }

    శూన్య * vtcm_ptr;
    ఒకవేళ (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_కొలత) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_కొలత);
        తిరిగి AEE_ENOMEMORY;
    }

    ctx->vtcm_పునాది          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_కొలత          = vtcm_కొలత;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    తిరిగి 0;
}

స్థిర శూన్య vtcm_free(నిర్మాణం htp_context * ctx) {
    ఒకవేళ (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_పునాది = 0;
        ctx->vtcm_rctx = 0;
    }
}

స్థిర శూన్య htp_main_thread(శూన్య * context);
స్థిర శూన్య htp_packet_callback(dspqueue_t వరుస, పూర్ణాంకం error, శూన్య * context);
స్థిర శూన్య htp_error_callback(dspqueue_t వరుస, పూర్ణాంకం error, శూన్య * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
నిర్మాణం htp_power_profile {
    boolean  dcvs_enable;
    uint32_t bus_min_మూల;
    uint32_t bus_max_మూల;
    uint32_t bus_లక్ష్యం_మూల;
    uint32_t core_min_మూల;
    uint32_t core_max_మూల;
    uint32_t core_లక్ష్యం_మూల;
    boolean  sleep_disable;
    uint32_t hmx_min_మూల;    // only used when __HVX_ARCH__ >= 75
    uint32_t hmx_max_మూల;
    uint32_t hmx_లక్ష్యం_మూల;
    uint32_t hmx_perf_బహుళకం;
};

#నిర్వచించు HTP_POWER_PROFILE_నమోదు(min_c, max_c, hmx_c) \
    { \
        .dcvs_enable        = TRUE, \
        .bus_min_మూల     = (min_c), \
        .bus_max_మూల     = (max_c), \
        .bus_లక్ష్యం_మూల  = (max_c), \
        .core_min_మూల    = (min_c), \
        .core_max_మూల    = (max_c), \
        .core_లక్ష్యం_మూల = (max_c), \
        .sleep_disable      = TRUE, \
        .hmx_min_మూల     = (hmx_c), \
        .hmx_max_మూల     = (hmx_c), \
        .hmx_లక్ష్యం_మూల  = (hmx_c), \
        .hmx_perf_బహుళకం      = HAP_CLK_PERF_ఎగువ, \
    }

స్థిర స్థిరరాశి నిర్మాణం htp_power_profile htp_power_profiles[HTP_POWER_బహుళకం_లెక్క] = {
    [HTP_POWER_బహుళకం_దిగువ_POWER_SAVER] =
        HTP_POWER_PROFILE_నమోదు(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_బహుళకం_POWER_SAVER] =
        HTP_POWER_PROFILE_నమోదు(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_బహుళకం_ఎగువ_POWER_SAVER] =
        HTP_POWER_PROFILE_నమోదు(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_బహుళకం_దిగువ_సమతుల్య] =
        HTP_POWER_PROFILE_నమోదు(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_బహుళకం_సమతుల్య] =
        HTP_POWER_PROFILE_నమోదు(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_బహుళకం_ఎగువ_PERFORMANCE] =
        HTP_POWER_PROFILE_నమోదు(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_బహుళకం_SUSTAINED_ఎగువ_PERFORMANCE] =
        HTP_POWER_PROFILE_నమోదు(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_బహుళకం_వ్యవధి] = {
        .dcvs_enable        = FALSE,
        .bus_min_మూల     = HAP_DCVS_VCORNER_MAX,
        .bus_max_మూల     = HAP_DCVS_VCORNER_MAX,
        .bus_లక్ష్యం_మూల  = HAP_DCVS_VCORNER_MAX,
        .core_min_మూల    = HAP_DCVS_VCORNER_MAX,
        .core_max_మూల    = HAP_DCVS_VCORNER_MAX,
        .core_లక్ష్యం_మూల = HAP_DCVS_VCORNER_MAX,
        .sleep_disable      = TRUE,
        .hmx_min_మూల     = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_max_మూల     = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_లక్ష్యం_మూల  = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_perf_బహుళకం      = HAP_CLK_PERF_ఎగువ,
    },
};

#నిర్వచనం_రద్దు HTP_POWER_PROFILE_నమోదు

AEEResult htp_iface_ప్రారంభం(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_వరుస_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_బహుళకం) {
    నిర్మాణం htp_handle * h = (నిర్మాణం htp_handle *) handle;
    ఒకవేళ (!h) {
        తిరిగి AEE_EBADPARM;
    }

    ఒకవేళ (power_బహుళకం >= HTP_POWER_బహుళకం_లెక్క) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (గుర్తులేని) power_బహుళకం);
        power_బహుళకం = HTP_POWER_బహుళకం_వ్యవధి;
    }
    స్థిరరాశి నిర్మాణం htp_power_profile * pwr = &htp_power_profiles[power_బహుళకం];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (గుర్తులేని) power_బహుళకం, (పూర్ణాంకం) pwr->dcvs_enable,
         (గుర్తులేని) pwr->bus_min_మూల, (గుర్తులేని) pwr->bus_max_మూల, (గుర్తులేని) pwr->bus_లక్ష్యం_మూల,
         (గుర్తులేని) pwr->core_min_మూల, (గుర్తులేని) pwr->core_max_మూల, (గుర్తులేని) pwr->core_లక్ష్యం_మూల,
         (పూర్ణాంకం) pwr->sleep_disable,
         (గుర్తులేని) pwr->hmx_min_మూల, (గుర్తులేని) pwr->hmx_max_మూల, (గుర్తులేని) pwr->hmx_లక్ష్యం_మూల);

    ఒకవేళ (h->ctx) {
        FARF(ERROR, "Queue already open");
        తిరిగి AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    పూర్ణాంకం fastrpc_tid  = qurt_thread_get_id();
    పూర్ణాంకం fastrpc_prio = qurt_thread_get_ప్రాధాన్యత(fastrpc_tid);
    పూర్ణాంకం main_prio    = fastrpc_prio - 10;
    ఒకవేళ (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_వరుస = ఖాళీ;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    పూర్ణాంకం err = dspqueue_import(dsp_వరుస_id, ఖాళీ, ఖాళీ, (శూన్య *) h, &dsp_వరుస);
    ఒకవేళ (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(ఎగువ_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_వరుస_id, htp_packet_callback, htp_error_callback, (శూన్య *) h, &dsp_వరుస);
    }

    ఒకవేళ (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (గుర్తులేని) err);
        తిరిగి err;
    }

    qurt_sysenv_max_hthreads_t hw_దారాలు;
    qurt_sysenv_get_max_hw_దారాలు(&hw_దారాలు);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    ఒకవేళ (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    ఒకవేళ (n_hvx > hw_దారాలు.max_hthreads) {
        n_hvx = hw_దారాలు.max_hthreads;
    }
    ఒకవేళ (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t స్థానభ్రంశం = పరిమాణం(నిర్మాణం htp_context);

    // 2. main_stack
    size_t స్థానభ్రంశం_main_దొంతర = 0;
    size_t కొలత_main_దొంతర   = 0;
    ఒకవేళ (!use_callbacks) {
        స్థానభ్రంశం_main_దొంతర = hex_align_పైకి(స్థానభ్రంశం, 4096);
        కొలత_main_దొంతర   = MAIN_THREAD_దొంతర_కొలత;
        స్థానభ్రంశం = స్థానభ్రంశం_main_దొంతర + కొలత_main_దొంతర;
    }

    // 3. work_queue
    size_t wq_align  = work_వరుస_alignof();
    size_t స్థానభ్రంశం_wq = hex_align_పైకి(స్థానభ్రంశం, wq_align);
    size_t కొలత_wq   = work_వరుస_sizeof(n_hvx, WORK_వరుస_సామర్థ్యం, WORK_వరుస_దొంతర_కొలత);
    స్థానభ్రంశం = స్థానభ్రంశం_wq + కొలత_wq;

    // 4. dma_queue
    size_t dma_align = dma_వరుస_alignof();
    size_t స్థానభ్రంశం_dma = hex_align_పైకి(స్థానభ్రంశం, dma_align);
    size_t కొలత_dma = 0;
    కొరకు (uint32_t i = 0; i < n_hvx; i++) {
        కొలత_dma  = hex_align_పైకి(కొలత_dma, dma_వరుస_alignof());
        కొలత_dma += dma_వరుస_sizeof(256);
    }
    స్థానభ్రంశం = స్థానభ్రంశం_dma + కొలత_dma;

    // 5. hmx_queue
    size_t స్థానభ్రంశం_hmx = 0;
    size_t కొలత_hmx = 0;
    ఒకవేళ (n_hmx) {
        size_t hmx_align = hmx_వరుస_alignof();
        స్థానభ్రంశం_hmx = hex_align_పైకి(స్థానభ్రంశం, hmx_align);
        కొలత_hmx   = hmx_వరుస_sizeof(HMX_వరుస_సామర్థ్యం, HMX_వరుస_దొంతర_కొలత);
        స్థానభ్రంశం = స్థానభ్రంశం_hmx + కొలత_hmx;
    }

    size_t footprint = hex_align_పైకి(స్థానభ్రంశం, 128);

    శూన్య * ఖండం = memalign(4096, footprint);
    ఒకవేళ (!ఖండం) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_వరుస);
        తిరిగి AEE_ENOMEMORY;
    }
    memset(ఖండం, 0, footprint);

    h->ctx = (నిర్మాణం htp_context *) ఖండం;
    నిర్మాణం htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_వరుస   = dsp_వరుస;

    err = vtcm_alloc(ctx);
    ఒకవేళ (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        తిరిగి AEE_ENOMEMORY;
    }

    HAP_సమితి_FARFRuntime_Logging_Params(0xffff, ఖాళీ, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, పరిమాణం(HAP_power_request_t));
        request.type    = HAP_power_సమితి_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        ఒకవేళ ((err = HAP_power_సమితి((శూన్య *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            తిరిగి err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, పరిమాణం(request));

        request.type                              = HAP_power_సమితి_DCVS_v3;
        request.dcvs_v3.సమితి_dcvs_enable           = TRUE;
        request.dcvs_v3.dcvs_enable               = pwr->dcvs_enable;
        request.dcvs_v3.సమితి_bus_params            = TRUE;
        request.dcvs_v3.bus_params.min_మూల     = pwr->bus_min_మూల;
        request.dcvs_v3.bus_params.max_మూల     = pwr->bus_max_మూల;
        request.dcvs_v3.bus_params.లక్ష్యం_మూల  = pwr->bus_లక్ష్యం_మూల;
        request.dcvs_v3.సమితి_core_params           = TRUE;
        request.dcvs_v3.core_params.min_మూల    = pwr->core_min_మూల;
        request.dcvs_v3.core_params.max_మూల    = pwr->core_max_మూల;
        request.dcvs_v3.core_params.లక్ష్యం_మూల = pwr->core_లక్ష్యం_మూల;
        request.dcvs_v3.సమితి_sleep_disable         = TRUE;
        request.dcvs_v3.sleep_disable             = pwr->sleep_disable;

#ఒకవేళ (__HEXAGON_ARCH__ >= 79)
        HAP_సమితి_dcvs_v3_protected_bus_corners(&request, 1);
#అయితే_ముగింపు
        ఒకవేళ ((err = HAP_power_సమితి((శూన్య *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            తిరిగి err;
        }

        memset(&request, 0, పరిమాణం(request));
        request.type         = HAP_power_సమితి_HVX;
        request.hvx.power_పైకి = TRUE;
        ఒకవేళ ((err = HAP_power_సమితి((శూన్య *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            తిరిగి err;
        }
    }

#ఒకవేళ __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, పరిమాణం(HAP_power_request_t));
        request.type = HAP_power_సమితి_HMX_v2;
        request.hmx_v2.సమితి_power     = TRUE;
        request.hmx_v2.power_పైకి      = TRUE;
        request.hmx_v2.సమితి_clock     = TRUE;
        request.hmx_v2.లక్ష్యం_మూల = pwr->hmx_లక్ష్యం_మూల;
        request.hmx_v2.min_మూల    = pwr->hmx_min_మూల;
        request.hmx_v2.max_మూల    = pwr->hmx_max_మూల;
        request.hmx_v2.perf_బహుళకం     = pwr->hmx_perf_బహుళకం;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_సమితి((శూన్య *) ctx, &request);
        ఒకవేళ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            తిరిగి err;
        }
    }
#లేకపోతే
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, పరిమాణం(HAP_power_request_t));
        request.type         = HAP_power_సమితి_HMX;
        request.hmx.power_పైకి = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_సమితి((శూన్య *) ctx, &request);
        ఒకవేళ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            తిరిగి err;
        }
    }
#అయితే_ముగింపు

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_వరుస   = ఖాళీ;
    ఒకవేళ (n_hmx) {
        శూన్య * hmx_ptr = (శూన్య *) ((uintptr_t) ఖండం + స్థానభ్రంశం_hmx);
        ctx->hmx_వరుస = hmx_వరుస_init(hmx_ptr, HMX_వరుస_సామర్థ్యం, HMX_వరుస_దొంతర_కొలత, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(ఎగువ_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_దారాలు = n_hvx;
    ctx->n_దారాలు_div = init_fastdiv_విలువలు(ctx->n_దారాలు);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) ఖండం + స్థానభ్రంశం_dma);
    size_t కొలత_dma_q = dma_వరుస_sizeof(256);

    కొరకు (పూర్ణాంకం i = 0; i < ctx->n_దారాలు; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_పైకి((uintptr_t) dma_ptr_curr, dma_వరుస_alignof());
        ctx->dma[i] = dma_వరుస_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += కొలత_dma_q;
    }

    ctx->ddr_spad_కొలత = 512 * 1024; // 512 KB
    ctx->ddr_spad_పునాది = memalign(128, ctx->ddr_spad_కొలత);

    శూన్య * wq_ptr = (శూన్య *) ((uintptr_t) ఖండం + స్థానభ్రంశం_wq);
    ctx->work_వరుస = work_వరుస_init(wq_ptr, n_hvx, WORK_వరుస_సామర్థ్యం, WORK_వరుస_దొంతర_కొలత);

    ctx->main_దొంతర = ఖాళీ;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    ఒకవేళ (!use_callbacks) {
        // Start main compute thread
        ctx->main_దొంతర = (శూన్య *) ((uintptr_t) ఖండం + స్థానభ్రంశం_main_దొంతర);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_సమితి_దొంతర_addr(&attr, ctx->main_దొంతర);
        qurt_thread_attr_సమితి_దొంతర_కొలత(&attr, కొలత_main_దొంతర);
        qurt_thread_attr_సమితి_ప్రాధాన్యత(&attr, main_prio);
        qurt_thread_attr_సమితి_name(&attr, "htp-main");

        పూర్ణాంకం err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        ఒకవేళ (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            తిరిగి AEE_ENOMEMORY;
        }
    }

    FARF(ఎగువ_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_కొలత, ctx->vtcm_rctx, ctx->n_దారాలు, ctx->thread_id, ctx->thread_prio);

    తిరిగి AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    నిర్మాణం htp_handle * h = (నిర్మాణం htp_handle *) handle;
    ఒకవేళ (!h || !h->ctx) {
        తిరిగి AEE_EBADPARM;
    }
    నిర్మాణం htp_context * ctx = h->ctx;

    ఒకవేళ (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        పూర్ణాంకం స్థితి_2;
        (శూన్య) qurt_thread_join(ctx->main_thread, &స్థితి_2);
        ctx->main_thread = 0;
    }

    పూర్ణాంకం err = dspqueue_close(ctx->dsp_వరుస); ctx->dsp_వరుస = ఖాళీ;
    ఒకవేళ (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (గుర్తులేని) err);
        తిరిగి err;
    }

    work_వరుస_free(ctx->work_వరుస);

    కొరకు (పూర్ణాంకం i = 0; i < ctx->n_దారాలు; i++) {
        dma_వరుస_free(ctx->dma[i]);
    }

    ఒకవేళ (ctx->hmx_వరుస) {
        hmx_వరుస_free(ctx->hmx_వరుస);
        ctx->hmx_వరుస = ఖాళీ;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    ఒకవేళ (ctx->ddr_spad_పునాది) {
        విడుదల(ctx->ddr_spad_పునాది);
        ctx->ddr_spad_పునాది = ఖాళీ;
        ctx->ddr_spad_కొలత = 0;
    }

    విడుదల(ctx);
    h->ctx = ఖాళీ;

    తిరిగి AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_దారాలు, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_కొలత) {
    (శూన్య)handle;
    ఒకవేళ (!n_దారాలు || !n_hvx || !n_hmx || !vtcm_కొలత) {
        తిరిగి AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_దారాలు;
    qurt_sysenv_get_max_hw_దారాలు(&hw_దారాలు);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    ఒకవేళ (n_hvx_val > hw_దారాలు.max_hthreads) {
        n_hvx_val = hw_దారాలు.max_hthreads;
    }
    ఒకవేళ (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_దారాలు = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_కొలత = vtcm_query_కొలత();

    తిరిగి AEE_SUCCESS;
}

స్థిర శూన్య htp_error_callback(dspqueue_t వరుస, పూర్ణాంకం error, శూన్య * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (గుర్తులేని) error);
}

నిర్మాణం profile_దత్తాంశం {
    uint64_t usecs;
    uint64_t చక్రాలు_ప్రారంభం;
    uint64_t చక్రాలు_stop;
    uint32_t pmu_counters[HEX_అంకె_PMU_COUNTERS];
};

స్థిర అంతఃవరుస శూన్య profile_ప్రారంభం(uint32_t బహుళకం, నిర్మాణం profile_దత్తాంశం * d) {
    ఎంపిక (బహుళకం) {
        స్థితి HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        స్థితి HTP_PROF_BASIC:
        స్థితి HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_లెక్క();
            d->చక్రాలు_ప్రారంభం = hex_get_చక్రాలు();
            ఆపు;
        అప్రమేయ:
            ఆపు;
    }
}

స్థిర అంతఃవరుస శూన్య profile_stop(uint32_t బహుళకం, నిర్మాణం profile_దత్తాంశం * d) {
    uint32_t pmu_counters[HEX_అంకె_PMU_COUNTERS];
    ఎంపిక (బహుళకం) {
        స్థితి HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            కొరకు (పూర్ణాంకం i = 0; i < HEX_అంకె_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        స్థితి HTP_PROF_BASIC:
        స్థితి HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_లెక్క_గమ్యం_us(HAP_perf_get_qtimer_లెక్క() - d->usecs);
            d->చక్రాలు_stop = hex_get_చక్రాలు();
            ఆపు;
        అప్రమేయ:
            ఆపు;
    }
}

స్థిర పూర్ణాంకం op_fence(నిర్మాణం htp_ops_context * octx) {
    నిర్మాణం htp_context *ctx = octx->ctx;
    నిర్మాణం htp_thread_trace * tr = &ctx->trace[0];
    స్థిరరాశి uint32_t seq  = (uint32_t) octx->op_params[0];
    స్థిరరాశి uint32_t బహుళకం = (uint32_t) octx->op_params[1];

    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    స్థిరరాశి నిర్మాణం htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->దత్తాంశం;

    ఒకవేళ (బహుళకం == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_సమూహం_barrier(octx);

        ఒకవేళ (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->స్థితి_2);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(ఎగువ_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->స్థితి_2);
        తిరిగి octx->స్థితి_2;
    }

    పూర్ణాంకం స్థితి_2 = HTP_స్థితి_OK;
    uint64_t spins = 0;
    అయితే (1) {
        uint32_t sync_seq;
        uint32_t sync_స్థితి;
        htp_fence_read(sync_fence, &sync_seq, &sync_స్థితి);
        ఒకవేళ ((int32_t)(sync_seq - seq) >= 0) {
            ఒకవేళ (sync_స్థితి > HTP_స్థితి_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_స్థితి, sync_fence, seq);
                స్థితి_2 = sync_స్థితి;
            }
            ఆపు;
        }
        ఒకవేళ (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            స్థితి_2 = HTP_స్థితి_అంతర్గత_ERR;
            ఆపు;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(ఎగువ_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    తిరిగి స్థితి_2;
}

స్థిర పూర్ణాంకం op_mdev_సమూహం(నిర్మాణం htp_ops_context * octx) {
    నిర్మాణం htp_context * ctx = octx->ctx;
    స్థిరరాశి నిర్మాణం htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.లెక్క = (uint16_t) sync->ne[1];
    ఒకవేళ (ctx->mdev.లెక్క > 1) {
        ctx->mdev.లెక్క_div = init_fastdiv_విలువలు(ctx->mdev.లెక్క);
        ctx->mdev.fence_పునాది = (uint8_t *) sync->దత్తాంశం;
    }
    తిరిగి HTP_స్థితి_OK;
}

స్థిర పూర్ణాంకం execute_op(నిర్మాణం htp_ops_context * octx) {
    ఎంపిక (octx->op) {
        స్థితి HTP_OP_MDEV_సమూహం:
            తిరిగి op_mdev_సమూహం(octx);

        స్థితి HTP_OP_FENCE:
            తిరిగి op_fence(octx);

        స్థితి HTP_OP_ALLREDUCE:
        స్థితి HTP_OP_ALLREDUCE_కూడు:
            తిరిగి op_allreduce(octx);

        స్థితి HTP_OP_MUL_MAT:
        స్థితి HTP_OP_MUL_MAT_కూడు:
            తిరిగి op_matmul(octx);

        స్థితి HTP_OP_MUL_MAT_ID:
            తిరిగి op_matmul_id(octx);

        స్థితి HTP_OP_MUL_MAT_ID_NX:
            తిరిగి op_matmul_id_nx(octx);

        స్థితి HTP_OP_MUL_MAT_NX:
            తిరిగి op_matmul_nx(octx);

        స్థితి HTP_OP_MUL:
        స్థితి HTP_OP_కూడు:
        స్థితి HTP_OP_SUB:
        స్థితి HTP_OP_DIV:
        స్థితి HTP_OP_కూడు_ID:
            తిరిగి op_ద్విమాన(octx);

        స్థితి HTP_OP_NORM:
        స్థితి HTP_OP_RMS_NORM:
        స్థితి HTP_OP_RMS_NORM_MUL:
        స్థితి HTP_OP_SCALE:
        స్థితి HTP_OP_CLAMP:
        స్థితి HTP_OP_LEAKY_RELU:
        స్థితి HTP_OP_SQR:
        స్థితి HTP_OP_SQRT:
        స్థితి HTP_OP_UNARY_SOFTPLUS:
        స్థితి HTP_OP_UNARY_SIGMOID:
        స్థితి HTP_OP_UNARY_SILU:
        స్థితి HTP_OP_UNARY_GELU:
        స్థితి HTP_OP_UNARY_GELU_శీఘ్ర:
        స్థితి HTP_OP_UNARY_GELU_ERF:
        స్థితి HTP_OP_UNARY_NEG:
        స్థితి HTP_OP_UNARY_EXP:
        స్థితి HTP_OP_UNARY_TANH:
        స్థితి HTP_OP_UNARY_ABS:
        స్థితి HTP_OP_UNARY_LOG:
        స్థితి HTP_OP_UNARY_RELU:
        స్థితి HTP_OP_UNARY_అడుగు:
        స్థితి HTP_OP_L2_NORM:
        స్థితి HTP_OP_HADAMARD:
            తిరిగి op_unary(octx);

        స్థితి HTP_OP_GLU_SWIGLU:
        స్థితి HTP_OP_GLU_SWIGLU_OAI:
        స్థితి HTP_OP_GLU_SWIGLU_CLAMP:
        స్థితి HTP_OP_GLU_GEGLU:
        స్థితి HTP_OP_GLU_GEGLU_శీఘ్ర:
        స్థితి HTP_OP_GLU_GEGLU_ERF:
            తిరిగి op_activations(octx);

        స్థితి HTP_OP_SOFTMAX:
            తిరిగి op_softmax(octx);

        స్థితి HTP_OP_ROPE:
            తిరిగి op_rope(octx);

        స్థితి HTP_OP_FLASH_ATTN_EXT:
            తిరిగి op_flash_attn_ext(octx);

        స్థితి HTP_OP_సమితి_అడ్డువరుసలు:
            తిరిగి op_సమితి_అడ్డువరుసలు(octx);

        స్థితి HTP_OP_GET_అడ్డువరుసలు:
            తిరిగి op_get_అడ్డువరుసలు(octx);

        స్థితి HTP_OP_కూడిక:
            తిరిగి op_కూడిక(octx);

        స్థితి HTP_OP_కూడిక_అడ్డువరుసలు:
            తిరిగి op_కూడిక_అడ్డువరుసలు(octx);

        స్థితి HTP_OP_CPY:
        స్థితి HTP_OP_CPY_FENCE:
            తిరిగి op_cpy(octx);

        స్థితి HTP_OP_REPEAT:
            తిరిగి op_repeat(octx);

        స్థితి HTP_OP_ARGSORT:
            తిరిగి op_argsort(octx);

        స్థితి HTP_OP_శిఖరం_K:
            తిరిగి op_శిఖరం_k(octx);

        స్థితి HTP_OP_ARGMAX:
            తిరిగి op_argmax(octx);

        స్థితి HTP_OP_SSM_CONV:
            తిరిగి op_ssm_conv(octx);

        స్థితి HTP_OP_CUMSUM:
            తిరిగి op_cumsum(octx);

        స్థితి HTP_OP_FILL:
            తిరిగి op_fill(octx);

        స్థితి HTP_OP_DIAG:
            తిరిగి op_diag(octx);

        స్థితి HTP_OP_SOLVE_TRI:
            తిరిగి op_solve_tri(octx);

        స్థితి HTP_OP_PAD:
            తిరిగి op_pad(octx);

        స్థితి HTP_OP_IM2_నిలువు:
            తిరిగి op_im2col(octx);

        స్థితి HTP_OP_IM2_నిలువు_3_D:
            తిరిగి op_im2col_3d(octx);

        స్థితి HTP_OP_ROLL:
            తిరిగి op_roll(octx);

        స్థితి HTP_OP_POOL_2D:
            తిరిగి op_pool_2d(octx);

        స్థితి HTP_OP_POOL_1D:
            తిరిగి op_pool_1d(octx);

        స్థితి HTP_OP_CONCAT:
            తిరిగి op_concat(octx);

        స్థితి HTP_OP_GATED_DELTA_NET:
            తిరిగి op_gated_delta_net(octx);

        స్థితి HTP_OP_TRI:
            తిరిగి op_unary(octx);

        స్థితి HTP_OP_INVALID:
            ఆపు;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    తిరిగి HTP_స్థితి_NO_SUPPORT;
}

స్థిర అంతఃవరుస bool reuse_buf(నిర్మాణం htp_context *ctx, uint32_t *m_reuse, నిర్మాణం htp_buf_desc *b) {
    b->పునాది = ఖాళీ;

    కొరకు (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        నిర్మాణం htp_mmap *m = ctx->mmap + i;
        ఒకవేళ (m->కొలత && m->fd == b->fd && m->flags == b->flags) {
            b->పునాది   = m->పునాది;
            *m_reuse |= (1 << i);
            తిరిగి true;
        }
    }

    తిరిగి false;
}

స్థిర అంతఃవరుస శూన్య drop_mmap(నిర్మాణం htp_context *ctx, నిర్మాణం htp_mmap *m) {
    ఒకవేళ (m->కొలత) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (గుర్తులేని దీర్ఘ దీర్ఘ) m->పునాది, (గుర్తులేని దీర్ఘ దీర్ఘ) m->కొలత);
        htp_munmap(m->పునాది, m->కొలత, m->flags);
        m->కొలత = 0;
        m->పునాది = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

స్థిర అంతఃవరుస bool mmap_buf(నిర్మాణం htp_context *ctx, నిర్మాణం htp_buf_desc *b) {
    ఒకవేళ (b->పునాది) తిరిగి true; // already mapped

    // find unused mapping
    కొరకు (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        నిర్మాణం htp_mmap *m = &ctx->mmap[i];
        ఒకవేళ (!m->కొలత) {
            uint64_t va = htp_mmap(b->fd, b->కొలత, b->flags);
            ఒకవేళ (va == 0) {
                FARF(ఎగువ_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (గుర్తులేని దీర్ఘ దీర్ఘ) b->కొలత);
                తిరిగి false;
            }

            m->పునాది   = b->పునాది = va;
            m->fd     = b->fd;
            m->కొలత   = b->కొలత;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (గుర్తులేని దీర్ఘ దీర్ఘ) m->పునాది, (గుర్తులేని దీర్ఘ దీర్ఘ) m->కొలత, m->flags);
            తిరిగి true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    తిరిగి false;
}

స్థిర శూన్య prep_op_bufs(నిర్మాణం htp_context *ctx, నిర్మాణం htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    కొరకు (uint32_t i=0; i < n_bufs; i++) {
        నిర్మాణం htp_buf_desc *b = bufs + i;
        ఒకవేళ (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } లేకపోతే ఒకవేళ (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->కొలత;
        }
        FARF(ఎగువ_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (గుర్తులేని దీర్ఘ దీర్ఘ) b->పునాది, (గుర్తులేని దీర్ఘ దీర్ఘ) b->కొలత, b->flags);
    }

    ఒకవేళ (b_reuse == n_bufs) తిరిగి; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    కొరకు (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ఒకవేళ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].కొలత;
        }
    }

    FARF(ఎగువ_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    ఒకవేళ ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        కొరకు (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool వాడిన = m_reuse & (1<<i);
            ఒకవేళ (!వాడిన && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    కొరకు (uint32_t i=0; i < n_bufs; i++) {
        నిర్మాణం htp_buf_desc *b = bufs + i;
        ఒకవేళ (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            ఆపు;
        }
        FARF(ఎగువ_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (గుర్తులేని దీర్ఘ దీర్ఘ) b->పునాది, (గుర్తులేని దీర్ఘ దీర్ఘ) b->కొలత, b->flags);
    }

    ఒకవేళ (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(ఎగువ_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        కొరకు (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            ఒకవేళ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        కొరకు (uint32_t i=0; i < n_bufs; i++) {
            నిర్మాణం htp_buf_desc *b = bufs + i;
            ఒకవేళ (!(b->flags & HTP_BUF_EXTENDED)) {
                b->పునాది = 0;
            }
            ఒకవేళ (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (గుర్తులేని దీర్ఘ దీర్ఘ) b->కొలత);
                abort();
            }
            FARF(ఎగువ_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (గుర్తులేని దీర్ఘ దీర్ఘ) b->పునాది, (గుర్తులేని దీర్ఘ దీర్ఘ) b->కొలత, b->flags);
        }
    }
}

స్థిర శూన్య prep_tensor(నిర్మాణం htp_context *ctx, నిర్మాణం htp_buf_desc *bufs, నిర్మాణం htp_tensor *tens, uint32_t idx, నిర్మాణం htp_tensor *t) {
    uint64_t స్థానభ్రంశం = t->దత్తాంశం;
    uint32_t bi     = t->bi;

    t->దత్తాంశం  = bufs[bi].పునాది + స్థానభ్రంశం;  // update data to the actual pointer

    FARF(ఎగువ_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (గుర్తులేని దీర్ఘ దీర్ఘ) స్థానభ్రంశం, t->కొలత, (గుర్తులేని దీర్ఘ దీర్ఘ) t->దత్తాంశం,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

స్థిర శూన్య prep_tensors(నిర్మాణం htp_context *ctx, నిర్మాణం htp_buf_desc *bufs, నిర్మాణం htp_tensor *tens, uint32_t n_tens) {
    కొరకు (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

స్థిర శూన్య mdev_సమూహం_init(నిర్మాణం htp_context * ctx, స్థిరరాశి నిర్మాణం htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, పరిమాణం(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

స్థిర పూర్ణాంకం proc_op_req(నిర్మాణం htp_ops_context * octx, నిర్మాణం htp_buf_desc * bufs, uint32_t n_bufs,
                       నిర్మాణం htp_tensor * tens, uint32_t idx, నిర్మాణం htp_op_desc * op) {
    జ్ఞాపకం_నకలు(octx->op_params,     op->params, పరిమాణం(octx->op_params));
    జ్ఞాపకం_నకలు(octx->kernel_params, op->kernel_params, పరిమాణం(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_దారాలు     = octx->ctx->n_దారాలు;
    octx->n_దారాలు_div = octx->ctx->n_దారాలు_div;

    FARF(ఎగువ_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    కొరకు (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        ఒకవేళ (src_idx == 0xffff) {
            octx->src[i]     = ఖాళీ;
            కొనసాగు;
        }

        నిర్మాణం htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(ఎగువ_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (గుర్తులేని దీర్ఘ దీర్ఘ) src->దత్తాంశం, src->కొలత,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    కొరకు (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        ఒకవేళ (dst_idx == 0xffff) {
            octx->dsts[i]    = ఖాళీ;
            కొనసాగు;
        }
        నిర్మాణం htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(ఎగువ_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (గుర్తులేని దీర్ఘ దీర్ఘ) dst->దత్తాంశం, dst->కొలత,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_సమూహం_barrier(octx);

    పూర్ణాంకం స్థితి_2 = execute_op(octx);

    htp_ops_context_సమితి_స్థితి(octx, స్థితి_2);

    octx->src0_spad.src = ఖాళీ;
    octx->src1_spad.src = ఖాళీ;
    octx->src2_spad.src = ఖాళీ;
    octx->src3_spad.src = ఖాళీ;
    octx->dst_spad.src  = ఖాళీ;

    తిరిగి octx->స్థితి_2;
}

స్థిర శూన్య process_opbatch(నిర్మాణం htp_context * ctx, స్థిరరాశి నిర్మాణం htp_opbatch_req * req, స్థిరరాశి నిర్మాణం dspqueue_బఫరు * dbuf) {
    dspqueue_t వరుస = ctx->dsp_వరుస;
    పూర్ణాంకం err;

    స్థిరరాశి uint32_t n_bufs = req->n_bufs;
    స్థిరరాశి uint32_t n_tens = req->n_tensors;
    స్థిరరాశి uint32_t n_ops  = req->n_ops;

    స్థిరరాశి uint32_t b_కొలత = పరిమాణం(నిర్మాణం htp_buf_desc)  * n_bufs;
    స్థిరరాశి uint32_t t_కొలత = పరిమాణం(నిర్మాణం htp_tensor)    * n_tens;
    స్థిరరాశి uint32_t o_కొలత = పరిమాణం(నిర్మాణం htp_op_desc)   * n_ops;
    స్థిరరాశి uint32_t p_కొలత = పరిమాణం(నిర్మాణం htp_prof_desc) * n_ops;
    స్థిరరాశి uint32_t tr_కొలత = (HTP_MAX_NTHREADS + 1) * req->n_traces * పరిమాణం(నిర్మాణం htp_trace_desc);

    ఒకవేళ (dbuf->కొలత < b_కొలత + t_కొలత + o_కొలత + p_కొలత + tr_కొలత) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->కొలత, b_కొలత + t_కొలత + o_కొలత + p_కొలత + tr_కొలత);
        తిరిగి;
    }

    FARF(ఎగువ_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (గుర్తులేని దీర్ఘ దీర్ఘ) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->కొలత, b_కొలత, t_కొలత, o_కొలత);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    నిర్మాణం htp_buf_desc* bufs = (నిర్మాణం htp_buf_desc*)  m_ptr; m_ptr += b_కొలత;
    నిర్మాణం htp_tensor*   tens = (నిర్మాణం htp_tensor*)    m_ptr; m_ptr += t_కొలత;
    నిర్మాణం htp_op_desc*   ops = (నిర్మాణం htp_op_desc*)   m_ptr; m_ptr += o_కొలత;
    నిర్మాణం htp_prof_desc* pds = (నిర్మాణం htp_prof_desc*) m_ptr;

    నిర్మాణం profile_దత్తాంశం batch_prof;
    profile_ప్రారంభం(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, పరిమాణం(ctx->trace));
    ఒకవేళ (ctx->profiler == HTP_PROF_TRACE) {
        నిర్మాణం htp_trace_desc * trace_events = (నిర్మాణం htp_trace_desc *) (m_ptr + p_కొలత);
        కొరకు (పూర్ణాంకం t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_ప్రారంభం(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_ఖండం(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, పరిమాణం(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_ప్రారంభం(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    నిర్మాణం htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, పరిమాణం(*octx));
    octx->n_దారాలు     = ctx->n_దారాలు;
    octx->n_దారాలు_div = ctx->n_దారాలు_div;
    octx->ctx           = ctx;

    mdev_సమూహం_init(ctx, req);

    work_వరుస_wakeup(ctx->work_వరుస);
    ఒకవేళ (ctx->hmx_వరుస) {
        hmx_వరుస_wakeup(ctx->hmx_వరుస);
    }

    పూర్ణాంకం op_స్థితి = HTP_స్థితి_OK;
    octx->స్థితి_2  = HTP_స్థితి_OK;
    కొరకు (uint32_t i = 0; i < n_ops; i++) {
        నిర్మాణం profile_దత్తాంశం prof;

        profile_ప్రారంభం(ctx->profiler, &prof);

        op_స్థితి = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_సమితి_స్థితి(octx, op_స్థితి);

        ఒకవేళ (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].చక్రాలు_ప్రారంభం = prof.చక్రాలు_ప్రారంభం;
            pds[i].చక్రాలు_stop = prof.చక్రాలు_stop;
            కొరకు (పూర్ణాంకం j = 0; j < HEX_అంకె_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    ఒకవేళ (ctx->hmx_వరుస) {
        hmx_వరుస_suspend(ctx->hmx_వరుస);
        hmx_వరుస_flush(ctx->hmx_వరుస);
    }
    work_వరుస_suspend(ctx->work_వరుస);

    // Flush remaining dirty tensors at the end of the batch
    htp_trace_event_ప్రారంభం(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_mdev_సమూహం_barrier(octx);

    profile_stop(HTP_PROF_BASIC, &batch_prof);

    నిర్మాణం htp_opbatch_rsp rsp;
    memset(&rsp, 0, పరిమాణం(rsp));
    rsp.seq          = req->seq;
    rsp.స్థితి_2       = octx->స్థితి_2;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.చక్రాలు_ప్రారంభం = batch_prof.చక్రాలు_ప్రారంభం;
    rsp.చక్రాలు_stop  = batch_prof.చక్రాలు_stop;

    ఒకవేళ (ctx->profiler == HTP_PROF_TRACE) {
        కొరకు (పూర్ణాంకం t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].లెక్క;
        }
    }

    నిర్మాణం dspqueue_బఫరు write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_బఫరు_FLAG_FLUSH_SENDER | DSPQUEUE_బఫరు_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(వరుస, 0, 1, &write_dbuf, పరిమాణం(rsp), (స్థిరరాశి uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    ఒకవేళ (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (గుర్తులేని) err);
    }
}

#నిర్వచించు DSPQUEUE_READ_TIMEOUT_USEC 5000
#నిర్వచించు DSPQUEUE_POLL_TIMEOUT_USEC 100
#నిర్వచించు DSPQUEUE_POLL_లెక్క        100

స్థిర శూన్య process_ops(నిర్మాణం htp_context * ctx) {
    dspqueue_t వరుస = ctx->dsp_వరుస;
    పూర్ణాంకం err;

    uint32_t poll_లెక్క = DSPQUEUE_POLL_లెక్క;

    vtcm_acquire(ctx);

    అయితే (!atomic_ఎక్కించు(&ctx->vtcm_needs_release) && !atomic_ఎక్కించు(&ctx->killed)) {
        నిర్మాణం htp_opbatch_req req;
        uint32_t r_కొలత = పరిమాణం(req);

        నిర్మాణం dspqueue_బఫరు dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(వరుస, &flags, n_dbufs, &n_dbufs, &dbuf, r_కొలత, &r_కొలత, (uint8_t *) &req);
        ఒకవేళ (err == AEE_EWOULDBLOCK) {
            ఒకవేళ (--poll_లెక్క) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                కొనసాగు;
            }
            ఆపు;
        }

        ఒకవేళ (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (గుర్తులేని) err);
            ఆపు;
        }

        ఒకవేళ (r_కొలత < పరిమాణం(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_కొలత, n_dbufs);
            కొనసాగు;
        }

        // Reset poll count for valid requests
        poll_లెక్క = DSPQUEUE_POLL_లెక్క;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

స్థిర శూన్య htp_packet_callback(dspqueue_t వరుస, పూర్ణాంకం error, శూన్య * context) {
    (శూన్య) వరుస;
    (శూన్య) error;
    నిర్మాణం htp_handle * h = (నిర్మాణం htp_handle *) context;
    ఒకవేళ (h && h->ctx) {
        process_ops(h->ctx);
    }
}

స్థిర శూన్య htp_main_thread(శూన్య * context) {
    నిర్మాణం htp_context * ctx = (నిర్మాణం htp_context *) context;

    FARF(ఎగువ_2, "htp-main-thread: started");

    అయితే (!atomic_ఎక్కించు(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t అంకె_buffers = 0;
        uint32_t సందేశం_నిడివి = 0;

        పూర్ణాంకం err = dspqueue_peek(ctx->dsp_వరుస, &flags, &అంకె_buffers, &సందేశం_నిడివి, 50000);
        ఒకవేళ (err == 0) {
            process_ops(ctx);
        } లేకపోతే ఒకవేళ (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            కొనసాగు;
        } లేకపోతే {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (గుర్తులేని) err);
            ఆపు;
        }
    }

    FARF(ఎగువ_2, "htp-main-thread: stopped");
}
