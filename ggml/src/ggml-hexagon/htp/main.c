#ہدایت clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>
#شامل <AEEStdErr.h>
#شامل <dspqueue.h>
#شامل <HAP_compute_res.h>
#شامل <HAP_etm_config.h>
#شامل <HAP_mem.h>
#شامل <HAP_power.h>
#شامل <HAP_ps.h>
#شامل <HAP_dcvs.h>
#شامل <qurt.h>
#شامل <qurt_thread.h>
#شامل <qurt_memory.h>
#شامل <remote.h>
#شامل <string.h>
#شامل <stdatomic.h>

#شامل "hex-utils.h"
#شامل "dma-queue.h"
#شامل "hmx-queue.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "hex-bitmap.h"
#شامل "htp-ctx.h"
#شامل "htp-dcache.h"
#شامل "htp-ops.h"
#شامل "htp_iface.h"
#شامل "work-queue.h"
#شامل "hex-profile.h"
#شامل "allreduce-ops.h"
#شامل "htp-fence.h"

#تعریف_کرو HMX_قطار_استعداد     128
#تعریف_کرو HMX_قطار_ڈھیر_سائز   16384
#تعریف_کرو WORK_قطار_استعداد    16
#تعریف_کرو WORK_قطار_ڈھیر_سائز  16384
#تعریف_کرو MAIN_THREAD_ڈھیر_سائز 32768

_Static_assert(WORK_قطار_MAX_N_دھاگے >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

ساخت htp_handle {
    ساخت htp_context * ctx;
};

ساکن اندرونی uint64_t htp_mmap(uint32_t fd, uint64_t سائز, uint32_t flags) {
#اگر __HVX_ARCH__ > 79
    اگر (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, حجم(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = سائز;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        اگر (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (غیرعلامتی طویل طویل) سائز);
            واپس 0;
        }

        واپس payload.mmap.dsp_va;
    }
#ورنہ
    اگر (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        واپس 0;
    }
#اگر_ختم

    اگر (سائز > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (غیرعلامتی طویل طویل) سائز);
        واپس 0;
    }

    خالی * va = (خالی *)-1;
    برائے (صحیح retry = 0; retry < 2; retry++) {
#اگر __HVX_ARCH__ > 73
        va = HAP_mmap2(تہی, (size_t) سائز, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#ورنہ
        اگر (سائز > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (غیرعلامتی طویل طویل) سائز);
            abort();
        }
        va = HAP_mmap(تہی, (صحیح) سائز, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#اگر_ختم
        اگر (va != (خالی *)-1 && va != تہی) {
            واپس (uint64_t) (uintptr_t) va;
        }
        اگر (retry == 0) {
            FARF(بلند_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (غیرعلامتی طویل طویل) سائز);
        }
    }
    واپس 0;
}

ساکن اندرونی خالی htp_munmap(uint64_t va, uint64_t سائز, uint32_t flags) {
#اگر __HVX_ARCH__ > 79
    اگر (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, حجم(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = سائز;
        HAP_mem_request(&payload);
        واپس;
    }
#اگر_ختم

#اگر __HVX_ARCH__ > 73
    HAP_munmap2((خالی *) (uintptr_t) va, (size_t) سائز);
#ورنہ
    HAP_munmap((خالی *) (uintptr_t) va, (صحیح) سائز);
#اگر_ختم
}

AEEResult htp_iface_open(ثابت حرف * uri, remote_handle64 * handle) {
    (خالی) uri;
    ساخت htp_handle * h = صفر_حافظہ_دو(1, حجم(*h));
    اگر (h == تہی) {
        واپس AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    واپس AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    ساخت htp_handle * h = (ساخت htp_handle *) handle;
    اگر (!h) {
        واپس AEE_EBADPARM;
    }

    صحیح err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    اگر (err) {
        اگر (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } ورنہ {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    واپس err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t منوال, ثابت htp_iface_pmu_conf* pmu_conf) {
    ساخت htp_handle * h = (ساخت htp_handle *) handle;
    اگر (!h || !h->ctx) {
        واپس AEE_EBADPARM;
    }
    ساخت htp_context * ctx = h->ctx;

    اگر (منوال == HTP_PROF_PMU) {
        ثابت uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        برائے (; i < HEX_عدد_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        برائے (i = 0; i < HEX_عدد_PMU_COUNTERS; i++) {
            cfg |= (((events[i] >> 8) & 3) << (i * 2));
        }

        FARF(ALWAYS, "Configuring PMU registers: evtcfg = 0x%x, evtcfg1 = 0x%x, pmucfg = 0x%x", evtcfg, evtcfg1, cfg);

        // Configure PMU registers
        qurt_pmu_سیٹ(QURT_PMUCFG,     cfg);
        qurt_pmu_سیٹ(QURT_PMUEVTCFG,  evtcfg);
        qurt_pmu_سیٹ(QURT_PMUEVTCFG1, evtcfg1);
        qurt_pmu_enable(1);
    }

    ctx->profiler = منوال;

    واپس AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    ساخت htp_handle * h = (ساخت htp_handle *) handle;
    اگر (!h) {
        واپس AEE_EBADPARM;
    }

    ساخت htp_context * ctx = h->ctx;
    اگر (ctx) {
        اگر (ctx->dsp_قطار) {
            FARF(ERROR, "Closing handle with queue still open");
            واپس AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        برائے (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            اگر (ctx->mmap[i].سائز) {
                htp_munmap(ctx->mmap[i].بنیاد, ctx->mmap[i].سائز, ctx->mmap[i].flags);
                ctx->mmap[i].سائز = 0;
                ctx->mmap[i].بنیاد = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        اگر (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        اگر (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        آزاد(ctx);
        h->ctx = تہی;
    }

    آزاد(h);
    واپس AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t سائز) {
    ساخت htp_handle * h = (ساخت htp_handle *) handle;
    اگر (!h || !h->ctx) {
        واپس AEE_EBADPARM;
    }
    ساخت htp_context * ctx = h->ctx;

    // See if we already have this mapping
    برائے (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ساخت htp_mmap *m = &ctx->mmap[i];
        اگر (m->fd == fd) {
            واپس AEE_SUCCESS;
        }
    }

    // Add new mapping
    برائے (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ساخت htp_mmap *m = &ctx->mmap[i];
        اگر (!m->سائز) {
            FARF(بلند_2, "mmap : fd %u size %llu", fd, (غیرعلامتی طویل طویل) سائز);
            uint64_t va = htp_mmap(fd, سائز, 0);
            اگر (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (غیرعلامتی طویل طویل) سائز);
                واپس AEE_EFAILED;
            }

            m->بنیاد   = va;
            m->fd     = fd;
            m->سائز   = سائز;
            m->flags  = 0;

            واپس AEE_SUCCESS;
        }
    }

    واپس AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    ساخت htp_handle * h = (ساخت htp_handle *) handle;
    اگر (!h || !h->ctx) {
        واپس AEE_EBADPARM;
    }
    ساخت htp_context * ctx = h->ctx;

    برائے (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ساخت htp_mmap *m = &ctx->mmap[i];
        اگر (fd < 0 || m->fd == fd) {
            FARF(بلند_2, "unmmap : base 0x%llx fd %u size %llu", (غیرعلامتی طویل طویل) m->بنیاد, m->fd, (غیرعلامتی طویل طویل) m->سائز);
            htp_munmap(m->بنیاد, m->سائز, m->flags);
            m->سائز   = 0;
            m->بنیاد   = تہی;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    واپس AEE_SUCCESS;
}

ساکن خالی vtcm_acquire(ساخت htp_context * ctx) {
    اگر (!atomic_بھرو(&ctx->vtcm_valid)) {
        صحیح err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        اگر (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (غیرعلامتی)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_ترجیح(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

ساکن خالی vtcm_release(ساخت htp_context * ctx) {
    اگر (atomic_بھرو(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

ساکن صحیح vtcm_release_callback(غیرعلامتی صحیح rctx, خالی * state) {
    ساخت htp_context * ctx = (ساخت htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    واپس 0;
}

ساکن غیرعلامتی صحیح vtcm_query_سائز(خالی) {
    غیرعلامتی صحیح vtcm_سائز = 8 * 1024 * 1024;  // 8MB default
    غیرعلامتی صحیح vtcm_avail;
    compute_res_vtcm_صفحہ_t کل_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_سائز, &کل_layout, &vtcm_avail, &avail_layout);
    واپس vtcm_سائز;
}

ساکن صحیح vtcm_alloc(ساخت htp_context * ctx) {
    غیرعلامتی صحیح vtcm_سائز = vtcm_query_سائز();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_سیٹ_serialize(&attr, 0);
    HAP_compute_res_attr_سیٹ_cache_منوال(&attr, 1);
    HAP_compute_res_attr_سیٹ_vtcm_param_v2(&attr, vtcm_سائز, vtcm_سائز, vtcm_سائز); // single page
    HAP_compute_res_attr_سیٹ_release_callback(&attr, vtcm_release_callback, (خالی *) ctx);
    HAP_compute_res_attr_سیٹ_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    اگر (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_سائز);
        واپس AEE_ENOMEMORY;
    }

    خالی * vtcm_ptr;
    اگر (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_سائز) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_سائز);
        واپس AEE_ENOMEMORY;
    }

    ctx->vtcm_بنیاد          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_سائز          = vtcm_سائز;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    واپس 0;
}

ساکن خالی vtcm_free(ساخت htp_context * ctx) {
    اگر (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_بنیاد = 0;
        ctx->vtcm_rctx = 0;
    }
}

ساکن خالی htp_main_thread(خالی * context);
ساکن خالی htp_packet_callback(dspqueue_t قطار, صحیح error, خالی * context);
ساکن خالی htp_error_callback(dspqueue_t قطار, صحیح error, خالی * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
ساخت htp_power_profile {
    boolean  dcvs_enable;
    uint32_t bus_min_کونا;
    uint32_t bus_max_کونا;
    uint32_t bus_ہدف_کونا;
    uint32_t core_min_کونا;
    uint32_t core_max_کونا;
    uint32_t core_ہدف_کونا;
    boolean  sleep_disable;
    uint32_t hmx_min_کونا;    // only used when __HVX_ARCH__ >= 75
    uint32_t hmx_max_کونا;
    uint32_t hmx_ہدف_کونا;
    uint32_t hmx_perf_منوال;
};

#تعریف_کرو HTP_POWER_PROFILE_اندراج(min_c, max_c, hmx_c) \
    { \
        .dcvs_enable        = TRUE, \
        .bus_min_کونا     = (min_c), \
        .bus_max_کونا     = (max_c), \
        .bus_ہدف_کونا  = (max_c), \
        .core_min_کونا    = (min_c), \
        .core_max_کونا    = (max_c), \
        .core_ہدف_کونا = (max_c), \
        .sleep_disable      = TRUE, \
        .hmx_min_کونا     = (hmx_c), \
        .hmx_max_کونا     = (hmx_c), \
        .hmx_ہدف_کونا  = (hmx_c), \
        .hmx_perf_منوال      = HAP_CLK_PERF_بلند, \
    }

ساکن ثابت ساخت htp_power_profile htp_power_profiles[HTP_POWER_منوال_گنتی] = {
    [HTP_POWER_منوال_کم_POWER_SAVER] =
        HTP_POWER_PROFILE_اندراج(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_منوال_POWER_SAVER] =
        HTP_POWER_PROFILE_اندراج(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_منوال_بلند_POWER_SAVER] =
        HTP_POWER_PROFILE_اندراج(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_منوال_کم_متوازن] =
        HTP_POWER_PROFILE_اندراج(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_منوال_متوازن] =
        HTP_POWER_PROFILE_اندراج(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_منوال_بلند_PERFORMANCE] =
        HTP_POWER_PROFILE_اندراج(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_منوال_SUSTAINED_بلند_PERFORMANCE] =
        HTP_POWER_PROFILE_اندراج(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_منوال_مدت] = {
        .dcvs_enable        = FALSE,
        .bus_min_کونا     = HAP_DCVS_VCORNER_MAX,
        .bus_max_کونا     = HAP_DCVS_VCORNER_MAX,
        .bus_ہدف_کونا  = HAP_DCVS_VCORNER_MAX,
        .core_min_کونا    = HAP_DCVS_VCORNER_MAX,
        .core_max_کونا    = HAP_DCVS_VCORNER_MAX,
        .core_ہدف_کونا = HAP_DCVS_VCORNER_MAX,
        .sleep_disable      = TRUE,
        .hmx_min_کونا     = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_max_کونا     = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_ہدف_کونا  = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_perf_منوال      = HAP_CLK_PERF_بلند,
    },
};

#تعریف_منسوخ HTP_POWER_PROFILE_اندراج

AEEResult htp_iface_آغاز(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_قطار_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_منوال) {
    ساخت htp_handle * h = (ساخت htp_handle *) handle;
    اگر (!h) {
        واپس AEE_EBADPARM;
    }

    اگر (power_منوال >= HTP_POWER_منوال_گنتی) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (غیرعلامتی) power_منوال);
        power_منوال = HTP_POWER_منوال_مدت;
    }
    ثابت ساخت htp_power_profile * pwr = &htp_power_profiles[power_منوال];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (غیرعلامتی) power_منوال, (صحیح) pwr->dcvs_enable,
         (غیرعلامتی) pwr->bus_min_کونا, (غیرعلامتی) pwr->bus_max_کونا, (غیرعلامتی) pwr->bus_ہدف_کونا,
         (غیرعلامتی) pwr->core_min_کونا, (غیرعلامتی) pwr->core_max_کونا, (غیرعلامتی) pwr->core_ہدف_کونا,
         (صحیح) pwr->sleep_disable,
         (غیرعلامتی) pwr->hmx_min_کونا, (غیرعلامتی) pwr->hmx_max_کونا, (غیرعلامتی) pwr->hmx_ہدف_کونا);

    اگر (h->ctx) {
        FARF(ERROR, "Queue already open");
        واپس AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    صحیح fastrpc_tid  = qurt_thread_get_id();
    صحیح fastrpc_prio = qurt_thread_get_ترجیح(fastrpc_tid);
    صحیح main_prio    = fastrpc_prio - 10;
    اگر (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_قطار = تہی;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    صحیح err = dspqueue_import(dsp_قطار_id, تہی, تہی, (خالی *) h, &dsp_قطار);
    اگر (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(بلند_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_قطار_id, htp_packet_callback, htp_error_callback, (خالی *) h, &dsp_قطار);
    }

    اگر (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (غیرعلامتی) err);
        واپس err;
    }

    qurt_sysenv_max_hthreads_t hw_دھاگے;
    qurt_sysenv_get_max_hw_دھاگے(&hw_دھاگے);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    اگر (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    اگر (n_hvx > hw_دھاگے.max_hthreads) {
        n_hvx = hw_دھاگے.max_hthreads;
    }
    اگر (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t انحراف = حجم(ساخت htp_context);

    // 2. main_stack
    size_t انحراف_main_ڈھیر = 0;
    size_t سائز_main_ڈھیر   = 0;
    اگر (!use_callbacks) {
        انحراف_main_ڈھیر = hex_align_اوپر(انحراف, 4096);
        سائز_main_ڈھیر   = MAIN_THREAD_ڈھیر_سائز;
        انحراف = انحراف_main_ڈھیر + سائز_main_ڈھیر;
    }

    // 3. work_queue
    size_t wq_align  = work_قطار_alignof();
    size_t انحراف_wq = hex_align_اوپر(انحراف, wq_align);
    size_t سائز_wq   = work_قطار_sizeof(n_hvx, WORK_قطار_استعداد, WORK_قطار_ڈھیر_سائز);
    انحراف = انحراف_wq + سائز_wq;

    // 4. dma_queue
    size_t dma_align = dma_قطار_alignof();
    size_t انحراف_dma = hex_align_اوپر(انحراف, dma_align);
    size_t سائز_dma = 0;
    برائے (uint32_t i = 0; i < n_hvx; i++) {
        سائز_dma  = hex_align_اوپر(سائز_dma, dma_قطار_alignof());
        سائز_dma += dma_قطار_sizeof(256);
    }
    انحراف = انحراف_dma + سائز_dma;

    // 5. hmx_queue
    size_t انحراف_hmx = 0;
    size_t سائز_hmx = 0;
    اگر (n_hmx) {
        size_t hmx_align = hmx_قطار_alignof();
        انحراف_hmx = hex_align_اوپر(انحراف, hmx_align);
        سائز_hmx   = hmx_قطار_sizeof(HMX_قطار_استعداد, HMX_قطار_ڈھیر_سائز);
        انحراف = انحراف_hmx + سائز_hmx;
    }

    size_t footprint = hex_align_اوپر(انحراف, 128);

    خالی * بلاک = memalign(4096, footprint);
    اگر (!بلاک) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_قطار);
        واپس AEE_ENOMEMORY;
    }
    memset(بلاک, 0, footprint);

    h->ctx = (ساخت htp_context *) بلاک;
    ساخت htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_قطار   = dsp_قطار;

    err = vtcm_alloc(ctx);
    اگر (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        واپس AEE_ENOMEMORY;
    }

    HAP_سیٹ_FARFRuntime_Logging_Params(0xffff, تہی, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, حجم(HAP_power_request_t));
        request.type    = HAP_power_سیٹ_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        اگر ((err = HAP_power_سیٹ((خالی *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            واپس err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, حجم(request));

        request.type                              = HAP_power_سیٹ_DCVS_v3;
        request.dcvs_v3.سیٹ_dcvs_enable           = TRUE;
        request.dcvs_v3.dcvs_enable               = pwr->dcvs_enable;
        request.dcvs_v3.سیٹ_bus_params            = TRUE;
        request.dcvs_v3.bus_params.min_کونا     = pwr->bus_min_کونا;
        request.dcvs_v3.bus_params.max_کونا     = pwr->bus_max_کونا;
        request.dcvs_v3.bus_params.ہدف_کونا  = pwr->bus_ہدف_کونا;
        request.dcvs_v3.سیٹ_core_params           = TRUE;
        request.dcvs_v3.core_params.min_کونا    = pwr->core_min_کونا;
        request.dcvs_v3.core_params.max_کونا    = pwr->core_max_کونا;
        request.dcvs_v3.core_params.ہدف_کونا = pwr->core_ہدف_کونا;
        request.dcvs_v3.سیٹ_sleep_disable         = TRUE;
        request.dcvs_v3.sleep_disable             = pwr->sleep_disable;

#اگر (__HEXAGON_ARCH__ >= 79)
        HAP_سیٹ_dcvs_v3_protected_bus_corners(&request, 1);
#اگر_ختم
        اگر ((err = HAP_power_سیٹ((خالی *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            واپس err;
        }

        memset(&request, 0, حجم(request));
        request.type         = HAP_power_سیٹ_HVX;
        request.hvx.power_اوپر = TRUE;
        اگر ((err = HAP_power_سیٹ((خالی *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            واپس err;
        }
    }

#اگر __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, حجم(HAP_power_request_t));
        request.type = HAP_power_سیٹ_HMX_v2;
        request.hmx_v2.سیٹ_power     = TRUE;
        request.hmx_v2.power_اوپر      = TRUE;
        request.hmx_v2.سیٹ_clock     = TRUE;
        request.hmx_v2.ہدف_کونا = pwr->hmx_ہدف_کونا;
        request.hmx_v2.min_کونا    = pwr->hmx_min_کونا;
        request.hmx_v2.max_کونا    = pwr->hmx_max_کونا;
        request.hmx_v2.perf_منوال     = pwr->hmx_perf_منوال;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_سیٹ((خالی *) ctx, &request);
        اگر (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            واپس err;
        }
    }
#ورنہ
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, حجم(HAP_power_request_t));
        request.type         = HAP_power_سیٹ_HMX;
        request.hmx.power_اوپر = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_سیٹ((خالی *) ctx, &request);
        اگر (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            واپس err;
        }
    }
#اگر_ختم

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_قطار   = تہی;
    اگر (n_hmx) {
        خالی * hmx_ptr = (خالی *) ((uintptr_t) بلاک + انحراف_hmx);
        ctx->hmx_قطار = hmx_قطار_init(hmx_ptr, HMX_قطار_استعداد, HMX_قطار_ڈھیر_سائز, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(بلند_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_دھاگے = n_hvx;
    ctx->n_دھاگے_div = init_fastdiv_اقدار(ctx->n_دھاگے);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) بلاک + انحراف_dma);
    size_t سائز_dma_q = dma_قطار_sizeof(256);

    برائے (صحیح i = 0; i < ctx->n_دھاگے; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_اوپر((uintptr_t) dma_ptr_curr, dma_قطار_alignof());
        ctx->dma[i] = dma_قطار_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += سائز_dma_q;
    }

    ctx->ddr_spad_سائز = 512 * 1024; // 512 KB
    ctx->ddr_spad_بنیاد = memalign(128, ctx->ddr_spad_سائز);

    خالی * wq_ptr = (خالی *) ((uintptr_t) بلاک + انحراف_wq);
    ctx->work_قطار = work_قطار_init(wq_ptr, n_hvx, WORK_قطار_استعداد, WORK_قطار_ڈھیر_سائز);

    ctx->main_ڈھیر = تہی;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    اگر (!use_callbacks) {
        // Start main compute thread
        ctx->main_ڈھیر = (خالی *) ((uintptr_t) بلاک + انحراف_main_ڈھیر);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_سیٹ_ڈھیر_addr(&attr, ctx->main_ڈھیر);
        qurt_thread_attr_سیٹ_ڈھیر_سائز(&attr, سائز_main_ڈھیر);
        qurt_thread_attr_سیٹ_ترجیح(&attr, main_prio);
        qurt_thread_attr_سیٹ_name(&attr, "htp-main");

        صحیح err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        اگر (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            واپس AEE_ENOMEMORY;
        }
    }

    FARF(بلند_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_سائز, ctx->vtcm_rctx, ctx->n_دھاگے, ctx->thread_id, ctx->thread_prio);

    واپس AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    ساخت htp_handle * h = (ساخت htp_handle *) handle;
    اگر (!h || !h->ctx) {
        واپس AEE_EBADPARM;
    }
    ساخت htp_context * ctx = h->ctx;

    اگر (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        صحیح حالت_2;
        (خالی) qurt_thread_join(ctx->main_thread, &حالت_2);
        ctx->main_thread = 0;
    }

    صحیح err = dspqueue_close(ctx->dsp_قطار); ctx->dsp_قطار = تہی;
    اگر (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (غیرعلامتی) err);
        واپس err;
    }

    work_قطار_free(ctx->work_قطار);

    برائے (صحیح i = 0; i < ctx->n_دھاگے; i++) {
        dma_قطار_free(ctx->dma[i]);
    }

    اگر (ctx->hmx_قطار) {
        hmx_قطار_free(ctx->hmx_قطار);
        ctx->hmx_قطار = تہی;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    اگر (ctx->ddr_spad_بنیاد) {
        آزاد(ctx->ddr_spad_بنیاد);
        ctx->ddr_spad_بنیاد = تہی;
        ctx->ddr_spad_سائز = 0;
    }

    آزاد(ctx);
    h->ctx = تہی;

    واپس AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_دھاگے, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_سائز) {
    (خالی)handle;
    اگر (!n_دھاگے || !n_hvx || !n_hmx || !vtcm_سائز) {
        واپس AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_دھاگے;
    qurt_sysenv_get_max_hw_دھاگے(&hw_دھاگے);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    اگر (n_hvx_val > hw_دھاگے.max_hthreads) {
        n_hvx_val = hw_دھاگے.max_hthreads;
    }
    اگر (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_دھاگے = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_سائز = vtcm_query_سائز();

    واپس AEE_SUCCESS;
}

ساکن خالی htp_error_callback(dspqueue_t قطار, صحیح error, خالی * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (غیرعلامتی) error);
}

ساخت profile_ڈیٹا {
    uint64_t usecs;
    uint64_t ادوار_آغاز;
    uint64_t ادوار_stop;
    uint32_t pmu_counters[HEX_عدد_PMU_COUNTERS];
};

ساکن اندرونی خالی profile_آغاز(uint32_t منوال, ساخت profile_ڈیٹا * d) {
    انتخاب (منوال) {
        حالت HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        حالت HTP_PROF_BASIC:
        حالت HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_گنتی();
            d->ادوار_آغاز = hex_get_ادوار();
            توڑو;
        طےشدہ:
            توڑو;
    }
}

ساکن اندرونی خالی profile_stop(uint32_t منوال, ساخت profile_ڈیٹا * d) {
    uint32_t pmu_counters[HEX_عدد_PMU_COUNTERS];
    انتخاب (منوال) {
        حالت HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            برائے (صحیح i = 0; i < HEX_عدد_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        حالت HTP_PROF_BASIC:
        حالت HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_گنتی_مقصد_us(HAP_perf_get_qtimer_گنتی() - d->usecs);
            d->ادوار_stop = hex_get_ادوار();
            توڑو;
        طےشدہ:
            توڑو;
    }
}

ساکن صحیح op_fence(ساخت htp_ops_context * octx) {
    ساخت htp_context *ctx = octx->ctx;
    ساخت htp_thread_trace * tr = &ctx->trace[0];
    ثابت uint32_t seq  = (uint32_t) octx->op_params[0];
    ثابت uint32_t منوال = (uint32_t) octx->op_params[1];

    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ثابت ساخت htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->ڈیٹا;

    اگر (منوال == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_گروہ_barrier(octx);

        اگر (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->حالت_2);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(بلند_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->حالت_2);
        واپس octx->حالت_2;
    }

    صحیح حالت_2 = HTP_حالت_OK;
    uint64_t spins = 0;
    جب (1) {
        uint32_t sync_seq;
        uint32_t sync_حالت;
        htp_fence_read(sync_fence, &sync_seq, &sync_حالت);
        اگر ((int32_t)(sync_seq - seq) >= 0) {
            اگر (sync_حالت > HTP_حالت_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_حالت, sync_fence, seq);
                حالت_2 = sync_حالت;
            }
            توڑو;
        }
        اگر (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            حالت_2 = HTP_حالت_اندرونی_ERR;
            توڑو;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(بلند_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    واپس حالت_2;
}

ساکن صحیح op_mdev_گروہ(ساخت htp_ops_context * octx) {
    ساخت htp_context * ctx = octx->ctx;
    ثابت ساخت htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.گنتی = (uint16_t) sync->ne[1];
    اگر (ctx->mdev.گنتی > 1) {
        ctx->mdev.گنتی_div = init_fastdiv_اقدار(ctx->mdev.گنتی);
        ctx->mdev.fence_بنیاد = (uint8_t *) sync->ڈیٹا;
    }
    واپس HTP_حالت_OK;
}

ساکن صحیح execute_op(ساخت htp_ops_context * octx) {
    انتخاب (octx->op) {
        حالت HTP_OP_MDEV_گروہ:
            واپس op_mdev_گروہ(octx);

        حالت HTP_OP_FENCE:
            واپس op_fence(octx);

        حالت HTP_OP_ALLREDUCE:
        حالت HTP_OP_ALLREDUCE_جوڑو:
            واپس op_allreduce(octx);

        حالت HTP_OP_MUL_MAT:
        حالت HTP_OP_MUL_MAT_جوڑو:
            واپس op_matmul(octx);

        حالت HTP_OP_MUL_MAT_ID:
            واپس op_matmul_id(octx);

        حالت HTP_OP_MUL_MAT_ID_NX:
            واپس op_matmul_id_nx(octx);

        حالت HTP_OP_MUL_MAT_NX:
            واپس op_matmul_nx(octx);

        حالت HTP_OP_MUL:
        حالت HTP_OP_جوڑو:
        حالت HTP_OP_SUB:
        حالت HTP_OP_DIV:
        حالت HTP_OP_جوڑو_ID:
            واپس op_ثنائی(octx);

        حالت HTP_OP_NORM:
        حالت HTP_OP_RMS_NORM:
        حالت HTP_OP_RMS_NORM_MUL:
        حالت HTP_OP_SCALE:
        حالت HTP_OP_CLAMP:
        حالت HTP_OP_LEAKY_RELU:
        حالت HTP_OP_SQR:
        حالت HTP_OP_SQRT:
        حالت HTP_OP_UNARY_SOFTPLUS:
        حالت HTP_OP_UNARY_SIGMOID:
        حالت HTP_OP_UNARY_SILU:
        حالت HTP_OP_UNARY_GELU:
        حالت HTP_OP_UNARY_GELU_تیز:
        حالت HTP_OP_UNARY_GELU_ERF:
        حالت HTP_OP_UNARY_NEG:
        حالت HTP_OP_UNARY_EXP:
        حالت HTP_OP_UNARY_TANH:
        حالت HTP_OP_UNARY_ABS:
        حالت HTP_OP_UNARY_LOG:
        حالت HTP_OP_UNARY_RELU:
        حالت HTP_OP_UNARY_قدم:
        حالت HTP_OP_L2_NORM:
        حالت HTP_OP_HADAMARD:
            واپس op_unary(octx);

        حالت HTP_OP_GLU_SWIGLU:
        حالت HTP_OP_GLU_SWIGLU_OAI:
        حالت HTP_OP_GLU_SWIGLU_CLAMP:
        حالت HTP_OP_GLU_GEGLU:
        حالت HTP_OP_GLU_GEGLU_تیز:
        حالت HTP_OP_GLU_GEGLU_ERF:
            واپس op_activations(octx);

        حالت HTP_OP_SOFTMAX:
            واپس op_softmax(octx);

        حالت HTP_OP_ROPE:
            واپس op_rope(octx);

        حالت HTP_OP_FLASH_ATTN_EXT:
            واپس op_flash_attn_ext(octx);

        حالت HTP_OP_سیٹ_صفیں:
            واپس op_سیٹ_صفیں(octx);

        حالت HTP_OP_GET_صفیں:
            واپس op_get_صفیں(octx);

        حالت HTP_OP_مجموع:
            واپس op_مجموع(octx);

        حالت HTP_OP_مجموع_صفیں:
            واپس op_مجموع_صفیں(octx);

        حالت HTP_OP_CPY:
        حالت HTP_OP_CPY_FENCE:
            واپس op_cpy(octx);

        حالت HTP_OP_REPEAT:
            واپس op_repeat(octx);

        حالت HTP_OP_ARGSORT:
            واپس op_argsort(octx);

        حالت HTP_OP_چوٹی_K:
            واپس op_چوٹی_k(octx);

        حالت HTP_OP_ARGMAX:
            واپس op_argmax(octx);

        حالت HTP_OP_SSM_CONV:
            واپس op_ssm_conv(octx);

        حالت HTP_OP_CUMSUM:
            واپس op_cumsum(octx);

        حالت HTP_OP_FILL:
            واپس op_fill(octx);

        حالت HTP_OP_DIAG:
            واپس op_diag(octx);

        حالت HTP_OP_SOLVE_TRI:
            واپس op_solve_tri(octx);

        حالت HTP_OP_PAD:
            واپس op_pad(octx);

        حالت HTP_OP_IM2_ستون:
            واپس op_im2col(octx);

        حالت HTP_OP_IM2_ستون_3_D:
            واپس op_im2col_3d(octx);

        حالت HTP_OP_ROLL:
            واپس op_roll(octx);

        حالت HTP_OP_POOL_2D:
            واپس op_pool_2d(octx);

        حالت HTP_OP_POOL_1D:
            واپس op_pool_1d(octx);

        حالت HTP_OP_CONCAT:
            واپس op_concat(octx);

        حالت HTP_OP_GATED_DELTA_NET:
            واپس op_gated_delta_net(octx);

        حالت HTP_OP_TRI:
            واپس op_unary(octx);

        حالت HTP_OP_INVALID:
            توڑو;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    واپس HTP_حالت_NO_SUPPORT;
}

ساکن اندرونی bool reuse_buf(ساخت htp_context *ctx, uint32_t *m_reuse, ساخت htp_buf_desc *b) {
    b->بنیاد = تہی;

    برائے (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ساخت htp_mmap *m = ctx->mmap + i;
        اگر (m->سائز && m->fd == b->fd && m->flags == b->flags) {
            b->بنیاد   = m->بنیاد;
            *m_reuse |= (1 << i);
            واپس true;
        }
    }

    واپس false;
}

ساکن اندرونی خالی drop_mmap(ساخت htp_context *ctx, ساخت htp_mmap *m) {
    اگر (m->سائز) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (غیرعلامتی طویل طویل) m->بنیاد, (غیرعلامتی طویل طویل) m->سائز);
        htp_munmap(m->بنیاد, m->سائز, m->flags);
        m->سائز = 0;
        m->بنیاد = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

ساکن اندرونی bool mmap_buf(ساخت htp_context *ctx, ساخت htp_buf_desc *b) {
    اگر (b->بنیاد) واپس true; // already mapped

    // find unused mapping
    برائے (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        ساخت htp_mmap *m = &ctx->mmap[i];
        اگر (!m->سائز) {
            uint64_t va = htp_mmap(b->fd, b->سائز, b->flags);
            اگر (va == 0) {
                FARF(بلند_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (غیرعلامتی طویل طویل) b->سائز);
                واپس false;
            }

            m->بنیاد   = b->بنیاد = va;
            m->fd     = b->fd;
            m->سائز   = b->سائز;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (غیرعلامتی طویل طویل) m->بنیاد, (غیرعلامتی طویل طویل) m->سائز, m->flags);
            واپس true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    واپس false;
}

ساکن خالی prep_op_bufs(ساخت htp_context *ctx, ساخت htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    برائے (uint32_t i=0; i < n_bufs; i++) {
        ساخت htp_buf_desc *b = bufs + i;
        اگر (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } ورنہ اگر (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->سائز;
        }
        FARF(بلند_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (غیرعلامتی طویل طویل) b->بنیاد, (غیرعلامتی طویل طویل) b->سائز, b->flags);
    }

    اگر (b_reuse == n_bufs) واپس; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    برائے (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        اگر (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].سائز;
        }
    }

    FARF(بلند_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    اگر ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        برائے (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool مستعمل = m_reuse & (1<<i);
            اگر (!مستعمل && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    برائے (uint32_t i=0; i < n_bufs; i++) {
        ساخت htp_buf_desc *b = bufs + i;
        اگر (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            توڑو;
        }
        FARF(بلند_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (غیرعلامتی طویل طویل) b->بنیاد, (غیرعلامتی طویل طویل) b->سائز, b->flags);
    }

    اگر (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(بلند_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        برائے (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            اگر (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        برائے (uint32_t i=0; i < n_bufs; i++) {
            ساخت htp_buf_desc *b = bufs + i;
            اگر (!(b->flags & HTP_BUF_EXTENDED)) {
                b->بنیاد = 0;
            }
            اگر (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (غیرعلامتی طویل طویل) b->سائز);
                abort();
            }
            FARF(بلند_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (غیرعلامتی طویل طویل) b->بنیاد, (غیرعلامتی طویل طویل) b->سائز, b->flags);
        }
    }
}

ساکن خالی prep_tensor(ساخت htp_context *ctx, ساخت htp_buf_desc *bufs, ساخت htp_tensor *tens, uint32_t idx, ساخت htp_tensor *t) {
    uint64_t انحراف = t->ڈیٹا;
    uint32_t bi     = t->bi;

    t->ڈیٹا  = bufs[bi].بنیاد + انحراف;  // update data to the actual pointer

    FARF(بلند_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (غیرعلامتی طویل طویل) انحراف, t->سائز, (غیرعلامتی طویل طویل) t->ڈیٹا,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

ساکن خالی prep_tensors(ساخت htp_context *ctx, ساخت htp_buf_desc *bufs, ساخت htp_tensor *tens, uint32_t n_tens) {
    برائے (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

ساکن خالی mdev_گروہ_init(ساخت htp_context * ctx, ثابت ساخت htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, حجم(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

ساکن صحیح proc_op_req(ساخت htp_ops_context * octx, ساخت htp_buf_desc * bufs, uint32_t n_bufs,
                       ساخت htp_tensor * tens, uint32_t idx, ساخت htp_op_desc * op) {
    حافظہ_نقل(octx->op_params,     op->params, حجم(octx->op_params));
    حافظہ_نقل(octx->kernel_params, op->kernel_params, حجم(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_دھاگے     = octx->ctx->n_دھاگے;
    octx->n_دھاگے_div = octx->ctx->n_دھاگے_div;

    FARF(بلند_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    برائے (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        اگر (src_idx == 0xffff) {
            octx->src[i]     = تہی;
            جاری;
        }

        ساخت htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(بلند_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (غیرعلامتی طویل طویل) src->ڈیٹا, src->سائز,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    برائے (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        اگر (dst_idx == 0xffff) {
            octx->dsts[i]    = تہی;
            جاری;
        }
        ساخت htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(بلند_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (غیرعلامتی طویل طویل) dst->ڈیٹا, dst->سائز,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_گروہ_barrier(octx);

    صحیح حالت_2 = execute_op(octx);

    htp_ops_context_سیٹ_حالت(octx, حالت_2);

    octx->src0_spad.src = تہی;
    octx->src1_spad.src = تہی;
    octx->src2_spad.src = تہی;
    octx->src3_spad.src = تہی;
    octx->dst_spad.src  = تہی;

    واپس octx->حالت_2;
}

ساکن خالی process_opbatch(ساخت htp_context * ctx, ثابت ساخت htp_opbatch_req * req, ثابت ساخت dspqueue_بفر * dbuf) {
    dspqueue_t قطار = ctx->dsp_قطار;
    صحیح err;

    ثابت uint32_t n_bufs = req->n_bufs;
    ثابت uint32_t n_tens = req->n_tensors;
    ثابت uint32_t n_ops  = req->n_ops;

    ثابت uint32_t b_سائز = حجم(ساخت htp_buf_desc)  * n_bufs;
    ثابت uint32_t t_سائز = حجم(ساخت htp_tensor)    * n_tens;
    ثابت uint32_t o_سائز = حجم(ساخت htp_op_desc)   * n_ops;
    ثابت uint32_t p_سائز = حجم(ساخت htp_prof_desc) * n_ops;
    ثابت uint32_t tr_سائز = (HTP_MAX_NTHREADS + 1) * req->n_traces * حجم(ساخت htp_trace_desc);

    اگر (dbuf->سائز < b_سائز + t_سائز + o_سائز + p_سائز + tr_سائز) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->سائز, b_سائز + t_سائز + o_سائز + p_سائز + tr_سائز);
        واپس;
    }

    FARF(بلند_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (غیرعلامتی طویل طویل) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->سائز, b_سائز, t_سائز, o_سائز);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    ساخت htp_buf_desc* bufs = (ساخت htp_buf_desc*)  m_ptr; m_ptr += b_سائز;
    ساخت htp_tensor*   tens = (ساخت htp_tensor*)    m_ptr; m_ptr += t_سائز;
    ساخت htp_op_desc*   ops = (ساخت htp_op_desc*)   m_ptr; m_ptr += o_سائز;
    ساخت htp_prof_desc* pds = (ساخت htp_prof_desc*) m_ptr;

    ساخت profile_ڈیٹا batch_prof;
    profile_آغاز(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, حجم(ctx->trace));
    اگر (ctx->profiler == HTP_PROF_TRACE) {
        ساخت htp_trace_desc * trace_events = (ساخت htp_trace_desc *) (m_ptr + p_سائز);
        برائے (صحیح t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_آغاز(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_بلاک(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_آغاز(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    ساخت htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, حجم(*octx));
    octx->n_دھاگے     = ctx->n_دھاگے;
    octx->n_دھاگے_div = ctx->n_دھاگے_div;
    octx->ctx           = ctx;

    mdev_گروہ_init(ctx, req);

    work_قطار_wakeup(ctx->work_قطار);
    اگر (ctx->hmx_قطار) {
        hmx_قطار_wakeup(ctx->hmx_قطار);
    }

    صحیح op_حالت = HTP_حالت_OK;
    octx->حالت_2  = HTP_حالت_OK;
    برائے (uint32_t i = 0; i < n_ops; i++) {
        ساخت profile_ڈیٹا prof;

        profile_آغاز(ctx->profiler, &prof);

        op_حالت = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_سیٹ_حالت(octx, op_حالت);

        اگر (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].ادوار_آغاز = prof.ادوار_آغاز;
            pds[i].ادوار_stop = prof.ادوار_stop;
            برائے (صحیح j = 0; j < HEX_عدد_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    اگر (ctx->hmx_قطار) {
        hmx_قطار_suspend(ctx->hmx_قطار);
        hmx_قطار_flush(ctx->hmx_قطار);
    }
    work_قطار_suspend(ctx->work_قطار);

    // Flush remaining dirty tensors at the end of the batch
    htp_trace_event_آغاز(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_mdev_گروہ_barrier(octx);

    profile_stop(HTP_PROF_BASIC, &batch_prof);

    ساخت htp_opbatch_rsp rsp;
    memset(&rsp, 0, حجم(rsp));
    rsp.seq          = req->seq;
    rsp.حالت_2       = octx->حالت_2;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.ادوار_آغاز = batch_prof.ادوار_آغاز;
    rsp.ادوار_stop  = batch_prof.ادوار_stop;

    اگر (ctx->profiler == HTP_PROF_TRACE) {
        برائے (صحیح t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].گنتی;
        }
    }

    ساخت dspqueue_بفر write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_بفر_FLAG_FLUSH_SENDER | DSPQUEUE_بفر_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(قطار, 0, 1, &write_dbuf, حجم(rsp), (ثابت uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    اگر (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (غیرعلامتی) err);
    }
}

#تعریف_کرو DSPQUEUE_READ_TIMEOUT_USEC 5000
#تعریف_کرو DSPQUEUE_POLL_TIMEOUT_USEC 100
#تعریف_کرو DSPQUEUE_POLL_گنتی        100

ساکن خالی process_ops(ساخت htp_context * ctx) {
    dspqueue_t قطار = ctx->dsp_قطار;
    صحیح err;

    uint32_t poll_گنتی = DSPQUEUE_POLL_گنتی;

    vtcm_acquire(ctx);

    جب (!atomic_بھرو(&ctx->vtcm_needs_release) && !atomic_بھرو(&ctx->killed)) {
        ساخت htp_opbatch_req req;
        uint32_t r_سائز = حجم(req);

        ساخت dspqueue_بفر dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(قطار, &flags, n_dbufs, &n_dbufs, &dbuf, r_سائز, &r_سائز, (uint8_t *) &req);
        اگر (err == AEE_EWOULDBLOCK) {
            اگر (--poll_گنتی) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                جاری;
            }
            توڑو;
        }

        اگر (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (غیرعلامتی) err);
            توڑو;
        }

        اگر (r_سائز < حجم(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_سائز, n_dbufs);
            جاری;
        }

        // Reset poll count for valid requests
        poll_گنتی = DSPQUEUE_POLL_گنتی;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

ساکن خالی htp_packet_callback(dspqueue_t قطار, صحیح error, خالی * context) {
    (خالی) قطار;
    (خالی) error;
    ساخت htp_handle * h = (ساخت htp_handle *) context;
    اگر (h && h->ctx) {
        process_ops(h->ctx);
    }
}

ساکن خالی htp_main_thread(خالی * context) {
    ساخت htp_context * ctx = (ساخت htp_context *) context;

    FARF(بلند_2, "htp-main-thread: started");

    جب (!atomic_بھرو(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t عدد_buffers = 0;
        uint32_t پیغام_لمبائی = 0;

        صحیح err = dspqueue_peek(ctx->dsp_قطار, &flags, &عدد_buffers, &پیغام_لمبائی, 50000);
        اگر (err == 0) {
            process_ops(ctx);
        } ورنہ اگر (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            جاری;
        } ورنہ {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (غیرعلامتی) err);
            توڑو;
        }
    }

    FARF(بلند_2, "htp-main-thread: stopped");
}
