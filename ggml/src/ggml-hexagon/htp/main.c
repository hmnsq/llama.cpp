#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>
#ଅନ୍ତର୍ଭାଵ <AEEStdErr.h>
#ଅନ୍ତର୍ଭାଵ <dspqueue.h>
#ଅନ୍ତର୍ଭାଵ <HAP_compute_res.h>
#ଅନ୍ତର୍ଭାଵ <HAP_etm_config.h>
#ଅନ୍ତର୍ଭାଵ <HAP_mem.h>
#ଅନ୍ତର୍ଭାଵ <HAP_power.h>
#ଅନ୍ତର୍ଭାଵ <HAP_ps.h>
#ଅନ୍ତର୍ଭାଵ <HAP_dcvs.h>
#ଅନ୍ତର୍ଭାଵ <qurt.h>
#ଅନ୍ତର୍ଭାଵ <qurt_thread.h>
#ଅନ୍ତର୍ଭାଵ <qurt_memory.h>
#ଅନ୍ତର୍ଭାଵ <remote.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <stdatomic.h>

#ଅନ୍ତର୍ଭାଵ "hex-utils.h"
#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "hmx-queue.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-bitmap.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-dcache.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp_iface.h"
#ଅନ୍ତର୍ଭାଵ "work-queue.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "allreduce-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-fence.h"

#define HMX_QUEUE_CAPACITY     128
#define HMX_QUEUE_ଗଦା_SIZE   16384
#define WORK_QUEUE_CAPACITY    16
#define WORK_QUEUE_ଗଦା_SIZE  16384
#define MAIN_THREAD_ଗଦା_SIZE 32768

_Static_assert(WORK_QUEUE_MAX_N_THREADS >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

ସଂରଚନା htp_handle {
    ସଂରଚନା htp_context * ctx;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint64_t htp_mmap(uint32_t fd, uint64_t size, uint32_t flags) {
#ଯଦି __HVX_ARCH__ > 79
    ଯଦି (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, ଆକାର(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = size;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        ଯଦି (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (unsigned long long) size);
            ଫେରନ୍ତୁ 0;
        }

        ଫେରନ୍ତୁ payload.mmap.dsp_va;
    }
#ଅନ୍ୟଥା
    ଯଦି (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        ଫେରନ୍ତୁ 0;
    }
#endif

    ଯଦି (size > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (unsigned long long) size);
        ଫେରନ୍ତୁ 0;
    }

    ଶୂନ୍ୟ * va = (ଶୂନ୍ୟ *)-1;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା retry = 0; retry < 2; retry++) {
#ଯଦି __HVX_ARCH__ > 73
        va = HAP_mmap2(ଖାଲି, (size_t) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#ଅନ୍ୟଥା
        ଯଦି (size > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (unsigned long long) size);
            abort();
        }
        va = HAP_mmap(ଖାଲି, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
        ଯଦି (va != (ଶୂନ୍ୟ *)-1 && va != ଖାଲି) {
            ଫେରନ୍ତୁ (uint64_t) (uintptr_t) va;
        }
        ଯଦି (retry == 0) {
            FARF(ଉପର_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (unsigned long long) size);
        }
    }
    ଫେରନ୍ତୁ 0;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_munmap(uint64_t va, uint64_t size, uint32_t flags) {
#ଯଦି __HVX_ARCH__ > 79
    ଯଦି (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, ଆକାର(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = size;
        HAP_mem_request(&payload);
        ଫେରନ୍ତୁ;
    }
#endif

#ଯଦି __HVX_ARCH__ > 73
    HAP_munmap2((ଶୂନ୍ୟ *) (uintptr_t) va, (size_t) size);
#ଅନ୍ୟଥା
    HAP_munmap((ଶୂନ୍ୟ *) (uintptr_t) va, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) size);
#endif
}

AEEResult htp_iface_open(ଅଚଳ ଅକ୍ଷର * uri, remote_handle64 * handle) {
    (ଶୂନ୍ୟ) uri;
    ସଂରଚନା htp_handle * h = calloc(1, ଆକାର(*h));
    ଯଦି (h == ଖାଲି) {
        ଫେରନ୍ତୁ AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    ଫେରନ୍ତୁ AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    ସଂରଚନା htp_handle * h = (ସଂରଚନା htp_handle *) handle;
    ଯଦି (!h) {
        ଫେରନ୍ତୁ AEE_EBADPARM;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    ଯଦି (err) {
        ଯଦି (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } ଅନ୍ୟଥା {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    ଫେରନ୍ତୁ err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, ଅଚଳ htp_iface_pmu_conf* pmu_conf) {
    ସଂରଚନା htp_handle * h = (ସଂରଚନା htp_handle *) handle;
    ଯଦି (!h || !h->ctx) {
        ଫେରନ୍ତୁ AEE_EBADPARM;
    }
    ସଂରଚନା htp_context * ctx = h->ctx;

    ଯଦି (mode == HTP_PROF_PMU) {
        ଅଚଳ uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        ପାଇଁ (; i < HEX_ଅଙ୍କ_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        ପାଇଁ (i = 0; i < HEX_ଅଙ୍କ_PMU_COUNTERS; i++) {
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

    ଫେରନ୍ତୁ AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    ସଂରଚନା htp_handle * h = (ସଂରଚନା htp_handle *) handle;
    ଯଦି (!h) {
        ଫେରନ୍ତୁ AEE_EBADPARM;
    }

    ସଂରଚନା htp_context * ctx = h->ctx;
    ଯଦି (ctx) {
        ଯଦି (ctx->dsp_queue) {
            FARF(ERROR, "Closing handle with queue still open");
            ଫେରନ୍ତୁ AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        ପାଇଁ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            ଯଦି (ctx->mmap[i].size) {
                htp_munmap(ctx->mmap[i].base, ctx->mmap[i].size, ctx->mmap[i].flags);
                ctx->mmap[i].size = 0;
                ctx->mmap[i].base = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        ଯଦି (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        ଯଦି (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        ମୁକ୍ତ(ctx);
        h->ctx = ଖାଲି;
    }

    ମୁକ୍ତ(h);
    ଫେରନ୍ତୁ AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t size) {
    ସଂରଚନା htp_handle * h = (ସଂରଚନା htp_handle *) handle;
    ଯଦି (!h || !h->ctx) {
        ଫେରନ୍ତୁ AEE_EBADPARM;
    }
    ସଂରଚନା htp_context * ctx = h->ctx;

    // See if we already have this mapping
    ପାଇଁ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ସଂରଚନା htp_mmap *m = &ctx->mmap[i];
        ଯଦି (m->fd == fd) {
            ଫେରନ୍ତୁ AEE_SUCCESS;
        }
    }

    // Add new mapping
    ପାଇଁ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ସଂରଚନା htp_mmap *m = &ctx->mmap[i];
        ଯଦି (!m->size) {
            FARF(ଉପର_2, "mmap : fd %u size %llu", fd, (unsigned long long) size);
            uint64_t va = htp_mmap(fd, size, 0);
            ଯଦି (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (unsigned long long) size);
                ଫେରନ୍ତୁ AEE_EFAILED;
            }

            m->base   = va;
            m->fd     = fd;
            m->size   = size;
            m->flags  = 0;

            ଫେରନ୍ତୁ AEE_SUCCESS;
        }
    }

    ଫେରନ୍ତୁ AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    ସଂରଚନା htp_handle * h = (ସଂରଚନା htp_handle *) handle;
    ଯଦି (!h || !h->ctx) {
        ଫେରନ୍ତୁ AEE_EBADPARM;
    }
    ସଂରଚନା htp_context * ctx = h->ctx;

    ପାଇଁ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ସଂରଚନା htp_mmap *m = &ctx->mmap[i];
        ଯଦି (fd < 0 || m->fd == fd) {
            FARF(ଉପର_2, "unmmap : base 0x%llx fd %u size %llu", (unsigned long long) m->base, m->fd, (unsigned long long) m->size);
            htp_munmap(m->base, m->size, m->flags);
            m->size   = 0;
            m->base   = ଖାଲି;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    ଫେରନ୍ତୁ AEE_SUCCESS;
}

ସ୍ଥିର ଶୂନ୍ୟ vtcm_acquire(ସଂରଚନା htp_context * ctx) {
    ଯଦି (!atomic_load(&ctx->vtcm_valid)) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        ଯଦି (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (unsigned)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ vtcm_release(ସଂରଚନା htp_context * ctx) {
    ଯଦି (atomic_load(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_release_callback(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା rctx, ଶୂନ୍ୟ * state) {
    ସଂରଚନା htp_context * ctx = (ସଂରଚନା htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    ଫେରନ୍ତୁ 0;
}

ସ୍ଥିର unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_query_size(ଶୂନ୍ୟ) {
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_size = 8 * 1024 * 1024;  // 8MB default
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_avail;
    compute_res_vtcm_page_t total_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_size, &total_layout, &vtcm_avail, &avail_layout);
    ଫେରନ୍ତୁ vtcm_size;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_alloc(ସଂରଚନା htp_context * ctx) {
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_size = vtcm_query_size();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (ଶୂନ୍ୟ *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    ଯଦି (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_size);
        ଫେରନ୍ତୁ AEE_ENOMEMORY;
    }

    ଶୂନ୍ୟ * vtcm_ptr;
    ଯଦି (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_size);
        ଫେରନ୍ତୁ AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size          = vtcm_size;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    ଫେରନ୍ତୁ 0;
}

ସ୍ଥିର ଶୂନ୍ୟ vtcm_free(ସଂରଚନା htp_context * ctx) {
    ଯଦି (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ htp_main_thread(ଶୂନ୍ୟ * context);
ସ୍ଥିର ଶୂନ୍ୟ htp_packet_callback(dspqueue_t queue, ପୂର୍ଣ୍ଣସଂଖ୍ୟା error, ଶୂନ୍ୟ * context);
ସ୍ଥିର ଶୂନ୍ୟ htp_error_callback(dspqueue_t queue, ପୂର୍ଣ୍ଣସଂଖ୍ୟା error, ଶୂନ୍ୟ * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
ସଂରଚନା htp_power_profile {
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
        .hmx_perf_mode      = HAP_CLK_PERF_ଉପର, \
    }

ସ୍ଥିର ଅଚଳ ସଂରଚନା htp_power_profile htp_power_profiles[HTP_POWER_MODE_ଗଣନା] = {
    [HTP_POWER_MODE_ତଳ_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_MODE_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_MODE_ଉପର_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_MODE_ତଳ_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_MODE_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_MODE_ଉପର_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_SUSTAINED_ଉପର_PERFORMANCE] =
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
        .hmx_perf_mode      = HAP_CLK_PERF_ଉପର,
    },
};

#undef HTP_POWER_PROFILE_ENTRY

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_mode) {
    ସଂରଚନା htp_handle * h = (ସଂରଚନା htp_handle *) handle;
    ଯଦି (!h) {
        ଫେରନ୍ତୁ AEE_EBADPARM;
    }

    ଯଦି (power_mode >= HTP_POWER_MODE_ଗଣନା) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (unsigned) power_mode);
        power_mode = HTP_POWER_MODE_BURST;
    }
    ଅଚଳ ସଂରଚନା htp_power_profile * pwr = &htp_power_profiles[power_mode];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (unsigned) power_mode, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) pwr->dcvs_enable,
         (unsigned) pwr->bus_min_corner, (unsigned) pwr->bus_max_corner, (unsigned) pwr->bus_target_corner,
         (unsigned) pwr->core_min_corner, (unsigned) pwr->core_max_corner, (unsigned) pwr->core_target_corner,
         (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) pwr->sleep_disable,
         (unsigned) pwr->hmx_min_corner, (unsigned) pwr->hmx_max_corner, (unsigned) pwr->hmx_target_corner);

    ଯଦି (h->ctx) {
        FARF(ERROR, "Queue already open");
        ଫେରନ୍ତୁ AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା fastrpc_tid  = qurt_thread_get_id();
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା fastrpc_prio = qurt_thread_get_priority(fastrpc_tid);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା main_prio    = fastrpc_prio - 10;
    ଯଦି (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_queue = ଖାଲି;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = dspqueue_import(dsp_queue_id, ଖାଲି, ଖାଲି, (ଶୂନ୍ୟ *) h, &dsp_queue);
    ଯଦି (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(ଉପର_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (ଶୂନ୍ୟ *) h, &dsp_queue);
    }

    ଯଦି (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        ଫେରନ୍ତୁ err;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    ଯଦି (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    ଯଦି (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    ଯଦି (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t offset = ଆକାର(ସଂରଚନା htp_context);

    // 2. main_stack
    size_t offset_main_ଗଦା = 0;
    size_t size_main_ଗଦା   = 0;
    ଯଦି (!use_callbacks) {
        offset_main_ଗଦା = hex_align_up(offset, 4096);
        size_main_ଗଦା   = MAIN_THREAD_ଗଦା_SIZE;
        offset = offset_main_ଗଦା + size_main_ଗଦା;
    }

    // 3. work_queue
    size_t wq_align  = work_queue_alignof();
    size_t offset_wq = hex_align_up(offset, wq_align);
    size_t size_wq   = work_queue_sizeof(n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_ଗଦା_SIZE);
    offset = offset_wq + size_wq;

    // 4. dma_queue
    size_t dma_align = dma_queue_alignof();
    size_t offset_dma = hex_align_up(offset, dma_align);
    size_t size_dma = 0;
    ପାଇଁ (uint32_t i = 0; i < n_hvx; i++) {
        size_dma  = hex_align_up(size_dma, dma_queue_alignof());
        size_dma += dma_queue_sizeof(256);
    }
    offset = offset_dma + size_dma;

    // 5. hmx_queue
    size_t offset_hmx = 0;
    size_t size_hmx = 0;
    ଯଦି (n_hmx) {
        size_t hmx_align = hmx_queue_alignof();
        offset_hmx = hex_align_up(offset, hmx_align);
        size_hmx   = hmx_queue_sizeof(HMX_QUEUE_CAPACITY, HMX_QUEUE_ଗଦା_SIZE);
        offset = offset_hmx + size_hmx;
    }

    size_t footprint = hex_align_up(offset, 128);

    ଶୂନ୍ୟ * block = memalign(4096, footprint);
    ଯଦି (!block) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_queue);
        ଫେରନ୍ତୁ AEE_ENOMEMORY;
    }
    memset(block, 0, footprint);

    h->ctx = (ସଂରଚନା htp_context *) block;
    ସଂରଚନା htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_queue   = dsp_queue;

    err = vtcm_alloc(ctx);
    ଯଦି (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        ଫେରନ୍ତୁ AEE_ENOMEMORY;
    }

    HAP_setFARFRuntimeLoggingParams(0xffff, ଖାଲି, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, ଆକାର(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        ଯଦି ((err = HAP_power_set((ଶୂନ୍ୟ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ଫେରନ୍ତୁ err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, ଆକାର(request));

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

#ଯଦି (__HEXAGON_ARCH__ >= 79)
        HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
#endif
        ଯଦି ((err = HAP_power_set((ଶୂନ୍ୟ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ଫେରନ୍ତୁ err;
        }

        memset(&request, 0, ଆକାର(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        ଯଦି ((err = HAP_power_set((ଶୂନ୍ୟ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ଫେରନ୍ତୁ err;
        }
    }

#ଯଦି __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, ଆକାର(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_power     = TRUE;
        request.hmx_v2.power_up      = TRUE;
        request.hmx_v2.set_clock     = TRUE;
        request.hmx_v2.target_corner = pwr->hmx_target_corner;
        request.hmx_v2.min_corner    = pwr->hmx_min_corner;
        request.hmx_v2.max_corner    = pwr->hmx_max_corner;
        request.hmx_v2.perf_mode     = pwr->hmx_perf_mode;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((ଶୂନ୍ୟ *) ctx, &request);
        ଯଦି (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            ଫେରନ୍ତୁ err;
        }
    }
#ଅନ୍ୟଥା
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, ଆକାର(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((ଶୂନ୍ୟ *) ctx, &request);
        ଯଦି (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            ଫେରନ୍ତୁ err;
        }
    }
#endif

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_queue   = ଖାଲି;
    ଯଦି (n_hmx) {
        ଶୂନ୍ୟ * hmx_ptr = (ଶୂନ୍ୟ *) ((uintptr_t) block + offset_hmx);
        ctx->hmx_queue = hmx_queue_init(hmx_ptr, HMX_QUEUE_CAPACITY, HMX_QUEUE_ଗଦା_SIZE, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(ଉପର_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_threads = n_hvx;
    ctx->n_threads_div = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ctx->n_threads);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) block + offset_dma);
    size_t size_dma_q = dma_queue_sizeof(256);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ctx->n_threads; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_up((uintptr_t) dma_ptr_curr, dma_queue_alignof());
        ctx->dma[i] = dma_queue_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += size_dma_q;
    }

    ctx->ddr_spad_size = 512 * 1024; // 512 KB
    ctx->ddr_spad_base = memalign(128, ctx->ddr_spad_size);

    ଶୂନ୍ୟ * wq_ptr = (ଶୂନ୍ୟ *) ((uintptr_t) block + offset_wq);
    ctx->work_queue = work_queue_init(wq_ptr, n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_ଗଦା_SIZE);

    ctx->main_ଗଦା = ଖାଲି;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    ଯଦି (!use_callbacks) {
        // Start main compute thread
        ctx->main_ଗଦା = (ଶୂନ୍ୟ *) ((uintptr_t) block + offset_main_ଗଦା);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_ଗଦା_addr(&attr, ctx->main_ଗଦା);
        qurt_thread_attr_set_ଗଦା_size(&attr, size_main_ଗଦା);
        qurt_thread_attr_set_priority(&attr, main_prio);
        qurt_thread_attr_set_name(&attr, "htp-main");

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        ଯଦି (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            ଫେରନ୍ତୁ AEE_ENOMEMORY;
        }
    }

    FARF(ଉପର_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_size, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    ଫେରନ୍ତୁ AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    ସଂରଚନା htp_handle * h = (ସଂରଚନା htp_handle *) handle;
    ଯଦି (!h || !h->ctx) {
        ଫେରନ୍ତୁ AEE_EBADPARM;
    }
    ସଂରଚନା htp_context * ctx = h->ctx;

    ଯଦି (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା status;
        (ଶୂନ୍ୟ) qurt_thread_join(ctx->main_thread, &status);
        ctx->main_thread = 0;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = dspqueue_close(ctx->dsp_queue); ctx->dsp_queue = ଖାଲି;
    ଯଦି (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (unsigned) err);
        ଫେରନ୍ତୁ err;
    }

    work_queue_free(ctx->work_queue);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ctx->n_threads; i++) {
        dma_queue_free(ctx->dma[i]);
    }

    ଯଦି (ctx->hmx_queue) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = ଖାଲି;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    ଯଦି (ctx->ddr_spad_base) {
        ମୁକ୍ତ(ctx->ddr_spad_base);
        ctx->ddr_spad_base = ଖାଲି;
        ctx->ddr_spad_size = 0;
    }

    ମୁକ୍ତ(ctx);
    h->ctx = ଖାଲି;

    ଫେରନ୍ତୁ AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_threads, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_size) {
    (ଶୂନ୍ୟ)handle;
    ଯଦି (!n_threads || !n_hvx || !n_hmx || !vtcm_size) {
        ଫେରନ୍ତୁ AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    ଯଦି (n_hvx_val > hw_threads.max_hthreads) {
        n_hvx_val = hw_threads.max_hthreads;
    }
    ଯଦି (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_threads = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_size = vtcm_query_size();

    ଫେରନ୍ତୁ AEE_SUCCESS;
}

ସ୍ଥିର ଶୂନ୍ୟ htp_error_callback(dspqueue_t queue, ପୂର୍ଣ୍ଣସଂଖ୍ୟା error, ଶୂନ୍ୟ * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (unsigned) error);
}

ସଂରଚନା profile_data {
    uint64_t usecs;
    uint64_t cycles_start;
    uint64_t cycles_stop;
    uint32_t pmu_counters[HEX_ଅଙ୍କ_PMU_COUNTERS];
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ profile_start(uint32_t mode, ସଂରଚନା profile_data * d) {
    ବାଛ (mode) {
        ସ୍ଥିତି HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        ସ୍ଥିତି HTP_PROF_BASIC:
        ସ୍ଥିତି HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_ଗଣନା();
            d->cycles_start = hex_get_cycles();
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଭାଙ୍ଗ;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ profile_stop(uint32_t mode, ସଂରଚନା profile_data * d) {
    uint32_t pmu_counters[HEX_ଅଙ୍କ_PMU_COUNTERS];
    ବାଛ (mode) {
        ସ୍ଥିତି HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < HEX_ଅଙ୍କ_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        ସ୍ଥିତି HTP_PROF_BASIC:
        ସ୍ଥିତି HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_ଗଣନା_to_us(HAP_perf_get_qtimer_ଗଣନା() - d->usecs);
            d->cycles_stop = hex_get_cycles();
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଭାଙ୍ଗ;
    }
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_fence(ସଂରଚନା htp_ops_context * octx) {
    ସଂରଚନା htp_context *ctx = octx->ctx;
    ସଂରଚନା htp_thread_trace * tr = &ctx->trace[0];
    ଅଚଳ uint32_t seq  = (uint32_t) octx->op_params[0];
    ଅଚଳ uint32_t mode = (uint32_t) octx->op_params[1];

    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ଅଚଳ ସଂରଚନା htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;

    ଯଦି (mode == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_group_barrier(octx);

        ଯଦି (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->status);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(ଉପର_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        ଫେରନ୍ତୁ octx->status;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା status = HTP_STATUS_OK;
    uint64_t spins = 0;
    ଯେତେବେଳେ (1) {
        uint32_t sync_seq;
        uint32_t sync_status;
        htp_fence_read(sync_fence, &sync_seq, &sync_status);
        ଯଦି ((int32_t)(sync_seq - seq) >= 0) {
            ଯଦି (sync_status > HTP_STATUS_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_status, sync_fence, seq);
                status = sync_status;
            }
            ଭାଙ୍ଗ;
        }
        ଯଦି (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            status = HTP_STATUS_INTERNAL_ERR;
            ଭାଙ୍ଗ;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(ଉପର_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    ଫେରନ୍ତୁ status;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_mdev_group(ସଂରଚନା htp_ops_context * octx) {
    ସଂରଚନା htp_context * ctx = octx->ctx;
    ଅଚଳ ସଂରଚନା htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.ଗଣନା_2 = (uint16_t) sync->ne[1];
    ଯଦି (ctx->mdev.ଗଣନା_2 > 1) {
        ctx->mdev.ଗଣନା_div = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ctx->mdev.ଗଣନା_2);
        ctx->mdev.fence_base = (uint8_t *) sync->data;
    }
    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା execute_op(ସଂରଚନା htp_ops_context * octx) {
    ବାଛ (octx->op) {
        ସ୍ଥିତି HTP_OP_MDEV_GROUP:
            ଫେରନ୍ତୁ op_mdev_group(octx);

        ସ୍ଥିତି HTP_OP_FENCE:
            ଫେରନ୍ତୁ op_fence(octx);

        ସ୍ଥିତି HTP_OP_ALLREDUCE:
        ସ୍ଥିତି HTP_OP_ALLREDUCE_ADD:
            ଫେରନ୍ତୁ op_allreduce(octx);

        ସ୍ଥିତି HTP_OP_MUL_MAT:
        ସ୍ଥିତି HTP_OP_MUL_MAT_ADD:
            ଫେରନ୍ତୁ op_matmul(octx);

        ସ୍ଥିତି HTP_OP_MUL_MAT_ID:
            ଫେରନ୍ତୁ op_matmul_id(octx);

        ସ୍ଥିତି HTP_OP_MUL_MAT_ID_NX:
            ଫେରନ୍ତୁ op_matmul_id_nx(octx);

        ସ୍ଥିତି HTP_OP_MUL_MAT_NX:
            ଫେରନ୍ତୁ op_matmul_nx(octx);

        ସ୍ଥିତି HTP_OP_MUL:
        ସ୍ଥିତି HTP_OP_ADD:
        ସ୍ଥିତି HTP_OP_SUB:
        ସ୍ଥିତି HTP_OP_DIV:
        ସ୍ଥିତି HTP_OP_ADD_ID:
            ଫେରନ୍ତୁ op_ଦ୍ୱିମିକ(octx);

        ସ୍ଥିତି HTP_OP_NORM:
        ସ୍ଥିତି HTP_OP_RMS_NORM:
        ସ୍ଥିତି HTP_OP_RMS_NORM_MUL:
        ସ୍ଥିତି HTP_OP_SCALE:
        ସ୍ଥିତି HTP_OP_CLAMP:
        ସ୍ଥିତି HTP_OP_LEAKY_RELU:
        ସ୍ଥିତି HTP_OP_SQR:
        ସ୍ଥିତି HTP_OP_SQRT:
        ସ୍ଥିତି HTP_OP_UNARY_SOFTPLUS:
        ସ୍ଥିତି HTP_OP_UNARY_SIGMOID:
        ସ୍ଥିତି HTP_OP_UNARY_SILU:
        ସ୍ଥିତି HTP_OP_UNARY_GELU:
        ସ୍ଥିତି HTP_OP_UNARY_GELU_QUICK:
        ସ୍ଥିତି HTP_OP_UNARY_GELU_ERF:
        ସ୍ଥିତି HTP_OP_UNARY_NEG:
        ସ୍ଥିତି HTP_OP_UNARY_EXP:
        ସ୍ଥିତି HTP_OP_UNARY_TANH:
        ସ୍ଥିତି HTP_OP_UNARY_ABS:
        ସ୍ଥିତି HTP_OP_UNARY_LOG:
        ସ୍ଥିତି HTP_OP_UNARY_RELU:
        ସ୍ଥିତି HTP_OP_UNARY_STEP:
        ସ୍ଥିତି HTP_OP_L2_NORM:
        ସ୍ଥିତି HTP_OP_HADAMARD:
            ଫେରନ୍ତୁ op_unary(octx);

        ସ୍ଥିତି HTP_OP_GLU_SWIGLU:
        ସ୍ଥିତି HTP_OP_GLU_SWIGLU_OAI:
        ସ୍ଥିତି HTP_OP_GLU_SWIGLU_CLAMP:
        ସ୍ଥିତି HTP_OP_GLU_GEGLU:
        ସ୍ଥିତି HTP_OP_GLU_GEGLU_QUICK:
        ସ୍ଥିତି HTP_OP_GLU_GEGLU_ERF:
            ଫେରନ୍ତୁ op_activations(octx);

        ସ୍ଥିତି HTP_OP_SOFTMAX:
            ଫେରନ୍ତୁ op_softmax(octx);

        ସ୍ଥିତି HTP_OP_ROPE:
            ଫେରନ୍ତୁ op_rope(octx);

        ସ୍ଥିତି HTP_OP_FLASH_ATTN_EXT:
            ଫେରନ୍ତୁ op_flash_attn_ext(octx);

        ସ୍ଥିତି HTP_OP_SET_ROWS:
            ଫେରନ୍ତୁ op_set_rows(octx);

        ସ୍ଥିତି HTP_OP_GET_ROWS:
            ଫେରନ୍ତୁ op_get_rows(octx);

        ସ୍ଥିତି HTP_OP_SUM:
            ଫେରନ୍ତୁ op_sum(octx);

        ସ୍ଥିତି HTP_OP_SUM_ROWS:
            ଫେରନ୍ତୁ op_sum_rows(octx);

        ସ୍ଥିତି HTP_OP_CPY:
        ସ୍ଥିତି HTP_OP_CPY_FENCE:
            ଫେରନ୍ତୁ op_cpy(octx);

        ସ୍ଥିତି HTP_OP_REPEAT:
            ଫେରନ୍ତୁ op_repeat(octx);

        ସ୍ଥିତି HTP_OP_ARGSORT:
            ଫେରନ୍ତୁ op_argsort(octx);

        ସ୍ଥିତି HTP_OP_ଶୀର୍ଷ_K:
            ଫେରନ୍ତୁ op_ଶୀର୍ଷ_k(octx);

        ସ୍ଥିତି HTP_OP_ARGMAX:
            ଫେରନ୍ତୁ op_argmax(octx);

        ସ୍ଥିତି HTP_OP_SSM_CONV:
            ଫେରନ୍ତୁ op_ssm_conv(octx);

        ସ୍ଥିତି HTP_OP_CUMSUM:
            ଫେରନ୍ତୁ op_cumsum(octx);

        ସ୍ଥିତି HTP_OP_FILL:
            ଫେରନ୍ତୁ op_fill(octx);

        ସ୍ଥିତି HTP_OP_DIAG:
            ଫେରନ୍ତୁ op_diag(octx);

        ସ୍ଥିତି HTP_OP_SOLVE_TRI:
            ଫେରନ୍ତୁ op_solve_tri(octx);

        ସ୍ଥିତି HTP_OP_PAD:
            ଫେରନ୍ତୁ op_pad(octx);

        ସ୍ଥିତି HTP_OP_IM2COL:
            ଫେରନ୍ତୁ op_im2col(octx);

        ସ୍ଥିତି HTP_OP_IM2COL_3D:
            ଫେରନ୍ତୁ op_im2col_3d(octx);

        ସ୍ଥିତି HTP_OP_ROLL:
            ଫେରନ୍ତୁ op_roll(octx);

        ସ୍ଥିତି HTP_OP_POOL_2D:
            ଫେରନ୍ତୁ op_pool_2d(octx);

        ସ୍ଥିତି HTP_OP_POOL_1D:
            ଫେରନ୍ତୁ op_pool_1d(octx);

        ସ୍ଥିତି HTP_OP_CONCAT:
            ଫେରନ୍ତୁ op_concat(octx);

        ସ୍ଥିତି HTP_OP_GATED_DELTA_NET:
            ଫେରନ୍ତୁ op_gated_delta_net(octx);

        ସ୍ଥିତି HTP_OP_TRI:
            ଫେରନ୍ତୁ op_unary(octx);

        ସ୍ଥିତି HTP_OP_INVALID:
            ଭାଙ୍ଗ;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool reuse_buf(ସଂରଚନା htp_context *ctx, uint32_t *m_reuse, ସଂରଚନା htp_buf_desc *b) {
    b->base = ଖାଲି;

    ପାଇଁ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ସଂରଚନା htp_mmap *m = ctx->mmap + i;
        ଯଦି (m->size && m->fd == b->fd && m->flags == b->flags) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            ଫେରନ୍ତୁ true;
        }
    }

    ଫେରନ୍ତୁ false;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ drop_mmap(ସଂରଚନା htp_context *ctx, ସଂରଚନା htp_mmap *m) {
    ଯଦି (m->size) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (unsigned long long) m->base, (unsigned long long) m->size);
        htp_munmap(m->base, m->size, m->flags);
        m->size = 0;
        m->base = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool mmap_buf(ସଂରଚନା htp_context *ctx, ସଂରଚନା htp_buf_desc *b) {
    ଯଦି (b->base) ଫେରନ୍ତୁ true; // already mapped

    // find unused mapping
    ପାଇଁ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        ସଂରଚନା htp_mmap *m = &ctx->mmap[i];
        ଯଦି (!m->size) {
            uint64_t va = htp_mmap(b->fd, b->size, b->flags);
            ଯଦି (va == 0) {
                FARF(ଉପର_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (unsigned long long) b->size);
                ଫେରନ୍ତୁ false;
            }

            m->base   = b->base = va;
            m->fd     = b->fd;
            m->size   = b->size;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (unsigned long long) m->base, (unsigned long long) m->size, m->flags);
            ଫେରନ୍ତୁ true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    ଫେରନ୍ତୁ false;
}

ସ୍ଥିର ଶୂନ୍ୟ prep_op_bufs(ସଂରଚନା htp_context *ctx, ସଂରଚନା htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    ପାଇଁ (uint32_t i=0; i < n_bufs; i++) {
        ସଂରଚନା htp_buf_desc *b = bufs + i;
        ଯଦି (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } ଅନ୍ୟଥା ଯଦି (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->size;
        }
        FARF(ଉପର_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    ଯଦି (b_reuse == n_bufs) ଫେରନ୍ତୁ; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    ପାଇଁ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ଯଦି (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].size;
        }
    }

    FARF(ଉପର_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    ଯଦି ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        ପାଇଁ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            ଯଦି (!used && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    ପାଇଁ (uint32_t i=0; i < n_bufs; i++) {
        ସଂରଚନା htp_buf_desc *b = bufs + i;
        ଯଦି (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            ଭାଙ୍ଗ;
        }
        FARF(ଉପର_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    ଯଦି (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(ଉପର_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        ପାଇଁ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            ଯଦି (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        ପାଇଁ (uint32_t i=0; i < n_bufs; i++) {
            ସଂରଚନା htp_buf_desc *b = bufs + i;
            ଯଦି (!(b->flags & HTP_BUF_EXTENDED)) {
                b->base = 0;
            }
            ଯଦି (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (unsigned long long) b->size);
                abort();
            }
            FARF(ଉପର_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ prep_tensor(ସଂରଚନା htp_context *ctx, ସଂରଚନା htp_buf_desc *bufs, ସଂରଚନା htp_tensor *tens, uint32_t idx, ସଂରଚନା htp_tensor *t) {
    uint64_t offset = t->data;
    uint32_t bi     = t->bi;

    t->data  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(ଉପର_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (unsigned long long) offset, t->size, (unsigned long long) t->data,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

ସ୍ଥିର ଶୂନ୍ୟ prep_tensors(ସଂରଚନା htp_context *ctx, ସଂରଚନା htp_buf_desc *bufs, ସଂରଚନା htp_tensor *tens, uint32_t n_tens) {
    ପାଇଁ (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mdev_group_init(ସଂରଚନା htp_context * ctx, ଅଚଳ ସଂରଚନା htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, ଆକାର(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା proc_op_req(ସଂରଚନା htp_ops_context * octx, ସଂରଚନା htp_buf_desc * bufs, uint32_t n_bufs,
                       ସଂରଚନା htp_tensor * tens, uint32_t idx, ସଂରଚନା htp_op_desc * op) {
    memcpy(octx->op_params,     op->params, ଆକାର(octx->op_params));
    memcpy(octx->kernel_params, op->kernel_params, ଆକାର(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_threads     = octx->ctx->n_threads;
    octx->n_threads_div = octx->ctx->n_threads_div;

    FARF(ଉପର_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    ପାଇଁ (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        ଯଦି (src_idx == 0xffff) {
            octx->src[i]     = ଖାଲି;
            ଜାରି;
        }

        ସଂରଚନା htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(ଉପର_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (unsigned long long) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    ପାଇଁ (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        ଯଦି (dst_idx == 0xffff) {
            octx->dsts[i]    = ଖାଲି;
            ଜାରି;
        }
        ସଂରଚନା htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(ଉପର_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (unsigned long long) dst->data, dst->size,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_group_barrier(octx);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା status = execute_op(octx);

    htp_ops_context_set_status(octx, status);

    octx->src0_spad.src = ଖାଲି;
    octx->src1_spad.src = ଖାଲି;
    octx->src2_spad.src = ଖାଲି;
    octx->src3_spad.src = ଖାଲି;
    octx->dst_spad.src  = ଖାଲି;

    ଫେରନ୍ତୁ octx->status;
}

ସ୍ଥିର ଶୂନ୍ୟ process_opbatch(ସଂରଚନା htp_context * ctx, ଅଚଳ ସଂରଚନା htp_opbatch_req * req, ଅଚଳ ସଂରଚନା dspqueue_buffer * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err;

    ଅଚଳ uint32_t n_bufs = req->n_bufs;
    ଅଚଳ uint32_t n_tens = req->n_tensors;
    ଅଚଳ uint32_t n_ops  = req->n_ops;

    ଅଚଳ uint32_t b_size = ଆକାର(ସଂରଚନା htp_buf_desc)  * n_bufs;
    ଅଚଳ uint32_t t_size = ଆକାର(ସଂରଚନା htp_tensor)    * n_tens;
    ଅଚଳ uint32_t o_size = ଆକାର(ସଂରଚନା htp_op_desc)   * n_ops;
    ଅଚଳ uint32_t p_size = ଆକାର(ସଂରଚନା htp_prof_desc) * n_ops;
    ଅଚଳ uint32_t tr_size = (HTP_MAX_NTHREADS + 1) * req->n_traces * ଆକାର(ସଂରଚନା htp_trace_desc);

    ଯଦି (dbuf->size < b_size + t_size + o_size + p_size + tr_size) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->size, b_size + t_size + o_size + p_size + tr_size);
        ଫେରନ୍ତୁ;
    }

    FARF(ଉପର_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (unsigned long long) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->size, b_size, t_size, o_size);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    ସଂରଚନା htp_buf_desc* bufs = (ସଂରଚନା htp_buf_desc*)  m_ptr; m_ptr += b_size;
    ସଂରଚନା htp_tensor*   tens = (ସଂରଚନା htp_tensor*)    m_ptr; m_ptr += t_size;
    ସଂରଚନା htp_op_desc*   ops = (ସଂରଚନା htp_op_desc*)   m_ptr; m_ptr += o_size;
    ସଂରଚନା htp_prof_desc* pds = (ସଂରଚନା htp_prof_desc*) m_ptr;

    ସଂରଚନା profile_data batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, ଆକାର(ctx->trace));
    ଯଦି (ctx->profiler == HTP_PROF_TRACE) {
        ସଂରଚନା htp_trace_desc * trace_events = (ସଂରଚନା htp_trace_desc *) (m_ptr + p_size);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, ଆକାର(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    ସଂରଚନା htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, ଆକାର(*octx));
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
    octx->ctx           = ctx;

    mdev_group_init(ctx, req);

    work_queue_wakeup(ctx->work_queue);
    ଯଦି (ctx->hmx_queue) {
        hmx_queue_wakeup(ctx->hmx_queue);
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_status = HTP_STATUS_OK;
    octx->status  = HTP_STATUS_OK;
    ପାଇଁ (uint32_t i = 0; i < n_ops; i++) {
        ସଂରଚନା profile_data prof;

        profile_start(ctx->profiler, &prof);

        op_status = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_set_status(octx, op_status);

        ଯଦି (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].cycles_start = prof.cycles_start;
            pds[i].cycles_stop = prof.cycles_stop;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < HEX_ଅଙ୍କ_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    ଯଦି (ctx->hmx_queue) {
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

    ସଂରଚନା htp_opbatch_rsp rsp;
    memset(&rsp, 0, ଆକାର(rsp));
    rsp.seq          = req->seq;
    rsp.status       = octx->status;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.cycles_start = batch_prof.cycles_start;
    rsp.cycles_stop  = batch_prof.cycles_stop;

    ଯଦି (ctx->profiler == HTP_PROF_TRACE) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].ଗଣନା_2;
        }
    }

    ସଂରଚନା dspqueue_buffer write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(queue, 0, 1, &write_dbuf, ଆକାର(rsp), (ଅଚଳ uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    ଯଦି (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
    }
}

#define DSPQUEUE_READ_TIMEOUT_USEC 5000
#define DSPQUEUE_POLL_TIMEOUT_USEC 100
#define DSPQUEUE_POLL_ଗଣନା        100

ସ୍ଥିର ଶୂନ୍ୟ process_ops(ସଂରଚନା htp_context * ctx) {
    dspqueue_t queue = ctx->dsp_queue;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err;

    uint32_t poll_ଗଣନା = DSPQUEUE_POLL_ଗଣନା;

    vtcm_acquire(ctx);

    ଯେତେବେଳେ (!atomic_load(&ctx->vtcm_needs_release) && !atomic_load(&ctx->killed)) {
        ସଂରଚନା htp_opbatch_req req;
        uint32_t r_size = ଆକାର(req);

        ସଂରଚନା dspqueue_buffer dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_size, &r_size, (uint8_t *) &req);
        ଯଦି (err == AEE_EWOULDBLOCK) {
            ଯଦି (--poll_ଗଣନା) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                ଜାରି;
            }
            ଭାଙ୍ଗ;
        }

        ଯଦି (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            ଭାଙ୍ଗ;
        }

        ଯଦି (r_size < ଆକାର(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_size, n_dbufs);
            ଜାରି;
        }

        // Reset poll count for valid requests
        poll_ଗଣନା = DSPQUEUE_POLL_ଗଣନା;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

ସ୍ଥିର ଶୂନ୍ୟ htp_packet_callback(dspqueue_t queue, ପୂର୍ଣ୍ଣସଂଖ୍ୟା error, ଶୂନ୍ୟ * context) {
    (ଶୂନ୍ୟ) queue;
    (ଶୂନ୍ୟ) error;
    ସଂରଚନା htp_handle * h = (ସଂରଚନା htp_handle *) context;
    ଯଦି (h && h->ctx) {
        process_ops(h->ctx);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ htp_main_thread(ଶୂନ୍ୟ * context) {
    ସଂରଚନା htp_context * ctx = (ସଂରଚନା htp_context *) context;

    FARF(ଉପର_2, "htp-main-thread: started");

    ଯେତେବେଳେ (!atomic_load(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t ଅଙ୍କ_buffers = 0;
        uint32_t message_length = 0;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = dspqueue_peek(ctx->dsp_queue, &flags, &ଅଙ୍କ_buffers, &message_length, 50000);
        ଯଦି (err == 0) {
            process_ops(ctx);
        } ଅନ୍ୟଥା ଯଦି (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            ଜାରି;
        } ଅନ୍ୟଥା {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (unsigned) err);
            ଭାଙ୍ଗ;
        }
    }

    FARF(ଉପର_2, "htp-main-thread: stopped");
}
