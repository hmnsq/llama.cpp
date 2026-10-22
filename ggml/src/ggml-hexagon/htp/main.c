#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>
#ضمن <AEEStdErr.h>
#ضمن <dspqueue.h>
#ضمن <HAP_compute_res.h>
#ضمن <HAP_etm_config.h>
#ضمن <HAP_mem.h>
#ضمن <HAP_power.h>
#ضمن <HAP_ps.h>
#ضمن <HAP_dcvs.h>
#ضمن <qurt.h>
#ضمن <qurt_thread.h>
#ضمن <qurt_memory.h>
#ضمن <remote.h>
#ضمن <string.h>
#ضمن <stdatomic.h>

#ضمن "hex-utils.h"
#ضمن "dma-queue.h"
#ضمن "hmx-queue.h"

#define GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-bitmap.h"
#ضمن "htp-ctx.h"
#ضمن "htp-dcache.h"
#ضمن "htp-ops.h"
#ضمن "htp_iface.h"
#ضمن "work-queue.h"
#ضمن "hex-profile.h"
#ضمن "allreduce-ops.h"
#ضمن "htp-fence.h"

#define HMX_QUEUE_CAPACITY     128
#define HMX_QUEUE_تر_SIZE   16384
#define WORK_QUEUE_CAPACITY    16
#define WORK_QUEUE_تر_SIZE  16384
#define MAIN_THREAD_تر_SIZE 32768

_Static_assert(WORK_QUEUE_MAX_N_THREADS >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

بنية htp_handle {
    بنية htp_context * ctx;
};

ساكن مضمن uint64_t htp_mmap(uint32_t fd, uint64_t size, uint32_t flags) {
#إذا __HVX_ARCH__ > 79
    إذا (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, حجم(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = size;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        إذا (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (unsigned long long) size);
            أرجع 0;
        }

        أرجع payload.mmap.dsp_va;
    }
#وإلا
    إذا (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        أرجع 0;
    }
#endif

    إذا (size > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (unsigned long long) size);
        أرجع 0;
    }

    فراغ * va = (فراغ *)-1;
    لكل (صحيح retry = 0; retry < 2; retry++) {
#إذا __HVX_ARCH__ > 73
        va = HAP_mmap2(عدم, (size_t) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#وإلا
        إذا (size > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (unsigned long long) size);
            abort();
        }
        va = HAP_mmap(عدم, (صحيح) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
        إذا (va != (فراغ *)-1 && va != عدم) {
            أرجع (uint64_t) (uintptr_t) va;
        }
        إذا (retry == 0) {
            FARF(سم_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (unsigned long long) size);
        }
    }
    أرجع 0;
}

ساكن مضمن فراغ htp_munmap(uint64_t va, uint64_t size, uint32_t flags) {
#إذا __HVX_ARCH__ > 79
    إذا (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, حجم(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = size;
        HAP_mem_request(&payload);
        أرجع;
    }
#endif

#إذا __HVX_ARCH__ > 73
    HAP_munmap2((فراغ *) (uintptr_t) va, (size_t) size);
#وإلا
    HAP_munmap((فراغ *) (uintptr_t) va, (صحيح) size);
#endif
}

AEEResult htp_iface_open(ثابت حرف * uri, remote_handle64 * handle) {
    (فراغ) uri;
    بنية htp_handle * h = calloc(1, حجم(*h));
    إذا (h == عدم) {
        أرجع AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    أرجع AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    بنية htp_handle * h = (بنية htp_handle *) handle;
    إذا (!h) {
        أرجع AEE_EBADPARM;
    }

    صحيح err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    إذا (err) {
        إذا (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } وإلا {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    أرجع err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, ثابت htp_iface_pmu_conf* pmu_conf) {
    بنية htp_handle * h = (بنية htp_handle *) handle;
    إذا (!h || !h->ctx) {
        أرجع AEE_EBADPARM;
    }
    بنية htp_context * ctx = h->ctx;

    إذا (mode == HTP_PROF_PMU) {
        ثابت uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        لكل (; i < HEX_لمب_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        لكل (i = 0; i < HEX_لمب_PMU_COUNTERS; i++) {
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

    أرجع AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    بنية htp_handle * h = (بنية htp_handle *) handle;
    إذا (!h) {
        أرجع AEE_EBADPARM;
    }

    بنية htp_context * ctx = h->ctx;
    إذا (ctx) {
        إذا (ctx->dsp_queue) {
            FARF(ERROR, "Closing handle with queue still open");
            أرجع AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        لكل (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            إذا (ctx->mmap[i].size) {
                htp_munmap(ctx->mmap[i].base, ctx->mmap[i].size, ctx->mmap[i].flags);
                ctx->mmap[i].size = 0;
                ctx->mmap[i].base = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        إذا (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        إذا (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        حرر(ctx);
        h->ctx = عدم;
    }

    حرر(h);
    أرجع AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t size) {
    بنية htp_handle * h = (بنية htp_handle *) handle;
    إذا (!h || !h->ctx) {
        أرجع AEE_EBADPARM;
    }
    بنية htp_context * ctx = h->ctx;

    // See if we already have this mapping
    لكل (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        بنية htp_mmap *m = &ctx->mmap[i];
        إذا (m->fd == fd) {
            أرجع AEE_SUCCESS;
        }
    }

    // Add new mapping
    لكل (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        بنية htp_mmap *m = &ctx->mmap[i];
        إذا (!m->size) {
            FARF(سم_2, "mmap : fd %u size %llu", fd, (unsigned long long) size);
            uint64_t va = htp_mmap(fd, size, 0);
            إذا (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (unsigned long long) size);
                أرجع AEE_EFAILED;
            }

            m->base   = va;
            m->fd     = fd;
            m->size   = size;
            m->flags  = 0;

            أرجع AEE_SUCCESS;
        }
    }

    أرجع AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    بنية htp_handle * h = (بنية htp_handle *) handle;
    إذا (!h || !h->ctx) {
        أرجع AEE_EBADPARM;
    }
    بنية htp_context * ctx = h->ctx;

    لكل (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        بنية htp_mmap *m = &ctx->mmap[i];
        إذا (fd < 0 || m->fd == fd) {
            FARF(سم_2, "unmmap : base 0x%llx fd %u size %llu", (unsigned long long) m->base, m->fd, (unsigned long long) m->size);
            htp_munmap(m->base, m->size, m->flags);
            m->size   = 0;
            m->base   = عدم;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    أرجع AEE_SUCCESS;
}

ساكن فراغ vtcm_acquire(بنية htp_context * ctx) {
    إذا (!atomic_load(&ctx->vtcm_valid)) {
        صحيح err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        إذا (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (unsigned)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

ساكن فراغ vtcm_release(بنية htp_context * ctx) {
    إذا (atomic_load(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

ساكن صحيح vtcm_release_callback(unsigned صحيح rctx, فراغ * state) {
    بنية htp_context * ctx = (بنية htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    أرجع 0;
}

ساكن unsigned صحيح vtcm_query_size(فراغ) {
    unsigned صحيح vtcm_size = 8 * 1024 * 1024;  // 8MB default
    unsigned صحيح vtcm_avail;
    compute_res_vtcm_page_t total_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_size, &total_layout, &vtcm_avail, &avail_layout);
    أرجع vtcm_size;
}

ساكن صحيح vtcm_alloc(بنية htp_context * ctx) {
    unsigned صحيح vtcm_size = vtcm_query_size();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (فراغ *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    إذا (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_size);
        أرجع AEE_ENOMEMORY;
    }

    فراغ * vtcm_ptr;
    إذا (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_size);
        أرجع AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size          = vtcm_size;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    أرجع 0;
}

ساكن فراغ vtcm_free(بنية htp_context * ctx) {
    إذا (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

ساكن فراغ htp_main_thread(فراغ * context);
ساكن فراغ htp_packet_callback(dspqueue_t queue, صحيح error, فراغ * context);
ساكن فراغ htp_error_callback(dspqueue_t queue, صحيح error, فراغ * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
بنية htp_power_profile {
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
        .hmx_perf_mode      = HAP_CLK_PERF_سم, \
    }

ساكن ثابت بنية htp_power_profile htp_power_profiles[HTP_POWER_MODE_كدي] = {
    [HTP_POWER_MODE_كس_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_MODE_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_MODE_سم_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_MODE_كس_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_MODE_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_MODE_سم_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_SUSTAINED_سم_PERFORMANCE] =
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
        .hmx_perf_mode      = HAP_CLK_PERF_سم,
    },
};

#undef HTP_POWER_PROFILE_ENTRY

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_mode) {
    بنية htp_handle * h = (بنية htp_handle *) handle;
    إذا (!h) {
        أرجع AEE_EBADPARM;
    }

    إذا (power_mode >= HTP_POWER_MODE_كدي) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (unsigned) power_mode);
        power_mode = HTP_POWER_MODE_BURST;
    }
    ثابت بنية htp_power_profile * pwr = &htp_power_profiles[power_mode];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (unsigned) power_mode, (صحيح) pwr->dcvs_enable,
         (unsigned) pwr->bus_min_corner, (unsigned) pwr->bus_max_corner, (unsigned) pwr->bus_target_corner,
         (unsigned) pwr->core_min_corner, (unsigned) pwr->core_max_corner, (unsigned) pwr->core_target_corner,
         (صحيح) pwr->sleep_disable,
         (unsigned) pwr->hmx_min_corner, (unsigned) pwr->hmx_max_corner, (unsigned) pwr->hmx_target_corner);

    إذا (h->ctx) {
        FARF(ERROR, "Queue already open");
        أرجع AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    صحيح fastrpc_tid  = qurt_thread_get_id();
    صحيح fastrpc_prio = qurt_thread_get_priority(fastrpc_tid);
    صحيح main_prio    = fastrpc_prio - 10;
    إذا (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_queue = عدم;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    صحيح err = dspqueue_import(dsp_queue_id, عدم, عدم, (فراغ *) h, &dsp_queue);
    إذا (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(سم_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (فراغ *) h, &dsp_queue);
    }

    إذا (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        أرجع err;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    إذا (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    إذا (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    إذا (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t offset = حجم(بنية htp_context);

    // 2. main_stack
    size_t offset_main_تر = 0;
    size_t size_main_تر   = 0;
    إذا (!use_callbacks) {
        offset_main_تر = hex_align_up(offset, 4096);
        size_main_تر   = MAIN_THREAD_تر_SIZE;
        offset = offset_main_تر + size_main_تر;
    }

    // 3. work_queue
    size_t wq_align  = work_queue_alignof();
    size_t offset_wq = hex_align_up(offset, wq_align);
    size_t size_wq   = work_queue_sizeof(n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_تر_SIZE);
    offset = offset_wq + size_wq;

    // 4. dma_queue
    size_t dma_align = dma_queue_alignof();
    size_t offset_dma = hex_align_up(offset, dma_align);
    size_t size_dma = 0;
    لكل (uint32_t i = 0; i < n_hvx; i++) {
        size_dma  = hex_align_up(size_dma, dma_queue_alignof());
        size_dma += dma_queue_sizeof(256);
    }
    offset = offset_dma + size_dma;

    // 5. hmx_queue
    size_t offset_hmx = 0;
    size_t size_hmx = 0;
    إذا (n_hmx) {
        size_t hmx_align = hmx_queue_alignof();
        offset_hmx = hex_align_up(offset, hmx_align);
        size_hmx   = hmx_queue_sizeof(HMX_QUEUE_CAPACITY, HMX_QUEUE_تر_SIZE);
        offset = offset_hmx + size_hmx;
    }

    size_t footprint = hex_align_up(offset, 128);

    فراغ * block = memalign(4096, footprint);
    إذا (!block) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_queue);
        أرجع AEE_ENOMEMORY;
    }
    memset(block, 0, footprint);

    h->ctx = (بنية htp_context *) block;
    بنية htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_queue   = dsp_queue;

    err = vtcm_alloc(ctx);
    إذا (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        أرجع AEE_ENOMEMORY;
    }

    HAP_setFARFRuntimeLoggingParams(0xffff, عدم, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, حجم(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        إذا ((err = HAP_power_set((فراغ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            أرجع err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, حجم(request));

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

#إذا (__HEXAGON_ARCH__ >= 79)
        HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
#endif
        إذا ((err = HAP_power_set((فراغ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            أرجع err;
        }

        memset(&request, 0, حجم(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        إذا ((err = HAP_power_set((فراغ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            أرجع err;
        }
    }

#إذا __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, حجم(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_power     = TRUE;
        request.hmx_v2.power_up      = TRUE;
        request.hmx_v2.set_clock     = TRUE;
        request.hmx_v2.target_corner = pwr->hmx_target_corner;
        request.hmx_v2.min_corner    = pwr->hmx_min_corner;
        request.hmx_v2.max_corner    = pwr->hmx_max_corner;
        request.hmx_v2.perf_mode     = pwr->hmx_perf_mode;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((فراغ *) ctx, &request);
        إذا (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            أرجع err;
        }
    }
#وإلا
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, حجم(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((فراغ *) ctx, &request);
        إذا (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            أرجع err;
        }
    }
#endif

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_queue   = عدم;
    إذا (n_hmx) {
        فراغ * hmx_ptr = (فراغ *) ((uintptr_t) block + offset_hmx);
        ctx->hmx_queue = hmx_queue_init(hmx_ptr, HMX_QUEUE_CAPACITY, HMX_QUEUE_تر_SIZE, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(سم_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_threads = n_hvx;
    ctx->n_threads_div = init_fastdiv_درجوجي(ctx->n_threads);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) block + offset_dma);
    size_t size_dma_q = dma_queue_sizeof(256);

    لكل (صحيح i = 0; i < ctx->n_threads; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_up((uintptr_t) dma_ptr_curr, dma_queue_alignof());
        ctx->dma[i] = dma_queue_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += size_dma_q;
    }

    ctx->ddr_spad_size = 512 * 1024; // 512 KB
    ctx->ddr_spad_base = memalign(128, ctx->ddr_spad_size);

    فراغ * wq_ptr = (فراغ *) ((uintptr_t) block + offset_wq);
    ctx->work_queue = work_queue_init(wq_ptr, n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_تر_SIZE);

    ctx->main_تر = عدم;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    إذا (!use_callbacks) {
        // Start main compute thread
        ctx->main_تر = (فراغ *) ((uintptr_t) block + offset_main_تر);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_تر_addr(&attr, ctx->main_تر);
        qurt_thread_attr_set_تر_size(&attr, size_main_تر);
        qurt_thread_attr_set_priority(&attr, main_prio);
        qurt_thread_attr_set_name(&attr, "htp-main");

        صحيح err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        إذا (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            أرجع AEE_ENOMEMORY;
        }
    }

    FARF(سم_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_size, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    أرجع AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    بنية htp_handle * h = (بنية htp_handle *) handle;
    إذا (!h || !h->ctx) {
        أرجع AEE_EBADPARM;
    }
    بنية htp_context * ctx = h->ctx;

    إذا (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        صحيح status;
        (فراغ) qurt_thread_join(ctx->main_thread, &status);
        ctx->main_thread = 0;
    }

    صحيح err = dspqueue_close(ctx->dsp_queue); ctx->dsp_queue = عدم;
    إذا (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (unsigned) err);
        أرجع err;
    }

    work_queue_free(ctx->work_queue);

    لكل (صحيح i = 0; i < ctx->n_threads; i++) {
        dma_queue_free(ctx->dma[i]);
    }

    إذا (ctx->hmx_queue) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = عدم;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    إذا (ctx->ddr_spad_base) {
        حرر(ctx->ddr_spad_base);
        ctx->ddr_spad_base = عدم;
        ctx->ddr_spad_size = 0;
    }

    حرر(ctx);
    h->ctx = عدم;

    أرجع AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_threads, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_size) {
    (فراغ)handle;
    إذا (!n_threads || !n_hvx || !n_hmx || !vtcm_size) {
        أرجع AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    إذا (n_hvx_val > hw_threads.max_hthreads) {
        n_hvx_val = hw_threads.max_hthreads;
    }
    إذا (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_threads = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_size = vtcm_query_size();

    أرجع AEE_SUCCESS;
}

ساكن فراغ htp_error_callback(dspqueue_t queue, صحيح error, فراغ * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (unsigned) error);
}

بنية profile_data {
    uint64_t usecs;
    uint64_t cycles_start;
    uint64_t cycles_stop;
    uint32_t pmu_counters[HEX_لمب_PMU_COUNTERS];
};

ساكن مضمن فراغ profile_start(uint32_t mode, بنية profile_data * d) {
    بدل (mode) {
        حالة HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        حالة HTP_PROF_BASIC:
        حالة HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_كدي();
            d->cycles_start = hex_get_cycles();
            اكسر;
        افتراضي:
            اكسر;
    }
}

ساكن مضمن فراغ profile_stop(uint32_t mode, بنية profile_data * d) {
    uint32_t pmu_counters[HEX_لمب_PMU_COUNTERS];
    بدل (mode) {
        حالة HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            لكل (صحيح i = 0; i < HEX_لمب_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        حالة HTP_PROF_BASIC:
        حالة HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_كدي_to_us(HAP_perf_get_qtimer_كدي() - d->usecs);
            d->cycles_stop = hex_get_cycles();
            اكسر;
        افتراضي:
            اكسر;
    }
}

ساكن صحيح op_fence(بنية htp_ops_context * octx) {
    بنية htp_context *ctx = octx->ctx;
    بنية htp_thread_trace * tr = &ctx->trace[0];
    ثابت uint32_t seq  = (uint32_t) octx->op_params[0];
    ثابت uint32_t mode = (uint32_t) octx->op_params[1];

    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ثابت بنية htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;

    إذا (mode == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_group_barrier(octx);

        إذا (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->status);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(سم_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        أرجع octx->status;
    }

    صحيح status = HTP_STATUS_OK;
    uint64_t spins = 0;
    طالما (1) {
        uint32_t sync_seq;
        uint32_t sync_status;
        htp_fence_read(sync_fence, &sync_seq, &sync_status);
        إذا ((int32_t)(sync_seq - seq) >= 0) {
            إذا (sync_status > HTP_STATUS_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_status, sync_fence, seq);
                status = sync_status;
            }
            اكسر;
        }
        إذا (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            status = HTP_STATUS_INTERNAL_ERR;
            اكسر;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(سم_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    أرجع status;
}

ساكن صحيح op_mdev_group(بنية htp_ops_context * octx) {
    بنية htp_context * ctx = octx->ctx;
    ثابت بنية htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.كدي = (uint16_t) sync->ne[1];
    إذا (ctx->mdev.كدي > 1) {
        ctx->mdev.كدي_div = init_fastdiv_درجوجي(ctx->mdev.كدي);
        ctx->mdev.fence_base = (uint8_t *) sync->data;
    }
    أرجع HTP_STATUS_OK;
}

ساكن صحيح execute_op(بنية htp_ops_context * octx) {
    بدل (octx->op) {
        حالة HTP_OP_MDEV_GROUP:
            أرجع op_mdev_group(octx);

        حالة HTP_OP_FENCE:
            أرجع op_fence(octx);

        حالة HTP_OP_ALLREDUCE:
        حالة HTP_OP_ALLREDUCE_ADD:
            أرجع op_allreduce(octx);

        حالة HTP_OP_MUL_MAT:
        حالة HTP_OP_MUL_MAT_ADD:
            أرجع op_matmul(octx);

        حالة HTP_OP_MUL_MAT_ID:
            أرجع op_matmul_id(octx);

        حالة HTP_OP_MUL_MAT_ID_NX:
            أرجع op_matmul_id_nx(octx);

        حالة HTP_OP_MUL_MAT_NX:
            أرجع op_matmul_nx(octx);

        حالة HTP_OP_MUL:
        حالة HTP_OP_ADD:
        حالة HTP_OP_SUB:
        حالة HTP_OP_DIV:
        حالة HTP_OP_ADD_ID:
            أرجع op_نبيو(octx);

        حالة HTP_OP_NORM:
        حالة HTP_OP_RMS_NORM:
        حالة HTP_OP_RMS_NORM_MUL:
        حالة HTP_OP_SCALE:
        حالة HTP_OP_CLAMP:
        حالة HTP_OP_LEAKY_RELU:
        حالة HTP_OP_SQR:
        حالة HTP_OP_SQRT:
        حالة HTP_OP_UNARY_SOFTPLUS:
        حالة HTP_OP_UNARY_SIGMOID:
        حالة HTP_OP_UNARY_SILU:
        حالة HTP_OP_UNARY_GELU:
        حالة HTP_OP_UNARY_GELU_QUICK:
        حالة HTP_OP_UNARY_GELU_ERF:
        حالة HTP_OP_UNARY_NEG:
        حالة HTP_OP_UNARY_EXP:
        حالة HTP_OP_UNARY_TANH:
        حالة HTP_OP_UNARY_ABS:
        حالة HTP_OP_UNARY_LOG:
        حالة HTP_OP_UNARY_RELU:
        حالة HTP_OP_UNARY_STEP:
        حالة HTP_OP_L2_NORM:
        حالة HTP_OP_HADAMARD:
            أرجع op_unary(octx);

        حالة HTP_OP_GLU_SWIGLU:
        حالة HTP_OP_GLU_SWIGLU_OAI:
        حالة HTP_OP_GLU_SWIGLU_CLAMP:
        حالة HTP_OP_GLU_GEGLU:
        حالة HTP_OP_GLU_GEGLU_QUICK:
        حالة HTP_OP_GLU_GEGLU_ERF:
            أرجع op_activations(octx);

        حالة HTP_OP_SOFTMAX:
            أرجع op_softmax(octx);

        حالة HTP_OP_ROPE:
            أرجع op_rope(octx);

        حالة HTP_OP_FLASH_ATTN_EXT:
            أرجع op_flash_attn_ext(octx);

        حالة HTP_OP_SET_ROWS:
            أرجع op_set_rows(octx);

        حالة HTP_OP_GET_ROWS:
            أرجع op_get_rows(octx);

        حالة HTP_OP_SUM:
            أرجع op_sum(octx);

        حالة HTP_OP_SUM_ROWS:
            أرجع op_sum_rows(octx);

        حالة HTP_OP_CPY:
        حالة HTP_OP_CPY_FENCE:
            أرجع op_cpy(octx);

        حالة HTP_OP_REPEAT:
            أرجع op_repeat(octx);

        حالة HTP_OP_ARGSORT:
            أرجع op_argsort(octx);

        حالة HTP_OP_كولي_K:
            أرجع op_كولي_k(octx);

        حالة HTP_OP_ARGMAX:
            أرجع op_argmax(octx);

        حالة HTP_OP_SSM_CONV:
            أرجع op_ssm_conv(octx);

        حالة HTP_OP_CUMSUM:
            أرجع op_cumsum(octx);

        حالة HTP_OP_FILL:
            أرجع op_fill(octx);

        حالة HTP_OP_DIAG:
            أرجع op_diag(octx);

        حالة HTP_OP_SOLVE_TRI:
            أرجع op_solve_tri(octx);

        حالة HTP_OP_PAD:
            أرجع op_pad(octx);

        حالة HTP_OP_IM2COL:
            أرجع op_im2col(octx);

        حالة HTP_OP_IM2COL_3D:
            أرجع op_im2col_3d(octx);

        حالة HTP_OP_ROLL:
            أرجع op_roll(octx);

        حالة HTP_OP_POOL_2D:
            أرجع op_pool_2d(octx);

        حالة HTP_OP_POOL_1D:
            أرجع op_pool_1d(octx);

        حالة HTP_OP_CONCAT:
            أرجع op_concat(octx);

        حالة HTP_OP_GATED_DELTA_NET:
            أرجع op_gated_delta_net(octx);

        حالة HTP_OP_TRI:
            أرجع op_unary(octx);

        حالة HTP_OP_INVALID:
            اكسر;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    أرجع HTP_STATUS_NO_SUPPORT;
}

ساكن مضمن bool reuse_buf(بنية htp_context *ctx, uint32_t *m_reuse, بنية htp_buf_desc *b) {
    b->base = عدم;

    لكل (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        بنية htp_mmap *m = ctx->mmap + i;
        إذا (m->size && m->fd == b->fd && m->flags == b->flags) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            أرجع true;
        }
    }

    أرجع false;
}

ساكن مضمن فراغ drop_mmap(بنية htp_context *ctx, بنية htp_mmap *m) {
    إذا (m->size) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (unsigned long long) m->base, (unsigned long long) m->size);
        htp_munmap(m->base, m->size, m->flags);
        m->size = 0;
        m->base = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

ساكن مضمن bool mmap_buf(بنية htp_context *ctx, بنية htp_buf_desc *b) {
    إذا (b->base) أرجع true; // already mapped

    // find unused mapping
    لكل (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        بنية htp_mmap *m = &ctx->mmap[i];
        إذا (!m->size) {
            uint64_t va = htp_mmap(b->fd, b->size, b->flags);
            إذا (va == 0) {
                FARF(سم_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (unsigned long long) b->size);
                أرجع false;
            }

            m->base   = b->base = va;
            m->fd     = b->fd;
            m->size   = b->size;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (unsigned long long) m->base, (unsigned long long) m->size, m->flags);
            أرجع true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    أرجع false;
}

ساكن فراغ prep_op_bufs(بنية htp_context *ctx, بنية htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    لكل (uint32_t i=0; i < n_bufs; i++) {
        بنية htp_buf_desc *b = bufs + i;
        إذا (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } وإلا إذا (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->size;
        }
        FARF(سم_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    إذا (b_reuse == n_bufs) أرجع; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    لكل (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        إذا (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].size;
        }
    }

    FARF(سم_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    إذا ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        لكل (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            إذا (!used && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    لكل (uint32_t i=0; i < n_bufs; i++) {
        بنية htp_buf_desc *b = bufs + i;
        إذا (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            اكسر;
        }
        FARF(سم_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    إذا (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(سم_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        لكل (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            إذا (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        لكل (uint32_t i=0; i < n_bufs; i++) {
            بنية htp_buf_desc *b = bufs + i;
            إذا (!(b->flags & HTP_BUF_EXTENDED)) {
                b->base = 0;
            }
            إذا (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (unsigned long long) b->size);
                abort();
            }
            FARF(سم_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
        }
    }
}

ساكن فراغ prep_tensor(بنية htp_context *ctx, بنية htp_buf_desc *bufs, بنية htp_tensor *tens, uint32_t idx, بنية htp_tensor *t) {
    uint64_t offset = t->data;
    uint32_t bi     = t->bi;

    t->data  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(سم_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (unsigned long long) offset, t->size, (unsigned long long) t->data,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

ساكن فراغ prep_tensors(بنية htp_context *ctx, بنية htp_buf_desc *bufs, بنية htp_tensor *tens, uint32_t n_tens) {
    لكل (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

ساكن فراغ mdev_group_init(بنية htp_context * ctx, ثابت بنية htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, حجم(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

ساكن صحيح proc_op_req(بنية htp_ops_context * octx, بنية htp_buf_desc * bufs, uint32_t n_bufs,
                       بنية htp_tensor * tens, uint32_t idx, بنية htp_op_desc * op) {
    memcpy(octx->op_params,     op->params, حجم(octx->op_params));
    memcpy(octx->kernel_params, op->kernel_params, حجم(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_threads     = octx->ctx->n_threads;
    octx->n_threads_div = octx->ctx->n_threads_div;

    FARF(سم_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    لكل (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        إذا (src_idx == 0xffff) {
            octx->src[i]     = عدم;
            استمر;
        }

        بنية htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(سم_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (unsigned long long) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    لكل (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        إذا (dst_idx == 0xffff) {
            octx->dsts[i]    = عدم;
            استمر;
        }
        بنية htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(سم_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (unsigned long long) dst->data, dst->size,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_group_barrier(octx);

    صحيح status = execute_op(octx);

    htp_ops_context_set_status(octx, status);

    octx->src0_spad.src = عدم;
    octx->src1_spad.src = عدم;
    octx->src2_spad.src = عدم;
    octx->src3_spad.src = عدم;
    octx->dst_spad.src  = عدم;

    أرجع octx->status;
}

ساكن فراغ process_opbatch(بنية htp_context * ctx, ثابت بنية htp_opbatch_req * req, ثابت بنية dspqueue_buffer * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    صحيح err;

    ثابت uint32_t n_bufs = req->n_bufs;
    ثابت uint32_t n_tens = req->n_tensors;
    ثابت uint32_t n_ops  = req->n_ops;

    ثابت uint32_t b_size = حجم(بنية htp_buf_desc)  * n_bufs;
    ثابت uint32_t t_size = حجم(بنية htp_tensor)    * n_tens;
    ثابت uint32_t o_size = حجم(بنية htp_op_desc)   * n_ops;
    ثابت uint32_t p_size = حجم(بنية htp_prof_desc) * n_ops;
    ثابت uint32_t tr_size = (HTP_MAX_NTHREADS + 1) * req->n_traces * حجم(بنية htp_trace_desc);

    إذا (dbuf->size < b_size + t_size + o_size + p_size + tr_size) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->size, b_size + t_size + o_size + p_size + tr_size);
        أرجع;
    }

    FARF(سم_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (unsigned long long) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->size, b_size, t_size, o_size);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    بنية htp_buf_desc* bufs = (بنية htp_buf_desc*)  m_ptr; m_ptr += b_size;
    بنية htp_tensor*   tens = (بنية htp_tensor*)    m_ptr; m_ptr += t_size;
    بنية htp_op_desc*   ops = (بنية htp_op_desc*)   m_ptr; m_ptr += o_size;
    بنية htp_prof_desc* pds = (بنية htp_prof_desc*) m_ptr;

    بنية profile_data batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, حجم(ctx->trace));
    إذا (ctx->profiler == HTP_PROF_TRACE) {
        بنية htp_trace_desc * trace_events = (بنية htp_trace_desc *) (m_ptr + p_size);
        لكل (صحيح t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    بنية htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, حجم(*octx));
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
    octx->ctx           = ctx;

    mdev_group_init(ctx, req);

    work_queue_wakeup(ctx->work_queue);
    إذا (ctx->hmx_queue) {
        hmx_queue_wakeup(ctx->hmx_queue);
    }

    صحيح op_status = HTP_STATUS_OK;
    octx->status  = HTP_STATUS_OK;
    لكل (uint32_t i = 0; i < n_ops; i++) {
        بنية profile_data prof;

        profile_start(ctx->profiler, &prof);

        op_status = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_set_status(octx, op_status);

        إذا (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].cycles_start = prof.cycles_start;
            pds[i].cycles_stop = prof.cycles_stop;
            لكل (صحيح j = 0; j < HEX_لمب_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    إذا (ctx->hmx_queue) {
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

    بنية htp_opbatch_rsp rsp;
    memset(&rsp, 0, حجم(rsp));
    rsp.seq          = req->seq;
    rsp.status       = octx->status;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.cycles_start = batch_prof.cycles_start;
    rsp.cycles_stop  = batch_prof.cycles_stop;

    إذا (ctx->profiler == HTP_PROF_TRACE) {
        لكل (صحيح t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].كدي;
        }
    }

    بنية dspqueue_buffer write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(queue, 0, 1, &write_dbuf, حجم(rsp), (ثابت uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    إذا (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
    }
}

#define DSPQUEUE_READ_TIMEOUT_USEC 5000
#define DSPQUEUE_POLL_TIMEOUT_USEC 100
#define DSPQUEUE_POLL_كدي        100

ساكن فراغ process_ops(بنية htp_context * ctx) {
    dspqueue_t queue = ctx->dsp_queue;
    صحيح err;

    uint32_t poll_كدي = DSPQUEUE_POLL_كدي;

    vtcm_acquire(ctx);

    طالما (!atomic_load(&ctx->vtcm_needs_release) && !atomic_load(&ctx->killed)) {
        بنية htp_opbatch_req req;
        uint32_t r_size = حجم(req);

        بنية dspqueue_buffer dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_size, &r_size, (uint8_t *) &req);
        إذا (err == AEE_EWOULDBLOCK) {
            إذا (--poll_كدي) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                استمر;
            }
            اكسر;
        }

        إذا (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            اكسر;
        }

        إذا (r_size < حجم(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_size, n_dbufs);
            استمر;
        }

        // Reset poll count for valid requests
        poll_كدي = DSPQUEUE_POLL_كدي;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

ساكن فراغ htp_packet_callback(dspqueue_t queue, صحيح error, فراغ * context) {
    (فراغ) queue;
    (فراغ) error;
    بنية htp_handle * h = (بنية htp_handle *) context;
    إذا (h && h->ctx) {
        process_ops(h->ctx);
    }
}

ساكن فراغ htp_main_thread(فراغ * context) {
    بنية htp_context * ctx = (بنية htp_context *) context;

    FARF(سم_2, "htp-main-thread: started");

    طالما (!atomic_load(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t لمب_buffers = 0;
        uint32_t message_length = 0;

        صحيح err = dspqueue_peek(ctx->dsp_queue, &flags, &لمب_buffers, &message_length, 50000);
        إذا (err == 0) {
            process_ops(ctx);
        } وإلا إذا (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            استمر;
        } وإلا {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (unsigned) err);
            اكسر;
        }
    }

    FARF(سم_2, "htp-main-thread: stopped");
}
