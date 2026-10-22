#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাৱ <HAP_farf.h>
#অন্তর্ভাৱ <HAP_perf.h>
#অন্তর্ভাৱ <AEEStdErr.h>
#অন্তর্ভাৱ <dspqueue.h>
#অন্তর্ভাৱ <HAP_compute_res.h>
#অন্তর্ভাৱ <HAP_etm_config.h>
#অন্তর্ভাৱ <HAP_mem.h>
#অন্তর্ভাৱ <HAP_power.h>
#অন্তর্ভাৱ <HAP_ps.h>
#অন্তর্ভাৱ <HAP_dcvs.h>
#অন্তর্ভাৱ <qurt.h>
#অন্তর্ভাৱ <qurt_thread.h>
#অন্তর্ভাৱ <qurt_memory.h>
#অন্তর্ভাৱ <remote.h>
#অন্তর্ভাৱ <string.h>
#অন্তর্ভাৱ <stdatomic.h>

#অন্তর্ভাৱ "hex-utils.h"
#অন্তর্ভাৱ "dma-queue.h"
#অন্তর্ভাৱ "hmx-queue.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাৱ "ggml-common.h"
#অন্তর্ভাৱ "hex-bitmap.h"
#অন্তর্ভাৱ "htp-ctx.h"
#অন্তর্ভাৱ "htp-dcache.h"
#অন্তর্ভাৱ "htp-ops.h"
#অন্তর্ভাৱ "htp_iface.h"
#অন্তর্ভাৱ "work-queue.h"
#অন্তর্ভাৱ "hex-profile.h"
#অন্তর্ভাৱ "allreduce-ops.h"
#অন্তর্ভাৱ "htp-fence.h"

#define HMX_QUEUE_CAPACITY     128
#define HMX_QUEUE_স্তূপ_SIZE   16384
#define WORK_QUEUE_CAPACITY    16
#define WORK_QUEUE_স্তূপ_SIZE  16384
#define MAIN_THREAD_স্তূপ_SIZE 32768

_Static_assert(WORK_QUEUE_MAX_N_THREADS >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

সংরচনা htp_handle {
    সংরচনা htp_context * ctx;
};

স্থির অন্তঃস্থ uint64_t htp_mmap(uint32_t fd, uint64_t size, uint32_t flags) {
#যদি __HVX_ARCH__ > 79
    যদি (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, পরিমাণ(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = size;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        যদি (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (unsigned long long) size);
            ঘূৰাই 0;
        }

        ঘূৰাই payload.mmap.dsp_va;
    }
#অন্যথা
    যদি (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        ঘূৰাই 0;
    }
#endif

    যদি (size > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (unsigned long long) size);
        ঘূৰাই 0;
    }

    শূন্য * va = (শূন্য *)-1;
    বাবে (পূর্ণাঙ্ক retry = 0; retry < 2; retry++) {
#যদি __HVX_ARCH__ > 73
        va = HAP_mmap2(রিক্ত, (size_t) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#অন্যথা
        যদি (size > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (unsigned long long) size);
            abort();
        }
        va = HAP_mmap(রিক্ত, (পূর্ণাঙ্ক) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
        যদি (va != (শূন্য *)-1 && va != রিক্ত) {
            ঘূৰাই (uint64_t) (uintptr_t) va;
        }
        যদি (retry == 0) {
            FARF(উপৰেৰ_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (unsigned long long) size);
        }
    }
    ঘূৰাই 0;
}

স্থির অন্তঃস্থ শূন্য htp_munmap(uint64_t va, uint64_t size, uint32_t flags) {
#যদি __HVX_ARCH__ > 79
    যদি (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, পরিমাণ(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = size;
        HAP_mem_request(&payload);
        ঘূৰাই;
    }
#endif

#যদি __HVX_ARCH__ > 73
    HAP_munmap2((শূন্য *) (uintptr_t) va, (size_t) size);
#অন্যথা
    HAP_munmap((শূন্য *) (uintptr_t) va, (পূর্ণাঙ্ক) size);
#endif
}

AEEResult htp_iface_open(ধ্রুৱ ৱর্ণ * uri, remote_handle64 * handle) {
    (শূন্য) uri;
    সংরচনা htp_handle * h = calloc(1, পরিমাণ(*h));
    যদি (h == রিক্ত) {
        ঘূৰাই AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    ঘূৰাই AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    সংরচনা htp_handle * h = (সংরচনা htp_handle *) handle;
    যদি (!h) {
        ঘূৰাই AEE_EBADPARM;
    }

    পূর্ণাঙ্ক err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    যদি (err) {
        যদি (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } অন্যথা {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    ঘূৰাই err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, ধ্রুৱ htp_iface_pmu_conf* pmu_conf) {
    সংরচনা htp_handle * h = (সংরচনা htp_handle *) handle;
    যদি (!h || !h->ctx) {
        ঘূৰাই AEE_EBADPARM;
    }
    সংরচনা htp_context * ctx = h->ctx;

    যদি (mode == HTP_PROF_PMU) {
        ধ্রুৱ uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        বাবে (; i < HEX_অঙ্ক_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        বাবে (i = 0; i < HEX_অঙ্ক_PMU_COUNTERS; i++) {
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

    ঘূৰাই AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    সংরচনা htp_handle * h = (সংরচনা htp_handle *) handle;
    যদি (!h) {
        ঘূৰাই AEE_EBADPARM;
    }

    সংরচনা htp_context * ctx = h->ctx;
    যদি (ctx) {
        যদি (ctx->dsp_queue) {
            FARF(ERROR, "Closing handle with queue still open");
            ঘূৰাই AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        বাবে (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            যদি (ctx->mmap[i].size) {
                htp_munmap(ctx->mmap[i].base, ctx->mmap[i].size, ctx->mmap[i].flags);
                ctx->mmap[i].size = 0;
                ctx->mmap[i].base = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        যদি (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        যদি (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        মুক্ত(ctx);
        h->ctx = রিক্ত;
    }

    মুক্ত(h);
    ঘূৰাই AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t size) {
    সংরচনা htp_handle * h = (সংরচনা htp_handle *) handle;
    যদি (!h || !h->ctx) {
        ঘূৰাই AEE_EBADPARM;
    }
    সংরচনা htp_context * ctx = h->ctx;

    // See if we already have this mapping
    বাবে (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        সংরচনা htp_mmap *m = &ctx->mmap[i];
        যদি (m->fd == fd) {
            ঘূৰাই AEE_SUCCESS;
        }
    }

    // Add new mapping
    বাবে (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        সংরচনা htp_mmap *m = &ctx->mmap[i];
        যদি (!m->size) {
            FARF(উপৰেৰ_2, "mmap : fd %u size %llu", fd, (unsigned long long) size);
            uint64_t va = htp_mmap(fd, size, 0);
            যদি (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (unsigned long long) size);
                ঘূৰাই AEE_EFAILED;
            }

            m->base   = va;
            m->fd     = fd;
            m->size   = size;
            m->flags  = 0;

            ঘূৰাই AEE_SUCCESS;
        }
    }

    ঘূৰাই AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    সংরচনা htp_handle * h = (সংরচনা htp_handle *) handle;
    যদি (!h || !h->ctx) {
        ঘূৰাই AEE_EBADPARM;
    }
    সংরচনা htp_context * ctx = h->ctx;

    বাবে (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        সংরচনা htp_mmap *m = &ctx->mmap[i];
        যদি (fd < 0 || m->fd == fd) {
            FARF(উপৰেৰ_2, "unmmap : base 0x%llx fd %u size %llu", (unsigned long long) m->base, m->fd, (unsigned long long) m->size);
            htp_munmap(m->base, m->size, m->flags);
            m->size   = 0;
            m->base   = রিক্ত;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    ঘূৰাই AEE_SUCCESS;
}

স্থির শূন্য vtcm_acquire(সংরচনা htp_context * ctx) {
    যদি (!atomic_load(&ctx->vtcm_valid)) {
        পূর্ণাঙ্ক err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        যদি (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (unsigned)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

স্থির শূন্য vtcm_release(সংরচনা htp_context * ctx) {
    যদি (atomic_load(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

স্থির পূর্ণাঙ্ক vtcm_release_callback(unsigned পূর্ণাঙ্ক rctx, শূন্য * state) {
    সংরচনা htp_context * ctx = (সংরচনা htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    ঘূৰাই 0;
}

স্থির unsigned পূর্ণাঙ্ক vtcm_query_size(শূন্য) {
    unsigned পূর্ণাঙ্ক vtcm_size = 8 * 1024 * 1024;  // 8MB default
    unsigned পূর্ণাঙ্ক vtcm_avail;
    compute_res_vtcm_page_t total_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_size, &total_layout, &vtcm_avail, &avail_layout);
    ঘূৰাই vtcm_size;
}

স্থির পূর্ণাঙ্ক vtcm_alloc(সংরচনা htp_context * ctx) {
    unsigned পূর্ণাঙ্ক vtcm_size = vtcm_query_size();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (শূন্য *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    যদি (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_size);
        ঘূৰাই AEE_ENOMEMORY;
    }

    শূন্য * vtcm_ptr;
    যদি (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_size);
        ঘূৰাই AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size          = vtcm_size;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    ঘূৰাই 0;
}

স্থির শূন্য vtcm_free(সংরচনা htp_context * ctx) {
    যদি (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

স্থির শূন্য htp_main_thread(শূন্য * context);
স্থির শূন্য htp_packet_callback(dspqueue_t queue, পূর্ণাঙ্ক error, শূন্য * context);
স্থির শূন্য htp_error_callback(dspqueue_t queue, পূর্ণাঙ্ক error, শূন্য * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
সংরচনা htp_power_profile {
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
        .hmx_perf_mode      = HAP_CLK_PERF_উপৰেৰ, \
    }

স্থির ধ্রুৱ সংরচনা htp_power_profile htp_power_profiles[HTP_POWER_MODE_গুনতি] = {
    [HTP_POWER_MODE_নিচেৰ_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_MODE_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_MODE_উপৰেৰ_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_MODE_নিচেৰ_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_MODE_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_MODE_উপৰেৰ_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_SUSTAINED_উপৰেৰ_PERFORMANCE] =
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
        .hmx_perf_mode      = HAP_CLK_PERF_উপৰেৰ,
    },
};

#undef HTP_POWER_PROFILE_ENTRY

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_mode) {
    সংরচনা htp_handle * h = (সংরচনা htp_handle *) handle;
    যদি (!h) {
        ঘূৰাই AEE_EBADPARM;
    }

    যদি (power_mode >= HTP_POWER_MODE_গুনতি) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (unsigned) power_mode);
        power_mode = HTP_POWER_MODE_BURST;
    }
    ধ্রুৱ সংরচনা htp_power_profile * pwr = &htp_power_profiles[power_mode];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (unsigned) power_mode, (পূর্ণাঙ্ক) pwr->dcvs_enable,
         (unsigned) pwr->bus_min_corner, (unsigned) pwr->bus_max_corner, (unsigned) pwr->bus_target_corner,
         (unsigned) pwr->core_min_corner, (unsigned) pwr->core_max_corner, (unsigned) pwr->core_target_corner,
         (পূর্ণাঙ্ক) pwr->sleep_disable,
         (unsigned) pwr->hmx_min_corner, (unsigned) pwr->hmx_max_corner, (unsigned) pwr->hmx_target_corner);

    যদি (h->ctx) {
        FARF(ERROR, "Queue already open");
        ঘূৰাই AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    পূর্ণাঙ্ক fastrpc_tid  = qurt_thread_get_id();
    পূর্ণাঙ্ক fastrpc_prio = qurt_thread_get_priority(fastrpc_tid);
    পূর্ণাঙ্ক main_prio    = fastrpc_prio - 10;
    যদি (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_queue = রিক্ত;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    পূর্ণাঙ্ক err = dspqueue_import(dsp_queue_id, রিক্ত, রিক্ত, (শূন্য *) h, &dsp_queue);
    যদি (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(উপৰেৰ_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (শূন্য *) h, &dsp_queue);
    }

    যদি (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        ঘূৰাই err;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    যদি (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    যদি (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    যদি (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t offset = পরিমাণ(সংরচনা htp_context);

    // 2. main_stack
    size_t offset_main_স্তূপ = 0;
    size_t size_main_স্তূপ   = 0;
    যদি (!use_callbacks) {
        offset_main_স্তূপ = hex_align_up(offset, 4096);
        size_main_স্তূপ   = MAIN_THREAD_স্তূপ_SIZE;
        offset = offset_main_স্তূপ + size_main_স্তূপ;
    }

    // 3. work_queue
    size_t wq_align  = work_queue_alignof();
    size_t offset_wq = hex_align_up(offset, wq_align);
    size_t size_wq   = work_queue_sizeof(n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_স্তূপ_SIZE);
    offset = offset_wq + size_wq;

    // 4. dma_queue
    size_t dma_align = dma_queue_alignof();
    size_t offset_dma = hex_align_up(offset, dma_align);
    size_t size_dma = 0;
    বাবে (uint32_t i = 0; i < n_hvx; i++) {
        size_dma  = hex_align_up(size_dma, dma_queue_alignof());
        size_dma += dma_queue_sizeof(256);
    }
    offset = offset_dma + size_dma;

    // 5. hmx_queue
    size_t offset_hmx = 0;
    size_t size_hmx = 0;
    যদি (n_hmx) {
        size_t hmx_align = hmx_queue_alignof();
        offset_hmx = hex_align_up(offset, hmx_align);
        size_hmx   = hmx_queue_sizeof(HMX_QUEUE_CAPACITY, HMX_QUEUE_স্তূপ_SIZE);
        offset = offset_hmx + size_hmx;
    }

    size_t footprint = hex_align_up(offset, 128);

    শূন্য * block = memalign(4096, footprint);
    যদি (!block) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_queue);
        ঘূৰাই AEE_ENOMEMORY;
    }
    memset(block, 0, footprint);

    h->ctx = (সংরচনা htp_context *) block;
    সংরচনা htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_queue   = dsp_queue;

    err = vtcm_alloc(ctx);
    যদি (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        ঘূৰাই AEE_ENOMEMORY;
    }

    HAP_setFARFRuntimeLoggingParams(0xffff, রিক্ত, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, পরিমাণ(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        যদি ((err = HAP_power_set((শূন্য *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ঘূৰাই err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, পরিমাণ(request));

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

#যদি (__HEXAGON_ARCH__ >= 79)
        HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
#endif
        যদি ((err = HAP_power_set((শূন্য *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ঘূৰাই err;
        }

        memset(&request, 0, পরিমাণ(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        যদি ((err = HAP_power_set((শূন্য *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ঘূৰাই err;
        }
    }

#যদি __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, পরিমাণ(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_power     = TRUE;
        request.hmx_v2.power_up      = TRUE;
        request.hmx_v2.set_clock     = TRUE;
        request.hmx_v2.target_corner = pwr->hmx_target_corner;
        request.hmx_v2.min_corner    = pwr->hmx_min_corner;
        request.hmx_v2.max_corner    = pwr->hmx_max_corner;
        request.hmx_v2.perf_mode     = pwr->hmx_perf_mode;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((শূন্য *) ctx, &request);
        যদি (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            ঘূৰাই err;
        }
    }
#অন্যথা
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, পরিমাণ(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((শূন্য *) ctx, &request);
        যদি (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            ঘূৰাই err;
        }
    }
#endif

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_queue   = রিক্ত;
    যদি (n_hmx) {
        শূন্য * hmx_ptr = (শূন্য *) ((uintptr_t) block + offset_hmx);
        ctx->hmx_queue = hmx_queue_init(hmx_ptr, HMX_QUEUE_CAPACITY, HMX_QUEUE_স্তূপ_SIZE, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(উপৰেৰ_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_threads = n_hvx;
    ctx->n_threads_div = init_fastdiv_মানসমূহ(ctx->n_threads);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) block + offset_dma);
    size_t size_dma_q = dma_queue_sizeof(256);

    বাবে (পূর্ণাঙ্ক i = 0; i < ctx->n_threads; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_up((uintptr_t) dma_ptr_curr, dma_queue_alignof());
        ctx->dma[i] = dma_queue_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += size_dma_q;
    }

    ctx->ddr_spad_size = 512 * 1024; // 512 KB
    ctx->ddr_spad_base = memalign(128, ctx->ddr_spad_size);

    শূন্য * wq_ptr = (শূন্য *) ((uintptr_t) block + offset_wq);
    ctx->work_queue = work_queue_init(wq_ptr, n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_স্তূপ_SIZE);

    ctx->main_স্তূপ = রিক্ত;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    যদি (!use_callbacks) {
        // Start main compute thread
        ctx->main_স্তূপ = (শূন্য *) ((uintptr_t) block + offset_main_স্তূপ);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_স্তূপ_addr(&attr, ctx->main_স্তূপ);
        qurt_thread_attr_set_স্তূপ_size(&attr, size_main_স্তূপ);
        qurt_thread_attr_set_priority(&attr, main_prio);
        qurt_thread_attr_set_name(&attr, "htp-main");

        পূর্ণাঙ্ক err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        যদি (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            ঘূৰাই AEE_ENOMEMORY;
        }
    }

    FARF(উপৰেৰ_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_size, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    ঘূৰাই AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    সংরচনা htp_handle * h = (সংরচনা htp_handle *) handle;
    যদি (!h || !h->ctx) {
        ঘূৰাই AEE_EBADPARM;
    }
    সংরচনা htp_context * ctx = h->ctx;

    যদি (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        পূর্ণাঙ্ক status;
        (শূন্য) qurt_thread_join(ctx->main_thread, &status);
        ctx->main_thread = 0;
    }

    পূর্ণাঙ্ক err = dspqueue_close(ctx->dsp_queue); ctx->dsp_queue = রিক্ত;
    যদি (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (unsigned) err);
        ঘূৰাই err;
    }

    work_queue_free(ctx->work_queue);

    বাবে (পূর্ণাঙ্ক i = 0; i < ctx->n_threads; i++) {
        dma_queue_free(ctx->dma[i]);
    }

    যদি (ctx->hmx_queue) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = রিক্ত;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    যদি (ctx->ddr_spad_base) {
        মুক্ত(ctx->ddr_spad_base);
        ctx->ddr_spad_base = রিক্ত;
        ctx->ddr_spad_size = 0;
    }

    মুক্ত(ctx);
    h->ctx = রিক্ত;

    ঘূৰাই AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_threads, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_size) {
    (শূন্য)handle;
    যদি (!n_threads || !n_hvx || !n_hmx || !vtcm_size) {
        ঘূৰাই AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    যদি (n_hvx_val > hw_threads.max_hthreads) {
        n_hvx_val = hw_threads.max_hthreads;
    }
    যদি (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_threads = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_size = vtcm_query_size();

    ঘূৰাই AEE_SUCCESS;
}

স্থির শূন্য htp_error_callback(dspqueue_t queue, পূর্ণাঙ্ক error, শূন্য * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (unsigned) error);
}

সংরচনা profile_data {
    uint64_t usecs;
    uint64_t cycles_start;
    uint64_t cycles_stop;
    uint32_t pmu_counters[HEX_অঙ্ক_PMU_COUNTERS];
};

স্থির অন্তঃস্থ শূন্য profile_start(uint32_t mode, সংরচনা profile_data * d) {
    ৱিকল্প (mode) {
        পক্ষ HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        পক্ষ HTP_PROF_BASIC:
        পক্ষ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_গুনতি();
            d->cycles_start = hex_get_cycles();
            ভঙ্গ;
        স্ৱভাৱ:
            ভঙ্গ;
    }
}

স্থির অন্তঃস্থ শূন্য profile_stop(uint32_t mode, সংরচনা profile_data * d) {
    uint32_t pmu_counters[HEX_অঙ্ক_PMU_COUNTERS];
    ৱিকল্প (mode) {
        পক্ষ HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            বাবে (পূর্ণাঙ্ক i = 0; i < HEX_অঙ্ক_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        পক্ষ HTP_PROF_BASIC:
        পক্ষ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_গুনতি_to_us(HAP_perf_get_qtimer_গুনতি() - d->usecs);
            d->cycles_stop = hex_get_cycles();
            ভঙ্গ;
        স্ৱভাৱ:
            ভঙ্গ;
    }
}

স্থির পূর্ণাঙ্ক op_fence(সংরচনা htp_ops_context * octx) {
    সংরচনা htp_context *ctx = octx->ctx;
    সংরচনা htp_thread_trace * tr = &ctx->trace[0];
    ধ্রুৱ uint32_t seq  = (uint32_t) octx->op_params[0];
    ধ্রুৱ uint32_t mode = (uint32_t) octx->op_params[1];

    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ধ্রুৱ সংরচনা htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;

    যদি (mode == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_group_barrier(octx);

        যদি (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->status);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(উপৰেৰ_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        ঘূৰাই octx->status;
    }

    পূর্ণাঙ্ক status = HTP_STATUS_OK;
    uint64_t spins = 0;
    যেতিয়ালৈ (1) {
        uint32_t sync_seq;
        uint32_t sync_status;
        htp_fence_read(sync_fence, &sync_seq, &sync_status);
        যদি ((int32_t)(sync_seq - seq) >= 0) {
            যদি (sync_status > HTP_STATUS_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_status, sync_fence, seq);
                status = sync_status;
            }
            ভঙ্গ;
        }
        যদি (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            status = HTP_STATUS_INTERNAL_ERR;
            ভঙ্গ;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(উপৰেৰ_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    ঘূৰাই status;
}

স্থির পূর্ণাঙ্ক op_mdev_group(সংরচনা htp_ops_context * octx) {
    সংরচনা htp_context * ctx = octx->ctx;
    ধ্রুৱ সংরচনা htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.গুনতি = (uint16_t) sync->ne[1];
    যদি (ctx->mdev.গুনতি > 1) {
        ctx->mdev.গুনতি_div = init_fastdiv_মানসমূহ(ctx->mdev.গুনতি);
        ctx->mdev.fence_base = (uint8_t *) sync->data;
    }
    ঘূৰাই HTP_STATUS_OK;
}

স্থির পূর্ণাঙ্ক execute_op(সংরচনা htp_ops_context * octx) {
    ৱিকল্প (octx->op) {
        পক্ষ HTP_OP_MDEV_GROUP:
            ঘূৰাই op_mdev_group(octx);

        পক্ষ HTP_OP_FENCE:
            ঘূৰাই op_fence(octx);

        পক্ষ HTP_OP_ALLREDUCE:
        পক্ষ HTP_OP_ALLREDUCE_ADD:
            ঘূৰাই op_allreduce(octx);

        পক্ষ HTP_OP_MUL_MAT:
        পক্ষ HTP_OP_MUL_MAT_ADD:
            ঘূৰাই op_matmul(octx);

        পক্ষ HTP_OP_MUL_MAT_ID:
            ঘূৰাই op_matmul_id(octx);

        পক্ষ HTP_OP_MUL_MAT_ID_NX:
            ঘূৰাই op_matmul_id_nx(octx);

        পক্ষ HTP_OP_MUL_MAT_NX:
            ঘূৰাই op_matmul_nx(octx);

        পক্ষ HTP_OP_MUL:
        পক্ষ HTP_OP_ADD:
        পক্ষ HTP_OP_SUB:
        পক্ষ HTP_OP_DIV:
        পক্ষ HTP_OP_ADD_ID:
            ঘূৰাই op_দ্বিমিক(octx);

        পক্ষ HTP_OP_NORM:
        পক্ষ HTP_OP_RMS_NORM:
        পক্ষ HTP_OP_RMS_NORM_MUL:
        পক্ষ HTP_OP_SCALE:
        পক্ষ HTP_OP_CLAMP:
        পক্ষ HTP_OP_LEAKY_RELU:
        পক্ষ HTP_OP_SQR:
        পক্ষ HTP_OP_SQRT:
        পক্ষ HTP_OP_UNARY_SOFTPLUS:
        পক্ষ HTP_OP_UNARY_SIGMOID:
        পক্ষ HTP_OP_UNARY_SILU:
        পক্ষ HTP_OP_UNARY_GELU:
        পক্ষ HTP_OP_UNARY_GELU_QUICK:
        পক্ষ HTP_OP_UNARY_GELU_ERF:
        পক্ষ HTP_OP_UNARY_NEG:
        পক্ষ HTP_OP_UNARY_EXP:
        পক্ষ HTP_OP_UNARY_TANH:
        পক্ষ HTP_OP_UNARY_ABS:
        পক্ষ HTP_OP_UNARY_LOG:
        পক্ষ HTP_OP_UNARY_RELU:
        পক্ষ HTP_OP_UNARY_STEP:
        পক্ষ HTP_OP_L2_NORM:
        পক্ষ HTP_OP_HADAMARD:
            ঘূৰাই op_unary(octx);

        পক্ষ HTP_OP_GLU_SWIGLU:
        পক্ষ HTP_OP_GLU_SWIGLU_OAI:
        পক্ষ HTP_OP_GLU_SWIGLU_CLAMP:
        পক্ষ HTP_OP_GLU_GEGLU:
        পক্ষ HTP_OP_GLU_GEGLU_QUICK:
        পক্ষ HTP_OP_GLU_GEGLU_ERF:
            ঘূৰাই op_activations(octx);

        পক্ষ HTP_OP_SOFTMAX:
            ঘূৰাই op_softmax(octx);

        পক্ষ HTP_OP_ROPE:
            ঘূৰাই op_rope(octx);

        পক্ষ HTP_OP_FLASH_ATTN_EXT:
            ঘূৰাই op_flash_attn_ext(octx);

        পক্ষ HTP_OP_SET_ROWS:
            ঘূৰাই op_set_rows(octx);

        পক্ষ HTP_OP_GET_ROWS:
            ঘূৰাই op_get_rows(octx);

        পক্ষ HTP_OP_SUM:
            ঘূৰাই op_sum(octx);

        পক্ষ HTP_OP_SUM_ROWS:
            ঘূৰাই op_sum_rows(octx);

        পক্ষ HTP_OP_CPY:
        পক্ষ HTP_OP_CPY_FENCE:
            ঘূৰাই op_cpy(octx);

        পক্ষ HTP_OP_REPEAT:
            ঘূৰাই op_repeat(octx);

        পক্ষ HTP_OP_ARGSORT:
            ঘূৰাই op_argsort(octx);

        পক্ষ HTP_OP_শীৰ্ষ_K:
            ঘূৰাই op_শীৰ্ষ_k(octx);

        পক্ষ HTP_OP_ARGMAX:
            ঘূৰাই op_argmax(octx);

        পক্ষ HTP_OP_SSM_CONV:
            ঘূৰাই op_ssm_conv(octx);

        পক্ষ HTP_OP_CUMSUM:
            ঘূৰাই op_cumsum(octx);

        পক্ষ HTP_OP_FILL:
            ঘূৰাই op_fill(octx);

        পক্ষ HTP_OP_DIAG:
            ঘূৰাই op_diag(octx);

        পক্ষ HTP_OP_SOLVE_TRI:
            ঘূৰাই op_solve_tri(octx);

        পক্ষ HTP_OP_PAD:
            ঘূৰাই op_pad(octx);

        পক্ষ HTP_OP_IM2COL:
            ঘূৰাই op_im2col(octx);

        পক্ষ HTP_OP_IM2COL_3D:
            ঘূৰাই op_im2col_3d(octx);

        পক্ষ HTP_OP_ROLL:
            ঘূৰাই op_roll(octx);

        পক্ষ HTP_OP_POOL_2D:
            ঘূৰাই op_pool_2d(octx);

        পক্ষ HTP_OP_POOL_1D:
            ঘূৰাই op_pool_1d(octx);

        পক্ষ HTP_OP_CONCAT:
            ঘূৰাই op_concat(octx);

        পক্ষ HTP_OP_GATED_DELTA_NET:
            ঘূৰাই op_gated_delta_net(octx);

        পক্ষ HTP_OP_TRI:
            ঘূৰাই op_unary(octx);

        পক্ষ HTP_OP_INVALID:
            ভঙ্গ;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    ঘূৰাই HTP_STATUS_NO_SUPPORT;
}

স্থির অন্তঃস্থ bool reuse_buf(সংরচনা htp_context *ctx, uint32_t *m_reuse, সংরচনা htp_buf_desc *b) {
    b->base = রিক্ত;

    বাবে (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        সংরচনা htp_mmap *m = ctx->mmap + i;
        যদি (m->size && m->fd == b->fd && m->flags == b->flags) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            ঘূৰাই true;
        }
    }

    ঘূৰাই false;
}

স্থির অন্তঃস্থ শূন্য drop_mmap(সংরচনা htp_context *ctx, সংরচনা htp_mmap *m) {
    যদি (m->size) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (unsigned long long) m->base, (unsigned long long) m->size);
        htp_munmap(m->base, m->size, m->flags);
        m->size = 0;
        m->base = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

স্থির অন্তঃস্থ bool mmap_buf(সংরচনা htp_context *ctx, সংরচনা htp_buf_desc *b) {
    যদি (b->base) ঘূৰাই true; // already mapped

    // find unused mapping
    বাবে (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        সংরচনা htp_mmap *m = &ctx->mmap[i];
        যদি (!m->size) {
            uint64_t va = htp_mmap(b->fd, b->size, b->flags);
            যদি (va == 0) {
                FARF(উপৰেৰ_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (unsigned long long) b->size);
                ঘূৰাই false;
            }

            m->base   = b->base = va;
            m->fd     = b->fd;
            m->size   = b->size;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (unsigned long long) m->base, (unsigned long long) m->size, m->flags);
            ঘূৰাই true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    ঘূৰাই false;
}

স্থির শূন্য prep_op_bufs(সংরচনা htp_context *ctx, সংরচনা htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    বাবে (uint32_t i=0; i < n_bufs; i++) {
        সংরচনা htp_buf_desc *b = bufs + i;
        যদি (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } অন্যথা যদি (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->size;
        }
        FARF(উপৰেৰ_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    যদি (b_reuse == n_bufs) ঘূৰাই; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    বাবে (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        যদি (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].size;
        }
    }

    FARF(উপৰেৰ_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    যদি ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        বাবে (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            যদি (!used && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    বাবে (uint32_t i=0; i < n_bufs; i++) {
        সংরচনা htp_buf_desc *b = bufs + i;
        যদি (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            ভঙ্গ;
        }
        FARF(উপৰেৰ_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    যদি (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(উপৰেৰ_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        বাবে (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            যদি (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        বাবে (uint32_t i=0; i < n_bufs; i++) {
            সংরচনা htp_buf_desc *b = bufs + i;
            যদি (!(b->flags & HTP_BUF_EXTENDED)) {
                b->base = 0;
            }
            যদি (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (unsigned long long) b->size);
                abort();
            }
            FARF(উপৰেৰ_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
        }
    }
}

স্থির শূন্য prep_tensor(সংরচনা htp_context *ctx, সংরচনা htp_buf_desc *bufs, সংরচনা htp_tensor *tens, uint32_t idx, সংরচনা htp_tensor *t) {
    uint64_t offset = t->data;
    uint32_t bi     = t->bi;

    t->data  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(উপৰেৰ_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (unsigned long long) offset, t->size, (unsigned long long) t->data,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

স্থির শূন্য prep_tensors(সংরচনা htp_context *ctx, সংরচনা htp_buf_desc *bufs, সংরচনা htp_tensor *tens, uint32_t n_tens) {
    বাবে (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

স্থির শূন্য mdev_group_init(সংরচনা htp_context * ctx, ধ্রুৱ সংরচনা htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, পরিমাণ(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

স্থির পূর্ণাঙ্ক proc_op_req(সংরচনা htp_ops_context * octx, সংরচনা htp_buf_desc * bufs, uint32_t n_bufs,
                       সংরচনা htp_tensor * tens, uint32_t idx, সংরচনা htp_op_desc * op) {
    memcpy(octx->op_params,     op->params, পরিমাণ(octx->op_params));
    memcpy(octx->kernel_params, op->kernel_params, পরিমাণ(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_threads     = octx->ctx->n_threads;
    octx->n_threads_div = octx->ctx->n_threads_div;

    FARF(উপৰেৰ_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    বাবে (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        যদি (src_idx == 0xffff) {
            octx->src[i]     = রিক্ত;
            অনুৱর্ত;
        }

        সংরচনা htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(উপৰেৰ_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (unsigned long long) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    বাবে (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        যদি (dst_idx == 0xffff) {
            octx->dsts[i]    = রিক্ত;
            অনুৱর্ত;
        }
        সংরচনা htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(উপৰেৰ_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (unsigned long long) dst->data, dst->size,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_group_barrier(octx);

    পূর্ণাঙ্ক status = execute_op(octx);

    htp_ops_context_set_status(octx, status);

    octx->src0_spad.src = রিক্ত;
    octx->src1_spad.src = রিক্ত;
    octx->src2_spad.src = রিক্ত;
    octx->src3_spad.src = রিক্ত;
    octx->dst_spad.src  = রিক্ত;

    ঘূৰাই octx->status;
}

স্থির শূন্য process_opbatch(সংরচনা htp_context * ctx, ধ্রুৱ সংরচনা htp_opbatch_req * req, ধ্রুৱ সংরচনা dspqueue_buffer * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    পূর্ণাঙ্ক err;

    ধ্রুৱ uint32_t n_bufs = req->n_bufs;
    ধ্রুৱ uint32_t n_tens = req->n_tensors;
    ধ্রুৱ uint32_t n_ops  = req->n_ops;

    ধ্রুৱ uint32_t b_size = পরিমাণ(সংরচনা htp_buf_desc)  * n_bufs;
    ধ্রুৱ uint32_t t_size = পরিমাণ(সংরচনা htp_tensor)    * n_tens;
    ধ্রুৱ uint32_t o_size = পরিমাণ(সংরচনা htp_op_desc)   * n_ops;
    ধ্রুৱ uint32_t p_size = পরিমাণ(সংরচনা htp_prof_desc) * n_ops;
    ধ্রুৱ uint32_t tr_size = (HTP_MAX_NTHREADS + 1) * req->n_traces * পরিমাণ(সংরচনা htp_trace_desc);

    যদি (dbuf->size < b_size + t_size + o_size + p_size + tr_size) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->size, b_size + t_size + o_size + p_size + tr_size);
        ঘূৰাই;
    }

    FARF(উপৰেৰ_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (unsigned long long) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->size, b_size, t_size, o_size);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    সংরচনা htp_buf_desc* bufs = (সংরচনা htp_buf_desc*)  m_ptr; m_ptr += b_size;
    সংরচনা htp_tensor*   tens = (সংরচনা htp_tensor*)    m_ptr; m_ptr += t_size;
    সংরচনা htp_op_desc*   ops = (সংরচনা htp_op_desc*)   m_ptr; m_ptr += o_size;
    সংরচনা htp_prof_desc* pds = (সংরচনা htp_prof_desc*) m_ptr;

    সংরচনা profile_data batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, পরিমাণ(ctx->trace));
    যদি (ctx->profiler == HTP_PROF_TRACE) {
        সংরচনা htp_trace_desc * trace_events = (সংরচনা htp_trace_desc *) (m_ptr + p_size);
        বাবে (পূর্ণাঙ্ক t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, পরিমাণ(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    সংরচনা htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, পরিমাণ(*octx));
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
    octx->ctx           = ctx;

    mdev_group_init(ctx, req);

    work_queue_wakeup(ctx->work_queue);
    যদি (ctx->hmx_queue) {
        hmx_queue_wakeup(ctx->hmx_queue);
    }

    পূর্ণাঙ্ক op_status = HTP_STATUS_OK;
    octx->status  = HTP_STATUS_OK;
    বাবে (uint32_t i = 0; i < n_ops; i++) {
        সংরচনা profile_data prof;

        profile_start(ctx->profiler, &prof);

        op_status = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_set_status(octx, op_status);

        যদি (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].cycles_start = prof.cycles_start;
            pds[i].cycles_stop = prof.cycles_stop;
            বাবে (পূর্ণাঙ্ক j = 0; j < HEX_অঙ্ক_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    যদি (ctx->hmx_queue) {
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

    সংরচনা htp_opbatch_rsp rsp;
    memset(&rsp, 0, পরিমাণ(rsp));
    rsp.seq          = req->seq;
    rsp.status       = octx->status;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.cycles_start = batch_prof.cycles_start;
    rsp.cycles_stop  = batch_prof.cycles_stop;

    যদি (ctx->profiler == HTP_PROF_TRACE) {
        বাবে (পূর্ণাঙ্ক t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].গুনতি;
        }
    }

    সংরচনা dspqueue_buffer write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(queue, 0, 1, &write_dbuf, পরিমাণ(rsp), (ধ্রুৱ uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    যদি (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
    }
}

#define DSPQUEUE_READ_TIMEOUT_USEC 5000
#define DSPQUEUE_POLL_TIMEOUT_USEC 100
#define DSPQUEUE_POLL_গুনতি        100

স্থির শূন্য process_ops(সংরচনা htp_context * ctx) {
    dspqueue_t queue = ctx->dsp_queue;
    পূর্ণাঙ্ক err;

    uint32_t poll_গুনতি = DSPQUEUE_POLL_গুনতি;

    vtcm_acquire(ctx);

    যেতিয়ালৈ (!atomic_load(&ctx->vtcm_needs_release) && !atomic_load(&ctx->killed)) {
        সংরচনা htp_opbatch_req req;
        uint32_t r_size = পরিমাণ(req);

        সংরচনা dspqueue_buffer dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_size, &r_size, (uint8_t *) &req);
        যদি (err == AEE_EWOULDBLOCK) {
            যদি (--poll_গুনতি) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                অনুৱর্ত;
            }
            ভঙ্গ;
        }

        যদি (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            ভঙ্গ;
        }

        যদি (r_size < পরিমাণ(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_size, n_dbufs);
            অনুৱর্ত;
        }

        // Reset poll count for valid requests
        poll_গুনতি = DSPQUEUE_POLL_গুনতি;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

স্থির শূন্য htp_packet_callback(dspqueue_t queue, পূর্ণাঙ্ক error, শূন্য * context) {
    (শূন্য) queue;
    (শূন্য) error;
    সংরচনা htp_handle * h = (সংরচনা htp_handle *) context;
    যদি (h && h->ctx) {
        process_ops(h->ctx);
    }
}

স্থির শূন্য htp_main_thread(শূন্য * context) {
    সংরচনা htp_context * ctx = (সংরচনা htp_context *) context;

    FARF(উপৰেৰ_2, "htp-main-thread: started");

    যেতিয়ালৈ (!atomic_load(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t অঙ্ক_buffers = 0;
        uint32_t message_length = 0;

        পূর্ণাঙ্ক err = dspqueue_peek(ctx->dsp_queue, &flags, &অঙ্ক_buffers, &message_length, 50000);
        যদি (err == 0) {
            process_ops(ctx);
        } অন্যথা যদি (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            অনুৱর্ত;
        } অন্যথা {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (unsigned) err);
            ভঙ্গ;
        }
    }

    FARF(উপৰেৰ_2, "htp-main-thread: stopped");
}
