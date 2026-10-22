#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <AEEStdErr.h>
#include <dspqueue.h>
#include <HAP_compute_res.h>
#include <HAP_etm_config.h>
#include <HAP_mem.h>
#include <HAP_power.h>
#include <HAP_ps.h>
#include <HAP_dcvs.h>
#include <qurt.h>
#include <qurt_thread.h>
#include <qurt_memory.h>
#include <remote.h>
#include <string.h>
#include <stdatomic.h>

#include "hex-utils.h"
#include "dma-queue.h"
#include "hmx-queue.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-bitmap.h"
#include "htp-ctx.h"
#include "htp-dcache.h"
#include "htp-ops.h"
#include "htp_iface.h"
#include "work-queue.h"
#include "hex-profile.h"
#include "allreduce-ops.h"
#include "htp-fence.h"

#define HMX_QUEUE_CAPACITY     128
#define HMX_QUEUE_STACK_SIZE   16384
#define WORK_QUEUE_CAPACITY    16
#define WORK_QUEUE_STACK_SIZE  16384
#define MAIN_THREAD_STACK_SIZE 32768

_Static_assert(WORK_QUEUE_MAX_N_THREADS >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

sa_nracanaa htp_handle {
    sa_nracanaa htp_context * ctx;
};

static anta_hstha uint64_t htp_mmap(uint32_t fd, uint64_t size, uint32_t flags) {
#အကယ်၍ __HVX_ARCH__ > 79
    အကယ်၍ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, parimaa_na(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = size;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        အကယ်၍ (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (unsigned long long) size);
            ပြန်ပေး 0;
        }

        ပြန်ပေး payload.mmap.dsp_va;
    }
#မဟုတ်ပါက
    အကယ်၍ (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        ပြန်ပေး 0;
    }
#endif

    အကယ်၍ (size > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (unsigned long long) size);
        ပြန်ပေး 0;
    }

    void * va = (void *)-1;
    အတွက် (ကိန်းပြည့် retry = 0; retry < 2; retry++) {
#အကယ်၍ __HVX_ARCH__ > 73
        va = HAP_mmap2(NULL, (size_t) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#မဟုတ်ပါက
        အကယ်၍ (size > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (unsigned long long) size);
            abort();
        }
        va = HAP_mmap(NULL, (ကိန်းပြည့်) size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
        အကယ်၍ (va != (void *)-1 && va != NULL) {
            ပြန်ပေး (uint64_t) (uintptr_t) va;
        }
        အကယ်၍ (retry == 0) {
            FARF(HIGH, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (unsigned long long) size);
        }
    }
    ပြန်ပေး 0;
}

static anta_hstha void htp_munmap(uint64_t va, uint64_t size, uint32_t flags) {
#အကယ်၍ __HVX_ARCH__ > 79
    အကယ်၍ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, parimaa_na(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = size;
        HAP_mem_request(&payload);
        ပြန်ပေး;
    }
#endif

#အကယ်၍ __HVX_ARCH__ > 73
    HAP_munmap2((void *) (uintptr_t) va, (size_t) size);
#မဟုတ်ပါက
    HAP_munmap((void *) (uintptr_t) va, (ကိန်းပြည့်) size);
#endif
}

AEEResult htp_iface_open(const var_na * uri, remote_handle64 * handle) {
    (void) uri;
    sa_nracanaa htp_handle * h = calloc(1, parimaa_na(*h));
    အကယ်၍ (h == NULL) {
        ပြန်ပေး AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    ပြန်ပေး AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    sa_nracanaa htp_handle * h = (sa_nracanaa htp_handle *) handle;
    အကယ်၍ (!h) {
        ပြန်ပေး AEE_EBADPARM;
    }

    ကိန်းပြည့် err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    အကယ်၍ (err) {
        အကယ်၍ (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } မဟုတ်ပါက {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    ပြန်ပေး err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, const htp_iface_pmu_conf* pmu_conf) {
    sa_nracanaa htp_handle * h = (sa_nracanaa htp_handle *) handle;
    အကယ်၍ (!h || !h->ctx) {
        ပြန်ပေး AEE_EBADPARM;
    }
    sa_nracanaa htp_context * ctx = h->ctx;

    အကယ်၍ (mode == HTP_PROF_PMU) {
        const uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        အတွက် (; i < HEX_NUM_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        အတွက် (i = 0; i < HEX_NUM_PMU_COUNTERS; i++) {
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

    ပြန်ပေး AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    sa_nracanaa htp_handle * h = (sa_nracanaa htp_handle *) handle;
    အကယ်၍ (!h) {
        ပြန်ပေး AEE_EBADPARM;
    }

    sa_nracanaa htp_context * ctx = h->ctx;
    အကယ်၍ (ctx) {
        အကယ်၍ (ctx->dsp_queue) {
            FARF(ERROR, "Closing handle with queue still open");
            ပြန်ပေး AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        အတွက် (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            အကယ်၍ (ctx->mmap[i].size) {
                htp_munmap(ctx->mmap[i].base, ctx->mmap[i].size, ctx->mmap[i].flags);
                ctx->mmap[i].size = 0;
                ctx->mmap[i].base = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        အကယ်၍ (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        အကယ်၍ (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        free(ctx);
        h->ctx = NULL;
    }

    free(h);
    ပြန်ပေး AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t size) {
    sa_nracanaa htp_handle * h = (sa_nracanaa htp_handle *) handle;
    အကယ်၍ (!h || !h->ctx) {
        ပြန်ပေး AEE_EBADPARM;
    }
    sa_nracanaa htp_context * ctx = h->ctx;

    // See if we already have this mapping
    အတွက် (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        sa_nracanaa htp_mmap *m = &ctx->mmap[i];
        အကယ်၍ (m->fd == fd) {
            ပြန်ပေး AEE_SUCCESS;
        }
    }

    // Add new mapping
    အတွက် (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        sa_nracanaa htp_mmap *m = &ctx->mmap[i];
        အကယ်၍ (!m->size) {
            FARF(HIGH, "mmap : fd %u size %llu", fd, (unsigned long long) size);
            uint64_t va = htp_mmap(fd, size, 0);
            အကယ်၍ (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (unsigned long long) size);
                ပြန်ပေး AEE_EFAILED;
            }

            m->base   = va;
            m->fd     = fd;
            m->size   = size;
            m->flags  = 0;

            ပြန်ပေး AEE_SUCCESS;
        }
    }

    ပြန်ပေး AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    sa_nracanaa htp_handle * h = (sa_nracanaa htp_handle *) handle;
    အကယ်၍ (!h || !h->ctx) {
        ပြန်ပေး AEE_EBADPARM;
    }
    sa_nracanaa htp_context * ctx = h->ctx;

    အတွက် (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        sa_nracanaa htp_mmap *m = &ctx->mmap[i];
        အကယ်၍ (fd < 0 || m->fd == fd) {
            FARF(HIGH, "unmmap : base 0x%llx fd %u size %llu", (unsigned long long) m->base, m->fd, (unsigned long long) m->size);
            htp_munmap(m->base, m->size, m->flags);
            m->size   = 0;
            m->base   = NULL;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    ပြန်ပေး AEE_SUCCESS;
}

static void vtcm_acquire(sa_nracanaa htp_context * ctx) {
    အကယ်၍ (!atomic_load(&ctx->vtcm_valid)) {
        ကိန်းပြည့် err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        အကယ်၍ (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (unsigned)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

static void vtcm_release(sa_nracanaa htp_context * ctx) {
    အကယ်၍ (atomic_load(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

static ကိန်းပြည့် vtcm_release_callback(unsigned ကိန်းပြည့် rctx, void * state) {
    sa_nracanaa htp_context * ctx = (sa_nracanaa htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    ပြန်ပေး 0;
}

static unsigned ကိန်းပြည့် vtcm_query_size(void) {
    unsigned ကိန်းပြည့် vtcm_size = 8 * 1024 * 1024;  // 8MB default
    unsigned ကိန်းပြည့် vtcm_avail;
    compute_res_vtcm_page_t total_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_size, &total_layout, &vtcm_avail, &avail_layout);
    ပြန်ပေး vtcm_size;
}

static ကိန်းပြည့် vtcm_alloc(sa_nracanaa htp_context * ctx) {
    unsigned ကိန်းပြည့် vtcm_size = vtcm_query_size();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (void *) ctx);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    အကယ်၍ (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_size);
        ပြန်ပေး AEE_ENOMEMORY;
    }

    void * vtcm_ptr;
    အကယ်၍ (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_size);
        ပြန်ပေး AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size          = vtcm_size;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    ပြန်ပေး 0;
}

static void vtcm_free(sa_nracanaa htp_context * ctx) {
    အကယ်၍ (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

static void htp_main_thread(void * context);
static void htp_packet_callback(dspqueue_t queue, ကိန်းပြည့် error, void * context);
static void htp_error_callback(dspqueue_t queue, ကိန်းပြည့် error, void * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
sa_nracanaa htp_power_profile {
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
        .hmx_perf_mode      = HAP_CLK_PERF_HIGH, \
    }

static const sa_nracanaa htp_power_profile htp_power_profiles[HTP_POWER_MODE_COUNT] = {
    [HTP_POWER_MODE_LOW_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_MODE_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_MODE_HIGH_POWER_SAVER] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_MODE_LOW_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_MODE_BALANCED] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_MODE_HIGH_PERFORMANCE] =
        HTP_POWER_PROFILE_ENTRY(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_MODE_SUSTAINED_HIGH_PERFORMANCE] =
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
        .hmx_perf_mode      = HAP_CLK_PERF_HIGH,
    },
};

#undef HTP_POWER_PROFILE_ENTRY

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_mode) {
    sa_nracanaa htp_handle * h = (sa_nracanaa htp_handle *) handle;
    အကယ်၍ (!h) {
        ပြန်ပေး AEE_EBADPARM;
    }

    အကယ်၍ (power_mode >= HTP_POWER_MODE_COUNT) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (unsigned) power_mode);
        power_mode = HTP_POWER_MODE_BURST;
    }
    const sa_nracanaa htp_power_profile * pwr = &htp_power_profiles[power_mode];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (unsigned) power_mode, (ကိန်းပြည့်) pwr->dcvs_enable,
         (unsigned) pwr->bus_min_corner, (unsigned) pwr->bus_max_corner, (unsigned) pwr->bus_target_corner,
         (unsigned) pwr->core_min_corner, (unsigned) pwr->core_max_corner, (unsigned) pwr->core_target_corner,
         (ကိန်းပြည့်) pwr->sleep_disable,
         (unsigned) pwr->hmx_min_corner, (unsigned) pwr->hmx_max_corner, (unsigned) pwr->hmx_target_corner);

    အကယ်၍ (h->ctx) {
        FARF(ERROR, "Queue already open");
        ပြန်ပေး AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    ကိန်းပြည့် fastrpc_tid  = qurt_thread_get_id();
    ကိန်းပြည့် fastrpc_prio = qurt_thread_get_priority(fastrpc_tid);
    ကိန်းပြည့် main_prio    = fastrpc_prio - 10;
    အကယ်၍ (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_queue = NULL;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    ကိန်းပြည့် err = dspqueue_import(dsp_queue_id, NULL, NULL, (void *) h, &dsp_queue);
    အကယ်၍ (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(HIGH, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_queue_id, htp_packet_callback, htp_error_callback, (void *) h, &dsp_queue);
    }

    အကယ်၍ (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        ပြန်ပေး err;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    အကယ်၍ (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    အကယ်၍ (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    အကယ်၍ (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t offset = parimaa_na(sa_nracanaa htp_context);

    // 2. main_stack
    size_t offset_main_stack = 0;
    size_t size_main_stack   = 0;
    အကယ်၍ (!use_callbacks) {
        offset_main_stack = hex_align_up(offset, 4096);
        size_main_stack   = MAIN_THREAD_STACK_SIZE;
        offset = offset_main_stack + size_main_stack;
    }

    // 3. work_queue
    size_t wq_align  = work_queue_alignof();
    size_t offset_wq = hex_align_up(offset, wq_align);
    size_t size_wq   = work_queue_sizeof(n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_STACK_SIZE);
    offset = offset_wq + size_wq;

    // 4. dma_queue
    size_t dma_align = dma_queue_alignof();
    size_t offset_dma = hex_align_up(offset, dma_align);
    size_t size_dma = 0;
    အတွက် (uint32_t i = 0; i < n_hvx; i++) {
        size_dma  = hex_align_up(size_dma, dma_queue_alignof());
        size_dma += dma_queue_sizeof(256);
    }
    offset = offset_dma + size_dma;

    // 5. hmx_queue
    size_t offset_hmx = 0;
    size_t size_hmx = 0;
    အကယ်၍ (n_hmx) {
        size_t hmx_align = hmx_queue_alignof();
        offset_hmx = hex_align_up(offset, hmx_align);
        size_hmx   = hmx_queue_sizeof(HMX_QUEUE_CAPACITY, HMX_QUEUE_STACK_SIZE);
        offset = offset_hmx + size_hmx;
    }

    size_t footprint = hex_align_up(offset, 128);

    void * block = memalign(4096, footprint);
    အကယ်၍ (!block) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_queue);
        ပြန်ပေး AEE_ENOMEMORY;
    }
    memset(block, 0, footprint);

    h->ctx = (sa_nracanaa htp_context *) block;
    sa_nracanaa htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_queue   = dsp_queue;

    err = vtcm_alloc(ctx);
    အကယ်၍ (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        ပြန်ပေး AEE_ENOMEMORY;
    }

    HAP_setFARFRuntimeLoggingParams(0xffff, NULL, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, parimaa_na(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        အကယ်၍ ((err = HAP_power_set((void *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ပြန်ပေး err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, parimaa_na(request));

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

#အကယ်၍ (__HEXAGON_ARCH__ >= 79)
        HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
#endif
        အကယ်၍ ((err = HAP_power_set((void *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ပြန်ပေး err;
        }

        memset(&request, 0, parimaa_na(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        အကယ်၍ ((err = HAP_power_set((void *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ပြန်ပေး err;
        }
    }

#အကယ်၍ __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, parimaa_na(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_power     = TRUE;
        request.hmx_v2.power_up      = TRUE;
        request.hmx_v2.set_clock     = TRUE;
        request.hmx_v2.target_corner = pwr->hmx_target_corner;
        request.hmx_v2.min_corner    = pwr->hmx_min_corner;
        request.hmx_v2.max_corner    = pwr->hmx_max_corner;
        request.hmx_v2.perf_mode     = pwr->hmx_perf_mode;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((void *) ctx, &request);
        အကယ်၍ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            ပြန်ပေး err;
        }
    }
#မဟုတ်ပါက
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, parimaa_na(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((void *) ctx, &request);
        အကယ်၍ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            ပြန်ပေး err;
        }
    }
#endif

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_queue   = NULL;
    အကယ်၍ (n_hmx) {
        void * hmx_ptr = (void *) ((uintptr_t) block + offset_hmx);
        ctx->hmx_queue = hmx_queue_init(hmx_ptr, HMX_QUEUE_CAPACITY, HMX_QUEUE_STACK_SIZE, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(HIGH, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_threads = n_hvx;
    ctx->n_threads_div = init_fastdiv_values(ctx->n_threads);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) block + offset_dma);
    size_t size_dma_q = dma_queue_sizeof(256);

    အတွက် (ကိန်းပြည့် i = 0; i < ctx->n_threads; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_up((uintptr_t) dma_ptr_curr, dma_queue_alignof());
        ctx->dma[i] = dma_queue_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += size_dma_q;
    }

    ctx->ddr_spad_size = 512 * 1024; // 512 KB
    ctx->ddr_spad_base = memalign(128, ctx->ddr_spad_size);

    void * wq_ptr = (void *) ((uintptr_t) block + offset_wq);
    ctx->work_queue = work_queue_init(wq_ptr, n_hvx, WORK_QUEUE_CAPACITY, WORK_QUEUE_STACK_SIZE);

    ctx->main_stack = NULL;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    အကယ်၍ (!use_callbacks) {
        // Start main compute thread
        ctx->main_stack = (void *) ((uintptr_t) block + offset_main_stack);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, ctx->main_stack);
        qurt_thread_attr_set_stack_size(&attr, size_main_stack);
        qurt_thread_attr_set_priority(&attr, main_prio);
        qurt_thread_attr_set_name(&attr, "htp-main");

        ကိန်းပြည့် err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        အကယ်၍ (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            ပြန်ပေး AEE_ENOMEMORY;
        }
    }

    FARF(HIGH, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_size, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    ပြန်ပေး AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    sa_nracanaa htp_handle * h = (sa_nracanaa htp_handle *) handle;
    အကယ်၍ (!h || !h->ctx) {
        ပြန်ပေး AEE_EBADPARM;
    }
    sa_nracanaa htp_context * ctx = h->ctx;

    အကယ်၍ (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        ကိန်းပြည့် status;
        (void) qurt_thread_join(ctx->main_thread, &status);
        ctx->main_thread = 0;
    }

    ကိန်းပြည့် err = dspqueue_close(ctx->dsp_queue); ctx->dsp_queue = NULL;
    အကယ်၍ (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (unsigned) err);
        ပြန်ပေး err;
    }

    work_queue_free(ctx->work_queue);

    အတွက် (ကိန်းပြည့် i = 0; i < ctx->n_threads; i++) {
        dma_queue_free(ctx->dma[i]);
    }

    အကယ်၍ (ctx->hmx_queue) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = NULL;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    အကယ်၍ (ctx->ddr_spad_base) {
        free(ctx->ddr_spad_base);
        ctx->ddr_spad_base = NULL;
        ctx->ddr_spad_size = 0;
    }

    free(ctx);
    h->ctx = NULL;

    ပြန်ပေး AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_threads, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_size) {
    (void)handle;
    အကယ်၍ (!n_threads || !n_hvx || !n_hmx || !vtcm_size) {
        ပြန်ပေး AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    အကယ်၍ (n_hvx_val > hw_threads.max_hthreads) {
        n_hvx_val = hw_threads.max_hthreads;
    }
    အကယ်၍ (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_threads = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_size = vtcm_query_size();

    ပြန်ပေး AEE_SUCCESS;
}

static void htp_error_callback(dspqueue_t queue, ကိန်းပြည့် error, void * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (unsigned) error);
}

sa_nracanaa profile_data {
    uint64_t usecs;
    uint64_t cycles_start;
    uint64_t cycles_stop;
    uint32_t pmu_counters[HEX_NUM_PMU_COUNTERS];
};

static anta_hstha void profile_start(uint32_t mode, sa_nracanaa profile_data * d) {
    switch (mode) {
        pak_sa HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        pak_sa HTP_PROF_BASIC:
        pak_sa HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_count();
            d->cycles_start = hex_get_cycles();
            bha_nga;
        default:
            bha_nga;
    }
}

static anta_hstha void profile_stop(uint32_t mode, sa_nracanaa profile_data * d) {
    uint32_t pmu_counters[HEX_NUM_PMU_COUNTERS];
    switch (mode) {
        pak_sa HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            အတွက် (ကိန်းပြည့် i = 0; i < HEX_NUM_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        pak_sa HTP_PROF_BASIC:
        pak_sa HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - d->usecs);
            d->cycles_stop = hex_get_cycles();
            bha_nga;
        default:
            bha_nga;
    }
}

static ကိန်းပြည့် op_fence(sa_nracanaa htp_ops_context * octx) {
    sa_nracanaa htp_context *ctx = octx->ctx;
    sa_nracanaa htp_thread_trace * tr = &ctx->trace[0];
    const uint32_t seq  = (uint32_t) octx->op_params[0];
    const uint32_t mode = (uint32_t) octx->op_params[1];

    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    const sa_nracanaa htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;

    အကယ်၍ (mode == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_group_barrier(octx);

        အကယ်၍ (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->status);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(HIGH, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        ပြန်ပေး octx->status;
    }

    ကိန်းပြည့် status = HTP_STATUS_OK;
    uint64_t spins = 0;
    နေစဉ် (1) {
        uint32_t sync_seq;
        uint32_t sync_status;
        htp_fence_read(sync_fence, &sync_seq, &sync_status);
        အကယ်၍ ((int32_t)(sync_seq - seq) >= 0) {
            အကယ်၍ (sync_status > HTP_STATUS_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_status, sync_fence, seq);
                status = sync_status;
            }
            bha_nga;
        }
        အကယ်၍ (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            status = HTP_STATUS_INTERNAL_ERR;
            bha_nga;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(HIGH, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    ပြန်ပေး status;
}

static ကိန်းပြည့် op_mdev_group(sa_nracanaa htp_ops_context * octx) {
    sa_nracanaa htp_context * ctx = octx->ctx;
    const sa_nracanaa htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.count = (uint16_t) sync->ne[1];
    အကယ်၍ (ctx->mdev.count > 1) {
        ctx->mdev.count_div = init_fastdiv_values(ctx->mdev.count);
        ctx->mdev.fence_base = (uint8_t *) sync->data;
    }
    ပြန်ပေး HTP_STATUS_OK;
}

static ကိန်းပြည့် execute_op(sa_nracanaa htp_ops_context * octx) {
    switch (octx->op) {
        pak_sa HTP_OP_MDEV_GROUP:
            ပြန်ပေး op_mdev_group(octx);

        pak_sa HTP_OP_FENCE:
            ပြန်ပေး op_fence(octx);

        pak_sa HTP_OP_ALLREDUCE:
        pak_sa HTP_OP_ALLREDUCE_ADD:
            ပြန်ပေး op_allreduce(octx);

        pak_sa HTP_OP_MUL_MAT:
        pak_sa HTP_OP_MUL_MAT_ADD:
            ပြန်ပေး op_matmul(octx);

        pak_sa HTP_OP_MUL_MAT_ID:
            ပြန်ပေး op_matmul_id(octx);

        pak_sa HTP_OP_MUL_MAT_ID_NX:
            ပြန်ပေး op_matmul_id_nx(octx);

        pak_sa HTP_OP_MUL_MAT_NX:
            ပြန်ပေး op_matmul_nx(octx);

        pak_sa HTP_OP_MUL:
        pak_sa HTP_OP_ADD:
        pak_sa HTP_OP_SUB:
        pak_sa HTP_OP_DIV:
        pak_sa HTP_OP_ADD_ID:
            ပြန်ပေး op_binary(octx);

        pak_sa HTP_OP_NORM:
        pak_sa HTP_OP_RMS_NORM:
        pak_sa HTP_OP_RMS_NORM_MUL:
        pak_sa HTP_OP_SCALE:
        pak_sa HTP_OP_CLAMP:
        pak_sa HTP_OP_LEAKY_RELU:
        pak_sa HTP_OP_SQR:
        pak_sa HTP_OP_SQRT:
        pak_sa HTP_OP_UNARY_SOFTPLUS:
        pak_sa HTP_OP_UNARY_SIGMOID:
        pak_sa HTP_OP_UNARY_SILU:
        pak_sa HTP_OP_UNARY_GELU:
        pak_sa HTP_OP_UNARY_GELU_QUICK:
        pak_sa HTP_OP_UNARY_GELU_ERF:
        pak_sa HTP_OP_UNARY_NEG:
        pak_sa HTP_OP_UNARY_EXP:
        pak_sa HTP_OP_UNARY_TANH:
        pak_sa HTP_OP_UNARY_ABS:
        pak_sa HTP_OP_UNARY_LOG:
        pak_sa HTP_OP_UNARY_RELU:
        pak_sa HTP_OP_UNARY_STEP:
        pak_sa HTP_OP_L2_NORM:
        pak_sa HTP_OP_HADAMARD:
            ပြန်ပေး op_unary(octx);

        pak_sa HTP_OP_GLU_SWIGLU:
        pak_sa HTP_OP_GLU_SWIGLU_OAI:
        pak_sa HTP_OP_GLU_SWIGLU_CLAMP:
        pak_sa HTP_OP_GLU_GEGLU:
        pak_sa HTP_OP_GLU_GEGLU_QUICK:
        pak_sa HTP_OP_GLU_GEGLU_ERF:
            ပြန်ပေး op_activations(octx);

        pak_sa HTP_OP_SOFTMAX:
            ပြန်ပေး op_softmax(octx);

        pak_sa HTP_OP_ROPE:
            ပြန်ပေး op_rope(octx);

        pak_sa HTP_OP_FLASH_ATTN_EXT:
            ပြန်ပေး op_flash_attn_ext(octx);

        pak_sa HTP_OP_SET_ROWS:
            ပြန်ပေး op_set_rows(octx);

        pak_sa HTP_OP_GET_ROWS:
            ပြန်ပေး op_get_rows(octx);

        pak_sa HTP_OP_SUM:
            ပြန်ပေး op_sum(octx);

        pak_sa HTP_OP_SUM_ROWS:
            ပြန်ပေး op_sum_rows(octx);

        pak_sa HTP_OP_CPY:
        pak_sa HTP_OP_CPY_FENCE:
            ပြန်ပေး op_cpy(octx);

        pak_sa HTP_OP_REPEAT:
            ပြန်ပေး op_repeat(octx);

        pak_sa HTP_OP_ARGSORT:
            ပြန်ပေး op_argsort(octx);

        pak_sa HTP_OP_TOP_K:
            ပြန်ပေး op_top_k(octx);

        pak_sa HTP_OP_ARGMAX:
            ပြန်ပေး op_argmax(octx);

        pak_sa HTP_OP_SSM_CONV:
            ပြန်ပေး op_ssm_conv(octx);

        pak_sa HTP_OP_CUMSUM:
            ပြန်ပေး op_cumsum(octx);

        pak_sa HTP_OP_FILL:
            ပြန်ပေး op_fill(octx);

        pak_sa HTP_OP_DIAG:
            ပြန်ပေး op_diag(octx);

        pak_sa HTP_OP_SOLVE_TRI:
            ပြန်ပေး op_solve_tri(octx);

        pak_sa HTP_OP_PAD:
            ပြန်ပေး op_pad(octx);

        pak_sa HTP_OP_IM2COL:
            ပြန်ပေး op_im2col(octx);

        pak_sa HTP_OP_IM2COL_3D:
            ပြန်ပေး op_im2col_3d(octx);

        pak_sa HTP_OP_ROLL:
            ပြန်ပေး op_roll(octx);

        pak_sa HTP_OP_POOL_2D:
            ပြန်ပေး op_pool_2d(octx);

        pak_sa HTP_OP_POOL_1D:
            ပြန်ပေး op_pool_1d(octx);

        pak_sa HTP_OP_CONCAT:
            ပြန်ပေး op_concat(octx);

        pak_sa HTP_OP_GATED_DELTA_NET:
            ပြန်ပေး op_gated_delta_net(octx);

        pak_sa HTP_OP_TRI:
            ပြန်ပေး op_unary(octx);

        pak_sa HTP_OP_INVALID:
            bha_nga;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    ပြန်ပေး HTP_STATUS_NO_SUPPORT;
}

static anta_hstha bool reuse_buf(sa_nracanaa htp_context *ctx, uint32_t *m_reuse, sa_nracanaa htp_buf_desc *b) {
    b->base = NULL;

    အတွက် (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        sa_nracanaa htp_mmap *m = ctx->mmap + i;
        အကယ်၍ (m->size && m->fd == b->fd && m->flags == b->flags) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            ပြန်ပေး true;
        }
    }

    ပြန်ပေး false;
}

static anta_hstha void drop_mmap(sa_nracanaa htp_context *ctx, sa_nracanaa htp_mmap *m) {
    အကယ်၍ (m->size) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (unsigned long long) m->base, (unsigned long long) m->size);
        htp_munmap(m->base, m->size, m->flags);
        m->size = 0;
        m->base = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

static anta_hstha bool mmap_buf(sa_nracanaa htp_context *ctx, sa_nracanaa htp_buf_desc *b) {
    အကယ်၍ (b->base) ပြန်ပေး true; // already mapped

    // find unused mapping
    အတွက် (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        sa_nracanaa htp_mmap *m = &ctx->mmap[i];
        အကယ်၍ (!m->size) {
            uint64_t va = htp_mmap(b->fd, b->size, b->flags);
            အကယ်၍ (va == 0) {
                FARF(HIGH, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (unsigned long long) b->size);
                ပြန်ပေး false;
            }

            m->base   = b->base = va;
            m->fd     = b->fd;
            m->size   = b->size;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (unsigned long long) m->base, (unsigned long long) m->size, m->flags);
            ပြန်ပေး true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    ပြန်ပေး false;
}

static void prep_op_bufs(sa_nracanaa htp_context *ctx, sa_nracanaa htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    အတွက် (uint32_t i=0; i < n_bufs; i++) {
        sa_nracanaa htp_buf_desc *b = bufs + i;
        အကယ်၍ (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } မဟုတ်ပါက အကယ်၍ (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->size;
        }
        FARF(HIGH, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    အကယ်၍ (b_reuse == n_bufs) ပြန်ပေး; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    အတွက် (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        အကယ်၍ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].size;
        }
    }

    FARF(HIGH, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    အကယ်၍ ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        အတွက် (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            အကယ်၍ (!used && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    အတွက် (uint32_t i=0; i < n_bufs; i++) {
        sa_nracanaa htp_buf_desc *b = bufs + i;
        အကယ်၍ (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            bha_nga;
        }
        FARF(HIGH, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
    }

    အကယ်၍ (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(HIGH, "prep-bufs : dropping 32-bit mappings to defragment address space");
        အတွက် (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            အကယ်၍ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        အတွက် (uint32_t i=0; i < n_bufs; i++) {
            sa_nracanaa htp_buf_desc *b = bufs + i;
            အကယ်၍ (!(b->flags & HTP_BUF_EXTENDED)) {
                b->base = 0;
            }
            အကယ်၍ (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (unsigned long long) b->size);
                abort();
            }
            FARF(HIGH, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (unsigned long long) b->base, (unsigned long long) b->size, b->flags);
        }
    }
}

static void prep_tensor(sa_nracanaa htp_context *ctx, sa_nracanaa htp_buf_desc *bufs, sa_nracanaa htp_tensor *tens, uint32_t idx, sa_nracanaa htp_tensor *t) {
    uint64_t offset = t->data;
    uint32_t bi     = t->bi;

    t->data  = bufs[bi].base + offset;  // update data to the actual pointer

    FARF(HIGH, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (unsigned long long) offset, t->size, (unsigned long long) t->data,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

static void prep_tensors(sa_nracanaa htp_context *ctx, sa_nracanaa htp_buf_desc *bufs, sa_nracanaa htp_tensor *tens, uint32_t n_tens) {
    အတွက် (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

static void mdev_group_init(sa_nracanaa htp_context * ctx, const sa_nracanaa htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, parimaa_na(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

static ကိန်းပြည့် proc_op_req(sa_nracanaa htp_ops_context * octx, sa_nracanaa htp_buf_desc * bufs, uint32_t n_bufs,
                       sa_nracanaa htp_tensor * tens, uint32_t idx, sa_nracanaa htp_op_desc * op) {
    memcpy(octx->op_params,     op->params, parimaa_na(octx->op_params));
    memcpy(octx->kernel_params, op->kernel_params, parimaa_na(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_threads     = octx->ctx->n_threads;
    octx->n_threads_div = octx->ctx->n_threads_div;

    FARF(HIGH, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    အတွက် (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        အကယ်၍ (src_idx == 0xffff) {
            octx->src[i]     = NULL;
            continue;
        }

        sa_nracanaa htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(HIGH, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (unsigned long long) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    အတွက် (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        အကယ်၍ (dst_idx == 0xffff) {
            octx->dsts[i]    = NULL;
            continue;
        }
        sa_nracanaa htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(HIGH, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (unsigned long long) dst->data, dst->size,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_group_barrier(octx);

    ကိန်းပြည့် status = execute_op(octx);

    htp_ops_context_set_status(octx, status);

    octx->src0_spad.src = NULL;
    octx->src1_spad.src = NULL;
    octx->src2_spad.src = NULL;
    octx->src3_spad.src = NULL;
    octx->dst_spad.src  = NULL;

    ပြန်ပေး octx->status;
}

static void process_opbatch(sa_nracanaa htp_context * ctx, const sa_nracanaa htp_opbatch_req * req, const sa_nracanaa dspqueue_buffer * dbuf) {
    dspqueue_t queue = ctx->dsp_queue;
    ကိန်းပြည့် err;

    const uint32_t n_bufs = req->n_bufs;
    const uint32_t n_tens = req->n_tensors;
    const uint32_t n_ops  = req->n_ops;

    const uint32_t b_size = parimaa_na(sa_nracanaa htp_buf_desc)  * n_bufs;
    const uint32_t t_size = parimaa_na(sa_nracanaa htp_tensor)    * n_tens;
    const uint32_t o_size = parimaa_na(sa_nracanaa htp_op_desc)   * n_ops;
    const uint32_t p_size = parimaa_na(sa_nracanaa htp_prof_desc) * n_ops;
    const uint32_t tr_size = (HTP_MAX_NTHREADS + 1) * req->n_traces * parimaa_na(sa_nracanaa htp_trace_desc);

    အကယ်၍ (dbuf->size < b_size + t_size + o_size + p_size + tr_size) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->size, b_size + t_size + o_size + p_size + tr_size);
        ပြန်ပေး;
    }

    FARF(HIGH, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (unsigned long long) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->size, b_size, t_size, o_size);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    sa_nracanaa htp_buf_desc* bufs = (sa_nracanaa htp_buf_desc*)  m_ptr; m_ptr += b_size;
    sa_nracanaa htp_tensor*   tens = (sa_nracanaa htp_tensor*)    m_ptr; m_ptr += t_size;
    sa_nracanaa htp_op_desc*   ops = (sa_nracanaa htp_op_desc*)   m_ptr; m_ptr += o_size;
    sa_nracanaa htp_prof_desc* pds = (sa_nracanaa htp_prof_desc*) m_ptr;

    sa_nracanaa profile_data batch_prof;
    profile_start(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, parimaa_na(ctx->trace));
    အကယ်၍ (ctx->profiler == HTP_PROF_TRACE) {
        sa_nracanaa htp_trace_desc * trace_events = (sa_nracanaa htp_trace_desc *) (m_ptr + p_size);
        အတွက် (ကိန်းပြည့် t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, parimaa_na(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    sa_nracanaa htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, parimaa_na(*octx));
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
    octx->ctx           = ctx;

    mdev_group_init(ctx, req);

    work_queue_wakeup(ctx->work_queue);
    အကယ်၍ (ctx->hmx_queue) {
        hmx_queue_wakeup(ctx->hmx_queue);
    }

    ကိန်းပြည့် op_status = HTP_STATUS_OK;
    octx->status  = HTP_STATUS_OK;
    အတွက် (uint32_t i = 0; i < n_ops; i++) {
        sa_nracanaa profile_data prof;

        profile_start(ctx->profiler, &prof);

        op_status = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_set_status(octx, op_status);

        အကယ်၍ (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].cycles_start = prof.cycles_start;
            pds[i].cycles_stop = prof.cycles_stop;
            အတွက် (ကိန်းပြည့် j = 0; j < HEX_NUM_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    အကယ်၍ (ctx->hmx_queue) {
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

    sa_nracanaa htp_opbatch_rsp rsp;
    memset(&rsp, 0, parimaa_na(rsp));
    rsp.seq          = req->seq;
    rsp.status       = octx->status;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.cycles_start = batch_prof.cycles_start;
    rsp.cycles_stop  = batch_prof.cycles_stop;

    အကယ်၍ (ctx->profiler == HTP_PROF_TRACE) {
        အတွက် (ကိန်းပြည့် t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].count;
        }
    }

    sa_nracanaa dspqueue_buffer write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(queue, 0, 1, &write_dbuf, parimaa_na(rsp), (const uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    အကယ်၍ (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
    }
}

#define DSPQUEUE_READ_TIMEOUT_USEC 5000
#define DSPQUEUE_POLL_TIMEOUT_USEC 100
#define DSPQUEUE_POLL_COUNT        100

static void process_ops(sa_nracanaa htp_context * ctx) {
    dspqueue_t queue = ctx->dsp_queue;
    ကိန်းပြည့် err;

    uint32_t poll_count = DSPQUEUE_POLL_COUNT;

    vtcm_acquire(ctx);

    နေစဉ် (!atomic_load(&ctx->vtcm_needs_release) && !atomic_load(&ctx->killed)) {
        sa_nracanaa htp_opbatch_req req;
        uint32_t r_size = parimaa_na(req);

        sa_nracanaa dspqueue_buffer dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_size, &r_size, (uint8_t *) &req);
        အကယ်၍ (err == AEE_EWOULDBLOCK) {
            အကယ်၍ (--poll_count) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                continue;
            }
            bha_nga;
        }

        အကယ်၍ (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            bha_nga;
        }

        အကယ်၍ (r_size < parimaa_na(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_size, n_dbufs);
            continue;
        }

        // Reset poll count for valid requests
        poll_count = DSPQUEUE_POLL_COUNT;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

static void htp_packet_callback(dspqueue_t queue, ကိန်းပြည့် error, void * context) {
    (void) queue;
    (void) error;
    sa_nracanaa htp_handle * h = (sa_nracanaa htp_handle *) context;
    အကယ်၍ (h && h->ctx) {
        process_ops(h->ctx);
    }
}

static void htp_main_thread(void * context) {
    sa_nracanaa htp_context * ctx = (sa_nracanaa htp_context *) context;

    FARF(HIGH, "htp-main-thread: started");

    နေစဉ် (!atomic_load(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t num_buffers = 0;
        uint32_t message_length = 0;

        ကိန်းပြည့် err = dspqueue_peek(ctx->dsp_queue, &flags, &num_buffers, &message_length, 50000);
        အကယ်၍ (err == 0) {
            process_ops(ctx);
        } မဟုတ်ပါက အကယ်၍ (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            continue;
        } မဟုတ်ပါက {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (unsigned) err);
            bha_nga;
        }
    }

    FARF(HIGH, "htp-main-thread: stopped");
}
