#ಸೂಚನೆ clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

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

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
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

#ವ್ಯಾಖ್ಯಾನಿಸು HMX_ಸರತಿ_ಸಾಮರ್ಥ್ಯ     128
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_ಸರತಿ_ರಾಶಿ_ಗಾತ್ರ   16384
#ವ್ಯಾಖ್ಯಾನಿಸು WORK_ಸರತಿ_ಸಾಮರ್ಥ್ಯ    16
#ವ್ಯಾಖ್ಯಾನಿಸು WORK_ಸರತಿ_ರಾಶಿ_ಗಾತ್ರ  16384
#ವ್ಯಾಖ್ಯಾನಿಸು MAIN_THREAD_ರಾಶಿ_ಗಾತ್ರ 32768

_Static_assert(WORK_ಸರತಿ_MAX_N_ಎಳೆಗಳು >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

ರಚನೆ htp_handle {
    ರಚನೆ htp_context * ctx;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t htp_mmap(uint32_t fd, uint64_t ಗಾತ್ರ_2, uint32_t flags) {
#ಆದರೆ __HVX_ARCH__ > 79
    ಆದರೆ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, ಗಾತ್ರ(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = ಗಾತ್ರ_2;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        ಆದರೆ (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) ಗಾತ್ರ_2);
            ಹಿಂತಿರುಗಿ 0;
        }

        ಹಿಂತಿರುಗಿ payload.mmap.dsp_va;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಆದರೆ (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        ಹಿಂತಿರುಗಿ 0;
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (ಗಾತ್ರ_2 > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) ಗಾತ್ರ_2);
        ಹಿಂತಿರುಗಿ 0;
    }

    ಶೂನ್ಯ * va = (ಶೂನ್ಯ *)-1;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ retry = 0; retry < 2; retry++) {
#ಆದರೆ __HVX_ARCH__ > 73
        va = HAP_mmap2(ಖಾಲಿ, (size_t) ಗಾತ್ರ_2, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#ಇಲ್ಲದಿದ್ದರೆ
        ಆದರೆ (ಗಾತ್ರ_2 > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) ಗಾತ್ರ_2);
            abort();
        }
        va = HAP_mmap(ಖಾಲಿ, (ಪೂರ್ಣಾಂಕ) ಗಾತ್ರ_2, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (va != (ಶೂನ್ಯ *)-1 && va != ಖಾಲಿ) {
            ಹಿಂತಿರುಗಿ (uint64_t) (uintptr_t) va;
        }
        ಆದರೆ (retry == 0) {
            FARF(ಮೇಲಿನ_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) ಗಾತ್ರ_2);
        }
    }
    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_munmap(uint64_t va, uint64_t ಗಾತ್ರ_2, uint32_t flags) {
#ಆದರೆ __HVX_ARCH__ > 79
    ಆದರೆ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, ಗಾತ್ರ(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = ಗಾತ್ರ_2;
        HAP_mem_request(&payload);
        ಹಿಂತಿರುಗಿ;
    }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ __HVX_ARCH__ > 73
    HAP_munmap2((ಶೂನ್ಯ *) (uintptr_t) va, (size_t) ಗಾತ್ರ_2);
#ಇಲ್ಲದಿದ್ದರೆ
    HAP_munmap((ಶೂನ್ಯ *) (uintptr_t) va, (ಪೂರ್ಣಾಂಕ) ಗಾತ್ರ_2);
#ಆದರೆ_ಅಂತ್ಯ
}

AEEResult htp_iface_open(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * uri, remote_handle64 * handle) {
    (ಶೂನ್ಯ) uri;
    ರಚನೆ htp_handle * h = ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(1, ಗಾತ್ರ(*h));
    ಆದರೆ (h == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    ಹಿಂತಿರುಗಿ AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    ರಚನೆ htp_handle * h = (ರಚನೆ htp_handle *) handle;
    ಆದರೆ (!h) {
        ಹಿಂತಿರುಗಿ AEE_EBADPARM;
    }

    ಪೂರ್ಣಾಂಕ err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    ಆದರೆ (err) {
        ಆದರೆ (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } ಇಲ್ಲದಿದ್ದರೆ {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    ಹಿಂತಿರುಗಿ err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t ಬಹುಲಕ, ಸ್ಥಿರಾಂಕ htp_iface_pmu_conf* pmu_conf) {
    ರಚನೆ htp_handle * h = (ರಚನೆ htp_handle *) handle;
    ಆದರೆ (!h || !h->ctx) {
        ಹಿಂತಿರುಗಿ AEE_EBADPARM;
    }
    ರಚನೆ htp_context * ctx = h->ctx;

    ಆದರೆ (ಬಹುಲಕ == HTP_PROF_PMU) {
        ಸ್ಥಿರಾಂಕ uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        ಗಾಗಿ (; i < HEX_ಅಂಕೆ_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        ಗಾಗಿ (i = 0; i < HEX_ಅಂಕೆ_PMU_COUNTERS; i++) {
            cfg |= (((events[i] >> 8) & 3) << (i * 2));
        }

        FARF(ALWAYS, "Configuring PMU registers: evtcfg = 0x%x, evtcfg1 = 0x%x, pmucfg = 0x%x", evtcfg, evtcfg1, cfg);

        // Configure PMU registers
        qurt_pmu_ಗಣ(QURT_PMUCFG,     cfg);
        qurt_pmu_ಗಣ(QURT_PMUEVTCFG,  evtcfg);
        qurt_pmu_ಗಣ(QURT_PMUEVTCFG1, evtcfg1);
        qurt_pmu_enable(1);
    }

    ctx->profiler = ಬಹುಲಕ;

    ಹಿಂತಿರುಗಿ AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    ರಚನೆ htp_handle * h = (ರಚನೆ htp_handle *) handle;
    ಆದರೆ (!h) {
        ಹಿಂತಿರುಗಿ AEE_EBADPARM;
    }

    ರಚನೆ htp_context * ctx = h->ctx;
    ಆದರೆ (ctx) {
        ಆದರೆ (ctx->dsp_ಸರತಿ) {
            FARF(ERROR, "Closing handle with queue still open");
            ಹಿಂತಿರುಗಿ AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        ಗಾಗಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            ಆದರೆ (ctx->mmap[i].ಗಾತ್ರ_2) {
                htp_munmap(ctx->mmap[i].ತಳ, ctx->mmap[i].ಗಾತ್ರ_2, ctx->mmap[i].flags);
                ctx->mmap[i].ಗಾತ್ರ_2 = 0;
                ctx->mmap[i].ತಳ = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        ಆದರೆ (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        ಆದರೆ (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        ಬಿಡುಗಡೆ(ctx);
        h->ctx = ಖಾಲಿ;
    }

    ಬಿಡುಗಡೆ(h);
    ಹಿಂತಿರುಗಿ AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t ಗಾತ್ರ_2) {
    ರಚನೆ htp_handle * h = (ರಚನೆ htp_handle *) handle;
    ಆದರೆ (!h || !h->ctx) {
        ಹಿಂತಿರುಗಿ AEE_EBADPARM;
    }
    ರಚನೆ htp_context * ctx = h->ctx;

    // See if we already have this mapping
    ಗಾಗಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ರಚನೆ htp_mmap *m = &ctx->mmap[i];
        ಆದರೆ (m->fd == fd) {
            ಹಿಂತಿರುಗಿ AEE_SUCCESS;
        }
    }

    // Add new mapping
    ಗಾಗಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ರಚನೆ htp_mmap *m = &ctx->mmap[i];
        ಆದರೆ (!m->ಗಾತ್ರ_2) {
            FARF(ಮೇಲಿನ_2, "mmap : fd %u size %llu", fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) ಗಾತ್ರ_2);
            uint64_t va = htp_mmap(fd, ಗಾತ್ರ_2, 0);
            ಆದರೆ (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) ಗಾತ್ರ_2);
                ಹಿಂತಿರುಗಿ AEE_EFAILED;
            }

            m->ತಳ   = va;
            m->fd     = fd;
            m->ಗಾತ್ರ_2   = ಗಾತ್ರ_2;
            m->flags  = 0;

            ಹಿಂತಿರುಗಿ AEE_SUCCESS;
        }
    }

    ಹಿಂತಿರುಗಿ AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    ರಚನೆ htp_handle * h = (ರಚನೆ htp_handle *) handle;
    ಆದರೆ (!h || !h->ctx) {
        ಹಿಂತಿರುಗಿ AEE_EBADPARM;
    }
    ರಚನೆ htp_context * ctx = h->ctx;

    ಗಾಗಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ರಚನೆ htp_mmap *m = &ctx->mmap[i];
        ಆದರೆ (fd < 0 || m->fd == fd) {
            FARF(ಮೇಲಿನ_2, "unmmap : base 0x%llx fd %u size %llu", (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) m->ತಳ, m->fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) m->ಗಾತ್ರ_2);
            htp_munmap(m->ತಳ, m->ಗಾತ್ರ_2, m->flags);
            m->ಗಾತ್ರ_2   = 0;
            m->ತಳ   = ಖಾಲಿ;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    ಹಿಂತಿರುಗಿ AEE_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ vtcm_acquire(ರಚನೆ htp_context * ctx) {
    ಆದರೆ (!atomic_ತುಂಬು(&ctx->vtcm_valid)) {
        ಪೂರ್ಣಾಂಕ err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        ಆದರೆ (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (ಚಿಹ್ನೆರಹಿತ)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_ಆದ್ಯತೆ(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ vtcm_release(ರಚನೆ htp_context * ctx) {
    ಆದರೆ (atomic_ತುಂಬು(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ vtcm_release_callback(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ rctx, ಶೂನ್ಯ * state) {
    ರಚನೆ htp_context * ctx = (ರಚನೆ htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ vtcm_query_ಗಾತ್ರ(ಶೂನ್ಯ) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ vtcm_ಗಾತ್ರ = 8 * 1024 * 1024;  // 8MB default
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ vtcm_avail;
    compute_res_vtcm_ಪುಟ_t ಒಟ್ಟು_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_ಗಾತ್ರ, &ಒಟ್ಟು_layout, &vtcm_avail, &avail_layout);
    ಹಿಂತಿರುಗಿ vtcm_ಗಾತ್ರ;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ vtcm_alloc(ರಚನೆ htp_context * ctx) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ vtcm_ಗಾತ್ರ = vtcm_query_ಗಾತ್ರ();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_ಗಣ_serialize(&attr, 0);
    HAP_compute_res_attr_ಗಣ_cache_ಬಹುಲಕ(&attr, 1);
    HAP_compute_res_attr_ಗಣ_vtcm_param_v2(&attr, vtcm_ಗಾತ್ರ, vtcm_ಗಾತ್ರ, vtcm_ಗಾತ್ರ); // single page
    HAP_compute_res_attr_ಗಣ_release_callback(&attr, vtcm_release_callback, (ಶೂನ್ಯ *) ctx);
    HAP_compute_res_attr_ಗಣ_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    ಆದರೆ (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ AEE_ENOMEMORY;
    }

    ಶೂನ್ಯ * vtcm_ptr;
    ಆದರೆ (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_ಗಾತ್ರ) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ AEE_ENOMEMORY;
    }

    ctx->vtcm_ತಳ          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_ಗಾತ್ರ          = vtcm_ಗಾತ್ರ;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಶೂನ್ಯ vtcm_free(ರಚನೆ htp_context * ctx) {
    ಆದರೆ (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_ತಳ = 0;
        ctx->vtcm_rctx = 0;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ htp_main_thread(ಶೂನ್ಯ * context);
ಸ್ಥಿರ ಶೂನ್ಯ htp_packet_callback(dspqueue_t ಸರತಿ, ಪೂರ್ಣಾಂಕ error, ಶೂನ್ಯ * context);
ಸ್ಥಿರ ಶೂನ್ಯ htp_error_callback(dspqueue_t ಸರತಿ, ಪೂರ್ಣಾಂಕ error, ಶೂನ್ಯ * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
ರಚನೆ htp_power_profile {
    boolean  dcvs_enable;
    uint32_t bus_min_ಮೂಲೆ;
    uint32_t bus_max_ಮೂಲೆ;
    uint32_t bus_ಗುರಿ_ಮೂಲೆ;
    uint32_t core_min_ಮೂಲೆ;
    uint32_t core_max_ಮೂಲೆ;
    uint32_t core_ಗುರಿ_ಮೂಲೆ;
    boolean  sleep_disable;
    uint32_t hmx_min_ಮೂಲೆ;    // only used when __HVX_ARCH__ >= 75
    uint32_t hmx_max_ಮೂಲೆ;
    uint32_t hmx_ಗುರಿ_ಮೂಲೆ;
    uint32_t hmx_perf_ಬಹುಲಕ;
};

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_POWER_PROFILE_ನಮೂದು(min_c, max_c, hmx_c) \
    { \
        .dcvs_enable        = TRUE, \
        .bus_min_ಮೂಲೆ     = (min_c), \
        .bus_max_ಮೂಲೆ     = (max_c), \
        .bus_ಗುರಿ_ಮೂಲೆ  = (max_c), \
        .core_min_ಮೂಲೆ    = (min_c), \
        .core_max_ಮೂಲೆ    = (max_c), \
        .core_ಗುರಿ_ಮೂಲೆ = (max_c), \
        .sleep_disable      = TRUE, \
        .hmx_min_ಮೂಲೆ     = (hmx_c), \
        .hmx_max_ಮೂಲೆ     = (hmx_c), \
        .hmx_ಗುರಿ_ಮೂಲೆ  = (hmx_c), \
        .hmx_perf_ಬಹುಲಕ      = HAP_CLK_PERF_ಮೇಲಿನ, \
    }

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ htp_power_profile htp_power_profiles[HTP_POWER_ಬಹುಲಕ_ಎಣಿಕೆ] = {
    [HTP_POWER_ಬಹುಲಕ_ಕೆಳಗಿನ_POWER_SAVER] =
        HTP_POWER_PROFILE_ನಮೂದು(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_ಬಹುಲಕ_POWER_SAVER] =
        HTP_POWER_PROFILE_ನಮೂದು(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_ಬಹುಲಕ_ಮೇಲಿನ_POWER_SAVER] =
        HTP_POWER_PROFILE_ನಮೂದು(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_ಬಹುಲಕ_ಕೆಳಗಿನ_ಸಮತೋಲಿತ] =
        HTP_POWER_PROFILE_ನಮೂದು(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_ಬಹುಲಕ_ಸಮತೋಲಿತ] =
        HTP_POWER_PROFILE_ನಮೂದು(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_ಬಹುಲಕ_ಮೇಲಿನ_PERFORMANCE] =
        HTP_POWER_PROFILE_ನಮೂದು(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_ಬಹುಲಕ_SUSTAINED_ಮೇಲಿನ_PERFORMANCE] =
        HTP_POWER_PROFILE_ನಮೂದು(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_ಬಹುಲಕ_ಅವಧಿ] = {
        .dcvs_enable        = FALSE,
        .bus_min_ಮೂಲೆ     = HAP_DCVS_VCORNER_MAX,
        .bus_max_ಮೂಲೆ     = HAP_DCVS_VCORNER_MAX,
        .bus_ಗುರಿ_ಮೂಲೆ  = HAP_DCVS_VCORNER_MAX,
        .core_min_ಮೂಲೆ    = HAP_DCVS_VCORNER_MAX,
        .core_max_ಮೂಲೆ    = HAP_DCVS_VCORNER_MAX,
        .core_ಗುರಿ_ಮೂಲೆ = HAP_DCVS_VCORNER_MAX,
        .sleep_disable      = TRUE,
        .hmx_min_ಮೂಲೆ     = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_max_ಮೂಲೆ     = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_ಗುರಿ_ಮೂಲೆ  = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_perf_ಬಹುಲಕ      = HAP_CLK_PERF_ಮೇಲಿನ,
    },
};

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HTP_POWER_PROFILE_ನಮೂದು

AEEResult htp_iface_ಆರಂಭ(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_ಸರತಿ_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_ಬಹುಲಕ) {
    ರಚನೆ htp_handle * h = (ರಚನೆ htp_handle *) handle;
    ಆದರೆ (!h) {
        ಹಿಂತಿರುಗಿ AEE_EBADPARM;
    }

    ಆದರೆ (power_ಬಹುಲಕ >= HTP_POWER_ಬಹುಲಕ_ಎಣಿಕೆ) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (ಚಿಹ್ನೆರಹಿತ) power_ಬಹುಲಕ);
        power_ಬಹುಲಕ = HTP_POWER_ಬಹುಲಕ_ಅವಧಿ;
    }
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_power_profile * pwr = &htp_power_profiles[power_ಬಹುಲಕ];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (ಚಿಹ್ನೆರಹಿತ) power_ಬಹುಲಕ, (ಪೂರ್ಣಾಂಕ) pwr->dcvs_enable,
         (ಚಿಹ್ನೆರಹಿತ) pwr->bus_min_ಮೂಲೆ, (ಚಿಹ್ನೆರಹಿತ) pwr->bus_max_ಮೂಲೆ, (ಚಿಹ್ನೆರಹಿತ) pwr->bus_ಗುರಿ_ಮೂಲೆ,
         (ಚಿಹ್ನೆರಹಿತ) pwr->core_min_ಮೂಲೆ, (ಚಿಹ್ನೆರಹಿತ) pwr->core_max_ಮೂಲೆ, (ಚಿಹ್ನೆರಹಿತ) pwr->core_ಗುರಿ_ಮೂಲೆ,
         (ಪೂರ್ಣಾಂಕ) pwr->sleep_disable,
         (ಚಿಹ್ನೆರಹಿತ) pwr->hmx_min_ಮೂಲೆ, (ಚಿಹ್ನೆರಹಿತ) pwr->hmx_max_ಮೂಲೆ, (ಚಿಹ್ನೆರಹಿತ) pwr->hmx_ಗುರಿ_ಮೂಲೆ);

    ಆದರೆ (h->ctx) {
        FARF(ERROR, "Queue already open");
        ಹಿಂತಿರುಗಿ AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    ಪೂರ್ಣಾಂಕ fastrpc_tid  = qurt_thread_get_id();
    ಪೂರ್ಣಾಂಕ fastrpc_prio = qurt_thread_get_ಆದ್ಯತೆ(fastrpc_tid);
    ಪೂರ್ಣಾಂಕ main_prio    = fastrpc_prio - 10;
    ಆದರೆ (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_ಸರತಿ = ಖಾಲಿ;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    ಪೂರ್ಣಾಂಕ err = dspqueue_import(dsp_ಸರತಿ_id, ಖಾಲಿ, ಖಾಲಿ, (ಶೂನ್ಯ *) h, &dsp_ಸರತಿ);
    ಆದರೆ (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(ಮೇಲಿನ_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_ಸರತಿ_id, htp_packet_callback, htp_error_callback, (ಶೂನ್ಯ *) h, &dsp_ಸರತಿ);
    }

    ಆದರೆ (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (ಚಿಹ್ನೆರಹಿತ) err);
        ಹಿಂತಿರುಗಿ err;
    }

    qurt_sysenv_max_hthreads_t hw_ಎಳೆಗಳು;
    qurt_sysenv_get_max_hw_ಎಳೆಗಳು(&hw_ಎಳೆಗಳು);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    ಆದರೆ (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    ಆದರೆ (n_hvx > hw_ಎಳೆಗಳು.max_hthreads) {
        n_hvx = hw_ಎಳೆಗಳು.max_hthreads;
    }
    ಆದರೆ (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t ಸ್ಥಾನಾಂತರ = ಗಾತ್ರ(ರಚನೆ htp_context);

    // 2. main_stack
    size_t ಸ್ಥಾನಾಂತರ_main_ರಾಶಿ = 0;
    size_t ಗಾತ್ರ_main_ರಾಶಿ   = 0;
    ಆದರೆ (!use_callbacks) {
        ಸ್ಥಾನಾಂತರ_main_ರಾಶಿ = hex_align_ಮೇಲೆ(ಸ್ಥಾನಾಂತರ, 4096);
        ಗಾತ್ರ_main_ರಾಶಿ   = MAIN_THREAD_ರಾಶಿ_ಗಾತ್ರ;
        ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ_main_ರಾಶಿ + ಗಾತ್ರ_main_ರಾಶಿ;
    }

    // 3. work_queue
    size_t wq_align  = work_ಸರತಿ_alignof();
    size_t ಸ್ಥಾನಾಂತರ_wq = hex_align_ಮೇಲೆ(ಸ್ಥಾನಾಂತರ, wq_align);
    size_t ಗಾತ್ರ_wq   = work_ಸರತಿ_sizeof(n_hvx, WORK_ಸರತಿ_ಸಾಮರ್ಥ್ಯ, WORK_ಸರತಿ_ರಾಶಿ_ಗಾತ್ರ);
    ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ_wq + ಗಾತ್ರ_wq;

    // 4. dma_queue
    size_t dma_align = dma_ಸರತಿ_alignof();
    size_t ಸ್ಥಾನಾಂತರ_dma = hex_align_ಮೇಲೆ(ಸ್ಥಾನಾಂತರ, dma_align);
    size_t ಗಾತ್ರ_dma = 0;
    ಗಾಗಿ (uint32_t i = 0; i < n_hvx; i++) {
        ಗಾತ್ರ_dma  = hex_align_ಮೇಲೆ(ಗಾತ್ರ_dma, dma_ಸರತಿ_alignof());
        ಗಾತ್ರ_dma += dma_ಸರತಿ_sizeof(256);
    }
    ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ_dma + ಗಾತ್ರ_dma;

    // 5. hmx_queue
    size_t ಸ್ಥಾನಾಂತರ_hmx = 0;
    size_t ಗಾತ್ರ_hmx = 0;
    ಆದರೆ (n_hmx) {
        size_t hmx_align = hmx_ಸರತಿ_alignof();
        ಸ್ಥಾನಾಂತರ_hmx = hex_align_ಮೇಲೆ(ಸ್ಥಾನಾಂತರ, hmx_align);
        ಗಾತ್ರ_hmx   = hmx_ಸರತಿ_sizeof(HMX_ಸರತಿ_ಸಾಮರ್ಥ್ಯ, HMX_ಸರತಿ_ರಾಶಿ_ಗಾತ್ರ);
        ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ_hmx + ಗಾತ್ರ_hmx;
    }

    size_t footprint = hex_align_ಮೇಲೆ(ಸ್ಥಾನಾಂತರ, 128);

    ಶೂನ್ಯ * ಖಂಡ = memalign(4096, footprint);
    ಆದರೆ (!ಖಂಡ) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_ಸರತಿ);
        ಹಿಂತಿರುಗಿ AEE_ENOMEMORY;
    }
    memset(ಖಂಡ, 0, footprint);

    h->ctx = (ರಚನೆ htp_context *) ಖಂಡ;
    ರಚನೆ htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_ಸರತಿ   = dsp_ಸರತಿ;

    err = vtcm_alloc(ctx);
    ಆದರೆ (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        ಹಿಂತಿರುಗಿ AEE_ENOMEMORY;
    }

    HAP_ಗಣ_FARFRuntime_Logging_Params(0xffff, ಖಾಲಿ, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, ಗಾತ್ರ(HAP_power_request_t));
        request.type    = HAP_power_ಗಣ_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        ಆದರೆ ((err = HAP_power_ಗಣ((ಶೂನ್ಯ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ಹಿಂತಿರುಗಿ err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, ಗಾತ್ರ(request));

        request.type                              = HAP_power_ಗಣ_DCVS_v3;
        request.dcvs_v3.ಗಣ_dcvs_enable           = TRUE;
        request.dcvs_v3.dcvs_enable               = pwr->dcvs_enable;
        request.dcvs_v3.ಗಣ_bus_params            = TRUE;
        request.dcvs_v3.bus_params.min_ಮೂಲೆ     = pwr->bus_min_ಮೂಲೆ;
        request.dcvs_v3.bus_params.max_ಮೂಲೆ     = pwr->bus_max_ಮೂಲೆ;
        request.dcvs_v3.bus_params.ಗುರಿ_ಮೂಲೆ  = pwr->bus_ಗುರಿ_ಮೂಲೆ;
        request.dcvs_v3.ಗಣ_core_params           = TRUE;
        request.dcvs_v3.core_params.min_ಮೂಲೆ    = pwr->core_min_ಮೂಲೆ;
        request.dcvs_v3.core_params.max_ಮೂಲೆ    = pwr->core_max_ಮೂಲೆ;
        request.dcvs_v3.core_params.ಗುರಿ_ಮೂಲೆ = pwr->core_ಗುರಿ_ಮೂಲೆ;
        request.dcvs_v3.ಗಣ_sleep_disable         = TRUE;
        request.dcvs_v3.sleep_disable             = pwr->sleep_disable;

#ಆದರೆ (__HEXAGON_ARCH__ >= 79)
        HAP_ಗಣ_dcvs_v3_protected_bus_corners(&request, 1);
#ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ ((err = HAP_power_ಗಣ((ಶೂನ್ಯ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ಹಿಂತಿರುಗಿ err;
        }

        memset(&request, 0, ಗಾತ್ರ(request));
        request.type         = HAP_power_ಗಣ_HVX;
        request.hvx.power_ಮೇಲೆ = TRUE;
        ಆದರೆ ((err = HAP_power_ಗಣ((ಶೂನ್ಯ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ಹಿಂತಿರುಗಿ err;
        }
    }

#ಆದರೆ __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, ಗಾತ್ರ(HAP_power_request_t));
        request.type = HAP_power_ಗಣ_HMX_v2;
        request.hmx_v2.ಗಣ_power     = TRUE;
        request.hmx_v2.power_ಮೇಲೆ      = TRUE;
        request.hmx_v2.ಗಣ_clock     = TRUE;
        request.hmx_v2.ಗುರಿ_ಮೂಲೆ = pwr->hmx_ಗುರಿ_ಮೂಲೆ;
        request.hmx_v2.min_ಮೂಲೆ    = pwr->hmx_min_ಮೂಲೆ;
        request.hmx_v2.max_ಮೂಲೆ    = pwr->hmx_max_ಮೂಲೆ;
        request.hmx_v2.perf_ಬಹುಲಕ     = pwr->hmx_perf_ಬಹುಲಕ;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_ಗಣ((ಶೂನ್ಯ *) ctx, &request);
        ಆದರೆ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            ಹಿಂತಿರುಗಿ err;
        }
    }
#ಇಲ್ಲದಿದ್ದರೆ
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, ಗಾತ್ರ(HAP_power_request_t));
        request.type         = HAP_power_ಗಣ_HMX;
        request.hmx.power_ಮೇಲೆ = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_ಗಣ((ಶೂನ್ಯ *) ctx, &request);
        ಆದರೆ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            ಹಿಂತಿರುಗಿ err;
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_ಸರತಿ   = ಖಾಲಿ;
    ಆದರೆ (n_hmx) {
        ಶೂನ್ಯ * hmx_ptr = (ಶೂನ್ಯ *) ((uintptr_t) ಖಂಡ + ಸ್ಥಾನಾಂತರ_hmx);
        ctx->hmx_ಸರತಿ = hmx_ಸರತಿ_init(hmx_ptr, HMX_ಸರತಿ_ಸಾಮರ್ಥ್ಯ, HMX_ಸರತಿ_ರಾಶಿ_ಗಾತ್ರ, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(ಮೇಲಿನ_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_ಎಳೆಗಳು = n_hvx;
    ctx->n_ಎಳೆಗಳು_div = init_fastdiv_ಮೌಲ್ಯಗಳು(ctx->n_ಎಳೆಗಳು);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) ಖಂಡ + ಸ್ಥಾನಾಂತರ_dma);
    size_t ಗಾತ್ರ_dma_q = dma_ಸರತಿ_sizeof(256);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ctx->n_ಎಳೆಗಳು; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_ಮೇಲೆ((uintptr_t) dma_ptr_curr, dma_ಸರತಿ_alignof());
        ctx->dma[i] = dma_ಸರತಿ_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += ಗಾತ್ರ_dma_q;
    }

    ctx->ddr_spad_ಗಾತ್ರ = 512 * 1024; // 512 KB
    ctx->ddr_spad_ತಳ = memalign(128, ctx->ddr_spad_ಗಾತ್ರ);

    ಶೂನ್ಯ * wq_ptr = (ಶೂನ್ಯ *) ((uintptr_t) ಖಂಡ + ಸ್ಥಾನಾಂತರ_wq);
    ctx->work_ಸರತಿ = work_ಸರತಿ_init(wq_ptr, n_hvx, WORK_ಸರತಿ_ಸಾಮರ್ಥ್ಯ, WORK_ಸರತಿ_ರಾಶಿ_ಗಾತ್ರ);

    ctx->main_ರಾಶಿ = ಖಾಲಿ;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    ಆದರೆ (!use_callbacks) {
        // Start main compute thread
        ctx->main_ರಾಶಿ = (ಶೂನ್ಯ *) ((uintptr_t) ಖಂಡ + ಸ್ಥಾನಾಂತರ_main_ರಾಶಿ);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_ಗಣ_ರಾಶಿ_addr(&attr, ctx->main_ರಾಶಿ);
        qurt_thread_attr_ಗಣ_ರಾಶಿ_ಗಾತ್ರ(&attr, ಗಾತ್ರ_main_ರಾಶಿ);
        qurt_thread_attr_ಗಣ_ಆದ್ಯತೆ(&attr, main_prio);
        qurt_thread_attr_ಗಣ_name(&attr, "htp-main");

        ಪೂರ್ಣಾಂಕ err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        ಆದರೆ (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            ಹಿಂತಿರುಗಿ AEE_ENOMEMORY;
        }
    }

    FARF(ಮೇಲಿನ_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_ಗಾತ್ರ, ctx->vtcm_rctx, ctx->n_ಎಳೆಗಳು, ctx->thread_id, ctx->thread_prio);

    ಹಿಂತಿರುಗಿ AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    ರಚನೆ htp_handle * h = (ರಚನೆ htp_handle *) handle;
    ಆದರೆ (!h || !h->ctx) {
        ಹಿಂತಿರುಗಿ AEE_EBADPARM;
    }
    ರಚನೆ htp_context * ctx = h->ctx;

    ಆದರೆ (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2;
        (ಶೂನ್ಯ) qurt_thread_join(ctx->main_thread, &ಸ್ಥಿತಿ_2);
        ctx->main_thread = 0;
    }

    ಪೂರ್ಣಾಂಕ err = dspqueue_close(ctx->dsp_ಸರತಿ); ctx->dsp_ಸರತಿ = ಖಾಲಿ;
    ಆದರೆ (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (ಚಿಹ್ನೆರಹಿತ) err);
        ಹಿಂತಿರುಗಿ err;
    }

    work_ಸರತಿ_free(ctx->work_ಸರತಿ);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ctx->n_ಎಳೆಗಳು; i++) {
        dma_ಸರತಿ_free(ctx->dma[i]);
    }

    ಆದರೆ (ctx->hmx_ಸರತಿ) {
        hmx_ಸರತಿ_free(ctx->hmx_ಸರತಿ);
        ctx->hmx_ಸರತಿ = ಖಾಲಿ;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    ಆದರೆ (ctx->ddr_spad_ತಳ) {
        ಬಿಡುಗಡೆ(ctx->ddr_spad_ತಳ);
        ctx->ddr_spad_ತಳ = ಖಾಲಿ;
        ctx->ddr_spad_ಗಾತ್ರ = 0;
    }

    ಬಿಡುಗಡೆ(ctx);
    h->ctx = ಖಾಲಿ;

    ಹಿಂತಿರುಗಿ AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_ಎಳೆಗಳು, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_ಗಾತ್ರ) {
    (ಶೂನ್ಯ)handle;
    ಆದರೆ (!n_ಎಳೆಗಳು || !n_hvx || !n_hmx || !vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_ಎಳೆಗಳು;
    qurt_sysenv_get_max_hw_ಎಳೆಗಳು(&hw_ಎಳೆಗಳು);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    ಆದರೆ (n_hvx_val > hw_ಎಳೆಗಳು.max_hthreads) {
        n_hvx_val = hw_ಎಳೆಗಳು.max_hthreads;
    }
    ಆದರೆ (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_ಎಳೆಗಳು = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_ಗಾತ್ರ = vtcm_query_ಗಾತ್ರ();

    ಹಿಂತಿರುಗಿ AEE_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ htp_error_callback(dspqueue_t ಸರತಿ, ಪೂರ್ಣಾಂಕ error, ಶೂನ್ಯ * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (ಚಿಹ್ನೆರಹಿತ) error);
}

ರಚನೆ profile_ದತ್ತಾಂಶ {
    uint64_t usecs;
    uint64_t ಚಕ್ರಗಳು_ಆರಂಭ;
    uint64_t ಚಕ್ರಗಳು_stop;
    uint32_t pmu_counters[HEX_ಅಂಕೆ_PMU_COUNTERS];
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ profile_ಆರಂಭ(uint32_t ಬಹುಲಕ, ರಚನೆ profile_ದತ್ತಾಂಶ * d) {
    ಆಯ್ಕೆ (ಬಹುಲಕ) {
        ಸ್ಥಿತಿ HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        ಸ್ಥಿತಿ HTP_PROF_BASIC:
        ಸ್ಥಿತಿ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_ಎಣಿಕೆ();
            d->ಚಕ್ರಗಳು_ಆರಂಭ = hex_get_ಚಕ್ರಗಳು();
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ profile_stop(uint32_t ಬಹುಲಕ, ರಚನೆ profile_ದತ್ತಾಂಶ * d) {
    uint32_t pmu_counters[HEX_ಅಂಕೆ_PMU_COUNTERS];
    ಆಯ್ಕೆ (ಬಹುಲಕ) {
        ಸ್ಥಿತಿ HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < HEX_ಅಂಕೆ_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        ಸ್ಥಿತಿ HTP_PROF_BASIC:
        ಸ್ಥಿತಿ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_ಎಣಿಕೆ_ಗಮ್ಯ_us(HAP_perf_get_qtimer_ಎಣಿಕೆ() - d->usecs);
            d->ಚಕ್ರಗಳು_stop = hex_get_ಚಕ್ರಗಳು();
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ op_fence(ರಚನೆ htp_ops_context * octx) {
    ರಚನೆ htp_context *ctx = octx->ctx;
    ರಚನೆ htp_thread_trace * tr = &ctx->trace[0];
    ಸ್ಥಿರಾಂಕ uint32_t seq  = (uint32_t) octx->op_params[0];
    ಸ್ಥಿರಾಂಕ uint32_t ಬಹುಲಕ = (uint32_t) octx->op_params[1];

    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->ದತ್ತಾಂಶ;

    ಆದರೆ (ಬಹುಲಕ == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_ಗುಂಪು_barrier(octx);

        ಆದರೆ (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->ಸ್ಥಿತಿ_2);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(ಮೇಲಿನ_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->ಸ್ಥಿತಿ_2);
        ಹಿಂತಿರುಗಿ octx->ಸ್ಥಿತಿ_2;
    }

    ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = HTP_ಸ್ಥಿತಿ_OK;
    uint64_t spins = 0;
    ಆಗ (1) {
        uint32_t sync_seq;
        uint32_t sync_ಸ್ಥಿತಿ;
        htp_fence_read(sync_fence, &sync_seq, &sync_ಸ್ಥಿತಿ);
        ಆದರೆ ((int32_t)(sync_seq - seq) >= 0) {
            ಆದರೆ (sync_ಸ್ಥಿತಿ > HTP_ಸ್ಥಿತಿ_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_ಸ್ಥಿತಿ, sync_fence, seq);
                ಸ್ಥಿತಿ_2 = sync_ಸ್ಥಿತಿ;
            }
            ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            ಸ್ಥಿತಿ_2 = HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR;
            ನಿಲ್ಲಿಸು;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(ಮೇಲಿನ_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ op_mdev_ಗುಂಪು(ರಚನೆ htp_ops_context * octx) {
    ರಚನೆ htp_context * ctx = octx->ctx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.ಎಣಿಕೆ_2 = (uint16_t) sync->ne[1];
    ಆದರೆ (ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ctx->mdev.ಎಣಿಕೆ_div = init_fastdiv_ಮೌಲ್ಯಗಳು(ctx->mdev.ಎಣಿಕೆ_2);
        ctx->mdev.fence_ತಳ = (uint8_t *) sync->ದತ್ತಾಂಶ;
    }
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ execute_op(ರಚನೆ htp_ops_context * octx) {
    ಆಯ್ಕೆ (octx->op) {
        ಸ್ಥಿತಿ HTP_OP_MDEV_ಗುಂಪು:
            ಹಿಂತಿರುಗಿ op_mdev_ಗುಂಪು(octx);

        ಸ್ಥಿತಿ HTP_OP_FENCE:
            ಹಿಂತಿರುಗಿ op_fence(octx);

        ಸ್ಥಿತಿ HTP_OP_ALLREDUCE:
        ಸ್ಥಿತಿ HTP_OP_ALLREDUCE_ಕೂಡಿಸು:
            ಹಿಂತಿರುಗಿ op_allreduce(octx);

        ಸ್ಥಿತಿ HTP_OP_MUL_MAT:
        ಸ್ಥಿತಿ HTP_OP_MUL_MAT_ಕೂಡಿಸು:
            ಹಿಂತಿರುಗಿ op_matmul(octx);

        ಸ್ಥಿತಿ HTP_OP_MUL_MAT_ID:
            ಹಿಂತಿರುಗಿ op_matmul_id(octx);

        ಸ್ಥಿತಿ HTP_OP_MUL_MAT_ID_NX:
            ಹಿಂತಿರುಗಿ op_matmul_id_nx(octx);

        ಸ್ಥಿತಿ HTP_OP_MUL_MAT_NX:
            ಹಿಂತಿರುಗಿ op_matmul_nx(octx);

        ಸ್ಥಿತಿ HTP_OP_MUL:
        ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ HTP_OP_SUB:
        ಸ್ಥಿತಿ HTP_OP_DIV:
        ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು_ID:
            ಹಿಂತಿರುಗಿ op_ದ್ವಿಮಾನ(octx);

        ಸ್ಥಿತಿ HTP_OP_NORM:
        ಸ್ಥಿತಿ HTP_OP_RMS_NORM:
        ಸ್ಥಿತಿ HTP_OP_RMS_NORM_MUL:
        ಸ್ಥಿತಿ HTP_OP_SCALE:
        ಸ್ಥಿತಿ HTP_OP_CLAMP:
        ಸ್ಥಿತಿ HTP_OP_LEAKY_RELU:
        ಸ್ಥಿತಿ HTP_OP_SQR:
        ಸ್ಥಿತಿ HTP_OP_SQRT:
        ಸ್ಥಿತಿ HTP_OP_UNARY_SOFTPLUS:
        ಸ್ಥಿತಿ HTP_OP_UNARY_SIGMOID:
        ಸ್ಥಿತಿ HTP_OP_UNARY_SILU:
        ಸ್ಥಿತಿ HTP_OP_UNARY_GELU:
        ಸ್ಥಿತಿ HTP_OP_UNARY_GELU_ಶೀಘ್ರ:
        ಸ್ಥಿತಿ HTP_OP_UNARY_GELU_ERF:
        ಸ್ಥಿತಿ HTP_OP_UNARY_NEG:
        ಸ್ಥಿತಿ HTP_OP_UNARY_EXP:
        ಸ್ಥಿತಿ HTP_OP_UNARY_TANH:
        ಸ್ಥಿತಿ HTP_OP_UNARY_ABS:
        ಸ್ಥಿತಿ HTP_OP_UNARY_LOG:
        ಸ್ಥಿತಿ HTP_OP_UNARY_RELU:
        ಸ್ಥಿತಿ HTP_OP_UNARY_ಹೆಜ್ಜೆ:
        ಸ್ಥಿತಿ HTP_OP_L2_NORM:
        ಸ್ಥಿತಿ HTP_OP_HADAMARD:
            ಹಿಂತಿರುಗಿ op_unary(octx);

        ಸ್ಥಿತಿ HTP_OP_GLU_SWIGLU:
        ಸ್ಥಿತಿ HTP_OP_GLU_SWIGLU_OAI:
        ಸ್ಥಿತಿ HTP_OP_GLU_SWIGLU_CLAMP:
        ಸ್ಥಿತಿ HTP_OP_GLU_GEGLU:
        ಸ್ಥಿತಿ HTP_OP_GLU_GEGLU_ಶೀಘ್ರ:
        ಸ್ಥಿತಿ HTP_OP_GLU_GEGLU_ERF:
            ಹಿಂತಿರುಗಿ op_activations(octx);

        ಸ್ಥಿತಿ HTP_OP_SOFTMAX:
            ಹಿಂತಿರುಗಿ op_softmax(octx);

        ಸ್ಥಿತಿ HTP_OP_ROPE:
            ಹಿಂತಿರುಗಿ op_rope(octx);

        ಸ್ಥಿತಿ HTP_OP_FLASH_ATTN_EXT:
            ಹಿಂತಿರುಗಿ op_flash_attn_ext(octx);

        ಸ್ಥಿತಿ HTP_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            ಹಿಂತಿರುಗಿ op_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(octx);

        ಸ್ಥಿತಿ HTP_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            ಹಿಂತಿರುಗಿ op_get_ಅಡ್ಡಸಾಲುಗಳು(octx);

        ಸ್ಥಿತಿ HTP_OP_ಮೊತ್ತ:
            ಹಿಂತಿರುಗಿ op_ಮೊತ್ತ(octx);

        ಸ್ಥಿತಿ HTP_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            ಹಿಂತಿರುಗಿ op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(octx);

        ಸ್ಥಿತಿ HTP_OP_CPY:
        ಸ್ಥಿತಿ HTP_OP_CPY_FENCE:
            ಹಿಂತಿರುಗಿ op_cpy(octx);

        ಸ್ಥಿತಿ HTP_OP_REPEAT:
            ಹಿಂತಿರುಗಿ op_repeat(octx);

        ಸ್ಥಿತಿ HTP_OP_ARGSORT:
            ಹಿಂತಿರುಗಿ op_argsort(octx);

        ಸ್ಥಿತಿ HTP_OP_ತುದಿ_K:
            ಹಿಂತಿರುಗಿ op_ತುದಿ_k(octx);

        ಸ್ಥಿತಿ HTP_OP_ARGMAX:
            ಹಿಂತಿರುಗಿ op_argmax(octx);

        ಸ್ಥಿತಿ HTP_OP_SSM_CONV:
            ಹಿಂತಿರುಗಿ op_ssm_conv(octx);

        ಸ್ಥಿತಿ HTP_OP_CUMSUM:
            ಹಿಂತಿರುಗಿ op_cumsum(octx);

        ಸ್ಥಿತಿ HTP_OP_FILL:
            ಹಿಂತಿರುಗಿ op_fill(octx);

        ಸ್ಥಿತಿ HTP_OP_DIAG:
            ಹಿಂತಿರುಗಿ op_diag(octx);

        ಸ್ಥಿತಿ HTP_OP_SOLVE_TRI:
            ಹಿಂತಿರುಗಿ op_solve_tri(octx);

        ಸ್ಥಿತಿ HTP_OP_PAD:
            ಹಿಂತಿರುಗಿ op_pad(octx);

        ಸ್ಥಿತಿ HTP_OP_IM2_ಲಂಬಸಾಲು:
            ಹಿಂತಿರುಗಿ op_im2col(octx);

        ಸ್ಥಿತಿ HTP_OP_IM2_ಲಂಬಸಾಲು_3_D:
            ಹಿಂತಿರುಗಿ op_im2col_3d(octx);

        ಸ್ಥಿತಿ HTP_OP_ROLL:
            ಹಿಂತಿರುಗಿ op_roll(octx);

        ಸ್ಥಿತಿ HTP_OP_POOL_2D:
            ಹಿಂತಿರುಗಿ op_pool_2d(octx);

        ಸ್ಥಿತಿ HTP_OP_POOL_1D:
            ಹಿಂತಿರುಗಿ op_pool_1d(octx);

        ಸ್ಥಿತಿ HTP_OP_CONCAT:
            ಹಿಂತಿರುಗಿ op_concat(octx);

        ಸ್ಥಿತಿ HTP_OP_GATED_DELTA_NET:
            ಹಿಂತಿರುಗಿ op_gated_delta_net(octx);

        ಸ್ಥಿತಿ HTP_OP_TRI:
            ಹಿಂತಿರುಗಿ op_unary(octx);

        ಸ್ಥಿತಿ HTP_OP_INVALID:
            ನಿಲ್ಲಿಸು;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool reuse_buf(ರಚನೆ htp_context *ctx, uint32_t *m_reuse, ರಚನೆ htp_buf_desc *b) {
    b->ತಳ = ಖಾಲಿ;

    ಗಾಗಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ರಚನೆ htp_mmap *m = ctx->mmap + i;
        ಆದರೆ (m->ಗಾತ್ರ_2 && m->fd == b->fd && m->flags == b->flags) {
            b->ತಳ   = m->ತಳ;
            *m_reuse |= (1 << i);
            ಹಿಂತಿರುಗಿ true;
        }
    }

    ಹಿಂತಿರುಗಿ false;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ drop_mmap(ರಚನೆ htp_context *ctx, ರಚನೆ htp_mmap *m) {
    ಆದರೆ (m->ಗಾತ್ರ_2) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) m->ತಳ, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) m->ಗಾತ್ರ_2);
        htp_munmap(m->ತಳ, m->ಗಾತ್ರ_2, m->flags);
        m->ಗಾತ್ರ_2 = 0;
        m->ತಳ = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool mmap_buf(ರಚನೆ htp_context *ctx, ರಚನೆ htp_buf_desc *b) {
    ಆದರೆ (b->ತಳ) ಹಿಂತಿರುಗಿ true; // already mapped

    // find unused mapping
    ಗಾಗಿ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        ರಚನೆ htp_mmap *m = &ctx->mmap[i];
        ಆದರೆ (!m->ಗಾತ್ರ_2) {
            uint64_t va = htp_mmap(b->fd, b->ಗಾತ್ರ_2, b->flags);
            ಆದರೆ (va == 0) {
                FARF(ಮೇಲಿನ_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) b->ಗಾತ್ರ_2);
                ಹಿಂತಿರುಗಿ false;
            }

            m->ತಳ   = b->ತಳ = va;
            m->fd     = b->fd;
            m->ಗಾತ್ರ_2   = b->ಗಾತ್ರ_2;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) m->ತಳ, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) m->ಗಾತ್ರ_2, m->flags);
            ಹಿಂತಿರುಗಿ true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    ಹಿಂತಿರುಗಿ false;
}

ಸ್ಥಿರ ಶೂನ್ಯ prep_op_bufs(ರಚನೆ htp_context *ctx, ರಚನೆ htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    ಗಾಗಿ (uint32_t i=0; i < n_bufs; i++) {
        ರಚನೆ htp_buf_desc *b = bufs + i;
        ಆದರೆ (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->ಗಾತ್ರ_2;
        }
        FARF(ಮೇಲಿನ_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) b->ತಳ, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) b->ಗಾತ್ರ_2, b->flags);
    }

    ಆದರೆ (b_reuse == n_bufs) ಹಿಂತಿರುಗಿ; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    ಗಾಗಿ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ಆದರೆ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].ಗಾತ್ರ_2;
        }
    }

    FARF(ಮೇಲಿನ_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    ಆದರೆ ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        ಗಾಗಿ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool ಬಳಸಿದ = m_reuse & (1<<i);
            ಆದರೆ (!ಬಳಸಿದ && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    ಗಾಗಿ (uint32_t i=0; i < n_bufs; i++) {
        ರಚನೆ htp_buf_desc *b = bufs + i;
        ಆದರೆ (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            ನಿಲ್ಲಿಸು;
        }
        FARF(ಮೇಲಿನ_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) b->ತಳ, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) b->ಗಾತ್ರ_2, b->flags);
    }

    ಆದರೆ (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(ಮೇಲಿನ_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        ಗಾಗಿ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            ಆದರೆ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        ಗಾಗಿ (uint32_t i=0; i < n_bufs; i++) {
            ರಚನೆ htp_buf_desc *b = bufs + i;
            ಆದರೆ (!(b->flags & HTP_BUF_EXTENDED)) {
                b->ತಳ = 0;
            }
            ಆದರೆ (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) b->ಗಾತ್ರ_2);
                abort();
            }
            FARF(ಮೇಲಿನ_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) b->ತಳ, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) b->ಗಾತ್ರ_2, b->flags);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ prep_tensor(ರಚನೆ htp_context *ctx, ರಚನೆ htp_buf_desc *bufs, ರಚನೆ htp_tensor *tens, uint32_t idx, ರಚನೆ htp_tensor *t) {
    uint64_t ಸ್ಥಾನಾಂತರ = t->ದತ್ತಾಂಶ;
    uint32_t bi     = t->bi;

    t->ದತ್ತಾಂಶ  = bufs[bi].ತಳ + ಸ್ಥಾನಾಂತರ;  // update data to the actual pointer

    FARF(ಮೇಲಿನ_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) ಸ್ಥಾನಾಂತರ, t->ಗಾತ್ರ_2, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) t->ದತ್ತಾಂಶ,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

ಸ್ಥಿರ ಶೂನ್ಯ prep_tensors(ರಚನೆ htp_context *ctx, ರಚನೆ htp_buf_desc *bufs, ರಚನೆ htp_tensor *tens, uint32_t n_tens) {
    ಗಾಗಿ (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mdev_ಗುಂಪು_init(ರಚನೆ htp_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, ಗಾತ್ರ(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ proc_op_req(ರಚನೆ htp_ops_context * octx, ರಚನೆ htp_buf_desc * bufs, uint32_t n_bufs,
                       ರಚನೆ htp_tensor * tens, uint32_t idx, ರಚನೆ htp_op_desc * op) {
    ಸ್ಮರಣೆ_ನಕಲು(octx->op_params,     op->params, ಗಾತ್ರ(octx->op_params));
    ಸ್ಮರಣೆ_ನಕಲು(octx->kernel_params, op->kernel_params, ಗಾತ್ರ(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_ಎಳೆಗಳು     = octx->ctx->n_ಎಳೆಗಳು;
    octx->n_ಎಳೆಗಳು_div = octx->ctx->n_ಎಳೆಗಳು_div;

    FARF(ಮೇಲಿನ_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    ಗಾಗಿ (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        ಆದರೆ (src_idx == 0xffff) {
            octx->src[i]     = ಖಾಲಿ;
            ಮುಂದುವರಿ;
        }

        ರಚನೆ htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(ಮೇಲಿನ_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) src->ದತ್ತಾಂಶ, src->ಗಾತ್ರ_2,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    ಗಾಗಿ (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        ಆದರೆ (dst_idx == 0xffff) {
            octx->dsts[i]    = ಖಾಲಿ;
            ಮುಂದುವರಿ;
        }
        ರಚನೆ htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(ಮೇಲಿನ_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) dst->ದತ್ತಾಂಶ, dst->ಗಾತ್ರ_2,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_ಗುಂಪು_barrier(octx);

    ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = execute_op(octx);

    htp_ops_context_ಗಣ_ಸ್ಥಿತಿ(octx, ಸ್ಥಿತಿ_2);

    octx->src0_spad.src = ಖಾಲಿ;
    octx->src1_spad.src = ಖಾಲಿ;
    octx->src2_spad.src = ಖಾಲಿ;
    octx->src3_spad.src = ಖಾಲಿ;
    octx->dst_spad.src  = ಖಾಲಿ;

    ಹಿಂತಿರುಗಿ octx->ಸ್ಥಿತಿ_2;
}

ಸ್ಥಿರ ಶೂನ್ಯ process_opbatch(ರಚನೆ htp_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_opbatch_req * req, ಸ್ಥಿರಾಂಕ ರಚನೆ dspqueue_ಬಫರ್ * dbuf) {
    dspqueue_t ಸರತಿ = ctx->dsp_ಸರತಿ;
    ಪೂರ್ಣಾಂಕ err;

    ಸ್ಥಿರಾಂಕ uint32_t n_bufs = req->n_bufs;
    ಸ್ಥಿರಾಂಕ uint32_t n_tens = req->n_tensors;
    ಸ್ಥಿರಾಂಕ uint32_t n_ops  = req->n_ops;

    ಸ್ಥಿರಾಂಕ uint32_t b_ಗಾತ್ರ = ಗಾತ್ರ(ರಚನೆ htp_buf_desc)  * n_bufs;
    ಸ್ಥಿರಾಂಕ uint32_t t_ಗಾತ್ರ = ಗಾತ್ರ(ರಚನೆ htp_tensor)    * n_tens;
    ಸ್ಥಿರಾಂಕ uint32_t o_ಗಾತ್ರ = ಗಾತ್ರ(ರಚನೆ htp_op_desc)   * n_ops;
    ಸ್ಥಿರಾಂಕ uint32_t p_ಗಾತ್ರ = ಗಾತ್ರ(ರಚನೆ htp_prof_desc) * n_ops;
    ಸ್ಥಿರಾಂಕ uint32_t tr_ಗಾತ್ರ = (HTP_MAX_NTHREADS + 1) * req->n_traces * ಗಾತ್ರ(ರಚನೆ htp_trace_desc);

    ಆದರೆ (dbuf->ಗಾತ್ರ_2 < b_ಗಾತ್ರ + t_ಗಾತ್ರ + o_ಗಾತ್ರ + p_ಗಾತ್ರ + tr_ಗಾತ್ರ) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->ಗಾತ್ರ_2, b_ಗಾತ್ರ + t_ಗಾತ್ರ + o_ಗಾತ್ರ + p_ಗಾತ್ರ + tr_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ;
    }

    FARF(ಮೇಲಿನ_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->ಗಾತ್ರ_2, b_ಗಾತ್ರ, t_ಗಾತ್ರ, o_ಗಾತ್ರ);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    ರಚನೆ htp_buf_desc* bufs = (ರಚನೆ htp_buf_desc*)  m_ptr; m_ptr += b_ಗಾತ್ರ;
    ರಚನೆ htp_tensor*   tens = (ರಚನೆ htp_tensor*)    m_ptr; m_ptr += t_ಗಾತ್ರ;
    ರಚನೆ htp_op_desc*   ops = (ರಚನೆ htp_op_desc*)   m_ptr; m_ptr += o_ಗಾತ್ರ;
    ರಚನೆ htp_prof_desc* pds = (ರಚನೆ htp_prof_desc*) m_ptr;

    ರಚನೆ profile_ದತ್ತಾಂಶ batch_prof;
    profile_ಆರಂಭ(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, ಗಾತ್ರ(ctx->trace));
    ಆದರೆ (ctx->profiler == HTP_PROF_TRACE) {
        ರಚನೆ htp_trace_desc * trace_events = (ರಚನೆ htp_trace_desc *) (m_ptr + p_ಗಾತ್ರ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_ಆರಂಭ(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_ಖಂಡ(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, ಗಾತ್ರ(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_ಆರಂಭ(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    ರಚನೆ htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, ಗಾತ್ರ(*octx));
    octx->n_ಎಳೆಗಳು     = ctx->n_ಎಳೆಗಳು;
    octx->n_ಎಳೆಗಳು_div = ctx->n_ಎಳೆಗಳು_div;
    octx->ctx           = ctx;

    mdev_ಗುಂಪು_init(ctx, req);

    work_ಸರತಿ_wakeup(ctx->work_ಸರತಿ);
    ಆದರೆ (ctx->hmx_ಸರತಿ) {
        hmx_ಸರತಿ_wakeup(ctx->hmx_ಸರತಿ);
    }

    ಪೂರ್ಣಾಂಕ op_ಸ್ಥಿತಿ = HTP_ಸ್ಥಿತಿ_OK;
    octx->ಸ್ಥಿತಿ_2  = HTP_ಸ್ಥಿತಿ_OK;
    ಗಾಗಿ (uint32_t i = 0; i < n_ops; i++) {
        ರಚನೆ profile_ದತ್ತಾಂಶ prof;

        profile_ಆರಂಭ(ctx->profiler, &prof);

        op_ಸ್ಥಿತಿ = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_ಗಣ_ಸ್ಥಿತಿ(octx, op_ಸ್ಥಿತಿ);

        ಆದರೆ (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].ಚಕ್ರಗಳು_ಆರಂಭ = prof.ಚಕ್ರಗಳು_ಆರಂಭ;
            pds[i].ಚಕ್ರಗಳು_stop = prof.ಚಕ್ರಗಳು_stop;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < HEX_ಅಂಕೆ_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    ಆದರೆ (ctx->hmx_ಸರತಿ) {
        hmx_ಸರತಿ_suspend(ctx->hmx_ಸರತಿ);
        hmx_ಸರತಿ_flush(ctx->hmx_ಸರತಿ);
    }
    work_ಸರತಿ_suspend(ctx->work_ಸರತಿ);

    // Flush remaining dirty tensors at the end of the batch
    htp_trace_event_ಆರಂಭ(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_mdev_ಗುಂಪು_barrier(octx);

    profile_stop(HTP_PROF_BASIC, &batch_prof);

    ರಚನೆ htp_opbatch_rsp rsp;
    memset(&rsp, 0, ಗಾತ್ರ(rsp));
    rsp.seq          = req->seq;
    rsp.ಸ್ಥಿತಿ_2       = octx->ಸ್ಥಿತಿ_2;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.ಚಕ್ರಗಳು_ಆರಂಭ = batch_prof.ಚಕ್ರಗಳು_ಆರಂಭ;
    rsp.ಚಕ್ರಗಳು_stop  = batch_prof.ಚಕ್ರಗಳು_stop;

    ಆದರೆ (ctx->profiler == HTP_PROF_TRACE) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].ಎಣಿಕೆ_2;
        }
    }

    ರಚನೆ dspqueue_ಬಫರ್ write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_ಬಫರ್_FLAG_FLUSH_SENDER | DSPQUEUE_ಬಫರ್_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(ಸರತಿ, 0, 1, &write_dbuf, ಗಾತ್ರ(rsp), (ಸ್ಥಿರಾಂಕ uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    ಆದರೆ (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (ಚಿಹ್ನೆರಹಿತ) err);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು DSPQUEUE_READ_TIMEOUT_USEC 5000
#ವ್ಯಾಖ್ಯಾನಿಸು DSPQUEUE_POLL_TIMEOUT_USEC 100
#ವ್ಯಾಖ್ಯಾನಿಸು DSPQUEUE_POLL_ಎಣಿಕೆ        100

ಸ್ಥಿರ ಶೂನ್ಯ process_ops(ರಚನೆ htp_context * ctx) {
    dspqueue_t ಸರತಿ = ctx->dsp_ಸರತಿ;
    ಪೂರ್ಣಾಂಕ err;

    uint32_t poll_ಎಣಿಕೆ = DSPQUEUE_POLL_ಎಣಿಕೆ;

    vtcm_acquire(ctx);

    ಆಗ (!atomic_ತುಂಬು(&ctx->vtcm_needs_release) && !atomic_ತುಂಬು(&ctx->killed)) {
        ರಚನೆ htp_opbatch_req req;
        uint32_t r_ಗಾತ್ರ = ಗಾತ್ರ(req);

        ರಚನೆ dspqueue_ಬಫರ್ dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(ಸರತಿ, &flags, n_dbufs, &n_dbufs, &dbuf, r_ಗಾತ್ರ, &r_ಗಾತ್ರ, (uint8_t *) &req);
        ಆದರೆ (err == AEE_EWOULDBLOCK) {
            ಆದರೆ (--poll_ಎಣಿಕೆ) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                ಮುಂದುವರಿ;
            }
            ನಿಲ್ಲಿಸು;
        }

        ಆದರೆ (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (ಚಿಹ್ನೆರಹಿತ) err);
            ನಿಲ್ಲಿಸು;
        }

        ಆದರೆ (r_ಗಾತ್ರ < ಗಾತ್ರ(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_ಗಾತ್ರ, n_dbufs);
            ಮುಂದುವರಿ;
        }

        // Reset poll count for valid requests
        poll_ಎಣಿಕೆ = DSPQUEUE_POLL_ಎಣಿಕೆ;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ htp_packet_callback(dspqueue_t ಸರತಿ, ಪೂರ್ಣಾಂಕ error, ಶೂನ್ಯ * context) {
    (ಶೂನ್ಯ) ಸರತಿ;
    (ಶೂನ್ಯ) error;
    ರಚನೆ htp_handle * h = (ರಚನೆ htp_handle *) context;
    ಆದರೆ (h && h->ctx) {
        process_ops(h->ctx);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ htp_main_thread(ಶೂನ್ಯ * context) {
    ರಚನೆ htp_context * ctx = (ರಚನೆ htp_context *) context;

    FARF(ಮೇಲಿನ_2, "htp-main-thread: started");

    ಆಗ (!atomic_ತುಂಬು(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t ಅಂಕೆ_buffers = 0;
        uint32_t ಸಂದೇಶ_ಉದ್ದಳತೆ = 0;

        ಪೂರ್ಣಾಂಕ err = dspqueue_peek(ctx->dsp_ಸರತಿ, &flags, &ಅಂಕೆ_buffers, &ಸಂದೇಶ_ಉದ್ದಳತೆ, 50000);
        ಆದರೆ (err == 0) {
            process_ops(ctx);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            ಮುಂದುವರಿ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (ಚಿಹ್ನೆರಹಿತ) err);
            ನಿಲ್ಲಿಸು;
        }
    }

    FARF(ಮೇಲಿನ_2, "htp-main-thread: stopped");
}
