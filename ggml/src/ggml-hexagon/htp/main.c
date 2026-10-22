#ਹਦਾਇਤ clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <AEEStdErr.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <dspqueue.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_compute_res.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_etm_config.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_mem.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_power.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_ps.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_dcvs.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_thread.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_memory.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <remote.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdatomic.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-queue.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-bitmap.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-dcache.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp_iface.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "work-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "allreduce-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-fence.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_ਕਤਾਰ_ਸਮਰੱਥਾ     128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_ਕਤਾਰ_ਢੇਰ_ਮਾਪ   16384
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WORK_ਕਤਾਰ_ਸਮਰੱਥਾ    16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WORK_ਕਤਾਰ_ਢੇਰ_ਮਾਪ  16384
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAIN_THREAD_ਢੇਰ_ਮਾਪ 32768

_Static_assert(WORK_ਕਤਾਰ_MAX_N_ਧਾਗੇ >= HTP_MAX_NTHREADS,
               "work-queue thread cap must be >= HTP_MAX_NTHREADS");

ਬਣਤਰ htp_handle {
    ਬਣਤਰ htp_context * ctx;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t htp_mmap(uint32_t fd, uint64_t ਮਾਪ, uint32_t flags) {
#ਜੇ __HVX_ARCH__ > 79
    ਜੇ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, ਆਕਾਰ(payload));
        payload.request_id = HAP_MEM_MAP;
        payload.mmap.len   = ਮਾਪ;
        payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
        payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
        payload.mmap.fd    = fd;

        ਜੇ (HAP_mem_request(&payload) != 0) {
            FARF(ERROR, "extended mmap failed : fd %u size %llu", fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) ਮਾਪ);
            ਵਾਪਸ 0;
        }

        ਵਾਪਸ payload.mmap.dsp_va;
    }
#ਨਹੀਂਤਾਂ
    ਜੇ (flags & HTP_BUF_EXTENDED) {
        FARF(ERROR, "extended mmap is unsupported on v%d", __HVX_ARCH__);
        ਵਾਪਸ 0;
    }
#ਜੇ_ਅੰਤ

    ਜੇ (ਮਾਪ > UINT32_MAX) {
        FARF(ERROR, "mmap failed : size %llu exceeds 32-bit limit", (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) ਮਾਪ);
        ਵਾਪਸ 0;
    }

    ਖਾਲੀ * va = (ਖਾਲੀ *)-1;
    ਲਈ (ਪੂਰਨ retry = 0; retry < 2; retry++) {
#ਜੇ __HVX_ARCH__ > 73
        va = HAP_mmap2(ਨਿਰਾ, (size_t) ਮਾਪ, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#ਨਹੀਂਤਾਂ
        ਜੇ (ਮਾਪ > HTP_MMAP_MAX_VMEM) {
            FARF(ERROR, "mmap failed : size %llu exceeds 2GB limit for HAP_mmap", (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) ਮਾਪ);
            abort();
        }
        va = HAP_mmap(ਨਿਰਾ, (ਪੂਰਨ) ਮਾਪ, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#ਜੇ_ਅੰਤ
        ਜੇ (va != (ਖਾਲੀ *)-1 && va != ਨਿਰਾ) {
            ਵਾਪਸ (uint64_t) (uintptr_t) va;
        }
        ਜੇ (retry == 0) {
            FARF(ਉੱਚਾ_2, "mmap failed first try (va %p fd %u size %llu), retrying...", va, fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) ਮਾਪ);
        }
    }
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_munmap(uint64_t va, uint64_t ਮਾਪ, uint32_t flags) {
#ਜੇ __HVX_ARCH__ > 79
    ਜੇ (flags & HTP_BUF_EXTENDED) {
        HAP_mem_req_payload_t payload;
        memset(&payload, 0, ਆਕਾਰ(payload));
        payload.request_id   = HAP_MEM_UNMAP;
        payload.munmap.dsp_va = va;
        payload.munmap.len    = ਮਾਪ;
        HAP_mem_request(&payload);
        ਵਾਪਸ;
    }
#ਜੇ_ਅੰਤ

#ਜੇ __HVX_ARCH__ > 73
    HAP_munmap2((ਖਾਲੀ *) (uintptr_t) va, (size_t) ਮਾਪ);
#ਨਹੀਂਤਾਂ
    HAP_munmap((ਖਾਲੀ *) (uintptr_t) va, (ਪੂਰਨ) ਮਾਪ);
#ਜੇ_ਅੰਤ
}

AEEResult htp_iface_open(ਅਚਲ ਅੱਖਰ * uri, remote_handle64 * handle) {
    (ਖਾਲੀ) uri;
    ਬਣਤਰ htp_handle * h = ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(1, ਆਕਾਰ(*h));
    ਜੇ (h == ਨਿਰਾ) {
        ਵਾਪਸ AEE_ENOMEMORY;
    }

    *handle = (remote_handle64) h;
    ਵਾਪਸ AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    ਬਣਤਰ htp_handle * h = (ਬਣਤਰ htp_handle *) handle;
    ਜੇ (!h) {
        ਵਾਪਸ AEE_EBADPARM;
    }

    ਪੂਰਨ err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    ਜੇ (err) {
        ਜੇ (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } ਨਹੀਂਤਾਂ {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    ਵਾਪਸ err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t ਬਹੁਲਕ, ਅਚਲ htp_iface_pmu_conf* pmu_conf) {
    ਬਣਤਰ htp_handle * h = (ਬਣਤਰ htp_handle *) handle;
    ਜੇ (!h || !h->ctx) {
        ਵਾਪਸ AEE_EBADPARM;
    }
    ਬਣਤਰ htp_context * ctx = h->ctx;

    ਜੇ (ਬਹੁਲਕ == HTP_PROF_PMU) {
        ਅਚਲ uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        ਲਈ (; i < HEX_ਅੰਕ_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        ਲਈ (i = 0; i < HEX_ਅੰਕ_PMU_COUNTERS; i++) {
            cfg |= (((events[i] >> 8) & 3) << (i * 2));
        }

        FARF(ALWAYS, "Configuring PMU registers: evtcfg = 0x%x, evtcfg1 = 0x%x, pmucfg = 0x%x", evtcfg, evtcfg1, cfg);

        // Configure PMU registers
        qurt_pmu_ਸੈੱਟ(QURT_PMUCFG,     cfg);
        qurt_pmu_ਸੈੱਟ(QURT_PMUEVTCFG,  evtcfg);
        qurt_pmu_ਸੈੱਟ(QURT_PMUEVTCFG1, evtcfg1);
        qurt_pmu_enable(1);
    }

    ctx->profiler = ਬਹੁਲਕ;

    ਵਾਪਸ AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    ਬਣਤਰ htp_handle * h = (ਬਣਤਰ htp_handle *) handle;
    ਜੇ (!h) {
        ਵਾਪਸ AEE_EBADPARM;
    }

    ਬਣਤਰ htp_context * ctx = h->ctx;
    ਜੇ (ctx) {
        ਜੇ (ctx->dsp_ਕਤਾਰ) {
            FARF(ERROR, "Closing handle with queue still open");
            ਵਾਪਸ AEE_EITEMBUSY;
        }

        // release the mmaps (if any)
        ਲਈ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
            ਜੇ (ctx->mmap[i].ਮਾਪ) {
                htp_munmap(ctx->mmap[i].ਅਧਾਰ, ctx->mmap[i].ਮਾਪ, ctx->mmap[i].flags);
                ctx->mmap[i].ਮਾਪ = 0;
                ctx->mmap[i].ਅਧਾਰ = 0;
                ctx->mmap[i].fd   = -1;
                ctx->mmap[i].flags = 0;
            }
        }

        ਜੇ (ctx->profiler) {
            qurt_pmu_enable(1);
        }

        ਜੇ (ctx->etm) {
            HAP_user_etm_disable();
        }

        // Free the unified block (ctx is the base address of the block)
        ਮੁਕਤ(ctx);
        h->ctx = ਨਿਰਾ;
    }

    ਮੁਕਤ(h);
    ਵਾਪਸ AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint64_t ਮਾਪ) {
    ਬਣਤਰ htp_handle * h = (ਬਣਤਰ htp_handle *) handle;
    ਜੇ (!h || !h->ctx) {
        ਵਾਪਸ AEE_EBADPARM;
    }
    ਬਣਤਰ htp_context * ctx = h->ctx;

    // See if we already have this mapping
    ਲਈ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ਬਣਤਰ htp_mmap *m = &ctx->mmap[i];
        ਜੇ (m->fd == fd) {
            ਵਾਪਸ AEE_SUCCESS;
        }
    }

    // Add new mapping
    ਲਈ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ਬਣਤਰ htp_mmap *m = &ctx->mmap[i];
        ਜੇ (!m->ਮਾਪ) {
            FARF(ਉੱਚਾ_2, "mmap : fd %u size %llu", fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) ਮਾਪ);
            uint64_t va = htp_mmap(fd, ਮਾਪ, 0);
            ਜੇ (va == 0) {
                FARF(ERROR, "mmap failed : fd %u size %llu", fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) ਮਾਪ);
                ਵਾਪਸ AEE_EFAILED;
            }

            m->ਅਧਾਰ   = va;
            m->fd     = fd;
            m->ਮਾਪ   = ਮਾਪ;
            m->flags  = 0;

            ਵਾਪਸ AEE_SUCCESS;
        }
    }

    ਵਾਪਸ AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    ਬਣਤਰ htp_handle * h = (ਬਣਤਰ htp_handle *) handle;
    ਜੇ (!h || !h->ctx) {
        ਵਾਪਸ AEE_EBADPARM;
    }
    ਬਣਤਰ htp_context * ctx = h->ctx;

    ਲਈ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ਬਣਤਰ htp_mmap *m = &ctx->mmap[i];
        ਜੇ (fd < 0 || m->fd == fd) {
            FARF(ਉੱਚਾ_2, "unmmap : base 0x%llx fd %u size %llu", (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) m->ਅਧਾਰ, m->fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) m->ਮਾਪ);
            htp_munmap(m->ਅਧਾਰ, m->ਮਾਪ, m->flags);
            m->ਮਾਪ   = 0;
            m->ਅਧਾਰ   = ਨਿਰਾ;
            m->fd     = -1;
            m->flags  = 0;
        }
    }

    ਵਾਪਸ AEE_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ vtcm_acquire(ਬਣਤਰ htp_context * ctx) {
    ਜੇ (!atomic_ਲੱਦੋ(&ctx->vtcm_valid)) {
        ਪੂਰਨ err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 10000000u);
        ਜੇ (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (ਬਿਨਾਨਿਸ਼ਾਨੀ)err);
            abort();
        }

        atomic_store(&ctx->vtcm_needs_release, false);
        atomic_store(&ctx->vtcm_valid, true);

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_ਤਰਜੀਹ(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

ਸਥਿਰ ਖਾਲੀ vtcm_release(ਬਣਤਰ htp_context * ctx) {
    ਜੇ (atomic_ਲੱਦੋ(&ctx->vtcm_valid)) {
        atomic_store(&ctx->vtcm_valid, false);
        atomic_store(&ctx->vtcm_needs_release, false);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

ਸਥਿਰ ਪੂਰਨ vtcm_release_callback(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ rctx, ਖਾਲੀ * state) {
    ਬਣਤਰ htp_context * ctx = (ਬਣਤਰ htp_context *) state;
    atomic_store(&ctx->vtcm_needs_release, true);
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ vtcm_query_ਮਾਪ(ਖਾਲੀ) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ vtcm_ਮਾਪ = 8 * 1024 * 1024;  // 8MB default
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ vtcm_avail;
    compute_res_vtcm_ਪੰਨਾ_t ਕੁੱਲ_layout, avail_layout;
    // some images (e.g. FuSa cDSP) fail the query when the layout out-params are NULL
    HAP_compute_res_query_VTCM(0, &vtcm_ਮਾਪ, &ਕੁੱਲ_layout, &vtcm_avail, &avail_layout);
    ਵਾਪਸ vtcm_ਮਾਪ;
}

ਸਥਿਰ ਪੂਰਨ vtcm_alloc(ਬਣਤਰ htp_context * ctx) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ vtcm_ਮਾਪ = vtcm_query_ਮਾਪ();

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_ਸੈੱਟ_serialize(&attr, 0);
    HAP_compute_res_attr_ਸੈੱਟ_cache_ਬਹੁਲਕ(&attr, 1);
    HAP_compute_res_attr_ਸੈੱਟ_vtcm_param_v2(&attr, vtcm_ਮਾਪ, vtcm_ਮਾਪ, vtcm_ਮਾਪ); // single page
    HAP_compute_res_attr_ਸੈੱਟ_release_callback(&attr, vtcm_release_callback, (ਖਾਲੀ *) ctx);
    HAP_compute_res_attr_ਸੈੱਟ_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    ਜੇ (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_ਮਾਪ);
        ਵਾਪਸ AEE_ENOMEMORY;
    }

    ਖਾਲੀ * vtcm_ptr;
    ਜੇ (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_ਮਾਪ) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_ਮਾਪ);
        ਵਾਪਸ AEE_ENOMEMORY;
    }

    ctx->vtcm_ਅਧਾਰ          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_ਮਾਪ          = vtcm_ਮਾਪ;
    ctx->vtcm_rctx          = rctx;
    atomic_store(&ctx->vtcm_valid, false);
    atomic_store(&ctx->vtcm_needs_release, false);

    ਵਾਪਸ 0;
}

ਸਥਿਰ ਖਾਲੀ vtcm_free(ਬਣਤਰ htp_context * ctx) {
    ਜੇ (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_ਅਧਾਰ = 0;
        ctx->vtcm_rctx = 0;
    }
}

ਸਥਿਰ ਖਾਲੀ htp_main_thread(ਖਾਲੀ * context);
ਸਥਿਰ ਖਾਲੀ htp_packet_callback(dspqueue_t ਕਤਾਰ, ਪੂਰਨ error, ਖਾਲੀ * context);
ਸਥਿਰ ਖਾਲੀ htp_error_callback(dspqueue_t ਕਤਾਰ, ਪੂਰਨ error, ਖਾਲੀ * context);

// Non-BURST modes leave DCVS enabled and keep the HTP awake.
// BURST pins all corners to MAX and disables DCVS.
// Corner constants are from HAP_power.h (HAP_dcvs_voltage_corner_t / HAP_dcvs_exp_vcorner_t).
ਬਣਤਰ htp_power_profile {
    boolean  dcvs_enable;
    uint32_t bus_min_ਕੋਨਾ;
    uint32_t bus_max_ਕੋਨਾ;
    uint32_t bus_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ;
    uint32_t core_min_ਕੋਨਾ;
    uint32_t core_max_ਕੋਨਾ;
    uint32_t core_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ;
    boolean  sleep_disable;
    uint32_t hmx_min_ਕੋਨਾ;    // only used when __HVX_ARCH__ >= 75
    uint32_t hmx_max_ਕੋਨਾ;
    uint32_t hmx_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ;
    uint32_t hmx_perf_ਬਹੁਲਕ;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_POWER_PROFILE_ਇੰਦਰਾਜ(min_c, max_c, hmx_c) \
    { \
        .dcvs_enable        = TRUE, \
        .bus_min_ਕੋਨਾ     = (min_c), \
        .bus_max_ਕੋਨਾ     = (max_c), \
        .bus_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ  = (max_c), \
        .core_min_ਕੋਨਾ    = (min_c), \
        .core_max_ਕੋਨਾ    = (max_c), \
        .core_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ = (max_c), \
        .sleep_disable      = TRUE, \
        .hmx_min_ਕੋਨਾ     = (hmx_c), \
        .hmx_max_ਕੋਨਾ     = (hmx_c), \
        .hmx_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ  = (hmx_c), \
        .hmx_perf_ਬਹੁਲਕ      = HAP_CLK_PERF_ਉੱਚਾ, \
    }

ਸਥਿਰ ਅਚਲ ਬਣਤਰ htp_power_profile htp_power_profiles[HTP_POWER_ਬਹੁਲਕ_ਗਿਣਤੀ] = {
    [HTP_POWER_ਬਹੁਲਕ_ਨੀਵਾਂ_POWER_SAVER] =
        HTP_POWER_PROFILE_ਇੰਦਰਾਜ(HAP_DCVS_VCORNER_SVS2, HAP_DCVS_VCORNER_SVS2, HAP_DCVS_EXP_VCORNER_MIN_SVS),
    [HTP_POWER_ਬਹੁਲਕ_POWER_SAVER] =
        HTP_POWER_PROFILE_ਇੰਦਰਾਜ(HAP_DCVS_VCORNER_SVS, HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_EXP_VCORNER_SVS),
    [HTP_POWER_ਬਹੁਲਕ_ਉੱਚਾ_POWER_SAVER] =
        HTP_POWER_PROFILE_ਇੰਦਰਾਜ(HAP_DCVS_VCORNER_SVS_PLUS, HAP_DCVS_VCORNER_NOM, HAP_DCVS_EXP_VCORNER_SVS_L2),
    [HTP_POWER_ਬਹੁਲਕ_ਨੀਵਾਂ_ਸੰਤੁਲਿਤ] =
        HTP_POWER_PROFILE_ਇੰਦਰਾਜ(HAP_DCVS_VCORNER_NOM, HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_EXP_VCORNER_NOM),
    [HTP_POWER_ਬਹੁਲਕ_ਸੰਤੁਲਿਤ] =
        HTP_POWER_PROFILE_ਇੰਦਰਾਜ(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_EXP_VCORNER_NOM_L1),
    [HTP_POWER_ਬਹੁਲਕ_ਉੱਚਾ_PERFORMANCE] =
        HTP_POWER_PROFILE_ਇੰਦਰਾਜ(HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_ਬਹੁਲਕ_SUSTAINED_ਉੱਚਾ_PERFORMANCE] =
        HTP_POWER_PROFILE_ਇੰਦਰਾਜ(HAP_DCVS_VCORNER_NOM_PLUS, HAP_DCVS_VCORNER_TURBO_PLUS, HAP_DCVS_EXP_VCORNER_TUR_L2),
    [HTP_POWER_ਬਹੁਲਕ_ਮਿਆਦ] = {
        .dcvs_enable        = FALSE,
        .bus_min_ਕੋਨਾ     = HAP_DCVS_VCORNER_MAX,
        .bus_max_ਕੋਨਾ     = HAP_DCVS_VCORNER_MAX,
        .bus_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ  = HAP_DCVS_VCORNER_MAX,
        .core_min_ਕੋਨਾ    = HAP_DCVS_VCORNER_MAX,
        .core_max_ਕੋਨਾ    = HAP_DCVS_VCORNER_MAX,
        .core_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ = HAP_DCVS_VCORNER_MAX,
        .sleep_disable      = TRUE,
        .hmx_min_ਕੋਨਾ     = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_max_ਕੋਨਾ     = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ  = HAP_DCVS_EXP_VCORNER_MAX,
        .hmx_perf_ਬਹੁਲਕ      = HAP_CLK_PERF_ਉੱਚਾ,
    },
};

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HTP_POWER_PROFILE_ਇੰਦਰਾਜ

AEEResult htp_iface_ਸ਼ੁਰੂ(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_ਕਤਾਰ_id, uint32_t n_hvx, uint32_t n_hmx, uint64_t max_vmem, uint32_t power_ਬਹੁਲਕ) {
    ਬਣਤਰ htp_handle * h = (ਬਣਤਰ htp_handle *) handle;
    ਜੇ (!h) {
        ਵਾਪਸ AEE_EBADPARM;
    }

    ਜੇ (power_ਬਹੁਲਕ >= HTP_POWER_ਬਹੁਲਕ_ਗਿਣਤੀ) {
        FARF(ERROR, "invalid power-mode %u, falling back to burst", (ਬਿਨਾਨਿਸ਼ਾਨੀ) power_ਬਹੁਲਕ);
        power_ਬਹੁਲਕ = HTP_POWER_ਬਹੁਲਕ_ਮਿਆਦ;
    }
    ਅਚਲ ਬਣਤਰ htp_power_profile * pwr = &htp_power_profiles[power_ਬਹੁਲਕ];
    FARF(ALWAYS, "power-mode %u : dcvs-enable %d bus[%u,%u,%u] core[%u,%u,%u] sleep-disable %d hmx[%u,%u,%u]",
         (ਬਿਨਾਨਿਸ਼ਾਨੀ) power_ਬਹੁਲਕ, (ਪੂਰਨ) pwr->dcvs_enable,
         (ਬਿਨਾਨਿਸ਼ਾਨੀ) pwr->bus_min_ਕੋਨਾ, (ਬਿਨਾਨਿਸ਼ਾਨੀ) pwr->bus_max_ਕੋਨਾ, (ਬਿਨਾਨਿਸ਼ਾਨੀ) pwr->bus_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ,
         (ਬਿਨਾਨਿਸ਼ਾਨੀ) pwr->core_min_ਕੋਨਾ, (ਬਿਨਾਨਿਸ਼ਾਨੀ) pwr->core_max_ਕੋਨਾ, (ਬਿਨਾਨਿਸ਼ਾਨੀ) pwr->core_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ,
         (ਪੂਰਨ) pwr->sleep_disable,
         (ਬਿਨਾਨਿਸ਼ਾਨੀ) pwr->hmx_min_ਕੋਨਾ, (ਬਿਨਾਨਿਸ਼ਾਨੀ) pwr->hmx_max_ਕੋਨਾ, (ਬਿਨਾਨਿਸ਼ਾਨੀ) pwr->hmx_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ);

    ਜੇ (h->ctx) {
        FARF(ERROR, "Queue already open");
        ਵਾਪਸ AEE_EITEMBUSY;
    }

    // Cache the original FastRPC thread priority, then calculate compute priority
    ਪੂਰਨ fastrpc_tid  = qurt_thread_get_id();
    ਪੂਰਨ fastrpc_prio = qurt_thread_get_ਤਰਜੀਹ(fastrpc_tid);
    ਪੂਰਨ main_prio    = fastrpc_prio - 10;
    ਜੇ (main_prio < 1) main_prio = 1;

    dspqueue_t dsp_ਕਤਾਰ = ਨਿਰਾ;
    bool use_callbacks = false;

    // Import queue with NULL callbacks to avoid starting dspueue internal threads
    ਪੂਰਨ err = dspqueue_import(dsp_ਕਤਾਰ_id, ਨਿਰਾ, ਨਿਰਾ, (ਖਾਲੀ *) h, &dsp_ਕਤਾਰ);
    ਜੇ (err == AEE_EBADPARM) {
        // Fallback for devices that don't support NULL callbacks
        FARF(ਉੱਚਾ_2, "dspqueue import with NULL callbacks failed, trying with callbacks");
        use_callbacks = true;
        err = dspqueue_import(dsp_ਕਤਾਰ_id, htp_packet_callback, htp_error_callback, (ਖਾਲੀ *) h, &dsp_ਕਤਾਰ);
    }

    ਜੇ (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        ਵਾਪਸ err;
    }

    qurt_sysenv_max_hthreads_t hw_ਧਾਗੇ;
    qurt_sysenv_get_max_hw_ਧਾਗੇ(&hw_ਧਾਗੇ);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    ਜੇ (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    ਜੇ (n_hvx > hw_ਧਾਗੇ.max_hthreads) {
        n_hvx = hw_ਧਾਗੇ.max_hthreads;
    }
    ਜੇ (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    // layout segments of our contiguous block

    // 1. htp_context : sits at the base (block is 4K-aligned via memalign below)
    size_t ਵਿੱਥ = ਆਕਾਰ(ਬਣਤਰ htp_context);

    // 2. main_stack
    size_t ਵਿੱਥ_main_ਢੇਰ = 0;
    size_t ਮਾਪ_main_ਢੇਰ   = 0;
    ਜੇ (!use_callbacks) {
        ਵਿੱਥ_main_ਢੇਰ = hex_align_ਉੱਪਰ(ਵਿੱਥ, 4096);
        ਮਾਪ_main_ਢੇਰ   = MAIN_THREAD_ਢੇਰ_ਮਾਪ;
        ਵਿੱਥ = ਵਿੱਥ_main_ਢੇਰ + ਮਾਪ_main_ਢੇਰ;
    }

    // 3. work_queue
    size_t wq_align  = work_ਕਤਾਰ_alignof();
    size_t ਵਿੱਥ_wq = hex_align_ਉੱਪਰ(ਵਿੱਥ, wq_align);
    size_t ਮਾਪ_wq   = work_ਕਤਾਰ_sizeof(n_hvx, WORK_ਕਤਾਰ_ਸਮਰੱਥਾ, WORK_ਕਤਾਰ_ਢੇਰ_ਮਾਪ);
    ਵਿੱਥ = ਵਿੱਥ_wq + ਮਾਪ_wq;

    // 4. dma_queue
    size_t dma_align = dma_ਕਤਾਰ_alignof();
    size_t ਵਿੱਥ_dma = hex_align_ਉੱਪਰ(ਵਿੱਥ, dma_align);
    size_t ਮਾਪ_dma = 0;
    ਲਈ (uint32_t i = 0; i < n_hvx; i++) {
        ਮਾਪ_dma  = hex_align_ਉੱਪਰ(ਮਾਪ_dma, dma_ਕਤਾਰ_alignof());
        ਮਾਪ_dma += dma_ਕਤਾਰ_sizeof(256);
    }
    ਵਿੱਥ = ਵਿੱਥ_dma + ਮਾਪ_dma;

    // 5. hmx_queue
    size_t ਵਿੱਥ_hmx = 0;
    size_t ਮਾਪ_hmx = 0;
    ਜੇ (n_hmx) {
        size_t hmx_align = hmx_ਕਤਾਰ_alignof();
        ਵਿੱਥ_hmx = hex_align_ਉੱਪਰ(ਵਿੱਥ, hmx_align);
        ਮਾਪ_hmx   = hmx_ਕਤਾਰ_sizeof(HMX_ਕਤਾਰ_ਸਮਰੱਥਾ, HMX_ਕਤਾਰ_ਢੇਰ_ਮਾਪ);
        ਵਿੱਥ = ਵਿੱਥ_hmx + ਮਾਪ_hmx;
    }

    size_t footprint = hex_align_ਉੱਪਰ(ਵਿੱਥ, 128);

    ਖਾਲੀ * ਖੰਡ = memalign(4096, footprint);
    ਜੇ (!ਖੰਡ) {
        FARF(ERROR, "Unable to allocate unified block of size %zu\n", footprint);
        dspqueue_close(dsp_ਕਤਾਰ);
        ਵਾਪਸ AEE_ENOMEMORY;
    }
    memset(ਖੰਡ, 0, footprint);

    h->ctx = (ਬਣਤਰ htp_context *) ਖੰਡ;
    ਬਣਤਰ htp_context * ctx = h->ctx;
    ctx->footprint = footprint;

    ctx->thread_id   = fastrpc_tid;
    ctx->thread_prio = main_prio;
    ctx->max_vmem    = max_vmem;
    ctx->dsp_ਕਤਾਰ   = dsp_ਕਤਾਰ;

    err = vtcm_alloc(ctx);
    ਜੇ (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        htp_iface_stop(handle);
        ਵਾਪਸ AEE_ENOMEMORY;
    }

    HAP_ਸੈੱਟ_FARFRuntime_Logging_Params(0xffff, ਨਿਰਾ, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, ਆਕਾਰ(HAP_power_request_t));
        request.type    = HAP_power_ਸੈੱਟ_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        ਜੇ ((err = HAP_power_ਸੈੱਟ((ਖਾਲੀ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ਵਾਪਸ err;
        }
    }

    // DCVS setup
    {
        HAP_power_request_t request;
        memset(&request, 0, ਆਕਾਰ(request));

        request.type                              = HAP_power_ਸੈੱਟ_DCVS_v3;
        request.dcvs_v3.ਸੈੱਟ_dcvs_enable           = TRUE;
        request.dcvs_v3.dcvs_enable               = pwr->dcvs_enable;
        request.dcvs_v3.ਸੈੱਟ_bus_params            = TRUE;
        request.dcvs_v3.bus_params.min_ਕੋਨਾ     = pwr->bus_min_ਕੋਨਾ;
        request.dcvs_v3.bus_params.max_ਕੋਨਾ     = pwr->bus_max_ਕੋਨਾ;
        request.dcvs_v3.bus_params.ਨਿਸ਼ਾਨਾ_ਕੋਨਾ  = pwr->bus_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ;
        request.dcvs_v3.ਸੈੱਟ_core_params           = TRUE;
        request.dcvs_v3.core_params.min_ਕੋਨਾ    = pwr->core_min_ਕੋਨਾ;
        request.dcvs_v3.core_params.max_ਕੋਨਾ    = pwr->core_max_ਕੋਨਾ;
        request.dcvs_v3.core_params.ਨਿਸ਼ਾਨਾ_ਕੋਨਾ = pwr->core_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ;
        request.dcvs_v3.ਸੈੱਟ_sleep_disable         = TRUE;
        request.dcvs_v3.sleep_disable             = pwr->sleep_disable;

#ਜੇ (__HEXAGON_ARCH__ >= 79)
        HAP_ਸੈੱਟ_dcvs_v3_protected_bus_corners(&request, 1);
#ਜੇ_ਅੰਤ
        ਜੇ ((err = HAP_power_ਸੈੱਟ((ਖਾਲੀ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ਵਾਪਸ err;
        }

        memset(&request, 0, ਆਕਾਰ(request));
        request.type         = HAP_power_ਸੈੱਟ_HVX;
        request.hvx.power_ਉੱਪਰ = TRUE;
        ਜੇ ((err = HAP_power_ਸੈੱਟ((ਖਾਲੀ *) ctx, &request)) != 0) {
            htp_iface_stop(handle);
            ਵਾਪਸ err;
        }
    }

#ਜੇ __HVX_ARCH__ >= 75
    {
        // Power on HMX and set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, ਆਕਾਰ(HAP_power_request_t));
        request.type = HAP_power_ਸੈੱਟ_HMX_v2;
        request.hmx_v2.ਸੈੱਟ_power     = TRUE;
        request.hmx_v2.power_ਉੱਪਰ      = TRUE;
        request.hmx_v2.ਸੈੱਟ_clock     = TRUE;
        request.hmx_v2.ਨਿਸ਼ਾਨਾ_ਕੋਨਾ = pwr->hmx_ਨਿਸ਼ਾਨਾ_ਕੋਨਾ;
        request.hmx_v2.min_ਕੋਨਾ    = pwr->hmx_min_ਕੋਨਾ;
        request.hmx_v2.max_ਕੋਨਾ    = pwr->hmx_max_ਕੋਨਾ;
        request.hmx_v2.perf_ਬਹੁਲਕ     = pwr->hmx_perf_ਬਹੁਲਕ;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_ਸੈੱਟ((ਖਾਲੀ *) ctx, &request);
        ਜੇ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error setting HMX clock.");
            htp_iface_stop(handle);
            ਵਾਪਸ err;
        }
    }
#ਨਹੀਂਤਾਂ
    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, ਆਕਾਰ(HAP_power_request_t));
        request.type         = HAP_power_ਸੈੱਟ_HMX;
        request.hmx.power_ਉੱਪਰ = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_ਸੈੱਟ((ਖਾਲੀ *) ctx, &request);
        ਜੇ (err != AEE_SUCCESS) {
            FARF(ERROR, "ggml-hex: error powering on HMX.");
            htp_iface_stop(handle);
            ਵਾਪਸ err;
        }
    }
#ਜੇ_ਅੰਤ

    ctx->hmx_enabled = n_hmx;
    ctx->hmx_ਕਤਾਰ   = ਨਿਰਾ;
    ਜੇ (n_hmx) {
        ਖਾਲੀ * hmx_ptr = (ਖਾਲੀ *) ((uintptr_t) ਖੰਡ + ਵਿੱਥ_hmx);
        ctx->hmx_ਕਤਾਰ = hmx_ਕਤਾਰ_init(hmx_ptr, HMX_ਕਤਾਰ_ਸਮਰੱਥਾ, HMX_ਕਤਾਰ_ਢੇਰ_ਮਾਪ, ctx->vtcm_rctx, &ctx->trace[HTP_MAX_NTHREADS]);
    }
    FARF(ਉੱਚਾ_2, "HMX %s (n_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", n_hmx);

    ctx->n_ਧਾਗੇ = n_hvx;
    ctx->n_ਧਾਗੇ_div = init_fastdiv_ਮੁੱਲਾਂ(ctx->n_ਧਾਗੇ);

    // Initialize DMA queues
    uint8_t * dma_ptr_curr = (uint8_t *) ((uintptr_t) ਖੰਡ + ਵਿੱਥ_dma);
    size_t ਮਾਪ_dma_q = dma_ਕਤਾਰ_sizeof(256);

    ਲਈ (ਪੂਰਨ i = 0; i < ctx->n_ਧਾਗੇ; i++) {
        dma_ptr_curr = (uint8_t *) hex_align_ਉੱਪਰ((uintptr_t) dma_ptr_curr, dma_ਕਤਾਰ_alignof());
        ctx->dma[i] = dma_ਕਤਾਰ_init(dma_ptr_curr, 256, &ctx->trace[i]);
        dma_ptr_curr += ਮਾਪ_dma_q;
    }

    ctx->ddr_spad_ਮਾਪ = 512 * 1024; // 512 KB
    ctx->ddr_spad_ਅਧਾਰ = memalign(128, ctx->ddr_spad_ਮਾਪ);

    ਖਾਲੀ * wq_ptr = (ਖਾਲੀ *) ((uintptr_t) ਖੰਡ + ਵਿੱਥ_wq);
    ctx->work_ਕਤਾਰ = work_ਕਤਾਰ_init(wq_ptr, n_hvx, WORK_ਕਤਾਰ_ਸਮਰੱਥਾ, WORK_ਕਤਾਰ_ਢੇਰ_ਮਾਪ);

    ctx->main_ਢੇਰ = ਨਿਰਾ;
    ctx->main_thread = 0;
    atomic_store(&ctx->killed, false);

    ਜੇ (!use_callbacks) {
        // Start main compute thread
        ctx->main_ਢੇਰ = (ਖਾਲੀ *) ((uintptr_t) ਖੰਡ + ਵਿੱਥ_main_ਢੇਰ);

        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_ਸੈੱਟ_ਢੇਰ_addr(&attr, ctx->main_ਢੇਰ);
        qurt_thread_attr_ਸੈੱਟ_ਢੇਰ_ਮਾਪ(&attr, ਮਾਪ_main_ਢੇਰ);
        qurt_thread_attr_ਸੈੱਟ_ਤਰਜੀਹ(&attr, main_prio);
        qurt_thread_attr_ਸੈੱਟ_name(&attr, "htp-main");

        ਪੂਰਨ err_thread = qurt_thread_create(&ctx->main_thread, &attr, htp_main_thread, ctx);
        ਜੇ (err_thread) {
            FARF(ERROR, "Unable to create htp main thread: %d", err_thread);
            htp_iface_stop(handle);
            ਵਾਪਸ AEE_ENOMEMORY;
        }
    }

    FARF(ਉੱਚਾ_2, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_ਮਾਪ, ctx->vtcm_rctx, ctx->n_ਧਾਗੇ, ctx->thread_id, ctx->thread_prio);

    ਵਾਪਸ AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    ਬਣਤਰ htp_handle * h = (ਬਣਤਰ htp_handle *) handle;
    ਜੇ (!h || !h->ctx) {
        ਵਾਪਸ AEE_EBADPARM;
    }
    ਬਣਤਰ htp_context * ctx = h->ctx;

    ਜੇ (ctx->main_thread) {
        atomic_store(&ctx->killed, true);
        ਪੂਰਨ ਸਥਿਤੀ;
        (ਖਾਲੀ) qurt_thread_join(ctx->main_thread, &ਸਥਿਤੀ);
        ctx->main_thread = 0;
    }

    ਪੂਰਨ err = dspqueue_close(ctx->dsp_ਕਤਾਰ); ctx->dsp_ਕਤਾਰ = ਨਿਰਾ;
    ਜੇ (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        ਵਾਪਸ err;
    }

    work_ਕਤਾਰ_free(ctx->work_ਕਤਾਰ);

    ਲਈ (ਪੂਰਨ i = 0; i < ctx->n_ਧਾਗੇ; i++) {
        dma_ਕਤਾਰ_free(ctx->dma[i]);
    }

    ਜੇ (ctx->hmx_ਕਤਾਰ) {
        hmx_ਕਤਾਰ_free(ctx->hmx_ਕਤਾਰ);
        ctx->hmx_ਕਤਾਰ = ਨਿਰਾ;
    }
    ctx->hmx_enabled = false;

    vtcm_free(ctx);

    ਜੇ (ctx->ddr_spad_ਅਧਾਰ) {
        ਮੁਕਤ(ctx->ddr_spad_ਅਧਾਰ);
        ctx->ddr_spad_ਅਧਾਰ = ਨਿਰਾ;
        ctx->ddr_spad_ਮਾਪ = 0;
    }

    ਮੁਕਤ(ctx);
    h->ctx = ਨਿਰਾ;

    ਵਾਪਸ AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, uint32_t * n_ਧਾਗੇ, uint32_t * n_hvx, uint32_t * n_hmx, uint64_t * vtcm_ਮਾਪ) {
    (ਖਾਲੀ)handle;
    ਜੇ (!n_ਧਾਗੇ || !n_hvx || !n_hmx || !vtcm_ਮਾਪ) {
        ਵਾਪਸ AEE_EBADPARM;
    }

    qurt_sysenv_max_hthreads_t hw_ਧਾਗੇ;
    qurt_sysenv_get_max_hw_ਧਾਗੇ(&hw_ਧਾਗੇ);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    uint32_t n_hvx_val = hw_nhvx;
    ਜੇ (n_hvx_val > hw_ਧਾਗੇ.max_hthreads) {
        n_hvx_val = hw_ਧਾਗੇ.max_hthreads;
    }
    ਜੇ (n_hvx_val > HTP_MAX_NTHREADS) {
        n_hvx_val = HTP_MAX_NTHREADS;
    }

    // for now we force n_threads == n_hvx
    *n_ਧਾਗੇ = n_hvx_val;
    *n_hvx     = n_hvx_val;
    *n_hmx     = 1;

    *vtcm_ਮਾਪ = vtcm_query_ਮਾਪ();

    ਵਾਪਸ AEE_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ htp_error_callback(dspqueue_t ਕਤਾਰ, ਪੂਰਨ error, ਖਾਲੀ * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (ਬਿਨਾਨਿਸ਼ਾਨੀ) error);
}

ਬਣਤਰ profile_ਅੰਕੜੇ {
    uint64_t usecs;
    uint64_t ਚੱਕਰਾਂ_ਸ਼ੁਰੂ;
    uint64_t ਚੱਕਰਾਂ_stop;
    uint32_t pmu_counters[HEX_ਅੰਕ_PMU_COUNTERS];
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ profile_ਸ਼ੁਰੂ(uint32_t ਬਹੁਲਕ, ਬਣਤਰ profile_ਅੰਕੜੇ * d) {
    ਚੋਣ (ਬਹੁਲਕ) {
        ਹਾਲਤ HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        ਹਾਲਤ HTP_PROF_BASIC:
        ਹਾਲਤ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_get_qtimer_ਗਿਣਤੀ();
            d->ਚੱਕਰਾਂ_ਸ਼ੁਰੂ = hex_get_ਚੱਕਰਾਂ();
            ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ profile_stop(uint32_t ਬਹੁਲਕ, ਬਣਤਰ profile_ਅੰਕੜੇ * d) {
    uint32_t pmu_counters[HEX_ਅੰਕ_PMU_COUNTERS];
    ਚੋਣ (ਬਹੁਲਕ) {
        ਹਾਲਤ HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            ਲਈ (ਪੂਰਨ i = 0; i < HEX_ਅੰਕ_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        ਹਾਲਤ HTP_PROF_BASIC:
        ਹਾਲਤ HTP_PROF_TRACE:
            d->usecs  = HAP_perf_qtimer_ਗਿਣਤੀ_ਮੰਜ਼ਿਲ_us(HAP_perf_get_qtimer_ਗਿਣਤੀ() - d->usecs);
            d->ਚੱਕਰਾਂ_stop = hex_get_ਚੱਕਰਾਂ();
            ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
    }
}

ਸਥਿਰ ਪੂਰਨ op_fence(ਬਣਤਰ htp_ops_context * octx) {
    ਬਣਤਰ htp_context *ctx = octx->ctx;
    ਬਣਤਰ htp_thread_trace * tr = &ctx->trace[0];
    ਅਚਲ uint32_t seq  = (uint32_t) octx->op_params[0];
    ਅਚਲ uint32_t ਬਹੁਲਕ = (uint32_t) octx->op_params[1];

    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    ਅਚਲ ਬਣਤਰ htp_tensor * sync = octx->src[0];
    atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->ਅੰਕੜੇ;

    ਜੇ (ਬਹੁਲਕ == 1) {
        htp_dcache_flush_dirty_ranges(ctx);

        htp_mdev_ਸਮੂਹ_barrier(octx);

        ਜੇ (ctx->mdev.idx == 0) {
            htp_fence_write(sync_fence, seq, octx->ਸਥਿਤੀ);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        FARF(ਉੱਚਾ_2, "ggml-hex: sync-signal : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->ਸਥਿਤੀ);
        ਵਾਪਸ octx->ਸਥਿਤੀ;
    }

    ਪੂਰਨ ਸਥਿਤੀ = HTP_ਸਥਿਤੀ_OK;
    uint64_t spins = 0;
    ਜਦੋਂ (1) {
        uint32_t sync_seq;
        uint32_t sync_ਸਥਿਤੀ;
        htp_fence_read(sync_fence, &sync_seq, &sync_ਸਥਿਤੀ);
        ਜੇ ((int32_t)(sync_seq - seq) >= 0) {
            ਜੇ (sync_ਸਥਿਤੀ > HTP_ਸਥਿਤੀ_OK) {
                FARF(ERROR, "ggml-hex: sync-wait peer failed with status %u : fence %p seq 0x%x\n", sync_ਸਥਿਤੀ, sync_fence, seq);
                ਸਥਿਤੀ = sync_ਸਥਿਤੀ;
            }
            ਰੋਕੋ;
        }
        ਜੇ (++spins > HTP_FENCE_TIMEOUT) {
            FARF(ERROR, "ggml-hex: sync-wait TIMEOUT : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
            ਸਥਿਤੀ = HTP_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERR;
            ਰੋਕੋ;
        }
        hex_pause();
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    FARF(ਉੱਚਾ_2, "ggml-hex: sync-done : fence %p spins %llu seq 0x%x\n", sync_fence, spins, seq);
    ਵਾਪਸ ਸਥਿਤੀ;
}

ਸਥਿਰ ਪੂਰਨ op_mdev_ਸਮੂਹ(ਬਣਤਰ htp_ops_context * octx) {
    ਬਣਤਰ htp_context * ctx = octx->ctx;
    ਅਚਲ ਬਣਤਰ htp_tensor * sync = octx->src[0];
    ctx->mdev.idx   = (uint16_t) octx->op_params[0];
    ctx->mdev.ਗਿਣਤੀ_2 = (uint16_t) sync->ne[1];
    ਜੇ (ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ctx->mdev.ਗਿਣਤੀ_div = init_fastdiv_ਮੁੱਲਾਂ(ctx->mdev.ਗਿਣਤੀ_2);
        ctx->mdev.fence_ਅਧਾਰ = (uint8_t *) sync->ਅੰਕੜੇ;
    }
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਪੂਰਨ execute_op(ਬਣਤਰ htp_ops_context * octx) {
    ਚੋਣ (octx->op) {
        ਹਾਲਤ HTP_OP_MDEV_ਸਮੂਹ:
            ਵਾਪਸ op_mdev_ਸਮੂਹ(octx);

        ਹਾਲਤ HTP_OP_FENCE:
            ਵਾਪਸ op_fence(octx);

        ਹਾਲਤ HTP_OP_ALLREDUCE:
        ਹਾਲਤ HTP_OP_ALLREDUCE_ਜੋੜੋ:
            ਵਾਪਸ op_allreduce(octx);

        ਹਾਲਤ HTP_OP_MUL_MAT:
        ਹਾਲਤ HTP_OP_MUL_MAT_ਜੋੜੋ:
            ਵਾਪਸ op_matmul(octx);

        ਹਾਲਤ HTP_OP_MUL_MAT_ID:
            ਵਾਪਸ op_matmul_id(octx);

        ਹਾਲਤ HTP_OP_MUL_MAT_ID_NX:
            ਵਾਪਸ op_matmul_id_nx(octx);

        ਹਾਲਤ HTP_OP_MUL_MAT_NX:
            ਵਾਪਸ op_matmul_nx(octx);

        ਹਾਲਤ HTP_OP_MUL:
        ਹਾਲਤ HTP_OP_ਜੋੜੋ:
        ਹਾਲਤ HTP_OP_SUB:
        ਹਾਲਤ HTP_OP_DIV:
        ਹਾਲਤ HTP_OP_ਜੋੜੋ_ID:
            ਵਾਪਸ op_ਦੁਆਧਾਰੀ(octx);

        ਹਾਲਤ HTP_OP_NORM:
        ਹਾਲਤ HTP_OP_RMS_NORM:
        ਹਾਲਤ HTP_OP_RMS_NORM_MUL:
        ਹਾਲਤ HTP_OP_SCALE:
        ਹਾਲਤ HTP_OP_CLAMP:
        ਹਾਲਤ HTP_OP_LEAKY_RELU:
        ਹਾਲਤ HTP_OP_SQR:
        ਹਾਲਤ HTP_OP_SQRT:
        ਹਾਲਤ HTP_OP_UNARY_SOFTPLUS:
        ਹਾਲਤ HTP_OP_UNARY_SIGMOID:
        ਹਾਲਤ HTP_OP_UNARY_SILU:
        ਹਾਲਤ HTP_OP_UNARY_GELU:
        ਹਾਲਤ HTP_OP_UNARY_GELU_ਤੇਜ਼:
        ਹਾਲਤ HTP_OP_UNARY_GELU_ERF:
        ਹਾਲਤ HTP_OP_UNARY_NEG:
        ਹਾਲਤ HTP_OP_UNARY_EXP:
        ਹਾਲਤ HTP_OP_UNARY_TANH:
        ਹਾਲਤ HTP_OP_UNARY_ABS:
        ਹਾਲਤ HTP_OP_UNARY_LOG:
        ਹਾਲਤ HTP_OP_UNARY_RELU:
        ਹਾਲਤ HTP_OP_UNARY_ਕਦਮ:
        ਹਾਲਤ HTP_OP_L2_NORM:
        ਹਾਲਤ HTP_OP_HADAMARD:
            ਵਾਪਸ op_unary(octx);

        ਹਾਲਤ HTP_OP_GLU_SWIGLU:
        ਹਾਲਤ HTP_OP_GLU_SWIGLU_OAI:
        ਹਾਲਤ HTP_OP_GLU_SWIGLU_CLAMP:
        ਹਾਲਤ HTP_OP_GLU_GEGLU:
        ਹਾਲਤ HTP_OP_GLU_GEGLU_ਤੇਜ਼:
        ਹਾਲਤ HTP_OP_GLU_GEGLU_ERF:
            ਵਾਪਸ op_activations(octx);

        ਹਾਲਤ HTP_OP_SOFTMAX:
            ਵਾਪਸ op_softmax(octx);

        ਹਾਲਤ HTP_OP_ROPE:
            ਵਾਪਸ op_rope(octx);

        ਹਾਲਤ HTP_OP_FLASH_ATTN_EXT:
            ਵਾਪਸ op_flash_attn_ext(octx);

        ਹਾਲਤ HTP_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            ਵਾਪਸ op_ਸੈੱਟ_ਪੰਕਤੀਆਂ(octx);

        ਹਾਲਤ HTP_OP_GET_ਪੰਕਤੀਆਂ:
            ਵਾਪਸ op_get_ਪੰਕਤੀਆਂ(octx);

        ਹਾਲਤ HTP_OP_ਜੋੜ:
            ਵਾਪਸ op_ਜੋੜ(octx);

        ਹਾਲਤ HTP_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            ਵਾਪਸ op_ਜੋੜ_ਪੰਕਤੀਆਂ(octx);

        ਹਾਲਤ HTP_OP_CPY:
        ਹਾਲਤ HTP_OP_CPY_FENCE:
            ਵਾਪਸ op_cpy(octx);

        ਹਾਲਤ HTP_OP_REPEAT:
            ਵਾਪਸ op_repeat(octx);

        ਹਾਲਤ HTP_OP_ARGSORT:
            ਵਾਪਸ op_argsort(octx);

        ਹਾਲਤ HTP_OP_ਸਿਖਰ_K:
            ਵਾਪਸ op_ਸਿਖਰ_k(octx);

        ਹਾਲਤ HTP_OP_ARGMAX:
            ਵਾਪਸ op_argmax(octx);

        ਹਾਲਤ HTP_OP_SSM_CONV:
            ਵਾਪਸ op_ssm_conv(octx);

        ਹਾਲਤ HTP_OP_CUMSUM:
            ਵਾਪਸ op_cumsum(octx);

        ਹਾਲਤ HTP_OP_FILL:
            ਵਾਪਸ op_fill(octx);

        ਹਾਲਤ HTP_OP_DIAG:
            ਵਾਪਸ op_diag(octx);

        ਹਾਲਤ HTP_OP_SOLVE_TRI:
            ਵਾਪਸ op_solve_tri(octx);

        ਹਾਲਤ HTP_OP_PAD:
            ਵਾਪਸ op_pad(octx);

        ਹਾਲਤ HTP_OP_IM2_ਖਾਨਾ:
            ਵਾਪਸ op_im2col(octx);

        ਹਾਲਤ HTP_OP_IM2_ਖਾਨਾ_3_D:
            ਵਾਪਸ op_im2col_3d(octx);

        ਹਾਲਤ HTP_OP_ROLL:
            ਵਾਪਸ op_roll(octx);

        ਹਾਲਤ HTP_OP_POOL_2D:
            ਵਾਪਸ op_pool_2d(octx);

        ਹਾਲਤ HTP_OP_POOL_1D:
            ਵਾਪਸ op_pool_1d(octx);

        ਹਾਲਤ HTP_OP_CONCAT:
            ਵਾਪਸ op_concat(octx);

        ਹਾਲਤ HTP_OP_GATED_DELTA_NET:
            ਵਾਪਸ op_gated_delta_net(octx);

        ਹਾਲਤ HTP_OP_TRI:
            ਵਾਪਸ op_unary(octx);

        ਹਾਲਤ HTP_OP_INVALID:
            ਰੋਕੋ;
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool reuse_buf(ਬਣਤਰ htp_context *ctx, uint32_t *m_reuse, ਬਣਤਰ htp_buf_desc *b) {
    b->ਅਧਾਰ = ਨਿਰਾ;

    ਲਈ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ਬਣਤਰ htp_mmap *m = ctx->mmap + i;
        ਜੇ (m->ਮਾਪ && m->fd == b->fd && m->flags == b->flags) {
            b->ਅਧਾਰ   = m->ਅਧਾਰ;
            *m_reuse |= (1 << i);
            ਵਾਪਸ true;
        }
    }

    ਵਾਪਸ false;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ drop_mmap(ਬਣਤਰ htp_context *ctx, ਬਣਤਰ htp_mmap *m) {
    ਜੇ (m->ਮਾਪ) {
        FARF(ALWAYS, "unmap : fd %u base 0x%llx size %llu", m->fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) m->ਅਧਾਰ, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) m->ਮਾਪ);
        htp_munmap(m->ਅਧਾਰ, m->ਮਾਪ, m->flags);
        m->ਮਾਪ = 0;
        m->ਅਧਾਰ = 0;
        m->fd   = -1;
        m->flags = 0;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool mmap_buf(ਬਣਤਰ htp_context *ctx, ਬਣਤਰ htp_buf_desc *b) {
    ਜੇ (b->ਅਧਾਰ) ਵਾਪਸ true; // already mapped

    // find unused mapping
    ਲਈ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        ਬਣਤਰ htp_mmap *m = &ctx->mmap[i];
        ਜੇ (!m->ਮਾਪ) {
            uint64_t va = htp_mmap(b->fd, b->ਮਾਪ, b->flags);
            ਜੇ (va == 0) {
                FARF(ਉੱਚਾ_2, "mmap failed (will attempt defrag) : fd %u size %llu", b->fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) b->ਮਾਪ);
                ਵਾਪਸ false;
            }

            m->ਅਧਾਰ   = b->ਅਧਾਰ = va;
            m->fd     = b->fd;
            m->ਮਾਪ   = b->ਮਾਪ;
            m->flags  = b->flags;

            FARF(ALWAYS, "mmap : fd %u base 0x%llx size %llu flags 0x%x", m->fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) m->ਅਧਾਰ, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) m->ਮਾਪ, m->flags);
            ਵਾਪਸ true;
        }
    }

    FARF(ERROR, "mmap failed : exceeded mapping capacity limit of %u", HTP_MAX_MMAPS);
    ਵਾਪਸ false;
}

ਸਥਿਰ ਖਾਲੀ prep_op_bufs(ਬਣਤਰ htp_context *ctx, ਬਣਤਰ htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    uint64_t m_vmem  = 0; // mapped vmem
    uint64_t e_vmem  = 0; // extra  vmem

    // See what we can reuse
    ਲਈ (uint32_t i=0; i < n_bufs; i++) {
        ਬਣਤਰ htp_buf_desc *b = bufs + i;
        ਜੇ (reuse_buf(ctx, &m_reuse, b)) {
            b_reuse++;
        } ਨਹੀਂਤਾਂ ਜੇ (!(b->flags & HTP_BUF_EXTENDED)) {
            e_vmem += b->ਮਾਪ;
        }
        FARF(ਉੱਚਾ_2, "prep-buf #%u : pass0 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) b->ਅਧਾਰ, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) b->ਮਾਪ, b->flags);
    }

    ਜੇ (b_reuse == n_bufs) ਵਾਪਸ; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    ਲਈ (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        ਜੇ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
            m_vmem += ctx->mmap[i].ਮਾਪ;
        }
    }

    FARF(ਉੱਚਾ_2, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);

    ਜੇ ((m_vmem + e_vmem) > ctx->max_vmem) {
        // Drop unused mappings
        ਲਈ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool ਵਰਤਿਆ = m_reuse & (1<<i);
            ਜੇ (!ਵਰਤਿਆ && !(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }
    }

    // Create missing mappings (pass 1)
    bool mmap_ok = true;
    ਲਈ (uint32_t i=0; i < n_bufs; i++) {
        ਬਣਤਰ htp_buf_desc *b = bufs + i;
        ਜੇ (!mmap_buf(ctx, b)) {
            mmap_ok = false;
            ਰੋਕੋ;
        }
        FARF(ਉੱਚਾ_2, "prep-buf #%u : pass1 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) b->ਅਧਾਰ, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) b->ਮਾਪ, b->flags);
    }

    ਜੇ (!mmap_ok) {
        // Attempt defragmentation: drop 32-bit mappings and remap (pass 2)
        FARF(ਉੱਚਾ_2, "prep-bufs : dropping 32-bit mappings to defragment address space");
        ਲਈ (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            ਜੇ (!(ctx->mmap[i].flags & HTP_BUF_EXTENDED)) {
                drop_mmap(ctx, ctx->mmap + i);
            }
        }

        ਲਈ (uint32_t i=0; i < n_bufs; i++) {
            ਬਣਤਰ htp_buf_desc *b = bufs + i;
            ਜੇ (!(b->flags & HTP_BUF_EXTENDED)) {
                b->ਅਧਾਰ = 0;
            }
            ਜੇ (!mmap_buf(ctx, b)) {
                FARF(ERROR, "prep-bufs : mmap failed after defragmentation (fd %u size %llu)", b->fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) b->ਮਾਪ);
                abort();
            }
            FARF(ਉੱਚਾ_2, "prep-buf #%u : pass2 fd %u base 0x%llx size %llu flags 0x%x", i, b->fd, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) b->ਅਧਾਰ, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) b->ਮਾਪ, b->flags);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ prep_tensor(ਬਣਤਰ htp_context *ctx, ਬਣਤਰ htp_buf_desc *bufs, ਬਣਤਰ htp_tensor *tens, uint32_t idx, ਬਣਤਰ htp_tensor *t) {
    uint64_t ਵਿੱਥ = t->ਅੰਕੜੇ;
    uint32_t bi     = t->bi;

    t->ਅੰਕੜੇ  = bufs[bi].ਅਧਾਰ + ਵਿੱਥ;  // update data to the actual pointer

    FARF(ਉੱਚਾ_2, "prep-tensor #%u: bi %u offset %llu size %u data 0x%llx : %u:%u:%u:%u", idx, t->bi, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) ਵਿੱਥ, t->ਮਾਪ, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) t->ਅੰਕੜੇ,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

ਸਥਿਰ ਖਾਲੀ prep_tensors(ਬਣਤਰ htp_context *ctx, ਬਣਤਰ htp_buf_desc *bufs, ਬਣਤਰ htp_tensor *tens, uint32_t n_tens) {
    ਲਈ (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, tens, i, tens + i);
    }
}

ਸਥਿਰ ਖਾਲੀ mdev_ਸਮੂਹ_init(ਬਣਤਰ htp_context * ctx, ਅਚਲ ਬਣਤਰ htp_opbatch_req * req) {
    memset(&ctx->mdev, 0, ਆਕਾਰ(ctx->mdev));
    ctx->mdev.fence_seq = (uint32_t)((req->seq & 0xfffff) << 12);
}

ਸਥਿਰ ਪੂਰਨ proc_op_req(ਬਣਤਰ htp_ops_context * octx, ਬਣਤਰ htp_buf_desc * bufs, uint32_t n_bufs,
                       ਬਣਤਰ htp_tensor * tens, uint32_t idx, ਬਣਤਰ htp_op_desc * op) {
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(octx->op_params,     op->params, ਆਕਾਰ(octx->op_params));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(octx->kernel_params, op->kernel_params, ਆਕਾਰ(octx->kernel_params));
    octx->flags         = op->flags;
    octx->op            = op->opcode;
    octx->n_ਧਾਗੇ     = octx->ctx->n_ਧਾਗੇ;
    octx->n_ਧਾਗੇ_div = octx->ctx->n_ਧਾਗੇ_div;

    FARF(ਉੱਚਾ_2, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    ਲਈ (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        uint16_t src_idx = op->src[i];
        ਜੇ (src_idx == 0xffff) {
            octx->src[i]     = ਨਿਰਾ;
            ਜਾਰੀ;
        }

        ਬਣਤਰ htp_tensor *src = tens + src_idx;
        octx->src[i]     = src;

        FARF(ਉੱਚਾ_2, "prep-src #%u: data 0x%llx size %u : %u:%u:%u:%u", op->src[i], (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) src->ਅੰਕੜੇ, src->ਮਾਪ,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    htp_dcache_flush_dirty_tensors(octx->ctx, octx->src, HTP_OP_MAX_INPUTS);

    // Prep output tensors
    ਲਈ (uint32_t i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        uint16_t dst_idx = op->dst[i];
        ਜੇ (dst_idx == 0xffff) {
            octx->dsts[i]    = ਨਿਰਾ;
            ਜਾਰੀ;
        }
        ਬਣਤਰ htp_tensor *dst = tens + dst_idx;
        octx->dsts[i]    = dst;

        FARF(ਉੱਚਾ_2, "prep-dst[%u] #%u: data 0x%llx size %u : %u:%u:%u:%u", i, dst_idx, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) dst->ਅੰਕੜੇ, dst->ਮਾਪ,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    }

    htp_dcache_mark_dirty_tensors(octx->ctx, octx->dsts, HTP_OP_MAX_OUTPUTS);

    htp_mdev_ਸਮੂਹ_barrier(octx);

    ਪੂਰਨ ਸਥਿਤੀ = execute_op(octx);

    htp_ops_context_ਸੈੱਟ_ਸਥਿਤੀ(octx, ਸਥਿਤੀ);

    octx->src0_spad.src = ਨਿਰਾ;
    octx->src1_spad.src = ਨਿਰਾ;
    octx->src2_spad.src = ਨਿਰਾ;
    octx->src3_spad.src = ਨਿਰਾ;
    octx->dst_spad.src  = ਨਿਰਾ;

    ਵਾਪਸ octx->ਸਥਿਤੀ;
}

ਸਥਿਰ ਖਾਲੀ process_opbatch(ਬਣਤਰ htp_context * ctx, ਅਚਲ ਬਣਤਰ htp_opbatch_req * req, ਅਚਲ ਬਣਤਰ dspqueue_ਬਫਰ * dbuf) {
    dspqueue_t ਕਤਾਰ = ctx->dsp_ਕਤਾਰ;
    ਪੂਰਨ err;

    ਅਚਲ uint32_t n_bufs = req->n_bufs;
    ਅਚਲ uint32_t n_tens = req->n_tensors;
    ਅਚਲ uint32_t n_ops  = req->n_ops;

    ਅਚਲ uint32_t b_ਮਾਪ = ਆਕਾਰ(ਬਣਤਰ htp_buf_desc)  * n_bufs;
    ਅਚਲ uint32_t t_ਮਾਪ = ਆਕਾਰ(ਬਣਤਰ htp_tensor)    * n_tens;
    ਅਚਲ uint32_t o_ਮਾਪ = ਆਕਾਰ(ਬਣਤਰ htp_op_desc)   * n_ops;
    ਅਚਲ uint32_t p_ਮਾਪ = ਆਕਾਰ(ਬਣਤਰ htp_prof_desc) * n_ops;
    ਅਚਲ uint32_t tr_ਮਾਪ = (HTP_MAX_NTHREADS + 1) * req->n_traces * ਆਕਾਰ(ਬਣਤਰ htp_trace_desc);

    ਜੇ (dbuf->ਮਾਪ < b_ਮਾਪ + t_ਮਾਪ + o_ਮਾਪ + p_ਮਾਪ + tr_ਮਾਪ) {
        FARF(ERROR, "invalid opbatch memory block size %u (req %u)", dbuf->ਮਾਪ, b_ਮਾਪ + t_ਮਾਪ + o_ਮਾਪ + p_ਮਾਪ + tr_ਮਾਪ);
        ਵਾਪਸ;
    }

    FARF(ਉੱਚਾ_2, "processing opbatch #%llu: n-bufs %u n-tensors %u n-ops %u n-traces %u : m-size %u b-size %u t-size %u o-size %u", (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) req->seq,
            n_bufs, n_tens, n_ops, req->n_traces, dbuf->ਮਾਪ, b_ਮਾਪ, t_ਮਾਪ, o_ਮਾਪ);

    // Setup descriptor pointers
    uint8_t * m_ptr = dbuf->ptr;
    ਬਣਤਰ htp_buf_desc* bufs = (ਬਣਤਰ htp_buf_desc*)  m_ptr; m_ptr += b_ਮਾਪ;
    ਬਣਤਰ htp_tensor*   tens = (ਬਣਤਰ htp_tensor*)    m_ptr; m_ptr += t_ਮਾਪ;
    ਬਣਤਰ htp_op_desc*   ops = (ਬਣਤਰ htp_op_desc*)   m_ptr; m_ptr += o_ਮਾਪ;
    ਬਣਤਰ htp_prof_desc* pds = (ਬਣਤਰ htp_prof_desc*) m_ptr;

    ਬਣਤਰ profile_ਅੰਕੜੇ batch_prof;
    profile_ਸ਼ੁਰੂ(HTP_PROF_BASIC, &batch_prof);

    memset(ctx->trace, 0, ਆਕਾਰ(ctx->trace));
    ਜੇ (ctx->profiler == HTP_PROF_TRACE) {
        ਬਣਤਰ htp_trace_desc * trace_events = (ਬਣਤਰ htp_trace_desc *) (m_ptr + p_ਮਾਪ);
        ਲਈ (ਪੂਰਨ t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ctx->trace[t].events     = &trace_events[t * req->n_traces];
            ctx->trace[t].max_events = req->n_traces;
        }
    }

    // Clean cache at the start of the batch
    htp_trace_event_ਸ਼ੁਰੂ(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_ਖੰਡ(ctx, ctx->footprint);
    memset(ctx->dirty_ranges, 0, ਆਕਾਰ(ctx->dirty_ranges));
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_trace_event_ਸ਼ੁਰੂ(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);
    prep_op_bufs(ctx, bufs, n_bufs);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_BUFF, 0);

    prep_tensors(ctx, bufs, tens, n_tens);

    ਬਣਤਰ htp_ops_context *octx = &ctx->octx;
    memset(octx, 0, ਆਕਾਰ(*octx));
    octx->n_ਧਾਗੇ     = ctx->n_ਧਾਗੇ;
    octx->n_ਧਾਗੇ_div = ctx->n_ਧਾਗੇ_div;
    octx->ctx           = ctx;

    mdev_ਸਮੂਹ_init(ctx, req);

    work_ਕਤਾਰ_wakeup(ctx->work_ਕਤਾਰ);
    ਜੇ (ctx->hmx_ਕਤਾਰ) {
        hmx_ਕਤਾਰ_wakeup(ctx->hmx_ਕਤਾਰ);
    }

    ਪੂਰਨ op_ਸਥਿਤੀ = HTP_ਸਥਿਤੀ_OK;
    octx->ਸਥਿਤੀ  = HTP_ਸਥਿਤੀ_OK;
    ਲਈ (uint32_t i = 0; i < n_ops; i++) {
        ਬਣਤਰ profile_ਅੰਕੜੇ prof;

        profile_ਸ਼ੁਰੂ(ctx->profiler, &prof);

        op_ਸਥਿਤੀ = proc_op_req(octx, bufs, n_bufs, tens, i, &ops[i]);

        profile_stop(ctx->profiler, &prof);

        htp_ops_context_ਸੈੱਟ_ਸਥਿਤੀ(octx, op_ਸਥਿਤੀ);

        ਜੇ (ctx->profiler) {
            pds[i].opcode = ops[i].opcode;
            pds[i].usecs  = prof.usecs;
            pds[i].ਚੱਕਰਾਂ_ਸ਼ੁਰੂ = prof.ਚੱਕਰਾਂ_ਸ਼ੁਰੂ;
            pds[i].ਚੱਕਰਾਂ_stop = prof.ਚੱਕਰਾਂ_stop;
            ਲਈ (ਪੂਰਨ j = 0; j < HEX_ਅੰਕ_PMU_COUNTERS; j++) {
                pds[i].pmu[j] = prof.pmu_counters[j];
            }
        }
    }

    ਜੇ (ctx->hmx_ਕਤਾਰ) {
        hmx_ਕਤਾਰ_suspend(ctx->hmx_ਕਤਾਰ);
        hmx_ਕਤਾਰ_flush(ctx->hmx_ਕਤਾਰ);
    }
    work_ਕਤਾਰ_suspend(ctx->work_ਕਤਾਰ);

    // Flush remaining dirty tensors at the end of the batch
    htp_trace_event_ਸ਼ੁਰੂ(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_L2FLUSH, 0);

    htp_mdev_ਸਮੂਹ_barrier(octx);

    profile_stop(HTP_PROF_BASIC, &batch_prof);

    ਬਣਤਰ htp_opbatch_rsp rsp;
    memset(&rsp, 0, ਆਕਾਰ(rsp));
    rsp.seq          = req->seq;
    rsp.ਸਥਿਤੀ       = octx->ਸਥਿਤੀ;
    rsp.n_bufs       = n_bufs;
    rsp.n_tensors    = n_tens;
    rsp.n_ops        = n_ops;
    rsp.usecs        = batch_prof.usecs;
    rsp.ਚੱਕਰਾਂ_ਸ਼ੁਰੂ = batch_prof.ਚੱਕਰਾਂ_ਸ਼ੁਰੂ;
    rsp.ਚੱਕਰਾਂ_stop  = batch_prof.ਚੱਕਰਾਂ_stop;

    ਜੇ (ctx->profiler == HTP_PROF_TRACE) {
        ਲਈ (ਪੂਰਨ t = 0; t <= HTP_MAX_NTHREADS; t++) {
            rsp.n_traces[t] = ctx->trace[t].ਗਿਣਤੀ_2;
        }
    }

    ਬਣਤਰ dspqueue_ਬਫਰ write_dbuf = *dbuf;
    write_dbuf.flags = DSPQUEUE_ਬਫਰ_FLAG_FLUSH_SENDER | DSPQUEUE_ਬਫਰ_FLAG_INVALIDATE_RECIPIENT;

    err = dspqueue_write(ਕਤਾਰ, 0, 1, &write_dbuf, ਆਕਾਰ(rsp), (ਅਚਲ uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
    ਜੇ (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DSPQUEUE_READ_TIMEOUT_USEC 5000
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DSPQUEUE_POLL_TIMEOUT_USEC 100
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DSPQUEUE_POLL_ਗਿਣਤੀ        100

ਸਥਿਰ ਖਾਲੀ process_ops(ਬਣਤਰ htp_context * ctx) {
    dspqueue_t ਕਤਾਰ = ctx->dsp_ਕਤਾਰ;
    ਪੂਰਨ err;

    uint32_t poll_ਗਿਣਤੀ = DSPQUEUE_POLL_ਗਿਣਤੀ;

    vtcm_acquire(ctx);

    ਜਦੋਂ (!atomic_ਲੱਦੋ(&ctx->vtcm_needs_release) && !atomic_ਲੱਦੋ(&ctx->killed)) {
        ਬਣਤਰ htp_opbatch_req req;
        uint32_t r_ਮਾਪ = ਆਕਾਰ(req);

        ਬਣਤਰ dspqueue_ਬਫਰ dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(ਕਤਾਰ, &flags, n_dbufs, &n_dbufs, &dbuf, r_ਮਾਪ, &r_ਮਾਪ, (uint8_t *) &req);
        ਜੇ (err == AEE_EWOULDBLOCK) {
            ਜੇ (--poll_ਗਿਣਤੀ) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                ਜਾਰੀ;
            }
            ਰੋਕੋ;
        }

        ਜੇ (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
            ਰੋਕੋ;
        }

        ਜੇ (r_ਮਾਪ < ਆਕਾਰ(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_ਮਾਪ, n_dbufs);
            ਜਾਰੀ;
        }

        // Reset poll count for valid requests
        poll_ਗਿਣਤੀ = DSPQUEUE_POLL_ਗਿਣਤੀ;

        process_opbatch(ctx, &req, &dbuf);
    }

    vtcm_release(ctx);
}

ਸਥਿਰ ਖਾਲੀ htp_packet_callback(dspqueue_t ਕਤਾਰ, ਪੂਰਨ error, ਖਾਲੀ * context) {
    (ਖਾਲੀ) ਕਤਾਰ;
    (ਖਾਲੀ) error;
    ਬਣਤਰ htp_handle * h = (ਬਣਤਰ htp_handle *) context;
    ਜੇ (h && h->ctx) {
        process_ops(h->ctx);
    }
}

ਸਥਿਰ ਖਾਲੀ htp_main_thread(ਖਾਲੀ * context) {
    ਬਣਤਰ htp_context * ctx = (ਬਣਤਰ htp_context *) context;

    FARF(ਉੱਚਾ_2, "htp-main-thread: started");

    ਜਦੋਂ (!atomic_ਲੱਦੋ(&ctx->killed)) {
        uint32_t flags = 0;
        uint32_t ਅੰਕ_buffers = 0;
        uint32_t ਸੁਨੇਹਾ_ਲੰਮਾਈ = 0;

        ਪੂਰਨ err = dspqueue_peek(ctx->dsp_ਕਤਾਰ, &flags, &ਅੰਕ_buffers, &ਸੁਨੇਹਾ_ਲੰਮਾਈ, 50000);
        ਜੇ (err == 0) {
            process_ops(ctx);
        } ਨਹੀਂਤਾਂ ਜੇ (err == AEE_EWOULDBLOCK || err == AEE_EEXPIRED) {
            ਜਾਰੀ;
        } ਨਹੀਂਤਾਂ {
            FARF(ERROR, "dspqueue_peek failed: 0x%08x", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
            ਰੋਕੋ;
        }
    }

    FARF(ਉੱਚਾ_2, "htp-main-thread: stopped");
}
