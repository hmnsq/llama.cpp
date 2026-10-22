#ಅನ್ತರ್ಭಾವ "htp-dcache.h"

#ಅನ್ತರ್ಭಾವ <qurt.h>
#ಅನ್ತರ್ಭಾವ <qurt_memory.h>
#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "work-queue.h"

ರಚನೆ l2flush_range {
    uint32_t ಆರಂಭ;       // line-aligned start address
    uint32_t ಕೊನೆ;         // line-aligned end address
    uint32_t ಖಂಡ_ಮೊದಲ; // global block index of this range's first block
    uint32_t n_ಖಂಡಗಳು;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

ರಚನೆ l2flush_multi_task {
    ರಚನೆ htp_thread_trace * trace;
    ರಚನೆ l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  ಒಟ್ಟು_ಖಂಡಗಳು;
    uint32_t                  ಖಂಡಗಳು_ಪ್ರತಿ_thread;
};

ಸ್ಥಿರ ಶೂನ್ಯ flush_all_dcache(ರಚನೆ htp_context * ctx) {
    ರಚನೆ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_ಖಂಡ(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, ಗಾತ್ರ(ctx->dirty_ranges));
}

ಸ್ಥಿರ ಶೂನ್ಯ l2flush_multi_ಕೆಲಸಗಾರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ l2flush_multi_task * task = (ರಚನೆ l2flush_multi_task *) ದತ್ತಾಂಶ;
    (ಶೂನ್ಯ) n;

    ಸ್ಥಿರಾಂಕ uint32_t gb_ಮೊದಲ = i * task->ಖಂಡಗಳು_ಪ್ರತಿ_thread;
    uint32_t       gb_ಕೊನೆಯ  = gb_ಮೊದಲ + task->ಖಂಡಗಳು_ಪ್ರತಿ_thread;
    ಆದರೆ (gb_ಕೊನೆಯ > task->ಒಟ್ಟು_ಖಂಡಗಳು) {
        gb_ಕೊನೆಯ = task->ಒಟ್ಟು_ಖಂಡಗಳು;
    }
    ಆದರೆ (gb_ಮೊದಲ >= gb_ಕೊನೆಯ) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_L2FLUSH, gb_ಮೊದಲ);

    ಗಾಗಿ (uint32_t r = 0; r < task->n_ranges; r++) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ l2flush_range * rg = &task->ranges[r];
        ಸ್ಥಿರಾಂಕ uint32_t rb_ಮೊದಲ = rg->ಖಂಡ_ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ uint32_t rb_ಕೊನೆಯ  = rg->ಖಂಡ_ಮೊದಲ + rg->n_ಖಂಡಗಳು;

        ಸ್ಥಿರಾಂಕ uint32_t lo = gb_ಮೊದಲ > rb_ಮೊದಲ ? gb_ಮೊದಲ : rb_ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ uint32_t hi = gb_ಕೊನೆಯ  < rb_ಕೊನೆಯ  ? gb_ಕೊನೆಯ  : rb_ಕೊನೆಯ;
        ಆದರೆ (lo >= hi) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ uint32_t s = rg->ಆರಂಭ + (lo - rb_ಮೊದಲ) * HEX_L2_ಖಂಡ_ಗಾತ್ರ;
        uint32_t       e = rg->ಆರಂಭ + (hi - rb_ಮೊದಲ) * HEX_L2_ಖಂಡ_ಗಾತ್ರ;
        ಆದರೆ (e > rg->ಕೊನೆ) {
            e = rg->ಕೊನೆ;
        }
        hex_l2flush((ಶೂನ್ಯ *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_ಮೊದಲ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ವಿಲೀನ_dirty_ranges(ರಚನೆ htp_context * ctx) {
    ಗಾಗಿ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ಆದರೆ (!r->ಆರಂಭ) ಮುಂದುವರಿ;

        ಗಾಗಿ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            ರಚನೆ htp_dirty_range * s = &ctx->dirty_ranges[j];
            ಆದರೆ (i == j || !s->ಆರಂಭ || r->ಕೊನೆ < s->ಆರಂಭ || s->ಕೊನೆ < r->ಆರಂಭ) {
                j++;
                ಮುಂದುವರಿ;
            }

            r->ಆರಂಭ = MIN(r->ಆರಂಭ, s->ಆರಂಭ);
            r->ಕೊನೆ   = MAX(r->ಕೊನೆ, s->ಕೊನೆ);
            s->ಆರಂಭ = 0;
            s->ಕೊನೆ   = 0;
            j = 0;
        }
    }
}

ಶೂನ್ಯ htp_dcache_mark_dirty_tensors(ರಚನೆ htp_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸ್ಥಿರಾಂಕ * tensors, uint32_t n) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    ಗಾಗಿ (uint32_t i = 0; i < n; i++) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t = tensors[i];
        ಆದರೆ (!t || (t->flags & (HTP_TENSOR_ತೂಕ | HTP_TENSOR_FENCE))) {
            ಮುಂದುವರಿ;
        }

        uint32_t t_ಆರಂಭ = t->ದತ್ತಾಂಶ;
        uint32_t t_ಕೊನೆ   = t_ಆರಂಭ + t->ಗಾತ್ರ_2;

        bool merged = false;
        ಗಾಗಿ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[j];
            ಆದರೆ (!r->ಆರಂಭ) ಮುಂದುವರಿ;

            ಆದರೆ (r->ಆರಂಭ <= t_ಕೊನೆ && t_ಆರಂಭ <= r->ಕೊನೆ) {
                uint32_t new_ಆರಂಭ = (t_ಆರಂಭ < r->ಆರಂಭ) ? t_ಆರಂಭ : r->ಆರಂಭ;
                uint32_t new_ಕೊನೆ   = (t_ಕೊನೆ > r->ಕೊನೆ)     ? t_ಕೊನೆ   : r->ಕೊನೆ;
                r->ಆರಂಭ = new_ಆರಂಭ;
                r->ಕೊನೆ   = new_ಕೊನೆ;
                merged   = true;
            }
        }

        ಆದರೆ (!merged) {
            pending[n_pending++] = t;
        }
    }

    ವಿಲೀನ_dirty_ranges(ctx);

    ಆದರೆ (n_pending == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    uint32_t ಖಾಲಿಗಳು_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t ಸಕ್ರಿಯ_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ಸಕ್ರಿಯ = 0;
    uint32_t n_ಖಾಲಿಗಳು  = 0;
    ಗಾಗಿ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        ಆದರೆ (ctx->dirty_ranges[j].ಆರಂಭ) {
            ಸಕ್ರಿಯ_indices[n_ಸಕ್ರಿಯ++] = j;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಖಾಲಿಗಳು_indices[n_ಖಾಲಿಗಳು++]   = j;
        }
    }

    ಆದರೆ (n_pending <= n_ಖಾಲಿಗಳು) {
        ಗಾಗಿ (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = ಖಾಲಿಗಳು_indices[i];
            ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->ಆರಂಭ = pending[i]->ದತ್ತಾಂಶ;
            r->ಕೊನೆ   = pending[i]->ದತ್ತಾಂಶ + pending[i]->ಗಾತ್ರ_2;
        }
        ವಿಲೀನ_dirty_ranges(ctx);
        ಹಿಂತಿರುಗಿ;
    }

    uint32_t n_evict = n_pending - n_ಖಾಲಿಗಳು;
    uint32_t ಒಟ್ಟು_evict_ಗಾತ್ರ = 0;
    ಗಾಗಿ (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = ಸಕ್ರಿಯ_indices[i];
        ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[idx];
        ಒಟ್ಟು_evict_ಗಾತ್ರ += r->ಕೊನೆ - r->ಆರಂಭ;
    }

    ಆದರೆ (ಒಟ್ಟು_evict_ಗಾತ್ರ > HEX_L2_FLUSH_ALL_ಮಿತಿ) {
        flush_all_dcache(ctx);
        ಗಾಗಿ (uint32_t i = 0; i < n_pending; i++) {
            ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->ಆರಂಭ = pending[i]->ದತ್ತಾಂಶ;
            r->ಕೊನೆ   = pending[i]->ದತ್ತಾಂಶ + pending[i]->ಗಾತ್ರ_2;
        }
        ವಿಲೀನ_dirty_ranges(ctx);
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ಒಟ್ಟು_evict_ಗಾತ್ರ > HEX_L2_FLUSH_WQ_ಮಿತಿ && ctx->n_ಎಳೆಗಳು > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        ರಚನೆ l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t ಖಂಡ_acc = 0;
        ಗಾಗಿ (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = ಸಕ್ರಿಯ_indices[i];
            ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[idx];

            ರಚನೆ l2flush_range * rg = &task.ranges[i];
            rg->ಆರಂಭ = hex_align_ಕೆಳಕ್ಕೆ((size_t) r->ಆರಂಭ, HEX_L2_ಸಾಲು_ಗಾತ್ರ);
            rg->ಕೊನೆ   = hex_align_ಮೇಲೆ((size_t) r->ಕೊನೆ, HEX_L2_ಸಾಲು_ಗಾತ್ರ);
            rg->ಖಂಡ_ಮೊದಲ = ಖಂಡ_acc;
            rg->n_ಖಂಡಗಳು = (rg->ಕೊನೆ - rg->ಆರಂಭ + HEX_L2_ಖಂಡ_ಗಾತ್ರ - 1) / HEX_L2_ಖಂಡ_ಗಾತ್ರ;
            ಖಂಡ_acc += rg->n_ಖಂಡಗಳು;
        }

        task.ಒಟ್ಟು_ಖಂಡಗಳು      = ಖಂಡ_acc;
        task.ಖಂಡಗಳು_ಪ್ರತಿ_thread = fastdiv(ಖಂಡ_acc + ctx->n_ಎಳೆಗಳು - 1, &ctx->n_ಎಳೆಗಳು_div);

        work_ಸರತಿ_ಸತತ(ctx->work_ಸರತಿ, l2flush_multi_ಕೆಲಸಗಾರ, &task, ctx->n_ಎಳೆಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ರಚನೆ htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        ಗಾಗಿ (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = ಸಕ್ರಿಯ_indices[i];
            ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t ಗಾತ್ರ_2 = r->ಕೊನೆ - r->ಆರಂಭ;
            hex_l2flush((ಶೂನ್ಯ *) (uintptr_t) r->ಆರಂಭ, ಗಾತ್ರ_2);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    ಗಾಗಿ (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = ಸಕ್ರಿಯ_indices[i];
        ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->ಆರಂಭ = pending[i]->ದತ್ತಾಂಶ;
        r->ಕೊನೆ   = pending[i]->ದತ್ತಾಂಶ + pending[i]->ಗಾತ್ರ_2;
    }

    ಗಾಗಿ (uint32_t i = 0; i < n_ಖಾಲಿಗಳು; i++) {
        uint32_t idx = ಖಾಲಿಗಳು_indices[i];
        ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->ಆರಂಭ = pending[n_evict + i]->ದತ್ತಾಂಶ;
        r->ಕೊನೆ   = pending[n_evict + i]->ದತ್ತಾಂಶ + pending[n_evict + i]->ಗಾತ್ರ_2;
    }

    ವಿಲೀನ_dirty_ranges(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ make_tensor_clean(ರಚನೆ htp_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t) {
    ಆದರೆ (!t || (t->flags & (HTP_TENSOR_ತೂಕ | HTP_TENSOR_FENCE))) {
        ಹಿಂತಿರುಗಿ;
    }

    uint32_t t_ಆರಂಭ = t->ದತ್ತಾಂಶ;
    uint32_t t_ಕೊನೆ   = t_ಆರಂಭ + t->ಗಾತ್ರ_2;

    ಗಾಗಿ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ಆದರೆ (!r->ಆರಂಭ) ಮುಂದುವರಿ;

        ಆದರೆ (r->ಆರಂಭ < t_ಕೊನೆ && t_ಆರಂಭ < r->ಕೊನೆ) {
            ಆದರೆ (t_ಆರಂಭ <= r->ಆರಂಭ && r->ಕೊನೆ <= t_ಕೊನೆ) {
                r->ಆರಂಭ = 0;
                r->ಕೊನೆ   = 0;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (t_ಆರಂಭ <= r->ಆರಂಭ) {
                r->ಆರಂಭ = t_ಕೊನೆ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (r->ಕೊನೆ <= t_ಕೊನೆ) {
                r->ಕೊನೆ = t_ಆರಂಭ;
            }
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool is_tensor_dirty(ರಚನೆ htp_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t) {
    ಆದರೆ (!t || (t->flags & (HTP_TENSOR_ತೂಕ | HTP_TENSOR_FENCE))) {
        ಹಿಂತಿರುಗಿ false;
    }

    uint32_t t_ಆರಂಭ = t->ದತ್ತಾಂಶ;
    uint32_t t_ಕೊನೆ   = t_ಆರಂಭ + t->ಗಾತ್ರ_2;

    ಗಾಗಿ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ಆದರೆ (!r->ಆರಂಭ) ಮುಂದುವರಿ;

        ಆದರೆ (r->ಆರಂಭ < t_ಕೊನೆ && t_ಆರಂಭ < r->ಕೊನೆ) {
            ಹಿಂತಿರುಗಿ true;
        }
    }
    ಹಿಂತಿರುಗಿ false;
}

ಸ್ಥಿರ ಶೂನ್ಯ flush_dirty_ranges(ರಚನೆ htp_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_dirty_range * ranges, uint32_t n_ranges, uint64_t ಒಟ್ಟು_dirty) {
    ಆದರೆ (ಒಟ್ಟು_dirty >= HEX_L2_FLUSH_WQ_ಮಿತಿ && ctx->n_ಎಳೆಗಳು > 1) {
        ರಚನೆ l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t ಖಂಡ_acc = 0;
        ಗಾಗಿ (uint32_t i = 0; i < n_ranges; i++) {
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_dirty_range * r = &ranges[i];
            ರಚನೆ l2flush_range * rg = &task.ranges[i];
            rg->ಆರಂಭ = hex_align_ಕೆಳಕ್ಕೆ((size_t) r->ಆರಂಭ, HEX_L2_ಸಾಲು_ಗಾತ್ರ);
            rg->ಕೊನೆ   = hex_align_ಮೇಲೆ((size_t) r->ಕೊನೆ, HEX_L2_ಸಾಲು_ಗಾತ್ರ);
            rg->ಖಂಡ_ಮೊದಲ = ಖಂಡ_acc;
            rg->n_ಖಂಡಗಳು = (rg->ಕೊನೆ - rg->ಆರಂಭ + HEX_L2_ಖಂಡ_ಗಾತ್ರ - 1) / HEX_L2_ಖಂಡ_ಗಾತ್ರ;
            ಖಂಡ_acc += rg->n_ಖಂಡಗಳು;
        }

        task.ಒಟ್ಟು_ಖಂಡಗಳು      = ಖಂಡ_acc;
        task.ಖಂಡಗಳು_ಪ್ರತಿ_thread = fastdiv(ಖಂಡ_acc + ctx->n_ಎಳೆಗಳು - 1, &ctx->n_ಎಳೆಗಳು_div);

        work_ಸರತಿ_ಸತತ(ctx->work_ಸರತಿ, l2flush_multi_ಕೆಲಸಗಾರ, &task, ctx->n_ಎಳೆಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ರಚನೆ htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        ಗಾಗಿ (uint32_t i = 0; i < n_ranges; i++) {
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_dirty_range * r = &ranges[i];
            hex_l2flush((ಶೂನ್ಯ *) (uintptr_t) r->ಆರಂಭ, r->ಕೊನೆ - r->ಆರಂಭ);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

ಶೂನ್ಯ htp_dcache_flush_dirty_ranges(ರಚನೆ htp_context * ctx) {
    ರಚನೆ htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t ಒಟ್ಟು_dirty = 0;

    ಗಾಗಿ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ಆದರೆ (!r->ಆರಂಭ) {
            ಮುಂದುವರಿ;
        }
        ranges[n_ranges++] = *r;
        ಒಟ್ಟು_dirty += r->ಕೊನೆ - r->ಆರಂಭ;
    }

    ಆದರೆ (ಒಟ್ಟು_dirty == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ಒಟ್ಟು_dirty > HEX_L2_FLUSH_ALL_ಮಿತಿ) {
        flush_all_dcache(ctx);
        ಹಿಂತಿರುಗಿ;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, ಒಟ್ಟು_dirty);
    memset(ctx->dirty_ranges, 0, ಗಾತ್ರ(ctx->dirty_ranges));
}

ಶೂನ್ಯ htp_dcache_flush_dirty_tensors(ರಚನೆ htp_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸ್ಥಿರಾಂಕ * tensors, uint32_t n) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    ರಚನೆ htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t ಒಟ್ಟು_dirty = 0;

    ಗಾಗಿ (uint32_t i = 0; i < n; i++) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t = tensors[i];
        ಆದರೆ (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].ಆರಂಭ = t->ದತ್ತಾಂಶ;
            ranges[n_dirty - 1].ಕೊನೆ   = t->ದತ್ತಾಂಶ + t->ಗಾತ್ರ_2;
            ಒಟ್ಟು_dirty += t->ಗಾತ್ರ_2;
        }
    }

    ಆದರೆ (ಒಟ್ಟು_dirty == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ಒಟ್ಟು_dirty > HEX_L2_FLUSH_ALL_ಮಿತಿ) {
        flush_all_dcache(ctx);
        ಹಿಂತಿರುಗಿ;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, ಒಟ್ಟು_dirty);
    ಗಾಗಿ (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
