#ਅਨ੍ਤਰ੍ਭਾਵ "htp-dcache.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <qurt.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_memory.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "work-queue.h"

ਬਣਤਰ l2flush_range {
    uint32_t ਸ਼ੁਰੂ;       // line-aligned start address
    uint32_t ਸਿਰਾ;         // line-aligned end address
    uint32_t ਖੰਡ_ਪਹਿਲਾ; // global block index of this range's first block
    uint32_t n_ਖੰਡਾਂ;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

ਬਣਤਰ l2flush_multi_task {
    ਬਣਤਰ htp_thread_trace * trace;
    ਬਣਤਰ l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  ਕੁੱਲ_ਖੰਡਾਂ;
    uint32_t                  ਖੰਡਾਂ_ਪ੍ਰਤੀ_thread;
};

ਸਥਿਰ ਖਾਲੀ flush_all_dcache(ਬਣਤਰ htp_context * ctx) {
    ਬਣਤਰ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_ਖੰਡ(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, ਆਕਾਰ(ctx->dirty_ranges));
}

ਸਥਿਰ ਖਾਲੀ l2flush_multi_ਕਾਮਾ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ l2flush_multi_task * task = (ਬਣਤਰ l2flush_multi_task *) ਅੰਕੜੇ;
    (ਖਾਲੀ) n;

    ਅਚਲ uint32_t gb_ਪਹਿਲਾ = i * task->ਖੰਡਾਂ_ਪ੍ਰਤੀ_thread;
    uint32_t       gb_ਆਖਰੀ  = gb_ਪਹਿਲਾ + task->ਖੰਡਾਂ_ਪ੍ਰਤੀ_thread;
    ਜੇ (gb_ਆਖਰੀ > task->ਕੁੱਲ_ਖੰਡਾਂ) {
        gb_ਆਖਰੀ = task->ਕੁੱਲ_ਖੰਡਾਂ;
    }
    ਜੇ (gb_ਪਹਿਲਾ >= gb_ਆਖਰੀ) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_L2FLUSH, gb_ਪਹਿਲਾ);

    ਲਈ (uint32_t r = 0; r < task->n_ranges; r++) {
        ਅਚਲ ਬਣਤਰ l2flush_range * rg = &task->ranges[r];
        ਅਚਲ uint32_t rb_ਪਹਿਲਾ = rg->ਖੰਡ_ਪਹਿਲਾ;
        ਅਚਲ uint32_t rb_ਆਖਰੀ  = rg->ਖੰਡ_ਪਹਿਲਾ + rg->n_ਖੰਡਾਂ;

        ਅਚਲ uint32_t lo = gb_ਪਹਿਲਾ > rb_ਪਹਿਲਾ ? gb_ਪਹਿਲਾ : rb_ਪਹਿਲਾ;
        ਅਚਲ uint32_t hi = gb_ਆਖਰੀ  < rb_ਆਖਰੀ  ? gb_ਆਖਰੀ  : rb_ਆਖਰੀ;
        ਜੇ (lo >= hi) {
            ਜਾਰੀ;
        }

        ਅਚਲ uint32_t s = rg->ਸ਼ੁਰੂ + (lo - rb_ਪਹਿਲਾ) * HEX_L2_ਖੰਡ_ਮਾਪ;
        uint32_t       e = rg->ਸ਼ੁਰੂ + (hi - rb_ਪਹਿਲਾ) * HEX_L2_ਖੰਡ_ਮਾਪ;
        ਜੇ (e > rg->ਸਿਰਾ) {
            e = rg->ਸਿਰਾ;
        }
        hex_l2flush((ਖਾਲੀ *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_ਪਹਿਲਾ);
}

ਸਥਿਰ ਖਾਲੀ ਮਿਲਾਓ_dirty_ranges(ਬਣਤਰ htp_context * ctx) {
    ਲਈ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ਜੇ (!r->ਸ਼ੁਰੂ) ਜਾਰੀ;

        ਲਈ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            ਬਣਤਰ htp_dirty_range * s = &ctx->dirty_ranges[j];
            ਜੇ (i == j || !s->ਸ਼ੁਰੂ || r->ਸਿਰਾ < s->ਸ਼ੁਰੂ || s->ਸਿਰਾ < r->ਸ਼ੁਰੂ) {
                j++;
                ਜਾਰੀ;
            }

            r->ਸ਼ੁਰੂ = MIN(r->ਸ਼ੁਰੂ, s->ਸ਼ੁਰੂ);
            r->ਸਿਰਾ   = MAX(r->ਸਿਰਾ, s->ਸਿਰਾ);
            s->ਸ਼ੁਰੂ = 0;
            s->ਸਿਰਾ   = 0;
            j = 0;
        }
    }
}

ਖਾਲੀ htp_dcache_mark_dirty_tensors(ਬਣਤਰ htp_context * ctx, ਅਚਲ ਬਣਤਰ htp_tensor * ਅਚਲ * tensors, uint32_t n) {
    ਅਚਲ ਬਣਤਰ htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    ਲਈ (uint32_t i = 0; i < n; i++) {
        ਅਚਲ ਬਣਤਰ htp_tensor * t = tensors[i];
        ਜੇ (!t || (t->flags & (HTP_TENSOR_ਭਾਰ | HTP_TENSOR_FENCE))) {
            ਜਾਰੀ;
        }

        uint32_t t_ਸ਼ੁਰੂ = t->ਅੰਕੜੇ;
        uint32_t t_ਸਿਰਾ   = t_ਸ਼ੁਰੂ + t->ਮਾਪ;

        bool merged = false;
        ਲਈ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[j];
            ਜੇ (!r->ਸ਼ੁਰੂ) ਜਾਰੀ;

            ਜੇ (r->ਸ਼ੁਰੂ <= t_ਸਿਰਾ && t_ਸ਼ੁਰੂ <= r->ਸਿਰਾ) {
                uint32_t new_ਸ਼ੁਰੂ = (t_ਸ਼ੁਰੂ < r->ਸ਼ੁਰੂ) ? t_ਸ਼ੁਰੂ : r->ਸ਼ੁਰੂ;
                uint32_t new_ਸਿਰਾ   = (t_ਸਿਰਾ > r->ਸਿਰਾ)     ? t_ਸਿਰਾ   : r->ਸਿਰਾ;
                r->ਸ਼ੁਰੂ = new_ਸ਼ੁਰੂ;
                r->ਸਿਰਾ   = new_ਸਿਰਾ;
                merged   = true;
            }
        }

        ਜੇ (!merged) {
            pending[n_pending++] = t;
        }
    }

    ਮਿਲਾਓ_dirty_ranges(ctx);

    ਜੇ (n_pending == 0) {
        ਵਾਪਸ;
    }

    uint32_t ਖਾਲੀ_ਥਾਵਾਂ_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t ਸਰਗਰਮ_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ਸਰਗਰਮ = 0;
    uint32_t n_ਖਾਲੀ_ਥਾਵਾਂ  = 0;
    ਲਈ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        ਜੇ (ctx->dirty_ranges[j].ਸ਼ੁਰੂ) {
            ਸਰਗਰਮ_indices[n_ਸਰਗਰਮ++] = j;
        } ਨਹੀਂਤਾਂ {
            ਖਾਲੀ_ਥਾਵਾਂ_indices[n_ਖਾਲੀ_ਥਾਵਾਂ++]   = j;
        }
    }

    ਜੇ (n_pending <= n_ਖਾਲੀ_ਥਾਵਾਂ) {
        ਲਈ (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = ਖਾਲੀ_ਥਾਵਾਂ_indices[i];
            ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->ਸ਼ੁਰੂ = pending[i]->ਅੰਕੜੇ;
            r->ਸਿਰਾ   = pending[i]->ਅੰਕੜੇ + pending[i]->ਮਾਪ;
        }
        ਮਿਲਾਓ_dirty_ranges(ctx);
        ਵਾਪਸ;
    }

    uint32_t n_evict = n_pending - n_ਖਾਲੀ_ਥਾਵਾਂ;
    uint32_t ਕੁੱਲ_evict_ਮਾਪ = 0;
    ਲਈ (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = ਸਰਗਰਮ_indices[i];
        ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[idx];
        ਕੁੱਲ_evict_ਮਾਪ += r->ਸਿਰਾ - r->ਸ਼ੁਰੂ;
    }

    ਜੇ (ਕੁੱਲ_evict_ਮਾਪ > HEX_L2_FLUSH_ALL_ਹੱਦ) {
        flush_all_dcache(ctx);
        ਲਈ (uint32_t i = 0; i < n_pending; i++) {
            ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->ਸ਼ੁਰੂ = pending[i]->ਅੰਕੜੇ;
            r->ਸਿਰਾ   = pending[i]->ਅੰਕੜੇ + pending[i]->ਮਾਪ;
        }
        ਮਿਲਾਓ_dirty_ranges(ctx);
        ਵਾਪਸ;
    }

    ਜੇ (ਕੁੱਲ_evict_ਮਾਪ > HEX_L2_FLUSH_WQ_ਹੱਦ && ctx->n_ਧਾਗੇ > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        ਬਣਤਰ l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t ਖੰਡ_acc = 0;
        ਲਈ (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = ਸਰਗਰਮ_indices[i];
            ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[idx];

            ਬਣਤਰ l2flush_range * rg = &task.ranges[i];
            rg->ਸ਼ੁਰੂ = hex_align_ਥੱਲੇ((size_t) r->ਸ਼ੁਰੂ, HEX_L2_ਸਤਰ_ਮਾਪ);
            rg->ਸਿਰਾ   = hex_align_ਉੱਪਰ((size_t) r->ਸਿਰਾ, HEX_L2_ਸਤਰ_ਮਾਪ);
            rg->ਖੰਡ_ਪਹਿਲਾ = ਖੰਡ_acc;
            rg->n_ਖੰਡਾਂ = (rg->ਸਿਰਾ - rg->ਸ਼ੁਰੂ + HEX_L2_ਖੰਡ_ਮਾਪ - 1) / HEX_L2_ਖੰਡ_ਮਾਪ;
            ਖੰਡ_acc += rg->n_ਖੰਡਾਂ;
        }

        task.ਕੁੱਲ_ਖੰਡਾਂ      = ਖੰਡ_acc;
        task.ਖੰਡਾਂ_ਪ੍ਰਤੀ_thread = fastdiv(ਖੰਡ_acc + ctx->n_ਧਾਗੇ - 1, &ctx->n_ਧਾਗੇ_div);

        work_ਕਤਾਰ_ਲੜੀਵਾਰ(ctx->work_ਕਤਾਰ, l2flush_multi_ਕਾਮਾ, &task, ctx->n_ਧਾਗੇ);
    } ਨਹੀਂਤਾਂ {
        ਬਣਤਰ htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        ਲਈ (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = ਸਰਗਰਮ_indices[i];
            ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t ਮਾਪ = r->ਸਿਰਾ - r->ਸ਼ੁਰੂ;
            hex_l2flush((ਖਾਲੀ *) (uintptr_t) r->ਸ਼ੁਰੂ, ਮਾਪ);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    ਲਈ (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = ਸਰਗਰਮ_indices[i];
        ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->ਸ਼ੁਰੂ = pending[i]->ਅੰਕੜੇ;
        r->ਸਿਰਾ   = pending[i]->ਅੰਕੜੇ + pending[i]->ਮਾਪ;
    }

    ਲਈ (uint32_t i = 0; i < n_ਖਾਲੀ_ਥਾਵਾਂ; i++) {
        uint32_t idx = ਖਾਲੀ_ਥਾਵਾਂ_indices[i];
        ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->ਸ਼ੁਰੂ = pending[n_evict + i]->ਅੰਕੜੇ;
        r->ਸਿਰਾ   = pending[n_evict + i]->ਅੰਕੜੇ + pending[n_evict + i]->ਮਾਪ;
    }

    ਮਿਲਾਓ_dirty_ranges(ctx);
}

ਸਥਿਰ ਖਾਲੀ make_tensor_clean(ਬਣਤਰ htp_context * ctx, ਅਚਲ ਬਣਤਰ htp_tensor * t) {
    ਜੇ (!t || (t->flags & (HTP_TENSOR_ਭਾਰ | HTP_TENSOR_FENCE))) {
        ਵਾਪਸ;
    }

    uint32_t t_ਸ਼ੁਰੂ = t->ਅੰਕੜੇ;
    uint32_t t_ਸਿਰਾ   = t_ਸ਼ੁਰੂ + t->ਮਾਪ;

    ਲਈ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ਜੇ (!r->ਸ਼ੁਰੂ) ਜਾਰੀ;

        ਜੇ (r->ਸ਼ੁਰੂ < t_ਸਿਰਾ && t_ਸ਼ੁਰੂ < r->ਸਿਰਾ) {
            ਜੇ (t_ਸ਼ੁਰੂ <= r->ਸ਼ੁਰੂ && r->ਸਿਰਾ <= t_ਸਿਰਾ) {
                r->ਸ਼ੁਰੂ = 0;
                r->ਸਿਰਾ   = 0;
            } ਨਹੀਂਤਾਂ ਜੇ (t_ਸ਼ੁਰੂ <= r->ਸ਼ੁਰੂ) {
                r->ਸ਼ੁਰੂ = t_ਸਿਰਾ;
            } ਨਹੀਂਤਾਂ ਜੇ (r->ਸਿਰਾ <= t_ਸਿਰਾ) {
                r->ਸਿਰਾ = t_ਸ਼ੁਰੂ;
            }
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool is_tensor_dirty(ਬਣਤਰ htp_context * ctx, ਅਚਲ ਬਣਤਰ htp_tensor * t) {
    ਜੇ (!t || (t->flags & (HTP_TENSOR_ਭਾਰ | HTP_TENSOR_FENCE))) {
        ਵਾਪਸ false;
    }

    uint32_t t_ਸ਼ੁਰੂ = t->ਅੰਕੜੇ;
    uint32_t t_ਸਿਰਾ   = t_ਸ਼ੁਰੂ + t->ਮਾਪ;

    ਲਈ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ਜੇ (!r->ਸ਼ੁਰੂ) ਜਾਰੀ;

        ਜੇ (r->ਸ਼ੁਰੂ < t_ਸਿਰਾ && t_ਸ਼ੁਰੂ < r->ਸਿਰਾ) {
            ਵਾਪਸ true;
        }
    }
    ਵਾਪਸ false;
}

ਸਥਿਰ ਖਾਲੀ flush_dirty_ranges(ਬਣਤਰ htp_context * ctx, ਅਚਲ ਬਣਤਰ htp_dirty_range * ranges, uint32_t n_ranges, uint64_t ਕੁੱਲ_dirty) {
    ਜੇ (ਕੁੱਲ_dirty >= HEX_L2_FLUSH_WQ_ਹੱਦ && ctx->n_ਧਾਗੇ > 1) {
        ਬਣਤਰ l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t ਖੰਡ_acc = 0;
        ਲਈ (uint32_t i = 0; i < n_ranges; i++) {
            ਅਚਲ ਬਣਤਰ htp_dirty_range * r = &ranges[i];
            ਬਣਤਰ l2flush_range * rg = &task.ranges[i];
            rg->ਸ਼ੁਰੂ = hex_align_ਥੱਲੇ((size_t) r->ਸ਼ੁਰੂ, HEX_L2_ਸਤਰ_ਮਾਪ);
            rg->ਸਿਰਾ   = hex_align_ਉੱਪਰ((size_t) r->ਸਿਰਾ, HEX_L2_ਸਤਰ_ਮਾਪ);
            rg->ਖੰਡ_ਪਹਿਲਾ = ਖੰਡ_acc;
            rg->n_ਖੰਡਾਂ = (rg->ਸਿਰਾ - rg->ਸ਼ੁਰੂ + HEX_L2_ਖੰਡ_ਮਾਪ - 1) / HEX_L2_ਖੰਡ_ਮਾਪ;
            ਖੰਡ_acc += rg->n_ਖੰਡਾਂ;
        }

        task.ਕੁੱਲ_ਖੰਡਾਂ      = ਖੰਡ_acc;
        task.ਖੰਡਾਂ_ਪ੍ਰਤੀ_thread = fastdiv(ਖੰਡ_acc + ctx->n_ਧਾਗੇ - 1, &ctx->n_ਧਾਗੇ_div);

        work_ਕਤਾਰ_ਲੜੀਵਾਰ(ctx->work_ਕਤਾਰ, l2flush_multi_ਕਾਮਾ, &task, ctx->n_ਧਾਗੇ);
    } ਨਹੀਂਤਾਂ {
        ਬਣਤਰ htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        ਲਈ (uint32_t i = 0; i < n_ranges; i++) {
            ਅਚਲ ਬਣਤਰ htp_dirty_range * r = &ranges[i];
            hex_l2flush((ਖਾਲੀ *) (uintptr_t) r->ਸ਼ੁਰੂ, r->ਸਿਰਾ - r->ਸ਼ੁਰੂ);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

ਖਾਲੀ htp_dcache_flush_dirty_ranges(ਬਣਤਰ htp_context * ctx) {
    ਬਣਤਰ htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t ਕੁੱਲ_dirty = 0;

    ਲਈ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ਅਚਲ ਬਣਤਰ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ਜੇ (!r->ਸ਼ੁਰੂ) {
            ਜਾਰੀ;
        }
        ranges[n_ranges++] = *r;
        ਕੁੱਲ_dirty += r->ਸਿਰਾ - r->ਸ਼ੁਰੂ;
    }

    ਜੇ (ਕੁੱਲ_dirty == 0) {
        ਵਾਪਸ;
    }

    ਜੇ (ਕੁੱਲ_dirty > HEX_L2_FLUSH_ALL_ਹੱਦ) {
        flush_all_dcache(ctx);
        ਵਾਪਸ;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, ਕੁੱਲ_dirty);
    memset(ctx->dirty_ranges, 0, ਆਕਾਰ(ctx->dirty_ranges));
}

ਖਾਲੀ htp_dcache_flush_dirty_tensors(ਬਣਤਰ htp_context * ctx, ਅਚਲ ਬਣਤਰ htp_tensor * ਅਚਲ * tensors, uint32_t n) {
    ਅਚਲ ਬਣਤਰ htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    ਬਣਤਰ htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t ਕੁੱਲ_dirty = 0;

    ਲਈ (uint32_t i = 0; i < n; i++) {
        ਅਚਲ ਬਣਤਰ htp_tensor * t = tensors[i];
        ਜੇ (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].ਸ਼ੁਰੂ = t->ਅੰਕੜੇ;
            ranges[n_dirty - 1].ਸਿਰਾ   = t->ਅੰਕੜੇ + t->ਮਾਪ;
            ਕੁੱਲ_dirty += t->ਮਾਪ;
        }
    }

    ਜੇ (ਕੁੱਲ_dirty == 0) {
        ਵਾਪਸ;
    }

    ਜੇ (ਕੁੱਲ_dirty > HEX_L2_FLUSH_ALL_ਹੱਦ) {
        flush_all_dcache(ctx);
        ਵਾਪਸ;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, ਕੁੱਲ_dirty);
    ਲਈ (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
