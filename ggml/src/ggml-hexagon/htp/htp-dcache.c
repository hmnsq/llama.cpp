#ضمن "htp-dcache.h"

#ضمن <qurt.h>
#ضمن <qurt_memory.h>
#ضمن <HAP_farf.h>

#ضمن "hex-common.h"
#ضمن "hex-utils.h"
#ضمن "hex-fastdiv.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "work-queue.h"

بنية l2flush_range {
    uint32_t شروع;       // line-aligned start address
    uint32_t سرا;         // line-aligned end address
    uint32_t بلاک_پہلا; // global block index of this range's first block
    uint32_t n_بلاکاں;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

بنية l2flush_multi_task {
    بنية htp_thread_trace * trace;
    بنية l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  کل_بلاکاں;
    uint32_t                  بلاکاں_فی_thread;
};

ساكن فراغ flush_all_dcache(بنية htp_context * ctx) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_بلاک(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

ساكن فراغ l2flush_multi_کامہ(بنانشانی صحيح n, بنانشانی صحيح i, فراغ * انکڑے) {
    بنية l2flush_multi_task * task = (بنية l2flush_multi_task *) انکڑے;
    (فراغ) n;

    ثابت uint32_t gb_پہلا = i * task->بلاکاں_فی_thread;
    uint32_t       gb_آخری  = gb_پہلا + task->بلاکاں_فی_thread;
    إذا (gb_آخری > task->کل_بلاکاں) {
        gb_آخری = task->کل_بلاکاں;
    }
    إذا (gb_پہلا >= gb_آخری) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_L2FLUSH, gb_پہلا);

    لكل (uint32_t r = 0; r < task->n_ranges; r++) {
        ثابت بنية l2flush_range * rg = &task->ranges[r];
        ثابت uint32_t rb_پہلا = rg->بلاک_پہلا;
        ثابت uint32_t rb_آخری  = rg->بلاک_پہلا + rg->n_بلاکاں;

        ثابت uint32_t lo = gb_پہلا > rb_پہلا ? gb_پہلا : rb_پہلا;
        ثابت uint32_t hi = gb_آخری  < rb_آخری  ? gb_آخری  : rb_آخری;
        إذا (lo >= hi) {
            استمر;
        }

        ثابت uint32_t s = rg->شروع + (lo - rb_پہلا) * HEX_L2_بلاک_ناپ;
        uint32_t       e = rg->شروع + (hi - rb_پہلا) * HEX_L2_بلاک_ناپ;
        إذا (e > rg->سرا) {
            e = rg->سرا;
        }
        hex_l2flush((فراغ *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_پہلا);
}

ساكن فراغ ملاؤ_dirty_ranges(بنية htp_context * ctx) {
    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروع) استمر;

        لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            بنية htp_dirty_range * s = &ctx->dirty_ranges[j];
            إذا (i == j || !s->شروع || r->سرا < s->شروع || s->سرا < r->شروع) {
                j++;
                استمر;
            }

            r->شروع = MIN(r->شروع, s->شروع);
            r->سرا   = MAX(r->سرا, s->سرا);
            s->شروع = 0;
            s->سرا   = 0;
            j = 0;
        }
    }
}

فراغ htp_dcache_mark_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n) {
    ثابت بنية htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    لكل (uint32_t i = 0; i < n; i++) {
        ثابت بنية htp_tensor * t = tensors[i];
        إذا (!t || (t->flags & (HTP_TENSOR_بھار | HTP_TENSOR_FENCE))) {
            استمر;
        }

        uint32_t t_شروع = t->انکڑے;
        uint32_t t_سرا   = t_شروع + t->ناپ;

        bool merged = false;
        لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            بنية htp_dirty_range * r = &ctx->dirty_ranges[j];
            إذا (!r->شروع) استمر;

            إذا (r->شروع <= t_سرا && t_شروع <= r->سرا) {
                uint32_t new_شروع = (t_شروع < r->شروع) ? t_شروع : r->شروع;
                uint32_t new_سرا   = (t_سرا > r->سرا)     ? t_سرا   : r->سرا;
                r->شروع = new_شروع;
                r->سرا   = new_سرا;
                merged   = true;
            }
        }

        إذا (!merged) {
            pending[n_pending++] = t;
        }
    }

    ملاؤ_dirty_ranges(ctx);

    إذا (n_pending == 0) {
        أرجع;
    }

    uint32_t خالی_تھاواں_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t سرگرم_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_سرگرم = 0;
    uint32_t n_خالی_تھاواں  = 0;
    لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        إذا (ctx->dirty_ranges[j].شروع) {
            سرگرم_indices[n_سرگرم++] = j;
        } وإلا {
            خالی_تھاواں_indices[n_خالی_تھاواں++]   = j;
        }
    }

    إذا (n_pending <= n_خالی_تھاواں) {
        لكل (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = خالی_تھاواں_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->شروع = pending[i]->انکڑے;
            r->سرا   = pending[i]->انکڑے + pending[i]->ناپ;
        }
        ملاؤ_dirty_ranges(ctx);
        أرجع;
    }

    uint32_t n_evict = n_pending - n_خالی_تھاواں;
    uint32_t کل_evict_ناپ = 0;
    لكل (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = سرگرم_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        کل_evict_ناپ += r->سرا - r->شروع;
    }

    إذا (کل_evict_ناپ > HEX_L2_FLUSH_ALL_حد) {
        flush_all_dcache(ctx);
        لكل (uint32_t i = 0; i < n_pending; i++) {
            بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->شروع = pending[i]->انکڑے;
            r->سرا   = pending[i]->انکڑے + pending[i]->ناپ;
        }
        ملاؤ_dirty_ranges(ctx);
        أرجع;
    }

    إذا (کل_evict_ناپ > HEX_L2_FLUSH_WQ_حد && ctx->n_دھاگے > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        بنية l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t بلاک_acc = 0;
        لكل (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = سرگرم_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];

            بنية l2flush_range * rg = &task.ranges[i];
            rg->شروع = hex_align_تھلے((size_t) r->شروع, HEX_L2_سطر_ناپ);
            rg->سرا   = hex_align_اتے((size_t) r->سرا, HEX_L2_سطر_ناپ);
            rg->بلاک_پہلا = بلاک_acc;
            rg->n_بلاکاں = (rg->سرا - rg->شروع + HEX_L2_بلاک_ناپ - 1) / HEX_L2_بلاک_ناپ;
            بلاک_acc += rg->n_بلاکاں;
        }

        task.کل_بلاکاں      = بلاک_acc;
        task.بلاکاں_فی_thread = fastdiv(بلاک_acc + ctx->n_دھاگے - 1, &ctx->n_دھاگے_div);

        work_قطار_لگاتار(ctx->work_قطار, l2flush_multi_کامہ, &task, ctx->n_دھاگے);
    } وإلا {
        بنية htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        لكل (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = سرگرم_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t ناپ = r->سرا - r->شروع;
            hex_l2flush((فراغ *) (uintptr_t) r->شروع, ناپ);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    لكل (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = سرگرم_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->شروع = pending[i]->انکڑے;
        r->سرا   = pending[i]->انکڑے + pending[i]->ناپ;
    }

    لكل (uint32_t i = 0; i < n_خالی_تھاواں; i++) {
        uint32_t idx = خالی_تھاواں_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->شروع = pending[n_evict + i]->انکڑے;
        r->سرا   = pending[n_evict + i]->انکڑے + pending[n_evict + i]->ناپ;
    }

    ملاؤ_dirty_ranges(ctx);
}

ساكن فراغ make_tensor_clean(بنية htp_context * ctx, ثابت بنية htp_tensor * t) {
    إذا (!t || (t->flags & (HTP_TENSOR_بھار | HTP_TENSOR_FENCE))) {
        أرجع;
    }

    uint32_t t_شروع = t->انکڑے;
    uint32_t t_سرا   = t_شروع + t->ناپ;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروع) استمر;

        إذا (r->شروع < t_سرا && t_شروع < r->سرا) {
            إذا (t_شروع <= r->شروع && r->سرا <= t_سرا) {
                r->شروع = 0;
                r->سرا   = 0;
            } وإلا إذا (t_شروع <= r->شروع) {
                r->شروع = t_سرا;
            } وإلا إذا (r->سرا <= t_سرا) {
                r->سرا = t_شروع;
            }
        }
    }
}

ساكن مضمن bool is_tensor_dirty(بنية htp_context * ctx, ثابت بنية htp_tensor * t) {
    إذا (!t || (t->flags & (HTP_TENSOR_بھار | HTP_TENSOR_FENCE))) {
        أرجع false;
    }

    uint32_t t_شروع = t->انکڑے;
    uint32_t t_سرا   = t_شروع + t->ناپ;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروع) استمر;

        إذا (r->شروع < t_سرا && t_شروع < r->سرا) {
            أرجع true;
        }
    }
    أرجع false;
}

ساكن فراغ flush_dirty_ranges(بنية htp_context * ctx, ثابت بنية htp_dirty_range * ranges, uint32_t n_ranges, uint64_t کل_dirty) {
    إذا (کل_dirty >= HEX_L2_FLUSH_WQ_حد && ctx->n_دھاگے > 1) {
        بنية l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t بلاک_acc = 0;
        لكل (uint32_t i = 0; i < n_ranges; i++) {
            ثابت بنية htp_dirty_range * r = &ranges[i];
            بنية l2flush_range * rg = &task.ranges[i];
            rg->شروع = hex_align_تھلے((size_t) r->شروع, HEX_L2_سطر_ناپ);
            rg->سرا   = hex_align_اتے((size_t) r->سرا, HEX_L2_سطر_ناپ);
            rg->بلاک_پہلا = بلاک_acc;
            rg->n_بلاکاں = (rg->سرا - rg->شروع + HEX_L2_بلاک_ناپ - 1) / HEX_L2_بلاک_ناپ;
            بلاک_acc += rg->n_بلاکاں;
        }

        task.کل_بلاکاں      = بلاک_acc;
        task.بلاکاں_فی_thread = fastdiv(بلاک_acc + ctx->n_دھاگے - 1, &ctx->n_دھاگے_div);

        work_قطار_لگاتار(ctx->work_قطار, l2flush_multi_کامہ, &task, ctx->n_دھاگے);
    } وإلا {
        بنية htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        لكل (uint32_t i = 0; i < n_ranges; i++) {
            ثابت بنية htp_dirty_range * r = &ranges[i];
            hex_l2flush((فراغ *) (uintptr_t) r->شروع, r->سرا - r->شروع);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

فراغ htp_dcache_flush_dirty_ranges(بنية htp_context * ctx) {
    بنية htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t کل_dirty = 0;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ثابت بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروع) {
            استمر;
        }
        ranges[n_ranges++] = *r;
        کل_dirty += r->سرا - r->شروع;
    }

    إذا (کل_dirty == 0) {
        أرجع;
    }

    إذا (کل_dirty > HEX_L2_FLUSH_ALL_حد) {
        flush_all_dcache(ctx);
        أرجع;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, کل_dirty);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

فراغ htp_dcache_flush_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n) {
    ثابت بنية htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    بنية htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t کل_dirty = 0;

    لكل (uint32_t i = 0; i < n; i++) {
        ثابت بنية htp_tensor * t = tensors[i];
        إذا (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].شروع = t->انکڑے;
            ranges[n_dirty - 1].سرا   = t->انکڑے + t->ناپ;
            کل_dirty += t->ناپ;
        }
    }

    إذا (کل_dirty == 0) {
        أرجع;
    }

    إذا (کل_dirty > HEX_L2_FLUSH_ALL_حد) {
        flush_all_dcache(ctx);
        أرجع;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, کل_dirty);
    لكل (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
