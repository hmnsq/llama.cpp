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
    uint32_t شروعات;       // line-aligned start address
    uint32_t پڇاڙي;         // line-aligned end address
    uint32_t بلاڪ_پهريون; // global block index of this range's first block
    uint32_t n_بلاڪون;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

بنية l2flush_multi_task {
    بنية htp_thread_trace * trace;
    بنية l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  ڪل_بلاڪون;
    uint32_t                  بلاڪون_في_thread;
};

ساكن فراغ flush_all_dcache(بنية htp_context * ctx) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_بلاڪ(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

ساكن فراغ l2flush_multi_ڪم_ڪندڙ(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    بنية l2flush_multi_task * task = (بنية l2flush_multi_task *) معلومات;
    (فراغ) n;

    ثابت uint32_t gb_پهريون = i * task->بلاڪون_في_thread;
    uint32_t       gb_آخري  = gb_پهريون + task->بلاڪون_في_thread;
    إذا (gb_آخري > task->ڪل_بلاڪون) {
        gb_آخري = task->ڪل_بلاڪون;
    }
    إذا (gb_پهريون >= gb_آخري) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_L2FLUSH, gb_پهريون);

    لكل (uint32_t r = 0; r < task->n_ranges; r++) {
        ثابت بنية l2flush_range * rg = &task->ranges[r];
        ثابت uint32_t rb_پهريون = rg->بلاڪ_پهريون;
        ثابت uint32_t rb_آخري  = rg->بلاڪ_پهريون + rg->n_بلاڪون;

        ثابت uint32_t lo = gb_پهريون > rb_پهريون ? gb_پهريون : rb_پهريون;
        ثابت uint32_t hi = gb_آخري  < rb_آخري  ? gb_آخري  : rb_آخري;
        إذا (lo >= hi) {
            استمر;
        }

        ثابت uint32_t s = rg->شروعات + (lo - rb_پهريون) * HEX_L2_بلاڪ_سائيز;
        uint32_t       e = rg->شروعات + (hi - rb_پهريون) * HEX_L2_بلاڪ_سائيز;
        إذا (e > rg->پڇاڙي) {
            e = rg->پڇاڙي;
        }
        hex_l2flush((فراغ *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_پهريون);
}

ساكن فراغ ملايو_dirty_ranges(بنية htp_context * ctx) {
    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروعات) استمر;

        لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            بنية htp_dirty_range * s = &ctx->dirty_ranges[j];
            إذا (i == j || !s->شروعات || r->پڇاڙي < s->شروعات || s->پڇاڙي < r->شروعات) {
                j++;
                استمر;
            }

            r->شروعات = MIN(r->شروعات, s->شروعات);
            r->پڇاڙي   = MAX(r->پڇاڙي, s->پڇاڙي);
            s->شروعات = 0;
            s->پڇاڙي   = 0;
            j = 0;
        }
    }
}

فراغ htp_dcache_mark_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n) {
    ثابت بنية htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    لكل (uint32_t i = 0; i < n; i++) {
        ثابت بنية htp_tensor * t = tensors[i];
        إذا (!t || (t->flags & (HTP_TENSOR_وزن | HTP_TENSOR_FENCE))) {
            استمر;
        }

        uint32_t t_شروعات = t->معلومات;
        uint32_t t_پڇاڙي   = t_شروعات + t->سائيز;

        bool merged = false;
        لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            بنية htp_dirty_range * r = &ctx->dirty_ranges[j];
            إذا (!r->شروعات) استمر;

            إذا (r->شروعات <= t_پڇاڙي && t_شروعات <= r->پڇاڙي) {
                uint32_t new_شروعات = (t_شروعات < r->شروعات) ? t_شروعات : r->شروعات;
                uint32_t new_پڇاڙي   = (t_پڇاڙي > r->پڇاڙي)     ? t_پڇاڙي   : r->پڇاڙي;
                r->شروعات = new_شروعات;
                r->پڇاڙي   = new_پڇاڙي;
                merged   = true;
            }
        }

        إذا (!merged) {
            pending[n_pending++] = t;
        }
    }

    ملايو_dirty_ranges(ctx);

    إذا (n_pending == 0) {
        أرجع;
    }

    uint32_t خالي_جايون_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t سرگرم_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_سرگرم = 0;
    uint32_t n_خالي_جايون  = 0;
    لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        إذا (ctx->dirty_ranges[j].شروعات) {
            سرگرم_indices[n_سرگرم++] = j;
        } وإلا {
            خالي_جايون_indices[n_خالي_جايون++]   = j;
        }
    }

    إذا (n_pending <= n_خالي_جايون) {
        لكل (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = خالي_جايون_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->شروعات = pending[i]->معلومات;
            r->پڇاڙي   = pending[i]->معلومات + pending[i]->سائيز;
        }
        ملايو_dirty_ranges(ctx);
        أرجع;
    }

    uint32_t n_evict = n_pending - n_خالي_جايون;
    uint32_t ڪل_evict_سائيز = 0;
    لكل (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = سرگرم_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        ڪل_evict_سائيز += r->پڇاڙي - r->شروعات;
    }

    إذا (ڪل_evict_سائيز > HEX_L2_FLUSH_ALL_حد) {
        flush_all_dcache(ctx);
        لكل (uint32_t i = 0; i < n_pending; i++) {
            بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->شروعات = pending[i]->معلومات;
            r->پڇاڙي   = pending[i]->معلومات + pending[i]->سائيز;
        }
        ملايو_dirty_ranges(ctx);
        أرجع;
    }

    إذا (ڪل_evict_سائيز > HEX_L2_FLUSH_WQ_حد && ctx->n_ڌاڳا > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        بنية l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t بلاڪ_acc = 0;
        لكل (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = سرگرم_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];

            بنية l2flush_range * rg = &task.ranges[i];
            rg->شروعات = hex_align_هيٺ_ڏانهن((size_t) r->شروعات, HEX_L2_سٽ_سائيز);
            rg->پڇاڙي   = hex_align_مٿي((size_t) r->پڇاڙي, HEX_L2_سٽ_سائيز);
            rg->بلاڪ_پهريون = بلاڪ_acc;
            rg->n_بلاڪون = (rg->پڇاڙي - rg->شروعات + HEX_L2_بلاڪ_سائيز - 1) / HEX_L2_بلاڪ_سائيز;
            بلاڪ_acc += rg->n_بلاڪون;
        }

        task.ڪل_بلاڪون      = بلاڪ_acc;
        task.بلاڪون_في_thread = fastdiv(بلاڪ_acc + ctx->n_ڌاڳا - 1, &ctx->n_ڌاڳا_div);

        work_قطار_لاڳيتو(ctx->work_قطار, l2flush_multi_ڪم_ڪندڙ, &task, ctx->n_ڌاڳا);
    } وإلا {
        بنية htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        لكل (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = سرگرم_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t سائيز = r->پڇاڙي - r->شروعات;
            hex_l2flush((فراغ *) (uintptr_t) r->شروعات, سائيز);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    لكل (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = سرگرم_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->شروعات = pending[i]->معلومات;
        r->پڇاڙي   = pending[i]->معلومات + pending[i]->سائيز;
    }

    لكل (uint32_t i = 0; i < n_خالي_جايون; i++) {
        uint32_t idx = خالي_جايون_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->شروعات = pending[n_evict + i]->معلومات;
        r->پڇاڙي   = pending[n_evict + i]->معلومات + pending[n_evict + i]->سائيز;
    }

    ملايو_dirty_ranges(ctx);
}

ساكن فراغ make_tensor_clean(بنية htp_context * ctx, ثابت بنية htp_tensor * t) {
    إذا (!t || (t->flags & (HTP_TENSOR_وزن | HTP_TENSOR_FENCE))) {
        أرجع;
    }

    uint32_t t_شروعات = t->معلومات;
    uint32_t t_پڇاڙي   = t_شروعات + t->سائيز;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروعات) استمر;

        إذا (r->شروعات < t_پڇاڙي && t_شروعات < r->پڇاڙي) {
            إذا (t_شروعات <= r->شروعات && r->پڇاڙي <= t_پڇاڙي) {
                r->شروعات = 0;
                r->پڇاڙي   = 0;
            } وإلا إذا (t_شروعات <= r->شروعات) {
                r->شروعات = t_پڇاڙي;
            } وإلا إذا (r->پڇاڙي <= t_پڇاڙي) {
                r->پڇاڙي = t_شروعات;
            }
        }
    }
}

ساكن مضمن bool is_tensor_dirty(بنية htp_context * ctx, ثابت بنية htp_tensor * t) {
    إذا (!t || (t->flags & (HTP_TENSOR_وزن | HTP_TENSOR_FENCE))) {
        أرجع false;
    }

    uint32_t t_شروعات = t->معلومات;
    uint32_t t_پڇاڙي   = t_شروعات + t->سائيز;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروعات) استمر;

        إذا (r->شروعات < t_پڇاڙي && t_شروعات < r->پڇاڙي) {
            أرجع true;
        }
    }
    أرجع false;
}

ساكن فراغ flush_dirty_ranges(بنية htp_context * ctx, ثابت بنية htp_dirty_range * ranges, uint32_t n_ranges, uint64_t ڪل_dirty) {
    إذا (ڪل_dirty >= HEX_L2_FLUSH_WQ_حد && ctx->n_ڌاڳا > 1) {
        بنية l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t بلاڪ_acc = 0;
        لكل (uint32_t i = 0; i < n_ranges; i++) {
            ثابت بنية htp_dirty_range * r = &ranges[i];
            بنية l2flush_range * rg = &task.ranges[i];
            rg->شروعات = hex_align_هيٺ_ڏانهن((size_t) r->شروعات, HEX_L2_سٽ_سائيز);
            rg->پڇاڙي   = hex_align_مٿي((size_t) r->پڇاڙي, HEX_L2_سٽ_سائيز);
            rg->بلاڪ_پهريون = بلاڪ_acc;
            rg->n_بلاڪون = (rg->پڇاڙي - rg->شروعات + HEX_L2_بلاڪ_سائيز - 1) / HEX_L2_بلاڪ_سائيز;
            بلاڪ_acc += rg->n_بلاڪون;
        }

        task.ڪل_بلاڪون      = بلاڪ_acc;
        task.بلاڪون_في_thread = fastdiv(بلاڪ_acc + ctx->n_ڌاڳا - 1, &ctx->n_ڌاڳا_div);

        work_قطار_لاڳيتو(ctx->work_قطار, l2flush_multi_ڪم_ڪندڙ, &task, ctx->n_ڌاڳا);
    } وإلا {
        بنية htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        لكل (uint32_t i = 0; i < n_ranges; i++) {
            ثابت بنية htp_dirty_range * r = &ranges[i];
            hex_l2flush((فراغ *) (uintptr_t) r->شروعات, r->پڇاڙي - r->شروعات);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

فراغ htp_dcache_flush_dirty_ranges(بنية htp_context * ctx) {
    بنية htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t ڪل_dirty = 0;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ثابت بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروعات) {
            استمر;
        }
        ranges[n_ranges++] = *r;
        ڪل_dirty += r->پڇاڙي - r->شروعات;
    }

    إذا (ڪل_dirty == 0) {
        أرجع;
    }

    إذا (ڪل_dirty > HEX_L2_FLUSH_ALL_حد) {
        flush_all_dcache(ctx);
        أرجع;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, ڪل_dirty);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

فراغ htp_dcache_flush_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n) {
    ثابت بنية htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    بنية htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t ڪل_dirty = 0;

    لكل (uint32_t i = 0; i < n; i++) {
        ثابت بنية htp_tensor * t = tensors[i];
        إذا (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].شروعات = t->معلومات;
            ranges[n_dirty - 1].پڇاڙي   = t->معلومات + t->سائيز;
            ڪل_dirty += t->سائيز;
        }
    }

    إذا (ڪل_dirty == 0) {
        أرجع;
    }

    إذا (ڪل_dirty > HEX_L2_FLUSH_ALL_حد) {
        flush_all_dcache(ctx);
        أرجع;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, ڪل_dirty);
    لكل (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
