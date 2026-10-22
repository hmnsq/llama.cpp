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
    uint32_t پیل;       // line-aligned start address
    uint32_t پای_ټکی;         // line-aligned end address
    uint32_t بلاک_لومړی; // global block index of this range's first block
    uint32_t n_بلاکونه;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

بنية l2flush_multi_task {
    بنية htp_thread_trace * trace;
    بنية l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  ټول_بلاکونه;
    uint32_t                  بلاکونه_فی_thread;
};

ساكن فراغ flush_all_dcache(بنية htp_context * ctx) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_بلاک(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

ساكن فراغ l2flush_multi_کارګر(بینښه صحيح n, بینښه صحيح i, فراغ * معلومات) {
    بنية l2flush_multi_task * task = (بنية l2flush_multi_task *) معلومات;
    (فراغ) n;

    ثابت uint32_t gb_لومړی = i * task->بلاکونه_فی_thread;
    uint32_t       gb_وروستی  = gb_لومړی + task->بلاکونه_فی_thread;
    إذا (gb_وروستی > task->ټول_بلاکونه) {
        gb_وروستی = task->ټول_بلاکونه;
    }
    إذا (gb_لومړی >= gb_وروستی) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_L2FLUSH, gb_لومړی);

    لكل (uint32_t r = 0; r < task->n_ranges; r++) {
        ثابت بنية l2flush_range * rg = &task->ranges[r];
        ثابت uint32_t rb_لومړی = rg->بلاک_لومړی;
        ثابت uint32_t rb_وروستی  = rg->بلاک_لومړی + rg->n_بلاکونه;

        ثابت uint32_t lo = gb_لومړی > rb_لومړی ? gb_لومړی : rb_لومړی;
        ثابت uint32_t hi = gb_وروستی  < rb_وروستی  ? gb_وروستی  : rb_وروستی;
        إذا (lo >= hi) {
            استمر;
        }

        ثابت uint32_t s = rg->پیل + (lo - rb_لومړی) * HEX_L2_بلاک_اندازه;
        uint32_t       e = rg->پیل + (hi - rb_لومړی) * HEX_L2_بلاک_اندازه;
        إذا (e > rg->پای_ټکی) {
            e = rg->پای_ټکی;
        }
        hex_l2flush((فراغ *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_لومړی);
}

ساكن فراغ یوځای_dirty_ranges(بنية htp_context * ctx) {
    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->پیل) استمر;

        لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            بنية htp_dirty_range * s = &ctx->dirty_ranges[j];
            إذا (i == j || !s->پیل || r->پای_ټکی < s->پیل || s->پای_ټکی < r->پیل) {
                j++;
                استمر;
            }

            r->پیل = MIN(r->پیل, s->پیل);
            r->پای_ټکی   = MAX(r->پای_ټکی, s->پای_ټکی);
            s->پیل = 0;
            s->پای_ټکی   = 0;
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

        uint32_t t_پیل = t->معلومات;
        uint32_t t_پای_ټکی   = t_پیل + t->اندازه;

        bool merged = false;
        لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            بنية htp_dirty_range * r = &ctx->dirty_ranges[j];
            إذا (!r->پیل) استمر;

            إذا (r->پیل <= t_پای_ټکی && t_پیل <= r->پای_ټکی) {
                uint32_t new_پیل = (t_پیل < r->پیل) ? t_پیل : r->پیل;
                uint32_t new_پای_ټکی   = (t_پای_ټکی > r->پای_ټکی)     ? t_پای_ټکی   : r->پای_ټکی;
                r->پیل = new_پیل;
                r->پای_ټکی   = new_پای_ټکی;
                merged   = true;
            }
        }

        إذا (!merged) {
            pending[n_pending++] = t;
        }
    }

    یوځای_dirty_ranges(ctx);

    إذا (n_pending == 0) {
        أرجع;
    }

    uint32_t تشه_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t فعال_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_فعال = 0;
    uint32_t n_تشه  = 0;
    لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        إذا (ctx->dirty_ranges[j].پیل) {
            فعال_indices[n_فعال++] = j;
        } وإلا {
            تشه_indices[n_تشه++]   = j;
        }
    }

    إذا (n_pending <= n_تشه) {
        لكل (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = تشه_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->پیل = pending[i]->معلومات;
            r->پای_ټکی   = pending[i]->معلومات + pending[i]->اندازه;
        }
        یوځای_dirty_ranges(ctx);
        أرجع;
    }

    uint32_t n_evict = n_pending - n_تشه;
    uint32_t ټول_evict_اندازه = 0;
    لكل (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = فعال_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        ټول_evict_اندازه += r->پای_ټکی - r->پیل;
    }

    إذا (ټول_evict_اندازه > HEX_L2_FLUSH_ALL_پوله) {
        flush_all_dcache(ctx);
        لكل (uint32_t i = 0; i < n_pending; i++) {
            بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->پیل = pending[i]->معلومات;
            r->پای_ټکی   = pending[i]->معلومات + pending[i]->اندازه;
        }
        یوځای_dirty_ranges(ctx);
        أرجع;
    }

    إذا (ټول_evict_اندازه > HEX_L2_FLUSH_WQ_پوله && ctx->n_تارونه > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        بنية l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t بلاک_acc = 0;
        لكل (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = فعال_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];

            بنية l2flush_range * rg = &task.ranges[i];
            rg->پیل = hex_align_ښکته((size_t) r->پیل, HEX_L2_کرښه_اندازه);
            rg->پای_ټکی   = hex_align_پورته((size_t) r->پای_ټکی, HEX_L2_کرښه_اندازه);
            rg->بلاک_لومړی = بلاک_acc;
            rg->n_بلاکونه = (rg->پای_ټکی - rg->پیل + HEX_L2_بلاک_اندازه - 1) / HEX_L2_بلاک_اندازه;
            بلاک_acc += rg->n_بلاکونه;
        }

        task.ټول_بلاکونه      = بلاک_acc;
        task.بلاکونه_فی_thread = fastdiv(بلاک_acc + ctx->n_تارونه - 1, &ctx->n_تارونه_div);

        work_کتار_پرلپسې(ctx->work_کتار, l2flush_multi_کارګر, &task, ctx->n_تارونه);
    } وإلا {
        بنية htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        لكل (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = فعال_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t اندازه = r->پای_ټکی - r->پیل;
            hex_l2flush((فراغ *) (uintptr_t) r->پیل, اندازه);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    لكل (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = فعال_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->پیل = pending[i]->معلومات;
        r->پای_ټکی   = pending[i]->معلومات + pending[i]->اندازه;
    }

    لكل (uint32_t i = 0; i < n_تشه; i++) {
        uint32_t idx = تشه_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->پیل = pending[n_evict + i]->معلومات;
        r->پای_ټکی   = pending[n_evict + i]->معلومات + pending[n_evict + i]->اندازه;
    }

    یوځای_dirty_ranges(ctx);
}

ساكن فراغ make_tensor_clean(بنية htp_context * ctx, ثابت بنية htp_tensor * t) {
    إذا (!t || (t->flags & (HTP_TENSOR_وزن | HTP_TENSOR_FENCE))) {
        أرجع;
    }

    uint32_t t_پیل = t->معلومات;
    uint32_t t_پای_ټکی   = t_پیل + t->اندازه;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->پیل) استمر;

        إذا (r->پیل < t_پای_ټکی && t_پیل < r->پای_ټکی) {
            إذا (t_پیل <= r->پیل && r->پای_ټکی <= t_پای_ټکی) {
                r->پیل = 0;
                r->پای_ټکی   = 0;
            } وإلا إذا (t_پیل <= r->پیل) {
                r->پیل = t_پای_ټکی;
            } وإلا إذا (r->پای_ټکی <= t_پای_ټکی) {
                r->پای_ټکی = t_پیل;
            }
        }
    }
}

ساكن مضمن bool is_tensor_dirty(بنية htp_context * ctx, ثابت بنية htp_tensor * t) {
    إذا (!t || (t->flags & (HTP_TENSOR_وزن | HTP_TENSOR_FENCE))) {
        أرجع false;
    }

    uint32_t t_پیل = t->معلومات;
    uint32_t t_پای_ټکی   = t_پیل + t->اندازه;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->پیل) استمر;

        إذا (r->پیل < t_پای_ټکی && t_پیل < r->پای_ټکی) {
            أرجع true;
        }
    }
    أرجع false;
}

ساكن فراغ flush_dirty_ranges(بنية htp_context * ctx, ثابت بنية htp_dirty_range * ranges, uint32_t n_ranges, uint64_t ټول_dirty) {
    إذا (ټول_dirty >= HEX_L2_FLUSH_WQ_پوله && ctx->n_تارونه > 1) {
        بنية l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t بلاک_acc = 0;
        لكل (uint32_t i = 0; i < n_ranges; i++) {
            ثابت بنية htp_dirty_range * r = &ranges[i];
            بنية l2flush_range * rg = &task.ranges[i];
            rg->پیل = hex_align_ښکته((size_t) r->پیل, HEX_L2_کرښه_اندازه);
            rg->پای_ټکی   = hex_align_پورته((size_t) r->پای_ټکی, HEX_L2_کرښه_اندازه);
            rg->بلاک_لومړی = بلاک_acc;
            rg->n_بلاکونه = (rg->پای_ټکی - rg->پیل + HEX_L2_بلاک_اندازه - 1) / HEX_L2_بلاک_اندازه;
            بلاک_acc += rg->n_بلاکونه;
        }

        task.ټول_بلاکونه      = بلاک_acc;
        task.بلاکونه_فی_thread = fastdiv(بلاک_acc + ctx->n_تارونه - 1, &ctx->n_تارونه_div);

        work_کتار_پرلپسې(ctx->work_کتار, l2flush_multi_کارګر, &task, ctx->n_تارونه);
    } وإلا {
        بنية htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        لكل (uint32_t i = 0; i < n_ranges; i++) {
            ثابت بنية htp_dirty_range * r = &ranges[i];
            hex_l2flush((فراغ *) (uintptr_t) r->پیل, r->پای_ټکی - r->پیل);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

فراغ htp_dcache_flush_dirty_ranges(بنية htp_context * ctx) {
    بنية htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t ټول_dirty = 0;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ثابت بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->پیل) {
            استمر;
        }
        ranges[n_ranges++] = *r;
        ټول_dirty += r->پای_ټکی - r->پیل;
    }

    إذا (ټول_dirty == 0) {
        أرجع;
    }

    إذا (ټول_dirty > HEX_L2_FLUSH_ALL_پوله) {
        flush_all_dcache(ctx);
        أرجع;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, ټول_dirty);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

فراغ htp_dcache_flush_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n) {
    ثابت بنية htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    بنية htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t ټول_dirty = 0;

    لكل (uint32_t i = 0; i < n; i++) {
        ثابت بنية htp_tensor * t = tensors[i];
        إذا (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].پیل = t->معلومات;
            ranges[n_dirty - 1].پای_ټکی   = t->معلومات + t->اندازه;
            ټول_dirty += t->اندازه;
        }
    }

    إذا (ټول_dirty == 0) {
        أرجع;
    }

    إذا (ټول_dirty > HEX_L2_FLUSH_ALL_پوله) {
        flush_all_dcache(ctx);
        أرجع;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, ټول_dirty);
    لكل (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
