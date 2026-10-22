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
    uint32_t انتها;         // line-aligned end address
    uint32_t بلوک_اولی; // global block index of this range's first block
    uint32_t n_بلوکها;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

بنية l2flush_multi_task {
    بنية htp_thread_trace * trace;
    بنية l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  مجموع_بلوکها;
    uint32_t                  بلوکها_در_هر_thread;
};

ساكن فراغ flush_all_dcache(بنية htp_context * ctx) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_بلوک(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

ساكن فراغ l2flush_multi_کارگر(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    بنية l2flush_multi_task * task = (بنية l2flush_multi_task *) داده;
    (فراغ) n;

    ثابت uint32_t gb_اولی = i * task->بلوکها_در_هر_thread;
    uint32_t       gb_آخرین  = gb_اولی + task->بلوکها_در_هر_thread;
    إذا (gb_آخرین > task->مجموع_بلوکها) {
        gb_آخرین = task->مجموع_بلوکها;
    }
    إذا (gb_اولی >= gb_آخرین) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_L2FLUSH, gb_اولی);

    لكل (uint32_t r = 0; r < task->n_ranges; r++) {
        ثابت بنية l2flush_range * rg = &task->ranges[r];
        ثابت uint32_t rb_اولی = rg->بلوک_اولی;
        ثابت uint32_t rb_آخرین  = rg->بلوک_اولی + rg->n_بلوکها;

        ثابت uint32_t lo = gb_اولی > rb_اولی ? gb_اولی : rb_اولی;
        ثابت uint32_t hi = gb_آخرین  < rb_آخرین  ? gb_آخرین  : rb_آخرین;
        إذا (lo >= hi) {
            استمر;
        }

        ثابت uint32_t s = rg->شروع + (lo - rb_اولی) * HEX_L2_بلوک_اندازه;
        uint32_t       e = rg->شروع + (hi - rb_اولی) * HEX_L2_بلوک_اندازه;
        إذا (e > rg->انتها) {
            e = rg->انتها;
        }
        hex_l2flush((فراغ *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_اولی);
}

ساكن فراغ ادغام_dirty_ranges(بنية htp_context * ctx) {
    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروع) استمر;

        لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            بنية htp_dirty_range * s = &ctx->dirty_ranges[j];
            إذا (i == j || !s->شروع || r->انتها < s->شروع || s->انتها < r->شروع) {
                j++;
                استمر;
            }

            r->شروع = MIN(r->شروع, s->شروع);
            r->انتها   = MAX(r->انتها, s->انتها);
            s->شروع = 0;
            s->انتها   = 0;
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

        uint32_t t_شروع = t->داده;
        uint32_t t_انتها   = t_شروع + t->اندازه;

        bool merged = false;
        لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            بنية htp_dirty_range * r = &ctx->dirty_ranges[j];
            إذا (!r->شروع) استمر;

            إذا (r->شروع <= t_انتها && t_شروع <= r->انتها) {
                uint32_t new_شروع = (t_شروع < r->شروع) ? t_شروع : r->شروع;
                uint32_t new_انتها   = (t_انتها > r->انتها)     ? t_انتها   : r->انتها;
                r->شروع = new_شروع;
                r->انتها   = new_انتها;
                merged   = true;
            }
        }

        إذا (!merged) {
            pending[n_pending++] = t;
        }
    }

    ادغام_dirty_ranges(ctx);

    إذا (n_pending == 0) {
        أرجع;
    }

    uint32_t خالی_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t فعال_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_فعال = 0;
    uint32_t n_خالی  = 0;
    لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        إذا (ctx->dirty_ranges[j].شروع) {
            فعال_indices[n_فعال++] = j;
        } وإلا {
            خالی_indices[n_خالی++]   = j;
        }
    }

    إذا (n_pending <= n_خالی) {
        لكل (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = خالی_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->شروع = pending[i]->داده;
            r->انتها   = pending[i]->داده + pending[i]->اندازه;
        }
        ادغام_dirty_ranges(ctx);
        أرجع;
    }

    uint32_t n_evict = n_pending - n_خالی;
    uint32_t مجموع_evict_اندازه = 0;
    لكل (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = فعال_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        مجموع_evict_اندازه += r->انتها - r->شروع;
    }

    إذا (مجموع_evict_اندازه > HEX_L2_FLUSH_ALL_آستانه) {
        flush_all_dcache(ctx);
        لكل (uint32_t i = 0; i < n_pending; i++) {
            بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->شروع = pending[i]->داده;
            r->انتها   = pending[i]->داده + pending[i]->اندازه;
        }
        ادغام_dirty_ranges(ctx);
        أرجع;
    }

    إذا (مجموع_evict_اندازه > HEX_L2_FLUSH_WQ_آستانه && ctx->n_نخها > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        بنية l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t بلوک_acc = 0;
        لكل (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = فعال_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];

            بنية l2flush_range * rg = &task.ranges[i];
            rg->شروع = hex_align_پایین((size_t) r->شروع, HEX_L2_سطر_اندازه);
            rg->انتها   = hex_align_به_بالا((size_t) r->انتها, HEX_L2_سطر_اندازه);
            rg->بلوک_اولی = بلوک_acc;
            rg->n_بلوکها = (rg->انتها - rg->شروع + HEX_L2_بلوک_اندازه - 1) / HEX_L2_بلوک_اندازه;
            بلوک_acc += rg->n_بلوکها;
        }

        task.مجموع_بلوکها      = بلوک_acc;
        task.بلوکها_در_هر_thread = fastdiv(بلوک_acc + ctx->n_نخها - 1, &ctx->n_نخها_div);

        work_صف_پیاپی(ctx->work_صف, l2flush_multi_کارگر, &task, ctx->n_نخها);
    } وإلا {
        بنية htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        لكل (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = فعال_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t اندازه = r->انتها - r->شروع;
            hex_l2flush((فراغ *) (uintptr_t) r->شروع, اندازه);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    لكل (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = فعال_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->شروع = pending[i]->داده;
        r->انتها   = pending[i]->داده + pending[i]->اندازه;
    }

    لكل (uint32_t i = 0; i < n_خالی; i++) {
        uint32_t idx = خالی_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->شروع = pending[n_evict + i]->داده;
        r->انتها   = pending[n_evict + i]->داده + pending[n_evict + i]->اندازه;
    }

    ادغام_dirty_ranges(ctx);
}

ساكن فراغ make_tensor_clean(بنية htp_context * ctx, ثابت بنية htp_tensor * t) {
    إذا (!t || (t->flags & (HTP_TENSOR_وزن | HTP_TENSOR_FENCE))) {
        أرجع;
    }

    uint32_t t_شروع = t->داده;
    uint32_t t_انتها   = t_شروع + t->اندازه;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروع) استمر;

        إذا (r->شروع < t_انتها && t_شروع < r->انتها) {
            إذا (t_شروع <= r->شروع && r->انتها <= t_انتها) {
                r->شروع = 0;
                r->انتها   = 0;
            } وإلا إذا (t_شروع <= r->شروع) {
                r->شروع = t_انتها;
            } وإلا إذا (r->انتها <= t_انتها) {
                r->انتها = t_شروع;
            }
        }
    }
}

ساكن مضمن bool is_tensor_dirty(بنية htp_context * ctx, ثابت بنية htp_tensor * t) {
    إذا (!t || (t->flags & (HTP_TENSOR_وزن | HTP_TENSOR_FENCE))) {
        أرجع false;
    }

    uint32_t t_شروع = t->داده;
    uint32_t t_انتها   = t_شروع + t->اندازه;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروع) استمر;

        إذا (r->شروع < t_انتها && t_شروع < r->انتها) {
            أرجع true;
        }
    }
    أرجع false;
}

ساكن فراغ flush_dirty_ranges(بنية htp_context * ctx, ثابت بنية htp_dirty_range * ranges, uint32_t n_ranges, uint64_t مجموع_dirty) {
    إذا (مجموع_dirty >= HEX_L2_FLUSH_WQ_آستانه && ctx->n_نخها > 1) {
        بنية l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t بلوک_acc = 0;
        لكل (uint32_t i = 0; i < n_ranges; i++) {
            ثابت بنية htp_dirty_range * r = &ranges[i];
            بنية l2flush_range * rg = &task.ranges[i];
            rg->شروع = hex_align_پایین((size_t) r->شروع, HEX_L2_سطر_اندازه);
            rg->انتها   = hex_align_به_بالا((size_t) r->انتها, HEX_L2_سطر_اندازه);
            rg->بلوک_اولی = بلوک_acc;
            rg->n_بلوکها = (rg->انتها - rg->شروع + HEX_L2_بلوک_اندازه - 1) / HEX_L2_بلوک_اندازه;
            بلوک_acc += rg->n_بلوکها;
        }

        task.مجموع_بلوکها      = بلوک_acc;
        task.بلوکها_در_هر_thread = fastdiv(بلوک_acc + ctx->n_نخها - 1, &ctx->n_نخها_div);

        work_صف_پیاپی(ctx->work_صف, l2flush_multi_کارگر, &task, ctx->n_نخها);
    } وإلا {
        بنية htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        لكل (uint32_t i = 0; i < n_ranges; i++) {
            ثابت بنية htp_dirty_range * r = &ranges[i];
            hex_l2flush((فراغ *) (uintptr_t) r->شروع, r->انتها - r->شروع);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

فراغ htp_dcache_flush_dirty_ranges(بنية htp_context * ctx) {
    بنية htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t مجموع_dirty = 0;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ثابت بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        إذا (!r->شروع) {
            استمر;
        }
        ranges[n_ranges++] = *r;
        مجموع_dirty += r->انتها - r->شروع;
    }

    إذا (مجموع_dirty == 0) {
        أرجع;
    }

    إذا (مجموع_dirty > HEX_L2_FLUSH_ALL_آستانه) {
        flush_all_dcache(ctx);
        أرجع;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, مجموع_dirty);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

فراغ htp_dcache_flush_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n) {
    ثابت بنية htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    بنية htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t مجموع_dirty = 0;

    لكل (uint32_t i = 0; i < n; i++) {
        ثابت بنية htp_tensor * t = tensors[i];
        إذا (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].شروع = t->داده;
            ranges[n_dirty - 1].انتها   = t->داده + t->اندازه;
            مجموع_dirty += t->اندازه;
        }
    }

    إذا (مجموع_dirty == 0) {
        أرجع;
    }

    إذا (مجموع_dirty > HEX_L2_FLUSH_ALL_آستانه) {
        flush_all_dcache(ctx);
        أرجع;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, مجموع_dirty);
    لكل (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
