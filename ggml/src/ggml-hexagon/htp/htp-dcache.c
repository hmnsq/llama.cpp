#شامل "htp-dcache.h"

#شامل <qurt.h>
#شامل <qurt_memory.h>
#شامل <HAP_farf.h>

#شامل "hex-common.h"
#شامل "hex-utils.h"
#شامل "hex-fastdiv.h"
#شامل "hex-profile.h"
#شامل "htp-ctx.h"
#شامل "work-queue.h"

ساخت l2flush_range {
    uint32_t آغاز;       // line-aligned start address
    uint32_t انتہا;         // line-aligned end address
    uint32_t بلاک_پہلا; // global block index of this range's first block
    uint32_t n_بلاکیں;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

ساخت l2flush_multi_task {
    ساخت htp_thread_trace * trace;
    ساخت l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  کل_بلاکیں;
    uint32_t                  بلاکیں_فی_thread;
};

ساکن خالی flush_all_dcache(ساخت htp_context * ctx) {
    ساخت htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_بلاک(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

ساکن خالی l2flush_multi_کارکن(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    ساخت l2flush_multi_task * task = (ساخت l2flush_multi_task *) ڈیٹا;
    (خالی) n;

    ثابت uint32_t gb_پہلا = i * task->بلاکیں_فی_thread;
    uint32_t       gb_آخری  = gb_پہلا + task->بلاکیں_فی_thread;
    اگر (gb_آخری > task->کل_بلاکیں) {
        gb_آخری = task->کل_بلاکیں;
    }
    اگر (gb_پہلا >= gb_آخری) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_L2FLUSH, gb_پہلا);

    برائے (uint32_t r = 0; r < task->n_ranges; r++) {
        ثابت ساخت l2flush_range * rg = &task->ranges[r];
        ثابت uint32_t rb_پہلا = rg->بلاک_پہلا;
        ثابت uint32_t rb_آخری  = rg->بلاک_پہلا + rg->n_بلاکیں;

        ثابت uint32_t lo = gb_پہلا > rb_پہلا ? gb_پہلا : rb_پہلا;
        ثابت uint32_t hi = gb_آخری  < rb_آخری  ? gb_آخری  : rb_آخری;
        اگر (lo >= hi) {
            جاری;
        }

        ثابت uint32_t s = rg->آغاز + (lo - rb_پہلا) * HEX_L2_بلاک_سائز;
        uint32_t       e = rg->آغاز + (hi - rb_پہلا) * HEX_L2_بلاک_سائز;
        اگر (e > rg->انتہا) {
            e = rg->انتہا;
        }
        hex_l2flush((خالی *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_پہلا);
}

ساکن خالی ملاؤ_dirty_ranges(ساخت htp_context * ctx) {
    برائے (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ساخت htp_dirty_range * r = &ctx->dirty_ranges[i];
        اگر (!r->آغاز) جاری;

        برائے (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            ساخت htp_dirty_range * s = &ctx->dirty_ranges[j];
            اگر (i == j || !s->آغاز || r->انتہا < s->آغاز || s->انتہا < r->آغاز) {
                j++;
                جاری;
            }

            r->آغاز = MIN(r->آغاز, s->آغاز);
            r->انتہا   = MAX(r->انتہا, s->انتہا);
            s->آغاز = 0;
            s->انتہا   = 0;
            j = 0;
        }
    }
}

خالی htp_dcache_mark_dirty_tensors(ساخت htp_context * ctx, ثابت ساخت htp_tensor * ثابت * tensors, uint32_t n) {
    ثابت ساخت htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    برائے (uint32_t i = 0; i < n; i++) {
        ثابت ساخت htp_tensor * t = tensors[i];
        اگر (!t || (t->flags & (HTP_TENSOR_وزن | HTP_TENSOR_FENCE))) {
            جاری;
        }

        uint32_t t_آغاز = t->ڈیٹا;
        uint32_t t_انتہا   = t_آغاز + t->سائز;

        bool merged = false;
        برائے (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            ساخت htp_dirty_range * r = &ctx->dirty_ranges[j];
            اگر (!r->آغاز) جاری;

            اگر (r->آغاز <= t_انتہا && t_آغاز <= r->انتہا) {
                uint32_t new_آغاز = (t_آغاز < r->آغاز) ? t_آغاز : r->آغاز;
                uint32_t new_انتہا   = (t_انتہا > r->انتہا)     ? t_انتہا   : r->انتہا;
                r->آغاز = new_آغاز;
                r->انتہا   = new_انتہا;
                merged   = true;
            }
        }

        اگر (!merged) {
            pending[n_pending++] = t;
        }
    }

    ملاؤ_dirty_ranges(ctx);

    اگر (n_pending == 0) {
        واپس;
    }

    uint32_t خالی_خانے_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t فعال_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_فعال = 0;
    uint32_t n_خالی_خانے  = 0;
    برائے (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        اگر (ctx->dirty_ranges[j].آغاز) {
            فعال_indices[n_فعال++] = j;
        } ورنہ {
            خالی_خانے_indices[n_خالی_خانے++]   = j;
        }
    }

    اگر (n_pending <= n_خالی_خانے) {
        برائے (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = خالی_خانے_indices[i];
            ساخت htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->آغاز = pending[i]->ڈیٹا;
            r->انتہا   = pending[i]->ڈیٹا + pending[i]->سائز;
        }
        ملاؤ_dirty_ranges(ctx);
        واپس;
    }

    uint32_t n_evict = n_pending - n_خالی_خانے;
    uint32_t کل_evict_سائز = 0;
    برائے (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = فعال_indices[i];
        ساخت htp_dirty_range * r = &ctx->dirty_ranges[idx];
        کل_evict_سائز += r->انتہا - r->آغاز;
    }

    اگر (کل_evict_سائز > HEX_L2_FLUSH_ALL_حد) {
        flush_all_dcache(ctx);
        برائے (uint32_t i = 0; i < n_pending; i++) {
            ساخت htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->آغاز = pending[i]->ڈیٹا;
            r->انتہا   = pending[i]->ڈیٹا + pending[i]->سائز;
        }
        ملاؤ_dirty_ranges(ctx);
        واپس;
    }

    اگر (کل_evict_سائز > HEX_L2_FLUSH_WQ_حد && ctx->n_دھاگے > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        ساخت l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t بلاک_acc = 0;
        برائے (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = فعال_indices[i];
            ساخت htp_dirty_range * r = &ctx->dirty_ranges[idx];

            ساخت l2flush_range * rg = &task.ranges[i];
            rg->آغاز = hex_align_نیچے_کو((size_t) r->آغاز, HEX_L2_سطر_سائز);
            rg->انتہا   = hex_align_اوپر((size_t) r->انتہا, HEX_L2_سطر_سائز);
            rg->بلاک_پہلا = بلاک_acc;
            rg->n_بلاکیں = (rg->انتہا - rg->آغاز + HEX_L2_بلاک_سائز - 1) / HEX_L2_بلاک_سائز;
            بلاک_acc += rg->n_بلاکیں;
        }

        task.کل_بلاکیں      = بلاک_acc;
        task.بلاکیں_فی_thread = fastdiv(بلاک_acc + ctx->n_دھاگے - 1, &ctx->n_دھاگے_div);

        work_قطار_تسلسل(ctx->work_قطار, l2flush_multi_کارکن, &task, ctx->n_دھاگے);
    } ورنہ {
        ساخت htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        برائے (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = فعال_indices[i];
            ساخت htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t سائز = r->انتہا - r->آغاز;
            hex_l2flush((خالی *) (uintptr_t) r->آغاز, سائز);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    برائے (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = فعال_indices[i];
        ساخت htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->آغاز = pending[i]->ڈیٹا;
        r->انتہا   = pending[i]->ڈیٹا + pending[i]->سائز;
    }

    برائے (uint32_t i = 0; i < n_خالی_خانے; i++) {
        uint32_t idx = خالی_خانے_indices[i];
        ساخت htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->آغاز = pending[n_evict + i]->ڈیٹا;
        r->انتہا   = pending[n_evict + i]->ڈیٹا + pending[n_evict + i]->سائز;
    }

    ملاؤ_dirty_ranges(ctx);
}

ساکن خالی make_tensor_clean(ساخت htp_context * ctx, ثابت ساخت htp_tensor * t) {
    اگر (!t || (t->flags & (HTP_TENSOR_وزن | HTP_TENSOR_FENCE))) {
        واپس;
    }

    uint32_t t_آغاز = t->ڈیٹا;
    uint32_t t_انتہا   = t_آغاز + t->سائز;

    برائے (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ساخت htp_dirty_range * r = &ctx->dirty_ranges[i];
        اگر (!r->آغاز) جاری;

        اگر (r->آغاز < t_انتہا && t_آغاز < r->انتہا) {
            اگر (t_آغاز <= r->آغاز && r->انتہا <= t_انتہا) {
                r->آغاز = 0;
                r->انتہا   = 0;
            } ورنہ اگر (t_آغاز <= r->آغاز) {
                r->آغاز = t_انتہا;
            } ورنہ اگر (r->انتہا <= t_انتہا) {
                r->انتہا = t_آغاز;
            }
        }
    }
}

ساکن اندرونی bool is_tensor_dirty(ساخت htp_context * ctx, ثابت ساخت htp_tensor * t) {
    اگر (!t || (t->flags & (HTP_TENSOR_وزن | HTP_TENSOR_FENCE))) {
        واپس false;
    }

    uint32_t t_آغاز = t->ڈیٹا;
    uint32_t t_انتہا   = t_آغاز + t->سائز;

    برائے (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ساخت htp_dirty_range * r = &ctx->dirty_ranges[i];
        اگر (!r->آغاز) جاری;

        اگر (r->آغاز < t_انتہا && t_آغاز < r->انتہا) {
            واپس true;
        }
    }
    واپس false;
}

ساکن خالی flush_dirty_ranges(ساخت htp_context * ctx, ثابت ساخت htp_dirty_range * ranges, uint32_t n_ranges, uint64_t کل_dirty) {
    اگر (کل_dirty >= HEX_L2_FLUSH_WQ_حد && ctx->n_دھاگے > 1) {
        ساخت l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t بلاک_acc = 0;
        برائے (uint32_t i = 0; i < n_ranges; i++) {
            ثابت ساخت htp_dirty_range * r = &ranges[i];
            ساخت l2flush_range * rg = &task.ranges[i];
            rg->آغاز = hex_align_نیچے_کو((size_t) r->آغاز, HEX_L2_سطر_سائز);
            rg->انتہا   = hex_align_اوپر((size_t) r->انتہا, HEX_L2_سطر_سائز);
            rg->بلاک_پہلا = بلاک_acc;
            rg->n_بلاکیں = (rg->انتہا - rg->آغاز + HEX_L2_بلاک_سائز - 1) / HEX_L2_بلاک_سائز;
            بلاک_acc += rg->n_بلاکیں;
        }

        task.کل_بلاکیں      = بلاک_acc;
        task.بلاکیں_فی_thread = fastdiv(بلاک_acc + ctx->n_دھاگے - 1, &ctx->n_دھاگے_div);

        work_قطار_تسلسل(ctx->work_قطار, l2flush_multi_کارکن, &task, ctx->n_دھاگے);
    } ورنہ {
        ساخت htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        برائے (uint32_t i = 0; i < n_ranges; i++) {
            ثابت ساخت htp_dirty_range * r = &ranges[i];
            hex_l2flush((خالی *) (uintptr_t) r->آغاز, r->انتہا - r->آغاز);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

خالی htp_dcache_flush_dirty_ranges(ساخت htp_context * ctx) {
    ساخت htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t کل_dirty = 0;

    برائے (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ثابت ساخت htp_dirty_range * r = &ctx->dirty_ranges[i];
        اگر (!r->آغاز) {
            جاری;
        }
        ranges[n_ranges++] = *r;
        کل_dirty += r->انتہا - r->آغاز;
    }

    اگر (کل_dirty == 0) {
        واپس;
    }

    اگر (کل_dirty > HEX_L2_FLUSH_ALL_حد) {
        flush_all_dcache(ctx);
        واپس;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, کل_dirty);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

خالی htp_dcache_flush_dirty_tensors(ساخت htp_context * ctx, ثابت ساخت htp_tensor * ثابت * tensors, uint32_t n) {
    ثابت ساخت htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    ساخت htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t کل_dirty = 0;

    برائے (uint32_t i = 0; i < n; i++) {
        ثابت ساخت htp_tensor * t = tensors[i];
        اگر (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].آغاز = t->ڈیٹا;
            ranges[n_dirty - 1].انتہا   = t->ڈیٹا + t->سائز;
            کل_dirty += t->سائز;
        }
    }

    اگر (کل_dirty == 0) {
        واپس;
    }

    اگر (کل_dirty > HEX_L2_FLUSH_ALL_حد) {
        flush_all_dcache(ctx);
        واپس;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, کل_dirty);
    برائے (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
