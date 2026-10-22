#अन्तर्भाव "htp-dcache.h"

#अन्तर्भाव <qurt.h>
#अन्तर्भाव <qurt_memory.h>
#अन्तर्भाव <HAP_farf.h>

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "work-queue.h"

संरचना l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t अन्तः_2;         // line-aligned end address
    uint32_t खण्डम्_first; // global block index of this range's first block
    uint32_t n_खण्डानि;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

संरचना l2flush_multi_task {
    संरचना htp_thread_trace * trace;
    संरचना l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  योग_खण्डानि;
    uint32_t                  खण्डानि_प्रत्येकम्_thread;
};

स्थिर शून्य flush_all_dcache(संरचना htp_context * ctx) {
    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_खण्डम्(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    स्मृतिं_पूरय(ctx->dirty_ranges, 0, परिमाण(ctx->dirty_ranges));
}

स्थिर शून्य l2flush_multi_कार्यकर्ता(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    संरचना l2flush_multi_task * task = (संरचना l2flush_multi_task *) दत्त;
    (शून्य) n;

    ध्रुव uint32_t gb_first = i * task->खण्डानि_प्रत्येकम्_thread;
    uint32_t       gb_चरमः  = gb_first + task->खण्डानि_प्रत्येकम्_thread;
    यदि (gb_चरमः > task->योग_खण्डानि) {
        gb_चरमः = task->योग_खण्डानि;
    }
    यदि (gb_first >= gb_चरमः) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    प्रति (uint32_t r = 0; r < task->n_ranges; r++) {
        ध्रुव संरचना l2flush_range * rg = &task->ranges[r];
        ध्रुव uint32_t rb_first = rg->खण्डम्_first;
        ध्रुव uint32_t rb_चरमः  = rg->खण्डम्_first + rg->n_खण्डानि;

        ध्रुव uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        ध्रुव uint32_t hi = gb_चरमः  < rb_चरमः  ? gb_चरमः  : rb_चरमः;
        यदि (lo >= hi) {
            अनुवर्त;
        }

        ध्रुव uint32_t s = rg->start + (lo - rb_first) * HEX_L2_खण्डम्_आकार;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_खण्डम्_आकार;
        यदि (e > rg->अन्तः_2) {
            e = rg->अन्तः_2;
        }
        hex_l2flush((शून्य *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

स्थिर शून्य merge_dirty_ranges(संरचना htp_context * ctx) {
    प्रति (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[i];
        यदि (!r->start) अनुवर्त;

        प्रति (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            संरचना htp_dirty_range * s = &ctx->dirty_ranges[j];
            यदि (i == j || !s->start || r->अन्तः_2 < s->start || s->अन्तः_2 < r->start) {
                j++;
                अनुवर्त;
            }

            r->start = MIN(r->start, s->start);
            r->अन्तः_2   = MAX(r->अन्तः_2, s->अन्तः_2);
            s->start = 0;
            s->अन्तः_2   = 0;
            j = 0;
        }
    }
}

शून्य htp_dcache_mark_dirty_tensors(संरचना htp_context * ctx, ध्रुव संरचना htp_tensor * ध्रुव * tensors, uint32_t n) {
    ध्रुव संरचना htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    प्रति (uint32_t i = 0; i < n; i++) {
        ध्रुव संरचना htp_tensor * t = tensors[i];
        यदि (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            अनुवर्त;
        }

        uint32_t t_start = t->दत्त;
        uint32_t t_अन्तः   = t_start + t->आकार;

        bool merged = false;
        प्रति (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            संरचना htp_dirty_range * r = &ctx->dirty_ranges[j];
            यदि (!r->start) अनुवर्त;

            यदि (r->start <= t_अन्तः && t_start <= r->अन्तः_2) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_अन्तः   = (t_अन्तः > r->अन्तः_2)     ? t_अन्तः   : r->अन्तः_2;
                r->start = new_start;
                r->अन्तः_2   = new_अन्तः;
                merged   = true;
            }
        }

        यदि (!merged) {
            pending[n_pending++] = t;
        }
    }

    merge_dirty_ranges(ctx);

    यदि (n_pending == 0) {
        प्रत्यागम;
    }

    uint32_t रिक्तानि_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t सक्रिय_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_सक्रिय = 0;
    uint32_t n_रिक्तानि  = 0;
    प्रति (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        यदि (ctx->dirty_ranges[j].start) {
            सक्रिय_indices[n_सक्रिय++] = j;
        } अन्यथा {
            रिक्तानि_indices[n_रिक्तानि++]   = j;
        }
    }

    यदि (n_pending <= n_रिक्तानि) {
        प्रति (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = रिक्तानि_indices[i];
            संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->दत्त;
            r->अन्तः_2   = pending[i]->दत्त + pending[i]->आकार;
        }
        merge_dirty_ranges(ctx);
        प्रत्यागम;
    }

    uint32_t n_evict = n_pending - n_रिक्तानि;
    uint32_t योग_evict_आकार = 0;
    प्रति (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = सक्रिय_indices[i];
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];
        योग_evict_आकार += r->अन्तः_2 - r->start;
    }

    यदि (योग_evict_आकार > HEX_L2_FLUSH_ALL_देहली) {
        flush_all_dcache(ctx);
        प्रति (uint32_t i = 0; i < n_pending; i++) {
            संरचना htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->दत्त;
            r->अन्तः_2   = pending[i]->दत्त + pending[i]->आकार;
        }
        merge_dirty_ranges(ctx);
        प्रत्यागम;
    }

    यदि (योग_evict_आकार > HEX_L2_FLUSH_WQ_देहली && ctx->n_तन्तवः > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        संरचना l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t खण्डम्_acc = 0;
        प्रति (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = सक्रिय_indices[i];
            संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];

            संरचना l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_अधस्तात्((size_t) r->start, HEX_L2_पङ्क्तिः_आकार);
            rg->अन्तः_2   = hex_align_ऊर्ध्वम्((size_t) r->अन्तः_2, HEX_L2_पङ्क्तिः_आकार);
            rg->खण्डम्_first = खण्डम्_acc;
            rg->n_खण्डानि = (rg->अन्तः_2 - rg->start + HEX_L2_खण्डम्_आकार - 1) / HEX_L2_खण्डम्_आकार;
            खण्डम्_acc += rg->n_खण्डानि;
        }

        task.योग_खण्डानि      = खण्डम्_acc;
        task.खण्डानि_प्रत्येकम्_thread = fastdiv(खण्डम्_acc + ctx->n_तन्तवः - 1, &ctx->n_तन्तवः_div);

        work_queue_सततम्(ctx->work_queue, l2flush_multi_कार्यकर्ता, &task, ctx->n_तन्तवः);
    } अन्यथा {
        संरचना htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        प्रति (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = सक्रिय_indices[i];
            संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t आकार = r->अन्तः_2 - r->start;
            hex_l2flush((शून्य *) (uintptr_t) r->start, आकार);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    प्रति (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = सक्रिय_indices[i];
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->दत्त;
        r->अन्तः_2   = pending[i]->दत्त + pending[i]->आकार;
    }

    प्रति (uint32_t i = 0; i < n_रिक्तानि; i++) {
        uint32_t idx = रिक्तानि_indices[i];
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->दत्त;
        r->अन्तः_2   = pending[n_evict + i]->दत्त + pending[n_evict + i]->आकार;
    }

    merge_dirty_ranges(ctx);
}

स्थिर शून्य make_tensor_clean(संरचना htp_context * ctx, ध्रुव संरचना htp_tensor * t) {
    यदि (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        प्रत्यागम;
    }

    uint32_t t_start = t->दत्त;
    uint32_t t_अन्तः   = t_start + t->आकार;

    प्रति (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[i];
        यदि (!r->start) अनुवर्त;

        यदि (r->start < t_अन्तः && t_start < r->अन्तः_2) {
            यदि (t_start <= r->start && r->अन्तः_2 <= t_अन्तः) {
                r->start = 0;
                r->अन्तः_2   = 0;
            } अन्यथा यदि (t_start <= r->start) {
                r->start = t_अन्तः;
            } अन्यथा यदि (r->अन्तः_2 <= t_अन्तः) {
                r->अन्तः_2 = t_start;
            }
        }
    }
}

स्थिर अन्तःस्थ bool is_tensor_dirty(संरचना htp_context * ctx, ध्रुव संरचना htp_tensor * t) {
    यदि (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        प्रत्यागम false;
    }

    uint32_t t_start = t->दत्त;
    uint32_t t_अन्तः   = t_start + t->आकार;

    प्रति (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[i];
        यदि (!r->start) अनुवर्त;

        यदि (r->start < t_अन्तः && t_start < r->अन्तः_2) {
            प्रत्यागम true;
        }
    }
    प्रत्यागम false;
}

स्थिर शून्य flush_dirty_ranges(संरचना htp_context * ctx, ध्रुव संरचना htp_dirty_range * ranges, uint32_t n_ranges, uint64_t योग_dirty) {
    यदि (योग_dirty >= HEX_L2_FLUSH_WQ_देहली && ctx->n_तन्तवः > 1) {
        संरचना l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t खण्डम्_acc = 0;
        प्रति (uint32_t i = 0; i < n_ranges; i++) {
            ध्रुव संरचना htp_dirty_range * r = &ranges[i];
            संरचना l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_अधस्तात्((size_t) r->start, HEX_L2_पङ्क्तिः_आकार);
            rg->अन्तः_2   = hex_align_ऊर्ध्वम्((size_t) r->अन्तः_2, HEX_L2_पङ्क्तिः_आकार);
            rg->खण्डम्_first = खण्डम्_acc;
            rg->n_खण्डानि = (rg->अन्तः_2 - rg->start + HEX_L2_खण्डम्_आकार - 1) / HEX_L2_खण्डम्_आकार;
            खण्डम्_acc += rg->n_खण्डानि;
        }

        task.योग_खण्डानि      = खण्डम्_acc;
        task.खण्डानि_प्रत्येकम्_thread = fastdiv(खण्डम्_acc + ctx->n_तन्तवः - 1, &ctx->n_तन्तवः_div);

        work_queue_सततम्(ctx->work_queue, l2flush_multi_कार्यकर्ता, &task, ctx->n_तन्तवः);
    } अन्यथा {
        संरचना htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        प्रति (uint32_t i = 0; i < n_ranges; i++) {
            ध्रुव संरचना htp_dirty_range * r = &ranges[i];
            hex_l2flush((शून्य *) (uintptr_t) r->start, r->अन्तः_2 - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

शून्य htp_dcache_flush_dirty_ranges(संरचना htp_context * ctx) {
    संरचना htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t योग_dirty = 0;

    प्रति (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ध्रुव संरचना htp_dirty_range * r = &ctx->dirty_ranges[i];
        यदि (!r->start) {
            अनुवर्त;
        }
        ranges[n_ranges++] = *r;
        योग_dirty += r->अन्तः_2 - r->start;
    }

    यदि (योग_dirty == 0) {
        प्रत्यागम;
    }

    यदि (योग_dirty > HEX_L2_FLUSH_ALL_देहली) {
        flush_all_dcache(ctx);
        प्रत्यागम;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, योग_dirty);
    स्मृतिं_पूरय(ctx->dirty_ranges, 0, परिमाण(ctx->dirty_ranges));
}

शून्य htp_dcache_flush_dirty_tensors(संरचना htp_context * ctx, ध्रुव संरचना htp_tensor * ध्रुव * tensors, uint32_t n) {
    ध्रुव संरचना htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    संरचना htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t योग_dirty = 0;

    प्रति (uint32_t i = 0; i < n; i++) {
        ध्रुव संरचना htp_tensor * t = tensors[i];
        यदि (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->दत्त;
            ranges[n_dirty - 1].अन्तः_2   = t->दत्त + t->आकार;
            योग_dirty += t->आकार;
        }
    }

    यदि (योग_dirty == 0) {
        प्रत्यागम;
    }

    यदि (योग_dirty > HEX_L2_FLUSH_ALL_देहली) {
        flush_all_dcache(ctx);
        प्रत्यागम;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, योग_dirty);
    प्रति (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
