#అన్తర్భావ "htp-dcache.h"

#అన్తర్భావ <qurt.h>
#అన్తర్భావ <qurt_memory.h>
#అన్తర్భావ <HAP_farf.h>

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-utils.h"
#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "work-queue.h"

నిర్మాణం l2flush_range {
    uint32_t ప్రారంభం;       // line-aligned start address
    uint32_t చివర;         // line-aligned end address
    uint32_t ఖండం_మొదటి; // global block index of this range's first block
    uint32_t n_ఖండాలు;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

నిర్మాణం l2flush_multi_task {
    నిర్మాణం htp_thread_trace * trace;
    నిర్మాణం l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  మొత్తం_ఖండాలు;
    uint32_t                  ఖండాలు_చొప్పున_thread;
};

స్థిర శూన్య flush_all_dcache(నిర్మాణం htp_context * ctx) {
    నిర్మాణం htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_ఖండం(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, పరిమాణం(ctx->dirty_ranges));
}

స్థిర శూన్య l2flush_multi_కార్మికుడు(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    నిర్మాణం l2flush_multi_task * task = (నిర్మాణం l2flush_multi_task *) దత్తాంశం;
    (శూన్య) n;

    స్థిరరాశి uint32_t gb_మొదటి = i * task->ఖండాలు_చొప్పున_thread;
    uint32_t       gb_చివరి  = gb_మొదటి + task->ఖండాలు_చొప్పున_thread;
    ఒకవేళ (gb_చివరి > task->మొత్తం_ఖండాలు) {
        gb_చివరి = task->మొత్తం_ఖండాలు;
    }
    ఒకవేళ (gb_మొదటి >= gb_చివరి) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_L2FLUSH, gb_మొదటి);

    కొరకు (uint32_t r = 0; r < task->n_ranges; r++) {
        స్థిరరాశి నిర్మాణం l2flush_range * rg = &task->ranges[r];
        స్థిరరాశి uint32_t rb_మొదటి = rg->ఖండం_మొదటి;
        స్థిరరాశి uint32_t rb_చివరి  = rg->ఖండం_మొదటి + rg->n_ఖండాలు;

        స్థిరరాశి uint32_t lo = gb_మొదటి > rb_మొదటి ? gb_మొదటి : rb_మొదటి;
        స్థిరరాశి uint32_t hi = gb_చివరి  < rb_చివరి  ? gb_చివరి  : rb_చివరి;
        ఒకవేళ (lo >= hi) {
            కొనసాగు;
        }

        స్థిరరాశి uint32_t s = rg->ప్రారంభం + (lo - rb_మొదటి) * HEX_L2_ఖండం_కొలత;
        uint32_t       e = rg->ప్రారంభం + (hi - rb_మొదటి) * HEX_L2_ఖండం_కొలత;
        ఒకవేళ (e > rg->చివర) {
            e = rg->చివర;
        }
        hex_l2flush((శూన్య *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_మొదటి);
}

స్థిర శూన్య విలీనం_dirty_ranges(నిర్మాణం htp_context * ctx) {
    కొరకు (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[i];
        ఒకవేళ (!r->ప్రారంభం) కొనసాగు;

        కొరకు (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            నిర్మాణం htp_dirty_range * s = &ctx->dirty_ranges[j];
            ఒకవేళ (i == j || !s->ప్రారంభం || r->చివర < s->ప్రారంభం || s->చివర < r->ప్రారంభం) {
                j++;
                కొనసాగు;
            }

            r->ప్రారంభం = MIN(r->ప్రారంభం, s->ప్రారంభం);
            r->చివర   = MAX(r->చివర, s->చివర);
            s->ప్రారంభం = 0;
            s->చివర   = 0;
            j = 0;
        }
    }
}

శూన్య htp_dcache_mark_dirty_tensors(నిర్మాణం htp_context * ctx, స్థిరరాశి నిర్మాణం htp_tensor * స్థిరరాశి * tensors, uint32_t n) {
    స్థిరరాశి నిర్మాణం htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    కొరకు (uint32_t i = 0; i < n; i++) {
        స్థిరరాశి నిర్మాణం htp_tensor * t = tensors[i];
        ఒకవేళ (!t || (t->flags & (HTP_TENSOR_బరువు | HTP_TENSOR_FENCE))) {
            కొనసాగు;
        }

        uint32_t t_ప్రారంభం = t->దత్తాంశం;
        uint32_t t_చివర   = t_ప్రారంభం + t->కొలత;

        bool merged = false;
        కొరకు (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[j];
            ఒకవేళ (!r->ప్రారంభం) కొనసాగు;

            ఒకవేళ (r->ప్రారంభం <= t_చివర && t_ప్రారంభం <= r->చివర) {
                uint32_t new_ప్రారంభం = (t_ప్రారంభం < r->ప్రారంభం) ? t_ప్రారంభం : r->ప్రారంభం;
                uint32_t new_చివర   = (t_చివర > r->చివర)     ? t_చివర   : r->చివర;
                r->ప్రారంభం = new_ప్రారంభం;
                r->చివర   = new_చివర;
                merged   = true;
            }
        }

        ఒకవేళ (!merged) {
            pending[n_pending++] = t;
        }
    }

    విలీనం_dirty_ranges(ctx);

    ఒకవేళ (n_pending == 0) {
        తిరిగి;
    }

    uint32_t ఖాళీలు_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t చురుకైన_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_చురుకైన = 0;
    uint32_t n_ఖాళీలు  = 0;
    కొరకు (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        ఒకవేళ (ctx->dirty_ranges[j].ప్రారంభం) {
            చురుకైన_indices[n_చురుకైన++] = j;
        } లేకపోతే {
            ఖాళీలు_indices[n_ఖాళీలు++]   = j;
        }
    }

    ఒకవేళ (n_pending <= n_ఖాళీలు) {
        కొరకు (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = ఖాళీలు_indices[i];
            నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->ప్రారంభం = pending[i]->దత్తాంశం;
            r->చివర   = pending[i]->దత్తాంశం + pending[i]->కొలత;
        }
        విలీనం_dirty_ranges(ctx);
        తిరిగి;
    }

    uint32_t n_evict = n_pending - n_ఖాళీలు;
    uint32_t మొత్తం_evict_కొలత = 0;
    కొరకు (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = చురుకైన_indices[i];
        నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[idx];
        మొత్తం_evict_కొలత += r->చివర - r->ప్రారంభం;
    }

    ఒకవేళ (మొత్తం_evict_కొలత > HEX_L2_FLUSH_ALL_పరిమితి) {
        flush_all_dcache(ctx);
        కొరకు (uint32_t i = 0; i < n_pending; i++) {
            నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->ప్రారంభం = pending[i]->దత్తాంశం;
            r->చివర   = pending[i]->దత్తాంశం + pending[i]->కొలత;
        }
        విలీనం_dirty_ranges(ctx);
        తిరిగి;
    }

    ఒకవేళ (మొత్తం_evict_కొలత > HEX_L2_FLUSH_WQ_పరిమితి && ctx->n_దారాలు > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        నిర్మాణం l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t ఖండం_acc = 0;
        కొరకు (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = చురుకైన_indices[i];
            నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[idx];

            నిర్మాణం l2flush_range * rg = &task.ranges[i];
            rg->ప్రారంభం = hex_align_కిందికి((size_t) r->ప్రారంభం, HEX_L2_పంక్తి_కొలత);
            rg->చివర   = hex_align_పైకి((size_t) r->చివర, HEX_L2_పంక్తి_కొలత);
            rg->ఖండం_మొదటి = ఖండం_acc;
            rg->n_ఖండాలు = (rg->చివర - rg->ప్రారంభం + HEX_L2_ఖండం_కొలత - 1) / HEX_L2_ఖండం_కొలత;
            ఖండం_acc += rg->n_ఖండాలు;
        }

        task.మొత్తం_ఖండాలు      = ఖండం_acc;
        task.ఖండాలు_చొప్పున_thread = fastdiv(ఖండం_acc + ctx->n_దారాలు - 1, &ctx->n_దారాలు_div);

        work_వరుస_నిరంతరం(ctx->work_వరుస, l2flush_multi_కార్మికుడు, &task, ctx->n_దారాలు);
    } లేకపోతే {
        నిర్మాణం htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        కొరకు (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = చురుకైన_indices[i];
            నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t కొలత = r->చివర - r->ప్రారంభం;
            hex_l2flush((శూన్య *) (uintptr_t) r->ప్రారంభం, కొలత);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    కొరకు (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = చురుకైన_indices[i];
        నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->ప్రారంభం = pending[i]->దత్తాంశం;
        r->చివర   = pending[i]->దత్తాంశం + pending[i]->కొలత;
    }

    కొరకు (uint32_t i = 0; i < n_ఖాళీలు; i++) {
        uint32_t idx = ఖాళీలు_indices[i];
        నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->ప్రారంభం = pending[n_evict + i]->దత్తాంశం;
        r->చివర   = pending[n_evict + i]->దత్తాంశం + pending[n_evict + i]->కొలత;
    }

    విలీనం_dirty_ranges(ctx);
}

స్థిర శూన్య make_tensor_clean(నిర్మాణం htp_context * ctx, స్థిరరాశి నిర్మాణం htp_tensor * t) {
    ఒకవేళ (!t || (t->flags & (HTP_TENSOR_బరువు | HTP_TENSOR_FENCE))) {
        తిరిగి;
    }

    uint32_t t_ప్రారంభం = t->దత్తాంశం;
    uint32_t t_చివర   = t_ప్రారంభం + t->కొలత;

    కొరకు (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[i];
        ఒకవేళ (!r->ప్రారంభం) కొనసాగు;

        ఒకవేళ (r->ప్రారంభం < t_చివర && t_ప్రారంభం < r->చివర) {
            ఒకవేళ (t_ప్రారంభం <= r->ప్రారంభం && r->చివర <= t_చివర) {
                r->ప్రారంభం = 0;
                r->చివర   = 0;
            } లేకపోతే ఒకవేళ (t_ప్రారంభం <= r->ప్రారంభం) {
                r->ప్రారంభం = t_చివర;
            } లేకపోతే ఒకవేళ (r->చివర <= t_చివర) {
                r->చివర = t_ప్రారంభం;
            }
        }
    }
}

స్థిర అంతఃవరుస bool is_tensor_dirty(నిర్మాణం htp_context * ctx, స్థిరరాశి నిర్మాణం htp_tensor * t) {
    ఒకవేళ (!t || (t->flags & (HTP_TENSOR_బరువు | HTP_TENSOR_FENCE))) {
        తిరిగి false;
    }

    uint32_t t_ప్రారంభం = t->దత్తాంశం;
    uint32_t t_చివర   = t_ప్రారంభం + t->కొలత;

    కొరకు (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[i];
        ఒకవేళ (!r->ప్రారంభం) కొనసాగు;

        ఒకవేళ (r->ప్రారంభం < t_చివర && t_ప్రారంభం < r->చివర) {
            తిరిగి true;
        }
    }
    తిరిగి false;
}

స్థిర శూన్య flush_dirty_ranges(నిర్మాణం htp_context * ctx, స్థిరరాశి నిర్మాణం htp_dirty_range * ranges, uint32_t n_ranges, uint64_t మొత్తం_dirty) {
    ఒకవేళ (మొత్తం_dirty >= HEX_L2_FLUSH_WQ_పరిమితి && ctx->n_దారాలు > 1) {
        నిర్మాణం l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t ఖండం_acc = 0;
        కొరకు (uint32_t i = 0; i < n_ranges; i++) {
            స్థిరరాశి నిర్మాణం htp_dirty_range * r = &ranges[i];
            నిర్మాణం l2flush_range * rg = &task.ranges[i];
            rg->ప్రారంభం = hex_align_కిందికి((size_t) r->ప్రారంభం, HEX_L2_పంక్తి_కొలత);
            rg->చివర   = hex_align_పైకి((size_t) r->చివర, HEX_L2_పంక్తి_కొలత);
            rg->ఖండం_మొదటి = ఖండం_acc;
            rg->n_ఖండాలు = (rg->చివర - rg->ప్రారంభం + HEX_L2_ఖండం_కొలత - 1) / HEX_L2_ఖండం_కొలత;
            ఖండం_acc += rg->n_ఖండాలు;
        }

        task.మొత్తం_ఖండాలు      = ఖండం_acc;
        task.ఖండాలు_చొప్పున_thread = fastdiv(ఖండం_acc + ctx->n_దారాలు - 1, &ctx->n_దారాలు_div);

        work_వరుస_నిరంతరం(ctx->work_వరుస, l2flush_multi_కార్మికుడు, &task, ctx->n_దారాలు);
    } లేకపోతే {
        నిర్మాణం htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        కొరకు (uint32_t i = 0; i < n_ranges; i++) {
            స్థిరరాశి నిర్మాణం htp_dirty_range * r = &ranges[i];
            hex_l2flush((శూన్య *) (uintptr_t) r->ప్రారంభం, r->చివర - r->ప్రారంభం);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

శూన్య htp_dcache_flush_dirty_ranges(నిర్మాణం htp_context * ctx) {
    నిర్మాణం htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t మొత్తం_dirty = 0;

    కొరకు (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        స్థిరరాశి నిర్మాణం htp_dirty_range * r = &ctx->dirty_ranges[i];
        ఒకవేళ (!r->ప్రారంభం) {
            కొనసాగు;
        }
        ranges[n_ranges++] = *r;
        మొత్తం_dirty += r->చివర - r->ప్రారంభం;
    }

    ఒకవేళ (మొత్తం_dirty == 0) {
        తిరిగి;
    }

    ఒకవేళ (మొత్తం_dirty > HEX_L2_FLUSH_ALL_పరిమితి) {
        flush_all_dcache(ctx);
        తిరిగి;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, మొత్తం_dirty);
    memset(ctx->dirty_ranges, 0, పరిమాణం(ctx->dirty_ranges));
}

శూన్య htp_dcache_flush_dirty_tensors(నిర్మాణం htp_context * ctx, స్థిరరాశి నిర్మాణం htp_tensor * స్థిరరాశి * tensors, uint32_t n) {
    స్థిరరాశి నిర్మాణం htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    నిర్మాణం htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t మొత్తం_dirty = 0;

    కొరకు (uint32_t i = 0; i < n; i++) {
        స్థిరరాశి నిర్మాణం htp_tensor * t = tensors[i];
        ఒకవేళ (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].ప్రారంభం = t->దత్తాంశం;
            ranges[n_dirty - 1].చివర   = t->దత్తాంశం + t->కొలత;
            మొత్తం_dirty += t->కొలత;
        }
    }

    ఒకవేళ (మొత్తం_dirty == 0) {
        తిరిగి;
    }

    ఒకవేళ (మొత్తం_dirty > HEX_L2_FLUSH_ALL_పరిమితి) {
        flush_all_dcache(ctx);
        తిరిగి;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, మొత్తం_dirty);
    కొరకు (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
