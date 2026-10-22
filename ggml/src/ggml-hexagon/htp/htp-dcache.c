#അന്തര്ഭാവ "htp-dcache.h"

#അന്തര്ഭാവ <qurt.h>
#അന്തര്ഭാവ <qurt_memory.h>
#അന്തര്ഭാവ <HAP_farf.h>

#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "hex-utils.h"
#അന്തര്ഭാവ "hex-fastdiv.h"
#അന്തര്ഭാവ "hex-profile.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "work-queue.h"

ഘടന l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

ഘടന l2flush_multi_task {
    ഘടന htp_thread_trace * trace;
    ഘടന l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  total_blocks;
    uint32_t                  blocks_per_thread;
};

സ്ഥിതം ശൂന്യം flush_all_dcache(ഘടന htp_context * ctx) {
    ഘടന htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, വലുപ്പം(ctx->dirty_ranges));
}

സ്ഥിതം ശൂന്യം l2flush_multi_worker(unsigned പൂർണ്ണസംഖ്യ n, unsigned പൂർണ്ണസംഖ്യ i, ശൂന്യം * data) {
    ഘടന l2flush_multi_task * task = (ഘടന l2flush_multi_task *) data;
    (ശൂന്യം) n;

    സ്ഥിരം uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    എങ്കിൽ (gb_last > task->total_blocks) {
        gb_last = task->total_blocks;
    }
    എങ്കിൽ (gb_first >= gb_last) {
        തിരികെ;
    }

    ഘടന htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    വേണ്ടി (uint32_t r = 0; r < task->n_ranges; r++) {
        സ്ഥിരം ഘടന l2flush_range * rg = &task->ranges[r];
        സ്ഥിരം uint32_t rb_first = rg->block_first;
        സ്ഥിരം uint32_t rb_last  = rg->block_first + rg->n_blocks;

        സ്ഥിരം uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        സ്ഥിരം uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        എങ്കിൽ (lo >= hi) {
            തുടരുക;
        }

        സ്ഥിരം uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_SIZE;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_SIZE;
        എങ്കിൽ (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((ശൂന്യം *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

സ്ഥിതം ശൂന്യം merge_dirty_ranges(ഘടന htp_context * ctx) {
    വേണ്ടി (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ഘടന htp_dirty_range * r = &ctx->dirty_ranges[i];
        എങ്കിൽ (!r->start) തുടരുക;

        വേണ്ടി (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            ഘടന htp_dirty_range * s = &ctx->dirty_ranges[j];
            എങ്കിൽ (i == j || !s->start || r->end < s->start || s->end < r->start) {
                j++;
                തുടരുക;
            }

            r->start = MIN(r->start, s->start);
            r->end   = MAX(r->end, s->end);
            s->start = 0;
            s->end   = 0;
            j = 0;
        }
    }
}

ശൂന്യം htp_dcache_mark_dirty_tensors(ഘടന htp_context * ctx, സ്ഥിരം ഘടന htp_tensor * സ്ഥിരം * tensors, uint32_t n) {
    സ്ഥിരം ഘടന htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    വേണ്ടി (uint32_t i = 0; i < n; i++) {
        സ്ഥിരം ഘടന htp_tensor * t = tensors[i];
        എങ്കിൽ (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            തുടരുക;
        }

        uint32_t t_start = t->data;
        uint32_t t_end   = t_start + t->size;

        bool merged = false;
        വേണ്ടി (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            ഘടന htp_dirty_range * r = &ctx->dirty_ranges[j];
            എങ്കിൽ (!r->start) തുടരുക;

            എങ്കിൽ (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end)     ? t_end   : r->end;
                r->start = new_start;
                r->end   = new_end;
                merged   = true;
            }
        }

        എങ്കിൽ (!merged) {
            pending[n_pending++] = t;
        }
    }

    merge_dirty_ranges(ctx);

    എങ്കിൽ (n_pending == 0) {
        തിരികെ;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    വേണ്ടി (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        എങ്കിൽ (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } അല്ലെങ്കിൽ {
            empty_indices[n_empty++]   = j;
        }
    }

    എങ്കിൽ (n_pending <= n_empty) {
        വേണ്ടി (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            ഘടന htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        തിരികെ;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t total_evict_size = 0;
    വേണ്ടി (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        ഘടന htp_dirty_range * r = &ctx->dirty_ranges[idx];
        total_evict_size += r->end - r->start;
    }

    എങ്കിൽ (total_evict_size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        വേണ്ടി (uint32_t i = 0; i < n_pending; i++) {
            ഘടന htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        തിരികെ;
    }

    എങ്കിൽ (total_evict_size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        ഘടന l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        വേണ്ടി (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            ഘടന htp_dirty_range * r = &ctx->dirty_ranges[idx];

            ഘടന l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_വരി_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_വരി_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } അല്ലെങ്കിൽ {
        ഘടന htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        വേണ്ടി (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            ഘടന htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t size = r->end - r->start;
            hex_l2flush((ശൂന്യം *) (uintptr_t) r->start, size);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    വേണ്ടി (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        ഘടന htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->data;
        r->end   = pending[i]->data + pending[i]->size;
    }

    വേണ്ടി (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        ഘടന htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->data;
        r->end   = pending[n_evict + i]->data + pending[n_evict + i]->size;
    }

    merge_dirty_ranges(ctx);
}

സ്ഥിതം ശൂന്യം make_tensor_clean(ഘടന htp_context * ctx, സ്ഥിരം ഘടന htp_tensor * t) {
    എങ്കിൽ (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        തിരികെ;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    വേണ്ടി (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ഘടന htp_dirty_range * r = &ctx->dirty_ranges[i];
        എങ്കിൽ (!r->start) തുടരുക;

        എങ്കിൽ (r->start < t_end && t_start < r->end) {
            എങ്കിൽ (t_start <= r->start && r->end <= t_end) {
                r->start = 0;
                r->end   = 0;
            } അല്ലെങ്കിൽ എങ്കിൽ (t_start <= r->start) {
                r->start = t_end;
            } അല്ലെങ്കിൽ എങ്കിൽ (r->end <= t_end) {
                r->end = t_start;
            }
        }
    }
}

സ്ഥിതം വരിയിൽ bool is_tensor_dirty(ഘടന htp_context * ctx, സ്ഥിരം ഘടന htp_tensor * t) {
    എങ്കിൽ (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        തിരികെ false;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    വേണ്ടി (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ഘടന htp_dirty_range * r = &ctx->dirty_ranges[i];
        എങ്കിൽ (!r->start) തുടരുക;

        എങ്കിൽ (r->start < t_end && t_start < r->end) {
            തിരികെ true;
        }
    }
    തിരികെ false;
}

സ്ഥിതം ശൂന്യം flush_dirty_ranges(ഘടന htp_context * ctx, സ്ഥിരം ഘടന htp_dirty_range * ranges, uint32_t n_ranges, uint64_t total_dirty) {
    എങ്കിൽ (total_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        ഘടന l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        വേണ്ടി (uint32_t i = 0; i < n_ranges; i++) {
            സ്ഥിരം ഘടന htp_dirty_range * r = &ranges[i];
            ഘടന l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_വരി_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_വരി_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } അല്ലെങ്കിൽ {
        ഘടന htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        വേണ്ടി (uint32_t i = 0; i < n_ranges; i++) {
            സ്ഥിരം ഘടന htp_dirty_range * r = &ranges[i];
            hex_l2flush((ശൂന്യം *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

ശൂന്യം htp_dcache_flush_dirty_ranges(ഘടന htp_context * ctx) {
    ഘടന htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t total_dirty = 0;

    വേണ്ടി (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        സ്ഥിരം ഘടന htp_dirty_range * r = &ctx->dirty_ranges[i];
        എങ്കിൽ (!r->start) {
            തുടരുക;
        }
        ranges[n_ranges++] = *r;
        total_dirty += r->end - r->start;
    }

    എങ്കിൽ (total_dirty == 0) {
        തിരികെ;
    }

    എങ്കിൽ (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        തിരികെ;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, total_dirty);
    memset(ctx->dirty_ranges, 0, വലുപ്പം(ctx->dirty_ranges));
}

ശൂന്യം htp_dcache_flush_dirty_tensors(ഘടന htp_context * ctx, സ്ഥിരം ഘടന htp_tensor * സ്ഥിരം * tensors, uint32_t n) {
    സ്ഥിരം ഘടന htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    ഘടന htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t total_dirty = 0;

    വേണ്ടി (uint32_t i = 0; i < n; i++) {
        സ്ഥിരം ഘടന htp_tensor * t = tensors[i];
        എങ്കിൽ (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->data;
            ranges[n_dirty - 1].end   = t->data + t->size;
            total_dirty += t->size;
        }
    }

    എങ്കിൽ (total_dirty == 0) {
        തിരികെ;
    }

    എങ്കിൽ (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        തിരികെ;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, total_dirty);
    വേണ്ടി (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
