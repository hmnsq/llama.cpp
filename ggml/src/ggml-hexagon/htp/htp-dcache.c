#ଅନ୍ତର୍ଭାଵ "htp-dcache.h"

#ଅନ୍ତର୍ଭାଵ <qurt.h>
#ଅନ୍ତର୍ଭାଵ <qurt_memory.h>
#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>

#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-utils.h"
#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "work-queue.h"

ସଂରଚନା l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

ସଂରଚନା l2flush_multi_task {
    ସଂରଚନା htp_thread_trace * trace;
    ସଂରଚନା l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  total_blocks;
    uint32_t                  blocks_per_thread;
};

ସ୍ଥିର ଶୂନ୍ୟ flush_all_dcache(ସଂରଚନା htp_context * ctx) {
    ସଂରଚନା htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, ଆକାର(ctx->dirty_ranges));
}

ସ୍ଥିର ଶୂନ୍ୟ l2flush_multi_worker(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ * data) {
    ସଂରଚନା l2flush_multi_task * task = (ସଂରଚନା l2flush_multi_task *) data;
    (ଶୂନ୍ୟ) n;

    ଅଚଳ uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    ଯଦି (gb_last > task->total_blocks) {
        gb_last = task->total_blocks;
    }
    ଯଦି (gb_first >= gb_last) {
        ଫେରନ୍ତୁ;
    }

    ସଂରଚନା htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    ପାଇଁ (uint32_t r = 0; r < task->n_ranges; r++) {
        ଅଚଳ ସଂରଚନା l2flush_range * rg = &task->ranges[r];
        ଅଚଳ uint32_t rb_first = rg->block_first;
        ଅଚଳ uint32_t rb_last  = rg->block_first + rg->n_blocks;

        ଅଚଳ uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        ଅଚଳ uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        ଯଦି (lo >= hi) {
            ଜାରି;
        }

        ଅଚଳ uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_SIZE;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_SIZE;
        ଯଦି (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((ଶୂନ୍ୟ *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

ସ୍ଥିର ଶୂନ୍ୟ merge_dirty_ranges(ସଂରଚନା htp_context * ctx) {
    ପାଇଁ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[i];
        ଯଦି (!r->start) ଜାରି;

        ପାଇଁ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            ସଂରଚନା htp_dirty_range * s = &ctx->dirty_ranges[j];
            ଯଦି (i == j || !s->start || r->end < s->start || s->end < r->start) {
                j++;
                ଜାରି;
            }

            r->start = MIN(r->start, s->start);
            r->end   = MAX(r->end, s->end);
            s->start = 0;
            s->end   = 0;
            j = 0;
        }
    }
}

ଶୂନ୍ୟ htp_dcache_mark_dirty_tensors(ସଂରଚନା htp_context * ctx, ଅଚଳ ସଂରଚନା htp_tensor * ଅଚଳ * tensors, uint32_t n) {
    ଅଚଳ ସଂରଚନା htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    ପାଇଁ (uint32_t i = 0; i < n; i++) {
        ଅଚଳ ସଂରଚନା htp_tensor * t = tensors[i];
        ଯଦି (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            ଜାରି;
        }

        uint32_t t_start = t->data;
        uint32_t t_end   = t_start + t->size;

        bool merged = false;
        ପାଇଁ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[j];
            ଯଦି (!r->start) ଜାରି;

            ଯଦି (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end)     ? t_end   : r->end;
                r->start = new_start;
                r->end   = new_end;
                merged   = true;
            }
        }

        ଯଦି (!merged) {
            pending[n_pending++] = t;
        }
    }

    merge_dirty_ranges(ctx);

    ଯଦି (n_pending == 0) {
        ଫେରନ୍ତୁ;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    ପାଇଁ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        ଯଦି (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } ଅନ୍ୟଥା {
            empty_indices[n_empty++]   = j;
        }
    }

    ଯଦି (n_pending <= n_empty) {
        ପାଇଁ (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        ଫେରନ୍ତୁ;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t total_evict_size = 0;
    ପାଇଁ (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[idx];
        total_evict_size += r->end - r->start;
    }

    ଯଦି (total_evict_size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        ପାଇଁ (uint32_t i = 0; i < n_pending; i++) {
            ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        ଫେରନ୍ତୁ;
    }

    ଯଦି (total_evict_size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        ସଂରଚନା l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        ପାଇଁ (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[idx];

            ସଂରଚନା l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_ଧାଡ଼ି_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_ଧାଡ଼ି_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } ଅନ୍ୟଥା {
        ସଂରଚନା htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        ପାଇଁ (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t size = r->end - r->start;
            hex_l2flush((ଶୂନ୍ୟ *) (uintptr_t) r->start, size);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    ପାଇଁ (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->data;
        r->end   = pending[i]->data + pending[i]->size;
    }

    ପାଇଁ (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->data;
        r->end   = pending[n_evict + i]->data + pending[n_evict + i]->size;
    }

    merge_dirty_ranges(ctx);
}

ସ୍ଥିର ଶୂନ୍ୟ make_tensor_clean(ସଂରଚନା htp_context * ctx, ଅଚଳ ସଂରଚନା htp_tensor * t) {
    ଯଦି (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        ଫେରନ୍ତୁ;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    ପାଇଁ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[i];
        ଯଦି (!r->start) ଜାରି;

        ଯଦି (r->start < t_end && t_start < r->end) {
            ଯଦି (t_start <= r->start && r->end <= t_end) {
                r->start = 0;
                r->end   = 0;
            } ଅନ୍ୟଥା ଯଦି (t_start <= r->start) {
                r->start = t_end;
            } ଅନ୍ୟଥା ଯଦି (r->end <= t_end) {
                r->end = t_start;
            }
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool is_tensor_dirty(ସଂରଚନା htp_context * ctx, ଅଚଳ ସଂରଚନା htp_tensor * t) {
    ଯଦି (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        ଫେରନ୍ତୁ false;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    ପାଇଁ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[i];
        ଯଦି (!r->start) ଜାରି;

        ଯଦି (r->start < t_end && t_start < r->end) {
            ଫେରନ୍ତୁ true;
        }
    }
    ଫେରନ୍ତୁ false;
}

ସ୍ଥିର ଶୂନ୍ୟ flush_dirty_ranges(ସଂରଚନା htp_context * ctx, ଅଚଳ ସଂରଚନା htp_dirty_range * ranges, uint32_t n_ranges, uint64_t total_dirty) {
    ଯଦି (total_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        ସଂରଚନା l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        ପାଇଁ (uint32_t i = 0; i < n_ranges; i++) {
            ଅଚଳ ସଂରଚନା htp_dirty_range * r = &ranges[i];
            ସଂରଚନା l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_ଧାଡ଼ି_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_ଧାଡ଼ି_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } ଅନ୍ୟଥା {
        ସଂରଚନା htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        ପାଇଁ (uint32_t i = 0; i < n_ranges; i++) {
            ଅଚଳ ସଂରଚନା htp_dirty_range * r = &ranges[i];
            hex_l2flush((ଶୂନ୍ୟ *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

ଶୂନ୍ୟ htp_dcache_flush_dirty_ranges(ସଂରଚନା htp_context * ctx) {
    ସଂରଚନା htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t total_dirty = 0;

    ପାଇଁ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ଅଚଳ ସଂରଚନା htp_dirty_range * r = &ctx->dirty_ranges[i];
        ଯଦି (!r->start) {
            ଜାରି;
        }
        ranges[n_ranges++] = *r;
        total_dirty += r->end - r->start;
    }

    ଯଦି (total_dirty == 0) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        ଫେରନ୍ତୁ;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, total_dirty);
    memset(ctx->dirty_ranges, 0, ଆକାର(ctx->dirty_ranges));
}

ଶୂନ୍ୟ htp_dcache_flush_dirty_tensors(ସଂରଚନା htp_context * ctx, ଅଚଳ ସଂରଚନା htp_tensor * ଅଚଳ * tensors, uint32_t n) {
    ଅଚଳ ସଂରଚନା htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    ସଂରଚନା htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t total_dirty = 0;

    ପାଇଁ (uint32_t i = 0; i < n; i++) {
        ଅଚଳ ସଂରଚନା htp_tensor * t = tensors[i];
        ଯଦି (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->data;
            ranges[n_dirty - 1].end   = t->data + t->size;
            total_dirty += t->size;
        }
    }

    ଯଦି (total_dirty == 0) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        ଫେରନ୍ତୁ;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, total_dirty);
    ପାଇଁ (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
