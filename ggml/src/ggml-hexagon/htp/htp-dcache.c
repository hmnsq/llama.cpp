#அந்தர்பாவ "htp-dcache.h"

#அந்தர்பாவ <qurt.h>
#அந்தர்பாவ <qurt_memory.h>
#அந்தர்பாவ <HAP_farf.h>

#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "hex-utils.h"
#அந்தர்பாவ "hex-fastdiv.h"
#அந்தர்பாவ "hex-profile.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "work-queue.h"

கட்டமைப்பு l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

கட்டமைப்பு l2flush_multi_task {
    கட்டமைப்பு htp_thread_trace * trace;
    கட்டமைப்பு l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  total_blocks;
    uint32_t                  blocks_per_thread;
};

நிலையான வெறுமை flush_all_dcache(கட்டமைப்பு htp_context * ctx) {
    கட்டமைப்பு htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, அளவு(ctx->dirty_ranges));
}

நிலையான வெறுமை l2flush_multi_worker(unsigned முழு n, unsigned முழு i, வெறுமை * data) {
    கட்டமைப்பு l2flush_multi_task * task = (கட்டமைப்பு l2flush_multi_task *) data;
    (வெறுமை) n;

    மாறிலி uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    எனில் (gb_last > task->total_blocks) {
        gb_last = task->total_blocks;
    }
    எனில் (gb_first >= gb_last) {
        திரும்பு;
    }

    கட்டமைப்பு htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    க்கு (uint32_t r = 0; r < task->n_ranges; r++) {
        மாறிலி கட்டமைப்பு l2flush_range * rg = &task->ranges[r];
        மாறிலி uint32_t rb_first = rg->block_first;
        மாறிலி uint32_t rb_last  = rg->block_first + rg->n_blocks;

        மாறிலி uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        மாறிலி uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        எனில் (lo >= hi) {
            தொடர்;
        }

        மாறிலி uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_SIZE;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_SIZE;
        எனில் (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((வெறுமை *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

நிலையான வெறுமை merge_dirty_ranges(கட்டமைப்பு htp_context * ctx) {
    க்கு (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[i];
        எனில் (!r->start) தொடர்;

        க்கு (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            கட்டமைப்பு htp_dirty_range * s = &ctx->dirty_ranges[j];
            எனில் (i == j || !s->start || r->end < s->start || s->end < r->start) {
                j++;
                தொடர்;
            }

            r->start = MIN(r->start, s->start);
            r->end   = MAX(r->end, s->end);
            s->start = 0;
            s->end   = 0;
            j = 0;
        }
    }
}

வெறுமை htp_dcache_mark_dirty_tensors(கட்டமைப்பு htp_context * ctx, மாறிலி கட்டமைப்பு htp_tensor * மாறிலி * tensors, uint32_t n) {
    மாறிலி கட்டமைப்பு htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    க்கு (uint32_t i = 0; i < n; i++) {
        மாறிலி கட்டமைப்பு htp_tensor * t = tensors[i];
        எனில் (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            தொடர்;
        }

        uint32_t t_start = t->data;
        uint32_t t_end   = t_start + t->size;

        bool merged = false;
        க்கு (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[j];
            எனில் (!r->start) தொடர்;

            எனில் (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end)     ? t_end   : r->end;
                r->start = new_start;
                r->end   = new_end;
                merged   = true;
            }
        }

        எனில் (!merged) {
            pending[n_pending++] = t;
        }
    }

    merge_dirty_ranges(ctx);

    எனில் (n_pending == 0) {
        திரும்பு;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    க்கு (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        எனில் (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } இல்லையெனில் {
            empty_indices[n_empty++]   = j;
        }
    }

    எனில் (n_pending <= n_empty) {
        க்கு (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        திரும்பு;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t total_evict_size = 0;
    க்கு (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[idx];
        total_evict_size += r->end - r->start;
    }

    எனில் (total_evict_size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        க்கு (uint32_t i = 0; i < n_pending; i++) {
            கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        திரும்பு;
    }

    எனில் (total_evict_size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        கட்டமைப்பு l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        க்கு (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[idx];

            கட்டமைப்பு l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_வரி_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_வரி_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } இல்லையெனில் {
        கட்டமைப்பு htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        க்கு (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t size = r->end - r->start;
            hex_l2flush((வெறுமை *) (uintptr_t) r->start, size);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    க்கு (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->data;
        r->end   = pending[i]->data + pending[i]->size;
    }

    க்கு (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->data;
        r->end   = pending[n_evict + i]->data + pending[n_evict + i]->size;
    }

    merge_dirty_ranges(ctx);
}

நிலையான வெறுமை make_tensor_clean(கட்டமைப்பு htp_context * ctx, மாறிலி கட்டமைப்பு htp_tensor * t) {
    எனில் (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        திரும்பு;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    க்கு (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[i];
        எனில் (!r->start) தொடர்;

        எனில் (r->start < t_end && t_start < r->end) {
            எனில் (t_start <= r->start && r->end <= t_end) {
                r->start = 0;
                r->end   = 0;
            } இல்லையெனில் எனில் (t_start <= r->start) {
                r->start = t_end;
            } இல்லையெனில் எனில் (r->end <= t_end) {
                r->end = t_start;
            }
        }
    }
}

நிலையான உள்வரி bool is_tensor_dirty(கட்டமைப்பு htp_context * ctx, மாறிலி கட்டமைப்பு htp_tensor * t) {
    எனில் (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        திரும்பு false;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    க்கு (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[i];
        எனில் (!r->start) தொடர்;

        எனில் (r->start < t_end && t_start < r->end) {
            திரும்பு true;
        }
    }
    திரும்பு false;
}

நிலையான வெறுமை flush_dirty_ranges(கட்டமைப்பு htp_context * ctx, மாறிலி கட்டமைப்பு htp_dirty_range * ranges, uint32_t n_ranges, uint64_t total_dirty) {
    எனில் (total_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        கட்டமைப்பு l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        க்கு (uint32_t i = 0; i < n_ranges; i++) {
            மாறிலி கட்டமைப்பு htp_dirty_range * r = &ranges[i];
            கட்டமைப்பு l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_வரி_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_வரி_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } இல்லையெனில் {
        கட்டமைப்பு htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        க்கு (uint32_t i = 0; i < n_ranges; i++) {
            மாறிலி கட்டமைப்பு htp_dirty_range * r = &ranges[i];
            hex_l2flush((வெறுமை *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

வெறுமை htp_dcache_flush_dirty_ranges(கட்டமைப்பு htp_context * ctx) {
    கட்டமைப்பு htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t total_dirty = 0;

    க்கு (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        மாறிலி கட்டமைப்பு htp_dirty_range * r = &ctx->dirty_ranges[i];
        எனில் (!r->start) {
            தொடர்;
        }
        ranges[n_ranges++] = *r;
        total_dirty += r->end - r->start;
    }

    எனில் (total_dirty == 0) {
        திரும்பு;
    }

    எனில் (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        திரும்பு;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, total_dirty);
    memset(ctx->dirty_ranges, 0, அளவு(ctx->dirty_ranges));
}

வெறுமை htp_dcache_flush_dirty_tensors(கட்டமைப்பு htp_context * ctx, மாறிலி கட்டமைப்பு htp_tensor * மாறிலி * tensors, uint32_t n) {
    மாறிலி கட்டமைப்பு htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    கட்டமைப்பு htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t total_dirty = 0;

    க்கு (uint32_t i = 0; i < n; i++) {
        மாறிலி கட்டமைப்பு htp_tensor * t = tensors[i];
        எனில் (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->data;
            ranges[n_dirty - 1].end   = t->data + t->size;
            total_dirty += t->size;
        }
    }

    எனில் (total_dirty == 0) {
        திரும்பு;
    }

    எனில் (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        திரும்பு;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, total_dirty);
    க்கு (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
