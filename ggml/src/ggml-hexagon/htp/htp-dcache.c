#ಅನ್ತರ್ಭಾವ "htp-dcache.h"

#ಅನ್ತರ್ಭಾವ <qurt.h>
#ಅನ್ತರ್ಭಾವ <qurt_memory.h>
#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "work-queue.h"

ಸಂರಚನಾ l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

ಸಂರಚನಾ l2flush_multi_task {
    ಸಂರಚನಾ htp_thread_trace * trace;
    ಸಂರಚನಾ l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  total_blocks;
    uint32_t                  blocks_per_thread;
};

ಸ್ಥಿರ ಶೂನ್ಯ flush_all_dcache(ಸಂರಚನಾ htp_context * ctx) {
    ಸಂರಚನಾ htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, ಪರಿಮಾಣ(ctx->dirty_ranges));
}

ಸ್ಥಿರ ಶೂನ್ಯ l2flush_multi_worker(unsigned ಪೂರ್ಣಾಙ್ಕ n, unsigned ಪೂರ್ಣಾಙ್ಕ i, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ l2flush_multi_task * task = (ಸಂರಚನಾ l2flush_multi_task *) data;
    (ಶೂನ್ಯ) n;

    ಧ್ರುವ uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    ಯದಿ (gb_last > task->total_blocks) {
        gb_last = task->total_blocks;
    }
    ಯದಿ (gb_first >= gb_last) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಸಂರಚನಾ htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    ಪ್ರತಿ (uint32_t r = 0; r < task->n_ranges; r++) {
        ಧ್ರುವ ಸಂರಚನಾ l2flush_range * rg = &task->ranges[r];
        ಧ್ರುವ uint32_t rb_first = rg->block_first;
        ಧ್ರುವ uint32_t rb_last  = rg->block_first + rg->n_blocks;

        ಧ್ರುವ uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        ಧ್ರುವ uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        ಯದಿ (lo >= hi) {
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_SIZE;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_SIZE;
        ಯದಿ (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((ಶೂನ್ಯ *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

ಸ್ಥಿರ ಶೂನ್ಯ merge_dirty_ranges(ಸಂರಚನಾ htp_context * ctx) {
    ಪ್ರತಿ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ಯದಿ (!r->start) ಅನುವರ್ತ;

        ಪ್ರತಿ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            ಸಂರಚನಾ htp_dirty_range * s = &ctx->dirty_ranges[j];
            ಯದಿ (i == j || !s->start || r->end < s->start || s->end < r->start) {
                j++;
                ಅನುವರ್ತ;
            }

            r->start = MIN(r->start, s->start);
            r->end   = MAX(r->end, s->end);
            s->start = 0;
            s->end   = 0;
            j = 0;
        }
    }
}

ಶೂನ್ಯ htp_dcache_mark_dirty_tensors(ಸಂರಚನಾ htp_context * ctx, ಧ್ರುವ ಸಂರಚನಾ htp_tensor * ಧ್ರುವ * tensors, uint32_t n) {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    ಪ್ರತಿ (uint32_t i = 0; i < n; i++) {
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t = tensors[i];
        ಯದಿ (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            ಅನುವರ್ತ;
        }

        uint32_t t_start = t->data;
        uint32_t t_end   = t_start + t->size;

        bool merged = false;
        ಪ್ರತಿ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[j];
            ಯದಿ (!r->start) ಅನುವರ್ತ;

            ಯದಿ (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end)     ? t_end   : r->end;
                r->start = new_start;
                r->end   = new_end;
                merged   = true;
            }
        }

        ಯದಿ (!merged) {
            pending[n_pending++] = t;
        }
    }

    merge_dirty_ranges(ctx);

    ಯದಿ (n_pending == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    ಪ್ರತಿ (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        ಯದಿ (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } ಅನ್ಯಥಾ {
            empty_indices[n_empty++]   = j;
        }
    }

    ಯದಿ (n_pending <= n_empty) {
        ಪ್ರತಿ (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        ಪ್ರತ್ಯಾಗಮ;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t total_evict_size = 0;
    ಪ್ರತಿ (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[idx];
        total_evict_size += r->end - r->start;
    }

    ಯದಿ (total_evict_size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        ಪ್ರತಿ (uint32_t i = 0; i < n_pending; i++) {
            ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (total_evict_size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        ಸಂರಚನಾ l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        ಪ್ರತಿ (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[idx];

            ಸಂರಚನಾ l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_ಸಾಲು_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_ಸಾಲು_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } ಅನ್ಯಥಾ {
        ಸಂರಚನಾ htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        ಪ್ರತಿ (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t size = r->end - r->start;
            hex_l2flush((ಶೂನ್ಯ *) (uintptr_t) r->start, size);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    ಪ್ರತಿ (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->data;
        r->end   = pending[i]->data + pending[i]->size;
    }

    ಪ್ರತಿ (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->data;
        r->end   = pending[n_evict + i]->data + pending[n_evict + i]->size;
    }

    merge_dirty_ranges(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ make_tensor_clean(ಸಂರಚನಾ htp_context * ctx, ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t) {
    ಯದಿ (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    ಪ್ರತಿ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ಯದಿ (!r->start) ಅನುವರ್ತ;

        ಯದಿ (r->start < t_end && t_start < r->end) {
            ಯದಿ (t_start <= r->start && r->end <= t_end) {
                r->start = 0;
                r->end   = 0;
            } ಅನ್ಯಥಾ ಯದಿ (t_start <= r->start) {
                r->start = t_end;
            } ಅನ್ಯಥಾ ಯದಿ (r->end <= t_end) {
                r->end = t_start;
            }
        }
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool is_tensor_dirty(ಸಂರಚನಾ htp_context * ctx, ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t) {
    ಯದಿ (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        ಪ್ರತ್ಯಾಗಮ false;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    ಪ್ರತಿ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ಯದಿ (!r->start) ಅನುವರ್ತ;

        ಯದಿ (r->start < t_end && t_start < r->end) {
            ಪ್ರತ್ಯಾಗಮ true;
        }
    }
    ಪ್ರತ್ಯಾಗಮ false;
}

ಸ್ಥಿರ ಶೂನ್ಯ flush_dirty_ranges(ಸಂರಚನಾ htp_context * ctx, ಧ್ರುವ ಸಂರಚನಾ htp_dirty_range * ranges, uint32_t n_ranges, uint64_t total_dirty) {
    ಯದಿ (total_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        ಸಂರಚನಾ l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        ಪ್ರತಿ (uint32_t i = 0; i < n_ranges; i++) {
            ಧ್ರುವ ಸಂರಚನಾ htp_dirty_range * r = &ranges[i];
            ಸಂರಚನಾ l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_ಸಾಲು_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_ಸಾಲು_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } ಅನ್ಯಥಾ {
        ಸಂರಚನಾ htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        ಪ್ರತಿ (uint32_t i = 0; i < n_ranges; i++) {
            ಧ್ರುವ ಸಂರಚನಾ htp_dirty_range * r = &ranges[i];
            hex_l2flush((ಶೂನ್ಯ *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

ಶೂನ್ಯ htp_dcache_flush_dirty_ranges(ಸಂರಚನಾ htp_context * ctx) {
    ಸಂರಚನಾ htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t total_dirty = 0;

    ಪ್ರತಿ (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ಧ್ರುವ ಸಂರಚನಾ htp_dirty_range * r = &ctx->dirty_ranges[i];
        ಯದಿ (!r->start) {
            ಅನುವರ್ತ;
        }
        ranges[n_ranges++] = *r;
        total_dirty += r->end - r->start;
    }

    ಯದಿ (total_dirty == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        ಪ್ರತ್ಯಾಗಮ;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, total_dirty);
    memset(ctx->dirty_ranges, 0, ಪರಿಮಾಣ(ctx->dirty_ranges));
}

ಶೂನ್ಯ htp_dcache_flush_dirty_tensors(ಸಂರಚನಾ htp_context * ctx, ಧ್ರುವ ಸಂರಚನಾ htp_tensor * ಧ್ರುವ * tensors, uint32_t n) {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    ಸಂರಚನಾ htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t total_dirty = 0;

    ಪ್ರತಿ (uint32_t i = 0; i < n; i++) {
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t = tensors[i];
        ಯದಿ (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->data;
            ranges[n_dirty - 1].end   = t->data + t->size;
            total_dirty += t->size;
        }
    }

    ಯದಿ (total_dirty == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        ಪ್ರತ್ಯಾಗಮ;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, total_dirty);
    ಪ್ರತಿ (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
