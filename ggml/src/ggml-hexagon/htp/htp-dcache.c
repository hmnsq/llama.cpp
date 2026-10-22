#অন্তর্ভাব "htp-dcache.h"

#অন্তর্ভাব <qurt.h>
#অন্তর্ভাব <qurt_memory.h>
#অন্তর্ভাব <HAP_farf.h>

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-utils.h"
#অন্তর্ভাব "hex-fastdiv.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "work-queue.h"

সংরচনা l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

সংরচনা l2flush_multi_task {
    সংরচনা htp_thread_trace * trace;
    সংরচনা l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  total_blocks;
    uint32_t                  blocks_per_thread;
};

স্থির শূন্য flush_all_dcache(সংরচনা htp_context * ctx) {
    সংরচনা htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, পরিমাণ(ctx->dirty_ranges));
}

স্থির শূন্য l2flush_multi_worker(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য * data) {
    সংরচনা l2flush_multi_task * task = (সংরচনা l2flush_multi_task *) data;
    (শূন্য) n;

    ধ্রুব uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    যদি (gb_last > task->total_blocks) {
        gb_last = task->total_blocks;
    }
    যদি (gb_first >= gb_last) {
        প্রত্যাগম;
    }

    সংরচনা htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    প্রতি (uint32_t r = 0; r < task->n_ranges; r++) {
        ধ্রুব সংরচনা l2flush_range * rg = &task->ranges[r];
        ধ্রুব uint32_t rb_first = rg->block_first;
        ধ্রুব uint32_t rb_last  = rg->block_first + rg->n_blocks;

        ধ্রুব uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        ধ্রুব uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        যদি (lo >= hi) {
            অনুবর্ত;
        }

        ধ্রুব uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_SIZE;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_SIZE;
        যদি (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((শূন্য *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

স্থির শূন্য merge_dirty_ranges(সংরচনা htp_context * ctx) {
    প্রতি (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[i];
        যদি (!r->start) অনুবর্ত;

        প্রতি (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            সংরচনা htp_dirty_range * s = &ctx->dirty_ranges[j];
            যদি (i == j || !s->start || r->end < s->start || s->end < r->start) {
                j++;
                অনুবর্ত;
            }

            r->start = MIN(r->start, s->start);
            r->end   = MAX(r->end, s->end);
            s->start = 0;
            s->end   = 0;
            j = 0;
        }
    }
}

শূন্য htp_dcache_mark_dirty_tensors(সংরচনা htp_context * ctx, ধ্রুব সংরচনা htp_tensor * ধ্রুব * tensors, uint32_t n) {
    ধ্রুব সংরচনা htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    প্রতি (uint32_t i = 0; i < n; i++) {
        ধ্রুব সংরচনা htp_tensor * t = tensors[i];
        যদি (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            অনুবর্ত;
        }

        uint32_t t_start = t->data;
        uint32_t t_end   = t_start + t->size;

        bool merged = false;
        প্রতি (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[j];
            যদি (!r->start) অনুবর্ত;

            যদি (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end)     ? t_end   : r->end;
                r->start = new_start;
                r->end   = new_end;
                merged   = true;
            }
        }

        যদি (!merged) {
            pending[n_pending++] = t;
        }
    }

    merge_dirty_ranges(ctx);

    যদি (n_pending == 0) {
        প্রত্যাগম;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    প্রতি (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        যদি (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } অন্যথা {
            empty_indices[n_empty++]   = j;
        }
    }

    যদি (n_pending <= n_empty) {
        প্রতি (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        প্রত্যাগম;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t total_evict_size = 0;
    প্রতি (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[idx];
        total_evict_size += r->end - r->start;
    }

    যদি (total_evict_size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        প্রতি (uint32_t i = 0; i < n_pending; i++) {
            সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        প্রত্যাগম;
    }

    যদি (total_evict_size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        সংরচনা l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        প্রতি (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[idx];

            সংরচনা l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_সারি_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_সারি_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } অন্যথা {
        সংরচনা htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        প্রতি (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t size = r->end - r->start;
            hex_l2flush((শূন্য *) (uintptr_t) r->start, size);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    প্রতি (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->data;
        r->end   = pending[i]->data + pending[i]->size;
    }

    প্রতি (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->data;
        r->end   = pending[n_evict + i]->data + pending[n_evict + i]->size;
    }

    merge_dirty_ranges(ctx);
}

স্থির শূন্য make_tensor_clean(সংরচনা htp_context * ctx, ধ্রুব সংরচনা htp_tensor * t) {
    যদি (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        প্রত্যাগম;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    প্রতি (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[i];
        যদি (!r->start) অনুবর্ত;

        যদি (r->start < t_end && t_start < r->end) {
            যদি (t_start <= r->start && r->end <= t_end) {
                r->start = 0;
                r->end   = 0;
            } অন্যথা যদি (t_start <= r->start) {
                r->start = t_end;
            } অন্যথা যদি (r->end <= t_end) {
                r->end = t_start;
            }
        }
    }
}

স্থির অন্তঃস্থ bool is_tensor_dirty(সংরচনা htp_context * ctx, ধ্রুব সংরচনা htp_tensor * t) {
    যদি (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        প্রত্যাগম false;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    প্রতি (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[i];
        যদি (!r->start) অনুবর্ত;

        যদি (r->start < t_end && t_start < r->end) {
            প্রত্যাগম true;
        }
    }
    প্রত্যাগম false;
}

স্থির শূন্য flush_dirty_ranges(সংরচনা htp_context * ctx, ধ্রুব সংরচনা htp_dirty_range * ranges, uint32_t n_ranges, uint64_t total_dirty) {
    যদি (total_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        সংরচনা l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        প্রতি (uint32_t i = 0; i < n_ranges; i++) {
            ধ্রুব সংরচনা htp_dirty_range * r = &ranges[i];
            সংরচনা l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_সারি_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_সারি_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } অন্যথা {
        সংরচনা htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        প্রতি (uint32_t i = 0; i < n_ranges; i++) {
            ধ্রুব সংরচনা htp_dirty_range * r = &ranges[i];
            hex_l2flush((শূন্য *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

শূন্য htp_dcache_flush_dirty_ranges(সংরচনা htp_context * ctx) {
    সংরচনা htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t total_dirty = 0;

    প্রতি (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ধ্রুব সংরচনা htp_dirty_range * r = &ctx->dirty_ranges[i];
        যদি (!r->start) {
            অনুবর্ত;
        }
        ranges[n_ranges++] = *r;
        total_dirty += r->end - r->start;
    }

    যদি (total_dirty == 0) {
        প্রত্যাগম;
    }

    যদি (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        প্রত্যাগম;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, total_dirty);
    memset(ctx->dirty_ranges, 0, পরিমাণ(ctx->dirty_ranges));
}

শূন্য htp_dcache_flush_dirty_tensors(সংরচনা htp_context * ctx, ধ্রুব সংরচনা htp_tensor * ধ্রুব * tensors, uint32_t n) {
    ধ্রুব সংরচনা htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    সংরচনা htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t total_dirty = 0;

    প্রতি (uint32_t i = 0; i < n; i++) {
        ধ্রুব সংরচনা htp_tensor * t = tensors[i];
        যদি (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->data;
            ranges[n_dirty - 1].end   = t->data + t->size;
            total_dirty += t->size;
        }
    }

    যদি (total_dirty == 0) {
        প্রত্যাগম;
    }

    যদি (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        প্রত্যাগম;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, total_dirty);
    প্রতি (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
