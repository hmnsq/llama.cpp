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

কাঠামো l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

কাঠামো l2flush_multi_task {
    কাঠামো htp_thread_trace * trace;
    কাঠামো l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  যোগ_blocks;
    uint32_t                  blocks_per_thread;
};

স্থির শূন্য flush_all_dcache(কাঠামো htp_context * ctx) {
    কাঠামো htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, আকার(ctx->dirty_ranges));
}

স্থির শূন্য l2flush_multi_worker(unsigned পূর্ণসংখ্যা n, unsigned পূর্ণসংখ্যা i, শূন্য * তথ্য) {
    কাঠামো l2flush_multi_task * task = (কাঠামো l2flush_multi_task *) তথ্য;
    (শূন্য) n;

    ধ্রুবক uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    যদি (gb_last > task->যোগ_blocks) {
        gb_last = task->যোগ_blocks;
    }
    যদি (gb_first >= gb_last) {
        ফেরত;
    }

    কাঠামো htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    জন্য (uint32_t r = 0; r < task->n_ranges; r++) {
        ধ্রুবক কাঠামো l2flush_range * rg = &task->ranges[r];
        ধ্রুবক uint32_t rb_first = rg->block_first;
        ধ্রুবক uint32_t rb_last  = rg->block_first + rg->n_blocks;

        ধ্রুবক uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        ধ্রুবক uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        যদি (lo >= hi) {
            চালু;
        }

        ধ্রুবক uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_আকার;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_আকার;
        যদি (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((শূন্য *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

স্থির শূন্য merge_dirty_ranges(কাঠামো htp_context * ctx) {
    জন্য (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[i];
        যদি (!r->start) চালু;

        জন্য (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            কাঠামো htp_dirty_range * s = &ctx->dirty_ranges[j];
            যদি (i == j || !s->start || r->end < s->start || s->end < r->start) {
                j++;
                চালু;
            }

            r->start = MIN(r->start, s->start);
            r->end   = MAX(r->end, s->end);
            s->start = 0;
            s->end   = 0;
            j = 0;
        }
    }
}

শূন্য htp_dcache_mark_dirty_tensors(কাঠামো htp_context * ctx, ধ্রুবক কাঠামো htp_tensor * ধ্রুবক * tensors, uint32_t n) {
    ধ্রুবক কাঠামো htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    জন্য (uint32_t i = 0; i < n; i++) {
        ধ্রুবক কাঠামো htp_tensor * t = tensors[i];
        যদি (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            চালু;
        }

        uint32_t t_start = t->তথ্য;
        uint32_t t_end   = t_start + t->আকার_2;

        bool merged = false;
        জন্য (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[j];
            যদি (!r->start) চালু;

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
        ফেরত;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    জন্য (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        যদি (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } অন্যথা {
            empty_indices[n_empty++]   = j;
        }
    }

    যদি (n_pending <= n_empty) {
        জন্য (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->তথ্য;
            r->end   = pending[i]->তথ্য + pending[i]->আকার_2;
        }
        merge_dirty_ranges(ctx);
        ফেরত;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t যোগ_evict_আকার = 0;
    জন্য (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[idx];
        যোগ_evict_আকার += r->end - r->start;
    }

    যদি (যোগ_evict_আকার > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        জন্য (uint32_t i = 0; i < n_pending; i++) {
            কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->তথ্য;
            r->end   = pending[i]->তথ্য + pending[i]->আকার_2;
        }
        merge_dirty_ranges(ctx);
        ফেরত;
    }

    যদি (যোগ_evict_আকার > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        কাঠামো l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        জন্য (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[idx];

            কাঠামো l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_সারি_আকার);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_সারি_আকার);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_আকার - 1) / HEX_L2_BLOCK_আকার;
            block_acc += rg->n_blocks;
        }

        task.যোগ_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } অন্যথা {
        কাঠামো htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        জন্য (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t আকার_2 = r->end - r->start;
            hex_l2flush((শূন্য *) (uintptr_t) r->start, আকার_2);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    জন্য (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->তথ্য;
        r->end   = pending[i]->তথ্য + pending[i]->আকার_2;
    }

    জন্য (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->তথ্য;
        r->end   = pending[n_evict + i]->তথ্য + pending[n_evict + i]->আকার_2;
    }

    merge_dirty_ranges(ctx);
}

স্থির শূন্য make_tensor_clean(কাঠামো htp_context * ctx, ধ্রুবক কাঠামো htp_tensor * t) {
    যদি (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        ফেরত;
    }

    uint32_t t_start = t->তথ্য;
    uint32_t t_end   = t_start + t->আকার_2;

    জন্য (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[i];
        যদি (!r->start) চালু;

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

স্থির অন্তঃরেখ bool is_tensor_dirty(কাঠামো htp_context * ctx, ধ্রুবক কাঠামো htp_tensor * t) {
    যদি (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        ফেরত false;
    }

    uint32_t t_start = t->তথ্য;
    uint32_t t_end   = t_start + t->আকার_2;

    জন্য (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[i];
        যদি (!r->start) চালু;

        যদি (r->start < t_end && t_start < r->end) {
            ফেরত true;
        }
    }
    ফেরত false;
}

স্থির শূন্য flush_dirty_ranges(কাঠামো htp_context * ctx, ধ্রুবক কাঠামো htp_dirty_range * ranges, uint32_t n_ranges, uint64_t যোগ_dirty) {
    যদি (যোগ_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        কাঠামো l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        জন্য (uint32_t i = 0; i < n_ranges; i++) {
            ধ্রুবক কাঠামো htp_dirty_range * r = &ranges[i];
            কাঠামো l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_সারি_আকার);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_সারি_আকার);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_আকার - 1) / HEX_L2_BLOCK_আকার;
            block_acc += rg->n_blocks;
        }

        task.যোগ_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } অন্যথা {
        কাঠামো htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        জন্য (uint32_t i = 0; i < n_ranges; i++) {
            ধ্রুবক কাঠামো htp_dirty_range * r = &ranges[i];
            hex_l2flush((শূন্য *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

শূন্য htp_dcache_flush_dirty_ranges(কাঠামো htp_context * ctx) {
    কাঠামো htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t যোগ_dirty = 0;

    জন্য (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ধ্রুবক কাঠামো htp_dirty_range * r = &ctx->dirty_ranges[i];
        যদি (!r->start) {
            চালু;
        }
        ranges[n_ranges++] = *r;
        যোগ_dirty += r->end - r->start;
    }

    যদি (যোগ_dirty == 0) {
        ফেরত;
    }

    যদি (যোগ_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        ফেরত;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, যোগ_dirty);
    memset(ctx->dirty_ranges, 0, আকার(ctx->dirty_ranges));
}

শূন্য htp_dcache_flush_dirty_tensors(কাঠামো htp_context * ctx, ধ্রুবক কাঠামো htp_tensor * ধ্রুবক * tensors, uint32_t n) {
    ধ্রুবক কাঠামো htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    কাঠামো htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t যোগ_dirty = 0;

    জন্য (uint32_t i = 0; i < n; i++) {
        ধ্রুবক কাঠামো htp_tensor * t = tensors[i];
        যদি (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->তথ্য;
            ranges[n_dirty - 1].end   = t->তথ্য + t->আকার_2;
            যোগ_dirty += t->আকার_2;
        }
    }

    যদি (যোগ_dirty == 0) {
        ফেরত;
    }

    যদি (যোগ_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        ফেরত;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, যোগ_dirty);
    জন্য (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
