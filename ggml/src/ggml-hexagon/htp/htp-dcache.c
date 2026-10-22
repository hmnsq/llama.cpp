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
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

بنية l2flush_multi_task {
    بنية htp_thread_trace * trace;
    بنية l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  total_blocks;
    uint32_t                  blocks_per_thread;
};

ساكن فراغ flush_all_dcache(بنية htp_context * ctx) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

ساكن فراغ l2flush_multi_worker(unsigned صحيح n, unsigned صحيح i, فراغ * data) {
    بنية l2flush_multi_task * task = (بنية l2flush_multi_task *) data;
    (فراغ) n;

    ثابت uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    𐤀𐤌 (gb_last > task->total_blocks) {
        gb_last = task->total_blocks;
    }
    𐤀𐤌 (gb_first >= gb_last) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    لكل (uint32_t r = 0; r < task->n_ranges; r++) {
        ثابت بنية l2flush_range * rg = &task->ranges[r];
        ثابت uint32_t rb_first = rg->block_first;
        ثابت uint32_t rb_last  = rg->block_first + rg->n_blocks;

        ثابت uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        ثابت uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        𐤀𐤌 (lo >= hi) {
            استمر;
        }

        ثابت uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_SIZE;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_SIZE;
        𐤀𐤌 (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((فراغ *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

ساكن فراغ merge_dirty_ranges(بنية htp_context * ctx) {
    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        𐤀𐤌 (!r->start) استمر;

        لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            بنية htp_dirty_range * s = &ctx->dirty_ranges[j];
            𐤀𐤌 (i == j || !s->start || r->end < s->start || s->end < r->start) {
                j++;
                استمر;
            }

            r->start = MIN(r->start, s->start);
            r->end   = MAX(r->end, s->end);
            s->start = 0;
            s->end   = 0;
            j = 0;
        }
    }
}

فراغ htp_dcache_mark_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n) {
    ثابت بنية htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    لكل (uint32_t i = 0; i < n; i++) {
        ثابت بنية htp_tensor * t = tensors[i];
        𐤀𐤌 (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            استمر;
        }

        uint32_t t_start = t->data;
        uint32_t t_end   = t_start + t->size;

        bool merged = false;
        لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            بنية htp_dirty_range * r = &ctx->dirty_ranges[j];
            𐤀𐤌 (!r->start) استمر;

            𐤀𐤌 (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end)     ? t_end   : r->end;
                r->start = new_start;
                r->end   = new_end;
                merged   = true;
            }
        }

        𐤀𐤌 (!merged) {
            pending[n_pending++] = t;
        }
    }

    merge_dirty_ranges(ctx);

    𐤀𐤌 (n_pending == 0) {
        أرجع;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    لكل (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        𐤀𐤌 (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } وإلا {
            empty_indices[n_empty++]   = j;
        }
    }

    𐤀𐤌 (n_pending <= n_empty) {
        لكل (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        أرجع;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t total_evict_size = 0;
    لكل (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        total_evict_size += r->end - r->start;
    }

    𐤀𐤌 (total_evict_size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        لكل (uint32_t i = 0; i < n_pending; i++) {
            بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        أرجع;
    }

    𐤀𐤌 (total_evict_size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        بنية l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        لكل (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];

            بنية l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_LINE_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_LINE_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } وإلا {
        بنية htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        لكل (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t size = r->end - r->start;
            hex_l2flush((فراغ *) (uintptr_t) r->start, size);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    لكل (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->data;
        r->end   = pending[i]->data + pending[i]->size;
    }

    لكل (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        بنية htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->data;
        r->end   = pending[n_evict + i]->data + pending[n_evict + i]->size;
    }

    merge_dirty_ranges(ctx);
}

ساكن فراغ make_tensor_clean(بنية htp_context * ctx, ثابت بنية htp_tensor * t) {
    𐤀𐤌 (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        أرجع;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        𐤀𐤌 (!r->start) استمر;

        𐤀𐤌 (r->start < t_end && t_start < r->end) {
            𐤀𐤌 (t_start <= r->start && r->end <= t_end) {
                r->start = 0;
                r->end   = 0;
            } وإلا 𐤀𐤌 (t_start <= r->start) {
                r->start = t_end;
            } وإلا 𐤀𐤌 (r->end <= t_end) {
                r->end = t_start;
            }
        }
    }
}

ساكن مضمن bool is_tensor_dirty(بنية htp_context * ctx, ثابت بنية htp_tensor * t) {
    𐤀𐤌 (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        أرجع false;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        𐤀𐤌 (!r->start) استمر;

        𐤀𐤌 (r->start < t_end && t_start < r->end) {
            أرجع true;
        }
    }
    أرجع false;
}

ساكن فراغ flush_dirty_ranges(بنية htp_context * ctx, ثابت بنية htp_dirty_range * ranges, uint32_t n_ranges, uint64_t total_dirty) {
    𐤀𐤌 (total_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        بنية l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        لكل (uint32_t i = 0; i < n_ranges; i++) {
            ثابت بنية htp_dirty_range * r = &ranges[i];
            بنية l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_LINE_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_LINE_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } وإلا {
        بنية htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        لكل (uint32_t i = 0; i < n_ranges; i++) {
            ثابت بنية htp_dirty_range * r = &ranges[i];
            hex_l2flush((فراغ *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

فراغ htp_dcache_flush_dirty_ranges(بنية htp_context * ctx) {
    بنية htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t total_dirty = 0;

    لكل (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        ثابت بنية htp_dirty_range * r = &ctx->dirty_ranges[i];
        𐤀𐤌 (!r->start) {
            استمر;
        }
        ranges[n_ranges++] = *r;
        total_dirty += r->end - r->start;
    }

    𐤀𐤌 (total_dirty == 0) {
        أرجع;
    }

    𐤀𐤌 (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        أرجع;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, total_dirty);
    memset(ctx->dirty_ranges, 0, حجم(ctx->dirty_ranges));
}

فراغ htp_dcache_flush_dirty_tensors(بنية htp_context * ctx, ثابت بنية htp_tensor * ثابت * tensors, uint32_t n) {
    ثابت بنية htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    بنية htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t total_dirty = 0;

    لكل (uint32_t i = 0; i < n; i++) {
        ثابت بنية htp_tensor * t = tensors[i];
        𐤀𐤌 (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->data;
            ranges[n_dirty - 1].end   = t->data + t->size;
            total_dirty += t->size;
        }
    }

    𐤀𐤌 (total_dirty == 0) {
        أرجع;
    }

    𐤀𐤌 (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        أرجع;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, total_dirty);
    لكل (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
