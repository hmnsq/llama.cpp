#અન્તર્ભાવ "htp-dcache.h"

#અન્તર્ભાવ <qurt.h>
#અન્તર્ભાવ <qurt_memory.h>
#અન્તર્ભાવ <HAP_farf.h>

#અન્તર્ભાવ "hex-common.h"
#અન્તર્ભાવ "hex-utils.h"
#અન્તર્ભાવ "hex-fastdiv.h"
#અન્તર્ભાવ "hex-profile.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "work-queue.h"

સંરચના l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

સંરચના l2flush_multi_task {
    સંરચના htp_thread_trace * trace;
    સંરચના l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  total_blocks;
    uint32_t                  blocks_per_thread;
};

સ્થિર શૂન્ય flush_all_dcache(સંરચના htp_context * ctx) {
    સંરચના htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, કદ(ctx->dirty_ranges));
}

સ્થિર શૂન્ય l2flush_multi_worker(unsigned પૂર્ણાંક n, unsigned પૂર્ણાંક i, શૂન્ય * data) {
    સંરચના l2flush_multi_task * task = (સંરચના l2flush_multi_task *) data;
    (શૂન્ય) n;

    અચલ uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    જો (gb_last > task->total_blocks) {
        gb_last = task->total_blocks;
    }
    જો (gb_first >= gb_last) {
        પાછા;
    }

    સંરચના htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    માટે (uint32_t r = 0; r < task->n_ranges; r++) {
        અચલ સંરચના l2flush_range * rg = &task->ranges[r];
        અચલ uint32_t rb_first = rg->block_first;
        અચલ uint32_t rb_last  = rg->block_first + rg->n_blocks;

        અચલ uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        અચલ uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        જો (lo >= hi) {
            ચાલુ;
        }

        અચલ uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_SIZE;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_SIZE;
        જો (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((શૂન્ય *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

સ્થિર શૂન્ય merge_dirty_ranges(સંરચના htp_context * ctx) {
    માટે (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        સંરચના htp_dirty_range * r = &ctx->dirty_ranges[i];
        જો (!r->start) ચાલુ;

        માટે (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            સંરચના htp_dirty_range * s = &ctx->dirty_ranges[j];
            જો (i == j || !s->start || r->end < s->start || s->end < r->start) {
                j++;
                ચાલુ;
            }

            r->start = MIN(r->start, s->start);
            r->end   = MAX(r->end, s->end);
            s->start = 0;
            s->end   = 0;
            j = 0;
        }
    }
}

શૂન્ય htp_dcache_mark_dirty_tensors(સંરચના htp_context * ctx, અચલ સંરચના htp_tensor * અચલ * tensors, uint32_t n) {
    અચલ સંરચના htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    માટે (uint32_t i = 0; i < n; i++) {
        અચલ સંરચના htp_tensor * t = tensors[i];
        જો (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            ચાલુ;
        }

        uint32_t t_start = t->data;
        uint32_t t_end   = t_start + t->size;

        bool merged = false;
        માટે (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            સંરચના htp_dirty_range * r = &ctx->dirty_ranges[j];
            જો (!r->start) ચાલુ;

            જો (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end)     ? t_end   : r->end;
                r->start = new_start;
                r->end   = new_end;
                merged   = true;
            }
        }

        જો (!merged) {
            pending[n_pending++] = t;
        }
    }

    merge_dirty_ranges(ctx);

    જો (n_pending == 0) {
        પાછા;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    માટે (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        જો (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } નહીંતર {
            empty_indices[n_empty++]   = j;
        }
    }

    જો (n_pending <= n_empty) {
        માટે (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            સંરચના htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        પાછા;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t total_evict_size = 0;
    માટે (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        સંરચના htp_dirty_range * r = &ctx->dirty_ranges[idx];
        total_evict_size += r->end - r->start;
    }

    જો (total_evict_size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        માટે (uint32_t i = 0; i < n_pending; i++) {
            સંરચના htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        પાછા;
    }

    જો (total_evict_size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        સંરચના l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        માટે (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            સંરચના htp_dirty_range * r = &ctx->dirty_ranges[idx];

            સંરચના l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_લીટી_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_લીટી_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } નહીંતર {
        સંરચના htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        માટે (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            સંરચના htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t size = r->end - r->start;
            hex_l2flush((શૂન્ય *) (uintptr_t) r->start, size);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    માટે (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        સંરચના htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->data;
        r->end   = pending[i]->data + pending[i]->size;
    }

    માટે (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        સંરચના htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->data;
        r->end   = pending[n_evict + i]->data + pending[n_evict + i]->size;
    }

    merge_dirty_ranges(ctx);
}

સ્થિર શૂન્ય make_tensor_clean(સંરચના htp_context * ctx, અચલ સંરચના htp_tensor * t) {
    જો (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        પાછા;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    માટે (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        સંરચના htp_dirty_range * r = &ctx->dirty_ranges[i];
        જો (!r->start) ચાલુ;

        જો (r->start < t_end && t_start < r->end) {
            જો (t_start <= r->start && r->end <= t_end) {
                r->start = 0;
                r->end   = 0;
            } નહીંતર જો (t_start <= r->start) {
                r->start = t_end;
            } નહીંતર જો (r->end <= t_end) {
                r->end = t_start;
            }
        }
    }
}

સ્થિર અંતઃરેખ bool is_tensor_dirty(સંરચના htp_context * ctx, અચલ સંરચના htp_tensor * t) {
    જો (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        પાછા false;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    માટે (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        સંરચના htp_dirty_range * r = &ctx->dirty_ranges[i];
        જો (!r->start) ચાલુ;

        જો (r->start < t_end && t_start < r->end) {
            પાછા true;
        }
    }
    પાછા false;
}

સ્થિર શૂન્ય flush_dirty_ranges(સંરચના htp_context * ctx, અચલ સંરચના htp_dirty_range * ranges, uint32_t n_ranges, uint64_t total_dirty) {
    જો (total_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        સંરચના l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        માટે (uint32_t i = 0; i < n_ranges; i++) {
            અચલ સંરચના htp_dirty_range * r = &ranges[i];
            સંરચના l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_લીટી_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_લીટી_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } નહીંતર {
        સંરચના htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        માટે (uint32_t i = 0; i < n_ranges; i++) {
            અચલ સંરચના htp_dirty_range * r = &ranges[i];
            hex_l2flush((શૂન્ય *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

શૂન્ય htp_dcache_flush_dirty_ranges(સંરચના htp_context * ctx) {
    સંરચના htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t total_dirty = 0;

    માટે (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        અચલ સંરચના htp_dirty_range * r = &ctx->dirty_ranges[i];
        જો (!r->start) {
            ચાલુ;
        }
        ranges[n_ranges++] = *r;
        total_dirty += r->end - r->start;
    }

    જો (total_dirty == 0) {
        પાછા;
    }

    જો (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        પાછા;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, total_dirty);
    memset(ctx->dirty_ranges, 0, કદ(ctx->dirty_ranges));
}

શૂન્ય htp_dcache_flush_dirty_tensors(સંરચના htp_context * ctx, અચલ સંરચના htp_tensor * અચલ * tensors, uint32_t n) {
    અચલ સંરચના htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    સંરચના htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t total_dirty = 0;

    માટે (uint32_t i = 0; i < n; i++) {
        અચલ સંરચના htp_tensor * t = tensors[i];
        જો (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->data;
            ranges[n_dirty - 1].end   = t->data + t->size;
            total_dirty += t->size;
        }
    }

    જો (total_dirty == 0) {
        પાછા;
    }

    જો (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        પાછા;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, total_dirty);
    માટે (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
