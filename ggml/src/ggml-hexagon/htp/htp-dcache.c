#समावेश "htp-dcache.h"

#समावेश <qurt.h>
#समावेश <qurt_memory.h>
#समावेश <HAP_farf.h>

#समावेश "hex-common.h"
#समावेश "hex-utils.h"
#समावेश "hex-fastdiv.h"
#समावेश "hex-profile.h"
#समावेश "htp-ctx.h"
#समावेश "work-queue.h"

काष्ठा l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

काष्ठा l2flush_multi_task {
    काष्ठा htp_thread_trace * trace;
    काष्ठा l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  योग_blocks;
    uint32_t                  blocks_per_thread;
};

अटल व्योम flush_all_dcache(काष्ठा htp_context * ctx) {
    काष्ठा htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    स_रखो(ctx->dirty_ranges, 0, माप(ctx->dirty_ranges));
}

अटल व्योम l2flush_multi_worker(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम * तथ्य) {
    काष्ठा l2flush_multi_task * task = (काष्ठा l2flush_multi_task *) तथ्य;
    (व्योम) n;

    स्थिर uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    अगर (gb_last > task->योग_blocks) {
        gb_last = task->योग_blocks;
    }
    अगर (gb_first >= gb_last) {
        वापस;
    }

    काष्ठा htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    क्रम (uint32_t r = 0; r < task->n_ranges; r++) {
        स्थिर काष्ठा l2flush_range * rg = &task->ranges[r];
        स्थिर uint32_t rb_first = rg->block_first;
        स्थिर uint32_t rb_last  = rg->block_first + rg->n_blocks;

        स्थिर uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        स्थिर uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        अगर (lo >= hi) {
            जारी;
        }

        स्थिर uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_आकार;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_आकार;
        अगर (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((व्योम *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

अटल व्योम merge_dirty_ranges(काष्ठा htp_context * ctx) {
    क्रम (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[i];
        अगर (!r->start) जारी;

        क्रम (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            काष्ठा htp_dirty_range * s = &ctx->dirty_ranges[j];
            अगर (i == j || !s->start || r->end < s->start || s->end < r->start) {
                j++;
                जारी;
            }

            r->start = MIN(r->start, s->start);
            r->end   = MAX(r->end, s->end);
            s->start = 0;
            s->end   = 0;
            j = 0;
        }
    }
}

व्योम htp_dcache_mark_dirty_tensors(काष्ठा htp_context * ctx, स्थिर काष्ठा htp_tensor * स्थिर * tensors, uint32_t n) {
    स्थिर काष्ठा htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    क्रम (uint32_t i = 0; i < n; i++) {
        स्थिर काष्ठा htp_tensor * t = tensors[i];
        अगर (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            जारी;
        }

        uint32_t t_start = t->तथ्य;
        uint32_t t_end   = t_start + t->आकार;

        bool merged = false;
        क्रम (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[j];
            अगर (!r->start) जारी;

            अगर (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end)     ? t_end   : r->end;
                r->start = new_start;
                r->end   = new_end;
                merged   = true;
            }
        }

        अगर (!merged) {
            pending[n_pending++] = t;
        }
    }

    merge_dirty_ranges(ctx);

    अगर (n_pending == 0) {
        वापस;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    क्रम (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        अगर (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } अन्यथा {
            empty_indices[n_empty++]   = j;
        }
    }

    अगर (n_pending <= n_empty) {
        क्रम (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->तथ्य;
            r->end   = pending[i]->तथ्य + pending[i]->आकार;
        }
        merge_dirty_ranges(ctx);
        वापस;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t योग_evict_आकार = 0;
    क्रम (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[idx];
        योग_evict_आकार += r->end - r->start;
    }

    अगर (योग_evict_आकार > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        क्रम (uint32_t i = 0; i < n_pending; i++) {
            काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->तथ्य;
            r->end   = pending[i]->तथ्य + pending[i]->आकार;
        }
        merge_dirty_ranges(ctx);
        वापस;
    }

    अगर (योग_evict_आकार > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        काष्ठा l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        क्रम (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[idx];

            काष्ठा l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((माप_प्रकार) r->start, HEX_L2_पंक्ति_आकार);
            rg->end   = hex_align_up((माप_प्रकार) r->end, HEX_L2_पंक्ति_आकार);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_आकार - 1) / HEX_L2_BLOCK_आकार;
            block_acc += rg->n_blocks;
        }

        task.योग_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } अन्यथा {
        काष्ठा htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        क्रम (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t आकार = r->end - r->start;
            hex_l2flush((व्योम *) (uintptr_t) r->start, आकार);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    क्रम (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->तथ्य;
        r->end   = pending[i]->तथ्य + pending[i]->आकार;
    }

    क्रम (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->तथ्य;
        r->end   = pending[n_evict + i]->तथ्य + pending[n_evict + i]->आकार;
    }

    merge_dirty_ranges(ctx);
}

अटल व्योम make_tensor_clean(काष्ठा htp_context * ctx, स्थिर काष्ठा htp_tensor * t) {
    अगर (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        वापस;
    }

    uint32_t t_start = t->तथ्य;
    uint32_t t_end   = t_start + t->आकार;

    क्रम (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[i];
        अगर (!r->start) जारी;

        अगर (r->start < t_end && t_start < r->end) {
            अगर (t_start <= r->start && r->end <= t_end) {
                r->start = 0;
                r->end   = 0;
            } अन्यथा अगर (t_start <= r->start) {
                r->start = t_end;
            } अन्यथा अगर (r->end <= t_end) {
                r->end = t_start;
            }
        }
    }
}

अटल अंतरभूत bool is_tensor_dirty(काष्ठा htp_context * ctx, स्थिर काष्ठा htp_tensor * t) {
    अगर (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        वापस false;
    }

    uint32_t t_start = t->तथ्य;
    uint32_t t_end   = t_start + t->आकार;

    क्रम (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[i];
        अगर (!r->start) जारी;

        अगर (r->start < t_end && t_start < r->end) {
            वापस true;
        }
    }
    वापस false;
}

अटल व्योम flush_dirty_ranges(काष्ठा htp_context * ctx, स्थिर काष्ठा htp_dirty_range * ranges, uint32_t n_ranges, uint64_t योग_dirty) {
    अगर (योग_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        काष्ठा l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        क्रम (uint32_t i = 0; i < n_ranges; i++) {
            स्थिर काष्ठा htp_dirty_range * r = &ranges[i];
            काष्ठा l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((माप_प्रकार) r->start, HEX_L2_पंक्ति_आकार);
            rg->end   = hex_align_up((माप_प्रकार) r->end, HEX_L2_पंक्ति_आकार);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_आकार - 1) / HEX_L2_BLOCK_आकार;
            block_acc += rg->n_blocks;
        }

        task.योग_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } अन्यथा {
        काष्ठा htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        क्रम (uint32_t i = 0; i < n_ranges; i++) {
            स्थिर काष्ठा htp_dirty_range * r = &ranges[i];
            hex_l2flush((व्योम *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

व्योम htp_dcache_flush_dirty_ranges(काष्ठा htp_context * ctx) {
    काष्ठा htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t योग_dirty = 0;

    क्रम (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        स्थिर काष्ठा htp_dirty_range * r = &ctx->dirty_ranges[i];
        अगर (!r->start) {
            जारी;
        }
        ranges[n_ranges++] = *r;
        योग_dirty += r->end - r->start;
    }

    अगर (योग_dirty == 0) {
        वापस;
    }

    अगर (योग_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        वापस;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, योग_dirty);
    स_रखो(ctx->dirty_ranges, 0, माप(ctx->dirty_ranges));
}

व्योम htp_dcache_flush_dirty_tensors(काष्ठा htp_context * ctx, स्थिर काष्ठा htp_tensor * स्थिर * tensors, uint32_t n) {
    स्थिर काष्ठा htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    काष्ठा htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t योग_dirty = 0;

    क्रम (uint32_t i = 0; i < n; i++) {
        स्थिर काष्ठा htp_tensor * t = tensors[i];
        अगर (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->तथ्य;
            ranges[n_dirty - 1].end   = t->तथ्य + t->आकार;
            योग_dirty += t->आकार;
        }
    }

    अगर (योग_dirty == 0) {
        वापस;
    }

    अगर (योग_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        वापस;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, योग_dirty);
    क्रम (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
