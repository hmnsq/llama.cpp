#अन्तर्भाव "htp-dcache.h"

#अन्तर्भाव <qurt.h>
#अन्तर्भाव <qurt_memory.h>
#अन्तर्भाव <HAP_farf.h>

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "work-queue.h"

संरचना l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

संरचना l2flush_multi_task {
    संरचना htp_thread_trace * trace;
    संरचना l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  total_blocks;
    uint32_t                  blocks_per_thread;
};

स्थिर शून्य flush_all_dcache(संरचना htp_context * ctx) {
    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    स_रखो(ctx->dirty_ranges, 0, परिमाण(ctx->dirty_ranges));
}

स्थिर शून्य l2flush_multi_worker(अचिन्हित पूर्णाङ्क n, अचिन्हित पूर्णाङ्क i, शून्य * data) {
    संरचना l2flush_multi_task * task = (संरचना l2flush_multi_task *) data;
    (शून्य) n;

    अचर uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    यदि (gb_last > task->total_blocks) {
        gb_last = task->total_blocks;
    }
    यदि (gb_first >= gb_last) {
        फर्काऊ;
    }

    संरचना htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    लागि (uint32_t r = 0; r < task->n_ranges; r++) {
        अचर संरचना l2flush_range * rg = &task->ranges[r];
        अचर uint32_t rb_first = rg->block_first;
        अचर uint32_t rb_last  = rg->block_first + rg->n_blocks;

        अचर uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        अचर uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        यदि (lo >= hi) {
            जारी;
        }

        अचर uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_SIZE;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_SIZE;
        यदि (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((शून्य *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

स्थिर शून्य merge_dirty_ranges(संरचना htp_context * ctx) {
    लागि (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[i];
        यदि (!r->start) जारी;

        लागि (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES;) {
            संरचना htp_dirty_range * s = &ctx->dirty_ranges[j];
            यदि (i == j || !s->start || r->end < s->start || s->end < r->start) {
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

शून्य htp_dcache_mark_dirty_tensors(संरचना htp_context * ctx, अचर संरचना htp_tensor * अचर * tensors, uint32_t n) {
    अचर संरचना htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    लागि (uint32_t i = 0; i < n; i++) {
        अचर संरचना htp_tensor * t = tensors[i];
        यदि (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            जारी;
        }

        uint32_t t_start = t->data;
        uint32_t t_end   = t_start + t->size;

        bool merged = false;
        लागि (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            संरचना htp_dirty_range * r = &ctx->dirty_ranges[j];
            यदि (!r->start) जारी;

            यदि (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end)     ? t_end   : r->end;
                r->start = new_start;
                r->end   = new_end;
                merged   = true;
            }
        }

        यदि (!merged) {
            pending[n_pending++] = t;
        }
    }

    merge_dirty_ranges(ctx);

    यदि (n_pending == 0) {
        फर्काऊ;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    लागि (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        यदि (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } अन्यथा {
            empty_indices[n_empty++]   = j;
        }
    }

    यदि (n_pending <= n_empty) {
        लागि (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        फर्काऊ;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t total_evict_size = 0;
    लागि (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];
        total_evict_size += r->end - r->start;
    }

    यदि (total_evict_size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        लागि (uint32_t i = 0; i < n_pending; i++) {
            संरचना htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
        }
        merge_dirty_ranges(ctx);
        फर्काऊ;
    }

    यदि (total_evict_size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_MAX_DIRTY_RANGES) {
        संरचना l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        लागि (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];

            संरचना l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((माप_प्रकार) r->start, HEX_L2_पङ्क्ति_SIZE);
            rg->end   = hex_align_up((माप_प्रकार) r->end, HEX_L2_पङ्क्ति_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } अन्यथा {
        संरचना htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        लागि (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t size = r->end - r->start;
            hex_l2flush((शून्य *) (uintptr_t) r->start, size);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    लागि (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->data;
        r->end   = pending[i]->data + pending[i]->size;
    }

    लागि (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->data;
        r->end   = pending[n_evict + i]->data + pending[n_evict + i]->size;
    }

    merge_dirty_ranges(ctx);
}

स्थिर शून्य make_tensor_clean(संरचना htp_context * ctx, अचर संरचना htp_tensor * t) {
    यदि (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        फर्काऊ;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    लागि (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[i];
        यदि (!r->start) जारी;

        यदि (r->start < t_end && t_start < r->end) {
            यदि (t_start <= r->start && r->end <= t_end) {
                r->start = 0;
                r->end   = 0;
            } अन्यथा यदि (t_start <= r->start) {
                r->start = t_end;
            } अन्यथा यदि (r->end <= t_end) {
                r->end = t_start;
            }
        }
    }
}

स्थिर अन्तःस्थ bool is_tensor_dirty(संरचना htp_context * ctx, अचर संरचना htp_tensor * t) {
    यदि (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
        फर्काऊ false;
    }

    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    लागि (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        संरचना htp_dirty_range * r = &ctx->dirty_ranges[i];
        यदि (!r->start) जारी;

        यदि (r->start < t_end && t_start < r->end) {
            फर्काऊ true;
        }
    }
    फर्काऊ false;
}

स्थिर शून्य flush_dirty_ranges(संरचना htp_context * ctx, अचर संरचना htp_dirty_range * ranges, uint32_t n_ranges, uint64_t total_dirty) {
    यदि (total_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        संरचना l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        लागि (uint32_t i = 0; i < n_ranges; i++) {
            अचर संरचना htp_dirty_range * r = &ranges[i];
            संरचना l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((माप_प्रकार) r->start, HEX_L2_पङ्क्ति_SIZE);
            rg->end   = hex_align_up((माप_प्रकार) r->end, HEX_L2_पङ्क्ति_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } अन्यथा {
        संरचना htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        लागि (uint32_t i = 0; i < n_ranges; i++) {
            अचर संरचना htp_dirty_range * r = &ranges[i];
            hex_l2flush((शून्य *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

शून्य htp_dcache_flush_dirty_ranges(संरचना htp_context * ctx) {
    संरचना htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t total_dirty = 0;

    लागि (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        अचर संरचना htp_dirty_range * r = &ctx->dirty_ranges[i];
        यदि (!r->start) {
            जारी;
        }
        ranges[n_ranges++] = *r;
        total_dirty += r->end - r->start;
    }

    यदि (total_dirty == 0) {
        फर्काऊ;
    }

    यदि (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        फर्काऊ;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, total_dirty);
    स_रखो(ctx->dirty_ranges, 0, परिमाण(ctx->dirty_ranges));
}

शून्य htp_dcache_flush_dirty_tensors(संरचना htp_context * ctx, अचर संरचना htp_tensor * अचर * tensors, uint32_t n) {
    अचर संरचना htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    संरचना htp_dirty_range ranges[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t total_dirty = 0;

    लागि (uint32_t i = 0; i < n; i++) {
        अचर संरचना htp_tensor * t = tensors[i];
        यदि (is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
            ranges[n_dirty - 1].start = t->data;
            ranges[n_dirty - 1].end   = t->data + t->size;
            total_dirty += t->size;
        }
    }

    यदि (total_dirty == 0) {
        फर्काऊ;
    }

    यदि (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        फर्काऊ;
    }

    flush_dirty_ranges(ctx, ranges, n_dirty, total_dirty);
    लागि (uint32_t i = 0; i < n_dirty; i++) {
        make_tensor_clean(ctx, dirty_tensors[i]);
    }
}
