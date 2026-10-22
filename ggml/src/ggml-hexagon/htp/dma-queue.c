#अन्तर्भाव "dma-queue.h"

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>

#pragma clang diagnostic ignored "-Wunused-function"

स्थिर ओळीत uint32_t pow2_ceil(uint32_t x) {
    जर (x <= 1) {
        परत 1;
    }
    पूर्णांक p = 2;
    x--;
    पर्यंत (x >>= 1) {
        p <<= 1;
    }
    परत p;
}

स्थिर ओळीत uintptr_t align_up(uintptr_t addr, माप_प्रकार align) {
    परत (addr + align - 1) & ~(align - 1);
}

स्थिर ओळीत माप_प्रकार dma_ring_sizeof(माप_प्रकार capacity) {
    capacity = pow2_ceil(capacity);

    माप_प्रकार size_r      = आकार(dma_ring);
    माप_प्रकार offset_desc = align_up(size_r, HEX_L2_ओळ_SIZE);
    माप_प्रकार size_desc   = capacity * आकार(dma_descriptor_2d);
    माप_प्रकार offset_data = align_up(offset_desc + size_desc, HEX_L2_ओळ_SIZE);
    माप_प्रकार size_data   = capacity * आकार(dma_data);

    परत offset_data + size_data;
}

स्थिर ओळीत dma_ring * dma_ring_init(शून्य * ptr, माप_प्रकार capacity, संरचना htp_thread_trace * trace) {
    capacity = pow2_ceil(capacity);

    माप_प्रकार size_r      = आकार(dma_ring);
    माप_प्रकार offset_desc = align_up(size_r, HEX_L2_ओळ_SIZE);
    माप_प्रकार size_desc   = capacity * आकार(dma_descriptor_2d);
    माप_प्रकार offset_data = align_up(offset_desc + size_desc, HEX_L2_ओळ_SIZE);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->capacity  = capacity;
    r->idx_mask  = capacity - 1;
    r->ढकला_idx  = 0;
    r->काढा_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + offset_desc);
    r->data      = (dma_data *) ((uintptr_t) ptr + offset_data);
    r->tail      = &r->desc[capacity - 1];

    परत r;
}

माप_प्रकार dma_queue_sizeof(माप_प्रकार capacity) {
    माप_प्रकार size_q    = आकार(dma_queue);
    माप_प्रकार offset_r0 = align_up(size_q, HEX_L2_ओळ_SIZE);
    माप_प्रकार size_r0   = dma_ring_sizeof(capacity);
    माप_प्रकार offset_r1 = align_up(offset_r0 + size_r0, HEX_L2_ओळ_SIZE);
    माप_प्रकार size_r1   = dma_ring_sizeof(DMA_FALLBACK_CAPACITY);

    परत offset_r1 + size_r1;
}

माप_प्रकार dma_queue_alignof(शून्य) {
    परत HEX_L2_ओळ_SIZE;
}

dma_queue_t dma_queue_init(शून्य * ptr, माप_प्रकार capacity, संरचना htp_thread_trace * trace) {
    माप_प्रकार total_size = dma_queue_sizeof(capacity);
    स_रखो(ptr, 0, total_size);

    dma_queue * q = (dma_queue *) ptr;

    माप_प्रकार size_q    = आकार(dma_queue);
    माप_प्रकार offset_r0 = align_up(size_q, HEX_L2_ओळ_SIZE);
    माप_प्रकार size_r0   = dma_ring_sizeof(capacity);
    माप_प्रकार offset_r1 = align_up(offset_r0 + size_r0, HEX_L2_ओळ_SIZE);

    q->ring0 = dma_ring_init((शून्य *) ((uintptr_t) ptr + offset_r0), capacity, trace);
    q->ring1 = dma_ring_init((शून्य *) ((uintptr_t) ptr + offset_r1), DMA_FALLBACK_CAPACITY, trace);
    q->alias = false;

    FARF(वरचा_2, "dma-queue: capacity %u, unified memory size %zu\n", (अचिन्हित) capacity, total_size);

    परत q;
}

शून्य dma_queue_free(dma_queue_t q) {
    (शून्य) q;
}

माप_प्रकार dma_queue_alias_sizeof(शून्य) {
    परत आकार(dma_queue);
}

dma_queue_t dma_queue_alias_init(शून्य * ptr, dma_queue_t main_q) {
    dma_queue * q = (dma_queue *) ptr;
    स_रखो(q, 0, आकार(dma_queue));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    परत q;
}

शून्य dma_queue_alias_free(dma_queue_t q) {
    (शून्य) q;
}

bool dma_queue_ढकला_fallback_2d(dma_queue * q, dma_data ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_size, माप_प्रकार nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    जर (((r0->ढकला_idx + 1) & r0->idx_mask) == r0->काढा_idx) {
        परत false;
    }

    r1->tail = r0->tail;

    माप_प्रकार rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    पर्यंत (rem_rows > 0) {
        स्थिरांक uint32_t cur_rows = MIN(rem_rows, DMA_MAX_NROWS);
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        जर (!dma_ring_ढकला_single_2d(r1, cur_data, dst_stride, src_stride, row_size, cur_rows)) {
            dma_ring_flush(r1);
            dma_ring_ढकला_single_2d(r1, cur_data, dst_stride, src_stride, row_size, cur_rows);
        }
        cur_dst  += cur_rows * dst_stride;
        cur_src  += cur_rows * src_stride;
        rem_rows -= cur_rows;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    परत dma_ring_ढकला_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_queue_ढकला_fallback_contig(dma_queue * q, dma_data ddata, माप_प्रकार total) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    जर (((r0->ढकला_idx + 1) & r0->idx_mask) == r0->काढा_idx) {
        परत false;
    }

    r1->tail = r0->tail;

    माप_प्रकार rem_bytes   = total;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    पर्यंत (rem_bytes > 0) {
        स्थिरांक uint32_t cur_bytes = MIN(rem_bytes, DMA_SAFE_CHUNK_SIZE);
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        जर (!dma_ring_ढकला_single_contig(r1, cur_data, cur_bytes)) {
            dma_ring_flush(r1);
            dma_ring_ढकला_single_contig(r1, cur_data, cur_bytes);
        }
        cur_dst   += cur_bytes;
        cur_src   += cur_bytes;
        rem_bytes -= cur_bytes;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    परत dma_ring_ढकला_single_1d(r0, ddata, /*size=*/ 0);
}

#जर __HVX_ARCH__ < 75

bool dma_queue_ढकला_fallback_1d(dma_queue * q, dma_data ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_size, माप_प्रकार nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    जर (((r0->ढकला_idx + 1) & r0->idx_mask) == r0->काढा_idx) {
        परत false;
    }

    r1->tail = r0->tail;

    माप_प्रकार rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    पर्यंत (rem_rows > 0) {
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        जर (!dma_ring_ढकला_single_1d(r1, cur_data, row_size)) {
            dma_ring_flush(r1);
            dma_ring_ढकला_single_1d(r1, cur_data, row_size);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_rows -= 1;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    परत dma_ring_ढकला_single_1d(r0, ddata, /*size=*/ 0);
}

#endif
