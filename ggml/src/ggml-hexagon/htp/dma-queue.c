#अन्तर्भाव "dma-queue.h"

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>

#pragma clang diagnostic ignored "-Wunused-function"

स्थिर अन्तःस्थ uint32_t pow2_ceil(uint32_t x) {
    यदि (x <= 1) {
        फर्काऊ 1;
    }
    पूर्णाङ्क p = 2;
    x--;
    जबसम्म (x >>= 1) {
        p <<= 1;
    }
    फर्काऊ p;
}

स्थिर अन्तःस्थ uintptr_t align_up(uintptr_t addr, माप_प्रकार align) {
    फर्काऊ (addr + align - 1) & ~(align - 1);
}

स्थिर अन्तःस्थ माप_प्रकार dma_ring_sizeof(माप_प्रकार capacity) {
    capacity = pow2_ceil(capacity);

    माप_प्रकार size_r      = परिमाण(dma_ring);
    माप_प्रकार offset_desc = align_up(size_r, HEX_L2_पङ्क्ति_SIZE);
    माप_प्रकार size_desc   = capacity * परिमाण(dma_descriptor_2d);
    माप_प्रकार offset_data = align_up(offset_desc + size_desc, HEX_L2_पङ्क्ति_SIZE);
    माप_प्रकार size_data   = capacity * परिमाण(dma_data);

    फर्काऊ offset_data + size_data;
}

स्थिर अन्तःस्थ dma_ring * dma_ring_init(शून्य * ptr, माप_प्रकार capacity, संरचना htp_thread_trace * trace) {
    capacity = pow2_ceil(capacity);

    माप_प्रकार size_r      = परिमाण(dma_ring);
    माप_प्रकार offset_desc = align_up(size_r, HEX_L2_पङ्क्ति_SIZE);
    माप_प्रकार size_desc   = capacity * परिमाण(dma_descriptor_2d);
    माप_प्रकार offset_data = align_up(offset_desc + size_desc, HEX_L2_पङ्क्ति_SIZE);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->capacity  = capacity;
    r->idx_mask  = capacity - 1;
    r->थप_idx  = 0;
    r->झिक_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + offset_desc);
    r->data      = (dma_data *) ((uintptr_t) ptr + offset_data);
    r->tail      = &r->desc[capacity - 1];

    फर्काऊ r;
}

माप_प्रकार dma_queue_sizeof(माप_प्रकार capacity) {
    माप_प्रकार size_q    = परिमाण(dma_queue);
    माप_प्रकार offset_r0 = align_up(size_q, HEX_L2_पङ्क्ति_SIZE);
    माप_प्रकार size_r0   = dma_ring_sizeof(capacity);
    माप_प्रकार offset_r1 = align_up(offset_r0 + size_r0, HEX_L2_पङ्क्ति_SIZE);
    माप_प्रकार size_r1   = dma_ring_sizeof(DMA_FALLBACK_CAPACITY);

    फर्काऊ offset_r1 + size_r1;
}

माप_प्रकार dma_queue_alignof(शून्य) {
    फर्काऊ HEX_L2_पङ्क्ति_SIZE;
}

dma_queue_t dma_queue_init(शून्य * ptr, माप_प्रकार capacity, संरचना htp_thread_trace * trace) {
    माप_प्रकार total_size = dma_queue_sizeof(capacity);
    स_रखो(ptr, 0, total_size);

    dma_queue * q = (dma_queue *) ptr;

    माप_प्रकार size_q    = परिमाण(dma_queue);
    माप_प्रकार offset_r0 = align_up(size_q, HEX_L2_पङ्क्ति_SIZE);
    माप_प्रकार size_r0   = dma_ring_sizeof(capacity);
    माप_प्रकार offset_r1 = align_up(offset_r0 + size_r0, HEX_L2_पङ्क्ति_SIZE);

    q->ring0 = dma_ring_init((शून्य *) ((uintptr_t) ptr + offset_r0), capacity, trace);
    q->ring1 = dma_ring_init((शून्य *) ((uintptr_t) ptr + offset_r1), DMA_FALLBACK_CAPACITY, trace);
    q->alias = false;

    FARF(माथिल्लो_2, "dma-queue: capacity %u, unified memory size %zu\n", (अचिन्हित) capacity, total_size);

    फर्काऊ q;
}

शून्य dma_queue_free(dma_queue_t q) {
    (शून्य) q;
}

माप_प्रकार dma_queue_alias_sizeof(शून्य) {
    फर्काऊ परिमाण(dma_queue);
}

dma_queue_t dma_queue_alias_init(शून्य * ptr, dma_queue_t main_q) {
    dma_queue * q = (dma_queue *) ptr;
    स_रखो(q, 0, परिमाण(dma_queue));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    फर्काऊ q;
}

शून्य dma_queue_alias_free(dma_queue_t q) {
    (शून्य) q;
}

bool dma_queue_थप_fallback_2d(dma_queue * q, dma_data ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_size, माप_प्रकार nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    यदि (((r0->थप_idx + 1) & r0->idx_mask) == r0->झिक_idx) {
        फर्काऊ false;
    }

    r1->tail = r0->tail;

    माप_प्रकार rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    जबसम्म (rem_rows > 0) {
        अचर uint32_t cur_rows = MIN(rem_rows, DMA_MAX_NROWS);
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        यदि (!dma_ring_थप_single_2d(r1, cur_data, dst_stride, src_stride, row_size, cur_rows)) {
            dma_ring_flush(r1);
            dma_ring_थप_single_2d(r1, cur_data, dst_stride, src_stride, row_size, cur_rows);
        }
        cur_dst  += cur_rows * dst_stride;
        cur_src  += cur_rows * src_stride;
        rem_rows -= cur_rows;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    फर्काऊ dma_ring_थप_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_queue_थप_fallback_contig(dma_queue * q, dma_data ddata, माप_प्रकार total) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    यदि (((r0->थप_idx + 1) & r0->idx_mask) == r0->झिक_idx) {
        फर्काऊ false;
    }

    r1->tail = r0->tail;

    माप_प्रकार rem_bytes   = total;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    जबसम्म (rem_bytes > 0) {
        अचर uint32_t cur_bytes = MIN(rem_bytes, DMA_SAFE_CHUNK_SIZE);
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        यदि (!dma_ring_थप_single_contig(r1, cur_data, cur_bytes)) {
            dma_ring_flush(r1);
            dma_ring_थप_single_contig(r1, cur_data, cur_bytes);
        }
        cur_dst   += cur_bytes;
        cur_src   += cur_bytes;
        rem_bytes -= cur_bytes;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    फर्काऊ dma_ring_थप_single_1d(r0, ddata, /*size=*/ 0);
}

#यदि __HVX_ARCH__ < 75

bool dma_queue_थप_fallback_1d(dma_queue * q, dma_data ddata, माप_प्रकार dst_stride, माप_प्रकार src_stride, माप_प्रकार row_size, माप_प्रकार nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    यदि (((r0->थप_idx + 1) & r0->idx_mask) == r0->झिक_idx) {
        फर्काऊ false;
    }

    r1->tail = r0->tail;

    माप_प्रकार rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    जबसम्म (rem_rows > 0) {
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        यदि (!dma_ring_थप_single_1d(r1, cur_data, row_size)) {
            dma_ring_flush(r1);
            dma_ring_थप_single_1d(r1, cur_data, row_size);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_rows -= 1;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    फर्काऊ dma_ring_थप_single_1d(r0, ddata, /*size=*/ 0);
}

#endif
