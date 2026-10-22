#अन्तर्भाव "dma-queue.h"

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>

#pragma clang diagnostic ignored "-Wunused-function"

स्थिर अन्तःस्थ uint32_t pow2_ceil(uint32_t x) {
    सचे (x <= 1) {
        पटिनिवत्त 1;
    }
    पूर्णाङ्क p = 2;
    x--;
    याव (x >>= 1) {
        p <<= 1;
    }
    पटिनिवत्त p;
}

स्थिर अन्तःस्थ uintptr_t align_up(uintptr_t addr, size_t align) {
    पटिनिवत्त (addr + align - 1) & ~(align - 1);
}

स्थिर अन्तःस्थ size_t dma_ring_sizeof(size_t capacity) {
    capacity = pow2_ceil(capacity);

    size_t size_r      = परिमाण(dma_ring);
    size_t offset_desc = align_up(size_r, HEX_L2_पङ्क्तिः_SIZE);
    size_t size_desc   = capacity * परिमाण(dma_descriptor_2d);
    size_t offset_data = align_up(offset_desc + size_desc, HEX_L2_पङ्क्तिः_SIZE);
    size_t size_data   = capacity * परिमाण(dma_data);

    पटिनिवत्त offset_data + size_data;
}

स्थिर अन्तःस्थ dma_ring * dma_ring_init(शून्य * ptr, size_t capacity, संरचना htp_thread_trace * trace) {
    capacity = pow2_ceil(capacity);

    size_t size_r      = परिमाण(dma_ring);
    size_t offset_desc = align_up(size_r, HEX_L2_पङ्क्तिः_SIZE);
    size_t size_desc   = capacity * परिमाण(dma_descriptor_2d);
    size_t offset_data = align_up(offset_desc + size_desc, HEX_L2_पङ्क्तिः_SIZE);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->capacity  = capacity;
    r->idx_mask  = capacity - 1;
    r->प्रक्षिप_idx  = 0;
    r->निष्कासय_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + offset_desc);
    r->data      = (dma_data *) ((uintptr_t) ptr + offset_data);
    r->tail      = &r->desc[capacity - 1];

    पटिनिवत्त r;
}

size_t dma_queue_sizeof(size_t capacity) {
    size_t size_q    = परिमाण(dma_queue);
    size_t offset_r0 = align_up(size_q, HEX_L2_पङ्क्तिः_SIZE);
    size_t size_r0   = dma_ring_sizeof(capacity);
    size_t offset_r1 = align_up(offset_r0 + size_r0, HEX_L2_पङ्क्तिः_SIZE);
    size_t size_r1   = dma_ring_sizeof(DMA_FALLBACK_CAPACITY);

    पटिनिवत्त offset_r1 + size_r1;
}

size_t dma_queue_alignof(शून्य) {
    पटिनिवत्त HEX_L2_पङ्क्तिः_SIZE;
}

dma_queue_t dma_queue_init(शून्य * ptr, size_t capacity, संरचना htp_thread_trace * trace) {
    size_t total_size = dma_queue_sizeof(capacity);
    memset(ptr, 0, total_size);

    dma_queue * q = (dma_queue *) ptr;

    size_t size_q    = परिमाण(dma_queue);
    size_t offset_r0 = align_up(size_q, HEX_L2_पङ्क्तिः_SIZE);
    size_t size_r0   = dma_ring_sizeof(capacity);
    size_t offset_r1 = align_up(offset_r0 + size_r0, HEX_L2_पङ्क्तिः_SIZE);

    q->ring0 = dma_ring_init((शून्य *) ((uintptr_t) ptr + offset_r0), capacity, trace);
    q->ring1 = dma_ring_init((शून्य *) ((uintptr_t) ptr + offset_r1), DMA_FALLBACK_CAPACITY, trace);
    q->alias = false;

    FARF(उच्चः_2, "dma-queue: capacity %u, unified memory size %zu\n", (unsigned) capacity, total_size);

    पटिनिवत्त q;
}

शून्य dma_queue_free(dma_queue_t q) {
    (शून्य) q;
}

size_t dma_queue_alias_sizeof(शून्य) {
    पटिनिवत्त परिमाण(dma_queue);
}

dma_queue_t dma_queue_alias_init(शून्य * ptr, dma_queue_t main_q) {
    dma_queue * q = (dma_queue *) ptr;
    memset(q, 0, परिमाण(dma_queue));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    पटिनिवत्त q;
}

शून्य dma_queue_alias_free(dma_queue_t q) {
    (शून्य) q;
}

bool dma_queue_प्रक्षिप_fallback_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    सचे (((r0->प्रक्षिप_idx + 1) & r0->idx_mask) == r0->निष्कासय_idx) {
        पटिनिवत्त false;
    }

    r1->tail = r0->tail;

    size_t rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    याव (rem_rows > 0) {
        ध्रुव uint32_t cur_rows = MIN(rem_rows, DMA_MAX_NROWS);
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        सचे (!dma_ring_प्रक्षिप_single_2d(r1, cur_data, dst_stride, src_stride, row_size, cur_rows)) {
            dma_ring_flush(r1);
            dma_ring_प्रक्षिप_single_2d(r1, cur_data, dst_stride, src_stride, row_size, cur_rows);
        }
        cur_dst  += cur_rows * dst_stride;
        cur_src  += cur_rows * src_stride;
        rem_rows -= cur_rows;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    पटिनिवत्त dma_ring_प्रक्षिप_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_queue_प्रक्षिप_fallback_contig(dma_queue * q, dma_data ddata, size_t total) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    सचे (((r0->प्रक्षिप_idx + 1) & r0->idx_mask) == r0->निष्कासय_idx) {
        पटिनिवत्त false;
    }

    r1->tail = r0->tail;

    size_t rem_bytes   = total;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    याव (rem_bytes > 0) {
        ध्रुव uint32_t cur_bytes = MIN(rem_bytes, DMA_SAFE_CHUNK_SIZE);
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        सचे (!dma_ring_प्रक्षिप_single_contig(r1, cur_data, cur_bytes)) {
            dma_ring_flush(r1);
            dma_ring_प्रक्षिप_single_contig(r1, cur_data, cur_bytes);
        }
        cur_dst   += cur_bytes;
        cur_src   += cur_bytes;
        rem_bytes -= cur_bytes;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    पटिनिवत्त dma_ring_प्रक्षिप_single_1d(r0, ddata, /*size=*/ 0);
}

#सचे __HVX_ARCH__ < 75

bool dma_queue_प्रक्षिप_fallback_1d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    सचे (((r0->प्रक्षिप_idx + 1) & r0->idx_mask) == r0->निष्कासय_idx) {
        पटिनिवत्त false;
    }

    r1->tail = r0->tail;

    size_t rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    याव (rem_rows > 0) {
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        सचे (!dma_ring_प्रक्षिप_single_1d(r1, cur_data, row_size)) {
            dma_ring_flush(r1);
            dma_ring_प्रक्षिप_single_1d(r1, cur_data, row_size);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_rows -= 1;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    पटिनिवत्त dma_ring_प्रक्षिप_single_1d(r0, ddata, /*size=*/ 0);
}

#endif
