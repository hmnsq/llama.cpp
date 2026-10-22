#ضمن "dma-queue.h"

#ضمن <stdbool.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#pragma clang diagnostic ignored "-Wunused-function"

ساكن مضمن uint32_t pow2_ceil(uint32_t x) {
    إذا (x <= 1) {
        أرجع 1;
    }
    صحيح p = 2;
    x--;
    طالما (x >>= 1) {
        p <<= 1;
    }
    أرجع p;
}

ساكن مضمن uintptr_t align_up(uintptr_t addr, size_t align) {
    أرجع (addr + align - 1) & ~(align - 1);
}

ساكن مضمن size_t dma_ring_sizeof(size_t capacity) {
    capacity = pow2_ceil(capacity);

    size_t size_r      = حجم(dma_ring);
    size_t offset_desc = align_up(size_r, HEX_L2_سطر_SIZE);
    size_t size_desc   = capacity * حجم(dma_descriptor_2d);
    size_t offset_data = align_up(offset_desc + size_desc, HEX_L2_سطر_SIZE);
    size_t size_data   = capacity * حجم(dma_data);

    أرجع offset_data + size_data;
}

ساكن مضمن dma_ring * dma_ring_init(فراغ * ptr, size_t capacity, بنية htp_thread_trace * trace) {
    capacity = pow2_ceil(capacity);

    size_t size_r      = حجم(dma_ring);
    size_t offset_desc = align_up(size_r, HEX_L2_سطر_SIZE);
    size_t size_desc   = capacity * حجم(dma_descriptor_2d);
    size_t offset_data = align_up(offset_desc + size_desc, HEX_L2_سطر_SIZE);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->capacity  = capacity;
    r->idx_mask  = capacity - 1;
    r->ادفع_idx  = 0;
    r->اسحب_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + offset_desc);
    r->data      = (dma_data *) ((uintptr_t) ptr + offset_data);
    r->tail      = &r->desc[capacity - 1];

    أرجع r;
}

size_t dma_queue_sizeof(size_t capacity) {
    size_t size_q    = حجم(dma_queue);
    size_t offset_r0 = align_up(size_q, HEX_L2_سطر_SIZE);
    size_t size_r0   = dma_ring_sizeof(capacity);
    size_t offset_r1 = align_up(offset_r0 + size_r0, HEX_L2_سطر_SIZE);
    size_t size_r1   = dma_ring_sizeof(DMA_FALLBACK_CAPACITY);

    أرجع offset_r1 + size_r1;
}

size_t dma_queue_alignof(فراغ) {
    أرجع HEX_L2_سطر_SIZE;
}

dma_queue_t dma_queue_init(فراغ * ptr, size_t capacity, بنية htp_thread_trace * trace) {
    size_t total_size = dma_queue_sizeof(capacity);
    memset(ptr, 0, total_size);

    dma_queue * q = (dma_queue *) ptr;

    size_t size_q    = حجم(dma_queue);
    size_t offset_r0 = align_up(size_q, HEX_L2_سطر_SIZE);
    size_t size_r0   = dma_ring_sizeof(capacity);
    size_t offset_r1 = align_up(offset_r0 + size_r0, HEX_L2_سطر_SIZE);

    q->ring0 = dma_ring_init((فراغ *) ((uintptr_t) ptr + offset_r0), capacity, trace);
    q->ring1 = dma_ring_init((فراغ *) ((uintptr_t) ptr + offset_r1), DMA_FALLBACK_CAPACITY, trace);
    q->alias = false;

    FARF(عالي_2, "dma-queue: capacity %u, unified memory size %zu\n", (unsigned) capacity, total_size);

    أرجع q;
}

فراغ dma_queue_free(dma_queue_t q) {
    (فراغ) q;
}

size_t dma_queue_alias_sizeof(فراغ) {
    أرجع حجم(dma_queue);
}

dma_queue_t dma_queue_alias_init(فراغ * ptr, dma_queue_t main_q) {
    dma_queue * q = (dma_queue *) ptr;
    memset(q, 0, حجم(dma_queue));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    أرجع q;
}

فراغ dma_queue_alias_free(dma_queue_t q) {
    (فراغ) q;
}

bool dma_queue_ادفع_fallback_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ادفع_idx + 1) & r0->idx_mask) == r0->اسحب_idx) {
        أرجع false;
    }

    r1->tail = r0->tail;

    size_t rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_rows > 0) {
        ثابت uint32_t cur_rows = MIN(rem_rows, DMA_MAX_NROWS);
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        إذا (!dma_ring_ادفع_single_2d(r1, cur_data, dst_stride, src_stride, row_size, cur_rows)) {
            dma_ring_flush(r1);
            dma_ring_ادفع_single_2d(r1, cur_data, dst_stride, src_stride, row_size, cur_rows);
        }
        cur_dst  += cur_rows * dst_stride;
        cur_src  += cur_rows * src_stride;
        rem_rows -= cur_rows;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    أرجع dma_ring_ادفع_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_queue_ادفع_fallback_contig(dma_queue * q, dma_data ddata, size_t total) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ادفع_idx + 1) & r0->idx_mask) == r0->اسحب_idx) {
        أرجع false;
    }

    r1->tail = r0->tail;

    size_t rem_bytes   = total;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_bytes > 0) {
        ثابت uint32_t cur_bytes = MIN(rem_bytes, DMA_SAFE_CHUNK_SIZE);
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        إذا (!dma_ring_ادفع_single_contig(r1, cur_data, cur_bytes)) {
            dma_ring_flush(r1);
            dma_ring_ادفع_single_contig(r1, cur_data, cur_bytes);
        }
        cur_dst   += cur_bytes;
        cur_src   += cur_bytes;
        rem_bytes -= cur_bytes;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    أرجع dma_ring_ادفع_single_1d(r0, ddata, /*size=*/ 0);
}

#إذا __HVX_ARCH__ < 75

bool dma_queue_ادفع_fallback_1d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ادفع_idx + 1) & r0->idx_mask) == r0->اسحب_idx) {
        أرجع false;
    }

    r1->tail = r0->tail;

    size_t rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_rows > 0) {
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        إذا (!dma_ring_ادفع_single_1d(r1, cur_data, row_size)) {
            dma_ring_flush(r1);
            dma_ring_ادفع_single_1d(r1, cur_data, row_size);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_rows -= 1;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    أرجع dma_ring_ادفع_single_1d(r0, ddata, /*size=*/ 0);
}

#endif
