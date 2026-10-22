#অন্তর্ভাব "dma-queue.h"

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <string.h>

#pragma clang diagnostic ignored "-Wunused-function"

স্থির অন্তঃস্থ uint32_t pow2_ceil(uint32_t x) {
    যদি (x <= 1) {
        প্রত্যাগম 1;
    }
    পূর্ণাঙ্ক p = 2;
    x--;
    যাবত্ (x >>= 1) {
        p <<= 1;
    }
    প্রত্যাগম p;
}

স্থির অন্তঃস্থ uintptr_t align_up(uintptr_t addr, size_t align) {
    প্রত্যাগম (addr + align - 1) & ~(align - 1);
}

স্থির অন্তঃস্থ size_t dma_ring_sizeof(size_t capacity) {
    capacity = pow2_ceil(capacity);

    size_t size_r      = পরিমাণ(dma_ring);
    size_t offset_desc = align_up(size_r, HEX_L2_সারি_SIZE);
    size_t size_desc   = capacity * পরিমাণ(dma_descriptor_2d);
    size_t offset_data = align_up(offset_desc + size_desc, HEX_L2_সারি_SIZE);
    size_t size_data   = capacity * পরিমাণ(dma_data);

    প্রত্যাগম offset_data + size_data;
}

স্থির অন্তঃস্থ dma_ring * dma_ring_init(শূন্য * ptr, size_t capacity, সংরচনা htp_thread_trace * trace) {
    capacity = pow2_ceil(capacity);

    size_t size_r      = পরিমাণ(dma_ring);
    size_t offset_desc = align_up(size_r, HEX_L2_সারি_SIZE);
    size_t size_desc   = capacity * পরিমাণ(dma_descriptor_2d);
    size_t offset_data = align_up(offset_desc + size_desc, HEX_L2_সারি_SIZE);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->capacity  = capacity;
    r->idx_mask  = capacity - 1;
    r->ঠেলো_idx  = 0;
    r->বের_করো_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + offset_desc);
    r->data      = (dma_data *) ((uintptr_t) ptr + offset_data);
    r->tail      = &r->desc[capacity - 1];

    প্রত্যাগম r;
}

size_t dma_queue_sizeof(size_t capacity) {
    size_t size_q    = পরিমাণ(dma_queue);
    size_t offset_r0 = align_up(size_q, HEX_L2_সারি_SIZE);
    size_t size_r0   = dma_ring_sizeof(capacity);
    size_t offset_r1 = align_up(offset_r0 + size_r0, HEX_L2_সারি_SIZE);
    size_t size_r1   = dma_ring_sizeof(DMA_FALLBACK_CAPACITY);

    প্রত্যাগম offset_r1 + size_r1;
}

size_t dma_queue_alignof(শূন্য) {
    প্রত্যাগম HEX_L2_সারি_SIZE;
}

dma_queue_t dma_queue_init(শূন্য * ptr, size_t capacity, সংরচনা htp_thread_trace * trace) {
    size_t total_size = dma_queue_sizeof(capacity);
    memset(ptr, 0, total_size);

    dma_queue * q = (dma_queue *) ptr;

    size_t size_q    = পরিমাণ(dma_queue);
    size_t offset_r0 = align_up(size_q, HEX_L2_সারি_SIZE);
    size_t size_r0   = dma_ring_sizeof(capacity);
    size_t offset_r1 = align_up(offset_r0 + size_r0, HEX_L2_সারি_SIZE);

    q->ring0 = dma_ring_init((শূন্য *) ((uintptr_t) ptr + offset_r0), capacity, trace);
    q->ring1 = dma_ring_init((শূন্য *) ((uintptr_t) ptr + offset_r1), DMA_FALLBACK_CAPACITY, trace);
    q->alias = false;

    FARF(উপরের_2, "dma-queue: capacity %u, unified memory size %zu\n", (unsigned) capacity, total_size);

    প্রত্যাগম q;
}

শূন্য dma_queue_free(dma_queue_t q) {
    (শূন্য) q;
}

size_t dma_queue_alias_sizeof(শূন্য) {
    প্রত্যাগম পরিমাণ(dma_queue);
}

dma_queue_t dma_queue_alias_init(শূন্য * ptr, dma_queue_t main_q) {
    dma_queue * q = (dma_queue *) ptr;
    memset(q, 0, পরিমাণ(dma_queue));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    প্রত্যাগম q;
}

শূন্য dma_queue_alias_free(dma_queue_t q) {
    (শূন্য) q;
}

bool dma_queue_ঠেলো_fallback_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    যদি (((r0->ঠেলো_idx + 1) & r0->idx_mask) == r0->বের_করো_idx) {
        প্রত্যাগম false;
    }

    r1->tail = r0->tail;

    size_t rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    যাবত্ (rem_rows > 0) {
        ধ্রুব uint32_t cur_rows = MIN(rem_rows, DMA_MAX_NROWS);
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        যদি (!dma_ring_ঠেলো_single_2d(r1, cur_data, dst_stride, src_stride, row_size, cur_rows)) {
            dma_ring_flush(r1);
            dma_ring_ঠেলো_single_2d(r1, cur_data, dst_stride, src_stride, row_size, cur_rows);
        }
        cur_dst  += cur_rows * dst_stride;
        cur_src  += cur_rows * src_stride;
        rem_rows -= cur_rows;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    প্রত্যাগম dma_ring_ঠেলো_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_queue_ঠেলো_fallback_contig(dma_queue * q, dma_data ddata, size_t total) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    যদি (((r0->ঠেলো_idx + 1) & r0->idx_mask) == r0->বের_করো_idx) {
        প্রত্যাগম false;
    }

    r1->tail = r0->tail;

    size_t rem_bytes   = total;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    যাবত্ (rem_bytes > 0) {
        ধ্রুব uint32_t cur_bytes = MIN(rem_bytes, DMA_SAFE_CHUNK_SIZE);
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        যদি (!dma_ring_ঠেলো_single_contig(r1, cur_data, cur_bytes)) {
            dma_ring_flush(r1);
            dma_ring_ঠেলো_single_contig(r1, cur_data, cur_bytes);
        }
        cur_dst   += cur_bytes;
        cur_src   += cur_bytes;
        rem_bytes -= cur_bytes;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    প্রত্যাগম dma_ring_ঠেলো_single_1d(r0, ddata, /*size=*/ 0);
}

#যদি __HVX_ARCH__ < 75

bool dma_queue_ঠেলো_fallback_1d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    যদি (((r0->ঠেলো_idx + 1) & r0->idx_mask) == r0->বের_করো_idx) {
        প্রত্যাগম false;
    }

    r1->tail = r0->tail;

    size_t rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    যাবত্ (rem_rows > 0) {
        dma_data cur_data = dma_make_data(cur_dst, cur_src);
        যদি (!dma_ring_ঠেলো_single_1d(r1, cur_data, row_size)) {
            dma_ring_flush(r1);
            dma_ring_ঠেলো_single_1d(r1, cur_data, row_size);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_rows -= 1;
    }

    dma_ring_flush(r1);
    r0->tail = r1->tail;

    প্রত্যাগম dma_ring_ঠেলো_single_1d(r0, ddata, /*size=*/ 0);
}

#endif
