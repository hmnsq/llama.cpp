#অন্তর্ভাব "dma-queue.h"

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <string.h>

#pragma clang diagnostic ignored "-Wunused-function"

স্থির অন্তঃরেখ uint32_t pow2_ceil(uint32_t x) {
    যদি (x <= 1) {
        ফেরত 1;
    }
    পূর্ণসংখ্যা p = 2;
    x--;
    যখন (x >>= 1) {
        p <<= 1;
    }
    ফেরত p;
}

স্থির অন্তঃরেখ uintptr_t align_up(uintptr_t addr, size_t align) {
    ফেরত (addr + align - 1) & ~(align - 1);
}

স্থির অন্তঃরেখ size_t dma_ring_sizeof(size_t capacity) {
    capacity = pow2_ceil(capacity);

    size_t আকার_r      = আকার(dma_ring);
    size_t offset_desc = align_up(আকার_r, HEX_L2_সারি_আকার);
    size_t আকার_desc   = capacity * আকার(dma_descriptor_2d);
    size_t offset_তথ্য = align_up(offset_desc + আকার_desc, HEX_L2_সারি_আকার);
    size_t আকার_তথ্য   = capacity * আকার(dma_তথ্য);

    ফেরত offset_তথ্য + আকার_তথ্য;
}

স্থির অন্তঃরেখ dma_ring * dma_ring_init(শূন্য * ptr, size_t capacity, কাঠামো htp_thread_trace * trace) {
    capacity = pow2_ceil(capacity);

    size_t আকার_r      = আকার(dma_ring);
    size_t offset_desc = align_up(আকার_r, HEX_L2_সারি_আকার);
    size_t আকার_desc   = capacity * আকার(dma_descriptor_2d);
    size_t offset_তথ্য = align_up(offset_desc + আকার_desc, HEX_L2_সারি_আকার);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->capacity  = capacity;
    r->idx_mask  = capacity - 1;
    r->ঠেলো_idx  = 0;
    r->বের_করো_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + offset_desc);
    r->তথ্য      = (dma_তথ্য *) ((uintptr_t) ptr + offset_তথ্য);
    r->লেজ      = &r->desc[capacity - 1];

    ফেরত r;
}

size_t dma_queue_sizeof(size_t capacity) {
    size_t আকার_q    = আকার(dma_queue);
    size_t offset_r0 = align_up(আকার_q, HEX_L2_সারি_আকার);
    size_t আকার_r0   = dma_ring_sizeof(capacity);
    size_t offset_r1 = align_up(offset_r0 + আকার_r0, HEX_L2_সারি_আকার);
    size_t আকার_r1   = dma_ring_sizeof(DMA_FALLBACK_CAPACITY);

    ফেরত offset_r1 + আকার_r1;
}

size_t dma_queue_alignof(শূন্য) {
    ফেরত HEX_L2_সারি_আকার;
}

dma_queue_t dma_queue_init(শূন্য * ptr, size_t capacity, কাঠামো htp_thread_trace * trace) {
    size_t যোগ_আকার = dma_queue_sizeof(capacity);
    memset(ptr, 0, যোগ_আকার);

    dma_queue * q = (dma_queue *) ptr;

    size_t আকার_q    = আকার(dma_queue);
    size_t offset_r0 = align_up(আকার_q, HEX_L2_সারি_আকার);
    size_t আকার_r0   = dma_ring_sizeof(capacity);
    size_t offset_r1 = align_up(offset_r0 + আকার_r0, HEX_L2_সারি_আকার);

    q->ring0 = dma_ring_init((শূন্য *) ((uintptr_t) ptr + offset_r0), capacity, trace);
    q->ring1 = dma_ring_init((শূন্য *) ((uintptr_t) ptr + offset_r1), DMA_FALLBACK_CAPACITY, trace);
    q->alias = false;

    FARF(উপরের_2, "dma-queue: capacity %u, unified memory size %zu\n", (unsigned) capacity, যোগ_আকার);

    ফেরত q;
}

শূন্য dma_queue_free(dma_queue_t q) {
    (শূন্য) q;
}

size_t dma_queue_alias_sizeof(শূন্য) {
    ফেরত আকার(dma_queue);
}

dma_queue_t dma_queue_alias_init(শূন্য * ptr, dma_queue_t main_q) {
    dma_queue * q = (dma_queue *) ptr;
    memset(q, 0, আকার(dma_queue));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    ফেরত q;
}

শূন্য dma_queue_alias_free(dma_queue_t q) {
    (শূন্য) q;
}

bool dma_queue_ঠেলো_fallback_2d(dma_queue * q, dma_তথ্য ddata, size_t dst_stride, size_t src_stride, size_t row_আকার, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    যদি (((r0->ঠেলো_idx + 1) & r0->idx_mask) == r0->বের_করো_idx) {
        ফেরত false;
    }

    r1->লেজ = r0->লেজ;

    size_t rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    যখন (rem_rows > 0) {
        ধ্রুবক uint32_t cur_rows = MIN(rem_rows, DMA_MAX_NROWS);
        dma_তথ্য cur_তথ্য = dma_make_তথ্য(cur_dst, cur_src);
        যদি (!dma_ring_ঠেলো_single_2d(r1, cur_তথ্য, dst_stride, src_stride, row_আকার, cur_rows)) {
            dma_ring_flush(r1);
            dma_ring_ঠেলো_single_2d(r1, cur_তথ্য, dst_stride, src_stride, row_আকার, cur_rows);
        }
        cur_dst  += cur_rows * dst_stride;
        cur_src  += cur_rows * src_stride;
        rem_rows -= cur_rows;
    }

    dma_ring_flush(r1);
    r0->লেজ = r1->লেজ;

    ফেরত dma_ring_ঠেলো_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_queue_ঠেলো_fallback_contig(dma_queue * q, dma_তথ্য ddata, size_t যোগ) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    যদি (((r0->ঠেলো_idx + 1) & r0->idx_mask) == r0->বের_করো_idx) {
        ফেরত false;
    }

    r1->লেজ = r0->লেজ;

    size_t rem_bytes   = যোগ;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    যখন (rem_bytes > 0) {
        ধ্রুবক uint32_t cur_bytes = MIN(rem_bytes, DMA_SAFE_CHUNK_আকার);
        dma_তথ্য cur_তথ্য = dma_make_তথ্য(cur_dst, cur_src);
        যদি (!dma_ring_ঠেলো_single_contig(r1, cur_তথ্য, cur_bytes)) {
            dma_ring_flush(r1);
            dma_ring_ঠেলো_single_contig(r1, cur_তথ্য, cur_bytes);
        }
        cur_dst   += cur_bytes;
        cur_src   += cur_bytes;
        rem_bytes -= cur_bytes;
    }

    dma_ring_flush(r1);
    r0->লেজ = r1->লেজ;

    ফেরত dma_ring_ঠেলো_single_1d(r0, ddata, /*size=*/ 0);
}

#যদি __HVX_ARCH__ < 75

bool dma_queue_ঠেলো_fallback_1d(dma_queue * q, dma_তথ্য ddata, size_t dst_stride, size_t src_stride, size_t row_আকার, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    যদি (((r0->ঠেলো_idx + 1) & r0->idx_mask) == r0->বের_করো_idx) {
        ফেরত false;
    }

    r1->লেজ = r0->লেজ;

    size_t rem_rows    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    যখন (rem_rows > 0) {
        dma_তথ্য cur_তথ্য = dma_make_তথ্য(cur_dst, cur_src);
        যদি (!dma_ring_ঠেলো_single_1d(r1, cur_তথ্য, row_আকার)) {
            dma_ring_flush(r1);
            dma_ring_ঠেলো_single_1d(r1, cur_তথ্য, row_আকার);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_rows -= 1;
    }

    dma_ring_flush(r1);
    r0->লেজ = r1->লেজ;

    ফেরত dma_ring_ঠেলো_single_1d(r0, ddata, /*size=*/ 0);
}

#endif
