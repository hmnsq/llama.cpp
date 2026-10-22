#ضمن "dma-queue.h"

#ضمن <stdbool.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#ہدایت clang diagnostic ignored "-Wunused-function"

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

ساكن مضمن uintptr_t align_اتے(uintptr_t addr, size_t align) {
    أرجع (addr + align - 1) & ~(align - 1);
}

ساكن مضمن size_t dma_ring_sizeof(size_t سمرتھا) {
    سمرتھا = pow2_ceil(سمرتھا);

    size_t ناپ_r      = حجم(dma_ring);
    size_t وتھ_desc = align_اتے(ناپ_r, HEX_L2_سطر_ناپ);
    size_t ناپ_desc   = سمرتھا * حجم(dma_descriptor_2d);
    size_t وتھ_انکڑے = align_اتے(وتھ_desc + ناپ_desc, HEX_L2_سطر_ناپ);
    size_t ناپ_انکڑے   = سمرتھا * حجم(dma_انکڑے);

    أرجع وتھ_انکڑے + ناپ_انکڑے;
}

ساكن مضمن dma_ring * dma_ring_init(فراغ * ptr, size_t سمرتھا, بنية htp_thread_trace * trace) {
    سمرتھا = pow2_ceil(سمرتھا);

    size_t ناپ_r      = حجم(dma_ring);
    size_t وتھ_desc = align_اتے(ناپ_r, HEX_L2_سطر_ناپ);
    size_t ناپ_desc   = سمرتھا * حجم(dma_descriptor_2d);
    size_t وتھ_انکڑے = align_اتے(وتھ_desc + ناپ_desc, HEX_L2_سطر_ناپ);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->سمرتھا  = سمرتھا;
    r->idx_نقاب  = سمرتھا - 1;
    r->ڈالو_idx  = 0;
    r->نکالو_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + وتھ_desc);
    r->انکڑے      = (dma_انکڑے *) ((uintptr_t) ptr + وتھ_انکڑے);
    r->پوچھ      = &r->desc[سمرتھا - 1];

    أرجع r;
}

size_t dma_قطار_sizeof(size_t سمرتھا) {
    size_t ناپ_q    = حجم(dma_قطار);
    size_t وتھ_r0 = align_اتے(ناپ_q, HEX_L2_سطر_ناپ);
    size_t ناپ_r0   = dma_ring_sizeof(سمرتھا);
    size_t وتھ_r1 = align_اتے(وتھ_r0 + ناپ_r0, HEX_L2_سطر_ناپ);
    size_t ناپ_r1   = dma_ring_sizeof(DMA_FALLBACK_سمرتھا);

    أرجع وتھ_r1 + ناپ_r1;
}

size_t dma_قطار_alignof(فراغ) {
    أرجع HEX_L2_سطر_ناپ;
}

dma_قطار_t dma_قطار_init(فراغ * ptr, size_t سمرتھا, بنية htp_thread_trace * trace) {
    size_t کل_ناپ = dma_قطار_sizeof(سمرتھا);
    memset(ptr, 0, کل_ناپ);

    dma_قطار * q = (dma_قطار *) ptr;

    size_t ناپ_q    = حجم(dma_قطار);
    size_t وتھ_r0 = align_اتے(ناپ_q, HEX_L2_سطر_ناپ);
    size_t ناپ_r0   = dma_ring_sizeof(سمرتھا);
    size_t وتھ_r1 = align_اتے(وتھ_r0 + ناپ_r0, HEX_L2_سطر_ناپ);

    q->ring0 = dma_ring_init((فراغ *) ((uintptr_t) ptr + وتھ_r0), سمرتھا, trace);
    q->ring1 = dma_ring_init((فراغ *) ((uintptr_t) ptr + وتھ_r1), DMA_FALLBACK_سمرتھا, trace);
    q->alias = false;

    FARF(بلند_2, "dma-queue: capacity %u, unified memory size %zu\n", (بنانشانی) سمرتھا, کل_ناپ);

    أرجع q;
}

فراغ dma_قطار_free(dma_قطار_t q) {
    (فراغ) q;
}

size_t dma_قطار_alias_sizeof(فراغ) {
    أرجع حجم(dma_قطار);
}

dma_قطار_t dma_قطار_alias_init(فراغ * ptr, dma_قطار_t main_q) {
    dma_قطار * q = (dma_قطار *) ptr;
    memset(q, 0, حجم(dma_قطار));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    أرجع q;
}

فراغ dma_قطار_alias_free(dma_قطار_t q) {
    (فراغ) q;
}

bool dma_قطار_ڈالو_fallback_2d(dma_قطار * q, dma_انکڑے ddata, size_t dst_stride, size_t src_stride, size_t صف_ناپ, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ڈالو_idx + 1) & r0->idx_نقاب) == r0->نکالو_idx) {
        أرجع false;
    }

    r1->پوچھ = r0->پوچھ;

    size_t rem_صفاں    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_صفاں > 0) {
        ثابت uint32_t cur_صفاں = MIN(rem_صفاں, DMA_MAX_NROWS);
        dma_انکڑے cur_انکڑے = dma_make_انکڑے(cur_dst, cur_src);
        إذا (!dma_ring_ڈالو_single_2d(r1, cur_انکڑے, dst_stride, src_stride, صف_ناپ, cur_صفاں)) {
            dma_ring_flush(r1);
            dma_ring_ڈالو_single_2d(r1, cur_انکڑے, dst_stride, src_stride, صف_ناپ, cur_صفاں);
        }
        cur_dst  += cur_صفاں * dst_stride;
        cur_src  += cur_صفاں * src_stride;
        rem_صفاں -= cur_صفاں;
    }

    dma_ring_flush(r1);
    r0->پوچھ = r1->پوچھ;

    أرجع dma_ring_ڈالو_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_قطار_ڈالو_fallback_contig(dma_قطار * q, dma_انکڑے ddata, size_t کل) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ڈالو_idx + 1) & r0->idx_نقاب) == r0->نکالو_idx) {
        أرجع false;
    }

    r1->پوچھ = r0->پوچھ;

    size_t rem_بائٹاں   = کل;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_بائٹاں > 0) {
        ثابت uint32_t cur_بائٹاں = MIN(rem_بائٹاں, DMA_محفوظ_CHUNK_ناپ);
        dma_انکڑے cur_انکڑے = dma_make_انکڑے(cur_dst, cur_src);
        إذا (!dma_ring_ڈالو_single_contig(r1, cur_انکڑے, cur_بائٹاں)) {
            dma_ring_flush(r1);
            dma_ring_ڈالو_single_contig(r1, cur_انکڑے, cur_بائٹاں);
        }
        cur_dst   += cur_بائٹاں;
        cur_src   += cur_بائٹاں;
        rem_بائٹاں -= cur_بائٹاں;
    }

    dma_ring_flush(r1);
    r0->پوچھ = r1->پوچھ;

    أرجع dma_ring_ڈالو_single_1d(r0, ddata, /*size=*/ 0);
}

#إذا __HVX_ARCH__ < 75

bool dma_قطار_ڈالو_fallback_1d(dma_قطار * q, dma_انکڑے ddata, size_t dst_stride, size_t src_stride, size_t صف_ناپ, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ڈالو_idx + 1) & r0->idx_نقاب) == r0->نکالو_idx) {
        أرجع false;
    }

    r1->پوچھ = r0->پوچھ;

    size_t rem_صفاں    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_صفاں > 0) {
        dma_انکڑے cur_انکڑے = dma_make_انکڑے(cur_dst, cur_src);
        إذا (!dma_ring_ڈالو_single_1d(r1, cur_انکڑے, صف_ناپ)) {
            dma_ring_flush(r1);
            dma_ring_ڈالو_single_1d(r1, cur_انکڑے, صف_ناپ);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_صفاں -= 1;
    }

    dma_ring_flush(r1);
    r0->پوچھ = r1->پوچھ;

    أرجع dma_ring_ڈالو_single_1d(r0, ddata, /*size=*/ 0);
}

#جے_انت
