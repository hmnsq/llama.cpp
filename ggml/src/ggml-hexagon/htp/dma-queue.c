#ضمن "dma-queue.h"

#ضمن <stdbool.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#هدايت clang diagnostic ignored "-Wunused-function"

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

ساكن مضمن uintptr_t align_مٿي(uintptr_t addr, size_t align) {
    أرجع (addr + align - 1) & ~(align - 1);
}

ساكن مضمن size_t dma_ring_sizeof(size_t صلاحيت) {
    صلاحيت = pow2_ceil(صلاحيت);

    size_t سائيز_r      = حجم(dma_ring);
    size_t وٿي_desc = align_مٿي(سائيز_r, HEX_L2_سٽ_سائيز);
    size_t سائيز_desc   = صلاحيت * حجم(dma_descriptor_2d);
    size_t وٿي_معلومات = align_مٿي(وٿي_desc + سائيز_desc, HEX_L2_سٽ_سائيز);
    size_t سائيز_معلومات   = صلاحيت * حجم(dma_معلومات);

    أرجع وٿي_معلومات + سائيز_معلومات;
}

ساكن مضمن dma_ring * dma_ring_init(فراغ * ptr, size_t صلاحيت, بنية htp_thread_trace * trace) {
    صلاحيت = pow2_ceil(صلاحيت);

    size_t سائيز_r      = حجم(dma_ring);
    size_t وٿي_desc = align_مٿي(سائيز_r, HEX_L2_سٽ_سائيز);
    size_t سائيز_desc   = صلاحيت * حجم(dma_descriptor_2d);
    size_t وٿي_معلومات = align_مٿي(وٿي_desc + سائيز_desc, HEX_L2_سٽ_سائيز);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->صلاحيت  = صلاحيت;
    r->idx_نقاب  = صلاحيت - 1;
    r->ڌڪو_idx  = 0;
    r->ڪڍو_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + وٿي_desc);
    r->معلومات      = (dma_معلومات *) ((uintptr_t) ptr + وٿي_معلومات);
    r->پڇ      = &r->desc[صلاحيت - 1];

    أرجع r;
}

size_t dma_قطار_sizeof(size_t صلاحيت) {
    size_t سائيز_q    = حجم(dma_قطار);
    size_t وٿي_r0 = align_مٿي(سائيز_q, HEX_L2_سٽ_سائيز);
    size_t سائيز_r0   = dma_ring_sizeof(صلاحيت);
    size_t وٿي_r1 = align_مٿي(وٿي_r0 + سائيز_r0, HEX_L2_سٽ_سائيز);
    size_t سائيز_r1   = dma_ring_sizeof(DMA_FALLBACK_صلاحيت);

    أرجع وٿي_r1 + سائيز_r1;
}

size_t dma_قطار_alignof(فراغ) {
    أرجع HEX_L2_سٽ_سائيز;
}

dma_قطار_t dma_قطار_init(فراغ * ptr, size_t صلاحيت, بنية htp_thread_trace * trace) {
    size_t ڪل_سائيز = dma_قطار_sizeof(صلاحيت);
    memset(ptr, 0, ڪل_سائيز);

    dma_قطار * q = (dma_قطار *) ptr;

    size_t سائيز_q    = حجم(dma_قطار);
    size_t وٿي_r0 = align_مٿي(سائيز_q, HEX_L2_سٽ_سائيز);
    size_t سائيز_r0   = dma_ring_sizeof(صلاحيت);
    size_t وٿي_r1 = align_مٿي(وٿي_r0 + سائيز_r0, HEX_L2_سٽ_سائيز);

    q->ring0 = dma_ring_init((فراغ *) ((uintptr_t) ptr + وٿي_r0), صلاحيت, trace);
    q->ring1 = dma_ring_init((فراغ *) ((uintptr_t) ptr + وٿي_r1), DMA_FALLBACK_صلاحيت, trace);
    q->alias = false;

    FARF(مٿاهون_2, "dma-queue: capacity %u, unified memory size %zu\n", (بنانشان) صلاحيت, ڪل_سائيز);

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

bool dma_قطار_ڌڪو_fallback_2d(dma_قطار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t صف_سائيز, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ڌڪو_idx + 1) & r0->idx_نقاب) == r0->ڪڍو_idx) {
        أرجع false;
    }

    r1->پڇ = r0->پڇ;

    size_t rem_صفون    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_صفون > 0) {
        ثابت uint32_t cur_صفون = MIN(rem_صفون, DMA_MAX_NROWS);
        dma_معلومات cur_معلومات = dma_make_معلومات(cur_dst, cur_src);
        إذا (!dma_ring_ڌڪو_single_2d(r1, cur_معلومات, dst_stride, src_stride, صف_سائيز, cur_صفون)) {
            dma_ring_flush(r1);
            dma_ring_ڌڪو_single_2d(r1, cur_معلومات, dst_stride, src_stride, صف_سائيز, cur_صفون);
        }
        cur_dst  += cur_صفون * dst_stride;
        cur_src  += cur_صفون * src_stride;
        rem_صفون -= cur_صفون;
    }

    dma_ring_flush(r1);
    r0->پڇ = r1->پڇ;

    أرجع dma_ring_ڌڪو_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_قطار_ڌڪو_fallback_contig(dma_قطار * q, dma_معلومات ddata, size_t ڪل) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ڌڪو_idx + 1) & r0->idx_نقاب) == r0->ڪڍو_idx) {
        أرجع false;
    }

    r1->پڇ = r0->پڇ;

    size_t rem_بائيٽون   = ڪل;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_بائيٽون > 0) {
        ثابت uint32_t cur_بائيٽون = MIN(rem_بائيٽون, DMA_محفوظ_CHUNK_سائيز);
        dma_معلومات cur_معلومات = dma_make_معلومات(cur_dst, cur_src);
        إذا (!dma_ring_ڌڪو_single_contig(r1, cur_معلومات, cur_بائيٽون)) {
            dma_ring_flush(r1);
            dma_ring_ڌڪو_single_contig(r1, cur_معلومات, cur_بائيٽون);
        }
        cur_dst   += cur_بائيٽون;
        cur_src   += cur_بائيٽون;
        rem_بائيٽون -= cur_بائيٽون;
    }

    dma_ring_flush(r1);
    r0->پڇ = r1->پڇ;

    أرجع dma_ring_ڌڪو_single_1d(r0, ddata, /*size=*/ 0);
}

#إذا __HVX_ARCH__ < 75

bool dma_قطار_ڌڪو_fallback_1d(dma_قطار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t صف_سائيز, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ڌڪو_idx + 1) & r0->idx_نقاب) == r0->ڪڍو_idx) {
        أرجع false;
    }

    r1->پڇ = r0->پڇ;

    size_t rem_صفون    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_صفون > 0) {
        dma_معلومات cur_معلومات = dma_make_معلومات(cur_dst, cur_src);
        إذا (!dma_ring_ڌڪو_single_1d(r1, cur_معلومات, صف_سائيز)) {
            dma_ring_flush(r1);
            dma_ring_ڌڪو_single_1d(r1, cur_معلومات, صف_سائيز);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_صفون -= 1;
    }

    dma_ring_flush(r1);
    r0->پڇ = r1->پڇ;

    أرجع dma_ring_ڌڪو_single_1d(r0, ddata, /*size=*/ 0);
}

#جيڪڏهن_پورو
