#ضمن "dma-queue.h"

#ضمن <stdbool.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#رهنمود clang diagnostic ignored "-Wunused-function"

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

ساكن مضمن uintptr_t align_به_بالا(uintptr_t addr, size_t align) {
    أرجع (addr + align - 1) & ~(align - 1);
}

ساكن مضمن size_t dma_ring_sizeof(size_t ظرفیت) {
    ظرفیت = pow2_ceil(ظرفیت);

    size_t اندازه_r      = حجم(dma_ring);
    size_t جابه_جایی_desc = align_به_بالا(اندازه_r, HEX_L2_سطر_اندازه);
    size_t اندازه_desc   = ظرفیت * حجم(dma_descriptor_2d);
    size_t جابه_جایی_داده = align_به_بالا(جابه_جایی_desc + اندازه_desc, HEX_L2_سطر_اندازه);
    size_t اندازه_داده   = ظرفیت * حجم(dma_داده);

    أرجع جابه_جایی_داده + اندازه_داده;
}

ساكن مضمن dma_ring * dma_ring_init(فراغ * ptr, size_t ظرفیت, بنية htp_thread_trace * trace) {
    ظرفیت = pow2_ceil(ظرفیت);

    size_t اندازه_r      = حجم(dma_ring);
    size_t جابه_جایی_desc = align_به_بالا(اندازه_r, HEX_L2_سطر_اندازه);
    size_t اندازه_desc   = ظرفیت * حجم(dma_descriptor_2d);
    size_t جابه_جایی_داده = align_به_بالا(جابه_جایی_desc + اندازه_desc, HEX_L2_سطر_اندازه);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->ظرفیت  = ظرفیت;
    r->idx_ماسک  = ظرفیت - 1;
    r->افزودن_idx  = 0;
    r->برداشت_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + جابه_جایی_desc);
    r->داده      = (dma_داده *) ((uintptr_t) ptr + جابه_جایی_داده);
    r->دم      = &r->desc[ظرفیت - 1];

    أرجع r;
}

size_t dma_صف_sizeof(size_t ظرفیت) {
    size_t اندازه_q    = حجم(dma_صف);
    size_t جابه_جایی_r0 = align_به_بالا(اندازه_q, HEX_L2_سطر_اندازه);
    size_t اندازه_r0   = dma_ring_sizeof(ظرفیت);
    size_t جابه_جایی_r1 = align_به_بالا(جابه_جایی_r0 + اندازه_r0, HEX_L2_سطر_اندازه);
    size_t اندازه_r1   = dma_ring_sizeof(DMA_FALLBACK_ظرفیت);

    أرجع جابه_جایی_r1 + اندازه_r1;
}

size_t dma_صف_alignof(فراغ) {
    أرجع HEX_L2_سطر_اندازه;
}

dma_صف_t dma_صف_init(فراغ * ptr, size_t ظرفیت, بنية htp_thread_trace * trace) {
    size_t مجموع_اندازه = dma_صف_sizeof(ظرفیت);
    memset(ptr, 0, مجموع_اندازه);

    dma_صف * q = (dma_صف *) ptr;

    size_t اندازه_q    = حجم(dma_صف);
    size_t جابه_جایی_r0 = align_به_بالا(اندازه_q, HEX_L2_سطر_اندازه);
    size_t اندازه_r0   = dma_ring_sizeof(ظرفیت);
    size_t جابه_جایی_r1 = align_به_بالا(جابه_جایی_r0 + اندازه_r0, HEX_L2_سطر_اندازه);

    q->ring0 = dma_ring_init((فراغ *) ((uintptr_t) ptr + جابه_جایی_r0), ظرفیت, trace);
    q->ring1 = dma_ring_init((فراغ *) ((uintptr_t) ptr + جابه_جایی_r1), DMA_FALLBACK_ظرفیت, trace);
    q->alias = false;

    FARF(بالا_2, "dma-queue: capacity %u, unified memory size %zu\n", (بیعلامت) ظرفیت, مجموع_اندازه);

    أرجع q;
}

فراغ dma_صف_free(dma_صف_t q) {
    (فراغ) q;
}

size_t dma_صف_alias_sizeof(فراغ) {
    أرجع حجم(dma_صف);
}

dma_صف_t dma_صف_alias_init(فراغ * ptr, dma_صف_t main_q) {
    dma_صف * q = (dma_صف *) ptr;
    memset(q, 0, حجم(dma_صف));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    أرجع q;
}

فراغ dma_صف_alias_free(dma_صف_t q) {
    (فراغ) q;
}

bool dma_صف_افزودن_fallback_2d(dma_صف * q, dma_داده ddata, size_t dst_stride, size_t src_stride, size_t ردیف_اندازه, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->افزودن_idx + 1) & r0->idx_ماسک) == r0->برداشت_idx) {
        أرجع false;
    }

    r1->دم = r0->دم;

    size_t rem_ردیفها    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_ردیفها > 0) {
        ثابت uint32_t cur_ردیفها = MIN(rem_ردیفها, DMA_MAX_NROWS);
        dma_داده cur_داده = dma_make_داده(cur_dst, cur_src);
        إذا (!dma_ring_افزودن_single_2d(r1, cur_داده, dst_stride, src_stride, ردیف_اندازه, cur_ردیفها)) {
            dma_ring_flush(r1);
            dma_ring_افزودن_single_2d(r1, cur_داده, dst_stride, src_stride, ردیف_اندازه, cur_ردیفها);
        }
        cur_dst  += cur_ردیفها * dst_stride;
        cur_src  += cur_ردیفها * src_stride;
        rem_ردیفها -= cur_ردیفها;
    }

    dma_ring_flush(r1);
    r0->دم = r1->دم;

    أرجع dma_ring_افزودن_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_صف_افزودن_fallback_contig(dma_صف * q, dma_داده ddata, size_t مجموع) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->افزودن_idx + 1) & r0->idx_ماسک) == r0->برداشت_idx) {
        أرجع false;
    }

    r1->دم = r0->دم;

    size_t rem_بایتها   = مجموع;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_بایتها > 0) {
        ثابت uint32_t cur_بایتها = MIN(rem_بایتها, DMA_امن_CHUNK_اندازه);
        dma_داده cur_داده = dma_make_داده(cur_dst, cur_src);
        إذا (!dma_ring_افزودن_single_contig(r1, cur_داده, cur_بایتها)) {
            dma_ring_flush(r1);
            dma_ring_افزودن_single_contig(r1, cur_داده, cur_بایتها);
        }
        cur_dst   += cur_بایتها;
        cur_src   += cur_بایتها;
        rem_بایتها -= cur_بایتها;
    }

    dma_ring_flush(r1);
    r0->دم = r1->دم;

    أرجع dma_ring_افزودن_single_1d(r0, ddata, /*size=*/ 0);
}

#إذا __HVX_ARCH__ < 75

bool dma_صف_افزودن_fallback_1d(dma_صف * q, dma_داده ddata, size_t dst_stride, size_t src_stride, size_t ردیف_اندازه, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->افزودن_idx + 1) & r0->idx_ماسک) == r0->برداشت_idx) {
        أرجع false;
    }

    r1->دم = r0->دم;

    size_t rem_ردیفها    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_ردیفها > 0) {
        dma_داده cur_داده = dma_make_داده(cur_dst, cur_src);
        إذا (!dma_ring_افزودن_single_1d(r1, cur_داده, ردیف_اندازه)) {
            dma_ring_flush(r1);
            dma_ring_افزودن_single_1d(r1, cur_داده, ردیف_اندازه);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_ردیفها -= 1;
    }

    dma_ring_flush(r1);
    r0->دم = r1->دم;

    أرجع dma_ring_افزودن_single_1d(r0, ddata, /*size=*/ 0);
}

#پایان_اگر
