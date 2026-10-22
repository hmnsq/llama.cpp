#ضمن "dma-queue.h"

#ضمن <stdbool.h>
#ضمن <stdlib.h>
#ضمن <string.h>

#لارښوونه clang diagnostic ignored "-Wunused-function"

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

ساكن مضمن uintptr_t align_پورته(uintptr_t addr, size_t align) {
    أرجع (addr + align - 1) & ~(align - 1);
}

ساكن مضمن size_t dma_ring_sizeof(size_t ظرفیت) {
    ظرفیت = pow2_ceil(ظرفیت);

    size_t اندازه_r      = حجم(dma_ring);
    size_t دننه_واټن_desc = align_پورته(اندازه_r, HEX_L2_کرښه_اندازه);
    size_t اندازه_desc   = ظرفیت * حجم(dma_descriptor_2d);
    size_t دننه_واټن_معلومات = align_پورته(دننه_واټن_desc + اندازه_desc, HEX_L2_کرښه_اندازه);
    size_t اندازه_معلومات   = ظرفیت * حجم(dma_معلومات);

    أرجع دننه_واټن_معلومات + اندازه_معلومات;
}

ساكن مضمن dma_ring * dma_ring_init(فراغ * ptr, size_t ظرفیت, بنية htp_thread_trace * trace) {
    ظرفیت = pow2_ceil(ظرفیت);

    size_t اندازه_r      = حجم(dma_ring);
    size_t دننه_واټن_desc = align_پورته(اندازه_r, HEX_L2_کرښه_اندازه);
    size_t اندازه_desc   = ظرفیت * حجم(dma_descriptor_2d);
    size_t دننه_واټن_معلومات = align_پورته(دننه_واټن_desc + اندازه_desc, HEX_L2_کرښه_اندازه);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->ظرفیت  = ظرفیت;
    r->idx_ماسک  = ظرفیت - 1;
    r->ټېل_idx  = 0;
    r->راوباسه_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + دننه_واټن_desc);
    r->معلومات      = (dma_معلومات *) ((uintptr_t) ptr + دننه_واټن_معلومات);
    r->لکۍ      = &r->desc[ظرفیت - 1];

    أرجع r;
}

size_t dma_کتار_sizeof(size_t ظرفیت) {
    size_t اندازه_q    = حجم(dma_کتار);
    size_t دننه_واټن_r0 = align_پورته(اندازه_q, HEX_L2_کرښه_اندازه);
    size_t اندازه_r0   = dma_ring_sizeof(ظرفیت);
    size_t دننه_واټن_r1 = align_پورته(دننه_واټن_r0 + اندازه_r0, HEX_L2_کرښه_اندازه);
    size_t اندازه_r1   = dma_ring_sizeof(DMA_FALLBACK_ظرفیت);

    أرجع دننه_واټن_r1 + اندازه_r1;
}

size_t dma_کتار_alignof(فراغ) {
    أرجع HEX_L2_کرښه_اندازه;
}

dma_کتار_t dma_کتار_init(فراغ * ptr, size_t ظرفیت, بنية htp_thread_trace * trace) {
    size_t ټول_اندازه = dma_کتار_sizeof(ظرفیت);
    memset(ptr, 0, ټول_اندازه);

    dma_کتار * q = (dma_کتار *) ptr;

    size_t اندازه_q    = حجم(dma_کتار);
    size_t دننه_واټن_r0 = align_پورته(اندازه_q, HEX_L2_کرښه_اندازه);
    size_t اندازه_r0   = dma_ring_sizeof(ظرفیت);
    size_t دننه_واټن_r1 = align_پورته(دننه_واټن_r0 + اندازه_r0, HEX_L2_کرښه_اندازه);

    q->ring0 = dma_ring_init((فراغ *) ((uintptr_t) ptr + دننه_واټن_r0), ظرفیت, trace);
    q->ring1 = dma_ring_init((فراغ *) ((uintptr_t) ptr + دننه_واټن_r1), DMA_FALLBACK_ظرفیت, trace);
    q->alias = false;

    FARF(لوړ_2, "dma-queue: capacity %u, unified memory size %zu\n", (بینښه) ظرفیت, ټول_اندازه);

    أرجع q;
}

فراغ dma_کتار_free(dma_کتار_t q) {
    (فراغ) q;
}

size_t dma_کتار_alias_sizeof(فراغ) {
    أرجع حجم(dma_کتار);
}

dma_کتار_t dma_کتار_alias_init(فراغ * ptr, dma_کتار_t main_q) {
    dma_کتار * q = (dma_کتار *) ptr;
    memset(q, 0, حجم(dma_کتار));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    أرجع q;
}

فراغ dma_کتار_alias_free(dma_کتار_t q) {
    (فراغ) q;
}

bool dma_کتار_ټېل_fallback_2d(dma_کتار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t لیکه_اندازه, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ټېل_idx + 1) & r0->idx_ماسک) == r0->راوباسه_idx) {
        أرجع false;
    }

    r1->لکۍ = r0->لکۍ;

    size_t rem_لیکې    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_لیکې > 0) {
        ثابت uint32_t cur_لیکې = MIN(rem_لیکې, DMA_MAX_NROWS);
        dma_معلومات cur_معلومات = dma_make_معلومات(cur_dst, cur_src);
        إذا (!dma_ring_ټېل_single_2d(r1, cur_معلومات, dst_stride, src_stride, لیکه_اندازه, cur_لیکې)) {
            dma_ring_flush(r1);
            dma_ring_ټېل_single_2d(r1, cur_معلومات, dst_stride, src_stride, لیکه_اندازه, cur_لیکې);
        }
        cur_dst  += cur_لیکې * dst_stride;
        cur_src  += cur_لیکې * src_stride;
        rem_لیکې -= cur_لیکې;
    }

    dma_ring_flush(r1);
    r0->لکۍ = r1->لکۍ;

    أرجع dma_ring_ټېل_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_کتار_ټېل_fallback_contig(dma_کتار * q, dma_معلومات ddata, size_t ټول) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ټېل_idx + 1) & r0->idx_ماسک) == r0->راوباسه_idx) {
        أرجع false;
    }

    r1->لکۍ = r0->لکۍ;

    size_t rem_بایټونه   = ټول;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_بایټونه > 0) {
        ثابت uint32_t cur_بایټونه = MIN(rem_بایټونه, DMA_خوندي_CHUNK_اندازه);
        dma_معلومات cur_معلومات = dma_make_معلومات(cur_dst, cur_src);
        إذا (!dma_ring_ټېل_single_contig(r1, cur_معلومات, cur_بایټونه)) {
            dma_ring_flush(r1);
            dma_ring_ټېل_single_contig(r1, cur_معلومات, cur_بایټونه);
        }
        cur_dst   += cur_بایټونه;
        cur_src   += cur_بایټونه;
        rem_بایټونه -= cur_بایټونه;
    }

    dma_ring_flush(r1);
    r0->لکۍ = r1->لکۍ;

    أرجع dma_ring_ټېل_single_1d(r0, ddata, /*size=*/ 0);
}

#إذا __HVX_ARCH__ < 75

bool dma_کتار_ټېل_fallback_1d(dma_کتار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t لیکه_اندازه, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    إذا (((r0->ټېل_idx + 1) & r0->idx_ماسک) == r0->راوباسه_idx) {
        أرجع false;
    }

    r1->لکۍ = r0->لکۍ;

    size_t rem_لیکې    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    طالما (rem_لیکې > 0) {
        dma_معلومات cur_معلومات = dma_make_معلومات(cur_dst, cur_src);
        إذا (!dma_ring_ټېل_single_1d(r1, cur_معلومات, لیکه_اندازه)) {
            dma_ring_flush(r1);
            dma_ring_ټېل_single_1d(r1, cur_معلومات, لیکه_اندازه);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_لیکې -= 1;
    }

    dma_ring_flush(r1);
    r0->لکۍ = r1->لکۍ;

    أرجع dma_ring_ټېل_single_1d(r0, ddata, /*size=*/ 0);
}

#د_که_پای
