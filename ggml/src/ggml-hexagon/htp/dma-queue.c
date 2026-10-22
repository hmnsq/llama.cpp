#شامل "dma-queue.h"

#شامل <stdbool.h>
#شامل <stdlib.h>
#شامل <string.h>

#ہدایت clang diagnostic ignored "-Wunused-function"

ساکن اندرونی uint32_t pow2_ceil(uint32_t x) {
    اگر (x <= 1) {
        واپس 1;
    }
    صحیح p = 2;
    x--;
    جب (x >>= 1) {
        p <<= 1;
    }
    واپس p;
}

ساکن اندرونی uintptr_t align_اوپر(uintptr_t addr, size_t align) {
    واپس (addr + align - 1) & ~(align - 1);
}

ساکن اندرونی size_t dma_ring_sizeof(size_t استعداد) {
    استعداد = pow2_ceil(استعداد);

    size_t سائز_r      = حجم(dma_ring);
    size_t انحراف_desc = align_اوپر(سائز_r, HEX_L2_سطر_سائز);
    size_t سائز_desc   = استعداد * حجم(dma_descriptor_2d);
    size_t انحراف_ڈیٹا = align_اوپر(انحراف_desc + سائز_desc, HEX_L2_سطر_سائز);
    size_t سائز_ڈیٹا   = استعداد * حجم(dma_ڈیٹا);

    واپس انحراف_ڈیٹا + سائز_ڈیٹا;
}

ساکن اندرونی dma_ring * dma_ring_init(خالی * ptr, size_t استعداد, ساخت htp_thread_trace * trace) {
    استعداد = pow2_ceil(استعداد);

    size_t سائز_r      = حجم(dma_ring);
    size_t انحراف_desc = align_اوپر(سائز_r, HEX_L2_سطر_سائز);
    size_t سائز_desc   = استعداد * حجم(dma_descriptor_2d);
    size_t انحراف_ڈیٹا = align_اوپر(انحراف_desc + سائز_desc, HEX_L2_سطر_سائز);

    dma_ring * r = (dma_ring *) ptr;
    r->trace     = trace;
    r->استعداد  = استعداد;
    r->idx_نقاب  = استعداد - 1;
    r->ڈالو_idx  = 0;
    r->نکالو_idx   = 0;
    r->desc      = (dma_descriptor_2d *) ((uintptr_t) ptr + انحراف_desc);
    r->ڈیٹا      = (dma_ڈیٹا *) ((uintptr_t) ptr + انحراف_ڈیٹا);
    r->دم      = &r->desc[استعداد - 1];

    واپس r;
}

size_t dma_قطار_sizeof(size_t استعداد) {
    size_t سائز_q    = حجم(dma_قطار);
    size_t انحراف_r0 = align_اوپر(سائز_q, HEX_L2_سطر_سائز);
    size_t سائز_r0   = dma_ring_sizeof(استعداد);
    size_t انحراف_r1 = align_اوپر(انحراف_r0 + سائز_r0, HEX_L2_سطر_سائز);
    size_t سائز_r1   = dma_ring_sizeof(DMA_FALLBACK_استعداد);

    واپس انحراف_r1 + سائز_r1;
}

size_t dma_قطار_alignof(خالی) {
    واپس HEX_L2_سطر_سائز;
}

dma_قطار_t dma_قطار_init(خالی * ptr, size_t استعداد, ساخت htp_thread_trace * trace) {
    size_t کل_سائز = dma_قطار_sizeof(استعداد);
    memset(ptr, 0, کل_سائز);

    dma_قطار * q = (dma_قطار *) ptr;

    size_t سائز_q    = حجم(dma_قطار);
    size_t انحراف_r0 = align_اوپر(سائز_q, HEX_L2_سطر_سائز);
    size_t سائز_r0   = dma_ring_sizeof(استعداد);
    size_t انحراف_r1 = align_اوپر(انحراف_r0 + سائز_r0, HEX_L2_سطر_سائز);

    q->ring0 = dma_ring_init((خالی *) ((uintptr_t) ptr + انحراف_r0), استعداد, trace);
    q->ring1 = dma_ring_init((خالی *) ((uintptr_t) ptr + انحراف_r1), DMA_FALLBACK_استعداد, trace);
    q->alias = false;

    FARF(بلند_2, "dma-queue: capacity %u, unified memory size %zu\n", (غیرعلامتی) استعداد, کل_سائز);

    واپس q;
}

خالی dma_قطار_free(dma_قطار_t q) {
    (خالی) q;
}

size_t dma_قطار_alias_sizeof(خالی) {
    واپس حجم(dma_قطار);
}

dma_قطار_t dma_قطار_alias_init(خالی * ptr, dma_قطار_t main_q) {
    dma_قطار * q = (dma_قطار *) ptr;
    memset(q, 0, حجم(dma_قطار));

    q->ring0 = main_q->ring0;
    q->ring1 = main_q->ring1;
    q->alias = true;

    واپس q;
}

خالی dma_قطار_alias_free(dma_قطار_t q) {
    (خالی) q;
}

bool dma_قطار_ڈالو_fallback_2d(dma_قطار * q, dma_ڈیٹا ddata, size_t dst_stride, size_t src_stride, size_t صف_سائز, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    اگر (((r0->ڈالو_idx + 1) & r0->idx_نقاب) == r0->نکالو_idx) {
        واپس false;
    }

    r1->دم = r0->دم;

    size_t rem_صفیں    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    جب (rem_صفیں > 0) {
        ثابت uint32_t cur_صفیں = MIN(rem_صفیں, DMA_MAX_NROWS);
        dma_ڈیٹا cur_ڈیٹا = dma_make_ڈیٹا(cur_dst, cur_src);
        اگر (!dma_ring_ڈالو_single_2d(r1, cur_ڈیٹا, dst_stride, src_stride, صف_سائز, cur_صفیں)) {
            dma_ring_flush(r1);
            dma_ring_ڈالو_single_2d(r1, cur_ڈیٹا, dst_stride, src_stride, صف_سائز, cur_صفیں);
        }
        cur_dst  += cur_صفیں * dst_stride;
        cur_src  += cur_صفیں * src_stride;
        rem_صفیں -= cur_صفیں;
    }

    dma_ring_flush(r1);
    r0->دم = r1->دم;

    واپس dma_ring_ڈالو_single_2d(r0, ddata, 0, 0, 0, /*nrows=*/ 0);
}

bool dma_قطار_ڈالو_fallback_contig(dma_قطار * q, dma_ڈیٹا ddata, size_t کل) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    اگر (((r0->ڈالو_idx + 1) & r0->idx_نقاب) == r0->نکالو_idx) {
        واپس false;
    }

    r1->دم = r0->دم;

    size_t rem_بائٹیں   = کل;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    جب (rem_بائٹیں > 0) {
        ثابت uint32_t cur_بائٹیں = MIN(rem_بائٹیں, DMA_محفوظ_CHUNK_سائز);
        dma_ڈیٹا cur_ڈیٹا = dma_make_ڈیٹا(cur_dst, cur_src);
        اگر (!dma_ring_ڈالو_single_contig(r1, cur_ڈیٹا, cur_بائٹیں)) {
            dma_ring_flush(r1);
            dma_ring_ڈالو_single_contig(r1, cur_ڈیٹا, cur_بائٹیں);
        }
        cur_dst   += cur_بائٹیں;
        cur_src   += cur_بائٹیں;
        rem_بائٹیں -= cur_بائٹیں;
    }

    dma_ring_flush(r1);
    r0->دم = r1->دم;

    واپس dma_ring_ڈالو_single_1d(r0, ddata, /*size=*/ 0);
}

#اگر __HVX_ARCH__ < 75

bool dma_قطار_ڈالو_fallback_1d(dma_قطار * q, dma_ڈیٹا ddata, size_t dst_stride, size_t src_stride, size_t صف_سائز, size_t nrows) {
    dma_ring * r0 = q->ring0;
    dma_ring * r1 = q->ring1;

    اگر (((r0->ڈالو_idx + 1) & r0->idx_نقاب) == r0->نکالو_idx) {
        واپس false;
    }

    r1->دم = r0->دم;

    size_t rem_صفیں    = nrows;
    dma_addr_t cur_dst = ddata.dst;
    dma_addr_t cur_src = ddata.src;

    جب (rem_صفیں > 0) {
        dma_ڈیٹا cur_ڈیٹا = dma_make_ڈیٹا(cur_dst, cur_src);
        اگر (!dma_ring_ڈالو_single_1d(r1, cur_ڈیٹا, صف_سائز)) {
            dma_ring_flush(r1);
            dma_ring_ڈالو_single_1d(r1, cur_ڈیٹا, صف_سائز);
        }
        cur_dst  += dst_stride;
        cur_src  += src_stride;
        rem_صفیں -= 1;
    }

    dma_ring_flush(r1);
    r0->دم = r1->دم;

    واپس dma_ring_ڈالو_single_1d(r0, ddata, /*size=*/ 0);
}

#اگر_ختم
