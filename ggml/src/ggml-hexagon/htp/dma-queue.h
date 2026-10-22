#که_نه_تعریف_شوی HTP_DMA_H
#تعریف_کړه HTP_DMA_H

#ضمن <HAP_farf.h>
#ضمن <hexagon_types.h>
#ضمن <assert.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <string.h>
#ضمن "hex-utils.h"

#ضمن "hex-profile.h"

#که_تعریف_شوی __cplusplus
خارجي "C" {
#د_که_پای

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
تعريفنوع بنية dma_descriptor_1d_s {
    فراغ *   بل;
    uint32_t اندازه:24;
    uint32_t desc_اندازه:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t بشپړ:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#إذا __HVX_ARCH__ < 75

تعريفنوع بنية dma_descriptor_2d_s {
    فراغ *   بل;
    uint32_t reserved0:24;
    uint32_t desc_اندازه:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t بشپړ:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t لیکه_اندازه:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_دننه_واټن:16;
    uint32_t dst_دننه_واټن:16;
} dma_descriptor_2d;

#وإلا

تعريفنوع بنية dma_descriptor_2d_s {
    فراغ *   بل;
    uint32_t dst_stride:24;
    uint32_t desc_اندازه:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t بشپړ:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#إذا __HVX_ARCH__ > 79
    uint32_t src_پاسنی:8;
    uint32_t dst_پاسنی:8;
    uint32_t ویش:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#وإلا
    uint32_t reserved0:24;
#د_که_پای
    uint32_t لیکه_اندازه:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t دننه_واټن:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#د_که_پای

#إذا __HVX_ARCH__ > 79
تعريفنوع uint64_t dma_addr_t;
#وإلا
تعريفنوع uint32_t dma_addr_t;
#د_که_پای

تعريفنوع بنية {
    dma_addr_t dst;
    dma_addr_t src;
} dma_معلومات;

// Hardware descriptor field limits
#تعریف_کړه DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#تعریف_کړه DMA_MAX_اندازه_16_B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#تعریف_کړه DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#تعریف_کړه DMA_MAX_اندازه_24_B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#تعریف_کړه DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#تعریف_کړه DMA_خوندي_CHUNK_اندازه    0x00F00000u    // ~15MB safe contiguous chunk size

#إذا __HVX_ARCH__ < 75
#تعریف_کړه DMA_MAX_2_D_لیکه_اندازه    DMA_MAX_اندازه_16_B
#تعریف_کړه DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#وإلا
#تعریف_کړه DMA_MAX_2_D_لیکه_اندازه    DMA_MAX_اندازه_24_B
#تعریف_کړه DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#د_که_پای

#تعریف_کړه DMA_FALLBACK_ظرفیت  16u            // descriptors in secondary fallback ring

تعريفنوع بنية dma_ring_s dma_ring;
بنية dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * لکۍ;      // tail pointer
    dma_معلومات          * معلومات;      // dst/src data
    uint32_t            ټېل_idx;
    uint32_t            راوباسه_idx;
    uint32_t            ظرفیت;
    uint32_t            idx_ماسک;
    بنية htp_thread_trace * trace;
};

تعريفنوع بنية dma_کتار_s dma_کتار;
تعريفنوع dma_کتار * dma_کتار_t;

بنية dma_کتار_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_کتار_sizeof(size_t ظرفیت);
size_t      dma_کتار_alignof(فراغ);
dma_کتار_t dma_کتار_init(فراغ * ptr, size_t ظرفیت, بنية htp_thread_trace * trace);
فراغ        dma_کتار_free(dma_کتار_t q);

size_t      dma_کتار_alias_sizeof(فراغ);
dma_کتار_t dma_کتار_alias_init(فراغ * ptr, dma_کتار_t main_q);
فراغ        dma_کتار_alias_free(dma_کتار_t q);

bool        dma_کتار_ټېل_fallback_2d(dma_کتار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t لیکه_اندازه, size_t nrows);
bool        dma_کتار_ټېل_fallback_contig(dma_کتار * q, dma_معلومات ddata, size_t ټول);
#إذا __HVX_ARCH__ < 75
bool        dma_کتار_ټېل_fallback_1d(dma_کتار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t لیکه_اندازه, size_t nrows);
#د_که_پای

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
ساكن مضمن فراغ dmstart(فراغ * بل) {
    asm متطاير(" release(%0):at" : : "r"(بل));
    asm متطاير(" dmstart(%0)" : : "r"(بل));
}

ساكن مضمن فراغ dmlink(فراغ * cur, فراغ * بل) {
    asm متطاير(" release(%0):at" : : "r"(بل));
    asm متطاير(" dmlink(%0, %1)" : : "r"(cur), "r"(بل));
}

ساكن مضمن بینښه صحيح dmpoll(فراغ) {
    بینښه صحيح ret = 0;
    asm متطاير(" %0 = dmpoll" : "=r"(ret) : : "memory");
    أرجع ret;
}

ساكن مضمن بینښه صحيح dmwait(فراغ) {
    بینښه صحيح ret = 0;
    asm متطاير(" %0 = dmwait" : "=r"(ret) : : "memory");
    أرجع ret;
}

ساكن مضمن dma_معلومات dma_make_معلومات_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_معلومات d = { dst, src };
    أرجع d;
}

#تعریف_کړه dma_make_معلومات(dst, src) dma_make_معلومات_impl((dma_addr_t) (dst), (dma_addr_t) (src))

ساكن مضمن bool dma_ring_ټېل_single_1d(dma_ring * r, dma_معلومات ddata, size_t اندازه) {
#إذا __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 32) || اندازه == 0);
#د_که_پای

    إذا (((r->ټېل_idx + 1) & r->idx_ماسک) == r->راوباسه_idx) {
        أرجع false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->ټېل_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->اندازه = اندازه;

    r->معلومات[r->ټېل_idx] = ddata;

    htp_trace_event_پیل(r->trace, HTP_TRACE_EVT_DMA, r->ټېل_idx);

    إذا (اندازه) {
        desc->بل       = عدم;
        desc->desc_اندازه  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->ترتیب      = 0;
        desc->بشپړ       = 0;

        dmlink(r->لکۍ, desc);
        r->لکۍ = (dma_descriptor_2d *) desc;
    } وإلا {
        desc->desc_اندازه = 0;
        desc->بشپړ      = 1;
    }

    r->ټېل_idx = (r->ټېل_idx + 1) & r->idx_ماسک;
    أرجع true;
}

ساكن مضمن bool dma_ring_ټېل_single_2d(dma_ring * r, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t لیکه_اندازه, size_t nrows) {
#إذا __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ثابت uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ثابت uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ثابت bool is_ext     = (src_hi | dst_hi) != 0;
#د_که_پای

    إذا (((r->ټېل_idx + 1) & r->idx_ماسک) == r->راوباسه_idx) {
        أرجع false;
    }

    dma_descriptor_2d * desc = &r->desc[r->ټېل_idx];

    desc->بل           = عدم;
    desc->reserved1      = 0;
    desc->desc_اندازه      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->ترتیب          = 0;
    desc->بشپړ           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->لیکه_اندازه       = لیکه_اندازه;

#إذا __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_دننه_واټن     = 0;
    desc->dst_دننه_واټن     = 0;
#وإلا
#إذا __HVX_ARCH__ > 79
    desc->src_پاسنی      = src_hi;
    desc->dst_پاسنی      = dst_hi;
    desc->ویش     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#وإلا
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#د_که_پای
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->دننه_واټن         = 0;
#د_که_پای

    r->معلومات[r->ټېل_idx] = ddata;

    htp_trace_event_پیل(r->trace, HTP_TRACE_EVT_DMA, r->ټېل_idx);

    إذا (nrows) {
        dmlink(r->لکۍ, desc);
        r->لکۍ = desc;
    } وإلا {
        desc->بشپړ = 1;
    }

    r->ټېل_idx = (r->ټېل_idx + 1) & r->idx_ماسک;
    أرجع true;
}

#إذا __HVX_ARCH__ < 75
ساكن مضمن bool dma_ring_ټېل_single_contig(dma_ring * r, dma_معلومات ddata, size_t اندازه) {
    أرجع dma_ring_ټېل_single_1d(r, ddata, اندازه);
}
#وإلا
ساكن مضمن bool dma_ring_ټېل_single_contig(dma_ring * r, dma_معلومات ddata, size_t اندازه) {
    أرجع dma_ring_ټېل_single_2d(r, ddata, اندازه, اندازه, اندازه, 1);
}
#د_که_پای

ساكن مضمن dma_معلومات dma_ring_راوباسه(dma_ring * r) {
    dma_معلومات ddata = { 0 };

    إذا (r->ټېل_idx == r->راوباسه_idx) {
        أرجع ddata;
    }

    ddata = r->معلومات[r->راوباسه_idx];

    متطاير dma_descriptor_2d * desc = &r->desc[r->راوباسه_idx];

    // Wait for desc to complete
    إذا (!desc->بشپړ) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        طالما (!desc->بشپړ) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->راوباسه_idx);

    r->راوباسه_idx = (r->راوباسه_idx + 1) & r->idx_ماسک;
    أرجع ddata;
}

ساكن مضمن dma_معلومات dma_ring_راوباسه_nowait(dma_ring * r) {
    dma_معلومات ddata = { 0 };

    إذا (r->ټېل_idx == r->راوباسه_idx) {
        أرجع ddata;
    }

    ddata = r->معلومات[r->راوباسه_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->راوباسه_idx);

    r->راوباسه_idx = (r->راوباسه_idx + 1) & r->idx_ماسک;
    أرجع ddata;
}

ساكن مضمن bool dma_ring_تشه(dma_ring * r) {
    أرجع r->ټېل_idx == r->راوباسه_idx;
}

ساكن مضمن فراغ dma_ring_flush(dma_ring * r) {
    طالما (!dma_ring_تشه(r)) {
        dma_ring_راوباسه(r);
    }
}

ساكن مضمن uint32_t dma_ring_ژوروالی(dma_ring * r) {
    أرجع (r->ټېل_idx - r->راوباسه_idx) & r->idx_ماسک;
}

ساكن مضمن uint32_t dma_ring_ظرفیت(dma_ring * r) {
    أرجع r->ظرفیت;
}

ساكن مضمن bool dma_کتار_ټېل_single_1d(dma_کتار * q, dma_معلومات ddata, size_t اندازه) {
    أرجع dma_ring_ټېل_single_1d(q->ring0, ddata, اندازه);
}

ساكن مضمن bool dma_کتار_ټېل_single_2d(dma_کتار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t لیکه_اندازه, size_t nrows) {
    أرجع dma_ring_ټېل_single_2d(q->ring0, ddata, dst_stride, src_stride, لیکه_اندازه, nrows);
}

ساكن مضمن dma_معلومات dma_کتار_راوباسه(dma_کتار * q) {
    أرجع dma_ring_راوباسه(q->ring0);
}

ساكن مضمن dma_معلومات dma_کتار_راوباسه_nowait(dma_کتار * q) {
    أرجع dma_ring_راوباسه_nowait(q->ring0);
}

ساكن مضمن bool dma_کتار_تشه(dma_کتار * q) {
    أرجع dma_ring_تشه(q->ring0);
}

ساكن مضمن فراغ dma_کتار_flush(dma_کتار * q) {
    dma_ring_flush(q->ring0);
}

ساكن مضمن uint32_t dma_کتار_ژوروالی(dma_کتار * q) {
    أرجع dma_ring_ژوروالی(q->ring0);
}

ساكن مضمن uint32_t dma_کتار_ظرفیت(dma_کتار * q) {
    أرجع dma_ring_ظرفیت(q->ring0);
}

ساكن مضمن bool dma_کتار_ټېل(dma_کتار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t لیکه_اندازه, size_t nrows) {
    إذا (__builtin_expect(nrows == 0, 0)) {
        أرجع dma_ring_ټېل_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    إذا (nrows == 1 || !((لیکه_اندازه ^ src_stride) | (لیکه_اندازه ^ dst_stride))) {
        ثابت size_t ټول = لیکه_اندازه * nrows;
        إذا (__builtin_expect(ټول <= DMA_MAX_اندازه_24_B, 1)) {
            أرجع dma_ring_ټېل_single_contig(q->ring0, ddata, ټول);
        }
        أرجع dma_کتار_ټېل_fallback_contig(q, ddata, ټول);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    إذا (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (لیکه_اندازه | src_stride | dst_stride) <= DMA_MAX_2_D_لیکه_اندازه, 1)) {
        أرجع dma_ring_ټېل_single_2d(q->ring0, ddata, dst_stride, src_stride, لیکه_اندازه, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#إذا __HVX_ARCH__ < 75
    إذا (لیکه_اندازه <= DMA_MAX_اندازه_16_B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        أرجع dma_کتار_ټېل_fallback_2d(q, ddata, dst_stride, src_stride, لیکه_اندازه, nrows);
    }
    أرجع dma_کتار_ټېل_fallback_1d(q, ddata, dst_stride, src_stride, لیکه_اندازه, nrows);
#وإلا
    أرجع dma_کتار_ټېل_fallback_2d(q, ddata, dst_stride, src_stride, لیکه_اندازه, nrows);
#د_که_پای
}

ساكن مضمن فراغ dma_sync_read(dma_کتار * dma_q, فراغ * dst, dma_addr_t src, size_t بایټونه) {
    ثابت uint32_t b = (uint32_t) بایټونه;
    إذا (b > 0) {
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst, src), b, b, b, 1);
        dma_کتار_راوباسه(dma_q);
    }
}

ساكن مضمن فراغ dma_sync_write(dma_کتار * dma_q, dma_addr_t dst, ثابت فراغ * src, size_t بایټونه) {
    ثابت uint32_t b = (uint32_t) بایټونه;
    إذا (b > 0) {
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst, src), b, b, b, 1);
        dma_کتار_راوباسه(dma_q);
    }
}

#تعریف_کړه DMA_CACHE_MAX_اندازه 256U

// Fully assoc LRU cache
تعريفنوع بنية {
    uint8_t *بنسټ;
    uint32_t کرښه_اندازه;
    uint32_t ظرفیت;
    dma_addr_t src[DMA_CACHE_MAX_اندازه];
    uint16_t age[DMA_CACHE_MAX_اندازه];
} dma_cache_fa;

ساكن مضمن فراغ dma_cache_fa_init(dma_cache_fa *c, uint8_t *بنسټ, uint32_t کرښه_اندازه, uint32_t ظرفیت)
{
    c->ظرفیت  = (ظرفیت > DMA_CACHE_MAX_اندازه) ? DMA_CACHE_MAX_اندازه : ظرفیت;
    c->بنسټ      = بنسټ;
    c->کرښه_اندازه = کرښه_اندازه;

    لكل (بینښه i=0; i < c->ظرفیت; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

ساكن مضمن bool dma_cache_fa_ټېل(dma_کتار *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t لیکه_اندازه, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    لكل (بینښه i=0; i < c->ظرفیت; i++) {
        إذا (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->بنسټ + (i * c->کرښه_اندازه); nrows = 0; // dummy dma
        } وإلا {
            c->age[i]++;
            إذا (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    إذا (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->بنسټ + o_idx * c->کرښه_اندازه; // normal nrows dma
        أرجع dma_کتار_ټېل(q, dma_make_معلومات(dst, src_addr), dst_stride, src_stride, لیکه_اندازه, nrows);
    }

    أرجع dma_کتار_ټېل_single_1d(q, dma_make_معلومات(dst, src_addr), 0);
}

// Direct mapped cache
تعريفنوع بنية {
    uint8_t *بنسټ;
    uint32_t کرښه_اندازه;
    uint32_t ظرفیت;
    uint32_t idx_ماسک;
    dma_addr_t src[DMA_CACHE_MAX_اندازه];
} dma_cache_dm;

ساكن مضمن فراغ dma_cache_dm_init(dma_cache_dm *c, uint8_t *بنسټ, uint32_t کرښه_اندازه, uint32_t ظرفیت)
{
    c->ظرفیت  = (ظرفیت > DMA_CACHE_MAX_اندازه) ? DMA_CACHE_MAX_اندازه : ظرفیت;
    c->idx_ماسک  = c->ظرفیت - 1;
    c->بنسټ      = بنسټ;
    c->کرښه_اندازه = کرښه_اندازه;

    لكل (بینښه i=0; i < c->ظرفیت; i++) {
        c->src[i] = 0;
    }
}

ساكن مضمن bool dma_cache_dm_ټېل(dma_کتار *q, dma_cache_dm *c, uint32_t خونه, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t لیکه_اندازه, uint32_t nrows)
{
    ثابت uint32_t i = خونه & c->idx_ماسک;
    uint8_t * dst = c->بنسټ + (i * c->کرښه_اندازه);

    إذا (c->src[i] == src_addr) {
        أرجع dma_کتار_ټېل_single_1d(q, dma_make_معلومات(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    أرجع dma_کتار_ټېل(q, dma_make_معلومات(dst, src_addr), dst_stride, src_stride, لیکه_اندازه, nrows);
}

#که_تعریف_شوی __cplusplus
}  // extern "C"
#د_که_پای

#د_که_پای /* HTP_DMA_H */
