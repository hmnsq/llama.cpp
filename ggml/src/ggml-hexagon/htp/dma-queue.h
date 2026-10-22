#اگر_تعریف_نشده HTP_DMA_H
#تعریف_کن HTP_DMA_H

#ضمن <HAP_farf.h>
#ضمن <hexagon_types.h>
#ضمن <assert.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <string.h>
#ضمن "hex-utils.h"

#ضمن "hex-profile.h"

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
تعريفنوع بنية dma_descriptor_1d_s {
    فراغ *   بعدی;
    uint32_t اندازه:24;
    uint32_t desc_اندازه:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t انجامشده:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#إذا __HVX_ARCH__ < 75

تعريفنوع بنية dma_descriptor_2d_s {
    فراغ *   بعدی;
    uint32_t reserved0:24;
    uint32_t desc_اندازه:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t انجامشده:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t ردیف_اندازه:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_جابه_جایی:16;
    uint32_t dst_جابه_جایی:16;
} dma_descriptor_2d;

#وإلا

تعريفنوع بنية dma_descriptor_2d_s {
    فراغ *   بعدی;
    uint32_t dst_stride:24;
    uint32_t desc_اندازه:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t انجامشده:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#إذا __HVX_ARCH__ > 79
    uint32_t src_فوقانی:8;
    uint32_t dst_فوقانی:8;
    uint32_t تخصیص:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#وإلا
    uint32_t reserved0:24;
#پایان_اگر
    uint32_t ردیف_اندازه:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t جابه_جایی:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#پایان_اگر

#إذا __HVX_ARCH__ > 79
تعريفنوع uint64_t dma_addr_t;
#وإلا
تعريفنوع uint32_t dma_addr_t;
#پایان_اگر

تعريفنوع بنية {
    dma_addr_t dst;
    dma_addr_t src;
} dma_داده;

// Hardware descriptor field limits
#تعریف_کن DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#تعریف_کن DMA_MAX_اندازه_16_B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#تعریف_کن DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#تعریف_کن DMA_MAX_اندازه_24_B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#تعریف_کن DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#تعریف_کن DMA_امن_CHUNK_اندازه    0x00F00000u    // ~15MB safe contiguous chunk size

#إذا __HVX_ARCH__ < 75
#تعریف_کن DMA_MAX_2_D_ردیف_اندازه    DMA_MAX_اندازه_16_B
#تعریف_کن DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#وإلا
#تعریف_کن DMA_MAX_2_D_ردیف_اندازه    DMA_MAX_اندازه_24_B
#تعریف_کن DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#پایان_اگر

#تعریف_کن DMA_FALLBACK_ظرفیت  16u            // descriptors in secondary fallback ring

تعريفنوع بنية dma_ring_s dma_ring;
بنية dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * دم;      // tail pointer
    dma_داده          * داده;      // dst/src data
    uint32_t            افزودن_idx;
    uint32_t            برداشت_idx;
    uint32_t            ظرفیت;
    uint32_t            idx_ماسک;
    بنية htp_thread_trace * trace;
};

تعريفنوع بنية dma_صف_s dma_صف;
تعريفنوع dma_صف * dma_صف_t;

بنية dma_صف_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_صف_sizeof(size_t ظرفیت);
size_t      dma_صف_alignof(فراغ);
dma_صف_t dma_صف_init(فراغ * ptr, size_t ظرفیت, بنية htp_thread_trace * trace);
فراغ        dma_صف_free(dma_صف_t q);

size_t      dma_صف_alias_sizeof(فراغ);
dma_صف_t dma_صف_alias_init(فراغ * ptr, dma_صف_t main_q);
فراغ        dma_صف_alias_free(dma_صف_t q);

bool        dma_صف_افزودن_fallback_2d(dma_صف * q, dma_داده ddata, size_t dst_stride, size_t src_stride, size_t ردیف_اندازه, size_t nrows);
bool        dma_صف_افزودن_fallback_contig(dma_صف * q, dma_داده ddata, size_t مجموع);
#إذا __HVX_ARCH__ < 75
bool        dma_صف_افزودن_fallback_1d(dma_صف * q, dma_داده ddata, size_t dst_stride, size_t src_stride, size_t ردیف_اندازه, size_t nrows);
#پایان_اگر

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
ساكن مضمن فراغ dmstart(فراغ * بعدی) {
    asm متطاير(" release(%0):at" : : "r"(بعدی));
    asm متطاير(" dmstart(%0)" : : "r"(بعدی));
}

ساكن مضمن فراغ dmlink(فراغ * cur, فراغ * بعدی) {
    asm متطاير(" release(%0):at" : : "r"(بعدی));
    asm متطاير(" dmlink(%0, %1)" : : "r"(cur), "r"(بعدی));
}

ساكن مضمن بیعلامت صحيح dmpoll(فراغ) {
    بیعلامت صحيح ret = 0;
    asm متطاير(" %0 = dmpoll" : "=r"(ret) : : "memory");
    أرجع ret;
}

ساكن مضمن بیعلامت صحيح dmwait(فراغ) {
    بیعلامت صحيح ret = 0;
    asm متطاير(" %0 = dmwait" : "=r"(ret) : : "memory");
    أرجع ret;
}

ساكن مضمن dma_داده dma_make_داده_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_داده d = { dst, src };
    أرجع d;
}

#تعریف_کن dma_make_داده(dst, src) dma_make_داده_impl((dma_addr_t) (dst), (dma_addr_t) (src))

ساكن مضمن bool dma_ring_افزودن_single_1d(dma_ring * r, dma_داده ddata, size_t اندازه) {
#إذا __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 32) || اندازه == 0);
#پایان_اگر

    إذا (((r->افزودن_idx + 1) & r->idx_ماسک) == r->برداشت_idx) {
        أرجع false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->افزودن_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->اندازه = اندازه;

    r->داده[r->افزودن_idx] = ddata;

    htp_trace_event_شروع(r->trace, HTP_TRACE_EVT_DMA, r->افزودن_idx);

    إذا (اندازه) {
        desc->بعدی       = عدم;
        desc->desc_اندازه  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->ترتیب      = 0;
        desc->انجامشده       = 0;

        dmlink(r->دم, desc);
        r->دم = (dma_descriptor_2d *) desc;
    } وإلا {
        desc->desc_اندازه = 0;
        desc->انجامشده      = 1;
    }

    r->افزودن_idx = (r->افزودن_idx + 1) & r->idx_ماسک;
    أرجع true;
}

ساكن مضمن bool dma_ring_افزودن_single_2d(dma_ring * r, dma_داده ddata, size_t dst_stride, size_t src_stride, size_t ردیف_اندازه, size_t nrows) {
#إذا __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ثابت uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ثابت uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ثابت bool is_ext     = (src_hi | dst_hi) != 0;
#پایان_اگر

    إذا (((r->افزودن_idx + 1) & r->idx_ماسک) == r->برداشت_idx) {
        أرجع false;
    }

    dma_descriptor_2d * desc = &r->desc[r->افزودن_idx];

    desc->بعدی           = عدم;
    desc->reserved1      = 0;
    desc->desc_اندازه      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->ترتیب          = 0;
    desc->انجامشده           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->ردیف_اندازه       = ردیف_اندازه;

#إذا __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_جابه_جایی     = 0;
    desc->dst_جابه_جایی     = 0;
#وإلا
#إذا __HVX_ARCH__ > 79
    desc->src_فوقانی      = src_hi;
    desc->dst_فوقانی      = dst_hi;
    desc->تخصیص     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#وإلا
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#پایان_اگر
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->جابه_جایی         = 0;
#پایان_اگر

    r->داده[r->افزودن_idx] = ddata;

    htp_trace_event_شروع(r->trace, HTP_TRACE_EVT_DMA, r->افزودن_idx);

    إذا (nrows) {
        dmlink(r->دم, desc);
        r->دم = desc;
    } وإلا {
        desc->انجامشده = 1;
    }

    r->افزودن_idx = (r->افزودن_idx + 1) & r->idx_ماسک;
    أرجع true;
}

#إذا __HVX_ARCH__ < 75
ساكن مضمن bool dma_ring_افزودن_single_contig(dma_ring * r, dma_داده ddata, size_t اندازه) {
    أرجع dma_ring_افزودن_single_1d(r, ddata, اندازه);
}
#وإلا
ساكن مضمن bool dma_ring_افزودن_single_contig(dma_ring * r, dma_داده ddata, size_t اندازه) {
    أرجع dma_ring_افزودن_single_2d(r, ddata, اندازه, اندازه, اندازه, 1);
}
#پایان_اگر

ساكن مضمن dma_داده dma_ring_برداشت(dma_ring * r) {
    dma_داده ddata = { 0 };

    إذا (r->افزودن_idx == r->برداشت_idx) {
        أرجع ddata;
    }

    ddata = r->داده[r->برداشت_idx];

    متطاير dma_descriptor_2d * desc = &r->desc[r->برداشت_idx];

    // Wait for desc to complete
    إذا (!desc->انجامشده) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        طالما (!desc->انجامشده) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->برداشت_idx);

    r->برداشت_idx = (r->برداشت_idx + 1) & r->idx_ماسک;
    أرجع ddata;
}

ساكن مضمن dma_داده dma_ring_برداشت_nowait(dma_ring * r) {
    dma_داده ddata = { 0 };

    إذا (r->افزودن_idx == r->برداشت_idx) {
        أرجع ddata;
    }

    ddata = r->داده[r->برداشت_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->برداشت_idx);

    r->برداشت_idx = (r->برداشت_idx + 1) & r->idx_ماسک;
    أرجع ddata;
}

ساكن مضمن bool dma_ring_خالی(dma_ring * r) {
    أرجع r->افزودن_idx == r->برداشت_idx;
}

ساكن مضمن فراغ dma_ring_flush(dma_ring * r) {
    طالما (!dma_ring_خالی(r)) {
        dma_ring_برداشت(r);
    }
}

ساكن مضمن uint32_t dma_ring_عمق(dma_ring * r) {
    أرجع (r->افزودن_idx - r->برداشت_idx) & r->idx_ماسک;
}

ساكن مضمن uint32_t dma_ring_ظرفیت(dma_ring * r) {
    أرجع r->ظرفیت;
}

ساكن مضمن bool dma_صف_افزودن_single_1d(dma_صف * q, dma_داده ddata, size_t اندازه) {
    أرجع dma_ring_افزودن_single_1d(q->ring0, ddata, اندازه);
}

ساكن مضمن bool dma_صف_افزودن_single_2d(dma_صف * q, dma_داده ddata, size_t dst_stride, size_t src_stride, size_t ردیف_اندازه, size_t nrows) {
    أرجع dma_ring_افزودن_single_2d(q->ring0, ddata, dst_stride, src_stride, ردیف_اندازه, nrows);
}

ساكن مضمن dma_داده dma_صف_برداشت(dma_صف * q) {
    أرجع dma_ring_برداشت(q->ring0);
}

ساكن مضمن dma_داده dma_صف_برداشت_nowait(dma_صف * q) {
    أرجع dma_ring_برداشت_nowait(q->ring0);
}

ساكن مضمن bool dma_صف_خالی(dma_صف * q) {
    أرجع dma_ring_خالی(q->ring0);
}

ساكن مضمن فراغ dma_صف_flush(dma_صف * q) {
    dma_ring_flush(q->ring0);
}

ساكن مضمن uint32_t dma_صف_عمق(dma_صف * q) {
    أرجع dma_ring_عمق(q->ring0);
}

ساكن مضمن uint32_t dma_صف_ظرفیت(dma_صف * q) {
    أرجع dma_ring_ظرفیت(q->ring0);
}

ساكن مضمن bool dma_صف_افزودن(dma_صف * q, dma_داده ddata, size_t dst_stride, size_t src_stride, size_t ردیف_اندازه, size_t nrows) {
    إذا (__builtin_expect(nrows == 0, 0)) {
        أرجع dma_ring_افزودن_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    إذا (nrows == 1 || !((ردیف_اندازه ^ src_stride) | (ردیف_اندازه ^ dst_stride))) {
        ثابت size_t مجموع = ردیف_اندازه * nrows;
        إذا (__builtin_expect(مجموع <= DMA_MAX_اندازه_24_B, 1)) {
            أرجع dma_ring_افزودن_single_contig(q->ring0, ddata, مجموع);
        }
        أرجع dma_صف_افزودن_fallback_contig(q, ddata, مجموع);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    إذا (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (ردیف_اندازه | src_stride | dst_stride) <= DMA_MAX_2_D_ردیف_اندازه, 1)) {
        أرجع dma_ring_افزودن_single_2d(q->ring0, ddata, dst_stride, src_stride, ردیف_اندازه, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#إذا __HVX_ARCH__ < 75
    إذا (ردیف_اندازه <= DMA_MAX_اندازه_16_B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        أرجع dma_صف_افزودن_fallback_2d(q, ddata, dst_stride, src_stride, ردیف_اندازه, nrows);
    }
    أرجع dma_صف_افزودن_fallback_1d(q, ddata, dst_stride, src_stride, ردیف_اندازه, nrows);
#وإلا
    أرجع dma_صف_افزودن_fallback_2d(q, ddata, dst_stride, src_stride, ردیف_اندازه, nrows);
#پایان_اگر
}

ساكن مضمن فراغ dma_sync_read(dma_صف * dma_q, فراغ * dst, dma_addr_t src, size_t بایتها) {
    ثابت uint32_t b = (uint32_t) بایتها;
    إذا (b > 0) {
        dma_صف_افزودن(dma_q, dma_make_داده(dst, src), b, b, b, 1);
        dma_صف_برداشت(dma_q);
    }
}

ساكن مضمن فراغ dma_sync_write(dma_صف * dma_q, dma_addr_t dst, ثابت فراغ * src, size_t بایتها) {
    ثابت uint32_t b = (uint32_t) بایتها;
    إذا (b > 0) {
        dma_صف_افزودن(dma_q, dma_make_داده(dst, src), b, b, b, 1);
        dma_صف_برداشت(dma_q);
    }
}

#تعریف_کن DMA_CACHE_MAX_اندازه 256U

// Fully assoc LRU cache
تعريفنوع بنية {
    uint8_t *پایه;
    uint32_t سطر_اندازه;
    uint32_t ظرفیت;
    dma_addr_t src[DMA_CACHE_MAX_اندازه];
    uint16_t age[DMA_CACHE_MAX_اندازه];
} dma_cache_fa;

ساكن مضمن فراغ dma_cache_fa_init(dma_cache_fa *c, uint8_t *پایه, uint32_t سطر_اندازه, uint32_t ظرفیت)
{
    c->ظرفیت  = (ظرفیت > DMA_CACHE_MAX_اندازه) ? DMA_CACHE_MAX_اندازه : ظرفیت;
    c->پایه      = پایه;
    c->سطر_اندازه = سطر_اندازه;

    لكل (بیعلامت i=0; i < c->ظرفیت; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

ساكن مضمن bool dma_cache_fa_افزودن(dma_صف *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t ردیف_اندازه, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    لكل (بیعلامت i=0; i < c->ظرفیت; i++) {
        إذا (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->پایه + (i * c->سطر_اندازه); nrows = 0; // dummy dma
        } وإلا {
            c->age[i]++;
            إذا (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    إذا (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->پایه + o_idx * c->سطر_اندازه; // normal nrows dma
        أرجع dma_صف_افزودن(q, dma_make_داده(dst, src_addr), dst_stride, src_stride, ردیف_اندازه, nrows);
    }

    أرجع dma_صف_افزودن_single_1d(q, dma_make_داده(dst, src_addr), 0);
}

// Direct mapped cache
تعريفنوع بنية {
    uint8_t *پایه;
    uint32_t سطر_اندازه;
    uint32_t ظرفیت;
    uint32_t idx_ماسک;
    dma_addr_t src[DMA_CACHE_MAX_اندازه];
} dma_cache_dm;

ساكن مضمن فراغ dma_cache_dm_init(dma_cache_dm *c, uint8_t *پایه, uint32_t سطر_اندازه, uint32_t ظرفیت)
{
    c->ظرفیت  = (ظرفیت > DMA_CACHE_MAX_اندازه) ? DMA_CACHE_MAX_اندازه : ظرفیت;
    c->idx_ماسک  = c->ظرفیت - 1;
    c->پایه      = پایه;
    c->سطر_اندازه = سطر_اندازه;

    لكل (بیعلامت i=0; i < c->ظرفیت; i++) {
        c->src[i] = 0;
    }
}

ساكن مضمن bool dma_cache_dm_افزودن(dma_صف *q, dma_cache_dm *c, uint32_t جایگاه, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t ردیف_اندازه, uint32_t nrows)
{
    ثابت uint32_t i = جایگاه & c->idx_ماسک;
    uint8_t * dst = c->پایه + (i * c->سطر_اندازه);

    إذا (c->src[i] == src_addr) {
        أرجع dma_صف_افزودن_single_1d(q, dma_make_داده(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    أرجع dma_صف_افزودن(q, dma_make_داده(dst, src_addr), dst_stride, src_stride, ردیف_اندازه, nrows);
}

#اگر_تعریف_شده __cplusplus
}  // extern "C"
#پایان_اگر

#پایان_اگر /* HTP_DMA_H */
