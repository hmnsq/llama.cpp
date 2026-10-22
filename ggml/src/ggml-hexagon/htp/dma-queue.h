#جيڪڏهن_وصف_نه_ٿيل HTP_DMA_H
#وصف_ڪريو HTP_DMA_H

#ضمن <HAP_farf.h>
#ضمن <hexagon_types.h>
#ضمن <assert.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <string.h>
#ضمن "hex-utils.h"

#ضمن "hex-profile.h"

#جيڪڏهن_وصف_ٿيل __cplusplus
خارجي "C" {
#جيڪڏهن_پورو

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
تعريفنوع بنية dma_descriptor_1d_s {
    فراغ *   ايندڙ;
    uint32_t سائيز:24;
    uint32_t desc_سائيز:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتيب:1;
    uint32_t پورو:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#إذا __HVX_ARCH__ < 75

تعريفنوع بنية dma_descriptor_2d_s {
    فراغ *   ايندڙ;
    uint32_t reserved0:24;
    uint32_t desc_سائيز:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتيب:1;
    uint32_t پورو:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t صف_سائيز:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_وٿي:16;
    uint32_t dst_وٿي:16;
} dma_descriptor_2d;

#وإلا

تعريفنوع بنية dma_descriptor_2d_s {
    فراغ *   ايندڙ;
    uint32_t dst_stride:24;
    uint32_t desc_سائيز:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتيب:1;
    uint32_t پورو:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#إذا __HVX_ARCH__ > 79
    uint32_t src_مٿانهون:8;
    uint32_t dst_مٿانهون:8;
    uint32_t ورهاست:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#وإلا
    uint32_t reserved0:24;
#جيڪڏهن_پورو
    uint32_t صف_سائيز:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t وٿي:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#جيڪڏهن_پورو

#إذا __HVX_ARCH__ > 79
تعريفنوع uint64_t dma_addr_t;
#وإلا
تعريفنوع uint32_t dma_addr_t;
#جيڪڏهن_پورو

تعريفنوع بنية {
    dma_addr_t dst;
    dma_addr_t src;
} dma_معلومات;

// Hardware descriptor field limits
#وصف_ڪريو DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#وصف_ڪريو DMA_MAX_سائيز_16_B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#وصف_ڪريو DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#وصف_ڪريو DMA_MAX_سائيز_24_B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#وصف_ڪريو DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#وصف_ڪريو DMA_محفوظ_CHUNK_سائيز    0x00F00000u    // ~15MB safe contiguous chunk size

#إذا __HVX_ARCH__ < 75
#وصف_ڪريو DMA_MAX_2_D_صف_سائيز    DMA_MAX_سائيز_16_B
#وصف_ڪريو DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#وإلا
#وصف_ڪريو DMA_MAX_2_D_صف_سائيز    DMA_MAX_سائيز_24_B
#وصف_ڪريو DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#جيڪڏهن_پورو

#وصف_ڪريو DMA_FALLBACK_صلاحيت  16u            // descriptors in secondary fallback ring

تعريفنوع بنية dma_ring_s dma_ring;
بنية dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * پڇ;      // tail pointer
    dma_معلومات          * معلومات;      // dst/src data
    uint32_t            ڌڪو_idx;
    uint32_t            ڪڍو_idx;
    uint32_t            صلاحيت;
    uint32_t            idx_نقاب;
    بنية htp_thread_trace * trace;
};

تعريفنوع بنية dma_قطار_s dma_قطار;
تعريفنوع dma_قطار * dma_قطار_t;

بنية dma_قطار_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_قطار_sizeof(size_t صلاحيت);
size_t      dma_قطار_alignof(فراغ);
dma_قطار_t dma_قطار_init(فراغ * ptr, size_t صلاحيت, بنية htp_thread_trace * trace);
فراغ        dma_قطار_free(dma_قطار_t q);

size_t      dma_قطار_alias_sizeof(فراغ);
dma_قطار_t dma_قطار_alias_init(فراغ * ptr, dma_قطار_t main_q);
فراغ        dma_قطار_alias_free(dma_قطار_t q);

bool        dma_قطار_ڌڪو_fallback_2d(dma_قطار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t صف_سائيز, size_t nrows);
bool        dma_قطار_ڌڪو_fallback_contig(dma_قطار * q, dma_معلومات ddata, size_t ڪل);
#إذا __HVX_ARCH__ < 75
bool        dma_قطار_ڌڪو_fallback_1d(dma_قطار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t صف_سائيز, size_t nrows);
#جيڪڏهن_پورو

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
ساكن مضمن فراغ dmstart(فراغ * ايندڙ) {
    asm متطاير(" release(%0):at" : : "r"(ايندڙ));
    asm متطاير(" dmstart(%0)" : : "r"(ايندڙ));
}

ساكن مضمن فراغ dmlink(فراغ * cur, فراغ * ايندڙ) {
    asm متطاير(" release(%0):at" : : "r"(ايندڙ));
    asm متطاير(" dmlink(%0, %1)" : : "r"(cur), "r"(ايندڙ));
}

ساكن مضمن بنانشان صحيح dmpoll(فراغ) {
    بنانشان صحيح ret = 0;
    asm متطاير(" %0 = dmpoll" : "=r"(ret) : : "memory");
    أرجع ret;
}

ساكن مضمن بنانشان صحيح dmwait(فراغ) {
    بنانشان صحيح ret = 0;
    asm متطاير(" %0 = dmwait" : "=r"(ret) : : "memory");
    أرجع ret;
}

ساكن مضمن dma_معلومات dma_make_معلومات_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_معلومات d = { dst, src };
    أرجع d;
}

#وصف_ڪريو dma_make_معلومات(dst, src) dma_make_معلومات_impl((dma_addr_t) (dst), (dma_addr_t) (src))

ساكن مضمن bool dma_ring_ڌڪو_single_1d(dma_ring * r, dma_معلومات ddata, size_t سائيز) {
#إذا __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 32) || سائيز == 0);
#جيڪڏهن_پورو

    إذا (((r->ڌڪو_idx + 1) & r->idx_نقاب) == r->ڪڍو_idx) {
        أرجع false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->ڌڪو_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->سائيز = سائيز;

    r->معلومات[r->ڌڪو_idx] = ddata;

    htp_trace_event_شروعات(r->trace, HTP_TRACE_EVT_DMA, r->ڌڪو_idx);

    إذا (سائيز) {
        desc->ايندڙ       = عدم;
        desc->desc_سائيز  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->ترتيب      = 0;
        desc->پورو       = 0;

        dmlink(r->پڇ, desc);
        r->پڇ = (dma_descriptor_2d *) desc;
    } وإلا {
        desc->desc_سائيز = 0;
        desc->پورو      = 1;
    }

    r->ڌڪو_idx = (r->ڌڪو_idx + 1) & r->idx_نقاب;
    أرجع true;
}

ساكن مضمن bool dma_ring_ڌڪو_single_2d(dma_ring * r, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t صف_سائيز, size_t nrows) {
#إذا __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ثابت uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ثابت uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ثابت bool is_ext     = (src_hi | dst_hi) != 0;
#جيڪڏهن_پورو

    إذا (((r->ڌڪو_idx + 1) & r->idx_نقاب) == r->ڪڍو_idx) {
        أرجع false;
    }

    dma_descriptor_2d * desc = &r->desc[r->ڌڪو_idx];

    desc->ايندڙ           = عدم;
    desc->reserved1      = 0;
    desc->desc_سائيز      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->ترتيب          = 0;
    desc->پورو           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->صف_سائيز       = صف_سائيز;

#إذا __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_وٿي     = 0;
    desc->dst_وٿي     = 0;
#وإلا
#إذا __HVX_ARCH__ > 79
    desc->src_مٿانهون      = src_hi;
    desc->dst_مٿانهون      = dst_hi;
    desc->ورهاست     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#وإلا
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#جيڪڏهن_پورو
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->وٿي         = 0;
#جيڪڏهن_پورو

    r->معلومات[r->ڌڪو_idx] = ddata;

    htp_trace_event_شروعات(r->trace, HTP_TRACE_EVT_DMA, r->ڌڪو_idx);

    إذا (nrows) {
        dmlink(r->پڇ, desc);
        r->پڇ = desc;
    } وإلا {
        desc->پورو = 1;
    }

    r->ڌڪو_idx = (r->ڌڪو_idx + 1) & r->idx_نقاب;
    أرجع true;
}

#إذا __HVX_ARCH__ < 75
ساكن مضمن bool dma_ring_ڌڪو_single_contig(dma_ring * r, dma_معلومات ddata, size_t سائيز) {
    أرجع dma_ring_ڌڪو_single_1d(r, ddata, سائيز);
}
#وإلا
ساكن مضمن bool dma_ring_ڌڪو_single_contig(dma_ring * r, dma_معلومات ddata, size_t سائيز) {
    أرجع dma_ring_ڌڪو_single_2d(r, ddata, سائيز, سائيز, سائيز, 1);
}
#جيڪڏهن_پورو

ساكن مضمن dma_معلومات dma_ring_ڪڍو(dma_ring * r) {
    dma_معلومات ddata = { 0 };

    إذا (r->ڌڪو_idx == r->ڪڍو_idx) {
        أرجع ddata;
    }

    ddata = r->معلومات[r->ڪڍو_idx];

    متطاير dma_descriptor_2d * desc = &r->desc[r->ڪڍو_idx];

    // Wait for desc to complete
    إذا (!desc->پورو) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        طالما (!desc->پورو) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->ڪڍو_idx);

    r->ڪڍو_idx = (r->ڪڍو_idx + 1) & r->idx_نقاب;
    أرجع ddata;
}

ساكن مضمن dma_معلومات dma_ring_ڪڍو_nowait(dma_ring * r) {
    dma_معلومات ddata = { 0 };

    إذا (r->ڌڪو_idx == r->ڪڍو_idx) {
        أرجع ddata;
    }

    ddata = r->معلومات[r->ڪڍو_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->ڪڍو_idx);

    r->ڪڍو_idx = (r->ڪڍو_idx + 1) & r->idx_نقاب;
    أرجع ddata;
}

ساكن مضمن bool dma_ring_خالي_جايون(dma_ring * r) {
    أرجع r->ڌڪو_idx == r->ڪڍو_idx;
}

ساكن مضمن فراغ dma_ring_flush(dma_ring * r) {
    طالما (!dma_ring_خالي_جايون(r)) {
        dma_ring_ڪڍو(r);
    }
}

ساكن مضمن uint32_t dma_ring_اونهائي(dma_ring * r) {
    أرجع (r->ڌڪو_idx - r->ڪڍو_idx) & r->idx_نقاب;
}

ساكن مضمن uint32_t dma_ring_صلاحيت(dma_ring * r) {
    أرجع r->صلاحيت;
}

ساكن مضمن bool dma_قطار_ڌڪو_single_1d(dma_قطار * q, dma_معلومات ddata, size_t سائيز) {
    أرجع dma_ring_ڌڪو_single_1d(q->ring0, ddata, سائيز);
}

ساكن مضمن bool dma_قطار_ڌڪو_single_2d(dma_قطار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t صف_سائيز, size_t nrows) {
    أرجع dma_ring_ڌڪو_single_2d(q->ring0, ddata, dst_stride, src_stride, صف_سائيز, nrows);
}

ساكن مضمن dma_معلومات dma_قطار_ڪڍو(dma_قطار * q) {
    أرجع dma_ring_ڪڍو(q->ring0);
}

ساكن مضمن dma_معلومات dma_قطار_ڪڍو_nowait(dma_قطار * q) {
    أرجع dma_ring_ڪڍو_nowait(q->ring0);
}

ساكن مضمن bool dma_قطار_خالي_جايون(dma_قطار * q) {
    أرجع dma_ring_خالي_جايون(q->ring0);
}

ساكن مضمن فراغ dma_قطار_flush(dma_قطار * q) {
    dma_ring_flush(q->ring0);
}

ساكن مضمن uint32_t dma_قطار_اونهائي(dma_قطار * q) {
    أرجع dma_ring_اونهائي(q->ring0);
}

ساكن مضمن uint32_t dma_قطار_صلاحيت(dma_قطار * q) {
    أرجع dma_ring_صلاحيت(q->ring0);
}

ساكن مضمن bool dma_قطار_ڌڪو(dma_قطار * q, dma_معلومات ddata, size_t dst_stride, size_t src_stride, size_t صف_سائيز, size_t nrows) {
    إذا (__builtin_expect(nrows == 0, 0)) {
        أرجع dma_ring_ڌڪو_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    إذا (nrows == 1 || !((صف_سائيز ^ src_stride) | (صف_سائيز ^ dst_stride))) {
        ثابت size_t ڪل = صف_سائيز * nrows;
        إذا (__builtin_expect(ڪل <= DMA_MAX_سائيز_24_B, 1)) {
            أرجع dma_ring_ڌڪو_single_contig(q->ring0, ddata, ڪل);
        }
        أرجع dma_قطار_ڌڪو_fallback_contig(q, ddata, ڪل);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    إذا (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (صف_سائيز | src_stride | dst_stride) <= DMA_MAX_2_D_صف_سائيز, 1)) {
        أرجع dma_ring_ڌڪو_single_2d(q->ring0, ddata, dst_stride, src_stride, صف_سائيز, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#إذا __HVX_ARCH__ < 75
    إذا (صف_سائيز <= DMA_MAX_سائيز_16_B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        أرجع dma_قطار_ڌڪو_fallback_2d(q, ddata, dst_stride, src_stride, صف_سائيز, nrows);
    }
    أرجع dma_قطار_ڌڪو_fallback_1d(q, ddata, dst_stride, src_stride, صف_سائيز, nrows);
#وإلا
    أرجع dma_قطار_ڌڪو_fallback_2d(q, ddata, dst_stride, src_stride, صف_سائيز, nrows);
#جيڪڏهن_پورو
}

ساكن مضمن فراغ dma_sync_read(dma_قطار * dma_q, فراغ * dst, dma_addr_t src, size_t بائيٽون) {
    ثابت uint32_t b = (uint32_t) بائيٽون;
    إذا (b > 0) {
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst, src), b, b, b, 1);
        dma_قطار_ڪڍو(dma_q);
    }
}

ساكن مضمن فراغ dma_sync_write(dma_قطار * dma_q, dma_addr_t dst, ثابت فراغ * src, size_t بائيٽون) {
    ثابت uint32_t b = (uint32_t) بائيٽون;
    إذا (b > 0) {
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst, src), b, b, b, 1);
        dma_قطار_ڪڍو(dma_q);
    }
}

#وصف_ڪريو DMA_CACHE_MAX_سائيز 256U

// Fully assoc LRU cache
تعريفنوع بنية {
    uint8_t *بنياد;
    uint32_t سٽ_سائيز;
    uint32_t صلاحيت;
    dma_addr_t src[DMA_CACHE_MAX_سائيز];
    uint16_t age[DMA_CACHE_MAX_سائيز];
} dma_cache_fa;

ساكن مضمن فراغ dma_cache_fa_init(dma_cache_fa *c, uint8_t *بنياد, uint32_t سٽ_سائيز, uint32_t صلاحيت)
{
    c->صلاحيت  = (صلاحيت > DMA_CACHE_MAX_سائيز) ? DMA_CACHE_MAX_سائيز : صلاحيت;
    c->بنياد      = بنياد;
    c->سٽ_سائيز = سٽ_سائيز;

    لكل (بنانشان i=0; i < c->صلاحيت; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

ساكن مضمن bool dma_cache_fa_ڌڪو(dma_قطار *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t صف_سائيز, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    لكل (بنانشان i=0; i < c->صلاحيت; i++) {
        إذا (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->بنياد + (i * c->سٽ_سائيز); nrows = 0; // dummy dma
        } وإلا {
            c->age[i]++;
            إذا (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    إذا (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->بنياد + o_idx * c->سٽ_سائيز; // normal nrows dma
        أرجع dma_قطار_ڌڪو(q, dma_make_معلومات(dst, src_addr), dst_stride, src_stride, صف_سائيز, nrows);
    }

    أرجع dma_قطار_ڌڪو_single_1d(q, dma_make_معلومات(dst, src_addr), 0);
}

// Direct mapped cache
تعريفنوع بنية {
    uint8_t *بنياد;
    uint32_t سٽ_سائيز;
    uint32_t صلاحيت;
    uint32_t idx_نقاب;
    dma_addr_t src[DMA_CACHE_MAX_سائيز];
} dma_cache_dm;

ساكن مضمن فراغ dma_cache_dm_init(dma_cache_dm *c, uint8_t *بنياد, uint32_t سٽ_سائيز, uint32_t صلاحيت)
{
    c->صلاحيت  = (صلاحيت > DMA_CACHE_MAX_سائيز) ? DMA_CACHE_MAX_سائيز : صلاحيت;
    c->idx_نقاب  = c->صلاحيت - 1;
    c->بنياد      = بنياد;
    c->سٽ_سائيز = سٽ_سائيز;

    لكل (بنانشان i=0; i < c->صلاحيت; i++) {
        c->src[i] = 0;
    }
}

ساكن مضمن bool dma_cache_dm_ڌڪو(dma_قطار *q, dma_cache_dm *c, uint32_t جڳهه, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t صف_سائيز, uint32_t nrows)
{
    ثابت uint32_t i = جڳهه & c->idx_نقاب;
    uint8_t * dst = c->بنياد + (i * c->سٽ_سائيز);

    إذا (c->src[i] == src_addr) {
        أرجع dma_قطار_ڌڪو_single_1d(q, dma_make_معلومات(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    أرجع dma_قطار_ڌڪو(q, dma_make_معلومات(dst, src_addr), dst_stride, src_stride, صف_سائيز, nrows);
}

#جيڪڏهن_وصف_ٿيل __cplusplus
}  // extern "C"
#جيڪڏهن_پورو

#جيڪڏهن_پورو /* HTP_DMA_H */
