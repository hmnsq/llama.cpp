#جے_بنا_تعریف HTP_DMA_H
#تعریف_کرو HTP_DMA_H

#ضمن <HAP_farf.h>
#ضمن <hexagon_types.h>
#ضمن <assert.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <string.h>
#ضمن "hex-utils.h"

#ضمن "hex-profile.h"

#جے_تعریف_شدہ __cplusplus
خارجي "C" {
#جے_انت

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
تعريفنوع بنية dma_descriptor_1d_s {
    فراغ *   اگلا;
    uint32_t ناپ:24;
    uint32_t desc_ناپ:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t پورا:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#إذا __HVX_ARCH__ < 75

تعريفنوع بنية dma_descriptor_2d_s {
    فراغ *   اگلا;
    uint32_t reserved0:24;
    uint32_t desc_ناپ:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t پورا:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t صف_ناپ:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_وتھ:16;
    uint32_t dst_وتھ:16;
} dma_descriptor_2d;

#وإلا

تعريفنوع بنية dma_descriptor_2d_s {
    فراغ *   اگلا;
    uint32_t dst_stride:24;
    uint32_t desc_ناپ:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t پورا:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#إذا __HVX_ARCH__ > 79
    uint32_t src_اتلا:8;
    uint32_t dst_اتلا:8;
    uint32_t ونڈ:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#وإلا
    uint32_t reserved0:24;
#جے_انت
    uint32_t صف_ناپ:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t وتھ:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#جے_انت

#إذا __HVX_ARCH__ > 79
تعريفنوع uint64_t dma_addr_t;
#وإلا
تعريفنوع uint32_t dma_addr_t;
#جے_انت

تعريفنوع بنية {
    dma_addr_t dst;
    dma_addr_t src;
} dma_انکڑے;

// Hardware descriptor field limits
#تعریف_کرو DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#تعریف_کرو DMA_MAX_ناپ_16_B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#تعریف_کرو DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#تعریف_کرو DMA_MAX_ناپ_24_B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#تعریف_کرو DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#تعریف_کرو DMA_محفوظ_CHUNK_ناپ    0x00F00000u    // ~15MB safe contiguous chunk size

#إذا __HVX_ARCH__ < 75
#تعریف_کرو DMA_MAX_2_D_صف_ناپ    DMA_MAX_ناپ_16_B
#تعریف_کرو DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#وإلا
#تعریف_کرو DMA_MAX_2_D_صف_ناپ    DMA_MAX_ناپ_24_B
#تعریف_کرو DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#جے_انت

#تعریف_کرو DMA_FALLBACK_سمرتھا  16u            // descriptors in secondary fallback ring

تعريفنوع بنية dma_ring_s dma_ring;
بنية dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * پوچھ;      // tail pointer
    dma_انکڑے          * انکڑے;      // dst/src data
    uint32_t            ڈالو_idx;
    uint32_t            نکالو_idx;
    uint32_t            سمرتھا;
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

size_t      dma_قطار_sizeof(size_t سمرتھا);
size_t      dma_قطار_alignof(فراغ);
dma_قطار_t dma_قطار_init(فراغ * ptr, size_t سمرتھا, بنية htp_thread_trace * trace);
فراغ        dma_قطار_free(dma_قطار_t q);

size_t      dma_قطار_alias_sizeof(فراغ);
dma_قطار_t dma_قطار_alias_init(فراغ * ptr, dma_قطار_t main_q);
فراغ        dma_قطار_alias_free(dma_قطار_t q);

bool        dma_قطار_ڈالو_fallback_2d(dma_قطار * q, dma_انکڑے ddata, size_t dst_stride, size_t src_stride, size_t صف_ناپ, size_t nrows);
bool        dma_قطار_ڈالو_fallback_contig(dma_قطار * q, dma_انکڑے ddata, size_t کل);
#إذا __HVX_ARCH__ < 75
bool        dma_قطار_ڈالو_fallback_1d(dma_قطار * q, dma_انکڑے ddata, size_t dst_stride, size_t src_stride, size_t صف_ناپ, size_t nrows);
#جے_انت

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
ساكن مضمن فراغ dmstart(فراغ * اگلا) {
    asm متطاير(" release(%0):at" : : "r"(اگلا));
    asm متطاير(" dmstart(%0)" : : "r"(اگلا));
}

ساكن مضمن فراغ dmlink(فراغ * cur, فراغ * اگلا) {
    asm متطاير(" release(%0):at" : : "r"(اگلا));
    asm متطاير(" dmlink(%0, %1)" : : "r"(cur), "r"(اگلا));
}

ساكن مضمن بنانشانی صحيح dmpoll(فراغ) {
    بنانشانی صحيح ret = 0;
    asm متطاير(" %0 = dmpoll" : "=r"(ret) : : "memory");
    أرجع ret;
}

ساكن مضمن بنانشانی صحيح dmwait(فراغ) {
    بنانشانی صحيح ret = 0;
    asm متطاير(" %0 = dmwait" : "=r"(ret) : : "memory");
    أرجع ret;
}

ساكن مضمن dma_انکڑے dma_make_انکڑے_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_انکڑے d = { dst, src };
    أرجع d;
}

#تعریف_کرو dma_make_انکڑے(dst, src) dma_make_انکڑے_impl((dma_addr_t) (dst), (dma_addr_t) (src))

ساكن مضمن bool dma_ring_ڈالو_single_1d(dma_ring * r, dma_انکڑے ddata, size_t ناپ) {
#إذا __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 32) || ناپ == 0);
#جے_انت

    إذا (((r->ڈالو_idx + 1) & r->idx_نقاب) == r->نکالو_idx) {
        أرجع false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->ڈالو_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->ناپ = ناپ;

    r->انکڑے[r->ڈالو_idx] = ddata;

    htp_trace_event_شروع(r->trace, HTP_TRACE_EVT_DMA, r->ڈالو_idx);

    إذا (ناپ) {
        desc->اگلا       = عدم;
        desc->desc_ناپ  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->ترتیب      = 0;
        desc->پورا       = 0;

        dmlink(r->پوچھ, desc);
        r->پوچھ = (dma_descriptor_2d *) desc;
    } وإلا {
        desc->desc_ناپ = 0;
        desc->پورا      = 1;
    }

    r->ڈالو_idx = (r->ڈالو_idx + 1) & r->idx_نقاب;
    أرجع true;
}

ساكن مضمن bool dma_ring_ڈالو_single_2d(dma_ring * r, dma_انکڑے ddata, size_t dst_stride, size_t src_stride, size_t صف_ناپ, size_t nrows) {
#إذا __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ثابت uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ثابت uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ثابت bool is_ext     = (src_hi | dst_hi) != 0;
#جے_انت

    إذا (((r->ڈالو_idx + 1) & r->idx_نقاب) == r->نکالو_idx) {
        أرجع false;
    }

    dma_descriptor_2d * desc = &r->desc[r->ڈالو_idx];

    desc->اگلا           = عدم;
    desc->reserved1      = 0;
    desc->desc_ناپ      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->ترتیب          = 0;
    desc->پورا           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->صف_ناپ       = صف_ناپ;

#إذا __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_وتھ     = 0;
    desc->dst_وتھ     = 0;
#وإلا
#إذا __HVX_ARCH__ > 79
    desc->src_اتلا      = src_hi;
    desc->dst_اتلا      = dst_hi;
    desc->ونڈ     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#وإلا
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#جے_انت
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->وتھ         = 0;
#جے_انت

    r->انکڑے[r->ڈالو_idx] = ddata;

    htp_trace_event_شروع(r->trace, HTP_TRACE_EVT_DMA, r->ڈالو_idx);

    إذا (nrows) {
        dmlink(r->پوچھ, desc);
        r->پوچھ = desc;
    } وإلا {
        desc->پورا = 1;
    }

    r->ڈالو_idx = (r->ڈالو_idx + 1) & r->idx_نقاب;
    أرجع true;
}

#إذا __HVX_ARCH__ < 75
ساكن مضمن bool dma_ring_ڈالو_single_contig(dma_ring * r, dma_انکڑے ddata, size_t ناپ) {
    أرجع dma_ring_ڈالو_single_1d(r, ddata, ناپ);
}
#وإلا
ساكن مضمن bool dma_ring_ڈالو_single_contig(dma_ring * r, dma_انکڑے ddata, size_t ناپ) {
    أرجع dma_ring_ڈالو_single_2d(r, ddata, ناپ, ناپ, ناپ, 1);
}
#جے_انت

ساكن مضمن dma_انکڑے dma_ring_نکالو(dma_ring * r) {
    dma_انکڑے ddata = { 0 };

    إذا (r->ڈالو_idx == r->نکالو_idx) {
        أرجع ddata;
    }

    ddata = r->انکڑے[r->نکالو_idx];

    متطاير dma_descriptor_2d * desc = &r->desc[r->نکالو_idx];

    // Wait for desc to complete
    إذا (!desc->پورا) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        طالما (!desc->پورا) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->نکالو_idx);

    r->نکالو_idx = (r->نکالو_idx + 1) & r->idx_نقاب;
    أرجع ddata;
}

ساكن مضمن dma_انکڑے dma_ring_نکالو_nowait(dma_ring * r) {
    dma_انکڑے ddata = { 0 };

    إذا (r->ڈالو_idx == r->نکالو_idx) {
        أرجع ddata;
    }

    ddata = r->انکڑے[r->نکالو_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->نکالو_idx);

    r->نکالو_idx = (r->نکالو_idx + 1) & r->idx_نقاب;
    أرجع ddata;
}

ساكن مضمن bool dma_ring_خالی_تھاواں(dma_ring * r) {
    أرجع r->ڈالو_idx == r->نکالو_idx;
}

ساكن مضمن فراغ dma_ring_flush(dma_ring * r) {
    طالما (!dma_ring_خالی_تھاواں(r)) {
        dma_ring_نکالو(r);
    }
}

ساكن مضمن uint32_t dma_ring_ڈونگھائی(dma_ring * r) {
    أرجع (r->ڈالو_idx - r->نکالو_idx) & r->idx_نقاب;
}

ساكن مضمن uint32_t dma_ring_سمرتھا(dma_ring * r) {
    أرجع r->سمرتھا;
}

ساكن مضمن bool dma_قطار_ڈالو_single_1d(dma_قطار * q, dma_انکڑے ddata, size_t ناپ) {
    أرجع dma_ring_ڈالو_single_1d(q->ring0, ddata, ناپ);
}

ساكن مضمن bool dma_قطار_ڈالو_single_2d(dma_قطار * q, dma_انکڑے ddata, size_t dst_stride, size_t src_stride, size_t صف_ناپ, size_t nrows) {
    أرجع dma_ring_ڈالو_single_2d(q->ring0, ddata, dst_stride, src_stride, صف_ناپ, nrows);
}

ساكن مضمن dma_انکڑے dma_قطار_نکالو(dma_قطار * q) {
    أرجع dma_ring_نکالو(q->ring0);
}

ساكن مضمن dma_انکڑے dma_قطار_نکالو_nowait(dma_قطار * q) {
    أرجع dma_ring_نکالو_nowait(q->ring0);
}

ساكن مضمن bool dma_قطار_خالی_تھاواں(dma_قطار * q) {
    أرجع dma_ring_خالی_تھاواں(q->ring0);
}

ساكن مضمن فراغ dma_قطار_flush(dma_قطار * q) {
    dma_ring_flush(q->ring0);
}

ساكن مضمن uint32_t dma_قطار_ڈونگھائی(dma_قطار * q) {
    أرجع dma_ring_ڈونگھائی(q->ring0);
}

ساكن مضمن uint32_t dma_قطار_سمرتھا(dma_قطار * q) {
    أرجع dma_ring_سمرتھا(q->ring0);
}

ساكن مضمن bool dma_قطار_ڈالو(dma_قطار * q, dma_انکڑے ddata, size_t dst_stride, size_t src_stride, size_t صف_ناپ, size_t nrows) {
    إذا (__builtin_expect(nrows == 0, 0)) {
        أرجع dma_ring_ڈالو_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    إذا (nrows == 1 || !((صف_ناپ ^ src_stride) | (صف_ناپ ^ dst_stride))) {
        ثابت size_t کل = صف_ناپ * nrows;
        إذا (__builtin_expect(کل <= DMA_MAX_ناپ_24_B, 1)) {
            أرجع dma_ring_ڈالو_single_contig(q->ring0, ddata, کل);
        }
        أرجع dma_قطار_ڈالو_fallback_contig(q, ddata, کل);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    إذا (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (صف_ناپ | src_stride | dst_stride) <= DMA_MAX_2_D_صف_ناپ, 1)) {
        أرجع dma_ring_ڈالو_single_2d(q->ring0, ddata, dst_stride, src_stride, صف_ناپ, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#إذا __HVX_ARCH__ < 75
    إذا (صف_ناپ <= DMA_MAX_ناپ_16_B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        أرجع dma_قطار_ڈالو_fallback_2d(q, ddata, dst_stride, src_stride, صف_ناپ, nrows);
    }
    أرجع dma_قطار_ڈالو_fallback_1d(q, ddata, dst_stride, src_stride, صف_ناپ, nrows);
#وإلا
    أرجع dma_قطار_ڈالو_fallback_2d(q, ddata, dst_stride, src_stride, صف_ناپ, nrows);
#جے_انت
}

ساكن مضمن فراغ dma_sync_read(dma_قطار * dma_q, فراغ * dst, dma_addr_t src, size_t بائٹاں) {
    ثابت uint32_t b = (uint32_t) بائٹاں;
    إذا (b > 0) {
        dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst, src), b, b, b, 1);
        dma_قطار_نکالو(dma_q);
    }
}

ساكن مضمن فراغ dma_sync_write(dma_قطار * dma_q, dma_addr_t dst, ثابت فراغ * src, size_t بائٹاں) {
    ثابت uint32_t b = (uint32_t) بائٹاں;
    إذا (b > 0) {
        dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst, src), b, b, b, 1);
        dma_قطار_نکالو(dma_q);
    }
}

#تعریف_کرو DMA_CACHE_MAX_ناپ 256U

// Fully assoc LRU cache
تعريفنوع بنية {
    uint8_t *ادھار;
    uint32_t سطر_ناپ;
    uint32_t سمرتھا;
    dma_addr_t src[DMA_CACHE_MAX_ناپ];
    uint16_t age[DMA_CACHE_MAX_ناپ];
} dma_cache_fa;

ساكن مضمن فراغ dma_cache_fa_init(dma_cache_fa *c, uint8_t *ادھار, uint32_t سطر_ناپ, uint32_t سمرتھا)
{
    c->سمرتھا  = (سمرتھا > DMA_CACHE_MAX_ناپ) ? DMA_CACHE_MAX_ناپ : سمرتھا;
    c->ادھار      = ادھار;
    c->سطر_ناپ = سطر_ناپ;

    لكل (بنانشانی i=0; i < c->سمرتھا; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

ساكن مضمن bool dma_cache_fa_ڈالو(dma_قطار *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t صف_ناپ, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    لكل (بنانشانی i=0; i < c->سمرتھا; i++) {
        إذا (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->ادھار + (i * c->سطر_ناپ); nrows = 0; // dummy dma
        } وإلا {
            c->age[i]++;
            إذا (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    إذا (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->ادھار + o_idx * c->سطر_ناپ; // normal nrows dma
        أرجع dma_قطار_ڈالو(q, dma_make_انکڑے(dst, src_addr), dst_stride, src_stride, صف_ناپ, nrows);
    }

    أرجع dma_قطار_ڈالو_single_1d(q, dma_make_انکڑے(dst, src_addr), 0);
}

// Direct mapped cache
تعريفنوع بنية {
    uint8_t *ادھار;
    uint32_t سطر_ناپ;
    uint32_t سمرتھا;
    uint32_t idx_نقاب;
    dma_addr_t src[DMA_CACHE_MAX_ناپ];
} dma_cache_dm;

ساكن مضمن فراغ dma_cache_dm_init(dma_cache_dm *c, uint8_t *ادھار, uint32_t سطر_ناپ, uint32_t سمرتھا)
{
    c->سمرتھا  = (سمرتھا > DMA_CACHE_MAX_ناپ) ? DMA_CACHE_MAX_ناپ : سمرتھا;
    c->idx_نقاب  = c->سمرتھا - 1;
    c->ادھار      = ادھار;
    c->سطر_ناپ = سطر_ناپ;

    لكل (بنانشانی i=0; i < c->سمرتھا; i++) {
        c->src[i] = 0;
    }
}

ساكن مضمن bool dma_cache_dm_ڈالو(dma_قطار *q, dma_cache_dm *c, uint32_t تھاں, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t صف_ناپ, uint32_t nrows)
{
    ثابت uint32_t i = تھاں & c->idx_نقاب;
    uint8_t * dst = c->ادھار + (i * c->سطر_ناپ);

    إذا (c->src[i] == src_addr) {
        أرجع dma_قطار_ڈالو_single_1d(q, dma_make_انکڑے(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    أرجع dma_قطار_ڈالو(q, dma_make_انکڑے(dst, src_addr), dst_stride, src_stride, صف_ناپ, nrows);
}

#جے_تعریف_شدہ __cplusplus
}  // extern "C"
#جے_انت

#جے_انت /* HTP_DMA_H */
