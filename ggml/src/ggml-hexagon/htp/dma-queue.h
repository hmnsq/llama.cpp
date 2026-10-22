#اگر_غیر_تعریف_شدہ HTP_DMA_H
#تعریف_کرو HTP_DMA_H

#شامل <HAP_farf.h>
#شامل <hexagon_types.h>
#شامل <assert.h>
#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <string.h>
#شامل "hex-utils.h"

#شامل "hex-profile.h"

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
قسمتعریف ساخت dma_descriptor_1d_s {
    خالی *   اگلا;
    uint32_t سائز:24;
    uint32_t desc_سائز:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t مکمل:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#اگر __HVX_ARCH__ < 75

قسمتعریف ساخت dma_descriptor_2d_s {
    خالی *   اگلا;
    uint32_t reserved0:24;
    uint32_t desc_سائز:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t مکمل:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t صف_سائز:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_انحراف:16;
    uint32_t dst_انحراف:16;
} dma_descriptor_2d;

#ورنہ

قسمتعریف ساخت dma_descriptor_2d_s {
    خالی *   اگلا;
    uint32_t dst_stride:24;
    uint32_t desc_سائز:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ترتیب:1;
    uint32_t مکمل:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#اگر __HVX_ARCH__ > 79
    uint32_t src_بالائی:8;
    uint32_t dst_بالائی:8;
    uint32_t تخصیص:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#ورنہ
    uint32_t reserved0:24;
#اگر_ختم
    uint32_t صف_سائز:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t انحراف:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#اگر_ختم

#اگر __HVX_ARCH__ > 79
قسمتعریف uint64_t dma_addr_t;
#ورنہ
قسمتعریف uint32_t dma_addr_t;
#اگر_ختم

قسمتعریف ساخت {
    dma_addr_t dst;
    dma_addr_t src;
} dma_ڈیٹا;

// Hardware descriptor field limits
#تعریف_کرو DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#تعریف_کرو DMA_MAX_سائز_16_B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#تعریف_کرو DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#تعریف_کرو DMA_MAX_سائز_24_B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#تعریف_کرو DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#تعریف_کرو DMA_محفوظ_CHUNK_سائز    0x00F00000u    // ~15MB safe contiguous chunk size

#اگر __HVX_ARCH__ < 75
#تعریف_کرو DMA_MAX_2_D_صف_سائز    DMA_MAX_سائز_16_B
#تعریف_کرو DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#ورنہ
#تعریف_کرو DMA_MAX_2_D_صف_سائز    DMA_MAX_سائز_24_B
#تعریف_کرو DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#اگر_ختم

#تعریف_کرو DMA_FALLBACK_استعداد  16u            // descriptors in secondary fallback ring

قسمتعریف ساخت dma_ring_s dma_ring;
ساخت dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * دم;      // tail pointer
    dma_ڈیٹا          * ڈیٹا;      // dst/src data
    uint32_t            ڈالو_idx;
    uint32_t            نکالو_idx;
    uint32_t            استعداد;
    uint32_t            idx_نقاب;
    ساخت htp_thread_trace * trace;
};

قسمتعریف ساخت dma_قطار_s dma_قطار;
قسمتعریف dma_قطار * dma_قطار_t;

ساخت dma_قطار_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_قطار_sizeof(size_t استعداد);
size_t      dma_قطار_alignof(خالی);
dma_قطار_t dma_قطار_init(خالی * ptr, size_t استعداد, ساخت htp_thread_trace * trace);
خالی        dma_قطار_free(dma_قطار_t q);

size_t      dma_قطار_alias_sizeof(خالی);
dma_قطار_t dma_قطار_alias_init(خالی * ptr, dma_قطار_t main_q);
خالی        dma_قطار_alias_free(dma_قطار_t q);

bool        dma_قطار_ڈالو_fallback_2d(dma_قطار * q, dma_ڈیٹا ddata, size_t dst_stride, size_t src_stride, size_t صف_سائز, size_t nrows);
bool        dma_قطار_ڈالو_fallback_contig(dma_قطار * q, dma_ڈیٹا ddata, size_t کل);
#اگر __HVX_ARCH__ < 75
bool        dma_قطار_ڈالو_fallback_1d(dma_قطار * q, dma_ڈیٹا ddata, size_t dst_stride, size_t src_stride, size_t صف_سائز, size_t nrows);
#اگر_ختم

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
ساکن اندرونی خالی dmstart(خالی * اگلا) {
    asm متلون(" release(%0):at" : : "r"(اگلا));
    asm متلون(" dmstart(%0)" : : "r"(اگلا));
}

ساکن اندرونی خالی dmlink(خالی * cur, خالی * اگلا) {
    asm متلون(" release(%0):at" : : "r"(اگلا));
    asm متلون(" dmlink(%0, %1)" : : "r"(cur), "r"(اگلا));
}

ساکن اندرونی غیرعلامتی صحیح dmpoll(خالی) {
    غیرعلامتی صحیح ret = 0;
    asm متلون(" %0 = dmpoll" : "=r"(ret) : : "memory");
    واپس ret;
}

ساکن اندرونی غیرعلامتی صحیح dmwait(خالی) {
    غیرعلامتی صحیح ret = 0;
    asm متلون(" %0 = dmwait" : "=r"(ret) : : "memory");
    واپس ret;
}

ساکن اندرونی dma_ڈیٹا dma_make_ڈیٹا_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_ڈیٹا d = { dst, src };
    واپس d;
}

#تعریف_کرو dma_make_ڈیٹا(dst, src) dma_make_ڈیٹا_impl((dma_addr_t) (dst), (dma_addr_t) (src))

ساکن اندرونی bool dma_ring_ڈالو_single_1d(dma_ring * r, dma_ڈیٹا ddata, size_t سائز) {
#اگر __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 32) || سائز == 0);
#اگر_ختم

    اگر (((r->ڈالو_idx + 1) & r->idx_نقاب) == r->نکالو_idx) {
        واپس false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->ڈالو_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->سائز = سائز;

    r->ڈیٹا[r->ڈالو_idx] = ddata;

    htp_trace_event_آغاز(r->trace, HTP_TRACE_EVT_DMA, r->ڈالو_idx);

    اگر (سائز) {
        desc->اگلا       = تہی;
        desc->desc_سائز  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->ترتیب      = 0;
        desc->مکمل       = 0;

        dmlink(r->دم, desc);
        r->دم = (dma_descriptor_2d *) desc;
    } ورنہ {
        desc->desc_سائز = 0;
        desc->مکمل      = 1;
    }

    r->ڈالو_idx = (r->ڈالو_idx + 1) & r->idx_نقاب;
    واپس true;
}

ساکن اندرونی bool dma_ring_ڈالو_single_2d(dma_ring * r, dma_ڈیٹا ddata, size_t dst_stride, size_t src_stride, size_t صف_سائز, size_t nrows) {
#اگر __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ثابت uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ثابت uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ثابت bool is_ext     = (src_hi | dst_hi) != 0;
#اگر_ختم

    اگر (((r->ڈالو_idx + 1) & r->idx_نقاب) == r->نکالو_idx) {
        واپس false;
    }

    dma_descriptor_2d * desc = &r->desc[r->ڈالو_idx];

    desc->اگلا           = تہی;
    desc->reserved1      = 0;
    desc->desc_سائز      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->ترتیب          = 0;
    desc->مکمل           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->صف_سائز       = صف_سائز;

#اگر __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_انحراف     = 0;
    desc->dst_انحراف     = 0;
#ورنہ
#اگر __HVX_ARCH__ > 79
    desc->src_بالائی      = src_hi;
    desc->dst_بالائی      = dst_hi;
    desc->تخصیص     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#ورنہ
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#اگر_ختم
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->انحراف         = 0;
#اگر_ختم

    r->ڈیٹا[r->ڈالو_idx] = ddata;

    htp_trace_event_آغاز(r->trace, HTP_TRACE_EVT_DMA, r->ڈالو_idx);

    اگر (nrows) {
        dmlink(r->دم, desc);
        r->دم = desc;
    } ورنہ {
        desc->مکمل = 1;
    }

    r->ڈالو_idx = (r->ڈالو_idx + 1) & r->idx_نقاب;
    واپس true;
}

#اگر __HVX_ARCH__ < 75
ساکن اندرونی bool dma_ring_ڈالو_single_contig(dma_ring * r, dma_ڈیٹا ddata, size_t سائز) {
    واپس dma_ring_ڈالو_single_1d(r, ddata, سائز);
}
#ورنہ
ساکن اندرونی bool dma_ring_ڈالو_single_contig(dma_ring * r, dma_ڈیٹا ddata, size_t سائز) {
    واپس dma_ring_ڈالو_single_2d(r, ddata, سائز, سائز, سائز, 1);
}
#اگر_ختم

ساکن اندرونی dma_ڈیٹا dma_ring_نکالو(dma_ring * r) {
    dma_ڈیٹا ddata = { 0 };

    اگر (r->ڈالو_idx == r->نکالو_idx) {
        واپس ddata;
    }

    ddata = r->ڈیٹا[r->نکالو_idx];

    متلون dma_descriptor_2d * desc = &r->desc[r->نکالو_idx];

    // Wait for desc to complete
    اگر (!desc->مکمل) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        جب (!desc->مکمل) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->نکالو_idx);

    r->نکالو_idx = (r->نکالو_idx + 1) & r->idx_نقاب;
    واپس ddata;
}

ساکن اندرونی dma_ڈیٹا dma_ring_نکالو_nowait(dma_ring * r) {
    dma_ڈیٹا ddata = { 0 };

    اگر (r->ڈالو_idx == r->نکالو_idx) {
        واپس ddata;
    }

    ddata = r->ڈیٹا[r->نکالو_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->نکالو_idx);

    r->نکالو_idx = (r->نکالو_idx + 1) & r->idx_نقاب;
    واپس ddata;
}

ساکن اندرونی bool dma_ring_خالی_خانے(dma_ring * r) {
    واپس r->ڈالو_idx == r->نکالو_idx;
}

ساکن اندرونی خالی dma_ring_flush(dma_ring * r) {
    جب (!dma_ring_خالی_خانے(r)) {
        dma_ring_نکالو(r);
    }
}

ساکن اندرونی uint32_t dma_ring_گہرائی(dma_ring * r) {
    واپس (r->ڈالو_idx - r->نکالو_idx) & r->idx_نقاب;
}

ساکن اندرونی uint32_t dma_ring_استعداد(dma_ring * r) {
    واپس r->استعداد;
}

ساکن اندرونی bool dma_قطار_ڈالو_single_1d(dma_قطار * q, dma_ڈیٹا ddata, size_t سائز) {
    واپس dma_ring_ڈالو_single_1d(q->ring0, ddata, سائز);
}

ساکن اندرونی bool dma_قطار_ڈالو_single_2d(dma_قطار * q, dma_ڈیٹا ddata, size_t dst_stride, size_t src_stride, size_t صف_سائز, size_t nrows) {
    واپس dma_ring_ڈالو_single_2d(q->ring0, ddata, dst_stride, src_stride, صف_سائز, nrows);
}

ساکن اندرونی dma_ڈیٹا dma_قطار_نکالو(dma_قطار * q) {
    واپس dma_ring_نکالو(q->ring0);
}

ساکن اندرونی dma_ڈیٹا dma_قطار_نکالو_nowait(dma_قطار * q) {
    واپس dma_ring_نکالو_nowait(q->ring0);
}

ساکن اندرونی bool dma_قطار_خالی_خانے(dma_قطار * q) {
    واپس dma_ring_خالی_خانے(q->ring0);
}

ساکن اندرونی خالی dma_قطار_flush(dma_قطار * q) {
    dma_ring_flush(q->ring0);
}

ساکن اندرونی uint32_t dma_قطار_گہرائی(dma_قطار * q) {
    واپس dma_ring_گہرائی(q->ring0);
}

ساکن اندرونی uint32_t dma_قطار_استعداد(dma_قطار * q) {
    واپس dma_ring_استعداد(q->ring0);
}

ساکن اندرونی bool dma_قطار_ڈالو(dma_قطار * q, dma_ڈیٹا ddata, size_t dst_stride, size_t src_stride, size_t صف_سائز, size_t nrows) {
    اگر (__builtin_expect(nrows == 0, 0)) {
        واپس dma_ring_ڈالو_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    اگر (nrows == 1 || !((صف_سائز ^ src_stride) | (صف_سائز ^ dst_stride))) {
        ثابت size_t کل = صف_سائز * nrows;
        اگر (__builtin_expect(کل <= DMA_MAX_سائز_24_B, 1)) {
            واپس dma_ring_ڈالو_single_contig(q->ring0, ddata, کل);
        }
        واپس dma_قطار_ڈالو_fallback_contig(q, ddata, کل);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    اگر (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (صف_سائز | src_stride | dst_stride) <= DMA_MAX_2_D_صف_سائز, 1)) {
        واپس dma_ring_ڈالو_single_2d(q->ring0, ddata, dst_stride, src_stride, صف_سائز, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#اگر __HVX_ARCH__ < 75
    اگر (صف_سائز <= DMA_MAX_سائز_16_B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        واپس dma_قطار_ڈالو_fallback_2d(q, ddata, dst_stride, src_stride, صف_سائز, nrows);
    }
    واپس dma_قطار_ڈالو_fallback_1d(q, ddata, dst_stride, src_stride, صف_سائز, nrows);
#ورنہ
    واپس dma_قطار_ڈالو_fallback_2d(q, ddata, dst_stride, src_stride, صف_سائز, nrows);
#اگر_ختم
}

ساکن اندرونی خالی dma_sync_read(dma_قطار * dma_q, خالی * dst, dma_addr_t src, size_t بائٹیں) {
    ثابت uint32_t b = (uint32_t) بائٹیں;
    اگر (b > 0) {
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst, src), b, b, b, 1);
        dma_قطار_نکالو(dma_q);
    }
}

ساکن اندرونی خالی dma_sync_write(dma_قطار * dma_q, dma_addr_t dst, ثابت خالی * src, size_t بائٹیں) {
    ثابت uint32_t b = (uint32_t) بائٹیں;
    اگر (b > 0) {
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst, src), b, b, b, 1);
        dma_قطار_نکالو(dma_q);
    }
}

#تعریف_کرو DMA_CACHE_MAX_سائز 256U

// Fully assoc LRU cache
قسمتعریف ساخت {
    uint8_t *بنیاد;
    uint32_t سطر_سائز;
    uint32_t استعداد;
    dma_addr_t src[DMA_CACHE_MAX_سائز];
    uint16_t age[DMA_CACHE_MAX_سائز];
} dma_cache_fa;

ساکن اندرونی خالی dma_cache_fa_init(dma_cache_fa *c, uint8_t *بنیاد, uint32_t سطر_سائز, uint32_t استعداد)
{
    c->استعداد  = (استعداد > DMA_CACHE_MAX_سائز) ? DMA_CACHE_MAX_سائز : استعداد;
    c->بنیاد      = بنیاد;
    c->سطر_سائز = سطر_سائز;

    برائے (غیرعلامتی i=0; i < c->استعداد; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

ساکن اندرونی bool dma_cache_fa_ڈالو(dma_قطار *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t صف_سائز, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    برائے (غیرعلامتی i=0; i < c->استعداد; i++) {
        اگر (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->بنیاد + (i * c->سطر_سائز); nrows = 0; // dummy dma
        } ورنہ {
            c->age[i]++;
            اگر (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    اگر (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->بنیاد + o_idx * c->سطر_سائز; // normal nrows dma
        واپس dma_قطار_ڈالو(q, dma_make_ڈیٹا(dst, src_addr), dst_stride, src_stride, صف_سائز, nrows);
    }

    واپس dma_قطار_ڈالو_single_1d(q, dma_make_ڈیٹا(dst, src_addr), 0);
}

// Direct mapped cache
قسمتعریف ساخت {
    uint8_t *بنیاد;
    uint32_t سطر_سائز;
    uint32_t استعداد;
    uint32_t idx_نقاب;
    dma_addr_t src[DMA_CACHE_MAX_سائز];
} dma_cache_dm;

ساکن اندرونی خالی dma_cache_dm_init(dma_cache_dm *c, uint8_t *بنیاد, uint32_t سطر_سائز, uint32_t استعداد)
{
    c->استعداد  = (استعداد > DMA_CACHE_MAX_سائز) ? DMA_CACHE_MAX_سائز : استعداد;
    c->idx_نقاب  = c->استعداد - 1;
    c->بنیاد      = بنیاد;
    c->سطر_سائز = سطر_سائز;

    برائے (غیرعلامتی i=0; i < c->استعداد; i++) {
        c->src[i] = 0;
    }
}

ساکن اندرونی bool dma_cache_dm_ڈالو(dma_قطار *q, dma_cache_dm *c, uint32_t خانہ, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t صف_سائز, uint32_t nrows)
{
    ثابت uint32_t i = خانہ & c->idx_نقاب;
    uint8_t * dst = c->بنیاد + (i * c->سطر_سائز);

    اگر (c->src[i] == src_addr) {
        واپس dma_قطار_ڈالو_single_1d(q, dma_make_ڈیٹا(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    واپس dma_قطار_ڈالو(q, dma_make_ڈیٹا(dst, src_addr), dst_stride, src_stride, صف_سائز, nrows);
}

#اگر_تعریف_شدہ __cplusplus
}  // extern "C"
#اگر_ختم

#اگر_ختم /* HTP_DMA_H */
