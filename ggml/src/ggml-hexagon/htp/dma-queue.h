#ifndef HTP_DMA_H
#define HTP_DMA_H

#ضمن <HAP_farf.h>
#ضمن <hexagon_types.h>
#ضمن <assert.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <string.h>
#ضمن "hex-utils.h"

#ضمن "hex-profile.h"

#ifdef __cplusplus
خارجي "C" {
#endif

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
تعريفنوع بنية dma_descriptor_1d_s {
    فراغ *   next;
    uint32_t size:24;
    uint32_t desc_size:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#𐤀𐤌 __HVX_ARCH__ < 75

تعريفنوع بنية dma_descriptor_2d_s {
    فراغ *   next;
    uint32_t reserved0:24;
    uint32_t desc_size:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t row_size:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_offset:16;
    uint32_t dst_offset:16;
} dma_descriptor_2d;

#وإلا

تعريفنوع بنية dma_descriptor_2d_s {
    فراغ *   next;
    uint32_t dst_stride:24;
    uint32_t desc_size:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#𐤀𐤌 __HVX_ARCH__ > 79
    uint32_t src_upper:8;
    uint32_t dst_upper:8;
    uint32_t allocation:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#وإلا
    uint32_t reserved0:24;
#endif
    uint32_t row_size:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t offset:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#endif

#𐤀𐤌 __HVX_ARCH__ > 79
تعريفنوع uint64_t dma_addr_t;
#وإلا
تعريفنوع uint32_t dma_addr_t;
#endif

تعريفنوع بنية {
    dma_addr_t dst;
    dma_addr_t src;
} dma_data;

// Hardware descriptor field limits
#define DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#define DMA_MAX_SIZE_16B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#define DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#define DMA_MAX_SIZE_24B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#define DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#define DMA_SAFE_CHUNK_SIZE    0x00F00000u    // ~15MB safe contiguous chunk size

#𐤀𐤌 __HVX_ARCH__ < 75
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_16B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#وإلا
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_24B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#endif

#define DMA_FALLBACK_CAPACITY  16u            // descriptors in secondary fallback ring

تعريفنوع بنية dma_ring_s dma_ring;
بنية dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * tail;      // tail pointer
    dma_data          * data;      // dst/src data
    uint32_t            push_idx;
    uint32_t            pop_idx;
    uint32_t            capacity;
    uint32_t            idx_mask;
    بنية htp_thread_trace * trace;
};

تعريفنوع بنية dma_queue_s dma_queue;
تعريفنوع dma_queue * dma_queue_t;

بنية dma_queue_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_queue_sizeof(size_t capacity);
size_t      dma_queue_alignof(فراغ);
dma_queue_t dma_queue_init(فراغ * ptr, size_t capacity, بنية htp_thread_trace * trace);
فراغ        dma_queue_free(dma_queue_t q);

size_t      dma_queue_alias_sizeof(فراغ);
dma_queue_t dma_queue_alias_init(فراغ * ptr, dma_queue_t main_q);
فراغ        dma_queue_alias_free(dma_queue_t q);

bool        dma_queue_push_fallback_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
bool        dma_queue_push_fallback_contig(dma_queue * q, dma_data ddata, size_t total);
#𐤀𐤌 __HVX_ARCH__ < 75
bool        dma_queue_push_fallback_1d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
#endif

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
ساكن مضمن فراغ dmstart(فراغ * next) {
    asm متطاير(" release(%0):at" : : "r"(next));
    asm متطاير(" dmstart(%0)" : : "r"(next));
}

ساكن مضمن فراغ dmlink(فراغ * cur, فراغ * next) {
    asm متطاير(" release(%0):at" : : "r"(next));
    asm متطاير(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

ساكن مضمن unsigned صحيح dmpoll(فراغ) {
    unsigned صحيح ret = 0;
    asm متطاير(" %0 = dmpoll" : "=r"(ret) : : "memory");
    أرجع ret;
}

ساكن مضمن unsigned صحيح dmwait(فراغ) {
    unsigned صحيح ret = 0;
    asm متطاير(" %0 = dmwait" : "=r"(ret) : : "memory");
    أرجع ret;
}

ساكن مضمن dma_data dma_make_data_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_data d = { dst, src };
    أرجع d;
}

#define dma_make_data(dst, src) dma_make_data_impl((dma_addr_t) (dst), (dma_addr_t) (src))

ساكن مضمن bool dma_ring_push_single_1d(dma_ring * r, dma_data ddata, size_t size) {
#𐤀𐤌 __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 32) || size == 0);
#endif

    𐤀𐤌 (((r->push_idx + 1) & r->idx_mask) == r->pop_idx) {
        أرجع false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->push_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->size = size;

    r->data[r->push_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->push_idx);

    𐤀𐤌 (size) {
        desc->next       = عدم;
        desc->desc_size  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->order      = 0;
        desc->done       = 0;

        dmlink(r->tail, desc);
        r->tail = (dma_descriptor_2d *) desc;
    } وإلا {
        desc->desc_size = 0;
        desc->done      = 1;
    }

    r->push_idx = (r->push_idx + 1) & r->idx_mask;
    أرجع true;
}

ساكن مضمن bool dma_ring_push_single_2d(dma_ring * r, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
#𐤀𐤌 __HVX_ARCH__ > 79
    أكد(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ثابت uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ثابت uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ثابت bool is_ext     = (src_hi | dst_hi) != 0;
#endif

    𐤀𐤌 (((r->push_idx + 1) & r->idx_mask) == r->pop_idx) {
        أرجع false;
    }

    dma_descriptor_2d * desc = &r->desc[r->push_idx];

    desc->next           = عدم;
    desc->reserved1      = 0;
    desc->desc_size      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->order          = 0;
    desc->done           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->row_size       = row_size;

#𐤀𐤌 __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_offset     = 0;
    desc->dst_offset     = 0;
#وإلا
#𐤀𐤌 __HVX_ARCH__ > 79
    desc->src_upper      = src_hi;
    desc->dst_upper      = dst_hi;
    desc->allocation     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#وإلا
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#endif
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->offset         = 0;
#endif

    r->data[r->push_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->push_idx);

    𐤀𐤌 (nrows) {
        dmlink(r->tail, desc);
        r->tail = desc;
    } وإلا {
        desc->done = 1;
    }

    r->push_idx = (r->push_idx + 1) & r->idx_mask;
    أرجع true;
}

#𐤀𐤌 __HVX_ARCH__ < 75
ساكن مضمن bool dma_ring_push_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    أرجع dma_ring_push_single_1d(r, ddata, size);
}
#وإلا
ساكن مضمن bool dma_ring_push_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    أرجع dma_ring_push_single_2d(r, ddata, size, size, size, 1);
}
#endif

ساكن مضمن dma_data dma_ring_pop(dma_ring * r) {
    dma_data ddata = { 0 };

    𐤀𐤌 (r->push_idx == r->pop_idx) {
        أرجع ddata;
    }

    ddata = r->data[r->pop_idx];

    متطاير dma_descriptor_2d * desc = &r->desc[r->pop_idx];

    // Wait for desc to complete
    𐤀𐤌 (!desc->done) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        طالما (!desc->done) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->pop_idx);

    r->pop_idx = (r->pop_idx + 1) & r->idx_mask;
    أرجع ddata;
}

ساكن مضمن dma_data dma_ring_pop_nowait(dma_ring * r) {
    dma_data ddata = { 0 };

    𐤀𐤌 (r->push_idx == r->pop_idx) {
        أرجع ddata;
    }

    ddata = r->data[r->pop_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->pop_idx);

    r->pop_idx = (r->pop_idx + 1) & r->idx_mask;
    أرجع ddata;
}

ساكن مضمن bool dma_ring_empty(dma_ring * r) {
    أرجع r->push_idx == r->pop_idx;
}

ساكن مضمن فراغ dma_ring_flush(dma_ring * r) {
    طالما (!dma_ring_empty(r)) {
        dma_ring_pop(r);
    }
}

ساكن مضمن uint32_t dma_ring_depth(dma_ring * r) {
    أرجع (r->push_idx - r->pop_idx) & r->idx_mask;
}

ساكن مضمن uint32_t dma_ring_capacity(dma_ring * r) {
    أرجع r->capacity;
}

ساكن مضمن bool dma_queue_push_single_1d(dma_queue * q, dma_data ddata, size_t size) {
    أرجع dma_ring_push_single_1d(q->ring0, ddata, size);
}

ساكن مضمن bool dma_queue_push_single_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    أرجع dma_ring_push_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
}

ساكن مضمن dma_data dma_queue_pop(dma_queue * q) {
    أرجع dma_ring_pop(q->ring0);
}

ساكن مضمن dma_data dma_queue_pop_nowait(dma_queue * q) {
    أرجع dma_ring_pop_nowait(q->ring0);
}

ساكن مضمن bool dma_queue_empty(dma_queue * q) {
    أرجع dma_ring_empty(q->ring0);
}

ساكن مضمن فراغ dma_queue_flush(dma_queue * q) {
    dma_ring_flush(q->ring0);
}

ساكن مضمن uint32_t dma_queue_depth(dma_queue * q) {
    أرجع dma_ring_depth(q->ring0);
}

ساكن مضمن uint32_t dma_queue_capacity(dma_queue * q) {
    أرجع dma_ring_capacity(q->ring0);
}

ساكن مضمن bool dma_queue_push(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    𐤀𐤌 (__builtin_expect(nrows == 0, 0)) {
        أرجع dma_ring_push_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    𐤀𐤌 (nrows == 1 || !((row_size ^ src_stride) | (row_size ^ dst_stride))) {
        ثابت size_t total = row_size * nrows;
        𐤀𐤌 (__builtin_expect(total <= DMA_MAX_SIZE_24B, 1)) {
            أرجع dma_ring_push_single_contig(q->ring0, ddata, total);
        }
        أرجع dma_queue_push_fallback_contig(q, ddata, total);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    𐤀𐤌 (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (row_size | src_stride | dst_stride) <= DMA_MAX_2D_ROW_SIZE, 1)) {
        أرجع dma_ring_push_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#𐤀𐤌 __HVX_ARCH__ < 75
    𐤀𐤌 (row_size <= DMA_MAX_SIZE_16B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        أرجع dma_queue_push_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
    }
    أرجع dma_queue_push_fallback_1d(q, ddata, dst_stride, src_stride, row_size, nrows);
#وإلا
    أرجع dma_queue_push_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
#endif
}

ساكن مضمن فراغ dma_sync_read(dma_queue * dma_q, فراغ * dst, dma_addr_t src, size_t bytes) {
    ثابت uint32_t b = (uint32_t) bytes;
    𐤀𐤌 (b > 0) {
        dma_queue_push(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_pop(dma_q);
    }
}

ساكن مضمن فراغ dma_sync_write(dma_queue * dma_q, dma_addr_t dst, ثابت فراغ * src, size_t bytes) {
    ثابت uint32_t b = (uint32_t) bytes;
    𐤀𐤌 (b > 0) {
        dma_queue_push(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_pop(dma_q);
    }
}

#define DMA_CACHE_MAX_SIZE 256U

// Fully assoc LRU cache
تعريفنوع بنية {
    uint8_t *base;
    uint32_t line_size;
    uint32_t capacity;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
    uint16_t age[DMA_CACHE_MAX_SIZE];
} dma_cache_fa;

ساكن مضمن فراغ dma_cache_fa_init(dma_cache_fa *c, uint8_t *base, uint32_t line_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->line_size = line_size;

    لكل (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

ساكن مضمن bool dma_cache_fa_push(dma_queue *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    لكل (unsigned i=0; i < c->capacity; i++) {
        𐤀𐤌 (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->base + (i * c->line_size); nrows = 0; // dummy dma
        } وإلا {
            c->age[i]++;
            𐤀𐤌 (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    𐤀𐤌 (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->base + o_idx * c->line_size; // normal nrows dma
        أرجع dma_queue_push(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
    }

    أرجع dma_queue_push_single_1d(q, dma_make_data(dst, src_addr), 0);
}

// Direct mapped cache
تعريفنوع بنية {
    uint8_t *base;
    uint32_t line_size;
    uint32_t capacity;
    uint32_t idx_mask;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
} dma_cache_dm;

ساكن مضمن فراغ dma_cache_dm_init(dma_cache_dm *c, uint8_t *base, uint32_t line_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->idx_mask  = c->capacity - 1;
    c->base      = base;
    c->line_size = line_size;

    لكل (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
    }
}

ساكن مضمن bool dma_cache_dm_push(dma_queue *q, dma_cache_dm *c, uint32_t slot, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    ثابت uint32_t i = slot & c->idx_mask;
    uint8_t * dst = c->base + (i * c->line_size);

    𐤀𐤌 (c->src[i] == src_addr) {
        أرجع dma_queue_push_single_1d(q, dma_make_data(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    أرجع dma_queue_push(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
