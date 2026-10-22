#ifndef HTP_DMA_H
#define HTP_DMA_H

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <hexagon_types.h>
#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <string.h>
#അന്തര്ഭാവ "hex-utils.h"

#അന്തര്ഭാവ "hex-profile.h"

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
തരനിർവചനം ഘടന dma_descriptor_1d_s {
    ശൂന്യം *   next;
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

#എങ്കിൽ __HVX_ARCH__ < 75

തരനിർവചനം ഘടന dma_descriptor_2d_s {
    ശൂന്യം *   next;
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

#അല്ലെങ്കിൽ

തരനിർവചനം ഘടന dma_descriptor_2d_s {
    ശൂന്യം *   next;
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
#എങ്കിൽ __HVX_ARCH__ > 79
    uint32_t src_upper:8;
    uint32_t dst_upper:8;
    uint32_t allocation:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#അല്ലെങ്കിൽ
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

#എങ്കിൽ __HVX_ARCH__ > 79
തരനിർവചനം uint64_t dma_addr_t;
#അല്ലെങ്കിൽ
തരനിർവചനം uint32_t dma_addr_t;
#endif

തരനിർവചനം ഘടന {
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

#എങ്കിൽ __HVX_ARCH__ < 75
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_16B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#അല്ലെങ്കിൽ
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_24B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#endif

#define DMA_FALLBACK_CAPACITY  16u            // descriptors in secondary fallback ring

തരനിർവചനം ഘടന dma_ring_s dma_ring;
ഘടന dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * tail;      // tail pointer
    dma_data          * data;      // dst/src data
    uint32_t            തള്ളുക_idx;
    uint32_t            എടുക്കുക_idx;
    uint32_t            capacity;
    uint32_t            idx_mask;
    ഘടന htp_thread_trace * trace;
};

തരനിർവചനം ഘടന dma_queue_s dma_queue;
തരനിർവചനം dma_queue * dma_queue_t;

ഘടന dma_queue_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_queue_sizeof(size_t capacity);
size_t      dma_queue_alignof(ശൂന്യം);
dma_queue_t dma_queue_init(ശൂന്യം * ptr, size_t capacity, ഘടന htp_thread_trace * trace);
ശൂന്യം        dma_queue_free(dma_queue_t q);

size_t      dma_queue_alias_sizeof(ശൂന്യം);
dma_queue_t dma_queue_alias_init(ശൂന്യം * ptr, dma_queue_t main_q);
ശൂന്യം        dma_queue_alias_free(dma_queue_t q);

bool        dma_queue_തള്ളുക_fallback_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
bool        dma_queue_തള്ളുക_fallback_contig(dma_queue * q, dma_data ddata, size_t total);
#എങ്കിൽ __HVX_ARCH__ < 75
bool        dma_queue_തള്ളുക_fallback_1d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
#endif

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
സ്ഥിതം വരിയിൽ ശൂന്യം dmstart(ശൂന്യം * next) {
    asm അസ്ഥിര(" release(%0):at" : : "r"(next));
    asm അസ്ഥിര(" dmstart(%0)" : : "r"(next));
}

സ്ഥിതം വരിയിൽ ശൂന്യം dmlink(ശൂന്യം * cur, ശൂന്യം * next) {
    asm അസ്ഥിര(" release(%0):at" : : "r"(next));
    asm അസ്ഥിര(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

സ്ഥിതം വരിയിൽ unsigned പൂർണ്ണസംഖ്യ dmpoll(ശൂന്യം) {
    unsigned പൂർണ്ണസംഖ്യ ret = 0;
    asm അസ്ഥിര(" %0 = dmpoll" : "=r"(ret) : : "memory");
    തിരികെ ret;
}

സ്ഥിതം വരിയിൽ unsigned പൂർണ്ണസംഖ്യ dmwait(ശൂന്യം) {
    unsigned പൂർണ്ണസംഖ്യ ret = 0;
    asm അസ്ഥിര(" %0 = dmwait" : "=r"(ret) : : "memory");
    തിരികെ ret;
}

സ്ഥിതം വരിയിൽ dma_data dma_make_data_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_data d = { dst, src };
    തിരികെ d;
}

#define dma_make_data(dst, src) dma_make_data_impl((dma_addr_t) (dst), (dma_addr_t) (src))

സ്ഥിതം വരിയിൽ bool dma_ring_തള്ളുക_single_1d(dma_ring * r, dma_data ddata, size_t size) {
#എങ്കിൽ __HVX_ARCH__ > 79
    പ്രതിജ്ഞാ(!((ddata.src | ddata.dst) >> 32) || size == 0);
#endif

    എങ്കിൽ (((r->തള്ളുക_idx + 1) & r->idx_mask) == r->എടുക്കുക_idx) {
        തിരികെ false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->തള്ളുക_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->size = size;

    r->data[r->തള്ളുക_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->തള്ളുക_idx);

    എങ്കിൽ (size) {
        desc->next       = ശൂന്യ;
        desc->desc_size  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->order      = 0;
        desc->done       = 0;

        dmlink(r->tail, desc);
        r->tail = (dma_descriptor_2d *) desc;
    } അല്ലെങ്കിൽ {
        desc->desc_size = 0;
        desc->done      = 1;
    }

    r->തള്ളുക_idx = (r->തള്ളുക_idx + 1) & r->idx_mask;
    തിരികെ true;
}

സ്ഥിതം വരിയിൽ bool dma_ring_തള്ളുക_single_2d(dma_ring * r, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
#എങ്കിൽ __HVX_ARCH__ > 79
    പ്രതിജ്ഞാ(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    സ്ഥിരം uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    സ്ഥിരം uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    സ്ഥിരം bool is_ext     = (src_hi | dst_hi) != 0;
#endif

    എങ്കിൽ (((r->തള്ളുക_idx + 1) & r->idx_mask) == r->എടുക്കുക_idx) {
        തിരികെ false;
    }

    dma_descriptor_2d * desc = &r->desc[r->തള്ളുക_idx];

    desc->next           = ശൂന്യ;
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

#എങ്കിൽ __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_offset     = 0;
    desc->dst_offset     = 0;
#അല്ലെങ്കിൽ
#എങ്കിൽ __HVX_ARCH__ > 79
    desc->src_upper      = src_hi;
    desc->dst_upper      = dst_hi;
    desc->allocation     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#അല്ലെങ്കിൽ
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#endif
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->offset         = 0;
#endif

    r->data[r->തള്ളുക_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->തള്ളുക_idx);

    എങ്കിൽ (nrows) {
        dmlink(r->tail, desc);
        r->tail = desc;
    } അല്ലെങ്കിൽ {
        desc->done = 1;
    }

    r->തള്ളുക_idx = (r->തള്ളുക_idx + 1) & r->idx_mask;
    തിരികെ true;
}

#എങ്കിൽ __HVX_ARCH__ < 75
സ്ഥിതം വരിയിൽ bool dma_ring_തള്ളുക_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    തിരികെ dma_ring_തള്ളുക_single_1d(r, ddata, size);
}
#അല്ലെങ്കിൽ
സ്ഥിതം വരിയിൽ bool dma_ring_തള്ളുക_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    തിരികെ dma_ring_തള്ളുക_single_2d(r, ddata, size, size, size, 1);
}
#endif

സ്ഥിതം വരിയിൽ dma_data dma_ring_എടുക്കുക(dma_ring * r) {
    dma_data ddata = { 0 };

    എങ്കിൽ (r->തള്ളുക_idx == r->എടുക്കുക_idx) {
        തിരികെ ddata;
    }

    ddata = r->data[r->എടുക്കുക_idx];

    അസ്ഥിര dma_descriptor_2d * desc = &r->desc[r->എടുക്കുക_idx];

    // Wait for desc to complete
    എങ്കിൽ (!desc->done) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        ആകുമ്പോൾ (!desc->done) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->എടുക്കുക_idx);

    r->എടുക്കുക_idx = (r->എടുക്കുക_idx + 1) & r->idx_mask;
    തിരികെ ddata;
}

സ്ഥിതം വരിയിൽ dma_data dma_ring_എടുക്കുക_nowait(dma_ring * r) {
    dma_data ddata = { 0 };

    എങ്കിൽ (r->തള്ളുക_idx == r->എടുക്കുക_idx) {
        തിരികെ ddata;
    }

    ddata = r->data[r->എടുക്കുക_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->എടുക്കുക_idx);

    r->എടുക്കുക_idx = (r->എടുക്കുക_idx + 1) & r->idx_mask;
    തിരികെ ddata;
}

സ്ഥിതം വരിയിൽ bool dma_ring_empty(dma_ring * r) {
    തിരികെ r->തള്ളുക_idx == r->എടുക്കുക_idx;
}

സ്ഥിതം വരിയിൽ ശൂന്യം dma_ring_flush(dma_ring * r) {
    ആകുമ്പോൾ (!dma_ring_empty(r)) {
        dma_ring_എടുക്കുക(r);
    }
}

സ്ഥിതം വരിയിൽ uint32_t dma_ring_depth(dma_ring * r) {
    തിരികെ (r->തള്ളുക_idx - r->എടുക്കുക_idx) & r->idx_mask;
}

സ്ഥിതം വരിയിൽ uint32_t dma_ring_capacity(dma_ring * r) {
    തിരികെ r->capacity;
}

സ്ഥിതം വരിയിൽ bool dma_queue_തള്ളുക_single_1d(dma_queue * q, dma_data ddata, size_t size) {
    തിരികെ dma_ring_തള്ളുക_single_1d(q->ring0, ddata, size);
}

സ്ഥിതം വരിയിൽ bool dma_queue_തള്ളുക_single_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    തിരികെ dma_ring_തള്ളുക_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
}

സ്ഥിതം വരിയിൽ dma_data dma_queue_എടുക്കുക(dma_queue * q) {
    തിരികെ dma_ring_എടുക്കുക(q->ring0);
}

സ്ഥിതം വരിയിൽ dma_data dma_queue_എടുക്കുക_nowait(dma_queue * q) {
    തിരികെ dma_ring_എടുക്കുക_nowait(q->ring0);
}

സ്ഥിതം വരിയിൽ bool dma_queue_empty(dma_queue * q) {
    തിരികെ dma_ring_empty(q->ring0);
}

സ്ഥിതം വരിയിൽ ശൂന്യം dma_queue_flush(dma_queue * q) {
    dma_ring_flush(q->ring0);
}

സ്ഥിതം വരിയിൽ uint32_t dma_queue_depth(dma_queue * q) {
    തിരികെ dma_ring_depth(q->ring0);
}

സ്ഥിതം വരിയിൽ uint32_t dma_queue_capacity(dma_queue * q) {
    തിരികെ dma_ring_capacity(q->ring0);
}

സ്ഥിതം വരിയിൽ bool dma_queue_തള്ളുക(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    എങ്കിൽ (__builtin_expect(nrows == 0, 0)) {
        തിരികെ dma_ring_തള്ളുക_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    എങ്കിൽ (nrows == 1 || !((row_size ^ src_stride) | (row_size ^ dst_stride))) {
        സ്ഥിരം size_t total = row_size * nrows;
        എങ്കിൽ (__builtin_expect(total <= DMA_MAX_SIZE_24B, 1)) {
            തിരികെ dma_ring_തള്ളുക_single_contig(q->ring0, ddata, total);
        }
        തിരികെ dma_queue_തള്ളുക_fallback_contig(q, ddata, total);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    എങ്കിൽ (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (row_size | src_stride | dst_stride) <= DMA_MAX_2D_ROW_SIZE, 1)) {
        തിരികെ dma_ring_തള്ളുക_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#എങ്കിൽ __HVX_ARCH__ < 75
    എങ്കിൽ (row_size <= DMA_MAX_SIZE_16B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        തിരികെ dma_queue_തള്ളുക_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
    }
    തിരികെ dma_queue_തള്ളുക_fallback_1d(q, ddata, dst_stride, src_stride, row_size, nrows);
#അല്ലെങ്കിൽ
    തിരികെ dma_queue_തള്ളുക_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
#endif
}

സ്ഥിതം വരിയിൽ ശൂന്യം dma_sync_read(dma_queue * dma_q, ശൂന്യം * dst, dma_addr_t src, size_t bytes) {
    സ്ഥിരം uint32_t b = (uint32_t) bytes;
    എങ്കിൽ (b > 0) {
        dma_queue_തള്ളുക(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_എടുക്കുക(dma_q);
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം dma_sync_write(dma_queue * dma_q, dma_addr_t dst, സ്ഥിരം ശൂന്യം * src, size_t bytes) {
    സ്ഥിരം uint32_t b = (uint32_t) bytes;
    എങ്കിൽ (b > 0) {
        dma_queue_തള്ളുക(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_എടുക്കുക(dma_q);
    }
}

#define DMA_CACHE_MAX_SIZE 256U

// Fully assoc LRU cache
തരനിർവചനം ഘടന {
    uint8_t *base;
    uint32_t വരി_size;
    uint32_t capacity;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
    uint16_t age[DMA_CACHE_MAX_SIZE];
} dma_cache_fa;

സ്ഥിതം വരിയിൽ ശൂന്യം dma_cache_fa_init(dma_cache_fa *c, uint8_t *base, uint32_t വരി_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->വരി_size = വരി_size;

    വേണ്ടി (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

സ്ഥിതം വരിയിൽ bool dma_cache_fa_തള്ളുക(dma_queue *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    വേണ്ടി (unsigned i=0; i < c->capacity; i++) {
        എങ്കിൽ (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->base + (i * c->വരി_size); nrows = 0; // dummy dma
        } അല്ലെങ്കിൽ {
            c->age[i]++;
            എങ്കിൽ (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    എങ്കിൽ (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->base + o_idx * c->വരി_size; // normal nrows dma
        തിരികെ dma_queue_തള്ളുക(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
    }

    തിരികെ dma_queue_തള്ളുക_single_1d(q, dma_make_data(dst, src_addr), 0);
}

// Direct mapped cache
തരനിർവചനം ഘടന {
    uint8_t *base;
    uint32_t വരി_size;
    uint32_t capacity;
    uint32_t idx_mask;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
} dma_cache_dm;

സ്ഥിതം വരിയിൽ ശൂന്യം dma_cache_dm_init(dma_cache_dm *c, uint8_t *base, uint32_t വരി_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->idx_mask  = c->capacity - 1;
    c->base      = base;
    c->വരി_size = വരി_size;

    വേണ്ടി (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
    }
}

സ്ഥിതം വരിയിൽ bool dma_cache_dm_തള്ളുക(dma_queue *q, dma_cache_dm *c, uint32_t slot, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    സ്ഥിരം uint32_t i = slot & c->idx_mask;
    uint8_t * dst = c->base + (i * c->വരി_size);

    എങ്കിൽ (c->src[i] == src_addr) {
        തിരികെ dma_queue_തള്ളുക_single_1d(q, dma_make_data(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    തിരികെ dma_queue_തള്ളുക(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
