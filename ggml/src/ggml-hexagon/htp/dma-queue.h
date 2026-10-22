#ifndef HTP_DMA_H
#define HTP_DMA_H

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <hexagon_types.h>
#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <string.h>
#அந்தர்பாவ "hex-utils.h"

#அந்தர்பாவ "hex-profile.h"

#ifdef __cplusplus
வெளி "C" {
#endif

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
வகைவரை கட்டமைப்பு dma_descriptor_1d_s {
    வெறுமை *   next;
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

#எனில் __HVX_ARCH__ < 75

வகைவரை கட்டமைப்பு dma_descriptor_2d_s {
    வெறுமை *   next;
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

#இல்லையெனில்

வகைவரை கட்டமைப்பு dma_descriptor_2d_s {
    வெறுமை *   next;
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
#எனில் __HVX_ARCH__ > 79
    uint32_t src_upper:8;
    uint32_t dst_upper:8;
    uint32_t allocation:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#இல்லையெனில்
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

#எனில் __HVX_ARCH__ > 79
வகைவரை uint64_t dma_addr_t;
#இல்லையெனில்
வகைவரை uint32_t dma_addr_t;
#endif

வகைவரை கட்டமைப்பு {
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

#எனில் __HVX_ARCH__ < 75
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_16B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#இல்லையெனில்
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_24B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#endif

#define DMA_FALLBACK_CAPACITY  16u            // descriptors in secondary fallback ring

வகைவரை கட்டமைப்பு dma_ring_s dma_ring;
கட்டமைப்பு dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * tail;      // tail pointer
    dma_data          * data;      // dst/src data
    uint32_t            தள்ளு_idx;
    uint32_t            எடு_idx;
    uint32_t            capacity;
    uint32_t            idx_mask;
    கட்டமைப்பு htp_thread_trace * trace;
};

வகைவரை கட்டமைப்பு dma_queue_s dma_queue;
வகைவரை dma_queue * dma_queue_t;

கட்டமைப்பு dma_queue_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_queue_sizeof(size_t capacity);
size_t      dma_queue_alignof(வெறுமை);
dma_queue_t dma_queue_init(வெறுமை * ptr, size_t capacity, கட்டமைப்பு htp_thread_trace * trace);
வெறுமை        dma_queue_free(dma_queue_t q);

size_t      dma_queue_alias_sizeof(வெறுமை);
dma_queue_t dma_queue_alias_init(வெறுமை * ptr, dma_queue_t main_q);
வெறுமை        dma_queue_alias_free(dma_queue_t q);

bool        dma_queue_தள்ளு_fallback_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
bool        dma_queue_தள்ளு_fallback_contig(dma_queue * q, dma_data ddata, size_t total);
#எனில் __HVX_ARCH__ < 75
bool        dma_queue_தள்ளு_fallback_1d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
#endif

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
நிலையான உள்வரி வெறுமை dmstart(வெறுமை * next) {
    asm நிலையற்ற(" release(%0):at" : : "r"(next));
    asm நிலையற்ற(" dmstart(%0)" : : "r"(next));
}

நிலையான உள்வரி வெறுமை dmlink(வெறுமை * cur, வெறுமை * next) {
    asm நிலையற்ற(" release(%0):at" : : "r"(next));
    asm நிலையற்ற(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

நிலையான உள்வரி unsigned முழு dmpoll(வெறுமை) {
    unsigned முழு ret = 0;
    asm நிலையற்ற(" %0 = dmpoll" : "=r"(ret) : : "memory");
    திரும்பு ret;
}

நிலையான உள்வரி unsigned முழு dmwait(வெறுமை) {
    unsigned முழு ret = 0;
    asm நிலையற்ற(" %0 = dmwait" : "=r"(ret) : : "memory");
    திரும்பு ret;
}

நிலையான உள்வரி dma_data dma_make_data_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_data d = { dst, src };
    திரும்பு d;
}

#define dma_make_data(dst, src) dma_make_data_impl((dma_addr_t) (dst), (dma_addr_t) (src))

நிலையான உள்வரி bool dma_ring_தள்ளு_single_1d(dma_ring * r, dma_data ddata, size_t size) {
#எனில் __HVX_ARCH__ > 79
    ப்ரதிஜ்ஞா(!((ddata.src | ddata.dst) >> 32) || size == 0);
#endif

    எனில் (((r->தள்ளு_idx + 1) & r->idx_mask) == r->எடு_idx) {
        திரும்பு false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->தள்ளு_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->size = size;

    r->data[r->தள்ளு_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->தள்ளு_idx);

    எனில் (size) {
        desc->next       = வெற்று;
        desc->desc_size  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->order      = 0;
        desc->done       = 0;

        dmlink(r->tail, desc);
        r->tail = (dma_descriptor_2d *) desc;
    } இல்லையெனில் {
        desc->desc_size = 0;
        desc->done      = 1;
    }

    r->தள்ளு_idx = (r->தள்ளு_idx + 1) & r->idx_mask;
    திரும்பு true;
}

நிலையான உள்வரி bool dma_ring_தள்ளு_single_2d(dma_ring * r, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
#எனில் __HVX_ARCH__ > 79
    ப்ரதிஜ்ஞா(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    மாறிலி uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    மாறிலி uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    மாறிலி bool is_ext     = (src_hi | dst_hi) != 0;
#endif

    எனில் (((r->தள்ளு_idx + 1) & r->idx_mask) == r->எடு_idx) {
        திரும்பு false;
    }

    dma_descriptor_2d * desc = &r->desc[r->தள்ளு_idx];

    desc->next           = வெற்று;
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

#எனில் __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_offset     = 0;
    desc->dst_offset     = 0;
#இல்லையெனில்
#எனில் __HVX_ARCH__ > 79
    desc->src_upper      = src_hi;
    desc->dst_upper      = dst_hi;
    desc->allocation     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#இல்லையெனில்
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#endif
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->offset         = 0;
#endif

    r->data[r->தள்ளு_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->தள்ளு_idx);

    எனில் (nrows) {
        dmlink(r->tail, desc);
        r->tail = desc;
    } இல்லையெனில் {
        desc->done = 1;
    }

    r->தள்ளு_idx = (r->தள்ளு_idx + 1) & r->idx_mask;
    திரும்பு true;
}

#எனில் __HVX_ARCH__ < 75
நிலையான உள்வரி bool dma_ring_தள்ளு_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    திரும்பு dma_ring_தள்ளு_single_1d(r, ddata, size);
}
#இல்லையெனில்
நிலையான உள்வரி bool dma_ring_தள்ளு_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    திரும்பு dma_ring_தள்ளு_single_2d(r, ddata, size, size, size, 1);
}
#endif

நிலையான உள்வரி dma_data dma_ring_எடு(dma_ring * r) {
    dma_data ddata = { 0 };

    எனில் (r->தள்ளு_idx == r->எடு_idx) {
        திரும்பு ddata;
    }

    ddata = r->data[r->எடு_idx];

    நிலையற்ற dma_descriptor_2d * desc = &r->desc[r->எடு_idx];

    // Wait for desc to complete
    எனில் (!desc->done) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        போது (!desc->done) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->எடு_idx);

    r->எடு_idx = (r->எடு_idx + 1) & r->idx_mask;
    திரும்பு ddata;
}

நிலையான உள்வரி dma_data dma_ring_எடு_nowait(dma_ring * r) {
    dma_data ddata = { 0 };

    எனில் (r->தள்ளு_idx == r->எடு_idx) {
        திரும்பு ddata;
    }

    ddata = r->data[r->எடு_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->எடு_idx);

    r->எடு_idx = (r->எடு_idx + 1) & r->idx_mask;
    திரும்பு ddata;
}

நிலையான உள்வரி bool dma_ring_empty(dma_ring * r) {
    திரும்பு r->தள்ளு_idx == r->எடு_idx;
}

நிலையான உள்வரி வெறுமை dma_ring_flush(dma_ring * r) {
    போது (!dma_ring_empty(r)) {
        dma_ring_எடு(r);
    }
}

நிலையான உள்வரி uint32_t dma_ring_depth(dma_ring * r) {
    திரும்பு (r->தள்ளு_idx - r->எடு_idx) & r->idx_mask;
}

நிலையான உள்வரி uint32_t dma_ring_capacity(dma_ring * r) {
    திரும்பு r->capacity;
}

நிலையான உள்வரி bool dma_queue_தள்ளு_single_1d(dma_queue * q, dma_data ddata, size_t size) {
    திரும்பு dma_ring_தள்ளு_single_1d(q->ring0, ddata, size);
}

நிலையான உள்வரி bool dma_queue_தள்ளு_single_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    திரும்பு dma_ring_தள்ளு_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
}

நிலையான உள்வரி dma_data dma_queue_எடு(dma_queue * q) {
    திரும்பு dma_ring_எடு(q->ring0);
}

நிலையான உள்வரி dma_data dma_queue_எடு_nowait(dma_queue * q) {
    திரும்பு dma_ring_எடு_nowait(q->ring0);
}

நிலையான உள்வரி bool dma_queue_empty(dma_queue * q) {
    திரும்பு dma_ring_empty(q->ring0);
}

நிலையான உள்வரி வெறுமை dma_queue_flush(dma_queue * q) {
    dma_ring_flush(q->ring0);
}

நிலையான உள்வரி uint32_t dma_queue_depth(dma_queue * q) {
    திரும்பு dma_ring_depth(q->ring0);
}

நிலையான உள்வரி uint32_t dma_queue_capacity(dma_queue * q) {
    திரும்பு dma_ring_capacity(q->ring0);
}

நிலையான உள்வரி bool dma_queue_தள்ளு(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    எனில் (__builtin_expect(nrows == 0, 0)) {
        திரும்பு dma_ring_தள்ளு_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    எனில் (nrows == 1 || !((row_size ^ src_stride) | (row_size ^ dst_stride))) {
        மாறிலி size_t total = row_size * nrows;
        எனில் (__builtin_expect(total <= DMA_MAX_SIZE_24B, 1)) {
            திரும்பு dma_ring_தள்ளு_single_contig(q->ring0, ddata, total);
        }
        திரும்பு dma_queue_தள்ளு_fallback_contig(q, ddata, total);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    எனில் (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (row_size | src_stride | dst_stride) <= DMA_MAX_2D_ROW_SIZE, 1)) {
        திரும்பு dma_ring_தள்ளு_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#எனில் __HVX_ARCH__ < 75
    எனில் (row_size <= DMA_MAX_SIZE_16B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        திரும்பு dma_queue_தள்ளு_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
    }
    திரும்பு dma_queue_தள்ளு_fallback_1d(q, ddata, dst_stride, src_stride, row_size, nrows);
#இல்லையெனில்
    திரும்பு dma_queue_தள்ளு_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
#endif
}

நிலையான உள்வரி வெறுமை dma_sync_read(dma_queue * dma_q, வெறுமை * dst, dma_addr_t src, size_t bytes) {
    மாறிலி uint32_t b = (uint32_t) bytes;
    எனில் (b > 0) {
        dma_queue_தள்ளு(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_எடு(dma_q);
    }
}

நிலையான உள்வரி வெறுமை dma_sync_write(dma_queue * dma_q, dma_addr_t dst, மாறிலி வெறுமை * src, size_t bytes) {
    மாறிலி uint32_t b = (uint32_t) bytes;
    எனில் (b > 0) {
        dma_queue_தள்ளு(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_எடு(dma_q);
    }
}

#define DMA_CACHE_MAX_SIZE 256U

// Fully assoc LRU cache
வகைவரை கட்டமைப்பு {
    uint8_t *base;
    uint32_t வரி_size;
    uint32_t capacity;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
    uint16_t age[DMA_CACHE_MAX_SIZE];
} dma_cache_fa;

நிலையான உள்வரி வெறுமை dma_cache_fa_init(dma_cache_fa *c, uint8_t *base, uint32_t வரி_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->வரி_size = வரி_size;

    க்கு (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

நிலையான உள்வரி bool dma_cache_fa_தள்ளு(dma_queue *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    க்கு (unsigned i=0; i < c->capacity; i++) {
        எனில் (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->base + (i * c->வரி_size); nrows = 0; // dummy dma
        } இல்லையெனில் {
            c->age[i]++;
            எனில் (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    எனில் (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->base + o_idx * c->வரி_size; // normal nrows dma
        திரும்பு dma_queue_தள்ளு(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
    }

    திரும்பு dma_queue_தள்ளு_single_1d(q, dma_make_data(dst, src_addr), 0);
}

// Direct mapped cache
வகைவரை கட்டமைப்பு {
    uint8_t *base;
    uint32_t வரி_size;
    uint32_t capacity;
    uint32_t idx_mask;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
} dma_cache_dm;

நிலையான உள்வரி வெறுமை dma_cache_dm_init(dma_cache_dm *c, uint8_t *base, uint32_t வரி_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->idx_mask  = c->capacity - 1;
    c->base      = base;
    c->வரி_size = வரி_size;

    க்கு (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
    }
}

நிலையான உள்வரி bool dma_cache_dm_தள்ளு(dma_queue *q, dma_cache_dm *c, uint32_t slot, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    மாறிலி uint32_t i = slot & c->idx_mask;
    uint8_t * dst = c->base + (i * c->வரி_size);

    எனில் (c->src[i] == src_addr) {
        திரும்பு dma_queue_தள்ளு_single_1d(q, dma_make_data(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    திரும்பு dma_queue_தள்ளு(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
