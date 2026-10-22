#ifndef HTP_DMA_H
#define HTP_DMA_H

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <hexagon_types.h>
#अन्तर्भाव <assert.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव "hex-utils.h"

#अन्तर्भाव "hex-profile.h"

#ifdef __cplusplus
बाह्य "C" {
#endif

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
प्रकारनिर्देश संरचना dma_descriptor_1d_s {
    शून्य *   next;
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

#जइ __HVX_ARCH__ < 75

प्रकारनिर्देश संरचना dma_descriptor_2d_s {
    शून्य *   next;
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

#अण्णहा

प्रकारनिर्देश संरचना dma_descriptor_2d_s {
    शून्य *   next;
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
#जइ __HVX_ARCH__ > 79
    uint32_t src_upper:8;
    uint32_t dst_upper:8;
    uint32_t allocation:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#अण्णहा
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

#जइ __HVX_ARCH__ > 79
प्रकारनिर्देश uint64_t dma_addr_t;
#अण्णहा
प्रकारनिर्देश uint32_t dma_addr_t;
#endif

प्रकारनिर्देश संरचना {
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

#जइ __HVX_ARCH__ < 75
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_16B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#अण्णहा
#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_24B
#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#endif

#define DMA_FALLBACK_CAPACITY  16u            // descriptors in secondary fallback ring

प्रकारनिर्देश संरचना dma_ring_s dma_ring;
संरचना dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * tail;      // tail pointer
    dma_data          * data;      // dst/src data
    uint32_t            प्रक्षिप_idx;
    uint32_t            निष्कासय_idx;
    uint32_t            capacity;
    uint32_t            idx_mask;
    संरचना htp_thread_trace * trace;
};

प्रकारनिर्देश संरचना dma_queue_s dma_queue;
प्रकारनिर्देश dma_queue * dma_queue_t;

संरचना dma_queue_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_queue_sizeof(size_t capacity);
size_t      dma_queue_alignof(शून्य);
dma_queue_t dma_queue_init(शून्य * ptr, size_t capacity, संरचना htp_thread_trace * trace);
शून्य        dma_queue_free(dma_queue_t q);

size_t      dma_queue_alias_sizeof(शून्य);
dma_queue_t dma_queue_alias_init(शून्य * ptr, dma_queue_t main_q);
शून्य        dma_queue_alias_free(dma_queue_t q);

bool        dma_queue_प्रक्षिप_fallback_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
bool        dma_queue_प्रक्षिप_fallback_contig(dma_queue * q, dma_data ddata, size_t total);
#जइ __HVX_ARCH__ < 75
bool        dma_queue_प्रक्षिप_fallback_1d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
#endif

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
स्थिर अन्तःस्थ शून्य dmstart(शून्य * next) {
    asm चञ्चल(" release(%0):at" : : "r"(next));
    asm चञ्चल(" dmstart(%0)" : : "r"(next));
}

स्थिर अन्तःस्थ शून्य dmlink(शून्य * cur, शून्य * next) {
    asm चञ्चल(" release(%0):at" : : "r"(next));
    asm चञ्चल(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

स्थिर अन्तःस्थ unsigned पूर्णाङ्क dmpoll(शून्य) {
    unsigned पूर्णाङ्क ret = 0;
    asm चञ्चल(" %0 = dmpoll" : "=r"(ret) : : "memory");
    प्रत्यागम ret;
}

स्थिर अन्तःस्थ unsigned पूर्णाङ्क dmwait(शून्य) {
    unsigned पूर्णाङ्क ret = 0;
    asm चञ्चल(" %0 = dmwait" : "=r"(ret) : : "memory");
    प्रत्यागम ret;
}

स्थिर अन्तःस्थ dma_data dma_make_data_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_data d = { dst, src };
    प्रत्यागम d;
}

#define dma_make_data(dst, src) dma_make_data_impl((dma_addr_t) (dst), (dma_addr_t) (src))

स्थिर अन्तःस्थ bool dma_ring_प्रक्षिप_single_1d(dma_ring * r, dma_data ddata, size_t size) {
#जइ __HVX_ARCH__ > 79
    प्रतिज्ञा(!((ddata.src | ddata.dst) >> 32) || size == 0);
#endif

    जइ (((r->प्रक्षिप_idx + 1) & r->idx_mask) == r->निष्कासय_idx) {
        प्रत्यागम false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->प्रक्षिप_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->size = size;

    r->data[r->प्रक्षिप_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->प्रक्षिप_idx);

    जइ (size) {
        desc->next       = रिक्त;
        desc->desc_size  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->order      = 0;
        desc->done       = 0;

        dmlink(r->tail, desc);
        r->tail = (dma_descriptor_2d *) desc;
    } अण्णहा {
        desc->desc_size = 0;
        desc->done      = 1;
    }

    r->प्रक्षिप_idx = (r->प्रक्षिप_idx + 1) & r->idx_mask;
    प्रत्यागम true;
}

स्थिर अन्तःस्थ bool dma_ring_प्रक्षिप_single_2d(dma_ring * r, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
#जइ __HVX_ARCH__ > 79
    प्रतिज्ञा(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ध्रुव uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ध्रुव uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ध्रुव bool is_ext     = (src_hi | dst_hi) != 0;
#endif

    जइ (((r->प्रक्षिप_idx + 1) & r->idx_mask) == r->निष्कासय_idx) {
        प्रत्यागम false;
    }

    dma_descriptor_2d * desc = &r->desc[r->प्रक्षिप_idx];

    desc->next           = रिक्त;
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

#जइ __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_offset     = 0;
    desc->dst_offset     = 0;
#अण्णहा
#जइ __HVX_ARCH__ > 79
    desc->src_upper      = src_hi;
    desc->dst_upper      = dst_hi;
    desc->allocation     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#अण्णहा
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#endif
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->offset         = 0;
#endif

    r->data[r->प्रक्षिप_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->प्रक्षिप_idx);

    जइ (nrows) {
        dmlink(r->tail, desc);
        r->tail = desc;
    } अण्णहा {
        desc->done = 1;
    }

    r->प्रक्षिप_idx = (r->प्रक्षिप_idx + 1) & r->idx_mask;
    प्रत्यागम true;
}

#जइ __HVX_ARCH__ < 75
स्थिर अन्तःस्थ bool dma_ring_प्रक्षिप_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    प्रत्यागम dma_ring_प्रक्षिप_single_1d(r, ddata, size);
}
#अण्णहा
स्थिर अन्तःस्थ bool dma_ring_प्रक्षिप_single_contig(dma_ring * r, dma_data ddata, size_t size) {
    प्रत्यागम dma_ring_प्रक्षिप_single_2d(r, ddata, size, size, size, 1);
}
#endif

स्थिर अन्तःस्थ dma_data dma_ring_निष्कासय(dma_ring * r) {
    dma_data ddata = { 0 };

    जइ (r->प्रक्षिप_idx == r->निष्कासय_idx) {
        प्रत्यागम ddata;
    }

    ddata = r->data[r->निष्कासय_idx];

    चञ्चल dma_descriptor_2d * desc = &r->desc[r->निष्कासय_idx];

    // Wait for desc to complete
    जइ (!desc->done) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        जाव (!desc->done) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->निष्कासय_idx);

    r->निष्कासय_idx = (r->निष्कासय_idx + 1) & r->idx_mask;
    प्रत्यागम ddata;
}

स्थिर अन्तःस्थ dma_data dma_ring_निष्कासय_nowait(dma_ring * r) {
    dma_data ddata = { 0 };

    जइ (r->प्रक्षिप_idx == r->निष्कासय_idx) {
        प्रत्यागम ddata;
    }

    ddata = r->data[r->निष्कासय_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->निष्कासय_idx);

    r->निष्कासय_idx = (r->निष्कासय_idx + 1) & r->idx_mask;
    प्रत्यागम ddata;
}

स्थिर अन्तःस्थ bool dma_ring_empty(dma_ring * r) {
    प्रत्यागम r->प्रक्षिप_idx == r->निष्कासय_idx;
}

स्थिर अन्तःस्थ शून्य dma_ring_flush(dma_ring * r) {
    जाव (!dma_ring_empty(r)) {
        dma_ring_निष्कासय(r);
    }
}

स्थिर अन्तःस्थ uint32_t dma_ring_depth(dma_ring * r) {
    प्रत्यागम (r->प्रक्षिप_idx - r->निष्कासय_idx) & r->idx_mask;
}

स्थिर अन्तःस्थ uint32_t dma_ring_capacity(dma_ring * r) {
    प्रत्यागम r->capacity;
}

स्थिर अन्तःस्थ bool dma_queue_प्रक्षिप_single_1d(dma_queue * q, dma_data ddata, size_t size) {
    प्रत्यागम dma_ring_प्रक्षिप_single_1d(q->ring0, ddata, size);
}

स्थिर अन्तःस्थ bool dma_queue_प्रक्षिप_single_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    प्रत्यागम dma_ring_प्रक्षिप_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
}

स्थिर अन्तःस्थ dma_data dma_queue_निष्कासय(dma_queue * q) {
    प्रत्यागम dma_ring_निष्कासय(q->ring0);
}

स्थिर अन्तःस्थ dma_data dma_queue_निष्कासय_nowait(dma_queue * q) {
    प्रत्यागम dma_ring_निष्कासय_nowait(q->ring0);
}

स्थिर अन्तःस्थ bool dma_queue_empty(dma_queue * q) {
    प्रत्यागम dma_ring_empty(q->ring0);
}

स्थिर अन्तःस्थ शून्य dma_queue_flush(dma_queue * q) {
    dma_ring_flush(q->ring0);
}

स्थिर अन्तःस्थ uint32_t dma_queue_depth(dma_queue * q) {
    प्रत्यागम dma_ring_depth(q->ring0);
}

स्थिर अन्तःस्थ uint32_t dma_queue_capacity(dma_queue * q) {
    प्रत्यागम dma_ring_capacity(q->ring0);
}

स्थिर अन्तःस्थ bool dma_queue_प्रक्षिप(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    जइ (__builtin_expect(nrows == 0, 0)) {
        प्रत्यागम dma_ring_प्रक्षिप_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    जइ (nrows == 1 || !((row_size ^ src_stride) | (row_size ^ dst_stride))) {
        ध्रुव size_t total = row_size * nrows;
        जइ (__builtin_expect(total <= DMA_MAX_SIZE_24B, 1)) {
            प्रत्यागम dma_ring_प्रक्षिप_single_contig(q->ring0, ddata, total);
        }
        प्रत्यागम dma_queue_प्रक्षिप_fallback_contig(q, ddata, total);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    जइ (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (row_size | src_stride | dst_stride) <= DMA_MAX_2D_ROW_SIZE, 1)) {
        प्रत्यागम dma_ring_प्रक्षिप_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#जइ __HVX_ARCH__ < 75
    जइ (row_size <= DMA_MAX_SIZE_16B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        प्रत्यागम dma_queue_प्रक्षिप_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
    }
    प्रत्यागम dma_queue_प्रक्षिप_fallback_1d(q, ddata, dst_stride, src_stride, row_size, nrows);
#अण्णहा
    प्रत्यागम dma_queue_प्रक्षिप_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
#endif
}

स्थिर अन्तःस्थ शून्य dma_sync_read(dma_queue * dma_q, शून्य * dst, dma_addr_t src, size_t bytes) {
    ध्रुव uint32_t b = (uint32_t) bytes;
    जइ (b > 0) {
        dma_queue_प्रक्षिप(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_निष्कासय(dma_q);
    }
}

स्थिर अन्तःस्थ शून्य dma_sync_write(dma_queue * dma_q, dma_addr_t dst, ध्रुव शून्य * src, size_t bytes) {
    ध्रुव uint32_t b = (uint32_t) bytes;
    जइ (b > 0) {
        dma_queue_प्रक्षिप(dma_q, dma_make_data(dst, src), b, b, b, 1);
        dma_queue_निष्कासय(dma_q);
    }
}

#define DMA_CACHE_MAX_SIZE 256U

// Fully assoc LRU cache
प्रकारनिर्देश संरचना {
    uint8_t *base;
    uint32_t पङ्क्तिः_size;
    uint32_t capacity;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
    uint16_t age[DMA_CACHE_MAX_SIZE];
} dma_cache_fa;

स्थिर अन्तःस्थ शून्य dma_cache_fa_init(dma_cache_fa *c, uint8_t *base, uint32_t पङ्क्तिः_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->पङ्क्तिः_size = पङ्क्तिः_size;

    प्रति (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

स्थिर अन्तःस्थ bool dma_cache_fa_प्रक्षिप(dma_queue *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    प्रति (unsigned i=0; i < c->capacity; i++) {
        जइ (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->base + (i * c->पङ्क्तिः_size); nrows = 0; // dummy dma
        } अण्णहा {
            c->age[i]++;
            जइ (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    जइ (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->base + o_idx * c->पङ्क्तिः_size; // normal nrows dma
        प्रत्यागम dma_queue_प्रक्षिप(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
    }

    प्रत्यागम dma_queue_प्रक्षिप_single_1d(q, dma_make_data(dst, src_addr), 0);
}

// Direct mapped cache
प्रकारनिर्देश संरचना {
    uint8_t *base;
    uint32_t पङ्क्तिः_size;
    uint32_t capacity;
    uint32_t idx_mask;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
} dma_cache_dm;

स्थिर अन्तःस्थ शून्य dma_cache_dm_init(dma_cache_dm *c, uint8_t *base, uint32_t पङ्क्तिः_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->idx_mask  = c->capacity - 1;
    c->base      = base;
    c->पङ्क्तिः_size = पङ्क्तिः_size;

    प्रति (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
    }
}

स्थिर अन्तःस्थ bool dma_cache_dm_प्रक्षिप(dma_queue *q, dma_cache_dm *c, uint32_t slot, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    ध्रुव uint32_t i = slot & c->idx_mask;
    uint8_t * dst = c->base + (i * c->पङ्क्तिः_size);

    जइ (c->src[i] == src_addr) {
        प्रत्यागम dma_queue_प्रक्षिप_single_1d(q, dma_make_data(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    प्रत्यागम dma_queue_प्रक्षिप(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
