#यदि_अपरिभाषितम् HTP_DMA_H
#परिभाषय HTP_DMA_H

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <hexagon_types.h>
#अन्तर्भाव <assert.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव "hex-utils.h"

#अन्तर्भाव "hex-profile.h"

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
प्रकारनिर्देश संरचना dma_descriptor_1d_s {
    शून्य *   अग्र;
    uint32_t आकार:24;
    uint32_t desc_आकार:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t क्रमः:1;
    uint32_t समाप्तम्:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#यदि __HVX_ARCH__ < 75

प्रकारनिर्देश संरचना dma_descriptor_2d_s {
    शून्य *   अग्र;
    uint32_t reserved0:24;
    uint32_t desc_आकार:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t क्रमः:1;
    uint32_t समाप्तम्:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t row_आकार:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_विस्थापनम्:16;
    uint32_t dst_विस्थापनम्:16;
} dma_descriptor_2d;

#अन्यथा

प्रकारनिर्देश संरचना dma_descriptor_2d_s {
    शून्य *   अग्र;
    uint32_t dst_stride:24;
    uint32_t desc_आकार:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t क्रमः:1;
    uint32_t समाप्तम्:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#यदि __HVX_ARCH__ > 79
    uint32_t src_ऊर्ध्वस्थम्:8;
    uint32_t dst_ऊर्ध्वस्थम्:8;
    uint32_t आवण्टनम्:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#अन्यथा
    uint32_t reserved0:24;
#यदि_अन्तः
    uint32_t row_आकार:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t विस्थापनम्:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#यदि_अन्तः

#यदि __HVX_ARCH__ > 79
प्रकारनिर्देश uint64_t dma_addr_t;
#अन्यथा
प्रकारनिर्देश uint32_t dma_addr_t;
#यदि_अन्तः

प्रकारनिर्देश संरचना {
    dma_addr_t dst;
    dma_addr_t src;
} dma_दत्त;

// Hardware descriptor field limits
#परिभाषय DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#परिभाषय DMA_MAX_आकार_16_B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#परिभाषय DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#परिभाषय DMA_MAX_आकार_24_B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#परिभाषय DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#परिभाषय DMA_SAFE_CHUNK_आकार    0x00F00000u    // ~15MB safe contiguous chunk size

#यदि __HVX_ARCH__ < 75
#परिभाषय DMA_MAX_2_D_ROW_आकार    DMA_MAX_आकार_16_B
#परिभाषय DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#अन्यथा
#परिभाषय DMA_MAX_2_D_ROW_आकार    DMA_MAX_आकार_24_B
#परिभाषय DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#यदि_अन्तः

#परिभाषय DMA_FALLBACK_क्षमता  16u            // descriptors in secondary fallback ring

प्रकारनिर्देश संरचना dma_ring_s dma_ring;
संरचना dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * पुच्छ;      // tail pointer
    dma_दत्त          * दत्त;      // dst/src data
    uint32_t            प्रक्षिप_idx;
    uint32_t            निष्कासय_idx;
    uint32_t            क्षमता;
    uint32_t            idx_आवरणम्;
    संरचना htp_thread_trace * trace;
};

प्रकारनिर्देश संरचना dma_queue_s dma_queue;
प्रकारनिर्देश dma_queue * dma_queue_t;

संरचना dma_queue_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_queue_sizeof(size_t क्षमता);
size_t      dma_queue_alignof(शून्य);
dma_queue_t dma_queue_init(शून्य * ptr, size_t क्षमता, संरचना htp_thread_trace * trace);
शून्य        dma_queue_free(dma_queue_t q);

size_t      dma_queue_alias_sizeof(शून्य);
dma_queue_t dma_queue_alias_init(शून्य * ptr, dma_queue_t main_q);
शून्य        dma_queue_alias_free(dma_queue_t q);

bool        dma_queue_प्रक्षिप_fallback_2d(dma_queue * q, dma_दत्त ddata, size_t dst_stride, size_t src_stride, size_t row_आकार, size_t nrows);
bool        dma_queue_प्रक्षिप_fallback_contig(dma_queue * q, dma_दत्त ddata, size_t योग);
#यदि __HVX_ARCH__ < 75
bool        dma_queue_प्रक्षिप_fallback_1d(dma_queue * q, dma_दत्त ddata, size_t dst_stride, size_t src_stride, size_t row_आकार, size_t nrows);
#यदि_अन्तः

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
स्थिर अन्तःस्थ शून्य dmstart(शून्य * अग्र) {
    asm चञ्चल(" release(%0):at" : : "r"(अग्र));
    asm चञ्चल(" dmstart(%0)" : : "r"(अग्र));
}

स्थिर अन्तःस्थ शून्य dmlink(शून्य * cur, शून्य * अग्र) {
    asm चञ्चल(" release(%0):at" : : "r"(अग्र));
    asm चञ्चल(" dmlink(%0, %1)" : : "r"(cur), "r"(अग्र));
}

स्थिर अन्तःस्थ अचिह्नित पूर्णाङ्क dmpoll(शून्य) {
    अचिह्नित पूर्णाङ्क ret = 0;
    asm चञ्चल(" %0 = dmpoll" : "=r"(ret) : : "memory");
    प्रत्यागम ret;
}

स्थिर अन्तःस्थ अचिह्नित पूर्णाङ्क dmwait(शून्य) {
    अचिह्नित पूर्णाङ्क ret = 0;
    asm चञ्चल(" %0 = dmwait" : "=r"(ret) : : "memory");
    प्रत्यागम ret;
}

स्थिर अन्तःस्थ dma_दत्त dma_make_दत्त_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_दत्त d = { dst, src };
    प्रत्यागम d;
}

#परिभाषय dma_make_दत्त(dst, src) dma_make_दत्त_impl((dma_addr_t) (dst), (dma_addr_t) (src))

स्थिर अन्तःस्थ bool dma_ring_प्रक्षिप_single_1d(dma_ring * r, dma_दत्त ddata, size_t आकार) {
#यदि __HVX_ARCH__ > 79
    प्रतिज्ञा(!((ddata.src | ddata.dst) >> 32) || आकार == 0);
#यदि_अन्तः

    यदि (((r->प्रक्षिप_idx + 1) & r->idx_आवरणम्) == r->निष्कासय_idx) {
        प्रत्यागम false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->प्रक्षिप_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->आकार = आकार;

    r->दत्त[r->प्रक्षिप_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->प्रक्षिप_idx);

    यदि (आकार) {
        desc->अग्र       = रिक्त;
        desc->desc_आकार  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->क्रमः      = 0;
        desc->समाप्तम्       = 0;

        dmlink(r->पुच्छ, desc);
        r->पुच्छ = (dma_descriptor_2d *) desc;
    } अन्यथा {
        desc->desc_आकार = 0;
        desc->समाप्तम्      = 1;
    }

    r->प्रक्षिप_idx = (r->प्रक्षिप_idx + 1) & r->idx_आवरणम्;
    प्रत्यागम true;
}

स्थिर अन्तःस्थ bool dma_ring_प्रक्षिप_single_2d(dma_ring * r, dma_दत्त ddata, size_t dst_stride, size_t src_stride, size_t row_आकार, size_t nrows) {
#यदि __HVX_ARCH__ > 79
    प्रतिज्ञा(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ध्रुव uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ध्रुव uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ध्रुव bool is_ext     = (src_hi | dst_hi) != 0;
#यदि_अन्तः

    यदि (((r->प्रक्षिप_idx + 1) & r->idx_आवरणम्) == r->निष्कासय_idx) {
        प्रत्यागम false;
    }

    dma_descriptor_2d * desc = &r->desc[r->प्रक्षिप_idx];

    desc->अग्र           = रिक्त;
    desc->reserved1      = 0;
    desc->desc_आकार      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->क्रमः          = 0;
    desc->समाप्तम्           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->row_आकार       = row_आकार;

#यदि __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_विस्थापनम्     = 0;
    desc->dst_विस्थापनम्     = 0;
#अन्यथा
#यदि __HVX_ARCH__ > 79
    desc->src_ऊर्ध्वस्थम्      = src_hi;
    desc->dst_ऊर्ध्वस्थम्      = dst_hi;
    desc->आवण्टनम्     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#अन्यथा
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#यदि_अन्तः
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->विस्थापनम्         = 0;
#यदि_अन्तः

    r->दत्त[r->प्रक्षिप_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->प्रक्षिप_idx);

    यदि (nrows) {
        dmlink(r->पुच्छ, desc);
        r->पुच्छ = desc;
    } अन्यथा {
        desc->समाप्तम् = 1;
    }

    r->प्रक्षिप_idx = (r->प्रक्षिप_idx + 1) & r->idx_आवरणम्;
    प्रत्यागम true;
}

#यदि __HVX_ARCH__ < 75
स्थिर अन्तःस्थ bool dma_ring_प्रक्षिप_single_contig(dma_ring * r, dma_दत्त ddata, size_t आकार) {
    प्रत्यागम dma_ring_प्रक्षिप_single_1d(r, ddata, आकार);
}
#अन्यथा
स्थिर अन्तःस्थ bool dma_ring_प्रक्षिप_single_contig(dma_ring * r, dma_दत्त ddata, size_t आकार) {
    प्रत्यागम dma_ring_प्रक्षिप_single_2d(r, ddata, आकार, आकार, आकार, 1);
}
#यदि_अन्तः

स्थिर अन्तःस्थ dma_दत्त dma_ring_निष्कासय(dma_ring * r) {
    dma_दत्त ddata = { 0 };

    यदि (r->प्रक्षिप_idx == r->निष्कासय_idx) {
        प्रत्यागम ddata;
    }

    ddata = r->दत्त[r->निष्कासय_idx];

    चञ्चल dma_descriptor_2d * desc = &r->desc[r->निष्कासय_idx];

    // Wait for desc to complete
    यदि (!desc->समाप्तम्) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        यावत् (!desc->समाप्तम्) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->निष्कासय_idx);

    r->निष्कासय_idx = (r->निष्कासय_idx + 1) & r->idx_आवरणम्;
    प्रत्यागम ddata;
}

स्थिर अन्तःस्थ dma_दत्त dma_ring_निष्कासय_nowait(dma_ring * r) {
    dma_दत्त ddata = { 0 };

    यदि (r->प्रक्षिप_idx == r->निष्कासय_idx) {
        प्रत्यागम ddata;
    }

    ddata = r->दत्त[r->निष्कासय_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->निष्कासय_idx);

    r->निष्कासय_idx = (r->निष्कासय_idx + 1) & r->idx_आवरणम्;
    प्रत्यागम ddata;
}

स्थिर अन्तःस्थ bool dma_ring_रिक्तानि(dma_ring * r) {
    प्रत्यागम r->प्रक्षिप_idx == r->निष्कासय_idx;
}

स्थिर अन्तःस्थ शून्य dma_ring_flush(dma_ring * r) {
    यावत् (!dma_ring_रिक्तानि(r)) {
        dma_ring_निष्कासय(r);
    }
}

स्थिर अन्तःस्थ uint32_t dma_ring_गभीरता(dma_ring * r) {
    प्रत्यागम (r->प्रक्षिप_idx - r->निष्कासय_idx) & r->idx_आवरणम्;
}

स्थिर अन्तःस्थ uint32_t dma_ring_क्षमता(dma_ring * r) {
    प्रत्यागम r->क्षमता;
}

स्थिर अन्तःस्थ bool dma_queue_प्रक्षिप_single_1d(dma_queue * q, dma_दत्त ddata, size_t आकार) {
    प्रत्यागम dma_ring_प्रक्षिप_single_1d(q->ring0, ddata, आकार);
}

स्थिर अन्तःस्थ bool dma_queue_प्रक्षिप_single_2d(dma_queue * q, dma_दत्त ddata, size_t dst_stride, size_t src_stride, size_t row_आकार, size_t nrows) {
    प्रत्यागम dma_ring_प्रक्षिप_single_2d(q->ring0, ddata, dst_stride, src_stride, row_आकार, nrows);
}

स्थिर अन्तःस्थ dma_दत्त dma_queue_निष्कासय(dma_queue * q) {
    प्रत्यागम dma_ring_निष्कासय(q->ring0);
}

स्थिर अन्तःस्थ dma_दत्त dma_queue_निष्कासय_nowait(dma_queue * q) {
    प्रत्यागम dma_ring_निष्कासय_nowait(q->ring0);
}

स्थिर अन्तःस्थ bool dma_queue_रिक्तानि(dma_queue * q) {
    प्रत्यागम dma_ring_रिक्तानि(q->ring0);
}

स्थिर अन्तःस्थ शून्य dma_queue_flush(dma_queue * q) {
    dma_ring_flush(q->ring0);
}

स्थिर अन्तःस्थ uint32_t dma_queue_गभीरता(dma_queue * q) {
    प्रत्यागम dma_ring_गभीरता(q->ring0);
}

स्थिर अन्तःस्थ uint32_t dma_queue_क्षमता(dma_queue * q) {
    प्रत्यागम dma_ring_क्षमता(q->ring0);
}

स्थिर अन्तःस्थ bool dma_queue_प्रक्षिप(dma_queue * q, dma_दत्त ddata, size_t dst_stride, size_t src_stride, size_t row_आकार, size_t nrows) {
    यदि (__builtin_expect(nrows == 0, 0)) {
        प्रत्यागम dma_ring_प्रक्षिप_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    यदि (nrows == 1 || !((row_आकार ^ src_stride) | (row_आकार ^ dst_stride))) {
        ध्रुव size_t योग = row_आकार * nrows;
        यदि (__builtin_expect(योग <= DMA_MAX_आकार_24_B, 1)) {
            प्रत्यागम dma_ring_प्रक्षिप_single_contig(q->ring0, ddata, योग);
        }
        प्रत्यागम dma_queue_प्रक्षिप_fallback_contig(q, ddata, योग);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    यदि (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (row_आकार | src_stride | dst_stride) <= DMA_MAX_2_D_ROW_आकार, 1)) {
        प्रत्यागम dma_ring_प्रक्षिप_single_2d(q->ring0, ddata, dst_stride, src_stride, row_आकार, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#यदि __HVX_ARCH__ < 75
    यदि (row_आकार <= DMA_MAX_आकार_16_B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        प्रत्यागम dma_queue_प्रक्षिप_fallback_2d(q, ddata, dst_stride, src_stride, row_आकार, nrows);
    }
    प्रत्यागम dma_queue_प्रक्षिप_fallback_1d(q, ddata, dst_stride, src_stride, row_आकार, nrows);
#अन्यथा
    प्रत्यागम dma_queue_प्रक्षिप_fallback_2d(q, ddata, dst_stride, src_stride, row_आकार, nrows);
#यदि_अन्तः
}

स्थिर अन्तःस्थ शून्य dma_sync_read(dma_queue * dma_q, शून्य * dst, dma_addr_t src, size_t अष्टकानि) {
    ध्रुव uint32_t b = (uint32_t) अष्टकानि;
    यदि (b > 0) {
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst, src), b, b, b, 1);
        dma_queue_निष्कासय(dma_q);
    }
}

स्थिर अन्तःस्थ शून्य dma_sync_write(dma_queue * dma_q, dma_addr_t dst, ध्रुव शून्य * src, size_t अष्टकानि) {
    ध्रुव uint32_t b = (uint32_t) अष्टकानि;
    यदि (b > 0) {
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst, src), b, b, b, 1);
        dma_queue_निष्कासय(dma_q);
    }
}

#परिभाषय DMA_CACHE_MAX_आकार 256U

// Fully assoc LRU cache
प्रकारनिर्देश संरचना {
    uint8_t *आधारः;
    uint32_t पङ्क्तिः_आकार;
    uint32_t क्षमता;
    dma_addr_t src[DMA_CACHE_MAX_आकार];
    uint16_t age[DMA_CACHE_MAX_आकार];
} dma_cache_fa;

स्थिर अन्तःस्थ शून्य dma_cache_fa_init(dma_cache_fa *c, uint8_t *आधारः, uint32_t पङ्क्तिः_आकार, uint32_t क्षमता)
{
    c->क्षमता  = (क्षमता > DMA_CACHE_MAX_आकार) ? DMA_CACHE_MAX_आकार : क्षमता;
    c->आधारः      = आधारः;
    c->पङ्क्तिः_आकार = पङ्क्तिः_आकार;

    प्रति (अचिह्नित i=0; i < c->क्षमता; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

स्थिर अन्तःस्थ bool dma_cache_fa_प्रक्षिप(dma_queue *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_आकार, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    प्रति (अचिह्नित i=0; i < c->क्षमता; i++) {
        यदि (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->आधारः + (i * c->पङ्क्तिः_आकार); nrows = 0; // dummy dma
        } अन्यथा {
            c->age[i]++;
            यदि (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    यदि (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->आधारः + o_idx * c->पङ्क्तिः_आकार; // normal nrows dma
        प्रत्यागम dma_queue_प्रक्षिप(q, dma_make_दत्त(dst, src_addr), dst_stride, src_stride, row_आकार, nrows);
    }

    प्रत्यागम dma_queue_प्रक्षिप_single_1d(q, dma_make_दत्त(dst, src_addr), 0);
}

// Direct mapped cache
प्रकारनिर्देश संरचना {
    uint8_t *आधारः;
    uint32_t पङ्क्तिः_आकार;
    uint32_t क्षमता;
    uint32_t idx_आवरणम्;
    dma_addr_t src[DMA_CACHE_MAX_आकार];
} dma_cache_dm;

स्थिर अन्तःस्थ शून्य dma_cache_dm_init(dma_cache_dm *c, uint8_t *आधारः, uint32_t पङ्क्तिः_आकार, uint32_t क्षमता)
{
    c->क्षमता  = (क्षमता > DMA_CACHE_MAX_आकार) ? DMA_CACHE_MAX_आकार : क्षमता;
    c->idx_आवरणम्  = c->क्षमता - 1;
    c->आधारः      = आधारः;
    c->पङ्क्तिः_आकार = पङ्क्तिः_आकार;

    प्रति (अचिह्नित i=0; i < c->क्षमता; i++) {
        c->src[i] = 0;
    }
}

स्थिर अन्तःस्थ bool dma_cache_dm_प्रक्षिप(dma_queue *q, dma_cache_dm *c, uint32_t slot, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_आकार, uint32_t nrows)
{
    ध्रुव uint32_t i = slot & c->idx_आवरणम्;
    uint8_t * dst = c->आधारः + (i * c->पङ्क्तिः_आकार);

    यदि (c->src[i] == src_addr) {
        प्रत्यागम dma_queue_प्रक्षिप_single_1d(q, dma_make_दत्त(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    प्रत्यागम dma_queue_प्रक्षिप(q, dma_make_दत्त(dst, src_addr), dst_stride, src_stride, row_आकार, nrows);
}

#यदि_परिभाषितम् __cplusplus
}  // extern "C"
#यदि_अन्तः

#यदि_अन्तः /* HTP_DMA_H */
