#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_DMA_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_DMA_H

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <hexagon_types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ dma_descriptor_1d_s {
    ਖਾਲੀ *   ਅਗਲਾ;
    uint32_t ਮਾਪ:24;
    uint32_t desc_ਮਾਪ:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ਕ੍ਰਮ:1;
    uint32_t ਪੂਰਾ:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#ਜੇ __HVX_ARCH__ < 75

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ dma_descriptor_2d_s {
    ਖਾਲੀ *   ਅਗਲਾ;
    uint32_t reserved0:24;
    uint32_t desc_ਮਾਪ:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ਕ੍ਰਮ:1;
    uint32_t ਪੂਰਾ:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t ਪੰਕਤੀ_ਮਾਪ:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_ਵਿੱਥ:16;
    uint32_t dst_ਵਿੱਥ:16;
} dma_descriptor_2d;

#ਨਹੀਂਤਾਂ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ dma_descriptor_2d_s {
    ਖਾਲੀ *   ਅਗਲਾ;
    uint32_t dst_stride:24;
    uint32_t desc_ਮਾਪ:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ਕ੍ਰਮ:1;
    uint32_t ਪੂਰਾ:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#ਜੇ __HVX_ARCH__ > 79
    uint32_t src_ਉੱਤਲਾ:8;
    uint32_t dst_ਉੱਤਲਾ:8;
    uint32_t ਵੰਡ:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#ਨਹੀਂਤਾਂ
    uint32_t reserved0:24;
#ਜੇ_ਅੰਤ
    uint32_t ਪੰਕਤੀ_ਮਾਪ:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t ਵਿੱਥ:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#ਜੇ_ਅੰਤ

#ਜੇ __HVX_ARCH__ > 79
ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint64_t dma_addr_t;
#ਨਹੀਂਤਾਂ
ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint32_t dma_addr_t;
#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    dma_addr_t dst;
    dma_addr_t src;
} dma_ਅੰਕੜੇ;

// Hardware descriptor field limits
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_MAX_ਮਾਪ_16_B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_MAX_ਮਾਪ_24_B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_ਸੁਰੱਖਿਅਤ_CHUNK_ਮਾਪ    0x00F00000u    // ~15MB safe contiguous chunk size

#ਜੇ __HVX_ARCH__ < 75
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_MAX_2_D_ਪੰਕਤੀ_ਮਾਪ    DMA_MAX_ਮਾਪ_16_B
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_MAX_2_D_ਪੰਕਤੀ_ਮਾਪ    DMA_MAX_ਮਾਪ_24_B
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_FALLBACK_ਸਮਰੱਥਾ  16u            // descriptors in secondary fallback ring

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ dma_ring_s dma_ring;
ਬਣਤਰ dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * ਪੂਛ;      // tail pointer
    dma_ਅੰਕੜੇ          * ਅੰਕੜੇ;      // dst/src data
    uint32_t            ਧੱਕੋ_idx;
    uint32_t            ਕੱਢੋ_idx;
    uint32_t            ਸਮਰੱਥਾ;
    uint32_t            idx_ਨਕਾਬ;
    ਬਣਤਰ htp_thread_trace * trace;
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ dma_ਕਤਾਰ_s dma_ਕਤਾਰ;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ dma_ਕਤਾਰ * dma_ਕਤਾਰ_t;

ਬਣਤਰ dma_ਕਤਾਰ_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_ਕਤਾਰ_sizeof(size_t ਸਮਰੱਥਾ);
size_t      dma_ਕਤਾਰ_alignof(ਖਾਲੀ);
dma_ਕਤਾਰ_t dma_ਕਤਾਰ_init(ਖਾਲੀ * ptr, size_t ਸਮਰੱਥਾ, ਬਣਤਰ htp_thread_trace * trace);
ਖਾਲੀ        dma_ਕਤਾਰ_free(dma_ਕਤਾਰ_t q);

size_t      dma_ਕਤਾਰ_alias_sizeof(ਖਾਲੀ);
dma_ਕਤਾਰ_t dma_ਕਤਾਰ_alias_init(ਖਾਲੀ * ptr, dma_ਕਤਾਰ_t main_q);
ਖਾਲੀ        dma_ਕਤਾਰ_alias_free(dma_ਕਤਾਰ_t q);

bool        dma_ਕਤਾਰ_ਧੱਕੋ_fallback_2d(dma_ਕਤਾਰ * q, dma_ਅੰਕੜੇ ddata, size_t dst_stride, size_t src_stride, size_t ਪੰਕਤੀ_ਮਾਪ, size_t nrows);
bool        dma_ਕਤਾਰ_ਧੱਕੋ_fallback_contig(dma_ਕਤਾਰ * q, dma_ਅੰਕੜੇ ddata, size_t ਕੁੱਲ);
#ਜੇ __HVX_ARCH__ < 75
bool        dma_ਕਤਾਰ_ਧੱਕੋ_fallback_1d(dma_ਕਤਾਰ * q, dma_ਅੰਕੜੇ ddata, size_t dst_stride, size_t src_stride, size_t ਪੰਕਤੀ_ਮਾਪ, size_t nrows);
#ਜੇ_ਅੰਤ

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dmstart(ਖਾਲੀ * ਅਗਲਾ) {
    asm ਅਸਥਿਰ(" release(%0):at" : : "r"(ਅਗਲਾ));
    asm ਅਸਥਿਰ(" dmstart(%0)" : : "r"(ਅਗਲਾ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dmlink(ਖਾਲੀ * cur, ਖਾਲੀ * ਅਗਲਾ) {
    asm ਅਸਥਿਰ(" release(%0):at" : : "r"(ਅਗਲਾ));
    asm ਅਸਥਿਰ(" dmlink(%0, %1)" : : "r"(cur), "r"(ਅਗਲਾ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ dmpoll(ਖਾਲੀ) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ret = 0;
    asm ਅਸਥਿਰ(" %0 = dmpoll" : "=r"(ret) : : "memory");
    ਵਾਪਸ ret;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ dmwait(ਖਾਲੀ) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ret = 0;
    asm ਅਸਥਿਰ(" %0 = dmwait" : "=r"(ret) : : "memory");
    ਵਾਪਸ ret;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_ਅੰਕੜੇ dma_make_ਅੰਕੜੇ_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_ਅੰਕੜੇ d = { dst, src };
    ਵਾਪਸ d;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ dma_make_ਅੰਕੜੇ(dst, src) dma_make_ਅੰਕੜੇ_impl((dma_addr_t) (dst), (dma_addr_t) (src))

ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_ring_ਧੱਕੋ_single_1d(dma_ring * r, dma_ਅੰਕੜੇ ddata, size_t ਮਾਪ) {
#ਜੇ __HVX_ARCH__ > 79
    ਪ੍ਰਤਿਜ੍ਞਾ(!((ddata.src | ddata.dst) >> 32) || ਮਾਪ == 0);
#ਜੇ_ਅੰਤ

    ਜੇ (((r->ਧੱਕੋ_idx + 1) & r->idx_ਨਕਾਬ) == r->ਕੱਢੋ_idx) {
        ਵਾਪਸ false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->ਧੱਕੋ_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->ਮਾਪ = ਮਾਪ;

    r->ਅੰਕੜੇ[r->ਧੱਕੋ_idx] = ddata;

    htp_trace_event_ਸ਼ੁਰੂ(r->trace, HTP_TRACE_EVT_DMA, r->ਧੱਕੋ_idx);

    ਜੇ (ਮਾਪ) {
        desc->ਅਗਲਾ       = ਨਿਰਾ;
        desc->desc_ਮਾਪ  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->ਕ੍ਰਮ      = 0;
        desc->ਪੂਰਾ       = 0;

        dmlink(r->ਪੂਛ, desc);
        r->ਪੂਛ = (dma_descriptor_2d *) desc;
    } ਨਹੀਂਤਾਂ {
        desc->desc_ਮਾਪ = 0;
        desc->ਪੂਰਾ      = 1;
    }

    r->ਧੱਕੋ_idx = (r->ਧੱਕੋ_idx + 1) & r->idx_ਨਕਾਬ;
    ਵਾਪਸ true;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_ring_ਧੱਕੋ_single_2d(dma_ring * r, dma_ਅੰਕੜੇ ddata, size_t dst_stride, size_t src_stride, size_t ਪੰਕਤੀ_ਮਾਪ, size_t nrows) {
#ਜੇ __HVX_ARCH__ > 79
    ਪ੍ਰਤਿਜ੍ਞਾ(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ਅਚਲ uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ਅਚਲ uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ਅਚਲ bool is_ext     = (src_hi | dst_hi) != 0;
#ਜੇ_ਅੰਤ

    ਜੇ (((r->ਧੱਕੋ_idx + 1) & r->idx_ਨਕਾਬ) == r->ਕੱਢੋ_idx) {
        ਵਾਪਸ false;
    }

    dma_descriptor_2d * desc = &r->desc[r->ਧੱਕੋ_idx];

    desc->ਅਗਲਾ           = ਨਿਰਾ;
    desc->reserved1      = 0;
    desc->desc_ਮਾਪ      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->ਕ੍ਰਮ          = 0;
    desc->ਪੂਰਾ           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->ਪੰਕਤੀ_ਮਾਪ       = ਪੰਕਤੀ_ਮਾਪ;

#ਜੇ __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_ਵਿੱਥ     = 0;
    desc->dst_ਵਿੱਥ     = 0;
#ਨਹੀਂਤਾਂ
#ਜੇ __HVX_ARCH__ > 79
    desc->src_ਉੱਤਲਾ      = src_hi;
    desc->dst_ਉੱਤਲਾ      = dst_hi;
    desc->ਵੰਡ     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#ਨਹੀਂਤਾਂ
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#ਜੇ_ਅੰਤ
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->ਵਿੱਥ         = 0;
#ਜੇ_ਅੰਤ

    r->ਅੰਕੜੇ[r->ਧੱਕੋ_idx] = ddata;

    htp_trace_event_ਸ਼ੁਰੂ(r->trace, HTP_TRACE_EVT_DMA, r->ਧੱਕੋ_idx);

    ਜੇ (nrows) {
        dmlink(r->ਪੂਛ, desc);
        r->ਪੂਛ = desc;
    } ਨਹੀਂਤਾਂ {
        desc->ਪੂਰਾ = 1;
    }

    r->ਧੱਕੋ_idx = (r->ਧੱਕੋ_idx + 1) & r->idx_ਨਕਾਬ;
    ਵਾਪਸ true;
}

#ਜੇ __HVX_ARCH__ < 75
ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_ring_ਧੱਕੋ_single_contig(dma_ring * r, dma_ਅੰਕੜੇ ddata, size_t ਮਾਪ) {
    ਵਾਪਸ dma_ring_ਧੱਕੋ_single_1d(r, ddata, ਮਾਪ);
}
#ਨਹੀਂਤਾਂ
ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_ring_ਧੱਕੋ_single_contig(dma_ring * r, dma_ਅੰਕੜੇ ddata, size_t ਮਾਪ) {
    ਵਾਪਸ dma_ring_ਧੱਕੋ_single_2d(r, ddata, ਮਾਪ, ਮਾਪ, ਮਾਪ, 1);
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_ਅੰਕੜੇ dma_ring_ਕੱਢੋ(dma_ring * r) {
    dma_ਅੰਕੜੇ ddata = { 0 };

    ਜੇ (r->ਧੱਕੋ_idx == r->ਕੱਢੋ_idx) {
        ਵਾਪਸ ddata;
    }

    ddata = r->ਅੰਕੜੇ[r->ਕੱਢੋ_idx];

    ਅਸਥਿਰ dma_descriptor_2d * desc = &r->desc[r->ਕੱਢੋ_idx];

    // Wait for desc to complete
    ਜੇ (!desc->ਪੂਰਾ) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        ਜਦੋਂ (!desc->ਪੂਰਾ) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->ਕੱਢੋ_idx);

    r->ਕੱਢੋ_idx = (r->ਕੱਢੋ_idx + 1) & r->idx_ਨਕਾਬ;
    ਵਾਪਸ ddata;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_ਅੰਕੜੇ dma_ring_ਕੱਢੋ_nowait(dma_ring * r) {
    dma_ਅੰਕੜੇ ddata = { 0 };

    ਜੇ (r->ਧੱਕੋ_idx == r->ਕੱਢੋ_idx) {
        ਵਾਪਸ ddata;
    }

    ddata = r->ਅੰਕੜੇ[r->ਕੱਢੋ_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->ਕੱਢੋ_idx);

    r->ਕੱਢੋ_idx = (r->ਕੱਢੋ_idx + 1) & r->idx_ਨਕਾਬ;
    ਵਾਪਸ ddata;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_ring_ਖਾਲੀ_ਥਾਵਾਂ(dma_ring * r) {
    ਵਾਪਸ r->ਧੱਕੋ_idx == r->ਕੱਢੋ_idx;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dma_ring_flush(dma_ring * r) {
    ਜਦੋਂ (!dma_ring_ਖਾਲੀ_ਥਾਵਾਂ(r)) {
        dma_ring_ਕੱਢੋ(r);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t dma_ring_ਡੂੰਘਾਈ(dma_ring * r) {
    ਵਾਪਸ (r->ਧੱਕੋ_idx - r->ਕੱਢੋ_idx) & r->idx_ਨਕਾਬ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t dma_ring_ਸਮਰੱਥਾ(dma_ring * r) {
    ਵਾਪਸ r->ਸਮਰੱਥਾ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_ਕਤਾਰ_ਧੱਕੋ_single_1d(dma_ਕਤਾਰ * q, dma_ਅੰਕੜੇ ddata, size_t ਮਾਪ) {
    ਵਾਪਸ dma_ring_ਧੱਕੋ_single_1d(q->ring0, ddata, ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_ਕਤਾਰ_ਧੱਕੋ_single_2d(dma_ਕਤਾਰ * q, dma_ਅੰਕੜੇ ddata, size_t dst_stride, size_t src_stride, size_t ਪੰਕਤੀ_ਮਾਪ, size_t nrows) {
    ਵਾਪਸ dma_ring_ਧੱਕੋ_single_2d(q->ring0, ddata, dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, nrows);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_ਅੰਕੜੇ dma_ਕਤਾਰ_ਕੱਢੋ(dma_ਕਤਾਰ * q) {
    ਵਾਪਸ dma_ring_ਕੱਢੋ(q->ring0);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_ਅੰਕੜੇ dma_ਕਤਾਰ_ਕੱਢੋ_nowait(dma_ਕਤਾਰ * q) {
    ਵਾਪਸ dma_ring_ਕੱਢੋ_nowait(q->ring0);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_ਕਤਾਰ_ਖਾਲੀ_ਥਾਵਾਂ(dma_ਕਤਾਰ * q) {
    ਵਾਪਸ dma_ring_ਖਾਲੀ_ਥਾਵਾਂ(q->ring0);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dma_ਕਤਾਰ_flush(dma_ਕਤਾਰ * q) {
    dma_ring_flush(q->ring0);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t dma_ਕਤਾਰ_ਡੂੰਘਾਈ(dma_ਕਤਾਰ * q) {
    ਵਾਪਸ dma_ring_ਡੂੰਘਾਈ(q->ring0);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t dma_ਕਤਾਰ_ਸਮਰੱਥਾ(dma_ਕਤਾਰ * q) {
    ਵਾਪਸ dma_ring_ਸਮਰੱਥਾ(q->ring0);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_ਕਤਾਰ_ਧੱਕੋ(dma_ਕਤਾਰ * q, dma_ਅੰਕੜੇ ddata, size_t dst_stride, size_t src_stride, size_t ਪੰਕਤੀ_ਮਾਪ, size_t nrows) {
    ਜੇ (__builtin_expect(nrows == 0, 0)) {
        ਵਾਪਸ dma_ring_ਧੱਕੋ_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    ਜੇ (nrows == 1 || !((ਪੰਕਤੀ_ਮਾਪ ^ src_stride) | (ਪੰਕਤੀ_ਮਾਪ ^ dst_stride))) {
        ਅਚਲ size_t ਕੁੱਲ = ਪੰਕਤੀ_ਮਾਪ * nrows;
        ਜੇ (__builtin_expect(ਕੁੱਲ <= DMA_MAX_ਮਾਪ_24_B, 1)) {
            ਵਾਪਸ dma_ring_ਧੱਕੋ_single_contig(q->ring0, ddata, ਕੁੱਲ);
        }
        ਵਾਪਸ dma_ਕਤਾਰ_ਧੱਕੋ_fallback_contig(q, ddata, ਕੁੱਲ);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    ਜੇ (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (ਪੰਕਤੀ_ਮਾਪ | src_stride | dst_stride) <= DMA_MAX_2_D_ਪੰਕਤੀ_ਮਾਪ, 1)) {
        ਵਾਪਸ dma_ring_ਧੱਕੋ_single_2d(q->ring0, ddata, dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#ਜੇ __HVX_ARCH__ < 75
    ਜੇ (ਪੰਕਤੀ_ਮਾਪ <= DMA_MAX_ਮਾਪ_16_B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        ਵਾਪਸ dma_ਕਤਾਰ_ਧੱਕੋ_fallback_2d(q, ddata, dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, nrows);
    }
    ਵਾਪਸ dma_ਕਤਾਰ_ਧੱਕੋ_fallback_1d(q, ddata, dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, nrows);
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ dma_ਕਤਾਰ_ਧੱਕੋ_fallback_2d(q, ddata, dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, nrows);
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dma_sync_read(dma_ਕਤਾਰ * dma_q, ਖਾਲੀ * dst, dma_addr_t src, size_t ਬਾਈਟਾਂ) {
    ਅਚਲ uint32_t b = (uint32_t) ਬਾਈਟਾਂ;
    ਜੇ (b > 0) {
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst, src), b, b, b, 1);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dma_sync_write(dma_ਕਤਾਰ * dma_q, dma_addr_t dst, ਅਚਲ ਖਾਲੀ * src, size_t ਬਾਈਟਾਂ) {
    ਅਚਲ uint32_t b = (uint32_t) ਬਾਈਟਾਂ;
    ਜੇ (b > 0) {
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst, src), b, b, b, 1);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DMA_CACHE_MAX_ਮਾਪ 256U

// Fully assoc LRU cache
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    uint8_t *ਅਧਾਰ;
    uint32_t ਸਤਰ_ਮਾਪ;
    uint32_t ਸਮਰੱਥਾ;
    dma_addr_t src[DMA_CACHE_MAX_ਮਾਪ];
    uint16_t age[DMA_CACHE_MAX_ਮਾਪ];
} dma_cache_fa;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dma_cache_fa_init(dma_cache_fa *c, uint8_t *ਅਧਾਰ, uint32_t ਸਤਰ_ਮਾਪ, uint32_t ਸਮਰੱਥਾ)
{
    c->ਸਮਰੱਥਾ  = (ਸਮਰੱਥਾ > DMA_CACHE_MAX_ਮਾਪ) ? DMA_CACHE_MAX_ਮਾਪ : ਸਮਰੱਥਾ;
    c->ਅਧਾਰ      = ਅਧਾਰ;
    c->ਸਤਰ_ਮਾਪ = ਸਤਰ_ਮਾਪ;

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ i=0; i < c->ਸਮਰੱਥਾ; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_cache_fa_ਧੱਕੋ(dma_ਕਤਾਰ *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t ਪੰਕਤੀ_ਮਾਪ, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ i=0; i < c->ਸਮਰੱਥਾ; i++) {
        ਜੇ (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->ਅਧਾਰ + (i * c->ਸਤਰ_ਮਾਪ); nrows = 0; // dummy dma
        } ਨਹੀਂਤਾਂ {
            c->age[i]++;
            ਜੇ (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    ਜੇ (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->ਅਧਾਰ + o_idx * c->ਸਤਰ_ਮਾਪ; // normal nrows dma
        ਵਾਪਸ dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(dst, src_addr), dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, nrows);
    }

    ਵਾਪਸ dma_ਕਤਾਰ_ਧੱਕੋ_single_1d(q, dma_make_ਅੰਕੜੇ(dst, src_addr), 0);
}

// Direct mapped cache
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    uint8_t *ਅਧਾਰ;
    uint32_t ਸਤਰ_ਮਾਪ;
    uint32_t ਸਮਰੱਥਾ;
    uint32_t idx_ਨਕਾਬ;
    dma_addr_t src[DMA_CACHE_MAX_ਮਾਪ];
} dma_cache_dm;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dma_cache_dm_init(dma_cache_dm *c, uint8_t *ਅਧਾਰ, uint32_t ਸਤਰ_ਮਾਪ, uint32_t ਸਮਰੱਥਾ)
{
    c->ਸਮਰੱਥਾ  = (ਸਮਰੱਥਾ > DMA_CACHE_MAX_ਮਾਪ) ? DMA_CACHE_MAX_ਮਾਪ : ਸਮਰੱਥਾ;
    c->idx_ਨਕਾਬ  = c->ਸਮਰੱਥਾ - 1;
    c->ਅਧਾਰ      = ਅਧਾਰ;
    c->ਸਤਰ_ਮਾਪ = ਸਤਰ_ਮਾਪ;

    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ i=0; i < c->ਸਮਰੱਥਾ; i++) {
        c->src[i] = 0;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool dma_cache_dm_ਧੱਕੋ(dma_ਕਤਾਰ *q, dma_cache_dm *c, uint32_t ਥਾਂ, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t ਪੰਕਤੀ_ਮਾਪ, uint32_t nrows)
{
    ਅਚਲ uint32_t i = ਥਾਂ & c->idx_ਨਕਾਬ;
    uint8_t * dst = c->ਅਧਾਰ + (i * c->ਸਤਰ_ਮਾਪ);

    ਜੇ (c->src[i] == src_addr) {
        ਵਾਪਸ dma_ਕਤਾਰ_ਧੱਕੋ_single_1d(q, dma_make_ਅੰਕੜੇ(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    ਵਾਪਸ dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(dst, src_addr), dst_stride, src_stride, ਪੰਕਤੀ_ਮਾਪ, nrows);
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}  // extern "C"
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ /* HTP_DMA_H */
