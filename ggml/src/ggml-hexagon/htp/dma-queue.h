#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_DMA_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_DMA_H

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <hexagon_types.h>
#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ "hex-utils.h"

#ಅನ್ತರ್ಭಾವ "hex-profile.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ dma_descriptor_1d_s {
    ಶೂನ್ಯ *   ಮುಂದಿನ;
    uint32_t ಗಾತ್ರ_2:24;
    uint32_t desc_ಗಾತ್ರ:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ಕ್ರಮ:1;
    uint32_t ಮುಗಿದ:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#ಆದರೆ __HVX_ARCH__ < 75

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ dma_descriptor_2d_s {
    ಶೂನ್ಯ *   ಮುಂದಿನ;
    uint32_t reserved0:24;
    uint32_t desc_ಗಾತ್ರ:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ಕ್ರಮ:1;
    uint32_t ಮುಗಿದ:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_ಸ್ಥಾನಾಂತರ:16;
    uint32_t dst_ಸ್ಥಾನಾಂತರ:16;
} dma_descriptor_2d;

#ಇಲ್ಲದಿದ್ದರೆ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ dma_descriptor_2d_s {
    ಶೂನ್ಯ *   ಮುಂದಿನ;
    uint32_t dst_stride:24;
    uint32_t desc_ಗಾತ್ರ:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t ಕ್ರಮ:1;
    uint32_t ಮುಗಿದ:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#ಆದರೆ __HVX_ARCH__ > 79
    uint32_t src_ಊರ್ಧ್ವ:8;
    uint32_t dst_ಊರ್ಧ್ವ:8;
    uint32_t ಹಂಚಿಕೆ:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#ಇಲ್ಲದಿದ್ದರೆ
    uint32_t reserved0:24;
#ಆದರೆ_ಅಂತ್ಯ
    uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t ಸ್ಥಾನಾಂತರ:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ __HVX_ARCH__ > 79
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint64_t dma_addr_t;
#ಇಲ್ಲದಿದ್ದರೆ
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint32_t dma_addr_t;
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    dma_addr_t dst;
    dma_addr_t src;
} dma_ದತ್ತಾಂಶ;

// Hardware descriptor field limits
#ವ್ಯಾಖ್ಯಾನಿಸು DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#ವ್ಯಾಖ್ಯಾನಿಸು DMA_MAX_ಗಾತ್ರ_16_B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#ವ್ಯಾಖ್ಯಾನಿಸು DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#ವ್ಯಾಖ್ಯಾನಿಸು DMA_MAX_ಗಾತ್ರ_24_B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#ವ್ಯಾಖ್ಯಾನಿಸು DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#ವ್ಯಾಖ್ಯಾನಿಸು DMA_ಸುರಕ್ಷಿತ_CHUNK_ಗಾತ್ರ    0x00F00000u    // ~15MB safe contiguous chunk size

#ಆದರೆ __HVX_ARCH__ < 75
#ವ್ಯಾಖ್ಯಾನಿಸು DMA_MAX_2_D_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ    DMA_MAX_ಗಾತ್ರ_16_B
#ವ್ಯಾಖ್ಯಾನಿಸು DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು DMA_MAX_2_D_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ    DMA_MAX_ಗಾತ್ರ_24_B
#ವ್ಯಾಖ್ಯಾನಿಸು DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು DMA_FALLBACK_ಸಾಮರ್ಥ್ಯ  16u            // descriptors in secondary fallback ring

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ dma_ring_s dma_ring;
ರಚನೆ dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * ಬಾಲ;      // tail pointer
    dma_ದತ್ತಾಂಶ          * ದತ್ತಾಂಶ;      // dst/src data
    uint32_t            ತಳ್ಳು_idx;
    uint32_t            ತೆಗೆ_idx;
    uint32_t            ಸಾಮರ್ಥ್ಯ;
    uint32_t            idx_ಮುಖವಾಡ;
    ರಚನೆ htp_thread_trace * trace;
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ dma_ಸರತಿ_s dma_ಸರತಿ;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ dma_ಸರತಿ * dma_ಸರತಿ_t;

ರಚನೆ dma_ಸರತಿ_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_ಸರತಿ_sizeof(size_t ಸಾಮರ್ಥ್ಯ);
size_t      dma_ಸರತಿ_alignof(ಶೂನ್ಯ);
dma_ಸರತಿ_t dma_ಸರತಿ_init(ಶೂನ್ಯ * ptr, size_t ಸಾಮರ್ಥ್ಯ, ರಚನೆ htp_thread_trace * trace);
ಶೂನ್ಯ        dma_ಸರತಿ_free(dma_ಸರತಿ_t q);

size_t      dma_ಸರತಿ_alias_sizeof(ಶೂನ್ಯ);
dma_ಸರತಿ_t dma_ಸರತಿ_alias_init(ಶೂನ್ಯ * ptr, dma_ಸರತಿ_t main_q);
ಶೂನ್ಯ        dma_ಸರತಿ_alias_free(dma_ಸರತಿ_t q);

bool        dma_ಸರತಿ_ತಳ್ಳು_fallback_2d(dma_ಸರತಿ * q, dma_ದತ್ತಾಂಶ ddata, size_t dst_stride, size_t src_stride, size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, size_t nrows);
bool        dma_ಸರತಿ_ತಳ್ಳು_fallback_contig(dma_ಸರತಿ * q, dma_ದತ್ತಾಂಶ ddata, size_t ಒಟ್ಟು);
#ಆದರೆ __HVX_ARCH__ < 75
bool        dma_ಸರತಿ_ತಳ್ಳು_fallback_1d(dma_ಸರತಿ * q, dma_ದತ್ತಾಂಶ ddata, size_t dst_stride, size_t src_stride, size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, size_t nrows);
#ಆದರೆ_ಅಂತ್ಯ

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dmstart(ಶೂನ್ಯ * ಮುಂದಿನ) {
    asm ಅಸ್ಥಿರ(" release(%0):at" : : "r"(ಮುಂದಿನ));
    asm ಅಸ್ಥಿರ(" dmstart(%0)" : : "r"(ಮುಂದಿನ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dmlink(ಶೂನ್ಯ * cur, ಶೂನ್ಯ * ಮುಂದಿನ) {
    asm ಅಸ್ಥಿರ(" release(%0):at" : : "r"(ಮುಂದಿನ));
    asm ಅಸ್ಥಿರ(" dmlink(%0, %1)" : : "r"(cur), "r"(ಮುಂದಿನ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ dmpoll(ಶೂನ್ಯ) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ret = 0;
    asm ಅಸ್ಥಿರ(" %0 = dmpoll" : "=r"(ret) : : "memory");
    ಹಿಂತಿರುಗಿ ret;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ dmwait(ಶೂನ್ಯ) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ret = 0;
    asm ಅಸ್ಥಿರ(" %0 = dmwait" : "=r"(ret) : : "memory");
    ಹಿಂತಿರುಗಿ ret;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_ದತ್ತಾಂಶ dma_make_ದತ್ತಾಂಶ_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_ದತ್ತಾಂಶ d = { dst, src };
    ಹಿಂತಿರುಗಿ d;
}

#ವ್ಯಾಖ್ಯಾನಿಸು dma_make_ದತ್ತಾಂಶ(dst, src) dma_make_ದತ್ತಾಂಶ_impl((dma_addr_t) (dst), (dma_addr_t) (src))

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_ring_ತಳ್ಳು_single_1d(dma_ring * r, dma_ದತ್ತಾಂಶ ddata, size_t ಗಾತ್ರ_2) {
#ಆದರೆ __HVX_ARCH__ > 79
    ಪ್ರತಿಜ್ಞಾ(!((ddata.src | ddata.dst) >> 32) || ಗಾತ್ರ_2 == 0);
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (((r->ತಳ್ಳು_idx + 1) & r->idx_ಮುಖವಾಡ) == r->ತೆಗೆ_idx) {
        ಹಿಂತಿರುಗಿ false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->ತಳ್ಳು_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->ಗಾತ್ರ_2 = ಗಾತ್ರ_2;

    r->ದತ್ತಾಂಶ[r->ತಳ್ಳು_idx] = ddata;

    htp_trace_event_ಆರಂಭ(r->trace, HTP_TRACE_EVT_DMA, r->ತಳ್ಳು_idx);

    ಆದರೆ (ಗಾತ್ರ_2) {
        desc->ಮುಂದಿನ       = ಖಾಲಿ;
        desc->desc_ಗಾತ್ರ  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->ಕ್ರಮ      = 0;
        desc->ಮುಗಿದ       = 0;

        dmlink(r->ಬಾಲ, desc);
        r->ಬಾಲ = (dma_descriptor_2d *) desc;
    } ಇಲ್ಲದಿದ್ದರೆ {
        desc->desc_ಗಾತ್ರ = 0;
        desc->ಮುಗಿದ      = 1;
    }

    r->ತಳ್ಳು_idx = (r->ತಳ್ಳು_idx + 1) & r->idx_ಮುಖವಾಡ;
    ಹಿಂತಿರುಗಿ true;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_ring_ತಳ್ಳು_single_2d(dma_ring * r, dma_ದತ್ತಾಂಶ ddata, size_t dst_stride, size_t src_stride, size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, size_t nrows) {
#ಆದರೆ __HVX_ARCH__ > 79
    ಪ್ರತಿಜ್ಞಾ(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
    ಸ್ಥಿರಾಂಕ uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    ಸ್ಥಿರಾಂಕ uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    ಸ್ಥಿರಾಂಕ bool is_ext     = (src_hi | dst_hi) != 0;
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (((r->ತಳ್ಳು_idx + 1) & r->idx_ಮುಖವಾಡ) == r->ತೆಗೆ_idx) {
        ಹಿಂತಿರುಗಿ false;
    }

    dma_descriptor_2d * desc = &r->desc[r->ತಳ್ಳು_idx];

    desc->ಮುಂದಿನ           = ಖಾಲಿ;
    desc->reserved1      = 0;
    desc->desc_ಗಾತ್ರ      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->ಕ್ರಮ          = 0;
    desc->ಮುಗಿದ           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->ಅಡ್ಡಸಾಲು_ಗಾತ್ರ       = ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

#ಆದರೆ __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_ಸ್ಥಾನಾಂತರ     = 0;
    desc->dst_ಸ್ಥಾನಾಂತರ     = 0;
#ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ __HVX_ARCH__ > 79
    desc->src_ಊರ್ಧ್ವ      = src_hi;
    desc->dst_ಊರ್ಧ್ವ      = dst_hi;
    desc->ಹಂಚಿಕೆ     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#ಇಲ್ಲದಿದ್ದರೆ
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#ಆದರೆ_ಅಂತ್ಯ
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->ಸ್ಥಾನಾಂತರ         = 0;
#ಆದರೆ_ಅಂತ್ಯ

    r->ದತ್ತಾಂಶ[r->ತಳ್ಳು_idx] = ddata;

    htp_trace_event_ಆರಂಭ(r->trace, HTP_TRACE_EVT_DMA, r->ತಳ್ಳು_idx);

    ಆದರೆ (nrows) {
        dmlink(r->ಬಾಲ, desc);
        r->ಬಾಲ = desc;
    } ಇಲ್ಲದಿದ್ದರೆ {
        desc->ಮುಗಿದ = 1;
    }

    r->ತಳ್ಳು_idx = (r->ತಳ್ಳು_idx + 1) & r->idx_ಮುಖವಾಡ;
    ಹಿಂತಿರುಗಿ true;
}

#ಆದರೆ __HVX_ARCH__ < 75
ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_ring_ತಳ್ಳು_single_contig(dma_ring * r, dma_ದತ್ತಾಂಶ ddata, size_t ಗಾತ್ರ_2) {
    ಹಿಂತಿರುಗಿ dma_ring_ತಳ್ಳು_single_1d(r, ddata, ಗಾತ್ರ_2);
}
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_ring_ತಳ್ಳು_single_contig(dma_ring * r, dma_ದತ್ತಾಂಶ ddata, size_t ಗಾತ್ರ_2) {
    ಹಿಂತಿರುಗಿ dma_ring_ತಳ್ಳು_single_2d(r, ddata, ಗಾತ್ರ_2, ಗಾತ್ರ_2, ಗಾತ್ರ_2, 1);
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_ದತ್ತಾಂಶ dma_ring_ತೆಗೆ(dma_ring * r) {
    dma_ದತ್ತಾಂಶ ddata = { 0 };

    ಆದರೆ (r->ತಳ್ಳು_idx == r->ತೆಗೆ_idx) {
        ಹಿಂತಿರುಗಿ ddata;
    }

    ddata = r->ದತ್ತಾಂಶ[r->ತೆಗೆ_idx];

    ಅಸ್ಥಿರ dma_descriptor_2d * desc = &r->desc[r->ತೆಗೆ_idx];

    // Wait for desc to complete
    ಆದರೆ (!desc->ಮುಗಿದ) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        ಆಗ (!desc->ಮುಗಿದ) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->ತೆಗೆ_idx);

    r->ತೆಗೆ_idx = (r->ತೆಗೆ_idx + 1) & r->idx_ಮುಖವಾಡ;
    ಹಿಂತಿರುಗಿ ddata;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_ದತ್ತಾಂಶ dma_ring_ತೆಗೆ_nowait(dma_ring * r) {
    dma_ದತ್ತಾಂಶ ddata = { 0 };

    ಆದರೆ (r->ತಳ್ಳು_idx == r->ತೆಗೆ_idx) {
        ಹಿಂತಿರುಗಿ ddata;
    }

    ddata = r->ದತ್ತಾಂಶ[r->ತೆಗೆ_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->ತೆಗೆ_idx);

    r->ತೆಗೆ_idx = (r->ತೆಗೆ_idx + 1) & r->idx_ಮುಖವಾಡ;
    ಹಿಂತಿರುಗಿ ddata;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_ring_ಖಾಲಿಗಳು(dma_ring * r) {
    ಹಿಂತಿರುಗಿ r->ತಳ್ಳು_idx == r->ತೆಗೆ_idx;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dma_ring_flush(dma_ring * r) {
    ಆಗ (!dma_ring_ಖಾಲಿಗಳು(r)) {
        dma_ring_ತೆಗೆ(r);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t dma_ring_ಆಳ(dma_ring * r) {
    ಹಿಂತಿರುಗಿ (r->ತಳ್ಳು_idx - r->ತೆಗೆ_idx) & r->idx_ಮುಖವಾಡ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t dma_ring_ಸಾಮರ್ಥ್ಯ(dma_ring * r) {
    ಹಿಂತಿರುಗಿ r->ಸಾಮರ್ಥ್ಯ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_ಸರತಿ_ತಳ್ಳು_single_1d(dma_ಸರತಿ * q, dma_ದತ್ತಾಂಶ ddata, size_t ಗಾತ್ರ_2) {
    ಹಿಂತಿರುಗಿ dma_ring_ತಳ್ಳು_single_1d(q->ring0, ddata, ಗಾತ್ರ_2);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_ಸರತಿ_ತಳ್ಳು_single_2d(dma_ಸರತಿ * q, dma_ದತ್ತಾಂಶ ddata, size_t dst_stride, size_t src_stride, size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, size_t nrows) {
    ಹಿಂತಿರುಗಿ dma_ring_ತಳ್ಳು_single_2d(q->ring0, ddata, dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_ದತ್ತಾಂಶ dma_ಸರತಿ_ತೆಗೆ(dma_ಸರತಿ * q) {
    ಹಿಂತಿರುಗಿ dma_ring_ತೆಗೆ(q->ring0);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_ದತ್ತಾಂಶ dma_ಸರತಿ_ತೆಗೆ_nowait(dma_ಸರತಿ * q) {
    ಹಿಂತಿರುಗಿ dma_ring_ತೆಗೆ_nowait(q->ring0);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_ಸರತಿ_ಖಾಲಿಗಳು(dma_ಸರತಿ * q) {
    ಹಿಂತಿರುಗಿ dma_ring_ಖಾಲಿಗಳು(q->ring0);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dma_ಸರತಿ_flush(dma_ಸರತಿ * q) {
    dma_ring_flush(q->ring0);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t dma_ಸರತಿ_ಆಳ(dma_ಸರತಿ * q) {
    ಹಿಂತಿರುಗಿ dma_ring_ಆಳ(q->ring0);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t dma_ಸರತಿ_ಸಾಮರ್ಥ್ಯ(dma_ಸರತಿ * q) {
    ಹಿಂತಿರುಗಿ dma_ring_ಸಾಮರ್ಥ್ಯ(q->ring0);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_ಸರತಿ_ತಳ್ಳು(dma_ಸರತಿ * q, dma_ದತ್ತಾಂಶ ddata, size_t dst_stride, size_t src_stride, size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, size_t nrows) {
    ಆದರೆ (__builtin_expect(nrows == 0, 0)) {
        ಹಿಂತಿರುಗಿ dma_ring_ತಳ್ಳು_single_1d(q->ring0, ddata, 0);
    }

    // 1. Hot path: Contiguous or single-row (80-90% of calls)
    ಆದರೆ (nrows == 1 || !((ಅಡ್ಡಸಾಲು_ಗಾತ್ರ ^ src_stride) | (ಅಡ್ಡಸಾಲು_ಗಾತ್ರ ^ dst_stride))) {
        ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು = ಅಡ್ಡಸಾಲು_ಗಾತ್ರ * nrows;
        ಆದರೆ (__builtin_expect(ಒಟ್ಟು <= DMA_MAX_ಗಾತ್ರ_24_B, 1)) {
            ಹಿಂತಿರುಗಿ dma_ring_ತಳ್ಳು_single_contig(q->ring0, ddata, ಒಟ್ಟು);
        }
        ಹಿಂತಿರುಗಿ dma_ಸರತಿ_ತಳ್ಳು_fallback_contig(q, ddata, ಒಟ್ಟು);
    }

    // 2. Hot path: Standard strided 2D (10-20% of calls)
    ಆದರೆ (__builtin_expect(nrows <= DMA_MAX_NROWS &&
                         (ಅಡ್ಡಸಾಲು_ಗಾತ್ರ | src_stride | dst_stride) <= DMA_MAX_2_D_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1)) {
        ಹಿಂತಿರುಗಿ dma_ring_ತಳ್ಳು_single_2d(q->ring0, ddata, dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows);
    }

    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
#ಆದರೆ __HVX_ARCH__ < 75
    ಆದರೆ (ಅಡ್ಡಸಾಲು_ಗಾತ್ರ <= DMA_MAX_ಗಾತ್ರ_16_B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
        ಹಿಂತಿರುಗಿ dma_ಸರತಿ_ತಳ್ಳು_fallback_2d(q, ddata, dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows);
    }
    ಹಿಂತಿರುಗಿ dma_ಸರತಿ_ತಳ್ಳು_fallback_1d(q, ddata, dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows);
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ dma_ಸರತಿ_ತಳ್ಳು_fallback_2d(q, ddata, dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows);
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dma_sync_read(dma_ಸರತಿ * dma_q, ಶೂನ್ಯ * dst, dma_addr_t src, size_t ಬೈಟುಗಳು) {
    ಸ್ಥಿರಾಂಕ uint32_t b = (uint32_t) ಬೈಟುಗಳು;
    ಆದರೆ (b > 0) {
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst, src), b, b, b, 1);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dma_sync_write(dma_ಸರತಿ * dma_q, dma_addr_t dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t ಬೈಟುಗಳು) {
    ಸ್ಥಿರಾಂಕ uint32_t b = (uint32_t) ಬೈಟುಗಳು;
    ಆದರೆ (b > 0) {
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst, src), b, b, b, 1);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು DMA_CACHE_MAX_ಗಾತ್ರ 256U

// Fully assoc LRU cache
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    uint8_t *ತಳ;
    uint32_t ಸಾಲು_ಗಾತ್ರ;
    uint32_t ಸಾಮರ್ಥ್ಯ;
    dma_addr_t src[DMA_CACHE_MAX_ಗಾತ್ರ];
    uint16_t age[DMA_CACHE_MAX_ಗಾತ್ರ];
} dma_cache_fa;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dma_cache_fa_init(dma_cache_fa *c, uint8_t *ತಳ, uint32_t ಸಾಲು_ಗಾತ್ರ, uint32_t ಸಾಮರ್ಥ್ಯ)
{
    c->ಸಾಮರ್ಥ್ಯ  = (ಸಾಮರ್ಥ್ಯ > DMA_CACHE_MAX_ಗಾತ್ರ) ? DMA_CACHE_MAX_ಗಾತ್ರ : ಸಾಮರ್ಥ್ಯ;
    c->ತಳ      = ತಳ;
    c->ಸಾಲು_ಗಾತ್ರ = ಸಾಲು_ಗಾತ್ರ;

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ i=0; i < c->ಸಾಮರ್ಥ್ಯ; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_cache_fa_ತಳ್ಳು(dma_ಸರತಿ *q, dma_cache_fa *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ i=0; i < c->ಸಾಮರ್ಥ್ಯ; i++) {
        ಆದರೆ (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->ತಳ + (i * c->ಸಾಲು_ಗಾತ್ರ); nrows = 0; // dummy dma
        } ಇಲ್ಲದಿದ್ದರೆ {
            c->age[i]++;
            ಆದರೆ (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    ಆದರೆ (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->ತಳ + o_idx * c->ಸಾಲು_ಗಾತ್ರ; // normal nrows dma
        ಹಿಂತಿರುಗಿ dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(dst, src_addr), dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows);
    }

    ಹಿಂತಿರುಗಿ dma_ಸರತಿ_ತಳ್ಳು_single_1d(q, dma_make_ದತ್ತಾಂಶ(dst, src_addr), 0);
}

// Direct mapped cache
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    uint8_t *ತಳ;
    uint32_t ಸಾಲು_ಗಾತ್ರ;
    uint32_t ಸಾಮರ್ಥ್ಯ;
    uint32_t idx_ಮುಖವಾಡ;
    dma_addr_t src[DMA_CACHE_MAX_ಗಾತ್ರ];
} dma_cache_dm;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dma_cache_dm_init(dma_cache_dm *c, uint8_t *ತಳ, uint32_t ಸಾಲು_ಗಾತ್ರ, uint32_t ಸಾಮರ್ಥ್ಯ)
{
    c->ಸಾಮರ್ಥ್ಯ  = (ಸಾಮರ್ಥ್ಯ > DMA_CACHE_MAX_ಗಾತ್ರ) ? DMA_CACHE_MAX_ಗಾತ್ರ : ಸಾಮರ್ಥ್ಯ;
    c->idx_ಮುಖವಾಡ  = c->ಸಾಮರ್ಥ್ಯ - 1;
    c->ತಳ      = ತಳ;
    c->ಸಾಲು_ಗಾತ್ರ = ಸಾಲು_ಗಾತ್ರ;

    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ i=0; i < c->ಸಾಮರ್ಥ್ಯ; i++) {
        c->src[i] = 0;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool dma_cache_dm_ತಳ್ಳು(dma_ಸರತಿ *q, dma_cache_dm *c, uint32_t ಸ್ಥಾನ, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, uint32_t nrows)
{
    ಸ್ಥಿರಾಂಕ uint32_t i = ಸ್ಥಾನ & c->idx_ಮುಖವಾಡ;
    uint8_t * dst = c->ತಳ + (i * c->ಸಾಲು_ಗಾತ್ರ);

    ಆದರೆ (c->src[i] == src_addr) {
        ಹಿಂತಿರುಗಿ dma_ಸರತಿ_ತಳ್ಳು_single_1d(q, dma_make_ದತ್ತಾಂಶ(dst, src_addr), 0); // dummy dma
    }

    c->src[i] = src_addr;
    ಹಿಂತಿರುಗಿ dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(dst, src_addr), dst_stride, src_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows);
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}  // extern "C"
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ /* HTP_DMA_H */
