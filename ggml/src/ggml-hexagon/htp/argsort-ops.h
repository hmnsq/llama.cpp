#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_ARGSORT_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ARGSORT_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"

ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params {
    int32_t  n_ਧਾਗੇ;
    int32_t  ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    int32_t  ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    int32_t  ਪੰਕਤੀ_ਸਿਰਾ;
    int32_t  ne00;
    int32_t  k;
    int32_t  ਕ੍ਰਮ;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_ਸਿਖਰ_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_ਮਾਪ;
    int32_t  phase1_ਥਾਂ_ਮਾਪ;
    int32_t  ਮਿਲਾਓ_ਮੁੱਲਾਂ_off;
    int32_t  ਮਿਲਾਓ_indices_off;
    int32_t  ਮਿਲਾਓ_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

ਬਣਤਰ htp_ਛਾਂਟੋ_vtcm_layout {
    size_t   ਕੁੱਲ_ਬਾਈਟਾਂ;
    size_t   phase1_ਥਾਂ_ਮਾਪ;
    size_t   ਮਿਲਾਓ_ਮੁੱਲਾਂ_off;
    size_t   ਮਿਲਾਓ_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t ਮਿਲਾਓ_elems;
    uint32_t n_ਧਾਗੇ;
    uint32_t n_slots;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_ਛਾਂਟੋ_solve_layout(
    ਬਣਤਰ htp_ਛਾਂਟੋ_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ,
    uint32_t k,
    uint32_t n_ਧਾਗੇ_max,
    size_t vtcm_budget,
    bool is_ਸਿਖਰ_k) {

    memset(layout, 0, ਆਕਾਰ(*layout));

    ਜੇ (ਕੁੱਲ_ਪੰਕਤੀਆਂ > 1) {
        uint32_t n_ਧਾਗੇ = ਕੁੱਲ_ਪੰਕਤੀਆਂ < n_ਧਾਗੇ_max ? ਕੁੱਲ_ਪੰਕਤੀਆਂ : n_ਧਾਗੇ_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        ਜਦੋਂ (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t ਮੁੱਲਾਂ_ਮਾਪ  = ((ne00_padded * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) + 127) & ~127;
        size_t indices_ਮਾਪ = ((ne00_padded * ਆਕਾਰ(int32_t)) + 127) & ~127;
        size_t spad_ਪ੍ਰਤੀ_ਥਾਂ = ((ਮੁੱਲਾਂ_ਮਾਪ + indices_ਮਾਪ) + 255) & ~255;

        uint32_t n_slots = 2;
        ਜੇ (spad_ਪ੍ਰਤੀ_ਥਾਂ * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_ਪ੍ਰਤੀ_thread = spad_ਪ੍ਰਤੀ_ਥਾਂ * n_slots;

        ਜਦੋਂ (n_ਧਾਗੇ > 1 && (spad_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ) > vtcm_budget) {
            n_ਧਾਗੇ--;
        }

        size_t ਕੁੱਲ_ਬਾਈਟਾਂ = spad_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
        ਜੇ (ਕੁੱਲ_ਬਾਈਟਾਂ > vtcm_budget) {
            ਵਾਪਸ false;
        }

        layout->ਕੁੱਲ_ਬਾਈਟਾਂ      = ਕੁੱਲ_ਬਾਈਟਾਂ;
        layout->phase1_ਥਾਂ_ਮਾਪ = spad_ਪ੍ਰਤੀ_ਥਾਂ;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_ਧਾਗੇ        = n_ਧਾਗੇ;
        layout->n_slots          = n_slots;
        ਵਾਪਸ true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    ਜਦੋਂ (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    ਜੇ (ne00 > 1024) {
        ਜਦੋਂ (n_chunks * 2 <= n_ਧਾਗੇ_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_ਮੁੱਲਾਂ_ਮਾਪ  = ((chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) + 127) & ~127;
    size_t phase1_indices_ਮਾਪ = ((chunk_elems * ਆਕਾਰ(int32_t)) + 127) & ~127;
    size_t phase1_ਥਾਂ_ਮਾਪ    = ((phase1_ਮੁੱਲਾਂ_ਮਾਪ + phase1_indices_ਮਾਪ) + 255) & ~255;
    size_t phase1_ਕੁੱਲ_ਮਾਪ   = phase1_ਥਾਂ_ਮਾਪ * n_chunks;

    size_t ਮਿਲਾਓ_ਮੁੱਲਾਂ_ਮਾਪ  = 0;
    size_t ਮਿਲਾਓ_indices_ਮਾਪ = 0;
    size_t ਮਿਲਾਓ_ਮੁੱਲਾਂ_off   = phase1_ਕੁੱਲ_ਮਾਪ;
    size_t ਮਿਲਾਓ_indices_off  = 0;
    uint32_t ਮਿਲਾਓ_elems      = 0;

    ਜੇ (n_chunks > 1) {
        ਜੇ (is_ਸਿਖਰ_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t ਕੁੱਲ_candidates = n_chunks * local_k;
            uint32_t ਮਿਲਾਓ_n_vec = (ਕੁੱਲ_candidates + 31) / 32;
            uint32_t ਮਿਲਾਓ_n_vec_pow2 = 1;
            ਜਦੋਂ (ਮਿਲਾਓ_n_vec_pow2 < ਮਿਲਾਓ_n_vec) ਮਿਲਾਓ_n_vec_pow2 <<= 1;
            ਮਿਲਾਓ_elems = ਮਿਲਾਓ_n_vec_pow2 * 32;
        } ਨਹੀਂਤਾਂ {
            ਮਿਲਾਓ_elems = n_vec_pow2 * 32;
        }
        ਮਿਲਾਓ_ਮੁੱਲਾਂ_ਮਾਪ  = ((ਮਿਲਾਓ_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) + 127) & ~127;
        ਮਿਲਾਓ_indices_ਮਾਪ = ((ਮਿਲਾਓ_elems * ਆਕਾਰ(int32_t)) + 127) & ~127;
        ਮਿਲਾਓ_indices_off  = ਮਿਲਾਓ_ਮੁੱਲਾਂ_off + ਮਿਲਾਓ_ਮੁੱਲਾਂ_ਮਾਪ;
    }

    size_t ਕੁੱਲ_ਬਾਈਟਾਂ = phase1_ਕੁੱਲ_ਮਾਪ + ਮਿਲਾਓ_ਮੁੱਲਾਂ_ਮਾਪ + ਮਿਲਾਓ_indices_ਮਾਪ;

    ਜੇ (ਕੁੱਲ_ਬਾਈਟਾਂ > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_ਮੁੱਲਾਂ_ਮਾਪ  = ((chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) + 127) & ~127;
        phase1_indices_ਮਾਪ = ((chunk_elems * ਆਕਾਰ(int32_t)) + 127) & ~127;
        phase1_ਥਾਂ_ਮਾਪ    = ((phase1_ਮੁੱਲਾਂ_ਮਾਪ + phase1_indices_ਮਾਪ) + 255) & ~255;
        phase1_ਕੁੱਲ_ਮਾਪ   = phase1_ਥਾਂ_ਮਾਪ;
        ਮਿਲਾਓ_ਮੁੱਲਾਂ_ਮਾਪ   = 0;
        ਮਿਲਾਓ_indices_ਮਾਪ  = 0;
        ਮਿਲਾਓ_ਮੁੱਲਾਂ_off    = phase1_ਕੁੱਲ_ਮਾਪ;
        ਮਿਲਾਓ_indices_off   = 0;
        ਮਿਲਾਓ_elems         = 0;
        ਕੁੱਲ_ਬਾਈਟਾਂ         = phase1_ਕੁੱਲ_ਮਾਪ;
    }

    ਜੇ (ਕੁੱਲ_ਬਾਈਟਾਂ > vtcm_budget) {
        ਵਾਪਸ false;
    }

    uint32_t n_slots = 1;
    ਜੇ (n_chunks == 1 && phase1_ਥਾਂ_ਮਾਪ * 2 <= vtcm_budget) {
        n_slots = 2;
        ਕੁੱਲ_ਬਾਈਟਾਂ = phase1_ਥਾਂ_ਮਾਪ * 2;
    }

    layout->ਕੁੱਲ_ਬਾਈਟਾਂ       = ਕੁੱਲ_ਬਾਈਟਾਂ;
    layout->phase1_ਥਾਂ_ਮਾਪ  = phase1_ਥਾਂ_ਮਾਪ;
    layout->ਮਿਲਾਓ_ਮੁੱਲਾਂ_off  = ਮਿਲਾਓ_ਮੁੱਲਾਂ_off;
    layout->ਮਿਲਾਓ_indices_off = ਮਿਲਾਓ_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->ਮਿਲਾਓ_elems       = ਮਿਲਾਓ_elems;
    layout->n_ਧਾਗੇ         = n_chunks;
    layout->n_slots           = n_slots;
    ਵਾਪਸ true;
}

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ // HTP_ARGSORT_OPS_H
