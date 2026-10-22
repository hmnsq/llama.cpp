#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_ARGSORT_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ARGSORT_OPS_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"

ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params {
    int32_t  n_ಎಳೆಗಳು;
    int32_t  ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    int32_t  ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    int32_t  ಅಡ್ಡಸಾಲು_ಕೊನೆ;
    int32_t  ne00;
    int32_t  k;
    int32_t  ಕ್ರಮ;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_ತುದಿ_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_ಗಾತ್ರ;
    int32_t  phase1_ಸ್ಥಾನ_ಗಾತ್ರ;
    int32_t  ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off;
    int32_t  ವಿಲೀನ_indices_off;
    int32_t  ವಿಲೀನ_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

ರಚನೆ htp_ವಿಂಗಡಿಸು_vtcm_layout {
    size_t   ಒಟ್ಟು_ಬೈಟುಗಳು;
    size_t   phase1_ಸ್ಥಾನ_ಗಾತ್ರ;
    size_t   ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off;
    size_t   ವಿಲೀನ_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t ವಿಲೀನ_elems;
    uint32_t n_ಎಳೆಗಳು;
    uint32_t n_slots;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_ವಿಂಗಡಿಸು_solve_layout(
    ರಚನೆ htp_ವಿಂಗಡಿಸು_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು,
    uint32_t k,
    uint32_t n_ಎಳೆಗಳು_max,
    size_t vtcm_budget,
    bool is_ತುದಿ_k) {

    memset(layout, 0, ಗಾತ್ರ(*layout));

    ಆದರೆ (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು > 1) {
        uint32_t n_ಎಳೆಗಳು = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು < n_ಎಳೆಗಳು_max ? ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು : n_ಎಳೆಗಳು_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        ಆಗ (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t ಮೌಲ್ಯಗಳು_ಗಾತ್ರ  = ((ne00_padded * ಗಾತ್ರ(ದಶಮಾಂಶ)) + 127) & ~127;
        size_t indices_ಗಾತ್ರ = ((ne00_padded * ಗಾತ್ರ(int32_t)) + 127) & ~127;
        size_t spad_ಪ್ರತಿ_ಸ್ಥಾನ = ((ಮೌಲ್ಯಗಳು_ಗಾತ್ರ + indices_ಗಾತ್ರ) + 255) & ~255;

        uint32_t n_slots = 2;
        ಆದರೆ (spad_ಪ್ರತಿ_ಸ್ಥಾನ * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_ಪ್ರತಿ_thread = spad_ಪ್ರತಿ_ಸ್ಥಾನ * n_slots;

        ಆಗ (n_ಎಳೆಗಳು > 1 && (spad_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು) > vtcm_budget) {
            n_ಎಳೆಗಳು--;
        }

        size_t ಒಟ್ಟು_ಬೈಟುಗಳು = spad_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
        ಆದರೆ (ಒಟ್ಟು_ಬೈಟುಗಳು > vtcm_budget) {
            ಹಿಂತಿರುಗಿ false;
        }

        layout->ಒಟ್ಟು_ಬೈಟುಗಳು      = ಒಟ್ಟು_ಬೈಟುಗಳು;
        layout->phase1_ಸ್ಥಾನ_ಗಾತ್ರ = spad_ಪ್ರತಿ_ಸ್ಥಾನ;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_ಎಳೆಗಳು        = n_ಎಳೆಗಳು;
        layout->n_slots          = n_slots;
        ಹಿಂತಿರುಗಿ true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    ಆಗ (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    ಆದರೆ (ne00 > 1024) {
        ಆಗ (n_chunks * 2 <= n_ಎಳೆಗಳು_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_ಮೌಲ್ಯಗಳು_ಗಾತ್ರ  = ((chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ)) + 127) & ~127;
    size_t phase1_indices_ಗಾತ್ರ = ((chunk_elems * ಗಾತ್ರ(int32_t)) + 127) & ~127;
    size_t phase1_ಸ್ಥಾನ_ಗಾತ್ರ    = ((phase1_ಮೌಲ್ಯಗಳು_ಗಾತ್ರ + phase1_indices_ಗಾತ್ರ) + 255) & ~255;
    size_t phase1_ಒಟ್ಟು_ಗಾತ್ರ   = phase1_ಸ್ಥಾನ_ಗಾತ್ರ * n_chunks;

    size_t ವಿಲೀನ_ಮೌಲ್ಯಗಳು_ಗಾತ್ರ  = 0;
    size_t ವಿಲೀನ_indices_ಗಾತ್ರ = 0;
    size_t ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off   = phase1_ಒಟ್ಟು_ಗಾತ್ರ;
    size_t ವಿಲೀನ_indices_off  = 0;
    uint32_t ವಿಲೀನ_elems      = 0;

    ಆದರೆ (n_chunks > 1) {
        ಆದರೆ (is_ತುದಿ_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t ಒಟ್ಟು_candidates = n_chunks * local_k;
            uint32_t ವಿಲೀನ_n_vec = (ಒಟ್ಟು_candidates + 31) / 32;
            uint32_t ವಿಲೀನ_n_vec_pow2 = 1;
            ಆಗ (ವಿಲೀನ_n_vec_pow2 < ವಿಲೀನ_n_vec) ವಿಲೀನ_n_vec_pow2 <<= 1;
            ವಿಲೀನ_elems = ವಿಲೀನ_n_vec_pow2 * 32;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ವಿಲೀನ_elems = n_vec_pow2 * 32;
        }
        ವಿಲೀನ_ಮೌಲ್ಯಗಳು_ಗಾತ್ರ  = ((ವಿಲೀನ_elems * ಗಾತ್ರ(ದಶಮಾಂಶ)) + 127) & ~127;
        ವಿಲೀನ_indices_ಗಾತ್ರ = ((ವಿಲೀನ_elems * ಗಾತ್ರ(int32_t)) + 127) & ~127;
        ವಿಲೀನ_indices_off  = ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off + ವಿಲೀನ_ಮೌಲ್ಯಗಳು_ಗಾತ್ರ;
    }

    size_t ಒಟ್ಟು_ಬೈಟುಗಳು = phase1_ಒಟ್ಟು_ಗಾತ್ರ + ವಿಲೀನ_ಮೌಲ್ಯಗಳು_ಗಾತ್ರ + ವಿಲೀನ_indices_ಗಾತ್ರ;

    ಆದರೆ (ಒಟ್ಟು_ಬೈಟುಗಳು > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_ಮೌಲ್ಯಗಳು_ಗಾತ್ರ  = ((chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ)) + 127) & ~127;
        phase1_indices_ಗಾತ್ರ = ((chunk_elems * ಗಾತ್ರ(int32_t)) + 127) & ~127;
        phase1_ಸ್ಥಾನ_ಗಾತ್ರ    = ((phase1_ಮೌಲ್ಯಗಳು_ಗಾತ್ರ + phase1_indices_ಗಾತ್ರ) + 255) & ~255;
        phase1_ಒಟ್ಟು_ಗಾತ್ರ   = phase1_ಸ್ಥಾನ_ಗಾತ್ರ;
        ವಿಲೀನ_ಮೌಲ್ಯಗಳು_ಗಾತ್ರ   = 0;
        ವಿಲೀನ_indices_ಗಾತ್ರ  = 0;
        ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off    = phase1_ಒಟ್ಟು_ಗಾತ್ರ;
        ವಿಲೀನ_indices_off   = 0;
        ವಿಲೀನ_elems         = 0;
        ಒಟ್ಟು_ಬೈಟುಗಳು         = phase1_ಒಟ್ಟು_ಗಾತ್ರ;
    }

    ಆದರೆ (ಒಟ್ಟು_ಬೈಟುಗಳು > vtcm_budget) {
        ಹಿಂತಿರುಗಿ false;
    }

    uint32_t n_slots = 1;
    ಆದರೆ (n_chunks == 1 && phase1_ಸ್ಥಾನ_ಗಾತ್ರ * 2 <= vtcm_budget) {
        n_slots = 2;
        ಒಟ್ಟು_ಬೈಟುಗಳು = phase1_ಸ್ಥಾನ_ಗಾತ್ರ * 2;
    }

    layout->ಒಟ್ಟು_ಬೈಟುಗಳು       = ಒಟ್ಟು_ಬೈಟುಗಳು;
    layout->phase1_ಸ್ಥಾನ_ಗಾತ್ರ  = phase1_ಸ್ಥಾನ_ಗಾತ್ರ;
    layout->ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off  = ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off;
    layout->ವಿಲೀನ_indices_off = ವಿಲೀನ_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->ವಿಲೀನ_elems       = ವಿಲೀನ_elems;
    layout->n_ಎಳೆಗಳು         = n_chunks;
    layout->n_slots           = n_slots;
    ಹಿಂತಿರುಗಿ true;
}

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ // HTP_ARGSORT_OPS_H
