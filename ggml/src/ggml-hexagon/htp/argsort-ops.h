#ifndef HTP_ARGSORT_OPS_H
#define HTP_ARGSORT_OPS_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"

ಸಂರಚನಾ htp_sort_kernel_params {
    int32_t  n_threads;
    int32_t  total_rows;
    int32_t  row_start;
    int32_t  row_end;
    int32_t  ne00;
    int32_t  k;
    int32_t  order;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_ತುದಿ_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_size;
    int32_t  phase1_slot_size;
    int32_t  merge_ಮೌಲ್ಯಗಳು_off;
    int32_t  merge_indices_off;
    int32_t  merge_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

ಸಂರಚನಾ htp_sort_vtcm_layout {
    size_t   total_bytes;
    size_t   phase1_slot_size;
    size_t   merge_ಮೌಲ್ಯಗಳು_off;
    size_t   merge_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t merge_elems;
    uint32_t n_threads;
    uint32_t n_slots;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_sort_solve_layout(
    ಸಂರಚನಾ htp_sort_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t total_rows,
    uint32_t k,
    uint32_t n_threads_max,
    size_t vtcm_budget,
    bool is_ತುದಿ_k) {

    memset(layout, 0, ಪರಿಮಾಣ(*layout));

    ಯದಿ (total_rows > 1) {
        uint32_t n_threads = total_rows < n_threads_max ? total_rows : n_threads_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        ಯಾವತ್ (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t ಮೌಲ್ಯಗಳು_size  = ((ne00_padded * ಪರಿಮಾಣ(ಭಿನ್ನ)) + 127) & ~127;
        size_t indices_size = ((ne00_padded * ಪರಿಮಾಣ(int32_t)) + 127) & ~127;
        size_t spad_per_slot = ((ಮೌಲ್ಯಗಳು_size + indices_size) + 255) & ~255;

        uint32_t n_slots = 2;
        ಯದಿ (spad_per_slot * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_per_thread = spad_per_slot * n_slots;

        ಯಾವತ್ (n_threads > 1 && (spad_per_thread * n_threads) > vtcm_budget) {
            n_threads--;
        }

        size_t total_bytes = spad_per_thread * n_threads;
        ಯದಿ (total_bytes > vtcm_budget) {
            ಪ್ರತ್ಯಾಗಮ false;
        }

        layout->total_bytes      = total_bytes;
        layout->phase1_slot_size = spad_per_slot;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_threads        = n_threads;
        layout->n_slots          = n_slots;
        ಪ್ರತ್ಯಾಗಮ true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    ಯಾವತ್ (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    ಯದಿ (ne00 > 1024) {
        ಯಾವತ್ (n_chunks * 2 <= n_threads_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_ಮೌಲ್ಯಗಳು_size  = ((chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ)) + 127) & ~127;
    size_t phase1_indices_size = ((chunk_elems * ಪರಿಮಾಣ(int32_t)) + 127) & ~127;
    size_t phase1_slot_size    = ((phase1_ಮೌಲ್ಯಗಳು_size + phase1_indices_size) + 255) & ~255;
    size_t phase1_total_size   = phase1_slot_size * n_chunks;

    size_t merge_ಮೌಲ್ಯಗಳು_size  = 0;
    size_t merge_indices_size = 0;
    size_t merge_ಮೌಲ್ಯಗಳು_off   = phase1_total_size;
    size_t merge_indices_off  = 0;
    uint32_t merge_elems      = 0;

    ಯದಿ (n_chunks > 1) {
        ಯದಿ (is_ತುದಿ_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t total_candidates = n_chunks * local_k;
            uint32_t merge_n_vec = (total_candidates + 31) / 32;
            uint32_t merge_n_vec_pow2 = 1;
            ಯಾವತ್ (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
            merge_elems = merge_n_vec_pow2 * 32;
        } ಅನ್ಯಥಾ {
            merge_elems = n_vec_pow2 * 32;
        }
        merge_ಮೌಲ್ಯಗಳು_size  = ((merge_elems * ಪರಿಮಾಣ(ಭಿನ್ನ)) + 127) & ~127;
        merge_indices_size = ((merge_elems * ಪರಿಮಾಣ(int32_t)) + 127) & ~127;
        merge_indices_off  = merge_ಮೌಲ್ಯಗಳು_off + merge_ಮೌಲ್ಯಗಳು_size;
    }

    size_t total_bytes = phase1_total_size + merge_ಮೌಲ್ಯಗಳು_size + merge_indices_size;

    ಯದಿ (total_bytes > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_ಮೌಲ್ಯಗಳು_size  = ((chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ)) + 127) & ~127;
        phase1_indices_size = ((chunk_elems * ಪರಿಮಾಣ(int32_t)) + 127) & ~127;
        phase1_slot_size    = ((phase1_ಮೌಲ್ಯಗಳು_size + phase1_indices_size) + 255) & ~255;
        phase1_total_size   = phase1_slot_size;
        merge_ಮೌಲ್ಯಗಳು_size   = 0;
        merge_indices_size  = 0;
        merge_ಮೌಲ್ಯಗಳು_off    = phase1_total_size;
        merge_indices_off   = 0;
        merge_elems         = 0;
        total_bytes         = phase1_total_size;
    }

    ಯದಿ (total_bytes > vtcm_budget) {
        ಪ್ರತ್ಯಾಗಮ false;
    }

    uint32_t n_slots = 1;
    ಯದಿ (n_chunks == 1 && phase1_slot_size * 2 <= vtcm_budget) {
        n_slots = 2;
        total_bytes = phase1_slot_size * 2;
    }

    layout->total_bytes       = total_bytes;
    layout->phase1_slot_size  = phase1_slot_size;
    layout->merge_ಮೌಲ್ಯಗಳು_off  = merge_ಮೌಲ್ಯಗಳು_off;
    layout->merge_indices_off = merge_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->merge_elems       = merge_elems;
    layout->n_threads         = n_chunks;
    layout->n_slots           = n_slots;
    ಪ್ರತ್ಯಾಗಮ true;
}

#ಯದಿ defined(__cplusplus)
static_assert(ಪರಿಮಾಣ(ಸಂರಚನಾ htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#ಅನ್ಯಥಾ
_Static_assert(ಪರಿಮಾಣ(ಸಂರಚನಾ htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_ARGSORT_OPS_H
