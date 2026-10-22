#ifndef HTP_ARGSORT_OPS_H
#define HTP_ARGSORT_OPS_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "hex-fastdiv.h"

संरचना htp_sort_kernel_params {
    int32_t  n_threads;
    int32_t  total_rows;
    int32_t  row_start;
    int32_t  row_end;
    int32_t  ne00;
    int32_t  k;
    int32_t  order;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_शिखरम्_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_size;
    int32_t  phase1_slot_size;
    int32_t  merge_मानानि_off;
    int32_t  merge_indices_off;
    int32_t  merge_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

संरचना htp_sort_vtcm_layout {
    size_t   total_bytes;
    size_t   phase1_slot_size;
    size_t   merge_मानानि_off;
    size_t   merge_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t merge_elems;
    uint32_t n_threads;
    uint32_t n_slots;
};

स्थिर अन्तःस्थ bool htp_sort_solve_layout(
    संरचना htp_sort_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t total_rows,
    uint32_t k,
    uint32_t n_threads_max,
    size_t vtcm_budget,
    bool is_शिखरम्_k) {

    memset(layout, 0, परिमाण(*layout));

    सचे (total_rows > 1) {
        uint32_t n_threads = total_rows < n_threads_max ? total_rows : n_threads_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        याव (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t मानानि_size  = ((ne00_padded * परिमाण(भिन्न)) + 127) & ~127;
        size_t indices_size = ((ne00_padded * परिमाण(int32_t)) + 127) & ~127;
        size_t spad_per_slot = ((मानानि_size + indices_size) + 255) & ~255;

        uint32_t n_slots = 2;
        सचे (spad_per_slot * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_per_thread = spad_per_slot * n_slots;

        याव (n_threads > 1 && (spad_per_thread * n_threads) > vtcm_budget) {
            n_threads--;
        }

        size_t total_bytes = spad_per_thread * n_threads;
        सचे (total_bytes > vtcm_budget) {
            पटिनिवत्त false;
        }

        layout->total_bytes      = total_bytes;
        layout->phase1_slot_size = spad_per_slot;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_threads        = n_threads;
        layout->n_slots          = n_slots;
        पटिनिवत्त true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    याव (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    सचे (ne00 > 1024) {
        याव (n_chunks * 2 <= n_threads_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_मानानि_size  = ((chunk_elems * परिमाण(भिन्न)) + 127) & ~127;
    size_t phase1_indices_size = ((chunk_elems * परिमाण(int32_t)) + 127) & ~127;
    size_t phase1_slot_size    = ((phase1_मानानि_size + phase1_indices_size) + 255) & ~255;
    size_t phase1_total_size   = phase1_slot_size * n_chunks;

    size_t merge_मानानि_size  = 0;
    size_t merge_indices_size = 0;
    size_t merge_मानानि_off   = phase1_total_size;
    size_t merge_indices_off  = 0;
    uint32_t merge_elems      = 0;

    सचे (n_chunks > 1) {
        सचे (is_शिखरम्_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t total_candidates = n_chunks * local_k;
            uint32_t merge_n_vec = (total_candidates + 31) / 32;
            uint32_t merge_n_vec_pow2 = 1;
            याव (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
            merge_elems = merge_n_vec_pow2 * 32;
        } अञ्ञथा {
            merge_elems = n_vec_pow2 * 32;
        }
        merge_मानानि_size  = ((merge_elems * परिमाण(भिन्न)) + 127) & ~127;
        merge_indices_size = ((merge_elems * परिमाण(int32_t)) + 127) & ~127;
        merge_indices_off  = merge_मानानि_off + merge_मानानि_size;
    }

    size_t total_bytes = phase1_total_size + merge_मानानि_size + merge_indices_size;

    सचे (total_bytes > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_मानानि_size  = ((chunk_elems * परिमाण(भिन्न)) + 127) & ~127;
        phase1_indices_size = ((chunk_elems * परिमाण(int32_t)) + 127) & ~127;
        phase1_slot_size    = ((phase1_मानानि_size + phase1_indices_size) + 255) & ~255;
        phase1_total_size   = phase1_slot_size;
        merge_मानानि_size   = 0;
        merge_indices_size  = 0;
        merge_मानानि_off    = phase1_total_size;
        merge_indices_off   = 0;
        merge_elems         = 0;
        total_bytes         = phase1_total_size;
    }

    सचे (total_bytes > vtcm_budget) {
        पटिनिवत्त false;
    }

    uint32_t n_slots = 1;
    सचे (n_chunks == 1 && phase1_slot_size * 2 <= vtcm_budget) {
        n_slots = 2;
        total_bytes = phase1_slot_size * 2;
    }

    layout->total_bytes       = total_bytes;
    layout->phase1_slot_size  = phase1_slot_size;
    layout->merge_मानानि_off  = merge_मानानि_off;
    layout->merge_indices_off = merge_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->merge_elems       = merge_elems;
    layout->n_threads         = n_chunks;
    layout->n_slots           = n_slots;
    पटिनिवत्त true;
}

#सचे defined(__cplusplus)
static_assert(परिमाण(संरचना htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#अञ्ञथा
_Static_assert(परिमाण(संरचना htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_ARGSORT_OPS_H
