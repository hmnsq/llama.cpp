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
    int32_t  is_शीर्ष_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_size;
    int32_t  phase1_slot_size;
    int32_t  merge_मानहरू_off;
    int32_t  merge_indices_off;
    int32_t  merge_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

संरचना htp_sort_vtcm_layout {
    माप_प्रकार   total_bytes;
    माप_प्रकार   phase1_slot_size;
    माप_प्रकार   merge_मानहरू_off;
    माप_प्रकार   merge_indices_off;
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
    माप_प्रकार vtcm_budget,
    bool is_शीर्ष_k) {

    स_रखो(layout, 0, परिमाण(*layout));

    यदि (total_rows > 1) {
        uint32_t n_threads = total_rows < n_threads_max ? total_rows : n_threads_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        जबसम्म (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        माप_प्रकार मानहरू_size  = ((ne00_padded * परिमाण(भिन्न)) + 127) & ~127;
        माप_प्रकार indices_size = ((ne00_padded * परिमाण(int32_t)) + 127) & ~127;
        माप_प्रकार spad_per_slot = ((मानहरू_size + indices_size) + 255) & ~255;

        uint32_t n_slots = 2;
        यदि (spad_per_slot * 2 > vtcm_budget) {
            n_slots = 1;
        }

        माप_प्रकार spad_per_thread = spad_per_slot * n_slots;

        जबसम्म (n_threads > 1 && (spad_per_thread * n_threads) > vtcm_budget) {
            n_threads--;
        }

        माप_प्रकार total_bytes = spad_per_thread * n_threads;
        यदि (total_bytes > vtcm_budget) {
            फर्काऊ false;
        }

        layout->total_bytes      = total_bytes;
        layout->phase1_slot_size = spad_per_slot;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_threads        = n_threads;
        layout->n_slots          = n_slots;
        फर्काऊ true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    जबसम्म (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    यदि (ne00 > 1024) {
        जबसम्म (n_chunks * 2 <= n_threads_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    माप_प्रकार phase1_मानहरू_size  = ((chunk_elems * परिमाण(भिन्न)) + 127) & ~127;
    माप_प्रकार phase1_indices_size = ((chunk_elems * परिमाण(int32_t)) + 127) & ~127;
    माप_प्रकार phase1_slot_size    = ((phase1_मानहरू_size + phase1_indices_size) + 255) & ~255;
    माप_प्रकार phase1_total_size   = phase1_slot_size * n_chunks;

    माप_प्रकार merge_मानहरू_size  = 0;
    माप_प्रकार merge_indices_size = 0;
    माप_प्रकार merge_मानहरू_off   = phase1_total_size;
    माप_प्रकार merge_indices_off  = 0;
    uint32_t merge_elems      = 0;

    यदि (n_chunks > 1) {
        यदि (is_शीर्ष_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t total_candidates = n_chunks * local_k;
            uint32_t merge_n_vec = (total_candidates + 31) / 32;
            uint32_t merge_n_vec_pow2 = 1;
            जबसम्म (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
            merge_elems = merge_n_vec_pow2 * 32;
        } अन्यथा {
            merge_elems = n_vec_pow2 * 32;
        }
        merge_मानहरू_size  = ((merge_elems * परिमाण(भिन्न)) + 127) & ~127;
        merge_indices_size = ((merge_elems * परिमाण(int32_t)) + 127) & ~127;
        merge_indices_off  = merge_मानहरू_off + merge_मानहरू_size;
    }

    माप_प्रकार total_bytes = phase1_total_size + merge_मानहरू_size + merge_indices_size;

    यदि (total_bytes > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_मानहरू_size  = ((chunk_elems * परिमाण(भिन्न)) + 127) & ~127;
        phase1_indices_size = ((chunk_elems * परिमाण(int32_t)) + 127) & ~127;
        phase1_slot_size    = ((phase1_मानहरू_size + phase1_indices_size) + 255) & ~255;
        phase1_total_size   = phase1_slot_size;
        merge_मानहरू_size   = 0;
        merge_indices_size  = 0;
        merge_मानहरू_off    = phase1_total_size;
        merge_indices_off   = 0;
        merge_elems         = 0;
        total_bytes         = phase1_total_size;
    }

    यदि (total_bytes > vtcm_budget) {
        फर्काऊ false;
    }

    uint32_t n_slots = 1;
    यदि (n_chunks == 1 && phase1_slot_size * 2 <= vtcm_budget) {
        n_slots = 2;
        total_bytes = phase1_slot_size * 2;
    }

    layout->total_bytes       = total_bytes;
    layout->phase1_slot_size  = phase1_slot_size;
    layout->merge_मानहरू_off  = merge_मानहरू_off;
    layout->merge_indices_off = merge_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->merge_elems       = merge_elems;
    layout->n_threads         = n_chunks;
    layout->n_slots           = n_slots;
    फर्काऊ true;
}

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_ARGSORT_OPS_H
