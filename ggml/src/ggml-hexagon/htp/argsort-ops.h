#अगर_अघोषित HTP_ARGSORT_OPS_H
#घोषणा HTP_ARGSORT_OPS_H

#समावेश <stdint.h>
#समावेश <stddef.h>
#समावेश <stdbool.h>
#समावेश <string.h>

#समावेश "hex-fastdiv.h"

काष्ठा htp_sort_kernel_params {
    int32_t  n_threads;
    int32_t  योग_rows;
    int32_t  row_start;
    int32_t  row_end;
    int32_t  ne00;
    int32_t  k;
    int32_t  order;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_शीर्ष_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_आकार;
    int32_t  phase1_slot_आकार;
    int32_t  merge_मूल्यावली_off;
    int32_t  merge_indices_off;
    int32_t  merge_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

काष्ठा htp_sort_vtcm_layout {
    माप_प्रकार   योग_bytes;
    माप_प्रकार   phase1_slot_आकार;
    माप_प्रकार   merge_मूल्यावली_off;
    माप_प्रकार   merge_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t merge_elems;
    uint32_t n_threads;
    uint32_t n_slots;
};

अटल अंतरभूत bool htp_sort_solve_layout(
    काष्ठा htp_sort_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t योग_rows,
    uint32_t k,
    uint32_t n_threads_max,
    माप_प्रकार vtcm_budget,
    bool is_शीर्ष_k) {

    स_रखो(layout, 0, माप(*layout));

    अगर (योग_rows > 1) {
        uint32_t n_threads = योग_rows < n_threads_max ? योग_rows : n_threads_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        जबतक (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        माप_प्रकार मूल्यावली_आकार  = ((ne00_padded * माप(भग्न)) + 127) & ~127;
        माप_प्रकार indices_आकार = ((ne00_padded * माप(int32_t)) + 127) & ~127;
        माप_प्रकार spad_per_slot = ((मूल्यावली_आकार + indices_आकार) + 255) & ~255;

        uint32_t n_slots = 2;
        अगर (spad_per_slot * 2 > vtcm_budget) {
            n_slots = 1;
        }

        माप_प्रकार spad_per_thread = spad_per_slot * n_slots;

        जबतक (n_threads > 1 && (spad_per_thread * n_threads) > vtcm_budget) {
            n_threads--;
        }

        माप_प्रकार योग_bytes = spad_per_thread * n_threads;
        अगर (योग_bytes > vtcm_budget) {
            वापस false;
        }

        layout->योग_bytes      = योग_bytes;
        layout->phase1_slot_आकार = spad_per_slot;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_threads        = n_threads;
        layout->n_slots          = n_slots;
        वापस true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    जबतक (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    अगर (ne00 > 1024) {
        जबतक (n_chunks * 2 <= n_threads_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    माप_प्रकार phase1_मूल्यावली_आकार  = ((chunk_elems * माप(भग्न)) + 127) & ~127;
    माप_प्रकार phase1_indices_आकार = ((chunk_elems * माप(int32_t)) + 127) & ~127;
    माप_प्रकार phase1_slot_आकार    = ((phase1_मूल्यावली_आकार + phase1_indices_आकार) + 255) & ~255;
    माप_प्रकार phase1_योग_आकार   = phase1_slot_आकार * n_chunks;

    माप_प्रकार merge_मूल्यावली_आकार  = 0;
    माप_प्रकार merge_indices_आकार = 0;
    माप_प्रकार merge_मूल्यावली_off   = phase1_योग_आकार;
    माप_प्रकार merge_indices_off  = 0;
    uint32_t merge_elems      = 0;

    अगर (n_chunks > 1) {
        अगर (is_शीर्ष_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t योग_candidates = n_chunks * local_k;
            uint32_t merge_n_vec = (योग_candidates + 31) / 32;
            uint32_t merge_n_vec_pow2 = 1;
            जबतक (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
            merge_elems = merge_n_vec_pow2 * 32;
        } अन्यथा {
            merge_elems = n_vec_pow2 * 32;
        }
        merge_मूल्यावली_आकार  = ((merge_elems * माप(भग्न)) + 127) & ~127;
        merge_indices_आकार = ((merge_elems * माप(int32_t)) + 127) & ~127;
        merge_indices_off  = merge_मूल्यावली_off + merge_मूल्यावली_आकार;
    }

    माप_प्रकार योग_bytes = phase1_योग_आकार + merge_मूल्यावली_आकार + merge_indices_आकार;

    अगर (योग_bytes > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_मूल्यावली_आकार  = ((chunk_elems * माप(भग्न)) + 127) & ~127;
        phase1_indices_आकार = ((chunk_elems * माप(int32_t)) + 127) & ~127;
        phase1_slot_आकार    = ((phase1_मूल्यावली_आकार + phase1_indices_आकार) + 255) & ~255;
        phase1_योग_आकार   = phase1_slot_आकार;
        merge_मूल्यावली_आकार   = 0;
        merge_indices_आकार  = 0;
        merge_मूल्यावली_off    = phase1_योग_आकार;
        merge_indices_off   = 0;
        merge_elems         = 0;
        योग_bytes         = phase1_योग_आकार;
    }

    अगर (योग_bytes > vtcm_budget) {
        वापस false;
    }

    uint32_t n_slots = 1;
    अगर (n_chunks == 1 && phase1_slot_आकार * 2 <= vtcm_budget) {
        n_slots = 2;
        योग_bytes = phase1_slot_आकार * 2;
    }

    layout->योग_bytes       = योग_bytes;
    layout->phase1_slot_आकार  = phase1_slot_आकार;
    layout->merge_मूल्यावली_off  = merge_मूल्यावली_off;
    layout->merge_indices_off = merge_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->merge_elems       = merge_elems;
    layout->n_threads         = n_chunks;
    layout->n_slots           = n_slots;
    वापस true;
}

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

#पूर्ण_अगर // HTP_ARGSORT_OPS_H
