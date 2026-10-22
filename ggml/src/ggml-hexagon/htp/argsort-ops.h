#ifndef HTP_ARGSORT_OPS_H
#define HTP_ARGSORT_OPS_H

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <string.h>

#অন্তর্ভাব "hex-fastdiv.h"

কাঠামো htp_sort_kernel_params {
    int32_t  n_threads;
    int32_t  যোগ_rows;
    int32_t  row_start;
    int32_t  row_end;
    int32_t  ne00;
    int32_t  k;
    int32_t  order;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_শীর্ষ_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_আকার;
    int32_t  phase1_slot_আকার;
    int32_t  merge_মানসমূহ_off;
    int32_t  merge_indices_off;
    int32_t  merge_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

কাঠামো htp_sort_vtcm_layout {
    size_t   যোগ_bytes;
    size_t   phase1_slot_আকার;
    size_t   merge_মানসমূহ_off;
    size_t   merge_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t merge_elems;
    uint32_t n_threads;
    uint32_t n_slots;
};

স্থির অন্তঃরেখ bool htp_sort_solve_layout(
    কাঠামো htp_sort_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t যোগ_rows,
    uint32_t k,
    uint32_t n_threads_max,
    size_t vtcm_budget,
    bool is_শীর্ষ_k) {

    memset(layout, 0, আকার(*layout));

    যদি (যোগ_rows > 1) {
        uint32_t n_threads = যোগ_rows < n_threads_max ? যোগ_rows : n_threads_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        যখন (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t মানসমূহ_আকার  = ((ne00_padded * আকার(দশমিক)) + 127) & ~127;
        size_t indices_আকার = ((ne00_padded * আকার(int32_t)) + 127) & ~127;
        size_t spad_per_slot = ((মানসমূহ_আকার + indices_আকার) + 255) & ~255;

        uint32_t n_slots = 2;
        যদি (spad_per_slot * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_per_thread = spad_per_slot * n_slots;

        যখন (n_threads > 1 && (spad_per_thread * n_threads) > vtcm_budget) {
            n_threads--;
        }

        size_t যোগ_bytes = spad_per_thread * n_threads;
        যদি (যোগ_bytes > vtcm_budget) {
            ফেরত false;
        }

        layout->যোগ_bytes      = যোগ_bytes;
        layout->phase1_slot_আকার = spad_per_slot;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_threads        = n_threads;
        layout->n_slots          = n_slots;
        ফেরত true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    যখন (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    যদি (ne00 > 1024) {
        যখন (n_chunks * 2 <= n_threads_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_মানসমূহ_আকার  = ((chunk_elems * আকার(দশমিক)) + 127) & ~127;
    size_t phase1_indices_আকার = ((chunk_elems * আকার(int32_t)) + 127) & ~127;
    size_t phase1_slot_আকার    = ((phase1_মানসমূহ_আকার + phase1_indices_আকার) + 255) & ~255;
    size_t phase1_যোগ_আকার   = phase1_slot_আকার * n_chunks;

    size_t merge_মানসমূহ_আকার  = 0;
    size_t merge_indices_আকার = 0;
    size_t merge_মানসমূহ_off   = phase1_যোগ_আকার;
    size_t merge_indices_off  = 0;
    uint32_t merge_elems      = 0;

    যদি (n_chunks > 1) {
        যদি (is_শীর্ষ_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t যোগ_candidates = n_chunks * local_k;
            uint32_t merge_n_vec = (যোগ_candidates + 31) / 32;
            uint32_t merge_n_vec_pow2 = 1;
            যখন (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
            merge_elems = merge_n_vec_pow2 * 32;
        } অন্যথা {
            merge_elems = n_vec_pow2 * 32;
        }
        merge_মানসমূহ_আকার  = ((merge_elems * আকার(দশমিক)) + 127) & ~127;
        merge_indices_আকার = ((merge_elems * আকার(int32_t)) + 127) & ~127;
        merge_indices_off  = merge_মানসমূহ_off + merge_মানসমূহ_আকার;
    }

    size_t যোগ_bytes = phase1_যোগ_আকার + merge_মানসমূহ_আকার + merge_indices_আকার;

    যদি (যোগ_bytes > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_মানসমূহ_আকার  = ((chunk_elems * আকার(দশমিক)) + 127) & ~127;
        phase1_indices_আকার = ((chunk_elems * আকার(int32_t)) + 127) & ~127;
        phase1_slot_আকার    = ((phase1_মানসমূহ_আকার + phase1_indices_আকার) + 255) & ~255;
        phase1_যোগ_আকার   = phase1_slot_আকার;
        merge_মানসমূহ_আকার   = 0;
        merge_indices_আকার  = 0;
        merge_মানসমূহ_off    = phase1_যোগ_আকার;
        merge_indices_off   = 0;
        merge_elems         = 0;
        যোগ_bytes         = phase1_যোগ_আকার;
    }

    যদি (যোগ_bytes > vtcm_budget) {
        ফেরত false;
    }

    uint32_t n_slots = 1;
    যদি (n_chunks == 1 && phase1_slot_আকার * 2 <= vtcm_budget) {
        n_slots = 2;
        যোগ_bytes = phase1_slot_আকার * 2;
    }

    layout->যোগ_bytes       = যোগ_bytes;
    layout->phase1_slot_আকার  = phase1_slot_আকার;
    layout->merge_মানসমূহ_off  = merge_মানসমূহ_off;
    layout->merge_indices_off = merge_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->merge_elems       = merge_elems;
    layout->n_threads         = n_chunks;
    layout->n_slots           = n_slots;
    ফেরত true;
}

#যদি defined(__cplusplus)
static_assert(আকার(কাঠামো htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(আকার(কাঠামো htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_ARGSORT_OPS_H
