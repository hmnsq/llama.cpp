#ifndef HTP_ARGSORT_OPS_H
#define HTP_ARGSORT_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>
#ضمن <string.h>

#ضمن "hex-fastdiv.h"

بنية htp_sort_kernel_params {
    int32_t  n_threads;
    int32_t  total_rows;
    int32_t  row_start;
    int32_t  row_end;
    int32_t  ne00;
    int32_t  k;
    int32_t  order;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_چوققا_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_size;
    int32_t  phase1_slot_size;
    int32_t  merge_قىممەتلەر_off;
    int32_t  merge_indices_off;
    int32_t  merge_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

بنية htp_sort_vtcm_layout {
    size_t   total_bytes;
    size_t   phase1_slot_size;
    size_t   merge_قىممەتلەر_off;
    size_t   merge_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t merge_elems;
    uint32_t n_threads;
    uint32_t n_slots;
};

ساكن مضمن bool htp_sort_solve_layout(
    بنية htp_sort_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t total_rows,
    uint32_t k,
    uint32_t n_threads_max,
    size_t vtcm_budget,
    bool is_چوققا_k) {

    memset(layout, 0, حجم(*layout));

    إذا (total_rows > 1) {
        uint32_t n_threads = total_rows < n_threads_max ? total_rows : n_threads_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        طالما (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t قىممەتلەر_size  = ((ne00_padded * حجم(عائم)) + 127) & ~127;
        size_t indices_size = ((ne00_padded * حجم(int32_t)) + 127) & ~127;
        size_t spad_per_slot = ((قىممەتلەر_size + indices_size) + 255) & ~255;

        uint32_t n_slots = 2;
        إذا (spad_per_slot * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_per_thread = spad_per_slot * n_slots;

        طالما (n_threads > 1 && (spad_per_thread * n_threads) > vtcm_budget) {
            n_threads--;
        }

        size_t total_bytes = spad_per_thread * n_threads;
        إذا (total_bytes > vtcm_budget) {
            أرجع false;
        }

        layout->total_bytes      = total_bytes;
        layout->phase1_slot_size = spad_per_slot;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_threads        = n_threads;
        layout->n_slots          = n_slots;
        أرجع true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    طالما (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    إذا (ne00 > 1024) {
        طالما (n_chunks * 2 <= n_threads_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_قىممەتلەر_size  = ((chunk_elems * حجم(عائم)) + 127) & ~127;
    size_t phase1_indices_size = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
    size_t phase1_slot_size    = ((phase1_قىممەتلەر_size + phase1_indices_size) + 255) & ~255;
    size_t phase1_total_size   = phase1_slot_size * n_chunks;

    size_t merge_قىممەتلەر_size  = 0;
    size_t merge_indices_size = 0;
    size_t merge_قىممەتلەر_off   = phase1_total_size;
    size_t merge_indices_off  = 0;
    uint32_t merge_elems      = 0;

    إذا (n_chunks > 1) {
        إذا (is_چوققا_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t total_candidates = n_chunks * local_k;
            uint32_t merge_n_vec = (total_candidates + 31) / 32;
            uint32_t merge_n_vec_pow2 = 1;
            طالما (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
            merge_elems = merge_n_vec_pow2 * 32;
        } وإلا {
            merge_elems = n_vec_pow2 * 32;
        }
        merge_قىممەتلەر_size  = ((merge_elems * حجم(عائم)) + 127) & ~127;
        merge_indices_size = ((merge_elems * حجم(int32_t)) + 127) & ~127;
        merge_indices_off  = merge_قىممەتلەر_off + merge_قىممەتلەر_size;
    }

    size_t total_bytes = phase1_total_size + merge_قىممەتلەر_size + merge_indices_size;

    إذا (total_bytes > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_قىممەتلەر_size  = ((chunk_elems * حجم(عائم)) + 127) & ~127;
        phase1_indices_size = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
        phase1_slot_size    = ((phase1_قىممەتلەر_size + phase1_indices_size) + 255) & ~255;
        phase1_total_size   = phase1_slot_size;
        merge_قىممەتلەر_size   = 0;
        merge_indices_size  = 0;
        merge_قىممەتلەر_off    = phase1_total_size;
        merge_indices_off   = 0;
        merge_elems         = 0;
        total_bytes         = phase1_total_size;
    }

    إذا (total_bytes > vtcm_budget) {
        أرجع false;
    }

    uint32_t n_slots = 1;
    إذا (n_chunks == 1 && phase1_slot_size * 2 <= vtcm_budget) {
        n_slots = 2;
        total_bytes = phase1_slot_size * 2;
    }

    layout->total_bytes       = total_bytes;
    layout->phase1_slot_size  = phase1_slot_size;
    layout->merge_قىممەتلەر_off  = merge_قىممەتلەر_off;
    layout->merge_indices_off = merge_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->merge_elems       = merge_elems;
    layout->n_threads         = n_chunks;
    layout->n_slots           = n_slots;
    أرجع true;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_ARGSORT_OPS_H
