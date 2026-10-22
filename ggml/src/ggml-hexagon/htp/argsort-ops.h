#جيڪڏهن_وصف_نه_ٿيل HTP_ARGSORT_OPS_H
#وصف_ڪريو HTP_ARGSORT_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>
#ضمن <string.h>

#ضمن "hex-fastdiv.h"

بنية htp_ترتيب_ڏيو_kernel_params {
    int32_t  n_ڌاڳا;
    int32_t  ڪل_صفون;
    int32_t  صف_شروعات;
    int32_t  صف_پڇاڙي;
    int32_t  ne00;
    int32_t  k;
    int32_t  ترتيب;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_چوٽي_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_سائيز;
    int32_t  phase1_جڳهه_سائيز;
    int32_t  ملايو_قدرون_off;
    int32_t  ملايو_indices_off;
    int32_t  ملايو_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

بنية htp_ترتيب_ڏيو_vtcm_layout {
    size_t   ڪل_بائيٽون;
    size_t   phase1_جڳهه_سائيز;
    size_t   ملايو_قدرون_off;
    size_t   ملايو_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t ملايو_elems;
    uint32_t n_ڌاڳا;
    uint32_t n_slots;
};

ساكن مضمن bool htp_ترتيب_ڏيو_solve_layout(
    بنية htp_ترتيب_ڏيو_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ڪل_صفون,
    uint32_t k,
    uint32_t n_ڌاڳا_max,
    size_t vtcm_budget,
    bool is_چوٽي_k) {

    memset(layout, 0, حجم(*layout));

    إذا (ڪل_صفون > 1) {
        uint32_t n_ڌاڳا = ڪل_صفون < n_ڌاڳا_max ? ڪل_صفون : n_ڌاڳا_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        طالما (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t قدرون_سائيز  = ((ne00_padded * حجم(عائم)) + 127) & ~127;
        size_t indices_سائيز = ((ne00_padded * حجم(int32_t)) + 127) & ~127;
        size_t spad_في_جڳهه = ((قدرون_سائيز + indices_سائيز) + 255) & ~255;

        uint32_t n_slots = 2;
        إذا (spad_في_جڳهه * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_في_thread = spad_في_جڳهه * n_slots;

        طالما (n_ڌاڳا > 1 && (spad_في_thread * n_ڌاڳا) > vtcm_budget) {
            n_ڌاڳا--;
        }

        size_t ڪل_بائيٽون = spad_في_thread * n_ڌاڳا;
        إذا (ڪل_بائيٽون > vtcm_budget) {
            أرجع false;
        }

        layout->ڪل_بائيٽون      = ڪل_بائيٽون;
        layout->phase1_جڳهه_سائيز = spad_في_جڳهه;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_ڌاڳا        = n_ڌاڳا;
        layout->n_slots          = n_slots;
        أرجع true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    طالما (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    إذا (ne00 > 1024) {
        طالما (n_chunks * 2 <= n_ڌاڳا_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_قدرون_سائيز  = ((chunk_elems * حجم(عائم)) + 127) & ~127;
    size_t phase1_indices_سائيز = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
    size_t phase1_جڳهه_سائيز    = ((phase1_قدرون_سائيز + phase1_indices_سائيز) + 255) & ~255;
    size_t phase1_ڪل_سائيز   = phase1_جڳهه_سائيز * n_chunks;

    size_t ملايو_قدرون_سائيز  = 0;
    size_t ملايو_indices_سائيز = 0;
    size_t ملايو_قدرون_off   = phase1_ڪل_سائيز;
    size_t ملايو_indices_off  = 0;
    uint32_t ملايو_elems      = 0;

    إذا (n_chunks > 1) {
        إذا (is_چوٽي_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t ڪل_candidates = n_chunks * local_k;
            uint32_t ملايو_n_vec = (ڪل_candidates + 31) / 32;
            uint32_t ملايو_n_vec_pow2 = 1;
            طالما (ملايو_n_vec_pow2 < ملايو_n_vec) ملايو_n_vec_pow2 <<= 1;
            ملايو_elems = ملايو_n_vec_pow2 * 32;
        } وإلا {
            ملايو_elems = n_vec_pow2 * 32;
        }
        ملايو_قدرون_سائيز  = ((ملايو_elems * حجم(عائم)) + 127) & ~127;
        ملايو_indices_سائيز = ((ملايو_elems * حجم(int32_t)) + 127) & ~127;
        ملايو_indices_off  = ملايو_قدرون_off + ملايو_قدرون_سائيز;
    }

    size_t ڪل_بائيٽون = phase1_ڪل_سائيز + ملايو_قدرون_سائيز + ملايو_indices_سائيز;

    إذا (ڪل_بائيٽون > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_قدرون_سائيز  = ((chunk_elems * حجم(عائم)) + 127) & ~127;
        phase1_indices_سائيز = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
        phase1_جڳهه_سائيز    = ((phase1_قدرون_سائيز + phase1_indices_سائيز) + 255) & ~255;
        phase1_ڪل_سائيز   = phase1_جڳهه_سائيز;
        ملايو_قدرون_سائيز   = 0;
        ملايو_indices_سائيز  = 0;
        ملايو_قدرون_off    = phase1_ڪل_سائيز;
        ملايو_indices_off   = 0;
        ملايو_elems         = 0;
        ڪل_بائيٽون         = phase1_ڪل_سائيز;
    }

    إذا (ڪل_بائيٽون > vtcm_budget) {
        أرجع false;
    }

    uint32_t n_slots = 1;
    إذا (n_chunks == 1 && phase1_جڳهه_سائيز * 2 <= vtcm_budget) {
        n_slots = 2;
        ڪل_بائيٽون = phase1_جڳهه_سائيز * 2;
    }

    layout->ڪل_بائيٽون       = ڪل_بائيٽون;
    layout->phase1_جڳهه_سائيز  = phase1_جڳهه_سائيز;
    layout->ملايو_قدرون_off  = ملايو_قدرون_off;
    layout->ملايو_indices_off = ملايو_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->ملايو_elems       = ملايو_elems;
    layout->n_ڌاڳا         = n_chunks;
    layout->n_slots           = n_slots;
    أرجع true;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_ترتيب_ڏيو_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_ترتيب_ڏيو_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

#جيڪڏهن_پورو // HTP_ARGSORT_OPS_H
