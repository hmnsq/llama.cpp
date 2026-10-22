#اگر_تعریف_نشده HTP_ARGSORT_OPS_H
#تعریف_کن HTP_ARGSORT_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>
#ضمن <string.h>

#ضمن "hex-fastdiv.h"

بنية htp_مرتب_سازی_kernel_params {
    int32_t  n_نخها;
    int32_t  مجموع_ردیفها;
    int32_t  ردیف_شروع;
    int32_t  ردیف_انتها;
    int32_t  ne00;
    int32_t  k;
    int32_t  ترتیب;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_بالاترین_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_اندازه;
    int32_t  phase1_جایگاه_اندازه;
    int32_t  ادغام_مقادیر_off;
    int32_t  ادغام_indices_off;
    int32_t  ادغام_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

بنية htp_مرتب_سازی_vtcm_layout {
    size_t   مجموع_بایتها;
    size_t   phase1_جایگاه_اندازه;
    size_t   ادغام_مقادیر_off;
    size_t   ادغام_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t ادغام_elems;
    uint32_t n_نخها;
    uint32_t n_slots;
};

ساكن مضمن bool htp_مرتب_سازی_solve_layout(
    بنية htp_مرتب_سازی_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t مجموع_ردیفها,
    uint32_t k,
    uint32_t n_نخها_max,
    size_t vtcm_budget,
    bool is_بالاترین_k) {

    memset(layout, 0, حجم(*layout));

    إذا (مجموع_ردیفها > 1) {
        uint32_t n_نخها = مجموع_ردیفها < n_نخها_max ? مجموع_ردیفها : n_نخها_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        طالما (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t مقادیر_اندازه  = ((ne00_padded * حجم(عائم)) + 127) & ~127;
        size_t indices_اندازه = ((ne00_padded * حجم(int32_t)) + 127) & ~127;
        size_t spad_در_هر_جایگاه = ((مقادیر_اندازه + indices_اندازه) + 255) & ~255;

        uint32_t n_slots = 2;
        إذا (spad_در_هر_جایگاه * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_در_هر_thread = spad_در_هر_جایگاه * n_slots;

        طالما (n_نخها > 1 && (spad_در_هر_thread * n_نخها) > vtcm_budget) {
            n_نخها--;
        }

        size_t مجموع_بایتها = spad_در_هر_thread * n_نخها;
        إذا (مجموع_بایتها > vtcm_budget) {
            أرجع false;
        }

        layout->مجموع_بایتها      = مجموع_بایتها;
        layout->phase1_جایگاه_اندازه = spad_در_هر_جایگاه;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_نخها        = n_نخها;
        layout->n_slots          = n_slots;
        أرجع true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    طالما (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    إذا (ne00 > 1024) {
        طالما (n_chunks * 2 <= n_نخها_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_مقادیر_اندازه  = ((chunk_elems * حجم(عائم)) + 127) & ~127;
    size_t phase1_indices_اندازه = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
    size_t phase1_جایگاه_اندازه    = ((phase1_مقادیر_اندازه + phase1_indices_اندازه) + 255) & ~255;
    size_t phase1_مجموع_اندازه   = phase1_جایگاه_اندازه * n_chunks;

    size_t ادغام_مقادیر_اندازه  = 0;
    size_t ادغام_indices_اندازه = 0;
    size_t ادغام_مقادیر_off   = phase1_مجموع_اندازه;
    size_t ادغام_indices_off  = 0;
    uint32_t ادغام_elems      = 0;

    إذا (n_chunks > 1) {
        إذا (is_بالاترین_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t مجموع_candidates = n_chunks * local_k;
            uint32_t ادغام_n_vec = (مجموع_candidates + 31) / 32;
            uint32_t ادغام_n_vec_pow2 = 1;
            طالما (ادغام_n_vec_pow2 < ادغام_n_vec) ادغام_n_vec_pow2 <<= 1;
            ادغام_elems = ادغام_n_vec_pow2 * 32;
        } وإلا {
            ادغام_elems = n_vec_pow2 * 32;
        }
        ادغام_مقادیر_اندازه  = ((ادغام_elems * حجم(عائم)) + 127) & ~127;
        ادغام_indices_اندازه = ((ادغام_elems * حجم(int32_t)) + 127) & ~127;
        ادغام_indices_off  = ادغام_مقادیر_off + ادغام_مقادیر_اندازه;
    }

    size_t مجموع_بایتها = phase1_مجموع_اندازه + ادغام_مقادیر_اندازه + ادغام_indices_اندازه;

    إذا (مجموع_بایتها > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_مقادیر_اندازه  = ((chunk_elems * حجم(عائم)) + 127) & ~127;
        phase1_indices_اندازه = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
        phase1_جایگاه_اندازه    = ((phase1_مقادیر_اندازه + phase1_indices_اندازه) + 255) & ~255;
        phase1_مجموع_اندازه   = phase1_جایگاه_اندازه;
        ادغام_مقادیر_اندازه   = 0;
        ادغام_indices_اندازه  = 0;
        ادغام_مقادیر_off    = phase1_مجموع_اندازه;
        ادغام_indices_off   = 0;
        ادغام_elems         = 0;
        مجموع_بایتها         = phase1_مجموع_اندازه;
    }

    إذا (مجموع_بایتها > vtcm_budget) {
        أرجع false;
    }

    uint32_t n_slots = 1;
    إذا (n_chunks == 1 && phase1_جایگاه_اندازه * 2 <= vtcm_budget) {
        n_slots = 2;
        مجموع_بایتها = phase1_جایگاه_اندازه * 2;
    }

    layout->مجموع_بایتها       = مجموع_بایتها;
    layout->phase1_جایگاه_اندازه  = phase1_جایگاه_اندازه;
    layout->ادغام_مقادیر_off  = ادغام_مقادیر_off;
    layout->ادغام_indices_off = ادغام_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->ادغام_elems       = ادغام_elems;
    layout->n_نخها         = n_chunks;
    layout->n_slots           = n_slots;
    أرجع true;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_مرتب_سازی_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_مرتب_سازی_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#پایان_اگر

#پایان_اگر // HTP_ARGSORT_OPS_H
