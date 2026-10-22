#جے_بنا_تعریف HTP_ARGSORT_OPS_H
#تعریف_کرو HTP_ARGSORT_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>
#ضمن <string.h>

#ضمن "hex-fastdiv.h"

بنية htp_ترتیب_دیو_kernel_params {
    int32_t  n_دھاگے;
    int32_t  کل_صفاں;
    int32_t  صف_شروع;
    int32_t  صف_سرا;
    int32_t  ne00;
    int32_t  k;
    int32_t  ترتیب;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_چوٹی_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_ناپ;
    int32_t  phase1_تھاں_ناپ;
    int32_t  ملاؤ_اقدار_off;
    int32_t  ملاؤ_indices_off;
    int32_t  ملاؤ_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

بنية htp_ترتیب_دیو_vtcm_layout {
    size_t   کل_بائٹاں;
    size_t   phase1_تھاں_ناپ;
    size_t   ملاؤ_اقدار_off;
    size_t   ملاؤ_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t ملاؤ_elems;
    uint32_t n_دھاگے;
    uint32_t n_slots;
};

ساكن مضمن bool htp_ترتیب_دیو_solve_layout(
    بنية htp_ترتیب_دیو_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t کل_صفاں,
    uint32_t k,
    uint32_t n_دھاگے_max,
    size_t vtcm_budget,
    bool is_چوٹی_k) {

    memset(layout, 0, حجم(*layout));

    إذا (کل_صفاں > 1) {
        uint32_t n_دھاگے = کل_صفاں < n_دھاگے_max ? کل_صفاں : n_دھاگے_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        طالما (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t اقدار_ناپ  = ((ne00_padded * حجم(عائم)) + 127) & ~127;
        size_t indices_ناپ = ((ne00_padded * حجم(int32_t)) + 127) & ~127;
        size_t spad_فی_تھاں = ((اقدار_ناپ + indices_ناپ) + 255) & ~255;

        uint32_t n_slots = 2;
        إذا (spad_فی_تھاں * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_فی_thread = spad_فی_تھاں * n_slots;

        طالما (n_دھاگے > 1 && (spad_فی_thread * n_دھاگے) > vtcm_budget) {
            n_دھاگے--;
        }

        size_t کل_بائٹاں = spad_فی_thread * n_دھاگے;
        إذا (کل_بائٹاں > vtcm_budget) {
            أرجع false;
        }

        layout->کل_بائٹاں      = کل_بائٹاں;
        layout->phase1_تھاں_ناپ = spad_فی_تھاں;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_دھاگے        = n_دھاگے;
        layout->n_slots          = n_slots;
        أرجع true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    طالما (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    إذا (ne00 > 1024) {
        طالما (n_chunks * 2 <= n_دھاگے_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_اقدار_ناپ  = ((chunk_elems * حجم(عائم)) + 127) & ~127;
    size_t phase1_indices_ناپ = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
    size_t phase1_تھاں_ناپ    = ((phase1_اقدار_ناپ + phase1_indices_ناپ) + 255) & ~255;
    size_t phase1_کل_ناپ   = phase1_تھاں_ناپ * n_chunks;

    size_t ملاؤ_اقدار_ناپ  = 0;
    size_t ملاؤ_indices_ناپ = 0;
    size_t ملاؤ_اقدار_off   = phase1_کل_ناپ;
    size_t ملاؤ_indices_off  = 0;
    uint32_t ملاؤ_elems      = 0;

    إذا (n_chunks > 1) {
        إذا (is_چوٹی_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t کل_candidates = n_chunks * local_k;
            uint32_t ملاؤ_n_vec = (کل_candidates + 31) / 32;
            uint32_t ملاؤ_n_vec_pow2 = 1;
            طالما (ملاؤ_n_vec_pow2 < ملاؤ_n_vec) ملاؤ_n_vec_pow2 <<= 1;
            ملاؤ_elems = ملاؤ_n_vec_pow2 * 32;
        } وإلا {
            ملاؤ_elems = n_vec_pow2 * 32;
        }
        ملاؤ_اقدار_ناپ  = ((ملاؤ_elems * حجم(عائم)) + 127) & ~127;
        ملاؤ_indices_ناپ = ((ملاؤ_elems * حجم(int32_t)) + 127) & ~127;
        ملاؤ_indices_off  = ملاؤ_اقدار_off + ملاؤ_اقدار_ناپ;
    }

    size_t کل_بائٹاں = phase1_کل_ناپ + ملاؤ_اقدار_ناپ + ملاؤ_indices_ناپ;

    إذا (کل_بائٹاں > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_اقدار_ناپ  = ((chunk_elems * حجم(عائم)) + 127) & ~127;
        phase1_indices_ناپ = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
        phase1_تھاں_ناپ    = ((phase1_اقدار_ناپ + phase1_indices_ناپ) + 255) & ~255;
        phase1_کل_ناپ   = phase1_تھاں_ناپ;
        ملاؤ_اقدار_ناپ   = 0;
        ملاؤ_indices_ناپ  = 0;
        ملاؤ_اقدار_off    = phase1_کل_ناپ;
        ملاؤ_indices_off   = 0;
        ملاؤ_elems         = 0;
        کل_بائٹاں         = phase1_کل_ناپ;
    }

    إذا (کل_بائٹاں > vtcm_budget) {
        أرجع false;
    }

    uint32_t n_slots = 1;
    إذا (n_chunks == 1 && phase1_تھاں_ناپ * 2 <= vtcm_budget) {
        n_slots = 2;
        کل_بائٹاں = phase1_تھاں_ناپ * 2;
    }

    layout->کل_بائٹاں       = کل_بائٹاں;
    layout->phase1_تھاں_ناپ  = phase1_تھاں_ناپ;
    layout->ملاؤ_اقدار_off  = ملاؤ_اقدار_off;
    layout->ملاؤ_indices_off = ملاؤ_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->ملاؤ_elems       = ملاؤ_elems;
    layout->n_دھاگے         = n_chunks;
    layout->n_slots           = n_slots;
    أرجع true;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_ترتیب_دیو_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_ترتیب_دیو_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#جے_انت

#جے_انت // HTP_ARGSORT_OPS_H
