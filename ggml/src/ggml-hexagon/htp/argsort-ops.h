#اگر_غیر_تعریف_شدہ HTP_ARGSORT_OPS_H
#تعریف_کرو HTP_ARGSORT_OPS_H

#شامل <stdint.h>
#شامل <stddef.h>
#شامل <stdbool.h>
#شامل <string.h>

#شامل "hex-fastdiv.h"

ساخت htp_ترتیب_دو_kernel_params {
    int32_t  n_دھاگے;
    int32_t  کل_صفیں;
    int32_t  صف_آغاز;
    int32_t  صف_انتہا;
    int32_t  ne00;
    int32_t  k;
    int32_t  ترتیب;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_چوٹی_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_سائز;
    int32_t  phase1_خانہ_سائز;
    int32_t  ملاؤ_اقدار_off;
    int32_t  ملاؤ_indices_off;
    int32_t  ملاؤ_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

ساخت htp_ترتیب_دو_vtcm_layout {
    size_t   کل_بائٹیں;
    size_t   phase1_خانہ_سائز;
    size_t   ملاؤ_اقدار_off;
    size_t   ملاؤ_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t ملاؤ_elems;
    uint32_t n_دھاگے;
    uint32_t n_slots;
};

ساکن اندرونی bool htp_ترتیب_دو_solve_layout(
    ساخت htp_ترتیب_دو_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t کل_صفیں,
    uint32_t k,
    uint32_t n_دھاگے_max,
    size_t vtcm_budget,
    bool is_چوٹی_k) {

    memset(layout, 0, حجم(*layout));

    اگر (کل_صفیں > 1) {
        uint32_t n_دھاگے = کل_صفیں < n_دھاگے_max ? کل_صفیں : n_دھاگے_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        جب (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t اقدار_سائز  = ((ne00_padded * حجم(اعشاری)) + 127) & ~127;
        size_t indices_سائز = ((ne00_padded * حجم(int32_t)) + 127) & ~127;
        size_t spad_فی_خانہ = ((اقدار_سائز + indices_سائز) + 255) & ~255;

        uint32_t n_slots = 2;
        اگر (spad_فی_خانہ * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_فی_thread = spad_فی_خانہ * n_slots;

        جب (n_دھاگے > 1 && (spad_فی_thread * n_دھاگے) > vtcm_budget) {
            n_دھاگے--;
        }

        size_t کل_بائٹیں = spad_فی_thread * n_دھاگے;
        اگر (کل_بائٹیں > vtcm_budget) {
            واپس false;
        }

        layout->کل_بائٹیں      = کل_بائٹیں;
        layout->phase1_خانہ_سائز = spad_فی_خانہ;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_دھاگے        = n_دھاگے;
        layout->n_slots          = n_slots;
        واپس true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    جب (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    اگر (ne00 > 1024) {
        جب (n_chunks * 2 <= n_دھاگے_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_اقدار_سائز  = ((chunk_elems * حجم(اعشاری)) + 127) & ~127;
    size_t phase1_indices_سائز = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
    size_t phase1_خانہ_سائز    = ((phase1_اقدار_سائز + phase1_indices_سائز) + 255) & ~255;
    size_t phase1_کل_سائز   = phase1_خانہ_سائز * n_chunks;

    size_t ملاؤ_اقدار_سائز  = 0;
    size_t ملاؤ_indices_سائز = 0;
    size_t ملاؤ_اقدار_off   = phase1_کل_سائز;
    size_t ملاؤ_indices_off  = 0;
    uint32_t ملاؤ_elems      = 0;

    اگر (n_chunks > 1) {
        اگر (is_چوٹی_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t کل_candidates = n_chunks * local_k;
            uint32_t ملاؤ_n_vec = (کل_candidates + 31) / 32;
            uint32_t ملاؤ_n_vec_pow2 = 1;
            جب (ملاؤ_n_vec_pow2 < ملاؤ_n_vec) ملاؤ_n_vec_pow2 <<= 1;
            ملاؤ_elems = ملاؤ_n_vec_pow2 * 32;
        } ورنہ {
            ملاؤ_elems = n_vec_pow2 * 32;
        }
        ملاؤ_اقدار_سائز  = ((ملاؤ_elems * حجم(اعشاری)) + 127) & ~127;
        ملاؤ_indices_سائز = ((ملاؤ_elems * حجم(int32_t)) + 127) & ~127;
        ملاؤ_indices_off  = ملاؤ_اقدار_off + ملاؤ_اقدار_سائز;
    }

    size_t کل_بائٹیں = phase1_کل_سائز + ملاؤ_اقدار_سائز + ملاؤ_indices_سائز;

    اگر (کل_بائٹیں > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_اقدار_سائز  = ((chunk_elems * حجم(اعشاری)) + 127) & ~127;
        phase1_indices_سائز = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
        phase1_خانہ_سائز    = ((phase1_اقدار_سائز + phase1_indices_سائز) + 255) & ~255;
        phase1_کل_سائز   = phase1_خانہ_سائز;
        ملاؤ_اقدار_سائز   = 0;
        ملاؤ_indices_سائز  = 0;
        ملاؤ_اقدار_off    = phase1_کل_سائز;
        ملاؤ_indices_off   = 0;
        ملاؤ_elems         = 0;
        کل_بائٹیں         = phase1_کل_سائز;
    }

    اگر (کل_بائٹیں > vtcm_budget) {
        واپس false;
    }

    uint32_t n_slots = 1;
    اگر (n_chunks == 1 && phase1_خانہ_سائز * 2 <= vtcm_budget) {
        n_slots = 2;
        کل_بائٹیں = phase1_خانہ_سائز * 2;
    }

    layout->کل_بائٹیں       = کل_بائٹیں;
    layout->phase1_خانہ_سائز  = phase1_خانہ_سائز;
    layout->ملاؤ_اقدار_off  = ملاؤ_اقدار_off;
    layout->ملاؤ_indices_off = ملاؤ_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->ملاؤ_elems       = ملاؤ_elems;
    layout->n_دھاگے         = n_chunks;
    layout->n_slots           = n_slots;
    واپس true;
}

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_ترتیب_دو_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_ترتیب_دو_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#اگر_ختم

#اگر_ختم // HTP_ARGSORT_OPS_H
