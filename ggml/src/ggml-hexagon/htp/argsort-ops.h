#که_نه_تعریف_شوی HTP_ARGSORT_OPS_H
#تعریف_کړه HTP_ARGSORT_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>
#ضمن <string.h>

#ضمن "hex-fastdiv.h"

بنية htp_ترتیبول_kernel_params {
    int32_t  n_تارونه;
    int32_t  ټول_لیکې;
    int32_t  لیکه_پیل;
    int32_t  لیکه_پای_ټکی;
    int32_t  ne00;
    int32_t  k;
    int32_t  ترتیب;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_سر_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_اندازه;
    int32_t  phase1_خونه_اندازه;
    int32_t  یوځای_ارزښتونه_off;
    int32_t  یوځای_indices_off;
    int32_t  یوځای_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

بنية htp_ترتیبول_vtcm_layout {
    size_t   ټول_بایټونه;
    size_t   phase1_خونه_اندازه;
    size_t   یوځای_ارزښتونه_off;
    size_t   یوځای_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t یوځای_elems;
    uint32_t n_تارونه;
    uint32_t n_slots;
};

ساكن مضمن bool htp_ترتیبول_solve_layout(
    بنية htp_ترتیبول_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ټول_لیکې,
    uint32_t k,
    uint32_t n_تارونه_max,
    size_t vtcm_budget,
    bool is_سر_k) {

    memset(layout, 0, حجم(*layout));

    إذا (ټول_لیکې > 1) {
        uint32_t n_تارونه = ټول_لیکې < n_تارونه_max ? ټول_لیکې : n_تارونه_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        طالما (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t ارزښتونه_اندازه  = ((ne00_padded * حجم(عائم)) + 127) & ~127;
        size_t indices_اندازه = ((ne00_padded * حجم(int32_t)) + 127) & ~127;
        size_t spad_فی_خونه = ((ارزښتونه_اندازه + indices_اندازه) + 255) & ~255;

        uint32_t n_slots = 2;
        إذا (spad_فی_خونه * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_فی_thread = spad_فی_خونه * n_slots;

        طالما (n_تارونه > 1 && (spad_فی_thread * n_تارونه) > vtcm_budget) {
            n_تارونه--;
        }

        size_t ټول_بایټونه = spad_فی_thread * n_تارونه;
        إذا (ټول_بایټونه > vtcm_budget) {
            أرجع false;
        }

        layout->ټول_بایټونه      = ټول_بایټونه;
        layout->phase1_خونه_اندازه = spad_فی_خونه;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_تارونه        = n_تارونه;
        layout->n_slots          = n_slots;
        أرجع true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    طالما (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    إذا (ne00 > 1024) {
        طالما (n_chunks * 2 <= n_تارونه_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_ارزښتونه_اندازه  = ((chunk_elems * حجم(عائم)) + 127) & ~127;
    size_t phase1_indices_اندازه = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
    size_t phase1_خونه_اندازه    = ((phase1_ارزښتونه_اندازه + phase1_indices_اندازه) + 255) & ~255;
    size_t phase1_ټول_اندازه   = phase1_خونه_اندازه * n_chunks;

    size_t یوځای_ارزښتونه_اندازه  = 0;
    size_t یوځای_indices_اندازه = 0;
    size_t یوځای_ارزښتونه_off   = phase1_ټول_اندازه;
    size_t یوځای_indices_off  = 0;
    uint32_t یوځای_elems      = 0;

    إذا (n_chunks > 1) {
        إذا (is_سر_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t ټول_candidates = n_chunks * local_k;
            uint32_t یوځای_n_vec = (ټول_candidates + 31) / 32;
            uint32_t یوځای_n_vec_pow2 = 1;
            طالما (یوځای_n_vec_pow2 < یوځای_n_vec) یوځای_n_vec_pow2 <<= 1;
            یوځای_elems = یوځای_n_vec_pow2 * 32;
        } وإلا {
            یوځای_elems = n_vec_pow2 * 32;
        }
        یوځای_ارزښتونه_اندازه  = ((یوځای_elems * حجم(عائم)) + 127) & ~127;
        یوځای_indices_اندازه = ((یوځای_elems * حجم(int32_t)) + 127) & ~127;
        یوځای_indices_off  = یوځای_ارزښتونه_off + یوځای_ارزښتونه_اندازه;
    }

    size_t ټول_بایټونه = phase1_ټول_اندازه + یوځای_ارزښتونه_اندازه + یوځای_indices_اندازه;

    إذا (ټول_بایټونه > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_ارزښتونه_اندازه  = ((chunk_elems * حجم(عائم)) + 127) & ~127;
        phase1_indices_اندازه = ((chunk_elems * حجم(int32_t)) + 127) & ~127;
        phase1_خونه_اندازه    = ((phase1_ارزښتونه_اندازه + phase1_indices_اندازه) + 255) & ~255;
        phase1_ټول_اندازه   = phase1_خونه_اندازه;
        یوځای_ارزښتونه_اندازه   = 0;
        یوځای_indices_اندازه  = 0;
        یوځای_ارزښتونه_off    = phase1_ټول_اندازه;
        یوځای_indices_off   = 0;
        یوځای_elems         = 0;
        ټول_بایټونه         = phase1_ټول_اندازه;
    }

    إذا (ټول_بایټونه > vtcm_budget) {
        أرجع false;
    }

    uint32_t n_slots = 1;
    إذا (n_chunks == 1 && phase1_خونه_اندازه * 2 <= vtcm_budget) {
        n_slots = 2;
        ټول_بایټونه = phase1_خونه_اندازه * 2;
    }

    layout->ټول_بایټونه       = ټول_بایټونه;
    layout->phase1_خونه_اندازه  = phase1_خونه_اندازه;
    layout->یوځای_ارزښتونه_off  = یوځای_ارزښتونه_off;
    layout->یوځای_indices_off = یوځای_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->یوځای_elems       = یوځای_elems;
    layout->n_تارونه         = n_chunks;
    layout->n_slots           = n_slots;
    أرجع true;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_ترتیبول_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_ترتیبول_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#د_که_پای

#د_که_پای // HTP_ARGSORT_OPS_H
