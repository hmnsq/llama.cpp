#నిర్వచితం_కాకపోతే HTP_ARGSORT_OPS_H
#నిర్వచించు HTP_ARGSORT_OPS_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "hex-fastdiv.h"

నిర్మాణం htp_క్రమీకరించు_kernel_params {
    int32_t  n_దారాలు;
    int32_t  మొత్తం_అడ్డువరుసలు;
    int32_t  అడ్డువరుస_ప్రారంభం;
    int32_t  అడ్డువరుస_చివర;
    int32_t  ne00;
    int32_t  k;
    int32_t  క్రమం;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    int32_t  is_శిఖరం_k;          // 1 if TOP_K, 0 if ARGSORT
    int32_t  use_dma;           // 1 if DMA enabled
    int32_t  chunk_elems;
    int32_t  n_chunks;
    int32_t  vtcm_కొలత;
    int32_t  phase1_స్థానం_కొలత;
    int32_t  విలీనం_విలువలు_off;
    int32_t  విలీనం_indices_off;
    int32_t  విలీనం_elems;
    int32_t  n_slots;
    int32_t  pad[15];
};

నిర్మాణం htp_క్రమీకరించు_vtcm_layout {
    size_t   మొత్తం_బైట్లు;
    size_t   phase1_స్థానం_కొలత;
    size_t   విలీనం_విలువలు_off;
    size_t   విలీనం_indices_off;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t విలీనం_elems;
    uint32_t n_దారాలు;
    uint32_t n_slots;
};

స్థిర అంతఃవరుస bool htp_క్రమీకరించు_solve_layout(
    నిర్మాణం htp_క్రమీకరించు_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t మొత్తం_అడ్డువరుసలు,
    uint32_t k,
    uint32_t n_దారాలు_max,
    size_t vtcm_budget,
    bool is_శిఖరం_k) {

    memset(layout, 0, పరిమాణం(*layout));

    ఒకవేళ (మొత్తం_అడ్డువరుసలు > 1) {
        uint32_t n_దారాలు = మొత్తం_అడ్డువరుసలు < n_దారాలు_max ? మొత్తం_అడ్డువరుసలు : n_దారాలు_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        అయితే (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t విలువలు_కొలత  = ((ne00_padded * పరిమాణం(దశాంశం)) + 127) & ~127;
        size_t indices_కొలత = ((ne00_padded * పరిమాణం(int32_t)) + 127) & ~127;
        size_t spad_చొప్పున_స్థానం = ((విలువలు_కొలత + indices_కొలత) + 255) & ~255;

        uint32_t n_slots = 2;
        ఒకవేళ (spad_చొప్పున_స్థానం * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_చొప్పున_thread = spad_చొప్పున_స్థానం * n_slots;

        అయితే (n_దారాలు > 1 && (spad_చొప్పున_thread * n_దారాలు) > vtcm_budget) {
            n_దారాలు--;
        }

        size_t మొత్తం_బైట్లు = spad_చొప్పున_thread * n_దారాలు;
        ఒకవేళ (మొత్తం_బైట్లు > vtcm_budget) {
            తిరిగి false;
        }

        layout->మొత్తం_బైట్లు      = మొత్తం_బైట్లు;
        layout->phase1_స్థానం_కొలత = spad_చొప్పున_స్థానం;
        layout->chunk_elems      = ne00_padded;
        layout->n_chunks         = 1;
        layout->n_దారాలు        = n_దారాలు;
        layout->n_slots          = n_slots;
        తిరిగి true;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    అయితే (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    ఒకవేళ (ne00 > 1024) {
        అయితే (n_chunks * 2 <= n_దారాలు_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;

    size_t phase1_విలువలు_కొలత  = ((chunk_elems * పరిమాణం(దశాంశం)) + 127) & ~127;
    size_t phase1_indices_కొలత = ((chunk_elems * పరిమాణం(int32_t)) + 127) & ~127;
    size_t phase1_స్థానం_కొలత    = ((phase1_విలువలు_కొలత + phase1_indices_కొలత) + 255) & ~255;
    size_t phase1_మొత్తం_కొలత   = phase1_స్థానం_కొలత * n_chunks;

    size_t విలీనం_విలువలు_కొలత  = 0;
    size_t విలీనం_indices_కొలత = 0;
    size_t విలీనం_విలువలు_off   = phase1_మొత్తం_కొలత;
    size_t విలీనం_indices_off  = 0;
    uint32_t విలీనం_elems      = 0;

    ఒకవేళ (n_chunks > 1) {
        ఒకవేళ (is_శిఖరం_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t మొత్తం_candidates = n_chunks * local_k;
            uint32_t విలీనం_n_vec = (మొత్తం_candidates + 31) / 32;
            uint32_t విలీనం_n_vec_pow2 = 1;
            అయితే (విలీనం_n_vec_pow2 < విలీనం_n_vec) విలీనం_n_vec_pow2 <<= 1;
            విలీనం_elems = విలీనం_n_vec_pow2 * 32;
        } లేకపోతే {
            విలీనం_elems = n_vec_pow2 * 32;
        }
        విలీనం_విలువలు_కొలత  = ((విలీనం_elems * పరిమాణం(దశాంశం)) + 127) & ~127;
        విలీనం_indices_కొలత = ((విలీనం_elems * పరిమాణం(int32_t)) + 127) & ~127;
        విలీనం_indices_off  = విలీనం_విలువలు_off + విలీనం_విలువలు_కొలత;
    }

    size_t మొత్తం_బైట్లు = phase1_మొత్తం_కొలత + విలీనం_విలువలు_కొలత + విలీనం_indices_కొలత;

    ఒకవేళ (మొత్తం_బైట్లు > vtcm_budget && n_chunks > 1) {
        n_chunks = 1;
        chunk_elems = n_vec_pow2 * 32;
        phase1_విలువలు_కొలత  = ((chunk_elems * పరిమాణం(దశాంశం)) + 127) & ~127;
        phase1_indices_కొలత = ((chunk_elems * పరిమాణం(int32_t)) + 127) & ~127;
        phase1_స్థానం_కొలత    = ((phase1_విలువలు_కొలత + phase1_indices_కొలత) + 255) & ~255;
        phase1_మొత్తం_కొలత   = phase1_స్థానం_కొలత;
        విలీనం_విలువలు_కొలత   = 0;
        విలీనం_indices_కొలత  = 0;
        విలీనం_విలువలు_off    = phase1_మొత్తం_కొలత;
        విలీనం_indices_off   = 0;
        విలీనం_elems         = 0;
        మొత్తం_బైట్లు         = phase1_మొత్తం_కొలత;
    }

    ఒకవేళ (మొత్తం_బైట్లు > vtcm_budget) {
        తిరిగి false;
    }

    uint32_t n_slots = 1;
    ఒకవేళ (n_chunks == 1 && phase1_స్థానం_కొలత * 2 <= vtcm_budget) {
        n_slots = 2;
        మొత్తం_బైట్లు = phase1_స్థానం_కొలత * 2;
    }

    layout->మొత్తం_బైట్లు       = మొత్తం_బైట్లు;
    layout->phase1_స్థానం_కొలత  = phase1_స్థానం_కొలత;
    layout->విలీనం_విలువలు_off  = విలీనం_విలువలు_off;
    layout->విలీనం_indices_off = విలీనం_indices_off;
    layout->chunk_elems       = chunk_elems;
    layout->n_chunks          = n_chunks;
    layout->విలీనం_elems       = విలీనం_elems;
    layout->n_దారాలు         = n_chunks;
    layout->n_slots           = n_slots;
    తిరిగి true;
}

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_క్రమీకరించు_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_క్రమీకరించు_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

#అయితే_ముగింపు // HTP_ARGSORT_OPS_H
