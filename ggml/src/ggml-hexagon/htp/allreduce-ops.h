#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ ALLREDUCE_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು ALLREDUCE_OPS_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ALLREDUCE_MAX_RANKS 4

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

ಎಣಿಕೆ htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

ಎಣಿಕೆ htp_allreduce_ಬಹುಲಕ {
    HTP_ALLREDUCE_ತುಂಬಿದ           = 0, // all ranks reduce full tensor
    HTP_ALLREDUCE_SHARDED_FANOUT = 1, // each rank reduces 1/N shard and fans out to all buffers
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_allreduce_vtcm_ಬಫರ್_ಎಣಿಕೆ(
    uint32_t n_ranks,
    uint32_t n_ಎಳೆಗಳು,
    bool has_ಕೂಡಿಸು,
    bool is_ಅಡ್ಡಸಾಲು_bcast
) {
    ಹಿಂತಿರುಗಿ (size_t) (n_ranks + 1) * n_ಎಳೆಗಳು + (has_ಕೂಡಿಸು ? (is_ಅಡ್ಡಸಾಲು_bcast ? 1 : n_ಎಳೆಗಳು) : 0);
}

ರಚನೆ htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_ಎಳೆಗಳು;
    int32_t ಖಂಡ_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_ಪ್ರತಿ_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    int32_t vtcm_ಗಾತ್ರ;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    int32_t rank_elem_ಆರಂಭ;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_ಅಡ್ಡಸಾಲು_bcast;
    int32_t ಬಹುಲಕ;                 // enum htp_allreduce_mode: FULL or SHARDED_FANOUT
};

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ /* ALLREDUCE_OPS_H */
