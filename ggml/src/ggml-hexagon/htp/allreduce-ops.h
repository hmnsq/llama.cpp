#ifndef ALLREDUCE_OPS_H
#define ALLREDUCE_OPS_H

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdbool.h>

#define HTP_ALLREDUCE_MAX_RANKS 4

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

গণনা htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

গণনা htp_allreduce_mode {
    HTP_ALLREDUCE_FULL           = 0, // all ranks reduce full tensor
    HTP_ALLREDUCE_SHARDED_FANOUT = 1, // each rank reduces 1/N shard and fans out to all buffers
};

স্থির অন্তঃরেখ size_t htp_allreduce_vtcm_বাফার_গুনতি(
    uint32_t n_ranks,
    uint32_t n_threads,
    bool has_add,
    bool is_row_bcast
) {
    ফেরত (size_t) (n_ranks + 1) * n_threads + (has_add ? (is_row_bcast ? 1 : n_threads) : 0);
}

কাঠামো htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_threads;
    int32_t block_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_per_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_আকার_per_thread;
    int32_t vtcm_আকার;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t row_আকার_aligned;
    int32_t rank_elem_start;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_row_bcast;
    int32_t mode;                 // enum htp_allreduce_mode: FULL or SHARDED_FANOUT
};

#ifdef __cplusplus
}
#endif

#endif /* ALLREDUCE_OPS_H */
