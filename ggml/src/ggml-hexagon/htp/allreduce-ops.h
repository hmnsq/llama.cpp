#अगर_अघोषित ALLREDUCE_OPS_H
#घोषणा ALLREDUCE_OPS_H

#समावेश <stdint.h>
#समावेश <stddef.h>
#समावेश <stdbool.h>

#घोषणा HTP_ALLREDUCE_MAX_RANKS 4

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

क्रमागत htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

क्रमागत htp_allreduce_mode {
    HTP_ALLREDUCE_FULL           = 0, // all ranks reduce full tensor
    HTP_ALLREDUCE_SHARDED_FANOUT = 1, // each rank reduces 1/N shard and fans out to all buffers
};

अटल अंतरभूत माप_प्रकार htp_allreduce_vtcm_बफर_गिनती(
    uint32_t n_ranks,
    uint32_t n_threads,
    bool has_add,
    bool is_row_bcast
) {
    वापस (माप_प्रकार) (n_ranks + 1) * n_threads + (has_add ? (is_row_bcast ? 1 : n_threads) : 0);
}

काष्ठा htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_threads;
    int32_t block_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_per_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_आकार_per_thread;
    int32_t vtcm_आकार;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t row_आकार_aligned;
    int32_t rank_elem_start;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_row_bcast;
    int32_t mode;                 // enum htp_allreduce_mode: FULL or SHARDED_FANOUT
};

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर

#पूर्ण_अगर /* ALLREDUCE_OPS_H */
