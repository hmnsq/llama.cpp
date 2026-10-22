#यदि_अपरिभाषितम् ALLREDUCE_OPS_H
#परिभाषय ALLREDUCE_OPS_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>

#परिभाषय HTP_ALLREDUCE_MAX_RANKS 4

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

गणना htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

गणना htp_allreduce_बहुलकम् {
    HTP_ALLREDUCE_पूर्णम्           = 0, // all ranks reduce full tensor
    HTP_ALLREDUCE_SHARDED_FANOUT = 1, // each rank reduces 1/N shard and fans out to all buffers
};

स्थिर अन्तःस्थ size_t htp_allreduce_vtcm_बफर_गणनम्(
    uint32_t n_ranks,
    uint32_t n_तन्तवः,
    bool has_सङ्कलय,
    bool is_row_bcast
) {
    प्रत्यागम (size_t) (n_ranks + 1) * n_तन्तवः + (has_सङ्कलय ? (is_row_bcast ? 1 : n_तन्तवः) : 0);
}

संरचना htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_तन्तवः;
    int32_t खण्डम्_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_प्रत्येकम्_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_आकार_प्रत्येकम्_thread;
    int32_t vtcm_आकार;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t row_आकार_aligned;
    int32_t rank_elem_start;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_row_bcast;
    int32_t बहुलकम्;                 // enum htp_allreduce_mode: FULL or SHARDED_FANOUT
};

#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः

#यदि_अन्तः /* ALLREDUCE_OPS_H */
