#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ ALLREDUCE_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ALLREDUCE_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ALLREDUCE_MAX_RANKS 4

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

ਗਿਣਤੀ htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

ਗਿਣਤੀ htp_allreduce_ਬਹੁਲਕ {
    HTP_ALLREDUCE_ਭਰਿਆ           = 0, // all ranks reduce full tensor
    HTP_ALLREDUCE_SHARDED_FANOUT = 1, // each rank reduces 1/N shard and fans out to all buffers
};

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_allreduce_vtcm_ਬਫਰ_ਗਿਣਤੀ(
    uint32_t n_ranks,
    uint32_t n_ਧਾਗੇ,
    bool has_ਜੋੜੋ,
    bool is_ਪੰਕਤੀ_bcast
) {
    ਵਾਪਸ (size_t) (n_ranks + 1) * n_ਧਾਗੇ + (has_ਜੋੜੋ ? (is_ਪੰਕਤੀ_bcast ? 1 : n_ਧਾਗੇ) : 0);
}

ਬਣਤਰ htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_ਧਾਗੇ;
    int32_t ਖੰਡ_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_ਪ੍ਰਤੀ_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    int32_t vtcm_ਮਾਪ;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t ਪੰਕਤੀ_ਮਾਪ_aligned;
    int32_t rank_elem_ਸ਼ੁਰੂ;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_ਪੰਕਤੀ_bcast;
    int32_t ਬਹੁਲਕ;                 // enum htp_allreduce_mode: FULL or SHARDED_FANOUT
};

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ /* ALLREDUCE_OPS_H */
