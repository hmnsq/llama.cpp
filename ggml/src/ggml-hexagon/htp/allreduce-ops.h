#నిర్వచితం_కాకపోతే ALLREDUCE_OPS_H
#నిర్వచించు ALLREDUCE_OPS_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdbool.h>

#నిర్వచించు HTP_ALLREDUCE_MAX_RANKS 4

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

గణన htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

గణన htp_allreduce_బహుళకం {
    HTP_ALLREDUCE_నిండిన           = 0, // all ranks reduce full tensor
    HTP_ALLREDUCE_SHARDED_FANOUT = 1, // each rank reduces 1/N shard and fans out to all buffers
};

స్థిర అంతఃవరుస size_t htp_allreduce_vtcm_బఫరు_లెక్క(
    uint32_t n_ranks,
    uint32_t n_దారాలు,
    bool has_కూడు,
    bool is_అడ్డువరుస_bcast
) {
    తిరిగి (size_t) (n_ranks + 1) * n_దారాలు + (has_కూడు ? (is_అడ్డువరుస_bcast ? 1 : n_దారాలు) : 0);
}

నిర్మాణం htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_దారాలు;
    int32_t ఖండం_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_చొప్పున_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_కొలత_చొప్పున_thread;
    int32_t vtcm_కొలత;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t అడ్డువరుస_కొలత_aligned;
    int32_t rank_elem_ప్రారంభం;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_అడ్డువరుస_bcast;
    int32_t బహుళకం;                 // enum htp_allreduce_mode: FULL or SHARDED_FANOUT
};

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు

#అయితే_ముగింపు /* ALLREDUCE_OPS_H */
