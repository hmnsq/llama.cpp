#جے_بنا_تعریف ALLREDUCE_OPS_H
#تعریف_کرو ALLREDUCE_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#تعریف_کرو HTP_ALLREDUCE_MAX_RANKS 4

#جے_تعریف_شدہ __cplusplus
خارجي "C" {
#جے_انت

تعداد htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

تعداد htp_allreduce_منوال {
    HTP_ALLREDUCE_بھریا           = 0, // all ranks reduce full tensor
    HTP_ALLREDUCE_SHARDED_FANOUT = 1, // each rank reduces 1/N shard and fans out to all buffers
};

ساكن مضمن size_t htp_allreduce_vtcm_بفر_گنتی(
    uint32_t n_ranks,
    uint32_t n_دھاگے,
    bool has_جوڑو,
    bool is_صف_bcast
) {
    أرجع (size_t) (n_ranks + 1) * n_دھاگے + (has_جوڑو ? (is_صف_bcast ? 1 : n_دھاگے) : 0);
}

بنية htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_دھاگے;
    int32_t بلاک_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_فی_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_ناپ_فی_thread;
    int32_t vtcm_ناپ;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t صف_ناپ_aligned;
    int32_t rank_elem_شروع;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_صف_bcast;
    int32_t منوال;                 // enum htp_allreduce_mode: FULL or SHARDED_FANOUT
};

#جے_تعریف_شدہ __cplusplus
}
#جے_انت

#جے_انت /* ALLREDUCE_OPS_H */
