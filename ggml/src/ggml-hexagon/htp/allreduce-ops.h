#که_نه_تعریف_شوی ALLREDUCE_OPS_H
#تعریف_کړه ALLREDUCE_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#تعریف_کړه HTP_ALLREDUCE_MAX_RANKS 4

#که_تعریف_شوی __cplusplus
خارجي "C" {
#د_که_پای

تعداد htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

تعداد htp_allreduce_ډېرپېښ {
    HTP_ALLREDUCE_ډک           = 0, // all ranks reduce full tensor
    HTP_ALLREDUCE_SHARDED_FANOUT = 1, // each rank reduces 1/N shard and fans out to all buffers
};

ساكن مضمن size_t htp_allreduce_vtcm_بفر_شمېر(
    uint32_t n_ranks,
    uint32_t n_تارونه,
    bool has_جمع_کړه,
    bool is_لیکه_bcast
) {
    أرجع (size_t) (n_ranks + 1) * n_تارونه + (has_جمع_کړه ? (is_لیکه_bcast ? 1 : n_تارونه) : 0);
}

بنية htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_تارونه;
    int32_t بلاک_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_فی_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_اندازه_فی_thread;
    int32_t vtcm_اندازه;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t لیکه_اندازه_aligned;
    int32_t rank_elem_پیل;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_لیکه_bcast;
    int32_t ډېرپېښ;                 // enum htp_allreduce_mode: FULL or SHARDED_FANOUT
};

#که_تعریف_شوی __cplusplus
}
#د_که_پای

#د_که_پای /* ALLREDUCE_OPS_H */
