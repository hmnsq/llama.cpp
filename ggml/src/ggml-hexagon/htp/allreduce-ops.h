#اگر_تعریف_نشده ALLREDUCE_OPS_H
#تعریف_کن ALLREDUCE_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#تعریف_کن HTP_ALLREDUCE_MAX_RANKS 4

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

تعداد htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

تعداد htp_allreduce_نما {
    HTP_ALLREDUCE_پر           = 0, // all ranks reduce full tensor
    HTP_ALLREDUCE_SHARDED_FANOUT = 1, // each rank reduces 1/N shard and fans out to all buffers
};

ساكن مضمن size_t htp_allreduce_vtcm_میانگیر_شمار(
    uint32_t n_ranks,
    uint32_t n_نخها,
    bool has_جمع_کن,
    bool is_ردیف_bcast
) {
    أرجع (size_t) (n_ranks + 1) * n_نخها + (has_جمع_کن ? (is_ردیف_bcast ? 1 : n_نخها) : 0);
}

بنية htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_نخها;
    int32_t بلوک_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_در_هر_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_اندازه_در_هر_thread;
    int32_t vtcm_اندازه;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t ردیف_اندازه_aligned;
    int32_t rank_elem_شروع;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_ردیف_bcast;
    int32_t نما;                 // enum htp_allreduce_mode: FULL or SHARDED_FANOUT
};

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر

#پایان_اگر /* ALLREDUCE_OPS_H */
