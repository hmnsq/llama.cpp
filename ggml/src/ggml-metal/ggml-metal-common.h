// helper functions for ggml-metal that are too difficult to implement in Objective-C

#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

ਬਣਤਰ ggml_tensor;
ਬਣਤਰ ggml_cgraph;

ਗਿਣਤੀ ggml_mem_range_type {
    MEM_RANGE_TYPE_SRC = 0,
    MEM_RANGE_TYPE_DST = 1,
};

// a helper object that can be used for reordering operations to improve concurrency
//
// the fundamental idea is that a set of tasks (either ggml ops, or something else) can run concurrently if they
//   don't write to a memory that is being read by another task or written to by another task in the set
//
// with this structure, we can add tasks to the set, setting memory constraints. we can also check if a new task
//   can be added to the set without violating the constraints (i.e. if it can be executed concurrently with the
//   tasks already in the set)
//
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_mem_ranges * ggml_mem_ranges_t;

ggml_mem_ranges_t ggml_mem_ranges_init(ਪੂਰਨ debug);
ਖਾਲੀ ggml_mem_ranges_free(ggml_mem_ranges_t mrs);

// remove all ranges from the set
ਖਾਲੀ ggml_mem_ranges_reset(ggml_mem_ranges_t mrs);

// add src or dst ranges to track
bool ggml_mem_ranges_ਜੋੜੋ(ggml_mem_ranges_t mrs, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor);

// return false if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor);

// reorder the nodes in the graph to improve concurrency, while respecting the fusions of a device with props
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
ਖਾਲੀ ggml_ਗ੍ਰਾਫ_optimize(ਬਣਤਰ ggml_cgraph * gf);

// mat-mat vs mat-vec dispatch; used by both supports_op and ggml_metal_op_mul_mat*
bool ggml_metal_op_mul_mat_use_fwht (ਅਚਲ ਬਣਤਰ ggml_tensor * op, size_t max_tg_mem);
bool ggml_metal_op_mul_mat_use_mm   (ਅਚਲ ਬਣਤਰ ggml_tensor * op, bool has_simdgroup_mm);
bool ggml_metal_op_mul_mat_id_use_mm(ਅਚਲ ਬਣਤਰ ggml_tensor * op, bool has_simdgroup_mm);

bool ggml_metal_op_mul_mat_use_mma    (ਅਚਲ ਬਣਤਰ ggml_tensor * op);
bool ggml_metal_op_mul_mat_may_use_mma(ਅਚਲ ਬਣਤਰ ggml_tensor * op); // graph structure only

// the few-row MMA kernel for a src0 type and rt src1 tiles: per 32-weight block (q4_0, q8_0 with one tile), q5_K, or the generic 64-weight chunk kernel
ਗਿਣਤੀ ggml_metal_mma_ਕਿਸਮ { GGML_METAL_MMA_ਕਿਸਮ_BLK, GGML_METAL_MMA_ਕਿਸਮ_Q5_K, GGML_METAL_MMA_ਕਿਸਮ_GEN };
ਗਿਣਤੀ ggml_metal_mma_ਕਿਸਮ ggml_metal_mul_mv_mma_ਕਿਸਮ(ਗਿਣਤੀ ggml_type type, ਪੂਰਨ rt);

// the src1 tiles of the few-row MMA kernels for mat-mul op: one 8-row tile, or two above 8 rows
ਪੂਰਨ ggml_metal_mul_mv_mma_rt(ਅਚਲ ਬਣਤਰ ggml_tensor * op);

// the weights of K per simdgroup step of the few-row MMA kernel for a src0 type and rt src1 tiles, 0 if none takes the type
int64_t ggml_metal_mul_mv_mma_k_ਕਦਮ(ਗਿਣਤੀ ggml_type type, ਪੂਰਨ rt);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ
