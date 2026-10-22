// helper functions for ggml-metal that are too difficult to implement in Objective-C

#సూచన once

#అన్తర్భావ "ggml.h"

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stddef.h>

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

నిర్మాణం ggml_tensor;
నిర్మాణం ggml_cgraph;

గణన ggml_mem_range_type {
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
రకనిర్వచనం నిర్మాణం ggml_mem_ranges * ggml_mem_ranges_t;

ggml_mem_ranges_t ggml_mem_ranges_init(పూర్ణాంకం debug);
శూన్య ggml_mem_ranges_free(ggml_mem_ranges_t mrs);

// remove all ranges from the set
శూన్య ggml_mem_ranges_reset(ggml_mem_ranges_t mrs);

// add src or dst ranges to track
bool ggml_mem_ranges_కూడు(ggml_mem_ranges_t mrs, స్థిరరాశి నిర్మాణం ggml_tensor * tensor);

// return false if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, స్థిరరాశి నిర్మాణం ggml_tensor * tensor);

// reorder the nodes in the graph to improve concurrency, while respecting the fusions of a device with props
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
శూన్య ggml_లేఖాచిత్రం_optimize(నిర్మాణం ggml_cgraph * gf);

// mat-mat vs mat-vec dispatch; used by both supports_op and ggml_metal_op_mul_mat*
bool ggml_metal_op_mul_mat_use_fwht (స్థిరరాశి నిర్మాణం ggml_tensor * op, size_t max_tg_mem);
bool ggml_metal_op_mul_mat_use_mm   (స్థిరరాశి నిర్మాణం ggml_tensor * op, bool has_simdgroup_mm);
bool ggml_metal_op_mul_mat_id_use_mm(స్థిరరాశి నిర్మాణం ggml_tensor * op, bool has_simdgroup_mm);

bool ggml_metal_op_mul_mat_use_mma    (స్థిరరాశి నిర్మాణం ggml_tensor * op);
bool ggml_metal_op_mul_mat_may_use_mma(స్థిరరాశి నిర్మాణం ggml_tensor * op); // graph structure only

// the few-row MMA kernel for a src0 type and rt src1 tiles: per 32-weight block (q4_0, q8_0 with one tile), q5_K, or the generic 64-weight chunk kernel
గణన ggml_metal_mma_రకం { GGML_METAL_MMA_రకం_BLK, GGML_METAL_MMA_రకం_Q5_K, GGML_METAL_MMA_రకం_GEN };
గణన ggml_metal_mma_రకం ggml_metal_mul_mv_mma_రకం(గణన ggml_type type, పూర్ణాంకం rt);

// the src1 tiles of the few-row MMA kernels for mat-mul op: one 8-row tile, or two above 8 rows
పూర్ణాంకం ggml_metal_mul_mv_mma_rt(స్థిరరాశి నిర్మాణం ggml_tensor * op);

// the weights of K per simdgroup step of the few-row MMA kernel for a src0 type and rt src1 tiles, 0 if none takes the type
int64_t ggml_metal_mul_mv_mma_k_అడుగు(గణన ggml_type type, పూర్ణాంకం rt);

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
