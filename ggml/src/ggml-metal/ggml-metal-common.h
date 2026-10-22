// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

#अन्तर्भाव "ggml.h"

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stddef.h>

#ifdef __cplusplus
बाह्य "C" {
#endif

संरचना ggml_tensor;
संरचना ggml_cgraph;

गणना ggml_mem_range_type {
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
प्रकारनिर्देश संरचना ggml_mem_ranges * ggml_mem_ranges_t;

ggml_mem_ranges_t ggml_mem_ranges_init(पूर्णाङ्क debug);
शून्य ggml_mem_ranges_free(ggml_mem_ranges_t mrs);

// remove all ranges from the set
शून्य ggml_mem_ranges_reset(ggml_mem_ranges_t mrs);

// add src or dst ranges to track
bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ध्रुव संरचना ggml_tensor * tensor);

// return false if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ध्रुव संरचना ggml_tensor * tensor);

// reorder the nodes in the graph to improve concurrency, while respecting the fusions of a device with props
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
शून्य ggml_graph_optimize(संरचना ggml_cgraph * gf);

// mat-mat vs mat-vec dispatch; used by both supports_op and ggml_metal_op_mul_mat*
bool ggml_metal_op_mul_mat_use_fwht (ध्रुव संरचना ggml_tensor * op, माप_प्रकार max_tg_mem);
bool ggml_metal_op_mul_mat_use_mm   (ध्रुव संरचना ggml_tensor * op, bool has_simdgroup_mm);
bool ggml_metal_op_mul_mat_id_use_mm(ध्रुव संरचना ggml_tensor * op, bool has_simdgroup_mm);

bool ggml_metal_op_mul_mat_use_mma    (ध्रुव संरचना ggml_tensor * op);
bool ggml_metal_op_mul_mat_may_use_mma(ध्रुव संरचना ggml_tensor * op); // graph structure only

// the few-row MMA kernel for a src0 type and rt src1 tiles: per 32-weight block (q4_0, q8_0 with one tile), q5_K, or the generic 64-weight chunk kernel
गणना ggml_metal_mma_kind { GGML_METAL_MMA_KIND_BLK, GGML_METAL_MMA_KIND_Q5_K, GGML_METAL_MMA_KIND_GEN };
गणना ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(गणना ggml_type type, पूर्णाङ्क rt);

// the src1 tiles of the few-row MMA kernels for mat-mul op: one 8-row tile, or two above 8 rows
पूर्णाङ्क ggml_metal_mul_mv_mma_rt(ध्रुव संरचना ggml_tensor * op);

// the weights of K per simdgroup step of the few-row MMA kernel for a src0 type and rt src1 tiles, 0 if none takes the type
int64_t ggml_metal_mul_mv_mma_k_step(गणना ggml_type type, पूर्णाङ्क rt);

#ifdef __cplusplus
}
#endif
