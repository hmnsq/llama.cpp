/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ಅನ್ತರ್ಭಾವ "acl_tensor.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstring>

acl_ದತ್ತಾಂಶ_Type ggml_cann_type_mapping(ggml_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಹಿಂತಿರುಗಿ ACL_FLOAT;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ಹಿಂತಿರುಗಿ ACL_FLOAT16;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಹಿಂತಿರುಗಿ ACL_BF16;
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            ಹಿಂತಿರುಗಿ ACL_INT8;
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            ಹಿಂತಿರುಗಿ ACL_INT16;
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            ಹಿಂತಿರುಗಿ ACL_INT32;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಹಿಂತಿರುಗಿ ACL_INT4;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ ACL_INT8;
        ಸ್ಥಿತಿ GGML_TYPE_I64:
            ಹಿಂತಿರುಗಿ ACL_INT64;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ACL_DT_UNDEFINED;
    }
}

acl_tensor_ptr ggml_cann_create_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                                       int64_t *           ne,
                                       size_t *            nb,
                                       int64_t             dims,
                                       aclFormat           format,
                                       size_t              ಸ್ಥಾನಾಂತರ) {
    // If tensor is bcasted, Up to GGML_MAX_DIMS additional dimensions will be
    // added.
    int64_t acl_ne[GGML_MAX_DIMS * 2], acl_stride[GGML_MAX_DIMS * 2];

    ಆದರೆ (ne == ಶೂನ್ಯಸೂಚಕ) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
            acl_ne[i]     = tensor->ne[i];
            // The step size of acl is in elements.
            acl_stride[i] = tensor->nb[i] / ggml_element_ಗಾತ್ರ(tensor);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // With bcast
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < dims; i++) {
            acl_ne[i]     = ne[i];
            acl_stride[i] = nb[i] / ggml_element_ಗಾತ್ರ(tensor);
        }
    }

    int64_t final_dims      = (dims == 0 ? GGML_MAX_DIMS : dims);
    int64_t acl_storage_len = 1;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < final_dims; i++) {
        acl_storage_len += (acl_ne[i] - 1) * acl_stride[i];
    }
    size_t elem_ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ / ggml_element_ಗಾತ್ರ(tensor);
    acl_storage_len += elem_ಸ್ಥಾನಾಂತರ;

    // Reverse ne and stride.
    std::reverse(acl_ne, acl_ne + final_dims);
    std::reverse(acl_stride, acl_stride + final_dims);

    aclTensor * raw = aclCreateTensor(acl_ne, final_dims, ggml_cann_type_mapping(tensor->type), acl_stride, elem_ಸ್ಥಾನಾಂತರ,
                                      format, &acl_storage_len, 1, tensor->ದತ್ತಾಂಶ);

    ಹಿಂತಿರುಗಿ acl_tensor_ptr(raw);
}

acl_int_array_ptr ggml_cann_create_int_array(ಸ್ಥಿರಾಂಕ int64_t * value, uint64_t size) {
    aclIntArray * raw = aclCreateIntArray(value, size);
    ಹಿಂತಿರುಗಿ acl_int_array_ptr(raw);
}

acl_scalar_ptr ggml_cann_create_scalar(ಶೂನ್ಯ * value, acl_ದತ್ತಾಂಶ_Type ದತ್ತಾಂಶ_Type) {
    aclScalar * raw = aclCreateScalar(value, ದತ್ತಾಂಶ_Type);
    ಹಿಂತಿರುಗಿ acl_scalar_ptr(raw);
}

ತಾರ್ಕಿಕ ggml_cann_need_bcast(ಸ್ಥಿರಾಂಕ ggml_tensor * t0, ಸ್ಥಿರಾಂಕ ggml_tensor * t1) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
        ಆದರೆ (t1->ne[i] != t0->ne[i] && t1->ne[i] != 1) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

int64_t ggml_cann_get_bcast_shape(ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
                                  ಸ್ಥಿರಾಂಕ ggml_tensor * src1,
                                  int64_t *           bcast_src0_ne,
                                  int64_t *           bcast_src1_ne,
                                  size_t *            bcast_src0_nb,
                                  size_t *            bcast_src1_nb) {
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    ಪೂರ್ಣಾಂಕ bcast_dim_cnt = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr                   = src0->ne[i] / src1->ne[i];
        bcast_src0_ne[bcast_dim_cnt] = src0->ne[i] / nr;
        bcast_src1_ne[bcast_dim_cnt] = src1->ne[i];
        bcast_src0_nb[bcast_dim_cnt] = src0->nb[i];
        bcast_src1_nb[bcast_dim_cnt] = src1->nb[i];
        bcast_dim_cnt++;
        ಆದರೆ (nr != 1) {
            // Need to add an extra dim.
            bcast_src0_ne[bcast_dim_cnt] = nr;
            bcast_src1_ne[bcast_dim_cnt] = 1;
            bcast_src0_nb[bcast_dim_cnt] = bcast_src0_nb[bcast_dim_cnt - 1] * bcast_src0_ne[bcast_dim_cnt - 1];
            bcast_src1_nb[bcast_dim_cnt] = bcast_src1_nb[bcast_dim_cnt - 1] * bcast_src1_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    ಹಿಂತಿರುಗಿ bcast_dim_cnt;
}

int64_t ggml_cann_get_mulmat_bcast_shape(ಸ್ಥಿರಾಂಕ int64_t * ಒಳಹರಿವು_ne,
                                         ಸ್ಥಿರಾಂಕ int64_t * ತೂಕ_ne,
                                         ಸ್ಥಿರಾಂಕ int64_t * dst_ne,
                                         ಸ್ಥಿರಾಂಕ size_t *  ಒಳಹರಿವು_nb,
                                         ಸ್ಥಿರಾಂಕ size_t *  ತೂಕ_nb,
                                         ಸ್ಥಿರಾಂಕ size_t *  dst_nb,
                                         int64_t *       bcast_ಒಳಹರಿವು_ne,
                                         int64_t *       bcast_ತೂಕ_ne,
                                         int64_t *       bcast_dst_ne,
                                         size_t *        bcast_ಒಳಹರಿವು_nb,
                                         size_t *        bcast_ತೂಕ_nb,
                                         size_t *        bcast_dst_nb) {
    // input and dst shoule in same shape, except first two dims.
    GGML_ASSERT(ಒಳಹರಿವು_ne[2] == dst_ne[2]);
    GGML_ASSERT(ಒಳಹರಿವು_ne[3] == dst_ne[3]);

    ಪೂರ್ಣಾಂಕ bcast_dim_cnt = 0;

    // For mul_mat, a dimension needs to be added before the dimension that
    // weight needs to be expanded to satisfy the bcast rule of matrix
    // multiplication.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = ಒಳಹರಿವು_ne[i] / ತೂಕ_ne[i];
        // Do not use bcast in the first two dimensions because we only support
        // the bcast batch dimension. Just copy them.
        ಆದರೆ (i < 2 || nr == 1) {
            bcast_ಒಳಹರಿವು_ne[bcast_dim_cnt]  = ಒಳಹರಿವು_ne[i];
            bcast_ತೂಕ_ne[bcast_dim_cnt] = ತೂಕ_ne[i];
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i];

            bcast_ಒಳಹರಿವು_nb[bcast_dim_cnt]  = ಒಳಹರಿವು_nb[i];
            bcast_ತೂಕ_nb[bcast_dim_cnt] = ತೂಕ_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_dim_cnt++;
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Need to add an extra dim.
            bcast_ಒಳಹರಿವು_ne[bcast_dim_cnt]  = nr;
            bcast_dst_ne[bcast_dim_cnt]    = nr;
            bcast_ತೂಕ_ne[bcast_dim_cnt] = 1;
            bcast_ಒಳಹರಿವು_nb[bcast_dim_cnt]  = ಒಳಹರಿವು_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_ತೂಕ_nb[bcast_dim_cnt] = ತೂಕ_nb[i];
            bcast_dim_cnt++;

            bcast_ಒಳಹರಿವು_ne[bcast_dim_cnt]  = ಒಳಹರಿವು_ne[i] / nr;
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i] / nr;
            bcast_ತೂಕ_ne[bcast_dim_cnt] = ತೂಕ_ne[i];
            bcast_ಒಳಹರಿವು_nb[bcast_dim_cnt]  = bcast_ಒಳಹರಿವು_nb[bcast_dim_cnt - 1] * bcast_ಒಳಹರಿವು_ne[bcast_dim_cnt - 1];
            bcast_dst_nb[bcast_dim_cnt]    = bcast_dst_nb[bcast_dim_cnt - 1] * bcast_dst_ne[bcast_dim_cnt - 1];
            bcast_ತೂಕ_nb[bcast_dim_cnt] = bcast_ತೂಕ_nb[bcast_dim_cnt - 1] * bcast_ತೂಕ_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    ಹಿಂತಿರುಗಿ bcast_dim_cnt;
}
