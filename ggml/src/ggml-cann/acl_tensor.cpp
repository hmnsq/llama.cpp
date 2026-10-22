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

#अन्तर्भाव "acl_tensor.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cstring>

aclDataType ggml_cann_type_mapping(ggml_type type) {
    विकल्प (type) {
        पक्ष GGML_TYPE_F32:
            प्रत्यागम ACL_FLOAT;
        पक्ष GGML_TYPE_F16:
            प्रत्यागम ACL_FLOAT16;
        पक्ष GGML_TYPE_BF16:
            प्रत्यागम ACL_BF16;
        पक्ष GGML_TYPE_I8:
            प्रत्यागम ACL_INT8;
        पक्ष GGML_TYPE_I16:
            प्रत्यागम ACL_INT16;
        पक्ष GGML_TYPE_I32:
            प्रत्यागम ACL_INT32;
        पक्ष GGML_TYPE_Q4_0:
            प्रत्यागम ACL_INT4;
        पक्ष GGML_TYPE_Q8_0:
            प्रत्यागम ACL_INT8;
        पक्ष GGML_TYPE_I64:
            प्रत्यागम ACL_INT64;
        स्वभाव:
            प्रत्यागम ACL_DT_UNDEFINED;
    }
}

acl_tensor_ptr ggml_cann_create_tensor(ध्रुव ggml_tensor * tensor,
                                       int64_t *           ne,
                                       size_t *            nb,
                                       int64_t             dims,
                                       aclFormat           format,
                                       size_t              offset) {
    // If tensor is bcasted, Up to GGML_MAX_DIMS additional dimensions will be
    // added.
    int64_t acl_ne[GGML_MAX_DIMS * 2], acl_stride[GGML_MAX_DIMS * 2];

    जइ (ne == nullptr) {
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
            acl_ne[i]     = tensor->ne[i];
            // The step size of acl is in elements.
            acl_stride[i] = tensor->nb[i] / ggml_element_size(tensor);
        }
    } अण्णहा {
        // With bcast
        प्रति (पूर्णाङ्क i = 0; i < dims; i++) {
            acl_ne[i]     = ne[i];
            acl_stride[i] = nb[i] / ggml_element_size(tensor);
        }
    }

    int64_t final_dims      = (dims == 0 ? GGML_MAX_DIMS : dims);
    int64_t acl_storage_len = 1;
    प्रति (पूर्णाङ्क i = 0; i < final_dims; i++) {
        acl_storage_len += (acl_ne[i] - 1) * acl_stride[i];
    }
    size_t elem_offset = offset / ggml_element_size(tensor);
    acl_storage_len += elem_offset;

    // Reverse ne and stride.
    std::reverse(acl_ne, acl_ne + final_dims);
    std::reverse(acl_stride, acl_stride + final_dims);

    aclTensor * raw = aclCreateTensor(acl_ne, final_dims, ggml_cann_type_mapping(tensor->type), acl_stride, elem_offset,
                                      format, &acl_storage_len, 1, tensor->data);

    प्रत्यागम acl_tensor_ptr(raw);
}

acl_int_array_ptr ggml_cann_create_int_array(ध्रुव int64_t * value, uint64_t size) {
    aclIntArray * raw = aclCreateIntArray(value, size);
    प्रत्यागम acl_int_array_ptr(raw);
}

acl_scalar_ptr ggml_cann_create_scalar(शून्य * value, aclDataType dataType) {
    aclScalar * raw = aclCreateScalar(value, dataType);
    प्रत्यागम acl_scalar_ptr(raw);
}

तार्किक ggml_cann_need_bcast(ध्रुव ggml_tensor * t0, ध्रुव ggml_tensor * t1) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        जइ (t1->ne[i] != t0->ne[i] && t1->ne[i] != 1) {
            प्रत्यागम सच्च;
        }
    }
    प्रत्यागम मिच्छ;
}

int64_t ggml_cann_get_bcast_shape(ध्रुव ggml_tensor * src0,
                                  ध्रुव ggml_tensor * src1,
                                  int64_t *           bcast_src0_ne,
                                  int64_t *           bcast_src1_ne,
                                  size_t *            bcast_src0_nb,
                                  size_t *            bcast_src1_nb) {
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    पूर्णाङ्क bcast_dim_cnt = 0;
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr                   = src0->ne[i] / src1->ne[i];
        bcast_src0_ne[bcast_dim_cnt] = src0->ne[i] / nr;
        bcast_src1_ne[bcast_dim_cnt] = src1->ne[i];
        bcast_src0_nb[bcast_dim_cnt] = src0->nb[i];
        bcast_src1_nb[bcast_dim_cnt] = src1->nb[i];
        bcast_dim_cnt++;
        जइ (nr != 1) {
            // Need to add an extra dim.
            bcast_src0_ne[bcast_dim_cnt] = nr;
            bcast_src1_ne[bcast_dim_cnt] = 1;
            bcast_src0_nb[bcast_dim_cnt] = bcast_src0_nb[bcast_dim_cnt - 1] * bcast_src0_ne[bcast_dim_cnt - 1];
            bcast_src1_nb[bcast_dim_cnt] = bcast_src1_nb[bcast_dim_cnt - 1] * bcast_src1_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    प्रत्यागम bcast_dim_cnt;
}

int64_t ggml_cann_get_mulmat_bcast_shape(ध्रुव int64_t * निवेशः_ne,
                                         ध्रुव int64_t * weight_ne,
                                         ध्रुव int64_t * dst_ne,
                                         ध्रुव size_t *  निवेशः_nb,
                                         ध्रुव size_t *  weight_nb,
                                         ध्रुव size_t *  dst_nb,
                                         int64_t *       bcast_निवेशः_ne,
                                         int64_t *       bcast_weight_ne,
                                         int64_t *       bcast_dst_ne,
                                         size_t *        bcast_निवेशः_nb,
                                         size_t *        bcast_weight_nb,
                                         size_t *        bcast_dst_nb) {
    // input and dst shoule in same shape, except first two dims.
    GGML_ASSERT(निवेशः_ne[2] == dst_ne[2]);
    GGML_ASSERT(निवेशः_ne[3] == dst_ne[3]);

    पूर्णाङ्क bcast_dim_cnt = 0;

    // For mul_mat, a dimension needs to be added before the dimension that
    // weight needs to be expanded to satisfy the bcast rule of matrix
    // multiplication.
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = निवेशः_ne[i] / weight_ne[i];
        // Do not use bcast in the first two dimensions because we only support
        // the bcast batch dimension. Just copy them.
        जइ (i < 2 || nr == 1) {
            bcast_निवेशः_ne[bcast_dim_cnt]  = निवेशः_ne[i];
            bcast_weight_ne[bcast_dim_cnt] = weight_ne[i];
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i];

            bcast_निवेशः_nb[bcast_dim_cnt]  = निवेशः_nb[i];
            bcast_weight_nb[bcast_dim_cnt] = weight_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_dim_cnt++;
        } अण्णहा {
            // Need to add an extra dim.
            bcast_निवेशः_ne[bcast_dim_cnt]  = nr;
            bcast_dst_ne[bcast_dim_cnt]    = nr;
            bcast_weight_ne[bcast_dim_cnt] = 1;
            bcast_निवेशः_nb[bcast_dim_cnt]  = निवेशः_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_weight_nb[bcast_dim_cnt] = weight_nb[i];
            bcast_dim_cnt++;

            bcast_निवेशः_ne[bcast_dim_cnt]  = निवेशः_ne[i] / nr;
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i] / nr;
            bcast_weight_ne[bcast_dim_cnt] = weight_ne[i];
            bcast_निवेशः_nb[bcast_dim_cnt]  = bcast_निवेशः_nb[bcast_dim_cnt - 1] * bcast_निवेशः_ne[bcast_dim_cnt - 1];
            bcast_dst_nb[bcast_dim_cnt]    = bcast_dst_nb[bcast_dim_cnt - 1] * bcast_dst_ne[bcast_dim_cnt - 1];
            bcast_weight_nb[bcast_dim_cnt] = bcast_weight_nb[bcast_dim_cnt - 1] * bcast_weight_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    प्रत्यागम bcast_dim_cnt;
}
