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
            फर्काऊ ACL_FLOAT;
        पक्ष GGML_TYPE_F16:
            फर्काऊ ACL_FLOAT16;
        पक्ष GGML_TYPE_BF16:
            फर्काऊ ACL_BF16;
        पक्ष GGML_TYPE_I8:
            फर्काऊ ACL_INT8;
        पक्ष GGML_TYPE_I16:
            फर्काऊ ACL_INT16;
        पक्ष GGML_TYPE_I32:
            फर्काऊ ACL_INT32;
        पक्ष GGML_TYPE_Q4_0:
            फर्काऊ ACL_INT4;
        पक्ष GGML_TYPE_Q8_0:
            फर्काऊ ACL_INT8;
        पक्ष GGML_TYPE_I64:
            फर्काऊ ACL_INT64;
        स्वभाव:
            फर्काऊ ACL_DT_UNDEFINED;
    }
}

acl_tensor_ptr ggml_cann_create_tensor(अचर ggml_tensor * tensor,
                                       int64_t *           ne,
                                       माप_प्रकार *            nb,
                                       int64_t             dims,
                                       aclFormat           format,
                                       माप_प्रकार              offset) {
    // If tensor is bcasted, Up to GGML_MAX_DIMS additional dimensions will be
    // added.
    int64_t acl_ne[GGML_MAX_DIMS * 2], acl_stride[GGML_MAX_DIMS * 2];

    यदि (ne == nullptr) {
        लागि (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
            acl_ne[i]     = tensor->ne[i];
            // The step size of acl is in elements.
            acl_stride[i] = tensor->nb[i] / ggml_element_size(tensor);
        }
    } अन्यथा {
        // With bcast
        लागि (पूर्णाङ्क i = 0; i < dims; i++) {
            acl_ne[i]     = ne[i];
            acl_stride[i] = nb[i] / ggml_element_size(tensor);
        }
    }

    int64_t final_dims      = (dims == 0 ? GGML_MAX_DIMS : dims);
    int64_t acl_storage_len = 1;
    लागि (पूर्णाङ्क i = 0; i < final_dims; i++) {
        acl_storage_len += (acl_ne[i] - 1) * acl_stride[i];
    }
    माप_प्रकार elem_offset = offset / ggml_element_size(tensor);
    acl_storage_len += elem_offset;

    // Reverse ne and stride.
    std::reverse(acl_ne, acl_ne + final_dims);
    std::reverse(acl_stride, acl_stride + final_dims);

    aclTensor * raw = aclCreateTensor(acl_ne, final_dims, ggml_cann_type_mapping(tensor->type), acl_stride, elem_offset,
                                      format, &acl_storage_len, 1, tensor->data);

    फर्काऊ acl_tensor_ptr(raw);
}

acl_int_array_ptr ggml_cann_create_int_array(अचर int64_t * value, uint64_t size) {
    aclIntArray * raw = aclCreateIntArray(value, size);
    फर्काऊ acl_int_array_ptr(raw);
}

acl_scalar_ptr ggml_cann_create_scalar(शून्य * value, aclDataType dataType) {
    aclScalar * raw = aclCreateScalar(value, dataType);
    फर्काऊ acl_scalar_ptr(raw);
}

तार्किक ggml_cann_need_bcast(अचर ggml_tensor * t0, अचर ggml_tensor * t1) {
    लागि (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        यदि (t1->ne[i] != t0->ne[i] && t1->ne[i] != 1) {
            फर्काऊ सत्य;
        }
    }
    फर्काऊ असत्य;
}

int64_t ggml_cann_get_bcast_shape(अचर ggml_tensor * src0,
                                  अचर ggml_tensor * src1,
                                  int64_t *           bcast_src0_ne,
                                  int64_t *           bcast_src1_ne,
                                  माप_प्रकार *            bcast_src0_nb,
                                  माप_प्रकार *            bcast_src1_nb) {
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    पूर्णाङ्क bcast_dim_cnt = 0;
    लागि (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr                   = src0->ne[i] / src1->ne[i];
        bcast_src0_ne[bcast_dim_cnt] = src0->ne[i] / nr;
        bcast_src1_ne[bcast_dim_cnt] = src1->ne[i];
        bcast_src0_nb[bcast_dim_cnt] = src0->nb[i];
        bcast_src1_nb[bcast_dim_cnt] = src1->nb[i];
        bcast_dim_cnt++;
        यदि (nr != 1) {
            // Need to add an extra dim.
            bcast_src0_ne[bcast_dim_cnt] = nr;
            bcast_src1_ne[bcast_dim_cnt] = 1;
            bcast_src0_nb[bcast_dim_cnt] = bcast_src0_nb[bcast_dim_cnt - 1] * bcast_src0_ne[bcast_dim_cnt - 1];
            bcast_src1_nb[bcast_dim_cnt] = bcast_src1_nb[bcast_dim_cnt - 1] * bcast_src1_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    फर्काऊ bcast_dim_cnt;
}

int64_t ggml_cann_get_mulmat_bcast_shape(अचर int64_t * आगत_ne,
                                         अचर int64_t * weight_ne,
                                         अचर int64_t * dst_ne,
                                         अचर माप_प्रकार *  आगत_nb,
                                         अचर माप_प्रकार *  weight_nb,
                                         अचर माप_प्रकार *  dst_nb,
                                         int64_t *       bcast_आगत_ne,
                                         int64_t *       bcast_weight_ne,
                                         int64_t *       bcast_dst_ne,
                                         माप_प्रकार *        bcast_आगत_nb,
                                         माप_प्रकार *        bcast_weight_nb,
                                         माप_प्रकार *        bcast_dst_nb) {
    // input and dst shoule in same shape, except first two dims.
    GGML_ASSERT(आगत_ne[2] == dst_ne[2]);
    GGML_ASSERT(आगत_ne[3] == dst_ne[3]);

    पूर्णाङ्क bcast_dim_cnt = 0;

    // For mul_mat, a dimension needs to be added before the dimension that
    // weight needs to be expanded to satisfy the bcast rule of matrix
    // multiplication.
    लागि (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = आगत_ne[i] / weight_ne[i];
        // Do not use bcast in the first two dimensions because we only support
        // the bcast batch dimension. Just copy them.
        यदि (i < 2 || nr == 1) {
            bcast_आगत_ne[bcast_dim_cnt]  = आगत_ne[i];
            bcast_weight_ne[bcast_dim_cnt] = weight_ne[i];
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i];

            bcast_आगत_nb[bcast_dim_cnt]  = आगत_nb[i];
            bcast_weight_nb[bcast_dim_cnt] = weight_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_dim_cnt++;
        } अन्यथा {
            // Need to add an extra dim.
            bcast_आगत_ne[bcast_dim_cnt]  = nr;
            bcast_dst_ne[bcast_dim_cnt]    = nr;
            bcast_weight_ne[bcast_dim_cnt] = 1;
            bcast_आगत_nb[bcast_dim_cnt]  = आगत_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_weight_nb[bcast_dim_cnt] = weight_nb[i];
            bcast_dim_cnt++;

            bcast_आगत_ne[bcast_dim_cnt]  = आगत_ne[i] / nr;
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i] / nr;
            bcast_weight_ne[bcast_dim_cnt] = weight_ne[i];
            bcast_आगत_nb[bcast_dim_cnt]  = bcast_आगत_nb[bcast_dim_cnt - 1] * bcast_आगत_ne[bcast_dim_cnt - 1];
            bcast_dst_nb[bcast_dim_cnt]    = bcast_dst_nb[bcast_dim_cnt - 1] * bcast_dst_ne[bcast_dim_cnt - 1];
            bcast_weight_nb[bcast_dim_cnt] = bcast_weight_nb[bcast_dim_cnt - 1] * bcast_weight_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    फर्काऊ bcast_dim_cnt;
}
