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

#شامل "acl_tensor.h"

#شامل <algorithm>
#شامل <cstring>

acl_ڈیٹا_Type ggml_cann_type_mapping(ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_F32:
            واپس ACL_FLOAT;
        حالت GGML_TYPE_F16:
            واپس ACL_FLOAT16;
        حالت GGML_TYPE_BF16:
            واپس ACL_BF16;
        حالت GGML_TYPE_I8:
            واپس ACL_INT8;
        حالت GGML_TYPE_I16:
            واپس ACL_INT16;
        حالت GGML_TYPE_I32:
            واپس ACL_INT32;
        حالت GGML_TYPE_Q4_0:
            واپس ACL_INT4;
        حالت GGML_TYPE_Q8_0:
            واپس ACL_INT8;
        حالت GGML_TYPE_I64:
            واپس ACL_INT64;
        طےشدہ:
            واپس ACL_DT_UNDEFINED;
    }
}

acl_tensor_ptr ggml_cann_create_tensor(ثابت ggml_tensor * tensor,
                                       int64_t *           ne,
                                       size_t *            nb,
                                       int64_t             dims,
                                       aclFormat           format,
                                       size_t              انحراف) {
    // If tensor is bcasted, Up to GGML_MAX_DIMS additional dimensions will be
    // added.
    int64_t acl_ne[GGML_MAX_DIMS * 2], acl_stride[GGML_MAX_DIMS * 2];

    اگر (ne == خالی_اشاریہ) {
        برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
            acl_ne[i]     = tensor->ne[i];
            // The step size of acl is in elements.
            acl_stride[i] = tensor->nb[i] / ggml_element_سائز(tensor);
        }
    } ورنہ {
        // With bcast
        برائے (صحیح i = 0; i < dims; i++) {
            acl_ne[i]     = ne[i];
            acl_stride[i] = nb[i] / ggml_element_سائز(tensor);
        }
    }

    int64_t final_dims      = (dims == 0 ? GGML_MAX_DIMS : dims);
    int64_t acl_storage_len = 1;
    برائے (صحیح i = 0; i < final_dims; i++) {
        acl_storage_len += (acl_ne[i] - 1) * acl_stride[i];
    }
    size_t elem_انحراف = انحراف / ggml_element_سائز(tensor);
    acl_storage_len += elem_انحراف;

    // Reverse ne and stride.
    std::reverse(acl_ne, acl_ne + final_dims);
    std::reverse(acl_stride, acl_stride + final_dims);

    aclTensor * raw = aclCreateTensor(acl_ne, final_dims, ggml_cann_type_mapping(tensor->type), acl_stride, elem_انحراف,
                                      format, &acl_storage_len, 1, tensor->ڈیٹا);

    واپس acl_tensor_ptr(raw);
}

acl_int_array_ptr ggml_cann_create_int_array(ثابت int64_t * value, uint64_t size) {
    aclIntArray * raw = aclCreateIntArray(value, size);
    واپس acl_int_array_ptr(raw);
}

acl_scalar_ptr ggml_cann_create_scalar(خالی * value, acl_ڈیٹا_Type ڈیٹا_Type) {
    aclScalar * raw = aclCreateScalar(value, ڈیٹا_Type);
    واپس acl_scalar_ptr(raw);
}

منطقی ggml_cann_need_bcast(ثابت ggml_tensor * t0, ثابت ggml_tensor * t1) {
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        اگر (t1->ne[i] != t0->ne[i] && t1->ne[i] != 1) {
            واپس سچ;
        }
    }
    واپس جھوٹ;
}

int64_t ggml_cann_get_bcast_shape(ثابت ggml_tensor * src0,
                                  ثابت ggml_tensor * src1,
                                  int64_t *           bcast_src0_ne,
                                  int64_t *           bcast_src1_ne,
                                  size_t *            bcast_src0_nb,
                                  size_t *            bcast_src1_nb) {
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    صحیح bcast_dim_cnt = 0;
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr                   = src0->ne[i] / src1->ne[i];
        bcast_src0_ne[bcast_dim_cnt] = src0->ne[i] / nr;
        bcast_src1_ne[bcast_dim_cnt] = src1->ne[i];
        bcast_src0_nb[bcast_dim_cnt] = src0->nb[i];
        bcast_src1_nb[bcast_dim_cnt] = src1->nb[i];
        bcast_dim_cnt++;
        اگر (nr != 1) {
            // Need to add an extra dim.
            bcast_src0_ne[bcast_dim_cnt] = nr;
            bcast_src1_ne[bcast_dim_cnt] = 1;
            bcast_src0_nb[bcast_dim_cnt] = bcast_src0_nb[bcast_dim_cnt - 1] * bcast_src0_ne[bcast_dim_cnt - 1];
            bcast_src1_nb[bcast_dim_cnt] = bcast_src1_nb[bcast_dim_cnt - 1] * bcast_src1_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    واپس bcast_dim_cnt;
}

int64_t ggml_cann_get_mulmat_bcast_shape(ثابت int64_t * داخلہ_ne,
                                         ثابت int64_t * وزن_ne,
                                         ثابت int64_t * dst_ne,
                                         ثابت size_t *  داخلہ_nb,
                                         ثابت size_t *  وزن_nb,
                                         ثابت size_t *  dst_nb,
                                         int64_t *       bcast_داخلہ_ne,
                                         int64_t *       bcast_وزن_ne,
                                         int64_t *       bcast_dst_ne,
                                         size_t *        bcast_داخلہ_nb,
                                         size_t *        bcast_وزن_nb,
                                         size_t *        bcast_dst_nb) {
    // input and dst shoule in same shape, except first two dims.
    GGML_ASSERT(داخلہ_ne[2] == dst_ne[2]);
    GGML_ASSERT(داخلہ_ne[3] == dst_ne[3]);

    صحیح bcast_dim_cnt = 0;

    // For mul_mat, a dimension needs to be added before the dimension that
    // weight needs to be expanded to satisfy the bcast rule of matrix
    // multiplication.
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = داخلہ_ne[i] / وزن_ne[i];
        // Do not use bcast in the first two dimensions because we only support
        // the bcast batch dimension. Just copy them.
        اگر (i < 2 || nr == 1) {
            bcast_داخلہ_ne[bcast_dim_cnt]  = داخلہ_ne[i];
            bcast_وزن_ne[bcast_dim_cnt] = وزن_ne[i];
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i];

            bcast_داخلہ_nb[bcast_dim_cnt]  = داخلہ_nb[i];
            bcast_وزن_nb[bcast_dim_cnt] = وزن_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_dim_cnt++;
        } ورنہ {
            // Need to add an extra dim.
            bcast_داخلہ_ne[bcast_dim_cnt]  = nr;
            bcast_dst_ne[bcast_dim_cnt]    = nr;
            bcast_وزن_ne[bcast_dim_cnt] = 1;
            bcast_داخلہ_nb[bcast_dim_cnt]  = داخلہ_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_وزن_nb[bcast_dim_cnt] = وزن_nb[i];
            bcast_dim_cnt++;

            bcast_داخلہ_ne[bcast_dim_cnt]  = داخلہ_ne[i] / nr;
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i] / nr;
            bcast_وزن_ne[bcast_dim_cnt] = وزن_ne[i];
            bcast_داخلہ_nb[bcast_dim_cnt]  = bcast_داخلہ_nb[bcast_dim_cnt - 1] * bcast_داخلہ_ne[bcast_dim_cnt - 1];
            bcast_dst_nb[bcast_dim_cnt]    = bcast_dst_nb[bcast_dim_cnt - 1] * bcast_dst_ne[bcast_dim_cnt - 1];
            bcast_وزن_nb[bcast_dim_cnt] = bcast_وزن_nb[bcast_dim_cnt - 1] * bcast_وزن_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    واپس bcast_dim_cnt;
}
