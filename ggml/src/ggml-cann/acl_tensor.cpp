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

#ਅਨ੍ਤਰ੍ਭਾਵ "acl_tensor.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>

acl_ਅੰਕੜੇ_Type ggml_cann_type_mapping(ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਵਾਪਸ ACL_FLOAT;
        ਹਾਲਤ GGML_TYPE_F16:
            ਵਾਪਸ ACL_FLOAT16;
        ਹਾਲਤ GGML_TYPE_BF16:
            ਵਾਪਸ ACL_BF16;
        ਹਾਲਤ GGML_TYPE_I8:
            ਵਾਪਸ ACL_INT8;
        ਹਾਲਤ GGML_TYPE_I16:
            ਵਾਪਸ ACL_INT16;
        ਹਾਲਤ GGML_TYPE_I32:
            ਵਾਪਸ ACL_INT32;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਵਾਪਸ ACL_INT4;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ ACL_INT8;
        ਹਾਲਤ GGML_TYPE_I64:
            ਵਾਪਸ ACL_INT64;
        ਮੂਲ:
            ਵਾਪਸ ACL_DT_UNDEFINED;
    }
}

acl_tensor_ptr ggml_cann_create_tensor(ਅਚਲ ggml_tensor * tensor,
                                       int64_t *           ne,
                                       size_t *            nb,
                                       int64_t             dims,
                                       aclFormat           format,
                                       size_t              ਵਿੱਥ) {
    // If tensor is bcasted, Up to GGML_MAX_DIMS additional dimensions will be
    // added.
    int64_t acl_ne[GGML_MAX_DIMS * 2], acl_stride[GGML_MAX_DIMS * 2];

    ਜੇ (ne == ਖਾਲੀ_ਸੂਚਕ) {
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
            acl_ne[i]     = tensor->ne[i];
            // The step size of acl is in elements.
            acl_stride[i] = tensor->nb[i] / ggml_element_ਮਾਪ(tensor);
        }
    } ਨਹੀਂਤਾਂ {
        // With bcast
        ਲਈ (ਪੂਰਨ i = 0; i < dims; i++) {
            acl_ne[i]     = ne[i];
            acl_stride[i] = nb[i] / ggml_element_ਮਾਪ(tensor);
        }
    }

    int64_t final_dims      = (dims == 0 ? GGML_MAX_DIMS : dims);
    int64_t acl_storage_len = 1;
    ਲਈ (ਪੂਰਨ i = 0; i < final_dims; i++) {
        acl_storage_len += (acl_ne[i] - 1) * acl_stride[i];
    }
    size_t elem_ਵਿੱਥ = ਵਿੱਥ / ggml_element_ਮਾਪ(tensor);
    acl_storage_len += elem_ਵਿੱਥ;

    // Reverse ne and stride.
    std::reverse(acl_ne, acl_ne + final_dims);
    std::reverse(acl_stride, acl_stride + final_dims);

    aclTensor * raw = aclCreateTensor(acl_ne, final_dims, ggml_cann_type_mapping(tensor->type), acl_stride, elem_ਵਿੱਥ,
                                      format, &acl_storage_len, 1, tensor->ਅੰਕੜੇ);

    ਵਾਪਸ acl_tensor_ptr(raw);
}

acl_int_array_ptr ggml_cann_create_int_array(ਅਚਲ int64_t * value, uint64_t size) {
    aclIntArray * raw = aclCreateIntArray(value, size);
    ਵਾਪਸ acl_int_array_ptr(raw);
}

acl_scalar_ptr ggml_cann_create_scalar(ਖਾਲੀ * value, acl_ਅੰਕੜੇ_Type ਅੰਕੜੇ_Type) {
    aclScalar * raw = aclCreateScalar(value, ਅੰਕੜੇ_Type);
    ਵਾਪਸ acl_scalar_ptr(raw);
}

ਤਾਰਕਿਕ ggml_cann_need_bcast(ਅਚਲ ggml_tensor * t0, ਅਚਲ ggml_tensor * t1) {
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
        ਜੇ (t1->ne[i] != t0->ne[i] && t1->ne[i] != 1) {
            ਵਾਪਸ ਸੱਚ;
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

int64_t ggml_cann_get_bcast_shape(ਅਚਲ ggml_tensor * src0,
                                  ਅਚਲ ggml_tensor * src1,
                                  int64_t *           bcast_src0_ne,
                                  int64_t *           bcast_src1_ne,
                                  size_t *            bcast_src0_nb,
                                  size_t *            bcast_src1_nb) {
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    ਪੂਰਨ bcast_dim_cnt = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr                   = src0->ne[i] / src1->ne[i];
        bcast_src0_ne[bcast_dim_cnt] = src0->ne[i] / nr;
        bcast_src1_ne[bcast_dim_cnt] = src1->ne[i];
        bcast_src0_nb[bcast_dim_cnt] = src0->nb[i];
        bcast_src1_nb[bcast_dim_cnt] = src1->nb[i];
        bcast_dim_cnt++;
        ਜੇ (nr != 1) {
            // Need to add an extra dim.
            bcast_src0_ne[bcast_dim_cnt] = nr;
            bcast_src1_ne[bcast_dim_cnt] = 1;
            bcast_src0_nb[bcast_dim_cnt] = bcast_src0_nb[bcast_dim_cnt - 1] * bcast_src0_ne[bcast_dim_cnt - 1];
            bcast_src1_nb[bcast_dim_cnt] = bcast_src1_nb[bcast_dim_cnt - 1] * bcast_src1_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    ਵਾਪਸ bcast_dim_cnt;
}

int64_t ggml_cann_get_mulmat_bcast_shape(ਅਚਲ int64_t * ਨਿਵੇਸ਼_ne,
                                         ਅਚਲ int64_t * ਭਾਰ_ne,
                                         ਅਚਲ int64_t * dst_ne,
                                         ਅਚਲ size_t *  ਨਿਵੇਸ਼_nb,
                                         ਅਚਲ size_t *  ਭਾਰ_nb,
                                         ਅਚਲ size_t *  dst_nb,
                                         int64_t *       bcast_ਨਿਵੇਸ਼_ne,
                                         int64_t *       bcast_ਭਾਰ_ne,
                                         int64_t *       bcast_dst_ne,
                                         size_t *        bcast_ਨਿਵੇਸ਼_nb,
                                         size_t *        bcast_ਭਾਰ_nb,
                                         size_t *        bcast_dst_nb) {
    // input and dst shoule in same shape, except first two dims.
    GGML_ASSERT(ਨਿਵੇਸ਼_ne[2] == dst_ne[2]);
    GGML_ASSERT(ਨਿਵੇਸ਼_ne[3] == dst_ne[3]);

    ਪੂਰਨ bcast_dim_cnt = 0;

    // For mul_mat, a dimension needs to be added before the dimension that
    // weight needs to be expanded to satisfy the bcast rule of matrix
    // multiplication.
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = ਨਿਵੇਸ਼_ne[i] / ਭਾਰ_ne[i];
        // Do not use bcast in the first two dimensions because we only support
        // the bcast batch dimension. Just copy them.
        ਜੇ (i < 2 || nr == 1) {
            bcast_ਨਿਵੇਸ਼_ne[bcast_dim_cnt]  = ਨਿਵੇਸ਼_ne[i];
            bcast_ਭਾਰ_ne[bcast_dim_cnt] = ਭਾਰ_ne[i];
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i];

            bcast_ਨਿਵੇਸ਼_nb[bcast_dim_cnt]  = ਨਿਵੇਸ਼_nb[i];
            bcast_ਭਾਰ_nb[bcast_dim_cnt] = ਭਾਰ_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_dim_cnt++;
        } ਨਹੀਂਤਾਂ {
            // Need to add an extra dim.
            bcast_ਨਿਵੇਸ਼_ne[bcast_dim_cnt]  = nr;
            bcast_dst_ne[bcast_dim_cnt]    = nr;
            bcast_ਭਾਰ_ne[bcast_dim_cnt] = 1;
            bcast_ਨਿਵੇਸ਼_nb[bcast_dim_cnt]  = ਨਿਵੇਸ਼_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_ਭਾਰ_nb[bcast_dim_cnt] = ਭਾਰ_nb[i];
            bcast_dim_cnt++;

            bcast_ਨਿਵੇਸ਼_ne[bcast_dim_cnt]  = ਨਿਵੇਸ਼_ne[i] / nr;
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i] / nr;
            bcast_ਭਾਰ_ne[bcast_dim_cnt] = ਭਾਰ_ne[i];
            bcast_ਨਿਵੇਸ਼_nb[bcast_dim_cnt]  = bcast_ਨਿਵੇਸ਼_nb[bcast_dim_cnt - 1] * bcast_ਨਿਵੇਸ਼_ne[bcast_dim_cnt - 1];
            bcast_dst_nb[bcast_dim_cnt]    = bcast_dst_nb[bcast_dim_cnt - 1] * bcast_dst_ne[bcast_dim_cnt - 1];
            bcast_ਭਾਰ_nb[bcast_dim_cnt] = bcast_ਭਾਰ_nb[bcast_dim_cnt - 1] * bcast_ਭਾਰ_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    ਵਾਪਸ bcast_dim_cnt;
}
