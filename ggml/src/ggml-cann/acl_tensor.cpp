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

#అన్తర్భావ "acl_tensor.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cstring>

acl_దత్తాంశం_Type ggml_cann_type_mapping(ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_F32:
            తిరిగి ACL_FLOAT;
        స్థితి GGML_TYPE_F16:
            తిరిగి ACL_FLOAT16;
        స్థితి GGML_TYPE_BF16:
            తిరిగి ACL_BF16;
        స్థితి GGML_TYPE_I8:
            తిరిగి ACL_INT8;
        స్థితి GGML_TYPE_I16:
            తిరిగి ACL_INT16;
        స్థితి GGML_TYPE_I32:
            తిరిగి ACL_INT32;
        స్థితి GGML_TYPE_Q4_0:
            తిరిగి ACL_INT4;
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి ACL_INT8;
        స్థితి GGML_TYPE_I64:
            తిరిగి ACL_INT64;
        అప్రమేయ:
            తిరిగి ACL_DT_UNDEFINED;
    }
}

acl_tensor_ptr ggml_cann_create_tensor(స్థిరరాశి ggml_tensor * tensor,
                                       int64_t *           ne,
                                       size_t *            nb,
                                       int64_t             dims,
                                       aclFormat           format,
                                       size_t              స్థానభ్రంశం) {
    // If tensor is bcasted, Up to GGML_MAX_DIMS additional dimensions will be
    // added.
    int64_t acl_ne[GGML_MAX_DIMS * 2], acl_stride[GGML_MAX_DIMS * 2];

    ఒకవేళ (ne == శూన్యసూచిక) {
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
            acl_ne[i]     = tensor->ne[i];
            // The step size of acl is in elements.
            acl_stride[i] = tensor->nb[i] / ggml_element_కొలత(tensor);
        }
    } లేకపోతే {
        // With bcast
        కొరకు (పూర్ణాంకం i = 0; i < dims; i++) {
            acl_ne[i]     = ne[i];
            acl_stride[i] = nb[i] / ggml_element_కొలత(tensor);
        }
    }

    int64_t final_dims      = (dims == 0 ? GGML_MAX_DIMS : dims);
    int64_t acl_storage_len = 1;
    కొరకు (పూర్ణాంకం i = 0; i < final_dims; i++) {
        acl_storage_len += (acl_ne[i] - 1) * acl_stride[i];
    }
    size_t elem_స్థానభ్రంశం = స్థానభ్రంశం / ggml_element_కొలత(tensor);
    acl_storage_len += elem_స్థానభ్రంశం;

    // Reverse ne and stride.
    std::reverse(acl_ne, acl_ne + final_dims);
    std::reverse(acl_stride, acl_stride + final_dims);

    aclTensor * raw = aclCreateTensor(acl_ne, final_dims, ggml_cann_type_mapping(tensor->type), acl_stride, elem_స్థానభ్రంశం,
                                      format, &acl_storage_len, 1, tensor->దత్తాంశం);

    తిరిగి acl_tensor_ptr(raw);
}

acl_int_array_ptr ggml_cann_create_int_array(స్థిరరాశి int64_t * value, uint64_t size) {
    aclIntArray * raw = aclCreateIntArray(value, size);
    తిరిగి acl_int_array_ptr(raw);
}

acl_scalar_ptr ggml_cann_create_scalar(శూన్య * value, acl_దత్తాంశం_Type దత్తాంశం_Type) {
    aclScalar * raw = aclCreateScalar(value, దత్తాంశం_Type);
    తిరిగి acl_scalar_ptr(raw);
}

తార్కిక ggml_cann_need_bcast(స్థిరరాశి ggml_tensor * t0, స్థిరరాశి ggml_tensor * t1) {
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
        ఒకవేళ (t1->ne[i] != t0->ne[i] && t1->ne[i] != 1) {
            తిరిగి నిజం;
        }
    }
    తిరిగి అబద్ధం;
}

int64_t ggml_cann_get_bcast_shape(స్థిరరాశి ggml_tensor * src0,
                                  స్థిరరాశి ggml_tensor * src1,
                                  int64_t *           bcast_src0_ne,
                                  int64_t *           bcast_src1_ne,
                                  size_t *            bcast_src0_nb,
                                  size_t *            bcast_src1_nb) {
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    పూర్ణాంకం bcast_dim_cnt = 0;
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr                   = src0->ne[i] / src1->ne[i];
        bcast_src0_ne[bcast_dim_cnt] = src0->ne[i] / nr;
        bcast_src1_ne[bcast_dim_cnt] = src1->ne[i];
        bcast_src0_nb[bcast_dim_cnt] = src0->nb[i];
        bcast_src1_nb[bcast_dim_cnt] = src1->nb[i];
        bcast_dim_cnt++;
        ఒకవేళ (nr != 1) {
            // Need to add an extra dim.
            bcast_src0_ne[bcast_dim_cnt] = nr;
            bcast_src1_ne[bcast_dim_cnt] = 1;
            bcast_src0_nb[bcast_dim_cnt] = bcast_src0_nb[bcast_dim_cnt - 1] * bcast_src0_ne[bcast_dim_cnt - 1];
            bcast_src1_nb[bcast_dim_cnt] = bcast_src1_nb[bcast_dim_cnt - 1] * bcast_src1_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    తిరిగి bcast_dim_cnt;
}

int64_t ggml_cann_get_mulmat_bcast_shape(స్థిరరాశి int64_t * ఆదానం_ne,
                                         స్థిరరాశి int64_t * బరువు_ne,
                                         స్థిరరాశి int64_t * dst_ne,
                                         స్థిరరాశి size_t *  ఆదానం_nb,
                                         స్థిరరాశి size_t *  బరువు_nb,
                                         స్థిరరాశి size_t *  dst_nb,
                                         int64_t *       bcast_ఆదానం_ne,
                                         int64_t *       bcast_బరువు_ne,
                                         int64_t *       bcast_dst_ne,
                                         size_t *        bcast_ఆదానం_nb,
                                         size_t *        bcast_బరువు_nb,
                                         size_t *        bcast_dst_nb) {
    // input and dst shoule in same shape, except first two dims.
    GGML_ASSERT(ఆదానం_ne[2] == dst_ne[2]);
    GGML_ASSERT(ఆదానం_ne[3] == dst_ne[3]);

    పూర్ణాంకం bcast_dim_cnt = 0;

    // For mul_mat, a dimension needs to be added before the dimension that
    // weight needs to be expanded to satisfy the bcast rule of matrix
    // multiplication.
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = ఆదానం_ne[i] / బరువు_ne[i];
        // Do not use bcast in the first two dimensions because we only support
        // the bcast batch dimension. Just copy them.
        ఒకవేళ (i < 2 || nr == 1) {
            bcast_ఆదానం_ne[bcast_dim_cnt]  = ఆదానం_ne[i];
            bcast_బరువు_ne[bcast_dim_cnt] = బరువు_ne[i];
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i];

            bcast_ఆదానం_nb[bcast_dim_cnt]  = ఆదానం_nb[i];
            bcast_బరువు_nb[bcast_dim_cnt] = బరువు_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_dim_cnt++;
        } లేకపోతే {
            // Need to add an extra dim.
            bcast_ఆదానం_ne[bcast_dim_cnt]  = nr;
            bcast_dst_ne[bcast_dim_cnt]    = nr;
            bcast_బరువు_ne[bcast_dim_cnt] = 1;
            bcast_ఆదానం_nb[bcast_dim_cnt]  = ఆదానం_nb[i];
            bcast_dst_nb[bcast_dim_cnt]    = dst_nb[i];
            bcast_బరువు_nb[bcast_dim_cnt] = బరువు_nb[i];
            bcast_dim_cnt++;

            bcast_ఆదానం_ne[bcast_dim_cnt]  = ఆదానం_ne[i] / nr;
            bcast_dst_ne[bcast_dim_cnt]    = dst_ne[i] / nr;
            bcast_బరువు_ne[bcast_dim_cnt] = బరువు_ne[i];
            bcast_ఆదానం_nb[bcast_dim_cnt]  = bcast_ఆదానం_nb[bcast_dim_cnt - 1] * bcast_ఆదానం_ne[bcast_dim_cnt - 1];
            bcast_dst_nb[bcast_dim_cnt]    = bcast_dst_nb[bcast_dim_cnt - 1] * bcast_dst_ne[bcast_dim_cnt - 1];
            bcast_బరువు_nb[bcast_dim_cnt] = bcast_బరువు_nb[bcast_dim_cnt - 1] * bcast_బరువు_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    తిరిగి bcast_dim_cnt;
}
