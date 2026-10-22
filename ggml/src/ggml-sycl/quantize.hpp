/***************************************************************************
 *
 *  Copyright (C) 2025 Codeplay Software Ltd.
 *  Copyright (C) 2025 Intel Corporation
 *
 *  MIT License
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  quantize.hpp
 *
 *  Description:
 *     Sycl backend specific quantization functions
 **************************************************************************/

#رهنمود once

#ضمن <sycl/nd_item.hpp>

#ضمن "ggml-sycl/dpct/helper.hpp"

قالب <صحيح Elements_در_هر_WI>
__dpct_inline__ ساكن فراغ quantize_q8_1_impl(ثابت عائم * __restrict__ x,
                                               sycl::vec<int8_t, Elements_در_هر_WI> & quantized_مقادیر, عائم & d,
                                               عائم & جمع, ثابت sycl::nd_قلم<1> & it) {
    خودکار subgroup_id = it.get_گروه(0);
    خودکار wi_id       = it.get_local_id(0);

    sycl::vec<عائم, Elements_در_هر_WI> wi_f32_vals;

    خودکار float_ptr_جابه_جایی = subgroup_id * QK8_1 + Elements_در_هر_WI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ثابت sycl::vec<عائم, Elements_در_هر_WI> *>(x + float_ptr_جابه_جایی);

    عائم amax = 0.0f;

#رهنمود unroll(Elements_در_هر_WI)
    لكل (صحيح i = 0; i < Elements_در_هر_WI; i++) {
        جمع += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_مقادیر[i] = 0;
    }
    جمع  = sycl::reduce_over_گروه(it.get_sub_گروه(), جمع, sycl::plus<عائم>());
    amax = sycl::reduce_over_گروه(it.get_sub_گروه(), amax, sycl::بیشینه<عائم>());
    d    = amax == 0 ? 1 : amax / 127;

#رهنمود unroll(Elements_در_هر_WI)
    لكل (صحيح i = 0; i < Elements_در_هر_WI; i++) {
        quantized_مقادیر[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
قالب <صحيح Elements_در_هر_WI> بنية no_quantize_q8_1 {
    فراغ عملگر()(ثابت عائم *, فراغ *, صحيح, صحيح, ثابت sycl::nd_قلم<1> &) ثابت {}
};

قالب <صحيح Elements_در_هر_WI> بنية quantize_and_reorder_q8_1_soa {
    __dpct_inline__ فراغ عملگر()(ثابت عائم * __restrict__ x, فراغ * reordered_q8_tensor, ثابت صحيح kx,
                                    ثابت صحيح kx_padded, ثابت sycl::nd_قلم<1> & it) ثابت {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        خودکار subgroup_id = it.get_گروه(0);
        خودکار wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, Elements_در_هر_WI> quantized_مقادیر;
        عائم                            d   = 0.0f;
        عائم                            جمع = 0.0f;
        quantize_q8_1_impl<Elements_در_هر_WI>(x, quantized_مقادیر, d, جمع, it);

        ثابت صحيح رقم_بلوکها_در_هر_ردیف = kx / QK8_1;
        خودکار      ردیف                = subgroup_id / رقم_بلوکها_در_هر_ردیف;
        خودکار      ستون                = subgroup_id % رقم_بلوکها_در_هر_ردیف;
        خودکار      ردیف_جابه_جایی         = ردیف * (kx_padded / QK8_1) * حجم(بلوک_q8_1);
        خودکار      ستون_جابه_جایی         = QK8_1 * ستون + wi_id * Elements_در_هر_WI;

        خودکار quant_ptr = (int8_t *) ((حرف *) reordered_q8_tensor + ردیف_جابه_جایی + ستون_جابه_جایی);
        *reinterpret_cast<sycl::vec<int8_t, Elements_در_هر_WI> *>(quant_ptr) = quantized_مقادیر;

        خودکار ds_ptr = (sycl::half2 *) ((حرف *) reordered_q8_tensor + ردیف_جابه_جایی + kx + ستون * حجم(sycl::half2));
        إذا (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(جمع));
        }
    }
};

قالب <صحيح Elements_در_هر_WI> بنية quantize_q8_1 {
    __dpct_inline__ فراغ عملگر()(ثابت عائم * __restrict__ x, فراغ * q8_tensor, ثابت صحيح kx, ثابت صحيح kx_padded,
                                    ثابت sycl::nd_قلم<1> & it) ثابت {
        خودکار subgroup_id = it.get_گروه(0);
        خودکار wi_id       = it.get_local_id(0);

        ثابت صحيح رقم_بلوکها_در_هر_ردیف = kx / QK8_1;
        خودکار      ردیف                = subgroup_id / رقم_بلوکها_در_هر_ردیف;
        ثابت صحيح pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, Elements_در_هر_WI> quantized_مقادیر;
        عائم                            d   = 0.0f;
        عائم                            جمع = 0.0f;
        quantize_q8_1_impl<Elements_در_هر_WI>(x, quantized_مقادیر, d, جمع, it);

        بلوک_q8_1 * quant_ptr = (بلوک_q8_1 *) q8_tensor;
        خودکار         بلوک_id  = subgroup_id % رقم_بلوکها_در_هر_ردیف + ردیف * pitch;

        int8_t * qs                                               = &(quant_ptr[بلوک_id].qs[wi_id * Elements_در_هر_WI]);
        *reinterpret_cast<sycl::vec<int8_t, Elements_در_هر_WI> *>(qs) = quantized_مقادیر;
        إذا (wi_id == 0) {
            quant_ptr[بلوک_id].ds = sycl::half2(sycl::half(d), sycl::half(جمع));
        }
    }
};

قالب <قالب <صحيح> نام_نوع quantize_f>
فراغ quantize_ردیف_q8_1_sycl(ثابت عائم * x, فراغ * vy, ثابت صحيح kx, ثابت صحيح ky, ثابت صحيح kx_padded,
                            dpct::صف_ptr stream) {
    static_assert(QK8_1 % WARP_اندازه == 0);
    خودکار local_range      = std::size_t(WARP_اندازه);
    خودکار رقم_quant_بلوکها = ky * (kx / QK8_1);
    خودکار global_range     = رقم_quant_بلوکها * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_قلم<1> it) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                             quantize_f<QK8_1 / WARP_اندازه>()(x, vy, kx, kx_padded, it);
                         });
}
