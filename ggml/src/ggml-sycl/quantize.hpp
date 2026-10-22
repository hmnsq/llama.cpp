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

#ہدایت once

#ضمن <sycl/nd_item.hpp>

#ضمن "ggml-sycl/dpct/helper.hpp"

سانچا <صحيح Elements_فی_WI>
__dpct_inline__ ساكن فراغ quantize_q8_1_impl(ثابت عائم * __restrict__ x,
                                               sycl::vec<int8_t, Elements_فی_WI> & quantized_اقدار, عائم & d,
                                               عائم & جوڑ, ثابت sycl::nd_شے<1> & it) {
    خودکار subgroup_id = it.get_گروہ(0);
    خودکار wi_id       = it.get_local_id(0);

    sycl::vec<عائم, Elements_فی_WI> wi_f32_vals;

    خودکار float_ptr_وتھ = subgroup_id * QK8_1 + Elements_فی_WI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ثابت sycl::vec<عائم, Elements_فی_WI> *>(x + float_ptr_وتھ);

    عائم amax = 0.0f;

#ہدایت unroll(Elements_فی_WI)
    لكل (صحيح i = 0; i < Elements_فی_WI; i++) {
        جوڑ += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_اقدار[i] = 0;
    }
    جوڑ  = sycl::reduce_over_گروہ(it.get_sub_گروہ(), جوڑ, sycl::plus<عائم>());
    amax = sycl::reduce_over_گروہ(it.get_sub_گروہ(), amax, sycl::ودھ_توں_ودھ<عائم>());
    d    = amax == 0 ? 1 : amax / 127;

#ہدایت unroll(Elements_فی_WI)
    لكل (صحيح i = 0; i < Elements_فی_WI; i++) {
        quantized_اقدار[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
سانچا <صحيح Elements_فی_WI> بنية no_quantize_q8_1 {
    فراغ عامل()(ثابت عائم *, فراغ *, صحيح, صحيح, ثابت sycl::nd_شے<1> &) ثابت {}
};

سانچا <صحيح Elements_فی_WI> بنية quantize_and_reorder_q8_1_soa {
    __dpct_inline__ فراغ عامل()(ثابت عائم * __restrict__ x, فراغ * reordered_q8_tensor, ثابت صحيح kx,
                                    ثابت صحيح kx_padded, ثابت sycl::nd_شے<1> & it) ثابت {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        خودکار subgroup_id = it.get_گروہ(0);
        خودکار wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, Elements_فی_WI> quantized_اقدار;
        عائم                            d   = 0.0f;
        عائم                            جوڑ = 0.0f;
        quantize_q8_1_impl<Elements_فی_WI>(x, quantized_اقدار, d, جوڑ, it);

        ثابت صحيح عدد_بلاکاں_فی_صف = kx / QK8_1;
        خودکار      صف                = subgroup_id / عدد_بلاکاں_فی_صف;
        خودکار      خانہ                = subgroup_id % عدد_بلاکاں_فی_صف;
        خودکار      صف_وتھ         = صف * (kx_padded / QK8_1) * حجم(بلاک_q8_1);
        خودکار      خانہ_وتھ         = QK8_1 * خانہ + wi_id * Elements_فی_WI;

        خودکار quant_ptr = (int8_t *) ((حرف *) reordered_q8_tensor + صف_وتھ + خانہ_وتھ);
        *reinterpret_cast<sycl::vec<int8_t, Elements_فی_WI> *>(quant_ptr) = quantized_اقدار;

        خودکار ds_ptr = (sycl::half2 *) ((حرف *) reordered_q8_tensor + صف_وتھ + kx + خانہ * حجم(sycl::half2));
        إذا (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(جوڑ));
        }
    }
};

سانچا <صحيح Elements_فی_WI> بنية quantize_q8_1 {
    __dpct_inline__ فراغ عامل()(ثابت عائم * __restrict__ x, فراغ * q8_tensor, ثابت صحيح kx, ثابت صحيح kx_padded,
                                    ثابت sycl::nd_شے<1> & it) ثابت {
        خودکار subgroup_id = it.get_گروہ(0);
        خودکار wi_id       = it.get_local_id(0);

        ثابت صحيح عدد_بلاکاں_فی_صف = kx / QK8_1;
        خودکار      صف                = subgroup_id / عدد_بلاکاں_فی_صف;
        ثابت صحيح pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, Elements_فی_WI> quantized_اقدار;
        عائم                            d   = 0.0f;
        عائم                            جوڑ = 0.0f;
        quantize_q8_1_impl<Elements_فی_WI>(x, quantized_اقدار, d, جوڑ, it);

        بلاک_q8_1 * quant_ptr = (بلاک_q8_1 *) q8_tensor;
        خودکار         بلاک_id  = subgroup_id % عدد_بلاکاں_فی_صف + صف * pitch;

        int8_t * qs                                               = &(quant_ptr[بلاک_id].qs[wi_id * Elements_فی_WI]);
        *reinterpret_cast<sycl::vec<int8_t, Elements_فی_WI> *>(qs) = quantized_اقدار;
        إذا (wi_id == 0) {
            quant_ptr[بلاک_id].ds = sycl::half2(sycl::half(d), sycl::half(جوڑ));
        }
    }
};

سانچا <سانچا <صحيح> قسم_ناں quantize_f>
فراغ quantize_صف_q8_1_sycl(ثابت عائم * x, فراغ * vy, ثابت صحيح kx, ثابت صحيح ky, ثابت صحيح kx_padded,
                            dpct::قطار_ptr stream) {
    static_assert(QK8_1 % WARP_ناپ == 0);
    خودکار local_range      = std::size_t(WARP_ناپ);
    خودکار عدد_quant_بلاکاں = ky * (kx / QK8_1);
    خودکار global_range     = عدد_quant_بلاکاں * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_شے<1> it) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             quantize_f<QK8_1 / WARP_ناپ>()(x, vy, kx, kx_padded, it);
                         });
}
