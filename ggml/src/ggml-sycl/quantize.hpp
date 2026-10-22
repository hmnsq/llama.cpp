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

#شامل <sycl/nd_item.hpp>

#شامل "ggml-sycl/dpct/helper.hpp"

سانچہ <صحیح Elements_فی_WI>
__dpct_inline__ ساکن خالی quantize_q8_1_impl(ثابت اعشاری * __restrict__ x,
                                               sycl::vec<int8_t, Elements_فی_WI> & quantized_اقدار, اعشاری & d,
                                               اعشاری & مجموع, ثابت sycl::nd_شے<1> & it) {
    خودکار subgroup_id = it.get_گروہ(0);
    خودکار wi_id       = it.get_local_id(0);

    sycl::vec<اعشاری, Elements_فی_WI> wi_f32_vals;

    خودکار float_ptr_انحراف = subgroup_id * QK8_1 + Elements_فی_WI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ثابت sycl::vec<اعشاری, Elements_فی_WI> *>(x + float_ptr_انحراف);

    اعشاری amax = 0.0f;

#ہدایت unroll(Elements_فی_WI)
    برائے (صحیح i = 0; i < Elements_فی_WI; i++) {
        مجموع += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_اقدار[i] = 0;
    }
    مجموع  = sycl::reduce_over_گروہ(it.get_sub_گروہ(), مجموع, sycl::plus<اعشاری>());
    amax = sycl::reduce_over_گروہ(it.get_sub_گروہ(), amax, sycl::زیادہ_سے_زیادہ<اعشاری>());
    d    = amax == 0 ? 1 : amax / 127;

#ہدایت unroll(Elements_فی_WI)
    برائے (صحیح i = 0; i < Elements_فی_WI; i++) {
        quantized_اقدار[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
سانچہ <صحیح Elements_فی_WI> ساخت no_quantize_q8_1 {
    خالی عامل()(ثابت اعشاری *, خالی *, صحیح, صحیح, ثابت sycl::nd_شے<1> &) ثابت {}
};

سانچہ <صحیح Elements_فی_WI> ساخت quantize_and_reorder_q8_1_soa {
    __dpct_inline__ خالی عامل()(ثابت اعشاری * __restrict__ x, خالی * reordered_q8_tensor, ثابت صحیح kx,
                                    ثابت صحیح kx_padded, ثابت sycl::nd_شے<1> & it) ثابت {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        خودکار subgroup_id = it.get_گروہ(0);
        خودکار wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, Elements_فی_WI> quantized_اقدار;
        اعشاری                            d   = 0.0f;
        اعشاری                            مجموع = 0.0f;
        quantize_q8_1_impl<Elements_فی_WI>(x, quantized_اقدار, d, مجموع, it);

        ثابت صحیح عدد_بلاکیں_فی_صف = kx / QK8_1;
        خودکار      صف                = subgroup_id / عدد_بلاکیں_فی_صف;
        خودکار      ستون                = subgroup_id % عدد_بلاکیں_فی_صف;
        خودکار      صف_انحراف         = صف * (kx_padded / QK8_1) * حجم(بلاک_q8_1);
        خودکار      ستون_انحراف         = QK8_1 * ستون + wi_id * Elements_فی_WI;

        خودکار quant_ptr = (int8_t *) ((حرف *) reordered_q8_tensor + صف_انحراف + ستون_انحراف);
        *reinterpret_cast<sycl::vec<int8_t, Elements_فی_WI> *>(quant_ptr) = quantized_اقدار;

        خودکار ds_ptr = (sycl::half2 *) ((حرف *) reordered_q8_tensor + صف_انحراف + kx + ستون * حجم(sycl::half2));
        اگر (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(مجموع));
        }
    }
};

سانچہ <صحیح Elements_فی_WI> ساخت quantize_q8_1 {
    __dpct_inline__ خالی عامل()(ثابت اعشاری * __restrict__ x, خالی * q8_tensor, ثابت صحیح kx, ثابت صحیح kx_padded,
                                    ثابت sycl::nd_شے<1> & it) ثابت {
        خودکار subgroup_id = it.get_گروہ(0);
        خودکار wi_id       = it.get_local_id(0);

        ثابت صحیح عدد_بلاکیں_فی_صف = kx / QK8_1;
        خودکار      صف                = subgroup_id / عدد_بلاکیں_فی_صف;
        ثابت صحیح pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, Elements_فی_WI> quantized_اقدار;
        اعشاری                            d   = 0.0f;
        اعشاری                            مجموع = 0.0f;
        quantize_q8_1_impl<Elements_فی_WI>(x, quantized_اقدار, d, مجموع, it);

        بلاک_q8_1 * quant_ptr = (بلاک_q8_1 *) q8_tensor;
        خودکار         بلاک_id  = subgroup_id % عدد_بلاکیں_فی_صف + صف * pitch;

        int8_t * qs                                               = &(quant_ptr[بلاک_id].qs[wi_id * Elements_فی_WI]);
        *reinterpret_cast<sycl::vec<int8_t, Elements_فی_WI> *>(qs) = quantized_اقدار;
        اگر (wi_id == 0) {
            quant_ptr[بلاک_id].ds = sycl::half2(sycl::half(d), sycl::half(مجموع));
        }
    }
};

سانچہ <سانچہ <صحیح> قسم_نام quantize_f>
خالی quantize_صف_q8_1_sycl(ثابت اعشاری * x, خالی * vy, ثابت صحیح kx, ثابت صحیح ky, ثابت صحیح kx_padded,
                            dpct::قطار_ptr stream) {
    static_assert(QK8_1 % WARP_سائز == 0);
    خودکار local_range      = std::size_t(WARP_سائز);
    خودکار عدد_quant_بلاکیں = ky * (kx / QK8_1);
    خودکار global_range     = عدد_quant_بلاکیں * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_شے<1> it) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             quantize_f<QK8_1 / WARP_سائز>()(x, vy, kx, kx_padded, it);
                         });
}
