#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"
#شامل "ggml.h"

#شامل <cmath>
#شامل <cstddef>
#شامل <memory>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/interpolate.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_upscale(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    استعمال Interpolate = ov::op::v4::Interpolate;
    خودکار داخلہ = process_view_داخلہ_new(context, 0);

    ثابت خودکار داخلہ_shape = context.get_داخلہ_shape(0).مقصد_shape();
    ثابت خودکار output_shape = context.get_output_shape().مقصد_shape();

    اگر (داخلہ_shape == output_shape) {
        واپس rename_outputs_with_suffix({داخلہ}, context.get_name());
    }

    ov::Output<ov::Node> res = داخلہ;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    برائے (size_t axis = 0; axis < 2; ++axis) {
        ثابت size_t اندر_dim = داخلہ_shape[axis];
        ثابت size_t برآمد_dim = output_shape[axis];
        اگر (اندر_dim != برآمد_dim) {
            std::vector<int64_t> indices(برآمد_dim);
            برائے (size_t i = 0; i < برآمد_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * اندر_dim) / برآمد_dim);
            }
            خودکار indices_گرہ = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            خودکار axis_گرہ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_گرہ, axis_گرہ);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    اگر (داخلہ_shape[2] != output_shape[2] || داخلہ_shape[3] != output_shape[3]) {
        ثابت int32_t * op_params = context.get_output_op_params();
        ثابت int32_t منوال_flags = op_params != خالی_اشاریہ ? op_params[0] : 0;

        ثابت صحیح op_case = context.get_op_case();
        ثابت منطقی align_corners = (منوال_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ثابت منطقی antialias = (منوال_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        اگر (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ثابت size_t H_اندر = داخلہ_shape[2];
            ثابت size_t W_اندر = داخلہ_shape[3];
            ثابت size_t H_برآمد = output_shape[2];
            ثابت size_t W_برآمد = output_shape[3];

            ثابت اعشاری pixel_انحراف = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<اعشاری> Wy_ڈیٹا(H_برآمد * H_اندر, 0.0f);
            ثابت اعشاری sf1 = static_cast<اعشاری>(H_برآمد) / H_اندر;
            ثابت اعشاری support1 = std::max(1.0f, 1.0f / sf1);
            ثابت اعشاری invscale1 = 1.0f / support1;

            برائے (size_t i1 = 0; i1 < H_برآمد; ++i1) {
                ثابت اعشاری y = (static_cast<اعشاری>(i1) + pixel_انحراف) / sf1;
                ثابت خودکار y_آغاز = static_cast<int64_t>(y - support1 + pixel_انحراف);
                ثابت size_t y_min = y_آغاز > 0 ? static_cast<size_t>(y_آغاز) : 0;
                ثابت خودکار y_انتہا = static_cast<int64_t>(y + support1 + pixel_انحراف);
                ثابت size_t y_max = y_انتہا > 0 ? std::min<size_t>(static_cast<size_t>(y_انتہا), H_اندر) : 0;

                اعشاری کل_وزن = 0.0f;
                برائے (size_t sy = y_min; sy < y_max; ++sy) {
                    اعشاری diff = std::مطلق_قدر((static_cast<اعشاری>(sy) - y + pixel_انحراف) * invscale1);
                    اعشاری وزن = std::max(1.0f - diff, 0.0f);
                    Wy_ڈیٹا[i1 * H_اندر + sy] = وزن;
                    کل_وزن += وزن;
                }
                اگر (کل_وزن > 0.0f) {
                    برائے (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_ڈیٹا[i1 * H_اندر + sy] /= کل_وزن;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<اعشاری> Wx_ڈیٹا(W_اندر * W_برآمد, 0.0f);
            ثابت اعشاری sf0 = static_cast<اعشاری>(W_برآمد) / W_اندر;
            ثابت اعشاری support0 = std::max(1.0f, 1.0f / sf0);
            ثابت اعشاری invscale0 = 1.0f / support0;

            برائے (size_t i0 = 0; i0 < W_برآمد; ++i0) {
                ثابت اعشاری x = (static_cast<اعشاری>(i0) + pixel_انحراف) / sf0;
                ثابت خودکار x_آغاز = static_cast<int64_t>(x - support0 + pixel_انحراف);
                ثابت size_t x_min = x_آغاز > 0 ? static_cast<size_t>(x_آغاز) : 0;
                ثابت خودکار x_انتہا = static_cast<int64_t>(x + support0 + pixel_انحراف);
                ثابت size_t x_max = x_انتہا > 0 ? std::min<size_t>(static_cast<size_t>(x_انتہا), W_اندر) : 0;

                اعشاری کل_وزن = 0.0f;
                برائے (size_t sx = x_min; sx < x_max; ++sx) {
                    اعشاری diff = std::مطلق_قدر((static_cast<اعشاری>(sx) - x + pixel_انحراف) * invscale0);
                    اعشاری وزن = std::max(1.0f - diff, 0.0f);
                    Wx_ڈیٹا[sx * W_برآمد + i0] = وزن;
                    کل_وزن += وزن;
                }
                اگر (کل_وزن > 0.0f) {
                    برائے (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_ڈیٹا[sx * W_برآمد + i0] /= کل_وزن;
                    }
                }
            }

            خودکار Wy_گرہ = ov::op::v0::Constant::create(ov::element::f32, {H_برآمد, H_اندر}, Wy_ڈیٹا);
            خودکار Wx_گرہ = ov::op::v0::Constant::create(ov::element::f32, {W_اندر, W_برآمد}, Wx_ڈیٹا);

            خودکار res_y = std::make_shared<ov::op::v0::MatMul>(Wy_گرہ, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_گرہ);
        } ورنہ {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_منوال = Interpolate::Shape_Calc_منوال::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_انتہا = {0, 0, 0, 0};

            انتخاب (op_case) {
            حالت 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::Interpolate_منوال::قریب_ترین;
                attrs.قریب_ترین_منوال = Interpolate::قریب_ترین_منوال_2::FLOOR;
                attrs.coordinate_transformation_منوال = Interpolate::Coordinate_Transform_منوال::ASYMMETRIC;
                توڑو;
            حالت 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::Interpolate_منوال::خطی;
                attrs.coordinate_transformation_منوال = align_corners
                    ? Interpolate::Coordinate_Transform_منوال::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_منوال::HALF_PIXEL;
                توڑو;
            حالت 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::Interpolate_منوال::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_منوال = align_corners
                    ? Interpolate::Coordinate_Transform_منوال::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_منوال::HALF_PIXEL;
                توڑو;
            طےشدہ:
                FRONT_انتہا_OP_CONVERSION_CHECK(جھوٹ, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> ہدف_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<اعشاری> scales_vec = {
                static_cast<اعشاری>(output_shape[2]) / static_cast<اعشاری>(داخلہ_shape[2]),
                static_cast<اعشاری>(output_shape[3]) / static_cast<اعشاری>(داخلہ_shape[3]),
            };

            خودکار ہدف_shape_گرہ = ov::op::v0::Constant::create(ov::element::i64, {2}, ہدف_shape_vec);
            خودکار scales_گرہ = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            خودکار axes_گرہ = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, ہدف_shape_گرہ, scales_گرہ, axes_گرہ, attrs);
        }
    }

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
