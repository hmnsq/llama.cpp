#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "ggml.h"

#ضمن <cmath>
#ضمن <cstddef>
#ضمن <memory>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/interpolate.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_وکتور translate_upscale(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    استفاده Interpolate = ov::op::v4::Interpolate;
    خودکار درون_داد = process_view_درون_داد_new(context, 0);

    ثابت خودکار درون_داد_shape = context.get_درون_داد_shape(0).مقصد_shape();
    ثابت خودکار output_shape = context.get_output_shape().مقصد_shape();

    إذا (درون_داد_shape == output_shape) {
        أرجع rename_outputs_with_suffix({درون_داد}, context.get_name());
    }

    ov::Output<ov::Node> res = درون_داد;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    لكل (size_t axis = 0; axis < 2; ++axis) {
        ثابت size_t درون_dim = درون_داد_shape[axis];
        ثابت size_t بیرون_dim = output_shape[axis];
        إذا (درون_dim != بیرون_dim) {
            std::vector<int64_t> indices(بیرون_dim);
            لكل (size_t i = 0; i < بیرون_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * درون_dim) / بیرون_dim);
            }
            خودکار indices_گره = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            خودکار axis_گره = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_گره, axis_گره);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    إذا (درون_داد_shape[2] != output_shape[2] || درون_داد_shape[3] != output_shape[3]) {
        ثابت int32_t * op_params = context.get_output_op_params();
        ثابت int32_t نما_flags = op_params != اشاره_تهی ? op_params[0] : 0;

        ثابت صحيح op_case = context.get_op_case();
        ثابت منطقي align_corners = (نما_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ثابت منطقي antialias = (نما_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        إذا (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ثابت size_t H_درون = درون_داد_shape[2];
            ثابت size_t W_درون = درون_داد_shape[3];
            ثابت size_t H_بیرون = output_shape[2];
            ثابت size_t W_بیرون = output_shape[3];

            ثابت عائم pixel_جابه_جایی = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<عائم> Wy_داده(H_بیرون * H_درون, 0.0f);
            ثابت عائم sf1 = static_cast<عائم>(H_بیرون) / H_درون;
            ثابت عائم support1 = std::max(1.0f, 1.0f / sf1);
            ثابت عائم invscale1 = 1.0f / support1;

            لكل (size_t i1 = 0; i1 < H_بیرون; ++i1) {
                ثابت عائم y = (static_cast<عائم>(i1) + pixel_جابه_جایی) / sf1;
                ثابت خودکار y_شروع = static_cast<int64_t>(y - support1 + pixel_جابه_جایی);
                ثابت size_t y_min = y_شروع > 0 ? static_cast<size_t>(y_شروع) : 0;
                ثابت خودکار y_انتها = static_cast<int64_t>(y + support1 + pixel_جابه_جایی);
                ثابت size_t y_max = y_انتها > 0 ? std::min<size_t>(static_cast<size_t>(y_انتها), H_درون) : 0;

                عائم مجموع_وزن = 0.0f;
                لكل (size_t sy = y_min; sy < y_max; ++sy) {
                    عائم diff = std::قدر_مطلق((static_cast<عائم>(sy) - y + pixel_جابه_جایی) * invscale1);
                    عائم وزن = std::max(1.0f - diff, 0.0f);
                    Wy_داده[i1 * H_درون + sy] = وزن;
                    مجموع_وزن += وزن;
                }
                إذا (مجموع_وزن > 0.0f) {
                    لكل (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_داده[i1 * H_درون + sy] /= مجموع_وزن;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<عائم> Wx_داده(W_درون * W_بیرون, 0.0f);
            ثابت عائم sf0 = static_cast<عائم>(W_بیرون) / W_درون;
            ثابت عائم support0 = std::max(1.0f, 1.0f / sf0);
            ثابت عائم invscale0 = 1.0f / support0;

            لكل (size_t i0 = 0; i0 < W_بیرون; ++i0) {
                ثابت عائم x = (static_cast<عائم>(i0) + pixel_جابه_جایی) / sf0;
                ثابت خودکار x_شروع = static_cast<int64_t>(x - support0 + pixel_جابه_جایی);
                ثابت size_t x_min = x_شروع > 0 ? static_cast<size_t>(x_شروع) : 0;
                ثابت خودکار x_انتها = static_cast<int64_t>(x + support0 + pixel_جابه_جایی);
                ثابت size_t x_max = x_انتها > 0 ? std::min<size_t>(static_cast<size_t>(x_انتها), W_درون) : 0;

                عائم مجموع_وزن = 0.0f;
                لكل (size_t sx = x_min; sx < x_max; ++sx) {
                    عائم diff = std::قدر_مطلق((static_cast<عائم>(sx) - x + pixel_جابه_جایی) * invscale0);
                    عائم وزن = std::max(1.0f - diff, 0.0f);
                    Wx_داده[sx * W_بیرون + i0] = وزن;
                    مجموع_وزن += وزن;
                }
                إذا (مجموع_وزن > 0.0f) {
                    لكل (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_داده[sx * W_بیرون + i0] /= مجموع_وزن;
                    }
                }
            }

            خودکار Wy_گره = ov::op::v0::Constant::create(ov::element::f32, {H_بیرون, H_درون}, Wy_داده);
            خودکار Wx_گره = ov::op::v0::Constant::create(ov::element::f32, {W_درون, W_بیرون}, Wx_داده);

            خودکار res_y = std::make_shared<ov::op::v0::MatMul>(Wy_گره, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_گره);
        } وإلا {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_نما = Interpolate::Shape_Calc_نما::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_انتها = {0, 0, 0, 0};

            بدل (op_case) {
            حالة 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::Interpolate_نما::نزدیکترین;
                attrs.نزدیکترین_نما = Interpolate::نزدیکترین_نما_2::FLOOR;
                attrs.coordinate_transformation_نما = Interpolate::Coordinate_Transform_نما::ASYMMETRIC;
                اكسر;
            حالة 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::Interpolate_نما::خطی;
                attrs.coordinate_transformation_نما = align_corners
                    ? Interpolate::Coordinate_Transform_نما::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_نما::HALF_PIXEL;
                اكسر;
            حالة 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::Interpolate_نما::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_نما = align_corners
                    ? Interpolate::Coordinate_Transform_نما::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_نما::HALF_PIXEL;
                اكسر;
            افتراضي:
                FRONT_انتها_OP_CONVERSION_CHECK(خطأ, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> هدف_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<عائم> scales_vec = {
                static_cast<عائم>(output_shape[2]) / static_cast<عائم>(درون_داد_shape[2]),
                static_cast<عائم>(output_shape[3]) / static_cast<عائم>(درون_داد_shape[3]),
            };

            خودکار هدف_shape_گره = ov::op::v0::Constant::create(ov::element::i64, {2}, هدف_shape_vec);
            خودکار scales_گره = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            خودکار axes_گره = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, هدف_shape_گره, scales_گره, axes_گره, attrs);
        }
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
