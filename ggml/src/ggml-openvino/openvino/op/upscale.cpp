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

Output_ويڪٽر translate_upscale(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    استعمال Interpolate = ov::op::v4::Interpolate;
    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);

    ثابت خودڪار داخلا_shape = context.get_داخلا_shape(0).منزل_shape();
    ثابت خودڪار output_shape = context.get_output_shape().منزل_shape();

    إذا (داخلا_shape == output_shape) {
        أرجع rename_outputs_with_suffix({داخلا_2}, context.get_name());
    }

    ov::Output<ov::Node> res = داخلا_2;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    لكل (size_t axis = 0; axis < 2; ++axis) {
        ثابت size_t اندريون_dim = داخلا_shape[axis];
        ثابت size_t ٻاهريون_dim = output_shape[axis];
        إذا (اندريون_dim != ٻاهريون_dim) {
            std::vector<int64_t> indices(ٻاهريون_dim);
            لكل (size_t i = 0; i < ٻاهريون_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * اندريون_dim) / ٻاهريون_dim);
            }
            خودڪار indices_ڳنڍ = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            خودڪار axis_ڳنڍ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_ڳنڍ, axis_ڳنڍ);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    إذا (داخلا_shape[2] != output_shape[2] || داخلا_shape[3] != output_shape[3]) {
        ثابت int32_t * op_params = context.get_output_op_params();
        ثابت int32_t گهڻائي_flags = op_params != خالي_اشارو ? op_params[0] : 0;

        ثابت صحيح op_case = context.get_op_case();
        ثابت منطقي align_corners = (گهڻائي_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        ثابت منطقي antialias = (گهڻائي_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        إذا (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            ثابت size_t H_اندريون = داخلا_shape[2];
            ثابت size_t W_اندريون = داخلا_shape[3];
            ثابت size_t H_ٻاهريون = output_shape[2];
            ثابت size_t W_ٻاهريون = output_shape[3];

            ثابت عائم pixel_وٿي = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<عائم> Wy_معلومات(H_ٻاهريون * H_اندريون, 0.0f);
            ثابت عائم sf1 = static_cast<عائم>(H_ٻاهريون) / H_اندريون;
            ثابت عائم support1 = std::max(1.0f, 1.0f / sf1);
            ثابت عائم invscale1 = 1.0f / support1;

            لكل (size_t i1 = 0; i1 < H_ٻاهريون; ++i1) {
                ثابت عائم y = (static_cast<عائم>(i1) + pixel_وٿي) / sf1;
                ثابت خودڪار y_شروعات = static_cast<int64_t>(y - support1 + pixel_وٿي);
                ثابت size_t y_min = y_شروعات > 0 ? static_cast<size_t>(y_شروعات) : 0;
                ثابت خودڪار y_پڇاڙي = static_cast<int64_t>(y + support1 + pixel_وٿي);
                ثابت size_t y_max = y_پڇاڙي > 0 ? std::min<size_t>(static_cast<size_t>(y_پڇاڙي), H_اندريون) : 0;

                عائم ڪل_وزن = 0.0f;
                لكل (size_t sy = y_min; sy < y_max; ++sy) {
                    عائم diff = std::مطلق_قدر((static_cast<عائم>(sy) - y + pixel_وٿي) * invscale1);
                    عائم وزن = std::max(1.0f - diff, 0.0f);
                    Wy_معلومات[i1 * H_اندريون + sy] = وزن;
                    ڪل_وزن += وزن;
                }
                إذا (ڪل_وزن > 0.0f) {
                    لكل (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_معلومات[i1 * H_اندريون + sy] /= ڪل_وزن;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<عائم> Wx_معلومات(W_اندريون * W_ٻاهريون, 0.0f);
            ثابت عائم sf0 = static_cast<عائم>(W_ٻاهريون) / W_اندريون;
            ثابت عائم support0 = std::max(1.0f, 1.0f / sf0);
            ثابت عائم invscale0 = 1.0f / support0;

            لكل (size_t i0 = 0; i0 < W_ٻاهريون; ++i0) {
                ثابت عائم x = (static_cast<عائم>(i0) + pixel_وٿي) / sf0;
                ثابت خودڪار x_شروعات = static_cast<int64_t>(x - support0 + pixel_وٿي);
                ثابت size_t x_min = x_شروعات > 0 ? static_cast<size_t>(x_شروعات) : 0;
                ثابت خودڪار x_پڇاڙي = static_cast<int64_t>(x + support0 + pixel_وٿي);
                ثابت size_t x_max = x_پڇاڙي > 0 ? std::min<size_t>(static_cast<size_t>(x_پڇاڙي), W_اندريون) : 0;

                عائم ڪل_وزن = 0.0f;
                لكل (size_t sx = x_min; sx < x_max; ++sx) {
                    عائم diff = std::مطلق_قدر((static_cast<عائم>(sx) - x + pixel_وٿي) * invscale0);
                    عائم وزن = std::max(1.0f - diff, 0.0f);
                    Wx_معلومات[sx * W_ٻاهريون + i0] = وزن;
                    ڪل_وزن += وزن;
                }
                إذا (ڪل_وزن > 0.0f) {
                    لكل (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_معلومات[sx * W_ٻاهريون + i0] /= ڪل_وزن;
                    }
                }
            }

            خودڪار Wy_ڳنڍ = ov::op::v0::Constant::create(ov::element::f32, {H_ٻاهريون, H_اندريون}, Wy_معلومات);
            خودڪار Wx_ڳنڍ = ov::op::v0::Constant::create(ov::element::f32, {W_اندريون, W_ٻاهريون}, Wx_معلومات);

            خودڪار res_y = std::make_shared<ov::op::v0::MatMul>(Wy_ڳنڍ, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_ڳنڍ);
        } وإلا {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_گهڻائي = Interpolate::Shape_Calc_گهڻائي::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_پڇاڙي = {0, 0, 0, 0};

            بدل (op_case) {
            حالة 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::Interpolate_گهڻائي::سڀ_کان_ويجهو;
                attrs.سڀ_کان_ويجهو_گهڻائي = Interpolate::سڀ_کان_ويجهو_گهڻائي_2::FLOOR;
                attrs.coordinate_transformation_گهڻائي = Interpolate::Coordinate_Transform_گهڻائي::ASYMMETRIC;
                اكسر;
            حالة 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::Interpolate_گهڻائي::لڪيري;
                attrs.coordinate_transformation_گهڻائي = align_corners
                    ? Interpolate::Coordinate_Transform_گهڻائي::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_گهڻائي::HALF_PIXEL;
                اكسر;
            حالة 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::Interpolate_گهڻائي::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_گهڻائي = align_corners
                    ? Interpolate::Coordinate_Transform_گهڻائي::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_گهڻائي::HALF_PIXEL;
                اكسر;
            افتراضي:
                FRONT_پڇاڙي_OP_CONVERSION_CHECK(خطأ, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> هدف_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<عائم> scales_vec = {
                static_cast<عائم>(output_shape[2]) / static_cast<عائم>(داخلا_shape[2]),
                static_cast<عائم>(output_shape[3]) / static_cast<عائم>(داخلا_shape[3]),
            };

            خودڪار هدف_shape_ڳنڍ = ov::op::v0::Constant::create(ov::element::i64, {2}, هدف_shape_vec);
            خودڪار scales_ڳنڍ = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            خودڪار axes_ڳنڍ = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, هدف_shape_ڳنڍ, scales_ڳنڍ, axes_ڳنڍ, attrs);
        }
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
