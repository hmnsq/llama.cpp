#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"
#అన్తర్భావ "ggml.h"

#అన్తర్భావ <cmath>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/interpolate.hpp>
#అన్తర్భావ <openvino/op/matmul.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_upscale(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    ఉపయోగించి Interpolate = ov::op::v4::Interpolate;
    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);

    స్థిరరాశి స్వయంచాలక ఆదానం_shape = context.get_ఆదానం_shape(0).గమ్యం_shape();
    స్థిరరాశి స్వయంచాలక output_shape = context.get_output_shape().గమ్యం_shape();

    ఒకవేళ (ఆదానం_shape == output_shape) {
        తిరిగి rename_outputs_with_suffix({ఆదానం}, context.get_name());
    }

    ov::Output<ov::Node> res = ఆదానం;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    కొరకు (size_t axis = 0; axis < 2; ++axis) {
        స్థిరరాశి size_t లోపల_dim = ఆదానం_shape[axis];
        స్థిరరాశి size_t బహిర్గతం_dim = output_shape[axis];
        ఒకవేళ (లోపల_dim != బహిర్గతం_dim) {
            std::vector<int64_t> indices(బహిర్గతం_dim);
            కొరకు (size_t i = 0; i < బహిర్గతం_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * లోపల_dim) / బహిర్గతం_dim);
            }
            స్వయంచాలక indices_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            స్వయంచాలక axis_కణుపు = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_కణుపు, axis_కణుపు);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    ఒకవేళ (ఆదానం_shape[2] != output_shape[2] || ఆదానం_shape[3] != output_shape[3]) {
        స్థిరరాశి int32_t * op_params = context.get_output_op_params();
        స్థిరరాశి int32_t బహుళకం_flags = op_params != శూన్యసూచిక ? op_params[0] : 0;

        స్థిరరాశి పూర్ణాంకం op_case = context.get_op_case();
        స్థిరరాశి తార్కిక align_corners = (బహుళకం_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        స్థిరరాశి తార్కిక antialias = (బహుళకం_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        ఒకవేళ (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            స్థిరరాశి size_t H_లోపల = ఆదానం_shape[2];
            స్థిరరాశి size_t W_లోపల = ఆదానం_shape[3];
            స్థిరరాశి size_t H_బహిర్గతం = output_shape[2];
            స్థిరరాశి size_t W_బహిర్గతం = output_shape[3];

            స్థిరరాశి దశాంశం pixel_స్థానభ్రంశం = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<దశాంశం> Wy_దత్తాంశం(H_బహిర్గతం * H_లోపల, 0.0f);
            స్థిరరాశి దశాంశం sf1 = static_cast<దశాంశం>(H_బహిర్గతం) / H_లోపల;
            స్థిరరాశి దశాంశం support1 = std::max(1.0f, 1.0f / sf1);
            స్థిరరాశి దశాంశం invscale1 = 1.0f / support1;

            కొరకు (size_t i1 = 0; i1 < H_బహిర్గతం; ++i1) {
                స్థిరరాశి దశాంశం y = (static_cast<దశాంశం>(i1) + pixel_స్థానభ్రంశం) / sf1;
                స్థిరరాశి స్వయంచాలక y_ప్రారంభం = static_cast<int64_t>(y - support1 + pixel_స్థానభ్రంశం);
                స్థిరరాశి size_t y_min = y_ప్రారంభం > 0 ? static_cast<size_t>(y_ప్రారంభం) : 0;
                స్థిరరాశి స్వయంచాలక y_చివర = static_cast<int64_t>(y + support1 + pixel_స్థానభ్రంశం);
                స్థిరరాశి size_t y_max = y_చివర > 0 ? std::min<size_t>(static_cast<size_t>(y_చివర), H_లోపల) : 0;

                దశాంశం మొత్తం_బరువు = 0.0f;
                కొరకు (size_t sy = y_min; sy < y_max; ++sy) {
                    దశాంశం diff = std::నిరపేక్ష((static_cast<దశాంశం>(sy) - y + pixel_స్థానభ్రంశం) * invscale1);
                    దశాంశం బరువు = std::max(1.0f - diff, 0.0f);
                    Wy_దత్తాంశం[i1 * H_లోపల + sy] = బరువు;
                    మొత్తం_బరువు += బరువు;
                }
                ఒకవేళ (మొత్తం_బరువు > 0.0f) {
                    కొరకు (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_దత్తాంశం[i1 * H_లోపల + sy] /= మొత్తం_బరువు;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<దశాంశం> Wx_దత్తాంశం(W_లోపల * W_బహిర్గతం, 0.0f);
            స్థిరరాశి దశాంశం sf0 = static_cast<దశాంశం>(W_బహిర్గతం) / W_లోపల;
            స్థిరరాశి దశాంశం support0 = std::max(1.0f, 1.0f / sf0);
            స్థిరరాశి దశాంశం invscale0 = 1.0f / support0;

            కొరకు (size_t i0 = 0; i0 < W_బహిర్గతం; ++i0) {
                స్థిరరాశి దశాంశం x = (static_cast<దశాంశం>(i0) + pixel_స్థానభ్రంశం) / sf0;
                స్థిరరాశి స్వయంచాలక x_ప్రారంభం = static_cast<int64_t>(x - support0 + pixel_స్థానభ్రంశం);
                స్థిరరాశి size_t x_min = x_ప్రారంభం > 0 ? static_cast<size_t>(x_ప్రారంభం) : 0;
                స్థిరరాశి స్వయంచాలక x_చివర = static_cast<int64_t>(x + support0 + pixel_స్థానభ్రంశం);
                స్థిరరాశి size_t x_max = x_చివర > 0 ? std::min<size_t>(static_cast<size_t>(x_చివర), W_లోపల) : 0;

                దశాంశం మొత్తం_బరువు = 0.0f;
                కొరకు (size_t sx = x_min; sx < x_max; ++sx) {
                    దశాంశం diff = std::నిరపేక్ష((static_cast<దశాంశం>(sx) - x + pixel_స్థానభ్రంశం) * invscale0);
                    దశాంశం బరువు = std::max(1.0f - diff, 0.0f);
                    Wx_దత్తాంశం[sx * W_బహిర్గతం + i0] = బరువు;
                    మొత్తం_బరువు += బరువు;
                }
                ఒకవేళ (మొత్తం_బరువు > 0.0f) {
                    కొరకు (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_దత్తాంశం[sx * W_బహిర్గతం + i0] /= మొత్తం_బరువు;
                    }
                }
            }

            స్వయంచాలక Wy_కణుపు = ov::op::v0::Constant::create(ov::element::f32, {H_బహిర్గతం, H_లోపల}, Wy_దత్తాంశం);
            స్వయంచాలక Wx_కణుపు = ov::op::v0::Constant::create(ov::element::f32, {W_లోపల, W_బహిర్గతం}, Wx_దత్తాంశం);

            స్వయంచాలక res_y = std::make_shared<ov::op::v0::MatMul>(Wy_కణుపు, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_కణుపు);
        } లేకపోతే {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_బహుళకం = Interpolate::Shape_Calc_బహుళకం::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_చివర = {0, 0, 0, 0};

            ఎంపిక (op_case) {
            స్థితి 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::Interpolate_బహుళకం::సమీప;
                attrs.సమీప_బహుళకం = Interpolate::సమీప_బహుళకం_2::FLOOR;
                attrs.coordinate_transformation_బహుళకం = Interpolate::Coordinate_Transform_బహుళకం::ASYMMETRIC;
                ఆపు;
            స్థితి 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::Interpolate_బహుళకం::రేఖీయ;
                attrs.coordinate_transformation_బహుళకం = align_corners
                    ? Interpolate::Coordinate_Transform_బహుళకం::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_బహుళకం::HALF_PIXEL;
                ఆపు;
            స్థితి 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::Interpolate_బహుళకం::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_బహుళకం = align_corners
                    ? Interpolate::Coordinate_Transform_బహుళకం::ALIGN_CORNERS
                    : Interpolate::Coordinate_Transform_బహుళకం::HALF_PIXEL;
                ఆపు;
            అప్రమేయ:
                FRONT_చివర_OP_CONVERSION_CHECK(అబద్ధం, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> లక్ష్యం_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<దశాంశం> scales_vec = {
                static_cast<దశాంశం>(output_shape[2]) / static_cast<దశాంశం>(ఆదానం_shape[2]),
                static_cast<దశాంశం>(output_shape[3]) / static_cast<దశాంశం>(ఆదానం_shape[3]),
            };

            స్వయంచాలక లక్ష్యం_shape_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {2}, లక్ష్యం_shape_vec);
            స్వయంచాలక scales_కణుపు = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            స్వయంచాలక axes_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, లక్ష్యం_shape_కణుపు, scales_కణుపు, axes_కణుపు, attrs);
        }
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
