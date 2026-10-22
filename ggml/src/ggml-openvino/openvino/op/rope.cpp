#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/split.hpp>
#అన్తర్భావ <openvino/op/subtract.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <openvino/op/variadic_split.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_rope(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 3);

    పూర్ణాంకం op_case = context.get_op_case();

    ov::Output<Node> res;

    స్వయంచాలక దత్తాంశం_కణుపు = process_view_ఆదానం_new(context, 0).get_కణుపు_shared_ptr();
    స్వయంచాలక output_shape = context.get_output_shape().గమ్యం_shape();
    int32_t * op_params = context.get_output_op_params();
    స్థిరరాశి పూర్ణాంకం mode = op_case;
    స్థిరరాశి int64_t తల_dim = static_cast<int64_t>(output_shape[3]);
    స్థిరరాశి int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    స్థిరరాశి int64_t n_dims = configured_n_dims == 0 ? తల_dim : configured_n_dims;
    స్థిరరాశి int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr పూర్ణాంకం TYPE_సాధారణ = 0;
    constexpr పూర్ణాంకం TYPE_NEOX = 1;
    constexpr పూర్ణాంకం TYPE_IMROPE = 2;
    constexpr పూర్ణాంకం TYPE_VISION = 3;
    constexpr పూర్ణాంకం TYPE_MROPE = 4;

    Output<Node> cos_theta_కణుపు;
    Output<Node> sin_theta_కణుపు;
    ఒకవేళ (context.has_ఆదానం("rope_cos")) {
        cos_theta_కణుపు = context.get_ఆదానం("rope_cos");
        sin_theta_కణుపు = context.get_ఆదానం("rope_sin");
    } లేకపోతే {
        std::string cache_కీలకం = "rope_sin_cos";
        కొరకు (పూర్ణాంకం i = 0; i < 15; i++) {
            cache_కీలకం += "_" + std::to_string(op_params[i]);
        }
        ఒకవేళ (context.get_ఆదానం_కొలత() == 3) {
            cache_కీలకం += "_ff_" + context.get_ఆదానం_పేర్లు()[2];
        }
        ఒకవేళ (context.has_ఆదానం(cache_కీలకం + "_cos")) {
            cos_theta_కణుపు = context.get_ఆదానం(cache_కీలకం + "_cos");
            sin_theta_కణుపు = context.get_ఆదానం(cache_కీలకం + "_sin");
        } లేకపోతే {
            స్వయంచాలక inp_pos = context.get_ఆదానం(1).get_కణుపు_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_బరువు;
            ఒకవేళ (context.get_ఆదానం_కొలత() == 3) {
                rope_freqs_బరువు = context.get_ఆదానం(2).get_కణుపు_shared_ptr();
            }
            స్వయంచాలక sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_బరువు, mode, అబద్ధం, తల_dim);
            sin_theta_కణుపు = sin_cos.మొదటి;
            cos_theta_కణుపు = sin_cos.second;
            context.put_shared(cache_కీలకం + "_cos", cos_theta_కణుపు);
            context.put_shared(cache_కీలకం + "_sin", sin_theta_కణుపు);
        }
    }

    స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (దత్తాంశం_కణుపు->get_element_type() != ov::element::f32) {
        దత్తాంశం_కణుపు = std::make_shared<ov::op::v0::Convert>(దత్తాంశం_కణుపు, ov::element::f32);
    }

    స్థిరరాశి int64_t మొత్తం_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_చివర_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_చివర_OP_CONVERSION_CHECK(n_dims > 0 && మొత్తం_rope_dims + n_offs <= తల_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    ఒకవేళ (context.is_stateful()) {
        స్థిరరాశి int64_t batch = static_cast<int64_t>(output_shape[0]);
        స్వయంచాలక r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        దత్తాంశం_కణుపు = std::make_shared<ov::op::v1::Reshape>(దత్తాంశం_కణుపు, r4_shape, అబద్ధం);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    ఒకవేళ (mode == TYPE_సాధారణ) {
        స్వయంచాలక axis_చివరి = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        స్వయంచాలక అడుగు_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        స్థిరరాశి int64_t batch = static_cast<int64_t>(output_shape[0]);
        స్థిరరాశి int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        స్థిరరాశి int64_t half = n_dims / 2;
        స్వయంచాలక rot_ప్రారంభం = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        స్వయంచాలక rot_చివర = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        స్వయంచాలక rot_దత్తాంశం = std::make_shared<ov::op::v8::ముక్క>(దత్తాంశం_కణుపు, rot_ప్రారంభం, rot_చివర, అడుగు_one, axis_చివరి);

        స్వయంచాలక neg_one_f = ov::op::v0::Constant::create(దత్తాంశం_కణుపు->get_element_type(), ov::Shape{}, {-1.0f});

        స్వయంచాలక paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        స్వయంచాలక x_paired = std::make_shared<ov::op::v1::Reshape>(rot_దత్తాంశం, paired_shape, అబద్ధం);

        స్వయంచాలక విభజించు_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        స్వయంచాలక దత్తాంశం_విభజించు = std::make_shared<ov::op::v1::విభజించు_2>(x_paired, విభజించు_axis, 2);
        Output<Node> x0 = దత్తాంశం_విభజించు->outputs()[0];
        Output<Node> x1 = దత్తాంశం_విభజించు->outputs()[1];

        స్వయంచాలక x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        స్వయంచాలక x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{x1_neg, x0}, -1);

        స్వయంచాలక flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        స్వయంచాలక x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, అబద్ధం);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        స్వయంచాలక expand_cos_sin = [&](స్థిరరాశి Output<Node>& cs) {
            స్వయంచాలక cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            స్వయంచాలక bcast_లక్ష్యం = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            స్వయంచాలక bcast = std::make_shared<ov::op::v3::ప్రసారం>(
                cs_unsq, bcast_లక్ష్యం, ov::op::ప్రసారం_Type::BIDIRECTIONAL);
            స్వయంచాలక flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            తిరిగి std::make_shared<ov::op::v1::Reshape>(bcast, flat, నిజం);
        };
        Output<Node> cos_నిండిన = expand_cos_sin(cos_theta_కణుపు);
        Output<Node> sin_నిండిన = expand_cos_sin(sin_theta_కణుపు);

        స్వయంచాలక y1 = std::make_shared<ov::op::v1::Multiply>(rot_దత్తాంశం, cos_నిండిన);
        స్వయంచాలక y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_నిండిన);
        స్వయంచాలక rotated = std::make_shared<ov::op::v1::కూడు>(y1, y2);

        ov::Output_సదిశ concat_parts;
        ఒకవేళ (n_offs > 0) {
            స్వయంచాలక తల_ప్రారంభం = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            స్వయంచాలక తల_చివర = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            స్వయంచాలక తల = std::make_shared<ov::op::v8::ముక్క>(దత్తాంశం_కణుపు, తల_ప్రారంభం, తల_చివర, అడుగు_one, axis_చివరి);
            concat_parts.push_back(తల);
        }
        concat_parts.push_back(rotated);
        ఒకవేళ (n_offs + n_dims < తల_dim) {
            స్వయంచాలక తోక_ప్రారంభం = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            స్వయంచాలక తోక_చివర = ov::op::v0::Constant::create(ov::element::i64, {1}, {తల_dim});
            స్వయంచాలక తోక = std::make_shared<ov::op::v8::ముక్క>(దత్తాంశం_కణుపు, తోక_ప్రారంభం, తోక_చివర, అడుగు_one, axis_చివరి);
            concat_parts.push_back(తోక);
        }
        ఒకవేళ (concat_parts.size() == 1) {
            res = rotated;
        } లేకపోతే {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } లేకపోతే ఒకవేళ (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        స్థిరరాశి int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        స్థిరరాశి int64_t rot_dims = 2 * half;

        ఒకవేళ (mode != TYPE_NEOX) {
            స్వయంచాలక cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_కణుపు = std::make_shared<ov::op::v1::Reshape>(cos_theta_కణుపు, cos_sin_shape, నిజం);
            sin_theta_కణుపు = std::make_shared<ov::op::v1::Reshape>(sin_theta_కణుపు, cos_sin_shape, నిజం);
        }

        స్వయంచాలక axis_చివరి = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        స్వయంచాలక అడుగు_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_దత్తాంశం = దత్తాంశం_కణుపు;
        ఒకవేళ (n_offs > 0 || n_offs + rot_dims < తల_dim) {
            స్వయంచాలక rot_ప్రారంభం = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            స్వయంచాలక rot_చివర = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_దత్తాంశం = std::make_shared<ov::op::v8::ముక్క>(దత్తాంశం_కణుపు, rot_ప్రారంభం, rot_చివర, అడుగు_one, axis_చివరి);
        }

        స్వయంచాలక neg_one_f = ov::op::v0::Constant::create(దత్తాంశం_కణుపు->get_element_type(), ov::Shape{}, {-1.0f});

        స్వయంచాలక విభజించు_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        స్వయంచాలక విభజించు_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        స్వయంచాలక దత్తాంశం_విభజించు = std::make_shared<ov::op::v1::Variadic_విభజించు>(rot_దత్తాంశం, విభజించు_axis, విభజించు_lengths);
        Output<Node> x1 = దత్తాంశం_విభజించు->outputs()[0];
        Output<Node> x2 = దత్తాంశం_విభజించు->outputs()[1];

        స్వయంచాలక x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        స్వయంచాలక x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{x2_neg, x1}, -1);

        స్వయంచాలక cos_నిండిన = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{cos_theta_కణుపు, cos_theta_కణుపు}, -1);
        స్వయంచాలక sin_నిండిన = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{sin_theta_కణుపు, sin_theta_కణుపు}, -1);

        స్వయంచాలక y1 = std::make_shared<ov::op::v1::Multiply>(rot_దత్తాంశం, cos_నిండిన);
        స్వయంచాలక y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_నిండిన);
        స్వయంచాలక rotated = std::make_shared<ov::op::v1::కూడు>(y1, y2);

        ov::Output_సదిశ concat_parts;
        ఒకవేళ (n_offs > 0) {
            స్వయంచాలక తల_ప్రారంభం = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            స్వయంచాలక తల_చివర = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            స్వయంచాలక తల = std::make_shared<ov::op::v8::ముక్క>(దత్తాంశం_కణుపు, తల_ప్రారంభం, తల_చివర, అడుగు_one, axis_చివరి);
            concat_parts.push_back(తల);
        }
        concat_parts.push_back(rotated);
        ఒకవేళ (n_offs + rot_dims < తల_dim) {
            స్వయంచాలక తోక_ప్రారంభం = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            స్వయంచాలక తోక_చివర = ov::op::v0::Constant::create(ov::element::i64, {1}, {తల_dim});
            స్వయంచాలక తోక = std::make_shared<ov::op::v8::ముక్క>(దత్తాంశం_కణుపు, తోక_ప్రారంభం, తోక_చివర, అడుగు_one, axis_చివరి);
            concat_parts.push_back(తోక);
        }
        ఒకవేళ (concat_parts.size() == 1) {
            res = rotated;
        } లేకపోతే {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
