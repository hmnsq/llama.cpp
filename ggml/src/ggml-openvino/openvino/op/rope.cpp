#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/split.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <openvino/op/variadic_split.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_rope(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 2, 3);

    पूर्णांक op_case = context.get_op_case();

    ov::Output<Node> res;

    स्वतः data_गाठ = process_view_आदान_new(context, 0).get_गाठ_shared_ptr();
    स्वतः output_shape = context.get_output_shape().to_shape();
    int32_t * op_params = context.get_output_op_params();
    स्थिरांक पूर्णांक mode = op_case;
    स्थिरांक int64_t head_dim = static_cast<int64_t>(output_shape[3]);
    स्थिरांक int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    स्थिरांक int64_t n_dims = configured_n_dims == 0 ? head_dim : configured_n_dims;
    स्थिरांक int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr पूर्णांक TYPE_NORMAL = 0;
    constexpr पूर्णांक TYPE_NEOX = 1;
    constexpr पूर्णांक TYPE_IMROPE = 2;
    constexpr पूर्णांक TYPE_VISION = 3;
    constexpr पूर्णांक TYPE_MROPE = 4;

    Output<Node> cos_theta_गाठ;
    Output<Node> sin_theta_गाठ;
    जर (context.has_आदान("rope_cos")) {
        cos_theta_गाठ = context.get_आदान("rope_cos");
        sin_theta_गाठ = context.get_आदान("rope_sin");
    } अन्यथा {
        std::string cache_किल्ली = "rope_sin_cos";
        साठी (पूर्णांक i = 0; i < 15; i++) {
            cache_किल्ली += "_" + std::to_string(op_params[i]);
        }
        जर (context.get_आदान_size() == 3) {
            cache_किल्ली += "_ff_" + context.get_आदान_names()[2];
        }
        जर (context.has_आदान(cache_किल्ली + "_cos")) {
            cos_theta_गाठ = context.get_आदान(cache_किल्ली + "_cos");
            sin_theta_गाठ = context.get_आदान(cache_किल्ली + "_sin");
        } अन्यथा {
            स्वतः inp_pos = context.get_आदान(1).get_गाठ_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_weight;
            जर (context.get_आदान_size() == 3) {
                rope_freqs_weight = context.get_आदान(2).get_गाठ_shared_ptr();
            }
            स्वतः sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight, mode, असत्य, head_dim);
            sin_theta_गाठ = sin_cos.first;
            cos_theta_गाठ = sin_cos.second;
            context.put_shared(cache_किल्ली + "_cos", cos_theta_गाठ);
            context.put_shared(cache_किल्ली + "_sin", sin_theta_गाठ);
        }
    }

    स्वतः output_type = context.get_output_type();
    जर (data_गाठ->get_element_type() != ov::element::f32) {
        data_गाठ = std::make_shared<ov::op::v0::Convert>(data_गाठ, ov::element::f32);
    }

    स्थिरांक int64_t total_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_END_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_END_OP_CONVERSION_CHECK(n_dims > 0 && total_rope_dims + n_offs <= head_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    जर (context.is_stateful()) {
        स्थिरांक int64_t batch = static_cast<int64_t>(output_shape[0]);
        स्वतः r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        data_गाठ = std::make_shared<ov::op::v1::Reshape>(data_गाठ, r4_shape, असत्य);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    जर (mode == TYPE_NORMAL) {
        स्वतः axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        स्वतः step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        स्थिरांक int64_t batch = static_cast<int64_t>(output_shape[0]);
        स्थिरांक int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        स्थिरांक int64_t half = n_dims / 2;
        स्वतः rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        स्वतः rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        स्वतः rot_data = std::make_shared<ov::op::v8::Slice>(data_गाठ, rot_start, rot_end, step_one, axis_last);

        स्वतः neg_one_f = ov::op::v0::Constant::create(data_गाठ->get_element_type(), ov::Shape{}, {-1.0f});

        स्वतः paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        स्वतः x_paired = std::make_shared<ov::op::v1::Reshape>(rot_data, paired_shape, असत्य);

        स्वतः split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        स्वतः data_split = std::make_shared<ov::op::v1::Split>(x_paired, split_axis, 2);
        Output<Node> x0 = data_split->outputs()[0];
        Output<Node> x1 = data_split->outputs()[1];

        स्वतः x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        स्वतः x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_neg, x0}, -1);

        स्वतः flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        स्वतः x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, असत्य);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        स्वतः expand_cos_sin = [&](स्थिरांक Output<Node>& cs) {
            स्वतः cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            स्वतः bcast_target = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            स्वतः bcast = std::make_shared<ov::op::v3::Broadcast>(
                cs_unsq, bcast_target, ov::op::BroadcastType::BIDIRECTIONAL);
            स्वतः flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            परत std::make_shared<ov::op::v1::Reshape>(bcast, flat, सत्य);
        };
        Output<Node> cos_full = expand_cos_sin(cos_theta_गाठ);
        Output<Node> sin_full = expand_cos_sin(sin_theta_गाठ);

        स्वतः y1 = std::make_shared<ov::op::v1::Multiply>(rot_data, cos_full);
        स्वतः y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_full);
        स्वतः rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        जर (n_offs > 0) {
            स्वतः head_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            स्वतः head_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            स्वतः head = std::make_shared<ov::op::v8::Slice>(data_गाठ, head_start, head_end, step_one, axis_last);
            concat_parts.push_back(head);
        }
        concat_parts.push_back(rotated);
        जर (n_offs + n_dims < head_dim) {
            स्वतः tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            स्वतः tail_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            स्वतः tail = std::make_shared<ov::op::v8::Slice>(data_गाठ, tail_start, tail_end, step_one, axis_last);
            concat_parts.push_back(tail);
        }
        जर (concat_parts.size() == 1) {
            res = rotated;
        } अन्यथा {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } अन्यथा जर (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        स्थिरांक int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        स्थिरांक int64_t rot_dims = 2 * half;

        जर (mode != TYPE_NEOX) {
            स्वतः cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_गाठ = std::make_shared<ov::op::v1::Reshape>(cos_theta_गाठ, cos_sin_shape, सत्य);
            sin_theta_गाठ = std::make_shared<ov::op::v1::Reshape>(sin_theta_गाठ, cos_sin_shape, सत्य);
        }

        स्वतः axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        स्वतः step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_data = data_गाठ;
        जर (n_offs > 0 || n_offs + rot_dims < head_dim) {
            स्वतः rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            स्वतः rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_data = std::make_shared<ov::op::v8::Slice>(data_गाठ, rot_start, rot_end, step_one, axis_last);
        }

        स्वतः neg_one_f = ov::op::v0::Constant::create(data_गाठ->get_element_type(), ov::Shape{}, {-1.0f});

        स्वतः split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        स्वतः split_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        स्वतः data_split = std::make_shared<ov::op::v1::VariadicSplit>(rot_data, split_axis, split_lengths);
        Output<Node> x1 = data_split->outputs()[0];
        Output<Node> x2 = data_split->outputs()[1];

        स्वतः x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        स्वतः x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x2_neg, x1}, -1);

        स्वतः cos_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_theta_गाठ, cos_theta_गाठ}, -1);
        स्वतः sin_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_theta_गाठ, sin_theta_गाठ}, -1);

        स्वतः y1 = std::make_shared<ov::op::v1::Multiply>(rot_data, cos_full);
        स्वतः y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_full);
        स्वतः rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        जर (n_offs > 0) {
            स्वतः head_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            स्वतः head_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            स्वतः head = std::make_shared<ov::op::v8::Slice>(data_गाठ, head_start, head_end, step_one, axis_last);
            concat_parts.push_back(head);
        }
        concat_parts.push_back(rotated);
        जर (n_offs + rot_dims < head_dim) {
            स्वतः tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            स्वतः tail_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            स्वतः tail = std::make_shared<ov::op::v8::Slice>(data_गाठ, tail_start, tail_end, step_one, axis_last);
            concat_parts.push_back(tail);
        }
        जर (concat_parts.size() == 1) {
            res = rotated;
        } अन्यथा {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    जर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
