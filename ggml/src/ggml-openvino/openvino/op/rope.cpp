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

OutputVector translate_rope(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 3);

    पूर्णाङ्क op_case = context.get_op_case();

    ov::Output<Node> res;

    auto data_गाँठ = process_view_निवेश_new(context, 0).get_गाँठ_shared_ptr();
    auto output_shape = context.get_output_shape().to_shape();
    int32_t * op_params = context.get_output_op_params();
    ध्रुव पूर्णाङ्क mode = op_case;
    ध्रुव int64_t head_dim = static_cast<int64_t>(output_shape[3]);
    ध्रुव int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    ध्रुव int64_t n_dims = configured_n_dims == 0 ? head_dim : configured_n_dims;
    ध्रुव int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr पूर्णाङ्क TYPE_NORMAL = 0;
    constexpr पूर्णाङ्क TYPE_NEOX = 1;
    constexpr पूर्णाङ्क TYPE_IMROPE = 2;
    constexpr पूर्णाङ्क TYPE_VISION = 3;
    constexpr पूर्णाङ्क TYPE_MROPE = 4;

    Output<Node> cos_theta_गाँठ;
    Output<Node> sin_theta_गाँठ;
    यदि (context.has_निवेश("rope_cos")) {
        cos_theta_गाँठ = context.get_निवेश("rope_cos");
        sin_theta_गाँठ = context.get_निवेश("rope_sin");
    } अन्यथा {
        std::string cache_कुंजी = "rope_sin_cos";
        प्रति (पूर्णाङ्क i = 0; i < 15; i++) {
            cache_कुंजी += "_" + std::to_string(op_params[i]);
        }
        यदि (context.get_निवेश_size() == 3) {
            cache_कुंजी += "_ff_" + context.get_निवेश_names()[2];
        }
        यदि (context.has_निवेश(cache_कुंजी + "_cos")) {
            cos_theta_गाँठ = context.get_निवेश(cache_कुंजी + "_cos");
            sin_theta_गाँठ = context.get_निवेश(cache_कुंजी + "_sin");
        } अन्यथा {
            auto inp_pos = context.get_निवेश(1).get_गाँठ_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_weight;
            यदि (context.get_निवेश_size() == 3) {
                rope_freqs_weight = context.get_निवेश(2).get_गाँठ_shared_ptr();
            }
            auto sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight, mode, असत्य, head_dim);
            sin_theta_गाँठ = sin_cos.first;
            cos_theta_गाँठ = sin_cos.second;
            context.put_shared(cache_कुंजी + "_cos", cos_theta_गाँठ);
            context.put_shared(cache_कुंजी + "_sin", sin_theta_गाँठ);
        }
    }

    auto output_type = context.get_output_type();
    यदि (data_गाँठ->get_element_type() != ov::element::f32) {
        data_गाँठ = std::make_shared<ov::op::v0::Convert>(data_गाँठ, ov::element::f32);
    }

    ध्रुव int64_t total_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_END_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_END_OP_CONVERSION_CHECK(n_dims > 0 && total_rope_dims + n_offs <= head_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    यदि (context.is_stateful()) {
        ध्रुव int64_t batch = static_cast<int64_t>(output_shape[0]);
        auto r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        data_गाँठ = std::make_shared<ov::op::v1::Reshape>(data_गाँठ, r4_shape, असत्य);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    यदि (mode == TYPE_NORMAL) {
        auto axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ध्रुव int64_t batch = static_cast<int64_t>(output_shape[0]);
        ध्रुव int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        ध्रुव int64_t half = n_dims / 2;
        auto rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        auto rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        auto rot_data = std::make_shared<ov::op::v8::Slice>(data_गाँठ, rot_start, rot_end, step_one, axis_last);

        auto neg_one_f = ov::op::v0::Constant::create(data_गाँठ->get_element_type(), ov::Shape{}, {-1.0f});

        auto paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        auto x_paired = std::make_shared<ov::op::v1::Reshape>(rot_data, paired_shape, असत्य);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto data_split = std::make_shared<ov::op::v1::Split>(x_paired, split_axis, 2);
        Output<Node> x0 = data_split->outputs()[0];
        Output<Node> x1 = data_split->outputs()[1];

        auto x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        auto x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_neg, x0}, -1);

        auto flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        auto x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, असत्य);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        auto expand_cos_sin = [&](ध्रुव Output<Node>& cs) {
            auto cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            auto bcast_target = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            auto bcast = std::make_shared<ov::op::v3::Broadcast>(
                cs_unsq, bcast_target, ov::op::BroadcastType::BIDIRECTIONAL);
            auto flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            प्रत्यागम std::make_shared<ov::op::v1::Reshape>(bcast, flat, सत्य);
        };
        Output<Node> cos_full = expand_cos_sin(cos_theta_गाँठ);
        Output<Node> sin_full = expand_cos_sin(sin_theta_गाँठ);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(rot_data, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_full);
        auto rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        यदि (n_offs > 0) {
            auto head_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto head_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto head = std::make_shared<ov::op::v8::Slice>(data_गाँठ, head_start, head_end, step_one, axis_last);
            concat_parts.push_back(head);
        }
        concat_parts.push_back(rotated);
        यदि (n_offs + n_dims < head_dim) {
            auto tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            auto tail_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            auto tail = std::make_shared<ov::op::v8::Slice>(data_गाँठ, tail_start, tail_end, step_one, axis_last);
            concat_parts.push_back(tail);
        }
        यदि (concat_parts.size() == 1) {
            res = rotated;
        } अन्यथा {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } अन्यथा यदि (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        ध्रुव int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        ध्रुव int64_t rot_dims = 2 * half;

        यदि (mode != TYPE_NEOX) {
            auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_गाँठ = std::make_shared<ov::op::v1::Reshape>(cos_theta_गाँठ, cos_sin_shape, सत्य);
            sin_theta_गाँठ = std::make_shared<ov::op::v1::Reshape>(sin_theta_गाँठ, cos_sin_shape, सत्य);
        }

        auto axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_data = data_गाँठ;
        यदि (n_offs > 0 || n_offs + rot_dims < head_dim) {
            auto rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_data = std::make_shared<ov::op::v8::Slice>(data_गाँठ, rot_start, rot_end, step_one, axis_last);
        }

        auto neg_one_f = ov::op::v0::Constant::create(data_गाँठ->get_element_type(), ov::Shape{}, {-1.0f});

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        auto data_split = std::make_shared<ov::op::v1::VariadicSplit>(rot_data, split_axis, split_lengths);
        Output<Node> x1 = data_split->outputs()[0];
        Output<Node> x2 = data_split->outputs()[1];

        auto x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        auto x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x2_neg, x1}, -1);

        auto cos_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_theta_गाँठ, cos_theta_गाँठ}, -1);
        auto sin_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_theta_गाँठ, sin_theta_गाँठ}, -1);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(rot_data, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_full);
        auto rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        यदि (n_offs > 0) {
            auto head_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto head_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto head = std::make_shared<ov::op::v8::Slice>(data_गाँठ, head_start, head_end, step_one, axis_last);
            concat_parts.push_back(head);
        }
        concat_parts.push_back(rotated);
        यदि (n_offs + rot_dims < head_dim) {
            auto tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            auto tail_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            auto tail = std::make_shared<ov::op::v8::Slice>(data_गाँठ, tail_start, tail_end, step_one, axis_last);
            concat_parts.push_back(tail);
        }
        यदि (concat_parts.size() == 1) {
            res = rotated;
        } अन्यथा {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
