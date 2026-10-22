#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/node.hpp>
#அந்தர்பாவ <openvino/core/node_output.hpp>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/broadcast.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/split.hpp>
#அந்தர்பாவ <openvino/op/subtract.hpp>
#அந்தர்பாவ <openvino/op/unsqueeze.hpp>
#அந்தர்பாவ <openvino/op/variadic_split.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_rope(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 2, 3);

    முழு op_case = context.get_op_case();

    ov::Output<Node> res;

    auto data_முனை = process_view_உள்ளீடு_new(context, 0).get_முனை_shared_ptr();
    auto output_shape = context.get_output_shape().to_shape();
    int32_t * op_params = context.get_output_op_params();
    மாறிலி முழு mode = op_case;
    மாறிலி int64_t head_dim = static_cast<int64_t>(output_shape[3]);
    மாறிலி int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    மாறிலி int64_t n_dims = configured_n_dims == 0 ? head_dim : configured_n_dims;
    மாறிலி int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr முழு TYPE_NORMAL = 0;
    constexpr முழு TYPE_NEOX = 1;
    constexpr முழு TYPE_IMROPE = 2;
    constexpr முழு TYPE_VISION = 3;
    constexpr முழு TYPE_MROPE = 4;

    Output<Node> cos_theta_முனை;
    Output<Node> sin_theta_முனை;
    எனில் (context.has_உள்ளீடு("rope_cos")) {
        cos_theta_முனை = context.get_உள்ளீடு("rope_cos");
        sin_theta_முனை = context.get_உள்ளீடு("rope_sin");
    } இல்லையெனில் {
        std::string cache_சாவி = "rope_sin_cos";
        க்கு (முழு i = 0; i < 15; i++) {
            cache_சாவி += "_" + std::to_string(op_params[i]);
        }
        எனில் (context.get_உள்ளீடு_size() == 3) {
            cache_சாவி += "_ff_" + context.get_உள்ளீடு_names()[2];
        }
        எனில் (context.has_உள்ளீடு(cache_சாவி + "_cos")) {
            cos_theta_முனை = context.get_உள்ளீடு(cache_சாவி + "_cos");
            sin_theta_முனை = context.get_உள்ளீடு(cache_சாவி + "_sin");
        } இல்லையெனில் {
            auto inp_pos = context.get_உள்ளீடு(1).get_முனை_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_weight;
            எனில் (context.get_உள்ளீடு_size() == 3) {
                rope_freqs_weight = context.get_உள்ளீடு(2).get_முனை_shared_ptr();
            }
            auto sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight, mode, பொய், head_dim);
            sin_theta_முனை = sin_cos.first;
            cos_theta_முனை = sin_cos.second;
            context.put_shared(cache_சாவி + "_cos", cos_theta_முனை);
            context.put_shared(cache_சாவி + "_sin", sin_theta_முனை);
        }
    }

    auto output_type = context.get_output_type();
    எனில் (data_முனை->get_element_type() != ov::element::f32) {
        data_முனை = std::make_shared<ov::op::v0::Convert>(data_முனை, ov::element::f32);
    }

    மாறிலி int64_t total_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_END_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_END_OP_CONVERSION_CHECK(n_dims > 0 && total_rope_dims + n_offs <= head_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    எனில் (context.is_stateful()) {
        மாறிலி int64_t batch = static_cast<int64_t>(output_shape[0]);
        auto r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        data_முனை = std::make_shared<ov::op::v1::Reshape>(data_முனை, r4_shape, பொய்);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    எனில் (mode == TYPE_NORMAL) {
        auto axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        மாறிலி int64_t batch = static_cast<int64_t>(output_shape[0]);
        மாறிலி int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        மாறிலி int64_t half = n_dims / 2;
        auto rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        auto rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        auto rot_data = std::make_shared<ov::op::v8::Slice>(data_முனை, rot_start, rot_end, step_one, axis_last);

        auto neg_one_f = ov::op::v0::Constant::create(data_முனை->get_element_type(), ov::Shape{}, {-1.0f});

        auto paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        auto x_paired = std::make_shared<ov::op::v1::Reshape>(rot_data, paired_shape, பொய்);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto data_split = std::make_shared<ov::op::v1::Split>(x_paired, split_axis, 2);
        Output<Node> x0 = data_split->outputs()[0];
        Output<Node> x1 = data_split->outputs()[1];

        auto x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        auto x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_neg, x0}, -1);

        auto flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        auto x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, பொய்);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        auto expand_cos_sin = [&](மாறிலி Output<Node>& cs) {
            auto cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            auto bcast_target = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            auto bcast = std::make_shared<ov::op::v3::Broadcast>(
                cs_unsq, bcast_target, ov::op::BroadcastType::BIDIRECTIONAL);
            auto flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            திரும்பு std::make_shared<ov::op::v1::Reshape>(bcast, flat, உண்மை);
        };
        Output<Node> cos_full = expand_cos_sin(cos_theta_முனை);
        Output<Node> sin_full = expand_cos_sin(sin_theta_முனை);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(rot_data, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_full);
        auto rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        எனில் (n_offs > 0) {
            auto head_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto head_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto head = std::make_shared<ov::op::v8::Slice>(data_முனை, head_start, head_end, step_one, axis_last);
            concat_parts.push_back(head);
        }
        concat_parts.push_back(rotated);
        எனில் (n_offs + n_dims < head_dim) {
            auto tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            auto tail_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            auto tail = std::make_shared<ov::op::v8::Slice>(data_முனை, tail_start, tail_end, step_one, axis_last);
            concat_parts.push_back(tail);
        }
        எனில் (concat_parts.size() == 1) {
            res = rotated;
        } இல்லையெனில் {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } இல்லையெனில் எனில் (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        மாறிலி int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        மாறிலி int64_t rot_dims = 2 * half;

        எனில் (mode != TYPE_NEOX) {
            auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_முனை = std::make_shared<ov::op::v1::Reshape>(cos_theta_முனை, cos_sin_shape, உண்மை);
            sin_theta_முனை = std::make_shared<ov::op::v1::Reshape>(sin_theta_முனை, cos_sin_shape, உண்மை);
        }

        auto axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_data = data_முனை;
        எனில் (n_offs > 0 || n_offs + rot_dims < head_dim) {
            auto rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_data = std::make_shared<ov::op::v8::Slice>(data_முனை, rot_start, rot_end, step_one, axis_last);
        }

        auto neg_one_f = ov::op::v0::Constant::create(data_முனை->get_element_type(), ov::Shape{}, {-1.0f});

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        auto data_split = std::make_shared<ov::op::v1::VariadicSplit>(rot_data, split_axis, split_lengths);
        Output<Node> x1 = data_split->outputs()[0];
        Output<Node> x2 = data_split->outputs()[1];

        auto x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        auto x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x2_neg, x1}, -1);

        auto cos_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_theta_முனை, cos_theta_முனை}, -1);
        auto sin_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_theta_முனை, sin_theta_முனை}, -1);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(rot_data, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_full);
        auto rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        எனில் (n_offs > 0) {
            auto head_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto head_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto head = std::make_shared<ov::op::v8::Slice>(data_முனை, head_start, head_end, step_one, axis_last);
            concat_parts.push_back(head);
        }
        concat_parts.push_back(rotated);
        எனில் (n_offs + rot_dims < head_dim) {
            auto tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            auto tail_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            auto tail = std::make_shared<ov::op::v8::Slice>(data_முனை, tail_start, tail_end, step_one, axis_last);
            concat_parts.push_back(tail);
        }
        எனில் (concat_parts.size() == 1) {
            res = rotated;
        } இல்லையெனில் {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    எனில் (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
