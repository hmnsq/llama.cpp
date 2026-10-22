#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node_output.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/add.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/broadcast.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/concat.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/multiply.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/slice.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/split.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/subtract.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/unsqueeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/variadic_split.hpp>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_rope(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 3);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_case = context.get_op_case();

    ov::Output<Node> res;

    auto data_ଗଣ୍ଠି = process_view_ନିବେଶ_new(context, 0).get_ଗଣ୍ଠି_shared_ptr();
    auto output_shape = context.get_output_shape().to_shape();
    int32_t * op_params = context.get_output_op_params();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode = op_case;
    ଅଚଳ int64_t head_dim = static_cast<int64_t>(output_shape[3]);
    ଅଚଳ int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    ଅଚଳ int64_t n_dims = configured_n_dims == 0 ? head_dim : configured_n_dims;
    ଅଚଳ int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା TYPE_NORMAL = 0;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା TYPE_NEOX = 1;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା TYPE_IMROPE = 2;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା TYPE_VISION = 3;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା TYPE_MROPE = 4;

    Output<Node> cos_theta_ଗଣ୍ଠି;
    Output<Node> sin_theta_ଗଣ୍ଠି;
    ଯଦି (context.has_ନିବେଶ("rope_cos")) {
        cos_theta_ଗଣ୍ଠି = context.get_ନିବେଶ("rope_cos");
        sin_theta_ଗଣ୍ଠି = context.get_ନିବେଶ("rope_sin");
    } ଅନ୍ୟଥା {
        std::string cache_ଚାବି = "rope_sin_cos";
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 15; i++) {
            cache_ଚାବି += "_" + std::to_string(op_params[i]);
        }
        ଯଦି (context.get_ନିବେଶ_size() == 3) {
            cache_ଚାବି += "_ff_" + context.get_ନିବେଶ_names()[2];
        }
        ଯଦି (context.has_ନିବେଶ(cache_ଚାବି + "_cos")) {
            cos_theta_ଗଣ୍ଠି = context.get_ନିବେଶ(cache_ଚାବି + "_cos");
            sin_theta_ଗଣ୍ଠି = context.get_ନିବେଶ(cache_ଚାବି + "_sin");
        } ଅନ୍ୟଥା {
            auto inp_pos = context.get_ନିବେଶ(1).get_ଗଣ୍ଠି_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_weight;
            ଯଦି (context.get_ନିବେଶ_size() == 3) {
                rope_freqs_weight = context.get_ନିବେଶ(2).get_ଗଣ୍ଠି_shared_ptr();
            }
            auto sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight, mode, ମିଥ୍ୟା, head_dim);
            sin_theta_ଗଣ୍ଠି = sin_cos.first;
            cos_theta_ଗଣ୍ଠି = sin_cos.second;
            context.put_shared(cache_ଚାବି + "_cos", cos_theta_ଗଣ୍ଠି);
            context.put_shared(cache_ଚାବି + "_sin", sin_theta_ଗଣ୍ଠି);
        }
    }

    auto output_type = context.get_output_type();
    ଯଦି (data_ଗଣ୍ଠି->get_element_type() != ov::element::f32) {
        data_ଗଣ୍ଠି = std::make_shared<ov::op::v0::Convert>(data_ଗଣ୍ଠି, ov::element::f32);
    }

    ଅଚଳ int64_t total_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_END_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_END_OP_CONVERSION_CHECK(n_dims > 0 && total_rope_dims + n_offs <= head_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    ଯଦି (context.is_stateful()) {
        ଅଚଳ int64_t batch = static_cast<int64_t>(output_shape[0]);
        auto r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        data_ଗଣ୍ଠି = std::make_shared<ov::op::v1::Reshape>(data_ଗଣ୍ଠି, r4_shape, ମିଥ୍ୟା);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    ଯଦି (mode == TYPE_NORMAL) {
        auto axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ଅଚଳ int64_t batch = static_cast<int64_t>(output_shape[0]);
        ଅଚଳ int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        ଅଚଳ int64_t half = n_dims / 2;
        auto rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        auto rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        auto rot_data = std::make_shared<ov::op::v8::Slice>(data_ଗଣ୍ଠି, rot_start, rot_end, step_one, axis_last);

        auto neg_one_f = ov::op::v0::Constant::create(data_ଗଣ୍ଠି->get_element_type(), ov::Shape{}, {-1.0f});

        auto paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        auto x_paired = std::make_shared<ov::op::v1::Reshape>(rot_data, paired_shape, ମିଥ୍ୟା);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto data_split = std::make_shared<ov::op::v1::Split>(x_paired, split_axis, 2);
        Output<Node> x0 = data_split->outputs()[0];
        Output<Node> x1 = data_split->outputs()[1];

        auto x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        auto x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_neg, x0}, -1);

        auto flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        auto x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, ମିଥ୍ୟା);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        auto expand_cos_sin = [&](ଅଚଳ Output<Node>& cs) {
            auto cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            auto bcast_target = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            auto bcast = std::make_shared<ov::op::v3::Broadcast>(
                cs_unsq, bcast_target, ov::op::BroadcastType::BIDIRECTIONAL);
            auto flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            ଫେରନ୍ତୁ std::make_shared<ov::op::v1::Reshape>(bcast, flat, ସତ୍ୟ);
        };
        Output<Node> cos_full = expand_cos_sin(cos_theta_ଗଣ୍ଠି);
        Output<Node> sin_full = expand_cos_sin(sin_theta_ଗଣ୍ଠି);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(rot_data, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_full);
        auto rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        ଯଦି (n_offs > 0) {
            auto head_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto head_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto head = std::make_shared<ov::op::v8::Slice>(data_ଗଣ୍ଠି, head_start, head_end, step_one, axis_last);
            concat_parts.push_back(head);
        }
        concat_parts.push_back(rotated);
        ଯଦି (n_offs + n_dims < head_dim) {
            auto tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            auto tail_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            auto tail = std::make_shared<ov::op::v8::Slice>(data_ଗଣ୍ଠି, tail_start, tail_end, step_one, axis_last);
            concat_parts.push_back(tail);
        }
        ଯଦି (concat_parts.size() == 1) {
            res = rotated;
        } ଅନ୍ୟଥା {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } ଅନ୍ୟଥା ଯଦି (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        ଅଚଳ int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        ଅଚଳ int64_t rot_dims = 2 * half;

        ଯଦି (mode != TYPE_NEOX) {
            auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_ଗଣ୍ଠି = std::make_shared<ov::op::v1::Reshape>(cos_theta_ଗଣ୍ଠି, cos_sin_shape, ସତ୍ୟ);
            sin_theta_ଗଣ୍ଠି = std::make_shared<ov::op::v1::Reshape>(sin_theta_ଗଣ୍ଠି, cos_sin_shape, ସତ୍ୟ);
        }

        auto axis_last = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_data = data_ଗଣ୍ଠି;
        ଯଦି (n_offs > 0 || n_offs + rot_dims < head_dim) {
            auto rot_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto rot_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_data = std::make_shared<ov::op::v8::Slice>(data_ଗଣ୍ଠି, rot_start, rot_end, step_one, axis_last);
        }

        auto neg_one_f = ov::op::v0::Constant::create(data_ଗଣ୍ଠି->get_element_type(), ov::Shape{}, {-1.0f});

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        auto data_split = std::make_shared<ov::op::v1::VariadicSplit>(rot_data, split_axis, split_lengths);
        Output<Node> x1 = data_split->outputs()[0];
        Output<Node> x2 = data_split->outputs()[1];

        auto x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        auto x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x2_neg, x1}, -1);

        auto cos_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_theta_ଗଣ୍ଠି, cos_theta_ଗଣ୍ଠି}, -1);
        auto sin_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_theta_ଗଣ୍ଠି, sin_theta_ଗଣ୍ଠି}, -1);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(rot_data, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_full);
        auto rotated = std::make_shared<ov::op::v1::Add>(y1, y2);

        ov::OutputVector concat_parts;
        ଯଦି (n_offs > 0) {
            auto head_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto head_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            auto head = std::make_shared<ov::op::v8::Slice>(data_ଗଣ୍ଠି, head_start, head_end, step_one, axis_last);
            concat_parts.push_back(head);
        }
        concat_parts.push_back(rotated);
        ଯଦି (n_offs + rot_dims < head_dim) {
            auto tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            auto tail_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            auto tail = std::make_shared<ov::op::v8::Slice>(data_ଗଣ୍ଠି, tail_start, tail_end, step_one, axis_last);
            concat_parts.push_back(tail);
        }
        ଯଦି (concat_parts.size() == 1) {
            res = rotated;
        } ଅନ୍ୟଥା {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    ଯଦି (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
