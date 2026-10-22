#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <cmath>
#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/op/broadcast.hpp>
#অন্তর্ভাৱ <openvino/frontend/exception.hpp>
#অন্তর্ভাৱ <openvino/op/add.hpp>
#অন্তর্ভাৱ <openvino/op/concat.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/convert.hpp>
#অন্তর্ভাৱ <openvino/op/multiply.hpp>
#অন্তর্ভাৱ <openvino/op/reshape.hpp>
#অন্তর্ভাৱ <openvino/op/shape_of.hpp>
#অন্তর্ভাৱ <openvino/op/slice.hpp>
#অন্তর্ভাৱ <openvino/op/softmax.hpp>
#অন্তর্ভাৱ <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

স্থির তার্কিক is_static_one(ধ্রুৱ ov::Dimension & dim) {
    ঘূৰাই dim.is_static() && dim.get_length() == 1;
}

স্থির তার্কিক same_static_dim(ধ্রুৱ ov::Dimension & lhs, ধ্রুৱ ov::Dimension & rhs) {
    ঘূৰাই lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

স্থির তার্কিক is_attention_sinks_প্ৰবেশ_shape(ধ্রুৱ ov::PartialShape & candidate, ধ্রুৱ ov::PartialShape & logits_shape) {
    যদি (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        ঘূৰাই মিছা;
    }

    ঘূৰাই is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 3);

    ভিন্ন scale = 1.0f;
    ভিন্ন max_bias = 0.0f;
    memcpy(&scale, (ভিন্ন *) context.get_output_op_params() + 0, পরিমাণ(ভিন্ন));
    memcpy(&max_bias, (ভিন্ন *) context.get_output_op_params() + 1, পরিমাণ(ভিন্ন));

    ov::Output<ov::Node> logits = context.get_প্ৰবেশ(0);
    ধ্রুৱ তার্কিক second_প্ৰবেশ_is_sinks =
        context.get_প্ৰবেশ_size() == 2 && is_attention_sinks_প্ৰবেশ_shape(context.get_প্ৰবেশ_shape(1), context.get_output_shape());
    ধ্রুৱ তার্কিক has_mask = context.get_প্ৰবেশ_size() > 1 && !second_প্ৰবেশ_is_sinks;
    ধ্রুৱ তার্কিক has_sinks = second_প্ৰবেশ_is_sinks || context.get_প্ৰবেশ_size() > 2;
    ধ্রুৱ size_t sinks_প্ৰবেশ_idx = second_প্ৰবেশ_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    যদি (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ভিন্ন>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    যদি (has_mask) {
        ov::Output<ov::Node> mask = context.get_প্ৰবেশ(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        যদি (context.get_প্ৰবেশ_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        যদি (context.has_প্ৰবেশ(mask_name)) {
            mask = context.get_প্ৰবেশ(mask_name);
        }

        যদি (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        যদি (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ধ্রুৱ uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ধ্রুৱ uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<ভিন্ন>(n_head))));
            ধ্রুৱ ভিন্ন m0 = std::pow(2.0f, -(max_bias) / static_cast<ভিন্ন>(n_head_log2));
            ধ্রুৱ ভিন্ন m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<ভিন্ন>(n_head_log2));

            std::vector<ভিন্ন> slopes(n_head);
            বাবে (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<ভিন্ন>(h + 1)) :
                                              std::pow(m1, static_cast<ভিন্ন>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_গ্ৰন্থি =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            যদি (slope_গ্ৰন্থি.get_element_type() != mask.get_element_type()) {
                slope_গ্ৰন্থি = std::make_shared<ov::op::v0::Convert>(slope_গ্ৰন্থি, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_গ্ৰন্থি, slope_shape, মিছা);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_প্ৰবেশ = logits;
    যদি (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_প্ৰবেশ(sinks_প্ৰবেশ_idx);
        যদি (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, মিছা);

        auto logits_shape = std::make_shared<ov::op::v3::ShapeOf>(logits, ov::element::i64);
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        auto four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        auto shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        auto sink_prefix_shape = std::make_shared<ov::op::v8::Slice>(logits_shape, zero, three, one, shape_axis);
        auto sink_last_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto sink_broadcast_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{sink_prefix_shape, sink_last_dim}, 0);
        auto sink_column = std::make_shared<ov::op::v3::Broadcast>(sinks_4d, sink_broadcast_shape,
                                                                   ov::op::BroadcastType::BIDIRECTIONAL);
        softmax_প্ৰবেশ = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_প্ৰবেশ, -1);
        auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        auto res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(softmax_প্ৰবেশ, -1);

    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
