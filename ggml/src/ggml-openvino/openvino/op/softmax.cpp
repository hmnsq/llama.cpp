#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/op/broadcast.hpp>
#অন্তর্ভাব <openvino/frontend/exception.hpp>
#অন্তর্ভাব <openvino/op/add.hpp>
#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/multiply.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>
#অন্তর্ভাব <openvino/op/softmax.hpp>
#অন্তর্ভাব <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

স্থির যৌক্তিক is_static_one(ধ্রুবক ov::Dimension & dim) {
    ফেরত dim.is_static() && dim.get_দৈর্ঘ্য() == 1;
}

স্থির যৌক্তিক same_static_dim(ধ্রুবক ov::Dimension & lhs, ধ্রুবক ov::Dimension & rhs) {
    ফেরত lhs.is_static() && rhs.is_static() && lhs.get_দৈর্ঘ্য() == rhs.get_দৈর্ঘ্য();
}

স্থির যৌক্তিক is_attention_sinks_প্রবেশ_shape(ধ্রুবক ov::PartialShape & candidate, ধ্রুবক ov::PartialShape & logits_shape) {
    যদি (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_দৈর্ঘ্য() != 4 ||
        logits_shape.rank().get_দৈর্ঘ্য() != 4) {
        ফেরত মিথ্যা;
    }

    ফেরত is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 3);

    দশমিক scale = 1.0f;
    দশমিক max_bias = 0.0f;
    memcpy(&scale, (দশমিক *) context.get_output_op_params() + 0, আকার(দশমিক));
    memcpy(&max_bias, (দশমিক *) context.get_output_op_params() + 1, আকার(দশমিক));

    ov::Output<ov::Node> logits = context.get_প্রবেশ(0);
    ধ্রুবক যৌক্তিক second_প্রবেশ_is_sinks =
        context.get_প্রবেশ_আকার() == 2 && is_attention_sinks_প্রবেশ_shape(context.get_প্রবেশ_shape(1), context.get_output_shape());
    ধ্রুবক যৌক্তিক has_mask = context.get_প্রবেশ_আকার() > 1 && !second_প্রবেশ_is_sinks;
    ধ্রুবক যৌক্তিক has_sinks = second_প্রবেশ_is_sinks || context.get_প্রবেশ_আকার() > 2;
    ধ্রুবক size_t sinks_প্রবেশ_idx = second_প্রবেশ_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    যদি (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<দশমিক>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    যদি (has_mask) {
        ov::Output<ov::Node> mask = context.get_প্রবেশ(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        যদি (context.get_প্রবেশ_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        যদি (context.has_প্রবেশ(mask_name)) {
            mask = context.get_প্রবেশ(mask_name);
        }

        যদি (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        যদি (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ধ্রুবক uint32_t n_শীর্ষ = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_শীর্ষ > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ধ্রুবক uint32_t n_শীর্ষ_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<দশমিক>(n_শীর্ষ))));
            ধ্রুবক দশমিক m0 = std::pow(2.0f, -(max_bias) / static_cast<দশমিক>(n_শীর্ষ_log2));
            ধ্রুবক দশমিক m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<দশমিক>(n_শীর্ষ_log2));

            std::vector<দশমিক> slopes(n_শীর্ষ);
            জন্য (uint32_t h = 0; h < n_শীর্ষ; ++h) {
                slopes[h] = h < n_শীর্ষ_log2 ? std::pow(m0, static_cast<দশমিক>(h + 1)) :
                                              std::pow(m1, static_cast<দশমিক>(2 * (h - n_শীর্ষ_log2) + 1));
            }

            ov::Output<ov::Node> slope_গ্রন্থি =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_শীর্ষ}, slopes);
            যদি (slope_গ্রন্থি.get_element_type() != mask.get_element_type()) {
                slope_গ্রন্থি = std::make_shared<ov::op::v0::Convert>(slope_গ্রন্থি, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_শীর্ষ), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_গ্রন্থি, slope_shape, মিথ্যা);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_প্রবেশ = logits;
    যদি (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_প্রবেশ(sinks_প্রবেশ_idx);
        যদি (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, মিথ্যা);

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
        softmax_প্রবেশ = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_প্রবেশ, -1);
        auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        auto res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        ফেরত rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(softmax_প্রবেশ, -1);

    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
