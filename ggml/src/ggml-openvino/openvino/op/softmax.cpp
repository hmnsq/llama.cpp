#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/op/broadcast.hpp>
#അന്തര്ഭാവ <openvino/frontend/exception.hpp>
#അന്തര്ഭാവ <openvino/op/add.hpp>
#അന്തര്ഭാവ <openvino/op/concat.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/op/multiply.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/shape_of.hpp>
#അന്തര്ഭാവ <openvino/op/slice.hpp>
#അന്തര്ഭാവ <openvino/op/softmax.hpp>
#അന്തര്ഭാവ <vector>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

സ്ഥിതം തർക്കം is_static_one(സ്ഥിരം ov::Dimension & dim) {
    തിരികെ dim.is_static() && dim.get_length() == 1;
}

സ്ഥിതം തർക്കം same_static_dim(സ്ഥിരം ov::Dimension & lhs, സ്ഥിരം ov::Dimension & rhs) {
    തിരികെ lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

സ്ഥിതം തർക്കം is_attention_sinks_നിവേശം_shape(സ്ഥിരം ov::PartialShape & candidate, സ്ഥിരം ov::PartialShape & logits_shape) {
    എങ്കിൽ (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        തിരികെ അസത്യം;
    }

    തിരികെ is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 3);

    ദശാംശം scale = 1.0f;
    ദശാംശം max_bias = 0.0f;
    memcpy(&scale, (ദശാംശം *) context.get_output_op_params() + 0, വലുപ്പം(ദശാംശം));
    memcpy(&max_bias, (ദശാംശം *) context.get_output_op_params() + 1, വലുപ്പം(ദശാംശം));

    ov::Output<ov::Node> logits = context.get_നിവേശം(0);
    സ്ഥിരം തർക്കം second_നിവേശം_is_sinks =
        context.get_നിവേശം_size() == 2 && is_attention_sinks_നിവേശം_shape(context.get_നിവേശം_shape(1), context.get_output_shape());
    സ്ഥിരം തർക്കം has_mask = context.get_നിവേശം_size() > 1 && !second_നിവേശം_is_sinks;
    സ്ഥിരം തർക്കം has_sinks = second_നിവേശം_is_sinks || context.get_നിവേശം_size() > 2;
    സ്ഥിരം size_t sinks_നിവേശം_idx = second_നിവേശം_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    എങ്കിൽ (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ദശാംശം>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    എങ്കിൽ (has_mask) {
        ov::Output<ov::Node> mask = context.get_നിവേശം(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        എങ്കിൽ (context.get_നിവേശം_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        എങ്കിൽ (context.has_നിവേശം(mask_name)) {
            mask = context.get_നിവേശം(mask_name);
        }

        എങ്കിൽ (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        എങ്കിൽ (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            സ്ഥിരം uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            സ്ഥിരം uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<ദശാംശം>(n_head))));
            സ്ഥിരം ദശാംശം m0 = std::pow(2.0f, -(max_bias) / static_cast<ദശാംശം>(n_head_log2));
            സ്ഥിരം ദശാംശം m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<ദശാംശം>(n_head_log2));

            std::vector<ദശാംശം> slopes(n_head);
            വേണ്ടി (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<ദശാംശം>(h + 1)) :
                                              std::pow(m1, static_cast<ദശാംശം>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_കണ്ണി =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            എങ്കിൽ (slope_കണ്ണി.get_element_type() != mask.get_element_type()) {
                slope_കണ്ണി = std::make_shared<ov::op::v0::Convert>(slope_കണ്ണി, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_കണ്ണി, slope_shape, അസത്യം);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_നിവേശം = logits;
    എങ്കിൽ (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_നിവേശം(sinks_നിവേശം_idx);
        എങ്കിൽ (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, അസത്യം);

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
        softmax_നിവേശം = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_നിവേശം, -1);
        auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        auto res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        തിരികെ rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(softmax_നിവേശം, -1);

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
