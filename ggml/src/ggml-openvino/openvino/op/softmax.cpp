#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/exception.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/softmax.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_static_one(ಸ್ಥಿರಾಂಕ ov::Dimension & dim) {
    ಹಿಂತಿರುಗಿ dim.is_static() && dim.get_ಉದ್ದಳತೆ() == 1;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ same_static_dim(ಸ್ಥಿರಾಂಕ ov::Dimension & lhs, ಸ್ಥಿರಾಂಕ ov::Dimension & rhs) {
    ಹಿಂತಿರುಗಿ lhs.is_static() && rhs.is_static() && lhs.get_ಉದ್ದಳತೆ() == rhs.get_ಉದ್ದಳತೆ();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_attention_sinks_ಒಳಹರಿವು_shape(ಸ್ಥಿರಾಂಕ ov::PartialShape & candidate, ಸ್ಥಿರಾಂಕ ov::PartialShape & logits_shape) {
    ಆದರೆ (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_ಉದ್ದಳತೆ() != 4 ||
        logits_shape.rank().get_ಉದ್ದಳತೆ() != 4) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
Output_ಸದಿಶ translate_soft_max(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 3);

    ದಶಮಾಂಶ scale = 1.0f;
    ದಶಮಾಂಶ max_bias = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, (ದಶಮಾಂಶ *) context.get_output_op_params() + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ದಶಮಾಂಶ *) context.get_output_op_params() + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ov::Output<ov::Node> logits = context.get_ಒಳಹರಿವು(0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಎರಡನೇ_ಒಳಹರಿವು_is_sinks =
        context.get_ಒಳಹರಿವು_ಗಾತ್ರ() == 2 && is_attention_sinks_ಒಳಹರಿವು_shape(context.get_ಒಳಹರಿವು_shape(1), context.get_output_shape());
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_ಮುಖವಾಡ = context.get_ಒಳಹರಿವು_ಗಾತ್ರ() > 1 && !ಎರಡನೇ_ಒಳಹರಿವು_is_sinks;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_sinks = ಎರಡನೇ_ಒಳಹರಿವು_is_sinks || context.get_ಒಳಹರಿವು_ಗಾತ್ರ() > 2;
    ಸ್ಥಿರಾಂಕ size_t sinks_ಒಳಹರಿವು_idx = ಎರಡನೇ_ಒಳಹರಿವು_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    ಆದರೆ (scale != 1.0f) {
        ಸ್ವಯಂಚಾಲಿತ scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ದಶಮಾಂಶ>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_ಕೊನೆ_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_ಮುಖವಾಡ),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    ಆದರೆ (has_ಮುಖವಾಡ) {
        ov::Output<ov::Node> ಮುಖವಾಡ = context.get_ಒಳಹರಿವು(1);

        // For stateful
        std::string ಮುಖವಾಡ_name = "KQ_mask_sliced";
        ಆದರೆ (context.get_ಒಳಹರಿವು_ಹೆಸರುಗಳು()[1].find("swa") != std::string::npos) {
            ಮುಖವಾಡ_name = "KQ_mask_swa_sliced";
        }
        ಆದರೆ (context.has_ಒಳಹರಿವು(ಮುಖವಾಡ_name)) {
            ಮುಖವಾಡ = context.get_ಒಳಹರಿವು(ಮುಖವಾಡ_name);
        }

        ಆದರೆ (ಮುಖವಾಡ.get_element_type() != logits.get_element_type()) {
            ಮುಖವಾಡ = std::make_shared<ov::op::v0::Convert>(ಮುಖವಾಡ, logits.get_element_type());
        }

        ಆದರೆ (max_bias > 0.0f) {
            ಸ್ವಯಂಚಾಲಿತ ಹೊರಹೋಗು_shape = context.get_output_shape().ಗಮ್ಯ_shape();
            FRONT_ಕೊನೆ_CHECK_IMPLEMENTED(ಹೊರಹೋಗು_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ = static_cast<uint32_t>(ಹೊರಹೋಗು_shape[1]);
            FRONT_ಕೊನೆ_CHECK_IMPLEMENTED(n_ತಲೆ > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<ದಶಮಾಂಶ>(n_ತಲೆ))));
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0 = std::ಘಾತ(2.0f, -(max_bias) / static_cast<ದಶಮಾಂಶ>(n_ತಲೆ_log2));
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = std::ಘಾತ(2.0f, -(max_bias / 2.0f) / static_cast<ದಶಮಾಂಶ>(n_ತಲೆ_log2));

            std::vector<ದಶಮಾಂಶ> slopes(n_ತಲೆ);
            ಗಾಗಿ (uint32_t h = 0; h < n_ತಲೆ; ++h) {
                slopes[h] = h < n_ತಲೆ_log2 ? std::ಘಾತ(m0, static_cast<ದಶಮಾಂಶ>(h + 1)) :
                                              std::ಘಾತ(m1, static_cast<ದಶಮಾಂಶ>(2 * (h - n_ತಲೆ_log2) + 1));
            }

            ov::Output<ov::Node> ಇಳಿಜಾರು_ಗಂಟು =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_ತಲೆ}, slopes);
            ಆದರೆ (ಇಳಿಜಾರು_ಗಂಟು.get_element_type() != ಮುಖವಾಡ.get_element_type()) {
                ಇಳಿಜಾರು_ಗಂಟು = std::make_shared<ov::op::v0::Convert>(ಇಳಿಜಾರು_ಗಂಟು, ಮುಖವಾಡ.get_element_type());
            }

            ಸ್ವಯಂಚಾಲಿತ ಇಳಿಜಾರು_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_ತಲೆ), 1, 1});
            ಸ್ವಯಂಚಾಲಿತ ಇಳಿಜಾರು_4d = std::make_shared<ov::op::v1::Reshape>(ಇಳಿಜಾರು_ಗಂಟು, ಇಳಿಜಾರು_shape, ಸುಳ್ಳು);
            ಮುಖವಾಡ = std::make_shared<ov::op::v1::Multiply>(ಮುಖವಾಡ, ಇಳಿಜಾರು_4d);
        }

        logits = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(logits, ಮುಖವಾಡ);
    }

    ov::Output<ov::Node> softmax_ಒಳಹರಿವು = logits;
    ಆದರೆ (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_ಒಳಹರಿವು(sinks_ಒಳಹರಿವು_idx);
        ಆದರೆ (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        ಸ್ವಯಂಚಾಲಿತ sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        ಸ್ವಯಂಚಾಲಿತ sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, ಸುಳ್ಳು);

        ಸ್ವಯಂಚಾಲಿತ logits_shape = std::make_shared<ov::op::v3::Shape_ದ>(logits, ov::element::i64);
        ಸ್ವಯಂಚಾಲಿತ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ಸ್ವಯಂಚಾಲಿತ three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        ಸ್ವಯಂಚಾಲಿತ four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        ಸ್ವಯಂಚಾಲಿತ shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        ಸ್ವಯಂಚಾಲಿತ sink_ಪೂರ್ವಪ್ರತ್ಯಯ_shape = std::make_shared<ov::op::v8::ತುಂಡು>(logits_shape, zero, three, one, shape_axis);
        ಸ್ವಯಂಚಾಲಿತ sink_ಕೊನೆಯ_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ಸ್ವಯಂಚಾಲಿತ sink_ಪ್ರಸಾರ_shape = std::make_shared<ov::op::v0::Concat>(
            ov::Output_ಸದಿಶ{sink_ಪೂರ್ವಪ್ರತ್ಯಯ_shape, sink_ಕೊನೆಯ_dim}, 0);
        ಸ್ವಯಂಚಾಲಿತ sink_ಸ್ತಂಭ = std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(sinks_4d, sink_ಪ್ರಸಾರ_shape,
                                                                   ov::op::ಪ್ರಸಾರ_Type::BIDIRECTIONAL);
        softmax_ಒಳಹರಿವು = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{logits, sink_ಸ್ತಂಭ}, 3);

        ಸ್ವಯಂಚಾಲಿತ softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_ಒಳಹರಿವು, -1);
        ಸ್ವಯಂಚಾಲಿತ original_ಕೊನೆಯ_dim = std::make_shared<ov::op::v8::ತುಂಡು>(logits_shape, three, four, one, shape_axis);
        ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v8::ತುಂಡು>(softmax_with_sink, zero, original_ಕೊನೆಯ_dim, one, three);

        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v8::Softmax>(softmax_ಒಳಹರಿವು, -1);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
