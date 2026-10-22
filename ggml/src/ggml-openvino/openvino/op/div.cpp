#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/type.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/divide.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/sigmoid.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/swish.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/tile.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/util/precision_sensitive_attribute.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

ನಾಮಾಕಾಶ {

ತಾರ್ಕಿಕ is_silu_div_ನಮೂನೆ(ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & numerator,
                         ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & denominator,
                         ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಆದರೆ (context.get_ಒಳಹರಿವು_ಗಾತ್ರ() != 2) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * unary_op = reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_unary_op *>(context.get_ಒಳಹರಿವು_op_params(0));
    ಆದರೆ (unary_op == ಶೂನ್ಯಸೂಚಕ || *unary_op != GGML_UNARY_OP_SILU) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ denom_ಗಂಟು = denominator.get_ಗಂಟು_shared_ptr();

    ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_ಗಂಟು_shared_ptr())) {
        ಹಿಂತಿರುಗಿ swish->ಒಳಹರಿವು_ಮೌಲ್ಯ(0).get_ಗಂಟು_shared_ptr() == denom_ಗಂಟು;
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ov::Output<ov::Node> repeat_ಒಳಹರಿವು_ಗಮ್ಯ_ಹೊಂದಿಕೆ(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context,
                                           ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & ಒಳಹರಿವು,
                                           ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & ಗುರಿ,
                                           size_t ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ) {
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_shape = context.get_ಒಳಹರಿವು_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಗುರಿ_shape = context.get_ಒಳಹರಿವು_shape(0);

    ಆದರೆ (ಒಳಹರಿವು_shape == ಗುರಿ_shape) {
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು;
    }

    ಆದರೆ (ಒಳಹರಿವು_shape.rank().is_static() && ಗುರಿ_shape.rank().is_static()) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ rank = static_cast<size_t>(ಒಳಹರಿವು_shape.rank().get_ಉದ್ದಳತೆ());
        std::vector<int64_t> repeats(rank, 1);
        ತಾರ್ಕಿಕ needs_repeat = ಸುಳ್ಳು;

        ಗಾಗಿ (size_t axis = 0; axis < rank; ++axis) {
            FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಒಳಹರಿವು_shape[axis].is_static() && ಗುರಿ_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ಸ್ಥಿರಾಂಕ int64_t ಒಳಹರಿವು_dim = ಒಳಹರಿವು_shape[axis].get_ಉದ್ದಳತೆ();
            ಸ್ಥಿರಾಂಕ int64_t ಗುರಿ_dim = ಗುರಿ_shape[axis].get_ಉದ್ದಳತೆ();

            FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಒಳಹರಿವು_dim > 0 && ಗುರಿ_dim > 0 && ಗುರಿ_dim % ಒಳಹರಿವು_dim == 0,
                                          "DIV input shape ", ಒಳಹರಿವು_shape, " cannot repeat to match ", ಗುರಿ_shape);

            repeats[axis] = ಗುರಿ_dim / ಒಳಹರಿವು_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        ಆದರೆ (!needs_repeat) {
            ಹಿಂತಿರುಗಿ ಒಳಹರಿವು;
        }

        ಸ್ವಯಂಚಾಲಿತ repeats_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        ಹಿಂತಿರುಗಿ std::make_shared<ov::op::v0::Tile>(ಒಳಹರಿವು, repeats_ಗಂಟು);
    }

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_shape_ಗಂಟು = std::make_shared<ov::op::v3::Shape_ದ>(ಒಳಹರಿವು, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ ಗುರಿ_shape_ಗಂಟು = std::make_shared<ov::op::v3::Shape_ದ>(ಗುರಿ, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ repeats_ಗಂಟು = std::make_shared<ov::op::v1::Divide>(ಗುರಿ_shape_ಗಂಟು, ಒಳಹರಿವು_shape_ಗಂಟು);
    ಹಿಂತಿರುಗಿ std::make_shared<ov::op::v0::Tile>(ಒಳಹರಿವು, repeats_ಗಂಟು);
}

}  // namespace

Output_ಸದಿಶ translate_div(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_0 = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_1 = process_view_ಒಳಹರಿವು_new(context, 1);

    ಆದರೆ (is_silu_div_ನಮೂನೆ(ಒಳಹರಿವು_0, ಒಳಹರಿವು_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(ಒಳಹರಿವು_1);
        ಆದರೆ (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
    }

    ಒಳಹರಿವು_1 = repeat_ಒಳಹರಿವು_ಗಮ್ಯ_ಹೊಂದಿಕೆ(context, ಒಳಹರಿವು_1, ಒಳಹರಿವು_0, 1);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_f32_compute = ಒಳಹರಿವು_0.get_element_type() != ov::element::f32 ||
                                 ಒಳಹರಿವು_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    ಆದರೆ (use_f32_compute) {
        ಒಳಹರಿವು_0 = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು_0, ov::element::f32);
        ಒಳಹರಿವು_1 = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(ಒಳಹರಿವು_0, ಒಳಹರಿವು_1);
    ಆದರೆ (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_ಗಂಟು_shared_ptr()->ಒಳಹರಿವು(0));
        ov::mark_as_precision_sensitive(res.get_ಗಂಟು_shared_ptr()->ಒಳಹರಿವು(1));
    }
    ಆದರೆ (res.get_element_type() != output_type) {
        ಸ್ವಯಂಚಾಲಿತ output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        ಆದರೆ (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->ಒಳಹರಿವು(0));
        }
        res = output_convert;
    }
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
