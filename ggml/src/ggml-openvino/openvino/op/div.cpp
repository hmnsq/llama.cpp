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

ತಾರ್ಕಿಕ is_silu_div_pattern(ಧ್ರುವ ov::Output<ov::Node> & numerator,
                         ಧ್ರುವ ov::Output<ov::Node> & denominator,
                         ಧ್ರುವ ಗಂಟು_Context & context) {
    ಯದಿ (context.get_ಒಳಹರಿವು_size() != 2) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ auto * unary_op = reinterpret_cast<ಧ್ರುವ ggml_unary_op *>(context.get_ಒಳಹರಿವು_op_params(0));
    ಯದಿ (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ auto denom_ಗಂಟು = denominator.get_ಗಂಟು_shared_ptr();

    ಯದಿ (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_ಗಂಟು_shared_ptr())) {
        ಪ್ರತ್ಯಾಗಮ swish->ಒಳಹರಿವು_ಮೌಲ್ಯ(0).get_ಗಂಟು_shared_ptr() == denom_ಗಂಟು;
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ov::Output<ov::Node> repeat_ಒಳಹರಿವು_to_match(ಧ್ರುವ ಗಂಟು_Context & context,
                                           ಧ್ರುವ ov::Output<ov::Node> & ಒಳಹರಿವು,
                                           ಧ್ರುವ ov::Output<ov::Node> & target,
                                           size_t ಒಳಹರಿವು_index) {
    ಧ್ರುವ auto ಒಳಹರಿವು_shape = context.get_ಒಳಹರಿವು_shape(ಒಳಹರಿವು_index);
    ಧ್ರುವ auto target_shape = context.get_ಒಳಹರಿವು_shape(0);

    ಯದಿ (ಒಳಹರಿವು_shape == target_shape) {
        ಪ್ರತ್ಯಾಗಮ ಒಳಹರಿವು;
    }

    ಯದಿ (ಒಳಹರಿವು_shape.rank().is_static() && target_shape.rank().is_static()) {
        ಧ್ರುವ auto rank = static_cast<size_t>(ಒಳಹರಿವು_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        ತಾರ್ಕಿಕ needs_repeat = ಅಸತ್ಯ;

        ಪ್ರತಿ (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(ಒಳಹರಿವು_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ಧ್ರುವ int64_t ಒಳಹರಿವು_dim = ಒಳಹರಿವು_shape[axis].get_length();
            ಧ್ರುವ int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(ಒಳಹರಿವು_dim > 0 && target_dim > 0 && target_dim % ಒಳಹರಿವು_dim == 0,
                                          "DIV input shape ", ಒಳಹರಿವು_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / ಒಳಹರಿವು_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        ಯದಿ (!needs_repeat) {
            ಪ್ರತ್ಯಾಗಮ ಒಳಹರಿವು;
        }

        auto repeats_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        ಪ್ರತ್ಯಾಗಮ std::make_shared<ov::op::v0::Tile>(ಒಳಹರಿವು, repeats_ಗಂಟು);
    }

    auto ಒಳಹರಿವು_shape_ಗಂಟು = std::make_shared<ov::op::v3::ShapeOf>(ಒಳಹರಿವು, ov::element::i64);
    auto target_shape_ಗಂಟು = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_ಗಂಟು = std::make_shared<ov::op::v1::Divide>(target_shape_ಗಂಟು, ಒಳಹರಿವು_shape_ಗಂಟು);
    ಪ್ರತ್ಯಾಗಮ std::make_shared<ov::op::v0::Tile>(ಒಳಹರಿವು, repeats_ಗಂಟು);
}

}  // namespace

OutputVector translate_div(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    auto ಒಳಹರಿವು_0 = process_view_ಒಳಹರಿವು_new(context, 0);
    auto ಒಳಹರಿವು_1 = process_view_ಒಳಹರಿವು_new(context, 1);

    ಯದಿ (is_silu_div_pattern(ಒಳಹರಿವು_0, ಒಳಹರಿವು_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(ಒಳಹರಿವು_1);
        ಯದಿ (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
    }

    ಒಳಹರಿವು_1 = repeat_ಒಳಹರಿವು_to_match(context, ಒಳಹರಿವು_1, ಒಳಹರಿವು_0, 1);

    ಧ್ರುವ auto output_type = context.get_output_type();
    ಧ್ರುವ ತಾರ್ಕಿಕ use_f32_compute = ಒಳಹರಿವು_0.get_element_type() != ov::element::f32 ||
                                 ಒಳಹರಿವು_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    ಯದಿ (use_f32_compute) {
        ಒಳಹರಿವು_0 = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು_0, ov::element::f32);
        ಒಳಹರಿವು_1 = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(ಒಳಹರಿವು_0, ಒಳಹರಿವು_1);
    ಯದಿ (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_ಗಂಟು_shared_ptr()->ಒಳಹರಿವು(0));
        ov::mark_as_precision_sensitive(res.get_ಗಂಟು_shared_ptr()->ಒಳಹರಿವು(1));
    }
    ಯದಿ (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        ಯದಿ (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->ಒಳಹರಿವು(0));
        }
        res = output_convert;
    }
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
