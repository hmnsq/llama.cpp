#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"
#অন্তর্ভাৱ "ggml.h"

#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/core/type.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/convert.hpp>
#অন্তর্ভাৱ <openvino/op/divide.hpp>
#অন্তর্ভাৱ <openvino/op/multiply.hpp>
#অন্তর্ভাৱ <openvino/op/shape_of.hpp>
#অন্তর্ভাৱ <openvino/op/sigmoid.hpp>
#অন্তর্ভাৱ <openvino/op/swish.hpp>
#অন্তর্ভাৱ <openvino/op/tile.hpp>
#অন্তর্ভাৱ <openvino/op/util/precision_sensitive_attribute.hpp>
#অন্তর্ভাৱ <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

নামাকাশ {

তার্কিক is_silu_div_pattern(ধ্রুৱ ov::Output<ov::Node> & numerator,
                         ধ্রুৱ ov::Output<ov::Node> & denominator,
                         ধ্রুৱ গ্ৰন্থি_Context & context) {
    যদি (context.get_প্ৰবেশ_size() != 2) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ auto * unary_op = reinterpret_cast<ধ্রুৱ ggml_unary_op *>(context.get_প্ৰবেশ_op_params(0));
    যদি (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ auto denom_গ্ৰন্থি = denominator.get_গ্ৰন্থি_shared_ptr();

    যদি (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_গ্ৰন্থি_shared_ptr())) {
        ঘূৰাই swish->প্ৰবেশ_মান(0).get_গ্ৰন্থি_shared_ptr() == denom_গ্ৰন্থি;
    }
    ঘূৰাই মিছা;
}

ov::Output<ov::Node> repeat_প্ৰবেশ_to_match(ধ্রুৱ গ্ৰন্থি_Context & context,
                                           ধ্রুৱ ov::Output<ov::Node> & প্ৰবেশ,
                                           ধ্রুৱ ov::Output<ov::Node> & target,
                                           size_t প্ৰবেশ_index) {
    ধ্রুৱ auto প্ৰবেশ_shape = context.get_প্ৰবেশ_shape(প্ৰবেশ_index);
    ধ্রুৱ auto target_shape = context.get_প্ৰবেশ_shape(0);

    যদি (প্ৰবেশ_shape == target_shape) {
        ঘূৰাই প্ৰবেশ;
    }

    যদি (প্ৰবেশ_shape.rank().is_static() && target_shape.rank().is_static()) {
        ধ্রুৱ auto rank = static_cast<size_t>(প্ৰবেশ_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        তার্কিক needs_repeat = মিছা;

        বাবে (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(প্ৰবেশ_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ধ্রুৱ int64_t প্ৰবেশ_dim = প্ৰবেশ_shape[axis].get_length();
            ধ্রুৱ int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(প্ৰবেশ_dim > 0 && target_dim > 0 && target_dim % প্ৰবেশ_dim == 0,
                                          "DIV input shape ", প্ৰবেশ_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / প্ৰবেশ_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        যদি (!needs_repeat) {
            ঘূৰাই প্ৰবেশ;
        }

        auto repeats_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        ঘূৰাই std::make_shared<ov::op::v0::Tile>(প্ৰবেশ, repeats_গ্ৰন্থি);
    }

    auto প্ৰবেশ_shape_গ্ৰন্থি = std::make_shared<ov::op::v3::ShapeOf>(প্ৰবেশ, ov::element::i64);
    auto target_shape_গ্ৰন্থি = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_গ্ৰন্থি = std::make_shared<ov::op::v1::Divide>(target_shape_গ্ৰন্থি, প্ৰবেশ_shape_গ্ৰন্থি);
    ঘূৰাই std::make_shared<ov::op::v0::Tile>(প্ৰবেশ, repeats_গ্ৰন্থি);
}

}  // namespace

OutputVector translate_div(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    auto প্ৰবেশ_0 = process_view_প্ৰবেশ_new(context, 0);
    auto প্ৰবেশ_1 = process_view_প্ৰবেশ_new(context, 1);

    যদি (is_silu_div_pattern(প্ৰবেশ_0, প্ৰবেশ_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(প্ৰবেশ_1);
        যদি (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
    }

    প্ৰবেশ_1 = repeat_প্ৰবেশ_to_match(context, প্ৰবেশ_1, প্ৰবেশ_0, 1);

    ধ্রুৱ auto output_type = context.get_output_type();
    ধ্রুৱ তার্কিক use_f32_compute = প্ৰবেশ_0.get_element_type() != ov::element::f32 ||
                                 প্ৰবেশ_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    যদি (use_f32_compute) {
        প্ৰবেশ_0 = std::make_shared<ov::op::v0::Convert>(প্ৰবেশ_0, ov::element::f32);
        প্ৰবেশ_1 = std::make_shared<ov::op::v0::Convert>(প্ৰবেশ_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(প্ৰবেশ_0, প্ৰবেশ_1);
    যদি (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_গ্ৰন্থি_shared_ptr()->প্ৰবেশ(0));
        ov::mark_as_precision_sensitive(res.get_গ্ৰন্থি_shared_ptr()->প্ৰবেশ(1));
    }
    যদি (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        যদি (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->প্ৰবেশ(0));
        }
        res = output_convert;
    }
    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
