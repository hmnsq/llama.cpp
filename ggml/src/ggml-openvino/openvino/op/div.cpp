#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"
#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/core/type.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/divide.hpp>
#অন্তর্ভাব <openvino/op/multiply.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>
#অন্তর্ভাব <openvino/op/sigmoid.hpp>
#অন্তর্ভাব <openvino/op/swish.hpp>
#অন্তর্ভাব <openvino/op/tile.hpp>
#অন্তর্ভাব <openvino/op/util/precision_sensitive_attribute.hpp>
#অন্তর্ভাব <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

নামাকাশ {

তার্কিক is_silu_div_pattern(ধ্রুব ov::Output<ov::Node> & numerator,
                         ধ্রুব ov::Output<ov::Node> & denominator,
                         ধ্রুব গ্রন্থি_Context & context) {
    যদি (context.get_প্রবেশ_size() != 2) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব auto * unary_op = reinterpret_cast<ধ্রুব ggml_unary_op *>(context.get_প্রবেশ_op_params(0));
    যদি (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব auto denom_গ্রন্থি = denominator.get_গ্রন্থি_shared_ptr();

    যদি (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_গ্রন্থি_shared_ptr())) {
        প্রত্যাগম swish->প্রবেশ_মান(0).get_গ্রন্থি_shared_ptr() == denom_গ্রন্থি;
    }
    প্রত্যাগম অসত্য;
}

ov::Output<ov::Node> repeat_প্রবেশ_to_match(ধ্রুব গ্রন্থি_Context & context,
                                           ধ্রুব ov::Output<ov::Node> & প্রবেশ,
                                           ধ্রুব ov::Output<ov::Node> & target,
                                           size_t প্রবেশ_index) {
    ধ্রুব auto প্রবেশ_shape = context.get_প্রবেশ_shape(প্রবেশ_index);
    ধ্রুব auto target_shape = context.get_প্রবেশ_shape(0);

    যদি (প্রবেশ_shape == target_shape) {
        প্রত্যাগম প্রবেশ;
    }

    যদি (প্রবেশ_shape.rank().is_static() && target_shape.rank().is_static()) {
        ধ্রুব auto rank = static_cast<size_t>(প্রবেশ_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        তার্কিক needs_repeat = অসত্য;

        প্রতি (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(প্রবেশ_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ধ্রুব int64_t প্রবেশ_dim = প্রবেশ_shape[axis].get_length();
            ধ্রুব int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(প্রবেশ_dim > 0 && target_dim > 0 && target_dim % প্রবেশ_dim == 0,
                                          "DIV input shape ", প্রবেশ_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / প্রবেশ_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        যদি (!needs_repeat) {
            প্রত্যাগম প্রবেশ;
        }

        auto repeats_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        প্রত্যাগম std::make_shared<ov::op::v0::Tile>(প্রবেশ, repeats_গ্রন্থি);
    }

    auto প্রবেশ_shape_গ্রন্থি = std::make_shared<ov::op::v3::ShapeOf>(প্রবেশ, ov::element::i64);
    auto target_shape_গ্রন্থি = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_গ্রন্থি = std::make_shared<ov::op::v1::Divide>(target_shape_গ্রন্থি, প্রবেশ_shape_গ্রন্থি);
    প্রত্যাগম std::make_shared<ov::op::v0::Tile>(প্রবেশ, repeats_গ্রন্থি);
}

}  // namespace

OutputVector translate_div(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    auto প্রবেশ_0 = process_view_প্রবেশ_new(context, 0);
    auto প্রবেশ_1 = process_view_প্রবেশ_new(context, 1);

    যদি (is_silu_div_pattern(প্রবেশ_0, প্রবেশ_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(প্রবেশ_1);
        যদি (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
    }

    প্রবেশ_1 = repeat_প্রবেশ_to_match(context, প্রবেশ_1, প্রবেশ_0, 1);

    ধ্রুব auto output_type = context.get_output_type();
    ধ্রুব তার্কিক use_f32_compute = প্রবেশ_0.get_element_type() != ov::element::f32 ||
                                 প্রবেশ_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    যদি (use_f32_compute) {
        প্রবেশ_0 = std::make_shared<ov::op::v0::Convert>(প্রবেশ_0, ov::element::f32);
        প্রবেশ_1 = std::make_shared<ov::op::v0::Convert>(প্রবেশ_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(প্রবেশ_0, প্রবেশ_1);
    যদি (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_গ্রন্থি_shared_ptr()->প্রবেশ(0));
        ov::mark_as_precision_sensitive(res.get_গ্রন্থি_shared_ptr()->প্রবেশ(1));
    }
    যদি (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        যদি (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->প্রবেশ(0));
        }
        res = output_convert;
    }
    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
