#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"
#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/type.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/divide.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/shape_of.hpp>
#அந்தர்பாவ <openvino/op/sigmoid.hpp>
#அந்தர்பாவ <openvino/op/swish.hpp>
#அந்தர்பாவ <openvino/op/tile.hpp>
#அந்தர்பாவ <openvino/op/util/precision_sensitive_attribute.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

நாமாகாஶ {

தருக்க is_silu_div_pattern(மாறிலி ov::Output<ov::Node> & numerator,
                         மாறிலி ov::Output<ov::Node> & denominator,
                         மாறிலி முனை_Context & context) {
    எனில் (context.get_உள்ளீடு_size() != 2) {
        திரும்பு பொய்;
    }

    மாறிலி auto * unary_op = reinterpret_cast<மாறிலி ggml_unary_op *>(context.get_உள்ளீடு_op_params(0));
    எனில் (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        திரும்பு பொய்;
    }

    மாறிலி auto denom_முனை = denominator.get_முனை_shared_ptr();

    எனில் (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_முனை_shared_ptr())) {
        திரும்பு swish->உள்ளீடு_மதிப்பு(0).get_முனை_shared_ptr() == denom_முனை;
    }
    திரும்பு பொய்;
}

ov::Output<ov::Node> repeat_உள்ளீடு_to_match(மாறிலி முனை_Context & context,
                                           மாறிலி ov::Output<ov::Node> & உள்ளீடு,
                                           மாறிலி ov::Output<ov::Node> & target,
                                           size_t உள்ளீடு_index) {
    மாறிலி auto உள்ளீடு_shape = context.get_உள்ளீடு_shape(உள்ளீடு_index);
    மாறிலி auto target_shape = context.get_உள்ளீடு_shape(0);

    எனில் (உள்ளீடு_shape == target_shape) {
        திரும்பு உள்ளீடு;
    }

    எனில் (உள்ளீடு_shape.rank().is_static() && target_shape.rank().is_static()) {
        மாறிலி auto rank = static_cast<size_t>(உள்ளீடு_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        தருக்க needs_repeat = பொய்;

        க்கு (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(உள்ளீடு_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            மாறிலி int64_t உள்ளீடு_dim = உள்ளீடு_shape[axis].get_length();
            மாறிலி int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(உள்ளீடு_dim > 0 && target_dim > 0 && target_dim % உள்ளீடு_dim == 0,
                                          "DIV input shape ", உள்ளீடு_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / உள்ளீடு_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        எனில் (!needs_repeat) {
            திரும்பு உள்ளீடு;
        }

        auto repeats_முனை = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        திரும்பு std::make_shared<ov::op::v0::Tile>(உள்ளீடு, repeats_முனை);
    }

    auto உள்ளீடு_shape_முனை = std::make_shared<ov::op::v3::ShapeOf>(உள்ளீடு, ov::element::i64);
    auto target_shape_முனை = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_முனை = std::make_shared<ov::op::v1::Divide>(target_shape_முனை, உள்ளீடு_shape_முனை);
    திரும்பு std::make_shared<ov::op::v0::Tile>(உள்ளீடு, repeats_முனை);
}

}  // namespace

OutputVector translate_div(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 2, 2);

    auto உள்ளீடு_0 = process_view_உள்ளீடு_new(context, 0);
    auto உள்ளீடு_1 = process_view_உள்ளீடு_new(context, 1);

    எனில் (is_silu_div_pattern(உள்ளீடு_0, உள்ளீடு_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(உள்ளீடு_1);
        எனில் (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        திரும்பு rename_outputs_with_suffix({res}, context.get_name());
    }

    உள்ளீடு_1 = repeat_உள்ளீடு_to_match(context, உள்ளீடு_1, உள்ளீடு_0, 1);

    மாறிலி auto output_type = context.get_output_type();
    மாறிலி தருக்க use_f32_compute = உள்ளீடு_0.get_element_type() != ov::element::f32 ||
                                 உள்ளீடு_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    எனில் (use_f32_compute) {
        உள்ளீடு_0 = std::make_shared<ov::op::v0::Convert>(உள்ளீடு_0, ov::element::f32);
        உள்ளீடு_1 = std::make_shared<ov::op::v0::Convert>(உள்ளீடு_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(உள்ளீடு_0, உள்ளீடு_1);
    எனில் (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_முனை_shared_ptr()->உள்ளீடு(0));
        ov::mark_as_precision_sensitive(res.get_முனை_shared_ptr()->உள்ளீடு(1));
    }
    எனில் (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        எனில் (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->உள்ளீடு(0));
        }
        res = output_convert;
    }
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
