#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "ggml.h"

#ضمن <memory>
#ضمن <openvino/core/type.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/sigmoid.hpp>
#ضمن <openvino/op/swish.hpp>
#ضمن <openvino/op/tile.hpp>
#ضمن <openvino/op/util/precision_sensitive_attribute.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

نطاق {

منطقي is_silu_div_pattern(ثابت ov::Output<ov::Node> & numerator,
                         ثابت ov::Output<ov::Node> & denominator,
                         ثابت NodeContext & context) {
    ܐܢ (context.get_input_size() != 2) {
        ܦܢܐ ܕܓܠܐ;
    }

    ثابت auto * unary_op = reinterpret_cast<ثابت ggml_unary_op *>(context.get_input_op_params(0));
    ܐܢ (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        ܦܢܐ ܕܓܠܐ;
    }

    ثابت auto denom_node = denominator.get_node_shared_ptr();

    ܐܢ (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_node_shared_ptr())) {
        ܦܢܐ swish->input_value(0).get_node_shared_ptr() == denom_node;
    }
    ܦܢܐ ܕܓܠܐ;
}

ov::Output<ov::Node> repeat_input_to_match(ثابت NodeContext & context,
                                           ثابت ov::Output<ov::Node> & input,
                                           ثابت ov::Output<ov::Node> & target,
                                           size_t input_index) {
    ثابت auto input_shape = context.get_input_shape(input_index);
    ثابت auto target_shape = context.get_input_shape(0);

    ܐܢ (input_shape == target_shape) {
        ܦܢܐ input;
    }

    ܐܢ (input_shape.rank().is_static() && target_shape.rank().is_static()) {
        ثابت auto rank = static_cast<size_t>(input_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        منطقي needs_repeat = ܕܓܠܐ;

        لكل (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(input_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ثابت int64_t input_dim = input_shape[axis].get_length();
            ثابت int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(input_dim > 0 && target_dim > 0 && target_dim % input_dim == 0,
                                          "DIV input shape ", input_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / input_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        ܐܢ (!needs_repeat) {
            ܦܢܐ input;
        }

        auto repeats_node = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        ܦܢܐ std::make_shared<ov::op::v0::Tile>(input, repeats_node);
    }

    auto input_shape_node = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    auto target_shape_node = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_node = std::make_shared<ov::op::v1::Divide>(target_shape_node, input_shape_node);
    ܦܢܐ std::make_shared<ov::op::v0::Tile>(input, repeats_node);
}

}  // namespace

OutputVector translate_div(ثابت NodeContext & context) {
    num_inputs_check(context, 2, 2);

    auto input_0 = process_view_input_new(context, 0);
    auto input_1 = process_view_input_new(context, 1);

    ܐܢ (is_silu_div_pattern(input_0, input_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(input_1);
        ܐܢ (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        ܦܢܐ rename_outputs_with_suffix({res}, context.get_name());
    }

    input_1 = repeat_input_to_match(context, input_1, input_0, 1);

    ثابت auto output_type = context.get_output_type();
    ثابت منطقي use_f32_compute = input_0.get_element_type() != ov::element::f32 ||
                                 input_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    ܐܢ (use_f32_compute) {
        input_0 = std::make_shared<ov::op::v0::Convert>(input_0, ov::element::f32);
        input_1 = std::make_shared<ov::op::v0::Convert>(input_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(input_0, input_1);
    ܐܢ (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_node_shared_ptr()->input(0));
        ov::mark_as_precision_sensitive(res.get_node_shared_ptr()->input(1));
    }
    ܐܢ (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        ܐܢ (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->input(0));
        }
        res = output_convert;
    }
    ܦܢܐ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
