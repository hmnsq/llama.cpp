#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <array>
#ضمن <cstdint>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/pad.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <utility>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

نطاق {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> input,
                                            ثابت std::array<int32_t, 8> & pads,
                                            ثابت ov::Shape & input_shape) {
    ov::Output<ov::Node> result = std::move(input);

    ثابت std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    لكل (size_t axis = 0; axis < input_shape.size(); ++axis) {
        ثابت int64_t input_dim = static_cast<int64_t>(input_shape[axis]);
        ثابت int64_t pad_begin = pads_begin[axis];
        ثابت int64_t pad_end = pads_end[axis];

        𐡄𐡍 (pad_begin == 0 && pad_end == 0) {
            استمر;
        }

        FRONT_END_CHECK_IMPLEMENTED(input_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(input_dim + pad_begin + pad_end));
        لكل (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % input_dim;
            𐡄𐡍 (wrapped < 0) {
                wrapped += input_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    أرجع result;
}

}  // namespace

OutputVector translate_pad(ثابت NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);
    𐡄𐡍 (context.get_input_shape(0) == context.get_output_shape()) {
        أرجع rename_outputs_with_suffix({input}, context.get_name());
    }

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    ثابت std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ثابت منطقي circular = op_params[8] != 0;

    𐡄𐡍 (circular) {
        auto res = translate_circular_pad(input, pads, context.get_input_shape(0).to_shape());
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    ثابت std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_node = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_node = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_value = ov::op::v0::Constant::create(context.get_input_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(input, pads_begin_node, pads_end_node, pad_value, ov::op::PadMode::CONSTANT);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
