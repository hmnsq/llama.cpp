#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <array>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/gather.hpp>
#અન્તર્ભાવ <openvino/op/pad.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>
#અન્તર્ભાવ <utility>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

નામાકાશ {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> આગત,
                                            અચલ std::array<int32_t, 8> & pads,
                                            અચલ ov::Shape & આગત_shape) {
    ov::Output<ov::Node> result = std::move(આગત);

    અચલ std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    અચલ std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    માટે (size_t axis = 0; axis < આગત_shape.size(); ++axis) {
        અચલ int64_t આગત_dim = static_cast<int64_t>(આગત_shape[axis]);
        અચલ int64_t pad_begin = pads_begin[axis];
        અચલ int64_t pad_end = pads_end[axis];

        જો (pad_begin == 0 && pad_end == 0) {
            ચાલુ;
        }

        FRONT_END_CHECK_IMPLEMENTED(આગત_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(આગત_dim + pad_begin + pad_end));
        માટે (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % આગત_dim;
            જો (wrapped < 0) {
                wrapped += આગત_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    પાછા result;
}

}  // namespace

OutputVector translate_pad(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    auto આગત = process_view_આગત_new(context, 0);
    જો (context.get_આગત_shape(0) == context.get_output_shape()) {
        પાછા rename_outputs_with_suffix({આગત}, context.get_name());
    }

    અચલ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    અચલ std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    અચલ તાર્કિક circular = op_params[8] != 0;

    જો (circular) {
        auto res = translate_circular_pad(આગત, pads, context.get_આગત_shape(0).to_shape());
        પાછા rename_outputs_with_suffix({res}, context.get_name());
    }

    અચલ std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    અચલ std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_કિંમત = ov::op::v0::Constant::create(context.get_આગત_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(આગત, pads_begin_ગાંઠ, pads_end_ગાંઠ, pad_કિંમત, ov::op::PadMode::CONSTANT);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
