#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <array>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/gather.hpp>
#அந்தர்பாவ <openvino/op/pad.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/shape_of.hpp>
#அந்தர்பாவ <utility>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

நாமாகாஶ {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> உள்ளீடு,
                                            மாறிலி std::array<int32_t, 8> & pads,
                                            மாறிலி ov::Shape & உள்ளீடு_shape) {
    ov::Output<ov::Node> result = std::move(உள்ளீடு);

    மாறிலி std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    மாறிலி std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    க்கு (size_t axis = 0; axis < உள்ளீடு_shape.size(); ++axis) {
        மாறிலி int64_t உள்ளீடு_dim = static_cast<int64_t>(உள்ளீடு_shape[axis]);
        மாறிலி int64_t pad_begin = pads_begin[axis];
        மாறிலி int64_t pad_end = pads_end[axis];

        எனில் (pad_begin == 0 && pad_end == 0) {
            தொடர்;
        }

        FRONT_END_CHECK_IMPLEMENTED(உள்ளீடு_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(உள்ளீடு_dim + pad_begin + pad_end));
        க்கு (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % உள்ளீடு_dim;
            எனில் (wrapped < 0) {
                wrapped += உள்ளீடு_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    திரும்பு result;
}

}  // namespace

OutputVector translate_pad(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    எனில் (context.get_உள்ளீடு_shape(0) == context.get_output_shape()) {
        திரும்பு rename_outputs_with_suffix({உள்ளீடு}, context.get_name());
    }

    மாறிலி int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    மாறிலி std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    மாறிலி தருக்க circular = op_params[8] != 0;

    எனில் (circular) {
        auto res = translate_circular_pad(உள்ளீடு, pads, context.get_உள்ளீடு_shape(0).to_shape());
        திரும்பு rename_outputs_with_suffix({res}, context.get_name());
    }

    மாறிலி std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    மாறிலி std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_முனை = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_முனை = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_மதிப்பு = ov::op::v0::Constant::create(context.get_உள்ளீடு_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(உள்ளீடு, pads_begin_முனை, pads_end_முனை, pad_மதிப்பு, ov::op::PadMode::CONSTANT);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
