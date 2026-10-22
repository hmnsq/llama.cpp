#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <array>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/gather.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/pad.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/shape_of.hpp>
#ଅନ୍ତର୍ଭାଵ <utility>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

ନାମାକାଶ {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> ନିବେଶ,
                                            ଅଚଳ std::array<int32_t, 8> & pads,
                                            ଅଚଳ ov::Shape & ନିବେଶ_shape) {
    ov::Output<ov::Node> result = std::move(ନିବେଶ);

    ଅଚଳ std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ଅଚଳ std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    ପାଇଁ (size_t axis = 0; axis < ନିବେଶ_shape.size(); ++axis) {
        ଅଚଳ int64_t ନିବେଶ_dim = static_cast<int64_t>(ନିବେଶ_shape[axis]);
        ଅଚଳ int64_t pad_begin = pads_begin[axis];
        ଅଚଳ int64_t pad_end = pads_end[axis];

        ଯଦି (pad_begin == 0 && pad_end == 0) {
            ଜାରି;
        }

        FRONT_END_CHECK_IMPLEMENTED(ନିବେଶ_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(ନିବେଶ_dim + pad_begin + pad_end));
        ପାଇଁ (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % ନିବେଶ_dim;
            ଯଦି (wrapped < 0) {
                wrapped += ନିବେଶ_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    ଫେରନ୍ତୁ result;
}

}  // namespace

OutputVector translate_pad(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    ଯଦି (context.get_ନିବେଶ_shape(0) == context.get_output_shape()) {
        ଫେରନ୍ତୁ rename_outputs_with_suffix({ନିବେଶ}, context.get_name());
    }

    ଅଚଳ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    ଅଚଳ std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ଅଚଳ ତାର୍କିକ circular = op_params[8] != 0;

    ଯଦି (circular) {
        auto res = translate_circular_pad(ନିବେଶ, pads, context.get_ନିବେଶ_shape(0).to_shape());
        ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
    }

    ଅଚଳ std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ଅଚଳ std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_ମୂଲ୍ୟ = ov::op::v0::Constant::create(context.get_ନିବେଶ_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(ନିବେଶ, pads_begin_ଗଣ୍ଠି, pads_end_ଗଣ୍ଠି, pad_ମୂଲ୍ୟ, ov::op::PadMode::CONSTANT);

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
