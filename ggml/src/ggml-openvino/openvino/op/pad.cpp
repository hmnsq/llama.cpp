#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <array>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/gather.hpp>
#অন্তর্ভাব <openvino/op/pad.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>
#অন্তর্ভাব <utility>
#অন্তর্ভাব <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

নামাকাশ {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> প্রবেশ,
                                            ধ্রুব std::array<int32_t, 8> & pads,
                                            ধ্রুব ov::Shape & প্রবেশ_shape) {
    ov::Output<ov::Node> result = std::move(প্রবেশ);

    ধ্রুব std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ধ্রুব std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    প্রতি (size_t axis = 0; axis < প্রবেশ_shape.size(); ++axis) {
        ধ্রুব int64_t প্রবেশ_dim = static_cast<int64_t>(প্রবেশ_shape[axis]);
        ধ্রুব int64_t pad_begin = pads_begin[axis];
        ধ্রুব int64_t pad_end = pads_end[axis];

        যদি (pad_begin == 0 && pad_end == 0) {
            অনুবর্ত;
        }

        FRONT_END_CHECK_IMPLEMENTED(প্রবেশ_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(প্রবেশ_dim + pad_begin + pad_end));
        প্রতি (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % প্রবেশ_dim;
            যদি (wrapped < 0) {
                wrapped += প্রবেশ_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    প্রত্যাগম result;
}

}  // namespace

OutputVector translate_pad(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    যদি (context.get_প্রবেশ_shape(0) == context.get_output_shape()) {
        প্রত্যাগম rename_outputs_with_suffix({প্রবেশ}, context.get_name());
    }

    ধ্রুব int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    ধ্রুব std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ধ্রুব তার্কিক circular = op_params[8] != 0;

    যদি (circular) {
        auto res = translate_circular_pad(প্রবেশ, pads, context.get_প্রবেশ_shape(0).to_shape());
        প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
    }

    ধ্রুব std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ধ্রুব std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_মান = ov::op::v0::Constant::create(context.get_প্রবেশ_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(প্রবেশ, pads_begin_গ্রন্থি, pads_end_গ্রন্থি, pad_মান, ov::op::PadMode::CONSTANT);

    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
