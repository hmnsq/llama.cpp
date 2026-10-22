#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <array>
#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/gather.hpp>
#অন্তর্ভাৱ <openvino/op/pad.hpp>
#অন্তর্ভাৱ <openvino/op/reshape.hpp>
#অন্তর্ভাৱ <openvino/op/shape_of.hpp>
#অন্তর্ভাৱ <utility>
#অন্তর্ভাৱ <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

নামাকাশ {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> প্ৰবেশ,
                                            ধ্রুৱ std::array<int32_t, 8> & pads,
                                            ধ্রুৱ ov::Shape & প্ৰবেশ_shape) {
    ov::Output<ov::Node> result = std::move(প্ৰবেশ);

    ধ্রুৱ std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ধ্রুৱ std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    বাবে (size_t axis = 0; axis < প্ৰবেশ_shape.size(); ++axis) {
        ধ্রুৱ int64_t প্ৰবেশ_dim = static_cast<int64_t>(প্ৰবেশ_shape[axis]);
        ধ্রুৱ int64_t pad_begin = pads_begin[axis];
        ধ্রুৱ int64_t pad_end = pads_end[axis];

        যদি (pad_begin == 0 && pad_end == 0) {
            অনুৱর্ত;
        }

        FRONT_END_CHECK_IMPLEMENTED(প্ৰবেশ_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(প্ৰবেশ_dim + pad_begin + pad_end));
        বাবে (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % প্ৰবেশ_dim;
            যদি (wrapped < 0) {
                wrapped += প্ৰবেশ_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    ঘূৰাই result;
}

}  // namespace

OutputVector translate_pad(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্ৰবেশ = process_view_প্ৰবেশ_new(context, 0);
    যদি (context.get_প্ৰবেশ_shape(0) == context.get_output_shape()) {
        ঘূৰাই rename_outputs_with_suffix({প্ৰবেশ}, context.get_name());
    }

    ধ্রুৱ int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    ধ্রুৱ std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ধ্রুৱ তার্কিক circular = op_params[8] != 0;

    যদি (circular) {
        auto res = translate_circular_pad(প্ৰবেশ, pads, context.get_প্ৰবেশ_shape(0).to_shape());
        ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
    }

    ধ্রুৱ std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ধ্রুৱ std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_মান = ov::op::v0::Constant::create(context.get_প্ৰবেশ_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(প্ৰবেশ, pads_begin_গ্ৰন্থি, pads_end_গ্ৰন্থি, pad_মান, ov::op::PadMode::CONSTANT);

    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
