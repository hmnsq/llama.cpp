#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <array>
#شامل <cstdint>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/pad.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <utility>
#شامل <vector>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

فضاینام {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> درون_داد,
                                            ثابت std::array<int32_t, 8> & pads,
                                            ثابت ov::Shape & درون_داد_shape) {
    ov::Output<ov::Node> result = std::move(درون_داد);

    ثابت std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    برای (size_t axis = 0; axis < درون_داد_shape.size(); ++axis) {
        ثابت int64_t درون_داد_dim = static_cast<int64_t>(درون_داد_shape[axis]);
        ثابت int64_t pad_begin = pads_begin[axis];
        ثابت int64_t pad_end = pads_end[axis];

        اگر (pad_begin == 0 && pad_end == 0) {
            ادامه;
        }

        FRONT_END_CHECK_IMPLEMENTED(درون_داد_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(درون_داد_dim + pad_begin + pad_end));
        برای (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % درون_داد_dim;
            اگر (wrapped < 0) {
                wrapped += درون_داد_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    بازگشت result;
}

}  // namespace

OutputVector translate_pad(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto درون_داد = process_view_درون_داد_new(context, 0);
    اگر (context.get_درون_داد_shape(0) == context.get_output_shape()) {
        بازگشت rename_outputs_with_suffix({درون_داد}, context.get_name());
    }

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    ثابت std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ثابت منطقی circular = op_params[8] != 0;

    اگر (circular) {
        auto res = translate_circular_pad(درون_داد, pads, context.get_درون_داد_shape(0).to_shape());
        بازگشت rename_outputs_with_suffix({res}, context.get_name());
    }

    ثابت std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_گره = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_گره = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_مقدار = ov::op::v0::Constant::create(context.get_درون_داد_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(درون_داد, pads_begin_گره, pads_end_گره, pad_مقدار, ov::op::PadMode::CONSTANT);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
