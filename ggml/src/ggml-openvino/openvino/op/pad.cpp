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

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

نامخلا {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> داخلہ,
                                            ثابت std::array<int32_t, 8> & pads,
                                            ثابت ov::Shape & داخلہ_shape) {
    ov::Output<ov::Node> نتیجہ = std::move(داخلہ);

    ثابت std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::array<int32_t, 4> pads_انتہا = {pads[7], pads[5], pads[3], pads[1]};

    برائے (size_t axis = 0; axis < داخلہ_shape.size(); ++axis) {
        ثابت int64_t داخلہ_dim = static_cast<int64_t>(داخلہ_shape[axis]);
        ثابت int64_t pad_begin = pads_begin[axis];
        ثابت int64_t pad_انتہا = pads_انتہا[axis];

        اگر (pad_begin == 0 && pad_انتہا == 0) {
            جاری;
        }

        FRONT_انتہا_CHECK_IMPLEMENTED(داخلہ_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(داخلہ_dim + pad_begin + pad_انتہا));
        برائے (int64_t اشاریہ = 0; اشاریہ < static_cast<int64_t>(indices.size()); ++اشاریہ) {
            int64_t wrapped = (اشاریہ - pad_begin) % داخلہ_dim;
            اگر (wrapped < 0) {
                wrapped += داخلہ_dim;
            }
            indices[static_cast<size_t>(اشاریہ)] = wrapped;
        }

        خودکار gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        خودکار gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        نتیجہ = std::make_shared<ov::op::v8::Gather>(نتیجہ, gather_indices, gather_axis);
    }

    واپس نتیجہ;
}

}  // namespace

Output_سمتیہ translate_pad(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    اگر (context.get_داخلہ_shape(0) == context.get_output_shape()) {
        واپس rename_outputs_with_suffix({داخلہ}, context.get_name());
    }

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_انتہا_CHECK_IMPLEMENTED(op_params != خالی_اشاریہ, "PAD requires output op params");

    ثابت std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ثابت منطقی circular = op_params[8] != 0;

    اگر (circular) {
        خودکار res = translate_circular_pad(داخلہ, pads, context.get_داخلہ_shape(0).مقصد_shape());
        واپس rename_outputs_with_suffix({res}, context.get_name());
    }

    ثابت std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::vector<int64_t> pads_انتہا = {pads[7], pads[5], pads[3], pads[1]};

    خودکار pads_begin_گرہ = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    خودکار pads_انتہا_گرہ = ov::op::v0::Constant::create(ov::element::i64, {pads_انتہا.size()}, pads_انتہا);
    خودکار pad_قدر = ov::op::v0::Constant::create(context.get_داخلہ_type(0), ov::Shape{}, {0});
    خودکار res =
        std::make_shared<ov::op::v1::Pad>(داخلہ, pads_begin_گرہ, pads_انتہا_گرہ, pad_قدر, ov::op::Pad_منوال::CONSTANT);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
