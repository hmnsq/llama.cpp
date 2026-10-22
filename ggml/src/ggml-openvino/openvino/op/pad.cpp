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

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> درون_داد,
                                            ثابت std::array<int32_t, 8> & pads,
                                            ثابت ov::Shape & درون_داد_shape) {
    ov::Output<ov::Node> نتیجه = std::move(درون_داد);

    ثابت std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::array<int32_t, 4> pads_انتها = {pads[7], pads[5], pads[3], pads[1]};

    لكل (size_t axis = 0; axis < درون_داد_shape.size(); ++axis) {
        ثابت int64_t درون_داد_dim = static_cast<int64_t>(درون_داد_shape[axis]);
        ثابت int64_t pad_begin = pads_begin[axis];
        ثابت int64_t pad_انتها = pads_انتها[axis];

        إذا (pad_begin == 0 && pad_انتها == 0) {
            استمر;
        }

        FRONT_انتها_CHECK_IMPLEMENTED(درون_داد_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(درون_داد_dim + pad_begin + pad_انتها));
        لكل (int64_t شاخص = 0; شاخص < static_cast<int64_t>(indices.size()); ++شاخص) {
            int64_t wrapped = (شاخص - pad_begin) % درون_داد_dim;
            إذا (wrapped < 0) {
                wrapped += درون_داد_dim;
            }
            indices[static_cast<size_t>(شاخص)] = wrapped;
        }

        خودکار gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        خودکار gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        نتیجه = std::make_shared<ov::op::v8::Gather>(نتیجه, gather_indices, gather_axis);
    }

    أرجع نتیجه;
}

}  // namespace

Output_وکتور translate_pad(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    إذا (context.get_درون_داد_shape(0) == context.get_output_shape()) {
        أرجع rename_outputs_with_suffix({درون_داد}, context.get_name());
    }

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_انتها_CHECK_IMPLEMENTED(op_params != اشاره_تهی, "PAD requires output op params");

    ثابت std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ثابت منطقي circular = op_params[8] != 0;

    إذا (circular) {
        خودکار res = translate_circular_pad(درون_داد, pads, context.get_درون_داد_shape(0).مقصد_shape());
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    ثابت std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::vector<int64_t> pads_انتها = {pads[7], pads[5], pads[3], pads[1]};

    خودکار pads_begin_گره = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    خودکار pads_انتها_گره = ov::op::v0::Constant::create(ov::element::i64, {pads_انتها.size()}, pads_انتها);
    خودکار pad_مقدار = ov::op::v0::Constant::create(context.get_درون_داد_type(0), ov::Shape{}, {0});
    خودکار res =
        std::make_shared<ov::op::v1::Pad>(درون_داد, pads_begin_گره, pads_انتها_گره, pad_مقدار, ov::op::Pad_نما::CONSTANT);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
