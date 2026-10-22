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

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> داخلا_2,
                                            ثابت std::array<int32_t, 8> & pads,
                                            ثابت ov::Shape & داخلا_shape) {
    ov::Output<ov::Node> نتيجو = std::move(داخلا_2);

    ثابت std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::array<int32_t, 4> pads_پڇاڙي = {pads[7], pads[5], pads[3], pads[1]};

    لكل (size_t axis = 0; axis < داخلا_shape.size(); ++axis) {
        ثابت int64_t داخلا_dim = static_cast<int64_t>(داخلا_shape[axis]);
        ثابت int64_t pad_begin = pads_begin[axis];
        ثابت int64_t pad_پڇاڙي = pads_پڇاڙي[axis];

        إذا (pad_begin == 0 && pad_پڇاڙي == 0) {
            استمر;
        }

        FRONT_پڇاڙي_CHECK_IMPLEMENTED(داخلا_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(داخلا_dim + pad_begin + pad_پڇاڙي));
        لكل (int64_t اشاريو = 0; اشاريو < static_cast<int64_t>(indices.size()); ++اشاريو) {
            int64_t wrapped = (اشاريو - pad_begin) % داخلا_dim;
            إذا (wrapped < 0) {
                wrapped += داخلا_dim;
            }
            indices[static_cast<size_t>(اشاريو)] = wrapped;
        }

        خودڪار gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        خودڪار gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        نتيجو = std::make_shared<ov::op::v8::Gather>(نتيجو, gather_indices, gather_axis);
    }

    أرجع نتيجو;
}

}  // namespace

Output_ويڪٽر translate_pad(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    إذا (context.get_داخلا_shape(0) == context.get_output_shape()) {
        أرجع rename_outputs_with_suffix({داخلا_2}, context.get_name());
    }

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_پڇاڙي_CHECK_IMPLEMENTED(op_params != خالي_اشارو, "PAD requires output op params");

    ثابت std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ثابت منطقي circular = op_params[8] != 0;

    إذا (circular) {
        خودڪار res = translate_circular_pad(داخلا_2, pads, context.get_داخلا_shape(0).منزل_shape());
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    ثابت std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::vector<int64_t> pads_پڇاڙي = {pads[7], pads[5], pads[3], pads[1]};

    خودڪار pads_begin_ڳنڍ = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    خودڪار pads_پڇاڙي_ڳنڍ = ov::op::v0::Constant::create(ov::element::i64, {pads_پڇاڙي.size()}, pads_پڇاڙي);
    خودڪار pad_قدر = ov::op::v0::Constant::create(context.get_داخلا_type(0), ov::Shape{}, {0});
    خودڪار res =
        std::make_shared<ov::op::v1::Pad>(داخلا_2, pads_begin_ڳنڍ, pads_پڇاڙي_ڳنڍ, pad_قدر, ov::op::Pad_گهڻائي::CONSTANT);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
