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

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> ننوت,
                                            ثابت std::array<int32_t, 8> & pads,
                                            ثابت ov::Shape & ننوت_shape) {
    ov::Output<ov::Node> پایله = std::move(ننوت);

    ثابت std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::array<int32_t, 4> pads_پای_ټکی = {pads[7], pads[5], pads[3], pads[1]};

    لكل (size_t axis = 0; axis < ننوت_shape.size(); ++axis) {
        ثابت int64_t ننوت_dim = static_cast<int64_t>(ننوت_shape[axis]);
        ثابت int64_t pad_begin = pads_begin[axis];
        ثابت int64_t pad_پای_ټکی = pads_پای_ټکی[axis];

        إذا (pad_begin == 0 && pad_پای_ټکی == 0) {
            استمر;
        }

        FRONT_پای_ټکی_CHECK_IMPLEMENTED(ننوت_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(ننوت_dim + pad_begin + pad_پای_ټکی));
        لكل (int64_t شاخص = 0; شاخص < static_cast<int64_t>(indices.size()); ++شاخص) {
            int64_t wrapped = (شاخص - pad_begin) % ننوت_dim;
            إذا (wrapped < 0) {
                wrapped += ننوت_dim;
            }
            indices[static_cast<size_t>(شاخص)] = wrapped;
        }

        خپلکاری gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        خپلکاری gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        پایله = std::make_shared<ov::op::v8::Gather>(پایله, gather_indices, gather_axis);
    }

    أرجع پایله;
}

}  // namespace

Output_ویکټور translate_pad(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    إذا (context.get_ننوت_shape(0) == context.get_output_shape()) {
        أرجع rename_outputs_with_suffix({ننوت}, context.get_name());
    }

    ثابت int32_t * op_params = context.get_output_op_params();
    FRONT_پای_ټکی_CHECK_IMPLEMENTED(op_params != تش_اشاره, "PAD requires output op params");

    ثابت std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    ثابت منطقي circular = op_params[8] != 0;

    إذا (circular) {
        خپلکاری res = translate_circular_pad(ننوت, pads, context.get_ننوت_shape(0).منزل_shape());
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    ثابت std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    ثابت std::vector<int64_t> pads_پای_ټکی = {pads[7], pads[5], pads[3], pads[1]};

    خپلکاری pads_begin_غوټه = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    خپلکاری pads_پای_ټکی_غوټه = ov::op::v0::Constant::create(ov::element::i64, {pads_پای_ټکی.size()}, pads_پای_ټکی);
    خپلکاری pad_ارزښت = ov::op::v0::Constant::create(context.get_ننوت_type(0), ov::Shape{}, {0});
    خپلکاری res =
        std::make_shared<ov::op::v1::Pad>(ننوت, pads_begin_غوټه, pads_پای_ټکی_غوټه, pad_ارزښت, ov::op::Pad_ډېرپېښ::CONSTANT);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
