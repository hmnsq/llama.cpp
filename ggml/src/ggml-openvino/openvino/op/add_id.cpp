#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

ساكن ov::Output<ov::Node> reshape_جمع_کړه_id_ننوت_منزل_2d(ثابت ov::Output<ov::Node> & ننوت,
                                                       ثابت ov::PartialShape & ننوت_shape,
                                                       ثابت std::vector<صحيح> & dims) {
    ثابت خپلکاری & actual_shape = ننوت.get_partial_shape();
    إذا (actual_shape.rank().is_static() && actual_shape.rank().get_اوږدتیا() == 2) {
        أرجع ننوت;
    }

    إذا (ننوت_shape.rank().is_static() && ننوت_shape.rank().get_اوږدتیا() == 2) {
        أرجع ننوت;
    }

    خپلکاری shape = std::make_shared<ov::op::v3::Shape_د>(ننوت, ov::element::i64);
    أرجع std::make_shared<ov::op::v1::Reshape>(ننوت, get_dimensions(shape, dims), خطأ);
}

Output_ویکټور translate_جمع_کړه_id(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 3, 3);

    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    خپلکاری bias = process_view_ننوت_new(context, 1);
    خپلکاری ids = process_view_ننوت_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_جمع_کړه_id_ننوت_منزل_2d(bias, context.get_ننوت_shape(1), {2, 3});
    ids = reshape_جمع_کړه_id_ننوت_منزل_2d(ids, context.get_ننوت_shape(2), {2, 3});

    إذا (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    خپلکاری gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::Shape_د>(ننوت, ov::element::i64), خطأ);

    إذا (selected_bias.get_element_type() != ننوت.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, ننوت.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::جمع_کړه>(ننوت, selected_bias);
    ثابت خپلکاری output_type = context.get_output_type();
    إذا (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
