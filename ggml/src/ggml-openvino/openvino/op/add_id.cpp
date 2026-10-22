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

ساكن ov::Output<ov::Node> reshape_جمع_کن_id_درون_داد_مقصد_2d(ثابت ov::Output<ov::Node> & درون_داد,
                                                       ثابت ov::PartialShape & درون_داد_shape,
                                                       ثابت std::vector<صحيح> & dims) {
    ثابت خودکار & actual_shape = درون_داد.get_partial_shape();
    إذا (actual_shape.rank().is_static() && actual_shape.rank().get_درازی() == 2) {
        أرجع درون_داد;
    }

    إذا (درون_داد_shape.rank().is_static() && درون_داد_shape.rank().get_درازی() == 2) {
        أرجع درون_داد;
    }

    خودکار shape = std::make_shared<ov::op::v3::Shape_از>(درون_داد, ov::element::i64);
    أرجع std::make_shared<ov::op::v1::Reshape>(درون_داد, get_dimensions(shape, dims), خطأ);
}

Output_وکتور translate_جمع_کن_id(ثابت گره_Context & context) {
    رقم_inputs_check(context, 3, 3);

    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار bias = process_view_درون_داد_new(context, 1);
    خودکار ids = process_view_درون_داد_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_جمع_کن_id_درون_داد_مقصد_2d(bias, context.get_درون_داد_shape(1), {2, 3});
    ids = reshape_جمع_کن_id_درون_داد_مقصد_2d(ids, context.get_درون_داد_shape(2), {2, 3});

    إذا (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    خودکار gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::Shape_از>(درون_داد, ov::element::i64), خطأ);

    إذا (selected_bias.get_element_type() != درون_داد.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, درون_داد.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::جمع_کن>(درون_داد, selected_bias);
    ثابت خودکار output_type = context.get_output_type();
    إذا (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
