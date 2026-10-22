#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/avg_pool.hpp>
#ضمن <openvino/op/max_pool.hpp>
#ضمن <openvino/op/convert.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ويڪٽر translate_pool_2d(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);
    ثابت int32_t * params = context.get_output_op_params();

    ثابت صحيح k0 = params[1];
    ثابت صحيح k1 = params[2];
    ثابت صحيح s0 = params[3];
    ثابت صحيح s1 = params[4];
    ثابت صحيح p0 = params[5];
    ثابت صحيح p1 = params[6];

    ثابت صحيح op_case = context.get_op_case();
    ov::Output<Node> داخلا_2 = context.get_داخلا(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_پڇاڙي{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    بدل (op_case) {
    حالة 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(داخلا_2, strides, pads_begin, pads_پڇاڙي, kernel);
        اكسر;
    }
    حالة 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(داخلا_2, strides, pads_begin, pads_پڇاڙي, kernel, خطأ);
        اكسر;
    }
    افتراضي:
        اكسر;
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
