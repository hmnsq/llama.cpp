#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/avg_pool.hpp>
#شامل <openvino/op/max_pool.hpp>
#شامل <openvino/op/convert.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_pool_2d(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    ثابت int32_t * params = context.get_output_op_params();

    ثابت صحیح k0 = params[1];
    ثابت صحیح k1 = params[2];
    ثابت صحیح s0 = params[3];
    ثابت صحیح s1 = params[4];
    ثابت صحیح p0 = params[5];
    ثابت صحیح p1 = params[6];

    ثابت صحیح op_case = context.get_op_case();
    ov::Output<Node> داخلہ = context.get_داخلہ(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_انتہا{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    انتخاب (op_case) {
    حالت 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(داخلہ, strides, pads_begin, pads_انتہا, kernel);
        توڑو;
    }
    حالت 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(داخلہ, strides, pads_begin, pads_انتہا, kernel, جھوٹ);
        توڑو;
    }
    طےشدہ:
        توڑو;
    }
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
