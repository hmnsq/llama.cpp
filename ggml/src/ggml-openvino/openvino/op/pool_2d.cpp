#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/avg_pool.hpp>
#شامل <openvino/op/max_pool.hpp>
#شامل <openvino/op/convert.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_pool_2d(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    ثابت int32_t * params = context.get_output_op_params();

    ثابت صحیح k0 = params[1];
    ثابت صحیح k1 = params[2];
    ثابت صحیح s0 = params[3];
    ثابت صحیح s1 = params[4];
    ثابت صحیح p0 = params[5];
    ثابت صحیح p1 = params[6];

    ثابت صحیح op_case = context.get_op_case();
    ov::Output<Node> درون_داد = context.get_درون_داد(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_end{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    انتخاب (op_case) {
    حالت 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(درون_داد, strides, pads_begin, pads_end, kernel);
        بشکن;
    }
    حالت 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(درون_داد, strides, pads_begin, pads_end, kernel, نادرست);
        بشکن;
    }
    پیشفرض:
        بشکن;
    }
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
