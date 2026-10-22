#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <openvino/op/avg_pool.hpp>
#অন্তর্ভাব <openvino/op/max_pool.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_pool_2d(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    ধ্রুব int32_t * params = context.get_output_op_params();

    ধ্রুব পূর্ণাঙ্ক k0 = params[1];
    ধ্রুব পূর্ণাঙ্ক k1 = params[2];
    ধ্রুব পূর্ণাঙ্ক s0 = params[3];
    ধ্রুব পূর্ণাঙ্ক s1 = params[4];
    ধ্রুব পূর্ণাঙ্ক p0 = params[5];
    ধ্রুব পূর্ণাঙ্ক p1 = params[6];

    ধ্রুব পূর্ণাঙ্ক op_case = context.get_op_case();
    ov::Output<Node> প্রবেশ = context.get_প্রবেশ(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_end{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    বিকল্প (op_case) {
    পক্ষ 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(প্রবেশ, strides, pads_begin, pads_end, kernel);
        ভঙ্গ;
    }
    পক্ষ 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(প্রবেশ, strides, pads_begin, pads_end, kernel, অসত্য);
        ভঙ্গ;
    }
    স্বভাব:
        ভঙ্গ;
    }
    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
