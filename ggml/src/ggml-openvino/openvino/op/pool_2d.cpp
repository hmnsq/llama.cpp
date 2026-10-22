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

OutputVector translate_pool_2d(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    ধ্রুবক int32_t * params = context.get_output_op_params();

    ধ্রুবক পূর্ণসংখ্যা k0 = params[1];
    ধ্রুবক পূর্ণসংখ্যা k1 = params[2];
    ধ্রুবক পূর্ণসংখ্যা s0 = params[3];
    ধ্রুবক পূর্ণসংখ্যা s1 = params[4];
    ধ্রুবক পূর্ণসংখ্যা p0 = params[5];
    ধ্রুবক পূর্ণসংখ্যা p1 = params[6];

    ধ্রুবক পূর্ণসংখ্যা op_case = context.get_op_case();
    ov::Output<Node> প্রবেশ = context.get_প্রবেশ(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_end{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    নির্বাচন (op_case) {
    ক্ষেত্র 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(প্রবেশ, strides, pads_begin, pads_end, kernel);
        বিরতি;
    }
    ক্ষেত্র 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(প্রবেশ, strides, pads_begin, pads_end, kernel, মিথ্যা);
        বিরতি;
    }
    পূর্বনির্ধারিত:
        বিরতি;
    }
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
