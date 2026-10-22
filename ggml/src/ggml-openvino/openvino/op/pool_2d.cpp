#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <openvino/op/avg_pool.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/max_pool.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_pool_2d(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    ଅଚଳ int32_t * params = context.get_output_op_params();

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k0 = params[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k1 = params[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = params[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1 = params[4];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = params[5];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p1 = params[6];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_case = context.get_op_case();
    ov::Output<Node> ନିବେଶ = context.get_ନିବେଶ(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_end{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    ବାଛ (op_case) {
    ସ୍ଥିତି 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(ନିବେଶ, strides, pads_begin, pads_end, kernel);
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(ନିବେଶ, strides, pads_begin, pads_end, kernel, ମିଥ୍ୟା);
        ଭାଙ୍ଗ;
    }
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        ଭାଙ୍ଗ;
    }
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
