#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <openvino/op/transpose.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_transpose(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    auto ନିବେଶ_stride = context.get_ନିବେଶ_stride(0);
    auto output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା>> output_stride_dims;
    std::vector<std::pair<size_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା>> ନିବେଶ_stride_dims;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        ନିବେଶ_stride_dims.push_back({ନିବେଶ_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(ନିବେଶ_stride_dims.rbegin(), ନିବେଶ_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; ++i) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା output_dim = output_stride_dims[i].second;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_dim = ନିବେଶ_stride_dims[i].second;
        permute_order[output_dim] = ନିବେଶ_dim;
    }

    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        ନିବେଶ, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
