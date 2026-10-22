#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <openvino/op/transpose.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_transpose(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    auto প্ৰবেশ_stride = context.get_প্ৰবেশ_stride(0);
    auto output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, পূর্ণাঙ্ক>> output_stride_dims;
    std::vector<std::pair<size_t, পূর্ণাঙ্ক>> প্ৰবেশ_stride_dims;

    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        প্ৰবেশ_stride_dims.push_back({প্ৰবেশ_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(প্ৰবেশ_stride_dims.rbegin(), প্ৰবেশ_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) {
        পূর্ণাঙ্ক output_dim = output_stride_dims[i].second;
        পূর্ণাঙ্ক প্ৰবেশ_dim = প্ৰবেশ_stride_dims[i].second;
        permute_order[output_dim] = প্ৰবেশ_dim;
    }

    auto প্ৰবেশ = process_view_প্ৰবেশ_new(context, 0);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        প্ৰবেশ, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
