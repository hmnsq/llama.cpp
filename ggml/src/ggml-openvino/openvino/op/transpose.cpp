#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_transpose(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_stride = context.get_ਨਿਵੇਸ਼_stride(0);
    ਸਵੈਚਾਲਿਤ output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, ਪੂਰਨ>> output_stride_dims;
    std::vector<std::pair<size_t, ਪੂਰਨ>> ਨਿਵੇਸ਼_stride_dims;

    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        ਨਿਵੇਸ਼_stride_dims.push_back({ਨਿਵੇਸ਼_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(ਨਿਵੇਸ਼_stride_dims.rbegin(), ਨਿਵੇਸ਼_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> ਕ੍ਰਮ_ਬਦਲੋ_ਕ੍ਰਮ(4);
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        ਪੂਰਨ output_dim = output_stride_dims[i].second;
        ਪੂਰਨ ਨਿਵੇਸ਼_dim = ਨਿਵੇਸ਼_stride_dims[i].second;
        ਕ੍ਰਮ_ਬਦਲੋ_ਕ੍ਰਮ[output_dim] = ਨਿਵੇਸ਼_dim;
    }

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);

    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Transpose>(
        ਨਿਵੇਸ਼, ov::op::v0::Constant::create(ov::element::i64, {4}, ਕ੍ਰਮ_ਬਦਲੋ_ਕ੍ਰਮ));
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
