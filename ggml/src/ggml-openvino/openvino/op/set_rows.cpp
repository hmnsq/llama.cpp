#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/exception.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/scatter_elements_update.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/scatter_update.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 3, 3);

    ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ indices = process_view_ਨਿਵੇਸ਼_new(context, 1);
    ਸਵੈਚਾਲਿਤ dst = process_view_ਨਿਵੇਸ਼_new(context, 2);

    ਅੰਕੜੇ = std::make_shared<ov::op::v0::Convert>(ਅੰਕੜੇ, context.get_output_type());

    ਅਚਲ ਸਵੈਚਾਲਿਤ indices_shape = context.get_ਨਿਵੇਸ਼_shape(1);
    ਅਚਲ ਤਾਰਕਿਕ multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_ਲੰਮਾਈ() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_ਲੰਮਾਈ() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_ਲੰਮਾਈ() > 1));

    ਸਵੈਚਾਲਿਤ axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    ਜੇ (context.is_stateful()) {
        ਪੂਰਨ concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_ਲੰਮਾਈ();
        int64_t dim3 = dst.get_partial_shape()[3].get_ਲੰਮਾਈ();
        ਅੰਕੜੇ = std::make_shared<ov::op::v1::Reshape>(
            ਅੰਕੜੇ, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), ਝੂਠ);
        res = std::make_shared<ov::op::v0::Concat>(Output_ਸਦਿਸ਼{dst, ਅੰਕੜੇ}, concat_axis);
    } ਨਹੀਂਤਾਂ ਜੇ (multidim_indices) {
        ਸਵੈਚਾਲਿਤ updates_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(ਅੰਕੜੇ, ov::element::i64);

        ਸਵੈਚਾਲਿਤ indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ਸਵੈਚਾਲਿਤ indices_rank4_shape = std::make_shared<ov::op::v0::Concat>(Output_ਸਦਿਸ਼{get_dimensions(updates_shape, {0, 1, 2}), one}, 0);
        ਸਵੈਚਾਲਿਤ indices_rank4 = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rank4_shape, ਝੂਠ);
        ਸਵੈਚਾਲਿਤ broadcasted_indices = std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(indices_rank4, updates_shape);

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, ਅੰਕੜੇ, axes);
    } ਨਹੀਂਤਾਂ {
        ਸਵੈਚਾਲਿਤ ਪੰਕਤੀ_ਮਾਪ = context.get_ਨਿਵੇਸ਼_shape(2)[3].get_ਲੰਮਾਈ();
        ਸਵੈਚਾਲਿਤ ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_reshaped = std::make_shared<ov::op::v1::Reshape>(
            ਅੰਕੜੇ,
            ov::op::v0::Constant::create(ov::element::i64, {4},
                                         {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) ਪੰਕਤੀ_ਮਾਪ}),
            ਝੂਠ);
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, ਅੰਕੜੇ_reshaped, axes);
    }

    ਸਵੈਚਾਲਿਤ dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_ਗੰਢ_shared_ptr());
    ਜੇ (!multidim_indices && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        ਸਵੈਚਾਲਿਤ dst_shape_partial = dst_reshape->get_ਨਿਵੇਸ਼_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_ਲੰਮਾਈ(), dst_shape_partial[1].get_ਲੰਮਾਈ(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_ਲੰਮਾਈ() : -1,
                                          dst_shape_partial[3].get_ਲੰਮਾਈ()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    ਝੂਠ);
    }
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
