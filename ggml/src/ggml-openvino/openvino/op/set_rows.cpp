#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <cassert>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/core/node.hpp>
#অন্তর্ভাব <openvino/core/node_output.hpp>
#অন্তর্ভাব <openvino/core/type.hpp>
#অন্তর্ভাব <openvino/frontend/exception.hpp>
#অন্তর্ভাব <openvino/op/broadcast.hpp>
#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/gather.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/scatter_elements_update.hpp>
#অন্তর্ভাব <openvino/op/scatter_update.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>
#অন্তর্ভাব <openvino/op/squeeze.hpp>
#অন্তর্ভাব <openvino/op/transpose.hpp>
#অন্তর্ভাব <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_set_rows(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 3, 3);

    auto তথ্য = process_view_প্রবেশ_new(context, 0);
    auto indices = process_view_প্রবেশ_new(context, 1);
    auto dst = process_view_প্রবেশ_new(context, 2);

    তথ্য = std::make_shared<ov::op::v0::Convert>(তথ্য, context.get_output_type());

    ধ্রুবক auto indices_shape = context.get_প্রবেশ_shape(1);
    ধ্রুবক যৌক্তিক multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_দৈর্ঘ্য() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_দৈর্ঘ্য() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_দৈর্ঘ্য() > 1));

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    যদি (context.is_stateful()) {
        পূর্ণসংখ্যা concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_দৈর্ঘ্য();
        int64_t dim3 = dst.get_partial_shape()[3].get_দৈর্ঘ্য();
        তথ্য = std::make_shared<ov::op::v1::Reshape>(
            তথ্য, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), মিথ্যা);
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{dst, তথ্য}, concat_axis);
    } অন্যথা যদি (multidim_indices) {
        auto updates_shape = std::make_shared<ov::op::v3::ShapeOf>(তথ্য, ov::element::i64);

        auto indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto indices_rank4_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{get_dimensions(updates_shape, {0, 1, 2}), one}, 0);
        auto indices_rank4 = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rank4_shape, মিথ্যা);
        auto broadcasted_indices = std::make_shared<ov::op::v3::Broadcast>(indices_rank4, updates_shape);

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, তথ্য, axes);
    } অন্যথা {
        auto row_আকার = context.get_প্রবেশ_shape(2)[3].get_দৈর্ঘ্য();
        auto ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        auto তথ্য_reshaped = std::make_shared<ov::op::v1::Reshape>(
            তথ্য,
            ov::op::v0::Constant::create(ov::element::i64, {4},
                                         {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) row_আকার}),
            মিথ্যা);
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, তথ্য_reshaped, axes);
    }

    auto dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_গ্রন্থি_shared_ptr());
    যদি (!multidim_indices && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        auto dst_shape_partial = dst_reshape->get_প্রবেশ_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_দৈর্ঘ্য(), dst_shape_partial[1].get_দৈর্ঘ্য(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_দৈর্ঘ্য() : -1,
                                          dst_shape_partial[3].get_দৈর্ঘ্য()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    মিথ্যা);
    }
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
