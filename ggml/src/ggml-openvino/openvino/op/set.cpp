#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/exception.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/range.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reduce_prod.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/scatter_update.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
OutputVector translate_set(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    auto dst = process_view_ಒಳಹರಿವು_new(context, 0);
    auto src = process_view_ಒಳಹರಿವು_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    ಧ್ರುವ auto dst_stride = context.get_ಒಳಹರಿವು_stride(0);
    FRONT_END_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    ಧ್ರುವ auto * op_params = reinterpret_cast<ಧ್ರುವ uint32_t *>(context.get_output_op_params());
    ಧ್ರುವ size_t offset = static_cast<size_t>(op_params[3]);

    ಧ್ರುವ size_t elem_size = dst_stride.back();
    FRONT_END_OP_CONVERSION_CHECK(elem_size != 0 && offset % elem_size == 0,
                                  "SET offset must be aligned to destination element size");

    ಧ್ರುವ int64_t offset_elems = static_cast<int64_t>(offset / elem_size);

    auto dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        ಅಸತ್ಯ);

    auto src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        ಅಸತ್ಯ);

    auto src_shape = std::make_shared<ov::op::v3::ShapeOf>(src_flat, ov::element::i64);
    auto src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        ಅಸತ್ಯ);

    auto start = ov::op::v0::Constant::create(ov::element::i64, {}, {offset_elems});
    auto stop = std::make_shared<ov::op::v1::Add>(start, src_len);
    auto step = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    auto indices = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    auto updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    auto dst_shape = std::make_shared<ov::op::v3::ShapeOf>(dst, ov::element::i64);
    auto res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, ಅಸತ್ಯ);

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
