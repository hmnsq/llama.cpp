#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/range.hpp>
#अन्तर्भाव <openvino/op/reduce_prod.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/scatter_update.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
OutputVector translate_set(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 2, 2);

    स्वतः dst = process_view_आगत_new(context, 0);
    स्वतः src = process_view_आगत_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    अचर स्वतः dst_stride = context.get_आगत_stride(0);
    FRONT_END_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    अचर स्वतः * op_params = reinterpret_cast<अचर uint32_t *>(context.get_output_op_params());
    अचर माप_प्रकार offset = static_cast<माप_प्रकार>(op_params[3]);

    अचर माप_प्रकार elem_size = dst_stride.back();
    FRONT_END_OP_CONVERSION_CHECK(elem_size != 0 && offset % elem_size == 0,
                                  "SET offset must be aligned to destination element size");

    अचर int64_t offset_elems = static_cast<int64_t>(offset / elem_size);

    स्वतः dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        असत्य);

    स्वतः src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        असत्य);

    स्वतः src_shape = std::make_shared<ov::op::v3::ShapeOf>(src_flat, ov::element::i64);
    स्वतः src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        असत्य);

    स्वतः start = ov::op::v0::Constant::create(ov::element::i64, {}, {offset_elems});
    स्वतः stop = std::make_shared<ov::op::v1::Add>(start, src_len);
    स्वतः step = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    स्वतः indices = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);
    स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    स्वतः updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    स्वतः dst_shape = std::make_shared<ov::op::v3::ShapeOf>(dst, ov::element::i64);
    स्वतः res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, असत्य);

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
