#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <openvino/core/except.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/equal.hpp>
#समावेश <openvino/op/gather.hpp>
#समावेश <openvino/op/greater_eq.hpp>
#समावेश <openvino/op/if.hpp>
#समावेश <openvino/op/less.hpp>
#समावेश <openvino/op/logical_or.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/range.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/squeeze.hpp>
#समावेश <openvino/op/unsqueeze.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_scale(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    भग्न scale;
    भग्न bias;
    स_नकल(&scale, (भग्न *) context.get_output_op_params() + 0, माप(भग्न));
    स_नकल(&bias, (भग्न *) context.get_output_op_params() + 1, माप(भग्न));

    स्वतः scale_गाँठ = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<भग्न>{scale});

    अगर (context.get_op_case() == 2) {
        वापस {context.get_निवेश(0)};
    }

    अगर (context.get_op_case() == 1 && context.has_निवेश("cache_rs_reset_len")) {
        स्वतः cache_rs_reset_idx = context.get_निवेश("cache_rs_reset_idx");
        स्वतः cache_rs_reset_len = context.get_निवेश("cache_rs_reset_len");

        स्वतः cache_rs = context.get_निवेश(0);

        स्वतः cache_shape = std::make_shared<ov::op::v3::ShapeOf>(cache_rs, ov::element::i64);
        स्वतः n_slots_1d = std::make_shared<ov::op::v8::Gather>(
            cache_shape, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        स्वतः n_slots = std::make_shared<ov::op::v0::Squeeze>(n_slots_1d);

        स्वतः iota = std::make_shared<ov::op::v4::Range>(
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}), n_slots,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), ov::element::i64);

        स्वतः idx_plus_len = std::make_shared<ov::op::v1::Add>(cache_rs_reset_idx, cache_rs_reset_len);
        स्वतः less_than_idx = std::make_shared<ov::op::v1::Less>(iota, cache_rs_reset_idx);
        स्वतः greater_equal_idx_plus_len = std::make_shared<ov::op::v1::GreaterEqual>(iota, idx_plus_len);
        स्वतः keep_mask = std::make_shared<ov::op::v1::LogicalOr>(less_than_idx, greater_equal_idx_plus_len);

        स्वतः keep_mask_f32 = std::make_shared<ov::op::v0::Convert>(keep_mask, ov::element::f32);
        स्वतः keep_mask_reshape = std::make_shared<ov::op::v0::Unsqueeze>(
            keep_mask_f32, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        स्वतः cleared_cache_rs = std::make_shared<ov::op::v1::Multiply>(cache_rs, keep_mask_reshape);
        वापस rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    स्वतः scaled = std::make_shared<ov::op::v1::Multiply>(context.get_निवेश(0), scale_गाँठ);

    std::shared_ptr<ov::Node> res;
    अगर (bias != 0.0f) {
        स्वतः bias_गाँठ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<भग्न>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_गाँठ);
    } अन्यथा {
        res = scaled;
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
