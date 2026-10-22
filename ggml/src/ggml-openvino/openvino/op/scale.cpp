#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/core/except.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/equal.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/greater_eq.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/if.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/less.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/logical_or.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/range.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_scale(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ದಶಮಾಂಶ scale;
    ದಶಮಾಂಶ bias;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, (ದಶಮಾಂಶ *) context.get_output_op_params() + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&bias, (ದಶಮಾಂಶ *) context.get_output_op_params() + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ವಯಂಚಾಲಿತ scale_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ದಶಮಾಂಶ>{scale});

    ಆದರೆ (context.get_op_case() == 2) {
        ಹಿಂತಿರುಗಿ {context.get_ಒಳಹರಿವು(0)};
    }

    ಆದರೆ (context.get_op_case() == 1 && context.has_ಒಳಹರಿವು("cache_rs_reset_len")) {
        ಸ್ವಯಂಚಾಲಿತ cache_rs_reset_idx = context.get_ಒಳಹರಿವು("cache_rs_reset_idx");
        ಸ್ವಯಂಚಾಲಿತ cache_rs_reset_len = context.get_ಒಳಹರಿವು("cache_rs_reset_len");

        ಸ್ವಯಂಚಾಲಿತ cache_rs = context.get_ಒಳಹರಿವು(0);

        ಸ್ವಯಂಚಾಲಿತ cache_shape = std::make_shared<ov::op::v3::Shape_ದ>(cache_rs, ov::element::i64);
        ಸ್ವಯಂಚಾಲಿತ n_slots_1d = std::make_shared<ov::op::v8::Gather>(
            cache_shape, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        ಸ್ವಯಂಚಾಲಿತ n_slots = std::make_shared<ov::op::v0::Squeeze>(n_slots_1d);

        ಸ್ವಯಂಚಾಲಿತ iota = std::make_shared<ov::op::v4::Range>(
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}), n_slots,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), ov::element::i64);

        ಸ್ವಯಂಚಾಲಿತ idx_plus_len = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(cache_rs_reset_idx, cache_rs_reset_len);
        ಸ್ವಯಂಚಾಲಿತ less_than_idx = std::make_shared<ov::op::v1::Less>(iota, cache_rs_reset_idx);
        ಸ್ವಯಂಚಾಲಿತ greater_equal_idx_plus_len = std::make_shared<ov::op::v1::GreaterEqual>(iota, idx_plus_len);
        ಸ್ವಯಂಚಾಲಿತ keep_ಮುಖವಾಡ = std::make_shared<ov::op::v1::LogicalOr>(less_than_idx, greater_equal_idx_plus_len);

        ಸ್ವಯಂಚಾಲಿತ keep_ಮುಖವಾಡ_f32 = std::make_shared<ov::op::v0::Convert>(keep_ಮುಖವಾಡ, ov::element::f32);
        ಸ್ವಯಂಚಾಲಿತ keep_ಮುಖವಾಡ_reshape = std::make_shared<ov::op::v0::Unsqueeze>(
            keep_ಮುಖವಾಡ_f32, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        ಸ್ವಯಂಚಾಲಿತ cleared_cache_rs = std::make_shared<ov::op::v1::Multiply>(cache_rs, keep_ಮುಖವಾಡ_reshape);
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    ಸ್ವಯಂಚಾಲಿತ scaled = std::make_shared<ov::op::v1::Multiply>(context.get_ಒಳಹರಿವು(0), scale_ಗಂಟು);

    std::shared_ptr<ov::Node> res;
    ಆದರೆ (bias != 0.0f) {
        ಸ್ವಯಂಚಾಲಿತ bias_ಗಂಟು =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ದಶಮಾಂಶ>{bias});
        res = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(scaled, bias_ಗಂಟು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        res = scaled;
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
