#ضمن "align_eltwise_ranks.h"

#ضمن <numeric>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/sqrt.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
منطقي is_rms_norm_output(ثابت ov::Output<ov::Node> & value, صحيح depth = 1) {
    ثابت auto * عقدة = value.get_عقدة();
    إذا (!ov::is_type<ov::op::v1::Multiply>(عقدة)) {
        إرجاع خطأ;
    }
    لكل (ثابت auto & إدخال : عقدة->إدخال_قيم()) {
        ثابت auto * src = إدخال.get_عقدة();
        إذا (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_إدخال_عقدة_ptr(1))) {
            إرجاع true;
        }
        إذا (depth > 0 && is_rms_norm_output(إدخال, depth - 1)) {
            إرجاع true;
        }
    }
    إرجاع خطأ;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    ثابت auto callback = [هذا](ov::pass::pattern::Matcher & m) {
        auto عقدة = m.get_match_جذر();
        إذا (عقدة->get_إدخال_size() != 2) {
            إرجاع خطأ;
        }

        auto lhs = عقدة->إدخال_قيمة(0);
        auto rhs = عقدة->إدخال_قيمة(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        إذا (ov::is_type<ov::op::v0::Constant>(lhs.get_عقدة()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_عقدة())) {
            إرجاع خطأ;
        }

        ثابت auto lhs_rank = lhs.get_partial_shape().rank();
        ثابت auto rhs_rank = rhs.get_partial_shape().rank();
        إذا (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            إرجاع خطأ;
        }

        ثابت size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        ثابت auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        إذا (is_rms_norm_output(shorter)) {
            إرجاع خطأ;
        }
        ثابت int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        عقدة->إدخال(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_عقدة(unsqueeze);
        إرجاع true;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
