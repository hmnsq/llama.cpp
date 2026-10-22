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
منطقي is_rms_norm_output(ثابت ov::Output<ov::Node> & value, صحيح ڈونگھائی = 1) {
    ثابت خودکار * گرہ = value.get_گرہ();
    إذا (!ov::is_type<ov::op::v1::Multiply>(گرہ)) {
        أرجع خطأ;
    }
    لكل (ثابت خودکار & داخلہ : گرہ->داخلہ_اقدار()) {
        ثابت خودکار * src = داخلہ.get_گرہ();
        إذا (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_داخلہ_گرہ_ptr(1))) {
            أرجع سچ;
        }
        إذا (ڈونگھائی > 0 && is_rms_norm_output(داخلہ, ڈونگھائی - 1)) {
            أرجع سچ;
        }
    }
    أرجع خطأ;
}

}  // namespace

Align_Eltwise_عملیہ_Ranks::Align_Eltwise_عملیہ_Ranks() {
    خودکار eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::جوڑو, ov::op::v1::Multiply, ov::op::v1::گھٹاؤ>();

    ثابت خودکار callback = [هذا](ov::pass::pattern::Matcher & m) {
        خودکار گرہ = m.get_میل_جڑ();
        إذا (گرہ->get_داخلہ_ناپ() != 2) {
            أرجع خطأ;
        }

        خودکار lhs = گرہ->داخلہ_قدر(0);
        خودکار rhs = گرہ->داخلہ_قدر(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        إذا (ov::is_type<ov::op::v0::Constant>(lhs.get_گرہ()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_گرہ())) {
            أرجع خطأ;
        }

        ثابت خودکار lhs_rank = lhs.get_partial_shape().rank();
        ثابت خودکار rhs_rank = rhs.get_partial_shape().rank();
        إذا (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            أرجع خطأ;
        }

        ثابت size_t shorter_idx = lhs_rank.get_لمائی() < rhs_rank.get_لمائی() ? 0 : 1;
        ثابت خودکار & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        إذا (is_rms_norm_output(shorter)) {
            أرجع خطأ;
        }
        ثابت int64_t diff = std::مطلق_قدر(lhs_rank.get_لمائی() - rhs_rank.get_لمائی());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        خودکار unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        گرہ->داخلہ(shorter_idx).replace_سروت_output(unsqueeze->output(0));
        رجسٹر_new_گرہ(unsqueeze);
        أرجع سچ;
    };

    رجسٹر_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
