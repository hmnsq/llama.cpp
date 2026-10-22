#شامل "align_eltwise_ranks.h"

#شامل <numeric>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/divide.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/sqrt.hpp>
#شامل <openvino/op/subtract.hpp>
#شامل <openvino/op/unsqueeze.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا pass {

نامخلا {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
منطقی is_rms_norm_output(ثابت ov::Output<ov::Node> & value, صحیح گہرائی = 1) {
    ثابت خودکار * گرہ = value.get_گرہ();
    اگر (!ov::is_type<ov::op::v1::Multiply>(گرہ)) {
        واپس جھوٹ;
    }
    برائے (ثابت خودکار & داخلہ : گرہ->داخلہ_اقدار()) {
        ثابت خودکار * src = داخلہ.get_گرہ();
        اگر (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_داخلہ_گرہ_ptr(1))) {
            واپس سچ;
        }
        اگر (گہرائی > 0 && is_rms_norm_output(داخلہ, گہرائی - 1)) {
            واپس سچ;
        }
    }
    واپس جھوٹ;
}

}  // namespace

Align_Eltwise_عملیہ_Ranks::Align_Eltwise_عملیہ_Ranks() {
    خودکار eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::جوڑو, ov::op::v1::Multiply, ov::op::v1::تفریق>();

    ثابت خودکار callback = [یہ](ov::pass::pattern::Matcher & m) {
        خودکار گرہ = m.get_مطابقت_جڑ();
        اگر (گرہ->get_داخلہ_سائز() != 2) {
            واپس جھوٹ;
        }

        خودکار lhs = گرہ->داخلہ_قدر(0);
        خودکار rhs = گرہ->داخلہ_قدر(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        اگر (ov::is_type<ov::op::v0::Constant>(lhs.get_گرہ()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_گرہ())) {
            واپس جھوٹ;
        }

        ثابت خودکار lhs_rank = lhs.get_partial_shape().rank();
        ثابت خودکار rhs_rank = rhs.get_partial_shape().rank();
        اگر (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            واپس جھوٹ;
        }

        ثابت size_t shorter_idx = lhs_rank.get_لمبائی() < rhs_rank.get_لمبائی() ? 0 : 1;
        ثابت خودکار & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        اگر (is_rms_norm_output(shorter)) {
            واپس جھوٹ;
        }
        ثابت int64_t diff = std::مطلق_قدر(lhs_rank.get_لمبائی() - rhs_rank.get_لمبائی());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        خودکار unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        گرہ->داخلہ(shorter_idx).replace_ماخذ_output(unsqueeze->output(0));
        رجسٹر_new_گرہ(unsqueeze);
        واپس سچ;
    };

    رجسٹر_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
