#અન્તર્ભાવ "align_eltwise_ranks.h"

#અન્તર્ભાવ <numeric>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/divide.hpp>
#અન્તર્ભાવ <openvino/op/multiply.hpp>
#અન્તર્ભાવ <openvino/op/sqrt.hpp>
#અન્તર્ભાવ <openvino/op/subtract.hpp>
#અન્તર્ભાવ <openvino/op/unsqueeze.hpp>
#અન્તર્ભાવ <openvino/pass/pattern/op/wrap_type.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ pass {

નામાકાશ {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
તાર્કિક is_rms_norm_output(અચલ ov::Output<ov::Node> & value, પૂર્ણાંક depth = 1) {
    અચલ auto * ગાંઠ = value.get_ગાંઠ();
    જો (!ov::is_type<ov::op::v1::Multiply>(ગાંઠ)) {
        પાછા ખોટું;
    }
    માટે (અચલ auto & આગત : ગાંઠ->આગત_કિંમતો()) {
        અચલ auto * src = આગત.get_ગાંઠ();
        જો (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_આગત_ગાંઠ_ptr(1))) {
            પાછા સાચું;
        }
        જો (depth > 0 && is_rms_norm_output(આગત, depth - 1)) {
            પાછા સાચું;
        }
    }
    પાછા ખોટું;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    અચલ auto callback = [આ](ov::pass::pattern::Matcher & m) {
        auto ગાંઠ = m.get_match_મૂળ();
        જો (ગાંઠ->get_આગત_size() != 2) {
            પાછા ખોટું;
        }

        auto lhs = ગાંઠ->આગત_કિંમત(0);
        auto rhs = ગાંઠ->આગત_કિંમત(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        જો (ov::is_type<ov::op::v0::Constant>(lhs.get_ગાંઠ()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_ગાંઠ())) {
            પાછા ખોટું;
        }

        અચલ auto lhs_rank = lhs.get_partial_shape().rank();
        અચલ auto rhs_rank = rhs.get_partial_shape().rank();
        જો (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            પાછા ખોટું;
        }

        અચલ size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        અચલ auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        જો (is_rms_norm_output(shorter)) {
            પાછા ખોટું;
        }
        અચલ int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        ગાંઠ->આગત(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_ગાંઠ(unsqueeze);
        પાછા સાચું;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
