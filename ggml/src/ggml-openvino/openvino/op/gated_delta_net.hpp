#pragma once

#અન્તર્ભાવ "openvino/op/op.hpp"

નામાકાશ ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
વર્ગ OPENVINO_API GatedDeltaNet : જાહેર ov::op::Op {
જાહેર:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = મૂળભૂત;
    /// \brief Constructs a GatedDeltaNet operation.
    ///
    /// \param query Query tensor input.
    /// \param key Key tensor input.
    /// \param value Value tensor input.
    /// \param recurrent_state Initial recurrent state tensor.
    /// \param gate Gate tensor controlling state decay/update.
    /// \param beta Beta tensor scaling the delta update.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(અચલ Output<Node>& query,
                  અચલ Output<Node>& ચાવી,
                  અચલ Output<Node>& value,
                  અચલ Output<Node>& recurrent_state,
                  અચલ Output<Node>& gate,
                  અચલ Output<Node>& beta,
                  અચલ તાર્કિક fuse_qk_l2norm = ખોટું,
                  અચલ દશાંશ q_l2_norm_eps = 1e-6F,
                  અચલ દશાંશ k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(અચલ ov::OutputVector& args,
                  અચલ તાર્કિક fuse_qk_l2norm = ખોટું,
                  અચલ દશાંશ q_l2_norm_eps = 1e-6F,
                  અચલ દશાંશ k_l2_norm_eps = 1e-6F);
    શૂન્ય validate_and_infer_types() override;
    તાર્કિક visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(અચલ ov::OutputVector& new_args) અચલ override;
    તાર્કિક get_fuse_qk_l2norm() અચલ {
        પાછા m_fuse_qk_l2norm;
    }
    દશાંશ get_q_l2_norm_eps() અચલ {
        પાછા m_q_l2_norm_eps;
    }
    દશાંશ get_k_l2_norm_eps() અચલ {
        પાછા m_k_l2_norm_eps;
    }

ખાનગી:
    તાર્કિક m_fuse_qk_l2norm = ખોટું;
    દશાંશ m_q_l2_norm_eps = 1e-6F;
    દશાંશ m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
