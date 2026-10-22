#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/op/op.hpp"

ਨਾਮਾਕਾਸ਼ ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
ਵਰਗ OPENVINO_API GatedDeltaNet : ਜਨਤਕ ov::op::Op {
ਜਨਤਕ:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = ਮੂਲ;
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
    GatedDeltaNet(ਅਚਲ Output<Node>& query,
                  ਅਚਲ Output<Node>& ਕੁੰਜੀ,
                  ਅਚਲ Output<Node>& value,
                  ਅਚਲ Output<Node>& recurrent_state,
                  ਅਚਲ Output<Node>& gate,
                  ਅਚਲ Output<Node>& beta,
                  ਅਚਲ ਤਾਰਕਿਕ fuse_qk_l2norm = ਝੂਠ,
                  ਅਚਲ ਦਸ਼ਮਲਵ q_l2_norm_eps = 1e-6F,
                  ਅਚਲ ਦਸ਼ਮਲਵ k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(ਅਚਲ ov::Output_ਸਦਿਸ਼& ਦਲੀਲਾਂ_2,
                  ਅਚਲ ਤਾਰਕਿਕ fuse_qk_l2norm = ਝੂਠ,
                  ਅਚਲ ਦਸ਼ਮਲਵ q_l2_norm_eps = 1e-6F,
                  ਅਚਲ ਦਸ਼ਮਲਵ k_l2_norm_eps = 1e-6F);
    ਖਾਲੀ validate_and_infer_types() override;
    ਤਾਰਕਿਕ ਦੇਖੋ_ਗੁਣ(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(ਅਚਲ ov::Output_ਸਦਿਸ਼& new_ਦਲੀਲਾਂ) ਅਚਲ override;
    ਤਾਰਕਿਕ get_fuse_qk_l2norm() ਅਚਲ {
        ਵਾਪਸ m_fuse_qk_l2norm;
    }
    ਦਸ਼ਮਲਵ get_q_l2_norm_eps() ਅਚਲ {
        ਵਾਪਸ m_q_l2_norm_eps;
    }
    ਦਸ਼ਮਲਵ get_k_l2_norm_eps() ਅਚਲ {
        ਵਾਪਸ m_k_l2_norm_eps;
    }

ਨਿਜੀ:
    ਤਾਰਕਿਕ m_fuse_qk_l2norm = ਝੂਠ;
    ਦਸ਼ਮਲਵ m_q_l2_norm_eps = 1e-6F;
    ਦਸ਼ਮਲਵ m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
