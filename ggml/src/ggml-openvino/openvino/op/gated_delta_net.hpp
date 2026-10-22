#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "openvino/op/op.hpp"

ನಾಮಾಕಾಶ ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
ವರ್ಗ OPENVINO_API GatedDeltaNet : ಸಾರ್ವಜನಿಕ ov::op::Op {
ಸಾರ್ವಜನಿಕ:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = ಪೂರ್ವನಿಯೋಜಿತ;
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
    GatedDeltaNet(ಸ್ಥಿರಾಂಕ Output<Node>& query,
                  ಸ್ಥಿರಾಂಕ Output<Node>& ಕೀಲಿ,
                  ಸ್ಥಿರಾಂಕ Output<Node>& value,
                  ಸ್ಥಿರಾಂಕ Output<Node>& recurrent_state,
                  ಸ್ಥಿರಾಂಕ Output<Node>& gate,
                  ಸ್ಥಿರಾಂಕ Output<Node>& beta,
                  ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ fuse_qk_l2norm = ಸುಳ್ಳು,
                  ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ q_l2_norm_eps = 1e-6F,
                  ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(ಸ್ಥಿರಾಂಕ ov::Output_ಸದಿಶ& ವಾದಗಳು_2,
                  ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ fuse_qk_l2norm = ಸುಳ್ಳು,
                  ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ q_l2_norm_eps = 1e-6F,
                  ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ k_l2_norm_eps = 1e-6F);
    ಶೂನ್ಯ validate_and_infer_types() override;
    ತಾರ್ಕಿಕ ಭೇಟಿ_ಗುಣಲಕ್ಷಣಗಳು(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(ಸ್ಥಿರಾಂಕ ov::Output_ಸದಿಶ& new_ವಾದಗಳು) ಸ್ಥಿರಾಂಕ override;
    ತಾರ್ಕಿಕ get_fuse_qk_l2norm() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_fuse_qk_l2norm;
    }
    ದಶಮಾಂಶ get_q_l2_norm_eps() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_q_l2_norm_eps;
    }
    ದಶಮಾಂಶ get_k_l2_norm_eps() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_k_l2_norm_eps;
    }

ಖಾಸಗಿ:
    ತಾರ್ಕಿಕ m_fuse_qk_l2norm = ಸುಳ್ಳು;
    ದಶಮಾಂಶ m_q_l2_norm_eps = 1e-6F;
    ದಶಮಾಂಶ m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
