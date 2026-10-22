#pragma once

#अन्तर्भाव "openvino/op/op.hpp"

नामाकाश ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
वर्ग OPENVINO_API GatedDeltaNet : सार्वजनिक ov::op::Op {
सार्वजनिक:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = स्वभाव;
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
    GatedDeltaNet(अचर Output<Node>& query,
                  अचर Output<Node>& साँचो,
                  अचर Output<Node>& value,
                  अचर Output<Node>& recurrent_state,
                  अचर Output<Node>& gate,
                  अचर Output<Node>& beta,
                  अचर तार्किक fuse_qk_l2norm = असत्य,
                  अचर भिन्न q_l2_norm_eps = 1e-6F,
                  अचर भिन्न k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(अचर ov::OutputVector& args,
                  अचर तार्किक fuse_qk_l2norm = असत्य,
                  अचर भिन्न q_l2_norm_eps = 1e-6F,
                  अचर भिन्न k_l2_norm_eps = 1e-6F);
    शून्य validate_and_infer_types() override;
    तार्किक visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(अचर ov::OutputVector& new_args) अचर override;
    तार्किक get_fuse_qk_l2norm() अचर {
        फर्काऊ m_fuse_qk_l2norm;
    }
    भिन्न get_q_l2_norm_eps() अचर {
        फर्काऊ m_q_l2_norm_eps;
    }
    भिन्न get_k_l2_norm_eps() अचर {
        फर्काऊ m_k_l2_norm_eps;
    }

गुप्त:
    तार्किक m_fuse_qk_l2norm = असत्य;
    भिन्न m_q_l2_norm_eps = 1e-6F;
    भिन्न m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
