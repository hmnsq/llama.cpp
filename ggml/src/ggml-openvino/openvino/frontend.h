// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_ಕೊನೆ {
ಸಾರ್ವಜನಿಕ:
    Front_ಕೊನೆ();

    ಸ್ಥಿರ std::shared_ptr<Model> convert(ಸ್ಥಿರಾಂಕ ಒಳಹರಿವು_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
