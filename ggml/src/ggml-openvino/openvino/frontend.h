// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ಅನ್ತರ್ಭಾವ <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd {
ಸಾರ್ವಜನಿಕ:
    FrontEnd();

    ಸ್ಥಿರ std::shared_ptr<Model> convert(ಧ್ರುವ ಒಳಹರಿವು_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
